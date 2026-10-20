"""Traceable runs with source/data/split-verified resume and a single writer lock."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

import numpy as np

from data_manager import DataManager
from utils.checkpointing import run_lock, training_config
from utils.hypergraph import build_hypergraph
from utils.results import environment_info, metric_rows, save_aggregate, save_evaluation, sha256, write_json
from utils.runtime import get_device, set_global_seed
from utils.training import evaluate
from .data_prep import outer_splits, prepare_fold
from .init_wandb import init_wandb_run
from .training_loop import fit_model

LOG = logging.getLogger(__name__)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_or_write(path, value, resume, checkpoints_exist=False):
    if resume and path.is_file():
        if read_json(path) != value:
            raise ValueError(f"Resume fingerprint mismatch: {path.name}")
    elif resume and checkpoints_exist:
        raise ValueError(f"Missing provenance for existing checkpoints: {path}")
    else:
        write_json(path, value)


def run_experiment(config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    experiment = f"{config.method_name}__{config.experiment_name}"
    if config.resume_from:
        root = Path(config.resume_from)
        if not (root / "run_manifest.json").is_file():
            raise ValueError("resume_from must be a run directory created by this code version")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", experiment)
        root = Path(config.results_dir) / f"{safe_name}_{config.eval_protocol}_{run_id}"
        root.mkdir(parents=True, exist_ok=False)
    with run_lock(root):
        if config.resume_from:
            saved = read_json(root / "config.json")
            if training_config(saved) != training_config(config.to_dict()):
                raise ValueError("Resume configuration differs; use the original method, ablation and overrides")
            run_id = read_json(root / "run_manifest.json")["run_id"]
        else:
            write_json(root / "config.json", config.to_dict())
            write_json(root / "run_manifest.json", {"schema": 2, "run_id": run_id, "experiment": experiment})
        handler = logging.FileHandler(root / "run.log", encoding="utf-8")
        logging.getLogger().addHandler(handler)
        try:
            return _run_locked(config, root, run_id, experiment)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()


def _run_locked(config, root, run_id, experiment):
    resume = bool(config.resume_from)
    checkpoints_exist = any(root.glob("seed_*/fold_*/last_checkpoint.pt"))
    set_global_seed(config.seeds[0], config.deterministic, config.deterministic_strict)
    device = get_device(config.device)
    environment = environment_info(device)
    # Preflight errors do not overwrite the status of an earlier valid run.
    verify_or_write(root / "environment.json", environment, resume, checkpoints_exist)
    project = Path(__file__).resolve().parents[2]
    source_files = [p for p in project.rglob("*.py") if p.is_relative_to(project / "models") or
                    p.is_relative_to(project / "utils") or p.is_relative_to(project / "tests") or p.parent == project]
    source_files += list((project / "conf").rglob("*.yaml"))
    verify_or_write(root / "source_hashes.json", {p.relative_to(project).as_posix(): sha256(p) for p in sorted(source_files)},
                    resume, checkpoints_exist)
    data = DataManager(config).load()
    verify_or_write(root / "data_hashes.json", {p.name: sha256(p) for p in data.source_files}, resume, checkpoints_exist)
    write_json(root / "data_stats.json", data.stats)
    data.mapping_frame().to_csv(root / "entity_mapping.csv", index=False)
    for name, frame in data.rejected.items():
        frame.to_csv(root / f"removed_{name}.csv", index=False)
    features = {kind: vector.to(device) for kind, vector in data.features.items()}
    edge_index, edge_type = data.edge_index.to(device), data.edge_type.to(device)
    all_metric_rows, completed_folds = [], 0
    write_json(root / "status.json", {"status": "running", "run_id": run_id})
    try:
        for seed in config.seeds:
            for fold, outer_train, test_positive in outer_splits(data, config, seed):
                folder = root / f"seed_{seed}" / f"fold_{fold}"
                folder.mkdir(parents=True, exist_ok=resume)
                # Regenerate in a temporary directory, verify BEFORE replacing
                # provenance, and hash only split inputs (never output CSV files).
                with tempfile.TemporaryDirectory(prefix=".splits_", dir=root) as temporary:
                    staging = Path(temporary)
                    positives, labelled = prepare_fold(data, config, seed, fold, outer_train, test_positive, staging)
                    split_hashes = {p.name: sha256(p) for p in sorted(staging.glob("*.csv"))}
                    verify_or_write(folder / "split_hashes.json", split_hashes, resume, (folder / "last_checkpoint.pt").is_file())
                    for path in staging.iterdir():
                        shutil.copy2(path, folder / path.name)
                fold_status = folder / "fold_status.json"
                if resume and fold_status.is_file() and read_json(fold_status).get("status") == "complete":
                    if read_json(fold_status)["test_metrics_sha256"] != sha256(folder / "test_metrics.json"):
                        raise ValueError(f"Completed fold metrics were changed: {folder}")
                    metrics = read_json(folder / "test_metrics.json")
                    LOG.info("Reusing completed seed=%d fold=%d", seed, fold)
                else:
                    incidence = build_hypergraph(positives["train"], device) if config.use_hypergraph else None
                    degree = np.bincount(positives["train"].reshape(-1), minlength=sum(data.counts.values()))
                    write_json(folder / "candidate_degree_diagnostics.json", {
                        task: {"candidates": len(pool), "zero_degree": int((degree[pool] == 0).sum()),
                               "degree_1_to_3": int(((degree[pool] > 0) & (degree[pool] <= 3)).sum())}
                        for task, pool in data.candidates.items()
                    })
                    graph = (incidence, edge_index, edge_type)
                    wandb_run = init_wandb_run(config, run_id, seed, fold)
                    try:
                        model, thresholds = fit_model(config, features, graph, positives["train"], positives["validation"],
                                                      labelled["train"], labelled["validation"], data.candidates,
                                                      folder, seed, fold, wandb_run)
                        test_filter = np.concatenate([positives["train"], positives["validation"], positives["test"]])
                        metrics, artifacts, _ = evaluate(model, features, graph, labelled["test"], positives["test"],
                                                          test_filter, data.candidates, config, thresholds=thresholds)
                        save_evaluation(folder, "test", metrics, artifacts)
                        write_json(fold_status, {"status": "complete", "test_metrics_sha256": sha256(folder / "test_metrics.json")})
                        if wandb_run is not None:
                            for task, values in metrics.items():
                                for metric, value in values.items():
                                    wandb_run.summary[f"test/{task}/{metric}"] = value
                        del model
                    finally:
                        if wandb_run is not None:
                            wandb_run.finish()
                completed_folds += 1
                all_metric_rows.extend(metric_rows(run_id, experiment, config.eval_protocol, seed, fold, metrics))
                save_aggregate(root, all_metric_rows)
        write_json(root / "status.json", {"status": "complete", "run_id": run_id, "completed_folds": completed_folds})
        LOG.info("All artifacts saved to %s", root)
        return root
    except Exception as exc:
        write_json(root / "status.json", {"status": "failed", "completed_folds": completed_folds,
                                         "error_type": type(exc).__name__, "error": str(exc)})
        LOG.exception("Experiment failed; inspect %s", root)
        raise
