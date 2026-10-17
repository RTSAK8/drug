"""Paper-aligned joint training, repeated CV/external evaluation, and traceable outputs."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from uuid import uuid4

import numpy as np

from data_manager import DataManager
from utils.hypergraph import build_hypergraph
from utils.results import environment_info, metric_rows, save_aggregate, save_evaluation, sha256, write_json
from utils.runtime import get_device, set_global_seed
from utils.training import evaluate
from .data_prep import outer_splits, prepare_fold
from .init_wandb import init_wandb_run
from .training_loop import fit_model

LOG = logging.getLogger(__name__)


def run_experiment(config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", config.experiment_name)
    root = Path(config.results_dir) / f"{safe_name}_{config.eval_protocol}_{run_id}"
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config.to_dict())
    handler = logging.FileHandler(root / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        set_global_seed(config.seeds[0], config.deterministic)
        device = get_device(config.device)
        write_json(root / "environment.json", {**environment_info(), "selected_device": str(device)})
        project = Path(__file__).resolve().parents[2]
        source_files = [p for p in project.rglob("*.py") if p.is_relative_to(project / "models") or
                        p.is_relative_to(project / "utils") or p.is_relative_to(project / "tests") or p.parent == project]
        source_files += list((project / "conf").rglob("*.yaml"))
        write_json(root / "source_hashes.json", {str(p.relative_to(project)): sha256(p) for p in sorted(source_files)})
        data = DataManager(config).load()
        write_json(root / "data_stats.json", data.stats)
        write_json(root / "data_hashes.json", {p.name: sha256(p) for p in data.source_files})
        data.mapping_frame().to_csv(root / "entity_mapping.csv", index=False)
        for name, frame in data.rejected.items():
            frame.to_csv(root / f"removed_{name}.csv", index=False)
        features = {kind: vector.to(device) for kind, vector in data.features.items()}
        edge_index, edge_type = data.edge_index.to(device), data.edge_type.to(device)
        all_metric_rows = []
        for seed in config.seeds:
            for fold, outer_train, test_positive in outer_splits(data, config, seed):
                folder = root / f"seed_{seed}" / f"fold_{fold}"
                folder.mkdir(parents=True, exist_ok=False)
                positives, labelled = prepare_fold(data, config, seed, fold, outer_train, test_positive, folder)
                write_json(folder / "split_hashes.json", {p.name: sha256(p) for p in sorted(folder.glob("*.csv"))})
                incidence = build_hypergraph(positives["train"], device) if config.use_hypergraph else None
                graph = (incidence, edge_index, edge_type)
                wandb_run = init_wandb_run(config, run_id, seed, fold)
                try:
                    model, thresholds = fit_model(config, features, graph, positives["train"], positives["validation"],
                                                  labelled["train"], labelled["validation"], data.candidates,
                                                  folder, seed, fold, wandb_run)
                    # Only now is the outer test scored. Filter labels are used by metrics,
                    # never by the encoder/decoder or the selected checkpoint.
                    test_filter = np.concatenate([positives["train"], positives["validation"], positives["test"]])
                    metrics, artifacts, _ = evaluate(model, features, graph, labelled["test"], positives["test"],
                                                      test_filter, data.candidates, config, thresholds=thresholds)
                    save_evaluation(folder, "test", metrics, artifacts)
                    all_metric_rows.extend(metric_rows(run_id, config.experiment_name, config.eval_protocol, seed, fold, metrics))
                    save_aggregate(root, all_metric_rows)
                    if wandb_run is not None:
                        for task, values in metrics.items():
                            for metric, value in values.items():
                                wandb_run.summary[f"test/{task}/{metric}"] = value
                    LOG.info("Completed seed=%d fold=%d: %s", seed, fold, metrics)
                    del model
                finally:
                    if wandb_run is not None:
                        wandb_run.finish()
        write_json(root / "status.json", {"status": "complete", "run_id": run_id, "completed_folds":
                                         len(config.seeds) * (config.n_splits if config.eval_protocol == "cross_validation" else 1)})
        LOG.info("All artifacts saved to %s", root)
    except Exception as exc:
        write_json(root / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        LOG.exception("Experiment failed; inspect %s", root)
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
