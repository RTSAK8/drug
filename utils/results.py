"""Per-run artifacts and fixed-schema long-form results; never append wide folds."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_info() -> dict:
    versions = {}
    for package in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "hydra-core", "omegaconf", "wandb"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def save_evaluation(folder: Path, split: str, metrics: dict, artifacts: dict) -> None:
    write_json(folder / f"{split}_metrics.json", metrics)
    for task, task_artifacts in artifacts.items():
        for kind, frame in task_artifacts.items():
            frame.to_csv(folder / f"{split}_{task}_{kind}.csv", index=False)


def metric_rows(run_id, experiment, protocol, seed, fold, metrics):
    return [{"run_id": run_id, "experiment": experiment, "protocol": protocol, "seed": seed,
             "fold": fold, "task": task, "metric": metric, "value": value}
            for task, values in metrics.items() for metric, value in values.items()]


def save_aggregate(folder: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "metrics_long.csv", index=False)
    groups = ["experiment", "protocol", "seed", "task", "metric"]
    # First summarize folds WITHIN each repeat. Repeat variability is a separate table.
    per_seed = frame.groupby(groups)["value"].agg(mean="mean", fold_std="std", folds="count").reset_index()
    per_seed["fold_std"] = per_seed["fold_std"].fillna(0.0)
    per_seed.to_csv(folder / "per_seed_summary.csv", index=False)
    repeated = per_seed.groupby(["experiment", "protocol", "task", "metric"])["mean"].agg(
        mean="mean", repeat_std="std", repeats="count").reset_index()
    # With one repeat, standard deviation is unknown (NaN/blank), not evidence of zero variation.
    repeated.to_csv(folder / "summary.csv", index=False)
