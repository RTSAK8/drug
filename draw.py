"""Plot new metrics_long.csv files; reject duplicate runs and mixed protocols."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(path, frame):
    """Reject incomplete runs, altered splits and incompatible comparison inputs."""
    root = path.parent
    config = load_json(root / "config.json")
    status = load_json(root / "status.json")
    manifest = load_json(root / "run_manifest.json")
    if status.get("status") != "complete":
        raise ValueError(f"Incomplete/failed run cannot be compared: {root}")
    folds = config["n_splits"] if config["eval_protocol"] == "cross_validation" else 1
    expected = {(seed, fold, task) for seed in config["seeds"] for fold in range(folds) for task in config["tasks"]}
    if status.get("completed_folds") != len(config["seeds"]) * folds:
        raise ValueError(f"Unexpected completed-fold count: {root}")
    if (set(frame.run_id) != {manifest["run_id"]} or set(frame.experiment) != {manifest["experiment"]}
            or set(frame.protocol) != {config["eval_protocol"]}):
        raise ValueError(f"Metric identity differs from run manifest: {root}")
    for metric, subset in frame.groupby("metric"):
        actual = set(zip(subset.seed, subset.fold, subset.task))
        if actual != expected or len(subset) != len(expected):
            raise ValueError(f"Missing or duplicate repeat/fold/task for {metric}: {root}")
    inputs = {}
    for seed in config["seeds"]:
        for fold in range(folds):
            folder = root / f"seed_{seed}" / f"fold_{fold}"
            completion = load_json(folder / "fold_status.json")
            metrics_file = folder / "test_metrics.json"
            if completion.get("status") != "complete" or hashlib.sha256(metrics_file.read_bytes()).hexdigest() != completion["test_metrics_sha256"]:
                raise ValueError(f"Incomplete or modified fold metrics: {folder}")
            expected_metrics = load_json(metrics_file)
            for task, values in expected_metrics.items():
                selected = frame[(frame.seed == seed) & (frame.fold == fold) & (frame.task == task)]
                actual_metrics = dict(zip(selected.metric, selected.value))
                if actual_metrics.keys() != values.keys() or any(
                        not np.isclose(actual_metrics[key], value, rtol=1e-12, atol=1e-12) for key, value in values.items()):
                    raise ValueError(f"Long-form table differs from saved fold metrics: {folder}")
            hashes = load_json(folder / "split_hashes.json")
            for name, digest in hashes.items():
                if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Modified split file: {folder / name}")
            inputs[(seed, fold)] = hashes
    protocol_keys = ("eval_protocol", "n_splits", "validation_fraction", "negative_filter_scope",
                     "negative_sampling_strategy", "neg_ratio", "ranking_tie_policy", "hit_k", "recall_k",
                     "selection_metric", "task_weights", "tasks")
    protocol = {key: config[key] for key in protocol_keys}
    source = load_json(root / "source_hashes.json")
    data = load_json(root / "data_hashes.json")
    # Auxiliary relation subsets legitimately differ between graph ablations.
    core_data = {name: digest for name, digest in data.items() if name.endswith("embeddings.csv") or "triplets" in name}
    return protocol, source, core_data, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more metrics_long.csv paths")
    parser.add_argument("--protocol", choices=["independent_test", "cross_validation"])
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--allow-unpaired", action="store_true", help="Allow different seed sets; common seeds must still use identical splits")
    parser.add_argument("--metrics", nargs="+", default=["auc", "aupr", "mrr_filtered_query_macro", "hit@1_filtered", "recall@10_filtered_query_macro"])
    args = parser.parse_args()
    frames = []
    reference = None
    required = {"run_id", "experiment", "protocol", "seed", "fold", "task", "metric", "value"}
    for path in args.inputs:
        frame = pd.read_csv(path)
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} is not the new long-form schema: missing {required-set(frame.columns)}")
        if args.protocol and set(frame.protocol) != {args.protocol}:
            continue
        comparison = validate_run(path, frame)
        if reference is None:
            reference = comparison
        else:
            if comparison[:3] != reference[:3]:
                raise ValueError("Runs differ in evaluation protocol, source code or core data; compare them separately")
            common = comparison[3].keys() & reference[3].keys()
            if any(comparison[3][key] != reference[3][key] for key in common):
                raise ValueError("Runs have different split/candidate sets for the same seed/fold")
            if not args.allow_unpaired and comparison[3].keys() != reference[3].keys():
                raise ValueError("Paired comparison requires identical seed/fold sets; use --allow-unpaired only for an explicit unpaired analysis")
        frames.append(frame)
    if not frames:
        raise ValueError("No complete run matches the selected protocol")
    frame = pd.concat(frames, ignore_index=True)
    if args.protocol:
        frame = frame[frame.protocol == args.protocol]
    if frame.empty or frame.protocol.nunique() != 1:
        raise ValueError("Select exactly one nonempty evaluation protocol with --protocol")
    identity = ["experiment", "protocol", "seed", "fold", "task", "metric"]
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate experiment/seed/fold metrics: do not count reruns as independent seeds")
    if not np.isfinite(frame.value.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in metrics input")
    frame = frame.loc[~frame.metric.str.endswith("_count") & (frame.metric != "threshold_logit")]
    # Fold means first, then independent-seed statistics, matching runner output.
    per_seed = frame.groupby(["experiment", "protocol", "seed", "task", "metric"], as_index=False).value.mean()
    summary = per_seed.groupby(["experiment", "protocol", "task", "metric"]).value.agg(mean="mean", std="std", repeats="count").reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "plot_summary.csv", index=False)
    for task in sorted(summary.task.unique()):
        for metric in args.metrics:
            selected = summary[(summary.task == task) & (summary.metric == metric)].sort_values("experiment")
            if selected.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, len(selected) * 0.9), 4))
            x = np.arange(len(selected))
            ax.bar(x, selected["mean"], color="#3572a5")
            multiple = selected.repeats.to_numpy() > 1
            if multiple.any():
                ax.errorbar(x[multiple], selected.loc[multiple, "mean"], yerr=selected.loc[multiple, "std"],
                            fmt="none", ecolor="black", capsize=3)
            ax.set_xticks(x, selected.experiment, rotation=35, ha="right")
            ax.set_ylim(0, 1.02)
            ax.set_ylabel(metric)
            ax.set_title(f"{task} | {frame.protocol.iloc[0]}\nMean across seeds; error bars = seed SD (n > 1)")
            fig.tight_layout()
            filename = f"{task}_{metric.replace('@', '_at_')}"
            fig.savefig(args.output_dir / f"{filename}.png", dpi=200)
            fig.savefig(args.output_dir / f"{filename}.pdf")
            plt.close(fig)


if __name__ == "__main__":
    main()
