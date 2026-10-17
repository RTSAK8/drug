"""Plot new metrics_long.csv files; reject duplicate runs and mixed protocols."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more metrics_long.csv paths")
    parser.add_argument("--protocol", choices=["independent_test", "cross_validation"])
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--metrics", nargs="+", default=["auc", "aupr", "mrr_filtered_query_macro", "hit@1_filtered", "recall@10_filtered_query_macro"])
    args = parser.parse_args()
    frames = []
    required = {"experiment", "protocol", "seed", "fold", "task", "metric", "value"}
    for path in args.inputs:
        frame = pd.read_csv(path)
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} is not the new long-form schema: missing {required-set(frame.columns)}")
        frames.append(frame)
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
