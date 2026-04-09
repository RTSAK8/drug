import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_GROUPS = ["gene", "disease"]
TARGET_HIT_METRICS = ["hit@1", "hit@3", "hit@5"]
TARGET_WANDB_METRICS = [f"{g}/test/{m}" for g in TARGET_GROUPS for m in TARGET_HIT_METRICS]
METHOD_ORDER = ["HAN", "MLP", "RGCN", "BASEDRUGBANK", "OURS"]
INDEPENDENT_FILES = {
    "HAN": "han_independent_test.csv",
    "MLP": "mlp_independent_test.csv",
    "RGCN": "rgcn_independent_test.csv",
}
MANUAL_HIT_RESULTS = {
    "OURS": {
        "gene": {"hit@1": 0.0959, "hit@3": 0.2329, "hit@5": 0.3288},
        "disease": {"hit@1": 0.2740, "hit@3": 0.6575, "hit@5": 0.8082},
    },
    "BASEDRUGBANK": {
        "gene": {"hit@1": 0.0822, "hit@3": 0.1781, "hit@5": 0.2329},
        "disease": {"hit@1": 0.0835, "hit@3": 0.4521, "hit@5": 0.5753},
    },
}


def normalize_mode(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "gene" in text:
        return "gene"
    if "disease" in text:
        return "disease"
    return None


def parse_wandb_metric(metric: str) -> tuple[str, str] | None:
    parts = metric.split("/")
    if len(parts) != 3:
        return None
    group, split, metric_name = parts
    if split != "test":
        return None
    if group not in TARGET_GROUPS or metric_name not in TARGET_HIT_METRICS:
        return None
    return group, metric_name


def prepare_wandb_hit_best(wandb_stats_path: Path) -> pd.DataFrame:
    df = pd.read_csv(wandb_stats_path)
    required = {"series", "metric", "max", "count"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"wandb stats missing columns: {sorted(missing)}")

    df["max"] = pd.to_numeric(df["max"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    df = df[df["metric"].isin(TARGET_WANDB_METRICS)]
    df = df[(df["count"] > 0) & (df["max"].notna())]
    if df.empty:
        print("[WARN] No valid wandb hit@K rows found in stats file; OURS will be empty.")
        return pd.DataFrame(columns=["method", "group", "metric_name", "score"])

    non_no_resnet = df[df["series"] != "no_resnet"]
    if non_no_resnet.empty:
        print("[WARN] No non-no_resnet hit@K rows found; falling back to all series.")
    else:
        df = non_no_resnet

    idx = df.groupby("metric")["max"].idxmax()
    best = df.loc[idx, ["metric", "max"]].copy()

    parsed = best["metric"].map(parse_wandb_metric)
    best = best[parsed.notna()].copy()
    best[["group", "metric_name"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=best.index)
    best = best.rename(columns={"max": "score"})
    best["method"] = "OURS"

    return best[["method", "group", "metric_name", "score"]]


def prepare_independent_hit(independent_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for method, filename in INDEPENDENT_FILES.items():
        csv_path = independent_dir / filename
        if not csv_path.exists():
            print(f"[WARN] missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "mode" not in df.columns:
            print(f"[WARN] missing mode column: {csv_path}")
            continue

        df = df.copy()
        df["group"] = df["mode"].map(normalize_mode)
        df = df[df["group"].isin(TARGET_GROUPS)]

        for metric in TARGET_HIT_METRICS:
            if metric not in df.columns:
                print(f"[WARN] missing metric column {metric}: {csv_path}")
                continue

            numeric_col = pd.to_numeric(df[metric], errors="coerce")
            for group in TARGET_GROUPS:
                values = numeric_col[df["group"] == group].dropna()
                if values.empty:
                    score = np.nan
                else:
                    score = float(values.mean())

                rows.append(
                    {
                        "method": method,
                        "group": group,
                        "metric_name": metric,
                        "score": score,
                    }
                )

    if not rows:
        raise SystemExit("No independent method rows parsed.")

    return pd.DataFrame(rows)


def prepare_manual_hit_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, method_map in MANUAL_HIT_RESULTS.items():
        for group in TARGET_GROUPS:
            metric_map = method_map[group]
            for metric in TARGET_HIT_METRICS:
                rows.append(
                    {
                        "method": method,
                        "group": group,
                        "metric_name": metric,
                        "score": float(metric_map[metric]),
                    }
                )
    return pd.DataFrame(rows)


def build_hit_table(independent_df: pd.DataFrame, ours_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    table = pd.concat([independent_df, ours_df], ignore_index=True)

    manual_methods = set(manual_df["method"].unique())
    table = table[~table["method"].isin(manual_methods)]
    table = pd.concat([table, manual_df], ignore_index=True)

    return table.sort_values(by=["group", "metric_name", "method"]).reset_index(drop=True)


def aggregate_for_group_plot(table: pd.DataFrame, group: str) -> pd.DataFrame:
    filtered = table[table["group"] == group].copy()
    if filtered.empty:
        raise SystemExit(f"No rows left after group filtering: {group}")

    agg = (
        filtered.groupby(["method", "metric_name"], as_index=False)["score"]
        .mean()
        .sort_values(by=["method", "metric_name"])
    )

    pivot = agg.pivot(index="method", columns="metric_name", values="score")
    pivot = pivot.reindex(METHOD_ORDER)
    for metric in TARGET_HIT_METRICS:
        if metric not in pivot.columns:
            pivot[metric] = np.nan
    return pivot[TARGET_HIT_METRICS]


def draw_hitk_line_chart(plot_df: pd.DataFrame, group: str, output_path: Path) -> Path:
    x = np.arange(plot_df.shape[0])

    marker_map = {
        "hit@1": "^",
        "hit@3": "o",
        "hit@5": "s",
    }
    color_map = {
        "hit@1": "#2563EB",
        "hit@3": "#D97706",
        "hit@5": "#16A34A",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    for metric in TARGET_HIT_METRICS:
        y = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=marker_map[metric],
            markersize=7,
            linewidth=2.0,
            label=metric.upper(),
            color=color_map[metric],
        )

    all_values = pd.to_numeric(plot_df[TARGET_HIT_METRICS].stack(), errors="coerce").dropna()
    if all_values.empty:
        raise SystemExit("No numeric values to plot for hit@K line chart.")

    y_min = float(all_values.min())
    y_max = float(all_values.max())
    span = max(0.02, y_max - y_min)
    ax.set_ylim(max(0.0, y_min - 0.15 * span), min(1.02, y_max + 0.2 * span))

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index.tolist())
    ax.set_ylabel("Score")
    ax.set_title(f"Hit@K Line Chart ({group.upper()})", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(title="K", ncol=3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw a single line chart for hit@1/hit@3/hit@5 using the same data sources as before."
    )
    parser.add_argument(
        "--wandb-stats",
        type=Path,
        default=Path("results_old") / "wandb_test_metrics_series_stats.csv",
        help="Path to wandb series statistics CSV.",
    )
    parser.add_argument(
        "--independent-dir",
        type=Path,
        default=Path("results"),
        help="Directory that contains han/mlp/rgcn independent_test CSV files.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=TARGET_GROUPS,
        default=TARGET_GROUPS,
        help="Groups to draw (default: gene disease).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "comparison_plots",
        help="Directory to save per-group hit@K line charts.",
    )
    args = parser.parse_args()

    ours = prepare_wandb_hit_best(args.wandb_stats)
    independent = prepare_independent_hit(args.independent_dir)
    manual = prepare_manual_hit_rows()
    table = build_hit_table(independent, ours, manual)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for group in args.groups:
        plot_df = aggregate_for_group_plot(table, group)
        out_path = args.output_dir / f"hitk_line_chart_{group}.png"
        saved_paths.append(draw_hitk_line_chart(plot_df, group, out_path))

    for path in saved_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
