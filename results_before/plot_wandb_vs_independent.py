import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_GROUPS = ["gene", "disease"]
TARGET_METRICS = ["auc", "aupr", "f1_score", "hit@1", "hit@3", "hit@5"]
TARGET_WANDB_METRICS = [f"{g}/test/{m}" for g in TARGET_GROUPS for m in TARGET_METRICS]
METHOD_ORDER = ["HAN", "MLP", "RGCN", "BASEDRUGBANK"]
INDEPENDENT_FILES = {
    "HAN": "han_independent_test.csv",
    "MLP": "mlp_independent_test.csv",
    "RGCN": "rgcn_independent_test.csv",
}
BASEDRUGBANK_RESULTS = {
    "gene": {"auc": 0.9373, "aupr": 0.8506, "f1_score": 0.7927},
    "disease": {"auc": 0.9048, "aupr": 0.7442, "f1_score": 0.7755},
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
    if group not in TARGET_GROUPS or metric_name not in TARGET_METRICS:
        return None
    return group, metric_name


def prepare_wandb_best(wandb_stats_path: Path) -> pd.DataFrame:
    df = pd.read_csv(wandb_stats_path)
    required = {"series", "metric", "max", "count"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"wandb stats missing columns: {sorted(missing)}")

    df["max"] = pd.to_numeric(df["max"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    df = df[df["metric"].isin(TARGET_WANDB_METRICS)]
    df = df[df["series"] != "no_resnet"]
    df = df[(df["count"] > 0) & (df["max"].notna())]

    if df.empty:
        raise SystemExit("No valid wandb rows after filtering out no_resnet.")

    idx = df.groupby("metric")["max"].idxmax()
    best = df.loc[idx, ["metric", "series", "max"]].copy()

    parsed = best["metric"].map(parse_wandb_metric)
    best = best[parsed.notna()].copy()
    best[["group", "metric_name"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=best.index)
    best = best.rename(columns={"max": "wandb_best", "series": "wandb_best_series"})

    return best[["group", "metric_name", "wandb_best", "wandb_best_series"]]


def prepare_independent_methods(independent_dir: Path) -> pd.DataFrame:
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

        for metric in TARGET_METRICS:
            if metric not in df.columns:
                print(f"[WARN] missing metric column {metric}: {csv_path}")
                continue

            numeric_col = pd.to_numeric(df[metric], errors="coerce")
            for group in TARGET_GROUPS:
                values = numeric_col[df["group"] == group].dropna()
                if values.empty:
                    mean_val = None
                    min_val = None
                    count = 0
                else:
                    mean_val = float(values.mean())
                    min_val = float(values.min())
                    count = int(values.shape[0])

                rows.append(
                    {
                        "method": method,
                        "source_csv": filename,
                        "group": group,
                        "metric_name": metric,
                        "independent_mean": mean_val,
                        "independent_min": min_val,
                        "count": count,
                    }
                )

    # Add manually provided BaseDrugBank benchmark as another comparison method.
    for group in TARGET_GROUPS:
        for metric in ["auc", "aupr", "f1_score"]:
            value = BASEDRUGBANK_RESULTS[group][metric]
            rows.append(
                {
                    "method": "BASEDRUGBANK",
                    "source_csv": "manual_basedrugbank",
                    "group": group,
                    "metric_name": metric,
                    "independent_mean": value,
                    "independent_min": value,
                    "count": 1,
                }
            )

    if not rows:
        raise SystemExit("No independent method rows parsed.")

    out = pd.DataFrame(rows)
    return out.sort_values(by=["group", "metric_name", "method"]).reset_index(drop=True)


def build_comparison_table(independent_df: pd.DataFrame, wandb_best: pd.DataFrame) -> pd.DataFrame:
    merged = independent_df.merge(wandb_best, on=["group", "metric_name"], how="inner")
    if merged.empty:
        raise SystemExit("No overlapping metrics between independent methods and wandb best.")

    merged["delta_vs_method_mean"] = merged["wandb_best"] - merged["independent_mean"]
    merged["gain_vs_method_mean_pct"] = np.where(
        merged["independent_mean"] != 0,
        merged["delta_vs_method_mean"] / merged["independent_mean"] * 100.0,
        np.nan,
    )

    merged["delta_vs_method_min"] = merged["wandb_best"] - merged["independent_min"]
    merged["gain_vs_method_min_pct"] = np.where(
        merged["independent_min"] != 0,
        merged["delta_vs_method_min"] / merged["independent_min"] * 100.0,
        np.nan,
    )

    return merged.sort_values(by=["group", "metric_name", "method"]).reset_index(drop=True)


def draw_method_metric_chart(
    subset: pd.DataFrame,
    value_column: str,
    value_tag: str,
    output_dir: Path,
) -> Path:
    group = subset.iloc[0]["group"]
    metric = subset.iloc[0]["metric_name"]
    ours = float(subset.iloc[0]["wandb_best"])

    method_values = []
    for method in METHOD_ORDER:
        row = subset[subset["method"] == method]
        if row.empty:
            method_values.append(np.nan)
        else:
            method_values.append(float(row.iloc[0][value_column]))

    labels = [*METHOD_ORDER, "OURS"]
    values = [*method_values, ours]
    color_map = {
        "HAN": "#B0B7C3",
        "MLP": "#8B96A8",
        "RGCN": "#657289",
        "BASEDRUGBANK": "#34495E",
        "OURS": "#16A34A",
    }
    colors = [color_map[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1F2937", linewidth=1.0)

    numeric_vals = [v for v in values if pd.notna(v)]
    y_max = max(numeric_vals)
    y_min = min(numeric_vals)
    span = max(0.02, y_max - y_min)
    ax.set_ylim(max(0.0, y_min - 0.18 * span), min(1.02, y_max + 0.38 * span))

    for bar, value in zip(bars, values):
        if pd.isna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                ax.get_ylim()[0] + 0.02,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.01 * span,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(f"{group.upper()} | {metric.upper()}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    out_path = output_dir / f"compare_{group}_{metric}_{value_tag}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def draw_overview_chart(table: pd.DataFrame, value_column: str, output_dir: Path) -> Path:
    index_rows = [(g, m) for g in TARGET_GROUPS for m in TARGET_METRICS]

    method_pivot = table.pivot_table(
        index=["group", "metric_name"],
        columns="method",
        values=value_column,
        aggfunc="first",
    )
    ours_series = table.groupby(["group", "metric_name"], as_index=True)["wandb_best"].first()

    plot_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(index_rows, names=["group", "metric_name"]))
    for method in METHOD_ORDER:
        if method in method_pivot.columns:
            plot_df[method] = method_pivot[method]
        else:
            plot_df[method] = np.nan
    plot_df["OURS"] = ours_series
    plot_df = plot_df.reset_index()
    plot_df["label"] = plot_df["group"].str.upper() + "-" + plot_df["metric_name"].str.upper()

    x = np.arange(plot_df.shape[0])
    all_methods = [*METHOD_ORDER, "OURS"]
    n_methods = len(all_methods)
    width = 0.15

    fig, ax = plt.subplots(figsize=(13.0, 5.8))
    colors = {
        "HAN": "#B0B7C3",
        "MLP": "#8B96A8",
        "RGCN": "#657289",
        "BASEDRUGBANK": "#34495E",
        "OURS": "#16A34A",
    }

    center = (n_methods - 1) / 2.0
    for i, method in enumerate(all_methods):
        ax.bar(
            x + (i - center) * width,
            plot_df[method],
            width=width,
            label=method,
            color=colors[method],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=0)
    ax.set_ylabel("Score")
    label_map = {"independent_mean": "Mean", "independent_min": "Min"}
    view_label = label_map.get(value_column, value_column)
    ax.set_title(f"Method-wise Comparison Across 6 Metrics ({view_label})", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=5)

    plt.tight_layout()
    out_path = output_dir / f"compare_overview_{value_column}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HAN/MLP/RGCN independent_test methods separately against "
            "best wandb performance (excluding no_resnet)."
        )
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
        default=Path("results_old"),
        help="Directory that contains han/mlp/rgcn independent_test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "comparison_plots",
        help="Directory to save plots and comparison table.",
    )
    parser.add_argument(
        "--with-overview",
        action="store_true",
        help="Also save overview charts that combine all metrics into one figure.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_dir = args.output_dir / "single_metric_mean"
    min_dir = args.output_dir / "single_metric_min"
    mean_dir.mkdir(parents=True, exist_ok=True)
    min_dir.mkdir(parents=True, exist_ok=True)

    wandb_best = prepare_wandb_best(args.wandb_stats)
    independent = prepare_independent_methods(args.independent_dir)
    comparison = build_comparison_table(independent, wandb_best)

    table_out = args.output_dir / "comparison_table_by_method.csv"
    comparison.to_csv(table_out, index=False)

    saved_paths: list[Path] = []
    for group in TARGET_GROUPS:
        for metric in TARGET_METRICS:
            subset = comparison[
                (comparison["group"] == group) & (comparison["metric_name"] == metric)
            ]
            if subset.empty:
                continue
            saved_paths.append(
                draw_method_metric_chart(
                    subset,
                    value_column="independent_mean",
                    value_tag="mean",
                    output_dir=mean_dir,
                )
            )
            saved_paths.append(
                draw_method_metric_chart(
                    subset,
                    value_column="independent_min",
                    value_tag="min",
                    output_dir=min_dir,
                )
            )

    if args.with_overview:
        saved_paths.append(draw_overview_chart(comparison, "independent_mean", args.output_dir))
        saved_paths.append(draw_overview_chart(comparison, "independent_min", args.output_dir))

    print(f"Saved table: {table_out}")
    for path in saved_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()