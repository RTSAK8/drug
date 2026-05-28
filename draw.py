import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


TARGET_GROUPS = ["gene", "disease"]
TARGET_HIT_METRICS = ["hit@1", "hit@3", "hit@5"]
TARGET_SCORE_METRICS = ["auc", "aupr", "f1_score"]
METHOD_ORDER = ["HAN", "MLP", "RGCN", "BASEDRUGBANK", "OURS"]
METHOD_COLOR_MAP = {
    "HAN": "#2563EB",
    "MLP": "#D97706",
    "RGCN": "#16A34A",
    "BASEDRUGBANK": "#DC2626",
    "OURS": "#0EA5E9",
}


def get_ordered_methods(method_values: list[str]) -> list[str]:
    method_set = set(method_values)
    method_order = [m for m in METHOD_ORDER if m in method_set]
    extra = sorted([m for m in method_set if m not in METHOD_ORDER])
    return method_order + extra


def extract_series_name(run_name: object) -> str | None:
    if pd.isna(run_name):
        return None

    text = str(run_name).strip()
    if not text:
        return None

    # Merge names like final_kv_1 ... final_kv_5 into final_kv.
    match = re.match(r"^(.*)_\d+$", text)
    if match:
        return match.group(1)
    return text


def extract_method(series: str) -> str:
    lower = series.lower()
    if lower.startswith("han"):
        return "HAN"
    if lower.startswith("mlp"):
        return "MLP"
    if lower.startswith("rgcn"):
        return "RGCN"
    if lower.startswith("base"):
        return "BASEDRUGBANK"
    if lower.startswith("final") or lower.startswith("ours"):
        return "OURS"

    return series.split("_")[0].upper()


def load_metric_long_table(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise SystemExit(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "Name" not in df.columns:
        raise SystemExit("Missing required column: Name")

    all_metrics = TARGET_HIT_METRICS + TARGET_SCORE_METRICS
    available_cols: list[str] = []
    for group in TARGET_GROUPS:
        for metric in all_metrics:
            col = f"{group}.test.{metric}"
            if col in df.columns:
                available_cols.append(col)
            else:
                print(f"[WARN] Missing column, skipped: {col}")

    if not available_cols:
        raise SystemExit("No target test metric columns found in input CSV.")

    work = df[["Name", *available_cols]].copy()
    work["series"] = work["Name"].map(extract_series_name)
    work = work[work["series"].notna()].copy()
    work["method"] = work["series"].map(extract_method)

    rows: list[dict[str, object]] = []
    for _, row in work.iterrows():
        for group in TARGET_GROUPS:
            for metric in all_metrics:
                col = f"{group}.test.{metric}"
                if col not in work.columns:
                    continue
                value = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(value):
                    continue

                rows.append(
                    {
                        "series": row["series"],
                        "method": row["method"],
                        "group": group,
                        "metric_name": metric,
                        "score": float(value),
                    }
                )

    if not rows:
        raise SystemExit("No valid numeric values found in target metrics.")

    return pd.DataFrame(rows)


def aggregate_table(hit_long: pd.DataFrame, score_agg: str) -> pd.DataFrame:
    agg = (
        hit_long.groupby(["method", "group", "metric_name"], as_index=False)["score"]
        .agg(score_agg)
        .sort_values(by=["group", "metric_name", "method"])
        .reset_index(drop=True)
    )
    return agg


def build_plot_df(agg_table: pd.DataFrame, group: str, metric_names: list[str]) -> pd.DataFrame:
    group_df = agg_table[agg_table["group"] == group].copy()
    if group_df.empty:
        raise SystemExit(f"No data for group: {group}")

    pivot = group_df.pivot(index="method", columns="metric_name", values="score")

    ordered = get_ordered_methods(pivot.index.tolist())
    pivot = pivot.reindex(ordered)

    for metric in metric_names:
        if metric not in pivot.columns:
            pivot[metric] = np.nan

    return pivot[metric_names]


def get_ylims(plot_df: pd.DataFrame, metric_names: list[str]) -> tuple[float, float]:
    values = pd.to_numeric(plot_df[metric_names].stack(), errors="coerce").dropna()
    if values.empty:
        raise SystemExit("No numeric values left for plotting.")

    y_min = float(values.min())
    y_max = float(values.max())
    span = max(0.02, y_max - y_min)
    return max(0.0, y_min - 0.15 * span), min(1.02, y_max + 0.2 * span)


def draw_line_chart(
    plot_df: pd.DataFrame,
    group: str,
    metric_names: list[str],
    chart_title: str,
    output_path: Path,
) -> Path:
    x = np.arange(plot_df.shape[0])

    marker_map = {
        "hit@1": "^",
        "hit@3": "o",
        "hit@5": "s",
        "auc": "D",
        "aupr": "o",
        "f1_score": "^",
    }
    color_map = {
        "hit@1": "#2563EB",
        "hit@3": "#D97706",
        "hit@5": "#16A34A",
        "auc": "#0EA5E9",
        "aupr": "#F97316",
        "f1_score": "#DC2626",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for metric in metric_names:
        y = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker=marker_map.get(metric, "o"),
            markersize=7,
            linewidth=2.0,
            label=metric.upper(),
            color=color_map.get(metric, "#334155"),
        )

    ymin, ymax = get_ylims(plot_df, metric_names)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index.tolist())
    ax.set_ylabel("Score")
    ax.set_title(f"{chart_title} Line Chart ({group.upper()})", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(title="Metric", ncol=3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def draw_score_metric_detail_chart(
    metric_long: pd.DataFrame,
    agg_table: pd.DataFrame,
    group: str,
    metric_name: str,
    score_agg: str,
    output_path: Path,
) -> Path:
    detail_df = metric_long[
        (metric_long["group"] == group) & (metric_long["metric_name"] == metric_name)
    ].copy()
    detail_df["score"] = pd.to_numeric(detail_df["score"], errors="coerce")
    detail_df = detail_df[detail_df["score"].notna()]
    if detail_df.empty:
        raise SystemExit(f"No numeric rows for detail chart: group={group}, metric={metric_name}")

    methods = get_ordered_methods(detail_df["method"].tolist())
    x = np.arange(len(methods))
    fallback_colors = ["#475569", "#7C3AED", "#0F766E", "#B45309", "#BE123C"]
    method_colors = {
        method: METHOD_COLOR_MAP.get(method, fallback_colors[i % len(fallback_colors)])
        for i, method in enumerate(methods)
    }

    agg_df = agg_table[
        (agg_table["group"] == group) & (agg_table["metric_name"] == metric_name)
    ].copy()
    agg_df["score"] = pd.to_numeric(agg_df["score"], errors="coerce")
    agg_map = dict(zip(agg_df["method"], agg_df["score"]))
    bar_values = np.array([float(agg_map.get(m, np.nan)) for m in methods], dtype=float)

    distributions: list[np.ndarray] = []
    for method in methods:
        vals = detail_df.loc[detail_df["method"] == method, "score"].to_numpy(dtype=float)
        distributions.append(vals)

    fig, ax = plt.subplots(figsize=(10.2, 5.6))

    for i, method in enumerate(methods):
        base_color = method_colors[method]

        ax.bar(
            x[i],
            bar_values[i],
            width=0.62,
            color=base_color,
            edgecolor=base_color,
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )

        ax.boxplot(
            [distributions[i]],
            positions=[x[i]],
            widths=0.36,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": base_color,
                "alpha": 0.20,
                "edgecolor": base_color,
                "linewidth": 1.2,
            },
            whiskerprops={"color": base_color, "linewidth": 1.1},
            capprops={"color": base_color, "linewidth": 1.1},
            medianprops={"color": base_color, "linewidth": 1.6},
            zorder=2,
        )

    rng = np.random.default_rng(42)
    for i, vals in enumerate(distributions):
        jitter = rng.normal(loc=x[i], scale=0.045, size=vals.shape[0])
        base_color = method_colors[methods[i]]
        ax.scatter(
            jitter,
            vals,
            s=24,
            color=base_color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.80,
            zorder=3,
        )

    stacked = np.concatenate(distributions + [bar_values[~np.isnan(bar_values)]])
    y_min = float(np.min(stacked))
    y_max = float(np.max(stacked))
    span = max(0.02, y_max - y_min)
    ax.set_ylim(max(0.0, y_min - 0.15 * span), min(1.02, y_max + 0.2 * span))

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title(
        f"{metric_name.upper()} Detail Chart ({group.upper()})",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)

    legend_handles = [
        Patch(facecolor="#94A3B8", edgecolor="#64748B", alpha=0.50, label=f"{score_agg.upper()} bar"),
        Patch(facecolor="#94A3B8", edgecolor="#64748B", alpha=0.25, label="Box (run distribution)"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#475569",
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=6,
            label="Run points",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def draw_grouped_bar_chart(
    plot_df: pd.DataFrame,
    group: str,
    metric_names: list[str],
    chart_title: str,
    output_path: Path,
) -> Path:
    x = np.arange(plot_df.shape[0])
    width = 0.24 if len(metric_names) <= 3 else 0.18

    color_map = {
        "hit@1": "#2563EB",
        "hit@3": "#D97706",
        "hit@5": "#16A34A",
        "auc": "#0EA5E9",
        "aupr": "#F97316",
        "f1_score": "#DC2626",
    }

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    offset_center = (len(metric_names) - 1) / 2.0
    for i, metric in enumerate(metric_names):
        y = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(
            x + (i - offset_center) * width,
            y,
            width=width,
            label=metric.upper(),
            color=color_map.get(metric, "#334155"),
            alpha=0.9,
        )

    ymin, ymax = get_ylims(plot_df, metric_names)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index.tolist())
    ax.set_ylabel("Score")
    ax.set_title(f"{chart_title} Grouped Bar Chart ({group.upper()})", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(title="Metric", ncol=3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read results CSV and draw line/grouped-bar charts for gene/disease metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "results.csv",
        help="Input CSV path (default: results/results.csv).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=TARGET_GROUPS,
        default=TARGET_GROUPS,
        help="Groups to draw (default: gene disease).",
    )
    parser.add_argument(
        "--score-agg",
        choices=["mean", "max", "min", "median"],
        default="mean",
        help="Aggregation over multiple runs for each method/group/metric.",
    )
    parser.add_argument(
        "--metric-types",
        nargs="+",
        choices=["hitk", "score"],
        default=["hitk", "score"],
        help="Metric groups to draw: hitk (hit@1/3/5), score (auc/aupr/f1_score).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "comparison_plots",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("outputs") / "comparison_plots" / "metrics_aggregated_table.csv",
        help="Path to save aggregated plotting table.",
    )
    args = parser.parse_args()

    metric_long = load_metric_long_table(args.input)
    agg_table = aggregate_table(metric_long, args.score_agg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    agg_table.to_csv(args.output_table, index=False)
    print(f"Saved aggregated table: {args.output_table}")

    saved_paths: list[Path] = []
    metric_groups = {
        "hitk": {
            "metrics": TARGET_HIT_METRICS,
            "title": "Hit@K",
            "prefix": "hitk",
        },
        "score": {
            "metrics": TARGET_SCORE_METRICS,
            "title": "AUC/AUPR/F1",
            "prefix": "score",
        },
    }

    for group in args.groups:
        for metric_type in args.metric_types:
            spec = metric_groups[metric_type]
            metrics = spec["metrics"]
            title = spec["title"]
            prefix = spec["prefix"]

            plot_df = build_plot_df(agg_table, group, metrics)

            line_path = args.output_dir / f"{prefix}_line_{group}.png"
            bar_path = args.output_dir / f"{prefix}_bar_{group}.png"

            saved_paths.append(draw_line_chart(plot_df, group, metrics, title, line_path))
            saved_paths.append(draw_grouped_bar_chart(plot_df, group, metrics, title, bar_path))

            if metric_type == "score":
                for single_metric in TARGET_SCORE_METRICS:
                    detail_path = args.output_dir / f"score_detail_{single_metric}_{group}.png"
                    saved_paths.append(
                        draw_score_metric_detail_chart(
                            metric_long=metric_long,
                            agg_table=agg_table,
                            group=group,
                            metric_name=single_metric,
                            score_agg=args.score_agg,
                            output_path=detail_path,
                        )
                    )

    for path in saved_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
