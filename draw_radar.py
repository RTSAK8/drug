"""Draw ICD-11 disease-category radar plots from per-fold category metric CSV files.

Typical use:
    python draw_radar.py \
      --inputs results/category_metrics \
      --metrics auc aupr recall@10 \
      --method-order OURS BASEDRUGBANK RGCN HAN MLP \
      --style example \
      --hide-radial-labels \
      --output outputs/icd11_radar.svg
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


DEFAULT_CATEGORY_ORDER = [
    "Infectious",
    "Neoplasms",
    "Hematologic & Immune",
    "Endocrine & Metabolic",
    "Neuropsychiatric & Sensory",
    "Cardiovascular",
    "Respiratory",
    "Digestive",
    "Skin & Musculoskeletal",
    "Developmental",
    "Symptoms & Care",
    "Others",
]

PAPER_CATEGORY_ORDER = [
    "Infectious",
    "Neoplasms",
    "Hematol.",
    "Endocrine",
    "Psychiatr.",
    "Neurol.",
    "Ophthalm.",
    "Cardiovasc.",
    "Respiratory",
    "Gastrointest.",
    "Dermatol.",
    "Musculosk.",
    "Others",
]

CATEGORY_LABELS_EN = {
    "Infectious": "Infectious",
    "Neoplasms": "Neoplasms",
    "Hematologic & Immune": "Hematol. & immune",
    "Endocrine & Metabolic": "Endocrine & metab.",
    "Neuropsychiatric & Sensory": "Neuropsych. & sensory",
    "Cardiovascular": "Cardiovascular",
    "Respiratory": "Respiratory",
    "Digestive": "Digestive",
    "Skin & Musculoskeletal": "Skin & musculoskel.",
    "Developmental": "Developmental",
    "Symptoms & Care": "Symptoms & care",
    "Hematol.": "Hematol.",
    "Endocrine": "Endocrine",
    "Psychiatr.": "Psychiatr.",
    "Neurol.": "Neurol.",
    "Ophthalm.": "Ophthalm.",
    "Cardiovasc.": "Cardiovasc.",
    "Gastrointest.": "Gastrointest.",
    "Dermatol.": "Dermatol.",
    "Musculosk.": "Musculosk.",
    "Genitourin.": "Genitourin.",
    "Others": "Others",
}

CATEGORY_LABELS_ZH = {
    "Infectious": "感染性疾病",
    "Neoplasms": "肿瘤",
    "Hematologic & Immune": "血液与免疫",
    "Endocrine & Metabolic": "内分泌与代谢",
    "Neuropsychiatric & Sensory": "神经精神与感觉",
    "Cardiovascular": "心血管",
    "Respiratory": "呼吸系统",
    "Digestive": "消化系统",
    "Skin & Musculoskeletal": "皮肤与肌骨",
    "Developmental": "发育相关",
    "Symptoms & Care": "症状与医疗接触",
    "Hematol.": "血液与免疫",
    "Endocrine": "内分泌与代谢",
    "Psychiatr.": "精神疾病",
    "Neurol.": "神经系统",
    "Ophthalm.": "眼科疾病",
    "Cardiovasc.": "心血管",
    "Gastrointest.": "胃肠道",
    "Dermatol.": "皮肤疾病",
    "Musculosk.": "肌肉骨骼",
    "Genitourin.": "泌尿生殖",
    "Others": "其他",
}

METHOD_COLORS = {
    "OURS": "#DC2626",
    "INTERACLE": "#DC2626",
    "HAN": "#A855A5",
    "MLP": "#F28E2B",
    "RGCN": "#169B62",
    "BASEDRUGBANK": "#2F6B8A",
    "BASE": "#2F6B8A",
    "GAT": "#4E79A7",
    "TXGNN": "#9C755F",
    "DREAMWALK": "#3FA7A3",
    "GTN": "#F28E2B",
}

METHOD_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", ">"]

ICD11_CHAPTER_TO_PAPER_CLASS = {
    "Certain infectious or parasitic diseases": "Infectious",
    "Neoplasms": "Neoplasms",
    "Diseases of the blood or blood-forming organs": "Hematol.",
    "Diseases of the immune system": "Hematol.",
    "Endocrine, nutritional or metabolic diseases": "Endocrine",
    "Mental, behavioural or neurodevelopmental disorders": "Psychiatr.",
    "Sleep-wake disorders": "Psychiatr.",
    "Diseases of the nervous system": "Neurol.",
    "Diseases of the visual system": "Ophthalm.",
    "Diseases of the circulatory system": "Cardiovasc.",
    "Diseases of the respiratory system": "Respiratory",
    "Diseases of the digestive system": "Gastrointest.",
    "Diseases of the skin": "Dermatol.",
    "Diseases of the musculoskeletal system or connective tissue": "Musculosk.",
    "Diseases of the genitourinary system": "Genitourin.",
    "Conditions related to sexual health": "Genitourin.",
    "Pregnancy, childbirth or the puerperium": "Genitourin.",
    "Certain conditions originating in the perinatal period": "Others",
    "Developmental anomalies": "Others",
    "Symptoms, signs or clinical findings, not elsewhere classified": "Others",
    "Factors influencing health status or contact with health services": "Others",
}

METRIC_TITLES = {
    "auc": "auROC",
    "aupr": "auPRC",
    "f1_score": "F1",
    "mrr": "MRR",
    "recall@1": "Recall@1",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "recall@50": "Recall@50",
    "recall@100": "Recall@100",
    "hit@1": "Hit@1",
    "hit@3": "Hit@3",
    "hit@5": "Hit@5",
}


def discover_csv_files(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.csv")))
        else:
            files.extend(Path(p) for p in sorted(glob.glob(item, recursive=True)))

    unique: List[Path] = []
    seen = set()
    for path in files:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_category_results(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    required = {"method", "category"}

    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Cannot read {path}: {exc}")
            continue
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["source_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise SystemExit(
            "No category-metric CSVs found. Run the modified experiment first; "
            "the ordinary overall results.csv does not contain disease-category detail."
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def apply_paper_category_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    if "category" not in frame.columns:
        raise ValueError("Category metric data must contain a 'category' column")

    work = frame.copy()

    def _convert(value: object) -> str:
        text = str(value).strip()
        if text in PAPER_CATEGORY_ORDER:
            return text
        return ICD11_CHAPTER_TO_PAPER_CLASS.get(text, "Others")

    work["category"] = work["category"].map(_convert)
    work["category_en"] = work["category"]
    work["category_zh"] = work["category"].map(CATEGORY_LABELS_ZH).fillna("其他")
    return work


def aggregate_results(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    aggregation: str,
) -> pd.DataFrame:
    for metric in metrics:
        if metric not in frame.columns:
            raise SystemExit(f"Missing metric column: {metric}")
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    group_cols = ["method", "category"]
    label_cols = [col for col in ("category_en", "category_zh") if col in frame.columns]

    if aggregation == "weighted_mean":
        rows = []
        for (method, category), group in frame.groupby(group_cols, dropna=False):
            row = {"method": method, "category": category}
            for col in label_cols:
                nonempty = group[col].dropna().astype(str)
                row[col] = nonempty.iloc[0] if not nonempty.empty else category

            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce")
                is_ranking_metric = (
                    metric.startswith("recall@")
                    or metric.startswith("hit@")
                    or metric == "mrr"
                )
                weight_col = "positive_query_count" if is_ranking_metric else "binary_sample_count"
                weight_source = (
                    group[weight_col]
                    if weight_col in group.columns
                    else pd.Series(1.0, index=group.index)
                )
                weights = pd.to_numeric(weight_source, errors="coerce").fillna(0.0)
                valid = values.notna() & (weights > 0)
                row[metric] = (
                    float(np.average(values[valid], weights=weights[valid]))
                    if valid.any()
                    else np.nan
                )

            row["source_rows"] = int(len(group))
            rows.append(row)
        return pd.DataFrame(rows)

    if aggregation not in {"mean", "median", "max", "min"}:
        raise SystemExit(f"Unsupported aggregation: {aggregation}")

    agg_dict = {metric: aggregation for metric in metrics}
    for col in label_cols:
        agg_dict[col] = "first"

    result = frame.groupby(group_cols, as_index=False, dropna=False).agg(agg_dict)
    counts = (
        frame.groupby(group_cols, as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "source_rows"})
    )
    return result.merge(counts, on=group_cols, how="left")


def resolve_methods(values: Iterable[str], requested: Sequence[str]) -> List[str]:
    available = [str(x) for x in pd.Series(list(values)).dropna().unique().tolist()]
    if requested:
        missing = [m for m in requested if m not in available]
        if missing:
            print(f"[WARN] Requested methods not found: {missing}")
        ordered = [m for m in requested if m in available]
        ordered += sorted(m for m in available if m not in ordered)
        return ordered

    preferred = ["GTN", "HAN", "DREAMWALK", "GAT", "RGCN", "TXGNN", "INTERACLE", "OURS"]
    return [m for m in preferred if m in available] + sorted(
        m for m in available if m not in preferred
    )


def resolve_categories(values: Iterable[str], requested: Sequence[str]) -> List[str]:
    available = [str(x) for x in pd.Series(list(values)).dropna().unique().tolist()]
    if requested:
        return [c for c in requested if c in available] + sorted(
            c for c in available if c not in requested
        )

    if any(c in available for c in ("Hematol.", "Psychiatr.", "Neurol.", "Cardiovasc.")):
        base_order = PAPER_CATEGORY_ORDER
    else:
        base_order = DEFAULT_CATEGORY_ORDER

    ordered = [c for c in base_order if c in available]
    ordered += sorted(c for c in available if c not in ordered and c != "Others")
    if "Others" in available and "Others" not in ordered:
        ordered.append("Others")
    return ordered


def filter_complete_categories(
    aggregate: pd.DataFrame,
    methods: Sequence[str],
    categories: Sequence[str],
    metrics: Sequence[str],
    missing_policy: str,
) -> tuple[pd.DataFrame, List[str]]:
    work = aggregate.copy()
    complete_categories: List[str] = []
    incomplete_categories: List[str] = []

    for category in categories:
        subset = work[(work["category"] == category) & (work["method"].isin(methods))]
        complete = True
        for method in methods:
            method_row = subset[subset["method"] == method]
            if method_row.empty or method_row[list(metrics)].iloc[0].isna().any():
                complete = False
                break
        if complete:
            complete_categories.append(category)
        else:
            incomplete_categories.append(category)

    if missing_policy == "drop":
        retained = complete_categories
        if incomplete_categories:
            print(
                "[WARN] Categories removed because at least one plotted value is missing: "
                f"{incomplete_categories}"
            )
        work = work[work["category"].isin(retained)]
    elif missing_policy == "zero":
        retained = list(categories)
        full_index = pd.MultiIndex.from_product([methods, retained], names=["method", "category"])
        work = work.set_index(["method", "category"]).reindex(full_index).reset_index()
        work[list(metrics)] = work[list(metrics)].fillna(0.0)
    elif missing_policy == "error":
        if incomplete_categories:
            raise SystemExit(
                "Missing method/category/metric values for categories: "
                f"{incomplete_categories}"
            )
        retained = complete_categories
    else:
        raise SystemExit(f"Unsupported missing policy: {missing_policy}")

    if len(retained) < 3:
        raise SystemExit("At least three complete disease categories are required for a radar plot.")
    return work, retained


def parse_radial_min(value: str, aggregate: pd.DataFrame, metrics: Sequence[str]) -> float:
    if value.lower() != "auto":
        return float(value)
    numeric = aggregate[list(metrics)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return 0.0
    minimum = max(0.0, float(finite.min()) - 0.05)
    return math.floor(minimum * 10.0) / 10.0


def category_labels(categories: Sequence[str], language: str) -> List[str]:
    mapping = CATEGORY_LABELS_ZH if language == "zh" else CATEGORY_LABELS_EN
    return [mapping.get(category, category) for category in categories]


def draw_radar(
    aggregate: pd.DataFrame,
    methods: Sequence[str],
    categories: Sequence[str],
    metrics: Sequence[str],
    output: Path,
    table_output: Path | None,
    language: str,
    radial_min: float,
    radial_max: float,
    title: str,
    mode_label: str,
    style: str = "default",
    panel_label: str = "",
    show_radial_tick_labels: bool = True,
) -> None:
    available_fonts = {item.name for item in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Arial",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "SimHei",
        "DejaVu Sans",
    ]
    selected_font = next((name for name in preferred_fonts if name in available_fonts), "DejaVu Sans")
    plt.rcParams["font.family"] = selected_font
    plt.rcParams["axes.unicode_minus"] = False

    n_axes = len(categories)
    angles = np.linspace(0.0, 2.0 * np.pi, n_axes, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    labels = category_labels(categories, language)

    is_example = style == "example"
    if is_example:
        figsize = (12.4, 3.25)
    else:
        width_per_panel = max(4.0, 0.38 * n_axes)
        figsize = (width_per_panel * len(metrics), max(4.8, width_per_panel + 0.6))

    fig, axes = plt.subplots(
        1,
        len(metrics),
        subplot_kw={"polar": True},
        figsize=figsize,
    )
    if len(metrics) == 1:
        axes = [axes]

    fallback_colors = plt.get_cmap("tab10")
    legend_handles = []

    for metric_index, (ax, metric) in enumerate(zip(axes, metrics)):
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)

        for method_index, method in enumerate(methods):
            method_frame = aggregate[aggregate["method"] == method].set_index("category")
            values = [float(method_frame.at[category, metric]) for category in categories]
            closed_values = np.asarray(values + values[:1], dtype=float)
            color = METHOD_COLORS.get(method.upper(), fallback_colors(method_index % 10))
            marker = "o" if is_example else METHOD_MARKERS[method_index % len(METHOD_MARKERS)]
            is_ours = method.upper() in {"OURS", "INTERACLE"}

            if is_example:
                linewidth = 1.80 if is_ours else 1.35
            else:
                linewidth = 2.0 if is_ours else 1.6

            line = ax.plot(
                closed_angles,
                closed_values,
                color=color,
                linewidth=linewidth,
                marker=marker,
                markersize=2.8 if is_example else 4.5,
                markeredgewidth=0.0 if is_example else 0.6,
                label=method,
                zorder=3,
            )[0]

            if not is_example:
                ax.fill(closed_angles, closed_values, color=color, alpha=0.035, zorder=1)

            if metric_index == 0:
                legend_handles.append(line)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=7.2 if is_example else 8.8)
        ax.tick_params(axis="x", pad=2 if is_example else 8)
        ax.set_ylim(radial_min, radial_max)

        if is_example:
            radial_ticks = np.asarray([0.4, 0.6, 0.8], dtype=float)
            radial_ticks = radial_ticks[(radial_ticks >= radial_min) & (radial_ticks < radial_max)]
        else:
            tick_start = math.ceil(radial_min * 10.0) / 10.0
            radial_ticks = np.arange(tick_start, radial_max + 1e-9, 0.1)
            if len(radial_ticks) > 6:
                radial_ticks = np.linspace(tick_start, radial_max, 5)

        ax.set_yticks(radial_ticks)
        if show_radial_tick_labels:
            ax.set_yticklabels(
                [f"{x:.1f}" for x in radial_ticks],
                fontsize=7.0 if is_example else 8,
                color="#737373",
            )
        else:
            ax.set_yticklabels([])

        ax.set_rlabel_position(15)
        ax.grid(color="#D9D9D9", linewidth=0.55 if is_example else 0.8, alpha=0.9)
        ax.spines["polar"].set_color("#D9D9D9")
        ax.spines["polar"].set_linewidth(0.55 if is_example else 0.8)

        metric_title = METRIC_TITLES.get(metric, metric)
        if mode_label:
            ax.set_title(
                f"{metric_title}\n{mode_label}",
                fontsize=9.5 if is_example else 12,
                fontweight="bold",
                pad=9 if is_example else 18,
            )
        else:
            ax.set_title(
                metric_title,
                fontsize=9.5 if is_example else 12,
                fontweight="bold",
                pad=9 if is_example else 18,
            )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.04)

    legend_y = 1.01 if is_example else 1.02
    fig.legend(
        handles=legend_handles,
        labels=methods,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=min(len(methods), 8),
        frameon=False,
        fontsize=8.0 if is_example else 9.5,
        handlelength=1.7,
        columnspacing=1.25,
    )

    if panel_label:
        fig.text(
            0.012,
            0.91 if is_example else 0.96,
            panel_label,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )

    if is_example:
        fig.subplots_adjust(top=0.76, bottom=0.10, left=0.045, right=0.985, wspace=0.30)
    else:
        fig.subplots_adjust(top=0.82, bottom=0.06, left=0.04, right=0.98, wspace=0.30)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", transparent=False)
    plt.close(fig)

    if table_output is not None:
        table_output.parent.mkdir(parents=True, exist_ok=True)
        aggregate.sort_values(["method", "category"]).to_csv(table_output, index=False)
        print(f"Saved radar data table: {table_output}")

    print(f"Saved radar plot: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw disease-category radar plots from ICD-11 category metric CSVs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results/category_metrics"],
        help="CSV file(s), directory/directories, or glob patterns.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["auc", "aupr", "recall@10"],
        help="Metrics shown as separate radar panels.",
    )
    parser.add_argument("--methods", nargs="*", default=[], help="Only include these methods.")
    parser.add_argument("--method-order", nargs="*", default=[], help="Preferred method order.")
    parser.add_argument("--categories", nargs="*", default=[], help="Preferred category order.")
    parser.add_argument("--mode", default="", help="Filter mode, e.g. disease or gene.")
    parser.add_argument("--protocol", default="", help="Filter protocol, e.g. cross_validation.")
    parser.add_argument("--ablation", default="", help="Filter ablation name.")
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median", "max", "min", "weighted_mean"],
        default="mean",
        help="Aggregation across folds/runs.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["drop", "zero", "error"],
        default="drop",
        help="How to handle absent method/category values.",
    )
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--radial-min", default="auto", help="Float or auto.")
    parser.add_argument("--radial-max", type=float, default=1.0)
    parser.add_argument("--title", default="")
    parser.add_argument("--mode-label", default="")
    parser.add_argument(
        "--style",
        choices=["default", "example"],
        default="default",
        help="example reproduces the compact three-panel paper style.",
    )
    parser.add_argument("--panel-label", default="", help="Optional panel label such as c.")
    parser.add_argument(
        "--hide-radial-labels",
        action="store_true",
        help="Hide the 0.x radial tick labels inside the radar chart.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "icd11_radar.svg",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("outputs") / "icd11_radar_data.csv",
    )
    args = parser.parse_args()

    paths = discover_csv_files(args.inputs)
    if not paths:
        raise SystemExit(f"No CSV files matched: {args.inputs}")
    frame = load_category_results(paths)

    if args.mode and "mode" in frame.columns:
        frame = frame[frame["mode"].astype(str) == args.mode]
    if args.protocol and "protocol" in frame.columns:
        frame = frame[frame["protocol"].astype(str) == args.protocol]
    if args.ablation and "ablation" in frame.columns:
        frame = frame[frame["ablation"].astype(str) == args.ablation]
    if args.methods:
        frame = frame[frame["method"].astype(str).isin(args.methods)]
    if frame.empty:
        raise SystemExit("No rows remain after applying filters.")

    if args.style == "example":
        frame = apply_paper_category_mapping(frame)

    aggregate = aggregate_results(frame, args.metrics, args.aggregation)
    methods = resolve_methods(aggregate["method"], args.method_order)
    if args.methods:
        methods = [m for m in methods if m in args.methods]
    categories = resolve_categories(aggregate["category"], args.categories)

    aggregate, categories = filter_complete_categories(
        aggregate=aggregate,
        methods=methods,
        categories=categories,
        metrics=args.metrics,
        missing_policy=args.missing_policy,
    )
    aggregate = aggregate[
        aggregate["method"].isin(methods) & aggregate["category"].isin(categories)
    ].copy()

    radial_min = parse_radial_min(args.radial_min, aggregate, args.metrics)
    if radial_min >= args.radial_max:
        raise SystemExit("radial-min must be smaller than radial-max")

    draw_radar(
        aggregate=aggregate,
        methods=methods,
        categories=categories,
        metrics=args.metrics,
        output=args.output,
        table_output=args.output_table,
        language=args.language,
        radial_min=radial_min,
        radial_max=args.radial_max,
        title=args.title,
        mode_label=args.mode_label,
        style=args.style,
        panel_label=args.panel_label,
        show_radial_tick_labels=not args.hide_radial_labels,
    )


if __name__ == "__main__":
    main()