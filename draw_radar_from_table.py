"""Draw a radar figure directly from a manually edited table.

Expected CSV format (single method):
category,auc,aupr,recall@10
Infectious,0.98,0.96,1.00
Neoplasms,0.88,0.74,0.62
...

Optional multi-method format:
method,category,auc,aupr,recall@10
OURS,Infectious,0.98,0.96,1.00
HAN,Infectious,0.83,0.70,0.88
...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from draw_radar import (
    PAPER_CATEGORY_ORDER,
    draw_radar,
    resolve_categories,
    resolve_methods,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw a radar chart directly from a manually edited CSV table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the manually edited CSV table.",
    )
    parser.add_argument(
        "--method",
        default="OURS",
        help="Default method name when the input CSV has no 'method' column.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["auc", "aupr", "recall@10"],
        help="Metric columns to plot.",
    )
    parser.add_argument(
        "--language",
        choices=["en", "zh"],
        default="en",
    )
    parser.add_argument(
        "--style",
        choices=["default", "example"],
        default="example",
    )
    parser.add_argument(
        "--panel-label",
        default="c",
    )
    parser.add_argument(
        "--hide-radial-labels",
        action="store_true",
        help="Hide the 0.x radial tick labels inside the radar chart.",
    )
    parser.add_argument(
        "--radial-min",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--radial-max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "manual_radar.svg",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("outputs") / "manual_radar_used_data.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "category" not in df.columns:
        raise SystemExit("Input CSV must contain a 'category' column.")

    if "method" not in df.columns:
        df["method"] = str(args.method)

    for metric in args.metrics:
        if metric not in df.columns:
            raise SystemExit(f"Missing metric column: {metric}")
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if "category_en" not in df.columns:
        df["category_en"] = df["category"]
    if "category_zh" not in df.columns:
        df["category_zh"] = df["category"]

    methods = resolve_methods(df["method"], requested=[])
    categories = resolve_categories(df["category"], requested=PAPER_CATEGORY_ORDER)

    # 只保留实际存在的类别
    categories = [c for c in PAPER_CATEGORY_ORDER if c in set(df["category"].astype(str))]
    if "Others" in set(df["category"].astype(str)) and "Others" not in categories:
        categories.append("Others")

    if len(categories) < 3:
        raise SystemExit("At least three categories are required to draw a radar chart.")

    draw_radar(
        aggregate=df.copy(),
        methods=methods,
        categories=categories,
        metrics=args.metrics,
        output=args.output,
        table_output=args.output_table,
        language=args.language,
        radial_min=args.radial_min,
        radial_max=args.radial_max,
        title="",
        mode_label="",
        style=args.style,
        panel_label=args.panel_label,
        show_radial_tick_labels=not args.hide_radial_labels,
    )


if __name__ == "__main__":
    main()