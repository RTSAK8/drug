import argparse
import re
from pathlib import Path

import pandas as pd


TARGET_METRICS = [
    "gene/test/auc",
    "gene/test/aupr",
    "gene/test/f1_score",
    "gene/test/mrr",
    "gene/test/hit@1",
    "gene/test/hit@3",
    "gene/test/hit@5",
    "disease/test/auc",
    "disease/test/aupr",
    "disease/test/f1_score",
    "disease/test/mrr",
    "disease/test/hit@1",
    "disease/test/hit@3",
    "disease/test/hit@5",
]


def extract_series_name(run_name: object) -> str | None:
    if pd.isna(run_name):
        return None

    text = str(run_name).strip()
    if not text:
        return None

    # Group names like "final_kv_1" ... "final_kv_5" into "final_kv".
    match = re.match(r"^(.*)_\d+$", text)
    if match:
        return match.group(1)
    return text


def mean_without_one_min(values: pd.Series) -> float | None:
    if values.empty:
        return None
    if values.shape[0] <= 1:
        return None

    # Remove exactly one occurrence of the minimum value, then average the rest.
    idx_to_drop = values.idxmin()
    trimmed = values.drop(index=idx_to_drop)
    if trimmed.empty:
        return None
    return float(trimmed.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize selected wandb test metrics by run name series: max, min, variance, mean, and mean excluding one minimum."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("wandb_export_2026-03-28T15_49_52.564+08_00.csv"),
        help="Path to wandb export CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "wandb_test_metrics_series_stats.csv",
        help="Path to output summary CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    available_metrics = [col for col in TARGET_METRICS if col in df.columns]
    missing = [col for col in TARGET_METRICS if col not in df.columns]
    if missing:
        print(f"[INFO] Missing columns (skipped): {missing}")
    if not available_metrics:
        raise SystemExit("No target wandb test metric columns found in input file.")

    if "Name" not in df.columns:
        raise SystemExit("Missing required column: Name")

    work_df = df[["Name", *available_metrics]].copy()
    work_df["series"] = work_df["Name"].map(extract_series_name)
    work_df = work_df[work_df["series"].notna()]

    for col in available_metrics:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    summary_rows: list[dict[str, object]] = []

    all_series = sorted(work_df["series"].dropna().unique())
    for series in all_series:
        group_df = work_df[work_df["series"] == series]
        for metric in available_metrics:
            values = group_df[metric].dropna()

            if values.empty:
                max_val = None
                min_val = None
                var_val = None
                mean_val = None
                mean_wo_min_val = None
                count = 0
            else:
                max_val = values.max()
                min_val = values.min()
                var_val = values.var(ddof=0)
                mean_val = values.mean()
                mean_wo_min_val = mean_without_one_min(values)
                count = int(values.shape[0])

            summary_rows.append(
                {
                    "series": series,
                    "metric": metric,
                    "max": max_val,
                    "min": min_val,
                    "variance": var_val,
                    "mean": mean_val,
                    "mean_without_one_min": mean_wo_min_val,
                    "count": count,
                }
            )

    if not summary_rows:
        raise SystemExit("No valid numeric values found for target metrics.")

    out_df = pd.DataFrame(summary_rows)
    out_df = out_df.sort_values(by=["series", "metric"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved summary to: {args.output}")


if __name__ == "__main__":
    main()