import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_FILES = [
    "han_all_nodes_summary_new.csv",
    "han_independent_test.csv",
    "mlp_independent_test.csv",
    "mlp_summary.csv",
    "rgcn_independent_test.csv",
    "rgcn_summary_new.csv",
]


def normalize_mode(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "gene" in text:
        return "gene"
    if "disease" in text:
        return "disease"
    return None


def detect_metric_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "mode", "model", "ablation", "fold", "group"}
    metric_cols: list[str] = []

    for col in df.columns:
        if col in excluded:
            continue
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if numeric_series.notna().any():
            df[col] = numeric_series
            metric_cols.append(col)

    return metric_cols


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = df["mode"].map(normalize_mode)
    df = df[df["group"].notna()]

    if "fold" in df.columns:
        fold_text = df["fold"].astype(str).str.upper().str.strip()
        df = df[~fold_text.isin(["MEAN", "STD"])]

    return df


def summarize_file(file_path: Path) -> list[dict[str, object]]:
    raw_df = pd.read_csv(file_path)
    if "mode" not in raw_df.columns:
        return []

    df = clean_rows(raw_df)
    if df.empty:
        return []

    metric_cols = detect_metric_columns(df)
    if not metric_cols:
        return []

    rows: list[dict[str, object]] = []
    for group, group_df in df.groupby("group"):
        for metric in metric_cols:
            values = group_df[metric].dropna()
            if values.empty:
                continue

            rows.append(
                {
                    "source_csv": file_path.name,
                    "group": group,
                    "metric": metric,
                    "max": values.max(),
                    "min": values.min(),
                    "variance": values.var(ddof=0),
                    "count": int(values.shape[0]),
                }
            )

    return rows


def existing_files(results_dir: Path, file_names: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for name in file_names:
        candidate = results_dir / name
        if candidate.exists():
            paths.append(candidate)
        else:
            print(f"[WARN] File not found, skipped: {candidate}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute max/min/variance per metric split by gene/disease for result CSV files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result CSV files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="CSV filenames to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "metrics_gene_disease_stats.csv",
        help="Output summary CSV path.",
    )
    args = parser.parse_args()

    files = existing_files(args.results_dir, args.files)
    if not files:
        raise SystemExit("No input CSV files found.")

    summary_rows: list[dict[str, object]] = []
    for csv_path in files:
        summary_rows.extend(summarize_file(csv_path))

    if not summary_rows:
        raise SystemExit("No valid rows or numeric metrics found in selected files.")

    out_df = pd.DataFrame(summary_rows)
    out_df = out_df.sort_values(by=["source_csv", "group", "metric"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved summary to: {args.output}")


if __name__ == "__main__":
    main()