"""ICD-11 disease-category evaluation and result export utilities.

The important correction in this version is query-based grouping for missing-disease
classification.  Every positive and negative candidate belonging to one query is
assigned to the category of the query's true disease.  Candidate diseases are never
used to decide the category of a negative sample.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


DEFAULT_CLASS_COLUMNS = (
    "Paper_Radar_Class_EN",
    "Paper_Radar_Class_ZH",
    "ICD11_Chapter_Name_EN",
    "ICD11_Chapter_Name_ZH",
    "Radar_Class_EN",
    "Radar_Class_ZH",
)
DEFAULT_RECALL_K = (1, 5, 10, 50, 100)


def _normalise_scalar_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def _safe_filename(value: object) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def load_disease_category_mapping(
    mapping_path: str | os.PathLike[str],
    class_column: str = "Paper_Radar_Class_EN",
) -> Tuple[Dict[str, Dict[str, object]], str]:
    """Load the disease mapping and return a Disease_ID keyed dictionary."""
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Disease mapping file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name="Mapping")
    else:
        frame = pd.read_csv(path)

    if "Disease_ID" not in frame.columns:
        raise ValueError(f"Mapping file must contain Disease_ID: {path}")

    resolved_column = class_column
    if resolved_column not in frame.columns:
        resolved_column = next((col for col in DEFAULT_CLASS_COLUMNS if col in frame.columns), "")
    if not resolved_column:
        raise ValueError(
            f"No category column found in {path}. Requested={class_column}; "
            f"available={list(frame.columns)}"
        )

    use_paper_labels = resolved_column.startswith("Paper_Radar_Class")
    english_column = "Paper_Radar_Class_EN" if use_paper_labels else "Radar_Class_EN"
    chinese_column = "Paper_Radar_Class_ZH" if use_paper_labels else "Radar_Class_ZH"

    mapping: Dict[str, Dict[str, object]] = {}
    for _, row in frame.iterrows():
        key = _normalise_scalar_id(row["Disease_ID"])
        if not key:
            continue
        category = row.get(resolved_column)
        if pd.isna(category) or not str(category).strip():
            category = "Others"
        mapping[key] = {
            "category": str(category).strip(),
            "category_en": str(row.get(english_column, category)).strip(),
            "category_zh": str(row.get(chinese_column, category)).strip(),
            "chapter_code": _normalise_scalar_id(row.get("ICD11_Chapter_Code", "")),
            "chapter_en": str(row.get("ICD11_Chapter_Name_EN", "")).strip(),
            "chapter_zh": str(row.get("ICD11_Chapter_Name_ZH", "")).strip(),
            "disease_concept": str(row.get("Probable_Disease_Concept", "")).strip(),
            "mapping_confidence": str(row.get("Confidence", "")).strip(),
        }

    if not mapping:
        raise ValueError(f"Mapping file contains no usable Disease_ID rows: {path}")
    return mapping, resolved_column


def _original_disease_id(
    local_or_offset_disease_id: object,
    disease_offset: int,
    disease_idx_to_id: Mapping[int, object],
    is_offset: bool = True,
) -> str:
    value = int(local_or_offset_disease_id)
    local_idx = value - int(disease_offset) if is_offset else value
    raw = disease_idx_to_id.get(local_idx, local_idx)
    return _normalise_scalar_id(raw)


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    result = {
        "auc": np.nan,
        "aupr": np.nan,
        "f1_score": np.nan,
        "binary_metric_valid": False,
    }
    if y_true.size == 0:
        return result

    unique = np.unique(y_true)
    if unique.size >= 2:
        result["auc"] = float(roc_auc_score(y_true, y_score))
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        result["aupr"] = float(auc(recall, precision))
        result["binary_metric_valid"] = True

    y_hat = (y_score >= float(threshold)).astype(np.int64)
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    result["f1_score"] = float((2 * tp) / denom) if denom > 0 else 0.0
    return result


def compute_category_metrics(
    *,
    val_data: np.ndarray,
    y_pred: Sequence[float],
    val_pos: np.ndarray,
    ranks: Sequence[int],
    query_disease_ids: Optional[Sequence[int]],
    threshold: float,
    disease_offset: int,
    disease_idx_to_id: Mapping[int, object],
    mapping_by_id: Mapping[str, Mapping[str, object]],
    recall_k_list: Iterable[int] = DEFAULT_RECALL_K,
    unknown_category: str = "Others",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute category metrics using the true disease of each query.

    ``query_disease_ids`` must be aligned with ``val_data`` and contain local disease
    indices before graph offsets.  When it is omitted, positive/candidate disease IDs
    are used only as a backward-compatible fallback and a warning column is written.
    """
    val_data = np.asarray(val_data)
    val_pos = np.asarray(val_pos)
    y_pred_array = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(val_data) != len(y_pred_array):
        raise ValueError(f"val_data/y_pred length mismatch: {len(val_data)} vs {len(y_pred_array)}")

    query_grouping = "true_query_disease"
    if query_disease_ids is None:
        query_grouping = "candidate_disease_fallback"
        query_ids_array = val_data[:, 2].astype(np.int64) - int(disease_offset)
    else:
        query_ids_array = np.asarray(query_disease_ids, dtype=np.int64).reshape(-1)
        if len(query_ids_array) != len(val_data):
            raise ValueError(
                "query_disease_ids/val_data length mismatch: "
                f"{len(query_ids_array)} vs {len(val_data)}"
            )

    binary_rows: List[Dict[str, object]] = []
    for row_index, (row, score, query_local_id) in enumerate(
        zip(val_data, y_pred_array, query_ids_array)
    ):
        true_disease_id = _original_disease_id(
            query_local_id,
            disease_offset,
            disease_idx_to_id,
            is_offset=False,
        )
        candidate_disease_id = _original_disease_id(
            row[2],
            disease_offset,
            disease_idx_to_id,
            is_offset=True,
        )
        info = mapping_by_id.get(true_disease_id, {})
        binary_rows.append(
            {
                "row_index": row_index,
                "True_Disease_ID": true_disease_id,
                "Candidate_Disease_ID": candidate_disease_id,
                "category": str(info.get("category", unknown_category)),
                "category_en": str(info.get("category_en", unknown_category)),
                "category_zh": str(info.get("category_zh", unknown_category)),
                "chapter_code": str(info.get("chapter_code", "")),
                "chapter_en": str(info.get("chapter_en", "")),
                "chapter_zh": str(info.get("chapter_zh", "")),
                "y_true": int(row[3]),
                "y_score": float(score),
                "query_grouping": query_grouping,
            }
        )
    binary_df = pd.DataFrame(binary_rows)

    rank_count = min(len(val_pos), len(ranks))
    ranking_rows: List[Dict[str, object]] = []
    for query_index, (row, rank) in enumerate(zip(val_pos[:rank_count], list(ranks)[:rank_count])):
        disease_id = _original_disease_id(
            row[2], disease_offset, disease_idx_to_id, is_offset=True
        )
        info = mapping_by_id.get(disease_id, {})
        ranking_rows.append(
            {
                "query_index": query_index,
                "True_Disease_ID": disease_id,
                "category": str(info.get("category", unknown_category)),
                "category_en": str(info.get("category_en", unknown_category)),
                "category_zh": str(info.get("category_zh", unknown_category)),
                "chapter_code": str(info.get("chapter_code", "")),
                "chapter_en": str(info.get("chapter_en", "")),
                "chapter_zh": str(info.get("chapter_zh", "")),
                "rank": int(rank),
            }
        )
    ranking_df = pd.DataFrame(ranking_rows)

    categories = sorted(
        set(binary_df.get("category", pd.Series(dtype=str)).dropna().astype(str))
        | set(ranking_df.get("category", pd.Series(dtype=str)).dropna().astype(str))
    )

    rows: List[Dict[str, object]] = []
    k_values = sorted({int(k) for k in recall_k_list if int(k) > 0})
    for category in categories:
        binary_group = binary_df[binary_df["category"] == category]
        ranking_group = ranking_df[ranking_df["category"] == category]
        first = binary_group.iloc[0] if not binary_group.empty else ranking_group.iloc[0]

        binary_result = _binary_metrics(
            binary_group["y_true"].to_numpy(dtype=np.int64),
            binary_group["y_score"].to_numpy(dtype=np.float64),
            threshold,
        )
        rank_values = ranking_group["rank"].to_numpy(dtype=np.int64)

        result: Dict[str, object] = {
            "category": category,
            "category_en": first.get("category_en", category),
            "category_zh": first.get("category_zh", category),
            "chapter_codes": ";".join(
                sorted(
                    {
                        x
                        for x in binary_group.get("chapter_code", pd.Series(dtype=str)).astype(str)
                        if x
                    }
                )
            ),
            "auc": binary_result["auc"],
            "aupr": binary_result["aupr"],
            "f1_score": binary_result["f1_score"],
            "binary_metric_valid": bool(binary_result["binary_metric_valid"]),
            "mrr": float(np.mean(1.0 / rank_values)) if rank_values.size else np.nan,
            "binary_sample_count": int(len(binary_group)),
            "binary_positive_count": int(binary_group["y_true"].sum()) if not binary_group.empty else 0,
            "binary_negative_count": int((binary_group["y_true"] == 0).sum()) if not binary_group.empty else 0,
            "positive_query_count": int(len(ranking_group)),
            "query_grouping": query_grouping,
        }
        for k in k_values:
            value = float(np.mean(rank_values <= k)) if rank_values.size else np.nan
            result[f"hit@{k}"] = value
            result[f"recall@{k}"] = value
        rows.append(result)

    return pd.DataFrame(rows), binary_df, ranking_df


def save_category_metrics_for_fold(
    *,
    val_data: np.ndarray,
    y_pred: Sequence[float],
    val_pos: np.ndarray,
    ranks: Sequence[int],
    query_disease_ids: Optional[Sequence[int]],
    threshold: float,
    disease_offset: int,
    disease_idx_to_id: Mapping[int, object],
    mapping_path: str | os.PathLike[str],
    class_column: str,
    results_dir: str | os.PathLike[str],
    method: str,
    protocol: str,
    mode: str,
    ablation: str,
    fold: int,
    run_index: int = 0,
    random_state: Optional[int] = None,
    save_raw_records: bool = True,
) -> Optional[Path]:
    """Compute and save one category-metric CSV for a fold/run."""
    try:
        mapping_by_id, resolved_class_column = load_disease_category_mapping(
            mapping_path=mapping_path,
            class_column=class_column,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARN] ICD-11 category metrics skipped: {exc}")
        return None

    metrics_df, binary_df, ranking_df = compute_category_metrics(
        val_data=val_data,
        y_pred=y_pred,
        val_pos=val_pos,
        ranks=ranks,
        query_disease_ids=query_disease_ids,
        threshold=threshold,
        disease_offset=disease_offset,
        disease_idx_to_id=disease_idx_to_id,
        mapping_by_id=mapping_by_id,
    )
    if metrics_df.empty:
        print("[WARN] ICD-11 category metrics skipped: no category rows were produced")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    method_name = str(method or "unknown").strip() or "unknown"
    output_dir = Path(results_dir) / "category_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "method": method_name,
        "protocol": str(protocol),
        "mode": str(mode),
        "ablation": str(ablation),
        "fold": int(fold),
        "run_index": int(run_index),
        "random_state": "" if random_state is None else int(random_state),
        "threshold": float(threshold),
        "mapping_file": str(mapping_path),
        "class_column": resolved_class_column,
        "created_at": timestamp,
    }
    for key, value in reversed(list(metadata.items())):
        metrics_df.insert(0, key, value)

    filename = (
        f"{_safe_filename(method_name)}_{_safe_filename(protocol)}_"
        f"{_safe_filename(mode)}_{_safe_filename(ablation)}_"
        f"run{int(run_index)}_fold{int(fold)}_{timestamp}_pid{os.getpid()}.csv"
    )
    output_path = output_dir / filename
    metrics_df.to_csv(output_path, index=False)
    print(f"ICD-11 category metrics saved: {output_path}")

    if save_raw_records:
        raw_dir = Path(results_dir) / "category_metric_records"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem
        for frame, suffix in ((binary_df, "binary"), (ranking_df, "ranking")):
            frame = frame.copy()
            for key, value in reversed(list(metadata.items())):
                frame.insert(0, key, value)
            frame.to_csv(raw_dir / f"{stem}_{suffix}.csv", index=False)

    return output_path
