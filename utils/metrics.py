"""
评估指标工具函数
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


DEFAULT_HIT_K_LIST = [1, 3, 5]
DEFAULT_RECALL_K_LIST = [1, 5, 10, 50, 100]


def _rank_at_k(ranks: List[int], k_list: List[int], prefix: str) -> Dict[str, float]:
    """根据正样本 rank 计算 @K 指标。rank 从 1 开始，越小越好。"""
    if len(ranks) == 0:
        return {f"{prefix}@{k}": 0.0 for k in k_list}

    ranks_array = np.asarray(ranks, dtype=np.int64)
    metrics = {}
    for k in k_list:
        metrics[f"{prefix}@{k}"] = float(np.mean(ranks_array <= int(k)))
    return metrics


def calculate_hits_ranking_based(ranks: List[int], k_list: List[int] = None) -> Dict[str, float]:
    """
    基于全实体排序 rank 计算 Hit@K。

    Args:
        ranks: 每个正样本真实答案在候选实体全集中的排名列表，排名从 1 开始。
        k_list: 要计算的 K 值列表。

    Returns:
        包含各个 Hit@K 值的字典。
    """
    if k_list is None:
        k_list = DEFAULT_HIT_K_LIST
    return _rank_at_k(ranks, k_list, prefix="hit")


def calculate_recall_ranking_based(ranks: List[int], k_list: List[int] = None) -> Dict[str, float]:
    """
    基于全实体排序 rank 计算 Recall@K。

    当前 missing-gene / missing-disease 评估中，每个查询对应一个标注正答案；因此该实现等价于
    mean(真实答案 rank <= K)，但单独输出 recall@K 便于论文表格和日志统计。
    """
    if k_list is None:
        k_list = DEFAULT_RECALL_K_LIST
    return _rank_at_k(ranks, k_list, prefix="recall")


def calculate_ranking_metrics(
    ranks: List[int],
    hit_k_list: List[int] = None,
    recall_k_list: List[int] = None,
) -> Dict[str, float]:
    """一次性返回 Hit@K 和 Recall@K。"""
    metrics = {}
    metrics.update(calculate_hits_ranking_based(ranks, hit_k_list or DEFAULT_HIT_K_LIST))
    metrics.update(calculate_recall_ranking_based(ranks, recall_k_list or DEFAULT_RECALL_K_LIST))
    return metrics


def compute_f1_from_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """在给定阈值下计算 F1，避免在测试集上二次寻优阈值。"""
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred)
    y_hat = (y_pred >= float(threshold)).astype(np.int32)

    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))

    return float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-10))


def metrics_graph_with_hits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_list: List[int] = None,
    fixed_threshold: Optional[float] = None,
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """
    计算二分类指标和二分类 top-k 命中指标。

    注意：论文中的 MRR / Hit@K / Recall@K 应优先使用全实体 ranking rank 计算，
    即 calculate_hits_ranking_based / calculate_recall_ranking_based。
    此函数中的 hits 仅保留为二分类预测列表的兼容输出。
    """
    if k_list is None:
        k_list = DEFAULT_HIT_K_LIST

    # 计算 AUC
    auc_score = roc_auc_score(y_true, y_pred)

    # 计算 AUPR 和最佳阈值下的 F1 score
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr_score = auc(recall, precision)

    if fixed_threshold is None:
        # 计算最佳阈值下的 F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores = f1_scores[:-1]  # 移除最后一个值（对应无预测的情况）

        if len(f1_scores) > 0:
            best_threshold_idx = int(np.argmax(f1_scores))
            best_f1_score = float(f1_scores[best_threshold_idx])
            best_threshold = float(thresholds[best_threshold_idx])
        else:
            best_f1_score = 0.0
            best_threshold = 0.5
    else:
        best_threshold = float(fixed_threshold)
        best_f1_score = compute_f1_from_threshold(y_true, y_pred, best_threshold)

    hits = calculate_hits_binary_classification(y_true, y_pred, k_list)

    return auc_score, aupr_score, best_f1_score, best_threshold, hits


def calculate_hits_binary_classification(y_true: np.ndarray, y_pred: np.ndarray, k_list: List[int] = None) -> Dict[str, float]:
    """
    基于二分类预测列表计算 Hit@K。

    该函数只用于兼容旧日志。ranking 任务的 Hit@K / Recall@K 应使用 rank-based 版本。
    """
    if k_list is None:
        k_list = DEFAULT_HIT_K_LIST

    pred_true_pairs = list(zip(y_pred, y_true))
    pred_true_pairs.sort(key=lambda x: x[0], reverse=True)

    total_positives = np.sum(y_true)

    if total_positives == 0:
        return {f"hit@{k}": 0.0 for k in k_list}

    hits = {}
    for k in k_list:
        top_k_labels = [pair[1] for pair in pred_true_pairs[: int(k)]]
        hits[f"hit@{k}"] = 1.0 if sum(top_k_labels) > 0 else 0.0

    return hits
