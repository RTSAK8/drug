"""
评估指标工具函数
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple


def calculate_hits_ranking_based(ranks: List[int], k_list: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """
    基于排名计算Hit@K指标

    Args:
        ranks: 每个正样本的排名列表
        k_list: 要计算的K值列表

    Returns:
        包含各个Hit@K值的字典
    """
    if len(ranks) == 0:
        return {f"hit@{k}": 0.0 for k in k_list}

    ranks_array = np.array(ranks)
    hits = {}

    for k in k_list:
        hits_at_k = np.sum(ranks_array <= k) / len(ranks_array)
        hits[f"hit@{k}"] = float(hits_at_k)

    return hits


def compute_f1_from_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """在给定阈值下计算F1，避免在测试集上二次寻优阈值。"""
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
    k_list: List[int] = [1, 3, 5],
    fixed_threshold: Optional[float] = None,
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """
    计算二分类指标和Hit@K指标

    Args:
        y_true: 真实标签
        y_pred: 预测概率
        k_list: Hit@K的K值列表

    Returns:
        auc, aupr, f1_score, threshold, hits字典
    """
    # 计算AUC
    auc_score = roc_auc_score(y_true, y_pred)

    # 计算AUPR和最佳阈值下的F1 score
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr_score = auc(recall, precision)

    if fixed_threshold is None:
        # 计算最佳阈值下的F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores = f1_scores[:-1]  # 移除最后一个值（对应无预测的情况）

        if len(f1_scores) > 0:
            best_threshold_idx = np.argmax(f1_scores)
            best_f1_score = float(f1_scores[best_threshold_idx])
            best_threshold = float(thresholds[best_threshold_idx])
        else:
            best_f1_score = 0.0
            best_threshold = 0.5
    else:
        best_threshold = float(fixed_threshold)
        best_f1_score = compute_f1_from_threshold(y_true, y_pred, best_threshold)

    # 计算Hit@K 待review
    hits = calculate_hits_binary_classification(y_true, y_pred, k_list)

    return auc_score, aupr_score, best_f1_score, best_threshold, hits


def calculate_hits_binary_classification(y_true: np.ndarray, y_pred: np.ndarray, k_list: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """
    基于二分类预测计算Hit@K指标

    Args:
        y_true: 真实标签
        y_pred: 预测概率
        k_list: 要计算的K值列表

    Returns:
        包含各个Hit@K值的字典
    """
    # 将预测分数和真实标签配对并按预测分数降序排序
    pred_true_pairs = list(zip(y_pred, y_true))
    pred_true_pairs.sort(key=lambda x: x[0], reverse=True)

    # 计算正样本总数
    total_positives = np.sum(y_true)

    if total_positives == 0:
        return {f"hit@{k}": 0.0 for k in k_list}

    hits = {}
    for k in k_list:
        # 取前k个预测结果
        top_k_labels = [pair[1] for pair in pred_true_pairs[:k]]
        # 计算前k个中是否至少有一个正样本
        hits_at_k = 1.0 if sum(top_k_labels) > 0 else 0.0
        hits[f"hit@{k}"] = hits_at_k

    return hits
