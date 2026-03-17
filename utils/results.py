"""
结果保存工具函数
"""

import pandas as pd
import os
import json
from datetime import datetime
import numpy as np
from typing import Dict, List


def save_results(mode: str, ablation_name: str, fold: int, metrics: Dict, config: dict, results_dir: str = "results"):
    """
    保存实验结果

    Args:
        mode: 实验模式
        ablation_name: 消融实验名称
        fold: 折叠编号
        metrics: 评估指标字典
        config: 配置字典
        results_dir: 结果保存目录
    """
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(results_dir, f"{mode}_{ablation_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 保存指标
    metrics_file = os.path.join(exp_dir, f"fold_{fold}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # 保存配置
    config_file = os.path.join(exp_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # 保存到汇总文件
    summary_file = os.path.join(results_dir, "hypergraph_summary.csv")
    result_row = {
        "timestamp": timestamp,
        "mode": mode,
        "ablation": ablation_name,
        "fold": fold,
        "auc": metrics.get("auc", 0),
        "aupr": metrics.get("aupr", 0),
        "f1_score": metrics.get("f1_score", 0),
        "threshold": metrics.get("threshold", 0),
    }

    df = pd.DataFrame([result_row])
    if os.path.exists(summary_file):
        df.to_csv(summary_file, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_file, index=False)

    print(f"结果已保存到: {exp_dir}")


def save_predictions(val_data: np.ndarray, pred_scores: np.ndarray, mode: str, ablation_name: str, fold: int, results_dir: str = "results"):
    """
    保存预测结果
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(results_dir, f"{mode}_{ablation_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 创建预测结果DataFrame
    predictions_df = pd.DataFrame({"drug_id": val_data[:, 0], "gene_id": val_data[:, 1], "disease_id": val_data[:, 2], "true_label": val_data[:, 3], "pred_score": pred_scores})

    # 保存预测结果
    pred_file = os.path.join(exp_dir, f"fold_{fold}_predictions.csv")
    predictions_df.to_csv(pred_file, index=False)

    return pred_file


def save_summary_results(mode: str, ablation_name: str, avg_metrics: Dict, all_fold_metrics: List[Dict], config: dict, results_dir: str = "results"):
    """
    保存所有fold的汇总结果（平均值±标准差）

    Args:
        mode: 实验模式
        ablation_name: 消融实验名称
        avg_metrics: 平均指标字典（包含_mean和_std后缀）
        all_fold_metrics: 所有fold的原始指标列表
        config: 配置字典
        results_dir: 结果保存目录
    """
    import pandas as pd
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建汇总DataFrame（一行数据）
    summary_row = {
        "timestamp": timestamp,
        "mode": mode,
        "ablation": ablation_name,
        "n_folds": len(all_fold_metrics),
        # 平均值
        "auc_mean": avg_metrics.get("auc_mean", 0),
        "aupr_mean": avg_metrics.get("aupr_mean", 0),
        "f1_score_mean": avg_metrics.get("f1_score_mean", 0),
        # 'mrr_mean': avg_metrics.get('mrr_mean', 0),
        # 'hit@1_mean': avg_metrics.get('hit@1_mean', 0),
        # 'hit@3_mean': avg_metrics.get('hit@3_mean', 0),
        # 'hit@5_mean': avg_metrics.get('hit@5_mean', 0),
        # # 标准差
        # 'auc_std': avg_metrics.get('auc_std', 0),
        # 'aupr_std': avg_metrics.get('aupr_std', 0),
        # 'f1_score_std': avg_metrics.get('f1_score_std', 0),
        # 'mrr_std': avg_metrics.get('mrr_std', 0),
        # 'hit@1_std': avg_metrics.get('hit@1_std', 0),
        # 'hit@3_std': avg_metrics.get('hit@3_std', 0),
        # 'hit@5_std': avg_metrics.get('hit@5_std', 0),
    }

    # 可选：添加每个fold的原始值（展开为列）
    for fold_idx, fold_metrics in enumerate(all_fold_metrics):
        for metric_name, value in fold_metrics.items():
            summary_row[f"fold{fold_idx+1}_{metric_name}"] = value

    df = pd.DataFrame([summary_row])

    # 保存到CSV（追加模式）
    summary_file = os.path.join(results_dir, "hypergraph_summary_new.csv")
    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(summary_file):
        df.to_csv(summary_file, mode="a", header=False, index=False)
        print(f"\n汇总结果已追加到: {summary_file}")
    else:
        df.to_csv(summary_file, index=False)
        print(f"\n汇总结果已保存到: {summary_file}")

    # 同时保存详细JSON（包含配置）
    # exp_dir = os.path.join(results_dir, f"{mode}_{ablation_name}_{timestamp}")
    # os.makedirs(exp_dir, exist_ok=True)

    # import json
    # with open(os.path.join(exp_dir, "summary.json"), 'w') as f:
    #     json.dump({
    #         'avg_metrics': {k: float(v) for k, v in avg_metrics.items()},
    #         'all_folds': all_fold_metrics,
    #         'config': config
    #     }, f, indent=2)

    # print(f"详细结果已保存到: {exp_dir}/summary.json")
