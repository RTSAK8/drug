"""
训练工具函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================== 超图构建函数 ===========================

def build_hypergraph(data: np.ndarray, ppi_data: Optional[np.ndarray] = None,
                    dpi_data: Optional[np.ndarray] = None, ddi_data: Optional[np.ndarray] = None,
                    gene_hyperedges: Optional[List[List[int]]] = None,
                    protein_disease_data: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    构建超图结构
    
    Args:
        data: 三元组数据，形状 (n_triplets, 3) 
        ppi_data: PPI边，形状 (2, n_ppi_edges)
        dpi_data: 药物-蛋白质边，形状 (2, n_dpi_edges)
        ddi_data: 药物-药物边，形状 (2, n_ddi_edges)
        gene_hyperedges: 基因功能超边列表
        
    Returns:
        超图边索引，形状 (2, n_edges)
    """
    if data.size == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    # 处理三元组为三阶超边
    pairs = data[:, 0:3].reshape(1, -1)  # 形状: (1, 3*n_triplets)
    
    # 为每个三元组创建唯一的超边编号
    # 每个三元组的3个节点共享同一个超边编号
    n_triplets = len(data)
    hyper_edge_num = np.repeat(np.arange(n_triplets), 3).reshape(1, -1)
    
    # 当前超边偏移量
    current_offset = n_triplets
    
    # 边类型处理
    edge_types = [
        ('ppi', ppi_data, 2),  # 二元边，每个边2个节点
        ('dpi', dpi_data, 2),
        ('ddi', ddi_data, 2),
        ('protein_disease', protein_disease_data, 2)  
    ]
    
    # 添加各种二元边
    for edge_name, edge_data, nodes_per_edge in edge_types:
        if edge_data is not None and edge_data.shape[1] > 0:
            n_edges = edge_data.shape[1]  # 边的数量
            
            # 展开边对
            edge_pairs = edge_data.reshape(1, -1)  # 形状 (1, 2*n_edges)
            
            # 为每个边的2个节点创建超边编号
            edge_pairs_num = np.repeat(np.arange(n_edges) + current_offset, nodes_per_edge).reshape(1, -1)
            
            # 拼接
            pairs = np.concatenate((pairs, edge_pairs), axis=1)
            hyper_edge_num = np.concatenate((hyper_edge_num, edge_pairs_num), axis=1)
            
            # 更新偏移
            current_offset += n_edges
    
    # 添加基因功能超边
    if gene_hyperedges and len(gene_hyperedges) > 0:
        gene_hyperedge_pairs = []
        gene_hyperedge_nums = []
        
        for hyperedge_idx, hyperedge in enumerate(gene_hyperedges):
            hyperedge_id = current_offset + hyperedge_idx
            for gene_id in hyperedge:
                gene_hyperedge_pairs.append(gene_id)
                gene_hyperedge_nums.append(hyperedge_id)
        
        if gene_hyperedge_pairs:
            gene_hyperedge_pairs = np.array(gene_hyperedge_pairs).reshape(1, -1)
            gene_hyperedge_nums = np.array(gene_hyperedge_nums).reshape(1, -1)
            
            pairs = np.concatenate((pairs, gene_hyperedge_pairs), axis=1)
            hyper_edge_num = np.concatenate((hyper_edge_num, gene_hyperedge_nums), axis=1)
    
    # 构建超图矩阵
    hyper_graph = np.concatenate((pairs, hyper_edge_num), axis=0)
    return torch.from_numpy(hyper_graph).long().to(device)

# =========================== 评估指标函数 ===========================

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
        return {f'hit@{k}': 0.0 for k in k_list}
    
    ranks_array = np.array(ranks)
    hits = {}
    
    for k in k_list:
        hits_at_k = np.sum(ranks_array <= k) / len(ranks_array)
        hits[f'hit@{k}'] = float(hits_at_k)
    
    return hits


def metrics_graph_with_hits(y_true: np.ndarray, y_pred: np.ndarray, 
                           k_list: List[int] = [1, 3, 5]) -> Tuple[float, float, float, Dict[str, float]]:
    """
    计算二分类指标和Hit@K指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测概率
        k_list: Hit@K的K值列表
        
    Returns:
        auc, aupr, f1_score, hits字典
    """
    # 计算AUC
    auc_score = roc_auc_score(y_true, y_pred)
    
    # 计算AUPR和最佳阈值下的F1 score
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr_score = auc(recall, precision)
    
    # 计算最佳阈值下的F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = f1_scores[:-1]  # 移除最后一个值（对应无预测的情况）
    
    if len(f1_scores) > 0:
        best_threshold_idx = np.argmax(f1_scores)
        best_f1_score = f1_scores[best_threshold_idx]
        best_threshold = thresholds[best_threshold_idx]
    else:
        best_f1_score = 0.0
        best_threshold = 0.5
    best_threshold = float(best_threshold)
    # 计算Hit@K 待review
    hits = calculate_hits_binary_classification(y_true, y_pred, k_list)
    
    return auc_score, aupr_score, best_f1_score, best_threshold, hits
def calculate_hits_binary_classification(y_true: np.ndarray, y_pred: np.ndarray,
                                        k_list: List[int] = [1, 3, 5]) -> Dict[str, float]:
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
        return {f'hit@{k}': 0.0 for k in k_list}
    
    hits = {}
    for k in k_list:
        # 取前k个预测结果
        top_k_labels = [pair[1] for pair in pred_true_pairs[:k]]
        # 计算前k个中是否至少有一个正样本
        hits_at_k = 1.0 if sum(top_k_labels) > 0 else 0.0
        hits[f'hit@{k}'] = hits_at_k
    
    return hits

# =========================== 训练和验证函数 ===========================

def train_step(model: nn.Module, optimizer: torch.optim.Optimizer,
              loss_func: nn.Module, drug_emb: torch.Tensor,
              gene_emb: torch.Tensor, disease_emb: torch.Tensor,
              hp_pos: torch.Tensor, hp_neg: torch.Tensor,
              train_data: np.ndarray, train_pos: np.ndarray,
              epoch: int, alpha: float = 0.4) -> Dict[str, float]:
    """
    单步训练函数
    
    Returns:
        训练指标字典
    """
    model.train()
    optimizer.zero_grad()
    
    # 准备输入
    train_data_tensor = torch.from_numpy(train_data).long().to(device)
    train_pos_tensor = torch.from_numpy(train_pos).long().to(device)
    
    # 前向传播
    pred, cl_loss, mrr, ranks = model(
        drug_emb, gene_emb, disease_emb,
        hp_pos, hp_neg,
        train_data_tensor[:, 0],  # drug_id
        train_data_tensor[:, 1],  # gene_id
        train_data_tensor[:, 2],  # disease_id
        train_pos_tensor[:, 0],   # pos_drugid
        train_pos_tensor[:, 1],   # pos_geneid
        train_pos_tensor[:, 2]    # pos_diseaseid
    )
    
    # 计算损失
    target = torch.from_numpy(train_data[:, 3]).float().to(device)
    main_loss = loss_func(pred, target)
    total_loss = (1 - alpha) * main_loss + alpha * cl_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    # 计算Hit@K
    hits = calculate_hits_ranking_based(ranks, [1, 3, 5])
    
    # 打印训练信息
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1:03d}: Loss={total_loss.item():.4f}, '
              f'Main={main_loss.item():.4f}, CL={cl_loss.item():.4f}, '
              f'MRR={mrr:.4f}, Hit@1={hits["hit@1"]:.4f}, '
              f'Hit@3={hits["hit@3"]:.4f}, Hit@5={hits["hit@5"]:.4f}')
    
    return {
        'loss': float(total_loss.item()),
        'main_loss': float(main_loss.item()),
        'cl_loss': float(cl_loss.item()),
        'mrr': float(mrr),
        'hit@1': float(hits['hit@1']),
        'hit@3': float(hits['hit@3']),
        'hit@5': float(hits['hit@5']),
    }

def validate_step(model: nn.Module, drug_emb: torch.Tensor,
                 gene_emb: torch.Tensor, disease_emb: torch.Tensor,
                 hp_pos: torch.Tensor, hp_neg: torch.Tensor,
                 val_data: np.ndarray, val_pos: np.ndarray) -> Tuple[float, float, float, Dict]:
    """
    验证步骤
    
    Returns:
        auc, aupr, mrr, hits字典
    """
    model.eval()
    with torch.no_grad():
        val_data_tensor = torch.from_numpy(val_data).long().to(device)
        val_pos_tensor = torch.from_numpy(val_pos).long().to(device)
        
        # 前向传播
        pred, cl_loss, mrr, ranks = model(
            drug_emb, gene_emb, disease_emb,
            hp_pos, hp_neg,
            val_data_tensor[:, 0],
            val_data_tensor[:, 1],
            val_data_tensor[:, 2],
            val_pos_tensor[:, 0],
            val_pos_tensor[:, 1],
            val_pos_tensor[:, 2]
        )
        
        # 计算指标
        y_true = val_data[:, 3]
        y_pred = pred.cpu().numpy()
        
        auc_score, aupr_score, f1_score, threshold, hits = metrics_graph_with_hits(y_true, y_pred)
        
        return auc_score, aupr_score, f1_score, threshold, float(mrr), hits

# =========================== 数据预处理函数 ===========================

def apply_offsets(data: np.ndarray, gene_offset: int, disease_offset: int) -> np.ndarray:
    """
    对数据应用偏移量
    
    Args:
        data: 原始数据，形状 (n_samples, 4)
        gene_offset: 基因偏移量
        disease_offset: 疾病偏移量
        
    Returns:
        偏移后的数据
    """
    if data.size == 0:
        return data
    
    data = data.copy()
    data[:, 1] += gene_offset   # 基因ID + 偏移
    data[:, 2] += disease_offset  # 疾病ID + 偏移
    return data

def generate_negatives_for_fold(pos_samples: np.ndarray, drug_pool: List[int],
                               gene_pool: List[int], disease_pool: List[int],
                               global_exclude: set, replace_dims: List[str],
                               neg_ratio: int = 3) -> np.ndarray:
    """
    为当前fold生成负样本
    
    Args:
        pos_samples: 正样本
        drug_pool: 药物候选池
        gene_pool: 基因候选池
        disease_pool: 疾病候选池
        global_exclude: 全局排除集合
        replace_dims: 替换维度
        neg_ratio: 负样本比例
        
    Returns:
        负样本数组
    """
    negatives = []
    local_exclude = set(tuple(row[:3]) for row in pos_samples)
    
    for sample in pos_samples:
        d, g, c = sample[:3]
        
        # 生成药物维度负样本
        if "drug" in replace_dims and len(drug_pool) > 1:
            for _ in range(neg_ratio):
                new_d = np.random.choice([x for x in drug_pool if x != d])
                triplet = (new_d, g, c)
                if triplet not in local_exclude and triplet not in global_exclude:
                    negatives.append([new_d, g, c, 0])
                    local_exclude.add(triplet)
        
        # 生成基因维度负样本
        if "gene" in replace_dims and len(gene_pool) > 1:
            for _ in range(neg_ratio):
                new_g = np.random.choice([x for x in gene_pool if x != g])
                triplet = (d, new_g, c)
                if triplet not in local_exclude and triplet not in global_exclude:
                    negatives.append([d, new_g, c, 0])
                    local_exclude.add(triplet)
        
        # 生成疾病维度负样本
        if "disease" in replace_dims and len(disease_pool) > 1:
            for _ in range(neg_ratio):
                new_c = np.random.choice([x for x in disease_pool if x != c])
                triplet = (d, g, new_c)
                if triplet not in local_exclude and triplet not in global_exclude:
                    negatives.append([d, g, new_c, 0])
                    local_exclude.add(triplet)
    
    global_exclude.update(local_exclude)
    return np.array(negatives) if negatives else np.empty((0, 4))

# =========================== 结果保存函数 ===========================

import pandas as pd
import os
import json
from datetime import datetime

def save_results(mode: str, ablation_name: str, fold: int, metrics: Dict,
                 config: dict, results_dir: str = "results"):
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
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存配置
    config_file = os.path.join(exp_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 保存到汇总文件
    summary_file = os.path.join(results_dir, "hypergraph_summary.csv")
    result_row = {
        'timestamp': timestamp,
        'mode': mode,
        'ablation': ablation_name,
        'fold': fold,
        'auc': metrics.get('auc', 0),
        'aupr': metrics.get('aupr', 0),
        'f1_score': metrics.get('f1_score', 0),
        'threshold': metrics.get('threshold', 0),
    }
    
    df = pd.DataFrame([result_row])
    if os.path.exists(summary_file):
        df.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        df.to_csv(summary_file, index=False)
    
    print(f"结果已保存到: {exp_dir}")

def save_predictions(val_data: np.ndarray, pred_scores: np.ndarray,
                    mode: str, ablation_name: str, fold: int,
                    results_dir: str = "results"):
    """
    保存预测结果
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(results_dir, f"{mode}_{ablation_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建预测结果DataFrame
    predictions_df = pd.DataFrame({
        'drug_id': val_data[:, 0],
        'gene_id': val_data[:, 1],
        'disease_id': val_data[:, 2],
        'true_label': val_data[:, 3],
        'pred_score': pred_scores
    })
    
    # 保存预测结果
    pred_file = os.path.join(exp_dir, f"fold_{fold}_predictions.csv")
    predictions_df.to_csv(pred_file, index=False)
    
    return pred_file

def save_summary_results(mode: str, ablation_name: str, 
                        avg_metrics: Dict, all_fold_metrics: List[Dict],
                        config: dict, results_dir: str = "results"):
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
        'timestamp': timestamp,
        'mode': mode,
        'ablation': ablation_name,
        'n_folds': len(all_fold_metrics),
        
        # 平均值
        'auc_mean': avg_metrics.get('auc_mean', 0),
        'aupr_mean': avg_metrics.get('aupr_mean', 0),
        'f1_score_mean': avg_metrics.get('f1_score_mean', 0),
        'mrr_mean': avg_metrics.get('mrr_mean', 0),
        'hit@1_mean': avg_metrics.get('hit@1_mean', 0),
        'hit@3_mean': avg_metrics.get('hit@3_mean', 0),
        'hit@5_mean': avg_metrics.get('hit@5_mean', 0),

        # 标准差
        'auc_std': avg_metrics.get('auc_std', 0),
        'aupr_std': avg_metrics.get('aupr_std', 0),
        'f1_score_std': avg_metrics.get('f1_score_std', 0),
        'mrr_std': avg_metrics.get('mrr_std', 0),
        'hit@1_std': avg_metrics.get('hit@1_std', 0),
        'hit@3_std': avg_metrics.get('hit@3_std', 0),
        'hit@5_std': avg_metrics.get('hit@5_std', 0),
    }
    
    # 可选：添加每个fold的原始值（展开为列）
    for fold_idx, fold_metrics in enumerate(all_fold_metrics):
        for metric_name, value in fold_metrics.items():
            summary_row[f'fold{fold_idx+1}_{metric_name}'] = value
    
    df = pd.DataFrame([summary_row])
    
    # 保存到CSV（追加模式）
    summary_file = os.path.join(results_dir, "hypergraph_summary_new.csv")
    os.makedirs(results_dir, exist_ok=True)
    
    if os.path.exists(summary_file):
        df.to_csv(summary_file, mode='a', header=False, index=False)
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