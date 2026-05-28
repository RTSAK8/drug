"""
训练与验证工具函数
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

from .metrics import calculate_hits_ranking_based, metrics_graph_with_hits
from .runtime import get_device

device = get_device()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: nn.Module,
    drug_emb: torch.Tensor,
    gene_emb: torch.Tensor,
    disease_emb: torch.Tensor,
    hp_pos: torch.Tensor,
    hp_neg: torch.Tensor,
    bin_edge_index: torch.Tensor,
    bin_edge_type: torch.Tensor,
    train_data: np.ndarray,
    train_pos: np.ndarray,
    epoch: int,
    epoch_progress: float = 0.0,
    alpha: float = 0.4,
) -> Dict[str, float]:
    """
    单步训练函数

    Returns:
        训练阶段指标字典
    """
    model.train()
    optimizer.zero_grad()

    # 准备输入
    train_data_tensor = torch.from_numpy(train_data).long().to(device)
    train_pos_tensor = torch.from_numpy(train_pos).long().to(device)

    # 前向传播
    pred, cl_loss, mrr, ranks = model(
        drug_emb,
        gene_emb,
        disease_emb,
        hp_pos,
        hp_neg,
        train_data_tensor[:, 0],  # drug_id
        train_data_tensor[:, 1],  # gene_id
        train_data_tensor[:, 2],  # disease_id
        train_pos_tensor[:, 0],  # pos_drugid
        train_pos_tensor[:, 1],  # pos_geneid
        train_pos_tensor[:, 2],  # pos_diseaseid
        bin_edge_index,
        bin_edge_type,
        epoch_progress,
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
        print(f"Epoch {epoch+1:03d}: Loss={total_loss.item():.4f}, " f"Main={main_loss.item():.4f}, CL={cl_loss.item():.4f}, " f'MRR={mrr:.4f}, Hit@1={hits["hit@1"]:.4f}, ' f'Hit@3={hits["hit@3"]:.4f}, Hit@5={hits["hit@5"]:.4f}')

    return {
        "loss": float(total_loss.item()),
        "main_loss": float(main_loss.item()),
        "cl_loss": float(cl_loss.item()),
        "mrr": float(mrr),
        "hit@1": float(hits["hit@1"]),
        "hit@3": float(hits["hit@3"]),
        "hit@5": float(hits["hit@5"]),
    }


def validate_step(
    model: nn.Module,
    drug_emb: torch.Tensor,
    gene_emb: torch.Tensor,
    disease_emb: torch.Tensor,
    hp_pos: torch.Tensor,
    hp_neg: torch.Tensor,
    bin_edge_index: torch.Tensor,
    bin_edge_type: torch.Tensor,
    val_data: np.ndarray,
    val_pos: np.ndarray,
    epoch_progress: float = 1.0,
) -> Tuple[float, float, float, float, float, Dict]:
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
            drug_emb,
            gene_emb,
            disease_emb,
            hp_pos,
            hp_neg,
            val_data_tensor[:, 0],
            val_data_tensor[:, 1],
            val_data_tensor[:, 2],
            val_pos_tensor[:, 0],
            val_pos_tensor[:, 1],
            val_pos_tensor[:, 2],
            bin_edge_index,
            bin_edge_type,
            epoch_progress,
        )

        # 计算指标
        y_true = val_data[:, 3]
        y_pred = pred.cpu().numpy()

        auc_score, aupr_score, f1_score, threshold, hits = metrics_graph_with_hits(y_true, y_pred)

        return auc_score, aupr_score, f1_score, threshold, float(mrr), hits
