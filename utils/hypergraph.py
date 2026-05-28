"""
超图构建工具函数
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from .runtime import get_device

device = get_device()


def build_hypergraph(
    data: np.ndarray,
    ppi_data: Optional[np.ndarray] = None,
    dpi_data: Optional[np.ndarray] = None,
    ddi_data: Optional[np.ndarray] = None,
    gene_hyperedges: Optional[List[List[int]]] = None,
    protein_disease_data: Optional[np.ndarray] = None,
) -> torch.Tensor:
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

    # 当前超边偏移量（用于为 gene_hyperedges 分配新的 hyperedge_id）
    current_offset = n_triplets

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


def build_binary_edges(
    ppi_data: Optional[np.ndarray] = None,
    dpi_data: Optional[np.ndarray] = None,
    ddi_data: Optional[np.ndarray] = None,
    protein_disease_data: Optional[np.ndarray] = None,
    make_undirected: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a typed ordinary edge list for binary relations.

    Returns:
        edge_index: (2, E) long tensor
        edge_type: (E,) long tensor, where type ids are:
            0=ppi, 1=dpi, 2=ddi, 3=protein_disease
    """

    def _append(edges_list: List[np.ndarray], types_list: List[np.ndarray], edge_data: Optional[np.ndarray], rel_type: int):
        if edge_data is None or edge_data.size == 0 or edge_data.shape[1] == 0:
            return
        src = edge_data[0].astype(np.int64)
        dst = edge_data[1].astype(np.int64)

        if make_undirected:
            src = np.concatenate([src, dst], axis=0)
            dst = np.concatenate([dst, edge_data[0].astype(np.int64)], axis=0)

        edges_list.append(np.vstack([src, dst]))
        types_list.append(np.full((src.shape[0],), rel_type, dtype=np.int64))

    edges_np: List[np.ndarray] = []
    types_np: List[np.ndarray] = []

    _append(edges_np, types_np, ppi_data, 0)
    _append(edges_np, types_np, dpi_data, 1)
    _append(edges_np, types_np, ddi_data, 2)
    _append(edges_np, types_np, protein_disease_data, 3)

    if not edges_np:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    edge_index = np.concatenate(edges_np, axis=1)
    edge_type = np.concatenate(types_np, axis=0)

    return (
        torch.from_numpy(edge_index).long().to(device),
        torch.from_numpy(edge_type).long().to(device),
    )
