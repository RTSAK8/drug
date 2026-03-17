"""
数据预处理工具函数
"""

import numpy as np
from typing import List


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
    data[:, 1] += gene_offset  # 基因ID + 偏移
    data[:, 2] += disease_offset  # 疾病ID + 偏移
    return data


def generate_negatives_for_fold(
    pos_samples: np.ndarray,
    drug_pool: List[int],
    gene_pool: List[int],
    disease_pool: List[int],
    global_exclude: set,
    replace_dims: List[str],
    neg_ratio: int = 3,
) -> np.ndarray:
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
