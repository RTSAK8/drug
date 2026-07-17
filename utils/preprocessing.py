"""
数据预处理工具函数。

除原始负采样外，本文件新增了带 query disease 追踪的负采样函数。
在 missing-disease 评估中，负候选疾病必须归属于其来源正查询的真实疾病类别，
否则按候选疾病类别分组会导致某些类别只有正样本或只有负样本，进而使
AUROC/AUPRC 无法计算。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def apply_offsets(data: np.ndarray, gene_offset: int, disease_offset: int) -> np.ndarray:
    """对药物-基因-疾病样本应用全图节点偏移。"""
    if data.size == 0:
        return data

    data = data.copy()
    data[:, 1] += gene_offset
    data[:, 2] += disease_offset
    return data


def _candidate_pool(pool: List[int], original: int) -> List[int]:
    """返回排除原实体后的候选池。"""
    return [int(item) for item in pool if int(item) != int(original)]


def generate_negatives_for_fold(
    pos_samples: np.ndarray,
    drug_pool: List[int],
    gene_pool: List[int],
    disease_pool: List[int],
    global_exclude: set,
    replace_dims: List[str],
    neg_ratio: int = 3,
) -> np.ndarray:
    """为当前 fold 生成普通负样本。

    返回形状为 ``(n, 4)`` 的数组，列为 Drug_ID、Gene_ID、Disease_ID、Label。
    """
    negatives = []
    local_exclude = set(tuple(int(x) for x in row[:3]) for row in pos_samples)

    for sample in pos_samples:
        d, g, c = (int(sample[0]), int(sample[1]), int(sample[2]))

        if "drug" in replace_dims and len(drug_pool) > 1:
            candidates = _candidate_pool(drug_pool, d)
            for _ in range(int(neg_ratio)):
                if not candidates:
                    break
                new_d = int(np.random.choice(candidates))
                triplet = (new_d, g, c)
                if triplet not in local_exclude and triplet not in global_exclude:
                    negatives.append([new_d, g, c, 0])
                    local_exclude.add(triplet)

        if "gene" in replace_dims and len(gene_pool) > 1:
            candidates = _candidate_pool(gene_pool, g)
            for _ in range(int(neg_ratio)):
                if not candidates:
                    break
                new_g = int(np.random.choice(candidates))
                triplet = (d, new_g, c)
                if triplet not in local_exclude and triplet not in global_exclude:
                    negatives.append([d, new_g, c, 0])
                    local_exclude.add(triplet)

        if "disease" in replace_dims and len(disease_pool) > 1:
            candidates = _candidate_pool(disease_pool, c)
            for _ in range(int(neg_ratio)):
                if not candidates:
                    break
                new_c = int(np.random.choice(candidates))
                triplet = (d, g, new_c)
                if triplet not in local_exclude and triplet not in global_exclude:
                    negatives.append([d, g, new_c, 0])
                    local_exclude.add(triplet)

    global_exclude.update(local_exclude)
    return np.asarray(negatives, dtype=np.int64) if negatives else np.empty((0, 4), dtype=np.int64)


def generate_negatives_with_query_disease(
    pos_samples: np.ndarray,
    drug_pool: List[int],
    gene_pool: List[int],
    disease_pool: List[int],
    global_exclude: set,
    replace_dims: List[str],
    neg_ratio: int = 3,
    max_attempts: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成负样本并记录每个负样本来源查询的真实疾病。

    Returns
    -------
    negatives:
        形状 ``(n, 4)``，列为 Drug_ID、Gene_ID、候选 Disease_ID、Label。
    query_disease_ids:
        形状 ``(n,)``。每一项是生成该负样本的原正样本 Disease_ID。

    Notes
    -----
    对 missing-disease 任务，负样本的候选疾病可能属于另一个 ICD-11 类别，
    但该候选仍是在回答原正样本的查询。因此类别统计必须使用这里保存的
    ``query_disease_ids``，而不能使用负样本自身的候选 Disease_ID。
    """
    negatives = []
    query_disease_ids = []
    local_exclude = set(tuple(int(x) for x in row[:3]) for row in pos_samples)
    max_attempts = max(1, int(max_attempts))

    for sample in pos_samples:
        d, g, true_c = (int(sample[0]), int(sample[1]), int(sample[2]))

        replacement_specs = []
        if "drug" in replace_dims and len(drug_pool) > 1:
            replacement_specs.append(("drug", _candidate_pool(drug_pool, d)))
        if "gene" in replace_dims and len(gene_pool) > 1:
            replacement_specs.append(("gene", _candidate_pool(gene_pool, g)))
        if "disease" in replace_dims and len(disease_pool) > 1:
            replacement_specs.append(("disease", _candidate_pool(disease_pool, true_c)))

        for dimension, candidates in replacement_specs:
            generated = 0
            attempts = 0
            while generated < int(neg_ratio) and attempts < max_attempts:
                attempts += 1
                if not candidates:
                    break
                replacement = int(np.random.choice(candidates))

                if dimension == "drug":
                    triplet = (replacement, g, true_c)
                elif dimension == "gene":
                    triplet = (d, replacement, true_c)
                else:
                    triplet = (d, g, replacement)

                if triplet in local_exclude or triplet in global_exclude:
                    continue

                negatives.append([triplet[0], triplet[1], triplet[2], 0])
                query_disease_ids.append(true_c)
                local_exclude.add(triplet)
                generated += 1

    global_exclude.update(local_exclude)

    if not negatives:
        return np.empty((0, 4), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return (
        np.asarray(negatives, dtype=np.int64),
        np.asarray(query_disease_ids, dtype=np.int64),
    )
