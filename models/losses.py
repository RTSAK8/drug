"""
损失函数模块
"""

import torch
import torch.nn.functional as F


def integrate_qccl(model, query_part1_ids, query_part2_ids, answer_ids, entity_embeddings, temperature=0.07):
    """
    查询条件对比学习损失
    """
    batch_size = len(query_part1_ids)
    if batch_size == 0:
        return torch.tensor(0.0, device=entity_embeddings.device)

    # 1. 编码查询
    query_e1_embs = entity_embeddings[query_part1_ids]
    query_e2_embs = entity_embeddings[query_part2_ids]
    query_vectors_zq = model.query_encoder(query_e1_embs, query_e2_embs)

    # 2. 准备候选集
    candidate_embs = entity_embeddings[answer_ids]

    # 3. 生成条件化嵌入
    conditional_embs = model.conditional_entity_encoder(candidate_embs, query_vectors_zq)
    # print(f"Debug - conditional_embs shape: {conditional_embs.shape}")
    # print(f"Debug - query_vectors_zq shape: {query_vectors_zq.shape}")
    # 4. 计算相似度矩阵
    query_vectors_zq = F.normalize(query_vectors_zq, p=2, dim=1)
    conditional_embs = F.normalize(conditional_embs, p=2, dim=2)
    score_matrix = torch.einsum("bd,bjd->bj", query_vectors_zq, conditional_embs)

    # 5. 计算InfoNCE损失
    scaled_scores = score_matrix / temperature
    targets = torch.arange(batch_size, device=entity_embeddings.device)
    loss = F.cross_entropy(scaled_scores, targets)

    return loss
