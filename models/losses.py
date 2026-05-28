"""
损失函数模块
"""

from typing import Optional

import torch
import torch.nn.functional as F


def integrate_qccl(
    model,
    query_part1_ids,
    query_part2_ids,
    answer_ids,
    entity_embeddings,
    temperature=0.07,
    query_type: str = "gene",
    progress: float = 0.0,
    queue_ids: Optional[torch.Tensor] = None,
    queue_embeddings: Optional[torch.Tensor] = None,
    norm_mode: str = "l2",
    multi_positive_mode: str = "group",
    suppress_false_negatives: bool = False,
    use_hardness_reweight: bool = False,
    hardness_alpha: float = 0.25,
    hardness_tau: float = 0.5,
    use_query_hardness_reweight: bool = False,
    query_hardness_alpha: float = 1.0,
    query_hardness_tau: float = 1.0,
    enable_three_pool_schedule: bool = False,
    pool_schedule_start: float = 0.35,
    pool_schedule_end: float = 0.7,
    pool_schedule_mid_weight: float = 0.35,
    pool_schedule_late_weight: float = 0.75,
):
    """
    查询条件对比学习损失（QCCL v2）。

    支持以下增强：
    - Lite: 按 ID 聚合多正样本，避免同答案实体互为负样本。
    - Standard: 可选难负样本重加权 + 训练进度三池调度。
    - Advanced: 可选 memory queue 跨 batch 负样本。
    """

    device = entity_embeddings.device

    if not isinstance(query_part1_ids, torch.Tensor):
        query_part1_ids = torch.as_tensor(query_part1_ids, dtype=torch.long, device=device)
    else:
        query_part1_ids = query_part1_ids.to(device=device, dtype=torch.long)

    if not isinstance(query_part2_ids, torch.Tensor):
        query_part2_ids = torch.as_tensor(query_part2_ids, dtype=torch.long, device=device)
    else:
        query_part2_ids = query_part2_ids.to(device=device, dtype=torch.long)

    if not isinstance(answer_ids, torch.Tensor):
        answer_ids = torch.as_tensor(answer_ids, dtype=torch.long, device=device)
    else:
        answer_ids = answer_ids.to(device=device, dtype=torch.long)

    batch_size = int(answer_ids.numel())
    if batch_size == 0:
        return torch.tensor(0.0, device=device)

    # 1. 编码查询
    query_e1_embs = entity_embeddings[query_part1_ids]
    query_e2_embs = entity_embeddings[query_part2_ids]
    query_vectors_zq = model.query_encoder(query_e1_embs, query_e2_embs)

    # 2. 准备候选集（in-batch positives + optional queue negatives）
    base_candidate_ids = answer_ids
    base_candidate_embs = entity_embeddings[base_candidate_ids]

    if queue_ids is not None and queue_embeddings is not None and queue_ids.numel() > 0 and queue_embeddings.numel() > 0:
        queue_ids = queue_ids.to(device=device, dtype=torch.long).reshape(-1)
        queue_embeddings = queue_embeddings.to(device=device, dtype=base_candidate_embs.dtype).reshape(queue_ids.size(0), -1)
        candidate_ids = torch.cat([base_candidate_ids, queue_ids], dim=0)
        candidate_embs = torch.cat([base_candidate_embs, queue_embeddings], dim=0)
    else:
        candidate_ids = base_candidate_ids
        candidate_embs = base_candidate_embs

    num_candidates = int(candidate_ids.numel())

    # 3. 生成条件化嵌入
    conditional_embs = model.conditional_entity_encoder(candidate_embs, query_vectors_zq)

    # 4. 计算相似度矩阵
    norm_mode = str(norm_mode).lower().strip()
    if norm_mode in {"l2", "symmetric"}:
        query_vectors_zq = F.normalize(query_vectors_zq, p=2, dim=1)
        conditional_embs = F.normalize(conditional_embs, p=2, dim=2)
    elif norm_mode == "query_only":
        query_vectors_zq = F.normalize(query_vectors_zq, p=2, dim=1)
    elif norm_mode == "candidate_only":
        conditional_embs = F.normalize(conditional_embs, p=2, dim=2)
    score_matrix = torch.einsum("bd,bcd->bc", query_vectors_zq, conditional_embs)
    scaled_scores = score_matrix / max(float(temperature), 1e-6)

    # 5. 多正样本/掩码构造
    mode = str(multi_positive_mode).lower().strip()
    if mode not in {"group", "strict"}:
        mode = "group"

    if mode == "strict":
        positive_mask = torch.zeros((batch_size, num_candidates), dtype=torch.bool, device=device)
        diag_idx = torch.arange(batch_size, device=device)
        positive_mask[diag_idx, diag_idx] = True
    else:
        positive_mask = answer_ids.unsqueeze(1) == candidate_ids.unsqueeze(0)

    # 安全兜底：每行至少一个正样本
    has_positive = positive_mask.any(dim=1)
    if not has_positive.all():
        missing_rows = (~has_positive).nonzero(as_tuple=True)[0]
        positive_mask[missing_rows, missing_rows] = True

    negative_mask = ~positive_mask

    # 可选：抑制明显伪负样本（query 条件实体自身）
    if suppress_false_negatives:
        conflict_mask = (candidate_ids.unsqueeze(0) == query_part1_ids.unsqueeze(1)) | (candidate_ids.unsqueeze(0) == query_part2_ids.unsqueeze(1))
        negative_mask = negative_mask & (~conflict_mask)

    # 6. 负样本权重（standard/advanced）
    negative_weights = torch.ones_like(scaled_scores)

    # 6.1 三池调度：中期强调难负样本，后期增加 queue 负样本权重
    if enable_three_pool_schedule:
        p = float(progress)
        p = min(1.0, max(0.0, p))
        start = float(pool_schedule_start)
        end = float(pool_schedule_end)
        if end < start:
            start, end = end, start

        if p <= start:
            ramp = 0.0
        elif p >= end:
            ramp = 1.0
        else:
            ramp = (p - start) / max(end - start, 1e-6)

        mid_scale = max(0.0, float(pool_schedule_mid_weight)) * ramp
        late_scale = max(0.0, float(pool_schedule_late_weight)) * ramp

        col_index = torch.arange(num_candidates, device=device).unsqueeze(0).expand(batch_size, -1)
        in_batch_mask = col_index < batch_size
        queue_mask = col_index >= batch_size

        # 中期 pool: 在 in-batch negatives 中挑 top-k 相似度作为 hard negatives
        in_batch_neg_mask = negative_mask & in_batch_mask
        hard_mask = torch.zeros_like(in_batch_neg_mask)
        if in_batch_neg_mask.any() and mid_scale > 0.0:
            detached_scores = scaled_scores.detach().masked_fill(~in_batch_neg_mask, float("-inf"))
            topk = max(1, int(round(batch_size * 0.25)))
            topk = min(topk, batch_size)
            hard_idx = torch.topk(detached_scores, k=topk, dim=1).indices
            hard_mask.scatter_(1, hard_idx, True)
            hard_mask = hard_mask & in_batch_neg_mask

        if mid_scale > 0.0:
            negative_weights = torch.where(hard_mask, negative_weights * (1.0 + mid_scale), negative_weights)

        # 后期 pool: queue negatives
        if late_scale > 0.0:
            queue_neg_mask = negative_mask & queue_mask
            negative_weights = torch.where(queue_neg_mask, negative_weights * (1.0 + late_scale), negative_weights)

    # 6.2 难负样本重加权
    if use_hardness_reweight and negative_mask.any():
        alpha = max(0.0, float(hardness_alpha))
        tau = max(1e-6, float(hardness_tau))
        with torch.no_grad():
            pos_score_sum = (scaled_scores * positive_mask.float()).sum(dim=1, keepdim=True)
            pos_count = positive_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            pos_mean = pos_score_sum / pos_count
            hardness = torch.sigmoid((scaled_scores - pos_mean) / tau)
        negative_weights = torch.where(negative_mask, negative_weights * (1.0 + alpha * hardness), negative_weights)

    # 7. 带掩码 + 带权对比目标
    valid_mask = positive_mask | negative_mask
    masked_scores = scaled_scores.masked_fill(~valid_mask, float("-inf"))

    class_weights = torch.ones_like(masked_scores)
    class_weights = torch.where(negative_mask, negative_weights.clamp_min(1e-6), class_weights)
    weighted_scores = masked_scores + torch.log(class_weights)

    log_probs = F.log_softmax(weighted_scores, dim=1)
    pos_weights = positive_mask.float()
    pos_weights = pos_weights / pos_weights.sum(dim=1, keepdim=True).clamp_min(1.0)

    sample_loss = -(pos_weights * log_probs).sum(dim=1)

    if use_query_hardness_reweight:
        q_alpha = max(0.0, float(query_hardness_alpha))
        q_tau = max(1e-6, float(query_hardness_tau))
        with torch.no_grad():
            pos_score_sum = (scaled_scores * positive_mask.float()).sum(dim=1)
            pos_count = positive_mask.float().sum(dim=1).clamp_min(1.0)
            pos_mean = pos_score_sum / pos_count
            query_hardness = torch.sigmoid((-pos_mean) / q_tau)
            query_weights = 1.0 + q_alpha * query_hardness
        sample_loss = sample_loss * query_weights

    loss = sample_loss.mean()

    return loss
