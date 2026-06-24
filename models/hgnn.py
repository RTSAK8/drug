"""
主模型定义
"""

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .losses import integrate_qccl, integrate_qfal
from .query_conditioning import QueryEncoder, MultiLayerFiLMConditionalEntityEncoder
from utils.runtime import get_device

device = get_device()


class HGNN(nn.Module):
    """
    主模型
    """

    def __init__(
        self,
        bio_encoder,
        graph_encoder,
        decoder,
        out_channels,
        tau=0.07,
        replace_dims=None,
        loss_type=None,
        binary_graph_encoder: nn.Module = None,
        fusion_method: str = "gate",
        qccl_config: Optional[Mapping[str, Any]] = None,
        qfal_config: Optional[Mapping[str, Any]] = None,
    ):
        super(HGNN, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.binary_graph_encoder = binary_graph_encoder
        self.decoder = decoder
        self.replace_dims = replace_dims if replace_dims else []
        self.loss_type = loss_type if loss_type else []

        # How to fuse hypergraph branch and binary-relation branch when both are present.
        # - "gate": sigmoid gate weighted sum (default, backward compatible)
        # - "concat": concatenate then project back to out_channels
        self.fusion_method = str(fusion_method).lower().strip() if fusion_method is not None else "gate"
        if self.fusion_method not in {"gate", "concat"}:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}. Use 'gate' or 'concat'.")

        # Fusion gate between hypergraph encoder and relational binary-edge encoder.
        # sigmoid(gate)=1 -> use hypergraph only; 0 -> use binary graph only.
        self._fusion_gate = nn.Parameter(torch.tensor(0.0))

        # Concat fusion projector (keeps output dim unchanged so Decoder stays the same).
        self.concat_fuse_proj = nn.Linear(out_channels * 2, out_channels)

        # QCCL v2 runtime controls.
        self.qccl_cfg = self._resolve_qccl_cfg(qccl_config)
        self.qfal_cfg = self._resolve_qfal_cfg(qfal_config)
        self.qccl_progress = 0.0

        # Typed memory queues for advanced QCCL negatives.
        self.register_buffer("_qccl_queue_drug_embs", torch.empty((0, out_channels)), persistent=False)
        self.register_buffer("_qccl_queue_gene_embs", torch.empty((0, out_channels)), persistent=False)
        self.register_buffer("_qccl_queue_disease_embs", torch.empty((0, out_channels)), persistent=False)
        self.register_buffer("_qccl_queue_drug_ids", torch.empty((0,), dtype=torch.long), persistent=False)
        self.register_buffer("_qccl_queue_gene_ids", torch.empty((0,), dtype=torch.long), persistent=False)
        self.register_buffer("_qccl_queue_disease_ids", torch.empty((0,), dtype=torch.long), persistent=False)

        # 对比学习参数
        if "qccl" in self.loss_type:
            graph_embed_dim = out_channels
            cond_ratio = float(self.qccl_cfg.get("cond_dim_ratio", 0.5))
            cond_dim = max(1, int(round(graph_embed_dim * cond_ratio)))
            self.query_encoder = QueryEncoder(graph_embed_dim, cond_dim, dropout=float(self.qccl_cfg.get("query_dropout", 0.3)))
            self.conditional_entity_encoder = MultiLayerFiLMConditionalEntityEncoder(
                embed_dim=graph_embed_dim,
                cond_dim=cond_dim,
                num_layers=int(self.qccl_cfg.get("num_layers", 2)),
                ln=bool(self.qccl_cfg.get("film_layer_norm", False)),
            )

        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""

        def _reset(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        _reset(self.bio_encoder)
        _reset(self.graph_encoder)
        _reset(self.binary_graph_encoder)
        _reset(self.decoder)
        if hasattr(self, "concat_fuse_proj") and isinstance(self.concat_fuse_proj, nn.Linear):
            nn.init.xavier_uniform_(self.concat_fuse_proj.weight)
            if self.concat_fuse_proj.bias is not None:
                nn.init.zeros_(self.concat_fuse_proj.bias)
        if "qccl" in self.loss_type:
            _reset(self.query_encoder)
            _reset(self.conditional_entity_encoder)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def _resolve_qccl_cfg(cls, qccl_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "temperature": 0.3,
            "gene_query_temperature": None,
            "drug_query_temperature": None,
            "disease_query_temperature": None,
            "norm_mode": "l2",
            "multi_positive_mode": "group",
            "suppress_false_negatives": False,
            "use_hardness_reweight": False,
            "hardness_alpha": 0.25,
            "hardness_tau": 0.5,
            "use_query_hardness_reweight": False,
            "query_hardness_alpha": 1.0,
            "query_hardness_tau": 1.0,
            "enable_three_pool_schedule": False,
            "pool_schedule_start": 0.35,
            "pool_schedule_end": 0.7,
            "pool_schedule_mid_weight": 0.35,
            "pool_schedule_late_weight": 0.75,
            "enable_memory_queue": False,
            "memory_queue_size": 8192,
            "memory_queue_warmup_steps": 100,
            "memory_queue_negatives_per_query": 256,
            "memory_queue_unique_only": True,
            "cond_dim_ratio": 0.5,
            "num_layers": 2,
            "query_dropout": 0.3,
            "film_layer_norm": False,
        }
        cfg = dict(defaults)
        if qccl_config is not None:
            for key, value in dict(qccl_config).items():
                if key in cfg and value is not None:
                    cfg[key] = value

        cfg["temperature"] = max(1e-6, float(cfg["temperature"]))
        for key in ("gene_query_temperature", "drug_query_temperature", "disease_query_temperature"):
            if cfg[key] is not None:
                cfg[key] = max(1e-6, float(cfg[key]))

        cfg["norm_mode"] = str(cfg["norm_mode"]).lower().strip()
        if cfg["norm_mode"] not in {"l2", "symmetric", "query_only", "candidate_only", "none"}:
            cfg["norm_mode"] = "l2"

        cfg["multi_positive_mode"] = str(cfg["multi_positive_mode"]).lower().strip()
        if cfg["multi_positive_mode"] not in {"group", "strict"}:
            cfg["multi_positive_mode"] = "group"

        for key in (
            "suppress_false_negatives",
            "use_hardness_reweight",
            "use_query_hardness_reweight",
            "enable_three_pool_schedule",
            "enable_memory_queue",
            "memory_queue_unique_only",
            "film_layer_norm",
        ):
            cfg[key] = cls._to_bool(cfg[key])

        cfg["hardness_alpha"] = max(0.0, float(cfg["hardness_alpha"]))
        cfg["hardness_tau"] = max(1e-6, float(cfg["hardness_tau"]))
        cfg["query_hardness_alpha"] = max(0.0, float(cfg["query_hardness_alpha"]))
        cfg["query_hardness_tau"] = max(1e-6, float(cfg["query_hardness_tau"]))
        cfg["pool_schedule_start"] = min(1.0, max(0.0, float(cfg["pool_schedule_start"])))
        cfg["pool_schedule_end"] = min(1.0, max(0.0, float(cfg["pool_schedule_end"])))
        if cfg["pool_schedule_end"] < cfg["pool_schedule_start"]:
            cfg["pool_schedule_start"], cfg["pool_schedule_end"] = cfg["pool_schedule_end"], cfg["pool_schedule_start"]
        cfg["pool_schedule_mid_weight"] = min(1.0, max(0.0, float(cfg["pool_schedule_mid_weight"])))
        cfg["pool_schedule_late_weight"] = min(1.0, max(0.0, float(cfg["pool_schedule_late_weight"])))

        cfg["memory_queue_size"] = max(0, int(cfg["memory_queue_size"]))
        cfg["memory_queue_warmup_steps"] = max(0, int(cfg["memory_queue_warmup_steps"]))
        cfg["memory_queue_negatives_per_query"] = max(0, int(cfg["memory_queue_negatives_per_query"]))

        cfg["cond_dim_ratio"] = max(0.05, float(cfg["cond_dim_ratio"]))
        cfg["num_layers"] = max(1, int(cfg["num_layers"]))
        cfg["query_dropout"] = min(1.0, max(0.0, float(cfg["query_dropout"])))
        return cfg

    @classmethod
    def _resolve_qfal_cfg(cls, qfal_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "num_negatives": 64,
            "temperature": 0.2,
            "false_negative_weight": 0.05,
            "known_positive_masking": True,
            "self_adversarial": True,
            "adversarial_temperature": 1.0,
            "max_queries": 2048,
        }
        cfg = dict(defaults)
        if qfal_config is not None:
            for key, value in dict(qfal_config).items():
                if key in cfg and value is not None:
                    cfg[key] = value

        cfg["num_negatives"] = max(1, int(cfg["num_negatives"]))
        cfg["temperature"] = max(1e-6, float(cfg["temperature"]))
        cfg["false_negative_weight"] = min(1.0, max(0.0, float(cfg["false_negative_weight"])))
        cfg["known_positive_masking"] = cls._to_bool(cfg["known_positive_masking"])
        cfg["self_adversarial"] = cls._to_bool(cfg["self_adversarial"])
        cfg["adversarial_temperature"] = max(1e-6, float(cfg["adversarial_temperature"]))
        cfg["max_queries"] = max(0, int(cfg["max_queries"]))
        return cfg

    def set_qccl_progress(self, epoch_progress: Optional[float]) -> None:
        if epoch_progress is None:
            return
        try:
            value = float(epoch_progress)
        except (TypeError, ValueError):
            return
        self.qccl_progress = min(1.0, max(0.0, value))

    def _resolve_query_temperature(self, query_type: str) -> float:
        if query_type == "gene":
            value = self.qccl_cfg.get("gene_query_temperature")
        elif query_type == "drug":
            value = self.qccl_cfg.get("drug_query_temperature")
        elif query_type == "disease":
            value = self.qccl_cfg.get("disease_query_temperature")
        else:
            value = None
        if value is None:
            value = self.qccl_cfg.get("temperature", 0.3)
        return max(1e-6, float(value))

    @staticmethod
    def _queue_buffer_names(query_type: str) -> Tuple[str, str]:
        if query_type == "drug":
            return "_qccl_queue_drug_ids", "_qccl_queue_drug_embs"
        if query_type == "gene":
            return "_qccl_queue_gene_ids", "_qccl_queue_gene_embs"
        if query_type == "disease":
            return "_qccl_queue_disease_ids", "_qccl_queue_disease_embs"
        raise ValueError(f"Unknown query_type for queue: {query_type}")

    def get_qccl_queue(self, query_type: str):
        if not self.qccl_cfg.get("enable_memory_queue", False):
            return None, None

        ids_name, emb_name = self._queue_buffer_names(query_type)
        queue_ids = getattr(self, ids_name)
        queue_embs = getattr(self, emb_name)

        if queue_ids.numel() == 0 or queue_embs.numel() == 0:
            return None, None

        warmup_steps = int(self.qccl_cfg.get("memory_queue_warmup_steps", 0))
        if warmup_steps > 0 and queue_ids.size(0) < warmup_steps:
            return None, None

        negatives_per_query = int(self.qccl_cfg.get("memory_queue_negatives_per_query", 0))
        if negatives_per_query > 0 and queue_ids.size(0) > negatives_per_query:
            sampled = torch.randperm(queue_ids.size(0), device=queue_ids.device)[:negatives_per_query]
            queue_ids = queue_ids.index_select(0, sampled)
            queue_embs = queue_embs.index_select(0, sampled)

        return queue_ids, queue_embs

    def update_qccl_queue(self, query_type: str, new_ids: torch.Tensor, new_embeddings: torch.Tensor) -> None:
        if not self.qccl_cfg.get("enable_memory_queue", False):
            return
        if new_ids is None or new_embeddings is None:
            return

        if not isinstance(new_ids, torch.Tensor):
            new_ids = torch.as_tensor(new_ids, dtype=torch.long, device=new_embeddings.device)
        else:
            new_ids = new_ids.to(dtype=torch.long, device=new_embeddings.device)

        new_ids = new_ids.reshape(-1)
        if new_ids.numel() == 0:
            return

        new_embeddings = new_embeddings.reshape(new_ids.numel(), -1)

        ids_name, emb_name = self._queue_buffer_names(query_type)
        old_ids = getattr(self, ids_name).to(new_ids.device)
        old_embs = getattr(self, emb_name).to(new_embeddings.device)

        merged_ids = torch.cat([old_ids, new_ids], dim=0)
        merged_embs = torch.cat([old_embs, new_embeddings], dim=0)

        if self.qccl_cfg.get("memory_queue_unique_only", True):
            seen = set()
            keep_indices_rev = []
            for i in range(merged_ids.size(0) - 1, -1, -1):
                entity_id = int(merged_ids[i].item())
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                keep_indices_rev.append(i)
            keep_indices = torch.tensor(list(reversed(keep_indices_rev)), device=merged_ids.device, dtype=torch.long)
            merged_ids = merged_ids.index_select(0, keep_indices)
            merged_embs = merged_embs.index_select(0, keep_indices)

        max_queue_size = int(self.qccl_cfg.get("memory_queue_size", 0))
        if max_queue_size > 0 and merged_ids.size(0) > max_queue_size:
            merged_ids = merged_ids[-max_queue_size:]
            merged_embs = merged_embs[-max_queue_size:]

        setattr(self, ids_name, merged_ids.detach())
        setattr(self, emb_name, merged_embs.detach())

    def forward(
        self,
        drug_feature,
        gene_feature,
        disease_feature,
        edge_pos,
        edge_neg,
        drug_id,
        gene_id,
        disease_id,
        pos_drugid,
        pos_geneid,
        pos_diseaseid,
        bin_edge_index: torch.Tensor = None,
        bin_edge_type: torch.Tensor = None,
        epoch_progress: Optional[float] = None,
    ):
        """
        前向传播

        Args:
            drug_feature: 药物特征
            gene_feature: 基因特征
            disease_feature: 疾病特征
            edge_pos: 正样本超图边
            edge_neg: 负样本超图边
            drug_id, gene_id, disease_id: 查询三元组的ID
            pos_drugid, pos_geneid, pos_diseaseid: 正样本三元组的ID
        """
        self.set_qccl_progress(epoch_progress)

        # 1. 编码生物特征
        x_drug, x_gene, x_disease = self.bio_encoder(drug_feature, gene_feature, disease_feature)
        embed = torch.cat([x_drug, x_gene, x_disease], dim=0)

        # 2. 图编码（超图分支 + 二元关系分支）
        hyper_embed = self.graph_encoder(embed, edge_pos) if edge_pos is not None and edge_pos.numel() > 0 else None

        rel_embed = None
        if self.binary_graph_encoder is not None and bin_edge_index is not None and bin_edge_type is not None and bin_edge_index.numel() > 0 and bin_edge_type.numel() > 0:
            rel_embed = self.binary_graph_encoder(embed, bin_edge_index, bin_edge_type)

        if hyper_embed is not None and rel_embed is not None:
            if self.fusion_method == "concat":
                graph_embed_pos = self.concat_fuse_proj(torch.cat([hyper_embed, rel_embed], dim=1))
            else:
                gate = torch.sigmoid(self._fusion_gate)
                graph_embed_pos = gate * hyper_embed + (1.0 - gate) * rel_embed
        # 这部分删去不影响
        elif hyper_embed is not None:
            graph_embed_pos = hyper_embed
        elif rel_embed is not None:
            graph_embed_pos = rel_embed
        else:
            graph_embed_pos = embed

        # 3. 解码预测结果
        res = self.decoder(graph_embed_pos, drug_id, gene_id, disease_id)

        # 4. 计算对比学习损失
        CL_loss = torch.tensor(0.0, device=graph_embed_pos.device)
        if "qccl" in self.loss_type and len(pos_drugid) > 0:

            def _to_id_tensor(values):
                if isinstance(values, torch.Tensor):
                    return values.to(device=graph_embed_pos.device, dtype=torch.long)
                return torch.tensor(values, device=graph_embed_pos.device, dtype=torch.long)

            norm_mode = self.qccl_cfg.get("norm_mode", "l2")
            multi_positive_mode = self.qccl_cfg.get("multi_positive_mode", "group")

            def qccl_loss_for(query_head_ids, query_tail_ids, answer_ids, query_type: str):
                query_head_ids = _to_id_tensor(query_head_ids)
                query_tail_ids = _to_id_tensor(query_tail_ids)
                answer_ids = _to_id_tensor(answer_ids)

                queue_ids, queue_embeddings = self.get_qccl_queue(query_type)
                loss = integrate_qccl(
                    self,
                    query_head_ids,
                    query_tail_ids,
                    answer_ids,
                    entity_embeddings=graph_embed_pos,
                    temperature=self._resolve_query_temperature(query_type),
                    query_type=query_type,
                    progress=self.qccl_progress,
                    queue_ids=queue_ids,
                    queue_embeddings=queue_embeddings,
                    norm_mode=norm_mode,
                    multi_positive_mode=multi_positive_mode,
                    suppress_false_negatives=bool(self.qccl_cfg.get("suppress_false_negatives", False)),
                    use_hardness_reweight=bool(self.qccl_cfg.get("use_hardness_reweight", False)),
                    hardness_alpha=float(self.qccl_cfg.get("hardness_alpha", 0.25)),
                    hardness_tau=float(self.qccl_cfg.get("hardness_tau", 0.5)),
                    use_query_hardness_reweight=bool(self.qccl_cfg.get("use_query_hardness_reweight", False)),
                    query_hardness_alpha=float(self.qccl_cfg.get("query_hardness_alpha", 1.0)),
                    query_hardness_tau=float(self.qccl_cfg.get("query_hardness_tau", 1.0)),
                    enable_three_pool_schedule=bool(self.qccl_cfg.get("enable_three_pool_schedule", False)),
                    pool_schedule_start=float(self.qccl_cfg.get("pool_schedule_start", 0.35)),
                    pool_schedule_end=float(self.qccl_cfg.get("pool_schedule_end", 0.7)),
                    pool_schedule_mid_weight=float(self.qccl_cfg.get("pool_schedule_mid_weight", 0.35)),
                    pool_schedule_late_weight=float(self.qccl_cfg.get("pool_schedule_late_weight", 0.75)),
                )

                if self.training:
                    valid_mask = (answer_ids >= 0) & (answer_ids < graph_embed_pos.size(0))
                    if valid_mask.any():
                        valid_answer_ids = answer_ids[valid_mask]
                        valid_answer_embeddings = graph_embed_pos[valid_answer_ids].detach()
                        self.update_qccl_queue(query_type, valid_answer_ids, valid_answer_embeddings)

                return loss

            # 三类查询
            loss_gene = qccl_loss_for(pos_drugid, pos_diseaseid, pos_geneid, query_type="gene")
            loss_drug = qccl_loss_for(pos_geneid, pos_diseaseid, pos_drugid, query_type="drug")
            loss_disease = qccl_loss_for(pos_drugid, pos_geneid, pos_diseaseid, query_type="disease")
            CL_loss = (loss_gene + loss_drug + loss_disease) / 3

        # 5. 计算 QFAL：decoder-based listwise ranking loss
        QFAL_loss = torch.tensor(0.0, device=graph_embed_pos.device)
        if "qfal" in self.loss_type and self.training and len(pos_drugid) > 0:
            QFAL_loss = integrate_qfal(
                model=self,
                graph_embed=graph_embed_pos,
                pos_drug_ids=pos_drugid,
                pos_gene_ids=pos_geneid,
                pos_disease_ids=pos_diseaseid,
                num_drugs=x_drug.shape[0],
                num_genes=x_gene.shape[0],
                num_diseases=x_disease.shape[0],
                replace_dims=self.replace_dims,
                num_negatives=int(self.qfal_cfg.get("num_negatives", 64)),
                temperature=float(self.qfal_cfg.get("temperature", 0.2)),
                false_negative_weight=float(self.qfal_cfg.get("false_negative_weight", 0.05)),
                known_positive_masking=bool(self.qfal_cfg.get("known_positive_masking", True)),
                self_adversarial=bool(self.qfal_cfg.get("self_adversarial", True)),
                adversarial_temperature=float(self.qfal_cfg.get("adversarial_temperature", 1.0)),
                max_queries=int(self.qfal_cfg.get("max_queries", 2048)),
            )

        # 6. 计算排名指标
        mrr = torch.tensor(0.0, device=graph_embed_pos.device)
        all_ranks = []

        if self.replace_dims and len(pos_drugid) > 0 and pos_geneid is not None and pos_diseaseid is not None:
            mrr, all_ranks = self._calculate_ranking_metrics(graph_embed_pos, pos_drugid, pos_geneid, pos_diseaseid, x_drug.shape[0], x_gene.shape[0], x_disease.shape[0])

        return res, CL_loss, QFAL_loss, mrr, all_ranks

    def _calculate_ranking_metrics(self, graph_embed, pos_drugid, pos_geneid, pos_diseaseid, num_drugs, num_genes, num_diseases):
        """计算MRR和排名"""
        # 代码需要review，好像有点问题
        total_samples = len(pos_drugid)
        unique_drug_ids = torch.arange(0, num_drugs, device=device)
        unique_gene_ids = torch.arange(num_drugs, num_drugs + num_genes, device=device)
        unique_disease_ids = torch.arange(num_drugs + num_genes, num_drugs + num_genes + num_diseases, device=device)

        mrr = torch.tensor(0.0, device=device)
        all_ranks = []

        if not isinstance(pos_drugid, torch.Tensor):
            pos_drugid = torch.tensor(pos_drugid, device=device)
        if not isinstance(pos_geneid, torch.Tensor):
            pos_geneid = torch.tensor(pos_geneid, device=device)
        if not isinstance(pos_diseaseid, torch.Tensor):
            pos_diseaseid = torch.tensor(pos_diseaseid, device=device)

        # 向量化计算排名
        if self.replace_dims == ["drug"] and len(unique_drug_ids) > 1:
            mrr, all_ranks = self._rank_entities(graph_embed, pos_geneid, pos_diseaseid, pos_drugid, unique_drug_ids, total_samples, self.replace_dims[0])
        elif self.replace_dims == ["gene"] and len(unique_gene_ids) > 1:
            mrr, all_ranks = self._rank_entities(graph_embed, pos_drugid, pos_diseaseid, pos_geneid, unique_gene_ids, total_samples, self.replace_dims[0])
        elif self.replace_dims == ["disease"] and len(unique_disease_ids) > 1:
            mrr, all_ranks = self._rank_entities(graph_embed, pos_drugid, pos_geneid, pos_diseaseid, unique_disease_ids, total_samples, self.replace_dims[0])

        return mrr, all_ranks

    def _rank_entities(self, graph_embed, id1, id2, target_ids, candidate_ids, batch_size, entity_type):
        """通用排名计算函数"""
        # 构建批处理输入
        # 待review
        batch_id1 = id1.unsqueeze(1).expand(-1, len(candidate_ids))
        batch_id2 = id2.unsqueeze(1).expand(-1, len(candidate_ids))
        batch_candidates = candidate_ids.unsqueeze(0).expand(batch_size, -1)

        # 重塑为批量形式
        if entity_type == "drug":
            flat_scores = self.decoder(graph_embed, batch_candidates.reshape(-1), batch_id1.reshape(-1), batch_id2.reshape(-1))
        elif entity_type == "gene":
            flat_scores = self.decoder(graph_embed, batch_id1.reshape(-1), batch_candidates.reshape(-1), batch_id2.reshape(-1))
        else:  # disease
            flat_scores = self.decoder(graph_embed, batch_id1.reshape(-1), batch_id2.reshape(-1), batch_candidates.reshape(-1))

        scores = flat_scores.reshape(batch_size, -1)
        sorted_indices = torch.argsort(scores, dim=1, descending=True)

        # 获取排名
        candidate_matrix = candidate_ids.unsqueeze(0).expand(batch_size, -1)
        sorted_candidates = torch.gather(candidate_matrix, 1, sorted_indices)
        target_matrix = target_ids.unsqueeze(1).expand(-1, len(candidate_ids))

        match_matrix = sorted_candidates == target_matrix
        # 正常情况下 target 必然在候选全集中；这里保底处理异常映射，避免 rank 数量少于 batch_size。
        has_match = match_matrix.any(dim=1)
        first_match = match_matrix.to(torch.int64).argmax(dim=1) + 1
        fallback_rank = torch.full_like(first_match, fill_value=len(candidate_ids) + 1)
        ranks = torch.where(has_match, first_match, fallback_rank)

        all_ranks = ranks.cpu().numpy().tolist()
        reciprocal_ranks = 1.0 / ranks.float()
        mrr = reciprocal_ranks.mean()

        return mrr, all_ranks
