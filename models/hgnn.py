"""
主模型定义
"""

import torch
import torch.nn as nn

from .losses import integrate_qccl
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

        # 对比学习参数
        if "qccl" in self.loss_type:
            graph_embed_dim = out_channels
            cond_dim = graph_embed_dim // 2
            self.query_encoder = QueryEncoder(graph_embed_dim, cond_dim)
            self.conditional_entity_encoder = MultiLayerFiLMConditionalEntityEncoder(embed_dim=graph_embed_dim, cond_dim=cond_dim, num_layers=2)

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

    def forward(self, drug_feature, gene_feature, disease_feature, edge_pos, edge_neg, drug_id, gene_id, disease_id, pos_drugid, pos_geneid, pos_diseaseid, bin_edge_index: torch.Tensor = None, bin_edge_type: torch.Tensor = None):
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
        CL_loss = torch.tensor(0.0, device=device)
        if "qccl" in self.loss_type and len(pos_drugid) > 0:

            def qccl_loss_for(query_head_ids, query_tail_ids, answer_ids):
                return integrate_qccl(self, query_head_ids, query_tail_ids, answer_ids, entity_embeddings=graph_embed_pos, temperature=0.3)

            # 三类查询
            loss_gene = qccl_loss_for(pos_drugid, pos_diseaseid, pos_geneid)
            loss_drug = qccl_loss_for(pos_geneid, pos_diseaseid, pos_drugid)
            loss_disease = qccl_loss_for(pos_drugid, pos_geneid, pos_diseaseid)
            CL_loss = (loss_gene + loss_drug + loss_disease) / 3

        # 5. 计算排名指标
        mrr = torch.tensor(0.0, device=device)
        all_ranks = []

        if self.replace_dims and len(pos_drugid) > 0 and pos_geneid is not None and pos_diseaseid is not None:
            mrr, all_ranks = self._calculate_ranking_metrics(graph_embed_pos, pos_drugid, pos_geneid, pos_diseaseid, x_drug.shape[0], x_gene.shape[0], x_disease.shape[0])

        return res, CL_loss, mrr, all_ranks

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

        ranks = (sorted_candidates == target_matrix).nonzero(as_tuple=True)[1] + 1
        all_ranks = ranks.cpu().numpy().tolist()
        reciprocal_ranks = 1.0 / ranks.float()
        mrr = reciprocal_ranks.mean()

        return mrr, all_ranks
