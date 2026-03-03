"""
模型定义 - 超图对比学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import HypergraphConv, RGCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================== Query Encoder (QCCL) ===========================


class QueryEncoder(nn.Module):
    """
    查询编码器：将一个不完整的三元组编码成查询向量
    例如，对于 (drug, ?, disease) 查询，编码drug和disease的嵌入
    """

    def __init__(self, embed_dim: int, cond_dim: int):
        super(QueryEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(embed_dim, cond_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, e1_embs: torch.Tensor, e3_embs: torch.Tensor) -> torch.Tensor:
        query_input = torch.cat([e1_embs, e3_embs], dim=1)
        return self.mlp(query_input)


class FiLMLayer(nn.Module):
    """
    使用查询向量动态调制候选实体的特征
    """

    def __init__(self, embed_dim, cond_dim, ln=True):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, embed_dim)  # 缩放参数
        self.beta = nn.Linear(cond_dim, embed_dim)  # 偏移参数
        self.ln = nn.LayerNorm(embed_dim) if ln else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.gamma, self.beta]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, z_q):
        # h: [batch_size, embed_dim]
        # z_q: [batch_size, cond_dim]
        γ = self.gamma(z_q)  # [batch_size, embed_dim]
        β = self.beta(z_q)  # [batch_size, embed_dim]
        out = γ * h + β
        out = self.ln(out)
        return out + h  # 残差连接


class MultiLayerFiLMConditionalEntityEncoder(nn.Module):
    """
    多层FiLM条件实体编码器
    """

    def __init__(self, embed_dim, cond_dim, num_layers=2, ln=False):
        super().__init__()
        self.layers = nn.ModuleList([FiLMLayer(embed_dim, cond_dim, ln=ln) for _ in range(num_layers)])
        self.proj = nn.Linear(embed_dim, cond_dim)  # 投影到头维度
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, candidate_embs, query_vectors_zq):
        B, E = candidate_embs.shape
        h = candidate_embs.unsqueeze(0).expand(B, -1, -1)  # [B,B,E]
        for layer in self.layers:
            h = layer(h, query_vectors_zq)
        return self.proj(h)  # 投影到查询向量维度


# =========================== 主模型组件 ===========================


class BioEncoder(nn.Module):
    """
    生物特征编码器：编码药物、基因、疾病的原始特征
    """

    def __init__(self, dim_drug, dim_gene, dim_disease, encoder_output_dim):
        super(BioEncoder, self).__init__()

        # 药物编码器
        self.fc_drug = nn.Linear(dim_drug, encoder_output_dim)
        self.bn_drug = nn.BatchNorm1d(encoder_output_dim)

        # 基因编码器
        self.fc_gene = nn.Linear(dim_gene, encoder_output_dim)
        self.bn_gene = nn.BatchNorm1d(encoder_output_dim)

        # 疾病编码器
        self.fc_disease = nn.Linear(dim_disease, encoder_output_dim)
        self.bn_disease = nn.BatchNorm1d(encoder_output_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_feature, gene_feature, disease_feature):
        x_drug = self.act(self.bn_drug(self.fc_drug(drug_feature)))
        x_gene = self.act(self.bn_gene(self.fc_gene(gene_feature)))
        x_disease = self.act(self.bn_disease(self.fc_disease(disease_feature)))

        x_drug = self.dropout(x_drug)
        x_gene = self.dropout(x_gene)
        x_disease = self.dropout(x_disease)

        return x_drug, x_gene, x_disease


class HgnnEncoder(nn.Module):
    """
    超图神经网络编码器
    """

    def __init__(
        self,
        in_channels: int,
        dim_1: int,
        use_residual: bool = False,
        jk: str = "last",
        norm: str = "batch",
        hyperedge_dropout: float = 0.0,
        dropout: float = 0.4,
    ):
        super(HgnnEncoder, self).__init__()

        self.use_residual = bool(use_residual)
        self.jk = jk
        self.norm = norm
        self.hyperedge_dropout = float(hyperedge_dropout)

        # 三层超图卷积
        self.conv1 = HypergraphConv(in_channels, dim_1)
        self.conv2 = HypergraphConv(dim_1, dim_1 // 2)
        self.conv3 = HypergraphConv(dim_1 // 2, dim_1 // 4)

        def _make_norm(num_features: int) -> nn.Module:
            if self.norm == "layer":
                return nn.LayerNorm(num_features)
            return nn.BatchNorm1d(num_features)

        self.norm1 = _make_norm(dim_1)
        self.norm2 = _make_norm(dim_1 // 2)
        self.norm3 = _make_norm(dim_1 // 4)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projections (dimension-matching)
        if self.use_residual:
            self.res_proj1 = nn.Linear(in_channels, dim_1, bias=False)
            self.res_proj2 = nn.Linear(dim_1, dim_1 // 2, bias=False)
            self.res_proj3 = nn.Linear(dim_1 // 2, dim_1 // 4, bias=False)
        else:
            self.res_proj1 = None
            self.res_proj2 = None
            self.res_proj3 = None

        # Jumping Knowledge (multi-layer fusion)
        out_dim = dim_1 // 4
        self._jk_out_dim = out_dim
        if self.jk not in {"last", "sum", "mean", "concat"}:
            raise ValueError(f"Unsupported jk mode: {self.jk}")

        if self.jk != "last":
            # Project all layer outputs to out_dim, then fuse.
            self.jk_proj1 = nn.Linear(dim_1, out_dim, bias=False)
            self.jk_proj2 = nn.Linear(dim_1 // 2, out_dim, bias=False)
            self.jk_proj3 = nn.Identity()
            if self.jk == "concat":
                self.jk_fuse = nn.Linear(out_dim * 3, out_dim)
            else:
                self.jk_fuse = nn.Identity()
        else:
            self.jk_proj1 = None
            self.jk_proj2 = None
            self.jk_proj3 = None
            self.jk_fuse = None

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                try:
                    module.reset_parameters()
                except TypeError:
                    pass

        # Ensure linear layers have sane init even if nested modules don't expose reset.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _drop_hyperedge_incidence(self, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0 or not self.training or self.hyperedge_dropout <= 0.0:
            return edge_index

        keep_prob = 1.0 - self.hyperedge_dropout
        mask = torch.rand(edge_index.size(1), device=edge_index.device) < keep_prob
        if not torch.any(mask):
            # Keep at least one incidence to avoid empty edge_index.
            mask[torch.randint(0, edge_index.size(1), (1,), device=edge_index.device)] = True
        return edge_index[:, mask]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index = self._drop_hyperedge_incidence(edge_index)

        # Layer 1
        x_in = x
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = self.act(x1)
        x1 = self.dropout(x1)
        if self.use_residual and self.res_proj1 is not None:
            x1 = x1 + self.res_proj1(x_in)

        # Layer 2
        x_in = x1
        x2 = self.conv2(x1, edge_index)
        x2 = self.norm2(x2)
        x2 = self.act(x2)
        x2 = self.dropout(x2)
        if self.use_residual and self.res_proj2 is not None:
            x2 = x2 + self.res_proj2(x_in)

        # Layer 3
        x_in = x2
        x3 = self.conv3(x2, edge_index)
        x3 = self.norm3(x3)
        x3 = self.act(x3)
        if self.use_residual and self.res_proj3 is not None:
            x3 = x3 + self.res_proj3(x_in)

        if self.jk == "last":
            return x3

        p1 = self.jk_proj1(x1)
        p2 = self.jk_proj2(x2)
        p3 = x3

        if self.jk == "concat":
            fused = torch.cat([p1, p2, p3], dim=1)
            return self.jk_fuse(fused)
        if self.jk == "sum":
            return p1 + p2 + p3
        # mean
        return (p1 + p2 + p3) / 3.0


class RelationalGraphEncoder(nn.Module):
    """Relational GNN encoder for binary edges with explicit relation types.

    Uses RGCNConv over a single node space with an edge_type vector.
    """

    def __init__(self, in_channels: int, dim_1: int, num_relations: int = 4, use_residual: bool = False, dropout: float = 0.4):
        super().__init__()
        self.num_relations = num_relations
        self.use_residual = bool(use_residual)

        self.conv1 = RGCNConv(in_channels, dim_1, num_relations=num_relations)
        self.conv2 = RGCNConv(dim_1, dim_1 // 2, num_relations=num_relations)
        self.conv3 = RGCNConv(dim_1 // 2, dim_1 // 4, num_relations=num_relations)

        self.bn1 = nn.BatchNorm1d(dim_1)
        self.bn2 = nn.BatchNorm1d(dim_1 // 2)
        self.bn3 = nn.BatchNorm1d(dim_1 // 4)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projections (dimension-matching)
        if self.use_residual:
            self.res_proj1 = nn.Linear(in_channels, dim_1, bias=False)
            self.res_proj2 = nn.Linear(dim_1, dim_1 // 2, bias=False)
            self.res_proj3 = nn.Linear(dim_1 // 2, dim_1 // 4, bias=False)
        else:
            self.res_proj1 = None
            self.res_proj2 = None
            self.res_proj3 = None

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                try:
                    module.reset_parameters()
                except TypeError:
                    pass

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_type is None or edge_index.numel() == 0:
            # No binary edges included in this ablation.
            return torch.zeros((x.size(0), self.bn3.num_features), device=x.device, dtype=x.dtype)

        x_in = x
        x1 = self.conv1(x, edge_index, edge_type)
        x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.dropout(x1)
        if self.use_residual and self.res_proj1 is not None:
            x1 = x1 + self.res_proj1(x_in)

        x_in = x1
        x2 = self.conv2(x1, edge_index, edge_type)
        x2 = self.bn2(x2)
        x2 = self.act(x2)
        x2 = self.dropout(x2)
        if self.use_residual and self.res_proj2 is not None:
            x2 = x2 + self.res_proj2(x_in)

        x_in = x2
        x3 = self.conv3(x2, edge_index, edge_type)
        x3 = self.bn3(x3)
        x3 = self.act(x3)
        if self.use_residual and self.res_proj3 is not None:
            x3 = x3 + self.res_proj3(x_in)
        return x3


class Decoder(nn.Module):
    """
    解码器：预测三元组关联概率
    """

    def __init__(self, decoder_input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(decoder_input_dim, decoder_input_dim // 2)
        self.bn1 = nn.BatchNorm1d(decoder_input_dim // 2)

        self.fc2 = nn.Linear(decoder_input_dim // 2, decoder_input_dim // 4)
        self.bn2 = nn.BatchNorm1d(decoder_input_dim // 4)

        self.fc3 = nn.Linear(decoder_input_dim // 4, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, drug_id, gene_id, disease_id):
        # 拼接三个节点的特征
        concatenated_features = torch.cat([graph_embed[drug_id, :], graph_embed[gene_id, :], graph_embed[disease_id, :]], dim=1)

        # 三层MLP
        x = self.act(self.fc1(concatenated_features))
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.act(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return torch.sigmoid(x.squeeze(dim=1))


# =========================== QCCL损失函数 ===========================
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


# =========================== 主模型 ===========================


class HGNN(nn.Module):
    """
    主模型
    """

    def __init__(self, bio_encoder, graph_encoder, decoder, out_channels, tau=0.07, replace_dims=None, loss_type=None, binary_graph_encoder: nn.Module = None):
        super(HGNN, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.binary_graph_encoder = binary_graph_encoder
        self.decoder = decoder
        self.replace_dims = replace_dims if replace_dims else []
        self.loss_type = loss_type if loss_type else []

        # Fusion gate between hypergraph encoder and relational binary-edge encoder.
        # sigmoid(gate)=1 -> use hypergraph only; 0 -> use binary graph only.
        self._fusion_gate = nn.Parameter(torch.tensor(0.0))

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
            gate = torch.sigmoid(self._fusion_gate)
            graph_embed_pos = gate * hyper_embed + (1.0 - gate) * rel_embed
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
