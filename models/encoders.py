"""
编码器模块
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, RGCNConv


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
