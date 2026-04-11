"""
查询条件化相关模块（QCCL）
"""

import torch
import torch.nn as nn


class QueryEncoder(nn.Module):
    """
    查询编码器：将一个不完整的三元组编码成查询向量
    例如，对于 (drug, ?, disease) 查询，编码drug和disease的嵌入
    """

    def __init__(self, embed_dim: int, cond_dim: int, dropout: float = 0.3):
        super(QueryEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, cond_dim))
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
        """
        Args:
            candidate_embs: [num_candidates, embed_dim]
            query_vectors_zq: [num_queries, cond_dim]
        Returns:
            [num_queries, num_candidates, cond_dim]
        """
        num_queries, _ = query_vectors_zq.shape
        num_candidates, _ = candidate_embs.shape

        # Expand candidates across query axis and apply per-query FiLM parameters.
        h = candidate_embs.unsqueeze(0).expand(num_queries, num_candidates, -1)  # [Q,C,E]
        z_q = query_vectors_zq.unsqueeze(1)  # [Q,1,D]

        for layer in self.layers:
            h = layer(h, z_q)
        return self.proj(h)  # 投影到查询向量维度
