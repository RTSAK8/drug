"""
解码器模块
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    解码器：预测三元组关联 logit。

    注意：这里不再做 sigmoid。训练时使用 BCEWithLogitsLoss，
    AUROC/AP直接使用logit；sigmoid仅提供未校准的分数。
    """

    def __init__(self, decoder_input_dim, dropout=0.4, normalization="batch", interactions=False):
        super(Decoder, self).__init__()
        self.interactions = interactions
        norms = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "none": lambda size: nn.Identity()}
        if normalization not in norms:
            raise ValueError(f"Unknown decoder normalization: {normalization}")
        self.fc1 = nn.Linear(decoder_input_dim * (2 if interactions else 1), decoder_input_dim // 2)
        # Keep names for compatibility with the paper architecture's state dict.
        self.bn1 = norms[normalization](decoder_input_dim // 2)

        self.fc2 = nn.Linear(decoder_input_dim // 2, decoder_input_dim // 4)
        self.bn2 = norms[normalization](decoder_input_dim // 4)

        self.fc3 = nn.Linear(decoder_input_dim // 4, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, drug_id, gene_id, disease_id):
        # 拼接三个节点的特征
        drug, gene, disease = graph_embed[drug_id], graph_embed[gene_id], graph_embed[disease_id]
        parts = [drug, gene, disease]
        if self.interactions:
            parts.extend([drug * gene, drug * disease, gene * disease])
        concatenated_features = torch.cat(parts, dim=1)

        # 三层MLP
        x = self.act(self.fc1(concatenated_features))
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.act(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        logits = self.fc3(x).squeeze(dim=1)
        return logits
