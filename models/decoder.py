"""
解码器模块
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    解码器：预测三元组关联 logit。

    注意：这里不再做 sigmoid。训练时使用 BCEWithLogitsLoss，
    评估 AUC/AUPR/F1 时再在外部做 torch.sigmoid(logits)。
    """

    def __init__(self, decoder_input_dim, dropout=0.4):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(decoder_input_dim, decoder_input_dim // 2)
        self.bn1 = nn.BatchNorm1d(decoder_input_dim // 2)

        self.fc2 = nn.Linear(decoder_input_dim // 2, decoder_input_dim // 4)
        self.bn2 = nn.BatchNorm1d(decoder_input_dim // 4)

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
        concatenated_features = torch.cat([graph_embed[drug_id, :], graph_embed[gene_id, :], graph_embed[disease_id, :]], dim=1)

        # 三层MLP
        x = self.act(self.fc1(concatenated_features))
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.act(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)

        logits = self.fc3(x).squeeze(dim=1)
        return logits
