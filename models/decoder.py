"""
解码器模块
"""

import torch
import torch.nn as nn


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
