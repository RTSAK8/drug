"""Pure representation/scoring model: no loss, labels, sampling, or metrics in forward."""

import torch
from torch import nn

from config import Config
from utils.hypergraph import relation_schema
from .decoder import Decoder
from .encoders import BioEncoder, GraphBranch


class HGNN(nn.Module):
    def __init__(self, config: Config, feature_dims: dict[str, int]):
        super().__init__()
        output_dim = config.hidden_dim // 4
        self.bio_encoder = BioEncoder(feature_dims, config.encoder_output_dim, config.feature_dropout)
        self.hypergraph_encoder = GraphBranch(
            config.encoder_output_dim, config.hidden_dim, config.hgnn_dropout,
            config.hgnn_use_residual, "hypergraph", config.hypergraph_normalization,
            config.hgnn_hyperedge_dropout,
            final_residual=config.final_layer_residual, feature_residual=config.feature_residual,
        ) if config.use_hypergraph else None
        self.relational_encoder = GraphBranch(
            config.encoder_output_dim, config.hidden_dim, config.rgcn_dropout,
            config.rgcn_use_residual, "relational",
            final_residual=config.final_layer_residual, feature_residual=config.feature_residual,
            num_relations=len(relation_schema(config.directional_relations)),
        ) if config.use_relational_graph else None
        self.fusion_method = config.fusion_method
        if config.use_hypergraph and config.use_relational_graph:
            if self.fusion_method == "concat":
                self.fusion = nn.Linear(2 * output_dim, output_dim)
            else:
                self.gate = nn.Parameter(torch.tensor(0.0))
        if not config.use_hypergraph and not config.use_relational_graph:
            self.feature_projection = nn.Linear(config.encoder_output_dim, output_dim)
        self.decoder = Decoder(3 * output_dim, dropout=config.decoder_dropout,
                               normalization=config.decoder_normalization, interactions=config.decoder_interactions)

    def encode(self, features, incidence, edge_index, edge_type) -> torch.Tensor:
        x = self.bio_encoder(features)
        hyper = self.hypergraph_encoder(x, incidence) if self.hypergraph_encoder is not None else None
        relational = self.relational_encoder(x, edge_index, edge_type) if self.relational_encoder is not None else None
        if hyper is not None and relational is not None:
            if self.fusion_method == "concat":
                return self.fusion(torch.cat([hyper, relational], dim=1))
            gate = torch.sigmoid(self.gate)
            return gate * hyper + (1 - gate) * relational
        if hyper is not None:
            return hyper
        if relational is not None:
            return relational
        return self.feature_projection(x)

    def score(self, embeddings: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
        return self.decoder(embeddings, triplets[:, 0], triplets[:, 1], triplets[:, 2])

    def forward(self, features, incidence, edge_index, edge_type, triplets):
        return self.score(self.encode(features, incidence, edge_index, edge_type), triplets)
