"""Feature projections and the two graph branches shown in the paper."""

import math

import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class BioEncoder(nn.Module):
    def __init__(self, dimensions: dict[str, int], output_dim: int, dropout: float):
        super().__init__()
        self.projections = nn.ModuleDict({kind: nn.Sequential(
            nn.Linear(dimensions[kind], output_dim), nn.BatchNorm1d(output_dim),
            nn.ReLU(), nn.Dropout(dropout),
        ) for kind in ("drug", "gene", "disease")})

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.projections[kind](features[kind]) for kind in ("drug", "gene", "disease")])


class HypergraphLayer(nn.Module):
    """Equation (4), with unit hyperedge weights and no attention.

    symmetric: Dv^-1/2 H De^-1 H^T Dv^-1/2 X Theta.
    random_walk is retained as an explicit alternative for legacy comparisons.
    """

    def __init__(self, in_channels: int, out_channels: int, normalization: str):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.normalization = normalization

    def forward(self, x: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        nodes, edges = incidence
        num_nodes = x.size(0)
        num_edges = int(edges.max().item()) + 1
        x = self.linear(x)
        node_degree = torch.bincount(nodes, minlength=num_nodes).to(x.dtype)
        edge_degree = torch.bincount(edges, minlength=num_edges).to(x.dtype).clamp_min(1)
        power = -0.5 if self.normalization == "symmetric" else -1.0
        scale = node_degree.clamp_min(1).pow(power)
        scale = scale.masked_fill(node_degree == 0, 0)
        source = x * scale[:, None] if self.normalization == "symmetric" else x
        edge_features = x.new_zeros((num_edges, x.size(1))).index_add(0, edges, source[nodes])
        edge_features = edge_features / edge_degree[:, None]
        output = x.new_zeros(x.shape).index_add(0, nodes, edge_features[edges])
        return output * scale[:, None] + self.bias


class GraphBranch(nn.Module):
    """Three layers; gated projected residuals only on the first two layers."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, residual: bool,
                 kind: str, normalization: str = "symmetric", hyperedge_dropout: float = 0.0):
        super().__init__()
        self.kind = kind
        self.hyperedge_dropout = hyperedge_dropout
        dims = [input_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4]
        self.convs = nn.ModuleList([
            HypergraphLayer(dims[i], dims[i+1], normalization) if kind == "hypergraph"
            else RGCNConv(dims[i], dims[i+1], num_relations=4)
            for i in range(3)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(dims[i+1]) for i in range(3)])
        self.residual = residual
        self.projections = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) for i in range(2)]) if residual else nn.ModuleList()
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(math.log(0.2 / 0.8))) for _ in range(2)]) if residual else nn.ParameterList()
        self.dropout = nn.Dropout(dropout)

    def _drop_hyperedges(self, incidence: torch.Tensor) -> torch.Tensor:
        if not self.training or self.hyperedge_dropout == 0:
            return incidence
        # Remove whole hyperedges, never individual nodes from a triplet.
        num_edges = int(incidence[1].max().item()) + 1
        keep = torch.rand(num_edges, device=incidence.device) >= self.hyperedge_dropout
        if not bool(keep.any()):
            keep[torch.randint(num_edges, (1,), device=incidence.device)] = True
        retained = incidence[:, keep[incidence[1]]]
        _, compact = torch.unique(retained[1], sorted=True, return_inverse=True)
        return torch.stack([retained[0], compact])

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_types=None) -> torch.Tensor:
        if self.kind == "hypergraph":
            edges = self._drop_hyperedges(edges)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            incoming = x
            x = conv(x, edges) if self.kind == "hypergraph" else conv(x, edges, edge_types)
            x = norm(x)
            if self.residual and i < 2:
                x = x + torch.sigmoid(self.gates[i]) * self.projections[i](incoming)
            x = torch.relu(x)
            if i < 2:
                x = self.dropout(x)
        return x
