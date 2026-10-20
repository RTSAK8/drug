"""Build positive training incidence and deduplicated typed auxiliary edges."""

import numpy as np
import torch


def build_hypergraph(positive_triplets: np.ndarray, device: torch.device) -> torch.Tensor:
    rows = np.asarray(positive_triplets, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 3 or len(rows) == 0:
        raise ValueError("A training hypergraph requires a nonempty (N,3) positive array")
    incidence = np.stack([rows.reshape(-1), np.repeat(np.arange(len(rows)), 3)])
    return torch.as_tensor(incidence, dtype=torch.long, device=device)


def relation_schema(directional: bool = False) -> dict[str, int]:
    schema = {"ppi": 0, "dpi": 1, "ddi": 2, "protein_disease": 3}
    if directional:
        schema.update(dpi_reverse=4, protein_disease_reverse=5)
    return schema


def build_binary_edges(relation_arrays: dict[str, np.ndarray], directional: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    relation_ids = relation_schema(directional)
    rows = []
    for name, edges in relation_arrays.items():
        if edges.size == 0:
            continue
        reverse = edges[:, ::-1]
        forward_type = relation_ids[name]
        reverse_type = relation_ids.get(f"{name}_reverse", forward_type)
        for pairs, relation_id in ((edges, forward_type), (reverse, reverse_type)):
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            rows.append(np.column_stack([pairs, np.full(len(pairs), relation_id)]))
    if not rows:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)
    typed = np.unique(np.concatenate(rows).astype(np.int64), axis=0)
    return torch.from_numpy(typed[:, :2].T.copy()), torch.from_numpy(typed[:, 2].copy())
