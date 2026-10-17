"""Build positive training incidence and deduplicated typed auxiliary edges."""

import numpy as np
import torch


def build_hypergraph(positive_triplets: np.ndarray, device: torch.device) -> torch.Tensor:
    rows = np.asarray(positive_triplets, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 3 or len(rows) == 0:
        raise ValueError("A training hypergraph requires a nonempty (N,3) positive array")
    incidence = np.stack([rows.reshape(-1), np.repeat(np.arange(len(rows)), 3)])
    return torch.as_tensor(incidence, dtype=torch.long, device=device)


def build_binary_edges(relation_arrays: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    relation_ids = {"ppi": 0, "dpi": 1, "ddi": 2, "protein_disease": 3}
    rows = []
    for name, edges in relation_arrays.items():
        if edges.size == 0:
            continue
        reverse = edges[:, ::-1]
        both = np.concatenate([edges, reverse], axis=0)
        both = both[both[:, 0] != both[:, 1]]
        rows.append(np.column_stack([both, np.full(len(both), relation_ids[name])]))
    if not rows:
        return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long)
    typed = np.unique(np.concatenate(rows).astype(np.int64), axis=0)
    return torch.from_numpy(typed[:, :2].T.copy()), torch.from_numpy(typed[:, 2].copy())
