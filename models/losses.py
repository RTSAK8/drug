"""Paper equations (10)-(14): query-macro QFAL, using training positives only."""

import numpy as np
import torch

from utils.preprocessing import make_triplet


def qfal_loss(model, embeddings: torch.Tensor, known_answers: dict, candidates: np.ndarray,
              task: str, config, rng: np.random.Generator) -> torch.Tensor:
    """Sample unique queries, one target per query, and distinct non-target candidates.

    known_answers is built once from the entire training partition BEFORE query
    subsampling, so down-weighting never forgets an answer outside the minibatch.
    Averaging repeated target draws estimates the mean over targets within query.
    Self-adversarial weights are detached and use raw logits, as in equation (13).
    """
    if len(candidates) < 2 or not known_answers:
        raise ValueError(f"QFAL {task} requires queries and >=2 candidate entities")
    keys = sorted(known_answers)
    if config.qfal_max_queries and len(keys) > config.qfal_max_queries:
        selected = rng.choice(len(keys), config.qfal_max_queries, replace=False)
        keys = [keys[int(i)] for i in selected]
    count = min(config.qfal_num_negatives, len(candidates) - 1)
    rows, alpha_rows = [], []
    for key in keys:
        positives = known_answers[key]
        target = int(rng.choice(sorted(positives)))
        available = candidates[candidates != target]
        negatives = rng.choice(available, size=count, replace=False)
        rows.extend(make_triplet(key, int(answer), task) for answer in [target, *negatives])
        alpha_rows.append([
            config.qfal_false_negative_weight if config.qfal_known_positive_masking and int(a) in positives else 1.0
            for a in negatives
        ])
    batch = torch.as_tensor(rows, device=embeddings.device, dtype=torch.long)
    scores = model.score(embeddings, batch).reshape(len(keys), count + 1)
    weights = torch.as_tensor(alpha_rows, device=scores.device, dtype=scores.dtype)
    if config.qfal_self_adversarial:
        with torch.no_grad():
            adversarial = torch.softmax(scores[:, 1:] / config.qfal_adversarial_temperature, dim=1) * count
        weights = weights * adversarial
    # epsilon=0 means exactly excluded, not clamp_min(1e-8).
    log_weights = torch.where(weights > 0, torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny)), -torch.inf)
    scaled = scores / config.qfal_temperature
    denominator_terms = torch.cat([scaled[:, :1], scaled[:, 1:] + log_weights], dim=1)
    return (torch.logsumexp(denominator_terms, dim=1) - scaled[:, 0]).mean()
