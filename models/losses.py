"""Paper QFAL plus explicitly selected training-positive exclusion variants."""

from collections import defaultdict

import numpy as np
import torch

from utils.preprocessing import make_triplet


def qfal_loss(model, embeddings: torch.Tensor, known_answers: dict, candidates: np.ndarray,
              task: str, config, rng: np.random.Generator, sampler=None, return_stats=False):
    """Average within each query, then across queries; never accept held-out labels.

    attenuate: paper equations (10)-(14), including attenuation AFTER softmax.
    exclude: sample one target, exclude every other known training answer before
             negative sampling and adversarial normalization.
    multi_positive: score every known answer, average its individual listwise loss
                    against the SAME unlabelled candidates (no positive competition).
    """
    if len(candidates) < 2 or not known_answers:
        raise ValueError(f"QFAL {task} requires queries and >=2 candidate entities")
    if sampler is None:
        keys = sorted(known_answers)
        if config.qfal_max_queries and len(keys) > config.qfal_max_queries:
            keys = [keys[int(i)] for i in rng.choice(len(keys), config.qfal_max_queries, replace=False)]
    else:
        rng = sampler.rng
        keys = sampler.queries(config.qfal_max_queries, config.qfal_query_sampling)
    mode = config.qfal_positive_mode
    rows, segments = [], []
    skipped, negative_total = 0, 0
    for key in keys:
        positives = known_answers[key]
        if mode == "attenuate":
            target = sampler.target(key, config.qfal_query_sampling) if sampler else int(rng.choice(sorted(positives)))
            targets = [target]
            available = candidates[candidates != target]
        else:
            available = candidates[~np.isin(candidates, sorted(positives))]
            if len(available) == 0:
                # No ranking competitor exists. Do not call BN with one dummy row.
                skipped += 1
                continue
            if mode == "multi_positive":
                targets = sampler.all_targets(key) if sampler else sorted(positives)
            else:
                targets = [sampler.target(key, config.qfal_query_sampling) if sampler else int(rng.choice(sorted(positives)))]
        count = min(config.qfal_num_negatives, len(available))
        negatives = rng.choice(available, size=count, replace=False)
        alpha = [config.qfal_false_negative_weight
                 if mode == "attenuate" and config.qfal_known_positive_masking and int(a) in positives else 1.0
                 for a in negatives]
        segments.append((len(rows), len(targets), count, alpha))
        rows.extend(make_triplet(key, int(a), task) for a in [*targets, *negatives])
        negative_total += count
    statistics = {"queries_selected": len(keys), "queries_used": len(segments), "queries_without_negatives": skipped,
                  "candidate_count": negative_total, "scored_triplets": len(rows)}
    if sampler is not None:
        statistics.update(sampler.diagnostics())
    if not segments:
        loss = embeddings.sum() * 0.0
        return (loss, statistics) if return_stats else loss
    batch = torch.as_tensor(rows, device=embeddings.device, dtype=torch.long)
    all_scores = model.score(embeddings, batch)
    groups = defaultdict(list)
    for offset, num_targets, count, alpha in segments:
        groups[(num_targets, count)].append((offset, alpha))
    losses = []
    for (num_targets, count), group in groups.items():
        offsets = torch.tensor([item[0] for item in group], device=embeddings.device)
        indices = offsets[:, None] + torch.arange(num_targets + count, device=embeddings.device)[None, :]
        scores = all_scores[indices]
        positive_scores, negative_scores = scores[:, :num_targets], scores[:, num_targets:]
        weights = scores.new_tensor([item[1] for item in group])
        if config.qfal_self_adversarial:
            with torch.no_grad():
                adversarial = torch.softmax(negative_scores / config.qfal_adversarial_temperature, dim=1) * count
            weights = weights * adversarial
        log_weights = torch.where(weights > 0, weights.clamp_min(torch.finfo(weights.dtype).tiny).log(), -torch.inf)
        targets_scaled = positive_scores / config.qfal_temperature
        negative_terms = negative_scores / config.qfal_temperature + log_weights
        # Include a finite target in EVERY logsumexp. All-masked negatives then
        # produce zero loss and finite gradients, avoiding logsumexp(all -inf).
        denominator = torch.cat([targets_scaled[:, :, None],
                                 negative_terms[:, None, :].expand(-1, num_targets, -1)], dim=2)
        losses.append((torch.logsumexp(denominator, dim=2) - targets_scaled).mean(dim=1))
    loss = torch.cat(losses).mean()
    return (loss, statistics) if return_stats else loss
