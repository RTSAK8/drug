"""Training losses and label-free deterministic inference are kept separate."""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models.losses import qfal_loss
from .metrics import classification_metrics, predict_logits, rank_queries, select_threshold


def train_step(model, optimizer, features, graph, labelled_tensors, known_answers,
               candidates, config, rng) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    embeddings = model.encode(features, *graph)
    bce_losses, ranking_losses = [], []
    for task in config.tasks:
        if config.bce_weight > 0:
            batch = labelled_tensors[task]
            logits = model.score(embeddings, batch[:, :3])
            bce_losses.append(F.binary_cross_entropy_with_logits(logits, batch[:, 3].to(logits.dtype)))
        if config.qfal_weight > 0:
            ranking_losses.append(qfal_loss(model, embeddings, known_answers[task], candidates[task], task, config, rng))
    zero = embeddings.new_zeros(())
    bce = torch.stack(bce_losses).mean() if bce_losses else zero
    ranking = torch.stack(ranking_losses).mean() if ranking_losses else zero
    loss = config.bce_weight * bce + config.qfal_weight * ranking
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Training loss is not finite")
    loss.backward()
    if config.gradient_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "bce_loss": float(bce.detach()), "qfal_loss": float(ranking.detach())}


@torch.inference_mode()
def evaluate(model, features, graph, labelled, positives, filter_positives, candidates,
             config, thresholds=None):
    """No loss function is called here. Test thresholds must come from validation."""
    model.eval()
    embeddings = model.encode(features, *graph)
    results, artifacts, selected = {}, {}, {}
    for task in config.tasks:
        data = labelled[task]
        logits = predict_logits(model, embeddings, data[:, :3], config.prediction_batch_size)
        threshold = select_threshold(data[:, 3], logits) if thresholds is None else float(thresholds[task])
        selected[task] = threshold
        metrics = classification_metrics(data[:, 3], logits, threshold)
        ranking, ranks, recalls = rank_queries(model, embeddings, positives, filter_positives,
                                              candidates[task], task, config)
        metrics.update(ranking)
        prediction = pd.DataFrame(data, columns=["drug_id", "gene_id", "disease_id", "label"])
        prediction["logit"] = logits
        prediction["probability"] = torch.sigmoid(torch.from_numpy(logits)).numpy()
        prediction["predicted_label"] = (logits >= threshold).astype(np.int64)
        results[task] = metrics
        artifacts[task] = {"predictions": prediction, "ranks": ranks, "query_recall": recalls}
    return results, artifacts, selected
