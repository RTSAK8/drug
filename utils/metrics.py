"""Classification, raw/filtered ranks, and genuine multi-answer query recall."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .preprocessing import answer_map, make_triplet


def select_threshold(labels: np.ndarray, logits: np.ndarray) -> float:
    """Calibrate on validation data only. Return a threshold in logit space."""
    precision, recall, thresholds = precision_recall_curve(labels, logits)
    if len(thresholds) == 0:
        raise ValueError("Cannot calibrate a threshold on empty validation predictions")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float) -> dict:
    if set(np.unique(labels)) != {0, 1} or not np.isfinite(logits).all():
        raise ValueError("Expected finite scores and both binary classes")
    precision, recall, _ = precision_recall_curve(labels, logits)
    return {
        "auc": float(roc_auc_score(labels, logits)),
        "aupr": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(labels, logits)),
        "f1": float(f1_score(labels, logits >= threshold, zero_division=0)),
        "threshold_logit": float(threshold),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
    }


def _rank(scores: np.ndarray, target_index: int, mask: np.ndarray, tie_policy: str) -> float:
    target = scores[target_index]
    higher = int(np.count_nonzero((scores > target) & mask))
    tied = int(np.count_nonzero((scores == target) & mask))
    if tied < 1:
        raise ValueError("The target must remain in the ranking candidate set")
    if tie_policy == "optimistic":
        return float(higher + 1)
    if tie_policy == "pessimistic":
        return float(higher + tied)
    return float(higher + 1 + (tied - 1) / 2)


@torch.inference_mode()
def predict_logits(model, embeddings: torch.Tensor, triplets: np.ndarray, batch_size: int) -> np.ndarray:
    if model.training:
        raise RuntimeError("Prediction requires model.eval()")
    pieces = []
    for start in range(0, len(triplets), batch_size):
        batch = torch.as_tensor(triplets[start:start+batch_size, :3], dtype=torch.long, device=embeddings.device)
        pieces.append(model.score(embeddings, batch).cpu().numpy())
    if not pieces:
        raise ValueError("Cannot score an empty dataset")
    result = np.concatenate(pieces)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite prediction logits")
    return result


@torch.inference_mode()
def rank_queries(model, embeddings, positives, known_positives, candidates, task, config):
    """Score unique queries in blocks; compute both rank conventions from one score array.

    Per-target filtering excludes all known answers except the current target.
    Multi-answer recall instead retains ALL evaluated answers of a query, and
    removes known positives outside that evaluation answer set. Recall ties use
    ascending global entity ID as a deterministic top-K secondary key.
    """
    if model.training:
        raise RuntimeError("Ranking requires model.eval()")
    evaluated = answer_map(positives, task)
    known = answer_map(known_positives, task)
    index = {int(answer): i for i, answer in enumerate(candidates)}
    keys = sorted(evaluated)
    ranks, recalls = [], []
    for start in range(0, len(keys), config.ranking_query_batch_size):
        group = keys[start:start+config.ranking_query_batch_size]
        score_chunks = []
        for offset in range(0, len(candidates), config.ranking_candidate_batch_size):
            pool = candidates[offset:offset+config.ranking_candidate_batch_size]
            triplets = np.asarray([make_triplet(key, int(a), task) for key in group for a in pool], dtype=np.int64)
            scores = predict_logits(model, embeddings, triplets, config.prediction_batch_size)
            score_chunks.append(scores.reshape(len(group), len(pool)))
        all_scores = np.concatenate(score_chunks, axis=1)
        for row, key in enumerate(group):
            scores = all_scores[row]
            targets = evaluated[key]
            if not targets.issubset(index):
                raise ValueError(f"Target outside candidate vocabulary for {task} query {key}")
            all_mask = np.ones(len(candidates), dtype=bool)
            other_known = known.get(key, set())
            for target in sorted(targets):
                target_index = index[target]
                filtered_mask = all_mask.copy()
                for answer in other_known - {target}:
                    if answer in index:
                        filtered_mask[index[answer]] = False
                ranks.append({
                    "task": task, "query_id_1": key[0], "query_id_2": key[1], "target_id": target,
                    "raw_rank": _rank(scores, target_index, all_mask, config.ranking_tie_policy),
                    "filtered_rank": _rank(scores, target_index, filtered_mask, config.ranking_tie_policy),
                    "raw_candidates": len(candidates), "filtered_candidates": int(filtered_mask.sum()),
                })
            for convention in ("raw", "filtered"):
                keep = all_mask.copy()
                if convention == "filtered":
                    for answer in other_known - targets:
                        if answer in index:
                            keep[index[answer]] = False
                eligible = np.flatnonzero(keep)
                ordering = eligible[np.lexsort((candidates[eligible], -scores[eligible]))]
                for k in config.recall_k:
                    found = len(set(map(int, candidates[ordering[:k]])) & targets)
                    recalls.append({"task": task, "query_id_1": key[0], "query_id_2": key[1],
                                    "convention": convention, "k": k, "num_answers": len(targets),
                                    "retrieved_answers": found, "recall": found / len(targets)})
    rank_frame = pd.DataFrame(ranks)
    recall_frame = pd.DataFrame(recalls)
    if rank_frame.empty:
        raise ValueError("Ranking needs positive evaluation targets")
    metrics = {"ranking_target_count": len(rank_frame), "ranking_query_count": len(keys)}
    for convention in ("raw", "filtered"):
        values = rank_frame[f"{convention}_rank"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 1):
            raise ValueError("Invalid ranks")
        metrics[f"mrr_{convention}"] = float(np.mean(1 / values))
        per_query = rank_frame[["query_id_1", "query_id_2"]].copy()
        per_query["rr"] = 1 / values
        metrics[f"mrr_{convention}_query_macro"] = float(per_query.groupby(["query_id_1", "query_id_2"])["rr"].mean().mean())
        for k in config.hit_k:
            metrics[f"hit@{k}_{convention}"] = float(np.mean(values <= k))
        if 1 in config.hit_k and metrics[f"mrr_{convention}"] + 1e-8 < metrics[f"hit@1_{convention}"]:
            raise AssertionError("MRR must be at least Hits@1")
        for k in config.recall_k:
            mask = (recall_frame["convention"] == convention) & (recall_frame["k"] == k)
            metrics[f"recall@{k}_{convention}_query_macro"] = float(recall_frame.loc[mask, "recall"].mean())
    if np.any(rank_frame.filtered_rank > rank_frame.raw_rank):
        raise AssertionError("Filtered rank cannot be worse than raw rank")
    return metrics, rank_frame, recall_frame
