"""Leakage-explicit split construction and independent random streams for samples."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from utils.preprocessing import labelled_data, sample_negatives, triplet_set
from utils.results import write_json
from utils.runtime import derived_seed


def outer_splits(data, config, seed):
    if config.eval_protocol == "independent_test":
        yield 0, data.positives.copy(), data.external.copy()
        return
    if len(data.positives) < config.n_splits:
        raise ValueError("Not enough unique positive triplets for the requested folds")
    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(data.positives)):
        yield fold, data.positives[train_idx], data.positives[test_idx]


def prepare_fold(data, config, seed, fold, outer_train, test_positive, folder):
    if len(outer_train) < 3:
        raise ValueError("Outer training partition is too small for inner validation")
    train_positive, validation_positive = train_test_split(
        outer_train, test_size=config.validation_fraction, shuffle=True,
        random_state=derived_seed(seed, fold, 1),
    )
    positives = {"train": train_positive, "validation": validation_positive, "test": test_positive}
    sets = {key: triplet_set(value) for key, value in positives.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise AssertionError("Positive partitions overlap after deduplication")
    all_known = sets["train"] | sets["validation"] | sets["test"]
    negatives = {split: {} for split in positives}
    statistics = {}
    # Train-only is the strict external protocol. Global known-positive filtering is
    # optional only in CV, explicitly recorded, and must never be confused with external independence.
    train_exclude = sets["train"] if config.negative_filter_scope == "train_only" else all_known
    for task_index, task in enumerate(config.tasks):
        negatives["train"][task], statistics[f"train_{task}"] = sample_negatives(
            train_positive, data.candidates[task], train_exclude, task, config.neg_ratio,
            np.random.default_rng(derived_seed(seed, fold, 10+task_index)),
        )
    all_train_negative = set().union(*(triplet_set(n) for n in negatives["train"].values()))
    # Inner validation construction does not consult outer/external test positives.
    validation_exclude = sets["train"] | sets["validation"] | all_train_negative
    if config.negative_filter_scope == "all_known":
        validation_exclude |= sets["test"]
    for task_index, task in enumerate(config.tasks):
        negatives["validation"][task], statistics[f"validation_{task}"] = sample_negatives(
            validation_positive, data.candidates[task], validation_exclude, task, config.neg_ratio,
            np.random.default_rng(derived_seed(seed, fold, 20+task_index)),
        )
    all_validation_negative = set().union(*(triplet_set(n) for n in negatives["validation"].values()))
    test_exclude = all_known | all_train_negative | all_validation_negative
    for task_index, task in enumerate(config.tasks):
        negatives["test"][task], statistics[f"test_{task}"] = sample_negatives(
            test_positive, data.candidates[task], test_exclude, task, config.neg_ratio,
            np.random.default_rng(derived_seed(seed, fold, 30+task_index)),
        )
    all_test_negative = set().union(*(triplet_set(n) for n in negatives["test"].values()))
    if all_train_negative & all_validation_negative or all_train_negative & all_test_negative or all_validation_negative & all_test_negative:
        raise AssertionError("Negative triplets overlap between partitions")
    # Do not silently use held-out labels to clean training. Record this PU-label
    # uncertainty instead: an unknown sampled negative may later be a held-out positive.
    collisions = {
        "train_negative_vs_validation_positive": sorted(all_train_negative & sets["validation"]),
        "train_negative_vs_test_positive": sorted(all_train_negative & sets["test"]),
        "validation_negative_vs_test_positive": sorted(all_validation_negative & sets["test"]),
    }
    write_json(folder / "sampling_stats.json", statistics)
    write_json(folder / "heldout_label_collision_audit.json", collisions)
    labelled = {split: {} for split in positives}
    manifest = []
    for split, rows in positives.items():
        manifest.extend({"split": split, "drug_id": int(d), "gene_id": int(g), "disease_id": int(s)} for d,g,s in rows)
        for task in config.tasks:
            labelled[split][task] = labelled_data(rows, negatives[split][task])
            pd.DataFrame(labelled[split][task], columns=["drug_id", "gene_id", "disease_id", "label"]).to_csv(
                folder / f"{split}_{task}_labelled.csv", index=False)
    pd.DataFrame(manifest).to_csv(folder / "positive_split_manifest.csv", index=False)
    return positives, labelled
