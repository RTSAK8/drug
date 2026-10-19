"""One optimizer update per epoch; validation selection and complete resume state."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.hgnn import HGNN
from utils.checkpointing import atomic_torch_save, capture_random_state, restore_random_state, training_config
from utils.preprocessing import answer_map, labelled_data, sample_negatives, triplet_set
from utils.query_sampling import QuerySampler
from utils.results import save_evaluation, write_json
from utils.runtime import derived_seed, set_global_seed
from utils.training import evaluate, train_step

LOG = logging.getLogger(__name__)
CHECKPOINT_VERSION = 2


def refresh_negatives(config, train_positive, candidates, seed, fold, epoch, device):
    """Only training positives and a label-independent reservation enter this path."""
    tensors, statistics = {}, {}
    for task in config.tasks:
        task_id = {"gene": 0, "disease": 1}[task]
        rng = np.random.default_rng(derived_seed(seed, fold, 1000 + 2 * epoch + task_id))
        negatives, statistics[task] = sample_negatives(
            train_positive, candidates[task], triplet_set(train_positive), task, config.neg_ratio, rng,
            partition="train", partition_seed=derived_seed(seed, fold, 40),
        )
        tensors[task] = torch.as_tensor(labelled_data(train_positive, negatives), dtype=torch.long, device=device)
    return tensors, statistics


def selection_score(metrics: dict, config) -> float:
    values = [metrics[task][config.selection_metric] for task in config.tasks]
    weights = [config.task_weights[task] for task in config.tasks]
    score = float(np.average(values, weights=weights))
    if not np.isfinite(score):
        raise FloatingPointError("Validation selection metric is not finite")
    return score


def fit_model(config, features, graph, train_positive, validation_positive,
              train_labelled, validation_labelled, candidates, folder: Path,
              seed: int, fold: int, wandb_run=None):
    # Test labels, test candidates selected for classification, and test positives
    # are deliberately absent from this API.
    model_seed = derived_seed(seed, fold, 50)
    set_global_seed(model_seed, config.deterministic, config.deterministic_strict)
    device = next(iter(features.values())).device
    model = HGNN(config, {k: int(v.shape[1]) for k, v in features.items()}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.scheduler_factor, patience=config.scheduler_patience,
            threshold=config.early_stopping_min_delta, threshold_mode="abs", min_lr=config.min_learning_rate,
        )
    train_tensors = {task: torch.as_tensor(rows, dtype=torch.long, device=device) for task, rows in train_labelled.items()}
    known = {task: answer_map(train_positive, task) for task in config.tasks}
    validation_filter = np.concatenate([train_positive, validation_positive])
    rng = np.random.default_rng(derived_seed(seed, fold, 51))
    samplers = {task: QuerySampler(known[task], np.random.default_rng(
        derived_seed(seed, fold, 60 + {"gene": 0, "disease": 1}[task]))) for task in config.tasks}
    best_score, best_epoch, stale = -float("inf"), 0, 0
    history, refresh_history = [], []
    checkpoint_path, last_path = folder / "best_model.pt", folder / "last_checkpoint.pt"
    best_payload, completed_epoch = None, 0
    if config.resume_from and last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state.get("version") != CHECKPOINT_VERSION or state["seed"] != seed or state["fold"] != fold:
            raise ValueError("Unsupported or mismatched resume checkpoint")
        if training_config(state["config"]) != training_config(config.to_dict()):
            raise ValueError("Training configuration differs from the resume checkpoint")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        for task, sampler in samplers.items():
            sampler.load_state_dict(state["samplers"][task])
        rng.bit_generator.state = state["generator_state"]
        train_tensors = {task: rows.to(device) for task, rows in state["train_labelled"].items()}
        best_payload, best_score = state["best_payload"], state["best_score"]
        best_epoch, stale, completed_epoch = state["best_epoch"], state["stale"], state["epoch"]
        history, refresh_history = state["history"], state["refresh_history"]
        # last_checkpoint is the single commit point; recover the matching best
        # snapshot even if interruption occurred between the two atomic saves.
        atomic_torch_save(checkpoint_path, best_payload)
        restore_random_state(state["random_state"])
        del state
        LOG.info("Resuming seed=%d fold=%d after update %d", seed, fold, completed_epoch)

    for epoch in range(completed_epoch + 1, config.epochs + 1):
        if stale >= config.early_stopping_patience:
            break
        if config.bce_weight > 0 and config.negative_refresh_every and epoch > 1 and (epoch - 1) % config.negative_refresh_every == 0:
            train_tensors, refresh_stats = refresh_negatives(config, train_positive, candidates, seed, fold, epoch, device)
            refresh_history.append({"epoch": epoch, "tasks": refresh_stats})
        current_lr = float(optimizer.param_groups[0]["lr"])
        training = train_step(model, optimizer, features, graph, train_tensors, known, candidates, config, rng, samplers)
        record = {"epoch": epoch, "optimizer_updates": epoch, "learning_rate": current_lr, **training}
        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            validation, _, _ = evaluate(model, features, graph, validation_labelled, validation_positive,
                                        validation_filter, candidates, config)
            score = selection_score(validation, config)
            record["validation_selection_score"] = score
            for task, metrics in validation.items():
                record.update({f"validation/{task}/{key}": value for key, value in metrics.items()})
            if score > best_score + config.early_stopping_min_delta:
                best_score, best_epoch, stale = score, epoch, 0
                best_payload = {
                    "version": CHECKPOINT_VERSION,
                    "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "epoch": epoch, "selection_score": score, "selection_metric": config.selection_metric,
                    "task_weights": config.task_weights, "config": config.to_dict(),
                    "seed": seed, "model_seed": model_seed, "fold": fold,
                }
                atomic_torch_save(checkpoint_path, best_payload)
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step(score)
            record["next_learning_rate"] = float(optimizer.param_groups[0]["lr"])
            LOG.info("Update %d loss=%.5f val_%s=%.5f best=%d patience=%d/%d",
                     epoch, training["loss"], config.selection_metric, score, best_epoch, stale, config.early_stopping_patience)
        history.append(record)
        completed_epoch = epoch
        if wandb_run is not None:
            wandb_run.log(record, step=epoch)
        if epoch == 1 or epoch % config.checkpoint_every == 0 or epoch == config.epochs or stale >= config.early_stopping_patience:
            atomic_torch_save(last_path, {
                "version": CHECKPOINT_VERSION, "config": config.to_dict(), "seed": seed, "fold": fold,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                "best_payload": best_payload, "history": history, "refresh_history": refresh_history,
                "samplers": {task: sampler.state_dict() for task, sampler in samplers.items()},
                "generator_state": rng.bit_generator.state, "random_state": capture_random_state(),
                "train_labelled": {task: rows.detach().cpu() for task, rows in train_tensors.items()},
            })
            pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    if best_payload is None:
        raise RuntimeError("Training finished without a finite validation checkpoint")
    pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
    write_json(folder / "negative_refresh_history.json", refresh_history)
    write_json(folder / "query_coverage.json", {task: sampler.diagnostics() for task, sampler in samplers.items()})
    model.load_state_dict(best_payload["model_state"])
    model.eval()
    validation, artifacts, thresholds = evaluate(model, features, graph, validation_labelled,
                                                 validation_positive, validation_filter, candidates, config)
    save_evaluation(folder, "validation", validation, artifacts)
    write_json(folder / "thresholds.json", {"source": "inner_validation_best_f1", "logit_thresholds": thresholds})
    best_payload["thresholds_logit"] = thresholds
    atomic_torch_save(checkpoint_path, best_payload)
    write_json(folder / "training_summary.json", {
        "best_epoch": best_epoch, "epochs_completed": completed_epoch, "optimizer_updates": completed_epoch,
        "best_validation_score": best_score, "model_seed": model_seed, "selection_metric": config.selection_metric,
        "task_weights": config.task_weights, "early_stopped": completed_epoch < config.epochs,
    })
    return model, thresholds
