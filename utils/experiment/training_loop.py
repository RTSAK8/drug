"""Fit on training data; select/checkpoint/calibrate exclusively on inner validation."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import HGNN
from utils.preprocessing import answer_map
from utils.results import save_evaluation, write_json
from utils.runtime import derived_seed, set_global_seed
from utils.training import evaluate, train_step

LOG = logging.getLogger(__name__)


def fit_model(config, features, graph, train_positive, validation_positive,
              train_labelled, validation_labelled, candidates, folder: Path,
              seed: int, fold: int, wandb_run=None):
    # No test triplets or test labels are accepted by this function.
    model_seed = derived_seed(seed, fold, 50)
    set_global_seed(model_seed, config.deterministic)
    model = HGNN(config, {k: int(v.shape[1]) for k, v in features.items()}).to(next(iter(features.values())).device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    device = next(model.parameters()).device
    train_tensors = {task: torch.as_tensor(rows, dtype=torch.long, device=device) for task, rows in train_labelled.items()}
    known = {task: answer_map(train_positive, task) for task in config.tasks}
    validation_filter = np.concatenate([train_positive, validation_positive])
    rng = np.random.default_rng(derived_seed(seed, fold, 51))
    best_score, best_epoch, stale = -float("inf"), 0, 0
    history = []
    checkpoint_path = folder / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        training = train_step(model, optimizer, features, graph, train_tensors, known, candidates, config, rng)
        record = {"epoch": epoch, **training}
        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            validation, _, _ = evaluate(model, features, graph, validation_labelled, validation_positive,
                                        validation_filter, candidates, config)
            score = float(np.mean([validation[t]["mrr_filtered_query_macro"] for t in config.tasks]))
            if not np.isfinite(score):
                raise FloatingPointError("Validation selection metric is not finite")
            record["validation_selection_mrr"] = score
            for task, metrics in validation.items():
                record.update({f"validation/{task}/{key}": value for key, value in metrics.items()})
            if score > best_score + config.early_stopping_min_delta:
                best_score, best_epoch, stale = score, epoch, 0
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "selection_score": score,
                            "selection_metric": "mean_task_filtered_query_macro_mrr", "config": config.to_dict(),
                            "seed": seed, "model_seed": model_seed, "fold": fold}, temporary)
                temporary.replace(checkpoint_path)
            else:
                stale += 1
            LOG.info("Epoch %d loss=%.5f val_macro_MRR=%.5f best_epoch=%d patience=%d/%d",
                     epoch, training["loss"], score, best_epoch, stale, config.early_stopping_patience)
        history.append(record)
        if wandb_run is not None:
            wandb_run.log(record, step=epoch)
        if "validation_selection_mrr" in record:
            pd.DataFrame(history).to_csv(folder / "training_history.csv", index=False)
        if stale >= config.early_stopping_patience:
            break
    if not checkpoint_path.is_file():
        raise RuntimeError("Training finished without a finite validation checkpoint")
    # Load only a checkpoint created by this run. No arbitrary pickle objects are needed.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    validation, artifacts, thresholds = evaluate(model, features, graph, validation_labelled,
                                                 validation_positive, validation_filter, candidates, config)
    save_evaluation(folder, "validation", validation, artifacts)
    write_json(folder / "thresholds.json", {"source": "inner_validation_best_f1", "logit_thresholds": thresholds})
    checkpoint["thresholds_logit"] = thresholds
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    write_json(folder / "training_summary.json", {"best_epoch": best_epoch, "epochs_completed": epoch,
               "best_validation_mrr": best_score, "model_seed": model_seed,
               "selection_metric": "mean_task_filtered_query_macro_mrr", "early_stopped": epoch < config.epochs})
    return model, thresholds
