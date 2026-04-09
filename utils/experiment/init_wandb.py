import warnings

from config import Config

try:
    import wandb
except ImportError:
    wandb = None


def _wandb_enabled(config: Config) -> bool:
    """仅按配置决定是否启用wandb。"""
    return bool(getattr(config, "use_wandb", False))


def init_wandb_run(config: Config, data_manager, ablation_count: int):
    """创建单个wandb run；在同一project内按mode前缀分栏记录。"""
    if not _wandb_enabled(config):
        return None
    if wandb is None:
        warnings.warn("检测到已启用wandb，但当前环境未安装wandb，已跳过日志记录。")
        return None

    project = getattr(config, "wandb_project", "protein-hgnn")
    entity = getattr(config, "wandb_entity", None)
    wandb_name_prefix = str(getattr(config, "wandb_name_prefix", "XXX"))
    wandb_run_index = int(getattr(config, "wandb_run_index", 1))
    run_name = f"{wandb_name_prefix}_{wandb_run_index}"

    run = wandb.init(
        project=project,
        entity=entity,
        group="mode_comparison",
        name=run_name,
        config=vars(config),
        reinit=True,
    )

    # Keep global monotonic _step for logging safety, but render curves with local epoch per mode.
    for mode_key in ("gene", "disease"):
        run.define_metric(f"{mode_key}/train_epoch", hidden=True)
        run.define_metric(f"{mode_key}/train_total_loss", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_main_loss", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_cl_loss", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_mrr", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_hit@1", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_hit@3", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/train_hit@5", step_metric=f"{mode_key}/train_epoch")
        run.define_metric(f"{mode_key}/val_epoch", hidden=True)
        run.define_metric(f"{mode_key}/val/*", step_metric=f"{mode_key}/val_epoch")

    # run.summary["n_ablations"] = int(ablation_count)
    run.summary["num_drugs"] = int(len(data_manager.drug_to_idx))
    run.summary["num_genes"] = int(len(data_manager.gene_to_idx))
    run.summary["num_diseases"] = int(len(data_manager.disease_to_idx))
    return run
