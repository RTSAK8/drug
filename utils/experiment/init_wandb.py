"""Optional logging; one W&B run per repeat/fold, with no metric overwrites."""


def init_wandb_run(config, run_id: str, seed: int, fold: int):
    if not config.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("use_wandb=true requires wandb; install it or disable logging") from exc
    return wandb.init(project=config.wandb_project, entity=config.wandb_entity or None,
                      name=f"{config.experiment_name}_{seed}_fold{fold}_{run_id}",
                      group=run_id, config={**config.to_dict(), "seed": seed, "fold": fold}, reinit=True)
