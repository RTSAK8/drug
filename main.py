import hydra
from omegaconf import DictConfig

from config import Config
from utils.experiment import run_experiment


@hydra.main(version_base=None, config_path="conf", config_name="new_loss")
def main(cfg: DictConfig):
    config = Config.from_omegaconf(cfg)
    run_experiment(config)


if __name__ == "__main__":
    main()
