"""Hydra entry point; load the training stack only when an experiment starts."""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("This project requires Python >= 3.10; check the active conda environment.")

import hydra
from omegaconf import DictConfig

from config import Config


@hydra.main(version_base="1.3", config_path="conf", config_name="default")
def main(cfg: DictConfig):
    config = Config.from_omegaconf(cfg)
    # Import the implementation explicitly. Do not depend on an older package
    # __init__.py re-export; --cfg/--help must not import torch/PyG or load data.
    from utils.experiment.runner import run_experiment

    run_experiment(config)


if __name__ == "__main__":
    main()
