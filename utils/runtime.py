"""Device selection and reproducible random-number streams."""

import os
import random

import numpy as np
import torch


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch installation cannot access CUDA")
    return device


def set_global_seed(seed: int, deterministic: bool = True, strict: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    # Some PyG scatter operations have no deterministic CUDA kernel; warn explicitly.
    torch.use_deterministic_algorithms(deterministic, warn_only=not strict)


def derived_seed(seed: int, fold: int, stream: int) -> int:
    return int(np.random.SeedSequence([seed, fold, stream]).generate_state(1)[0])
