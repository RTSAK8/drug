"""
Runtime helpers for consistent device and seed management.
"""

import random

import numpy as np
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    """Return the shared runtime device used across modules."""
    return DEVICE


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
