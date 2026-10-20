"""Atomic, weights_only-compatible checkpoints and resumable random states."""

from contextlib import contextmanager
import os
from pathlib import Path
import random

import numpy as np
import torch


def atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_random_state() -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def training_config(config: dict) -> dict:
    # Device, epochs and data paths deliberately remain strict for honest resume.
    ignored = {"resume_from", "results_dir", "use_wandb", "wandb_project", "wandb_entity"}
    return {key: value for key, value in config.items() if key not in ignored}


@contextmanager
def run_lock(root: Path):
    """OS lock released on process exit, including crashes; the file may remain."""
    with (root / ".run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This experiment directory is already in use") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("This experiment directory is already in use") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
