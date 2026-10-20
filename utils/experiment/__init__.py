"""Experiment API with lazy exports; importing the package does not load training."""

__all__ = ["run_experiment"]


def __getattr__(name):
    if name == "run_experiment":
        from .runner import run_experiment

        return run_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
