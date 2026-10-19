"""Public model API; obsolete encoder/loss exports are intentionally absent."""

__all__ = ["HGNN"]


def __getattr__(name):
    if name == "HGNN":
        from .hgnn import HGNN

        return HGNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
