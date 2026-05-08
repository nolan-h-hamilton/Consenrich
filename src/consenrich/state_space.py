"""State-space construction and fitting entrypoints."""

from __future__ import annotations

from .core import (
    constructMatrixF,
    constructMatrixQ,
    getPrimaryState,
    runConsenrich,
)


__all__ = [
    "constructMatrixF",
    "constructMatrixQ",
    "getPrimaryState",
    "runConsenrich",
]
