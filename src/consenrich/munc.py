"""MUNC variance-trend estimation entrypoints."""

from __future__ import annotations

from .core import (
    EB_computePriorStrength,
    PSplineLogVarianceTrend,
    evalPSplineLogVarianceTrend,
    fitPSplineLogVarianceTrend,
    getMuncTrack,
)


__all__ = [
    "EB_computePriorStrength",
    "PSplineLogVarianceTrend",
    "evalPSplineLogVarianceTrend",
    "fitPSplineLogVarianceTrend",
    "getMuncTrack",
]
