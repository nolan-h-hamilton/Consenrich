"""MUNC variance-trend estimation entrypoints."""

from __future__ import annotations

from .core import (
    EB_computePriorStrength,
    EB_computePooledPriorStrength,
    PooledMuncVarianceTrend,
    PSplineLogVarianceTrend,
    evalPSplineLogVarianceTrend,
    fitPooledMuncVarianceTrend,
    fitPSplineLogVarianceTrend,
    getMuncTrack,
)


__all__ = [
    "EB_computePriorStrength",
    "EB_computePooledPriorStrength",
    "PooledMuncVarianceTrend",
    "PSplineLogVarianceTrend",
    "evalPSplineLogVarianceTrend",
    "fitPooledMuncVarianceTrend",
    "fitPSplineLogVarianceTrend",
    "getMuncTrack",
]
