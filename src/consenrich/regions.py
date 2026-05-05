"""Region masks, sparse-region helpers, and length-selection utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .core import (
    _sparseSupportWeights,
    chooseDependenceLength,
    chooseFeatureLength,
    getBedMask,
)


@lru_cache(maxsize=8)
def _readSparseRegionsByChrom(sparseBedFile: str) -> dict[str, np.ndarray]:
    sparseFrame = pd.read_csv(
        sparseBedFile,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64},
        engine="c",
    )
    sparseFrame = sparseFrame[sparseFrame["end"] > sparseFrame["start"]]
    sparseRegionsByChrom: dict[str, np.ndarray] = {}
    for chromName, chromFrame in sparseFrame.groupby("chrom", sort=False):
        chromRegions = chromFrame.loc[:, ["start", "end"]].to_numpy(
            dtype=np.int64,
            copy=True,
        )
        order = np.argsort(chromRegions[:, 0], kind="mergesort")
        sparseRegionsByChrom[str(chromName)] = chromRegions[order, :]
    return sparseRegionsByChrom


def _loadSparseIntervalIndices(
    sparseBedFile: str,
    chromosome: str,
    intervals: np.ndarray,
) -> np.ndarray:
    sparseRegions = _readSparseRegionsByChrom(str(sparseBedFile)).get(
        str(chromosome),
        np.empty((0, 2), dtype=np.int64),
    )
    if sparseRegions.size == 0:
        return np.empty(0, dtype=np.intp)

    intervalStarts = np.asarray(intervals, dtype=np.int64)
    if intervalStarts.size == 0:
        return np.empty(0, dtype=np.intp)
    if intervalStarts.size == 1:
        intervalSize = 1
    else:
        intervalSize = int(intervalStarts[1] - intervalStarts[0])
        if intervalSize <= 0:
            raise ValueError("intervals must be strictly increasing")
    intervalEnds = intervalStarts + int(intervalSize)

    sparseMask = np.zeros(intervalStarts.size, dtype=bool)
    for bedStart, bedEnd in sparseRegions:
        firstIdx = int(np.searchsorted(intervalEnds, int(bedStart), side="right"))
        lastIdx = int(np.searchsorted(intervalStarts, int(bedEnd), side="left"))
        if firstIdx < 0:
            firstIdx = 0
        if lastIdx > intervalStarts.size:
            lastIdx = intervalStarts.size
        if lastIdx > firstIdx:
            sparseMask[firstIdx:lastIdx] = True

    sparseIdx = np.flatnonzero(sparseMask)
    if sparseIdx.size == 0:
        return np.empty(0, dtype=np.intp)
    return sparseIdx.astype(np.intp, copy=False)


__all__ = [
    "_loadSparseIntervalIndices",
    "_readSparseRegionsByChrom",
    "_sparseSupportWeights",
    "chooseDependenceLength",
    "chooseFeatureLength",
    "getBedMask",
]
