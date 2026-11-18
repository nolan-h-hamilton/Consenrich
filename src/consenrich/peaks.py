# -*- coding: utf-8 -*-
r"""Peak-calling helpers for ROCCO segmentation from Consenrich tracks."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from ccounts import assemblyBlacklistRanges
from scipy import stats

from . import cconsenrich
from . import _normalization as normalization

from .constants import (
    EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER,
    MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER,
    MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z,
    MATCHING_DEFAULT_GAMMA,
    MATCHING_DEFAULT_MAX_REGION_BP,
    MATCHING_DEFAULT_MERGE_TOLERANCE_BP,
    MATCHING_DEFAULT_MIN_MEAN_SIGNAL,
    MATCHING_DEFAULT_NUM_REGION_REPLAYS,
    MATCHING_DEFAULT_PEAK_MODE,
    MATCHING_DEFAULT_SELECTION_PENALTY_MULTIPLIER,
    MATCHING_DEFAULT_USE_LOCAL_BOOTSTRAP_RADIUS,
    MATCHING_PEAK_MODES,
    MATCHING_SUPPORTED_UNCERTAINTY_SCORE_MODES,
    NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    NESTED_ROCCO_ITERS_DEFAULT,
    NESTED_ROCCO_JACCARD_DEFAULT,
    NESTED_ROCCO_MIN_CHILD_STEPS,
    NESTED_ROCCO_MIN_PARENT_STEPS,
    OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES,
    OUTPUT_DEFAULT_PLOT_NULL_CALIBRATION_DIAGNOSTICS,
    ROCCO_BUDGET_MAX,
    ROCCO_BUDGET_MIN,
    ROCCO_MAX_ITER_DEFAULT,
    ROCCO_NUM_BOOTSTRAP_DEFAULT,
    ROCCO_THRESHOLD_Z_DEFAULT,
)

logger = logging.getLogger(__name__)

_TINY = float(np.finfo(np.float64).tiny)
_ROCCO_BUDGET_MIN = ROCCO_BUDGET_MIN
_ROCCO_BUDGET_MAX = ROCCO_BUDGET_MAX
_ROCCO_THRESHOLD_Z_DEFAULT = ROCCO_THRESHOLD_Z_DEFAULT
_ROCCO_NUM_BOOTSTRAP_DEFAULT = ROCCO_NUM_BOOTSTRAP_DEFAULT
_ROCCO_MAX_ITER_DEFAULT = ROCCO_MAX_ITER_DEFAULT
_MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE = MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE
_MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z = MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z
_MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z = MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z
_MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER = (
    MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER
)
_MATCHING_DEFAULT_PEAK_MODE = MATCHING_DEFAULT_PEAK_MODE
_OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES = OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES
_OUTPUT_DEFAULT_PLOT_NULL_CALIBRATION_DIAGNOSTICS = (
    OUTPUT_DEFAULT_PLOT_NULL_CALIBRATION_DIAGNOSTICS
)
_NESTED_ROCCO_ITERS_DEFAULT = NESTED_ROCCO_ITERS_DEFAULT
_NESTED_ROCCO_JACCARD_DEFAULT = NESTED_ROCCO_JACCARD_DEFAULT
_NESTED_ROCCO_MIN_PARENT_STEPS = NESTED_ROCCO_MIN_PARENT_STEPS
_NESTED_ROCCO_MIN_CHILD_STEPS = NESTED_ROCCO_MIN_CHILD_STEPS
_NESTED_ROCCO_BUDGET_SCALE_DEFAULT = NESTED_ROCCO_BUDGET_SCALE_DEFAULT
_NESTED_ROCCO_PARENT_EDGE_COST = 1.0e-12
_EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER = (
    EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
)
_NULL_REPLAY_MAX_SEGMENTS = 20000
_NULL_REPLAY_MAX_SEGMENTS_PER_VIEW = 1000
_NULL_REPLAY_THREADS = 4
_ROCCO_BUDGET_SHRINKAGE_MIN_CHROMOSOMES = 4
_ROCCO_TAIL_VARIANCE_BLOCK_SPANS = 8
_ROCCO_BUDGET_BLOCK_TAIL_BLOCKS = 200
_ROCCO_BUDGET_BLOCK_MIN_BP = 40_000_000
_ROCCO_SELECTION_PENALTY_MIN = 0.01
_ROCCO_LOCAL_BOOTSTRAP_RADIUS_LIMIT_BP = 1_000_000
_ROCCO_DIAGNOSTIC_MAX_BOOTSTRAP_SAMPLES_PER_BLOCK = 512


class peakArtifacts(NamedTuple):
    narrowPeak: str | None
    gappedPeak: str | None
    metadata: str | None
    nullCalibrationDiagnostics: str | None = None


class _peakRecord(NamedTuple):
    chromosome: str
    startBP: int
    endBP: int
    summitBP: int
    family: Literal["narrow", "broad"]
    rawScore: float
    signalValue: float
    pValue: float
    qValue: float
    blocks: tuple[tuple[int, int], ...]


class _regionalSignalData(NamedTuple):
    coveredBPPrefix: np.ndarray
    signalMass: np.ndarray


class _candidateReplayData(NamedTuple):
    thresholdViews: Mapping[str, Any]
    commonCandidateStats: tuple[tuple[tuple[int, int], float], ...]
    nullStatsByDraw: tuple[np.ndarray, ...]
    pooledNullStats: np.ndarray
    nullCandidateCounts: tuple[int, ...]


def _logRoccoProgress(
    message: str,
    *args: Any,
    fields: Mapping[str, Any] | None = None,
) -> None:
    extra: Dict[str, Any] = {
        "consenrich_console_subphase": True,
        "consenrich_console_progress": True,
        "consenrich_event": "rocco.progress",
    }
    if fields:
        extra["consenrich_fields"] = dict(fields)
    logger.info(message, *args, extra=extra, stacklevel=2)


def _asFloatVector(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"`{name}` must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` contains non-finite values")
    return arr


def _validateExportFilterUncertaintyMultiplier(value: float) -> float:
    value_ = float(value)
    if not np.isfinite(value_) or value_ < 0.0:
        raise ValueError(
            "`exportFilterUncertaintyMultiplier` must be finite and non-negative"
        )
    return value_


def _validateMinMeanSignal(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("`minMeanSignal` must be a finite number or None")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("`minMeanSignal` must be a finite number or None") from exc
    if not np.isfinite(score):
        raise ValueError("`minMeanSignal` must be a finite number or None")
    return score


def _validateBroadWeakThresholdZ(value: float) -> float:
    thresholdZ = float(value)
    if not np.isfinite(thresholdZ) or thresholdZ < 0.0:
        raise ValueError("`broadWeakThresholdZ` must be finite and non-negative")
    return thresholdZ


def _validateBroadSize(name: str, value: int | None, required: bool) -> int | None:
    if value is None:
        if required:
            raise ValueError(f"`{name}` must be a positive integer for broad peaks")
        return None
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be a positive integer or None")
    sizeBP = int(value)
    if sizeBP != value or sizeBP <= 0:
        raise ValueError(f"`{name}` must be a positive integer or None")
    return sizeBP


def _validateDurationBP(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be a positive integer")
    durationBP = int(value)
    if durationBP != value or durationBP <= 0:
        raise ValueError(f"`{name}` must be a positive integer")
    return durationBP


def _validateNumRegionReplays(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("`numRegionReplays` must be a positive integer")
    numReplays = int(value)
    if numReplays != value or numReplays <= 0:
        raise ValueError("`numRegionReplays` must be a positive integer")
    return numReplays


def _readBlacklistIntervalsByChrom(
    blacklistBedFile: str | None, genomeName: str | None = None
) -> Dict[str, np.ndarray]:
    if blacklistBedFile is None:
        if genomeName is None:
            return {}
        try:
            chromosomeRanges = assemblyBlacklistRanges[genomeName]
        except KeyError as error:
            raise ValueError(
                f"No blacklist ranges available for assembly {genomeName!r}"
            ) from error
        return {
            chromosome: np.asarray(ranges, dtype=np.int64)
            for chromosome, ranges in chromosomeRanges.items()
        }
    path = Path(blacklistBedFile)
    if not path.exists():
        raise ValueError(f"Could not find blacklist BED file {blacklistBedFile}")
    intervalsByChrom: Dict[str, List[Tuple[int, int]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue
            chrom = str(parts[0])
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            intervalsByChrom.setdefault(chrom, []).append((start, end))

    mergedByChrom: Dict[str, np.ndarray] = {}
    for chrom, rows in intervalsByChrom.items():
        rows.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in rows:
            if not merged or start > merged[-1][1]:
                merged.append((int(start), int(end)))
            else:
                prevStart, prevEnd = merged[-1]
                merged[-1] = (prevStart, max(prevEnd, int(end)))
        mergedByChrom[chrom] = np.asarray(merged, dtype=np.int64)
    return mergedByChrom


def _intervalOverlapsBlacklist(
    chromosome: str,
    start: int,
    end: int,
    blacklistByChrom: Mapping[str, np.ndarray],
) -> bool:
    intervals = blacklistByChrom.get(str(chromosome))
    if intervals is None or intervals.size == 0:
        return False
    start_ = int(start)
    end_ = int(end)
    if end_ <= start_:
        return False
    idx = int(np.searchsorted(intervals[:, 0], end_, side="left"))
    if idx <= 0:
        return False
    return bool(int(intervals[idx - 1, 1]) > start_)


def _thresholdZKey(thresholdZ: float) -> str:
    z = float(thresholdZ)
    return repr(0.0 if z == 0.0 else z)


def _resolveThresholdZGrid(
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    thresholdZGrid: Iterable[float] | None = None,
) -> List[float]:
    values: List[float] = []
    seen: set[str] = set()
    for value in [
        thresholdZ,
        *(() if thresholdZGrid is None else thresholdZGrid),
    ]:
        if isinstance(value, bool):
            raise ValueError("threshold z values must be finite and non-negative")
        value_ = float(value)
        if not np.isfinite(value_) or value_ < 0.0:
            raise ValueError("threshold z values must be finite and non-negative")
        key = _thresholdZKey(value_)
        if key not in seen:
            seen.add(key)
            values.append(value_)
    values.sort()
    return values


def _halfSampleMode(sortedValues: np.ndarray) -> float:
    vals = np.asarray(sortedValues, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return 0.0
    if n == 1:
        return float(vals[0])
    if n == 2:
        return float(np.mean(vals))
    if n == 3:
        leftWidth = float(vals[1] - vals[0])
        rightWidth = float(vals[2] - vals[1])
        return (
            float(np.mean(vals[:2]))
            if leftWidth <= rightWidth
            else float(np.mean(vals[1:]))
        )

    window = int(math.ceil(n / 2))
    bestStart = 0
    bestWidth = float(vals[window - 1] - vals[0])
    for start in range(1, n - window + 1):
        width = float(vals[start + window - 1] - vals[start])
        if width < bestWidth:
            bestWidth = width
            bestStart = start
    return _halfSampleMode(vals[bestStart : bestStart + window])


def consenrichStateScoreTrack(
    state: npt.ArrayLike,
    uncertainty: npt.ArrayLike | None = None,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
) -> np.ndarray:
    r"""Build the Consenrich-derived ROCCO score track."""
    state_ = _asFloatVector("state", state)
    mode = normalization.validateChoice(
        uncertaintyScoreMode,
        default=_MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
        supported=MATCHING_SUPPORTED_UNCERTAINTY_SCORE_MODES,
        paramName="uncertaintyScoreMode",
    )
    z_ = normalization.validateFiniteNonnegativeFloat(
        uncertaintyScoreZ,
        paramName="uncertaintyScoreZ",
    )
    if uncertainty is not None:
        uncertainty_ = _asFloatVector("uncertainty", uncertainty)
        if uncertainty_.size != state_.size:
            raise ValueError("`uncertainty` must match `state` length")
    elif mode == "lower_confidence":
        raise ValueError(
            "`lower_confidence` uncertaintyScoreMode requires `uncertainty`"
        )
    if mode != "lower_confidence":
        return state_
    if np.any(uncertainty_ < 0.0):
        raise ValueError("`uncertainty` must be non-negative for lower_confidence")
    rawScore = state_ - z_ * uncertainty_
    maxState = float(np.max(state_))
    if np.isfinite(maxState) and maxState > 0.0:
        return np.maximum(rawScore, -2.0 * maxState)
    return rawScore


def _selectRobustNullSupport(
    values: np.ndarray,
    bulkQuantile: float = 0.60,
) -> Tuple[np.ndarray, Dict[str, float | str | int]]:
    z = _asFloatVector("values", values)
    n = int(z.size)
    bulkQuantile_ = float(np.clip(bulkQuantile, 0.05, 0.95))
    minSupport = max(16, int(math.ceil(0.05 * n)))

    cutoff = float(
        np.quantile(
            z,
            bulkQuantile_,
            method="interpolated_inverted_cdf",
        )
    )
    lowerBulk = z[z <= cutoff]
    bulkSource = "lower_bulk"
    if lowerBulk.size < minSupport:
        lowerBulk = z
        bulkSource = "full_track"

    bulkSorted = np.sort(np.asarray(lowerBulk, dtype=np.float64))
    provisionalCenter = float(
        _halfSampleMode(bulkSorted) if bulkSorted.size >= 4 else np.median(bulkSorted)
    )
    centerMethod = (
        f"{bulkSource}_half_sample_mode"
        if bulkSorted.size >= 4
        else f"{bulkSource}_median"
    )
    bulkResiduals = bulkSorted - provisionalCenter
    bulkMad = 1.4826 * float(
        np.median(np.abs(bulkResiduals - np.median(bulkResiduals)))
    )
    bulkIqr = (
        float(stats.iqr(bulkResiduals, rng=(25, 75))) / 1.349
        if bulkResiduals.size >= 4
        else 0.0
    )
    bulkStd = float(np.std(bulkResiduals, ddof=1)) if bulkResiduals.size >= 2 else 0.0
    supportScale = float(max(bulkMad, bulkIqr, bulkStd, 1.0e-6))
    supportRadius = float(
        max(
            2.5 * supportScale,
            (
                float(
                    np.quantile(
                        np.abs(bulkResiduals),
                        0.50,
                        method="interpolated_inverted_cdf",
                    )
                )
                if bulkResiduals.size >= 4
                else supportScale
            ),
            1.0e-6,
        )
    )
    support = z[np.abs(z - provisionalCenter) <= supportRadius]
    if support.size >= minSupport:
        method = "mode_centered_central_support"
    else:
        order = np.argsort(np.abs(z - provisionalCenter))
        support = z[np.asarray(order[:minSupport], dtype=np.int64)]
        method = "mode_centered_nearest_support"

    details: Dict[str, float | str | int] = {
        "null_method": str(method),
        "support_size": int(support.size),
        "lower_bulk_size": int(lowerBulk.size),
        "bulk_quantile": float(bulkQuantile_),
        "bulk_cutoff": float(cutoff),
        "provisional_center": float(provisionalCenter),
        "center_method": str(centerMethod),
        "support_radius": float(supportRadius),
        "support_scale": float(supportScale),
    }
    return np.asarray(support, dtype=np.float64), details


def estimateROCCONull(
    scoreTrack: npt.ArrayLike,
    bulkQuantile: float = 0.60,
) -> Tuple[float, float, Dict[str, float | str]]:
    r"""Estimate a robust null center and scale from mode-centered central support."""
    nullCenter, nullScale, details, _lowerMagnitudes = _estimateROCCONull(
        scoreTrack,
        bulkQuantile=bulkQuantile,
    )
    return nullCenter, nullScale, details


def _estimateROCCONull(
    scoreTrack: npt.ArrayLike,
    bulkQuantile: float = 0.60,
) -> Tuple[float, float, Dict[str, float | str], np.ndarray]:
    z = _asFloatVector("scoreTrack", scoreTrack)
    _support, supportMeta = _selectRobustNullSupport(
        z,
        bulkQuantile=bulkQuantile,
    )

    nullCenter = float(supportMeta["provisional_center"])
    centerScale = float(max(float(supportMeta["support_scale"]), 1.0e-6))
    for _iteration in range(20):
        standardized = (z - nullCenter) / (3.0 * centerScale)
        centerMask = np.abs(standardized) < 1.0
        if np.count_nonzero(centerMask) < 8:
            raise ValueError("ROCCO null center requires eight central residuals")
        centerWeights = (1.0 - standardized[centerMask] ** 2) ** 2
        nextCenter = float(
            np.dot(centerWeights, z[centerMask]) / np.sum(centerWeights)
        )
        if abs(nextCenter - nullCenter) <= 1.0e-10 * max(centerScale, 1.0):
            nullCenter = nextCenter
            break
        nullCenter = nextCenter
    lowerMagnitudes = nullCenter - z[z < nullCenter]
    if lowerMagnitudes.size < 8:
        raise ValueError("ROCCO null estimation requires eight lower residuals")
    nullScale = float(
        np.median(lowerMagnitudes) / float(stats.norm.ppf(0.75))
    )
    if nullScale <= _TINY:
        raise ValueError("ROCCO lower-residual scale must be positive")
    scaleMethod = "lower_residual_median"

    details: Dict[str, float | str] = {
        "null_method": str(supportMeta["null_method"]),
        "scale_method": str(scaleMethod),
        "support_size": int(supportMeta["support_size"]),
        "lower_bulk_size": int(supportMeta["lower_bulk_size"]),
        "bulk_quantile": float(supportMeta["bulk_quantile"]),
        "bulk_cutoff": float(supportMeta["bulk_cutoff"]),
        "provisional_center": float(supportMeta["provisional_center"]),
        "center_method": "lower_bulk_half_sample_mode_tukey_biweight",
        "support_radius": float(supportMeta["support_radius"]),
        "support_scale": float(supportMeta["support_scale"]),
        "null_center": float(nullCenter),
        "null_scale": float(nullScale),
        "null_center_shift_from_zero": float(nullCenter),
        "support_fraction": float(supportMeta["support_size"] / max(int(z.size), 1)),
    }
    return nullCenter, nullScale, details, lowerMagnitudes


def _resolveROCCOBootstrapGeometry(
    size: int,
    intervals: npt.ArrayLike | None,
    ends: npt.ArrayLike | None,
) -> Tuple[np.ndarray, np.ndarray]:
    size_ = int(size)
    if intervals is None and ends is None:
        return (
            np.asarray([0, size_], dtype=np.int64),
            np.ones(size_, dtype=np.float64),
        )
    if intervals is None or ends is None:
        raise ValueError("`intervals` and `ends` must be provided together")
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    if intervals_.size != size_ or ends_.size != size_:
        raise ValueError("`intervals`, `ends`, and score track must match length")
    widths = ends_ - intervals_
    if np.any(widths <= 0):
        raise ValueError("ROCCO intervals must have positive width")
    if size_ > 1 and np.any(intervals_[1:] < ends_[:-1]):
        raise ValueError("ROCCO intervals must not overlap")
    boundaries = np.flatnonzero(intervals_[1:] != ends_[:-1]) + 1
    segmentOffsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.asarray(boundaries, dtype=np.int64),
            np.asarray([size_], dtype=np.int64),
        )
    )
    return segmentOffsets, np.asarray(widths, dtype=np.float64)


def _calibrateStationaryNullBootstrap(
    scoreTrack: np.ndarray,
    template: np.ndarray,
    nullCenter: float,
    nullScale: float,
    residualSpanIntervals: int,
    segmentOffsets: npt.ArrayLike | None = None,
    coverageWeights: npt.ArrayLike | None = None,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    thresholdZGrid: Iterable[float] | None = None,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    randomSeed: int = 0,
    templateMeta: Dict[str, Any] | None = None,
    progressLabel: str | None = None,
    useLocalBootStrapRadius: bool = MATCHING_DEFAULT_USE_LOCAL_BOOTSTRAP_RADIUS,
    runBootstrap: bool = True,
) -> Dict[str, Any]:
    scoreTrack_ = _asFloatVector("scoreTrack", scoreTrack)
    template_ = _asFloatVector("template", template)
    zGrid = _resolveThresholdZGrid(
        thresholdZ=thresholdZ,
        thresholdZGrid=thresholdZGrid,
    )
    residualSpanIntervals_ = int(residualSpanIntervals)
    if residualSpanIntervals_ <= 0:
        raise ValueError("`residualSpanIntervals` must be positive")
    if isinstance(numBootstrap, bool) or int(numBootstrap) != numBootstrap:
        raise ValueError("`numBootstrap` must be an integer of at least 8")
    numBootstrap_ = int(numBootstrap)
    if numBootstrap_ < 8:
        raise ValueError("`numBootstrap` must be an integer of at least 8")
    nullScale_ = float(nullScale)
    if not np.isfinite(nullScale_) or nullScale_ <= 0.0:
        raise ValueError("`nullScale` must be finite and positive")
    if segmentOffsets is None and coverageWeights is None:
        segmentOffsets_ = np.asarray([0, template_.size], dtype=np.int64)
        coverageWeights_ = np.ones(template_.size, dtype=np.float64)
    elif segmentOffsets is None or coverageWeights is None:
        raise ValueError(
            "`segmentOffsets` and `coverageWeights` must be provided together"
        )
    else:
        segmentOffsets_ = np.asarray(segmentOffsets, dtype=np.int64).ravel()
        coverageWeights_ = _asFloatVector("coverageWeights", coverageWeights)
        if coverageWeights_.size != template_.size:
            raise ValueError("`coverageWeights` must match template length")
        if np.any(coverageWeights_ <= 0.0):
            raise ValueError("`coverageWeights` must be positive")
        if (
            segmentOffsets_.size < 2
            or int(segmentOffsets_[0]) != 0
            or int(segmentOffsets_[-1]) != template_.size
            or np.any(segmentOffsets_[1:] <= segmentOffsets_[:-1])
        ):
            raise ValueError(
                "`segmentOffsets` must be increasing endpoints from zero to n"
            )
    if not isinstance(useLocalBootStrapRadius, (bool, np.bool_)):
        raise ValueError("`useLocalBootStrapRadius` must be boolean")
    useLocalBootStrapRadius_ = bool(useLocalBootStrapRadius)
    bootstrapBinBP = int(max(round(float(np.median(coverageWeights_))), 1))
    maxLocalRadiusIntervals = (
        int(_ROCCO_LOCAL_BOOTSTRAP_RADIUS_LIMIT_BP // bootstrapBinBP)
        if useLocalBootStrapRadius_
        else -1
    )
    if useLocalBootStrapRadius_:
        bootstrapLocalRadiusMinBins: int | None = None
        bootstrapLocalRadiusMaxBins: int | None = None
        bootstrapLocalRadiusLimitHitSegmentCount = 0
        for segmentIndex in range(segmentOffsets_.size - 1):
            segmentLength = int(
                segmentOffsets_[segmentIndex + 1] - segmentOffsets_[segmentIndex]
            )
            if segmentLength <= 1:
                radiusSquare = 0
            elif residualSpanIntervals_ >= segmentLength:
                radiusSquare = segmentLength - 1
            else:
                radiusProduct = residualSpanIntervals_ * segmentLength
                radiusRoot = math.isqrt(radiusProduct)
                radiusSquare = (
                    radiusRoot
                    if radiusRoot * radiusRoot == radiusProduct
                    else radiusRoot + 1
                )
                radiusSquare = min(radiusSquare, segmentLength - 1)
            effectiveRadius = min(radiusSquare, maxLocalRadiusIntervals)
            bootstrapLocalRadiusMinBins = (
                effectiveRadius
                if bootstrapLocalRadiusMinBins is None
                else min(bootstrapLocalRadiusMinBins, effectiveRadius)
            )
            bootstrapLocalRadiusMaxBins = (
                effectiveRadius
                if bootstrapLocalRadiusMaxBins is None
                else max(bootstrapLocalRadiusMaxBins, effectiveRadius)
            )
            if maxLocalRadiusIntervals < radiusSquare:
                bootstrapLocalRadiusLimitHitSegmentCount += 1
    else:
        bootstrapLocalRadiusMinBins = None
        bootstrapLocalRadiusMaxBins = None
        bootstrapLocalRadiusLimitHitSegmentCount = 0
    totalWeight = float(np.sum(coverageWeights_))
    progressTotal = numBootstrap_
    progressMarks = {
        int(math.ceil(progressTotal * fraction / 4.0)) for fraction in range(1, 5)
    }

    thresholdViews: Dict[str, Dict[str, Any]] = {}
    thresholdMetrics: Dict[str, Dict[str, Any]] = {}
    thresholdOffsets: Dict[str, float] = {}
    primaryKey = _thresholdZKey(float(thresholdZ))
    nullOcc = np.full(numBootstrap_, np.nan)
    farNullOcc = np.full(numBootstrap_, np.nan)
    templateMeta_ = {} if templateMeta is None else dict(templateMeta)
    for z in zGrid:
        key = _thresholdZKey(z)
        z_ = float(z)
        tailAlpha = float(stats.norm.sf(z_))
        tailQuantile = 1.0 - tailAlpha
        thresholdOffset = float(
            max(
                float(
                    np.quantile(
                        template_,
                        tailQuantile,
                        method="interpolated_inverted_cdf",
                    )
                ),
                0.0,
            )
        )
        threshold = float(nullCenter + thresholdOffset)
        thresholdOffsets[key] = float(thresholdOffset)
        trackTailOccupancy = float(
            np.dot(coverageWeights_, scoreTrack_ > threshold) / totalWeight
        )
        farTrackTailOccupancy = float(
            np.dot(coverageWeights_, scoreTrack_ > threshold + thresholdOffset)
            / totalWeight
        )
        nullMeta = {
            "null_center": float(nullCenter),
            "null_scale": float(nullScale_),
            "threshold": float(threshold),
            "threshold_z": float(z_),
            "tail_alpha": float(tailAlpha),
        }
        thresholdViews[key] = {
            "threshold_z": float(z_),
            "null_center": float(nullCenter),
            "null_scale": float(nullScale_),
            "threshold": float(threshold),
            "null_meta": dict(nullMeta),
            "template": np.asarray(template_, dtype=np.float64),
            "template_meta": dict(templateMeta_),
        }
        thresholdMetrics[key] = {
            "threshold_z": float(z_),
            "threshold": float(threshold),
            "track_tail_occupancy": float(trackTailOccupancy),
            "far_track_tail_occupancy": float(farTrackTailOccupancy),
        }

    rng = np.random.default_rng(int(randomSeed))
    for b in range(numBootstrap_ if runBootstrap else 0):
        draw = np.asarray(
            cconsenrich.cStationaryNullBootstrapDraw(
                template_,
                segmentOffsets_,
                coverageWeights_,
                residualSpanIntervals_,
                rng,
                maxLocalRadiusIntervals,
            ),
            dtype=np.float64,
        )
        nullOcc[b] = float(
            np.dot(coverageWeights_, draw > thresholdOffsets[primaryKey])
            / totalWeight
        )
        farNullOcc[b] = float(
            np.dot(coverageWeights_, draw > 2.0 * thresholdOffsets[primaryKey])
            / totalWeight
        )
        progressDone = b + 1
        if progressLabel is not None and progressDone in progressMarks:
            _logRoccoProgress(
                "ROCCO %s: null calibration %d/%d draws",
                progressLabel,
                progressDone,
                progressTotal,
                fields={
                    "stage": "nullCalibration",
                    "completed": progressDone,
                    "total": progressTotal,
                },
            )

    metrics = thresholdMetrics[primaryKey]
    nullTailOccupancy = float(np.mean(nullOcc))
    nullTailOccupancySD = float(np.std(nullOcc, ddof=1))
    nullTailMCSE = float(nullTailOccupancySD / math.sqrt(numBootstrap_))
    signedTailExcess = float(
        float(metrics["track_tail_occupancy"]) - nullTailOccupancy
    )
    budgetOccupancyRaw = float(max(signedTailExcess, 0.0))
    metrics.update(
        {
            "null_tail_occupancy": float(nullTailOccupancy),
            "null_tail_occupancy_sd": float(nullTailOccupancySD),
            "null_tail_mc_se": float(nullTailMCSE),
            "signed_tail_excess": float(signedTailExcess),
            "far_tail_excess": float(
                float(metrics["far_track_tail_occupancy"]) - np.mean(farNullOcc)
            ),
            "budget_occupancy_raw": float(budgetOccupancyRaw),
            "null_tail_occupancy_draws": nullOcc,
            "far_null_tail_occupancy_draws": farNullOcc,
        }
    )

    return {
        "bootstrap_method": "stationary_bootstrap",
        "num_bootstrap": int(numBootstrap_),
        "residual_span_intervals": int(residualSpanIntervals_),
        "bootstrap_block_length": int(residualSpanIntervals_),
        "use_local_bootstrap_radius": bool(useLocalBootStrapRadius_),
        "local_radius_limit_bp": int(_ROCCO_LOCAL_BOOTSTRAP_RADIUS_LIMIT_BP),
        "bootstrap_bin_bp": int(bootstrapBinBP),
        "max_local_radius_intervals": int(maxLocalRadiusIntervals),
        "bootstrap_local_radius_min_bins": bootstrapLocalRadiusMinBins,
        "bootstrap_local_radius_max_bins": bootstrapLocalRadiusMaxBins,
        "bootstrap_local_radius_limit_hit_segment_count": int(
            bootstrapLocalRadiusLimitHitSegmentCount
        ),
        "segment_offsets": np.asarray(segmentOffsets_, dtype=np.int64),
        "coverage_weights": np.asarray(coverageWeights_, dtype=np.float64),
        "threshold_views": thresholdViews,
        "threshold_metrics": thresholdMetrics,
        "budget_z_grid": [float(z) for z in zGrid],
        "primary_key": primaryKey,
    }


def _prepareROCCOBaseScore(
    state: npt.ArrayLike,
    uncertainty: npt.ArrayLike | None = None,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
) -> Dict[str, Any]:
    state_ = _asFloatVector("state", state)
    return {
        "score_track": np.asarray(
            consenrichStateScoreTrack(
                state_,
                uncertainty=uncertainty,
                uncertaintyScoreMode=uncertaintyScoreMode,
                uncertaintyScoreZ=uncertaintyScoreZ,
            ),
            dtype=np.float64,
        )
    }


def _prepareNullResidualTemplate(
    scoreTrack: np.ndarray,
    nullCenter: float,
    coverageWeights: npt.ArrayLike | None = None,
    *,
    lowerTail: Tuple[np.ndarray, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    centered = np.asarray(scoreTrack, dtype=np.float64) - float(nullCenter)
    if coverageWeights is None:
        coverageWeights_ = np.ones(centered.size, dtype=np.float64)
    else:
        coverageWeights_ = _asFloatVector("coverageWeights", coverageWeights)
        if coverageWeights_.size != centered.size:
            raise ValueError("`coverageWeights` must match score track length")
        if np.any(coverageWeights_ <= 0.0):
            raise ValueError("`coverageWeights` must be positive")
    if lowerTail is None:
        lowerMagnitudes = -centered[centered < 0.0]
        if lowerMagnitudes.size < 8:
            raise ValueError("ROCCO null template requires eight lower residuals")
        lowerScale = float(
            np.median(lowerMagnitudes) / float(stats.norm.ppf(0.75))
        )
        if lowerScale <= _TINY:
            raise ValueError("ROCCO lower-residual scale must be positive")
    else:
        lowerMagnitudes, lowerScale = lowerTail

    template = centered.copy()
    positiveMask = centered > 0.0
    positiveCount = int(np.count_nonzero(positiveMask))
    if positiveCount:
        positiveRanks = stats.rankdata(centered[positiveMask], method="average")
        positiveQuantiles = (positiveRanks - 0.5) / float(positiveCount)
        sortedLowerMagnitudes = np.sort(lowerMagnitudes)
        virtualIndices = positiveQuantiles * float(lowerMagnitudes.size) - 1.0
        lowerIndices = np.floor(virtualIndices).astype(np.int64)
        fractions = virtualIndices - lowerIndices
        upperIndices = np.clip(lowerIndices + 1, 0, lowerMagnitudes.size - 1)
        lowerIndices = np.clip(lowerIndices, 0, lowerMagnitudes.size - 1)
        template[positiveMask] = (
            sortedLowerMagnitudes[lowerIndices] * (1.0 - fractions)
            + sortedLowerMagnitudes[upperIndices] * fractions
        )
    lowerCap, upperCap = np.quantile(
        template,
        (0.001, 0.999),
        method="interpolated_inverted_cdf",
    )
    template = np.clip(template, lowerCap, upperCap)
    totalWeight = float(np.sum(coverageWeights_))
    template = template - float(np.dot(coverageWeights_, template) / totalWeight)
    templateStd = float(
        math.sqrt(np.dot(coverageWeights_, template * template) / totalWeight)
    )
    if templateStd <= _TINY:
        raise ValueError("ROCCO null template scale must be positive")
    template = template * (lowerScale / templateStd)

    return template, {
        "template_scale": float(lowerScale),
        "template_mean": float(
            np.dot(coverageWeights_, template) / totalWeight
        ),
        "template_lower_size": float(lowerMagnitudes.size),
        "template_clip_lower": float(lowerCap),
        "template_clip_upper": float(upperCap),
    }


def _prepareROCCOScoreAndNull(
    signal: npt.ArrayLike,
    residualSpanIntervals: int,
    uncertainty: npt.ArrayLike | None = None,
    intervals: npt.ArrayLike | None = None,
    ends: npt.ArrayLike | None = None,
    bulkQuantile: float = 0.60,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    randomSeed: int = 0,
    thresholdZGrid: Iterable[float] | None = None,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    progressLabel: str | None = None,
    useLocalBootStrapRadius: bool = MATCHING_DEFAULT_USE_LOCAL_BOOTSTRAP_RADIUS,
    runBootstrap: bool = True,
) -> Dict[str, Any]:
    r"""Prepare the direct Consenrich score track and robust null for ROCCO budgeting."""
    prepared = _prepareROCCOBaseScore(
        signal,
        uncertainty=uncertainty,
        uncertaintyScoreMode=uncertaintyScoreMode,
        uncertaintyScoreZ=uncertaintyScoreZ,
    )
    scoreTrack = np.asarray(prepared["score_track"], dtype=np.float64)
    segmentOffsets, coverageWeights = _resolveROCCOBootstrapGeometry(
        scoreTrack.size,
        intervals,
        ends,
    )
    zGrid = _resolveThresholdZGrid(
        thresholdZ=thresholdZ,
        thresholdZGrid=thresholdZGrid,
    )
    nullCenter, nullScale, _nullDetails, lowerMagnitudes = _estimateROCCONull(
        scoreTrack,
        bulkQuantile=bulkQuantile,
    )
    template, templateMeta = _prepareNullResidualTemplate(
        scoreTrack,
        nullCenter,
        coverageWeights=coverageWeights,
        lowerTail=(lowerMagnitudes, nullScale),
    )
    del lowerMagnitudes
    calibration = _calibrateStationaryNullBootstrap(
        scoreTrack,
        template,
        nullCenter,
        nullScale,
        residualSpanIntervals=int(residualSpanIntervals),
        segmentOffsets=segmentOffsets,
        coverageWeights=coverageWeights,
        thresholdZ=thresholdZ,
        thresholdZGrid=zGrid,
        numBootstrap=numBootstrap,
        randomSeed=randomSeed,
        templateMeta=templateMeta,
        progressLabel=progressLabel,
        useLocalBootStrapRadius=useLocalBootStrapRadius,
        runBootstrap=runBootstrap,
    )
    thresholdViews = dict(calibration["threshold_views"])
    primaryKey = str(calibration["primary_key"])
    primaryView = thresholdViews[primaryKey]
    return {
        "score_track": scoreTrack,
        "null_center": float(primaryView["null_center"]),
        "null_scale": float(primaryView["null_scale"]),
        "threshold": float(primaryView["threshold"]),
        "template": np.asarray(primaryView["template"], dtype=np.float64),
        "threshold_views": thresholdViews,
        "threshold_metrics": dict(calibration["threshold_metrics"]),
        "null_calibration": {
            "bootstrap_method": str(calibration["bootstrap_method"]),
            "num_bootstrap": int(calibration["num_bootstrap"]),
            "residual_span_intervals": int(
                calibration["residual_span_intervals"]
            ),
            "bootstrap_block_length": int(
                calibration["bootstrap_block_length"]
            ),
            "use_local_bootstrap_radius": bool(
                calibration["use_local_bootstrap_radius"]
            ),
            "local_radius_limit_bp": int(calibration["local_radius_limit_bp"]),
            "bootstrap_bin_bp": int(calibration["bootstrap_bin_bp"]),
            "max_local_radius_intervals": int(
                calibration["max_local_radius_intervals"]
            ),
            "bootstrap_local_radius_min_bins": calibration[
                "bootstrap_local_radius_min_bins"
            ],
            "bootstrap_local_radius_max_bins": calibration[
                "bootstrap_local_radius_max_bins"
            ],
            "bootstrap_local_radius_limit_hit_segment_count": int(
                calibration["bootstrap_local_radius_limit_hit_segment_count"]
            ),
            "segment_offsets": np.asarray(
                calibration["segment_offsets"], dtype=np.int64
            ),
            "coverage_weights": np.asarray(
                calibration["coverage_weights"], dtype=np.float64
            ),
            "random_seed": int(randomSeed),
            "primary_key": str(calibration["primary_key"]),
            "threshold_z_grid": [float(z) for z in calibration["budget_z_grid"]],
        },
    }


def _estimateROCCOTrackTailVariance(
    tailIndicator: npt.ArrayLike,
    segmentOffsets: npt.ArrayLike,
    coverageWeights: npt.ArrayLike,
    minBlockMass: float,
) -> Tuple[float, Dict[str, float | int]]:
    tailIndicator_ = np.asarray(tailIndicator).ravel()
    coverageWeights_ = _asFloatVector("coverageWeights", coverageWeights)
    segmentOffsets_ = np.asarray(segmentOffsets, dtype=np.int64).ravel()
    minBlockMass_ = float(minBlockMass)
    if tailIndicator_.size != coverageWeights_.size:
        raise ValueError("tail indicator and coverage weights must match length")
    if np.any(coverageWeights_ <= 0.0):
        raise ValueError("coverage weights must be positive")
    if (
        segmentOffsets_.size < 2
        or int(segmentOffsets_[0]) != 0
        or int(segmentOffsets_[-1]) != tailIndicator_.size
        or np.any(segmentOffsets_[1:] <= segmentOffsets_[:-1])
    ):
        raise ValueError(
            "segment offsets must be increasing endpoints from zero to n"
        )
    if not np.isfinite(minBlockMass_) or minBlockMass_ <= 0.0:
        raise ValueError("minimum block mass must be finite and positive")

    weightPrefix = np.empty(coverageWeights_.size + 1, dtype=np.float64)
    weightPrefix[0] = 0.0
    np.cumsum(coverageWeights_, out=weightPrefix[1:])
    blockBounds: List[Tuple[int, int]] = []
    mergedTerminalCount = 0
    terminalMergeMass = 0.5 * minBlockMass_
    for segmentStart, segmentEnd in zip(
        segmentOffsets_[:-1], segmentOffsets_[1:]
    ):
        blockStart = int(segmentStart)
        segmentEnd_ = int(segmentEnd)
        firstBlock = len(blockBounds)
        while blockStart < segmentEnd_:
            remainingMass = float(
                weightPrefix[segmentEnd_] - weightPrefix[blockStart]
            )
            if remainingMass < minBlockMass_:
                if (
                    len(blockBounds) > firstBlock
                    and remainingMass < terminalMergeMass
                ):
                    priorStart, _priorEnd = blockBounds[-1]
                    blockBounds[-1] = (priorStart, segmentEnd_)
                    mergedTerminalCount += 1
                else:
                    blockBounds.append((blockStart, segmentEnd_))
                break
            targetMass = float(weightPrefix[blockStart] + minBlockMass_)
            blockEnd = int(np.searchsorted(weightPrefix, targetMass, side="left"))
            blockBounds.append((blockStart, blockEnd))
            blockStart = blockEnd

    blockMasses = np.fromiter(
        (
            weightPrefix[blockEnd] - weightPrefix[blockStart]
            for blockStart, blockEnd in blockBounds
        ),
        dtype=np.float64,
        count=len(blockBounds),
    )
    tailMasses = np.fromiter(
        (
            np.dot(
                coverageWeights_[blockStart:blockEnd],
                tailIndicator_[blockStart:blockEnd],
            )
            for blockStart, blockEnd in blockBounds
        ),
        dtype=np.float64,
        count=len(blockBounds),
    )
    totalMass = float(weightPrefix[-1])
    tailOccupancy = float(np.sum(tailMasses) / totalMass)
    blockResiduals = tailMasses - tailOccupancy * blockMasses
    blockCount = int(blockMasses.size)
    trackTailVariance = (
        float(
            (blockCount / (blockCount - 1.0))
            * np.dot(blockResiduals, blockResiduals)
            / (totalMass * totalMass)
        )
        if blockCount > 1
        else 0.0
    )
    return trackTailVariance, {
        "block_count": int(blockCount),
        "underfilled_block_count": int(
            np.count_nonzero(blockMasses < minBlockMass_)
        ),
        "merged_terminal_count": int(mergedTerminalCount),
    }


def _estimateBudgetForPreparedROCCOScore(
    prepared: Dict[str, Any],
    budgetMin: float = _ROCCO_BUDGET_MIN,
    budgetMax: float = _ROCCO_BUDGET_MAX,
    returnDetails: bool = False,
    *,
    minBlockMass: float | None = None,
) -> float | Tuple[float, Dict[str, Any]]:
    r"""Estimate a ROCCO budget from a shared stationary-bootstrap calibration."""
    scoreTrack = np.asarray(prepared["score_track"], dtype=np.float64)
    thresholdViews = dict(prepared["threshold_views"])
    thresholdMetrics = dict(prepared["threshold_metrics"])
    calibration = dict(prepared["null_calibration"])
    if calibration.get("bootstrap_method") != "stationary_bootstrap":
        raise ValueError("`bootstrap_method` must be 'stationary_bootstrap'")
    primaryKey = str(calibration["primary_key"])
    primaryView = dict(thresholdViews[primaryKey])
    primaryMetrics = dict(thresholdMetrics[primaryKey])
    budgetMin_ = float(max(budgetMin, 0.0))
    budgetMax_ = float(max(budgetMax, budgetMin_))

    coverageWeights = np.asarray(
        calibration["coverage_weights"], dtype=np.float64
    )
    minBlockMass_ = (
        float(
            _ROCCO_TAIL_VARIANCE_BLOCK_SPANS
            * int(calibration["residual_span_intervals"])
            * float(np.median(coverageWeights))
        )
        if minBlockMass is None
        else float(minBlockMass)
    )
    threshold = float(primaryView["threshold"])
    tail = scoreTrack > threshold
    farTail = scoreTrack > 2.0 * threshold - float(primaryView["null_center"])
    band = tail & ~farTail
    nullDraws = np.asarray(
        primaryMetrics["null_tail_occupancy_draws"], dtype=np.float64
    )
    farNullDraws = np.asarray(
        primaryMetrics["far_null_tail_occupancy_draws"], dtype=np.float64
    )
    tailExcess = float(primaryMetrics["signed_tail_excess"])
    farTailExcess = float(primaryMetrics["far_tail_excess"])
    difference = tailExcess - farTailExcess
    bandVariance, _bandDetails = _estimateROCCOTrackTailVariance(
        band, calibration["segment_offsets"], coverageWeights, minBlockMass_
    )
    differenceSE = float(
        math.sqrt(
            bandVariance + np.var(nullDraws - farNullDraws, ddof=1) / nullDraws.size
        )
    )
    tailWeight = (
        float(stats.norm.cdf(difference / differenceSE))
        if differenceSE > 0.0
        else float(difference >= 0.0)
    )
    signedTailExcess = float(
        tailWeight * tailExcess + (1.0 - tailWeight) * farTailExcess
    )
    budgetRaw = float(max(signedTailExcess, 0.0))
    budget = float(np.clip(budgetRaw, budgetMin_, budgetMax_))
    if not returnDetails:
        return budget
    trackTailVariance, _blockDetails = _estimateROCCOTrackTailVariance(
        farTail + tailWeight * band,
        calibration["segment_offsets"],
        coverageWeights,
        minBlockMass_,
    )
    nullTailMCSE = float(
        np.std(tailWeight * nullDraws + (1.0 - tailWeight) * farNullDraws, ddof=1)
        / math.sqrt(nullDraws.size)
    )
    signedTailExcessSE = float(
        math.sqrt(trackTailVariance + nullTailMCSE * nullTailMCSE)
    )
    return budget, {
        "track_tail_occupancy": float(primaryMetrics["track_tail_occupancy"]),
        "null_tail_occupancy": float(primaryMetrics["null_tail_occupancy"]),
        "null_tail_occupancy_draws": nullDraws,
        "null_tail_mc_se": float(nullTailMCSE),
        "tail_excess": tailExcess,
        "far_tail_excess": farTailExcess,
        "tail_excess_weight": tailWeight,
        "signed_tail_excess": signedTailExcess,
        "track_tail_variance": float(trackTailVariance),
        "signed_tail_excess_se": float(signedTailExcessSE),
    }


def _estimateROCCOBlockBudgets(
    signal: np.ndarray,
    uncertainty: np.ndarray | None,
    intervals: np.ndarray,
    ends: np.ndarray,
    residualSpanIntervals: int,
    minBlockMass: float,
    prepareArgs: Mapping[str, Any],
    blockCount: int | None = None,
    progressLabel: str | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    detailNames = (
        "track_tail_occupancy",
        "null_tail_occupancy",
        "null_tail_occupancy_draws",
        "null_tail_mc_se",
        "tail_excess",
        "far_tail_excess",
        "tail_excess_weight",
        "signed_tail_excess",
        "track_tail_variance",
        "signed_tail_excess_se",
    )
    if blockCount is None:
        blockCount = int(
            float(np.sum(ends - intervals))
            // max(
                _ROCCO_BUDGET_BLOCK_MIN_BP,
                _ROCCO_BUDGET_BLOCK_TAIL_BLOCKS * float(minBlockMass),
            )
        )
    bounds = (
        np.linspace(0, signal.size, max(int(blockCount), 1) + 1)
        .astype(np.int64)
        .tolist()
    )
    prepared = _prepareROCCOScoreAndNull(
        signal,
        residualSpanIntervals,
        uncertainty=uncertainty,
        intervals=intervals,
        ends=ends,
        progressLabel=progressLabel,
        runBootstrap=len(bounds) == 2,
        **prepareArgs,
    )
    blocks: List[Dict[str, Any]] = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        block: Dict[str, Any] = {
            "start": start,
            "stop": stop,
            **dict.fromkeys(detailNames, 0.0),
            "null_tail_occupancy_draws": np.zeros(0, dtype=np.float64),
            "tail_excess_weight": 1.0,
        }
        blocks.append(block)
        try:
            blockPrepared = (
                prepared
                if len(bounds) == 2
                else _prepareROCCOScoreAndNull(
                    signal[start:stop],
                    residualSpanIntervals,
                    uncertainty=None if uncertainty is None else uncertainty[start:stop],
                    intervals=intervals[start:stop],
                    ends=ends[start:stop],
                    **prepareArgs,
                )
            )
        except ValueError:
            continue
        _budget, details = _estimateBudgetForPreparedROCCOScore(
            blockPrepared, returnDetails=True, minBlockMass=minBlockMass
        )
        block.update({name: details[name] for name in detailNames})
    return prepared, blocks


def _shrinkROCCOChromosomeBudgets(
    observations: Mapping[Any, Tuple[float, float]],
    *,
    enabled: bool = True,
    selectionPenalty: float | None = None,
    budgetMin: float = _ROCCO_BUDGET_MIN,
    budgetMax: float = _ROCCO_BUDGET_MAX,
) -> Tuple[Dict[Any, float], Dict[Any, float], Dict[str, Any]]:
    r"""Light shrinkage of chromosome-specific budgets to their genome-wide average"""
    if not observations:
        raise ValueError("chromosome budget shrinkage requires observations")
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("enabled must be boolean")

    chromosomeNames = tuple(sorted(observations))
    signedTailExcess = np.asarray(
        [float(observations[name][0]) for name in chromosomeNames],
        dtype=np.float64,
    )
    signedTailExcessSE = np.asarray(
        [float(observations[name][1]) for name in chromosomeNames],
        dtype=np.float64,
    )
    if np.any(signedTailExcessSE < 0.0):
        raise ValueError("signed-tail-excess errors must be non-negative")

    budgetMin_ = float(max(float(budgetMin), 0.0))
    budgetMax_ = float(max(float(budgetMax), budgetMin_))
    cohortSize = len(chromosomeNames)
    pooledMean = float(np.mean(signedTailExcess))
    sampleVariance = (
        float(np.var(signedTailExcess, ddof=1)) if cohortSize > 1 else None
    )
    signedTailExcessVariances = signedTailExcessSE * signedTailExcessSE
    meanSignedTailExcessVariance = float(np.mean(signedTailExcessVariances))
    betweenChromosomeVariance = (
        None
        if sampleVariance is None
        else float(max(sampleVariance - meanSignedTailExcessVariance, 0.0))
    )

    skipReason: str | None = None
    if not bool(enabled):
        skipReason = "disabled"
    elif selectionPenalty is not None:
        skipReason = "selectionPenaltyOverride"
    elif cohortSize < _ROCCO_BUDGET_SHRINKAGE_MIN_CHROMOSOMES:
        skipReason = "tooFewChromosomes"

    retention = np.ones(cohortSize, dtype=np.float64)
    shrunkValues = signedTailExcess.copy()
    if skipReason is None:
        tauSquared = float(betweenChromosomeVariance)
        positiveVariance = signedTailExcessVariances > 0.0
        retention[positiveVariance] = tauSquared / (
            tauSquared + signedTailExcessVariances[positiveVariance]
        )
        shrunkValues = pooledMean + retention * (signedTailExcess - pooledMean)

    budgetValues = np.clip(
        np.maximum(shrunkValues, 0.0),
        budgetMin_,
        budgetMax_,
    )
    budgetsByChrom = {
        name: float(value) for name, value in zip(chromosomeNames, budgetValues)
    }
    retentionByChrom = {
        name: float(value) for name, value in zip(chromosomeNames, retention)
    }
    metadata: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": skipReason is None,
        "method": "normalNormalMomentEB",
        "cohortSize": int(cohortSize),
        "pooledMeanSignedTailExcess": float(pooledMean),
        "sampleVariance": sampleVariance,
        "meanSignedTailExcessVariance": float(meanSignedTailExcessVariance),
        "betweenChromosomeVariance": betweenChromosomeVariance,
        "minimumRetention": float(np.min(retention)),
    }
    if skipReason is not None:
        metadata["skipReason"] = skipReason
    return budgetsByChrom, retentionByChrom, metadata


def solveChromROCCO(
    scores: npt.ArrayLike,
    budget: float | None = None,
    gamma: float = 0.5,
    selectionPenalty: float | None = None,
    maxIter: int = _ROCCO_MAX_ITER_DEFAULT,
    returnDetails: bool = False,
    selectionPenaltyMultiplier: float = MATCHING_DEFAULT_SELECTION_PENALTY_MULTIPLIER,
) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, Dict[str, Any]]:
    scores_ = _asFloatVector("scores", scores)
    gamma_ = float(gamma)
    if not np.isfinite(gamma_) or gamma_ < 0.0:
        raise ValueError("`gamma` must be finite and non-negative")
    if isinstance(selectionPenaltyMultiplier, (bool, np.bool_)):
        raise ValueError("`selectionPenaltyMultiplier` must be finite and positive")
    multiplier = float(selectionPenaltyMultiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("`selectionPenaltyMultiplier` must be finite and positive")
    solution, objective, _penalized, _selected, selectionPenalty_ = (
        cconsenrich.csolveChromROCCOExact(
            scores_,
            budget=budget,
            gamma=gamma_,
            selectionPenalty=selectionPenalty,
            maxIter=int(maxIter),
        )
    )
    baseSelectionPenalty = float(selectionPenalty_)
    if selectionPenalty is None and budget is not None:
        baseSelectionPenalty = max(baseSelectionPenalty, _ROCCO_SELECTION_PENALTY_MIN)
    if multiplier != 1.0 or baseSelectionPenalty != selectionPenalty_:
        selectionPenalty_ = (
            baseSelectionPenalty * multiplier
            if baseSelectionPenalty >= 0.0
            else baseSelectionPenalty / multiplier
        )
        if not np.isfinite(selectionPenalty_):
            raise ValueError("Scaled selection penalty must be finite")
        solution, objective, _penalized, _selected, selectionPenalty_ = (
            cconsenrich.csolveChromROCCOExact(
                scores_,
                gamma=gamma_,
                selectionPenalty=selectionPenalty_,
                maxIter=int(maxIter),
            )
        )
    if not returnDetails:
        return np.asarray(solution, dtype=np.uint8), float(objective)
    details = {"selection_penalty": float(selectionPenalty_)}
    return np.asarray(solution, dtype=np.uint8), float(objective), details


def _solveChromROCCOBlocks(
    scores: np.ndarray,
    blockBounds: Sequence[Tuple[int, int]],
    blockBudgets: Sequence[float],
    gamma: float,
    selectionPenalty: float | None = None,
    selectionPenaltyMultiplier: float = MATCHING_DEFAULT_SELECTION_PENALTY_MULTIPLIER,
) -> Tuple[np.ndarray, float, Dict[str, Any], np.ndarray]:
    blockSizes = [stop - start for start, stop in blockBounds]
    if selectionPenalty is not None or len(blockBounds) == 1:
        solution, objective, details = solveChromROCCO(
            scores,
            budget=float(np.average(blockBudgets, weights=blockSizes)),
            gamma=gamma,
            selectionPenalty=selectionPenalty,
            returnDetails=True,
            selectionPenaltyMultiplier=selectionPenaltyMultiplier,
        )
        return solution, objective, details, scores
    blockPenalties = [
        float(
            solveChromROCCO(
                scores[start:stop],
                budget=float(budget),
                gamma=gamma,
                returnDetails=True,
                selectionPenaltyMultiplier=selectionPenaltyMultiplier,
            )[2]["selection_penalty"]
        )
        for (start, stop), budget in zip(blockBounds, blockBudgets)
    ]
    meanPenalty = float(np.average(blockPenalties, weights=blockSizes))
    solveScores = scores
    if max(blockPenalties) > min(blockPenalties):
        solveScores = np.array(scores, dtype=np.float64)
        for (start, stop), penalty in zip(blockBounds, blockPenalties):
            solveScores[start:stop] -= penalty - meanPenalty
    solution, objective, details = solveChromROCCO(
        solveScores,
        gamma=gamma,
        selectionPenalty=meanPenalty,
        returnDetails=True,
    )
    details["block_selection_penalties"] = blockPenalties
    return solution, objective, details, solveScores


def _selectedRunBounds(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask_ = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    n = int(mask_.size)
    i = 0
    while i < n:
        if not bool(mask_[i]):
            i += 1
            continue
        start = i
        while i + 1 < n and bool(mask_[i + 1]):
            i += 1
        runs.append((int(start), int(i)))
        i += 1
    return runs


def _selectedCoordinateRunBounds(
    mask: np.ndarray,
    intervals: np.ndarray,
    ends: np.ndarray,
) -> List[Tuple[int, int]]:
    mask_ = np.asarray(mask, dtype=bool)
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    if mask_.ndim != 1:
        raise ValueError("`mask` must be one-dimensional")
    return cconsenrich.cSelectedCoordinateRunBounds(mask_, intervals_, ends_)


def _splitBroadRunsByWidth(
    runs: Sequence[Tuple[int, int]],
    intervals: npt.ArrayLike,
    ends: npt.ArrayLike,
    maxRegionBP: int,
) -> List[Tuple[int, int]]:
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    if intervals_.size != ends_.size:
        raise ValueError("`intervals` and `ends` must match length")
    maxRegionBP_ = int(maxRegionBP)
    if maxRegionBP_ <= 0:
        raise ValueError("`maxRegionBP` must be positive")

    boundedRuns: List[Tuple[int, int]] = []
    for runStart, runEnd in runs:
        runStart_ = int(runStart)
        runEnd_ = int(runEnd)
        if runStart_ < 0 or runEnd_ < runStart_ or runEnd_ >= intervals_.size:
            raise ValueError("broad support run indices are invalid")
        chunkStart = runStart_
        while chunkStart <= runEnd_:
            if int(ends_[chunkStart]) - int(intervals_[chunkStart]) > maxRegionBP_:
                raise RuntimeError("source interval exceeds `maxRegionBP`")
            chunkEnd = chunkStart
            while (
                chunkEnd < runEnd_
                and int(ends_[chunkEnd + 1]) - int(intervals_[chunkStart])
                <= maxRegionBP_
            ):
                chunkEnd += 1
            boundedRuns.append((int(chunkStart), int(chunkEnd)))
            chunkStart = chunkEnd + 1
    return boundedRuns


def _validatedOrderedRuns(
    name: str,
    runs: Sequence[tuple[int, int]],
    size: int,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    previousEnd = -1
    for start, end in runs:
        start_ = int(start)
        end_ = int(end)
        if start_ < 0 or end_ < start_ or end_ >= int(size):
            raise ValueError(f"{name} indices are invalid")
        if start_ <= previousEnd:
            raise ValueError(f"{name} must be ordered and disjoint")
        out.append((start_, end_))
        previousEnd = end_
    return out


def _runsTouchingReferenceRuns(
    candidateRuns: Sequence[tuple[int, int]],
    referenceRuns: Sequence[tuple[int, int]],
    intervals: npt.ArrayLike,
    ends: npt.ArrayLike,
) -> list[tuple[int, int]]:
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    if intervals_.size != ends_.size:
        raise ValueError("`intervals` and `ends` must match length")
    candidateRuns_ = _validatedOrderedRuns(
        "candidate runs",
        candidateRuns,
        intervals_.size,
    )
    referenceRuns_ = _validatedOrderedRuns(
        "reference runs",
        referenceRuns,
        intervals_.size,
    )
    touchingRuns: list[tuple[int, int]] = []
    referenceIndex = 0
    for candidateStart, candidateEnd in candidateRuns_:
        candidateStartBP = int(intervals_[candidateStart])
        candidateEndBP = int(ends_[candidateEnd])
        while (
            referenceIndex < len(referenceRuns_)
            and int(ends_[referenceRuns_[referenceIndex][1]])
            < candidateStartBP
        ):
            referenceIndex += 1
        if (
            referenceIndex < len(referenceRuns_)
            and int(intervals_[referenceRuns_[referenceIndex][0]])
            <= candidateEndBP
        ):
            touchingRuns.append((candidateStart, candidateEnd))
    return touchingRuns


def _mergeBroadRunsByObjective(
    runs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    intervals: np.ndarray,
    ends: np.ndarray,
    chromosome: str,
    selectionPenalty: float,
    boundaryCost: float,
    mergeToleranceBP: int,
    maxRegionBP: int,
    blacklistByChrom: Mapping[str, np.ndarray],
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    scores_ = np.asarray(scores, dtype=np.float64).ravel()
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    if scores_.size != intervals_.size or scores_.size != ends_.size:
        raise ValueError("`scores`, `intervals`, and `ends` must match length")
    selectionPenalty_ = float(max(float(selectionPenalty), 0.0))
    boundaryCost_ = float(max(float(boundaryCost), 0.0))
    mergeToleranceBP_ = int(mergeToleranceBP)
    maxRegionBP_ = int(maxRegionBP)
    if mergeToleranceBP_ <= 0 or maxRegionBP_ <= 0:
        raise ValueError("broad merge sizes must be positive")
    atomicRuns = _validatedOrderedRuns("broad atomic runs", runs, scores_.size)
    for start, end in atomicRuns:
        if int(ends_[end]) - int(intervals_[start]) > maxRegionBP_:
            raise RuntimeError("atomic broad run exceeds `maxRegionBP`")
    if not atomicRuns:
        return [], {
            "policy": "width_constrained_objective_dp",
            "num_input_runs": 0,
            "num_output_runs": 0,
            "num_gaps_evaluated": 0,
            "num_gaps_merged": 0,
            "num_gaps_blocked_by_blacklist": 0,
            "num_gaps_blocked_by_distance": 0,
            "num_gaps_blocked_by_gain": 0,
            "merge_tolerance_bp": int(mergeToleranceBP_),
            "max_region_bp": int(maxRegionBP_),
            "retained_utility": 0.0,
        }

    edgeGains: List[float] = []
    eligible: List[bool] = []
    blockedBlacklist = 0
    blockedDistance = 0
    blockedGain = 0
    for (leftStart, leftEnd), (rightStart, rightEnd) in zip(
        atomicRuns[:-1], atomicRuns[1:]
    ):
        del leftStart, rightEnd
        gapStartBP = int(ends_[leftEnd])
        gapEndBP = int(intervals_[rightStart])
        gapBP = int(max(gapEndBP - gapStartBP, 0))
        gapExcess = scores_[leftEnd + 1 : rightStart] - selectionPenalty_
        gapScore = float(
            np.sum(np.where(gapExcess >= 0.0, gapExcess, 0.5 * gapExcess))
        )
        gain = float(gapScore + 2.0 * boundaryCost_)
        distanceOK = gapBP <= mergeToleranceBP_
        blacklistOK = not _intervalOverlapsBlacklist(
            str(chromosome), gapStartBP, gapEndBP, blacklistByChrom
        )
        gainOK = gain > 0.0
        if not distanceOK:
            blockedDistance += 1
        elif not blacklistOK:
            blockedBlacklist += 1
        elif not gainOK:
            blockedGain += 1
        edgeGains.append(gain)
        eligible.append(bool(distanceOK and blacklistOK and gainOK))

    def _partitionChain(first: int, stop: int) -> Tuple[List[Tuple[int, int]], float]:
        chainRuns = atomicRuns[first:stop]
        chainGains = edgeGains[first : stop - 1]
        m = len(chainRuns)
        edgePrefix = np.empty(m, dtype=np.float64)
        edgePrefix[0] = 0.0
        for index, gain in enumerate(chainGains, start=1):
            edgePrefix[index] = float(edgePrefix[index - 1] + gain)
        utilities = np.full(m + 1, -math.inf, dtype=np.float64)
        groups = np.full(m + 1, m + 1, dtype=np.int64)
        predecessors = np.full(m + 1, -1, dtype=np.int64)
        utilities[0] = 0.0
        groups[0] = 0
        candidates = deque([0])
        for j in range(1, m + 1):
            groupEndBP = int(ends_[chainRuns[j - 1][1]])
            while candidates and groupEndBP - int(
                intervals_[chainRuns[candidates[0]][0]]
            ) > maxRegionBP_:
                candidates.popleft()
            if not candidates:
                raise RuntimeError("no feasible broad partition")
            predecessor = int(candidates[0])
            utilities[j] = float(
                edgePrefix[j - 1]
                + float(utilities[predecessor] - edgePrefix[predecessor])
            )
            groups[j] = int(groups[predecessor] + 1)
            predecessors[j] = predecessor
            if j < m:
                candidateValue = float(utilities[j] - edgePrefix[j])
                while candidates:
                    tail = int(candidates[-1])
                    tailValue = float(utilities[tail] - edgePrefix[tail])
                    if candidateValue > tailValue or (
                        candidateValue == tailValue
                        and int(groups[j]) < int(groups[tail])
                    ):
                        candidates.pop()
                    else:
                        break
                candidates.append(j)

        out: List[Tuple[int, int]] = []
        j = m
        while j > 0:
            k = int(predecessors[j])
            if k < 0 or k >= j:
                raise RuntimeError("no optimal broad partition path")
            out.append((int(chainRuns[k][0]), int(chainRuns[j - 1][1])))
            j = k
        out.reverse()
        return out, float(utilities[m])

    mergedRuns: List[Tuple[int, int]] = []
    retainedUtility = 0.0
    chainStart = 0
    for edgeIndex, edgeEligible in enumerate(eligible):
        if edgeEligible:
            continue
        partition, utility = _partitionChain(chainStart, edgeIndex + 1)
        mergedRuns.extend(partition)
        retainedUtility += utility
        chainStart = edgeIndex + 1
    partition, utility = _partitionChain(chainStart, len(atomicRuns))
    mergedRuns.extend(partition)
    retainedUtility += utility
    gapsMerged = int(len(atomicRuns) - len(mergedRuns))
    return mergedRuns, {
        "policy": "width_constrained_objective_dp",
        "num_input_runs": int(len(runs)),
        "num_output_runs": int(len(mergedRuns)),
        "num_gaps_evaluated": int(max(len(runs) - 1, 0)),
        "num_gaps_merged": int(gapsMerged),
        "num_gaps_blocked_by_blacklist": int(blockedBlacklist),
        "num_gaps_blocked_by_distance": int(blockedDistance),
        "num_gaps_blocked_by_gain": int(blockedGain),
        "selection_penalty": float(selectionPenalty_),
        "boundary_cost": float(boundaryCost_),
        "merge_tolerance_bp": int(mergeToleranceBP_),
        "max_region_bp": int(maxRegionBP_),
        "retained_utility": float(retainedUtility),
    }


def _maskJaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_ = np.asarray(a, dtype=bool)
    b_ = np.asarray(b, dtype=bool)
    union = int(np.sum(a_ | b_))
    if union == 0:
        return 1.0
    return float(np.sum(a_ & b_) / union)


def _selectedRunLengthBP(
    start: int,
    end: int,
    intervals: np.ndarray | None,
    ends: np.ndarray | None,
) -> int:
    if intervals is None or ends is None:
        return int(end - start + 1)
    return int(max(int(ends[end]) - int(intervals[start]), 0))


def _minimumChildBinsForRegion(
    start: int,
    end: int,
    intervals: np.ndarray | None,
    ends: np.ndarray | None,
    minRegionBP: int | None,
    minRegionBins: int,
) -> int:
    regionBins = int(max(int(end) - int(start) + 1, 1))
    minBins = int(max(int(minRegionBins), 1))
    if minRegionBP is not None and intervals is not None and ends is not None:
        widths = np.asarray(
            ends[start : end + 1] - intervals[start : end + 1],
            dtype=np.int64,
        )
        widths = widths[widths > 0]
        if widths.size > 0:
            stepBP = int(max(int(np.median(widths)), 1))
            minBins = int(max(1, math.ceil(float(minRegionBP) / float(stepBP))))
    return int(min(regionBins, max(minBins, 1)))


def _nestedSoftSelectionPenalty(
    scores: np.ndarray,
    selectionPenalty: float,
    budgetScale: float,
) -> float:
    budgetScale_ = float(np.clip(float(budgetScale), 0.0, 1.0))
    positive = np.asarray(scores, dtype=np.float64)
    positive = positive[positive > 0.0]
    positiveSpread = 0.0
    if positive.size > 1:
        positiveSpread = float(
            np.quantile(positive, 0.75) - np.quantile(positive, 0.25)
        )
    if (not np.isfinite(positiveSpread)) or positiveSpread < 0.0:
        positiveSpread = 0.0
    return float(
        max(float(selectionPenalty), 0.0) + (1.0 - budgetScale_) * positiveSpread
    )


def _parentConditionedSubpeakObjective(
    scores: np.ndarray,
    mask: np.ndarray,
    boundaryCosts: np.ndarray,
    selectionPenalty: float,
    runPenalty: float = 0.0,
) -> Tuple[float, float, float, float]:
    scores_ = np.asarray(scores, dtype=np.float64)
    mask_ = np.asarray(mask, dtype=bool)
    costs_ = np.asarray(boundaryCosts, dtype=np.float64)
    runPenalty_ = float(runPenalty)
    selected = float(np.sum(scores_[mask_]))
    boundaryPenalty = 0.0
    runCount = 0
    previous = False
    for i, current in enumerate(mask_.tolist()):
        current_ = bool(current)
        if current_ != previous:
            boundaryPenalty += float(costs_[i])
            if current_:
                runCount += 1
        previous = current_
    if previous:
        boundaryPenalty += float(costs_[mask_.size])
    runPenaltyTotal = float(runPenalty_ * float(runCount))
    objective = float(selected - boundaryPenalty - runPenaltyTotal)
    penalized = float(objective - float(selectionPenalty) * float(np.sum(mask_)))
    return objective, penalized, float(boundaryPenalty), float(runPenaltyTotal)


def _empiricalReplaySegmentPValues(
    observedStats: npt.ArrayLike,
    sortedNullStats: npt.ArrayLike,
) -> np.ndarray:
    observed = np.asarray(observedStats, dtype=np.float64).ravel()
    nullStats = np.asarray(sortedNullStats, dtype=np.float64).ravel()
    if observed.size == 0:
        return np.asarray([], dtype=np.float64)
    if nullStats.size == 0:
        return np.ones(observed.size, dtype=np.float64)
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(nullStats)):
        raise ValueError("replay segment statistics contain non-finite values")
    denominator = float(nullStats.size + 1)
    tailStarts = np.searchsorted(nullStats, observed, side="left")
    out = (1.0 + (nullStats.size - tailStarts).astype(np.float64)) / denominator
    return np.clip(out, 0.0, 1.0)


def _replayFDRQValues(
    observedStats: npt.ArrayLike,
    sortedNullStatsByDraw: Iterable[npt.ArrayLike],
) -> np.ndarray:
    observed = np.asarray(observedStats, dtype=np.float64).ravel()
    if observed.size == 0:
        return np.asarray([], dtype=np.float64)
    nullDraws = [
        np.asarray(draw, dtype=np.float64).ravel()
        for draw in sortedNullStatsByDraw
    ]
    if not np.all(np.isfinite(observed)) or any(
        not np.all(np.isfinite(draw)) for draw in nullDraws
    ):
        raise ValueError("replay FDR statistics contain non-finite values")
    statsSorted = np.sort(observed)
    order = np.argsort(-observed, kind="mergesort")
    thresholds = observed[order]
    observedAtThreshold = (
        statsSorted.size
        - np.searchsorted(statsSorted, thresholds, side="left")
    )
    replayPseudocount = (
        1.0 / float(len(nullDraws) + 1) if len(nullDraws) > 0 else 1.0
    )
    expectedNull = np.zeros(observed.size, dtype=np.float64)
    if nullDraws:
        nullAtThreshold = np.zeros(observed.size, dtype=np.int64)
        for draw in nullDraws:
            nullAtThreshold += draw.size - np.searchsorted(
                draw,
                thresholds,
                side="left",
            )
        expectedNull = nullAtThreshold.astype(np.float64) / float(len(nullDraws))
    rawFdr = np.clip(
        (expectedNull + replayPseudocount)
        / np.maximum(observedAtThreshold, 1).astype(np.float64),
        0.0,
        1.0,
    )

    qValues = np.ones(observed.size, dtype=np.float64)
    qValues[order] = np.minimum.accumulate(rawFdr[::-1])[::-1]
    return np.clip(qValues, 0.0, 1.0)


def _resolveMultiscaleCandidateBins(
    n: int,
    morphologySpan: int | None = None,
    lowerSpan: int | None = None,
    upperSpan: int | None = None,
    explicitScales: Iterable[int] | None = None,
) -> List[int]:
    n_ = int(max(int(n), 1))
    raw: List[int] = []
    if explicitScales is not None:
        raw.extend(int(scale) for scale in explicitScales)
    else:
        span = 0 if morphologySpan is None else int(morphologySpan)
        lower = span if lowerSpan is None else int(lowerSpan)
        upper = span if upperSpan is None else int(upperSpan)
        raw.extend(
            [
                1,
                max(2, int(round(max(lower, 1) / 2.0))),
                max(2, lower),
                max(2, span),
                max(2, upper),
            ]
        )
    out: List[int] = []
    seen: set[int] = set()
    for scale in raw:
        scale_ = int(min(max(int(scale), 1), n_))
        if scale_ not in seen:
            seen.add(scale_)
            out.append(scale_)
    out.sort()
    return out


def _segmentScoreAgainstThresholdView(
    scores: np.ndarray,
    start: int,
    end: int,
    view: Mapping[str, Any],
) -> Dict[str, float]:
    scores_ = np.asarray(scores, dtype=np.float64)
    start_ = int(max(int(start), 0))
    end_ = int(min(int(end), int(scores_.size) - 1))
    if end_ < start_:
        return {"score": 0.0}
    threshold = float(view.get("threshold", 0.0))
    nullScale = float(max(float(view.get("null_scale", 1.0)), _TINY))
    excess = np.clip((scores_[start_ : end_ + 1] - threshold) / nullScale, 0.0, None)
    integrated = float(np.sum(excess))
    length = int(max(end_ - start_ + 1, 1))
    score = float(integrated / math.sqrt(float(length)))
    return {"score": float(score)}


def _bestSegmentScoreAcrossThresholdViews(
    scores: np.ndarray,
    start: int,
    end: int,
    thresholdViews: Mapping[str, Any],
) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    for key, viewAny in thresholdViews.items():
        if not isinstance(viewAny, Mapping):
            continue
        stats_ = _segmentScoreAgainstThresholdView(scores, start, end, viewAny)
        candidate = {
            **stats_,
            "threshold_key": str(key),
            "threshold_z": float(viewAny.get("threshold_z", 0.0)),
            "threshold": float(viewAny.get("threshold", 0.0)),
            "null_scale": float(viewAny.get("null_scale", 1.0)),
        }
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate
    if best is None:
        best = {
            "score": 0.0,
            "threshold_key": "",
            "threshold_z": 0.0,
            "threshold": 0.0,
            "null_scale": 1.0,
        }
    return best


def _multiscaleCandidateSegments(
    scores: npt.ArrayLike,
    thresholdViews: Mapping[str, Any],
    scaleBins: Iterable[int] | None = None,
    minRunBins: int = 1,
    maxGapBins: int = 0,
    maxSegments: int | None = _NULL_REPLAY_MAX_SEGMENTS,
    maxSegmentsPerView: int | None = _NULL_REPLAY_MAX_SEGMENTS_PER_VIEW,
) -> List[Dict[str, Any]]:
    scores_ = _asFloatVector("scores", scores)
    scales = _resolveMultiscaleCandidateBins(
        int(scores_.size),
        explicitScales=scaleBins,
    )
    minRunBins_ = int(max(int(minRunBins), 1))
    maxGapBins_ = int(max(int(maxGapBins), 0))
    maxSegments_ = (
        None
        if maxSegments is None or int(maxSegments) <= 0
        else int(max(int(maxSegments), 1))
    )
    maxSegmentsPerView_ = (
        None
        if maxSegmentsPerView is None or int(maxSegmentsPerView) <= 0
        else int(max(int(maxSegmentsPerView), 1))
    )
    thresholdKeys: List[str] = []
    thresholdValues: List[float] = []
    nullScaleValues: List[float] = []
    for key, viewAny in thresholdViews.items():
        if not isinstance(viewAny, Mapping):
            continue
        thresholdKeys.append(str(key))
        thresholdValues.append(float(viewAny.get("threshold", 0.0)))
        nullScaleValues.append(float(max(float(viewAny.get("null_scale", 1.0)), _TINY)))

    nativeRows = cconsenrich.cMultiscaleCandidateSegmentStats(
        scores_,
        np.asarray(scales, dtype=np.int64),
        np.asarray(thresholdValues, dtype=np.float64),
        np.asarray(nullScaleValues, dtype=np.float64),
        minRunBins_,
        maxGapBins_,
        0 if maxSegmentsPerView_ is None else int(maxSegmentsPerView_),
    )
    (
        startArr,
        endArr,
        scaleArr,
        viewArr,
        scoreArr,
        _integratedArr,
        _meanArr,
        _maxArr,
        _eligibleCount,
        _perViewCapHitCount,
        _perViewDiscardedCount,
    ) = nativeRows
    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int, str]] = set()
    for rowIdx in range(int(np.asarray(startArr).size)):
        viewIdx = int(viewArr[rowIdx])
        key = thresholdKeys[viewIdx]
        start = int(startArr[rowIdx])
        end = int(endArr[rowIdx])
        scale = int(scaleArr[rowIdx])
        dedupeKey = (start, end, scale, key)
        if dedupeKey in seen:
            continue
        seen.add(dedupeKey)
        candidates.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "scale_bins": int(scale),
                "threshold_key": str(key),
                "score": float(scoreArr[rowIdx]),
            }
        )
    if maxSegments_ is not None and len(candidates) > maxSegments_:
        candidates = sorted(
            candidates,
            key=lambda candidate: float(candidate.get("score", 0.0)),
            reverse=True,
        )[:maxSegments_]
        candidates.sort(
            key=lambda candidate: (
                int(candidate["start_idx"]),
                int(candidate["end_idx"]),
                int(candidate["scale_bins"]),
                str(candidate["threshold_key"]),
            )
        )
    return candidates


def _resolveRoccoMorphologySpanDetails(
    featureSpanBins: int,
) -> Dict[str, int]:
    point = int(featureSpanBins)
    if point <= 0:
        raise ValueError("`featureSpanBins` must be positive")
    lower = int(max(point // 2, 1))
    upper = int(max(2 * point, point))
    return {"point": point, "lower": lower, "upper": upper}


def _thresholdViewsForNullReplay(
    thresholdViews: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    replayViews: Dict[str, Dict[str, Any]] = {}
    for key, viewAny in thresholdViews.items():
        if not isinstance(viewAny, Mapping):
            continue
        nullCenter = float(viewAny.get("null_center", 0.0))
        threshold = float(viewAny.get("threshold", 0.0))
        replayViews[str(key)] = {
            "threshold_z": float(viewAny.get("threshold_z", 0.0)),
            "threshold": float(threshold - nullCenter),
            "null_scale": float(viewAny.get("null_scale", 1.0)),
        }
    if not replayViews:
        raise ValueError("null calibration has no threshold views")
    return replayViews


def _recordIndexBounds(
    record: _peakRecord,
    intervals: np.ndarray,
    ends: np.ndarray,
) -> Tuple[int, int]:
    startIdx = int(np.searchsorted(ends, int(record.startBP), side="right"))
    endIdx = int(np.searchsorted(intervals, int(record.endBP), side="left") - 1)
    if (
        startIdx < 0
        or endIdx < startIdx
        or endIdx >= intervals.size
        or int(intervals[startIdx]) >= int(record.endBP)
        or int(ends[endIdx]) <= int(record.startBP)
    ):
        raise RuntimeError("peak record has no covered source interval")
    return startIdx, endIdx


def _buildCandidateReplayData(
    scores: npt.ArrayLike,
    prepared: Mapping[str, Any],
    featureSpanBins: int,
    numRegionReplays: int,
    progressLabel: str | None = None,
) -> _candidateReplayData:
    scores_ = _asFloatVector("scores", scores)
    thresholdViews = prepared.get("threshold_views")
    calibration = prepared.get("null_calibration")
    template = prepared.get("template")
    if (
        not isinstance(thresholdViews, Mapping)
        or not isinstance(calibration, Mapping)
        or template is None
    ):
        raise ValueError("null calibration is incomplete")
    template_ = np.asarray(template, dtype=np.float64).ravel()
    if template_.size != scores_.size:
        raise ValueError("null template length does not match scores")

    if calibration.get("bootstrap_method") != "stationary_bootstrap":
        raise ValueError("`bootstrap_method` must be 'stationary_bootstrap'")
    bootstrapBlockLength = int(calibration["bootstrap_block_length"])
    useLocalBootStrapRadius = calibration["use_local_bootstrap_radius"]
    if not isinstance(useLocalBootStrapRadius, (bool, np.bool_)):
        raise ValueError("null calibration radius flag must be boolean")
    maxLocalRadiusIntervalsValue = calibration["max_local_radius_intervals"]
    if (
        isinstance(maxLocalRadiusIntervalsValue, (bool, np.bool_))
        or int(maxLocalRadiusIntervalsValue) != maxLocalRadiusIntervalsValue
    ):
        raise ValueError("null calibration radius must be an integer")
    maxLocalRadiusIntervals = int(maxLocalRadiusIntervalsValue)
    if maxLocalRadiusIntervals < -1:
        raise ValueError("null calibration radius must be -1 or non-negative")
    if bool(useLocalBootStrapRadius) != (maxLocalRadiusIntervals >= 0):
        raise ValueError("null calibration radius fields are inconsistent")
    randomSeed = int(calibration["random_seed"])
    segmentOffsets = np.asarray(calibration["segment_offsets"], dtype=np.int64)
    coverageWeights = np.asarray(
        calibration["coverage_weights"], dtype=np.float64
    )
    morphology = _resolveRoccoMorphologySpanDetails(featureSpanBins)
    scaleBins = _resolveMultiscaleCandidateBins(
        int(scores_.size),
        morphologySpan=int(morphology["point"]),
        lowerSpan=int(morphology["lower"]),
        upperSpan=int(morphology["upper"]),
    )
    dataCandidates = _multiscaleCandidateSegments(
        scores_,
        thresholdViews,
        scaleBins=scaleBins,
        minRunBins=1,
        maxSegments=_NULL_REPLAY_MAX_SEGMENTS,
        maxSegmentsPerView=_NULL_REPLAY_MAX_SEGMENTS_PER_VIEW,
    )
    commonCandidateStats: dict[tuple[int, int], float] = {}
    for candidate in dataCandidates:
        key = (int(candidate["start_idx"]), int(candidate["end_idx"]))
        statistic = float(candidate["score"])
        if (
            key not in commonCandidateStats
            or statistic >= commonCandidateStats[key]
        ):
            commonCandidateStats[key] = statistic
    replayViews = _thresholdViewsForNullReplay(thresholdViews)
    rng = np.random.default_rng(randomSeed)
    replayTotal = int(numRegionReplays)
    replayMarks = {
        int(math.ceil(replayTotal * fraction / 4.0)) for fraction in range(1, 5)
    }
    nullStatsByDraw: list[Any] = [None] * replayTotal
    drawLock = threading.Lock()
    drawOrder = iter(range(replayTotal))

    def replayDraw(_: int) -> None:
        with drawLock:
            drawIndex = next(drawOrder)
            draw = np.asarray(
                cconsenrich.cStationaryNullBootstrapDraw(
                    template_,
                    segmentOffsets,
                    coverageWeights,
                    bootstrapBlockLength,
                    rng,
                    maxLocalRadiusIntervals,
                ),
                dtype=np.float64,
            )
        nullCandidates = _multiscaleCandidateSegments(
            draw,
            replayViews,
            scaleBins=scaleBins,
            minRunBins=1,
            maxSegments=_NULL_REPLAY_MAX_SEGMENTS,
            maxSegmentsPerView=_NULL_REPLAY_MAX_SEGMENTS_PER_VIEW,
        )
        drawStats = np.asarray(
            [float(candidate["score"]) for candidate in nullCandidates],
            dtype=np.float64,
        )
        drawStats.sort()
        drawStats.setflags(write=False)
        nullStatsByDraw[drawIndex] = drawStats

    with ThreadPoolExecutor(max_workers=_NULL_REPLAY_THREADS) as pool:
        for replayDone, _ in enumerate(pool.map(replayDraw, range(replayTotal)), 1):
            if progressLabel is not None and replayDone in replayMarks:
                _logRoccoProgress(
                    "ROCCO %s: candidate replay %d/%d",
                    progressLabel,
                    replayDone,
                    replayTotal,
                    fields={
                        "stage": "candidateReplay",
                        "completed": replayDone,
                        "total": replayTotal,
                    },
                )

    nonemptyNullStats = [draw for draw in nullStatsByDraw if draw.size > 0]
    pooledNullStats = (
        np.concatenate(nonemptyNullStats)
        if nonemptyNullStats
        else np.asarray([], dtype=np.float64)
    )
    pooledNullStats.sort()
    pooledNullStats.setflags(write=False)
    return _candidateReplayData(
        thresholdViews=thresholdViews,
        commonCandidateStats=tuple(sorted(commonCandidateStats.items())),
        nullStatsByDraw=tuple(nullStatsByDraw),
        pooledNullStats=pooledNullStats,
        nullCandidateCounts=tuple(int(draw.size) for draw in nullStatsByDraw),
    )


def _scorePeakRecords(
    records: Sequence[_peakRecord],
    scores: npt.ArrayLike,
    replayData: _candidateReplayData,
    intervals: npt.ArrayLike,
    ends: npt.ArrayLike,
) -> list[_peakRecord]:
    scores_ = _asFloatVector("scores", scores)
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    if intervals_.size != scores_.size or ends_.size != scores_.size:
        raise ValueError("`intervals`, `ends`, and `scores` must match length")
    thresholdViews = replayData.thresholdViews
    candidateStats = dict(replayData.commonCandidateStats)
    recordBounds: list[tuple[int, int]] = []
    for record in records:
        bounds = _recordIndexBounds(record, intervals_, ends_)
        recordBounds.append(bounds)
        statistic = float(
            _bestSegmentScoreAcrossThresholdViews(
                scores_,
                bounds[0],
                bounds[1],
                thresholdViews,
            )["score"]
        )
        if bounds not in candidateStats or statistic >= candidateStats[bounds]:
            candidateStats[bounds] = statistic

    orderedBounds = sorted(candidateStats)
    observedStats = np.asarray(
        [candidateStats[bounds] for bounds in orderedBounds],
        dtype=np.float64,
    )
    candidateIndex = {bounds: index for index, bounds in enumerate(orderedBounds)}
    pValues = _empiricalReplaySegmentPValues(
        observedStats,
        replayData.pooledNullStats,
    )
    qValues = np.maximum(
        _replayFDRQValues(observedStats, replayData.nullStatsByDraw),
        pValues,
    )
    scored = [
        record._replace(
            pValue=float(pValues[candidateIndex[bounds]]),
            qValue=float(qValues[candidateIndex[bounds]]),
        )
        for record, bounds in zip(records, recordBounds)
    ]
    return scored


def _solveParentConditionedSubpeaks(
    scores: np.ndarray,
    boundaryCosts: npt.ArrayLike,
    selectionPenalty: float,
    minRunBins: int,
    requiredIndex: int | None = None,
    runPenalty: float = 0.0,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    scores_ = np.asarray(scores, dtype=np.float64)
    costs_ = np.asarray(boundaryCosts, dtype=np.float64).ravel()
    penalty_ = float(selectionPenalty)
    runPenalty_ = float(runPenalty)

    n = int(scores_.size)
    requiredBin = None if requiredIndex is None else int(requiredIndex)
    if requiredBin is not None and (requiredBin < 0 or requiredBin >= n):
        raise ValueError("`requiredIndex` is outside `scores`")
    minRunBins_ = int(min(max(int(minRunBins), 1), n))
    mask = cconsenrich.cSolveParentConditionedSubpeaks(
        scores_,
        costs_,
        penalty_,
        minRunBins_,
        requiredBin,
        runPenalty_,
    )
    (
        objective,
        penalizedObjective,
        boundaryPenalty,
        runPenaltyTotal,
    ) = _parentConditionedSubpeakObjective(
        scores_,
        mask,
        costs_,
        penalty_,
        runPenalty_,
    )
    selectedCount = int(np.sum(mask))
    if requiredBin is not None and not bool(mask[requiredBin]):
        raise RuntimeError(
            "parent-conditioned subpeak DP violated required bin constraint"
        )
    runs = _selectedRunBounds(mask)
    return (
        mask,
        float(objective),
        {
            "mode": "parent_conditioned_min_run_dp",
            "penalized_objective": float(penalizedObjective),
            "selected_count": int(selectedCount),
            "selected_fraction": float(selectedCount / max(n, 1)),
            "selection_penalty": float(penalty_),
            "run_penalty": float(runPenalty_),
            "run_penalty_total": float(runPenaltyTotal),
            "boundary_cost_min": float(np.min(costs_)),
            "boundary_cost_max": float(np.max(costs_)),
            "boundary_penalty": float(boundaryPenalty),
            "min_run_bins": int(minRunBins_),
            "num_runs": int(len(runs)),
            "required_index": None if requiredBin is None else int(requiredBin),
            "required_selected": bool(
                True if requiredBin is None else mask[requiredBin]
            ),
            "required_fallback_window": False,
        },
    )


def _refineNestedROCCOSolution(
    scores: npt.ArrayLike,
    solution: npt.ArrayLike,
    gamma: float,
    selectionPenalty: float,
    nestedRoccoIters: int = _NESTED_ROCCO_ITERS_DEFAULT,
    nestedRoccoBudgetScale: float = _NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    jaccardThreshold: float = _NESTED_ROCCO_JACCARD_DEFAULT,
    intervals: npt.ArrayLike | None = None,
    ends: npt.ArrayLike | None = None,
    rawScores: npt.ArrayLike | None = None,
    minRegionBP: int | None = None,
    minRegionBins: int = _NESTED_ROCCO_MIN_CHILD_STEPS,
) -> np.ndarray:
    r"""Run local ROCCO refinements inside selected first-pass regions.

    For each eligible parent or child region ``R``, solve an exact local chain
    problem with ``localGamma = 0.25 * gamma``, a hard minimum selected-run
    length, and a mandatory required bin at the strongest local evidence bin. When
    ``nestedRoccoBudgetScale < 1``, translate the scale into a soft per-bin
    penalty rather than a hard local quota. This keeps nested ROCCO as a
    refinement step: children may shrink or split a parent, but every parent
    contributes at least one child.
    """
    scores_ = _asFloatVector("scores", scores)
    rawScores_ = scores_
    if rawScores is not None:
        rawScores_ = _asFloatVector("rawScores", rawScores)
        if rawScores_.size != scores_.size:
            raise ValueError("`rawScores` must match `scores` length")
    current = np.asarray(solution, dtype=np.uint8).ravel() > 0
    if current.size != scores_.size:
        raise ValueError("`solution` must match `scores` length")
    inputSelection = current.copy()

    intervals_: np.ndarray | None = None
    ends_: np.ndarray | None = None
    if intervals is not None or ends is not None:
        if intervals is None or ends is None:
            raise ValueError("`intervals` and `ends` must be supplied together")
        intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
        ends_ = np.asarray(ends, dtype=np.int64).ravel()
        if intervals_.size != scores_.size or ends_.size != scores_.size:
            raise ValueError("`intervals` and `ends` must match `scores` length")

    maxIters = max(int(nestedRoccoIters), 0)
    parentGamma = float(gamma)
    if not np.isfinite(parentGamma) or parentGamma < 0.0:
        raise ValueError("`gamma` must be finite and non-negative")
    selectionPenalty_ = float(selectionPenalty)
    if not np.isfinite(selectionPenalty_):
        raise ValueError("`selectionPenalty` must be finite")
    budgetScale = float(nestedRoccoBudgetScale)
    if not np.isfinite(budgetScale):
        raise ValueError("`nestedRoccoBudgetScale` must be finite")
    budgetScale = float(np.clip(budgetScale, 0.0, 1.0))
    jaccardThreshold_ = float(np.clip(jaccardThreshold, 0.0, 1.0))
    minRegionBins_ = max(int(minRegionBins), 1)
    minRegionBP_ = None if minRegionBP is None else max(int(minRegionBP), 0)
    localGamma = 0.25 * parentGamma
    if maxIters == 0:
        return current.astype(np.uint8)

    frontier = _selectedRunBounds(current)
    for iterIdx in range(maxIters):
        previous = current.copy()
        refined = current.copy()
        nextFrontier: List[Tuple[int, int]] = []
        iterBudgetScale = budgetScale if iterIdx == 0 else 1.0
        for start, end in frontier:
            regionLengthBP = _selectedRunLengthBP(
                start, end, intervals_, ends_
            )
            if (minRegionBP_ is not None and regionLengthBP < minRegionBP_) or (
                minRegionBP_ is None and end - start + 1 < minRegionBins_
            ):
                continue
            localScores = scores_[start : end + 1]
            localRawScores = rawScores_[start : end + 1]
            localMinChildBins = _minimumChildBinsForRegion(
                start,
                end,
                intervals_,
                ends_,
                minRegionBP_,
                minRegionBins_,
            )
            localSolverScores = localScores
            if iterIdx > 0:
                localSolverScores = np.asarray(
                    localScores - float(np.quantile(localScores, 0.25)),
                    dtype=np.float64,
                )
            localSelectionPenalty = _nestedSoftSelectionPenalty(
                localSolverScores,
                0.0 if iterIdx > 0 else selectionPenalty_,
                iterBudgetScale,
            )
            requiredLocal = int(np.argmax(localRawScores))
            nLocal = int(end - start + 1)
            internalBoundaryCost = float(
                max(localGamma, 1000.0 * _NESTED_ROCCO_PARENT_EDGE_COST)
            )
            localBoundaryCosts = np.full(
                nLocal + 1,
                internalBoundaryCost,
                dtype=np.float64,
            )
            localBoundaryCosts[0] = float(_NESTED_ROCCO_PARENT_EDGE_COST)
            localBoundaryCosts[-1] = float(_NESTED_ROCCO_PARENT_EDGE_COST)
            localSelection, _localObjective, localDetails = (
                _solveParentConditionedSubpeaks(
                    localSolverScores,
                    boundaryCosts=localBoundaryCosts,
                    selectionPenalty=localSelectionPenalty,
                    minRunBins=localMinChildBins,
                    requiredIndex=requiredLocal,
                    runPenalty=internalBoundaryCost,
                )
            )
            if not bool(np.any(localSelection)):
                raise RuntimeError("parent-conditioned subpeak solve selected no bins")
            if not bool(localSelection[requiredLocal]):
                raise RuntimeError(
                    "parent-conditioned subpeak solve violated required bin"
                )
            localRuns = _selectedRunBounds(localSelection)
            childWidthsOK = all(
                right - left + 1 >= localMinChildBins
                for left, right in localRuns
            )
            retainedSelection = np.ones(nLocal, dtype=bool)
            _parentObjective, retainedPenalized, _boundary, _runPenalty = (
                _parentConditionedSubpeakObjective(
                    localSolverScores,
                    retainedSelection,
                    localBoundaryCosts,
                    localSelectionPenalty,
                    runPenalty=internalBoundaryCost,
                )
            )
            gain = float(
                float(localDetails["penalized_objective"]) - retainedPenalized
            )
            changed = not np.array_equal(localSelection, retainedSelection)
            accept = bool(
                childWidthsOK
                and gain > 0.0
                and (len(localRuns) >= 2 or (len(localRuns) == 1 and changed))
            )
            if not accept:
                continue
            refined[start : end + 1] = localSelection
            nextFrontier.extend(
                (int(start + left), int(start + right))
                for left, right in localRuns
            )

        if bool(np.any(refined & ~inputSelection)):
            raise RuntimeError("nested ROCCO selection left the input solution")
        jaccard = _maskJaccard(previous, refined)
        current = refined
        if (
            np.array_equal(current, previous)
            or not nextFrontier
            or jaccard >= jaccardThreshold_
        ):
            break
        frontier = nextFrontier
    return current.astype(np.uint8)


def _readAlignedConsenrichBedGraphs(
    stateBedGraphFile: str,
    uncertaintyBedGraphFile: str | None = None,
    exportSignalBedGraphFile: str | None = None,
    chromosomes: Iterable[str] | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    colsState = ["chromosome", "start", "end", "state"]
    stateDF = pd.read_csv(
        stateBedGraphFile,
        sep="\t",
        header=None,
        names=colsState,
        dtype={
            "chromosome": str,
            "start": np.int64,
            "end": np.int64,
            "state": np.float64,
        },
    )
    stateOrder = np.lexsort((
        stateDF["end"].to_numpy(copy=False),
        stateDF["start"].to_numpy(copy=False),
        pd.factorize(stateDF["chromosome"], sort=True, use_na_sentinel=False)[0],
    ))
    stateDF = stateDF.iloc[stateOrder].reset_index(drop=True)

    uncertaintyDF: pd.DataFrame | None = None
    if uncertaintyBedGraphFile is not None:
        colsUnc = ["chromosome", "start", "end", "uncertainty"]
        uncertaintyDF = pd.read_csv(
            uncertaintyBedGraphFile,
            sep="\t",
            header=None,
            names=colsUnc,
            dtype={
                "chromosome": str,
                "start": np.int64,
                "end": np.int64,
                "uncertainty": np.float64,
            },
        )
        uncertaintyOrder = np.lexsort((
            uncertaintyDF["end"].to_numpy(copy=False),
            uncertaintyDF["start"].to_numpy(copy=False),
            pd.factorize(uncertaintyDF["chromosome"], sort=True, use_na_sentinel=False)[0],
        ))
        uncertaintyDF = uncertaintyDF.iloc[uncertaintyOrder].reset_index(drop=True)
        if not stateDF[["chromosome", "start", "end"]].equals(
            uncertaintyDF[["chromosome", "start", "end"]]
        ):
            raise ValueError(
                "`stateBedGraphFile` and `uncertaintyBedGraphFile` are not aligned."
            )

    exportSignalDF: pd.DataFrame | None = None
    if exportSignalBedGraphFile is not None:
        colsExportSignal = ["chromosome", "start", "end", "export_signal"]
        exportSignalDF = pd.read_csv(
            exportSignalBedGraphFile,
            sep="\t",
            header=None,
            names=colsExportSignal,
            dtype={
                "chromosome": str,
                "start": np.int64,
                "end": np.int64,
                "export_signal": np.float64,
            },
        )
        exportOrder = np.lexsort((
            exportSignalDF["end"].to_numpy(copy=False),
            exportSignalDF["start"].to_numpy(copy=False),
            pd.factorize(exportSignalDF["chromosome"], sort=True, use_na_sentinel=False)[0],
        ))
        exportSignalDF = exportSignalDF.iloc[exportOrder].reset_index(drop=True)
        if not stateDF[["chromosome", "start", "end"]].equals(
            exportSignalDF[["chromosome", "start", "end"]]
        ):
            raise ValueError(
                "`stateBedGraphFile` and `exportSignalBedGraphFile` are not aligned."
            )

    allowedChroms = set(chromosomes) if chromosomes is not None else None
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for chromosome, chromStateDF in stateDF.groupby("chromosome", sort=False):
        if allowedChroms is not None and chromosome not in allowedChroms:
            continue
        chromSlice = slice(int(chromStateDF.index[0]), int(chromStateDF.index[-1]) + 1)
        chromUncDF = None
        if uncertaintyDF is not None:
            chromUncDF = uncertaintyDF.iloc[chromSlice]
        chromExportSignalDF = None
        if exportSignalDF is not None:
            chromExportSignalDF = exportSignalDF.iloc[chromSlice]
        out[str(chromosome)] = {
            "intervals": chromStateDF["start"].to_numpy(dtype=np.int64, copy=True),
            "ends": chromStateDF["end"].to_numpy(dtype=np.int64, copy=True),
            "state": chromStateDF["state"].to_numpy(dtype=np.float64, copy=True),
            "uncertainty": (
                chromUncDF["uncertainty"].to_numpy(dtype=np.float64, copy=True)
                if chromUncDF is not None
                else None
            ),
            "export_signal": (
                chromExportSignalDF["export_signal"].to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
                if chromExportSignalDF is not None
                else None
            ),
        }
    return out


def _buildRegionalSignalData(
    coverageWeights: npt.ArrayLike,
    signal: npt.ArrayLike,
) -> _regionalSignalData:
    coverageWeights_ = np.asarray(coverageWeights)
    if coverageWeights_.ndim != 1:
        raise ValueError("`coverageWeights` must be one-dimensional")
    if coverageWeights_.size == 0:
        raise ValueError("`coverageWeights` must be non-empty")
    if not np.issubdtype(coverageWeights_.dtype, np.integer):
        raise ValueError("coverageWeights must contain integer BP widths")
    coverageWeights_ = coverageWeights_.astype(np.int64, copy=False)
    signal_ = _asFloatVector("exportSignal", signal)
    if coverageWeights_.size != signal_.size:
        raise ValueError("coverageWeights and exportSignal must match length")
    if np.any(coverageWeights_ <= 0.0):
        raise ValueError("coverageWeights must be positive")
    coveredBPPrefix = np.empty(coverageWeights_.size + 1, dtype=np.int64)
    coveredBPPrefix[0] = 0
    np.cumsum(coverageWeights_, dtype=np.int64, out=coveredBPPrefix[1:])
    return _regionalSignalData(
        coveredBPPrefix=coveredBPPrefix,
        signalMass=np.multiply(signal_, coverageWeights_, dtype=np.float64),
    )


def _regionalMeanSignal(
    signalData: _regionalSignalData,
    startIdx: int,
    endIdx: int,
) -> float:
    startIdx_ = int(startIdx)
    endIdx_ = int(endIdx)
    size = int(signalData.coveredBPPrefix.size - 1)
    if startIdx_ < 0 or endIdx_ < startIdx_ or endIdx_ >= size:
        raise ValueError("regional signal indices are invalid")
    coveredBP = float(
        signalData.coveredBPPrefix[endIdx_ + 1]
        - signalData.coveredBPPrefix[startIdx_]
    )
    if coveredBP <= 0.0:
        raise RuntimeError("peak region has zero covered BP")
    signalMass = cconsenrich.cRangeSums(
        signalData.signalMass,
        np.asarray([startIdx_], dtype=np.int64),
        np.asarray([endIdx_ + 1], dtype=np.int64),
    )[0]
    return float(signalMass / coveredBP)


def _narrowRecordsFromSolution(
    chromosome: str,
    intervals: npt.ArrayLike,
    ends: npt.ArrayLike,
    signal: npt.ArrayLike,
    scores: npt.ArrayLike,
    solution: npt.ArrayLike,
    signalData: _regionalSignalData,
) -> List[_peakRecord]:
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    signal_ = np.asarray(signal, dtype=np.float64).ravel()
    scores_ = np.asarray(scores, dtype=np.float64).ravel()
    solution_ = np.asarray(solution, dtype=np.uint8).ravel()
    if not (
        intervals_.size
        == ends_.size
        == signal_.size
        == scores_.size
        == solution_.size
    ):
        raise ValueError("narrow record inputs must match length")
    if signalData.coveredBPPrefix.size != signal_.size + 1:
        raise ValueError("regional signal data does not match input length")
    records: List[_peakRecord] = []
    for startIdx, endIdx in _selectedCoordinateRunBounds(
        solution_, intervals_, ends_
    ):
        startBP = int(intervals_[startIdx])
        endBP = int(ends_[endIdx])
        summitIdx = int(startIdx + np.argmax(signal_[startIdx : endIdx + 1]))
        summitBP = int(
            intervals_[summitIdx]
            + (int(ends_[summitIdx]) - int(intervals_[summitIdx])) // 2
        )
        records.append(
            _peakRecord(
                chromosome=str(chromosome),
                startBP=startBP,
                endBP=endBP,
                summitBP=summitBP,
                family="narrow",
                rawScore=float(np.max(scores_[startIdx : endIdx + 1])),
                signalValue=_regionalMeanSignal(signalData, startIdx, endIdx),
                pValue=1.0,
                qValue=1.0,
                blocks=((startBP, endBP),),
            )
        )
    return records


def _broadRecordsFromRuns(
    chromosome: str,
    intervals: npt.ArrayLike,
    ends: npt.ArrayLike,
    signal: npt.ArrayLike,
    scores: npt.ArrayLike,
    parentRuns: Sequence[Tuple[int, int]],
    supportRuns: Sequence[Tuple[int, int]],
    blockRuns: Sequence[Tuple[int, int]],
    signalData: _regionalSignalData,
) -> List[_peakRecord]:
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    signal_ = np.asarray(signal, dtype=np.float64).ravel()
    scores_ = np.asarray(scores, dtype=np.float64).ravel()
    if not (intervals_.size == ends_.size == signal_.size == scores_.size):
        raise ValueError("broad record inputs must match length")
    if signalData.coveredBPPrefix.size != signal_.size + 1:
        raise ValueError("regional signal data does not match input length")
    parentRuns_ = _validatedOrderedRuns(
        "broad parent runs",
        parentRuns,
        scores_.size,
    )
    supportRuns_ = _validatedOrderedRuns(
        "broad support runs",
        supportRuns,
        scores_.size,
    )
    blockRuns_ = _validatedOrderedRuns(
        "broad block runs",
        blockRuns,
        scores_.size,
    )
    records: List[_peakRecord] = []
    supportLeft = 0
    supportRight = 0
    blockLeft = 0
    blockRight = 0
    for parentStartIdx, parentEndIdx in parentRuns_:
        while (
            supportLeft < len(supportRuns_)
            and supportRuns_[supportLeft][1] < parentStartIdx
        ):
            supportLeft += 1
        supportRight = max(supportRight, supportLeft)
        while (
            supportRight < len(supportRuns_)
            and supportRuns_[supportRight][0] <= parentEndIdx
        ):
            supportRight += 1
        overlappingSupportRuns = supportRuns_[supportLeft:supportRight]
        if not overlappingSupportRuns:
            raise RuntimeError("broad parent has no support blocks")
        startBP = int(intervals_[overlappingSupportRuns[0][0]])
        endBP = int(ends_[overlappingSupportRuns[-1][1]])
        firstIdx = int(overlappingSupportRuns[0][0])
        lastIdx = int(overlappingSupportRuns[-1][1])
        while (
            blockLeft < len(blockRuns_)
            and blockRuns_[blockLeft][1] < parentStartIdx
        ):
            blockLeft += 1
        blockRight = max(blockRight, blockLeft)
        while (
            blockRight < len(blockRuns_)
            and blockRuns_[blockRight][0] <= parentEndIdx
        ):
            blockRight += 1
        blockCoordinates = [
            (
                int(intervals_[max(int(startIdx), int(parentStartIdx))]),
                int(ends_[min(int(endIdx), int(parentEndIdx))]),
            )
            for startIdx, endIdx in blockRuns_[blockLeft:blockRight]
        ]
        blocks: List[Tuple[int, int]] = []
        for blockStart, blockEnd in blockCoordinates:
            if blocks and blockStart <= blocks[-1][1]:
                blocks[-1] = (blocks[-1][0], max(blocks[-1][1], blockEnd))
            else:
                blocks.append((blockStart, blockEnd))
        if len(blocks) <= 1:
            blocks = [(startBP, endBP)]
        else:
            blocks[0] = (startBP, blocks[0][1])
            blocks[-1] = (blocks[-1][0], endBP)
        summitIdx = int(firstIdx + np.argmax(signal_[firstIdx : lastIdx + 1]))
        summitBP = int(
            intervals_[summitIdx]
            + (int(ends_[summitIdx]) - int(intervals_[summitIdx])) // 2
        )
        records.append(
            _peakRecord(
                chromosome=str(chromosome),
                startBP=startBP,
                endBP=endBP,
                summitBP=summitBP,
                family="broad",
                rawScore=float(np.mean(scores_[firstIdx : lastIdx + 1])),
                signalValue=_regionalMeanSignal(signalData, firstIdx, lastIdx),
                pValue=1.0,
                qValue=1.0,
                blocks=tuple(blocks),
            )
        )
    return records


def _filterPeakRecordsByBlacklist(
    records: Sequence[_peakRecord],
    blacklistByChrom: Mapping[str, np.ndarray],
) -> Tuple[List[_peakRecord], int]:
    retained = [
        record
        for record in records
        if not _intervalOverlapsBlacklist(
            record.chromosome,
            record.startBP,
            record.endBP,
            blacklistByChrom,
        )
    ]
    return retained, int(len(records) - len(retained))


def _filterPeakRecordsByUncertainty(
    records: Sequence[_peakRecord],
    intervals: np.ndarray,
    ends: np.ndarray,
    signal: np.ndarray,
    uncertainty: np.ndarray | None,
    multiplier: float,
) -> List[_peakRecord]:
    if uncertainty is None:
        return list(records)
    uncertainty_ = np.asarray(uncertainty, dtype=np.float64).ravel()
    signal_ = np.asarray(signal, dtype=np.float64).ravel()
    if uncertainty_.size != signal_.size:
        raise ValueError("uncertainty and signal must match length")
    retained: List[_peakRecord] = []
    for record in records:
        startIdx, endIdx = _recordIndexBounds(record, intervals, ends)
        localUncertainty = uncertainty_[startIdx : endIdx + 1]
        localUncertainty = localUncertainty[np.isfinite(localUncertainty)]
        if localUncertainty.size == 0:
            retained.append(record)
            continue
        if float(np.median(signal_[startIdx : endIdx + 1])) >= (
            -float(multiplier) * float(np.median(localUncertainty))
        ):
            retained.append(record)
    return retained


def _scaledRecordScores(
    records: Sequence[_peakRecord],
    scoreFloor: float,
    scoreCeil: float,
) -> List[int]:
    if not records:
        return []
    rawScores = np.asarray([record.rawScore for record in records], dtype=np.float64)
    minimum = float(np.min(rawScores))
    span = max(float(np.max(rawScores)) - minimum, 1.0e-12)
    return [
        int(
            round(
                scoreFloor
                + (scoreCeil - scoreFloor)
                * ((float(record.rawScore) - minimum) / span)
            )
        )
        for record in records
    ]


def _narrowPeakRows(
    records: Sequence[_peakRecord],
    prefix: str = "consenrichROCCO",
) -> List[List[str | int | float]]:
    scores = _scaledRecordScores(records, 250.0, 1000.0)
    rows: List[List[str | int | float]] = []
    for index, (record, score) in enumerate(zip(records, scores), start=1):
        rows.append(
            [
                record.chromosome,
                record.startBP,
                record.endBP,
                f"{prefix}_{record.chromosome}_{index}",
                score,
                ".",
                record.signalValue,
                _negativeLog10OrMissing(record.pValue),
                _negativeLog10OrMissing(record.qValue),
                int(record.summitBP - record.startBP),
            ]
        )
    return rows


def _gappedPeakRows(
    records: Sequence[_peakRecord],
    prefix: str = "consenrichROCCO",
) -> List[List[str | int | float]]:
    scores = _scaledRecordScores(records, 250.0, 1000.0)
    rows: List[List[str | int | float]] = []
    for index, (record, score) in enumerate(zip(records, scores), start=1):
        blockSizes = [end - start for start, end in record.blocks]
        blockStarts = [start - record.startBP for start, _end in record.blocks]
        rows.append(
            [
                record.chromosome,
                record.startBP,
                record.endBP,
                f"{prefix}_{record.chromosome}_{index}",
                score,
                ".",
                0,
                0,
                0,
                len(record.blocks),
                ",".join(str(value) for value in blockSizes),
                ",".join(str(value) for value in blockStarts),
                record.signalValue,
                _negativeLog10OrMissing(record.pValue),
                _negativeLog10OrMissing(record.qValue),
            ]
        )
    return rows


def _negativeLog10OrMissing(value: Any) -> float | int:
    if value is None:
        return -1
    numeric = float(value)
    if numeric <= 0.0 or numeric > 1.0:
        raise ValueError("empirical p/q values must lie in (0, 1]")
    return float(-math.log10(numeric))


def _plotROCCONullCalibrationDiagnostics(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    dpi: int = 400,
) -> bool:
    if not rows:
        logger.info(
            "nullCalibrationDiagnostics.plot skipped because no chromosome fits exist."
        )
        return False
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        logger.warning(
            "plotNullCalibrationDiagnostics=True but matplotlib is not installed. "
            "Skipped ROCCO null-calibration diagnostics."
        )
        return False

    chromosomeNames = [str(row["block"]) for row in rows]
    nullTailOccupancyDraws = np.concatenate(
        [
            np.asarray(row["nullTailOccupancyDraws"], dtype=np.float64).reshape(-1)
            for row in rows
        ]
    )
    nullTailOccupancy = np.asarray(
        [float(row["nullTailOccupancy"]) for row in rows], dtype=np.float64
    )
    thresholdZ = float(rows[0]["thresholdZ"])
    tailAlpha = float(rows[0]["tailAlpha"])
    signedTailExcess = np.asarray(
        [float(row["signedTailExcess"]) for row in rows], dtype=np.float64
    )
    budgetLocal = np.asarray(
        [float(row["budgetLocal"]) for row in rows], dtype=np.float64
    )
    budget = np.asarray([float(row["budget"]) for row in rows], dtype=np.float64)

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.8, 9.2),
        constrained_layout=True,
    )
    nullAx, excessAx, budgetAx, pairedAx = list(np.ravel(axes))
    navyBlue = "#003B73"
    burntOrange = "#C65A1E"
    darkBlack = "#050505"
    gridColor = "#D8D8D8"

    nullAx.hist(
        nullTailOccupancyDraws,
        bins="auto",
        color=navyBlue,
        alpha=0.78,
        edgecolor=darkBlack,
        linewidth=0.35,
    )
    nullAx.axvline(
        float(np.median(nullTailOccupancyDraws)),
        color=burntOrange,
        linewidth=1.4,
        linestyle="--",
        label="pooled median",
    )
    nullAx.axvline(
        tailAlpha,
        color=darkBlack,
        linewidth=1.2,
        linestyle=":",
        label=rf"normal tail target ($z={thresholdZ:g}$)",
    )
    nullAx.set_title("Bootstrap Null Tail Occupancy", color=darkBlack)
    nullAx.set_xlabel("tail occupancy fraction", color=darkBlack)
    nullAx.set_ylabel("bootstrap draws", color=darkBlack)
    nullAx.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    nullAx.legend(loc="best", fontsize=8, frameon=False)

    comparisonBins = np.histogram_bin_edges(
        np.concatenate((signedTailExcess, budget)),
        bins="auto",
    )
    excessAx.hist(
        signedTailExcess,
        bins=comparisonBins,
        color=burntOrange,
        alpha=0.66,
        edgecolor=darkBlack,
        linewidth=0.35,
        label="signed excess",
    )
    excessAx.hist(
        budget,
        bins=comparisonBins,
        color=navyBlue,
        alpha=0.58,
        edgecolor=darkBlack,
        linewidth=0.35,
        label="shrunk budget",
    )
    excessAx.axvline(0.0, color=darkBlack, linewidth=0.9)
    excessAx.set_title("Block Signed Excess and Budgets", color=darkBlack)
    excessAx.set_xlabel("tail occupancy fraction", color=darkBlack)
    excessAx.set_ylabel("blocks", color=darkBlack)
    excessAx.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    excessAx.legend(loc="best", fontsize=8, frameon=False)

    budgetLimit = float(max(np.max(budgetLocal), np.max(budget), 0.01) * 1.06)
    budgetAx.plot(
        [0.0, budgetLimit],
        [0.0, budgetLimit],
        color=darkBlack,
        linewidth=1.0,
        linestyle="--",
        label="identity",
    )
    budgetAx.scatter(
        budgetLocal,
        budget,
        s=42,
        color=navyBlue,
        edgecolors=darkBlack,
        linewidths=0.45,
        alpha=0.86,
        zorder=3,
    )
    for chromosome, localValue, shrunkValue in zip(
        chromosomeNames,
        budgetLocal,
        budget,
    ):
        budgetAx.annotate(
            chromosome,
            xy=(float(localValue), float(shrunkValue)),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=darkBlack,
        )
    budgetAx.set_xlim(0.0, budgetLimit)
    budgetAx.set_ylim(0.0, budgetLimit)
    budgetAx.set_aspect("equal", adjustable="box")
    budgetAx.set_title("Local and Shrunk Block Budgets", color=darkBlack)
    budgetAx.set_xlabel("local budget fraction", color=darkBlack)
    budgetAx.set_ylabel("shrunk budget fraction", color=darkBlack)
    budgetAx.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    budgetAx.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    budgetAx.legend(loc="best", fontsize=8, frameon=False)

    chromosomeX = np.arange(len(chromosomeNames), dtype=np.float64)
    barWidth = 0.44
    pairedAx.bar(
        chromosomeX - barWidth / 2.0,
        nullTailOccupancy,
        width=barWidth,
        color=navyBlue,
        alpha=0.78,
        edgecolor=darkBlack,
        linewidth=0.35,
        label="bootstrap null mean",
    )
    pairedAx.bar(
        chromosomeX + barWidth / 2.0,
        signedTailExcess,
        width=barWidth,
        color=burntOrange,
        alpha=0.78,
        edgecolor=darkBlack,
        linewidth=0.35,
        label="signed excess",
    )
    pairedAx.axhline(0.0, color=darkBlack, linewidth=0.9)
    pairedAx.set_xticks(chromosomeX)
    pairedAx.set_xticklabels(chromosomeNames, rotation=90)
    pairedAx.set_title("Bootstrap Null and Signed Excess", color=darkBlack)
    pairedAx.set_xlabel("block", color=darkBlack)
    pairedAx.set_ylabel("tail occupancy fraction", color=darkBlack)
    pairedAx.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    pairedAx.legend(loc="best", fontsize=8, frameon=False)

    for axis in (nullAx, excessAx, budgetAx, pairedAx):
        axis.set_axisbelow(True)
        axis.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    figure.suptitle(
        "ROCCO Null Calibration and Block Budgets",
        color=darkBlack,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=int(dpi))
    plt.close(figure)
    logger.info(
        "nullCalibrationDiagnostics.output wrote %s dpi=%d",
        target,
        int(dpi),
    )
    return True


def _writeRoccoMetadata(
    metaPath: str,
    meta: Mapping[str, Any],
    maxNonTrackFileBytes: int = _OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES,
) -> None:
    maxBytes = int(maxNonTrackFileBytes)
    if maxBytes < 0:
        raise ValueError("`maxNonTrackFileBytes` must be non-negative")
    try:
        payload = json.dumps(
            dict(meta),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("ROCCO metadata is not valid JSON") from exc
    if maxBytes and len(payload) > maxBytes:
        raise ValueError(
            f"ROCCO metadata exceeds maxNonTrackFileBytes: "
            f"{len(payload)} > {maxBytes}"
        )

    target = Path(metaPath)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporaryPath: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporaryPath = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporaryPath, target)
    except Exception:
        if temporaryPath is not None:
            temporaryPath.unlink(missing_ok=True)
        raise


def solveRocco(
    signalBedGraphFile: str,
    residualDurationBP: int,
    featureDurationBP: int,
    *,
    uncertaintyBedGraphFile: str | None = None,
    exportSignalBedGraphFile: str | None = None,
    chromosomes: Iterable[str] | None = None,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    gamma: float = MATCHING_DEFAULT_GAMMA,
    selectionPenalty: float | None = None,
    selectionPenaltyMultiplier: float = MATCHING_DEFAULT_SELECTION_PENALTY_MULTIPLIER,
    nestedRoccoIters: int = _NESTED_ROCCO_ITERS_DEFAULT,
    nestedRoccoBudgetScale: float = _NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    exportFilterUncertaintyMultiplier: float = (
        _EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
    ),
    peakMode: Literal["narrow", "broad", "both"] = _MATCHING_DEFAULT_PEAK_MODE,
    broadWeakThresholdZ: float = _MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z,
    mergeToleranceBP: int | None = MATCHING_DEFAULT_MERGE_TOLERANCE_BP,
    maxRegionBP: int | None = MATCHING_DEFAULT_MAX_REGION_BP,
    minMeanSignal: float | None = MATCHING_DEFAULT_MIN_MEAN_SIGNAL,
    numRegionReplays: int = MATCHING_DEFAULT_NUM_REGION_REPLAYS,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    useLocalBootStrapRadius: bool = MATCHING_DEFAULT_USE_LOCAL_BOOTSTRAP_RADIUS,
    shrinkChromosomeBudgets: bool = True,
    budgetMin: float = _ROCCO_BUDGET_MIN,
    budgetMax: float = _ROCCO_BUDGET_MAX,
    plotNullCalibrationDiagnostics: bool = (
        _OUTPUT_DEFAULT_PLOT_NULL_CALIBRATION_DIAGNOSTICS
    ),
    randSeed: int = 42,
    outPath: str | None = None,
    metaPath: str | None = None,
    maxNonTrackFileBytes: int = _OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES,
    blacklistBedFile: str | None = None,
    genomeName: str | None = None,
    writeMetadata: bool = True,
) -> peakArtifacts:
    residualDurationBP_ = _validateDurationBP(
        "residualDurationBP", residualDurationBP
    )
    featureDurationBP_ = _validateDurationBP("featureDurationBP", featureDurationBP)
    peakMode_ = normalization.validateChoice(
        peakMode,
        default=_MATCHING_DEFAULT_PEAK_MODE,
        supported=MATCHING_PEAK_MODES,
        paramName="peakMode",
    )
    broadEnabled = peakMode_ in {"broad", "both"}
    mergeToleranceBP_ = _validateBroadSize(
        "mergeToleranceBP", mergeToleranceBP, broadEnabled
    )
    maxRegionBP_ = _validateBroadSize("maxRegionBP", maxRegionBP, broadEnabled)
    minMeanSignal_ = _validateMinMeanSignal(minMeanSignal)
    numRegionReplays_ = _validateNumRegionReplays(numRegionReplays)
    exportFilterUncertaintyMultiplier_ = _validateExportFilterUncertaintyMultiplier(
        exportFilterUncertaintyMultiplier
    )
    broadWeakThresholdZ_ = _validateBroadWeakThresholdZ(broadWeakThresholdZ)
    thresholdZ_ = float(thresholdZ)
    if not np.isfinite(thresholdZ_) or thresholdZ_ < 0.0:
        raise ValueError("thresholdZ must be finite and non-negative")
    gamma_ = normalization.validateFiniteNonnegativeFloat(gamma, paramName="gamma")
    if isinstance(selectionPenaltyMultiplier, (bool, np.bool_)):
        raise ValueError("selectionPenaltyMultiplier must be finite and positive")
    selectionPenaltyMultiplier_ = float(selectionPenaltyMultiplier)
    if (
        not np.isfinite(selectionPenaltyMultiplier_)
        or selectionPenaltyMultiplier_ <= 0.0
    ):
        raise ValueError("selectionPenaltyMultiplier must be finite and positive")
    if isinstance(numBootstrap, bool) or int(numBootstrap) != numBootstrap:
        raise ValueError("numBootstrap must be an integer of at least 8")
    numBootstrap_ = int(numBootstrap)
    if numBootstrap_ < 8:
        raise ValueError("numBootstrap must be an integer of at least 8")
    if broadEnabled and broadWeakThresholdZ_ > thresholdZ_:
        logger.warning(
            "broadWeakThresholdZ=%g exceeds thresholdZ=%g. "
            "Setting broadWeakThresholdZ to %g.",
            broadWeakThresholdZ_,
            thresholdZ_,
            thresholdZ_,
        )
        broadWeakThresholdZ_ = thresholdZ_
    uncertaintyScoreMode_ = normalization.validateChoice(
        uncertaintyScoreMode,
        default=_MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
        supported=MATCHING_SUPPORTED_UNCERTAINTY_SCORE_MODES,
        paramName="uncertaintyScoreMode",
    )
    uncertaintyScoreZ_ = normalization.validateFiniteNonnegativeFloat(
        uncertaintyScoreZ,
        paramName="uncertaintyScoreZ",
    )
    if not isinstance(useLocalBootStrapRadius, (bool, np.bool_)):
        raise ValueError("useLocalBootStrapRadius must be boolean")
    useLocalBootStrapRadius_ = bool(useLocalBootStrapRadius)
    if not isinstance(shrinkChromosomeBudgets, (bool, np.bool_)):
        raise ValueError("shrinkChromosomeBudgets must be boolean")
    shrinkChromosomeBudgets_ = bool(shrinkChromosomeBudgets)
    if not isinstance(plotNullCalibrationDiagnostics, (bool, np.bool_)):
        raise ValueError("plotNullCalibrationDiagnostics must be boolean")
    plotNullCalibrationDiagnostics_ = bool(plotNullCalibrationDiagnostics)
    if uncertaintyScoreMode_ == "lower_confidence" and uncertaintyBedGraphFile is None:
        raise ValueError(
            "lower_confidence uncertaintyScoreMode requires an uncertainty bedGraph"
        )

    chromosomeNames = (
        None if chromosomes is None else tuple(str(chromosome) for chromosome in chromosomes)
    )
    if chromosomeNames is not None and len(set(chromosomeNames)) != len(
        chromosomeNames
    ):
        raise ValueError("chromosomes must not contain duplicates")
    signalBase = Path(signalBedGraphFile)
    narrowPath: str | None = None
    gappedPath: str | None = None
    if peakMode_ == "narrow":
        narrowPath = (
            str(signalBase.with_name(f"{signalBase.stem}_rocco.narrowPeak"))
            if outPath is None
            else str(outPath)
        )
    elif peakMode_ == "broad":
        gappedPath = (
            str(signalBase.with_name(f"{signalBase.stem}_rocco.gappedPeak"))
            if outPath is None
            else str(outPath)
        )
        if Path(gappedPath).suffix != ".gappedPeak":
            raise ValueError("broad peakMode requires a .gappedPeak output path")
    else:
        if outPath is None:
            narrowPath = str(
                signalBase.with_name(f"{signalBase.stem}_rocco.narrowPeak")
            )
            gappedPath = str(
                signalBase.with_name(f"{signalBase.stem}_rocco.gappedPeak")
            )
        elif Path(outPath).suffix == ".narrowPeak":
            narrowPath = str(outPath)
            gappedPath = str(Path(outPath).with_suffix(".gappedPeak"))
        elif Path(outPath).suffix == ".gappedPeak":
            gappedPath = str(outPath)
            narrowPath = str(Path(outPath).with_suffix(".narrowPeak"))
        else:
            raise ValueError(
                "both peakMode requires outPath ending in .narrowPeak or .gappedPeak"
            )
    if writeMetadata:
        if metaPath is None:
            selectedPath = narrowPath if narrowPath is not None else gappedPath
            metaPath = f"{selectedPath}.json"
        metadataPath: str | None = str(metaPath)
    else:
        metadataPath = None
    selectedPeakPath = narrowPath if narrowPath is not None else gappedPath
    nullCalibrationDiagnosticsCandidatePath = (
        f"{selectedPeakPath}.nullCalibration.png"
        if plotNullCalibrationDiagnostics_
        else None
    )

    peakStart = time.perf_counter()
    _logRoccoProgress(
        "ROCCO: loading input tracks",
        fields={"stage": "inputLoad"},
    )
    blacklistByChrom = _readBlacklistIntervalsByChrom(blacklistBedFile, genomeName)
    chromData = _readAlignedConsenrichBedGraphs(
        signalBedGraphFile,
        uncertaintyBedGraphFile=uncertaintyBedGraphFile,
        exportSignalBedGraphFile=exportSignalBedGraphFile,
        chromosomes=chromosomeNames,
    )
    if chromosomeNames is not None:
        missingChromosomes = tuple(
            chromosome for chromosome in chromosomeNames if chromosome not in chromData
        )
        if missingChromosomes:
            raise ValueError(
                "requested chromosomes are absent from the signal track: "
                + ", ".join(missingChromosomes)
            )
    chromosomeCount = len(chromData)
    inputBinCount = sum(
        int(np.asarray(data["state"]).size) for data in chromData.values()
    )
    _logRoccoProgress(
        "ROCCO: loaded %d chromosomes, %d bins",
        chromosomeCount,
        inputBinCount,
        fields={
            "stage": "inputLoaded",
            "chromosomeCount": chromosomeCount,
            "binCount": inputBinCount,
        },
    )
    thresholdValues = [float(thresholdZ_)]
    if broadEnabled:
        thresholdValues.append(float(broadWeakThresholdZ_))
    thresholdGrid = tuple(sorted(set(thresholdValues)))
    narrowRows: List[List[str | int | float]] = []
    broadRows: List[List[str | int | float]] = []
    chromosomeMetadata: Dict[str, Any] = {}
    diagnosticRows: List[Dict[str, Any]] = []
    totalCalled = {"narrow": 0, "broad": 0}
    totalEmitted = {"narrow": 0, "broad": 0}

    spoolContext = tempfile.TemporaryDirectory(prefix="consenrich-rocco-budget-")
    try:
        spoolDirectory = Path(spoolContext.name)
        calibratedChromosomes: Dict[str, Dict[str, Any]] = {}
        budgetObservations: Dict[Tuple[str, int], Tuple[float, float]] = {}
        for chromIndex, (chromosome, data) in enumerate(chromData.items()):
            signal = np.asarray(data["state"], dtype=np.float64)
            intervals = np.asarray(data["intervals"], dtype=np.int64)
            ends = np.asarray(data["ends"], dtype=np.int64)
            uncertainty = (
                None
                if data["uncertainty"] is None
                else np.asarray(data["uncertainty"], dtype=np.float64)
            )
            if signal.size == 0:
                continue
            chromosomeLabel = f"[{chromIndex + 1}/{chromosomeCount} {str(chromosome)}]"
            coveredBP = int(np.sum(ends - intervals))
            _logRoccoProgress(
                "ROCCO %s: calibrating %d bins, %d bp",
                chromosomeLabel,
                int(signal.size),
                coveredBP,
                fields={
                    "stage": "chromosomeCalibration",
                    "chromosome": str(chromosome),
                    "chromosomeIndex": chromIndex + 1,
                    "chromosomeCount": chromosomeCount,
                    "binCount": int(signal.size),
                    "coveredBP": coveredBP,
                },
            )
            binBP = int(max(round(float(np.median(ends - intervals))), 1))
            residualSpanBins = int(
                max(math.ceil(float(residualDurationBP_) / float(binBP)), 1)
            )
            featureSpanBins = int(
                max(math.ceil(float(featureDurationBP_) / float(binBP)), 1)
            )
            prepareArgs = {
                "thresholdZ": thresholdZ_,
                "numBootstrap": numBootstrap_,
                "randomSeed": int(randSeed),
                "thresholdZGrid": thresholdGrid,
                "uncertaintyScoreMode": uncertaintyScoreMode_,
                "uncertaintyScoreZ": uncertaintyScoreZ_,
                "useLocalBootStrapRadius": useLocalBootStrapRadius_,
            }
            prepared, budgetBlocks = _estimateROCCOBlockBudgets(
                signal,
                uncertainty,
                intervals,
                ends,
                residualSpanBins,
                float(_ROCCO_TAIL_VARIANCE_BLOCK_SPANS * residualSpanBins * binBP),
                prepareArgs,
                progressLabel=chromosomeLabel,
            )
            scoreTrack = np.asarray(prepared["score_track"], dtype=np.float64)
            template = np.asarray(prepared["template"], dtype=np.float64)
            scorePath = spoolDirectory / f"{chromIndex:04d}.score.npy"
            templatePath = spoolDirectory / f"{chromIndex:04d}.template.npy"
            np.save(scorePath, scoreTrack, allow_pickle=False)
            np.save(templatePath, template, allow_pickle=False)

            compactThresholdViews: Dict[str, Dict[str, Any]] = {}
            for key, viewAny in dict(prepared["threshold_views"]).items():
                compactView = dict(viewAny)
                compactView.pop("template", None)
                compactThresholdViews[str(key)] = compactView
            compactCalibration = dict(prepared["null_calibration"])
            compactCalibration.pop("segment_offsets", None)
            compactCalibration.pop("coverage_weights", None)
            compactThresholdMetrics: Dict[str, Dict[str, Any]] = {}
            for key, metricsAny in dict(prepared["threshold_metrics"]).items():
                compactMetrics = dict(metricsAny)
                compactMetrics.pop("null_tail_occupancy_draws", None)
                compactMetrics.pop("far_null_tail_occupancy_draws", None)
                compactThresholdMetrics[str(key)] = compactMetrics
            compactPrepared = dict(prepared)
            compactPrepared.pop("score_track", None)
            compactPrepared.pop("template", None)
            compactPrepared["threshold_views"] = compactThresholdViews
            compactPrepared["threshold_metrics"] = compactThresholdMetrics
            compactPrepared["null_calibration"] = compactCalibration
            calibratedChromosomes[str(chromosome)] = {
                "prepared": compactPrepared,
                "scorePath": scorePath,
                "templatePath": templatePath,
                "binBP": int(binBP),
                "residualSpanBins": int(residualSpanBins),
                "featureSpanBins": int(featureSpanBins),
                "budgetBlocks": budgetBlocks,
            }
            for blockIndex, block in enumerate(budgetBlocks):
                budgetObservations[(str(chromosome), blockIndex)] = (
                    float(block["signed_tail_excess"]),
                    float(block["signed_tail_excess_se"]),
                )
            del compactPrepared, prepared, scoreTrack, template

        budgetsByChrom, budgetRetentionByChrom, budgetShrinkageMetadata = (
            _shrinkROCCOChromosomeBudgets(
                budgetObservations,
                enabled=shrinkChromosomeBudgets_,
                selectionPenalty=selectionPenalty,
                budgetMin=budgetMin,
                budgetMax=budgetMax,
            )
        )
        atMin = [key[0] for key, value in budgetsByChrom.items() if value <= budgetMin]
        atMax = [key[0] for key, value in budgetsByChrom.items() if value >= budgetMax]
        if selectionPenalty is None and (atMin or atMax):
            logger.warning(
                "ROCCO budget at matchingParams.budgetMin=%g in %d block(s) [%s] and at "
                "matchingParams.budgetMax=%g in %d block(s) [%s]. Consider changing "
                "these bounds, the counting mode (samParams.defaultCountMode, "
                "scParams.defaultCountMode), the normalization "
                "(countingParams.normMethod, countingParams.scaleFactors), or setting "
                "the selection penalty directly (matchingParams.selectionPenalty).",
                budgetMin,
                len(atMin),
                ", ".join(sorted(set(atMin))),
                budgetMax,
                len(atMax),
                ", ".join(sorted(set(atMax))),
            )

        for chromIndex, (chromosome, data) in enumerate(chromData.items()):
            chromosomeStart = time.perf_counter()
            signal = np.asarray(data["state"], dtype=np.float64)
            intervals = np.asarray(data["intervals"], dtype=np.int64)
            ends = np.asarray(data["ends"], dtype=np.int64)
            uncertainty = (
                None
                if data["uncertainty"] is None
                else np.asarray(data["uncertainty"], dtype=np.float64)
            )
            exportSignal = (
                signal
                if data["export_signal"] is None
                else np.asarray(data["export_signal"], dtype=np.float64)
            )
            if signal.size == 0:
                continue
            chromosomeLabel = (
                f"[{chromIndex + 1}/{chromosomeCount} {str(chromosome)}]"
            )
            coveredBP = int(np.sum(ends - intervals))
            _logRoccoProgress(
                "ROCCO %s: %d bins, %d bp",
                chromosomeLabel,
                int(signal.size),
                coveredBP,
                fields={
                    "stage": "chromosomeStart",
                    "chromosome": str(chromosome),
                    "chromosomeIndex": chromIndex + 1,
                    "chromosomeCount": chromosomeCount,
                    "binCount": int(signal.size),
                    "coveredBP": coveredBP,
                },
            )
            calibrated = calibratedChromosomes[str(chromosome)]
            binBP = int(calibrated["binBP"])
            residualSpanBins = int(calibrated["residualSpanBins"])
            featureSpanBins = int(calibrated["featureSpanBins"])
            scoreTrack = np.load(
                calibrated["scorePath"],
                mmap_mode="r+",
                allow_pickle=False,
            )
            template = np.load(
                calibrated["templatePath"],
                mmap_mode="r+",
                allow_pickle=False,
            )
            prepared = dict(calibrated["prepared"])
            prepared["template"] = template
            segmentOffsets, coverageWeights = _resolveROCCOBootstrapGeometry(
                scoreTrack.size,
                intervals,
                ends,
            )
            regionalSignalData = _buildRegionalSignalData(
                ends - intervals,
                exportSignal,
            )
            prepared["null_calibration"] = {
                **dict(prepared["null_calibration"]),
                "segment_offsets": segmentOffsets,
                "coverage_weights": coverageWeights,
            }
            primaryKey = str(prepared["null_calibration"]["primary_key"])
            primaryMetrics = dict(prepared["threshold_metrics"][primaryKey])
            tailAlpha = float(
                prepared["threshold_views"][primaryKey]["null_meta"]["tail_alpha"]
            )
            budgetBlocks = calibrated["budgetBlocks"]
            blockKeys = [
                (str(chromosome), blockIndex)
                for blockIndex in range(len(budgetBlocks))
            ]
            blockBounds = [
                (int(block["start"]), int(block["stop"])) for block in budgetBlocks
            ]
            blockSizes = [stop - start for start, stop in blockBounds]
            blockShares = np.asarray(blockSizes, dtype=np.float64) / sum(blockSizes)
            blockBudgets = [float(budgetsByChrom[key]) for key in blockKeys]
            blockLocalBudgets = [
                float(
                    np.clip(
                        block["signed_tail_excess"],
                        budgetMin,
                        budgetMax,
                    )
                )
                for block in budgetBlocks
            ]
            budget = float(np.dot(blockShares, blockBudgets))
            budgetLocal = float(np.dot(blockShares, blockLocalBudgets))

            def blockValues(name: str) -> np.ndarray:
                return np.asarray(
                    [block[name] for block in budgetBlocks], dtype=np.float64
                )

            if plotNullCalibrationDiagnostics_:
                diagnosticRows.extend(
                    {
                        "block": (
                            str(chromosome)
                            if len(budgetBlocks) == 1
                            else f"{chromosome}:{blockIndex + 1}"
                        ),
                        "nullTailOccupancyDraws": block["null_tail_occupancy_draws"][
                            :_ROCCO_DIAGNOSTIC_MAX_BOOTSTRAP_SAMPLES_PER_BLOCK
                        ],
                        "nullTailOccupancy": float(block["null_tail_occupancy"]),
                        "thresholdZ": float(thresholdZ_),
                        "tailAlpha": tailAlpha,
                        "signedTailExcess": float(block["signed_tail_excess"]),
                        "budgetLocal": blockLocalBudgets[blockIndex],
                        "budget": blockBudgets[blockIndex],
                    }
                    for blockIndex, block in enumerate(budgetBlocks)
                )
            _logRoccoProgress(
                "ROCCO %s: solving ROCCO constrained optimization problem... "
                "γ=%.8g",
                chromosomeLabel,
                float(gamma_),
                fields={
                    "stage": "segmentation",
                    "chromosome": str(chromosome),
                    "budget": float(budget),
                    "gamma": float(gamma_),
                },
            )
            solution, objective, solveDetails, solveScores = _solveChromROCCOBlocks(
                scoreTrack,
                blockBounds,
                blockBudgets,
                gamma=float(gamma_),
                selectionPenalty=selectionPenalty,
                selectionPenaltyMultiplier=selectionPenaltyMultiplier_,
            )
            blockPenalties = solveDetails.get(
                "block_selection_penalties",
                [float(solveDetails["selection_penalty"])] * len(budgetBlocks),
            )
            selectedCount = int(np.count_nonzero(solution))
            selectedFraction = float(selectedCount / max(scoreTrack.size, 1))
            _logRoccoProgress(
                "ROCCO %s: λ=%.8g --> %d/%d bins (%.6f), γ=%.8g",
                chromosomeLabel,
                float(solveDetails["selection_penalty"]),
                selectedCount,
                int(scoreTrack.size),
                selectedFraction,
                float(gamma_),
                fields={
                    "stage": "segmentationSolution",
                    "chromosome": str(chromosome),
                    "selectionPenalty": float(solveDetails["selection_penalty"]),
                    "selectionPenaltyMultiplier": selectionPenaltyMultiplier_,
                    "gamma": float(gamma_),
                    "selectedCount": selectedCount,
                    "binCount": int(scoreTrack.size),
                    "selectedFraction": selectedFraction,
                },
            )
            refinedSolution = _refineNestedROCCOSolution(
                solveScores,
                np.asarray(solution, dtype=np.uint8),
                gamma=float(gamma_),
                selectionPenalty=float(solveDetails["selection_penalty"]),
                nestedRoccoIters=int(max(int(nestedRoccoIters), 0)),
                nestedRoccoBudgetScale=nestedRoccoBudgetScale,
                jaccardThreshold=_NESTED_ROCCO_JACCARD_DEFAULT,
                intervals=intervals,
                ends=ends,
                rawScores=scoreTrack,
                minRegionBP=int(
                    _NESTED_ROCCO_MIN_PARENT_STEPS * max(int(binBP), 1)
                ),
                minRegionBins=int(_NESTED_ROCCO_MIN_CHILD_STEPS),
            )
            del objective
            solution_ = np.asarray(refinedSolution, dtype=np.uint8)
            strongRuns = _selectedCoordinateRunBounds(solution_, intervals, ends)
            weakView: Dict[str, Any] | None = None
            if broadEnabled:
                weakView = dict(
                    prepared["threshold_views"][_thresholdZKey(broadWeakThresholdZ_)]
                )
            bridgeBoundaryCost = float(
                _MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER * float(gamma_)
            )

            narrowChromRecords: List[_peakRecord] = []
            broadChromRecords: List[_peakRecord] = []
            if peakMode_ in {"narrow", "both"}:
                narrowChromRecords = _narrowRecordsFromSolution(
                    str(chromosome),
                    intervals,
                    ends,
                    exportSignal,
                    scoreTrack,
                    solution_,
                    regionalSignalData,
                )
            if broadEnabled:
                assert weakView is not None
                weakSupportMask = np.asarray(
                    scoreTrack > float(weakView["threshold"]),
                    dtype=np.uint8,
                )
                weakRuns = _selectedCoordinateRunBounds(
                    weakSupportMask, intervals, ends
                )
                supportMask = solution_.copy()
                for weakStart, weakEnd in _runsTouchingReferenceRuns(
                    weakRuns,
                    strongRuns,
                    intervals,
                    ends,
                ):
                    supportMask[weakStart : weakEnd + 1] = 1
                supportRuns = _selectedCoordinateRunBounds(
                    supportMask, intervals, ends
                )
                supportRuns = _splitBroadRunsByWidth(
                    supportRuns,
                    intervals,
                    ends,
                    int(maxRegionBP_),
                )
                parentRuns, _mergeDetails = _mergeBroadRunsByObjective(
                    supportRuns,
                    solveScores,
                    intervals,
                    ends,
                    str(chromosome),
                    selectionPenalty=float(solveDetails["selection_penalty"]),
                    boundaryCost=float(bridgeBoundaryCost),
                    mergeToleranceBP=int(mergeToleranceBP_),
                    maxRegionBP=int(maxRegionBP_),
                    blacklistByChrom=blacklistByChrom,
                )
                broadChromRecords = _broadRecordsFromRuns(
                    str(chromosome),
                    intervals,
                    ends,
                    exportSignal,
                    scoreTrack,
                    parentRuns,
                    supportRuns,
                    strongRuns,
                    regionalSignalData,
                )

            del regionalSignalData
            calledNarrow = len(narrowChromRecords)
            calledBroad = len(broadChromRecords)
            narrowChromRecords = _filterPeakRecordsByUncertainty(
                narrowChromRecords,
                intervals,
                ends,
                exportSignal,
                uncertainty,
                exportFilterUncertaintyMultiplier_,
            )
            broadChromRecords = _filterPeakRecordsByUncertainty(
                broadChromRecords,
                intervals,
                ends,
                exportSignal,
                uncertainty,
                exportFilterUncertaintyMultiplier_,
            )
            narrowChromRecords, narrowBlacklistDropped = _filterPeakRecordsByBlacklist(
                narrowChromRecords, blacklistByChrom
            )
            broadChromRecords, broadBlacklistDropped = _filterPeakRecordsByBlacklist(
                broadChromRecords, blacklistByChrom
            )

            replayData = _buildCandidateReplayData(
                scoreTrack,
                prepared,
                featureSpanBins,
                numRegionReplays_,
                progressLabel=chromosomeLabel,
            )
            narrowReplayCounts: Tuple[int, ...] = ()
            broadReplayCounts: Tuple[int, ...] = ()
            if peakMode_ in {"narrow", "both"}:
                narrowChromRecords = _scorePeakRecords(
                    narrowChromRecords,
                    scoreTrack,
                    replayData,
                    intervals,
                    ends,
                )
                narrowReplayCounts = replayData.nullCandidateCounts
                if minMeanSignal_ is not None:
                    narrowChromRecords = [
                        record
                        for record in narrowChromRecords
                        if record.signalValue >= minMeanSignal_
                    ]
                narrowRows.extend(_narrowPeakRows(narrowChromRecords))
            if broadEnabled:
                broadChromRecords = _scorePeakRecords(
                    broadChromRecords,
                    scoreTrack,
                    replayData,
                    intervals,
                    ends,
                )
                broadReplayCounts = replayData.nullCandidateCounts
                if minMeanSignal_ is not None:
                    broadChromRecords = [
                        record
                        for record in broadChromRecords
                        if record.signalValue >= minMeanSignal_
                    ]
                broadRows.extend(_gappedPeakRows(broadChromRecords))

            emittedNarrow = len(narrowChromRecords)
            emittedBroad = len(broadChromRecords)
            totalCalled["narrow"] += calledNarrow
            totalCalled["broad"] += calledBroad
            totalEmitted["narrow"] += emittedNarrow
            totalEmitted["broad"] += emittedBroad
            chromosomeMetadata[str(chromosome)] = {
                "binCount": int(signal.size),
                "coveredBP": coveredBP,
                "blacklistIntervalCount": int(
                    np.asarray(
                        blacklistByChrom.get(str(chromosome), np.empty((0, 2)))
                    ).shape[0]
                ),
                "blacklistDroppedCount": int(
                    narrowBlacklistDropped + broadBlacklistDropped
                ),
                "fit": {
                    "residualSpanBins": int(residualSpanBins),
                    "featureSpanBins": int(featureSpanBins),
                    "nullCenter": float(prepared["null_center"]),
                    "nullScale": float(prepared["null_scale"]),
                    "threshold": float(primaryMetrics["threshold"]),
                    "thresholdZ": float(primaryMetrics["threshold_z"]),
                    "tailAlpha": tailAlpha,
                    "bootstrapMethod": str(
                        prepared["null_calibration"]["bootstrap_method"]
                    ),
                    "numBootstrap": int(
                        prepared["null_calibration"]["num_bootstrap"]
                    ),
                    "bootstrapBlockLength": int(
                        prepared["null_calibration"]["bootstrap_block_length"]
                    ),
                    "useLocalBootStrapRadius": bool(
                        prepared["null_calibration"][
                            "use_local_bootstrap_radius"
                        ]
                    ),
                    "localBootstrapRadiusLimitBP": int(
                        prepared["null_calibration"]["local_radius_limit_bp"]
                    ),
                    "bootstrapBinBP": int(
                        prepared["null_calibration"]["bootstrap_bin_bp"]
                    ),
                    "maxLocalRadiusIntervals": int(
                        prepared["null_calibration"][
                            "max_local_radius_intervals"
                        ]
                    ),
                    "bootstrapLocalRadiusMinBins": prepared[
                        "null_calibration"
                    ]["bootstrap_local_radius_min_bins"],
                    "bootstrapLocalRadiusMaxBins": prepared[
                        "null_calibration"
                    ]["bootstrap_local_radius_max_bins"],
                    "bootstrapLocalRadiusLimitHitSegmentCount": int(
                        prepared["null_calibration"][
                            "bootstrap_local_radius_limit_hit_segment_count"
                        ]
                    ),
                    "nullTailOccupancy": float(
                        np.dot(blockShares, blockValues("null_tail_occupancy"))
                    ),
                    "nullTailMCSE": float(
                        np.linalg.norm(blockShares * blockValues("null_tail_mc_se"))
                    ),
                    "trackTailOccupancy": float(
                        np.dot(blockShares, blockValues("track_tail_occupancy"))
                    ),
                    "signedTailExcess": float(
                        np.dot(blockShares, blockValues("signed_tail_excess"))
                    ),
                    "trackTailVariance": float(
                        np.dot(
                            blockShares * blockShares,
                            blockValues("track_tail_variance"),
                        )
                    ),
                    "signedTailExcessSE": float(
                        np.linalg.norm(
                            blockShares * blockValues("signed_tail_excess_se")
                        )
                    ),
                    "budgetLocal": float(budgetLocal),
                    "budgetRetention": float(
                        np.dot(
                            blockShares,
                            [budgetRetentionByChrom[key] for key in blockKeys],
                        )
                    ),
                    "budget": float(budget),
                    "budgetBlocks": [
                        {
                            "startBP": int(intervals[start]),
                            "endBP": int(ends[stop - 1]),
                            "tailExcess": float(block["tail_excess"]),
                            "farTailExcess": float(block["far_tail_excess"]),
                            "tailExcessWeight": float(block["tail_excess_weight"]),
                            "signedTailExcess": float(block["signed_tail_excess"]),
                            "signedTailExcessSE": float(
                                block["signed_tail_excess_se"]
                            ),
                            "budget": float(blockBudget),
                            "selectionPenalty": float(blockPenalty),
                        }
                        for block, (start, stop), blockBudget, blockPenalty in zip(
                            budgetBlocks, blockBounds, blockBudgets, blockPenalties
                        )
                    ],
                    "gamma": float(gamma_),
                    "selectionPenalty": float(solveDetails["selection_penalty"]),
                    "selectionPenaltyMultiplier": selectionPenaltyMultiplier_,
                },
                "narrow": (
                    {
                        "calledCount": int(calledNarrow),
                        "emittedCount": int(emittedNarrow),
                        "replayNullCandidateCounts": [
                            int(value) for value in narrowReplayCounts
                        ],
                    }
                    if peakMode_ in {"narrow", "both"}
                    else None
                ),
                "broad": (
                    {
                        "weakNullCenter": float(weakView["null_center"]),
                        "weakNullScale": float(weakView["null_scale"]),
                        "weakThreshold": float(weakView["threshold"]),
                        "bridgeBoundaryCost": float(bridgeBoundaryCost),
                        "calledCount": int(calledBroad),
                        "emittedCount": int(emittedBroad),
                        "replayNullCandidateCounts": [
                            int(value) for value in broadReplayCounts
                        ],
                    }
                    if broadEnabled
                    else None
                ),
            }
            chromosomeElapsed = time.perf_counter() - chromosomeStart
            _logRoccoProgress(
                "ROCCO %s: complete, narrow=%d broad=%d, %.1fs",
                chromosomeLabel,
                emittedNarrow,
                emittedBroad,
                chromosomeElapsed,
                fields={
                    "stage": "chromosomeComplete",
                    "chromosome": str(chromosome),
                    "chromosomeIndex": chromIndex + 1,
                    "chromosomeCount": chromosomeCount,
                    "narrowCount": emittedNarrow,
                    "broadCount": emittedBroad,
                    "elapsedSeconds": chromosomeElapsed,
                },
            )

            del (
                prepared,
                scoreTrack,
                solveScores,
                template,
                weakView,
                segmentOffsets,
                coverageWeights,
                uncertainty,
            )

    finally:
        spoolContext.cleanup()

    narrowRows.sort(key=lambda row: (str(row[0]), int(row[1]), int(row[2])))
    broadRows.sort(key=lambda row: (str(row[0]), int(row[1]), int(row[2])))
    _logRoccoProgress(
        "ROCCO: writing outputs, narrow=%d broad=%d",
        len(narrowRows),
        len(broadRows),
        fields={
            "stage": "outputWrite",
            "narrowCount": len(narrowRows),
            "broadCount": len(broadRows),
        },
    )
    if narrowPath is not None:
        narrowTarget = Path(narrowPath)
        narrowTarget.parent.mkdir(parents=True, exist_ok=True)
        with narrowTarget.open("w", encoding="utf-8") as handle:
            for row in narrowRows:
                handle.write("\t".join(map(str, row)) + "\n")
    if gappedPath is not None:
        gappedTarget = Path(gappedPath)
        gappedTarget.parent.mkdir(parents=True, exist_ok=True)
        with gappedTarget.open("w", encoding="utf-8") as handle:
            for row in broadRows:
                handle.write("\t".join(map(str, row)) + "\n")

    nullCalibrationDiagnosticsPath: str | None = None
    if nullCalibrationDiagnosticsCandidatePath is not None:
        if _plotROCCONullCalibrationDiagnostics(
            diagnosticRows,
            nullCalibrationDiagnosticsCandidatePath,
        ):
            nullCalibrationDiagnosticsPath = str(
                nullCalibrationDiagnosticsCandidatePath
            )

    def _qMember(enabled: bool) -> Dict[str, Any] | None:
        if not enabled:
            return None
        return {
            "method": "stationaryBootstrapCandidateReplay",
            "scope": "chromosome",
            "statistic": "widthAdjustedMassBins",
            "dataConstructor": "multiscalePlusExports",
            "nullConstructor": "multiscaleOccurrences",
            "empiricalPCap": True,
            "numRegionReplays": int(numRegionReplays_),
        }

    metadata = {
        "inputs": {
            "signalBedGraph": str(signalBedGraphFile),
            "uncertaintyBedGraph": (
                None
                if uncertaintyBedGraphFile is None
                else str(uncertaintyBedGraphFile)
            ),
            "exportSignalBedGraph": (
                None
                if exportSignalBedGraphFile is None
                else str(exportSignalBedGraphFile)
            ),
            "blacklistBed": (
                None if blacklistBedFile is None else str(blacklistBedFile)
            ),
            "blacklistAssembly": None if blacklistBedFile is not None else genomeName,
        },
        "outputs": {
            "narrowPeak": narrowPath,
            "gappedPeak": gappedPath,
            "nullCalibrationDiagnostics": nullCalibrationDiagnosticsPath,
        },
        "settings": {
            "residualDurationBP": int(residualDurationBP_),
            "featureDurationBP": int(featureDurationBP_),
            "chromosomes": (
                None if chromosomeNames is None else list(chromosomeNames)
            ),
            "numBootstrap": int(numBootstrap_),
            "thresholdZ": float(thresholdZ_),
            "gamma": float(gamma_),
            "selectionPenalty": (
                None if selectionPenalty is None else float(selectionPenalty)
            ),
            "selectionPenaltyMultiplier": selectionPenaltyMultiplier_,
            "nestedRoccoIters": int(max(int(nestedRoccoIters), 0)),
            "nestedRoccoBudgetScale": float(nestedRoccoBudgetScale),
            "exportFilterUncertaintyMultiplier": float(
                exportFilterUncertaintyMultiplier_
            ),
            "peakMode": str(peakMode_),
            "broadWeakThresholdZ": float(broadWeakThresholdZ_),
            "mergeToleranceBP": mergeToleranceBP_,
            "maxRegionBP": maxRegionBP_,
            "minMeanSignal": minMeanSignal_,
            "numRegionReplays": int(numRegionReplays_),
            "uncertaintyScoreMode": str(uncertaintyScoreMode_),
            "uncertaintyScoreZ": float(uncertaintyScoreZ_),
            "useLocalBootStrapRadius": bool(useLocalBootStrapRadius_),
            "localBootstrapRadiusLimitBP": int(
                _ROCCO_LOCAL_BOOTSTRAP_RADIUS_LIMIT_BP
            ),
            "shrinkChromosomeBudgets": bool(shrinkChromosomeBudgets_),
            "plotNullCalibrationDiagnostics": bool(plotNullCalibrationDiagnostics_),
            "randSeed": int(randSeed),
        },
        "budgetShrinkage": budgetShrinkageMetadata,
        "chromosomes": chromosomeMetadata,
        "qValues": {
            "narrow": _qMember(peakMode_ in {"narrow", "both"}),
            "broad": _qMember(broadEnabled),
        },
        "counts": {
            "narrow": (
                {
                    "calledCount": int(totalCalled["narrow"]),
                    "emittedCount": int(totalEmitted["narrow"]),
                }
                if peakMode_ in {"narrow", "both"}
                else None
            ),
            "broad": (
                {
                    "calledCount": int(totalCalled["broad"]),
                    "emittedCount": int(totalEmitted["broad"]),
                }
                if broadEnabled
                else None
            ),
        },
    }
    if metadataPath is not None:
        _writeRoccoMetadata(
            metadataPath,
            metadata,
            maxNonTrackFileBytes=maxNonTrackFileBytes,
        )
    peakElapsed = time.perf_counter() - peakStart
    _logRoccoProgress(
        "ROCCO: complete in %.1fs",
        peakElapsed,
        fields={
            "stage": "complete",
            "elapsedSeconds": peakElapsed,
        },
    )
    return peakArtifacts(
        narrowPeak=narrowPath,
        gappedPeak=gappedPath,
        metadata=metadataPath,
        nullCalibrationDiagnostics=nullCalibrationDiagnosticsPath,
    )
