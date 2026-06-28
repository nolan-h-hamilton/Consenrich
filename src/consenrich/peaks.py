# -*- coding: utf-8 -*-
r"""Peak-calling helpers for ROCCO segmentation from Consenrich tracks."""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from . import cconsenrich
from . import core
from ._normalization import (
    normalize_matching_uncertainty_score_mode as _sharedNormalizeUncertaintyScoreMode,
    validate_uncertainty_score_z as _sharedValidateUncertaintyScoreZ,
)

from .constants import (
    EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER,
    MASSIVE_SUBPEAK_CLEANUP_DEFAULT,
    MASSIVE_SUBPEAK_MAX_DEPTH,
    MASSIVE_SUBPEAK_MAX_FRACTION,
    MASSIVE_SUBPEAK_MIN_BP,
    MASSIVE_SUBPEAK_MIN_CHILD_BP,
    MASSIVE_SUBPEAK_MIN_CHILD_FRACTION,
    MASSIVE_SUBPEAK_MIN_LOG_GAP,
    MASSIVE_SUBPEAK_MIN_PEAKS,
    MASSIVE_SUBPEAK_SPLIT_QUANTILE,
    MASSIVE_SUBPEAK_SPLIT_Z,
    MASSIVE_SUBPEAK_WIDTH_ALPHA,
    MASSIVE_SUBPEAK_WIDTH_BULK_QUANTILE,
    MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    MATCHING_DEFAULT_BROAD_BRIDGE_DIP_PENALTY_FRACTION,
    MATCHING_DEFAULT_BROAD_MAX_GAP_BP,
    MATCHING_DEFAULT_BROAD_MIN_PEAK_BP,
    MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER,
    MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z,
    MATCHING_DEFAULT_METADATA_DETAIL,
    MATCHING_DEFAULT_MIN_PEAK_SCORE,
    MATCHING_DEFAULT_PEAK_MODE,
    MATCHING_METADATA_DETAILS,
    MATCHING_PEAK_MODES,
    MATCHING_SUPPORTED_UNCERTAINTY_SCORE_MODES,
    NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    NESTED_ROCCO_ITERS_DEFAULT,
    NESTED_ROCCO_JACCARD_DEFAULT,
    NESTED_ROCCO_MIN_CHILD_STEPS,
    NESTED_ROCCO_MIN_PARENT_STEPS,
    NESTED_ROCCO_SUBTASK_MAX_ITER,
    OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES,
    ROCCO_BUDGET_MAX,
    ROCCO_BUDGET_MIN,
    ROCCO_BUDGET_Z_GRID,
    ROCCO_MAX_ITER_DEFAULT,
    ROCCO_MIN_PEAK_BP,
    ROCCO_NULL_QUANTILE,
    ROCCO_NUM_BOOTSTRAP_DEFAULT,
    ROCCO_THRESHOLD_Z_DEFAULT,
)

logger = logging.getLogger(__name__)

_TINY = float(np.finfo(np.float64).tiny)
_ROCCO_BUDGET_MIN = ROCCO_BUDGET_MIN
_ROCCO_BUDGET_MAX = ROCCO_BUDGET_MAX
_ROCCO_NULL_QUANTILE = ROCCO_NULL_QUANTILE
_ROCCO_THRESHOLD_Z_DEFAULT = ROCCO_THRESHOLD_Z_DEFAULT
_ROCCO_NUM_BOOTSTRAP_DEFAULT = ROCCO_NUM_BOOTSTRAP_DEFAULT
_ROCCO_BUDGET_Z_GRID = ROCCO_BUDGET_Z_GRID
_ROCCO_MAX_ITER_DEFAULT = ROCCO_MAX_ITER_DEFAULT
_ROCCO_MIN_PEAK_BP = ROCCO_MIN_PEAK_BP
_MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE = MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE
_MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z = MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z
_MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z = MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z
_MATCHING_DEFAULT_BROAD_MAX_GAP_BP = MATCHING_DEFAULT_BROAD_MAX_GAP_BP
_MATCHING_DEFAULT_BROAD_MIN_PEAK_BP = MATCHING_DEFAULT_BROAD_MIN_PEAK_BP
_MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER = (
    MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER
)
_MATCHING_DEFAULT_METADATA_DETAIL = MATCHING_DEFAULT_METADATA_DETAIL
_MATCHING_DEFAULT_MIN_PEAK_SCORE = MATCHING_DEFAULT_MIN_PEAK_SCORE
_MATCHING_DEFAULT_PEAK_MODE = MATCHING_DEFAULT_PEAK_MODE
_OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES = OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES
_NESTED_ROCCO_ITERS_DEFAULT = NESTED_ROCCO_ITERS_DEFAULT
_NESTED_ROCCO_JACCARD_DEFAULT = NESTED_ROCCO_JACCARD_DEFAULT
_NESTED_ROCCO_MIN_PARENT_STEPS = NESTED_ROCCO_MIN_PARENT_STEPS
_NESTED_ROCCO_MIN_CHILD_STEPS = NESTED_ROCCO_MIN_CHILD_STEPS
_NESTED_ROCCO_BUDGET_SCALE_DEFAULT = NESTED_ROCCO_BUDGET_SCALE_DEFAULT
_NESTED_ROCCO_SUBTASK_MAX_ITER = NESTED_ROCCO_SUBTASK_MAX_ITER
_NESTED_ROCCO_BUDGET_POLICY = "soft_selection_penalty"
_NESTED_ROCCO_PARENT_EDGE_COST = 1.0e-12
_BROAD_BRIDGE_DIP_PENALTY_FRACTION = (
    MATCHING_DEFAULT_BROAD_BRIDGE_DIP_PENALTY_FRACTION
)
_EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER = (
    EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
)
_MASSIVE_SUBPEAK_CLEANUP_DEFAULT = MASSIVE_SUBPEAK_CLEANUP_DEFAULT
_MASSIVE_SUBPEAK_MIN_BP = MASSIVE_SUBPEAK_MIN_BP
_MASSIVE_SUBPEAK_WIDTH_ALPHA = MASSIVE_SUBPEAK_WIDTH_ALPHA
_MASSIVE_SUBPEAK_WIDTH_BULK_QUANTILE = MASSIVE_SUBPEAK_WIDTH_BULK_QUANTILE
_MASSIVE_SUBPEAK_MAX_FRACTION = MASSIVE_SUBPEAK_MAX_FRACTION
_MASSIVE_SUBPEAK_MIN_LOG_GAP = MASSIVE_SUBPEAK_MIN_LOG_GAP
_MASSIVE_SUBPEAK_MIN_PEAKS = MASSIVE_SUBPEAK_MIN_PEAKS
_MASSIVE_SUBPEAK_SPLIT_QUANTILE = MASSIVE_SUBPEAK_SPLIT_QUANTILE
_MASSIVE_SUBPEAK_SPLIT_Z = MASSIVE_SUBPEAK_SPLIT_Z
_MASSIVE_SUBPEAK_MAX_DEPTH = MASSIVE_SUBPEAK_MAX_DEPTH
_MASSIVE_SUBPEAK_MIN_CHILD_BP = MASSIVE_SUBPEAK_MIN_CHILD_BP
_MASSIVE_SUBPEAK_MIN_CHILD_FRACTION = MASSIVE_SUBPEAK_MIN_CHILD_FRACTION
_MASSIVE_SUBPEAK_TRIGGER_Z_CAP = 3.0
_DWB_PEAK_SCORING_MAX_REPLAYS = 48
_DWB_PEAK_SCORING_MAX_SEGMENTS = 10000
_DWB_PEAK_SCORING_MAX_SEGMENTS_PER_VIEW = 500


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


def _validateMinPeakScore(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("`minPeakScore` must be a finite number or None")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("`minPeakScore` must be a finite number or None") from exc
    if not np.isfinite(score):
        raise ValueError("`minPeakScore` must be a finite number or None")
    return score


def _normalizeUncertaintyScoreMode(value: str | None) -> str:
    return _sharedNormalizeUncertaintyScoreMode(
        value,
        config_name="uncertaintyScoreMode",
        allow_consenrich_state_alias=True,
    )


def _validateUncertaintyScoreZ(value: float) -> float:
    return _sharedValidateUncertaintyScoreZ(
        value,
        config_name="uncertaintyScoreZ",
    )


def _normalizeRoccoMetadataDetail(value: str | None) -> str:
    raw = _MATCHING_DEFAULT_METADATA_DETAIL if value is None else value
    key = str(raw).strip().lower().replace("-", "_")
    if key in {"compact", "summary", "summarized", "summarised"}:
        return "compact"
    if key in {"full", "all", "verbose"}:
        return "full"
    supported = ", ".join(MATCHING_METADATA_DETAILS)
    raise ValueError(
        f"Unsupported metadataDetail {value!r}; supported values: {supported}."
    )


def _normalizeRoccoPeakMode(value: str | None) -> str:
    raw = _MATCHING_DEFAULT_PEAK_MODE if value is None else value
    peakMode = str(raw)
    if peakMode in MATCHING_PEAK_MODES:
        return peakMode
    supported = ", ".join(MATCHING_PEAK_MODES)
    raise ValueError(f"Unsupported peakMode {value!r}. Supported values: {supported}.")


def _validateBroadWeakThresholdZ(value: float) -> float:
    thresholdZ = float(value)
    if not np.isfinite(thresholdZ) or thresholdZ < 0.0:
        raise ValueError("`broadWeakThresholdZ` must be finite and non-negative")
    return thresholdZ


def _validateBroadMaxGapBP(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("`broadMaxGapBP` must be an integer or None")
    gapBP = int(value)
    if gapBP != value or gapBP < 0:
        raise ValueError("`broadMaxGapBP` must be a non-negative integer or None")
    return gapBP


def _readBlacklistIntervalsByChrom(blacklistBedFile: str | None) -> Dict[str, np.ndarray]:
    if blacklistBedFile is None:
        return {}
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


def _filterNarrowPeakRowsByBlacklist(
    rows: List[List[str | int | float]],
    rowsMeta: List[Dict[str, Any]],
    blacklistByChrom: Mapping[str, np.ndarray],
) -> tuple[List[List[str | int | float]], List[Dict[str, Any]], int]:
    if not blacklistByChrom:
        return list(rows), list(rowsMeta), 0
    keptRows: List[List[str | int | float]] = []
    keptMeta: List[Dict[str, Any]] = []
    dropped = 0
    for row, meta in zip(rows, rowsMeta):
        if _intervalOverlapsBlacklist(str(row[0]), int(row[1]), int(row[2]), blacklistByChrom):
            dropped += 1
            continue
        keptRows.append(row)
        keptMeta.append(meta)
    return keptRows, keptMeta, int(dropped)


def _thresholdZKey(thresholdZ: float) -> str:
    z = float(thresholdZ)
    return f"{z:.6f}".rstrip("0").rstrip(".")


def _resolveThresholdZGrid(
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    thresholdZGrid: Iterable[float] | None = None,
) -> List[float]:
    values: List[float] = []
    seen: set[str] = set()
    for value in [
        thresholdZ,
        *(_ROCCO_BUDGET_Z_GRID if thresholdZGrid is None else thresholdZGrid),
    ]:
        value_ = float(max(float(value), 0.0))
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
    returnDetails: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    r"""Build the Consenrich-derived ROCCO score track."""
    state_ = _asFloatVector("state", state)
    mode = _normalizeUncertaintyScoreMode(uncertaintyScoreMode)
    z_ = _validateUncertaintyScoreZ(uncertaintyScoreZ)
    scoreTrack = state_
    seMode = "none"
    uncertaintyAvailable = False
    uncertaintyUsed = False
    uncertaintyMin = None
    uncertaintyMedian = None
    uncertaintyMax = None
    lowerConfidenceScoreFloor = None
    lowerConfidenceScoreFloorHits = 0
    if uncertainty is not None:
        uncertainty_ = _asFloatVector("uncertainty", uncertainty)
        if uncertainty_.size != state_.size:
            raise ValueError("`uncertainty` must match `state` length")
        seMode = "ignored"
        uncertaintyAvailable = True
        uncertaintyMin = float(np.min(uncertainty_))
        uncertaintyMedian = float(np.median(uncertainty_))
        uncertaintyMax = float(np.max(uncertainty_))
    elif mode == "lower_confidence":
        raise ValueError(
            "`lower_confidence` uncertaintyScoreMode requires `uncertainty`"
        )

    if mode == "lower_confidence":
        if np.any(uncertainty_ < 0.0):
            raise ValueError("`uncertainty` must be non-negative for lower_confidence")
        rawScore = state_ - z_ * uncertainty_
        maxState = float(np.max(state_))
        if np.isfinite(maxState) and maxState > 0.0:
            lowerConfidenceScoreFloor = float(-2.0 * maxState)
            scoreTrack = np.maximum(rawScore, lowerConfidenceScoreFloor)
            lowerConfidenceScoreFloorHits = int(
                np.sum(rawScore < lowerConfidenceScoreFloor)
            )
        else:
            scoreTrack = rawScore
        seMode = "used"
        uncertaintyUsed = True

    if not returnDetails:
        return scoreTrack

    details: Dict[str, Any] = {
        "score_mode": "consenrich_state" if mode == "state" else "lower_confidence",
        "uncertainty_score_mode": str(mode),
        "uncertainty_score_z": float(z_),
        "se_mode": str(seMode),
        "uncertainty_available": bool(uncertaintyAvailable),
        "uncertainty_used": bool(uncertaintyUsed),
        "state_median": float(np.median(state_)),
        "state_abs_median": float(np.median(np.abs(state_))),
        "state_min": float(np.min(state_)),
        "state_max": float(np.max(state_)),
        "score_median": float(np.median(scoreTrack)),
        "score_min": float(np.min(scoreTrack)),
        "score_max": float(np.max(scoreTrack)),
        "uncertainty_min": uncertaintyMin,
        "uncertainty_median": uncertaintyMedian,
        "uncertainty_max": uncertaintyMax,
        "lower_confidence_score_floor": lowerConfidenceScoreFloor,
        "lower_confidence_score_floor_hits": int(lowerConfidenceScoreFloorHits),
    }
    return scoreTrack, details


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
    z = _asFloatVector("scoreTrack", scoreTrack)
    support, supportMeta = _selectRobustNullSupport(
        z,
        bulkQuantile=bulkQuantile,
    )

    nullCenter = float(supportMeta["provisional_center"])
    centeredSupport = support - nullCenter
    scaleBase = centeredSupport
    scaleMethod = "central_support_mad"

    scaleMad = 1.4826 * float(np.median(np.abs(scaleBase - np.median(scaleBase))))
    scaleIqr = (
        float(stats.iqr(scaleBase, rng=(25, 75))) / 1.349
        if scaleBase.size >= 4
        else 0.0
    )
    scaleStd = float(np.std(scaleBase, ddof=1)) if scaleBase.size >= 2 else 0.0
    nullScale = float(max(scaleMad, scaleIqr, scaleStd, 1.0e-6))

    details: Dict[str, float | str] = {
        "null_method": str(supportMeta["null_method"]),
        "scale_method": str(scaleMethod),
        "support_size": int(supportMeta["support_size"]),
        "lower_bulk_size": int(supportMeta["lower_bulk_size"]),
        "bulk_quantile": float(supportMeta["bulk_quantile"]),
        "bulk_cutoff": float(supportMeta["bulk_cutoff"]),
        "provisional_center": float(supportMeta["provisional_center"]),
        "center_method": str(supportMeta["center_method"]),
        "support_radius": float(supportMeta["support_radius"]),
        "support_scale": float(supportMeta["support_scale"]),
        "null_center": float(nullCenter),
        "null_scale": float(nullScale),
        "null_center_shift_from_zero": float(nullCenter),
        "support_fraction": float(supportMeta["support_size"] / max(int(z.size), 1)),
    }
    return nullCenter, nullScale, details


def _estimateTemplateScale(template: np.ndarray) -> float:
    centeredTemplate = _asFloatVector("template", template)
    scaleMad = 1.4826 * float(
        np.median(np.abs(centeredTemplate - np.median(centeredTemplate)))
    )
    scaleIqr = (
        float(stats.iqr(centeredTemplate, rng=(25, 75))) / 1.349
        if centeredTemplate.size >= 4
        else 0.0
    )
    scaleStd = (
        float(np.std(centeredTemplate, ddof=1)) if centeredTemplate.size >= 2 else 0.0
    )
    return float(max(scaleMad, scaleIqr, scaleStd, 1.0e-6))


def _calibrateStationaryNullDWB(
    scoreTrack: np.ndarray,
    template: np.ndarray,
    nullCenter: float,
    nullScale: float,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    thresholdZGrid: Iterable[float] | None = None,
    dependenceSpan: int | None = None,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    kernel: str = "bartlett",
    randomSeed: int = 0,
    calibrationQuantile: float = _ROCCO_NULL_QUANTILE,
    pooledNullFloor: Dict[str, Any] | None = None,
    templateMeta: Dict[str, Any] | None = None,
    coreMeta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scoreTrack_ = _asFloatVector("scoreTrack", scoreTrack)
    template_ = _asFloatVector("template", template)
    zGrid = _resolveThresholdZGrid(
        thresholdZ=thresholdZ,
        thresholdZGrid=thresholdZGrid,
    )
    dependenceSpanDetails = _resolveRoccoDependenceSpanDetails(
        scoreTrack_,
        dependenceSpan=dependenceSpan,
    )
    dependenceSpan_ = int(dependenceSpanDetails["point"])
    numBootstrap_ = max(int(numBootstrap), 8)
    calibrationQuantile_ = float(np.clip(calibrationQuantile, 0.50, 0.999))
    kernel_ = str(kernel)
    panelId = (
        f"stationary_null_dwb:{int(randomSeed)}:{int(dependenceSpan_)}:"
        f"{int(numBootstrap_)}:{kernel_}:{int(template_.size)}"
    )
    rng = np.random.default_rng(int(randomSeed))
    upperTailOffsets: Dict[str, np.ndarray] = {
        _thresholdZKey(z): np.empty(numBootstrap_, dtype=np.float64) for z in zGrid
    }

    for b in range(numBootstrap_):
        draw = np.asarray(
            cconsenrich.cStationaryNullDWBDraw(
                template_,
                dependenceSpan_,
                rng,
                kernel_,
            ),
            dtype=np.float64,
        )
        for z in zGrid:
            tailAlpha = float(stats.norm.sf(float(max(z, 0.0))))
            tailQuantile = 1.0 - tailAlpha if float(z) > 0.0 else 0.5
            upperTailOffsets[_thresholdZKey(z)][b] = float(
                np.quantile(
                    draw,
                    tailQuantile,
                    method="interpolated_inverted_cdf",
                )
            )

    thresholdViews: Dict[str, Dict[str, Any]] = {}
    thresholdMetrics: Dict[str, Dict[str, Any]] = {}
    nullOccByKey: Dict[str, np.ndarray] = {
        _thresholdZKey(z): np.empty(numBootstrap_, dtype=np.float64) for z in zGrid
    }
    nullSoftByKey: Dict[str, np.ndarray] = {
        _thresholdZKey(z): np.empty(numBootstrap_, dtype=np.float64) for z in zGrid
    }
    coreMeta_ = {} if coreMeta is None else dict(coreMeta)
    templateMeta_ = {} if templateMeta is None else dict(templateMeta)
    for z in zGrid:
        key = _thresholdZKey(z)
        tailAlpha = float(stats.norm.sf(float(max(z, 0.0))))
        empiricalUpper = float(
            np.quantile(
                upperTailOffsets[key],
                calibrationQuantile_,
                method="interpolated_inverted_cdf",
            )
        )
        pooledView = _resolvePooledNullFloorView(pooledNullFloor, z)
        pooledThresholdFloor = 0.0
        pooledScaleFloor = 0.0
        pooledSource = "none"
        if pooledView is not None:
            pooledThresholdFloor = float(
                max(pooledView.get("threshold_offset_floor", 0.0), 0.0)
            )
            pooledScaleFloor = float(max(pooledView.get("null_scale_floor", 0.0), 0.0))
            pooledSource = str(pooledView.get("source", "external_floor"))

        z_ = float(max(z, 0.0))
        thresholdOffset = float(max(empiricalUpper, pooledThresholdFloor, 0.0))
        empiricalScale = (
            float(max(float(nullScale), thresholdOffset / float(z), 1.0e-6))
            if float(z) > 0.0
            else float(max(float(nullScale), thresholdOffset, 1.0e-6))
        )
        threshold = float(nullCenter + thresholdOffset)
        nullScale_ = float(max(empiricalScale, pooledScaleFloor, 1.0e-6))
        observedExcess = np.clip(
            (scoreTrack_ - threshold) / max(nullScale_, _TINY),
            0.0,
            None,
        )
        nullMeta = {
            **templateMeta_,
            "null_method": "stationary_null_dwb",
            "null_calibration_method": "stationary_null_dwb",
            "core_null_method": str(coreMeta_.get("null_method", "unknown")),
            "core_scale_method": str(coreMeta_.get("scale_method", "unknown")),
            "core_null_scale": float(nullScale),
            "threshold_offset_local": float(empiricalUpper),
            "threshold_offset_floor": float(pooledThresholdFloor),
            "threshold_offset": float(thresholdOffset),
            "threshold_offset_local_empirical": float(empiricalUpper),
            "null_scale_local": float(empiricalScale),
            "null_scale_floor": float(pooledScaleFloor),
            "null_scale": float(nullScale_),
            "pooled_floor_source": str(pooledSource),
            "pooled_floor_applied": bool(
                (pooledThresholdFloor > empiricalUpper + 1.0e-12)
                or (pooledScaleFloor > empiricalScale + 1.0e-12)
            ),
            "threshold": float(threshold),
            "threshold_z": float(z_),
            "null_center": float(nullCenter),
            "tail_method": "stationary_null_dwb",
            "tail_alpha": float(tailAlpha),
            "bootstrap_quantile": float(calibrationQuantile_),
            "num_bootstrap": int(numBootstrap_),
            "dependence_span": int(dependenceSpan_),
            "dwb_bandwidth": int(dependenceSpan_),
            "dwb_bandwidth_lower": int(dependenceSpanDetails["lower"]),
            "dwb_bandwidth_upper": int(dependenceSpanDetails["upper"]),
            "dwb_bandwidth_method": str(dependenceSpanDetails["method"]),
            "dependence_span_lower": int(dependenceSpanDetails["lower"]),
            "dependence_span_upper": int(dependenceSpanDetails["upper"]),
            "dependence_span_method": str(dependenceSpanDetails["method"]),
            "context_span_lower": int(dependenceSpanDetails["lower"]),
            "context_span_upper": int(dependenceSpanDetails["upper"]),
            "context_span_method": str(dependenceSpanDetails["method"]),
            "kernel": str(kernel_),
            "dwb_panel_id": str(panelId),
            "dwb_panel_reused": True,
            "bootstrap_upper_tail_offset": float(empiricalUpper),
            "upper_tail_offset_mean": float(np.mean(upperTailOffsets[key])),
            "upper_tail_offset_sd": (
                float(np.std(upperTailOffsets[key], ddof=1))
                if numBootstrap_ > 1
                else 0.0
            ),
            "threshold_offset": float(thresholdOffset),
            "empirical_null_scale": float(empiricalScale),
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
            "null_center": float(nullCenter),
            "null_scale": float(nullScale_),
            "observed_tail_occupancy": float(np.mean(scoreTrack_ > threshold)),
            "observed_soft_tail": float(np.mean(observedExcess)),
            "budget_occupancy_raw": 0.0,
            "budget_soft_raw": 0.0,
            "null_meta": dict(nullMeta),
            "template_meta": dict(templateMeta_),
            "dwb_panel_id": str(panelId),
        }

    rng = np.random.default_rng(int(randomSeed))
    for b in range(numBootstrap_):
        draw = np.asarray(
            cconsenrich.cStationaryNullDWBDraw(
                template_,
                dependenceSpan_,
                rng,
                kernel_,
            ),
            dtype=np.float64,
        )
        for z in zGrid:
            key = _thresholdZKey(z)
            view = thresholdViews[key]
            thresholdOffset = float(view["threshold"]) - float(nullCenter)
            nullScale_ = float(view["null_scale"])
            nullOccByKey[key][b] = float(np.mean(draw > thresholdOffset))
            nullSoftByKey[key][b] = float(
                np.mean(
                    np.clip(
                        (draw - thresholdOffset) / max(nullScale_, _TINY),
                        0.0,
                        None,
                    )
                )
            )

    for z in zGrid:
        key = _thresholdZKey(z)
        metrics = thresholdMetrics[key]
        nullOcc = np.asarray(nullOccByKey[key], dtype=np.float64)
        nullSoft = np.asarray(nullSoftByKey[key], dtype=np.float64)
        nullOccCal = float(
            np.quantile(
                nullOcc, calibrationQuantile_, method="interpolated_inverted_cdf"
            )
        )
        nullSoftCal = float(
            np.quantile(
                nullSoft, calibrationQuantile_, method="interpolated_inverted_cdf"
            )
        )
        observedTailOccupancy = float(metrics["observed_tail_occupancy"])
        nullTailOccupancyCalibrated = float(nullOccCal)
        budgetOccupancyRaw = observedTailOccupancy - nullTailOccupancyCalibrated
        if not np.isfinite(budgetOccupancyRaw):
            budgetOccupancyRaw = 0.0
        budgetOccupancyRaw = float(max(budgetOccupancyRaw, 0.0))
        metrics.update(
            {
                "null_tail_occupancy": float(np.mean(nullOcc)),
                "null_tail_occupancy_calibrated": float(nullOccCal),
                "null_tail_occupancy_sd": (
                    float(np.std(nullOcc, ddof=1)) if numBootstrap_ > 1 else 0.0
                ),
                "null_soft_tail": float(np.mean(nullSoft)),
                "null_soft_tail_calibrated": float(nullSoftCal),
                "null_soft_tail_sd": (
                    float(np.std(nullSoft, ddof=1)) if numBootstrap_ > 1 else 0.0
                ),
                "budget_occupancy_raw": float(budgetOccupancyRaw),
                "budget_soft_raw": float(
                    np.clip(
                        float(metrics["observed_soft_tail"]) - nullSoftCal, 0.0, 1.0
                    )
                ),
                "null_quantile": float(calibrationQuantile_),
            }
        )

    return {
        "method": "stationary_null_dwb",
        "null_calibration_method": "stationary_null_dwb",
        "kernel": str(kernel_),
        "num_bootstrap": int(numBootstrap_),
        "null_quantile": float(calibrationQuantile_),
        "dependence_span": int(dependenceSpan_),
        "dwb_bandwidth": int(dependenceSpan_),
        "dwb_bandwidth_lower": int(dependenceSpanDetails["lower"]),
        "dwb_bandwidth_upper": int(dependenceSpanDetails["upper"]),
        "dwb_bandwidth_method": str(dependenceSpanDetails["method"]),
        "dependence_span_lower": int(dependenceSpanDetails["lower"]),
        "dependence_span_upper": int(dependenceSpanDetails["upper"]),
        "dependence_span_method": str(dependenceSpanDetails["method"]),
        "context_span_lower": int(dependenceSpanDetails["lower"]),
        "context_span_upper": int(dependenceSpanDetails["upper"]),
        "context_span_method": str(dependenceSpanDetails["method"]),
        "dwb_panel_id": str(panelId),
        "dwb_panel_reused": True,
        "threshold_views": thresholdViews,
        "threshold_metrics": thresholdMetrics,
        "budget_z_grid": [float(z) for z in zGrid],
        "primary_key": _thresholdZKey(float(thresholdZ)),
    }


def _estimateEmpiricalMirroredNullForROCCO(
    scoreTrack: np.ndarray,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    bulkQuantile: float = 0.60,
    pooledNullFloor: Dict[str, Any] | None = None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    nullCenter, coreNullScale, coreMeta = estimateROCCONull(
        scoreTrack,
        bulkQuantile=bulkQuantile,
    )
    template, templateMeta = _prepareNullResidualTemplate(
        scoreTrack,
        nullCenter,
        coreNullScale,
        bulkQuantile=bulkQuantile,
    )
    calibration = _calibrateStationaryNullDWB(
        scoreTrack,
        template,
        nullCenter,
        coreNullScale,
        thresholdZ=thresholdZ,
        thresholdZGrid=[thresholdZ],
        randomSeed=0,
        calibrationQuantile=_ROCCO_NULL_QUANTILE,
        pooledNullFloor=pooledNullFloor,
        templateMeta=templateMeta,
        coreMeta=coreMeta,
    )
    key = _thresholdZKey(float(thresholdZ))
    view = dict(calibration["threshold_views"][key])
    metrics = dict(calibration["threshold_metrics"][key])
    details: Dict[str, Any] = {
        **dict(view["null_meta"]),
        **dict(metrics),
    }
    nullScale = float(view["null_scale"])
    threshold = float(view["threshold"])
    return nullCenter, nullScale, threshold, details


def _isAutosomeName(chromosome: str) -> bool:
    chrom = str(chromosome).strip().lower()
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    if not chrom.isdigit():
        return False
    chromNum = int(chrom)
    return 1 <= chromNum <= 22


def _prepareROCCOBaseScore(
    state: npt.ArrayLike,
    uncertainty: npt.ArrayLike | None = None,
    bulkQuantile: float = 0.60,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
) -> Dict[str, Any]:
    state_ = _asFloatVector("state", state)
    stateNullCenter, stateNullScale, stateNullMeta = estimateROCCONull(
        state_,
        bulkQuantile=bulkQuantile,
    )
    scoreTrack, scoreMeta = consenrichStateScoreTrack(
        state_,
        uncertainty=uncertainty,
        uncertaintyScoreMode=uncertaintyScoreMode,
        uncertaintyScoreZ=uncertaintyScoreZ,
        returnDetails=True,
    )
    return {
        "score_track": np.asarray(scoreTrack, dtype=np.float64),
        "state_null_center": float(stateNullCenter),
        "state_null_scale": float(stateNullScale),
        "score_meta": dict(scoreMeta),
        "state_null_meta": dict(stateNullMeta),
    }


def _estimateAutosomalNullFloorForROCCO(
    basePreparedByChrom: Dict[str, Dict[str, Any]],
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    bulkQuantile: float = 0.60,
    thresholdZGrid: Iterable[float] | None = None,
) -> Dict[str, Any]:
    if len(basePreparedByChrom) == 0:
        return {
            "source": "none",
            "threshold_offset_floor": 0.0,
            "null_scale_floor": 0.0,
            "chromosome_count": 0,
            "tail_support_size": 0,
            "threshold_views": {},
            "budget_z_grid": [],
        }

    autosomeChroms = [chrom for chrom in basePreparedByChrom if _isAutosomeName(chrom)]
    selectedChroms = (
        autosomeChroms if len(autosomeChroms) > 0 else list(basePreparedByChrom)
    )
    pooledTemplateParts: List[np.ndarray] = []
    for chrom in selectedChroms:
        scoreTrack = np.asarray(
            basePreparedByChrom[chrom]["score_track"], dtype=np.float64
        )
        nullCenter, coreNullScale, _ = estimateROCCONull(
            scoreTrack,
            bulkQuantile=bulkQuantile,
        )
        template, _ = _prepareNullResidualTemplate(
            scoreTrack,
            nullCenter,
            coreNullScale,
            bulkQuantile=bulkQuantile,
        )
        pooledTemplateParts.append(np.asarray(template, dtype=np.float64))

    pooledTemplate = np.concatenate(pooledTemplateParts).astype(np.float64, copy=False)
    pooledScale = _estimateTemplateScale(pooledTemplate)
    zGrid = _resolveThresholdZGrid(
        thresholdZ=thresholdZ,
        thresholdZGrid=thresholdZGrid,
    )
    calibration = _calibrateStationaryNullDWB(
        pooledTemplate,
        pooledTemplate,
        0.0,
        pooledScale,
        thresholdZ=thresholdZ,
        thresholdZGrid=zGrid,
        randomSeed=0,
        calibrationQuantile=_ROCCO_NULL_QUANTILE,
        templateMeta={"tail_support_size": int(pooledTemplate.size)},
        coreMeta={
            "null_method": "pooled_template",
            "scale_method": "pooled_template_scale",
        },
    )
    thresholdViews: Dict[str, Dict[str, Any]] = {}
    primaryKey = _thresholdZKey(float(thresholdZ))
    primaryMeta: Dict[str, Any] | None = None
    for z in zGrid:
        pooledMeta = dict(
            calibration["threshold_views"][_thresholdZKey(z)]["null_meta"]
        )
        thresholdOffsetFloor = float(pooledMeta["threshold_offset"])
        scaleFloor = float(pooledMeta["null_scale"])
        meta = {
            "threshold_z": float(z),
            "threshold_offset_floor": float(thresholdOffsetFloor),
            "null_scale_floor": float(scaleFloor),
            **pooledMeta,
        }
        thresholdViews[_thresholdZKey(z)] = meta
        if _thresholdZKey(z) == primaryKey:
            primaryMeta = meta

    if primaryMeta is None:
        primaryMeta = thresholdViews[_thresholdZKey(zGrid[0])]
    return {
        "source": (
            "autosomal_pool" if len(autosomeChroms) > 0 else "all_chromosomes_pool"
        ),
        "chromosome_count": int(len(selectedChroms)),
        "tail_support_size": int(pooledTemplate.size),
        "threshold_offset_floor": float(primaryMeta["threshold_offset_floor"]),
        "null_scale_floor": float(primaryMeta["null_scale_floor"]),
        "threshold_z": float(primaryMeta["threshold_z"]),
        "threshold_views": thresholdViews,
        "budget_z_grid": [float(z) for z in zGrid],
        "chromosomes": [str(chrom) for chrom in selectedChroms],
        **primaryMeta,
    }


def _resolvePooledNullFloorView(
    pooledNullFloor: Dict[str, Any] | None,
    thresholdZ: float,
) -> Dict[str, Any] | None:
    if pooledNullFloor is None:
        return None

    thresholdViews = pooledNullFloor.get("threshold_views")
    if isinstance(thresholdViews, dict):
        match = thresholdViews.get(_thresholdZKey(thresholdZ))
        if isinstance(match, dict):
            return dict(match)

    return {
        "source": str(pooledNullFloor.get("source", "external_floor")),
        "threshold_offset_floor": float(
            max(pooledNullFloor.get("threshold_offset_floor", 0.0), 0.0)
        ),
        "null_scale_floor": float(
            max(pooledNullFloor.get("null_scale_floor", 0.0), 0.0)
        ),
    }


def _resolveRoccoDependenceSpanDetails(
    values: np.ndarray,
    dependenceSpan: int | None = None,
) -> Dict[str, int | str]:
    if dependenceSpan is not None:
        span = max(int(dependenceSpan), 2)
        return {
            "point": int(span),
            "lower": int(span),
            "upper": int(span),
            "method": "fixed",
        }

    n = int(values.size)
    try:
        if n >= 100:
            positiveVals = np.clip(values, 0.0, None)
            contextSize, contextLower, contextUpper, featureDetails = (
                core.chooseFeatureLength(
                    positiveVals,
                    minSpan=3,
                    maxSpan=min(64, max(12, n // 8)),
                )
            )
            point = max(int(contextSize), 2)
            lower = max(min(int(contextLower), point), 2)
            upper = max(int(contextUpper), point)
            return {
                "point": int(point),
                "lower": int(lower),
                "upper": int(upper),
                "method": str(featureDetails.get("method", "chooseFeatureLength")),
            }
    except Exception as ex:
        logger.info("chooseFeatureLength fallback for ROCCO budget span: %s", ex)

    fallback = max(min(int(round(np.sqrt(n))), 64), 4)
    return {
        "point": int(fallback),
        "lower": int(fallback),
        "upper": int(fallback),
        "method": "sqrt_fallback",
    }


def _prepareNullResidualTemplate(
    scoreTrack: np.ndarray,
    nullCenter: float,
    nullScale: float,
    bulkQuantile: float = 0.60,
) -> Tuple[np.ndarray, Dict[str, float]]:
    centered = np.asarray(scoreTrack, dtype=np.float64) - float(nullCenter)
    support, supportMeta = _selectRobustNullSupport(
        np.asarray(scoreTrack, dtype=np.float64),
        bulkQuantile=bulkQuantile,
    )
    bulkVals = np.asarray(support, dtype=np.float64) - float(nullCenter)
    if bulkVals.size < 4:
        bulkVals = centered

    clipAbs = float(
        max(
            (
                np.quantile(
                    np.abs(bulkVals),
                    0.95,
                    method="interpolated_inverted_cdf",
                )
                if bulkVals.size > 0
                else 0.0
            ),
            nullScale,
            1.0e-6,
        )
    )

    template = np.clip(centered, -clipAbs, clipAbs).astype(np.float64, copy=False)
    template = template - float(np.mean(template))
    templateStd = float(np.std(template, ddof=1)) if template.size >= 2 else 0.0
    if np.isfinite(templateStd) and templateStd > _TINY:
        template = template * (float(nullScale) / templateStd)
    else:
        template = np.full_like(template, 0.0, dtype=np.float64)

    return template, {
        "clip_abs": float(clipAbs),
        "template_std": float(np.std(template, ddof=1)) if template.size >= 2 else 0.0,
        "template_mean": float(np.mean(template)),
        "template_support_radius": float(supportMeta["support_radius"]),
        "template_support_size": float(supportMeta["support_size"]),
    }


def _prepareROCCOScoreAndNull(
    state: npt.ArrayLike,
    uncertainty: npt.ArrayLike | None = None,
    bulkQuantile: float = 0.60,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    dependenceSpan: int | None = None,
    kernel: str = "bartlett",
    randomSeed: int = 0,
    nullQuantile: float = _ROCCO_NULL_QUANTILE,
    pooledNullFloor: Dict[str, Any] | None = None,
    thresholdZGrid: Iterable[float] | None = None,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
) -> Dict[str, Any]:
    r"""Prepare the direct Consenrich score track and robust null for ROCCO budgeting."""
    prepared = _prepareROCCOBaseScore(
        state,
        uncertainty=uncertainty,
        bulkQuantile=bulkQuantile,
        uncertaintyScoreMode=uncertaintyScoreMode,
        uncertaintyScoreZ=uncertaintyScoreZ,
    )
    scoreTrack = np.asarray(prepared["score_track"], dtype=np.float64)
    zGrid = _resolveThresholdZGrid(
        thresholdZ=thresholdZ,
        thresholdZGrid=thresholdZGrid,
    )
    nullCenter, coreNullScale, coreMeta = estimateROCCONull(
        scoreTrack,
        bulkQuantile=bulkQuantile,
    )
    template, templateMeta = _prepareNullResidualTemplate(
        scoreTrack,
        nullCenter,
        coreNullScale,
        bulkQuantile=bulkQuantile,
    )
    calibration = _calibrateStationaryNullDWB(
        scoreTrack,
        template,
        nullCenter,
        coreNullScale,
        thresholdZ=thresholdZ,
        thresholdZGrid=zGrid,
        dependenceSpan=dependenceSpan,
        numBootstrap=numBootstrap,
        kernel=kernel,
        randomSeed=randomSeed,
        calibrationQuantile=nullQuantile,
        pooledNullFloor=pooledNullFloor,
        templateMeta=templateMeta,
        coreMeta=coreMeta,
    )
    thresholdViews = dict(calibration["threshold_views"])
    primaryKey = str(calibration["primary_key"])
    primaryView = thresholdViews.get(primaryKey)
    if primaryView is None:
        primaryView = thresholdViews[_thresholdZKey(zGrid[0])]
    return {
        **prepared,
        "score_track": scoreTrack,
        "null_center": float(primaryView["null_center"]),
        "null_scale": float(primaryView["null_scale"]),
        "threshold": float(primaryView["threshold"]),
        "null_meta": dict(primaryView["null_meta"]),
        "template": np.asarray(primaryView["template"], dtype=np.float64),
        "template_meta": dict(primaryView["template_meta"]),
        "threshold_views": thresholdViews,
        "threshold_metrics": dict(calibration["threshold_metrics"]),
        "budget_z_grid": [float(z) for z in zGrid],
        "dwb_calibration": {
            "method": str(calibration["method"]),
            "null_calibration_method": str(calibration["null_calibration_method"]),
            "kernel": str(calibration["kernel"]),
            "num_bootstrap": int(calibration["num_bootstrap"]),
            "null_quantile": float(calibration["null_quantile"]),
            "dependence_span": int(calibration["dependence_span"]),
            "dwb_bandwidth": int(calibration["dwb_bandwidth"]),
            "dwb_bandwidth_lower": int(calibration["dwb_bandwidth_lower"]),
            "dwb_bandwidth_upper": int(calibration["dwb_bandwidth_upper"]),
            "dwb_bandwidth_method": str(calibration["dwb_bandwidth_method"]),
            "dependence_span_lower": int(calibration["dependence_span_lower"]),
            "dependence_span_upper": int(calibration["dependence_span_upper"]),
            "dependence_span_method": str(calibration["dependence_span_method"]),
            "context_span_lower": int(calibration["context_span_lower"]),
            "context_span_upper": int(calibration["context_span_upper"]),
            "context_span_method": str(calibration["context_span_method"]),
            "dwb_panel_id": str(calibration["dwb_panel_id"]),
            "dwb_panel_reused": bool(calibration["dwb_panel_reused"]),
            "random_seed": int(randomSeed),
            "primary_key": str(calibration["primary_key"]),
            "threshold_z_grid": [float(z) for z in calibration["budget_z_grid"]],
        },
        "pooled_null_floor": None if pooledNullFloor is None else dict(pooledNullFloor),
    }


def _estimateEffectiveSampleSize(
    values: np.ndarray,
    maxLag: int,
) -> Tuple[float, float, int]:
    n_eff, tau, lags_used = cconsenrich.cEstimateEffectiveSampleSize(
        values,
        int(maxLag),
    )
    return float(n_eff), float(tau), int(lags_used)

def _preparedStationaryNullDWBMatches(
    prepared: Dict[str, Any],
    *,
    scoreTrack: np.ndarray,
    numBootstrap: int,
    dependenceSpan: int | None,
    kernel: str,
    thresholdZ: float,
    thresholdZGrid: Iterable[float] | None,
    randomSeed: int,
    nullQuantile: float,
) -> bool:
    calibration = prepared.get("dwb_calibration")
    if not isinstance(calibration, dict):
        return False

    expectedGrid = _resolveThresholdZGrid(
        thresholdZ=thresholdZ,
        thresholdZGrid=(
            prepared.get("budget_z_grid") if thresholdZGrid is None else thresholdZGrid
        ),
    )
    expectedSpan = _resolveRoccoDependenceSpanDetails(
        scoreTrack,
        dependenceSpan=dependenceSpan,
    )
    return (
        str(calibration.get("method")) == "stationary_null_dwb"
        and int(calibration.get("num_bootstrap", -1)) == max(int(numBootstrap), 8)
        and int(calibration.get("dependence_span", -1)) == int(expectedSpan["point"])
        and str(calibration.get("kernel", "")).strip().lower().replace("-", "_")
        == str(kernel).strip().lower().replace("-", "_")
        and int(calibration.get("random_seed", -1)) == int(randomSeed)
        and float(calibration.get("null_quantile", -1.0))
        == float(np.clip(nullQuantile, 0.50, 0.999))
        and str(calibration.get("primary_key", ""))
        == _thresholdZKey(float(max(thresholdZ, 0.0)))
        and list(calibration.get("threshold_z_grid", []))
        == [float(z) for z in expectedGrid]
    )


def _estimateBudgetForPreparedROCCOScore(
    prepared: Dict[str, Any],
    statistic: str = "occupancy",
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    dependenceSpan: int | None = None,
    kernel: str = "bartlett",
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    bulkQuantile: float = 0.60,
    randomSeed: int = 0,
    nullQuantile: float = _ROCCO_NULL_QUANTILE,
    budgetMin: float = _ROCCO_BUDGET_MIN,
    budgetMax: float = _ROCCO_BUDGET_MAX,
    thresholdZGrid: Iterable[float] | None = None,
    returnDetails: bool = False,
) -> float | Tuple[float, Dict[str, Any]]:
    r"""Estimate a ROCCO budget from a shared stationary-null DWB calibration."""
    scoreTrack = np.asarray(prepared["score_track"], dtype=np.float64)
    scoreMeta = dict(prepared["score_meta"])
    stateNullMeta = dict(prepared["state_null_meta"])
    zGrid = _resolveThresholdZGrid(
        thresholdZ=thresholdZ,
        thresholdZGrid=(
            prepared.get("budget_z_grid") if thresholdZGrid is None else thresholdZGrid
        ),
    )
    numBootstrap_ = max(int(numBootstrap), 16)
    nullQuantile_ = float(np.clip(nullQuantile, 0.50, 0.999))
    budgetMin_ = float(max(budgetMin, 0.0))
    budgetMax_ = float(max(budgetMax, budgetMin_))
    primaryKey = _thresholdZKey(float(max(thresholdZ, 0.0)))
    statistic_ = str(statistic).strip().lower().replace("-", "_")
    if statistic_ in {
        "integrated",
        "integrated_excess",
        "integrated_excess_tail",
        "excess",
    }:
        selectedStatistic = "integrated_excess_tail"
    elif statistic_ in {
        "occupancy",
        "calibrated_occupancy",
        "calibrated_tail_occupancy",
        "tail_fraction",
        "fraction",
        "tail",
        "tail_occupancy",
    }:
        selectedStatistic = "tail_occupancy"
    elif statistic_ in {"soft", "soft_occupancy", "soft_tail"}:
        selectedStatistic = "soft"
    else:
        raise ValueError(f"Unknown budget statistic: {statistic}")

    if _preparedStationaryNullDWBMatches(
        prepared,
        scoreTrack=scoreTrack,
        numBootstrap=numBootstrap,
        dependenceSpan=dependenceSpan,
        kernel=kernel,
        thresholdZ=thresholdZ,
        thresholdZGrid=thresholdZGrid,
        randomSeed=randomSeed,
        nullQuantile=nullQuantile,
    ):
        thresholdViews = dict(prepared.get("threshold_views", {}))
        thresholdMetrics = dict(prepared.get("threshold_metrics", {}))
        calibrationMeta = dict(prepared.get("dwb_calibration", {}))
    else:
        calibration = _calibrateStationaryNullDWB(
            scoreTrack,
            np.asarray(prepared["template"], dtype=np.float64),
            float(prepared["null_center"]),
            float(prepared["null_meta"].get("core_null_scale", prepared["null_scale"])),
            thresholdZ=thresholdZ,
            thresholdZGrid=zGrid,
            dependenceSpan=dependenceSpan,
            numBootstrap=numBootstrap,
            kernel=kernel,
            randomSeed=randomSeed,
            calibrationQuantile=nullQuantile,
            pooledNullFloor=prepared.get("pooled_null_floor"),
            templateMeta=dict(prepared.get("template_meta", {})),
            coreMeta=dict(prepared.get("state_null_meta", {})),
        )
        thresholdViews = dict(calibration["threshold_views"])
        thresholdMetrics = dict(calibration["threshold_metrics"])
        calibrationMeta = {
            "method": str(calibration["method"]),
            "null_calibration_method": str(calibration["null_calibration_method"]),
            "kernel": str(calibration["kernel"]),
            "num_bootstrap": int(calibration["num_bootstrap"]),
            "null_quantile": float(calibration["null_quantile"]),
            "dependence_span": int(calibration["dependence_span"]),
            "dwb_bandwidth": int(calibration["dwb_bandwidth"]),
            "dwb_bandwidth_lower": int(calibration["dwb_bandwidth_lower"]),
            "dwb_bandwidth_upper": int(calibration["dwb_bandwidth_upper"]),
            "dwb_bandwidth_method": str(calibration["dwb_bandwidth_method"]),
            "dependence_span_lower": int(calibration["dependence_span_lower"]),
            "dependence_span_upper": int(calibration["dependence_span_upper"]),
            "dependence_span_method": str(calibration["dependence_span_method"]),
            "context_span_lower": int(calibration["context_span_lower"]),
            "context_span_upper": int(calibration["context_span_upper"]),
            "context_span_method": str(calibration["context_span_method"]),
            "dwb_panel_id": str(calibration["dwb_panel_id"]),
            "dwb_panel_reused": bool(calibration["dwb_panel_reused"]),
            "random_seed": int(randomSeed),
            "primary_key": str(calibration["primary_key"]),
            "threshold_z_grid": [float(z) for z in calibration["budget_z_grid"]],
        }

    primaryView = thresholdViews.get(primaryKey)
    if primaryView is None:
        primaryView = thresholdViews[_thresholdZKey(zGrid[0])]
        primaryKey = _thresholdZKey(zGrid[0])
    primaryMetrics = dict(thresholdMetrics[primaryKey])

    integratedSeriesParts: List[np.ndarray] = []
    for z in zGrid:
        view = thresholdViews[_thresholdZKey(z)]
        integratedSeriesParts.append(
            np.clip(
                (scoreTrack - float(view["threshold"]))
                / max(float(view["null_scale"]), _TINY),
                0.0,
                None,
            )
        )

    budgetIntegratedRaw = float(
        np.mean(
            [float(metrics["budget_soft_raw"]) for metrics in thresholdMetrics.values()]
        )
    )
    primaryBudgetOccupancyRaw = float(primaryMetrics["budget_occupancy_raw"])
    primaryBudgetSoftRaw = float(primaryMetrics["budget_soft_raw"])
    budgetOccupancy = float(np.clip(primaryBudgetOccupancyRaw, budgetMin_, budgetMax_))
    budgetSoft = float(np.clip(primaryBudgetSoftRaw, budgetMin_, budgetMax_))
    budgetIntegrated = float(np.clip(budgetIntegratedRaw, budgetMin_, budgetMax_))

    if selectedStatistic == "tail_occupancy":
        budget = budgetOccupancy
        budgetRaw = primaryBudgetOccupancyRaw
        essSeries = (scoreTrack > float(primaryView["threshold"])).astype(np.float64)
    elif selectedStatistic == "soft":
        budget = budgetSoft
        budgetRaw = primaryBudgetSoftRaw
        essSeries = np.clip(
            (scoreTrack - float(primaryView["threshold"]))
            / max(float(primaryView["null_scale"]), _TINY),
            0.0,
            None,
        )
    else:
        budget = budgetIntegrated
        budgetRaw = budgetIntegratedRaw
        essSeries = (
            np.mean(np.vstack(integratedSeriesParts), axis=0)
            if len(integratedSeriesParts) > 0
            else np.zeros_like(scoreTrack, dtype=np.float64)
        )

    effectiveTotalCount, tauInt, essLagsUsed = _estimateEffectiveSampleSize(
        essSeries,
        maxLag=max(
            4 * int(calibrationMeta["dependence_span"]),
            int(calibrationMeta["dependence_span"]),
        ),
    )

    if not returnDetails:
        return budget

    budgetModel = {
        "integrated_excess_tail": "dwb_integrated_excess_tail",
        "tail_occupancy": "dwb_tail_occupancy",
        "soft": "dwb_soft_tail",
    }[selectedStatistic]
    details: Dict[str, Any] = {
        "method": "stationary_null_dwb",
        "null_calibration_method": "stationary_null_dwb",
        "statistic": str(selectedStatistic),
        "kernel": str(calibrationMeta["kernel"]),
        "num_bootstrap": int(calibrationMeta["num_bootstrap"]),
        "null_quantile": float(nullQuantile_),
        "dependence_span": int(calibrationMeta["dependence_span"]),
        "dwb_bandwidth": int(calibrationMeta["dwb_bandwidth"]),
        "dwb_bandwidth_lower": int(calibrationMeta["dwb_bandwidth_lower"]),
        "dwb_bandwidth_upper": int(calibrationMeta["dwb_bandwidth_upper"]),
        "dwb_bandwidth_method": str(calibrationMeta["dwb_bandwidth_method"]),
        "dependence_span_lower": int(calibrationMeta["dependence_span_lower"]),
        "dependence_span_upper": int(calibrationMeta["dependence_span_upper"]),
        "dependence_span_method": str(calibrationMeta["dependence_span_method"]),
        "context_span_lower": int(calibrationMeta["context_span_lower"]),
        "context_span_upper": int(calibrationMeta["context_span_upper"]),
        "context_span_method": str(calibrationMeta["context_span_method"]),
        "dwb_panel_id": str(calibrationMeta["dwb_panel_id"]),
        "dwb_panel_reused": bool(calibrationMeta["dwb_panel_reused"]),
        "threshold": float(primaryView["threshold"]),
        "threshold_z": float(thresholdZ),
        "threshold_z_grid": [float(z) for z in zGrid],
        "bulk_quantile": float(bulkQuantile),
        "budget_model": str(budgetModel),
        "null_center": float(primaryView["null_center"]),
        "null_scale": float(primaryView["null_scale"]),
        "state_null_center": float(prepared["state_null_center"]),
        "state_null_scale": float(prepared["state_null_scale"]),
        "pooled_null_floor_source": str(
            prepared.get("pooled_null_floor", {}).get("source", "none")
            if prepared.get("pooled_null_floor") is not None
            else "none"
        ),
        "observed_tail_occupancy": float(primaryMetrics["observed_tail_occupancy"]),
        "null_tail_occupancy": float(primaryMetrics["null_tail_occupancy"]),
        "null_tail_occupancy_calibrated": float(
            primaryMetrics["null_tail_occupancy_calibrated"]
        ),
        "null_tail_occupancy_sd": float(primaryMetrics["null_tail_occupancy_sd"]),
        "observed_soft_tail": float(primaryMetrics["observed_soft_tail"]),
        "null_soft_tail": float(primaryMetrics["null_soft_tail"]),
        "null_soft_tail_calibrated": float(primaryMetrics["null_soft_tail_calibrated"]),
        "null_soft_tail_sd": float(primaryMetrics["null_soft_tail_sd"]),
        "budget_min": float(budgetMin_),
        "budget_max": float(budgetMax_),
        "budget_raw": float(budgetRaw),
        "budget_clipped": bool(abs(budget - budgetRaw) > 1.0e-12),
        "budget_occupancy_raw": float(primaryBudgetOccupancyRaw),
        "budget_soft_raw": float(primaryBudgetSoftRaw),
        "budget_integrated_raw": float(budgetIntegratedRaw),
        "budget_occupancy": float(budgetOccupancy),
        "budget_soft": float(budgetSoft),
        "budget_integrated": float(budgetIntegrated),
        "effective_count": float(budget * effectiveTotalCount),
        "effective_total_count": float(effectiveTotalCount),
        "autocorrelation_time": float(tauInt),
        "ess_lags_used": int(essLagsUsed),
        "threshold_metrics": thresholdMetrics,
    }
    details.update(scoreMeta)
    details.update(stateNullMeta)
    details.update(dict(primaryView["null_meta"]))
    details.update(dict(primaryView["template_meta"]))
    return budget, details


def getROCCOBudget(
    state: npt.ArrayLike,
    uncertainty: npt.ArrayLike | None = None,
    statistic: str = "occupancy",
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    dependenceSpan: int | None = None,
    kernel: str = "bartlett",
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    bulkQuantile: float = 0.60,
    randomSeed: int = 0,
    nullQuantile: float = _ROCCO_NULL_QUANTILE,
    pooledNullFloor: Dict[str, Any] | None = None,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    returnDetails: bool = False,
) -> float | Tuple[float, Dict[str, Any]]:
    r"""Estimate a chromosome 'budget' from the fitted Consenrich state."""
    prepared = _prepareROCCOScoreAndNull(
        state,
        uncertainty=uncertainty,
        bulkQuantile=bulkQuantile,
        thresholdZ=thresholdZ,
        numBootstrap=numBootstrap,
        dependenceSpan=dependenceSpan,
        kernel=kernel,
        randomSeed=randomSeed,
        nullQuantile=nullQuantile,
        pooledNullFloor=pooledNullFloor,
        uncertaintyScoreMode=uncertaintyScoreMode,
        uncertaintyScoreZ=uncertaintyScoreZ,
    )
    result = _estimateBudgetForPreparedROCCOScore(
        prepared,
        statistic=statistic,
        numBootstrap=numBootstrap,
        dependenceSpan=dependenceSpan,
        kernel=kernel,
        thresholdZ=thresholdZ,
        bulkQuantile=bulkQuantile,
        randomSeed=randomSeed,
        nullQuantile=nullQuantile,
        returnDetails=returnDetails,
    )
    return result


def shrinkROCCOBudgets(
    effectiveCounts: Dict[str, float],
    effectiveTotals: Dict[str, float],
    posteriorQuantile: float | None = None,
    minPriorConcentration: float = 8.0,
    minBudget: float = 0.0,
    maxBudget: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    r"""Apply a simple beta-binomial EB shrinkage to chromosome-wise budget estimates."""
    chroms = sorted(set(effectiveCounts) & set(effectiveTotals))
    if len(chroms) == 0:
        raise ValueError("No overlapping chromosome keys found.")

    successes = np.asarray(
        [max(float(effectiveCounts[c]), 0.0) for c in chroms],
        dtype=np.float64,
    )
    totals = np.asarray(
        [max(float(effectiveTotals[c]), 1.0) for c in chroms],
        dtype=np.float64,
    )
    successes = np.minimum(successes, totals)
    rawBudgets = successes / np.maximum(totals, 1.0)
    pooledRate = float(np.sum(successes) / np.sum(totals))
    if pooledRate <= 1.0e-12 and float(np.sum(successes)) <= 1.0e-12:
        shrunkZero = {chrom: 0.0 for chrom in chroms}
        return shrunkZero, {
            "genome_wide_budget": 0.0,
            "alpha_hat": 0.0,
            "beta_hat": 1.0,
            "prior_concentration": float(max(minPriorConcentration, 2.0)),
            "min_prior_concentration": float(max(minPriorConcentration, 2.0)),
            "posterior_estimator": "degenerate_zero",
            "posterior_quantile": None,
            "min_budget": float(max(minBudget, 0.0)),
            "max_budget": float(max(maxBudget, max(minBudget, 0.0))),
        }

    if len(chroms) == 1:
        shrunkSingle = {
            chroms[0]: float(
                np.clip(
                    rawBudgets[0],
                    max(minBudget, 0.0),
                    max(maxBudget, max(minBudget, 0.0)),
                )
            )
        }
        return shrunkSingle, {
            "genome_wide_budget": float(pooledRate),
            "alpha_hat": float(max(pooledRate, 1.0e-6)),
            "beta_hat": float(max(1.0 - pooledRate, 1.0e-6)),
            "prior_concentration": 0.0,
            "prior_concentration_raw": 0.0,
            "prior_concentration_cap": 0.0,
            "prior_concentration_capped": False,
            "min_prior_concentration": float(max(minPriorConcentration, 2.0)),
            "posterior_estimator": "none_single_chromosome",
            "posterior_quantile": (
                None
                if posteriorQuantile is None
                else float(np.clip(posteriorQuantile, 1.0e-4, 0.9999))
            ),
            "min_budget": float(max(minBudget, 0.0)),
            "max_budget": float(max(maxBudget, max(minBudget, 0.0))),
        }
    else:
        observedVar = float(np.var(rawBudgets, ddof=1))
        theoreticalMinVar = float(np.mean(pooledRate * (1.0 - pooledRate) / totals))
        excessVar = max(observedVar - theoreticalMinVar, 1.0e-8)
        concentrationRaw = max(
            (pooledRate * (1.0 - pooledRate) / excessVar) - 1.0,
            float(max(minPriorConcentration, 2.0)),
        )
        concentrationCap = float(
            max(
                float(max(minPriorConcentration, 2.0)),
                float(np.median(np.sqrt(np.maximum(totals, 1.0)))),
            )
        )
        concentration = min(concentrationRaw, concentrationCap)

    alphaHat = max(pooledRate * concentration, 1.0e-3)
    betaHat = max((1.0 - pooledRate) * concentration, 1.0e-3)
    q: float | None
    if posteriorQuantile is None:
        q = None
    else:
        q = float(np.clip(posteriorQuantile, 1.0e-4, 0.9999))
    minBudget_ = float(max(minBudget, 0.0))
    maxBudget_ = float(max(maxBudget, minBudget_))

    shrunk: Dict[str, float] = {}
    for idx, chrom in enumerate(chroms):
        if q is None:
            posterior = float(
                (successes[idx] + alphaHat) / max(totals[idx] + alphaHat + betaHat, 1.0)
            )
        else:
            posterior = float(
                stats.beta.ppf(
                    q,
                    successes[idx] + alphaHat,
                    max(totals[idx] - successes[idx], 0.0) + betaHat,
                )
            )
        if not np.isfinite(posterior):
            posterior = pooledRate
        shrunk[chrom] = float(np.clip(posterior, minBudget_, maxBudget_))

    meta = {
        "genome_wide_budget": float(pooledRate),
        "alpha_hat": float(alphaHat),
        "beta_hat": float(betaHat),
        "prior_concentration": float(concentration),
        "prior_concentration_raw": float(concentrationRaw if len(chroms) > 1 else 0.0),
        "prior_concentration_cap": float(concentrationCap if len(chroms) > 1 else 0.0),
        "prior_concentration_capped": bool(
            len(chroms) > 1 and concentration < concentrationRaw - 1.0e-12
        ),
        "min_prior_concentration": float(max(minPriorConcentration, 2.0)),
        "posterior_estimator": "mean" if q is None else "quantile",
        "posterior_quantile": None if q is None else float(q),
        "min_budget": float(minBudget_),
        "max_budget": float(maxBudget_),
    }
    return shrunk, meta


def estimateROCCOGamma(
    scoreTrack: npt.ArrayLike,
    dependenceSpan: int | None = None,
    gammaSpan: int | None = None,
    gamma: float | None = 0.5,
    gammaScale: float = 0.5,
    clipMin: float = 0.5,
    clipMax: float | None = 50.0,
    nullCenter: float | None = None,
    threshold: float | None = None,
    returnDetails: bool = False,
) -> float | Tuple[float, Dict[str, float | str]]:
    r"""Estimate a constant ROCCO boundary penalty from score scale and context size."""
    if gamma is None:
        gamma_ = 0.5
        if not returnDetails:
            return gamma_
        return gamma_, {"method": "fixed", "gamma": float(gamma_)}

    gamma_ = float(gamma)
    if not np.isfinite(gamma_):
        raise ValueError("`gamma` must be finite")
    if gamma_ >= 0.0:
        if not returnDetails:
            return gamma_
        return gamma_, {"method": "fixed", "gamma": float(gamma_)}

    scores = _asFloatVector("scoreTrack", scoreTrack)
    dependenceSpanDetails = _resolveRoccoDependenceSpanDetails(
        scores,
        dependenceSpan=dependenceSpan,
    )
    dependenceSpan_ = int(dependenceSpanDetails["point"])
    gammaSpan_ = (
        max(int(gammaSpan), 2)
        if gammaSpan is not None
        else int(dependenceSpanDetails["lower"])
    )
    referenceLevel = 0.0
    referenceMethod = "zero"
    if threshold is not None and np.isfinite(float(threshold)):
        referenceLevel = float(threshold)
        referenceMethod = "threshold"
    elif nullCenter is not None and np.isfinite(float(nullCenter)):
        referenceLevel = float(nullCenter)
        referenceMethod = "null_center"

    positiveScores = scores - referenceLevel
    positiveScores = positiveScores[positiveScores > 0.0]
    positiveScale = (
        float(
            np.median(
                positiveScores,
            )
        )
        if positiveScores.size > 0
        else 1.0
    )
    gammaRaw = float(max(float(gammaScale), 0.0) * float(gammaSpan_) * positiveScale)
    gamma_ = float(max(gammaRaw, float(max(clipMin, 0.0))))
    if clipMax is not None:
        gamma_ = float(min(gamma_, float(max(clipMax, clipMin))))

    if not returnDetails:
        return gamma_

    details = {
        "method": "dependence_span_lower_score_scale",
        "dependence_span": int(dependenceSpan_),
        "gamma_span": int(gammaSpan_),
        "dependence_span_lower": int(dependenceSpanDetails["lower"]),
        "dependence_span_upper": int(dependenceSpanDetails["upper"]),
        "dependence_span_method": str(dependenceSpanDetails["method"]),
        "context_span_lower": int(dependenceSpanDetails["lower"]),
        "context_span_upper": int(dependenceSpanDetails["upper"]),
        "context_span_method": str(dependenceSpanDetails["method"]),
        "reference_method": str(referenceMethod),
        "reference_level": float(referenceLevel),
        "positive_score_median": float(positiveScale),
        "gamma_scale": float(gammaScale),
        "gamma_raw": float(gammaRaw),
        "gamma": float(gamma_),
    }
    if clipMax is not None:
        details["gamma_clip_max"] = float(max(clipMax, clipMin))
    details["gamma_clip_min"] = float(max(clipMin, 0.0))
    return gamma_, details


def solveChromROCCO(
    scores: npt.ArrayLike,
    budget: float | None = None,
    gamma: float = 0.5,
    selectionPenalty: float | None = None,
    maxIter: int = _ROCCO_MAX_ITER_DEFAULT,
    returnDetails: bool = False,
) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, Dict[str, Any]]:
    scores_ = _asFloatVector("scores", scores)
    gamma_ = float(gamma)
    if not np.isfinite(gamma_) or gamma_ < 0.0:
        raise ValueError("`gamma` must be finite and non-negative")
    solution, objective, penalizedObjective, selectedCount, selectionPenalty_ = (
        cconsenrich.csolveChromROCCOExact(
            scores_,
            budget=budget,
            gamma=gamma_,
            selectionPenalty=selectionPenalty,
            maxIter=int(maxIter),
        )
    )
    budgetFallbackUsed = False
    budgetTargetCount: int | None = None
    if budget is not None and selectionPenalty is None:
        budget_ = float(budget)
        if np.isfinite(budget_):
            budgetTargetCount = int(
                min(max(math.floor(scores_.size * budget_), 0), scores_.size)
            )
        if (
            int(selectedCount) == 0
            and budgetTargetCount is not None
            and budgetTargetCount > 0
        ):
            fallbackMask, fallbackUsed = _bestContiguousBudgetFallbackMask(
                scores_,
                budgetTargetCount,
                0.0,
                gamma_,
                maxRelativeRange=0.05,
            )
            if fallbackUsed:
                solution = fallbackMask.astype(np.uint8)
                selectedCount = int(np.sum(solution))
                objective = _roccoObjectiveForSolution(
                    scores_,
                    np.asarray(solution, dtype=np.uint8),
                    gamma_,
                )
                penalizedObjective = float(objective) - float(
                    selectionPenalty_
                ) * float(selectedCount)
                budgetFallbackUsed = True
    if not returnDetails:
        return np.asarray(solution, dtype=np.uint8), float(objective)

    details = {
        "penalized_objective": float(penalizedObjective),
        "selected_count": int(selectedCount),
        "selected_fraction": float(selectedCount / max(scores_.size, 1)),
        "selection_penalty": float(selectionPenalty_),
        "gamma": float(gamma_),
        "max_iter": int(maxIter),
        "budget_target_count": budgetTargetCount,
        "budget_fallback_window": bool(budgetFallbackUsed),
    }
    return np.asarray(solution, dtype=np.uint8), float(objective), details


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
    if intervals_.size != mask_.size or ends_.size != mask_.size:
        raise ValueError("`intervals`, `ends`, and `mask` must match length")
    runs: List[Tuple[int, int]] = []
    n = int(mask_.size)
    i = 0
    while i < n:
        if not bool(mask_[i]):
            i += 1
            continue
        start = i
        while (
            i + 1 < n
            and bool(mask_[i + 1])
            and int(ends_[i]) == int(intervals_[i + 1])
        ):
            i += 1
        runs.append((int(start), int(i)))
        i += 1
    return runs


def _mergeBroadRunsByObjective(
    runs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    intervals: np.ndarray,
    ends: np.ndarray,
    chromosome: str,
    selectionPenalty: float,
    boundaryCost: float,
    maxGapBP: int,
    blacklistByChrom: Mapping[str, np.ndarray],
    runPenalty: float = 0.0,
    dipPenaltyFraction: float = 1.0,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    dipPenaltyFraction_ = float(np.clip(float(dipPenaltyFraction), 0.0, 1.0))
    if not runs:
        return [], {
            "policy": (
                "soft_dip_objective_delta"
                if dipPenaltyFraction_ < 1.0
                else "objective_delta"
            ),
            "num_input_runs": 0,
            "num_output_runs": 0,
            "num_gaps_evaluated": 0,
            "num_gaps_merged": 0,
            "num_gaps_blocked_by_blacklist": 0,
            "num_gaps_blocked_by_distance": 0,
            "max_gap_bp": int(maxGapBP),
            "run_penalty": float(max(float(runPenalty), 0.0)),
            "dip_penalty_fraction": float(dipPenaltyFraction_),
        }
    scores_ = np.asarray(scores, dtype=np.float64).ravel()
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    if scores_.size != intervals_.size or scores_.size != ends_.size:
        raise ValueError("`scores`, `intervals`, and `ends` must match length")
    selectionPenalty_ = float(max(float(selectionPenalty), 0.0))
    boundaryCost_ = float(max(float(boundaryCost), 0.0))
    runPenalty_ = float(max(float(runPenalty), 0.0))
    maxGapBP_ = int(max(int(maxGapBP), 0))
    mergedRuns: List[Tuple[int, int]] = []
    gapsEvaluated = 0
    gapsMerged = 0
    blockedBlacklist = 0
    blockedDistance = 0
    activeStart, activeEnd = int(runs[0][0]), int(runs[0][1])
    for nextStart, nextEnd in runs[1:]:
        nextStart_ = int(nextStart)
        nextEnd_ = int(nextEnd)
        gapsEvaluated += 1
        gapStartBP = int(ends_[activeEnd])
        gapEndBP = int(intervals_[nextStart_])
        gapBP = int(max(gapEndBP - gapStartBP, 0))
        if gapBP > maxGapBP_:
            blockedDistance += 1
            mergedRuns.append((int(activeStart), int(activeEnd)))
            activeStart, activeEnd = nextStart_, nextEnd_
            continue
        if _intervalOverlapsBlacklist(
            str(chromosome),
            gapStartBP,
            gapEndBP,
            blacklistByChrom,
        ):
            blockedBlacklist += 1
            mergedRuns.append((int(activeStart), int(activeEnd)))
            activeStart, activeEnd = nextStart_, nextEnd_
            continue
        if nextStart_ - activeEnd <= 1:
            gapScore = 0.0
        else:
            gapScores = scores_[activeEnd + 1 : nextStart_]
            gapExcess = np.asarray(gapScores - selectionPenalty_, dtype=np.float64)
            gapScore = float(
                np.sum(
                    np.where(
                        gapExcess < 0.0,
                        dipPenaltyFraction_ * gapExcess,
                        gapExcess,
                    )
                )
            )
        mergeGain = float(gapScore + 2.0 * boundaryCost_ + runPenalty_)
        if mergeGain > 0.0:
            gapsMerged += 1
            activeEnd = nextEnd_
            continue
        mergedRuns.append((int(activeStart), int(activeEnd)))
        activeStart, activeEnd = nextStart_, nextEnd_
    mergedRuns.append((int(activeStart), int(activeEnd)))
    return mergedRuns, {
        "policy": (
            "soft_dip_objective_delta"
            if dipPenaltyFraction_ < 1.0
            else "objective_delta"
        ),
        "num_input_runs": int(len(runs)),
        "num_output_runs": int(len(mergedRuns)),
        "num_gaps_evaluated": int(gapsEvaluated),
        "num_gaps_merged": int(gapsMerged),
        "num_gaps_blocked_by_blacklist": int(blockedBlacklist),
        "num_gaps_blocked_by_distance": int(blockedDistance),
        "selection_penalty": float(selectionPenalty_),
        "boundary_cost": float(boundaryCost_),
        "run_penalty": float(runPenalty_),
        "max_gap_bp": int(maxGapBP_),
        "dip_penalty_fraction": float(dipPenaltyFraction_),
    }


def _maskJaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_ = np.asarray(a, dtype=bool)
    b_ = np.asarray(b, dtype=bool)
    union = int(np.sum(a_ | b_))
    if union == 0:
        return 1.0
    return float(np.sum(a_ & b_) / union)


def _roccoObjectiveForSolution(
    scores: np.ndarray,
    solution: np.ndarray,
    gamma: float,
) -> float:
    scores_ = np.asarray(scores, dtype=np.float64)
    solution_ = np.asarray(solution, dtype=np.uint8)
    if scores_.size != solution_.size:
        raise ValueError("`scores` and `solution` must have the same length")
    objective = float(np.sum(scores_[solution_ > 0]))
    if solution_.size > 1:
        objective -= float(max(float(gamma), 0.0)) * float(
            np.sum(np.diff(solution_) != 0)
        )
    return objective


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


def _nestedSoftBudgetTargetCount(
    regionBins: int,
    budgetScale: float,
    minChildBins: int,
) -> int:
    regionBins_ = int(max(int(regionBins), 1))
    scaled = int(math.floor(float(regionBins_) * float(np.clip(budgetScale, 0.0, 1.0))))
    return int(min(regionBins_, max(int(minChildBins), scaled, 1)))


def _positiveScoreScale(scores: np.ndarray) -> float:
    scores_ = np.asarray(scores, dtype=np.float64)
    positive = scores_[scores_ > 0.0]
    if positive.size > 0:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(np.abs(scores_)))
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = 0.0
    return scale


def _nestedSoftSelectionPenalty(
    scores: np.ndarray,
    selectionPenalty: float,
    budgetScale: float,
) -> Tuple[float, Dict[str, float]]:
    budgetScale_ = float(np.clip(float(budgetScale), 0.0, 1.0))
    basePenalty = float(max(float(selectionPenalty), 0.0))
    positiveScale = _positiveScoreScale(scores)
    positive = np.asarray(scores, dtype=np.float64)
    positive = positive[positive > 0.0]
    positiveSpread = 0.0
    if positive.size > 1:
        positiveSpread = float(
            np.quantile(positive, 0.75) - np.quantile(positive, 0.25)
        )
    if (not np.isfinite(positiveSpread)) or positiveSpread < 0.0:
        positiveSpread = 0.0
    extraPenalty = float((1.0 - budgetScale_) * positiveSpread)
    penalty = float(basePenalty + extraPenalty)
    return penalty, {
        "base_penalty": float(basePenalty),
        "extra_penalty": float(extraPenalty),
        "positive_score_scale": float(positiveScale),
        "positive_score_spread": float(positiveSpread),
        "budget_scale": float(budgetScale_),
    }


def _asParentBoundaryCosts(boundaryCosts: npt.ArrayLike, n: int) -> np.ndarray:
    n_ = int(max(int(n), 1))
    arr = np.asarray(boundaryCosts, dtype=np.float64).ravel()
    if arr.size == 1:
        out = np.full(n_ + 1, float(arr[0]), dtype=np.float64)
    elif arr.size == n_ + 1:
        out = arr.astype(np.float64, copy=True)
    else:
        raise ValueError(
            "`boundaryCosts` must be scalar or have length len(scores) + 1"
        )
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("`boundaryCosts` must be finite and non-negative")
    return out


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


def _bhQValues(pValues: npt.ArrayLike) -> np.ndarray:
    p = np.asarray(pValues, dtype=np.float64).ravel()
    if p.size == 0:
        return np.asarray([], dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise ValueError("`pValues` contains non-finite values")
    p = np.clip(p, 0.0, 1.0)
    order = np.argsort(p, kind="mergesort")
    out = np.empty_like(p)
    previous = 1.0
    n = int(p.size)
    for rank in range(n - 1, -1, -1):
        idx = int(order[rank])
        value = float(p[idx]) * float(n) / float(rank + 1)
        previous = min(previous, value)
        out[idx] = min(previous, 1.0)
    return out


def _empiricalReplaySegmentPValues(
    observedStats: npt.ArrayLike,
    nullStatsByDraw: Iterable[npt.ArrayLike],
) -> np.ndarray:
    observed = np.asarray(observedStats, dtype=np.float64).ravel()
    nullParts: List[np.ndarray] = []
    for draw in nullStatsByDraw:
        draw_ = np.asarray(draw, dtype=np.float64).ravel()
        if draw_.size > 0:
            nullParts.append(draw_)
    if observed.size == 0:
        return np.asarray([], dtype=np.float64)
    if len(nullParts) == 0:
        return np.ones(observed.size, dtype=np.float64)
    nullStats = np.concatenate(nullParts)
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(nullStats)):
        raise ValueError("replay segment statistics contain non-finite values")
    nullStats.sort()
    denominator = float(nullStats.size + 1)
    tailStarts = np.searchsorted(nullStats, observed, side="left")
    out = (1.0 + (nullStats.size - tailStarts).astype(np.float64)) / denominator
    return np.clip(out, 0.0, 1.0)


def _replayFDRQValues(
    observedStats: npt.ArrayLike,
    nullStatsByDraw: Iterable[npt.ArrayLike],
) -> np.ndarray:
    observed = np.asarray(observedStats, dtype=np.float64).ravel()
    if observed.size == 0:
        return np.asarray([], dtype=np.float64)
    nullDraws = [
        np.asarray(draw, dtype=np.float64).ravel()
        for draw in nullStatsByDraw
    ]
    if not np.all(np.isfinite(observed)) or any(
        not np.all(np.isfinite(draw)) for draw in nullDraws
    ):
        raise ValueError("replay FDR statistics contain non-finite values")
    for draw in nullDraws:
        draw.sort()
    statsSorted = np.sort(observed)
    order = np.argsort(-observed, kind="mergesort")
    rawFdr = np.ones(observed.size, dtype=np.float64)
    replayPseudocount = 1.0 / float(len(nullDraws) + 1) if len(nullDraws) > 0 else 1.0
    for rank, idx in enumerate(order):
        threshold = float(observed[idx])
        observedAtThreshold = int(
            statsSorted.size
            - np.searchsorted(statsSorted, threshold, side="left")
        )
        expectedNull = float(
            np.mean(
                [
                    draw.size - np.searchsorted(draw, threshold, side="left")
                    for draw in nullDraws
                ]
            )
            if len(nullDraws) > 0
            else 0.0
        )
        rawFdr[rank] = float(
            np.clip(
                (expectedNull + replayPseudocount)
                / float(max(observedAtThreshold, 1)),
                0.0,
                1.0,
            )
        )

    qValues = np.ones(observed.size, dtype=np.float64)
    running = 1.0
    for rank in range(observed.size - 1, -1, -1):
        running = min(running, float(rawFdr[rank]))
        qValues[int(order[rank])] = float(running)
    return np.clip(qValues, 0.0, 1.0)


def _resolveMultiscaleCandidateBins(
    n: int,
    dependenceSpan: int | None = None,
    lowerSpan: int | None = None,
    upperSpan: int | None = None,
    explicitScales: Iterable[int] | None = None,
) -> List[int]:
    n_ = int(max(int(n), 1))
    raw: List[int] = []
    if explicitScales is not None:
        raw.extend(int(scale) for scale in explicitScales)
    else:
        span = 0 if dependenceSpan is None else int(dependenceSpan)
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
        return {
            "score": 0.0,
            "integrated_excess": 0.0,
            "mean_excess": 0.0,
            "max_excess": 0.0,
        }
    threshold = float(view.get("threshold", 0.0))
    nullScale = float(max(float(view.get("null_scale", 1.0)), _TINY))
    excess = np.clip((scores_[start_ : end_ + 1] - threshold) / nullScale, 0.0, None)
    integrated = float(np.sum(excess))
    length = int(max(end_ - start_ + 1, 1))
    score = float(integrated / math.sqrt(float(length)))
    return {
        "score": float(score),
        "integrated_excess": float(integrated),
        "mean_excess": float(np.mean(excess)) if excess.size else 0.0,
        "max_excess": float(np.max(excess)) if excess.size else 0.0,
    }


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
            "integrated_excess": 0.0,
            "mean_excess": 0.0,
            "max_excess": 0.0,
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
    maxSegments: int | None = _DWB_PEAK_SCORING_MAX_SEGMENTS,
    maxSegmentsPerView: int | None = _DWB_PEAK_SCORING_MAX_SEGMENTS_PER_VIEW,
    returnDiagnostics: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
    thresholdZValues: List[float] = []
    thresholdValues: List[float] = []
    nullScaleValues: List[float] = []
    for key, viewAny in thresholdViews.items():
        if not isinstance(viewAny, Mapping):
            continue
        thresholdKeys.append(str(key))
        thresholdZValues.append(float(viewAny.get("threshold_z", 0.0)))
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
        integratedArr,
        meanArr,
        maxArr,
        eligibleCount,
        perViewCapHitCount,
        perViewDiscardedCount,
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
                "threshold_z": float(thresholdZValues[viewIdx]),
                "threshold": float(thresholdValues[viewIdx]),
                "null_scale": float(nullScaleValues[viewIdx]),
                "score": float(scoreArr[rowIdx]),
                "integrated_excess": float(integratedArr[rowIdx]),
                "mean_excess": float(meanArr[rowIdx]),
                "max_excess": float(maxArr[rowIdx]),
            }
        )
    preTotalCapCount = int(len(candidates))
    totalCapHit = bool(maxSegments_ is not None and len(candidates) > maxSegments_)
    totalDiscardedCount = 0
    if totalCapHit:
        totalDiscardedCount = int(len(candidates) - int(maxSegments_))
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
    if returnDiagnostics:
        diagnostics = {
            "eligible_candidate_count": int(eligibleCount),
            "candidate_count_before_total_cap": int(preTotalCapCount),
            "candidate_count": int(len(candidates)),
            "cap_hit": bool(perViewCapHitCount > 0 or totalCapHit),
            "per_view_cap_hit_count": int(perViewCapHitCount),
            "total_cap_hit": bool(totalCapHit),
            "discarded_by_per_view_cap": int(perViewDiscardedCount),
            "discarded_by_total_cap": int(totalDiscardedCount),
            "max_segments": None if maxSegments_ is None else int(maxSegments_),
            "max_segments_per_view": (
                None if maxSegmentsPerView_ is None else int(maxSegmentsPerView_)
            ),
        }
        return candidates, diagnostics
    return candidates


def _summarizeMultiscaleCandidates(
    candidates: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates_ = [dict(candidate) for candidate in candidates]
    byScale: Dict[str, int] = {}
    byThreshold: Dict[str, int] = {}
    scores: List[float] = []
    for candidate in candidates_:
        scaleKey = str(int(candidate.get("scale_bins", 1)))
        thresholdKey = str(candidate.get("threshold_key", ""))
        byScale[scaleKey] = int(byScale.get(scaleKey, 0) + 1)
        byThreshold[thresholdKey] = int(byThreshold.get(thresholdKey, 0) + 1)
        score = float(candidate.get("score", 0.0))
        if np.isfinite(score):
            scores.append(score)
    scoreArr = np.asarray(scores, dtype=np.float64)
    return {
        "num_candidates": int(len(candidates_)),
        "by_scale_bins": byScale,
        "by_threshold": byThreshold,
        "max_score": float(np.max(scoreArr)) if scoreArr.size else 0.0,
        "median_score": float(np.median(scoreArr)) if scoreArr.size else 0.0,
    }


def _generateROCCOMultiscaleCandidateSegments(
    scores: npt.ArrayLike,
    intervals: npt.ArrayLike | None = None,
    ends: npt.ArrayLike | None = None,
    threshold: float = 0.0,
    scales: Iterable[int] | None = None,
    minRunBins: int = 1,
    returnDetails: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    r"""Generate threshold-run ROCCO candidates over multiple smoothing scales."""
    scores_ = _asFloatVector("scores", scores)
    intervals_: np.ndarray | None = None
    ends_: np.ndarray | None = None
    if intervals is not None or ends is not None:
        if intervals is None or ends is None:
            raise ValueError("`intervals` and `ends` must be supplied together")
        intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
        ends_ = np.asarray(ends, dtype=np.int64).ravel()
        if intervals_.size != scores_.size or ends_.size != scores_.size:
            raise ValueError("`intervals` and `ends` must match `scores` length")
    scaleBins = _resolveMultiscaleCandidateBins(
        int(scores_.size),
        explicitScales=(scales if scales is not None else (1,)),
    )
    thresholdViews = {
        "primary": {
            "threshold_z": 0.0,
            "threshold": float(threshold),
            "null_scale": 1.0,
        }
    }
    rawCandidates, candidateDiagnostics = _multiscaleCandidateSegments(
        scores_,
        thresholdViews,
        scaleBins=scaleBins,
        minRunBins=int(max(int(minRunBins), 1)),
        returnDiagnostics=True,
    )
    candidates: List[Dict[str, Any]] = []
    for candidate in rawCandidates:
        startIdx = int(candidate["start_idx"])
        endIdx = int(candidate["end_idx"])
        start = int(intervals_[startIdx]) if intervals_ is not None else int(startIdx)
        end = int(ends_[endIdx]) if ends_ is not None else int(endIdx + 1)
        candidates.append(
            {
                **dict(candidate),
                "start": int(start),
                "end": int(end),
                "score_statistic": float(candidate.get("score", 0.0)),
            }
        )
    details = {
        "method": "multiscale_rocco_candidates",
        "threshold": float(threshold),
        "scales": [int(scale) for scale in scaleBins],
        "min_run_bins": int(max(int(minRunBins), 1)),
        "num_candidates": int(len(candidates)),
        "cap_hit": bool(candidateDiagnostics["cap_hit"]),
        "per_view_cap_hit_count": int(
            candidateDiagnostics["per_view_cap_hit_count"]
        ),
        "total_cap_hit": bool(candidateDiagnostics["total_cap_hit"]),
        "discarded_by_per_view_cap": int(
            candidateDiagnostics["discarded_by_per_view_cap"]
        ),
        "discarded_by_total_cap": int(
            candidateDiagnostics["discarded_by_total_cap"]
        ),
        "candidate_scales": sorted(
            {int(candidate["scale_bins"]) for candidate in candidates}
            | {int(scale) for scale in scaleBins}
        ),
    }
    if returnDetails:
        return candidates, details
    return candidates


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
    return replayViews


def _addDWBPeakScoringToPeakMeta(
    peakMeta: List[Dict[str, Any]],
    scores: npt.ArrayLike,
    prepared: Mapping[str, Any],
    exportDetails: Dict[str, Any] | None = None,
    minRunBins: int = 1,
    intervals: npt.ArrayLike | None = None,
    ends: npt.ArrayLike | None = None,
) -> Dict[str, Any]:
    r"""Annotate exported peak metadata with DWB null-replay empirical p/q values."""
    details = {} if exportDetails is None else exportDetails
    if len(peakMeta) == 0:
        summary = {
            "enabled": False,
            "reason": "no_peaks",
            "num_peaks": 0,
        }
        details["dwb_peak_scoring"] = summary
        return summary

    scores_ = _asFloatVector("scores", scores)
    intervals_: np.ndarray | None = None
    ends_: np.ndarray | None = None
    if intervals is not None or ends is not None:
        if intervals is None or ends is None:
            raise ValueError("`intervals` and `ends` must be supplied together")
        intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
        ends_ = np.asarray(ends, dtype=np.int64).ravel()
        if intervals_.size != scores_.size or ends_.size != scores_.size:
            raise ValueError("`intervals` and `ends` must match `scores` length")
    thresholdViews = prepared.get("threshold_views", {})
    calibration = prepared.get("dwb_calibration", {})
    template = prepared.get("template")
    if (
        not isinstance(thresholdViews, Mapping)
        or not isinstance(calibration, Mapping)
        or template is None
    ):
        summary = {
            "enabled": False,
            "reason": "missing_dwb_calibration",
            "num_peaks": int(len(peakMeta)),
        }
        details["dwb_peak_scoring"] = summary
        return summary

    template_ = np.asarray(template, dtype=np.float64).ravel()
    if template_.size != scores_.size:
        summary = {
            "enabled": False,
            "reason": "template_length_mismatch",
            "num_peaks": int(len(peakMeta)),
        }
        details["dwb_peak_scoring"] = summary
        return summary

    dependenceSpan = int(calibration.get("dependence_span", 2))
    lowerSpan = int(calibration.get("dependence_span_lower", dependenceSpan))
    upperSpan = int(calibration.get("dependence_span_upper", dependenceSpan))
    kernel = str(calibration.get("kernel", "bartlett"))
    numBootstrap = int(max(int(calibration.get("num_bootstrap", 0)), 0))
    numReplay = int(min(numBootstrap, _DWB_PEAK_SCORING_MAX_REPLAYS))
    randomSeed = int(calibration.get("random_seed", 0))
    panelId = str(calibration.get("dwb_panel_id", ""))
    if numBootstrap <= 0:
        summary = {
            "enabled": False,
            "reason": "no_bootstrap_draws",
            "num_peaks": int(len(peakMeta)),
        }
        details["dwb_peak_scoring"] = summary
        return summary

    scaleBins = _resolveMultiscaleCandidateBins(
        int(scores_.size),
        dependenceSpan=dependenceSpan,
        lowerSpan=lowerSpan,
        upperSpan=upperSpan,
    )
    minRunBins_ = int(max(int(minRunBins), 1))
    observedCandidates, observedCandidateDiagnostics = _multiscaleCandidateSegments(
        scores_,
        thresholdViews,
        scaleBins=scaleBins,
        minRunBins=minRunBins_,
        maxSegments=_DWB_PEAK_SCORING_MAX_SEGMENTS,
        maxSegmentsPerView=_DWB_PEAK_SCORING_MAX_SEGMENTS_PER_VIEW,
        returnDiagnostics=True,
    )
    candidateBySpan: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _candidateCoordinates(startIdx: int, endIdx: int) -> Tuple[int, int]:
        if intervals_ is not None and ends_ is not None:
            return int(intervals_[startIdx]), int(ends_[endIdx])
        return int(startIdx), int(endIdx + 1)

    def _mergeCandidateRegion(
        startIdx: int,
        endIdx: int,
        scoreDetail: Mapping[str, Any],
        source: str,
        scaleBinsValue: int | None = None,
        overlapCount: int | None = None,
    ) -> None:
        startIdx_ = int(max(int(startIdx), 0))
        endIdx_ = int(min(max(int(endIdx), startIdx_), int(scores_.size) - 1))
        key = (startIdx_, endIdx_)
        chromStart, chromEnd = _candidateCoordinates(startIdx_, endIdx_)
        existing = candidateBySpan.get(key)
        if existing is None:
            existing = {
                "candidate_id": "",
                "start_idx": int(startIdx_),
                "end_idx": int(endIdx_),
                "start": int(chromStart),
                "end": int(chromEnd),
                "exported": False,
                "candidate_sources": [],
                "candidate_scale_bins": [],
                "candidate_threshold_keys": [],
                "score": 0.0,
                "integrated_excess": 0.0,
                "mean_excess": 0.0,
                "max_excess": 0.0,
                "threshold_key": "",
                "threshold_z": 0.0,
                "threshold": 0.0,
                "null_scale": 1.0,
            }
            candidateBySpan[key] = existing
        sources = list(existing.get("candidate_sources", []))
        if source not in sources:
            sources.append(str(source))
        existing["candidate_sources"] = sources
        if scaleBinsValue is not None:
            scaleValues = list(existing.get("candidate_scale_bins", []))
            scaleValue = int(scaleBinsValue)
            if scaleValue not in scaleValues:
                scaleValues.append(scaleValue)
            existing["candidate_scale_bins"] = sorted(scaleValues)
        thresholdKeys = list(existing.get("candidate_threshold_keys", []))
        thresholdKey = str(scoreDetail.get("threshold_key", ""))
        if thresholdKey and thresholdKey not in thresholdKeys:
            thresholdKeys.append(thresholdKey)
        existing["candidate_threshold_keys"] = thresholdKeys
        if overlapCount is not None:
            existing["overlapping_multiscale_candidate_count"] = max(
                int(existing.get("overlapping_multiscale_candidate_count", 0)),
                int(overlapCount),
            )
        if float(scoreDetail.get("score", 0.0)) >= float(existing.get("score", 0.0)):
            existing.update(
                {
                    "score": float(scoreDetail.get("score", 0.0)),
                    "integrated_excess": float(
                        scoreDetail.get("integrated_excess", 0.0)
                    ),
                    "mean_excess": float(scoreDetail.get("mean_excess", 0.0)),
                    "max_excess": float(scoreDetail.get("max_excess", 0.0)),
                    "threshold_key": thresholdKey,
                    "threshold_z": float(scoreDetail.get("threshold_z", 0.0)),
                    "threshold": float(scoreDetail.get("threshold", 0.0)),
                    "null_scale": float(scoreDetail.get("null_scale", 1.0)),
                }
            )

    for candidate in observedCandidates:
        _mergeCandidateRegion(
            int(candidate["start_idx"]),
            int(candidate["end_idx"]),
            candidate,
            source="multiscale_threshold_run",
            scaleBinsValue=int(candidate.get("scale_bins", 1)),
        )
    observedStarts = np.asarray(
        [int(candidate["start_idx"]) for candidate in observedCandidates],
        dtype=np.int64,
    )
    observedEnds = np.asarray(
        [int(candidate["end_idx"]) for candidate in observedCandidates],
        dtype=np.int64,
    )
    observedScales = np.asarray(
        [int(candidate.get("scale_bins", 1)) for candidate in observedCandidates],
        dtype=np.int64,
    )
    for meta in peakMeta:
        startIdx = int(meta.get("start_idx", meta.get("child_start_idx", 0)))
        endIdx = int(meta.get("end_idx", meta.get("child_end_idx", startIdx)))
        scoreDetail = _bestSegmentScoreAcrossThresholdViews(
            scores_,
            startIdx,
            endIdx,
            thresholdViews,
        )
        _mergeCandidateRegion(
            startIdx,
            endIdx,
            scoreDetail,
            source="exported_peak",
        )
        if observedStarts.size > 0:
            overlapMask = (observedEnds >= int(startIdx)) & (
                observedStarts <= int(endIdx)
            )
            overlapCount = int(np.sum(overlapMask))
            if overlapCount > 0:
                for scaleValue in np.unique(observedScales[overlapMask]):
                    _mergeCandidateRegion(
                        startIdx,
                        endIdx,
                        scoreDetail,
                        source="multiscale_threshold_overlap",
                        scaleBinsValue=int(scaleValue),
                        overlapCount=overlapCount,
                    )
        candidateBySpan[(int(startIdx), int(endIdx))]["exported"] = True
    candidateDetails = sorted(
        (dict(candidate) for candidate in candidateBySpan.values()),
        key=lambda item: (int(item["start_idx"]), int(item["end_idx"])),
    )
    for idx, candidate in enumerate(candidateDetails, start=1):
        candidate["candidate_id"] = f"dwb_candidate_{idx}"
    details["multiscale_candidate_generation"] = {
        "method": "threshold_grid_moving_average_runs",
        "scale_bins": [int(scale) for scale in scaleBins],
        "min_run_bins": int(minRunBins_),
        "max_segments": int(_DWB_PEAK_SCORING_MAX_SEGMENTS),
        "max_segments_per_scale_threshold": int(
            _DWB_PEAK_SCORING_MAX_SEGMENTS_PER_VIEW
        ),
        **_summarizeMultiscaleCandidates(observedCandidates),
        "cap_hit": bool(observedCandidateDiagnostics["cap_hit"]),
        "per_view_cap_hit_count": int(
            observedCandidateDiagnostics["per_view_cap_hit_count"]
        ),
        "total_cap_hit": bool(observedCandidateDiagnostics["total_cap_hit"]),
        "discarded_by_per_view_cap": int(
            observedCandidateDiagnostics["discarded_by_per_view_cap"]
        ),
        "discarded_by_total_cap": int(
            observedCandidateDiagnostics["discarded_by_total_cap"]
        ),
        "eligible_candidate_count": int(
            observedCandidateDiagnostics["eligible_candidate_count"]
        ),
        "candidate_count_before_total_cap": int(
            observedCandidateDiagnostics["candidate_count_before_total_cap"]
        ),
        "deduplicated_candidate_regions": int(len(candidateDetails)),
        "exported_peak_regions": int(len(peakMeta)),
    }

    replayViews = _thresholdViewsForNullReplay(thresholdViews)
    rng = np.random.default_rng(randomSeed)
    nullCandidateCounts = np.zeros(numReplay, dtype=np.int64)
    metricKeys = {
        "summit_excess": "max_excess",
        "integrated_excess": "integrated_excess",
        "width_adjusted_mass": "score",
    }
    observedMetricStats: Dict[str, np.ndarray] = {
        metric: np.asarray(
            [float(candidate[sourceKey]) for candidate in candidateDetails],
            dtype=np.float64,
        )
        for metric, sourceKey in metricKeys.items()
    }
    nullMetricStatsByDraw: Dict[str, List[np.ndarray]] = {
        metric: [] for metric in metricKeys
    }
    nullCapHitDraws = 0
    nullPerViewCapHitDraws = 0
    nullTotalCapHitDraws = 0
    nullDiscardedByPerViewCap = 0
    nullDiscardedByTotalCap = 0
    for drawIdx in range(numReplay):
        draw = np.asarray(
            cconsenrich.cStationaryNullDWBDraw(
                template_,
                dependenceSpan,
                rng,
                kernel,
            ),
            dtype=np.float64,
        )
        nullCandidates, nullCandidateDiagnostics = _multiscaleCandidateSegments(
            draw,
            replayViews,
            scaleBins=scaleBins,
            minRunBins=minRunBins_,
            maxSegments=_DWB_PEAK_SCORING_MAX_SEGMENTS,
            maxSegmentsPerView=_DWB_PEAK_SCORING_MAX_SEGMENTS_PER_VIEW,
            returnDiagnostics=True,
        )
        if bool(nullCandidateDiagnostics["cap_hit"]):
            nullCapHitDraws += 1
        if int(nullCandidateDiagnostics["per_view_cap_hit_count"]) > 0:
            nullPerViewCapHitDraws += 1
        if bool(nullCandidateDiagnostics["total_cap_hit"]):
            nullTotalCapHitDraws += 1
        nullDiscardedByPerViewCap += int(
            nullCandidateDiagnostics["discarded_by_per_view_cap"]
        )
        nullDiscardedByTotalCap += int(
            nullCandidateDiagnostics["discarded_by_total_cap"]
        )
        nullCandidateCounts[drawIdx] = int(len(nullCandidates))
        for metric, sourceKey in metricKeys.items():
            nullMetricStatsByDraw[metric].append(
                np.asarray(
                    [
                        float(candidate.get(sourceKey, 0.0))
                        for candidate in nullCandidates
                    ],
                    dtype=np.float64,
                )
            )

    pValuesByMetric = {
        metric: _empiricalReplaySegmentPValues(
            observedMetricStats[metric],
            nullMetricStatsByDraw[metric],
        )
        for metric in metricKeys
    }
    qValuesByMetric = {
        metric: np.maximum(
            _replayFDRQValues(
                observedMetricStats[metric],
                nullMetricStatsByDraw[metric],
            ),
            pValuesByMetric[metric],
        )
        for metric in metricKeys
    }
    observedArr = observedMetricStats["width_adjusted_mass"]
    pValues = pValuesByMetric["width_adjusted_mass"]
    qValues = qValuesByMetric["width_adjusted_mass"]
    nullPrimaryMaxScores = np.asarray(
        [
            float(np.max(draw)) if draw.size else 0.0
            for draw in nullMetricStatsByDraw["width_adjusted_mass"]
        ],
        dtype=np.float64,
    )
    nullMaxQ95 = (
        float(np.quantile(nullPrimaryMaxScores, 0.95, method="interpolated_inverted_cdf"))
        if nullPrimaryMaxScores.size
        else 0.0
    )
    nullMaxMean = (
        float(np.mean(nullPrimaryMaxScores)) if nullPrimaryMaxScores.size else 0.0
    )
    nullCandidateMean = (
        float(np.mean(nullCandidateCounts)) if nullCandidateCounts.size else 0.0
    )
    nullCandidateQ95 = (
        float(np.quantile(nullCandidateCounts, 0.95, method="interpolated_inverted_cdf"))
        if nullCandidateCounts.size
        else 0.0
    )
    nullCandidateQ95 = float(max(nullCandidateQ95, nullCandidateMean))
    falseSegmentDiagnostics = {
        "method": "stationary_null_dwb_null_replay",
        "num_replays": int(numReplay),
        "budget_num_bootstrap": int(numBootstrap),
        "observed_segment_count": int(len(peakMeta)),
        "observed_candidate_count": int(len(candidateDetails)),
        "false_segment_count_mean": float(nullCandidateMean),
        "false_segment_count_q95": float(nullCandidateQ95),
        "false_segment_fdr_estimate": float(
            np.clip(nullCandidateMean / float(max(len(peakMeta), 1)), 0.0, 1.0)
        ),
        "candidate_fdr_estimate": float(
            np.clip(nullCandidateMean / float(max(len(candidateDetails), 1)), 0.0, 1.0)
        ),
        "dwb_panel_id": str(panelId),
        "kernel": str(kernel),
        "dependence_span": int(dependenceSpan),
        "scale_bins": [int(scale) for scale in scaleBins],
        "candidate_cap_hit": bool(nullCapHitDraws > 0),
        "candidate_cap_hit_draws": int(nullCapHitDraws),
        "candidate_per_view_cap_hit_draws": int(nullPerViewCapHitDraws),
        "candidate_total_cap_hit_draws": int(nullTotalCapHitDraws),
        "candidate_discarded_by_per_view_cap": int(nullDiscardedByPerViewCap),
        "candidate_discarded_by_total_cap": int(nullDiscardedByTotalCap),
        "max_segments": int(_DWB_PEAK_SCORING_MAX_SEGMENTS),
        "max_segments_per_scale_threshold": int(
            _DWB_PEAK_SCORING_MAX_SEGMENTS_PER_VIEW
        ),
    }
    if bool(observedCandidateDiagnostics["cap_hit"]) or nullCapHitDraws > 0:
        observedDiscardedCount = int(
            observedCandidateDiagnostics["discarded_by_per_view_cap"]
        ) + int(observedCandidateDiagnostics["discarded_by_total_cap"])
        nullDiscardedCount = int(nullDiscardedByPerViewCap) + int(
            nullDiscardedByTotalCap
        )
        logger.info(
            "DWB peak-scoring candidate caps hit panel=%s observed=%s "
            "null_draws=%d null_per_view_draws=%d null_total_draws=%d "
            "observed_discarded=%d null_discarded=%d max_segments=%d "
            "max_segments_per_view=%d",
            str(panelId),
            bool(observedCandidateDiagnostics["cap_hit"]),
            int(nullCapHitDraws),
            int(nullPerViewCapHitDraws),
            int(nullTotalCapHitDraws),
            int(observedDiscardedCount),
            int(nullDiscardedCount),
            int(_DWB_PEAK_SCORING_MAX_SEGMENTS),
            int(_DWB_PEAK_SCORING_MAX_SEGMENTS_PER_VIEW),
        )
    details["null_replay_false_segment_diagnostics"] = falseSegmentDiagnostics
    primaryNullStats = (
        np.concatenate(nullMetricStatsByDraw["width_adjusted_mass"])
        if any(draw.size for draw in nullMetricStatsByDraw["width_adjusted_mass"])
        else np.asarray([], dtype=np.float64)
    )

    for idx, candidate in enumerate(candidateDetails):
        pValue = float(pValues[idx])
        qValue = float(qValues[idx])
        statistic = float(observedArr[idx])
        candidate["dwb_peak_score"] = float(observedArr[idx])
        candidate["dwb_peak_score_method"] = (
            "max_threshold_sqrt_normalized_integrated_excess"
        )
        candidate["dwb_peak_empirical_p"] = pValue
        candidate["dwb_peak_empirical_q"] = qValue
        candidate["dwb_empirical_method"] = "stationary_null_dwb_peak_replay"
        candidate["dwb_empirical_panel_id"] = str(panelId)
        candidate["dwb_empirical_null_replays"] = int(numReplay)
        candidate["dwb_empirical_p"] = pValue
        candidate["dwb_empirical_q"] = qValue
        candidate["dwb_empirical_statistic"] = statistic
        candidate["dwb_empirical_q_method"] = "dwb_replay_fdr_candidate_segments"
        for metric in metricKeys:
            metricStat = float(observedMetricStats[metric][idx])
            metricP = float(pValuesByMetric[metric][idx])
            metricQ = float(qValuesByMetric[metric][idx])
            candidate[metric] = metricStat
            candidate[f"{metric}_p"] = metricP
            candidate[f"{metric}_q"] = metricQ
        candidate["dwb_peak_null_exceedances"] = int(
            np.sum(primaryNullStats >= float(observedArr[idx]))
            if primaryNullStats.size
            else 0
        )
        candidate["dwb_peak_scoring_threshold_key"] = str(
            candidate["threshold_key"]
        )
        candidate["dwb_peak_scoring_threshold_z"] = float(candidate["threshold_z"])
        candidate["dwb_peak_integrated_excess"] = float(candidate["integrated_excess"])
        candidate["dwb_peak_mean_excess"] = float(candidate["mean_excess"])
        candidate["dwb_peak_max_excess"] = float(candidate["max_excess"])
        candidate["dwb_null_replay_num_draws"] = int(numReplay)
        candidate["dwb_null_replay_max_score_mean"] = float(nullMaxMean)
        candidate["dwb_null_replay_max_score_q95"] = float(nullMaxQ95)
        candidate["dwb_null_replay_candidate_count_mean"] = float(nullCandidateMean)
        candidate["dwb_null_replay_candidate_count_q95"] = float(nullCandidateQ95)

    scoredCandidatesBySpan = {
        (int(candidate["start_idx"]), int(candidate["end_idx"])): candidate
        for candidate in candidateDetails
    }
    for meta in peakMeta:
        startIdx = int(meta.get("start_idx", meta.get("child_start_idx", 0)))
        endIdx = int(meta.get("end_idx", meta.get("child_end_idx", startIdx)))
        candidate = scoredCandidatesBySpan[(startIdx, endIdx)]
        for key in (
            "candidate_id",
            "candidate_sources",
            "candidate_scale_bins",
            "candidate_threshold_keys",
            "overlapping_multiscale_candidate_count",
            "dwb_peak_score",
            "dwb_peak_score_method",
            "dwb_peak_empirical_p",
            "dwb_peak_empirical_q",
            "dwb_empirical_method",
            "dwb_empirical_panel_id",
            "dwb_empirical_null_replays",
            "dwb_empirical_p",
            "dwb_empirical_q",
            "dwb_empirical_statistic",
            "dwb_empirical_q_method",
            "dwb_peak_null_exceedances",
            "dwb_peak_scoring_threshold_key",
            "dwb_peak_scoring_threshold_z",
            "dwb_peak_integrated_excess",
            "dwb_peak_mean_excess",
            "dwb_peak_max_excess",
            "dwb_null_replay_num_draws",
            "dwb_null_replay_max_score_mean",
            "dwb_null_replay_max_score_q95",
            "dwb_null_replay_candidate_count_mean",
            "dwb_null_replay_candidate_count_q95",
        ):
            meta[key] = candidate.get(
                key,
                0 if key == "overlapping_multiscale_candidate_count" else None,
            )
        for metric in metricKeys:
            meta[metric] = candidate[metric]
            meta[f"{metric}_p"] = candidate[f"{metric}_p"]
            meta[f"{metric}_q"] = candidate[f"{metric}_q"]
    details["candidate_details"] = candidateDetails
    details["candidate_significance"] = {
        "method": "stationary_dwb_null_replay_multiscale_segments",
        "p_value": "empirical_replay_segment_tail",
        "q_value": "dwb_replay_fdr_candidate_segments",
        "primary_metric": "width_adjusted_mass",
        "num_candidates": int(len(candidateDetails)),
        "num_exported_candidates": int(
            sum(1 for candidate in candidateDetails if bool(candidate.get("exported")))
        ),
        "dwb_panel_id": str(panelId),
    }

    summary = {
        "enabled": True,
        "method": "stationary_dwb_null_replay_multiscale_segments",
        "p_value": "empirical_replay_segment_tail",
        "q_value": "dwb_replay_fdr_candidate_segments",
        "primary_metric": "width_adjusted_mass",
        "metrics": sorted(metricKeys),
        "num_peaks": int(len(peakMeta)),
        "num_candidate_regions": int(len(candidateDetails)),
        "num_bootstrap": int(numReplay),
        "budget_num_bootstrap": int(numBootstrap),
        "random_seed": int(randomSeed),
        "kernel": str(kernel),
        "dwb_panel_id": str(panelId),
        "dependence_span": int(dependenceSpan),
        "scale_bins": [int(scale) for scale in scaleBins],
        "min_run_bins": int(minRunBins_),
        "observed_candidate_count": int(len(candidateDetails)),
        "raw_multiscale_candidate_count": int(len(observedCandidates)),
        "observed_candidate_cap_hit": bool(observedCandidateDiagnostics["cap_hit"]),
        "null_replay_candidate_cap_hit": bool(nullCapHitDraws > 0),
        "null_replay_candidate_cap_hit_draws": int(nullCapHitDraws),
        "null_replay_candidate_count_mean": float(nullCandidateMean),
        "null_replay_candidate_count_q95": float(nullCandidateQ95),
        "null_replay_max_score_mean": float(nullMaxMean),
        "null_replay_max_score_q95": float(nullMaxQ95),
        "min_p": float(np.min(pValues)) if pValues.size else None,
        "min_q": float(np.min(qValues)) if qValues.size else None,
        "false_segment_diagnostics": falseSegmentDiagnostics,
    }
    details["dwb_peak_scoring"] = summary
    return summary


def _massiveSubpeakWidthScores(
    widthsBP: npt.ArrayLike,
    bulkQuantile: float = _MASSIVE_SUBPEAK_WIDTH_BULK_QUANTILE,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    widths = np.asarray(widthsBP, dtype=np.float64).ravel()
    positive = widths[np.isfinite(widths) & (widths > 0.0)]
    if positive.size == 0:
        empty = np.ones_like(widths, dtype=np.float64)
        return empty, empty, {"center": 0.0, "scale": 1.0}
    logs = np.log(np.maximum(widths, 1.0))
    validLogs = np.log(positive)
    q = float(np.clip(float(bulkQuantile), 0.5, 0.99))
    cutoff = float(np.quantile(validLogs, q))
    bulk = validLogs[validLogs <= cutoff]
    if bulk.size < max(5, int(math.ceil(0.1 * validLogs.size))):
        bulk = validLogs
    center = float(np.median(bulk))
    scale = 1.4826 * float(np.median(np.abs(bulk - center)))
    if (not np.isfinite(scale)) or scale <= 1.0e-12:
        scale = float(np.quantile(bulk, 0.75) - np.quantile(bulk, 0.25)) / 1.349
    if (not np.isfinite(scale)) or scale <= 1.0e-12:
        scale = 1.0
    z = (logs - center) / scale
    p = stats.norm.sf(z)
    if not np.all(np.isfinite(p)):
        raise RuntimeError("width-score p-values contain non-finite values")
    p = np.clip(p, 0.0, 1.0)
    qValues = _bhQValues(p)
    return p, qValues, {"center": float(center), "scale": float(scale)}


def _learnMassiveSubpeakWidthPolicy(
    widthsBP: npt.ArrayLike,
    enabled: bool = _MASSIVE_SUBPEAK_CLEANUP_DEFAULT,
    minBP: int = _MASSIVE_SUBPEAK_MIN_BP,
    alpha: float = _MASSIVE_SUBPEAK_WIDTH_ALPHA,
    bulkQuantile: float = _MASSIVE_SUBPEAK_WIDTH_BULK_QUANTILE,
    maxFraction: float = _MASSIVE_SUBPEAK_MAX_FRACTION,
    minLogGap: float = _MASSIVE_SUBPEAK_MIN_LOG_GAP,
    minPeaks: int = _MASSIVE_SUBPEAK_MIN_PEAKS,
) -> Dict[str, Any]:
    widths = np.asarray(widthsBP, dtype=np.float64).ravel()
    valid = widths[np.isfinite(widths) & (widths > 0.0)]
    details: Dict[str, Any] = {
        "enabled": bool(enabled),
        "method": "robust_log_width_tail_gap",
        "width_threshold_bp": None,
        "gap_width_threshold_bp": None,
        "width_cap_bp": None,
        "width_cap_z": float(_MASSIVE_SUBPEAK_TRIGGER_Z_CAP),
        "min_bp": int(max(int(minBP), 1)),
        "contract_width_bp": int(max(int(minBP), 1)),
        "alpha": float(alpha),
        "bulk_quantile": float(bulkQuantile),
        "max_fraction": float(maxFraction),
        "min_log_gap": float(minLogGap),
        "min_peaks": int(max(int(minPeaks), 1)),
        "num_peaks": int(valid.size),
        "num_width_tail_candidates": 0,
        "num_width_cluster_candidates": 0,
        "selected_log_gap": None,
        "null_center": None,
        "null_scale": None,
        "active": False,
    }
    if (not bool(enabled)) or valid.size < int(details["min_peaks"]):
        return details
    pValues, qValues, scoreMeta = _massiveSubpeakWidthScores(
        valid,
        bulkQuantile=float(bulkQuantile),
    )
    details["null_center"] = float(scoreMeta["center"])
    details["null_scale"] = float(scoreMeta["scale"])
    alpha_ = float(np.clip(float(alpha), 0.0, 1.0))
    minBP_ = float(details["min_bp"])
    tail = (
        (valid >= minBP_)
        & np.isfinite(qValues)
        & (qValues <= alpha_)
    )
    details["num_width_tail_candidates"] = int(np.sum(tail))
    if not bool(np.any(tail)):
        return details
    uniqueWidths = np.unique(np.sort(valid))
    maxCount = int(max(1, math.floor(float(maxFraction) * float(valid.size))))
    candidates: List[Tuple[float, int, float]] = []
    for left, right in zip(uniqueWidths[:-1], uniqueWidths[1:]):
        if right < minBP_:
            continue
        logGap = float(np.log(right) - np.log(left))
        if logGap < float(minLogGap):
            continue
        rightMask = valid >= float(right)
        rightCount = int(np.sum(rightMask))
        if rightCount < 1 or rightCount > maxCount:
            continue
        if not bool(np.all(qValues[rightMask] <= alpha_)):
            continue
        candidates.append((float(right), int(rightCount), float(logGap)))
    if not candidates:
        return details
    gapThreshold, gapClusterCount, logGap = min(candidates, key=lambda item: item[0])
    widthCap = math.exp(
        float(scoreMeta["center"])
        + float(_MASSIVE_SUBPEAK_TRIGGER_Z_CAP) * float(scoreMeta["scale"])
    )
    if (not np.isfinite(widthCap)) or widthCap <= 0.0:
        widthCap = float(gapThreshold)
    widthCap = float(max(minBP_, widthCap))
    threshold = float(max(minBP_, min(float(gapThreshold), widthCap)))
    details["gap_width_threshold_bp"] = int(round(float(gapThreshold)))
    details["width_cap_bp"] = int(round(float(widthCap)))
    details["width_threshold_bp"] = int(round(float(threshold)))
    details["num_width_cluster_candidates"] = int(np.sum(valid >= float(threshold)))
    details["num_width_tail_gap_candidates"] = int(gapClusterCount)
    details["selected_log_gap"] = float(logGap)
    details["active"] = True
    return details


def _massiveSubpeakWidthPValue(
    widthBP: float,
    policy: Mapping[str, Any] | None,
) -> Tuple[float | None, float | None]:
    if policy is None:
        return None, None
    center = policy.get("null_center")
    scale = policy.get("null_scale")
    if center is None or scale is None:
        return None, None
    scale_ = float(scale)
    if (not np.isfinite(scale_)) or scale_ <= 0.0:
        return None, None
    width_ = max(float(widthBP), 1.0)
    z = (math.log(width_) - float(center)) / scale_
    p = float(stats.norm.sf(z))
    return float(np.clip(p, 0.0, 1.0)), None


def _bestMassiveSubpeakSplit(
    scores: np.ndarray,
    boundaryCost: float,
    minRunBins: int,
    splitQuantile: float = _MASSIVE_SUBPEAK_SPLIT_QUANTILE,
) -> Dict[str, Any] | None:
    scores_ = np.asarray(scores, dtype=np.float64)
    n = int(scores_.size)
    minBins = int(max(int(minRunBins), 1))
    if n < 3 * minBins:
        return None
    baseline = float(np.quantile(scores_, float(np.clip(splitQuantile, 0.0, 1.0))))
    scale = 1.4826 * float(np.median(np.abs(scores_ - np.median(scores_))))
    if (not np.isfinite(scale)) or scale <= 1.0e-12:
        scale = float(np.quantile(scores_, 0.75) - np.quantile(scores_, 0.25)) / 1.349
    if (not np.isfinite(scale)) or scale <= 1.0e-12:
        scale = 1.0
    contrast = baseline - scores_
    prefix = np.concatenate(([0.0], np.cumsum(contrast, dtype=np.float64)))
    bestSum = -math.inf
    bestStart = -1
    bestEnd = -1
    bestPrefix = math.inf
    bestPrefixIndex = -1
    lastEnd = n - minBins - 1
    for end in range(minBins + minBins - 1, lastEnd + 1):
        allowedStart = end - minBins + 1
        if allowedStart >= minBins:
            candidatePrefix = float(prefix[allowedStart])
            if candidatePrefix < bestPrefix:
                bestPrefix = candidatePrefix
                bestPrefixIndex = int(allowedStart)
        if bestPrefixIndex < 0:
            continue
        gapSum = float(prefix[end + 1] - bestPrefix)
        if gapSum > bestSum:
            bestSum = float(gapSum)
            bestStart = int(bestPrefixIndex)
            bestEnd = int(end)
    if bestStart < 0 or bestEnd < bestStart:
        return None
    gapBins = int(bestEnd - bestStart + 1)
    gain = float(bestSum - 2.0 * max(float(boundaryCost), 0.0))
    z = float(bestSum / (float(scale) * math.sqrt(max(gapBins, 1))))
    return {
        "gap_start_local": int(bestStart),
        "gap_end_local": int(bestEnd),
        "gap_bins": int(gapBins),
        "baseline": float(baseline),
        "scale": float(scale),
        "deficit": float(bestSum),
        "gain": float(gain),
        "z": float(z),
    }


def _contractMassiveSubpeakSegment(
    startIdx: int,
    endIdx: int,
    scores: np.ndarray,
    intervals: np.ndarray,
    ends: np.ndarray,
    thresholdBP: float,
    minRunBins: int,
    splitQuantile: float = _MASSIVE_SUBPEAK_SPLIT_QUANTILE,
) -> Tuple[int, int, Dict[str, Any]] | None:
    scores_ = np.asarray(scores, dtype=np.float64)
    intervals_ = np.asarray(intervals, dtype=np.int64)
    ends_ = np.asarray(ends, dtype=np.int64)
    start = int(startIdx)
    end = int(endIdx)
    if start < 0 or end < start or end >= scores_.size:
        return None

    localScores = scores_[start : end + 1]
    localStarts = intervals_[start : end + 1]
    localEnds = ends_[start : end + 1]
    n = int(localScores.size)
    if n == 0:
        return None
    originalWidth = int(max(int(localEnds[-1]) - int(localStarts[0]), 0))
    threshold_ = float(thresholdBP)
    if (not np.isfinite(threshold_)) or threshold_ <= 0.0:
        return None
    if not np.all(np.isfinite(localScores)):
        raise ValueError("`scores` contains non-finite values")

    mids = 0.5 * (localStarts.astype(np.float64) + localEnds.astype(np.float64))
    maxScore = float(np.max(localScores))
    maxMask = localScores >= maxScore - max(1.0e-12, abs(maxScore) * 1.0e-12)
    centerAbs = float(np.median(mids[maxMask])) if bool(np.any(maxMask)) else float(
        0.5 * (int(localStarts[0]) + int(localEnds[-1]))
    )
    centerLocal = int(np.argmin(np.abs(mids - centerAbs)))
    minBins = int(max(int(minRunBins), 1))

    low = float(np.quantile(localScores, float(np.clip(splitQuantile, 0.0, 1.0))))
    dynamic = float(maxScore - low)
    if np.isfinite(dynamic) and dynamic > 1.0e-12:
        floor = float(low + 0.5 * dynamic)
        keep = localScores >= floor
        if bool(keep[centerLocal]):
            left = centerLocal
            while left > 0 and bool(keep[left - 1]):
                left -= 1
            right = centerLocal
            while right + 1 < n and bool(keep[right + 1]):
                right += 1
            width = int(max(int(localEnds[right]) - int(localStarts[left]), 0))
            if (
                right - left + 1 >= minBins
                and width < threshold_
                and width < originalWidth
            ):
                return int(start + left), int(start + right), {
                    "mode": "adaptive_core",
                    "core_score_floor": float(floor),
                    "core_width_bp": int(width),
                }

    left = centerLocal
    right = centerLocal
    while True:
        choices: List[Tuple[float, float, int]] = []
        if left > 0:
            widthLeft = int(max(int(localEnds[right]) - int(localStarts[left - 1]), 0))
            if widthLeft < threshold_:
                choices.append(
                    (
                        float(localScores[left - 1]),
                        -abs(float(mids[left - 1]) - centerAbs),
                        -1,
                    )
                )
        if right + 1 < n:
            widthRight = int(
                max(int(localEnds[right + 1]) - int(localStarts[left]), 0)
            )
            if widthRight < threshold_:
                choices.append(
                    (
                        float(localScores[right + 1]),
                        -abs(float(mids[right + 1]) - centerAbs),
                        1,
                    )
                )
        if not choices:
            break
        _score, _distance, direction = max(choices)
        if int(direction) < 0:
            left -= 1
        else:
            right += 1

    width = int(max(int(localEnds[right]) - int(localStarts[left]), 0))
    if right - left + 1 < minBins or width >= threshold_ or width >= originalWidth:
        return None
    return int(start + left), int(start + right), {
        "mode": "width_capped_core",
        "core_score_floor": None,
        "core_width_bp": int(width),
    }


def _makeSubpeakSegment(
    startIdx: int,
    endIdx: int,
    state: np.ndarray,
    originalStartIdx: int,
    originalEndIdx: int,
    numSubpeaks: int,
    splitFromParent: bool,
    objective: float,
    boundaryPenalty: float,
    cleanupCandidate: bool = False,
    cleanupApplied: bool = False,
    cleanupDetails: Mapping[str, Any] | None = None,
    cleanupMode: str | None = None,
) -> Dict[str, int | float | bool | str | None]:
    start = int(startIdx)
    end = int(endIdx)
    state_ = np.asarray(state, dtype=np.float64)
    summitLocal = int(np.argmax(state_[start : end + 1]))
    details = {} if cleanupDetails is None else dict(cleanupDetails)
    return {
        "start_idx": int(start),
        "end_idx": int(end),
        "summit_idx": int(start + summitLocal),
        "segment_length_bins": int(max(end - start + 1, 0)),
        "num_subpeaks": int(numSubpeaks),
        "split_from_parent": bool(splitFromParent),
        "subpeak_objective": float(objective),
        "subpeak_boundary_penalty": float(boundaryPenalty),
        "massive_subpeak_cleanup_candidate": bool(cleanupCandidate),
        "massive_subpeak_cleanup_applied": bool(cleanupApplied),
        "massive_subpeak_cleanup_mode": cleanupMode,
        "massive_subpeak_split_gain": (
            None if details.get("gain") is None else float(details["gain"])
        ),
        "massive_subpeak_split_z": (
            None if details.get("z") is None else float(details["z"])
        ),
        "massive_subpeak_gap_bins": (
            None if details.get("gap_bins") is None else int(details["gap_bins"])
        ),
        "massive_subpeak_core_width_bp": (
            None
            if details.get("core_width_bp") is None
            else int(details["core_width_bp"])
        ),
        "massive_subpeak_core_score_floor": (
            None
            if details.get("core_score_floor") is None
            else float(details["core_score_floor"])
        ),
        "massive_subpeak_parent_start_idx": int(originalStartIdx),
        "massive_subpeak_parent_end_idx": int(originalEndIdx),
    }


def _solveParentConditionedSubpeaks(
    scores: np.ndarray,
    boundaryCosts: npt.ArrayLike,
    selectionPenalty: float,
    minRunBins: int,
    requiredIndex: int | None = None,
    runPenalty: float = 0.0,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    scores_ = np.asarray(scores, dtype=np.float64)
    if scores_.ndim != 1 or scores_.size == 0:
        raise ValueError("`scores` must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(scores_)):
        raise ValueError("`scores` contains non-finite values")
    costs_ = _asParentBoundaryCosts(boundaryCosts, int(scores_.size))
    penalty_ = float(selectionPenalty)
    if not np.isfinite(penalty_):
        raise ValueError("`selectionPenalty` must be finite")
    runPenalty_ = float(runPenalty)
    if not np.isfinite(runPenalty_) or runPenalty_ < 0.0:
        raise ValueError("`runPenalty` must be finite and non-negative")

    n = int(scores_.size)
    requiredBin = None if requiredIndex is None else int(requiredIndex)
    if requiredBin is not None and (requiredBin < 0 or requiredBin >= n):
        raise ValueError("`requiredIndex` is outside `scores`")
    minRunBins_ = int(min(max(int(minRunBins), 1), n))
    numStates = int(minRunBins_ + 1)
    negInf = -math.inf
    eps = 1.0e-12
    largeCount = n + 1

    prevValues = np.full(numStates, negInf, dtype=np.float64)
    prevCounts = np.full(numStates, largeCount, dtype=np.int64)
    prevValues[0] = 0.0
    prevCounts[0] = 0
    backState = np.full((n, numStates), -1, dtype=np.int16)

    def _better(
        value: float,
        count: int,
        bestValue: float,
        bestCount: int,
    ) -> bool:
        if value > bestValue + eps:
            return True
        if abs(value - bestValue) <= eps and count < bestCount:
            return True
        return False

    def _update(
        values: np.ndarray,
        counts: np.ndarray,
        newState: int,
        value: float,
        count: int,
        prevState: int,
        i: int,
    ) -> None:
        if _better(
            float(value),
            int(count),
            float(values[newState]),
            int(counts[newState]),
        ):
            values[newState] = float(value)
            counts[newState] = int(count)
            backState[i, newState] = int(prevState)

    for i in range(n):
        adjustedScore = float(scores_[i] - penalty_)
        newValues = np.full(numStates, negInf, dtype=np.float64)
        newCounts = np.full(numStates, largeCount, dtype=np.int64)
        transitionCost = float(costs_[i])
        forceOn = bool(requiredBin is not None and i == requiredBin)

        if not forceOn:
            if np.isfinite(prevValues[0]):
                _update(
                    newValues,
                    newCounts,
                    0,
                    float(prevValues[0]),
                    int(prevCounts[0]),
                    0,
                    i,
                )
            if np.isfinite(prevValues[minRunBins_]):
                _update(
                    newValues,
                    newCounts,
                    0,
                    float(prevValues[minRunBins_] - transitionCost),
                    int(prevCounts[minRunBins_]),
                    minRunBins_,
                    i,
                )

        if np.isfinite(prevValues[0]):
            _update(
                newValues,
                newCounts,
                1,
                float(
                    prevValues[0]
                    - transitionCost
                    - runPenalty_
                    + adjustedScore
                ),
                int(prevCounts[0] + 1),
                0,
                i,
            )
        for state in range(1, minRunBins_):
            if not np.isfinite(prevValues[state]):
                continue
            _update(
                newValues,
                newCounts,
                state + 1,
                float(prevValues[state] + adjustedScore),
                int(prevCounts[state] + 1),
                state,
                i,
            )
        if np.isfinite(prevValues[minRunBins_]):
            _update(
                newValues,
                newCounts,
                minRunBins_,
                float(prevValues[minRunBins_] + adjustedScore),
                int(prevCounts[minRunBins_] + 1),
                minRunBins_,
                i,
            )

        prevValues = newValues
        prevCounts = newCounts

    finalCandidates = [
        (float(prevValues[0]), int(prevCounts[0]), 0),
        (
            float(prevValues[minRunBins_] - costs_[n]),
            int(prevCounts[minRunBins_]),
            minRunBins_,
        ),
    ]
    bestValue, bestCount, bestState = max(
        finalCandidates,
        key=lambda item: (item[0], -item[1]),
    )
    if not np.isfinite(bestValue):
        raise RuntimeError("parent-conditioned subpeak DP found no feasible path")
    mask = np.zeros(n, dtype=bool)
    state = int(bestState)
    for i in range(n - 1, -1, -1):
        if state > 0:
            mask[i] = True
        prevState = int(backState[i, state])
        if prevState < 0:
            break
        state = prevState
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


def _bestContiguousBudgetFallbackMask(
    scores: np.ndarray,
    targetCount: int,
    selectionPenalty: float,
    gamma: float,
    maxRelativeRange: float | None = None,
) -> Tuple[np.ndarray, bool]:
    scores_ = np.asarray(scores, dtype=np.float64)
    n = int(scores_.size)
    target = int(min(max(int(targetCount), 1), n))
    out = np.zeros(n, dtype=bool)
    if n == 0 or target <= 0:
        return out, False

    adjusted = scores_ - float(selectionPenalty)
    cumsum = np.concatenate(([0.0], np.cumsum(adjusted, dtype=np.float64)))
    bestObjective = -math.inf
    bestStart = 0
    for start in range(0, n - target + 1):
        end = start + target - 1
        windowScore = float(cumsum[end + 1] - cumsum[start])
        switchCount = int(start > 0) + int(end < n - 1)
        objective = windowScore - float(max(float(gamma), 0.0)) * float(switchCount)
        if objective > bestObjective:
            bestObjective = objective
            bestStart = start

    if not np.isfinite(bestObjective) or bestObjective <= 0.0:
        return out, False
    if maxRelativeRange is not None:
        maxRelativeRange_ = float(max(float(maxRelativeRange), 0.0))
        bestWindow = scores_[bestStart : bestStart + target]
        windowRange = float(np.max(bestWindow) - np.min(bestWindow))
        windowScale = float(max(np.max(np.abs(bestWindow)), 1.0e-6))
        if windowRange > maxRelativeRange_ * windowScale:
            return out, False
    out[bestStart : bestStart + target] = True
    return out, True


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
    diagnostics: bool = False,
    diagnosticLabel: str | None = None,
    diagnosticDetailPath: str | Path | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
    localGamma = 0.25 * parentGamma
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
    subproblemMaxIter = int(_NESTED_ROCCO_SUBTASK_MAX_ITER)
    diagnosticDetailPath_ = (
        None if diagnosticDetailPath is None else str(diagnosticDetailPath)
    )

    history: List[Dict[str, Any]] = []
    hierarchy: List[Dict[str, Any]] = []
    layers: List[Dict[str, Any]] = []
    nodeById: Dict[int, Dict[str, Any]] = {}
    rootIds: List[int] = []
    leafIds: set[int] = set()
    frontierIds: List[int] = []
    nextNodeId = 1

    def _rangeStartEnd(start: int, end: int) -> Tuple[int, int]:
        if intervals_ is not None and ends_ is not None:
            return int(intervals_[start]), int(ends_[end])
        return int(start), int(end + 1)

    def _addNode(layer: int, parentId: int | None, start: int, end: int) -> int:
        nonlocal nextNodeId
        nodeId = int(nextNodeId)
        nextNodeId += 1
        requiredLocal = int(np.argmax(rawScores_[start : end + 1]))
        rangeStart, rangeEnd = _rangeStartEnd(start, end)
        node = {
            "id": int(nodeId),
            "layer": int(layer),
            "parent_id": None if parentId is None else int(parentId),
            "start_idx": int(start),
            "end_idx": int(end),
            "range_start": int(rangeStart),
            "range_end": int(rangeEnd),
            "length_bins": int(end - start + 1),
            "selected_count": int(end - start + 1),
            "required_idx": int(start + requiredLocal),
            "summit_idx": int(start + requiredLocal),
            "raw_score_max": float(rawScores_[start + requiredLocal]),
            "score_sum": float(np.sum(scores_[start : end + 1])),
            "child_ids": [],
            "direct_child_count": 0,
            "child_count": 1,
            "leaf_count": 1,
            "candidate_child_count": 1,
            "child_widths_bins": [],
            "split_gain": 0.0,
            "split_gate": False,
            "decision": "leaf",
            "objective": None,
            "penalized_objective": None,
            "parent_penalized_objective": None,
            "boundary_penalty": None,
            "run_penalty": None,
            "run_penalty_total": None,
            "selection_penalty": None,
            "min_child_bins": int(minRegionBins_),
        }
        nodeById[nodeId] = node
        hierarchy.append(node)
        return int(nodeId)

    def _flatSelection(ids: set[int]) -> np.ndarray:
        out = np.zeros_like(inputSelection, dtype=bool)
        for nodeId in ids:
            node = nodeById[int(nodeId)]
            out[int(node["start_idx"]) : int(node["end_idx"]) + 1] = True
        return out

    def _refreshLayers() -> None:
        layers.clear()
        if not hierarchy:
            return
        maxLayer = int(max(int(node["layer"]) for node in hierarchy))
        for layerIdx in range(maxLayer + 1):
            layerNodes = [
                node for node in hierarchy if int(node["layer"]) == int(layerIdx)
            ]
            layers.append(
                {
                    "layer": int(layerIdx),
                    "node_count": int(len(layerNodes)),
                    "root_count": int(
                        sum(node["parent_id"] is None for node in layerNodes)
                    ),
                    "parent_count": int(len(layerNodes)),
                    "split_count": int(
                        sum(node["decision"] == "split" for node in layerNodes)
                    ),
                    "unsplit_count": int(
                        sum(node["decision"] != "split" for node in layerNodes)
                    ),
                    "leaf_count": int(
                        sum(int(node["id"]) in leafIds for node in layerNodes)
                    ),
                    "child_count": int(
                        sum(int(node["child_count"]) for node in layerNodes)
                    ),
                    "selected_count": int(
                        sum(int(node["selected_count"]) for node in layerNodes)
                    ),
                }
            )

    for start, end in _selectedRunBounds(current):
        nodeId = _addNode(0, None, int(start), int(end))
        rootIds.append(int(nodeId))
        leafIds.add(int(nodeId))
        frontierIds.append(int(nodeId))
    _refreshLayers()

    details: Dict[str, Any] = {
        "enabled": bool(maxIters > 0),
        "requested_iters": int(maxIters),
        "completed_iters": 0,
        "stop_reason": "disabled" if maxIters == 0 else "not_started",
        "jaccard_threshold": float(jaccardThreshold_),
        "parent_gamma": float(parentGamma),
        "local_gamma": float(localGamma),
        "selection_penalty": float(selectionPenalty_),
        "budget_scale": float(budgetScale),
        "budget_policy": _NESTED_ROCCO_BUDGET_POLICY,
        "score_shift": float(selectionPenalty_),
        "subproblem_mode": "parent_conditioned_min_run_dp",
        "subproblem_max_iter": int(subproblemMaxIter),
        "parent_edge_boundary_cost": float(_NESTED_ROCCO_PARENT_EDGE_COST),
        "min_region_bins": int(minRegionBins_),
        "min_region_bp": None if minRegionBP_ is None else int(minRegionBP_),
        "min_child_bins": int(minRegionBins_),
        "required_bin_policy": "argmax_raw_score_leftmost",
        "diagnostic_detail_path": diagnosticDetailPath_,
        "initial_selected_count": int(np.sum(current)),
        "final_selected_count": int(np.sum(current)),
        "root_ids": rootIds,
        "leaf_node_ids": sorted(int(nodeId) for nodeId in leafIds),
        "hierarchy": hierarchy,
        "layers": layers,
        "history": history,
    }
    if maxIters == 0:
        return current.astype(np.uint8), details

    diagnostics_ = bool(diagnostics)
    label_ = "" if diagnosticLabel is None else f" {diagnosticLabel}"
    diagnosticDetailHandle = None
    if diagnostics_ and diagnosticDetailPath_ is not None:
        diagnosticDetailPathObj = Path(diagnosticDetailPath_)
        diagnosticDetailPathObj.parent.mkdir(parents=True, exist_ok=True)
        diagnosticDetailHandle = diagnosticDetailPathObj.open("a", encoding="utf-8")

    def _writeSubproblemDiagnostic(row: Dict[str, Any]) -> None:
        if diagnosticDetailHandle is None:
            return
        diagnosticDetailHandle.write(json.dumps(row, sort_keys=True) + "\n")

    for iterIdx in range(maxIters):
        previous = current
        parentNodeIds = list(frontierIds)
        frontierIds = []
        skippedShort = 0
        splitParents = 0
        refinedParents = 0
        retainedParents = 0
        iterBudgetScale = budgetScale if iterIdx == 0 else 1.0
        expandedShortChildRuns = 0
        expandedShortChildBins = 0
        emptyLocalSolutions = 0
        budgetFallbackWindows = 0
        softBudgetPenaltyRegions = 0
        localPenaltyExtraTotal = 0.0
        localPenaltyExtraMax = 0.0
        requiredFallbackWindows = 0
        parentErasureViolations = 0
        requiredBinViolations = 0
        if diagnostics_:
            logger.info(
                "nested ROCCO%s iter=%d start parent_regions=%d selected=%d budget_scale=%.4g local_gamma=%.6g selection_penalty=%.6g",
                label_,
                int(iterIdx + 1),
                int(len(parentNodeIds)),
                int(np.sum(previous)),
                float(iterBudgetScale),
                float(localGamma),
                float(selectionPenalty_),
            )
        for regionIdx, parentId in enumerate(parentNodeIds, start=1):
            parentNode = nodeById[int(parentId)]
            start = int(parentNode["start_idx"])
            end = int(parentNode["end_idx"])
            regionLengthBP = _selectedRunLengthBP(start, end, intervals_, ends_)
            if (minRegionBP_ is not None and regionLengthBP < minRegionBP_) or (
                minRegionBP_ is None and (end - start + 1) < minRegionBins_
            ):
                parentNode["decision"] = "skipped_short"
                parentNode["child_count"] = 1
                parentNode["leaf_count"] = 1
                retainedParents += 1
                skippedShort += 1
                if diagnostics_:
                    _writeSubproblemDiagnostic(
                        {
                            "event": "subproblem",
                            "status": "skipped_short",
                            "chromosome": diagnosticLabel,
                            "iter": int(iterIdx + 1),
                            "layer": int(iterIdx + 1),
                            "region": int(regionIdx),
                            "parent_id": int(parentId),
                            "bins": int(end - start + 1),
                            "bp": int(regionLengthBP),
                            "budget_policy": _NESTED_ROCCO_BUDGET_POLICY,
                            "child_count": 1,
                            "split_gain": 0.0,
                            "split_gate": False,
                            "decision": "skipped_short",
                        }
                    )
                continue
            localScores = scores_[start : end + 1]
            localMinChildBins = _minimumChildBinsForRegion(
                start,
                end,
                intervals_,
                ends_,
                minRegionBP_,
                minRegionBins_,
            )
            localSoftBudgetTarget = _nestedSoftBudgetTargetCount(
                end - start + 1,
                iterBudgetScale,
                localMinChildBins,
            )
            localRawScores = rawScores_[start : end + 1]
            requiredLocal = int(np.argmax(localRawScores))
            localScoreFloor = 0.0
            localSolverScores = localScores
            if iterIdx > 0:
                localScoreFloor = float(np.quantile(localScores, 0.25))
                localSolverScores = np.asarray(
                    localScores - localScoreFloor,
                    dtype=np.float64,
                )
            localPenaltyDetails: Dict[str, float]
            localSelectionPenalty, localPenaltyDetails = _nestedSoftSelectionPenalty(
                localSolverScores,
                0.0 if iterIdx > 0 else selectionPenalty_,
                iterBudgetScale,
            )
            localBudgetPenalty = float(localSelectionPenalty)
            if iterBudgetScale < 1.0:
                localMode = "parent_conditioned_min_run_soft_budget"
                softBudgetPenaltyRegions += 1
            else:
                localMode = "parent_conditioned_min_run_dp"
            nLocal = int(end - start + 1)
            internalBoundaryCost = float(
                max(float(localGamma), 1000.0 * _NESTED_ROCCO_PARENT_EDGE_COST)
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
            localRequiredFallbackUsed = bool(
                localDetails.get("required_fallback_window", False)
            )
            penaltyExtra = float(localPenaltyDetails["extra_penalty"])
            localPenaltyExtraTotal += penaltyExtra
            localPenaltyExtraMax = float(max(localPenaltyExtraMax, penaltyExtra))
            if not bool(np.any(localSelection)):
                emptyLocalSolutions += 1
                raise RuntimeError("parent-conditioned subpeak solve selected no bins")
            if not bool(localSelection[requiredLocal]):
                requiredBinViolations += 1
                raise RuntimeError(
                    "parent-conditioned subpeak solve violated required bin"
                )
            if localRequiredFallbackUsed:
                requiredFallbackWindows += 1
            localRuns = _selectedRunBounds(localSelection)
            candidateChildCount = int(len(localRuns))
            childWidths = [int(right - left + 1) for left, right in localRuns]
            childWidthsOk = bool(
                candidateChildCount >= 1
                and all(width >= int(localMinChildBins) for width in childWidths)
            )
            retainedSelection = np.ones(nLocal, dtype=bool)
            (
                _parentObjective,
                retainedPenalized,
                _parentBoundaryPenalty,
                _parentRunPenalty,
            ) = (
                _parentConditionedSubpeakObjective(
                    localSolverScores,
                    retainedSelection,
                    localBoundaryCosts,
                    localSelectionPenalty,
                    runPenalty=internalBoundaryCost,
                )
            )
            splitGain = float(
                float(localDetails["penalized_objective"]) - float(retainedPenalized)
            )
            changedSelection = not bool(
                np.array_equal(localSelection, retainedSelection)
            )
            splitGate = bool(
                candidateChildCount >= 2 and childWidthsOk and splitGain > 0.0
            )
            refineGate = bool(
                candidateChildCount == 1
                and childWidthsOk
                and changedSelection
                and splitGain > 0.0
            )
            decision = (
                "split"
                if splitGate
                else ("shrink" if refineGate else "keep_parent")
            )
            parentNode["decision"] = str(decision)
            parentNode["candidate_child_count"] = int(candidateChildCount)
            parentNode["child_widths_bins"] = [int(width) for width in childWidths]
            parentNode["split_gain"] = float(splitGain)
            parentNode["split_gate"] = bool(splitGate)
            parentNode["objective"] = float(_localObjective)
            parentNode["penalized_objective"] = float(
                localDetails["penalized_objective"]
            )
            parentNode["parent_penalized_objective"] = float(retainedPenalized)
            parentNode["boundary_penalty"] = float(localDetails["boundary_penalty"])
            parentNode["run_penalty"] = float(localDetails["run_penalty"])
            parentNode["run_penalty_total"] = float(localDetails["run_penalty_total"])
            parentNode["selection_penalty"] = float(localSelectionPenalty)
            parentNode["local_score_floor"] = float(localScoreFloor)
            parentNode["min_child_bins"] = int(localMinChildBins)
            if splitGate or refineGate:
                childIds: List[int] = []
                for localLeft, localRight in localRuns:
                    childId = _addNode(
                        int(iterIdx + 1),
                        int(parentId),
                        int(start + localLeft),
                        int(start + localRight),
                    )
                    childIds.append(int(childId))
                    leafIds.add(int(childId))
                    frontierIds.append(int(childId))
                leafIds.remove(int(parentId))
                parentNode["child_ids"] = childIds
                parentNode["direct_child_count"] = int(len(childIds))
                parentNode["child_count"] = int(len(childIds))
                parentNode["leaf_count"] = int(len(childIds))
                if splitGate:
                    splitParents += 1
                else:
                    refinedParents += 1
            else:
                parentNode["child_ids"] = []
                parentNode["direct_child_count"] = 0
                parentNode["child_count"] = 1
                parentNode["leaf_count"] = 1
                retainedParents += 1
            if diagnostics_:
                regionStartBP, regionEndBP = _rangeStartEnd(start, end)
                selectedLocal = int(np.sum(localSelection))
                selectedNonPositive = int(
                    np.sum(localSelection & (localRawScores <= 0.0))
                )
                _writeSubproblemDiagnostic(
                    {
                        "event": "subproblem",
                        "status": "solved",
                        "chromosome": diagnosticLabel,
                        "iter": int(iterIdx + 1),
                        "layer": int(iterIdx + 1),
                        "region": int(regionIdx),
                        "parent_id": int(parentId),
                        "mode": str(localMode),
                        "bins": int(end - start + 1),
                        "bp": int(regionLengthBP),
                        "range_start": int(regionStartBP),
                        "range_end": int(regionEndBP),
                        "selected": int(selectedLocal),
                        "selected_possible": int(end - start + 1),
                        "nonpos_selected": int(selectedNonPositive),
                        "min_child_bins": int(localMinChildBins),
                        "budget_target": int(localSoftBudgetTarget),
                        "soft_budget_target": int(localSoftBudgetTarget),
                        "budget_policy": _NESTED_ROCCO_BUDGET_POLICY,
                        "required_local": int(requiredLocal),
                        "required_selected": bool(localSelection[requiredLocal]),
                        "empty_solution": False,
                        "required_fallback": bool(localRequiredFallbackUsed),
                        "objective": float(_localObjective),
                        "penalized": float(localDetails["penalized_objective"]),
                        "solver_penalty": float(localDetails["selection_penalty"]),
                        "budget_penalty": float(localBudgetPenalty),
                        "soft_penalty_extra": float(
                            localPenaltyDetails["extra_penalty"]
                        ),
                        "run_penalty": float(localDetails["run_penalty"]),
                        "run_penalty_total": float(
                            localDetails["run_penalty_total"]
                        ),
                        "local_score_floor": float(localScoreFloor),
                        "score_min": float(np.min(localRawScores)),
                        "score_max": float(np.max(localRawScores)),
                        "score_mean": float(np.mean(localRawScores)),
                        "child_count": int(parentNode["child_count"]),
                        "candidate_child_count": int(candidateChildCount),
                        "split_gain": float(splitGain),
                        "split_gate": bool(splitGate),
                        "decision": str(decision),
                    }
                )

        newSelection = _flatSelection(leafIds)
        if bool(np.any(newSelection & ~inputSelection)):
            raise RuntimeError("nested ROCCO selection left the input solution")
        jaccard = _maskJaccard(previous, newSelection)
        selectedBefore = int(np.sum(previous))
        selectedAfter = int(np.sum(newSelection))
        objectivePrevious = _roccoObjectiveForSolution(
            scores_,
            previous.astype(np.uint8),
            parentGamma,
        )
        objectiveAfter = _roccoObjectiveForSolution(
            scores_,
            newSelection.astype(np.uint8),
            parentGamma,
        )
        previousRuns = _selectedRunBounds(previous)
        runsAfter = _selectedRunBounds(newSelection)
        peakCountMonotonicityViolations = int(
            max(len(previousRuns) - len(runsAfter), 0)
        )
        coverageExpansionViolations = int(max(selectedAfter - selectedBefore, 0))
        _refreshLayers()
        details["leaf_node_ids"] = sorted(int(nodeId) for nodeId in leafIds)
        history.append(
            {
                "iter": int(iterIdx + 1),
                "layer": int(iterIdx + 1),
                "num_parent_peaks": int(len(parentNodeIds)),
                "num_parent_peaks_after": int(len(runsAfter)),
                "num_input_regions": int(len(parentNodeIds)),
                "num_next_parent_nodes": int(len(frontierIds)),
                "num_split_parent_nodes": int(splitParents),
                "num_refined_parent_nodes": int(refinedParents),
                "num_retained_parent_nodes": int(retainedParents),
                "num_skipped_short_regions": int(skippedShort),
                "num_budget_constrained_regions": int(softBudgetPenaltyRegions),
                "num_soft_budget_penalty_regions": int(softBudgetPenaltyRegions),
                "num_empty_local_solutions": int(emptyLocalSolutions),
                "num_budget_fallback_windows": int(budgetFallbackWindows),
                "num_required_fallback_windows": int(requiredFallbackWindows),
                "num_parent_erasure_violations": int(parentErasureViolations),
                "num_required_bin_violations": int(requiredBinViolations),
                "num_peak_count_monotonicity_violations": int(
                    peakCountMonotonicityViolations
                ),
                "num_coverage_expansion_violations": int(coverageExpansionViolations),
                "num_short_child_runs_expanded": int(expandedShortChildRuns),
                "num_short_child_bins_added": int(expandedShortChildBins),
                "local_penalty_extra_mean": float(
                    localPenaltyExtraTotal / max(len(parentNodeIds) - skippedShort, 1)
                ),
                "local_penalty_extra_max": float(localPenaltyExtraMax),
                "budget_scale": float(iterBudgetScale),
                "budget_policy": _NESTED_ROCCO_BUDGET_POLICY,
                "selected_count_before": int(selectedBefore),
                "selected_count_after": int(selectedAfter),
                "selected_count_delta": int(selectedAfter - selectedBefore),
                "objective": float(objectiveAfter),
                "objective_previous": float(objectivePrevious),
                "objective_delta": float(objectiveAfter - objectivePrevious),
                "jaccard": float(jaccard),
            }
        )
        if diagnostics_:
            logger.info(
                "nested ROCCO%s iter=%d done selected_before=%d selected_after=%d parent_regions=%d parent_regions_after=%d skipped_short=%d budget_constrained=%d empty_solutions=%d required_fallback=%d parent_erasure_violations=%d required_bin_violations=%d peak_count_violations=%d coverage_expansion_violations=%d short_child_expanded=%d local_penalty_extra_mean=%.6g objective=%.6g objective_delta=%.6g jaccard=%.6f",
                label_,
                int(iterIdx + 1),
                int(selectedBefore),
                int(selectedAfter),
                int(len(parentNodeIds)),
                int(len(runsAfter)),
                int(skippedShort),
                int(softBudgetPenaltyRegions),
                int(emptyLocalSolutions),
                int(requiredFallbackWindows),
                int(parentErasureViolations),
                int(requiredBinViolations),
                int(peakCountMonotonicityViolations),
                int(coverageExpansionViolations),
                int(expandedShortChildRuns),
                float(
                    localPenaltyExtraTotal / max(len(parentNodeIds) - skippedShort, 1)
                ),
                float(objectiveAfter),
                float(objectiveAfter - objectivePrevious),
                float(jaccard),
            )

        details["completed_iters"] = int(iterIdx + 1)
        current = newSelection
        details["final_selected_count"] = int(selectedAfter)

        if np.array_equal(newSelection, previous):
            details["stop_reason"] = "mask_equal"
            if diagnostics_:
                logger.info(
                    "nested ROCCO%s stop iter=%d reason=mask_equal",
                    label_,
                    int(iterIdx + 1),
                )
            break
        if not frontierIds:
            details["stop_reason"] = "no_splits"
            if diagnostics_:
                logger.info(
                    "nested ROCCO%s stop iter=%d reason=no_splits",
                    label_,
                    int(iterIdx + 1),
                )
            break
        if jaccard >= jaccardThreshold_:
            details["stop_reason"] = "jaccard"
            if diagnostics_:
                logger.info(
                    "nested ROCCO%s stop iter=%d reason=jaccard threshold=%.6f observed=%.6f",
                    label_,
                    int(iterIdx + 1),
                    float(jaccardThreshold_),
                    float(jaccard),
                )
            break
    else:
        details["stop_reason"] = "max_iter"
        if diagnostics_:
            logger.info(
                "nested ROCCO%s stop iter=%d reason=max_iter",
                label_,
                int(maxIters),
            )

    if diagnosticDetailHandle is not None:
        diagnosticDetailHandle.close()
    return current.astype(np.uint8), details


def _readAlignedConsenrichBedGraphs(
    stateBedGraphFile: str,
    uncertaintyBedGraphFile: str | None = None,
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
    stateDF.sort_values(
        by=["chromosome", "start", "end"],
        kind="mergesort",
        inplace=True,
    )
    stateDF.reset_index(drop=True, inplace=True)

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
        uncertaintyDF.sort_values(
            by=["chromosome", "start", "end"],
            kind="mergesort",
            inplace=True,
        )
        uncertaintyDF.reset_index(drop=True, inplace=True)
        if not stateDF[["chromosome", "start", "end"]].equals(
            uncertaintyDF[["chromosome", "start", "end"]]
        ):
            raise ValueError(
                "`stateBedGraphFile` and `uncertaintyBedGraphFile` are not aligned."
            )

    allowedChroms = set(chromosomes) if chromosomes is not None else None
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for chromosome, chromStateDF in stateDF.groupby("chromosome", sort=False):
        if allowedChroms is not None and chromosome not in allowedChroms:
            continue
        chromUncDF = None
        if uncertaintyDF is not None:
            chromUncDF = uncertaintyDF[uncertaintyDF["chromosome"] == chromosome]
        out[str(chromosome)] = {
            "intervals": chromStateDF["start"].to_numpy(dtype=np.int64, copy=True),
            "ends": chromStateDF["end"].to_numpy(dtype=np.int64, copy=True),
            "state": chromStateDF["state"].to_numpy(dtype=np.float64, copy=True),
            "uncertainty": (
                chromUncDF["uncertainty"].to_numpy(dtype=np.float64, copy=True)
                if chromUncDF is not None
                else None
            ),
        }
    return out


def _solveParentConditionedSubpeakSegments(
    segmentScores: np.ndarray,
    segmentState: np.ndarray,
    startIdx: int,
    endIdx: int,
    selectionPenalty: float,
    boundaryCost: float,
    minRunBins: int,
) -> List[Dict[str, int | float | bool]]:
    segScores = np.asarray(segmentScores, dtype=np.float64)
    segState = np.asarray(segmentState, dtype=np.float64)
    if segScores.size != segState.size:
        raise ValueError("`segmentScores` and `segmentState` must match")
    requiredLocal = int(np.argmax(segScores))
    localMask, _objective, details = _solveParentConditionedSubpeaks(
        segScores,
        boundaryCosts=float(max(float(boundaryCost), 0.0)),
        selectionPenalty=float(selectionPenalty),
        minRunBins=int(max(int(minRunBins), 1)),
        requiredIndex=requiredLocal,
    )
    runs = _selectedRunBounds(localMask)
    if len(runs) == 0:
        summitLocal = int(np.argmax(segState))
        return [
            {
                "start_idx": int(startIdx),
                "end_idx": int(endIdx),
                "summit_idx": int(startIdx + summitLocal),
                "segment_length_bins": int(max(endIdx - startIdx + 1, 0)),
                "num_subpeaks": 1,
                "split_from_parent": False,
                "subpeak_objective": float(details["penalized_objective"]),
                "subpeak_boundary_penalty": float(details["boundary_penalty"]),
                "massive_subpeak_cleanup_candidate": False,
                "massive_subpeak_cleanup_applied": False,
                "massive_subpeak_split_gain": None,
                "massive_subpeak_split_z": None,
                "massive_subpeak_gap_bins": None,
                "massive_subpeak_parent_start_idx": int(startIdx),
                "massive_subpeak_parent_end_idx": int(endIdx),
            }
        ]
    out: List[Dict[str, int | float | bool]] = []
    numSubpeaks = int(len(runs))
    for localLeft, localRight in runs:
        childState = segState[localLeft : localRight + 1]
        summitLocal = int(localLeft + np.argmax(childState))
        out.append(
            {
                "start_idx": int(startIdx + localLeft),
                "end_idx": int(startIdx + localRight),
                "summit_idx": int(startIdx + summitLocal),
                "segment_length_bins": int(localRight - localLeft + 1),
                "num_subpeaks": int(numSubpeaks),
                "split_from_parent": bool(
                    numSubpeaks > 1 or localLeft > 0 or localRight < segState.size - 1
                ),
                "subpeak_objective": float(details["penalized_objective"]),
                "subpeak_boundary_penalty": float(details["boundary_penalty"]),
                "massive_subpeak_cleanup_candidate": False,
                "massive_subpeak_cleanup_applied": False,
                "massive_subpeak_split_gain": None,
                "massive_subpeak_split_z": None,
                "massive_subpeak_gap_bins": None,
                "massive_subpeak_parent_start_idx": int(startIdx),
                "massive_subpeak_parent_end_idx": int(endIdx),
            }
        )
    return out


def _forceMassiveSubpeakSegments(
    child: Mapping[str, Any],
    scores: np.ndarray,
    state: np.ndarray,
    intervals: np.ndarray,
    ends: np.ndarray,
    widthPolicy: Mapping[str, Any] | None,
    boundaryCost: float,
    minRunBins: int,
    splitQuantile: float = _MASSIVE_SUBPEAK_SPLIT_QUANTILE,
    minSplitZ: float = _MASSIVE_SUBPEAK_SPLIT_Z,
    maxDepth: int = _MASSIVE_SUBPEAK_MAX_DEPTH,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    threshold = None if widthPolicy is None else widthPolicy.get("width_threshold_bp")
    if threshold is None:
        return [dict(child)], {
            "candidates": 0,
            "splits": 0,
            "segments_added": 0,
            "evaluated": 0,
            "contracts": 0,
        }
    threshold_ = float(threshold)
    contractThreshold = float(
        widthPolicy.get(
            "contract_width_bp",
            widthPolicy.get("min_bp", _MASSIVE_SUBPEAK_MIN_BP),
        )
    )
    if (not np.isfinite(contractThreshold)) or contractThreshold <= 0.0:
        contractThreshold = float(_MASSIVE_SUBPEAK_MIN_BP)
    contractThreshold = float(min(contractThreshold, threshold_))
    scores_ = np.asarray(scores, dtype=np.float64)
    state_ = np.asarray(state, dtype=np.float64)
    intervals_ = np.asarray(intervals, dtype=np.int64)
    ends_ = np.asarray(ends, dtype=np.int64)
    originalStart = int(child["start_idx"])
    originalEnd = int(child["end_idx"])
    binWidths = np.asarray(
        ends_[originalStart : originalEnd + 1]
        - intervals_[originalStart : originalEnd + 1],
        dtype=np.int64,
    )
    binWidths = binWidths[binWidths > 0]
    stepBP = int(max(int(np.median(binWidths)), 1)) if binWidths.size else 1
    minBins = int(
        max(
            int(minRunBins),
            int(math.ceil(float(_MASSIVE_SUBPEAK_MIN_CHILD_BP) / float(stepBP))),
            int(
                math.ceil(
                    float(originalEnd - originalStart + 1)
                    * float(_MASSIVE_SUBPEAK_MIN_CHILD_FRACTION)
                )
            ),
            1,
        )
    )
    contractMinBins = int(
        max(
            int(minRunBins),
            int(math.ceil(float(_MASSIVE_SUBPEAK_MIN_CHILD_BP) / float(stepBP))),
            1,
        )
    )
    maxDepth_ = int(max(int(maxDepth), 0))
    counts = {
        "candidates": 0,
        "splits": 0,
        "segments_added": 0,
        "evaluated": 0,
        "contracts": 0,
    }
    objective = float(child.get("subpeak_objective", 0.0))
    boundaryPenalty = float(child.get("subpeak_boundary_penalty", 0.0))

    def _contract_or_keep(
        start: int,
        end: int,
        split: Mapping[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        contraction = _contractMassiveSubpeakSegment(
            start,
            end,
            scores_,
            intervals_,
            ends_,
            thresholdBP=contractThreshold,
            minRunBins=contractMinBins,
            splitQuantile=float(splitQuantile),
        )
        if contraction is None:
            return [
                _makeSubpeakSegment(
                    start,
                    end,
                    state_,
                    originalStart,
                    originalEnd,
                    1,
                    bool(start != originalStart or end != originalEnd),
                    objective,
                    boundaryPenalty,
                    cleanupCandidate=True,
                    cleanupApplied=False,
                    cleanupDetails=split,
                    cleanupMode="unsplit",
                )
            ]
        contractStart, contractEnd, contractDetails = contraction
        details = {} if split is None else dict(split)
        details.update(contractDetails)
        counts["contracts"] += 1
        return [
            _makeSubpeakSegment(
                contractStart,
                contractEnd,
                state_,
                originalStart,
                originalEnd,
                1,
                True,
                objective,
                boundaryPenalty,
                cleanupCandidate=True,
                cleanupApplied=True,
                cleanupDetails=details,
                cleanupMode=str(contractDetails["mode"]),
            )
        ]

    def _recurse(start: int, end: int, depth: int) -> List[Dict[str, Any]]:
        widthBP = int(max(int(ends_[end]) - int(intervals_[start]), 0))
        needsSplitSearch = bool(widthBP >= threshold_)
        needsChildContract = bool(depth > 0 and widthBP >= contractThreshold)
        candidate = bool(needsSplitSearch or needsChildContract)
        if not candidate:
            return [
                _makeSubpeakSegment(
                    start,
                    end,
                    state_,
                    originalStart,
                    originalEnd,
                    1,
                    bool(start != originalStart or end != originalEnd),
                    objective,
                    boundaryPenalty,
                    cleanupCandidate=candidate,
                    cleanupApplied=False,
                )
            ]
        if not needsSplitSearch:
            counts["candidates"] += 1
            counts["evaluated"] += 1
            return _contract_or_keep(start, end, None)
        if depth >= maxDepth_:
            counts["candidates"] += 1
            counts["evaluated"] += 1
            return _contract_or_keep(start, end, None)
        counts["candidates"] += 1
        counts["evaluated"] += 1
        localScores = scores_[start : end + 1]
        split = _bestMassiveSubpeakSplit(
            localScores,
            boundaryCost=float(boundaryCost),
            minRunBins=minBins,
            splitQuantile=float(splitQuantile),
        )
        if (
            split is None
            or float(split["gain"]) <= 0.0
            or float(split["z"]) < float(minSplitZ)
        ):
            return _contract_or_keep(start, end, split)
        gapStart = int(start + int(split["gap_start_local"]))
        gapEnd = int(start + int(split["gap_end_local"]))
        leftEnd = int(gapStart - 1)
        rightStart = int(gapEnd + 1)
        if leftEnd - start + 1 < minBins or end - rightStart + 1 < minBins:
            return _contract_or_keep(start, end, split)
        counts["splits"] += 1
        left = _recurse(start, leftEnd, depth + 1)
        right = _recurse(rightStart, end, depth + 1)
        merged = left + right
        for segment in merged:
            segment["massive_subpeak_cleanup_candidate"] = True
            segment["massive_subpeak_cleanup_applied"] = True
            segment["massive_subpeak_cleanup_mode"] = (
                segment.get("massive_subpeak_cleanup_mode") or "split"
            )
            segment["massive_subpeak_split_gain"] = float(split["gain"])
            segment["massive_subpeak_split_z"] = float(split["z"])
            segment["massive_subpeak_gap_bins"] = int(split["gap_bins"])
        return merged

    out = _recurse(originalStart, originalEnd, 0)
    if len(out) > 1:
        counts["segments_added"] += int(len(out) - 1)
    for segment in out:
        segment["num_subpeaks"] = int(len(out))
        segment["split_from_parent"] = bool(
            len(out) > 1
            or int(segment["start_idx"]) != originalStart
            or int(segment["end_idx"]) != originalEnd
            or bool(child.get("split_from_parent", False))
        )
    return out, counts


def _trimChildSegmentAroundSummit(
    childStartIdx: int,
    childEndIdx: int,
    summitIdx: int,
    scores: np.ndarray,
    trimScoreFloor: float | None,
) -> Tuple[int, int, bool]:
    if trimScoreFloor is None:
        return int(childStartIdx), int(childEndIdx), False
    floor = float(trimScoreFloor)
    if not np.isfinite(floor):
        return int(childStartIdx), int(childEndIdx), False

    start = int(childStartIdx)
    end = int(childEndIdx)
    summit = int(np.clip(int(summitIdx), start, end))
    localSummit = int(summit - start)
    childScores = np.asarray(scores[start : end + 1], dtype=np.float64)
    if childScores.size == 0:
        return start, end, False

    keep = childScores > floor
    if not bool(keep[localSummit]):
        return start, end, False

    left = localSummit
    while left > 0 and bool(keep[left - 1]):
        left -= 1
    right = localSummit
    while right + 1 < keep.size and bool(keep[right + 1]):
        right += 1

    trimmedStart = int(start + left)
    trimmedEnd = int(start + right)
    trimmed = bool(trimmedStart != start or trimmedEnd != end)
    return trimmedStart, trimmedEnd, trimmed


def _coerceHierarchyNodeId(node: Mapping[str, Any]) -> str:
    for key in ("peak_id", "peakId", "node_id", "nodeId", "id"):
        value = node.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Nested hierarchy nodes require a peak id")


def _coerceHierarchyIdx(node: Mapping[str, Any], snakeKey: str, camelKey: str) -> int:
    if snakeKey in node:
        return int(node[snakeKey])
    if camelKey in node:
        return int(node[camelKey])
    raise ValueError(f"Nested hierarchy nodes require `{snakeKey}`")


def _coerceRoccoNestedHierarchy(
    nestedHierarchy: Any,
) -> Dict[str, Any] | None:
    if nestedHierarchy is None:
        return None
    if isinstance(nestedHierarchy, Mapping):
        if "nodes" in nestedHierarchy:
            sourceNodes = nestedHierarchy["nodes"]
        elif "roots" in nestedHierarchy:
            sourceNodes = nestedHierarchy["roots"]
        else:
            sourceNodes = [nestedHierarchy]
    elif isinstance(nestedHierarchy, (list, tuple)):
        sourceNodes = nestedHierarchy
    else:
        raise ValueError("`nestedHierarchy` must be a mapping or sequence")
    if not isinstance(sourceNodes, (list, tuple)):
        raise ValueError("Nested hierarchy `nodes` must be a sequence")

    flatNodes: List[Dict[str, Any]] = []
    seenNodeIds: set[str] = set()

    def addNode(nodeAny: Any, parentId: str | None, layer: int) -> None:
        if not isinstance(nodeAny, Mapping):
            raise ValueError("Nested hierarchy nodes must be mappings")
        node = dict(nodeAny)
        peakId = _coerceHierarchyNodeId(node)
        if peakId in seenNodeIds:
            raise ValueError(f"Duplicate nested hierarchy peak id: {peakId}")
        seenNodeIds.add(peakId)
        startIdx = _coerceHierarchyIdx(node, "start_idx", "startIdx")
        endIdx = _coerceHierarchyIdx(node, "end_idx", "endIdx")
        if startIdx > endIdx:
            raise ValueError("Nested hierarchy node start_idx exceeds end_idx")
        suppliedParent = node.get(
            "parent_peak_id",
            node.get(
                "parentPeakId",
                node.get("parent_id", node.get("parentId", parentId)),
            ),
        )
        if suppliedParent is not None and parentId is not None:
            suppliedParent = str(suppliedParent)
            if suppliedParent != parentId:
                raise ValueError("Nested hierarchy parent ids disagree")
        parentPeakId = None if suppliedParent is None else str(suppliedParent)
        suppliedLayer = node.get("layer")
        layerSupplied = suppliedLayer is not None
        layer_ = int(layer if suppliedLayer is None else suppliedLayer)
        if layer_ < 0:
            raise ValueError("Nested hierarchy layer must be non-negative")
        if parentId is not None and suppliedLayer is not None and layer_ != layer:
            raise ValueError("Nested hierarchy layer disagrees with tree depth")
        flatNodes.append(
            {
                "peak_id": peakId,
                "parent_peak_id": parentPeakId,
                "root_peak_id": node.get("root_peak_id", node.get("rootPeakId")),
                "layer": int(layer_),
                "start_idx": int(startIdx),
                "end_idx": int(endIdx),
                "start": (
                    None
                    if node.get("start") is None
                    else int(node.get("start"))
                ),
                "end": None if node.get("end") is None else int(node.get("end")),
                "children": [],
                "layer_supplied": bool(layerSupplied),
            }
        )
        children = node.get("children", [])
        if children is None:
            children = []
        if not isinstance(children, (list, tuple)):
            raise ValueError("Nested hierarchy node children must be a sequence")
        for child in children:
            addNode(child, peakId, layer_ + 1)

    for sourceNode in sourceNodes:
        addNode(sourceNode, None, 0)

    nodesById = {str(node["peak_id"]): node for node in flatNodes}
    childrenById: Dict[str, List[Dict[str, Any]]] = {
        str(node["peak_id"]): [] for node in flatNodes
    }
    roots: List[Dict[str, Any]] = []
    for node in flatNodes:
        parentId = node["parent_peak_id"]
        if parentId is None:
            roots.append(node)
            continue
        if parentId not in nodesById:
            raise ValueError(f"Unknown nested hierarchy parent id: {parentId}")
        parent = nodesById[parentId]
        if int(parent["start_idx"]) > int(node["start_idx"]) or int(
            node["end_idx"]
        ) > int(parent["end_idx"]):
            raise ValueError("Nested hierarchy child is outside its parent")
        childrenById[parentId].append(node)
        parent["children"].append(node)
    if flatNodes and not roots:
        raise ValueError("Nested hierarchy requires at least one root")

    def assignRoot(node: Dict[str, Any], rootPeakId: str, layer: int) -> None:
        if bool(node["layer_supplied"]) and int(node["layer"]) != int(layer):
            raise ValueError("Nested hierarchy layer disagrees with parent links")
        node["root_peak_id"] = str(rootPeakId)
        node["layer"] = int(layer)
        for child in node["children"]:
            assignRoot(child, rootPeakId, layer + 1)

    for root in roots:
        assignRoot(root, str(root["peak_id"]), 0)
    for node in flatNodes:
        node.pop("layer_supplied", None)

    def validateSiblings(nodes: List[Dict[str, Any]]) -> None:
        lastEnd = -1
        for node in nodes:
            startIdx = int(node["start_idx"])
            endIdx = int(node["end_idx"])
            if startIdx <= lastEnd:
                raise ValueError("Nested hierarchy siblings overlap")
            lastEnd = endIdx

    for nodeChildren in childrenById.values():
        nodeChildren.sort(
            key=lambda child: (int(child["start_idx"]), int(child["end_idx"]))
        )
        validateSiblings(nodeChildren)
    roots.sort(key=lambda root: (int(root["start_idx"]), int(root["end_idx"])))
    validateSiblings(roots)
    for node in flatNodes:
        node["child_starts"] = [int(child["start_idx"]) for child in node["children"]]
    return {
        "nodes": flatNodes,
        "roots": roots,
        "root_starts": [int(root["start_idx"]) for root in roots],
        "nodes_by_id": nodesById,
        "children_by_id": childrenById,
    }


def _resolveRoccoExportHierarchy(
    nestedHierarchy: Any = None,
    nestedDetails: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if nestedHierarchy is not None:
        return _coerceRoccoNestedHierarchy(nestedHierarchy)
    if nestedDetails is None:
        return None
    for key in ("nestedHierarchy", "nested_hierarchy", "hierarchy"):
        if key in nestedDetails:
            return _coerceRoccoNestedHierarchy(nestedDetails[key])
    return None


def _summarizeRoccoHierarchy(
    hierarchy: Mapping[str, Any] | None,
    rowMeta: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    if hierarchy is None:
        return {}
    nodes = list(hierarchy.get("nodes", []))
    roots = list(hierarchy.get("roots", []))
    layerCounts: Dict[str, int] = {}
    for node in nodes:
        layerKey = str(int(node["layer"]))
        layerCounts[layerKey] = int(layerCounts.get(layerKey, 0) + 1)
    exportedLayerCounts: Dict[str, int] = {}
    exportedRoots: set[str] = set()
    exportedParents: set[str] = set()
    exportedWithoutParent = 0
    for peak in rowMeta:
        if "layer" in peak:
            layerKey = str(int(peak["layer"]))
            exportedLayerCounts[layerKey] = int(
                exportedLayerCounts.get(layerKey, 0) + 1
            )
        rootPeakId = peak.get("root_peak_id")
        if rootPeakId is not None:
            exportedRoots.add(str(rootPeakId))
        parentPeakId = peak.get("parent_peak_id")
        if parentPeakId is None:
            exportedWithoutParent += 1
        else:
            exportedParents.add(str(parentPeakId))
    parentNodeIds = {
        str(node["peak_id"])
        for node in nodes
        if node.get("parent_peak_id") is None or len(node.get("children", [])) > 0
    }
    supportParentIds = {str(root["peak_id"]) for root in roots}
    return {
        "supplied": True,
        "node_count": int(len(nodes)),
        "root_count": int(len(roots)),
        "layer_counts": layerCounts,
        "exported_layer_counts": exportedLayerCounts,
        "exported_with_parent": int(len(rowMeta) - exportedWithoutParent),
        "exported_without_parent": int(exportedWithoutParent),
        "parent_node_count": int(len(parentNodeIds)),
        "parent_node_retained": int(len(parentNodeIds & exportedParents)),
        "parent_node_dropped": int(len(parentNodeIds - exportedParents)),
        "support_parent_count": int(len(supportParentIds)),
        "support_parent_retained": int(len(supportParentIds & exportedRoots)),
        "support_parent_dropped": int(len(supportParentIds - exportedRoots)),
    }


def _findRoccoHierarchyParentForPeak(
    hierarchy: Mapping[str, Any],
    peak: Mapping[str, Any],
) -> Dict[str, Any]:
    peakStartIdx = int(peak["start_idx"])
    peakEndIdx = int(peak["end_idx"])
    roots = list(hierarchy.get("roots", []))
    rootStarts = list(hierarchy.get("root_starts", []))
    rootIndex = int(bisect_right(rootStarts, peakStartIdx) - 1)
    if rootIndex < 0:
        raise ValueError("Exported peak is outside supplied nested hierarchy")
    containingRoot = roots[rootIndex]
    if int(containingRoot["start_idx"]) > peakStartIdx or peakEndIdx > int(
        containingRoot["end_idx"]
    ):
        raise ValueError("Exported peak is outside supplied nested hierarchy")

    strictParent = containingRoot
    node = containingRoot
    while True:
        children = list(node.get("children", []))
        childStarts = list(node.get("child_starts", []))
        childIndex = int(bisect_right(childStarts, peakStartIdx) - 1)
        if childIndex < 0:
            break
        nextNode = children[childIndex]
        if int(nextNode["start_idx"]) > peakStartIdx or peakEndIdx > int(
            nextNode["end_idx"]
        ):
            break
        if int(nextNode["start_idx"]) < peakStartIdx or peakEndIdx < int(
            nextNode["end_idx"]
        ):
            strictParent = nextNode
        node = nextNode
    return strictParent


def _annotateRoccoPeakHierarchy(
    rowMeta: List[Dict[str, Any]],
    hierarchy: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if hierarchy is None:
        return {}
    assignments: List[Tuple[int, Dict[str, Any]]] = []
    for rowIndex, peak in enumerate(rowMeta):
        parent = _findRoccoHierarchyParentForPeak(hierarchy, peak)
        assignments.append((rowIndex, parent))
    siblingGroups: Dict[str, List[int]] = {}
    for rowIndex, parent in assignments:
        parentPeakId = str(parent["peak_id"])
        siblingGroups.setdefault(parentPeakId, []).append(rowIndex)
    for siblingIndexes in siblingGroups.values():
        siblingIndexes.sort(
            key=lambda idx: (
                int(rowMeta[idx]["start_idx"]),
                int(rowMeta[idx]["end_idx"]),
                str(rowMeta[idx]["name"]),
            )
        )
    ordinalByRowIndex: Dict[int, int] = {}
    for siblingIndexes in siblingGroups.values():
        for ordinal, siblingIndex in enumerate(siblingIndexes, start=1):
            ordinalByRowIndex[int(siblingIndex)] = int(ordinal)
    for rowIndex, parent in assignments:
        parentPeakId = str(parent["peak_id"])
        siblings = siblingGroups[parentPeakId]
        rowMeta[rowIndex].update(
            {
                "root_peak_id": str(parent["root_peak_id"]),
                "parent_peak_id": parentPeakId,
                "layer": int(parent["layer"]) + 1,
                "child_ordinal": int(ordinalByRowIndex[rowIndex]),
                "sibling_count": int(len(siblings)),
                "parent_start_idx": int(parent["start_idx"]),
                "parent_end_idx": int(parent["end_idx"]),
            }
        )
    return _summarizeRoccoHierarchy(hierarchy, rowMeta)


def _buildRoccoNestedHierarchy(
    chromosome: str,
    intervals: np.ndarray,
    ends: np.ndarray,
    parentSolution: np.ndarray,
    childSolution: np.ndarray,
    prefix: str,
) -> Dict[str, Any]:
    parentSolution_ = np.asarray(parentSolution, dtype=np.uint8).ravel()
    childSolution_ = np.asarray(childSolution, dtype=np.uint8).ravel()
    if parentSolution_.size != childSolution_.size:
        raise ValueError("Parent and child solutions must match length")
    parentRuns = _selectedRunBounds(parentSolution_ > 0)
    childRuns = _selectedRunBounds(childSolution_ > 0)
    nodes: List[Dict[str, Any]] = []
    childAssigned = [False] * len(childRuns)
    childCursor = 0
    for parentOrdinal, (parentStart, parentEnd) in enumerate(parentRuns, start=1):
        parentPeakId = f"{prefix}_{chromosome}_parent_{parentOrdinal}"
        children: List[Dict[str, Any]] = []
        while childCursor < len(childRuns) and childRuns[childCursor][1] < parentStart:
            childCursor += 1
        localCursor = childCursor
        childOrdinal = 1
        while localCursor < len(childRuns):
            childStart, childEnd = childRuns[localCursor]
            if childStart > parentEnd:
                break
            if parentStart <= childStart and childEnd <= parentEnd:
                childAssigned[localCursor] = True
                children.append(
                    {
                        "peak_id": (
                            f"{prefix}_{chromosome}_parent_{parentOrdinal}"
                            f"_child_{childOrdinal}"
                        ),
                        "parent_peak_id": parentPeakId,
                        "root_peak_id": parentPeakId,
                        "layer": 1,
                        "start_idx": int(childStart),
                        "end_idx": int(childEnd),
                        "start": int(intervals[childStart]),
                        "end": int(ends[childEnd]),
                    }
                )
                childOrdinal += 1
            localCursor += 1
        nodes.append(
            {
                "peak_id": parentPeakId,
                "root_peak_id": parentPeakId,
                "layer": 0,
                "start_idx": int(parentStart),
                "end_idx": int(parentEnd),
                "start": int(intervals[parentStart]),
                "end": int(ends[parentEnd]),
                "children": children,
            }
        )
    if not all(childAssigned):
        raise RuntimeError("Final nested ROCCO child run is outside all support parents")
    return {
        "source": "first_pass_parent_final_child_runs",
        "nodes": nodes,
    }


def _solutionToChromNarrowPeakRows(
    chromosome: str,
    intervals: np.ndarray,
    ends: np.ndarray,
    state: np.ndarray,
    scores: np.ndarray,
    solution: np.ndarray,
    prefix: str,
    nullScale: float,
    uncertainty: np.ndarray | None = None,
    splitSubpeaks: bool = True,
    trimScoreFloor: float | None = 0.0,
    subpeakSelectionPenalty: float | None = None,
    subpeakBoundaryCost: float | None = None,
    minSubpeakBins: int = 1,
    massiveSubpeakCleanup: bool = False,
    massiveSubpeakWidthPolicy: Mapping[str, Any] | None = None,
    massiveSubpeakSplitQuantile: float = _MASSIVE_SUBPEAK_SPLIT_QUANTILE,
    massiveSubpeakSplitZ: float = _MASSIVE_SUBPEAK_SPLIT_Z,
    dropMedianSignalBelowNegativeLocalP: bool = True,
    exportFilterUncertaintyMultiplier: float = (
        _EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
    ),
    scoreFloor: float = 250.0,
    scoreCeil: float = 1000.0,
    nestedHierarchy: Any = None,
    nestedDetails: Mapping[str, Any] | None = None,
    returnExportDetails: bool = False,
) -> (
    Tuple[List[List[str | int | float]], List[Dict[str, Any]]]
    | Tuple[List[List[str | int | float]], List[Dict[str, Any]], Dict[str, Any]]
):
    rowsRaw: List[Dict[str, float | int | str]] = []
    rowsMeta: List[Dict[str, Any]] = []
    intervals = np.asarray(intervals, dtype=np.int64).ravel()
    ends = np.asarray(ends, dtype=np.int64).ravel()
    state_ = np.asarray(state, dtype=np.float64).ravel()
    scores_ = np.asarray(scores, dtype=np.float64).ravel()
    solution_ = np.asarray(solution, dtype=np.uint8).ravel()
    hierarchy = _resolveRoccoExportHierarchy(nestedHierarchy, nestedDetails)
    if (
        intervals.size != state_.size
        or ends.size != state_.size
        or scores_.size != state_.size
        or solution_.size != state_.size
    ):
        raise ValueError(
            "`intervals`, `ends`, `state`, `scores`, and `solution` must match length"
        )
    uncertainty_: np.ndarray | None = None
    if uncertainty is not None:
        uncertainty_ = np.asarray(uncertainty, dtype=np.float64).ravel()
        if uncertainty_.size != state_.size:
            raise ValueError("`uncertainty` must match `state` length")
    exportFilterUncertaintyMultiplier_ = _validateExportFilterUncertaintyMultiplier(
        exportFilterUncertaintyMultiplier
    )
    exportDetails: Dict[str, Any] = {
        "num_candidate_segments": 0,
        "num_segments_dropped_median_signal_local_p": 0,
        "num_segments_dropped_min_peak_bp": 0,
        "min_peak_bp": int(_ROCCO_MIN_PEAK_BP),
        "median_signal_local_p_multiplier": float(exportFilterUncertaintyMultiplier_),
        "median_signal_local_p_filter_active": bool(
            dropMedianSignalBelowNegativeLocalP and uncertainty_ is not None
        ),
        "massive_subpeak_cleanup_active": bool(
            massiveSubpeakCleanup
            and massiveSubpeakWidthPolicy is not None
            and bool(massiveSubpeakWidthPolicy.get("active", False))
        ),
        "massive_subpeak_width_policy": (
            None
            if massiveSubpeakWidthPolicy is None
            else dict(massiveSubpeakWidthPolicy)
        ),
        "num_massive_subpeak_candidates": 0,
        "num_massive_subpeak_splits": 0,
        "num_massive_subpeak_segments_added": 0,
        "num_massive_subpeak_evaluated": 0,
        "num_massive_subpeak_contracts": 0,
        "num_coordinate_gap_splits": 0,
    }
    n = int(solution_.size)
    i = 0
    while i < n:
        if int(solution_[i]) <= 0:
            i += 1
            continue
        startIdx = i
        while (
            i + 1 < n
            and int(solution_[i + 1]) > 0
            and int(ends[i]) == int(intervals[i + 1])
        ):
            i += 1
        endIdx = i
        if (
            endIdx + 1 < n
            and int(solution_[endIdx + 1]) > 0
            and int(ends[endIdx]) != int(intervals[endIdx + 1])
        ):
            exportDetails["num_coordinate_gap_splits"] += 1

        segState = np.asarray(state_[startIdx : endIdx + 1], dtype=np.float64)
        if splitSubpeaks:
            childSegments = _solveParentConditionedSubpeakSegments(
                scores_[startIdx : endIdx + 1],
                segState,
                startIdx=startIdx,
                endIdx=endIdx,
                selectionPenalty=(
                    float(max(float(nullScale), 0.0))
                    if subpeakSelectionPenalty is None
                    else float(subpeakSelectionPenalty)
                ),
                boundaryCost=(
                    float(max(float(nullScale), 0.0))
                    if subpeakBoundaryCost is None
                    else float(subpeakBoundaryCost)
                ),
                minRunBins=int(max(int(minSubpeakBins), 1)),
            )
        else:
            summitLocal = int(np.argmax(segState))
            childSegments = [
                {
                    "start_idx": int(startIdx),
                    "end_idx": int(endIdx),
                    "summit_idx": int(startIdx + summitLocal),
                    "segment_length_bins": int(max(endIdx - startIdx + 1, 0)),
                    "num_subpeaks": 1,
                    "split_from_parent": False,
                    "subpeak_objective": 0.0,
                    "subpeak_boundary_penalty": 0.0,
                    "massive_subpeak_cleanup_candidate": False,
                    "massive_subpeak_cleanup_applied": False,
                    "massive_subpeak_split_gain": None,
                    "massive_subpeak_split_z": None,
                    "massive_subpeak_gap_bins": None,
                    "massive_subpeak_parent_start_idx": int(startIdx),
                    "massive_subpeak_parent_end_idx": int(endIdx),
                }
            ]
        if bool(exportDetails["massive_subpeak_cleanup_active"]):
            forcedSegments: List[Dict[str, Any]] = []
            for child in childSegments:
                forced, forceCounts = _forceMassiveSubpeakSegments(
                    child,
                    scores_,
                    state_,
                    intervals,
                    ends,
                    massiveSubpeakWidthPolicy,
                    boundaryCost=(
                        float(max(float(nullScale), 0.0))
                        if subpeakBoundaryCost is None
                        else float(subpeakBoundaryCost)
                    ),
                    minRunBins=int(max(int(minSubpeakBins), 1)),
                    splitQuantile=float(massiveSubpeakSplitQuantile),
                    minSplitZ=float(massiveSubpeakSplitZ),
                )
                exportDetails["num_massive_subpeak_candidates"] += int(
                    forceCounts["candidates"]
                )
                exportDetails["num_massive_subpeak_splits"] += int(
                    forceCounts["splits"]
                )
                exportDetails["num_massive_subpeak_segments_added"] += int(
                    forceCounts["segments_added"]
                )
                exportDetails["num_massive_subpeak_evaluated"] += int(
                    forceCounts["evaluated"]
                )
                exportDetails["num_massive_subpeak_contracts"] += int(
                    forceCounts["contracts"]
                )
                forcedSegments.extend(forced)
            childSegments = forcedSegments
        for child in childSegments:
            untrimmedStartIdx = int(child["start_idx"])
            untrimmedEndIdx = int(child["end_idx"])
            summitIdx = int(child["summit_idx"])
            childStartIdx, childEndIdx, wasTrimmed = _trimChildSegmentAroundSummit(
                untrimmedStartIdx,
                untrimmedEndIdx,
                summitIdx,
                scores_,
                trimScoreFloor,
            )
            childState = np.asarray(
                state_[childStartIdx : childEndIdx + 1], dtype=np.float64
            )
            childScores = np.asarray(
                scores_[childStartIdx : childEndIdx + 1], dtype=np.float64
            )
            exportDetails["num_candidate_segments"] += 1
            medianState = float(np.median(childState))
            localMedianP = None
            medianSignalThreshold = None
            if dropMedianSignalBelowNegativeLocalP and uncertainty_ is not None:
                localP = np.asarray(
                    uncertainty_[childStartIdx : childEndIdx + 1],
                    dtype=np.float64,
                )
                localP = localP[np.isfinite(localP)]
                if localP.size > 0:
                    localMedianP = float(np.median(localP))
                    medianSignalThreshold = float(
                        -exportFilterUncertaintyMultiplier_ * localMedianP
                    )
                    if medianState < medianSignalThreshold:
                        exportDetails["num_segments_dropped_median_signal_local_p"] += 1
                        continue
            summitAbs = int(
                intervals[summitIdx]
                + max(1, int((ends[summitIdx] - intervals[summitIdx]) // 2))
            )
            chromStart = int(intervals[childStartIdx])
            chromEnd = int(ends[childEndIdx])
            if chromEnd - chromStart < int(_ROCCO_MIN_PEAK_BP):
                exportDetails["num_segments_dropped_min_peak_bp"] += 1
                continue
            widthP, widthQ = _massiveSubpeakWidthPValue(
                float(chromEnd - chromStart),
                massiveSubpeakWidthPolicy,
            )
            peakOffset = int(max(0, summitAbs - chromStart))
            peakName = f"{prefix}_{chromosome}_{len(rowsRaw)+1}"
            rowsRaw.append(
                {
                    "chromosome": str(chromosome),
                    "start": chromStart,
                    "end": chromEnd,
                    "signal": float(np.max(childState)),
                    "raw_score": float(np.max(childScores)),
                    "peak": peakOffset,
                    "name": str(peakName),
                }
            )
            rowsMeta.append(
                {
                    "name": str(peakName),
                    "chromosome": str(chromosome),
                    "start": int(chromStart),
                    "end": int(chromEnd),
                    "summit": int(summitAbs),
                    "start_idx": int(childStartIdx),
                    "end_idx": int(childEndIdx),
                    "summit_idx": int(summitIdx),
                    "untrimmed_start_idx": int(untrimmedStartIdx),
                    "untrimmed_end_idx": int(untrimmedEndIdx),
                    "segment_length_bins": int(child["segment_length_bins"]),
                    "median_state": float(medianState),
                    "local_median_p": (
                        None if localMedianP is None else float(localMedianP)
                    ),
                    "median_signal_threshold": (
                        None
                        if medianSignalThreshold is None
                        else float(medianSignalThreshold)
                    ),
                    "max_state": float(np.max(childState)),
                    "max_score": float(np.max(childScores)),
                    "num_subpeaks": int(child["num_subpeaks"]),
                    "split_from_parent": bool(child["split_from_parent"]),
                    "subpeak_objective": float(child["subpeak_objective"]),
                    "subpeak_boundary_penalty": float(
                        child["subpeak_boundary_penalty"]
                    ),
                    "massive_subpeak_cleanup_candidate": bool(
                        child.get("massive_subpeak_cleanup_candidate", False)
                    ),
                    "massive_subpeak_cleanup_applied": bool(
                        child.get("massive_subpeak_cleanup_applied", False)
                    ),
                    "massive_subpeak_cleanup_mode": child.get(
                        "massive_subpeak_cleanup_mode"
                    ),
                    "massive_subpeak_width_p": (
                        None if widthP is None else float(widthP)
                    ),
                    "massive_subpeak_width_q": (
                        None if widthQ is None else float(widthQ)
                    ),
                    "massive_subpeak_width_threshold_bp": (
                        None
                        if massiveSubpeakWidthPolicy is None
                        or massiveSubpeakWidthPolicy.get("width_threshold_bp") is None
                        else int(massiveSubpeakWidthPolicy["width_threshold_bp"])
                    ),
                    "massive_subpeak_split_gain": (
                        None
                        if child.get("massive_subpeak_split_gain") is None
                        else float(child["massive_subpeak_split_gain"])
                    ),
                    "massive_subpeak_split_z": (
                        None
                        if child.get("massive_subpeak_split_z") is None
                        else float(child["massive_subpeak_split_z"])
                    ),
                    "massive_subpeak_gap_bins": (
                        None
                        if child.get("massive_subpeak_gap_bins") is None
                        else int(child["massive_subpeak_gap_bins"])
                    ),
                    "massive_subpeak_core_width_bp": (
                        None
                        if child.get("massive_subpeak_core_width_bp") is None
                        else int(child["massive_subpeak_core_width_bp"])
                    ),
                    "massive_subpeak_core_score_floor": (
                        None
                        if child.get("massive_subpeak_core_score_floor") is None
                        else float(child["massive_subpeak_core_score_floor"])
                    ),
                    "trimmed_from_parent": bool(wasTrimmed),
                    "trim_score_floor": (
                        None if trimScoreFloor is None else float(trimScoreFloor)
                    ),
                    "untrimmed_start": int(intervals[untrimmedStartIdx]),
                    "untrimmed_end": int(ends[untrimmedEndIdx]),
                }
            )
        i += 1

    if len(rowsRaw) == 0:
        if returnExportDetails:
            exportDetails["num_segments_kept"] = 0
            hierarchySummary = _annotateRoccoPeakHierarchy(rowsMeta, hierarchy)
            if hierarchySummary:
                exportDetails["nested_hierarchy_summary"] = hierarchySummary
            return [], [], exportDetails
        return [], []

    rawScores = np.asarray([row["raw_score"] for row in rowsRaw], dtype=np.float64)
    minScore = float(np.min(rawScores))
    maxScore = float(np.max(rawScores))
    span = max(maxScore - minScore, 1.0e-12)
    outRows: List[List[str | int | float]] = []
    for row in rowsRaw:
        scaled = scoreFloor + (scoreCeil - scoreFloor) * (
            (float(row["raw_score"]) - minScore) / span
        )
        outRows.append(
            [
                str(row["chromosome"]),
                int(row["start"]),
                int(row["end"]),
                str(row["name"]),
                int(round(scaled)),
                ".",
                float(row["signal"]),
                -1,
                -1,
                int(row["peak"]),
            ]
        )
    exportDetails["num_segments_kept"] = int(len(outRows))
    hierarchySummary = _annotateRoccoPeakHierarchy(rowsMeta, hierarchy)
    if hierarchySummary:
        exportDetails["nested_hierarchy_summary"] = hierarchySummary
    if returnExportDetails:
        return outRows, rowsMeta, exportDetails
    return outRows, rowsMeta


def _blocksForBroadParent(
    parentStartIdx: int,
    parentEndIdx: int,
    childRuns: Sequence[Tuple[int, int]],
    intervals: np.ndarray,
    ends: np.ndarray,
) -> List[Tuple[int, int]]:
    parentStartBP = int(intervals[parentStartIdx])
    parentEndBP = int(ends[parentEndIdx])
    blocks: List[Tuple[int, int]] = []
    for childStart, childEnd in childRuns:
        if int(childEnd) < int(parentStartIdx) or int(childStart) > int(parentEndIdx):
            continue
        blockStartIdx = int(max(int(childStart), int(parentStartIdx)))
        blockEndIdx = int(min(int(childEnd), int(parentEndIdx)))
        blockStartBP = int(max(int(intervals[blockStartIdx]), parentStartBP))
        blockEndBP = int(min(int(ends[blockEndIdx]), parentEndBP))
        if blockEndBP > blockStartBP:
            blocks.append((blockStartBP, blockEndBP))
    if not blocks:
        blocks = [(parentStartBP, parentEndBP)]
    blocks.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in blocks:
        if not merged or int(start) > int(merged[-1][1]):
            merged.append((int(start), int(end)))
        else:
            prevStart, prevEnd = merged[-1]
            merged[-1] = (int(prevStart), max(int(prevEnd), int(end)))
    if int(merged[0][0]) > parentStartBP:
        merged.insert(0, (parentStartBP, parentStartBP + 1))
    if int(merged[-1][1]) < parentEndBP:
        merged.append((parentEndBP - 1, parentEndBP))
    return merged


def _solutionToChromBroadPeakRows(
    chromosome: str,
    intervals: np.ndarray,
    ends: np.ndarray,
    state: np.ndarray,
    scores: np.ndarray,
    parentSolution: np.ndarray,
    childSolution: np.ndarray,
    prefix: str,
    uncertainty: np.ndarray | None = None,
    exportFilterUncertaintyMultiplier: float = (
        _EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
    ),
    minPeakBP: int = _MATCHING_DEFAULT_BROAD_MIN_PEAK_BP,
    scoreFloor: float = 250.0,
    scoreCeil: float = 1000.0,
    returnExportDetails: bool = False,
) -> (
    Tuple[List[List[str | int | float]], List[List[str | int | float]], List[Dict[str, Any]]]
    | Tuple[
        List[List[str | int | float]],
        List[List[str | int | float]],
        List[Dict[str, Any]],
        Dict[str, Any],
    ]
):
    intervals_ = np.asarray(intervals, dtype=np.int64).ravel()
    ends_ = np.asarray(ends, dtype=np.int64).ravel()
    state_ = np.asarray(state, dtype=np.float64).ravel()
    scores_ = np.asarray(scores, dtype=np.float64).ravel()
    parentSolution_ = np.asarray(parentSolution, dtype=np.uint8).ravel()
    childSolution_ = np.asarray(childSolution, dtype=np.uint8).ravel()
    if (
        intervals_.size != state_.size
        or ends_.size != state_.size
        or scores_.size != state_.size
        or parentSolution_.size != state_.size
        or childSolution_.size != state_.size
    ):
        raise ValueError(
            "`intervals`, `ends`, `state`, `scores`, and solutions must match length"
        )
    uncertainty_: np.ndarray | None = None
    if uncertainty is not None:
        uncertainty_ = np.asarray(uncertainty, dtype=np.float64).ravel()
        if uncertainty_.size != state_.size:
            raise ValueError("`uncertainty` must match `state` length")
    exportFilterUncertaintyMultiplier_ = _validateExportFilterUncertaintyMultiplier(
        exportFilterUncertaintyMultiplier
    )
    minPeakBP_ = int(max(int(minPeakBP), 1))
    exportDetails: Dict[str, Any] = {
        "num_candidate_segments": 0,
        "num_segments_dropped_median_signal_local_p": 0,
        "num_segments_dropped_min_peak_bp": 0,
        "min_peak_bp": int(minPeakBP_),
        "median_signal_local_p_multiplier": float(exportFilterUncertaintyMultiplier_),
        "median_signal_local_p_filter_active": bool(uncertainty_ is not None),
        "num_broad_parent_segments": 0,
        "num_gapped_peak_blocks": 0,
    }
    parentRuns = _selectedCoordinateRunBounds(parentSolution_, intervals_, ends_)
    childRuns = _selectedCoordinateRunBounds(childSolution_, intervals_, ends_)
    rowsRaw: List[Dict[str, Any]] = []
    rowsMeta: List[Dict[str, Any]] = []
    for parentStartIdx, parentEndIdx in parentRuns:
        parentState = np.asarray(
            state_[parentStartIdx : parentEndIdx + 1], dtype=np.float64
        )
        parentScores = np.asarray(
            scores_[parentStartIdx : parentEndIdx + 1], dtype=np.float64
        )
        exportDetails["num_candidate_segments"] += 1
        medianState = float(np.median(parentState))
        localMedianP = None
        medianSignalThreshold = None
        if uncertainty_ is not None:
            localP = np.asarray(
                uncertainty_[parentStartIdx : parentEndIdx + 1],
                dtype=np.float64,
            )
            localP = localP[np.isfinite(localP)]
            if localP.size > 0:
                localMedianP = float(np.median(localP))
                medianSignalThreshold = float(
                    -exportFilterUncertaintyMultiplier_ * localMedianP
                )
                if medianState < medianSignalThreshold:
                    exportDetails["num_segments_dropped_median_signal_local_p"] += 1
                    continue
        chromStart = int(intervals_[parentStartIdx])
        chromEnd = int(ends_[parentEndIdx])
        if chromEnd - chromStart < minPeakBP_:
            exportDetails["num_segments_dropped_min_peak_bp"] += 1
            continue
        summitLocal = int(np.argmax(parentState))
        summitIdx = int(parentStartIdx + summitLocal)
        summitAbs = int(
            intervals_[summitIdx]
            + max(1, int((ends_[summitIdx] - intervals_[summitIdx]) // 2))
        )
        blocks = _blocksForBroadParent(
            int(parentStartIdx),
            int(parentEndIdx),
            childRuns,
            intervals_,
            ends_,
        )
        blockSizes = [int(end - start) for start, end in blocks]
        blockStarts = [int(start - chromStart) for start, _end in blocks]
        if blockStarts[0] != 0:
            raise RuntimeError("gappedPeak first block must start at parent start")
        if blockStarts[-1] + blockSizes[-1] != chromEnd - chromStart:
            raise RuntimeError("gappedPeak final block must end at parent end")
        peakName = f"{prefix}_{chromosome}_{len(rowsRaw) + 1}"
        rowsRaw.append(
            {
                "chromosome": str(chromosome),
                "start": int(chromStart),
                "end": int(chromEnd),
                "name": str(peakName),
                "signal": float(np.mean(parentState)),
                "raw_score": float(np.mean(parentScores)),
                "block_sizes": blockSizes,
                "block_starts": blockStarts,
            }
        )
        rowsMeta.append(
            {
                "name": str(peakName),
                "chromosome": str(chromosome),
                "start": int(chromStart),
                "end": int(chromEnd),
                "summit": int(summitAbs),
                "start_idx": int(parentStartIdx),
                "end_idx": int(parentEndIdx),
                "summit_idx": int(summitIdx),
                "median_state": float(medianState),
                "mean_state": float(np.mean(parentState)),
                "max_state": float(np.max(parentState)),
                "mean_score": float(np.mean(parentScores)),
                "max_score": float(np.max(parentScores)),
                "local_median_p": (
                    None if localMedianP is None else float(localMedianP)
                ),
                "median_signal_threshold": (
                    None
                    if medianSignalThreshold is None
                    else float(medianSignalThreshold)
                ),
                "block_count": int(len(blocks)),
                "block_sizes": [int(value) for value in blockSizes],
                "block_starts": [int(value) for value in blockStarts],
            }
        )
    if len(rowsRaw) == 0:
        exportDetails["num_segments_kept"] = 0
        if returnExportDetails:
            return [], [], [], exportDetails
        return [], [], []
    rawScores = np.asarray([row["raw_score"] for row in rowsRaw], dtype=np.float64)
    minScore = float(np.min(rawScores))
    maxScore = float(np.max(rawScores))
    span = max(maxScore - minScore, 1.0e-12)
    broadRows: List[List[str | int | float]] = []
    gappedRows: List[List[str | int | float]] = []
    for row in rowsRaw:
        scaled = scoreFloor + (scoreCeil - scoreFloor) * (
            (float(row["raw_score"]) - minScore) / span
        )
        score = int(round(scaled))
        blockSizesText = ",".join(str(int(value)) for value in row["block_sizes"])
        blockStartsText = ",".join(str(int(value)) for value in row["block_starts"])
        broadRows.append(
            [
                str(row["chromosome"]),
                int(row["start"]),
                int(row["end"]),
                str(row["name"]),
                score,
                ".",
                float(row["signal"]),
                -1,
                -1,
            ]
        )
        gappedRows.append(
            [
                str(row["chromosome"]),
                int(row["start"]),
                int(row["end"]),
                str(row["name"]),
                score,
                ".",
                0,
                0,
                0,
                int(len(row["block_sizes"])),
                blockSizesText,
                blockStartsText,
                float(row["signal"]),
                -1,
                -1,
            ]
        )
    exportDetails["num_segments_kept"] = int(len(broadRows))
    exportDetails["num_broad_parent_segments"] = int(len(broadRows))
    exportDetails["num_gapped_peak_blocks"] = int(
        sum(int(row[9]) for row in gappedRows)
    )
    if returnExportDetails:
        return broadRows, gappedRows, rowsMeta, exportDetails
    return broadRows, gappedRows, rowsMeta


def _pairedGappedRowsForPeakMeta(
    gappedRows: Sequence[List[str | int | float]],
    peakMeta: Sequence[Mapping[str, Any]],
) -> List[List[str | int | float]]:
    byName = {str(row[3]): list(row) for row in gappedRows}
    paired: List[List[str | int | float]] = []
    for meta in peakMeta:
        name = str(meta["name"])
        if name not in byName:
            raise RuntimeError("Missing gappedPeak row for broad peak")
        paired.append(byName[name])
    return paired


def _negativeLog10OrMissing(value: Any) -> float | int:
    if value is None:
        return -1
    numeric = float(value)
    if numeric <= 0.0 or numeric > 1.0:
        raise ValueError("empirical p/q values must lie in (0, 1]")
    return float(-math.log10(numeric))


def _fillBroadRowsDWBValues(
    broadRows: List[List[str | int | float]],
    gappedRows: List[List[str | int | float]],
    peakMeta: Sequence[Mapping[str, Any]],
) -> None:
    metaByName = {str(meta["name"]): meta for meta in peakMeta}
    for row in broadRows:
        meta = metaByName[str(row[3])]
        row[7] = _negativeLog10OrMissing(meta.get("dwb_peak_empirical_p"))
        row[8] = _negativeLog10OrMissing(meta.get("dwb_peak_empirical_q"))
    for row in gappedRows:
        meta = metaByName[str(row[3])]
        row[13] = _negativeLog10OrMissing(meta.get("dwb_peak_empirical_p"))
        row[14] = _negativeLog10OrMissing(meta.get("dwb_peak_empirical_q"))


def _fileInventoryEntry(path: str | None, kind: str) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"kind": str(kind), "path": path, "exists": False, "bytes": None}
    if path is None:
        return entry
    pathObj = Path(path)
    entry["exists"] = bool(pathObj.exists())
    if pathObj.exists():
        try:
            entry["bytes"] = int(pathObj.stat().st_size)
        except OSError:
            entry["bytes"] = None
    return entry


def _summarizePeakWidthsFromRows(
    rows: Iterable[List[str | int | float]],
) -> Dict[str, Any]:
    widths = [int(row[2]) - int(row[1]) for row in rows]
    if not widths:
        return {
            "exported_peak_count": 0,
            "total_peak_bp": 0,
            "min_width_bp": None,
            "median_width_bp": None,
            "max_width_bp": None,
        }
    widthsArr = np.asarray(widths, dtype=np.int64)
    return {
        "exported_peak_count": int(widthsArr.size),
        "total_peak_bp": int(np.sum(widthsArr)),
        "min_width_bp": int(np.min(widthsArr)),
        "median_width_bp": float(np.median(widthsArr)),
        "max_width_bp": int(np.max(widthsArr)),
    }


def _buildRoccoSummary(
    *,
    outPath: str,
    metaPath: str | None,
    nestedRoccoSubproblemDetailsPath: str | None,
    gappedPath: str | None = None,
    rows: List[List[str | int | float]],
    meta: Mapping[str, Any],
) -> Dict[str, Any]:
    widthSummary = _summarizePeakWidthsFromRows(rows)
    chromosomesMeta = meta.get("chromosomes", {})
    perChrom: Dict[str, Any] = {}
    nestedStops: Dict[str, Any] = {}
    if isinstance(chromosomesMeta, Mapping):
        for chromosome, chromMetaAny in chromosomesMeta.items():
            if not isinstance(chromMetaAny, Mapping):
                continue
            peakDetails = chromMetaAny.get("peak_details", [])
            peakCount = int(chromMetaAny.get("num_segments", 0))
            totalBP = 0
            if isinstance(peakDetails, list):
                for peak in peakDetails:
                    if not isinstance(peak, Mapping):
                        continue
                    totalBP += max(
                        int(peak.get("end", 0)) - int(peak.get("start", 0)),
                        0,
                    )
            nestedDetails = chromMetaAny.get("nested_rocco_details", {})
            nestedSummary: Dict[str, Any] = {}
            if isinstance(nestedDetails, Mapping):
                history = nestedDetails.get("history", [])
                lastHistory = history[-1] if isinstance(history, list) and history else {}
                nestedSummary = {
                    "enabled": bool(nestedDetails.get("enabled", False)),
                    "requested_iters": int(nestedDetails.get("requested_iters", 0)),
                    "completed_iters": int(nestedDetails.get("completed_iters", 0)),
                    "stop_reason": str(nestedDetails.get("stop_reason", "")),
                    "budget_scale": float(nestedDetails.get("budget_scale", 0.0)),
                    "last_jaccard": (
                        None
                        if not isinstance(lastHistory, Mapping)
                        or lastHistory.get("jaccard") is None
                        else float(lastHistory["jaccard"])
                    ),
                }
                nestedStops[str(chromosome)] = {
                    "completed_iters": nestedSummary["completed_iters"],
                    "stop_reason": nestedSummary["stop_reason"],
                    "last_jaccard": nestedSummary["last_jaccard"],
                }
            hierarchySummary = chromMetaAny.get("nested_hierarchy_summary", {})
            if not isinstance(hierarchySummary, Mapping):
                hierarchySummary = {}
            perChrom[str(chromosome)] = {
                "exported_peak_count": peakCount,
                "total_peak_bp": int(totalBP),
                "parent_peak_count": int(chromMetaAny.get("num_parent_segments", 0)),
                "candidate_peak_count": int(
                    chromMetaAny.get("num_candidate_segments", 0)
                ),
                "dropped_median_signal_local_p": int(
                    chromMetaAny.get("num_segments_dropped_median_signal_local_p", 0)
                ),
                "dropped_min_peak_bp": int(
                    chromMetaAny.get("num_segments_dropped_min_peak_bp", 0)
                ),
                "dropped_min_peak_score": int(
                    chromMetaAny.get("num_segments_dropped_min_peak_score", 0)
                ),
                "dwb_peak_scoring": dict(
                    chromMetaAny.get("export_details", {}).get(
                        "dwb_peak_scoring",
                        {},
                    )
                    if isinstance(chromMetaAny.get("export_details", {}), Mapping)
                    else {}
                ),
                "nested_rocco": nestedSummary,
                "nested_hierarchy": dict(hierarchySummary),
            }
    blacklist = meta.get("blacklist_filter", {})
    if not isinstance(blacklist, Mapping):
        blacklist = {}
    settings = meta.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}
    outputFormat = str(settings.get("peak_output_format", "narrowPeak"))
    inventory = [
        _fileInventoryEntry(outPath, outputFormat),
        _fileInventoryEntry(metaPath, "metadata_json"),
    ]
    if gappedPath is not None:
        inventory.append(_fileInventoryEntry(gappedPath, "gappedPeak"))
    if nestedRoccoSubproblemDetailsPath is not None:
        inventory.append(
            _fileInventoryEntry(
                nestedRoccoSubproblemDetailsPath,
                "nested_rocco_subproblems_jsonl",
            )
        )
    summary: Dict[str, Any] = {
        "peak_path": str(outPath),
        "peak_output_format": outputFormat,
        "narrowPeak_path": str(outPath) if outputFormat == "narrowPeak" else None,
        "broadPeak_path": str(outPath) if outputFormat == "broadPeak" else None,
        "gappedPeak_path": gappedPath,
        "peak_paths": (
            [str(outPath)]
            if gappedPath is None
            else [str(outPath), str(gappedPath)]
        ),
        "metadata_json_path": None if metaPath is None else str(metaPath),
        "nested_jsonl_path": nestedRoccoSubproblemDetailsPath,
        **widthSummary,
        "blacklist": {
            "blacklist_bed": blacklist.get("blacklist_bed"),
            "policy": blacklist.get("policy"),
            "dropped": int(blacklist.get("dropped", 0)),
            "kept": int(blacklist.get("kept", widthSummary["exported_peak_count"])),
        },
        "per_chrom": perChrom,
        "nested_rocco": {
            "requested_iters": int(settings.get("nested_rocco_iters", 0)),
            "budget_scale": float(settings.get("nested_rocco_budget_scale", 0.0)),
            "budget_policy": str(
                settings.get("nested_rocco_budget_policy", "")
            ),
            "diagnostics": bool(settings.get("nested_rocco_diagnostics", False)),
            "subproblem_details": nestedRoccoSubproblemDetailsPath,
            "stops": nestedStops,
        },
        "settings": {
            "budget_method": settings.get("budget_method"),
            "null_calibration_method": settings.get("null_calibration_method"),
            "num_bootstrap": int(settings.get("num_bootstrap", 0)),
            "threshold_z": float(settings.get("threshold_z", 0.0)),
            "rand_seed": int(settings.get("rand_seed", 0)),
            "min_peak_score": settings.get("min_peak_score"),
            "peak_output_format": outputFormat,
        },
        "files": inventory,
    }
    return summary


def _logRoccoSummary(summary: Mapping[str, Any]) -> None:
    width = summary.get("median_width_bp")
    widthText = "NA" if width is None else f"{float(width):.1f}"
    logger.info(
        "rocco.summary peaks=%d total_bp=%d width_bp[min/median/max]=%s/%s/%s blacklist_dropped=%d blacklist_kept=%d peak=%s format=%s metadata=%s nested_jsonl=%s",
        int(summary.get("exported_peak_count", 0)),
        int(summary.get("total_peak_bp", 0)),
        "NA" if summary.get("min_width_bp") is None else str(summary["min_width_bp"]),
        widthText,
        "NA" if summary.get("max_width_bp") is None else str(summary["max_width_bp"]),
        int(dict(summary.get("blacklist", {})).get("dropped", 0)),
        int(dict(summary.get("blacklist", {})).get("kept", 0)),
        summary.get("peak_path"),
        summary.get("peak_output_format"),
        summary.get("metadata_json_path"),
        summary.get("nested_jsonl_path"),
    )


def _logOutputInventory(summary: Mapping[str, Any]) -> None:
    files = summary.get("files", [])
    if not isinstance(files, list):
        return
    parts = []
    for entryAny in files:
        if not isinstance(entryAny, Mapping):
            continue
        exists = "yes" if bool(entryAny.get("exists", False)) else "no"
        byteText = "NA" if entryAny.get("bytes") is None else str(entryAny["bytes"])
        parts.append(
            f"{entryAny.get('kind')}:{entryAny.get('path')} exists={exists} bytes={byteText}"
        )
    if parts:
        logger.info("output.inventory %s", " | ".join(parts))


def _compactRoccoMetadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = dict(meta)
    compact["metadata_detail"] = "compact"
    settings = dict(compact.get("settings", {}))
    settings["metadata_detail"] = "compact"
    settings["metadata_omits"] = [
        "candidate_details",
        "peak_details",
        "nested_hierarchy",
        "nested_rocco_details.hierarchy",
    ]
    compact["settings"] = settings
    chromCompact: Dict[str, Any] = {}
    chromosomes = meta.get("chromosomes", {})
    if isinstance(chromosomes, Mapping):
        for chromosome, chromMetaAny in chromosomes.items():
            if not isinstance(chromMetaAny, Mapping):
                continue
            chromMeta = dict(chromMetaAny)
            candidateDetails = chromMeta.pop("candidate_details", [])
            peakDetails = chromMeta.pop("peak_details", [])
            nestedHierarchy = chromMeta.pop("nested_hierarchy", {})
            nestedRoccoDetails = chromMeta.get("nested_rocco_details")
            if isinstance(nestedRoccoDetails, Mapping):
                nestedRoccoDetails_ = dict(nestedRoccoDetails)
                nestedRoccoHierarchy = nestedRoccoDetails_.pop("hierarchy", [])
                if isinstance(nestedRoccoHierarchy, list):
                    nestedRoccoDetails_["hierarchy_nodes_omitted"] = int(
                        len(nestedRoccoHierarchy)
                    )
                chromMeta["nested_rocco_details"] = nestedRoccoDetails_
            exportDetails = dict(chromMeta.get("export_details", {}))
            exportCandidateDetails = exportDetails.pop("candidate_details", [])
            candidateCount = max(
                len(candidateDetails) if isinstance(candidateDetails, list) else 0,
                len(exportCandidateDetails)
                if isinstance(exportCandidateDetails, list)
                else 0,
            )
            peakCount = len(peakDetails) if isinstance(peakDetails, list) else 0
            exportDetails["candidate_details_omitted"] = int(candidateCount)
            hierarchySummaryAny = chromMeta.get(
                "nested_hierarchy_summary",
                exportDetails.get("nested_hierarchy_summary", {}),
            )
            hierarchySummary = (
                dict(hierarchySummaryAny)
                if isinstance(hierarchySummaryAny, Mapping)
                else {}
            )
            if hierarchySummary:
                exportDetails["nested_hierarchy_summary"] = hierarchySummary
            chromMeta["export_details"] = exportDetails
            chromMeta["candidate_details_omitted"] = int(candidateCount)
            chromMeta["peak_details_omitted"] = int(peakCount)
            if hierarchySummary:
                chromMeta["nested_hierarchy_summary"] = hierarchySummary
            if isinstance(nestedHierarchy, Mapping):
                nestedNodes = nestedHierarchy.get("nodes", [])
                chromMeta["nested_hierarchy_nodes_omitted"] = int(
                    hierarchySummary.get(
                        "node_count",
                        len(nestedNodes) if isinstance(nestedNodes, list) else 0,
                    )
                )
            chromCompact[str(chromosome)] = chromMeta
    compact["chromosomes"] = chromCompact
    return compact


def _boundedJsonValue(value: Any, *, depth: int = 2) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return {"omitted_type": "ndarray", "shape": [int(x) for x in value.shape]}
    if isinstance(value, Mapping):
        if depth <= 0:
            return {"omitted_type": "mapping", "keys": int(len(value))}
        return {
            str(key): _boundedJsonValue(child, depth=depth - 1)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return {"omitted_type": "sequence", "length": int(len(value))}
    return str(value)


def _boundedRoccoMetadata(
    meta: Mapping[str, Any],
    *,
    requestedDetail: str,
    maxBytes: int,
    attemptedBytes: int,
) -> Dict[str, Any]:
    settings = dict(meta.get("settings", {}))
    settings["metadata_detail"] = "bounded"
    settings["metadata_requested_detail"] = str(requestedDetail)
    settings["metadata_byte_cap"] = int(maxBytes)
    settings["metadata_attempted_bytes"] = int(attemptedBytes)
    settings["metadata_omits"] = [
        "candidate_details",
        "peak_details",
        "nested_hierarchy",
        "nested_rocco_details",
        "nested_details",
        "large_sequence_values",
    ]
    bounded: Dict[str, Any] = {
        "metadata_detail": "bounded",
        "settings": _boundedJsonValue(settings, depth=2),
        "pooled_null_floor": _boundedJsonValue(meta.get("pooled_null_floor"), depth=2),
        "budget_shrinkage": _boundedJsonValue(meta.get("budget_shrinkage"), depth=2),
        "blacklist_filter": _boundedJsonValue(
            meta.get("blacklist_filter", {}),
            depth=2,
        ),
        "massive_subpeak_width_policy": _boundedJsonValue(
            meta.get("massive_subpeak_width_policy", {}),
            depth=2,
        ),
        "chromosomes": {},
    }
    chromosomes = meta.get("chromosomes", {})
    if isinstance(chromosomes, Mapping):
        for chromosome, chromMetaAny in chromosomes.items():
            if not isinstance(chromMetaAny, Mapping):
                continue
            peakDetails = chromMetaAny.get("peak_details", [])
            exportDetails = chromMetaAny.get("export_details", {})
            exportCandidateDetails = (
                exportDetails.get("candidate_details", [])
                if isinstance(exportDetails, Mapping)
                else []
            )
            chromBounded: Dict[str, Any] = {}
            for key in (
                "n_loci",
                "interval_bp",
                "budget",
                "objective",
                "first_pass_objective",
                "num_parent_segments",
                "num_segments",
                "num_candidate_segments",
                "num_segments_dropped_median_signal_local_p",
                "num_segments_dropped_min_peak_bp",
                "num_segments_dropped_min_peak_score",
                "export_trim_score_floor",
                "state_diagnostics",
                "budget_details",
                "gamma_details",
                "solve_details",
                "nested_hierarchy_summary",
                "null_replay_false_segment_diagnostics",
            ):
                if key in chromMetaAny:
                    chromBounded[key] = _boundedJsonValue(chromMetaAny[key], depth=2)
            chromBounded["candidate_details_omitted"] = int(
                len(exportCandidateDetails)
                if isinstance(exportCandidateDetails, list)
                else chromMetaAny.get("candidate_details_omitted", 0)
            )
            chromBounded["peak_details_omitted"] = int(
                len(peakDetails)
                if isinstance(peakDetails, list)
                else chromMetaAny.get("peak_details_omitted", 0)
            )
            bounded["chromosomes"][str(chromosome)] = chromBounded
    return bounded


def _writeRoccoMetadata(
    metaPath: str,
    meta: Mapping[str, Any],
    *,
    metadataDetail: str,
    maxNonTrackFileBytes: int,
) -> None:
    path = Path(metaPath)

    def writePayload(payload: Mapping[str, Any]) -> int:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return int(path.stat().st_size)

    maxBytes = int(max(maxNonTrackFileBytes, 0))
    payload = meta if metadataDetail == "full" else _compactRoccoMetadata(meta)
    attemptedBytes = writePayload(payload)
    if maxBytes <= 0 or attemptedBytes <= maxBytes:
        return
    if metadataDetail == "full":
        compact = _compactRoccoMetadata(meta)
        compactBytes = writePayload(compact)
        if compactBytes <= maxBytes:
            logger.warning(
                "ROCCO metadata detail downgraded from full to compact because %s "
                "would exceed the non-track file cap: full_bytes=%d cap_bytes=%d",
                metaPath,
                attemptedBytes,
                maxBytes,
            )
            return
        attemptedBytes = compactBytes
    bounded = _boundedRoccoMetadata(
        meta,
        requestedDetail=metadataDetail,
        maxBytes=maxBytes,
        attemptedBytes=attemptedBytes,
    )
    boundedBytes = writePayload(bounded)
    logger.warning(
        "ROCCO metadata bounded to keep non-track file under policy cap: "
        "path=%s requested_detail=%s attempted_bytes=%d bounded_bytes=%d cap_bytes=%d",
        metaPath,
        metadataDetail,
        attemptedBytes,
        boundedBytes,
        maxBytes,
    )


def solveRocco(
    stateBedGraphFile: str,
    uncertaintyBedGraphFile: str | None = None,
    chromosomes: Iterable[str] | None = None,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    dependenceSpan: int | None = None,
    gamma: float | None = 0.5,
    selectionPenalty: float | None = None,
    gammaScale: float = 0.5,
    nestedRoccoIters: int = _NESTED_ROCCO_ITERS_DEFAULT,
    nestedRoccoBudgetScale: float = _NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    massiveSubpeakCleanup: bool = _MASSIVE_SUBPEAK_CLEANUP_DEFAULT,
    exportFilterUncertaintyMultiplier: float = (
        _EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
    ),
    minPeakScore: float | None = _MATCHING_DEFAULT_MIN_PEAK_SCORE,
    peakMode: str = _MATCHING_DEFAULT_PEAK_MODE,
    broadWeakThresholdZ: float = _MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z,
    broadMaxGapBP: int | None = _MATCHING_DEFAULT_BROAD_MAX_GAP_BP,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    randSeed: int = 42,
    outPath: str | None = None,
    metaPath: str | None = None,
    verbose: bool = False,
    metadataDetail: str = _MATCHING_DEFAULT_METADATA_DETAIL,
    maxNonTrackFileBytes: int = _OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES,
    stateDiagnosticsByChromosome: Mapping[str, Any] | None = None,
    blacklistBedFile: str | None = None,
    writeMetadata: bool = True,
    returnSummary: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    r"""Run Consenrich+ROCCO peak caller directly on bedGraphs."""
    exportFilterUncertaintyMultiplier_ = _validateExportFilterUncertaintyMultiplier(
        exportFilterUncertaintyMultiplier
    )
    minPeakScore_ = _validateMinPeakScore(minPeakScore)
    peakMode_ = _normalizeRoccoPeakMode(peakMode)
    broadWeakThresholdZ_ = _validateBroadWeakThresholdZ(broadWeakThresholdZ)
    broadMaxGapBP_ = _validateBroadMaxGapBP(broadMaxGapBP)
    thresholdZ_ = float(thresholdZ)
    if peakMode_ == "broad" and broadWeakThresholdZ_ > thresholdZ_:
        raise ValueError("`broadWeakThresholdZ` cannot exceed `thresholdZ`")
    outputFormat = "broadPeak" if peakMode_ == "broad" else "narrowPeak"
    requestedNestedRoccoIters = int(max(int(nestedRoccoIters), 0))
    effectiveNestedRoccoIters = (
        min(requestedNestedRoccoIters, 1)
        if peakMode_ == "broad"
        else requestedNestedRoccoIters
    )
    effectiveMassiveSubpeakCleanup = bool(massiveSubpeakCleanup) and peakMode_ == "narrow"
    uncertaintyScoreMode_ = _normalizeUncertaintyScoreMode(uncertaintyScoreMode)
    uncertaintyScoreZ_ = _validateUncertaintyScoreZ(uncertaintyScoreZ)
    metadataDetail_ = _normalizeRoccoMetadataDetail(metadataDetail)
    if uncertaintyScoreMode_ == "lower_confidence" and uncertaintyBedGraphFile is None:
        raise ValueError(
            "`lower_confidence` uncertaintyScoreMode requires an uncertainty bedGraph"
        )
    blacklistByChrom = _readBlacklistIntervalsByChrom(blacklistBedFile)
    chromData = _readAlignedConsenrichBedGraphs(
        stateBedGraphFile,
        uncertaintyBedGraphFile=uncertaintyBedGraphFile,
        chromosomes=chromosomes,
    )
    stateBase = Path(stateBedGraphFile)
    if outPath is None:
        outPath = str(stateBase.with_name(f"{stateBase.stem}_rocco.{outputFormat}"))
    gappedPath: str | None = None
    if peakMode_ == "broad":
        gappedPath = str(Path(outPath).with_suffix(".gappedPeak"))
    if writeMetadata and metaPath is None:
        metaPath = f"{outPath}.json"
    if not writeMetadata:
        metaPath = None
    nestedRoccoSubproblemDetailsPath: str | None = None
    if verbose:
        nestedRoccoSubproblemDetailsPath = f"{outPath}.nested_rocco_subproblems.jsonl"
        Path(nestedRoccoSubproblemDetailsPath).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(nestedRoccoSubproblemDetailsPath).write_text("", encoding="utf-8")
        logger.info(
            "writing nested ROCCO subproblem solving details to %s",
            nestedRoccoSubproblemDetailsPath,
        )

    allRows: List[List[str | int | float]] = []
    allGappedRows: List[List[str | int | float]] = []
    meta: Dict[str, Any] = {
        "settings": {
            "state_bedgraph": str(stateBedGraphFile),
            "uncertainty_bedgraph": (
                None
                if uncertaintyBedGraphFile is None
                else str(uncertaintyBedGraphFile)
            ),
            "blacklist_bed": None if blacklistBedFile is None else str(blacklistBedFile),
            "blacklist_filter_policy": "drop_any_overlap",
            "budget_method": "dwb_tail_occupancy",
            "null_calibration_method": "stationary_null_dwb",
            "peak_scoring_method": "stationary_dwb_null_replay_multiscale_segments",
            "num_bootstrap": int(numBootstrap),
            "threshold_z": float(thresholdZ),
            "peak_mode": str(peakMode_),
            "broad_weak_threshold_z": float(broadWeakThresholdZ_),
            "broad_max_gap_bp": broadMaxGapBP_,
            "broad_min_peak_bp": int(_MATCHING_DEFAULT_BROAD_MIN_PEAK_BP),
            "broad_parent_gamma_multiplier": float(
                _MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER
            ),
            "broad_bridge_dip_penalty_fraction": float(
                _BROAD_BRIDGE_DIP_PENALTY_FRACTION
            ),
            "null_quantile": float(_ROCCO_NULL_QUANTILE),
            "threshold_z_grid": [float(z) for z in _resolveThresholdZGrid(thresholdZ)],
            "nested_rocco_iters": int(effectiveNestedRoccoIters),
            "nested_rocco_requested_iters": int(requestedNestedRoccoIters),
            "nested_rocco_budget_scale": float(
                np.clip(float(nestedRoccoBudgetScale), 0.0, 1.0)
            ),
            "nested_rocco_jaccard": float(_NESTED_ROCCO_JACCARD_DEFAULT),
            "nested_rocco_min_parent_steps": int(_NESTED_ROCCO_MIN_PARENT_STEPS),
            "nested_rocco_min_child_steps": int(_NESTED_ROCCO_MIN_CHILD_STEPS),
            "nested_rocco_subproblem_policy": "parent_conditioned_min_run_dp",
            "nested_rocco_budget_policy": _NESTED_ROCCO_BUDGET_POLICY,
            "nested_rocco_diagnostics": bool(verbose),
            "nested_rocco_subproblem_details": nestedRoccoSubproblemDetailsPath,
            "peak_output_format": str(outputFormat),
            "rocco_min_peak_bp": int(_ROCCO_MIN_PEAK_BP),
            "massive_subpeak_cleanup": bool(effectiveMassiveSubpeakCleanup),
            "massive_subpeak_cleanup_requested": bool(massiveSubpeakCleanup),
            "massive_subpeak_cleanup_policy": (
                "robust_log_width_tail_gap_split_or_contract"
            ),
            "massive_subpeak_min_bp": int(_MASSIVE_SUBPEAK_MIN_BP),
            "massive_subpeak_width_alpha": float(_MASSIVE_SUBPEAK_WIDTH_ALPHA),
            "massive_subpeak_max_fraction": float(_MASSIVE_SUBPEAK_MAX_FRACTION),
            "massive_subpeak_min_log_gap": float(_MASSIVE_SUBPEAK_MIN_LOG_GAP),
            "massive_subpeak_split_quantile": float(_MASSIVE_SUBPEAK_SPLIT_QUANTILE),
            "massive_subpeak_split_z": float(_MASSIVE_SUBPEAK_SPLIT_Z),
            "massive_subpeak_min_child_bp": int(_MASSIVE_SUBPEAK_MIN_CHILD_BP),
            "massive_subpeak_min_child_fraction": float(
                _MASSIVE_SUBPEAK_MIN_CHILD_FRACTION
            ),
            "export_trim": "summit_component_score_above_floor",
            "export_trim_score_floor": 0.0,
            "export_filter": "drop_median_signal_below_negative_local_median_p",
            "export_filter_threshold": f"-{exportFilterUncertaintyMultiplier_:g} * median(local_uncertainty)",
            "export_filter_uncertainty_multiplier": float(
                exportFilterUncertaintyMultiplier_
            ),
            "min_peak_score": None if minPeakScore_ is None else float(minPeakScore_),
            "min_peak_score_field": "signalValue",
            "min_peak_score_narrowpeak_column": 7,
            "export_filter_uses_uncertainty_bedgraph": True,
            "uncertainty_score_mode": str(uncertaintyScoreMode_),
            "uncertainty_score_z": float(uncertaintyScoreZ_),
            "dwb_null_enabled": True,
            "metadata_detail": metadataDetail_,
            "rand_seed": int(randSeed),
        },
        "pooled_null_floor": None,
        "budget_shrinkage": None,
        "blacklist_filter": {
            "blacklist_bed": None if blacklistBedFile is None else str(blacklistBedFile),
            "policy": "drop_any_overlap",
            "dropped": 0,
            "kept": 0,
        },
        "chromosomes": {},
    }
    chromWork: Dict[str, Dict[str, Any]] = {}

    for chromIndex, (chromosome, data) in enumerate(chromData.items()):
        state = np.asarray(data["state"], dtype=np.float64)
        intervals = np.asarray(data["intervals"], dtype=np.int64)
        ends = np.asarray(data["ends"], dtype=np.int64)
        uncertainty = (
            None
            if data["uncertainty"] is None
            else np.asarray(data["uncertainty"], dtype=np.float64)
        )
        if state.size == 0:
            continue

        prepared = _prepareROCCOScoreAndNull(
            state,
            uncertainty=uncertainty,
            thresholdZ=thresholdZ_,
            numBootstrap=numBootstrap,
            dependenceSpan=dependenceSpan,
            kernel="bartlett",
            randomSeed=int(randSeed) + chromIndex,
            nullQuantile=_ROCCO_NULL_QUANTILE,
            thresholdZGrid=_ROCCO_BUDGET_Z_GRID,
            uncertaintyScoreMode=uncertaintyScoreMode_,
            uncertaintyScoreZ=uncertaintyScoreZ_,
        )
        scoreTrack = np.asarray(prepared["score_track"], dtype=np.float64)
        budgetRaw, budgetDetails = _estimateBudgetForPreparedROCCOScore(
            prepared,
            statistic="occupancy",
            numBootstrap=numBootstrap,
            dependenceSpan=dependenceSpan,
            thresholdZ=thresholdZ_,
            randomSeed=int(randSeed) + chromIndex,
            nullQuantile=_ROCCO_NULL_QUANTILE,
            returnDetails=True,
        )
        gamma_, gammaDetails = estimateROCCOGamma(
            scoreTrack,
            dependenceSpan=budgetDetails["dependence_span"],
            gammaSpan=budgetDetails.get(
                "context_span_lower",
                budgetDetails["dependence_span"],
            ),
            gamma=gamma,
            gammaScale=gammaScale,
            nullCenter=float(prepared["null_center"]),
            threshold=float(prepared["threshold"]),
            returnDetails=True,
        )
        weakPrepared = None
        weakBudgetRaw = None
        weakBudgetDetails = None
        weakGamma = None
        weakGammaDetails = None
        weakScoreTrack = None
        if peakMode_ == "broad":
            weakPrepared = _prepareROCCOScoreAndNull(
                state,
                uncertainty=uncertainty,
                thresholdZ=broadWeakThresholdZ_,
                numBootstrap=numBootstrap,
                dependenceSpan=dependenceSpan,
                kernel="bartlett",
                randomSeed=int(randSeed) + chromIndex,
                nullQuantile=_ROCCO_NULL_QUANTILE,
                thresholdZGrid=_ROCCO_BUDGET_Z_GRID,
                uncertaintyScoreMode=uncertaintyScoreMode_,
                uncertaintyScoreZ=uncertaintyScoreZ_,
            )
            weakScoreTrack = np.asarray(weakPrepared["score_track"], dtype=np.float64)
            weakBudgetRaw, weakBudgetDetails = _estimateBudgetForPreparedROCCOScore(
                weakPrepared,
                statistic="occupancy",
                numBootstrap=numBootstrap,
                dependenceSpan=dependenceSpan,
                thresholdZ=broadWeakThresholdZ_,
                randomSeed=int(randSeed) + chromIndex,
                nullQuantile=_ROCCO_NULL_QUANTILE,
                returnDetails=True,
            )
            weakGamma = float(
                _MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER * float(gamma_)
            )
            weakGammaDetails = {
                "method": "child_gamma_multiplier",
                "child_gamma": float(gamma_),
                "parent_gamma_multiplier": float(
                    _MATCHING_DEFAULT_BROAD_PARENT_GAMMA_MULTIPLIER
                ),
                "gamma": float(weakGamma),
            }
        chromWork[str(chromosome)] = {
            "state": state,
            "uncertainty": uncertainty,
            "intervals": intervals,
            "ends": ends,
            "score_track": scoreTrack,
            "null_scale": float(prepared["null_scale"]),
            "budget_raw": float(budgetRaw),
            "budget_details": dict(budgetDetails),
            "gamma": float(gamma_),
            "gamma_details": dict(gammaDetails),
            "weak_prepared": weakPrepared,
            "weak_score_track": weakScoreTrack,
            "weak_budget_raw": weakBudgetRaw,
            "weak_budget_details": weakBudgetDetails,
            "weak_gamma": weakGamma,
            "weak_gamma_details": weakGammaDetails,
            "interval_bp": int(np.median(ends - intervals)),
            "prepared": prepared,
        }

    chromResults: Dict[str, Dict[str, Any]] = {}
    initialPeakWidthsBP: List[int] = []

    for chromosome, work in chromWork.items():
        state = np.asarray(work["state"], dtype=np.float64)
        intervals = np.asarray(work["intervals"], dtype=np.int64)
        ends = np.asarray(work["ends"], dtype=np.int64)
        scoreTrack = np.asarray(work["score_track"], dtype=np.float64)
        uncertainty = (
            None
            if work["uncertainty"] is None
            else np.asarray(work["uncertainty"], dtype=np.float64)
        )
        nullScale = float(work["null_scale"])
        budget = float(work["budget_raw"])
        budgetDetails = dict(work["budget_details"])
        budgetDetails["budget_pre_shrink"] = float(work["budget_raw"])
        budgetDetails["budget_post_shrink"] = float(budget)
        budgetDetails["budget_shrink_delta"] = 0.0
        budgetDetails["budget_shrinkage_meta"] = None
        nestedMinRegionBP = int(
            _NESTED_ROCCO_MIN_PARENT_STEPS * max(int(work["interval_bp"]), 1)
        )
        exportMinSubpeakBins = int(_NESTED_ROCCO_MIN_CHILD_STEPS)

        solution, objective, solveDetails = solveChromROCCO(
            scoreTrack,
            budget=budget,
            gamma=float(work["gamma"]),
            selectionPenalty=selectionPenalty,
            returnDetails=True,
        )
        firstPassSolution = np.asarray(solution, dtype=np.uint8)
        refinedSolution, nestedDetails = _refineNestedROCCOSolution(
            scoreTrack,
            firstPassSolution,
            gamma=float(work["gamma"]),
            selectionPenalty=float(solveDetails["selection_penalty"]),
            nestedRoccoIters=effectiveNestedRoccoIters,
            nestedRoccoBudgetScale=nestedRoccoBudgetScale,
            jaccardThreshold=_NESTED_ROCCO_JACCARD_DEFAULT,
            intervals=intervals,
            ends=ends,
            rawScores=scoreTrack,
            minRegionBP=int(nestedMinRegionBP),
            minRegionBins=int(exportMinSubpeakBins),
            diagnostics=bool(verbose),
            diagnosticLabel=str(chromosome),
            diagnosticDetailPath=nestedRoccoSubproblemDetailsPath,
        )
        solution = np.asarray(refinedSolution, dtype=np.uint8)
        finalObjective = _roccoObjectiveForSolution(
            scoreTrack,
            solution,
            gamma=float(work["gamma"]),
        )
        solveDetails = dict(solveDetails)
        solveDetails["first_pass_selected_count"] = int(np.sum(firstPassSolution))
        solveDetails["final_selected_count"] = int(np.sum(solution))
        solveDetails["nested_rocco_iters"] = int(effectiveNestedRoccoIters)
        solveDetails["nested_rocco_requested_iters"] = int(
            requestedNestedRoccoIters
        )
        solveDetails["nested_rocco_budget_scale"] = float(
            np.clip(float(nestedRoccoBudgetScale), 0.0, 1.0)
        )
        solveDetails["nested_rocco_budget_policy"] = _NESTED_ROCCO_BUDGET_POLICY
        solveDetails["nested_rocco_stop_reason"] = str(nestedDetails["stop_reason"])
        exportTrimScoreFloor = 0.0
        roccoPrefix = "consenrichROCCO"
        broadParentSolution = None
        broadParentDetails: Dict[str, Any] | None = None
        if peakMode_ == "broad":
            weakScoreTrack = np.asarray(work["weak_score_track"], dtype=np.float64)
            weakBudgetDetails = dict(work["weak_budget_details"])
            weakSolution, weakObjective, weakSolveDetails = solveChromROCCO(
                weakScoreTrack,
                budget=float(work["weak_budget_raw"]),
                gamma=float(work["weak_gamma"]),
                selectionPenalty=selectionPenalty,
                returnDetails=True,
            )
            envelopeSolution = (
                (np.asarray(weakSolution, dtype=np.uint8) > 0)
                | (np.asarray(solution, dtype=np.uint8) > 0)
            ).astype(np.uint8)
            anchorSolution = np.asarray(envelopeSolution, dtype=bool)
            weakSupportThreshold = float(work["weak_prepared"]["threshold"])
            weakSupportSolution = np.asarray(
                weakScoreTrack >= weakSupportThreshold,
                dtype=np.uint8,
            )
            weakSupportRuns = _selectedCoordinateRunBounds(
                weakSupportSolution,
                intervals,
                ends,
            )
            weakRuns = []
            for supportStart, supportEnd in weakSupportRuns:
                if np.any(anchorSolution[int(supportStart) : int(supportEnd) + 1]):
                    weakRuns.append((int(supportStart), int(supportEnd)))
            if not weakRuns:
                weakRuns = _selectedCoordinateRunBounds(
                    envelopeSolution,
                    intervals,
                    ends,
                )
            broadMaxGapResolved = (
                int(broadMaxGapBP_)
                if broadMaxGapBP_ is not None
                else int(
                    4
                    * int(weakBudgetDetails["dependence_span"])
                    * max(int(work["interval_bp"]), 1)
                )
            )
            mergedRuns, mergeDetails = _mergeBroadRunsByObjective(
                weakRuns,
                weakScoreTrack,
                intervals,
                ends,
                str(chromosome),
                selectionPenalty=float(weakSolveDetails["selection_penalty"]),
                boundaryCost=float(work["weak_gamma"]),
                maxGapBP=int(broadMaxGapResolved),
                blacklistByChrom=blacklistByChrom,
                dipPenaltyFraction=float(_BROAD_BRIDGE_DIP_PENALTY_FRACTION),
            )
            broadParentSolution = np.zeros_like(envelopeSolution, dtype=np.uint8)
            for parentStart, parentEnd in mergedRuns:
                broadParentSolution[int(parentStart) : int(parentEnd) + 1] = 1
            broadParentDetails = {
                "enabled": True,
                "weak_threshold_z": float(broadWeakThresholdZ_),
                "weak_budget": float(work["weak_budget_raw"]),
                "weak_objective": float(weakObjective),
                "weak_solve_details": dict(weakSolveDetails),
                "weak_budget_details": weakBudgetDetails,
                "weak_gamma": float(work["weak_gamma"]),
                "weak_gamma_details": dict(work["weak_gamma_details"]),
                "strong_child_union_applied": True,
                "weak_support_threshold": float(weakSupportThreshold),
                "weak_support_run_count": int(len(weakSupportRuns)),
                "weak_support_touching_run_count": int(len(weakRuns)),
                "broad_bridge_dip_penalty_fraction": float(
                    _BROAD_BRIDGE_DIP_PENALTY_FRACTION
                ),
                "max_gap_bp": int(broadMaxGapResolved),
                "merge_details": mergeDetails,
            }
            rows, gappedRows, peakMeta, exportDetails = _solutionToChromBroadPeakRows(
                str(chromosome),
                intervals,
                ends,
                state,
                np.asarray(weakScoreTrack, dtype=np.float64),
                broadParentSolution,
                solution,
                prefix=roccoPrefix,
                uncertainty=uncertainty,
                exportFilterUncertaintyMultiplier=float(
                    exportFilterUncertaintyMultiplier_
                ),
                minPeakBP=int(_MATCHING_DEFAULT_BROAD_MIN_PEAK_BP),
                returnExportDetails=True,
            )
            initialPeakWidthsBP.extend(
                [int(meta_["end"]) - int(meta_["start"]) for meta_ in peakMeta]
            )
            nestedHierarchy = None
        else:
            nestedHierarchy = _buildRoccoNestedHierarchy(
                str(chromosome),
                intervals,
                ends,
                firstPassSolution,
                solution,
                roccoPrefix,
            )
            rows, peakMeta, exportDetails = _solutionToChromNarrowPeakRows(
                str(chromosome),
                intervals,
                ends,
                state,
                np.asarray(scoreTrack, dtype=np.float64),
                solution,
                prefix=roccoPrefix,
                nullScale=float(nullScale),
                uncertainty=uncertainty,
                trimScoreFloor=float(exportTrimScoreFloor),
                subpeakSelectionPenalty=float(solveDetails["selection_penalty"]),
                subpeakBoundaryCost=float(0.25 * float(work["gamma"])),
                minSubpeakBins=int(exportMinSubpeakBins),
                exportFilterUncertaintyMultiplier=float(
                    exportFilterUncertaintyMultiplier_
                ),
                nestedHierarchy=nestedHierarchy,
                returnExportDetails=True,
            )
            gappedRows = []
            initialPeakWidthsBP.extend(
                [int(meta_["end"]) - int(meta_["start"]) for meta_ in peakMeta]
            )
        chromResults[str(chromosome)] = {
            "state": state,
            "intervals": intervals,
            "ends": ends,
            "uncertainty": uncertainty,
            "score_track": scoreTrack,
            "null_scale": float(nullScale),
            "budget": float(budget),
            "budget_details": budgetDetails,
            "objective": float(objective),
            "final_objective": float(finalObjective),
            "solve_details": solveDetails,
            "nested_details": nestedDetails,
            "nested_hierarchy": nestedHierarchy,
            "broad_parent_solution": broadParentSolution,
            "broad_parent_details": broadParentDetails,
            "first_pass_solution": firstPassSolution,
            "solution": solution,
            "export_trim_score_floor": float(exportTrimScoreFloor),
            "initial_rows": rows,
            "initial_gapped_rows": gappedRows,
            "initial_peak_meta": peakMeta,
            "initial_export_details": exportDetails,
            "nested_min_region_bp": int(nestedMinRegionBP),
            "export_min_subpeak_bins": int(exportMinSubpeakBins),
            "work": work,
        }

    massiveWidthPolicy = _learnMassiveSubpeakWidthPolicy(
        initialPeakWidthsBP,
        enabled=bool(effectiveMassiveSubpeakCleanup),
        minBP=int(_MASSIVE_SUBPEAK_MIN_BP),
        alpha=float(_MASSIVE_SUBPEAK_WIDTH_ALPHA),
        bulkQuantile=float(_MASSIVE_SUBPEAK_WIDTH_BULK_QUANTILE),
        maxFraction=float(_MASSIVE_SUBPEAK_MAX_FRACTION),
        minLogGap=float(_MASSIVE_SUBPEAK_MIN_LOG_GAP),
        minPeaks=int(_MASSIVE_SUBPEAK_MIN_PEAKS),
    )
    meta["massive_subpeak_width_policy"] = dict(massiveWidthPolicy)

    blacklistDroppedTotal = 0
    for chromosome, result in chromResults.items():
        work = dict(result["work"])
        state = np.asarray(result["state"], dtype=np.float64)
        intervals = np.asarray(result["intervals"], dtype=np.int64)
        ends = np.asarray(result["ends"], dtype=np.int64)
        scoreTrack = np.asarray(result["score_track"], dtype=np.float64)
        uncertainty = result["uncertainty"]
        solution = np.asarray(result["solution"], dtype=np.uint8)
        solveDetails = dict(result["solve_details"])
        nestedHierarchy = result["nested_hierarchy"]
        exportTrimScoreFloor = float(result["export_trim_score_floor"])
        if bool(massiveWidthPolicy.get("active", False)):
            rows, peakMeta, exportDetails = _solutionToChromNarrowPeakRows(
                str(chromosome),
                intervals,
                ends,
                state,
                scoreTrack,
                solution,
                prefix="consenrichROCCO",
                nullScale=float(result["null_scale"]),
                uncertainty=uncertainty,
                trimScoreFloor=float(exportTrimScoreFloor),
                subpeakSelectionPenalty=float(solveDetails["selection_penalty"]),
                subpeakBoundaryCost=float(0.25 * float(work["gamma"])),
                minSubpeakBins=int(result["export_min_subpeak_bins"]),
                massiveSubpeakCleanup=True,
                massiveSubpeakWidthPolicy=massiveWidthPolicy,
                massiveSubpeakSplitQuantile=float(_MASSIVE_SUBPEAK_SPLIT_QUANTILE),
                massiveSubpeakSplitZ=float(_MASSIVE_SUBPEAK_SPLIT_Z),
                exportFilterUncertaintyMultiplier=float(
                    exportFilterUncertaintyMultiplier_
                ),
                nestedHierarchy=nestedHierarchy,
                returnExportDetails=True,
            )
            gappedRows = []
        else:
            rows = list(result["initial_rows"])
            gappedRows = list(result["initial_gapped_rows"])
            peakMeta = list(result["initial_peak_meta"])
            exportDetails = dict(result["initial_export_details"])
            exportDetails["massive_subpeak_width_policy"] = dict(massiveWidthPolicy)
        exportDetails["peak_output_format"] = str(outputFormat)
        exportDetails["nested_min_region_bp"] = int(result["nested_min_region_bp"])
        exportDetails["export_min_subpeak_bins"] = int(
            result["export_min_subpeak_bins"]
        )
        rows, peakMeta, blacklistDropped = _filterNarrowPeakRowsByBlacklist(
            rows,
            peakMeta,
            blacklistByChrom,
        )
        if peakMode_ == "broad":
            gappedRows = _pairedGappedRowsForPeakMeta(gappedRows, peakMeta)
        blacklistDroppedTotal += int(blacklistDropped)
        scoringTrack = scoreTrack
        scoringPrepared = work.get("prepared", {})
        if peakMode_ == "broad":
            scoringTrack = np.asarray(work["weak_score_track"], dtype=np.float64)
            scoringPrepared = work.get("weak_prepared", {})
        _addDWBPeakScoringToPeakMeta(
            peakMeta,
            scoringTrack,
            scoringPrepared,
            exportDetails=exportDetails,
            minRunBins=int(_NESTED_ROCCO_MIN_CHILD_STEPS),
            intervals=intervals,
            ends=ends,
        )
        exportDetails["min_peak_score"] = (
            None if minPeakScore_ is None else float(minPeakScore_)
        )
        exportDetails["min_peak_score_field"] = "signalValue"
        minPeakScoreColumnIndex = 6
        exportDetails["min_peak_score_output_column"] = int(
            minPeakScoreColumnIndex + 1
        )
        exportDetails["min_peak_score_narrowpeak_column"] = 7
        exportDetails["min_peak_score_filter_active"] = minPeakScore_ is not None
        exportDetails["num_segments_min_peak_score_evaluated"] = int(len(rows))
        exportDetails["num_segments_dropped_min_peak_score"] = 0
        if minPeakScore_ is not None:
            retainedRows: List[List[str | int | float]] = []
            retainedPeakMeta: List[Dict[str, Any]] = []
            for row, peak in zip(rows, peakMeta):
                if float(row[minPeakScoreColumnIndex]) >= minPeakScore_:
                    retainedRows.append(row)
                    retainedPeakMeta.append(peak)
            exportDetails["num_segments_dropped_min_peak_score"] = int(
                len(rows) - len(retainedRows)
            )
            rows = retainedRows
            peakMeta = retainedPeakMeta
            if peakMode_ == "broad":
                gappedRows = _pairedGappedRowsForPeakMeta(gappedRows, peakMeta)
        if peakMode_ == "broad":
            _fillBroadRowsDWBValues(rows, gappedRows, peakMeta)
        nullReplayDiagnostics = dict(
            exportDetails.get("null_replay_false_segment_diagnostics", {})
        )
        if (
            bool(verbose)
            and nullReplayDiagnostics
            and int(max(int(nestedRoccoIters), 0)) == 0
        ):
            logger.info(
                "null replay false-segment diagnostics %s replays=%d observed=%d false_mean=%.6g false_q95=%.6g fdr=%.6g",
                str(chromosome),
                int(nullReplayDiagnostics.get("num_replays", 0)),
                int(nullReplayDiagnostics.get("observed_segment_count", 0)),
                float(nullReplayDiagnostics.get("false_segment_count_mean", 0.0)),
                float(nullReplayDiagnostics.get("false_segment_count_q95", 0.0)),
                float(nullReplayDiagnostics.get("false_segment_fdr_estimate", 0.0)),
            )
            if nestedRoccoSubproblemDetailsPath is not None:
                with open(
                    nestedRoccoSubproblemDetailsPath,
                    "a",
                    encoding="utf-8",
                ) as detailHandle:
                    detailHandle.write(
                        json.dumps(
                            {
                                "event": "null_replay_false_segments",
                                "chromosome": str(chromosome),
                                **nullReplayDiagnostics,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        exportDetails["blacklist_filter"] = {
            "blacklist_bed": None if blacklistBedFile is None else str(blacklistBedFile),
            "policy": "drop_any_overlap",
            "dropped": int(blacklistDropped),
            "kept": int(len(rows)),
        }
        exportDetails["num_segments_kept_before_blacklist"] = int(
            exportDetails.get("num_segments_kept", len(rows) + int(blacklistDropped))
        )
        exportDetails["num_segments_kept"] = int(len(rows))
        hierarchy = _resolveRoccoExportHierarchy(nestedHierarchy)
        hierarchySummary = _summarizeRoccoHierarchy(hierarchy, peakMeta)
        if hierarchySummary:
            exportDetails["nested_hierarchy_summary"] = hierarchySummary
        allRows.extend(rows)
        if peakMode_ == "broad":
            allGappedRows.extend(gappedRows)

        firstPassSolution = np.asarray(result["first_pass_solution"], dtype=np.uint8)
        meta["chromosomes"][str(chromosome)] = {
            "n_loci": int(state.size),
            "interval_bp": int(work["interval_bp"]),
            "state_diagnostics": dict(
                (stateDiagnosticsByChromosome or {}).get(str(chromosome), {})
            ),
            "budget": float(result["budget"]),
            "objective": float(result["final_objective"]),
            "first_pass_objective": float(result["objective"]),
            "num_parent_segments": int(
                np.sum(
                    np.diff(
                        np.pad(
                            np.asarray(firstPassSolution, dtype=np.int8),
                            (1, 1),
                            mode="constant",
                        )
                    )
                    == 1
                )
            ),
            "num_segments": int(len(rows)),
            "num_candidate_segments": int(exportDetails["num_candidate_segments"]),
            "num_segments_dropped_median_signal_local_p": int(
                exportDetails["num_segments_dropped_median_signal_local_p"]
            ),
            "num_segments_dropped_min_peak_bp": int(
                exportDetails["num_segments_dropped_min_peak_bp"]
            ),
            "num_segments_dropped_min_peak_score": int(
                exportDetails["num_segments_dropped_min_peak_score"]
            ),
            "budget_details": dict(result["budget_details"]),
            "gamma_details": dict(work["gamma_details"]),
            "solve_details": solveDetails,
            "nested_rocco_details": result["nested_details"],
            "broad_parent_details": result["broad_parent_details"],
            "nested_hierarchy": nestedHierarchy,
            "nested_hierarchy_summary": dict(
                exportDetails.get("nested_hierarchy_summary", {})
            ),
            "nested_min_region_bp": int(result["nested_min_region_bp"]),
            "export_min_subpeak_bins": int(result["export_min_subpeak_bins"]),
            "null_replay_false_segment_diagnostics": nullReplayDiagnostics,
            "export_trim_score_floor": float(exportTrimScoreFloor),
            "export_details": exportDetails,
            "candidate_details": list(exportDetails.get("candidate_details", [])),
            "candidate_significance": dict(
                exportDetails.get("candidate_significance", {})
            ),
            "peak_details": peakMeta,
        }

    allRows.sort(key=lambda row: (str(row[0]), int(row[1]), int(row[2])))
    allGappedRows.sort(key=lambda row: (str(row[0]), int(row[1]), int(row[2])))
    meta["blacklist_filter"] = {
        "blacklist_bed": None if blacklistBedFile is None else str(blacklistBedFile),
        "policy": "drop_any_overlap",
        "dropped": int(blacklistDroppedTotal),
        "kept": int(len(allRows)),
    }
    with open(outPath, "w", encoding="utf-8") as handle:
        for row in allRows:
            handle.write("\t".join(map(str, row)) + "\n")
    if gappedPath is not None:
        with open(gappedPath, "w", encoding="utf-8") as handle:
            for row in allGappedRows:
                handle.write("\t".join(map(str, row)) + "\n")

    if metaPath is not None:
        _writeRoccoMetadata(
            metaPath,
            meta,
            metadataDetail=metadataDetail_,
            maxNonTrackFileBytes=maxNonTrackFileBytes,
        )
    summary = _buildRoccoSummary(
        outPath=str(outPath),
        metaPath=str(metaPath),
        nestedRoccoSubproblemDetailsPath=nestedRoccoSubproblemDetailsPath,
        gappedPath=gappedPath,
        rows=allRows,
        meta=meta,
    )
    _logRoccoSummary(summary)
    _logOutputInventory(summary)
    if returnSummary:
        return str(outPath), summary
    return str(outPath)


_ROCCO_CUTOFF_REPORT_SWEEPS: Tuple[
    Tuple[str, str, str, Tuple[float, ...], Mapping[str, Any], bool],
    ...,
] = (
    (
        "thresholdZ",
        "matchingParams.thresholdZ",
        "thresholdZ",
        (1.5, 2.5, 3.0),
        {},
        False,
    ),
    (
        "gamma",
        "matchingParams.gamma",
        "gamma",
        (0.1, 0.5, 1.0),
        {},
        False,
    ),
    (
        "uncertaintyScoreZ",
        "matchingParams.uncertaintyScoreZ",
        "uncertaintyScoreZ",
        (0.5, 1.0, 1.5),
        {"uncertaintyScoreMode": "lower_confidence"},
        True,
    ),
    (
        "nestedRoccoBudgetScale",
        "matchingParams.nestedRoccoBudgetScale",
        "nestedRoccoBudgetScale",
        (0.5, 1.0),
        {},
        False,
    ),
)

_ROCCO_CUTOFF_REPORT_FIELDS: Tuple[str, ...] = (
    "sweep",
    "parameter",
    "value",
    "thresholdZ",
    "gamma",
    "selectionPenalty",
    "gammaScale",
    "nestedRoccoIters",
    "nestedRoccoBudgetScale",
    "exportFilterUncertaintyMultiplier",
    "minPeakScore",
    "uncertaintyScoreMode",
    "uncertaintyScoreZ",
    "numBootstrap",
    "dependenceSpan",
    "peak_count",
    "total_peak_bp",
    "min_width_bp",
    "median_width_bp",
    "max_width_bp",
    "blacklist_dropped",
    "blacklist_kept",
    "per_chrom_peak_counts",
    "narrowPeak_path",
)


def _cutoffValueText(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        valueFloat = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(valueFloat):
        return f"{valueFloat:.10g}"
    return str(value)


def _cutoffValueKey(value: Any) -> str:
    text = _cutoffValueText(value).strip().lower()
    replacements = {
        ".": "p",
        "-": "m",
        "+": "p",
    }
    pieces: List[str] = []
    prevUnderscore = False
    for char in text:
        if char.isalnum():
            pieces.append(char)
            prevUnderscore = False
        elif char in replacements:
            pieces.append(replacements[char])
            prevUnderscore = False
        elif not prevUnderscore:
            pieces.append("_")
            prevUnderscore = True
    key = "".join(pieces).strip("_")
    return key or "value"


def _sameCutoffValue(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        leftFloat = float(left)
        rightFloat = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    return bool(np.isclose(leftFloat, rightFloat, rtol=0.0, atol=1.0e-12))


def _cutoffPerChromCounts(summary: Mapping[str, Any]) -> str:
    perChrom = summary.get("per_chrom", {})
    if not isinstance(perChrom, Mapping):
        return ""
    parts: List[str] = []
    for chrom in sorted(str(key) for key in perChrom.keys()):
        chromSummary = perChrom.get(chrom, {})
        if not isinstance(chromSummary, Mapping):
            continue
        parts.append(f"{chrom}:{int(chromSummary.get('exported_peak_count', 0))}")
    return ",".join(parts)


def _summarizeNarrowPeakFile(
    path: Path,
    templateSummary: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    rows: List[List[str | int | float]] = []
    perChrom: Dict[str, Dict[str, int]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                chrom = str(parts[0])
                start = int(parts[1])
                end = int(parts[2])
                rows.append([chrom, start, end])
                chromSummary = perChrom.setdefault(
                    chrom,
                    {"exported_peak_count": 0, "total_peak_bp": 0},
                )
                chromSummary["exported_peak_count"] += 1
                chromSummary["total_peak_bp"] += max(end - start, 0)

    widthSummary = _summarizePeakWidthsFromRows(rows)
    templateBlacklist = (
        templateSummary.get("blacklist", {})
        if isinstance(templateSummary, Mapping)
        else {}
    )
    if not isinstance(templateBlacklist, Mapping):
        templateBlacklist = {}
    return {
        "narrowPeak_path": str(path),
        "metadata_json_path": None,
        **widthSummary,
        "blacklist": {
            "blacklist_bed": templateBlacklist.get("blacklist_bed"),
            "policy": templateBlacklist.get("policy", "drop_any_overlap"),
            "dropped": int(templateBlacklist.get("dropped", 0)),
            "kept": int(templateBlacklist.get("kept", widthSummary["exported_peak_count"])),
        },
        "per_chrom": perChrom,
    }


def _cutoffSummaryRow(
    *,
    sweep: str,
    parameter: str,
    value: Any,
    settings: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, str]:
    blacklist = summary.get("blacklist", {})
    if not isinstance(blacklist, Mapping):
        blacklist = {}
    row = {
        "sweep": str(sweep),
        "parameter": str(parameter),
        "value": _cutoffValueText(value),
        "thresholdZ": _cutoffValueText(settings["thresholdZ"]),
        "gamma": _cutoffValueText(settings["gamma"]),
        "selectionPenalty": _cutoffValueText(settings["selectionPenalty"]),
        "gammaScale": _cutoffValueText(settings["gammaScale"]),
        "nestedRoccoIters": _cutoffValueText(settings["nestedRoccoIters"]),
        "nestedRoccoBudgetScale": _cutoffValueText(settings["nestedRoccoBudgetScale"]),
        "exportFilterUncertaintyMultiplier": _cutoffValueText(
            settings["exportFilterUncertaintyMultiplier"]
        ),
        "minPeakScore": _cutoffValueText(settings["minPeakScore"]),
        "uncertaintyScoreMode": _cutoffValueText(settings["uncertaintyScoreMode"]),
        "uncertaintyScoreZ": _cutoffValueText(settings["uncertaintyScoreZ"]),
        "numBootstrap": _cutoffValueText(settings["numBootstrap"]),
        "dependenceSpan": _cutoffValueText(settings["dependenceSpan"]),
        "peak_count": _cutoffValueText(summary.get("exported_peak_count")),
        "total_peak_bp": _cutoffValueText(summary.get("total_peak_bp")),
        "min_width_bp": _cutoffValueText(summary.get("min_width_bp")),
        "median_width_bp": _cutoffValueText(summary.get("median_width_bp")),
        "max_width_bp": _cutoffValueText(summary.get("max_width_bp")),
        "blacklist_dropped": _cutoffValueText(blacklist.get("dropped", 0)),
        "blacklist_kept": _cutoffValueText(blacklist.get("kept", 0)),
        "per_chrom_peak_counts": _cutoffPerChromCounts(summary),
        "narrowPeak_path": str(summary.get("narrowPeak_path", "")),
    }
    return row


def solveRoccoCutoffReport(
    stateBedGraphFile: str,
    uncertaintyBedGraphFile: str | None = None,
    chromosomes: Iterable[str] | None = None,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    dependenceSpan: int | None = None,
    gamma: float | None = 0.5,
    selectionPenalty: float | None = None,
    gammaScale: float = 0.5,
    nestedRoccoIters: int = _NESTED_ROCCO_ITERS_DEFAULT,
    nestedRoccoBudgetScale: float = _NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    massiveSubpeakCleanup: bool = _MASSIVE_SUBPEAK_CLEANUP_DEFAULT,
    exportFilterUncertaintyMultiplier: float = (
        _EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
    ),
    minPeakScore: float | None = _MATCHING_DEFAULT_MIN_PEAK_SCORE,
    uncertaintyScoreMode: str = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    uncertaintyScoreZ: float = _MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    randSeed: int = 42,
    outDir: str | None = None,
    blacklistBedFile: str | None = None,
    baselineNarrowPeakFile: str | None = None,
    baselineSummary: Mapping[str, Any] | None = None,
) -> str:
    minPeakScore_ = _validateMinPeakScore(minPeakScore)
    stateBase = Path(stateBedGraphFile)
    chromosomesTuple = None if chromosomes is None else tuple(chromosomes)
    reportDir = (
        Path(outDir)
        if outDir is not None
        else stateBase.with_name(f"{stateBase.stem}_rocco_cutoff_analysis")
    )
    reportDir.mkdir(parents=True, exist_ok=True)

    baselineSettings: Dict[str, Any] = {
        "numBootstrap": int(numBootstrap),
        "thresholdZ": float(thresholdZ),
        "dependenceSpan": dependenceSpan,
        "gamma": gamma,
        "selectionPenalty": selectionPenalty,
        "gammaScale": float(gammaScale),
        "nestedRoccoIters": int(nestedRoccoIters),
        "nestedRoccoBudgetScale": float(nestedRoccoBudgetScale),
        "massiveSubpeakCleanup": bool(massiveSubpeakCleanup),
        "exportFilterUncertaintyMultiplier": float(exportFilterUncertaintyMultiplier),
        "minPeakScore": minPeakScore_,
        "uncertaintyScoreMode": str(uncertaintyScoreMode),
        "uncertaintyScoreZ": float(uncertaintyScoreZ),
        "randSeed": int(randSeed),
    }

    rows: List[Dict[str, str]] = []
    settingFields = tuple(baselineSettings.keys())
    seenSettings = {tuple(baselineSettings[field] for field in settingFields)}

    def runSweep(
        *,
        sweep: str,
        parameter: str,
        value: Any,
        settings: Mapping[str, Any],
        sourceNarrowPeakFile: str | None = None,
        sourceSummary: Mapping[str, Any] | None = None,
    ) -> None:
        fileStem = f"rocco_{sweep}"
        if parameter:
            fileStem = f"{fileStem}_{_cutoffValueKey(value)}"
        outPath = reportDir / f"{fileStem}.narrowPeak"
        staleMetaPath = reportDir / f"{fileStem}.narrowPeak.json"
        if staleMetaPath.exists():
            staleMetaPath.unlink()
        if sourceNarrowPeakFile is not None:
            sourcePath = Path(sourceNarrowPeakFile)
            if sourcePath.resolve() != outPath.resolve():
                shutil.copyfile(sourcePath, outPath)
            summary = _summarizeNarrowPeakFile(outPath, sourceSummary)
        else:
            _resultPath, summary = solveRocco(
                stateBedGraphFile,
                uncertaintyBedGraphFile=uncertaintyBedGraphFile,
                chromosomes=chromosomesTuple,
                numBootstrap=int(settings["numBootstrap"]),
                thresholdZ=float(settings["thresholdZ"]),
                dependenceSpan=settings["dependenceSpan"],
                gamma=settings["gamma"],
                selectionPenalty=settings["selectionPenalty"],
                gammaScale=float(settings["gammaScale"]),
                nestedRoccoIters=int(settings["nestedRoccoIters"]),
                nestedRoccoBudgetScale=float(settings["nestedRoccoBudgetScale"]),
                massiveSubpeakCleanup=bool(settings["massiveSubpeakCleanup"]),
                exportFilterUncertaintyMultiplier=float(
                    settings["exportFilterUncertaintyMultiplier"]
                ),
                minPeakScore=settings["minPeakScore"],
                uncertaintyScoreMode=str(settings["uncertaintyScoreMode"]),
                uncertaintyScoreZ=float(settings["uncertaintyScoreZ"]),
                randSeed=int(settings["randSeed"]),
                outPath=str(outPath),
                verbose=False,
                blacklistBedFile=blacklistBedFile,
                writeMetadata=False,
                returnSummary=True,
            )
        rows.append(
            _cutoffSummaryRow(
                sweep=sweep,
                parameter=parameter,
                value=value,
                settings=settings,
                summary=summary,
            )
        )

    runSweep(
        sweep="baseline",
        parameter="",
        value="baseline",
        settings=baselineSettings,
        sourceNarrowPeakFile=baselineNarrowPeakFile,
        sourceSummary=baselineSummary,
    )
    for (
        settingKey,
        configName,
        settingName,
        values,
        overrides,
        requiresUncertainty,
    ) in _ROCCO_CUTOFF_REPORT_SWEEPS:
        if requiresUncertainty and uncertaintyBedGraphFile is None:
            logger.warning(
                "Skipping ROCCO cutoff sweep %s because no uncertainty bedGraph is available.",
                settingKey,
            )
            continue
        baselineValue = baselineSettings[settingKey]
        for value in values:
            if not overrides and _sameCutoffValue(value, baselineValue):
                continue
            settings = dict(baselineSettings)
            settings[settingName] = value
            settings.update(dict(overrides))
            settingValues = tuple(settings[field] for field in settingFields)
            if settingValues in seenSettings:
                continue
            seenSettings.add(settingValues)
            runSweep(
                sweep=settingKey,
                parameter=configName,
                value=value,
                settings=settings,
            )

    summaryPath = reportDir / "cutoff_summary.tsv"
    with summaryPath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(_ROCCO_CUTOFF_REPORT_FIELDS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "rocco.cutoff_report rows=%d report_dir=%s summary=%s",
        len(rows),
        str(reportDir),
        str(summaryPath),
    )
    return str(reportDir)
