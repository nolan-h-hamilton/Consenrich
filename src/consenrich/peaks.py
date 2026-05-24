# -*- coding: utf-8 -*-
r"""Peak-calling helpers for ROCCO segmentation from Consenrich tracks."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from . import cconsenrich
from . import core
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
    NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    NESTED_ROCCO_ITERS_DEFAULT,
    NESTED_ROCCO_JACCARD_DEFAULT,
    NESTED_ROCCO_MIN_CHILD_STEPS,
    NESTED_ROCCO_MIN_PARENT_STEPS,
    NESTED_ROCCO_SUBTASK_MAX_ITER,
    ROCCO_BUDGET_MAX,
    ROCCO_BUDGET_MIN,
    ROCCO_BUDGET_Z_GRID,
    ROCCO_MAX_ITER_DEFAULT,
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
_NESTED_ROCCO_ITERS_DEFAULT = NESTED_ROCCO_ITERS_DEFAULT
_NESTED_ROCCO_JACCARD_DEFAULT = NESTED_ROCCO_JACCARD_DEFAULT
_NESTED_ROCCO_MIN_PARENT_STEPS = NESTED_ROCCO_MIN_PARENT_STEPS
_NESTED_ROCCO_MIN_CHILD_STEPS = NESTED_ROCCO_MIN_CHILD_STEPS
_NESTED_ROCCO_BUDGET_SCALE_DEFAULT = NESTED_ROCCO_BUDGET_SCALE_DEFAULT
_NESTED_ROCCO_SUBTASK_MAX_ITER = NESTED_ROCCO_SUBTASK_MAX_ITER
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
    returnDetails: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float | str | bool]]:
    r"""Use the fitted Consenrich state directly as the ROCCO score track."""
    state_ = _asFloatVector("state", state)
    seMode = "none"
    uncertaintyAvailable = False
    if uncertainty is not None:
        uncertainty_ = _asFloatVector("uncertainty", uncertainty)
        if uncertainty_.size != state_.size:
            raise ValueError("`uncertainty` must match `state` length")
        seMode = "ignored"
        uncertaintyAvailable = True

    if not returnDetails:
        return state_

    details: Dict[str, float | str | bool] = {
        "score_mode": "consenrich_state",
        "se_mode": str(seMode),
        "uncertainty_available": bool(uncertaintyAvailable),
        "uncertainty_used": False,
        "state_median": float(np.median(state_)),
        "state_abs_median": float(np.median(np.abs(state_))),
        "state_min": float(np.min(state_)),
        "state_max": float(np.max(state_)),
    }
    return state_, details


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
        multipliers = _generateDWBMultipliers(
            template_.size,
            dependenceSpan_,
            rng,
            kernel=kernel_,
        )
        draw = np.asarray(template_, dtype=np.float64) * np.asarray(
            multipliers, dtype=np.float64
        )
        draw -= float(np.mean(draw))
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
        multipliers = _generateDWBMultipliers(
            template_.size,
            dependenceSpan_,
            rng,
            kernel=kernel_,
        )
        draw = np.asarray(template_, dtype=np.float64) * np.asarray(
            multipliers, dtype=np.float64
        )
        draw -= float(np.mean(draw))
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
) -> Dict[str, Any]:
    state_ = _asFloatVector("state", state)
    stateNullCenter, stateNullScale, stateNullMeta = estimateROCCONull(
        state_,
        bulkQuantile=bulkQuantile,
    )
    scoreTrack, scoreMeta = consenrichStateScoreTrack(
        state_,
        uncertainty=uncertainty,
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
) -> Dict[str, Any]:
    r"""Prepare the direct Consenrich score track and robust null for ROCCO budgeting."""
    prepared = _prepareROCCOBaseScore(
        state,
        uncertainty=uncertainty,
        bulkQuantile=bulkQuantile,
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


def _bartlettKernel(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax <= 1.0, 1.0 - ax, 0.0)


def _parzenKernel(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    out = np.zeros_like(ax, dtype=np.float64)
    mask1 = ax <= 0.5
    mask2 = (ax > 0.5) & (ax <= 1.0)
    out[mask1] = 1.0 - 6.0 * ax[mask1] ** 2 + 6.0 * ax[mask1] ** 3
    out[mask2] = 2.0 * (1.0 - ax[mask2]) ** 3
    return out


def _quadraticSpectralKernel(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    out = np.zeros_like(ax, dtype=np.float64)
    zeroMask = ax < 1.0e-12
    out[zeroMask] = 1.0
    nz = ~zeroMask
    if np.any(nz):
        y = (6.0 * np.pi * ax[nz]) / 5.0
        out[nz] = (25.0 / (12.0 * np.pi * np.pi * ax[nz] * ax[nz])) * (
            (np.sin(y) / np.maximum(y, 1.0e-12)) - np.cos(y)
        )
    return out


def _kernelValues(name: str, lags: np.ndarray, bandwidth: int) -> np.ndarray:
    kernelName = str(name).strip().lower().replace("-", "_")
    x = np.abs(lags.astype(np.float64, copy=False)) / max(float(bandwidth), 1.0)
    if kernelName in {"bartlett", "triangle", "triangular"}:
        return _bartlettKernel(x)
    if kernelName in {"parzen"}:
        return _parzenKernel(x)
    if kernelName in {"qs", "quadratic_spectral", "quadraticspectral"}:
        return _quadraticSpectralKernel(x)
    raise ValueError(f"Unknown DWB kernel: {name}")


def _generateDWBMultipliers(
    n: int,
    bandwidth: int,
    rng: np.random.Generator,
    kernel: str = "bartlett",
) -> np.ndarray:
    bandwidth_ = max(int(bandwidth), 2)
    kernelName = str(kernel).strip().lower().replace("-", "_")
    maxLag = (
        bandwidth_
        if kernelName not in {"qs", "quadratic_spectral", "quadraticspectral"}
        else max(8 * bandwidth_, 32)
    )
    lags = np.arange(-maxLag, maxLag + 1, dtype=np.int64)
    weights = _kernelValues(kernel, lags, bandwidth_)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= math.sqrt(max(float(np.sum(weights * weights)), _TINY))

    noise = rng.standard_normal(int(n + 2 * maxLag))
    multipliers = np.convolve(noise, weights, mode="valid")[:n]
    multipliers = np.asarray(multipliers, dtype=np.float64)
    multipliers -= float(np.mean(multipliers))
    sd = float(np.std(multipliers, ddof=1)) if n >= 2 else 0.0
    if not np.isfinite(sd) or sd <= _TINY:
        return np.ones(n, dtype=np.float64)
    return multipliers / sd


def _estimateEffectiveSampleSize(
    values: np.ndarray,
    maxLag: int,
) -> Tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n < 2:
        return float(n), 1.0, 0

    x = x - float(np.mean(x))
    var = float(np.dot(x, x) / max(n, 1))
    if not np.isfinite(var) or var <= _TINY:
        return float(n), 1.0, 0

    maxLag_ = max(1, min(int(maxLag), n - 1))
    tau = 1.0
    lagsUsed = 0
    for lag in range(1, maxLag_ + 1):
        cov = float(np.dot(x[:-lag], x[lag:]) / max(n - lag, 1))
        rho = cov / var
        if (not np.isfinite(rho)) or rho <= 0.0:
            break
        tau += 2.0 * rho
        lagsUsed = lag

    tau = float(max(tau, 1.0))
    return float(n / tau), tau, lagsUsed


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


def _nestedBudgetTargetCount(
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
) -> Tuple[float, float, float]:
    scores_ = np.asarray(scores, dtype=np.float64)
    mask_ = np.asarray(mask, dtype=bool)
    costs_ = np.asarray(boundaryCosts, dtype=np.float64)
    selected = float(np.sum(scores_[mask_]))
    boundaryPenalty = 0.0
    previous = False
    for i, current in enumerate(mask_.tolist()):
        current_ = bool(current)
        if current_ != previous:
            boundaryPenalty += float(costs_[i])
        previous = current_
    if previous:
        boundaryPenalty += float(costs_[mask_.size])
    objective = float(selected - boundaryPenalty)
    penalized = float(objective - float(selectionPenalty) * float(np.sum(mask_)))
    return objective, penalized, float(boundaryPenalty)


def _bhQValues(pValues: npt.ArrayLike) -> np.ndarray:
    p = np.asarray(pValues, dtype=np.float64).ravel()
    if p.size == 0:
        return np.asarray([], dtype=np.float64)
    p = np.clip(np.where(np.isfinite(p), p, 1.0), 0.0, 1.0)
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
    p = np.clip(np.where(np.isfinite(p), p, 1.0), 0.0, 1.0)
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
        "min_bp": int(max(int(minBP), 1)),
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
        widths,
        bulkQuantile=float(bulkQuantile),
    )
    details["null_center"] = float(scoreMeta["center"])
    details["null_scale"] = float(scoreMeta["scale"])
    alpha_ = float(np.clip(float(alpha), 0.0, 1.0))
    minBP_ = float(details["min_bp"])
    tail = (
        np.isfinite(widths)
        & (widths >= minBP_)
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
        rightMask = widths >= float(right)
        rightCount = int(np.sum(rightMask))
        if rightCount < 1 or rightCount > maxCount:
            continue
        if not bool(np.all(qValues[rightMask] <= alpha_)):
            continue
        candidates.append((float(right), int(rightCount), float(logGap)))
    if not candidates:
        return details
    threshold, clusterCount, logGap = min(candidates, key=lambda item: item[0])
    details["width_threshold_bp"] = int(round(float(threshold)))
    details["num_width_cluster_candidates"] = int(clusterCount)
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
) -> Dict[str, int | float | bool | None]:
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
        "massive_subpeak_split_gain": (
            None if details.get("gain") is None else float(details["gain"])
        ),
        "massive_subpeak_split_z": (
            None if details.get("z") is None else float(details["z"])
        ),
        "massive_subpeak_gap_bins": (
            None if details.get("gap_bins") is None else int(details["gap_bins"])
        ),
        "massive_subpeak_parent_start_idx": int(originalStartIdx),
        "massive_subpeak_parent_end_idx": int(originalEndIdx),
    }


def _solveParentConditionedSubpeaks(
    scores: np.ndarray,
    boundaryCosts: npt.ArrayLike,
    selectionPenalty: float,
    minRunBins: int,
    anchorIndex: int | None = None,
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

    n = int(scores_.size)
    anchor = None if anchorIndex is None else int(np.clip(int(anchorIndex), 0, n - 1))
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
        forceOn = bool(anchor is not None and i == anchor)

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
                float(prevValues[0] - transitionCost + adjustedScore),
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
    objective, penalizedObjective, boundaryPenalty = _parentConditionedSubpeakObjective(
        scores_,
        mask,
        costs_,
        penalty_,
    )
    selectedCount = int(np.sum(mask))
    if anchor is not None and not bool(mask[anchor]):
        raise RuntimeError("parent-conditioned subpeak DP violated anchor constraint")
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
            "boundary_cost_min": float(np.min(costs_)),
            "boundary_cost_max": float(np.max(costs_)),
            "boundary_penalty": float(boundaryPenalty),
            "min_run_bins": int(minRunBins_),
            "num_runs": int(len(runs)),
            "anchor_index": None if anchor is None else int(anchor),
            "anchor_selected": bool(True if anchor is None else mask[anchor]),
            "anchor_fallback_window": False,
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
    r"""Run anchored local ROCCO refinements inside selected first-pass regions.

    For each eligible parent or child region ``R``, solve an exact local chain
    problem with ``localGamma = 0.25 * gamma``, a hard minimum selected-run
    length, and a mandatory anchor at the strongest local evidence bin. When
    ``nestedRoccoBudgetScale < 1``, translate the scale into a soft per-bin
    penalty rather than a hard local quota. This keeps nested ROCCO as a
    refinement step: children may shrink or split a parent, but every parent
    contributes at least one anchored child.
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

    history: List[Dict[str, float | int | str]] = []
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
        "score_shift": float(selectionPenalty_),
        "subproblem_mode": "parent_conditioned_min_run_dp",
        "subproblem_max_iter": int(subproblemMaxIter),
        "min_region_bins": int(minRegionBins_),
        "min_region_bp": None if minRegionBP_ is None else int(minRegionBP_),
        "min_child_bins": int(minRegionBins_),
        "anchor_policy": "argmax_raw_score_leftmost",
        "diagnostic_detail_path": diagnosticDetailPath_,
        "initial_selected_count": int(np.sum(current)),
        "final_selected_count": int(np.sum(current)),
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
        newMask = np.zeros_like(previous, dtype=bool)
        runs = _selectedRunBounds(previous)
        skippedShort = 0
        iterBudgetScale = budgetScale if iterIdx == 0 else 1.0
        expandedShortChildRuns = 0
        expandedShortChildBins = 0
        emptyLocalSolutions = 0
        budgetFallbackWindows = 0
        budgetConstrainedRegions = 0
        localPenaltyExtraTotal = 0.0
        localPenaltyExtraMax = 0.0
        anchorFallbackWindows = 0
        parentErasureViolations = 0
        anchorSurvivalViolations = 0
        if diagnostics_:
            logger.info(
                "nested ROCCO%s iter=%d start parent_regions=%d selected=%d budget_scale=%.4g local_gamma=%.6g selection_penalty=%.6g",
                label_,
                int(iterIdx + 1),
                int(len(runs)),
                int(np.sum(previous)),
                float(iterBudgetScale),
                float(localGamma),
                float(selectionPenalty_),
            )
        for regionIdx, (start, end) in enumerate(runs, start=1):
            regionLengthBP = _selectedRunLengthBP(start, end, intervals_, ends_)
            if (minRegionBP_ is not None and regionLengthBP < minRegionBP_) or (
                minRegionBP_ is None and (end - start + 1) < minRegionBins_
            ):
                newMask[start : end + 1] = previous[start : end + 1]
                skippedShort += 1
                if diagnostics_:
                    _writeSubproblemDiagnostic(
                        {
                            "event": "subproblem",
                            "status": "skipped_short",
                            "chromosome": diagnosticLabel,
                            "iter": int(iterIdx + 1),
                            "region": int(regionIdx),
                            "bins": int(end - start + 1),
                            "bp": int(regionLengthBP),
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
            localBudgetTarget = _nestedBudgetTargetCount(
                end - start + 1,
                iterBudgetScale,
                localMinChildBins,
            )
            localRawScores = rawScores_[start : end + 1]
            anchorLocal = int(np.argmax(localRawScores))
            localBudgetPenalty = float("nan")
            localPenaltyDetails: Dict[str, float] = {
                "base_penalty": float(selectionPenalty_),
                "extra_penalty": 0.0,
                "positive_score_scale": 0.0,
                "positive_score_spread": 0.0,
                "budget_scale": float(iterBudgetScale),
            }
            if iterBudgetScale < 1.0:
                localSelectionPenalty, localPenaltyDetails = (
                    _nestedSoftSelectionPenalty(
                        localScores,
                        selectionPenalty_,
                        iterBudgetScale,
                    )
                )
                localBudgetPenalty = float(localSelectionPenalty)
                localMask, _localObjective, localDetails = (
                    _solveParentConditionedSubpeaks(
                        localScores,
                        boundaryCosts=localGamma,
                        selectionPenalty=localSelectionPenalty,
                        minRunBins=localMinChildBins,
                        anchorIndex=anchorLocal,
                    )
                )
                localMode = "parent_conditioned_min_run_soft_budget"
                budgetConstrainedRegions += 1
            else:
                localSelectionPenalty, localPenaltyDetails = (
                    _nestedSoftSelectionPenalty(
                        localScores,
                        selectionPenalty_,
                        iterBudgetScale,
                    )
                )
                localBudgetPenalty = float(localSelectionPenalty)
                localMask, _localObjective, localDetails = (
                    _solveParentConditionedSubpeaks(
                        localScores,
                        boundaryCosts=localGamma,
                        selectionPenalty=localSelectionPenalty,
                        minRunBins=localMinChildBins,
                        anchorIndex=anchorLocal,
                    )
                )
                localMode = "parent_conditioned_min_run_dp"
            localEmptySolution = False
            localAnchorFallbackUsed = bool(
                localDetails.get("anchor_fallback_window", False)
            )
            penaltyExtra = float(localPenaltyDetails["extra_penalty"])
            localPenaltyExtraTotal += penaltyExtra
            localPenaltyExtraMax = float(max(localPenaltyExtraMax, penaltyExtra))
            if not bool(np.any(localMask)):
                localEmptySolution = True
                emptyLocalSolutions += 1
                parentErasureViolations += 1
                localMask = previous[start : end + 1].copy()
            if not bool(localMask[anchorLocal]):
                anchorSurvivalViolations += 1
            if localAnchorFallbackUsed:
                anchorFallbackWindows += 1
            newMask[start : end + 1] = localMask
            if diagnostics_:
                if intervals_ is not None and ends_ is not None:
                    regionStartBP = int(intervals_[start])
                    regionEndBP = int(ends_[end])
                else:
                    regionStartBP = int(start)
                    regionEndBP = int(end + 1)
                selectedLocal = int(np.sum(localMask))
                selectedNonPositive = int(np.sum(localMask & (localRawScores <= 0.0)))
                _writeSubproblemDiagnostic(
                    {
                        "event": "subproblem",
                        "status": "solved",
                        "chromosome": diagnosticLabel,
                        "iter": int(iterIdx + 1),
                        "region": int(regionIdx),
                        "mode": str(localMode),
                        "bins": int(end - start + 1),
                        "bp": int(regionLengthBP),
                        "range_start": int(regionStartBP),
                        "range_end": int(regionEndBP),
                        "selected": int(selectedLocal),
                        "selected_possible": int(end - start + 1),
                        "nonpos_selected": int(selectedNonPositive),
                        "min_child_bins": int(localMinChildBins),
                        "budget_target": int(localBudgetTarget),
                        "anchor_local": int(anchorLocal),
                        "anchor_selected": bool(localMask[anchorLocal]),
                        "empty_solution": bool(localEmptySolution),
                        "anchor_fallback": bool(localAnchorFallbackUsed),
                        "objective": float(_localObjective),
                        "penalized": float(localDetails["penalized_objective"]),
                        "solver_penalty": float(localDetails["selection_penalty"]),
                        "budget_penalty": float(localBudgetPenalty),
                        "soft_penalty_extra": float(
                            localPenaltyDetails["extra_penalty"]
                        ),
                        "score_min": float(np.min(localRawScores)),
                        "score_max": float(np.max(localRawScores)),
                        "score_mean": float(np.mean(localRawScores)),
                    }
                )

        newMask &= previous
        jaccard = _maskJaccard(previous, newMask)
        selectedBefore = int(np.sum(previous))
        selectedAfter = int(np.sum(newMask))
        objectivePrevious = _roccoObjectiveForSolution(
            scores_,
            previous.astype(np.uint8),
            parentGamma,
        )
        objectiveAfter = _roccoObjectiveForSolution(
            scores_,
            newMask.astype(np.uint8),
            parentGamma,
        )
        runsAfter = _selectedRunBounds(newMask)
        peakCountMonotonicityViolations = int(max(len(runs) - len(runsAfter), 0))
        coverageExpansionViolations = int(max(selectedAfter - selectedBefore, 0))
        history.append(
            {
                "iter": int(iterIdx + 1),
                "num_parent_peaks": int(len(runs)),
                "num_parent_peaks_after": int(len(runsAfter)),
                "num_input_regions": int(len(runs)),
                "num_skipped_short_regions": int(skippedShort),
                "num_budget_constrained_regions": int(budgetConstrainedRegions),
                "num_empty_local_solutions": int(emptyLocalSolutions),
                "num_budget_fallback_windows": int(budgetFallbackWindows),
                "num_anchor_fallback_windows": int(anchorFallbackWindows),
                "num_parent_erasure_violations": int(parentErasureViolations),
                "num_anchor_survival_violations": int(anchorSurvivalViolations),
                "num_peak_count_monotonicity_violations": int(
                    peakCountMonotonicityViolations
                ),
                "num_coverage_expansion_violations": int(coverageExpansionViolations),
                "num_short_child_runs_expanded": int(expandedShortChildRuns),
                "num_short_child_bins_added": int(expandedShortChildBins),
                "local_penalty_extra_mean": float(
                    localPenaltyExtraTotal / max(len(runs) - skippedShort, 1)
                ),
                "local_penalty_extra_max": float(localPenaltyExtraMax),
                "budget_scale": float(iterBudgetScale),
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
                "nested ROCCO%s iter=%d done selected_before=%d selected_after=%d parent_regions=%d parent_regions_after=%d skipped_short=%d budget_constrained=%d empty_solutions=%d anchor_fallback=%d parent_erasure_violations=%d anchor_survival_violations=%d peak_count_violations=%d coverage_expansion_violations=%d short_child_expanded=%d local_penalty_extra_mean=%.6g objective=%.6g objective_delta=%.6g jaccard=%.6f",
                label_,
                int(iterIdx + 1),
                int(selectedBefore),
                int(selectedAfter),
                int(len(runs)),
                int(len(runsAfter)),
                int(skippedShort),
                int(budgetConstrainedRegions),
                int(emptyLocalSolutions),
                int(anchorFallbackWindows),
                int(parentErasureViolations),
                int(anchorSurvivalViolations),
                int(peakCountMonotonicityViolations),
                int(coverageExpansionViolations),
                int(expandedShortChildRuns),
                float(localPenaltyExtraTotal / max(len(runs) - skippedShort, 1)),
                float(objectiveAfter),
                float(objectiveAfter - objectivePrevious),
                float(jaccard),
            )

        details["completed_iters"] = int(iterIdx + 1)
        current = newMask
        details["final_selected_count"] = int(selectedAfter)

        if np.array_equal(newMask, previous):
            details["stop_reason"] = "mask_equal"
            if diagnostics_:
                logger.info(
                    "nested ROCCO%s stop iter=%d reason=mask_equal",
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
    anchorLocal = int(np.argmax(segScores))
    localMask, _objective, details = _solveParentConditionedSubpeaks(
        segScores,
        boundaryCosts=float(max(float(boundaryCost), 0.0)),
        selectionPenalty=float(selectionPenalty),
        minRunBins=int(max(int(minRunBins), 1)),
        anchorIndex=anchorLocal,
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
        }
    threshold_ = float(threshold)
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
    maxDepth_ = int(max(int(maxDepth), 0))
    counts = {"candidates": 0, "splits": 0, "segments_added": 0, "evaluated": 0}
    objective = float(child.get("subpeak_objective", 0.0))
    boundaryPenalty = float(child.get("subpeak_boundary_penalty", 0.0))

    def _recurse(start: int, end: int, depth: int) -> List[Dict[str, Any]]:
        widthBP = int(max(int(ends_[end]) - int(intervals_[start]), 0))
        candidate = bool(widthBP >= threshold_)
        if not candidate or depth >= maxDepth_:
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
                )
            ]
        gapStart = int(start + int(split["gap_start_local"]))
        gapEnd = int(start + int(split["gap_end_local"]))
        leftEnd = int(gapStart - 1)
        rightStart = int(gapEnd + 1)
        if leftEnd - start + 1 < minBins or end - rightStart + 1 < minBins:
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
                )
            ]
        counts["splits"] += 1
        left = _recurse(start, leftEnd, depth + 1)
        right = _recurse(rightStart, end, depth + 1)
        merged = left + right
        for segment in merged:
            segment["massive_subpeak_cleanup_candidate"] = True
            segment["massive_subpeak_cleanup_applied"] = True
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
    returnExportDetails: bool = False,
) -> (
    Tuple[List[List[str | int | float]], List[Dict[str, Any]]]
    | Tuple[List[List[str | int | float]], List[Dict[str, Any]], Dict[str, Any]]
):
    rowsRaw: List[Dict[str, float | int | str]] = []
    rowsMeta: List[Dict[str, Any]] = []
    state_ = np.asarray(state, dtype=np.float64)
    scores_ = np.asarray(scores, dtype=np.float64)
    uncertainty_: np.ndarray | None = None
    if uncertainty is not None:
        uncertainty_ = np.asarray(uncertainty, dtype=np.float64).ravel()
        if uncertainty_.size != state_.size:
            raise ValueError("`uncertainty` must match `state` length")
    exportFilterUncertaintyMultiplier_ = _validateExportFilterUncertaintyMultiplier(
        exportFilterUncertaintyMultiplier
    )
    n = int(solution.size)
    exportDetails: Dict[str, Any] = {
        "num_candidate_segments": 0,
        "num_segments_dropped_median_signal_local_p": 0,
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
    }
    i = 0
    while i < n:
        if int(solution[i]) <= 0:
            i += 1
            continue
        startIdx = i
        while i + 1 < n and int(solution[i + 1]) > 0:
            i += 1
        endIdx = i

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
                ".",
                ".",
                int(row["peak"]),
            ]
        )
    exportDetails["num_segments_kept"] = int(len(outRows))
    if returnExportDetails:
        return outRows, rowsMeta, exportDetails
    return outRows, rowsMeta


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
    metaPath: str,
    nestedRoccoSubproblemDetailsPath: str | None,
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
                "nested_rocco": nestedSummary,
            }
    blacklist = meta.get("blacklist_filter", {})
    if not isinstance(blacklist, Mapping):
        blacklist = {}
    settings = meta.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}
    inventory = [
        _fileInventoryEntry(outPath, "narrowPeak"),
        _fileInventoryEntry(metaPath, "metadata_json"),
    ]
    if nestedRoccoSubproblemDetailsPath is not None:
        inventory.append(
            _fileInventoryEntry(
                nestedRoccoSubproblemDetailsPath,
                "nested_rocco_subproblems_jsonl",
            )
        )
    summary: Dict[str, Any] = {
        "narrowPeak_path": str(outPath),
        "metadata_json_path": str(metaPath),
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
        },
        "files": inventory,
    }
    return summary


def _logRoccoSummary(summary: Mapping[str, Any]) -> None:
    width = summary.get("median_width_bp")
    widthText = "NA" if width is None else f"{float(width):.1f}"
    logger.info(
        "rocco.summary peaks=%d total_bp=%d width_bp[min/median/max]=%s/%s/%s blacklist_dropped=%d blacklist_kept=%d narrowPeak=%s metadata=%s nested_jsonl=%s",
        int(summary.get("exported_peak_count", 0)),
        int(summary.get("total_peak_bp", 0)),
        "NA" if summary.get("min_width_bp") is None else str(summary["min_width_bp"]),
        widthText,
        "NA" if summary.get("max_width_bp") is None else str(summary["max_width_bp"]),
        int(dict(summary.get("blacklist", {})).get("dropped", 0)),
        int(dict(summary.get("blacklist", {})).get("kept", 0)),
        summary.get("narrowPeak_path"),
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
    randSeed: int = 42,
    outPath: str | None = None,
    metaPath: str | None = None,
    verbose: bool = False,
    stateDiagnosticsByChromosome: Mapping[str, Any] | None = None,
    blacklistBedFile: str | None = None,
    returnSummary: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    r"""Run Consenrich+ROCCO peak caller directly on bedGraphs."""
    exportFilterUncertaintyMultiplier_ = _validateExportFilterUncertaintyMultiplier(
        exportFilterUncertaintyMultiplier
    )
    blacklistByChrom = _readBlacklistIntervalsByChrom(blacklistBedFile)
    chromData = _readAlignedConsenrichBedGraphs(
        stateBedGraphFile,
        uncertaintyBedGraphFile=uncertaintyBedGraphFile,
        chromosomes=chromosomes,
    )
    stateBase = Path(stateBedGraphFile)
    if outPath is None:
        outPath = str(stateBase.with_name(f"{stateBase.stem}_rocco.narrowPeak"))
    if metaPath is None:
        metaPath = f"{outPath}.json"
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
            "num_bootstrap": int(numBootstrap),
            "threshold_z": float(thresholdZ),
            "null_quantile": float(_ROCCO_NULL_QUANTILE),
            "threshold_z_grid": [float(z) for z in _resolveThresholdZGrid(thresholdZ)],
            "nested_rocco_iters": int(max(int(nestedRoccoIters), 0)),
            "nested_rocco_budget_scale": float(
                np.clip(float(nestedRoccoBudgetScale), 0.0, 1.0)
            ),
            "nested_rocco_jaccard": float(_NESTED_ROCCO_JACCARD_DEFAULT),
            "nested_rocco_min_parent_steps": int(_NESTED_ROCCO_MIN_PARENT_STEPS),
            "nested_rocco_min_child_steps": int(_NESTED_ROCCO_MIN_CHILD_STEPS),
            "nested_rocco_subproblem_policy": "parent_conditioned_min_run_dp",
            "nested_rocco_diagnostics": bool(verbose),
            "nested_rocco_subproblem_details": nestedRoccoSubproblemDetailsPath,
            "massive_subpeak_cleanup": bool(massiveSubpeakCleanup),
            "massive_subpeak_cleanup_policy": (
                "robust_log_width_tail_gap_plus_valley_deficit"
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
            "export_filter_uses_uncertainty_bedgraph": True,
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
            thresholdZ=thresholdZ,
            numBootstrap=numBootstrap,
            dependenceSpan=dependenceSpan,
            kernel="bartlett",
            randomSeed=int(randSeed) + chromIndex,
            nullQuantile=_ROCCO_NULL_QUANTILE,
            thresholdZGrid=_ROCCO_BUDGET_Z_GRID,
        )
        scoreTrack = np.asarray(prepared["score_track"], dtype=np.float64)
        budgetRaw, budgetDetails = _estimateBudgetForPreparedROCCOScore(
            prepared,
            statistic="occupancy",
            numBootstrap=numBootstrap,
            dependenceSpan=dependenceSpan,
            thresholdZ=thresholdZ,
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
            "interval_bp": int(np.median(ends - intervals)),
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
            nestedRoccoIters=nestedRoccoIters,
            nestedRoccoBudgetScale=nestedRoccoBudgetScale,
            jaccardThreshold=_NESTED_ROCCO_JACCARD_DEFAULT,
            intervals=intervals,
            ends=ends,
            rawScores=scoreTrack,
            minRegionBP=int(
                _NESTED_ROCCO_MIN_PARENT_STEPS * max(int(work["interval_bp"]), 1)
            ),
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
        solveDetails["nested_rocco_iters"] = int(max(int(nestedRoccoIters), 0))
        solveDetails["nested_rocco_budget_scale"] = float(
            np.clip(float(nestedRoccoBudgetScale), 0.0, 1.0)
        )
        solveDetails["nested_rocco_stop_reason"] = str(nestedDetails["stop_reason"])
        exportTrimScoreFloor = 0.0
        rows, peakMeta, exportDetails = _solutionToChromNarrowPeakRows(
            str(chromosome),
            intervals,
            ends,
            state,
            np.asarray(scoreTrack, dtype=np.float64),
            solution,
            prefix="consenrichROCCO",
            nullScale=float(nullScale),
            uncertainty=uncertainty,
            trimScoreFloor=float(exportTrimScoreFloor),
            subpeakSelectionPenalty=float(solveDetails["selection_penalty"]),
            subpeakBoundaryCost=float(0.25 * float(work["gamma"])),
            minSubpeakBins=int(_NESTED_ROCCO_MIN_CHILD_STEPS),
            exportFilterUncertaintyMultiplier=float(exportFilterUncertaintyMultiplier_),
            returnExportDetails=True,
        )
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
            "first_pass_solution": firstPassSolution,
            "solution": solution,
            "export_trim_score_floor": float(exportTrimScoreFloor),
            "initial_rows": rows,
            "initial_peak_meta": peakMeta,
            "initial_export_details": exportDetails,
            "work": work,
        }

    massiveWidthPolicy = _learnMassiveSubpeakWidthPolicy(
        initialPeakWidthsBP,
        enabled=bool(massiveSubpeakCleanup),
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
                minSubpeakBins=int(_NESTED_ROCCO_MIN_CHILD_STEPS),
                massiveSubpeakCleanup=True,
                massiveSubpeakWidthPolicy=massiveWidthPolicy,
                massiveSubpeakSplitQuantile=float(_MASSIVE_SUBPEAK_SPLIT_QUANTILE),
                massiveSubpeakSplitZ=float(_MASSIVE_SUBPEAK_SPLIT_Z),
                exportFilterUncertaintyMultiplier=float(
                    exportFilterUncertaintyMultiplier_
                ),
                returnExportDetails=True,
            )
        else:
            rows = list(result["initial_rows"])
            peakMeta = list(result["initial_peak_meta"])
            exportDetails = dict(result["initial_export_details"])
            exportDetails["massive_subpeak_width_policy"] = dict(massiveWidthPolicy)
        rows, peakMeta, blacklistDropped = _filterNarrowPeakRowsByBlacklist(
            rows,
            peakMeta,
            blacklistByChrom,
        )
        blacklistDroppedTotal += int(blacklistDropped)
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
        allRows.extend(rows)

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
            "budget_details": dict(result["budget_details"]),
            "gamma_details": dict(work["gamma_details"]),
            "solve_details": solveDetails,
            "nested_rocco_details": result["nested_details"],
            "export_trim_score_floor": float(exportTrimScoreFloor),
            "export_details": exportDetails,
            "peak_details": peakMeta,
        }

    allRows.sort(key=lambda row: (str(row[0]), int(row[1]), int(row[2])))
    meta["blacklist_filter"] = {
        "blacklist_bed": None if blacklistBedFile is None else str(blacklistBedFile),
        "policy": "drop_any_overlap",
        "dropped": int(blacklistDroppedTotal),
        "kept": int(len(allRows)),
    }
    with open(outPath, "w", encoding="utf-8") as handle:
        for row in allRows:
            handle.write("\t".join(map(str, row)) + "\n")

    with open(metaPath, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    summary = _buildRoccoSummary(
        outPath=str(outPath),
        metaPath=str(metaPath),
        nestedRoccoSubproblemDetailsPath=nestedRoccoSubproblemDetailsPath,
        rows=allRows,
        meta=meta,
    )
    _logRoccoSummary(summary)
    _logOutputInventory(summary)
    if returnSummary:
        return str(outPath), summary
    return str(outPath)
