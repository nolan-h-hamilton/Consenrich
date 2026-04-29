# -*- coding: utf-8 -*-
r"""Peak-calling helpers for ROCCO-style segmentation from Consenrich tracks."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import signal, stats

from . import cconsenrich
from . import core

logger = logging.getLogger(__name__)

_TINY = float(np.finfo(np.float64).tiny)
_ROCCO_BUDGET_MIN = 0.001
_ROCCO_BUDGET_MAX = 0.10
_ROCCO_NULL_QUANTILE = 0.80
_ROCCO_THRESHOLD_Z_DEFAULT = 2.0
_ROCCO_NUM_BOOTSTRAP_DEFAULT = 128
_ROCCO_BUDGET_Z_GRID = (1.5, 2.0, 2.5, 3.0)
_ROCCO_MAX_ITER_DEFAULT = 60
_NESTED_ROCCO_ITERS_DEFAULT = 3
_NESTED_ROCCO_JACCARD_DEFAULT = 0.999
_NESTED_ROCCO_MIN_PARENT_STEPS = 5
_NESTED_ROCCO_MIN_CHILD_STEPS = _NESTED_ROCCO_MIN_PARENT_STEPS
_NESTED_ROCCO_BUDGET_SCALE_DEFAULT = 0.5
_NESTED_ROCCO_SUBPROBLEM_MAX_ITER = 5
_EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER = 2.5


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


def studentizedScoreTrack(
    state: npt.ArrayLike,
    uncertainty: npt.ArrayLike | None = None,
    tau0: float = 1.0,
    returnDetails: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float | str]]:
    r"""Build a studentized score track from Consenrich outputs."""
    state_ = _asFloatVector("state", state)
    n = int(state_.size)
    tau0_ = float(max(tau0, 0.0))

    seMode = "identity"
    if uncertainty is None:
        effectiveSE = np.ones(n, dtype=np.float64)
    else:
        effectiveSE = _asFloatVector("uncertainty", uncertainty)
        if effectiveSE.size != n:
            raise ValueError("`uncertainty` must match `state` length")
        seMode = "uncertainty"

    effectiveSE = np.maximum(effectiveSE, 0.0)
    denom = np.sqrt(effectiveSE * effectiveSE + tau0_ * tau0_)
    denom = np.maximum(denom, 1.0e-6)
    scoreTrack = state_ / denom
    if not returnDetails:
        return scoreTrack

    details: Dict[str, float | str] = {
        "se_mode": str(seMode),
        "tau0": float(tau0_),
        "state_abs_median": float(np.median(np.abs(state_))),
        "uncertainty_median": float(np.median(effectiveSE)),
        "uncertainty_min": float(np.min(effectiveSE)),
        "uncertainty_max": float(np.max(effectiveSE)),
        "score_denom_median": float(np.median(denom)),
        "score_denom_min": float(np.min(denom)),
        "score_denom_max": float(np.max(denom)),
    }
    return scoreTrack, details


def shrinkageScoreTrack(
    state: npt.ArrayLike,
    uncertainty: npt.ArrayLike | None = None,
    nullCenter: float = 0.0,
    tau0: float = 1.0,
    returnDetails: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float | str]]:
    r"""Build a posterior-mean-style shrinkage score track for ROCCO."""
    state_ = _asFloatVector("state", state)
    n = int(state_.size)
    tau0_ = float(max(tau0, 0.0))
    nullCenter_ = float(nullCenter)
    priorVariance = float(tau0_ * tau0_)

    seMode = "identity"
    if uncertainty is None:
        effectiveSE = np.zeros(n, dtype=np.float64)
        shrinkWeights = np.ones(n, dtype=np.float64)
    else:
        effectiveSE = _asFloatVector("uncertainty", uncertainty)
        if effectiveSE.size != n:
            raise ValueError("`uncertainty` must match `state` length")
        effectiveSE = np.maximum(effectiveSE, 0.0)
        seMode = "uncertainty"
        if priorVariance <= 0.0:
            shrinkWeights = np.zeros(n, dtype=np.float64)
        else:
            shrinkWeights = priorVariance / np.maximum(
                priorVariance + effectiveSE * effectiveSE,
                1.0e-12,
            )

    centeredState = state_ - nullCenter_
    scoreTrack = shrinkWeights * centeredState
    if not returnDetails:
        return scoreTrack

    details: Dict[str, float | str] = {
        "score_mode": "posterior_mean_shrinkage",
        "se_mode": str(seMode),
        "tau0": float(tau0_),
        "prior_variance": float(priorVariance),
        "null_center_input": float(nullCenter_),
        "centered_state_abs_median": float(np.median(np.abs(centeredState))),
        "uncertainty_median": float(np.median(effectiveSE)),
        "uncertainty_min": float(np.min(effectiveSE)),
        "uncertainty_max": float(np.max(effectiveSE)),
        "shrink_weight_median": float(np.median(shrinkWeights)),
        "shrink_weight_min": float(np.min(shrinkWeights)),
        "shrink_weight_max": float(np.max(shrinkWeights)),
    }
    return scoreTrack, details


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
                "budget_occupancy_raw": float(
                    np.clip(
                        float(metrics["observed_tail_occupancy"]) - nullOccCal, 0.0, 1.0
                    )
                ),
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
    tau0: float = 1.0,
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
    scoreMeta["tau0"] = float(max(float(tau0), 0.0))
    scoreMeta["tau0_used"] = False
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
            contextSize, contextLower, contextUpper = core.getContextSize(
                positiveVals,
                minSpan=3,
                maxSpan=min(64, max(12, n // 8)),
            )
            point = max(int(contextSize), 2)
            lower = max(min(int(contextLower), point), 2)
            upper = max(int(contextUpper), point)
            return {
                "point": int(point),
                "lower": int(lower),
                "upper": int(upper),
                "method": "getContextSize",
            }
    except Exception as ex:
        logger.info("getContextSize fallback for ROCCO budget span: %s", ex)

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
    tau0: float = 1.0,
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
        tau0=tau0,
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
    statistic: str = "integrated",
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    dependenceSpan: int | None = None,
    kernel: str = "bartlett",
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    bulkQuantile: float = 0.60,
    randomSeed: int = 0,
    tau0: float = 1.0,
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
        "tau0": float(max(tau0, 0.0)),
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
    tau0: float = 1.0,
    statistic: str = "integrated",
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
        tau0=tau0,
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
        tau0=tau0,
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


def _solveMinRunPenalizedChainROCCO(
    scores: np.ndarray,
    gamma: float,
    selectionPenalty: float,
    minRunBins: int,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    scores_ = np.asarray(scores, dtype=np.float64)
    if scores_.ndim != 1 or scores_.size == 0:
        raise ValueError("`scores` must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(scores_)):
        raise ValueError("`scores` contains non-finite values")
    gamma_ = float(max(float(gamma), 0.0))
    penalty_ = float(selectionPenalty)
    if not np.isfinite(penalty_):
        raise ValueError("`selectionPenalty` must be finite")

    n = int(scores_.size)
    minRunBins_ = int(min(max(int(minRunBins), 1), n))
    numStates = int(minRunBins_ + 1)
    negInf = -math.inf
    eps = 1.0e-12
    largeCount = n + 1

    prevValues = np.full(numStates, negInf, dtype=np.float64)
    prevCounts = np.full(numStates, largeCount, dtype=np.int64)
    prevValues[0] = 0.0
    prevCounts[0] = 0
    back = np.full((n, numStates), -1, dtype=np.int16)

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

    for i in range(n):
        adjustedScore = float(scores_[i] - penalty_)
        newValues = np.full(numStates, negInf, dtype=np.float64)
        newCounts = np.full(numStates, largeCount, dtype=np.int64)
        transitionCost = 0.0 if i == 0 else gamma_

        if np.isfinite(prevValues[0]):
            newValues[0] = prevValues[0]
            newCounts[0] = prevCounts[0]
            back[i, 0] = 0

            candidateValue = prevValues[0] - transitionCost + adjustedScore
            candidateCount = int(prevCounts[0] + 1)
            if _better(candidateValue, candidateCount, newValues[1], newCounts[1]):
                newValues[1] = candidateValue
                newCounts[1] = candidateCount
                back[i, 1] = 0

        if np.isfinite(prevValues[minRunBins_]):
            candidateValue = prevValues[minRunBins_] - transitionCost
            candidateCount = int(prevCounts[minRunBins_])
            if _better(candidateValue, candidateCount, newValues[0], newCounts[0]):
                newValues[0] = candidateValue
                newCounts[0] = candidateCount
                back[i, 0] = minRunBins_

        for state in range(1, minRunBins_):
            if not np.isfinite(prevValues[state]):
                continue
            nextState = int(state + 1)
            candidateValue = prevValues[state] + adjustedScore
            candidateCount = int(prevCounts[state] + 1)
            if _better(
                candidateValue,
                candidateCount,
                newValues[nextState],
                newCounts[nextState],
            ):
                newValues[nextState] = candidateValue
                newCounts[nextState] = candidateCount
                back[i, nextState] = state

        if np.isfinite(prevValues[minRunBins_]):
            candidateValue = prevValues[minRunBins_] + adjustedScore
            candidateCount = int(prevCounts[minRunBins_] + 1)
            if _better(
                candidateValue,
                candidateCount,
                newValues[minRunBins_],
                newCounts[minRunBins_],
            ):
                newValues[minRunBins_] = candidateValue
                newCounts[minRunBins_] = candidateCount
                back[i, minRunBins_] = minRunBins_

        prevValues = newValues
        prevCounts = newCounts

    bestState = 0
    bestValue = float(prevValues[0])
    bestCount = int(prevCounts[0])
    if _better(
        float(prevValues[minRunBins_]),
        int(prevCounts[minRunBins_]),
        bestValue,
        bestCount,
    ):
        bestState = int(minRunBins_)
        bestValue = float(prevValues[minRunBins_])
        bestCount = int(prevCounts[minRunBins_])

    mask = np.zeros(n, dtype=bool)
    state = int(bestState)
    for i in range(n - 1, -1, -1):
        if state > 0:
            mask[i] = True
        prevState = int(back[i, state])
        if prevState < 0:
            break
        state = prevState

    solution = mask.astype(np.uint8)
    objective = _roccoObjectiveForSolution(scores_, solution, gamma_)
    penalizedObjective = float(objective - penalty_ * float(bestCount))
    runs = _selectedRunBounds(mask)
    return (
        mask,
        float(objective),
        {
            "mode": "min_run_penalty",
            "penalized_objective": float(penalizedObjective),
            "selected_count": int(bestCount),
            "selected_fraction": float(bestCount / max(n, 1)),
            "selection_penalty": float(penalty_),
            "gamma": float(gamma_),
            "min_run_bins": int(minRunBins_),
            "num_runs": int(len(runs)),
        },
    )


def _anchorFallbackWindowMask(
    scores: np.ndarray,
    anchorIndex: int,
    minRunBins: int,
) -> np.ndarray:
    scores_ = np.asarray(scores, dtype=np.float64)
    n = int(scores_.size)
    mask = np.zeros(n, dtype=bool)
    if n <= 0:
        return mask
    anchor = int(np.clip(int(anchorIndex), 0, n - 1))
    width = int(min(max(int(minRunBins), 1), n))
    leftMin = int(max(0, anchor - width + 1))
    leftMax = int(min(anchor, n - width))
    bestStart = leftMin
    bestScore = -math.inf
    cumsum = np.concatenate(([0.0], np.cumsum(scores_, dtype=np.float64)))
    for start in range(leftMin, leftMax + 1):
        score = float(cumsum[start + width] - cumsum[start])
        if score > bestScore:
            bestScore = score
            bestStart = int(start)
    mask[bestStart : bestStart + width] = True
    return mask


def _solveAnchoredMinRunPenalizedChainROCCO(
    scores: np.ndarray,
    gamma: float,
    selectionPenalty: float,
    minRunBins: int,
    anchorIndex: int,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    scores_ = np.asarray(scores, dtype=np.float64)
    if scores_.ndim != 1 or scores_.size == 0:
        raise ValueError("`scores` must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(scores_)):
        raise ValueError("`scores` contains non-finite values")
    gamma_ = float(max(float(gamma), 0.0))
    penalty_ = float(selectionPenalty)
    if not np.isfinite(penalty_):
        raise ValueError("`selectionPenalty` must be finite")

    n = int(scores_.size)
    anchor = int(np.clip(int(anchorIndex), 0, n - 1))
    minRunBins_ = int(min(max(int(minRunBins), 1), n))
    numStates = int(minRunBins_ + 1)
    negInf = -math.inf
    eps = 1.0e-12
    largeCount = n + 1

    prevValues = np.full((2, numStates), negInf, dtype=np.float64)
    prevCounts = np.full((2, numStates), largeCount, dtype=np.int64)
    prevValues[0, 0] = 0.0
    prevCounts[0, 0] = 0
    backState = np.full((n, 2, numStates), -1, dtype=np.int16)
    backSeen = np.full((n, 2, numStates), -1, dtype=np.int8)

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
        newSeen: int,
        newState: int,
        value: float,
        count: int,
        prevSeen: int,
        prevState: int,
        i: int,
    ) -> None:
        if _better(
            float(value),
            int(count),
            float(values[newSeen, newState]),
            int(counts[newSeen, newState]),
        ):
            values[newSeen, newState] = float(value)
            counts[newSeen, newState] = int(count)
            backSeen[i, newSeen, newState] = int(prevSeen)
            backState[i, newSeen, newState] = int(prevState)

    for i in range(n):
        adjustedScore = float(scores_[i] - penalty_)
        newValues = np.full((2, numStates), negInf, dtype=np.float64)
        newCounts = np.full((2, numStates), largeCount, dtype=np.int64)
        transitionCost = 0.0 if i == 0 else gamma_
        forceOn = bool(i == anchor)

        for seen in (0, 1):
            # Choose x_i = 0. A run may end only after reaching minRunBins_.
            if not forceOn:
                if np.isfinite(prevValues[seen, 0]):
                    _update(
                        newValues,
                        newCounts,
                        seen,
                        0,
                        float(prevValues[seen, 0]),
                        int(prevCounts[seen, 0]),
                        seen,
                        0,
                        i,
                    )
                if np.isfinite(prevValues[seen, minRunBins_]):
                    _update(
                        newValues,
                        newCounts,
                        seen,
                        0,
                        float(prevValues[seen, minRunBins_] - transitionCost),
                        int(prevCounts[seen, minRunBins_]),
                        seen,
                        minRunBins_,
                        i,
                    )

            # Choose x_i = 1.
            newSeen = int(seen or forceOn)
            if np.isfinite(prevValues[seen, 0]):
                _update(
                    newValues,
                    newCounts,
                    newSeen,
                    1,
                    float(prevValues[seen, 0] - transitionCost + adjustedScore),
                    int(prevCounts[seen, 0] + 1),
                    seen,
                    0,
                    i,
                )
            for state in range(1, minRunBins_):
                if not np.isfinite(prevValues[seen, state]):
                    continue
                _update(
                    newValues,
                    newCounts,
                    newSeen,
                    state + 1,
                    float(prevValues[seen, state] + adjustedScore),
                    int(prevCounts[seen, state] + 1),
                    seen,
                    state,
                    i,
                )
            if np.isfinite(prevValues[seen, minRunBins_]):
                _update(
                    newValues,
                    newCounts,
                    newSeen,
                    minRunBins_,
                    float(prevValues[seen, minRunBins_] + adjustedScore),
                    int(prevCounts[seen, minRunBins_] + 1),
                    seen,
                    minRunBins_,
                    i,
                )

        prevValues = newValues
        prevCounts = newCounts

    finalCandidates = [
        (float(prevValues[1, 0]), int(prevCounts[1, 0]), 1, 0),
        (
            float(prevValues[1, minRunBins_]),
            int(prevCounts[1, minRunBins_]),
            1,
            minRunBins_,
        ),
    ]
    bestValue, bestCount, bestSeen, bestState = max(
        finalCandidates,
        key=lambda item: (item[0], -item[1]),
    )
    fallbackUsed = False
    if not np.isfinite(bestValue):
        mask = _anchorFallbackWindowMask(scores_, anchor, minRunBins_)
        fallbackUsed = True
    else:
        mask = np.zeros(n, dtype=bool)
        seen = int(bestSeen)
        state = int(bestState)
        for i in range(n - 1, -1, -1):
            if state > 0:
                mask[i] = True
            prevState = int(backState[i, seen, state])
            prevSeen = int(backSeen[i, seen, state])
            if prevState < 0 or prevSeen < 0:
                break
            state = prevState
            seen = prevSeen

    if not bool(mask[anchor]):
        fallbackMask = _anchorFallbackWindowMask(scores_, anchor, minRunBins_)
        mask |= fallbackMask
        fallbackUsed = True

    solution = mask.astype(np.uint8)
    objective = _roccoObjectiveForSolution(scores_, solution, gamma_)
    selectedCount = int(np.sum(solution))
    penalizedObjective = float(objective - penalty_ * float(selectedCount))
    runs = _selectedRunBounds(mask)
    return (
        mask,
        float(objective),
        {
            "mode": "anchored_min_run_penalty",
            "penalized_objective": float(penalizedObjective),
            "selected_count": int(selectedCount),
            "selected_fraction": float(selectedCount / max(n, 1)),
            "selection_penalty": float(penalty_),
            "gamma": float(gamma_),
            "min_run_bins": int(minRunBins_),
            "num_runs": int(len(runs)),
            "anchor_index": int(anchor),
            "anchor_selected": bool(mask[anchor]),
            "anchor_fallback_window": bool(fallbackUsed),
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
    subproblemMaxIter = int(_NESTED_ROCCO_SUBPROBLEM_MAX_ITER)
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
        "subproblem_mode": "anchored_min_run_penalty",
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
                    _solveAnchoredMinRunPenalizedChainROCCO(
                        localScores,
                        gamma=localGamma,
                        selectionPenalty=localSelectionPenalty,
                        minRunBins=localMinChildBins,
                        anchorIndex=anchorLocal,
                    )
                )
                localMode = "anchored_min_run_soft_budget"
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
                    _solveAnchoredMinRunPenalizedChainROCCO(
                        localScores,
                        gamma=localGamma,
                        selectionPenalty=localSelectionPenalty,
                        minRunBins=localMinChildBins,
                        anchorIndex=anchorLocal,
                    )
                )
                localMode = "anchored_min_run_penalty"
            localEmptySolution = False
            localAnchorFallbackUsed = bool(
                localDetails.get("anchor_fallback_window", False)
            )
            expandedRuns = 0
            expandedBins = 0
            penaltyExtra = float(localPenaltyDetails["extra_penalty"])
            localPenaltyExtraTotal += penaltyExtra
            localPenaltyExtraMax = float(max(localPenaltyExtraMax, penaltyExtra))
            if not bool(np.any(localMask)):
                localEmptySolution = True
                emptyLocalSolutions += 1
                parentErasureViolations += 1
                localMask = _anchorFallbackWindowMask(
                    localScores,
                    anchorLocal,
                    localMinChildBins,
                )
                localAnchorFallbackUsed = True
            if not bool(localMask[anchorLocal]):
                anchorSurvivalViolations += 1
                localMask = np.asarray(localMask, dtype=bool)
                localMask |= _anchorFallbackWindowMask(
                    localScores,
                    anchorLocal,
                    localMinChildBins,
                )
                localAnchorFallbackUsed = True
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
                        "expanded_short_runs": int(
                            expandedRuns if not localEmptySolution else 0
                        ),
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


def _selectSubpeakSummits(
    segmentState: np.ndarray,
    nullScale: float,
    contextSpan: int,
) -> np.ndarray:
    segState = np.asarray(segmentState, dtype=np.float64)
    n = int(segState.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    globalPeak = int(np.argmax(segState))
    if n < 4:
        return np.asarray([globalPeak], dtype=np.int64)

    candidatePeaks, _ = signal.find_peaks(
        segState,
    )
    if candidatePeaks.size == 0:
        return np.asarray([globalPeak], dtype=np.int64)

    blockMaxExcess = float(max(np.max(segState) - np.min(segState), 0.0))
    nullProminence = float(max(float(nullScale), 1.0e-6))
    prominenceThreshold = float(
        max(nullProminence, min(0.25 * blockMaxExcess, 4.0 * nullProminence))
    )
    if candidatePeaks.size > 0:
        prominences = signal.peak_prominences(segState, candidatePeaks)[0]
        candidatePeaks = candidatePeaks[prominences >= prominenceThreshold]
    if candidatePeaks.size == 0:
        return np.asarray([globalPeak], dtype=np.int64)
    if globalPeak not in set(candidatePeaks.tolist()):
        candidatePeaks = np.sort(
            np.unique(
                np.concatenate(
                    [
                        np.asarray(candidatePeaks, dtype=np.int64),
                        np.asarray([globalPeak], dtype=np.int64),
                    ]
                )
            )
        )

    kept = sorted(set(int(idx) for idx in candidatePeaks.tolist()))

    if len(kept) <= 1:
        return np.asarray([globalPeak], dtype=np.int64)

    while len(kept) > 1:
        removed = False
        for i in range(len(kept) - 1):
            left = int(kept[i])
            right = int(kept[i + 1])
            valley = left + int(np.argmin(segState[left : right + 1]))
            leftProm = float(segState[left] - segState[valley])
            rightProm = float(segState[right] - segState[valley])
            if min(leftProm, rightProm) >= prominenceThreshold:
                continue
            if float(segState[left]) >= float(segState[right]):
                drop = right
            else:
                drop = left
            kept = [idx for idx in kept if idx != drop]
            removed = True
            break
        if not removed:
            break

    if len(kept) == 0:
        kept = [globalPeak]
    if globalPeak not in kept:
        kept.append(globalPeak)
    kept = sorted(set(kept))
    return np.asarray(kept, dtype=np.int64)


def _splitSelectedSegment(
    segmentState: np.ndarray,
    startIdx: int,
    endIdx: int,
    nullScale: float,
    contextSpan: int,
) -> List[Dict[str, int | float | bool]]:
    segState = np.asarray(segmentState, dtype=np.float64)
    summits = _selectSubpeakSummits(
        segState,
        nullScale=float(nullScale),
        contextSpan=int(contextSpan),
    )
    if summits.size <= 1:
        summitLocal = int(summits[0]) if summits.size == 1 else int(np.argmax(segState))
        return [
            {
                "start_idx": int(startIdx),
                "end_idx": int(endIdx),
                "summit_idx": int(startIdx + summitLocal),
                "segment_length_bins": int(max(endIdx - startIdx + 1, 0)),
                "num_subpeaks": 1,
                "split_from_parent": False,
            }
        ]

    splitPoints: List[int] = []
    sortedSummits = np.sort(summits.astype(np.int64, copy=False))
    for left, right in zip(sortedSummits[:-1], sortedSummits[1:]):
        valleyLocal = int(left + np.argmin(segState[left : right + 1]))
        splitPoints.append(valleyLocal)

    localRanges: List[Tuple[int, int]] = []
    localStart = 0
    for splitLocal in splitPoints:
        splitLocal_ = int(splitLocal)
        if splitLocal_ > localStart:
            localRanges.append((localStart, splitLocal_ - 1))
        localStart = int(splitLocal) + 1
    if localStart <= int(segState.size - 1):
        localRanges.append((localStart, int(segState.size - 1)))

    out: List[Dict[str, int | float | bool]] = []
    numSubpeaks = int(len(localRanges))
    for localLeft, localRight in localRanges:
        childState = segState[localLeft : localRight + 1]
        summitLocal = int(localLeft + np.argmax(childState))
        out.append(
            {
                "start_idx": int(startIdx + localLeft),
                "end_idx": int(startIdx + localRight),
                "summit_idx": int(startIdx + summitLocal),
                "segment_length_bins": int(localRight - localLeft + 1),
                "num_subpeaks": int(numSubpeaks),
                "split_from_parent": True,
            }
        )
    return out


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
    contextSpan: int,
    uncertainty: np.ndarray | None = None,
    splitSubpeaks: bool = True,
    trimScoreFloor: float | None = 0.0,
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
            childSegments = _splitSelectedSegment(
                segState,
                startIdx=startIdx,
                endIdx=endIdx,
                nullScale=float(nullScale),
                contextSpan=int(contextSpan),
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
                }
            ]
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


def solveRocco(
    stateBedGraphFile: str,
    uncertaintyBedGraphFile: str | None = None,
    chromosomes: Iterable[str] | None = None,
    tau0: float = 1.0,
    numBootstrap: int = _ROCCO_NUM_BOOTSTRAP_DEFAULT,
    thresholdZ: float = _ROCCO_THRESHOLD_Z_DEFAULT,
    dependenceSpan: int | None = None,
    gamma: float | None = 0.5,
    selectionPenalty: float | None = None,
    gammaScale: float = 0.5,
    nestedRoccoIters: int = _NESTED_ROCCO_ITERS_DEFAULT,
    nestedRoccoBudgetScale: float = _NESTED_ROCCO_BUDGET_SCALE_DEFAULT,
    exportFilterUncertaintyMultiplier: float = (
        _EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER
    ),
    randSeed: int = 42,
    outPath: str | None = None,
    metaPath: str | None = None,
    verbose: bool = False,
) -> str:
    r"""Run Consenrich+ROCCO peak caller directly on bedGraphs."""
    exportFilterUncertaintyMultiplier_ = _validateExportFilterUncertaintyMultiplier(
        exportFilterUncertaintyMultiplier
    )
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
            "tau0": float(tau0),
            "budget_method": "dwb_integrated_excess_tail",
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
            "nested_rocco_subproblem_policy": "anchored_exact_min_run_soft_budget_penalty",
            "nested_rocco_diagnostics": bool(verbose),
            "nested_rocco_subproblem_details": nestedRoccoSubproblemDetailsPath,
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
            tau0=tau0,
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
            statistic="integrated",
            numBootstrap=numBootstrap,
            dependenceSpan=dependenceSpan,
            thresholdZ=thresholdZ,
            randomSeed=int(randSeed) + chromIndex,
            tau0=tau0,
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
            contextSpan=int(budgetDetails.get("dependence_span", 4)),
            uncertainty=uncertainty,
            trimScoreFloor=float(exportTrimScoreFloor),
            exportFilterUncertaintyMultiplier=float(exportFilterUncertaintyMultiplier_),
            returnExportDetails=True,
        )
        allRows.extend(rows)

        meta["chromosomes"][str(chromosome)] = {
            "n_loci": int(state.size),
            "interval_bp": int(work["interval_bp"]),
            "budget": float(budget),
            "objective": float(finalObjective),
            "first_pass_objective": float(objective),
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
            "budget_details": budgetDetails,
            "gamma_details": dict(work["gamma_details"]),
            "solve_details": solveDetails,
            "nested_rocco_details": nestedDetails,
            "export_trim_score_floor": float(exportTrimScoreFloor),
            "export_details": exportDetails,
            "peak_details": peakMeta,
        }

    allRows.sort(key=lambda row: (str(row[0]), int(row[1]), int(row[2])))
    with open(outPath, "w", encoding="utf-8") as handle:
        for row in allRows:
            handle.write("\t".join(map(str, row)) + "\n")

    with open(metaPath, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    return str(outPath)
