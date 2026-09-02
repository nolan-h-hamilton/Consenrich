"""(Experimental) State-uncertainty calibration helpers."""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from . import core
from . import diagnostics
from . import segshrink
from . import cuncertainty as _cuncertainty
from . import _logging as _logging_utils
from ._normalization import weighted_quantile


logger = logging.getLogger(__name__)


TARGET_CALIBRATION_BLOCK_SPLIT_SEED_OFFSET = 20_000
TARGET_CALIBRATION_FRACTION = 0.5
_TARGET_BOUND_SCOPE = "chromosome_selected_target_conditional_exchangeability"
_COVERAGE_ESTIMAND = "delete_block_target_signal_perturbation"
_COVERAGE_SCOPE = "all_valid_rows_reuse_diagnostic"
_FIT_COVERAGE_SCOPE = "factor_fit_rows_reuse_diagnostic"
_TARGET_ROLE_SELECTED = "selected"
_TARGET_ROLE_DESCRIPTIVE = "descriptive"
_PERTURBATION_SCORE_DEFINITION = "masked_minus_full_target_signal_over_delta_sd"
_TARGET_PERTURBATION_SCORE_DEFINITION = (
    "max_abs_masked_minus_full_target_signal_over_delta_sd_by_block"
)
_CALIBRATION_REPLAY_KEYS = (
    "residual",
    "pDelta",
    "intervalIndex",
    "fitRows",
    "targetBlockMask",
    "deletedObservationAll",
    "coverageCodeAll",
    "coverageCodeFit",
    "summaryDecile",
)
_CALIBRATION_REPLAY_DTYPES = {
    "residual": np.dtype(np.float64),
    "pDelta": np.dtype(np.float64),
    "intervalIndex": np.dtype(np.int64),
    "fitRows": np.dtype(np.int64),
    "targetBlockMask": np.dtype(np.uint8),
    "deletedObservationAll": np.dtype(np.int64),
    "coverageCodeAll": np.dtype(np.int32),
    "coverageCodeFit": np.dtype(np.int32),
    "summaryDecile": np.dtype(np.int32),
}
_COVERAGE_CODE_NAMES = {
    0: "signal_abs_q00_20",
    1: "signal_abs_q20_40",
    2: "signal_abs_q40_60",
    3: "signal_abs_q60_80",
    4: "signal_abs_q80_100",
}
_DELETE_BLOCK_SOURCE_INVALID = np.uint8(0)
_DELETE_BLOCK_SOURCE_COVARIANCE_DIFFERENCE = np.uint8(1)
_DELETE_BLOCK_SOURCE_HELDOUT_INFORMATION = np.uint8(2)
_DELETE_BLOCK_SOURCE_HELDOUT_INFORMATION_FALLBACK = np.uint8(3)
DELETE_BLOCK_VARIANCE_SOURCE_LABELS = np.asarray(
    (
        "invalid",
        "covariance_difference",
        "heldout_information",
        "heldout_information_fallback",
    ),
    dtype=object,
)
_DELETE_BLOCK_REASON_OTHER = np.uint8(0)
_DELETE_BLOCK_REASON_NO_DELETED_INFORMATION = np.uint8(1)
_DELETE_BLOCK_REASON_H_OUT_OF_BOUNDS = np.uint8(2)
_DELETE_BLOCK_REASON_COVARIANCE_DELTA_NONPOSITIVE = np.uint8(3)
_DELETE_BLOCK_REASON_INFORMATION_DELTA_INVALID = np.uint8(4)
_DELETE_BLOCK_REASON_NONFINITE_STATE_DELTA = np.uint8(5)
_DELETE_BLOCK_REASON_NONFINITE_COVARIANCE = np.uint8(6)
_DELETE_BLOCK_REASON_VALID = np.uint8(7)
DELETE_BLOCK_INVALID_REASON_LABELS = np.asarray(
    (
        "other",
        "no_deleted_information",
        "h_out_of_bounds",
        "covariance_delta_nonpositive",
        "information_delta_invalid",
        "nonfinite_state_delta",
        "nonfinite_covariance",
        "valid",
    ),
    dtype=object,
)
_DELETE_BLOCK_INVALID_REASON_COUNT_CODES = (
    (_DELETE_BLOCK_REASON_NO_DELETED_INFORMATION, "no_deleted_information"),
    (_DELETE_BLOCK_REASON_H_OUT_OF_BOUNDS, "h_out_of_bounds"),
    (_DELETE_BLOCK_REASON_COVARIANCE_DELTA_NONPOSITIVE, "covariance_delta_nonpositive"),
    (_DELETE_BLOCK_REASON_INFORMATION_DELTA_INVALID, "information_delta_invalid"),
    (_DELETE_BLOCK_REASON_NONFINITE_STATE_DELTA, "nonfinite_state_delta"),
    (_DELETE_BLOCK_REASON_NONFINITE_COVARIANCE, "nonfinite_covariance"),
    (_DELETE_BLOCK_REASON_OTHER, "other"),
)

DELETE_BLOCK_CALIBRATION_LOG_COLUMNS = [
    "record_type",
    "event",
    "chromosome",
    "fold",
    "interval_index",
    "block_index",
    "blockIDX",
    "chrom_start",
    "uncertainty_decile",
    "high_signal",
    "stratum",
    "target",
    "target_role",
    "alpha",
    "delta",
    "q",
    "q_source",
    "k",
    "tail_probability",
    "finite_bound",
    "bound_available",
    "bound_scope",
    "reason",
    "n",
    "coverage_before",
    "coverage_after",
    "coverage_estimand",
    "coverage_scope",
    "mean_width_before",
    "mean_width_after",
    "median_width_before",
    "median_width_after",
    "q90_width_before",
    "q90_width_after",
    "residual",
    "deleted_target_signal_delta",
    "target_signal_full",
    "target_signal_masked",
    "P00_full",
    "P00_masked",
    "covariance_delta",
    "total_information",
    "kept_information",
    "heldout_information",
    "heldout_information_fraction",
    "deleted_replicates",
    "deleted_observations",
    "delta_variance",
    "delta_variance_source",
    "row_weight",
    "sd_before",
    "sd_after",
    "a_state",
    "factor_segment",
    "segment_raw_factor",
    "segment_bootstrap_variance",
    "segment_shrinkage_weight",
    "contig_shrinkage_weight",
    "key",
    "value",
]


class uncertaintyCalibrationResult(NamedTuple):
    factor: np.ndarray
    calibratedUncertainty: np.ndarray
    summary: pd.DataFrame
    scores: pd.DataFrame
    model: dict[str, Any]


def _firstSet(params: core.uncertaintyCalibrationParams, *names: str, default: Any = None):
    for name in names:
        if hasattr(params, name):
            value = getattr(params, name)
            if value is not None:
                return value
    return default


def _jsonSafe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA:
        return None
    if isinstance(value, np.ndarray):
        return [_jsonSafe(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonSafe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonSafe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        valueFloat = float(value)
        if np.isnan(valueFloat):
            return {"nonfinite": "nan"}
        if np.isposinf(valueFloat):
            return {"nonfinite": "inf"}
        if np.isneginf(valueFloat):
            return {"nonfinite": "-inf"}
        return valueFloat
    return value


def _initJsonlFile(path: str | Path) -> Path:
    return _logging_utils.init_jsonl_log(path)


def _jsonlRecords(rows: list[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        rowRecords = [
            dict(zip(rows.columns, values))
            for values in rows.itertuples(index=False, name=None)
        ]
    else:
        rowRecords = [dict(row) for row in rows]
    records: list[dict[str, Any]] = []
    for row in rowRecords:
        converted = {str(key): _jsonSafe(value) for key, value in row.items()}
        records.append({key: value for key, value in converted.items() if value is not None})
    return records


def _appendJsonlRecords(path: str | Path, rows: list[dict[str, Any]] | pd.DataFrame) -> int:
    records = _jsonlRecords(rows)
    if not records:
        return 0
    return _logging_utils.append_jsonl_log(path, records)


def _calibrationKeyValueRows(
    *,
    recordType: str,
    event: str,
    chromosome: str | None,
    values: dict[str, Any],
    fold: int | None = None,
) -> list[dict[str, Any]]:
    record = {
        "record_type": recordType,
        "event": event,
        "chromosome": chromosome,
        "fold": None if fold is None else int(fold),
        **{str(key): value for key, value in values.items()},
    }
    return [record]


def _ensureCalibrationLog(path: str | Path) -> Path:
    logPath = Path(path)
    if not logPath.exists():
        _initJsonlFile(logPath)
    return logPath


def _factorBounds(params: core.uncertaintyCalibrationParams) -> tuple[float, float]:
    factorMin = float(
        max(
            _firstSet(
                params,
                "factorMin",
                "minFactor",
                default=core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
            ),
            core.UNCERTAINTY_CALIBRATION_FACTOR_MIN_FLOOR,
        )
    )
    factorMax = float(
        max(
            _firstSet(
                params,
                "factorMax",
                "maxFactor",
                default=core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
            ),
            factorMin * core.UNCERTAINTY_CALIBRATION_FACTOR_MAX_MIN_RATIO,
        )
    )
    return factorMin, factorMax


def _maxScoreRows(params: core.uncertaintyCalibrationParams) -> int:
    return int(
        _firstSet(
            params,
            "maxHeldoutCells",
            "maxScores",
            default=core.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES,
        )
    )


def _calibrationPad() -> float:
    return float(core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD)


def _resolveBlockSizeIntervals(
    blockSizeBP: int | str | None,
    intervalSizeBP: int,
    n: int,
    folds: int | None = None,
) -> int:
    blockLen = diagnostics.resolveUncertaintyBlockSizeIntervals(
        blockSizeBP,
        intervalSizeBP,
        n,
        folds=folds,
    )
    return int(min(blockLen, max(int(n), 1)))


def _makeFoldSpec(
    *,
    m: int,
    n: int,
    blockLen: int,
    folds: int,
    deletionProbability: float,
    seed: int,
) -> list[np.ndarray]:
    if folds < core.UNCERTAINTY_CALIBRATION_MIN_FOLDS:
        raise ValueError("uncertainty calibration requires at least two folds")
    if not (np.isfinite(deletionProbability) and 0.0 < deletionProbability < 1.0):
        raise ValueError("delete-block deletion probability must be in (0, 1)")
    blockFold, repsByBlockCount, repsByBlock = _cuncertainty.cmakeFoldSpec(
        int(m),
        int(n),
        int(blockLen),
        int(folds),
        float(deletionProbability),
        int(seed),
    )
    return (
        np.ascontiguousarray(blockFold, dtype=np.int32),
        np.ascontiguousarray(repsByBlockCount, dtype=np.intp),
        np.ascontiguousarray(repsByBlock, dtype=np.intp),
    )


def _featureMatrix(
    *,
    state: np.ndarray,
    stateVar: np.ndarray,
    matrixMunc: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    featureNames = list(core.UNCERTAINTY_CALIBRATION_FEATURE_NAMES)
    X, center, scale = _cuncertainty.cfeatureMatrix(
        np.ascontiguousarray(np.asarray(state, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.maximum(
            np.asarray(stateVar, dtype=np.float64).reshape(-1),
            core.UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR,
        )),
        np.ascontiguousarray(matrixMunc, dtype=np.float64),
        float(core.UNCERTAINTY_CALIBRATION_FEATURE_HIGH_SIGNAL_QUANTILE),
        float(core.UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR),
        float(core.UNCERTAINTY_CALIBRATION_FEATURE_MAD_NORMAL_SCALE),
        float(core.UNCERTAINTY_CALIBRATION_FEATURE_SCALE_FLOOR),
    )
    return (
        np.asarray(X, dtype=np.float64),
        featureNames,
        np.asarray(center, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )

def _normalZ(target: float) -> float:
    target = float(
        np.clip(
            target,
            core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
            1.0 - core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
        )
    )
    return float(stats.norm.ppf(0.5 + 0.5 * target))


def _targetCalibrationDelta(params: core.uncertaintyCalibrationParams) -> float | None:
    rawDelta = getattr(params, "targetCalibrationDelta", None)
    if rawDelta is None:
        return None
    delta = float(rawDelta)
    if not np.isfinite(delta) or delta <= 0.0:
        return None
    return float(
        np.clip(
            delta,
            core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
            1.0 - core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
        )
    )


def _pacOrderIndex(N: int, target: float, delta: float) -> int | None:
    N = int(N)
    if N < 1:
        return None
    p = float(target)
    delta = float(delta)
    if not (0.0 < p < 1.0 and 0.0 < delta < 1.0):
        return None
    kGrid = np.arange(1, N + 1, dtype=np.int64)
    tails = stats.binom.sf(kGrid - 1, N, p)
    ok = np.flatnonzero(tails <= delta)
    if ok.size == 0:
        return None
    return int(kGrid[int(ok[0])])


def _minBlocksForFiniteBound(target: float, delta: float) -> int | None:
    p = float(target)
    delta = float(delta)
    if not (0.0 < p < 1.0 and 0.0 < delta < 1.0):
        return None
    return int(np.ceil(np.log(delta) / np.log(p)))


def _targetCalibrationSplit(
    blockIndex: np.ndarray,
    *,
    enabled: bool,
    seed: int,
) -> dict[str, Any]:
    blockIndex = np.asarray(blockIndex, dtype=np.int64).reshape(-1)
    valid = blockIndex >= 0
    uniqueBlocks = np.unique(blockIndex[valid])
    blockCount = int(uniqueBlocks[-1] + 1) if uniqueBlocks.size else 0
    scaleMask = np.ones(blockIndex.shape[0], dtype=bool)
    targetMask = np.zeros(blockIndex.shape[0], dtype=bool)
    targetBlockMask = np.zeros(blockCount, dtype=np.uint8)
    if not enabled or uniqueBlocks.size < 2:
        return {
            "enabled": bool(enabled),
            "seed": int(seed),
            "blocks_total": int(uniqueBlocks.size),
            "scale_blocks": uniqueBlocks.astype(np.int64, copy=False),
            "target_blocks": np.empty(0, dtype=np.int64),
            "scale_mask": scaleMask,
            "target_mask": targetMask,
            "target_block_mask": targetBlockMask,
        }

    rng = np.random.default_rng(int(seed))
    permuted = np.asarray(rng.permutation(uniqueBlocks), dtype=np.int64)
    targetCount = int(np.ceil(TARGET_CALIBRATION_FRACTION * float(uniqueBlocks.size)))
    targetCount = int(np.clip(targetCount, 1, uniqueBlocks.size - 1))
    targetBlocks = np.sort(permuted[:targetCount])
    scaleBlocks = np.sort(permuted[targetCount:])
    targetMask = np.isin(blockIndex, targetBlocks)
    scaleMask = np.isin(blockIndex, scaleBlocks)
    targetBlockMask[targetBlocks] = 1
    return {
        "enabled": True,
        "seed": int(seed),
        "blocks_total": int(uniqueBlocks.size),
        "scale_blocks": scaleBlocks,
        "target_blocks": targetBlocks,
        "scale_mask": scaleMask,
        "target_mask": targetMask,
        "target_block_mask": targetBlockMask,
    }


def _targetCalibrationBounds(
    blockScores: np.ndarray,
    *,
    targets: tuple[float, ...],
    delta: float,
) -> list[dict[str, Any]]:
    scores = np.asarray(blockScores, dtype=np.float64).reshape(-1)
    scores = np.sort(scores[np.isfinite(scores)])
    N = int(scores.size)
    bounds: list[dict[str, Any]] = []
    targetValues = tuple(
        float(
            np.clip(
                target,
                core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
                1.0 - core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
            )
        )
        for target in targets
    )
    selectedIndex = int(np.argmax(targetValues))
    for targetIndex, targetClipped in enumerate(targetValues):
        targetRole = (
            _TARGET_ROLE_SELECTED
            if targetIndex == selectedIndex
            else _TARGET_ROLE_DESCRIPTIVE
        )
        k = _pacOrderIndex(N, targetClipped, delta)
        minBlocks = _minBlocksForFiniteBound(targetClipped, delta)
        if k is None:
            tail = (
                None
                if N == 0
                else float(stats.binom.sf(N - 1, N, targetClipped))
            )
            qValue = None if N == 0 else float(scores[-1])
            bounds.append(
                {
                    "target": targetClipped,
                    "alpha": float(1.0 - targetClipped),
                    "target_role": targetRole,
                    "delta": float(delta),
                    "N": N,
                    "k": None,
                    "q": qValue,
                    "q_source": (
                        "empirical_max_without_finite_order_bound"
                        if targetIndex == selectedIndex
                        else "descriptive_empirical_max"
                    ),
                    "bound_available": False,
                    "bound_scope": (
                        _TARGET_BOUND_SCOPE if targetIndex == selectedIndex else None
                    ),
                    "binomial_tail": tail,
                    "allowed_blocks_above_q": None,
                    "min_blocks_for_any_finite_bound": minBlocks,
                }
            )
            continue
        tail = float(stats.binom.sf(k - 1, N, targetClipped))
        bounds.append(
            {
                "target": targetClipped,
                "alpha": float(1.0 - targetClipped),
                "target_role": targetRole,
                "delta": float(delta),
                "N": N,
                "k": int(k),
                "q": float(scores[k - 1]),
                "q_source": (
                    "exchangeability_conditional_order_statistic"
                    if targetIndex == selectedIndex
                    else "descriptive_order_statistic"
                ),
                "bound_available": bool(targetIndex == selectedIndex),
                "bound_scope": (
                    _TARGET_BOUND_SCOPE if targetIndex == selectedIndex else None
                ),
                "binomial_tail": tail,
                "allowed_blocks_above_q": int(N - k),
                "min_blocks_for_any_finite_bound": minBlocks,
            }
        )
    return bounds


def _targetCalibrationScaleBound(
    bounds: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bounds:
        return None
    return max(bounds, key=lambda row: float(row.get("target", 0.0)))


def _targetCalibrationTrackScale(
    targetScaleBound: dict[str, Any] | None,
) -> dict[str, Any]:
    if targetScaleBound is None:
        return {
            "scale": 1.0,
            "target": None,
            "target_z": None,
            "q": None,
            "q_source": None,
            "bound_available": False,
            "bound_scope": _TARGET_BOUND_SCOPE,
            "scaled": False,
            "reason": "no_target_bound",
        }
    target = float(targetScaleBound.get("target", np.nan))
    qValue = targetScaleBound.get("q")
    targetZ = _normalZ(target) if np.isfinite(target) else np.nan
    boundAvailable = bool(targetScaleBound.get("bound_available", False))
    if not boundAvailable:
        return {
            "scale": 1.0,
            "target": target if np.isfinite(target) else None,
            "target_z": float(targetZ) if np.isfinite(targetZ) else None,
            "q": None if qValue is None else float(qValue),
            "q_source": targetScaleBound.get("q_source"),
            "bound_available": False,
            "bound_scope": _TARGET_BOUND_SCOPE,
            "scaled": False,
            "reason": "finite_order_bound_unavailable",
        }
    if qValue is None:
        return {
            "scale": 1.0,
            "target": target if np.isfinite(target) else None,
            "target_z": float(targetZ) if np.isfinite(targetZ) else None,
            "q": None,
            "q_source": targetScaleBound.get("q_source"),
            "bound_available": True,
            "bound_scope": _TARGET_BOUND_SCOPE,
            "scaled": False,
            "reason": "no_finite_target_bound",
        }
    qFloat = float(qValue)
    if not (np.isfinite(qFloat) and qFloat > 0.0 and np.isfinite(targetZ) and targetZ > 0.0):
        return {
            "scale": 1.0,
            "target": target if np.isfinite(target) else None,
            "target_z": float(targetZ) if np.isfinite(targetZ) else None,
            "q": qFloat if np.isfinite(qFloat) else None,
            "q_source": targetScaleBound.get("q_source"),
            "bound_available": True,
            "bound_scope": _TARGET_BOUND_SCOPE,
            "scaled": False,
            "reason": "nonfinite_target_bound",
        }
    return {
        "scale": float(qFloat / targetZ),
        "target": target,
        "target_z": float(targetZ),
        "q": qFloat,
        "q_source": targetScaleBound.get("q_source"),
        "bound_available": True,
        "bound_scope": _TARGET_BOUND_SCOPE,
        "scaled": True,
        "reason": "scaled_by_exchangeability_conditional_order_bound_q_over_z",
    }


def _samplePositionsByCode(
    codes: np.ndarray,
    *,
    maxRows: int,
    seed: int,
) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    n = int(codes.size)
    maxRows = int(maxRows)
    if maxRows <= 0 or n <= maxRows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    uniqueCodes, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    groupCount = int(uniqueCodes.size)
    quota = np.floor(counts.astype(np.float64) * (float(maxRows) / float(n))).astype(np.int64)
    quota = np.minimum(counts, quota)
    quota[counts > 0] = np.maximum(quota[counts > 0], 1)
    extra = int(maxRows - int(np.sum(quota)))
    if extra > 0:
        fractional = counts.astype(np.float64) * (float(maxRows) / float(n)) - quota
        order = np.lexsort((uniqueCodes, -fractional))
        for group in order:
            if extra <= 0:
                break
            if quota[group] < counts[group]:
                quota[group] += 1
                extra -= 1
    elif extra < 0:
        order = np.lexsort((uniqueCodes, quota))
        for group in order:
            if extra >= 0:
                break
            if quota[group] > 1:
                quota[group] -= 1
                extra += 1
    pieces: list[np.ndarray] = []
    for group in range(groupCount):
        take = int(quota[group])
        if take <= 0:
            continue
        idx = np.flatnonzero(inverse == group)
        if idx.size > take:
            idx = rng.choice(idx, size=take, replace=False)
        pieces.append(np.asarray(idx, dtype=np.int64))
    if not pieces:
        return np.arange(min(n, maxRows), dtype=np.int64)
    out = np.sort(np.concatenate(pieces).astype(np.int64, copy=False))
    if out.size > maxRows:
        out = np.sort(rng.choice(out, size=maxRows, replace=False).astype(np.int64))
    return out


def _signalLevelCoverageCodes(signalAbs: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    signalAbs = np.asarray(signalAbs, dtype=np.float64).reshape(-1)
    if signalAbs.size == 0:
        return np.empty(0, dtype=np.int32), {}
    if not np.all(np.isfinite(signalAbs)):
        raise ValueError("state coverage signal contains nonfinite values")
    quantiles = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    cuts = np.quantile(signalAbs, quantiles)
    codes = np.searchsorted(cuts[1:], signalAbs, side="left").astype(np.int32, copy=False)
    names = {int(idx): _COVERAGE_CODE_NAMES[int(idx)] for idx in np.unique(codes)}
    return np.ascontiguousarray(codes, dtype=np.int32), names


def _coverageRowsFromCodes(
    *,
    residual: np.ndarray,
    sdBefore: np.ndarray,
    sdAfter: np.ndarray,
    coverageCode: np.ndarray,
    targets: tuple[float, ...],
    coverageScope: str,
) -> list[dict[str, float | int | str | None]]:
    groupCode = np.ascontiguousarray(coverageCode, dtype=np.int32)
    if groupCode.ndim != 1 or groupCode.shape[0] != np.asarray(residual).size:
        raise ValueError("state coverage codes must align with residual rows")
    if np.any((groupCode < 0) | (groupCode > 4)):
        raise ValueError("state coverage codes must be in [0, 4]")
    targetsArray = np.ascontiguousarray(tuple(float(t) for t in targets), dtype=np.float64)
    selectedTarget = float(np.max(targetsArray))
    targetZ = np.ascontiguousarray([_normalZ(target) for target in targetsArray], dtype=np.float64)
    rows = _cuncertainty.csummarizeCoverageWidths(
        np.ascontiguousarray(residual, dtype=np.float64),
        np.ascontiguousarray(sdBefore, dtype=np.float64),
        np.ascontiguousarray(sdAfter, dtype=np.float64),
        groupCode,
        targetsArray,
        targetZ,
        0.5,
        float(core.UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE),
    )
    out: list[dict[str, float | int | str | None]] = []
    for idx, group in enumerate(np.asarray(rows["group"], dtype=np.int32)):
        target = float(rows["target"][idx])
        out.append(
            {
                "stratum": (
                    "overall" if int(group) < 0 else _COVERAGE_CODE_NAMES[int(group)]
                ),
                "target": target,
                "target_role": (
                    _TARGET_ROLE_SELECTED
                    if target == selectedTarget
                    else _TARGET_ROLE_DESCRIPTIVE
                ),
                "z": _normalZ(target),
                "n": int(rows["n"][idx]),
                "coverage_before": float(rows["coverage_before"][idx]),
                "coverage_after": float(rows["coverage_after"][idx]),
                "mean_width_before": float(rows["mean_width_before"][idx]),
                "mean_width_after": float(rows["mean_width_after"][idx]),
                "median_width_before": float(rows["median_width_before"][idx]),
                "median_width_after": float(rows["median_width_after"][idx]),
                "coverage_estimand": _COVERAGE_ESTIMAND,
                "coverage_scope": str(coverageScope),
            }
        )
    return out


def _summarizeScores(
    *,
    residual: np.ndarray,
    sdBefore: np.ndarray,
    sdAfter: np.ndarray,
    uncertaintyDecile: np.ndarray,
    targets: tuple[float, ...],
) -> pd.DataFrame:
    decile = np.asarray(uncertaintyDecile, dtype=np.int32).reshape(-1)
    targetsArray = np.ascontiguousarray(tuple(float(t) for t in targets), dtype=np.float64)
    selectedTarget = float(np.max(targetsArray))
    targetZ = np.ascontiguousarray([_normalZ(target) for target in targetsArray], dtype=np.float64)
    summaryDict = _cuncertainty.csummarizeCoverageWidths(
        np.ascontiguousarray(residual, dtype=np.float64),
        np.ascontiguousarray(sdBefore, dtype=np.float64),
        np.ascontiguousarray(sdAfter, dtype=np.float64),
        np.ascontiguousarray(decile, dtype=np.int32),
        targetsArray,
        targetZ,
        float(core.UNCERTAINTY_CALIBRATION_SUMMARY_MEDIAN_QUANTILE),
        float(core.UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE),
    )
    summary = pd.DataFrame(summaryDict)
    summary["stratum"] = [
        "overall" if int(group) < 0 else f"uncertainty_decile_{int(group)}"
        for group in summary.pop("group")
    ]
    orderedColumns = [
        "stratum",
        "target",
        "n",
        "coverage_before",
        "coverage_after",
        "mean_width_before",
        "mean_width_after",
        "median_width_before",
        "median_width_after",
        "q90_width_before",
        "q90_width_after",
    ]
    summary = summary[orderedColumns]
    summary["target_role"] = np.where(
        summary["target"].to_numpy(dtype=np.float64) == selectedTarget,
        _TARGET_ROLE_SELECTED,
        _TARGET_ROLE_DESCRIPTIVE,
    )
    summary["coverage_estimand"] = _COVERAGE_ESTIMAND
    summary["coverage_scope"] = _FIT_COVERAGE_SCOPE
    return summary


def _deleteBlockFactorDistribution(factor: np.ndarray) -> dict[str, Any]:
    factorValues = np.asarray(factor, dtype=np.float64).reshape(-1)
    if factorValues.size == 0:
        raise ValueError("delete-block factor is empty")
    if not np.all(np.isfinite(factorValues)) or np.any(factorValues <= 0.0):
        raise ValueError("delete-block factor must be finite and positive")
    quantileMethod = "linear"
    factorQ05, factorQ95 = np.quantile(
        factorValues,
        [0.05, 0.95],
        method=quantileMethod,
    )
    factorMedian = float(np.median(factorValues))
    sdFactorValues = np.sqrt(factorValues)
    sdFactorMedian = float(np.median(sdFactorValues))
    sdFactorQ05, sdFactorQ95 = np.quantile(
        sdFactorValues,
        [0.05, 0.95],
        method=quantileMethod,
    )
    return {
        "count": int(factorValues.size),
        "median": factorMedian,
        "unscaled_mad": float(np.median(np.abs(factorValues - factorMedian))),
        "q05": float(factorQ05),
        "q95": float(factorQ95),
        "min": float(np.min(factorValues)),
        "max": float(np.max(factorValues)),
        "sd_multiplier_median": sdFactorMedian,
        "sd_multiplier_unscaled_mad": float(
            np.median(np.abs(sdFactorValues - sdFactorMedian))
        ),
        "sd_multiplier_q05": float(sdFactorQ05),
        "sd_multiplier_q95": float(sdFactorQ95),
        "sd_multiplier_min": float(np.min(sdFactorValues)),
        "sd_multiplier_max": float(np.max(sdFactorValues)),
        "quantile_method": quantileMethod,
    }


def _validateCalibrationReplayArrays(
    replayData: Mapping[str, Any],
    *,
    intervalCount: int,
    blockLenIntervals: int,
    positiveFloor: float,
) -> dict[str, np.ndarray]:
    replayKeys = set(replayData)
    expectedKeys = set(_CALIBRATION_REPLAY_KEYS)
    if replayKeys != expectedKeys:
        missing = sorted(expectedKeys - replayKeys)
        extra = sorted(replayKeys - expectedKeys)
        raise ValueError(
            "uncertainty calibration replay keys do not match the contract: "
            f"missing={missing} extra={extra}"
        )
    if int(intervalCount) < 1:
        raise ValueError("uncertainty calibration replay interval count must be positive")
    if int(blockLenIntervals) < 1:
        raise ValueError("uncertainty calibration replay block length must be positive")
    if not np.isfinite(positiveFloor) or float(positiveFloor) <= 0.0:
        raise ValueError("uncertainty calibration replay positive floor must be positive")
    arrays: dict[str, np.ndarray] = {}
    for key in _CALIBRATION_REPLAY_KEYS:
        value = np.asarray(replayData[key])
        expectedDtype = _CALIBRATION_REPLAY_DTYPES[key]
        if value.dtype != expectedDtype:
            raise ValueError(
                f"uncertainty calibration replay {key} must have dtype {expectedDtype}"
            )
        if value.ndim != 1:
            raise ValueError(f"uncertainty calibration replay {key} must be a vector")
        arrays[key] = np.ascontiguousarray(value, dtype=expectedDtype)

    rowCount = int(arrays["residual"].size)
    if rowCount < 1:
        raise ValueError("uncertainty calibration replay has no perturbation rows")
    for key in (
        "pDelta",
        "intervalIndex",
        "deletedObservationAll",
        "coverageCodeAll",
    ):
        if arrays[key].size != rowCount:
            raise ValueError(
                f"uncertainty calibration replay {key} does not match residual rows"
            )
    fitCount = int(arrays["fitRows"].size)
    if fitCount < 1:
        raise ValueError("uncertainty calibration replay has no factor-fit rows")
    for key in ("coverageCodeFit", "summaryDecile"):
        if arrays[key].size != fitCount:
            raise ValueError(
                f"uncertainty calibration replay {key} does not match fit rows"
            )

    residual = arrays["residual"]
    pDelta = arrays["pDelta"]
    intervalIndex = arrays["intervalIndex"]
    fitRows = arrays["fitRows"]
    targetBlockMask = arrays["targetBlockMask"]
    deletedObservationAll = arrays["deletedObservationAll"]
    coverageCodeAll = arrays["coverageCodeAll"]
    coverageCodeFit = arrays["coverageCodeFit"]
    summaryDecile = arrays["summaryDecile"]
    if not np.all(np.isfinite(residual)):
        raise ValueError("uncertainty calibration replay residual must be finite")
    if not np.all(np.isfinite(pDelta)) or np.any(pDelta <= float(positiveFloor)):
        raise ValueError(
            "uncertainty calibration replay pDelta must exceed the positive floor"
        )
    if np.any(intervalIndex < 0) or np.any(intervalIndex >= int(intervalCount)):
        raise ValueError("uncertainty calibration replay interval index is out of bounds")
    if np.any(fitRows < 0) or np.any(fitRows >= rowCount):
        raise ValueError("uncertainty calibration replay fit row is out of bounds")
    if fitRows.size > 1 and np.any(np.diff(fitRows) <= 0):
        raise ValueError(
            "uncertainty calibration replay fit rows must be strictly increasing"
        )
    blockIndex = intervalIndex // int(blockLenIntervals)
    expectedBlockCount = int(np.max(blockIndex)) + 1
    if targetBlockMask.size != expectedBlockCount:
        raise ValueError(
            "uncertainty calibration replay target mask does not match rebuilt blocks"
        )
    if np.any((targetBlockMask != 0) & (targetBlockMask != 1)):
        raise ValueError("uncertainty calibration replay target mask must be binary")
    presentBlockMask = np.zeros(expectedBlockCount, dtype=bool)
    presentBlockMask[np.unique(blockIndex)] = True
    if np.any(targetBlockMask[~presentBlockMask] != 0):
        raise ValueError(
            "uncertainty calibration replay selects a block without perturbation rows"
        )
    if np.any(deletedObservationAll < 1):
        raise ValueError(
            "uncertainty calibration replay deleted-observation counts must be positive"
        )
    if np.any((coverageCodeAll < 0) | (coverageCodeAll > 4)):
        raise ValueError("uncertainty calibration replay all-row coverage code is invalid")
    if np.any((coverageCodeFit < 0) | (coverageCodeFit > 4)):
        raise ValueError("uncertainty calibration replay fit-row coverage code is invalid")
    if np.any((summaryDecile < -1) | (summaryDecile > 9)):
        raise ValueError("uncertainty calibration replay summary decile is invalid")
    return arrays


def _writeCalibrationReplay(
    replayPath: str | Path,
    replayData: Mapping[str, np.ndarray],
) -> None:
    path = Path(replayPath)
    temporaryPath: str | None = None
    fileDescriptor, temporaryPath = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fileDescriptor, "wb") as handle:
            np.savez(handle, **{key: replayData[key] for key in _CALIBRATION_REPLAY_KEYS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporaryPath, path)
        temporaryPath = None
    finally:
        if temporaryPath is not None:
            try:
                os.unlink(temporaryPath)
            except FileNotFoundError:
                pass


def _loadCalibrationReplay(
    replayPath: str | Path,
    *,
    intervalCount: int,
    blockLenIntervals: int,
    positiveFloor: float,
) -> dict[str, np.ndarray]:
    with np.load(Path(replayPath), allow_pickle=False) as replay:
        return _validateCalibrationReplayArrays(
            {key: replay[key] for key in replay.files},
            intervalCount=intervalCount,
            blockLenIntervals=blockLenIntervals,
            positiveFloor=positiveFloor,
        )


def _evaluateDeleteBlockCalibration(
    *,
    factorRaw: np.ndarray,
    fullP: np.ndarray,
    residual: np.ndarray,
    pDelta: np.ndarray,
    intervalIndex: np.ndarray,
    fitRows: np.ndarray,
    blockIndex: np.ndarray,
    targetBlockMask: np.ndarray,
    deletedObservationAll: np.ndarray,
    coverageCodeAll: np.ndarray,
    coverageCodeFit: np.ndarray,
    summaryDecile: np.ndarray,
    targets: tuple[float, ...],
    targetCalibrationEnabled: bool,
    targetDelta: float | None,
    scaleByTargetCalibration: bool,
    positiveFloor: float,
) -> dict[str, Any]:
    factorRaw = np.ascontiguousarray(factorRaw, dtype=np.float64)
    fullP = np.ascontiguousarray(fullP, dtype=np.float64)
    if factorRaw.ndim != 1 or fullP.ndim != 1 or factorRaw.shape != fullP.shape:
        raise ValueError("delete-block factor and fullP must be aligned vectors")
    if not np.all(np.isfinite(factorRaw)) or np.any(factorRaw <= 0.0):
        raise ValueError("delete-block raw factor must be finite and positive")
    if not np.all(np.isfinite(fullP)) or np.any(fullP <= 0.0):
        raise ValueError("delete-block fullP must be finite and positive")
    factor = np.maximum(factorRaw, 1.0)
    residual = np.ascontiguousarray(residual, dtype=np.float64)
    pDelta = np.ascontiguousarray(pDelta, dtype=np.float64)
    intervalIndex = np.ascontiguousarray(intervalIndex, dtype=np.int64)
    fitRows = np.ascontiguousarray(fitRows, dtype=np.int64)
    blockIndex = np.ascontiguousarray(blockIndex, dtype=np.int64)
    targetBlockMask = np.ascontiguousarray(targetBlockMask, dtype=np.uint8)
    deletedObservationAll = np.ascontiguousarray(
        deletedObservationAll,
        dtype=np.int64,
    )
    targets = tuple(float(target) for target in targets)

    targetBlockIds = np.empty(0, dtype=np.int64)
    targetBlockScores = np.empty(0, dtype=np.float64)
    targetBlockCellCounts = np.empty(0, dtype=np.int64)
    targetBounds: list[dict[str, Any]] = []
    if targetCalibrationEnabled:
        if targetDelta is None:
            raise ValueError("enabled target calibration requires delta")
        targetBlockIds, targetBlockScores, targetBlockCellCounts = (
            _cuncertainty.cdeleteBlockBlockScores(
                residual,
                pDelta,
                factor,
                intervalIndex,
                blockIndex,
                targetBlockMask,
                heldoutCounts=deletedObservationAll,
                varianceFloor=float(positiveFloor),
            )
        )
        targetBlockIds = np.asarray(targetBlockIds, dtype=np.int64)
        targetBlockScores = np.asarray(targetBlockScores, dtype=np.float64)
        targetBlockCellCounts = np.asarray(targetBlockCellCounts, dtype=np.int64)
        targetBounds = _targetCalibrationBounds(
            targetBlockScores,
            targets=targets,
            delta=float(targetDelta),
        )

    targetScaleBound = _targetCalibrationScaleBound(targetBounds)
    targetScaleInfo = _targetCalibrationTrackScale(targetScaleBound)
    uncertaintyTrackScale = 1.0
    uncertaintyTrackScaled = False
    uncertaintyTrackScaleReason = "target_calibration_disabled"
    if targetCalibrationEnabled:
        uncertaintyTrackScaleReason = "scale_disabled_by_config"
        if scaleByTargetCalibration:
            uncertaintyTrackScale = float(targetScaleInfo["scale"])
            uncertaintyTrackScaled = bool(targetScaleInfo["scaled"])
            uncertaintyTrackScaleReason = str(targetScaleInfo["reason"])

    effectiveFactor = np.maximum(
        factor * uncertaintyTrackScale * uncertaintyTrackScale,
        1.0,
    )
    postFitDiagnostics = _cuncertainty.cdeleteBlockPostFitDiagnostics(
        residual,
        pDelta,
        effectiveFactor,
        intervalIndex,
        blockIndex,
        targetBlockMask,
        fitRows,
        float(positiveFloor),
    )
    sdBeforeAll = np.asarray(postFitDiagnostics["sd_before_all"], dtype=np.float64)
    sdAfterAll = np.asarray(postFitDiagnostics["sd_after_all"], dtype=np.float64)
    sdBeforeFit = np.asarray(postFitDiagnostics["sd_before_fit"], dtype=np.float64)
    sdAfterFit = np.asarray(postFitDiagnostics["sd_after_fit"], dtype=np.float64)
    heldFactorFit = np.asarray(
        postFitDiagnostics["held_factor_fit"],
        dtype=np.float64,
    )
    rawSD = np.sqrt(np.maximum(fullP, float(positiveFloor)))
    baseSD = np.sqrt(np.maximum(factor * fullP, float(positiveFloor)))
    scaledBaseSD = baseSD * uncertaintyTrackScale
    modelSEFloorHits = int(
        np.count_nonzero((factorRaw < 1.0) | (scaledBaseSD < rawSD))
    )
    calibrated = np.maximum(scaledBaseSD, rawSD).astype(np.float32)
    stateCoverage = _coverageRowsFromCodes(
        residual=residual,
        sdBefore=sdBeforeAll,
        sdAfter=sdAfterAll,
        coverageCode=coverageCodeAll,
        targets=targets,
        coverageScope=_COVERAGE_SCOPE,
    )
    stateCoverageFit = _coverageRowsFromCodes(
        residual=residual[fitRows],
        sdBefore=sdBeforeFit,
        sdAfter=sdAfterFit,
        coverageCode=coverageCodeFit,
        targets=targets,
        coverageScope=_FIT_COVERAGE_SCOPE,
    )
    summary = _summarizeScores(
        residual=residual[fitRows],
        sdBefore=sdBeforeFit,
        sdAfter=sdAfterFit,
        uncertaintyDecile=summaryDecile,
        targets=targets,
    )
    presentBlocks = np.unique(blockIndex)
    targetBlockCount = int(np.count_nonzero(targetBlockMask[presentBlocks]))
    targetMetadata = {
        "enabled": bool(targetCalibrationEnabled),
        "delta": None if targetDelta is None else float(targetDelta),
        "target_block_fraction": float(TARGET_CALIBRATION_FRACTION),
        "blocks_total": int(presentBlocks.size),
        "blocks_scale": int(presentBlocks.size - targetBlockCount),
        "blocks_target": targetBlockCount,
        "blocks_target_scored": int(targetBlockScores.size),
        "target_block_cells": int(np.sum(targetBlockCellCounts)),
        "scale_uncertainty_by_target_calibration": bool(scaleByTargetCalibration),
        "uncertainty_track_scaled": bool(uncertaintyTrackScaled),
        "uncertainty_track_scale": float(uncertaintyTrackScale),
        "uncertainty_track_scale_target": targetScaleInfo["target"],
        "uncertainty_track_scale_target_z": targetScaleInfo["target_z"],
        "uncertainty_track_scale_q": targetScaleInfo["q"],
        "uncertainty_track_scale_q_source": targetScaleInfo["q_source"],
        "uncertainty_track_scale_bound_available": bool(
            targetScaleInfo["bound_available"]
        ),
        "uncertainty_track_scale_bound_scope": _TARGET_BOUND_SCOPE,
        "uncertainty_track_scale_reason": uncertaintyTrackScaleReason,
        "score_definition": _TARGET_PERTURBATION_SCORE_DEFINITION,
        "bounds": targetBounds,
    }
    return {
        "factor": factor.astype(np.float32),
        "calibrated": calibrated,
        "summary": summary,
        "stateCoverage": stateCoverage,
        "stateCoverageFit": stateCoverageFit,
        "sdBeforeAll": sdBeforeAll,
        "sdAfterAll": sdAfterAll,
        "sdBeforeFit": sdBeforeFit,
        "sdAfterFit": sdAfterFit,
        "heldFactorFit": heldFactorFit,
        "targetBlockIds": targetBlockIds,
        "targetBlockScores": targetBlockScores,
        "targetBlockCellCounts": targetBlockCellCounts,
        "targetCalibration": targetMetadata,
        "modelSEFloorHits": modelSEFloorHits,
        "factorDistribution": _deleteBlockFactorDistribution(
            factor.astype(np.float32)
        ),
    }


def _evaluateSegShrinkReplay(
    *,
    factorRaw: np.ndarray,
    fullP: np.ndarray,
    calibrationModel: dict[str, Any],
    replayPath: str | Path,
    positiveFloor: float,
) -> dict[str, Any]:
    if not isinstance(calibrationModel, Mapping):
        raise ValueError("segShrink calibration model must be a mapping")
    model = copy.deepcopy(dict(calibrationModel))
    fullPArray = np.asarray(fullP)
    factorRawArray = np.asarray(factorRaw)
    if fullPArray.dtype != np.dtype(np.float64):
        raise ValueError("segShrink replay fullP must have dtype float64")
    if factorRawArray.dtype != np.dtype(np.float64):
        raise ValueError("segShrink replay raw factor must have dtype float64")
    if fullPArray.ndim != 1 or factorRawArray.ndim != 1:
        raise ValueError("segShrink replay fullP and raw factor must be vectors")
    if fullPArray.shape != factorRawArray.shape or fullPArray.size == 0:
        raise ValueError("segShrink replay fullP and raw factor must be aligned")
    if not np.all(np.isfinite(fullPArray)) or np.any(fullPArray <= 0.0):
        raise ValueError("segShrink replay fullP must be finite and positive")
    if not np.all(np.isfinite(factorRawArray)) or np.any(factorRawArray <= 0.0):
        raise ValueError("segShrink replay raw factor must be finite and positive")
    if not np.isfinite(positiveFloor) or float(positiveFloor) <= 0.0:
        raise ValueError("segShrink replay positive floor must be finite and positive")
    foldRefits = model.get("fold_refits")
    if not isinstance(foldRefits, Mapping):
        raise ValueError("segShrink replay model requires fold_refits")
    blockLenValue = foldRefits.get("block_len_intervals")
    if isinstance(blockLenValue, (bool, np.bool_)) or not isinstance(
        blockLenValue,
        (int, np.integer),
    ):
        raise ValueError("segShrink replay model block length must be an integer")
    blockLenIntervals = int(blockLenValue)
    if blockLenIntervals < 1:
        raise ValueError("segShrink replay model block length must be positive")
    targetsValue = model.get("targets")
    if not isinstance(targetsValue, (list, tuple)) or len(targetsValue) < 1:
        raise ValueError("segShrink replay model requires targets")
    targets = tuple(float(target) for target in targetsValue)
    if not all(np.isfinite(target) and 0.0 < target < 1.0 for target in targets):
        raise ValueError("segShrink replay model targets must be probabilities")
    targetCalibration = model.get("target_calibration")
    if not isinstance(targetCalibration, Mapping):
        raise ValueError("segShrink replay model requires target calibration")
    for key in (
        "enabled",
        "delta",
        "scale_uncertainty_by_target_calibration",
    ):
        if key not in targetCalibration:
            raise ValueError(f"segShrink replay target calibration requires {key}")
    targetCalibrationEnabled = targetCalibration["enabled"]
    scaleByTargetCalibration = targetCalibration[
        "scale_uncertainty_by_target_calibration"
    ]
    if not isinstance(targetCalibrationEnabled, (bool, np.bool_)):
        raise ValueError("segShrink replay target-calibration enabled flag must be boolean")
    if not isinstance(scaleByTargetCalibration, (bool, np.bool_)):
        raise ValueError("segShrink replay target-calibration scale flag must be boolean")
    targetDeltaValue = targetCalibration["delta"]
    if bool(targetCalibrationEnabled):
        if targetDeltaValue is None:
            raise ValueError("segShrink replay enabled target calibration requires delta")
        targetDelta = float(targetDeltaValue)
        if not np.isfinite(targetDelta) or not 0.0 < targetDelta < 1.0:
            raise ValueError("segShrink replay target delta must be a probability")
    else:
        if targetDeltaValue is not None:
            raise ValueError("segShrink replay disabled target calibration requires null delta")
        targetDelta = None

    replayArrays = _loadCalibrationReplay(
        replayPath,
        intervalCount=int(fullPArray.size),
        blockLenIntervals=blockLenIntervals,
        positiveFloor=float(positiveFloor),
    )
    if not bool(targetCalibrationEnabled) and np.any(
        replayArrays["targetBlockMask"] != 0
    ):
        raise ValueError("segShrink replay disabled target calibration selects target blocks")
    blockIndex = np.ascontiguousarray(
        replayArrays["intervalIndex"] // blockLenIntervals,
        dtype=np.int64,
    )
    evaluated = _evaluateDeleteBlockCalibration(
        factorRaw=np.ascontiguousarray(factorRawArray, dtype=np.float64),
        fullP=np.ascontiguousarray(fullPArray, dtype=np.float64),
        residual=replayArrays["residual"],
        pDelta=replayArrays["pDelta"],
        intervalIndex=replayArrays["intervalIndex"],
        fitRows=replayArrays["fitRows"],
        blockIndex=blockIndex,
        targetBlockMask=replayArrays["targetBlockMask"],
        deletedObservationAll=replayArrays["deletedObservationAll"],
        coverageCodeAll=replayArrays["coverageCodeAll"],
        coverageCodeFit=replayArrays["coverageCodeFit"],
        summaryDecile=replayArrays["summaryDecile"],
        targets=targets,
        targetCalibrationEnabled=bool(targetCalibrationEnabled),
        targetDelta=targetDelta,
        scaleByTargetCalibration=bool(scaleByTargetCalibration),
        positiveFloor=float(positiveFloor),
    )
    replayedTargetCalibration = dict(targetCalibration)
    replayedTargetCalibration.update(evaluated["targetCalibration"])
    model.update(
        {
            "score_definition": _PERTURBATION_SCORE_DEFINITION,
            "coverage_estimand": _COVERAGE_ESTIMAND,
            "coverage_scope": _COVERAGE_SCOPE,
            "coverage_fit_scope": _FIT_COVERAGE_SCOPE,
            "delete_block_factor_distribution": evaluated["factorDistribution"],
            "model_se_floor_applied": True,
            "model_se_floor_hits": int(evaluated["modelSEFloorHits"]),
            "rows_valid": int(replayArrays["residual"].size),
            "rows_fit": int(replayArrays["fitRows"].size),
            "diagnostic_score_rows": 0,
            "state_uncertainty_coverage": evaluated["stateCoverage"],
            "state_uncertainty_coverage_fit": evaluated["stateCoverageFit"],
            "target_calibration": replayedTargetCalibration,
        }
    )
    return {
        "factor": evaluated["factor"],
        "calibrated": evaluated["calibrated"],
        "summary": evaluated["summary"],
        "model": model,
    }


def _diagnosticsRecords(
    *,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    model: dict[str, Any],
    chromosome: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for recordType, event, frame in (
        ("score_sample", "delete_block_calibration.score_sample", scores),
        ("summary", "delete_block_calibration.summary", summary),
    ):
        for row in frame.to_dict(orient="records"):
            record = {"record_type": recordType, "event": event}
            if chromosome is not None:
                record["chromosome"] = str(chromosome)
            record.update(row)
            records.append(record)
    modelRecord = {
        "record_type": "model",
        "event": "delete_block_calibration.model",
    }
    if chromosome is not None:
        modelRecord["chromosome"] = str(chromosome)
    modelRecord.update(model)
    records.append(modelRecord)
    return records


def _coverageLogPayload(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        if row.get("coverage_after") is None:
            continue
        parts.append(
            "target={target:.3g} after calibration: {coverage_after:.3f} n={n}".format(
                **row
            )
        )
    return " ".join(parts)


def _normalizeUncertaintyCalibrationMode(value: str | None) -> str:
    if value is None:
        return core.UNCERTAINTY_CALIBRATION_DEFAULT_MODE
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "delete_block": core.UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE,
        "state_delete_block": core.UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE,
        "delete_block_state": core.UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized != core.UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE:
        raise ValueError(f"unsupported uncertainty calibration mode: {value!r}")
    return normalized


def _normalizeDeleteBlockVarianceMode(value: str | None) -> str:
    if value is None:
        return core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_VARIANCE_MODE
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "covdiff": "covariance_difference",
        "p_diff": "covariance_difference",
        "info": "heldout_information",
        "information": "heldout_information",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in core.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_VARIANCE_MODES:
        raise ValueError(f"unsupported delete-block variance mode: {value!r}")
    return normalized


def _normalizeDeleteBlockTargetSignal(value: str | None) -> str:
    if value is None:
        return core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_TARGET_SIGNAL
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {"signal": "state_plus_background"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in core.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_TARGET_SIGNALS:
        raise ValueError(f"unsupported delete-block target signal: {value!r}")
    return normalized


def _normalizeDeleteBlockFactorModel(value: str | None) -> str:
    if value is None:
        return core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_MODEL
    normalized = str(value).strip()
    if normalized not in core.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_MODELS:
        raise ValueError(
            f"unsupported delete-block factor model: {value!r}; supported values: "
            f"{', '.join(core.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_MODELS)}"
        )
    return normalized


def _normalizeDeleteBlockScoreWeightMode(value: str | None) -> str:
    if value is None:
        return core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_SCORE_WEIGHT_MODE
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in core.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_SCORE_WEIGHT_MODES:
        raise ValueError(f"unsupported delete-block score weight mode: {value!r}")
    return normalized


def _activeObservationMask(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    originalObservationMask: np.ndarray | None,
    pad: float,
) -> np.ndarray:
    active = (
        np.isfinite(matrixData)
        & np.isfinite(matrixMunc)
        & (matrixMunc < 0.5 * float(core.UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE))
        & np.isfinite(matrixMunc + np.float32(pad))
        & ((matrixMunc + np.float32(pad)) > 0.0)
    )
    if originalObservationMask is not None:
        original = np.asarray(originalObservationMask)
        if original.shape != active.shape:
            raise ValueError("originalObservationMask must match matrixData shape")
        active &= original.astype(bool)
    return np.ascontiguousarray(active, dtype=np.uint8)


def _observationLambdaValues(
    n: int,
    *,
    lambdaExp: np.ndarray | None = None,
    useLambda: bool = False,
    lambdaMin: float = 1.0,
    lambdaMax: float = 1.0,
) -> np.ndarray:
    if not useLambda:
        return np.empty(0, dtype=np.float64)
    if lambdaExp is None:
        raise ValueError(
            "deleteBlockUseLambdaInInformation=True requires fullObservationPrecision"
        )
    lambdaMin = float(lambdaMin)
    lambdaMax = float(lambdaMax)
    if not (
        np.isfinite(lambdaMin)
        and np.isfinite(lambdaMax)
        and lambdaMin <= lambdaMax
    ):
        raise ValueError("observation precision multiplier bounds are invalid")
    lam = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
    if lam.shape[0] != int(n):
        raise ValueError("fullObservationPrecision must match interval count")
    if not np.all(np.isfinite(lam)):
        raise ValueError("fullObservationPrecision must be finite")
    return np.ascontiguousarray(np.clip(lam, lambdaMin, lambdaMax), dtype=np.float64)


def _chooseDeleteBlockDeltaVariance(
    P00Full: np.ndarray,
    P00Masked: np.ndarray,
    h: np.ndarray,
    *,
    mode: str,
    minDeltaVariance: float,
    minInformationFraction: float,
    maxInformationFraction: float,
    positiveFloor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P00Full = np.asarray(P00Full, dtype=np.float64).reshape(-1)
    P00Masked = np.asarray(P00Masked, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    minDelta = float(max(minDeltaVariance, positiveFloor))
    covDelta = P00Masked - P00Full
    covValid = (
        np.isfinite(P00Full)
        & np.isfinite(P00Masked)
        & (P00Full > 0.0)
        & (P00Masked > 0.0)
        & np.isfinite(covDelta)
        & (covDelta > minDelta)
    )
    hValid = (
        np.isfinite(P00Full)
        & (P00Full > 0.0)
        & np.isfinite(h)
        & (h >= float(minInformationFraction))
        & (h <= float(maxInformationFraction))
        & (h < 1.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        infoDelta = P00Full * h / (1.0 - h)
    infoValid = hValid & np.isfinite(infoDelta) & (infoDelta > minDelta)

    deltaVariance = np.full(P00Full.shape[0], np.nan, dtype=np.float64)
    sourceCode = np.full(
        P00Full.shape[0],
        _DELETE_BLOCK_SOURCE_INVALID,
        dtype=np.uint8,
    )
    valid = np.zeros(P00Full.shape[0], dtype=bool)
    reasonCode = np.full(P00Full.shape[0], _DELETE_BLOCK_REASON_OTHER, dtype=np.uint8)

    if mode == "covariance_difference":
        valid = covValid
        deltaVariance[valid] = covDelta[valid]
        sourceCode[valid] = _DELETE_BLOCK_SOURCE_COVARIANCE_DIFFERENCE
    elif mode == "heldout_information":
        valid = infoValid
        deltaVariance[valid] = infoDelta[valid]
        sourceCode[valid] = _DELETE_BLOCK_SOURCE_HELDOUT_INFORMATION
    elif mode == "hybrid":
        valid = covValid | infoValid
        deltaVariance[covValid] = covDelta[covValid]
        sourceCode[covValid] = _DELETE_BLOCK_SOURCE_COVARIANCE_DIFFERENCE
        fallback = ~covValid & infoValid
        deltaVariance[fallback] = infoDelta[fallback]
        sourceCode[fallback] = _DELETE_BLOCK_SOURCE_HELDOUT_INFORMATION_FALLBACK
    else:
        raise AssertionError(f"unhandled delete-block variance mode: {mode}")

    reasonCode[valid] = _DELETE_BLOCK_REASON_VALID
    reasonCode[
        ~np.isfinite(P00Full) | (P00Full <= 0.0)
    ] = _DELETE_BLOCK_REASON_NONFINITE_COVARIANCE
    reasonCode[
        np.isfinite(P00Full) & (P00Full > 0.0) & ~hValid
    ] = _DELETE_BLOCK_REASON_H_OUT_OF_BOUNDS
    reasonCode[
        np.isfinite(P00Full)
        & (P00Full > 0.0)
        & hValid
        & ~covValid
        & ~infoValid
    ] = _DELETE_BLOCK_REASON_INFORMATION_DELTA_INVALID
    reasonCode[
        np.isfinite(P00Full)
        & np.isfinite(P00Masked)
        & (P00Full > 0.0)
        & (P00Masked > 0.0)
        & np.isfinite(covDelta)
        & (covDelta <= minDelta)
    ] = _DELETE_BLOCK_REASON_COVARIANCE_DELTA_NONPOSITIVE
    reasonCode[valid] = _DELETE_BLOCK_REASON_VALID
    return deltaVariance, sourceCode, valid, reasonCode


def _deleteBlockRowWeights(h: np.ndarray, params: core.uncertaintyCalibrationParams) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    clipped = np.clip(
        h,
        float(params.deleteBlockMinInformationFraction),
        float(params.deleteBlockMaxInformationFraction),
    )
    mode = _normalizeDeleteBlockScoreWeightMode(params.deleteBlockScoreWeightMode)
    if mode == "uniform":
        return np.ones_like(clipped, dtype=np.float64)
    if mode == "information_fraction":
        return clipped.astype(np.float64, copy=False)
    if mode == "sqrt_information_fraction":
        return np.sqrt(clipped).astype(np.float64, copy=False)
    raise AssertionError(f"unhandled delete-block weight mode: {mode}")


def _replicateDependenceEstimateFromEvidence(
    *,
    zWeightedSum: float,
    weightSum: float,
    blockCount: int,
    pairCount: int,
    rhoUpperBound: float,
) -> dict[str, Any]:
    zWeightedSum = float(zWeightedSum)
    weightSum = float(weightSum)
    rhoUpperBound = float(rhoUpperBound)
    zMean = 0.0
    zShrunk = 0.0
    zSE = None
    rawRho = 0.0
    rho = 0.0
    if weightSum > 0.0:
        zMean = zWeightedSum / weightSum
        zSEValue = float(np.sqrt(1.0 / weightSum))
        zSE = zSEValue
        if zMean > zSEValue:
            zShrunk = zMean - zSEValue
        elif zMean < -zSEValue:
            zShrunk = zMean + zSEValue
        rawRho = float(np.tanh(zMean))
        rho = float(np.tanh(zShrunk))
        rho = min(max(rho, 0.0), rhoUpperBound)
    return {
        "rho": float(rho),
        "raw_rho": float(rawRho),
        "fisher_z_mean": float(zMean),
        "fisher_z_shrunk": float(zShrunk),
        "fisher_z_se": zSE,
        "block_count": int(blockCount),
        "pair_count": int(pairCount),
        "weight_sum": float(weightSum),
        "rho_upper_bound": float(rhoUpperBound),
    }


def _deleteBlockScoreSamplingCodes(
    *,
    foldIndex: np.ndarray,
    intervalIndex: np.ndarray,
    pDelta: np.ndarray,
    fullState: np.ndarray,
    sourceCode: np.ndarray,
) -> np.ndarray:
    try:
        deltaDecile = pd.qcut(
            np.asarray(pDelta, dtype=np.float64),
            q=core.UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES,
            labels=False,
            duplicates="drop",
        )
        deltaCode = np.nan_to_num(np.asarray(deltaDecile, dtype=np.float64), nan=0.0).astype(np.int64)
    except ValueError:
        deltaCode = np.zeros(np.asarray(pDelta).shape[0], dtype=np.int64)
    fullStateArr = np.asarray(fullState, dtype=np.float64)
    stateAbs = np.abs(fullStateArr[np.asarray(intervalIndex, dtype=np.int64)])
    stateCut = (
        float(np.nanquantile(np.abs(fullStateArr), core.UNCERTAINTY_CALIBRATION_SCORE_STATE_ABS_QUANTILE))
        if fullStateArr.size
        else np.inf
    )
    highSignal = (stateAbs >= stateCut).astype(np.int64)
    sourceCodeArr = np.asarray(sourceCode, dtype=np.int64)
    return (
        np.asarray(foldIndex, dtype=np.int64)
        * core.UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE
        + sourceCodeArr * 128
        + deltaCode * 2
        + highSignal
    )


def _weightedQuantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    return float(weighted_quantile(values, weights, float(q)))


def _fitDeleteBlockGlobalFactor(
    *,
    residual: np.ndarray,
    pDelta: np.ndarray,
    rowWeight: np.ndarray,
    params: core.uncertaintyCalibrationParams,
) -> tuple[float, dict[str, Any]]:
    residual = np.asarray(residual, dtype=np.float64)
    pDelta = np.asarray(pDelta, dtype=np.float64)
    rowWeight = np.asarray(rowWeight, dtype=np.float64)
    valid = (
        np.isfinite(residual)
        & np.isfinite(pDelta)
        & (pDelta > core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR)
        & np.isfinite(rowWeight)
        & (rowWeight > 0.0)
    )
    if not np.any(valid):
        raise ValueError("delete-block state factor fit has no valid score rows")
    ratio = np.abs(residual[valid]) / np.sqrt(pDelta[valid])
    weights = rowWeight[valid]
    target = max(tuple(float(t) for t in params.targets))
    z = _normalZ(target)
    sdMultiplier = _weightedQuantile(ratio, weights, target) / z
    factorMin, factorMax = _factorBounds(params)
    factor = float(np.clip(sdMultiplier * sdMultiplier, factorMin, factorMax))
    return factor, {
        "success": True,
        "factor_model": "global",
        "global_factor": factor,
        "global_sd_multiplier": float(np.sqrt(factor)),
        "global_factor_target": float(target),
        "global_factor_target_z": float(z),
    }


def calibrateChromosomeStateUncertainty(
    *,
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    fullState: np.ndarray,
    fullCovar: np.ndarray | None = None,
    fullP: np.ndarray | None = None,
    fullBackground: np.ndarray | None = None,
    fullObservationPrecision: np.ndarray | None = None,
    originalObservationMask: np.ndarray | None = None,
    intervals: np.ndarray | None = None,
    intervalSizeBP: int,
    params: core.uncertaintyCalibrationParams,
    runKwargs: dict[str, Any],
    outPrefix: str | None = None,
    diagnosticsLogPath: str | Path | None = None,
    chromosome: str | None = None,
    calibrationReplayPath: str | Path | None = None,
) -> uncertaintyCalibrationResult:
    totalStart = time.perf_counter()
    timings: dict[str, float] = {}
    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.ascontiguousarray(matrixMunc, dtype=np.float32)
    m, n = matrixData.shape
    if m < 1:
        raise ValueError("uncertainty calibration requires at least one replicate")
    padValue = float(runKwargs.get("pad", _calibrationPad()))
    activeMask = _activeObservationMask(
        matrixData,
        matrixMunc,
        originalObservationMask,
        padValue,
    )
    eligibleReplicates = np.flatnonzero(np.any(activeMask != 0, axis=1))
    eligibleReplicateCount = int(eligibleReplicates.size)
    if eligibleReplicateCount < 1:
        raise ValueError("uncertainty calibration requires an active observation")
    replicateDependenceRhoSetting = getattr(
        params,
        "deleteBlockReplicateDependenceRho",
        core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_REPLICATE_DEPENDENCE_RHO,
    )
    replicateDependenceAuto = (
        isinstance(replicateDependenceRhoSetting, str)
        and replicateDependenceRhoSetting
        == core.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_REPLICATE_DEPENDENCE_RHO_AUTO
    )
    if replicateDependenceAuto:
        replicateDependenceRho = 0.0
    else:
        if isinstance(replicateDependenceRhoSetting, str):
            raise ValueError(
                "deleteBlockReplicateDependenceRho must be 'auto' or in [0, 1)"
            )
        if isinstance(replicateDependenceRhoSetting, (bool, np.bool_)):
            raise ValueError(
                "deleteBlockReplicateDependenceRho must be 'auto' or in [0, 1)"
            )
        replicateDependenceRho = float(replicateDependenceRhoSetting)
        if not (
            np.isfinite(replicateDependenceRho)
            and 0.0 <= replicateDependenceRho < 1.0
        ):
            raise ValueError(
                "deleteBlockReplicateDependenceRho must be 'auto' or in [0, 1)"
            )
        if replicateDependenceRho > 0.0 and eligibleReplicateCount < 2:
            raise ValueError(
                "deleteBlockReplicateDependenceRho > 0 requires at least two samples"
            )
    replicateDependenceEstimate: dict[str, Any] = {
        "source": "auto" if replicateDependenceAuto else "fixed",
        "estimator": (
            "delete_block_sampled_leave_pair_out_fisher_z"
            if replicateDependenceAuto
            else None
        ),
        "raw_rho": None if replicateDependenceAuto else float(replicateDependenceRho),
        "fisher_z_mean": None,
        "fisher_z_shrunk": None,
        "fisher_z_se": None,
        "block_count": None,
        "pair_count": None,
        "weight_sum": None,
        "rho_upper_bound": None,
    }
    if replicateDependenceAuto and eligibleReplicateCount < 2:
        logger.info(
            "uncertaintyCalibration.replicateDependence.auto skipped samples=%s "
            "resolvedRho=0",
            eligibleReplicateCount,
        )
    intervalsArr = (
        np.arange(n, dtype=np.int64) * int(intervalSizeBP)
        if intervals is None
        else np.asarray(intervals, dtype=np.int64)
    )
    if intervalsArr.shape[0] != n:
        raise ValueError("intervals must match the number of matrixData columns")
    calibrationMode = _normalizeUncertaintyCalibrationMode(getattr(params, "mode", None))
    varianceMode = _normalizeDeleteBlockVarianceMode(params.deleteBlockVarianceMode)
    targetSignal = _normalizeDeleteBlockTargetSignal(params.deleteBlockTargetSignal)
    factorModel = _normalizeDeleteBlockFactorModel(params.deleteBlockFactorModel)
    weightMode = _normalizeDeleteBlockScoreWeightMode(params.deleteBlockScoreWeightMode)
    folds = max(int(params.folds), core.UNCERTAINTY_CALIBRATION_MIN_FOLDS)
    blockLen = _resolveBlockSizeIntervals(
        params.blockSizeBP,
        intervalSizeBP,
        n,
        folds=folds,
    )
    deletionProbability = float(params.deleteBlockDeletionProbability)
    if not (np.isfinite(deletionProbability) and 0.0 < deletionProbability < 1.0):
        raise ValueError("deleteBlockDeletionProbability must be in (0, 1)")
    logger.info(
        "uncertaintyCalibration.start mode=delete_block_state intervals=%s "
        "samples=%s folds=%s blockLen=%s deleteBlockDeletionProbability=%s "
        "varianceMode=%s targetSignal=%s factorModel=%s",
        n,
        m,
        folds,
        blockLen,
        deletionProbability,
        varianceMode,
        targetSignal,
        factorModel,
    )
    stageStart = time.perf_counter()
    blockFold, repsByBlockCount, eligibleRepsByBlock = _makeFoldSpec(
        m=eligibleReplicateCount,
        n=n,
        blockLen=blockLen,
        folds=folds,
        deletionProbability=deletionProbability,
        seed=int(params.seed),
    )
    repsByBlock = np.full(
        (eligibleRepsByBlock.shape[0], m),
        -1,
        dtype=np.intp,
    )
    eligibleSlots = eligibleRepsByBlock >= 0
    repsByBlock[:, :eligibleReplicateCount][eligibleSlots] = eligibleReplicates[
        eligibleRepsByBlock[eligibleSlots]
    ]
    repsByBlock = np.ascontiguousarray(repsByBlock, dtype=np.intp)
    deleteCountsByBlock = np.asarray(
        repsByBlockCount,
        dtype=np.int64,
    ).reshape(-1)
    timings["make_masks_seconds"] = time.perf_counter() - stageStart
    timings["make_fold_spec_seconds"] = timings["make_masks_seconds"]
    fullStateArr = np.asarray(fullState, dtype=np.float64)
    fullState0 = fullStateArr[:, 0] if fullStateArr.ndim == 2 else fullStateArr.reshape(-1)
    if fullState0.shape[0] != n:
        raise ValueError("fullState must match the number of matrixData columns")
    stateRoughness = diagnostics.summarizeStateRoughness(
        fullState0,
        blockLenIntervals=blockLen,
        intervalSizeBP=intervalSizeBP,
    )
    if fullP is None:
        if fullCovar is None:
            raise ValueError("either fullP or fullCovar is required")
        fullCovarArr = np.asarray(fullCovar, dtype=np.float64)
        fullPArr = (
            fullCovarArr[:, 0, 0]
            if fullCovarArr.ndim == 3
            else fullCovarArr.reshape(-1)
        )
    else:
        fullPArr = np.asarray(fullP, dtype=np.float64).reshape(-1)
    if fullPArr.shape[0] != n:
        raise ValueError("fullP/fullCovar must match the number of matrixData columns")
    fullPArr = np.maximum(fullPArr, core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR)
    if fullBackground is None:
        fullBackgroundForRhoArr = np.zeros(n, dtype=np.float64)
    else:
        fullBackgroundForRhoArr = np.asarray(fullBackground, dtype=np.float64).reshape(-1)
        if fullBackgroundForRhoArr.shape[0] != n:
            raise ValueError("fullBackground must match the interval count")
    if targetSignal == "state_plus_background":
        if fullBackground is None:
            if bool(runKwargs.get("fitBackground", False)):
                raise ValueError(
                    "deleteBlockTargetSignal='state_plus_background' requires fullBackground"
                )
            fullBackgroundArr = np.zeros(n, dtype=np.float64)
        else:
            fullBackgroundArr = fullBackgroundForRhoArr
    else:
        fullBackgroundArr = np.zeros(n, dtype=np.float64)
    fullTargetSignal = fullState0 + fullBackgroundArr
    lambdaValues = _observationLambdaValues(
        n,
        lambdaExp=fullObservationPrecision,
        useLambda=bool(params.deleteBlockUseLambdaInInformation),
        lambdaMin=float(runKwargs.get("observationPrecisionMultiplierMin", 1.0)),
        lambdaMax=float(runKwargs.get("observationPrecisionMultiplierMax", 1.0)),
    )
    stageStart = time.perf_counter()
    totalInfoNominal = _cuncertainty.cobservationTotalInformation(
        matrixMunc,
        activeMask,
        lambdaValues,
        bool(params.deleteBlockUseLambdaInInformation),
        padValue,
    )
    if (not replicateDependenceAuto) and replicateDependenceRho > 0.0:
        totalInfoBase = _cuncertainty.cobservationTotalInformation(
            matrixMunc,
            activeMask,
            lambdaValues,
            bool(params.deleteBlockUseLambdaInInformation),
            padValue,
            replicateDependenceRho,
        )
    else:
        totalInfoBase = totalInfoNominal
    timings["total_information_seconds"] = time.perf_counter() - stageStart
    stageStart = time.perf_counter()
    featureMatrix, featureNames, featureCenter, featureScale = _featureMatrix(
        state=fullState0,
        stateVar=fullPArr,
        matrixMunc=matrixMunc,
    )
    timings["feature_matrix_seconds"] = time.perf_counter() - stageStart
    residualChunks: list[np.ndarray] = []
    pDeltaChunks: list[np.ndarray] = []
    iChunks: list[np.ndarray] = []
    foldChunks: list[np.ndarray] = []
    hChunks: list[np.ndarray] = []
    sourceCodeChunks: list[np.ndarray] = []
    rowWeightChunks: list[np.ndarray] = []
    stateMaskedChunks: list[np.ndarray] = []
    pMaskedChunks: list[np.ndarray] = []
    covDeltaChunks: list[np.ndarray] = []
    totalInfoChunks: list[np.ndarray] = []
    keptInfoChunks: list[np.ndarray] = []
    heldInfoChunks: list[np.ndarray] = []
    totalDeffChunks: list[np.ndarray] = []
    heldoutDeffChunks: list[np.ndarray] = []
    deletedReplicateChunks: list[np.ndarray] = []
    deletedObservationChunks: list[np.ndarray] = []
    invalidReasonCountByCode = np.zeros(
        DELETE_BLOCK_INVALID_REASON_LABELS.shape[0],
        dtype=np.int64,
    )
    rowsTotal = 0
    foldFailures = 0
    deletedReplicateIntervalTotal = 0
    deletedObservationIntervalTotal = 0
    foldDiagnosticRows: list[dict[str, Any]] = []
    foldRecords: list[dict[str, Any]] = []
    rhoZByFold = np.zeros(int(folds), dtype=np.float64)
    rhoWeightByFold = np.zeros(int(folds), dtype=np.float64)
    rhoBlockCountByFold = np.zeros(int(folds), dtype=np.int64)
    rhoPairCountByFold = np.zeros(int(folds), dtype=np.int64)
    rhoUpperBound = 0.25

    fitKwargs = dict(runKwargs)
    fitKwargs.setdefault("logRunRole", "delete-block state calibration fold")
    foldIndentLevel = max(0, int(fitKwargs.get("logIndentLevel", 0) or 0))
    fitKwargs["logIndentLevel"] = foldIndentLevel + 1
    if factorModel == segshrink.SEGSHRINK_MODEL:
        calibrationFixedBackgroundIters = max(int(params.calibrationECMIters), 2)
        calibrationOuterIters = min(max(int(params.calibrationOuterIters), 2), 4)
    else:
        calibrationFixedBackgroundIters = max(
            int(params.calibrationECMIters),
            core.UNCERTAINTY_CALIBRATION_MIN_CALIBRATION_ECM_ITERS,
        )
        calibrationOuterIters = max(1, int(params.calibrationOuterIters))
    fitKwargs["ECM_fixedBackgroundIters"] = calibrationFixedBackgroundIters
    fitKwargs["ECM_outerIters"] = calibrationOuterIters
    fitKwargs["ECM_minOuterIters"] = 1
    fitKwargs["processNoiseWarmupECMIters"] = (
        core.UNCERTAINTY_CALIBRATION_REFIT_PROCESS_NOISE_WARMUP_ECM_ITERS
    )
    fitKwargs["returnScales"] = True
    fitKwargs["returnBackground"] = True

    refitSeconds = 0.0
    extractSeconds = 0.0
    maskInformationSeconds = 0.0
    warmupMode = (
        "initialProcessQ"
        if fitKwargs.get("initialProcessQ") is not None
        else "processNoiseWarmup"
    )
    warmupDetail = (
        "initialProcessQ"
        if fitKwargs.get("initialProcessQ") is not None
        else (
            f"{int(core.PROCESS_DEFAULT_WARMUP_OUTER_PASSES)}x"
            f"{int(fitKwargs['processNoiseWarmupECMIters'])}"
        )
    )
    for fold in range(int(folds)):
        logger.info(
            "uncertaintyCalibration.fold.start fold=%s/%s intervals=%s warmupMode=%s warmupDetail=%s",
            int(fold + 1),
            int(folds),
            n,
            warmupMode,
            warmupDetail,
        )
        stageStart = time.perf_counter()
        foldInfo = _cuncertainty.cmakeFoldMaskAndInformation(
            int(m),
            int(n),
            int(blockLen),
            int(fold),
            blockFold,
            repsByBlockCount,
            repsByBlock,
            matrixMunc,
            activeMask,
            totalInfoBase,
            lambdaValues,
            bool(params.deleteBlockUseLambdaInInformation),
            padValue,
            replicateDependenceRho,
            replicateDependenceRho > 0.0,
        )
        if replicateDependenceRho > 0.0:
            mask, keptInfo, heldoutInfo, h, nominalHeldoutInfo = foldInfo
            nominalHeldoutInfo = np.asarray(nominalHeldoutInfo, dtype=np.float64)
        else:
            mask, keptInfo, heldoutInfo, h = foldInfo
            nominalHeldoutInfo = heldoutInfo
        foldMaskInformationSeconds = time.perf_counter() - stageStart
        maskInformationSeconds += foldMaskInformationSeconds
        deletedReplicates = np.sum(mask == 0, axis=0).astype(np.int64, copy=False)
        deletedObservations = np.sum(
            (mask == 0) & (activeMask != 0),
            axis=0,
        ).astype(np.int64, copy=False)
        deletedReplicateIntervalTotal += int(np.sum(deletedReplicates))
        deletedObservationIntervalTotal += int(np.sum(deletedObservations))
        stageStart = time.perf_counter()
        try:
            out = core.runConsenrich(
                matrixData,
                matrixMunc,
                observationMask=np.bitwise_and(mask, activeMask),
                **fitKwargs,
            )
        except Exception as exc:
            logger.warning(
                "uncertaintyCalibration.deleteBlock.fold.failed fold=%s/%s error=%s",
                int(fold + 1),
                int(folds),
                str(exc),
            )
            foldFailures += 1
            foldDiagnosticRows.extend(
                _calibrationKeyValueRows(
                    recordType="fold",
                    event="delete_block_calibration.fold.failed",
                    chromosome=chromosome,
                    fold=int(fold + 1),
                    values={
                        "status": "failed",
                        "error": str(exc),
                        "deleted_replicates": int(np.sum(deletedReplicates)),
                        "deleted_observations": int(np.sum(deletedObservations)),
                        "warmup_mode": warmupMode,
                        "warmup_detail": warmupDetail,
                    },
                )
            )
            continue
        foldRefitSeconds = time.perf_counter() - stageStart
        refitSeconds += foldRefitSeconds
        stateMasked, covarMasked = out[:2]
        stageStart = time.perf_counter()
        stateMaskedArr = np.asarray(stateMasked, dtype=np.float64)
        xMasked = (
            stateMaskedArr[:, 0]
            if stateMaskedArr.ndim == 2
            else stateMaskedArr.reshape(-1)
        )
        covarMaskedArr = np.asarray(covarMasked, dtype=np.float64)
        pMasked = (
            covarMaskedArr[:, 0, 0]
            if covarMaskedArr.ndim == 3
            else covarMaskedArr.reshape(-1)
        )
        if xMasked.shape[0] != n or pMasked.shape[0] != n:
            raise ValueError("masked fold output does not match interval count")
        if len(out) <= 5:
            if targetSignal == "state_plus_background" or replicateDependenceAuto:
                raise ValueError("delete-block calibration refit requires masked background output")
            backgroundMasked = np.zeros(n, dtype=np.float64)
        else:
            backgroundMasked = np.asarray(out[5], dtype=np.float64).reshape(-1)
            if backgroundMasked.shape[0] != n:
                raise ValueError("masked background output must match interval count")
        if replicateDependenceAuto and eligibleReplicateCount >= 2:
            evidence = _cuncertainty.cdeleteBlockReplicateDependenceRhoEvidence(
                matrixData,
                matrixMunc,
                activeMask,
                blockFold,
                repsByBlockCount,
                repsByBlock,
                np.ascontiguousarray(xMasked + backgroundMasked, dtype=np.float64),
                lambdaValues,
                bool(params.deleteBlockUseLambdaInInformation),
                padValue,
                int(blockLen),
                int(fold),
            )
            rhoZByFold[int(fold)] += float(evidence.get("fisher_z_weighted_sum", 0.0))
            rhoWeightByFold[int(fold)] += float(evidence.get("weight_sum", 0.0))
            rhoBlockCountByFold[int(fold)] += int(evidence.get("block_count", 0))
            rhoPairCountByFold[int(fold)] += int(evidence.get("pair_count", 0))
            rhoUpperBound = float(evidence.get("rho_upper_bound", rhoUpperBound))
        foldExtractSeconds = time.perf_counter() - stageStart
        extractSeconds += foldExtractSeconds
        logger.info(
            "uncertaintyCalibration.fold.refit.done fold=%s/%s "
            "deletedReplicates=%s deletedObservations=%s refitSeconds=%.3f "
            "extractSeconds=%.3f",
            int(fold + 1),
            int(folds),
            int(np.sum(deletedReplicates)),
            int(np.sum(deletedObservations)),
            float(foldRefitSeconds),
            float(foldExtractSeconds),
        )
        foldRecords.append(
            {
                "fold": int(fold),
                "mask": mask,
                "xMasked": np.ascontiguousarray(xMasked, dtype=np.float64),
                "pMasked": np.ascontiguousarray(pMasked, dtype=np.float64),
                "backgroundMasked": np.ascontiguousarray(backgroundMasked, dtype=np.float64),
                "deletedReplicates": deletedReplicates,
                "deletedObservations": deletedObservations,
                "refitSeconds": float(foldRefitSeconds),
                "extractSeconds": float(foldExtractSeconds),
            }
        )
    replicateDependenceRhoByFold = np.full(int(folds), float(replicateDependenceRho), dtype=np.float64)
    if replicateDependenceAuto and eligibleReplicateCount >= 2:
        pooledEstimate = _replicateDependenceEstimateFromEvidence(
            zWeightedSum=float(np.sum(rhoZByFold)),
            weightSum=float(np.sum(rhoWeightByFold)),
            blockCount=int(np.sum(rhoBlockCountByFold)),
            pairCount=int(np.sum(rhoPairCountByFold)),
            rhoUpperBound=float(rhoUpperBound),
        )
        replicateDependenceRho = float(pooledEstimate["rho"])
        foldEstimates = []
        for fold in range(int(folds)):
            foldEstimate = _replicateDependenceEstimateFromEvidence(
                zWeightedSum=float(np.sum(rhoZByFold) - rhoZByFold[fold]),
                weightSum=float(np.sum(rhoWeightByFold) - rhoWeightByFold[fold]),
                blockCount=int(np.sum(rhoBlockCountByFold) - rhoBlockCountByFold[fold]),
                pairCount=int(np.sum(rhoPairCountByFold) - rhoPairCountByFold[fold]),
                rhoUpperBound=float(rhoUpperBound),
            )
            foldEstimates.append(foldEstimate)
            replicateDependenceRhoByFold[fold] = float(foldEstimate["rho"])
        replicateDependenceEstimate.update(
            {
                key: pooledEstimate.get(key)
                for key in (
                    "raw_rho",
                    "fisher_z_mean",
                    "fisher_z_shrunk",
                    "fisher_z_se",
                    "block_count",
                    "pair_count",
                    "weight_sum",
                    "rho_upper_bound",
                )
            }
        )
        replicateDependenceEstimate["rho_by_fold"] = [
            float(foldEstimate["rho"]) for foldEstimate in foldEstimates
        ]
        replicateDependenceEstimate["raw_rho_by_fold"] = [
            float(foldEstimate["raw_rho"]) for foldEstimate in foldEstimates
        ]
        replicateDependenceEstimate["fisher_z_mean_by_fold"] = [
            float(foldEstimate["fisher_z_mean"]) for foldEstimate in foldEstimates
        ]
        replicateDependenceEstimate["fisher_z_se_by_fold"] = [
            foldEstimate["fisher_z_se"] for foldEstimate in foldEstimates
        ]
        replicateDependenceEstimate["information_scope"] = "leave_fold_out"
        replicateDependenceEstimate["same_fold_evidence_excluded"] = True
        replicateDependenceEstimate["support_passed"] = bool(
            pooledEstimate["weight_sum"] > 0.0
            and pooledEstimate["block_count"] > 0
            and pooledEstimate["pair_count"] > 0
        )
        logger.info(
            "uncertaintyCalibration.replicateDependence.auto rawRho=%.6g "
            "resolvedRho=%.6g blocks=%s pairs=%s",
            float(pooledEstimate.get("raw_rho", 0.0) or 0.0),
            replicateDependenceRho,
            pooledEstimate.get("block_count"),
            pooledEstimate.get("pair_count"),
        )

    for record in foldRecords:
        fold = int(record["fold"])
        rhoForFold = float(replicateDependenceRhoByFold[fold])
        stageStart = time.perf_counter()
        if rhoForFold > 0.0:
            totalInfoFold = _cuncertainty.cobservationTotalInformation(
                matrixMunc,
                activeMask,
                lambdaValues,
                bool(params.deleteBlockUseLambdaInInformation),
                padValue,
                rhoForFold,
            )
        else:
            totalInfoFold = totalInfoNominal
        foldInfo = _cuncertainty.cmakeFoldMaskAndInformation(
            int(m),
            int(n),
            int(blockLen),
            int(fold),
            blockFold,
            repsByBlockCount,
            repsByBlock,
            matrixMunc,
            activeMask,
            totalInfoFold,
            lambdaValues,
            bool(params.deleteBlockUseLambdaInInformation),
            padValue,
            rhoForFold,
            rhoForFold > 0.0,
        )
        if rhoForFold > 0.0:
            _mask, keptInfo, heldoutInfo, h, nominalHeldoutInfo = foldInfo
            nominalHeldoutInfo = np.asarray(nominalHeldoutInfo, dtype=np.float64)
        else:
            _mask, keptInfo, heldoutInfo, h = foldInfo
            nominalHeldoutInfo = heldoutInfo
        maskInformationSeconds += time.perf_counter() - stageStart

        stageStart = time.perf_counter()
        xMasked = np.asarray(record["xMasked"], dtype=np.float64)
        pMasked = np.asarray(record["pMasked"], dtype=np.float64)
        backgroundMasked = np.asarray(record["backgroundMasked"], dtype=np.float64)
        if targetSignal == "state_plus_background":
            signalMasked = xMasked + backgroundMasked
            signalFull = fullState0 + fullBackgroundArr
        else:
            signalMasked = xMasked
            signalFull = fullState0
        stateDelta = signalMasked - signalFull
        deltaVariance, sourceCode, valid, invalidReasonCode = _chooseDeleteBlockDeltaVariance(
            fullPArr,
            pMasked,
            h,
            mode=varianceMode,
            minDeltaVariance=float(params.deleteBlockMinDeltaVariance),
            minInformationFraction=float(params.deleteBlockMinInformationFraction),
            maxInformationFraction=float(params.deleteBlockMaxInformationFraction),
            positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        invalidReasonCode = np.asarray(invalidReasonCode, dtype=np.uint8)
        meaningfulDeletion = (
            np.isfinite(totalInfoFold)
            & (totalInfoFold > 0.0)
            & np.isfinite(heldoutInfo)
            & (heldoutInfo > 0.0)
        )
        invalidReasonCode[~meaningfulDeletion] = _DELETE_BLOCK_REASON_NO_DELETED_INFORMATION
        valid &= meaningfulDeletion
        finiteDelta = np.isfinite(stateDelta)
        invalidReasonCode[~finiteDelta] = _DELETE_BLOCK_REASON_NONFINITE_STATE_DELTA
        valid &= finiteDelta
        invalidReasonCode[valid] = _DELETE_BLOCK_REASON_VALID
        rowsTotal += int(n)
        invalidReasonCountByCode += np.bincount(
            invalidReasonCode[~valid].astype(np.int64, copy=False),
            minlength=DELETE_BLOCK_INVALID_REASON_LABELS.shape[0],
        )[: DELETE_BLOCK_INVALID_REASON_LABELS.shape[0]]
        foldExtractSeconds = time.perf_counter() - stageStart
        extractSeconds += foldExtractSeconds
        deletedReplicates = np.asarray(record["deletedReplicates"], dtype=np.int64)
        deletedObservations = np.asarray(record["deletedObservations"], dtype=np.int64)
        if not np.any(valid):
            logger.info(
                "uncertaintyCalibration.fold.done fold=%s/%s deleteBlockRows=0 "
                "deletedReplicates=%s deletedObservations=%s refitSeconds=%.3f "
                "extractSeconds=%.3f",
                int(fold + 1),
                int(folds),
                int(np.sum(deletedReplicates)),
                int(np.sum(deletedObservations)),
                float(record["refitSeconds"]),
                float(foldExtractSeconds),
            )
            foldDiagnosticRows.extend(
                _calibrationKeyValueRows(
                    recordType="fold",
                    event="delete_block_calibration.fold.done",
                    chromosome=chromosome,
                    fold=int(fold + 1),
                    values={
                        "status": "no_valid_rows",
                        "delete_block_rows": 0,
                        "deleted_replicates": int(np.sum(deletedReplicates)),
                        "deleted_observations": int(np.sum(deletedObservations)),
                        "replicate_dependence_rho": float(rhoForFold),
                        "refit_seconds": float(record["refitSeconds"]),
                        "extract_seconds": float(foldExtractSeconds),
                    },
                )
            )
            continue
        idx = np.flatnonzero(valid).astype(np.int64, copy=False)
        residualChunks.append(np.ascontiguousarray(stateDelta[idx], dtype=np.float64))
        pDeltaChunks.append(np.ascontiguousarray(deltaVariance[idx], dtype=np.float64))
        iChunks.append(idx.astype(np.int64, copy=False))
        foldChunks.append(np.full(idx.shape[0], int(fold), dtype=np.int32))
        hChunks.append(np.ascontiguousarray(h[idx], dtype=np.float64))
        sourceCodeChunks.append(np.ascontiguousarray(sourceCode[idx], dtype=np.uint8))
        rowWeightChunks.append(_deleteBlockRowWeights(h[idx], params))
        stateMaskedChunks.append(
            np.ascontiguousarray(signalMasked[idx], dtype=np.float64)
        )
        pMaskedChunks.append(np.ascontiguousarray(pMasked[idx], dtype=np.float64))
        covDeltaChunks.append(
            np.ascontiguousarray(pMasked[idx] - fullPArr[idx], dtype=np.float64)
        )
        totalInfoChunks.append(np.ascontiguousarray(totalInfoFold[idx], dtype=np.float64))
        keptInfoChunks.append(np.ascontiguousarray(keptInfo[idx], dtype=np.float64))
        heldInfoChunks.append(np.ascontiguousarray(heldoutInfo[idx], dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            totalDeffChunks.append(
                np.ascontiguousarray(
                    totalInfoNominal[idx] / totalInfoFold[idx],
                    dtype=np.float64,
                )
            )
            heldoutDeffChunks.append(
                np.ascontiguousarray(
                    nominalHeldoutInfo[idx] / heldoutInfo[idx],
                    dtype=np.float64,
                )
            )
        deletedReplicateChunks.append(
            np.ascontiguousarray(deletedReplicates[idx], dtype=np.int64)
        )
        deletedObservationChunks.append(
            np.ascontiguousarray(deletedObservations[idx], dtype=np.int64)
        )
        sourceCountByCodeFold = np.bincount(
            sourceCode[idx].astype(np.int64, copy=False),
            minlength=DELETE_BLOCK_VARIANCE_SOURCE_LABELS.shape[0],
        )
        covValidFractionFold = float(
            sourceCountByCodeFold[int(_DELETE_BLOCK_SOURCE_COVARIANCE_DIFFERENCE)]
            / max(int(idx.size), 1)
        )
        logger.info(
            "uncertaintyCalibration.fold.done fold=%s/%s deleteBlockRows=%s "
            "deletedReplicates=%s deletedObservations=%s varianceMode=%s "
            "covarianceDifferenceFraction=%.3f replicateDependenceRho=%.6g "
            "refitSeconds=%.3f extractSeconds=%.3f",
            int(fold + 1),
            int(folds),
            int(idx.size),
            int(np.sum(deletedReplicates)),
            int(np.sum(deletedObservations)),
            varianceMode,
            covValidFractionFold,
            float(rhoForFold),
            float(record["refitSeconds"]),
            float(foldExtractSeconds),
        )
        foldDiagnosticRows.extend(
            _calibrationKeyValueRows(
                recordType="fold",
                event="delete_block_calibration.fold.done",
                chromosome=chromosome,
                fold=int(fold + 1),
                values={
                    "status": "ok",
                    "delete_block_rows": int(idx.size),
                    "deleted_replicates": int(np.sum(deletedReplicates)),
                    "deleted_observations": int(np.sum(deletedObservations)),
                    "variance_mode": varianceMode,
                    "covariance_difference_fraction": covValidFractionFold,
                    "replicate_dependence_rho": float(rhoForFold),
                    "refit_seconds": float(record["refitSeconds"]),
                    "extract_seconds": float(foldExtractSeconds),
                },
            )
        )

    timings["masked_refits_seconds"] = refitSeconds
    timings["extract_scores_seconds"] = extractSeconds
    timings["mask_information_seconds"] = maskInformationSeconds

    if not residualChunks:
        raise ValueError("delete-block state uncertainty calibration produced no valid deleted-state rows")
    residual = np.concatenate(residualChunks)
    pDelta = np.concatenate(pDeltaChunks)
    intervalIndex = np.concatenate(iChunks)
    foldIndex = np.concatenate(foldChunks)
    hAll = np.concatenate(hChunks)
    sourceCodeAll = np.concatenate(sourceCodeChunks).astype(np.uint8, copy=False)
    rowWeight = np.concatenate(rowWeightChunks)
    stateMaskedAll = np.concatenate(stateMaskedChunks)
    pMaskedAll = np.concatenate(pMaskedChunks)
    covDeltaAll = np.concatenate(covDeltaChunks)
    totalInfoAll = np.concatenate(totalInfoChunks)
    keptInfoAll = np.concatenate(keptInfoChunks)
    heldInfoAll = np.concatenate(heldInfoChunks)
    totalDeffAll = np.concatenate(totalDeffChunks)
    heldoutDeffAll = np.concatenate(heldoutDeffChunks)
    deletedReplicateAll = np.concatenate(deletedReplicateChunks)
    deletedObservationAll = np.concatenate(deletedObservationChunks)
    blockIndex = (intervalIndex // int(blockLen)).astype(np.int64, copy=False)
    deletedBlockCount = int(np.sum(deleteCountsByBlock > 0))
    deletedReplicateBlockTotal = int(np.sum(deleteCountsByBlock))
    totalDeleteBlockRows = int(residual.size)
    if residual.size < int(params.minHeldoutCells):
        logger.warning(
            "uncertaintyCalibration.lowDeleteBlockRows deleteBlockRows=%s minHeldoutCells=%s; fitting with available rows",
            int(residual.size),
            int(params.minHeldoutCells),
        )
    targetDelta = _targetCalibrationDelta(params)
    targetCalibrationEnabled = targetDelta is not None
    targetSplit = _targetCalibrationSplit(
        blockIndex,
        enabled=targetCalibrationEnabled,
        seed=int(params.seed) + TARGET_CALIBRATION_BLOCK_SPLIT_SEED_OFFSET,
    )
    scaleRows = np.flatnonzero(np.asarray(targetSplit["scale_mask"], dtype=bool))
    if scaleRows.size == 0:
        scaleRows = np.arange(residual.size, dtype=np.int64)
    sampleCodes = _deleteBlockScoreSamplingCodes(
        foldIndex=foldIndex,
        intervalIndex=intervalIndex,
        pDelta=pDelta,
        fullState=fullTargetSignal,
        sourceCode=sourceCodeAll,
    )
    fitRowsLocal = _samplePositionsByCode(
        sampleCodes[scaleRows],
        maxRows=_maxScoreRows(params),
        seed=int(params.seed),
    )
    fitRows = scaleRows[fitRowsLocal]
    residualFit = residual[fitRows]
    pDeltaFit = pDelta[fitRows]
    intervalIndexFit = intervalIndex[fitRows]
    foldIndexFit = foldIndex[fitRows]
    logger.info(
        "uncertaintyCalibration.sample deleteBlockRows=%s fitRows=%s maxScores=%s",
        totalDeleteBlockRows,
        int(residualFit.size),
        _maxScoreRows(params),
    )
    coverageCodeAll, _coverageNameAll = _signalLevelCoverageCodes(
        np.abs(fullTargetSignal[intervalIndex])
    )
    coverageCodeFit, _coverageNameFit = _signalLevelCoverageCodes(
        np.abs(fullTargetSignal[intervalIndexFit])
    )
    try:
        uncertaintyDecile = np.asarray(
            pd.qcut(
                pDeltaFit,
                q=core.UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES,
                labels=False,
                duplicates="drop",
            ),
            dtype=np.float64,
        )
    except ValueError:
        uncertaintyDecile = np.zeros(pDeltaFit.shape[0], dtype=np.float64)
    summaryDecile = np.nan_to_num(uncertaintyDecile, nan=-1.0).astype(np.int32)
    replayData = _validateCalibrationReplayArrays(
        {
            "residual": np.ascontiguousarray(residual, dtype=np.float64),
            "pDelta": np.ascontiguousarray(pDelta, dtype=np.float64),
            "intervalIndex": np.ascontiguousarray(intervalIndex, dtype=np.int64),
            "fitRows": np.ascontiguousarray(fitRows, dtype=np.int64),
            "targetBlockMask": np.ascontiguousarray(
                targetSplit["target_block_mask"],
                dtype=np.uint8,
            ),
            "deletedObservationAll": np.ascontiguousarray(
                deletedObservationAll,
                dtype=np.int64,
            ),
            "coverageCodeAll": np.ascontiguousarray(coverageCodeAll, dtype=np.int32),
            "coverageCodeFit": np.ascontiguousarray(coverageCodeFit, dtype=np.int32),
            "summaryDecile": np.ascontiguousarray(summaryDecile, dtype=np.int32),
        },
        intervalCount=n,
        blockLenIntervals=int(blockLen),
        positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )
    deleteBlockApplyTarget = getattr(params, "deleteBlockApplyTargetCalibration", None)
    scaleByTargetCalibration = bool(
        params.scaleUncertaintyByTargetCalibration
        if deleteBlockApplyTarget is None
        else deleteBlockApplyTarget
    )
    stageStart = time.perf_counter()
    segShrinkFit: dict[str, Any] | None = None
    if factorModel == segshrink.SEGSHRINK_MODEL:
        targetForFactor = max(tuple(float(t) for t in params.targets))
        factorMin, factorMax = _factorBounds(params)
        segShrinkFit = segshrink.fitSingleContig(
            residual=residualFit,
            pDelta=pDeltaFit,
            rowWeight=rowWeight[fitRows],
            intervalIndex=intervalIndexFit,
            foldIndex=foldIndexFit,
            blockIDX=blockIndex[fitRows],
            fullP=fullPArr,
            target=targetForFactor,
            targetZ=_normalZ(targetForFactor),
            factorMin=factorMin,
            factorMax=factorMax,
            segmentCount=int(params.deleteBlockFactorSegmentCount),
            bootstrapReplicates=int(params.deleteBlockFactorBootstrapReplicates),
            seed=int(params.seed) + core.UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET,
            positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        modelMeta = dict(segShrinkFit["modelMeta"])
        segmentByInterval = np.asarray(
            segShrinkFit["segmentByInterval"],
            dtype=np.int32,
        )
        segmentFactorRaw = np.asarray(
            [row["factor"] for row in modelMeta["segmentShrinkage"]],
            dtype=np.float64,
        )
        if (
            segmentByInterval.shape != fullPArr.shape
            or segmentFactorRaw.size < 1
            or np.any(segmentByInterval < 0)
            or np.any(segmentByInterval >= segmentFactorRaw.size)
        ):
            raise ValueError("segShrink fitted segments do not match the interval factor")
        factorRaw = segmentFactorRaw[segmentByInterval]
        factorGlobal = float(modelMeta.get("global_factor", np.nan))
    else:
        factorGlobal, modelMeta = _fitDeleteBlockGlobalFactor(
            residual=residualFit,
            pDelta=pDeltaFit,
            rowWeight=rowWeight[fitRows],
            params=params,
        )
        factorRaw = np.full(n, float(factorGlobal), dtype=np.float64)
    timings["fit_factor_seconds"] = time.perf_counter() - stageStart
    modelMeta["refitPolicy"] = {
        "ECM_outerIters": int(calibrationOuterIters),
        "ECM_minOuterIters": 1,
        "ECM_fixedBackgroundIters": int(calibrationFixedBackgroundIters),
        "processNoiseWarmupECMIters": int(
            core.UNCERTAINTY_CALIBRATION_REFIT_PROCESS_NOISE_WARMUP_ECM_ITERS
        ),
    }
    stageStart = time.perf_counter()
    evaluated = _evaluateDeleteBlockCalibration(
        factorRaw=np.ascontiguousarray(factorRaw, dtype=np.float64),
        fullP=np.ascontiguousarray(fullPArr, dtype=np.float64),
        residual=replayData["residual"],
        pDelta=replayData["pDelta"],
        intervalIndex=replayData["intervalIndex"],
        fitRows=replayData["fitRows"],
        blockIndex=np.ascontiguousarray(blockIndex, dtype=np.int64),
        targetBlockMask=replayData["targetBlockMask"],
        deletedObservationAll=replayData["deletedObservationAll"],
        coverageCodeAll=replayData["coverageCodeAll"],
        coverageCodeFit=replayData["coverageCodeFit"],
        summaryDecile=replayData["summaryDecile"],
        targets=tuple(float(target) for target in params.targets),
        targetCalibrationEnabled=targetCalibrationEnabled,
        targetDelta=targetDelta,
        scaleByTargetCalibration=scaleByTargetCalibration,
        positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )
    factor = evaluated["factor"]
    calibrated = evaluated["calibrated"]
    summary = evaluated["summary"]
    stateCoverage = evaluated["stateCoverage"]
    stateCoverageFit = evaluated["stateCoverageFit"]
    sdBeforeAll = evaluated["sdBeforeAll"]
    sdAfterAll = evaluated["sdAfterAll"]
    sdBefore = evaluated["sdBeforeFit"]
    sdAfter = evaluated["sdAfterFit"]
    heldFactor = evaluated["heldFactorFit"]
    targetBlockIds = evaluated["targetBlockIds"]
    targetBlockScores = evaluated["targetBlockScores"]
    targetBlockCellCounts = evaluated["targetBlockCellCounts"]
    targetCalibrationMetadata = evaluated["targetCalibration"]
    targetCalibrationBounds = targetCalibrationMetadata["bounds"]
    modelSEFloorHits = int(evaluated["modelSEFloorHits"])
    deleteBlockFactorDistribution = evaluated["factorDistribution"]
    timings["evaluate_factor_seconds"] = time.perf_counter() - stageStart
    if calibrationReplayPath is not None:
        _writeCalibrationReplay(calibrationReplayPath, replayData)
    if calibrationReplayPath is None:
        targetLog = (
            logger.warning
            if (
                targetCalibrationEnabled
                and scaleByTargetCalibration
                and (
                    not targetCalibrationMetadata["uncertainty_track_scaled"]
                    or not targetCalibrationMetadata[
                        "uncertainty_track_scale_bound_available"
                    ]
                )
            )
            else logger.info
        )
        targetLog(
            "uncertaintyCalibration.target enabled=%s delta=%s deletionProbability=%.6g blocksTotal=%d blocksWithDeletion=%d blocksScale=%d blocksTarget=%d blocksTargetScored=%d targetBlockCells=%d selectedTarget=%s targetZ=%s q=%s qSource=%s boundAvailable=%s boundScope=%s scaleRequested=%s scaleApplied=%s scale=%.6g reason=%s",
            bool(targetCalibrationEnabled),
            None if targetDelta is None else float(targetDelta),
            float(deletionProbability),
            int(targetCalibrationMetadata["blocks_total"]),
            int(deletedBlockCount),
            int(targetCalibrationMetadata["blocks_scale"]),
            int(targetCalibrationMetadata["blocks_target"]),
            int(targetCalibrationMetadata["blocks_target_scored"]),
            int(targetCalibrationMetadata["target_block_cells"]),
            targetCalibrationMetadata["uncertainty_track_scale_target"],
            targetCalibrationMetadata["uncertainty_track_scale_target_z"],
            targetCalibrationMetadata["uncertainty_track_scale_q"],
            targetCalibrationMetadata["uncertainty_track_scale_q_source"],
            bool(
                targetCalibrationMetadata["uncertainty_track_scale_bound_available"]
            ),
            targetCalibrationMetadata["uncertainty_track_scale_bound_scope"],
            bool(scaleByTargetCalibration),
            bool(targetCalibrationMetadata["uncertainty_track_scaled"]),
            float(targetCalibrationMetadata["uncertainty_track_scale"]),
            targetCalibrationMetadata["uncertainty_track_scale_reason"],
        )
        coverageOverall = [
            row for row in stateCoverage if str(row.get("stratum", "")) == "overall"
        ]
        coverageFitOverall = [
            row
            for row in stateCoverageFit
            if str(row.get("stratum", "")) == "overall"
        ]
        if coverageOverall:
            logger.info(
                "uncertaintyCalibration.coverage.delete_block_all %s",
                _coverageLogPayload(coverageOverall),
            )
        if coverageFitOverall:
            logger.info(
                "uncertaintyCalibration.coverage.fit_sample %s",
                _coverageLogPayload(coverageFitOverall),
            )
    stageStart = time.perf_counter()
    sourceCodeFit = sourceCodeAll[fitRows]
    sourceFit = DELETE_BLOCK_VARIANCE_SOURCE_LABELS[
        sourceCodeFit.astype(np.int64, copy=False)
    ]
    hFit = hAll[fitRows]
    pMaskedFit = pMaskedAll[fitRows]
    covDeltaFit = covDeltaAll[fitRows]
    totalInfoFit = totalInfoAll[fitRows]
    keptInfoFit = keptInfoAll[fitRows]
    heldInfoFit = heldInfoAll[fitRows]
    deletedReplicateFit = deletedReplicateAll[fitRows]
    deletedObservationFit = deletedObservationAll[fitRows]
    rowWeightFit = rowWeight[fitRows]
    pDeltaFit = pDelta[fitRows]
    blockIndexFit = blockIndex[fitRows]
    stateMaskedFit = stateMaskedAll[fitRows]
    stateAbsAll = np.abs(fullTargetSignal)
    highSignalCut = (
        float(
            np.nanquantile(
                stateAbsAll,
                core.UNCERTAINTY_CALIBRATION_SCORE_STATE_ABS_QUANTILE,
            )
        )
        if stateAbsAll.size
        else np.inf
    )
    highSignalFit = stateAbsAll[intervalIndexFit] >= highSignalCut
    diagnosticRows = _samplePositionsByCode(
        _deleteBlockScoreSamplingCodes(
            foldIndex=foldIndexFit,
            intervalIndex=intervalIndexFit,
            pDelta=pDeltaFit,
            fullState=fullTargetSignal,
            sourceCode=sourceCodeFit,
        ),
        maxRows=int(params.maxDiagnosticRows),
        seed=int(params.seed) + core.UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET,
    )
    scoreData: dict[str, Any] = {
        "fold": foldIndexFit,
        "interval_index": intervalIndexFit,
        "block_index": blockIndexFit,
        "chrom_start": intervalsArr[intervalIndexFit],
        "residual": residualFit,
        "deleted_target_signal_delta": residualFit,
        "target_signal_full": fullTargetSignal[intervalIndexFit],
        "target_signal_masked": stateMaskedFit,
        "P00_full": fullPArr[intervalIndexFit],
        "P00_masked": pMaskedFit,
        "covariance_delta": covDeltaFit,
        "total_information": totalInfoFit,
        "kept_information": keptInfoFit,
        "heldout_information": heldInfoFit,
        "heldout_information_fraction": hFit,
        "deleted_replicates": deletedReplicateFit,
        "deleted_observations": deletedObservationFit,
        "delta_variance": pDeltaFit,
        "delta_variance_source": sourceFit,
        "row_weight": rowWeightFit,
        "sd_before": sdBefore,
        "sd_after": sdAfter,
        "a_state": heldFactor,
        "uncertainty_decile": uncertaintyDecile,
        "high_signal": highSignalFit,
    }
    if segShrinkFit is not None:
        segmentByInterval = np.asarray(segShrinkFit["segmentByInterval"], dtype=np.int32)
        fitSegment = segmentByInterval[intervalIndexFit]
        segmentRawLog = np.asarray(
            segShrinkFit["segmentRawLogFactor"],
            dtype=np.float64,
        )
        segmentRawFactor = np.full(fitSegment.shape[0], np.nan, dtype=np.float64)
        validSegment = (fitSegment >= 0) & (fitSegment < segmentRawLog.shape[0])
        segmentRawFactor[validSegment] = np.exp(segmentRawLog[fitSegment[validSegment]])
        segmentVariance = np.asarray(
            segShrinkFit["segmentBootstrapVariance"],
            dtype=np.float64,
        )
        segmentBootstrapVariance = np.full(
            fitSegment.shape[0],
            np.nan,
            dtype=np.float64,
        )
        validVarianceSegment = (fitSegment >= 0) & (fitSegment < segmentVariance.shape[0])
        segmentBootstrapVariance[validVarianceSegment] = segmentVariance[
            fitSegment[validVarianceSegment]
        ]
        segmentWeight = np.asarray(
            segShrinkFit["segmentShrinkageWeight"],
            dtype=np.float64,
        )
        segmentShrinkageWeight = np.full(fitSegment.shape[0], np.nan, dtype=np.float64)
        validWeightSegment = (fitSegment >= 0) & (fitSegment < segmentWeight.shape[0])
        segmentShrinkageWeight[validWeightSegment] = segmentWeight[
            fitSegment[validWeightSegment]
        ]
        contigShrinkageWeight = float(
            modelMeta.get("contigShrinkage", [{}])[0].get("shrinkageWeight", 0.0)
        )
        scoreData.update(
            {
                "factor_segment": fitSegment,
                "blockIDX": blockIndexFit.astype(np.int64, copy=False),
                "segment_raw_factor": segmentRawFactor,
                "segment_bootstrap_variance": segmentBootstrapVariance,
                "segment_shrinkage_weight": segmentShrinkageWeight,
                "contig_shrinkage_weight": contigShrinkageWeight,
            }
        )
    scores = pd.DataFrame(scoreData)
    scoresDiagnostics = scores.iloc[diagnosticRows, :].reset_index(drop=True)
    timings["summarize_scores_seconds"] = time.perf_counter() - stageStart
    sourceCountByCode = np.bincount(
        sourceCodeAll.astype(np.int64, copy=False),
        minlength=DELETE_BLOCK_VARIANCE_SOURCE_LABELS.shape[0],
    )
    sourceCounts = {
        "covariance_difference": int(
            sourceCountByCode[int(_DELETE_BLOCK_SOURCE_COVARIANCE_DIFFERENCE)]
        ),
        "heldout_information": int(
            sourceCountByCode[int(_DELETE_BLOCK_SOURCE_HELDOUT_INFORMATION)]
        ),
        "heldout_information_fallback": int(
            sourceCountByCode[int(_DELETE_BLOCK_SOURCE_HELDOUT_INFORMATION_FALLBACK)]
        ),
    }
    invalidReasonCounts = {
        label: int(invalidReasonCountByCode[int(code)])
        for code, label in _DELETE_BLOCK_INVALID_REASON_COUNT_CODES
    }
    covarianceDifferenceValidFraction = float(
        sourceCounts["covariance_difference"] / max(totalDeleteBlockRows, 1)
    )
    if (
        varianceMode == "hybrid"
        and totalDeleteBlockRows > 0
        and covarianceDifferenceValidFraction
        < float(params.deleteBlockFallbackMinValidFraction)
    ):
        logger.warning(
            "uncertaintyCalibration.deleteBlock.mostlyInformationFallback covarianceDifferenceFraction=%.3f minValidFraction=%.3f",
            covarianceDifferenceValidFraction,
            float(params.deleteBlockFallbackMinValidFraction),
        )
    finiteH = hAll[np.isfinite(hAll)]
    finiteTotalInfo = totalInfoAll[np.isfinite(totalInfoAll) & (totalInfoAll > 0.0)]
    finiteHeldInfo = heldInfoAll[np.isfinite(heldInfoAll) & (heldInfoAll > 0.0)]
    finiteTotalDeff = totalDeffAll[
        np.isfinite(totalDeffAll) & (totalDeffAll > 0.0)
    ]
    finiteHeldoutDeff = heldoutDeffAll[
        np.isfinite(heldoutDeffAll) & (heldoutDeffAll > 0.0)
    ]
    factorMin, factorMax = _factorBounds(params)
    factorOut = np.asarray(factor, dtype=np.float32)
    model = {
        **modelMeta,
        "mode": calibrationMode,
        "score_definition": _PERTURBATION_SCORE_DEFINITION,
        "coverage_estimand": _COVERAGE_ESTIMAND,
        "coverage_scope": _COVERAGE_SCOPE,
        "coverage_fit_scope": _FIT_COVERAGE_SCOPE,
        "target_signal": targetSignal,
        "variance_mode": varianceMode,
        "factor_model": factorModel,
        "calibration_policy": calibrationMode,
        "feature_names": featureNames,
        "feature_center": [float(x) for x in featureCenter],
        "feature_scale": [float(x) for x in featureScale],
        "factor_bound_min": float(factorMin),
        "factor_bound_max": float(factorMax),
        "delete_block_factor_distribution": deleteBlockFactorDistribution,
        "model_se_floor_applied": True,
        "model_se_floor_hits": modelSEFloorHits,
        "delete_block_deletion_probability": float(deletionProbability),
        "delete_block_deleted_blocks": deletedBlockCount,
        "delete_block_deleted_replicate_block_total": deletedReplicateBlockTotal,
        "delete_block_deleted_replicate_interval_total": int(
            deletedReplicateIntervalTotal
        ),
        "delete_block_deleted_observation_interval_total": int(
            deletedObservationIntervalTotal
        ),
        "rows_total": int(rowsTotal),
        "rows_valid": int(totalDeleteBlockRows),
        "rows_fit": int(residualFit.size),
        "rows_invalid": int(sum(invalidReasonCounts.values())),
        "invalid_reasons": {key: int(value) for key, value in invalidReasonCounts.items()},
        "variance_source_counts": sourceCounts,
        "covariance_difference_valid_fraction": covarianceDifferenceValidFraction,
        "information": {
            "use_lambda": bool(params.deleteBlockUseLambdaInInformation),
            "weight_mode": weightMode,
            "min_h": None if finiteH.size == 0 else float(np.min(finiteH)),
            "median_h": None if finiteH.size == 0 else float(np.median(finiteH)),
            "mean_h": None if finiteH.size == 0 else float(np.mean(finiteH)),
            "max_h": None if finiteH.size == 0 else float(np.max(finiteH)),
            "total_information_median": (
                None if finiteTotalInfo.size == 0 else float(np.median(finiteTotalInfo))
            ),
            "heldout_information_median": (
                None if finiteHeldInfo.size == 0 else float(np.median(finiteHeldInfo))
            ),
        },
        "replicate_dependence": {
            "method": "exchangeable",
            "source": replicateDependenceEstimate["source"],
            "estimator": replicateDependenceEstimate["estimator"],
            "rho": float(replicateDependenceRho),
            "applied": bool(np.any(replicateDependenceRhoByFold > 0.0)),
            "raw_rho": replicateDependenceEstimate["raw_rho"],
            "fisher_z_mean": replicateDependenceEstimate["fisher_z_mean"],
            "fisher_z_shrunk": replicateDependenceEstimate["fisher_z_shrunk"],
            "fisher_z_se": replicateDependenceEstimate["fisher_z_se"],
            "block_count": replicateDependenceEstimate["block_count"],
            "pair_count": replicateDependenceEstimate["pair_count"],
            "weight_sum": replicateDependenceEstimate["weight_sum"],
            "rho_upper_bound": replicateDependenceEstimate["rho_upper_bound"],
            "rho_by_fold": replicateDependenceEstimate.get("rho_by_fold"),
            "raw_rho_by_fold": replicateDependenceEstimate.get("raw_rho_by_fold"),
            "fisher_z_mean_by_fold": replicateDependenceEstimate.get(
                "fisher_z_mean_by_fold"
            ),
            "fisher_z_se_by_fold": replicateDependenceEstimate.get(
                "fisher_z_se_by_fold"
            ),
            "information_scope": replicateDependenceEstimate.get("information_scope"),
            "same_fold_evidence_excluded": replicateDependenceEstimate.get(
                "same_fold_evidence_excluded"
            ),
            "support_passed": replicateDependenceEstimate.get("support_passed"),
            "total_deff_median": (
                None
                if finiteTotalDeff.size == 0
                else float(np.median(finiteTotalDeff))
            ),
            "heldout_deff_median": (
                None
                if finiteHeldoutDeff.size == 0
                else float(np.median(finiteHeldoutDeff))
            ),
        },
        "fold_refits": {
            "folds": int(folds),
            "fold_failures": int(foldFailures),
            "block_len_intervals": int(blockLen),
            "delete_block_deletion_probability": float(deletionProbability),
            "blocks_total": int(deleteCountsByBlock.size),
            "blocks_with_deletion": deletedBlockCount,
            "deleted_replicate_block_total": deletedReplicateBlockTotal,
            "deleted_replicate_interval_total": int(deletedReplicateIntervalTotal),
            "deleted_observation_interval_total": int(deletedObservationIntervalTotal),
            "deleted_replicate_count_min": int(np.min(deleteCountsByBlock)),
            "deleted_replicate_count_mean": float(np.mean(deleteCountsByBlock)),
            "deleted_replicate_count_max": int(np.max(deleteCountsByBlock)),
            "calibration_ecm_iters": int(
                calibrationFixedBackgroundIters
                if factorModel == segshrink.SEGSHRINK_MODEL
                else params.calibrationECMIters
            ),
            "calibration_outer_iters": int(calibrationOuterIters),
            "refit_policy": {
                "ECM_outerIters": int(calibrationOuterIters),
                "ECM_minOuterIters": 1,
                **(
                    {
                        "ECM_fixedBackgroundIters": int(
                            calibrationFixedBackgroundIters
                        )
                    }
                    if factorModel == segshrink.SEGSHRINK_MODEL
                    else {}
                ),
                "returnBackground": True,
                "returnScales": True,
            },
        },
        "state_roughness": stateRoughness,
        "diagnostic_score_rows": int(scoresDiagnostics.shape[0]),
        "max_scores": int(_maxScoreRows(params)),
        "max_diagnostic_rows": int(params.maxDiagnosticRows),
        "folds": int(folds),
        "block_len_intervals": int(blockLen),
        "block_size_bp": (
            None
            if params.blockSizeBP is None
            else str(params.blockSizeBP)
        ),
        "targets": [float(target) for target in params.targets],
        "ridge": float(max(params.ridge, 0.0)),
        "state_uncertainty_coverage": stateCoverage,
        "state_uncertainty_coverage_fit": stateCoverageFit,
        "target_calibration": {
            "block_split_seed": int(targetSplit["seed"]),
            **targetCalibrationMetadata,
        },
    }
    timings["total_seconds"] = time.perf_counter() - totalStart
    model["timings_seconds"] = {key: float(value) for key, value in timings.items()}
    if calibrationReplayPath is None:
        logger.info(
            "uncertaintyCalibration.fit.done mode=delete_block_state deleteBlockRows=%s fitRows=%s globalFactor=%.6g elapsed=%.3fs",
            totalDeleteBlockRows,
            int(residualFit.size),
            float(factorGlobal),
            timings["total_seconds"],
        )
    if (diagnosticsLogPath is not None or outPrefix is not None) and bool(
        params.writeDiagnostics
    ):
        diagnosticsStart = time.perf_counter()
        logPath = _ensureCalibrationLog(
            diagnosticsLogPath
            if diagnosticsLogPath is not None
            else str(Path(str(outPrefix))) + ".delete_block_calibration.jsonl"
        )
        diagnosticsRecords = (
            []
            if calibrationReplayPath is not None
            else _diagnosticsRecords(
                scores=scoresDiagnostics,
                summary=summary,
                model=model,
                chromosome=chromosome,
            )
        )
        rowsWritten = _appendJsonlRecords(logPath, diagnosticsRecords)
        extraRows: list[dict[str, Any]] = []
        extraRows.extend(foldDiagnosticRows)
        extraRows.extend(
            _calibrationKeyValueRows(
                recordType="invalid_reason",
                event="delete_block_calibration.invalid_reason_counts",
                chromosome=chromosome,
                values={key: int(value) for key, value in invalidReasonCounts.items()},
            )
        )
        if calibrationReplayPath is None:
            extraRows.extend(
                {
                    "record_type": "target_bound",
                    "event": "delete_block_calibration.target_bound",
                    "chromosome": chromosome,
                    **dict(bound),
                }
                for bound in targetCalibrationBounds
            )
        rowsWritten += _appendJsonlRecords(logPath, extraRows)
        timings["diagnostics_seconds"] = time.perf_counter() - diagnosticsStart
        model["timings_seconds"] = {key: float(value) for key, value in timings.items()}
        _logging_utils.log_file_written(
            logger,
            event="uncertainty.delete_block_calibration_log",
            path=str(logPath),
            fields=(("chromosome", chromosome), ("rows", int(rowsWritten))),
            level=logging.INFO,
        )
    return uncertaintyCalibrationResult(
        factor=factorOut,
        calibratedUncertainty=calibrated,
        summary=summary,
        scores=scores,
        model=model,
    )


__all__ = [
    "calibrateChromosomeStateUncertainty",
    "uncertaintyCalibrationResult",
]
