"""State-uncertainty calibration helpers."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from . import core
from . import diagnostics
from . import segshrink
from . import cuncertainty as _cuncertainty
from . import _logging as _logging_utils
from ._normalization import weighted_quantile


logger = logging.getLogger(__name__)


TARGET_CALIBRATION_BLOCK_SPLIT_SEED_OFFSET = 20_000
TARGET_CALIBRATION_FRACTION = 0.5

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
    "alpha",
    "delta",
    "q",
    "q_source",
    "k",
    "tail_probability",
    "finite_bound",
    "certified",
    "reason",
    "n",
    "coverage_before",
    "coverage_after",
    "mean_width_before",
    "mean_width_after",
    "median_width_before",
    "median_width_after",
    "q90_width_before",
    "q90_width_after",
    "residual",
    "deleted_state_delta",
    "state_full",
    "state_masked",
    "P00_full",
    "P00_masked",
    "covariance_delta",
    "total_information",
    "kept_information",
    "heldout_information",
    "heldout_information_fraction",
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


def _progressEnabled(params: core.uncertaintyCalibrationParams) -> bool:
    return bool(params.writeDiagnostics) and _logging_utils.progress_enabled()


def _progress(iterable, *, params: core.uncertaintyCalibrationParams, **kwargs):
    if not _progressEnabled(params):
        return iterable
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, disable=False, **kwargs)


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
    if isinstance(value, np.ndarray):
        return [_jsonSafe(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonSafe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonSafe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        valueFloat = float(value)
        return valueFloat if np.isfinite(valueFloat) else None
    return value


def _diagnosticValue(value: Any) -> Any:
    safeValue = _jsonSafe(value)
    if isinstance(safeValue, (dict, list, tuple)):
        return json.dumps(safeValue, sort_keys=True)
    return safeValue


def _calibrationKeyValueRows(
    *,
    recordType: str,
    event: str,
    chromosome: str | None,
    values: dict[str, Any],
    fold: int | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "record_type": recordType,
            "event": event,
            "chromosome": chromosome,
            "fold": None if fold is None else int(fold),
            "key": str(key),
            "value": _diagnosticValue(value),
        }
        for key, value in values.items()
    ]


def _ensureCalibrationLog(path: str | Path) -> Path:
    logPath = Path(path)
    if not logPath.exists():
        _logging_utils.init_tsv_log(logPath, DELETE_BLOCK_CALIBRATION_LOG_COLUMNS)
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
) -> int:
    blockLen = diagnostics.resolveUncertaintyBlockSizeIntervals(
        blockSizeBP,
        intervalSizeBP,
        n,
    )
    return int(min(blockLen, max(int(n), 1)))


def _resolveHoldoutCount(m: int, fraction: float | None) -> int:
    if m < 1:
        return 0
    if m == 1:
        return 1
    frac = (1.0 / float(m)) if fraction is None else float(fraction)
    frac = float(
        np.clip(
            frac,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MIN,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MAX,
        )
    )
    return int(
        np.clip(
            round(frac * float(m)),
            core.UNCERTAINTY_CALIBRATION_MIN_HOLDOUT_REPLICATES,
            max(m - 1, core.UNCERTAINTY_CALIBRATION_MIN_HOLDOUT_REPLICATES),
        )
    )


def _makeFoldMasks(
    *,
    m: int,
    n: int,
    blockLen: int,
    folds: int,
    holdoutCount: int,
    seed: int,
) -> list[np.ndarray]:
    if folds < core.UNCERTAINTY_CALIBRATION_MIN_FOLDS:
        raise ValueError("uncertainty calibration requires at least two folds")
    if holdoutCount < core.UNCERTAINTY_CALIBRATION_MIN_HOLDOUT_REPLICATES:
        raise ValueError("uncertainty calibration requires at least one held-out replicate")
    masks3d = _cuncertainty.cmakeFoldMasks(
        int(m),
        int(n),
        int(blockLen),
        int(folds),
        int(holdoutCount),
        int(seed),
    )
    return [np.ascontiguousarray(masks3d[fold]) for fold in range(int(folds))]


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
    for target in tuple(float(x) for x in targets):
        targetClipped = float(
            np.clip(
                target,
                core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
                1.0 - core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
            )
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
                    "delta": float(delta),
                    "N": N,
                    "k": None,
                    "q": qValue,
                    "q_source": "empirical_max_uncertified",
                    "certified": False,
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
                "delta": float(delta),
                "N": N,
                "k": int(k),
                "q": float(scores[k - 1]),
                "q_source": "pac_order_statistic",
                "certified": True,
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
            "certified": False,
            "scaled": False,
            "reason": "no_target_bound",
        }
    target = float(targetScaleBound.get("target", np.nan))
    qValue = targetScaleBound.get("q")
    targetZ = _normalZ(target) if np.isfinite(target) else np.nan
    if qValue is None:
        return {
            "scale": 1.0,
            "target": target if np.isfinite(target) else None,
            "target_z": float(targetZ) if np.isfinite(targetZ) else None,
            "q": None,
            "q_source": targetScaleBound.get("q_source"),
            "certified": bool(targetScaleBound.get("certified", False)),
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
            "certified": bool(targetScaleBound.get("certified", False)),
            "scaled": False,
            "reason": "nonfinite_target_bound",
        }
    certified = bool(targetScaleBound.get("certified", False))
    return {
        "scale": float(qFloat / targetZ),
        "target": target,
        "target_z": float(targetZ),
        "q": qFloat,
        "q_source": targetScaleBound.get("q_source"),
        "certified": certified,
        "scaled": True,
        "reason": (
            "scaled_by_certified_target_bound_q_over_z"
            if certified
            else "scaled_by_uncertified_empirical_max_q_over_z"
        ),
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


def _signalLevelCoverageStrata(signalAbs: np.ndarray) -> dict[str, np.ndarray]:
    signalAbs = np.asarray(signalAbs, dtype=np.float64).reshape(-1)
    finite = np.isfinite(signalAbs)
    if not np.any(finite):
        return {}
    quantiles = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    cuts = np.nanquantile(signalAbs[finite], quantiles)
    strata: dict[str, np.ndarray] = {}
    for idx in range(len(quantiles) - 1):
        lo = float(cuts[idx])
        hi = float(cuts[idx + 1])
        if idx == 0:
            mask = finite & (signalAbs <= hi)
        else:
            mask = finite & (signalAbs > lo) & (signalAbs <= hi)
        if not np.any(mask):
            continue
        strata[
            f"signal_abs_q{int(quantiles[idx] * 100):02d}_{int(quantiles[idx + 1] * 100):02d}"
        ] = mask
    return strata


def _summarizeScores(
    *,
    scores: pd.DataFrame,
    targets: tuple[float, ...],
) -> pd.DataFrame:
    if "uncertainty_decile" in scores.columns:
        decile = (
            scores["uncertainty_decile"]
            .fillna(-1)
            .astype(np.int32)
            .to_numpy(copy=True)
        )
    else:
        decile = np.full(scores.shape[0], -1, dtype=np.int32)
    targetsArray = np.ascontiguousarray(tuple(float(t) for t in targets), dtype=np.float64)
    targetZ = np.ascontiguousarray([_normalZ(target) for target in targetsArray], dtype=np.float64)
    summaryDict = _cuncertainty.csummarizeCoverageWidths(
        np.ascontiguousarray(scores["residual"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(scores["sd_before"].to_numpy(dtype=np.float64)),
        np.ascontiguousarray(scores["sd_after"].to_numpy(dtype=np.float64)),
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
    return summary[orderedColumns]


def _diagnosticsTable(
    *,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    model: dict[str, Any],
    chromosome: str | None = None,
) -> pd.DataFrame:
    scoreRows = scores.copy()
    scoreRows.insert(0, "event", "delete_block_calibration.score_sample")
    scoreRows.insert(0, "record_type", "score_sample")
    summaryRows = summary.copy()
    summaryRows.insert(0, "event", "delete_block_calibration.summary")
    summaryRows.insert(0, "record_type", "summary")
    modelRows = pd.DataFrame(
        {
            "record_type": "model",
            "event": "delete_block_calibration.model",
            "key": list(model.keys()),
            "value": [_diagnosticValue(value) for value in model.values()],
        }
    )
    if chromosome is not None:
        for frame in (scoreRows, summaryRows, modelRows):
            frame.insert(2, "chromosome", str(chromosome))
    columns = list(
        dict.fromkeys([*scoreRows.columns, *summaryRows.columns, *modelRows.columns])
    )
    return pd.concat(
        [
            scoreRows.reindex(columns=columns),
            summaryRows.reindex(columns=columns),
            modelRows.reindex(columns=columns),
        ],
        ignore_index=True,
    )


def _coverageLogPayload(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        if row.get("coverage_after") is None:
            continue
        parts.append(
            "target={target:.3g} before={coverage_before:.3f} after={coverage_after:.3f} n={n}".format(
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
        raise ValueError(
            "predictive held-out residual uncertainty calibration has been removed; "
            "use uncertaintyCalibrationParams.mode='delete_block_state'"
        )
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
    return np.ascontiguousarray(active, dtype=bool)


def _observationInformationMatrix(
    matrixMunc: np.ndarray,
    *,
    activeMask: np.ndarray,
    pad: float,
    lambdaExp: np.ndarray | None = None,
    useLambda: bool = False,
    lambdaMin: float = 1.0,
    lambdaMax: float = 1.0,
) -> np.ndarray:
    baseVar = np.asarray(matrixMunc, dtype=np.float64) + float(pad)
    with np.errstate(divide="ignore", invalid="ignore"):
        infoCell = np.where(activeMask, 1.0 / baseVar, 0.0)
    if useLambda:
        if lambdaExp is None:
            raise ValueError(
                "deleteBlockUseLambdaInInformation=True requires fullObservationPrecision"
            )
        lam = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
        if lam.shape[0] != matrixMunc.shape[1]:
            raise ValueError("fullObservationPrecision must match interval count")
        lam = np.clip(lam, float(lambdaMin), float(lambdaMax))
        infoCell = infoCell * lam[None, :]
    infoCell = np.where(np.isfinite(infoCell) & (infoCell > 0.0), infoCell, 0.0)
    return np.ascontiguousarray(infoCell, dtype=np.float64)


def _heldoutInformationByInterval(
    infoCell: np.ndarray,
    activeMask: np.ndarray,
    foldMask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del activeMask
    totalInfo = np.sum(infoCell, axis=0, dtype=np.float64)
    keptInfo = np.sum(np.where(np.asarray(foldMask, dtype=bool), infoCell, 0.0), axis=0)
    heldoutInfo = totalInfo - keptInfo
    with np.errstate(divide="ignore", invalid="ignore"):
        h = heldoutInfo / totalInfo
    h = np.where(np.isfinite(h), h, np.nan)
    return totalInfo, keptInfo, heldoutInfo, h


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
    source = np.full(P00Full.shape[0], "invalid", dtype=object)
    valid = np.zeros(P00Full.shape[0], dtype=bool)
    reason = np.full(P00Full.shape[0], "invalid", dtype=object)

    if mode == "covariance_difference":
        valid = covValid
        deltaVariance[valid] = covDelta[valid]
        source[valid] = "covariance_difference"
    elif mode == "heldout_information":
        valid = infoValid
        deltaVariance[valid] = infoDelta[valid]
        source[valid] = "heldout_information"
    elif mode == "hybrid":
        valid = covValid | infoValid
        deltaVariance[covValid] = covDelta[covValid]
        source[covValid] = "covariance_difference"
        fallback = ~covValid & infoValid
        deltaVariance[fallback] = infoDelta[fallback]
        source[fallback] = "heldout_information_fallback"
    else:
        raise AssertionError(f"unhandled delete-block variance mode: {mode}")

    reason[valid] = "valid"
    reason[~np.isfinite(P00Full) | (P00Full <= 0.0)] = "nonfinite_covariance"
    reason[np.isfinite(P00Full) & (P00Full > 0.0) & ~hValid] = "h_out_of_bounds"
    reason[
        np.isfinite(P00Full)
        & (P00Full > 0.0)
        & hValid
        & ~covValid
        & ~infoValid
    ] = "information_delta_invalid"
    reason[
        np.isfinite(P00Full)
        & np.isfinite(P00Masked)
        & (P00Full > 0.0)
        & (P00Masked > 0.0)
        & np.isfinite(covDelta)
        & (covDelta <= minDelta)
    ] = "covariance_delta_nonpositive"
    reason[valid] = "valid"
    return deltaVariance, source, valid, reason


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


def _deleteBlockScoreSamplingCodes(
    *,
    foldIndex: np.ndarray,
    intervalIndex: np.ndarray,
    pDelta: np.ndarray,
    fullState: np.ndarray,
    source: np.ndarray,
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
    sourceCodeMap = {
        "covariance_difference": 0,
        "heldout_information": 1,
        "heldout_information_fallback": 2,
    }
    sourceCode = np.asarray(
        [sourceCodeMap.get(str(value), 3) for value in np.asarray(source, dtype=object)],
        dtype=np.int64,
    )
    return (
        np.asarray(foldIndex, dtype=np.int64)
        * core.UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE
        + sourceCode * 128
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


def _deleteBlockTargetBlockScores(
    *,
    residual: np.ndarray,
    pDelta: np.ndarray,
    factorByInterval: np.ndarray,
    intervalIndex: np.ndarray,
    blockIndex: np.ndarray,
    targetBlockMask: np.ndarray,
    positiveFloor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validBlocks, blockScore, blockCellCount = _cuncertainty.cdeleteBlockBlockScores(
        residual,
        pDelta,
        factorByInterval,
        intervalIndex,
        blockIndex,
        targetBlockMask,
        varianceFloor=float(positiveFloor),
    )
    return (
        np.asarray(validBlocks, dtype=np.int64),
        np.asarray(blockScore, dtype=np.float64),
        np.asarray(blockCellCount, dtype=np.int64),
    )

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
) -> uncertaintyCalibrationResult:
    totalStart = time.perf_counter()
    timings: dict[str, float] = {}
    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.ascontiguousarray(matrixMunc, dtype=np.float32)
    m, n = matrixData.shape
    if m < 1:
        raise ValueError("uncertainty calibration requires at least one replicate")
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
    blockLen = _resolveBlockSizeIntervals(params.blockSizeBP, intervalSizeBP, n)
    holdoutFractionRaw = _firstSet(
        params,
        "holdoutFraction",
        "heldoutReplicateFraction",
        default=None,
    )
    holdoutCount = _resolveHoldoutCount(m, holdoutFractionRaw)
    folds = max(int(params.folds), core.UNCERTAINTY_CALIBRATION_MIN_FOLDS)
    logger.info(
        "uncertaintyCalibration.start mode=delete_block_state intervals=%s samples=%s folds=%s blockLen=%s holdoutCount=%s varianceMode=%s targetSignal=%s factorModel=%s",
        n,
        m,
        folds,
        blockLen,
        holdoutCount,
        varianceMode,
        targetSignal,
        factorModel,
    )
    stageStart = time.perf_counter()
    masks = _makeFoldMasks(
        m=m,
        n=n,
        blockLen=blockLen,
        folds=folds,
        holdoutCount=holdoutCount,
        seed=int(params.seed),
    )
    timings["make_masks_seconds"] = time.perf_counter() - stageStart
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
    if targetSignal == "state_plus_background":
        if fullBackground is None:
            if bool(runKwargs.get("fitBackground", False)):
                raise ValueError(
                    "deleteBlockTargetSignal='state_plus_background' requires fullBackground"
                )
            fullBackgroundArr = np.zeros(n, dtype=np.float64)
        else:
            fullBackgroundArr = np.asarray(fullBackground, dtype=np.float64).reshape(-1)
            if fullBackgroundArr.shape[0] != n:
                raise ValueError("fullBackground must match the interval count")
    else:
        fullBackgroundArr = np.zeros(n, dtype=np.float64)
    padValue = float(runKwargs.get("pad", _calibrationPad()))
    activeMask = _activeObservationMask(
        matrixData,
        matrixMunc,
        originalObservationMask,
        padValue,
    )
    infoCell = _observationInformationMatrix(
        matrixMunc,
        activeMask=activeMask,
        pad=padValue,
        lambdaExp=fullObservationPrecision,
        useLambda=bool(params.deleteBlockUseLambdaInInformation),
        lambdaMin=float(runKwargs.get("observationPrecisionMultiplierMin", 1.0)),
        lambdaMax=float(runKwargs.get("observationPrecisionMultiplierMax", 1.0)),
    )
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
    sourceChunks: list[np.ndarray] = []
    rowWeightChunks: list[np.ndarray] = []
    stateMaskedChunks: list[np.ndarray] = []
    pMaskedChunks: list[np.ndarray] = []
    covDeltaChunks: list[np.ndarray] = []
    totalInfoChunks: list[np.ndarray] = []
    keptInfoChunks: list[np.ndarray] = []
    heldInfoChunks: list[np.ndarray] = []
    invalidReasonCounts = {
        "no_deleted_information": 0,
        "h_out_of_bounds": 0,
        "covariance_delta_nonpositive": 0,
        "information_delta_invalid": 0,
        "nonfinite_state_delta": 0,
        "nonfinite_covariance": 0,
        "other": 0,
    }
    rowsTotal = 0
    foldFailures = 0
    foldDiagnosticRows: list[dict[str, Any]] = []

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
    for fold, mask in _progress(
        list(enumerate(masks)),
        params=params,
        desc="Uncertainty calibration folds",
        unit="fold",
    ):
        logger.info(
            "uncertaintyCalibration.fold.start fold=%s/%s intervals=%s warmupMode=%s warmupDetail=%s",
            int(fold + 1),
            int(len(masks)),
            n,
            warmupMode,
            warmupDetail,
        )
        stageStart = time.perf_counter()
        try:
            out = core.runConsenrich(
                matrixData,
                matrixMunc,
                observationMask=mask,
                **fitKwargs,
            )
        except Exception as exc:
            logger.warning(
                "uncertaintyCalibration.deleteBlock.fold.failed fold=%s/%s error=%s",
                int(fold + 1),
                int(len(masks)),
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
        if targetSignal == "state_plus_background":
            if len(out) <= 5:
                raise ValueError(
                    "deleteBlockTargetSignal='state_plus_background' requires masked background output"
                )
            backgroundMasked = np.asarray(out[5], dtype=np.float64).reshape(-1)
            if backgroundMasked.shape[0] != n:
                raise ValueError("masked background output must match interval count")
            signalMasked = xMasked + backgroundMasked
            signalFull = fullState0 + fullBackgroundArr
        else:
            signalMasked = xMasked
            signalFull = fullState0
        stateDelta = signalMasked - signalFull
        totalInfo, keptInfo, heldoutInfo, h = _heldoutInformationByInterval(
            infoCell,
            activeMask,
            mask,
        )
        deltaVariance, source, valid, invalidReason = _chooseDeleteBlockDeltaVariance(
            fullPArr,
            pMasked,
            h,
            mode=varianceMode,
            minDeltaVariance=float(params.deleteBlockMinDeltaVariance),
            minInformationFraction=float(params.deleteBlockMinInformationFraction),
            maxInformationFraction=float(params.deleteBlockMaxInformationFraction),
            positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        invalidReason = np.asarray(invalidReason, dtype=object)
        meaningfulDeletion = (
            np.isfinite(totalInfo)
            & (totalInfo > 0.0)
            & np.isfinite(heldoutInfo)
            & (heldoutInfo > 0.0)
        )
        invalidReason[~meaningfulDeletion] = "no_deleted_information"
        valid &= meaningfulDeletion
        finiteDelta = np.isfinite(stateDelta)
        invalidReason[~finiteDelta] = "nonfinite_state_delta"
        valid &= finiteDelta
        invalidReason[valid] = "valid"
        rowsTotal += int(n)
        for reason in np.asarray(invalidReason[~valid], dtype=object):
            key = str(reason)
            invalidReasonCounts[key if key in invalidReasonCounts else "other"] += 1
        foldExtractSeconds = time.perf_counter() - stageStart
        extractSeconds += foldExtractSeconds
        if not np.any(valid):
            logger.info(
                "uncertaintyCalibration.fold.done fold=%s/%s deleteBlockRows=0 refitSeconds=%.3f extractSeconds=%.3f",
                int(fold + 1),
                int(len(masks)),
                float(foldRefitSeconds),
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
                        "refit_seconds": float(foldRefitSeconds),
                        "extract_seconds": float(foldExtractSeconds),
                    },
                )
            )
            continue
        idx = np.flatnonzero(valid).astype(np.int64, copy=False)
        foldExtractSeconds = time.perf_counter() - stageStart
        residualChunks.append(np.ascontiguousarray(stateDelta[idx], dtype=np.float64))
        pDeltaChunks.append(np.ascontiguousarray(deltaVariance[idx], dtype=np.float64))
        iChunks.append(idx.astype(np.int64, copy=False))
        foldChunks.append(
            np.full(idx.shape[0], int(fold), dtype=np.int32)
        )
        hChunks.append(np.ascontiguousarray(h[idx], dtype=np.float64))
        sourceChunks.append(np.asarray(source[idx], dtype=object))
        rowWeightChunks.append(_deleteBlockRowWeights(h[idx], params))
        stateMaskedChunks.append(np.ascontiguousarray(xMasked[idx], dtype=np.float64))
        pMaskedChunks.append(np.ascontiguousarray(pMasked[idx], dtype=np.float64))
        covDeltaChunks.append(
            np.ascontiguousarray(pMasked[idx] - fullPArr[idx], dtype=np.float64)
        )
        totalInfoChunks.append(np.ascontiguousarray(totalInfo[idx], dtype=np.float64))
        keptInfoChunks.append(np.ascontiguousarray(keptInfo[idx], dtype=np.float64))
        heldInfoChunks.append(np.ascontiguousarray(heldoutInfo[idx], dtype=np.float64))
        sourceCountsFold = {
            str(value): int(count)
            for value, count in zip(*np.unique(source[idx], return_counts=True))
        }
        covValidFractionFold = float(
            sourceCountsFold.get("covariance_difference", 0) / max(int(idx.size), 1)
        )
        logger.info(
            "uncertaintyCalibration.fold.done fold=%s/%s deleteBlockRows=%s varianceMode=%s covarianceDifferenceFraction=%.3f refitSeconds=%.3f extractSeconds=%.3f",
            int(fold + 1),
            int(len(masks)),
            int(idx.size),
            varianceMode,
            covValidFractionFold,
            float(foldRefitSeconds),
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
                    "variance_mode": varianceMode,
                    "covariance_difference_fraction": covValidFractionFold,
                    "refit_seconds": float(foldRefitSeconds),
                    "extract_seconds": float(foldExtractSeconds),
                },
            )
        )
    timings["masked_refits_seconds"] = refitSeconds
    timings["extract_scores_seconds"] = extractSeconds

    if not residualChunks:
        raise ValueError("delete-block state uncertainty calibration produced no valid deleted-state rows")
    residual = np.concatenate(residualChunks)
    pDelta = np.concatenate(pDeltaChunks)
    intervalIndex = np.concatenate(iChunks)
    foldIndex = np.concatenate(foldChunks)
    hAll = np.concatenate(hChunks)
    sourceAll = np.concatenate(sourceChunks).astype(object, copy=False)
    rowWeight = np.concatenate(rowWeightChunks)
    stateMaskedAll = np.concatenate(stateMaskedChunks)
    pMaskedAll = np.concatenate(pMaskedChunks)
    covDeltaAll = np.concatenate(covDeltaChunks)
    totalInfoAll = np.concatenate(totalInfoChunks)
    keptInfoAll = np.concatenate(keptInfoChunks)
    heldInfoAll = np.concatenate(heldInfoChunks)
    blockIndex = (intervalIndex // int(blockLen)).astype(np.int64, copy=False)
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
        fullState=fullState0,
        source=sourceAll,
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
        factor = np.asarray(segShrinkFit["factor"], dtype=np.float64)
        calibrated = np.asarray(segShrinkFit["calibrated"], dtype=np.float32)
        modelMeta = dict(segShrinkFit["modelMeta"])
        factorGlobal = float(modelMeta.get("global_factor", np.nan))
    else:
        factorGlobal, modelMeta = _fitDeleteBlockGlobalFactor(
            residual=residualFit,
            pDelta=pDeltaFit,
            rowWeight=rowWeight[fitRows],
            params=params,
        )
        factor = np.full(n, float(factorGlobal), dtype=np.float64)
        calibrated = np.sqrt(
            np.maximum(
                factor * fullPArr,
                core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
            )
        ).astype(np.float32)
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
    timings["evaluate_factor_seconds"] = time.perf_counter() - stageStart
    heldFactor = factor[intervalIndexFit].astype(np.float64)
    sdBeforeAll = np.sqrt(
        np.maximum(pDelta, core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR)
    )
    sdAfterAll = np.sqrt(
        np.maximum(
            factor[intervalIndex].astype(np.float64) * pDelta,
            core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
        )
    )
    sdBefore = sdBeforeAll[fitRows]
    sdAfter = sdAfterAll[fitRows]
    signalAbsHeldout = np.abs(fullState0[intervalIndex])
    stateCoverage = core.checkStateUncertaintyCoverage(
        residual,
        sdBeforeAll,
        sdAfterAll,
        targets=tuple(float(t) for t in params.targets),
        strata=_signalLevelCoverageStrata(signalAbsHeldout),
    )
    signalAbsFit = np.abs(fullState0[intervalIndexFit])
    stateCoverageFit = core.checkStateUncertaintyCoverage(
        residualFit,
        sdBefore,
        sdAfter,
        targets=tuple(float(t) for t in params.targets),
        strata=_signalLevelCoverageStrata(signalAbsFit),
    )
    targetBlockIds = np.empty(0, dtype=np.int64)
    targetBlockScores = np.empty(0, dtype=np.float64)
    targetBlockCellCounts = np.empty(0, dtype=np.int64)
    targetCalibrationBounds: list[dict[str, Any]] = []
    if targetCalibrationEnabled:
        targetBlockIds, targetBlockScores, targetBlockCellCounts = (
            _deleteBlockTargetBlockScores(
                residual=residual,
                pDelta=pDelta,
                factorByInterval=factor,
                intervalIndex=intervalIndex,
                blockIndex=blockIndex,
                targetBlockMask=np.asarray(targetSplit["target_block_mask"], dtype=np.uint8),
                positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
            )
        )
        targetCalibrationBounds = _targetCalibrationBounds(
            targetBlockScores,
            targets=tuple(float(t) for t in params.targets),
            delta=float(targetDelta),
        )
    deleteBlockApplyTarget = getattr(params, "deleteBlockApplyTargetCalibration", None)
    scaleByTargetCalibration = bool(
        params.scaleUncertaintyByTargetCalibration
        if deleteBlockApplyTarget is None
        else deleteBlockApplyTarget
    )
    targetScaleBound = _targetCalibrationScaleBound(targetCalibrationBounds)
    targetScaleInfo = _targetCalibrationTrackScale(targetScaleBound)
    uncertaintyTrackScale = 1.0
    uncertaintyTrackScaleTarget = None
    uncertaintyTrackScaleTargetZ = None
    uncertaintyTrackScaleQ = None
    uncertaintyTrackScaled = False
    uncertaintyTrackScaleCertified = False
    uncertaintyTrackScaleReason = "target_calibration_disabled"
    if targetCalibrationEnabled:
        uncertaintyTrackScaleReason = "scale_disabled_by_config"
        if scaleByTargetCalibration:
            uncertaintyTrackScaleReason = str(targetScaleInfo["reason"])
            uncertaintyTrackScaleTarget = targetScaleInfo["target"]
            uncertaintyTrackScaleTargetZ = targetScaleInfo["target_z"]
            uncertaintyTrackScaleQ = targetScaleInfo["q"]
            uncertaintyTrackScaleCertified = bool(targetScaleInfo["certified"])
            uncertaintyTrackScaled = bool(targetScaleInfo["scaled"])
            uncertaintyTrackScale = float(targetScaleInfo["scale"])
            if uncertaintyTrackScaled:
                uncertaintyTrackScaleReason = (
                    "delete_block_state_target_bound_divided_by_normal_z"
                )
            if uncertaintyTrackScaled:
                calibrated = (
                    np.asarray(calibrated, dtype=np.float32)
                    * np.float32(uncertaintyTrackScale)
                )
    targetQ = (
        None
        if targetScaleBound is None or targetScaleBound.get("q") is None
        else float(targetScaleBound.get("q"))
    )
    targetQSource = None if targetScaleBound is None else targetScaleBound.get("q_source")
    targetLog = (
        logger.warning
        if (
            targetCalibrationEnabled
            and scaleByTargetCalibration
            and (not uncertaintyTrackScaled or not uncertaintyTrackScaleCertified)
        )
        else logger.info
    )
    targetLog(
        "uncertaintyCalibration.target enabled=%s delta=%s blocksTotal=%d blocksScale=%d blocksTarget=%d blocksTargetScored=%d targetBlockCells=%d selectedTarget=%s targetZ=%s q=%s qSource=%s certified=%s scaleRequested=%s scaleApplied=%s scale=%.6g reason=%s",
        bool(targetCalibrationEnabled),
        None if targetDelta is None else float(targetDelta),
        int(targetSplit["blocks_total"]),
        int(np.asarray(targetSplit["scale_blocks"]).size),
        int(np.asarray(targetSplit["target_blocks"]).size),
        int(targetBlockScores.size),
        int(np.sum(targetBlockCellCounts)),
        uncertaintyTrackScaleTarget,
        uncertaintyTrackScaleTargetZ,
        targetQ,
        targetQSource,
        bool(uncertaintyTrackScaleCertified),
        bool(scaleByTargetCalibration),
        bool(uncertaintyTrackScaled),
        float(uncertaintyTrackScale),
        uncertaintyTrackScaleReason,
    )
    coverageOverall = [
        row for row in stateCoverage if str(row.get("stratum", "")) == "overall"
    ]
    coverageFitOverall = [
        row for row in stateCoverageFit if str(row.get("stratum", "")) == "overall"
    ]
    if coverageOverall:
        coveragePayload = _coverageLogPayload(coverageOverall)
        logger.info(
            "uncertaintyCalibration.coverage.delete_block_all %s",
            coveragePayload,
        )
    if coverageFitOverall:
        logger.info(
            "uncertaintyCalibration.coverage.fit_sample %s",
            _coverageLogPayload(coverageFitOverall),
        )
    stageStart = time.perf_counter()
    sourceFit = sourceAll[fitRows].astype(object, copy=False)
    hFit = hAll[fitRows]
    pMaskedFit = pMaskedAll[fitRows]
    covDeltaFit = covDeltaAll[fitRows]
    totalInfoFit = totalInfoAll[fitRows]
    keptInfoFit = keptInfoAll[fitRows]
    heldInfoFit = heldInfoAll[fitRows]
    rowWeightFit = rowWeight[fitRows]
    pDeltaFit = pDelta[fitRows]
    scores = pd.DataFrame(
        {
            "fold": foldIndexFit,
            "interval_index": intervalIndexFit,
            "block_index": blockIndex[fitRows],
            "chrom_start": intervalsArr[intervalIndexFit],
            "residual": residualFit,
            "deleted_state_delta": residualFit,
            "state_full": fullState0[intervalIndexFit],
            "state_masked": stateMaskedAll[fitRows],
            "P00_full": fullPArr[intervalIndexFit],
            "P00_masked": pMaskedFit,
            "covariance_delta": covDeltaFit,
            "total_information": totalInfoFit,
            "kept_information": keptInfoFit,
            "heldout_information": heldInfoFit,
            "heldout_information_fraction": hFit,
            "delta_variance": pDeltaFit,
            "delta_variance_source": sourceFit,
            "row_weight": rowWeightFit,
            "sd_before": sdBefore,
            "sd_after": sdAfter,
            "a_state": heldFactor,
        }
    )
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
        scores["factor_segment"] = fitSegment
        scores["blockIDX"] = scores["block_index"].to_numpy(dtype=np.int64)
        scores["segment_raw_factor"] = segmentRawFactor
        scores["segment_bootstrap_variance"] = segmentBootstrapVariance
        scores["segment_shrinkage_weight"] = segmentShrinkageWeight
        scores["contig_shrinkage_weight"] = contigShrinkageWeight
    try:
        scores["uncertainty_decile"] = pd.qcut(
            scores["delta_variance"],
            q=core.UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        scores["uncertainty_decile"] = 0
    stateAbsAll = np.abs(fullState0)
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
    scores["high_signal"] = (
        stateAbsAll[scores["interval_index"].to_numpy(dtype=np.int64)] >= highSignalCut
    )
    summary = _summarizeScores(scores=scores, targets=tuple(float(t) for t in params.targets))
    diagnosticRows = _samplePositionsByCode(
        _deleteBlockScoreSamplingCodes(
            foldIndex=scores["fold"].to_numpy(dtype=np.int64),
            intervalIndex=scores["interval_index"].to_numpy(dtype=np.int64),
            pDelta=scores["delta_variance"].to_numpy(dtype=np.float64),
            fullState=fullState0,
            source=scores["delta_variance_source"].to_numpy(dtype=object),
        ),
        maxRows=int(params.maxDiagnosticRows),
        seed=int(params.seed) + core.UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET,
    )
    scoresDiagnostics = scores.iloc[diagnosticRows, :].reset_index(drop=True)
    timings["summarize_scores_seconds"] = time.perf_counter() - stageStart
    sourceCounts = {
        "covariance_difference": int(np.count_nonzero(sourceAll == "covariance_difference")),
        "heldout_information": int(np.count_nonzero(sourceAll == "heldout_information")),
        "heldout_information_fallback": int(
            np.count_nonzero(sourceAll == "heldout_information_fallback")
        ),
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
    factorMin, factorMax = _factorBounds(params)
    model = {
        **modelMeta,
        "mode": calibrationMode,
        "score_definition": "deleted_state_delta_over_deleted_state_delta_sd",
        "target_signal": targetSignal,
        "variance_mode": varianceMode,
        "factor_model": factorModel,
        "calibration_policy": calibrationMode,
        "feature_names": featureNames,
        "feature_center": [float(x) for x in featureCenter],
        "feature_scale": [float(x) for x in featureScale],
        "factor_bound_min": float(factorMin),
        "factor_bound_max": float(factorMax),
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
        "fold_refits": {
            "folds": int(folds),
            "fold_failures": int(foldFailures),
            "block_len_intervals": int(blockLen),
            "holdout_count": int(holdoutCount),
            "holdout_fraction": float(holdoutCount / max(m, 1)),
            "holdout_fraction_config": (
                None if holdoutFractionRaw is None else float(holdoutFractionRaw)
            ),
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
            "enabled": bool(targetCalibrationEnabled),
            "delta": None if targetDelta is None else float(targetDelta),
            "block_split_seed": int(targetSplit["seed"]),
            "target_block_fraction": float(TARGET_CALIBRATION_FRACTION),
            "blocks_total": int(targetSplit["blocks_total"]),
            "blocks_scale": int(np.asarray(targetSplit["scale_blocks"]).size),
            "blocks_target": int(np.asarray(targetSplit["target_blocks"]).size),
            "blocks_target_scored": int(targetBlockScores.size),
            "target_block_cells": int(np.sum(targetBlockCellCounts)),
            "scale_uncertainty_by_target_calibration": bool(scaleByTargetCalibration),
            "uncertainty_track_scaled": bool(uncertaintyTrackScaled),
            "uncertainty_track_scale": float(uncertaintyTrackScale),
            "uncertainty_track_scale_target": uncertaintyTrackScaleTarget,
            "uncertainty_track_scale_target_z": uncertaintyTrackScaleTargetZ,
            "uncertainty_track_scale_q": uncertaintyTrackScaleQ,
            "uncertainty_track_scale_certified": bool(uncertaintyTrackScaleCertified),
            "uncertainty_track_scale_reason": uncertaintyTrackScaleReason,
            "score_definition": "max_abs_deleted_state_delta_over_deleted_state_delta_sd_by_block",
            "bounds": targetCalibrationBounds,
        },
    }
    timings["total_seconds"] = time.perf_counter() - totalStart
    model["timings_seconds"] = {key: float(value) for key, value in timings.items()}
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
            else str(Path(str(outPrefix))) + ".delete_block_calibration.log"
        )
        diagnosticsTable = _diagnosticsTable(
            scores=scoresDiagnostics,
            summary=summary,
            model=model,
            chromosome=chromosome,
        )
        rowsWritten = _logging_utils.append_tsv_log(
            logPath,
            diagnosticsTable,
            DELETE_BLOCK_CALIBRATION_LOG_COLUMNS,
        )
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
        extraRows.extend(
            {
                "record_type": "target_bound",
                "event": "delete_block_calibration.target_bound",
                "chromosome": chromosome,
                **{
                    key: _diagnosticValue(value)
                    for key, value in dict(bound).items()
                },
            }
            for bound in targetCalibrationBounds
        )
        rowsWritten += _logging_utils.append_tsv_log(
            logPath,
            extraRows,
            DELETE_BLOCK_CALIBRATION_LOG_COLUMNS,
        )
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
        factor=factor.astype(np.float32),
        calibratedUncertainty=calibrated,
        summary=summary,
        scores=scores,
        model=model,
    )


__all__ = [
    "calibrateChromosomeStateUncertainty",
    "uncertaintyCalibrationResult",
]
