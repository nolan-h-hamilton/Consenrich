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
from scipy import optimize, stats
from tqdm import tqdm

from . import core
from . import diagnostics
from . import cuncertainty as _cuncertainty


logger = logging.getLogger(__name__)


TARGET_CALIBRATION_BLOCK_SPLIT_SEED_OFFSET = 20_000
TARGET_CALIBRATION_FRACTION = 0.5


class uncertaintyCalibrationResult(NamedTuple):
    factor: np.ndarray
    calibratedUncertainty: np.ndarray
    summary: pd.DataFrame
    scores: pd.DataFrame
    model: dict[str, Any]


class observationVarianceFloorCalibrationResult(NamedTuple):
    minR: float
    trimmedMean: float
    target: float
    heldoutCells: int
    fitCells: int
    usedLambda: bool
    hitUpperBound: bool
    fallbackUsed: bool
    diagnostics: dict[str, Any]


def _progressEnabled(params: core.uncertaintyCalibrationParams) -> bool:
    return bool(params.writeDiagnostics) and bool(getattr(sys.stderr, "isatty", lambda: False)())


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


def _aObsPenalty(params: core.uncertaintyCalibrationParams) -> float:
    return float(
        max(
            _firstSet(
                params,
                "aObsPriorStrength",
                "aObsPenalty",
                default=core.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY,
            ),
            0.0,
        )
    )


def _wisWeight(params: core.uncertaintyCalibrationParams) -> float:
    return float(
        max(
            _firstSet(
                params,
                "wisWeight",
                default=core.UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT,
            ),
            0.0,
        )
    )


def _calibrationPad(
    params: core.uncertaintyCalibrationParams,
    pad: float | None,
) -> float:
    if pad is not None:
        return float(pad)
    return float(_firstSet(params, "pad", default=0.0) or 0.0)


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
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    stateVar = np.maximum(
        np.asarray(stateVar, dtype=np.float64).reshape(-1),
        core.UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR,
    )
    matrixMunc = np.asarray(matrixMunc, dtype=np.float64)
    finiteMunc = np.where(np.isfinite(matrixMunc), matrixMunc, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        meanMunc = np.nanmean(finiteMunc, axis=0)
    meanMunc = np.maximum(meanMunc, core.UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR)
    stateDelta = np.zeros_like(state)
    if state.size > 1:
        stateDelta[1:] = np.diff(state)
    absState = np.abs(state)
    highSignalCut = (
        float(np.nanquantile(absState, core.UNCERTAINTY_CALIBRATION_FEATURE_HIGH_SIGNAL_QUANTILE))
        if absState.size
        else np.inf
    )
    raw = np.column_stack(
        [
            np.log(stateVar),
            np.log(meanMunc),
            absState,
            np.abs(stateDelta),
            (absState > highSignalCut).astype(np.float64),
        ]
    )
    center = np.nanmedian(raw, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    mad = np.nanmedian(np.abs(raw - center[None, :]), axis=0)
    scale = mad * core.UNCERTAINTY_CALIBRATION_FEATURE_MAD_NORMAL_SCALE
    scale = np.where(
        np.isfinite(scale) & (scale > core.UNCERTAINTY_CALIBRATION_FEATURE_SCALE_FLOOR),
        scale,
        1.0,
    )
    standardized = (raw - center[None, :]) / scale[None, :]
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.column_stack([np.ones(state.size, dtype=np.float64), standardized])
    return X, featureNames, center, scale


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


def _scoreSamplingCodes(
    *,
    foldIndex: np.ndarray,
    repIndex: np.ndarray,
    intervalIndex: np.ndarray,
    pState: np.ndarray,
    fullState: np.ndarray,
) -> np.ndarray:
    try:
        sdDecile = pd.qcut(
            np.asarray(pState, dtype=np.float64),
            q=core.UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES,
            labels=False,
            duplicates="drop",
        )
        sdCode = np.nan_to_num(
            np.asarray(sdDecile, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.int64)
    except ValueError:
        sdCode = np.zeros(np.asarray(pState).shape[0], dtype=np.int64)
    stateAbs = np.abs(np.asarray(fullState, dtype=np.float64)[np.asarray(intervalIndex, dtype=np.int64)])
    stateCut = (
        float(
            np.nanquantile(
                np.abs(np.asarray(fullState, dtype=np.float64)),
                core.UNCERTAINTY_CALIBRATION_SCORE_STATE_ABS_QUANTILE,
            )
        )
        if np.asarray(fullState).size
        else np.inf
    )
    highSignal = (stateAbs >= stateCut).astype(np.int64)
    return (
        np.asarray(foldIndex, dtype=np.int64)
        * core.UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE
        + np.asarray(repIndex, dtype=np.int64)
        * core.UNCERTAINTY_CALIBRATION_SCORE_REPLICATE_CODE_STRIDE
        + sdCode * 2
        + highSignal
    )


def _scoreSamplingCodesWithoutState(
    *,
    foldIndex: np.ndarray,
    repIndex: np.ndarray,
    pState: np.ndarray,
) -> np.ndarray:
    try:
        sdDecile = pd.qcut(
            np.asarray(pState, dtype=np.float64),
            q=core.UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES,
            labels=False,
            duplicates="drop",
        )
        sdCode = np.nan_to_num(
            np.asarray(sdDecile, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.int64)
    except ValueError:
        sdCode = np.zeros(np.asarray(pState).shape[0], dtype=np.int64)
    return (
        np.asarray(foldIndex, dtype=np.int64)
        * core.UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE
        + np.asarray(repIndex, dtype=np.int64)
        * core.UNCERTAINTY_CALIBRATION_SCORE_REPLICATE_CODE_STRIDE
        + sdCode
    )


def _trimmedMean05To95(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size >= 20:
        lo, hi = np.quantile(arr, [0.05, 0.95])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if trimmed.size:
            arr = trimmed
    return float(np.mean(arr, dtype=np.float64))


def _observationFloorCalibrationMean(
    *,
    residual: np.ndarray,
    pState: np.ndarray,
    muncBase: np.ndarray,
    lambdaExp: np.ndarray | None,
    r: float,
    pad: float,
) -> float:
    base = np.maximum(np.asarray(muncBase, dtype=np.float64), float(r))
    obsVar = base + float(pad)
    if lambdaExp is not None:
        lam = np.maximum(
            np.asarray(lambdaExp, dtype=np.float64),
            core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
        )
        obsVar = obsVar / lam
    denom = np.maximum(
        np.asarray(pState, dtype=np.float64) + obsVar,
        core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
    )
    ratio = (np.asarray(residual, dtype=np.float64) ** 2) / denom
    return _trimmedMean05To95(ratio)


def _selectObservationVarianceFloor(
    *,
    residual: np.ndarray,
    pState: np.ndarray,
    muncBase: np.ndarray,
    lambdaExp: np.ndarray | None,
    pad: float,
    lower: float,
    upper: float,
    target: float = 1.0,
) -> tuple[float, float, bool]:
    lo = float(max(float(lower), 0.0))
    hi = float(max(float(upper), lo))
    target = float(target)
    loMean = _observationFloorCalibrationMean(
        residual=residual,
        pState=pState,
        muncBase=muncBase,
        lambdaExp=lambdaExp,
        r=lo,
        pad=pad,
    )
    if not np.isfinite(loMean):
        return lo, loMean, False
    if loMean <= target:
        return lo, loMean, False

    hiMean = _observationFloorCalibrationMean(
        residual=residual,
        pState=pState,
        muncBase=muncBase,
        lambdaExp=lambdaExp,
        r=hi,
        pad=pad,
    )
    if not np.isfinite(hiMean) or hiMean > target:
        return hi, hiMean, True

    bestR = hi
    bestMean = hiMean
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        midMean = _observationFloorCalibrationMean(
            residual=residual,
            pState=pState,
            muncBase=muncBase,
            lambdaExp=lambdaExp,
            r=mid,
            pad=pad,
        )
        if not np.isfinite(midMean):
            hi = mid
            continue
        if midMean <= target:
            bestR = mid
            bestMean = midMean
            hi = mid
        else:
            lo = mid
    return float(bestR), float(bestMean), False


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


def _fitFactorModel(
    *,
    residual: np.ndarray,
    pState: np.ndarray,
    obsVar: np.ndarray,
    featureByInterval: np.ndarray,
    intervalIndex: np.ndarray,
    params: core.uncertaintyCalibrationParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual = np.asarray(residual, dtype=np.float64)
    pState = np.maximum(
        np.asarray(pState, dtype=np.float64),
        core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
    )
    obsVar = np.maximum(
        np.asarray(obsVar, dtype=np.float64),
        core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
    )
    factorMin, factorMax = _factorBounds(params)
    raw = (residual * residual - obsVar) / pState
    raw = raw[np.isfinite(raw) & (raw > 0.0)]
    initFactor = 1.0 if raw.size == 0 else float(np.nanmedian(raw))
    initFactor = float(np.clip(initFactor, factorMin, factorMax))
    theta0 = np.zeros(featureByInterval.shape[1] + 1, dtype=np.float64)
    theta0[0] = np.log(initFactor)
    ridge = float(max(params.ridge, 0.0))
    aObsPenalty = _aObsPenalty(params)
    targets = tuple(float(t) for t in params.targets)
    targetArray = np.ascontiguousarray(targets, dtype=np.float64)
    targetZ = np.ascontiguousarray([_normalZ(target) for target in targets], dtype=np.float64)
    scaleWIS = max(
        float(np.nanmedian(np.abs(residual)))
        * core.UNCERTAINTY_CALIBRATION_WIS_SCALE_MULTIPLIER,
        core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
    )
    cFeature = np.ascontiguousarray(featureByInterval, dtype=np.float64)
    cInterval = np.ascontiguousarray(intervalIndex, dtype=np.int64)
    cResidual = np.ascontiguousarray(residual, dtype=np.float64)
    cPState = np.ascontiguousarray(pState, dtype=np.float64)
    cObsVar = np.ascontiguousarray(obsVar, dtype=np.float64)

    def objectiveAndGradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        value, gradient = _cuncertainty.cfactorObjectiveAndGradient(
            theta,
            cResidual,
            cPState,
            cObsVar,
            cFeature,
            cInterval,
            targetArray,
            targetZ,
            factorMin,
            factorMax,
            ridge,
            aObsPenalty,
            scaleWIS,
            float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN),
            float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX),
            float(core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR),
            _wisWeight(params),
            float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        return float(value), np.ascontiguousarray(gradient, dtype=np.float64)

    pbar = tqdm(
        total=None,
        desc="Fitting uncertainty factor",
        unit="iter",
        disable=not _progressEnabled(params),
        mininterval=0.5,
        leave=False,
        dynamic_ncols=True,
    )

    def callback(_theta: np.ndarray) -> None:
        pbar.update(1)

    try:
        result = optimize.minimize(
            objectiveAndGradient,
            theta0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
        )
    finally:
        pbar.close()
    theta = np.asarray(result.x if result.success else theta0, dtype=np.float64)
    beta = theta[:-1]
    aObs = float(
        np.exp(
            np.clip(
                theta[-1],
                np.log(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN),
                np.log(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX),
            )
        )
    )
    meta = {
        "success": bool(result.success),
        "objective": float(
            result.fun
            if np.isfinite(result.fun)
            else objectiveAndGradient(theta)[0]
        ),
        "message": str(result.message),
        "initial_factor": initFactor,
        "a_obs_factor": aObs,
        "a_obs_penalty": aObsPenalty,
    }
    return beta, meta


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
    scoreRows.insert(0, "record_type", "score")
    summaryRows = summary.copy()
    summaryRows.insert(0, "record_type", "summary")
    modelRows = pd.DataFrame(
        {
            "record_type": "model",
            "key": list(model.keys()),
            "value": [
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
                for value in model.values()
            ],
        }
    )
    if chromosome is not None:
        for frame in (scoreRows, summaryRows, modelRows):
            frame.insert(1, "chromosome", str(chromosome))
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


def _writeModelJson(path: str, model: dict[str, Any], chromosome: str | None) -> None:
    pathObj = Path(path)
    if chromosome is None:
        payload: dict[str, Any] = dict(model)
    else:
        payload = {"chromosomes": {str(chromosome): dict(model)}}
        if pathObj.exists():
            try:
                with open(pathObj, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if isinstance(existing, dict):
                    existingChroms = existing.get("chromosomes")
                    if isinstance(existingChroms, dict):
                        payload = existing
                        payload["chromosomes"][str(chromosome)] = dict(model)
            except Exception:
                pass
    with open(pathObj, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def estimateObservationVarianceFloorFromHeldout(
    *,
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    intervalSizeBP: int,
    params: core.uncertaintyCalibrationParams,
    runKwargs: dict[str, Any],
    pad: float | None = None,
    maxR: float | None = None,
    fallbackMinR: float = 1.0e-4,
    excludeIntervals: np.ndarray | None = None,
    chromosome: str | None = None,
) -> observationVarianceFloorCalibrationResult:
    totalStart = time.perf_counter()
    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.ascontiguousarray(matrixMunc, dtype=np.float32)
    m, n = matrixData.shape
    if m < 1:
        raise ValueError("observation-floor calibration requires at least one replicate")
    if matrixMunc.shape != matrixData.shape:
        raise ValueError("matrixMunc must match matrixData shape")

    padValue = _calibrationPad(params, pad)
    blockLen = _resolveBlockSizeIntervals(params.blockSizeBP, intervalSizeBP, n)
    holdoutFraction = _firstSet(
        params,
        "holdoutFraction",
        "heldoutReplicateFraction",
        default=None,
    )
    holdoutCount = _resolveHoldoutCount(m, holdoutFraction)
    folds = max(int(params.folds), core.UNCERTAINTY_CALIBRATION_MIN_FOLDS)
    upper = float(maxR) if maxR is not None and np.isfinite(float(maxR)) else float(np.nanmax(matrixMunc))
    if not np.isfinite(upper) or upper <= 0.0:
        upper = max(float(fallbackMinR), 1.0)
    upper = float(max(upper, 0.0))
    fallback = float(np.clip(float(fallbackMinR), 0.0, upper))
    logger.info(
        "observationFloorCalibration.start chrom=%s intervals=%s samples=%s folds=%s blockLen=%s holdoutReps=%s",
        chromosome,
        n,
        m,
        folds,
        blockLen,
        holdoutCount,
    )

    masks = _makeFoldMasks(
        m=m,
        n=n,
        blockLen=blockLen,
        folds=folds,
        holdoutCount=holdoutCount,
        seed=int(params.seed),
    )
    excludeMask = None
    if excludeIntervals is not None:
        excludeMask = np.asarray(excludeIntervals, dtype=bool).reshape(-1)
        if excludeMask.shape[0] != n:
            raise ValueError("excludeIntervals must match the number of matrixData columns")

    fitKwargs = dict(runKwargs)
    fitKwargs.setdefault("logRunRole", "observation-floor held-out fold")
    foldIndentLevel = max(0, int(fitKwargs.get("logIndentLevel", 0) or 0))
    fitKwargs["logIndentLevel"] = foldIndentLevel + 1
    fitKwargs["ECM_fixedBackgroundIters"] = max(
        int(params.calibrationECMIters),
        core.UNCERTAINTY_CALIBRATION_MIN_CALIBRATION_ECM_ITERS,
    )
    fitKwargs["ECM_outerIters"] = 1
    fitKwargs["ECM_minOuterIters"] = 1
    fitKwargs["processNoiseWarmupECMIters"] = (
        core.UNCERTAINTY_CALIBRATION_REFIT_PROCESS_NOISE_WARMUP_ECM_ITERS
    )
    fitKwargs["returnScales"] = True
    fitKwargs["returnReplicateOffsets"] = True
    fitKwargs["returnBackground"] = True
    fitKwargs["returnPrecisionDiagnostics"] = True

    residualChunks: list[np.ndarray] = []
    pChunks: list[np.ndarray] = []
    muncChunks: list[np.ndarray] = []
    lambdaChunks: list[np.ndarray] = []
    iChunks: list[np.ndarray] = []
    jChunks: list[np.ndarray] = []
    foldChunks: list[np.ndarray] = []
    usedLambda = False
    refitSeconds = 0.0
    extractSeconds = 0.0

    for fold, mask in _progress(
        list(enumerate(masks)),
        params=params,
        desc="Observation-floor folds",
        unit="fold",
    ):
        stageStart = time.perf_counter()
        out = core.runConsenrich(
            matrixData,
            matrixMunc,
            observationMask=mask,
            **fitKwargs,
        )
        refitSeconds += time.perf_counter() - stageStart
        (
            stateMasked,
            covarMasked,
            _resid,
            _track4,
            biasMasked,
            _blockMap,
            backgroundMasked,
        ) = out[:7]
        precisionDiagnostics = next(
            (
                item
                for item in out[7:]
                if isinstance(item, dict) and item.get("precision_track_diagnostics") is True
            ),
            {},
        )
        lambdaExp = precisionDiagnostics.get("lambdaExp") if precisionDiagnostics else None
        if lambdaExp is not None:
            lambdaExp = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
            if lambdaExp.shape[0] != n:
                lambdaExp = None

        stageStart = time.perf_counter()
        residual, pHeld, _rHeld, ii, jj, foldHeld = _cuncertainty.cextractHeldoutScores(
            matrixData,
            matrixMunc,
            np.ascontiguousarray(np.asarray(stateMasked)[:, 0], dtype=np.float32),
            np.ascontiguousarray(np.asarray(covarMasked)[:, 0, 0], dtype=np.float32),
            np.ascontiguousarray(biasMasked, dtype=np.float32),
            np.ascontiguousarray(backgroundMasked, dtype=np.float32),
            np.ascontiguousarray(mask, dtype=np.uint8),
            int(fold),
            float(padValue),
            float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        extractSeconds += time.perf_counter() - stageStart
        if ii.size == 0:
            continue
        valid = np.ones(ii.shape[0], dtype=bool)
        if excludeMask is not None:
            valid &= ~excludeMask[ii]
        if not np.any(valid):
            continue
        residual = residual[valid]
        pHeld = pHeld[valid]
        ii = ii[valid]
        jj = jj[valid]
        foldHeld = foldHeld[valid]
        muncBase = np.asarray(matrixMunc[jj, ii], dtype=np.float64)
        if lambdaExp is None:
            lambdaHeld = np.ones(residual.shape[0], dtype=np.float64)
        else:
            lambdaHeld = np.maximum(
                lambdaExp[ii].astype(np.float64, copy=False),
                core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
            )
            usedLambda = True
        validScores = (
            np.isfinite(residual)
            & np.isfinite(pHeld)
            & np.isfinite(muncBase)
            & np.isfinite(lambdaHeld)
            & (pHeld >= 0.0)
            & (muncBase >= 0.0)
            & (lambdaHeld > 0.0)
        )
        if not np.any(validScores):
            continue
        residualChunks.append(residual[validScores])
        pChunks.append(pHeld[validScores])
        muncChunks.append(muncBase[validScores])
        lambdaChunks.append(lambdaHeld[validScores])
        iChunks.append(ii[validScores].astype(np.int64))
        jChunks.append(jj[validScores].astype(np.int64))
        foldChunks.append(foldHeld[validScores].astype(np.int32))

    if not residualChunks:
        return observationVarianceFloorCalibrationResult(
            minR=fallback,
            trimmedMean=float("nan"),
            target=1.0,
            heldoutCells=0,
            fitCells=0,
            usedLambda=False,
            hitUpperBound=False,
            fallbackUsed=True,
            diagnostics={
                "reason": "no_heldout_scores",
                "fallback_min_r": fallback,
                "elapsed_seconds": time.perf_counter() - totalStart,
            },
        )

    residual = np.concatenate(residualChunks)
    pState = np.concatenate(pChunks)
    muncBase = np.concatenate(muncChunks)
    lambdaHeld = np.concatenate(lambdaChunks) if usedLambda else None
    intervalIndex = np.concatenate(iChunks)
    repIndex = np.concatenate(jChunks)
    foldIndex = np.concatenate(foldChunks)
    totalHeldoutCells = int(residual.size)
    codes = _scoreSamplingCodesWithoutState(
        foldIndex=foldIndex,
        repIndex=repIndex,
        pState=pState,
    )
    fitRows = _samplePositionsByCode(
        codes,
        maxRows=_maxScoreRows(params),
        seed=int(params.seed) + TARGET_CALIBRATION_BLOCK_SPLIT_SEED_OFFSET,
    )
    residualFit = residual[fitRows]
    pStateFit = pState[fitRows]
    muncFit = muncBase[fitRows]
    lambdaFit = None if lambdaHeld is None else lambdaHeld[fitRows]
    selected, score, hitUpper = _selectObservationVarianceFloor(
        residual=residualFit,
        pState=pStateFit,
        muncBase=muncFit,
        lambdaExp=lambdaFit,
        pad=padValue,
        lower=0.0,
        upper=upper,
        target=1.0,
    )
    if not np.isfinite(score):
        selected = fallback
        score = _observationFloorCalibrationMean(
            residual=residualFit,
            pState=pStateFit,
            muncBase=muncFit,
            lambdaExp=lambdaFit,
            r=selected,
            pad=padValue,
        )
        fallbackUsed = True
    else:
        fallbackUsed = False
    selected = float(np.clip(float(selected), 0.0, upper))
    logger.info(
        "observationFloorCalibration.done chrom=%s minR=%.6g trimmedMean=%.6g heldoutCells=%d fitCells=%d usedLambda=%s hitUpper=%s elapsed=%.3fs",
        chromosome,
        selected,
        float(score),
        totalHeldoutCells,
        int(residualFit.size),
        bool(usedLambda),
        bool(hitUpper),
        time.perf_counter() - totalStart,
    )
    return observationVarianceFloorCalibrationResult(
        minR=selected,
        trimmedMean=float(score),
        target=1.0,
        heldoutCells=totalHeldoutCells,
        fitCells=int(residualFit.size),
        usedLambda=bool(usedLambda),
        hitUpperBound=bool(hitUpper),
        fallbackUsed=bool(fallbackUsed),
        diagnostics={
            "block_len_intervals": int(blockLen),
            "folds": int(folds),
            "holdout_replicates": int(holdoutCount),
            "max_r": float(upper),
            "fallback_min_r": float(fallback),
            "refit_seconds": float(refitSeconds),
            "extract_seconds": float(extractSeconds),
            "elapsed_seconds": time.perf_counter() - totalStart,
        },
    )


def calibrateChromosomeStateUncertainty(
    *,
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    fullState: np.ndarray,
    fullCovar: np.ndarray | None = None,
    fullP: np.ndarray | None = None,
    fullReplicateBias: np.ndarray | None = None,
    intervals: np.ndarray | None = None,
    intervalSizeBP: int,
    params: core.uncertaintyCalibrationParams,
    runKwargs: dict[str, Any],
    pad: float | None = None,
    outPrefix: str | None = None,
    chromosome: str | None = None,
) -> uncertaintyCalibrationResult:
    totalStart = time.perf_counter()
    timings: dict[str, float] = {}
    # Retained for API compatibility; replicate bias is already reflected in runKwargs.
    _ = fullReplicateBias
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
    padValue = _calibrationPad(params, pad)
    blockLen = _resolveBlockSizeIntervals(params.blockSizeBP, intervalSizeBP, n)
    holdoutFraction = _firstSet(
        params,
        "holdoutFraction",
        "heldoutReplicateFraction",
        default=None,
    )
    holdoutCount = _resolveHoldoutCount(m, holdoutFraction)
    folds = max(int(params.folds), core.UNCERTAINTY_CALIBRATION_MIN_FOLDS)
    logger.info(
        "uncertaintyCalibration.start intervals=%s samples=%s folds=%s blockLen=%s holdoutReps=%s cython=%s",
        n,
        m,
        folds,
        blockLen,
        holdoutCount,
        1,
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
    stageStart = time.perf_counter()
    featureMatrix, featureNames, featureCenter, featureScale = _featureMatrix(
        state=fullState0,
        stateVar=fullPArr,
        matrixMunc=matrixMunc,
    )
    timings["feature_matrix_seconds"] = time.perf_counter() - stageStart
    residualChunks: list[np.ndarray] = []
    pChunks: list[np.ndarray] = []
    rChunks: list[np.ndarray] = []
    iChunks: list[np.ndarray] = []
    jChunks: list[np.ndarray] = []
    foldChunks: list[np.ndarray] = []

    fitKwargs = dict(runKwargs)
    fitKwargs.setdefault("logRunRole", "held-out fold")
    foldIndentLevel = max(0, int(fitKwargs.get("logIndentLevel", 0) or 0))
    fitKwargs["logIndentLevel"] = foldIndentLevel + 1
    fitKwargs["ECM_fixedBackgroundIters"] = max(
        int(params.calibrationECMIters),
        core.UNCERTAINTY_CALIBRATION_MIN_CALIBRATION_ECM_ITERS,
    )
    fitKwargs["ECM_outerIters"] = 1
    fitKwargs["ECM_minOuterIters"] = 1
    fitKwargs["processNoiseWarmupECMIters"] = (
        core.UNCERTAINTY_CALIBRATION_REFIT_PROCESS_NOISE_WARMUP_ECM_ITERS
    )
    fitKwargs["returnScales"] = True
    fitKwargs["returnReplicateOffsets"] = True
    fitKwargs["returnBackground"] = True

    refitSeconds = 0.0
    extractSeconds = 0.0
    extractProgress = tqdm(
        total=len(masks),
        desc="Extracting held-out scores",
        unit="fold",
        disable=not _progressEnabled(params),
        mininterval=0.5,
        leave=False,
        dynamic_ncols=True,
    )
    for fold, mask in _progress(
        list(enumerate(masks)),
        params=params,
        desc="Uncertainty calibration folds",
        unit="fold",
    ):
        core._logAsciiBlock(
            "uncertainty calibration fold",
            (
                ("fold", f"{int(fold + 1)}/{int(len(masks))}"),
                ("intervals", int(n)),
                ("total folds", int(len(masks))),
                (
                    (
                        "process noise warm-start"
                        if fitKwargs.get("initialProcessQ") is not None
                        else "process noise warmup"
                    ),
                    (
                        "initialProcessQ"
                        if fitKwargs.get("initialProcessQ") is not None
                        else (
                            f"{int(core.PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES)} "
                            f"outer passes x "
                            f"{int(fitKwargs['processNoiseWarmupECMIters'])} ECM iterations"
                        )
                    ),
                ),
            ),
            logger_=logger,
            indentLevel=foldIndentLevel,
        )
        logger.info("uncertaintyCalibration.fold.start fold=%s intervals=%s", fold, n)
        stageStart = time.perf_counter()
        out = core.runConsenrich(
            matrixData,
            matrixMunc,
            observationMask=mask,
            **fitKwargs,
        )
        refitSeconds += time.perf_counter() - stageStart
        (
            stateMasked,
            covarMasked,
            _resid,
            _track4,
            biasMasked,
            _blockMap,
            backgroundMasked,
        ) = out[:7]
        stageStart = time.perf_counter()
        residual, pHeld, rHeld, ii, jj, foldHeld = _cuncertainty.cextractHeldoutScores(
            matrixData,
            matrixMunc,
            np.ascontiguousarray(np.asarray(stateMasked)[:, 0], dtype=np.float32),
            np.ascontiguousarray(np.asarray(covarMasked)[:, 0, 0], dtype=np.float32),
            np.ascontiguousarray(biasMasked, dtype=np.float32),
            np.ascontiguousarray(backgroundMasked, dtype=np.float32),
            np.ascontiguousarray(mask, dtype=np.uint8),
            int(fold),
            float(padValue),
            float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        extractSeconds += time.perf_counter() - stageStart
        extractProgress.update(1)
        if ii.size == 0:
            logger.info("uncertaintyCalibration.fold.done fold=%s heldoutCells=0", fold)
            continue
        residualChunks.append(residual)
        pChunks.append(pHeld)
        rChunks.append(rHeld)
        iChunks.append(ii.astype(np.int64))
        jChunks.append(jj.astype(np.int64))
        foldChunks.append(foldHeld.astype(np.int32))
        logger.info("uncertaintyCalibration.fold.done fold=%s heldoutCells=%s", fold, ii.size)
    extractProgress.close()
    timings["masked_refits_seconds"] = refitSeconds
    timings["extract_scores_seconds"] = extractSeconds

    if not residualChunks:
        raise ValueError("uncertainty calibration produced no held-out cells")
    residual = np.concatenate(residualChunks)
    pState = np.concatenate(pChunks)
    obsVar = np.concatenate(rChunks)
    intervalIndex = np.concatenate(iChunks)
    repIndex = np.concatenate(jChunks)
    foldIndex = np.concatenate(foldChunks)
    blockIndex = (intervalIndex // int(blockLen)).astype(np.int64, copy=False)
    totalHeldoutCells = int(residual.size)
    if residual.size < int(params.minHeldoutCells):
        logger.warning(
            "uncertaintyCalibration.lowHeldoutCells heldoutCells=%s minHeldoutCells=%s; fitting with available cells",
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
    sampleCodes = _scoreSamplingCodes(
        foldIndex=foldIndex,
        repIndex=repIndex,
        intervalIndex=intervalIndex,
        pState=pState,
        fullState=fullState0,
    )
    fitRowsLocal = _samplePositionsByCode(
        sampleCodes[scaleRows],
        maxRows=_maxScoreRows(params),
        seed=int(params.seed),
    )
    fitRows = scaleRows[fitRowsLocal]
    residualFit = residual[fitRows]
    pStateFit = pState[fitRows]
    obsVarFit = obsVar[fitRows]
    intervalIndexFit = intervalIndex[fitRows]
    repIndexFit = repIndex[fitRows]
    foldIndexFit = foldIndex[fitRows]
    logger.info(
        "uncertaintyCalibration.sample heldoutCells=%s fitCells=%s maxHeldoutCells=%s",
        totalHeldoutCells,
        int(residualFit.size),
        _maxScoreRows(params),
    )
    stageStart = time.perf_counter()
    beta, modelMeta = _fitFactorModel(
        residual=residualFit,
        pState=pStateFit,
        obsVar=obsVarFit,
        featureByInterval=featureMatrix,
        intervalIndex=intervalIndexFit,
        params=params,
    )
    timings["fit_factor_seconds"] = time.perf_counter() - stageStart
    stageStart = time.perf_counter()
    factor, calibrated = _cuncertainty.cevaluateFactor(
        np.ascontiguousarray(featureMatrix, dtype=np.float64),
        np.ascontiguousarray(beta, dtype=np.float64),
        np.ascontiguousarray(fullPArr, dtype=np.float64),
        _factorBounds(params)[0],
        _factorBounds(params)[1],
    )
    timings["evaluate_factor_seconds"] = time.perf_counter() - stageStart
    heldFactor = factor[intervalIndexFit].astype(np.float64)
    sdBefore = np.sqrt(
        np.maximum(
            pStateFit + obsVarFit,
            core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
        )
    )
    aObsFactor = float(modelMeta.get("a_obs_factor", 1.0))
    sdAfter = np.sqrt(
        np.maximum(
            heldFactor * pStateFit + aObsFactor * obsVarFit,
            core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
        )
    )
    heldFactorAll = factor[intervalIndex].astype(np.float64)
    sdBeforeAll = np.sqrt(
        np.maximum(
            pState + obsVar,
            core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
        )
    )
    sdAfterAll = np.sqrt(
        np.maximum(
            heldFactorAll * pState + aObsFactor * obsVar,
            core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
        )
    )
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
            _cuncertainty.ctargetBlockScores(
                np.ascontiguousarray(residual, dtype=np.float64),
                np.ascontiguousarray(pState, dtype=np.float64),
                np.ascontiguousarray(obsVar, dtype=np.float64),
                np.ascontiguousarray(factor, dtype=np.float64),
                np.ascontiguousarray(intervalIndex, dtype=np.int64),
                np.ascontiguousarray(blockIndex, dtype=np.int64),
                np.ascontiguousarray(targetSplit["target_block_mask"], dtype=np.uint8),
                aObsFactor,
                float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
            )
        )
        targetCalibrationBounds = _targetCalibrationBounds(
            targetBlockScores,
            targets=tuple(float(t) for t in params.targets),
            delta=float(targetDelta),
        )
    scaleByTargetCalibration = bool(
        getattr(params, "scaleUncertaintyByTargetCalibration", True)
    )
    targetScaleBound = _targetCalibrationScaleBound(targetCalibrationBounds)
    uncertaintyTrackScale = 1.0
    uncertaintyTrackScaleTarget = None
    uncertaintyTrackScaled = False
    uncertaintyTrackScaleCertified = False
    uncertaintyTrackScaleReason = "target_calibration_disabled"
    if targetCalibrationEnabled:
        uncertaintyTrackScaleReason = "scale_disabled_by_config"
        if scaleByTargetCalibration:
            uncertaintyTrackScaleReason = "no_target_bound"
            if targetScaleBound is not None:
                uncertaintyTrackScaleTarget = float(targetScaleBound["target"])
                qValue = targetScaleBound.get("q")
                if qValue is not None:
                    qFloat = float(qValue)
                    if np.isfinite(qFloat) and qFloat > 0.0:
                        uncertaintyTrackScale = qFloat
                        uncertaintyTrackScaled = True
                        uncertaintyTrackScaleCertified = bool(
                            targetScaleBound.get("certified", False)
                        )
                        uncertaintyTrackScaleReason = (
                            "scaled_by_certified_target_bound"
                            if uncertaintyTrackScaleCertified
                            else "scaled_by_uncertified_empirical_max"
                        )
                        calibrated = (
                            np.asarray(calibrated, dtype=np.float32)
                            * np.float32(uncertaintyTrackScale)
                        )
                        if uncertaintyTrackScaleCertified:
                            logger.info(
                                "Target-calibrated uncertainty scaling applied: "
                                "target=%.6g delta=%.6g q=%.6g. To write unscaled "
                                "calibrated standard errors, set "
                                "`uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: false`.",
                                uncertaintyTrackScaleTarget,
                                float(targetDelta),
                                uncertaintyTrackScale,
                            )
                        else:
                            logger.warning(
                                "Target-calibrated uncertainty scaling applied using "
                                "uncertified empirical max multiplier: target=%.6g "
                                "delta=%.6g q=%.6g. The requested PAC bound was not "
                                "certified with the available target-calibration blocks. "
                                "To write unscaled calibrated standard errors, set "
                                "`uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: false`.",
                                uncertaintyTrackScaleTarget,
                                float(targetDelta),
                                uncertaintyTrackScale,
                            )
                    else:
                        uncertaintyTrackScaleReason = "nonfinite_target_bound"
                else:
                    uncertaintyTrackScaleReason = "no_finite_target_bound"
            if not uncertaintyTrackScaled:
                logger.warning(
                    "Target-calibrated uncertainty scaling was requested but no finite "
                    "multiplier was available for the selected target. Writing "
                    "unscaled calibrated standard errors. To disable target-calibrated "
                    "scaling explicitly, set "
                    "`uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: false`."
                )
        else:
            logger.info(
                "Target-calibrated uncertainty scaling disabled by configuration. "
                "Set `uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: true` "
                "to scale the uncertainty track by the selected certified multiplier."
            )
    coverageOverall = [
        row for row in stateCoverage if str(row.get("stratum", "")) == "overall"
    ]
    if coverageOverall:
        logger.info(
            "uncertaintyCalibration.coverage.all %s",
            " ".join(
                "target={target:.3g} before={coverage_before:.3f} after={coverage_after:.3f} n={n}".format(
                    **row
                )
                for row in coverageOverall
                if row.get("coverage_after") is not None
            ),
        )
    stageStart = time.perf_counter()
    scores = pd.DataFrame(
        {
            "fold": foldIndexFit,
            "replicate": repIndexFit,
            "interval_index": intervalIndexFit,
            "block_index": blockIndex[fitRows],
            "chrom_start": intervalsArr[intervalIndexFit],
            "residual": residualFit,
            "state_variance": pStateFit,
            "observation_variance": obsVarFit,
            "sd_before": sdBefore,
            "sd_after": sdAfter,
            "a_state": heldFactor,
            "a_obs": np.full_like(heldFactor, aObsFactor),
        }
    )
    try:
        scores["uncertainty_decile"] = pd.qcut(
            scores["state_variance"],
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
        _scoreSamplingCodes(
            foldIndex=scores["fold"].to_numpy(dtype=np.int64),
            repIndex=scores["replicate"].to_numpy(dtype=np.int64),
            intervalIndex=scores["interval_index"].to_numpy(dtype=np.int64),
            pState=scores["state_variance"].to_numpy(dtype=np.float64),
            fullState=fullState0,
        ),
        maxRows=int(params.maxDiagnosticRows),
        seed=int(params.seed) + core.UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET,
    )
    scoresDiagnostics = scores.iloc[diagnosticRows, :].reset_index(drop=True)
    timings["summarize_scores_seconds"] = time.perf_counter() - stageStart
    model = {
        **modelMeta,
        "beta": [float(x) for x in beta],
        "feature_names": featureNames,
        "feature_center": [float(x) for x in featureCenter],
        "feature_scale": [float(x) for x in featureScale],
        "block_len_intervals": int(blockLen),
        "state_roughness": stateRoughness,
        "holdout_replicates_per_block": int(holdoutCount),
        "heldout_cells": totalHeldoutCells,
        "fit_heldout_cells": int(residualFit.size),
        "diagnostic_score_rows": int(scoresDiagnostics.shape[0]),
        "max_scores": int(_maxScoreRows(params)),
        "max_diagnostic_rows": int(params.maxDiagnosticRows),
        "folds": int(folds),
        "block_size_bp": (
            None
            if params.blockSizeBP is None
            else str(params.blockSizeBP)
        ),
        "holdout_fraction": (
            None if holdoutFraction is None else float(holdoutFraction)
        ),
        "targets": [float(target) for target in params.targets],
        "ridge": float(max(params.ridge, 0.0)),
        "wis_weight": float(_wisWeight(params)),
        "a_obs_penalty": float(_aObsPenalty(params)),
        "factor_bound_min": float(_factorBounds(params)[0]),
        "factor_bound_max": float(_factorBounds(params)[1]),
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
            "uncertainty_track_scale_certified": bool(uncertaintyTrackScaleCertified),
            "uncertainty_track_scale_reason": uncertaintyTrackScaleReason,
            "score_definition": "max_abs_residual_over_predictive_sd_by_block",
            "bounds": targetCalibrationBounds,
        },
    }
    timings["total_seconds"] = time.perf_counter() - totalStart
    model["timings_seconds"] = {key: float(value) for key, value in timings.items()}
    logger.info(
        "uncertaintyCalibration.fit.done heldoutCells=%s objective=%.6g aObs=%.6g elapsed=%.3fs",
        totalHeldoutCells,
        float(modelMeta.get("objective", np.nan)),
        float(modelMeta.get("a_obs_factor", np.nan)),
        timings["total_seconds"],
    )
    if outPrefix is not None and bool(params.writeDiagnostics):
        prefix = Path(outPrefix)
        diagnosticsPath = str(prefix) + ".diagnostics.tsv.gz"
        modelPath = str(prefix) + ".model.json"
        diagnosticsTable = _diagnosticsTable(
            scores=scoresDiagnostics,
            summary=summary,
            model=model,
            chromosome=chromosome,
        )
        for path in _progress(
            [diagnosticsPath],
            params=params,
            desc="Writing uncertainty diagnostics",
            unit="file",
        ):
            pathObj = Path(path)
            diagnosticsTable.to_csv(
                path,
                sep="\t",
                index=False,
                mode="a" if pathObj.exists() else "w",
                header=not pathObj.exists(),
                compression="gzip",
            )
            logger.info("uncertaintyCalibration.output wrote %s", path)
        _writeModelJson(modelPath, model, chromosome)
        logger.info("uncertaintyCalibration.output wrote %s", modelPath)
    return uncertaintyCalibrationResult(
        factor=factor,
        calibratedUncertainty=calibrated,
        summary=summary,
        scores=scores,
        model=model,
    )


__all__ = [
    "calibrateChromosomeStateUncertainty",
    "estimateObservationVarianceFloorFromHeldout",
    "observationVarianceFloorCalibrationResult",
    "uncertaintyCalibrationResult",
]
