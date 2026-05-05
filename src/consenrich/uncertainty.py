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
from . import cuncertainty as _cuncertainty


logger = logging.getLogger(__name__)


class uncertaintyCalibrationResult(NamedTuple):
    factor: np.ndarray
    calibratedUncertainty: np.ndarray
    summary: pd.DataFrame
    scores: pd.DataFrame
    model: dict[str, Any]


def _progressEnabled(params: core.uncertaintyCalibrationParams) -> bool:
    return bool(params.writeDiagnostics) and bool(getattr(sys.stderr, "isatty", lambda: False)())


def _progress(iterable, *, params: core.uncertaintyCalibrationParams, **kwargs):
    return tqdm(iterable, disable=not _progressEnabled(params), **kwargs)


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
    if blockSizeBP is None or str(blockSizeBP).lower() == "auto":
        targetBP = max(
            core.UNCERTAINTY_CALIBRATION_AUTO_BLOCK_MIN_BP,
            core.UNCERTAINTY_CALIBRATION_AUTO_BLOCK_INTERVAL_MULTIPLIER
            * int(intervalSizeBP),
        )
    else:
        targetBP = int(blockSizeBP)
    minBlockIntervals = core.UNCERTAINTY_CALIBRATION_MIN_BLOCK_INTERVALS
    return int(
        np.clip(
            round(targetBP / max(int(intervalSizeBP), 1)),
            minBlockIntervals,
            max(n, minBlockIntervals),
        )
    )


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
    fitKwargs["EM_maxIters"] = max(
        int(params.calibrationEMIters),
        core.UNCERTAINTY_CALIBRATION_MIN_CALIBRATION_EM_ITERS,
    )
    fitKwargs["EM_outerIters"] = 1
    fitKwargs["returnScales"] = True
    fitKwargs["returnReplicateOffsets"] = True
    fitKwargs["applyJackknife"] = False
    fitKwargs["autoDeltaF"] = False

    refitSeconds = 0.0
    extractSeconds = 0.0
    extractProgress = tqdm(
        total=len(masks),
        desc="Extracting held-out scores",
        unit="fold",
        disable=not _progressEnabled(params),
    )
    for fold, mask in _progress(
        list(enumerate(masks)),
        params=params,
        desc="Uncertainty calibration folds",
        unit="fold",
    ):
        logger.info("uncertaintyCalibration.fold.start fold=%s intervals=%s", fold, n)
        stageStart = time.perf_counter()
        out = core.runConsenrich(
            matrixData,
            matrixMunc,
            observationMask=mask,
            **fitKwargs,
        )
        refitSeconds += time.perf_counter() - stageStart
        stateMasked, covarMasked, _resid, _track4, _qScale, biasMasked, _blockMap = out
        stageStart = time.perf_counter()
        residual, pHeld, rHeld, ii, jj, foldHeld = _cuncertainty.cextractHeldoutScores(
            matrixData,
            matrixMunc,
            np.ascontiguousarray(np.asarray(stateMasked)[:, 0], dtype=np.float32),
            np.ascontiguousarray(np.asarray(covarMasked)[:, 0, 0], dtype=np.float32),
            np.ascontiguousarray(biasMasked, dtype=np.float32),
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
    totalHeldoutCells = int(residual.size)
    if residual.size < int(params.minHeldoutCells):
        logger.warning(
            "uncertaintyCalibration.lowHeldoutCells heldoutCells=%s minHeldoutCells=%s; fitting with available cells",
            int(residual.size),
            int(params.minHeldoutCells),
        )
    sampleCodes = _scoreSamplingCodes(
        foldIndex=foldIndex,
        repIndex=repIndex,
        intervalIndex=intervalIndex,
        pState=pState,
        fullState=fullState0,
    )
    fitRows = _samplePositionsByCode(
        sampleCodes,
        maxRows=_maxScoreRows(params),
        seed=int(params.seed),
    )
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
    stageStart = time.perf_counter()
    scores = pd.DataFrame(
        {
            "fold": foldIndexFit,
            "replicate": repIndexFit,
            "interval_index": intervalIndexFit,
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
        "factor_min": float(np.min(factor)),
        "factor_median": float(np.median(factor)),
        "factor_max": float(np.max(factor)),
    }
    timings["total_seconds"] = time.perf_counter() - totalStart
    model["timings_seconds"] = {key: float(value) for key, value in timings.items()}
    logger.info(
        "uncertaintyCalibration.fit.done heldoutCells=%s objective=%.6g aObs=%.6g factorMin=%.6g factorMed=%.6g factorMax=%.6g elapsed=%.3fs",
        totalHeldoutCells,
        float(modelMeta.get("objective", np.nan)),
        float(modelMeta.get("a_obs_factor", np.nan)),
        model["factor_min"],
        model["factor_median"],
        model["factor_max"],
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


__all__ = ["calibrateChromosomeStateUncertainty", "uncertaintyCalibrationResult"]
