"""Post-fit empirical-Bayes shrinkage for Consenrich state estimates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

try:  # cython is optional in this mod. for now
    from . import cconsenrich as _cconsenrich
except Exception:  # pragma: no cover - exercised only when extension is absent.
    _cconsenrich = None


STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL = "spikeAndNormal"
STATE_SHRINKAGE_MODEL = STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL
STATE_SHRINKAGE_MODELS = (STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,)
_STATE_SHRINKAGE_POSITIVE_FLOOR = 1.0e-12
_STATE_SHRINKAGE_DEFAULT_MAX_ITER = 96
_STATE_SHRINKAGE_DEFAULT_TOL = 1.0e-5
_STATE_SHRINKAGE_DEFAULT_NULL_Z = 1.5
_STATE_SHRINKAGE_DEFAULT_MIN_NULL = 1.0e-2
_STATE_SHRINKAGE_DEFAULT_MAX_NULL = 1.0 - 1.0e-4


class stateShrinkPrior(NamedTuple):
    r"""Specifies genome-wide EB prior for state shrinkage (spike+normal, mixture)."""

    model: str
    priorNull: float
    priorScale: float
    priorVariance: float
    blockSize: int
    metadata: dict[str, Any]


class stateShrinkResult(NamedTuple):
    r"""Post-fit empirical-Bayes shrinkage result.

    ``shrunkState`` is the posterior mean under the fitted spike-and-normal
    prior, ``posteriorSd`` is the posterior standard deviation under that same
    prior, ``shrinkageFactor`` is ``shrunkState / state`` where defined, and
    ``nullProbability`` is the posterior probability assigned to the point-null
    component (prob. state = 0) of the fitted prior. ``slabPosteriorMean`` and
    ``slabPosteriorWeight`` are the posterior mean and weight of the normal slab
    component of the fitted prior.
    """

    shrunkState: np.ndarray
    posteriorSd: np.ndarray
    shrinkageFactor: np.ndarray
    nullProbability: np.ndarray
    slabPosteriorMean: np.ndarray
    slabPosteriorWeight: np.ndarray
    priorNull: float
    priorScale: float
    metadata: dict[str, Any]


def _metadataFloat(value: float) -> float | None:
    value_ = float(value)
    return value_ if np.isfinite(value_) else None


def _normalizeModel(model: str | None) -> str:
    if model is None:
        return STATE_SHRINKAGE_MODEL
    key = str(model).strip().replace("-", "_").replace(" ", "_").lower()
    keyCompact = key.replace("_", "")
    if keyCompact in {"spikeandnormal", "spikenormal", "spike", "pointnormal"}:
        return STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL
    supported = ", ".join(STATE_SHRINKAGE_MODELS)
    raise ValueError(
        f"Unsupported state shrinkage model {model!r}; supported values: {supported}."
    )


def _asFloatVector(name: str, values: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"`{name}` must be non-empty")
    return arr


def _loadStateVarianceItem(item: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(item, Mapping):
        stateSource = item.get("state")
        varianceSource = item.get("variance")
        statePath = item.get("statePath") or item.get("stateFile")
        variancePath = item.get("variancePath") or item.get("varianceFile")
        if stateSource is None:
            if statePath is None:
                raise ValueError("state shrinkage item is missing `state`/`statePath`")
            stateSource = np.load(statePath, allow_pickle=False, mmap_mode="r")
        if varianceSource is None:
            if variancePath is None:
                raise ValueError(
                    "state shrinkage item is missing `variance`/`variancePath`"
                )
            varianceSource = np.load(variancePath, allow_pickle=False, mmap_mode="r")
        state = _asFloatVector("state", stateSource)
        variance = _asFloatVector("stateVariance", varianceSource)
    else:
        try:
            stateSource, varianceSource = item
        except Exception as exc:  # pragma: no cover - defensive only.
            raise TypeError(
                "state shrinkage chunks must be mappings or (state, variance) pairs"
            ) from exc
        state = _asFloatVector("state", stateSource)
        variance = _asFloatVector("stateVariance", varianceSource)
    if state.shape != variance.shape:
        raise ValueError("`state` and `stateVariance` must have the same length")
    return state, variance


def _nullBounds(minNull: float, maxNull: float) -> tuple[float, float]:
    minNull_ = float(np.clip(minNull, _STATE_SHRINKAGE_DEFAULT_MIN_NULL, 0.5))
    maxNull_ = float(np.clip(maxNull, 0.5, _STATE_SHRINKAGE_DEFAULT_MAX_NULL))
    if minNull_ >= maxNull_:
        raise ValueError("`minNull` must be smaller than `maxNull`")
    return minNull_, maxNull_


def _safeBlockSize(blockSize: int | None) -> int:
    if blockSize is None:
        return 1
    return int(max(1, blockSize))


def _logNormalZeroDensity(x: np.ndarray, variance: np.ndarray) -> np.ndarray:
    var = np.maximum(
        np.asarray(variance, dtype=np.float64),
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )
    return -0.5 * (
        np.log(2.0 * math.pi * var) + (np.asarray(x, dtype=np.float64) ** 2) / var
    )


def _iterBlockWeights(state: np.ndarray, variance: np.ndarray, blockSize: int):
    blockSize_ = _safeBlockSize(blockSize)
    n = int(state.size)
    for start in range(0, n, blockSize_):
        end = min(start + blockSize_, n)
        blockValid = (
            np.isfinite(state[start:end])
            & np.isfinite(variance[start:end])
            & (variance[start:end] > 0.0)
        )
        validCount = int(np.count_nonzero(blockValid))
        if validCount <= 0:
            continue
        idx = np.nonzero(blockValid)[0] + start
        yield idx, 1.0 / float(validCount)


def _initialSumsPython(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    nullZ: float,
    blockSize: int,
) -> tuple[float, float, float, float, float, int]:
    totalWeight = 0.0
    centralWeight = 0.0
    excessMomentSum = 0.0
    varianceSum = 0.0
    stateSqSum = 0.0
    finiteCount = 0
    nullZ_ = float(max(nullZ, _STATE_SHRINKAGE_POSITIVE_FLOOR))
    for idx, weight in _iterBlockWeights(state, variance, blockSize):
        x = np.asarray(state[idx], dtype=np.float64)
        v = np.maximum(
            np.asarray(variance[idx], dtype=np.float64), _STATE_SHRINKAGE_POSITIVE_FLOOR
        )
        z = np.abs(x) / np.sqrt(v)
        totalWeight += weight * float(x.size)
        centralWeight += weight * float(np.count_nonzero(z <= nullZ_))
        excessMomentSum += weight * float(np.sum(x * x - v))
        varianceSum += weight * float(np.sum(v))
        stateSqSum += weight * float(np.sum(x * x))
        finiteCount += int(x.size)
    return (
        totalWeight,
        centralWeight,
        excessMomentSum,
        varianceSum,
        stateSqSum,
        finiteCount,
    )


def _initialSums(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    nullZ: float,
    blockSize: int,
) -> tuple[float, float, float, float, float, int]:
    func = getattr(_cconsenrich, "cstateShrinkSpikeNormalInitialSums", None)
    if func is not None:
        return tuple(func(state, variance, float(nullZ), int(blockSize)))  # type: ignore[return-value]
    # python if cython not available
    return _initialSumsPython(state, variance, nullZ=nullZ, blockSize=blockSize)


def _emStepPython(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorNull: float,
    priorVariance: float,
    blockSize: int,
) -> tuple[float, float, float, float, float, int]:
    pi0 = float(
        np.clip(
            priorNull,
            _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
            _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
        )
    )
    tau2 = float(max(priorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR))
    totalWeight = 0.0
    nullMass = 0.0
    slabMass = 0.0
    slabSecond = 0.0
    logLikelihood = 0.0
    finiteCount = 0
    for idx, weight in _iterBlockWeights(state, variance, blockSize):
        x = np.asarray(state[idx], dtype=np.float64)
        v = np.maximum(
            np.asarray(variance[idx], dtype=np.float64), _STATE_SHRINKAGE_POSITIVE_FLOOR
        )
        logNull = math.log(pi0) + _logNormalZeroDensity(x, v)
        logSlab = math.log1p(-pi0) + _logNormalZeroDensity(x, v + tau2)
        logDenom = np.logaddexp(logNull, logSlab)
        nullProb = np.exp(logNull - logDenom)
        slabWeight = np.maximum(1.0 - nullProb, 0.0)
        slabShrinkage = tau2 / (tau2 + v)
        slabMean = slabShrinkage * x
        slabVariance = slabShrinkage * v
        totalWeight += weight * float(x.size)
        nullMass += weight * float(np.sum(nullProb))
        slabMass += weight * float(np.sum(slabWeight))
        slabSecond += weight * float(
            np.sum(slabWeight * (slabVariance + slabMean * slabMean))
        )
        logLikelihood += weight * float(np.sum(logDenom))
        finiteCount += int(x.size)
    return totalWeight, nullMass, slabMass, slabSecond, logLikelihood, finiteCount


def _emStep(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorNull: float,
    priorVariance: float,
    blockSize: int,
) -> tuple[float, float, float, float, float, int]:
    func = getattr(_cconsenrich, "cstateShrinkSpikeNormalEMStep", None)
    if func is not None:
        return tuple(
            func(
                state,
                variance,
                float(priorNull),
                float(priorVariance),
                int(blockSize),
            )
        )  # type: ignore[return-value]
    return _emStepPython(
        state,
        variance,
        priorNull=priorNull,
        priorVariance=priorVariance,
        blockSize=blockSize,
    )


def fitStateShrinkagePrior(
    chunks: Sequence[Any],
    *,
    model: str | None = None,
    priorNull: float | None = None,
    priorScale: float | None = None,
    maxIter: int = _STATE_SHRINKAGE_DEFAULT_MAX_ITER,
    tol: float = _STATE_SHRINKAGE_DEFAULT_TOL,
    nullZ: float = _STATE_SHRINKAGE_DEFAULT_NULL_Z,
    minNull: float = _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
    maxNull: float = _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
    blockSize: int | None = 1,
) -> stateShrinkPrior:
    r"""Fit a genome-level spike-and-normal prior with block-level weights.

    Each block contributes total weight one, split equally across,
    positive-variance intervals in that block. This keeps dense correlated
    intervals from dominating the EB hyperparameter fit while leaving posterior
    shrinkage itself available for every interval.
    """

    model_ = _normalizeModel(model)
    if model_ != STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL:
        supported = ", ".join(STATE_SHRINKAGE_MODELS)
        raise ValueError(
            f"Unsupported state shrinkage model {model!r}; supported values: {supported}."
        )
    chunks_ = list(chunks)
    if not chunks_:
        raise ValueError("state shrinkage prior fit requires at least one chunk")

    minNull_, maxNull_ = _nullBounds(minNull, maxNull)
    blockSize_ = _safeBlockSize(blockSize)
    totalWeight = 0.0
    centralWeight = 0.0
    excessMomentSum = 0.0
    varianceSum = 0.0
    stateSqSum = 0.0
    finiteCount = 0
    intervalCount = 0
    chunkCount = 0
    for item in chunks_:
        state, variance = _loadStateVarianceItem(item)
        chunkCount += 1
        intervalCount += int(state.size)
        sums = _initialSums(
            state,
            variance,
            nullZ=float(nullZ),
            blockSize=blockSize_,
        )
        totalWeight += float(sums[0])
        centralWeight += float(sums[1])
        excessMomentSum += float(sums[2])
        varianceSum += float(sums[3])
        stateSqSum += float(sums[4])
        finiteCount += int(sums[5])

    if totalWeight <= 0.0 or finiteCount <= 0:
        raise ValueError(
            "state shrinkage prior fit has no finite positive-variance intervals"
        )

    estimateNull = priorNull is None
    estimateScale = priorScale is None

    if priorNull is None:
        expectedCentral = float(
            math.erf(
                float(max(nullZ, _STATE_SHRINKAGE_POSITIVE_FLOOR)) / math.sqrt(2.0)
            )
        )
        if not np.isfinite(expectedCentral) or expectedCentral <= 0.0:
            pi0 = 0.8
        else:
            pi0 = (centralWeight / totalWeight) / expectedCentral
        pi0 = float(np.clip(pi0, minNull_, maxNull_))
    else:
        pi0 = float(priorNull)
        if not np.isfinite(pi0) or pi0 <= 0.0 or pi0 >= 1.0:
            raise ValueError("`priorNull` must be finite and strictly between 0 and 1")
        pi0 = float(np.clip(pi0, minNull_, maxNull_))

    if priorScale is None:
        tau2 = excessMomentSum / max(totalWeight, _STATE_SHRINKAGE_POSITIVE_FLOOR)
        if not np.isfinite(tau2) or tau2 <= _STATE_SHRINKAGE_POSITIVE_FLOOR:
            tau2 = varianceSum / max(totalWeight, _STATE_SHRINKAGE_POSITIVE_FLOOR)
        tau2 = float(max(tau2, _STATE_SHRINKAGE_POSITIVE_FLOOR))
    else:
        tau = float(priorScale)
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("`priorScale` must be finite and positive")
        tau2 = tau * tau

    converged = False
    iterations = 0
    logLikelihood = float("nan")
    tol_ = float(max(tol, 0.0))
    maxIter_ = int(max(maxIter, 1))
    if estimateNull or estimateScale:
        for iteration in range(maxIter_):
            iterations = iteration + 1
            emTotalWeight = 0.0
            nullMass = 0.0
            slabMass = 0.0
            slabSecond = 0.0
            logLikelihood = 0.0
            for item in chunks_:
                state, variance = _loadStateVarianceItem(item)
                sums = _emStep(
                    state,
                    variance,
                    priorNull=pi0,
                    priorVariance=tau2,
                    blockSize=blockSize_,
                )
                emTotalWeight += float(sums[0])
                nullMass += float(sums[1])
                slabMass += float(sums[2])
                slabSecond += float(sums[3])
                logLikelihood += float(sums[4])
            nextPi0 = pi0
            nextTau2 = tau2
            if estimateNull and emTotalWeight > 0.0:
                nextPi0 = float(np.clip(nullMass / emTotalWeight, minNull_, maxNull_))
            if estimateScale and slabMass > _STATE_SHRINKAGE_POSITIVE_FLOOR:
                nextTau2 = slabSecond / slabMass
                if (
                    not np.isfinite(nextTau2)
                    or nextTau2 <= _STATE_SHRINKAGE_POSITIVE_FLOOR
                ):
                    nextTau2 = tau2
            relPi = abs(nextPi0 - pi0) / max(abs(pi0), _STATE_SHRINKAGE_POSITIVE_FLOOR)
            relTau = abs(nextTau2 - tau2) / max(
                abs(tau2),
                _STATE_SHRINKAGE_POSITIVE_FLOOR,
            )
            pi0, tau2 = nextPi0, max(nextTau2, _STATE_SHRINKAGE_POSITIVE_FLOOR)
            if max(relPi, relTau) <= tol_:
                converged = True
                break
    else:
        converged = True
        iterations = 0

    metadata = {
        "model": STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,
        "scope": "genome",
        "chunk_count": int(chunkCount),
        "interval_count": int(intervalCount),
        "finite_count": int(finiteCount),
        "effective_block_count": _metadataFloat(totalWeight),
        "block_size_intervals": int(blockSize_),
        "prior_null": _metadataFloat(pi0),
        "prior_scale": _metadataFloat(math.sqrt(tau2)),
        "prior_variance": _metadataFloat(tau2),
        "estimated_prior_null": bool(estimateNull),
        "estimated_prior_scale": bool(estimateScale),
        "iterations": int(iterations),
        "converged": bool(converged),
        "log_likelihood": _metadataFloat(logLikelihood),
    }
    return stateShrinkPrior(
        model=STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,
        priorNull=float(pi0),
        priorScale=float(math.sqrt(tau2)),
        priorVariance=float(tau2),
        blockSize=int(blockSize_),
        metadata=metadata,
    )


def _safeShrinkageFactor(state: np.ndarray, shrunkState: np.ndarray) -> np.ndarray:
    return np.divide(
        shrunkState,
        state,
        out=np.zeros_like(shrunkState, dtype=np.float64),
        where=np.abs(state) > _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )


def _posteriorPython(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorNull: float,
    priorVariance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pi0 = float(
        np.clip(
            priorNull,
            _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
            _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
        )
    )
    tau2 = float(max(priorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR))
    valid = np.isfinite(state) & np.isfinite(variance) & (variance > 0.0)
    varianceSafe = np.maximum(variance[valid], _STATE_SHRINKAGE_POSITIVE_FLOOR)
    stateValid = state[valid]

    logNull = math.log(pi0) + _logNormalZeroDensity(stateValid, varianceSafe)
    logSlab = math.log1p(-pi0) + _logNormalZeroDensity(stateValid, varianceSafe + tau2)
    logDenom = np.logaddexp(logNull, logSlab)
    nullProb = np.exp(logNull - logDenom)
    slabWeight = np.maximum(1.0 - nullProb, 0.0)
    slabShrinkage = tau2 / (tau2 + varianceSafe)
    slabMean = slabShrinkage * stateValid
    slabVariance = slabShrinkage * varianceSafe
    shrunkValid = slabWeight * slabMean
    posteriorSecond = slabWeight * (slabVariance + slabMean * slabMean)
    posteriorVariance = np.maximum(
        posteriorSecond - shrunkValid * shrunkValid,
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )
    posteriorSdValid = np.sqrt(posteriorVariance)
    shrinkFactorValid = _safeShrinkageFactor(stateValid, shrunkValid)

    shrunk = state.astype(np.float64, copy=True)
    posteriorSd = np.full_like(state, np.nan, dtype=np.float64)
    shrinkFactor = np.ones_like(state, dtype=np.float64)
    nullOut = np.full_like(state, np.nan, dtype=np.float64)
    slabMeanOut = np.full_like(state, np.nan, dtype=np.float64)
    slabWeightOut = np.full_like(state, np.nan, dtype=np.float64)
    shrunk[valid] = shrunkValid
    posteriorSd[valid] = posteriorSdValid
    shrinkFactor[valid] = shrinkFactorValid
    nullOut[valid] = nullProb
    slabMeanOut[valid] = slabMean
    slabWeightOut[valid] = slabWeight
    return shrunk, posteriorSd, shrinkFactor, nullOut, slabMeanOut, slabWeightOut


def _posterior(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorNull: float,
    priorVariance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    func = getattr(_cconsenrich, "cstateShrinkSpikeNormalPosterior", None)
    if func is not None:
        return tuple(
            func(
                state,
                variance,
                float(priorNull),
                float(priorVariance),
            )
        )  # type: ignore[return-value]
    return _posteriorPython(
        state,
        variance,
        priorNull=priorNull,
        priorVariance=priorVariance,
    )


def applyStateShrinkagePrior(
    state: npt.ArrayLike,
    stateVariance: npt.ArrayLike,
    prior: stateShrinkPrior,
) -> stateShrinkResult:
    r"""Apply a fitted spike-and-normal prior to one state vector."""

    stateArr = _asFloatVector("state", state)
    varianceArr = _asFloatVector("stateVariance", stateVariance)
    if stateArr.shape != varianceArr.shape:
        raise ValueError("`state` and `stateVariance` must have the same length")
    (
        shrunk,
        posteriorSd,
        shrinkFactor,
        nullProb,
        slabMean,
        slabWeight,
    ) = _posterior(
        stateArr,
        varianceArr,
        priorNull=prior.priorNull,
        priorVariance=prior.priorVariance,
    )
    valid = np.isfinite(stateArr) & np.isfinite(varianceArr) & (varianceArr > 0.0)
    metadata = {
        **dict(prior.metadata),
        "scope": "contig_apply",
        "interval_count": int(stateArr.size),
        "finite_count": int(np.count_nonzero(valid)),
        "state_abs_median_before": _metadataFloat(
            np.median(np.abs(stateArr[valid])) if np.any(valid) else float("nan")
        ),
        "state_abs_median_after": _metadataFloat(
            np.median(np.abs(shrunk[valid])) if np.any(valid) else float("nan")
        ),
        "shrinkage_factor_median": _metadataFloat(
            np.median(shrinkFactor[valid]) if np.any(valid) else float("nan")
        ),
        "null_probability_median": _metadataFloat(
            np.median(nullProb[valid]) if np.any(valid) else float("nan")
        ),
        "posterior_sd_median": _metadataFloat(
            np.median(posteriorSd[valid]) if np.any(valid) else float("nan")
        ),
    }
    return stateShrinkResult(
        shrunkState=np.asarray(shrunk, dtype=np.float32),
        posteriorSd=np.asarray(posteriorSd, dtype=np.float32),
        shrinkageFactor=np.asarray(shrinkFactor, dtype=np.float32),
        nullProbability=np.asarray(nullProb, dtype=np.float32),
        slabPosteriorMean=np.asarray(slabMean, dtype=np.float32),
        slabPosteriorWeight=np.asarray(slabWeight, dtype=np.float32),
        priorNull=float(prior.priorNull),
        priorScale=float(prior.priorScale),
        metadata=metadata,
    )


def shrinkStateEB(
    state: npt.ArrayLike,
    stateVariance: npt.ArrayLike,
    *,
    model: str | None = None,
    priorNull: float | None = None,
    priorScale: float | None = None,
    maxIter: int = _STATE_SHRINKAGE_DEFAULT_MAX_ITER,
    tol: float = _STATE_SHRINKAGE_DEFAULT_TOL,
    nullZ: float = _STATE_SHRINKAGE_DEFAULT_NULL_Z,
    minNull: float = _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
    maxNull: float = _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
    blockSize: int | None = 1,
    **_unused: Any,
) -> stateShrinkResult:
    r"""Shrink fitted state estimates toward zero using post-fit EB.

    This convenience API fits the spike-and-normal prior on the supplied vector
    and immediately applies it. Genome-level callers should prefer
    :func:`fitStateShrinkagePrior` followed by :func:`applyStateShrinkagePrior`.
    """

    stateArr = _asFloatVector("state", state)
    varianceArr = _asFloatVector("stateVariance", stateVariance)
    if stateArr.shape != varianceArr.shape:
        raise ValueError("`state` and `stateVariance` must have the same length")
    prior = fitStateShrinkagePrior(
        [(stateArr, varianceArr)],
        model=model,
        priorNull=priorNull,
        priorScale=priorScale,
        maxIter=maxIter,
        tol=tol,
        nullZ=nullZ,
        minNull=minNull,
        maxNull=maxNull,
        blockSize=blockSize,
    )
    return applyStateShrinkagePrior(stateArr, varianceArr, prior)
