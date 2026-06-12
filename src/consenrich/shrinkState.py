"""Post-fit empirical-Bayes shrinkage for Consenrich state estimates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from .cconsenrich import (
    cstateShrinkInitialSums as _cstateShrinkInitialSums,
    cstateShrinkMixtureEMStep as _cstateShrinkMixtureEMStep,
    cstateShrinkMixtureEMStepPrepared as _cstateShrinkMixtureEMStepPrepared,
    cstateShrinkMixturePosterior as _cstateShrinkMixturePosterior,
    cstateShrinkMixturePosteriorPrepared as _cstateShrinkMixturePosteriorPrepared,
)


STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE = "adaptiveNormalMixture"
STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL = "spikeAndNormal"
STATE_SHRINKAGE_MODEL = STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE
STATE_SHRINKAGE_MODELS = (
    STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE,
    STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,
)
_STATE_SHRINKAGE_POSITIVE_FLOOR = 1.0e-12
_STATE_SHRINKAGE_WEIGHT_FLOOR = 1.0e-12
_STATE_SHRINKAGE_DEFAULT_MAX_ITER = 96
_STATE_SHRINKAGE_DEFAULT_TOL = 1.0e-5
_STATE_SHRINKAGE_DEFAULT_NULL_Z = 1.5
_STATE_SHRINKAGE_DEFAULT_MIN_NULL = 1.0e-2
_STATE_SHRINKAGE_DEFAULT_MAX_NULL = 1.0 - 1.0e-4
_STATE_SHRINKAGE_DEFAULT_SLAB_SCALE_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


class stateShrinkPrior(NamedTuple):
    r"""Specifies genome-wide EB prior for state shrinkage (spike+normal, mixture)."""

    model: str
    priorNull: float
    priorScale: float
    priorVariance: float
    blockSize: int
    metadata: dict[str, Any]
    slabVariance: tuple[float, ...] = ()
    slabWeight: tuple[float, ...] = ()
    componentWeights: tuple[float, ...] = ()


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
    model_ = str(model)
    if model_ in STATE_SHRINKAGE_MODELS:
        return model_
    supported = ", ".join(STATE_SHRINKAGE_MODELS)
    raise ValueError(
        f"Unsupported state shrinkage model {model!r}; supported values: {supported}."
    )


def _asFloatVector(name: str, values: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"`{name}` must be non-empty")
    return arr


def _loadStateVarianceItem(item: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(item, Mapping):
        stateSource = item.get("state")
        varianceSource = item.get("variance")
        if stateSource is None or varianceSource is None:
            raise ValueError("state shrinkage item is missing `state`/`variance`")
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


def _sortedPositiveWeights(
    slabVariance: npt.ArrayLike,
    slabWeight: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    variance = np.asarray(slabVariance, dtype=np.float64).reshape(-1)
    weight = np.asarray(slabWeight, dtype=np.float64).reshape(-1)
    if variance.ndim != 1 or weight.ndim != 1 or variance.size == 0:
        raise ValueError("state shrinkage slabs must be non-empty one-dimensional arrays")
    if variance.shape != weight.shape:
        raise ValueError("state shrinkage slab variance and weight shapes differ")
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("state shrinkage slab variances must be finite and positive")
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValueError("state shrinkage slab weights must be finite and positive")
    order = np.argsort(variance, kind="mergesort")
    variance = np.maximum(variance[order], _STATE_SHRINKAGE_POSITIVE_FLOOR)
    weight = np.maximum(weight[order], _STATE_SHRINKAGE_WEIGHT_FLOOR)
    weight = weight / float(np.sum(weight))
    return variance, weight


def _priorSlabArrays(prior: stateShrinkPrior) -> tuple[np.ndarray, np.ndarray]:
    if len(prior.slabVariance) and len(prior.slabWeight):
        return _sortedPositiveWeights(prior.slabVariance, prior.slabWeight)
    return _sortedPositiveWeights((prior.priorVariance,), (1.0,))


def _componentWeights(priorNull: float, slabWeight: np.ndarray) -> tuple[float, ...]:
    pi0 = float(priorNull)
    return tuple([pi0] + [float((1.0 - pi0) * weight) for weight in slabWeight])


def _slabMultipliersForModel(model: str) -> tuple[float, ...]:
    if model == STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL:
        return (1.0,)
    return _STATE_SHRINKAGE_DEFAULT_SLAB_SCALE_MULTIPLIERS


def _logSlabPrior(priorNull: float, slabWeight: np.ndarray) -> np.ndarray:
    pi0 = float(priorNull)
    if not np.isfinite(pi0) or pi0 <= 0.0 or pi0 >= 1.0:
        raise ValueError("`priorNull` must be finite and strictly between 0 and 1")
    weight = np.asarray(slabWeight, dtype=np.float64).reshape(-1)
    weightTotal = float(np.sum(weight))
    if weight.size == 0 or weightTotal <= 0.0 or not np.isfinite(weightTotal):
        raise ValueError("state shrinkage slab weights must have positive total weight")
    return np.log(weight) + (math.log(1.0 - pi0) - math.log(weightTotal))


def _initialSums(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    nullZ: float,
    blockSize: int,
) -> tuple[float, float, float, float, int]:
    return tuple(
        _cstateShrinkInitialSums(state, variance, float(nullZ), int(blockSize))
    )  # type: ignore[return-value]


def _emStep(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorNull: float,
    slabVariance: np.ndarray,
    logSlabPrior: np.ndarray,
    blockSize: int,
) -> tuple[float, float, np.ndarray, np.ndarray, float, int]:
    out = _cstateShrinkMixtureEMStepPrepared(
        state,
        variance,
        float(priorNull),
        slabVariance,
        logSlabPrior,
        int(blockSize),
    )
    return (
        float(out[0]),
        float(out[1]),
        np.asarray(out[2], dtype=np.float64),
        np.asarray(out[3], dtype=np.float64),
        float(out[4]),
        int(out[5]),
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
    chunkItems = list(chunks)
    if not chunkItems:
        raise ValueError("state shrinkage prior fit requires at least one chunk")
    chunks_ = [_loadStateVarianceItem(item) for item in chunkItems]

    minNull_, maxNull_ = _nullBounds(minNull, maxNull)
    blockSize_ = _safeBlockSize(blockSize)
    totalWeight = 0.0
    centralWeight = 0.0
    excessMomentSum = 0.0
    varianceSum = 0.0
    finiteCount = 0
    intervalCount = 0
    chunkCount = 0
    for state, variance in chunks_:
        chunkCount += 1
        intervalCount += int(state.size)
        (
            chunkTotalWeight,
            chunkCentralWeight,
            chunkExcessMomentSum,
            chunkVarianceSum,
            chunkFiniteCount,
        ) = _initialSums(
            state,
            variance,
            nullZ=float(nullZ),
            blockSize=blockSize_,
        )
        totalWeight += float(chunkTotalWeight)
        centralWeight += float(chunkCentralWeight)
        excessMomentSum += float(chunkExcessMomentSum)
        varianceSum += float(chunkVarianceSum)
        finiteCount += int(chunkFiniteCount)

    if totalWeight <= 0.0 or finiteCount <= 0:
        raise ValueError(
            "state shrinkage prior fit has no finite positive-variance intervals"
        )

    estimateNull = priorNull is None

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
        momentVariance = excessMomentSum / max(
            totalWeight,
            _STATE_SHRINKAGE_POSITIVE_FLOOR,
        )
        if (
            not np.isfinite(momentVariance)
            or momentVariance <= _STATE_SHRINKAGE_POSITIVE_FLOOR
        ):
            momentVariance = varianceSum / max(
                totalWeight,
                _STATE_SHRINKAGE_POSITIVE_FLOOR,
            )
        if (
            not np.isfinite(momentVariance)
            or momentVariance <= _STATE_SHRINKAGE_POSITIVE_FLOOR
        ):
            raise ValueError("state shrinkage prior fit has no positive moment scale")
        baseScale = float(math.sqrt(momentVariance))
        estimateSlabScales = True
    else:
        baseScale = float(priorScale)
        if not np.isfinite(baseScale) or baseScale <= 0.0:
            raise ValueError("`priorScale` must be finite and positive")
        estimateSlabScales = False

    slabMultipliers = np.asarray(
        _slabMultipliersForModel(model_),
        dtype=np.float64,
    )
    slabVariance = np.maximum(
        np.square(float(baseScale) * slabMultipliers),
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )
    slabWeight = np.full(
        slabVariance.shape,
        1.0 / float(slabVariance.size),
        dtype=np.float64,
    )
    slabVariance, slabWeight = _sortedPositiveWeights(slabVariance, slabWeight)
    estimateSlabWeights = slabVariance.size > 1

    converged = False
    iterations = 0
    logLikelihood = float("nan")
    tol_ = float(max(tol, 0.0))
    maxIter_ = int(max(maxIter, 1))
    if estimateNull or estimateSlabScales or estimateSlabWeights:
        for iteration in range(maxIter_):
            iterations = iteration + 1
            emTotalWeight = 0.0
            nullMass = 0.0
            slabMass = np.zeros_like(slabWeight, dtype=np.float64)
            slabSecond = np.zeros_like(slabVariance, dtype=np.float64)
            logLikelihood = 0.0
            logSlabPrior = _logSlabPrior(pi0, slabWeight)
            for state, variance in chunks_:
                sums = _emStep(
                    state,
                    variance,
                    priorNull=pi0,
                    slabVariance=slabVariance,
                    logSlabPrior=logSlabPrior,
                    blockSize=blockSize_,
                )
                emTotalWeight += float(sums[0])
                nullMass += float(sums[1])
                slabMass += np.asarray(sums[2], dtype=np.float64)
                slabSecond += np.asarray(sums[3], dtype=np.float64)
                logLikelihood += float(sums[4])
            nextPi0 = pi0
            nextSlabWeight = slabWeight.copy()
            nextSlabVariance = slabVariance.copy()
            if estimateNull and emTotalWeight > 0.0:
                nextPi0 = float(np.clip(nullMass / emTotalWeight, minNull_, maxNull_))
            if estimateSlabWeights:
                massTotal = float(np.sum(slabMass))
                if massTotal > _STATE_SHRINKAGE_POSITIVE_FLOOR:
                    nextSlabWeight = np.maximum(
                        slabMass / massTotal,
                        _STATE_SHRINKAGE_WEIGHT_FLOOR,
                    )
                    nextSlabWeight = nextSlabWeight / float(np.sum(nextSlabWeight))
            if estimateSlabScales:
                active = slabMass > _STATE_SHRINKAGE_POSITIVE_FLOOR
                nextSlabVariance[active] = slabSecond[active] / slabMass[active]
                nextSlabVariance = np.maximum(
                    nextSlabVariance,
                    _STATE_SHRINKAGE_POSITIVE_FLOOR,
                )
                nextSlabVariance[~np.isfinite(nextSlabVariance)] = slabVariance[
                    ~np.isfinite(nextSlabVariance)
                ]
            nextSlabVariance, nextSlabWeight = _sortedPositiveWeights(
                nextSlabVariance,
                nextSlabWeight,
            )
            relPi = abs(nextPi0 - pi0) / max(abs(pi0), _STATE_SHRINKAGE_POSITIVE_FLOOR)
            relWeight = float(
                np.max(
                    np.abs(nextSlabWeight - slabWeight)
                    / np.maximum(np.abs(slabWeight), _STATE_SHRINKAGE_POSITIVE_FLOOR)
                )
            )
            relVariance = float(
                np.max(
                    np.abs(nextSlabVariance - slabVariance)
                    / np.maximum(
                        np.abs(slabVariance),
                        _STATE_SHRINKAGE_POSITIVE_FLOOR,
                    )
                )
            )
            pi0 = nextPi0
            slabVariance = nextSlabVariance
            slabWeight = nextSlabWeight
            if max(relPi, relWeight, relVariance) <= tol_:
                converged = True
                break
    else:
        converged = True
        iterations = 0

    priorVariance = float(np.sum(slabWeight * slabVariance))
    priorScaleOut = float(math.sqrt(max(priorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR)))
    componentWeights = _componentWeights(pi0, slabWeight)
    metadata = {
        "model": model_,
        "scope": "genome",
        "chunk_count": int(chunkCount),
        "interval_count": int(intervalCount),
        "finite_count": int(finiteCount),
        "effective_block_count": _metadataFloat(totalWeight),
        "block_size_intervals": int(blockSize_),
        "prior_null": _metadataFloat(pi0),
        "prior_scale": _metadataFloat(priorScaleOut),
        "prior_variance": _metadataFloat(priorVariance),
        "slab_count": int(slabVariance.size),
        "slab_variance": [float(value) for value in slabVariance],
        "slab_weight": [float(value) for value in slabWeight],
        "component_weights": [float(value) for value in componentWeights],
        "estimated_prior_null": bool(estimateNull),
        "estimated_prior_scale": bool(estimateSlabScales),
        "estimated_slab_weights": bool(estimateSlabWeights),
        "estimated_slab_scales": bool(estimateSlabScales),
        "iterations": int(iterations),
        "converged": bool(converged),
        "log_likelihood": _metadataFloat(logLikelihood),
    }
    return stateShrinkPrior(
        model=model_,
        priorNull=float(pi0),
        priorScale=priorScaleOut,
        priorVariance=priorVariance,
        blockSize=int(blockSize_),
        metadata=metadata,
        slabVariance=tuple(float(value) for value in slabVariance),
        slabWeight=tuple(float(value) for value in slabWeight),
        componentWeights=componentWeights,
    )


def _posterior(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorNull: float,
    slabVariance: np.ndarray,
    slabWeight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        _cstateShrinkMixturePosteriorPrepared(
            state,
            variance,
            float(priorNull),
            slabVariance,
            _logSlabPrior(priorNull, slabWeight),
        )
    )  # type: ignore[return-value]


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
    priorSlabVariance, priorSlabWeight = _priorSlabArrays(prior)
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
        slabVariance=priorSlabVariance,
        slabWeight=priorSlabWeight,
    )
    valid = np.isfinite(stateArr) & np.isfinite(varianceArr) & (varianceArr > 0.0)
    metadata = {
        **dict(prior.metadata),
        "scope": "contig_apply",
        "interval_count": int(stateArr.size),
        "finite_count": int(np.count_nonzero(valid)),
        "slab_count": int(priorSlabVariance.size),
        "slab_variance": [float(value) for value in priorSlabVariance],
        "slab_weight": [float(value) for value in priorSlabWeight],
        "component_weights": [
            float(value) for value in _componentWeights(prior.priorNull, priorSlabWeight)
        ],
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
