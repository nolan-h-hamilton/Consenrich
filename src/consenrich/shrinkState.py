"""Post-fit empirical-Bayes shrinkage for Consenrich state estimates."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt


STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE = "adaptiveNormalMixture"
STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL = "spikeAndNormal"
STATE_SHRINKAGE_MODEL = STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE
STATE_SHRINKAGE_MODELS = (
    STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE,
    STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,
)
_STATE_SHRINKAGE_POSITIVE_FLOOR = 1.0e-12
_STATE_SHRINKAGE_WEIGHT_FLOOR = 1.0e-12
_STATE_SHRINKAGE_DEFAULT_MAX_ITER = 64
_STATE_SHRINKAGE_DEFAULT_TOL = 1.0e-5
_STATE_SHRINKAGE_DEFAULT_NULL_Z = 1.0
_STATE_SHRINKAGE_DEFAULT_MIN_NULL = 1.0e-4
_STATE_SHRINKAGE_DEFAULT_MAX_NULL = 1.0 - 1.0e-4
_STATE_SHRINKAGE_DEFAULT_MIXTURE_COMPONENTS = 16
_STATE_SHRINKAGE_DEFAULT_CHUNK_SIZE = 131_072


class stateShrinkResult(NamedTuple):
    r"""Post-fit empirical-Bayes shrinkage result.

    ``shrunkState`` is the posterior mean under the fitted prior and
    ``shrinkageFactor`` is ``shrunkState / state`` where defined.
    ``nullProbability`` is the posterior mass assigned to the exact-zero
    component. For ``adaptiveNormalMixture``, the remaining prior mass is spread
    over a grid of zero-centered Normal components; for ``spikeAndNormal`` the
    remaining prior mass is a single Normal slab.
    """

    shrunkState: np.ndarray
    shrinkageFactor: np.ndarray
    nullProbability: np.ndarray
    slabPosteriorMean: np.ndarray
    slabPosteriorWeight: np.ndarray
    priorNull: float
    priorScale: float
    metadata: dict[str, Any]


def _asFloatVector(name: str, values: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"`{name}` must be non-empty")
    return arr


def _metadataFloat(value: float) -> float | None:
    value_ = float(value)
    return value_ if np.isfinite(value_) else None


def _iterChunks(n: int, chunkSize: int = _STATE_SHRINKAGE_DEFAULT_CHUNK_SIZE):
    chunkSize_ = int(max(1, chunkSize))
    for start in range(0, int(n), chunkSize_):
        yield start, min(start + chunkSize_, int(n))


def _normalizeModel(model: str | None) -> str:
    if model is None:
        return STATE_SHRINKAGE_MODEL
    key = str(model).strip().replace("-", "_").replace(" ", "_").lower()
    keyCompact = key.replace("_", "")
    if keyCompact in {
        "adaptivenormalmixture",
        "normalmixture",
        "mixturenormal",
        "ashr",
        "ashrlike",
        "adaptive",
    }:
        return STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE
    if keyCompact in {"spikeandnormal", "spikenormal", "spike", "pointnormal"}:
        return STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL
    supported = ", ".join(STATE_SHRINKAGE_MODELS)
    raise ValueError(
        f"Unsupported state shrinkage model {model!r}; supported values: {supported}."
    )


def _logNormalZeroDensity(x: np.ndarray, variance: np.ndarray) -> np.ndarray:
    var = np.maximum(
        np.asarray(variance, dtype=np.float64),
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )
    return -0.5 * (
        np.log(2.0 * math.pi * var)
        + (np.asarray(x, dtype=np.float64) ** 2) / var
    )


def _initialNullProbability(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    nullZ: float,
    minNull: float,
    maxNull: float,
) -> float:
    z = np.abs(state) / np.sqrt(
        np.maximum(variance, _STATE_SHRINKAGE_POSITIVE_FLOOR)
    )
    central = float(
        np.mean(z <= float(max(nullZ, _STATE_SHRINKAGE_POSITIVE_FLOOR)))
    )
    expectedCentral = float(
        math.erf(
            float(max(nullZ, _STATE_SHRINKAGE_POSITIVE_FLOOR)) / math.sqrt(2.0)
        )
    )
    if (
        not np.isfinite(central)
        or not np.isfinite(expectedCentral)
        or expectedCentral <= 0.0
    ):
        return float(np.clip(0.8, minNull, maxNull))
    return float(np.clip(central / expectedCentral, minNull, maxNull))


def _initialSlabVariance(state: np.ndarray, variance: np.ndarray) -> float:
    rawMoment = float(np.mean(state * state - variance))
    robustScale = float(np.median(np.abs(state)) / 0.6744897501960817)
    robustMoment = robustScale * robustScale - float(np.median(variance))
    candidates = [
        rawMoment,
        robustMoment,
        float(np.var(state, ddof=1)) if state.size > 1 else rawMoment,
    ]
    finitePositive = [
        value
        for value in candidates
        if np.isfinite(value) and value > _STATE_SHRINKAGE_POSITIVE_FLOOR
    ]
    if finitePositive:
        return float(max(finitePositive))
    fallback = float(max(np.nanmedian(variance), _STATE_SHRINKAGE_POSITIVE_FLOOR))
    return fallback


def _safeShrinkageFactor(state: np.ndarray, shrunkState: np.ndarray) -> np.ndarray:
    return np.divide(
        shrunkState,
        state,
        out=np.zeros_like(shrunkState, dtype=np.float64),
        where=np.abs(state) > _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )


def _posteriorMomentsSpikeAndNormal(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorNull: float,
    priorVariance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pi0 = float(
        np.clip(
            priorNull,
            _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
            _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
        )
    )
    tau2 = float(max(priorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR))
    variance = np.maximum(variance, _STATE_SHRINKAGE_POSITIVE_FLOOR)

    logNull = math.log(pi0) + _logNormalZeroDensity(state, variance)
    logSlab = math.log1p(-pi0) + _logNormalZeroDensity(state, variance + tau2)
    logDenom = np.logaddexp(logNull, logSlab)
    nullProb = np.exp(logNull - logDenom)
    slabWeight = np.maximum(1.0 - nullProb, 0.0)
    slabShrinkage = tau2 / (tau2 + variance)
    slabMean = slabShrinkage * state
    slabVariance = slabShrinkage * variance
    shrunkState = slabWeight * slabMean
    shrinkageFactor = _safeShrinkageFactor(state, shrunkState)
    return nullProb, slabWeight, slabMean, slabVariance, shrinkageFactor


def _sanitizeValidStateVariance(
    state: npt.ArrayLike,
    stateVariance: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stateArr = _asFloatVector("state", state)
    varianceArr = _asFloatVector("stateVariance", stateVariance)
    if stateArr.shape != varianceArr.shape:
        raise ValueError("`state` and `stateVariance` must have the same length")

    valid = np.isfinite(stateArr) & np.isfinite(varianceArr) & (varianceArr > 0.0)
    if not np.any(valid):
        raise ValueError("state shrinkage has no finite positive-variance intervals")

    stateValid = stateArr[valid]
    varianceValid = np.maximum(
        varianceArr[valid],
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )
    return stateArr, varianceArr, valid, stateValid, varianceValid


def _nullBounds(minNull: float, maxNull: float) -> tuple[float, float]:
    minNull_ = float(np.clip(minNull, _STATE_SHRINKAGE_DEFAULT_MIN_NULL, 0.5))
    maxNull_ = float(np.clip(maxNull, 0.5, _STATE_SHRINKAGE_DEFAULT_MAX_NULL))
    if minNull_ >= maxNull_:
        raise ValueError("`minNull` must be smaller than `maxNull`")
    return minNull_, maxNull_


def _buildAdaptiveNormalMixtureVariances(
    state: np.ndarray,
    variance: np.ndarray,
    *,
    priorScale: float | None,
    componentCount: int,
) -> tuple[np.ndarray, np.ndarray]:
    componentCount_ = int(max(2, componentCount))
    if priorScale is None:
        momentScale = math.sqrt(_initialSlabVariance(state, variance))
        robustScale = float(np.median(np.abs(state)) / 0.6744897501960817)
        observedScale = float(np.std(state, ddof=1)) if state.size > 1 else momentScale
        seScale = math.sqrt(float(np.median(variance)))
        scaleCandidates = [momentScale, robustScale, observedScale, seScale]
        maxScale = max(
            [
                value
                for value in scaleCandidates
                if np.isfinite(value) and value > _STATE_SHRINKAGE_POSITIVE_FLOOR
            ]
            or [math.sqrt(_STATE_SHRINKAGE_POSITIVE_FLOOR)]
        )
    else:
        maxScale = float(priorScale)
        if not np.isfinite(maxScale) or maxScale <= 0.0:
            raise ValueError("`priorScale` must be finite and positive")

    maxScale = float(max(maxScale, math.sqrt(_STATE_SHRINKAGE_POSITIVE_FLOOR)))
    minScale = float(max(maxScale * 1.0e-3, math.sqrt(_STATE_SHRINKAGE_POSITIVE_FLOOR)))
    if minScale >= maxScale:
        minScale = maxScale / 2.0
    if minScale <= 0.0 or not np.isfinite(minScale):
        minScale = math.sqrt(_STATE_SHRINKAGE_POSITIVE_FLOOR)
    if maxScale <= minScale:
        maxScale = minScale * 2.0

    positiveScales = np.geomspace(minScale, maxScale, componentCount_)
    componentVariances = np.concatenate(
        [np.zeros(1, dtype=np.float64), positiveScales * positiveScales]
    )
    componentVariances = np.maximum(componentVariances, 0.0)
    componentVariances[0] = 0.0
    return componentVariances, positiveScales


def _normalMixtureLogJoint(
    state: np.ndarray,
    variance: np.ndarray,
    componentVariances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    totalVariance = (
        np.maximum(variance, _STATE_SHRINKAGE_POSITIVE_FLOOR)[:, None]
        + componentVariances[None, :]
    )
    logWeights = np.log(np.maximum(weights, _STATE_SHRINKAGE_WEIGHT_FLOOR))
    return logWeights[None, :] - 0.5 * (
        np.log(2.0 * math.pi * totalVariance)
        + (state[:, None] * state[:, None]) / totalVariance
    )


def _fitAdaptiveNormalMixtureWeights(
    state: np.ndarray,
    variance: np.ndarray,
    componentVariances: np.ndarray,
    *,
    initialNull: float,
    fixedNull: bool,
    maxIter: int,
    tol: float,
) -> tuple[np.ndarray, int, bool, float]:
    componentCount = int(componentVariances.size)
    if componentCount < 2:
        raise ValueError("adaptive Normal mixture must contain at least two components")
    weights = np.empty(componentCount, dtype=np.float64)
    weights[0] = float(np.clip(initialNull, 0.0, 1.0))
    weights[1:] = (1.0 - weights[0]) / float(componentCount - 1)
    weights = np.maximum(weights, _STATE_SHRINKAGE_WEIGHT_FLOOR)
    weights /= float(np.sum(weights))
    if fixedNull:
        weights[0] = float(np.clip(initialNull, 0.0, 1.0))
        positiveMass = max(1.0 - weights[0], _STATE_SHRINKAGE_WEIGHT_FLOOR)
        weights[1:] = positiveMass * weights[1:] / float(np.sum(weights[1:]))

    maxIter_ = int(max(maxIter, 1))
    tol_ = float(max(tol, 0.0))
    converged = False
    iterations = 0
    logLikelihood = float("nan")
    n = int(state.size)
    for iteration in range(maxIter_):
        iterations = iteration + 1
        componentSums = np.zeros(componentCount, dtype=np.float64)
        logLikelihood = 0.0
        for start, end in _iterChunks(n):
            logJoint = _normalMixtureLogJoint(
                state[start:end],
                variance[start:end],
                componentVariances,
                weights,
            )
            logDenom = np.logaddexp.reduce(logJoint, axis=1)
            responsibility = np.exp(logJoint - logDenom[:, None])
            componentSums += np.sum(responsibility, axis=0)
            logLikelihood += float(np.sum(logDenom))

        nextWeights = componentSums / max(float(n), 1.0)
        nextWeights = np.maximum(nextWeights, _STATE_SHRINKAGE_WEIGHT_FLOOR)
        if fixedNull:
            nullWeight = float(weights[0])
            positiveSums = np.maximum(
                componentSums[1:],
                _STATE_SHRINKAGE_WEIGHT_FLOOR,
            )
            nextWeights[0] = nullWeight
            nextWeights[1:] = (1.0 - nullWeight) * positiveSums / float(
                np.sum(positiveSums)
            )
        else:
            nextWeights /= float(np.sum(nextWeights))

        delta = float(np.max(np.abs(nextWeights - weights)))
        weights = nextWeights
        if delta <= tol_:
            converged = True
            break

    return weights, iterations, converged, logLikelihood


def _posteriorMomentsAdaptiveNormalMixture(
    state: np.ndarray,
    variance: np.ndarray,
    componentVariances: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(state.size)
    nullProb = np.full(n, np.nan, dtype=np.float64)
    slabWeight = np.full(n, np.nan, dtype=np.float64)
    slabMean = np.full(n, np.nan, dtype=np.float64)
    shrunkState = np.full(n, np.nan, dtype=np.float64)
    shrinkageFactor = np.full(n, np.nan, dtype=np.float64)

    for start, end in _iterChunks(n):
        stateChunk = state[start:end]
        varianceChunk = np.maximum(
            variance[start:end],
            _STATE_SHRINKAGE_POSITIVE_FLOOR,
        )
        logJoint = _normalMixtureLogJoint(
            stateChunk,
            varianceChunk,
            componentVariances,
            weights,
        )
        logDenom = np.logaddexp.reduce(logJoint, axis=1)
        responsibility = np.exp(logJoint - logDenom[:, None])
        shrinkageByComponent = componentVariances[None, :] / (
            componentVariances[None, :] + varianceChunk[:, None]
        )
        componentMean = shrinkageByComponent * stateChunk[:, None]
        shrunkChunk = np.sum(responsibility * componentMean, axis=1)
        nullChunk = responsibility[:, 0]
        positiveWeight = np.maximum(1.0 - nullChunk, 0.0)
        positiveNumerator = np.sum(
            responsibility[:, 1:] * componentMean[:, 1:],
            axis=1,
        )
        positiveMean = np.divide(
            positiveNumerator,
            positiveWeight,
            out=np.zeros_like(positiveNumerator, dtype=np.float64),
            where=positiveWeight > _STATE_SHRINKAGE_POSITIVE_FLOOR,
        )
        nullProb[start:end] = nullChunk
        slabWeight[start:end] = positiveWeight
        slabMean[start:end] = positiveMean
        shrunkState[start:end] = shrunkChunk
        shrinkageFactor[start:end] = _safeShrinkageFactor(stateChunk, shrunkChunk)

    return nullProb, slabWeight, slabMean, shrunkState, shrinkageFactor


def _mixturePriorScale(
    componentVariances: np.ndarray,
    weights: np.ndarray,
) -> float:
    positiveMass = float(np.sum(weights[1:]))
    if positiveMass <= _STATE_SHRINKAGE_WEIGHT_FLOOR:
        return 0.0
    positiveVariance = float(np.sum(weights[1:] * componentVariances[1:]) / positiveMass)
    return math.sqrt(max(positiveVariance, 0.0))


def _shrinkStateAdaptiveNormalMixture(
    state: npt.ArrayLike,
    stateVariance: npt.ArrayLike,
    *,
    priorNull: float | None = None,
    priorScale: float | None = None,
    maxIter: int = _STATE_SHRINKAGE_DEFAULT_MAX_ITER,
    tol: float = _STATE_SHRINKAGE_DEFAULT_TOL,
    nullZ: float = _STATE_SHRINKAGE_DEFAULT_NULL_Z,
    minNull: float = _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
    maxNull: float = _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
    mixtureComponentCount: int = _STATE_SHRINKAGE_DEFAULT_MIXTURE_COMPONENTS,
) -> stateShrinkResult:
    r"""Shrink state estimates using an adaptive zero-centered Normal mixture.

    The working likelihood is ``state[i] | ell[i] ~ N(ell[i], stateVariance[i])``.
    The prior is ``sum_k pi[k] N(0, tau[k]^2)`` with an exact-zero first
    component ``tau[0]^2 = 0``. Component scales are fixed on an adaptive
    geometric grid and the mixture weights are estimated by EM.
    """

    stateArr, _, valid, stateValid, varianceValid = _sanitizeValidStateVariance(
        state,
        stateVariance,
    )
    minNull_, maxNull_ = _nullBounds(minNull, maxNull)

    if priorNull is None:
        pi0 = _initialNullProbability(
            stateValid,
            varianceValid,
            nullZ=float(nullZ),
            minNull=minNull_,
            maxNull=maxNull_,
        )
        fixedNull = False
    else:
        pi0 = float(priorNull)
        if not np.isfinite(pi0) or pi0 <= 0.0 or pi0 >= 1.0:
            raise ValueError("`priorNull` must be finite and strictly between 0 and 1")
        pi0 = float(np.clip(pi0, minNull_, maxNull_))
        fixedNull = True

    componentVariances, positiveScales = _buildAdaptiveNormalMixtureVariances(
        stateValid,
        varianceValid,
        priorScale=priorScale,
        componentCount=mixtureComponentCount,
    )
    weights, iterations, converged, logLikelihood = _fitAdaptiveNormalMixtureWeights(
        stateValid,
        varianceValid,
        componentVariances,
        initialNull=pi0,
        fixedNull=fixedNull,
        maxIter=maxIter,
        tol=tol,
    )
    (
        nullValid,
        slabWeightValid,
        slabMeanValid,
        shrunkValid,
        shrinkFactorValid,
    ) = _posteriorMomentsAdaptiveNormalMixture(
        stateValid,
        varianceValid,
        componentVariances,
        weights,
    )

    shrunk = stateArr.astype(np.float64, copy=True)
    nullProbOut = np.full_like(stateArr, np.nan, dtype=np.float64)
    slabMeanOut = np.full_like(stateArr, np.nan, dtype=np.float64)
    slabWeightOut = np.full_like(stateArr, np.nan, dtype=np.float64)
    shrinkFactor = np.ones_like(stateArr, dtype=np.float64)
    shrunk[valid] = shrunkValid
    nullProbOut[valid] = nullValid
    slabMeanOut[valid] = slabMeanValid
    slabWeightOut[valid] = slabWeightValid
    shrinkFactor[valid] = shrinkFactorValid

    priorScaleSummary = _mixturePriorScale(componentVariances, weights)
    metadata = {
        "model": STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE,
        "finite_count": int(np.count_nonzero(valid)),
        "interval_count": int(stateArr.size),
        "component_count": int(componentVariances.size),
        "positive_component_count": int(max(componentVariances.size - 1, 0)),
        "prior_null": _metadataFloat(weights[0]),
        "prior_scale": _metadataFloat(priorScaleSummary),
        "prior_variance": _metadataFloat(priorScaleSummary * priorScaleSummary),
        "component_scales": [
            _metadataFloat(math.sqrt(float(value))) for value in componentVariances
        ],
        "component_weights": [_metadataFloat(float(value)) for value in weights],
        "estimated_prior_null": bool(priorNull is None),
        "estimated_prior_scale": bool(priorScale is None),
        "fixed_null_weight": bool(fixedNull),
        "iterations": int(iterations),
        "converged": bool(converged),
        "log_likelihood": _metadataFloat(logLikelihood),
        "state_abs_median_before": _metadataFloat(np.median(np.abs(stateValid))),
        "state_abs_median_after": _metadataFloat(np.median(np.abs(shrunkValid))),
        "shrinkage_factor_median": _metadataFloat(np.median(shrinkFactorValid)),
        "null_probability_median": _metadataFloat(np.median(nullValid)),
        "mixture_min_positive_scale": _metadataFloat(np.min(positiveScales)),
        "mixture_max_positive_scale": _metadataFloat(np.max(positiveScales)),
    }

    return stateShrinkResult(
        shrunkState=shrunk.astype(np.float32),
        shrinkageFactor=shrinkFactor.astype(np.float32),
        nullProbability=nullProbOut.astype(np.float32),
        slabPosteriorMean=slabMeanOut.astype(np.float32),
        slabPosteriorWeight=slabWeightOut.astype(np.float32),
        priorNull=float(weights[0]),
        priorScale=float(priorScaleSummary),
        metadata=metadata,
    )


def _shrinkStateSpikeAndNormal(
    state: npt.ArrayLike,
    stateVariance: npt.ArrayLike,
    *,
    priorNull: float | None = None,
    priorScale: float | None = None,
    maxIter: int = _STATE_SHRINKAGE_DEFAULT_MAX_ITER,
    tol: float = _STATE_SHRINKAGE_DEFAULT_TOL,
    nullZ: float = _STATE_SHRINKAGE_DEFAULT_NULL_Z,
    minNull: float = _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
    maxNull: float = _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
) -> stateShrinkResult:
    r"""Shrink fitted state estimates toward zero using a spike-and-normal EB prior.

    The working likelihood is ``state[i] | ell[i] ~ N(ell[i], stateVariance[i])``.
    The prior is a point mass at zero plus a zero-centered normal slab. When
    ``priorNull`` or ``priorScale`` is not supplied, it is estimated by EM.
    """

    stateArr, _, valid, stateValid, varianceValid = _sanitizeValidStateVariance(
        state,
        stateVariance,
    )
    minNull_, maxNull_ = _nullBounds(minNull, maxNull)

    estimateNull = priorNull is None
    estimateScale = priorScale is None

    if priorNull is None:
        pi0 = _initialNullProbability(
            stateValid,
            varianceValid,
            nullZ=float(nullZ),
            minNull=minNull_,
            maxNull=maxNull_,
        )
    else:
        pi0 = float(priorNull)
        if not np.isfinite(pi0) or pi0 <= 0.0 or pi0 >= 1.0:
            raise ValueError("`priorNull` must be finite and strictly between 0 and 1")
        pi0 = float(np.clip(pi0, minNull_, maxNull_))

    if priorScale is None:
        tau2 = _initialSlabVariance(stateValid, varianceValid)
    else:
        tau = float(priorScale)
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("`priorScale` must be finite and positive")
        tau2 = tau * tau

    converged = False
    iterations = 0
    tol_ = float(max(tol, 0.0))
    maxIter_ = int(max(maxIter, 1))
    for iteration in range(maxIter_):
        iterations = iteration + 1
        nullProb, slabWeight, slabMean, slabVariance, _ = (
            _posteriorMomentsSpikeAndNormal(
                stateValid,
                varianceValid,
                priorNull=pi0,
                priorVariance=tau2,
            )
        )
        nextPi0 = pi0
        nextTau2 = tau2
        if estimateNull:
            nextPi0 = float(np.clip(np.mean(nullProb), minNull_, maxNull_))
        if estimateScale:
            slabMass = float(np.sum(slabWeight))
            if np.isfinite(slabMass) and slabMass > _STATE_SHRINKAGE_POSITIVE_FLOOR:
                nextTau2 = float(
                    np.sum(slabWeight * (slabVariance + slabMean * slabMean))
                    / slabMass
                )
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

    nullValid, slabWeightValid, slabMeanValid, _, shrinkFactorValid = (
        _posteriorMomentsSpikeAndNormal(
            stateValid,
            varianceValid,
            priorNull=pi0,
            priorVariance=tau2,
        )
    )
    shrunkValid = slabWeightValid * slabMeanValid

    shrunk = stateArr.astype(np.float64, copy=True)
    nullProbOut = np.full_like(stateArr, np.nan, dtype=np.float64)
    slabMeanOut = np.full_like(stateArr, np.nan, dtype=np.float64)
    slabWeightOut = np.full_like(stateArr, np.nan, dtype=np.float64)
    shrinkFactor = np.ones_like(stateArr, dtype=np.float64)
    shrunk[valid] = shrunkValid
    nullProbOut[valid] = nullValid
    slabMeanOut[valid] = slabMeanValid
    slabWeightOut[valid] = slabWeightValid
    shrinkFactor[valid] = shrinkFactorValid

    metadata = {
        "model": STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,
        "finite_count": int(np.count_nonzero(valid)),
        "interval_count": int(stateArr.size),
        "prior_null": _metadataFloat(pi0),
        "prior_scale": _metadataFloat(math.sqrt(tau2)),
        "prior_variance": _metadataFloat(tau2),
        "estimated_prior_null": bool(estimateNull),
        "estimated_prior_scale": bool(estimateScale),
        "iterations": int(iterations),
        "converged": bool(converged),
        "state_abs_median_before": _metadataFloat(np.median(np.abs(stateValid))),
        "state_abs_median_after": _metadataFloat(np.median(np.abs(shrunkValid))),
        "shrinkage_factor_median": _metadataFloat(np.median(shrinkFactorValid)),
        "null_probability_median": _metadataFloat(np.median(nullValid)),
    }

    return stateShrinkResult(
        shrunkState=shrunk.astype(np.float32),
        shrinkageFactor=shrinkFactor.astype(np.float32),
        nullProbability=nullProbOut.astype(np.float32),
        slabPosteriorMean=slabMeanOut.astype(np.float32),
        slabPosteriorWeight=slabWeightOut.astype(np.float32),
        priorNull=float(pi0),
        priorScale=float(math.sqrt(tau2)),
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
    mixtureComponentCount: int = _STATE_SHRINKAGE_DEFAULT_MIXTURE_COMPONENTS,
) -> stateShrinkResult:
    r"""Shrink fitted state estimates toward zero using post-fit EB.

    ``adaptiveNormalMixture`` is the default and uses an ashr-like mixture of
    zero-centered Normal priors with an exact-zero component. ``spikeAndNormal``
    preserves the original single-slab shrinker.
    """

    model_ = _normalizeModel(model)
    if model_ == STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE:
        return _shrinkStateAdaptiveNormalMixture(
            state,
            stateVariance,
            priorNull=priorNull,
            priorScale=priorScale,
            maxIter=maxIter,
            tol=tol,
            nullZ=nullZ,
            minNull=minNull,
            maxNull=maxNull,
            mixtureComponentCount=mixtureComponentCount,
        )
    if model_ == STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL:
        return _shrinkStateSpikeAndNormal(
            state,
            stateVariance,
            priorNull=priorNull,
            priorScale=priorScale,
            maxIter=maxIter,
            tol=tol,
            nullZ=nullZ,
            minNull=minNull,
            maxNull=maxNull,
        )
    supported = ", ".join(STATE_SHRINKAGE_MODELS)
    raise ValueError(
        f"Unsupported state shrinkage model {model!r}; supported values: {supported}."
    )
