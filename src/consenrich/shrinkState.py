"""(Experimental) Post-fit empirical-Bayes shrinkage for Consenrich state estimates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import special

from . import constants
from .cconsenrich import (
    cstateShrinkInitialSums as _cstateShrinkInitialSums,
    cstateShrinkMixtureEMStep as _cstateShrinkMixtureEMStep,
    cstateShrinkMixtureEMStepPrepared as _cstateShrinkMixtureEMStepPrepared,
    cstateShrinkMixturePosterior as _cstateShrinkMixturePosterior,
    cstateShrinkMixturePosteriorPrepared as _cstateShrinkMixturePosteriorPrepared,
)

STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE = "adaptiveNormalMixture"
STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL = "spikeAndNormal"
STATE_SHRINKAGE_MODEL_SPIKE_AND_STUDENT_T = "spikeAndStudentT"
STATE_SHRINKAGE_MODEL = STATE_SHRINKAGE_MODEL_SPIKE_AND_STUDENT_T
STATE_SHRINKAGE_MODELS = (
    STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE,
    STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,
    STATE_SHRINKAGE_MODEL_SPIKE_AND_STUDENT_T,
)
_STATE_SHRINKAGE_POSITIVE_FLOOR = 1.0e-12
_STATE_SHRINKAGE_WEIGHT_FLOOR = 1.0e-12
_STATE_SHRINKAGE_DEFAULT_MAX_ITER = 96
_STATE_SHRINKAGE_DEFAULT_TOL = 1.0e-5
_STATE_SHRINKAGE_DEFAULT_NULL_Z = 1.5
_STATE_SHRINKAGE_DEFAULT_MIN_NULL = 1.0e-2
_STATE_SHRINKAGE_DEFAULT_MAX_NULL = 1.0 - 1.0e-4
_STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT = (
    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT
)
_STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT_FRACTION = (
    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT_FRACTION
)
_STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT_MIN = (
    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT_MIN
)
_STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT_MAX = (
    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT_MAX
)
_STATE_SHRINKAGE_DEFAULT_SCALE_ANCHOR_WEIGHT = (
    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT
)
_STATE_SHRINKAGE_DEFAULT_SCALE_ANCHOR_WEIGHT_FRACTION = (
    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT_FRACTION
)
_STATE_SHRINKAGE_DEFAULT_SCALE_ANCHOR_WEIGHT_MIN = (
    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT_MIN
)
_STATE_SHRINKAGE_DEFAULT_SLAB_SCALE_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
_STATE_SHRINKAGE_DEFAULT_STUDENT_T_DF = 1.0
_STATE_SHRINKAGE_MIN_STUDENT_T_DF = 1.0
_STATE_SHRINKAGE_MAX_STUDENT_T_DF = 30.0
_STATE_SHRINKAGE_DEFAULT_STUDENT_T_QUADRATURE_ORDER = 12
_STATE_SHRINKAGE_MIN_STUDENT_T_QUADRATURE_ORDER = 8
_STATE_SHRINKAGE_MAX_STUDENT_T_QUADRATURE_ORDER = 96
_STATE_SHRINKAGE_DEFAULT_STUDENT_T_SLAB_VARIANCE_FLOOR_FACTOR = 4.0


class stateShrinkPrior(NamedTuple):
    r"""Specifies genome-wide EB prior for state shrinkage (spike+normal, mixture)."""

    model: str
    priorSpikeProp: float
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
    spikeProp: np.ndarray
    slabPosteriorMean: np.ndarray
    slabPosteriorWeight: np.ndarray
    priorSpikeProp: float
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
        raise ValueError(
            "state shrinkage slabs must be non-empty one-dimensional arrays"
        )
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


def _normalizeStudentTDF(studentTDF: float) -> float:
    if isinstance(studentTDF, bool):
        raise ValueError("`studentTDF` must be numeric with 1 <= studentTDF <= 30")
    try:
        df = float(studentTDF)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "`studentTDF` must be numeric with 1 <= studentTDF <= 30"
        ) from exc
    if (
        not np.isfinite(df)
        or df < _STATE_SHRINKAGE_MIN_STUDENT_T_DF
        or df > _STATE_SHRINKAGE_MAX_STUDENT_T_DF
    ):
        raise ValueError("`studentTDF` must be numeric with 1 <= studentTDF <= 30")
    return df


def _normalizeStudentTQuadratureOrder(studentTQuadratureOrder: int) -> int:
    if isinstance(studentTQuadratureOrder, bool):
        raise ValueError(
            "`studentTQuadratureOrder` must be an integer with 8 <= order <= 96"
        )
    try:
        orderFloat = float(studentTQuadratureOrder)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "`studentTQuadratureOrder` must be an integer with 8 <= order <= 96"
        ) from exc
    if not np.isfinite(orderFloat) or not orderFloat.is_integer():
        raise ValueError(
            "`studentTQuadratureOrder` must be an integer with 8 <= order <= 96"
        )
    order = int(orderFloat)
    if (
        order < _STATE_SHRINKAGE_MIN_STUDENT_T_QUADRATURE_ORDER
        or order > _STATE_SHRINKAGE_MAX_STUDENT_T_QUADRATURE_ORDER
    ):
        raise ValueError(
            "`studentTQuadratureOrder` must be an integer with 8 <= order <= 96"
        )
    return order


def _normalizeNonnegativeFloat(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be nonnegative")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be nonnegative") from exc
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"`{name}` must be nonnegative")
    return out


def _isStudentTModel(model: str) -> bool:
    return model == STATE_SHRINKAGE_MODEL_SPIKE_AND_STUDENT_T


def _studentTSlabArrays(
    priorScale: float,
    studentTDF: float,
    studentTQuadratureOrder: int,
    minSlabVariance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    scale = float(priorScale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("`priorScale` must be finite and positive")
    df = _normalizeStudentTDF(studentTDF)
    order = _normalizeStudentTQuadratureOrder(studentTQuadratureOrder)
    alpha = df / 2.0 - 1.0
    nodes, weights = special.roots_genlaguerre(order, alpha)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        nodes.size != order
        or weights.size != order
        or np.any(~np.isfinite(nodes))
        or np.any(~np.isfinite(weights))
        or np.any(nodes <= 0.0)
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Student-t quadrature produced invalid nodes or weights")
    if df > 2.0:
        slabMultiplier = (df - 2.0) / (2.0 * nodes)
        studentTScale = scale * math.sqrt((df - 2.0) / df)
    else:
        slabMultiplier = df / (2.0 * nodes)
        studentTScale = scale
    slabVariance = np.maximum(
        scale * scale * slabMultiplier,
        max(float(minSlabVariance), _STATE_SHRINKAGE_POSITIVE_FLOOR),
    )
    weight = weights / float(np.sum(weights))
    orderIndex = np.argsort(slabVariance, kind="mergesort")
    slabVariance = slabVariance[orderIndex]
    weight = weight[orderIndex]
    slabMultiplier = slabMultiplier[orderIndex]
    weight = np.maximum(weight, _STATE_SHRINKAGE_WEIGHT_FLOOR)
    weight = weight / float(np.sum(weight))
    return slabVariance, weight, slabMultiplier, studentTScale, alpha, order


def _studentTPriorVariance(
    slabVariance: np.ndarray,
    slabMultiplier: np.ndarray,
) -> float:
    ratio = np.asarray(slabVariance, dtype=np.float64) / np.asarray(
        slabMultiplier,
        dtype=np.float64,
    )
    return float(np.mean(ratio))


def _priorSlabArrays(prior: stateShrinkPrior) -> tuple[np.ndarray, np.ndarray]:
    if len(prior.slabVariance) and len(prior.slabWeight):
        return _sortedPositiveWeights(prior.slabVariance, prior.slabWeight)
    return _sortedPositiveWeights((prior.priorVariance,), (1.0,))


def _componentWeights(
    priorSpikeProp: float,
    slabWeight: np.ndarray,
) -> tuple[float, ...]:
    pi0 = float(priorSpikeProp)
    return tuple([pi0] + [float((1.0 - pi0) * weight) for weight in slabWeight])


def _modelComponentWeights(
    model: str,
    priorSpikeProp: float,
    slabWeight: np.ndarray,
) -> tuple[float, ...]:
    return _componentWeights(priorSpikeProp, slabWeight)


def _slabMultipliersForModel(model: str) -> tuple[float, ...]:
    if model == STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL:
        return (1.0,)
    if _isStudentTModel(model):
        raise ValueError("Student-t slabs require quadrature construction")
    return _STATE_SHRINKAGE_DEFAULT_SLAB_SCALE_MULTIPLIERS


def _logSlabPrior(priorSpikeProp: float, slabWeight: np.ndarray) -> np.ndarray:
    pi0 = float(priorSpikeProp)
    if not np.isfinite(pi0) or pi0 <= 0.0 or pi0 >= 1.0:
        raise ValueError("`priorSpikeProp` must be finite and strictly between 0 and 1")
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
    priorSpikeProp: float,
    slabVariance: np.ndarray,
    logSlabPrior: np.ndarray,
    blockSize: int,
) -> tuple[float, float, np.ndarray, np.ndarray, float, int]:
    out = _cstateShrinkMixtureEMStepPrepared(
        state,
        variance,
        float(priorSpikeProp),
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


def _chunksLogLikelihood(
    chunks: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    priorSpikeProp: float,
    slabVariance: np.ndarray,
    slabWeight: np.ndarray,
    blockSize: int,
) -> float:
    logSlabPrior = _logSlabPrior(priorSpikeProp, slabWeight)
    logLikelihood = 0.0
    for state, variance in chunks:
        logLikelihood += float(
            _emStep(
                state,
                variance,
                priorSpikeProp=priorSpikeProp,
                slabVariance=slabVariance,
                logSlabPrior=logSlabPrior,
                blockSize=blockSize,
            )[4]
        )
    return logLikelihood


def _logObjectivePenalty(
    *,
    priorSpikeProp: float,
    slabVariance: np.ndarray,
    slabMultiplier: np.ndarray | None,
    spikePseudoCount: float,
    slabPseudoCount: float,
    scaleVarianceAnchor: float,
    scalePriorWeight: float,
) -> float:
    penalty = 0.0
    if spikePseudoCount > 0.0:
        penalty += float(spikePseudoCount) * math.log(
            max(float(priorSpikeProp), _STATE_SHRINKAGE_POSITIVE_FLOOR)
        )
    if slabPseudoCount > 0.0:
        penalty += float(slabPseudoCount) * math.log(
            max(1.0 - float(priorSpikeProp), _STATE_SHRINKAGE_POSITIVE_FLOOR)
        )
    if scalePriorWeight > 0.0:
        if slabMultiplier is None:
            raise ValueError("Student-t slab multipliers are missing")
        scaleVariance = max(
            _studentTPriorVariance(slabVariance, slabMultiplier),
            _STATE_SHRINKAGE_POSITIVE_FLOOR,
        )
        anchor = max(float(scaleVarianceAnchor), _STATE_SHRINKAGE_POSITIVE_FLOOR)
        penalty += (
            -0.5
            * float(scalePriorWeight)
            * (math.log(scaleVariance) + anchor / scaleVariance)
        )
    return penalty


def fitStateShrinkagePrior(
    chunks: Sequence[Any],
    *,
    model: str | None = None,
    priorSpikeProp: float | None = None,
    priorScale: float | None = None,
    stateShrinkageSpikePseudoCount: float | None = (
        _STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT
    ),
    stateShrinkageScaleAnchorWeight: float | None = (
        _STATE_SHRINKAGE_DEFAULT_SCALE_ANCHOR_WEIGHT
    ),
    studentTDF: float = _STATE_SHRINKAGE_DEFAULT_STUDENT_T_DF,
    studentTQuadratureOrder: int = _STATE_SHRINKAGE_DEFAULT_STUDENT_T_QUADRATURE_ORDER,
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
    studentTModel = _isStudentTModel(model_)
    studentTDF_ = _normalizeStudentTDF(studentTDF)
    studentTQuadratureOrder_ = _normalizeStudentTQuadratureOrder(
        studentTQuadratureOrder
    )
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

    if stateShrinkageSpikePseudoCount is None:
        spikePseudoCount = float(
            np.clip(
                _STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT_FRACTION * totalWeight,
                _STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT_MIN,
                _STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT_MAX,
            )
        )
    else:
        if isinstance(stateShrinkageSpikePseudoCount, bool):
            raise ValueError("`stateShrinkageSpikePseudoCount` must be nonnegative")
        try:
            spikePseudoCount = float(stateShrinkageSpikePseudoCount)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "`stateShrinkageSpikePseudoCount` must be nonnegative"
            ) from exc
        if not np.isfinite(spikePseudoCount) or spikePseudoCount < 0.0:
            raise ValueError("`stateShrinkageSpikePseudoCount` must be nonnegative")
    slabPseudoCount = 0.0
    if stateShrinkageScaleAnchorWeight is None:
        scalePriorWeight = max(
            _STATE_SHRINKAGE_DEFAULT_SCALE_ANCHOR_WEIGHT_MIN,
            _STATE_SHRINKAGE_DEFAULT_SCALE_ANCHOR_WEIGHT_FRACTION * totalWeight,
        )
    else:
        scalePriorWeight = _normalizeNonnegativeFloat(
            "stateShrinkageScaleAnchorWeight",
            stateShrinkageScaleAnchorWeight,
        )

    estimateSpikeProp = priorSpikeProp is None

    if priorSpikeProp is None:
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
        pi0 = float(priorSpikeProp)
        if not np.isfinite(pi0) or pi0 <= 0.0 or pi0 >= 1.0:
            raise ValueError(
                "`priorSpikeProp` must be finite and strictly between 0 and 1"
            )
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
    scaleVarianceAnchor = max(
        float(baseScale * baseScale),
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )
    stateVarianceAnchor = max(
        float(varianceSum / totalWeight),
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )
    studentTSlabVarianceFloor = max(
        float(
            _STATE_SHRINKAGE_DEFAULT_STUDENT_T_SLAB_VARIANCE_FLOOR_FACTOR
            * stateVarianceAnchor
        ),
        _STATE_SHRINKAGE_POSITIVE_FLOOR,
    )

    studentTScale: float | None = None
    studentTQuadratureAlpha: float | None = None
    slabMultiplier: np.ndarray | None = None
    if studentTModel:
        (
            slabVariance,
            slabWeight,
            slabMultiplier,
            studentTScale,
            studentTQuadratureAlpha,
            studentTQuadratureOrder_,
        ) = _studentTSlabArrays(
            baseScale,
            studentTDF_,
            studentTQuadratureOrder_,
            studentTSlabVarianceFloor,
        )
        estimateSlabWeights = False
    else:
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
    if estimateSpikeProp or estimateSlabScales or estimateSlabWeights:
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
                    priorSpikeProp=pi0,
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
            if estimateSpikeProp and emTotalWeight > 0.0:
                totalMass = float(emTotalWeight)
                nextPi0 = float(
                    np.clip(
                        (nullMass + spikePseudoCount)
                        / (totalMass + spikePseudoCount + slabPseudoCount),
                        minNull_,
                        maxNull_,
                    )
                )
            if estimateSlabWeights:
                massTotal = float(np.sum(slabMass))
                if massTotal > _STATE_SHRINKAGE_POSITIVE_FLOOR:
                    nextSlabWeight = np.maximum(
                        slabMass / massTotal,
                        _STATE_SHRINKAGE_WEIGHT_FLOOR,
                    )
                    nextSlabWeight = nextSlabWeight / float(np.sum(nextSlabWeight))
            if estimateSlabScales:
                if studentTModel:
                    if slabMultiplier is None:
                        raise ValueError("Student-t slab multipliers are missing")
                    massTotal = float(np.sum(slabMass))
                    if massTotal + scalePriorWeight > _STATE_SHRINKAGE_POSITIVE_FLOOR:
                        scaleMomentSum = float(np.sum(slabSecond / slabMultiplier))
                        nextPriorVariance = float(
                            (scaleMomentSum + scalePriorWeight * scaleVarianceAnchor)
                            / (massTotal + scalePriorWeight)
                        )
                        nextPriorVariance = max(
                            nextPriorVariance,
                            _STATE_SHRINKAGE_POSITIVE_FLOOR,
                        )
                        nextSlabVariance = nextPriorVariance * slabMultiplier
                        nextSlabVariance = np.maximum(
                            nextSlabVariance,
                            studentTSlabVarianceFloor,
                        )
                else:
                    active = slabMass > _STATE_SHRINKAGE_POSITIVE_FLOOR
                    nextSlabVariance[active] = slabSecond[active] / slabMass[active]
                    nextSlabVariance = np.maximum(
                        nextSlabVariance,
                        _STATE_SHRINKAGE_POSITIVE_FLOOR,
                    )
                    nextSlabVariance[~np.isfinite(nextSlabVariance)] = slabVariance[
                        ~np.isfinite(nextSlabVariance)
                    ]
            if not studentTModel:
                nextSlabVariance, nextSlabWeight = _sortedPositiveWeights(
                    nextSlabVariance,
                    nextSlabWeight,
                )
            if studentTModel:
                objective = logLikelihood + _logObjectivePenalty(
                    priorSpikeProp=pi0,
                    slabVariance=slabVariance,
                    slabMultiplier=slabMultiplier,
                    spikePseudoCount=spikePseudoCount,
                    slabPseudoCount=slabPseudoCount,
                    scaleVarianceAnchor=scaleVarianceAnchor,
                    scalePriorWeight=scalePriorWeight,
                )
                nextLogLikelihood = _chunksLogLikelihood(
                    chunks_,
                    priorSpikeProp=nextPi0,
                    slabVariance=nextSlabVariance,
                    slabWeight=nextSlabWeight,
                    blockSize=blockSize_,
                )
                nextObjective = nextLogLikelihood + _logObjectivePenalty(
                    priorSpikeProp=nextPi0,
                    slabVariance=nextSlabVariance,
                    slabMultiplier=slabMultiplier,
                    spikePseudoCount=spikePseudoCount,
                    slabPseudoCount=slabPseudoCount,
                    scaleVarianceAnchor=scaleVarianceAnchor,
                    scalePriorWeight=scalePriorWeight,
                )
                if nextObjective < objective - max(tol_, 1.0e-12):
                    accepted = False
                    if estimateSlabScales and slabMultiplier is not None:
                        anchorPriorVariance = _studentTPriorVariance(
                            slabVariance,
                            slabMultiplier,
                        )
                        proposedPriorVariance = _studentTPriorVariance(
                            nextSlabVariance,
                            slabMultiplier,
                        )
                        logAnchor = math.log(
                            max(anchorPriorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR)
                        )
                        logProposed = math.log(
                            max(proposedPriorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR)
                        )
                        for backtrackStep in range(12):
                            fraction = 0.5 ** float(backtrackStep + 1)
                            trialPriorVariance = math.exp(
                                logAnchor + fraction * (logProposed - logAnchor)
                            )
                            trialSlabVariance = trialPriorVariance * slabMultiplier
                            trialSlabVariance = np.maximum(
                                trialSlabVariance,
                                studentTSlabVarianceFloor,
                            )
                            trialLogLikelihood = _chunksLogLikelihood(
                                chunks_,
                                priorSpikeProp=nextPi0,
                                slabVariance=trialSlabVariance,
                                slabWeight=nextSlabWeight,
                                blockSize=blockSize_,
                            )
                            trialObjective = trialLogLikelihood + _logObjectivePenalty(
                                priorSpikeProp=nextPi0,
                                slabVariance=trialSlabVariance,
                                slabMultiplier=slabMultiplier,
                                spikePseudoCount=spikePseudoCount,
                                slabPseudoCount=slabPseudoCount,
                                scaleVarianceAnchor=scaleVarianceAnchor,
                                scalePriorWeight=scalePriorWeight,
                            )
                            if trialObjective >= objective - max(tol_, 1.0e-12):
                                nextSlabVariance = trialSlabVariance
                                nextLogLikelihood = trialLogLikelihood
                                accepted = True
                                break
                    if not accepted:
                        nextPi0 = pi0
                        nextSlabVariance = slabVariance
                        nextSlabWeight = slabWeight
                        nextLogLikelihood = logLikelihood
                        converged = False
                logLikelihood = nextLogLikelihood
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
            if (
                studentTModel
                and not converged
                and relPi == 0.0
                and relWeight == 0.0
                and relVariance == 0.0
            ):
                break
    else:
        converged = True
        iterations = 0

    if studentTModel:
        if slabMultiplier is None:
            raise ValueError("Student-t slab multipliers are missing")
        scaleVariance = _studentTPriorVariance(slabVariance, slabMultiplier)
        if not estimateSlabScales:
            priorScaleOut = float(baseScale)
            if studentTDF_ > 2.0:
                priorVariance = float(baseScale * baseScale)
                studentTScale = float(
                    baseScale * math.sqrt((studentTDF_ - 2.0) / studentTDF_)
                )
                priorVarianceDefined = True
            else:
                priorVariance = float("nan")
                studentTScale = float(baseScale)
                priorVarianceDefined = False
        elif studentTDF_ > 2.0:
            priorVariance = scaleVariance
            studentTScale = math.sqrt(
                max(priorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR)
                * (studentTDF_ - 2.0)
                / studentTDF_
            )
            priorScaleOut = float(
                math.sqrt(max(priorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR))
            )
            priorVarianceDefined = True
        else:
            priorVariance = float("nan")
            studentTScale = math.sqrt(
                max(scaleVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR)
            )
            priorScaleOut = float(studentTScale)
            priorVarianceDefined = False
    else:
        priorVariance = float(np.sum(slabWeight * slabVariance))
        priorScaleOut = float(
            math.sqrt(max(priorVariance, _STATE_SHRINKAGE_POSITIVE_FLOOR))
        )
        priorVarianceDefined = True
    componentWeights = _modelComponentWeights(model_, pi0, slabWeight)
    metadata = {
        "model": model_,
        "slab_family": (
            "studentTNormalScaleMixture" if studentTModel else "normalMixture"
        ),
        "has_point_mass": True,
        "scope": "genome",
        "chunk_count": int(chunkCount),
        "interval_count": int(intervalCount),
        "finite_count": int(finiteCount),
        "effective_block_count": _metadataFloat(totalWeight),
        "block_size_intervals": int(blockSize_),
        "prior_spike_prop": _metadataFloat(pi0),
        "prior_scale": _metadataFloat(priorScaleOut),
        "prior_variance": _metadataFloat(priorVariance),
        "prior_variance_defined": bool(priorVarianceDefined),
        "slab_count": int(slabVariance.size),
        "slab_variance": [float(value) for value in slabVariance],
        "slab_weight": [float(value) for value in slabWeight],
        "component_weights": [float(value) for value in componentWeights],
        "estimated_prior_spike_prop": bool(estimateSpikeProp),
        "spike_pseudo_count": _metadataFloat(spikePseudoCount),
        "estimated_prior_scale": bool(estimateSlabScales),
        "estimated_slab_weights": bool(estimateSlabWeights),
        "estimated_slab_scales": bool(estimateSlabScales),
        "scale_anchor_weight": _metadataFloat(scalePriorWeight),
        "state_variance_anchor": _metadataFloat(stateVarianceAnchor),
        "iterations": int(iterations),
        "converged": bool(converged),
        "log_likelihood": _metadataFloat(logLikelihood),
    }
    if studentTModel:
        metadata.update(
            {
                "student_t_df": _metadataFloat(studentTDF_),
                "student_t_scale": _metadataFloat(float(studentTScale)),
                "student_t_quadrature_order": int(studentTQuadratureOrder_),
                "student_t_quadrature_alpha": _metadataFloat(
                    float(studentTQuadratureAlpha)
                ),
                "student_t_min_slab_variance": _metadataFloat(
                    studentTSlabVarianceFloor
                ),
                "student_t_min_slab_scale": _metadataFloat(
                    math.sqrt(studentTSlabVarianceFloor)
                ),
                "student_t_slab_variance_floor_factor": _metadataFloat(
                    _STATE_SHRINKAGE_DEFAULT_STUDENT_T_SLAB_VARIANCE_FLOOR_FACTOR
                ),
                "slab_multiplier": [float(value) for value in slabMultiplier],
            }
        )
    return stateShrinkPrior(
        model=model_,
        priorSpikeProp=float(pi0),
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
    priorSpikeProp: float,
    slabVariance: np.ndarray,
    slabWeight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        _cstateShrinkMixturePosteriorPrepared(
            state,
            variance,
            float(priorSpikeProp),
            slabVariance,
            _logSlabPrior(priorSpikeProp, slabWeight),
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
        spikeProp,
        slabMean,
        slabWeight,
    ) = _posterior(
        stateArr,
        varianceArr,
        priorSpikeProp=float(prior.priorSpikeProp),
        slabVariance=priorSlabVariance,
        slabWeight=priorSlabWeight,
    )
    valid = np.isfinite(stateArr) & np.isfinite(varianceArr) & (varianceArr > 0.0)
    metadata = {
        **dict(prior.metadata),
        "scope": "contig_apply",
        "has_point_mass": True,
        "interval_count": int(stateArr.size),
        "finite_count": int(np.count_nonzero(valid)),
        "slab_count": int(priorSlabVariance.size),
        "slab_variance": [float(value) for value in priorSlabVariance],
        "slab_weight": [float(value) for value in priorSlabWeight],
        "component_weights": [
            float(value)
            for value in _modelComponentWeights(
                prior.model,
                prior.priorSpikeProp,
                priorSlabWeight,
            )
        ],
        "state_abs_median_before": _metadataFloat(
            np.median(np.abs(stateArr[valid])) if np.any(valid) else float("nan")
        ),
        "state_abs_median_after": _metadataFloat(
            np.median(np.abs(shrunk[valid])) if np.any(valid) else float("nan")
        ),
        "spike_prop_median": _metadataFloat(
            np.median(spikeProp[valid]) if np.any(valid) else float("nan")
        ),
        "posterior_sd_median": _metadataFloat(
            np.median(posteriorSd[valid]) if np.any(valid) else float("nan")
        ),
    }
    return stateShrinkResult(
        shrunkState=np.asarray(shrunk, dtype=np.float32),
        posteriorSd=np.asarray(posteriorSd, dtype=np.float32),
        spikeProp=np.asarray(spikeProp, dtype=np.float32),
        slabPosteriorMean=np.asarray(slabMean, dtype=np.float32),
        slabPosteriorWeight=np.asarray(slabWeight, dtype=np.float32),
        priorSpikeProp=float(prior.priorSpikeProp),
        priorScale=float(prior.priorScale),
        metadata=metadata,
    )


def shrinkStateEB(
    state: npt.ArrayLike,
    stateVariance: npt.ArrayLike,
    *,
    model: str | None = None,
    priorSpikeProp: float | None = None,
    priorScale: float | None = None,
    stateShrinkageSpikePseudoCount: float | None = (
        _STATE_SHRINKAGE_DEFAULT_SPIKE_PSEUDO_COUNT
    ),
    stateShrinkageScaleAnchorWeight: float | None = (
        _STATE_SHRINKAGE_DEFAULT_SCALE_ANCHOR_WEIGHT
    ),
    studentTDF: float = _STATE_SHRINKAGE_DEFAULT_STUDENT_T_DF,
    studentTQuadratureOrder: int = _STATE_SHRINKAGE_DEFAULT_STUDENT_T_QUADRATURE_ORDER,
    maxIter: int = _STATE_SHRINKAGE_DEFAULT_MAX_ITER,
    tol: float = _STATE_SHRINKAGE_DEFAULT_TOL,
    nullZ: float = _STATE_SHRINKAGE_DEFAULT_NULL_Z,
    minNull: float = _STATE_SHRINKAGE_DEFAULT_MIN_NULL,
    maxNull: float = _STATE_SHRINKAGE_DEFAULT_MAX_NULL,
    blockSize: int | None = 1,
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
        priorSpikeProp=priorSpikeProp,
        priorScale=priorScale,
        stateShrinkageSpikePseudoCount=stateShrinkageSpikePseudoCount,
        stateShrinkageScaleAnchorWeight=stateShrinkageScaleAnchorWeight,
        studentTDF=studentTDF,
        studentTQuadratureOrder=studentTQuadratureOrder,
        maxIter=maxIter,
        tol=tol,
        nullZ=nullZ,
        minNull=minNull,
        maxNull=maxNull,
        blockSize=blockSize,
    )
    return applyStateShrinkagePrior(stateArr, varianceArr, prior)
