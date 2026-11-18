# -*- coding: utf-8 -*-

# the name of this file is unfortunately misleading --
# 'core' as in 'central' or 'basic', not the literal
# `consenrich.core` module

import inspect
import logging
import math
import os
import tempfile
from typing import Tuple, List, Optional
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats
import scipy.signal as spySig  # renamed to avoid conflict with any `signal` variables

from consenrich import _normalization as normalization
import consenrich.core as core
import consenrich.constants as constants
import consenrich.consenrich as consenrichRuntime
import consenrich.cconsenrich as cconsenrich
import consenrich.diagnostics as diagnostics
import consenrich.misc_util as misc_util
import consenrich.peaks as peaks
import consenrich.shrinkState as shrinkState

TESTS_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TESTS_DIR / "data"
FRAGMENTS_DIR = TEST_DATA_DIR / "fragments"


def test_stateShrinkageCenteredMomentsAndFittedLikelihood():
    state = np.asarray([0.0, -0.2, 1.0, 1e4, 1e5, 1e6, -1e7])
    variance = np.full(state.size, 1e-6)
    for tau, slabWeight in (
        (np.asarray([1.44]), np.asarray([1.0])),
        (np.asarray([0.02, 1.44, 25.0]), np.asarray([0.2, 0.5, 0.3])),
    ):
        result = cconsenrich.cstateShrinkMixturePosterior(
            state, variance, 0.4, tau, slabWeight
        )
        expectedMean, expectedSD = [], []
        for x, v in zip(state, variance):
            componentVariance = np.concatenate(([0.0], tau))
            priorWeight = np.concatenate(([0.4], 0.6 * slabWeight))
            logWeight = np.log(priorWeight) + stats.norm.logpdf(
                x, scale=np.sqrt(v + componentVariance)
            )
            weight = np.exp(logWeight - np.max(logWeight))
            weight /= math.fsum(weight)
            componentMean = componentVariance / (v + componentVariance) * x
            conditionalVariance = componentVariance / (v + componentVariance) * v
            mean = math.fsum(weight * componentMean)
            postVariance = math.fsum(
                weight * (conditionalVariance + (componentMean - mean) ** 2)
            )
            expectedMean.append(mean)
            expectedSD.append(math.sqrt(max(postVariance, 1e-12)))
        np.testing.assert_allclose(result[0], expectedMean, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(result[1], expectedSD, rtol=1e-6)
    repeatedSlab = cconsenrich.cstateShrinkMixturePosterior(
        [1e20, -1e20], [1e-6, 1e-6], 0.4, [1.44] * 4, [0.1, 0.2, 0.3, 0.4]
    )
    np.testing.assert_allclose(repeatedSlab[1], math.sqrt(1.44e-6 / (1.44 + 1e-6)), rtol=1e-6)

    rng = np.random.default_rng(42)
    state = np.concatenate((rng.normal(0.0, 0.3, 200), rng.normal(4.0, 1.0, 100)))
    variance = np.full(state.size, 0.09)
    for family in shrinkState.STATE_SHRINKAGE_MODELS:
        for fixed in (False, True):
            prior = shrinkState.fitStateShrinkagePrior(
                [(state, variance)], model=family, maxIter=1, blockSize=2,
                priorSpikeProp=0.4 if fixed else None,
                priorScale=1.2 if fixed else None,
                stateShrinkageSpikePseudoCount=0.0,
                stateShrinkageScaleAnchorWeight=0.0,
            )
            tau, slabWeight = shrinkState._priorSlabArrays(prior)
            logDensity = stats.norm.logpdf(
                state[:, None], scale=np.sqrt(variance[:, None] + np.r_[0.0, tau])
            )
            logDensity += np.log(np.r_[prior.priorSpikeProp, (1-prior.priorSpikeProp)*slabWeight])
            expected = np.logaddexp.reduce(logDensity, axis=1).reshape(-1, 2).mean(axis=1).sum()
            assert prior.metadata["log_likelihood"] == pytest.approx(expected, rel=1e-12)


def test_roughness_ldl_guardrails(caplog):
    assert constants.FIT_BACKGROUND_LENGTH_SCALE_FLOOR_BP == 5_000
    assert constants.FIT_BACKGROUND_LENGTH_SCALE_CAP_BP == 100_000
    for scaleIntervals, configuredBP, binBP, multiplier, provided, expected in (
        (1, 0, 50, 8.0, False, 101),
        (13, -1, 50, 8.0, False, 105),
        (None, 100, 50, 8.0, False, 3),
        (1, 0, 300, 8.0, True, 17),
        (171, 0, 50, 16.0, False, 1999),
        (171, 0, 50, 16.0, True, 171),
        (200, 0, 50, 16.0, False, 1999),
        (400, 0, 50, -1.0, True, 401),
        (2400, 0, 50, None, True, 1999),
        (None, 20_000, 50, 16.0, True, 401),
        (None, 200_000, 50, None, True, 4001),
        (None, 101, 50, 8.0, False, 3),
        (None, 1, 50, None, False, 1),
        (None, 50, 50, "unused", False, 1),
        (None, 100_001, 100_001, -1.0, True, 1),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            span = consenrichRuntime._resolveRuntimeBackgroundBlockLen(
                scaleIntervals, configuredBP, binBP, multiplier,
                lengthScaleProvided=provided,
            )
        assert span == expected
        warnings = [
            record for record in caplog.records
            if "backgroundBlockSizeBP" in record.getMessage()
        ]
        adjusted = configuredBP > 0 and span * binBP != configuredBP
        assert len(warnings) == int(adjusted)
        if configuredBP > 0:
            assert span % 2 == 1 and span * binBP >= configuredBP
        else:
            assert span % 2 == 1 and 5_000 <= span * binBP <= 100_000
        if adjusted:
            message = warnings[0].getMessage()
            assert warnings[0].levelno == logging.WARNING
            assert f"from {configuredBP} bp to {span * binBP} bp" in message
            assert f"({span} bins at {binBP} bp)" in message
    with pytest.raises(ValueError, match="counting bin"):
        consenrichRuntime._resolveRuntimeBackgroundBlockLen(
            1, 0, 100_001, 8.0, lengthScaleProvided=False,
        )
    for multiplier in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            consenrichRuntime._resolveRuntimeBackgroundBlockLen(
                400, 0, 50, multiplier, lengthScaleProvided=False,
            )
    for configuredBP in (True, False, 1.5, "100"):
        with pytest.raises(ValueError, match="backgroundBlockSizeBP"):
            consenrichRuntime._resolveRuntimeBackgroundBlockLen(
                1, configuredBP, 50, 8.0,
                lengthScaleProvided=False,
            )

    residualMatrix = np.ones((1, 64), dtype=np.float32)
    invVarMatrix = np.full((1, 64), 8.53348, dtype=np.float32)

    for span, smoothness in ((401, 64.0), (1771, 64.0), (2499, 64.0), (6427, 128.0)):
        fitted = core.solveZeroCenteredBackground(
            residualMatrix=residualMatrix,
            invVarMatrix=invVarMatrix,
            blockLenIntervals=span,
            backgroundSmoothness=smoothness,
            zeroCenter=False,
            useNonnegative=False,
        )
        assert fitted.dtype == np.float32
        np.testing.assert_allclose(fitted, 1.0, atol=1.0e-7, rtol=0.0)

    with pytest.raises(RuntimeError, match="rank deficient"):
        cconsenrich.csolveZeroCenteredBackground(
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            0.0,
            False,
            lamFirst=0.0,
        )

    singleton = cconsenrich.csolveZeroCenteredBackground(
        np.asarray([2.0], dtype=np.float64),
        np.asarray([8.0], dtype=np.float64),
        9.0,
        False,
        lamFirst=6.0,
    )
    np.testing.assert_allclose(singleton, np.asarray([4.0]))
    empty = cconsenrich.csolveZeroCenteredBackground(np.empty(0), np.empty(0), 0.0, False)
    assert empty.shape == (0,)


@pytest.mark.parametrize(
    ("span", "smoothness", "expectedPenalties"),
    ((3, 2.0, (4.5, 10.125)), (1, 64.0, (16.0, 4.0))),
)
def test_background_difference_penalties_match_sparse_reference(span, smoothness, expectedPenalties):
    intervalCount = 7
    weightTrack = np.asarray(
        [0.7, 1.4, 2.2, 0.9, 3.1, 1.8, 0.6],
        dtype=np.float64,
    )
    rhsTrack = np.asarray(
        [0.3, -1.1, 2.4, 0.8, -0.7, 1.6, -0.2],
        dtype=np.float64,
    )
    lamFirst, lamSecond = core._backgroundPenaltyWeightsFromSpan(span, smoothness)
    assert (lamFirst, lamSecond) == expectedPenalties
    firstPenalty = core._buildFirstDiffPenalty(intervalCount)
    secondPenalty = core._buildSecondDiffPenalty(intervalCount)
    systemMat = core.sparse.diags(
        weightTrack,
        offsets=0,
        format="csc",
    )
    systemMat += lamFirst * firstPenalty + lamSecond * secondPenalty

    for zeroCenter in (False, True):
        actual = core.solveZeroCenteredBackground(
            residualMatrix=np.zeros((1, intervalCount), dtype=np.float32),
            invVarMatrix=np.ones((1, intervalCount), dtype=np.float32),
            blockLenIntervals=span,
            backgroundSmoothness=smoothness,
            zeroCenter=zeroCenter,
            useNonnegative=False,
            weightTrack=weightTrack,
            rhsTrack=rhsTrack,
        )
        if zeroCenter:
            constraint = np.ones(intervalCount, dtype=np.float64)
            kktMat = core.sparse.bmat(
                [
                    [systemMat, constraint[:, None]],
                    [constraint[None, :], None],
                ],
                format="csc",
            )
            expected = core.sparse_linalg.spsolve(
                kktMat,
                np.concatenate([rhsTrack, np.zeros(1, dtype=np.float64)]),
            )[:-1]
        else:
            expected = core.sparse_linalg.spsolve(systemMat, rhsTrack)
        np.testing.assert_allclose(actual, expected, atol=1.0e-4)

    np.testing.assert_allclose(
        core._backgroundPenaltyDiagonal(intervalCount, lamFirst, lamSecond),
        systemMat.diagonal() - weightTrack,
    )
    backgroundTarget = np.linspace(-0.4, 0.7, intervalCount, dtype=np.float64)
    total, first, second = core._backgroundObjectivePenalty(
        background=backgroundTarget,
        blockLenIntervals=span,
        backgroundSmoothness=smoothness,
    )
    np.testing.assert_allclose(
        total - first - second,
        0.0,
        atol=1.0e-12,
    )






def test_process_precision_auto_min_rejects_too_small_max():
    assert core._processKappaConvexityLowerBound(
        robustTNu=4.0,
        stateDim=2,
    ) == pytest.approx(0.7501)
    assert core._processKappaConvexityLowerBound(
        robustTNu=16.0,
        stateDim=2,
    ) == pytest.approx(0.5626)
    with pytest.raises(ValueError, match="auto .*convexity-preserving lower bound"):
        core._checkProcessPrecisionMultiplierBounds(
            minValue=-1.0,
            maxValue=0.5,
            robustTNu=8.0,
            stateDim=2,
        )


def test_localSumsRetainSmallTotals():
    samples = (
        [1.0e16, 1.0, 1.0e-12, 2.0, 0.0, 3.0e-8, 4.0, 1.0e-12],
        [1.0e16, 1.0, -1.0e16, 2.0, -2.0, 1.0e-12, 0.0, -1.0e-12],
    )
    for dtype in (np.float32, np.float64):
        for sample in samples:
            storage = np.empty(2 * len(sample), dtype=dtype)
            storage[::2] = sample
            values = storage[::2]
            values.flags.writeable = False
            size = values.size
            for width in (1, 2, 3, 5, size, size + 7):
                effectiveWidth = min(width, size)
                for shiftEdges in (False, True):
                    for leftOffset in (-1, effectiveWidth // 2):
                        offset = leftOffset if leftOffset >= 0 else (
                            effectiveWidth // 2 if shiftEdges else (effectiveWidth - 1) // 2
                        )
                        expected = []
                        for index in range(size):
                            left = index - offset
                            if shiftEdges:
                                left = min(max(left, 0), size - effectiveWidth)
                            right = min(size, left + effectiveWidth)
                            expected.append(math.fsum(values[max(0, left):right]))
                        actual = cconsenrich.cWindowSums(
                            values, width, shiftEdges=shiftEdges, leftOffset=leftOffset
                        )
                        assert actual.dtype == np.float64
                        np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=0.0)
            starts = np.asarray([4, 0, 2, size, size - 2], dtype=np.int64)
            ends = np.asarray([size, 4, 2, size, size], dtype=np.int64)
            expected = [math.fsum(values[left:right]) for left, right in zip(starts, ends)]
            np.testing.assert_allclose(
                cconsenrich.cRangeSums(values, starts, ends), expected, rtol=2.0e-15, atol=0.0
            )

    values = np.full(8192, 1.0e-12, dtype=np.float32)
    values[:4096] = 4.0
    np.testing.assert_array_equal(
        cconsenrich.cWindowSums(values, 3)[4098:-2],
        np.full(values.size - 4100, 3.0 * float(values[-1])),
    )
    assert cconsenrich.cWindowSums([], 3).size == 0
    assert cconsenrich.cRangeSums([], [], []).size == 0
    for width in (0, -1):
        with pytest.raises(ValueError):
            cconsenrich.cWindowSums(values, width)
    for args, kwargs in ((([[1.0]], 1), {}), ((values, 3), {"leftOffset": 3})):
        with pytest.raises(ValueError):
            cconsenrich.cWindowSums(*args, **kwargs)
    for starts, ends in (([-1], [1]), ([0], [values.size + 1]), ([2], [1]),
                         ([0, 1], [1]), ([[0]], [[1]]), ([0.5], [1])):
        with pytest.raises((TypeError, ValueError)):
            cconsenrich.cRangeSums(values, starts, ends)

    values = np.full(301, 1.0e-5)
    values[0] = 1.0e12
    for function in (cconsenrich.cMovingAverageSame, cconsenrich.cmovingAverageSame):
        for width in (2, 3, 256, 257):
            offset = (
                width // 2 if function is cconsenrich.cmovingAverageSame and width <= 256
                else (width - 1) // 2
            )
            expected = [
                math.fsum(values[max(0, index - offset):min(values.size, index - offset + width)])
                / width
                for index in range(values.size)
            ]
            np.testing.assert_allclose(function(values, width), expected, rtol=2.0e-15, atol=0.0)

    result = cconsenrich.cMultiscaleCandidateSegmentStats(
        [1.0e12, 0.0, 1.0e-5, 1.0e-5, 0.0], [1], [0.0], [1.0]
    )
    np.testing.assert_array_equal(result[0], [0, 2])
    np.testing.assert_array_equal(result[1], [0, 3])
    np.testing.assert_array_equal(result[5], [1.0e12, 2.0e-5])




def test_munc_count_model_variance_is_lower_bound():
    totalVariance = np.asarray([0.02, 0.20, 0.03], dtype=np.float32)
    count_noise = np.asarray([0.10, 0.05, np.nan], dtype=np.float32)

    combined = core.applyMuncCountModelVarianceFloor(
        totalVariance,
        count_noise,
        varianceFloor=1.0e-6,
    )

    assert combined[0] == pytest.approx(0.10)
    assert combined[1] == pytest.approx(0.20)
    assert combined[2] == pytest.approx(0.03)



def _emaReference(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    if x.dtype == np.float32:
        alpha_ = np.float32(alpha)
        oneMinusAlpha = np.float32(1.0) - alpha_
    else:
        alpha_ = float(alpha)
        oneMinusAlpha = 1.0 - alpha_

    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha_ * x[i] + oneMinusAlpha * out[i - 1]
    for i in range(x.shape[0] - 2, -1, -1):
        out[i] = alpha_ * out[i] + oneMinusAlpha * out[i + 1]
    return out


def _levelKalmanReference(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    *,
    levelQ: float,
    stateInit: float,
    stateCovarInit: float,
    pad: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    data = np.asarray(matrixData, dtype=np.float64)
    munc = np.asarray(matrixMunc, dtype=np.float64)
    trackCount, intervalCount = data.shape
    stateForward = np.empty((intervalCount, 1), dtype=np.float64)
    covForward = np.empty((intervalCount, 1, 1), dtype=np.float64)
    pNoiseForward = np.empty((max(intervalCount, 1), 1, 1), dtype=np.float64)

    x = float(stateInit)
    p = float(stateCovarInit)
    for k in range(intervalCount):
        p += float(levelQ)
        sumInvR = 0.0
        sumInvRInnov = 0.0
        for j in range(trackCount):
            r = max(float(munc[j, k]) + float(pad), 1.0e-12)
            invR = 1.0 / r
            sumInvR += invR
            sumInvRInnov += invR * (float(data[j, k]) - x)
        innovScale = 1.0 + p * sumInvR
        x += p * sumInvRInnov / innovScale
        p /= innovScale
        stateForward[k, 0] = x
        covForward[k, 0, 0] = p
        if k > 0:
            pNoiseForward[k - 1, 0, 0] = float(levelQ)

    stateSmoothed = np.empty_like(stateForward)
    covSmoothed = np.empty_like(covForward)
    lagCovSmoothed = np.empty((max(intervalCount - 1, 1), 1, 1), dtype=np.float64)
    stateSmoothed[-1, 0] = stateForward[-1, 0]
    covSmoothed[-1, 0, 0] = covForward[-1, 0, 0]
    for k in range(intervalCount - 2, -1, -1):
        pf = covForward[k, 0, 0]
        pPred = max(pf + pNoiseForward[k, 0, 0], 1.0e-12)
        gain = pf / pPred
        stateSmoothed[k, 0] = stateForward[k, 0] + gain * (
            stateSmoothed[k + 1, 0] - stateForward[k, 0]
        )
        covSmoothed[k, 0, 0] = max(
            pf + gain * gain * (covSmoothed[k + 1, 0, 0] - pPred),
            0.0,
        )
        lagCovSmoothed[k, 0, 0] = pf + gain * (covSmoothed[k + 1, 0, 0] - pPred)

    residuals = data.T - stateSmoothed[:, 0:1]
    return (
        stateForward,
        covForward,
        pNoiseForward,
        stateSmoothed,
        covSmoothed,
        lagCovSmoothed,
        residuals,
    )


def test_munc_eb_prior_g_uncertainty_modes_are_limited():
    params = dict(
        default=constants.OBSERVATION_DEFAULT_MUNC_EB_PRIOR_G_UNCERTAINTY_MODE,
        supported=constants.MUNC_SUPPORTED_EB_PRIOR_G_UNCERTAINTY_MODES,
        paramName="observationParams.muncEBPrior.gUncertaintyMode",
    )
    for value in (None, "proxy", "disabled"):
        assert normalization.validateChoice(value, **params) == (
            "proxy" if value is None else value
        )
    for value in ("diagonal", "exact"):
        with pytest.raises(ValueError, match="gUncertaintyMode"):
            normalization.validateChoice(value, **params)


@pytest.mark.parametrize(
    "qKwargs, expectedMessage",
    (
        ({"minQ": 2.0e-4, "maxQ": 1.0}, "minQ"),
        ({"minQ": 1.0e-6, "maxQ": 5.0e-5}, "maxQ"),
    ),
)
def test_core_fixed_process_q_respects_q_bounds(qKwargs, expectedMessage):
    matrixData = np.zeros((2, 4), dtype=np.float32)
    matrixMunc = np.ones((2, 4), dtype=np.float32)
    kwargs = {
        "deltaF": 0.1,
        "stateInit": 0.0,
        "stateCovarInit": 1.0,
        "boundState": False,
        "stateLowerBound": 0.0,
        "stateUpperBound": 0.0,
        "blockLenIntervals": 2,
        "ECM_fixedBackgroundIters": 1,
        "processNoiseCalibration": core.PROCESS_NOISE_CALIBRATION_FIXED,
    }
    kwargs.update(qKwargs)

    with pytest.raises(ValueError, match=expectedMessage):
        core.runConsenrich(matrixData, matrixMunc, **kwargs)


@pytest.mark.parametrize(
    "scaleObs,scaleProcess",
    [(True, False), (False, True)],
)
def test_run_consenrich_passes_precision_options_to_fixed_background_ecm(
    monkeypatch,
    scaleObs,
    scaleProcess,
):
    capturedKwargs = []
    capturedWarmStarts = []

    def fakeCFixedBackgroundECM(**kwargs):
        capturedKwargs.append(
            (
                kwargs["t_innerIters"],
                kwargs["ECM_scaleObsPrecisionToMedian"],
                kwargs["ECM_scaleProcessPrecisionToMedian"],
                kwargs["obsPrecisionWarmStartIsMedianScaled"],
                kwargs["processPrecisionWarmStartIsMedianScaled"],
                kwargs["ECM_robustTNu"],
                kwargs["ECM_processRobustTNu"],
            )
        )
        capturedWarmStarts.append(
            (
                np.asarray(kwargs["lambdaExpInit"], dtype=np.float32).copy(),
                np.asarray(kwargs["processPrecExpInit"], dtype=np.float32).copy(),
            )
        )
        matrixData = np.asarray(kwargs["matrixData"], dtype=np.float32)
        trackCount, intervalCount = matrixData.shape
        stateDim = 2
        return (
            1,
            0.0,
            np.zeros((intervalCount, stateDim), dtype=np.float32),
            np.zeros((intervalCount, stateDim, stateDim), dtype=np.float32),
            np.zeros((intervalCount - 1, stateDim, stateDim), dtype=np.float32),
            np.zeros((intervalCount, trackCount), dtype=np.float32),
            np.ones(intervalCount, dtype=np.float32),
            np.ones(intervalCount, dtype=np.float32),
            {"converged": True},
        )

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", fakeCFixedBackgroundECM)
    initialLambda = np.asarray([0.1, 0.2, 0.4, 0.8, 1.6], dtype=np.float32)
    initialKappa = np.asarray([999.0, 0.1, 0.2, 0.4, 4.0], dtype=np.float32)
    initialLambdaCopy = initialLambda.copy()
    initialKappaCopy = initialKappa.copy()

    runKwargs = {
        "deltaF": 0.1,
        "minQ": 1.0e-4,
        "maxQ": 1.0,
        "stateInit": 0.0,
        "stateCovarInit": 1.0,
        "boundState": False,
        "stateLowerBound": 0.0,
        "stateUpperBound": 0.0,
        "blockLenIntervals": 2,
        "ECM_fixedBackgroundIters": 1,
        "ECM_outerIters": 1,
        "ECM_minOuterIters": 1,
        "ECM_robustTNu": 5.0,
        "ECM_processRobustTNu": 11.0,
        "ECM_useObsPrecisionReweighting": True,
        "ECM_useProcessPrecisionReweighting": True,
        "processPrecisionMultiplierMin": 1.0e-4,
        "processPrecisionMultiplierMax": 10.0,
        "ECM_scaleObsPrecisionToMedian": scaleObs,
        "ECM_scaleProcessPrecisionToMedian": scaleProcess,
        "initialObservationPrecision": initialLambda,
        "initialProcessPrecision": initialKappa,
        "returnPrecisionDiagnostics": True,
        "fitBackground": False,
        "processNoiseCalibration": core.PROCESS_NOISE_CALIBRATION_FIXED,
    }
    with pytest.raises(ValueError, match="t_innerIters"):
        core.runConsenrich(
            np.zeros((2, 5), dtype=np.float32),
            np.ones((2, 5), dtype=np.float32),
            **runKwargs,
            t_innerIters=1.5,
        )

    out = core.runConsenrich(
        np.zeros((2, 5), dtype=np.float32),
        np.ones((2, 5), dtype=np.float32),
        **runKwargs,
        t_innerIters=4,
    )

    assert capturedKwargs == [
        (4, scaleObs, scaleProcess, scaleObs, scaleProcess, 5.0, 11.0)
    ]
    lambdaWarmStart, kappaWarmStart = capturedWarmStarts[0]
    if scaleObs:
        np.testing.assert_allclose(
            lambdaWarmStart,
            [0.25, 0.5, 1.0, 2.0, 4.0],
        )
    if scaleProcess:
        np.testing.assert_allclose(
            kappaWarmStart,
            [1.0, 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 10.0],
            rtol=2.0e-6,
        )
    np.testing.assert_array_equal(initialLambda, initialLambdaCopy)
    np.testing.assert_array_equal(initialKappa, initialKappaCopy)
    precisionDiagnostics = out[-1]
    assert precisionDiagnostics["scale_obs_precision_to_median"] is scaleObs
    assert precisionDiagnostics["scale_process_precision_to_median"] is scaleProcess
    assert precisionDiagnostics["observation_robust_t_nu"] == pytest.approx(5.0)
    assert precisionDiagnostics["process_robust_t_nu"] == pytest.approx(11.0)


@pytest.mark.parametrize(
    "precisionKwargs,kernelPrecisionKwargs",
    [
        (
            {
                "ECM_useObsPrecisionReweighting": True,
                "ECM_scaleObsPrecisionToMedian": True,
                "observationPrecisionMultiplierMin": 1.1,
                "observationPrecisionMultiplierMax": 2.0,
            },
            {
                "ECM_useObsPrecisionReweighting": True,
                "ECM_scaleObsPrecisionToMedian": True,
                "obsPrecisionMultiplierMin": 1.1,
                "obsPrecisionMultiplierMax": 2.0,
            },
        ),
        (
            {
                "ECM_scaleProcessPrecisionToMedian": True,
                "processPrecisionMultiplierMin": 0.2,
                "processPrecisionMultiplierMax": 0.9,
            },
            {
                "ECM_scaleProcessPrecisionToMedian": True,
                "procPrecisionMultiplierMin": 0.2,
                "procPrecisionMultiplierMax": 0.9,
            },
        ),
    ],
)
def test_run_consenrich_requires_median_scaling_bounds_to_contain_one(
    precisionKwargs,
    kernelPrecisionKwargs,
):
    with pytest.raises(ValueError, match="bounds must contain 1"):
        core.runConsenrich(
            np.zeros((1, 6), dtype=np.float32),
            np.ones((1, 6), dtype=np.float32),
            deltaF=0.1,
            minQ=1.0e-4,
            maxQ=0.5,
            stateInit=0.0,
            stateCovarInit=1.0,
            boundState=False,
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            blockLenIntervals=3,
            **precisionKwargs,
        )
    with pytest.raises(ValueError, match="bounds must contain 1"):
        cconsenrich.cfixedBackgroundECMLevel(
            matrixData=np.zeros((1, 5), dtype=np.float32),
            matrixPluginMuncInit=np.ones((1, 5), dtype=np.float32),
            matrixQ0=np.asarray([[0.1]], dtype=np.float32),
            intervalToBlockMap=np.zeros(5, dtype=np.int32),
            blockCount=1,
            stateInit=0.0,
            stateCovarInit=1.0,
            logIterations=False,
            **kernelPrecisionKwargs,
        )


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _caseCEMAUsesSameBidirectionalKernelForFloat32AndFloat64(dtype):
    x = np.array([0.0, 2.0, -1.0, 5.0, 4.0, 7.0], dtype=dtype)
    alpha = 0.35

    out = cconsenrich.cEMA(x, alpha)
    expected = _emaReference(x, alpha)

    assert out.dtype == dtype
    if dtype == np.float32:
        np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-7)
    else:
        np.testing.assert_allclose(out, expected, rtol=1.0e-14, atol=1.0e-14)
@pytest.mark.correctness
def _caseMuncObservationMomentSeedPassUsesOmegaMomentsAndFloors():
    matrixData = np.asarray(
        [[1.0, 2.0, 4.0], [1.5, 1.0, 5.0]],
        dtype=np.float32,
    )
    matrixMunc = np.asarray(
        [[0.4, 0.6, 0.8], [0.5, 0.7, 0.9]],
        dtype=np.float32,
    )
    stateMean = np.asarray([1.25, 1.5, 3.0], dtype=np.float32)
    stateVariance = np.asarray([0.05, 0.10, 0.20], dtype=np.float32)
    gVariance = np.asarray([0.00, 0.20, 0.40], dtype=np.float32)
    countFloor = np.asarray(
        [[0.01, 0.10, 0.20], [0.02, 0.15, 0.25]],
        dtype=np.float32,
    )
    background = np.asarray([0.05, -0.10, 0.20], dtype=np.float32)
    pad = 0.05
    nu = 5.0
    omegaMin = 0.10
    omegaMax = 0.80
    varianceFloor = 1.0e-4
    varianceCap = 2.0

    def expectedArrays(enabled, studentT, useG, useCountFloor):
        gTrack = gVariance.astype(np.float64) if useG else np.zeros(stateMean.size)
        moment = (
            np.square(
                matrixData.astype(np.float64)
                - background.astype(np.float64)[None, :]
                - stateMean[None, :]
            )
            + stateVariance.astype(np.float64)[None, :]
            + gTrack[None, :]
        )
        count = countFloor.astype(np.float64) if useCountFloor else np.zeros_like(moment)
        baseVariance = matrixMunc.astype(np.float64) + pad
        if enabled and studentT:
            rho = (nu + 1.0) / (nu + moment / baseVariance)
            dbar = np.mean(moment / baseVariance, axis=0)
            omegaRaw = (nu + 1.0) / (nu + dbar)
            omega = np.clip(omegaRaw, omegaMin, omegaMax)
            local = omega[None, :] * rho * moment - pad - count
        else:
            rho = np.ones(matrixData.shape, dtype=np.float64)
            omegaRaw = np.ones(stateMean.size, dtype=np.float64)
            omega = np.ones(stateMean.size, dtype=np.float64)
            local = moment - pad - count
        total = local + count
        local = np.clip(local, varianceFloor, varianceCap)
        total = local + count
        total = np.clip(total, varianceFloor, varianceCap)
        return moment, rho, omegaRaw, omega, local, total

    out = cconsenrich.cMuncObservationMomentSeedPass(
        matrixData,
        matrixMunc,
        stateMean,
        stateVariance,
        background=background,
        gVariance=gVariance,
        countFloor=countFloor,
        pad=pad,
        studentTdf=nu,
        useSeedWeights=True,
        updateWeights=True,
        omegaMin=omegaMin,
        omegaMax=omegaMax,
        varianceFloor=varianceFloor,
        varianceCap=varianceCap,
        enabled=True,
        studentT=True,
        dOmega=nu,
    )
    for observed, expected in zip(
        out,
        expectedArrays(True, True, True, True),
    ):
        np.testing.assert_allclose(observed, expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(out[5], dtype=np.float64),
        np.asarray(out[4], dtype=np.float64) + countFloor,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    for branchKwargs, branchExpected in (
        ({"enabled": True, "studentT": False}, expectedArrays(True, False, False, False)),
        ({"enabled": False, "studentT": True}, expectedArrays(False, True, False, False)),
    ):
        branchOut = cconsenrich.cMuncObservationMomentSeedPass(
            matrixData,
            matrixMunc,
            stateMean,
            stateVariance,
            background=background,
            pad=pad,
            studentTdf=nu,
            useSeedWeights=True,
            updateWeights=True,
            omegaMin=omegaMin,
            omegaMax=omegaMax,
            varianceFloor=varianceFloor,
            varianceCap=varianceCap,
            dOmega=nu,
            **branchKwargs,
        )
        for observed, expected in zip(branchOut, branchExpected):
            np.testing.assert_allclose(observed, expected, rtol=1.0e-6, atol=1.0e-6)

    smallData = np.full((1, 1), 1.0e-5, dtype=np.float32)
    smallFloor = np.full((1, 1), 0.01, dtype=np.float32)
    smallMoment = float(smallData[0, 0]) ** 2 + float(smallFloor[0, 0])
    for studentT, updateWeights in ((False, True), (True, False), (True, True)):
        weighted = (9.0 / (8.0 + smallMoment / float(smallFloor[0, 0]))) ** 2 if studentT and updateWeights else 1.0
        result = cconsenrich.cMuncObservationMomentSeedPass(
            smallData, smallFloor, np.zeros(1, dtype=np.float32), smallFloor[0],
            countFloor=smallFloor, pad=0.0, studentTdf=8.0, dOmega=8.0,
            studentT=studentT, updateWeights=updateWeights,
            rhoIn=np.ones((1, 1), dtype=np.float32),
        )
        expectedLocal = weighted * smallMoment - float(smallFloor[0, 0])
        np.testing.assert_allclose(result[4], expectedLocal, rtol=1.0e-7, atol=0.0)


@pytest.mark.correctness
def _caseFinalizeMuncEBTrackPreservesCountFloorSentinel():
    local = np.asarray([0.05, 0.20, 4.00, 0.001], dtype=np.float64)
    prior = np.asarray([0.35, 0.60, 10.00, 0.02], dtype=np.float32)
    countFloor = np.asarray([np.nan, 0.25, 2.00, 0.00], dtype=np.float32)

    out, diagnostics = cconsenrich.cFinalizeMuncEBTrack(
        local,
        priorVarianceTrack=prior,
        countFloor=countFloor,
        nuLocal=2.0,
        nuPrior=3.0,
        useEB=True,
        varianceFloor=0.01,
        varianceCap=5.0,
    )

    expected = np.asarray([0.23, 0.69, 5.0, 0.016], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)
    assert diagnostics["supportCount"] == 3
    assert diagnostics["supportFraction"] == pytest.approx(0.75)
    assert diagnostics["countFloorFiniteCount"] == 3
    assert diagnostics["countFloorAddedCount"] == 2
    assert diagnostics["countFloorMissingCount"] == 1
    assert diagnostics["finalShrinkagePairCount"] == 4
    assert diagnostics["finalShrinkagePairFraction"] == pytest.approx(1.0)

    noEBOut, noEBDiagnostics = cconsenrich.cFinalizeMuncEBTrack(
        local,
        countFloor=countFloor,
        useEB=False,
        varianceFloor=0.01,
        varianceCap=5.0,
    )
    np.testing.assert_allclose(
        noEBOut,
        np.asarray([0.05, 0.45, 5.0, 0.01], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert noEBDiagnostics["finalShrinkagePairCount"] == 0

    with pytest.raises(ValueError, match="countFloor"):
        cconsenrich.cFinalizeMuncEBTrack(
            local,
            priorVarianceTrack=prior,
            countFloor=np.asarray([0.0, np.inf, 0.0, 0.0], dtype=np.float32),
            nuLocal=2.0,
            nuPrior=3.0,
            varianceFloor=0.01,
            varianceCap=5.0,
        )


@pytest.mark.correctness
def _caseMuncSmoothDenseLocalEvidenceUsesCenteredWindows():
    localEvidence = np.asarray(
        [[1.0, 100.0, 3.0, 5.0, 7.0], [2.0, 4.0, 8.0, 16.0, 32.0]],
        dtype=np.float32,
    )
    excludeMask = np.asarray([0, 1, 0, 0, 0], dtype=np.uint8)
    observed = cconsenrich.cMuncSmoothDenseLocalEvidence(
        localEvidence,
        3,
        excludeMask=excludeMask,
        eps=1.0e-4,
    )
    expected = np.asarray(
        [
            [2.0, 2.0, 4.0, 5.0, 5.0],
            [5.0, 5.0, 12.0, 56.0 / 3.0, 56.0 / 3.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(observed, expected, rtol=1.0e-6, atol=1.0e-6)

    floor = np.float32(1.0e-12)
    tinyExcess = np.nextafter(np.nextafter(floor, np.float32(np.inf)), np.float32(np.inf))
    intervalCount = 16384
    localEvidence = np.full((2, intervalCount), floor, dtype=np.float32)
    localEvidence[1, :] = tinyExcess
    localEvidence[0, 53::331] = np.float32(987654321.0)
    localEvidence[1, 71::283] = np.float32(123456789.0)
    localEvidence[1, 5000:5300] = floor / np.float32(4.0)
    excludeMask = np.zeros(intervalCount, dtype=np.uint8)
    excludeMask[2000:2250] = 1
    excludeMask[53::331] = 1
    rowExcludeMask = np.stack((excludeMask, np.roll(excludeMask, 127)))
    cases = [(localEvidence, width, excluded) for width, excluded in
             ((100, None), (101, excludeMask), (100, rowExcludeMask))]
    extreme = np.full((5, 65), 2.0 * floor, dtype=np.float32)
    scales = np.asarray([1.0e10, 1.0e20, 1.0e25, 1.0e30, 3.0e38], dtype=np.float32)
    extreme[:, ::17] = scales[:, None]
    extreme[:, 1::17] = (scales * np.float32(1.0e-15))[:, None]
    excluded = np.zeros_like(extreme, dtype=np.uint8)
    excluded[:, 20:35] = 1
    excluded[::2, ::17] = 1
    cases.extend((extreme, width, excludedCells) for width in (1, 2, 3, 7, 16, 65, 70)
                 for excludedCells in (None, excluded[0], excluded))
    cases.extend((np.empty(shape, dtype=np.float32), 3, None) for shape in ((2, 0), (0, 65)))
    for evidence, windowIntervals, excluded in cases:
        intervalCount = evidence.shape[1]
        allowed = np.ones_like(evidence, dtype=bool) if excluded is None else np.broadcast_to(
            excluded == 0, evidence.shape
        )
        expected = np.empty_like(evidence)
        for row in range(evidence.shape[0]):
            for index in range(intervalCount):
                left = min(max(0, index - windowIntervals // 2), max(0, intervalCount - windowIntervals))
                right = min(intervalCount, left + windowIntervals)
                values = evidence[row, left:right][allowed[row, left:right]]
                expected[row, index] = max(
                    float(floor),
                    math.fsum(values) / values.size if values.size else float(evidence[row, index]),
                )
        actual = cconsenrich.cMuncSmoothDenseLocalEvidence(
            evidence, windowIntervals, excludeMask=excluded, eps=float(floor)
        )
        np.testing.assert_array_equal(actual, expected)



















def test_stableTransformsAndWeightedQuantiles():
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 90
        values = np.arange(1001, dtype=np.float64)
        weights = np.concatenate(([1e16], np.ones(1000)))
        q = 0.99999999999995
        target = sum(Decimal(float(w)) for w in weights) * Decimal(q)
        expected = float((target - Decimal(1e16)).to_integral_value(rounding="ROUND_CEILING"))
        assert normalization.weightedQuantile(values, weights, q) == expected
        assert normalization.weightedQuantileInterpolated(values, weights, q) == pytest.approx(
            float(target - Decimal(1e16)), rel=1e-15,
        )
    for function in (normalization.weightedQuantile, normalization.weightedQuantileInterpolated):
        for scale in (1.0, 1e308, 1e-308):
            np.testing.assert_array_equal(function([0., 1.], [scale, scale], [[0., 0.5, 1.]]), [[0., 0., 1.]])
        np.testing.assert_array_equal(function([2., 0., 1.], [1., 1., 2.], [0., 0.25, 0.75, 1.]), [0., 0., 1., 2.])










def test_run_consenrich_default_c_stdout_stderr_silent(capfd):
    rng = np.random.default_rng(11)
    n = 18
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    matrixData = np.vstack(
        [
            grid + 0.01 * rng.normal(size=n),
            grid + 0.02 * rng.normal(size=n) + 0.05,
        ]
    ).astype(np.float32)
    matrixMunc = np.full_like(matrixData, 0.1, dtype=np.float32)

    core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-4,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=6,
        pad=1.0e-4,
        ECM_fixedBackgroundIters=1,
        ECM_fixedBackgroundRtol=0.0,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        ECM_backgroundShiftRtol=0.0,
        ECM_outerNLLRtol=0.0,
        fitBackground=False,
        processNoiseCalibration=core.PROCESS_NOISE_CALIBRATION_FIXED,
        returnScales=True,
    )

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""












@pytest.mark.correctness
def test_core_center_mb_applies_default_savgol_filter_in_place(monkeypatch):
    cases = (
        (
            np.array(
                [
                    [1.0, 2.0, 100.0, 4.0, 5.0],
                    [-8.0, -4.0, -2.0, 0.0, 3.0],
                ],
                dtype=np.float32,
            ),
            100_000,
            5,
        ),
        (
            np.array([2.0, 8.0, 1.0, 6.0], dtype=np.float64),
            50_000,
            11,
        ),
    )
    for tracks, intervalSizeBP, expectedWindow in cases:
        original = tracks.copy()

        stats_ = core.centerMBInPlace(
            tracks,
            intervalSizeBP=intervalSizeBP,
            filterWindowBP=500_000,
            centerMBMethod="medfilt",
        )

        originalTracks = original.reshape(1, -1) if original.ndim == 1 else original
        expectedFilter = ndi.median_filter(
            originalTracks,
            size=(1, expectedWindow),
            mode="nearest",
        )
        expectedCentered = originalTracks - expectedFilter
        centeredTracks = tracks.reshape(1, -1) if tracks.ndim == 1 else tracks

        assert stats_["applied"] is True
        assert stats_["applied_tracks"] == originalTracks.shape[0]
        assert stats_["meanTrackValBeforeCenterMB"] == pytest.approx(
            float(np.mean(originalTracks))
        )
        assert stats_["meanTrackValAfterCenterMB"] == pytest.approx(
            float(np.mean(expectedCentered))
        )
        assert stats_["stdTrackValBeforeCenterMB"] == pytest.approx(
            float(np.std(originalTracks))
        )
        assert stats_["stdTrackValAfterCenterMB"] == pytest.approx(
            float(np.std(expectedCentered))
        )
        np.testing.assert_allclose(centeredTracks, expectedCentered)
        assert not np.allclose(centeredTracks, originalTracks)

    medianCalls = []

    def captureMedianFilter(track, *, size, mode):
        medianCalls.append((track.copy(), size, mode))
        return np.zeros_like(track)

    productionBinTracks = np.array(
        [[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]],
        dtype=np.float32,
    )
    with monkeypatch.context() as mp:
        mp.setattr(core.ndimage, "median_filter", captureMedianFilter)
        core.centerMBInPlace(
            productionBinTracks,
            intervalSizeBP=50,
            filterWindowBP=500_000,
            centerMBMethod="medfilt",
        )

    assert len(medianCalls) == 2
    assert all(size == 10_001 for _, size, _ in medianCalls)
    assert all(mode == "nearest" for _, _, mode in medianCalls)
    np.testing.assert_array_equal(medianCalls[0][0], [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(medianCalls[1][0], [5.0, 4.0, 6.0])

    savgolTracks = np.array(
        [
            [1.0, 4.0, 12.0, 5.0, 2.0],
            [3.0, 3.0, 9.0, 3.0, 3.0],
        ],
        dtype=np.float64,
    )
    savgolOriginal = savgolTracks.copy()
    savgolStats = core.centerMBInPlace(
        savgolTracks,
        intervalSizeBP=250_000,
    )
    savgolFilter = spySig.savgol_filter(
        savgolOriginal,
        window_length=5,
        polyorder=0,
        mode="nearest",
        axis=1,
    )

    assert savgolStats["applied"] is True
    assert savgolStats["applied_tracks"] == 2
    assert savgolStats["meanTrackValBeforeCenterMB"] == pytest.approx(
        float(np.mean(savgolOriginal))
    )
    assert savgolStats["meanTrackValAfterCenterMB"] == pytest.approx(
        float(np.mean(savgolOriginal - savgolFilter))
    )
    assert savgolStats["stdTrackValBeforeCenterMB"] == pytest.approx(
        float(np.std(savgolOriginal))
    )
    assert savgolStats["stdTrackValAfterCenterMB"] == pytest.approx(
        float(np.std(savgolOriginal - savgolFilter))
    )
    np.testing.assert_allclose(
        savgolTracks,
        savgolOriginal - savgolFilter,
        atol=8.0 * np.finfo(np.float64).eps,
    )

    plateauTrack = np.full(8192, 1.0e-12, dtype=np.float64)
    plateauTrack[:4096] = 4.0
    core.centerMBInPlace(plateauTrack, intervalSizeBP=50, filterWindowBP=150)
    np.testing.assert_allclose(plateauTrack[4098:], 0.0, rtol=0.0, atol=1.0e-27)

    with pytest.raises(ValueError, match="centerMBMethod"):
        core.centerMBInPlace(
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            intervalSizeBP=500_000,
            centerMBMethod="mean",
        )


def _writeSyntheticBam(tmp_path: Path, fileName: str, records: list[dict]) -> Path:
    pysam = pytest.importorskip("pysam")
    bamPath = tmp_path / fileName
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1000}],
    }

    with pysam.AlignmentFile(str(bamPath), "wb", header=header) as bamHandle:
        for recordSpec in records:
            readLength = int(recordSpec.get("length", 20))
            record = pysam.AlignedSegment()
            record.query_name = str(recordSpec["name"])
            record.query_sequence = str(recordSpec.get("sequence", "A" * readLength))
            record.flag = int(recordSpec.get("flag", 0))
            record.reference_id = 0
            record.reference_start = int(recordSpec["start"])
            record.mapping_quality = int(recordSpec.get("mapq", 60))
            record.cigar = ((0, readLength),)
            record.next_reference_id = int(recordSpec.get("next_reference_id", -1))
            record.next_reference_start = int(recordSpec.get("next_start", -1))
            record.template_length = int(recordSpec.get("template_length", 0))
            record.query_qualities = pysam.qualitystring_to_array("I" * readLength)
            bamHandle.write(record)

    pysam.index(str(bamPath))
    return bamPath






@pytest.mark.peaks
def _caseMatchExistingBedGraph():
    np.random.seed(42)
    with tempfile.TemporaryDirectory() as tempFolder:
        stateBedGraphPath = Path(tempFolder) / "toy_state.bedGraph"
        uncertaintyBedGraphPath = Path(tempFolder) / "toy_uncertainty.bedGraph"
        fakeVals = []
        for i in range(1000):
            if (i % 100) <= 10:
                # add in about ~10~ peak regions
                fakeVals.append(max(np.random.poisson(lam=5), 1))
            else:
                # add in background poisson(1) for BG
                fakeVals.append(np.random.poisson(lam=1))

        fakeVals = np.array(fakeVals).astype(float)
        stateFrame = pd.DataFrame(
            {
                "chromosome": ["chr2"] * 1000,
                "start": list(range(0, 10_000, 10)),
                "end": list(range(10, 10_010, 10)),
                "value": spySig.fftconvolve(
                    fakeVals,
                    np.ones(10) / 10,  # smooth out over ~100bp~
                    mode="same",
                ),
            }
        )
        stateFrame = stateFrame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        uncertaintyFrame = stateFrame.copy()
        uncertaintyFrame["value"] = 1.0
        stateFrame.to_csv(stateBedGraphPath, sep="\t", header=False, index=False)
        uncertaintyFrame.to_csv(
            uncertaintyBedGraphPath, sep="\t", header=False, index=False
        )
        artifacts = peaks.solveRocco(
            str(stateBedGraphPath),
            100,
            500,
            uncertaintyBedGraphFile=str(uncertaintyBedGraphPath),
            peakMode="narrow",
            numBootstrap=32,
            thresholdZ=2.0,
            randSeed=42,
        )
        outputPath = artifacts.narrowPeak
        assert outputPath is not None
        assert os.path.isfile(outputPath)
        with open(outputPath, "r") as fileHandle:
            lineStrings = fileHandle.readlines()

        assert len(lineStrings) >= 1
        assert lineStrings[0].startswith("chr2")


@pytest.mark.correctness
def _caseZeroCenteredBackgroundUpdate():
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    residualMatrix = np.vstack(
        [
            0.6 * np.sin(np.linspace(0.0, np.pi, 64, dtype=np.float32)) + 0.1 * x,
            0.6 * np.sin(np.linspace(0.0, np.pi, 64, dtype=np.float32)) - 0.1 * x,
        ]
    ).astype(np.float32)
    invVarMatrix = np.ones_like(residualMatrix, dtype=np.float32)

    background = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )

    assert background.shape == (64,)
    assert np.isfinite(background).all()
    assert abs(float(np.mean(background))) < 1.0e-5
    assert float(np.std(background)) > 0.0

    weightTrack = np.sum(invVarMatrix.astype(np.float64), axis=0)
    rhsTrack = np.sum(
        invVarMatrix.astype(np.float64) * residualMatrix.astype(np.float64),
        axis=0,
    )
    lamFirst, lamSecond = core._backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lamFirst * core._buildFirstDiffPenalty(weightTrack.size))
    systemMat = systemMat + (lamSecond * core._buildSecondDiffPenalty(weightTrack.size))
    constraintVec = np.ones(weightTrack.size, dtype=np.float64)
    kktMat = core.sparse.bmat(
        [
            [systemMat, constraintVec[:, None]],
            [constraintVec[None, :], None],
        ],
        format="csc",
    )
    rhsVec = np.concatenate([rhsTrack, np.zeros(1, dtype=np.float64)])
    expected = core.sparse_linalg.spsolve(kktMat, rhsVec)[:-1]
    expected -= np.mean(expected, dtype=np.float64)

    assert np.allclose(background, expected.astype(np.float32), atol=1.0e-4)


@pytest.mark.correctness
def _caseZeroCenteredBackgroundUpdateUsesPrecisionWeights():
    n = 52
    x = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    residualMatrix = np.vstack(
        [
            0.35 * np.sin(x) + 0.15,
            -0.10 * np.cos(x) - 0.05,
            np.linspace(-0.2, 0.2, n, dtype=np.float32),
        ]
    ).astype(np.float32)
    invVarMatrix = np.vstack(
        [
            np.linspace(0.4, 2.0, n, dtype=np.float32),
            np.linspace(2.0, 0.6, n, dtype=np.float32),
            np.full(n, 0.9, dtype=np.float32),
        ]
    )

    background = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=7,
        backgroundSmoothness=1.3,
    )

    weightTrack = np.sum(invVarMatrix.astype(np.float64), axis=0)
    rhsTrack = np.sum(
        invVarMatrix.astype(np.float64) * residualMatrix.astype(np.float64),
        axis=0,
    )
    lamFirst, lamSecond = core._backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=7,
        backgroundSmoothness=1.3,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lamFirst * core._buildFirstDiffPenalty(weightTrack.size))
    systemMat = systemMat + (lamSecond * core._buildSecondDiffPenalty(weightTrack.size))
    constraintVec = np.ones(weightTrack.size, dtype=np.float64)
    kktMat = core.sparse.bmat(
        [
            [systemMat, constraintVec[:, None]],
            [constraintVec[None, :], None],
        ],
        format="csc",
    )
    rhsVec = np.concatenate([rhsTrack, np.zeros(1, dtype=np.float64)])
    expected = core.sparse_linalg.spsolve(kktMat, rhsVec)[:-1]

    unweightedRhs = np.sum(residualMatrix.astype(np.float64), axis=0)
    assert not np.allclose(rhsTrack, unweightedRhs, atol=1.0e-3)
    assert np.allclose(background, expected.astype(np.float32), atol=1.0e-4)


@pytest.mark.correctness
def _caseZeroCenteredBackgroundUpdateMatchesSparseReference():
    residualMatrix = np.vstack(
        [
            np.linspace(-0.25, 0.25, 96, dtype=np.float32),
            np.linspace(0.25, -0.25, 96, dtype=np.float32),
        ]
    )
    invVarMatrix = np.ones_like(residualMatrix, dtype=np.float32)

    background = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )

    assert background.shape == (96,)
    assert np.isfinite(background).all()
    assert abs(float(np.mean(background))) < 1.0e-5

    weightTrack = np.sum(invVarMatrix.astype(np.float64), axis=0)
    rhsTrack = np.sum(
        invVarMatrix.astype(np.float64) * residualMatrix.astype(np.float64),
        axis=0,
    )
    lamFirst, lamSecond = core._backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lamFirst * core._buildFirstDiffPenalty(weightTrack.size))
    systemMat = systemMat + (lamSecond * core._buildSecondDiffPenalty(weightTrack.size))
    constraintVec = np.ones(weightTrack.size, dtype=np.float64)
    kktMat = core.sparse.bmat(
        [[systemMat, constraintVec[:, None]], [constraintVec[None, :], None]],
        format="csc",
    )
    rhsVec = np.concatenate([rhsTrack, np.zeros(1, dtype=np.float64)])
    expected = core.sparse_linalg.spsolve(kktMat, rhsVec)[:-1]
    np.testing.assert_allclose(background, expected.astype(np.float32), atol=1.0e-5)


@pytest.mark.correctness
def _caseBackgroundUpdateCanSkipZeroCentering():
    n = 72
    x = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    residualMatrix = np.vstack(
        [
            0.75 + 0.2 * np.sin(x),
            0.75 + 0.1 * np.cos(x),
        ]
    ).astype(np.float32)
    invVarMatrix = np.ones_like(residualMatrix, dtype=np.float32)

    centered = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )
    uncentered = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
        zeroCenter=False,
    )

    weightTrack = np.sum(invVarMatrix.astype(np.float64), axis=0)
    rhsTrack = np.sum(
        invVarMatrix.astype(np.float64) * residualMatrix.astype(np.float64),
        axis=0,
    )
    lamFirst, lamSecond = core._backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lamFirst * core._buildFirstDiffPenalty(weightTrack.size))
    systemMat = systemMat + (lamSecond * core._buildSecondDiffPenalty(weightTrack.size))
    expectedUncentered = core.sparse_linalg.spsolve(systemMat.tocsc(), rhsTrack)

    assert abs(float(np.mean(centered))) < 1.0e-5
    assert abs(float(np.mean(uncentered))) > 0.5
    assert np.allclose(
        uncentered,
        expectedUncentered.astype(np.float32),
        atol=1.0e-4,
    )


@pytest.mark.correctness
def _caseBackgroundUpdateCanEnforceNonnegativeConstraint():
    n = 80
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    residualMatrix = np.vstack(
        [
            -0.45 + 1.25 * np.exp(-((x - 0.15) ** 2) / 0.04),
            -0.35 + 1.00 * np.exp(-((x + 0.10) ** 2) / 0.06),
        ]
    ).astype(np.float32)
    invVarMatrix = np.ones_like(residualMatrix, dtype=np.float32)

    unconstrained = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=4,
        backgroundSmoothness=0.5,
        zeroCenter=False,
        useNonnegative=False,
    )
    constrained = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=4,
        backgroundSmoothness=0.5,
        zeroCenter=False,
        useNonnegative=True,
    )
    disabled = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=4,
        backgroundSmoothness=0.5,
        zeroCenter=False,
        useNonnegative=True,
        backgroundNegativePenaltyMultiplier=None,
    )

    def negativeEnergy(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        return float(np.sum(np.minimum(arr, 0.0) ** 2, dtype=np.float64))

    assert float(np.min(unconstrained)) < 0.0
    assert np.isfinite(constrained).all()
    assert float(np.max(constrained)) > 0.0
    assert not np.allclose(constrained, unconstrained)
    assert negativeEnergy(constrained) < negativeEnergy(unconstrained)
    np.testing.assert_allclose(disabled, unconstrained)

    stiffX = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    stiffResidualMatrix = np.vstack(
        [
            -0.02 + 0.08 * np.exp(-((stiffX - 0.2) ** 2) / 0.08),
            -0.018 + 0.07 * np.exp(-((stiffX + 0.1) ** 2) / 0.09),
        ]
    ).astype(np.float32)
    stiffInvVarMatrix = np.ones_like(stiffResidualMatrix, dtype=np.float32)
    stiffUnconstrained = core.solveZeroCenteredBackground(
        residualMatrix=stiffResidualMatrix,
        invVarMatrix=stiffInvVarMatrix,
        blockLenIntervals=80,
        backgroundSmoothness=1.0,
        zeroCenter=False,
        useNonnegative=False,
    ).astype(np.float64)
    stiffClipped = np.maximum(stiffUnconstrained, 0.0)
    stiffConstrained = core.solveZeroCenteredBackground(
        residualMatrix=stiffResidualMatrix,
        invVarMatrix=stiffInvVarMatrix,
        blockLenIntervals=80,
        backgroundSmoothness=1.0,
        zeroCenter=False,
        useNonnegative=True,
    ).astype(np.float64)

    assert float(np.min(stiffUnconstrained)) < 0.0
    assert np.isfinite(stiffConstrained).all()
    assert negativeEnergy(stiffConstrained) < negativeEnergy(stiffUnconstrained)
    assert not np.allclose(stiffConstrained, stiffClipped)


@pytest.mark.correctness
def _caseBackgroundUpdateReusesStatsAndInitialActiveSet():
    n = 90
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    residualMatrix = np.vstack(
        [
            -0.15 + 0.9 * np.exp(-((x - 0.2) ** 2) / 0.05),
            -0.10 + 0.7 * np.exp(-((x + 0.25) ** 2) / 0.08),
            0.05 * np.sin(np.linspace(0.0, 3.0 * np.pi, n, dtype=np.float32)),
        ]
    ).astype(np.float32)
    invVarMatrix = np.vstack(
        [
            np.linspace(0.7, 1.8, n, dtype=np.float32),
            np.linspace(1.4, 0.6, n, dtype=np.float32),
            np.full(n, 1.1, dtype=np.float32),
        ]
    )
    weightTrack = np.sum(invVarMatrix.astype(np.float64), axis=0)
    rhsTrack = np.sum(
        invVarMatrix.astype(np.float64) * residualMatrix.astype(np.float64),
        axis=0,
    )
    nativeWeightTrack, nativeRhsTrack, nativeSupportCount = (
        cconsenrich.cbackgroundWeightedStatsWithSupport(
            residualMatrix,
            invVarMatrix,
        )
    )
    np.testing.assert_allclose(nativeWeightTrack, weightTrack, rtol=1.0e-7)
    np.testing.assert_allclose(nativeRhsTrack, rhsTrack, rtol=1.0e-7)
    assert nativeSupportCount == n

    for useNonnegative in (False, True):
        plain = core.solveZeroCenteredBackground(
            residualMatrix=residualMatrix,
            invVarMatrix=invVarMatrix,
            blockLenIntervals=5,
            backgroundSmoothness=0.8,
            zeroCenter=False,
            useNonnegative=useNonnegative,
        )
        reused = core.solveZeroCenteredBackground(
            residualMatrix=residualMatrix,
            invVarMatrix=invVarMatrix,
            blockLenIntervals=5,
            backgroundSmoothness=0.8,
            zeroCenter=False,
            useNonnegative=useNonnegative,
            initialBackground=plain,
            weightTrack=weightTrack,
            rhsTrack=rhsTrack,
        )
        np.testing.assert_allclose(reused, plain, atol=1.0e-5)

    zeroWeights = np.zeros_like(invVarMatrix, dtype=np.float32)
    nativeWeightTrack, nativeRhsTrack, nativeSupportCount = (
        cconsenrich.cbackgroundWeightedStatsWithSupport(
            residualMatrix,
            zeroWeights,
        )
    )
    assert nativeSupportCount == 0
    np.testing.assert_array_equal(nativeWeightTrack, np.zeros(n, dtype=np.float64))
    np.testing.assert_array_equal(nativeRhsTrack, np.zeros(n, dtype=np.float64))
    zeroBackground = core.solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=zeroWeights,
        blockLenIntervals=5,
        backgroundSmoothness=0.8,
        zeroCenter=False,
        useNonnegative=True,
    )
    np.testing.assert_array_equal(zeroBackground, np.zeros(n, dtype=np.float32))


@pytest.mark.correctness
def _caseFinalForwardNISUsesMeanFinalForwardDiagnostic():
    assert core._finalForwardNIS(
        np.array([0.25, 0.5, 1.25, np.nan], dtype=np.float32)
    ) == pytest.approx(
        (0.25 + 0.5 + 1.25) / 3.0,
    )
    assert np.isnan(core._finalForwardNIS(np.array([], dtype=np.float32)))
    assert np.isnan(core._finalForwardNIS(np.array([np.nan], dtype=np.float32)))
    with pytest.raises(ValueError):
        core._finalForwardNIS(np.zeros((2, 2), dtype=np.float32))


@pytest.mark.correctness
def _caseFinalForwardGainSummaryUsesReplicateContigRows():
    p00Forward = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float32)
    stateCovarForward = np.zeros((4, 2, 2), dtype=np.float32)
    stateCovarForward[:, 0, 0] = p00Forward
    matrixMunc = np.array(
        [
            [0.9, 0.9, 0.9, 0.9],
            [0.4, 0.6, 0.8, 1.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    lambdaExp = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)

    summary = core._finalForwardReplicateGainContigSummary(
        stateCovarForward=stateCovarForward,
        matrixMunc=matrixMunc,
        lambdaExp=lambdaExp,
        pad=0.1,
        obsPrecisionMultiplierMin=0.2,
        obsPrecisionMultiplierMax=5.0,
    )
    expectedGains = (
        p00Forward.astype(np.float64)[None, :]
        * lambdaExp.astype(np.float64)[None, :]
        / (matrixMunc.astype(np.float64) + 0.1)
    )
    expectedAverages = expectedGains.mean(axis=1)
    expectedMedians = np.median(expectedGains, axis=1)

    np.testing.assert_allclose(summary["mean"], expectedAverages)
    np.testing.assert_allclose(summary["median"], expectedMedians)
    np.testing.assert_array_equal(
        summary["count"], np.full(matrixMunc.shape[0], matrixMunc.shape[1])
    )


@pytest.mark.correctness
def test_perIntervalOutputDiagnosticTracks():
    stateCovarForward = np.zeros((3, 2, 2), dtype=np.float32)
    stateCovarForward[:, 0, 0] = [0.4, 0.5, 0.6]
    stateCovarForward[:, 0, 1] = [0.03, 0.04, 0.05]
    stateCovarForward[:, 1, 0] = stateCovarForward[:, 0, 1]
    stateCovarForward[:, 1, 1] = [0.2, 0.25, 0.3]
    matrixMunc = np.asarray(
        [[0.9, 1.9, 0.4], [1.1, 0.1, 0.6]],
        dtype=np.float32,
    )
    matrixQ0 = np.asarray([[0.2, 0.0], [0.0, 0.05]], dtype=np.float32)
    matrixF = np.asarray([[1.0, 0.1], [0.0, 1.0]], dtype=np.float32)
    lambdaExp = np.asarray([1.0, 2.0, 0.5], dtype=np.float32)
    processPrecExp = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)

    trackArgs = dict(
        stateCovarForward=stateCovarForward,
        matrixMunc=matrixMunc,
        matrixQ0=matrixQ0,
        matrixF=matrixF,
        stateCovarInit=1.0,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        lambdaExp=lambdaExp,
        processPrecExp=processPrecExp,
        pNoiseForward=None,
        pad=0.1,
        obsPrecisionMultiplierMin=0.25,
        obsPrecisionMultiplierMax=4.0,
        procPrecisionMultiplierMin=0.25,
        procPrecisionMultiplierMax=4.0,
    )

    tracks = core._perIntervalOutputDiagnosticTracks(**trackArgs)

    np.testing.assert_allclose(tracks["preKappaQLevel"], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(tracks["preKappaQTrend"], [0.05, 0.05, 0.05])
    np.testing.assert_allclose(tracks["effectiveQLevel"], [0.2, 0.1, 0.05])
    np.testing.assert_allclose(tracks["effectiveQTrend"], [0.05, 0.025, 0.0125])
    qMeta = core._processQDiagnosticsMetadata(
        matrixQ0=matrixQ0,
        intervalCount=3,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        processPrecExp=processPrecExp,
        pNoiseForward=None,
        processPrecisionRequested=True,
        processPrecisionEffective=True,
        procPrecisionMultiplierMin=0.25,
        procPrecisionMultiplierMax=4.0,
    )
    assert qMeta["effectiveQTraceMin"] == pytest.approx(0.0625)
    assert qMeta["effectiveQTraceMedian"] == pytest.approx(0.125)
    assert qMeta["effectiveQTraceMax"] == pytest.approx(0.25)
    expectedTrace = np.sum(
        (matrixMunc.astype(np.float64) + 0.1) / lambdaExp[None, :],
        axis=0,
    )
    np.testing.assert_allclose(tracks["muncTrace"], expectedTrace)

    sumInvR0 = (1.0 / 1.0) + (1.0 / 1.2)
    pred00 = 1.21
    pred10 = 0.1
    denom = 1.0 + pred00 * sumInvR0
    assert tracks["sumGain0"][0] == pytest.approx(pred00 * sumInvR0 / denom)
    assert tracks["sumGain1"][0] == pytest.approx(pred10 * sumInvR0 / denom)

    intervalCount = 65_539
    for stateDim in (1, 2):
        covar = np.resize(
            stateCovarForward[:, :stateDim, :stateDim],
            (intervalCount, stateDim, stateDim),
        )
        localArgs = dict(trackArgs)
        localArgs.update(
            stateCovarForward=covar,
            matrixMunc=np.resize(matrixMunc, (2, intervalCount)),
            lambdaExp=np.resize(lambdaExp, intervalCount),
            stateModel=(
                core.STATE_MODEL_LEVEL if stateDim == 1 else core.STATE_MODEL_LEVEL_TREND
            ),
        )
        for noiseSource in ("precision", "stored", "fixed"):
            precision = (
                np.resize(processPrecExp, intervalCount)
                if noiseSource == "precision" else None
            )
            noise = (
                np.resize(
                    matrixQ0[:stateDim, :stateDim] * 2,
                    (intervalCount - 1, stateDim, stateDim),
                )
                if noiseSource == "stored" else None
            )
            localArgs.update(processPrecExp=precision, pNoiseForward=noise)
            result = core._perIntervalOutputDiagnosticTracks(**localArgs)
            f = matrixF[:stateDim, :stateDim].astype(np.float64)
            for index in (0, 1, 2, 65_535, 65_536, intervalCount - 1):
                q = matrixQ0[:stateDim, :stateDim].astype(np.float64)
                if precision is not None:
                    q /= precision[index]
                elif noise is not None and index > 0:
                    q = noise[index - 1].astype(np.float64)
                covariance = (
                    np.eye(stateDim) if index == 0
                    else covar[index - 1].astype(np.float64)
                )
                predicted = f @ covariance @ f.T + q
                invR = np.sum(
                    localArgs["lambdaExp"][index]
                    / (localArgs["matrixMunc"][:, index].astype(np.float64) + 0.1)
                )
                expectedGain = predicted[:, 0] * invR / (1 + predicted[0, 0] * invR)
                assert result["sumGain0"][index] == pytest.approx(
                    expectedGain[0], rel=1e-6
                )
                assert result["sumGain1"][index] == pytest.approx(
                    0.0 if stateDim == 1 else expectedGain[1], rel=1e-6
                )


@pytest.mark.correctness
def _caseBackgroundPenaltyWeightsScaleByDifferenceOrder():
    lamFirst, lamSecond = core._backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=12,
        backgroundSmoothness=0.5,
    )

    assert lamFirst == pytest.approx(0.5 * (12.0**2) / 4.0)
    assert lamSecond == pytest.approx(0.5 * (12.0**4) / 16.0)
    assert core._backgroundPenaltyFromSpan(
        blockLenIntervals=12,
        backgroundSmoothness=0.5,
    ) == pytest.approx(lamSecond)


@pytest.mark.correctness
def _caseCFixedBackgroundECMTinyTrackUsesFiniteFallback():
    n = 5
    m = 2
    matrixData = np.vstack(
        [
            np.linspace(0.2, 1.0, n, dtype=np.float32),
            np.linspace(0.35, 1.15, n, dtype=np.float32),
        ]
    )
    matrixMunc = np.full((m, n), 0.08, dtype=np.float32)
    matrixF = core.constructMatrixF(0.1).astype(np.float32, copy=False)
    matrixQ0 = core.constructMatrixQ(minDiagQ=1.0e-5).astype(np.float32, copy=False)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)

    out = cconsenrich.cfixedBackgroundECM(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0,
        ECM_fixedBackgroundIters=3,
        ECM_fixedBackgroundRtol=0.0,
        pad=1.0e-4,
        ECM_robustTNu=8.0,
        ECM_useObsPrecisionReweighting=True,
        ECM_useProcessPrecisionReweighting=True,
        returnIntermediates=True,
        returnDiagnostics=True,
        t_innerIters=5,
        logIterations=False,
    )

    (
        itersDone,
        finalNLL,
        stateSmoothed,
        stateCovarSmoothed,
        lagCovSmoothed,
        postFitResiduals,
        lambdaExp,
        processPrecExp,
        ecmDiagnostics,
    ) = out

    assert itersDone == 0
    assert np.isfinite(finalNLL)
    assert ecmDiagnostics["skipped"] is True
    assert ecmDiagnostics["skip_reason"] == "too_few_intervals"
    assert ecmDiagnostics["fallback"] == "filter_smoother_only"
    assert ecmDiagnostics["final_nll"] == pytest.approx(finalNLL)
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert lagCovSmoothed.shape[0] >= n - 1
    assert postFitResiduals.shape == (n, m)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.all(np.isfinite(stateCovarSmoothed))
    assert np.all(np.isfinite(lagCovSmoothed[: n - 1]))
    assert np.all(np.isfinite(postFitResiduals))
    assert np.all(np.isfinite(lambdaExp))
    assert np.all(np.isfinite(processPrecExp))


@pytest.mark.correctness
def _caseCFixedBackgroundECMLevelTinyTrackUsesFiniteFallback():
    n = 5
    m = 2
    matrixData = np.vstack(
        [
            np.linspace(-0.1, 0.3, n, dtype=np.float32),
            np.linspace(0.0, 0.4, n, dtype=np.float32),
        ]
    )
    matrixMunc = np.full((m, n), 0.05, dtype=np.float32)
    matrixQ0 = np.asarray([[1.0e-4]], dtype=np.float32)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)

    out = cconsenrich.cfixedBackgroundECMLevel(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0,
        ECM_fixedBackgroundIters=3,
        ECM_fixedBackgroundRtol=0.0,
        pad=1.0e-4,
        ECM_robustTNu=8.0,
        ECM_useObsPrecisionReweighting=True,
        ECM_useProcessPrecisionReweighting=True,
        returnIntermediates=True,
        returnDiagnostics=True,
        t_innerIters=5,
        logIterations=False,
    )

    (
        itersDone,
        finalNLL,
        stateSmoothed,
        stateCovarSmoothed,
        lagCovSmoothed,
        postFitResiduals,
        lambdaExp,
        processPrecExp,
        ecmDiagnostics,
    ) = out

    assert itersDone == 0
    assert np.isfinite(finalNLL)
    assert ecmDiagnostics["skipped"] is True
    assert ecmDiagnostics["skip_reason"] == "too_few_intervals"
    assert ecmDiagnostics["fallback"] == "filter_smoother_only"
    assert ecmDiagnostics["final_nll"] == pytest.approx(finalNLL)
    assert stateSmoothed.shape == (n, 1)
    assert stateCovarSmoothed.shape == (n, 1, 1)
    assert lagCovSmoothed.shape[0] >= n - 1
    assert postFitResiduals.shape == (n, m)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.all(np.isfinite(stateCovarSmoothed))
    assert np.all(np.isfinite(lagCovSmoothed[: n - 1]))
    assert np.all(np.isfinite(postFitResiduals))
    assert np.all(np.isfinite(lambdaExp))
    assert np.all(np.isfinite(processPrecExp))


@pytest.mark.correctness
def _caseObservationPrecisionIsIntervalLevelOnly():
    n = 10
    m = 2
    matrixData = np.zeros((m, n), dtype=np.float32)
    matrixMunc = np.full((m, n), 0.2, dtype=np.float32)
    matrixF = core.constructMatrixF(0.1).astype(np.float32, copy=False)
    matrixQ0 = core.constructMatrixQ(
        minDiagQ=1.0e-4,
    ).astype(np.float32, copy=False)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)

    out = cconsenrich.cfixedBackgroundECM(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0,
        ECM_fixedBackgroundIters=1,
        ECM_fixedBackgroundRtol=0.0,
        ECM_useObsPrecisionReweighting=True,
        ECM_useProcessPrecisionReweighting=False,
        ECM_scaleProcessPrecisionToMedian=False,
        returnIntermediates=True,
        t_innerIters=1,
    )
    assert np.asarray(out[6]).shape == (n,)

    with pytest.raises(ValueError):
        cconsenrich.cforwardPass(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixMunc,
            matrixF=matrixF,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=1,
            stateInit=0.0,
            stateCovarInit=1.0,
            lambdaExp=np.ones((m, n), dtype=np.float32),
            ECM_useObsPrecisionReweighting=True,
        )

    with pytest.raises(ValueError):
        core.runConsenrich(
            matrixData,
            matrixMunc,
            deltaF=0.1,
            minQ=1.0e-4,
            maxQ=0.5,
            stateInit=0.0,
            stateCovarInit=1.0,
            boundState=False,
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            blockLenIntervals=4,
            ECM_fixedBackgroundIters=1,
            initialObservationPrecision=np.ones((1, n), dtype=np.float32),
        )


def _checkCFixedBackgroundPrecisionUpdates(levelOnly, scaleObs, scaleProcess):
    n = 8
    m = 2
    grid = np.linspace(-0.5, 0.8, n, dtype=np.float32)
    matrixData = np.vstack(
        [
            grid
            + np.asarray(
                [0.0, 0.2, -0.1, 0.4, 0.0, -0.3, 0.1, 0.6],
                dtype=np.float32,
            ),
            grid
            + np.asarray(
                [0.1, -0.2, 0.2, -0.1, 0.3, 0.1, -0.4, 0.2],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    matrixMunc = np.vstack(
        [
            np.linspace(0.05, 0.20, n),
            np.linspace(0.12, 0.30, n),
        ]
    ).astype(np.float32)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)
    observationNu = 5.0
    processNu = 11.0
    pad = 0.01
    multiplierMin = 0.97
    multiplierMax = 1.03

    if levelOnly:
        matrixQ0 = np.asarray([[0.04]], dtype=np.float32)
        out = cconsenrich.cfixedBackgroundECMLevel(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixMunc,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=1,
            stateInit=0.0,
            stateCovarInit=1.0,
            ECM_fixedBackgroundIters=1,
            ECM_fixedBackgroundRtol=0.0,
            pad=pad,
            ECM_robustTNu=observationNu,
            ECM_processRobustTNu=processNu,
            obsPrecisionMultiplierMin=multiplierMin,
            obsPrecisionMultiplierMax=multiplierMax,
            procPrecisionMultiplierMin=multiplierMin,
            procPrecisionMultiplierMax=multiplierMax,
            ECM_useObsPrecisionReweighting=True,
            ECM_useProcessPrecisionReweighting=True,
            ECM_scaleObsPrecisionToMedian=scaleObs,
            ECM_scaleProcessPrecisionToMedian=scaleProcess,
            returnIntermediates=True,
            returnDiagnostics=True,
            t_innerIters=1,
            logIterations=False,
        )
        matrixF = np.asarray([[1.0]], dtype=np.float64)
        stateDim = 1
    else:
        matrixF = core.constructMatrixF(0.3).astype(np.float32)
        matrixQ0 = core.constructMatrixQ(
            minDiagQ=1.0e-5,
            Q00=0.04,
            Q01=0.0,
            Q10=0.0,
            Q11=0.02,
        ).astype(np.float32)
        out = cconsenrich.cfixedBackgroundECM(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixMunc,
            matrixF=matrixF,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=1,
            stateInit=0.0,
            stateCovarInit=1.0,
            ECM_fixedBackgroundIters=1,
            ECM_fixedBackgroundRtol=0.0,
            pad=pad,
            ECM_robustTNu=observationNu,
            ECM_processRobustTNu=processNu,
            obsPrecisionMultiplierMin=multiplierMin,
            obsPrecisionMultiplierMax=multiplierMax,
            procPrecisionMultiplierMin=multiplierMin,
            procPrecisionMultiplierMax=multiplierMax,
            ECM_useObsPrecisionReweighting=True,
            ECM_useProcessPrecisionReweighting=True,
            ECM_scaleObsPrecisionToMedian=scaleObs,
            ECM_scaleProcessPrecisionToMedian=scaleProcess,
            returnIntermediates=True,
            returnDiagnostics=True,
            t_innerIters=1,
            logIterations=False,
        )
        stateDim = 2

    stateSmoothed = np.asarray(out[2], dtype=np.float64)
    stateCovarSmoothed = np.asarray(out[3], dtype=np.float64)
    lagCovSmoothed = np.asarray(out[4], dtype=np.float64)
    lambdaExp = np.asarray(out[6], dtype=np.float64)
    processPrecExp = np.asarray(out[7], dtype=np.float64)

    np.testing.assert_array_equal(
        out[5],
        (matrixData.T.astype(np.float64) - stateSmoothed[:, :1]).astype(np.float32),
    )

    expectedLambda = []
    for k in range(n):
        obsU2 = 0.0
        p00 = max(float(stateCovarSmoothed[k, 0, 0]), 0.0)
        for j in range(m):
            obsU2 += (
                (float(matrixData[j, k]) - float(stateSmoothed[k, 0])) ** 2 + p00
            ) / (float(matrixMunc[j, k]) + pad)
        expectedLambda.append(
            (observationNu + m) / (observationNu + obsU2)
        )
    expectedLambda = np.asarray(expectedLambda, dtype=np.float64)
    if scaleObs:
        expectedLambda /= np.median(expectedLambda)
    expectedLambda = np.clip(expectedLambda, multiplierMin, multiplierMax)

    expectedProcess = []
    qInv = np.linalg.inv(np.asarray(matrixQ0[:stateDim, :stateDim], dtype=np.float64))
    f = np.asarray(matrixF[:stateDim, :stateDim], dtype=np.float64)
    for k in range(n - 1):
        x = stateSmoothed[k, :stateDim]
        y = stateSmoothed[k + 1, :stateDim]
        innovation = y - (f @ x)
        delta = float(innovation @ qInv @ innovation)
        expectedProcess.append(
            (processNu + stateDim) / (processNu + max(delta, 0.0))
        )
    expectedProcess = np.asarray(expectedProcess, dtype=np.float64)
    if scaleProcess:
        expectedProcess /= np.median(expectedProcess)
    expectedProcess = np.concatenate(
        ([1.0], np.clip(expectedProcess, multiplierMin, multiplierMax))
    )

    np.testing.assert_allclose(lambdaExp, expectedLambda, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(
        processPrecExp,
        expectedProcess,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert out[8]["scale_obs_precision_to_median"] is scaleObs
    assert out[8]["scale_process_precision_to_median"] is scaleProcess

    if scaleObs or scaleProcess:
        return

    if levelOnly:
        disabled = cconsenrich.cfixedBackgroundECMLevel(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixMunc,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=1,
            stateInit=0.0,
            stateCovarInit=1.0,
            ECM_fixedBackgroundIters=1,
            ECM_useObsPrecisionReweighting=False,
            ECM_useProcessPrecisionReweighting=False,
            ECM_scaleProcessPrecisionToMedian=False,
            returnIntermediates=True,
            t_innerIters=1,
            logIterations=False,
        )
    else:
        disabled = cconsenrich.cfixedBackgroundECM(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixMunc,
            matrixF=matrixF.astype(np.float32),
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=1,
            stateInit=0.0,
            stateCovarInit=1.0,
            ECM_fixedBackgroundIters=1,
            ECM_useObsPrecisionReweighting=False,
            ECM_useProcessPrecisionReweighting=False,
            ECM_scaleProcessPrecisionToMedian=False,
            returnIntermediates=True,
            t_innerIters=1,
            logIterations=False,
        )
    assert disabled[6] is None
    assert disabled[7] is None


@pytest.mark.correctness
@pytest.mark.parametrize("levelOnly", [True, False])
@pytest.mark.parametrize(
    "scaleObs,scaleProcess",
    [(False, False), (True, False), (False, True)],
)
def test_cfixed_background_precision_updates_match_student_t_equations(
    levelOnly,
    scaleObs,
    scaleProcess,
):
    _checkCFixedBackgroundPrecisionUpdates(levelOnly, scaleObs, scaleProcess)


@pytest.mark.correctness
@pytest.mark.parametrize("levelOnly", [True, False])
def test_cfixed_background_precision_median_scaling_applies_to_warm_starts(
    levelOnly,
):
    intervalCount = 5
    kernel = cconsenrich.cfixedBackgroundECMLevel
    modelKwargs = {
        "matrixQ0": np.asarray([[0.1]], dtype=np.float32),
    }
    if not levelOnly:
        kernel = cconsenrich.cfixedBackgroundECM
        modelKwargs = {
            "matrixF": np.eye(2, dtype=np.float32),
            "matrixQ0": np.diag([0.1, 0.05]).astype(np.float32),
        }
    kernelKwargs = dict(
        matrixData=np.zeros((1, intervalCount), dtype=np.float32),
        matrixPluginMuncInit=np.ones((1, intervalCount), dtype=np.float32),
        intervalToBlockMap=np.zeros(intervalCount, dtype=np.int32),
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0,
        obsPrecisionMultiplierMin=0.2,
        obsPrecisionMultiplierMax=5.0,
        procPrecisionMultiplierMin=0.2,
        procPrecisionMultiplierMax=5.0,
        ECM_useObsPrecisionReweighting=True,
        ECM_useProcessPrecisionReweighting=True,
        ECM_processRobustTNu=6.0,
        ECM_scaleObsPrecisionToMedian=True,
        ECM_scaleProcessPrecisionToMedian=True,
        returnIntermediates=True,
        logIterations=False,
    )
    kernelKwargs.update(modelKwargs)
    out = kernel(
        **kernelKwargs,
        lambdaExpInit=np.asarray([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32),
        processPrecExpInit=np.asarray(
            [999.0, 0.01, 0.01, 1.0, 2.0],
            dtype=np.float32,
        ),
    )

    assert out[0] == 0
    np.testing.assert_allclose(out[6], [0.25, 0.5, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        out[7],
        [1.0, 0.2, 0.2, 1.980198, 3.960396],
        rtol=2.0e-6,
    )
    preparedLambda = np.asarray([0.2, 0.2, 1.5, 3.0, 5.0], dtype=np.float32)
    preparedOut = kernel(
        **kernelKwargs,
        lambdaExpInit=preparedLambda,
        processPrecExpInit=np.asarray(out[7], dtype=np.float32),
        obsPrecisionWarmStartIsMedianScaled=True,
        processPrecisionWarmStartIsMedianScaled=True,
    )
    np.testing.assert_array_equal(preparedOut[6], preparedLambda)
    np.testing.assert_array_equal(preparedOut[7], out[7])

    pointwiseSeedKwargs = dict(kernelKwargs)
    pointwiseSeedKwargs["matrixData"] = np.asarray(
        [[0.0, 0.0, 2.0, 2.0, 2.0]],
        dtype=np.float32,
    )
    pointwiseSeedOut = kernel(**pointwiseSeedKwargs)
    samplingVariance = 2.0 * (1.0 + 1.0e-4)
    transitionStatistic = (4.0 - samplingVariance) / 0.1
    rawFlatPrecision = 7.0 / 6.0
    rawJumpPrecision = 7.0 / (6.0 + transitionStatistic)
    np.testing.assert_allclose(
        pointwiseSeedOut[7],
        [1.0, 1.0, rawJumpPrecision / rawFlatPrecision, 1.0, 1.0],
        rtol=2.0e-6,
    )


@pytest.mark.correctness
def _caseSummarizeStateRoughnessUsesHoldoutBlocksAndSignalStrata():
    state = np.asarray(
        [
            0.0,
            1.0,
            2.0,
            5.0,
            6.0,
            9.0,
            10.0,
            12.0,
            16.0,
            30.0,
            34.0,
            42.0,
        ],
        dtype=np.float64,
    )

    summary = diagnostics.summarizeStateRoughness(
        state,
        blockLenIntervals=3,
        intervalSizeBP=25,
    )

    assert summary["block_len_intervals"] == 3
    assert summary["block_len_bp"] == 75
    assert summary["n_blocks"] == 4
    assert summary["n_differences"] == 8
    assert summary["overall_mean_abs_diff"] == pytest.approx(3.0)
    assert summary["block_mean_abs_diff_median"] == pytest.approx(2.5)
    assert summary["block_mean_abs_diff_q90"] == pytest.approx(5.1)
    strata = {row["stratum"]: row for row in summary["signal_strata"]}
    assert strata["signal_abs_q00_50"]["mean_abs_diff"] == pytest.approx(1.5)
    assert strata["signal_abs_q50_90"]["mean_abs_diff"] == pytest.approx(3.0)
    assert strata["signal_abs_q90_100"]["mean_abs_diff"] == pytest.approx(6.0)


@pytest.mark.correctness
def _caseSummarizePrecisionBoundaryHitsSkipsFirstProcessWeight():
    summary = diagnostics.summarizePrecisionBoundaryHits(
        observationPrecision=np.asarray(
            [[0.1, 1.0, 10.0], [0.10000001, 10.0, 3.0]],
            dtype=np.float64,
        ),
        observationPrecisionMin=0.1,
        observationPrecisionMax=10.0,
        processPrecision=np.asarray([0.2, 0.2, 5.0, 3.0, 0.2], dtype=np.float64),
        processPrecisionMin=0.2,
        processPrecisionMax=5.0,
    )

    assert summary["observation"]["total"] == 6
    assert summary["observation"]["lower"] == 2
    assert summary["observation"]["upper"] == 2
    assert summary["process"]["total"] == 4
    assert summary["process"]["lower"] == 2
    assert summary["process"]["upper"] == 1


@pytest.mark.correctness
def _caseExpectedTransitionResidualSumsUsesLagOrientationAndDeltaF():
    matrixF = core.constructMatrixF(0.5).astype(np.float64, copy=False)
    stateSmoothed = np.asarray(
        [
            [0.0, 1.0],
            [0.6, 1.2],
            [1.25, 1.0],
            [1.75, 1.1],
        ],
        dtype=np.float64,
    )
    stateCovarSmoothed = np.zeros((4, 2, 2), dtype=np.float64)
    lagCovSmoothed = np.zeros((3, 2, 2), dtype=np.float64)

    sumLevel, sumTrend, transitionCount = core._computeExpectedTransitionResidualSums(
        stateSmoothed=stateSmoothed,
        stateCovarSmoothed=stateCovarSmoothed,
        lagCovSmoothed=lagCovSmoothed,
        matrixF=matrixF,
    )

    assert transitionCount == 3
    assert sumLevel == pytest.approx((0.1**2) + (0.05**2) + (0.0**2))
    assert sumTrend == pytest.approx((0.2**2) + ((-0.2) ** 2) + (0.1**2))


@pytest.mark.correctness
def _caseExpectedTransitionResidualSumsMatchesPythonReference():
    rng = np.random.default_rng(123)
    n = 19
    matrixF = np.asarray([[1.0, 0.35], [0.02, 0.97]], dtype=np.float64)
    stateSmoothed = rng.normal(size=(n, 2)).astype(np.float64)
    rawCov = rng.normal(scale=0.1, size=(n, 2, 2)).astype(np.float64)
    stateCovarSmoothed = rawCov + np.swapaxes(rawCov, 1, 2)
    lagCovSmoothed = rng.normal(scale=0.05, size=(n - 1, 2, 2)).astype(np.float64)

    expectedLevel = 0.0
    expectedTrend = 0.0
    ft = matrixF.T
    for k in range(n - 1):
        x0 = stateSmoothed[k]
        x1 = stateSmoothed[k + 1]
        exx0 = stateCovarSmoothed[k] + np.outer(x0, x0)
        exx1 = stateCovarSmoothed[k + 1] + np.outer(x1, x1)
        ex0x1 = lagCovSmoothed[k] + np.outer(x0, x1)
        ex1x0 = ex0x1.T
        residualSecondMoment = (
            exx1 - (ex1x0 @ ft) - (matrixF @ ex0x1) + (matrixF @ exx0 @ ft)
        )
        expectedLevel += max(float(residualSecondMoment[0, 0]), 0.0)
        expectedTrend += max(float(residualSecondMoment[1, 1]), 0.0)

    sumLevel, sumTrend, transitionCount = core._computeExpectedTransitionResidualSums(
        stateSmoothed=stateSmoothed,
        stateCovarSmoothed=stateCovarSmoothed,
        lagCovSmoothed=lagCovSmoothed,
        matrixF=matrixF,
    )

    assert transitionCount == n - 1
    assert sumLevel == pytest.approx(expectedLevel, rel=1.0e-12, abs=1.0e-12)
    assert sumTrend == pytest.approx(expectedTrend, rel=1.0e-12, abs=1.0e-12)


@pytest.mark.correctness
def _caseNormalizeStateModelAcceptsCanonicalValuesOnly():
    params = dict(
        default=constants.STATE_MODEL_LEVEL_TREND,
        supported=constants.STATE_MODEL_MODES,
        paramName="stateModel",
    )
    for value in (None, "levelTrend", "level"):
        assert normalization.validateChoice(value, **params) == (
            "levelTrend" if value is None else value
        )
    for badMode in ("level-trend", "two_state", "one-state", "scalar", "levelSlope"):
        with pytest.raises(ValueError, match="stateModel"):
            normalization.validateChoice(badMode, **params)


@pytest.mark.correctness
def _caseExpectedLevelTransitionResidualSumsMatchesPythonReference():
    stateSmoothed = np.asarray([[0.0], [0.4], [0.9], [0.7]], dtype=np.float64)
    stateCovarSmoothed = np.asarray([0.10, 0.20, 0.15, 0.12], dtype=np.float64).reshape(
        -1, 1, 1
    )
    lagCovSmoothed = np.asarray([0.03, 0.04, 0.02], dtype=np.float64).reshape(-1, 1, 1)

    expected = 0.0
    for k in range(stateSmoothed.shape[0] - 1):
        x0 = stateSmoothed[k, 0]
        x1 = stateSmoothed[k + 1, 0]
        expected += max(
            float(
                stateCovarSmoothed[k + 1, 0, 0]
                + x1 * x1
                - 2.0 * (lagCovSmoothed[k, 0, 0] + x0 * x1)
                + stateCovarSmoothed[k, 0, 0]
                + x0 * x0
            ),
            0.0,
        )

    sumLevel, sumTrend, transitionCount = (
        core._computeExpectedLevelTransitionResidualSums(
            stateSmoothed=stateSmoothed,
            stateCovarSmoothed=stateCovarSmoothed,
            lagCovSmoothed=lagCovSmoothed,
        )
    )

    assert transitionCount == 3
    assert sumLevel == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
    assert sumTrend == pytest.approx(0.0)

    for stateDim in (1, 2):
        for meanValue, varianceValue in ((1.0e8, 1.0), (100.0, 1.0e-12)):
            means = np.full((2, stateDim), meanValue, dtype=np.float64)
            covariance = np.repeat((np.eye(stateDim) * varianceValue)[None], 2, axis=0)
            lagCovariance = covariance[:1] * 0.5
            if stateDim == 1:
                result = cconsenrich.cExpectedTransitionResidualSumsLevel(
                    means, covariance, lagCovariance,
                )
            else:
                result = cconsenrich.cExpectedTransitionResidualSums(
                    means, covariance, lagCovariance, np.eye(2),
                )
            assert result[0] == pytest.approx(varianceValue, rel=1.0e-14, abs=0.0)
            assert result[1] == pytest.approx(varianceValue if stateDim == 2 else 0.0, rel=1.0e-14, abs=0.0)
            assert result[2] == 1


@pytest.mark.correctness
def _caseLevelForwardBackwardMatchesPythonReference():
    matrixData = np.asarray(
        [
            [0.2, 0.4, 0.5, 0.7, 0.1, -0.2, 0.0],
            [0.1, 0.3, 0.65, 0.5, 0.0, -0.1, 0.2],
        ],
        dtype=np.float32,
    )
    matrixMunc = np.asarray(
        [
            [0.40, 0.35, 0.30, 0.42, 0.38, 0.33, 0.31],
            [0.45, 0.37, 0.34, 0.40, 0.36, 0.35, 0.32],
        ],
        dtype=np.float32,
    )
    n = matrixData.shape[1]
    levelQ = 0.06
    stateInit = -0.1
    stateCovarInit = 0.8
    pad = 0.02
    intervalToBlockMap = np.zeros(n, dtype=np.int32)
    matrixQ0 = np.asarray([[levelQ, 0.0], [0.0, 0.5]], dtype=np.float32)
    stateForward = np.empty((n, 1), dtype=np.float32)
    covForward = np.empty((n, 1, 1), dtype=np.float32)
    pNoiseForward = np.empty((n, 1, 1), dtype=np.float32)
    vectorD = np.empty(n, dtype=np.float32)

    phiHat, _sentinel, outD, sumNLL = cconsenrich.cforwardPassLevel(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=stateInit,
        stateCovarInit=stateCovarInit,
        pad=pad,
        stateForward=stateForward,
        stateCovarForward=covForward,
        pNoiseForward=pNoiseForward,
        vectorD=vectorD,
        returnNLL=True,
        ECM_useObsPrecisionReweighting=False,
        ECM_useProcessPrecisionReweighting=False,
    )
    stateSmoothed, covSmoothed, lagCovSmoothed, residuals = (
        cconsenrich.cbackwardPassLevel(
            matrixData=matrixData,
            stateForward=stateForward,
            stateCovarForward=covForward,
            pNoiseForward=pNoiseForward,
        )
    )

    refForward, refCovForward, _refPNoise, refState, refCov, refLag, refResiduals = (
        _levelKalmanReference(
            matrixData,
            matrixMunc,
            levelQ=levelQ,
            stateInit=stateInit,
            stateCovarInit=stateCovarInit,
            pad=pad,
        )
    )

    assert np.isfinite(float(phiHat))
    assert np.isfinite(float(sumNLL))
    assert np.asarray(outD).shape == (n,)
    np.testing.assert_allclose(stateForward, refForward, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(covForward, refCovForward, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(stateSmoothed, refState, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(covSmoothed, refCov, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(
        lagCovSmoothed[: n - 1], refLag[: n - 1], rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(residuals, refResiduals, rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.parametrize("stateDim", [1, 2])
def test_observationSufficientStats(stateDim):
    rng = np.random.default_rng(9017)
    for trackCount in (1, 10):
        intervalCount = 257
        data = rng.normal(size=(trackCount, intervalCount)).astype(np.float32)
        variance = np.exp(rng.normal(size=data.shape)).astype(np.float32)
        variance[:, 20:25] = 1.0e30
        variance[0, 40:60] = 1.0e30
        pad = float(np.float32(0.01))
        active = variance < 5.0e29
        weights = np.where(active, 1.0 / (variance.astype(np.float64) + pad), 0.0)
        weightSum = weights.sum(axis=0)
        mean = np.divide(
            (weights * data).sum(axis=0), weightSum,
            out=np.zeros(intervalCount), where=weightSum > 0,
        )
        stats = np.column_stack((
            weightSum, mean, (weights * (data - mean) ** 2).sum(axis=0),
            np.where(active, np.log(variance.astype(np.float64) + pad), 0.0).sum(axis=0),
            active.sum(axis=0),
        ))
        args = dict(
            matrixData=data, matrixPluginMuncInit=variance,
            matrixQ0=np.eye(stateDim, dtype=np.float32) * 0.01,
            intervalToBlockMap=np.zeros(intervalCount, dtype=np.int32), blockCount=1,
            stateInit=0.0, stateCovarInit=1.0, pad=pad, returnNLL=True,
            lambdaExp=rng.uniform(0.25, 4.0, intervalCount).astype(np.float32),
            processPrecExp=rng.uniform(0.25, 4.0, intervalCount).astype(np.float32),
        )
        forward = cconsenrich.cforwardPassLevel
        if stateDim == 2:
            forward = cconsenrich.cforwardPass
            args["matrixF"] = np.asarray([[1.0, 0.3], [0.0, 1.0]], dtype=np.float32)
        outputs = []
        for observationStats in (None, stats):
            state = np.empty((intervalCount, stateDim), dtype=np.float32)
            covariance = np.empty((intervalCount, stateDim, stateDim), dtype=np.float32)
            noise = np.empty_like(covariance)
            statistic = np.empty(intervalCount, dtype=np.float32)
            result = forward(
                **args, observationStats=observationStats, stateForward=state,
                stateCovarForward=covariance, pNoiseForward=noise, vectorD=statistic,
            )
            outputs.append((state, covariance, noise[:-1], statistic, result[3]))
            skippedState = np.empty_like(state)
            skippedCovariance = np.empty_like(covariance)
            skippedNoise = np.empty_like(noise)
            unusedStatistic = np.full(intervalCount, -123.0, dtype=np.float32)
            for computeNLL in (False, True):
                skipped = forward(
                    **dict(args, returnNLL=computeNLL), observationStats=observationStats,
                    computeStatistic=False, stateForward=skippedState,
                    stateCovarForward=skippedCovariance, pNoiseForward=skippedNoise,
                    vectorD=unusedStatistic,
                )
                np.testing.assert_array_equal(skippedState, state)
                np.testing.assert_array_equal(skippedCovariance, covariance)
                np.testing.assert_array_equal(skippedNoise[:-1], noise[:-1])
                np.testing.assert_array_equal(unusedStatistic, -123.0)
                assert skipped[2] is None
                if computeNLL:
                    assert skipped[3] == result[3]
        for collapsed, direct in zip(outputs[1], outputs[0], strict=True):
            np.testing.assert_allclose(collapsed, direct, rtol=2.0e-6, atol=2.0e-7)
    for computeNLL in (False, True):
        empty = forward(**dict(
            args, matrixData=data[:, :0].copy(),
            matrixPluginMuncInit=variance[:, :0].copy(),
            computeStatistic=False, returnNLL=computeNLL,
        ))
        assert empty == ((0.0, 0, None, 0.0) if computeNLL else (0.0, 0, None))

    for observedValue in (0.0, 100.0, 1.0e4):
        tinyVariance = np.float32(1.0e-12)
        measurementVariance = max(float(tinyVariance), 1.0e-12)
        totalVariance = 1.0e4 + float(tinyVariance) + measurementVariance
        expectedNIS = observedValue ** 2 / totalVariance
        expectedNLL = 0.5 * (np.log(2.0 * np.pi * totalVariance) + expectedNIS)
        singleArgs = dict(
            matrixData=np.array([[observedValue]], dtype=np.float32),
            matrixPluginMuncInit=np.array([[tinyVariance]], dtype=np.float32),
            matrixQ0=np.eye(stateDim, dtype=np.float32) * tinyVariance,
            intervalToBlockMap=np.zeros(1, dtype=np.int32), blockCount=1,
            stateInit=0.0, stateCovarInit=1.0e4, pad=0.0, returnNLL=True,
            ECM_useObsPrecisionReweighting=False,
            ECM_useProcessPrecisionReweighting=False,
        )
        if stateDim == 2:
            singleArgs["matrixF"] = np.eye(2, dtype=np.float32)
        singleStats = np.array([[1.0 / measurementVariance, observedValue, 0.0,
                                 np.log(measurementVariance), 1.0]])
        for observationStats in (None, singleStats):
            result = forward(**singleArgs, observationStats=observationStats,
                             vectorD=np.empty(1, dtype=np.float32))
            np.testing.assert_allclose(result[2], expectedNIS, rtol=1.0e-7, atol=0.0)
            assert result[3] == pytest.approx(expectedNLL, rel=1.0e-13)


@pytest.mark.correctness
def test_forwardBackwardResidualBuffers():
    for check in (
        _caseExpectedTransitionResidualSumsUsesLagOrientationAndDeltaF,
        _caseExpectedTransitionResidualSumsMatchesPythonReference,
        _caseExpectedLevelTransitionResidualSumsMatchesPythonReference,
        _caseLevelForwardBackwardMatchesPythonReference,
    ):
        check()
    matrixData = np.asarray(
        [
            [0.25, 0.10, 0.45, 0.75, 0.20, -0.10, 0.05],
            [0.15, 0.20, 0.35, 0.65, 0.10, -0.20, 0.15],
        ],
        dtype=np.float32,
    )
    matrixMunc = np.asarray(
        [
            [0.20, 0.25, 0.18, 0.22, 0.30, 0.28, 0.24],
            [0.27, 0.23, 0.31, 0.19, 0.29, 0.25, 0.33],
        ],
        dtype=np.float32,
    )
    n = matrixData.shape[1]
    matrixF = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matrixQ0 = np.asarray([[0.045, 0.0], [0.0, 0.125]], dtype=np.float32)
    sharedKwargs = {
        "matrixData": matrixData,
        "matrixPluginMuncInit": matrixMunc,
        "intervalToBlockMap": np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int32),
        "blockCount": 3,
        "stateInit": -0.15,
        "stateCovarInit": 0.7,
        "pad": 0.015,
        "returnNLL": True,
        "storeNLLInD": True,
        "lambdaExp": np.asarray(
            [0.10, 0.50, 1.20, 3.50, 7.00, 0.80, 2.20],
            dtype=np.float32,
        ),
        "processPrecExp": np.asarray(
            [1.00, 0.20, 0.75, 2.50, 6.00, 1.25, 0.40],
            dtype=np.float32,
        ),
        "obsPrecisionMultiplierMin": 0.25,
        "obsPrecisionMultiplierMax": 4.0,
        "procPrecisionMultiplierMin": 0.25,
        "procPrecisionMultiplierMax": 4.0,
        "ECM_useObsPrecisionReweighting": True,
    }
    levelStore = {
        "stateForward": np.empty((n, 1), dtype=np.float32),
        "stateCovarForward": np.empty((n, 1, 1), dtype=np.float32),
        "pNoiseForward": np.empty((n, 1, 1), dtype=np.float32),
        "vectorD": np.empty(n, dtype=np.float32),
    }
    fullStore = {
        "stateForward": np.empty((n, 2), dtype=np.float32),
        "stateCovarForward": np.empty((n, 2, 2), dtype=np.float32),
        "pNoiseForward": np.empty((n, 2, 2), dtype=np.float32),
        "vectorD": np.empty(n, dtype=np.float32),
    }

    levelPhiHat, levelSentinel, levelDOut, levelNLL = cconsenrich.cforwardPassLevel(
        **sharedKwargs,
        matrixQ0=matrixQ0,
        **levelStore,
    )
    fullPhiHat, fullSentinel, fullDOut, fullNLL = cconsenrich.cforwardPass(
        **sharedKwargs,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        **fullStore,
    )

    assert levelSentinel == 0
    assert fullSentinel == 0
    assert levelDOut is levelStore["vectorD"]
    assert fullDOut is fullStore["vectorD"]
    assert fullPhiHat == pytest.approx(levelPhiHat, rel=2.0e-6, abs=2.0e-6)
    assert fullNLL == pytest.approx(levelNLL, rel=2.0e-6, abs=2.0e-6)
    for observed, expected in (
        (fullStore["vectorD"], levelStore["vectorD"]),
        (fullStore["stateForward"][:, :1], levelStore["stateForward"]),
        (fullStore["stateCovarForward"][:, :1, :1], levelStore["stateCovarForward"]),
        (
            fullStore["pNoiseForward"][: n - 1, :1, :1],
            levelStore["pNoiseForward"][: n - 1],
        ),
    ):
        np.testing.assert_allclose(observed, expected, rtol=2.0e-6, atol=2.0e-6)

    levelSmooth = cconsenrich.cbackwardPassLevel(
        matrixData=matrixData,
        stateForward=levelStore["stateForward"],
        stateCovarForward=levelStore["stateCovarForward"],
        pNoiseForward=levelStore["pNoiseForward"],
    )
    fullSmooth = cconsenrich.cbackwardPass(
        matrixData=matrixData,
        matrixF=matrixF,
        stateForward=fullStore["stateForward"],
        stateCovarForward=fullStore["stateCovarForward"],
        pNoiseForward=fullStore["pNoiseForward"],
    )
    for observed, expected in (
        (fullSmooth[0][:, :1], levelSmooth[0]),
        (fullSmooth[1][:, :1, :1], levelSmooth[1]),
        (fullSmooth[2][: n - 1, :1, :1], levelSmooth[2][: n - 1]),
        (fullSmooth[3], levelSmooth[3]),
    ):
        np.testing.assert_allclose(observed, expected, rtol=2.0e-6, atol=2.0e-6)
    for smoother, stored, expected, extraArgs in (
        (cconsenrich.cbackwardPassLevel, levelStore, levelSmooth, {}),
        (cconsenrich.cbackwardPass, fullStore, fullSmooth, {"matrixF": matrixF}),
    ):
        for computeLagCovariance in (True, False):
            lagBuffer = np.full_like(expected[2], -456.0)
            residualBuffer = np.full_like(expected[3], -123.0)
            withoutResiduals = smoother(
                matrixData=matrixData,
                stateForward=stored["stateForward"],
                stateCovarForward=stored["stateCovarForward"],
                pNoiseForward=stored["pNoiseForward"],
                postFitResiduals=residualBuffer,
                computeResiduals=False,
                computeLagCovariance=computeLagCovariance,
                lagCovSmoothed=lagBuffer,
                **extraArgs,
            )
            for result, reference in zip(withoutResiduals[:2], expected[:2]):
                np.testing.assert_array_equal(result, reference)
            assert withoutResiduals[3] is None
            np.testing.assert_array_equal(residualBuffer, -123.0)
            if computeLagCovariance:
                np.testing.assert_array_equal(withoutResiduals[2], expected[2])
            else:
                assert withoutResiduals[2] is None
                np.testing.assert_array_equal(lagBuffer, -456.0)

    for stateDim in (1, 2):
        forward = cconsenrich.cforwardPassLevel if stateDim == 1 else cconsenrich.cforwardPass
        backward = cconsenrich.cbackwardPassLevel if stateDim == 1 else cconsenrich.cbackwardPass
        extraArgs = {} if stateDim == 1 else {"matrixF": np.eye(2, dtype=np.float32)}
        for processVariance in (1.0e-12, 0.01):
            data = np.zeros((1, 2), dtype=np.float32)
            stored = dict(
                stateForward=np.empty((2, stateDim), dtype=np.float32),
                stateCovarForward=np.empty((2, stateDim, stateDim), dtype=np.float32),
                pNoiseForward=np.empty((2, stateDim, stateDim), dtype=np.float32),
            )
            forward(
                matrixData=data,
                matrixPluginMuncInit=np.array([[1.0e30, 1.0e-12]], dtype=np.float32),
                matrixQ0=np.eye(stateDim, dtype=np.float32) * processVariance,
                intervalToBlockMap=np.zeros(2, dtype=np.int32), blockCount=1,
                stateInit=0.0, stateCovarInit=1.0e4, pad=0.0,
                ECM_useObsPrecisionReweighting=False,
                ECM_useProcessPrecisionReweighting=False,
                **stored, **extraArgs,
            )
            smoothed = backward(matrixData=data, **stored, **extraArgs)
            filteredVariance = float(stored["stateCovarForward"][0, 0, 0])
            nextVariance = float(stored["stateCovarForward"][1, 0, 0])
            noiseVariance = float(stored["pNoiseForward"][0, 0, 0])
            gain = filteredVariance / (filteredVariance + noiseVariance)
            expectedVariance = gain * noiseVariance + gain ** 2 * nextVariance
            assert smoothed[1][0, 0, 0] == pytest.approx(expectedVariance, rel=2.0e-7, abs=0.0)
            assert smoothed[2][0, 0, 0] == pytest.approx(gain * nextVariance, rel=2.0e-7, abs=0.0)
        if stateDim == 2:
            filteredCovariance = np.array([
                [[1.0, 0.0], [0.0, 1.0e-16]],
                [[0.5, 0.0], [0.0, 1.0e-16]],
            ], dtype=np.float32)
            rankDeficient = backward(
                matrixData=data, stateForward=np.zeros((2, 2), dtype=np.float32),
                stateCovarForward=filteredCovariance,
                pNoiseForward=np.zeros((2, 2, 2), dtype=np.float32), **extraArgs,
            )
            np.testing.assert_allclose(rankDeficient[1][0], filteredCovariance[1], rtol=1.0e-7, atol=0.0)
            np.testing.assert_allclose(rankDeficient[2][0], filteredCovariance[1], rtol=1.0e-7, atol=0.0)



@pytest.mark.correctness
def _caseInitialProcessNoiseSeedRecoversRandomWalkScale():
    rng = np.random.default_rng(122)
    n = 160
    m = 4
    qTrue = 1.0e-2
    obsVar = 2.0e-3
    latent = np.cumsum(rng.normal(0.0, np.sqrt(qTrue), size=n))
    matrixData = np.vstack(
        [latent + rng.normal(0.0, np.sqrt(obsVar), size=n) for _ in range(m)]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), obsVar, dtype=np.float32)

    matrixQ, diagnostics = core._estimateInitialProcessNoiseFromData(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        pad=1.0e-4,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        minQ=1.0e-5,
        maxQ=1.0,
        deltaF=1.0,
        robustTNu=8.0,
    )

    assert diagnostics["qSeedSource"] == "sameTrackEB"
    assert 0.3 * qTrue <= diagnostics["qSeedLevelFinal"] <= 3.0 * qTrue
    assert matrixQ[0, 0] == pytest.approx(diagnostics["qSeedLevelFinal"])
    assert matrixQ[1, 1] == pytest.approx(diagnostics["qSeedLevelFinal"])


@pytest.mark.correctness
def _caseInitialProcessNoiseSeedCapsDominantLowMuncArtifact():
    n = 30
    m = 5
    matrixData = np.zeros((m, n), dtype=np.float32)
    matrixData[0, n // 2 :] = 100.0
    matrixMunc = np.full((m, n), 0.1, dtype=np.float32)
    matrixMunc[0, :] = 1.0e-9

    matrixQ, diagnostics = core._estimateInitialProcessNoiseFromData(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        pad=1.0e-4,
        stateModel=core.STATE_MODEL_LEVEL,
        minQ=1.0e-5,
        maxQ=10.0,
        deltaF=1.0,
        robustTNu=8.0,
    )

    assert diagnostics["qSeedSource"] == "sameTrackEB"
    assert diagnostics["qSeedPrecisionCapFraction"] > 0.0
    assert diagnostics["qSeedLevelFinal"] < 1.0e-3
    assert matrixQ[0, 0] == pytest.approx(diagnostics["qSeedLevelFinal"])


@pytest.mark.correctness
def _caseInitialProcessNoiseSeedFallsBackToPooledEbForSparseOverlap():
    n = 20
    matrixData = np.full((2, n), np.nan, dtype=np.float32)
    matrixData[0, ::2] = 0.0
    matrixData[1, 1::2] = 1.0
    matrixMunc = np.full((2, n), 0.1, dtype=np.float32)

    matrixQ, diagnostics = core._estimateInitialProcessNoiseFromData(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        pad=1.0e-4,
        stateModel=core.STATE_MODEL_LEVEL,
        minQ=1.0e-4,
        maxQ=1.0,
        deltaF=1.0,
        robustTNu=8.0,
    )

    assert diagnostics["qSeedSource"] == "pooledEB"
    assert diagnostics["qSeedTransitionCount"] == n - 1
    assert np.isfinite(matrixQ).all()
    assert matrixQ[0, 0] > 1.0e-4


@pytest.mark.correctness
def _caseMuncSeedQKernelsMatchReference():
    base = np.linspace(-0.25, 0.95, 14, dtype=np.float64)
    matrixData = np.vstack(
        [
            base,
            base + 0.035,
            base - 0.025,
            base + 0.02 * np.sin(np.linspace(0.0, 2.0 * np.pi, base.size)),
        ]
    )
    matrixData[2, 5] = np.nan
    matrixData[3, 9] = np.nan
    obsVar = np.full(matrixData.shape, 0.04, dtype=np.float64)
    obsVar[0, :] = 1.0e-5
    active = np.isfinite(matrixData)

    def robustLocation(values, weights):
        x = np.asarray(values, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if x.size == 1:
            return float(x[0])
        loc = float(np.median(x))
        scale = 1.4826 * float(np.median(np.abs(x - loc)))
        if scale <= 1.0e-12:
            return loc
        c = 1.345
        for _ in range(4):
            resid = x - loc
            huber = np.minimum(1.0, (c * scale) / np.maximum(np.abs(resid), 1.0e-12))
            eff = w * huber
            nextLoc = float(np.sum(eff * x) / np.sum(eff))
            if abs(nextLoc - loc) <= 1.0e-10 * max(1.0, abs(loc)):
                loc = nextLoc
                break
            loc = nextLoc
        return loc

    def sampleIndices(itemCount, sampleCount):
        if sampleCount <= 0 or sampleCount >= itemCount:
            sampleCount = itemCount
        return np.asarray(
            [
                (((2 * sampleIndex) + 1) * itemCount) // (2 * sampleCount)
                for sampleIndex in range(sampleCount)
            ],
            dtype=np.intp,
        )

    def sameTrackReference(
        data,
        obs,
        activeRows,
        precisionCapQuantile,
        precisionCapMultiplier,
        maxTransitionSamples=0,
        precisionSampleCap=core._QINIT_PRECISION_SAMPLE_CAP,
    ):
        maxTransitionCount = activeRows.shape[1] - 1
        useTransitionSampling = (
            maxTransitionSamples > 0 and maxTransitionSamples < maxTransitionCount
        )
        transitionIndex = sampleIndices(maxTransitionCount, maxTransitionSamples)
        pairPrecisions = []
        perTransition = []
        for k in transitionIndex:
            rows = np.flatnonzero(activeRows[:, k] & activeRows[:, k + 1])
            rawPrecision = 1.0 / (obs[rows, k] + obs[rows, k + 1])
            pairPrecisions.extend(rawPrecision.tolist())
            perTransition.append((k, rows, rawPrecision))
        positivePrecision = np.asarray(pairPrecisions, dtype=np.float64)
        precisionForCap = positivePrecision
        if (
            useTransitionSampling
            and precisionSampleCap > 0
            and precisionForCap.size > precisionSampleCap
        ):
            precisionForCap = precisionForCap[
                sampleIndices(precisionForCap.size, precisionSampleCap)
            ]
        cap = min(
            float(np.quantile(precisionForCap, float(precisionCapQuantile))),
            float(precisionCapMultiplier) * float(np.median(precisionForCap)),
        )
        deltaValues = []
        samplingValues = []
        weightValues = []
        pairCount = 0
        for k, rows, rawPrecision in perTransition:
            if rows.size == 0:
                continue
            d = data[rows, k + 1] - data[rows, k]
            p = np.minimum(rawPrecision, cap)
            pairCount += int(d.size)
            sumP = float(np.sum(p))
            sumP2 = float(np.sum(p * p))
            deltaValues.append(robustLocation(d, p))
            samplingValues.append(1.0 / sumP)
            weightValues.append(max((sumP * sumP) / sumP2, 1.0))
        return (
            np.asarray(deltaValues, dtype=np.float64),
            np.asarray(samplingValues, dtype=np.float64),
            np.asarray(weightValues, dtype=np.float64),
            {
                "pairCount": pairCount,
                "sampledPairCount": int(positivePrecision.size),
                "precisionSamplePairCount": int(precisionForCap.size),
                "precisionCap": cap,
                "precisionCapFraction": float(np.mean(positivePrecision > cap)),
                "sampledTransitionCount": int(transitionIndex.size),
                "transitionSampleFraction": float(
                    transitionIndex.size / maxTransitionCount
                ),
                "precisionSampleCap": int(precisionSampleCap),
                "maxTransitionSamples": int(maxTransitionSamples),
                "sampledTransitionIndices": transitionIndex.tolist()
                if transitionIndex.size <= 1024
                else None,
            },
        )

    def pooledReference(data, obs, activeRows):
        weights = np.where(activeRows, 1.0 / obs, 0.0)
        weightSum = np.sum(weights, axis=0, dtype=np.float64)
        pooledMean = np.divide(
            np.sum(np.where(activeRows, data * weights, 0.0), axis=0, dtype=np.float64),
            weightSum,
            out=np.full(data.shape[1], np.nan, dtype=np.float64),
            where=weightSum > 0.0,
        )
        pooledVar = np.divide(
            1.0,
            weightSum,
            out=np.full(data.shape[1], np.nan, dtype=np.float64),
            where=weightSum > 0.0,
        )
        valid = (
            np.isfinite(pooledMean[1:])
            & np.isfinite(pooledMean[:-1])
            & np.isfinite(pooledVar[1:])
            & np.isfinite(pooledVar[:-1])
        )
        sampling = pooledVar[1:][valid] + pooledVar[:-1][valid]
        return (
            pooledMean[1:][valid] - pooledMean[:-1][valid],
            sampling,
            1.0 / np.maximum(sampling, np.finfo(np.float64).tiny),
        )

    def weightedQuantile(values, weights, q):
        return float(
            normalization.weightedQuantileInterpolated(
                np.asarray(values, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
                np.asarray([q], dtype=np.float64),
            )[0]
        )

    def posteriorReference(
        deltas, sampling, weights, source, gridSize=core._QINIT_GRID_SIZE, qCap=1.0
    ):
        qFloor = 1.0e-5
        delta = np.asarray(deltas, dtype=np.float64)
        s2 = np.asarray(sampling, dtype=np.float64)
        transitionWeight = np.asarray(weights, dtype=np.float64)
        sumW = float(np.sum(transitionWeight))
        sumW2 = float(np.sum(transitionWeight * transitionWeight))
        effectiveCount = (sumW * sumW) / sumW2
        center = weightedQuantile(delta, transitionWeight, 0.5)
        robustScale = 1.4826 * weightedQuantile(
            np.abs(delta - center), transitionWeight, 0.5
        )
        medianS2 = weightedQuantile(s2, transitionWeight, 0.5)
        qPrior = max(robustScale * robustScale - medianS2, qFloor)
        deconvolved = np.maximum(delta * delta - s2, 0.0)
        qTransition90 = weightedQuantile(deconvolved, transitionWeight, 0.9)
        grid = np.exp(np.linspace(math.log(qFloor), math.log(qCap), gridSize))
        nu = 8.0
        normalizedWeights = transitionWeight / max(
            weightedQuantile(transitionWeight, transitionWeight, 0.5),
            np.finfo(np.float64).tiny,
        )
        normalizedWeights = np.clip(normalizedWeights, 0.25, 4.0)
        logPriorCenter = math.log(max(qPrior, qFloor))
        logPriorSd = core._QINIT_PRIOR_LOG_SD
        logPost = np.empty(grid.shape, dtype=np.float64)
        for idx, q in enumerate(grid):
            var = np.maximum(q + s2, np.finfo(np.float64).tiny)
            scale = np.sqrt(var)
            logLike = stats.t.logpdf(delta / scale, df=nu) - np.log(scale)
            logPrior = -0.5 * ((math.log(q) - logPriorCenter) / logPriorSd) ** 2
            logPost[idx] = float(np.sum(normalizedWeights * logLike) + logPrior)
        posterior = np.exp(logPost - np.max(logPost))
        posterior = posterior / float(np.sum(posterior))
        cdf = np.cumsum(posterior)
        return {
            "ok": True,
            "source": source,
            "reason": "ok",
            "transitionCount": int(delta.size),
            "effectiveTransitionCount": float(effectiveCount),
            "medianSamplingVariance": float(medianS2),
            "priorLevel": float(qPrior),
            "posteriorModeLevel": float(grid[int(np.argmax(posterior))]),
            "posteriorMedianLevel": float(np.interp(0.5, cdf, grid)),
            "posteriorQ05Level": float(np.interp(0.05, cdf, grid)),
            "posteriorQ95Level": float(np.interp(0.95, cdf, grid)),
            "transitionQ90": float(qTransition90),
        }

    sameExpected = sameTrackReference(
        matrixData,
        obsVar,
        active,
        core._QINIT_PRECISION_CAP_QUANTILE,
        core._QINIT_PRECISION_CAP_MULTIPLIER,
    )
    sameActual = cconsenrich.cEstimateSameTrackProcessNoiseTransitions(
        matrixData,
        obsVar,
        active,
        core._QINIT_PRECISION_CAP_QUANTILE,
        core._QINIT_PRECISION_CAP_MULTIPLIER,
    )
    sameWrapper = core._estimateSameTrackProcessNoiseTransitions(
        matrixData=matrixData,
        obsVar=obsVar,
        finiteMask=active,
    )
    for actual in (sameActual, sameWrapper):
        np.testing.assert_allclose(actual[0], sameExpected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual[1], sameExpected[1], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual[2], sameExpected[2], rtol=2e-12, atol=2e-12)
        assert actual[3]["pairCount"] == sameExpected[3]["pairCount"]
        assert actual[3]["precisionCap"] == pytest.approx(
            sameExpected[3]["precisionCap"], rel=2e-12, abs=2e-12
        )
        assert actual[3]["precisionCapFraction"] == pytest.approx(
            sameExpected[3]["precisionCapFraction"], rel=2e-12, abs=2e-12
        )

    uncappedExpected = sameTrackReference(matrixData, obsVar, active, 1.0, 1.0e12)
    uncappedActual = cconsenrich.cEstimateSameTrackProcessNoiseTransitions(
        matrixData,
        obsVar,
        active,
        1.0,
        1.0e12,
        0,
    )
    np.testing.assert_allclose(
        uncappedActual[0], uncappedExpected[0], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        uncappedActual[1], uncappedExpected[1], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        uncappedActual[2], uncappedExpected[2], rtol=2e-12, atol=2e-12
    )
    assert uncappedActual[3]["pairCount"] == uncappedExpected[3]["pairCount"]
    assert uncappedActual[3]["precisionCapFraction"] == pytest.approx(0.0)

    sampleBase = np.linspace(-0.2, 1.4, 33, dtype=np.float64)
    sampleData = np.vstack(
        [
            sampleBase + 0.02 * np.sin(3.0 * sampleBase),
            sampleBase + 0.05 + 0.03 * np.cos(2.0 * sampleBase),
            sampleBase - 0.04 + 0.01 * np.sin(5.0 * sampleBase),
        ]
    )
    sampleObs = np.full(sampleData.shape, 0.035, dtype=np.float64)
    sampleActive = np.ones(sampleData.shape, dtype=bool)
    sampleCap = 8
    midpointIndex = np.floor(
        (np.arange(sampleCap, dtype=np.float64) + 0.5)
        * (sampleData.shape[1] - 1)
        / sampleCap
    ).astype(np.int64)
    fullExpected = sameTrackReference(sampleData, sampleObs, sampleActive, 1.0, 1.0e12)
    sampledExpected = sameTrackReference(
        sampleData,
        sampleObs,
        sampleActive,
        1.0,
        1.0e12,
        sampleCap,
        precisionSampleCap=7,
    )
    sampledActual = cconsenrich.cEstimateSameTrackProcessNoiseTransitions(
        sampleData,
        sampleObs,
        sampleActive,
        1.0,
        1.0e12,
        sampleCap,
        7,
    )
    assert sampledActual[0].shape == (sampleCap,)
    assert sampledActual[3]["pairCount"] == sampleData.shape[0] * sampleCap
    assert sampledActual[3]["sampledPairCount"] == sampledExpected[3][
        "sampledPairCount"
    ]
    assert sampledActual[3]["precisionSamplePairCount"] == 7
    assert sampledActual[3]["precisionSampleCap"] == 7
    assert sampledActual[3]["sampledTransitionIndices"] == midpointIndex.tolist()
    np.testing.assert_allclose(
        sampledActual[0], fullExpected[0][midpointIndex], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        sampledActual[0], sampledExpected[0], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        sampledActual[1], sampledExpected[1], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        sampledActual[2], sampledExpected[2], rtol=2e-12, atol=2e-12
    )

    pooledExpected = pooledReference(matrixData, obsVar, active)
    pooledActual = cconsenrich.cEstimatePooledProcessNoiseTransitions(
        matrixData,
        obsVar,
        active,
    )
    pooledWrapper = core._estimatePooledProcessNoiseTransitions(
        matrixData=matrixData,
        obsVar=obsVar,
        finiteMask=active,
    )
    for actual in (pooledActual, pooledWrapper):
        np.testing.assert_allclose(actual[0], pooledExpected[0], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual[1], pooledExpected[1], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual[2], pooledExpected[2], rtol=2e-12, atol=2e-12)

    posteriorExpected = posteriorReference(
        sameExpected[0], sameExpected[1], sameExpected[2], "sameTrackEB"
    )
    posteriorActual = core._qSeedPosteriorFromTransitions(
        deltas=sameActual[0],
        samplingVariances=sameActual[1],
        transitionWeights=sameActual[2],
        qFloor=1.0e-5,
        qCap=1.0,
        robustTNu=8.0,
        source="sameTrackEB",
        qSeedPriorLevel=1.0e-5,
    )
    for key in ("ok", "source", "reason", "transitionCount"):
        assert posteriorActual[key] == posteriorExpected[key]
    for key in (
        "effectiveTransitionCount",
        "medianSamplingVariance",
        "priorLevel",
        "posteriorModeLevel",
        "posteriorMedianLevel",
        "posteriorQ05Level",
        "posteriorQ95Level",
        "transitionQ90",
    ):
        assert posteriorActual[key] == pytest.approx(
            posteriorExpected[key], rel=1e-8, abs=1e-12
        )

    posteriorCalls = []
    nativePosterior = cconsenrich.cQSeedPosteriorFromTransitions

    def spyPosterior(*args):
        posteriorCalls.append(args)
        return nativePosterior(*args)

    try:
        cconsenrich.cQSeedPosteriorFromTransitions = spyPosterior
        core._qSeedPosteriorFromTransitions(
            deltas=sameActual[0],
            samplingVariances=sameActual[1],
            transitionWeights=sameActual[2],
            qFloor=1.0e-5,
            qCap=1.0,
            robustTNu=8.0,
            source="sameTrackEB",
            qSeedPriorLevel=1.0e-5,
        )
    finally:
        cconsenrich.cQSeedPosteriorFromTransitions = nativePosterior
    assert posteriorCalls[0][-1] == core._QINIT_GRID_SIZE

    syntheticCount = 96
    syntheticX = (np.arange(syntheticCount, dtype=np.float64) + 0.5) / syntheticCount
    syntheticDeltas = (
        0.05 * np.sin(2.0 * np.pi * syntheticX)
        + 0.0175 * np.sin(6.0 * np.pi * syntheticX)
        + 0.006 * np.cos(10.0 * np.pi * syntheticX)
    )
    syntheticSampling = np.full(syntheticCount, 1.0e-4, dtype=np.float64)
    syntheticWeights = 1.0 + 0.35 * np.cos(2.0 * np.pi * syntheticX)
    synthetic64 = cconsenrich.cQSeedPosteriorFromTransitions(
        syntheticDeltas,
        syntheticSampling,
        syntheticWeights,
        1.0e-5,
        1.0e-2,
        8.0,
        "sameTrackEB",
        1.0e-5,
        core._QINIT_MIN_TRANSITIONS,
        core._QINIT_PRIOR_LOG_SD,
        core._QINIT_DEFAULT_T_NU,
        core._QINIT_GRID_SIZE,
    )
    synthetic256 = cconsenrich.cQSeedPosteriorFromTransitions(
        syntheticDeltas,
        syntheticSampling,
        syntheticWeights,
        1.0e-5,
        1.0e-2,
        8.0,
        "sameTrackEB",
        1.0e-5,
        core._QINIT_MIN_TRANSITIONS,
        core._QINIT_PRIOR_LOG_SD,
        core._QINIT_DEFAULT_T_NU,
        256,
    )
    for key in ("ok", "source", "reason", "transitionCount"):
        assert synthetic64[key] == synthetic256[key]
    for key in (
        "posteriorMedianLevel",
        "posteriorModeLevel",
        "posteriorQ05Level",
        "posteriorQ95Level",
    ):
        assert synthetic64[key] == pytest.approx(synthetic256[key], rel=0.35)

    with pytest.raises(ValueError, match="samplingVariances"):
        cconsenrich.cQSeedPosteriorFromTransitions(
            np.ones(8, dtype=np.float64),
            -np.ones(8, dtype=np.float64),
            np.ones(8, dtype=np.float64),
            1.0e-5,
            1.0,
            8.0,
            "bad",
            1.0e-5,
            core._QINIT_MIN_TRANSITIONS,
            core._QINIT_PRIOR_LOG_SD,
            core._QINIT_DEFAULT_T_NU,
            core._QINIT_GRID_SIZE,
        )
@pytest.mark.correctness
def _caseRunConsenrichOuterPassSmoke():
    rng = np.random.default_rng(0)
    n = 64
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signalTrack = np.sin(grid).astype(np.float32)
    backgroundTrack = np.linspace(-0.25, 0.25, n, dtype=np.float32)
    matrixData = np.vstack(
        [
            signalTrack + backgroundTrack + 0.05 * rng.normal(size=n) - 0.04,
            signalTrack + backgroundTrack + 0.05 * rng.normal(size=n),
            signalTrack + backgroundTrack + 0.05 * rng.normal(size=n) + 0.03,
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.2, dtype=np.float32)
    assert core._signChangePerKB(
        np.asarray([1.0, -0.005, 1.0], dtype=np.float32),
        intervalSizeBP=1000,
    ) == pytest.approx(0.0)
    assert core._signChangePerKB(
        np.asarray([1.0, -0.02, 1.0], dtype=np.float32),
        intervalSizeBP=1000,
    ) == pytest.approx(2.0 / 3.0)
    assert core._relativeSignChangePerKB(
        np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        np.asarray([[1.0, 3.0, 1.0], [3.0, 1.0, 3.0]], dtype=np.float32),
        np.asarray([[0.1, 0.1, 0.1], [10.0, 10.0, 10.0]], dtype=np.float32),
        intervalSizeBP=1000,
        pad=0.0,
    ) == pytest.approx(2.0 / 3.0)
    relativeData = np.asarray(
        [[1.0, 3.0, 1.0], [3.0, 1.0, 3.0]],
        dtype=np.float32,
    )
    relativeBackground = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    assert core._relativeSignChangePerKB(
        np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
        relativeData + relativeBackground[None, :],
        np.asarray([[0.1, 0.1, 0.1], [10.0, 10.0, 10.0]], dtype=np.float32),
        intervalSizeBP=1000,
        background=relativeBackground,
        pad=0.0,
    ) == pytest.approx(2.0 / 3.0)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-6,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        intervalSizeBP=1000,
        ECM_fixedBackgroundIters=3,
        ECM_outerIters=2,
        processNoiseWarmupECMIters=1,
        ECM_useObsPrecisionReweighting=True,
        trackOptimizationPath=True,
        returnPrecisionDiagnostics=True,
        returnDiagnostics=True,
    )

    (
        stateSmoothed,
        stateCovarSmoothed,
        postFitResiduals,
        NIS,
        *_rest,
        precisionDiagnostics,
        diagnostics,
    ) = out
    assert len(out) == 7
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert precisionDiagnostics["precision_track_diagnostics"] is True
    outputTracks = precisionDiagnostics["outputTracks"]
    assert tuple(sorted(outputTracks)) == (
        "baseQLevel",
        "baseQTrend",
        "effectiveQLevel",
        "effectiveQTrend",
        "muncTrace",
        "preKappaQLevel",
        "preKappaQTrend",
        "sumGain0",
        "sumGain1",
    )
    for track in outputTracks.values():
        assert np.asarray(track).shape == (n,)
    assert diagnostics["final_forward_nis"] == pytest.approx(float(np.mean(NIS)))
    qDiagnostics = diagnostics["process_q_diagnostics"]
    assert qDiagnostics["effectiveQTraceMin"] <= qDiagnostics["effectiveQTraceMedian"]
    assert qDiagnostics["effectiveQTraceMedian"] <= qDiagnostics["effectiveQTraceMax"]
    obsTrace = diagnostics["observation_r_trace"]
    expectedObservationTrace = m * (0.2 + 0.0001)
    assert obsTrace["min"] == pytest.approx(expectedObservationTrace)
    assert obsTrace["median"] == pytest.approx(expectedObservationTrace)
    assert obsTrace["max"] == pytest.approx(expectedObservationTrace)
    lambdaTrack = np.asarray(precisionDiagnostics["lambdaExp"], dtype=np.float64)
    processPrecisionTrack = np.asarray(
        precisionDiagnostics["processPrecExp"],
        dtype=np.float64,
    )
    assert lambdaTrack.shape == (n,)
    assert processPrecisionTrack.shape == (n,)
    np.testing.assert_allclose(
        outputTracks["muncTrace"],
        expectedObservationTrace / lambdaTrack,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    gainSummary = diagnostics["final_forward_gain_contig_summary"]
    assert len(gainSummary["mean"]) == m
    assert len(gainSummary["median"]) == m
    assert len(gainSummary["sd"]) == m
    assert len(gainSummary["iqr"]) == m
    assert len(gainSummary["count"]) == m
    assert all(value >= 0.0 for value in gainSummary["sd"])
    assert all(value >= 0.0 for value in gainSummary["iqr"])
    assert all(value >= 0 for value in gainSummary["count"])
    assert "background_prior" not in diagnostics
    fitDiagnostics = diagnostics["post_process_noise_fit"]["fixed_background_ecm"]
    assert fitDiagnostics
    backgroundPassDiagnostics = [
        item for item in fitDiagnostics if not item.get("final_fixed_background_ecm")
    ]
    assert backgroundPassDiagnostics
    assert "observation_lambda_mean" in backgroundPassDiagnostics[-1]
    assert "observation_lambda_median" in backgroundPassDiagnostics[-1]
    assert "background_objective_per_cell" in backgroundPassDiagnostics[-1]
    assert "background_objective_change_per_cell" in backgroundPassDiagnostics[-1]
    assert "background_objective_threshold_per_cell" in backgroundPassDiagnostics[-1]
    assert "observation_lambda_lower_bound_hits" in backgroundPassDiagnostics[-1]
    assert "observation_lambda_upper_bound_hits" in backgroundPassDiagnostics[-1]
    assert "process_kappa_lower_bound_hits" in backgroundPassDiagnostics[-1]
    assert "process_kappa_upper_bound_hits" in backgroundPassDiagnostics[-1]
    assert (
        0.0
        <= backgroundPassDiagnostics[-1]["observation_lambda_lower_bound_hits"]
        <= 1.0
    )
    assert (
        0.0
        <= backgroundPassDiagnostics[-1]["observation_lambda_upper_bound_hits"]
        <= 1.0
    )
    assert 0.0 <= backgroundPassDiagnostics[-1]["process_kappa_lower_bound_hits"] <= 1.0
    assert 0.0 <= backgroundPassDiagnostics[-1]["process_kappa_upper_bound_hits"] <= 1.0
    assert backgroundPassDiagnostics[-1]["relative_sign_change_per_kb"] >= 0.0
    finalFixedDiagnostics = fitDiagnostics[-1]
    assert finalFixedDiagnostics["final_fixed_background_ecm"] is True
    assert "final_abs_rel_change" in finalFixedDiagnostics
    assert "stable_iters" in finalFixedDiagnostics
    assert "patience_target" in finalFixedDiagnostics
    assert finalFixedDiagnostics["relative_sign_change_per_kb"] >= 0.0
    assert "background_objective_per_cell" not in finalFixedDiagnostics
    assert "outer_objective_per_cell" not in finalFixedDiagnostics
    traceRows = finalFixedDiagnostics["optimization_path"]
    assert traceRows
    assert [row["iter"] for row in traceRows] == sorted(
        row["iter"] for row in traceRows
    )
    assert traceRows[0]["reset_iteration"] is True
    assert traceRows[0]["change"] is None
    assert traceRows[0]["threshold"] is None

    backgroundOut = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-6,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        processNoiseWarmupECMIters=1,
        returnBackground=True,
    )
    assert len(backgroundOut) == 6
    background = np.asarray(backgroundOut[-1])
    assert background.shape == (n,)
    assert np.isfinite(background).all()
    assert all(np.isfinite(float(row["objective_value"])) for row in traceRows)
    assert all(row["objective_name"] == "nll" for row in traceRows)
    qInfo = diagnostics["process_noise_calibration"]
    assert qInfo["processNoisePolicy"] == "fixedDiagonal"
    assert qInfo["processNoiseCalibrationStatus"] == "estimated"
    assert "".join(("block", "Mode")) not in qInfo
    assert "_".join(("process", "q", "calibration")) not in diagnostics
    expectedLevelQ = qInfo["qSeedLevelFinal"]
    expectedTrendQ = qInfo["qSeedTrendFinal"]
    assert qInfo["processNoiseCalibrationReason"] == "data_derived_q_estimate"
    assert qInfo["preKappaQLevel"] == pytest.approx(expectedLevelQ, rel=5.0e-6)
    assert qInfo["preKappaQTrend"] == pytest.approx(expectedTrendQ, rel=5.0e-6)
    np.testing.assert_allclose(
        qInfo["matrixQ0Final"],
        np.diag([expectedLevelQ, expectedTrendQ]),
        rtol=5.0e-6,
    )
@pytest.mark.correctness
def _caseRunConsenrichLevelStateModelSmoke():
    rng = np.random.default_rng(100)
    n = 42
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signalTrack = (0.4 * np.sin(grid) + 0.15 * np.cos(2.0 * grid)).astype(np.float32)
    matrixData = np.vstack(
        [
            signalTrack + 0.04 * rng.normal(size=n) - 0.02,
            signalTrack + 0.04 * rng.normal(size=n),
            signalTrack + 0.04 * rng.normal(size=n) + 0.03,
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.10, dtype=np.float32)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        stateModel="level",
        deltaF=-10.0,
        minQ=1.0e-4,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=7,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        ECM_useProcessPrecisionReweighting=True,
        processNoiseWarmupECMIters=1,
        returnDiagnostics=True,
    )

    (
        stateSmoothed,
        stateCovarSmoothed,
        postFitResiduals,
        NIS,
        _blockMap,
        runDiagnostics,
    ) = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.all(np.isfinite(stateCovarSmoothed))
    assert np.all(np.isfinite(NIS))
    np.testing.assert_array_equal(stateSmoothed[:, 1], np.zeros(n, dtype=np.float32))
    np.testing.assert_array_equal(
        stateCovarSmoothed[:, 0, 1], np.zeros(n, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        stateCovarSmoothed[:, 1, 0], np.zeros(n, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        stateCovarSmoothed[:, 1, 1], np.zeros(n, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        core.getPrimaryState(stateSmoothed),
        np.round(stateSmoothed[:, 0].astype(np.float32), decimals=5),
    )
    qInfo = runDiagnostics["process_noise_calibration"]
    assert runDiagnostics["state_model"] == core.STATE_MODEL_LEVEL
    assert qInfo["processNoisePolicy"] == "fixedDiagonal"
    expectedLevelQ = qInfo["qSeedLevelFinal"]
    assert qInfo["processNoiseCalibrationReason"] == "data_derived_q_estimate"
    assert qInfo["preKappaQLevel"] == pytest.approx(expectedLevelQ, rel=5.0e-6)
    assert qInfo["preKappaQTrend"] == pytest.approx(0.0)
    assert qInfo["effectiveTrendLevelRatio"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        qInfo["matrixQ0Final"],
        [[expectedLevelQ]],
        rtol=5.0e-6,
    )


@pytest.mark.correctness
def _caseRunConsenrichInitialProcessQSkipsWarmup(monkeypatch):
    rng = np.random.default_rng(17)
    n = 30
    m = 3
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    matrixData = np.vstack(
        [grid + 0.02 * rng.normal(size=n) + offset for offset in (-0.01, 0.0, 0.01)]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.10, dtype=np.float32)

    originalECM = cconsenrich.cfixedBackgroundECM
    ecmModes = []

    def _spyECM(*args, **kwargs):
        result = originalECM(*args, **kwargs)
        ecmModes.append(bool(kwargs.get("ECM_useProcessPrecisionReweighting")))
        return result

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", _spyECM)

    def failSeedEstimator(**_kwargs):
        raise AssertionError("explicit process Q must bypass data estimation")

    monkeypatch.setattr(core, "_estimateInitialProcessNoiseFromData", failSeedEstimator)

    initialQ = np.diag([1.0e-3, 1.0e-4]).astype(np.float32)
    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=1.0,
        minQ=1.0e-4,
        maxQ=0.5,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_fixedBackgroundIters=2,
        ECM_useProcessPrecisionReweighting=True,
        fitBackground=False,
        initialProcessQ=initialQ,
        returnDiagnostics=True,
    )

    diagnostics = out[-1]
    assert ecmModes == [True]
    qInfo = diagnostics["process_noise_calibration"]
    assert qInfo["processNoiseCalibrationStatus"] == "skipped"
    assert qInfo["processNoiseCalibrationReason"] == "initial_process_q"
    assert qInfo["warmStartProcessNoise"] == 1.0
    np.testing.assert_allclose(
        qInfo["matrixQ0Final"],
        initialQ,
        rtol=0.0,
        atol=0.0,
    )
    assert diagnostics["post_process_noise_fit"]["warm_start"]["background"] is False


@pytest.mark.correctness
def _caseRunConsenrichFixedDiagonalUsesDataQ(monkeypatch):
    rng = np.random.default_rng(3)
    n = 36
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    matrixData = np.vstack(
        [
            np.sin(grid) + 0.2 * rng.normal(size=n) + offset
            for offset in (-0.1, 0.0, 0.1)
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.05, dtype=np.float32)

    seedQ = np.diag([2.0e-5, 3.0e-5]).astype(np.float32)
    seedCalls = []

    def fakeSeedEstimator(**kwargs):
        seedCalls.append(kwargs)
        return seedQ.copy(), {
            "qSeedSource": "test",
            "qSeedTransitionCount": n - 1,
            "qSeedLevelFinal": float(seedQ[0, 0]),
            "qSeedTrendFinal": float(seedQ[1, 1]),
        }

    monkeypatch.setattr(core, "_estimateInitialProcessNoiseFromData", fakeSeedEstimator)

    originalECM = cconsenrich.cfixedBackgroundECM
    ecmCalls = []

    def spyECM(*args, **kwargs):
        ecmCalls.append(
            {
                "matrixQ0": np.asarray(kwargs["matrixQ0"], dtype=np.float64).copy(),
                "useProcPrec": bool(kwargs["ECM_useProcessPrecisionReweighting"]),
            }
        )
        return originalECM(*args, **kwargs)

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", spyECM)

    initialKappa = np.linspace(0.5, 1.8, n, dtype=np.float32)
    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.2,
        minQ=1.0e-6,
        maxQ=1.0,
        processNoiseCalibration=core.PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=6,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        fitBackground=False,
        ECM_useProcessPrecisionReweighting=True,
        initialProcessPrecision=initialKappa,
        qSeedPriorLevel=7.0e-6,
        returnPrecisionDiagnostics=True,
        returnDiagnostics=True,
    )

    precisionDiagnostics = out[-2]
    runDiagnostics = out[-1]
    qInfo = runDiagnostics["process_noise_calibration"]
    assert len(seedCalls) == 1
    seedCall = seedCalls[0]
    np.testing.assert_array_equal(seedCall["matrixData"], matrixData)
    np.testing.assert_array_equal(seedCall["matrixMunc"], matrixMunc)
    assert seedCall["pad"] == pytest.approx(1.0e-4)
    assert seedCall["stateModel"] == core.STATE_MODEL_LEVEL_TREND
    assert seedCall["minQ"] == pytest.approx(1.0e-6)
    assert seedCall["maxQ"] == pytest.approx(1.0)
    assert seedCall["deltaF"] == pytest.approx(0.2)
    assert (
        seedCall["robustTNu"]
        == constants.FIT_DEFAULT_ROBUST_T_NU
        == pytest.approx(8.0)
    )
    assert seedCall["qSeedPriorLevel"] == pytest.approx(7.0e-6)
    assert len(ecmCalls) == 1
    np.testing.assert_allclose(
        ecmCalls[0]["matrixQ0"],
        seedQ,
        rtol=0.0,
        atol=1.0e-10,
    )
    assert ecmCalls[0]["useProcPrec"] is True
    assert "process_noise_warmup_fit" not in runDiagnostics
    assert qInfo["processNoisePolicy"] == "fixedDiagonal"
    assert qInfo["processNoiseCalibrationStatus"] == "estimated"
    assert qInfo["processNoiseCalibrationReason"] == "data_derived_q_estimate"
    assert qInfo["qSeedSource"] == "test"
    assert qInfo["validTransitionCount"] == n - 1
    np.testing.assert_allclose(
        qInfo["matrixQ0Final"],
        seedQ,
        rtol=0.0,
        atol=1.0e-10,
    )
    outputTracks = precisionDiagnostics["outputTracks"]
    np.testing.assert_allclose(outputTracks["baseQLevel"], seedQ[0, 0])
    np.testing.assert_allclose(outputTracks["baseQTrend"], seedQ[1, 1])
    np.testing.assert_allclose(outputTracks["preKappaQLevel"], seedQ[0, 0])
    np.testing.assert_allclose(outputTracks["preKappaQTrend"], seedQ[1, 1])
    assert np.any(
        np.abs(outputTracks["effectiveQLevel"] - outputTracks["preKappaQLevel"])
        > 1.0e-9
    )


@pytest.mark.correctness
def test_core_estimate_provisional_background_helper():
    n = 40
    background = np.full(n, 1.75, dtype=np.float32)
    matrixData = np.broadcast_to(background[None, :], (3, n)).astype(np.float32)
    matrixMunc = np.full_like(matrixData, 0.05, dtype=np.float32)

    estimated, diagnostics = core.estimateProvisionalBackground(
        matrixData,
        matrixMunc,
        blockLenIntervals=8,
        useNonnegativeBackground=False,
        zeroCenterBackground=False,
        returnDiagnostics=True,
    )

    assert estimated.shape == (n,)
    assert np.all(np.isfinite(estimated))
    assert np.max(np.abs(estimated - background)) < 1.0e-4
    assert diagnostics["applied"] is True
    assert diagnostics["source"] == "banded_weighted_data"


@pytest.mark.correctness
def test_core_run_consenrich_initial_background_reaches_final_fit():
    rng = np.random.default_rng(41)
    n = 24
    m = 2
    initialBackground = np.linspace(0.20, 0.35, n, dtype=np.float32)
    latent = np.linspace(0.0, 0.1, n, dtype=np.float32)
    matrixData = (
        initialBackground[None, :]
        + latent[None, :]
        + 0.005 * rng.normal(size=(m, n))
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.08, dtype=np.float32)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=1.0,
        minQ=1.0e-4,
        maxQ=0.5,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=6,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        processNoiseWarmupECMIters=1,
        initialBackground=initialBackground,
        returnDiagnostics=True,
    )

    diagnostics = out[-1]
    warmStart = diagnostics["post_process_noise_fit"]["warm_start"]
    assert warmStart["background"] is True
    assert warmStart["background_prepass"] is False


@pytest.mark.correctness
def test_core_run_consenrich_weighted_rms_shift_gate(monkeypatch):
    matrixData = np.asarray([[100.0, 0.0]], dtype=np.float32)
    matrixMunc = np.asarray([[0.01, 1.0]], dtype=np.float32)
    seedG = np.asarray([100.0, 0.0], dtype=np.float32)
    proposalG = np.asarray([100.1, 10.0], dtype=np.float32)
    capturedWeights = []

    def fakeECM(**kwargs):
        trackCount, intervalCount = kwargs["matrixDataLocal"].shape
        stateDim = 1 if kwargs["stateModelMode"] == core.STATE_MODEL_LEVEL else 2
        state = np.zeros((intervalCount, stateDim), dtype=np.float32)
        covar = np.zeros((intervalCount, stateDim, stateDim), dtype=np.float32)
        residuals = np.zeros((trackCount, intervalCount), dtype=np.float32)
        return core._FixedBackgroundECMResult(
            iters_done=1,
            nll=1.0,
            state_smoothed=state,
            state_covar_smoothed=covar,
            lag_covar_smoothed=covar,
            post_fit_residuals=residuals,
            lambda_exp=None,
            process_prec_exp=None,
            diagnostics={"converged": True, "nll_increase_count": 0},
        )

    def fakeSolve(**kwargs):
        capturedWeights.append(np.asarray(kwargs["weightTrack"], dtype=np.float64))
        return proposalG

    monkeypatch.setattr(core, "_runFixedBackgroundECMPhase", fakeECM)
    monkeypatch.setattr(core, "solveZeroCenteredBackground", fakeSolve)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=1.0,
        minQ=1.0e-4,
        maxQ=0.5,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=2,
        pad=0.0,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        ECM_backgroundShiftRtol=0.05,
        initialBackground=seedG,
        initialProcessQ=np.diag([1.0e-3, 1.0e-4]).astype(np.float32),
        useNonnegativeBackground=False,
        returnDiagnostics=True,
    )

    weights = np.asarray([100.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(capturedWeights, [weights], rtol=0.0, atol=1.0e-10)
    shiftDelta = proposalG.astype(np.float64) - seedG.astype(np.float64)
    expectedShift = math.sqrt(float(np.dot(weights, shiftDelta * shiftDelta)) / 101.0)
    expectedScale = max(
        math.sqrt(float(np.dot(weights, proposalG * proposalG)) / 101.0),
        math.sqrt(float(np.dot(weights, seedG * seedG)) / 101.0),
        1.0,
    )
    maxShift = float(np.max(np.abs(shiftDelta)))
    maxThreshold = 0.05 * max(
        float(np.max(np.abs(proposalG))),
        float(np.max(np.abs(seedG))),
        1.0,
    )

    loopDiagnostics = out[-1]["post_process_noise_fit"]["fixed_background_ecm"][0]
    assert loopDiagnostics["background_shift"] == pytest.approx(expectedShift)
    assert loopDiagnostics["background_shift_threshold"] == pytest.approx(
        0.05 * expectedScale
    )
    assert loopDiagnostics["background_shift_stable"] is True
    assert maxShift > maxThreshold


@pytest.mark.correctness
def _caseRunConsenrichOuterPassRequiresThreeIterationsDespiteTolerance(monkeypatch):
    rng = np.random.default_rng(23)
    n = 32
    m = 3
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    matrixData = np.vstack(
        [
            grid + 0.02 * rng.normal(size=n) - 0.01,
            grid + 0.02 * rng.normal(size=n),
            grid + 0.02 * rng.normal(size=n) + 0.01,
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.2, dtype=np.float32)

    originalECM = cconsenrich.cfixedBackgroundECM
    calls = []

    def _spyECM(*args, **kwargs):
        result = originalECM(*args, **kwargs)
        calls.append(1)
        return result

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", _spyECM)

    commonKwargs = dict(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        deltaF=0.2,
        minQ=1.0e-3,
        maxQ=0.5,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_fixedBackgroundIters=3,
        ECM_fixedBackgroundRtol=1.0e9,
        ECM_backgroundShiftRtol=1.0e9,
        initialProcessQ=np.diag([1.0e-3, 1.0e-5]).astype(np.float32),
    )

    out = core.runConsenrich(
        **commonKwargs,
        ECM_outerIters=5,
        ECM_outerNLLRtol=1.0e9,
        returnDiagnostics=True,
    )
    assert len(calls) == 4
    assert out[-1]["post_process_noise_fit"]["actual_outer_passes"] == 3
    assert (
        out[-1]["post_process_noise_fit"]["fixed_background_ecm"][-1][
            "final_fixed_background_ecm"
        ]
        is True
    )

    calls.clear()
    core.runConsenrich(**commonKwargs, ECM_outerIters=2, ECM_outerNLLRtol=1.0e9)
    assert len(calls) == 4

    calls.clear()
    core.runConsenrich(
        **commonKwargs,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        ECM_outerNLLRtol=1.0e9,
    )
    assert len(calls) == 2

    calls.clear()
    core.runConsenrich(
        **commonKwargs,
        ECM_outerIters=4,
        ECM_minOuterIters=1,
        ECM_outerNLLRtol=0.0,
    )
    assert len(calls) > 1


@pytest.mark.correctness
def _caseGetBedMaskUsesIntervalSpanOverlap(tmp_path):
    bedPath = tmp_path / "regions.bed"
    bedPath.write_text("chrTest\t110\t120\n", encoding="utf-8")
    intervals = np.array([0, 100, 200], dtype=np.uint32)

    mask = core.getBedMask("chrTest", str(bedPath), intervals)

    assert mask.tolist() == [False, True, False]


@pytest.mark.correctness
def _caseSparseSupportWeightsUseExponentialDistanceDecay():
    weights = core._sparseSupportWeights(
        np.array([2], dtype=np.intp),
        intervalCount=6,
        ellIntervals=2.0,
        supportPrior=1.0,
    )
    expectedNEff = np.exp(-np.abs(np.arange(6) - 2) / 2.0)
    expected = expectedNEff / (expectedNEff + 1.0)

    assert np.allclose(weights, expected)
    assert weights[2] > weights[1] > weights[0]

    duplicateWeights = core._sparseSupportWeights(
        np.array([-5, 2, 2, 99], dtype=np.intp),
        intervalCount=6,
        ellIntervals=2.0,
        supportPrior=1.0,
    )
    np.testing.assert_allclose(duplicateWeights, expected.astype(np.float32))

    hardSupport = core._sparseSupportWeights(
        np.array([2], dtype=np.intp),
        intervalCount=6,
        ellIntervals=0.0,
        supportPrior=1.0,
    )
    np.testing.assert_array_equal(
        hardSupport,
        np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


@pytest.mark.correctness
def _casePSplineLogVarianceTrendRecoversNonmonotoneShape():
    rng = np.random.default_rng(123)
    amplitudes = np.linspace(0.0, 30.0, 240, dtype=np.float64)
    x = np.log1p(amplitudes)
    trueLogVariance = -1.2 + 0.45 * np.sin(2.7 * x) + 0.08 * (x - 2.0) ** 2
    noisyVariance = np.exp(trueLogVariance + rng.normal(0.0, 0.06, size=x.size))

    trend = core.fitPSplineLogVarianceTrend(
        amplitudes,
        noisyVariance,
        trendNumBasis=28,
        trendMinEdf=4.0,
        trendMaxEdf=22.0,
        trendLambdaGridSize=31,
        eps=1.0e-6,
    )
    fittedLogVariance = np.log(
        core.evalPSplineLogVarianceTrend(trend, amplitudes, eps=1.0e-6)
    )

    assert np.mean((fittedLogVariance - trueLogVariance) ** 2) < 0.02
    assert np.any(np.diff(fittedLogVariance) > 0.0)
    assert np.any(np.diff(fittedLogVariance) < 0.0)


@pytest.mark.correctness
def _casePSplineSignedPredictorDistinguishesPositiveAndNegativeMeans():
    means = np.linspace(-12.0, 12.0, 360, dtype=np.float64)
    x = np.sign(means) * np.log1p(np.abs(means))
    variances = np.exp(-0.4 + 0.6 * x)

    trend = core.fitPSplineLogVarianceTrend(
        means,
        variances,
        trendNumBasis=24,
        trendMinObsPerBasis=4.0,
        trendMinEdf=3.0,
        trendMaxEdf=18.0,
        trendLambdaGridSize=21,
        eps=1.0e-8,
    )
    pred = core.evalPSplineLogVarianceTrend(
        trend,
        np.array([-8.0, 8.0], dtype=np.float64),
        eps=1.0e-8,
    )

    assert pred[1] > 4.0 * pred[0]


@pytest.mark.correctness
def _casePSplinePredictionClampsToTrainingBoundary():
    amplitudes = np.linspace(1.0, 8.0, 80, dtype=np.float64)
    variances = np.exp(0.1 + 0.2 * np.sin(np.log1p(amplitudes)))
    trend = core.fitPSplineLogVarianceTrend(
        amplitudes,
        variances,
        trendNumBasis=12,
        trendMinEdf=3.0,
        trendLambdaGridSize=21,
        eps=1.0e-8,
    )

    lo = np.expm1(trend.xMin)
    hi = np.expm1(trend.xMax)
    predOutside = core.evalPSplineLogVarianceTrend(
        trend,
        np.array([0.0, 1000.0], dtype=np.float64),
        eps=1.0e-8,
    )
    predBoundary = core.evalPSplineLogVarianceTrend(
        trend,
        np.array([lo, hi], dtype=np.float64),
        eps=1.0e-8,
    )

    assert np.all(np.isfinite(predOutside))
    assert np.allclose(predOutside, predBoundary)


@pytest.mark.correctness
def _casePSplineEvaluationMatchesDenseDesign():
    amplitudes = np.linspace(-20.0, 20.0, 80, dtype=np.float64)
    signedPredictor = np.sign(amplitudes) * np.log1p(np.abs(amplitudes))
    variances = np.exp(0.2 + 0.1 * np.sin(signedPredictor))
    trend = core.fitPSplineLogVarianceTrend(
        amplitudes,
        variances,
        trendNumBasis=12,
        trendMinObsPerBasis=2.0,
        trendLambdaGridSize=17,
        eps=1.0e-8,
    )

    x = np.clip(signedPredictor, trend.xMin, trend.xMax)
    design = core._bsplineDesign(x, trend.knots, trend.degree)
    assert design.nnz <= amplitudes.size * (trend.degree + 1)
    denseDesign = design.toarray()
    expected = np.exp(denseDesign @ trend.beta)
    observed = core.evalPSplineLogVarianceTrend(trend, amplitudes, eps=1.0e-8)

    assert hasattr(cconsenrich, "cEvalPSplineLogVarianceTrend")
    assert np.allclose(observed, expected.astype(np.float32), rtol=1.0e-6)
    gram = denseDesign.T @ denseDesign
    penalty = core._coefficientDiffPenalty(gram.shape[0], 2)
    ridge = 1.0e-10 * max(np.trace(gram) / gram.shape[0], 1.0)
    system = gram + trend.lambdaHat * penalty + ridge * np.eye(gram.shape[0])
    denseBeta = np.linalg.solve(system, denseDesign.T @ np.log(variances))
    denseEDF = np.trace(np.linalg.solve(system, gram))
    residual = np.log(variances) - denseDesign @ denseBeta
    denseGCV = np.mean(residual ** 2) / (1.0 - denseEDF / variances.size) ** 2
    np.testing.assert_allclose(trend.beta, denseBeta, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose([trend.edf, trend.gcv], [denseEDF, denseGCV], rtol=1.0e-9)


@pytest.mark.correctness
def _casePSplineLimitsBasisCountByWeightedSupport():
    amplitudes = np.linspace(0.0, 10.0, 120, dtype=np.float64)
    variances = np.exp(0.2 + 0.1 * np.sin(np.log1p(amplitudes)))

    trend = core.fitPSplineLogVarianceTrend(
        amplitudes,
        variances,
        trendNumBasis=60,
        trendMinObsPerBasis=20.0,
        trendMinEdf=3.0,
        trendLambdaGridSize=11,
        eps=1.0e-8,
    )

    assert trend.diagnostics["requested_num_basis"] == 60
    assert trend.diagnostics["num_basis"] <= 6
    assert trend.diagnostics["trend_n_eff"] == pytest.approx(120.0)


@pytest.mark.correctness
def _casePooledMuncTrendUsesSharedShapeAndSampleStrength():
    rng = np.random.default_rng(2025)
    meansBase = np.linspace(-10.0, 10.0, 300, dtype=np.float64)
    residualSd = np.array([0.03, 0.25, 0.60], dtype=np.float64)
    means = np.tile(meansBase, residualSd.size)
    sampleIndex = np.repeat(np.arange(residualSd.size), meansBase.size)
    x = np.sign(means) * np.log1p(np.abs(means))
    sharedVariance = np.exp(-0.7 + 0.25 * np.sin(2.0 * x) + 0.08 * x)
    blockVariances = (
        sharedVariance
        * np.exp(rng.normal(0.0, residualSd[sampleIndex], means.size))
    )

    factorCount = 0
    factorize = core.linalg.cho_factor

    def countedFactorize(*args, **kwargs):
        nonlocal factorCount
        factorCount += 1
        return factorize(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core.linalg, "cho_factor", countedFactorize)
        pooled = core.fitPooledMuncVarianceTrend(
            means,
            blockVariances,
            sampleIndex,
            trendNumBasis=30,
            trendMinObsPerBasis=8.0,
            trendMinEdf=3.0,
            trendMaxEdf=20.0,
            trendLambdaGridSize=21,
            eps=1.0e-8,
        )
    assert factorCount == 21

    predicted = core.evalPSplineLogVarianceTrend(
        pooled.trend,
        meansBase,
        eps=1.0e-8,
    )
    nu0BySample = []
    for sample in range(residualSd.size):
        sampleMask = sampleIndex == sample
        nu0BySample.append(
            core.EB_computePooledPriorStrength(
                blockVariances[sampleMask],
                core.evalPSplineLogVarianceTrend(
                    pooled.trend,
                    means[sampleMask],
                    eps=1.0e-8,
                ),
                Nu_local=12.0,
                sampleIndex=sampleIndex[sampleMask],
                chromosomeIndex=np.zeros(np.count_nonzero(sampleMask), dtype=np.int64),
                blockStarts=np.arange(np.count_nonzero(sampleMask), dtype=np.int64),
                thinBinSize=1,
                localLogVarianceNoise=np.full(
                    np.count_nonzero(sampleMask),
                    0.02,
                    dtype=np.float64,
                ),
            )
        )

    xBase = x[: meansBase.size]
    sharedTarget = np.exp(-0.7 + 0.25 * np.sin(2.0 * xBase) + 0.08 * xBase)
    assert np.allclose(
        pooled.replicateVarianceFactors,
        np.ones(residualSd.size),
        atol=0.05,
    )
    assert np.mean(np.abs(np.log(predicted) - np.log(sharedTarget))) < 0.35
    assert len({round(float(value), 4) for value in nu0BySample}) > 1
    assert pooled.diagnostics["predictor"] == "signed_log1p"
    assert (
        pooled.diagnostics["replicate_factor_fit"]
        == "weighted_median_log_residual"
    )

    trueFactors = np.asarray([0.64, 1.0, 1.44], dtype=np.float64)
    shiftedVariances = blockVariances * trueFactors[sampleIndex]
    shifted = core.fitPooledMuncVarianceTrend(
        means,
        shiftedVariances,
        sampleIndex,
        trendNumBasis=30,
        trendMinObsPerBasis=8.0,
        trendMinEdf=3.0,
        trendMaxEdf=20.0,
        trendLambdaGridSize=21,
        eps=1.0e-8,
    )
    centeredTrue = trueFactors / np.exp(np.mean(np.log(trueFactors)))
    assert np.allclose(shifted.replicateVarianceFactors, centeredTrue, rtol=0.08)


@pytest.mark.correctness
def _caseMuncTrendRejectsInvalidVarianceValues():
    means = np.linspace(-2.0, 2.0, 8, dtype=np.float64)
    samples = np.arange(means.size, dtype=np.int64) % 2
    message = "blockVariances must contain only finite positive values"

    for badValue in (-1.0, 0.0, np.nan, np.inf):
        variances = np.linspace(1.0, 2.0, means.size, dtype=np.float64)
        variances[3] = badValue

        with pytest.raises(ValueError, match=message):
            core.fitPSplineLogVarianceTrend(
                means,
                variances,
                trendNumBasis=6,
                trendMinObsPerBasis=1.0,
                trendLambdaGridSize=5,
                eps=1.0e-8,
            )
        with pytest.raises(ValueError, match=message):
            core.fitPooledMuncVarianceTrend(
                means,
                variances,
                samples,
                trendNumBasis=6,
                trendMinObsPerBasis=1.0,
                trendLambdaGridSize=5,
                eps=1.0e-8,
            )


@pytest.mark.correctness
def _casePSplineGuardedGCVAppliesEdfCap():
    rng = np.random.default_rng(321)
    amplitudes = np.linspace(0.0, 10.0, 500, dtype=np.float64)
    trueLogVariance = 0.1 + 0.05 * np.sin(2.0 * np.log1p(amplitudes))
    variances = np.exp(trueLogVariance + rng.normal(0.0, 0.12, size=amplitudes.size))

    trend = core.fitPSplineLogVarianceTrend(
        amplitudes,
        variances,
        trendNumBasis=60,
        trendMinObsPerBasis=1.0,
        trendLambdaGridSize=41,
        trendMinEdf=3.0,
        trendMaxEdf=30.0,
        eps=1.0e-8,
    )

    assert trend.diagnostics["trend_max_edf"] == pytest.approx(30.0)
    assert trend.edf <= 30.0 + 1.0e-6


@pytest.mark.correctness
def _casePSplineUsesQuantileKnotsFromSupport():
    lowSupport = np.linspace(0.0, 1.0, 95, dtype=np.float64)
    highSupport = np.linspace(20.0, 30.0, 5, dtype=np.float64)
    amplitudes = np.concatenate([lowSupport, highSupport])
    variances = np.exp(0.1 + 0.05 * np.sin(np.log1p(amplitudes)))

    trend = core.fitPSplineLogVarianceTrend(
        amplitudes,
        variances,
        trendNumBasis=14,
        trendMinObsPerBasis=1.0,
        trendMinEdf=3.0,
        trendLambdaGridSize=11,
        eps=1.0e-8,
    )
    internal = np.unique(trend.knots)[1:-1]

    assert trend.diagnostics["knot_mode"] == "weighted_quantile"
    assert internal.size > 2
    assert not np.allclose(np.diff(internal), np.diff(internal)[0])
    assert np.count_nonzero(internal <= np.log1p(1.0)) > np.count_nonzero(
        internal > np.log1p(20.0)
    )


@pytest.mark.correctness
def _casePSplinePredictionClipsBeforeFloat32Overflow():
    trend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([1000.0], dtype=np.float64),
        xMin=0.0,
        xMax=1.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=1,
        diagnostics={},
    )

    pred = core.evalPSplineLogVarianceTrend(
        trend,
        np.array([0.0, 1.0], dtype=np.float64),
        eps=1.0e-6,
        maxVariance=1000.0,
    )

    assert np.all(np.isfinite(pred))
    assert np.all(pred <= np.float32(1000.0))


@pytest.mark.correctness
def _caseCheckStateUncertaintyCoverageOverallAndStrata():
    target = 2.0 * stats.norm.cdf(1.0) - 1.0
    residual = np.array([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
    before = np.ones_like(residual)
    after = np.full_like(residual, 2.0)
    strata = {
        "low_signal": np.array([True, True, False, False]),
        "high_signal": np.array([False, False, True, True]),
    }

    rows = core.checkStateUncertaintyCoverage(
        residual,
        before,
        after,
        targets=(target,),
        strata=strata,
    )

    byStratum = {row["stratum"]: row for row in rows}
    assert byStratum["overall"]["n"] == 4
    assert byStratum["overall"]["coverage_before"] == pytest.approx(0.5)
    assert byStratum["overall"]["coverage_after"] == pytest.approx(0.75)
    assert byStratum["overall"]["mean_width_before"] == pytest.approx(2.0)
    assert byStratum["overall"]["mean_width_after"] == pytest.approx(4.0)
    assert byStratum["low_signal"]["coverage_before"] == pytest.approx(1.0)
    assert byStratum["high_signal"]["coverage_after"] == pytest.approx(0.5)


@pytest.mark.correctness
def _caseEBPriorStrengthBoundaryIsUsable():
    assert core._coerceEBPriorStrength(4.0) == pytest.approx(4.0)
    assert core._coerceEBPriorStrength(4) == pytest.approx(4.0)
    assert core._coerceEBPriorStrength(3.999999) is None
    assert core._coerceEBPriorStrength(float("nan")) is None


@pytest.mark.correctness
def _caseEBPriorStrengthUsesThinnedVariancePairs():
    n = 25
    candidateIdx = np.arange(n, dtype=np.intp)
    globalVars = np.ones(n, dtype=np.float64)
    localVars = np.exp(0.32 * np.sin(candidateIdx) + 0.013 * candidateIdx)

    expectedUnpooled = core._computePriorStrengthFromCandidateIdx(
        localVars,
        globalVars,
        100.0,
        candidateIdx[candidateIdx % 4 == 0],
    )
    observedUnpooled = core.EB_computePriorStrength(
        localVars,
        globalVars,
        Nu_local=100.0,
        thinStride=4,
    )
    np.testing.assert_allclose(observedUnpooled, expectedUnpooled)

    pooledLogRatios = np.asarray(
        [0.00, 0.95, 0.42, -0.36, 0.88, -0.20, 0.30, -0.75],
        dtype=np.float64,
    )
    pooledLocalVars = np.exp(pooledLogRatios)
    pooledGlobalVars = np.ones(pooledLogRatios.size, dtype=np.float64)
    tupleKeyExpectedIdx = np.asarray([0, 2, 3, 5, 6], dtype=np.intp)
    expectedPooled = core._computePriorStrengthFromCandidateIdx(
        pooledLocalVars,
        pooledGlobalVars,
        100.0,
        tupleKeyExpectedIdx,
    )
    observedPooled = core.EB_computePooledPriorStrength(
        pooledLocalVars,
        pooledGlobalVars,
        Nu_local=100.0,
        sampleIndex=np.asarray([0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int64),
        chromosomeIndex=np.asarray([0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int64),
        blockStarts=np.asarray([0, 4, 0, 0, 6, 0, 12, 18], dtype=np.int64),
        thinBinSize=10,
    )
    expectedWithoutTupleKeys = core._computePriorStrengthFromCandidateIdx(
        pooledLocalVars,
        pooledGlobalVars,
        100.0,
        np.asarray([0, 6], dtype=np.intp),
    )

    np.testing.assert_allclose(observedPooled, expectedPooled)
    assert observedPooled != pytest.approx(expectedWithoutTupleKeys)


@pytest.mark.correctness
def _caseApplyBlacklistMuncFloorUsesNonBlacklistQuantile():
    munc = np.asarray(
        [
            [1.0, 2.0, 0.01, 4.0, np.nan],
            [10.0, np.nan, 0.1, 20.0, 0.05],
        ],
        dtype=np.float32,
    )
    blacklistMask = np.asarray([False, False, True, False, True])

    floors = core.applyBlacklistMuncFloor(munc, blacklistMask, minR=0.5)

    assert floors[0] == pytest.approx(np.quantile([1.0, 2.0, 4.0], 0.05))
    assert floors[1] == pytest.approx(np.quantile([10.0, 20.0], 0.05))
    assert munc[0, 2] >= floors[0]
    assert munc[0, 4] >= floors[0]
    assert munc[1, 2] >= floors[1]
    assert munc[1, 4] >= floors[1]
    assert np.isnan(munc[1, 1])


@pytest.mark.correctness
def _caseResolveMuncMinRFloorUsesMuncQuantileWhenNegative():
    munc = np.asarray(
        [
            [0.2, 1.0, np.nan, -2.0],
            [0.0, 0.5, 2.0, np.inf],
        ],
        dtype=np.float32,
    )

    resolved = core.resolveMuncMinRFloor(munc, minR=-1.0)

    expected = np.quantile([0.2, 1.0, 0.0, 0.5, 2.0], 0.05)
    assert resolved == pytest.approx(expected)
    assert core.resolveMuncMinRFloor(munc, minR=0.125) == pytest.approx(0.125)


@pytest.mark.correctness
def _caseApplyBlacklistMuncFloorUsesAutoFloorWhenMinRNegative():
    munc = np.asarray(
        [
            [0.2, 1.0, 0.01, 4.0, np.nan],
            [0.0, 0.5, 0.02, 2.0, np.inf],
        ],
        dtype=np.float32,
    )
    blacklistMask = np.asarray([False, False, True, False, True])
    baseFloor = core.resolveMuncMinRFloor(munc, minR=-1.0)

    floors = core.applyBlacklistMuncFloor(munc, blacklistMask, minR=-1.0)

    assert np.all(floors >= baseFloor)
    assert munc[0, 2] >= floors[0]
    assert munc[0, 4] >= floors[0]
    assert munc[1, 2] >= floors[1]
    assert munc[1, 4] >= floors[1]


@pytest.mark.correctness
def _caseGetMuncTrackRejectsSparseLocalVariancePaths():
    intervals = np.arange(0, 400, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.6, intervals.size, dtype=np.float32)
    localVarianceTrack = np.full(values.size, 0.5, dtype=np.float32)
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(0.5)], dtype=np.float64),
        xMin=0.0,
        xMax=0.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=1,
        diagnostics={},
    )
    cases = (
        (
            "sparse-nearest MUNC",
            {
                "sparseIntervalIndices": np.array([1, 4, 7], dtype=np.intp),
            },
        ),
        ("sparse-nearest MUNC", {"numNearest": 3}),
        (
            "restrictLocalVarianceToSparseBed",
            {
                "sparseRegionMask": np.ones(values.size, dtype=np.uint8),
                "restrictLocalVarianceToSparseBed": True,
            },
        ),
    )

    for message, kwargs in cases:
        with pytest.raises(ValueError, match=message):
            core.getMuncTrack(
                chromosome="chrTest",
                intervals=intervals,
                values=values,
                intervalSizeBP=25,
                muncTrendBlockSizeBP=125,
                muncLocalWindowSizeBP=150,
                pooledTrend=pooledTrend,
                localVarianceTrack=localVarianceTrack,
                **kwargs,
            )


@pytest.mark.correctness
def _caseGetMuncTrackClipsHugePriorBeforeShrinkage(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.2, intervals.size, dtype=np.float32)
    localVarTrack = np.linspace(0.05, 0.2, intervals.size, dtype=np.float32)
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([0.0], dtype=np.float64),
        xMin=0.0,
        xMax=0.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=1,
        diagnostics={},
    )

    def _fakeFitPSplineLogVarianceTrend(*args, **kwargs):
        pytest.fail("pooled trend should be reused")

    def _fakeEvalPSplineLogVarianceTrend(*args, **kwargs):
        return np.full(values.shape, 1.0e100, dtype=np.float64)

    monkeypatch.setattr(
        core, "fitPSplineLogVarianceTrend", _fakeFitPSplineLogVarianceTrend
    )
    monkeypatch.setattr(
        core, "evalPSplineLogVarianceTrend", _fakeEvalPSplineLogVarianceTrend
    )
    monkeypatch.setattr(
        core,
        "EB_computePriorStrength",
        lambda *args, **kwargs: 2_000_001.0,
    )

    muncTrack, _ = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        muncTrendBlockSizeBP=125,
        muncLocalWindowSizeBP=150,
        samplingIters=64,
        EB_localQuantile=-1.0,
        EB_use=True,
        pooledTrend=pooledTrend,
        localVarianceTrack=localVarTrack,
        varianceCap=0.75,
    )

    assert np.all(np.isfinite(muncTrack))
    assert np.all(muncTrack <= np.float32(0.75))


@pytest.mark.correctness
def _caseGetMuncTrackCapsPriorStrengthAtFiftyTimesLocalDf(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.2, intervals.size, dtype=np.float32)
    localVarTrack = np.full(intervals.size, 1.0, dtype=np.float32)
    priorVarTrack = np.full(intervals.size, 0.01, dtype=np.float32)
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(0.01)], dtype=np.float64),
        xMin=0.0,
        xMax=0.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=1,
        diagnostics={},
    )
    monkeypatch.setattr(
        core,
        "fitPSplineLogVarianceTrend",
        lambda *args, **kwargs: pytest.fail("pooled trend should be reused"),
    )
    monkeypatch.setattr(
        core,
        "evalPSplineLogVarianceTrend",
        lambda *args, **kwargs: priorVarTrack.copy(),
    )
    monkeypatch.setattr(
        core,
        "EB_computePriorStrength",
        lambda *args, **kwargs: 2_000_001.0,
    )

    muncTrack, _ = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        muncTrendBlockSizeBP=125,
        muncLocalWindowSizeBP=150,
        samplingIters=64,
        EB_localQuantile=-1.0,
        EB_setNuL=10,
        EB_use=True,
        pooledTrend=pooledTrend,
        localVarianceTrack=localVarTrack,
        varianceFloor=0.0,
        varianceCap=10.0,
    )

    expected = (10.0 * localVarTrack + 500.0 * priorVarTrack) / 510.0

    assert np.allclose(muncTrack, expected.astype(np.float32))


def test_core_munc_uses_kalman_local_evidence_for_shrinkage(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 500, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.5, intervals.size, dtype=np.float32)
    localVarTrack = np.full(intervals.size, 2.0, dtype=np.float32)
    priorVarTrack = np.full(intervals.size, 10.0, dtype=np.float32)
    tinyMask = np.zeros(intervals.size, dtype=bool)
    tinyMask[-4:] = True
    localVarTrack[tinyMask] = np.array([2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6])
    priorVarTrack[tinyMask] = np.array([6.0e-6, 7.0e-6, 8.0e-6, 9.0e-6])
    countModelVarianceFloor = np.zeros(intervals.size, dtype=np.float32)
    countModelVarianceFloor[1] = 5.5
    countModelVarianceFloor[tinyMask] = np.array(
        [1.5e-6, 2.0e-6, 2.5e-6, 3.0e-6],
        dtype=np.float32,
    )
    localScale = (1.0e-2 * float(np.median(localVarTrack))) + 1.0e-4
    priorScale = (1.0e-2 * float(np.median(priorVarTrack))) + 1.0e-4
    expectedCandidateMask = (localVarTrack > localScale) & (
        priorVarTrack > priorScale
    )
    np.testing.assert_array_equal(expectedCandidateMask, ~tinyMask)
    seen = {}
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(10.0)], dtype=np.float64),
        xMin=0.0,
        xMax=0.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=1,
        diagnostics={},
    )

    def _fakePriorStrength(local, prior, Nu_local, *args, **kwargs):
        seen["Nu_L"] = float(Nu_local)
        assert "candidateMask" in kwargs
        seen["candidateMask"] = np.asarray(kwargs["candidateMask"], dtype=bool)
        np.testing.assert_allclose(local, localVarTrack)
        np.testing.assert_allclose(prior, priorVarTrack)
        return 4.0

    monkeypatch.setattr(
        core,
        "fitPSplineLogVarianceTrend",
        lambda *args, **kwargs: pytest.fail("pooled trend should be reused"),
    )
    monkeypatch.setattr(
        core,
        "evalPSplineLogVarianceTrend",
        lambda *args, **kwargs: priorVarTrack.copy(),
    )
    monkeypatch.setattr(core, "EB_computePriorStrength", _fakePriorStrength)

    for nuLKwargs, expectedNuL in (
        ({"lengthScaleIntervals": 1}, 7.0),
        ({"lengthScaleIntervals": 2}, 4.0),
        ({"EB_effectiveNuL": 12.0}, 12.0),
        ({"EB_setNuL": 9, "EB_effectiveNuL": 12.0}, 9.0),
    ):
        seen.clear()
        muncTrack, _ = core.getMuncTrack(
            chromosome="chrTest",
            intervals=intervals,
            values=values,
            intervalSizeBP=25,
            muncTrendBlockSizeBP=125,
            muncLocalWindowSizeBP=250,
            samplingIters=64,
            EB_localQuantile=-1.0,
            EB_use=True,
            pooledTrend=pooledTrend,
            localVarianceTrack=localVarTrack,
            countModelVarianceFloor=countModelVarianceFloor,
            varianceFloor=0.0,
            varianceCap=20.0,
            **nuLKwargs,
        )

        weighted = (expectedNuL * localVarTrack + 4.0 * priorVarTrack) / (
            expectedNuL + 4.0
        )
        expected = weighted + countModelVarianceFloor

        assert seen["Nu_L"] == pytest.approx(expectedNuL)
        np.testing.assert_array_equal(seen["candidateMask"], expectedCandidateMask)
        np.testing.assert_allclose(
            muncTrack[tinyMask],
            expected[tinyMask].astype(np.float32),
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            muncTrack,
            expected.astype(np.float32),
            rtol=1.0e-6,
        )
        assert np.all(muncTrack >= countModelVarianceFloor)


def test_core_munc_eb_finalization_uses_native_count_floor_nan_sentinel(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.2, intervals.size, dtype=np.float32)
    localVarianceTrack = np.linspace(0.2, 1.3, intervals.size, dtype=np.float32)
    priorVarianceTrack = np.linspace(1.5, 0.4, intervals.size, dtype=np.float32)
    countModelVarianceFloor = np.array(
        [
            0.10,
            np.nan,
            0.00,
            0.20,
            np.nan,
            0.05,
            0.30,
            np.nan,
            0.00,
            0.15,
            0.25,
            np.nan,
        ],
        dtype=np.float32,
    )
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(1.0)], dtype=np.float64),
        xMin=0.0,
        xMax=0.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=1,
        diagnostics={},
    )
    nativeFinalize = cconsenrich.cFinalizeMuncEBTrack
    seen = {}

    def spyFinalize(localTrack, priorVarianceTrack=None, countFloor=None, **kwargs):
        if kwargs.get("useEB", True) and countFloor is not None:
            seen["countFloor"] = np.asarray(countFloor, dtype=np.float32).copy()
            seen["kwargs"] = dict(kwargs)
        return nativeFinalize(
            localTrack,
            priorVarianceTrack=priorVarianceTrack,
            countFloor=countFloor,
            **kwargs,
        )

    monkeypatch.setattr(cconsenrich, "cFinalizeMuncEBTrack", spyFinalize)
    monkeypatch.setattr(
        core,
        "fitPSplineLogVarianceTrend",
        lambda *args, **kwargs: pytest.fail("fit called"),
    )
    monkeypatch.setattr(
        core,
        "evalPSplineLogVarianceTrend",
        lambda *args, **kwargs: priorVarianceTrack.copy(),
    )
    monkeypatch.setattr(
        core,
        "EB_computePriorStrength",
        lambda *args, **kwargs: pytest.fail("prior strength called"),
    )

    muncTrack, _ = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        muncTrendBlockSizeBP=125,
        muncLocalWindowSizeBP=150,
        samplingIters=64,
        EB_use=True,
        EB_setNuL=6,
        EB_pooledNu0=4.0,
        pooledTrend=pooledTrend,
        localVarianceTrack=localVarianceTrack,
        countModelVarianceFloor=countModelVarianceFloor,
        varianceFloor=1.0e-6,
        varianceCap=20.0,
    )

    expected, _ = nativeFinalize(
        localVarianceTrack,
        priorVarianceTrack=priorVarianceTrack,
        countFloor=countModelVarianceFloor,
        nuLocal=6.0,
        nuPrior=4.0,
        varianceFloor=1.0e-6,
        varianceCap=20.0,
        useEB=True,
    )
    noFloorIndex = 1
    finiteFloorIndex = 0
    expectedNoFloor = (
        6.0 * float(localVarianceTrack[noFloorIndex])
        + 4.0 * float(priorVarianceTrack[noFloorIndex])
    ) / 10.0
    expectedFiniteFloor = (
        6.0 * float(localVarianceTrack[finiteFloorIndex])
        + 4.0 * float(priorVarianceTrack[finiteFloorIndex])
    ) / 10.0 + float(countModelVarianceFloor[finiteFloorIndex])

    assert seen["kwargs"]["nuLocal"] == pytest.approx(6.0)
    assert seen["kwargs"]["nuPrior"] == pytest.approx(4.0)
    assert np.isnan(seen["countFloor"][noFloorIndex])
    np.testing.assert_allclose(muncTrack, expected)
    assert muncTrack[noFloorIndex] == pytest.approx(expectedNoFloor)
    assert muncTrack[finiteFloorIndex] == pytest.approx(expectedFiniteFloor)


@pytest.mark.correctness
def _caseGetMuncTrackUsesSuppliedPooledTrendAndPriorMean(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(-1.0, 1.0, intervals.size, dtype=np.float32)
    localVarTrack = np.full(intervals.size, 9.0, dtype=np.float32)
    priorMeanTrack = np.full(intervals.size, 2.0, dtype=np.float32)
    priorVarTrack = np.linspace(2.0, 3.0, intervals.size, dtype=np.float32)
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(2.0)], dtype=np.float64),
        xMin=-1.0,
        xMax=1.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=8,
        diagnostics={},
    )
    monkeypatch.setattr(
        core,
        "evalPSplineLogVarianceTrend",
        lambda *args, **kwargs: pytest.fail("prior variance should be reused"),
    )
    monkeypatch.setattr(
        core,
        "fitPSplineLogVarianceTrend",
        lambda *args, **kwargs: pytest.fail("pooled trend should be reused"),
    )
    monkeypatch.setattr(
        core,
        "EB_computePriorStrength",
        lambda *args, **kwargs: pytest.fail("pooled Nu_0 should be reused"),
    )
    caplog.clear()
    caplog.set_level(logging.INFO, logger=core.logger.name)

    muncTrack, _ = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        muncTrendBlockSizeBP=125,
        muncLocalWindowSizeBP=150,
        samplingIters=64,
        EB_localQuantile=-1.0,
        EB_setNuL=10,
        EB_use=True,
        pooledTrend=pooledTrend,
        priorMeanTrack=priorMeanTrack,
        priorVarianceTrack=priorVarTrack,
        EB_pooledNu0=4.0,
        localVarianceTrack=localVarTrack,
        varianceFloor=0.0,
        varianceCap=20.0,
        sampleFile="ENCFF12345_sampleA.bam",
    )

    expected = (10.0 * localVarTrack + 4.0 * priorVarTrack) / 14.0
    assert np.allclose(muncTrack, expected.astype(np.float32))
    assert "sampleFile=ENCFF12" in caplog.text
    assert "sample_file=ENCFF12" in caplog.text
    assert "ENCFF123" not in caplog.text


@pytest.mark.correctness
def _caseGetMuncTrackSmoothsPriorMeanWithEMA(monkeypatch: pytest.MonkeyPatch):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(-2.0, 2.0, intervals.size, dtype=np.float32)
    smoothedValues = np.linspace(0.25, 1.25, intervals.size, dtype=np.float32)
    localVarTrack = np.full(intervals.size, 4.0, dtype=np.float32)
    priorVarTrack = np.linspace(1.0, 2.0, intervals.size, dtype=np.float32)
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(1.0)], dtype=np.float64),
        xMin=-2.0,
        xMax=2.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=8,
        diagnostics={},
    )
    evalInputs = []

    def fakeEMA(arr, alpha):
        assert np.asarray(arr).shape == values.shape
        assert 0.0 < float(alpha) <= 1.0
        return smoothedValues.copy()

    def fakeEval(_trend, predictor, **kwargs):
        predictorArr = np.asarray(predictor, dtype=np.float32).reshape(-1)
        evalInputs.append(predictorArr.copy())
        return priorVarTrack.copy()

    monkeypatch.setattr(cconsenrich, "cEMA", fakeEMA)
    monkeypatch.setattr(core, "evalPSplineLogVarianceTrend", fakeEval)
    monkeypatch.setattr(
        core,
        "fitPSplineLogVarianceTrend",
        lambda *args, **kwargs: pytest.fail("pooled trend should be reused"),
    )
    monkeypatch.setattr(
        core,
        "EB_computePriorStrength",
        lambda *args, **kwargs: pytest.fail("pooled Nu_0 should be reused"),
    )

    muncTrack, supportFraction = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        muncTrendBlockSizeBP=125,
        muncLocalWindowSizeBP=150,
        samplingIters=64,
        EB_use=True,
        EB_setNuL=6,
        EB_pooledNu0=4.0,
        pooledTrend=pooledTrend,
        localVarianceTrack=localVarTrack,
        varianceFloor=0.0,
        varianceCap=20.0,
    )

    expected = ((6.0 * localVarTrack) + (4.0 * priorVarTrack)) / 10.0
    assert supportFraction == pytest.approx(1.0)
    assert any(
        item.shape == smoothedValues.shape and np.allclose(item, smoothedValues)
        for item in evalInputs
    )
    np.testing.assert_allclose(muncTrack, expected.astype(np.float32))


@pytest.mark.correctness
def _caseGetMuncTrackAppliesReplicateVarianceFactor():
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(-1.0, 1.0, intervals.size, dtype=np.float32)
    localVarianceTrack = np.linspace(0.5, 1.6, intervals.size, dtype=np.float32)
    priorVarianceTrack = np.linspace(1.0, 2.1, intervals.size, dtype=np.float32)
    pooledTrend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(2.0)], dtype=np.float64),
        xMin=-1.0,
        xMax=1.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=8,
        diagnostics={},
    )

    muncTrack, _ = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        muncTrendBlockSizeBP=125,
        muncLocalWindowSizeBP=150,
        pooledTrend=pooledTrend,
        priorVarianceTrack=priorVarianceTrack,
        replicateVarianceFactor=1.5,
        EB_use=True,
        EB_setNuL=6,
        EB_pooledNu0=4.0,
        localVarianceTrack=localVarianceTrack,
        varianceFloor=1.0e-6,
        varianceCap=20.0,
    )

    expected = (6.0 * localVarianceTrack + 4.0 * priorVarianceTrack * 1.5) / 10.0
    np.testing.assert_allclose(muncTrack, expected.astype(np.float32), rtol=1.0e-6)


@pytest.mark.correctness
def test_core_pooled_prior_strength_uses_block_log_variance_noise():
    n = 100
    logRatio = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    localVariances = np.exp(logRatio)
    globalVariances = np.ones(n, dtype=np.float64)

    nominalNu0 = core.EB_computePooledPriorStrength(
        localVariances,
        globalVariances,
        Nu_local=100.0,
    )
    noiseAwareNu0 = core.EB_computePooledPriorStrength(
        localVariances,
        globalVariances,
        Nu_local=100.0,
        localLogVarianceNoise=np.full(n, 0.25, dtype=np.float64),
    )

    assert noiseAwareNu0 > nominalNu0
    assert core.EB_computePooledPriorStrength(
        np.ones(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        Nu_local=100.0,
    ) == pytest.approx(4.0)

    thinLocalVariances = np.exp(np.linspace(-0.2, 0.2, 8, dtype=np.float64))
    assert core.EB_computePooledPriorStrength(
        thinLocalVariances,
        np.ones(8, dtype=np.float64),
        Nu_local=100.0,
        sampleIndex=np.zeros(8, dtype=np.int64),
        chromosomeIndex=np.zeros(8, dtype=np.int64),
        blockStarts=np.zeros(8, dtype=np.int64),
        thinBinSize=10,
    ) == pytest.approx(4.0)


@pytest.mark.correctness
def _caseMuncSizingAndVarianceModels():
    intervalSizeBP = 25
    with pytest.raises(ValueError, match="length scale"):
        core._resolveMuncRuntimeSizing(intervalSizeBP=intervalSizeBP)

    explicitSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        muncTrendBlockSizeBP=250,
        muncLocalWindowSizeBP=500,
    )
    assert explicitSizing.trendBlockSizeBP == 250
    assert explicitSizing.localWindowSizeBP == 500

    lengthScaleSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        lengthScaleIntervals=17,
        muncTrendBlockSizeBP=None,
        muncLocalWindowSizeBP=None,
        muncTrendBlockLengthScaleMultiplier=1.5,
        muncLocalWindowLengthScaleMultiplier=2.5,
    )
    assert lengthScaleSizing.usedLengthScale is True
    assert lengthScaleSizing.lengthScaleIntervals == 17
    assert lengthScaleSizing.trendBlockIntervals == 26
    assert lengthScaleSizing.localWindowIntervals == 43

    for binBP, spanIntervals, explicitBP, expectedBP in (
        (50, 200, None, (20000, 40000)),
        (50, 8000, None, (800000, 1000000)),
        (50, 10**400, None, (1000000, 1000000)),
        (300, 1667, None, (999900, 999900)),
        (300, None, 10**400, (999900, 999900)),
    ):
        boundedSizing = core._resolveMuncRuntimeSizing(
            intervalSizeBP=binBP,
            lengthScaleIntervals=spanIntervals,
            muncTrendBlockSizeBP=explicitBP,
            muncLocalWindowSizeBP=explicitBP,
        )
        assert (
            boundedSizing.trendBlockSizeBP, boundedSizing.localWindowSizeBP
        ) == expectedBP
        assert boundedSizing.trendBlockIntervals * binBP == expectedBP[0]
        assert boundedSizing.localWindowIntervals * binBP == expectedBP[1]


@pytest.mark.correctness
def _caseChooseFeatureLengthBootstrapWidthVarianceAndContextCompat():
    grid = np.arange(512, dtype=np.float64)
    vals = np.zeros_like(grid)
    for center, amp, sigma in [
        (64.0, 4.0, 4.0),
        (160.0, 6.0, 7.0),
        (256.0, 5.0, 5.5),
        (352.0, 7.0, 8.0),
        (448.0, 4.5, 6.0),
    ]:
        vals += amp * np.exp(-0.5 * ((grid - center) / sigma) ** 2)

    pointEstimate, widthLower, widthUpper, diagnostics = core.chooseFeatureLength(
        vals.astype(np.float32),
        minSpan=3,
        maxSpan=64,
        bandZ=1.0,
        maxOrder=5,
    )

    assert pointEstimate >= 3
    assert pointEstimate <= 64
    assert widthLower >= 1
    assert widthUpper >= widthLower
    assert diagnostics["method"] == "feature_peak_width_random_effects"


@pytest.mark.correctness
def _caseChromSizesKeepSexChromosomesByDefault():
    with tempfile.TemporaryDirectory() as tempDir:
        sizesPath = Path(tempDir) / "chrom.sizes"
        sizesPath.write_text(
            "\n".join(
                [
                    "chr1\t1000",
                    "chrX\t900",
                    "chrY\t800",
                    "chrM\t700",
                    "chr1_KI270706v1_random\t600",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        chromSizes = misc_util.getChromSizesDict(str(sizesPath))

        assert {"chr1", "chrX", "chrY", "chrM"} <= set(chromSizes)
        assert "chr1_KI270706v1_random" not in chromSizes
        assert "chrY" not in misc_util.getChromSizesDict(
            str(sizesPath), excludeChroms=["chrY"]
        )
























@pytest.mark.correctness
def _caseNormalizeCountModeRejectsNoncanonicalModes():
    for badMode in [
        "Coverage",
        " coverage ",
        "cov",
        "cut",
        "cutsites",
        "5p",
        "five_prime",
        "centre",
        "conservedfractionaloverlap",
        "conserved-fractional-overlap",
        "conserved_fractional_overlap",
    ]:
        with pytest.raises(ValueError, match="Unsupported countMode"):
            normalization.validateCountMode(badMode, "coverage")






























def _pairedNormalizationRecords() -> list[dict]:
    return [
        {
            "name": "pair1",
            "start": 100,
            "flag": 99,
            "next_reference_id": 0,
            "next_start": 160,
            "template_length": 80,
        },
        {
            "name": "pair1",
            "start": 160,
            "flag": 147,
            "next_reference_id": 0,
            "next_start": 100,
            "template_length": -80,
        },
        {
            "name": "pair2",
            "start": 300,
            "flag": 99,
            "next_reference_id": 0,
            "next_start": 360,
            "template_length": 80,
        },
        {
            "name": "pair2",
            "start": 360,
            "flag": 147,
            "next_reference_id": 0,
            "next_start": 300,
            "template_length": -80,
        },
    ]
























def _run_with_monkeypatch(monkeypatch, func, *args):
    with monkeypatch.context() as mp:
        return func(*args, mp)
def test_core_em_loop_contracts(monkeypatch, contract_case):
    contract_case(
        "outer pass smoke",
        _caseRunConsenrichOuterPassSmoke,
    )
    for label, func in (
        (
            "initial process noise skips warmup",
            _caseRunConsenrichInitialProcessQSkipsWarmup,
        ),
        (
            "data-derived fixed diagonal process noise",
            _caseRunConsenrichFixedDiagonalUsesDataQ,
        ),
        (
            "outer-pass minimum iterations",
            _caseRunConsenrichOuterPassRequiresThreeIterationsDespiteTolerance,
        ),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)
    contract_case("level state-model smoke", _caseRunConsenrichLevelStateModelSmoke)


def test_core_pspline_sparse_support_and_trend_contracts(
    tmp_path,
    monkeypatch,
    caplog,
    contract_case,
):
    contract_case(
        "BED mask interval overlap", _caseGetBedMaskUsesIntervalSpanOverlap, tmp_path
    )
    for label, func in (
        ("sparse support decay", _caseSparseSupportWeightsUseExponentialDistanceDecay),
        (
            "nonmonotone P-spline trend",
            _casePSplineLogVarianceTrendRecoversNonmonotoneShape,
        ),
        (
            "signed P-spline predictor",
            _casePSplineSignedPredictorDistinguishesPositiveAndNegativeMeans,
        ),
        ("P-spline boundary clamp", _casePSplinePredictionClampsToTrainingBoundary),
        ("P-spline eval", _casePSplineEvaluationMatchesDenseDesign),
        ("P-spline basis support limit", _casePSplineLimitsBasisCountByWeightedSupport),
        (
            "pooled MUNC shared trend",
            _casePooledMuncTrendUsesSharedShapeAndSampleStrength,
        ),
        (
            "MUNC trend invalid variance rejection",
            _caseMuncTrendRejectsInvalidVarianceValues,
        ),
        ("MUNC centered evidence windows", _caseMuncSmoothDenseLocalEvidenceUsesCenteredWindows),
        ("MUNC observation moments", _caseMuncObservationMomentSeedPassUsesOmegaMomentsAndFloors),
        ("MUNC window bounds", _caseMuncSizingAndVarianceModels),
        (
            "EB prior strength thinning",
            _caseEBPriorStrengthUsesThinnedVariancePairs,
        ),
        (
            "MUNC sparse local variance rejection",
            _caseGetMuncTrackRejectsSparseLocalVariancePaths,
        ),
        ("P-spline guarded GCV", _casePSplineGuardedGCVAppliesEdfCap),
        ("P-spline quantile knots", _casePSplineUsesQuantileKnotsFromSupport),
        ("P-spline float32 clipping", _casePSplinePredictionClipsBeforeFloat32Overflow),
        ("MUNC blacklist floor", _caseApplyBlacklistMuncFloorUsesNonBlacklistQuantile),
        (
            "MUNC auto minR quantile",
            _caseResolveMuncMinRFloorUsesMuncQuantileWhenNegative,
        ),
        (
            "MUNC blacklist auto minR floor",
            _caseApplyBlacklistMuncFloorUsesAutoFloorWhenMinRNegative,
        ),
    ):
        contract_case(label, func)
    contract_case(
        "MUNC supplied pooled trend",
        _caseGetMuncTrackUsesSuppliedPooledTrendAndPriorMean,
        monkeypatch,
        caplog,
    )
    contract_case(
        "MUNC EMA prior mean",
        _caseGetMuncTrackSmoothsPriorMeanWithEMA,
        monkeypatch,
    )
    contract_case(
        "MUNC applies replicate factor",
        _caseGetMuncTrackAppliesReplicateVarianceFactor,
    )
