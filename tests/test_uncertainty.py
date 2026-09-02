# -*- coding: utf-8 -*-

import logging
import math
import numpy as np
import json
import pandas as pd
import pytest

import consenrich.cuncertainty as cuncertainty
import consenrich.consenrich as consenrichRuntime
import consenrich.core as core
import consenrich.diagnostics as diagnostic_utils
import consenrich.segshrink as segshrink
import consenrich.uncertainty as uncertainty


def _smallRunKwargs():
    return dict(
        deltaF=0.15,
        minQ=1.0e-4,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=12,
        pad=1.0e-4,
        ECM_fixedBackgroundIters=2,
        ECM_fixedBackgroundRtol=0.0,
        ECM_robustTNu=8.0,
        ECM_useObsPrecisionReweighting=True,
        ECM_useProcessPrecisionReweighting=True,
        ECM_outerIters=1,
        ECM_backgroundShiftRtol=0.0,
        returnScales=True,
    )


def _caseDeleteBlockGlobalFactorUsesWeightedQuantile():
    residual = np.array([0.5, 1.0, 3.0], dtype=np.float64)
    pDelta = np.ones_like(residual)
    rowWeight = np.ones_like(residual)
    params = core.uncertaintyCalibrationParams(
        targets=(0.8,),
        factorMin=core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
        factorMax=core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
        minHeldoutCells=1,
    )

    factor, meta = uncertainty._fitDeleteBlockGlobalFactor(
        residual=residual,
        pDelta=pDelta,
        rowWeight=rowWeight,
        params=params,
    )

    expected = (3.0 / uncertainty._normalZ(0.8)) ** 2
    assert factor == pytest.approx(expected)
    assert meta["factor_model"] == "global"
    assert meta["global_factor"] == pytest.approx(factor)


def _caseSegShrinkFactorModelStrictContract():
    assert uncertainty._normalizeDeleteBlockFactorModel(None) == "segShrink"
    assert uncertainty._normalizeDeleteBlockFactorModel("global") == "global"
    assert uncertainty._normalizeDeleteBlockFactorModel("segShrink") == "segShrink"
    for value in ("seg-shrink", "seg_shrink", "segshrink", "SegShrink"):
        with pytest.raises(ValueError, match="factor model"):
            uncertainty._normalizeDeleteBlockFactorModel(value)


def _casePacOrderIndexExamples():
    assert uncertainty._pacOrderIndex(59, 0.95, 0.05) == 59
    assert uncertainty._pacOrderIndex(100, 0.95, 0.05) == 99
    assert uncertainty._pacOrderIndex(200, 0.95, 0.05) == 196
    assert uncertainty._pacOrderIndex(500, 0.95, 0.05) == 484
    assert uncertainty._pacOrderIndex(58, 0.95, 0.05) is None
    assert uncertainty._minBlocksForFiniteBound(0.95, 0.05) == 59
    bounds = uncertainty._targetCalibrationBounds(
        np.arange(58, dtype=np.float64),
        targets=(0.95,),
        delta=0.05,
    )
    assert bounds[0]["bound_available"] is False
    assert bounds[0]["q"] == 57.0
    assert bounds[0]["q_source"] == "empirical_max_without_finite_order_bound"


def _caseDeleteBlockInformationApproximation():
    m = 3
    n = 7
    blockLen = 2
    folds = 3
    deletionProbability = 0.4
    seed = 1
    blockCount = (n + blockLen - 1) // blockLen
    seededBlockFold, seededRepsByBlockCount, seededRepsByBlock = (
        cuncertainty.cmakeFoldSpec(
            m,
            n,
            blockLen,
            folds,
            deletionProbability,
            seed,
        )
    )
    rng = np.random.default_rng(seed)
    blockOrder = rng.permutation(blockCount).astype(np.int32, copy=False)
    wantBlockFold = np.empty(blockCount, dtype=np.int32)
    wantBlockFold[blockOrder] = np.arange(blockCount, dtype=np.int32) % folds
    wantRepsByBlockCount = np.empty(blockCount, dtype=np.intp)
    wantRepsByBlock = np.full((blockCount, m), -1, dtype=np.intp)
    for block in range(blockCount):
        deleteCount = int(rng.binomial(m, deletionProbability))
        while deleteCount < 1 or deleteCount >= m:
            deleteCount = int(rng.binomial(m, deletionProbability))
        wantRepsByBlockCount[block] = deleteCount
        wantRepsByBlock[block, :deleteCount] = rng.choice(
            m, size=deleteCount, replace=False
        )
    assert np.array_equal(seededBlockFold, wantBlockFold)
    assert np.array_equal(seededRepsByBlockCount, wantRepsByBlockCount)
    assert np.array_equal(seededRepsByBlock, wantRepsByBlock)
    assert np.all(seededRepsByBlockCount >= 1)

    infoCell = np.array(
        [
            [1.0, 2.0, 4.0],
            [3.0, 2.0, 4.0],
        ],
        dtype=np.float64,
    )
    matrixMunc = 1.0 / infoCell
    activeMask = np.ones_like(infoCell, dtype=np.uint8)
    blockFold = np.array([1, 0, 0], dtype=np.int32)
    repsByBlockCount = np.array([1, 2, 1], dtype=np.intp)
    repsByBlock = np.array([[0, -1], [0, 1], [1, -1]], dtype=np.intp)
    totalInfo = cuncertainty.cobservationTotalInformation(
        matrixMunc,
        activeMask,
        np.empty(0, dtype=np.float64),
        False,
        0.0,
    )
    _foldMask, keptInfo, deletedInfo, h = cuncertainty.cmakeFoldMaskAndInformation(
        2,
        3,
        1,
        0,
        blockFold,
        repsByBlockCount,
        repsByBlock,
        matrixMunc,
        activeMask,
        totalInfo,
        np.empty(0, dtype=np.float64),
        False,
        0.0,
    )
    assert np.array_equal(
        _foldMask,
        [
            [1, 0, 1],
            [1, 0, 0],
        ],
    )

    assert np.allclose(totalInfo, [4.0, 4.0, 8.0])
    assert np.allclose(keptInfo, [4.0, 0.0, 4.0])
    assert np.allclose(deletedInfo, [0.0, 4.0, 4.0])
    assert np.allclose(h, [0.0, 1.0, 0.5])

    rho = 0.5
    totalInfoRho = cuncertainty.cobservationTotalInformation(
        matrixMunc,
        activeMask,
        np.empty(0, dtype=np.float64),
        False,
        0.0,
        rho,
    )

    def _exchangeableInfo(weights, rhoValue):
        weights = np.asarray(weights, dtype=np.float64)
        adjusted = (
            float(np.sum(weights)) / (1.0 - rhoValue)
            - rhoValue
            * float(np.sum(np.sqrt(weights))) ** 2
            / ((1.0 - rhoValue) * (1.0 - rhoValue + rhoValue * weights.size))
        )
        return min(float(np.sum(weights)), adjusted)

    assert totalInfoRho[0] == pytest.approx(_exchangeableInfo([1.0, 3.0], rho))
    assert totalInfoRho[1] == pytest.approx(4.0 / (1.0 + rho))
    assert totalInfoRho[2] == pytest.approx(8.0 / (1.0 + rho))

    foldResultRho = cuncertainty.cmakeFoldMaskAndInformation(
        2,
        3,
        1,
        0,
        blockFold,
        repsByBlockCount,
        repsByBlock,
        matrixMunc,
        activeMask,
        totalInfoRho,
        np.empty(0, dtype=np.float64),
        False,
        0.0,
        rho,
        True,
    )
    _maskRho, keptRho, deletedRho, hRho, nominalDeletedRho = foldResultRho
    assert np.array_equal(_maskRho, _foldMask)
    assert np.allclose(keptRho, [totalInfoRho[0], 0.0, 4.0])
    assert np.allclose(deletedRho, [0.0, totalInfoRho[1], totalInfoRho[2] - 4.0])
    assert np.allclose(hRho, [0.0, 1.0, (totalInfoRho[2] - 4.0) / totalInfoRho[2]])
    assert np.allclose(nominalDeletedRho, [0.0, 4.0, 4.0])
    assert hRho[2] < h[2]

    equalMunc = np.full((4, 2), 2.0, dtype=np.float64)
    equalActive = np.ones_like(equalMunc, dtype=np.uint8)
    equalInfo = cuncertainty.cobservationTotalInformation(
        equalMunc,
        equalActive,
        np.empty(0, dtype=np.float64),
        False,
        0.0,
        rho,
    )
    assert np.allclose(equalInfo, [2.0 / (1.0 + 3.0 * rho)] * 2)

    cappedMunc = np.array([[1.0], [0.01]], dtype=np.float64)
    cappedInfo = cuncertainty.cobservationTotalInformation(
        cappedMunc,
        np.ones_like(cappedMunc, dtype=np.uint8),
        np.empty(0, dtype=np.float64),
        False,
        0.0,
        0.25,
    )
    assert cappedInfo[0] == pytest.approx(101.0)

    delta, source, valid, reason = uncertainty._chooseDeleteBlockDeltaVariance(
        np.array([1.0, 1.0, 2.0], dtype=np.float64),
        np.array([1.1, 1.1, 2.1], dtype=np.float64),
        h,
        mode="heldout_information",
        minDeltaVariance=1.0e-12,
        minInformationFraction=0.01,
        maxInformationFraction=0.95,
        positiveFloor=1.0e-12,
    )

    assert valid.tolist() == [False, False, True]
    assert np.all(np.isnan(delta[:2]))
    assert delta[2] == pytest.approx(2.0)
    assert uncertainty.DELETE_BLOCK_VARIANCE_SOURCE_LABELS[source].tolist() == [
        "invalid",
        "invalid",
        "heldout_information",
    ]
    assert uncertainty.DELETE_BLOCK_INVALID_REASON_LABELS[reason].tolist() == [
        "h_out_of_bounds",
        "h_out_of_bounds",
        "valid",
    ]


def _caseReplicateDependenceGaussianCoverage():
    rng = np.random.default_rng(13)
    draws = 40_000
    m = 4
    target = 0.95
    z = uncertainty._normalZ(target)

    for rho in (0.0, 0.5, 1.0):
        shared = rng.normal(size=(draws, 1))
        independent = rng.normal(size=(draws, m))
        samples = np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * independent
        estimate = np.mean(samples, axis=1)
        if rho == 1.0:
            correctedVariance = 1.0
        else:
            correctedVariance = (1.0 + (m - 1.0) * rho) / m
        naiveVariance = 1.0 / m
        correctedCoverage = np.mean(np.abs(estimate) <= z * np.sqrt(correctedVariance))
        naiveCoverage = np.mean(np.abs(estimate) <= z * np.sqrt(naiveVariance))
        assert correctedCoverage == pytest.approx(target, abs=0.015)
        if rho == 0.0:
            assert naiveCoverage == pytest.approx(target, abs=0.015)
        elif rho == 0.5:
            assert naiveCoverage < 0.90
        else:
            assert naiveCoverage < 0.80


def _caseReplicateDependenceDeleteBlockEvidence():
    n = 40
    blockLen = 10
    t = np.linspace(0.0, 4.0 * np.pi, n, dtype=np.float64)
    shared = np.sin(t)
    matrixData = np.vstack(
        [
            shared + 0.05 * np.cos(t),
            shared - 0.03 * np.sin(2.0 * t),
            shared + 0.04 * np.cos(2.0 * t),
            shared - 0.02 * np.sin(3.0 * t),
        ]
    ).astype(np.float64)
    matrixMunc = np.ones((4, n), dtype=np.float64)
    activeMask = np.ones((4, n), dtype=np.uint8)
    blockFold = np.zeros(4, dtype=np.int32)
    repsByBlockCount = np.full(4, 2, dtype=np.intp)
    repsByBlock = np.array(
        [[0, 1, -1, -1], [2, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1]],
        dtype=np.intp,
    )
    signal = np.zeros(n, dtype=np.float64)
    lambdaExp = np.empty(0, dtype=np.float64)

    evidence = cuncertainty.cdeleteBlockReplicateDependenceRhoEvidence(
        matrixData,
        matrixMunc,
        activeMask,
        blockFold,
        repsByBlockCount,
        repsByBlock,
        signal,
        lambdaExp,
        False,
        0.0,
        blockLen,
        0,
    )
    estimate = uncertainty._replicateDependenceEstimateFromEvidence(
        zWeightedSum=evidence["fisher_z_weighted_sum"],
        weightSum=evidence["weight_sum"],
        blockCount=evidence["block_count"],
        pairCount=evidence["pair_count"],
        rhoUpperBound=evidence["rho_upper_bound"],
    )
    assert evidence["block_count"] == 4
    assert evidence["pair_count"] == 4
    assert estimate["rho"] > 0.0
    assert estimate["rho"] <= evidence["rho_upper_bound"]

    noPairEvidence = cuncertainty.cdeleteBlockReplicateDependenceRhoEvidence(
        matrixData,
        matrixMunc,
        activeMask,
        blockFold,
        np.ones(4, dtype=np.intp),
        np.array([[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1], [3, -1, -1, -1]], dtype=np.intp),
        signal,
        lambdaExp,
        False,
        0.0,
        blockLen,
        0,
    )
    assert noPairEvidence["pair_count"] == 0
    assert noPairEvidence["weight_sum"] == pytest.approx(0.0)


def _caseDeleteBlockVarianceModeSelection():
    pFull = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    pMasked = np.array([1.5, 1.0, 0.8], dtype=np.float64)
    h = np.array([0.25, 0.5, 0.5], dtype=np.float64)

    delta, source, valid, reason = uncertainty._chooseDeleteBlockDeltaVariance(
        pFull,
        pMasked,
        h,
        mode="hybrid",
        minDeltaVariance=1.0e-12,
        minInformationFraction=0.01,
        maxInformationFraction=0.95,
        positiveFloor=1.0e-12,
    )

    assert valid.tolist() == [True, True, True]
    assert np.allclose(delta, [0.5, 1.0, 1.0])
    assert uncertainty.DELETE_BLOCK_VARIANCE_SOURCE_LABELS[source].tolist() == [
        "covariance_difference",
        "heldout_information_fallback",
        "heldout_information_fallback",
    ]
    assert uncertainty.DELETE_BLOCK_INVALID_REASON_LABELS[reason].tolist() == [
        "valid",
        "valid",
        "valid",
    ]

    delta, source, valid, reason = uncertainty._chooseDeleteBlockDeltaVariance(
        pFull,
        pMasked,
        h,
        mode="covariance_difference",
        minDeltaVariance=1.0e-12,
        minInformationFraction=0.01,
        maxInformationFraction=0.95,
        positiveFloor=1.0e-12,
    )

    assert valid.tolist() == [True, False, False]
    assert np.isfinite(delta[0])
    assert uncertainty.DELETE_BLOCK_VARIANCE_SOURCE_LABELS[source].tolist() == [
        "covariance_difference",
        "invalid",
        "invalid",
    ]
    assert uncertainty.DELETE_BLOCK_INVALID_REASON_LABELS[reason].tolist() == [
        "valid",
        "covariance_delta_nonpositive",
        "covariance_delta_nonpositive",
    ]


def _caseTargetCalibrationTrackScaleUsesQOverZ():
    target = 0.95
    z = uncertainty._normalZ(target)
    info = uncertainty._targetCalibrationTrackScale(
        {
            "target": target,
            "q": 2.0 * z,
            "q_source": "exchangeability_conditional_order_statistic",
            "bound_available": True,
            "bound_scope": "chromosome_selected_target_conditional_exchangeability",
        }
    )

    assert info["scaled"] is True
    assert info["bound_available"] is True
    assert info["target_z"] == pytest.approx(z)
    assert info["scale"] == pytest.approx(2.0)
    assert info["reason"] == "scaled_by_exchangeability_conditional_order_bound_q_over_z"


def _caseAutoBlockSizeForShortContigs():
    assert diagnostic_utils.resolveUncertaintyBlockSizeIntervals(
        None,
        25,
        800,
        folds=2,
    ) == 400
    assert diagnostic_utils.resolveUncertaintyBlockSizeIntervals(
        "auto",
        25,
        800,
        folds=4,
    ) == 200
    assert diagnostic_utils.resolveUncertaintyBlockSizeIntervals(
        None,
        25,
        20_000,
        folds=4,
    ) == 400
    assert diagnostic_utils.resolveUncertaintyBlockSizeIntervals(
        50_000,
        25,
        800,
        folds=4,
    ) == 800
    assert uncertainty._resolveBlockSizeIntervals(None, 25, 6, folds=4) == 6


def _pythonFeatureMatrix(state, stateVar, matrixMunc):
    state = np.asarray(state, dtype=np.float64)
    stateVar = np.maximum(
        np.asarray(stateVar, dtype=np.float64),
        core.UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR,
    )
    obsMean = np.maximum(
        np.nanmean(np.asarray(matrixMunc, dtype=np.float64), axis=0),
        core.UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR,
    )
    slope = np.zeros_like(state)
    slope[1:] = np.diff(state)
    raw = np.column_stack(
        [
            np.log(stateVar),
            np.log(obsMean),
            np.abs(state),
            np.abs(slope),
            (
                np.abs(state)
                > np.nanquantile(
                    np.abs(state),
                    core.UNCERTAINTY_CALIBRATION_FEATURE_HIGH_SIGNAL_QUANTILE,
                )
            ).astype(np.float64),
        ]
    )
    center = np.nanmedian(raw, axis=0)
    scale = (
        np.nanmedian(np.abs(raw - center[None, :]), axis=0)
        * core.UNCERTAINTY_CALIBRATION_FEATURE_MAD_NORMAL_SCALE
    )
    scale = np.where(
        np.isfinite(scale)
        & (scale > core.UNCERTAINTY_CALIBRATION_FEATURE_SCALE_FLOOR),
        scale,
        1.0,
    )
    z = np.nan_to_num((raw - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    return np.column_stack([np.ones(state.size, dtype=np.float64), z]), center, scale


def _caseCythonFeatureMatrixMatchesPythonForFloat32AndFloat64():
    state = np.array([0.2, 0.5, -0.1, 0.8, 1.3, 0.4], dtype=np.float64)
    stateVar = np.array([0.1, 0.2, 0.15, 0.4, 0.8, 0.3], dtype=np.float64)
    matrixMunc = np.array(
        [[0.2, 0.3, 0.4, 0.2, 0.7, 0.9], [0.1, 0.5, 0.3, 0.6, 0.8, 1.0]],
        dtype=np.float64,
    )
    expectedX, expectedCenter, expectedScale = _pythonFeatureMatrix(state, stateVar, matrixMunc)

    for dtype in (np.float32, np.float64):
        X, center, scale = cuncertainty.cfeatureMatrix(
            np.ascontiguousarray(state, dtype=dtype),
            np.ascontiguousarray(stateVar, dtype=dtype),
            np.ascontiguousarray(matrixMunc, dtype=dtype),
            float(core.UNCERTAINTY_CALIBRATION_FEATURE_HIGH_SIGNAL_QUANTILE),
            float(core.UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR),
            float(core.UNCERTAINTY_CALIBRATION_FEATURE_MAD_NORMAL_SCALE),
            float(core.UNCERTAINTY_CALIBRATION_FEATURE_SCALE_FLOOR),
        )
        assert np.allclose(X, expectedX, atol=1.0e-6)
        assert np.allclose(center, expectedCenter, atol=1.0e-6)
        assert np.allclose(scale, expectedScale, atol=1.0e-6)


def _caseCythonFactorEvaluation():
    pState = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    features = np.ascontiguousarray(np.column_stack([np.ones(4), np.arange(4)]), dtype=np.float64)
    beta = np.array([np.log(2.0), 0.0], dtype=np.float64)
    factor, calibrated = cuncertainty.cevaluateFactor(
        features,
        beta,
        pState.astype(np.float64),
        core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
        core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
    )
    assert np.allclose(factor, 2.0)
    assert np.allclose(calibrated, np.sqrt(2.0 * pState))


def _caseCythonDeletedStateScoresAndDeleteBlockScores():
    fullState = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    deletedState = np.array([9.0, 22.0, 31.0, 37.0], dtype=np.float64)
    deletedStateVar = np.array([1.0, 4.0, 0.5, 9.0], dtype=np.float64)
    activeMask = np.array(
        [
            [1, 0, 1, 1],
            [1, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    foldMask = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )

    residual, pState, ii, fold, heldoutCount, keptCount = (
        cuncertainty.cextractDeletedStateScores(
            fullState,
            deletedState,
            deletedStateVar,
            activeMask,
            foldMask,
            2,
            float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
    )

    assert ii.tolist() == [0, 1, 3]
    assert fold.tolist() == [2, 2, 2]
    assert np.allclose(residual, [1.0, -2.0, 3.0])
    assert np.allclose(pState, [1.0, 4.0, 9.0])
    assert heldoutCount.tolist() == [1, 1, 1]
    assert keptCount.tolist() == [1, 0, 1]

    factorByInterval = np.array([1.0, 4.0, 1.0, 1.0], dtype=np.float64)
    blockIndex = (ii // 2).astype(np.int64, copy=False)
    targetMask = np.array([1, 1], dtype=np.uint8)

    blocks, scores, counts = cuncertainty.cdeleteBlockBlockScores(
        residual,
        pState,
        factorByInterval,
        ii,
        blockIndex,
        targetMask,
        heldoutCount,
        float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )

    assert blocks.tolist() == [0, 1]
    assert np.allclose(scores, [1.0, 1.0])
    assert counts.tolist() == [2, 1]


def _caseCythonSummaryContracts():
    residual = np.array([-1.0, -0.2, 0.1, 1.3], dtype=np.float64)
    pDelta = np.full(4, 0.5, dtype=np.float64)
    targets = np.array(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[:2], dtype=np.float64)
    targetZ = np.array([uncertainty._normalZ(t) for t in targets], dtype=np.float64)

    summary = cuncertainty.csummarizeCoverageWidths(
        residual,
        np.sqrt(pDelta),
        np.sqrt(2.0 * pDelta),
        np.array([0, 0, 1, 1], dtype=np.int32),
        targets,
        targetZ,
        float(core.UNCERTAINTY_CALIBRATION_SUMMARY_MEDIAN_QUANTILE),
        float(core.UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE),
    )
    assert set(summary) >= {"group", "target", "coverage_before", "mean_width_after"}
    assert summary["group"].tolist() == [-1, 0, 1, -1, 0, 1]


def _caseSegShrinkCythonParityContract():
    segment = cuncertainty.csegShrinkSegmentCodes(10, 5)
    assert segment.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert cuncertainty.csegShrinkSegmentCodes(3, 5).tolist() == [0, 1, 2]

    contigScope, segmentScope = cuncertainty.csegShrinkScopeCodes(
        2,
        segment,
        np.array([0, 1, 8, 9, -1, 10], dtype=np.int64),
    )
    assert contigScope.tolist() == [2, 2, 2, 2, 2, 2]
    assert segmentScope.tolist() == [10, 10, 14, 14, -1, -1]

    blockIDX = np.array([2, 2, 3, 3], dtype=np.int64)
    group = cuncertainty.csegShrinkGroupCodes(
        1,
        np.array([0, 1, 0, 1], dtype=np.int64),
        blockIDX,
    )
    assert group.tolist() == [10, 14, 11, 15]

    multipliers = segshrink.bootstrapMultipliers(
        groupCount=3,
        replicateCount=9,
        seed=17,
    )
    assert multipliers.shape == (9, 3)
    assert np.array_equal(
        multipliers,
        segshrink.bootstrapMultipliers(
            groupCount=3,
            replicateCount=9,
            seed=17,
        ),
    )

    baseLog, bootLog = cuncertainty.csegShrinkBootstrapLogFactorsCompact(
        np.array([1.0, 2.0, 3.0, 2.0, 4.0, 8.0], dtype=np.float64),
        np.ones(6, dtype=np.float64),
        np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
        np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
        np.array([0, 3, 6], dtype=np.int64),
        0.5,
        1.0,
        0.01,
        100.0,
    )
    assert np.allclose(baseLog, np.log([4.0, 16.0]))
    assert bootLog.shape == (2, 3)
    assert np.allclose(bootLog[:, 0], baseLog)
    assert np.allclose(bootLog[:, 1], np.log([1.0, 4.0]))
    assert np.allclose(bootLog[:, 2], np.log([4.0, 16.0]))

    empiricalBayes = cuncertainty.csegShrinkEmpiricalBayes(
        0.0,
        np.array([0.2, -0.1], dtype=np.float64),
        np.array([0.02, 0.03], dtype=np.float64),
        np.array([0.5, 0.1, -0.2], dtype=np.float64),
        np.array([0.04, 0.05, 0.04], dtype=np.float64),
        np.array([0, 0, 1], dtype=np.int32),
    )
    assert empiricalBayes["tauContigSq"] >= 0.0
    assert empiricalBayes["tauSegmentSq"] >= 0.0
    assert np.all(np.isfinite(empiricalBayes["segmentTheta"]))
    assert np.all((empiricalBayes["segmentAlpha"] >= 0.0) & (empiricalBayes["segmentAlpha"] <= 1.0))

    factor, calibrated = cuncertainty.csegShrinkApplyFactors(
        np.array([0, 1, 2, 1], dtype=np.int32),
        np.log(np.array([4.0, 1.0, 0.25], dtype=np.float64)),
        np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float64),
        float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )
    assert np.allclose(factor, [4.0, 1.0, 0.25, 1.0])
    assert np.allclose(calibrated, [2.0, 2.0, 1.5, 4.0])


def _caseCalibrateChromosomeStateUncertaintySmoke(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=uncertainty.logger.name)
    caplog.clear()
    rng = np.random.default_rng(123)
    n = 48
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signal = np.sin(grid).astype(np.float32)
    matrixData = np.vstack(
        [
            signal + 0.03 * rng.normal(size=n),
            signal + 0.04 * rng.normal(size=n) + 0.02,
            signal + 0.05 * rng.normal(size=n) - 0.02,
        ]
    ).astype(np.float32)
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    full = core.runConsenrich(matrixData, matrixMunc, **_smallRunKwargs())
    fullState, fullCovar, _resid, _track4, _blockMap = full

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=120,
        calibrationECMIters=1,
        minHeldoutCells=1,
        maxHeldoutCells=12,
        maxDiagnosticRows=5,
        targets=core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[:2],
        writeDiagnostics=True,
        seed=77,
    )

    result = uncertainty.calibrateChromosomeStateUncertainty(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        fullState=fullState,
        fullCovar=fullCovar,
        fullBackground=np.zeros(n, dtype=np.float32),
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=_smallRunKwargs(),
        outPrefix=str(tmp_path / "cal"),
    )

    assert result.factor.shape == (n,)
    assert result.calibratedUncertainty.shape == (n,)
    assert np.all(
        result.calibratedUncertainty + 1.0e-7 >= np.sqrt(fullCovar[:, 0, 0])
    )
    assert np.all(np.isfinite(result.factor))
    assert np.all(result.factor > 0.0)
    assert {"coverage_before", "coverage_after", "mean_width_after"} <= set(
        result.summary.columns
    )
    diagnosticsPath = tmp_path / "cal.delete_block_calibration.jsonl"
    assert diagnosticsPath.exists()
    diagnostics = [
        json.loads(line)
        for line in diagnosticsPath.read_text(encoding="utf-8").splitlines()
    ]
    recordTypes = {record["record_type"] for record in diagnostics}
    assert {"score_sample", "summary", "model", "fold", "invalid_reason"} <= recordTypes
    assert sum(record["record_type"] == "score_sample" for record in diagnostics) <= 5
    assert not (tmp_path / "cal.delete_block_calibration.log").exists()
    assert not (tmp_path / "cal.diagnostics.tsv.gz").exists()
    assert not (tmp_path / "cal.model.json").exists()
    modelPath = tmp_path / "cal.model.json"
    assert not modelPath.exists()
    model = result.model
    factorValues = np.asarray(result.factor, dtype=np.float64)
    factorMedian = float(np.median(factorValues))
    factorDistribution = model["delete_block_factor_distribution"]
    assert factorDistribution["count"] == int(factorValues.size)
    assert factorDistribution["median"] == pytest.approx(factorMedian)
    assert factorDistribution["unscaled_mad"] == pytest.approx(
        float(np.median(np.abs(factorValues - factorMedian)))
    )
    assert factorDistribution["q05"] == pytest.approx(
        float(
            np.quantile(
                factorValues,
                0.05,
                method=factorDistribution["quantile_method"],
            )
        )
    )
    assert factorDistribution["q95"] == pytest.approx(
        float(
            np.quantile(
                factorValues,
                0.95,
                method=factorDistribution["quantile_method"],
            )
        )
    )
    assert factorDistribution["min"] == pytest.approx(float(np.min(factorValues)))
    assert factorDistribution["max"] == pytest.approx(float(np.max(factorValues)))
    sdFactorValues = np.sqrt(factorValues)
    sdFactorMedian = float(np.median(sdFactorValues))
    assert factorDistribution["sd_multiplier_median"] == pytest.approx(
        sdFactorMedian
    )
    assert factorDistribution["sd_multiplier_unscaled_mad"] == pytest.approx(
        float(np.median(np.abs(sdFactorValues - sdFactorMedian)))
    )
    assert factorDistribution["sd_multiplier_q05"] == pytest.approx(
        float(
            np.quantile(
                sdFactorValues,
                0.05,
                method=factorDistribution["quantile_method"],
            )
        )
    )
    assert factorDistribution["sd_multiplier_q95"] == pytest.approx(
        float(
            np.quantile(
                sdFactorValues,
                0.95,
                method=factorDistribution["quantile_method"],
            )
        )
    )
    assert factorDistribution["sd_multiplier_min"] == pytest.approx(
        float(np.min(sdFactorValues))
    )
    assert factorDistribution["sd_multiplier_max"] == pytest.approx(
        float(np.max(sdFactorValues))
    )
    flatFactorKeys = {
        "factor_count",
        "factor_min",
        "factor_median",
        "factor_unscaled_mad",
        "factor_q05",
        "factor_q95",
        "factor_max",
        "factor_sd_multiplier_median",
    }
    assert model["mode"] == "delete_block_state"
    assert model["score_definition"] == "masked_minus_full_target_signal_over_delta_sd"
    assert model["factor_model"] == "segShrink"
    assert model["model_se_floor_applied"] is True
    assert model["model_se_floor_hits"] >= 0
    assert model["replicate_dependence"]["source"] == "auto"
    assert 0.0 <= model["replicate_dependence"]["rho"] < 1.0
    assert model["replicate_dependence"]["total_deff_median"] >= 1.0
    assert model["factorModel"] == "segShrink"
    assert (
        model["segmentCount"]
        == min(
            n,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_SEGMENT_COUNT,
        )
    )
    assert (
        model["bootstrapReplicates"]
        == core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_BOOTSTRAP_REPLICATES
    )
    assert "global_factor" in model
    assert "objective" not in model
    assert model["target_calibration"]["enabled"] is True
    assert model["target_calibration"]["delta"] == params.targetCalibrationDelta
    assert model["target_calibration"]["score_definition"] == (
        "max_abs_masked_minus_full_target_signal_over_delta_sd_by_block"
    )
    bounds = model["target_calibration"]["bounds"]
    assert len(bounds) == len(params.targets)
    assert [(row["target_role"], row["bound_available"], row["bound_scope"])
            for row in bounds] == [
        ("descriptive", False, None),
        ("selected", True, "chromosome_selected_target_conditional_exchangeability")]
    assert isinstance(
        model["target_calibration"]["scale_uncertainty_by_target_calibration"],
        bool,
    )
    if model["target_calibration"]["uncertainty_track_scaled"]:
        assert model["target_calibration"]["uncertainty_track_scale"] == pytest.approx(
            model["target_calibration"]["uncertainty_track_scale_q"]
            / model["target_calibration"]["uncertainty_track_scale_target_z"]
        )
    assert flatFactorKeys.isdisjoint(model)
    assert {"holdout_replicates_per_block", "heldout_cells", "fit_heldout_cells"}.isdisjoint(
        model
    )
    assert "predictive" not in json.dumps(model).lower()
    assert model["state_roughness"]["block_len_intervals"] == (
        diagnostic_utils.resolveUncertaintyBlockSizeIntervals(
            params.blockSizeBP,
            25,
            n,
        )
    )
    assert model["state_roughness"]["overall_mean_abs_diff"] is not None
    assert model["rows_valid"] >= model["rows_fit"]
    assert model["rows_fit"] <= 12
    assert model["fold_refits"]["delete_block_deletion_probability"] == pytest.approx(
        params.deleteBlockDeletionProbability
    )
    assert model["fold_refits"]["deleted_replicate_count_max"] >= 1
    assert sum(model["variance_source_counts"].values()) == model["rows_valid"]
    assert model["diagnostic_score_rows"] <= 5
    coverageRows = model["state_uncertainty_coverage"]
    coverageFitRows = model["state_uncertainty_coverage_fit"]
    assert any(row["stratum"] == "overall" for row in coverageRows)
    assert any(str(row["stratum"]).startswith("signal_abs_q") for row in coverageRows)
    overallRows = [row for row in coverageRows if row["stratum"] == "overall"]
    overallFitRows = [row for row in coverageFitRows if row["stratum"] == "overall"]
    assert {row["target"] for row in overallRows} == set(params.targets)
    assert {row["target"] for row in overallFitRows} == set(params.targets)
    assert all(row["n"] == model["rows_valid"] for row in overallRows)
    assert all(row["n"] == model["rows_fit"] for row in overallFitRows)
    assert all("coverage_before" in row and "coverage_after" in row for row in coverageRows)
    assert {row["coverage_scope"] for row in coverageRows} == {"all_valid_rows_reuse_diagnostic"}
    assert {row["coverage_scope"] for row in coverageFitRows} == {"factor_fit_rows_reuse_diagnostic"}
    assert set(result.summary["coverage_scope"]) == {"factor_fit_rows_reuse_diagnostic"}
    assert "replicate" not in result.scores.columns
    assert "observation_variance" not in result.scores.columns
    assert {"deleted_target_signal_delta", "target_signal_full",
            "target_signal_masked"} <= set(result.scores.columns)
    assert "deleted_state_delta" not in result.scores.columns
    np.testing.assert_allclose(result.scores["deleted_target_signal_delta"],
                               result.scores["target_signal_masked"] - result.scores["target_signal_full"])
    assert "deleted_replicates" in result.scores.columns
    assert "deleted_observations" in result.scores.columns
    assert np.all(result.scores["deleted_replicates"] >= 1)
    assert np.all(result.scores["deleted_observations"] >= 1)
    assert "delta_variance" in result.scores.columns
    assert "delta_variance_source" in result.scores.columns
    modelRecord = next(
        record for record in diagnostics if record["record_type"] == "model"
    )
    assert flatFactorKeys.isdisjoint(modelRecord)
    assert (
        modelRecord["delete_block_factor_distribution"]
        == model["delete_block_factor_distribution"]
    )
    assert not (tmp_path / "cal.summary.tsv").exists()
    assert not (tmp_path / "cal.scores.tsv.gz").exists()
    assert "uncertaintyCalibration.target enabled=True" in caplog.text
    assert "blocksTargetScored=" in caplog.text
    assert "mode=delete_block_state" in caplog.text
    assert "deleteBlockRows=" in caplog.text
    assert "uncertaintyCalibration.coverage.delete_block_all" in caplog.text
    assert "uncertaintyCalibration.coverage.fit_sample" in caplog.text











def _caseCalibrationRefitsUseCheapProcessNoiseWarmup(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=uncertainty.logger.name)
    caplog.clear()
    n = 32
    m = 8
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signal = np.sin(grid).astype(np.float32)
    offsets = np.linspace(-0.035, 0.035, m, dtype=np.float32)
    matrixData = np.vstack([signal + offset for offset in offsets]).astype(np.float32)
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    originalObservationMask = np.ones_like(matrixData, dtype=np.uint8)
    originalObservationMask[0, :] = 0
    fullState = np.column_stack(
        [signal, np.gradient(signal).astype(np.float32)]
    ).astype(np.float32)
    fullCovar = np.zeros((n, 2, 2), dtype=np.float32)
    fullCovar[:, 0, 0] = 0.05
    fullCovar[:, 1, 1] = 0.01
    capturedKwargs = []
    capturedMasks = []

    def _fakeRunConsenrich(matrixDataArg, _matrixMuncArg, *, observationMask, **kwargs):
        capturedKwargs.append(dict(kwargs))
        capturedMasks.append(np.asarray(observationMask, dtype=np.uint8).copy())
        residual = np.asarray(matrixDataArg, dtype=np.float32) - fullState[:, 0][None, :]
        return (
            fullState,
            fullCovar,
            residual.T,
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    monkeypatch.setattr(core, "runConsenrich", _fakeRunConsenrich)

    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = True
    runKwargs["processNoiseWarmupECMIters"] = 5
    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=100,
        calibrationECMIters=2,
        deleteBlockDeletionProbability=0.2,
        minHeldoutCells=1,
        maxHeldoutCells=24,
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        seed=21,
    )

    result = uncertainty.calibrateChromosomeStateUncertainty(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        fullState=fullState,
        fullCovar=fullCovar,
        fullBackground=np.zeros(n, dtype=np.float32),
        originalObservationMask=originalObservationMask,
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=runKwargs,
    )

    assert len(capturedKwargs) == params.folds
    assert len(capturedMasks) == params.folds
    assert all(np.all(mask[0, :] == 0) for mask in capturedMasks)
    assert all(kwargs.get("fitBackground") is True for kwargs in capturedKwargs)
    assert all(
        kwargs["ECM_outerIters"] == params.calibrationOuterIters
        for kwargs in capturedKwargs
    )
    assert all(
        kwargs["ECM_minOuterIters"] == 1 for kwargs in capturedKwargs
    )
    assert all(kwargs["ECM_fixedBackgroundIters"] == 2 for kwargs in capturedKwargs)
    assert all(
        kwargs["processNoiseWarmupECMIters"]
        == core.UNCERTAINTY_CALIBRATION_REFIT_PROCESS_NOISE_WARMUP_ECM_ITERS
        for kwargs in capturedKwargs
    )
    assert all(
        "processQWarmupOuterIters" not in kwargs
        for kwargs in capturedKwargs
    )
    foldRefits = result.model["fold_refits"]
    assert foldRefits["delete_block_deletion_probability"] == pytest.approx(
        params.deleteBlockDeletionProbability
    )
    assert foldRefits["deleted_replicate_count_min"] >= 1
    assert "holdout_count" not in foldRefits
    assert "holdout_fraction" not in foldRefits
    assert "deleteBlockDeletionProbability=0.2" in caplog.text
    assert "holdoutCount=" not in caplog.text
    blockLen = diagnostic_utils.resolveUncertaintyBlockSizeIntervals(
        params.blockSizeBP,
        25,
        n,
        folds=params.folds,
    )
    blockCount = (n + blockLen - 1) // blockLen
    rng = np.random.default_rng(params.seed)
    rng.permutation(blockCount)
    eligibleReplicateCount = m - 1
    expectedDeletedByBlock = np.empty(blockCount, dtype=np.int64)
    for block in range(blockCount):
        deleteCount = int(
            rng.binomial(
                eligibleReplicateCount,
                params.deleteBlockDeletionProbability,
            )
        )
        while deleteCount < 1 or deleteCount >= eligibleReplicateCount:
            deleteCount = int(
                rng.binomial(
                    eligibleReplicateCount,
                    params.deleteBlockDeletionProbability,
                )
            )
        expectedDeletedByBlock[block] = deleteCount
        rng.choice(eligibleReplicateCount, size=deleteCount, replace=False)
    maskStack = np.stack(capturedMasks, axis=0)
    deletedByBlock = np.empty(blockCount, dtype=np.int64)
    for block in range(blockCount):
        start = block * blockLen
        deletedByFold = np.sum(
            (maskStack[:, :, start] == 0)
            & (originalObservationMask[:, start][None, :] != 0),
            axis=1,
        )
        assert np.count_nonzero(deletedByFold) == 1
        deletedByBlock[block] = int(np.max(deletedByFold))
    assert np.array_equal(deletedByBlock, expectedDeletedByBlock)
    assert deletedByBlock.min() >= 1
    assert deletedByBlock.max() < eligibleReplicateCount
    assert len(set(deletedByBlock.tolist())) > 1
    combinedDeletedByInterval = np.sum(
        [np.sum(mask == 0, axis=0) for mask in capturedMasks],
        axis=0,
    )
    expectedDeletedByInterval = (
        np.repeat(expectedDeletedByBlock, blockLen)[:n] + params.folds
    )
    assert np.array_equal(
        combinedDeletedByInterval,
        expectedDeletedByInterval,
    )


def _caseUncertaintyCalibrationRouting():
    for calibrationEnabled in (False, True):
        for writeUncertainty in (False, True):
            for useStateShrinkage in (False, True):
                assert consenrichRuntime._uncertaintyCalibrationIsRequired(
                    calibrationEnabled,
                    writeUncertainty,
                    useStateShrinkage,
                ) is bool(
                    calibrationEnabled
                    and (writeUncertainty or useStateShrinkage)
                )


def test_stateShrinkageVariancePreservesUncertaintyInput():
    calibrated = np.array([0.0, 0.25, 0.75, 1.5], dtype=np.float32)
    finalized = segshrink.combinePreparedContigs(
        [
            {
                "chromosome": "chrTest",
                "model": {},
                "calibrated": calibrated,
            }
        ],
        positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )
    assert finalized[0]["calibrated"] is calibrated

    stridedFloat64 = np.array(
        [[0.125, -1.0], [0.5, -1.0], [2.0, -1.0]],
        dtype=np.float64,
    )[:, 0]
    assert not stridedFloat64.flags.c_contiguous

    for uncertaintyValues in (finalized[0]["calibrated"], stridedFloat64):
        uncertaintyBefore = uncertaintyValues.copy()
        variance = consenrichRuntime._stateShrinkageVariance(uncertaintyValues)
        expected = np.maximum(
            np.square(uncertaintyBefore.astype(np.float32)),
            np.float32(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )

        np.testing.assert_array_equal(uncertaintyValues, uncertaintyBefore)
        np.testing.assert_array_equal(variance, expected)
        assert variance.dtype == np.float32
        assert variance.flags.c_contiguous
        assert not np.shares_memory(variance, uncertaintyValues)


def _caseSegShrinkCalibrationContract(monkeypatch):
    n = 40
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signal = np.sin(grid).astype(np.float32)
    matrixData = np.vstack(
        [signal - 0.02, signal + 0.01, signal + 0.03]).astype(np.float32)
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    fullState = np.column_stack(
        [signal, np.gradient(signal).astype(np.float32)]).astype(np.float32)
    fullCovar = np.zeros((n, 2, 2), dtype=np.float32)
    fullCovar[:, 0, 0] = 0.05
    fullCovar[:, 1, 1] = 0.01
    capturedKwargs = []

    def _fakeRunConsenrich(matrixDataArg, _matrixMuncArg, *, observationMask, **kwargs):
        capturedKwargs.append(dict(kwargs))
        deleted = np.mean(np.asarray(observationMask, dtype=np.float32) == 0, axis=0)
        maskedState = fullState.copy()
        maskedState[:, 0] = maskedState[:, 0] + 1.0
        maskedCovar = fullCovar.copy()
        maskedCovar[:, 0, 0] = maskedCovar[:, 0, 0] + 0.04 + 0.01 * deleted
        residual = np.asarray(matrixDataArg, dtype=np.float32) - maskedState[:, 0][None, :]
        return (maskedState, maskedCovar, residual.T, np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32))
    monkeypatch.setattr(core, "runConsenrich", _fakeRunConsenrich)
    target = core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0]
    targetZ = 0.6744897501960817
    def _fixedTargetBounds(blockScores, *, targets, delta):
        assert tuple(targets) == (target,)
        return [{
            "target": target, "alpha": 1.0 - target, "delta": delta,
            "N": int(np.size(blockScores)), "k": 1, "q": 2.0 * targetZ,
            "q_source": "exchangeability_conditional_order_statistic",
            "bound_available": True,
            "bound_scope": "chromosome_selected_target_conditional_exchangeability",
            "binomial_tail": 0.0, "allowed_blocks_above_q": 0,
            "min_blocks_for_any_finite_bound": 1,
        }]

    targetBoundsFunction = uncertainty._targetCalibrationBounds
    monkeypatch.setattr(uncertainty, "_targetCalibrationBounds", _fixedTargetBounds)
    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=100,
        calibrationECMIters=1,
        calibrationOuterIters=9,
        minHeldoutCells=1,
        maxHeldoutCells=40,
        targets=(target,),
        targetCalibrationDelta=0.5, scaleUncertaintyByTargetCalibration=True,
        deleteBlockVarianceMode="covariance_difference",
        deleteBlockReplicateDependenceRho=0.25,
        deleteBlockFactorModel="segShrink",
        deleteBlockFactorSegmentCount=4,
        deleteBlockFactorBootstrapReplicates=8,
        seed=41,
    )

    result = uncertainty.calibrateChromosomeStateUncertainty(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        fullState=fullState,
        fullCovar=fullCovar,
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=_smallRunKwargs(),
    )
    monkeypatch.setattr(uncertainty, "_targetCalibrationBounds", targetBoundsFunction)

    model = result.model
    assert model["factor_model"] == "segShrink"
    assert model["replicate_dependence"]["source"] == "fixed"
    assert model["replicate_dependence"]["rho"] == pytest.approx(0.25)
    assert model["replicate_dependence"]["applied"] is True
    assert model["replicate_dependence"]["total_deff_median"] >= 1.0
    assert model["replicate_dependence"]["heldout_deff_median"] >= 1.0
    assert model["hierarchyScope"] == "singleProcessedContig"
    assert model["processedContigCount"] == 1
    assert model["blockIDXUnitCount"] >= 1
    assert set(model["refitPolicy"]) >= {
        "ECM_outerIters", "ECM_minOuterIters", "ECM_fixedBackgroundIters",
        "processNoiseWarmupECMIters"}
    assert model["refitPolicy"]["ECM_outerIters"] == 4
    assert model["refitPolicy"]["ECM_minOuterIters"] == 1
    assert model["refitPolicy"]["ECM_fixedBackgroundIters"] == 2
    assert len(model["segmentShrinkage"]) == 4
    assert {"blockIDX", "factor_segment", "segment_shrinkage_weight"} <= set(result.scores)
    assert all((row["ECM_outerIters"], row["ECM_fixedBackgroundIters"],
                row["ECM_minOuterIters"]) == (4, 2, 1) for row in capturedKwargs)
    targetMeta = model["target_calibration"]
    assert targetMeta["uncertainty_track_scale"] == pytest.approx(2.0)
    assert targetMeta["uncertainty_track_scale_q"] == pytest.approx(2.0 * targetZ)
    assert targetMeta["uncertainty_track_scale_bound_available"] is True
    assert targetMeta["uncertainty_track_scale_bound_scope"] == "chromosome_selected_target_conditional_exchangeability"
    assert targetMeta["score_definition"] == "max_abs_masked_minus_full_target_signal_over_delta_sd_by_block"
    factor = np.asarray(result.factor, dtype=np.float64)
    expectedTrack = np.maximum(np.sqrt(np.maximum(
        factor * fullCovar[:, 0, 0], core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR)) * 2.0,
        np.sqrt(fullCovar[:, 0, 0])).astype(np.float32)
    np.testing.assert_allclose(result.calibratedUncertainty, expectedTrack)
    assert not np.allclose(expectedTrack, np.sqrt(factor * fullCovar[:, 0, 0]))
    scoreInterval = result.scores["interval_index"].to_numpy(dtype=np.int64)
    scoreDelta = result.scores["delta_variance"].to_numpy(dtype=np.float64)
    baseSD = np.sqrt(np.maximum(factor[scoreInterval] * scoreDelta, core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR))
    expectedSD = baseSD * 2.0
    np.testing.assert_allclose(result.scores["sd_after"], expectedSD)
    assert not np.allclose(expectedSD, baseSD)
    absResidual = np.abs(result.scores["residual"].to_numpy(dtype=np.float64))
    unscaledCoverage = np.mean(absResidual <= targetZ * baseSD)
    expectedCoverage = np.mean(absResidual <= targetZ * expectedSD)
    assert expectedCoverage != unscaledCoverage
    summaryOverall = result.summary.query("stratum == 'overall'").iloc[0]
    fitOverall = next(row for row in model["state_uncertainty_coverage_fit"] if row["stratum"] == "overall")
    assert summaryOverall["coverage_after"] == pytest.approx(expectedCoverage)
    assert fitOverall["coverage_after"] == pytest.approx(expectedCoverage)
    assert summaryOverall["mean_width_after"] == pytest.approx(2.0 * targetZ * np.mean(expectedSD))
    assert model["model_se_floor_hits"] == 0
    assert model["coverage_estimand"] == "delete_block_target_signal_perturbation"
    assert model["coverage_scope"] == "all_valid_rows_reuse_diagnostic"


def _runSampledLPOAutoRhoCase(
    monkeypatch,
    *,
    m=4,
    deletePairs=True,
    useBg=False,
    varianceMode="covariance_difference",
):
    n = 64
    blockLen = 8
    blockCount = n // blockLen
    pattern = np.tile(
        np.array([-1.0, -0.4, 0.2, 0.8, 1.0, 0.5, -0.2, -0.7], dtype=np.float32),
        blockCount,
    )
    bg = (0.4 * pattern).astype(np.float32) if useBg else np.zeros(n, dtype=np.float32)
    matrixData = np.zeros((m, n), dtype=np.float32)
    if useBg:
        matrixData[:] = bg[None, :]
    matrixMunc = np.full((m, n), 0.1, dtype=np.float32)
    fullState = np.zeros((n, 2), dtype=np.float32)
    fullCovar = np.zeros((n, 2, 2), dtype=np.float32)
    fullCovar[:, 0, 0] = 0.25
    fullCovar[:, 1, 1] = 0.01

    def _fakeMakeFoldSpec(**_kwargs):
        blockFold = (np.arange(blockCount, dtype=np.int32) % 2).astype(np.int32)
        if deletePairs:
            pairTemplates = np.array(
                [[0, 1, -1, -1], [2, 3, -1, -1], [0, 2, -1, -1], [1, 3, -1, -1]],
                dtype=np.intp,
            )
            repsByBlock = pairTemplates[
                np.arange(blockCount, dtype=np.int64) % pairTemplates.shape[0]
            ].copy()
            repsByBlockCount = np.full(blockCount, 2, dtype=np.intp)
        else:
            repsByBlock = np.full((blockCount, m), -1, dtype=np.intp)
            repsByBlock[:, 0] = np.arange(blockCount, dtype=np.intp) % m
            repsByBlockCount = np.ones(blockCount, dtype=np.intp)
        return blockFold, repsByBlockCount, repsByBlock

    def _fakeRunConsenrich(matrixDataArg, _matrixMuncArg, *, observationMask, **_kwargs):
        deleted = np.mean(np.asarray(observationMask, dtype=np.float32) == 0, axis=0)
        maskedState = fullState.copy()
        if not useBg:
            maskedState[:, 0] = 0.5 * deleted * pattern
        maskedCovar = fullCovar.copy()
        maskedCovar[:, 0, 0] = fullCovar[:, 0, 0] + 0.05 + 0.02 * deleted
        signal = maskedState[:, 0] + bg
        residual = np.asarray(matrixDataArg, dtype=np.float32) - signal[None, :]
        return (
            maskedState,
            maskedCovar,
            residual.T,
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.int32),
            bg.copy(),
        )

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=200,
        calibrationECMIters=1,
        minHeldoutCells=1,
        maxHeldoutCells=64,
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        targetCalibrationDelta=None,
        deleteBlockVarianceMode=varianceMode,
        deleteBlockReplicateDependenceRho="auto",
        deleteBlockTargetSignal="state_plus_background" if useBg else "state",
        deleteBlockFactorModel="global",
        seed=71,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = bool(useBg)

    with monkeypatch.context() as scopedPatch:
        scopedPatch.setattr(uncertainty, "_makeFoldSpec", _fakeMakeFoldSpec)
        scopedPatch.setattr(core, "runConsenrich", _fakeRunConsenrich)
        return uncertainty.calibrateChromosomeStateUncertainty(
            matrixData=matrixData,
            matrixMunc=matrixMunc,
            fullState=fullState,
            fullCovar=fullCovar,
            fullBackground=bg if useBg else None,
            intervals=np.arange(n, dtype=np.int64) * 25,
            intervalSizeBP=25,
            params=params,
            runKwargs=runKwargs,
        )


def _caseSampledLPOAutoRhoUsesDeleteBlockRefits(monkeypatch):
    result = _runSampledLPOAutoRhoCase(monkeypatch)
    dep = result.model["replicate_dependence"]

    assert dep["source"] == "auto"
    assert dep["estimator"] == "delete_block_sampled_leave_pair_out_fisher_z"
    assert dep["rho"] > 0.0
    assert dep["applied"] is True
    assert dep["block_count"] > 0
    assert dep["pair_count"] > 0
    assert dep["support_passed"] is True
    assert dep["same_fold_evidence_excluded"] is True


def _caseSampledLPOAutoRhoNoDeletedPairs(monkeypatch):
    result = _runSampledLPOAutoRhoCase(monkeypatch, m=2, deletePairs=False)
    dep = result.model["replicate_dependence"]

    assert dep["source"] == "auto"
    assert dep["rho"] == pytest.approx(0.0)
    assert dep["applied"] is False
    assert dep["pair_count"] == 0
    assert result.model["rows_valid"] > 0


def _caseSampledLPOAutoRhoSubtractsMaskedBg(monkeypatch):
    result = _runSampledLPOAutoRhoCase(monkeypatch, useBg=True)
    dep = result.model["replicate_dependence"]

    assert result.model["target_signal"] == "state_plus_background"
    assert result.model["fold_refits"]["deleted_replicate_count_min"] == 2
    assert dep["rho"] == pytest.approx(0.0)


def _caseSampledLPOAutoRhoRecomputesEffectiveInformation(monkeypatch):
    result = _runSampledLPOAutoRhoCase(
        monkeypatch,
        varianceMode="heldout_information",
    )
    row = result.scores.iloc[0]
    rho = result.model["replicate_dependence"]["rho_by_fold"][int(row["fold"])]
    sampleInfo = 1.0 / (0.1 + _smallRunKwargs()["pad"])
    total = 4.0 * sampleInfo / (1.0 + 3.0 * rho)
    kept = 2.0 * sampleInfo / (1.0 + rho)
    held = total - kept
    h = held / total

    assert rho > 0.0
    assert row["total_information"] == pytest.approx(total)
    assert row["kept_information"] == pytest.approx(kept)
    assert row["heldout_information"] == pytest.approx(held)
    assert row["heldout_information_fraction"] == pytest.approx(h)
    assert row["delta_variance"] == pytest.approx(0.25 * h / (1.0 - h))
    assert row["delta_variance_source"] == "heldout_information"


def _caseCalibrationFloorAppliesToGlobalAndSegShrink(monkeypatch):
    n = 24
    m = 4
    matrixData = np.zeros((m, n), dtype=np.float32)
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    fullP = np.linspace(0.04, 0.21, n, dtype=np.float64)
    fullState = np.zeros((n, 2), dtype=np.float32)
    fullCovar = np.zeros((n, 2, 2), dtype=np.float64)
    fullCovar[:, 0, 0] = fullP
    fullCovar[:, 1, 1] = 0.01

    def _fakeRunConsenrich(matrixDataArg, _matrixMuncArg, *, observationMask, **_kwargs):
        deleted = np.mean(np.asarray(observationMask, dtype=np.float32) == 0, axis=0)
        maskedState = fullState.copy()
        maskedState[:, 0] = 0.05 + 0.02 * deleted
        maskedCovar = fullCovar.copy()
        maskedCovar[:, 0, 0] = fullP + 0.03 + 0.01 * deleted
        residual = np.asarray(matrixDataArg, dtype=np.float32) - maskedState[:, 0][None, :]
        return (
            maskedState,
            maskedCovar,
            residual.T,
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    def _fakeGlobalFactor(*, params, **_kwargs):
        target = float(max(params.targets))
        return 0.25, {
            "success": True,
            "factor_model": "global",
            "global_factor": 0.25,
            "global_sd_multiplier": 0.5,
            "global_factor_target": target,
            "global_factor_target_z": uncertainty._normalZ(target),
        }

    def _fakeSegShrinkFit(*, fullP, target, targetZ, **_kwargs):
        fullPArr = np.asarray(fullP, dtype=np.float64).reshape(-1)
        factor = np.full(fullPArr.shape, 0.25, dtype=np.float64)
        return {
            "factor": factor,
            "calibrated": np.sqrt(factor * fullPArr).astype(np.float32),
            "segmentByInterval": np.zeros(fullPArr.shape[0], dtype=np.int32),
            "segmentRawLogFactor": np.log(np.array([0.25], dtype=np.float64)),
            "segmentBootstrapVariance": np.array([0.0], dtype=np.float64),
            "segmentShrinkageWeight": np.array([1.0], dtype=np.float64),
            "modelMeta": {
                "success": True,
                "factor_model": "segShrink",
                "factorModel": "segShrink",
                "global_factor": 0.25,
                "global_sd_multiplier": 0.5,
                "global_factor_target": float(target),
                "global_factor_target_z": float(targetZ),
                "segmentShrinkage": [{"factor": 0.25}],
            },
        }

    monkeypatch.setattr(core, "runConsenrich", _fakeRunConsenrich)
    monkeypatch.setattr(uncertainty, "_fitDeleteBlockGlobalFactor", _fakeGlobalFactor)
    monkeypatch.setattr(segshrink, "fitSingleContig", _fakeSegShrinkFit)

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=100,
        calibrationECMIters=1,
        minHeldoutCells=1,
        maxHeldoutCells=48,
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        targetCalibrationDelta=None,
        deleteBlockDeletionProbability=0.5,
        deleteBlockVarianceMode="covariance_difference",
        seed=59,
    )

    for factorModel, covarianceKwargs in (
        ("global", {"fullCovar": fullCovar}),
        ("segShrink", {"fullP": fullP}),
    ):
        result = uncertainty.calibrateChromosomeStateUncertainty(
            matrixData=matrixData,
            matrixMunc=matrixMunc,
            fullState=fullState,
            intervals=np.arange(n, dtype=np.int64) * 25,
            intervalSizeBP=25,
            params=params._replace(deleteBlockFactorModel=factorModel),
            runKwargs=_smallRunKwargs(),
            **covarianceKwargs,
        )

        floor = np.sqrt(fullP).astype(np.float32)
        assert result.model["factor_model"] == factorModel
        assert result.model["model_se_floor_hits"] == n
        assert np.all(result.factor >= 1.0)
        assert np.all(result.calibratedUncertainty + 1.0e-7 >= floor)
        assert np.any(np.isclose(result.calibratedUncertainty, floor))


def _caseSegShrinkProcessedContigContract(tmp_path, monkeypatch):
    positiveFloor = float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR)
    allScope = "all_valid_rows_reuse_diagnostic"
    fitScope = "factor_fit_rows_reuse_diagnostic"
    boundScope = "chromosome_selected_target_conditional_exchangeability"
    targetSignal = core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_TARGET_SIGNAL
    with pytest.raises(ValueError, match="no processed contigs"):
        segshrink.combinePreparedContigs([], positiveFloor=positiveFloor)
    directCalibrated = np.asarray([0.5, 1.0], dtype=np.float32)
    direct = segshrink.combinePreparedContigs(
        [{"model": {}, "factor": np.ones(2), "calibrated": directCalibrated}],
        positiveFloor=positiveFloor)[0]
    assert direct["calibrated"] is directCalibrated
    assert direct["model"]["hierarchyScope"] == "singleProcessedContig"
    n = 8
    fitRows = np.asarray([0, 2, 4, 6], dtype=np.int64)
    signalAbs = np.arange(n, dtype=np.float64)
    cuts = np.quantile(signalAbs, np.linspace(0.0, 1.0, 6))
    coverageCode = np.searchsorted(cuts[1:], signalAbs, side="left").astype(np.int32)
    pDelta = np.asarray([1.0, 4.0] * 4, dtype=np.float64)
    targets = (0.25, 0.5)
    selectedTarget, selectedZ, targetDelta = 0.5, 0.6744897501960817, 0.3
    prepared = []
    replayArrays = []
    replaySpecs = (
        ("chrA", 0.25, (0.1, 0.4), (1, 0, 1, 0), (8, 6, 1000, 900, 10, 9, 800, 700)),
        ("chrC", 16.0, (8.0, 32.0), (1, 0, 0, 0), (20, 18, 2, 1, 4, 3, 6, 5)),
    )
    for ordinal, (chromosome, rawFactor, segmentRaw, targetMask, residual) in enumerate(replaySpecs):
        arrays = {
            "residual": np.asarray(residual, dtype=np.float64),
            "pDelta": pDelta.copy(),
            "intervalIndex": np.arange(n, dtype=np.int64),
            "fitRows": fitRows.copy(),
            "targetBlockMask": np.asarray(targetMask, dtype=np.uint8),
            "deletedObservationAll": np.arange(1, n + 1, dtype=np.int64),
            "coverageCodeAll": coverageCode.copy(),
            "coverageCodeFit": coverageCode[fitRows].copy(),
            "summaryDecile": np.asarray([0, 1, 0, 1], dtype=np.int32),
        }
        replayPath = tmp_path / f"{chromosome}.npz"
        uncertainty._writeCalibrationReplay(replayPath, arrays)
        if ordinal == 0:
            loaded = uncertainty._loadCalibrationReplay(
                replayPath, intervalCount=n, blockLenIntervals=2, positiveFloor=positiveFloor)
            assert set(loaded) == set(arrays)
            assert all(
                loaded[key].dtype == value.dtype and np.array_equal(loaded[key], value)
                for key, value in arrays.items())
            destinationBytes = replayPath.read_bytes()
            with monkeypatch.context() as scopedPatch:
                scopedPatch.setattr(
                    uncertainty.np, "savez",
                    lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("writer failure")))
                with pytest.raises(RuntimeError, match="writer failure"):
                    uncertainty._writeCalibrationReplay(replayPath, arrays)
            assert replayPath.read_bytes() == destinationBytes
            assert not list(tmp_path.glob(f".{replayPath.name}.*.tmp"))
        model = {
            "factor_model": "segShrink", "targets": list(targets),
            "target_signal": targetSignal, "block_len_intervals": 2,
            "fold_refits": {"block_len_intervals": 2}, "coverage_estimand": "stale",
            "coverage_scope": "stale", "score_definition": "stale",
            "diagnostic_score_rows": -1, "model_se_floor_hits": -1,
            "delete_block_factor_distribution": {"count": -1},
            "state_uncertainty_coverage": [{"stratum": "stale"}],
            "state_uncertainty_coverage_fit": [{"stratum": "stale"}],
            "target_calibration": {
                "enabled": True, "delta": targetDelta,
                "scale_uncertainty_by_target_calibration": True,
                "bounds": [{"q_source": "stale"}],
            },
            "contigShrinkage": [{"rawFactor": rawFactor, "bootstrapVariance": 0.05}],
            "segmentShrinkage": [
                {"segment": segment, "rows": n // 2, "rawFactor": value,
                 "bootstrapVariance": 0.05}
                for segment, value in enumerate(segmentRaw)
            ],
        }
        fullP = np.linspace(0.5, 2.0, n, dtype=np.float64)
        if ordinal == 0:
            fullP[0] = positiveFloor / 4.0
        prepared.append({
            "chromosome": chromosome, "intervals": np.arange(n, dtype=np.int64) * 25,
            "fullP": fullP, "model": model, "calibrationReplayPath": str(replayPath),
            "summaryRowIndex": ordinal,
        })
        replayArrays.append(arrays)
    modelBefore = [json.dumps(item["model"], sort_keys=True) for item in prepared]
    malformedPath = tmp_path / "malformed.npz"
    malformedArrays = {key: value.copy() for key, value in replayArrays[1].items()}
    malformedArrays["pDelta"][0] = positiveFloor
    uncertainty._writeCalibrationReplay(malformedPath, malformedArrays)
    badInputs = (
        ([{**prepared[0], "calibrationReplayPath": str(tmp_path / "missing.npz")}, prepared[1]], "does not exist"),
        ([{key: value for key, value in prepared[0].items() if key != "calibrationReplayPath"}, prepared[1]], "keys"),
        ([prepared[0], {**prepared[1], "calibrationReplayPath": str(malformedPath)}], "positive floor"),
    )
    incompatibilities = (
        ({"targets": [0.25, 0.6]}, {}, "selected maximum target"),
        ({"target_signal": "state"}, {}, "target signal"),
        ({}, {"scale_uncertainty_by_target_calibration": False}, "scale flag"),
        ({}, {"delta": 0.2}, "target-calibration delta"),
    )
    with monkeypatch.context() as scopedPatch:
        scopedPatch.setattr(
            segshrink._cuncertainty, "csegShrinkEmpiricalBayes",
            lambda *_args, **_kwargs: pytest.fail("EB reached invalid pooled input"))
        scopedPatch.setattr(
            uncertainty, "_evaluateSegShrinkReplay",
            lambda **_kwargs: pytest.fail("replay evaluated invalid pooled input"))
        for badPrepared, message in badInputs:
            with pytest.raises(ValueError, match=message):
                segshrink.combinePreparedContigs(badPrepared, positiveFloor=positiveFloor)
        for modelPatch, calibrationPatch, message in incompatibilities:
            badModel = json.loads(json.dumps(prepared[1]["model"]))
            badModel.update(modelPatch)
            badModel["target_calibration"].update(calibrationPatch)
            with pytest.raises(ValueError, match=message):
                segshrink.combinePreparedContigs(
                    [prepared[0], {**prepared[1], "model": badModel}], positiveFloor=positiveFloor)
    finalized = segshrink.combinePreparedContigs(prepared, positiveFloor=positiveFloor)
    assert [item["chromosome"] for item in finalized] == ["chrA", "chrC"]
    assert [json.dumps(item["model"], sort_keys=True) for item in prepared] == modelBefore
    assert all("calibrationReplayPath" not in item for item in finalized)
    assert all(path.exists() for path in (tmp_path / "chrA.npz", tmp_path / "chrC.npz"))
    stratumNames = {f"signal_abs_q{lower:02d}_{lower + 20:02d}" for lower in range(0, 100, 20)}
    for item, arrays, original in zip(finalized, replayArrays, prepared):
        model = item["model"]
        assert (model["hierarchyScope"], model["processedContigCount"]) == ("processedGenome", 2)
        assert model["target_signal"] == targetSignal
        assert model["coverage_estimand"] == "delete_block_target_signal_perturbation"
        assert (model["coverage_scope"], model["coverage_fit_scope"]) == (allScope, fitScope)
        assert model["score_definition"] == "masked_minus_full_target_signal_over_delta_sd"
        assert model["diagnostic_score_rows"] == 0
        assert "stale" not in json.dumps(model)
        segmentFactor = np.asarray([row["factor"] for row in model["segmentShrinkage"]])
        segmentCode = np.minimum(np.arange(n) * segmentFactor.size // n, segmentFactor.size - 1)
        factorRaw = segmentFactor[segmentCode]
        expectedFactor = np.maximum(factorRaw, 1.0)
        np.testing.assert_allclose(item["factor"], expectedFactor)
        assert np.any(segmentFactor != np.asarray([row["rawFactor"] for row in original["model"]["segmentShrinkage"]]))
        blockIndex = arrays["intervalIndex"] // 2
        score = np.abs(arrays["residual"]) / np.sqrt(
            expectedFactor[arrays["intervalIndex"]] * arrays["pDelta"])
        selectedBlocks = np.flatnonzero(arrays["targetBlockMask"])
        blockScores = np.asarray([np.max(score[blockIndex == block]) for block in selectedBlocks])
        N = int(blockScores.size)
        order = next(
            (k for k in range(1, N + 1) if sum(
                math.comb(N, j) * selectedTarget**j * (1.0 - selectedTarget) ** (N - j)
                for j in range(k, N + 1)) <= targetDelta), None)
        q = float(np.max(blockScores) if order is None else np.sort(blockScores)[order - 1])
        bounds = model["target_calibration"]["bounds"]
        bound = next(row for row in bounds if row["target_role"] == "selected")
        assert [(row["target_role"], row["bound_available"], row["bound_scope"])
                for row in bounds] == [
            ("descriptive", False, None),
            ("selected", order is not None, boundScope),
        ]
        assert (bound["N"], bound["k"]) == (N, order)
        assert bound["q"] == pytest.approx(q)
        scale = q / selectedZ if order is not None else 1.0
        assert model["target_calibration"]["uncertainty_track_scale"] == pytest.approx(scale)
        assert model["target_calibration"]["uncertainty_track_scale_bound_available"] is (order is not None)
        expectedReason = "scaled_by_exchangeability_conditional_order_bound_q_over_z" if order else "finite_order_bound_unavailable"
        assert model["target_calibration"]["uncertainty_track_scale_reason"] == expectedReason
        if order is None:
            assert bound["q_source"] == "empirical_max_without_finite_order_bound"
        else:
            assert np.max(score[~np.isin(blockIndex, selectedBlocks)]) > q
        effectiveFactor = np.maximum(expectedFactor * scale**2, 1.0)
        rawSD = np.sqrt(np.maximum(item["fullP"], positiveFloor))
        baseSD = np.sqrt(np.maximum(expectedFactor * item["fullP"], positiveFloor))
        expectedCalibrated = np.maximum(baseSD * scale, rawSD).astype(np.float32)
        np.testing.assert_allclose(item["calibrated"], expectedCalibrated)
        np.testing.assert_allclose(
            consenrichRuntime._stateShrinkageVariance(item["calibrated"]),
            np.maximum(expectedCalibrated**2, np.float32(positiveFloor)))
        assert model["model_se_floor_hits"] == np.count_nonzero(
            (factorRaw < 1.0) | (baseSD * scale < rawSD))
        if item["chromosome"] == "chrA":
            assert item["fullP"][0] < positiveFloor
            assert baseSD[0] == pytest.approx(np.sqrt(positiveFloor))
        sdAfter = np.sqrt(np.maximum(effectiveFactor[arrays["intervalIndex"]] * arrays["pDelta"], positiveFloor))
        coverageRows = model["state_uncertainty_coverage"]
        coverageFitRows = model["state_uncertainty_coverage_fit"]
        assert {row["stratum"] for row in coverageRows} == {"overall", *stratumNames}
        assert ({row["coverage_scope"] for row in coverageRows},
                {row["coverage_scope"] for row in coverageFitRows},
                set(item["summary"]["coverage_scope"])) == ({allScope}, {fitScope}, {fitScope})
        overall = next(row for row in coverageRows if row["stratum"] == "overall"
                       and row["target"] == selectedTarget)
        assert overall["n"] == n
        assert overall["coverage_after"] == pytest.approx(
            np.mean(np.abs(arrays["residual"]) <= selectedZ * sdAfter))
        assert overall["mean_width_after"] == pytest.approx(2.0 * selectedZ * np.mean(sdAfter))
        summaryOverall = item["summary"].query("stratum == 'overall' and target == @selectedTarget").iloc[0]
        assert summaryOverall["coverage_after"] == pytest.approx(
            np.mean(np.abs(arrays["residual"][fitRows]) <= selectedZ * sdAfter[fitRows]))
        assert summaryOverall["q90_width_after"] == pytest.approx(
            2.0 * selectedZ * np.quantile(sdAfter[fitRows], 0.9))
        distribution = model["delete_block_factor_distribution"]
        assert [distribution[key] for key in ("count", "median", "unscaled_mad")] == pytest.approx([
            n, np.median(expectedFactor), np.median(np.abs(expectedFactor - np.median(expectedFactor)))])
        expectedBlocks = np.asarray([np.mean(effectiveFactor[start:start + 2])
                                     for start in range(0, n, 2)])
        np.testing.assert_allclose(
            consenrichRuntime._deleteBlockBlockFactorValues(item["factor"], model),
            expectedBlocks)
        plotRows = consenrichRuntime._deleteBlockCoverageRowsForPlot(
            chromosome=item["chromosome"], calibrationModel=model, summary=item["summary"])
        assert {row["target_role"] for row in plotRows} == {"selected", "descriptive"}
        assert all(row["selected_target"] == selectedTarget for row in plotRows)
    logPath = tmp_path / "pooled.jsonl"
    consenrichRuntime._writeJsonlRecords(logPath, [])
    for item in finalized:
        consenrichRuntime._appendPooledDeleteBlockDiagnostics(
            logPath, item["chromosome"], item["summary"], item["model"])
    records = [json.loads(line) for line in logPath.read_text().splitlines()]
    assert not any(row["record_type"] == "score_sample" for row in records)
    for item in finalized:
        chromosomeRecords = [row for row in records if row["chromosome"] == item["chromosome"]]
        assert [sum(row["record_type"] == kind for row in chromosomeRecords) for kind in
                ("model", "summary", "target_bound")] == [
            1, len(item["summary"]), len(item["model"]["target_calibration"]["bounds"])
        ]
        summaryRecords = [row for row in chromosomeRecords if row["record_type"] == "summary"]
        boundRecords = [row for row in chromosomeRecords if row["record_type"] == "target_bound"]
        assert {row["target_role"] for row in summaryRecords} == {"selected", "descriptive"}
        assert {row["coverage_scope"] for row in summaryRecords} == {fitScope}
        selectedBound = next(row for row in boundRecords if row["target_role"] == "selected")
        descriptiveBound = next(row for row in boundRecords if row["target_role"] == "descriptive")
        assert selectedBound["bound_scope"] == boundScope
        assert (descriptiveBound["bound_available"], descriptiveBound.get("bound_scope")) == (False, None)


def _caseDeleteBlockCalibrationReportsRefitFailures(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=uncertainty.logger.name)
    caplog.clear()
    n = 24
    m = 2
    matrixData = np.zeros((m, n), dtype=np.float32)
    matrixMunc = np.full_like(matrixData, 0.1, dtype=np.float32)
    fullState = np.zeros((n, 2), dtype=np.float32)
    fullCovar = np.zeros((n, 2, 2), dtype=np.float32)
    fullCovar[:, 0, 0] = 0.1
    fullCovar[:, 1, 1] = 0.01

    def _failingRunConsenrich(*_args, **_kwargs):
        raise RuntimeError("planned refit failure")

    monkeypatch.setattr(core, "runConsenrich", _failingRunConsenrich)

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=100,
        calibrationECMIters=1,
        minHeldoutCells=1,
        maxHeldoutCells=12,
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        seed=31,
    )

    with pytest.raises(
        ValueError,
        match="delete-block state uncertainty calibration produced no valid deleted-state rows",
    ):
        uncertainty.calibrateChromosomeStateUncertainty(
            matrixData=matrixData,
            matrixMunc=matrixMunc,
            fullState=fullState,
            fullCovar=fullCovar,
            intervals=np.arange(n, dtype=np.int64) * 25,
            intervalSizeBP=25,
            params=params,
            runKwargs=_smallRunKwargs(),
        )

    assert "uncertaintyCalibration.deleteBlock.fold.failed" in caplog.text
    assert "planned refit failure" in caplog.text



def _caseCalibrateChromosomeStateUncertaintySingleReplicate(tmp_path):
    n = 36
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    matrixData = np.sin(grid).astype(np.float32)[None, :]
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    runKwargs = _smallRunKwargs()
    full = core.runConsenrich(matrixData, matrixMunc, **runKwargs)
    fullState, fullCovar, _resid, _track4, _blockMap = full

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=100,
        calibrationECMIters=1,
        minHeldoutCells=1000,
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        writeDiagnostics=True,
        seed=13,
    )

    result = uncertainty.calibrateChromosomeStateUncertainty(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        fullState=fullState,
        fullCovar=fullCovar,
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=runKwargs,
        outPrefix=str(tmp_path / "single"),
    )

    assert result.calibratedUncertainty.shape == (n,)
    assert np.all(np.isfinite(result.calibratedUncertainty))
    assert (tmp_path / "single.delete_block_calibration.jsonl").exists()
    assert not (tmp_path / "single.delete_block_calibration.log").exists()
    assert not (tmp_path / "single.diagnostics.tsv.gz").exists()
    assert not (tmp_path / "single.model.json").exists()


def test_uncertainty_cython_contracts(contract_case):
    for label, func in (
        ("feature matrix matches Python", _caseCythonFeatureMatrixMatchesPythonForFloat32AndFloat64),
        ("factor evaluation", _caseCythonFactorEvaluation),
        ("factor model strict contract", _caseSegShrinkFactorModelStrictContract),
        ("pac order examples", _casePacOrderIndexExamples),
        ("delete-block information", _caseDeleteBlockInformationApproximation),
        ("delete-block variance mode", _caseDeleteBlockVarianceModeSelection),
        ("replicate dependence Gaussian coverage", _caseReplicateDependenceGaussianCoverage),
        ("replicate dependence delete-block evidence", _caseReplicateDependenceDeleteBlockEvidence),
        ("delete-state block scores", _caseCythonDeletedStateScoresAndDeleteBlockScores),
        ("summary contracts", _caseCythonSummaryContracts),
        ("segShrink Cython parity", _caseSegShrinkCythonParityContract),
    ):
        contract_case(label, func)


def test_uncertainty_calibration_smoke_contract(tmp_path, monkeypatch, caplog, contract_case):
    contract_case(
        "uncertainty calibration routing",
        _caseUncertaintyCalibrationRouting,
    )
    contract_case(
        "calibration smoke",
        _caseCalibrateChromosomeStateUncertaintySmoke,
        tmp_path,
        caplog,
    )
    contract_case(
        "cheap Q warmup policy for calibration refits",
        _caseCalibrationRefitsUseCheapProcessNoiseWarmup,
        monkeypatch,
        caplog,
    )
    contract_case(
        "segShrink calibration",
        _caseSegShrinkCalibrationContract,
        monkeypatch,
    )
    contract_case(
        "sampled LPO auto rho refit source",
        _caseSampledLPOAutoRhoUsesDeleteBlockRefits,
        monkeypatch,
    )
    contract_case(
        "sampled LPO auto rho no deleted pairs",
        _caseSampledLPOAutoRhoNoDeletedPairs,
        monkeypatch,
    )
    contract_case(
        "sampled LPO auto rho bg subtraction",
        _caseSampledLPOAutoRhoSubtractsMaskedBg,
        monkeypatch,
    )
    contract_case(
        "sampled LPO auto rho information recompute",
        _caseSampledLPOAutoRhoRecomputesEffectiveInformation,
        monkeypatch,
    )
    contract_case(
        "calibration floor after factor paths",
        _caseCalibrationFloorAppliesToGlobalAndSegShrink,
        monkeypatch,
    )
    contract_case(
        "segShrink processed contigs",
        _caseSegShrinkProcessedContigContract,
        tmp_path,
        monkeypatch,
    )
    contract_case(
        "delete-block refit failure handling",
        _caseDeleteBlockCalibrationReportsRefitFailures,
        monkeypatch,
        caplog,
    )


def test_uncertainty_single_replicate_contract(tmp_path, contract_case):
    contract_case(
        "single-replicate calibration",
        _caseCalibrateChromosomeStateUncertaintySingleReplicate,
        tmp_path,
    )
