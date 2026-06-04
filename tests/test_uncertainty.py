# -*- coding: utf-8 -*-

import logging
import numpy as np
import json
import pandas as pd
import pytest

import consenrich.cuncertainty as cuncertainty
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
    assert bounds[0]["certified"] is False
    assert bounds[0]["q"] == 57.0
    assert bounds[0]["q_source"] == "empirical_max_uncertified"


def _caseDeleteBlockInformationApproximation():
    m = 3
    n = 7
    blockLen = 2
    folds = 3
    holdoutCount = 1
    seed = 19
    blockCount = (n + blockLen - 1) // blockLen
    seededBlockFold, seededRepsByBlock = cuncertainty.cmakeFoldSpec(
        m,
        n,
        blockLen,
        folds,
        holdoutCount,
        seed,
    )
    rng = np.random.default_rng(seed)
    blockOrder = rng.permutation(blockCount).astype(np.int32, copy=False)
    wantBlockFold = np.empty(blockCount, dtype=np.int32)
    wantBlockFold[blockOrder] = np.arange(blockCount, dtype=np.int32) % folds
    wantRepsByBlock = np.empty((blockCount, holdoutCount), dtype=np.intp)
    for block in range(blockCount):
        wantRepsByBlock[block, :] = rng.choice(m, size=holdoutCount, replace=False)
    assert np.array_equal(seededBlockFold, wantBlockFold)
    assert np.array_equal(seededRepsByBlock, wantRepsByBlock)

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
    repsByBlock = np.array([[0], [0], [1]], dtype=np.intp)
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
            [1, 1, 0],
        ],
    )

    assert np.allclose(totalInfo, [4.0, 4.0, 8.0])
    assert np.allclose(keptInfo, [4.0, 2.0, 4.0])
    assert np.allclose(deletedInfo, [0.0, 2.0, 4.0])
    assert np.allclose(h, [0.0, 0.5, 0.5])

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

    assert valid.tolist() == [False, True, True]
    assert np.isnan(delta[0])
    assert np.allclose(delta[1:], [1.0, 2.0])
    assert uncertainty.DELETE_BLOCK_VARIANCE_SOURCE_LABELS[source].tolist() == [
        "invalid",
        "heldout_information",
        "heldout_information",
    ]
    assert uncertainty.DELETE_BLOCK_INVALID_REASON_LABELS[reason].tolist() == [
        "h_out_of_bounds",
        "valid",
        "valid",
    ]


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
            "q_source": "pac_order_statistic",
            "certified": True,
        }
    )

    assert info["scaled"] is True
    assert info["certified"] is True
    assert info["target_z"] == pytest.approx(z)
    assert info["scale"] == pytest.approx(2.0)
    assert info["reason"] == "scaled_by_certified_target_bound_q_over_z"


def _caseOldPredictiveHeldoutModeUnsupported():
    for oldMode in ("predictive_holdout", "predictive-heldout", "heldout_residual"):
        with pytest.raises(ValueError, match="predictive held-out residual.*removed"):
            uncertainty._normalizeUncertaintyCalibrationMode(oldMode)


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
    ) == 2_000
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
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=_smallRunKwargs(),
        outPrefix=str(tmp_path / "cal"),
    )

    assert result.factor.shape == (n,)
    assert result.calibratedUncertainty.shape == (n,)
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
    assert model["mode"] == "delete_block_state"
    assert model["score_definition"] == "deleted_state_delta_over_deleted_state_delta_sd"
    assert model["factor_model"] == "segShrink"
    assert model["factorModel"] == "segShrink"
    assert (
        model["segmentCount"]
        == core.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_SEGMENT_COUNT
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
        "max_abs_deleted_state_delta_over_deleted_state_delta_sd_by_block"
    )
    assert len(model["target_calibration"]["bounds"]) == len(params.targets)
    assert isinstance(
        model["target_calibration"]["scale_uncertainty_by_target_calibration"],
        bool,
    )
    if model["target_calibration"]["uncertainty_track_scaled"]:
        assert model["target_calibration"]["uncertainty_track_scale"] == pytest.approx(
            model["target_calibration"]["uncertainty_track_scale_q"]
            / model["target_calibration"]["uncertainty_track_scale_target_z"]
        )
    assert {"factor_min", "factor_median", "factor_max"}.isdisjoint(model)
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
    assert model["fold_refits"]["holdout_count"] >= 1
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
    assert "replicate" not in result.scores.columns
    assert "observation_variance" not in result.scores.columns
    assert "deleted_state_delta" in result.scores.columns
    assert "delta_variance" in result.scores.columns
    assert "delta_variance_source" in result.scores.columns
    modelRecord = next(
        record for record in diagnostics if record["record_type"] == "model"
    )
    assert {"factor_min", "factor_median", "factor_max"}.isdisjoint(modelRecord)
    assert not (tmp_path / "cal.summary.tsv").exists()
    assert not (tmp_path / "cal.scores.tsv.gz").exists()
    assert "uncertaintyCalibration.target enabled=True" in caplog.text
    assert "blocksTargetScored=" in caplog.text
    assert "mode=delete_block_state" in caplog.text
    assert "deleteBlockRows=" in caplog.text
    assert "uncertaintyCalibration.coverage.delete_block_all" in caplog.text
    assert "uncertaintyCalibration.coverage.fit_sample" in caplog.text











def _caseCalibrationRefitsUseCheapProcessNoiseWarmup(monkeypatch):
    n = 32
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signal = np.sin(grid).astype(np.float32)
    matrixData = np.vstack(
        [
            signal - 0.02,
            signal + 0.01,
            signal + 0.03,
        ]
    ).astype(np.float32)
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
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
        minHeldoutCells=1,
        maxHeldoutCells=24,
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        seed=21,
    )

    uncertainty.calibrateChromosomeStateUncertainty(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        fullState=fullState,
        fullCovar=fullCovar,
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=runKwargs,
    )

    assert len(capturedKwargs) == params.folds
    assert len(capturedMasks) == params.folds
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
    holdoutCount = uncertainty._resolveHoldoutCount(m, params.holdoutFraction)
    for mask in capturedMasks:
        deletedByInterval = np.sum(mask == 0, axis=0)
        assert set(np.unique(deletedByInterval)).issubset({0, holdoutCount})
    combinedDeletedByInterval = np.sum(
        [np.sum(mask == 0, axis=0) for mask in capturedMasks],
        axis=0,
    )
    assert np.all(combinedDeletedByInterval == holdoutCount)


def _caseSegShrinkCalibrationContract(monkeypatch):
    n = 40
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signal = np.sin(grid).astype(np.float32)
    matrixData = np.vstack(
        [
            signal - 0.02,
            signal + 0.01,
            signal + 0.03,
        ]
    ).astype(np.float32)
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    fullState = np.column_stack(
        [signal, np.gradient(signal).astype(np.float32)]
    ).astype(np.float32)
    fullCovar = np.zeros((n, 2, 2), dtype=np.float32)
    fullCovar[:, 0, 0] = 0.05
    fullCovar[:, 1, 1] = 0.01
    capturedKwargs = []

    def _fakeRunConsenrich(matrixDataArg, _matrixMuncArg, *, observationMask, **kwargs):
        capturedKwargs.append(dict(kwargs))
        deleted = np.mean(np.asarray(observationMask, dtype=np.float32) == 0, axis=0)
        maskedState = fullState.copy()
        maskedState[:, 0] = maskedState[:, 0] + 0.05 * deleted
        maskedCovar = fullCovar.copy()
        maskedCovar[:, 0, 0] = maskedCovar[:, 0, 0] + 0.04 + 0.01 * deleted
        residual = np.asarray(matrixDataArg, dtype=np.float32) - maskedState[:, 0][None, :]
        return (
            maskedState,
            maskedCovar,
            residual.T,
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    monkeypatch.setattr(core, "runConsenrich", _fakeRunConsenrich)

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=100,
        calibrationECMIters=1,
        calibrationOuterIters=9,
        minHeldoutCells=1,
        maxHeldoutCells=40,
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        deleteBlockVarianceMode="covariance_difference",
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

    model = result.model
    assert result.factor.shape == (n,)
    assert np.all(np.isfinite(result.factor))
    assert np.all(result.factor > 0.0)
    assert model["factor_model"] == "segShrink"
    assert model["factorModel"] == "segShrink"
    assert model["hierarchyScope"] == "singleProcessedContig"
    assert model["processedContigCount"] == 1
    assert model["segmentCount"] == 4
    assert model["bootstrapReplicates"] == 8
    assert model["blockIDXUnitCount"] >= 1
    assert set(model["refitPolicy"]) >= {
        "ECM_outerIters",
        "ECM_minOuterIters",
        "ECM_fixedBackgroundIters",
        "processNoiseWarmupECMIters",
    }
    assert model["refitPolicy"]["ECM_outerIters"] == 4
    assert model["refitPolicy"]["ECM_minOuterIters"] == 1
    assert model["refitPolicy"]["ECM_fixedBackgroundIters"] == 2
    assert len(model["segmentShrinkage"]) == 4
    assert "blockIDX" in result.scores.columns
    assert "factor_segment" in result.scores.columns
    assert "segment_shrinkage_weight" in result.scores.columns
    assert all(kwargs["ECM_outerIters"] == 4 for kwargs in capturedKwargs)
    assert all(kwargs["ECM_fixedBackgroundIters"] == 2 for kwargs in capturedKwargs)
    assert all(kwargs["ECM_minOuterIters"] == 1 for kwargs in capturedKwargs)
    overall = [
        row for row in model["state_uncertainty_coverage_fit"]
        if row["stratum"] == "overall"
    ]
    assert overall and all("coverage_after" in row for row in overall)


def _caseSegShrinkProcessedContigContract():
    with pytest.raises(ValueError, match="no processed contigs"):
        segshrink.combinePreparedContigs(
            [],
            positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )

    prepared = []
    for chromosome, rawFactor, variance in (
        ("chrA", 1.0, 0.25),
        ("chrC", 4.0, 0.5),
    ):
        prepared.append(
            {
                "chromosome": chromosome,
                "fullP": np.ones(6, dtype=np.float64),
                "model": {
                    "global_factor": rawFactor,
                    "contigShrinkage": [
                        {
                            "rawFactor": rawFactor,
                            "bootstrapVariance": variance,
                        }
                    ],
                    "segmentShrinkage": [
                        {
                            "segment": 0,
                            "rows": 3,
                            "rawFactor": rawFactor,
                            "bootstrapVariance": variance,
                            "shrinkageWeight": 1.0,
                            "factor": rawFactor,
                            "fallbackReason": "none",
                        },
                        {
                            "segment": 1,
                            "rows": 3,
                            "rawFactor": rawFactor * 1.5,
                            "bootstrapVariance": variance,
                            "shrinkageWeight": 1.0,
                            "factor": rawFactor * 1.5,
                            "fallbackReason": "none",
                        },
                    ],
                },
            }
        )

    finalized = segshrink.combinePreparedContigs(
        prepared,
        positiveFloor=float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )
    assert [item["chromosome"] for item in finalized] == ["chrA", "chrC"]
    expectedGenomeLog = (
        np.log(1.0) / 0.25 + np.log(4.0) / 0.5
    ) / (1.0 / 0.25 + 1.0 / 0.5)
    for item in finalized:
        model = item["model"]
        assert model["hierarchyScope"] == "processedGenome"
        assert model["processedContigCount"] == 2
        assert model["genomeFactor"] == pytest.approx(float(np.exp(expectedGenomeLog)))
        assert {row["chromosome"] for row in model["contigShrinkage"]} == {
            "chrA",
            "chrC",
        }
        assert item["calibrated"].shape == (6,)
        assert np.all(np.isfinite(item["calibrated"]))


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


def test_uncertainty_factor_model_contract(contract_case):
    contract_case(
        "delete-block global factor uses weighted quantile",
        _caseDeleteBlockGlobalFactorUsesWeightedQuantile,
    )
    contract_case(
        "segShrink factor model is strict camelCase",
        _caseSegShrinkFactorModelStrictContract,
    )
    contract_case("PAC order index examples", _casePacOrderIndexExamples)
    contract_case(
        "delete-block information approximation",
        _caseDeleteBlockInformationApproximation,
    )
    contract_case(
        "delete-block variance mode selection",
        _caseDeleteBlockVarianceModeSelection,
    )
    contract_case(
        "target calibration q over z scaling",
        _caseTargetCalibrationTrackScaleUsesQOverZ,
    )
    contract_case(
        "old predictive held-out calibration unsupported",
        _caseOldPredictiveHeldoutModeUnsupported,
    )
    contract_case(
        "auto block sizing for short contigs",
        _caseAutoBlockSizeForShortContigs,
    )


def test_uncertainty_cython_contracts(contract_case):
    for label, func in (
        ("feature matrix matches Python", _caseCythonFeatureMatrixMatchesPythonForFloat32AndFloat64),
        ("factor evaluation", _caseCythonFactorEvaluation),
        ("delete-state block scores", _caseCythonDeletedStateScoresAndDeleteBlockScores),
        ("summary contracts", _caseCythonSummaryContracts),
        ("segShrink Cython parity", _caseSegShrinkCythonParityContract),
    ):
        contract_case(label, func)


def test_uncertainty_calibration_smoke_contract(tmp_path, monkeypatch, caplog, contract_case):
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
    )
    contract_case(
        "segShrink calibration",
        _caseSegShrinkCalibrationContract,
        monkeypatch,
    )
    contract_case(
        "segShrink processed contigs",
        _caseSegShrinkProcessedContigContract,
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
