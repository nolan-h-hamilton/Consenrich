# -*- coding: utf-8 -*-

import logging
import numpy as np
import json
import pytest

import consenrich.cuncertainty as cuncertainty
import consenrich.core as core
import consenrich.diagnostics as diagnostic_utils
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
        returnReplicateOffsets=True,
    )


def _caseFitFactorModelAllowsInflationAndDeflation():
    n = 20
    pState = np.ones(n, dtype=np.float64)
    obsVar = np.full(n, 0.1, dtype=np.float64)
    features = np.ones((n, 1), dtype=np.float64)
    intervalIndex = np.arange(n, dtype=np.int64)
    params = core.uncertaintyCalibrationParams(
        targets=(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[0],),
        factorMin=core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
        factorMax=core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
        ridge=0.0,
        minHeldoutCells=1,
    )

    smallResidual = np.zeros(n, dtype=np.float64)
    betaSmall, _ = uncertainty._fitFactorModel(
        residual=smallResidual,
        pState=pState,
        obsVar=obsVar,
        featureByInterval=features,
        intervalIndex=intervalIndex,
        params=params,
    )

    largeResidual = np.full(n, 5.0, dtype=np.float64)
    betaLarge, _ = uncertainty._fitFactorModel(
        residual=largeResidual,
        pState=pState,
        obsVar=obsVar,
        featureByInterval=features,
        intervalIndex=intervalIndex,
        params=params,
    )

    assert float(np.exp(betaSmall[0])) < 1.0
    assert float(np.exp(betaLarge[0])) > 1.0


def _caseObservationVarianceFloorSelectorSolvesTrimmedTarget():
    n = 64
    pState = np.full(n, 0.2, dtype=np.float64)
    muncBase = np.zeros(n, dtype=np.float64)
    targetR = 0.4
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    residual = np.sqrt(pState + targetR + pad)

    selected, score, hitUpper = uncertainty._selectObservationVarianceFloor(
        residual=residual,
        pState=pState,
        muncBase=muncBase,
        lambdaExp=None,
        pad=pad,
        lower=0.0,
        upper=2.0,
    )

    assert abs(selected - targetR) < 1.0e-6
    assert abs(score - 1.0) < 1.0e-6
    assert not hitUpper

    lambdaExp = np.full(n, 2.0, dtype=np.float64)
    residualWithLambda = np.sqrt(pState + (targetR + pad) / lambdaExp)
    selectedWithLambda, scoreWithLambda, hitUpperWithLambda = (
        uncertainty._selectObservationVarianceFloor(
            residual=residualWithLambda,
            pState=pState,
            muncBase=muncBase,
            lambdaExp=lambdaExp,
            pad=pad,
            lower=0.0,
            upper=2.0,
        )
    )

    assert abs(selectedWithLambda - targetR) < 1.0e-6
    assert abs(scoreWithLambda - 1.0) < 1.0e-6
    assert not hitUpperWithLambda

    selectedAlreadyCalibrated, scoreAlreadyCalibrated, _ = (
        uncertainty._selectObservationVarianceFloor(
            residual=0.5 * np.sqrt(pState + pad),
            pState=pState,
            muncBase=muncBase,
            lambdaExp=None,
            pad=pad,
            lower=0.0,
            upper=2.0,
        )
    )

    assert selectedAlreadyCalibrated == 0.0
    assert scoreAlreadyCalibrated < 1.0


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


def _caseCythonHeldoutExtractionAndFactorEvaluation():
    matrixData = np.arange(12, dtype=np.float32).reshape(3, 4)
    matrixMunc = np.full((3, 4), 0.2, dtype=np.float32)
    state = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    pState = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    bias = np.array([0.0, 0.1, -0.2], dtype=np.float32)
    background = np.array([0.0, 0.2, -0.1, 0.3], dtype=np.float32)
    mask = np.ones((3, 4), dtype=np.uint8)
    mask[0, 1] = 0
    mask[2, 3] = 0

    residual, pHeld, rHeld, ii, jj, fold = cuncertainty.cextractHeldoutScores(
        matrixData,
        matrixMunc,
        state,
        pState,
        bias,
        background,
        mask,
        2,
        1.0e-4,
        float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )

    assert ii.tolist() == [1, 3]
    assert jj.tolist() == [0, 2]
    assert fold.tolist() == [2, 2]
    assert np.allclose(
        residual,
        [
            matrixData[0, 1] - background[1] - state[1],
            matrixData[2, 3] - background[3] - state[3] - bias[2],
        ],
    )
    assert np.allclose(pHeld, [pState[1], pState[3]])
    assert np.allclose(rHeld, [0.2001, 0.2001])

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


def _caseCythonTargetBlockScores():
    residual = np.array([1.0, 4.0, 3.0, 100.0, np.nan, 6.0], dtype=np.float64)
    pState = np.ones(residual.size, dtype=np.float64)
    obsVar = np.zeros(residual.size, dtype=np.float64)
    factorByInterval = np.array([1.0, 4.0, 1.0, 1.0], dtype=np.float64)
    intervalIndex = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    blockIndex = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    targetMask = np.array([1, 0, 1], dtype=np.uint8)

    blocks, scores, counts = cuncertainty.ctargetBlockScores(
        residual,
        pState,
        obsVar,
        factorByInterval,
        intervalIndex,
        blockIndex,
        targetMask,
        1.0,
        float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )

    assert blocks.tolist() == [0, 2]
    assert np.allclose(scores, [2.0, 3.0])
    assert counts.tolist() == [2, 1]


def _caseCythonObjectiveAndSummaryContracts():
    residual = np.array([-1.0, -0.2, 0.1, 1.3], dtype=np.float64)
    pState = np.full(4, 0.5, dtype=np.float64)
    obsVar = np.full(4, 0.2, dtype=np.float64)
    features = np.ones((4, 1), dtype=np.float64)
    intervalIndex = np.arange(4, dtype=np.int64)
    targets = np.array(core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS[:2], dtype=np.float64)
    targetZ = np.array([uncertainty._normalZ(t) for t in targets], dtype=np.float64)
    theta = np.array([0.0, 0.0], dtype=np.float64)

    value = cuncertainty.cfactorObjective(
        theta,
        residual,
        pState,
        obsVar,
        features,
        intervalIndex,
        targets,
        targetZ,
        core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
        core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
        0.0,
        core.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH,
        1.0,
        float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN),
        float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX),
        float(core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR),
        float(core.UNCERTAINTY_CALIBRATION_WIS_WEIGHT),
        float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )
    assert np.isfinite(value)
    valueWithGrad, gradient = cuncertainty.cfactorObjectiveAndGradient(
        theta,
        residual,
        pState,
        obsVar,
        features,
        intervalIndex,
        targets,
        targetZ,
        core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
        core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
        0.0,
        core.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH,
        1.0,
        float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN),
        float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX),
        float(core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR),
        float(core.UNCERTAINTY_CALIBRATION_WIS_WEIGHT),
        float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )
    assert np.isclose(valueWithGrad, value)
    assert gradient.shape == theta.shape
    eps = 1.0e-6
    for k in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[k] += eps
        minus[k] -= eps
        valuePlus = cuncertainty.cfactorObjective(
            plus,
            residual,
            pState,
            obsVar,
            features,
            intervalIndex,
            targets,
            targetZ,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
            0.0,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH,
            1.0,
            float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN),
            float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX),
            float(core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR),
            float(core.UNCERTAINTY_CALIBRATION_WIS_WEIGHT),
            float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        valueMinus = cuncertainty.cfactorObjective(
            minus,
            residual,
            pState,
            obsVar,
            features,
            intervalIndex,
            targets,
            targetZ,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
            0.0,
            core.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH,
            1.0,
            float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN),
            float(core.UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX),
            float(core.UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR),
            float(core.UNCERTAINTY_CALIBRATION_WIS_WEIGHT),
            float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        assert np.isclose(
            gradient[k],
            (valuePlus - valueMinus) / (2.0 * eps),
            rtol=1.0e-4,
            atol=1.0e-5,
        )

    summary = cuncertainty.csummarizeCoverageWidths(
        residual,
        np.sqrt(pState + obsVar),
        np.sqrt(2.0 * pState + obsVar),
        np.array([0, 0, 1, 1], dtype=np.int32),
        targets,
        targetZ,
        float(core.UNCERTAINTY_CALIBRATION_SUMMARY_MEDIAN_QUANTILE),
        float(core.UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE),
    )
    assert set(summary) >= {"group", "target", "coverage_before", "mean_width_after"}
    assert summary["group"].tolist() == [-1, 0, 1, -1, 0, 1]


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
    fullState, fullCovar, _resid, _track4, replicateBias, _blockMap = full

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=120,
        heldoutReplicateFraction=1.0 / m,
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
        fullReplicateBias=replicateBias,
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
    diagnosticsPath = tmp_path / "cal.diagnostics.tsv.gz"
    assert diagnosticsPath.exists()
    diagnostics = np.genfromtxt(
        diagnosticsPath,
        delimiter="\t",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert {"score", "summary", "model"} <= set(np.atleast_1d(diagnostics["record_type"]))
    assert np.sum(np.atleast_1d(diagnostics["record_type"]) == "score") <= 5
    modelPath = tmp_path / "cal.model.json"
    assert modelPath.exists()
    with open(modelPath, "r", encoding="utf-8") as handle:
        model = json.load(handle)
    assert "objective" in model
    assert model["target_calibration"]["enabled"] is True
    assert model["target_calibration"]["delta"] == params.targetCalibrationDelta
    assert model["target_calibration"]["score_definition"] == (
        "max_abs_residual_over_predictive_sd_by_block"
    )
    assert len(model["target_calibration"]["bounds"]) == len(params.targets)
    assert model["target_calibration"]["scale_uncertainty_by_target_calibration"] is True
    assert model["target_calibration"]["uncertainty_track_scaled"] is True
    assert {"factor_min", "factor_median", "factor_max"}.isdisjoint(model)
    assert model["state_roughness"]["block_len_intervals"] == (
        diagnostic_utils.resolveUncertaintyBlockSizeIntervals(
            params.blockSizeBP,
            25,
            n,
        )
    )
    assert model["state_roughness"]["overall_mean_abs_diff"] is not None
    assert model["heldout_cells"] >= model["fit_heldout_cells"]
    assert model["fit_heldout_cells"] <= 12
    assert model["diagnostic_score_rows"] <= 5
    coverageRows = model["state_uncertainty_coverage"]
    coverageFitRows = model["state_uncertainty_coverage_fit"]
    assert any(row["stratum"] == "overall" for row in coverageRows)
    assert any(str(row["stratum"]).startswith("signal_abs_q") for row in coverageRows)
    overallRows = [row for row in coverageRows if row["stratum"] == "overall"]
    overallFitRows = [row for row in coverageFitRows if row["stratum"] == "overall"]
    assert {row["target"] for row in overallRows} == set(params.targets)
    assert {row["target"] for row in overallFitRows} == set(params.targets)
    assert all(row["n"] == model["heldout_cells"] for row in overallRows)
    assert all(row["n"] == model["fit_heldout_cells"] for row in overallFitRows)
    assert all("coverage_before" in row and "coverage_after" in row for row in coverageRows)
    modelKeys = np.atleast_1d(diagnostics["key"])[
        np.atleast_1d(diagnostics["record_type"]) == "model"
    ]
    assert {"factor_min", "factor_median", "factor_max"}.isdisjoint(modelKeys)
    assert not (tmp_path / "cal.summary.tsv").exists()
    assert not (tmp_path / "cal.scores.tsv.gz").exists()
    assert "uncertaintyCalibration.target enabled=True" in caplog.text
    assert "blocksTargetScored=" in caplog.text
    assert "uncertaintyCalibration.coverage.crossfit_all" in caplog.text
    assert "uncertaintyCalibration.coverage.fit_sample" in caplog.text


def _caseObservationVarianceFloorHeldoutSmoke(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=uncertainty.logger.name)
    caplog.clear()
    m, n = 4, 16
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    targetR = 0.4
    pState = np.full(n, 0.2, dtype=np.float32)
    lambdaExp = np.full(n, 2.0, dtype=np.float32)
    background = np.linspace(-0.1, 0.1, n, dtype=np.float32)
    residual = np.sqrt(pState + (targetR + pad) / lambdaExp).astype(np.float32)
    matrixData = np.tile(background + residual, (m, 1)).astype(np.float32)
    matrixMunc = np.zeros((m, n), dtype=np.float32)
    calls = []

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        calls.append((observationMask, kwargs))
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        postFitResiduals = np.zeros((n, m), dtype=np.float32)
        nis = np.zeros(n, dtype=np.float32)
        bias = np.zeros(m, dtype=np.float32)
        blockMap = np.arange(n, dtype=np.int32)
        precisionDiagnostics = {
            "precision_track_diagnostics": True,
            "lambdaExp": lambdaExp,
        }
        return (
            state,
            covar,
            postFitResiduals,
            nis,
            bias,
            blockMap,
            background,
            precisionDiagnostics,
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=8,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=17,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = True
    commonBackground = background.copy()
    runKwargs["initialBackground"] = commonBackground

    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=runKwargs,
        maxR=2.0,
        fallbackMinR=1.0e-4,
        chromosome="chrTest",
    )

    assert calls
    assert abs(result.minR - targetR) < 1.0e-5
    assert abs(result.trimmedMean - 1.0) < 1.0e-5
    assert result.usedLambda
    assert result.heldoutCells == 2 * n
    assert result.diagnostics["background_mode"] == "fixed_common"
    assert result.diagnostics["fixed_common_background"] is True
    assert result.diagnostics["common_background_source"] == "provided_initialBackground"
    assert result.diagnostics["holdout_replicates"] == 2
    assert uncertainty._resolveObservationFloorHoldoutCount(2) == 1
    assert uncertainty._resolveObservationFloorHoldoutCount(3) == 2
    assert uncertainty._resolveObservationFloorHoldoutCount(4) == 2
    assert uncertainty._resolveObservationFloorHoldoutCount(10) == 5
    assert runKwargs["fitBackground"] is True
    assert runKwargs["initialBackground"] is commonBackground
    for _mask, kwargs in calls:
        assert kwargs.get("fitBackground") is False
        assert kwargs.get("returnBackground") is True
        assert np.allclose(kwargs.get("initialBackground"), commonBackground)
        assert kwargs.get("initialBackground") is not commonBackground
    for mask, _kwargs in calls:
        heldoutByInterval = np.sum(np.asarray(mask) == 0, axis=0)
        assert set(np.unique(heldoutByInterval)).issubset({0, 2})
    combinedHeldoutByInterval = np.sum(
        [np.sum(np.asarray(mask) == 0, axis=0) for mask, _kwargs in calls],
        axis=0,
    )
    assert np.all(combinedHeldoutByInterval == 2)
    assert "observationFloorCalibration.start chrom=chrTest" in caplog.text
    assert "observationFloorCalibration.fold.done chrom=chrTest" in caplog.text
    assert "observationFloorCalibration.candidates chrom=chrTest" in caplog.text
    assert "observationFloorCalibration.done chrom=chrTest" in caplog.text
    assert "selectedSource=heldout_lambda" in caplog.text
    assert "lambdaFreeGuardApplied=False" in caplog.text
    assert "contrastFloorApplied=False" in caplog.text


def _caseObservationVarianceFloorMissingBackgroundModes(monkeypatch):
    m, n = 2, 12
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    targetR = 0.3
    pState = np.full(n, 0.15, dtype=np.float32)
    residual = np.sqrt(pState + targetR + pad).astype(np.float32)
    matrixData = np.tile(residual, (m, 1)).astype(np.float32)
    matrixMunc = np.zeros((m, n), dtype=np.float32)
    calls = []

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        calls.append((observationMask, kwargs))
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=6,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=19,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = True
    with pytest.raises(ValueError, match="fixed initialBackground"):
        uncertainty.estimateObservationVarianceFloorFromHeldout(
            matrixData=matrixData,
            matrixMunc=matrixMunc,
            intervalSizeBP=1,
            params=params,
            runKwargs=runKwargs,
            maxR=2.0,
            fallbackMinR=1.0e-4,
            chromosome="chrMissing",
        )
    assert not calls
    assert runKwargs["fitBackground"] is True
    assert "initialBackground" not in runKwargs

    zeroRunKwargs = _smallRunKwargs()
    zeroRunKwargs["fitBackground"] = False
    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=zeroRunKwargs,
        maxR=2.0,
        fallbackMinR=1.0e-4,
        chromosome="chrZero",
    )
    assert abs(result.minR - targetR) < 1.0e-5
    assert abs(result.trimmedMean - 1.0) < 1.0e-5
    assert result.diagnostics["background_mode"] == "fixed_zero"
    assert result.diagnostics["fixed_common_background"] is False
    assert result.diagnostics["common_background_source"] == "zero_fitBackground_false"
    assert zeroRunKwargs["fitBackground"] is False
    assert "initialBackground" not in zeroRunKwargs
    assert calls
    for _mask, kwargs in calls:
        assert kwargs.get("fitBackground") is False
        assert kwargs.get("returnBackground") is True
        assert kwargs.get("initialBackground") is None


def _caseObservationVarianceFloorSafetyCushion(monkeypatch):
    m, n = 2, 12
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    pState = np.full(n, 0.2, dtype=np.float32)
    residual = np.sqrt(pState + pad).astype(np.float32)
    matrixData = np.tile(residual, (m, 1)).astype(np.float32)
    matrixMunc = np.zeros((m, n), dtype=np.float32)

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=6,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=23,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = False
    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=runKwargs,
        maxR=1.0,
        fallbackMinR=0.0,
        chromosome="chrCushion",
    )

    cushion = uncertainty._OBSERVATION_VARIANCE_FLOOR_SAFETY_CUSHION
    expectedScore = float((pState[0] + pad) / (pState[0] + pad + cushion))
    assert result.minR == cushion
    assert abs(result.trimmedMean - expectedScore) < 1.0e-6
    assert result.diagnostics["raw_selected_min_r"] < cushion
    assert result.diagnostics["safety_cushion_min_r"] == cushion
    assert result.diagnostics["safety_cushion_applied"] is True


def _caseObservationVarianceFloorFallbackGuard(monkeypatch):
    m, n = 2, 12
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    fallbackMinR = 0.05
    pState = np.full(n, 0.2, dtype=np.float32)
    residual = np.sqrt(pState + pad).astype(np.float32)
    matrixData = np.tile(residual, (m, 1)).astype(np.float32)
    matrixMunc = np.zeros((m, n), dtype=np.float32)

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=6,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=24,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = False
    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=runKwargs,
        maxR=1.0,
        fallbackMinR=fallbackMinR,
        chromosome="chrFallbackGuard",
    )

    expectedScore = float((pState[0] + pad) / (pState[0] + pad + fallbackMinR))
    assert result.minR == fallbackMinR
    assert abs(result.trimmedMean - expectedScore) < 1.0e-6
    assert not result.fallbackUsed
    assert result.diagnostics["raw_selected_min_r"] < fallbackMinR
    assert result.diagnostics["selection_floor_guard_min_r"] == fallbackMinR
    assert result.diagnostics["selection_floor_guard_applied"] is True
    assert result.diagnostics["selected_source"] == "fallback_guard"


def _caseObservationVarianceFloorResidualVarianceBeatsMuncPad(monkeypatch):
    m, n = 3, 15
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    fallbackMinR = 1.0e-4
    targetR = 0.35
    pState = np.full(n, 0.11, dtype=np.float32)
    background = np.linspace(0.25, -0.15, n, dtype=np.float32)
    matrixMunc = np.tile(np.linspace(0.04, 0.09, n, dtype=np.float32), (m, 1))
    residual = np.sqrt(pState + targetR + pad).astype(np.float32)
    matrixData = np.tile(background + residual, (m, 1)).astype(np.float32)
    calls = []

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        calls.append((np.asarray(observationMask, dtype=np.uint8).copy(), kwargs))
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            background,
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=5,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=29,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = True
    runKwargs["initialBackground"] = background.copy()

    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=runKwargs,
        maxR=2.0,
        fallbackMinR=fallbackMinR,
        chromosome="chrMuncPad",
    )

    assert abs(result.minR - targetR) < 1.0e-5
    assert result.minR > 1000.0 * fallbackMinR
    assert abs(result.trimmedMean - 1.0) < 1.0e-5
    assert result.heldoutCells == 2 * n
    assert result.fitCells == result.heldoutCells
    assert result.diagnostics["holdout_replicates"] == 2
    assert result.diagnostics["background_mode"] == "fixed_common"
    assert result.diagnostics["max_fixed_background_deviation"] == 0.0
    assert not result.fallbackUsed
    assert calls


def _caseObservationVarianceFloorSparseHeldoutIntervals(monkeypatch):
    m, n = 4, 18
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    targetR = 0.28
    pState = np.full(n, 0.07, dtype=np.float32)
    matrixMunc = np.tile(np.linspace(0.03, 0.12, n, dtype=np.float32), (m, 1))
    lambdaExp = np.linspace(0.8, 2.4, n, dtype=np.float32)
    residual = np.sqrt(pState + (targetR + pad) / lambdaExp)
    matrixData = np.tile(residual.astype(np.float32), (m, 1))
    keepIntervals = np.array([2, 9, 15], dtype=np.int64)
    excludeIntervals = np.ones(n, dtype=bool)
    excludeIntervals[keepIntervals] = False

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
            {
                "precision_track_diagnostics": True,
                "lambdaExp": lambdaExp,
            },
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=6,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=31,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = False

    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=runKwargs,
        maxR=2.0,
        fallbackMinR=1.0e-4,
        excludeIntervals=excludeIntervals,
        chromosome="chrSparse",
    )

    assert abs(result.minR - targetR) < 1.0e-5
    assert abs(result.trimmedMean - 1.0) < 1.0e-5
    assert result.usedLambda
    assert result.heldoutCells == 2 * keepIntervals.size
    assert result.fitCells == result.heldoutCells
    assert result.diagnostics["holdout_replicates"] == 2
    assert result.diagnostics["background_mode"] == "fixed_zero"


def _caseObservationVarianceFloorLambdaFreeGuard(monkeypatch):
    m, n = 2, 14
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    targetR = 0.3
    pState = np.full(n, 0.1, dtype=np.float32)
    lambdaExp = np.full(n, 0.1, dtype=np.float32)
    residual = np.sqrt(pState + targetR + pad).astype(np.float32)
    matrixData = np.tile(residual, (m, 1)).astype(np.float32)
    matrixMunc = np.zeros((m, n), dtype=np.float32)

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
            {
                "precision_track_diagnostics": True,
                "lambdaExp": lambdaExp,
            },
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=7,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=32,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = False

    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=runKwargs,
        maxR=1.0,
        fallbackMinR=1.0e-4,
        chromosome="chrLambdaGuard",
    )

    assert result.minR == pytest.approx(targetR, abs=1.0e-5)
    assert result.diagnostics["innovation_selected_min_r"] < 0.05
    assert result.diagnostics["innovation_no_lambda_selected_min_r"] == pytest.approx(
        targetR,
        abs=1.0e-5,
    )
    assert result.diagnostics["lambda_free_guard_applied"] is True
    assert result.diagnostics["selected_source"] == "heldout_lambda_free"
    assert result.diagnostics["median_lambda"] == pytest.approx(0.1)
    assert result.diagnostics["fraction_lambda_lt_0p5"] == 1.0


def _caseObservationVarianceFloorReplicateContrastGuard(monkeypatch):
    m, n = 4, 10
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    targetR = 0.24
    amplitude = np.float32(np.sqrt(0.5 * (targetR + pad)))
    pState = np.full(n, 1.5, dtype=np.float32)
    matrixData = np.zeros((m, n), dtype=np.float32)
    matrixData[0, :] = amplitude
    matrixData[1, :] = -amplitude
    matrixMunc = np.zeros((m, n), dtype=np.float32)

    masks = []
    mask0 = np.ones((m, n), dtype=np.uint8)
    mask0[0:2, :] = 0
    masks.append(mask0)
    masks.append(np.ones((m, n), dtype=np.uint8))

    def fakeMakeFoldMasks(**kwargs):
        return [mask.copy() for mask in masks]

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(uncertainty, "_makeFoldMasks", fakeMakeFoldMasks)
        scoped.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
        params = core.uncertaintyCalibrationParams(
            folds=2,
            blockSizeBP=5,
            calibrationECMIters=1,
            maxScores=10_000,
            writeDiagnostics=False,
            seed=33,
        )
        runKwargs = _smallRunKwargs()
        runKwargs["fitBackground"] = False

        result = uncertainty.estimateObservationVarianceFloorFromHeldout(
            matrixData=matrixData,
            matrixMunc=matrixMunc,
            intervalSizeBP=1,
            params=params,
            runKwargs=runKwargs,
            maxR=1.0,
            fallbackMinR=1.0e-4,
            chromosome="chrContrastGuard",
        )

        assert result.minR == pytest.approx(targetR, abs=1.0e-5)
        assert result.trimmedMean < 1.0
        assert result.heldoutCells == 2 * n
        assert result.diagnostics["contrast_cells"] == n
        assert result.diagnostics["innovation_selected_min_r"] == 0.0
        assert result.diagnostics["contrast_selected_min_r"] == pytest.approx(
            targetR,
            abs=1.0e-5,
        )
        assert result.diagnostics["contrast_floor_applied"] is True
        assert result.diagnostics["selected_source"] == "replicate_contrast"


def _caseObservationVarianceFloorNoHeldoutScoresFallbackDiagnostics(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=uncertainty.logger.name)
    caplog.clear()
    m, n = 2, 10
    pad = core.UNCERTAINTY_CALIBRATION_DEFAULT_PAD
    fallbackMinR = 3.0e-4
    pState = np.full(n, 0.1, dtype=np.float32)
    matrixData = np.zeros((m, n), dtype=np.float32)
    matrixMunc = np.zeros((m, n), dtype=np.float32)

    def fakeRunConsenrich(matrixDataArg, matrixMuncArg, observationMask=None, **kwargs):
        state = np.zeros((n, 2), dtype=np.float32)
        covar = np.zeros((n, 2, 2), dtype=np.float32)
        covar[:, 0, 0] = pState
        return (
            state,
            covar,
            np.zeros((n, m), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.arange(n, dtype=np.int32),
            np.zeros(n, dtype=np.float32),
        )

    monkeypatch.setattr(uncertainty.core, "runConsenrich", fakeRunConsenrich)
    params = core.uncertaintyCalibrationParams(
        folds=2,
        blockSizeBP=5,
        calibrationECMIters=1,
        maxScores=10_000,
        writeDiagnostics=False,
        seed=37,
    )
    runKwargs = _smallRunKwargs()
    runKwargs["fitBackground"] = False

    result = uncertainty.estimateObservationVarianceFloorFromHeldout(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        intervalSizeBP=1,
        params=params,
        runKwargs=runKwargs,
        maxR=1.0,
        fallbackMinR=fallbackMinR,
        excludeIntervals=np.ones(n, dtype=bool),
        chromosome="chrFallback",
    )

    cushion = uncertainty._OBSERVATION_VARIANCE_FLOOR_SAFETY_CUSHION
    assert result.minR == cushion
    assert result.heldoutCells == 0
    assert result.fitCells == 0
    assert not result.usedLambda
    assert result.fallbackUsed
    assert result.diagnostics["reason"] == "no_heldout_scores"
    assert result.diagnostics["fallback_min_r"] == cushion
    assert result.diagnostics["safety_cushion_min_r"] == cushion
    assert result.diagnostics["safety_cushion_applied"] is False
    assert result.diagnostics["background_mode"] == "fixed_zero"
    assert result.diagnostics["common_background_source"] == "zero_fitBackground_false"
    assert result.diagnostics["fixed_common_background"] is False
    assert "observationFloorCalibration.done chrom=chrFallback" in caplog.text
    assert "selectedSource=fallback_no_heldout" in caplog.text
    assert "heldoutCells=0 fitCells=0 contrastCells=0" in caplog.text
    assert "fallbackUsed=True" in caplog.text


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
    replicateBias = np.zeros(m, dtype=np.float32)
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
            replicateBias,
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
        heldoutReplicateFraction=1.0 / m,
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
        fullReplicateBias=replicateBias,
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=runKwargs,
    )

    assert len(capturedKwargs) == params.folds
    assert len(capturedMasks) == params.folds
    assert all(kwargs.get("fitBackground") is True for kwargs in capturedKwargs)
    assert all(kwargs["ECM_outerIters"] == 1 for kwargs in capturedKwargs)
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
    for mask in capturedMasks:
        heldoutByInterval = np.sum(mask == 0, axis=0)
        assert set(np.unique(heldoutByInterval)).issubset({0, 1})
    combinedHeldoutByInterval = np.sum(
        [np.sum(mask == 0, axis=0) for mask in capturedMasks],
        axis=0,
    )
    assert np.all(combinedHeldoutByInterval == 1)



def _caseCalibrateChromosomeStateUncertaintySingleReplicate(tmp_path):
    n = 36
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    matrixData = np.sin(grid).astype(np.float32)[None, :]
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    runKwargs = _smallRunKwargs()
    full = core.runConsenrich(matrixData, matrixMunc, **runKwargs)
    fullState, fullCovar, _resid, _track4, replicateBias, _blockMap = full

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
        fullReplicateBias=replicateBias,
        intervals=np.arange(n, dtype=np.int64) * 25,
        intervalSizeBP=25,
        params=params,
        runKwargs=runKwargs,
        outPrefix=str(tmp_path / "single"),
    )

    assert result.calibratedUncertainty.shape == (n,)
    assert np.all(np.isfinite(result.calibratedUncertainty))
    assert (tmp_path / "single.diagnostics.tsv.gz").exists()
    assert (tmp_path / "single.model.json").exists()


def test_uncertainty_factor_model_contract(contract_case):
    contract_case(
        "factor model allows inflation and deflation",
        _caseFitFactorModelAllowsInflationAndDeflation,
    )
    contract_case(
        "observation variance floor selector solves trimmed target",
        _caseObservationVarianceFloorSelectorSolvesTrimmedTarget,
    )
    contract_case("PAC order index examples", _casePacOrderIndexExamples)


def test_uncertainty_cython_contracts(contract_case):
    for label, func in (
        ("feature matrix matches Python", _caseCythonFeatureMatrixMatchesPythonForFloat32AndFloat64),
        ("heldout extraction and factor evaluation", _caseCythonHeldoutExtractionAndFactorEvaluation),
        ("target block scores", _caseCythonTargetBlockScores),
        ("objective and summary contracts", _caseCythonObjectiveAndSummaryContracts),
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
        "held-out observation variance floor smoke",
        _caseObservationVarianceFloorHeldoutSmoke,
        monkeypatch,
        caplog,
    )
    contract_case(
        "observation variance floor missing background modes",
        _caseObservationVarianceFloorMissingBackgroundModes,
        monkeypatch,
    )
    contract_case(
        "observation variance floor safety cushion",
        _caseObservationVarianceFloorSafetyCushion,
        monkeypatch,
    )
    contract_case(
        "observation variance floor fallback guard",
        _caseObservationVarianceFloorFallbackGuard,
        monkeypatch,
    )
    contract_case(
        "observation variance floor uses residual variance above MUNC plus pad",
        _caseObservationVarianceFloorResidualVarianceBeatsMuncPad,
        monkeypatch,
    )
    contract_case(
        "observation variance floor sparse heldout intervals",
        _caseObservationVarianceFloorSparseHeldoutIntervals,
        monkeypatch,
    )
    contract_case(
        "observation variance floor lambda-free guard",
        _caseObservationVarianceFloorLambdaFreeGuard,
        monkeypatch,
    )
    contract_case(
        "observation variance floor replicate contrast guard",
        _caseObservationVarianceFloorReplicateContrastGuard,
        monkeypatch,
    )
    contract_case(
        "observation variance floor no-score fallback diagnostics",
        _caseObservationVarianceFloorNoHeldoutScoresFallbackDiagnostics,
        monkeypatch,
        caplog,
    )
    contract_case(
        "cheap Q warmup policy for calibration refits",
        _caseCalibrationRefitsUseCheapProcessNoiseWarmup,
        monkeypatch,
    )


def test_uncertainty_single_replicate_contract(tmp_path, contract_case):
    contract_case(
        "single-replicate calibration",
        _caseCalibrateChromosomeStateUncertaintySingleReplicate,
        tmp_path,
    )
