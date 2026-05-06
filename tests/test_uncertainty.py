# -*- coding: utf-8 -*-

import numpy as np
import json

import consenrich.cuncertainty as cuncertainty
import consenrich.core as core
import consenrich.uncertainty as uncertainty


def _smallRunKwargs():
    return dict(
        deltaF=0.15,
        minQ=1.0e-4,
        maxQ=1.0,
        offDiagQ=0.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=12,
        pad=1.0e-4,
        disableCalibration=False,
        EM_maxIters=2,
        EM_innerRtol=0.0,
        EM_tNu=8.0,
        EM_useObsPrecReweight=True,
        EM_useProcPrecReweight=True,
        EM_useReplicateBias=True,
        EM_outerIters=1,
        EM_outerRtol=0.0,
        returnScales=True,
        returnReplicateOffsets=True,
        applyJackknife=False,
    )


def testFitFactorModelAllowsInflationAndDeflation():
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


def testCythonFeatureMatrixMatchesPythonForFloat32AndFloat64():
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


def testCythonHeldoutExtractionAndFactorEvaluation():
    matrixData = np.arange(12, dtype=np.float32).reshape(3, 4)
    matrixMunc = np.full((3, 4), 0.2, dtype=np.float32)
    state = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    pState = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    bias = np.array([0.0, 0.1, -0.2], dtype=np.float32)
    mask = np.ones((3, 4), dtype=np.uint8)
    mask[0, 1] = 0
    mask[2, 3] = 0

    residual, pHeld, rHeld, ii, jj, fold = cuncertainty.cextractHeldoutScores(
        matrixData,
        matrixMunc,
        state,
        pState,
        bias,
        mask,
        2,
        1.0e-4,
        float(core.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
    )

    assert ii.tolist() == [1, 3]
    assert jj.tolist() == [0, 2]
    assert fold.tolist() == [2, 2]
    assert np.allclose(residual, [matrixData[0, 1] - state[1], matrixData[2, 3] - state[3] - bias[2]])
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


def testCythonObjectiveAndSummaryContracts():
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


def testCalibrateChromosomeStateUncertaintySmoke(tmp_path):
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
    fullState, fullCovar, _resid, _track4, _qScale, replicateBias, _blockMap = full

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=120,
        heldoutReplicateFraction=1.0 / m,
        calibrationEMIters=1,
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
        pad=1.0e-4,
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
    assert {"factor_min", "factor_median", "factor_max"}.isdisjoint(model)
    assert model["heldout_cells"] >= model["fit_heldout_cells"]
    assert model["fit_heldout_cells"] <= 12
    assert model["diagnostic_score_rows"] <= 5
    modelKeys = np.atleast_1d(diagnostics["key"])[
        np.atleast_1d(diagnostics["record_type"]) == "model"
    ]
    assert {"factor_min", "factor_median", "factor_max"}.isdisjoint(modelKeys)
    assert not (tmp_path / "cal.summary.tsv").exists()
    assert not (tmp_path / "cal.scores.tsv.gz").exists()


def testCalibrateChromosomeStateUncertaintySingleReplicate(tmp_path):
    n = 36
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    matrixData = np.sin(grid).astype(np.float32)[None, :]
    matrixMunc = np.full_like(matrixData, 0.08, dtype=np.float32)
    runKwargs = _smallRunKwargs()
    full = core.runConsenrich(matrixData, matrixMunc, **runKwargs)
    fullState, fullCovar, _resid, _track4, _qScale, replicateBias, _blockMap = full

    params = core.uncertaintyCalibrationParams(
        enabled=True,
        folds=2,
        blockSizeBP=100,
        calibrationEMIters=1,
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
        pad=1.0e-4,
        outPrefix=str(tmp_path / "single"),
    )

    assert result.calibratedUncertainty.shape == (n,)
    assert np.all(np.isfinite(result.calibratedUncertainty))
    assert (tmp_path / "single.diagnostics.tsv.gz").exists()
    assert (tmp_path / "single.model.json").exists()
