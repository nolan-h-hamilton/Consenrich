# -*- coding: utf-8 -*-

# the name of this file is unfortunately misleading --
# 'core' as in 'central' or 'basic', not the literal
# `consenrich.core` module

import logging
import math
import os
import tempfile
from types import SimpleNamespace
from typing import Tuple, List, Optional
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
import scipy.ndimage as ndi
import scipy.stats as stats
import scipy.signal as spySig  # renamed to avoid conflict with any `signal` variables

import consenrich.core as core
import consenrich.constants as constants
import consenrich.consenrich as consenrichRuntime
import consenrich.ccounts as ccounts
import consenrich.cconsenrich as cconsenrich
import consenrich.diagnostics as diagnostics
import consenrich.detrorm as detrorm
import consenrich.misc_util as misc_util
import consenrich.peaks as peaks

TESTS_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TESTS_DIR / "data"
FRAGMENTS_DIR = TEST_DATA_DIR / "fragments"


def test_process_precision_auto_min_rejects_too_small_max():
    with pytest.raises(ValueError, match="auto .*convexity-preserving lower bound"):
        core._checkProcessPrecisionMultiplierBounds(
            minValue=-1.0,
            maxValue=0.5,
            robustTNu=8.0,
            stateDim=2,
        )


def test_transform_count_variance_floor_is_transform_dependent_and_applied():
    counts = np.array([0.0, 1.0, 4.0], dtype=np.float32)
    log1Floor = core.transformCountVarianceFloor(
        counts,
        [1.0],
        transformMethod="log",
        logOffset=1.0,
        logMult=1.0,
        transformInputOffset=1.0,
        transformInputScale=1.0,
        transformOutputScale=1.0,
        transformShape=1.0,
    )
    log2Floor = core.transformCountVarianceFloor(
        counts,
        [1.0],
        transformMethod="log",
        logOffset=2.0,
        logMult=1.0,
        transformInputOffset=2.0,
        transformInputScale=1.0,
        transformOutputScale=1.0,
        transformShape=1.0,
    )

    assert np.all(np.isfinite(log1Floor))
    assert log1Floor[0] > 0.0
    assert log2Floor[0] < log1Floor[0]
    expectedLog1 = (counts.astype(np.float64) + 0.5) / np.square(
        counts.astype(np.float64) + 1.5
    )
    np.testing.assert_allclose(log1Floor, expectedLog1.astype(np.float32))

    scaledCounts = np.asarray([0.5, 2.0, 8.0], dtype=np.float32)
    rawNoiseMass = np.asarray([0.125, 1.5, 3.0], dtype=np.float32)
    pseudoVarianceMass = 0.125
    rawDefaultFloor = core.transformCountVarianceFloor(
        scaledCounts,
        [0.25],
        rawNoiseMass=rawNoiseMass,
        transformMethod="identity",
        transformInputScale=1.0,
        transformOutputScale=1.0,
    )
    rawPseudoFloor = core.transformCountVarianceFloor(
        scaledCounts,
        [0.25],
        rawNoiseMass=rawNoiseMass,
        countNoisePseudoVarianceMass=pseudoVarianceMass,
        transformMethod="identity",
        transformInputScale=1.0,
        transformOutputScale=1.0,
    )
    np.testing.assert_allclose(
        rawDefaultFloor,
        ((rawNoiseMass + 0.5) * 0.25 * 0.25).astype(np.float32),
    )
    np.testing.assert_allclose(
        rawPseudoFloor,
        ((rawNoiseMass + pseudoVarianceMass) * 0.25 * 0.25).astype(np.float32),
    )

    munc = np.full_like(log1Floor, 0.01, dtype=np.float32)
    floored = core.applyMuncCountModelVarianceFloor(
        munc,
        log1Floor,
        varianceFloor=1.0e-6,
    )
    assert np.all(floored >= log1Floor)
    assert floored[0] == pytest.approx(log1Floor[0])


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


_REMOVED_EM_PREFIX = "E" + "M" + "_"


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


def _monoLogReference(x: np.ndarray, offset: float, scale: float) -> np.ndarray:
    out = np.empty_like(x)
    if x.dtype == np.float32:
        offset_ = np.float32(offset if offset > 0.0 else 1.0)
        scale_ = np.float32(scale)
        for i, value in enumerate(x):
            u = np.float32(value + offset_)
            if u <= np.float32(0.0):
                u = offset_
            out[i] = np.float32(scale_ * np.float32(math.log(float(u))))
    else:
        offset_ = float(offset if offset > 0.0 else 1.0)
        scale_ = float(scale)
        for i, value in enumerate(x):
            u = float(value) + offset_
            if u <= 0.0:
                u = offset_
            out[i] = scale_ * math.log(u)
    return out


def _logRatioReference(
    treatment: np.ndarray,
    control: np.ndarray,
    offset: float,
    scale: float,
) -> np.ndarray:
    out = np.empty_like(treatment)
    if treatment.dtype == np.float32:
        offset_ = np.float32(offset if offset > 0.0 else 1.0)
        scale_ = np.float32(scale)
        for i, (treatValue, controlValue) in enumerate(zip(treatment, control)):
            t = np.float32(treatValue + offset_)
            c = np.float32(controlValue + offset_)
            if t <= np.float32(0.0):
                t = offset_
            if c <= np.float32(0.0):
                c = offset_
            out[i] = np.float32(
                scale_ * np.float32(math.log(float(t)) - math.log(float(c)))
            )
    else:
        offset_ = float(offset if offset > 0.0 else 1.0)
        scale_ = float(scale)
        for i, (treatValue, controlValue) in enumerate(zip(treatment, control)):
            t = float(treatValue) + offset_
            c = float(controlValue) + offset_
            if t <= 0.0:
                t = offset_
            if c <= 0.0:
                c = offset_
            out[i] = scale_ * (math.log(t) - math.log(c))
    return out


def _countTransformReference(
    x: np.ndarray,
    *,
    mode: str,
    inputOffset: float = 0.0,
    inputScale: float = 1.0,
    outputScale: float = 1.0,
    outputOffset: float = 0.0,
    shape: float = 1.0,
) -> np.ndarray:
    z = x.astype(np.float64) + float(inputOffset)
    inputScale_ = float(inputScale)
    outputScale_ = float(outputScale)
    outputOffset_ = float(outputOffset)
    shape_ = float(shape)
    key = mode.replace("-", "").replace("_", "").lower()
    if key == "log":
        z = np.where(z <= 0.0, float(inputOffset), z)
        u = z / inputScale_
        u = np.where(u <= 0.0, 1.0, u)
        core_ = np.log(u)
    elif key == "sqrt":
        core_ = np.sqrt(np.maximum(z / inputScale_, 0.0))
    elif key == "anscombe":
        core_ = np.sqrt(np.maximum(z / inputScale_, 0.0))
    elif key == "asinh":
        core_ = np.arcsinh(z / inputScale_)
    elif key == "asinhsqrt":
        core_ = np.arcsinh(np.sqrt(np.maximum(z, 0.0)) / inputScale_)
    elif key == "generalizedlog":
        u = z / inputScale_
        core_ = np.log((u + np.sqrt((u * u) + (shape_ * shape_))) / shape_)
    elif key == "identity":
        core_ = z / inputScale_
    else:
        raise AssertionError(f"unexpected reference mode {mode!r}")
    return (outputOffset_ + outputScale_ * core_).astype(x.dtype)


def _countTransformDiffReference(
    treatment: np.ndarray,
    control: np.ndarray,
    **kwargs,
) -> np.ndarray:
    noOutputOffset = dict(kwargs)
    noOutputOffset["outputOffset"] = 0.0
    return (
        _countTransformReference(treatment, **noOutputOffset)
        - _countTransformReference(control, **noOutputOffset)
    ).astype(treatment.dtype)


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


def _expectedCSF(chromMat: np.ndarray, centerMedian: bool = True) -> np.ndarray:
    chromMat_ = np.ascontiguousarray(chromMat, dtype=np.float32)
    logMat = np.log(chromMat_.astype(np.float64))
    refLog = logMat.mean(axis=0)
    scaleFactors = np.exp(np.median(logMat - refLog, axis=1))
    scaleFactors = np.clip(scaleFactors, 0.2, 5.0)

    if centerMedian:
        centerLog = np.median(np.log(scaleFactors + 1.0e-8))
        scaleFactors = np.clip(scaleFactors / np.exp(centerLog), 0.2, 5.0)

    return 1.0 / scaleFactors


@pytest.mark.correctness
def test_runtime_munc_dense_seed_builds_pooled_prior(tmp_path, monkeypatch):
    intervalCount = 20
    intervalSizeBP = 1_000_000
    sampleCount = 2
    chromSize = intervalCount * intervalSizeBP
    chromSizesPath = tmp_path / "chrom.sizes"
    chromSizesPath.write_text(f"chrTest\t{chromSize}\n", encoding="utf-8")
    configPath = tmp_path / "config.yaml"
    configPath.write_text("experimentName: runtimeMuncDenseSeed\n", encoding="utf-8")
    baseMatrix = np.vstack(
        (
            np.arange(intervalCount, dtype=np.float32),
            np.arange(intervalCount, dtype=np.float32) + np.float32(0.25),
        )
    )

    class StopAfterMuncPrior(RuntimeError):
        pass

    def makeConfig(
        ebUse,
        *,
        seedWeightEnabled=constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_ENABLED,
        seedWeightStudentT=constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_T,
    ):
        sources = [
            core.inputSource(
                path=f"sample{sampleIndex}.bedGraph",
                sourceKind=core.BEDGRAPH_SOURCE_KIND,
                role="treatment",
            )
            for sampleIndex in range(sampleCount)
        ]
        return {
            "experimentName": "runtimeMuncDenseSeed",
            "genomeArgs": core.genomeParams(
                genomeName="testGenome",
                chromSizesFile=str(chromSizesPath),
                blacklistFile=None,
                sparseBedFile=None,
                genomeCovariateCacheDir=None,
                chromosomes=["chrTest"],
                excludeChroms=[],
                excludeForNorm=[],
            ),
            "inputArgs": core.inputParams(
                bamFiles=[source.path for source in sources],
                bamFilesControl=None,
                treatmentSources=sources,
                controlSources=[],
            ),
            "outputArgs": core.outputParams(
                convertToBigWig=False,
                roundDigits=4,
                writeUncertainty=False,
                writeRunSummary=False,
            ),
            "countingArgs": core.countingParams(
                intervalSizeBP=intervalSizeBP,
                backgroundBlockSizeBP=intervalSizeBP,
                scaleFactors=[1.0] * sampleCount,
                scaleFactorsControl=None,
                normMethod="CPM",
                fragmentsGroupNorm=None,
                fixControl=False,
                logOffset=1.0,
                logMult=1.0,
                transformMethod="identity",
                centerMB=False,
            ),
            "scArgs": core.scParams(),
            "processArgs": core.processParams(
                stateModel=core.STATE_MODEL_LEVEL,
                deltaF=1.0,
                minQ=1.0e-4,
                maxQ=1.0,
            ),
            "observationArgs": core.observationParams(
                minR=None,
                maxR=10.0,
                samplingIters=4,
                EB_use=ebUse,
                EB_setNu0=None,
                EB_setNuL=8,
                trendNumBasis=4,
                trendMinObsPerBasis=1.0,
                trendMinEdf=1.0,
                trendMaxEdf=4.0,
                trendLambdaMin=1.0e-4,
                trendLambdaMax=1.0e4,
                trendLambdaGridSize=5,
                numNearest=None,
                sparseSupportScaleBP=None,
                sparseSupportPrior=None,
                pad=1.0e-4,
                precisionMultiplierMin=0.5,
                precisionMultiplierMax=2.0,
                useCountNoiseFloor=False,
                muncTrendBlockSizeBP=5 * intervalSizeBP,
                muncLocalWindowSizeBP=4 * intervalSizeBP,
                muncTrendBlockDependenceMultiplier=1.0,
                muncEBPriorStrata=None,
                muncEBPriorSupportMinQ=0.0,
                muncEBPriorSupportMaxQ=1.0,
                muncEBPriorMaxExtrapolatedFraction=0.0,
                muncEBPriorGUncertaintyMode="disabled",
                muncCovariatesEnabled=False,
                muncCovariatesFeatures=(),
                muncSeedWeightEnabled=seedWeightEnabled,
                muncSeedWeightPasses=2,
                muncSeedWeightStudentT=seedWeightStudentT,
                muncSeedProcessMinQ=2.0e-5,
                muncSeedProcessMaxQ=7.0e-4,
            ),
            "stateArgs": core.stateParams(
                stateInit=0.0,
                stateCovarInit=1.0,
                boundState=False,
                stateLowerBound=0.0,
                stateUpperBound=0.0,
            ),
            "uncertaintyCalibrationArgs": core.uncertaintyCalibrationParams(
                enabled=False,
            ),
            "samArgs": core.samParams(
                samThreads=1,
                samFlagExclude=0,
                oneReadPerBin=0,
                chunkSize=1,
                inferFragmentLength=0,
            ),
            "matchingArgs": core.matchingParams(
                enabled=False,
                randSeed=1,
                numBootstrap=0,
                thresholdZ=3.0,
                dependenceSpan=None,
                gamma=0.25,
                selectionPenalty=None,
                gammaScale=1.0,
                nestedRoccoIters=0,
                nestedRoccoBudgetScale=1.0,
                exportFilterUncertaintyMultiplier=1.0,
            ),
            "fitArgs": core.fitParams(
                ECM_fixedBackgroundIters=1,
                ECM_outerIters=1,
                ECM_minOuterIters=1,
                ECM_backgroundLengthScaleMultiplier=1.0,
                fitBackground=False,
                useNonnegativeBackground=False,
                backgroundNegativePenaltyMultiplier=None,
            ),
        }

    monkeypatch.setattr(
        consenrichRuntime.sys,
        "argv",
        ["consenrich", "--config", str(configPath)],
    )
    monkeypatch.setattr(
        consenrichRuntime,
        "_configureCliLogging",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        consenrichRuntime,
        "_initializeDiagnosticLogs",
        lambda paths: None,
    )
    monkeypatch.setattr(consenrichRuntime.os, "listdir", lambda path: [])
    monkeypatch.setattr(
        consenrichRuntime.misc_util,
        "getChromSizesDict",
        lambda *args, **kwargs: {"chrTest": chromSize},
    )
    monkeypatch.setattr(
        core,
        "getChromRangesJoint",
        lambda *args, **kwargs: (0, chromSize),
    )
    monkeypatch.setattr(
        core,
        "readSegments",
        lambda *args, **kwargs: np.ascontiguousarray(baseMatrix, dtype=np.float32),
    )
    monkeypatch.setattr(cconsenrich, "cTransformInPlace", lambda *args, **kwargs: None)

    def failTrendFit(blockMeans, blockVariances, sampleIndex, **kwargs):
        pooledFitCallsByFlag[activeFlag] = {
            "blockMeans": np.asarray(blockMeans, dtype=np.float64).copy(),
            "blockVariances": np.asarray(blockVariances, dtype=np.float64).copy(),
            "sampleIndex": np.asarray(sampleIndex, dtype=np.int64).copy(),
            "weights": np.asarray(kwargs["weights"], dtype=np.float64).copy(),
        }
        raise StopAfterMuncPrior()

    monkeypatch.setattr(core, "fitPooledMuncVarianceTrend", failTrendFit)

    seedQCallsByFlag = {}
    seedMomentCallsByFlag = {}
    pooledFitCallsByFlag = {}
    activeSeedUseWeightsByFlag = {}
    seedProcessQ = np.diag([2.0e-4, 3.0e-4]).astype(np.float32)

    def fakeSeedProcessQ(matrixData, matrixMunc, **kwargs):
        data = np.asarray(matrixData, dtype=np.float32)
        seedQCallsByFlag.setdefault(activeFlag, []).append(
            (
                int(round(float(data[0, 0]))),
                int(data.shape[1]),
                kwargs["processNoiseCalibration"],
                float(kwargs["minQ"]),
                float(kwargs["maxQ"]),
            )
        )
        return seedProcessQ, {
            "qSeedSource": "test",
            "qSeedReason": "ok",
            "qSeedLevelFinal": float(seedProcessQ[0, 0]),
            "qSeedTrendFinal": float(seedProcessQ[1, 1]),
        }

    monkeypatch.setattr(
        core,
        "_estimateInitialProcessNoiseFromData",
        fakeSeedProcessQ,
    )

    def fakeMuncObservationMomentSeedPass(
        matrixData,
        matrixMunc,
        stateMean,
        stateVariance,
        **kwargs,
    ):
        data = np.asarray(matrixData, dtype=np.float32)
        munc = np.asarray(matrixMunc, dtype=np.float32)
        state = np.asarray(stateMean, dtype=np.float32)
        stateVar = np.asarray(stateVariance, dtype=np.float32)
        assert data.shape == (sampleCount, intervalCount)
        assert munc.shape == data.shape
        assert state.shape == (intervalCount,)
        assert stateVar.shape == (intervalCount,)
        passCalls = seedMomentCallsByFlag.setdefault(activeFlag, [])
        passIndex = len(passCalls)
        updateWeights = bool(kwargs["updateWeights"])
        expectedUseWeights = activeSeedUseWeightsByFlag[activeFlag]
        omegaIn = np.asarray(kwargs["omegaIn"], dtype=np.float32)
        rhoIn = np.asarray(kwargs["rhoIn"], dtype=np.float32)
        if passIndex == 0:
            assert updateWeights is True
            np.testing.assert_allclose(omegaIn, np.ones(intervalCount, dtype=np.float32))
            np.testing.assert_allclose(rhoIn, np.ones(data.shape, dtype=np.float32))
        elif passIndex < 2:
            assert updateWeights is True
            if expectedUseWeights:
                np.testing.assert_allclose(omegaIn, passCalls[-1]["omegaOut"])
                np.testing.assert_allclose(rhoIn, passCalls[-1]["rhoOut"])
            else:
                np.testing.assert_allclose(
                    omegaIn,
                    np.ones(intervalCount, dtype=np.float32),
                )
                np.testing.assert_allclose(rhoIn, np.ones(data.shape, dtype=np.float32))
        else:
            assert updateWeights is False
            if expectedUseWeights:
                np.testing.assert_allclose(omegaIn, passCalls[-1]["omegaOut"])
                np.testing.assert_allclose(rhoIn, passCalls[-1]["rhoOut"])
            else:
                np.testing.assert_allclose(
                    omegaIn,
                    np.ones(intervalCount, dtype=np.float32),
                )
                np.testing.assert_allclose(rhoIn, np.ones(data.shape, dtype=np.float32))
        assert kwargs["useSeedWeights"] is expectedUseWeights
        assert kwargs["studentTdf"] == pytest.approx(
            constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_TDF
        )

        background = np.asarray(kwargs["background"], dtype=np.float32).reshape(1, -1)
        gVariance = np.asarray(kwargs["gVariance"], dtype=np.float32).reshape(1, -1)
        moment = (
            np.square(data.astype(np.float64) - background - state.reshape(1, -1))
            + stateVar.reshape(1, -1)
            + gVariance
        ).astype(np.float32)
        local = np.maximum(moment - np.float32(kwargs["pad"]), 1.0e-6).astype(
            np.float32
        )
        seedMunc = local.copy()
        omegaOut = np.full(
            intervalCount,
            0.50 + 0.10 * min(passIndex, 2),
            dtype=np.float32,
        )
        rhoOut = np.full(data.shape, 1.20 + 0.10 * min(passIndex, 2), dtype=np.float32)
        rhoPattern = np.tile(
            np.asarray(
                [
                    4.0,
                    0.25,
                    0.25,
                    0.25,
                    0.25,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            intervalCount // 10,
        )
        rhoOut *= rhoPattern.reshape(1, -1)
        passCalls.append(
            {
                "updateWeights": updateWeights,
                "useSeedWeights": kwargs["useSeedWeights"],
                "omegaOut": omegaOut.copy(),
                "rhoOut": rhoOut.copy(),
            }
        )
        return moment, rhoOut, omegaOut.copy(), omegaOut, local, seedMunc

    monkeypatch.setattr(
        cconsenrich,
        "cMuncObservationMomentSeedPass",
        fakeMuncObservationMomentSeedPass,
        raising=False,
    )

    def fakeMuncSmoothDenseLocalEvidence(
        localEvidence,
        windowIntervals,
        *,
        excludeMask=None,
        eps=1.0e-12,
    ):
        evidence = np.asarray(localEvidence, dtype=np.float32)
        window = max(1, int(windowIntervals))
        mask = (
            np.zeros(evidence.shape[1], dtype=np.uint8)
            if excludeMask is None
            else np.asarray(excludeMask, dtype=np.uint8).reshape(-1)
        )
        out = np.empty_like(evidence)
        half = window // 2
        for rowIndex in range(evidence.shape[0]):
            for intervalIndex in range(evidence.shape[1]):
                left = max(0, intervalIndex - half)
                right = left + window
                if right > evidence.shape[1]:
                    right = evidence.shape[1]
                    left = max(0, right - window)
                allowed = mask[left:right] == 0
                if np.any(allowed):
                    value = float(np.mean(evidence[rowIndex, left:right][allowed]))
                else:
                    value = float(evidence[rowIndex, intervalIndex])
                out[rowIndex, intervalIndex] = max(value, float(eps))
        return np.ascontiguousarray(out, dtype=np.float32)

    monkeypatch.setattr(
        cconsenrich,
        "cMuncSmoothDenseLocalEvidence",
        fakeMuncSmoothDenseLocalEvidence,
        raising=False,
    )

    runtimeCases = (
        ("ebOff", False, True, True),
        ("ebOn", True, True, True),
        ("studentTOff", True, True, False),
        ("seedWeightsOff", True, False, True),
    )
    for activeFlag, ebUse, seedWeightEnabled, seedWeightStudentT in runtimeCases:
        activeSeedUseWeightsByFlag[activeFlag] = bool(
            seedWeightEnabled and seedWeightStudentT
        )
        monkeypatch.setattr(
            consenrichRuntime,
            "readConfig",
            lambda path,
            flag=ebUse,
            enabled=seedWeightEnabled,
            studentT=seedWeightStudentT: makeConfig(
                flag,
                seedWeightEnabled=enabled,
                seedWeightStudentT=studentT,
            ),
        )
        with pytest.raises(StopAfterMuncPrior):
            consenrichRuntime.main()

    for activeFlag, _ebUse, _seedWeightEnabled, _seedWeightStudentT in runtimeCases:
        assert seedQCallsByFlag[activeFlag] == [(0, 20, "seed", 2.0e-5, 7.0e-4)]
        assert [call["updateWeights"] for call in seedMomentCallsByFlag[activeFlag]] == [
            True,
            True,
            False,
        ]
        fitCall = pooledFitCallsByFlag[activeFlag]
        expectedBlockCount = intervalCount // 5
        assert fitCall["blockMeans"].shape == (sampleCount * expectedBlockCount,)
        assert fitCall["blockVariances"].shape == fitCall["blockMeans"].shape
        np.testing.assert_array_equal(
            np.bincount(fitCall["sampleIndex"], minlength=sampleCount),
            np.full(sampleCount, expectedBlockCount, dtype=np.int64),
        )
        expectedNuEff = (
            np.asarray([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
            if activeSeedUseWeightsByFlag[activeFlag]
            else np.full(expectedBlockCount, 5.0, dtype=np.float64)
        )
        expectedWeights = np.tile(
            1.0 / core.trigamma(expectedNuEff / 2.0),
            sampleCount,
        )
        np.testing.assert_allclose(fitCall["weights"], expectedWeights, rtol=1.0e-6)
    assert all(
        call["useSeedWeights"] is True for call in seedMomentCallsByFlag["ebOff"]
    )
    assert all(call["useSeedWeights"] is True for call in seedMomentCallsByFlag["ebOn"])
    assert all(
        call["useSeedWeights"] is False
        for call in seedMomentCallsByFlag["studentTOff"]
    )
    assert all(
        call["useSeedWeights"] is False
        for call in seedMomentCallsByFlag["seedWeightsOff"]
    )


def test_munc_eb_prior_g_uncertainty_modes_are_limited():
    assert core._normalizeMuncEBPriorGUncertaintyMode(None) == "proxy"
    assert core._normalizeMuncEBPriorGUncertaintyMode("proxy") == "proxy"
    assert core._normalizeMuncEBPriorGUncertaintyMode("diagonal") == "proxy"
    assert core._normalizeMuncEBPriorGUncertaintyMode("disabled") == "disabled"

    with pytest.raises(ValueError, match="g uncertainty"):
        core._normalizeMuncEBPriorGUncertaintyMode("exact")


@pytest.mark.parametrize(
    "qKwargs, expectedMessage",
    (
        ({"qPriorTrend": 5.0e-4}, "qPriorTrend"),
        ({"qPriorLevel": 2.0e-3}, "qPriorLevel"),
    ),
)
def test_core_process_prior_q_respects_q_bounds(qKwargs, expectedMessage):
    matrixData = np.zeros((2, 4), dtype=np.float32)
    matrixMunc = np.ones((2, 4), dtype=np.float32)
    kwargs = {
        "deltaF": 0.1,
        "minQ": 1.0e-3,
        "maxQ": 1.0e-3,
        "qPriorLevel": 1.0e-3,
        "qPriorTrend": 1.0e-3,
        "stateInit": 0.0,
        "stateCovarInit": 1.0,
        "boundState": False,
        "stateLowerBound": 0.0,
        "stateUpperBound": 0.0,
        "blockLenIntervals": 2,
        "ECM_fixedBackgroundIters": 1,
    }
    kwargs.update(qKwargs)

    with pytest.raises(ValueError, match=expectedMessage):
        core.runConsenrich(matrixData, matrixMunc, **kwargs)


def test_run_consenrich_passes_t_inner_iters_to_fixed_background_ecm(monkeypatch):
    capturedInnerIters = []

    def fakeCFixedBackgroundECM(**kwargs):
        capturedInnerIters.append(kwargs["t_innerIters"])
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
            None,
            None,
            {"converged": True},
        )

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", fakeCFixedBackgroundECM)

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
        "ECM_useObsPrecisionReweighting": False,
        "ECM_useProcessPrecisionReweighting": False,
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

    core.runConsenrich(
        np.zeros((2, 5), dtype=np.float32),
        np.ones((2, 5), dtype=np.float32),
        **runKwargs,
        t_innerIters=4,
    )

    assert capturedInnerIters == [4]


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
def _casePuncBridgeSymbolsAreRenamed():
    for symbol in (
        "cPuncObservationInformation",
        "cpuncObservationInformation",
        "crebasePuncIntervalScales",
    ):
        assert hasattr(cconsenrich, symbol)
    for symbol in (
        "c" + "T" + "uncObservationInformation",
        "c" + "t" + "uncObservationInformation",
        "crebase" + "T" + "uncIntervalScales",
    ):
        assert not hasattr(cconsenrich, symbol)


@pytest.mark.correctness
def _casePuncObservationInformationKernelUsesLambdaClampForFloat32AndFloat64():
    matrix = np.asarray(
        [[0.10, 0.20, 0.40], [0.30, 0.50, 0.70]],
        dtype=np.float64,
    )
    lambdaExp = np.asarray([0.01, 2.0, 20.0], dtype=np.float64)
    lambdaClipped = np.asarray([0.5, 2.0, 4.0], dtype=np.float64)
    expected = np.sum(lambdaClipped[None, :] / (matrix + 0.05), axis=0)

    for dtype in (np.float32, np.float64):
        out = cconsenrich.cPuncObservationInformation(
            matrix.astype(dtype),
            0.05,
            lambdaExp,
            0.5,
            4.0,
        )
        np.testing.assert_allclose(out, expected, rtol=2.0e-7, atol=2.0e-7)

    unitOut = cconsenrich.cPuncObservationInformation(
        matrix.astype(np.float32),
        0.05,
        None,
        0.5,
        4.0,
    )
    np.testing.assert_allclose(
        unitOut,
        np.sum(1.0 / (matrix + 0.05), axis=0),
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    with pytest.raises(ValueError, match="lambdaExp length"):
        cconsenrich.cPuncObservationInformation(
            matrix.astype(np.float32),
            0.05,
            np.ones(2, dtype=np.float64),
            0.5,
            4.0,
        )


@pytest.mark.correctness
def _casePuncEvidenceKernelAcceptsFullSeedQ():
    levelState = np.asarray([[0.0], [1.0], [3.0], [6.0]], dtype=np.float32)
    levelCov = np.zeros((4, 1, 1), dtype=np.float32)
    levelLag = np.zeros((3, 1, 1), dtype=np.float32)
    levelSeedQ = np.diag([2.0, 99.0]).astype(np.float64)

    levelEvidence, levelDiagnostics = cconsenrich.cExpectedTransitionProcessEvidence(
        levelState,
        levelCov,
        levelLag,
        levelSeedQ,
    )
    np.testing.assert_allclose(
        levelEvidence,
        np.asarray([0.5, 2.0, 4.5], dtype=np.float64),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert levelDiagnostics["state_dim"] == 1

    matrixF = np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    fullState = np.asarray(
        [[0.0, 1.0], [1.0, 1.5], [2.5, 1.0], [3.5, 0.5]],
        dtype=np.float64,
    )
    fullCov = np.zeros((4, 2, 2), dtype=np.float64)
    fullLag = np.zeros((3, 2, 2), dtype=np.float64)
    fullSeedQ = np.diag([4.0, 2.0, 77.0]).astype(np.float64)

    fullEvidence, fullDiagnostics = cconsenrich.cExpectedTransitionProcessEvidence(
        fullState,
        fullCov,
        fullLag,
        fullSeedQ,
        matrixF=matrixF,
    )
    np.testing.assert_allclose(
        fullEvidence,
        np.full(3, 0.0625, dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert fullDiagnostics["state_dim"] == 2


@pytest.mark.correctness
def _casePuncScaleRebaseKernelUsesCanonicalStateModel():
    seedQ = np.diag([1.0, 4.0, 9.0]).astype(np.float64)
    baseQ = np.diag([2.0, 1.0, 99.0]).astype(np.float64)
    rawScale = np.asarray([3.0, 12.0], dtype=np.float64)

    levelScale, levelMaxErr, levelMedErr = cconsenrich.crebasePuncIntervalScales(
        seedQ,
        baseQ,
        rawScale,
        core.STATE_MODEL_LEVEL,
    )
    np.testing.assert_allclose(levelScale, rawScale / 2.0, rtol=1.0e-7, atol=1.0e-7)
    assert levelMaxErr == pytest.approx(0.0)
    assert levelMedErr == pytest.approx(0.0)

    fullScale, fullMaxErr, fullMedErr = cconsenrich.crebasePuncIntervalScales(
        seedQ,
        baseQ,
        rawScale,
        core.STATE_MODEL_LEVEL_TREND,
    )
    expected = np.sqrt((seedQ[0, 0] * rawScale / baseQ[0, 0]) * (seedQ[1, 1] * rawScale / baseQ[1, 1]))
    np.testing.assert_allclose(fullScale, expected, rtol=1.0e-6, atol=1.0e-6)
    assert fullMaxErr > 0.0
    assert fullMedErr > 0.0

    with pytest.raises(ValueError, match="stateModel"):
        cconsenrich.crebasePuncIntervalScales(seedQ, baseQ, rawScale, "other")


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


@pytest.mark.correctness
def _caseEffectiveSampleSizeSupportsMuncEvidenceMode():
    oscillatingEvidence = np.exp(
        np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    )
    correlatedEvidence = np.exp(
        np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    )
    activeMask = np.ones(oscillatingEvidence.size, dtype=np.uint8)
    _essOsc, etaOsc, lagsOsc = cconsenrich.cEstimateEffectiveSampleSize(
        oscillatingEvidence,
        1,
        activeMask=activeMask,
        logPositive=True,
        windowIntervals=4,
    )
    _essCorr, etaCorr, lagsCorr = cconsenrich.cEstimateEffectiveSampleSize(
        correlatedEvidence,
        3,
        activeMask=activeMask,
        logPositive=True,
        windowIntervals=4,
    )
    assert etaOsc == pytest.approx(1.0)
    assert lagsOsc == 0
    assert 1.0 < etaCorr <= 4.0
    assert lagsCorr > 0

    maskedEvidence = np.asarray([1.0, 2.0, 0.0, 4.0, 8.0], dtype=np.float64)
    maskedActive = np.asarray([1, 1, 0, 1, 1], dtype=np.uint8)
    _essMasked, etaMasked, _lagsMasked = cconsenrich.cEstimateEffectiveSampleSize(
        maskedEvidence,
        2,
        activeMask=maskedActive,
        logPositive=True,
        windowIntervals=4,
    )
    assert 1.0 <= etaMasked <= 4.0
    with pytest.raises(ValueError, match="positive"):
        cconsenrich.cEstimateEffectiveSampleSize(
            maskedEvidence,
            2,
            activeMask=np.ones(maskedEvidence.size, dtype=np.uint8),
            logPositive=True,
            windowIntervals=4,
        )


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _caseMonoFuncUsesSameLogKernelForFloat32AndFloat64(dtype):
    x = np.array([-3.0, -0.25, 0.0, 2.5, 9.0], dtype=dtype)

    out, sentinel = cconsenrich.monoFunc(x, offset=0.75, scale=1.7)
    expected = _monoLogReference(x, offset=0.75, scale=1.7)

    assert sentinel == pytest.approx(-1.0)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, expected)


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _caseTransformLogRatioKernelMatchesReferenceForFloat32AndFloat64(dtype):
    treatment = np.array([9.0, -4.0, 0.0, 12.0, 4.0], dtype=dtype)
    control = np.array([2.0, 5.0, -3.0, 3.0, 4.0], dtype=dtype)
    out = np.empty_like(treatment)

    returned = cconsenrich.cTransformWithInputInto(
        treatment,
        control,
        out,
        logOffset=0.5,
        logMult=1.3,
    )
    allocated = cconsenrich.cTransformWithInput(
        treatment,
        control,
        logOffset=0.5,
        logMult=1.3,
    )
    expected = _logRatioReference(treatment, control, offset=0.5, scale=1.3)

    assert returned is out
    assert out.dtype == dtype
    assert allocated.dtype == dtype
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(allocated, expected)


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _caseTransformPureLogPathMatchesMonoReferenceForFloat32AndFloat64(dtype):
    x = np.array([-2.0, -0.5, 0.0, 3.0, 8.0], dtype=dtype)
    inPlace = x.copy()

    returned = cconsenrich.cTransformInPlace(
        inPlace,
        logOffset=0.75,
        logMult=1.7,
    )
    allocated = cconsenrich.cTransform(
        x,
        logOffset=0.75,
        logMult=1.7,
    )
    expected = _monoLogReference(x, offset=0.75, scale=1.7)

    assert returned is inPlace
    assert inPlace.dtype == dtype
    assert allocated.dtype == dtype
    np.testing.assert_array_equal(inPlace, expected)
    np.testing.assert_array_equal(allocated, expected)


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _caseGenericTransformModesMatchReference(dtype):
    x = np.array([-1.0, 0.0, 1.0, 4.0, 16.0], dtype=dtype)
    modeCases = (
        (
            "sqrt",
            dict(
                inputOffset=0.25,
                inputScale=2.0,
                outputScale=1.7,
                outputOffset=-0.2,
            ),
        ),
        (
            "anscombe",
            dict(
                inputOffset=0.375,
                inputScale=1.0,
                outputScale=2.0,
                outputOffset=0.0,
            ),
        ),
        (
            "asinh",
            dict(
                inputOffset=-0.5,
                inputScale=2.0,
                outputScale=0.75,
                outputOffset=0.1,
            ),
        ),
        (
            "asinh_sqrt",
            dict(
                inputOffset=0.0,
                inputScale=2.0,
                outputScale=2.0,
                outputOffset=0.0,
            ),
        ),
        (
            "generalized_log",
            dict(
                inputOffset=0.0,
                inputScale=2.0,
                outputScale=1.2,
                outputOffset=-0.1,
                shape=0.75,
            ),
        ),
        (
            "identity",
            dict(
                inputOffset=-1.0,
                inputScale=2.0,
                outputScale=3.0,
                outputOffset=0.5,
            ),
        ),
    )

    for mode, kwargs in modeCases:
        inPlace = x.copy()
        returned = cconsenrich.cTransformInPlace(inPlace, mode=mode, **kwargs)
        allocated = cconsenrich.cTransform(x, mode=mode, **kwargs)
        expected = _countTransformReference(x, mode=mode, **kwargs)

        assert returned is inPlace
        assert inPlace.dtype == dtype
        assert allocated.dtype == dtype
        if dtype == np.float32:
            np.testing.assert_allclose(inPlace, expected, rtol=1.0e-6, atol=1.0e-6)
            np.testing.assert_allclose(allocated, expected, rtol=1.0e-6, atol=1.0e-6)
        else:
            np.testing.assert_allclose(inPlace, expected, rtol=1.0e-12, atol=1.0e-12)
            np.testing.assert_allclose(allocated, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _caseGenericTransformWithInputUsesTransformDifferences(dtype):
    treatment = np.array([0.0, 4.0, 16.0, 25.0], dtype=dtype)
    control = np.array([0.0, 1.0, 4.0, 36.0], dtype=dtype)
    modeCases = (
        ("sqrt", dict(inputOffset=0.25, inputScale=2.0, outputScale=1.5)),
        ("anscombe", dict(inputOffset=0.375, outputScale=2.0)),
        ("asinh", dict(inputOffset=-0.5, inputScale=2.0, outputScale=0.8)),
        ("asinh_sqrt", dict(inputOffset=0.0, inputScale=2.0, outputScale=2.0)),
        (
            "generalized_log",
            dict(inputOffset=0.0, inputScale=2.0, outputScale=1.1, shape=0.75),
        ),
        ("identity", dict(inputOffset=0.0, inputScale=2.0, outputScale=3.0)),
    )

    for mode, kwargs in modeCases:
        out = np.empty_like(treatment)
        returned = cconsenrich.cTransformWithInputInto(
            treatment,
            control,
            out,
            mode=mode,
            outputOffset=99.0,
            **kwargs,
        )
        allocated = cconsenrich.cTransformWithInput(
            treatment,
            control,
            mode=mode,
            outputOffset=99.0,
            **kwargs,
        )
        expected = _countTransformDiffReference(
            treatment,
            control,
            mode=mode,
            outputOffset=99.0,
            **kwargs,
        )

        assert returned is out
        assert out.dtype == dtype
        assert allocated.dtype == dtype
        if dtype == np.float32:
            np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)
            np.testing.assert_allclose(allocated, expected, rtol=1.0e-6, atol=1.0e-6)
        else:
            np.testing.assert_allclose(out, expected, rtol=1.0e-12, atol=1.0e-12)
            np.testing.assert_allclose(allocated, expected, rtol=1.0e-12, atol=1.0e-12)
        assert out[1] > 0.0
        assert out[-1] < 0.0


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _caseAnscombePresetUsesCanonicalDefaults(dtype):
    x = np.array([0.0, 1.0, 4.0, 16.0], dtype=dtype)
    treatment = np.array([0.0, 4.0, 16.0], dtype=dtype)
    control = np.array([1.0, 1.0, 4.0], dtype=dtype)

    transformed = cconsenrich.cTransform(x, mode="anscombe")
    expected = _countTransformReference(
        x,
        mode="anscombe",
        inputOffset=0.375,
        outputScale=2.0,
    )
    diff = cconsenrich.cTransformWithInput(treatment, control, mode="anscombe")
    expectedDiff = _countTransformDiffReference(
        treatment,
        control,
        mode="anscombe",
        inputOffset=0.375,
        outputScale=2.0,
    )

    if dtype == np.float32:
        np.testing.assert_allclose(transformed, expected, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(diff, expectedDiff, rtol=1.0e-6, atol=1.0e-6)
    else:
        np.testing.assert_allclose(transformed, expected, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(diff, expectedDiff, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.correctness
def _caseGenericTransformInvalidOptionsFailGracefully():
    x = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="mode"):
        cconsenrich.cTransform(x, mode="banana")
    with pytest.raises(ValueError, match="inputScale"):
        cconsenrich.cTransform(x, mode="asinh", inputScale=0.0)
    with pytest.raises(ValueError, match="shape"):
        cconsenrich.cTransform(x, mode="generalized_log", shape=0.0)


@pytest.mark.correctness
def _caseCSFMedianSelectionHandlesOddLengthDuplicates():
    base = (20.0 + (np.arange(601, dtype=np.float32) % 17)).astype(np.float32)
    factors = np.array([0.75, 1.0, 1.0], dtype=np.float32)
    chromMat = factors[:, None] * base[None, :]

    out = cconsenrich.cSF(chromMat, centerMedian=True, minRefDist=1)

    np.testing.assert_allclose(out, _expectedCSF(chromMat), rtol=1.0e-7, atol=1.0e-7)


@pytest.mark.correctness
def _caseCSFMedianSelectionHandlesEvenLengthDuplicates():
    base = (30.0 + (np.arange(600, dtype=np.float32) % 23)).astype(np.float32)
    factors = np.array([0.75, 1.25, 1.25, 2.0], dtype=np.float32)
    chromMat = factors[:, None] * base[None, :]

    out = cconsenrich.cSF(chromMat, centerMedian=True, minRefDist=1)

    np.testing.assert_allclose(out, _expectedCSF(chromMat), rtol=1.0e-7, atol=1.0e-7)


def test_csf_default_c_stdout_stderr_silent(capfd):
    base = (25.0 + (np.arange(601, dtype=np.float32) % 19)).astype(np.float32)
    chromMat = np.array([0.8, 1.0, 1.3], dtype=np.float32)[:, None] * base[None, :]

    cconsenrich.cSF(chromMat, centerMedian=True, minRefDist=1)

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


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
def _caseCTransformWithInputReturnsFloat32LogRatio():
    treatment = np.array([9.0, 0.0, 4.0], dtype=np.float32)
    control = np.array([2.0, 5.0, 4.0], dtype=np.float32)

    out = cconsenrich.cTransformWithInput(
        treatment,
        control,
        logOffset=1.0,
        logMult=1.0,
    )
    expected = np.log(treatment + 1.0) - np.log(control + 1.0)

    assert out.dtype == np.float32
    assert np.allclose(out, expected.astype(np.float32))
    assert out[1] < 0.0
    assert out[2] == pytest.approx(0.0)


@pytest.mark.correctness
def _caseCTransformWithInputReturnsFloat64LogRatio():
    treatment = np.array([15.0, 2.0], dtype=np.float64)
    control = np.array([3.0, 8.0], dtype=np.float64)

    out = cconsenrich.cTransformWithInput(
        treatment,
        control,
        logOffset=0.5,
        logMult=1.0 / np.log(2.0),
    )
    expected = (np.log(treatment + 0.5) - np.log(control + 0.5)) / np.log(2.0)

    assert out.dtype == np.float64
    assert np.allclose(out, expected)
    assert out[1] < 0.0


@pytest.mark.correctness
def _caseCTransformWithInputIntoWritesOutputInPlace():
    treatment = np.array([9.0, 0.0, 4.0], dtype=np.float32)
    control = np.array([2.0, 5.0, 4.0], dtype=np.float32)
    out = np.empty_like(treatment)

    returned = cconsenrich.cTransformWithInputInto(
        treatment,
        control,
        out,
        logOffset=1.0,
        logMult=1.0,
    )
    expected = cconsenrich.cTransformWithInput(
        treatment,
        control,
        logOffset=1.0,
        logMult=1.0,
    )

    assert returned is out
    assert out.dtype == np.float32
    assert np.allclose(out, expected)
    assert out[1] < 0.0


@pytest.mark.correctness
def _caseCTransformInPlacePureLogMutatesFloat32Array():
    x = np.array([0.0, 3.0, 8.0], dtype=np.float32)
    original = x.copy()

    returned = cconsenrich.cTransformInPlace(
        x,
        logOffset=1.0,
        logMult=2.0,
    )

    assert returned is x
    assert x.dtype == np.float32
    assert np.allclose(x, (2.0 * np.log(original + 1.0)).astype(np.float32))


@pytest.mark.correctness
def _caseCTransformInPlaceMatchesAllocatingTransformForFloat64():
    x = np.linspace(0.0, 5.0, 256, dtype=np.float64)
    x[120:140] += 10.0
    logged = _monoLogReference(x, offset=1.0, scale=1.0)
    expected = cconsenrich.cTransform(
        x,
        logOffset=1.0,
        logMult=1.0,
    )
    in_place = x.copy()

    returned = cconsenrich.cTransformInPlace(
        in_place,
        logOffset=1.0,
        logMult=1.0,
    )

    assert returned is in_place
    assert in_place.dtype == np.float64
    assert np.allclose(in_place, expected)
    assert np.allclose(in_place, logged)


@pytest.mark.correctness
def _caseCenterMBAppliesMedianFilterInPlace():
    cases = (
        (
            np.array(
                [
                    [1.0, 2.0, 100.0, 4.0, 5.0],
                    [-8.0, -4.0, -2.0, 0.0, 3.0],
                ],
                dtype=np.float32,
            ),
            25,
            40_001,
        ),
        (
            np.array([2.0, 8.0, 1.0, 6.0], dtype=np.float64),
            500_000,
            3,
        ),
    )
    for tracks, intervalSizeBP, expectedWindow in cases:
        original = tracks.copy()

        stats_ = core.centerMBInPlace(
            tracks,
            intervalSizeBP=intervalSizeBP,
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
        intervalSizeBP=500_000,
        centerMBMethod="savgol",
    )
    savgolFilter = spySig.savgol_filter(
        savgolOriginal,
        window_length=3,
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
    np.testing.assert_allclose(savgolTracks, savgolOriginal - savgolFilter)

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


@pytest.mark.correctness
def _caseSingleEndDetection():
    # case: single-end BAM
    bamFiles = [str(TESTS_DIR / "smallTest.bam")]
    pairedEndStatus = misc_util.alignmentFilesArePairedEnd(bamFiles, maxReads=1_000)
    assert isinstance(pairedEndStatus, list)
    assert len(pairedEndStatus) == 1
    assert isinstance(pairedEndStatus[0], bool)
    assert pairedEndStatus[0] is False


@pytest.mark.correctness
def _casePairedEndDetection():
    # case: paired-end BAM
    bamFiles = [str(TESTS_DIR / "smallTest2.bam")]
    pairedEndStatus = misc_util.alignmentFilesArePairedEnd(bamFiles, maxReads=1_000)
    assert isinstance(pairedEndStatus, list)
    assert len(pairedEndStatus) == 1
    assert isinstance(pairedEndStatus[0], bool)
    assert pairedEndStatus[0] is True


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
        outputPath = peaks.solveRocco(
            stateBedGraphFile=str(stateBedGraphPath),
            uncertaintyBedGraphFile=str(uncertaintyBedGraphPath),
            numBootstrap=32,
            thresholdZ=2.0,
            randSeed=42,
        )
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
def _casePerIntervalOutputDiagnosticsUseEffectiveNoiseAndGainComponents():
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

    tracks = core._perIntervalOutputDiagnosticTracks(
        stateCovarForward=stateCovarForward,
        matrixMunc=matrixMunc,
        matrixQ0=matrixQ0,
        matrixF=matrixF,
        stateCovarInit=1.0,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        lambdaExp=lambdaExp,
        processPrecExp=processPrecExp,
        processQScale=np.asarray([9.0, 2.0, 3.0], dtype=np.float32),
        pNoiseForward=None,
        pad=0.1,
        obsPrecisionMultiplierMin=0.25,
        obsPrecisionMultiplierMax=4.0,
        procPrecisionMultiplierMin=0.25,
        procPrecisionMultiplierMax=4.0,
    )

    np.testing.assert_allclose(tracks["preKappaQLevel"], [0.2, 0.4, 0.6])
    np.testing.assert_allclose(tracks["preKappaQTrend"], [0.05, 0.1, 0.15])
    np.testing.assert_allclose(tracks["effectiveQLevel"], [0.2, 0.2, 0.15])
    np.testing.assert_allclose(tracks["effectiveQTrend"], [0.05, 0.05, 0.0375])
    np.testing.assert_allclose(tracks["puncQScale"], [1.0, 2.0, 3.0])
    qMeta = core._processQDiagnosticsMetadata(
        matrixQ0=matrixQ0,
        intervalCount=3,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        processPrecExp=processPrecExp,
        processQScale=np.asarray([9.0, 2.0, 3.0], dtype=np.float32),
        pNoiseForward=None,
        useAPN=False,
        processPrecisionRequested=True,
        processPrecisionEffective=True,
        procPrecisionMultiplierMin=0.25,
        procPrecisionMultiplierMax=4.0,
    )
    assert qMeta["effectiveQTraceMin"] == pytest.approx(0.1875)
    assert qMeta["effectiveQTraceMedian"] == pytest.approx(0.25)
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
        ECM_useAPN=False,
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
        ECM_useAPN=False,
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


def _checkCFixedBackgroundPrecisionUpdates(levelOnly):
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
    nu = 5.0
    pad = 0.01

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
            ECM_robustTNu=nu,
            obsPrecisionMultiplierMin=0.1,
            obsPrecisionMultiplierMax=10.0,
            procPrecisionMultiplierMin=0.1,
            procPrecisionMultiplierMax=10.0,
            ECM_useObsPrecisionReweighting=True,
            ECM_useProcessPrecisionReweighting=True,
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
            ECM_robustTNu=nu,
            obsPrecisionMultiplierMin=0.1,
            obsPrecisionMultiplierMax=10.0,
            procPrecisionMultiplierMin=0.1,
            procPrecisionMultiplierMax=10.0,
            ECM_useObsPrecisionReweighting=True,
            ECM_useProcessPrecisionReweighting=True,
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

    expectedLambda = []
    for k in range(n):
        obsU2 = 0.0
        p00 = max(float(stateCovarSmoothed[k, 0, 0]), 0.0)
        for j in range(m):
            obsU2 += (
                (float(matrixData[j, k]) - float(stateSmoothed[k, 0])) ** 2 + p00
            ) / (float(matrixMunc[j, k]) + pad)
        expectedLambda.append((nu + m) / (nu + obsU2))
    expectedLambda = np.clip(expectedLambda, 0.1, 10.0)

    expectedProcess = [1.0]
    qInv = np.linalg.inv(np.asarray(matrixQ0[:stateDim, :stateDim], dtype=np.float64))
    f = np.asarray(matrixF[:stateDim, :stateDim], dtype=np.float64)
    for k in range(n - 1):
        x = stateSmoothed[k, :stateDim]
        y = stateSmoothed[k + 1, :stateDim]
        covX = stateCovarSmoothed[k, :stateDim, :stateDim]
        covY = stateCovarSmoothed[k + 1, :stateDim, :stateDim]
        cross = lagCovSmoothed[k, :stateDim, :stateDim]
        exx = covX + np.outer(x, x)
        eyy = covY + np.outer(y, y)
        exy = cross + np.outer(x, y)
        innovationMoment = eyy - (exy.T @ f.T) - (f @ exy) + (f @ exx @ f.T)
        diagonal = np.diag_indices(stateDim)
        innovationMoment[diagonal] = np.maximum(innovationMoment[diagonal], 0.0)
        delta = float(np.sum(qInv * innovationMoment.T))
        expectedProcess.append((nu + stateDim) / (nu + max(delta, 0.0)))
    expectedProcess = np.clip(expectedProcess, 0.1, 10.0)

    np.testing.assert_allclose(lambdaExp, expectedLambda, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(
        processPrecExp,
        expectedProcess,
        rtol=2.0e-6,
        atol=2.0e-6,
    )

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
            returnIntermediates=True,
            t_innerIters=1,
            logIterations=False,
        )
    assert disabled[6] is None
    assert disabled[7] is None


@pytest.mark.correctness
def _caseCFixedBackgroundPrecisionUpdatesMatchStudentTEquations():
    _checkCFixedBackgroundPrecisionUpdates(True)
    _checkCFixedBackgroundPrecisionUpdates(False)


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
def _caseFitParamsDropsProcBlockScaleOptions():
    removedFields = {
        _REMOVED_EM_PREFIX + "scaleToMedian",
        _REMOVED_EM_PREFIX + "alphaEMA",
        _REMOVED_EM_PREFIX + "scaleLOW",
        _REMOVED_EM_PREFIX + "scaleHIGH",
        _REMOVED_EM_PREFIX + "useProcBlockScale",
        _REMOVED_EM_PREFIX + "useReplicateScale",
        _REMOVED_EM_PREFIX + "repScaleLOW",
        _REMOVED_EM_PREFIX + "repScaleHIGH",
    }
    assert removedFields.isdisjoint(core.fitParams._fields)


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
    assert core._normalizeStateModel(None) == core.STATE_MODEL_LEVEL_TREND
    assert core._normalizeStateModel("levelTrend") == core.STATE_MODEL_LEVEL_TREND
    assert core._normalizeStateModel("level") == core.STATE_MODEL_LEVEL
    for badMode in ("level-trend", "two_state", "one-state", "scalar", "levelSlope"):
        with pytest.raises(ValueError, match="stateModel"):
            core._normalizeStateModel(badMode)


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
        ECM_useAPN=False,
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


@pytest.mark.correctness
def _caseLevelEmbeddedForwardBackwardAgreementWithPrecisionMultipliers():
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
        "processQScale": np.asarray(
            [1.00, 0.70, 1.80, 0.55, 1.20, 2.40, 0.90],
            dtype=np.float32,
        ),
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


@pytest.mark.correctness
def _casePuncProcessNoiseCalibrationRebasesClampedBaseQ():
    intervalCount = 16
    seedQLevel = 1.0e-2
    desiredEvidence = 100.0
    stateStep = float(np.sqrt(seedQLevel * desiredEvidence))
    state = (np.arange(intervalCount, dtype=np.float64) * stateStep).reshape(-1, 1)
    warmupFit = {
        "stateSmoothed": state.astype(np.float32),
        "stateCovarSmoothed": np.zeros((intervalCount, 1, 1), dtype=np.float32),
        "lagCovSmoothed": np.zeros((intervalCount, 1, 1), dtype=np.float32),
        "matrixMunc": np.vstack(
            [
                np.linspace(0.05, 0.25, intervalCount),
                np.linspace(0.25, 0.05, intervalCount),
            ]
        ).astype(np.float32),
        "lambdaExp": np.linspace(0.5, 2.0, intervalCount, dtype=np.float32),
    }

    matrixQ, processQScale, info = core._fitPuncProcessNoise(
        warmupFit=warmupFit,
        matrixMunc=warmupFit["matrixMunc"],
        matrixF=np.asarray([[1.0]], dtype=np.float32),
        seedQ=np.diag([seedQLevel, seedQLevel]).astype(np.float32),
        stateModel=core.STATE_MODEL_LEVEL,
        pad=1.0e-4,
        minQ=1.0e-4,
        maxQ=2.0e-2,
        blockLenIntervals=3,
        processCovariates=None,
        puncLocalWindowMultiplier=1.0,
        puncDependenceMultiplier=1.0,
        puncMinScale=0.1,
        puncMaxScale=10.0,
        puncMinWindowWeight=0.0,
        puncPriorRidge=1.0e-3,
        puncLevelBufferZ=0.0,
        puncUseReliabilityWeightedWindows=True,
        observationPrecisionMultiplierMin=0.25,
        observationPrecisionMultiplierMax=4.0,
    )

    assert info["processNoisePolicy"] == "punc"
    assert info["priorDesignColumnCount"] == 2
    assert info["priorDesignColumns"] == ("intercept", "stateLevelMidpoint")
    assert info["processCovariateCount"] == 0
    assert info["puncPriorDfSource"] == "method_of_moments"
    assert info["puncLevelBufferZ"] == pytest.approx(0.0)
    assert info["puncUseReliabilityWeightedWindows"] is True
    assert info["puncLevelBufferEnabled"] is False
    assert np.isfinite(info["puncPriorDf"])
    assert info["puncPriorDf"] >= 4.0
    assert info["puncPriorDfMomentWindowCount"] > 0
    assert "processQScale" not in info
    assert matrixQ[0, 0] == pytest.approx(2.0e-2)
    assert processQScale[0] == pytest.approx(1.0)
    np.testing.assert_allclose(processQScale[1:], np.full(intervalCount - 1, 5.0), rtol=5.0e-5)
    assert info["baseQClampChanged"] is True
    assert info["baseQClampMaxRelativeChange"] > 0.0
    assert info["qScaleDecompositionMaxLogError"] == pytest.approx(0.0, abs=1.0e-6)
    assert info["preKappaQLevel"] == pytest.approx(float(matrixQ[0, 0]))
    assert info["preKappaQTrend"] == pytest.approx(0.0)


@pytest.mark.correctness
def _casePuncReliabilityWeightedWindowsSwitchesLocalEvidence(monkeypatch):
    intervalCount = 8
    evidenceTarget = np.asarray([1.0, 16.0, 4.0, 25.0, 9.0, 36.0, 16.0])
    increments = np.sqrt(evidenceTarget)
    state = np.concatenate(([0.0], np.cumsum(increments))).reshape(-1, 1)
    warmupFit = {
        "stateSmoothed": state.astype(np.float32),
        "stateCovarSmoothed": np.zeros((intervalCount, 1, 1), dtype=np.float32),
        "lagCovSmoothed": np.zeros((intervalCount, 1, 1), dtype=np.float32),
        "matrixMunc": np.ones((1, intervalCount), dtype=np.float32),
        "lambdaExp": np.asarray(
            [100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        ),
    }
    captured: list[dict[str, np.ndarray]] = []

    def capturePriorDf(localEvidence, localPrior, nuLocal, weights, **kwargs):
        captured.append(
            {
                "localEvidence": np.asarray(localEvidence, dtype=np.float64).copy(),
                "nuLocal": np.asarray(nuLocal, dtype=np.float64).copy(),
                "weights": np.asarray(weights, dtype=np.float64).copy(),
            }
        )
        return (
            4.0,
            1.0,
            {
                "puncPriorDfMomentWindowCount": int(np.size(localEvidence)),
                "puncPriorDfMomentEffectiveWindowCount": float(np.size(localEvidence)),
                "puncPriorDfMomentLogRatioVariance": 0.0,
                "puncPriorDfMomentSamplingVariance": 0.0,
                "puncPriorDfMomentExcessVariance": 0.0,
                "puncPriorDfMomentScale": 1.0,
                "puncPriorDfMomentWinsorLower": float("nan"),
                "puncPriorDfMomentWinsorUpper": float("nan"),
                "puncPriorDfMomentReason": "ok",
            },
        )

    monkeypatch.setattr(core, "_estimatePuncPriorDfMethodOfMoments", capturePriorDf)

    def run(useWeighted):
        _matrixQ, _processQScale, info = core._fitPuncProcessNoise(
            warmupFit=warmupFit,
            matrixMunc=warmupFit["matrixMunc"],
            matrixF=np.asarray([[1.0]], dtype=np.float32),
            seedQ=np.diag([1.0, 1.0]).astype(np.float32),
            stateModel=core.STATE_MODEL_LEVEL,
            pad=1.0e-4,
            minQ=1.0e-4,
            maxQ=100.0,
            blockLenIntervals=3,
            processCovariates=None,
            puncLocalWindowMultiplier=1.0,
            puncDependenceMultiplier=1.0,
            puncMinScale=0.01,
            puncMaxScale=100.0,
            puncMinWindowWeight=0.0,
            puncPriorRidge=1.0e-3,
            puncLevelBufferZ=0.0,
            puncUseReliabilityWeightedWindows=useWeighted,
            observationPrecisionMultiplierMin=0.01,
            observationPrecisionMultiplierMax=200.0,
        )
        return info

    weightedInfo = run(True)
    unitInfo = run(False)

    assert weightedInfo["puncUseReliabilityWeightedWindows"] is True
    assert unitInfo["puncUseReliabilityWeightedWindows"] is False
    assert len(captured) == 2
    weightedWindows, unitWindows = captured
    assert weightedInfo["validTransitionCount"] == unitInfo["validTransitionCount"]
    assert weightedInfo["windowCount"] == unitInfo["windowCount"]
    np.testing.assert_allclose(
        unitWindows["weights"],
        np.asarray([2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0]),
    )
    assert not np.allclose(
        weightedWindows["localEvidence"],
        unitWindows["localEvidence"],
    )
    assert weightedWindows["nuLocal"][0] < unitWindows["nuLocal"][0]


@pytest.mark.correctness
def _casePuncPriorDfMethodOfMomentsRecoversKnownFDispersion():
    n = 1000
    nuLocal = 20.0
    priorDfTrue = 12.0
    priorScaleTrue = 1.5
    probs = (np.arange(n, dtype=np.float64) + 0.5) / n
    localEvidence = priorScaleTrue * stats.f.ppf(probs, nuLocal, priorDfTrue)
    localPrior = np.ones(n, dtype=np.float64)
    nu = np.full(n, nuLocal, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64)

    priorDf, priorScale, diagnostics = core._estimatePuncPriorDfMethodOfMoments(
        localEvidence,
        localPrior,
        nu,
        weights,
        winsorTail=0.0,
        minScale=0.1,
        maxScale=10.0,
    )

    assert diagnostics["puncPriorDfMomentReason"] == "ok"
    assert priorDf == pytest.approx(priorDfTrue, rel=0.02)
    assert priorScale == pytest.approx(priorScaleTrue, rel=0.02)


@pytest.mark.correctness
def _casePuncPriorDfMethodOfMomentsHandlesDegenerateInputs():
    localEvidence = np.asarray([np.nan, 1.0, 1.0, np.inf, 0.0, 1.0], dtype=np.float64)
    localPrior = np.ones_like(localEvidence)
    nu = np.full_like(localEvidence, 8.0)
    weights = np.ones_like(localEvidence)

    priorDf, priorScale, diagnostics = core._estimatePuncPriorDfMethodOfMoments(
        localEvidence,
        localPrior,
        nu,
        weights,
        minWindows=2,
    )

    assert priorDf == pytest.approx(1.0e6)
    assert np.isfinite(priorScale)
    assert diagnostics["puncPriorDfMomentReason"] == "no_excess_dispersion"


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
        puncMaxScale=4.0,
        processNoiseCalibration=core.PROCESS_NOISE_CALIBRATION_PUNC,
        robustTNu=8.0,
    )

    assert diagnostics["qSeedSource"] == "sameTrackEB"
    assert 0.3 * qTrue <= diagnostics["qSeedLevelFinal"] <= 3.0 * qTrue
    assert matrixQ[0, 0] == pytest.approx(diagnostics["qSeedLevelFinal"])
    assert matrixQ[1, 1] == pytest.approx(
        diagnostics["qSeedLevelFinal"] * core.PROCESS_DEFAULT_PUNC_TREND_SEED_RATIO
    )


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
        puncMaxScale=4.0,
        processNoiseCalibration=core.PROCESS_NOISE_CALIBRATION_PUNC,
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
        puncMaxScale=4.0,
        processNoiseCalibration=core.PROCESS_NOISE_CALIBRATION_PUNC,
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
            core._weightedQuantile(
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
def _casePuncDeadbandPriorShrinksNearNullPriorScale():
    intervalCount = 24
    state = np.zeros((intervalCount, 1), dtype=np.float64)
    state[intervalCount // 2 :, 0] = 5.0
    stateCov = np.ones((intervalCount, 1, 1), dtype=np.float64) * 0.04
    lagCov = np.zeros((intervalCount, 1, 1), dtype=np.float64)
    lagCov[:-1, 0, 0] = 0.02
    warmupFit = {
        "stateSmoothed": state.astype(np.float32),
        "stateCovarSmoothed": stateCov.astype(np.float32),
        "lagCovSmoothed": lagCov.astype(np.float32),
        "matrixMunc": np.full((3, intervalCount), 0.1, dtype=np.float32),
        "lambdaExp": np.ones(intervalCount, dtype=np.float32),
    }

    _matrixQ, _processQScale, info = core._fitPuncProcessNoise(
        warmupFit=warmupFit,
        matrixMunc=warmupFit["matrixMunc"],
        matrixF=np.asarray([[1.0]], dtype=np.float32),
        seedQ=np.diag([1.0e-2, 1.0e-2]).astype(np.float32),
        stateModel=core.STATE_MODEL_LEVEL,
        pad=1.0e-4,
        minQ=1.0e-5,
        maxQ=1.0,
        blockLenIntervals=3,
        processCovariates=None,
        puncLocalWindowMultiplier=1.0,
        puncDependenceMultiplier=1.0,
        puncMinScale=0.25,
        puncMaxScale=4.0,
        puncMinWindowWeight=0.0,
        puncPriorRidge=1.0e-3,
        puncLevelBufferZ=1.64,
        puncUseReliabilityWeightedWindows=True,
        observationPrecisionMultiplierMin=0.25,
        observationPrecisionMultiplierMax=4.0,
    )

    before = info["puncPriorScaleBeforeDeadbandSummary"]
    after = info["puncPriorScaleAfterDeadbandSummary"]
    assert info["puncDeadbandPriorEnabled"] is True
    assert info["puncDeadbandMeanProbability"] > 0.0
    assert info["puncDeadbandHighProbabilityFraction"] > 0.0
    assert info["puncDeadbandNullScale"] == pytest.approx(0.25)
    assert after["min"] < before["min"]
    assert info["puncDeadbandHighProbabilityThreshold"] == pytest.approx(0.8)
    assert info["puncDeadbandHighTransitionCount"] > 0
    assert info["puncDeadbandHighTransitionFraction"] > 0.0
    assert info["puncDeadbandHighRawScaleSummary"]["median"] is not None
    assert info["puncDeadbandHighTransitionScaleSummary"]["median"] is not None
    assert info["puncDeadbandHighRebasedProcessQScaleSummary"]["median"] is not None
    assert info["puncDeadbandHighQLevelSummary"]["median"] is not None
    assert info["puncDeadbandHighQTrendSummary"]["median"] == pytest.approx(0.0)


@pytest.mark.correctness
def _casePuncDeadbandPriorNegligibleOutsideDeadband():
    intervalCount = 20
    state = np.full((intervalCount, 1), 10.0, dtype=np.float64)
    stateCov = np.ones((intervalCount, 1, 1), dtype=np.float64) * 0.01
    lagCov = np.zeros((intervalCount, 1, 1), dtype=np.float64)
    lagCov[:-1, 0, 0] = 0.005
    warmupFit = {
        "stateSmoothed": state.astype(np.float32),
        "stateCovarSmoothed": stateCov.astype(np.float32),
        "lagCovSmoothed": lagCov.astype(np.float32),
        "matrixMunc": np.full((2, intervalCount), 0.1, dtype=np.float32),
        "lambdaExp": np.ones(intervalCount, dtype=np.float32),
    }

    _matrixQ, _processQScale, info = core._fitPuncProcessNoise(
        warmupFit=warmupFit,
        matrixMunc=warmupFit["matrixMunc"],
        matrixF=np.asarray([[1.0]], dtype=np.float32),
        seedQ=np.diag([1.0e-2, 1.0e-2]).astype(np.float32),
        stateModel=core.STATE_MODEL_LEVEL,
        pad=1.0e-4,
        minQ=1.0e-5,
        maxQ=1.0,
        blockLenIntervals=3,
        processCovariates=None,
        puncLocalWindowMultiplier=1.0,
        puncDependenceMultiplier=1.0,
        puncMinScale=0.25,
        puncMaxScale=4.0,
        puncMinWindowWeight=0.0,
        puncPriorRidge=1.0e-3,
        puncLevelBufferZ=1.64,
        puncUseReliabilityWeightedWindows=True,
        observationPrecisionMultiplierMin=0.25,
        observationPrecisionMultiplierMax=4.0,
    )

    before = info["puncPriorScaleBeforeDeadbandSummary"]
    after = info["puncPriorScaleAfterDeadbandSummary"]
    assert info["puncDeadbandPriorEnabled"] is True
    assert info["puncDeadbandMeanProbability"] < 1.0e-6
    assert after["median"] == pytest.approx(before["median"])
    assert info["puncDeadbandHighTransitionCount"] == 0
    assert info["puncDeadbandHighRawScaleSummary"]["median"] is None
    assert info["puncDeadbandHighTransitionScaleSummary"]["median"] is None
    assert info["puncDeadbandHighRebasedProcessQScaleSummary"]["median"] is None
    assert info["puncDeadbandHighQLevelSummary"]["median"] is None


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
        minQ=1.0e-3,
        maxQ=1.0,
        qPriorLevel=1.0e-3,
        qPriorTrend=1.0e-3,
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
        "puncQScale",
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
    warmupDiagnostics = diagnostics["process_noise_warmup_fit"]
    assert warmupDiagnostics is not None
    assert warmupDiagnostics["fixed_background_ecm"]
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
        minQ=1.0e-3,
        maxQ=1.0,
        qPriorLevel=1.0e-3,
        qPriorTrend=1.0e-3,
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
    assert qInfo["processNoisePolicy"] == "punc"
    assert "".join(("block", "Mode")) not in qInfo
    assert "_".join(("process", "q", "calibration")) not in diagnostics
    assert qInfo["preKappaQLevel"] > 0.0
    assert qInfo["preKappaQTrend"] > 0.0
    assert "processQScaleSummary" in qInfo
    assert "processQScale" not in qInfo
    assert "_puncDeadbandHighIntervalMask" not in qInfo
    assert "puncDeadbandHighKappaSummary" in qInfo
    assert "puncDeadbandHighEffectiveQLevelSummary" in qInfo
    assert "puncDeadbandHighEffectiveQTrendSummary" in qInfo


@pytest.mark.correctness
def _caseRunConsenrichFlatWarmupInitializerUsesMinQ():
    n = 24
    m = 3
    matrixData = np.full((m, n), 0.25, dtype=np.float32)
    matrixMunc = np.full((m, n), 0.20, dtype=np.float32)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=5.0e-2,
        maxQ=1.0,
        qPriorLevel=5.0e-2,
        qPriorTrend=5.0e-2,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=6,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        ECM_useProcessPrecisionReweighting=False,
        processNoiseWarmupECMIters=1,
        returnDiagnostics=True,
    )

    qInfo = out[-1]["process_noise_calibration"]
    assert qInfo["processNoisePolicy"] == "punc"
    assert qInfo["preKappaQLevel"] >= 5.0e-2
    assert qInfo["qFloor"] == pytest.approx(5.0e-2)
    assert qInfo["qSeedSource"] == "sameTrackEB"
    assert qInfo["qSeedLevelFinal"] < 1.25 * qInfo["qFloor"]
    assert qInfo["matrixQ0Final"][0][0] == pytest.approx(qInfo["preKappaQLevel"])


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
        ECM_useAPN=False,
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
        np.round(stateSmoothed[:, 0].astype(np.float32), decimals=4),
    )
    qInfo = runDiagnostics["process_noise_calibration"]
    assert runDiagnostics["state_model"] == core.STATE_MODEL_LEVEL
    assert qInfo["processNoisePolicy"] == "punc"
    assert qInfo["preKappaQLevel"] > 0.0
    assert qInfo["preKappaQTrend"] == pytest.approx(0.0)
    assert qInfo["effectiveTrendLevelRatio"] == pytest.approx(0.0)
    assert qInfo["priorDesignColumnCount"] == 2


@pytest.mark.correctness
def _caseRunConsenrichProcessNoiseWarmupRestoresFinalReweighting(monkeypatch):
    rng = np.random.default_rng(7)
    n = 36
    m = 3
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    signalTrack = (2.0 * grid + 0.2 * np.sin(2.0 * np.pi * grid)).astype(np.float32)
    matrixData = np.vstack(
        [
            signalTrack + 0.03 * rng.normal(size=n) - 0.02,
            signalTrack + 0.03 * rng.normal(size=n),
            signalTrack + 0.03 * rng.normal(size=n) + 0.01,
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.12, dtype=np.float32)

    originalECM = cconsenrich.cfixedBackgroundECM
    ecmModes = []
    ecmLogIterations = []

    def _spyECM(*args, **kwargs):
        result = originalECM(*args, **kwargs)
        ecmLogIterations.append(bool(kwargs.get("logIterations", True)))
        ecmModes.append(
            (
                bool(kwargs.get("ECM_useProcessPrecisionReweighting")),
                bool(kwargs.get("ECM_useAPN")),
            )
        )
        return result

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", _spyECM)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=1.0,
        minQ=1.0e-3,
        maxQ=0.5,
        qPriorLevel=1.0e-3,
        qPriorTrend=1.0e-3,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        ECM_useProcessPrecisionReweighting=True,
        ECM_useAPN=False,
        processNoiseWarmupECMIters=1,
        returnDiagnostics=True,
    )

    firstPostQIndex = next(
        idx for idx, mode in enumerate(ecmModes) if mode == (True, False)
    )
    assert firstPostQIndex >= 1
    assert all(mode == (False, False) for mode in ecmModes[:firstPostQIndex])
    assert all(mode == (True, False) for mode in ecmModes[firstPostQIndex:])
    assert all(flag is False for flag in ecmLogIterations)
    stateSmoothed, stateCovarSmoothed, *_ = out
    diagnostics = out[-1]
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.all(np.isfinite(stateCovarSmoothed))
    assert diagnostics["process_noise_warmup_fit"]["actual_outer_passes"] >= 1
    assert diagnostics["post_process_noise_fit"]["actual_outer_passes"] >= 1
    assert diagnostics["post_process_noise_fit"]["outer_stop_reason"] in {
        "background_shift_and_nll",
        "background_objective_inner_stable",
        "max_outer_passes",
        "max_outer_passes_inner_ecm_unconverged",
        "max_outer_passes_objective",
        "max_outer_passes_patience",
    }
    assert "outer_nll_change" in diagnostics["post_process_noise_fit"]
    assert "outer_objective_change_per_cell" in diagnostics["post_process_noise_fit"]
    assert diagnostics["post_process_noise_fit"]["outer_patience_target"] == 2
    assert (
        diagnostics["process_noise_calibration"]["processNoisePolicy"]
        == "punc"
    )
    assert diagnostics["process_noise_calibration"]["warmupECMIters"] == pytest.approx(
        1.0
    )
    assert diagnostics["process_noise_calibration"][
        "warmupOuterPasses"
    ] == pytest.approx(float(core.PROCESS_DEFAULT_WARMUP_OUTER_PASSES))
    assert diagnostics["process_noise_calibration"]["preKappaQLevel"] >= (1.0e-3)


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
        ecmModes.append(
            (
                bool(kwargs.get("ECM_useProcessPrecisionReweighting")),
                bool(kwargs.get("ECM_useAPN")),
            )
        )
        return result

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", _spyECM)

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
        ECM_useAPN=False,
        fitBackground=False,
        initialProcessQ=initialQ,
        returnDiagnostics=True,
    )

    diagnostics = out[-1]
    assert ecmModes == [(True, False)]
    assert diagnostics["process_noise_warmup_fit"] is None
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
def _caseRunConsenrichFixedDiagonalSkipsPunc(monkeypatch):
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

    def failSeedEstimator(*_args, **_kwargs):
        raise AssertionError("seed estimator should not run")

    def failPuncEstimator(*_args, **_kwargs):
        raise AssertionError("PUNC estimator should not run")

    monkeypatch.setattr(core, "_estimateInitialProcessNoiseFromData", failSeedEstimator)
    monkeypatch.setattr(core, "_fitPuncProcessNoise", failPuncEstimator)

    initialKappa = np.linspace(0.5, 1.8, n, dtype=np.float32)
    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.2,
        minQ=1.0e-6,
        maxQ=1.0,
        qPriorLevel=2.0e-3,
        qPriorTrend=5.0e-4,
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
        returnPrecisionDiagnostics=True,
        returnDiagnostics=True,
    )

    precisionDiagnostics = out[-2]
    runDiagnostics = out[-1]
    qInfo = runDiagnostics["process_noise_calibration"]
    assert runDiagnostics["process_noise_warmup_fit"] is None
    assert qInfo["processNoisePolicy"] == "fixedDiagonal"
    assert qInfo["processNoiseCalibrationReason"] == "fixed_diagonal"
    assert qInfo["puncStagesActive"] is False
    assert all(qInfo[key] is False for key in core._PUNC_STAGE_TOGGLE_KEYS)
    np.testing.assert_allclose(
        qInfo["matrixQ0Final"],
        np.diag([2.0e-3, 5.0e-4]),
        rtol=0.0,
        atol=1.0e-10,
    )
    outputTracks = precisionDiagnostics["outputTracks"]
    np.testing.assert_allclose(outputTracks["puncQScale"], np.ones(n))
    assert np.any(
        np.abs(outputTracks["effectiveQLevel"] - outputTracks["preKappaQLevel"])
        > 1.0e-9
    )

    levelOut = core.runConsenrich(
        matrixData,
        matrixMunc,
        stateModel="level",
        deltaF=-10.0,
        minQ=1.0e-6,
        maxQ=1.0,
        qPriorLevel=3.0e-3,
        qPriorTrend=7.0e-4,
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
        returnPrecisionDiagnostics=True,
        returnDiagnostics=True,
    )
    levelPrecisionDiagnostics = levelOut[-2]
    levelRunDiagnostics = levelOut[-1]
    levelQInfo = levelRunDiagnostics["process_noise_calibration"]
    np.testing.assert_allclose(levelQInfo["matrixQ0Final"], [[3.0e-3]])
    assert levelQInfo["preKappaQTrend"] == pytest.approx(0.0)
    assert levelQInfo["effectiveTrendLevelRatio"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        levelPrecisionDiagnostics["outputTracks"]["puncQScale"],
        np.ones(n),
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
def test_core_run_consenrich_initial_background_reaches_process_noise_warmup():
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
    warmStart = diagnostics["process_noise_warmup_fit"]["warm_start"]
    assert warmStart["background"] is True
    assert warmStart["background_prepass"] is False
    assert diagnostics["post_process_noise_fit"]["warm_start"]["background"] is True


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
        qPriorLevel=1.0e-3,
        qPriorTrend=1.0e-3,
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
    expected = np.exp(core._bsplineDesign(x, trend.knots, trend.degree) @ trend.beta)
    observed = core.evalPSplineLogVarianceTrend(trend, amplitudes, eps=1.0e-8)

    assert hasattr(cconsenrich, "cEvalPSplineLogVarianceTrend")
    assert np.allclose(observed, expected.astype(np.float32), rtol=1.0e-6)


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
    assert np.allclose(pooled.replicateVarianceFactors, np.ones(residualSd.size))
    assert np.mean(np.abs(np.log(predicted) - np.log(sharedTarget))) < 0.35
    assert len({round(float(value), 4) for value in nu0BySample}) > 1
    assert pooled.diagnostics["predictor"] == "signed_log1p"
    assert pooled.diagnostics["replicate_factor_fit"] == "disabled"


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
def _caseNonnegativeRidgeFailsWhenNNLSSolverFails(monkeypatch):
    def failNNLS(*args, **kwargs):
        raise RuntimeError("iteration limit")

    monkeypatch.setattr(core.optimize, "nnls", failNNLS)

    with pytest.raises(
        RuntimeError,
        match="nonnegative ridge NNLS failed",
    ) as excInfo:
        means = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
        baseline = np.ones(means.size, dtype=np.float64)
        core.fitMuncAdditiveCovariateModel(
            means,
            baseline + 1.0,
            baseline,
            np.ones((means.size, 1), dtype=np.float64),
            np.zeros(means.size, dtype=np.int64),
            sampleCount=1,
            minBlocksPerReplicate=1,
        )

    assert isinstance(excInfo.value.__cause__, RuntimeError)
    assert str(excInfo.value.__cause__) == "iteration limit"


@pytest.mark.correctness
def _caseMuncAdditiveCovariateModelFitsReplicateSpecificExcessAndFallback():
    means0 = np.linspace(-3.0, 3.0, 80, dtype=np.float64)
    means1 = np.linspace(-3.0, 3.0, 80, dtype=np.float64)
    means2 = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
    means = np.concatenate([means0, means1, means2])
    sampleIndex = np.concatenate(
        [
            np.zeros(means0.size, dtype=np.int64),
            np.ones(means1.size, dtype=np.int64),
            np.full(means2.size, 2, dtype=np.int64),
        ]
    )
    baseline = 2.0 + 0.05 * np.abs(means)
    excess = np.where(sampleIndex == 0, 0.5, np.where(sampleIndex == 1, 3.0, 7.0))
    covariates = np.ones((means.size, 1), dtype=np.float32)

    model = core.fitMuncAdditiveCovariateModel(
        means,
        baseline + excess,
        baseline,
        covariates,
        sampleIndex,
        featureNames=("repeat_frac",),
        sampleCount=3,
        basisCount=1,
        ridge=1.0e-8,
        minBlocksPerReplicate=20,
    )

    probeMeans = np.zeros(4, dtype=np.float64)
    probeCovariates = np.ones((probeMeans.size, 1), dtype=np.float32)
    add0 = core.evalMuncAdditiveCovariateModel(model, probeMeans, probeCovariates, 0)
    add1 = core.evalMuncAdditiveCovariateModel(model, probeMeans, probeCovariates, 1)
    add2 = core.evalMuncAdditiveCovariateModel(model, probeMeans, probeCovariates, 2)

    assert model.featureNames == ("repeat_frac",)
    assert model.replicateUsesPooled.tolist() == [False, False, True]
    assert float(np.median(add0)) == pytest.approx(0.5, abs=0.05)
    assert float(np.median(add1)) == pytest.approx(3.0, abs=0.05)
    assert float(np.median(add2)) > float(np.median(add0))
    assert float(np.median(add2)) < 3.0


@pytest.mark.correctness
def _caseMuncAdditiveCovariateModelTreatsMissingCovariatesAsMissing():
    means = np.zeros(6, dtype=np.float64)
    baseline = np.full(means.size, 2.0, dtype=np.float64)
    variances = np.asarray([5.0, 100.0, 5.0, 100.0, 5.0, 100.0], dtype=np.float64)
    covariates = np.asarray([[1.0], [np.nan], [1.0], [np.nan], [1.0], [np.nan]])
    sampleIndex = np.zeros(means.size, dtype=np.int64)

    model = core.fitMuncAdditiveCovariateModel(
        means,
        variances,
        baseline,
        covariates,
        sampleIndex,
        featureNames=("repeat_frac",),
        sampleCount=1,
        basisCount=1,
        ridge=1.0e-8,
        minBlocksPerReplicate=1,
    )

    probeMeans = np.zeros(3, dtype=np.float64)
    probeCovariates = np.asarray([[1.0], [np.nan], [2.0]], dtype=np.float32)
    additional = core.evalMuncAdditiveCovariateModel(
        model,
        probeMeans,
        probeCovariates,
        0,
    )

    assert model.diagnostics["valid_pairs"] == 3
    np.testing.assert_allclose(additional, np.asarray([3.0, 0.0, 6.0]), atol=1.0e-5)


@pytest.mark.correctness
def _caseGetMuncTrackAppliesAdditiveCovariatesBeforeEBShrinkage():
    values = np.zeros(40, dtype=np.float32)
    intervals = np.arange(values.size, dtype=np.uint32) * np.uint32(25)
    trend = core.PSplineLogVarianceTrend(
        knots=np.empty(0, dtype=np.float64),
        degree=-1,
        beta=np.array([np.log(2.0)], dtype=np.float64),
        xMin=0.0,
        xMax=0.0,
        lambdaHat=0.0,
        edf=1.0,
        gcv=0.0,
        lambdaAtBoundary=False,
        finiteCount=1,
        diagnostics={"fallback": "test_constant"},
    )
    model = core.MuncAdditiveCovariateModel(
        featureNames=("gc_dev",),
        basisEdges=np.array([-np.inf, np.inf], dtype=np.float64),
        basisMetadata={"type": "quantile_indicator", "predictor": "signed_log1p"},
        pooledCoefficients=np.array([[3.0]], dtype=np.float64),
        perReplicateCoefficients=np.array(
            [
                [[1.0]],
                [[4.0]],
            ],
            dtype=np.float64,
        ),
        replicateUsesPooled=np.array([False, False], dtype=bool),
        diagnostics={},
    )
    covariates = np.ones((values.size, 1), dtype=np.float32)
    localTrack0 = np.full(values.size, 3.0, dtype=np.float32)
    localTrack1 = np.full(values.size, 6.0, dtype=np.float32)

    track0, _ = core.getMuncTrack(
        "chrTest",
        intervals,
        values,
        25,
        muncTrendBlockSizeBP=100,
        muncLocalWindowSizeBP=100,
        EB_use=True,
        EB_localQuantile=-1.0,
        EB_setNuL=4,
        EB_pooledNu0=1.0e9,
        pooledTrend=trend,
        covariateTrack=covariates,
        additiveCovariateModel=model,
        replicateIndex=0,
        localVarianceTrack=localTrack0,
        varianceFloor=1.0e-6,
    )
    track1, _ = core.getMuncTrack(
        "chrTest",
        intervals,
        values,
        25,
        muncTrendBlockSizeBP=100,
        muncLocalWindowSizeBP=100,
        EB_use=True,
        EB_localQuantile=-1.0,
        EB_setNuL=4,
        EB_pooledNu0=1.0e9,
        pooledTrend=trend,
        covariateTrack=covariates,
        additiveCovariateModel=model,
        replicateIndex=1,
        localVarianceTrack=localTrack1,
        varianceFloor=1.0e-6,
    )

    np.testing.assert_allclose(track0, np.full(values.size, 3.0), rtol=1.0e-5)
    np.testing.assert_allclose(track1, np.full(values.size, 6.0), rtol=1.0e-5)


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
def _caseLinearEnvelopeParameterIsAbsent():
    removed = "EB" + "_minLin"
    assert removed not in core.observationParams._fields

    with pytest.raises(TypeError):
        core.fitPSplineLogVarianceTrend(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([1.0, 1.1, 1.0, 1.2]),
            **{removed: 10.0},
        )


@pytest.mark.correctness
def _caseMonotonePoolingSourceSymbolsAbsent():
    removed = "P" + "AVA"
    sourcePaths = [
        Path(core.__file__),
        Path(core.__file__).parent / "cconsenrich.pyx",
    ]

    for sourcePath in sourcePaths:
        assert removed not in sourcePath.read_text(encoding="utf-8")


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
        ({}, 7.0),
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
def _caseGetMuncTrackRejectsReplicateVarianceFactor():
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(-1.0, 1.0, intervals.size, dtype=np.float32)
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

    with pytest.raises(ValueError, match="replicateVarianceFactor"):
        core.getMuncTrack(
            chromosome="chrTest",
            intervals=intervals,
            values=values,
            intervalSizeBP=25,
            muncTrendBlockSizeBP=125,
            muncLocalWindowSizeBP=150,
            pooledTrend=pooledTrend,
            replicateVarianceFactor=1.5,
        )


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
    fallbackSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        muncTrendBlockSizeBP=None,
        muncLocalWindowSizeBP=None,
    )
    assert fallbackSizing.trendBlockSource == "fallback default"
    assert fallbackSizing.localWindowSource == "fallback default"

    explicitSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        muncTrendBlockSizeBP=250,
        muncLocalWindowSizeBP=500,
    )
    assert explicitSizing.trendBlockSizeBP == 250
    assert explicitSizing.localWindowSizeBP == 500

    dependenceSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceSpanIntervals=17,
        muncTrendBlockSizeBP=None,
        muncLocalWindowSizeBP=None,
        muncTrendBlockDependenceMultiplier=1.5,
        muncLocalWindowDependenceMultiplier=2.5,
    )
    assert dependenceSizing.usedDependenceSpan is True
    assert dependenceSizing.dependenceSpanIntervals == 17
    assert dependenceSizing.trendBlockIntervals == 26
    assert dependenceSizing.localWindowIntervals == 43


@pytest.mark.correctness
def _caseRunConsenrichAPNSmoke():
    rng = np.random.default_rng(123)
    n = 48
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signalTrack = np.sin(grid).astype(np.float32)
    matrixData = np.vstack(
        [
            signalTrack + 0.08 * rng.normal(size=n) - 0.03,
            signalTrack + 0.08 * rng.normal(size=n),
            signalTrack + 0.08 * rng.normal(size=n) + 0.02,
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.15, dtype=np.float32)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
        maxQ=0.5,
        qPriorLevel=1.0e-3,
        qPriorTrend=1.0e-3,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_fixedBackgroundIters=2,
        ECM_outerIters=1,
        ECM_useProcessPrecisionReweighting=True,
        ECM_useAPN=True,
        processNoiseCalibration="seed",
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(NIS))


@pytest.mark.correctness
def _caseRunConsenrichAlwaysRunsECMWithAPN(
    monkeypatch: pytest.MonkeyPatch,
):
    rng = np.random.default_rng(321)
    n = 40
    m = 3
    signalTrack = np.cos(np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32))
    matrixData = np.vstack(
        [
            signalTrack + 0.05 * rng.normal(size=n) - 0.02,
            signalTrack + 0.05 * rng.normal(size=n),
            signalTrack + 0.05 * rng.normal(size=n) + 0.01,
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.2, dtype=np.float32)

    originalECM = cconsenrich.cfixedBackgroundECM
    calls = []

    def _spyECM(*args, **kwargs):
        result = originalECM(*args, **kwargs)
        calls.append(
            (
                bool(kwargs.get("ECM_useAPN")),
                bool(kwargs.get("ECM_useProcessPrecisionReweighting")),
            )
        )
        return result

    monkeypatch.setattr(cconsenrich, "cfixedBackgroundECM", _spyECM)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
        maxQ=0.5,
        qPriorLevel=1.0e-3,
        qPriorTrend=1.0e-3,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_useAPN=True,
        ECM_useProcessPrecisionReweighting=True,
        processNoiseCalibration="seed",
    )

    assert calls
    assert calls[-1] == (True, False)
    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(stateSmoothed))


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
def _caseChooseDependenceSpanSamplesAutosomesAndReportsDiagnostics():
    rng = np.random.default_rng(123)
    n = 1024

    def _matrix(phi: float) -> np.ndarray:
        base = np.empty(n, dtype=np.float64)
        base[0] = rng.normal(scale=0.5)
        for i in range(1, n):
            base[i] = phi * base[i - 1] + rng.normal(scale=0.5)
        return np.vstack(
            [
                base + rng.normal(scale=0.05, size=n),
                base + rng.normal(scale=0.05, size=n),
                base + rng.normal(scale=0.05, size=n),
            ]
        ).astype(np.float32)

    chromosomeNames = ["chr1", "chr2", "chrX", "chrY", "chrM", "chr1_alt"]
    chromosomeMatrices = [
        _matrix(0.80),
        _matrix(0.65),
        _matrix(0.95),
        _matrix(0.95),
        _matrix(0.95),
        _matrix(0.95),
    ]

    pointSpan, lowerSpan, upperSpan, diagnostics = cconsenrich.cchooseDependenceSpan(
        chromosomeNames,
        chromosomeMatrices,
        intervalSizeBP=25,
        numBlocks=100,
        randSeed=77,
    )
    repeat = cconsenrich.cchooseDependenceSpan(
        chromosomeNames,
        chromosomeMatrices,
        intervalSizeBP=25,
        numBlocks=100,
        randSeed=77,
    )

    assert (pointSpan, lowerSpan, upperSpan) == repeat[:3]
    assert diagnostics["sampled_chromosomes"] == repeat[3]["sampled_chromosomes"]
    assert diagnostics["sampled_width_bp"] == repeat[3]["sampled_width_bp"]
    assert diagnostics["num_blocks"] == 100
    assert 3 <= lowerSpan <= pointSpan <= upperSpan
    assert diagnostics["min_span"] == 20
    assert diagnostics["context_size_bp"] == pointSpan * 50 + 1
    assert diagnostics["estimand"] == "acf_abs_consecutive_crossing"
    assert diagnostics["point_threshold"] == pytest.approx(0.10)
    assert diagnostics["acfPointThreshold"] == pytest.approx(
        diagnostics["point_threshold"]
    )
    assert diagnostics["lower_threshold"] == 0.20
    assert diagnostics["upper_threshold"] == 0.10
    assert diagnostics["acf_required_crossings"] == 5
    assert diagnostics["acfRequiredCrossings"] == diagnostics["acf_required_crossings"]
    assert diagnostics["minSpan"] == diagnostics["min_span"]
    assert diagnostics["maxSpan"] == diagnostics["max_span"]
    assert "crossingLag" in diagnostics
    assert diagnostics["pooled_right_censored_fraction"] == pytest.approx(
        diagnostics["right_censored_fraction"]
    )
    assert "right_censored_blocks" in diagnostics
    assert all(5_000 <= width <= 100_000 for width in diagnostics["sampled_width_bp"])
    assert set(diagnostics["sampled_chromosomes"]) <= {"chr1", "chr2"}
    excluded = set(diagnostics["excluded_nonstandard_chromosomes"])
    assert {"chrX", "chrY", "chrM", "chr1_alt"} <= excluded


@pytest.mark.correctness
def _caseChooseDependenceSpanWeightsDenseBlocksAboveSparseBlocks():
    n = 4096
    intervalSizeBP = 25
    params = {
        "intervalSizeBP": intervalSizeBP,
        "numBlocks": 120,
        "randSeed": 202,
        "blockMedianBP": 12_000.0,
        "blockSigma": 0.45,
        "blockMinBP": 4_000,
        "blockMaxBP": 40_000,
        "minContextBP": 500,
        "maxContextBP": 8_000,
        "priorMedianSpan": 18.0,
        "priorLogSd": 1.0,
    }

    def _smoothNoiseTrack(
        rng: np.random.Generator,
        scale: float,
    ) -> np.ndarray:
        raw = rng.normal(scale=scale, size=n + 48)
        kernel = np.exp(-np.arange(49, dtype=np.float64) / 10.0)
        kernel = kernel / math.sqrt(float(np.sum(np.square(kernel))))
        return np.convolve(raw, kernel, mode="valid")[:n]

    def _makeMatrices(includePeaks: bool) -> tuple[list[str], list[np.ndarray]]:
        rng = np.random.default_rng(11)
        grid = np.arange(n, dtype=np.float64)
        chromosomeNames = []
        chromosomeMatrices = []
        for chromIndex in range(2):
            latent = _smoothNoiseTrack(rng, 0.20)
            for peakIndex, center in enumerate(
                [
                    420 + 97 * chromIndex,
                    1250 + 113 * chromIndex,
                    2280 + 71 * chromIndex,
                    3320 + 53 * chromIndex,
                ]
            ):
                width = 26 + 8 * peakIndex + 2 * chromIndex
                amplitude = 3.2 + 0.3 * peakIndex
                if includePeaks:
                    latent += amplitude * np.exp(-np.abs(grid - center) / width)
            replicateTracks = [
                latent + rng.normal(scale=0.08 + 0.01 * rep, size=n)
                for rep in range(3)
            ]
            chromosomeNames.append(f"chr{chromIndex + 1}")
            chromosomeMatrices.append(np.vstack(replicateTracks).astype(np.float32))
        return chromosomeNames, chromosomeMatrices

    shortPoint, _, shortUpper, shortDiagnostics = cconsenrich.cchooseDependenceSpan(
        *_makeMatrices(False),
        **params,
    )
    peakPoint, peakLower, peakUpper, peakDiagnostics = cconsenrich.cchooseDependenceSpan(
        *_makeMatrices(True),
        **params,
    )
    assert shortDiagnostics["sampled_width_bp"] == peakDiagnostics["sampled_width_bp"]
    assert (
        shortDiagnostics["sampled_width_median_bp"]
        == peakDiagnostics["sampled_width_median_bp"]
    )
    assert (
        shortDiagnostics["sampled_chromosomes"]
        == peakDiagnostics["sampled_chromosomes"]
    )
    assert peakDiagnostics["sampled_width_median_bp"] == pytest.approx(11836.0)
    assert peakDiagnostics["blocks_valid"] == params["numBlocks"]
    assert shortDiagnostics["blocks_valid"] == params["numBlocks"]

    def assertDensityReliabilityDiagnostics(diag):
        for key in (
            "sampled_density_reliability",
            "pooled_density_reliability",
            "pooled_density_reliability_relative_weight",
            "block_density_reliability_summary",
            "density_reliability_weighting_used",
            "density_reliability_effective_blocks",
            "acf_evidence_threshold_nats",
            "acf_evidence_snr_threshold",
            "sampled_acf_evidence_nats",
            "sampled_acf_evidence_snr",
            "sampled_acf_evidence_start_lag",
            "pooled_acf_evidence_nats",
            "acf_evidence_summary",
            "acf_evidence_passed_blocks",
            "low_acf_evidence_blocks",
            "density_reliability_effective_blocks_after_acf_gate",
        ):
            assert key in diag

        sampledDensityScores = np.asarray(
            diag["sampled_density_reliability"],
            dtype=np.float64,
        )
        pooledDensityScores = np.asarray(
            diag["pooled_density_reliability"],
            dtype=np.float64,
        )
        pooledPositiveMeans = np.asarray(
            diag["pooled_positive_signal_mean"],
            dtype=np.float64,
        )
        pooledPositiveESS = np.asarray(
            diag["pooled_positive_signal_ess_fraction"],
            dtype=np.float64,
        )
        densityWeights = np.asarray(
            diag["pooled_density_reliability_relative_weight"],
            dtype=np.float64,
        )
        sampledAcfEvidence = np.asarray(
            diag["sampled_acf_evidence_nats"],
            dtype=np.float64,
        )
        sampledAcfSNR = np.asarray(
            diag["sampled_acf_evidence_snr"],
            dtype=np.float64,
        )
        sampledAcfStartLags = np.asarray(
            diag["sampled_acf_evidence_start_lag"],
            dtype=np.int64,
        )
        pooledAcfEvidence = np.asarray(
            diag["pooled_acf_evidence_nats"],
            dtype=np.float64,
        )
        assert len(sampledDensityScores) == len(diag["sampled_width_bp"])
        assert len(diag["sampled_row_index"]) == len(diag["sampled_width_bp"])
        assert len(sampledAcfEvidence) == len(diag["sampled_width_bp"])
        assert len(sampledAcfSNR) == len(diag["sampled_width_bp"])
        assert len(sampledAcfStartLags) == len(diag["sampled_width_bp"])
        assert len(pooledDensityScores) == diag["blocks_valid"]
        assert len(densityWeights) == diag["blocks_valid"]
        assert len(pooledAcfEvidence) == diag["blocks_valid"]
        assert min(sampledDensityScores) >= 0.0
        assert min(pooledDensityScores) >= 0.0
        assert min(densityWeights) >= 0.0
        assert min(sampledAcfEvidence) >= 0.0
        assert min(sampledAcfSNR) >= 0.0
        assert set(diag["sampled_row_index"]) <= {0, 1, 2}
        assert diag["density_reliability_weighting_used"] is True
        assert diag["acf_evidence_threshold_nats"] == pytest.approx(2.0)
        assert diag["acf_evidence_snr_threshold"] > 0.0
        assert diag["acf_evidence_passed_blocks"] == diag["blocks_valid"]
        assert (
            diag["acf_evidence_passed_blocks"] + diag["low_acf_evidence_blocks"]
            == len(diag["sampled_width_bp"])
        )
        assert (
            0.0
            < diag["density_reliability_effective_blocks_after_acf_gate"]
            <= diag["density_reliability_effective_blocks"]
        )
        assert (
            diag["block_weight_score"]
            == "positive_signal_mean_x_sqrt_positive_signal_ess_fraction"
        )
        np.testing.assert_allclose(
            pooledDensityScores,
            pooledPositiveMeans * np.sqrt(pooledPositiveESS),
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        assert diag["block_density_reliability_summary"]["count"] == (
            diag["blocks_valid"]
        )
        assert diag["method"] == "sampled_row_block_spectral_EB"
        assert diag["spectral_pooling"] == "density_reliability_log_periodogram_EB"
        assert diag["spectral_nfft"] >= 2 * diag["max_span"] + 2
        assert 0.0 <= diag["spectral_shrink_median"] <= 1.0
        assert 0.0 < diag["density_reliability_effective_blocks"] <= (
            diag["blocks_valid"]
        )
        assert float(np.mean(densityWeights)) == pytest.approx(1.0)

    for diag in (shortDiagnostics, peakDiagnostics):
        assertDensityReliabilityDiagnostics(diag)

    densityScores = np.asarray(
        peakDiagnostics["pooled_density_reliability"],
        dtype=np.float64,
    )
    densityWeights = np.asarray(
        peakDiagnostics["pooled_density_reliability_relative_weight"],
        dtype=np.float64,
    )
    scoreOrder = np.argsort(densityScores)
    tailCount = max(3, densityScores.size // 4)
    sparseBlocks = scoreOrder[:tailCount]
    denseBlocks = scoreOrder[-tailCount:]
    assert float(np.median(densityScores[denseBlocks])) > float(
        np.median(densityScores[sparseBlocks])
    )
    assert float(np.median(densityWeights[denseBlocks])) > float(
        np.median(densityWeights[sparseBlocks])
    )
    assert densityWeights[int(scoreOrder[-1])] > densityWeights[int(scoreOrder[0])]
    assert peakLower > shortUpper
    assert peakPoint >= 3 * shortPoint
    assert peakDiagnostics["acf_evidence_passed_blocks"] == params["numBlocks"]
    assert peakDiagnostics["low_acf_evidence_blocks"] == 0
    assert peakDiagnostics[
        "density_reliability_effective_blocks_after_acf_gate"
    ] == pytest.approx(peakDiagnostics["density_reliability_effective_blocks"])

    edgeParams = dict(params)
    edgeParams["numBlocks"] = 24
    edgeParams["randSeed"] = 303
    silentMatrix = np.zeros((3, n), dtype=np.float32)
    _, _, _, silentDiagnostics = cconsenrich.cchooseDependenceSpan(
        ["chr1"],
        [silentMatrix],
        **edgeParams,
    )
    assert silentDiagnostics["blocks_valid"] == 0
    assert silentDiagnostics["fallback"] is True
    assert silentDiagnostics["density_reliability_weighting_used"] is False
    assert silentDiagnostics["density_reliability_effective_blocks"] == 0.0
    assert silentDiagnostics["acf_evidence_passed_blocks"] == 0
    assert silentDiagnostics["low_acf_evidence_blocks"] == edgeParams["numBlocks"]
    assert (
        silentDiagnostics["density_reliability_effective_blocks_after_acf_gate"]
        == 0.0
    )
    assert max(silentDiagnostics["sampled_density_reliability"]) == 0.0

    sparseNoiseMatrix = np.zeros((3, n), dtype=np.float32)
    sparseNoisePositions = np.arange(128, n, 512, dtype=np.int64)
    sparseNoiseMatrix[:, sparseNoisePositions] = np.asarray(
        [[1.0], [0.8], [1.2]],
        dtype=np.float32,
    )
    _, _, _, sparseNoiseDiagnostics = cconsenrich.cchooseDependenceSpan(
        ["chr1"],
        [sparseNoiseMatrix],
        **edgeParams,
    )
    assert sparseNoiseDiagnostics["blocks_valid"] == 0
    assert sparseNoiseDiagnostics["fallback"] is True
    assert sparseNoiseDiagnostics["acf_evidence_passed_blocks"] == 0
    assert (
        sparseNoiseDiagnostics["low_acf_evidence_blocks"]
        == edgeParams["numBlocks"]
    )
    assert (
        sparseNoiseDiagnostics[
            "density_reliability_effective_blocks_after_acf_gate"
        ]
        == 0.0
    )

    tinyNames, tinyMatrices = _makeMatrices(False)
    tinyMatrices = [matrix * np.float32(1.0e-10) for matrix in tinyMatrices]
    _, _, _, tinyDiagnostics = cconsenrich.cchooseDependenceSpan(
        tinyNames,
        tinyMatrices,
        **edgeParams,
    )
    assert tinyDiagnostics["blocks_valid"] > 0
    assert tinyDiagnostics["density_reliability_weighting_used"] is True
    assert 0.0 < tinyDiagnostics["density_reliability_effective_blocks"] <= (
        tinyDiagnostics["blocks_valid"]
    )

    ramp = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    rampMatrix = np.vstack([ramp - 0.01, ramp, ramp + 0.01]).astype(np.float32)
    _, _, _, rampDiagnostics = cconsenrich.cchooseDependenceSpan(
        ["chr1"],
        [rampMatrix],
        **edgeParams,
    )
    assert rampDiagnostics["method"] == "sampled_row_block_spectral_EB"
    assert rampDiagnostics["point_span"] >= rampDiagnostics["min_span"]
    assert rampDiagnostics["spectral_frequency_count"] == (
        rampDiagnostics["spectral_nfft"] // 2
    ) + 1


@pytest.mark.correctness
def _caseChooseDependenceSpanHandlesEdgeSpectraAndCrossingRule():
    intervalSizeBP = 25
    periodicParams = {
        "intervalSizeBP": intervalSizeBP,
        "numBlocks": 40,
        "randSeed": 505,
        "blockMedianBP": 6_000.0,
        "blockSigma": 0.20,
        "blockMinBP": 5_000,
        "blockMaxBP": 7_000,
        "minContextBP": 300,
        "maxContextBP": 6_000,
        "priorMedianSpan": 12.0,
        "priorLogSd": 1.0,
    }

    shortMatrix = np.linspace(-1.0, 1.0, 6, dtype=np.float32)[None, :]
    _, _, _, shortDiagnostics = cconsenrich.cchooseDependenceSpan(
        ["chr1"],
        [shortMatrix],
        intervalSizeBP=intervalSizeBP,
        numBlocks=6,
        randSeed=1,
        blockMedianBP=100.0,
        blockSigma=0.10,
        blockMinBP=100,
        blockMaxBP=120,
        minContextBP=500,
        maxContextBP=1_000,
        priorMedianSpan=10.0,
        priorLogSd=1.0,
    )
    assert shortDiagnostics["blocks_valid"] == 0
    assert shortDiagnostics["fallback"] is True
    assert shortDiagnostics["fallback_blocks"] == shortDiagnostics["num_blocks"]
    assert shortDiagnostics["point_span"] == shortDiagnostics["min_span"]

    flatMatrix = np.ones((3, 512), dtype=np.float32)
    _, _, _, flatDiagnostics = cconsenrich.cchooseDependenceSpan(
        ["chr1"],
        [flatMatrix],
        **periodicParams,
    )
    assert flatDiagnostics["blocks_valid"] == 0
    assert flatDiagnostics["fallback"] is True
    assert flatDiagnostics["fallback_blocks"] == periodicParams["numBlocks"]
    assert min(flatDiagnostics["sampled_point_span"]) >= flatDiagnostics["min_span"]
    assert max(flatDiagnostics["sampled_point_span"]) <= flatDiagnostics["max_span"]

    grid = np.arange(512, dtype=np.float64)
    periodicTrack = np.sin(2.0 * np.pi * grid / 12.0)
    periodicMatrix = np.vstack(
        [periodicTrack, periodicTrack + 0.01, periodicTrack - 0.01]
    ).astype(np.float32)
    periodicPoint, periodicLower, periodicUpper, periodicDiagnostics = (
        cconsenrich.cchooseDependenceSpan(
            ["chr1"],
            [periodicMatrix],
            **periodicParams,
        )
    )
    assert periodicDiagnostics["blocks_valid"] == periodicParams["numBlocks"]
    assert periodicDiagnostics["method"] == "sampled_row_block_spectral_EB"
    assert periodicDiagnostics["spectral_acf_first"] > 0.75
    assert periodicPoint >= int(0.80 * periodicDiagnostics["max_span"])
    assert 0.0 <= periodicDiagnostics["spectral_shrink_median"] <= 1.0
    assert periodicLower <= periodicPoint <= periodicUpper
    assert periodicUpper <= periodicDiagnostics["max_span"]

    crossingTrack = np.random.default_rng(1).normal(size=240)
    centered = crossingTrack - float(np.median(crossingTrack))
    scale = 1.4826 * float(np.median(np.abs(centered)))
    trackMedian = float(np.median(crossingTrack))
    lo = max(float(np.quantile(crossingTrack, 0.005)), trackMedian - 8.0 * scale)
    hi = min(float(np.quantile(crossingTrack, 0.995)), trackMedian + 8.0 * scale)
    clipped = np.clip(crossingTrack, lo, hi) - trackMedian
    gamma0 = float(np.mean(np.square(clipped)))
    acf = np.asarray(
        [
            float(
                np.dot(clipped[:-lag], clipped[lag:])
                / (clipped.size - lag)
                / gamma0
            )
            for lag in range(1, 61)
        ],
        dtype=np.float64,
    )
    crossingThreshold = 0.05
    firstSingleCrossing = next(
        lagIndex + 1
        for lagIndex, value in enumerate(acf)
        if abs(value) < crossingThreshold
    )
    firstTripleCrossing = next(
        lagIndex + 1
        for lagIndex in range(acf.size - 2)
        if (
            abs(acf[lagIndex]) < crossingThreshold
            and abs(acf[lagIndex + 1]) < crossingThreshold
            and abs(acf[lagIndex + 2]) < crossingThreshold
        )
    )
    assert firstSingleCrossing == 3
    assert firstTripleCrossing == 12

    crossingMatrix = np.vstack(
        [crossingTrack, crossingTrack + 0.001, crossingTrack - 0.001]
    ).astype(np.float32)
    _, _, _, crossingDiagnostics = cconsenrich.cchooseDependenceSpan(
        ["chr1"],
        [crossingMatrix],
        intervalSizeBP=intervalSizeBP,
        numBlocks=1,
        randSeed=99,
        blockMedianBP=6_000.0,
        blockSigma=0.001,
        blockMinBP=6_000,
        blockMaxBP=6_001,
        minContextBP=100,
        maxContextBP=3_000,
        priorMedianSpan=12.0,
        priorLogSd=1.0,
        acfMinEvidenceNats=0.0,
    )
    assert crossingDiagnostics["sampled_width_bp"] == [6_000]
    assert crossingDiagnostics["sampled_point_span"][0] > firstSingleCrossing
    assert crossingDiagnostics["sampled_point_span"][0] >= firstTripleCrossing
    assert crossingDiagnostics["right_censored_blocks"] == 0
    assert crossingDiagnostics["crossingLag"] is not None
    assert crossingDiagnostics["pooled_right_censored_fraction"] == pytest.approx(0.0)
    _, _, _, singleCrossingDiagnostics = cconsenrich.cchooseDependenceSpan(
        ["chr1"],
        [crossingMatrix],
        intervalSizeBP=intervalSizeBP,
        numBlocks=1,
        randSeed=99,
        blockMedianBP=6_000.0,
        blockSigma=0.001,
        blockMinBP=6_000,
        blockMaxBP=6_001,
        minContextBP=100,
        maxContextBP=3_000,
        priorMedianSpan=12.0,
        priorLogSd=1.0,
        acfMinEvidenceNats=0.0,
        acfRequiredCrossings=1,
    )
    assert singleCrossingDiagnostics["acf_required_crossings"] == 1
    assert (
        singleCrossingDiagnostics["sampled_point_span"][0]
        <= crossingDiagnostics["sampled_point_span"][0]
    )


@pytest.mark.correctness
def _caseChooseDependenceSpanHandlesRowNoiseAndPooledOutliers():
    def smoothNoiseTrack(
        rng: np.random.Generator,
        n: int,
        width: float,
        scale: float,
    ) -> np.ndarray:
        raw = rng.normal(scale=scale, size=n + 128)
        kernel = np.exp(-np.arange(129, dtype=np.float64) / float(width))
        kernel = kernel / math.sqrt(float(np.sum(np.square(kernel))))
        return np.convolve(raw, kernel, mode="valid")[:n]

    n = 2048
    baseTrack = smoothNoiseTrack(np.random.default_rng(7), n, 22.0, 0.20)
    cleanMatrix = np.vstack(
        [
            baseTrack - 0.02,
            baseTrack - 0.01,
            baseTrack,
            baseTrack + 0.01,
            baseTrack + 0.02,
        ]
    ).astype(np.float32)
    rowNoise = 100.0 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    noisyMatrix = cleanMatrix.copy()
    noisyMatrix[4] = rowNoise.astype(np.float32)
    rowNoiseParams = {
        "intervalSizeBP": 25,
        "numBlocks": 80,
        "randSeed": 404,
        "blockMedianBP": 12_000.0,
        "blockSigma": 0.35,
        "blockMinBP": 6_000,
        "blockMaxBP": 24_000,
        "minContextBP": 500,
        "maxContextBP": 6_000,
        "priorMedianSpan": 20.0,
        "priorLogSd": 1.0,
    }

    cleanPoint, cleanLower, cleanUpper, cleanDiagnostics = (
        cconsenrich.cchooseDependenceSpan(["chr1"], [cleanMatrix], **rowNoiseParams)
    )
    noisyPoint, noisyLower, noisyUpper, noisyDiagnostics = (
        cconsenrich.cchooseDependenceSpan(["chr1"], [noisyMatrix], **rowNoiseParams)
    )
    assert abs(noisyPoint - cleanPoint) <= 10
    assert noisyLower <= cleanUpper + 10
    assert cleanLower - noisyUpper <= 10
    assert noisyDiagnostics["sampled_width_bp"] == cleanDiagnostics["sampled_width_bp"]
    assert noisyDiagnostics["sampled_row_index"] == cleanDiagnostics["sampled_row_index"]
    assert noisyDiagnostics["blocks_valid"] >= int(0.90 * rowNoiseParams["numBlocks"])
    noisyDensityWeights = np.asarray(
        noisyDiagnostics["pooled_density_reliability_relative_weight"],
        dtype=np.float64,
    )
    assert np.all(np.isfinite(noisyDensityWeights))
    assert min(noisyDensityWeights) >= 0.0
    assert max(noisyDensityWeights) > 0.0
    assert float(
        np.mean(noisyDensityWeights[noisyDensityWeights > 0.0])
    ) == pytest.approx(
        1.0
    )

    outlierN = 8192
    rng = np.random.default_rng(123)
    grid = np.arange(outlierN, dtype=np.float64)
    latent = smoothNoiseTrack(rng, outlierN, 6.0, 0.15)
    for center in (1800.0, 5500.0):
        latent += 4.0 * np.exp(-0.5 * np.square((grid - center) / 160.0))
    outlierMatrix = np.vstack(
        [latent + rng.normal(scale=0.05, size=outlierN) for _ in range(4)]
    ).astype(np.float32)
    outlierPoint, outlierLower, outlierUpper, outlierDiagnostics = (
        cconsenrich.cchooseDependenceSpan(
            ["chr1"],
            [outlierMatrix],
            intervalSizeBP=25,
            numBlocks=120,
            randSeed=707,
            blockMedianBP=10_000.0,
            blockSigma=0.35,
            blockMinBP=5_000,
            blockMaxBP=18_000,
            minContextBP=500,
            maxContextBP=20_000,
            priorMedianSpan=18.0,
            priorLogSd=1.0,
        )
    )
    sampledSpans = np.asarray(
        outlierDiagnostics["sampled_point_span"],
        dtype=np.float64,
    )
    assert outlierDiagnostics["blocks_valid"] == 120
    assert outlierDiagnostics["density_reliability_effective_blocks"] >= 8.0
    assert outlierDiagnostics["robust_log_span_mad"] > 0.25
    assert 0.0 < outlierDiagnostics["spectral_shrink_median"] < 1.0
    assert outlierPoint <= int(np.quantile(sampledSpans, 0.95)) + 6
    assert outlierLower <= outlierPoint <= outlierUpper


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
def _caseReadSegmentsFragmentsGrouped():
    gzPath = FRAGMENTS_DIR / "small.fragments.tsv.gz"
    groupMapPath = FRAGMENTS_DIR / "barcode_groups.tsv"

    counts = core.readSegments(
        sources=[
            core.inputSource(
                path=str(gzPath),
                sourceKind="FRAGMENTS",
                barcodeGroupMapFile=str(groupMapPath),
                selectGroups=["clusterA"],
                countMode="cutsite",
            ),
            core.inputSource(
                path=str(gzPath),
                sourceKind="FRAGMENTS",
                barcodeGroupMapFile=str(groupMapPath),
                selectGroups=["clusterB"],
                countMode="cutsite",
            ),
        ],
        chromosome="chr1",
        start=0,
        end=40,
        intervalSizeBP=10,
        readLengths=[1, 1],
        scaleFactors=[1.0, 1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=0,
        maxInsertSize=0,
        inferFragmentLength=0,
        minMappingQuality=0,
        minTemplateLength=0,
    )

    assert counts.shape == (2, 4)
    assert np.allclose(counts[0], np.array([2.0, 2.0, 2.0, 4.0], dtype=np.float32))
    assert np.allclose(counts[1], np.array([0.0, 4.0, 2.0, 0.0], dtype=np.float32))


@pytest.mark.correctness
def _caseReadSegmentsFragmentsDefaultToConservedFractionalOverlap():
    gzPath = FRAGMENTS_DIR / "small.fragments.tsv.gz"
    allowListPath = FRAGMENTS_DIR / "allow_BC_A.txt"

    counts = core.readSegments(
        sources=[
            core.inputSource(
                path=str(gzPath),
                sourceKind="FRAGMENTS",
                barcodeAllowListFile=str(allowListPath),
            )
        ],
        chromosome="chr1",
        start=0,
        end=20,
        intervalSizeBP=10,
        readLengths=[1],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=0,
    )

    assert np.allclose(counts[0], np.array([1.0, 1.0], dtype=np.float32))


@pytest.mark.correctness
def _caseReadSegmentsFragmentsRespectModeAndMultiplicity(tmp_path):
    gzPath = FRAGMENTS_DIR / "small.fragments.tsv.gz"
    allowListPath = tmp_path / "allow_AB.txt"
    allowListPath.write_text("BC_A\nBC_B\n", encoding="ascii")

    expectedByMode = {
        "coverage": np.array([2.0, 5.0, 4.0, 3.0], dtype=np.float32),
        "cutsite": np.array([2.0, 6.0, 4.0, 4.0], dtype=np.float32),
        "center": np.array([0.0, 3.0, 4.0, 1.0], dtype=np.float32),
        "midpoint": np.array([0.0, 3.0, 4.0, 1.0], dtype=np.float32),
        constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP: np.array(
            [1.0, 2.5, 3.2777777, 1.2222222],
            dtype=np.float32,
        ),
    }
    expectedNoiseByMode = {
        "coverage": np.array([2.0, 5.0, 4.0, 3.0], dtype=np.float32),
        "cutsite": np.array([2.0, 8.0, 4.0, 6.0], dtype=np.float32),
        "center": np.array([0.0, 3.0, 4.0, 1.0], dtype=np.float32),
        "midpoint": np.array([0.0, 3.0, 4.0, 1.0], dtype=np.float32),
        constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP: np.array(
            [0.5, 1.625, 2.7052469, 1.0246914],
            dtype=np.float32,
        ),
    }

    for countMode, expected in expectedByMode.items():
        result = core.readSegments(
            sources=[
                core.inputSource(
                    path=str(gzPath),
                    sourceKind="FRAGMENTS",
                    barcodeAllowListFile=str(allowListPath),
                    countMode=countMode,
                )
            ],
            chromosome="chr1",
            start=0,
            end=40,
            intervalSizeBP=10,
            readLengths=[1],
            scaleFactors=[1.0],
            oneReadPerBin=0,
            samThreads=1,
            samFlagExclude=0,
            returnRawNoiseMass=True,
        )

        assert np.allclose(result.counts[0], expected), countMode
        assert np.allclose(
            result.rawNoiseMass[0],
            expectedNoiseByMode[countMode],
        ), countMode


@pytest.mark.correctness
def _caseReadSegmentsFragmentsUseDefaultFragmentCountMode(tmp_path):
    gzPath = FRAGMENTS_DIR / "small.fragments.tsv.gz"
    allowListPath = tmp_path / "allow_AB.txt"
    allowListPath.write_text("BC_A\nBC_B\n", encoding="ascii")

    counts = core.readSegments(
        sources=[
            core.inputSource(
                path=str(gzPath),
                sourceKind="FRAGMENTS",
                barcodeAllowListFile=str(allowListPath),
            )
        ],
        chromosome="chr1",
        start=0,
        end=40,
        intervalSizeBP=10,
        readLengths=[1],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=0,
        defaultFragmentCountMode="cutsite",
    )

    assert np.allclose(counts[0], np.array([2.0, 6.0, 4.0, 4.0], dtype=np.float32))


@pytest.mark.correctness
def _caseReadSegmentsFragmentsRejectFFP():
    gzPath = FRAGMENTS_DIR / "small.fragments.tsv.gz"

    for countMode in ("ffp", "ffp-center"):
        with pytest.raises(ValueError, match="ffp.*BAM"):
            core.readSegments(
                sources=[
                    core.inputSource(
                        path=str(gzPath),
                        sourceKind="FRAGMENTS",
                        countMode=countMode,
                    )
                ],
                chromosome="chr1",
                start=0,
                end=40,
                intervalSizeBP=10,
                readLengths=[1],
                scaleFactors=[1.0],
                oneReadPerBin=0,
                samThreads=1,
                samFlagExclude=0,
            )


@pytest.mark.correctness
def _caseCCountsCountBedGraphRegionWeightedByOverlap(tmp_path):
    bedGraphPath = tmp_path / "toy.bedGraph"
    bedGraphPath.write_text(
        "\n".join(
            [
                "track type=bedGraph",
                "chr1 0 25 2.0",
                "chr1 25 75 6.0",
                "chr1 75 100 10.0",
                "chr2 0 100 99.0",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    counts = ccounts.ccounts_countAlignmentRegion(
        str(bedGraphPath),
        "chr1",
        0,
        100,
        50,
        0,
        0,
        1,
        0,
        sourceKind="BEDGRAPH",
    )

    assert counts.dtype == np.float32
    assert np.allclose(counts, np.array([4.0, 8.0], dtype=np.float32))


@pytest.mark.correctness
def _caseCCountsCountIndexedBedGraphRegionWeightedByOverlap(tmp_path):
    pysam = pytest.importorskip("pysam")
    bedGraphPath = tmp_path / "toy.sorted.bedGraph"
    bedGraphPath.write_text(
        "\n".join(
            [
                "chr1\t0\t25\t2.0",
                "chr1\t25\t75\t6.0",
                "chr1\t75\t100\t10.0",
                "chr2\t0\t100\t99.0",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    indexedPath = Path(pysam.tabix_index(str(bedGraphPath), preset="bed", force=True))

    counts = ccounts.ccounts_countAlignmentRegion(
        str(indexedPath),
        "chr1",
        0,
        100,
        50,
        0,
        0,
        1,
        0,
        sourceKind="BEDGRAPH",
    )

    assert (tmp_path / "toy.sorted.bedGraph.gz.tbi").is_file()
    assert np.allclose(counts, np.array([4.0, 8.0], dtype=np.float32))


@pytest.mark.correctness
def _caseReadSegmentsBedGraphScalesNativeCounts(tmp_path):
    bedGraphPath = tmp_path / "toy.bdg"
    bedGraphPath.write_text(
        "chr1\t0\t10\t1.5\n" "chr1\t10\t20\t2.5\n" "chr1\t20\t30\t4.0\n",
        encoding="ascii",
    )

    result = core.readSegments(
        sources=[
            core.inputSource(
                path=str(bedGraphPath),
                sourceKind="BEDGRAPH",
            )
        ],
        chromosome="chr1",
        start=0,
        end=30,
        intervalSizeBP=10,
        readLengths=[0],
        scaleFactors=[2.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=0,
        returnRawNoiseMass=True,
    )

    assert result.counts.shape == (1, 3)
    assert np.allclose(result.counts[0], np.array([3.0, 5.0, 8.0], dtype=np.float32))
    assert np.all(np.isnan(result.rawNoiseMass[0]))


@pytest.mark.correctness
def _caseBedGraphChromRangeAndReadLength(tmp_path):
    bedGraphPath = tmp_path / "range.bedGraph"
    bedGraphPath.write_text(
        "chr2\t0\t10\t9\n" "chr1\t25\t50\t2\n" "chr1\t5\t20\t3\n",
        encoding="ascii",
    )

    assert core.getReadLength(str(bedGraphPath), 10, 100, 1, 0, "BEDGRAPH") == 0
    assert core.getChromRangesJoint(
        [str(bedGraphPath)],
        "chr1",
        chromSize=100,
        samThreads=1,
        samFlagExclude=0,
        sourceKinds=["BEDGRAPH"],
    ) == (5, 50)


@pytest.mark.correctness
def _caseSparseNativeChromRangeFallsBackToWholeChromosome(
    monkeypatch: pytest.MonkeyPatch,
):
    def _fakeChromRange(*args, **kwargs):
        return 1250, 0

    monkeypatch.setattr(
        core.ccounts,
        "ccounts_getAlignmentChromRange",
        _fakeChromRange,
    )

    assert core.getChromRanges(
        "sparse.bam",
        "chrSparse",
        chromLength=5000,
        samThreads=1,
        samFlagExclude=0,
        sourceKind="BAM",
    ) == (0, 5000)


@pytest.mark.correctness
def _caseNormalizeCountModeRejectsNoncanonicalModes():
    for badMode in [
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
            core._normalizeCountMode(badMode, "coverage")


@pytest.mark.correctness
def _caseNormalizeCountModeAcceptsFFP():
    assert core._normalizeCountMode("ffp", "coverage") == "ffp"
    assert core._normalizeCountMode("ffp-center", "coverage") == "ffp-center"
    assert core._nativeCountModeForPreset("ffp-center") == "center"
    assert core._normalizeCountMode("midpoint", "coverage") == "center"
    assert (
        core._normalizeCountMode(
            constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
            "coverage",
        )
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
    assert (
        core._nativeCountModeForPreset(
            constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
        )
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )


@pytest.mark.correctness
def _caseReadSegmentsBamPairedEndUsesTemplateSpan(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "paired.synthetic.bam",
        [
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
        ],
    )

    counts = core.readSegments(
        sources=[core.inputSource(path=str(bamPath), sourceKind="BAM")],
        chromosome="chr1",
        start=0,
        end=300,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        bamInputMode="fragments",
        defaultCountMode="coverage",
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10:18] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def _caseReadSegmentsBamConservedFractionalOverlap(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "fractional-overlap.synthetic.bam",
        [
            {"name": "r1", "start": 5, "flag": 0},
            {"name": "r2", "start": 18, "flag": 0},
        ],
    )

    result = core.readSegments(
        sources=[
            core.inputSource(
                path=str(bamPath),
                sourceKind="BAM",
                countMode=constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
            )
        ],
        chromosome="chr1",
        start=0,
        end=40,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        bamInputMode="reads",
        inferFragmentLength=0,
        returnRawNoiseMass=True,
    )

    expected = np.array([0.25, 0.6, 0.75, 0.4], dtype=np.float32)
    expectedNoise = np.array([0.0625, 0.26, 0.3125, 0.16], dtype=np.float32)
    np.testing.assert_allclose(result.counts[0], expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        result.rawNoiseMass[0],
        expectedNoise,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


@pytest.mark.correctness
def _caseReadSegmentsBamConservedFractionalRejectsOneReadPerBin(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "fractional-overlap-one-read-per-bin.synthetic.bam",
        [{"name": "r1", "start": 5, "flag": 0}],
    )

    with pytest.raises(ValueError, match="oneReadPerBin"):
        core.readSegments(
            sources=[
                core.inputSource(
                    path=str(bamPath),
                    sourceKind="BAM",
                    countMode=constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
                )
            ],
            chromosome="chr1",
            start=0,
            end=40,
            intervalSizeBP=10,
            readLengths=[20],
            scaleFactors=[1.0],
            oneReadPerBin=1,
            samThreads=1,
            samFlagExclude=3844,
            bamInputMode="reads",
            inferFragmentLength=0,
        )


@pytest.mark.correctness
def _caseReadSegmentsPairedBamCanUseRead1OnlySingleEndMode(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "paired-read1.synthetic.bam",
        [
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
        ],
    )

    counts = core.readSegments(
        sources=[core.inputSource(path=str(bamPath), sourceKind="BAM")],
        chromosome="chr1",
        start=0,
        end=300,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        bamInputMode="read1",
        defaultCountMode="coverage",
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10:12] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def _caseReadSegmentsBamPairedEndFFPCountsOnce(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "paired-read1-fiveprime.synthetic.bam",
        [
            {
                "name": "pair-forward-read1",
                "start": 100,
                "flag": 99,
                "next_reference_id": 0,
                "next_start": 160,
                "template_length": 80,
            },
            {
                "name": "pair-forward-read1",
                "start": 160,
                "flag": 147,
                "next_reference_id": 0,
                "next_start": 100,
                "template_length": -80,
            },
            {
                "name": "pair-reverse-read1",
                "start": 300,
                "flag": 163,
                "next_reference_id": 0,
                "next_start": 360,
                "template_length": 80,
            },
            {
                "name": "pair-reverse-read1",
                "start": 360,
                "flag": 83,
                "next_reference_id": 0,
                "next_start": 300,
                "template_length": -80,
            },
        ],
    )

    counts = core.readSegments(
        sources=[
            core.inputSource(
                path=str(bamPath),
                sourceKind="BAM",
                countMode="ffp",
            )
        ],
        chromosome="chr1",
        start=0,
        end=500,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        bamInputMode="fragments",
        inferFragmentLength=0,
    )

    expected = np.zeros(50, dtype=np.float32)
    expected[10] = 1.0
    expected[37] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def _caseReadSegmentsBamFFPCenterPresetUsesRead1EstimatedMidpoint(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "paired-ffp-center.synthetic.bam",
        [
            {
                "name": "pair-forward-read1",
                "start": 100,
                "flag": 99,
                "next_reference_id": 0,
                "next_start": 160,
                "template_length": 80,
            },
            {
                "name": "pair-forward-read1",
                "start": 160,
                "flag": 147,
                "next_reference_id": 0,
                "next_start": 100,
                "template_length": -80,
            },
            {
                "name": "pair-reverse-read1",
                "start": 300,
                "flag": 163,
                "next_reference_id": 0,
                "next_start": 360,
                "template_length": 80,
            },
            {
                "name": "pair-reverse-read1",
                "start": 360,
                "flag": 83,
                "next_reference_id": 0,
                "next_start": 300,
                "template_length": -80,
            },
        ],
    )

    fragmentLengthCalls = []

    def fakeFragmentLength(path, *, samThreads, samFlagExclude, maxInsertSize):
        fragmentLengthCalls.append(
            {
                "path": path,
                "samThreads": samThreads,
                "samFlagExclude": samFlagExclude,
                "maxInsertSize": maxInsertSize,
            }
        )
        return 80

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(core.cconsenrich, "cgetFragmentLength", fakeFragmentLength)
        counts = core.readSegments(
            sources=[
                core.inputSource(
                    path=str(bamPath),
                    sourceKind="BAM",
                    countMode="ffp-center",
                )
            ],
            chromosome="chr1",
            start=0,
            end=500,
            intervalSizeBP=10,
            readLengths=[20],
            scaleFactors=[1.0],
            oneReadPerBin=0,
            samThreads=1,
            samFlagExclude=3844,
            bamInputMode="auto",
            inferFragmentLength=0,
        )
    finally:
        mp.undo()

    expected = np.zeros(50, dtype=np.float32)
    expected[14] = 1.0
    expected[34] = 1.0
    assert np.allclose(counts[0], expected)
    assert fragmentLengthCalls
    assert fragmentLengthCalls[0]["samFlagExclude"] & 128


@pytest.mark.correctness
def _caseReadSegmentsBamFFPCenterRejectsConflictingBamInputMode(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "paired-ffp-center-conflict.synthetic.bam",
        [
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
        ],
    )

    with pytest.raises(ValueError, match="ffp-center.*bamInputMode.*auto.*read1"):
        core.readSegments(
            sources=[
                core.inputSource(
                    path=str(bamPath),
                    sourceKind="BAM",
                    countMode="ffp-center",
                )
            ],
            chromosome="chr1",
            start=0,
            end=300,
            intervalSizeBP=10,
            readLengths=[20],
            scaleFactors=[1.0],
            oneReadPerBin=0,
            samThreads=1,
            samFlagExclude=3844,
            bamInputMode="fragments",
            inferFragmentLength=0,
        )


@pytest.mark.correctness
def _caseReadSegmentsBamExtensionFetchesReadsOutsideRegion(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "extension-edge.synthetic.bam",
        [
            {"name": "forward-left-edge", "start": 80, "flag": 0},
            {"name": "reverse-right-edge", "start": 120, "flag": 16},
        ],
    )

    counts = core.readSegments(
        sources=[core.inputSource(path=str(bamPath), sourceKind="BAM")],
        chromosome="chr1",
        start=100,
        end=120,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        bamInputMode="reads",
        defaultCountMode="coverage",
        extendFrom5pBP=40,
        inferFragmentLength=0,
    )

    assert np.allclose(counts[0], np.array([2.0, 2.0], dtype=np.float32))


@pytest.mark.correctness
def _caseReadSegmentsBamPairedFragmentFetchesRead1OutsideRegion(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "paired-edge.synthetic.bam",
        [
            {
                "name": "pair-edge",
                "start": 80,
                "flag": 99,
                "next_reference_id": 0,
                "next_start": 120,
                "template_length": 60,
            },
            {
                "name": "pair-edge",
                "start": 120,
                "flag": 147,
                "next_reference_id": 0,
                "next_start": 80,
                "template_length": -60,
            },
        ],
    )

    counts = core.readSegments(
        sources=[core.inputSource(path=str(bamPath), sourceKind="BAM")],
        chromosome="chr1",
        start=100,
        end=120,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        bamInputMode="fragments",
        defaultCountMode="coverage",
        maxInsertSize=1000,
        inferFragmentLength=0,
    )

    assert np.allclose(counts[0], np.array([1.0, 1.0], dtype=np.float32))


@pytest.mark.correctness
def _caseReadSegmentsBamCountEndsOnlyUsesFivePrimePositions(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "ends.synthetic.bam",
        [
            {"name": "forward", "start": 100, "flag": 0},
            {"name": "reverse", "start": 160, "flag": 16},
        ],
    )

    counts = core.readSegments(
        sources=[core.inputSource(path=str(bamPath), sourceKind="BAM")],
        chromosome="chr1",
        start=0,
        end=300,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        defaultCountMode="cutsite",
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10] = 1.0
    expected[17] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def _caseReadSegmentsBamFFPMatchesSingleEndFivePrime(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "read1-fiveprime-single-end.synthetic.bam",
        [
            {"name": "forward", "start": 100, "flag": 0},
            {"name": "reverse", "start": 160, "flag": 16},
        ],
    )

    counts = core.readSegments(
        sources=[
            core.inputSource(
                path=str(bamPath),
                sourceKind="BAM",
                countMode="ffp",
            )
        ],
        chromosome="chr1",
        start=0,
        end=300,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        bamInputMode="reads",
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10] = 1.0
    expected[17] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def _caseReadBamSegmentsCountEndsOnlyUsesFivePrimePositions(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "ends.synthetic.bam",
        [
            {"name": "forward", "start": 100, "flag": 0},
            {"name": "reverse", "start": 160, "flag": 16},
        ],
    )

    counts = core.readBamSegments(
        bamFiles=[str(bamPath)],
        chromosome="chr1",
        start=0,
        end=300,
        intervalSizeBP=10,
        readLengths=[20],
        scaleFactors=[1.0],
        oneReadPerBin=0,
        samThreads=1,
        samFlagExclude=3844,
        defaultCountMode="cutsite",
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10] = 1.0
    expected[17] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def _caseFragmentsMappedCountUsesEmittedInsertionsAndSelectedCells():
    gzPath = FRAGMENTS_DIR / "small.fragments.tsv.gz"
    allowListPath = FRAGMENTS_DIR / "allow_BC_A.txt"

    mappedCount, _ = ccounts.ccounts_getAlignmentMappedReadCount(
        str(gzPath),
        sourceKind="FRAGMENTS",
        barcodeAllowListFile=str(allowListPath),
        countMode="cutsite",
    )
    cellCount = ccounts.ccounts_getFragmentCellCount(
        str(gzPath),
        barcodeAllowListFile=str(allowListPath),
    )
    scaleFactor = detrorm.getScaleFactorPerMillion(
        str(gzPath),
        [],
        10,
        sourceKind="FRAGMENTS",
        barcodeAllowListFile=str(allowListPath),
        countMode="cutsite",
        normMethod="RPKM",
        groupCellCount=cellCount,
        fragmentsGroupNorm="CELLS",
    )

    assert mappedCount == 12
    assert cellCount == 1
    assert scaleFactor == pytest.approx((1_000_000 / 12.0) * 100.0)


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


@pytest.mark.correctness
def _caseNormalizationCpmAndRpkmUseDistinctBinLengthScaling(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "single-normalization.synthetic.bam",
        [
            {"name": "r1", "start": 100, "flag": 0},
            {"name": "r2", "start": 200, "flag": 0},
            {"name": "r3", "start": 300, "flag": 0},
            {"name": "r4", "start": 400, "flag": 0},
        ],
    )

    cpmScale = detrorm.getScaleFactorPerMillion(
        str(bamPath),
        [],
        50,
        normMethod="CPM",
        sourceKind="BAM",
        bamInputMode="reads",
    )
    rpkmScale = detrorm.getScaleFactorPerMillion(
        str(bamPath),
        [],
        50,
        normMethod="RPKM",
        sourceKind="BAM",
        bamInputMode="reads",
    )

    assert cpmScale == pytest.approx(250_000.0)
    assert rpkmScale == pytest.approx(5_000_000.0)


@pytest.mark.correctness
def _caseNormalizationEgsPairedBamUsesFragmentSpanDenominator(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "egs-paired-normalization.synthetic.bam",
        _pairedNormalizationRecords(),
    )
    chromSizes = tmp_path / "chrom.sizes"
    chromSizes.write_text("chr1\t1000\n", encoding="ascii")

    scaleFactor = detrorm.getScaleFactor1x(
        str(bamPath),
        1000,
        80,
        [],
        str(chromSizes),
        1,
        sourceKind="BAM",
        bamInputMode="fragments",
        samFlagExclude=3844,
        countReadLength=20,
    )

    assert scaleFactor == pytest.approx(6.25)


@pytest.mark.correctness
def _caseNormalizationEgsConservedFractionalUsesMappedUnitsPerBin(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "egs-conserved-fractional.synthetic.bam",
        _pairedNormalizationRecords(),
    )
    chromSizes = tmp_path / "chrom.sizes"
    chromSizes.write_text("chr1\t1000\n", encoding="ascii")

    scaleFactor = detrorm.getScaleFactor1x(
        str(bamPath),
        1000,
        80,
        [],
        str(chromSizes),
        1,
        sourceKind="BAM",
        bamInputMode="fragments",
        samFlagExclude=3844,
        countMode=constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
        intervalSizeBP=50,
        countReadLength=20,
    )

    assert scaleFactor == pytest.approx(10.0)


@pytest.mark.correctness
def _caseNormalizationCpmPairedBamUsesFragmentDenominator(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "cpm-paired-normalization.synthetic.bam",
        _pairedNormalizationRecords(),
    )

    scaleFactor = detrorm.getScaleFactorPerMillion(
        str(bamPath),
        [],
        50,
        normMethod="CPM",
        sourceKind="BAM",
        bamInputMode="fragments",
        samFlagExclude=3844,
        readLength=20,
    )

    assert scaleFactor == pytest.approx(500_000.0)


@pytest.mark.correctness
def _caseNormalizationCpmPairedRead1UsesRead1Denominator(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "cpm-read1-normalization.synthetic.bam",
        _pairedNormalizationRecords(),
    )

    scaleFactor = detrorm.getScaleFactorPerMillion(
        str(bamPath),
        [],
        50,
        normMethod="CPM",
        sourceKind="BAM",
        bamInputMode="read1",
        samFlagExclude=3844,
    )

    assert scaleFactor == pytest.approx(500_000.0)


@pytest.mark.correctness
def _caseNormalizationCpmPairedFFPUsesOneEventPerPair(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "cpm-paired-read1-fiveprime.synthetic.bam",
        _pairedNormalizationRecords(),
    )

    scaleFactor = detrorm.getScaleFactorPerMillion(
        str(bamPath),
        [],
        50,
        normMethod="CPM",
        sourceKind="BAM",
        bamInputMode="fragments",
        samFlagExclude=3844,
        readLength=20,
        countMode="ffp",
    )

    assert scaleFactor == pytest.approx(500_000.0)


@pytest.mark.correctness
def _caseNormalizationDenominatorMatchesBamFilters(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "filtered-normalization.synthetic.bam",
        [
            {
                "name": "good",
                "start": 100,
                "flag": 99,
                "next_reference_id": 0,
                "next_start": 160,
                "template_length": 80,
            },
            {
                "name": "good",
                "start": 160,
                "flag": 147,
                "next_reference_id": 0,
                "next_start": 100,
                "template_length": -80,
            },
            {
                "name": "low-mapq",
                "start": 250,
                "flag": 99,
                "mapq": 5,
                "next_reference_id": 0,
                "next_start": 310,
                "template_length": 80,
            },
            {
                "name": "low-mapq",
                "start": 310,
                "flag": 147,
                "mapq": 5,
                "next_reference_id": 0,
                "next_start": 250,
                "template_length": -80,
            },
            {
                "name": "dup",
                "start": 400,
                "flag": 1123,
                "next_reference_id": 0,
                "next_start": 460,
                "template_length": 80,
            },
            {
                "name": "dup",
                "start": 460,
                "flag": 1171,
                "next_reference_id": 0,
                "next_start": 400,
                "template_length": -80,
            },
            {
                "name": "short",
                "start": 600,
                "flag": 99,
                "next_reference_id": 0,
                "next_start": 620,
                "template_length": 40,
            },
            {
                "name": "short",
                "start": 620,
                "flag": 147,
                "next_reference_id": 0,
                "next_start": 600,
                "template_length": -40,
            },
        ],
    )

    scaleFactor = detrorm.getScaleFactorPerMillion(
        str(bamPath),
        [],
        50,
        normMethod="CPM",
        sourceKind="BAM",
        bamInputMode="fragments",
        samFlagExclude=3844,
        minMappingQuality=10,
        minTemplateLength=50,
        readLength=20,
    )

    assert scaleFactor == pytest.approx(1_000_000.0)


@pytest.mark.correctness
def _casePairScaleFactorsDownscaleDeeperSampleMacs(monkeypatch):
    depths = {"treatment.bam": 10.0, "control.bam": 4.0}

    def fakeScaleFactor1x(
        bamFile,
        effectiveGenomeSize,
        readLength,
        excludeChroms,
        chromSizesFile,
        samThreads,
        **kwargs,
    ):
        return 1.0 / depths[bamFile]

    monkeypatch.setattr(detrorm, "getScaleFactor1x", fakeScaleFactor1x)

    scaleTreatment, scaleControl = detrorm.getPairScaleFactors(
        "treatment.bam",
        "control.bam",
        1000,
        1000,
        100,
        100,
        [],
        "chrom.sizes",
        1,
        25,
        normMethod="EGS",
        fixControl=False,
    )

    assert scaleTreatment == pytest.approx(0.4)
    assert scaleControl == pytest.approx(1.0)


@pytest.mark.correctness
def _casePairScaleFactorsCanDownscaleControlByDefault(monkeypatch):
    depths = {"treatment.bam": 4.0, "control.bam": 10.0}

    def fakeScaleFactor1x(
        bamFile,
        effectiveGenomeSize,
        readLength,
        excludeChroms,
        chromSizesFile,
        samThreads,
        **kwargs,
    ):
        return 1.0 / depths[bamFile]

    monkeypatch.setattr(detrorm, "getScaleFactor1x", fakeScaleFactor1x)

    scaleTreatment, scaleControl = detrorm.getPairScaleFactors(
        "treatment.bam",
        "control.bam",
        1000,
        1000,
        100,
        100,
        [],
        "chrom.sizes",
        1,
        25,
        normMethod="EGS",
    )

    assert scaleTreatment == pytest.approx(1.0)
    assert scaleControl == pytest.approx(0.4)


@pytest.mark.correctness
def _casePairScaleFactorsFixControlKeepsControlFullDepth(monkeypatch):
    depths = {"treatment.bam": 4.0, "control.bam": 10.0}

    def fakeScaleFactor1x(
        bamFile,
        effectiveGenomeSize,
        readLength,
        excludeChroms,
        chromSizesFile,
        samThreads,
        **kwargs,
    ):
        return 1.0 / depths[bamFile]

    monkeypatch.setattr(detrorm, "getScaleFactor1x", fakeScaleFactor1x)

    scaleTreatment, scaleControl = detrorm.getPairScaleFactors(
        "treatment.bam",
        "control.bam",
        1000,
        1000,
        100,
        100,
        [],
        "chrom.sizes",
        1,
        25,
        normMethod="EGS",
        fixControl=True,
    )

    assert scaleTreatment == pytest.approx(1.0)
    assert scaleControl == pytest.approx(1.0)


@pytest.mark.correctness
def _casePairScaleFactorsCpmUsesPerMillionForFragments(monkeypatch):
    depths = {"treatment.fragments.tsv.gz": 12.0, "control.fragments.tsv.gz": 3.0}
    calls = []

    def fakeScaleFactorPerMillion(
        bamFile,
        excludeChroms,
        intervalSizeBP,
        **kwargs,
    ):
        calls.append((bamFile, kwargs.get("sourceKind")))
        return 1.0 / depths[bamFile]

    def failScaleFactor1x(*args, **kwargs):
        raise AssertionError("CPM paired normalization should not use EGS/RPGC")

    monkeypatch.setattr(detrorm, "getScaleFactorPerMillion", fakeScaleFactorPerMillion)
    monkeypatch.setattr(detrorm, "getScaleFactor1x", failScaleFactor1x)

    scaleTreatment, scaleControl = detrorm.getPairScaleFactors(
        "treatment.fragments.tsv.gz",
        "control.fragments.tsv.gz",
        1000,
        1000,
        1,
        1,
        [],
        "chrom.sizes",
        1,
        25,
        normMethod="CPM",
        sourceKindA="FRAGMENTS",
        sourceKindB="FRAGMENTS",
    )

    assert calls == [
        ("treatment.fragments.tsv.gz", "FRAGMENTS"),
        ("control.fragments.tsv.gz", "FRAGMENTS"),
    ]
    assert scaleTreatment == pytest.approx(0.25)
    assert scaleControl == pytest.approx(1.0)


def _run_with_monkeypatch(monkeypatch, func, *args):
    with monkeypatch.context() as mp:
        return func(*args, mp)


def test_core_numeric_kernel_contracts(contract_case):
    for dtype in (np.float32, np.float64):
        contract_case(
            f"C EMA kernel {dtype}",
            _caseCEMAUsesSameBidirectionalKernelForFloat32AndFloat64,
            dtype,
        )
        contract_case(
            f"mono log kernel {dtype}",
            _caseMonoFuncUsesSameLogKernelForFloat32AndFloat64,
            dtype,
        )
        contract_case(
            f"log-ratio transform kernel {dtype}",
            _caseTransformLogRatioKernelMatchesReferenceForFloat32AndFloat64,
            dtype,
        )
        contract_case(
            f"pure-log transform kernel {dtype}",
            _caseTransformPureLogPathMatchesMonoReferenceForFloat32AndFloat64,
            dtype,
        )
        contract_case(
            f"generic transform modes {dtype}",
            _caseGenericTransformModesMatchReference,
            dtype,
        )
        contract_case(
            f"generic transform differences {dtype}",
            _caseGenericTransformWithInputUsesTransformDifferences,
            dtype,
        )
        contract_case(
            f"Anscombe transform preset {dtype}",
            _caseAnscombePresetUsesCanonicalDefaults,
            dtype,
        )
    for label, func in (
        ("PUNC bridge symbols", _casePuncBridgeSymbolsAreRenamed),
        (
            "PUNC info kernel",
            _casePuncObservationInformationKernelUsesLambdaClampForFloat32AndFloat64,
        ),
        ("PUNC evidence kernel", _casePuncEvidenceKernelAcceptsFullSeedQ),
        ("PUNC scale rebase kernel", _casePuncScaleRebaseKernelUsesCanonicalStateModel),
        (
            "MUNC seed moment kernel",
            _caseMuncObservationMomentSeedPassUsesOmegaMomentsAndFloors,
        ),
        (
            "MUNC EB finalize count floor sentinel",
            _caseFinalizeMuncEBTrackPreservesCountFloorSentinel,
        ),
        (
            "MUNC local evidence windows",
            _caseMuncSmoothDenseLocalEvidenceUsesCenteredWindows,
        ),
        (
            "MUNC ESS local evidence mode",
            _caseEffectiveSampleSizeSupportsMuncEvidenceMode,
        ),
        ("CSF odd median", _caseCSFMedianSelectionHandlesOddLengthDuplicates),
        ("CSF even median", _caseCSFMedianSelectionHandlesEvenLengthDuplicates),
        (
            "generic transform invalid options",
            _caseGenericTransformInvalidOptionsFailGracefully,
        ),
        ("transform input float32", _caseCTransformWithInputReturnsFloat32LogRatio),
        ("transform input float64", _caseCTransformWithInputReturnsFloat64LogRatio),
        ("transform into output", _caseCTransformWithInputIntoWritesOutputInPlace),
        ("in-place pure log float32", _caseCTransformInPlacePureLogMutatesFloat32Array),
        (
            "in-place transform float64",
            _caseCTransformInPlaceMatchesAllocatingTransformForFloat64,
        ),
        (
            "centerMB filters",
            _caseCenterMBAppliesMedianFilterInPlace,
        ),
        (
            "level forward-backward kernel",
            _caseLevelForwardBackwardMatchesPythonReference,
        ),
        (
            "level embedded forward-backward agreement",
            _caseLevelEmbeddedForwardBackwardAgreementWithPrecisionMultipliers,
        ),
        (
            "fixed ECM precision equations",
            _caseCFixedBackgroundPrecisionUpdatesMatchStudentTEquations,
        ),
        (
            "background solve stat reuse",
            _caseBackgroundUpdateReusesStatsAndInitialActiveSet,
        ),
    ):
        contract_case(label, func)


def test_core_state_diagnostics_and_transition_contracts(contract_case):
    for label, func in (
        ("final forward NIS", _caseFinalForwardNISUsesMeanFinalForwardDiagnostic),
        (
            "final forward gain summary",
            _caseFinalForwardGainSummaryUsesReplicateContigRows,
        ),
        (
            "per-interval output diagnostics",
            _casePerIntervalOutputDiagnosticsUseEffectiveNoiseAndGainComponents,
        ),
        (
            "state roughness summary",
            _caseSummarizeStateRoughnessUsesHoldoutBlocksAndSignalStrata,
        ),
        (
            "precision boundary summary",
            _caseSummarizePrecisionBoundaryHitsSkipsFirstProcessWeight,
        ),
        (
            "removed process block scale options",
            _caseFitParamsDropsProcBlockScaleOptions,
        ),
        (
            "state model normalization",
            _caseNormalizeStateModelAcceptsCanonicalValuesOnly,
        ),
        (
            "transition residual orientation",
            _caseExpectedTransitionResidualSumsUsesLagOrientationAndDeltaF,
        ),
        (
            "transition residual reference",
            _caseExpectedTransitionResidualSumsMatchesPythonReference,
        ),
        (
            "level transition residual reference",
            _caseExpectedLevelTransitionResidualSumsMatchesPythonReference,
        ),
        (
            "PUNC process noise clamp decomposition",
            _casePuncProcessNoiseCalibrationRebasesClampedBaseQ,
        ),
        (
            "PUNC prior df MoM recovers F dispersion",
            _casePuncPriorDfMethodOfMomentsRecoversKnownFDispersion,
        ),
        (
            "PUNC prior df MoM degenerate inputs",
            _casePuncPriorDfMethodOfMomentsHandlesDegenerateInputs,
        ),
        (
            "robust Q seed recovers random walk scale",
            _caseInitialProcessNoiseSeedRecoversRandomWalkScale,
        ),
        (
            "robust Q seed caps low MUNC artifact",
            _caseInitialProcessNoiseSeedCapsDominantLowMuncArtifact,
        ),
        (
            "robust Q seed pooled EB fallback",
            _caseInitialProcessNoiseSeedFallsBackToPooledEbForSparseOverlap,
        ),
        (
            "MUNC seed Q reference match",
            _caseMuncSeedQKernelsMatchReference,
        ),
        (
            "PUNC deadband prior shrinks near null",
            _casePuncDeadbandPriorShrinksNearNullPriorScale,
        ),
        (
            "PUNC deadband prior negligible outside deadband",
            _casePuncDeadbandPriorNegligibleOutsideDeadband,
        ),
        (
            "state uncertainty coverage",
            _caseCheckStateUncertaintyCoverageOverallAndStrata,
        ),
        ("linear envelope removed", _caseLinearEnvelopeParameterIsAbsent),
        ("monotone pooling removed", _caseMonotonePoolingSourceSymbolsAbsent),
    ):
        contract_case(label, func)


def test_core_punc_window_weight_contracts(monkeypatch, contract_case):
    contract_case(
        "PUNC reliability window switch",
        _run_with_monkeypatch,
        monkeypatch,
        _casePuncReliabilityWeightedWindowsSwitchesLocalEvidence,
    )


@pytest.mark.parametrize("toggleName", core._PUNC_STAGE_TOGGLE_KEYS)
def test_core_punc_stage_toggles_report_diagnostics(toggleName):
    intervalCount = 14
    level = np.linspace(-0.5, 0.6, intervalCount, dtype=np.float64)
    trend = np.gradient(level)
    state = np.column_stack([level, trend]).astype(np.float32)
    stateCov = np.zeros((intervalCount, 2, 2), dtype=np.float32)
    stateCov[:, 0, 0] = 0.05
    stateCov[:, 1, 1] = 0.02
    lagCov = np.zeros((intervalCount, 2, 2), dtype=np.float32)
    lagCov[:-1, 0, 0] = 0.01
    lagCov[:-1, 1, 1] = 0.005
    warmupFit = {
        "stateSmoothed": state,
        "stateCovarSmoothed": stateCov,
        "lagCovSmoothed": lagCov,
        "matrixMunc": np.full((2, intervalCount), 0.1, dtype=np.float32),
        "lambdaExp": np.ones(intervalCount, dtype=np.float32),
    }
    toggles = {key: True for key in core._PUNC_STAGE_TOGGLE_KEYS}
    toggles[toggleName] = False

    _matrixQ, processQScale, info = core._fitPuncProcessNoise(
        warmupFit=warmupFit,
        matrixMunc=warmupFit["matrixMunc"],
        matrixF=core.constructMatrixF(0.2),
        seedQ=np.diag([2.0e-3, 5.0e-4]).astype(np.float32),
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        pad=1.0e-4,
        minQ=1.0e-6,
        maxQ=1.0,
        blockLenIntervals=4,
        processCovariates=None,
        puncLocalWindowMultiplier=1.0,
        puncDependenceMultiplier=1.0,
        puncMinScale=0.25,
        puncMaxScale=4.0,
        puncMinWindowWeight=0.0,
        puncPriorDf=6.0,
        puncPriorRidge=1.0e-3,
        puncLevelBufferZ=0.0,
        puncUseReliabilityWeightedWindows=True,
        observationPrecisionMultiplierMin=0.25,
        observationPrecisionMultiplierMax=4.0,
        **toggles,
    )

    assert info["puncStagesActive"] is True
    assert info[toggleName] is False
    if toggleName == "puncUsePriorDfMoments":
        assert info["puncPriorDf"] == pytest.approx(6.0)
        assert info["puncPriorDfSource"] == "configured"
        assert info["puncPriorDfMomentReason"] == "disabled"
    if toggleName == "puncUseGlobalScale":
        assert info["globalScale"] == pytest.approx(1.0)
    assert processQScale.shape == (intervalCount,)


def test_core_em_loop_contracts(monkeypatch, contract_case):
    contract_case(
        "outer pass smoke",
        _caseRunConsenrichOuterPassSmoke,
    )
    contract_case(
        "flat process-noise initializer",
        _caseRunConsenrichFlatWarmupInitializerUsesMinQ,
    )
    for label, func in (
        (
            "process noise warmup",
            _caseRunConsenrichProcessNoiseWarmupRestoresFinalReweighting,
        ),
        (
            "initial process noise skips warmup",
            _caseRunConsenrichInitialProcessQSkipsWarmup,
        ),
        (
            "fixed diagonal process noise skips PUNC",
            _caseRunConsenrichFixedDiagonalSkipsPunc,
        ),
        (
            "outer-pass minimum iterations",
            _caseRunConsenrichOuterPassRequiresThreeIterationsDespiteTolerance,
        ),
        ("ECM always runs with APN", _caseRunConsenrichAlwaysRunsECMWithAPN),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)
    contract_case("APN smoke", _caseRunConsenrichAPNSmoke)
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
        (
            "EB prior strength thinning",
            _caseEBPriorStrengthUsesThinnedVariancePairs,
        ),
        (
            "MUNC sparse local variance rejection",
            _caseGetMuncTrackRejectsSparseLocalVariancePaths,
        ),
        (
            "MUNC additive covariate model",
            _caseMuncAdditiveCovariateModelFitsReplicateSpecificExcessAndFallback,
        ),
        (
            "MUNC missing additive covariates",
            _caseMuncAdditiveCovariateModelTreatsMissingCovariatesAsMissing,
        ),
        (
            "MUNC additive covariates before EB",
            _caseGetMuncTrackAppliesAdditiveCovariatesBeforeEBShrinkage,
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
        "nonnegative ridge NNLS failure",
        _caseNonnegativeRidgeFailsWhenNNLSSolverFails,
        monkeypatch,
    )
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
        "MUNC rejects replicate factor",
        _caseGetMuncTrackRejectsReplicateVarianceFactor,
    )


def test_core_dependence_selection_contracts(contract_case):
    contract_case(
        "feature length bootstrap/context compatibility",
        _caseChooseFeatureLengthBootstrapWidthVarianceAndContextCompat,
    )
    contract_case(
        "sampled correlation length autosome diagnostics",
        _caseChooseDependenceSpanSamplesAutosomesAndReportsDiagnostics,
    )
    contract_case(
        "density reliability correlation length weighting",
        _caseChooseDependenceSpanWeightsDenseBlocksAboveSparseBlocks,
    )
    contract_case(
        "correlation length edge spectra and crossing rule",
        _caseChooseDependenceSpanHandlesEdgeSpectraAndCrossingRule,
    )
    contract_case(
        "correlation length row noise and pooled outliers",
        _caseChooseDependenceSpanHandlesRowNoiseAndPooledOutliers,
    )
    contract_case(
        "chrom sizes preserve sex chromosomes",
        _caseChromSizesKeepSexChromosomesByDefault,
    )


def test_core_fragments_io_contracts(tmp_path, contract_case):
    for label, func, args in (
        ("fragments grouped", _caseReadSegmentsFragmentsGrouped, ()),
        (
            "fragments default conserved fractional overlap",
            _caseReadSegmentsFragmentsDefaultToConservedFractionalOverlap,
            (),
        ),
        (
            "fragments mode and multiplicity",
            _caseReadSegmentsFragmentsRespectModeAndMultiplicity,
            (tmp_path,),
        ),
        (
            "fragments default count mode",
            _caseReadSegmentsFragmentsUseDefaultFragmentCountMode,
            (tmp_path,),
        ),
        ("fragments reject ffp", _caseReadSegmentsFragmentsRejectFFP, ()),
        (
            "fragments mapped count",
            _caseFragmentsMappedCountUsesEmittedInsertionsAndSelectedCells,
            (),
        ),
    ):
        contract_case(label, func, *args)


def test_core_bedgraph_counting_contracts(tmp_path, contract_case):
    for label, func in (
        ("bedGraph weighted overlap", _caseCCountsCountBedGraphRegionWeightedByOverlap),
        (
            "indexed bedGraph weighted overlap",
            _caseCCountsCountIndexedBedGraphRegionWeightedByOverlap,
        ),
        ("bedGraph native scaling", _caseReadSegmentsBedGraphScalesNativeCounts),
        ("bedGraph range and read length", _caseBedGraphChromRangeAndReadLength),
    ):
        contract_case(label, func, tmp_path)


def test_core_bam_counting_contracts(tmp_path, monkeypatch, contract_case):
    contract_case(
        "sparse native chromosome range fallback",
        _run_with_monkeypatch,
        monkeypatch,
        _caseSparseNativeChromRangeFallsBackToWholeChromosome,
    )
    contract_case(
        "count mode noncanonical modes rejected",
        _caseNormalizeCountModeRejectsNoncanonicalModes,
    )
    contract_case("ffp count mode accepted", _caseNormalizeCountModeAcceptsFFP)
    for label, func in (
        ("paired BAM template span", _caseReadSegmentsBamPairedEndUsesTemplateSpan),
        (
            "paired BAM read1 single-end mode",
            _caseReadSegmentsPairedBamCanUseRead1OnlySingleEndMode,
        ),
        (
            "BAM conserved fractional overlap",
            _caseReadSegmentsBamConservedFractionalOverlap,
        ),
        (
            "BAM conserved fractional oneReadPerBin rejected",
            _caseReadSegmentsBamConservedFractionalRejectsOneReadPerBin,
        ),
        ("paired BAM ffp count mode", _caseReadSegmentsBamPairedEndFFPCountsOnce),
        (
            "paired BAM ffp-center preset",
            _caseReadSegmentsBamFFPCenterPresetUsesRead1EstimatedMidpoint,
        ),
        (
            "paired BAM ffp-center bamInputMode conflict",
            _caseReadSegmentsBamFFPCenterRejectsConflictingBamInputMode,
        ),
        (
            "BAM extension fetches reads outside region",
            _caseReadSegmentsBamExtensionFetchesReadsOutsideRegion,
        ),
        (
            "paired BAM fragment fetches read1 outside region",
            _caseReadSegmentsBamPairedFragmentFetchesRead1OutsideRegion,
        ),
        (
            "BAM count-ends 5-prime positions",
            _caseReadSegmentsBamCountEndsOnlyUsesFivePrimePositions,
        ),
        ("BAM ffp single-end parity", _caseReadSegmentsBamFFPMatchesSingleEndFivePrime),
        (
            "direct BAM count-ends 5-prime positions",
            _caseReadBamSegmentsCountEndsOnlyUsesFivePrimePositions,
        ),
    ):
        contract_case(label, func, tmp_path)


def test_core_normalization_scale_factor_contracts(tmp_path, contract_case):
    for label, func in (
        (
            "CPM and RPKM bin-length scaling",
            _caseNormalizationCpmAndRpkmUseDistinctBinLengthScaling,
        ),
        (
            "EGS paired BAM fragment span denominator",
            _caseNormalizationEgsPairedBamUsesFragmentSpanDenominator,
        ),
        (
            "EGS conserved fractional mapped units per bin",
            _caseNormalizationEgsConservedFractionalUsesMappedUnitsPerBin,
        ),
        (
            "CPM paired BAM fragment denominator",
            _caseNormalizationCpmPairedBamUsesFragmentDenominator,
        ),
        (
            "CPM paired BAM read1 denominator",
            _caseNormalizationCpmPairedRead1UsesRead1Denominator,
        ),
        (
            "CPM paired BAM ffp denominator",
            _caseNormalizationCpmPairedFFPUsesOneEventPerPair,
        ),
        (
            "normalization denominator BAM filters",
            _caseNormalizationDenominatorMatchesBamFilters,
        ),
    ):
        contract_case(label, func, tmp_path)


def test_core_pair_scale_factor_contracts(monkeypatch, contract_case):
    for label, func in (
        ("MACS treatment downscale", _casePairScaleFactorsDownscaleDeeperSampleMacs),
        (
            "control downscale by default",
            _casePairScaleFactorsCanDownscaleControlByDefault,
        ),
        (
            "fixed control keeps full depth",
            _casePairScaleFactorsFixControlKeepsControlFullDepth,
        ),
        (
            "CPM fragments use per-million scaling",
            _casePairScaleFactorsCpmUsesPerMillionForFragments,
        ),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)
