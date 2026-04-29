# -*- coding: utf-8 -*-

# the name of this file is unfortunately misleading --
# 'core' as in 'central' or 'basic', not the literal
# `consenrich.core` module

import math
import logging
import os
import re
import tempfile
from typing import Tuple, List, Optional
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
import scipy.stats as stats
import scipy.signal as spySig  # renamed to avoid conflict with any `signal` variables

import consenrich.core as core
import consenrich.ccounts as ccounts
import consenrich.cconsenrich as cconsenrich
import consenrich.detrorm as detrorm
import consenrich.misc_util as misc_util
import consenrich.peaks as peaks


TESTS_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TESTS_DIR / "data"
FRAGMENTS_DIR = TEST_DATA_DIR / "fragments"


@pytest.mark.correctness
def testDenseMeanBlockShrinkageIsConservative():
    rng = np.random.default_rng(7)
    x = np.concatenate(
        [
            rng.normal(1.0, 0.05, 1800),
            rng.normal(6.0, 0.15, 200),
        ]
    ).astype(np.float32)

    global_only = cconsenrich.cDenseMean(
        x,
        blockLenTarget=-1,
        itersEM=200,
    )
    block_shrunk = cconsenrich.cDenseMean(
        x,
        blockLenTarget=100,
        itersEM=200,
    )

    assert block_shrunk <= global_only + 1e-6
    assert block_shrunk < global_only - 0.25


@pytest.mark.correctness
def testDenseMeanBlockMeanElbowKeepsBackgroundEstimate():
    rng = np.random.default_rng(8)
    block_len = 12
    blocks = [
        rng.normal(1.0, 0.02, block_len)
        for _ in range(240)
    ] + [
        rng.normal(8.0, 0.02, block_len)
        for _ in range(80)
    ]
    x = np.concatenate(blocks).astype(np.float32)

    global_only = cconsenrich.cDenseMean(
        x,
        blockLenTarget=-1,
        itersEM=100,
    )
    block_shrunk = cconsenrich.cDenseMean(
        x,
        blockLenTarget=block_len,
        itersEM=100,
    )

    assert global_only > 7.0
    assert block_shrunk < 2.0
    assert block_shrunk < global_only - 5.0


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
def testSingleEndDetection():
    # case: single-end BAM
    bamFiles = [str(TESTS_DIR / "smallTest.bam")]
    pairedEndStatus = misc_util.alignmentFilesArePairedEnd(bamFiles, maxReads=1_000)
    assert isinstance(pairedEndStatus, list)
    assert len(pairedEndStatus) == 1
    assert isinstance(pairedEndStatus[0], bool)
    assert pairedEndStatus[0] is False


@pytest.mark.correctness
def testPairedEndDetection():
    # case: paired-end BAM
    bamFiles = [str(TESTS_DIR / "smallTest2.bam")]
    pairedEndStatus = misc_util.alignmentFilesArePairedEnd(bamFiles, maxReads=1_000)
    assert isinstance(pairedEndStatus, list)
    assert len(pairedEndStatus) == 1
    assert isinstance(pairedEndStatus[0], bool)
    assert pairedEndStatus[0] is True


@pytest.mark.peaks
def testMatchExistingBedGraph():
    np.random.seed(42)
    with tempfile.TemporaryDirectory() as tempFolder:
        stateBedGraphPath = Path(tempFolder) / "toy_state.bedGraph"
        uncertaintyBedGraphPath = Path(tempFolder) / "toy_uncertainty.bedGraph"
        fakeVals = []
        for i in range(1000):
            if (i % 100) <= 10:
                # add in about ~10~ peak-like regions
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
            tau0=1.0,
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
def testZeroCenteredBackgroundUpdate():
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    residualMatrix = np.vstack(
        [
            0.6 * np.sin(np.linspace(0.0, np.pi, 64, dtype=np.float32)) + 0.1 * x,
            0.6 * np.sin(np.linspace(0.0, np.pi, 64, dtype=np.float32)) - 0.1 * x,
        ]
    ).astype(np.float32)
    invVarMatrix = np.ones_like(residualMatrix, dtype=np.float32)

    background = core._solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )

    assert background.shape == (64,)
    assert np.isfinite(background).all()
    assert abs(float(np.mean(background))) < 1.0e-5
    assert float(np.std(background)) > 0.0


@pytest.mark.correctness
def testFitParamsDropsProcBlockScaleOptions():
    removedFields = {
        "EM_scaleToMedian",
        "EM_alphaEMA",
        "EM_scaleLOW",
        "EM_scaleHIGH",
        "EM_useProcBlockScale",
        "EM_useReplicateScale",
        "EM_repScaleLOW",
        "EM_repScaleHIGH",
    }
    assert removedFields.isdisjoint(core.fitParams._fields)


@pytest.mark.correctness
def testRunConsenrichOuterEMSmoke():
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

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
        maxQ=1.0,
        offDiagQ=0.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        autoDeltaF=False,
        EM_maxIters=3,
        EM_outerIters=2,
        applyJackknife=False,
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, qScale, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.allclose(qScale, 1.0)


@pytest.mark.correctness
def testBartlettEffectiveInfoCorrectionIID():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(4, 512)).astype(np.float32)

    diag = core._bartlettEffectiveInfoCorrection(z, bandwidth=16)

    assert diag["numSeries"] == 4
    assert diag["bandwidth"] == 16
    assert diag["gamma0"] > 0.0
    assert diag["correctionFactor"] >= 1.0
    assert diag["correctionFactor"] < 1.5
    assert diag["effectiveInfoFraction"] <= 1.0
    assert diag["effectiveInfoFraction"] > 0.65


@pytest.mark.correctness
def testBartlettEffectiveInfoCorrectionAR1PositiveAutocorrelation():
    rng = np.random.default_rng(8)
    m = 4
    n = 512
    phi = 0.85
    z = np.zeros((m, n), dtype=np.float64)
    innovations = rng.normal(size=(m, n))
    for j in range(m):
        for i in range(1, n):
            z[j, i] = phi * z[j, i - 1] + innovations[j, i]

    diag = core._bartlettEffectiveInfoCorrection(z.astype(np.float32), bandwidth=16)

    assert diag["numSeries"] == m
    assert diag["gamma0"] > 0.0
    assert diag["correctionFactor"] > 1.5
    assert diag["effectiveInfoFraction"] < 0.7


@pytest.mark.correctness
def testShrunkenBlockEffectiveInfoCorrectionTracksLocalHAC():
    rng = np.random.default_rng(9)
    n = 128
    z = np.zeros((1, n), dtype=np.float64)
    z[:, :64] = rng.normal(scale=0.4, size=(1, 64))
    z[:, 64:] = rng.normal(scale=2.0, size=(1, 64))

    diag = core._shrunkenBlockEffectiveInfoCorrection(
        z,
        bandwidth=4,
        blockLengthIntervals=64,
    )
    factors = np.asarray(diag["intervalFactors"], dtype=np.float64)

    assert diag["numBlocks"] == 2
    assert factors.shape == (n,)
    assert np.all(factors >= 1.0)
    assert float(np.median(factors[64:])) > float(np.median(factors[:64]))
    assert diag["blockFactorMax"] >= diag["blockFactorMin"]


@pytest.mark.correctness
def testRunConsenrichEffectiveInfoRescaleInflatesDefaultUncertainty(
    caplog: pytest.LogCaptureFixture,
):
    rng = np.random.default_rng(11)
    n = 128
    m = 3
    phi = 0.8
    targetVar = 0.2
    innovSD = np.sqrt(targetVar * (1.0 - phi * phi))
    matrixData = np.zeros((m, n), dtype=np.float32)
    for j in range(m):
        noise = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + innovSD * rng.normal()
        matrixData[j, :] = noise.astype(np.float32)
    matrixMunc = np.full((m, n), targetVar, dtype=np.float32)

    with caplog.at_level(logging.INFO):
        outNoCorrection = core.runConsenrich(
            matrixData,
            matrixMunc,
            deltaF=0.1,
            minQ=1.0e-3,
            maxQ=0.5,
            offDiagQ=0.0,
            stateInit=0.0,
            stateCovarInit=1.0,
            boundState=False,
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            blockLenIntervals=17,
            autoDeltaF=False,
            EM_maxIters=2,
            EM_outerIters=1,
            effectiveInfoRescale=False,
            applyJackknife=False,
        )
        outCorrected = core.runConsenrich(
            matrixData,
            matrixMunc,
            deltaF=0.1,
            minQ=1.0e-3,
            maxQ=0.5,
            offDiagQ=0.0,
            stateInit=0.0,
            stateCovarInit=1.0,
            boundState=False,
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            blockLenIntervals=17,
            autoDeltaF=False,
            EM_maxIters=2,
            EM_outerIters=1,
            effectiveInfoRescale=True,
            effectiveInfoBandwidthIntervals=8,
            applyJackknife=False,
        )
        outDefault = core.runConsenrich(
            matrixData,
            matrixMunc,
            deltaF=0.1,
            minQ=1.0e-3,
            maxQ=0.5,
            offDiagQ=0.0,
            stateInit=0.0,
            stateCovarInit=1.0,
            boundState=False,
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            blockLenIntervals=17,
            autoDeltaF=False,
            EM_maxIters=2,
            EM_outerIters=1,
            effectiveInfoBandwidthIntervals=8,
            applyJackknife=False,
        )

    _, covNoCorrection, *_ = outNoCorrection
    _, covCorrected, *_ = outCorrected
    _, covDefault, *_ = outDefault
    ratio = covCorrected[:, 0, 0] / np.maximum(covNoCorrection[:, 0, 0], 1.0e-12)

    assert np.all(np.isfinite(ratio))
    assert np.all(ratio >= 1.0)
    assert float(np.median(ratio)) > 1.0
    assert np.allclose(covDefault, covCorrected)
    assert any(
        "Effective-information uncertainty correction applied" in rec.message
        for rec in caplog.records
    )


@pytest.mark.correctness
def testGetMuncTrackSparseNearestPath(monkeypatch: pytest.MonkeyPatch):
    intervals = np.arange(0, 400, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.6, intervals.size, dtype=np.float32)
    sparseIntervalIndices = np.array([1, 5, 9, 13], dtype=np.intp)
    fakeVarTrack = np.linspace(0.2, 0.5, intervals.size, dtype=np.float32)
    fakeBlockMeans = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    fakeBlockVars = np.linspace(0.2, 0.6, 8, dtype=np.float32)

    def _fakeSparseNearest(*args, **kwargs):
        return (
            np.zeros(intervals.size, dtype=np.float32),
            fakeVarTrack.copy(),
        )

    def _forbidRolling(*args, **kwargs):
        raise AssertionError("rolling AR1 path should not be used")

    def _fakeMeanVarPairs(*args, **kwargs):
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    def _fakeFitVarianceFunction(*args, **kwargs):
        return {"slope": 1.0}

    def _fakeEvalVarianceFunction(opt, meanTrack, EB_minLin=1.0):
        meanTrackArr = np.asarray(meanTrack, dtype=np.float32)
        return np.maximum(0.25, meanTrackArr + 0.1).astype(np.float32)

    monkeypatch.setattr(
        cconsenrich,
        "cSparseNearestMeanVarTrack",
        _fakeSparseNearest,
    )
    monkeypatch.setattr(
        cconsenrich,
        "crolling_AR1_IVar",
        _forbidRolling,
    )
    monkeypatch.setattr(
        cconsenrich,
        "cmeanVarPairs",
        _fakeMeanVarPairs,
    )
    monkeypatch.setattr(core, "fitVarianceFunction", _fakeFitVarianceFunction)
    monkeypatch.setattr(core, "evalVarianceFunction", _fakeEvalVarianceFunction)

    muncTrack, support = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        samplingBlockSizeBP=125,
        samplingIters=64,
        sparseIntervalIndices=sparseIntervalIndices,
        numNearest=3,
        EB_use=True,
    )

    assert muncTrack.shape == values.shape
    assert np.isfinite(muncTrack).all()
    assert support > 0.0


@pytest.mark.correctness
def testGetMuncTrackSparseNearestDetrendsPriorBySignedLocalIntercept(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 400, 25, dtype=np.uint32)
    values = np.linspace(-0.4, 1.1, intervals.size, dtype=np.float32)
    sparseIntervalIndices = np.array([1, 5, 9, 13], dtype=np.intp)
    sparseMeanTrack = np.linspace(-0.2, 0.3, intervals.size, dtype=np.float32)
    sparseVarTrack = np.linspace(0.2, 0.5, intervals.size, dtype=np.float32)
    fakeBlockMeans = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    fakeBlockVars = np.linspace(0.2, 0.6, 8, dtype=np.float32)
    seen: dict[str, np.ndarray] = {}

    def _fakeSparseNearest(*args, **kwargs):
        return sparseMeanTrack.copy(), sparseVarTrack.copy()

    def _forbidRolling(*args, **kwargs):
        raise AssertionError("rolling AR1 path should not be used")

    def _fakeMeanVarPairs(intervalsArg, valuesArg, *args, **kwargs):
        seen["prior_fit_values"] = np.asarray(valuesArg).copy()
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    def _fakeEMA(meanTrackArg, alpha):
        seen["ema_input"] = np.asarray(meanTrackArg).copy()
        return np.asarray(meanTrackArg).copy()

    def _fakeFitVarianceFunction(*args, **kwargs):
        return {"slope": 1.0}

    def _fakeEvalVarianceFunction(opt, meanTrack, EB_minLin=1.0):
        seen["prior_eval_mean_track"] = np.asarray(meanTrack).copy()
        meanTrackArr = np.asarray(meanTrack, dtype=np.float32)
        return np.maximum(0.25, meanTrackArr + 0.1).astype(np.float32)

    monkeypatch.setattr(
        cconsenrich,
        "cSparseNearestMeanVarTrack",
        _fakeSparseNearest,
    )
    monkeypatch.setattr(
        cconsenrich,
        "crolling_AR1_IVar",
        _forbidRolling,
    )
    monkeypatch.setattr(
        cconsenrich,
        "cmeanVarPairs",
        _fakeMeanVarPairs,
    )
    monkeypatch.setattr(cconsenrich, "cEMA", _fakeEMA)
    monkeypatch.setattr(core, "fitVarianceFunction", _fakeFitVarianceFunction)
    monkeypatch.setattr(core, "evalVarianceFunction", _fakeEvalVarianceFunction)

    muncTrack, support = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        samplingBlockSizeBP=125,
        samplingIters=64,
        sparseIntervalIndices=sparseIntervalIndices,
        numNearest=3,
        EB_use=True,
    )

    expectedResidual = values.astype(np.float32) - sparseMeanTrack.astype(np.float32)
    expectedMagnitude = np.abs(expectedResidual)

    assert np.allclose(seen["prior_fit_values"], expectedResidual)
    assert np.allclose(seen["ema_input"], expectedResidual)
    assert np.allclose(seen["prior_eval_mean_track"], expectedMagnitude)
    assert muncTrack.shape == values.shape
    assert np.isfinite(muncTrack).all()
    assert support > 0.0


@pytest.mark.correctness
def testGetMuncTrackRestrictsRollingAR1ToSparseBed(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 400, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.6, intervals.size, dtype=np.float32)
    excludeMask = np.zeros(intervals.size, dtype=np.uint8)
    excludeMask[6] = 1
    sparseRegionMask = np.array(
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        dtype=np.uint8,
    )
    fakeRollingVarTrack = np.linspace(0.3, 0.7, intervals.size, dtype=np.float32)
    fakeBlockMeans = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    fakeBlockVars = np.linspace(0.2, 0.6, 8, dtype=np.float32)
    seen: dict[str, np.ndarray] = {}

    def _fakeRolling(valuesArg, blockLengthArg, excludeMaskArg, *args, **kwargs):
        seen["values"] = np.asarray(valuesArg).copy()
        seen["excludeMask"] = np.asarray(excludeMaskArg).copy()
        seen["blockLength"] = np.asarray([blockLengthArg], dtype=np.int64)
        return fakeRollingVarTrack.copy()

    def _fakeMeanVarPairs(*args, **kwargs):
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    def _fakeFitVarianceFunction(*args, **kwargs):
        return {"slope": 1.0}

    def _fakeEvalVarianceFunction(opt, meanTrack, EB_minLin=1.0):
        meanTrackArr = np.asarray(meanTrack, dtype=np.float32)
        return np.maximum(0.25, meanTrackArr + 0.1).astype(np.float32)

    monkeypatch.setattr(
        cconsenrich,
        "crolling_AR1_IVar",
        _fakeRolling,
    )
    monkeypatch.setattr(
        cconsenrich,
        "cmeanVarPairs",
        _fakeMeanVarPairs,
    )
    monkeypatch.setattr(core, "fitVarianceFunction", _fakeFitVarianceFunction)
    monkeypatch.setattr(core, "evalVarianceFunction", _fakeEvalVarianceFunction)

    muncTrack, support = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        samplingBlockSizeBP=125,
        samplingIters=64,
        excludeMask=excludeMask,
        sparseRegionMask=sparseRegionMask,
        restrictLocalAR1ToSparseBed=True,
        EB_use=True,
    )

    expectedExcludeMask = np.logical_or(
        excludeMask != 0,
        sparseRegionMask == 0,
    ).astype(np.uint8)

    assert np.array_equal(seen["values"], values)
    assert int(seen["blockLength"][0]) == 6
    assert np.array_equal(seen["excludeMask"], expectedExcludeMask)
    assert muncTrack.shape == values.shape
    assert np.isfinite(muncTrack).all()
    assert support > 0.0


@pytest.mark.correctness
def testRunConsenrichAPNSmoke():
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
        offDiagQ=0.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        autoDeltaF=False,
        EM_maxIters=2,
        EM_outerIters=1,
        EM_useProcPrecReweight=True,
        EM_useAPN=True,
        applyJackknife=False,
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, qScale, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(NIS))
    assert np.allclose(qScale, 1.0)


@pytest.mark.correctness
def testRunConsenrichDisableCalibrationUsesPluginAndAPN(
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

    def _forbidInnerEM(*args, **kwargs):
        raise AssertionError("cinnerEM should not run when calibration is disabled")

    monkeypatch.setattr(cconsenrich, "cinnerEM", _forbidInnerEM)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
        maxQ=0.5,
        offDiagQ=0.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        autoDeltaF=False,
        disableCalibration=True,
        EM_useAPN=True,
        EM_useProcPrecReweight=True,
        applyJackknife=False,
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, qScale, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.allclose(qScale, 1.0)


@pytest.mark.correctness
def testGetContextSizeBootstrapWidthVariance():
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

    pointEstimate, widthLower, widthUpper = core.getContextSize(
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


@pytest.mark.correctness
def testReadSegmentsFragmentsGrouped():
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
def testReadSegmentsFragmentsDefaultToCutSites():
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

    assert np.allclose(counts[0], np.array([2.0, 2.0], dtype=np.float32))


@pytest.mark.correctness
def testReadSegmentsFragmentsRespectModeAndMultiplicity(tmp_path):
    gzPath = FRAGMENTS_DIR / "small.fragments.tsv.gz"
    allowListPath = tmp_path / "allow_AB.txt"
    allowListPath.write_text("BC_A\nBC_B\n", encoding="ascii")

    expectedByMode = {
        "coverage": np.array([2.0, 5.0, 4.0, 3.0], dtype=np.float32),
        "cutsite": np.array([2.0, 6.0, 4.0, 4.0], dtype=np.float32),
        "center": np.array([0.0, 3.0, 4.0, 1.0], dtype=np.float32),
    }

    for countMode, expected in expectedByMode.items():
        counts = core.readSegments(
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
        )

        assert np.allclose(counts[0], expected), countMode


@pytest.mark.correctness
def testNormalizeCountModeRejectsLegacyAliases():
    for alias in ["cov", "cut", "cutsites", "5p", "five_prime", "centre", "midpoint"]:
        with pytest.raises(ValueError, match="Unsupported countMode"):
            core._normalizeCountMode(alias, "coverage")


@pytest.mark.correctness
def testReadSegmentsBamPairedEndUsesTemplateSpan(tmp_path):
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
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10:18] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def testReadSegmentsPairedBamCanUseRead1OnlySingleEndMode(tmp_path):
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
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10:12] = 1.0
    assert np.allclose(counts[0], expected)


@pytest.mark.correctness
def testReadSegmentsBamCountEndsOnlyUsesFivePrimePositions(tmp_path):
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
def testReadBamSegmentsCountEndsOnlyUsesFivePrimePositions(tmp_path):
    bamPath = _writeSyntheticBam(
        tmp_path,
        "legacy-ends.synthetic.bam",
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
def testFragmentsMappedCountUsesEmittedInsertionsAndSelectedCells():
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
        groupCellCount=cellCount,
        fragmentsGroupNorm="CELLS",
    )

    assert mappedCount == 12
    assert cellCount == 1
    assert scaleFactor == pytest.approx((1_000_000 / 12.0) * 100.0)
