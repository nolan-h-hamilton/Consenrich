# -*- coding: utf-8 -*-

# the name of this file is unfortunately misleading --
# 'core' as in 'central' or 'basic', not the literal
# `consenrich.core` module

import math
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
import consenrich.matching as matching
import consenrich.misc_util as misc_util


TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
FRAGMENTS_DIR = TEST_DATA_DIR / "fragments"


@pytest.mark.correctness
def testSingleEndDetection():
    # case: single-end BAM
    bamFiles = ["smallTest.bam"]
    pairedEndStatus = misc_util.bamsArePairedEnd(bamFiles, maxReads=1_000)
    assert isinstance(pairedEndStatus, list)
    assert len(pairedEndStatus) == 1
    assert isinstance(pairedEndStatus[0], bool)
    assert pairedEndStatus[0] is False


@pytest.mark.correctness
def testPairedEndDetection():
    # case: paired-end BAM
    bamFiles = ["smallTest2.bam"]
    pairedEndStatus = misc_util.bamsArePairedEnd(bamFiles, maxReads=1_000)
    assert isinstance(pairedEndStatus, list)
    assert len(pairedEndStatus) == 1
    assert isinstance(pairedEndStatus[0], bool)
    assert pairedEndStatus[0] is True


@pytest.mark.matching
def testmatchWaveletUnevenIntervals():
    np.random.seed(42)
    intervals = np.random.randint(0, 1000, size=100, dtype=int)
    intervals = np.unique(intervals)
    intervals.sort()
    values = np.random.poisson(lam=5, size=len(intervals)).astype(float)
    with pytest.raises(ValueError, match="spaced"):
        matching.matchWavelet(
            chromosome="chr1",
            intervals=intervals,
            values=values,
            templateNames=["haar"],
            cascadeLevels=[1],
            iters=1000,
        )


@pytest.mark.matching
def testMatchExistingBedGraph():
    np.random.seed(42)
    with tempfile.TemporaryDirectory() as tempFolder:
        bedGraphPath = Path(tempFolder) / "toyFile.bedGraph"
        fakeVals = []
        for i in range(1000):
            if (i % 100) <= 10:
                # add in about ~10~ peak-like regions
                fakeVals.append(max(np.random.poisson(lam=5), 1))
            else:
                # add in background poisson(1) for BG
                fakeVals.append(np.random.poisson(lam=1))

        fakeVals = np.array(fakeVals).astype(float)
        dataFrame = pd.DataFrame(
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
        dataFrame = dataFrame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        dataFrame.to_csv(bedGraphPath, sep="\t", header=False, index=False)
        outputPath = matching.runMatchingAlgorithm(
            bedGraphFile=str(bedGraphPath),
            templateNames=["haar"],
            cascadeLevels=[5],
            iters=5000,
            alpha=0.10,
            minSignalAtMaxima=-1,
            minMatchLengthBP=50,
        )
        assert outputPath is not None
        assert os.path.isfile(outputPath)
        with open(outputPath, "r") as fileHandle:
            lineStrings = fileHandle.readlines()

        # Not really the point of this test but
        # makes sure we're somewhat calibrated
        # Updated 15,3 to account for now-default BH correction
        assert len(lineStrings) <= 15  # more than 20 might indicate high FPR
        assert len(lineStrings) >= 3  # fewer than 5 might indicate low power


@pytest.mark.matching
def testMatchWaveletGlobalFallbackRetainsCandidates():
    intervals = np.arange(0, 400, 10, dtype=int)
    values = np.zeros(intervals.size, dtype=float)
    values[16:24] = np.array([0.0, 1.0, 3.0, 6.0, 6.0, 3.0, 1.0, 0.0], dtype=float)

    df, minMatchLengthBP = matching.matchWavelet(
        chromosome="chr1",
        intervals=intervals,
        values=values,
        templateNames=["haar"],
        cascadeLevels=[3],
        iters=1000,
        alpha=1.0,
        minMatchLengthBP=40,
        minSignalAtMaxima=-1,
        randSeed=42,
    )

    assert minMatchLengthBP == 40
    assert len(df) >= 1
    assert float(df["signal"].max()) >= 6.0


@pytest.mark.matching
def testMatchWaveletSplitEmpiricalNullIsOptional():
    intervals = np.arange(0, 1000, 10, dtype=int)
    values = np.zeros(intervals.size, dtype=float)
    values[18:25] = np.array([0.0, 2.0, 5.0, 8.0, 5.0, 2.0, 0.0], dtype=float)
    values[58:65] = np.array([0.0, 1.5, 4.0, 7.0, 4.0, 1.5, 0.0], dtype=float)

    dfGlobal, _ = matching.matchWavelet(
        chromosome="chr1",
        intervals=intervals,
        values=values,
        templateNames=["haar"],
        cascadeLevels=[3],
        iters=1000,
        alpha=1.0,
        minMatchLengthBP=40,
        minSignalAtMaxima=-1,
        randSeed=42,
    )
    dfSplit, _ = matching.matchWavelet(
        chromosome="chr1",
        intervals=intervals,
        values=values,
        templateNames=["haar"],
        cascadeLevels=[3],
        iters=1000,
        alpha=1.0,
        minMatchLengthBP=40,
        minSignalAtMaxima=-1,
        randSeed=42,
        useSplitEmpiricalNull=True,
    )

    assert len(dfGlobal) >= 1
    assert len(dfSplit) >= 1
    assert set(dfGlobal.columns) == set(dfSplit.columns)


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
def testIntervalMuncUpdateIsPositive():
    n = 48
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    stateSmoothed = np.column_stack(
        [
            np.sin(grid),
            np.gradient(np.sin(grid)).astype(np.float32),
        ]
    ).astype(np.float32)
    stateCovarSmoothed = np.zeros((n, 2, 2), dtype=np.float32)
    stateCovarSmoothed[:, 0, 0] = 0.05
    matrixData = np.vstack(
        [
            stateSmoothed[:, 0] + 0.10 * np.cos(grid),
            stateSmoothed[:, 0] - 0.08 * np.cos(grid),
            stateSmoothed[:, 0] + 0.05 * np.sin(2.0 * grid),
        ]
    ).astype(np.float32)

    track = core._estimateIntervalMuncTrack(
        matrixData=matrixData,
        stateSmoothed=stateSmoothed,
        stateCovarSmoothed=stateCovarSmoothed,
        replicateBias=np.zeros(m, dtype=np.float32),
        replicateScale=np.ones(m, dtype=np.float32),
        backgroundTrack=np.zeros(n, dtype=np.float32),
        lambdaExp=None,
        pad=1.0e-4,
        minR=1.0e-3,
        maxR=10.0,
        binQuantileCutoff=0.5,
        EB_minLin=0.1,
        EB_use=True,
    )

    assert track.shape == (n,)
    assert np.isfinite(track).all()
    assert np.min(track) > 0.0


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
        EM_useIntervalBackground=True,
        EM_useIntervalMunc=True,
        conformalRescale=False,
        applyJackknife=False,
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)


@pytest.mark.correctness
def testRunConsenrichOuterEMReusesObservationEBSettings(monkeypatch):
    captured: dict[str, object] = {}
    originalEstimator = core._estimateIntervalMuncTrack

    def _wrappedEstimator(*args, **kwargs):
        captured["binQuantileCutoff"] = kwargs.get("binQuantileCutoff")
        captured["EB_minLin"] = kwargs.get("EB_minLin")
        captured["EB_use"] = kwargs.get("EB_use")
        captured["EB_setNu0"] = kwargs.get("EB_setNu0")
        captured["EB_setNuL"] = kwargs.get("EB_setNuL")
        return originalEstimator(*args, **kwargs)

    monkeypatch.setattr(core, "_estimateIntervalMuncTrack", _wrappedEstimator)

    n = 48
    m = 3
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    signalTrack = np.sin(grid).astype(np.float32)
    matrixData = np.vstack(
        [
            signalTrack + 0.04 * np.cos(grid),
            signalTrack - 0.03 * np.cos(grid),
            signalTrack + 0.02 * np.sin(2.0 * grid),
        ]
    ).astype(np.float32)
    matrixMunc = np.full((m, n), 0.2, dtype=np.float32)

    core.runConsenrich(
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
        EM_maxIters=2,
        EM_outerIters=1,
        EM_useIntervalBackground=False,
        EM_useIntervalMunc=True,
        intervalMuncBinQuantileCutoff=0.8,
        intervalMuncEB_minLin=0.125,
        intervalMuncEB_use=False,
        intervalMuncEB_setNu0=17,
        intervalMuncEB_setNuL=11,
        conformalRescale=False,
        applyJackknife=False,
    )

    assert captured == {
        "binQuantileCutoff": 0.8,
        "EB_minLin": 0.125,
        "EB_use": False,
        "EB_setNu0": 17,
        "EB_setNuL": 11,
    }


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
        offsetStr="0,0",
        maxInsertSize=0,
        pairedEndMode=0,
        inferFragmentLength=0,
        countEndsOnly=False,
        minMappingQuality=0,
        minTemplateLength=0,
        fragmentLengths=[0, 0],
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
        fragmentLengths=[0],
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
            fragmentLengths=[0],
        )

        assert np.allclose(counts[0], expected), countMode


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
