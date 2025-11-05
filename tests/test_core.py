# -*- coding: utf-8 -*-

# the name of this file is unfortunately misleading --
# 'core' as in 'central' or 'basic', not the literal
# `consenrich.core` module

import math
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
import scipy.stats as stats
import scipy.signal as spySig # renamed to avoid conflict with any `signal` variables

import consenrich.core as core
import consenrich.cconsenrich as cconsenrich
import consenrich.matching as matching
import consenrich.misc_util as misc_util


@pytest.mark.correctness
def testConstantGetAverageLocalVarianceTrack(constantValue=10):
    # case: `values` is constant --> noise level should be zero, but due to clipping, `minR`
    values = np.ones(100) * constantValue
    stepSize = 1
    approximationWindowLengthBP = 10
    lowPassWindowLengthBP = 20
    minR = 1.0
    maxR = 100.0
    out = core.getAverageLocalVarianceTrack(
        values,
        stepSize,
        approximationWindowLengthBP,
        lowPassWindowLengthBP,
        minR,
        maxR,
    )
    np.testing.assert_allclose(out, np.ones_like(values) * minR)


@pytest.mark.correctness
def testMaxVarGetAverageLocalVarianceTrack(maxVariance=20):
    # case: values (length 1000) ~ Poisson(maxVariance*2) -->
    # mode(all noise levels) ~=~ maxVariance
    np.random.seed(42)
    values = np.random.poisson(lam=20, size=1000)
    stepSize = 1
    approximationWindowLengthBP = 10
    lowPassWindowLengthBP = 20
    minR = 0.0
    maxR = maxVariance
    out = core.getAverageLocalVarianceTrack(
        values,
        stepSize,
        approximationWindowLengthBP,
        lowPassWindowLengthBP,
        minR,
        maxR,
    )
    np.testing.assert_allclose(stats.mode(out)[0], maxR, rtol=0.001)


@pytest.mark.correctness
def testMatrixConstruction(
    deltaF=0.50, coefficients=[0.1, 0.2, 0.3, 0.4], minQ=0.25, offDiag=0.10
):
    # F
    m = len(coefficients)
    matrixF = core.constructMatrixF(deltaF)
    assert matrixF.shape == (2, 2)
    np.testing.assert_allclose(matrixF, np.array([[1.0, deltaF], [0.0, 1.0]]))

    # H
    matrixH = core.constructMatrixH(m, coefficients)
    assert matrixH.shape == (m, 2)
    np.testing.assert_allclose(matrixH[:, 0], coefficients)
    np.testing.assert_allclose(matrixH[:, 1], np.zeros(m))

    # Q
    matrixQ = core.constructMatrixQ(minQ, offDiag)
    assert matrixQ.shape == (2, 2)
    np.testing.assert_allclose(
        matrixQ, np.array([[minQ, offDiag], [offDiag, minQ]])
    )


@pytest.mark.chelpers
def testResidualCovarianceInversion():
    np.random.seed(42)
    m = 10
    muncMatrixIter = np.random.gamma(shape=2, scale=1.0, size=m) + 1
    priorCovarianceOO = 0.1
    residCovar = np.diag(muncMatrixIter) + (np.ones((m, m)) * priorCovarianceOO)

    invertedMatrix = cconsenrich.cinvertMatrixE(
        muncMatrixIter.astype(np.float32), np.float32(priorCovarianceOO)
    )
    np.testing.assert_allclose(
        invertedMatrix @ residCovar, np.eye(m), atol=1e-8
    )


@pytest.mark.chelpers
def testProcessNoiseAdjustment():
    np.random.seed(42)

    m = 100
    minQ = 0.25
    maxQ = 10.0
    offDiag = 0.0
    dStatAlpha = 3.0
    dStatd = 10.0
    dStatPC = 1.0
    inflatedQ = False

    matrixQ = np.array([[minQ, offDiag], [offDiag, minQ]], dtype=np.float32)
    matrixQCopy = matrixQ.copy()
    vectorY = (np.random.normal(0, 15, size=m)).astype(np.float32)
    dStat = np.mean(vectorY**2).astype(np.float32)
    dStatDiff = np.float32(
        np.sqrt(np.abs(dStat - dStatAlpha) * dStatd + dStatPC)
    )

    matrixQ, inflatedQ = cconsenrich.updateProcessNoiseCovariance(
        matrixQ,
        matrixQCopy,
        dStat,
        dStatAlpha,
        dStatd,
        dStatPC,
        inflatedQ,
        maxQ,
        minQ,
    )

    assert inflatedQ is True
    np.testing.assert_allclose(matrixQ, maxQ * np.eye(2), rtol=0.01)


@pytest.mark.correctness
def testbedMask(tmp_path):
    bedPath = tmp_path / "testTmp.bed"
    bedPath.write_text("chr1\t50\t2000\nchr1\t3000\t5000\nchr1\t10000\t20000\n")
    intervals = np.arange(500, 10_000, 25)
    mask = core.getBedMask("chr1", bedPath, intervals)

    # first test: mask and intervals equal length
    assert len(mask) == len(intervals)

    for i, interval_ in enumerate(intervals):
        if 50 <= interval_ < 2000 or 3000 <= interval_ < 5000:
            assert mask[i] == 1
        else:
            assert mask[i] == 0


@pytest.mark.correctness
def testgetPrecisionWeightedResidualWithCovar():
    np.random.seed(0)
    n, m = 5, 3
    postFitResiduals = np.random.randn(n, m).astype(np.float32)
    matrixMunc = (np.random.rand(m, n).astype(np.float32) * 2.0) + 0.5
    add_vec = np.random.rand(n).astype(np.float32) * 0.5
    stateCovarSmoothed = np.zeros((n, 2, 2), dtype=np.float32)
    stateCovarSmoothed[:, 0, 0] = add_vec
    totalUnc = matrixMunc + add_vec
    weights = 1.0 / totalUnc
    expected = (postFitResiduals * weights.T).sum(axis=1) / weights.sum(axis=0)
    out = core.getPrecisionWeightedResidual(
        postFitResiduals=postFitResiduals,
        matrixMunc=matrixMunc,
        roundPrecision=6,
        stateCovarSmoothed=stateCovarSmoothed,
    )
    np.testing.assert_allclose(
        out, expected.astype(np.float32), rtol=1e-6, atol=1e-6
    )


@pytest.mark.correctness
def testgetPrimaryStateF64():
    xVec = np.array(
        [[1.2349, 0.0], [-2.5551, 0.0], [10.4446, 0.0], [-0.5001, 0.0]],
        dtype=np.float64,
    )
    stateVectors = xVec[:, :]
    out = core.getPrimaryState(stateVectors, roundPrecision=3)
    np.testing.assert_array_equal(out.dtype, np.float32)
    np.testing.assert_allclose(
        out,
        np.array([1.235, -2.555, 10.445, -0.500], dtype=np.float32),
        rtol=0,
        atol=0,
    )


@pytest.mark.correctness
def testFragLen(threshold: float = 25, expected: float = 220):
    fragLens = []
    for i in range(50):
        fragLen = float(
            cconsenrich.cgetFragmentLength(
                "smallTest.bam",
                "chr6",
                32_000_000,
                35_000_000,
                randSeed=i,
            )
        )
        fragLens.append(fragLen)
    fragLens.sort()

    assert stats.iqr(fragLens) < 2 * threshold
    assert abs(np.median(fragLens) - expected) < threshold


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
                    np.ones(10) / 10, # smooth out over ~100bp~
                    mode="same",
                ),
            }
        )
        dataFrame.to_csv(bedGraphPath, sep="\t", header=False, index=False)
        outputPath = matching.matchExistingBedGraph(
            bedGraphFile=str(bedGraphPath),
            templateName="haar",
            cascadeLevel=5,
            alpha=0.10,
            merge=False,
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
        assert len(lineStrings) <= 15 # more than 20 might indicate high FPR
        assert len(lineStrings) >= 3  # fewer than 5 might indicate low power


@pytest.mark.matching
def testMergeMatches():
    TEST_FILE = "unmerged.test.narrowPeak"

    outFile = matching.mergeMatches(TEST_FILE, mergeGapBP=75)
    assert outFile and os.path.isfile(outFile), "No output 'merged' file found"

    name_re = re.compile(
        r"^consenrichPeak\|i=(?P<i>\d+)\|gap=(?P<gap>\d+)bp\|ct=(?P<ct>\d+)\|qRange=(?P<qmin>\d+\.\d{3})_(?P<qmax>\d+\.\d{3})$"
    )
    with open(outFile) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0, "no merged features"

    idx = 0
    for line in lines:
        idx += 1
        line_ = line.strip()
        fields = line_.split("\t")
        assert len(fields) == 10, f"Line {idx}: fewer than 10 narrowPeak fields"
        chrom, start_, end_, name, score, strand, sigAvg, pHM, qHM, point = fields[:10]
        record_ = name_re.match(name)
        assert record_, f"Could not parse feature name: {name}"

        gap = int(record_["gap"])
        ct = int(record_["ct"])
        assert gap == 75, "parsed mergeGapBP in feature name does not match expected"
        assert ct >= 1, "parsed count of merged peaks should be at least 1"

        qMinLog10 = float(record_["qmin"])
        qMaxLog10 = float(record_["qmax"])
        qMin = np.round(10 ** (- float(qMaxLog10)),3)
        qMax = np.round(10 ** (- float(qMinLog10)),3)

        qHarmonicMean = np.round(10 ** (- float(qHM)),3)
        assert qHarmonicMean >= qMin, f"harmonic mean of q-values should be greater/equal to minimum q-value: {line_}"
        assert qHarmonicMean <= qMax, f"harmonic mean of q-values should be less/equal to maximum q-value: {line_}"


@pytest.mark.matching
def testMergeMatchesReduction():

    TEST_FILE = "unmerged.test.narrowPeak"

    with open(TEST_FILE) as f:
        linesInit = [line.strip() for line in f if line.strip()]
    numInit = len(linesInit)

    outFile = matching.mergeMatches(TEST_FILE, mergeGapBP=75)

    assert outFile and os.path.isfile(outFile), (
        "No output file can be found after call to `mergeMatches`"
    )

    name_re = re.compile(
        r"^consenrichPeak\|i=(?P<i>\d+)\|gap=(?P<gap>\d+)bp\|ct=(?P<ct>\d+)\|qRange=(?P<qmin>\d+\.\d{3})_(?P<qmax>\d+\.\d{3})$"
    )
    with open(outFile) as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0, "no remaining features after merge"

    assert len(lines) > 25, (
        f"Unexpected: too few features remaining after merge {len(lines)},{numInit}"
    )

    assert len(lines) < 75, (
        f"Unexpected: too many features remaining after merge {len(lines)},{numInit}"
    )
