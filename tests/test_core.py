# -*- coding: utf-8 -*-

# the name of this file is unfortunately misleading --
# 'core' as in 'central' or 'basic', not the literal
# `consenrich.core` module

import math
import os
import logging
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
import consenrich.diagnostics as diagnostics
import consenrich.detrorm as detrorm
import consenrich.misc_util as misc_util
import consenrich.peaks as peaks


TESTS_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TESTS_DIR / "data"
FRAGMENTS_DIR = TEST_DATA_DIR / "fragments"
_LEGACY_ALGO_PREFIX = "E" + "M" + "_"


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
                scale_
                * np.float32(math.log(float(t)) - math.log(float(c)))
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


def _levelKalmanReference(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    *,
    qLevel: float,
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
        p += float(qLevel)
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
            pNoiseForward[k - 1, 0, 0] = float(qLevel)

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
def _caseAsciiPhaseLogFormattingIsCompactAndAttributed(caplog):
    block = core._formatAsciiLogBlock(
        "MUNC track start",
        (
            ("chromosome", "chr11"),
            ("MUNC variance EB", "enabled"),
            (
                "long value",
                "this value is too long for the table and should be omitted",
            ),
        ),
    )

    assert max(len(line) for line in block.splitlines()) <= 72
    assert "PHASE: MUNC TRACK START" in block
    assert "| MUNC variance EB" in block
    assert "| enabled" in block
    assert "long value" not in block
    indentedBlock = core._formatAsciiLogBlock(
        "MUNC track child",
        (("chromosome", "chr11"),),
        indentLevel=1,
    )
    assert indentedBlock.startswith("      +")
    assert "\n      | PHASE: MUNC TRACK CHILD" in indentedBlock

    caplog.set_level(logging.INFO, logger=core.logger.name)
    core._logAsciiBlock("MUNC track start", (("MUNC variance EB", "enabled"),))

    assert caplog.records[-1].funcName.endswith("AsciiPhaseLogFormattingIsCompactAndAttributed")
    assert caplog.records[-1].message.startswith("\n+")
    assert caplog.records[-1].message.endswith("\n")


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
def _caseSubtractGlobalMedianCentersEachTrackInPlace():
    tracks = np.array(
        [
            [1.0, 2.0, 100.0, 4.0, 5.0],
            [-8.0, -4.0, -2.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    original = tracks.copy()

    stats_ = core.subtractGlobalMedianInPlace(tracks)

    assert stats_["applied"] is True
    assert stats_["applied_tracks"] == tracks.shape[0]
    np.testing.assert_allclose(
        np.asarray(stats_["track_medians"]),
        np.median(original.astype(np.float64), axis=1),
    )
    np.testing.assert_allclose(
        np.median(tracks.astype(np.float64), axis=1),
        np.zeros(tracks.shape[0], dtype=np.float64),
        atol=1.0e-7,
    )
    assert not np.allclose(tracks, original)


@pytest.mark.correctness
def _caseQuantileFilterSubtractsUncenteredTrendInPlace():
    x = np.linspace(8.0, 10.0, 101, dtype=np.float32)
    x[50] += 4.0
    original = x.copy()
    expectedTrend = core.ndimage.median_filter(original, size=21, mode="nearest")
    expected = original - expectedTrend

    stats_ = core.quantileFilterDetrendInPlace(x, 20, quantile=0.5)

    assert stats_["applied"] is True
    assert stats_["window_intervals"] == 21
    assert stats_["detrend_quantile"] == pytest.approx(0.5)
    np.testing.assert_allclose(x, expected, rtol=1.0e-6, atol=1.0e-6)
    anchored = original - (expectedTrend - np.median(expectedTrend))
    assert np.max(np.abs(x - anchored)) > 8.0
    assert x[50] > 3.5


@pytest.mark.correctness
def _caseQuantileFilterDetrendUsesRequestedQuantile():
    x = np.linspace(8.0, 10.0, 101, dtype=np.float32)
    x[45:56] += np.linspace(0.0, 5.0, 11, dtype=np.float32)
    original = x.copy()
    expectedTrend = core.ndimage.percentile_filter(
        original,
        percentile=75.0,
        size=21,
        mode="nearest",
    )
    expected = original - expectedTrend

    stats_ = core.quantileFilterDetrendInPlace(x, 20, quantile=0.75)

    assert stats_["applied"] is True
    assert stats_["window_intervals"] == 21
    assert stats_["detrend_quantile"] == pytest.approx(0.75)
    np.testing.assert_allclose(x, expected, rtol=1.0e-6, atol=1.0e-6)


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
def _caseZeroCenteredBackgroundUpdate():
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

    background = core._solveZeroCenteredBackground(
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

    background = core._solveZeroCenteredBackground(
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

    centered = core._solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )
    uncentered = core._solveZeroCenteredBackground(
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

    unconstrained = core._solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=4,
        backgroundSmoothness=0.5,
        zeroCenter=False,
        useNonnegative=False,
    )
    constrained = core._solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=4,
        backgroundSmoothness=0.5,
        zeroCenter=False,
        useNonnegative=True,
    )
    disabled = core._solveZeroCenteredBackground(
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
    stiffUnconstrained = core._solveZeroCenteredBackground(
        residualMatrix=stiffResidualMatrix,
        invVarMatrix=stiffInvVarMatrix,
        blockLenIntervals=80,
        backgroundSmoothness=1.0,
        zeroCenter=False,
        useNonnegative=False,
    ).astype(np.float64)
    stiffClipped = np.maximum(stiffUnconstrained, 0.0)
    stiffConstrained = core._solveZeroCenteredBackground(
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
def _caseReplicateBiasIsAlwaysZeroCentered():
    n = 24
    m = 3
    matrixData = np.full((m, n), 2.0, dtype=np.float32)
    matrixMunc = np.full((m, n), 10.0, dtype=np.float32)
    matrixF = core.constructMatrixF(0.1).astype(np.float32, copy=False)
    matrixQ0 = core.constructMatrixQ(
        minDiagQ=1.0e-6,
    ).astype(np.float32, copy=False)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)

    commonKwargs = dict(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0e-8,
        ECM_fixedBackgroundIters=1,
        ECM_fixedBackgroundRtol=0.0,
        pad=1.0e-4,
        ECM_robustTNu=8.0,
        ECM_useObsPrecisionReweighting=False,
        ECM_useProcessPrecisionReweighting=False,
        ECM_useAPN=False,
        returnIntermediates=True,
        t_innerIters=1,
    )

    centeredBias = cconsenrich.cfixedBackgroundECM(**commonKwargs)[-1]

    assert abs(float(np.mean(centeredBias))) < 1.0e-5


@pytest.mark.correctness
def _caseCFixedBackgroundECMReplicateBiasUpdateMatchesPrecisionWeightedMinimizer():
    n = 48
    offsets = np.array([1.25, -0.75, 0.35], dtype=np.float32)
    trend = np.linspace(-0.1, 0.1, n, dtype=np.float32)
    matrixData = offsets[:, None] + np.vstack(
        [
            0.05 * trend,
            -0.02 * trend,
            0.01 * np.sin(np.linspace(0.0, np.pi, n, dtype=np.float32)),
        ]
    )
    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.vstack(
        [
            np.linspace(0.2, 0.6, n, dtype=np.float32),
            np.linspace(1.5, 0.4, n, dtype=np.float32),
            np.full(n, 0.85, dtype=np.float32),
        ]
    ).astype(np.float32)
    matrixF = core.constructMatrixF(0.1).astype(np.float32, copy=False)
    matrixQ0 = core.constructMatrixQ(
        minDiagQ=1.0e-12,
    ).astype(np.float32, copy=False)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)
    pad = 1.0e-4

    out = cconsenrich.cfixedBackgroundECM(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0e-10,
        ECM_fixedBackgroundIters=1,
        ECM_fixedBackgroundRtol=0.0,
        pad=pad,
        ECM_robustTNu=8.0,
        ECM_useObsPrecisionReweighting=False,
        ECM_useProcessPrecisionReweighting=False,
        ECM_useAPN=False,
        returnIntermediates=True,
        t_innerIters=1,
    )
    stateSmoothed = np.asarray(out[2], dtype=np.float64)[:, 0]
    replicateBias = np.asarray(out[-1], dtype=np.float64)

    invVar = 1.0 / np.maximum(matrixMunc.astype(np.float64) + pad, 1.0e-12)
    den = np.sum(invVar, axis=1)
    raw = (
        np.sum(
            invVar * (matrixData.astype(np.float64) - stateSmoothed[None, :]),
            axis=1,
        )
        / den
    )
    expected = raw - (np.sum(den * raw) / np.sum(den))

    assert np.allclose(replicateBias, expected, atol=1.0e-5)
    assert abs(float(np.sum(den * replicateBias))) < 1.0e-4


@pytest.mark.correctness
def _caseCFixedBackgroundECMReplicateBiasUsesFixedCenterConstraintWithRobustWeights():
    n = 56
    offsets = np.array([1.4, -0.6, 0.15], dtype=np.float32)
    grid = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    base = 0.2 * np.sin(grid)
    matrixData = offsets[:, None] + np.vstack(
        [
            base,
            base + 0.05 * np.cos(grid),
            base - 0.03 * np.sin(2.0 * grid),
        ]
    )
    matrixData[0, 7] += 4.0
    matrixData[1, 31] -= 3.5
    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.vstack(
        [
            np.linspace(0.12, 1.4, n, dtype=np.float32),
            np.linspace(1.2, 0.2, n, dtype=np.float32),
            np.full(n, 0.55, dtype=np.float32),
        ]
    ).astype(np.float32)
    matrixF = core.constructMatrixF(0.1).astype(np.float32, copy=False)
    matrixQ0 = core.constructMatrixQ(
        minDiagQ=1.0e-10,
    ).astype(np.float32, copy=False)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)
    pad = 1.0e-4

    out = cconsenrich.cfixedBackgroundECM(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0e-8,
        ECM_fixedBackgroundIters=1,
        ECM_fixedBackgroundRtol=0.0,
        pad=pad,
        ECM_robustTNu=2.5,
        ECM_useObsPrecisionReweighting=True,
        ECM_useProcessPrecisionReweighting=False,
        ECM_useAPN=False,
        obsPrecisionMultiplierMin=0.25,
        obsPrecisionMultiplierMax=4.0,
        returnIntermediates=True,
        t_innerIters=1,
    )
    stateSmoothed = np.asarray(out[2], dtype=np.float64)[:, 0]
    lambdaExp = np.asarray(out[6], dtype=np.float64)
    replicateBias = np.asarray(out[-1], dtype=np.float64)

    assert lambdaExp.shape == (n,)
    baseInvVar = 1.0 / np.maximum(matrixMunc.astype(np.float64) + pad, 1.0e-12)
    centerWeight = np.sum(baseInvVar, axis=1)
    currentInvVar = lambdaExp * baseInvVar
    den = np.sum(currentInvVar, axis=1)
    raw = (
        np.sum(
            currentInvVar * (matrixData.astype(np.float64) - stateSmoothed[None, :]),
            axis=1,
        )
        / den
    )
    alpha = np.sum(centerWeight * raw) / np.sum((centerWeight * centerWeight) / den)
    expected = raw - alpha * centerWeight / den

    movingCentered = raw - (np.sum(den * raw) / np.sum(den))
    assert np.max(np.abs(lambdaExp - 1.0)) > 1.0e-3
    assert not np.allclose(expected, movingCentered, atol=1.0e-4)
    assert np.allclose(replicateBias, expected, atol=1.0e-5)
    assert abs(float(np.sum(centerWeight * replicateBias))) < 1.0e-4


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
        _LEGACY_ALGO_PREFIX + "scaleToMedian",
        _LEGACY_ALGO_PREFIX + "alphaEMA",
        _LEGACY_ALGO_PREFIX + "scaleLOW",
        _LEGACY_ALGO_PREFIX + "scaleHIGH",
        _LEGACY_ALGO_PREFIX + "useProcBlockScale",
        _LEGACY_ALGO_PREFIX + "useReplicateScale",
        _LEGACY_ALGO_PREFIX + "repScaleLOW",
        _LEGACY_ALGO_PREFIX + "repScaleHIGH",
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
        residualSecondMoment = exx1 - (ex1x0 @ ft) - (matrixF @ ex0x1) + (
            matrixF @ exx0 @ ft
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
    for alias in ("level-trend", "two_state", "one-state", "scalar", "levelSlope"):
        with pytest.raises(ValueError, match="stateModel"):
            core._normalizeStateModel(alias)


@pytest.mark.correctness
def _caseExpectedLevelTransitionResidualSumsMatchesPythonReference():
    stateSmoothed = np.asarray([[0.0], [0.4], [0.9], [0.7]], dtype=np.float64)
    stateCovarSmoothed = np.asarray([0.10, 0.20, 0.15, 0.12], dtype=np.float64).reshape(
        -1, 1, 1
    )
    lagCovSmoothed = np.asarray([0.03, 0.04, 0.02], dtype=np.float64).reshape(
        -1, 1, 1
    )

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

    sumLevel, sumTrend, transitionCount = core._computeExpectedLevelTransitionResidualSums(
        stateSmoothed=stateSmoothed,
        stateCovarSmoothed=stateCovarSmoothed,
        lagCovSmoothed=lagCovSmoothed,
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
    qLevel = 0.06
    stateInit = -0.1
    stateCovarInit = 0.8
    pad = 0.02
    intervalToBlockMap = np.zeros(n, dtype=np.int32)
    matrixQ0 = np.asarray([[qLevel, 0.0], [0.0, 0.5]], dtype=np.float32)
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
    stateSmoothed, covSmoothed, lagCovSmoothed, residuals = cconsenrich.cbackwardPassLevel(
        matrixData=matrixData,
        stateForward=stateForward,
        stateCovarForward=covForward,
        pNoiseForward=pNoiseForward,
    )

    refForward, refCovForward, _refPNoise, refState, refCov, refLag, refResiduals = (
        _levelKalmanReference(
            matrixData,
            matrixMunc,
            qLevel=qLevel,
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
    np.testing.assert_allclose(lagCovSmoothed[: n - 1], refLag[: n - 1], rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(residuals, refResiduals, rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.correctness
def _caseWarmupProcessNoiseCalibrationReliabilityAndLevelModel():
    matrixF = core.constructMatrixF(1.0).astype(np.float32, copy=False)
    stateSmoothed = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.5, 0.7],
            [4.2, 1.1],
            [6.3, 1.4],
        ],
        dtype=np.float32,
    )
    stateCovarSmoothed = np.zeros((stateSmoothed.shape[0], 2, 2), dtype=np.float32)
    lagCovSmoothed = np.zeros((stateSmoothed.shape[0] - 1, 2, 2), dtype=np.float32)

    matrixQ, blockInfo = core._estimateWarmupProcessNoiseCalibration(
        stateSmoothed=stateSmoothed,
        stateCovarSmoothed=stateCovarSmoothed,
        lagCovSmoothed=lagCovSmoothed,
        matrixF=matrixF,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        minQ=1.0e-4,
        maxQ=10.0,
        regularizationStrength=1.0,
        regularizationRatio=0.001,
        blockLenIntervals=2,
    )
    _, changedKnobInfo = core._estimateWarmupProcessNoiseCalibration(
        stateSmoothed=stateSmoothed,
        stateCovarSmoothed=stateCovarSmoothed,
        lagCovSmoothed=lagCovSmoothed,
        matrixF=matrixF,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        minQ=1.0e-4,
        maxQ=10.0,
        regularizationStrength=99.0,
        regularizationRatio=0.9,
        blockLenIntervals=2,
    )

    assert matrixQ[0, 1] == pytest.approx(0.0)
    assert matrixQ[1, 0] == pytest.approx(0.0)
    assert blockInfo["processNoisePolicy"] == "EB_MAP"
    assert blockInfo["blockMode"] == "non_overlapping"
    assert blockInfo["validBlockCount"] > 0
    assert blockInfo["validRatioBlockCount"] > 0
    assert "repeatCount" not in blockInfo
    assert "logQLevelMean" not in blockInfo
    assert blockInfo["qLevel"] == pytest.approx(changedKnobInfo["qLevel"])
    assert blockInfo["qTrend"] != pytest.approx(changedKnobInfo["qTrend"])
    assert blockInfo["effectiveTrendLevelRatio"] > 0.0
    assert changedKnobInfo["effectiveTrendLevelRatio"] > blockInfo["effectiveTrendLevelRatio"]
    assert "q_level" not in blockInfo
    assert blockInfo["qFloor"] == pytest.approx(core.PROCESS_NOISE_NUMERICAL_FLOOR)

    levelState = np.asarray([[0.0], [0.2], [0.5], [0.9]], dtype=np.float32)
    levelCovar = np.zeros((levelState.shape[0], 1, 1), dtype=np.float32)
    levelLag = np.zeros((levelState.shape[0] - 1, 1, 1), dtype=np.float32)
    matrixQLevel, levelInfo = core._estimateWarmupProcessNoiseCalibration(
        stateSmoothed=levelState,
        stateCovarSmoothed=levelCovar,
        lagCovSmoothed=levelLag,
        matrixF=matrixF,
        stateModel=core.STATE_MODEL_LEVEL,
        minQ=1.0e-4,
        maxQ=1.0,
        regularizationStrength=1.0,
        regularizationRatio=0.5,
        blockLenIntervals=2,
    )
    assert matrixQLevel[0, 0] > 0.0
    assert levelInfo["qTrend"] == pytest.approx(0.0)
    assert levelInfo["effectiveTrendLevelRatio"] == pytest.approx(0.0)
    assert levelInfo["ratioPriorDfUsed"] == pytest.approx(0.0)
    assert levelInfo["q11PaddingOnly"] is True
    assert levelInfo["processNoisePolicy"] == "EB_MAP"

    matrixQClamped, clampedInfo = core._estimateWarmupProcessNoiseCalibration(
        stateSmoothed=stateSmoothed * 10.0,
        stateCovarSmoothed=stateCovarSmoothed,
        lagCovSmoothed=lagCovSmoothed,
        matrixF=matrixF,
        stateModel=core.STATE_MODEL_LEVEL_TREND,
        minQ=1.0e-4,
        maxQ=0.05,
        regularizationStrength=1.0,
        regularizationRatio=0.001,
        blockLenIntervals=2,
    )
    assert matrixQClamped[0, 0] == pytest.approx(0.05)
    assert clampedInfo["qLevel"] == pytest.approx(0.05)
    assert clampedInfo["hitQLevelCap"] is True


@pytest.mark.correctness
def _caseRunConsenrichOuterPassSmoke(caplog):
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
    assert core._stateSignChangePerKB(
        np.asarray([1.0, -0.005, 1.0], dtype=np.float32),
        intervalSizeBP=1000,
    ) == pytest.approx(0.0)
    assert core._stateSignChangePerKB(
        np.asarray([1.0, -0.02, 1.0], dtype=np.float32),
        intervalSizeBP=1000,
    ) == pytest.approx(2.0 / 3.0)

    caplog.set_level(logging.INFO, logger=core.logger.name)
    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
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
        trackOptimizationPath=True,
        returnDiagnostics=True,
    )

    (
        stateSmoothed,
        stateCovarSmoothed,
        postFitResiduals,
        NIS,
        *_rest,
        diagnostics,
    ) = out
    assert len(out) == 6
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert diagnostics["final_forward_nis"] == pytest.approx(float(np.mean(NIS)))
    gainSummary = diagnostics["final_forward_gain_contig_summary"]
    assert len(gainSummary["mean"]) == m
    assert len(gainSummary["median"]) == m
    assert len(gainSummary["sd"]) == m
    assert len(gainSummary["iqr"]) == m
    assert all(value >= 0.0 for value in gainSummary["sd"])
    assert all(value >= 0.0 for value in gainSummary["iqr"])
    assert "background_prior" not in diagnostics
    assert "PHASE: CORE START" in caplog.text
    assert "PHASE: POST-PROCESS-NOISE FIT" in caplog.text
    assert "      | PHASE: POST-PROCESS-NOISE FIT" in caplog.text
    assert "            | PHASE: INNER ECM FIT / FIXED-BACKGROUND" in caplog.text
    assert "proc precision weights" in caplog.text
    assert "backgroundTarget[" in caplog.text
    assert "backgroundSolve[" in caplog.text
    assert "backgroundObjectivePerCell=" in caplog.text
    assert "backgroundObjectiveChangePerCell=" in caplog.text
    assert "backgroundObjectiveThresholdPerCell=" in caplog.text
    assert "qLevelPriorMode=" in caplog.text
    assert "qLevelDataEstimate=" in caplog.text
    assert "qLevelPriorDf=" in caplog.text
    assert "ratioPriorDf=" in caplog.text
    assert "lambdaMean=" in caplog.text
    assert "lambdaMedian=" in caplog.text
    assert "signChangePerKB=" in caplog.text
    assert "lambdaLowerBoundHits=" in caplog.text
    assert "lambdaUpperBoundHits=" in caplog.text
    assert "kappaLowerBoundHits=" in caplog.text
    assert "kappaUpperBoundHits=" in caplog.text
    assert "PHASE: POST-PROCESS-NOISE FIT SUMMARY" in caplog.text
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
    assert (
        0.0 <= backgroundPassDiagnostics[-1]["process_kappa_lower_bound_hits"] <= 1.0
    )
    assert (
        0.0 <= backgroundPassDiagnostics[-1]["process_kappa_upper_bound_hits"] <= 1.0
    )
    assert backgroundPassDiagnostics[-1]["sign_change_per_kb"] >= 0.0
    finalFixedDiagnostics = fitDiagnostics[-1]
    assert finalFixedDiagnostics["final_fixed_background_ecm"] is True
    assert "final_abs_rel_change" in finalFixedDiagnostics
    assert "stable_iters" in finalFixedDiagnostics
    assert "patience_target" in finalFixedDiagnostics
    assert finalFixedDiagnostics["sign_change_per_kb"] >= 0.0
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
    assert qInfo["processNoisePolicy"] == "EB_MAP"
    assert qInfo["blockMode"] == "non_overlapping"
    assert diagnostics["process_q_calibration"] == qInfo
    assert qInfo["qLevel"] > 0.0
    assert qInfo["qTrend"] > 0.0

    outChangedKnobs = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_fixedBackgroundIters=3,
        ECM_outerIters=2,
        regularizationStrength=0.0,
        regularizationRatio=0.9,
        processNoiseWarmupECMIters=1,
        returnDiagnostics=True,
    )
    changedInfo = outChangedKnobs[-1]["process_noise_calibration"]
    assert changedInfo["processNoisePolicy"] == "EB_MAP"
    assert changedInfo["qTrend"] > 0.0


@pytest.mark.correctness
def _caseRunConsenrichFlatWarmupInitializerDoesNotUseMinQ():
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
    assert qInfo["processNoisePolicy"] == "EB_MAP"
    assert qInfo["qLevel"] >= core.PROCESS_NOISE_NUMERICAL_FLOOR
    assert qInfo["qLevel"] < 5.0e-2
    assert qInfo["matrixQ0Final"][0][0] == pytest.approx(qInfo["qLevel"])


@pytest.mark.correctness
def _caseRunConsenrichWarnsWhenNonnegativeBackgroundIsZeroCentered(caplog):
    n = 16
    m = 2
    grid = np.linspace(0.0, 1.0, n, dtype=np.float32)
    matrixData = np.vstack([grid, grid + 0.05]).astype(np.float32)
    matrixMunc = np.full((m, n), 0.2, dtype=np.float32)

    caplog.set_level(logging.WARNING, logger=core.logger.name)
    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=4,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        fitBackground=True,
        useNonnegativeBackground=True,
        ECM_zeroCenterBackground=True,
        initialProcessQ=np.diag([1.0e-3, 1.0e-5]).astype(np.float32),
        returnBackground=True,
        returnDiagnostics=True,
    )
    background = out[-2]

    assert "penalizing negative values may pull the shared background toward zero" in caplog.text
    assert "only the zero background feasible" not in caplog.text
    assert np.isfinite(background).all()
    assert abs(float(np.mean(background))) < 1.0e-5

    caplog.clear()
    core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=0.1,
        minQ=1.0e-3,
        maxQ=1.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=4,
        ECM_fixedBackgroundIters=1,
        ECM_outerIters=1,
        ECM_minOuterIters=1,
        fitBackground=True,
        useNonnegativeBackground=True,
        backgroundNegativePenaltyMultiplier=None,
        ECM_zeroCenterBackground=True,
        initialProcessQ=np.diag([1.0e-3, 1.0e-5]).astype(np.float32),
        returnDiagnostics=True,
    )
    assert "penalizing negative values may pull the shared background toward zero" not in caplog.text


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
        regularizationRatio=0.5,
        processNoiseWarmupECMIters=1,
        returnDiagnostics=True,
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, _blockMap, runDiagnostics = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.all(np.isfinite(stateCovarSmoothed))
    assert np.all(np.isfinite(NIS))
    np.testing.assert_array_equal(stateSmoothed[:, 1], np.zeros(n, dtype=np.float32))
    np.testing.assert_array_equal(stateCovarSmoothed[:, 0, 1], np.zeros(n, dtype=np.float32))
    np.testing.assert_array_equal(stateCovarSmoothed[:, 1, 0], np.zeros(n, dtype=np.float32))
    np.testing.assert_array_equal(stateCovarSmoothed[:, 1, 1], np.zeros(n, dtype=np.float32))
    np.testing.assert_array_equal(
        core.getPrimaryState(stateSmoothed),
        np.round(stateSmoothed[:, 0].astype(np.float32), decimals=4),
    )
    qInfo = runDiagnostics["process_noise_calibration"]
    assert runDiagnostics["state_model"] == core.STATE_MODEL_LEVEL
    assert qInfo["processNoisePolicy"] == "EB_MAP"
    assert qInfo["qLevel"] > 0.0
    assert qInfo["qTrend"] == pytest.approx(0.0)
    assert qInfo["effectiveTrendLevelRatio"] == pytest.approx(0.0)
    assert qInfo["ratioPriorDfUsed"] == pytest.approx(0.0)
    assert qInfo["q11PaddingOnly"] is True


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
    assert (
        len(ecmModes[:firstPostQIndex])
        >= core.PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES
    )
    assert all(mode == (False, False) for mode in ecmModes[:firstPostQIndex])
    assert all(mode == (True, False) for mode in ecmModes[firstPostQIndex:])
    assert all(flag is False for flag in ecmLogIterations[:firstPostQIndex])
    assert any(flag is True for flag in ecmLogIterations[firstPostQIndex:])
    assert ecmLogIterations[-1] is False
    stateSmoothed, stateCovarSmoothed, *_ = out
    diagnostics = out[-1]
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.all(np.isfinite(stateCovarSmoothed))
    assert (
        diagnostics["process_noise_warmup_fit"]["actual_outer_passes"]
        == core.PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES
    )
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
    assert diagnostics["process_noise_calibration"]["processNoisePolicy"] == "EB_MAP"
    assert diagnostics["process_noise_calibration"]["qLevel"] >= (
        core.PROCESS_NOISE_NUMERICAL_FLOOR
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
    assert diagnostics["process_noise_calibration"]["warmStartProcessNoise"] == 1.0
    assert diagnostics["post_process_noise_fit"]["warm_start"]["background"] is False


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
def _casePSplineCythonEvaluationMatchesDenseDesign():
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
def _casePooledMuncTrendRecoversReplicateVarianceFactors():
    rng = np.random.default_rng(2025)
    meansBase = np.linspace(-10.0, 10.0, 300, dtype=np.float64)
    factorsTrue = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    means = np.tile(meansBase, factorsTrue.size)
    sampleIndex = np.repeat(np.arange(factorsTrue.size), meansBase.size)
    x = np.sign(means) * np.log1p(np.abs(means))
    sharedVariance = np.exp(-0.7 + 0.25 * np.sin(2.0 * x) + 0.08 * x)
    blockVariances = (
        sharedVariance
        * factorsTrue[sampleIndex]
        * np.exp(rng.normal(0.0, 0.03, means.size))
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

    logMae = np.mean(np.abs(np.log(pooled.replicateVarianceFactors) - np.log(factorsTrue)))
    assert logMae < 0.08
    assert pooled.diagnostics["predictor"] == "signed_log1p"
    assert pooled.diagnostics["iterations"] <= 3


@pytest.mark.correctness
def _caseReplicateMuncPriorsDifferAndProcessMatchesSerial(tmp_path):
    meansBase = np.linspace(-8.0, 8.0, 160, dtype=np.float64)
    means = np.tile(meansBase, 2)
    sampleIndex = np.repeat([0, 1], meansBase.size)
    chromIndex = np.tile(np.repeat([0, 1], meansBase.size // 2), 2)
    starts = np.tile(np.arange(meansBase.size, dtype=np.int64), 2)
    x = np.sign(means) * np.log1p(np.abs(means))
    variances = np.empty_like(means)
    variances[sampleIndex == 0] = np.exp(-0.4 + 0.35 * x[sampleIndex == 0])
    variances[sampleIndex == 1] = np.exp(0.5 - 0.25 * x[sampleIndex == 1])

    common = dict(
        chromosomeIndex=chromIndex,
        blockStarts=starts,
        sampleCount=2,
        eps=1.0e-8,
        trendNumBasis=18,
        trendMinObsPerBasis=5.0,
        trendMinEdf=2.0,
        trendMaxEdf=12.0,
        trendLambdaGridSize=13,
        EB_setNuL=8,
        localWindowIntervals=11,
        thinBinSize=7,
    )
    serial = core.fitReplicateMuncVariancePriors(
        means,
        variances,
        sampleIndex,
        workers=1,
        **common,
    )
    process = core.fitReplicateMuncVariancePriors(
        means,
        variances,
        sampleIndex,
        workers=2,
        memmapDir=str(tmp_path),
        **common,
    )

    probe = np.array([-5.0, 0.0, 5.0], dtype=np.float64)
    serialPred0 = core.evalPSplineLogVarianceTrend(serial[0].trend, probe, eps=1.0e-8)
    serialPred1 = core.evalPSplineLogVarianceTrend(serial[1].trend, probe, eps=1.0e-8)
    processPred0 = core.evalPSplineLogVarianceTrend(process[0].trend, probe, eps=1.0e-8)
    processPred1 = core.evalPSplineLogVarianceTrend(process[1].trend, probe, eps=1.0e-8)

    assert len(serial) == 2
    assert not np.allclose(serialPred0, serialPred1)
    np.testing.assert_allclose(processPred0, serialPred0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(processPred1, serialPred1, rtol=0.0, atol=0.0)
    assert all(np.isfinite(prior.Nu_0) and prior.Nu_0 >= 4.0 for prior in serial)


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
def _casePSplineTrendSummaryLogsRelationship():
    amplitudes = np.linspace(0.0, 5.0, 30, dtype=np.float64)
    variances = np.exp(0.1 + 0.05 * amplitudes)
    trend = core.fitPSplineLogVarianceTrend(
        amplitudes,
        variances,
        trendNumBasis=8,
        trendMinObsPerBasis=3.0,
        trendLambdaGridSize=11,
        eps=1.0e-8,
    )

    summary = core._formatPSplineTrendSummary(
        trend,
        amplitudes,
        eps=1.0e-8,
        maxVariance=100.0,
        pointCount=5,
    )

    assert "MUNC P-spline signed-mean-SD trend" in summary
    assert "signed_mean->sd[" in summary
    assert "basis=" in summary
    assert "lambda=" in summary
    assert "edf_cap=" in summary
    assert "->" in summary


@pytest.mark.correctness
def _caseMuncVarianceDiagnosticsLogLocalGlobalFinalAndTailSupport():
    local = np.array([0.01, 0.04, 0.09, 0.16, np.nan], dtype=np.float64)
    global_ = np.array([0.04, 0.09, 0.16, 0.25, 0.36], dtype=np.float64)
    final = np.array([0.0225, 0.0625, 0.1225, 0.2025], dtype=np.float64)
    support = np.concatenate(
        [
            np.linspace(0.0, 1.0, 100, dtype=np.float64),
            np.array([2.0, 3.0, 4.0], dtype=np.float64),
        ]
    )

    summary = core._formatMuncVarianceDiagnostics(local, global_, final, support)

    assert "MUNC variance SD diagnostics" in summary
    assert "L[n=4" in summary
    assert "G[n=5" in summary
    assert "V0[n=4" in summary
    assert "p50=" in summary
    assert "tail_support(abs_signed_mean)" in summary
    assert "q95=" in summary
    assert "q99=" in summary
    assert "max=4" in summary


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
    n = 50
    globalVars = np.ones(n, dtype=np.float64)
    localVars = np.ones(n, dtype=np.float64)
    nonThinned = (np.arange(n) % 5) != 0
    localVars[nonThinned] = np.exp(
        np.where(np.arange(n)[nonThinned] % 2 == 0, 1.0, -1.0)
    )

    nuAll = core.EB_computePriorStrength(
        localVars, globalVars, Nu_local=10.0, thinStride=1
    )
    nuThinned = core.EB_computePriorStrength(
        localVars, globalVars, Nu_local=10.0, thinStride=5
    )

    assert nuThinned > 10.0 * nuAll


@pytest.mark.correctness
def _casePooledPriorStrengthThinsBySampleChromosomeAndBin(
    monkeypatch: pytest.MonkeyPatch,
):
    n = 12
    localVars = np.linspace(1.0, 2.0, n, dtype=np.float64)
    globalVars = np.ones(n, dtype=np.float64)
    sampleIndex = np.repeat([0, 1], 6)
    chromosomeIndex = np.tile(np.repeat([0, 1], 3), 2)
    blockStarts = np.tile(np.array([0, 4, 8, 0, 4, 8], dtype=np.int64), 2)
    seen: dict[str, np.ndarray] = {}

    def _fakeCompute(localArr, globalArr, _nuLocal, candidateIdx):
        seen["candidate_idx"] = np.asarray(candidateIdx, dtype=np.intp)
        return 17.0

    monkeypatch.setattr(core, "_computePriorStrengthFromCandidateIdx", _fakeCompute)

    nu0 = core.EB_computePooledPriorStrength(
        localVars,
        globalVars,
        Nu_local=8.0,
        sampleIndex=sampleIndex,
        chromosomeIndex=chromosomeIndex,
        blockStarts=blockStarts,
        thinBinSize=10,
    )

    assert nu0 == pytest.approx(17.0)
    assert seen["candidate_idx"].tolist() == [0, 3, 6, 9]


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
def _caseGetMuncTrackSparseNearestPath(monkeypatch: pytest.MonkeyPatch):
    intervals = np.arange(0, 400, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.6, intervals.size, dtype=np.float32)
    sparseIntervalIndices = np.array([1, 2, 3, 4, 9, 10, 11, 12], dtype=np.intp)
    fakeVarTrack = np.linspace(0.2, 0.5, intervals.size, dtype=np.float32)
    fakeRollingVarTrack = np.linspace(0.3, 0.7, intervals.size, dtype=np.float32)
    fakeBlockMeans = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    fakeBlockVars = np.linspace(0.2, 0.6, 8, dtype=np.float32)
    seen: dict[str, np.ndarray] = {}

    def _fakeSparseNearest(*args, **kwargs):
        seen["sparse_centers"] = np.asarray(args[1]).copy()
        seen["block_starts"] = np.asarray(args[2]).copy()
        seen["block_sizes"] = np.asarray(args[3]).copy()
        seen["sparse_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        return (
            np.zeros(intervals.size, dtype=np.float32),
            fakeVarTrack.copy(),
        )

    def _fakeRolling(*args, **kwargs):
        seen["rolling_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        return fakeRollingVarTrack.copy()

    def _fakeMeanVarPairs(*args, **kwargs):
        seen["block_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    def _fakeFitPSplineLogVarianceTrend(*args, **kwargs):
        return {"slope": 1.0}

    def _fakeEvalPSplineLogVarianceTrend(opt, meanTrack, *args, **kwargs):
        meanTrackArr = np.asarray(meanTrack, dtype=np.float32)
        return np.maximum(0.25, meanTrackArr + 0.1).astype(np.float32)

    def _fakeEBComputePriorStrength(localVars, _priorVars, _nuLocal, **kwargs):
        seen["local_vars"] = np.asarray(localVars).copy()
        seen["thin_stride"] = np.asarray([kwargs.get("thinStride", 1)], dtype=np.int64)
        return 10.0

    monkeypatch.setattr(
        cconsenrich,
        "cSparseNearestMeanVarTrack",
        _fakeSparseNearest,
    )
    monkeypatch.setattr(
        cconsenrich,
        "crollingMuncVariance",
        _fakeRolling,
    )
    monkeypatch.setattr(
        cconsenrich,
        "cmeanVarPairs",
        _fakeMeanVarPairs,
    )
    monkeypatch.setattr(
        core, "fitPSplineLogVarianceTrend", _fakeFitPSplineLogVarianceTrend
    )
    monkeypatch.setattr(
        core, "evalPSplineLogVarianceTrend", _fakeEvalPSplineLogVarianceTrend
    )
    monkeypatch.setattr(core, "EB_computePriorStrength", _fakeEBComputePriorStrength)

    muncTrack, support = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        samplingBlockSizeBP=125,
        samplingIters=64,
        sparseIntervalIndices=sparseIntervalIndices,
        numNearest=3,
        sparseSupportScaleBP=50,
        sparseSupportPrior=1.0,
        EB_localQuantile=-1.0,
        EB_use=True,
    )

    supportWeights = core._sparseSupportWeights(
        sparseIntervalIndices,
        intervals.size,
        ellIntervals=2.0,
        supportPrior=1.0,
    )
    expectedLocalVars = (
        supportWeights * fakeVarTrack.astype(np.float64)
        + (1.0 - supportWeights) * fakeRollingVarTrack.astype(np.float64)
    )
    sparseSet = set(sparseIntervalIndices.tolist())
    for blockStart, blockSize in zip(seen["block_starts"], seen["block_sizes"]):
        blockRange = range(int(blockStart), int(blockStart + blockSize))
        assert all(idx in sparseSet for idx in blockRange)

    assert np.array_equal(seen["sparse_centers"], sparseIntervalIndices)
    assert bool(seen["sparse_use_innovation_var"][0]) is False
    assert bool(seen["rolling_use_innovation_var"][0]) is False
    assert bool(seen["block_use_innovation_var"][0]) is False
    assert np.allclose(seen["local_vars"], expectedLocalVars)
    assert int(seen["thin_stride"][0]) == 6
    assert muncTrack.shape == values.shape
    assert np.isfinite(muncTrack).all()
    assert support > 0.0


@pytest.mark.correctness
def _caseGetMuncTrackClipsHugePriorBeforeShrinkage(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.2, intervals.size, dtype=np.float32)
    fakeBlockMeans = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    fakeBlockVars = np.linspace(0.2, 0.6, 8, dtype=np.float32)
    fakeRollingVarTrack = np.linspace(0.05, 0.2, intervals.size, dtype=np.float32)
    seen: dict[str, np.ndarray] = {}

    def _fakeMeanVarPairs(*args, **kwargs):
        seen["block_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    def _fakeRolling(*args, **kwargs):
        seen["rolling_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        return fakeRollingVarTrack.copy()

    def _fakeFitPSplineLogVarianceTrend(*args, **kwargs):
        return {"huge": True}

    def _fakeEvalPSplineLogVarianceTrend(*args, **kwargs):
        return np.full(values.shape, 1.0e100, dtype=np.float64)

    monkeypatch.setattr(cconsenrich, "cmeanVarPairs", _fakeMeanVarPairs)
    monkeypatch.setattr(cconsenrich, "crollingMuncVariance", _fakeRolling)
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
        samplingBlockSizeBP=125,
        samplingIters=64,
        EB_localQuantile=-1.0,
        EB_use=True,
        varianceCap=0.75,
    )

    assert np.all(np.isfinite(muncTrack))
    assert np.all(muncTrack <= np.float32(0.75))
    assert bool(seen["rolling_use_innovation_var"][0]) is False
    assert bool(seen["block_use_innovation_var"][0]) is False


@pytest.mark.correctness
def _caseGetMuncTrackCapsPriorStrengthAtFiftyTimesLocalDf(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(0.1, 1.2, intervals.size, dtype=np.float32)
    localVarTrack = np.full(intervals.size, 1.0, dtype=np.float32)
    priorVarTrack = np.full(intervals.size, 0.01, dtype=np.float32)
    fakeBlockMeans = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    fakeBlockVars = np.linspace(0.2, 0.6, 8, dtype=np.float32)

    def _fakeMeanVarPairs(*args, **kwargs):
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    monkeypatch.setattr(cconsenrich, "cmeanVarPairs", _fakeMeanVarPairs)
    monkeypatch.setattr(
        cconsenrich,
        "crollingMuncVariance",
        lambda *args, **kwargs: localVarTrack.copy(),
    )
    monkeypatch.setattr(
        core,
        "fitPSplineLogVarianceTrend",
        lambda *args, **kwargs: {"flat": True},
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
        samplingBlockSizeBP=125,
        samplingIters=64,
        EB_localQuantile=-1.0,
        EB_setNuL=10,
        EB_use=True,
        varianceFloor=0.0,
        varianceCap=10.0,
    )

    expected = (10.0 * localVarTrack + 500.0 * priorVarTrack) / 510.0

    assert np.allclose(muncTrack, expected.astype(np.float32))


@pytest.mark.correctness
def _caseGetMuncTrackUsesSuppliedPooledTrendFactorAndBoundaryNu0(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 300, 25, dtype=np.uint32)
    values = np.linspace(-1.0, 1.0, intervals.size, dtype=np.float32)
    localVarTrack = np.full(intervals.size, 9.0, dtype=np.float32)
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

    def _fakeMeanVarPairs(*args, **kwargs):
        pytest.fail("pooled trend should not resample block pairs")

    monkeypatch.setattr(cconsenrich, "cmeanVarPairs", _fakeMeanVarPairs)
    monkeypatch.setattr(
        cconsenrich,
        "crollingMuncVariance",
        lambda *args, **kwargs: localVarTrack.copy(),
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

    muncTrack, _ = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        samplingBlockSizeBP=125,
        samplingIters=64,
        EB_localQuantile=-1.0,
        EB_setNuL=10,
        EB_use=True,
        pooledTrend=pooledTrend,
        replicateVarianceFactor=3.0,
        EB_pooledNu0=4.0,
        varianceFloor=0.0,
        varianceCap=20.0,
    )

    expected = (10.0 * localVarTrack + 4.0 * np.float32(6.0)) / 14.0
    assert np.allclose(muncTrack, expected.astype(np.float32))


@pytest.mark.correctness
def _caseMeanVarPairSampleSizesStayWithinTrackLength():
    intervals = np.arange(1024, dtype=np.uint32)
    values = np.linspace(-0.5, 0.5, intervals.size, dtype=np.float32)
    excludeMask = np.zeros(intervals.size, dtype=np.uint8)

    means, variances, starts, ends = cconsenrich.cmeanVarPairs(
        intervals,
        values,
        900,
        256,
        42,
        excludeMask,
    )

    assert means.shape == (256,)
    assert variances.shape == (256,)
    assert starts.shape == ends.shape == (256,)
    assert np.all(starts >= 0)
    assert np.all(ends <= values.size)
    assert np.all(ends > starts)


@pytest.mark.correctness
def _caseRollingAR1CanReturnMarginalVarianceInsteadOfInnovation():
    rng = np.random.default_rng(44)
    n = 4096
    phi = 0.8
    values = np.zeros(n, dtype=np.float64)
    innovations = rng.normal(scale=np.sqrt(1.0 - phi * phi), size=n)
    values[0] = rng.normal()
    for i in range(1, n):
        values[i] = phi * values[i - 1] + innovations[i]

    excludeMask = np.zeros(n, dtype=np.uint8)
    innovationVar = cconsenrich.crolling_AR1_IVar(
        values.astype(np.float32),
        201,
        excludeMask,
        useInnovationVar=True,
    )
    marginalVar = cconsenrich.crolling_AR1_IVar(
        values.astype(np.float32),
        201,
        excludeMask,
        useInnovationVar=False,
    )

    assert float(np.median(marginalVar)) > 1.8 * float(np.median(innovationVar))
    assert float(np.median(marginalVar)) == pytest.approx(1.0, rel=0.25)


@pytest.mark.correctness
def _caseMuncSizingAndCythonVarianceModels():
    intervalSizeBP = 25
    legacySizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceContextBP=1000,
        samplingBlockSizeBP=125,
        muncTrendBlockSizeBP=None,
        muncLocalWindowSizeBP=None,
    )
    assert legacySizing.usedLegacySamplingBlockSize is True
    assert legacySizing.localWindowIntervals == legacySizing.trendBlockIntervals + 1

    explicitSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceContextBP=1000,
        samplingBlockSizeBP=125,
        muncTrendBlockSizeBP=250,
        muncLocalWindowSizeBP=500,
    )
    assert explicitSizing.usedLegacySamplingBlockSize is False
    assert explicitSizing.trendBlockSizeBP == 250
    assert explicitSizing.localWindowSizeBP == 500

    dependenceSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceContextBP=None,
        dependenceSpanIntervals=17,
        samplingBlockSizeBP=None,
        muncTrendBlockSizeBP=None,
        muncLocalWindowSizeBP=None,
        muncTrendBlockDependenceMultiplier=1.5,
        muncLocalWindowDependenceMultiplier=2.5,
    )
    assert dependenceSizing.usedDependenceSpan is True
    assert dependenceSizing.dependenceSpanIntervals == 17
    assert dependenceSizing.trendBlockIntervals == 26
    assert dependenceSizing.localWindowIntervals == 43

    rng = np.random.default_rng(123)
    n = 2048
    trend = np.linspace(-4.0, 4.0, n, dtype=np.float64)
    values = (trend + rng.normal(scale=0.1, size=n)).astype(np.float32)
    excludeMask = np.zeros(n, dtype=np.uint8)
    svarTrack = cconsenrich.crollingMuncVariance(
        values,
        101,
        excludeMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR,
    )
    ar1Track = cconsenrich.crollingMuncVariance(
        values,
        101,
        excludeMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_AR1,
        useInnovationVar=False,
    )
    firstDifferenceTrack = cconsenrich.crollingMuncVariance(
        values,
        101,
        excludeMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR_D1,
    )
    secondDifferenceTrack = cconsenrich.crollingMuncVariance(
        values,
        101,
        excludeMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR_D2,
    )

    assert np.all(np.isfinite(svarTrack))
    assert np.all(np.isfinite(ar1Track))
    assert np.all(np.isfinite(firstDifferenceTrack))
    assert np.all(np.isfinite(secondDifferenceTrack))
    assert float(np.median(firstDifferenceTrack)) < float(np.median(svarTrack))
    assert float(np.median(secondDifferenceTrack)) < float(np.median(svarTrack))

    referenceValues = (
        np.linspace(-0.5, 0.7, 96, dtype=np.float64)
        + rng.normal(scale=0.03, size=96)
    ).astype(np.float32)
    referenceMask = np.zeros(referenceValues.size, dtype=np.uint8)
    windowLength = 17
    fastFirstDifferenceTrack = cconsenrich.crollingMuncVariance(
        referenceValues,
        windowLength,
        referenceMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR_D1,
    )
    fastSecondDifferenceTrack = cconsenrich.crollingMuncVariance(
        referenceValues,
        windowLength,
        referenceMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR_D2,
    )
    starts = np.clip(
        np.arange(referenceValues.size) - (windowLength // 2),
        0,
        referenceValues.size - windowLength,
    )

    referenceFirstDifferenceTrack = np.empty(referenceValues.size, dtype=np.float64)
    for outIndex, start in enumerate(starts):
        window = referenceValues[start : start + windowLength].astype(np.float64)
        windowDiff = np.diff(window)
        referenceFirstDifferenceTrack[outIndex] = np.sum(windowDiff * windowDiff) / (
            2.0 * windowDiff.size
        )
    np.testing.assert_allclose(
        fastFirstDifferenceTrack,
        referenceFirstDifferenceTrack,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    referenceSecondDifferenceTrack = np.empty(referenceValues.size, dtype=np.float64)
    for outIndex, start in enumerate(starts):
        window = referenceValues[start : start + windowLength].astype(np.float64)
        windowDiff = np.diff(window, n=2)
        referenceSecondDifferenceTrack[outIndex] = np.sum(windowDiff * windowDiff) / (
            6.0 * windowDiff.size
        )
    np.testing.assert_allclose(
        fastSecondDifferenceTrack,
        referenceSecondDifferenceTrack,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    maskedReferenceMask = referenceMask.copy()
    maskedReferenceMask[20] = 1
    fastMaskedSecondDifferenceTrack = cconsenrich.crollingMuncVariance(
        referenceValues,
        windowLength,
        maskedReferenceMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR_D2,
    )
    referenceMaskedSecondDifferenceTrack = np.empty(referenceValues.size, dtype=np.float64)
    for outIndex, start in enumerate(starts):
        window = referenceValues[start : start + windowLength].astype(np.float64)
        windowMask = maskedReferenceMask[start : start + windowLength].astype(bool)
        validDiff = ~(windowMask[:-2] | windowMask[1:-1] | windowMask[2:])
        windowDiff = np.diff(window, n=2)[validDiff]
        referenceMaskedSecondDifferenceTrack[outIndex] = (
            np.sum(windowDiff * windowDiff) / (6.0 * windowDiff.size)
            if windowDiff.size
            else 0.0
        )
    np.testing.assert_allclose(
        fastMaskedSecondDifferenceTrack,
        referenceMaskedSecondDifferenceTrack,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    blockMeans, blockVars, blockStarts, blockEnds = cconsenrich.cmeanVarPairs(
        np.arange(referenceValues.size, dtype=np.uint32),
        referenceValues,
        windowLength,
        16,
        123,
        referenceMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR_D1,
    )
    assert np.all(np.isfinite(blockMeans))
    assert np.all(np.isfinite(blockVars))
    for blockVar, blockStart, blockEnd in zip(blockVars, blockStarts, blockEnds):
        block = referenceValues[blockStart:blockEnd].astype(np.float64)
        blockDiff = np.diff(block)
        referenceBlockVar = np.sum(blockDiff * blockDiff) / (2.0 * blockDiff.size)
        assert float(blockVar) == pytest.approx(float(referenceBlockVar), rel=1.0e-6)

    blockMeans, blockVars, blockStarts, blockEnds = cconsenrich.cmeanVarPairs(
        np.arange(referenceValues.size, dtype=np.uint32),
        referenceValues,
        windowLength,
        16,
        123,
        referenceMask,
        modelCode=core.MUNC_VARIANCE_MODEL_CODE_SVAR_D2,
    )
    assert np.all(np.isfinite(blockMeans))
    assert np.all(np.isfinite(blockVars))
    for blockMean, blockVar, blockStart, blockEnd in zip(
        blockMeans,
        blockVars,
        blockStarts,
        blockEnds,
    ):
        block = referenceValues[blockStart:blockEnd].astype(np.float64)
        blockDiff = np.diff(block, n=2)
        referenceBlockVar = np.sum(blockDiff * blockDiff) / (6.0 * blockDiff.size)
        assert float(blockMean) == pytest.approx(float(np.mean(block)), rel=1.0e-6)
        assert float(blockVar) == pytest.approx(float(referenceBlockVar), rel=1.0e-6)


@pytest.mark.correctness
def _caseGetMuncTrackSparseNearestDetrendsPriorBySignedLocalIntercept(
    monkeypatch: pytest.MonkeyPatch,
):
    intervals = np.arange(0, 400, 25, dtype=np.uint32)
    values = np.linspace(-0.4, 1.1, intervals.size, dtype=np.float32)
    sparseIntervalIndices = np.array([1, 2, 3, 4, 9, 10, 11, 12], dtype=np.intp)
    sparseMeanTrack = np.linspace(-0.2, 0.3, intervals.size, dtype=np.float32)
    sparseVarTrack = np.linspace(0.2, 0.5, intervals.size, dtype=np.float32)
    fakeRollingVarTrack = np.linspace(0.3, 0.7, intervals.size, dtype=np.float32)
    fakeBlockMeans = np.linspace(0.1, 1.0, 8, dtype=np.float32)
    fakeBlockVars = np.linspace(0.2, 0.6, 8, dtype=np.float32)
    seen: dict[str, np.ndarray] = {}

    def _fakeSparseNearest(*args, **kwargs):
        seen["sparse_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        return sparseMeanTrack.copy(), sparseVarTrack.copy()

    def _fakeRolling(*args, **kwargs):
        seen["rolling_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        return fakeRollingVarTrack.copy()

    def _fakeMeanVarPairs(_intervalsArg, valuesArg, *args, **kwargs):
        seen["prior_fit_values"] = np.asarray(valuesArg).copy()
        seen["block_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    def _fakeEMA(meanTrackArg, alpha):
        seen["ema_input"] = np.asarray(meanTrackArg).copy()
        return np.asarray(meanTrackArg).copy()

    def _fakeFitPSplineLogVarianceTrend(*args, **kwargs):
        return {"slope": 1.0}

    def _fakeEvalPSplineLogVarianceTrend(opt, meanTrack, *args, **kwargs):
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
        "crollingMuncVariance",
        _fakeRolling,
    )
    monkeypatch.setattr(
        cconsenrich,
        "cmeanVarPairs",
        _fakeMeanVarPairs,
    )
    monkeypatch.setattr(cconsenrich, "cEMA", _fakeEMA)
    monkeypatch.setattr(
        core, "fitPSplineLogVarianceTrend", _fakeFitPSplineLogVarianceTrend
    )
    monkeypatch.setattr(
        core, "evalPSplineLogVarianceTrend", _fakeEvalPSplineLogVarianceTrend
    )

    muncTrack, support = core.getMuncTrack(
        chromosome="chrTest",
        intervals=intervals,
        values=values,
        intervalSizeBP=25,
        samplingBlockSizeBP=125,
        samplingIters=64,
        sparseIntervalIndices=sparseIntervalIndices,
        numNearest=3,
        sparseSupportScaleBP=50,
        sparseSupportPrior=1.0,
        EB_localQuantile=-1.0,
        EB_use=True,
    )

    supportWeights = core._sparseSupportWeights(
        sparseIntervalIndices,
        intervals.size,
        ellIntervals=2.0,
        supportPrior=1.0,
    )
    expectedResidual = values.astype(np.float64) - (
        supportWeights * sparseMeanTrack.astype(np.float64)
    )
    assert np.allclose(seen["prior_fit_values"], expectedResidual)
    assert np.allclose(seen["ema_input"], expectedResidual)
    assert np.allclose(seen["prior_eval_mean_track"], expectedResidual)
    assert bool(seen["sparse_use_innovation_var"][0]) is False
    assert bool(seen["rolling_use_innovation_var"][0]) is False
    assert bool(seen["block_use_innovation_var"][0]) is False
    assert muncTrack.shape == values.shape
    assert np.isfinite(muncTrack).all()
    assert support > 0.0


@pytest.mark.correctness
def _caseGetMuncTrackRestrictsRollingAR1ToSparseBed(
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
        seen["rolling_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        return fakeRollingVarTrack.copy()

    def _fakeMeanVarPairs(*args, **kwargs):
        seen["block_use_innovation_var"] = np.asarray(
            [bool(kwargs.get("useInnovationVar", True))],
            dtype=bool,
        )
        starts = np.arange(fakeBlockMeans.size, dtype=np.intp)
        ends = starts + 1
        return fakeBlockMeans.copy(), fakeBlockVars.copy(), starts, ends

    def _fakeFitPSplineLogVarianceTrend(*args, **kwargs):
        return {"slope": 1.0}

    def _fakeEvalPSplineLogVarianceTrend(opt, meanTrack, *args, **kwargs):
        meanTrackArr = np.asarray(meanTrack, dtype=np.float32)
        return np.maximum(0.25, meanTrackArr + 0.1).astype(np.float32)

    monkeypatch.setattr(
        cconsenrich,
        "crollingMuncVariance",
        _fakeRolling,
    )
    monkeypatch.setattr(
        cconsenrich,
        "cmeanVarPairs",
        _fakeMeanVarPairs,
    )
    monkeypatch.setattr(
        core, "fitPSplineLogVarianceTrend", _fakeFitPSplineLogVarianceTrend
    )
    monkeypatch.setattr(
        core, "evalPSplineLogVarianceTrend", _fakeEvalPSplineLogVarianceTrend
    )

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
    assert bool(seen["rolling_use_innovation_var"][0]) is False
    assert bool(seen["block_use_innovation_var"][0]) is False
    assert muncTrack.shape == values.shape
    assert np.isfinite(muncTrack).all()
    assert support > 0.0


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
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        ECM_useAPN=True,
        ECM_useProcessPrecisionReweighting=True,
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
    assert diagnostics["context_size_bp"] == pointSpan * 50 + 1
    assert all(1000 <= width <= 1_000_000 for width in diagnostics["sampled_width_bp"])
    assert set(diagnostics["sampled_chromosomes"]) <= {"chr1", "chr2"}
    excluded = set(diagnostics["excluded_nonstandard_chromosomes"])
    assert {"chrX", "chrY", "chrM", "chr1_alt"} <= excluded


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
def _caseReadSegmentsFragmentsDefaultToCoverage():
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
def _caseReadSegmentsFragmentsRespectModeAndMultiplicity(tmp_path):
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
        "chr1\t0\t10\t1.5\n"
        "chr1\t10\t20\t2.5\n"
        "chr1\t20\t30\t4.0\n",
        encoding="ascii",
    )

    counts = core.readSegments(
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
    )

    assert counts.shape == (1, 3)
    assert np.allclose(counts[0], np.array([3.0, 5.0, 8.0], dtype=np.float32))


@pytest.mark.correctness
def _caseBedGraphChromRangeAndReadLength(tmp_path):
    bedGraphPath = tmp_path / "range.bedGraph"
    bedGraphPath.write_text(
        "chr2\t0\t10\t9\n"
        "chr1\t25\t50\t2\n"
        "chr1\t5\t20\t3\n",
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
def _caseNormalizeCountModeRejectsLegacyAliases():
    for alias in ["cov", "cut", "cutsites", "5p", "five_prime", "centre", "midpoint"]:
        with pytest.raises(ValueError, match="Unsupported countMode"):
            core._normalizeCountMode(alias, "coverage")


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
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10:18] = 1.0
    assert np.allclose(counts[0], expected)


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
        inferFragmentLength=0,
    )

    expected = np.zeros(30, dtype=np.float32)
    expected[10:12] = 1.0
    assert np.allclose(counts[0], expected)


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
def _caseReadBamSegmentsCountEndsOnlyUsesFivePrimePositions(tmp_path):
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
        groupCellCount=cellCount,
        fragmentsGroupNorm="CELLS",
    )

    assert mappedCount == 12
    assert cellCount == 1
    assert scaleFactor == pytest.approx((1_000_000 / 12.0) * 100.0)


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


def _run_with_monkeypatch(monkeypatch, func, *args):
    with monkeypatch.context() as mp:
        return func(*args, mp)


def test_core_numeric_kernel_contracts(caplog, contract_case):
    caplog.clear()
    contract_case("ASCII phase logging", _caseAsciiPhaseLogFormattingIsCompactAndAttributed, caplog)
    for dtype in (np.float32, np.float64):
        contract_case(f"C EMA kernel {dtype}", _caseCEMAUsesSameBidirectionalKernelForFloat32AndFloat64, dtype)
        contract_case(f"mono log kernel {dtype}", _caseMonoFuncUsesSameLogKernelForFloat32AndFloat64, dtype)
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
    for label, func in (
        ("CSF odd median", _caseCSFMedianSelectionHandlesOddLengthDuplicates),
        ("CSF even median", _caseCSFMedianSelectionHandlesEvenLengthDuplicates),
        ("transform input float32", _caseCTransformWithInputReturnsFloat32LogRatio),
        ("transform input float64", _caseCTransformWithInputReturnsFloat64LogRatio),
        ("transform into output", _caseCTransformWithInputIntoWritesOutputInPlace),
        ("in-place pure log float32", _caseCTransformInPlacePureLogMutatesFloat32Array),
        (
            "in-place transform float64",
            _caseCTransformInPlaceMatchesAllocatingTransformForFloat64,
        ),
        (
            "global median center",
            _caseSubtractGlobalMedianCentersEachTrackInPlace,
        ),
        (
            "quantile detrend uncentered",
            _caseQuantileFilterSubtractsUncenteredTrendInPlace,
        ),
        (
            "quantile detrend requested q",
            _caseQuantileFilterDetrendUsesRequestedQuantile,
        ),
        ("level forward-backward kernel", _caseLevelForwardBackwardMatchesPythonReference),
    ):
        contract_case(label, func)


def test_core_existing_peak_contracts(contract_case):
    for label, func in (
        ("single-end detection", _caseSingleEndDetection),
        ("paired-end detection", _casePairedEndDetection),
        ("existing bedGraph matching", _caseMatchExistingBedGraph),
    ):
        contract_case(label, func)


def test_core_background_bias_contracts(monkeypatch, contract_case):
    for label, func, args in (
        ("zero-centered background", _caseZeroCenteredBackgroundUpdate, ()),
        ("weighted background", _caseZeroCenteredBackgroundUpdateUsesPrecisionWeights, ()),
        (
            "background sparse reference",
            _caseZeroCenteredBackgroundUpdateMatchesSparseReference,
            (),
        ),
        ("skip zero-centering", _caseBackgroundUpdateCanSkipZeroCentering, ()),
        (
            "nonnegative background update",
            _caseBackgroundUpdateCanEnforceNonnegativeConstraint,
            (),
        ),
        (
            "background penalty scaling",
            _caseBackgroundPenaltyWeightsScaleByDifferenceOrder,
            (),
        ),
        ("replicate bias is always centered", _caseReplicateBiasIsAlwaysZeroCentered, ()),
        ("replicate-bias minimizer", _caseCFixedBackgroundECMReplicateBiasUpdateMatchesPrecisionWeightedMinimizer, ()),
        (
            "replicate-bias robust center",
            _caseCFixedBackgroundECMReplicateBiasUsesFixedCenterConstraintWithRobustWeights,
            (),
        ),
        (
            "interval-level observation precision",
            _caseObservationPrecisionIsIntervalLevelOnly,
            (),
        ),
    ):
        contract_case(label, func, *args)


def test_core_state_diagnostics_and_transition_contracts(contract_case):
    for label, func in (
        ("final forward NIS", _caseFinalForwardNISUsesMeanFinalForwardDiagnostic),
        (
            "final forward gain summary",
            _caseFinalForwardGainSummaryUsesReplicateContigRows,
        ),
        ("state roughness summary", _caseSummarizeStateRoughnessUsesHoldoutBlocksAndSignalStrata),
        ("precision boundary summary", _caseSummarizePrecisionBoundaryHitsSkipsFirstProcessWeight),
        ("removed process block scale options", _caseFitParamsDropsProcBlockScaleOptions),
        ("state model normalization", _caseNormalizeStateModelAcceptsCanonicalValuesOnly),
        ("transition residual orientation", _caseExpectedTransitionResidualSumsUsesLagOrientationAndDeltaF),
        ("transition residual reference", _caseExpectedTransitionResidualSumsMatchesPythonReference),
        ("level transition residual reference", _caseExpectedLevelTransitionResidualSumsMatchesPythonReference),
        (
            "block EB process noise",
            _caseWarmupProcessNoiseCalibrationReliabilityAndLevelModel,
        ),
        ("state uncertainty coverage", _caseCheckStateUncertaintyCoverageOverallAndStrata),
        ("linear envelope removed", _caseLinearEnvelopeParameterIsAbsent),
        ("monotone pooling removed", _caseMonotonePoolingSourceSymbolsAbsent),
    ):
        contract_case(label, func)


def test_core_em_loop_contracts(monkeypatch, caplog, contract_case):
    caplog.clear()
    contract_case(
        "outer pass smoke",
        _caseRunConsenrichOuterPassSmoke,
        caplog,
    )
    contract_case(
        "flat process-noise initializer",
        _caseRunConsenrichFlatWarmupInitializerDoesNotUseMinQ,
    )
    caplog.clear()
    contract_case(
        "nonnegative zero-centered background warning",
        _caseRunConsenrichWarnsWhenNonnegativeBackgroundIsZeroCentered,
        caplog,
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
            "outer-pass minimum iterations",
            _caseRunConsenrichOuterPassRequiresThreeIterationsDespiteTolerance,
        ),
        ("ECM always runs with APN", _caseRunConsenrichAlwaysRunsECMWithAPN),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)
    contract_case("APN smoke", _caseRunConsenrichAPNSmoke)
    contract_case("level state-model smoke", _caseRunConsenrichLevelStateModelSmoke)


def test_core_pspline_sparse_support_and_trend_contracts(tmp_path, contract_case):
    contract_case("BED mask interval overlap", _caseGetBedMaskUsesIntervalSpanOverlap, tmp_path)
    for label, func in (
        ("sparse support decay", _caseSparseSupportWeightsUseExponentialDistanceDecay),
        ("nonmonotone P-spline trend", _casePSplineLogVarianceTrendRecoversNonmonotoneShape),
        ("signed P-spline predictor", _casePSplineSignedPredictorDistinguishesPositiveAndNegativeMeans),
        ("P-spline boundary clamp", _casePSplinePredictionClampsToTrainingBoundary),
        ("P-spline Cython eval", _casePSplineCythonEvaluationMatchesDenseDesign),
        ("P-spline basis support limit", _casePSplineLimitsBasisCountByWeightedSupport),
        ("pooled MUNC trend factors", _casePooledMuncTrendRecoversReplicateVarianceFactors),
        ("P-spline guarded GCV", _casePSplineGuardedGCVAppliesEdfCap),
        ("P-spline quantile knots", _casePSplineUsesQuantileKnotsFromSupport),
        ("P-spline float32 clipping", _casePSplinePredictionClipsBeforeFloat32Overflow),
        ("P-spline trend logging", _casePSplineTrendSummaryLogsRelationship),
        ("MUNC variance diagnostics", _caseMuncVarianceDiagnosticsLogLocalGlobalFinalAndTailSupport),
    ):
        contract_case(label, func)
    contract_case(
        "replicate MUNC priors",
        _caseReplicateMuncPriorsDifferAndProcessMatchesSerial,
        tmp_path,
    )


def test_core_eb_prior_and_munc_contracts(monkeypatch, contract_case):
    for label, func in (
        ("EB prior strength boundary", _caseEBPriorStrengthBoundaryIsUsable),
        ("EB prior thinned pairs", _caseEBPriorStrengthUsesThinnedVariancePairs),
        ("mean-var sample sizes", _caseMeanVarPairSampleSizesStayWithinTrackLength),
        ("rolling AR1 marginal variance", _caseRollingAR1CanReturnMarginalVarianceInsteadOfInnovation),
        ("MUNC sizing and Cython variance models", _caseMuncSizingAndCythonVarianceModels),
        ("blacklist MUNC floor", _caseApplyBlacklistMuncFloorUsesNonBlacklistQuantile),
    ):
        contract_case(label, func)
    for label, func in (
        ("pooled prior thinning", _casePooledPriorStrengthThinsBySampleChromosomeAndBin),
        ("sparse-nearest MUNC path", _caseGetMuncTrackSparseNearestPath),
        ("huge prior clipping", _caseGetMuncTrackClipsHugePriorBeforeShrinkage),
        ("prior strength cap", _caseGetMuncTrackCapsPriorStrengthAtFiftyTimesLocalDf),
        ("pooled trend factor", _caseGetMuncTrackUsesSuppliedPooledTrendFactorAndBoundaryNu0),
        ("sparse-nearest detrended prior", _caseGetMuncTrackSparseNearestDetrendsPriorBySignedLocalIntercept),
        ("restrict rolling AR1 to sparse BED", _caseGetMuncTrackRestrictsRollingAR1ToSparseBed),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)


def test_core_dependence_selection_contracts(contract_case):
    contract_case(
        "feature length bootstrap/context compatibility",
        _caseChooseFeatureLengthBootstrapWidthVarianceAndContextCompat,
    )
    contract_case(
        "sampled dependence span autosome diagnostics",
        _caseChooseDependenceSpanSamplesAutosomesAndReportsDiagnostics,
    )
    contract_case(
        "chrom sizes preserve sex chromosomes",
        _caseChromSizesKeepSexChromosomesByDefault,
    )


def test_core_fragments_io_contracts(tmp_path, contract_case):
    for label, func, args in (
        ("fragments grouped", _caseReadSegmentsFragmentsGrouped, ()),
        ("fragments default coverage", _caseReadSegmentsFragmentsDefaultToCoverage, ()),
        ("fragments mode and multiplicity", _caseReadSegmentsFragmentsRespectModeAndMultiplicity, (tmp_path,)),
        ("fragments mapped count", _caseFragmentsMappedCountUsesEmittedInsertionsAndSelectedCells, ()),
    ):
        contract_case(label, func, *args)


def test_core_bedgraph_counting_contracts(tmp_path, contract_case):
    for label, func in (
        ("bedGraph weighted overlap", _caseCCountsCountBedGraphRegionWeightedByOverlap),
        ("indexed bedGraph weighted overlap", _caseCCountsCountIndexedBedGraphRegionWeightedByOverlap),
        ("bedGraph native scaling", _caseReadSegmentsBedGraphScalesNativeCounts),
        ("bedGraph range and read length", _caseBedGraphChromRangeAndReadLength),
    ):
        contract_case(label, func, tmp_path)


def test_core_bam_counting_contracts(tmp_path, contract_case):
    contract_case("count mode legacy aliases rejected", _caseNormalizeCountModeRejectsLegacyAliases)
    for label, func in (
        ("paired BAM template span", _caseReadSegmentsBamPairedEndUsesTemplateSpan),
        ("paired BAM read1 single-end mode", _caseReadSegmentsPairedBamCanUseRead1OnlySingleEndMode),
        ("BAM count-ends 5-prime positions", _caseReadSegmentsBamCountEndsOnlyUsesFivePrimePositions),
        ("direct BAM count-ends 5-prime positions", _caseReadBamSegmentsCountEndsOnlyUsesFivePrimePositions),
    ):
        contract_case(label, func, tmp_path)


def test_core_pair_scale_factor_contracts(monkeypatch, contract_case):
    for label, func in (
        ("MACS treatment downscale", _casePairScaleFactorsDownscaleDeeperSampleMacs),
        ("control downscale by default", _casePairScaleFactorsCanDownscaleControlByDefault),
        ("fixed control keeps full depth", _casePairScaleFactorsFixControlKeepsControlFullDepth),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)
