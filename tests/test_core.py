# -*- coding: utf-8 -*-

# the name of this file is unfortunately misleading --
# 'core' as in 'central' or 'basic', not the literal
# `consenrich.core` module

import math
import os
import inspect
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
def testAsciiPhaseLogFormattingIsCompactAndAttributed(caplog):
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

    caplog.set_level(logging.INFO, logger=core.logger.name)
    core._logAsciiBlock("MUNC track start", (("MUNC variance EB", "enabled"),))

    assert caplog.records[-1].funcName == "testAsciiPhaseLogFormattingIsCompactAndAttributed"
    assert caplog.records[-1].message.startswith("\n+")
    assert caplog.records[-1].message.endswith("\n")


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testCEMAUsesSameBidirectionalKernelForFloat32AndFloat64(dtype):
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
def testMonoFuncUsesSameLogKernelForFloat32AndFloat64(dtype):
    x = np.array([-3.0, -0.25, 0.0, 2.5, 9.0], dtype=dtype)

    out, sentinel = cconsenrich.monoFunc(x, offset=0.75, scale=1.7)
    expected = _monoLogReference(x, offset=0.75, scale=1.7)

    assert sentinel == pytest.approx(-1.0)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, expected)


@pytest.mark.correctness
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testTransformLogRatioKernelMatchesReferenceForFloat32AndFloat64(dtype):
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
def testTransformPureLogPathMatchesMonoReferenceForFloat32AndFloat64(dtype):
    x = np.array([-2.0, -0.5, 0.0, 3.0, 8.0], dtype=dtype)
    inPlace = x.copy()

    returned = cconsenrich.cTransformInPlace(
        inPlace,
        blockLength=5,
        w_global=0.0,
        logOffset=0.75,
        logMult=1.7,
    )
    allocated = cconsenrich.cTransform(
        x,
        blockLength=5,
        w_global=0.0,
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
def testCSFMedianSelectionHandlesOddLengthDuplicates():
    base = (20.0 + (np.arange(601, dtype=np.float32) % 17)).astype(np.float32)
    factors = np.array([0.75, 1.0, 1.0], dtype=np.float32)
    chromMat = factors[:, None] * base[None, :]

    out = cconsenrich.cSF(chromMat, centerMedian=True, minRefDist=1)

    np.testing.assert_allclose(out, _expectedCSF(chromMat), rtol=1.0e-7, atol=1.0e-7)


@pytest.mark.correctness
def testCSFMedianSelectionHandlesEvenLengthDuplicates():
    base = (30.0 + (np.arange(600, dtype=np.float32) % 23)).astype(np.float32)
    factors = np.array([0.75, 1.25, 1.25, 2.0], dtype=np.float32)
    chromMat = factors[:, None] * base[None, :]

    out = cconsenrich.cSF(chromMat, centerMedian=True, minRefDist=1)

    np.testing.assert_allclose(out, _expectedCSF(chromMat), rtol=1.0e-7, atol=1.0e-7)


@pytest.mark.correctness
def testCTransformWithInputReturnsFloat32LogRatio():
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
def testCTransformWithInputReturnsFloat64LogRatio():
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
def testCTransformWithInputIntoWritesOutputInPlace():
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
def testCTransformInPlacePureLogMutatesFloat32Array():
    x = np.array([0.0, 3.0, 8.0], dtype=np.float32)
    original = x.copy()

    returned = cconsenrich.cTransformInPlace(
        x,
        blockLength=5,
        w_global=0.0,
        logOffset=1.0,
        logMult=2.0,
    )

    assert returned is x
    assert x.dtype == np.float32
    assert np.allclose(x, (2.0 * np.log(original + 1.0)).astype(np.float32))


@pytest.mark.correctness
def testCTransformInPlaceMatchesAllocatingTransformForFloat64():
    x = np.linspace(0.0, 5.0, 256, dtype=np.float64)
    x[120:140] += 10.0
    logged = _monoLogReference(x, offset=1.0, scale=1.0)
    dense_offset = cconsenrich.cDenseMean(
        logged,
        blockLenTarget=21,
        blockQuantile=0.25,
    )
    expected = cconsenrich.cTransform(
        x,
        blockLength=21,
        w_global=1.0,
        logOffset=1.0,
        logMult=1.0,
        blockQuantile=0.25,
    )
    in_place = x.copy()

    returned = cconsenrich.cTransformInPlace(
        in_place,
        blockLength=21,
        w_global=1.0,
        logOffset=1.0,
        logMult=1.0,
        blockQuantile=0.25,
    )

    assert returned is in_place
    assert in_place.dtype == np.float64
    assert np.allclose(in_place, expected)
    assert np.allclose(in_place, logged - dense_offset)


@pytest.mark.correctness
def testDenseMeanUsesMedianOfBlockMediansByDefault():
    block_len = 5
    blocks = [
        np.array([0.0, 1.0, 2.0, 3.0, 20.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0, 7.0, 30.0], dtype=np.float32),
        np.array([8.0, 9.0, 10.0, 11.0, 40.0], dtype=np.float32),
    ]
    x = np.concatenate(blocks)
    expected = np.median([np.quantile(block, 0.5) for block in blocks])

    observed = cconsenrich.cDenseMean(
        x,
        blockLenTarget=block_len,
    )

    assert observed == pytest.approx(expected)


@pytest.mark.correctness
def testDenseMeanHonorsBlockQuantileArgument():
    block_len = 5
    blocks = [
        np.array([0.0, 1.0, 2.0, 3.0, 20.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0, 7.0, 30.0], dtype=np.float32),
        np.array([8.0, 9.0, 10.0, 11.0, 40.0], dtype=np.float32),
    ]
    x = np.concatenate(blocks)
    expected = np.median([np.quantile(block, 0.75) for block in blocks])

    observed = cconsenrich.cDenseMean(
        x,
        blockLenTarget=block_len,
        blockQuantile=0.75,
    )

    assert observed == pytest.approx(expected)


@pytest.mark.correctness
def testDenseMeanNonpositiveBlockLengthUsesWholeTrackMedian():
    x = np.array([1.0, 2.0, 3.0, 9.0, 20.0, 30.0], dtype=np.float32)
    expected = np.quantile(x, 0.5)

    observed = cconsenrich.cDenseMean(
        x,
        blockLenTarget=-1,
    )

    assert observed == pytest.approx(expected)



@pytest.mark.correctness
def testDenseMeanIncludesFinalPartialBlock():
    block_len = 4
    x = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 50.0], dtype=np.float32)
    expected = np.median(
        [
            np.quantile(x[0:4], 0.25),
            np.quantile(x[4:8], 0.25),
            np.quantile(x[8:9], 0.25),
        ]
    )

    observed = cconsenrich.cDenseMean(
        x,
        blockLenTarget=block_len,
        blockQuantile=0.25,
    )

    assert observed == pytest.approx(expected)


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

    weightTrack = np.sum(invVarMatrix.astype(np.float64), axis=0)
    rhsTrack = np.sum(
        invVarMatrix.astype(np.float64) * residualMatrix.astype(np.float64),
        axis=0,
    )
    lam = core._backgroundPenaltyFromSpan(
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lam * core._buildSecondDiffPenalty(weightTrack.size))
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
def testZeroCenteredBackgroundUpdateUsesPrecisionWeights():
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
    lam = core._backgroundPenaltyFromSpan(
        blockLenIntervals=7,
        backgroundSmoothness=1.3,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lam * core._buildSecondDiffPenalty(weightTrack.size))
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
def testZeroCenteredBackgroundUpdateDoesNotUseSparseFactorization(
    monkeypatch: pytest.MonkeyPatch,
):
    residualMatrix = np.vstack(
        [
            np.linspace(-0.25, 0.25, 96, dtype=np.float32),
            np.linspace(0.25, -0.25, 96, dtype=np.float32),
        ]
    )
    invVarMatrix = np.ones_like(residualMatrix, dtype=np.float32)

    def _forbidSpsolve(*args, **kwargs):
        raise AssertionError("background update should use the banded solver")

    monkeypatch.setattr(core.sparse_linalg, "spsolve", _forbidSpsolve)

    background = core._solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )

    assert background.shape == (96,)
    assert np.isfinite(background).all()
    assert abs(float(np.mean(background))) < 1.0e-5


@pytest.mark.correctness
def testBackgroundUpdateCanSkipZeroCentering():
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
    lam = core._backgroundPenaltyFromSpan(
        blockLenIntervals=8,
        backgroundSmoothness=1.0,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lam * core._buildSecondDiffPenalty(weightTrack.size))
    expectedUncentered = core.sparse_linalg.spsolve(systemMat.tocsc(), rhsTrack)

    assert abs(float(np.mean(centered))) < 1.0e-5
    assert abs(float(np.mean(uncentered))) > 0.5
    assert np.allclose(
        uncentered,
        expectedUncentered.astype(np.float32),
        atol=1.0e-4,
    )


@pytest.mark.correctness
def testFinalForwardNISUsesMeanFinalForwardDiagnostic():
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
def testBackgroundUpdateUsesPriorAsDiagonalPenalty():
    n = 40
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    residualMatrix = np.vstack(
        [
            0.2 * np.sin(2.0 * np.pi * x),
            0.2 * np.cos(2.0 * np.pi * x),
        ]
    ).astype(np.float32)
    invVarMatrix = np.ones_like(residualMatrix, dtype=np.float32)
    priorMean = np.linspace(0.4, 0.8, n, dtype=np.float32)
    priorPrecision = np.linspace(0.25, 0.75, n, dtype=np.float64)

    background = core._solveZeroCenteredBackground(
        residualMatrix=residualMatrix,
        invVarMatrix=invVarMatrix,
        blockLenIntervals=6,
        backgroundSmoothness=0.8,
        zeroCenter=False,
        priorMeanTrack=priorMean,
        priorPrecisionTrack=priorPrecision,
    )

    weightTrack = np.sum(invVarMatrix.astype(np.float64), axis=0) + priorPrecision
    rhsTrack = (
        np.sum(
            invVarMatrix.astype(np.float64) * residualMatrix.astype(np.float64),
            axis=0,
        )
        + priorPrecision * priorMean.astype(np.float64)
    )
    lam = core._backgroundPenaltyFromSpan(
        blockLenIntervals=6,
        backgroundSmoothness=0.8,
    )
    systemMat = core.sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (lam * core._buildSecondDiffPenalty(n))
    expected = core.sparse_linalg.spsolve(systemMat.tocsc(), rhsTrack)

    assert np.allclose(background, expected.astype(np.float32), atol=1.0e-4)


@pytest.mark.correctness
def testBackgroundPriorDefaultWindowMatchesBackgroundLengthScale():
    assert core._backgroundPriorWindowLength(
        intervalCount=101,
        blockLenIntervals=9,
    ) == 9
    assert core._backgroundPriorWindowLength(
        intervalCount=101,
        blockLenIntervals=8,
    ) == 8
    assert core._backgroundPriorWindowLength(
        intervalCount=50,
        blockLenIntervals=101,
    ) == 50


@pytest.mark.correctness
def testBackgroundPriorMeanVarianceTracksUseRobustLocalQuantile():
    n = 17
    baseline = np.ones(n, dtype=np.float32)
    baseline[8] = 20.0
    matrixData = np.vstack([baseline, baseline + 0.1]).astype(np.float32)
    matrixMunc = np.ones_like(matrixData, dtype=np.float32)

    priorMean, priorMeanVariance = core._backgroundPriorMeanVarianceTracks(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        pad=0.0,
        blockLenIntervals=3,
        backgroundPriorQuantile=0.25,
    )

    assert priorMean.shape == (n,)
    assert float(priorMean[8]) < 2.0
    assert np.isfinite(priorMean).all()
    assert priorMeanVariance.shape == (n,)
    assert np.all(priorMeanVariance > 0.0)
    assert np.isfinite(priorMeanVariance).all()


@pytest.mark.correctness
def testPositiveSequenceESSDetectsSerialDependence():
    dependent = np.repeat(np.array([0.0, 1.0], dtype=np.float64), 10)
    dependent = np.tile(dependent, 20)
    independentLike = np.tile(np.array([0.0, 1.0], dtype=np.float64), 200)

    essDependent, tauDependent, lagsDependent = (
        core._estimatePositiveSequenceEffectiveSampleSize(dependent, maxLag=30)
    )
    essIndependent, tauIndependent, lagsIndependent = (
        core._estimatePositiveSequenceEffectiveSampleSize(independentLike, maxLag=30)
    )

    assert tauDependent > 1.0
    assert lagsDependent > 0
    assert essDependent < dependent.size
    assert tauIndependent == pytest.approx(1.0)
    assert lagsIndependent == 0
    assert essIndependent == pytest.approx(float(independentLike.size))


@pytest.mark.correctness
def testBackgroundPriorQuantileVarianceUsesEffectiveWindow():
    target = np.repeat(np.array([0.0, 1.0], dtype=np.float32), 10)
    target = np.tile(target, 20)
    matrixData = np.vstack([target, target + 0.01]).astype(np.float32)
    matrixMunc = np.ones_like(matrixData, dtype=np.float32)

    _priorMean, priorMeanVariance, diagnostics = core._backgroundPriorMeanVarianceTracks(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        pad=0.0,
        blockLenIntervals=31,
        backgroundPriorQuantile=0.5,
        returnDiagnostics=True,
    )

    assert diagnostics["nominal_window"] == 31
    assert diagnostics["tau_int"] > 1.0
    assert diagnostics["effective_window"] < diagnostics["nominal_window"]
    assert np.isfinite(priorMeanVariance).all()
    assert np.all(priorMeanVariance > 0.0)


@pytest.mark.correctness
def testBackgroundPriorVarianceAndPrecisionUseBaselineUncertainty():
    priorMean = np.zeros(20, dtype=np.float64)
    target = np.tile(np.array([-0.6, 0.6], dtype=np.float64), 10)
    targetVariance = np.full(20, 0.05, dtype=np.float64)
    priorMeanVariance = np.full(20, 0.10, dtype=np.float64)

    variance = core._estimateBackgroundPriorVariance(
        targetTrack=target,
        targetVariance=targetVariance,
        priorMeanTrack=priorMean,
        priorMeanVarianceTrack=priorMeanVariance,
        trimQuantile=1.0,
    )
    assert variance > 0.05

    precision = core._backgroundPriorPrecisionTrack(
        backgroundPriorVariance=variance,
        priorMeanVarianceTrack=np.array([0.10, 0.40], dtype=np.float64),
    )

    assert np.all(precision > 0.0)
    assert precision[0] > precision[1]
    assert precision[0] == pytest.approx(1.0 / (variance + 0.10))


@pytest.mark.correctness
def testBackgroundPriorVariancePenaltyAvoidsBoundaryZero():
    n = 2000
    priorMean = np.zeros(n, dtype=np.float64)
    target = np.zeros(n, dtype=np.float64)
    targetVariance = np.full(n, 0.05, dtype=np.float64)
    priorMeanVariance = np.full(n, 0.10, dtype=np.float64)

    variance = core._estimateBackgroundPriorVariance(
        targetTrack=target,
        targetVariance=targetVariance,
        priorMeanTrack=priorMean,
        priorMeanVarianceTrack=priorMeanVariance,
        trimQuantile=1.0,
        penaltyShape=2.0,
        penaltyRate=0.0,
    )

    assert variance > 0.0
    assert variance < 1.0e-3


@pytest.mark.correctness
def testReplicateBiasCanSkipZeroCentering():
    n = 24
    m = 3
    matrixData = np.full((m, n), 2.0, dtype=np.float32)
    matrixMunc = np.full((m, n), 10.0, dtype=np.float32)
    matrixF = core.constructMatrixF(0.1).astype(np.float32, copy=False)
    matrixQ0 = core.constructMatrixQ(
        minDiagQ=1.0e-6,
        offDiagQ=0.0,
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
        EM_maxIters=1,
        EM_innerRtol=0.0,
        pad=1.0e-4,
        EM_tNu=8.0,
        EM_useObsPrecReweight=False,
        EM_useProcPrecReweight=False,
        EM_useAPN=False,
        EM_useReplicateBias=True,
        returnIntermediates=True,
        t_innerIters=1,
    )

    centeredBias = cconsenrich.cinnerEM(
        **commonKwargs,
        EM_zeroCenterReplicateBias=True,
    )[-1]
    uncenteredBias = cconsenrich.cinnerEM(
        **commonKwargs,
        EM_zeroCenterReplicateBias=False,
    )[-1]

    assert abs(float(np.mean(centeredBias))) < 1.0e-5
    assert abs(float(np.mean(uncenteredBias))) > 1.0
    assert np.all(np.asarray(uncenteredBias) > 1.0)


@pytest.mark.correctness
def testCinnerEMReplicateBiasUpdateMatchesPrecisionWeightedMinimizer():
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
        offDiagQ=0.0,
    ).astype(np.float32, copy=False)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)
    pad = 1.0e-4

    out = cconsenrich.cinnerEM(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0e-10,
        EM_maxIters=1,
        EM_innerRtol=0.0,
        pad=pad,
        EM_tNu=8.0,
        EM_useObsPrecReweight=False,
        EM_useProcPrecReweight=False,
        EM_useAPN=False,
        EM_useReplicateBias=True,
        EM_zeroCenterReplicateBias=True,
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
def testCinnerEMReplicateBiasUsesFixedCenterConstraintWithRobustWeights():
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
        offDiagQ=0.0,
    ).astype(np.float32, copy=False)
    intervalToBlockMap = np.zeros(n, dtype=np.int32)
    pad = 1.0e-4

    out = cconsenrich.cinnerEM(
        matrixData=matrixData,
        matrixPluginMuncInit=matrixMunc,
        matrixF=matrixF,
        matrixQ0=matrixQ0,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=1,
        stateInit=0.0,
        stateCovarInit=1.0e-8,
        EM_maxIters=1,
        EM_innerRtol=0.0,
        pad=pad,
        EM_tNu=2.5,
        EM_useObsPrecReweight=True,
        EM_useProcPrecReweight=False,
        EM_useAPN=False,
        EM_useReplicateBias=True,
        EM_zeroCenterReplicateBias=True,
        obsPrecisionMultiplierMin=0.25,
        obsPrecisionMultiplierMax=4.0,
        returnIntermediates=True,
        t_innerIters=1,
    )
    stateSmoothed = np.asarray(out[2], dtype=np.float64)[:, 0]
    lambdaExp = np.asarray(out[6], dtype=np.float64)
    replicateBias = np.asarray(out[-1], dtype=np.float64)

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
def testSummarizeStateRoughnessUsesHoldoutBlocksAndSignalStrata():
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
def testSummarizePrecisionBoundaryHitsSkipsFirstProcessWeight():
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
def testExpectedTransitionResidualSumsUsesLagOrientationAndDeltaF():
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
def testExpectedTransitionResidualSumsMatchesPythonReference():
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
def testRunConsenrichOuterEMSmoke(caplog):
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

    caplog.set_level(logging.INFO, logger=core.logger.name)
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
        EM_maxIters=3,
        EM_outerIters=2,
        processQCalibration="none",
        applyJackknife=False,
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
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert diagnostics["final_forward_nis"] == pytest.approx(float(np.mean(NIS)))
    assert diagnostics["background_prior"]["active"] is True
    assert "PHASE: CORE START" in caplog.text
    assert "PHASE: FINAL FIT" in caplog.text
    assert "PHASE: FINAL FIT SUMMARY" in caplog.text


@pytest.mark.correctness
def testRunConsenrichProcessQCalibrationWarmupRestoresFinalReweighting(monkeypatch):
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

    originalInnerEM = cconsenrich.cinnerEM
    innerEMModes = []

    def _spyInnerEM(*args, **kwargs):
        innerEMModes.append(
            (
                bool(kwargs.get("EM_useProcPrecReweight")),
                bool(kwargs.get("EM_useAPN")),
            )
        )
        return originalInnerEM(*args, **kwargs)

    monkeypatch.setattr(cconsenrich, "cinnerEM", _spyInnerEM)

    out = core.runConsenrich(
        matrixData,
        matrixMunc,
        deltaF=1.0,
        minQ=1.0e-3,
        maxQ=0.5,
        offDiagQ=0.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        EM_maxIters=1,
        EM_outerIters=1,
        EM_useProcPrecReweight=True,
        EM_useAPN=False,
        processQCalibration="regularizedDiagonal",
        processQCalibIters=1,
        applyJackknife=False,
    )

    assert innerEMModes[0] == (False, False)
    assert innerEMModes[-1] == (True, False)
    stateSmoothed, stateCovarSmoothed, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert np.all(np.isfinite(stateSmoothed))
    assert np.all(np.isfinite(stateCovarSmoothed))


@pytest.mark.correctness
def testRunConsenrichOuterLoopRequiresThreeIterationsDespiteTolerance(monkeypatch):
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

    originalInnerEM = cconsenrich.cinnerEM
    calls = []

    def _spyInnerEM(*args, **kwargs):
        calls.append(1)
        return originalInnerEM(*args, **kwargs)

    monkeypatch.setattr(cconsenrich, "cinnerEM", _spyInnerEM)

    commonKwargs = dict(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        deltaF=0.2,
        minQ=1.0e-3,
        maxQ=0.5,
        offDiagQ=0.0,
        stateInit=0.0,
        stateCovarInit=1.0,
        boundState=False,
        stateLowerBound=0.0,
        stateUpperBound=0.0,
        blockLenIntervals=8,
        EM_maxIters=1,
        EM_outerRtol=1.0e9,
        processQCalibration="none",
        applyJackknife=False,
    )

    core.runConsenrich(**commonKwargs, EM_outerIters=5)
    assert len(calls) == 3

    calls.clear()
    core.runConsenrich(**commonKwargs, EM_outerIters=2)
    assert len(calls) == 3


@pytest.mark.correctness
def testGetBedMaskUsesIntervalSpanOverlap(tmp_path):
    bedPath = tmp_path / "regions.bed"
    bedPath.write_text("chrTest\t110\t120\n", encoding="utf-8")
    intervals = np.array([0, 100, 200], dtype=np.uint32)

    mask = core.getBedMask("chrTest", str(bedPath), intervals)

    assert mask.tolist() == [False, True, False]


@pytest.mark.correctness
def testSparseSupportWeightsUseExponentialDistanceDecay():
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
def testPSplineLogVarianceTrendRecoversNonmonotoneShape():
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
def testPSplineSignedPredictorDistinguishesPositiveAndNegativeMeans():
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
def testPSplinePredictionClampsToTrainingBoundary():
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
def testPSplineCythonEvaluationMatchesDenseDesign():
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
def testPSplineLimitsBasisCountByWeightedSupport():
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
def testPooledMuncTrendRecoversReplicateVarianceFactors():
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
def testPSplineGuardedGCVAppliesDefaultEdfCap():
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
        eps=1.0e-8,
    )

    assert trend.diagnostics["trend_max_edf"] == pytest.approx(30.0)
    assert trend.edf <= 30.0 + 1.0e-6


@pytest.mark.correctness
def testPSplineUsesQuantileKnotsFromSupport():
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
def testPSplinePredictionClipsBeforeFloat32Overflow():
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
def testPSplineTrendSummaryLogsRelationship():
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
def testMuncVarianceDiagnosticsLogLocalGlobalFinalAndTailSupport():
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
def testDefaultProcessPrecisionMultiplierBoundsAreTight():
    assert core.processParams().precisionMultiplierMin == pytest.approx(0.5)
    assert core.processParams().precisionMultiplierMax == pytest.approx(2.0)

    signature = inspect.signature(core.runConsenrich)
    assert signature.parameters["processPrecisionMultiplierMin"].default == pytest.approx(0.5)
    assert signature.parameters["processPrecisionMultiplierMax"].default == pytest.approx(2.0)


@pytest.mark.correctness
def testCheckStateUncertaintyCoverageOverallAndStrata():
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
def testLinearEnvelopeParameterIsAbsent():
    removed = "EB" + "_minLin"
    assert removed not in core.observationParams._fields

    with pytest.raises(TypeError):
        core.fitPSplineLogVarianceTrend(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([1.0, 1.1, 1.0, 1.2]),
            **{removed: 10.0},
        )


@pytest.mark.correctness
def testMonotonePoolingSourceSymbolsAbsent():
    removed = "P" + "AVA"
    sourcePaths = [
        Path(core.__file__),
        Path(core.__file__).parent / "cconsenrich.pyx",
    ]

    for sourcePath in sourcePaths:
        assert removed not in sourcePath.read_text(encoding="utf-8")


@pytest.mark.correctness
def testEBPriorStrengthBoundaryIsUsable():
    assert core._coerceEBPriorStrength(4.0) == pytest.approx(4.0)
    assert core._coerceEBPriorStrength(4) == pytest.approx(4.0)
    assert core._coerceEBPriorStrength(3.999999) is None
    assert core._coerceEBPriorStrength(float("nan")) is None


@pytest.mark.correctness
def testEBPriorStrengthUsesThinnedVariancePairs():
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
def testPooledPriorStrengthThinsBySampleChromosomeAndBin(
    monkeypatch: pytest.MonkeyPatch,
):
    n = 12
    localVars = np.linspace(1.0, 2.0, n, dtype=np.float64)
    globalVars = np.ones(n, dtype=np.float64)
    sampleIndex = np.repeat([0, 1], 6)
    chromosomeIndex = np.tile(np.repeat([0, 1], 3), 2)
    blockStarts = np.tile(np.array([0, 4, 8, 0, 4, 8], dtype=np.int64), 2)
    seen: dict[str, np.ndarray] = {}

    def _fakeCompute(localArr, globalArr, nuLocal, candidateIdx):
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
def testGetMuncTrackSparseNearestPath(monkeypatch: pytest.MonkeyPatch):
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

    def _fakeEBComputePriorStrength(localVars, priorVars, nuLocal, **kwargs):
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
        "crolling_AR1_IVar",
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
def testGetMuncTrackClipsHugePriorBeforeShrinkage(
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
    monkeypatch.setattr(cconsenrich, "crolling_AR1_IVar", _fakeRolling)
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
def testGetMuncTrackCapsPriorStrengthAtFiftyTimesLocalDf(
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
        "crolling_AR1_IVar",
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
def testGetMuncTrackUsesSuppliedPooledTrendFactorAndBoundaryNu0(
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
        "crolling_AR1_IVar",
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
def testMeanVarPairSampleSizesStayWithinTrackLength():
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
def testRollingAR1CanReturnMarginalVarianceInsteadOfInnovation():
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
def testGetMuncTrackSparseNearestDetrendsPriorBySignedLocalIntercept(
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

    def _fakeMeanVarPairs(intervalsArg, valuesArg, *args, **kwargs):
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
        "crolling_AR1_IVar",
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
        "crolling_AR1_IVar",
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
        EM_maxIters=2,
        EM_outerIters=1,
        EM_useProcPrecReweight=True,
        EM_useAPN=True,
        applyJackknife=False,
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(NIS))


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
        disableCalibration=True,
        EM_useAPN=True,
        EM_useProcPrecReweight=True,
        applyJackknife=False,
    )

    stateSmoothed, stateCovarSmoothed, postFitResiduals, NIS, *_ = out
    assert stateSmoothed.shape == (n, 2)
    assert stateCovarSmoothed.shape == (n, 2, 2)
    assert postFitResiduals.shape == (n, m)
    assert NIS.shape == (n,)
    assert np.all(np.isfinite(stateSmoothed))


@pytest.mark.correctness
def testChooseFeatureLengthBootstrapWidthVarianceAndContextCompat():
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
def testChooseDependenceLengthUsesAutocorrelationAndIncrements():
    rng = np.random.default_rng(123)
    n = 512
    base = np.empty(n, dtype=np.float64)
    base[0] = 0.0
    for i in range(1, n):
        base[i] = 0.85 * base[i - 1] + rng.normal(scale=0.5)
    chromMat = np.vstack(
        [
            base + rng.normal(scale=0.05, size=n),
            base + rng.normal(scale=0.05, size=n),
            base + rng.normal(scale=0.05, size=n),
        ]
    ).astype(np.float32)

    pointSpan, lowerSpan, upperSpan, diagnostics = core.chooseDependenceLength(
        chromMat,
        intervalSizeBP=25,
        minSpan=3,
        maxSpan=64,
    )

    assert 3 <= pointSpan <= 64
    assert 3 <= lowerSpan <= pointSpan <= upperSpan <= 64
    assert diagnostics["method"] == "dependence_acf_increment"
    assert diagnostics["context_size_bp"] == pointSpan * 50 + 1
    assert diagnostics["finite_count"] == n
    assert len(diagnostics["acf"]) == 64
    assert len(diagnostics["increment_variance"]) == 64


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
def testReadSegmentsFragmentsDefaultToCoverage():
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
def testCCountsCountBedGraphRegionWeightedByOverlap(tmp_path):
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
def testCCountsCountIndexedBedGraphRegionWeightedByOverlap(tmp_path):
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
def testReadSegmentsBedGraphScalesNativeCounts(tmp_path):
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
def testBedGraphChromRangeAndReadLength(tmp_path):
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


@pytest.mark.correctness
def testPairScaleFactorsDownscaleDeeperSampleMacsStyle(monkeypatch):
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
def testPairScaleFactorsCanDownscaleControlByDefault(monkeypatch):
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
def testPairScaleFactorsFixControlKeepsControlFullDepth(monkeypatch):
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
