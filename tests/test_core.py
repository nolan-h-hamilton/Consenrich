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
import consenrich.cconsenrich as cconsenrich
import consenrich.matching as matching
import consenrich.misc_util as misc_util


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
