# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple
import logging
import re
import numpy as np
import pandas as pd
import pybedtools as bed
import pysam as sam

from scipy import signal, ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from .misc_util import getChromSizesDict
from .constants import EFFECTIVE_GENOME_SIZES
from .cconsenrich import cgetFragmentLength, cEMA


def getScaleFactor1x(
    bamFile: str,
    effectiveGenomeSize: int,
    readLength: int,
    excludeChroms: List[str],
    chromSizesFile: str,
    samThreads: int,
) -> float:
    r"""Generic normalization factor based on effective genome size and number of mapped reads in non-excluded chromosomes.

    :param bamFile: See :class:`consenrich.core.inputParams`.
    :type bamFile: str
    :param effectiveGenomeSize: Effective genome size in base pairs. See :func:`consenrich.constants.getEffectiveGenomeSize`.
    :type effectiveGenomeSize: int
    :param readLength: read length or fragment length
    :type readLength: int
    :param excludeChroms: List of chromosomes to exclude from the analysis.
    :type excludeChroms: List[str]
    :param chromSizesFile: Path to the chromosome sizes file.
    :type chromSizesFile: str
    :param samThreads: See :class:`consenrich.core.samParams`.
    :type samThreads: int
    :return: Scale factor for 1x normalization.
    :rtype: float
    """
    if excludeChroms is not None:
        if chromSizesFile is None:
            raise ValueError(
                "`excludeChroms` is provided...so must be `chromSizesFile`."
            )
        chromSizes: dict = getChromSizesDict(chromSizesFile)
        for chrom in excludeChroms:
            if chrom not in chromSizes:
                continue
            effectiveGenomeSize -= chromSizes[chrom]
    totalMappedReads: int = -1
    with sam.AlignmentFile(bamFile, "rb", threads=samThreads) as aln:
        totalMappedReads = aln.mapped
        if excludeChroms is not None:
            idxStats = aln.get_index_statistics()
            for element in idxStats:
                if element.contig in excludeChroms:
                    totalMappedReads -= element.mapped
    if totalMappedReads <= 0 or effectiveGenomeSize <= 0:
        raise ValueError(
            f"Negative EGS after removing excluded chromosomes or no mapped reads: EGS={effectiveGenomeSize}, totalMappedReads={totalMappedReads}."
        )

    return round(
        effectiveGenomeSize / (totalMappedReads * readLength), 5
    )


def getScaleFactorPerMillion(
    bamFile: str, excludeChroms: List[str], intervalSizeBP: int
) -> float:
    r"""Generic normalization factor based on number of mapped reads in non-excluded chromosomes.

    :param bamFile: See :class:`consenrich.core.inputParams`.
    :type bamFile: str
    :param excludeChroms: List of chromosomes to exclude when counting mapped reads.
    :type excludeChroms: List[str]
    :return: Scale factor accounting for number of mapped reads (only).
    :rtype: float
    """
    if not os.path.exists(bamFile):
        raise FileNotFoundError(f"BAM file {bamFile} does not exist.")
    totalMappedReads: int = 0
    with sam.AlignmentFile(bamFile, "rb") as aln:
        totalMappedReads = aln.mapped
        if excludeChroms is not None:
            idxStats = aln.get_index_statistics()
            for element in idxStats:
                if element.contig in excludeChroms:
                    totalMappedReads -= element.mapped
    if totalMappedReads <= 0:
        raise ValueError(
            f"After removing reads mapping to excluded chroms, totalMappedReads is {totalMappedReads}."
        )
    scalePM = round(
        (1_000_000 / totalMappedReads) * (1000 / intervalSizeBP), 5
    )
    return scalePM


def getPairScaleFactors(
    bamFileA: str,
    bamFileB: str,
    effectiveGenomeSizeA: int,
    effectiveGenomeSizeB: int,
    readLengthA: int,
    readLengthB: int,
    excludeChroms: List[str],
    chromSizesFile: str,
    samThreads: int,
    intervalSizeBP: int,
    normMethod: str = "EGS",
    fixControl: bool = True,
) -> Tuple[float, float]:
    r"""Scale treatment:control data based on effective genome size or reads per million.

    :param bamFileA: Alignment file for the 'treatment' sample.
    :type bamFileA: str
    :param bamFileB: Alignment file for the 'control' sample (e.g., input).
    :type bamFileB: str
    :param effectiveGenomeSizeA: Effective genome size for the treatment sample.
    :type effectiveGenomeSizeA: int
    :param effectiveGenomeSizeB: Effective genome size for the control sample.
    :type effectiveGenomeSizeB: int
    :param readLengthA: Read or fragment length for the treatment sample.
    :type readLengthA: int
    :param readLengthB: Read or fragment length for the control sample.
    :type readLengthB: int
    :param excludeChroms: List of chromosomes to exclude from the analysis.
    :type excludeChroms: List[str]
    :param chromSizesFile: Path to the chromosome sizes file.
    :type chromSizesFile: str
    :param samThreads: See :class:`consenrich.core.samParams`.
    :type samThreads: int
    :param intervalSizeBP: Step size for coverage calculation.
    :param: normMethod: Normalization method to use ("EGS" or "RPKM").
    :type normMethod: str
    :return: Tuple of scale factors for treatment and control samples.
    :rtype: Tuple[float, float]
    """

    if normMethod.upper() == "RPKM":
        scaleFactorA = getScaleFactorPerMillion(
            bamFileA,
            excludeChroms,
            intervalSizeBP,
        )
        scaleFactorB = getScaleFactorPerMillion(
            bamFileB,
            excludeChroms,
            intervalSizeBP,
        )
    else:
        scaleFactorA = getScaleFactor1x(
            bamFileA,
            effectiveGenomeSizeA,
            readLengthA,
            excludeChroms,
            chromSizesFile,
            samThreads,
        )
        scaleFactorB = getScaleFactor1x(
            bamFileB,
            effectiveGenomeSizeB,
            readLengthB,
            excludeChroms,
            chromSizesFile,
            samThreads,
        )

    coverageA = 1.0 / scaleFactorA if scaleFactorA > 0.0 else 0.0
    coverageB = 1.0 / scaleFactorB if scaleFactorB > 0.0 else 0.0

    if fixControl:
        # keep control full depth, never scale it down, never scale it up
        scaleFactorB = 1.0

        # only downscale treatment to the (unscaled) control, never upscale treatment
        if coverageA > coverageB and coverageA > 0.0:
            scaleFactorA = scaleFactorA * (coverageB / coverageA)
        else:
            scaleFactorA = 1.0
    else:
        # downscale higher --> lower (regardless of treatment/control status)
        if coverageA > coverageB and coverageA > 0.0:
            scaleFactorA = scaleFactorA * (coverageB / coverageA)
            scaleFactorB = 1.0
        elif coverageB > coverageA and coverageB > 0.0:
            scaleFactorB = scaleFactorB * (coverageA / coverageB)
            scaleFactorA = 1.0
        else:
            scaleFactorA = 1.0
            scaleFactorB = 1.0

    ratio = max(scaleFactorA, scaleFactorB) / max(
        1.0e-12, min(scaleFactorA, scaleFactorB)
    )
    if ratio > 5.0:
        logger.warning(
            f"Scale factors differ > 5x....\n"
            f"\n\tAre effective genome sizes {effectiveGenomeSizeA} and {effectiveGenomeSizeB} correct?"
            f"\n\tAre read/fragment lengths {readLengthA},{readLengthB} correct?"
        )

    return scaleFactorA, scaleFactorB