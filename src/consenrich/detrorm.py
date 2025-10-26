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
    return round(effectiveGenomeSize / (totalMappedReads * readLength), 4)


def getScaleFactorPerMillion(bamFile: str, excludeChroms: List[str]) -> float:
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
    scalePM = round(1_000_000 / totalMappedReads, 4)
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
    scaleDown: bool = True,
) -> Tuple[float, float]:
    r"""Get scaling constants that normalize two alignment files to each other (e.g. ChIP-seq treatment and control) with respect to sequence coverage.

    :param bamFileA: Path to the first BAM file.
    :type bamFileA: str
    :param bamFileB: Path to the second BAM file.
    :type bamFileB: str
    :param effectiveGenomeSizeA: Effective genome size for the first BAM file.
    :type effectiveGenomeSizeA: int
    :param effectiveGenomeSizeB: Effective genome size for the second BAM file.
    :type effectiveGenomeSizeB: int
    :param readLengthA: read length or fragment length for the first BAM file.
    :type readLengthA: int
    :param readLengthB: read length or fragment length for the second BAM file.
    :type readLengthB: int
    :param excludeChroms: List of chromosomes to exclude from the analysis.
    :type excludeChroms: List[str]
    :param chromSizesFile: Path to the chromosome sizes file.
    :type chromSizesFile: str
    :param samThreads: Number of threads to use for reading BAM files.
    :type samThreads: int
    :return: A tuple containing the scale factors for the first and second BAM files.
    :rtype: Tuple[float, float]
    """
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
    logger.info(
        f"Initial scale factors: {bamFileA}: {scaleFactorA}, {bamFileB}: {scaleFactorB}"
    )
    if not scaleDown:
        return scaleFactorA, scaleFactorB
    coverageA = 1 / scaleFactorA
    coverageB = 1 / scaleFactorB
    if coverageA < coverageB:
        scaleFactorB *= coverageA / coverageB
        scaleFactorA = 1.0
    else:
        scaleFactorA *= coverageB / coverageA
        scaleFactorB = 1.0

    logger.info(
        f"Final scale factors: {bamFileA}: {scaleFactorA}, {bamFileB}: {scaleFactorB}"
    )

    ratio = max(scaleFactorA, scaleFactorB) / min(scaleFactorA, scaleFactorB)
    if ratio > 5.0:
        logger.warning(
            f"Scale factors differ > 5x....\n"
            f"\n\tAre effective genome sizes {effectiveGenomeSizeA} and {effectiveGenomeSizeB} correct?"
            f"\n\tAre read/fragment lengths {readLengthA},{readLengthB} correct?"
        )
    return scaleFactorA, scaleFactorB


def detrendTrack(
    values: np.ndarray,
    stepSize: int,
    detrendWindowLengthBP: int,
    useOrderStatFilter: bool,
    usePolyFilter: bool,
    detrendTrackPercentile: float,
    detrendSavitzkyGolayDegree: int,
) -> np.ndarray:
    r"""Detrend tracks using either an order statistic filter or a polynomial filter.

    :param values: Values to detrend.
    :type values: np.ndarray
    :param stepSize: see :class:`consenrich.core.countingParams`.
    :type stepSize: int
    :param detrendWindowLengthBP: See :class:`consenrich.core.detrendParams`.
    :type detrendWindowLengthBP: int
    :param useOrderStatFilter: Whether to use a sliding order statistic filter.
    :type useOrderStatFilter: bool
    :param usePolyFilter: Whether to use a sliding polynomial/least squares filter.
    :type usePolyFilter: bool
    :param detrendTrackPercentile: Percentile to use for the order statistic filter.
    :type detrendTrackPercentile: float
    :param detrendSavitzkyGolayDegree: Degree of the polynomial for the Savitzky-Golay/Polynomial filter.
    :type detrendSavitzkyGolayDegree: int
    :return: Detrended values.
    :rtype: np.ndarray
    :raises ValueError: If the detrend window length is not greater than 3 times the step size
        or if the values length is less than the detrend window length.
    """
    bothSpecified: bool = False
    size = int(detrendWindowLengthBP / stepSize)
    if size % 2 == 0:
        size += 1
    if size < 3:
        raise ValueError("Required: windowLengthBP > 3*stepSize.")
    if len(values) < size:
        raise ValueError("values length must be greater than windowLength.")

    if useOrderStatFilter and usePolyFilter:
        logger.warning(
            "Both order statistic and polynomial filters specified...using order statistic filter."
        )
        bothSpecified = True

    if useOrderStatFilter or bothSpecified:
        return values - ndimage.percentile_filter(
            values, detrendTrackPercentile, size=size
        )
    elif usePolyFilter:
        return values - signal.savgol_filter(
            values, size, detrendSavitzkyGolayDegree
        )

    return values - ndimage.uniform_filter1d(values, size=size, mode="nearest")
