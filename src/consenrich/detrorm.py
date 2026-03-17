# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple
import logging
import re
import numpy as np
import pandas as pd

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
from . import ccounts


def getScaleFactor1x(
    bamFile: str,
    effectiveGenomeSize: int,
    readLength: int,
    excludeChroms: List[str],
    chromSizesFile: str,
    samThreads: int,
    sourceKind: str | None = None,
    barcodeAllowListFile: str | None = None,
    countMode: str | None = None,
    oneReadPerBin: int = 0,
    groupCellCount: int | None = None,
    fragmentsGroupNorm: str | None = None,
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
    if sourceKind is None:
        if str(bamFile).lower().endswith(".cram"):
            raise ValueError("CRAM inputs are no longer supported.")
        sourceKind = "BAM"
    sourceKind = str(sourceKind).upper()
    if sourceKind == "FRAGMENTS":
        raise ValueError(
            "EGS/RPGC normalization is not supported for fragments sources use CPM/RPKM instead"
        )

    totalMappedReads, _ = ccounts.ccounts_getAlignmentMappedReadCount(
        bamFile,
        excludeChromosomes=excludeChroms,
        threadCount=samThreads,
        sourceKind=sourceKind,
        barcodeAllowListFile=barcodeAllowListFile or "",
        countMode=countMode or "coverage",
        oneReadPerBin=oneReadPerBin,
    )
    if totalMappedReads <= 0 or effectiveGenomeSize <= 0:
        raise ValueError(
            f"Negative EGS after removing excluded chromosomes or no mapped reads: EGS={effectiveGenomeSize}, totalMappedReads={totalMappedReads}."
        )

    return round(effectiveGenomeSize / (totalMappedReads * readLength), 5)


def getScaleFactorPerMillion(
    bamFile: str,
    excludeChroms: List[str],
    intervalSizeBP: int,
    sourceKind: str | None = None,
    barcodeAllowListFile: str | None = None,
    countMode: str | None = None,
    oneReadPerBin: int = 0,
    groupCellCount: int | None = None,
    fragmentsGroupNorm: str | None = None,
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
    if sourceKind is None:
        if str(bamFile).lower().endswith(".cram"):
            raise ValueError("CRAM inputs are no longer supported.")
        sourceKind = "BAM"
    sourceKind = str(sourceKind).upper()
    totalMappedReads, _ = ccounts.ccounts_getAlignmentMappedReadCount(
        bamFile,
        excludeChromosomes=excludeChroms,
        sourceKind=sourceKind,
        barcodeAllowListFile=barcodeAllowListFile or "",
        countMode=countMode or "coverage",
        oneReadPerBin=oneReadPerBin,
    )
    if totalMappedReads <= 0:
        raise ValueError(
            f"After removing reads mapping to excluded chroms, totalMappedReads is {totalMappedReads}."
        )
    scalePM = round((1_000_000 / totalMappedReads) * (1000 / intervalSizeBP), 5)
    fragmentsGroupNorm = str(fragmentsGroupNorm or "NONE").strip().upper()
    if sourceKind == "FRAGMENTS" and fragmentsGroupNorm == "CELLS":
        if groupCellCount is None or groupCellCount <= 0:
            raise ValueError(
                "fragmentsGroupNorm=CELLS requires a positive selected cell count"
            )
        scalePM = scalePM / float(groupCellCount)
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
    sourceKindA: str | None = None,
    sourceKindB: str | None = None,
    barcodeAllowListFileA: str | None = None,
    barcodeAllowListFileB: str | None = None,
    countModeA: str | None = None,
    countModeB: str | None = None,
    oneReadPerBin: int = 0,
    groupCellCountA: int | None = None,
    groupCellCountB: int | None = None,
    fragmentsGroupNorm: str | None = None,
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
            sourceKind=sourceKindA,
            barcodeAllowListFile=barcodeAllowListFileA,
            countMode=countModeA,
            oneReadPerBin=oneReadPerBin,
            groupCellCount=groupCellCountA,
            fragmentsGroupNorm=fragmentsGroupNorm,
        )
        scaleFactorB = getScaleFactorPerMillion(
            bamFileB,
            excludeChroms,
            intervalSizeBP,
            sourceKind=sourceKindB,
            barcodeAllowListFile=barcodeAllowListFileB,
            countMode=countModeB,
            oneReadPerBin=oneReadPerBin,
            groupCellCount=groupCellCountB,
            fragmentsGroupNorm=fragmentsGroupNorm,
        )
    else:
        scaleFactorA = getScaleFactor1x(
            bamFileA,
            effectiveGenomeSizeA,
            readLengthA,
            excludeChroms,
            chromSizesFile,
            samThreads,
            sourceKind=sourceKindA,
            barcodeAllowListFile=barcodeAllowListFileA,
            countMode=countModeA,
            oneReadPerBin=oneReadPerBin,
            groupCellCount=groupCellCountA,
            fragmentsGroupNorm=fragmentsGroupNorm,
        )
        scaleFactorB = getScaleFactor1x(
            bamFileB,
            effectiveGenomeSizeB,
            readLengthB,
            excludeChroms,
            chromSizesFile,
            samThreads,
            sourceKind=sourceKindB,
            barcodeAllowListFile=barcodeAllowListFileB,
            countMode=countModeB,
            oneReadPerBin=oneReadPerBin,
            groupCellCount=groupCellCountB,
            fragmentsGroupNorm=fragmentsGroupNorm,
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
