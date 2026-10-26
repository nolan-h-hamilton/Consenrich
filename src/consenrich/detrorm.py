# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple
import logging
import re
import numpy as np
import pandas as pd

from scipy import signal, ndimage

logger = logging.getLogger(__name__)

from .misc_util import getChromSizesDict
from .constants import COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP, EFFECTIVE_GENOME_SIZES
from .cconsenrich import cgetFragmentLength, cEMA
from . import ccounts
from . import constants
from . import _normalization as normalization


def _bamPairedEndMode(bamInputMode: str | None) -> int:
    return 1 if normalization.validateBamInputMode(bamInputMode, default="reads", autoAsReads=True) == "fragments" else 0


def _resolveBamFlagExclude(samFlagExclude: int, bamInputMode: str | None) -> int:
    resolvedFlagExclude = int(samFlagExclude or 0)
    if normalization.validateBamInputMode(bamInputMode, default="reads", autoAsReads=True) == "read1":
        resolvedFlagExclude |= 8 | 128
    return resolvedFlagExclude


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
    bamInputMode: str | None = "reads",
    samFlagExclude: int = 0,
    minMappingQuality: int | None = 0,
    minTemplateLength: int | None = -1,
    maxInsertSize: int | None = 1000,
    extendBP: int | None = 0,
    intervalSizeBP: int | None = None,
    countReadLength: int | None = None,
    fragmentsUseReadSupport: bool = False,
    barcodeTag: str | None = None,
    effectiveGenomeSizeIsFinal: bool = False,
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
    if excludeChroms is not None and not effectiveGenomeSizeIsFinal:
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
            raise ValueError("CRAM inputs are unsupported.")
        sourceKind = "BAM"
    sourceKind = normalization.validateChoice(
        sourceKind, default="BAM", supported=constants.SUPPORTED_SOURCE_KINDS, paramName="sourceKind"
    )
    resolvedCountMode = normalization.validateCountMode(countMode, "coverage")
    conservedFractional = (
        resolvedCountMode == COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
    if sourceKind == "FRAGMENTS":
        if not conservedFractional:
            raise ValueError(
                "EGS/RPGC normalization for fragments requires "
                "conservedFractionalOverlap count mode"
            )
        if fragmentsUseReadSupport:
            raise ValueError(
                "EGS/RPGC normalization for fragments requires "
                "fragmentsUseReadSupport=False"
            )
        if normalization.validateChoice(
            fragmentsGroupNorm, default="NONE", supported=constants.COUNTING_SUPPORTED_FRAGMENTS_GROUP_NORMS, paramName="fragmentsGroupNorm"
        ) != "NONE":
            raise ValueError(
                "EGS/RPGC normalization for fragments requires "
                "fragmentsGroupNorm=NONE"
            )
    if int(oneReadPerBin) != 0:
        raise ValueError("EGS/RPGC normalization does not support oneReadPerBin")
    if resolvedCountMode != "coverage" and not conservedFractional:
        raise ValueError(
            "EGS/RPGC normalization requires coverage or "
            "conservedFractionalOverlap count mode"
        )

    denominatorExcludeChroms = excludeChroms
    if sourceKind == "FRAGMENTS":
        if chromSizesFile is None:
            raise ValueError("Fragments EGS/RPGC normalization requires chromSizesFile")
        territoryChroms = getChromSizesDict(chromSizesFile)
        denominatorExcludeChroms = list(excludeChroms or ())
        denominatorExcludeChroms.extend(
            chrom
            for chrom in ccounts.ccounts_getChromosomeNames(bamFile, sourceKind=sourceKind)
            if chrom not in territoryChroms and chrom not in denominatorExcludeChroms
        )
    countReadLength = int(countReadLength if countReadLength is not None else readLength)
    mappedCountArgs = {
        "excludeChromosomes": denominatorExcludeChroms,
        "threadCount": samThreads,
        "sourceKind": sourceKind,
        "barcodeAllowListFile": barcodeAllowListFile or "",
        "barcodeTag": barcodeTag if barcodeTag is not None else "CB",
        "countMode": resolvedCountMode,
        "oneReadPerBin": oneReadPerBin,
        "flagExclude": _resolveBamFlagExclude(samFlagExclude, bamInputMode),
        "minMappingQuality": int(minMappingQuality or 0),
        "minTemplateLength": int(
            minTemplateLength if minTemplateLength is not None else -1
        ),
        "maxInsertSize": int(maxInsertSize or 0),
        "pairedEndMode": _bamPairedEndMode(bamInputMode),
        "readLength": countReadLength,
        "extendBP": int(extendBP or 0),
        "fragmentsUseReadSupport": fragmentsUseReadSupport,
    }
    totalMappedSpanBP = 0
    if conservedFractional:
        totalMappedReads, _ = ccounts.ccounts_getAlignmentMappedReadCount(
            bamFile,
            **mappedCountArgs,
        )
    else:
        totalMappedReads, _, totalMappedSpanBP = (
            ccounts.ccounts_getAlignmentMappedReadCount(
                bamFile,
                returnSpanBP=1,
                **mappedCountArgs,
            )
        )
    if totalMappedReads <= 0 or effectiveGenomeSize <= 0:
        raise ValueError(
            f"Negative EGS after removing excluded chromosomes or no mapped reads: EGS={effectiveGenomeSize}, totalMappedReads={totalMappedReads}, totalMappedSpanBP={totalMappedSpanBP}."
        )
    if conservedFractional:
        if intervalSizeBP is None or int(intervalSizeBP) <= 0:
            raise ValueError(
                "EGS/RPGC normalization with conservedFractionalOverlap requires "
                "a positive intervalSizeBP"
            )
        # Normalize by event-counts, times bin size ( not fragment size!)
        return round(effectiveGenomeSize / (totalMappedReads * int(intervalSizeBP)), 5)
    if totalMappedSpanBP <= 0:
        raise ValueError(
            f"Negative EGS after removing excluded chromosomes or no mapped span: EGS={effectiveGenomeSize}, totalMappedReads={totalMappedReads}, totalMappedSpanBP={totalMappedSpanBP}."
        )

    return round(effectiveGenomeSize / totalMappedSpanBP, 5)


def getScaleFactorPerMillion(
    bamFile: str,
    excludeChroms: List[str],
    intervalSizeBP: int,
    normMethod: str = "RPKM",
    sourceKind: str | None = None,
    barcodeAllowListFile: str | None = None,
    countMode: str | None = None,
    oneReadPerBin: int = 0,
    samThreads: int = 0,
    groupCellCount: int | None = None,
    fragmentsGroupNorm: str | None = None,
    bamInputMode: str | None = "reads",
    samFlagExclude: int = 0,
    minMappingQuality: int | None = 0,
    minTemplateLength: int | None = -1,
    maxInsertSize: int | None = 1000,
    readLength: int | None = 0,
    extendBP: int | None = 0,
    fragmentsUseReadSupport: bool = False,
    barcodeTag: str | None = None,
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
    normMethodUpper = normalization.validateChoice(
        normMethod, default="RPKM", supported=("CPM", "RPKM"), paramName="normMethod"
    )
    if sourceKind is None:
        if str(bamFile).lower().endswith(".cram"):
            raise ValueError("CRAM inputs are unsupported.")
        sourceKind = "BAM"
    sourceKind = normalization.validateChoice(
        sourceKind, default="BAM", supported=constants.SUPPORTED_SOURCE_KINDS, paramName="sourceKind"
    )
    resolvedCountMode = normalization.validateCountMode(countMode, "coverage")
    totalMappedReads, _ = ccounts.ccounts_getAlignmentMappedReadCount(
        bamFile,
        excludeChromosomes=excludeChroms,
        threadCount=samThreads,
        sourceKind=sourceKind,
        barcodeAllowListFile=barcodeAllowListFile or "",
        countMode=resolvedCountMode,
        oneReadPerBin=oneReadPerBin,
        flagExclude=(
            _resolveBamFlagExclude(samFlagExclude, bamInputMode)
            if sourceKind == "BAM"
            else 0
        ),
        minMappingQuality=int(minMappingQuality or 0),
        minTemplateLength=int(minTemplateLength if minTemplateLength is not None else -1),
        maxInsertSize=int(maxInsertSize or 0),
        pairedEndMode=_bamPairedEndMode(bamInputMode) if sourceKind == "BAM" else 0,
        readLength=int(readLength or 0),
        extendBP=int(extendBP or 0),
        fragmentsUseReadSupport=fragmentsUseReadSupport,
        barcodeTag=barcodeTag if barcodeTag is not None else "CB",
    )
    if totalMappedReads <= 0:
        raise ValueError(
            f"After removing reads mapping to excluded chroms, totalMappedReads is {totalMappedReads}."
        )
    scalePM = 1_000_000 / totalMappedReads
    if normMethodUpper == "RPKM":
        if intervalSizeBP <= 0:
            raise ValueError("RPKM normalization requires a positive intervalSizeBP")
        scalePM *= 1000 / intervalSizeBP
    scalePM = round(scalePM, 5)
    fragmentsGroupNorm = normalization.validateChoice(
        fragmentsGroupNorm, default="NONE", supported=constants.COUNTING_SUPPORTED_FRAGMENTS_GROUP_NORMS, paramName="fragmentsGroupNorm"
    )
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
    fixControl: bool = False,
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
    bamInputModeA: str | None = "reads",
    bamInputModeB: str | None = "reads",
    samFlagExclude: int = 0,
    minMappingQuality: int | None = 0,
    minTemplateLength: int | None = -1,
    maxInsertSize: int | None = 1000,
    extendBPA: int | None = 0,
    extendBPB: int | None = 0,
    countReadLengthA: int | None = None,
    countReadLengthB: int | None = None,
    fragmentsUseReadSupportA: bool = False,
    fragmentsUseReadSupportB: bool = False,
    barcodeTagA: str | None = None,
    barcodeTagB: str | None = None,
    effectiveGenomeSizeIsFinalA: bool = False,
    effectiveGenomeSizeIsFinalB: bool = False,
    precomputedScaleFactors: tuple[float | None, float | None] | None = None,
) -> Tuple[float, float]:
    r"""Scale treatment:control data to a common sequencing depth.

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
    :param: normMethod: Normalization method to use ("EGS", "RPGC", "RPKM", or "CPM").
    :type normMethod: str
    This follows standard treatment/control normalization: the deeper sample is
    scaled down to the shallower sample, and the shallower sample is left at
    full depth. If ``fixControl`` is true, the control is always left at full
    depth and only the treatment may be downscaled.

    :return: Tuple of scale factors for treatment and control samples.
    :rtype: Tuple[float, float]
    """

    normMethodUpper = normalization.validateChoice(
        normMethod, default="EGS", supported=("RPKM", "CPM", "EGS", "RPGC"), paramName="normMethod"
    )
    scaleFactors = []
    sources = (
        (bamFileA, effectiveGenomeSizeA, readLengthA, sourceKindA, barcodeAllowListFileA,
         countModeA, groupCellCountA, bamInputModeA, extendBPA, countReadLengthA,
         fragmentsUseReadSupportA, barcodeTagA, effectiveGenomeSizeIsFinalA),
        (bamFileB, effectiveGenomeSizeB, readLengthB, sourceKindB, barcodeAllowListFileB,
         countModeB, groupCellCountB, bamInputModeB, extendBPB, countReadLengthB,
         fragmentsUseReadSupportB, barcodeTagB, effectiveGenomeSizeIsFinalB),
    )
    if precomputedScaleFactors is not None and len(precomputedScaleFactors) != 2:
        raise ValueError("precomputedScaleFactors must contain two values")
    for sourceIndex, (bamFile, effectiveGenomeSize, readLength, sourceKind, barcodeAllowListFile,
         countMode, groupCellCount, bamInputMode, extendBP, countReadLength,
         fragmentsUseReadSupport, barcodeTag, effectiveGenomeSizeIsFinal) in enumerate(sources):
        if precomputedScaleFactors is not None and precomputedScaleFactors[sourceIndex] is not None:
            scaleFactors.append(normalization.validatePositiveFloat(
                precomputedScaleFactors[sourceIndex], "precomputedScaleFactors"
            ))
            continue
        sharedArgs = dict(
            sourceKind=sourceKind,
            barcodeAllowListFile=barcodeAllowListFile,
            countMode=countMode,
            oneReadPerBin=oneReadPerBin,
            groupCellCount=groupCellCount,
            fragmentsGroupNorm=fragmentsGroupNorm,
            bamInputMode=bamInputMode,
            samFlagExclude=samFlagExclude,
            minMappingQuality=minMappingQuality,
            minTemplateLength=minTemplateLength,
            maxInsertSize=maxInsertSize,
            extendBP=extendBP,
            fragmentsUseReadSupport=fragmentsUseReadSupport,
            barcodeTag=barcodeTag,
        )
        if normMethodUpper in ("RPKM", "CPM"):
            scaleFactor = getScaleFactorPerMillion(
                bamFile, excludeChroms, intervalSizeBP,
                normMethod=normMethodUpper,
                samThreads=samThreads,
                readLength=countReadLength if countReadLength is not None else readLength,
                **sharedArgs,
            )
        else:
            scaleFactor = getScaleFactor1x(
                bamFile, effectiveGenomeSize, readLength, excludeChroms,
                chromSizesFile, samThreads,
                intervalSizeBP=intervalSizeBP,
                countReadLength=countReadLength,
                effectiveGenomeSizeIsFinal=effectiveGenomeSizeIsFinal,
                **sharedArgs,
            )
        scaleFactors.append(scaleFactor)
    scaleFactorA, scaleFactorB = scaleFactors

    depthA = 1.0 / scaleFactorA if scaleFactorA > 0.0 else 0.0
    depthB = 1.0 / scaleFactorB if scaleFactorB > 0.0 else 0.0
    if depthA <= 0.0 or depthB <= 0.0:
        raise ValueError(
            "Treatment/control depth estimates must be positive: "
            f"treatment={depthA}, control={depthB}."
        )

    if fixControl:
        # Keep control full depth. Downscale treatment only when it is deeper.
        scaleFactorB = 1.0
        if depthA > depthB:
            scaleFactorA = depthB / depthA
        else:
            scaleFactorA = 1.0
    else:
        # pair normalization: downscale the deeper sample to the
        # shallower sample, without upscaling either track.
        targetDepth = min(depthA, depthB)
        scaleFactorA = targetDepth / depthA
        scaleFactorB = targetDepth / depthB

    ratio = max(depthA, depthB) / max(1.0e-12, min(depthA, depthB))
    if ratio > 5.0:
        logger.warning(
            f"Treatment/control sequencing depths differ > 5x....\n"
            f"\n\tAre effective genome sizes {effectiveGenomeSizeA} and {effectiveGenomeSizeB} correct?"
            f"\n\tAre read/fragment lengths {readLengthA},{readLengthB} correct?"
        )

    return scaleFactorA, scaleFactorB
