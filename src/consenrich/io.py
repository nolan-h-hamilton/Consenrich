"""Input preparation, bedGraph, and output-conversion helpers."""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import consenrich.ccounts as ccounts
import consenrich.constants as constants
import consenrich.core as core
import consenrich.misc_util as misc_util
from ._version import __version__
from ._runtime import (
    get_available_memory_bytes as _sharedGetAvailableMemoryBytes,
    get_munc_worker_count as _sharedGetMuncWorkerCount,
    get_small_worker_count as _sharedGetSmallWorkerCount,
    thread_map as _sharedThreadMap,
)

logger = logging.getLogger(__name__)


def _getSmallWorkerCount(taskCount: int, maxWorkers: int = 4) -> int:
    return _sharedGetSmallWorkerCount(taskCount, max_workers=maxWorkers)


def _getAvailableMemoryBytes() -> int | None:
    return _sharedGetAvailableMemoryBytes()


_MEMORY_UNSET = object()


def _getMuncWorkerCount(
    numSamples: int,
    numIntervals: int,
    sharedArrays=(),
    availableMemoryBytes=_MEMORY_UNSET,
    logger_=logger,
) -> int:
    if availableMemoryBytes is _MEMORY_UNSET:
        availableMemoryBytes = _sharedGetAvailableMemoryBytes()
    return _sharedGetMuncWorkerCount(
        numSamples,
        numIntervals,
        shared_arrays=sharedArrays,
        available_memory_bytes=availableMemoryBytes,
        logger=logger_,
    )


def _threadMap(
    items,
    func,
    label: str,
    allowThreads: bool = True,
    minItems: int = 4,
    maxWorkers: int = 4,
):
    return _sharedThreadMap(
        items,
        func,
        label,
        logger=logger,
        allow_threads=allowThreads,
        min_items=minItems,
        max_workers=maxWorkers,
    )


@lru_cache(maxsize=1)
def _pyBigWigAvailable() -> bool:
    try:
        import pyBigWig  # noqa: F401
    except Exception:
        return False
    return True


def _checkSF(sf, logger, cut_=3.0):

    sf = np.asarray(sf, dtype=float)
    bad = ~np.isfinite(sf) | (sf <= 0)
    if np.any(bad):
        logger.warning(
            "Scaling factors contain non-finite or non-positive values: "
            f"nBad={bad.sum()}/{sf.size}: {sf[bad][:10]} ..."
        )

    v = sf[np.isfinite(sf) & (sf > 0)]
    if v.size < 3:
        return

    p05, p50, p95 = np.percentile(v, [5, 50, 95])
    ratio = p95 / p05

    if ratio > cut_:
        logger.warning(
            "Sample scaling factors (`countingParams.normMethod: SF`) used for "
            "library size/coverage normalization are heterogeneous: "
            "median=%g, p95/p05=%f > %f. __IF this is unexpected__, consider "
            "inspecting the alignment files.",
            p50,
            ratio,
            cut_,
        )


def _resolveExtendFrom5pBPPairs(
    treatmentExtendFrom5pBP: Optional[Sequence[Union[int, float]]],
    controlExtendFrom5pBP: Optional[Sequence[Union[int, float]]],
) -> Tuple[List[int], List[int]]:
    r"""Assign consistent single-end extension lengths to treatment and control BAM files.

    For single-end data, cross-correlation-based fragment estimates for control inputs
    can be much smaller than for treatment samples due to lack of structure. This creates
    artifacts during signal quantification and normalization steps, and it's common to use
    the treatment fragment length for both treatment and control samples. So we do that here.
    """

    if not treatmentExtendFrom5pBP:
        logger.warning("No treatment extension lengths provided...returning [],[]")
        return [], []

    n_treat = len(treatmentExtendFrom5pBP)

    if controlExtendFrom5pBP:
        if len(controlExtendFrom5pBP) == 1 and n_treat > 1:
            controlExtendFrom5pBP = list(controlExtendFrom5pBP) * n_treat
            logger.info(
                "Only one control extension length provided: broadcasting this value for all control BAM files."
            )
        elif len(controlExtendFrom5pBP) != n_treat:
            logger.warning(
                "Sizes of treatment and control extension length lists are incompatible...returning [],[]"
            )
            return [], []
        else:
            controlExtendFrom5pBP = list(controlExtendFrom5pBP)
    else:
        controlExtendFrom5pBP = list(treatmentExtendFrom5pBP)

    finalTreatment = [int(x) for x in treatmentExtendFrom5pBP]
    finalControl = [int(x) for x in treatmentExtendFrom5pBP]

    return finalTreatment, finalControl

def _listOrEmpty(list_):
    if list_ is None:
        return []
    return list_


def _normalizeScaleFactorList(
    scaleFactors: Optional[Sequence[float]],
    expectedCount: int,
    paramName: str,
) -> Optional[List[float]]:
    if scaleFactors is None:
        return None
    if expectedCount < 1:
        raise ValueError(f"`{paramName}` was provided but no matching inputs exist.")
    normalized = [float(scaleFactor) for scaleFactor in scaleFactors]
    if len(normalized) == expectedCount:
        return normalized
    if len(normalized) == 1:
        return normalized * expectedCount
    raise ValueError(
        f"`{paramName}` must contain 1 value or {expectedCount} values; "
        f"got {len(normalized)}."
    )


def _expandWildCards(pathList: List[str]) -> List[str]:
    expandedList: List[str] = []
    for pathEntry in pathList:
        if "*" in pathEntry or "?" in pathEntry or "[" in pathEntry:
            matchedList = sorted(glob.glob(pathEntry))
            if matchedList:
                expandedList.extend(matchedList)
            else:
                expandedList.append(pathEntry)
        else:
            expandedList.append(pathEntry)
    return expandedList


def _inferSourceKind(path: str) -> str:
    lowerPath = str(path).lower()
    if lowerPath.endswith(".cram"):
        raise ValueError("CRAM inputs are no longer supported.")
    if lowerPath.endswith((".bedgraph", ".bedgraph.gz", ".bdg", ".bdg.gz")):
        return core.BEDGRAPH_SOURCE_KIND
    if "fragments" in os.path.basename(lowerPath):
        return "FRAGMENTS"
    return "BAM"


def _isCompressedBedGraphPath(path: str) -> bool:
    return str(path).lower().endswith((".gz", ".bgz", ".bgzf"))


def _bedGraphTabixIndexExists(path: str) -> bool:
    return os.path.exists(f"{path}.tbi") or os.path.exists(f"{path}.csi")


def _existingIndexedBedGraphPath(path: str) -> Optional[str]:
    if _isCompressedBedGraphPath(path) and _bedGraphTabixIndexExists(path):
        return path
    compressedPath = f"{path}.gz"
    if (
        not _isCompressedBedGraphPath(path)
        and os.path.exists(compressedPath)
        and _bedGraphTabixIndexExists(compressedPath)
    ):
        return compressedPath
    return None


def _buildBedGraphTabixIndex(path: str) -> str:
    sourcePath = str(path)
    targetPath = (
        sourcePath if _isCompressedBedGraphPath(sourcePath) else f"{sourcePath}.gz"
    )
    sourceDir = os.path.dirname(os.path.abspath(sourcePath)) or "."
    tempCompressedPath = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="consenrich_bedgraph_index_",
            suffix=".bedGraph.gz",
            delete=False,
            dir=sourceDir,
        ) as tempCompressedHandle:
            tempCompressedPath = tempCompressedHandle.name

        ccounts.ccounts_indexBedGraphPath(sourcePath, tempCompressedPath)
        os.replace(tempCompressedPath, targetPath)
        for indexSuffix in [".tbi", ".csi"]:
            tempIndexPath = f"{tempCompressedPath}{indexSuffix}"
            targetIndexPath = f"{targetPath}{indexSuffix}"
            if os.path.exists(tempIndexPath):
                os.replace(tempIndexPath, targetIndexPath)
        tempCompressedPath = ""
        return targetPath
    finally:
        for tempPath in [
            tempCompressedPath,
            f"{tempCompressedPath}.tbi" if tempCompressedPath else "",
            f"{tempCompressedPath}.csi" if tempCompressedPath else "",
        ]:
            if tempPath and os.path.exists(tempPath):
                try:
                    os.remove(tempPath)
                except OSError:
                    pass


def _ensureBedGraphIndexed(path: str, logger_: logging.Logger = logger) -> str:
    indexedPath = _existingIndexedBedGraphPath(path)
    if indexedPath is not None:
        if indexedPath != path:
            logger_.info(
                "Using existing indexed bedGraph copy %s for unindexed input %s.",
                indexedPath,
                path,
            )
        return indexedPath

    targetPath = path if _isCompressedBedGraphPath(path) else f"{path}.gz"
    logger_.info(
        "BedGraph input %s has no tabix index; creating %s and tabix index for random access.",
        path,
        targetPath,
    )
    return _buildBedGraphTabixIndex(path)


def _prepareBedGraphSources(
    sources: Sequence[core.inputSource],
    logger_: logging.Logger = logger,
) -> List[core.inputSource]:
    preparedSources: List[core.inputSource] = []
    for source in sources:
        if str(source.sourceKind).upper() == core.BEDGRAPH_SOURCE_KIND:
            indexedPath = _ensureBedGraphIndexed(source.path, logger_)
            if indexedPath != source.path:
                source = source._replace(path=indexedPath)
        preparedSources.append(source)
    return preparedSources


def _normalizeSourceKind(sourceKind: Optional[str], path: str) -> str:
    if sourceKind is None:
        normalizedKind = _inferSourceKind(path)
    else:
        normalizedKind = str(sourceKind).strip().upper()
        if normalizedKind == "AUTO":
            normalizedKind = _inferSourceKind(path)

    if normalizedKind == "CRAM":
        raise ValueError("CRAM inputs are no longer supported.")
    if normalizedKind not in core.SUPPORTED_SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind `{normalizedKind}` for `{path}`.")
    return normalizedKind


def _coerceInputSource(
    sourceConfig: Union[str, Mapping[str, Any]],
    defaultRole: str,
    defaultBarcodeTag: str | None = None,
    defaultFragmentPositionMode: str | None = None,
) -> core.inputSource:
    if isinstance(sourceConfig, str):
        path = sourceConfig
        sourceKind = None
        sampleName = None
        role = defaultRole
        barcodeTag = defaultBarcodeTag
        barcodeAllowListFile = None
        barcodeGroupMapFile = None
        selectGroups = None
        countMode = None
        bamInputMode = None
        fragmentPositionMode = defaultFragmentPositionMode
    elif isinstance(sourceConfig, Mapping):
        path = str(sourceConfig.get("path", "")).strip()
        sourceKind = sourceConfig.get("format", sourceConfig.get("sourceKind", None))
        sampleName = sourceConfig.get("name", None)
        role = str(sourceConfig.get("role", defaultRole)).strip().lower()
        barcodeTag = sourceConfig.get("barcodeTag", defaultBarcodeTag)
        barcodeAllowListFile = sourceConfig.get("barcodeAllowListFile", None)
        barcodeGroupMapFile = sourceConfig.get("barcodeGroupMapFile", None)
        selectGroups = sourceConfig.get("selectGroups", None)
        countMode = sourceConfig.get("countMode", None)
        bamInputMode = sourceConfig.get("bamInputMode", None)
        fragmentPositionMode = sourceConfig.get(
            "fragmentPositionMode",
            defaultFragmentPositionMode,
        )
    else:
        raise TypeError("Each input source must be a path string or a mapping.")

    if not path:
        raise ValueError("Each input source requires a non-empty `path`.")

    if role not in ["treatment", "control"]:
        raise ValueError(f"Unsupported source role `{role}` for `{path}`.")

    if selectGroups is not None and not isinstance(selectGroups, list):
        raise ValueError(f"`selectGroups` must be a list for `{path}`.")

    pathList = _expandWildCards([path])
    if len(pathList) != 1:
        raise ValueError(
            f"Input source `{path}` expanded to {len(pathList)} paths. Use one entry per source."
        )
    normalizedPath = pathList[0]

    return core.inputSource(
        path=normalizedPath,
        sourceKind=_normalizeSourceKind(sourceKind, normalizedPath),
        role=role,
        sampleName=sampleName,
        barcodeTag=barcodeTag,
        barcodeAllowListFile=barcodeAllowListFile,
        barcodeGroupMapFile=barcodeGroupMapFile,
        selectGroups=selectGroups,
        countMode=countMode,
        bamInputMode=bamInputMode,
        fragmentPositionMode=fragmentPositionMode,
    )


def _buildPathInputSources(pathList: List[str], role: str) -> List[core.inputSource]:
    return [
        core.inputSource(
            path=path,
            sourceKind=_normalizeSourceKind(None, path),
            role=role,
        )
        for path in _expandWildCards(pathList)
    ]


def _getSourceCountMode(
    source: core.inputSource,
    defaultBamCountMode: str = "coverage",
    defaultFragmentCountMode: str = "coverage",
) -> str:
    if str(source.sourceKind).upper() == core.BEDGRAPH_SOURCE_KIND:
        return "coverage"
    countMode = core._normalizeCountMode(
        source.countMode,
        (
            defaultFragmentCountMode
            if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND
            else defaultBamCountMode
        ),
    )
    if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND:
        if countMode in {"ffp", "ffp-center"}:
            raise ValueError(f"countMode `{countMode}` requires BAM input")
        core._normalizeFragmentPositionMode(source.fragmentPositionMode)
    return countMode


def _prepareFragmentsNormalizationMetadata(
    sources: List[core.inputSource],
) -> tuple[List[str | None], List[int | None], List[str]]:
    allowListPaths: List[str | None] = []
    selectedCellCounts: List[int | None] = []
    tempPaths: List[str] = []

    for source in sources:
        if str(source.sourceKind).upper() != core.FRAGMENTS_SOURCE_KIND:
            allowListPaths.append(None)
            selectedCellCounts.append(None)
            continue
        allowListPath, tempPath = core._writeFragmentsAllowList(source)
        allowListPaths.append(allowListPath)
        selectedCellCount = core.getFragmentsSelectedBarcodeCount(source)
        if selectedCellCount is None:
            selectedCellCount = core.ccounts.ccounts_getFragmentCellCount(
                source.path,
                barcodeAllowListFile=allowListPath or "",
            )
        selectedCellCounts.append(selectedCellCount)
        if tempPath is not None:
            tempPaths.append(tempPath)

    return allowListPaths, selectedCellCounts, tempPaths


def checkControlsPresent(inputArgs: core.inputParams) -> bool:
    """Check if control BAM files are present in the input arguments.

    :param inputArgs: core.inputParams object
    :return: True if control BAM files are present, False otherwise.
    """
    return (
        bool(inputArgs.bamFilesControl)
        and isinstance(inputArgs.bamFilesControl, list)
        and len(inputArgs.bamFilesControl) > 0
    )


def getReadLengths(
    inputArgs: core.inputParams,
    countingArgs: core.countingParams,
    samArgs: core.samParams,
) -> List[int]:
    r"""Get read lengths for each BAM file in the input arguments.

    :param inputArgs: core.inputParams object containing BAM file paths.
    :param countingArgs: core.countingParams object containing number of reads.
    :param samArgs: core.samParams object containing SAM thread and flag exclude parameters.
    :return: List of read lengths for each BAM file.
    """
    if not inputArgs.bamFiles:
        raise ValueError("No BAM files provided in the input arguments.")

    if not isinstance(inputArgs.bamFiles, list) or len(inputArgs.bamFiles) == 0:
        raise ValueError("bam files list is empty")

    allowThreads = int(samArgs.samThreads) <= 1

    def _getReadLengthForTask(task) -> int:
        path, sourceKind = task
        return core.getReadLength(
            path,
            100,
            1000,
            samArgs.samThreads,
            samArgs.samFlagExclude,
            sourceKind=sourceKind,
        )

    treatmentSources = _listOrEmpty(getattr(inputArgs, "treatmentSources", None))
    if treatmentSources:
        return _threadMap(
            [
                (source.path, str(source.sourceKind).upper())
                for source in treatmentSources
            ],
            _getReadLengthForTask,
            "read lengths",
            allowThreads=allowThreads,
        )

    return _threadMap(
        [(bamFile, "BAM") for bamFile in inputArgs.bamFiles],
        _getReadLengthForTask,
        "read lengths",
        allowThreads=allowThreads,
    )


def checkMatchingEnabled(matchingArgs: core.matchingParams) -> bool:
    return bool(getattr(matchingArgs, "enabled", True))


def _inferMatchingUncertaintyBedGraph(matchBedGraph: str) -> Optional[str]:
    statePath = Path(matchBedGraph)
    if "_state." not in statePath.name:
        return None
    uncertaintyPath = statePath.with_name(
        statePath.name.replace("_state.", "_uncertainty.", 1)
    )
    if uncertaintyPath.exists():
        return str(uncertaintyPath)
    return None


def getEffectiveGenomeSizes(
    genomeArgs: core.genomeParams, readLengths: List[int]
) -> List[int]:
    r"""Get effective genome sizes for the given genome name and read lengths.
    :param genomeArgs: core.genomeParams object
    :param readLengths: List of read lengths for which to get effective genome sizes.
    :return: List of effective genome sizes corresponding to the read lengths.
    """
    genomeName = genomeArgs.genomeName
    if not genomeName or not isinstance(genomeName, str):
        raise ValueError("Genome name must be a non-empty string.")

    if not isinstance(readLengths, list) or len(readLengths) == 0:
        raise ValueError(
            "Read lengths must be a non-empty list. Try calling `getReadLengths` first."
        )
    return [
        (
            0
            if int(readLength) <= 0
            else constants.getEffectiveGenomeSize(genomeName, readLength)
        )
        for readLength in readLengths
    ]

def convertBedGraphToBigWig(
    experimentName,
    chromSizesFile,
    suffixes: Optional[List[str]] = None,
):
    if suffixes is None:
        # at least look for `state` bedGraph
        suffixes = ["state"]
    logger.info(
        "Attempting to generate bigWig files from bedGraph format via pyBigWig..."
    )
    for suffix in suffixes:
        bedgraph = f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
        if not os.path.exists(bedgraph):
            logger.warning(
                f"bedGraph file {bedgraph} does not exist. Skipping bigWig conversion."
            )
            continue
        if not os.path.exists(chromSizesFile):
            logger.warning(
                f"{chromSizesFile} does not exist. Skipping bigWig conversion."
            )
            return
        chromOrder = _readChromOrder(chromSizesFile)
        try:
            _validateBedGraphSorted(bedgraph, chromOrder=chromOrder)
        except Exception as e:
            logger.warning(
                "bedGraph %s failed sorted validation before bigWig conversion; "
                "sorting as a fallback:\n%s",
                bedgraph,
                e,
            )
            try:
                _sortBedGraphInPlace(bedgraph, chromOrder=chromOrder)
            except Exception as sortError:
                logger.warning(
                    f"Failed to sort {bedgraph} before bigWig conversion:\n{sortError}"
                )
                continue
        bigwig = f"{experimentName}_consenrich_{suffix}.v{__version__}.bw"
        logger.info(f"Start: {bedgraph} --> {bigwig}...")
        try:
            _convertBedGraphToBigWigPyBigWig(
                bedgraph,
                chromSizesFile,
                bigwig,
            )
        except Exception as e:
            logger.warning(
                f"pyBigWig bedGraph-->bigWig conversion for {bedgraph} raised:\n{e}\n"
            )
            continue
        if os.path.exists(bigwig) and os.path.getsize(bigwig) > 100:
            logger.info(f"Finished: converted {bedgraph} to {bigwig}.")


def _readChromSizes(chromSizesFile: str) -> List[Tuple[str, int]]:
    chromSizes: List[Tuple[str, int]] = []
    chromSizeByName: Dict[str, int] = {}
    with open(chromSizesFile, "r", encoding="utf-8") as handle:
        for lineNumber, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split()
            if len(parts) == 0 or parts[0].startswith("#"):
                continue
            if len(parts) < 2:
                raise ValueError(
                    f"Malformed chromosome sizes row {lineNumber} in {chromSizesFile}"
                )
            chrom = str(parts[0])
            try:
                chromSize = int(parts[1])
            except ValueError as e:
                raise ValueError(
                    f"Invalid chromosome size on row {lineNumber} in {chromSizesFile}"
                ) from e
            if chromSize <= 0:
                raise ValueError(
                    f"Chromosome {chrom} has non-positive size on row {lineNumber}"
                )
            if chrom in chromSizeByName:
                raise ValueError(f"Duplicate chromosome {chrom} in {chromSizesFile}")
            chromSizes.append((chrom, chromSize))
            chromSizeByName[chrom] = chromSize
    if len(chromSizes) == 0:
        raise ValueError(f"No chromosome sizes found in {chromSizesFile}")
    return chromSizes


def _readChromOrder(chromSizesFile: str) -> List[str]:
    return [chrom for chrom, _size in _readChromSizes(chromSizesFile)]


def _convertBedGraphToBigWigPyBigWig(
    bedgraphPath: str,
    chromSizesFile: str,
    bigwigPath: str,
    chunkSize: int = 200_000,
) -> None:
    try:
        import pyBigWig
    except Exception as e:
        raise RuntimeError(
            "pyBigWig is not installed; cannot convert bedGraph files to bigWig"
        ) from e

    chromSizes = _readChromSizes(chromSizesFile)
    chromSizeByName = dict(chromSizes)
    chromRankByName = {chrom: rank for rank, (chrom, _size) in enumerate(chromSizes)}

    chunkSize_ = max(int(chunkSize), 1)
    outDir = os.path.dirname(os.path.abspath(bigwigPath)) or "."
    tempPath = ""
    bw = None
    seenEntry = False
    lastChrom = ""
    lastStart = -1
    lastEnd = -1
    try:
        with tempfile.NamedTemporaryFile(
            prefix="consenrich_bigwig_",
            suffix=".bw",
            delete=False,
            dir=outDir,
        ) as tempHandle:
            tempPath = tempHandle.name
        bw = pyBigWig.open(tempPath, "w")
        bw.addHeader(chromSizes)
        chroms: List[str] = []
        starts: List[int] = []
        ends: List[int] = []
        values: List[float] = []
        with open(bedgraphPath, "r", encoding="utf-8") as handle:
            for lineNumber, line in enumerate(handle, start=1):
                stripped = line.strip()
                if (
                    not stripped
                    or stripped.startswith("#")
                    or stripped == "track"
                    or stripped.startswith("track ")
                    or stripped == "browser"
                    or stripped.startswith("browser ")
                ):
                    continue
                parts = stripped.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Malformed bedGraph row {lineNumber} in {bedgraphPath}: "
                        "expected 4 columns"
                    )
                chrom = str(parts[0])
                if chrom not in chromSizeByName:
                    raise ValueError(
                        f"Chromosome {chrom} on bedGraph row {lineNumber} is not "
                        f"present in {chromSizesFile}"
                    )
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bedGraph coordinates on row {lineNumber} in "
                        f"{bedgraphPath}"
                    ) from e
                try:
                    value = float(parts[3])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bedGraph value on row {lineNumber} in {bedgraphPath}"
                    ) from e
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite bedGraph value on row {lineNumber} in {bedgraphPath}"
                    )
                if start < 0:
                    raise ValueError(
                        f"Negative start coordinate on bedGraph row {lineNumber}"
                    )
                if end <= start:
                    raise ValueError(
                        f"End coordinate must be greater than start on bedGraph row "
                        f"{lineNumber}"
                    )
                if end > chromSizeByName[chrom]:
                    raise ValueError(
                        f"End coordinate {end} on bedGraph row {lineNumber} exceeds "
                        f"{chrom} size of {chromSizeByName[chrom]}"
                    )
                chromRank = chromRankByName[chrom]
                if seenEntry:
                    lastRank = chromRankByName[lastChrom]
                    if chromRank < lastRank or (
                        chrom == lastChrom and start < lastStart
                    ):
                        raise ValueError(
                            f"bedGraph input is not sorted at row {lineNumber}; sort "
                            "by chromosome sizes order, then start/end"
                        )
                    if chrom == lastChrom and start < lastEnd:
                        raise ValueError(
                            f"Overlapping bedGraph interval at row {lineNumber}"
                        )
                chroms.append(chrom)
                starts.append(start)
                ends.append(end)
                values.append(value)
                seenEntry = True
                lastChrom = chrom
                lastStart = start
                lastEnd = end
                if len(chroms) >= chunkSize_:
                    bw.addEntries(chroms, starts, ends=ends, values=values)
                    chroms.clear()
                    starts.clear()
                    ends.clear()
                    values.clear()
        if len(chroms) > 0:
            bw.addEntries(chroms, starts, ends=ends, values=values)
        if not seenEntry:
            raise ValueError(f"No bedGraph intervals found in {bedgraphPath}")
        bw.close()
        bw = None
        os.replace(tempPath, bigwigPath)
    finally:
        if bw is not None:
            bw.close()
        if tempPath and os.path.exists(tempPath):
            try:
                os.remove(tempPath)
            except OSError:
                pass


def _validateBedGraphSorted(
    bedgraphPath: str,
    chromOrder: Optional[Sequence[str]] = None,
) -> None:
    if not os.path.exists(bedgraphPath) or os.path.getsize(bedgraphPath) == 0:
        return

    chromRankByName = (
        {str(chrom): rank for rank, chrom in enumerate(chromOrder)}
        if chromOrder is not None
        else None
    )
    seenEntry = False
    seenChroms: set[str] = set()
    lastChrom = ""
    lastStart = -1
    lastEnd = -1
    lastRank = -1

    with open(bedgraphPath, "r", encoding="utf-8") as handle:
        for lineNumber, line in enumerate(handle, start=1):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped == "track"
                or stripped.startswith("track ")
                or stripped == "browser"
                or stripped.startswith("browser ")
            ):
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Malformed bedGraph row {lineNumber} in {bedgraphPath}: "
                    "expected 4 columns"
                )
            chrom = str(parts[0])
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError as e:
                raise ValueError(
                    f"Invalid bedGraph coordinates on row {lineNumber} in "
                    f"{bedgraphPath}"
                ) from e
            if start < 0:
                raise ValueError(
                    f"Negative start coordinate on bedGraph row {lineNumber}"
                )
            if end <= start:
                raise ValueError(
                    f"End coordinate must be greater than start on bedGraph row "
                    f"{lineNumber}"
                )
            if chromRankByName is None:
                chromRank = -1
            else:
                if chrom not in chromRankByName:
                    raise ValueError(
                        f"Chromosome {chrom} on bedGraph row {lineNumber} is not "
                        "present in the expected chromosome order"
                    )
                chromRank = chromRankByName[chrom]
            if seenEntry:
                if chrom == lastChrom:
                    if start < lastStart:
                        raise ValueError(
                            f"bedGraph input is not sorted at row {lineNumber}"
                        )
                    if start < lastEnd:
                        raise ValueError(
                            f"Overlapping bedGraph interval at row {lineNumber}"
                        )
                else:
                    if chrom in seenChroms:
                        raise ValueError(
                            f"Chromosome {chrom} reappears on bedGraph row "
                            f"{lineNumber}"
                        )
                    if chromRankByName is None:
                        if chrom < lastChrom:
                            raise ValueError(
                                f"bedGraph input is not sorted at row {lineNumber}"
                            )
                    elif chromRank < lastRank:
                        raise ValueError(
                            f"bedGraph input is not in chromosome order at row "
                            f"{lineNumber}"
                        )
            seenEntry = True
            seenChroms.add(chrom)
            lastChrom = chrom
            lastStart = start
            lastEnd = end
            lastRank = chromRank


def _sortBedGraphInPlace(
    bedgraphPath: str,
    chromOrder: Optional[Sequence[str]] = None,
) -> None:
    if not os.path.exists(bedgraphPath) or os.path.getsize(bedgraphPath) == 0:
        return

    sortPath = shutil.which("sort")
    if sortPath is not None and chromOrder is None:
        tempPath = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="consenrich_sort_",
                suffix=".bedGraph",
                delete=False,
                dir=os.path.dirname(os.path.abspath(bedgraphPath)) or ".",
            ) as tempHandle:
                tempPath = tempHandle.name
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            with open(tempPath, "w", encoding="utf-8") as outHandle:
                subprocess.run(
                    [
                        sortPath,
                        "-k1,1",
                        "-k2,2n",
                        "-k3,3n",
                        bedgraphPath,
                    ],
                    check=True,
                    stdout=outHandle,
                    env=env,
                )
            os.replace(tempPath, bedgraphPath)
            return
        finally:
            if tempPath and os.path.exists(tempPath):
                try:
                    os.remove(tempPath)
                except OSError:
                    pass

    headerLines: List[str] = []
    recordRows: List[List[str]] = []
    with open(bedgraphPath, "r", encoding="utf-8") as handle:
        for lineNumber, line in enumerate(handle, start=1):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped == "track"
                or stripped.startswith("track ")
                or stripped == "browser"
                or stripped.startswith("browser ")
            ):
                headerLines.append(line.rstrip("\n"))
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Malformed bedGraph row {lineNumber} in {bedgraphPath}: "
                    "expected 4 columns"
                )
            recordRows.append(parts)
    if not recordRows:
        with open(bedgraphPath, "w", encoding="utf-8") as handle:
            for headerLine in headerLines:
                handle.write(f"{headerLine}\n")
        return

    df = pd.DataFrame(
        recordRows,
        columns=["chromosome", "start", "end", "value"],
    ).astype(
        {
            "chromosome": str,
            "start": np.int64,
            "end": np.int64,
            "value": np.float64,
        }
    )
    if chromOrder is not None:
        chromRankByName = {str(chrom): rank for rank, chrom in enumerate(chromOrder)}
        df["_chromRank"] = df["chromosome"].map(chromRankByName)
        unknown = sorted(df.loc[df["_chromRank"].isna(), "chromosome"].unique())
        if unknown:
            raise ValueError(
                "bedGraph contains chromosomes not present in chromosome order: "
                + ", ".join(str(chrom) for chrom in unknown[:5])
            )
        df["_chromRank"] = df["_chromRank"].astype(np.int64)
        df.sort_values(
            by=["_chromRank", "start", "end"],
            kind="mergesort",
            inplace=True,
        )
        df.drop(columns=["_chromRank"], inplace=True)
    else:
        df.sort_values(
            by=["chromosome", "start", "end"],
            kind="mergesort",
            inplace=True,
        )
    with open(bedgraphPath, "w", encoding="utf-8") as handle:
        for headerLine in headerLines:
            handle.write(f"{headerLine}\n")
        df.to_csv(
            handle,
            sep="\t",
            header=False,
            index=False,
            float_format="%.4f",
            lineterminator="\n",
        )

__all__ = [
    "checkControlsPresent",
    "checkMatchingEnabled",
    "convertBedGraphToBigWig",
    "getEffectiveGenomeSizes",
    "getReadLengths",
]
