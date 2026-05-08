#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import math
from multiprocessing.pool import ThreadPool
import pprint
import os
import tempfile
import time
from pathlib import Path
from collections.abc import Mapping
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any, Union, Sequence
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import consenrich.core as core
import consenrich.ccounts as ccounts
import consenrich.diagnostics as diagnostics
import consenrich.misc_util as misc_util
import consenrich.constants as constants
import consenrich.detrorm as detrorm
import consenrich.peaks as peaks
import consenrich.cconsenrich as cconsenrich
from . import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _fmtDiagnosticFloat(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value_ = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value_):
        return "NA"
    return f"{value_:.6g}"


def _progress(iterable, **kwargs):
    disable = kwargs.pop("disable", not sys.stderr.isatty())
    if disable:
        return iterable
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, disable=False, **kwargs)


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


def _getSmallWorkerCount(taskCount: int, maxWorkers: int = 4) -> int:
    cpuCount = os.cpu_count() or 1
    return min(taskCount, max(1, cpuCount // 2), maxWorkers)


_MEMORY_UNSET = object()


def _getAvailableMemoryBytes() -> int | None:
    try:
        import psutil  # type: ignore

        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except Exception:
        pass

    try:
        with open("/proc/meminfo", "r", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        pageSize = int(os.sysconf("SC_PAGE_SIZE"))
        total = pages * pageSize
        if total > 0:
            return total // 4
    except (AttributeError, OSError, ValueError):
        pass

    return None


def _getMuncWorkerCount(
    numSamples: int,
    numIntervals: int,
    sharedArrays=(),
    availableMemoryBytes=_MEMORY_UNSET,
    logger_=logger,
) -> int:
    numSamples = int(numSamples)
    numIntervals = int(numIntervals)
    if numSamples <= 0:
        return 1

    cpuCount = os.cpu_count() or 1
    cpuWorkers = min(numSamples, max(1, cpuCount // 2))
    if cpuWorkers <= 1:
        return 1

    if availableMemoryBytes is _MEMORY_UNSET:
        availableMemoryBytes = _getAvailableMemoryBytes()
    if availableMemoryBytes is None:
        return cpuWorkers

    try:
        availableBytes = int(availableMemoryBytes)
    except (TypeError, ValueError):
        return cpuWorkers
    if availableBytes <= 0:
        return cpuWorkers

    sharedBytes = 0
    for arr in sharedArrays:
        if arr is not None:
            sharedBytes += int(getattr(arr, "nbytes", 0) or 0)

    scratchBytes = max(
        int(64 * max(numIntervals, 0)) + (64 * 1024 * 1024),
        128 * 1024 * 1024,
    )
    memoryBudget = int(max(0, availableBytes - sharedBytes) * 0.5)
    memoryWorkers = max(1, memoryBudget // scratchBytes)
    workers = max(1, min(cpuWorkers, memoryWorkers))
    if workers < cpuWorkers and logger_ is not None:
        logger_.info(
            "munc matrix: reducing workers from %d to %d based on memory estimate "
            "(available=%.2f GiB, shared=%.2f GiB, perWorker=%.2f GiB).",
            int(cpuWorkers),
            int(workers),
            availableBytes / (1024**3),
            sharedBytes / (1024**3),
            scratchBytes / (1024**3),
        )
    return workers


def _threadMapMaybe(
    items,
    func,
    label: str,
    allowThreads: bool = True,
    minItems: int = 4,
    maxWorkers: int = 4,
):
    itemList = list(items)
    if len(itemList) == 0:
        return []

    workerCount = _getSmallWorkerCount(len(itemList), maxWorkers=maxWorkers)
    usePool = allowThreads and len(itemList) >= minItems and workerCount > 1
    if usePool:
        logger.info(
            "%s: using ThreadPool with %d workers (n=%d).",
            label,
            int(workerCount),
            int(len(itemList)),
        )
        with ThreadPool(processes=int(workerCount)) as pool:
            return list(pool.map(func, itemList))

    return [func(item) for item in itemList]


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


def loadConfig(
    configSource: Union[str, Path, Mapping[str, Any]],
) -> Dict[str, Any]:
    r"""Load a YAML config from a path or accept an already-parsed mapping.

    If given a dict-like object, just return it. If given a path, try to load as YAML --> dict
    If given a path, try to load as YAML --> dict

    """
    if isinstance(configSource, Mapping):
        configData = configSource
    elif isinstance(configSource, (str, Path)):
        with open(configSource, "r") as fileHandle:
            configData = yaml.safe_load(fileHandle) or {}
    else:
        raise TypeError("`config` must be a path or a mapping/dict.")

    if not isinstance(configData, Mapping):
        raise TypeError("Top-level YAML must be a mapping/object.")
    return configData


def _cfgGet(
    configMap: Mapping[str, Any],
    dottedKey: str,
    defaultVal: Any = None,
) -> Any:
    r"""Support both dotted keys and yaml/dict-style nested access for configs."""

    # e.g., inputParams.bamFiles
    if dottedKey in configMap:
        return configMap[dottedKey]

    # e.g.,
    # inputParams:
    #   bamFiles: [...]
    currentVal: Any = configMap
    for keyPart in dottedKey.split("."):
        if isinstance(currentVal, Mapping) and keyPart in currentVal:
            currentVal = currentVal[keyPart]
        else:
            return defaultVal
    return currentVal


def _listOrEmpty(list_):
    if list_ is None:
        return []
    return list_


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


@lru_cache(maxsize=8)
def _readSparseRegionsByChrom(sparseBedFile: str) -> dict[str, np.ndarray]:
    sparseFrame = pd.read_csv(
        sparseBedFile,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64},
        engine="c",
    )
    sparseFrame = sparseFrame[sparseFrame["end"] > sparseFrame["start"]]
    sparseRegionsByChrom: dict[str, np.ndarray] = {}
    for chromName, chromFrame in sparseFrame.groupby("chrom", sort=False):
        chromRegions = chromFrame.loc[:, ["start", "end"]].to_numpy(
            dtype=np.int64,
            copy=True,
        )
        order = np.argsort(chromRegions[:, 0], kind="mergesort")
        sparseRegionsByChrom[str(chromName)] = chromRegions[order, :]
    return sparseRegionsByChrom


def _loadSparseIntervalIndices(
    sparseBedFile: str,
    chromosome: str,
    intervals: np.ndarray,
) -> np.ndarray:
    sparseRegions = _readSparseRegionsByChrom(str(sparseBedFile)).get(
        str(chromosome),
        np.empty((0, 2), dtype=np.int64),
    )
    if sparseRegions.size == 0:
        return np.empty(0, dtype=np.intp)

    intervalStarts = np.asarray(intervals, dtype=np.int64)
    if intervalStarts.size == 0:
        return np.empty(0, dtype=np.intp)
    if intervalStarts.size == 1:
        intervalSize = 1
    else:
        intervalSize = int(intervalStarts[1] - intervalStarts[0])
        if intervalSize <= 0:
            raise ValueError("intervals must be strictly increasing")
    intervalEnds = intervalStarts + int(intervalSize)

    sparseMask = np.zeros(intervalStarts.size, dtype=bool)
    for bedStart, bedEnd in sparseRegions:
        firstIdx = int(np.searchsorted(intervalEnds, int(bedStart), side="right"))
        lastIdx = int(np.searchsorted(intervalStarts, int(bedEnd), side="left"))
        if firstIdx < 0:
            firstIdx = 0
        if lastIdx > intervalStarts.size:
            lastIdx = intervalStarts.size
        if lastIdx > firstIdx:
            sparseMask[firstIdx:lastIdx] = True

    sparseIdx = np.flatnonzero(sparseMask)
    if sparseIdx.size == 0:
        return np.empty(0, dtype=np.intp)
    return sparseIdx.astype(np.intp, copy=False)


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
        return _threadMapMaybe(
            [
                (source.path, str(source.sourceKind).upper())
                for source in treatmentSources
            ],
            _getReadLengthForTask,
            "read lengths",
            allowThreads=allowThreads,
        )

    return _threadMapMaybe(
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


def getInputArgs(config_path: str) -> core.inputParams:
    configData = loadConfig(config_path)
    defaultBarcodeTag = _cfgGet(configData, "scParams.barcodeTag", "CB")
    defaultFragmentPositionMode = _cfgGet(
        configData,
        "scParams.defaultFragmentPositionMode",
        "insertionEndpoints",
    )
    core._normalizeFragmentPositionMode(defaultFragmentPositionMode)

    sampleConfigs = _cfgGet(configData, "inputParams.samples", None)
    treatmentSources: List[core.inputSource]
    controlSources: List[core.inputSource]
    if sampleConfigs is not None:
        if not isinstance(sampleConfigs, list) or len(sampleConfigs) == 0:
            raise ValueError("`inputParams.samples` must be a non-empty list.")
        allSources = [
            _coerceInputSource(
                sourceConfig,
                defaultRole="treatment",
                defaultBarcodeTag=defaultBarcodeTag,
                defaultFragmentPositionMode=defaultFragmentPositionMode,
            )
            for sourceConfig in sampleConfigs
        ]
        treatmentSources = [
            source for source in allSources if str(source.role).lower() == "treatment"
        ]
        controlSources = [
            source for source in allSources if str(source.role).lower() == "control"
        ]
    else:
        bamFilesRaw = _cfgGet(configData, "inputParams.bamFiles", []) or []
        bamFilesControlRaw = (
            _cfgGet(configData, "inputParams.bamFilesControl", []) or []
        )
        treatmentSources = _buildPathInputSources(bamFilesRaw, role="treatment")
        controlSources = _buildPathInputSources(
            bamFilesControlRaw,
            role="control",
        )

    if len(treatmentSources) == 0:
        raise ValueError("No input sources provided in the configuration.")

    if (
        len(controlSources) > 0
        and len(controlSources) != len(treatmentSources)
        and len(controlSources) != 1
    ):
        raise ValueError(
            "Number of control sources must be 0, 1, or the same as number of treatment sources"
        )

    treatmentSources = _prepareBedGraphSources(treatmentSources)
    controlSources = _prepareBedGraphSources(controlSources)

    if len(controlSources) == 1:
        logger.info(
            f"Only one control given: Using {controlSources[0].path} for all treatment files."
        )
        controlSources = controlSources * len(treatmentSources)

    bamFiles = core.getSourcePaths(treatmentSources)
    bamFilesControl = core.getSourcePaths(controlSources)

    if not bamFiles or not isinstance(bamFiles, list):
        raise ValueError("No input source paths found")

    for source in treatmentSources:
        if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS:
            misc_util.checkAlignmentFile(source.path)
        elif not os.path.exists(source.path):
            raise FileNotFoundError(f"Could not find {source.path}")

    if controlSources:
        for source in controlSources:
            if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS:
                misc_util.checkAlignmentFile(source.path)
            elif not os.path.exists(source.path):
                raise FileNotFoundError(f"Could not find {source.path}")

    return core.inputParams(
        bamFiles=bamFiles,
        bamFilesControl=bamFilesControl,
        treatmentSources=treatmentSources,
        controlSources=controlSources,
    )


def getOutputArgs(config_path: str) -> core.outputParams:
    configData = loadConfig(config_path)

    convertToBigWig_ = _cfgGet(
        configData,
        "outputParams.convertToBigWig",
        _pyBigWigAvailable(),
    )

    roundDigits_ = _cfgGet(configData, "outputParams.roundDigits", 3)
    writeUncertainty_ = _cfgGet(
        configData,
        "outputParams.writeUncertainty",
        True,
    )
    writeJackknifeSE_ = _cfgGet(
        configData,
        "outputParams.writeJackknifeSE",
        True,
    )
    applyJackknife_ = _cfgGet(
        configData,
        "outputParams.applyJackknife",
        False,
    )
    return core.outputParams(
        convertToBigWig=convertToBigWig_,
        roundDigits=roundDigits_,
        writeUncertainty=writeUncertainty_,
        writeJackknifeSE=writeJackknifeSE_,
        applyJackknife=applyJackknife_,
    )


def getGenomeArgs(config_path: str) -> core.genomeParams:
    configData = loadConfig(config_path)

    genomeName = _cfgGet(configData, "genomeParams.name", None)
    genomeLabel = constants.resolveGenomeName(genomeName)

    chromSizesFile: Optional[str] = None
    blacklistFile: Optional[str] = None
    sparseBedFile: Optional[str] = None
    chromosomesList: Optional[List[str]] = None

    excludeChromsList: List[str] = (
        _cfgGet(configData, "genomeParams.excludeChroms", []) or []
    )
    excludeForNormList: List[str] = (
        _cfgGet(configData, "genomeParams.excludeForNorm", []) or []
    )

    if genomeLabel:
        chromSizesFile = constants.getGenomeResourceFile(genomeLabel, "sizes")
        blacklistFile = constants.getGenomeResourceFile(genomeLabel, "blacklist")
        sparseBedFile = constants.getGenomeResourceFile(genomeLabel, "sparse")

    chromSizesOverride = _cfgGet(configData, "genomeParams.chromSizesFile", None)
    if chromSizesOverride:
        chromSizesFile = chromSizesOverride

    blacklistOverride = _cfgGet(configData, "genomeParams.blacklistFile", None)
    if blacklistOverride:
        blacklistFile = blacklistOverride

    sparseOverride = _cfgGet(configData, "genomeParams.sparseBedFile", None)
    if sparseOverride:
        sparseBedFile = sparseOverride

    if not chromSizesFile or not os.path.exists(chromSizesFile):
        raise FileNotFoundError(
            f"Chromosome sizes file {chromSizesFile} does not exist."
        )

    chromosomesConfig = _cfgGet(configData, "genomeParams.chromosomes", None)
    if chromosomesConfig is not None:
        chromosomesList = chromosomesConfig
    else:
        if chromSizesFile:
            chromosomesFrame = pd.read_csv(
                chromSizesFile,
                sep="\t",
                header=None,
                names=["chrom", "size"],
            )
            chromosomesList = list(chromosomesFrame["chrom"])
        else:
            raise ValueError(
                "No chromosomes provided in the configuration and no chromosome sizes file specified."
            )

    chromosomesList = [
        chromName.strip()
        for chromName in chromosomesList
        if chromName and chromName.strip()
    ]
    if excludeChromsList:
        chromosomesList = [
            chromName
            for chromName in chromosomesList
            if chromName not in excludeChromsList
        ]
    chromosomesList = list(dict.fromkeys(chromosomesList))
    if not chromosomesList:
        raise ValueError(
            "No valid chromosomes found after excluding specified chromosomes."
        )

    return core.genomeParams(
        genomeName=genomeLabel,
        chromSizesFile=chromSizesFile,
        blacklistFile=blacklistFile,
        sparseBedFile=sparseBedFile,
        chromosomes=chromosomesList,
        excludeChroms=excludeChromsList,
        excludeForNorm=excludeForNormList,
    )


def getStateArgs(config_path: str) -> core.stateParams:
    configData = loadConfig(config_path)

    stateInit_ = _cfgGet(configData, "stateParams.stateInit", 0.0)
    stateCovarInit_ = _cfgGet(
        configData,
        "stateParams.stateCovarInit",
        1000.0,
    )
    boundState_ = _cfgGet(
        configData,
        "stateParams.boundState",
        False,
    )
    stateLowerBound_ = _cfgGet(
        configData,
        "stateParams.stateLowerBound",
        0.0,
    )
    stateUpperBound_ = _cfgGet(
        configData,
        "stateParams.stateUpperBound",
        10000.0,
    )
    if boundState_:
        if stateLowerBound_ > stateUpperBound_:
            raise ValueError("`stateLowerBound` is greater than `stateUpperBound`.")
    return core.stateParams(
        stateInit=stateInit_,
        stateCovarInit=stateCovarInit_,
        boundState=boundState_,
        stateLowerBound=stateLowerBound_,
        stateUpperBound=stateUpperBound_,
    )


def getCountingArgs(config_path: str) -> core.countingParams:
    configData = loadConfig(config_path)

    intervalSizeBP = _cfgGet(configData, "countingParams.intervalSizeBP", 25)
    backgroundBlockSizeBP_ = _cfgGet(
        configData,
        "countingParams.backgroundBlockSizeBP",
        -1,
    )
    scaleFactorList = _cfgGet(configData, "countingParams.scaleFactors", None)
    scaleFactorsControlList = _cfgGet(
        configData, "countingParams.scaleFactorsControl", None
    )
    if scaleFactorList is not None and not isinstance(scaleFactorList, list):
        raise ValueError("`scaleFactors` should be a list of floats.")

    if scaleFactorsControlList is not None and not isinstance(
        scaleFactorsControlList, list
    ):
        raise ValueError("`scaleFactorsControl` should be a list of floats.")

    if (
        scaleFactorList is not None
        and scaleFactorsControlList is not None
        and len(scaleFactorList) != len(scaleFactorsControlList)
    ):
        if len(scaleFactorsControlList) == 1:
            scaleFactorsControlList = scaleFactorsControlList * len(scaleFactorList)
        else:
            raise ValueError(
                "control and treatment scale factors: must be equal length or 1 control"
            )

    normMethod_ = _cfgGet(
        configData,
        "countingParams.normMethod",
        "EGS",
    )
    if normMethod_.upper() not in ["EGS", "RPGC", "RPKM", "CPM", "SF"]:
        logger.warning(
            f"Unknown `countingParams.normMethod`...Using `EGS`...",
        )
        normMethod_ = "EGS"
    fragmentsGroupNorm_ = _cfgGet(
        configData,
        "countingParams.fragmentsGroupNorm",
        _cfgGet(configData, "scParams.fragmentsGroupNorm", "NONE"),
    )
    if str(fragmentsGroupNorm_).upper() not in ["NONE", "CELLS"]:
        raise ValueError(
            "`countingParams.fragmentsGroupNorm` must be `NONE` or `CELLS`."
        )

    fixControl_ = _cfgGet(
        configData,
        "countingParams.fixControl",
        False,
    )
    globalWeight_ = _cfgGet(
        configData,
        "countingParams.globalWeight",
        1000.0,
    )
    logOffset_ = _cfgGet(
        configData,
        "countingParams.logOffset",
        1.0,
    )
    logMult_ = _cfgGet(
        configData,
        "countingParams.logMult",
        1.0,
    )
    return core.countingParams(
        intervalSizeBP=intervalSizeBP,
        backgroundBlockSizeBP=backgroundBlockSizeBP_,
        scaleFactors=scaleFactorList,
        scaleFactorsControl=scaleFactorsControlList,
        normMethod=normMethod_,
        fragmentsGroupNorm=fragmentsGroupNorm_,
        fixControl=fixControl_,
        globalWeight=globalWeight_,
        logOffset=logOffset_,
        logMult=logMult_,
    )


def getScArgs(config_path: str) -> core.scParams:
    configData = loadConfig(config_path)

    barcodeTag_ = _cfgGet(configData, "scParams.barcodeTag", "CB")
    defaultCountMode_ = _cfgGet(
        configData,
        "scParams.defaultCountMode",
        "coverage",
    )
    if str(defaultCountMode_).strip().lower() not in [
        "coverage",
        "cutsite",
        "cutsites",
        "fiveprime",
        "5p",
        "center",
        "midpoint",
    ]:
        raise ValueError("`scParams.defaultCountMode` is not supported.")

    fragmentsGroupNorm_ = _cfgGet(
        configData,
        "scParams.fragmentsGroupNorm",
        _cfgGet(configData, "countingParams.fragmentsGroupNorm", "NONE"),
    )
    if str(fragmentsGroupNorm_).upper() not in ["NONE", "CELLS"]:
        raise ValueError("`scParams.fragmentsGroupNorm` must be `NONE` or `CELLS`.")

    defaultFragmentPositionMode_ = _cfgGet(
        configData,
        "scParams.defaultFragmentPositionMode",
        "insertionEndpoints",
    )
    core._normalizeFragmentPositionMode(defaultFragmentPositionMode_)
    return core.scParams(
        barcodeTag=barcodeTag_,
        defaultCountMode=defaultCountMode_,
        fragmentsGroupNorm=fragmentsGroupNorm_,
        defaultFragmentPositionMode=defaultFragmentPositionMode_,
    )


def getUncertaintyCalibrationArgs(
    config_path: str,
) -> core.uncertaintyCalibrationParams:
    configData = loadConfig(config_path)
    enabledDefault = True
    blockDefault = None
    padDefault = _cfgGet(configData, "observationParams.pad", 1.0e-4)
    maxScores = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.maxScores",
        _cfgGet(configData, "uncertaintyCalibration.maxScores", None),
    )
    maxHeldoutCells = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.maxHeldoutCells",
        _cfgGet(configData, "uncertaintyCalibration.maxHeldoutCells", None),
    )
    if maxScores is None and maxHeldoutCells is None:
        maxScores = core.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES
    return core.uncertaintyCalibrationParams(
        enabled=bool(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.enabled",
                _cfgGet(configData, "uncertaintyCalibration.enabled", enabledDefault),
            )
        ),
        folds=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.folds",
                _cfgGet(configData, "uncertaintyCalibration.folds", 5),
            )
        ),
        blockSizeBP=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.blockSizeBP",
            _cfgGet(configData, "uncertaintyCalibration.blockSizeBP", blockDefault),
        ),
        holdoutFraction=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.holdoutFraction",
            _cfgGet(configData, "uncertaintyCalibration.holdoutFraction", None),
        ),
        heldoutReplicateFraction=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.heldoutReplicateFraction",
            _cfgGet(
                configData,
                "uncertaintyCalibration.heldoutReplicateFraction",
                None,
            ),
        ),
        maxScores=int(maxScores) if maxScores is not None else maxScores,
        maxHeldoutCells=(
            int(maxHeldoutCells) if maxHeldoutCells is not None else maxHeldoutCells
        ),
        maxDiagnosticRows=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.maxDiagnosticRows",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.maxDiagnosticRows",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS,
                ),
            )
        ),
        minHeldoutCells=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.minHeldoutCells",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.minHeldoutCells",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS,
                ),
            )
        ),
        targets=tuple(
            float(x)
            for x in _cfgGet(
                configData,
                "uncertaintyCalibrationParams.targets",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.targets",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS,
                ),
            )
        ),
        minFactor=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.minFactor",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.minFactor",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
                ),
            )
        ),
        maxFactor=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.maxFactor",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.maxFactor",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
                ),
            )
        ),
        factorMin=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.factorMin",
            _cfgGet(configData, "uncertaintyCalibration.factorMin", None),
        ),
        factorMax=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.factorMax",
            _cfgGet(configData, "uncertaintyCalibration.factorMax", None),
        ),
        ridge=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.ridge",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.ridge",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE,
                ),
            )
        ),
        wisWeight=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.wisWeight",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.wisWeight",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT,
                ),
            )
        ),
        aObsPenalty=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.aObsPenalty",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.aObsPenalty",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY,
                ),
            )
        ),
        aObsPriorStrength=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.aObsPriorStrength",
            _cfgGet(configData, "uncertaintyCalibration.aObsPriorStrength", None),
        ),
        calibrationEMIters=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.calibrationEMIters",
                _cfgGet(configData, "uncertaintyCalibration.calibrationEMIters", 2),
            )
        ),
        seed=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.seed",
                _cfgGet(
                    configData,
                    "uncertaintyCalibration.seed",
                    core.UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
                ),
            )
        ),
        pad=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.pad",
            _cfgGet(configData, "uncertaintyCalibration.pad", padDefault),
        ),
        writeDiagnostics=bool(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.writeDiagnostics",
                _cfgGet(configData, "uncertaintyCalibration.writeDiagnostics", False),
            )
        ),
    )


def readConfig(config_path: str) -> Dict[str, Any]:
    r"""Read and parse the configuration file for Consenrich.

    :param config_path: Path to the YAML configuration file.
    :return: Dictionary containing all parsed configuration parameters.
    """
    configData = loadConfig(config_path)

    inputParams = getInputArgs(config_path)
    outputParams = getOutputArgs(config_path)
    genomeParams = getGenomeArgs(config_path)
    stateParams = getStateArgs(config_path)
    countingParams = getCountingArgs(config_path)
    scArgs = getScArgs(config_path)
    uncertaintyCalibrationArgs = getUncertaintyCalibrationArgs(config_path)
    experimentName = _cfgGet(configData, "experimentName", "consenrichExperiment")
    processQLevelTargetCfg = _cfgGet(
        configData,
        "processParams.processQLevelTarget",
        None,
    )
    processQTrendTargetCfg = _cfgGet(
        configData,
        "processParams.processQTrendTarget",
        None,
    )
    processDefaults = core.processParams()
    processArgs = core.processParams(
        deltaF=_cfgGet(configData, "processParams.deltaF", 1.0),
        minQ=_cfgGet(configData, "processParams.minQ", 1.0e-4),
        maxQ=_cfgGet(configData, "processParams.maxQ", 1000.0),
        offDiagQ=_cfgGet(
            configData,
            "processParams.offDiagQ",
            0.0,
        ),
        processQCalibration=_cfgGet(
            configData,
            "processParams.processQCalibration",
            core.PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL,
        ),
        processQCalibIters=int(
            _cfgGet(configData, "processParams.processQCalibIters", 5)
        ),
        processQLevelTarget=(
            None if processQLevelTargetCfg is None else float(processQLevelTargetCfg)
        ),
        processQTrendTarget=(
            None if processQTrendTargetCfg is None else float(processQTrendTargetCfg)
        ),
        processQLevelPriorWeight=float(
            _cfgGet(configData, "processParams.processQLevelPriorWeight", 0.05)
        ),
        processQTrendPriorWeight=float(
            _cfgGet(configData, "processParams.processQTrendPriorWeight", 25.0)
        ),
        precisionMultiplierMin=float(
            _cfgGet(
                configData,
                "processParams.precisionMultiplierMin",
                processDefaults.precisionMultiplierMin,
            )
        ),
        precisionMultiplierMax=float(
            _cfgGet(
                configData,
                "processParams.precisionMultiplierMax",
                processDefaults.precisionMultiplierMax,
            )
        ),
    )

    explicitSparseBedFile = _cfgGet(configData, "genomeParams.sparseBedFile", None)
    sparseBedAvailable = bool(
        genomeParams.sparseBedFile and os.path.exists(str(genomeParams.sparseBedFile))
    )
    numNearestRequested = int(
        _cfgGet(
            configData,
            "observationParams.numNearest",
            0,
        )
        or 0
    )
    if explicitSparseBedFile and numNearestRequested > 0:
        numNearestResolved = numNearestRequested
    else:
        numNearestResolved = 0
    restrictLocalAR1ToSparseBedRequested = bool(
        _cfgGet(
            configData,
            "observationParams.restrictLocalAR1ToSparseBed",
            False,
        )
    )
    if restrictLocalAR1ToSparseBedRequested and not sparseBedAvailable:
        logger.warning(
            "Requested `observationParams.restrictLocalAR1ToSparseBed`, but no "
            "readable sparse BED was resolved; disabling that option.",
        )
    restrictLocalAR1ToSparseBedResolved = bool(
        restrictLocalAR1ToSparseBedRequested and sparseBedAvailable
    )
    trendMaxEdfCfg = _cfgGet(configData, "observationParams.trendMaxEdf", 30.0)

    observationArgs = core.observationParams(
        minR=_cfgGet(configData, "observationParams.minR", -1.0),
        maxR=_cfgGet(configData, "observationParams.maxR", 1000.0),
        samplingIters=_cfgGet(
            configData,
            "observationParams.samplingIters",
            10_000,
        ),
        samplingBlockSizeBP=_cfgGet(
            configData,
            "observationParams.samplingBlockSizeBP",
            -1,
        ),
        EB_use=_cfgGet(
            configData,
            "observationParams.EB_use",
            True,
        ),
        EB_setNu0=_cfgGet(configData, "observationParams.EB_setNu0", None),
        EB_setNuL=_cfgGet(configData, "observationParams.EB_setNuL", None),
        trendNumBasis=int(_cfgGet(configData, "observationParams.trendNumBasis", 60)),
        trendMinObsPerBasis=float(
            _cfgGet(configData, "observationParams.trendMinObsPerBasis", 25.0)
        ),
        trendMinEdf=float(_cfgGet(configData, "observationParams.trendMinEdf", 3.0)),
        trendMaxEdf=None if trendMaxEdfCfg is None else float(trendMaxEdfCfg),
        trendLambdaMin=float(
            _cfgGet(configData, "observationParams.trendLambdaMin", 1.0e-6)
        ),
        trendLambdaMax=float(
            _cfgGet(configData, "observationParams.trendLambdaMax", 1.0e6)
        ),
        trendLambdaGridSize=int(
            _cfgGet(configData, "observationParams.trendLambdaGridSize", 41)
        ),
        numNearest=numNearestResolved,
        sparseSupportScaleBP=_cfgGet(
            configData,
            "observationParams.sparseSupportScaleBP",
            -1.0,
        ),
        sparseSupportPrior=float(
            _cfgGet(
                configData,
                "observationParams.sparseSupportPrior",
                1.0,
            )
        ),
        restrictLocalAR1ToSparseBed=restrictLocalAR1ToSparseBedResolved,
        blockQuantile=float(
            _cfgGet(configData, "observationParams.blockQuantile", 0.75)
        ),
        pad=_cfgGet(configData, "observationParams.pad", 1.0e-4),
        precisionMultiplierMin=float(
            _cfgGet(configData, "observationParams.precisionMultiplierMin", 0.1)
        ),
        precisionMultiplierMax=float(
            _cfgGet(configData, "observationParams.precisionMultiplierMax", 10.0)
        ),
    )

    EM_useAPN_ = bool(_cfgGet(configData, "fitParams.EM_useAPN", False))
    EM_use_ = bool(_cfgGet(configData, "fitParams.EM_use", True))

    fitArgs = core.fitParams(
        EM_maxIters=_cfgGet(configData, "fitParams.EM_maxIters", 50),
        EM_use=EM_use_,
        EM_innerRtol=_cfgGet(configData, "fitParams.EM_innerRtol", 1.0e-4),
        EM_tNu=_cfgGet(configData, "fitParams.EM_tNu", 8.0),
        EM_useObsPrecReweight=_cfgGet(
            configData,
            "fitParams.EM_useObsPrecReweight",
            True,
        ),
        EM_useProcPrecReweight=_cfgGet(
            configData,
            "fitParams.EM_useProcPrecReweight",
            True,
        )
        and (not EM_useAPN_),
        EM_useAPN=EM_useAPN_,
        EM_useReplicateBias=_cfgGet(
            configData,
            "fitParams.EM_useReplicateBias",
            True,
        ),
        fitBackground=_cfgGet(
            configData,
            "fitParams.fitBackground",
            True,
        ),
        EM_zeroCenterBackground=_cfgGet(
            configData,
            "fitParams.EM_zeroCenterBackground",
            True,
        ),
        EM_zeroCenterReplicateBias=_cfgGet(
            configData,
            "fitParams.EM_zeroCenterReplicateBias",
            True,
        ),
        EM_outerIters=_cfgGet(
            configData,
            "fitParams.EM_outerIters",
            8,
        ),
        EM_outerRtol=_cfgGet(
            configData,
            "fitParams.EM_outerRtol",
            0.01,
        ),
        EM_backgroundSmoothness=_cfgGet(
            configData,
            "fitParams.EM_backgroundSmoothness",
            1.0,
        ),
    )

    samThreads = _cfgGet(configData, "samParams.samThreads", 1)
    samFlagExclude = _cfgGet(
        configData,
        "samParams.samFlagExclude",
        3844,
    )
    minMappingQuality = _cfgGet(
        configData,
        "samParams.minMappingQuality",
        10,
    )
    oneReadPerBin = _cfgGet(configData, "samParams.oneReadPerBin", 0)
    chunkSize = _cfgGet(configData, "samParams.chunkSize", 500_000)
    bamInputMode = _cfgGet(configData, "samParams.bamInputMode", "auto")
    defaultCountMode = _cfgGet(configData, "samParams.defaultCountMode", "coverage")
    shiftForward5p = int(_cfgGet(configData, "samParams.shiftForward5p", 0))
    shiftReverse5p = int(_cfgGet(configData, "samParams.shiftReverse5p", 0))
    extendFrom5pBP = _cfgGet(configData, "samParams.extendFrom5pBP", None)
    maxInsertSize = _cfgGet(
        configData,
        "samParams.maxInsertSize",
        1000,
    )
    inferFragmentLength = _cfgGet(
        configData,
        "samParams.inferFragmentLength",
        None,
    )
    core._normalizeBamInputMode(bamInputMode)
    core._normalizeCountMode(defaultCountMode, "coverage")
    if extendFrom5pBP is not None and not isinstance(extendFrom5pBP, (int, list)):
        raise ValueError("`samParams.extendFrom5pBP` must be an integer or list.")
    if isinstance(extendFrom5pBP, list):
        extendFrom5pBP = [int(value) for value in extendFrom5pBP]

    samArgs = core.samParams(
        samThreads=samThreads,
        samFlagExclude=samFlagExclude,
        oneReadPerBin=oneReadPerBin,
        chunkSize=chunkSize,
        bamInputMode=bamInputMode,
        defaultCountMode=defaultCountMode,
        shiftForward5p=shiftForward5p,
        shiftReverse5p=shiftReverse5p,
        extendFrom5pBP=extendFrom5pBP,
        maxInsertSize=maxInsertSize,
        inferFragmentLength=inferFragmentLength,
        minMappingQuality=minMappingQuality,
        minTemplateLength=_cfgGet(
            configData,
            "samParams.minTemplateLength",
            -1,
        ),
    )

    matchingArgs = core.matchingParams(
        enabled=bool(_cfgGet(configData, "matchingParams.enabled", True)),
        randSeed=_cfgGet(configData, "matchingParams.randSeed", 42),
        tau0=float(_cfgGet(configData, "matchingParams.tau0", 1.0)),
        numBootstrap=int(_cfgGet(configData, "matchingParams.numBootstrap", 128)),
        thresholdZ=float(_cfgGet(configData, "matchingParams.thresholdZ", 2.0)),
        dependenceSpan=_cfgGet(configData, "matchingParams.dependenceSpan", None),
        gamma=_cfgGet(configData, "matchingParams.gamma", 0.5),
        selectionPenalty=_cfgGet(configData, "matchingParams.selectionPenalty", None),
        gammaScale=float(_cfgGet(configData, "matchingParams.gammaScale", 0.5)),
        nestedRoccoIters=int(_cfgGet(configData, "matchingParams.nestedRoccoIters", 3)),
        nestedRoccoBudgetScale=float(
            _cfgGet(configData, "matchingParams.nestedRoccoBudgetScale", 0.5)
        ),
        exportFilterUncertaintyMultiplier=float(
            _cfgGet(
                configData,
                "matchingParams.exportFilterUncertaintyMultiplier",
                2.0,
            )
        ),
    )

    return {
        "experimentName": experimentName,
        "genomeArgs": genomeParams,
        "inputArgs": inputParams,
        "outputArgs": outputParams,
        "countingArgs": countingParams,
        "scArgs": scArgs,
        "processArgs": processArgs,
        "observationArgs": observationArgs,
        "stateArgs": stateParams,
        "uncertaintyCalibrationArgs": uncertaintyCalibrationArgs,
        "samArgs": samArgs,
        "matchingArgs": matchingArgs,
        "fitArgs": fitArgs,
    }


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
        try:
            _sortBedGraphInPlace(bedgraph)
        except Exception as e:
            logger.warning(f"Failed to sort {bedgraph} before bigWig conversion:\n{e}")
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
        bw.addHeader(sorted(chromSizes, key=lambda item: item[0]))
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
                if seenEntry:
                    if chrom < lastChrom or (chrom == lastChrom and start < lastStart):
                        raise ValueError(
                            f"bedGraph input is not sorted at row {lineNumber}; sort "
                            "with LC_ALL=C sort -k1,1 -k2,2n -k3,3n"
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


def _sortBedGraphInPlace(bedgraphPath: str) -> None:
    if not os.path.exists(bedgraphPath) or os.path.getsize(bedgraphPath) == 0:
        return

    sortPath = shutil.which("sort")
    if sortPath is not None:
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

    df = pd.read_csv(
        bedgraphPath,
        sep="\t",
        header=None,
        names=["chromosome", "start", "end", "value"],
        dtype={
            "chromosome": str,
            "start": np.int64,
            "end": np.int64,
            "value": np.float64,
        },
    )
    df.sort_values(
        by=["chromosome", "start", "end"],
        kind="mergesort",
        inplace=True,
    )
    df.to_csv(
        bedgraphPath,
        sep="\t",
        header=False,
        index=False,
        float_format="%.4f",
        lineterminator="\n",
    )


def main():
    parser = argparse.ArgumentParser(description="Consenrich CLI")
    parser.add_argument(
        "--config",
        type=str,
        dest="config",
        help="Path to a YAML config file with parameters + arguments defined in `consenrich.core`",
    )

    # --- Post hoc ROCCO peak-calling arguments ---
    parser.add_argument(
        "--match-bedGraph",
        type=str,
        dest="matchBedGraph",
        help="Path to a Consenrich state bedGraph file",
    )
    parser.add_argument(
        "--match-uncertainty-bedGraph",
        type=str,
        default=None,
        dest="matchUncertaintyBedGraph",
        help="Optional uncertainty bedGraph paired with `--match-bedGraph`. If omitted, Consenrich looks for a sibling `_uncertainty` bedGraph.",
    )
    parser.add_argument(
        "--match-tau0",
        type=float,
        default=1.0,
        dest="matchTau0",
        help="Shrinkage-score pseudovariance parameter; direct ROCCO scoring uses the fitted state values.",
    )
    parser.add_argument(
        "--match-num-bootstrap",
        type=int,
        default=128,
        dest="matchNumBootstrap",
        help="Number of dependent wild-bootstrap null draws used for budget calibration.",
    )
    parser.add_argument(
        "--match-threshold-z",
        type=float,
        default=2.0,
        dest="matchThresholdZ",
        help="One-sided Gaussian z-threshold used when calibrating null tail occupancy.",
    )
    parser.add_argument(
        "--match-nested-rocco-iters",
        type=int,
        default=3,
        dest="matchNestedRoccoIters",
        help="Number of monotone nested ROCCO refinement iterations within first-pass peaks. Set to 0 to disable.",
    )
    parser.add_argument(
        "--match-nested-rocco-budget-scale",
        type=float,
        default=0.5,
        dest="matchNestedRoccoBudgetScale",
        help="Optional fraction of each eligible first-pass peak region available to nested ROCCO refinement.",
    )
    parser.add_argument(
        "--match-export-filter-c",
        type=float,
        default=2.5,
        dest="matchExportFilterUncertaintyMultiplier",
        help=(
            "Multiplier c in the final ROCCO export filter "
            "`medianState < -c * median(local uncertainty)`. Default: 2.5. "
            "Setting c=0 requires exported peaks to have positive median signal."
        ),
    )
    parser.add_argument("--match-seed", type=int, default=42, dest="matchRandSeed")
    parser.add_argument("--verbose", action="store_true", help="If set, logs config")
    parser.add_argument(
        "--verbose2",
        action="store_true",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Consenrich v{__version__}",
    )
    args = parser.parse_args()

    if args.matchBedGraph:
        if not os.path.exists(args.matchBedGraph):
            raise FileNotFoundError(
                f"bedGraph file {args.matchBedGraph} couldn't be found."
            )
        uncertaintyBedGraph = args.matchUncertaintyBedGraph
        if uncertaintyBedGraph is not None and not os.path.exists(uncertaintyBedGraph):
            raise FileNotFoundError(
                f"uncertainty bedGraph file {uncertaintyBedGraph} couldn't be found."
            )
        if uncertaintyBedGraph is None:
            uncertaintyBedGraph = _inferMatchingUncertaintyBedGraph(args.matchBedGraph)
        logger.info(
            "Running post hoc ROCCO peak caller using state bedGraph %s...",
            args.matchBedGraph,
        )
        outName = peaks.solveRocco(
            args.matchBedGraph,
            uncertaintyBedGraphFile=uncertaintyBedGraph,
            tau0=args.matchTau0,
            numBootstrap=args.matchNumBootstrap,
            thresholdZ=args.matchThresholdZ,
            nestedRoccoIters=args.matchNestedRoccoIters,
            nestedRoccoBudgetScale=args.matchNestedRoccoBudgetScale,
            exportFilterUncertaintyMultiplier=(
                args.matchExportFilterUncertaintyMultiplier
            ),
            randSeed=args.matchRandSeed,
            verbose=bool(args.verbose or args.verbose2),
        )
        logger.info("Finished post hoc ROCCO peak calling. Written to %s", outName)
        sys.exit(0)

    if not args.config:
        logger.info(
            "No config file provided, run with `--config <path_to_config.yaml>`"
        )
        logger.info("See documentation: https://nolan-h-hamilton.github.io/Consenrich/")
        sys.exit(1)

    if not os.path.exists(args.config):
        logger.info(f"Config file {args.config} does not exist.")
        logger.info("See documentation: https://nolan-h-hamilton.github.io/Consenrich/")
        sys.exit(1)

    config = readConfig(args.config)
    experimentName = config["experimentName"]
    genomeArgs = config["genomeArgs"]
    inputArgs = config["inputArgs"]
    outputArgs = config["outputArgs"]
    countingArgs = config["countingArgs"]
    scArgs = config["scArgs"]
    processArgs = config["processArgs"]
    observationArgs = config["observationArgs"]
    stateArgs = config["stateArgs"]
    uncertaintyCalibrationArgs = config["uncertaintyCalibrationArgs"]
    samArgs = config["samArgs"]
    matchingArgs = config["matchingArgs"]
    fitArgs = config["fitArgs"]
    treatmentSources = _listOrEmpty(getattr(inputArgs, "treatmentSources", None))
    controlSources = _listOrEmpty(getattr(inputArgs, "controlSources", None))
    if not treatmentSources:
        treatmentSources = _buildPathInputSources(inputArgs.bamFiles, role="treatment")
    if not controlSources:
        controlSources = _buildPathInputSources(
            _listOrEmpty(inputArgs.bamFilesControl),
            role="control",
        )
    bamFiles = core.getSourcePaths(treatmentSources)
    bamFilesControl = core.getSourcePaths(controlSources)
    numSamples = len(bamFiles)
    intervalSizeBP = countingArgs.intervalSizeBP
    excludeForNorm = genomeArgs.excludeForNorm
    chromSizes = genomeArgs.chromSizesFile
    deltaF_ = processArgs.deltaF
    minR_ = observationArgs.minR
    maxR_ = observationArgs.maxR
    minQ_ = processArgs.minQ
    maxQ_ = processArgs.maxQ
    offDiagQ_ = processArgs.offDiagQ
    samplingBlockSizeBP_ = observationArgs.samplingBlockSizeBP
    backgroundBlockSizeBP_ = countingArgs.backgroundBlockSizeBP
    backgroundBlockSizeIntervals = (
        -1
        if backgroundBlockSizeBP_ <= 0
        else int(backgroundBlockSizeBP_ / intervalSizeBP)
    )
    if samplingBlockSizeBP_ is None or samplingBlockSizeBP_ <= 0:
        samplingBlockSizeBP_ = countingArgs.backgroundBlockSizeBP
    vec_: Optional[Tuple[int, int, int]] = None
    waitForMatrix: bool = False
    normMethod_: Optional[str] = countingArgs.normMethod.upper()
    pad_ = observationArgs.pad if hasattr(observationArgs, "pad") else 1.0e-4
    if args.verbose2:
        args.verbose = True

    if args.verbose:
        try:
            logger.info(f"Consenrich [v{__version__}]: Initial Configuration\n")
            config_truncated = {
                k: v
                for k, v in config.items()
                if k not in ["inputArgs", "genomeArgs", "countingArgs"]
            }
            config_truncated["experimentName"] = experimentName
            config_truncated["inputArgs"] = inputArgs
            config_truncated["outputArgs"] = outputArgs
            config_truncated["genomeArgs"] = genomeArgs
            config_truncated["countingArgs"] = countingArgs
            config_truncated["scArgs"] = scArgs
            config_truncated["processArgs"] = processArgs
            config_truncated["observationArgs"] = observationArgs
            config_truncated["stateArgs"] = stateArgs
            config_truncated["uncertaintyCalibrationArgs"] = uncertaintyCalibrationArgs
            config_truncated["samArgs"] = samArgs
            pretty = pprint.pformat(
                config_truncated,
                indent=2,
                width=72,
                sort_dicts=True,
                compact=False,
            )
            logger.info(f"\n{pretty}\n")
        except Exception as e:
            logger.warning(f"Failed to print parsed config:\n{e}\n")

    if normMethod_ in ["SF"] and (len(bamFilesControl) > 0 or numSamples < 3):
        logger.warning(
            "`countingParams.normMethod` `SF` is not available when control inputs are present OR if < 3 treatment samples are given."
            "  --> using CPM/RPKM ..."
        )
        normMethod_ = "RPKM"

    controlsPresent = checkControlsPresent(inputArgs)
    if args.verbose:
        logger.info(f"controlsPresent: {controlsPresent}")
    anyFragments = any(
        str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND
        for source in treatmentSources + controlSources
    )
    anyBedGraph = any(
        str(source.sourceKind).upper() == core.BEDGRAPH_SOURCE_KIND
        for source in treatmentSources + controlSources
    )
    allBedGraph = all(
        str(source.sourceKind).upper() == core.BEDGRAPH_SOURCE_KIND
        for source in treatmentSources + controlSources
    )
    if anyFragments and normMethod_ in ["EGS", "RPGC"]:
        logger.warning(
            "Fragments inputs use insertion-based depth normalization not EGS/RPGC"
            "  --> using CPM/RPKM ..."
        )
        normMethod_ = "CPM"
    if (
        anyBedGraph
        and not allBedGraph
        and (
            countingArgs.scaleFactors is None
            or (controlsPresent and countingArgs.scaleFactorsControl is None)
        )
    ):
        raise ValueError(
            "Mixed BEDGRAPH and read-count sources require explicit "
            "`countingParams.scaleFactors`"
            + (" and `countingParams.scaleFactorsControl`." if controlsPresent else ".")
        )
    if allBedGraph and normMethod_ in ["EGS", "RPGC", "RPKM", "CPM"]:
        logger.info(
            "BEDGRAPH inputs are treated as already scaled tracks; using unit "
            "scale factors unless explicit scale factors are provided."
        )
    for source in treatmentSources + controlSources:
        if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND:
            core._normalizeFragmentPositionMode(
                source.fragmentPositionMode or scArgs.defaultFragmentPositionMode
            )
    readLengthsBamFiles = getReadLengths(inputArgs, countingArgs, samArgs)
    effectiveGenomeSizes = getEffectiveGenomeSizes(genomeArgs, readLengthsBamFiles)
    treatmentBamInputModes = [
        (
            core._resolveSourceBamInputMode(
                source,
                str(samArgs.bamInputMode or "auto"),
            )
            if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS
            else "reads"
        )
        for source in treatmentSources
    ]
    controlBamInputModes = [
        (
            core._resolveSourceBamInputMode(
                source,
                str(samArgs.bamInputMode or "auto"),
            )
            if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS
            else "reads"
        )
        for source in controlSources
    ]
    autoInferFragmentLength = (
        samArgs.inferFragmentLength is None
        and core._normalizeBamInputMode(samArgs.bamInputMode) == "auto"
    )
    inferFragmentLengthRequested = int(samArgs.inferFragmentLength or 0) > 0
    if autoInferFragmentLength and any(
        sourceBamInputMode in ("reads", "read1")
        for sourceBamInputMode in treatmentBamInputModes + controlBamInputModes
    ):
        logger.info(
            "samParams.bamInputMode=auto and samParams.inferFragmentLength omitted: "
            "single-end BAM sources will be extended by inferred fragment length."
        )
    treatmentCountModes = [
        _getSourceCountMode(
            source,
            str(samArgs.defaultCountMode or "coverage"),
            str(scArgs.defaultCountMode or "coverage"),
        )
        for source in treatmentSources
    ]
    controlCountModes = [
        _getSourceCountMode(
            source,
            str(samArgs.defaultCountMode or "coverage"),
            str(scArgs.defaultCountMode or "coverage"),
        )
        for source in controlSources
    ]
    treatmentAllowLists, treatmentSelectedCellCounts, treatmentNormTempPaths = (
        _prepareFragmentsNormalizationMetadata(treatmentSources)
    )
    controlAllowLists, controlSelectedCellCounts, controlNormTempPaths = (
        _prepareFragmentsNormalizationMetadata(controlSources)
    )

    peakCallingEnabled = checkMatchingEnabled(matchingArgs)
    if args.verbose:
        logger.info(f"peakCallingEnabled: {peakCallingEnabled}")
    scaleFactors = countingArgs.scaleFactors
    scaleFactorsControl = countingArgs.scaleFactorsControl
    characteristicFragmentLengthsTreatment: List[int] = []
    characteristicFragmentLengthsControl: List[int] = []
    countExtendFrom5pBPTreatment: List[int] = []
    countExtendFrom5pBPControl: List[int] = []
    setupAllowThreads = int(samArgs.samThreads) <= 1
    sf: np.ndarray = np.empty((numSamples,), dtype=float)
    configuredExtendFrom5pBPTreatment = core._resolveExtendFrom5pBP(
        samArgs.extendFrom5pBP,
        treatmentSources,
    )
    configuredExtendFrom5pBPControl = (
        list(configuredExtendFrom5pBPTreatment[: len(controlSources)])
        if controlsPresent
        else []
    )

    def _estimateFragmentLengthForSource(
        source: core.inputSource,
        sourceFlagExclude: int,
    ) -> int:
        if str(source.sourceKind).upper() not in core.ALIGNMENT_SOURCE_KINDS:
            return 0
        fragmentLength = int(
            cconsenrich.cgetFragmentLength(
                source.path,
                samThreads=samArgs.samThreads,
                samFlagExclude=sourceFlagExclude,
                maxInsertSize=samArgs.maxInsertSize,
            )
        )
        logger.info(
            "Estimated fragment length for %s: %d",
            source.path,
            fragmentLength,
        )
        return fragmentLength

    def _resolveCharacteristicFragmentLength(task) -> int:
        source, sourceBamInputMode, configuredExtendBP = task
        if str(source.sourceKind).upper() not in core.ALIGNMENT_SOURCE_KINDS:
            return 0
        if sourceBamInputMode == "fragments" and (
            int(configuredExtendBP) > 0 or inferFragmentLengthRequested
        ):
            raise ValueError(
                "`samParams.extendFrom5pBP` and `samParams.inferFragmentLength` "
                "require `bamInputMode` `reads` or `read1`."
            )
        if int(configuredExtendBP) > 0:
            return int(configuredExtendBP)
        return _estimateFragmentLengthForSource(
            source,
            core._resolveSourceFlagExclude(
                samArgs.samFlagExclude,
                sourceBamInputMode,
            ),
        )

    characteristicFragmentLengthsTreatment = _threadMapMaybe(
        zip(
            treatmentSources,
            treatmentBamInputModes,
            configuredExtendFrom5pBPTreatment,
        ),
        _resolveCharacteristicFragmentLength,
        "characteristic lengths",
        allowThreads=setupAllowThreads,
    )
    if controlsPresent:
        logger.info(
            "Using treatment-derived extension lengths for control BAM sources."
        )
        (
            characteristicFragmentLengthsTreatment,
            characteristicFragmentLengthsControl,
        ) = _resolveExtendFrom5pBPPairs(
            characteristicFragmentLengthsTreatment,
            characteristicFragmentLengthsControl,
        )

    def _resolveCountExtendFrom5pBP(
        source: core.inputSource,
        sourceBamInputMode: str,
        configuredExtendBP: int,
        characteristicFragmentLength: int,
    ) -> int:
        if str(source.sourceKind).upper() not in core.ALIGNMENT_SOURCE_KINDS:
            return 0
        if sourceBamInputMode == "fragments":
            return 0
        if int(configuredExtendBP) > 0:
            return int(configuredExtendBP)
        if inferFragmentLengthRequested or (
            autoInferFragmentLength and sourceBamInputMode in ("reads", "read1")
        ):
            return int(characteristicFragmentLength)
        return 0

    countExtendFrom5pBPTreatment = [
        _resolveCountExtendFrom5pBP(
            source,
            sourceBamInputMode,
            configuredExtendBP,
            characteristicFragmentLength,
        )
        for source, sourceBamInputMode, configuredExtendBP, characteristicFragmentLength in zip(
            treatmentSources,
            treatmentBamInputModes,
            configuredExtendFrom5pBPTreatment,
            characteristicFragmentLengthsTreatment,
        )
    ]
    if controlsPresent:
        countExtendFrom5pBPControl = [
            _resolveCountExtendFrom5pBP(
                source,
                sourceBamInputMode,
                configuredExtendBP,
                characteristicFragmentLength,
            )
            for source, sourceBamInputMode, configuredExtendBP, characteristicFragmentLength in zip(
                controlSources,
                controlBamInputModes,
                configuredExtendFrom5pBPControl,
                characteristicFragmentLengthsControl,
            )
        ]

    try:
        if controlsPresent:

            def _getReadLengthForSource(source: core.inputSource) -> int:
                return core.getReadLength(
                    source.path,
                    100,
                    1000,
                    samArgs.samThreads,
                    samArgs.samFlagExclude,
                    sourceKind=str(source.sourceKind).upper(),
                )

            readLengthsControlBamFiles = _threadMapMaybe(
                controlSources,
                _getReadLengthForSource,
                "control read lengths",
                allowThreads=setupAllowThreads,
            )
            effectiveGenomeSizesControl = getEffectiveGenomeSizes(
                genomeArgs,
                readLengthsControlBamFiles,
            )

            if scaleFactors is not None and scaleFactorsControl is not None:
                treatScaleFactors = scaleFactors
                controlScaleFactors = scaleFactorsControl
            elif allBedGraph:
                treatScaleFactors = scaleFactors or [1.0] * len(treatmentSources)
                controlScaleFactors = scaleFactorsControl or [1.0] * len(controlSources)
            else:

                def _getPairScaleFactors(task):
                    (
                        sourceA,
                        sourceB,
                        effectiveGenomeSizeA,
                        effectiveGenomeSizeB,
                        readLengthA,
                        readLengthB,
                        barcodeAllowListPathA,
                        barcodeAllowListPathB,
                        countModeA,
                        countModeB,
                        groupCellCountA,
                        groupCellCountB,
                    ) = task
                    return detrorm.getPairScaleFactors(
                        sourceA.path,
                        sourceB.path,
                        effectiveGenomeSizeA,
                        effectiveGenomeSizeB,
                        readLengthA,
                        readLengthB,
                        excludeForNorm,
                        chromSizes,
                        samArgs.samThreads,
                        intervalSizeBP,
                        normMethod=normMethod_,
                        fixControl=countingArgs.fixControl,
                        sourceKindA=str(sourceA.sourceKind).upper(),
                        sourceKindB=str(sourceB.sourceKind).upper(),
                        barcodeAllowListFileA=barcodeAllowListPathA,
                        barcodeAllowListFileB=barcodeAllowListPathB,
                        countModeA=countModeA,
                        countModeB=countModeB,
                        oneReadPerBin=samArgs.oneReadPerBin,
                        groupCellCountA=groupCellCountA,
                        groupCellCountB=groupCellCountB,
                        fragmentsGroupNorm=scArgs.fragmentsGroupNorm,
                    )

                pairScalingFactors = _threadMapMaybe(
                    zip(
                        treatmentSources,
                        controlSources,
                        effectiveGenomeSizes,
                        effectiveGenomeSizesControl,
                        characteristicFragmentLengthsTreatment,
                        characteristicFragmentLengthsControl,
                        treatmentAllowLists,
                        controlAllowLists,
                        treatmentCountModes,
                        controlCountModes,
                        treatmentSelectedCellCounts,
                        controlSelectedCellCounts,
                    ),
                    _getPairScaleFactors,
                    "pair scale factors",
                    allowThreads=setupAllowThreads,
                )
                treatScaleFactors = []
                controlScaleFactors = []
                for scaleFactorA, scaleFactorB in pairScalingFactors:
                    treatScaleFactors.append(scaleFactorA)
                    controlScaleFactors.append(scaleFactorB)

        else:
            treatScaleFactors = scaleFactors
            controlScaleFactors = scaleFactorsControl

        if scaleFactors is None and not controlsPresent:
            if allBedGraph and normMethod_ != "SF":
                scaleFactors = [1.0] * len(treatmentSources)
            elif normMethod_ in ["RPKM", "CPM"]:

                def _getScaleFactorPerMillion(task) -> float:
                    source, barcodeAllowListPath, countMode, groupCellCount = task
                    return detrorm.getScaleFactorPerMillion(
                        source.path,
                        excludeForNorm,
                        intervalSizeBP,
                        sourceKind=str(source.sourceKind).upper(),
                        barcodeAllowListFile=barcodeAllowListPath,
                        countMode=countMode,
                        oneReadPerBin=samArgs.oneReadPerBin,
                        groupCellCount=groupCellCount,
                        fragmentsGroupNorm=scArgs.fragmentsGroupNorm,
                    )

                scaleFactors = _threadMapMaybe(
                    zip(
                        treatmentSources,
                        treatmentAllowLists,
                        treatmentCountModes,
                        treatmentSelectedCellCounts,
                    ),
                    _getScaleFactorPerMillion,
                    "scale factors",
                    allowThreads=setupAllowThreads,
                )
            elif normMethod_ in ["EGS", "RPGC"]:

                def _getScaleFactor1x(task) -> float:
                    (
                        source,
                        effectiveGenomeSize,
                        readLength,
                        barcodeAllowListPath,
                        countMode,
                        groupCellCount,
                    ) = task
                    return detrorm.getScaleFactor1x(
                        source.path,
                        effectiveGenomeSize,
                        readLength,
                        excludeForNorm,
                        genomeArgs.chromSizesFile,
                        samArgs.samThreads,
                        sourceKind=str(source.sourceKind).upper(),
                        barcodeAllowListFile=barcodeAllowListPath,
                        countMode=countMode,
                        oneReadPerBin=samArgs.oneReadPerBin,
                        groupCellCount=groupCellCount,
                        fragmentsGroupNorm=scArgs.fragmentsGroupNorm,
                    )

                scaleFactors = _threadMapMaybe(
                    zip(
                        treatmentSources,
                        effectiveGenomeSizes,
                        characteristicFragmentLengthsTreatment,
                        treatmentAllowLists,
                        treatmentCountModes,
                        treatmentSelectedCellCounts,
                    ),
                    _getScaleFactor1x,
                    "scale factors",
                    allowThreads=setupAllowThreads,
                )
            elif normMethod_ in ["SF"]:
                waitForMatrix = True
    finally:
        for tempPath in treatmentNormTempPaths + controlNormTempPaths:
            try:
                os.remove(tempPath)
            except OSError:
                pass

    deltaF_ = core._resolveFixedDeltaF(deltaF_)
    logger.info("Using fixed deltaF=%.6f", deltaF_)

    chromSizesDict = misc_util.getChromSizesDict(
        genomeArgs.chromSizesFile,
        excludeChroms=genomeArgs.excludeChroms,
    )
    chromosomes = genomeArgs.chromosomes
    treatmentSourceKinds = [
        str(source.sourceKind).upper() for source in treatmentSources
    ]
    chromosomePlans: List[Dict[str, Any]] = []
    for chromosome in _progress(
        chromosomes,
        total=len(chromosomes),
        desc="Planning chromosomes",
        unit="chrom",
    ):
        chromosomeStart, chromosomeEnd = core.getChromRangesJoint(
            bamFiles,
            chromosome,
            chromSizesDict[chromosome],
            samArgs.samThreads,
            samArgs.samFlagExclude,
            sourceKinds=treatmentSourceKinds,
        )
        chromosomeStart = max(0, (chromosomeStart - (chromosomeStart % intervalSizeBP)))
        chromosomeEnd = max(0, (chromosomeEnd - (chromosomeEnd % intervalSizeBP)))
        numIntervals = (
            ((chromosomeEnd - chromosomeStart) + intervalSizeBP) - 1
        ) // intervalSizeBP
        chromosomePlans.append(
            {
                "chromosome": str(chromosome),
                "start": int(chromosomeStart),
                "end": int(chromosomeEnd),
                "numIntervals": int(numIntervals),
            }
        )

    if chromosomePlans:
        for file_ in os.listdir("."):
            if file_.startswith(f"consenrichOutput_{experimentName}") and (
                file_.endswith(".bedGraph") or file_.endswith(".narrowPeak")
            ):
                logger.warning(f"Overwriting: {file_}")
                os.remove(file_)

    stateDiagnosticsByChromosome: Dict[str, Any] = {}

    for c_, chromPlan in enumerate(
        _progress(
            chromosomePlans,
            total=len(chromosomePlans),
            desc="Processing chromosomes",
            unit="chrom",
        )
    ):
        chromosomeStartTime = time.perf_counter()
        chromosome = str(chromPlan["chromosome"])
        chromosomeStart = int(chromPlan["start"])
        chromosomeEnd = int(chromPlan["end"])
        numIntervals = int(chromPlan["numIntervals"])
        logger.info(
            "chromosome.start %s intervals=%d samples=%d",
            chromosome,
            int(numIntervals),
            int(numSamples),
        )
        intervals = np.arange(chromosomeStart, chromosomeEnd, intervalSizeBP)
        chromMat: np.ndarray = np.empty((numSamples, numIntervals), dtype=np.float32)
        muncMat: np.ndarray = np.empty_like(chromMat, dtype=np.float32)
        if controlsPresent:
            for j_, (bamA, bamB) in enumerate(
                _progress(
                    zip(bamFiles, bamFilesControl),
                    total=numSamples,
                    desc=f"Counting {chromosome}",
                    unit="sample",
                )
            ):
                countStart = time.perf_counter()
                logger.info(
                    "counting.start %s sample=%d/%d treatment=%s control=%s",
                    chromosome,
                    int(j_ + 1),
                    int(numSamples),
                    bamA,
                    bamB,
                )

                pairMatrix: np.ndarray = core.readSegments(
                    [
                        treatmentSources[j_],
                        controlSources[j_],
                    ],
                    chromosome,
                    chromosomeStart,
                    chromosomeEnd,
                    intervalSizeBP,
                    [
                        readLengthsBamFiles[j_],
                        readLengthsControlBamFiles[j_],
                    ],
                    [treatScaleFactors[j_], controlScaleFactors[j_]],
                    samArgs.oneReadPerBin,
                    samArgs.samThreads,
                    samArgs.samFlagExclude,
                    bamInputMode=samArgs.bamInputMode,
                    defaultCountMode=samArgs.defaultCountMode,
                    shiftForward5p=samArgs.shiftForward5p,
                    shiftReverse5p=samArgs.shiftReverse5p,
                    extendFrom5pBP=[
                        countExtendFrom5pBPTreatment[j_],
                        countExtendFrom5pBPControl[j_],
                    ],
                    maxInsertSize=samArgs.maxInsertSize,
                    inferFragmentLength=samArgs.inferFragmentLength,
                    minMappingQuality=samArgs.minMappingQuality,
                    minTemplateLength=samArgs.minTemplateLength,
                )
                cconsenrich.cTransformWithInputInto(
                    pairMatrix[0, :],
                    pairMatrix[1, :],
                    chromMat[j_, :],
                    logOffset=countingArgs.logOffset,
                    logMult=countingArgs.logMult,
                )
                logger.info(
                    "counting.done %s sample=%d/%d elapsed=%.3fs",
                    chromosome,
                    int(j_ + 1),
                    int(numSamples),
                    time.perf_counter() - countStart,
                )
        else:
            countStart = time.perf_counter()
            logger.info(
                "counting.start %s samples=%d intervals=%d samThreads=%d",
                chromosome,
                int(numSamples),
                int(numIntervals),
                int(samArgs.samThreads),
            )
            chromMat = core.readSegments(
                treatmentSources,
                chromosome,
                chromosomeStart,
                chromosomeEnd,
                intervalSizeBP,
                readLengthsBamFiles,
                (
                    np.ones(numSamples) if waitForMatrix else scaleFactors
                ),  # for SF, wait until matrix is built
                samArgs.oneReadPerBin,
                samArgs.samThreads,
                samArgs.samFlagExclude,
                bamInputMode=samArgs.bamInputMode,
                defaultCountMode=samArgs.defaultCountMode,
                shiftForward5p=samArgs.shiftForward5p,
                shiftReverse5p=samArgs.shiftReverse5p,
                extendFrom5pBP=countExtendFrom5pBPTreatment,
                maxInsertSize=samArgs.maxInsertSize,
                inferFragmentLength=samArgs.inferFragmentLength,
                minMappingQuality=samArgs.minMappingQuality,
                minTemplateLength=samArgs.minTemplateLength,
            )
            logger.info(
                "counting.done %s samples=%d elapsed=%.3fs",
                chromosome,
                int(numSamples),
                time.perf_counter() - countStart,
            )

        if backgroundBlockSizeBP_ < 0:
            depPoint, depLower, depUpper, depDiagnostics = core.chooseDependenceLength(
                chromMat,
                intervalSizeBP,
                minSpan=3,
                maxSpan=64,
            )
            vec_ = (int(depPoint), int(depLower), int(depUpper))
            backgroundBlockSizeBP_ = int(depDiagnostics["context_size_bp"])
            backgroundBlockSizeIntervals = backgroundBlockSizeBP_ // intervalSizeBP
            logger.info(
                "`countingParams.backgroundBlockSizeBP < 0` --> "
                "chooseDependenceLength(): %d bp (span=%d, lower=%d, upper=%d)",
                int(backgroundBlockSizeBP_),
                int(depPoint),
                int(depLower),
                int(depUpper),
            )

        if samplingBlockSizeBP_ < 0:
            if backgroundBlockSizeBP_ > 0:
                samplingBlockSizeBP_ = backgroundBlockSizeBP_
            else:
                depPoint, depLower, depUpper, depDiagnostics = (
                    core.chooseDependenceLength(
                        chromMat,
                        intervalSizeBP,
                        minSpan=3,
                        maxSpan=64,
                    )
                )
                vec_ = (int(depPoint), int(depLower), int(depUpper))
                samplingBlockSizeBP_ = int(depDiagnostics["context_size_bp"])
                logger.info(
                    "`observationParams.samplingBlockSizeBP < 0` --> "
                    "chooseDependenceLength(): %d bp (span=%d, lower=%d, upper=%d)",
                    int(samplingBlockSizeBP_),
                    int(depPoint),
                    int(depLower),
                    int(depUpper),
                )

        denseCenterContextSizeBP = (
            backgroundBlockSizeBP_ if backgroundBlockSizeBP_ > 0 else intervalSizeBP
        )
        denseCenterBlockSizeBP = min(
            max(10 * int(denseCenterContextSizeBP), 1000),
            50000,
        )
        denseCenterBlockSizeIntervals = max(
            3,
            int(math.ceil(float(denseCenterBlockSizeBP) / float(intervalSizeBP))),
        )
        if args.verbose2:
            logger.info(
                "dense centering blocks for %s: context=%d bp block=%d bp intervals=%d",
                chromosome,
                int(denseCenterContextSizeBP),
                int(denseCenterBlockSizeBP),
                int(denseCenterBlockSizeIntervals),
            )

        if waitForMatrix:
            if c_ == 0:
                sf = cconsenrich.cSF(chromMat)
                logger.info(
                    f"`countingParams.normMethod=SF` --> calculating scaling factors\n{sf}\n",
                )
                _checkSF(sf, logger)
            np.multiply(chromMat, sf[:, None], out=chromMat)

        # negative --> data-based
        if observationArgs.minR < 0.0 or observationArgs.maxR < 0.0:
            minR_ = 0.0
            maxR_ = 1e4
        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            minQ_ = 0.0
            maxQ_ = 1e4

        def _transformTrack(j: int) -> int:
            cconsenrich.cTransformInPlace(
                chromMat[j, :],
                blockLength=denseCenterBlockSizeIntervals,
                verbose=args.verbose2,
                w_global=countingArgs.globalWeight,
                logOffset=countingArgs.logOffset,
                logMult=countingArgs.logMult,
                blockQuantile=observationArgs.blockQuantile,
            )
            return j

        if controlsPresent:
            logger.info(
                "Skipping ordinary count transform: treatment/control tracks "
                "were already transformed as log-ratios.",
            )
        else:
            transformStart = time.perf_counter()
            transformWorkers = _getSmallWorkerCount(numSamples, maxWorkers=4)
            useParallelTransform = (
                numSamples >= 4 and chromMat.shape[1] >= 5000 and transformWorkers > 1
            )
            if useParallelTransform:
                logger.info(
                    "transform matrix: using ThreadPool with %d workers (numSamples=%d, numIntervals=%d).",
                    int(transformWorkers),
                    int(numSamples),
                    int(chromMat.shape[1]),
                )
                with ThreadPool(processes=int(transformWorkers)) as pool:
                    for _ in _progress(
                        pool.imap(_transformTrack, range(numSamples)),
                        total=numSamples,
                        desc="Transforming data",
                        unit="sample",
                    ):
                        pass
            else:
                for j in _progress(
                    range(numSamples),
                    desc="Transforming data",
                    unit="sample",
                ):
                    _transformTrack(j)
            logger.info(
                "transform.done %s samples=%d elapsed=%.3fs",
                chromosome,
                int(numSamples),
                time.perf_counter() - transformStart,
            )

        useSparseNearest = bool(
            observationArgs.numNearest is not None
            and int(observationArgs.numNearest) > 0
            and genomeArgs.sparseBedFile
        )
        useSparseRestrictedLocalAR1 = bool(
            getattr(observationArgs, "restrictLocalAR1ToSparseBed", False)
            and genomeArgs.sparseBedFile
        )

        sparseIntervalIndices = None
        sparseRegionMask = None
        muncIntervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32)
        muncEmptyExcludeMask = np.zeros(numIntervals, dtype=np.uint8)
        if useSparseNearest:
            sparseIntervalIndices = _loadSparseIntervalIndices(
                genomeArgs.sparseBedFile,
                chromosome,
                intervals,
            )
            logger.info(
                "munc matrix: using explicit sparse-bed nearest-neighbor local variance "
                "(chrom=%s, numNearest=%d, sparseIntervals=%d).",
                chromosome,
                int(observationArgs.numNearest),
                int(sparseIntervalIndices.size),
            )
        if useSparseRestrictedLocalAR1:
            sparseRegionMask = core.getBedMask(
                chromosome,
                genomeArgs.sparseBedFile,
                intervals,
            )
            logger.info(
                "munc matrix: restricting rolling local AR(1) observation variance to "
                "sparse-bed regions (chrom=%s, sparseIntervals=%d).",
                chromosome,
                int(np.count_nonzero(sparseRegionMask)),
            )

        def _fitMuncTrack(j: int) -> tuple[int, np.ndarray]:
            muncTrack, _ = core.getMuncTrack(
                chromosome,
                intervals,
                chromMat[j, :],
                intervalSizeBP,
                samplingIters=observationArgs.samplingIters,
                samplingBlockSizeBP=samplingBlockSizeBP_,
                randomSeed=42 + j,
                EB_use=observationArgs.EB_use,
                EB_setNu0=observationArgs.EB_setNu0,
                EB_setNuL=observationArgs.EB_setNuL,
                trendNumBasis=observationArgs.trendNumBasis,
                trendMinObsPerBasis=observationArgs.trendMinObsPerBasis,
                trendMinEdf=observationArgs.trendMinEdf,
                trendMaxEdf=observationArgs.trendMaxEdf,
                trendLambdaMin=observationArgs.trendLambdaMin,
                trendLambdaMax=observationArgs.trendLambdaMax,
                trendLambdaGridSize=observationArgs.trendLambdaGridSize,
                sparseIntervalIndices=sparseIntervalIndices,
                sparseRegionMask=sparseRegionMask,
                numNearest=int(observationArgs.numNearest or 0),
                sparseSupportScaleBP=observationArgs.sparseSupportScaleBP,
                sparseSupportPrior=float(observationArgs.sparseSupportPrior or 0.0),
                restrictLocalAR1ToSparseBed=bool(
                    getattr(observationArgs, "restrictLocalAR1ToSparseBed", False)
                ),
                verbose=args.verbose2,
                varianceFloor=minR_ if minR_ is not None and minR_ > 0.0 else None,
                varianceCap=maxR_ if maxR_ is not None and maxR_ > 0.0 else None,
                intervalsArr=muncIntervalsArr,
                excludeMaskArr=muncEmptyExcludeMask,
            )
            return j, muncTrack

        # this has become a bottleneck, so gentle multiprocessing
        muncWorkers = _getMuncWorkerCount(
            numSamples,
            chromMat.shape[1],
            sharedArrays=(
                chromMat,
                muncMat,
                muncIntervalsArr,
                muncEmptyExcludeMask,
                sparseIntervalIndices,
                sparseRegionMask,
            ),
        )
        useParallelMunc = (
            numSamples >= 4 and chromMat.shape[1] >= 5000 and muncWorkers > 1
        )
        muncStart = time.perf_counter()
        logger.info(
            "munc.start %s samples=%d intervals=%d workers=%d",
            chromosome,
            int(numSamples),
            int(chromMat.shape[1]),
            int(muncWorkers if useParallelMunc else 1),
        )
        if useParallelMunc:
            logger.info(
                "munc matrix: using ThreadPool with %d workers (numSamples=%d, numIntervals=%d).",
                int(muncWorkers),
                int(numSamples),
                int(chromMat.shape[1]),
            )
            with ThreadPool(processes=int(muncWorkers)) as pool:
                for j, muncTrack in _progress(
                    pool.imap(_fitMuncTrack, range(numSamples)),
                    total=numSamples,
                    desc="Fitting variance function f(|mu|;Theta)",
                    unit="sample",
                ):
                    muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)
        else:
            for j in _progress(
                range(numSamples),
                desc="Fitting variance function f(|mu|;Theta)",
                unit="sample",
            ):
                _, muncTrack = _fitMuncTrack(j)
                muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)
        logger.info(
            "munc.done %s samples=%d elapsed=%.3fs",
            chromosome,
            int(numSamples),
            time.perf_counter() - muncStart,
        )

        if observationArgs.minR < 0.0 or observationArgs.maxR < 0.0:
            finiteMunc = muncMat[np.isfinite(muncMat)]
            minR_ = np.float32(
                max(
                    np.quantile(finiteMunc, 0.01) if finiteMunc.size else 1.0e-4,
                    1.0e-4,
                )
            )
            logger.info(
                "observationParams.minR < 0 or observationParams.maxR < 0 --> applying minimal numerically stable bounds for conditioning",
            )
        muncMat = np.nan_to_num(
            muncMat.astype(np.float32, copy=False),
            nan=np.float32(minR_),
            posinf=np.float32(maxR_),
            neginf=np.float32(minR_),
        )
        np.clip(
            muncMat,
            np.float32(minR_),
            np.float32(maxR_),
            out=muncMat,
        )
        minQ_ = processArgs.minQ
        maxQ_ = processArgs.maxQ

        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            if minR_ is None:
                minR_ = np.float32(max(np.quantile(muncMat, 0.01), 1.0e-4))
            autoMinQ = max((0.01 * minR_) * (1 + deltaF_), 1.0e-4)
            logger.info(
                "processParams.minQ < 0 or processParams.maxQ < 0 --> applying minimal numerically stable bounds for conditioning",
            )
            if processArgs.minQ < 0.0:
                minQ_ = autoMinQ
            else:
                minQ_ = np.float32(processArgs.minQ)
            if processArgs.maxQ < 0.0:
                maxQ_ = minQ_
            else:
                maxQ_ = np.float32(max(processArgs.maxQ, minQ_))
        else:
            maxQ_ = np.float32(max(maxQ_, minQ_))
        logger.info(f"minR={minR_}, maxR={maxR_}, minQ={minQ_}, maxQ={maxQ_}")
        if not bool(fitArgs.EM_use):
            logger.info(
                "fitParams.EM_use=False --> skipping iterative EM calibration and using the plugin variance track directly"
            )
        logger.info(f">>>  Running consenrich: {chromosome}  <<<")
        blockLenIntervals_ = (
            4 * vec_[0] + 1
            if vec_ is not None
            else 2 * backgroundBlockSizeIntervals + 1
        )
        runStart = time.perf_counter()
        logger.info(
            "runConsenrich.start %s intervals=%d samples=%d blocks=%d",
            chromosome,
            int(numIntervals),
            int(numSamples),
            int(np.ceil(numIntervals / float(blockLenIntervals_))),
        )
        useCrossFitUncertainty = bool(
            outputArgs.writeUncertainty and uncertaintyCalibrationArgs.enabled
        )
        runResult = core.runConsenrich(
            chromMat,
            muncMat,
            deltaF_,
            minQ_,
            maxQ_,
            offDiagQ_,
            stateArgs.stateInit,
            stateArgs.stateCovarInit,
            stateArgs.boundState,
            stateArgs.stateLowerBound,
            stateArgs.stateUpperBound,
            blockLenIntervals=blockLenIntervals_,
            returnScales=True,
            returnReplicateOffsets=True,
            pad=pad_,
            disableCalibration=(not bool(fitArgs.EM_use)),
            EM_maxIters=fitArgs.EM_maxIters,
            EM_innerRtol=fitArgs.EM_innerRtol,
            EM_tNu=fitArgs.EM_tNu,
            EM_useObsPrecReweight=fitArgs.EM_useObsPrecReweight,
            EM_useProcPrecReweight=fitArgs.EM_useProcPrecReweight,
            EM_useAPN=fitArgs.EM_useAPN,
            EM_useReplicateBias=fitArgs.EM_useReplicateBias,
            fitBackground=fitArgs.fitBackground,
            EM_zeroCenterBackground=fitArgs.EM_zeroCenterBackground,
            EM_zeroCenterReplicateBias=fitArgs.EM_zeroCenterReplicateBias,
            EM_outerIters=fitArgs.EM_outerIters,
            EM_outerRtol=fitArgs.EM_outerRtol,
            EM_backgroundSmoothness=fitArgs.EM_backgroundSmoothness,
            processQCalibration=processArgs.processQCalibration,
            processQCalibIters=processArgs.processQCalibIters,
            processQLevelTarget=processArgs.processQLevelTarget,
            processQTrendTarget=processArgs.processQTrendTarget,
            processQLevelPriorWeight=processArgs.processQLevelPriorWeight,
            processQTrendPriorWeight=processArgs.processQTrendPriorWeight,
            observationPrecisionMultiplierMin=observationArgs.precisionMultiplierMin,
            observationPrecisionMultiplierMax=observationArgs.precisionMultiplierMax,
            processPrecisionMultiplierMin=processArgs.precisionMultiplierMin,
            processPrecisionMultiplierMax=processArgs.precisionMultiplierMax,
            applyJackknife=outputArgs.applyJackknife,
            returnDiagnostics=True,
        )
        (
            x,
            P,
            postFitResiduals,
            JackknifeSEVec,
            qScale,
            replicateBias,
            intervalToBlockMap,
        ) = runResult[:7]
        runDiagnostics = (
            runResult[7]
            if len(runResult) > 7 and isinstance(runResult[7], Mapping)
            else {}
        )
        logger.info(
            "runConsenrich.done %s elapsed=%.3fs",
            chromosome,
            time.perf_counter() - runStart,
        )
        replicateBias = np.asarray(replicateBias, dtype=np.float32)
        logger.info(
            "finalReplicateBias[%s]=%s",
            chromosome,
            np.array2string(
                replicateBias,
                precision=6,
                floatmode="fixed",
                separator=", ",
            ),
        )

        x_ = core.getPrimaryState(
            x,
            stateLowerBound=stateArgs.stateLowerBound,
            stateUpperBound=stateArgs.stateUpperBound,
            boundState=stateArgs.boundState,
        )
        roughnessBlockLen = diagnostics.resolveUncertaintyBlockSizeIntervals(
            uncertaintyCalibrationArgs.blockSizeBP,
            intervalSizeBP,
            len(x_),
        )
        stateRoughness = diagnostics.summarizeStateRoughness(
            x_,
            blockLenIntervals=roughnessBlockLen,
            intervalSizeBP=intervalSizeBP,
        )
        roughnessStrata = {
            str(row.get("stratum", "")): row
            for row in stateRoughness.get("signal_strata", [])
            if isinstance(row, Mapping)
        }
        logger.info(
            "stateRoughness[%s]: block=%d intervals (%s bp) meanAbsDiff=%s "
            "blockMedian=%s blockQ90=%s signalLow/Mid/High=%s/%s/%s",
            chromosome,
            int(stateRoughness["block_len_intervals"]),
            _fmtDiagnosticFloat(stateRoughness.get("block_len_bp")),
            _fmtDiagnosticFloat(stateRoughness.get("overall_mean_abs_diff")),
            _fmtDiagnosticFloat(stateRoughness.get("block_mean_abs_diff_median")),
            _fmtDiagnosticFloat(stateRoughness.get("block_mean_abs_diff_q90")),
            _fmtDiagnosticFloat(
                roughnessStrata.get("signal_abs_q00_50", {}).get("mean_abs_diff")
            ),
            _fmtDiagnosticFloat(
                roughnessStrata.get("signal_abs_q50_90", {}).get("mean_abs_diff")
            ),
            _fmtDiagnosticFloat(
                roughnessStrata.get("signal_abs_q90_100", {}).get("mean_abs_diff")
            ),
        )
        precisionBoundaryHits = dict(
            runDiagnostics.get("precision_reweighting_boundary_hits", {})
        )
        obsBoundaryHits = dict(precisionBoundaryHits.get("observation", {}))
        procBoundaryHits = dict(precisionBoundaryHits.get("process", {}))
        logger.info(
            "precisionReweight.boundaryHits[%s]: obs lower=%d upper=%d total=%d; "
            "proc lower=%d upper=%d total=%d",
            chromosome,
            int(obsBoundaryHits.get("lower", 0)),
            int(obsBoundaryHits.get("upper", 0)),
            int(obsBoundaryHits.get("total", 0)),
            int(procBoundaryHits.get("lower", 0)),
            int(procBoundaryHits.get("upper", 0)),
            int(procBoundaryHits.get("total", 0)),
        )
        stateDiagnosticsByChromosome[chromosome] = {
            "state_roughness": stateRoughness,
            "precision_reweighting_boundary_hits": precisionBoundaryHits,
        }
        P00_ = (P[:, 0, 0]).astype(np.float32, copy=False)
        uncertaintyTrack = np.sqrt(P00_).astype(np.float32, copy=False)

        if useCrossFitUncertainty:
            try:
                from consenrich import uncertainty as uncertainty_module
            except ImportError as exc:
                raise RuntimeError(
                    "Cross-fit uncertainty calibration requires the optional "
                    "`consenrich.uncertainty` module and `consenrich.cuncertainty` "
                    "extension. Build/install Consenrich with uncertainty support, "
                    "or set `uncertaintyCalibration.enabled: false`."
                ) from exc

            calibrationRunKwargs = dict(
                deltaF=deltaF_,
                minQ=minQ_,
                maxQ=maxQ_,
                offDiagQ=offDiagQ_,
                stateInit=stateArgs.stateInit,
                stateCovarInit=stateArgs.stateCovarInit,
                boundState=stateArgs.boundState,
                stateLowerBound=stateArgs.stateLowerBound,
                stateUpperBound=stateArgs.stateUpperBound,
                blockLenIntervals=blockLenIntervals_,
                returnScales=True,
                returnReplicateOffsets=True,
                pad=pad_,
                disableCalibration=(not bool(fitArgs.EM_use)),
                EM_maxIters=fitArgs.EM_maxIters,
                EM_innerRtol=fitArgs.EM_innerRtol,
                EM_tNu=fitArgs.EM_tNu,
                EM_useObsPrecReweight=fitArgs.EM_useObsPrecReweight,
                EM_useProcPrecReweight=fitArgs.EM_useProcPrecReweight,
                EM_useAPN=fitArgs.EM_useAPN,
                EM_useReplicateBias=fitArgs.EM_useReplicateBias,
                fitBackground=fitArgs.fitBackground,
                EM_zeroCenterBackground=fitArgs.EM_zeroCenterBackground,
                EM_zeroCenterReplicateBias=fitArgs.EM_zeroCenterReplicateBias,
                EM_outerIters=fitArgs.EM_outerIters,
                EM_outerRtol=fitArgs.EM_outerRtol,
                EM_backgroundSmoothness=fitArgs.EM_backgroundSmoothness,
                processQCalibration=processArgs.processQCalibration,
                processQCalibIters=processArgs.processQCalibIters,
                processQLevelTarget=processArgs.processQLevelTarget,
                processQTrendTarget=processArgs.processQTrendTarget,
                processQLevelPriorWeight=processArgs.processQLevelPriorWeight,
                processQTrendPriorWeight=processArgs.processQTrendPriorWeight,
                observationPrecisionMultiplierMin=observationArgs.precisionMultiplierMin,
                observationPrecisionMultiplierMax=observationArgs.precisionMultiplierMax,
                processPrecisionMultiplierMin=processArgs.precisionMultiplierMin,
                processPrecisionMultiplierMax=processArgs.precisionMultiplierMax,
                applyJackknife=False,
            )
            calibrationPrefix = (
                f"consenrichOutput_{experimentName}_uncertaintyCalibration"
                f".v{__version__}"
            )
            calibrationResult = uncertainty_module.calibrateChromosomeStateUncertainty(
                matrixData=chromMat,
                matrixMunc=muncMat,
                fullState=x,
                fullCovar=P,
                fullReplicateBias=replicateBias,
                intervals=intervals,
                intervalSizeBP=intervalSizeBP,
                params=uncertaintyCalibrationArgs,
                runKwargs=calibrationRunKwargs,
                pad=pad_,
                outPrefix=calibrationPrefix,
                chromosome=chromosome,
            )
            uncertaintyTrack = np.asarray(
                calibrationResult.calibratedUncertainty,
                dtype=np.float32,
            )
            logger.info(
                "Cross-fit uncertainty calibration applied for %s: "
                "aObs=%.6g heldoutCells=%d",
                chromosome,
                float(calibrationResult.model.get("a_obs_factor", np.nan)),
                int(calibrationResult.model.get("heldout_cells", 0)),
            )

        df = pd.DataFrame(
            {
                "Chromosome": chromosome,
                "Start": intervals,
                "End": intervals + intervalSizeBP,
                "State": x_,
            }
        )

        if outputArgs.writeUncertainty:
            df["uncertainty"] = uncertaintyTrack

        cols_ = ["Chromosome", "Start", "End", "State"]

        if outputArgs.writeUncertainty:
            cols_.append("uncertainty")

        if outputArgs.writeJackknifeSE and outputArgs.applyJackknife:
            cols_.append("JackknifeSE")
            df["JackknifeSE"] = JackknifeSEVec.astype(np.float32, copy=False)

        df = df[cols_]
        suffixes = ["state"]
        if outputArgs.writeUncertainty:
            suffixes.append("uncertainty")
        if outputArgs.writeJackknifeSE and outputArgs.applyJackknife:
            suffixes.append("JackknifeSE")

        writeStart = time.perf_counter()
        for col, suffix in _progress(
            zip(cols_[3:], suffixes),
            total=len(suffixes),
            desc=f"Writing {chromosome}",
            unit="track",
        ):
            logger.info(
                f"{chromosome}: writing/appending to: consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
            )
            df[["Chromosome", "Start", "End", col]].to_csv(
                f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph",
                sep="\t",
                header=False,
                index=False,
                mode="a",
                float_format="%.4f",
                lineterminator="\n",
            )
        logger.info(
            "chromosome.done %s elapsed=%.3fs outputElapsed=%.3fs",
            chromosome,
            time.perf_counter() - chromosomeStartTime,
            time.perf_counter() - writeStart,
        )

    logger.info("Finished: output in human-readable format")

    for suffix in _progress(
        suffixes,
        total=len(suffixes),
        desc="Sorting bedGraphs",
        unit="track",
    ):
        bedgraphPath = (
            f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
        )
        try:
            _sortBedGraphInPlace(bedgraphPath)
        except Exception as ex:
            logger.warning(f"Failed to sort {bedgraphPath}:\n{ex}")

    if outputArgs.convertToBigWig:
        convertBedGraphToBigWig(
            experimentName,
            genomeArgs.chromSizesFile,
            suffixes=suffixes,
        )

    if peakCallingEnabled:
        try:
            logger.info("running Consenrich+ROCCO for peaks...")
            stateBedGraphPath = (
                f"consenrichOutput_{experimentName}_state.v{__version__}.bedGraph"
            )
            uncertaintyBedGraphPath = (
                f"consenrichOutput_{experimentName}_uncertainty.v{__version__}.bedGraph"
            )
            if not os.path.exists(uncertaintyBedGraphPath):
                logger.warning(
                    "Uncertainty bedGraph %s was not found; proceeding without model-based uncertainty.",
                    uncertaintyBedGraphPath,
                )
                uncertaintyBedGraphPath = None
            outName = peaks.solveRocco(
                stateBedGraphPath,
                uncertaintyBedGraphFile=uncertaintyBedGraphPath,
                tau0=float(matchingArgs.tau0),
                numBootstrap=int(matchingArgs.numBootstrap),
                thresholdZ=float(matchingArgs.thresholdZ),
                dependenceSpan=matchingArgs.dependenceSpan,
                gamma=matchingArgs.gamma,
                selectionPenalty=matchingArgs.selectionPenalty,
                gammaScale=float(matchingArgs.gammaScale),
                nestedRoccoIters=int(matchingArgs.nestedRoccoIters),
                nestedRoccoBudgetScale=float(matchingArgs.nestedRoccoBudgetScale),
                exportFilterUncertaintyMultiplier=float(
                    matchingArgs.exportFilterUncertaintyMultiplier
                ),
                randSeed=matchingArgs.randSeed,
                verbose=bool(args.verbose),
                stateDiagnosticsByChromosome=stateDiagnosticsByChromosome,
            )

            logger.info("Finished ROCCO peak calling. Written to %s", outName)
        except Exception as ex_:
            logger.warning(
                f"ROCCO peak calling raised an exception:\n\n\t{ex_}\n"
                f"Skipping peak-calling step...try running post hoc via `consenrich --match-bedGraph <bedGraphFile>`\n"
                f"\tSee ``consenrich -h`` for more details.\n"
            )


if __name__ == "__main__":
    main()
