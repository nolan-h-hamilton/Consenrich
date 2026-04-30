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
from scipy import stats

import consenrich.core as core
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
    if "fragments" in os.path.basename(lowerPath):
        return "FRAGMENTS"
    return "BAM"


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
    defaultFragmentCountMode: str = "cutsite",
) -> str:
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


def _resolveDeltaFAutoParams(
    deltaF: float,
    intervalSizeBP: int,
    sources: Sequence[core.inputSource],
    readLengths: Sequence[int | float],
    characteristicLengths: Sequence[int | float],
) -> tuple[float, bool, float, float]:
    if np.isfinite(deltaF) and float(deltaF) > 0.0:
        deltaFFixed = float(deltaF)
        return deltaFFixed, False, deltaFFixed, deltaFFixed

    effectiveFragmentLengths: List[float] = []
    for source, readLength, fragmentLength in zip(
        sources, readLengths, characteristicLengths
    ):
        sourceKind = str(source.sourceKind).upper()
        candidateLength = float(fragmentLength)
        if sourceKind == core.FRAGMENTS_SOURCE_KIND and float(readLength) > 0.0:
            candidateLength = float(readLength)
        elif candidateLength <= 0.0 and float(readLength) > 0.0:
            candidateLength = float(readLength)

        if np.isfinite(candidateLength) and candidateLength > 0.0:
            effectiveFragmentLengths.append(candidateLength)

    if effectiveFragmentLengths:
        medianFragmentLength = float(np.median(effectiveFragmentLengths))
    else:
        medianFragmentLength = float(intervalSizeBP)

    deltaFCenter = float(
        np.clip(
            0.5 * float(intervalSizeBP) / max(medianFragmentLength, 1.0),
            1.0e-4,
            2.0,
        )
    )
    deltaFSearchLow = float(max(1.0e-4, deltaFCenter * math.exp(-0.25)))
    deltaFSearchHigh = float(max(deltaFSearchLow, deltaFCenter * math.exp(0.25)))
    return deltaFCenter, True, deltaFSearchLow, deltaFSearchHigh


def _prioritizeLargestChromosomePlan(
    chromosomePlans: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    planList = list(chromosomePlans)
    if len(planList) <= 1:
        return planList

    largestIdx = max(
        range(len(planList)),
        key=lambda idx: int(planList[idx].get("numIntervals", 0)),
    )
    if largestIdx == 0:
        return planList
    return [planList[largestIdx]] + planList[:largestIdx] + planList[largestIdx + 1 :]


@lru_cache(maxsize=8)
def _readSparseStartsByChrom(sparseBedFile: str) -> dict[str, np.ndarray]:
    sparseFrame = pd.read_csv(
        sparseBedFile,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["chrom", "start"],
        dtype={"chrom": str, "start": np.int64},
        engine="c",
    )
    sparseStartsByChrom: dict[str, np.ndarray] = {}
    for chromName, chromFrame in sparseFrame.groupby("chrom", sort=False):
        sparseStartsByChrom[str(chromName)] = chromFrame["start"].to_numpy(
            dtype=np.int64,
            copy=False,
        )
    return sparseStartsByChrom


def _loadSparseIntervalIndices(
    sparseBedFile: str,
    chromosome: str,
    intervals: np.ndarray,
) -> np.ndarray:
    sparseStarts = _readSparseStartsByChrom(str(sparseBedFile)).get(
        str(chromosome),
        np.empty(0, dtype=np.int64),
    )
    if sparseStarts.size == 0:
        return np.empty(0, dtype=np.intp)

    intervalStarts = np.asarray(intervals, dtype=np.int64)
    sparseIdx = np.searchsorted(intervalStarts, sparseStarts, side="right") - 1
    sparseIdx = sparseIdx[(sparseIdx >= 0) & (sparseIdx < intervalStarts.size)]
    if sparseIdx.size == 0:
        return np.empty(0, dtype=np.intp)
    return np.unique(sparseIdx.astype(np.intp, copy=False))


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
        constants.getEffectiveGenomeSize(genomeName, readLength)
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

    bamFiles = core.getSourcePaths(treatmentSources)
    bamFilesControl = core.getSourcePaths(controlSources)

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

    if len(controlSources) == 1:
        logger.info(
            f"Only one control given: Using {bamFilesControl[0]} for all treatment files."
        )
        controlSources = controlSources * len(treatmentSources)
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
        True if shutil.which("bedGraphToBigWig") else False,
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
    effectiveInfoRescale_ = _cfgGet(
        configData,
        "stateParams.effectiveInfoRescale",
        True,
    )
    effectiveInfoBlockLengthBP_ = _cfgGet(
        configData,
        "stateParams.effectiveInfoBlockLengthBP",
        _cfgGet(configData, "stateParams.effectiveInfoBlockLength", 50_000),
    )
    return core.stateParams(
        stateInit=stateInit_,
        stateCovarInit=stateCovarInit_,
        boundState=boundState_,
        stateLowerBound=stateLowerBound_,
        stateUpperBound=stateUpperBound_,
        effectiveInfoRescale=effectiveInfoRescale_,
        effectiveInfoBlockLengthBP=effectiveInfoBlockLengthBP_,
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
        True,
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


def _resolveEffectiveInfoBandwidthIntervals(
    contextVector: tuple[int, int, int] | None,
    blockLenIntervals: int,
) -> int:
    if contextVector is not None:
        return max(int(contextVector[0]), 1)
    return max(int((int(blockLenIntervals) - 1) // 4), 1)


def _resolveEffectiveInfoBlockLengthIntervals(
    effectiveInfoBlockLengthBP: int | float | None,
    intervalSizeBP: int,
    intervalCount: int,
) -> int:
    if effectiveInfoBlockLengthBP is None or float(effectiveInfoBlockLengthBP) <= 0.0:
        return max(int(intervalCount), 1)
    return max(
        1,
        min(
            int(np.ceil(float(effectiveInfoBlockLengthBP) / float(intervalSizeBP))),
            max(int(intervalCount), 1),
        ),
    )


def getScArgs(config_path: str) -> core.scParams:
    configData = loadConfig(config_path)

    barcodeTag_ = _cfgGet(configData, "scParams.barcodeTag", "CB")
    defaultCountMode_ = _cfgGet(
        configData,
        "scParams.defaultCountMode",
        "cutsite",
    )
    if str(defaultCountMode_).strip().lower() not in [
        "coverage",
        "cov",
        "cutsite",
        "cut",
        "cutsites",
        "fiveprime",
        "5p",
        "five_prime",
        "center",
        "centre",
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
    experimentName = _cfgGet(configData, "experimentName", "consenrichExperiment")
    processArgs = core.processParams(
        deltaF=_cfgGet(configData, "processParams.deltaF", -1.0),
        minQ=_cfgGet(configData, "processParams.minQ", 2.5e-4),
        maxQ=_cfgGet(configData, "processParams.maxQ", 1000.0),
        offDiagQ=_cfgGet(
            configData,
            "processParams.offDiagQ",
            0.0,
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
        binQuantileCutoff=_cfgGet(
            configData,
            "observationParams.binQuantileCutoff",
            0.5,
        ),
        EB_minLin=float(
            _cfgGet(
                configData,
                "observationParams.EB_minLin",
                0.0,
            )
        ),
        EB_use=_cfgGet(
            configData,
            "observationParams.EB_use",
            True,
        ),
        EB_setNu0=_cfgGet(configData, "observationParams.EB_setNu0", None),
        EB_setNuL=_cfgGet(configData, "observationParams.EB_setNuL", None),
        numNearest=numNearestResolved,
        restrictLocalAR1ToSparseBed=restrictLocalAR1ToSparseBedResolved,
        pad=_cfgGet(configData, "observationParams.pad", 1.0e-3),
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
        EM_repBiasShrink=_cfgGet(
            configData,
            "fitParams.EM_repBiasShrink",
            0.0,
        ),
        EM_outerIters=_cfgGet(
            configData,
            "fitParams.EM_outerIters",
            8,
        ),
        EM_outerRtol=_cfgGet(
            configData,
            "fitParams.EM_outerRtol",
            1.0e-2,
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
        0,
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
                2.5,
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
    path_ = ""
    warningMessage = (
        "Could not find UCSC bedGraphToBigWig binary utility."
        "If you need bigWig files instead of the default, human-readable bedGraph files,"
        "you can download the `bedGraphToBigWig` binary from https://hgdownload.soe.ucsc.edu/admin/exe/<operatingSystem, architecture>"
        "OR install via conda (conda install -c bioconda ucsc-bedgraphtobigwig)."
    )

    logger.info("Attempting to generate bigWig files from bedGraph format...")
    try:
        path_ = shutil.which("bedGraphToBigWig")
    except Exception:
        logger.warning(f"\n{warningMessage}\n")
        path_ = ""
    if path_ is None or len(path_) == 0:
        logger.warning(f"\n{warningMessage}\n")
    else:
        logger.info(f"Using bedGraphToBigWig from {path_}")
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
        ucscSucceeded = False
        if path_ is not None and len(path_) > 0:
            try:
                subprocess.run([path_, bedgraph, chromSizesFile, bigwig], check=True)
                ucscSucceeded = True
            except Exception as e:
                logger.warning(
                    f"bedGraph-->bigWig conversion with\n\n\t`bedGraphToBigWig {bedgraph} {chromSizesFile} {bigwig}`\nraised: \n{e}\n\n"
                )
        if not ucscSucceeded:
            try:
                logger.info(
                    "Trying pyBigWig streaming fallback for %s --> %s...",
                    bedgraph,
                    bigwig,
                )
                _convertBedGraphToBigWigPyBigWig(
                    bedgraph,
                    chromSizesFile,
                    bigwig,
                )
            except Exception as e:
                logger.warning(
                    f"pyBigWig bedGraph-->bigWig fallback for {bedgraph} raised:\n{e}\n"
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
            "pyBigWig is not installed; cannot use streaming bigWig fallback"
        ) from e

    chromSizes: List[Tuple[str, int]] = []
    with open(chromSizesFile, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split()
            if len(parts) < 2:
                continue
            chromSizes.append((str(parts[0]), int(parts[1])))
    if len(chromSizes) == 0:
        raise ValueError(f"No chromosome sizes found in {chromSizesFile}")

    chunkSize_ = max(int(chunkSize), 1)
    outDir = os.path.dirname(os.path.abspath(bigwigPath)) or "."
    tempPath = ""
    bw = None
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
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                chroms.append(str(parts[0]))
                starts.append(int(parts[1]))
                ends.append(int(parts[2]))
                values.append(float(parts[3]))
                if len(chroms) >= chunkSize_:
                    bw.addEntries(chroms, starts, ends=ends, values=values)
                    chroms.clear()
                    starts.clear()
                    ends.clear()
                    values.clear()
        if len(chroms) > 0:
            bw.addEntries(chroms, starts, ends=ends, values=values)
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
    samArgs = config["samArgs"]
    matchingArgs = config["matchingArgs"]
    fitArgs = config["fitArgs"]
    treatmentSources = _listOrEmpty(getattr(inputArgs, "treatmentSources", None))
    controlSources = _listOrEmpty(getattr(inputArgs, "controlSources", None))
    if not treatmentSources:
        treatmentSources = _buildPathInputSources(
            inputArgs.bamFiles, role="treatment"
        )
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
    vec_: Optional[np.ndarray] = None
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
    if anyFragments and normMethod_ in ["EGS", "RPGC"]:
        logger.warning(
            "Fragments inputs use insertion-based depth normalization not EGS/RPGC"
            "  --> using CPM/RPKM ..."
        )
        normMethod_ = "CPM"
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
    treatmentCountModes = [
        _getSourceCountMode(
            source,
            str(samArgs.defaultCountMode or "coverage"),
            str(scArgs.defaultCountMode or "cutsite"),
        )
        for source in treatmentSources
    ]
    controlCountModes = [
        _getSourceCountMode(
            source,
            str(samArgs.defaultCountMode or "coverage"),
            str(scArgs.defaultCountMode or "cutsite"),
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
        if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND:
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
        if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND:
            return 0
        if sourceBamInputMode == "fragments" and (
            int(configuredExtendBP) > 0 or int(samArgs.inferFragmentLength or 0) > 0
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
        if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND:
            return 0
        if sourceBamInputMode == "fragments":
            return 0
        if int(configuredExtendBP) > 0:
            return int(configuredExtendBP)
        if int(samArgs.inferFragmentLength or 0) > 0:
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
            effectiveGenomeSizesControl = [
                constants.getEffectiveGenomeSize(genomeArgs.genomeName, readLength)
                for readLength in readLengthsControlBamFiles
            ]

            if scaleFactors is not None and scaleFactorsControl is not None:
                treatScaleFactors = scaleFactors
                controlScaleFactors = scaleFactorsControl
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
            if normMethod_ in ["RPKM", "CPM"]:

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

    deltaFCenter_, autoDeltaF_, autoDeltaFLow_, autoDeltaFHigh_ = (
        _resolveDeltaFAutoParams(
            deltaF_,
            intervalSizeBP,
            treatmentSources,
            readLengthsBamFiles,
            characteristicFragmentLengthsTreatment,
        )
    )
    if autoDeltaF_:
        logger.info(
            "processParams.deltaF < 0 --> centering autoDeltaF at %.6f with bounds [%.6f, %.6f]",
            deltaFCenter_,
            autoDeltaFLow_,
            autoDeltaFHigh_,
        )
    else:
        logger.info("Using fixed deltaF=%.6f", deltaFCenter_)

    chromSizesDict = misc_util.getChromSizesDict(
        genomeArgs.chromSizesFile,
        excludeChroms=genomeArgs.excludeChroms,
    )
    chromosomes = genomeArgs.chromosomes
    treatmentSourceKinds = [
        str(source.sourceKind).upper() for source in treatmentSources
    ]
    chromosomePlans: List[Dict[str, Any]] = []
    for chromosome in chromosomes:
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

    if autoDeltaF_ and chromosomePlans:
        chromosomePlans = _prioritizeLargestChromosomePlan(chromosomePlans)

    autoDeltaFResolved = not bool(autoDeltaF_)

    if chromosomePlans:
        for file_ in os.listdir("."):
            if file_.startswith(f"consenrichOutput_{experimentName}") and (
                file_.endswith(".bedGraph") or file_.endswith(".narrowPeak")
            ):
                logger.warning(f"Overwriting: {file_}")
                os.remove(file_)

    for c_, chromPlan in enumerate(chromosomePlans):
        chromosome = str(chromPlan["chromosome"])
        chromosomeStart = int(chromPlan["start"])
        chromosomeEnd = int(chromPlan["end"])
        numIntervals = int(chromPlan["numIntervals"])
        intervals = np.arange(chromosomeStart, chromosomeEnd, intervalSizeBP)
        chromMat: np.ndarray = np.empty((numSamples, numIntervals), dtype=np.float32)
        muncMat: np.ndarray = np.empty_like(chromMat, dtype=np.float32)
        if controlsPresent:
            j_: int = 0
            for bamA, bamB in zip(bamFiles, bamFilesControl):
                logger.info(f"Counting (trt,ctrl) for {chromosome}: ({bamA}, {bamB})")

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
                logger.info(f"(trt,ctrl) for {chromosome}: ({bamA}, {bamB})")
                chromMat[j_, :] = np.maximum(pairMatrix[0, :] - pairMatrix[1, :], 0.0)
                j_ += 1
        else:
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

        if backgroundBlockSizeBP_ < 0:
            vec_ = core.getContextSize(
                stats.trim_mean(chromMat, proportiontocut=0.1, axis=0)
            )
            backgroundBlockSizeBP_ = vec_[0] * (2 * intervalSizeBP) + 1
            backgroundBlockSizeIntervals = backgroundBlockSizeBP_ // intervalSizeBP
            logger.info(
                f"`countingParams.backgroundBlockSizeBP < 0` --> getContextSize(): {backgroundBlockSizeBP_} bp"
            )

        if samplingBlockSizeBP_ < 0:
            if backgroundBlockSizeBP_ > 0:
                samplingBlockSizeBP_ = backgroundBlockSizeBP_
            else:
                samplingBlockSizeBP_ = (
                    core.getContextSize(
                        stats.trim_mean(chromMat, proportiontocut=0.1, axis=0)
                    )[0]
                    * (2 * intervalSizeBP)
                    + 1
                )
                logger.info(
                    f"`observationParams.samplingBlockSizeBP < 0` --> getContextSize(): {samplingBlockSizeBP_} bp"
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

        def _transformTrack(j: int) -> tuple[int, np.ndarray]:
            transformed = cconsenrich.cTransform(
                chromMat[j, :],
                blockLength=denseCenterBlockSizeIntervals,
                verbose=args.verbose2,
                w_global=countingArgs.globalWeight,
                logOffset=countingArgs.logOffset,
                logMult=countingArgs.logMult,
            )
            return j, np.asarray(transformed, dtype=np.float32)

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
                for j, transformed in tqdm(
                    pool.imap(_transformTrack, range(numSamples)),
                    total=numSamples,
                    desc="Transforming data",
                    unit=" sample ",
                ):
                    chromMat[j, :] = transformed
        else:
            for j in tqdm(
                range(numSamples),
                desc="Transforming data",
                unit=" sample ",
            ):
                logger.info(f"\n{chromosome}, sample {j + 1} / {numSamples}...")
                _, transformed = _transformTrack(j)
                chromMat[j, :] = transformed

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
                binQuantileCutoff=observationArgs.binQuantileCutoff,
                EB_minLin=observationArgs.EB_minLin,
                randomSeed=42 + j,
                EB_use=observationArgs.EB_use,
                EB_setNu0=observationArgs.EB_setNu0,
                EB_setNuL=observationArgs.EB_setNuL,
                sparseIntervalIndices=sparseIntervalIndices,
                sparseRegionMask=sparseRegionMask,
                numNearest=int(observationArgs.numNearest or 0),
                restrictLocalAR1ToSparseBed=bool(
                    getattr(observationArgs, "restrictLocalAR1ToSparseBed", False)
                ),
                verbose=args.verbose2,
            )
            return j, muncTrack

        # this has become a bottleneck, so gentle multiprocessing
        cpuCount = os.cpu_count() or 1
        muncWorkers = min(
            numSamples,
            max(1, cpuCount // 2),
        )
        useParallelMunc = (
            numSamples >= 4 and chromMat.shape[1] >= 5000 and muncWorkers > 1
        )
        if useParallelMunc:
            logger.info(
                "munc matrix: using ThreadPool with %d workers (numSamples=%d, numIntervals=%d).",
                int(muncWorkers),
                int(numSamples),
                int(chromMat.shape[1]),
            )
            with ThreadPool(processes=int(muncWorkers)) as pool:
                for j, muncTrack in tqdm(
                    pool.imap(_fitMuncTrack, range(numSamples)),
                    total=numSamples,
                    desc="Fitting variance function f(|mu|;Theta)",
                    unit=" sample ",
                ):
                    muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)
        else:
            for j in tqdm(
                range(numSamples),
                desc="Fitting variance function f(|mu|;Theta)",
                unit=" sample ",
            ):
                _, muncTrack = _fitMuncTrack(j)
                muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)

        if observationArgs.minR < 0.0 or observationArgs.maxR < 0.0:
            minR_ = np.float32(max(np.quantile(muncMat, 0.01), 1.0e-4))
            logger.info(
                "observationParams.minR < 0 or observationParams.maxR < 0 --> applying minimal numerically stable bounds for conditioning",
            )
            muncMat = muncMat.astype(np.float32, copy=False)
        minQ_ = processArgs.minQ
        maxQ_ = processArgs.maxQ

        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            if minR_ is None:
                minR_ = np.float32(max(np.quantile(muncMat, 0.01), 1.0e-4))
            autoMinQ = max((0.01 * minR_) * (1 + deltaFCenter_), 1.0e-4)
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
        if (not autoDeltaFResolved) and autoDeltaF_:
            deltaFCenter_ = core.estimateAutoDeltaF(
                matrixData=chromMat,
                matrixMunc=muncMat,
                minQ=float(minQ_),
                offDiagQ=float(offDiagQ_),
                stateInit=float(stateArgs.stateInit),
                stateCovarInit=float(stateArgs.stateCovarInit),
                blockLenIntervals=(
                    4 * vec_[0] + 1
                    if vec_ is not None
                    else 2 * backgroundBlockSizeIntervals + 1
                ),
                pad=float(pad_),
                autoDeltaF_low=float(autoDeltaFLow_),
                autoDeltaF_high=float(autoDeltaFHigh_),
                autoDeltaF_init=float(deltaFCenter_),
            )
            autoDeltaF_ = False
            autoDeltaFLow_ = float(deltaFCenter_)
            autoDeltaFHigh_ = float(deltaFCenter_)
            autoDeltaFResolved = True
            logger.info(
                "Using fixed deltaF=%.6f estimated from largest processed chromosome %s",
                deltaFCenter_,
                chromosome,
            )
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
        effectiveInfoBandwidthIntervals_ = _resolveEffectiveInfoBandwidthIntervals(
            vec_,
            blockLenIntervals_,
        )
        effectiveInfoBlockLengthIntervals_ = _resolveEffectiveInfoBlockLengthIntervals(
            stateArgs.effectiveInfoBlockLengthBP,
            intervalSizeBP,
            numIntervals,
        )
        logger.info(
            "Effective-information intervals for %s: bandwidth=%d blockLength=%d",
            chromosome,
            int(effectiveInfoBandwidthIntervals_),
            int(effectiveInfoBlockLengthIntervals_),
        )
        (
            x,
            P,
            postFitResiduals,
            JackknifeSEVec,
            qScale,
            replicateBias,
            intervalToBlockMap,
        ) = core.runConsenrich(
            chromMat,
            muncMat,
            deltaFCenter_,
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
            EM_zeroCenterBackground=fitArgs.EM_zeroCenterBackground,
            EM_zeroCenterReplicateBias=fitArgs.EM_zeroCenterReplicateBias,
            EM_repBiasShrink=fitArgs.EM_repBiasShrink,
            EM_outerIters=fitArgs.EM_outerIters,
            EM_outerRtol=fitArgs.EM_outerRtol,
            EM_backgroundSmoothness=fitArgs.EM_backgroundSmoothness,
            autoDeltaF=autoDeltaF_,
            autoDeltaF_low=autoDeltaFLow_,
            autoDeltaF_high=autoDeltaFHigh_,
            autoDeltaF_init=deltaFCenter_,
            applyJackknife=outputArgs.applyJackknife,
            effectiveInfoRescale=stateArgs.effectiveInfoRescale,
            effectiveInfoBandwidthIntervals=effectiveInfoBandwidthIntervals_,
            effectiveInfoBlockLengthIntervals=effectiveInfoBlockLengthIntervals_,
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
        P00_ = (P[:, 0, 0]).astype(np.float32, copy=False)

        df = pd.DataFrame(
            {
                "Chromosome": chromosome,
                "Start": intervals,
                "End": intervals + intervalSizeBP,
                "State": x_,
            }
        )

        if outputArgs.writeUncertainty:
            df["uncertainty"] = np.sqrt(P00_).astype(np.float32, copy=False)

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

        for col, suffix in zip(cols_[3:], suffixes):
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

    logger.info("Finished: output in human-readable format")

    for suffix in suffixes:
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
