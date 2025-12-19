#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import math
import pprint
import os
from pathlib import Path
from collections.abc import Mapping
from textwrap import dedent
from typing import List, Optional, Tuple, Dict, Any, Union, Sequence
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pysam
import pywt
import yaml
from tqdm import tqdm

import consenrich.core as core
import consenrich.misc_util as misc_util
import consenrich.constants as constants
import consenrich.detrorm as detrorm
import consenrich.matching as matching
import consenrich.cconsenrich as cconsenrich
from . import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _resolveFragmentLengthPairs(
    treatmentFragmentLengths: Optional[Sequence[Union[int, float]]],
    controlFragmentLengths: Optional[Sequence[Union[int, float]]],
) -> Tuple[List[int], List[int]]:
    r"""Assign consistent fragment length estimates to treatment and control BAM files.

    For single-end data, cross-correlation-based fragment estimates for control inputs
    can be much smaller than for treatment samples due to lack of structure. This creates
    artifacts during signal quantification and normalization steps, and it's common to use
    the treatment fragment length for both treatment and control samples. So we offer that here.
    """

    if not treatmentFragmentLengths:
        logger.warning(
            "No treatment fragment lengths provided...returning [],[]"
        )
        return [], []

    n_treat = len(treatmentFragmentLengths)

    if controlFragmentLengths:
        if len(controlFragmentLengths) == 1 and n_treat > 1:
            controlFragmentLengths = (
                list(controlFragmentLengths) * n_treat
            )
            logger.info(
                "Only one control fragment length provided: broadcasting this value for all control BAM files."
            )
        elif len(controlFragmentLengths) != n_treat:
            logger.warning(
                "Sizes of treatment and control fragment length lists are incompatible...returning [],[]"
            )
            return [], []
        else:
            controlFragmentLengths = list(controlFragmentLengths)
    else:
        controlFragmentLengths = list(treatmentFragmentLengths)

    finalTreatment = [int(x) for x in treatmentFragmentLengths]
    finalControl = [int(x) for x in treatmentFragmentLengths]

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
        raise ValueError(
            "No BAM files provided in the input arguments."
        )

    if (
        not isinstance(inputArgs.bamFiles, list)
        or len(inputArgs.bamFiles) == 0
    ):
        raise ValueError("bam files list is empty")

    return [
        core.getReadLength(
            bamFile,
            100,
            1000,
            samArgs.samThreads,
            samArgs.samFlagExclude,
        )
        for bamFile in inputArgs.bamFiles
    ]


def checkMatchingEnabled(matchingArgs: core.matchingParams) -> bool:
    matchingEnabled = (
        (matchingArgs.templateNames is not None)
        and isinstance(matchingArgs.templateNames, list)
        and len(matchingArgs.templateNames) > 0
    )
    matchingEnabled = (
        matchingEnabled
        and (matchingArgs.cascadeLevels is not None)
        and isinstance(matchingArgs.cascadeLevels, list)
        and len(matchingArgs.cascadeLevels) > 0
    )
    return matchingEnabled


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

    def expandWildCards(bamList: List[str]) -> List[str]:
        expandedList: List[str] = []
        for bamEntry in bamList:
            if "*" in bamEntry or "?" in bamEntry or "[" in bamEntry:
                matchedList = glob.glob(bamEntry)
            else:
                expandedList.append(bamEntry)
        return expandedList

    bamFilesRaw = (
        _cfgGet(configData, "inputParams.bamFiles", []) or []
    )
    bamFilesControlRaw = (
        _cfgGet(configData, "inputParams.bamFilesControl", []) or []
    )

    bamFiles = expandWildCards(bamFilesRaw)
    bamFilesControl = expandWildCards(bamFilesControlRaw)

    if len(bamFiles) == 0:
        raise ValueError(
            "No BAM files provided in the configuration."
        )

    if (
        len(bamFilesControl) > 0
        and len(bamFilesControl) != len(bamFiles)
        and len(bamFilesControl) != 1
    ):
        raise ValueError(
            "Number of control BAM files must be 0, 1, or the same as number of treatment files"
        )

    if len(bamFilesControl) == 1:
        logger.info(
            f"Only one control given: Using {bamFilesControl[0]} for all treatment files."
        )
        bamFilesControl = bamFilesControl * len(bamFiles)

    if not bamFiles or not isinstance(bamFiles, list):
        raise ValueError("No BAM files found")

    for bamFile in bamFiles:
        misc_util.checkBamFile(bamFile)

    if bamFilesControl:
        for bamFile in bamFilesControl:
            misc_util.checkBamFile(bamFile)

    pairedEndList = misc_util.bamsArePairedEnd(bamFiles)
    pairedEndConfig: Optional[bool] = _cfgGet(
        configData, "inputParams.pairedEnd", None
    )
    if pairedEndConfig is None:
        pairedEndConfig = all(pairedEndList)
        if pairedEndConfig:
            logger.info("Paired-end BAM files detected")
        else:
            logger.info("One or more single-end BAM files detected")

    return core.inputParams(
        bamFiles=bamFiles,
        bamFilesControl=bamFilesControl,
        pairedEnd=pairedEndConfig,
    )


def getOutputArgs(config_path: str) -> core.outputParams:
    configData = loadConfig(config_path)

    convertToBigWig_ = _cfgGet(
        configData,
        "outputParams.convertToBigWig",
        True if shutil.which("bedGraphToBigWig") else False,
    )

    roundDigits_ = _cfgGet(configData, "outputParams.roundDigits", 3)

    writeResiduals_ = _cfgGet(
        configData,
        "outputParams.writeResiduals",
        True,
    )

    writeMuncTrace: bool = _cfgGet(
        configData, "outputParams.writeMuncTrace", False
    )

    writeStateStd: bool = _cfgGet(
        configData,
        "outputParams.writeStateStd",
        True,
    )

    return core.outputParams(
        convertToBigWig=convertToBigWig_,
        roundDigits=roundDigits_,
        writeResiduals=writeResiduals_,
        writeMuncTrace=writeMuncTrace,
        writeStateStd=writeStateStd,
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
        chromSizesFile = constants.getGenomeResourceFile(
            genomeLabel, "sizes"
        )
        blacklistFile = constants.getGenomeResourceFile(
            genomeLabel, "blacklist"
        )
        sparseBedFile = constants.getGenomeResourceFile(
            genomeLabel, "sparse"
        )

    chromSizesOverride = _cfgGet(
        configData, "genomeParams.chromSizesFile", None
    )
    if chromSizesOverride:
        chromSizesFile = chromSizesOverride

    blacklistOverride = _cfgGet(
        configData, "genomeParams.blacklistFile", None
    )
    if blacklistOverride:
        blacklistFile = blacklistOverride

    sparseOverride = _cfgGet(
        configData, "genomeParams.sparseBedFile", None
    )
    if sparseOverride:
        sparseBedFile = sparseOverride

    if not chromSizesFile or not os.path.exists(chromSizesFile):
        raise FileNotFoundError(
            f"Chromosome sizes file {chromSizesFile} does not exist."
        )

    chromosomesConfig = _cfgGet(
        configData, "genomeParams.chromosomes", None
    )
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
        True,
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
            raise ValueError(
                "`stateLowerBound` is greater than `stateUpperBound`."
            )

    return core.stateParams(
        stateInit=stateInit_,
        stateCovarInit=stateCovarInit_,
        boundState=boundState_,
        stateLowerBound=stateLowerBound_,
        stateUpperBound=stateUpperBound_,
    )


def getCountingArgs(config_path: str) -> core.countingParams:
    configData = loadConfig(config_path)

    stepSize = _cfgGet(configData, "countingParams.stepSize", 25)
    backgroundWindowSizeBP = _cfgGet(
        configData, "countingParams.backgroundWindowSizeBP", 25_000,
    )
    scaleFactorList = _cfgGet(
        configData, "countingParams.scaleFactors", None
    )
    scaleFactorsControlList = _cfgGet(
        configData, "countingParams.scaleFactorsControl", None
    )
    if scaleFactorList is not None and not isinstance(
        scaleFactorList, list
    ):
        raise ValueError("`scaleFactors` should be a list of floats.")

    if scaleFactorsControlList is not None and not isinstance(
        scaleFactorsControlList, list
    ):
        raise ValueError(
            "`scaleFactorsControl` should be a list of floats."
        )

    if (
        scaleFactorList is not None
        and scaleFactorsControlList is not None
        and len(scaleFactorList) != len(scaleFactorsControlList)
    ):
        if len(scaleFactorsControlList) == 1:
            scaleFactorsControlList = scaleFactorsControlList * len(
                scaleFactorList
            )
        else:
            raise ValueError(
                "control and treatment scale factors: must be equal length or 1 control"
            )

    normMethod_ = _cfgGet(
        configData,
        "countingParams.normMethod",
        "EGS",
    )
    if normMethod_.upper() not in ["EGS", "RPKM"]:
        logger.warning(
            f"Unknown `countingParams.normMethod`...Using `EGS`...",
        )
        normMethod_ = "EGS"

    fragmentLengths: Optional[List[int]] = _cfgGet(
        configData,
        "countingParams.fragmentLengths",
        None,
    )
    fragmentLengthsControl: Optional[List[int]] = _cfgGet(
        configData,
        "countingParams.fragmentLengthsControl",
        None,
    )

    if fragmentLengths is not None and not isinstance(
        fragmentLengths, list
    ):
        raise ValueError(
            "`fragmentLengths` should be a list of integers."
        )
    if fragmentLengthsControl is not None and not isinstance(
        fragmentLengthsControl, list
    ):
        raise ValueError(
            "`fragmentLengthsControl` should be a list of integers."
        )
    if (
        fragmentLengths is not None
        and fragmentLengthsControl is not None
        and len(fragmentLengths) != len(fragmentLengthsControl)
    ):
        if len(fragmentLengthsControl) == 1:
            fragmentLengthsControl = fragmentLengthsControl * len(
                fragmentLengths
            )
        else:
            raise ValueError(
                "control and treatment fragment lengths: must be equal length or 1 control"
            )

    useTreatmentFragmentLengths_ = _cfgGet(
        configData,
        "countingParams.useTreatmentFragmentLengths",
        True,
    )

    return core.countingParams(
        stepSize=stepSize,
        backgroundWindowSizeBP=backgroundWindowSizeBP,
        scaleFactors=scaleFactorList,
        scaleFactorsControl=scaleFactorsControlList,
        normMethod=normMethod_,
        fragmentLengths=fragmentLengths,
        fragmentLengthsControl=fragmentLengthsControl,
        useTreatmentFragmentLengths=useTreatmentFragmentLengths_,
    )


def getPlotArgs(
    config_path: str, experimentName: str
) -> core.plotParams:
    configData = loadConfig(config_path)

    plotPrefix_ = _cfgGet(
        configData, "plotParams.plotPrefix", experimentName
    )

    plotStateEstimatesHistogram_ = _cfgGet(
        configData,
        "plotParams.plotStateEstimatesHistogram",
        False,
    )

    plotResidualsHistogram_ = _cfgGet(
        configData,
        "plotParams.plotResidualsHistogram",
        False,
    )

    plotStateStdHistogram_ = _cfgGet(
        configData,
        "plotParams.plotStateStdHistogram",
        False,
    )

    plotHeightInches_ = _cfgGet(
        configData,
        "plotParams.plotHeightInches",
        6.0,
    )

    plotWidthInches_ = _cfgGet(
        configData,
        "plotParams.plotWidthInches",
        8.0,
    )

    plotDPI_ = _cfgGet(
        configData,
        "plotParams.plotDPI",
        300,
    )

    plotDirectory_ = _cfgGet(
        configData,
        "plotParams.plotDirectory",
        os.path.join(
            os.getcwd(), f"{experimentName}_consenrichPlots"
        ),
    )

    if (
        int(plotStateEstimatesHistogram_)
        + int(plotResidualsHistogram_)
        + int(plotStateStdHistogram_)
        >= 1
    ):
        if plotDirectory_ is not None and (
            not os.path.exists(plotDirectory_)
            or not os.path.isdir(plotDirectory_)
        ):
            try:
                os.makedirs(plotDirectory_, exist_ok=True)
            except Exception as e:
                logger.warning(
                    f"Failed to create {plotDirectory_}:\n\t{e}\nUsing CWD."
                )
                plotDirectory_ = os.getcwd()
        elif plotDirectory_ is None:
            plotDirectory_ = os.getcwd()

        elif os.path.exists(plotDirectory_) and os.path.isdir(
            plotDirectory_
        ):
            logger.warning(
                f"Using existing plot directory: {plotDirectory_}"
            )
        else:
            logger.warning(
                f"Failed creating/identifying {plotDirectory_}...Using CWD."
            )
            plotDirectory_ = os.getcwd()

    return core.plotParams(
        plotPrefix=plotPrefix_,
        plotStateEstimatesHistogram=plotStateEstimatesHistogram_,
        plotResidualsHistogram=plotResidualsHistogram_,
        plotStateStdHistogram=plotStateStdHistogram_,
        plotHeightInches=plotHeightInches_,
        plotWidthInches=plotWidthInches_,
        plotDPI=plotDPI_,
        plotDirectory=plotDirectory_,
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
    matchingExcludeRegionsFileDefault: Optional[str] = (
        genomeParams.blacklistFile
    )

    experimentName = _cfgGet(
        configData, "experimentName", "consenrichExperiment"
    )

    processArgs = core.processParams(
        deltaF=_cfgGet(configData, "processParams.deltaF", -1.0),
        minQ=_cfgGet(configData, "processParams.minQ", -1.0),
        maxQ=_cfgGet(configData, "processParams.maxQ", 1000.0),
        offDiagQ=_cfgGet(
            configData, "processParams.offDiagQ", 0.0,
        ),
        dStatAlpha=_cfgGet(
            configData,
            "processParams.dStatAlpha",
            3.0,
        ),
        dStatd=_cfgGet(configData, "processParams.dStatd", 1.0),
        dStatPC=_cfgGet(configData, "processParams.dStatPC", 1.0),
        dStatUseMean=_cfgGet(
            configData,
            "processParams.dStatUseMean",
            True,
        ),
        scaleResidualsByP11=_cfgGet(
            configData,
            "processParams.scaleResidualsByP11",
            True,
        ),
    )

    plotArgs = getPlotArgs(config_path, experimentName)

    observationArgs = core.observationParams(
        minR=_cfgGet(configData, "observationParams.minR", -1.0),
        maxR=_cfgGet(configData, "observationParams.maxR", 1000.0),
        numNearest=_cfgGet(
            configData,
            "observationParams.numNearest",
            50,
        ),
        localWeight=_cfgGet(
            configData,
            "observationParams.localWeight",
            0.25,
        ),
        refitWeight=_cfgGet(
            configData,
            "observationParams.refitWeight",
            10.0,
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
        0,
    )
    oneReadPerBin = _cfgGet(configData, "samParams.oneReadPerBin", 0)
    chunkSize = _cfgGet(configData, "samParams.chunkSize", 1_000_000)
    offsetStr = _cfgGet(configData, "samParams.offsetStr", "0,0")
    maxInsertSize = _cfgGet(
        configData,
        "samParams.maxInsertSize",
        1000,
    )

    pairedEndDefault = (
        1
        if inputParams.pairedEnd is not None
        and int(inputParams.pairedEnd) > 0
        else 0
    )
    inferFragmentDefault = (
        1
        if inputParams.pairedEnd is not None
        and int(inputParams.pairedEnd) == 0
        else 0
    )

    samArgs = core.samParams(
        samThreads=samThreads,
        samFlagExclude=samFlagExclude,
        oneReadPerBin=oneReadPerBin,
        chunkSize=chunkSize,
        offsetStr=offsetStr,
        maxInsertSize=maxInsertSize,
        pairedEndMode=_cfgGet(
            configData,
            "samParams.pairedEndMode",
            pairedEndDefault,
        ),
        inferFragmentLength=_cfgGet(
            configData,
            "samParams.inferFragmentLength",
            inferFragmentDefault,
        ),
        countEndsOnly=_cfgGet(
            configData, "samParams.countEndsOnly", False
        ),
        minMappingQuality=minMappingQuality,
        minTemplateLength=_cfgGet(
            configData,
            "samParams.minTemplateLength",
            -1,
        ),
    )

    matchingArgs = core.matchingParams(
        templateNames=_cfgGet(
            configData, "matchingParams.templateNames", []
        ),
        cascadeLevels=_cfgGet(
            configData, "matchingParams.cascadeLevels", []
        ),
        iters=_cfgGet(configData, "matchingParams.iters", 25_000),
        alpha=_cfgGet(configData, "matchingParams.alpha", 0.05),
        minMatchLengthBP=_cfgGet(
            configData,
            "matchingParams.minMatchLengthBP",
            -1,
        ),
        maxNumMatches=_cfgGet(
            configData,
            "matchingParams.maxNumMatches",
            100_000,
        ),
        minSignalAtMaxima=_cfgGet(
            configData,
            "matchingParams.minSignalAtMaxima",
            "q:0.50",
        ),
        merge=_cfgGet(configData, "matchingParams.merge", True),
        mergeGapBP=_cfgGet(
            configData,
            "matchingParams.mergeGapBP",
            -1,
        ),
        useScalingFunction=_cfgGet(
            configData,
            "matchingParams.useScalingFunction",
            True,
        ),
        excludeRegionsBedFile=_cfgGet(
            configData,
            "matchingParams.excludeRegionsBedFile",
            matchingExcludeRegionsFileDefault,
        ),
        randSeed=_cfgGet(configData, "matchingParams.randSeed", 42),
        penalizeBy=_cfgGet(
            configData, "matchingParams.penalizeBy", None
        ),
        eps=_cfgGet(configData, "matchingParams.eps", 1.0e-2),
        autoLengthQuantile=_cfgGet(
            configData,
            "matchingParams.autoLengthQuantile",
            0.50,
        ),
        methodFDR=_cfgGet(
            configData,
            "matchingParams.methodFDR",
            None,
        ),
        massQuantileCutoff=_cfgGet(
            configData,
            "matchingParams.massQuantileCutoff",
            0.25,
        ),
    )

    return {
        "experimentName": experimentName,
        "genomeArgs": genomeParams,
        "inputArgs": inputParams,
        "outputArgs": outputParams,
        "countingArgs": countingParams,
        "processArgs": processArgs,
        "plotArgs": plotArgs,
        "observationArgs": observationArgs,
        "stateArgs": stateParams,
        "samArgs": samArgs,
        "matchingArgs": matchingArgs,
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

    logger.info(
        "Attempting to generate bigWig files from bedGraph format..."
    )
    try:
        path_ = shutil.which("bedGraphToBigWig")
    except Exception as e:
        logger.warning(f"\n{warningMessage}\n")
        return
    if path_ is None or len(path_) == 0:
        logger.warning(f"\n{warningMessage}\n")
        return
    logger.info(f"Using bedGraphToBigWig from {path_}")
    for suffix in suffixes:
        bedgraph = (
            f"consenrichOutput_{experimentName}_{suffix}.bedGraph"
        )
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
        bigwig = f"{experimentName}_consenrich_{suffix}.bw"
        logger.info(f"Start: {bedgraph} --> {bigwig}...")
        try:
            subprocess.run(
                [path_, bedgraph, chromSizesFile, bigwig], check=True
            )
        except Exception as e:
            logger.warning(
                f"bedGraph-->bigWig conversion with\n\n\t`bedGraphToBigWig {bedgraph} {chromSizesFile} {bigwig}`\nraised: \n{e}\n\n"
            )
            continue
        if os.path.exists(bigwig) and os.path.getsize(bigwig) > 100:
            logger.info(
                f"Finished: converted {bedgraph} to {bigwig}."
            )


def main():
    parser = argparse.ArgumentParser(description="Consenrich CLI")
    parser.add_argument(
        "--config",
        type=str,
        dest="config",
        help="Path to a YAML config file with parameters + arguments defined in `consenrich.core`",
    )

    # --- Matching-specific command-line arguments ---
    parser.add_argument(
        "--match-bedGraph",
        type=str,
        dest="matchBedGraph",
        help="Path to a bedGraph file of Consenrich estimates to match templates against.\
            If provided, *only* the matching algorithm is run (no other processing). Note that \
            some features in `consenrich.matching` may not be supported through this CLI interface.",
    )
    parser.add_argument(
        "--match-template",
        nargs="+",
        type=str,
        help="List of template names to use in matching. See PyWavelets discrete wavelet families: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#discrete-wavelets. \
            Needs to match `--match-level` in length",
        dest="matchTemplate",
    )

    parser.add_argument(
        "--match-level",
        nargs="+",
        type=int,
        help="List of cascade levels to use in matching. Needs to match `--match-template` in length",
        dest="matchLevel",
    )

    parser.add_argument(
        "--match-alpha",
        type=float,
        default=0.05,
        dest="matchAlpha",
        help="Cutoff qualifying candidate matches as significant (FDR-adjusted p-value < alpha).",
    )
    parser.add_argument(
        "--match-min-length",
        type=int,
        default=-1,
        dest="matchMinMatchLengthBP",
        help="Minimum length (bp) qualifying candidate matches. Set to -1 for auto calculation from data",
    )
    parser.add_argument(
        "--match-iters",
        type=int,
        default=25_000,
        dest="matchIters",
        help="Number of sampled blocks for estimating null distribution of match scores (cross correlations with templates).",
    )
    parser.add_argument(
        "--match-min-signal",
        type=str,
        default="q:0.50",
        dest="matchMinSignalAtMaxima",
        help="Minimum signal at local maxima in the response sequence that qualifies candidate matches\
            Can be an absolute value (e.g., `50.0`) or a quantile (e.g., `q:0.75` for 75th percentile).",
    )
    parser.add_argument(
        "--match-max-matches",
        type=int,
        default=1000000,
        dest="matchMaxNumMatches",
    )
    parser.add_argument(
        "--match-merge-gap",
        type=int,
        default=-1,
        dest="matchMergeGapBP",
        help="Maximum gap (bp) between candidate matches to merge into a single match.\
            Set to -1 for auto calculation from data.",
    )
    parser.add_argument(
        "--match-use-wavelet",
        action="store_true",
        dest="matchUseWavelet",
        help="If set, use the wavelet function at the given level rather than scaling function.",
    )
    parser.add_argument(
        "--match-seed", type=int, default=42, dest="matchRandSeed"
    )
    parser.add_argument(
        "--match-exclude-bed",
        type=str,
        default=None,
        dest="matchExcludeBed",
    )
    parser.add_argument(
        "--match-auto-length-quantile",
        type=float,
        default=0.90,
        dest="matchAutoLengthQuantile",
        help="Cutoff in standardized values to use when auto-calculating minimum match length and merge gap.",
    )
    parser.add_argument(
        "--match-method-fdr",
        type=str,
        default=None,
        dest="matchMethodFDR",
        help="Method for multiple hypothesis correction of p-values. (bh, by)",
    )
    parser.add_argument(
        "--match-mass-quantile-cutoff",
        type=float,
        default=0.25,
        dest="matchMassQuantileCutoff",
        help="Quantile cutoff for filtering initial (unmerged) matches based on their 'mass' (average signal value * length). Set to < 0 to disable",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="If set, logs config"
    )
    parser.add_argument(
        "--verbose2",
        action="store_true",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Consenrich version {__version__}",
    )
    args = parser.parse_args()

    if args.matchBedGraph:
        if not os.path.exists(args.matchBedGraph):
            raise FileNotFoundError(
                f"bedGraph file {args.matchBedGraph} couldn't be found."
            )
        logger.info(
            f"Running matching algorithm using bedGraph file {args.matchBedGraph}..."
        )

        outName = matching.runMatchingAlgorithm(
            args.matchBedGraph,
            args.matchTemplate,
            args.matchLevel,
            alpha=args.matchAlpha,
            minMatchLengthBP=args.matchMinMatchLengthBP,
            iters=args.matchIters,
            minSignalAtMaxima=args.matchMinSignalAtMaxima,
            maxNumMatches=args.matchMaxNumMatches,
            useScalingFunction=(not args.matchUseWavelet),
            mergeGapBP=args.matchMergeGapBP,
            excludeRegionsBedFile=args.matchExcludeBed,
            autoLengthQuantile=args.matchAutoLengthQuantile,
            methodFDR=args.matchMethodFDR.lower()
            if args.matchMethodFDR
            else None,
            randSeed=args.matchRandSeed,
            merge=True,  # always merge for CLI use -- either way, both files produced
            massQuantileCutoff=args.matchMassQuantileCutoff,
        )
        logger.info(f"Finished matching. Written to {outName}")
        sys.exit(0)

    if not args.config:
        logger.info(
            "No config file provided, run with `--config <path_to_config.yaml>`"
        )
        logger.info(
            "See documentation: https://nolan-h-hamilton.github.io/Consenrich/"
        )
        sys.exit(1)

    if not os.path.exists(args.config):
        logger.info(f"Config file {args.config} does not exist.")
        logger.info(
            "See documentation: https://nolan-h-hamilton.github.io/Consenrich/"
        )
        sys.exit(1)

    config = readConfig(args.config)
    experimentName = config["experimentName"]
    genomeArgs = config["genomeArgs"]
    inputArgs = config["inputArgs"]
    outputArgs = config["outputArgs"]
    countingArgs = config["countingArgs"]
    processArgs = config["processArgs"]
    observationArgs = config["observationArgs"]
    stateArgs = config["stateArgs"]
    samArgs = config["samArgs"]
    matchingArgs = config["matchingArgs"]
    plotArgs = config["plotArgs"]
    bamFiles = inputArgs.bamFiles
    bamFilesControl = inputArgs.bamFilesControl
    numSamples = len(bamFiles)
    numNearest = observationArgs.numNearest
    stepSize = countingArgs.stepSize
    excludeForNorm = genomeArgs.excludeForNorm
    chromSizes = genomeArgs.chromSizesFile
    initialTreatmentScaleFactors = []
    minMatchLengthBP_: Optional[int] = matchingArgs.minMatchLengthBP
    deltaF_ = processArgs.deltaF
    minR_ = observationArgs.minR
    maxR_ = observationArgs.maxR
    minQ_ = processArgs.minQ
    maxQ_ = processArgs.maxQ
    offDiagQ_ = processArgs.offDiagQ
    muncEps: float = 10e-2

    if args.verbose2:
        args.verbose = True

    if args.verbose:
        try:
            logger.info(f"Initial Configuration ({__version__})\n")
            config_truncated = {
                k: v
                for k, v in config.items()
                if k
                not in ["inputArgs", "genomeArgs", "countingArgs"]
            }
            config_truncated["experimentName"] = experimentName
            config_truncated["inputArgs"] = inputArgs
            config_truncated["outputArgs"] = outputArgs
            config_truncated["genomeArgs"] = genomeArgs
            config_truncated["countingArgs"] = countingArgs
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

    controlsPresent = checkControlsPresent(inputArgs)
    if args.verbose:
        logger.info(f"controlsPresent: {controlsPresent}")
    readLengthsBamFiles = getReadLengths(
        inputArgs, countingArgs, samArgs
    )
    effectiveGenomeSizes = getEffectiveGenomeSizes(
        genomeArgs, readLengthsBamFiles
    )

    matchingEnabled = checkMatchingEnabled(matchingArgs)
    if args.verbose:
        logger.info(f"matchingEnabled: {matchingEnabled}")
    scaleFactors = countingArgs.scaleFactors
    scaleFactorsControl = countingArgs.scaleFactorsControl

    fragmentLengthsTreatment: List[int] = []
    fragmentLengthsControl: List[int] = []

    if countingArgs.fragmentLengths is not None:
        fragmentLengthsTreatment = list(countingArgs.fragmentLengths)
    else:
        for bamFile in bamFiles:
            fragmentLengthsTreatment.append(
                cconsenrich.cgetFragmentLength(
                    bamFile,
                    samThreads=samArgs.samThreads,
                    samFlagExclude=samArgs.samFlagExclude,
                    maxInsertSize=samArgs.maxInsertSize,
                )
            )
            logger.info(
                f"Estimated fragment length for {bamFile}: {fragmentLengthsTreatment[-1]}"
            )
    if controlsPresent:
        readLengthsControlBamFiles = [
            core.getReadLength(
                bamFile,
                100,
                1000,
                samArgs.samThreads,
                samArgs.samFlagExclude,
            )
            for bamFile in bamFilesControl
        ]
        effectiveGenomeSizesControl = [
            constants.getEffectiveGenomeSize(
                genomeArgs.genomeName, readLength
            )
            for readLength in readLengthsControlBamFiles
        ]

        if countingArgs.fragmentLengthsControl is not None:
            fragmentLengthsControl = list(
                countingArgs.fragmentLengthsControl
            )
        elif not countingArgs.useTreatmentFragmentLengths:
            for bamFile in bamFilesControl:
                fragmentLengthsControl.append(
                    cconsenrich.cgetFragmentLength(
                        bamFile,
                        samThreads=samArgs.samThreads,
                        samFlagExclude=samArgs.samFlagExclude,
                        maxInsertSize=samArgs.maxInsertSize,
                    )
                )
                logger.info(
                    f"Estimated fragment length for {bamFile}: {fragmentLengthsControl[-1]}"
                )
        if countingArgs.useTreatmentFragmentLengths:
            logger.info(
                "`countingParams.useTreatmentFragmentLengths=True`"
                "`\n\t--> using treatment fraglens for control samples, too"
            )
            fragmentLengthsTreatment, fragmentLengthsControl = (
                _resolveFragmentLengthPairs(
                    fragmentLengthsTreatment, fragmentLengthsControl
                )
            )

        if (
            scaleFactors is not None
            and scaleFactorsControl is not None
        ):
            treatScaleFactors = scaleFactors
            controlScaleFactors = scaleFactorsControl
            # still make sure this is accessible
            initialTreatmentScaleFactors = [1.0] * len(bamFiles)
        else:
            try:
                initialTreatmentScaleFactors = [
                    detrorm.getScaleFactor1x(
                        bamFile,
                        effectiveGenomeSize,
                        readLength,
                        excludeForNorm,
                        genomeArgs.chromSizesFile,
                        samArgs.samThreads,
                    )
                    for bamFile, effectiveGenomeSize, readLength in zip(
                        bamFiles,
                        effectiveGenomeSizes,
                        fragmentLengthsTreatment,
                    )
                ]
            except Exception:
                initialTreatmentScaleFactors = [1.0] * len(bamFiles)

            pairScalingFactors = [
                detrorm.getPairScaleFactors(
                    bamFileA,
                    bamFileB,
                    effectiveGenomeSizeA,
                    effectiveGenomeSizeB,
                    readLengthA,
                    readLengthB,
                    excludeForNorm,
                    chromSizes,
                    samArgs.samThreads,
                    stepSize,
                    True,
                    normMethod=countingArgs.normMethod,
                )
                for bamFileA, bamFileB, effectiveGenomeSizeA, effectiveGenomeSizeB, readLengthA, readLengthB in zip(
                    bamFiles,
                    bamFilesControl,
                    effectiveGenomeSizes,
                    effectiveGenomeSizesControl,
                    fragmentLengthsTreatment,
                    fragmentLengthsControl,
                )
            ]

            treatScaleFactors = []
            controlScaleFactors = []
            for scaleFactorA, scaleFactorB in pairScalingFactors:
                treatScaleFactors.append(scaleFactorA)
                controlScaleFactors.append(scaleFactorB)

    else:
        treatScaleFactors = scaleFactors
        controlScaleFactors = scaleFactorsControl

    if scaleFactors is None and not controlsPresent:
        if countingArgs.normMethod.upper() == "RPKM":
            scaleFactors = [
                detrorm.getScaleFactorPerMillion(
                    bamFile,
                    excludeForNorm,
                    stepSize,
                )
                for bamFile in bamFiles
            ]
        else:
            scaleFactors = [
                detrorm.getScaleFactor1x(
                    bamFile,
                    effectiveGenomeSize,
                    readLength,
                    excludeForNorm,
                    genomeArgs.chromSizesFile,
                    samArgs.samThreads,
                )
                for bamFile, effectiveGenomeSize, readLength in zip(
                    bamFiles,
                    effectiveGenomeSizes,
                    fragmentLengthsTreatment,
                )
            ]
    chromSizesDict = misc_util.getChromSizesDict(
        genomeArgs.chromSizesFile,
        excludeChroms=genomeArgs.excludeChroms,
    )
    chromosomes = genomeArgs.chromosomes

    for c_, chromosome in enumerate(chromosomes):
        chromosomeStart, chromosomeEnd = core.getChromRangesJoint(
            bamFiles,
            chromosome,
            chromSizesDict[chromosome],
            samArgs.samThreads,
            samArgs.samFlagExclude,
        )
        chromosomeStart = max(
            0, (chromosomeStart - (chromosomeStart % stepSize))
        )
        chromosomeEnd = max(
            0, (chromosomeEnd - (chromosomeEnd % stepSize))
        )
        numIntervals = (
            ((chromosomeEnd - chromosomeStart) + stepSize) - 1
        ) // stepSize
        intervals = np.arange(
            chromosomeStart, chromosomeEnd, stepSize
        )

        if c_ == 0 and deltaF_ < 0:
            logger.info(
                f"`processParams.deltaF < 0` --> calling core.autoDeltaF()..."
            )
            deltaF_ = core.autoDeltaF(
                bamFiles,
                stepSize,
                fragmentLengths=fragmentLengthsTreatment,
            )

        chromMat: np.ndarray = np.empty(
            (numSamples, numIntervals), dtype=np.float32
        )
        if controlsPresent:
            j_: int = 0
            for bamA, bamB in zip(bamFiles, bamFilesControl):
                logger.info(
                    f"Counting (trt,ctrl) for {chromosome}: ({bamA}, {bamB})"
                )
                pairMatrix: np.ndarray = core.readBamSegments(
                    [bamA, bamB],
                    chromosome,
                    chromosomeStart,
                    chromosomeEnd,
                    stepSize,
                    [
                        readLengthsBamFiles[j_],
                        readLengthsControlBamFiles[j_],
                    ],
                    [treatScaleFactors[j_], controlScaleFactors[j_]],
                    samArgs.oneReadPerBin,
                    samArgs.samThreads,
                    samArgs.samFlagExclude,
                    offsetStr=samArgs.offsetStr,
                    maxInsertSize=samArgs.maxInsertSize,
                    pairedEndMode=samArgs.pairedEndMode,
                    inferFragmentLength=samArgs.inferFragmentLength,
                    countEndsOnly=samArgs.countEndsOnly,
                    minMappingQuality=samArgs.minMappingQuality,
                    minTemplateLength=samArgs.minTemplateLength,
                    fragmentLengths=[
                        fragmentLengthsTreatment[j_],
                        fragmentLengthsControl[j_],
                    ],
                )
                chromMat[j_, :] = pairMatrix[0, :] - pairMatrix[1, :]
                j_ += 1
        else:
            chromMat = core.readBamSegments(
                bamFiles,
                chromosome,
                chromosomeStart,
                chromosomeEnd,
                stepSize,
                readLengthsBamFiles,
                scaleFactors,
                samArgs.oneReadPerBin,
                samArgs.samThreads,
                samArgs.samFlagExclude,
                offsetStr=samArgs.offsetStr,
                maxInsertSize=samArgs.maxInsertSize,
                pairedEndMode=samArgs.pairedEndMode,
                inferFragmentLength=samArgs.inferFragmentLength,
                countEndsOnly=samArgs.countEndsOnly,
                minMappingQuality=samArgs.minMappingQuality,
                minTemplateLength=samArgs.minTemplateLength,
                fragmentLengths=fragmentLengthsTreatment,
            )
        sparseMap = None
        if genomeArgs.sparseBedFile and not genomeArgs.sparseBedFile.lower() not in ['none', 'false']:
            sparseMap = core.getSparseMap(
                chromosome,
                intervals,
                numNearest,
                genomeArgs.sparseBedFile,
            )

        # negative --> data-based
        if observationArgs.minR < 0.0 or observationArgs.maxR < 0.0:
            minR_ = 0.0
            maxR_ = 1e4
        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            minQ_ = 0.0
            maxQ_ = 1e4

        muncMat = np.empty_like(chromMat, dtype=np.float32)
        for j in range(numSamples):
            chromMat[j, :] = cconsenrich.carsinhRatio(
                chromMat[j, :],
                countingArgs.backgroundWindowSizeBP,
            )

            # compute munc track for each sample independently
            muncMat[j, :] = core.getMuncTrack(
                chromosome,
                intervals,
                chromMat[j, :],
                stepSize,
                minR_,
                maxR_,
                sparseMap,
                localWeight=observationArgs.localWeight,
                randomSeed=42 + j,
                textPlotMeanVarianceTrend=args.verbose2,
                refitWeight=observationArgs.refitWeight,
            )

        if observationArgs.minR < 0.0 or observationArgs.maxR < 0.0:
            kappa = 100.0 # conditioning given f32
            minR_ = np.float32(
                np.quantile(muncMat[muncMat > 0], 0.05)
            )

            colMax = muncMat.max(axis=0).astype(np.float32)
            colMin = np.maximum(
                muncMat.min(axis=0), (colMax / kappa)
            ).astype(np.float32)

            np.clip(muncMat, colMin, colMax, out=muncMat)
            muncMat += muncEps
            muncMat = muncMat.astype(np.float32, copy=False)
        minQ_ = processArgs.minQ
        maxQ_ = processArgs.maxQ

        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            if minR_ is None:
                minR_ = np.float32(
                    np.quantile(muncMat[muncMat > 0], 0.05)
                )
            # heuristic to keep us PD, on scale, growing with model's deltaF
            autoMinQ = (minR_*deltaF_)/numSamples + 2.0*offDiagQ_ + 1.0e-3

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


        logger.info(f">>>Running consenrich: {chromosome}<<<")
        x, P, y = core.runConsenrich(
            chromMat,
            muncMat,
            deltaF_,
            minQ_,
            maxQ_,
            offDiagQ_,
            processArgs.dStatAlpha,
            processArgs.dStatd,
            processArgs.dStatPC,
            processArgs.dStatUseMean,
            stateArgs.stateInit,
            stateArgs.stateCovarInit,
            stateArgs.boundState,
            stateArgs.stateLowerBound,
            stateArgs.stateUpperBound,
            samArgs.chunkSize,
            progressIter=25_000,
        )
        logger.info("Done.")
        logger.info(
            f"minQ={minQ_}, minR={minR_}"
        )
        x_ = core.getPrimaryState(x, stateLowerBound = stateArgs.stateLowerBound, stateUpperBound = stateArgs.stateUpperBound, boundState = stateArgs.boundState)
        y_ = core.getPrecisionWeightedResidual(y, muncMat, P)

        if plotArgs.plotStateEstimatesHistogram:
            core.plotStateEstimatesHistogram(
                chromosome,
                plotArgs.plotPrefix,
                x_,
                plotDirectory=plotArgs.plotDirectory,
            )

        if plotArgs.plotResidualsHistogram:
            core.plotResidualsHistogram(
                chromosome,
                plotArgs.plotPrefix,
                y,
                plotDirectory=plotArgs.plotDirectory,
            )

        if plotArgs.plotStateStdHistogram:
            core.plotStateStdHistogram(
                chromosome,
                plotArgs.plotPrefix,
                np.sqrt(P[:, 0, 0]),
                plotDirectory=plotArgs.plotDirectory,
            )

        df = pd.DataFrame(
            {
                "Chromosome": chromosome,
                "Start": intervals,
                "End": intervals + stepSize,
                "State": x_,
            }
        )

        if outputArgs.writeResiduals:
            df["Res"] = y_.astype(np.float32)  # FFR: cast necessary?
        if outputArgs.writeMuncTrace:
            munc_std = np.sqrt(
                np.mean(muncMat.astype(np.float64), axis=0)
            ).astype(np.float32)
            df["Munc"] = munc_std
        if outputArgs.writeStateStd:
            df["StateStd"] = np.sqrt(P[:, 0, 0]).astype(np.float32)
        cols_ = ["Chromosome", "Start", "End", "State"]
        if outputArgs.writeResiduals:
            cols_.append("Res")
        if outputArgs.writeMuncTrace:
            cols_.append("Munc")
        if outputArgs.writeStateStd:
            cols_.append("StateStd")
        df = df[cols_]
        suffixes = ["state"]
        if outputArgs.writeResiduals:
            suffixes.append("residuals")
        if outputArgs.writeMuncTrace:
            suffixes.append("muncTraces")
        if outputArgs.writeStateStd:
            suffixes.append("stdDevs")

        if (c_ == 0 and len(chromosomes) > 1) or (
            len(chromosomes) == 1
        ):
            for file_ in os.listdir("."):
                if file_.startswith(
                    f"consenrichOutput_{experimentName}"
                ) and (
                    file_.endswith(".bedGraph")
                    or file_.endswith(".narrowPeak")
                ):
                    logger.warning(f"Overwriting: {file_}")
                    os.remove(file_)

        for col, suffix in zip(cols_[3:], suffixes):
            logger.info(
                f"{chromosome}: writing/appending to: consenrichOutput_{experimentName}_{suffix}.bedGraph"
            )
            df[["Chromosome", "Start", "End", col]].to_csv(
                f"consenrichOutput_{experimentName}_{suffix}.bedGraph",
                sep="\t",
                header=False,
                index=False,
                mode="a",
                float_format="%.3f",
                lineterminator="\n",
            )

    logger.info("Finished: output in human-readable format")

    if outputArgs.convertToBigWig:
        convertBedGraphToBigWig(
            experimentName,
            genomeArgs.chromSizesFile,
            suffixes=suffixes,
        )

    if matchingEnabled:
        try:
            weightsBedGraph: str | None = None
            logger.info("Running matching algorithm...")
            if matchingArgs.penalizeBy is not None:
                if matchingArgs.penalizeBy.lower() in [
                    "stateuncertainty",
                    "statestddev",
                    "statestd",
                    "p11",
                ]:
                    weightsBedGraph = f"consenrichOutput_{experimentName}_stdDevs.bedGraph"
                elif matchingArgs.penalizeBy.lower() in [
                    "munc",
                    "munctrace",
                    "avgmunctrace",
                ]:
                    weightsBedGraph = f"consenrichOutput_{experimentName}_muncTraces.bedGraph"
                elif matchingArgs.penalizeBy.lower() == "none":
                    weightsBedGraph = None
                else:
                    weightsBedGraph = None

            outName = matching.runMatchingAlgorithm(
                f"consenrichOutput_{experimentName}_state.bedGraph",
                matchingArgs.templateNames,
                matchingArgs.cascadeLevels,
                matchingArgs.iters,
                alpha=matchingArgs.alpha,
                minMatchLengthBP=minMatchLengthBP_,
                maxNumMatches=matchingArgs.maxNumMatches,
                minSignalAtMaxima=matchingArgs.minSignalAtMaxima,
                useScalingFunction=matchingArgs.useScalingFunction,
                mergeGapBP=matchingArgs.mergeGapBP,
                excludeRegionsBedFile=matchingArgs.excludeRegionsBedFile,
                randSeed=matchingArgs.randSeed,
                weightsBedGraph=weightsBedGraph,
                eps=matchingArgs.eps,
                autoLengthQuantile=matchingArgs.autoLengthQuantile,
                methodFDR=matchingArgs.methodFDR.lower()
                if matchingArgs.methodFDR is not None
                else None,
                merge=matchingArgs.merge,
                massQuantileCutoff=matchingArgs.massQuantileCutoff,
            )

            logger.info(f"Finished matching. Written to {outName}")
        except Exception as ex_:
            logger.warning(
                f"Matching algorithm raised an exception:\n\n\t{ex_}\n"
                f"Skipping matching step...try running post-hoc via `consenrich --match-bedGraph <bedGraphFile>`\n"
                f"\tSee ``consenrich -h`` for more details.\n"
            )


if __name__ == "__main__":
    main()

