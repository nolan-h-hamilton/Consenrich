#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import pprint
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pysam
import yaml

import consenrich.core as core
import consenrich.misc_util as misc_util
import consenrich.constants as constants
import consenrich.detrorm as detrorm
import consenrich.matching as matching


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _listOrEmpty(list_):
    if list_ is None:
        return []
    return list_


def _getMinR(cfg, numBams: int) -> float:
    try:
        raw = cfg.get("observationParams.minR", None)
        return float(raw) if raw is not None else (1 / numBams) + 1e-4
    except (TypeError, ValueError, KeyError):
        fallBackMinR: float = 1.0e-2
        logger.warning(
            f"Invalid or missing 'observationParams.minR' in config. Using `{fallBackMinR}`."
        )
        return fallBackMinR


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

    return [
        core.getReadLength(
            bamFile,
            countingArgs.numReads,
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
    def _expandWildCards(bamList) -> List[str]:
        expanded = []
        for entry in bamList:
            if "*" in entry or "?" in entry or "[" in entry:
                matched = glob.glob(entry)
                expanded.extend(matched)
            else:
                expanded.append(entry)
        return expanded

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    bamFilesRaw = config.get("inputParams.bamFiles", [])
    bamFilesControlRaw = config.get("inputParams.bamFilesControl", [])
    bamFiles = _expandWildCards(bamFilesRaw)
    bamFilesControl = _expandWildCards(bamFilesControlRaw)
    if len(bamFiles) == 0:
        raise ValueError("No BAM files provided in the configuration.")
    if (
        len(bamFilesControl) > 0
        and len(bamFilesControl) != len(bamFiles)
        and len(bamFilesControl) != 1
    ):
        raise ValueError(
            "Number of control BAM files must be 0, 1, or the same as number of treatment files"
        )
    if len(bamFilesControl) == 1:
        # If there are multiple bamFiles, but 1 control, control is applied for all treatment files
        logger.info(
            f"Only one control given: Using {bamFilesControl[0]} for all treatment files."
        )
        bamFilesControl = bamFilesControl * len(bamFiles)

    if not bamFiles or not isinstance(bamFiles, list) or len(bamFiles) == 0:
        raise ValueError("No BAM files found")

    for i, bamFile in enumerate(bamFiles):
        misc_util.checkBamFile(bamFile)

    if bamFilesControl:
        for i, bamFile in enumerate(bamFilesControl):
            misc_util.checkBamFile(bamFile)

    return core.inputParams(bamFiles=bamFiles, bamFilesControl=bamFilesControl)


def getGenomeArgs(config_path: str) -> core.genomeParams:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    genomeName = config.get("genomeParams.name", None)
    genome = constants.resolveGenomeName(genomeName)
    chromSizesFile: Optional[str] = None
    blacklistFile: Optional[str] = None
    sparseBedFile: Optional[str] = None
    chromosomes: Optional[List[str]] = None
    excludeChroms: List[str] = config.get("genomeParams.excludeChroms", [])
    excludeForNorm: List[str] = config.get("genomeParams.excludeForNorm", [])
    if genome:
        chromSizesFile = constants.getGenomeResourceFile(genome, "sizes")
        blacklistFile = constants.getGenomeResourceFile(genome, "blacklist")
        sparseBedFile = constants.getGenomeResourceFile(genome, "sparse")
    if config.get("genomeParams.chromSizesFile", None):
        chromSizesFile = config["genomeParams.chromSizesFile"]
    if config.get("genomeParams.blacklistFile", None):
        blacklistFile = config["genomeParams.blacklistFile"]
    if config.get("genomeParams.sparseBedFile", None):
        sparseBedFile = config["genomeParams.sparseBedFile"]
    if not chromSizesFile or not os.path.exists(chromSizesFile):
        raise FileNotFoundError(
            f"Chromosome sizes file {chromSizesFile} does not exist."
        )
    if config.get("genomeParams.chromosomes", None):
        chromosomes = config["genomeParams.chromosomes"]
    else:
        if chromSizesFile:
            chromosomes = list(
                pd.read_csv(
                    chromSizesFile,
                    sep="\t",
                    header=None,
                    names=["chrom", "size"],
                )["chrom"]
            )
        else:
            raise ValueError(
                "No chromosomes provided in the configuration and no chromosome sizes file specified."
            )
    chromosomes = [chrom.strip() for chrom in chromosomes if chrom.strip()]
    if excludeChroms:
        chromosomes = [
            chrom for chrom in chromosomes if chrom not in excludeChroms
        ]
    if not chromosomes:
        raise ValueError(
            "No valid chromosomes found after excluding specified chromosomes."
        )
    return core.genomeParams(
        genomeName=genome,
        chromSizesFile=chromSizesFile,
        blacklistFile=blacklistFile,
        sparseBedFile=sparseBedFile,
        chromosomes=chromosomes,
        excludeChroms=excludeChroms,
        excludeForNorm=excludeForNorm,
    )


def getCountingArgs(config_path: str) -> core.countingParams:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    stepSize = config.get("countingParams.stepSize", 25)
    scaleDown = config.get("countingParams.scaleDown", True)
    scaleFactors = config.get("countingParams.scaleFactors", None)
    numReads = config.get("countingParams.numReads", 100)
    scaleFactorsControl = config.get("countingParams.scaleFactorsControl", None)
    applyAsinh = config.get("countingParams.applyAsinh", False)
    applyLog = config.get("countingParams.applyLog", False)
    if applyAsinh and applyLog:
        applyAsinh = True
        applyLog = False
        logger.warning(
            "Both `applyAsinh` and `applyLog` are set. Overriding `applyLog` to False."
        )
    rescaleToTreatmentCoverage = config.get(
        "countingParams.rescaleToTreatmentCoverage", True
    )
    if scaleFactors is not None and not isinstance(scaleFactors, list):
        raise ValueError("`scaleFactors` should be a list of floats.")
    if scaleFactorsControl is not None and not isinstance(
        scaleFactorsControl, list
    ):
        raise ValueError("`scaleFactorsControl` should be a list of floats.")
    if (
        scaleFactors is not None
        and scaleFactorsControl is not None
        and len(scaleFactors) != len(scaleFactorsControl)
    ):
        if len(scaleFactorsControl) == 1:
            scaleFactorsControl = scaleFactorsControl * len(scaleFactors)
        else:
            raise ValueError(
                "control and treatment scale factors: must be equal length or 1 control"
            )
    return core.countingParams(
        stepSize=stepSize,
        scaleDown=scaleDown,
        scaleFactors=scaleFactors,
        scaleFactorsControl=scaleFactorsControl,
        numReads=numReads,
        applyAsinh=applyAsinh,
        applyLog=applyLog,
        rescaleToTreatmentCoverage=rescaleToTreatmentCoverage,
    )


def readConfig(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    inputParams = getInputArgs(config_path)
    genomeParams = getGenomeArgs(config_path)
    countingParams = getCountingArgs(config_path)
    minR_default = _getMinR(config, len(inputParams.bamFiles))
    matchingExcludeRegionsBedFile_default: Optional[str] = (
        genomeParams.blacklistFile
    )
    return {
        "experimentName": config.get("experimentName", "consenrichExperiment"),
        "genomeArgs": genomeParams,
        "inputArgs": inputParams,
        "countingArgs": countingParams,
        "processArgs": core.processParams(
            deltaF=config.get("processParams.deltaF", 0.5),
            minQ=config.get("processParams.minQ", 0.25),
            maxQ=config.get("processParams.maxQ", 500.0),
            offDiagQ=config.get("processParams.offDiagQ", 0.0),
            dStatAlpha=config.get("processParams.dStatAlpha", 3.0),
            dStatd=config.get("processParams.dStatd", 10.0),
            dStatPC=config.get("processParams.dStatPC", 2.0),
        ),
        "observationArgs": core.observationParams(
            minR=minR_default,
            maxR=config.get("observationParams.maxR", 500.0),
            useALV=config.get("observationParams.useALV", False),
            useConstantNoiseLevel=config.get(
                "observationParams.useConstantNoiseLevel", False
            ),
            noGlobal=config.get("observationParams.noGlobal", False),
            numNearest=config.get("observationParams.numNearest", 25),
            localWeight=config.get("observationParams.localWeight", 0.333),
            globalWeight=config.get("observationParams.globalWeight", 0.667),
            approximationWindowLengthBP=config.get(
                "observationParams.approximationWindowLengthBP", 10000
            ),
            lowPassWindowLengthBP=config.get(
                "observationParams.lowPassWindowLengthBP", 20000
            ),
            lowPassFilterType=config.get(
                "observationParams.lowPassFilterType", "median"
            ),
            returnCenter=config.get("observationParams.returnCenter", True),
        ),
        "stateArgs": core.stateParams(
            stateInit=config.get("stateParams.stateInit", 0.0),
            stateCovarInit=config.get("stateParams.stateCovarInit", 100.0),
            boundState=config.get("stateParams.boundState", True),
            stateLowerBound=config.get("stateParams.stateLowerBound", 0.0),
            stateUpperBound=config.get("stateParams.stateUpperBound", 10000.0),
        ),
        "samArgs": core.samParams(
            samThreads=config.get("samParams.samThreads", 1),
            samFlagExclude=config.get("samParams.samFlagExclude", 3844),
            oneReadPerBin=config.get("samParams.oneReadPerBin", 0),
            chunkSize=config.get("samParams.chunkSize", 1000000),
            offsetStr=config.get("samParams.offsetStr", "0,0"),
            extendBP=config.get("samParams.extendBP", []),
            maxInsertSize=config.get("samParams.maxInsertSize", 1000),
            pairedEndMode=config.get("samParams.pairedEndMode", 0),
            inferFragmentLength=config.get("samParams.inferFragmentLength", 0),
        ),
        "detrendArgs": core.detrendParams(
            detrendWindowLengthBP=config.get(
                "detrendParams.detrendWindowLengthBP", 10000
            ),
            detrendTrackPercentile=config.get(
                "detrendParams.detrendTrackPercentile", 75.0
            ),
            usePolyFilter=config.get("detrendParams.usePolyFilter", False),
            detrendSavitzkyGolayDegree=config.get(
                "detrendParams.detrendSavitzkyGolayDegree", 2
            ),
            useOrderStatFilter=config.get(
                "detrendParams.useOrderStatFilter", True
            ),
        ),
        "matchingArgs": core.matchingParams(
            templateNames=config.get("matchingParams.templateNames", []),
            cascadeLevels=config.get("matchingParams.cascadeLevels", [2]),
            iters=config.get("matchingParams.iters", 25_000),
            alpha=config.get("matchingParams.alpha", 0.05),
            minMatchLengthBP=config.get(
                "matchingParams.minMatchLengthBP", None
            ),
            maxNumMatches=config.get("matchingParams.maxNumMatches", 100_000),
            minSignalAtMaxima=config.get(
                "matchingParams.minSignalAtMaxima", None
            ),
            merge=config.get("matchingParams.merge", True),
            mergeGapBP=config.get("matchingParams.mergeGapBP", 50),
            useScalingFunction=config.get(
                "matchingParams.useScalingFunction", True
            ),
            excludeRegionsBedFile=config.get(
                "matchingParams.excludeRegionsBedFile",
                matchingExcludeRegionsBedFile_default,
            ),
        ),
    }


def convertBedGraphToBigWig(experimentName, chromSizesFile):
    suffixes = ["state", "residuals"]
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
    except Exception as e:
        logger.warning(f"\n{warningMessage}\n")
        return
    if path_ is None or len(path_) == 0:
        logger.warning(f"\n{warningMessage}\n")
        return
    logger.info(f"Using bedGraphToBigWig from {path_}")
    for suffix in suffixes:
        bedgraph = f"consenrichOutput_{experimentName}_{suffix}.bedGraph"
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
            logger.info(f"Finished: converted {bedgraph} to {bigwig}.")


def main():
    parser = argparse.ArgumentParser(description="Consenrich CLI")
    parser.add_argument(
        "--config",
        type=str,
        dest="config",
        help="Path to a YAML config file with parameters + arguments defined in `consenrich.core`",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="If set, logs config"
    )
    args = parser.parse_args()

    if not args.config:
        logger.info(
            "No config file provided, run with `--config <path_to_config.yaml>`"
        )
        logger.info(
            "See documentation: https://nolan-h-hamilton.github.io/Consenrich/"
        )
        sys.exit(0)

    if not os.path.exists(args.config):
        logger.info(f"Config file {args.config} does not exist.")
        logger.info(
            "See documentation: https://nolan-h-hamilton.github.io/Consenrich/"
        )
        sys.exit(0)

    config = readConfig(args.config)
    experimentName = config["experimentName"]
    genomeArgs = config["genomeArgs"]
    inputArgs = config["inputArgs"]
    countingArgs = config["countingArgs"]
    processArgs = config["processArgs"]
    observationArgs = config["observationArgs"]
    stateArgs = config["stateArgs"]
    samArgs = config["samArgs"]
    detrendArgs = config["detrendArgs"]
    matchingArgs = config["matchingArgs"]
    bamFiles = inputArgs.bamFiles
    bamFilesControl = inputArgs.bamFilesControl
    numSamples = len(bamFiles)
    numNearest = observationArgs.numNearest
    stepSize = countingArgs.stepSize
    excludeForNorm = genomeArgs.excludeForNorm
    chromSizes = genomeArgs.chromSizesFile
    scaleDown = countingArgs.scaleDown
    extendBP_ = core.resolveExtendBP(samArgs.extendBP, bamFiles)
    initialTreatmentScaleFactors = []
    if args.verbose:
        try:
            logger.info("Configuration:\n")
            config_truncated = {
                k: v
                for k, v in config.items()
                if k not in ["inputArgs", "genomeArgs", "countingArgs"]
            }
            config_truncated["experimentName"] = experimentName
            config_truncated["inputArgs"] = inputArgs
            config_truncated["genomeArgs"] = genomeArgs
            config_truncated["countingArgs"] = countingArgs
            config_truncated["processArgs"] = processArgs
            config_truncated["observationArgs"] = observationArgs
            config_truncated["stateArgs"] = stateArgs
            config_truncated["samArgs"] = samArgs
            config_truncated["detrendArgs"] = detrendArgs
            pprint.pprint(config_truncated, indent=4)
        except Exception as e:
            logger.warning(f"Failed to print parsed config:\n{e}\n")

    controlsPresent = checkControlsPresent(inputArgs)
    if args.verbose:
        logger.info(f"controlsPresent: {controlsPresent}")
    readLengthsBamFiles = getReadLengths(inputArgs, countingArgs, samArgs)
    effectiveGenomeSizes = getEffectiveGenomeSizes(
        genomeArgs, readLengthsBamFiles
    )
    matchingEnabled = checkMatchingEnabled(matchingArgs)
    if args.verbose:
        logger.info(f"matchingEnabled: {matchingEnabled}")
    scaleFactors = countingArgs.scaleFactors
    scaleFactorsControl = countingArgs.scaleFactorsControl

    if controlsPresent:
        readLengthsControlBamFiles = [
            core.getReadLength(
                bamFile,
                countingArgs.numReads,
                1000,
                samArgs.samThreads,
                samArgs.samFlagExclude,
            )
            for bamFile in bamFilesControl
        ]
        effectiveGenomeSizesControl = [
            constants.getEffectiveGenomeSize(genomeArgs.genomeName, readLength)
            for readLength in readLengthsControlBamFiles
        ]

        if scaleFactors is not None and scaleFactorsControl is not None:
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
                        genomeArgs.excludeChroms,
                        genomeArgs.chromSizesFile,
                        samArgs.samThreads,
                    )
                    for bamFile, effectiveGenomeSize, readLength in zip(
                        bamFiles, effectiveGenomeSizes, readLengthsBamFiles
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
                    scaleDown,
                )
                for bamFileA, bamFileB, effectiveGenomeSizeA, effectiveGenomeSizeB, readLengthA, readLengthB in zip(
                    bamFiles,
                    bamFilesControl,
                    effectiveGenomeSizes,
                    effectiveGenomeSizesControl,
                    readLengthsBamFiles,
                    readLengthsControlBamFiles,
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
        scaleFactors = [
            detrorm.getScaleFactor1x(
                bamFile,
                effectiveGenomeSize,
                readLength,
                genomeArgs.excludeChroms,
                genomeArgs.chromSizesFile,
                samArgs.samThreads,
            )
            for bamFile, effectiveGenomeSize, readLength in zip(
                bamFiles, effectiveGenomeSizes, readLengthsBamFiles
            )
        ]
    chromSizesDict = misc_util.getChromSizesDict(
        genomeArgs.chromSizesFile, excludeChroms=genomeArgs.excludeChroms
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
        chromosomeEnd = max(0, (chromosomeEnd - (chromosomeEnd % stepSize)))
        numIntervals = (
            ((chromosomeEnd - chromosomeStart) + stepSize) - 1
        ) // stepSize
        intervals = np.arange(chromosomeStart, chromosomeEnd, stepSize)
        chromMat: np.ndarray = np.empty(
            (numSamples, numIntervals), dtype=np.float32
        )
        if controlsPresent:
            j_: int = 0
            finalSF = 1.0
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
                    [readLengthsBamFiles[j_], readLengthsControlBamFiles[j_]],
                    [treatScaleFactors[j_], controlScaleFactors[j_]],
                    samArgs.oneReadPerBin,
                    samArgs.samThreads,
                    samArgs.samFlagExclude,
                    offsetStr=samArgs.offsetStr,
                    extendBP=extendBP_[j_],
                    maxInsertSize=samArgs.maxInsertSize,
                    pairedEndMode=samArgs.pairedEndMode,
                    inferFragmentLength=samArgs.inferFragmentLength,
                    applyAsinh=countingArgs.applyAsinh,
                    applyLog=countingArgs.applyLog,
                )
                if countingArgs.rescaleToTreatmentCoverage:
                    finalSF = max(1.0, initialTreatmentScaleFactors[j_])
                chromMat[j_, :] = finalSF * (
                    pairMatrix[0, :] - pairMatrix[1, :]
                )
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
                extendBP=samArgs.extendBP,
                maxInsertSize=samArgs.maxInsertSize,
                pairedEndMode=samArgs.pairedEndMode,
                inferFragmentLength=samArgs.inferFragmentLength,
                applyAsinh=countingArgs.applyAsinh,
                applyLog=countingArgs.applyLog,
            )
        sparseMap = None
        if genomeArgs.sparseBedFile and not observationArgs.useALV:
            logger.info(f"Building sparse mapping for {chromosome}...")
            sparseMap = core.getSparseMap(
                chromosome, intervals, numNearest, genomeArgs.sparseBedFile
            )

        muncMat = np.empty_like(chromMat, dtype=np.float32)
        for j in range(numSamples):
            logger.info(f"Muncing {j + 1}/{numSamples} for {chromosome}...")
            muncMat[j, :] = core.getMuncTrack(
                chromosome,
                intervals,
                stepSize,
                chromMat[j, :],
                observationArgs.minR,
                observationArgs.maxR,
                observationArgs.useALV,
                observationArgs.useConstantNoiseLevel,
                observationArgs.noGlobal,
                observationArgs.localWeight,
                observationArgs.globalWeight,
                observationArgs.approximationWindowLengthBP,
                observationArgs.lowPassWindowLengthBP,
                observationArgs.returnCenter,
                sparseMap=sparseMap,
                lowPassFilterType=observationArgs.lowPassFilterType,
            )
            chromMat[j, :] = detrorm.detrendTrack(
                chromMat[j, :],
                stepSize,
                detrendArgs.detrendWindowLengthBP,
                detrendArgs.useOrderStatFilter,
                detrendArgs.usePolyFilter,
                detrendArgs.detrendTrackPercentile,
                detrendArgs.detrendSavitzkyGolayDegree,
            )
        logger.info(f">>>Running consenrich: {chromosome}<<<")

        x, P, y = core.runConsenrich(
            chromMat,
            muncMat,
            processArgs.deltaF,
            processArgs.minQ,
            processArgs.maxQ,
            processArgs.offDiagQ,
            processArgs.dStatAlpha,
            processArgs.dStatd,
            processArgs.dStatPC,
            stateArgs.stateInit,
            stateArgs.stateCovarInit,
            stateArgs.boundState,
            stateArgs.stateLowerBound,
            stateArgs.stateUpperBound,
            samArgs.chunkSize,
            progressIter=50_000,
        )
        logger.info("Done.")

        x_ = core.getPrimaryState(x)
        y_ = core.getPrecisionWeightedResidual(y, muncMat)

        df = pd.DataFrame(
            {
                "Chromosome": chromosome,
                "Start": intervals,
                "End": intervals + stepSize,
                "State": x_,
                "Res": y_,
            }
        )
        if c_ == 0 and len(chromosomes) > 1:
            for file_ in os.listdir("."):
                if file_.startswith(f"consenrichOutput_{experimentName}") and (
                    file_.endswith(".bedGraph") or file_.endswith(".narrowPeak")
                ):
                    logger.warning(f"Overwriting: {file_}")
                    os.remove(file_)

        for col, suffix in [("State", "state"), ("Res", "residuals")]:
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
        try:
            if matchingEnabled:
                matchingDF = matching.matchWavelet(
                    chromosome,
                    intervals,
                    x_,
                    matchingArgs.templateNames,
                    matchingArgs.cascadeLevels,
                    matchingArgs.iters,
                    matchingArgs.alpha,
                    matchingArgs.minMatchLengthBP,
                    matchingArgs.maxNumMatches,
                    matchingArgs.minSignalAtMaxima,
                    useScalingFunction=matchingArgs.useScalingFunction,
                    excludeRegionsBedFile=matchingArgs.excludeRegionsBedFile,
                )
                if not matchingDF.empty:
                    matchingDF.to_csv(
                        f"consenrichOutput_{experimentName}_matches.narrowPeak",
                        sep="\t",
                        header=False,
                        index=False,
                        mode="a",
                        float_format="%.3f",
                        lineterminator="\n",
                    )
        except Exception as e:
            logger.warning(
                f"Matching routine unsuccessful for {chromosome}...SKIPPING:\n{e}\n\n"
            )
            continue
    logger.info("Finished: output in human-readable format")
    convertBedGraphToBigWig(experimentName, genomeArgs.chromSizesFile)
    if matchingEnabled and matchingArgs.merge:
        try:
            matching.mergeMatches(
                f"consenrichOutput_{experimentName}_matches.narrowPeak",
                mergeGapBP=matchingArgs.mergeGapBP,
            )

        except Exception as e:
            logger.warning(f"Failed to merge matches...SKIPPING:\n{e}\n\n")
    logger.info("Done.")


if __name__ == "__main__":
    main()
