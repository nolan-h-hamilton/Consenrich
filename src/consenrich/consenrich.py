#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import gzip
import json
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
import yaml

import consenrich.core as core
import consenrich.misc_util as misc_util
import consenrich.constants as constants
import consenrich.detrorm as detrorm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _listOrEmpty(list_):
    if list_ is None:
        return []
    return list_

def getInputArgs(config_path:str) -> core.inputParams:
    def _expandWildCards(bamList) -> List[str]:
        expanded = []
        for entry in bamList:
            if '*' in entry or '?' in entry or '[' in entry:
                matched = glob.glob(entry)
                expanded.extend(matched)
            else:
                expanded.append(entry)
        return expanded
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    bamFilesRaw = config.get('inputParams.bamFiles', [])
    bamFilesControlRaw = config.get('inputParams.bamFilesControl', [])
    bamFiles = _expandWildCards(bamFilesRaw)
    bamFilesControl = _expandWildCards(bamFilesControlRaw)
    if len(bamFiles) == 0:
        raise ValueError("No BAM files provided in the configuration.")
    if len(bamFilesControl) > 0 and len(bamFilesControl) != len(bamFiles) and len(bamFilesControl) != 1:
        raise ValueError("Number of control BAM files must be 0, 1, or the same as number of treatment files")
    if len(bamFilesControl) == 1:
        # If there are multiple bamFiles, but 1 control, control is applied for all treatment files
        logger.info(f"Only one control given: Using {bamFilesControl[0]} for all treatment files.")
        bamFilesControl = bamFilesControl * len(bamFiles)
    return core.inputParams(
        bamFiles=bamFiles,
        bamFilesControl=bamFilesControl
    )


def getGenomeArgs(config_path: str) -> core.genomeParams:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    genomeName = config.get('genomeParams.name', None)
    genome = constants.resolveGenomeName(genomeName)
    chromSizesFile: Optional[str] = None
    blacklistFile: Optional[str] = None
    sparseBedFile: Optional[str] = None
    chromosomes: Optional[List[str]] = None
    excludeChroms: List[str] = config.get('genomeParams.excludeChroms', [])
    excludeForNorm: List[str] = config.get('genomeParams.excludeForNorm', [])
    if genome:
        chromSizesFile = constants.getGenomeResourceFile(genome, 'sizes')
        blacklistFile = constants.getGenomeResourceFile(genome, 'blacklist')
        sparseBedFile = constants.getGenomeResourceFile(genome, 'sparse')
    if config.get('genomeParams.chromSizesFile', None):
        chromSizesFile = config['genomeParams.chromSizesFile']
    if config.get('genomeParams.blacklistFile', None):
        blacklistFile = config['genomeParams.blacklistFile']
    if config.get('genomeParams.sparseBedFile', None):
        sparseBedFile = config['genomeParams.sparseBedFile']
    if not chromSizesFile or not os.path.exists(chromSizesFile):
        raise FileNotFoundError(f"Chromosome sizes file {chromSizesFile} does not exist.")
    if config.get('genomeParams.chromosomes', None):
        chromosomes = config['genomeParams.chromosomes']
    else:
        if chromSizesFile:
            chromosomes = list(pd.read_csv(chromSizesFile, sep='\t', header=None, names=['chrom', 'size'])['chrom'])
        else:
            raise ValueError("No chromosomes provided in the configuration and no chromosome sizes file specified.")
    chromosomes = [chrom.strip() for chrom in chromosomes if chrom.strip()]
    if excludeChroms:
        chromosomes = [chrom for chrom in chromosomes if chrom not in excludeChroms]
    if not chromosomes:
        raise ValueError("No valid chromosomes found after excluding specified chromosomes.")
    return core.genomeParams(
        genomeName=genome,
        chromSizesFile=chromSizesFile,
        blacklistFile=blacklistFile,
        sparseBedFile=sparseBedFile,
        chromosomes=chromosomes,
        excludeChroms=excludeChroms,
        excludeForNorm=excludeForNorm)

def getCountingArgs(config_path: str) -> core.countingParams:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    stepSize = config.get('countingParams.stepSize', 25)
    scaleDown = config.get('countingParams.scaleDown', True)
    scaleFactors = config.get('countingParams.scaleFactors', None)
    numReads = config.get('countingParams.numReads', 100)
    scaleFactorsControl = config.get('countingParams.scaleFactorsControl', None)
    if scaleFactors is not None and not isinstance(scaleFactors, list):
        raise ValueError("`scaleFactors` should be a list of floats.")
    if scaleFactorsControl is not None and not isinstance(scaleFactorsControl, list):
        raise ValueError("`scaleFactorsControl` should be a list of floats.")
    if scaleFactors is not None and scaleFactorsControl is not None and len(scaleFactors) != len(scaleFactorsControl):
        if len(scaleFactorsControl) == 1:
            scaleFactorsControl = scaleFactorsControl * len(scaleFactors)
        else:
            raise ValueError("control and treatment scale factors: must be equal length or 1 control")
    return core.countingParams(
        stepSize=stepSize,
        scaleDown=scaleDown,
        scaleFactors=scaleFactors,
        scaleFactorsControl=scaleFactorsControl,
        numReads=numReads
    )

def readConfig(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    inputParams = getInputArgs(config_path)
    genomeParams = getGenomeArgs(config_path)
    countingParams = getCountingArgs(config_path)
    minR_default = 1/len(inputParams.bamFiles) + 1.0e-4
    return {
        'experimentName': config.get('experimentName', 'consenrichExperiment'),
        'genomeArgs': genomeParams,
        'inputArgs': inputParams,
        'countingArgs': countingParams,
        'processArgs': core.processParams(
            deltaF=config.get('processParams.deltaF', 0.5),
            minQ=config.get('processParams.minQ', 0.25),
            maxQ=config.get('processParams.maxQ', 500.0),
            offDiagQ=config.get('processParams.offDiagQ', 0.0),
            dStatAlpha=config.get('processParams.dStatAlpha', 3.0),
            dStatd=config.get('processParams.dStatd', 10.0),
            dStatPC=config.get('processParams.dStatPC', 2.0)
        ),
        'observationArgs': core.observationParams(
            minR=minR_default,
            maxR=config.get('observationParams.maxR', 500.0),
            useALV=config.get('observationParams.useALV', False),
            useConstantNoiseLevel=config.get('observationParams.useConstantNoiseLevel', False),
            noGlobal=config.get('observationParams.noGlobal', False),
            numNearest=config.get('observationParams.numNearest', 25),
            localWeight=config.get('observationParams.localWeight', 0.333),
            globalWeight=config.get('observationParams.globalWeight', 0.667),
            approximationWindowLengthBP=config.get('observationParams.approximationWindowLengthBP', 10000),
            lowPassWindowLengthBP=config.get('observationParams.lowPassWindowLengthBP', 20000),
            returnCenter=config.get('observationParams.returnCenter', True)
        ),
        'stateArgs': core.stateParams(
            stateInit=config.get('stateParams.stateInit', 0.0),
            stateCovarInit=config.get('stateParams.stateCovarInit', 100.0),
            boundState=config.get('stateParams.boundState', True),
            stateLowerBound=config.get('stateParams.stateLowerBound', 0.0),
            stateUpperBound=config.get('stateParams.stateUpperBound', 10000.0)
        ),
        'samArgs': core.samParams(
            samThreads=config.get('samParams.samThreads', 1),
            samFlagExclude=config.get('samParams.samFlagExclude', 3844),
            oneReadPerBin=config.get('samParams.oneReadPerBin', 0),
            chunkSize=config.get('samParams.chunkSize', 1000000)
        ),
        'detrendArgs': core.detrendParams(
            detrendWindowLengthBP=config.get('detrendParams.detrendWindowLengthBP', 10000),
            detrendTrackPercentile=config.get('detrendParams.detrendTrackPercentile', 75.0),
            usePolyFilter=config.get('detrendParams.usePolyFilter', False),
            detrendSavitzkyGolayDegree=config.get('detrendParams.detrendSavitzkyGolayDegree', 2),
            useOrderStatFilter=config.get('detrendParams.useOrderStatFilter', True)
        )
    }

def convertBedGraphToBigWig(experimentName, chromSizesFile):
    suffixes = ['state', 'residuals']
    path_ = ""
    warningMessage = ("Could not find UCSC bedGraphToBigWig binary utility."
        "If you need bigWig files instead of the default, human-readable bedGraph files,"
        "you can download the `bedGraphToBigWig` binary from https://hgdownload.soe.ucsc.edu/admin/exe/<operatingSystem, architecture>"
        "OR install via conda (conda install -c bioconda ucsc-bedgraphtobigwig).")

    logger.info("Attempting to generate bigWig files from bedGraph format...")
    try:
        path_ = shutil.which('bedGraphToBigWig')
    except Exception as e:
        logger.warning(f"\n{warningMessage}\n")
        return
    if path_ is None or len(path_) == 0:
        logger.warning(f"\n{warningMessage}\n")
        return
    logger.info(f"Using bedGraphToBigWig from {path_}")

    for suffix in suffixes:
        bedgraph = f'consenrichOutput_{experimentName}_{suffix}.bedGraph'
        if not os.path.exists(bedgraph):
            logger.warning(f"bedGraph file {bedgraph} does not exist. Skipping sorting + bigWig conversion.")
            continue
        if not os.path.exists(chromSizesFile):
            logger.warning(f"{chromSizesFile} does not exist. Skipping sorting + bigWig conversion.")
            continue
        sorted_bedgraph = bedgraph.replace('.bedGraph', '_sorted.bedGraph')
        bigwig = f'{experimentName}_consenrich_{suffix}.bw'
        subprocess.run(['sort', '-k1,1', '-k2,2n', bedgraph, '-o', sorted_bedgraph], check=True)
        subprocess.run([path_, sorted_bedgraph, chromSizesFile, bigwig], check=True)
        if os.path.exists(bigwig) and os.path.getsize(bigwig) > 100:
            os.remove(sorted_bedgraph)
            logger.info(f"Converted {bedgraph} to {bigwig}.")


def main():
    parser = argparse.ArgumentParser(description="Consenrich CLI")
    parser.add_argument('--config', type=str, dest='config', help='Path to a YAML config file with parameters + arguments defined in `consenrich.core`')
    args = parser.parse_args()

    if not args.config:
        logger.info("No config file provided, run with `--config <path_to_config.yaml>`")
        logger.info("See documentation: https://nolan-h-hamilton.github.io/Consenrich/")
        sys.exit(0)

    if not os.path.exists(args.config):
        logger.info(f"Config file {args.config} does not exist.")
        logger.info("See documentation: https://nolan-h-hamilton.github.io/Consenrich/")
        sys.exit(0)

    config = readConfig(args.config)
    experimentName = config['experimentName']
    genomeArgs = config['genomeArgs']
    inputArgs = config['inputArgs']
    countingArgs = config['countingArgs']
    processArgs = config['processArgs']
    observationArgs = config['observationArgs']
    stateArgs = config['stateArgs']
    samArgs = config['samArgs']
    detrendArgs = config['detrendArgs']
    bamFiles = inputArgs.bamFiles
    bamFilesControl = inputArgs.bamFilesControl
    numSamples = len(bamFiles)
    numNearest = observationArgs.numNearest
    stepSize = countingArgs.stepSize
    excludeForNorm = genomeArgs.excludeForNorm
    chromSizes = genomeArgs.chromSizesFile
    scaleDown = countingArgs.scaleDown

    controlsPresent = inputArgs.bamFilesControl and isinstance(inputArgs.bamFilesControl, list) and len(inputArgs.bamFilesControl) > 0
    readLengthsBamFiles = [core.getReadLength(bamFile, countingArgs.numReads, 1000, samArgs.samThreads, samArgs.samFlagExclude)  for bamFile in bamFiles]
    effectiveGenomeSizes = [constants.getEffectiveGenomeSize(genomeArgs.genomeName, readLength) for readLength in readLengthsBamFiles]
    scaleFactors = countingArgs.scaleFactors
    scaleFactorsControl = countingArgs.scaleFactorsControl

    if controlsPresent:
        treatScaleFactors = []
        controlScaleFactors = []
        if scaleFactors is None or scaleFactorsControl is None:
            readLengthsControlBamFiles = [core.getReadLength(bamFile, countingArgs.numReads, 1000, samArgs.samThreads, samArgs.samFlagExclude) for bamFile in bamFilesControl]
            effectiveGenomeSizesControl = [constants.getEffectiveGenomeSize(genomeArgs.genomeName, readLength) for readLength in readLengthsControlBamFiles]
            pairScalingFactors = [
                detrorm.getPairScaleFactors(bamFileA, bamFileB, effectiveGenomeSizeA, effectiveGenomeSizeB,
                                        readLengthA, readLengthB, excludeForNorm,
                                        chromSizes, samArgs.samThreads, scaleDown)
                for bamFileA, bamFileB, effectiveGenomeSizeA, effectiveGenomeSizeB,
                    readLengthA, readLengthB in zip(bamFiles, bamFilesControl,
                                                    effectiveGenomeSizes, effectiveGenomeSizesControl,
                                                    readLengthsBamFiles, readLengthsControlBamFiles)
            ]
            for scaleFactorA, scaleFactorB in pairScalingFactors:
                treatScaleFactors.append(scaleFactorA)
                controlScaleFactors.append(scaleFactorB)
    else:
        treatScaleFactors = scaleFactors
        controlScaleFactors = scaleFactorsControl


    if scaleFactors is None:
        scaleFactors = [
            detrorm.getScaleFactor1x(bamFile, effectiveGenomeSize, readLength, genomeArgs.excludeChroms, genomeArgs.chromSizesFile, samArgs.samThreads)
            for bamFile, effectiveGenomeSize, readLength in zip(bamFiles, effectiveGenomeSizes, readLengthsBamFiles)
        ]
    chromSizesDict = misc_util.getChromSizesDict(genomeArgs.chromSizesFile, excludeChroms=genomeArgs.excludeChroms)
    chromosomes = genomeArgs.chromosomes


    for c_, chromosome in enumerate(chromosomes):
        chromosomeStart, chromosomeEnd = core.getChromRangesJoint(bamFiles, chromosome, chromSizesDict[chromosome], samArgs.samThreads, samArgs.samFlagExclude)
        chromosomeStart = max(0, (chromosomeStart - (chromosomeStart % stepSize)))
        chromosomeEnd = max(0, (chromosomeEnd - (chromosomeEnd % stepSize)))
        intervals = np.arange(chromosomeStart, chromosomeEnd + stepSize, stepSize)

        chromMat: np.ndarray = np.empty((numSamples, len(intervals)), dtype=np.float64)
        if controlsPresent:
            j_: int = 0
            for bamA, bamB in zip(bamFiles, bamFilesControl):
                pairMatrix: np.ndarray = core.readBamSegments(
                    [bamA, bamB], chromosome, chromosomeStart, chromosomeEnd, stepSize,
                    [readLengthsBamFiles[j_], readLengthsControlBamFiles[j_]], [treatScaleFactors[j_], controlScaleFactors[j_]],
                    samArgs.oneReadPerBin, samArgs.samThreads, samArgs.samFlagExclude
                )
                chromMat[j_, :] = scaleFactors[j_]*(pairMatrix[0, :] - pairMatrix[1, :])
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
                samArgs.samFlagExclude
            )
        sparseMap = None
        if genomeArgs.sparseBedFile and not observationArgs.useALV:
            logger.info(f"Building sparse mapping for {chromosome}...")
            sparseMap = core.getSparseMap(chromosome, intervals, numNearest, genomeArgs.sparseBedFile)

        muncMat = np.empty_like(chromMat, dtype=np.float64)
        for j in range(numSamples):
            logger.info(f"Muncing {j+1}/{numSamples} for {chromosome}...")
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
                sparseMap=sparseMap
            )
            chromMat[j, :] = detrorm.detrendTrack(
                chromMat[j, :],
                stepSize,
                detrendArgs.detrendWindowLengthBP,
                detrendArgs.useOrderStatFilter,
                detrendArgs.usePolyFilter,
                detrendArgs.detrendTrackPercentile,
                detrendArgs.detrendSavitzkyGolayDegree
            )
        logger.info(f">>>Running consenrich: {chromosome}<<<")

        x, P, y = core.runConsenrich(chromMat, muncMat,
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
        progressIter=50_000)
        logger.info(f"Done.")


        x_ = core.getPrimaryState(x)
        y_ = core.getPrecisionWeightedResidual(y, muncMat)

        df = pd.DataFrame({
            "Chromosome": chromosome,"Start": intervals, "End": intervals + stepSize,
            "State": x_,
            "Res": y_})
        if c_ == 0 and len(chromosomes) > 1:
            for file_ in os.listdir('.'):
                if file_.startswith(f"consenrichOutput_{experimentName}") and file_.endswith('.bedGraph'):
                    os.remove(file_)

        for col, suffix in [("State", "state"),
                            ("Res", "residuals")]:
            df[["Chromosome", "Start", "End", col]].to_csv(
                f"consenrichOutput_{experimentName}_{suffix}.bedGraph", sep="\t",
                header=False, index=False, mode="a", float_format="%.3f",
                lineterminator="\n",
            )
    logger.info("Finished. Output is in bedGraph format.")
    convertBedGraphToBigWig(experimentName, genomeArgs.chromSizesFile)
    logger.info("Done.")

if __name__ == "__main__":
    main()
