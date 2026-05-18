"""YAML/default parsing and CLI configuration contracts."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

import consenrich.constants as constants
import consenrich.core as core
import consenrich.misc_util as misc_util
from . import io as io_helpers

logger = logging.getLogger(__name__)

GENERIC_DEFAULT_CONFIGURATION = "generic"
SUPPORTED_DEFAULT_CONFIGURATIONS = (GENERIC_DEFAULT_CONFIGURATION,)
DEFAULT_CONFIGURATION_KEYS = (
    "configuration",
)

DEFAULT_CONFIGURATION_VALUES: dict[str, dict[str, Any]] = {
    GENERIC_DEFAULT_CONFIGURATION: {
        "fitParams.ECM_fixedBackgroundIters": 50,
        "fitParams.ECM_fixedBackgroundRtol": 1.0e-6,
        "fitParams.ECM_outerIters": 32,
        "fitParams.ECM_minOuterIters": None,
        "fitParams.ECM_backgroundShiftRtol": 1.0e-6,
        "fitParams.ECM_outerNLLRtol": 1.0e-4,
        "fitParams.ECM_backgroundSmoothness": 10.0,  #
        "fitParams.ECM_backgroundLengthScaleMultiplier": 16.0,  # 8, 16, 32
        "processParams.processQCalibration": (
            core.PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL
        ),
        "processParams.processQWarmupECMIters": 5,
        "processParams.processQWarmupOuterIters": (
            core.PROCESS_Q_CALIBRATION_DEFAULT_OUTER_ITERS
        ),
        "processParams.processQLevelPriorWeight": 1.0,
        "processParams.processQTrendPriorWeight": 10.0,
        "processParams.stateModel": core.STATE_MODEL_LEVEL_TREND,
        "processParams.precisionMultiplierMin": 0.1,
        "processParams.precisionMultiplierMax": 10.0,
        "observationParams.precisionMultiplierMin": 0.1,
        "observationParams.precisionMultiplierMax": 10.0,
        "countingParams.gentleDetrendQuantile": 0.5,
        "uncertaintyCalibrationParams.enabled": True,
    }
}

def loadConfig(
    configSource: Union[str, Path, Mapping[str, Any]],
) -> Dict[str, Any]:
    r"""Load a YAML config from a path or accept an already-parsed mapping.

    If given a mapping object, just return it. If given a path, try to load as YAML --> dict
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
    r"""Support both dotted keys and YAML nested mappings for configs."""

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

def _normalizeDefaultConfigurationName(value: Any) -> str:
    if value is None:
        return GENERIC_DEFAULT_CONFIGURATION
    name = str(value).strip().lower().replace("_", "-")
    if not name:
        return GENERIC_DEFAULT_CONFIGURATION
    if name != GENERIC_DEFAULT_CONFIGURATION:
        supported = ", ".join(SUPPORTED_DEFAULT_CONFIGURATIONS)
        raise ValueError(
            f"Unsupported default configuration {value!r}. "
            f"Supported default configurations: {supported}."
        )
    return name


def _getDefaultConfigurationName(configMap: Mapping[str, Any]) -> str:
    for key in DEFAULT_CONFIGURATION_KEYS:
        value = _cfgGet(configMap, key, None)
        if value is not None:
            return _normalizeDefaultConfigurationName(value)
    return GENERIC_DEFAULT_CONFIGURATION


def _cfgDefault(configMap: Mapping[str, Any], dottedKey: str) -> Any:
    configurationName = _getDefaultConfigurationName(configMap)
    return DEFAULT_CONFIGURATION_VALUES[configurationName][dottedKey]

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
            io_helpers._coerceInputSource(
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
        treatmentSources = io_helpers._buildPathInputSources(bamFilesRaw, role="treatment")
        controlSources = io_helpers._buildPathInputSources(
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

    treatmentSources = io_helpers._prepareBedGraphSources(treatmentSources)
    controlSources = io_helpers._prepareBedGraphSources(controlSources)

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
        io_helpers._pyBigWigAvailable(),
    )

    roundDigits_ = _cfgGet(configData, "outputParams.roundDigits", 3)
    writeUncertainty_ = _cfgGet(
        configData,
        "outputParams.writeUncertainty",
        True,
    )
    return core.outputParams(
        convertToBigWig=convertToBigWig_,
        roundDigits=roundDigits_,
        writeUncertainty=writeUncertainty_,
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
    replicateMedianDetrend_ = _cfgGet(
        configData,
        "countingParams.replicateMedianDetrend",
        True,
    )
    replicateMedianDetrendWindowMultiplier_ = _cfgGet(
        configData,
        "countingParams.replicateMedianDetrendWindowMultiplier",
        2.0,
    )
    gentleDetrendQuantile_ = _cfgGet(
        configData,
        "countingParams.gentleDetrendQuantile",
        0.5,
    )
    gentleDetrendQuantile_ = float(gentleDetrendQuantile_)
    if (
        not np.isfinite(gentleDetrendQuantile_)
        or gentleDetrendQuantile_ < 0.0
        or gentleDetrendQuantile_ > 1.0
    ):
        raise ValueError("countingParams.gentleDetrendQuantile must be between 0 and 1")
    return core.countingParams(
        intervalSizeBP=intervalSizeBP,
        backgroundBlockSizeBP=backgroundBlockSizeBP_,
        scaleFactors=scaleFactorList,
        scaleFactorsControl=scaleFactorsControlList,
        normMethod=normMethod_,
        fragmentsGroupNorm=fragmentsGroupNorm_,
        fixControl=fixControl_,
        logOffset=logOffset_,
        logMult=logMult_,
        replicateMedianDetrend=bool(replicateMedianDetrend_),
        replicateMedianDetrendWindowMultiplier=float(
            replicateMedianDetrendWindowMultiplier_
        ),
        gentleDetrendQuantile=gentleDetrendQuantile_,
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
        "fiveprime",
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
    enabledDefault = _cfgDefault(configData, "uncertaintyCalibrationParams.enabled")
    blockDefault = None
    padDefault = _cfgGet(configData, "observationParams.pad", 1.0e-4)
    maxScores = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.maxScores",
        None,
    )
    maxHeldoutCells = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.maxHeldoutCells",
        None,
    )
    if maxScores is None and maxHeldoutCells is None:
        maxScores = core.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES
    return core.uncertaintyCalibrationParams(
        enabled=bool(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.enabled",
                enabledDefault,
            )
        ),
        folds=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.folds",
                core.uncertaintyCalibrationParams().folds,
            )
        ),
        blockSizeBP=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.blockSizeBP",
            blockDefault,
        ),
        holdoutFraction=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.holdoutFraction",
            None,
        ),
        heldoutReplicateFraction=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.heldoutReplicateFraction",
            None,
        ),
        maxScores=int(maxScores) if maxScores is not None else maxScores,
        maxHeldoutCells=(
            int(maxHeldoutCells) if maxHeldoutCells is not None else maxHeldoutCells
        ),
        maxDiagnosticRows=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.maxDiagnosticRows",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS,
            )
        ),
        minHeldoutCells=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.minHeldoutCells",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS,
            )
        ),
        targets=tuple(
            float(x)
            for x in _cfgGet(
                configData,
                "uncertaintyCalibrationParams.targets",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS,
            )
        ),
        minFactor=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.minFactor",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
            )
        ),
        maxFactor=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.maxFactor",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
            )
        ),
        factorMin=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.factorMin",
            None,
        ),
        factorMax=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.factorMax",
            None,
        ),
        ridge=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.ridge",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE,
            )
        ),
        wisWeight=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.wisWeight",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT,
            )
        ),
        aObsPenalty=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.aObsPenalty",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY,
            )
        ),
        aObsPriorStrength=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.aObsPriorStrength",
            None,
        ),
        calibrationECMIters=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.calibrationECMIters",
                3,
            )
        ),
        seed=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.seed",
                core.UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
            )
        ),
        pad=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.pad",
            padDefault,
        ),
        writeDiagnostics=bool(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.writeDiagnostics",
                False,
            )
        ),
    )


def readConfig(config_path: str) -> Dict[str, Any]:
    r"""Read and parse the configuration file for Consenrich.

    :param config_path: Path to the YAML configuration file.
    :return: Dictionary containing all parsed configuration parameters.
    """
    configData = loadConfig(config_path)
    defaultConfiguration = _getDefaultConfigurationName(configData)

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
    processArgs = core.processParams(
        deltaF=_cfgGet(configData, "processParams.deltaF", 1.0),
        stateModel=_cfgGet(
            configData,
            "processParams.stateModel",
            _cfgDefault(configData, "processParams.stateModel"),
        ),
        minQ=_cfgGet(configData, "processParams.minQ", -1.0),
        maxQ=_cfgGet(configData, "processParams.maxQ", 1000.0),
        offDiagQ=_cfgGet(
            configData,
            "processParams.offDiagQ",
            0.0,
        ),
        processQCalibration=_cfgGet(
            configData,
            "processParams.processQCalibration",
            _cfgDefault(configData, "processParams.processQCalibration"),
        ),
        processQWarmupECMIters=int(
            _cfgGet(
                configData,
                "processParams.processQWarmupECMIters",
                _cfgDefault(configData, "processParams.processQWarmupECMIters"),
            )
        ),
        processQWarmupOuterIters=int(
            _cfgGet(
                configData,
                "processParams.processQWarmupOuterIters",
                _cfgDefault(configData, "processParams.processQWarmupOuterIters"),
            )
        ),
        processQLevelTarget=(
            None if processQLevelTargetCfg is None else float(processQLevelTargetCfg)
        ),
        processQTrendTarget=(
            None if processQTrendTargetCfg is None else float(processQTrendTargetCfg)
        ),
        processQLevelPriorWeight=float(
            _cfgGet(
                configData,
                "processParams.processQLevelPriorWeight",
                _cfgDefault(configData, "processParams.processQLevelPriorWeight"),
            )
        ),
        processQTrendPriorWeight=float(
            _cfgGet(
                configData,
                "processParams.processQTrendPriorWeight",
                _cfgDefault(configData, "processParams.processQTrendPriorWeight"),
            )
        ),
        precisionMultiplierMin=float(
            _cfgGet(
                configData,
                "processParams.precisionMultiplierMin",
                _cfgDefault(configData, "processParams.precisionMultiplierMin"),
            )
        ),
        precisionMultiplierMax=float(
            _cfgGet(
                configData,
                "processParams.precisionMultiplierMax",
                _cfgDefault(configData, "processParams.precisionMultiplierMax"),
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
        pad=_cfgGet(configData, "observationParams.pad", 1.0e-4),
        precisionMultiplierMin=float(
            _cfgGet(
                configData,
                "observationParams.precisionMultiplierMin",
                _cfgDefault(configData, "observationParams.precisionMultiplierMin"),
            )
        ),
        precisionMultiplierMax=float(
            _cfgGet(
                configData,
                "observationParams.precisionMultiplierMax",
                _cfgDefault(configData, "observationParams.precisionMultiplierMax"),
            )
        ),
        useReplicateTrends=bool(
            _cfgGet(configData, "observationParams.useReplicateTrends", False)
        ),
    )

    ECM_useAPN_ = bool(_cfgGet(configData, "fitParams.ECM_useAPN", False))

    fitArgs = core.fitParams(
        ECM_fixedBackgroundIters=_cfgGet(
            configData,
            "fitParams.ECM_fixedBackgroundIters",
            _cfgDefault(configData, "fitParams.ECM_fixedBackgroundIters"),
        ),
        ECM_fixedBackgroundRtol=_cfgGet(
            configData,
            "fitParams.ECM_fixedBackgroundRtol",
            _cfgDefault(configData, "fitParams.ECM_fixedBackgroundRtol"),
        ),
        ECM_robustTNu=_cfgGet(configData, "fitParams.ECM_robustTNu", 10.0),
        ECM_useObsPrecisionReweighting=_cfgGet(
            configData,
            "fitParams.ECM_useObsPrecisionReweighting",
            True,
        ),
        ECM_useProcessPrecisionReweighting=_cfgGet(
            configData,
            "fitParams.ECM_useProcessPrecisionReweighting",
            True,
        )
        and (not ECM_useAPN_),
        ECM_useAPN=ECM_useAPN_,
        fitBackground=_cfgGet(
            configData,
            "fitParams.fitBackground",
            True,
        ),
        ECM_zeroCenterBackground=_cfgGet(
            configData,
            "fitParams.ECM_zeroCenterBackground",
            False,
        ),
        ECM_zeroCenterReplicateBias=_cfgGet(
            configData,
            "fitParams.ECM_zeroCenterReplicateBias",
            True,
        ),
        ECM_outerIters=_cfgGet(
            configData,
            "fitParams.ECM_outerIters",
            _cfgDefault(configData, "fitParams.ECM_outerIters"),
        ),
        ECM_minOuterIters=_cfgGet(
            configData,
            "fitParams.ECM_minOuterIters",
            _cfgDefault(configData, "fitParams.ECM_minOuterIters"),
        ),
        ECM_backgroundShiftRtol=_cfgGet(
            configData,
            "fitParams.ECM_backgroundShiftRtol",
            _cfgDefault(configData, "fitParams.ECM_backgroundShiftRtol"),
        ),
        ECM_outerNLLRtol=_cfgGet(
            configData,
            "fitParams.ECM_outerNLLRtol",
            _cfgDefault(configData, "fitParams.ECM_outerNLLRtol"),
        ),
        ECM_backgroundSmoothness=_cfgGet(
            configData,
            "fitParams.ECM_backgroundSmoothness",
            _cfgDefault(configData, "fitParams.ECM_backgroundSmoothness"),
        ),
        ECM_backgroundLengthScaleMultiplier=_cfgGet(
            configData,
            "fitParams.ECM_backgroundLengthScaleMultiplier",
            _cfgDefault(configData, "fitParams.ECM_backgroundLengthScaleMultiplier"),
        ),
    )

    samThreads = _cfgGet(configData, "samParams.samThreads", 2)
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
        gamma=_cfgGet(configData, "matchingParams.gamma", 0.25),
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
        "defaultConfiguration": defaultConfiguration,
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

__all__ = [
    "DEFAULT_CONFIGURATION_KEYS",
    "DEFAULT_CONFIGURATION_VALUES",
    "GENERIC_DEFAULT_CONFIGURATION",
    "SUPPORTED_DEFAULT_CONFIGURATIONS",
    "_cfgDefault",
    "_cfgGet",
    "_getDefaultConfigurationName",
    "_normalizeDefaultConfigurationName",
    "getCountingArgs",
    "getGenomeArgs",
    "getInputArgs",
    "getOutputArgs",
    "getScArgs",
    "getStateArgs",
    "getUncertaintyCalibrationArgs",
    "loadConfig",
    "readConfig",
]
