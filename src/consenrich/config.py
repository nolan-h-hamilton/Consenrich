"""YAML/default parsing and CLI configuration contracts."""

from __future__ import annotations

import logging
import os
from collections import namedtuple
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

GENERIC_DEFAULT_CONFIGURATION = constants.GENERIC_DEFAULT_CONFIGURATION
SUPPORTED_DEFAULT_CONFIGURATIONS = constants.SUPPORTED_DEFAULT_CONFIGURATIONS
DEFAULT_CONFIGURATION_KEYS = constants.DEFAULT_CONFIGURATION_KEYS
DEFAULT_CONFIGURATION_VALUES = constants.DEFAULT_CONFIGURATION_VALUES

_PROCESS_ARGS_TYPE_CACHE: dict[tuple[str, ...], type] = {}


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


def _cfgHas(configMap: Mapping[str, Any], dottedKey: str) -> bool:
    if dottedKey in configMap:
        return True
    currentVal: Any = configMap
    for keyPart in dottedKey.split("."):
        if isinstance(currentVal, Mapping) and keyPart in currentVal:
            currentVal = currentVal[keyPart]
        else:
            return False
    return True


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


def _runtimeProcessParamsType(fields: tuple[str, ...]) -> type:
    processArgsType = _PROCESS_ARGS_TYPE_CACHE.get(fields)
    if processArgsType is None:
        processArgsType = namedtuple("processParams", fields)
        _PROCESS_ARGS_TYPE_CACHE[fields] = processArgsType
    return processArgsType


def _buildProcessArgs(
    baseValues: Mapping[str, Any],
    extraValues: Mapping[str, Any],
) -> Any:
    coreFields = tuple(getattr(core.processParams, "_fields", ()))
    coreValues = {key: baseValues[key] for key in coreFields if key in baseValues}
    coreExtras = {
        key: value for key, value in extraValues.items() if key in coreFields
    }
    if len(coreExtras) == len(extraValues):
        return core.processParams(**coreValues, **coreExtras)

    baseArgs = core.processParams(**coreValues, **coreExtras)
    extraFields = tuple(key for key in extraValues if key not in coreFields)
    fields = coreFields + extraFields
    processArgsType = _runtimeProcessParamsType(fields)
    return processArgsType(
        *(getattr(baseArgs, key) for key in coreFields),
        *(extraValues[key] for key in extraFields),
    )


def getInputArgs(config_path: str) -> core.inputParams:
    configData = loadConfig(config_path)
    defaultBarcodeTag = _cfgGet(
        configData,
        "scParams.barcodeTag",
        constants.SC_DEFAULT_BARCODE_TAG,
    )
    defaultFragmentPositionMode = _cfgGet(
        configData,
        "scParams.defaultFragmentPositionMode",
        constants.SC_DEFAULT_FRAGMENT_POSITION_MODE,
    )
    core._normalizeFragmentPositionMode(defaultFragmentPositionMode)

    sampleConfigs = _cfgGet(
        configData,
        "inputParams.samples",
        constants.INPUT_DEFAULT_SAMPLES,
    )
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
        bamFilesRaw = (
            _cfgGet(
                configData,
                "inputParams.bamFiles",
                constants.INPUT_DEFAULT_BAM_FILES,
            )
            or []
        )
        bamFilesControlRaw = (
            _cfgGet(
                configData,
                "inputParams.bamFilesControl",
                constants.INPUT_DEFAULT_BAM_FILES_CONTROL,
            )
            or []
        )
        treatmentSources = io_helpers._buildPathInputSources(
            bamFilesRaw, role="treatment"
        )
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

    roundDigits_ = _cfgGet(
        configData,
        "outputParams.roundDigits",
        constants.OUTPUT_DEFAULT_ROUND_DIGITS,
    )
    writeUncertainty_ = _cfgGet(
        configData,
        "outputParams.writeUncertainty",
        constants.OUTPUT_DEFAULT_WRITE_UNCERTAINTY,
    )
    saveBackgroundTracks_ = _cfgGet(
        configData,
        "outputParams.saveBackgroundTracks",
        _cfgDefault(configData, "outputParams.saveBackgroundTracks"),
    )
    plotOptimizationPath_ = _cfgGet(
        configData,
        "outputParams.plotOptimizationPath",
        constants.OUTPUT_DEFAULT_PLOT_OPTIMIZATION_PATH,
    )
    return core.outputParams(
        convertToBigWig=convertToBigWig_,
        roundDigits=roundDigits_,
        writeUncertainty=writeUncertainty_,
        saveBackgroundTracks=saveBackgroundTracks_,
        plotOptimizationPath=plotOptimizationPath_,
    )


def getGenomeArgs(config_path: str) -> core.genomeParams:
    configData = loadConfig(config_path)

    genomeName = _cfgGet(configData, "genomeParams.name", constants.GENOME_DEFAULT_NAME)
    genomeLabel = constants.resolveGenomeName(genomeName)

    chromSizesFile: Optional[str] = None
    blacklistFile: Optional[str] = None
    sparseBedFile: Optional[str] = None
    chromosomesList: Optional[List[str]] = None

    excludeChromsList: List[str] = (
        _cfgGet(
            configData,
            "genomeParams.excludeChroms",
            constants.GENOME_DEFAULT_EXCLUDE_CHROMS,
        )
        or []
    )
    excludeForNormList: List[str] = (
        _cfgGet(
            configData,
            "genomeParams.excludeForNorm",
            constants.GENOME_DEFAULT_EXCLUDE_FOR_NORM,
        )
        or []
    )

    if genomeLabel:
        chromSizesFile = constants.getGenomeResourceFile(genomeLabel, "sizes")
        blacklistFile = constants.getGenomeResourceFile(genomeLabel, "blacklist")
        sparseBedFile = constants.getGenomeResourceFile(genomeLabel, "sparse")

    chromSizesOverride = _cfgGet(
        configData,
        "genomeParams.chromSizesFile",
        constants.GENOME_DEFAULT_CHROM_SIZES_FILE,
    )
    if chromSizesOverride:
        chromSizesFile = chromSizesOverride

    blacklistOverride = _cfgGet(
        configData,
        "genomeParams.blacklistFile",
        constants.GENOME_DEFAULT_BLACKLIST_FILE,
    )
    if blacklistOverride:
        blacklistFile = blacklistOverride

    sparseOverride = _cfgGet(
        configData,
        "genomeParams.sparseBedFile",
        constants.GENOME_DEFAULT_SPARSE_BED_FILE,
    )
    if sparseOverride:
        sparseBedFile = sparseOverride

    if not chromSizesFile or not os.path.exists(chromSizesFile):
        raise FileNotFoundError(
            f"Chromosome sizes file {chromSizesFile} does not exist."
        )

    chromosomesConfig = _cfgGet(
        configData,
        "genomeParams.chromosomes",
        constants.GENOME_DEFAULT_CHROMOSOMES,
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

    stateInit_ = _cfgGet(configData, "stateParams.stateInit", constants.STATE_DEFAULT_INIT)
    stateCovarInit_ = _cfgGet(
        configData,
        "stateParams.stateCovarInit",
        constants.STATE_DEFAULT_COVAR_INIT,
    )
    boundState_ = _cfgGet(
        configData,
        "stateParams.boundState",
        constants.STATE_DEFAULT_BOUND_STATE,
    )
    stateLowerBound_ = _cfgGet(
        configData,
        "stateParams.stateLowerBound",
        constants.STATE_DEFAULT_LOWER_BOUND,
    )
    stateUpperBound_ = _cfgGet(
        configData,
        "stateParams.stateUpperBound",
        constants.STATE_DEFAULT_UPPER_BOUND,
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

    intervalSizeBP = _cfgGet(
        configData,
        "countingParams.intervalSizeBP",
        constants.COUNTING_DEFAULT_INTERVAL_SIZE_BP,
    )
    backgroundBlockSizeBP_ = _cfgGet(
        configData,
        "countingParams.backgroundBlockSizeBP",
        constants.COUNTING_DEFAULT_BACKGROUND_BLOCK_SIZE_BP,
    )
    scaleFactorList = _cfgGet(
        configData,
        "countingParams.scaleFactors",
        constants.COUNTING_DEFAULT_SCALE_FACTORS,
    )
    scaleFactorsControlList = _cfgGet(
        configData,
        "countingParams.scaleFactorsControl",
        constants.COUNTING_DEFAULT_SCALE_FACTORS_CONTROL,
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
        constants.COUNTING_DEFAULT_NORM_METHOD,
    )
    if normMethod_.upper() not in constants.COUNTING_SUPPORTED_NORM_METHODS:
        logger.warning(
            f"Unknown `countingParams.normMethod`...Using `{constants.COUNTING_DEFAULT_NORM_METHOD}`...",
        )
        normMethod_ = constants.COUNTING_DEFAULT_NORM_METHOD
    fragmentsGroupNorm_ = _cfgGet(
        configData,
        "countingParams.fragmentsGroupNorm",
        constants.COUNTING_DEFAULT_FRAGMENTS_GROUP_NORM,
    )
    if (
        str(fragmentsGroupNorm_).upper()
        not in constants.COUNTING_SUPPORTED_FRAGMENTS_GROUP_NORMS
    ):
        raise ValueError(
            "`countingParams.fragmentsGroupNorm` must be `NONE` or `CELLS`."
        )

    fixControl_ = _cfgGet(
        configData,
        "countingParams.fixControl",
        constants.COUNTING_DEFAULT_FIX_CONTROL,
    )
    logOffset_ = _cfgGet(
        configData,
        "countingParams.logOffset",
        constants.COUNTING_DEFAULT_LOG_OFFSET,
    )
    logMult_ = _cfgGet(
        configData,
        "countingParams.logMult",
        constants.COUNTING_DEFAULT_LOG_MULT,
    )
    subtractGlobalMedian_ = _cfgGet(
        configData,
        "countingParams.subtractGlobalMedian",
        _cfgDefault(configData, "countingParams.subtractGlobalMedian"),
    )
    replicateMedianDetrend_ = _cfgGet(
        configData,
        "countingParams.replicateMedianDetrend",
        constants.COUNTING_DEFAULT_REPLICATE_MEDIAN_DETREND,
    )
    replicateMedianDetrendWindowMultiplier_ = _cfgGet(
        configData,
        "countingParams.replicateMedianDetrendWindowMultiplier",
        constants.COUNTING_DEFAULT_REPLICATE_MEDIAN_DETREND_WINDOW_MULTIPLIER,
    )
    gentleDetrendQuantile_ = _cfgGet(
        configData,
        "countingParams.gentleDetrendQuantile",
        constants.COUNTING_DEFAULT_GENTLE_DETREND_QUANTILE,
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
        subtractGlobalMedian=bool(subtractGlobalMedian_),
        replicateMedianDetrend=bool(replicateMedianDetrend_),
        replicateMedianDetrendWindowMultiplier=float(
            replicateMedianDetrendWindowMultiplier_
        ),
        gentleDetrendQuantile=gentleDetrendQuantile_,
    )


def getScArgs(config_path: str) -> core.scParams:
    configData = loadConfig(config_path)

    barcodeTag_ = _cfgGet(
        configData,
        "scParams.barcodeTag",
        constants.SC_DEFAULT_BARCODE_TAG,
    )
    defaultCountMode_ = _cfgGet(
        configData,
        "scParams.defaultCountMode",
        constants.SC_DEFAULT_COUNT_MODE,
    )
    if str(defaultCountMode_).strip().lower() not in constants.SC_SUPPORTED_COUNT_MODES:
        raise ValueError("`scParams.defaultCountMode` is not supported.")

    fragmentsGroupNorm_ = _cfgGet(
        configData,
        "scParams.fragmentsGroupNorm",
        constants.SC_DEFAULT_FRAGMENTS_GROUP_NORM,
    )
    if (
        str(fragmentsGroupNorm_).upper()
        not in constants.COUNTING_SUPPORTED_FRAGMENTS_GROUP_NORMS
    ):
        raise ValueError("`scParams.fragmentsGroupNorm` must be `NONE` or `CELLS`.")

    defaultFragmentPositionMode_ = _cfgGet(
        configData,
        "scParams.defaultFragmentPositionMode",
        constants.SC_DEFAULT_FRAGMENT_POSITION_MODE,
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
    blockDefault = constants.UNCERTAINTY_CALIBRATION_DEFAULT_BLOCK_SIZE_BP
    padDefault = _cfgGet(
        configData,
        "observationParams.pad",
        constants.UNCERTAINTY_CALIBRATION_DEFAULT_PAD,
    )
    maxScores = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.maxScores",
        None,
    )
    maxHeldoutCells = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.maxHeldoutCells",
        constants.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_HELDOUT_CELLS,
    )
    if maxScores is None and maxHeldoutCells is None:
        maxScores = constants.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES
    enabledConfig = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.enabled",
        _cfgGet(configData, "uncertaintyCalibration.enabled", enabledDefault),
    )
    return core.uncertaintyCalibrationParams(
        enabled=bool(enabledConfig),
        folds=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.folds",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_FOLDS,
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
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS,
            )
        ),
        minHeldoutCells=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.minHeldoutCells",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS,
            )
        ),
        targets=tuple(
            float(x)
            for x in _cfgGet(
                configData,
                "uncertaintyCalibrationParams.targets",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS,
            )
        ),
        minFactor=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.minFactor",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
            )
        ),
        maxFactor=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.maxFactor",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
            )
        ),
        factorMin=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.factorMin",
            constants.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN_OVERRIDE,
        ),
        factorMax=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.factorMax",
            constants.UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX_OVERRIDE,
        ),
        ridge=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.ridge",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE,
            )
        ),
        wisWeight=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.wisWeight",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT,
            )
        ),
        aObsPenalty=float(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.aObsPenalty",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY,
            )
        ),
        aObsPriorStrength=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.aObsPriorStrength",
            constants.UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH_OVERRIDE,
        ),
        calibrationECMIters=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.calibrationECMIters",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_CALIBRATION_ECM_ITERS,
            )
        ),
        targetCalibrationDelta=_cfgGet(
            configData,
            "uncertaintyCalibrationParams.targetCalibrationDelta",
            constants.UNCERTAINTY_CALIBRATION_DEFAULT_TARGET_CALIBRATION_DELTA,
        ),
        scaleUncertaintyByTargetCalibration=bool(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_SCALE_UNCERTAINTY_BY_TARGET_CALIBRATION,
            )
        ),
        seed=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.seed",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
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
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_WRITE_DIAGNOSTICS,
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
    experimentName = _cfgGet(
        configData,
        "experimentName",
        constants.EXPERIMENT_DEFAULT_NAME,
    )
    regularizationStrengthKey = "processParams.regularizationStrength"
    mapRoughnessPenaltyKey = "processParams.processNoiseMapRoughnessPenalty"
    mapRoughnessPenaltyRaw = _cfgGet(configData, mapRoughnessPenaltyKey, None)
    regularizationStrength = float(
        _cfgGet(
            configData,
            regularizationStrengthKey,
            _cfgDefault(configData, regularizationStrengthKey),
        )
    )
    mapRoughnessPenalty = (
        regularizationStrength
        if mapRoughnessPenaltyRaw is None
        else float(mapRoughnessPenaltyRaw)
    )
    processNoiseWarmupECMIters = int(
        _cfgGet(
            configData,
            "processParams.processNoiseWarmupECMIters",
            _cfgDefault(configData, "processParams.processNoiseWarmupECMIters"),
        )
    )
    processNoiseWarmupOuterPasses = int(
        _cfgGet(
            configData,
            "processParams.processNoiseWarmupOuterPasses",
            _cfgDefault(configData, "processParams.processNoiseWarmupOuterPasses"),
        )
    )
    if processNoiseWarmupECMIters < 1:
        raise ValueError(
            "`processParams.processNoiseWarmupECMIters` must be a positive integer."
        )
    if processNoiseWarmupOuterPasses < 1:
        raise ValueError(
            "`processParams.processNoiseWarmupOuterPasses` must be a positive integer."
        )
    regularizationRatioConfigured = _cfgHas(
        configData,
        "processParams.regularizationRatio",
    )
    processArgs = _buildProcessArgs(
        {
            "deltaF": _cfgGet(
                configData,
                "processParams.deltaF",
                constants.PROCESS_DEFAULT_DELTA_F,
            ),
            "stateModel": _cfgGet(
                configData,
                "processParams.stateModel",
                _cfgDefault(configData, "processParams.stateModel"),
            ),
            "minQ": _cfgGet(
                configData,
                "processParams.minQ",
                constants.PROCESS_DEFAULT_MIN_Q,
            ),
            "maxQ": _cfgGet(
                configData,
                "processParams.maxQ",
                constants.PROCESS_DEFAULT_MAX_Q,
            ),
            "regularizationStrength": regularizationStrength,
            "regularizationRatio": float(
                _cfgGet(
                    configData,
                    "processParams.regularizationRatio",
                    _cfgDefault(configData, "processParams.regularizationRatio"),
                )
            ),
            "processNoiseWarmupECMIters": processNoiseWarmupECMIters,
            "precisionMultiplierMin": float(
                _cfgGet(
                    configData,
                    "processParams.precisionMultiplierMin",
                    _cfgDefault(configData, "processParams.precisionMultiplierMin"),
                )
            ),
            "precisionMultiplierMax": float(
                _cfgGet(
                    configData,
                    "processParams.precisionMultiplierMax",
                    _cfgDefault(configData, "processParams.precisionMultiplierMax"),
                )
            ),
        },
        {
            "processNoiseMapRoughnessPenalty": mapRoughnessPenalty,
            "processNoiseWarmupOuterPasses": processNoiseWarmupOuterPasses,
        },
    )
    if (
        regularizationRatioConfigured
        and core._normalizeStateModel(processArgs.stateModel) == core.STATE_MODEL_LEVEL
    ):
        logger.info(
            "processParams.regularizationRatio was provided but ignored because stateModel='level' has no trend state."
        )

    explicitSparseBedFile = _cfgGet(
        configData,
        "genomeParams.sparseBedFile",
        constants.GENOME_DEFAULT_SPARSE_BED_FILE,
    )
    sparseBedAvailable = bool(
        genomeParams.sparseBedFile and os.path.exists(str(genomeParams.sparseBedFile))
    )
    numNearestRequested = int(
        _cfgGet(
            configData,
            "observationParams.numNearest",
            constants.OBSERVATION_DEFAULT_NUM_NEAREST,
        )
        or 0
    )
    if explicitSparseBedFile and numNearestRequested > 0:
        numNearestResolved = numNearestRequested
    else:
        numNearestResolved = 0
    restrictLocalVarianceKey = "observationParams.restrictLocalVarianceToSparseBed"
    restrictLocalVarianceRequested = bool(
        _cfgGet(
            configData,
            restrictLocalVarianceKey,
            constants.OBSERVATION_DEFAULT_RESTRICT_LOCAL_VARIANCE_TO_SPARSE_BED,
        )
    )
    if restrictLocalVarianceRequested and not sparseBedAvailable:
        logger.warning(
            "Requested `%s`, but no "
            "readable sparse BED was resolved; disabling that option.",
            restrictLocalVarianceKey,
        )
    restrictLocalVarianceResolved = bool(
        restrictLocalVarianceRequested and sparseBedAvailable
    )
    trendMaxEdfCfg = _cfgGet(
        configData,
        "observationParams.trendMaxEdf",
        constants.OBSERVATION_DEFAULT_TREND_MAX_EDF,
    )
    muncVarianceModel = core._normalizeMuncVarianceModel(
        _cfgGet(
            configData,
            "observationParams.muncVarianceModel",
            _cfgDefault(configData, "observationParams.muncVarianceModel"),
        )
    )
    muncTrendBlockSizeBP = (
        _cfgGet(configData, "observationParams.muncTrendBlockSizeBP", None)
        if _cfgHas(configData, "observationParams.muncTrendBlockSizeBP")
        else _cfgDefault(configData, "observationParams.muncTrendBlockSizeBP")
    )
    if _cfgHas(configData, "observationParams.muncTrendBlockSizeBP") and muncTrendBlockSizeBP is None:
        muncTrendBlockSizeBP = -1
    muncLocalWindowSizeBP = (
        _cfgGet(configData, "observationParams.muncLocalWindowSizeBP", None)
        if _cfgHas(configData, "observationParams.muncLocalWindowSizeBP")
        else _cfgDefault(configData, "observationParams.muncLocalWindowSizeBP")
    )
    if _cfgHas(configData, "observationParams.muncLocalWindowSizeBP") and muncLocalWindowSizeBP is None:
        muncLocalWindowSizeBP = -1
    muncTrendBlockDependenceMultiplierRaw = _cfgGet(
        configData,
        "observationParams.muncTrendBlockDependenceMultiplier",
        _cfgDefault(
            configData,
            "observationParams.muncTrendBlockDependenceMultiplier",
        ),
    )
    if muncTrendBlockDependenceMultiplierRaw is None:
        muncTrendBlockDependenceMultiplierRaw = (
            constants.OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER
        )
    muncTrendBlockDependenceMultiplier = float(
        muncTrendBlockDependenceMultiplierRaw
    )
    if (
        not np.isfinite(muncTrendBlockDependenceMultiplier)
        or muncTrendBlockDependenceMultiplier <= 0.0
    ):
        raise ValueError(
            "`observationParams.muncTrendBlockDependenceMultiplier` must be positive."
        )
    muncLocalWindowDependenceMultiplierRaw = _cfgGet(
        configData,
        "observationParams.muncLocalWindowDependenceMultiplier",
        _cfgDefault(
            configData,
            "observationParams.muncLocalWindowDependenceMultiplier",
        ),
    )
    if muncLocalWindowDependenceMultiplierRaw is None:
        muncLocalWindowDependenceMultiplierRaw = (
            constants.OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER
        )
    muncLocalWindowDependenceMultiplier = float(
        muncLocalWindowDependenceMultiplierRaw
    )
    if (
        not np.isfinite(muncLocalWindowDependenceMultiplier)
        or muncLocalWindowDependenceMultiplier <= 0.0
    ):
        raise ValueError(
            "`observationParams.muncLocalWindowDependenceMultiplier` must be positive."
        )
    observationArgs = core.observationParams(
        minR=_cfgGet(configData, "observationParams.minR", constants.OBSERVATION_DEFAULT_MIN_R),
        maxR=_cfgGet(configData, "observationParams.maxR", constants.OBSERVATION_DEFAULT_MAX_R),
        samplingIters=_cfgGet(
            configData,
            "observationParams.samplingIters",
            constants.OBSERVATION_DEFAULT_SAMPLING_ITERS,
        ),
        samplingBlockSizeBP=_cfgGet(
            configData,
            "observationParams.samplingBlockSizeBP",
            constants.OBSERVATION_DEFAULT_SAMPLING_BLOCK_SIZE_BP,
        ),
        EB_use=_cfgGet(
            configData,
            "observationParams.EB_use",
            constants.OBSERVATION_DEFAULT_EB_USE,
        ),
        EB_setNu0=_cfgGet(
            configData,
            "observationParams.EB_setNu0",
            constants.OBSERVATION_DEFAULT_EB_SET_NU0,
        ),
        EB_setNuL=_cfgGet(
            configData,
            "observationParams.EB_setNuL",
            constants.OBSERVATION_DEFAULT_EB_SET_NUL,
        ),
        noDMVar=bool(
            _cfgGet(
                configData,
                "observationParams.noDMVar",
                constants.OBSERVATION_DEFAULT_NO_DM_VAR,
            )
        ),
        trendNumBasis=int(
            _cfgGet(
                configData,
                "observationParams.trendNumBasis",
                constants.OBSERVATION_DEFAULT_TREND_NUM_BASIS,
            )
        ),
        trendMinObsPerBasis=float(
            _cfgGet(
                configData,
                "observationParams.trendMinObsPerBasis",
                constants.OBSERVATION_DEFAULT_TREND_MIN_OBS_PER_BASIS,
            )
        ),
        trendMinEdf=float(
            _cfgGet(
                configData,
                "observationParams.trendMinEdf",
                constants.OBSERVATION_DEFAULT_TREND_MIN_EDF,
            )
        ),
        trendMaxEdf=None if trendMaxEdfCfg is None else float(trendMaxEdfCfg),
        trendLambdaMin=float(
            _cfgGet(
                configData,
                "observationParams.trendLambdaMin",
                constants.OBSERVATION_DEFAULT_TREND_LAMBDA_MIN,
            )
        ),
        trendLambdaMax=float(
            _cfgGet(
                configData,
                "observationParams.trendLambdaMax",
                constants.OBSERVATION_DEFAULT_TREND_LAMBDA_MAX,
            )
        ),
        trendLambdaGridSize=int(
            _cfgGet(
                configData,
                "observationParams.trendLambdaGridSize",
                constants.OBSERVATION_DEFAULT_TREND_LAMBDA_GRID_SIZE,
            )
        ),
        numNearest=numNearestResolved,
        sparseSupportScaleBP=_cfgGet(
            configData,
            "observationParams.sparseSupportScaleBP",
            constants.OBSERVATION_DEFAULT_SPARSE_SUPPORT_SCALE_BP,
        ),
        sparseSupportPrior=float(
            _cfgGet(
                configData,
                "observationParams.sparseSupportPrior",
                constants.OBSERVATION_DEFAULT_SPARSE_SUPPORT_PRIOR,
            )
        ),
        restrictLocalAR1ToSparseBed=restrictLocalVarianceResolved,
        pad=_cfgGet(configData, "observationParams.pad", constants.OBSERVATION_DEFAULT_PAD),
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
            _cfgGet(
                configData,
                "observationParams.useReplicateTrends",
                constants.OBSERVATION_DEFAULT_USE_REPLICATE_TRENDS,
            )
        ),
        muncVarianceModel=muncVarianceModel,
        muncTrendBlockSizeBP=muncTrendBlockSizeBP,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP,
        muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier,
        muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier,
        restrictLocalVarianceToSparseBed=restrictLocalVarianceResolved,
    )

    ECM_useAPN_ = bool(
        _cfgGet(configData, "fitParams.ECM_useAPN", constants.FIT_DEFAULT_USE_APN)
    )

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
        ECM_robustTNu=_cfgGet(
            configData,
            "fitParams.ECM_robustTNu",
            constants.FIT_DEFAULT_ROBUST_T_NU,
        ),
        ECM_useObsPrecisionReweighting=_cfgGet(
            configData,
            "fitParams.ECM_useObsPrecisionReweighting",
            constants.FIT_DEFAULT_USE_OBS_PRECISION_REWEIGHTING,
        ),
        ECM_useProcessPrecisionReweighting=_cfgGet(
            configData,
            "fitParams.ECM_useProcessPrecisionReweighting",
            constants.FIT_DEFAULT_USE_PROCESS_PRECISION_REWEIGHTING,
        )
        and (not ECM_useAPN_),
        ECM_useAPN=ECM_useAPN_,
        fitBackground=_cfgGet(
            configData,
            "fitParams.fitBackground",
            constants.FIT_DEFAULT_BACKGROUND,
        ),
        useNonnegativeBackground=_cfgGet(
            configData,
            "fitParams.useNonnegativeBackground",
            _cfgDefault(configData, "fitParams.useNonnegativeBackground"),
        ),
        backgroundNegativePenaltyMultiplier=_cfgGet(
            configData,
            "fitParams.backgroundNegativePenaltyMultiplier",
            _cfgDefault(
                configData,
                "fitParams.backgroundNegativePenaltyMultiplier",
            ),
        ),
        ECM_zeroCenterBackground=_cfgGet(
            configData,
            "fitParams.ECM_zeroCenterBackground",
            constants.FIT_DEFAULT_ZERO_CENTER_BACKGROUND,
        ),
        ECM_zeroCenterReplicateBias=_cfgGet(
            configData,
            "fitParams.ECM_zeroCenterReplicateBias",
            constants.FIT_DEFAULT_ZERO_CENTER_REPLICATE_BIAS,
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
            _cfgGet(
                configData,
                "fitParams.EM_backgroundSpanMultiplier",
                _cfgDefault(
                    configData,
                    "fitParams.ECM_backgroundLengthScaleMultiplier",
                ),
            ),
        ),
    )

    samThreads = _cfgGet(configData, "samParams.samThreads", constants.SAM_DEFAULT_THREADS)
    samFlagExclude = _cfgGet(
        configData,
        "samParams.samFlagExclude",
        constants.SAM_DEFAULT_FLAG_EXCLUDE,
    )
    minMappingQuality = _cfgGet(
        configData,
        "samParams.minMappingQuality",
        constants.SAM_DEFAULT_MIN_MAPPING_QUALITY,
    )
    oneReadPerBin = _cfgGet(
        configData,
        "samParams.oneReadPerBin",
        constants.SAM_DEFAULT_ONE_READ_PER_BIN,
    )
    chunkSize = _cfgGet(configData, "samParams.chunkSize", constants.SAM_DEFAULT_CHUNK_SIZE)
    bamInputMode = _cfgGet(
        configData,
        "samParams.bamInputMode",
        constants.SAM_DEFAULT_BAM_INPUT_MODE,
    )
    defaultCountMode = _cfgGet(
        configData,
        "samParams.defaultCountMode",
        constants.SAM_DEFAULT_COUNT_MODE,
    )
    shiftForward5p = int(
        _cfgGet(
            configData,
            "samParams.shiftForward5p",
            constants.SAM_DEFAULT_SHIFT_FORWARD_5P,
        )
    )
    shiftReverse5p = int(
        _cfgGet(
            configData,
            "samParams.shiftReverse5p",
            constants.SAM_DEFAULT_SHIFT_REVERSE_5P,
        )
    )
    extendFrom5pBP = _cfgGet(
        configData,
        "samParams.extendFrom5pBP",
        constants.SAM_DEFAULT_EXTEND_FROM_5P_BP,
    )
    maxInsertSize = _cfgGet(
        configData,
        "samParams.maxInsertSize",
        constants.SAM_DEFAULT_MAX_INSERT_SIZE,
    )
    inferFragmentLength = _cfgGet(
        configData,
        "samParams.inferFragmentLength",
        constants.SAM_DEFAULT_INFER_FRAGMENT_LENGTH,
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
            constants.SAM_DEFAULT_MIN_TEMPLATE_LENGTH,
        ),
    )

    matchingArgs = core.matchingParams(
        enabled=bool(
            _cfgGet(configData, "matchingParams.enabled", constants.MATCHING_DEFAULT_ENABLED)
        ),
        randSeed=_cfgGet(
            configData,
            "matchingParams.randSeed",
            constants.MATCHING_DEFAULT_RAND_SEED,
        ),
        numBootstrap=int(
            _cfgGet(
                configData,
                "matchingParams.numBootstrap",
                constants.MATCHING_DEFAULT_NUM_BOOTSTRAP,
            )
        ),
        thresholdZ=float(
            _cfgGet(
                configData,
                "matchingParams.thresholdZ",
                constants.MATCHING_DEFAULT_THRESHOLD_Z,
            )
        ),
        dependenceSpan=_cfgGet(
            configData,
            "matchingParams.dependenceSpan",
            constants.MATCHING_DEFAULT_DEPENDENCE_SPAN,
        ),
        gamma=_cfgGet(configData, "matchingParams.gamma", constants.MATCHING_DEFAULT_GAMMA),
        selectionPenalty=_cfgGet(
            configData,
            "matchingParams.selectionPenalty",
            constants.MATCHING_DEFAULT_SELECTION_PENALTY,
        ),
        gammaScale=float(
            _cfgGet(configData, "matchingParams.gammaScale", constants.MATCHING_DEFAULT_GAMMA_SCALE)
        ),
        nestedRoccoIters=int(
            _cfgGet(
                configData,
                "matchingParams.nestedRoccoIters",
                constants.MATCHING_DEFAULT_NESTED_ROCCO_ITERS,
            )
        ),
        nestedRoccoBudgetScale=float(
            _cfgGet(
                configData,
                "matchingParams.nestedRoccoBudgetScale",
                constants.MATCHING_DEFAULT_NESTED_ROCCO_BUDGET_SCALE,
            )
        ),
        exportFilterUncertaintyMultiplier=float(
            _cfgGet(
                configData,
                "matchingParams.exportFilterUncertaintyMultiplier",
                constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER,
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
    "_cfgHas",
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
