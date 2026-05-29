"""YAML/default parsing and CLI configuration contracts."""

from __future__ import annotations

import logging
import os
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

import consenrich.constants as constants
import consenrich.core as core
import consenrich.misc_util as misc_util
from consenrich.genome_covariates import (
    resolve_genome_covariate_feature_config,
    validate_genome_covariate_cache,
)
from . import io as io_helpers
from ._normalization import (
    enum_token_key as _sharedEnumTokenKey,
    normalize_config_enum as _sharedNormalizeConfigEnum,
    normalize_count_transform_method as _sharedNormalizeCountingTransformMethod,
    normalize_matching_uncertainty_score_mode as _sharedNormalizeMatchingUncertaintyScoreMode,
    normalize_process_noise_calibration as _sharedNormalizeProcessNoiseCalibration,
    validate_uncertainty_score_z as _sharedValidateMatchingUncertaintyScoreZ,
)

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

    If given a mapping object, just return it. If given a path, load it as YAML.

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


def _cfgGetFirst(
    configMap: Mapping[str, Any],
    dottedKeys: Sequence[str],
    defaultVal: Any = None,
) -> Any:
    for dottedKey in dottedKeys:
        if _cfgHas(configMap, dottedKey):
            return _cfgGet(configMap, dottedKey)
    return defaultVal


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


def _normalizeOutputDiagnosticTracks(value: Any) -> tuple[str, ...]:
    aliasByKey = {
        "slope": "slope",
        "trend": "slope",
        "prekappaqlevel": "preKappaQLevel",
        "prekappaqtrend": "preKappaQTrend",
        "effectiveqlevel": "effectiveQLevel",
        "effectiveqtrend": "effectiveQTrend",
        "tuncqscale": "tuncQScale",
        "munctrace": "muncTrace",
        "rtrace": "muncTrace",
        "sumgain0": "sumGain0",
        "sumgain1": "sumGain1",
    }
    allTracks = tuple(constants.OUTPUT_DIAGNOSTIC_TRACK_NAMES)
    if value is None or value is False:
        rawItems: list[Any] = []
    elif value is True:
        rawItems = ["all"]
    elif isinstance(value, str):
        text = value.strip()
        rawItems = [] if not text else [item.strip() for item in text.split(",")]
    elif isinstance(value, (list, tuple, set)):
        rawItems = list(value)
    else:
        raise ValueError(
            "`outputParams.diagnosticTracks` must be a list, comma-separated string, "
            "boolean, or null."
        )

    tracks: list[str] = []
    seen: set[str] = set()
    for item in rawItems:
        name = str(item).strip()
        if not name:
            continue
        key = name.replace("_", "").replace("-", "").lower()
        if key in {"none", "false", "off"}:
            continue
        if key == "all":
            for track in allTracks:
                if track not in seen:
                    tracks.append(track)
                    seen.add(track)
            continue
        canonical = aliasByKey.get(key)
        if canonical is None:
            supported = ", ".join(allTracks)
            raise ValueError(
                f"Unsupported output diagnostic track {name!r}. "
                f"Supported tracks: {supported}, or 'all'."
            )
        if canonical not in seen:
            tracks.append(canonical)
            seen.add(canonical)
    return tuple(tracks)


def _normalizeMatchingUncertaintyScoreMode(value: Any) -> str:
    return _sharedNormalizeMatchingUncertaintyScoreMode(value)


def _validateMatchingUncertaintyScoreZ(value: Any) -> float:
    return _sharedValidateMatchingUncertaintyScoreZ(value)


def _enumTokenKey(value: Any) -> str:
    return _sharedEnumTokenKey(value)


def _normalizeConfigEnum(
    value: Any,
    *,
    default: str,
    supported: Sequence[str],
    configName: str,
) -> str:
    return _sharedNormalizeConfigEnum(
        value,
        default=default,
        supported=supported,
        config_name=configName,
    )


def _normalizeCountingTransformMethod(value: Any) -> str:
    return _sharedNormalizeCountingTransformMethod(value)


def _coerceTransformFloat(
    value: Any,
    *,
    name: str,
    default: float | None = None,
    positive: bool = False,
) -> float | None:
    if value is None:
        return default
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return out


def _normalizeMuncCovariatesMode(value: Any) -> str:
    raw = constants.OBSERVATION_DEFAULT_MUNC_COVARIATES_MODE if value is None else value
    key = str(raw).strip().replace("-", "").replace("_", "").lower()
    canonicalByKey = {
        mode.replace("-", "").replace("_", "").lower(): mode
        for mode in constants.MUNC_SUPPORTED_COVARIATE_MODES
    }
    if key not in canonicalByKey:
        supported = ", ".join(constants.MUNC_SUPPORTED_COVARIATE_MODES)
        raise ValueError(
            f"Unsupported observationParams.muncCovariates.mode {raw!r}. "
            f"Supported modes: {supported}."
        )
    return canonicalByKey[key]


def _normalizeMuncCovariateFeatures(
    value: Any,
    *,
    availableFeatures: Sequence[str] | None = None,
) -> tuple[str, ...]:
    return resolve_genome_covariate_feature_config(
        value,
        default_features=constants.OBSERVATION_DEFAULT_MUNC_COVARIATE_FEATURES,
        available_features=availableFeatures,
        config_name="observationParams.muncCovariates.features",
    )


def _normalizeProcessNoiseCalibration(value: Any) -> str:
    return _sharedNormalizeProcessNoiseCalibration(value)


def _normalizeTuncCovariatesMode(value: Any) -> str:
    raw = constants.PROCESS_DEFAULT_TUNC_COVARIATES_MODE if value is None else value
    key = str(raw).strip().replace("-", "").replace("_", "").lower()
    canonicalByKey = {
        mode.replace("-", "").replace("_", "").lower(): mode
        for mode in constants.TUNC_SUPPORTED_COVARIATE_MODES
    }
    if key not in canonicalByKey:
        supported = ", ".join(constants.TUNC_SUPPORTED_COVARIATE_MODES)
        raise ValueError(
            f"Unsupported processParams.tuncProcessCovariates.mode {raw!r}. "
            f"Supported modes: {supported}."
        )
    return canonicalByKey[key]


def _normalizeTuncCovariateFeatures(
    value: Any,
    *,
    availableFeatures: Sequence[str] | None = None,
) -> tuple[str, ...]:
    return resolve_genome_covariate_feature_config(
        value,
        default_features=constants.PROCESS_DEFAULT_TUNC_COVARIATE_FEATURES,
        available_features=availableFeatures,
        config_name="processParams.tuncProcessCovariates.features",
    )


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
    coreExtras = {key: value for key, value in extraValues.items() if key in coreFields}
    publicExtraValues = {
        key: value
        for key, value in {**baseValues, **extraValues}.items()
        if key not in coreFields
    }
    extraFields = tuple(publicExtraValues)
    if not extraFields and len(coreFields) == len(
        getattr(core.processParams, "_fields", ())
    ):
        return core.processParams(**coreValues, **coreExtras)

    baseArgs = core.processParams(**coreValues, **coreExtras)
    fields = coreFields + extraFields
    processArgsType = _runtimeProcessParamsType(fields)
    return processArgsType(
        *(getattr(baseArgs, key) for key in coreFields),
        *(publicExtraValues[key] for key in extraFields),
    )


def getInputArgs(config_path: Union[str, Path, Mapping[str, Any]]) -> core.inputParams:
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


def getOutputArgs(config_path: Union[str, Path, Mapping[str, Any]]) -> core.outputParams:
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
    saveGains_ = _cfgGet(
        configData,
        "outputParams.saveGains",
        _cfgDefault(configData, "outputParams.saveGains"),
    )
    plotOptimizationPath_ = _cfgGet(
        configData,
        "outputParams.plotOptimizationPath",
        constants.OUTPUT_DEFAULT_PLOT_OPTIMIZATION_PATH,
    )
    cutoffReport_ = _cfgGet(
        configData,
        "outputParams.cutoffReport",
        _cfgDefault(configData, "outputParams.cutoffReport"),
    )
    diagnosticTracksRaw = _cfgGet(configData, "outputParams.diagnosticTracks", None)
    if diagnosticTracksRaw is None:
        diagnosticTracksRaw = _cfgGet(configData, "outputParams.tracks", None)
    if diagnosticTracksRaw is None:
        diagnosticTracksRaw = _cfgGet(
            configData,
            "outputParams.writeDiagnosticTracks",
            constants.OUTPUT_DEFAULT_DIAGNOSTIC_TRACKS,
        )
    diagnosticTracks_ = _normalizeOutputDiagnosticTracks(diagnosticTracksRaw)
    return core.outputParams(
        convertToBigWig=convertToBigWig_,
        roundDigits=roundDigits_,
        writeUncertainty=writeUncertainty_,
        saveBackgroundTracks=saveBackgroundTracks_,
        saveGains=saveGains_,
        plotOptimizationPath=plotOptimizationPath_,
        diagnosticTracks=diagnosticTracks_,
        cutoffReport=bool(cutoffReport_),
    )


def getGenomeArgs(config_path: Union[str, Path, Mapping[str, Any]]) -> core.genomeParams:
    configData = loadConfig(config_path)

    genomeName = _cfgGet(configData, "genomeParams.name", constants.GENOME_DEFAULT_NAME)
    genomeLabel = constants.resolveGenomeName(genomeName)

    chromSizesFile: Optional[str] = None
    blacklistFile: Optional[str] = None
    sparseBedFile: Optional[str] = None
    genomeCovariateCacheDir: Optional[str] = None
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

    genomeCovariateCacheDir = _cfgGet(
        configData,
        "genomeParams.genomeCovariateCacheDir",
        _cfgDefault(configData, "genomeParams.genomeCovariateCacheDir"),
    )
    if genomeCovariateCacheDir is not None:
        genomeCovariateCacheDir = str(genomeCovariateCacheDir)

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
        genomeCovariateCacheDir=genomeCovariateCacheDir,
        chromosomes=chromosomesList,
        excludeChroms=excludeChromsList,
        excludeForNorm=excludeForNormList,
    )


def getStateArgs(config_path: Union[str, Path, Mapping[str, Any]]) -> core.stateParams:
    configData = loadConfig(config_path)

    stateInit_ = _cfgGet(
        configData, "stateParams.stateInit", constants.STATE_DEFAULT_INIT
    )
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


def getCountingArgs(config_path: Union[str, Path, Mapping[str, Any]]) -> core.countingParams:
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
    transformMethodRaw = _cfgGetFirst(
        configData,
        (
            "countingParams.transformMethod",
            "countingParams.transform.method",
            "countingParams.countTransform",
        ),
        None,
    )
    if transformMethodRaw is None and _cfgHas(configData, "countingParams.transform"):
        transformConfig = _cfgGet(configData, "countingParams.transform")
        if not isinstance(transformConfig, Mapping):
            transformMethodRaw = transformConfig
    transformMethod_ = _normalizeCountingTransformMethod(
        constants.COUNTING_DEFAULT_TRANSFORM_METHOD
        if transformMethodRaw is None
        else transformMethodRaw
    )
    explicitLogOffset = _cfgHas(configData, "countingParams.logOffset")
    explicitLogMult = _cfgHas(configData, "countingParams.logMult")

    transformInputOffsetRaw = _cfgGetFirst(
        configData,
        (
            "countingParams.transformInputOffset",
            "countingParams.transform.inputOffset",
            "countingParams.transform.offset",
            "countingParams.transformOffset",
        ),
        None,
    )
    if transformInputOffsetRaw is None:
        if transformMethod_ == "log" or explicitLogOffset:
            transformInputOffsetRaw = logOffset_
        elif transformMethod_ == "anscombe":
            transformInputOffsetRaw = constants.COUNTING_ANSCOMBE_INPUT_OFFSET
        else:
            transformInputOffsetRaw = constants.COUNTING_DEFAULT_TRANSFORM_INPUT_OFFSET

    transformInputScaleRaw = _cfgGetFirst(
        configData,
        (
            "countingParams.transformInputScale",
            "countingParams.transform.inputScale",
            "countingParams.transform.scale",
            "countingParams.transformScale",
        ),
        constants.COUNTING_DEFAULT_TRANSFORM_INPUT_SCALE,
    )

    transformOutputScaleRaw = _cfgGetFirst(
        configData,
        (
            "countingParams.transformOutputScale",
            "countingParams.transform.outputScale",
            "countingParams.transform.multiplier",
            "countingParams.transformMultiplier",
            "countingParams.transform.outputMultiplier",
        ),
        None,
    )
    if transformOutputScaleRaw is None:
        if transformMethod_ == "log" or explicitLogMult:
            transformOutputScaleRaw = logMult_
        elif transformMethod_ == "anscombe":
            transformOutputScaleRaw = constants.COUNTING_ANSCOMBE_OUTPUT_SCALE
        else:
            transformOutputScaleRaw = constants.COUNTING_DEFAULT_TRANSFORM_OUTPUT_SCALE

    transformOutputOffsetRaw = _cfgGetFirst(
        configData,
        (
            "countingParams.transformOutputOffset",
            "countingParams.transform.outputOffset",
        ),
        constants.COUNTING_DEFAULT_TRANSFORM_OUTPUT_OFFSET,
    )
    transformShapeRaw = _cfgGetFirst(
        configData,
        (
            "countingParams.transformShape",
            "countingParams.transform.shape",
        ),
        constants.COUNTING_DEFAULT_TRANSFORM_SHAPE,
    )
    transformInputOffset_ = _coerceTransformFloat(
        transformInputOffsetRaw,
        name="countingParams.transformInputOffset",
        default=constants.COUNTING_DEFAULT_TRANSFORM_INPUT_OFFSET,
    )
    transformInputScale_ = _coerceTransformFloat(
        transformInputScaleRaw,
        name="countingParams.transformInputScale",
        default=constants.COUNTING_DEFAULT_TRANSFORM_INPUT_SCALE,
        positive=True,
    )
    transformOutputScale_ = _coerceTransformFloat(
        transformOutputScaleRaw,
        name="countingParams.transformOutputScale",
        default=constants.COUNTING_DEFAULT_TRANSFORM_OUTPUT_SCALE,
    )
    transformOutputOffset_ = _coerceTransformFloat(
        transformOutputOffsetRaw,
        name="countingParams.transformOutputOffset",
        default=constants.COUNTING_DEFAULT_TRANSFORM_OUTPUT_OFFSET,
    )
    transformShape_ = _coerceTransformFloat(
        transformShapeRaw,
        name="countingParams.transformShape",
        default=constants.COUNTING_DEFAULT_TRANSFORM_SHAPE,
        positive=True,
    )
    subtractGlobalMedian_ = _cfgGet(
        configData,
        "countingParams.subtractGlobalMedian",
        _cfgDefault(configData, "countingParams.subtractGlobalMedian"),
    )
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
        transformMethod=transformMethod_,
        transformInputOffset=transformInputOffset_,
        transformInputScale=transformInputScale_,
        transformOutputScale=transformOutputScale_,
        transformOutputOffset=transformOutputOffset_,
        transformShape=transformShape_,
        subtractGlobalMedian=bool(subtractGlobalMedian_),
    )


def getScArgs(config_path: Union[str, Path, Mapping[str, Any]]) -> core.scParams:
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
    config_path: Union[str, Path, Mapping[str, Any]],
) -> core.uncertaintyCalibrationParams:
    configData = loadConfig(config_path)
    enabledDefault = _cfgDefault(configData, "uncertaintyCalibrationParams.enabled")
    blockDefault = constants.UNCERTAINTY_CALIBRATION_DEFAULT_BLOCK_SIZE_BP
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
    mode = _normalizeConfigEnum(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.mode",
            _cfgDefault(configData, "uncertaintyCalibrationParams.mode"),
        ),
        default=constants.UNCERTAINTY_CALIBRATION_DEFAULT_MODE,
        supported=constants.UNCERTAINTY_CALIBRATION_MODES,
        configName="uncertaintyCalibrationParams.mode",
    )
    deleteBlockVarianceMode = _normalizeConfigEnum(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockVarianceMode",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockVarianceMode",
            ),
        ),
        default=constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_VARIANCE_MODE,
        supported=constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_VARIANCE_MODES,
        configName="uncertaintyCalibrationParams.deleteBlockVarianceMode",
    )
    deleteBlockTargetSignal = _normalizeConfigEnum(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockTargetSignal",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockTargetSignal",
            ),
        ),
        default=constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_TARGET_SIGNAL,
        supported=constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_TARGET_SIGNALS,
        configName="uncertaintyCalibrationParams.deleteBlockTargetSignal",
    )
    deleteBlockFactorModel = _normalizeConfigEnum(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockFactorModel",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockFactorModel",
            ),
        ),
        default=constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_MODEL,
        supported=constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_MODELS,
        configName="uncertaintyCalibrationParams.deleteBlockFactorModel",
    )
    deleteBlockScoreWeightMode = _normalizeConfigEnum(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockScoreWeightMode",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockScoreWeightMode",
            ),
        ),
        default=constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_SCORE_WEIGHT_MODE,
        supported=constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_SCORE_WEIGHT_MODES,
        configName="uncertaintyCalibrationParams.deleteBlockScoreWeightMode",
    )
    minInformationFraction = float(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockMinInformationFraction",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockMinInformationFraction",
            ),
        )
    )
    maxInformationFraction = float(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockMaxInformationFraction",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockMaxInformationFraction",
            ),
        )
    )
    if not (0.0 < minInformationFraction < maxInformationFraction < 1.0):
        raise ValueError(
            "uncertaintyCalibrationParams delete-block information fractions must "
            "satisfy 0 < min < max < 1"
        )
    minDeltaVariance = float(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockMinDeltaVariance",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockMinDeltaVariance",
            ),
        )
    )
    if not (np.isfinite(minDeltaVariance) and minDeltaVariance > 0.0):
        raise ValueError(
            "uncertaintyCalibrationParams.deleteBlockMinDeltaVariance must be positive"
        )
    fallbackMinValidFraction = float(
        _cfgGet(
            configData,
            "uncertaintyCalibrationParams.deleteBlockFallbackMinValidFraction",
            _cfgDefault(
                configData,
                "uncertaintyCalibrationParams.deleteBlockFallbackMinValidFraction",
            ),
        )
    )
    if not (0.0 <= fallbackMinValidFraction <= 1.0):
        raise ValueError(
            "uncertaintyCalibrationParams.deleteBlockFallbackMinValidFraction "
            "must be in [0, 1]"
        )
    applyTargetRaw = _cfgGet(
        configData,
        "uncertaintyCalibrationParams.deleteBlockApplyTargetCalibration",
        _cfgDefault(
            configData,
            "uncertaintyCalibrationParams.deleteBlockApplyTargetCalibration",
        ),
    )
    return core.uncertaintyCalibrationParams(
        enabled=bool(enabledConfig),
        mode=mode,
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
        deleteBlockVarianceMode=deleteBlockVarianceMode,
        deleteBlockUseLambdaInInformation=bool(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.deleteBlockUseLambdaInInformation",
                _cfgDefault(
                    configData,
                    "uncertaintyCalibrationParams.deleteBlockUseLambdaInInformation",
                ),
            )
        ),
        deleteBlockTargetSignal=deleteBlockTargetSignal,
        deleteBlockFactorModel=deleteBlockFactorModel,
        deleteBlockMinInformationFraction=minInformationFraction,
        deleteBlockMaxInformationFraction=maxInformationFraction,
        deleteBlockMinDeltaVariance=minDeltaVariance,
        deleteBlockFallbackMinValidFraction=fallbackMinValidFraction,
        deleteBlockScoreWeightMode=deleteBlockScoreWeightMode,
        deleteBlockApplyTargetCalibration=(
            None if applyTargetRaw is None else bool(applyTargetRaw)
        ),
        seed=int(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.seed",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
            )
        ),
        writeDiagnostics=bool(
            _cfgGet(
                configData,
                "uncertaintyCalibrationParams.writeDiagnostics",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_WRITE_DIAGNOSTICS,
            )
        ),
    )


def readConfig(config_path: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    r"""Read and parse the configuration file for Consenrich.

    :param config_path: Path to the YAML configuration file.
    :return: Dictionary containing all parsed configuration parameters.
    """
    configData = loadConfig(config_path)
    defaultConfiguration = _getDefaultConfigurationName(configData)

    inputParams = getInputArgs(configData)
    outputParams = getOutputArgs(configData)
    genomeParams = getGenomeArgs(configData)
    stateParams = getStateArgs(configData)
    countingParams = getCountingArgs(configData)
    scaleFactors = io_helpers._normalizeScaleFactorList(
        countingParams.scaleFactors,
        len(inputParams.bamFiles),
        "countingParams.scaleFactors",
    )
    scaleFactorsControl = countingParams.scaleFactorsControl
    if len(inputParams.bamFilesControl) > 0:
        scaleFactorsControl = io_helpers._normalizeScaleFactorList(
            countingParams.scaleFactorsControl,
            len(inputParams.bamFilesControl),
            "countingParams.scaleFactorsControl",
        )
    countingParams = countingParams._replace(
        scaleFactors=scaleFactors,
        scaleFactorsControl=scaleFactorsControl,
    )
    scArgs = getScArgs(configData)
    uncertaintyCalibrationArgs = getUncertaintyCalibrationArgs(configData)
    experimentName = _cfgGet(
        configData,
        "experimentName",
        constants.EXPERIMENT_DEFAULT_NAME,
    )
    processNoiseCalibration = _normalizeProcessNoiseCalibration(
        _cfgGet(
            configData,
            "processParams.processNoiseCalibration",
            _cfgDefault(configData, "processParams.processNoiseCalibration"),
        )
    )
    tuncLocalWindowMultiplier = _coerceTransformFloat(
        _cfgGet(
            configData,
            "processParams.tuncLocalWindowMultiplier",
            _cfgDefault(configData, "processParams.tuncLocalWindowMultiplier"),
        ),
        name="processParams.tuncLocalWindowMultiplier",
        positive=True,
    )
    tuncDependenceMultiplier = _coerceTransformFloat(
        _cfgGet(
            configData,
            "processParams.tuncDependenceMultiplier",
            _cfgDefault(configData, "processParams.tuncDependenceMultiplier"),
        ),
        name="processParams.tuncDependenceMultiplier",
        positive=True,
    )
    tuncMinScale = _coerceTransformFloat(
        _cfgGet(
            configData,
            "processParams.tuncMinScale",
            _cfgDefault(configData, "processParams.tuncMinScale"),
        ),
        name="processParams.tuncMinScale",
        positive=True,
    )
    tuncMaxScale = _coerceTransformFloat(
        _cfgGet(
            configData,
            "processParams.tuncMaxScale",
            _cfgDefault(configData, "processParams.tuncMaxScale"),
        ),
        name="processParams.tuncMaxScale",
        positive=True,
    )
    if float(tuncMaxScale) < float(tuncMinScale):
        raise ValueError(
            "`processParams.tuncMaxScale` must be greater than or equal to "
            "`processParams.tuncMinScale`."
        )
    tuncMinWindowWeight = _coerceTransformFloat(
        _cfgGet(
            configData,
            "processParams.tuncMinWindowWeight",
            _cfgDefault(configData, "processParams.tuncMinWindowWeight"),
        ),
        name="processParams.tuncMinWindowWeight",
        positive=True,
    )
    tuncPriorRidge = _coerceTransformFloat(
        _cfgGet(
            configData,
            "processParams.tuncPriorRidge",
            _cfgDefault(configData, "processParams.tuncPriorRidge"),
        ),
        name="processParams.tuncPriorRidge",
    )
    if float(tuncPriorRidge) < 0.0:
        raise ValueError("`processParams.tuncPriorRidge` must be non-negative.")
    tuncProcessCovariatesEnabled = bool(
        _cfgGet(
            configData,
            "processParams.tuncProcessCovariates.enabled",
            _cfgDefault(configData, "processParams.tuncProcessCovariates.enabled"),
        )
    )
    tuncProcessCovariatesMode = _normalizeTuncCovariatesMode(
        _cfgGet(
            configData,
            "processParams.tuncProcessCovariates.mode",
            _cfgDefault(configData, "processParams.tuncProcessCovariates.mode"),
        )
    )
    tuncProcessCovariatesFeaturesRaw = _cfgGet(
        configData,
        "processParams.tuncProcessCovariates.features",
        _cfgDefault(configData, "processParams.tuncProcessCovariates.features"),
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
    processGenomeCovariateValidation = None
    if tuncProcessCovariatesEnabled:
        if not genomeParams.genomeCovariateCacheDir:
            raise ValueError(
                "`genomeParams.genomeCovariateCacheDir` is required when "
                "`processParams.tuncProcessCovariates.enabled` is true."
            )
        processGenomeCovariateValidation = validate_genome_covariate_cache(
            genomeParams.genomeCovariateCacheDir,
            interval_size_bp=countingParams.intervalSizeBP,
        )
    tuncProcessCovariatesFeatures = _normalizeTuncCovariateFeatures(
        tuncProcessCovariatesFeaturesRaw,
        availableFeatures=(
            processGenomeCovariateValidation.features
            if processGenomeCovariateValidation is not None
            else None
        ),
    )
    if tuncProcessCovariatesEnabled and not tuncProcessCovariatesFeatures:
        raise ValueError(
            "`processParams.tuncProcessCovariates.features` must select at least "
            "one feature when TUNC process covariates are enabled."
        )
    if tuncProcessCovariatesEnabled and processGenomeCovariateValidation is not None:
        processGenomeCovariateValidation.validate_request(
            required_features=tuncProcessCovariatesFeatures,
            interval_size_bp=countingParams.intervalSizeBP,
            required_features_label="requested TUNC process features",
        )
    processArgs = _buildProcessArgs(
        {
            "deltaF": _cfgGet(
                configData,
                "processParams.deltaF",
                _cfgDefault(configData, "processParams.deltaF"),
            ),
            "stateModel": _cfgGet(
                configData,
                "processParams.stateModel",
                _cfgDefault(configData, "processParams.stateModel"),
            ),
            "minQ": _cfgGet(
                configData,
                "processParams.minQ",
                _cfgDefault(configData, "processParams.minQ"),
            ),
            "maxQ": _cfgGet(
                configData,
                "processParams.maxQ",
                _cfgDefault(configData, "processParams.maxQ"),
            ),
            "processNoiseCalibration": processNoiseCalibration,
            "tuncLocalWindowMultiplier": tuncLocalWindowMultiplier,
            "tuncDependenceMultiplier": tuncDependenceMultiplier,
            "tuncMinScale": tuncMinScale,
            "tuncMaxScale": tuncMaxScale,
            "tuncMinWindowWeight": tuncMinWindowWeight,
            "tuncPriorRidge": tuncPriorRidge,
            "tuncProcessCovariatesEnabled": tuncProcessCovariatesEnabled,
            "tuncProcessCovariatesMode": tuncProcessCovariatesMode,
            "tuncProcessCovariatesFeatures": tuncProcessCovariatesFeatures,
            "processNoiseWarmupECMIters": processNoiseWarmupECMIters,
            "processNoiseWarmupOuterPasses": processNoiseWarmupOuterPasses,
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
        {},
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
    muncAR1VarianceFunctional = core._normalizeMuncAR1VarianceFunctional(
        _cfgGetFirst(
            configData,
            (
                "observationParams.muncAR1VarianceFunctional",
                "observationParams.muncAR1ObservationVarianceFunctional",
            ),
            _cfgDefault(configData, "observationParams.muncAR1VarianceFunctional"),
        )
    )
    muncTrendBlockSizeBP = (
        _cfgGet(configData, "observationParams.muncTrendBlockSizeBP", None)
        if _cfgHas(configData, "observationParams.muncTrendBlockSizeBP")
        else _cfgDefault(configData, "observationParams.muncTrendBlockSizeBP")
    )
    if (
        _cfgHas(configData, "observationParams.muncTrendBlockSizeBP")
        and muncTrendBlockSizeBP is None
    ):
        muncTrendBlockSizeBP = -1
    muncLocalWindowSizeBP = (
        _cfgGet(configData, "observationParams.muncLocalWindowSizeBP", None)
        if _cfgHas(configData, "observationParams.muncLocalWindowSizeBP")
        else _cfgDefault(configData, "observationParams.muncLocalWindowSizeBP")
    )
    if (
        _cfgHas(configData, "observationParams.muncLocalWindowSizeBP")
        and muncLocalWindowSizeBP is None
    ):
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
    muncTrendBlockDependenceMultiplier = float(muncTrendBlockDependenceMultiplierRaw)
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
    muncLocalWindowDependenceMultiplier = float(muncLocalWindowDependenceMultiplierRaw)
    if (
        not np.isfinite(muncLocalWindowDependenceMultiplier)
        or muncLocalWindowDependenceMultiplier <= 0.0
    ):
        raise ValueError(
            "`observationParams.muncLocalWindowDependenceMultiplier` must be positive."
        )
    muncCovariatesEnabled = bool(
        _cfgGet(
            configData,
            "observationParams.muncCovariates.enabled",
            _cfgDefault(configData, "observationParams.muncCovariates.enabled"),
        )
    )
    muncCovariatesMode = _normalizeMuncCovariatesMode(
        _cfgGet(
            configData,
            "observationParams.muncCovariates.mode",
            _cfgDefault(configData, "observationParams.muncCovariates.mode"),
        )
    )
    muncCovariatesFeaturesRaw = _cfgGet(
        configData,
        "observationParams.muncCovariates.features",
        _cfgDefault(configData, "observationParams.muncCovariates.features"),
    )
    genomeCovariateValidation = None
    if muncCovariatesEnabled:
        if not genomeParams.genomeCovariateCacheDir:
            raise ValueError(
                "`genomeParams.genomeCovariateCacheDir` is required when "
                "`observationParams.muncCovariates.enabled` is true."
            )
        genomeCovariateValidation = validate_genome_covariate_cache(
            genomeParams.genomeCovariateCacheDir,
            interval_size_bp=countingParams.intervalSizeBP,
        )
    muncCovariatesFeatures = _normalizeMuncCovariateFeatures(
        muncCovariatesFeaturesRaw,
        availableFeatures=(
            genomeCovariateValidation.features
            if genomeCovariateValidation is not None
            else None
        ),
    )
    if muncCovariatesEnabled and genomeCovariateValidation is not None:
        genomeCovariateValidation.validate_request(
            required_features=muncCovariatesFeatures,
            interval_size_bp=countingParams.intervalSizeBP,
            required_features_label="requested MUNC features",
        )
    observationArgs = core.observationParams(
        minR=_cfgGet(
            configData, "observationParams.minR", constants.OBSERVATION_DEFAULT_MIN_R
        ),
        maxR=_cfgGet(
            configData, "observationParams.maxR", constants.OBSERVATION_DEFAULT_MAX_R
        ),
        samplingIters=_cfgGet(
            configData,
            "observationParams.samplingIters",
            constants.OBSERVATION_DEFAULT_SAMPLING_ITERS,
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
        pad=_cfgGet(
            configData, "observationParams.pad", constants.OBSERVATION_DEFAULT_PAD
        ),
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
        muncAR1VarianceFunctional=muncAR1VarianceFunctional,
        muncTrendBlockSizeBP=muncTrendBlockSizeBP,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP,
        muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier,
        muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier,
        restrictLocalVarianceToSparseBed=restrictLocalVarianceResolved,
        muncCovariatesEnabled=muncCovariatesEnabled,
        muncCovariatesMode=muncCovariatesMode,
        muncCovariatesFeatures=muncCovariatesFeatures,
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
            _cfgDefault(
                configData,
                "fitParams.ECM_backgroundLengthScaleMultiplier",
            ),
        ),
    )

    samThreads = _cfgGet(
        configData, "samParams.samThreads", constants.SAM_DEFAULT_THREADS
    )
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
    chunkSize = _cfgGet(
        configData, "samParams.chunkSize", constants.SAM_DEFAULT_CHUNK_SIZE
    )
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
            _cfgGet(
                configData, "matchingParams.enabled", constants.MATCHING_DEFAULT_ENABLED
            )
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
        gamma=_cfgGet(
            configData, "matchingParams.gamma", constants.MATCHING_DEFAULT_GAMMA
        ),
        selectionPenalty=_cfgGet(
            configData,
            "matchingParams.selectionPenalty",
            constants.MATCHING_DEFAULT_SELECTION_PENALTY,
        ),
        gammaScale=float(
            _cfgGet(
                configData,
                "matchingParams.gammaScale",
                constants.MATCHING_DEFAULT_GAMMA_SCALE,
            )
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
        uncertaintyScoreMode=_normalizeMatchingUncertaintyScoreMode(
            _cfgGet(
                configData,
                "matchingParams.uncertaintyScoreMode",
                constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
            )
        ),
        uncertaintyScoreZ=_validateMatchingUncertaintyScoreZ(
            _cfgGet(
                configData,
                "matchingParams.uncertaintyScoreZ",
                constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
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
