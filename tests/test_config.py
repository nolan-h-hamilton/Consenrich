import textwrap
import logging
from pathlib import Path

import numpy as np
import pytest

from consenrich.consenrich import readConfig
import consenrich.consenrich as consenrich
import consenrich.io as consenrich_io
import consenrich.regions as consenrich_regions
import consenrich.constants as constants
import consenrich.misc_util as misc_util


def writeConfigFile(tmpPath, fileName, yamlText):
    filePath = tmpPath / fileName
    filePath.write_text(textwrap.dedent(yamlText).strip() + "\n", encoding="utf-8")
    return filePath


def setupGenomeFiles(tmpPath, monkeypatch: pytest.MonkeyPatch) -> None:
    chromSizesPath = tmpPath / "testGenome.sizes"
    chromSizesPath.write_text("chrTest\t100000\n", encoding="utf-8")

    def fakeResolveGenomeName(genomeName: str) -> str:
        return genomeName

    def fakeGetGenomeResourceFile(genomeLabel: str, resourceName: str) -> str:
        if resourceName == "sizes":
            return str(chromSizesPath)
        return str(tmpPath / f"{genomeLabel}.{resourceName}.bed")

    monkeypatch.setattr(constants, "resolveGenomeName", fakeResolveGenomeName)
    monkeypatch.setattr(constants, "getGenomeResourceFile", fakeGetGenomeResourceFile)


def setupBamHelpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fakeCheckAlignmentFile(bamPath: str) -> None:
        return None

    def fakeAlignmentFilesArePairedEnd(bamList: list) -> list:
        return [False] * len(bamList)

    monkeypatch.setattr(misc_util, "checkAlignmentFile", fakeCheckAlignmentFile)
    monkeypatch.setattr(
        misc_util,
        "alignmentFilesArePairedEnd",
        fakeAlignmentFilesArePairedEnd,
    )


_CONFIG_SECTION_TO_PARSED_ARGS = {
    "fitParams": "fitArgs",
    "processParams": "processArgs",
    "observationParams": "observationArgs",
    "uncertaintyCalibration": "uncertaintyCalibrationArgs",
    "uncertaintyCalibrationParams": "uncertaintyCalibrationArgs",
}


def _assertParsedConfigValue(parsed, dottedKey: str, expected) -> None:
    section, field = dottedKey.split(".", 1)
    actual = getattr(parsed[_CONFIG_SECTION_TO_PARSED_ARGS[section]], field)
    if isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def _caseRuntimeBackgroundPriorWindowPairsWithRuntimeBlockLength():
    blockLen = consenrich._resolveRuntimeBackgroundBlockLen(
        vec_=(5, 4, 6),
        backgroundBlockSizeIntervals=11,
        lengthScaleMultiplier=8.0,
    )
    assert blockLen == 41
    assert (
        consenrich._resolveRuntimeBackgroundBlockLen(
            vec_=None,
            backgroundBlockSizeIntervals=5,
            lengthScaleMultiplier=8.0,
        )
        == 41
    )
    assert (
        consenrich._resolveRuntimeBackgroundBlockLen(
            vec_=(5, 4, 6),
            backgroundBlockSizeIntervals=11,
            lengthScaleMultiplier=4.0,
        )
        == 21
    )


def _case_readConfigRejectsBackgroundPriorWindowOverride(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_backgroundPriorWindowIntervals: 21
    """
    configPath = writeConfigFile(tmp_path, "config_prior_window_removed.yaml", configYaml)

    with pytest.raises(ValueError, match="EM_backgroundPriorWindowIntervals"):
        readConfig(str(configPath))


def _caseInitialConfigurationSummaryStaysCompact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [sampleA.bam, sampleB.bam, sampleC.bam]
    genomeParams.name: testGenome
    genomeParams.chromosomes: [chrTest]
    """
    configPath = writeConfigFile(tmp_path, "config_summary.yaml", configYaml)
    parsed = readConfig(str(configPath))

    caplog.set_level(logging.INFO, logger=consenrich.logger.name)
    consenrich._logInitialConfigurationSummary(parsed)

    assert "PHASE: INITIAL CONFIGURATION" in caplog.text
    assert "treatment inputs" in caplog.text
    assert "| treatment inputs" in caplog.text
    assert "| 3" in caplog.text
    assert "inputSource(" not in caplog.text
    assert "'countingArgs':" not in caplog.text


def _case_munc_worker_count_unknown_memory_uses_cpu_cap(monkeypatch):
    monkeypatch.setattr(consenrich_io.os, "cpu_count", lambda: 8)

    workers = consenrich_io._getMuncWorkerCount(
        10,
        1000,
        availableMemoryBytes=None,
        logger_=None,
    )

    assert workers == 4


def _case_munc_worker_count_low_memory_keeps_one_worker(monkeypatch):
    monkeypatch.setattr(consenrich_io.os, "cpu_count", lambda: 8)

    workers = consenrich_io._getMuncWorkerCount(
        10,
        1000,
        availableMemoryBytes=64 * 1024 * 1024,
        logger_=None,
    )

    assert workers == 1


def _case_munc_worker_count_moderate_memory_caps_below_cpu(monkeypatch):
    monkeypatch.setattr(consenrich_io.os, "cpu_count", lambda: 16)

    workers = consenrich_io._getMuncWorkerCount(
        10,
        1000,
        availableMemoryBytes=1024 * 1024 * 1024,
        logger_=None,
    )

    assert workers == 4


def _case_ensureInput():
    configYaml = f"""
    experimentName: testExperiment
    genomeParams.name: hg38
    """

    configPath = writeConfigFile(Path("."), "config_no_input.yaml", configYaml)
    try:
        readConfig(str(configPath))
    except ValueError as e:
        return
    else:
        assert False, "Expected ValueError not raised given empty `consenrich.core.inputParams`"


def _case_readConfigDottedAndNestedEquivalent(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    dottedYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam, smallTest2.bam]
    genomeParams.name: testGenome
    genomeParams.excludeChroms: [chrM]
    countingParams.intervalSizeBP: 50
    """

    nestedYaml = """
    experimentName: testExperiment
    inputParams:
      bamFiles:
        - smallTest.bam
        - smallTest2.bam
    genomeParams:
      name: testGenome
      excludeChroms:
        - chrM
    countingParams:
      intervalSizeBP: 50
    """

    dottedPath = writeConfigFile(tmp_path, "config_dotted.yaml", dottedYaml)
    nestedPath = writeConfigFile(tmp_path, "config_nested.yaml", nestedYaml)

    configDotted = readConfig(str(dottedPath))
    configNested = readConfig(str(nestedPath))

    assert configDotted["experimentName"] == "testExperiment"
    assert configNested["experimentName"] == "testExperiment"
    assert configDotted["experimentName"] == configNested["experimentName"]

    inputDotted = configDotted["inputArgs"]
    inputNested = configNested["inputArgs"]

    assert inputDotted.bamFiles == ["smallTest.bam", "smallTest2.bam"]
    assert inputNested.bamFiles == ["smallTest.bam", "smallTest2.bam"]
    assert inputDotted.bamFiles == inputNested.bamFiles

    assert bool(inputDotted.bamFilesControl) is False
    assert bool(inputNested.bamFilesControl) is False

    genomeDotted = configDotted["genomeArgs"]
    genomeNested = configNested["genomeArgs"]

    assert genomeDotted.genomeName == "testGenome"
    assert genomeNested.genomeName == "testGenome"
    assert genomeDotted.genomeName == genomeNested.genomeName

    assert genomeDotted.excludeChroms == ["chrM"]
    assert genomeNested.excludeChroms == ["chrM"]

    assert "chrTest" in genomeDotted.chromosomes
    assert "chrTest" in genomeNested.chromosomes
    assert genomeDotted.chromosomes == genomeNested.chromosomes

    countingDotted = configDotted["countingArgs"]
    countingNested = configNested["countingArgs"]

    assert countingDotted.intervalSizeBP == 50
    assert countingNested.intervalSizeBP == 50
    assert countingDotted.intervalSizeBP == countingNested.intervalSizeBP

    observationDotted = configDotted["observationArgs"]
    observationNested = configNested["observationArgs"]
    processDotted = configDotted["processArgs"]
    processNested = configNested["processArgs"]

    assert type(observationDotted) is type(observationNested)
    assert type(processDotted) is type(processNested)
    assert observationDotted == observationNested
    assert processDotted == processNested

    samDotted = configDotted["samArgs"]
    samNested = configNested["samArgs"]
    matchingDotted = configDotted["matchingArgs"]
    matchingNested = configNested["matchingArgs"]

    assert type(samDotted) is type(samNested)
    assert type(matchingDotted) is type(matchingNested)

    assert samDotted.samThreads == samNested.samThreads
    assert samDotted.defaultCountMode == "coverage"
    assert configDotted["scArgs"].defaultCountMode == "coverage"
    assert configNested["scArgs"].defaultCountMode == "coverage"
    assert matchingDotted.enabled == matchingNested.enabled
    assert matchingDotted.thresholdZ == matchingNested.thresholdZ
    assert matchingDotted.nestedRoccoIters == matchingNested.nestedRoccoIters
    assert matchingDotted.nestedRoccoBudgetScale == matchingNested.nestedRoccoBudgetScale


def _case_readConfigProcessQCalibrationOptions(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    processParams:
      processQCalibration: none
      processQCalibIters: 3
      processQCalibOuterIters: 2
      processQLevelTarget: 0.002
      processQTrendTarget: 0.00002
      processQLevelPriorWeight: 0.25
      processQTrendPriorWeight: 2.5
      precisionMultiplierMin: 0.5
      precisionMultiplierMax: 2.0
    observationParams:
      precisionMultiplierMin: 0.1
      precisionMultiplierMax: 8.0
    """

    configPath = writeConfigFile(tmp_path, "config_process_q.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    processArgs = configParsed["processArgs"]

    assert processArgs.processQCalibration == "none"
    assert processArgs.processQCalibIters == 3
    assert processArgs.processQCalibOuterIters == 2
    assert processArgs.processQLevelTarget == pytest.approx(0.002)
    assert processArgs.processQTrendTarget == pytest.approx(0.00002)
    assert processArgs.processQLevelPriorWeight == pytest.approx(0.25)
    assert processArgs.processQTrendPriorWeight == pytest.approx(2.5)
    assert processArgs.precisionMultiplierMin == pytest.approx(0.5)
    assert processArgs.precisionMultiplierMax == pytest.approx(2.0)
    assert configParsed["observationArgs"].precisionMultiplierMin == pytest.approx(0.1)
    assert configParsed["observationArgs"].precisionMultiplierMax == pytest.approx(8.0)


def _case_readConfigUsesGenericDefaultConfiguration(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    configuration: generic
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_generic_defaults.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["defaultConfiguration"] == "generic"
    defaults = consenrich.DEFAULT_CONFIGURATION_VALUES[
        consenrich.GENERIC_DEFAULT_CONFIGURATION
    ]
    for dottedKey, expected in defaults.items():
        _assertParsedConfigValue(parsed, dottedKey, expected)


def _caseGenericDefaultConfigurationUsesCanonicalUncertaintyKeys():
    defaults = consenrich.DEFAULT_CONFIGURATION_VALUES[
        consenrich.GENERIC_DEFAULT_CONFIGURATION
    ]

    assert defaults["uncertaintyCalibration.enabled"] is True
    assert not any(
        key.startswith("uncertaintyCalibrationParams.") for key in defaults
    )


def _case_readConfigGenericDefaultsStillAllowExplicitOverrides(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    configuration: generic
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_outerIters: 16
    fitParams.EM_backgroundLengthScaleMultiplier: 2.0
    fitParams.EM_backgroundPriorVariancePenaltyShape: 2.5
    processParams.processQTrendPriorWeight: 2.5
    processParams.precisionMultiplierMin: 0.5
    observationParams.precisionMultiplierMax: 4.0
    uncertaintyCalibration.enabled: false
    """

    configPath = writeConfigFile(tmp_path, "config_generic_override.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["defaultConfiguration"] == "generic"
    assert parsed["fitArgs"].EM_outerIters == 16
    assert parsed["fitArgs"].EM_backgroundLengthScaleMultiplier == pytest.approx(2.0)
    assert (
        parsed["fitArgs"].EM_backgroundPriorVariancePenaltyShape
        == pytest.approx(2.5)
    )
    assert parsed["processArgs"].processQTrendPriorWeight == pytest.approx(2.5)
    assert parsed["processArgs"].precisionMultiplierMin == pytest.approx(0.5)
    assert parsed["observationArgs"].precisionMultiplierMax == pytest.approx(4.0)
    assert parsed["uncertaintyCalibrationArgs"].enabled is False


def _case_readConfigRejectsUnknownDefaultConfiguration(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    configuration: narrow
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_unknown_defaults.yaml", configYaml)
    with pytest.raises(ValueError, match="Unsupported default configuration"):
        readConfig(str(configPath))


def _case_readConfigObservationTrendDefaultsRemoveLinearEnvelope(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_trend_defaults.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    observationArgs = configParsed["observationArgs"]
    removed = "EB" + "_minLin"

    assert removed not in observationArgs._fields
    assert observationArgs.trendNumBasis == 60
    assert observationArgs.trendMinObsPerBasis == 25.0
    assert observationArgs.trendMinEdf == 3.0
    assert observationArgs.trendMaxEdf == 30.0
    assert observationArgs.trendLambdaMin == 1.0e-6
    assert observationArgs.trendLambdaMax == 1.0e6
    assert observationArgs.trendLambdaGridSize == 41


def _case_readConfigObservationBlockQuantileDefaultAndOverride(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configDefault = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """
    configDefaultPath = writeConfigFile(
        tmp_path,
        "config_block_quantile_default.yaml",
        configDefault,
    )
    parsedDefault = readConfig(str(configDefaultPath))
    _assertParsedConfigValue(
        parsedDefault,
        "observationParams.blockQuantile",
        consenrich.DEFAULT_CONFIGURATION_VALUES[consenrich.GENERIC_DEFAULT_CONFIGURATION][
            "observationParams.blockQuantile"
        ],
    )

    configExplicit = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.blockQuantile: 0.25
    """
    configExplicitPath = writeConfigFile(
        tmp_path,
        "config_block_quantile_explicit.yaml",
        configExplicit,
    )
    parsedExplicit = readConfig(str(configExplicitPath))
    assert parsedExplicit["observationArgs"].blockQuantile == pytest.approx(0.25)


def _case_readConfigDeduplicatesChromosomes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.chromosomes: [chr1, chr2, chr1, chr2, chr3]
    """

    configPath = writeConfigFile(tmp_path, "config_dedup.yaml", configYaml)
    configParsed = readConfig(str(configPath))

    assert configParsed["genomeArgs"].chromosomes == ["chr1", "chr2", "chr3"]


def _case_readConfigAPNDisablesProcPrecReweight(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_useAPN: true
    fitParams.EM_useProcPrecReweight: true
    """

    configPath = writeConfigFile(tmp_path, "config_apn.yaml", configYaml)
    configParsed = readConfig(str(configPath))

    assert configParsed["fitArgs"].EM_useAPN is True
    assert configParsed["fitArgs"].EM_useProcPrecReweight is False


def _case_readConfigUsesEMUseField(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configFieldYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_use: false
    """
    configFieldPath = writeConfigFile(tmp_path, "config_em_use.yaml", configFieldYaml)
    parsedField = readConfig(str(configFieldPath))
    assert parsedField["fitArgs"].EM_use is False


def _case_readConfigUsesZeroCenterIdentifiabilityFields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configDefaultYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """
    parsedDefault = readConfig(
        str(
            writeConfigFile(
                tmp_path,
                "config_zero_center_default.yaml",
                configDefaultYaml,
            )
        )
    )
    defaultFitArgs = parsedDefault["fitArgs"]
    fitDefaults = consenrich.core.fitParams()
    assert defaultFitArgs.EM_zeroCenterBackground == fitDefaults.EM_zeroCenterBackground
    assert (
        defaultFitArgs.EM_zeroCenterReplicateBias
        == fitDefaults.EM_zeroCenterReplicateBias
    )
    for field in (
        "EM_backgroundLengthScaleMultiplier",
        "EM_useBackgroundPrior",
        "EM_backgroundPriorQuantile",
        "EM_backgroundPriorVariancePenaltyShape",
        "EM_backgroundPriorVariancePenaltyRate",
    ):
        assert hasattr(defaultFitArgs, field)

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_zeroCenterBackground: false
    fitParams.EM_zeroCenterReplicateBias: false
    fitParams.EM_backgroundLengthScaleMultiplier: 6
    fitParams.EM_useBackgroundPrior: false
    fitParams.EM_backgroundPriorQuantile: 0.5
    fitParams.EM_backgroundPriorTrimQuantile: 0.8
    fitParams.EM_backgroundPriorVariance: 0.25
    fitParams.EM_backgroundPriorVariancePenaltyShape: 2.5
    fitParams.EM_backgroundPriorVariancePenaltyRate: 0.01
    """
    parsedOverride = readConfig(
        str(
            writeConfigFile(
                tmp_path,
                "config_zero_center_override.yaml",
                configOverrideYaml,
            )
        )
    )
    assert parsedOverride["fitArgs"].EM_zeroCenterBackground is False
    assert parsedOverride["fitArgs"].EM_zeroCenterReplicateBias is False
    assert parsedOverride["fitArgs"].EM_backgroundLengthScaleMultiplier == pytest.approx(
        6.0
    )
    assert parsedOverride["fitArgs"].EM_useBackgroundPrior is False
    assert parsedOverride["fitArgs"].EM_backgroundPriorQuantile == pytest.approx(0.5)
    assert parsedOverride["fitArgs"].EM_backgroundPriorTrimQuantile == pytest.approx(0.8)
    assert parsedOverride["fitArgs"].EM_backgroundPriorVariance == pytest.approx(0.25)
    assert (
        parsedOverride["fitArgs"].EM_backgroundPriorVariancePenaltyShape
        == pytest.approx(2.5)
    )
    assert (
        parsedOverride["fitArgs"].EM_backgroundPriorVariancePenaltyRate
        == pytest.approx(0.01)
    )


def _case_readConfigDefaultsEMTNuToEightAndAllowsOverride(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configDefaultYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """
    parsedDefault = readConfig(
        str(writeConfigFile(tmp_path, "config_em_tnu_default.yaml", configDefaultYaml))
    )
    assert parsedDefault["fitArgs"].EM_tNu == pytest.approx(8.0)

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_tNu: 4.0
    """
    parsedOverride = readConfig(
        str(writeConfigFile(tmp_path, "config_em_tnu_override.yaml", configOverrideYaml))
    )
    assert parsedOverride["fitArgs"].EM_tNu == pytest.approx(4.0)


def _case_readConfigUsesInnerAndOuterEMToleranceFields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_innerRtol: 1.0e-6
    fitParams.EM_outerRtol: 2.5e-3
    """

    configPath = writeConfigFile(tmp_path, "config_em_tol.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["fitArgs"].EM_innerRtol == pytest.approx(1.0e-6)
    assert parsed["fitArgs"].EM_outerRtol == pytest.approx(2.5e-3)
    assert not hasattr(parsed["fitArgs"], "EM_rtol")


def _case_readConfigUsesUncertaintyCalibrationFields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configDefaultYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """
    parsedDefault = readConfig(
        str(
            writeConfigFile(
                tmp_path,
                "config_uncertainty_calibration_default.yaml",
                configDefaultYaml,
            )
        )
    )
    removedStateFields = tuple(
        "".join(parts)
        for parts in (
            ("effective", "InfoRescale"),
            ("effective", "InfoBlockLengthBP"),
            ("effective", "InfoBandwidthBP"),
        )
    )
    for removedField in removedStateFields:
        assert not hasattr(parsedDefault["stateArgs"], removedField)
    assert parsedDefault["uncertaintyCalibrationArgs"].enabled is True
    assert parsedDefault["uncertaintyCalibrationArgs"].blockSizeBP is None
    assert (
        parsedDefault["uncertaintyCalibrationArgs"].folds
        == consenrich.core.uncertaintyCalibrationParams().folds
    )

    configExplicitYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    uncertaintyCalibration.enabled: false
    uncertaintyCalibration.blockSizeBP: 25000
    uncertaintyCalibration.folds: 3
    uncertaintyCalibration.holdoutFraction: 0.2
    uncertaintyCalibration.maxScores: 1234
    uncertaintyCalibration.targets: [0.5, 0.9]
    """
    parsedExplicit = readConfig(
        str(
            writeConfigFile(
                tmp_path,
                "config_uncertainty_calibration_explicit.yaml",
                configExplicitYaml,
            )
        )
    )
    explicitArgs = parsedExplicit["uncertaintyCalibrationArgs"]
    assert explicitArgs.enabled is False
    assert explicitArgs.blockSizeBP == 25_000
    assert explicitArgs.folds == 3
    assert explicitArgs.holdoutFraction == pytest.approx(0.2)
    assert explicitArgs.maxScores == 1234
    assert explicitArgs.targets == (0.5, 0.9)


def _case_readConfigUncertaintyCalibrationLegacyAliasStillAccepted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    uncertaintyCalibrationParams.enabled: false
    uncertaintyCalibrationParams.maxScores: 4321
    uncertaintyCalibrationParams.targets: [0.5, 0.8]
    """
    parsed = readConfig(
        str(
            writeConfigFile(
                tmp_path,
                "config_uncertainty_calibration_legacy_alias.yaml",
                configYaml,
            )
        )
    )
    args = parsed["uncertaintyCalibrationArgs"]

    assert args.enabled is False
    assert args.maxScores == 4321
    assert args.targets == (0.5, 0.8)


def _case_readConfigNumNearestRequiresExplicitSparseBed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    sparseBedPath = tmp_path / "explicit_sparse.bed"
    sparseBedPath.write_text("chrTest\t0\t100\n", encoding="utf-8")

    configNoExplicit = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.numNearest: 17
    """
    configNoExplicitPath = writeConfigFile(
        tmp_path,
        "config_no_explicit_sparse.yaml",
        configNoExplicit,
    )
    parsedNoExplicit = readConfig(str(configNoExplicitPath))
    assert parsedNoExplicit["observationArgs"].numNearest == 0

    configExplicit = f"""
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.sparseBedFile: {sparseBedPath}
    observationParams.numNearest: 17
    """
    configExplicitPath = writeConfigFile(
        tmp_path,
        "config_explicit_sparse.yaml",
        configExplicit,
    )
    parsedExplicit = readConfig(str(configExplicitPath))
    assert parsedExplicit["observationArgs"].numNearest == 17


def _case_readConfigRestrictLocalAR1ToSparseBedRequiresAvailableSparseBed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    sparseBedPath = tmp_path / "explicit_sparse.bed"
    sparseBedPath.write_text("chrTest\t0\t100\n", encoding="utf-8")

    configNoSparse = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.restrictLocalAR1ToSparseBed: true
    """
    configNoSparsePath = writeConfigFile(
        tmp_path,
        "config_restrict_local_ar1_no_sparse.yaml",
        configNoSparse,
    )
    parsedNoSparse = readConfig(str(configNoSparsePath))
    assert parsedNoSparse["observationArgs"].restrictLocalAR1ToSparseBed is False

    configExplicitSparse = f"""
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.sparseBedFile: {sparseBedPath}
    observationParams.restrictLocalAR1ToSparseBed: true
    """
    configExplicitSparsePath = writeConfigFile(
        tmp_path,
        "config_restrict_local_ar1_explicit_sparse.yaml",
        configExplicitSparse,
    )
    parsedExplicitSparse = readConfig(str(configExplicitSparsePath))
    assert (
        parsedExplicitSparse["observationArgs"].restrictLocalAR1ToSparseBed is True
    )


def _case_loadSparseIntervalIndicesUsesBedSpan(tmp_path):
    sparseBedPath = tmp_path / "sparse_regions.bed"
    sparseBedPath.write_text(
        "\n".join(
            [
                "chrTest\t110\t120",
                "chrTest\t150\t226",
                "chrOther\t0\t1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    intervals = np.arange(0, 300, 50, dtype=np.uint32)

    indices = consenrich_regions._loadSparseIntervalIndices(
        str(sparseBedPath),
        "chrTest",
        intervals,
    )

    assert indices.tolist() == [2, 3, 4]


def _case_readConfigSampleSources(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    fragmentsPath = tmp_path / "smallTest.fragments.tsv.gz"
    fragmentsPath.write_text("", encoding="utf-8")
    groupMapPath = tmp_path / "groups.tsv"
    groupMapPath.write_text("BC_A\tclusterA\n", encoding="utf-8")

    sampleYaml = f"""
    experimentName: sampleExperiment
    inputParams:
      samples:
        - name: trt1
          path: smallTest.bam
          format: bam
          role: treatment
        - name: ctrl1
          path: smallTest2.bam
          format: bam
          role: control
        - name: clusterA
          path: {fragmentsPath}
          format: fragments
          role: treatment
          barcodeGroupMapFile: {groupMapPath}
          selectGroups: [clusterA]
          fragmentPositionMode: fragmentEndpoints
    genomeParams.name: testGenome
    countingParams.normMethod: CPM
    countingParams.fragmentsGroupNorm: CELLS
    """

    configPath = writeConfigFile(tmp_path, "config_samples.yaml", sampleYaml)
    configParsed = readConfig(str(configPath))
    inputArgs = configParsed["inputArgs"]

    assert inputArgs.bamFiles == ["smallTest.bam", str(fragmentsPath)]
    assert inputArgs.bamFilesControl == ["smallTest2.bam", "smallTest2.bam"]
    assert inputArgs.treatmentSources is not None
    assert inputArgs.controlSources is not None
    assert inputArgs.treatmentSources[0].sourceKind == "BAM"
    assert inputArgs.controlSources[0].role == "control"
    assert inputArgs.treatmentSources[0].sampleName == "trt1"
    assert inputArgs.treatmentSources[1].sourceKind == "FRAGMENTS"
    assert inputArgs.treatmentSources[1].selectGroups == ["clusterA"]
    assert inputArgs.treatmentSources[1].fragmentPositionMode == "fragmentEndpoints"
    assert configParsed["countingArgs"].normMethod == "CPM"
    assert configParsed["countingArgs"].fragmentsGroupNorm == "CELLS"


def _case_readConfigSamplesSupportBedGraph(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    bedGraphPath = tmp_path / "sample.bedGraph"
    bedGraphPath.write_text(
        "track type=bedGraph\nbrowser position chrTest:1-50\nchrTest\t0\t50\t3.0\n",
        encoding="ascii",
    )
    indexedPath = Path(f"{bedGraphPath}.gz")

    configYaml = f"""
    experimentName: sampleExperiment
    inputParams:
      samples:
        - path: {bedGraphPath}
          role: treatment
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_bedgraph.yaml", configYaml)
    caplog.set_level(logging.INFO, logger=consenrich.logger.name)
    configParsed = readConfig(str(configPath))
    inputArgs = configParsed["inputArgs"]

    assert inputArgs.treatmentSources is not None
    assert inputArgs.treatmentSources[0].sourceKind == "BEDGRAPH"
    assert inputArgs.bamFiles == [str(indexedPath)]
    assert inputArgs.treatmentSources[0].path == str(indexedPath)
    assert bedGraphPath.exists()
    assert indexedPath.exists()
    assert Path(f"{indexedPath}.tbi").exists()
    assert "has no tabix index" in caplog.text
    counts = consenrich.core.readSegments(
        inputArgs.treatmentSources,
        "chrTest",
        0,
        50,
        50,
        [0],
        [1.0],
        0,
        1,
        0,
    )
    np.testing.assert_allclose(counts, np.array([[3.0]], dtype=np.float32))


def _case_readConfigSamplesSupportExplicitBedGraphFormat(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    signalPath = tmp_path / "sample.signal"
    signalPath.write_text("chrTest\t0\t50\t3.0\n", encoding="ascii")
    indexedPath = Path(f"{signalPath}.gz")

    configYaml = f"""
    experimentName: sampleExperiment
    inputParams:
      samples:
        - path: {signalPath}
          format: bedGraph
          role: treatment
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_bedgraph_format.yaml", configYaml)
    caplog.set_level(logging.INFO, logger=consenrich.logger.name)
    configParsed = readConfig(str(configPath))
    inputArgs = configParsed["inputArgs"]

    assert inputArgs.treatmentSources is not None
    assert inputArgs.treatmentSources[0].sourceKind == "BEDGRAPH"
    assert inputArgs.bamFiles == [str(indexedPath)]
    assert indexedPath.exists()
    assert Path(f"{indexedPath}.tbi").exists()
    assert "has no tabix index" in caplog.text


def _case_readConfigScParamsProvideFragmentsDefaults(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    fragmentsPath = tmp_path / "smallTest.fragments.tsv.gz"
    fragmentsPath.write_text("", encoding="utf-8")

    configYaml = f"""
    experimentName: sampleExperiment
    inputParams:
      samples:
        - path: {fragmentsPath}
          format: fragments
          role: treatment
    genomeParams.name: testGenome
    scParams.defaultCountMode: center
    scParams.fragmentsGroupNorm: CELLS
    scParams.defaultFragmentPositionMode: fragmentEndpoints
    scParams.barcodeTag: CR
    """

    configPath = writeConfigFile(tmp_path, "config_sc_defaults.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    source = configParsed["inputArgs"].treatmentSources[0]

    assert source.countMode is None
    assert source.fragmentPositionMode == "fragmentEndpoints"
    assert source.barcodeTag == "CR"
    assert configParsed["scArgs"].defaultCountMode == "center"
    assert configParsed["scArgs"].fragmentsGroupNorm == "CELLS"
    assert configParsed["countingArgs"].fragmentsGroupNorm == "CELLS"


def _case_resolveExtendFrom5pBPPairsUsesTreatmentValuesForControls():
    treatment, control = consenrich._resolveExtendFrom5pBPPairs(
        [150, 180],
        [90, 110],
    )
    ioTreatment, ioControl = consenrich_io._resolveExtendFrom5pBPPairs(
        [150, 180],
        [90, 110],
    )

    assert treatment == [150, 180]
    assert control == [150, 180]
    assert ioTreatment == treatment
    assert ioControl == control


def _case_readConfigMatchingDefaultsToROCCO(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_matching_default.yaml", configYaml)
    configParsed = readConfig(str(configPath))

    assert configParsed["matchingArgs"].enabled is True
    assert configParsed["matchingArgs"].numBootstrap == 128
    assert configParsed["matchingArgs"].thresholdZ == pytest.approx(2.0)
    assert configParsed["matchingArgs"].gamma == pytest.approx(0.25)
    assert configParsed["matchingArgs"].nestedRoccoIters == 3
    assert configParsed["matchingArgs"].nestedRoccoBudgetScale == pytest.approx(0.5)
    assert not hasattr(configParsed["matchingArgs"], "minMatchLengthBP")
    assert not hasattr(configParsed["matchingArgs"], "merge")
    assert not hasattr(configParsed["matchingArgs"], "mergeGapBP")
    assert configParsed["countingArgs"].intervalSizeBP == 25
    assert not hasattr(configParsed["countingArgs"], "smoothSpanBP")


def _case_readConfigRejectsCRAMSources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams:
      samples:
        - path: sample.cram
          format: cram
          role: treatment
    genomeParams:
      name: hg38
    """

    configPath = writeConfigFile(tmp_path, "config_cram.yaml", configYaml)
    with pytest.raises(ValueError, match="CRAM inputs are no longer supported"):
        readConfig(str(configPath))


def _readBigWigIntervals(path: Path, chroms: list[str]) -> dict[str, list[tuple]]:
    pyBigWig = pytest.importorskip("pyBigWig")
    bw = pyBigWig.open(str(path))
    try:
        return {chrom: list(bw.intervals(chrom) or []) for chrom in chroms}
    finally:
        bw.close()


def _case_convertBedGraphToBigWigPyBigWigWritesExpectedTrack(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")

    bedGraphPath = tmp_path / "toy.bedGraph"
    chromSizesPath = tmp_path / "toy.chrom.sizes"
    pyBigWigPath = tmp_path / "pybigwig.bw"
    bedGraphPath.write_text(
        "\n".join(
                [
                    "track type=bedGraph name=toy",
                    "browser position chr1:1-20",
                    "chr1 0 10 0.5",
                    "chr1\t10\t20\t2.25",
                    "chr2\t0\t8\t2.0",
                    "chr10 0 5 10.0",
                ]
            )
        + "\n",
        encoding="ascii",
    )
    chromSizesPath.write_text(
        "chr1\t100\nchr2\t100\nchr10\t100\n",
        encoding="ascii",
    )

    consenrich._convertBedGraphToBigWigPyBigWig(
        str(bedGraphPath),
        str(chromSizesPath),
        str(pyBigWigPath),
        chunkSize=2,
    )

    chroms = ["chr1", "chr2", "chr10"]
    assert _readBigWigIntervals(pyBigWigPath, chroms) == {
        "chr1": [(0, 10, 0.5), (10, 20, 2.25)],
        "chr2": [(0, 8, 2.0)],
        "chr10": [(0, 5, 10.0)],
    }

    handle = pyBigWig.open(str(pyBigWigPath))
    try:
        header = handle.header()
    finally:
        handle.close()
    assert header["nBasesCovered"] == 33
    assert header["minVal"] == 0
    assert header["maxVal"] == 10
    assert header["sumData"] == 93
    assert header["sumSquared"] == 585


def _case_convertBedGraphToBigWigPyBigWigRejectsOutOfBounds(tmp_path):
    pytest.importorskip("pyBigWig")
    bedGraphPath = tmp_path / "bad.bedGraph"
    chromSizesPath = tmp_path / "bad.chrom.sizes"
    bigWigPath = tmp_path / "bad.bw"
    bedGraphPath.write_text("chr1\t90\t101\t1.0\n", encoding="ascii")
    chromSizesPath.write_text("chr1\t100\n", encoding="ascii")

    with pytest.raises(ValueError, match="exceeds chr1 size"):
        consenrich._convertBedGraphToBigWigPyBigWig(
            str(bedGraphPath),
            str(chromSizesPath),
            str(bigWigPath),
        )
    assert not bigWigPath.exists()


def _case_convertBedGraphToBigWigPyBigWigRejectsEmptyBedGraph(tmp_path):
    pytest.importorskip("pyBigWig")
    bedGraphPath = tmp_path / "empty.bedGraph"
    chromSizesPath = tmp_path / "empty.chrom.sizes"
    bigWigPath = tmp_path / "empty.bw"
    bedGraphPath.write_text(
        "track type=bedGraph name=empty\nbrowser position chr1:1-10\n",
        encoding="ascii",
    )
    chromSizesPath.write_text("chr1\t100\n", encoding="ascii")

    with pytest.raises(ValueError, match="No bedGraph intervals"):
        consenrich._convertBedGraphToBigWigPyBigWig(
            str(bedGraphPath),
            str(chromSizesPath),
            str(bigWigPath),
        )
    assert not bigWigPath.exists()




def _case_sortBedGraphInPlace(tmp_path):
    bedGraphPath = tmp_path / "toy.bedGraph"
    bedGraphPath.write_text(
        "\n".join(
            [
                "chr2\t20\t30\t2.0",
                "chr1\t10\t20\t1.0",
                "chr1\t0\t10\t0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    consenrich._sortBedGraphInPlace(str(bedGraphPath))

    assert bedGraphPath.read_text(encoding="utf-8").splitlines() == [
        "chr1\t0\t10\t0.5",
        "chr1\t10\t20\t1.0",
        "chr2\t20\t30\t2.0",
    ]


def _case_bedGraphValidationAcceptsGenomeOrderAndSortsFallback(tmp_path):
    bedGraphPath = tmp_path / "genome_order.bedGraph"
    bedGraphPath.write_text(
        "\n".join(
            [
                "chr2\t0\t10\t2.0",
                "chr2\t10\t20\t2.5",
                "chr1\t0\t10\t1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    consenrich._validateBedGraphSorted(str(bedGraphPath), chromOrder=["chr2", "chr1"])
    with pytest.raises(ValueError, match="chromosome order"):
        consenrich._validateBedGraphSorted(
            str(bedGraphPath),
            chromOrder=["chr1", "chr2"],
        )

    unsortedPath = tmp_path / "needs_genome_order_sort.bedGraph"
    unsortedPath.write_text(
        "\n".join(
            [
                "track type=bedGraph name=toy",
                "browser position chr2:1-20",
                "chr1\t10\t20\t1.5",
                "chr2\t10\t20\t2.5",
                "chr1\t0\t10\t1.0",
                "chr2\t0\t10\t2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    consenrich._sortBedGraphInPlace(str(unsortedPath), chromOrder=["chr2", "chr1"])

    assert unsortedPath.read_text(encoding="utf-8").splitlines() == [
        "track type=bedGraph name=toy",
        "browser position chr2:1-20",
        "chr2\t0\t10\t2.0000",
        "chr2\t10\t20\t2.5000",
        "chr1\t0\t10\t1.0000",
        "chr1\t10\t20\t1.5000",
    ]


def _case_resolveFixedDeltaFRequiresPositiveFinite():
    assert consenrich.core._resolveFixedDeltaF(0.25) == pytest.approx(0.25)

    for badDeltaF in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="deltaF"):
            consenrich.core._resolveFixedDeltaF(badDeltaF)


def _run_with_monkeypatch(monkeypatch, func, *args):
    with monkeypatch.context() as mp:
        return func(*args, mp)


def test_config_runtime_logging_and_validation_contracts(
    tmp_path, monkeypatch, caplog, contract_case
):
    contract_case(
        "runtime background prior window",
        _caseRuntimeBackgroundPriorWindowPairsWithRuntimeBlockLength,
    )
    contract_case(
        "background prior window override rejected",
        _run_with_monkeypatch,
        monkeypatch,
        _case_readConfigRejectsBackgroundPriorWindowOverride,
        tmp_path,
    )
    caplog.clear()
    with monkeypatch.context() as mp:
        contract_case(
            "compact initial configuration summary",
            _caseInitialConfigurationSummaryStaysCompact,
            tmp_path,
            mp,
            caplog,
        )
    contract_case("fixed deltaF validation", _case_resolveFixedDeltaFRequiresPositiveFinite)


def test_config_worker_and_input_helper_contracts(monkeypatch, contract_case):
    for label, func in (
        ("unknown memory worker cap", _case_munc_worker_count_unknown_memory_uses_cpu_cap),
        ("low memory keeps one worker", _case_munc_worker_count_low_memory_keeps_one_worker),
        ("moderate memory caps workers", _case_munc_worker_count_moderate_memory_caps_below_cpu),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)
    contract_case("input presence validation", _case_ensureInput)
    contract_case(
        "5p extension treatment/control compatibility",
        _case_resolveExtendFrom5pBPPairsUsesTreatmentValuesForControls,
    )


def test_config_parser_defaults_and_override_contracts(
    tmp_path, monkeypatch, contract_case
):
    for label, func in (
        ("dotted and nested config equivalence", _case_readConfigDottedAndNestedEquivalent),
        ("process Q calibration options", _case_readConfigProcessQCalibrationOptions),
        ("generic defaults", _case_readConfigUsesGenericDefaultConfiguration),
        ("generic overrides", _case_readConfigGenericDefaultsStillAllowExplicitOverrides),
        ("unknown default profile rejected", _case_readConfigRejectsUnknownDefaultConfiguration),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)
    contract_case(
        "canonical uncertainty default keys",
        _caseGenericDefaultConfigurationUsesCanonicalUncertaintyKeys,
    )


def test_config_model_parameter_field_contracts(tmp_path, monkeypatch, contract_case):
    for label, func in (
        ("observation trend defaults", _case_readConfigObservationTrendDefaultsRemoveLinearEnvelope),
        ("observation block quantile", _case_readConfigObservationBlockQuantileDefaultAndOverride),
        ("chromosome deduplication", _case_readConfigDeduplicatesChromosomes),
        ("APN disables process precision reweighting", _case_readConfigAPNDisablesProcPrecReweight),
        ("EM use field", _case_readConfigUsesEMUseField),
        ("zero-center identifiability fields", _case_readConfigUsesZeroCenterIdentifiabilityFields),
        ("EM t-nu defaults and override", _case_readConfigDefaultsEMTNuToEightAndAllowsOverride),
        ("inner and outer EM tolerance fields", _case_readConfigUsesInnerAndOuterEMToleranceFields),
        ("uncertainty calibration fields", _case_readConfigUsesUncertaintyCalibrationFields),
        ("legacy uncertainty aliases", _case_readConfigUncertaintyCalibrationLegacyAliasStillAccepted),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)


def test_config_sparse_sample_source_and_matching_contracts(
    tmp_path, monkeypatch, caplog, contract_case
):
    for label, func in (
        ("numNearest sparse-bed requirement", _case_readConfigNumNearestRequiresExplicitSparseBed),
        (
            "restrict local AR1 sparse-bed requirement",
            _case_readConfigRestrictLocalAR1ToSparseBedRequiresAvailableSparseBed,
        ),
        ("structured sample sources", _case_readConfigSampleSources),
        ("single-cell fragments defaults", _case_readConfigScParamsProvideFragmentsDefaults),
        ("matching defaults to ROCCO", _case_readConfigMatchingDefaultsToROCCO),
        ("CRAM sources rejected", _case_readConfigRejectsCRAMSources),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)
    contract_case("sparse interval loading", _case_loadSparseIntervalIndicesUsesBedSpan, tmp_path)
    for label, func in (
        ("bedGraph sample source", _case_readConfigSamplesSupportBedGraph),
        (
            "explicit bedGraph sample source",
            _case_readConfigSamplesSupportExplicitBedGraphFormat,
        ),
    ):
        caplog.clear()
        with monkeypatch.context() as mp:
            contract_case(label, func, tmp_path, mp, caplog)


def test_config_bedgraph_bigwig_io_contracts(tmp_path, contract_case):
    for label, func in (
        ("pyBigWig write", _case_convertBedGraphToBigWigPyBigWigWritesExpectedTrack),
        (
            "pyBigWig out-of-bounds rejection",
            _case_convertBedGraphToBigWigPyBigWigRejectsOutOfBounds,
        ),
        ("pyBigWig empty rejection", _case_convertBedGraphToBigWigPyBigWigRejectsEmptyBedGraph),
        ("bedGraph sort", _case_sortBedGraphInPlace),
        (
            "bedGraph genome-order validation",
            _case_bedGraphValidationAcceptsGenomeOrderAndSortsFallback,
        ),
    ):
        contract_case(label, func, tmp_path)
