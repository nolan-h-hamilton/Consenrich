import textwrap
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from consenrich.config import readConfig
import consenrich.consenrich as consenrich_cli
import consenrich.config as consenrich_config
import consenrich.core as consenrich_core
import consenrich.io as consenrich_io
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


def _caseRuntimeBackgroundSpanUsesLengthScaleMultiplier():
    coarseMinSpan, coarseMaxSpan = consenrich_cli._dependenceSpanBoundsFromContextBP(50)
    fineMinSpan, fineMaxSpan = consenrich_cli._dependenceSpanBoundsFromContextBP(25)
    assert fineMinSpan >= coarseMinSpan
    assert fineMaxSpan >= coarseMaxSpan
    assert abs(2 * coarseMaxSpan * 50 - 2 * fineMaxSpan * 25) <= 50
    contextBP = 3701
    coarseLen = consenrich_cli._resolveRuntimeBackgroundBlockLen(
        dependenceContextBP=contextBP,
        backgroundBlockSizeBP=-1,
        intervalSizeBP=50,
        lengthScaleMultiplier=16.0,
    )
    fineLen = consenrich_cli._resolveRuntimeBackgroundBlockLen(
        dependenceContextBP=contextBP,
        backgroundBlockSizeBP=-1,
        intervalSizeBP=25,
        lengthScaleMultiplier=16.0,
    )
    assert abs(coarseLen * 50 - fineLen * 25) <= 50

    blockLen = consenrich_cli._resolveRuntimeBackgroundBlockLen(
        dependenceContextBP=501,
        backgroundBlockSizeBP=550,
        intervalSizeBP=50,
        lengthScaleMultiplier=8.0,
    )
    assert blockLen == 41
    assert (
        consenrich_cli._resolveRuntimeBackgroundBlockLen(
            dependenceContextBP=None,
            backgroundBlockSizeBP=250,
            intervalSizeBP=50,
            lengthScaleMultiplier=8.0,
        )
        == 41
    )
    assert (
        consenrich_cli._resolveRuntimeBackgroundBlockLen(
            dependenceContextBP=501,
            backgroundBlockSizeBP=550,
            intervalSizeBP=50,
            lengthScaleMultiplier=4.0,
        )
        == 21
    )
    assert (
        consenrich_cli._resolveRuntimeReplicateDetrendWindow(
            dependenceContextBP=501,
            backgroundBlockSizeBP=550,
            intervalSizeBP=50,
            lengthScaleMultiplier=8.0,
            windowMultiplier=2.0,
        )
        == 81
    )
    assert (
        consenrich_cli._resolveRuntimeReplicateDetrendWindow(
            dependenceContextBP=None,
            backgroundBlockSizeBP=250,
            intervalSizeBP=50,
            lengthScaleMultiplier=4.0,
            windowMultiplier=2.0,
        )
        == 41
    )


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

    caplog.set_level(logging.INFO, logger=consenrich_cli.logger.name)
    consenrich_cli._logInitialConfigurationSummary(parsed)

    assert "PHASE: INITIAL CONFIGURATION" in caplog.text
    assert "treatment inputs" in caplog.text
    assert "| treatment inputs" in caplog.text
    assert "| 3" in caplog.text
    assert "inputSource(" not in caplog.text
    assert "'countingArgs':" not in caplog.text


def _caseReplicateGainFrameShowsIndentedIdFileMeanMedianSdAndIqr():
    sources = [
        consenrich_core.inputSource(
            path="/tmp/sampleA.bam",
            sourceKind="BAM",
            sampleName="sampleA",
        ),
        consenrich_core.inputSource(
            path="/tmp/sampleB.bam",
            sourceKind="BAM",
            sampleName="sampleB",
        ),
    ]
    frame = consenrich_cli._formatReplicateGainFrame(
        "chrTest",
        sources,
        [0.125, 0.25],
        [0.1, 0.2],
        [0.0125, 0.025],
        [0.05, 0.075],
        indentLevel=1,
    )

    assert frame.startswith("      +")
    assert "FINAL FORWARD-PASS GAINS [chrTest]" in frame
    assert "| mean" in frame
    assert "| median" in frame
    assert "| sd" in frame
    assert "| IQR" in frame
    assert "sampleA" in frame
    assert "/tmp/sampleA.bam" in frame
    assert "0.125" in frame
    assert "0.1" in frame
    assert "0.0125" in frame
    assert "0.075" in frame
    assert all(line.startswith(("      +", "      |")) for line in frame.splitlines())


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
        assert False, "Expected ValueError not raised given empty `consenrich_core.inputParams`"


def _case_readConfigDottedAndNestedEquivalent(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    dottedYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam, smallTest2.bam]
    genomeParams.name: testGenome
    genomeParams.excludeChroms: [chrM]
    countingParams.intervalSizeBP: 50
    outputParams.plotOptimizationPath: false
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
    outputParams:
      plotOptimizationPath: false
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

    outputDotted = configDotted["outputArgs"]
    outputNested = configNested["outputArgs"]
    assert outputDotted.plotOptimizationPath is False
    assert outputNested.plotOptimizationPath is False
    assert outputDotted == outputNested

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


def _case_readConfigProcessNoiseOptions(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    processParams:
      stateModel: level
      regularizationStrength: 0.25
      regularizationRatio: 0.005
      processNoiseWarmupECMIters: 7
      processQWarmupECMIters: 3
      precisionMultiplierMin: 0.5
      precisionMultiplierMax: 2.0
    observationParams:
      precisionMultiplierMin: 0.1
      precisionMultiplierMax: 8.0
      useReplicateTrends: true
    """

    configPath = writeConfigFile(tmp_path, "config_process_noise.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    processArgs = configParsed["processArgs"]

    assert processArgs.stateModel == constants.STATE_MODEL_LEVEL
    assert processArgs.regularizationStrength == pytest.approx(0.25)
    assert processArgs.regularizationRatio == pytest.approx(0.005)
    assert processArgs.processNoiseWarmupECMIters == 7
    assert not hasattr(processArgs, "processQWarmupECMIters")
    assert processArgs.precisionMultiplierMin == pytest.approx(0.5)
    assert processArgs.precisionMultiplierMax == pytest.approx(2.0)
    assert configParsed["observationArgs"].precisionMultiplierMin == pytest.approx(0.1)
    assert configParsed["observationArgs"].precisionMultiplierMax == pytest.approx(8.0)
    assert configParsed["observationArgs"].useReplicateTrends is True
    with pytest.raises(TypeError):
        consenrich_core.constructMatrixQ(1.0e-4, offDiagQ=0.0)
    with pytest.raises(TypeError):
        consenrich_core.runConsenrich(
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            1.0,
            1.0e-4,
            1.0,
            offDiagQ=0.0,
            stateInit=0.0,
            stateCovarInit=1.0,
            boundState=False,
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            blockLenIntervals=1,
        )


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
    assert parsed["processArgs"].stateModel == constants.STATE_MODEL_LEVEL_TREND
    assert (
        parsed["processArgs"].processNoiseWarmupECMIters
        == constants.PROCESS_NOISE_DEFAULT_WARMUP_ECM_ITERS
    )
    assert parsed["processArgs"].regularizationStrength == pytest.approx(
        constants.PROCESS_NOISE_DEFAULT_REGULARIZATION_STRENGTH
    )
    assert parsed["processArgs"].regularizationRatio == pytest.approx(
        constants.PROCESS_NOISE_DEFAULT_REGULARIZATION_RATIO
    )
    assert (
        parsed["observationArgs"].useReplicateTrends
        is constants.OBSERVATION_DEFAULT_USE_REPLICATE_TRENDS
    )


def _caseGenericDefaultConfigurationUsesCanonicalUncertaintyKeys():
    defaults = consenrich_config.DEFAULT_CONFIGURATION_VALUES[
        consenrich_config.GENERIC_DEFAULT_CONFIGURATION
    ]

    assert "uncertaintyCalibrationParams.enabled" in defaults
    assert not any(key.startswith("uncertaintyCalibration.") for key in defaults)


def _case_runtime_defaults_are_centralized(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: centralizedDefaults
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """
    configPath = writeConfigFile(tmp_path, "config_centralized_defaults.yaml", configYaml)
    parsed = readConfig(str(configPath))
    profile = constants.DEFAULT_CONFIGURATION_VALUES[
        constants.GENERIC_DEFAULT_CONFIGURATION
    ]

    assert consenrich_config.DEFAULT_CONFIGURATION_VALUES is constants.DEFAULT_CONFIGURATION_VALUES
    assert consenrich_config.GENERIC_DEFAULT_CONFIGURATION is constants.GENERIC_DEFAULT_CONFIGURATION
    assert (
        consenrich_config.SUPPORTED_DEFAULT_CONFIGURATIONS
        is constants.SUPPORTED_DEFAULT_CONFIGURATIONS
    )
    assert consenrich_config.DEFAULT_CONFIGURATION_KEYS is constants.DEFAULT_CONFIGURATION_KEYS

    assert parsed["defaultConfiguration"] == constants.GENERIC_DEFAULT_CONFIGURATION
    assert parsed["processArgs"].stateModel == profile["processParams.stateModel"]
    assert parsed["processArgs"].regularizationStrength == profile[
        "processParams.regularizationStrength"
    ]
    assert parsed["processArgs"].regularizationRatio == profile[
        "processParams.regularizationRatio"
    ]
    assert parsed["processArgs"].processNoiseWarmupECMIters == profile[
        "processParams.processNoiseWarmupECMIters"
    ]
    assert parsed["fitArgs"].ECM_outerIters == profile["fitParams.ECM_outerIters"]
    assert parsed["fitArgs"].ECM_backgroundLengthScaleMultiplier == profile[
        "fitParams.ECM_backgroundLengthScaleMultiplier"
    ]
    assert parsed["fitArgs"].useNonnegativeBackground == profile[
        "fitParams.useNonnegativeBackground"
    ]
    assert parsed["fitArgs"].backgroundNegativePenaltyMultiplier == profile[
        "fitParams.backgroundNegativePenaltyMultiplier"
    ]
    assert parsed["observationArgs"].muncVarianceModel == profile[
        "observationParams.muncVarianceModel"
    ]
    assert parsed["observationArgs"].muncTrendBlockSizeBP == profile[
        "observationParams.muncTrendBlockSizeBP"
    ]
    assert parsed["observationArgs"].muncLocalWindowSizeBP == profile[
        "observationParams.muncLocalWindowSizeBP"
    ]
    assert parsed["observationArgs"].muncTrendBlockDependenceMultiplier == profile[
        "observationParams.muncTrendBlockDependenceMultiplier"
    ]
    assert parsed["observationArgs"].muncLocalWindowDependenceMultiplier == profile[
        "observationParams.muncLocalWindowDependenceMultiplier"
    ]
    assert parsed["observationArgs"].restrictLocalVarianceToSparseBed == profile[
        "observationParams.restrictLocalVarianceToSparseBed"
    ]
    assert consenrich_core.observationParams(
        minR=parsed["observationArgs"].minR,
        maxR=parsed["observationArgs"].maxR,
        samplingIters=parsed["observationArgs"].samplingIters,
        samplingBlockSizeBP=parsed["observationArgs"].samplingBlockSizeBP,
        EB_use=parsed["observationArgs"].EB_use,
        EB_setNu0=parsed["observationArgs"].EB_setNu0,
        EB_setNuL=parsed["observationArgs"].EB_setNuL,
        trendNumBasis=parsed["observationArgs"].trendNumBasis,
        trendMinObsPerBasis=parsed["observationArgs"].trendMinObsPerBasis,
        trendMinEdf=parsed["observationArgs"].trendMinEdf,
        trendMaxEdf=parsed["observationArgs"].trendMaxEdf,
        trendLambdaMin=parsed["observationArgs"].trendLambdaMin,
        trendLambdaMax=parsed["observationArgs"].trendLambdaMax,
        trendLambdaGridSize=parsed["observationArgs"].trendLambdaGridSize,
        numNearest=parsed["observationArgs"].numNearest,
        sparseSupportScaleBP=parsed["observationArgs"].sparseSupportScaleBP,
        sparseSupportPrior=parsed["observationArgs"].sparseSupportPrior,
        restrictLocalAR1ToSparseBed=parsed["observationArgs"].restrictLocalAR1ToSparseBed,
        pad=parsed["observationArgs"].pad,
    ).muncVarianceModel == constants.OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL
    assert parsed["countingArgs"].subtractGlobalMedian == profile[
        "countingParams.subtractGlobalMedian"
    ]
    assert parsed["outputArgs"].saveBackgroundTracks == profile[
        "outputParams.saveBackgroundTracks"
    ]
    assert (
        parsed["outputArgs"].plotOptimizationPath
        is constants.OUTPUT_DEFAULT_PLOT_OPTIMIZATION_PATH
    )
    assert (
        consenrich_core.outputParams(
            convertToBigWig=parsed["outputArgs"].convertToBigWig,
            roundDigits=parsed["outputArgs"].roundDigits,
            writeUncertainty=parsed["outputArgs"].writeUncertainty,
        ).saveBackgroundTracks
        == constants.OUTPUT_DEFAULT_SAVE_BACKGROUND_TRACKS
    )
    assert (
        consenrich_core.fitParams().useNonnegativeBackground
        == constants.FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND
    )
    assert (
        consenrich_core.fitParams().backgroundNegativePenaltyMultiplier
        == constants.FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER
    )
    assert parsed["countingArgs"].replicateMedianDetrendWindowMultiplier == (
        constants.COUNTING_DEFAULT_REPLICATE_MEDIAN_DETREND_WINDOW_MULTIPLIER
    )
    assert parsed["matchingArgs"].exportFilterUncertaintyMultiplier == (
        constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER
    )
    assert consenrich_core.processParams().minQ == constants.PROCESS_DEFAULT_MIN_Q
    assert (
        consenrich_core.countingParams(
            intervalSizeBP=parsed["countingArgs"].intervalSizeBP,
            backgroundBlockSizeBP=parsed["countingArgs"].backgroundBlockSizeBP,
            scaleFactors=None,
            scaleFactorsControl=None,
            normMethod=parsed["countingArgs"].normMethod,
            fragmentsGroupNorm=parsed["countingArgs"].fragmentsGroupNorm,
            fixControl=parsed["countingArgs"].fixControl,
            logOffset=parsed["countingArgs"].logOffset,
            logMult=parsed["countingArgs"].logMult,
        ).replicateMedianDetrendWindowMultiplier
        == constants.COUNTING_DEFAULT_REPLICATE_MEDIAN_DETREND_WINDOW_MULTIPLIER
    )
    assert (
        consenrich_core.uncertaintyCalibrationParams().enabled
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_ENABLED
    )

    cliDefaults = consenrich_cli._buildArgParser().parse_args([])
    assert cliDefaults.matchTau0 == constants.MATCHING_DEFAULT_TAU0
    assert cliDefaults.matchNumBootstrap == constants.MATCHING_DEFAULT_NUM_BOOTSTRAP
    assert cliDefaults.matchThresholdZ == constants.MATCHING_DEFAULT_THRESHOLD_Z
    assert (
        cliDefaults.matchNestedRoccoIters
        == constants.MATCHING_DEFAULT_NESTED_ROCCO_ITERS
    )
    assert (
        cliDefaults.matchNestedRoccoBudgetScale
        == constants.MATCHING_DEFAULT_NESTED_ROCCO_BUDGET_SCALE
    )
    assert cliDefaults.matchExportFilterUncertaintyMultiplier == (
        constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER
    )
    assert cliDefaults.matchRandSeed == constants.MATCHING_DEFAULT_RAND_SEED


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
    fitParams.ECM_outerIters: 16
    fitParams.ECM_backgroundLengthScaleMultiplier: 2.0
    countingParams.replicateMedianDetrend: false
    countingParams.replicateMedianDetrendWindowMultiplier: 3.0
    countingParams.gentleDetrendQuantile: 0.75
    countingParams.subtractGlobalMedian: false
    processParams.regularizationStrength: 2.5
    processParams.regularizationRatio: 0.002
    processParams.precisionMultiplierMin: 0.5
    observationParams.precisionMultiplierMax: 4.0
    outputParams.saveBackgroundTracks: false
    uncertaintyCalibrationParams.enabled: false
    """

    configPath = writeConfigFile(tmp_path, "config_generic_override.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["defaultConfiguration"] == "generic"
    assert parsed["fitArgs"].ECM_outerIters == 16
    assert parsed["fitArgs"].ECM_backgroundLengthScaleMultiplier == pytest.approx(2.0)
    assert parsed["countingArgs"].replicateMedianDetrend is False
    assert parsed["countingArgs"].replicateMedianDetrendWindowMultiplier == pytest.approx(
        3.0
    )
    assert parsed["countingArgs"].gentleDetrendQuantile == pytest.approx(0.75)
    assert parsed["countingArgs"].subtractGlobalMedian is False
    assert parsed["processArgs"].regularizationStrength == pytest.approx(2.5)
    assert parsed["processArgs"].regularizationRatio == pytest.approx(0.002)
    assert parsed["processArgs"].precisionMultiplierMin == pytest.approx(0.5)
    assert parsed["observationArgs"].precisionMultiplierMax == pytest.approx(4.0)
    assert parsed["outputArgs"].saveBackgroundTracks is False
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


def _case_readConfigObservationTrendRemovesLinearEnvelope(
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
    fitParams.ECM_useAPN: true
    fitParams.ECM_useProcessPrecisionReweighting: true
    """

    configPath = writeConfigFile(tmp_path, "config_apn.yaml", configYaml)
    configParsed = readConfig(str(configPath))

    assert configParsed["fitArgs"].ECM_useAPN is True
    assert configParsed["fitArgs"].ECM_useProcessPrecisionReweighting is False


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
    assert not hasattr(defaultFitArgs, "ECM_backgroundPriorQuantile")
    assert not hasattr(defaultFitArgs, "ECM_backgroundPriorVariance")
    assert hasattr(defaultFitArgs, "ECM_backgroundLengthScaleMultiplier")

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.ECM_zeroCenterBackground: false
    fitParams.ECM_zeroCenterReplicateBias: false
    fitParams.useNonnegativeBackground: false
    fitParams.backgroundNegativePenaltyMultiplier: null
    fitParams.ECM_backgroundLengthScaleMultiplier: 6
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
    assert parsedOverride["fitArgs"].ECM_zeroCenterBackground is False
    assert parsedOverride["fitArgs"].ECM_zeroCenterReplicateBias is False
    assert parsedOverride["fitArgs"].useNonnegativeBackground is False
    assert parsedOverride["fitArgs"].backgroundNegativePenaltyMultiplier is None
    assert parsedOverride["fitArgs"].ECM_backgroundLengthScaleMultiplier == pytest.approx(
        6.0
    )


def _case_readConfigAllowsEMTNuOverride(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.ECM_robustTNu: 4.0
    """
    parsedOverride = readConfig(
        str(writeConfigFile(tmp_path, "config_em_tnu_override.yaml", configOverrideYaml))
    )
    assert parsedOverride["fitArgs"].ECM_robustTNu == pytest.approx(4.0)


def _case_readConfigUsesECMAndOuterPassToleranceFields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.ECM_fixedBackgroundRtol: 1.0e-6
    fitParams.ECM_backgroundShiftRtol: 2.5e-3
    fitParams.ECM_outerNLLRtol: 3.5e-4
    """

    configPath = writeConfigFile(tmp_path, "config_ecm_tol.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["fitArgs"].ECM_fixedBackgroundRtol == pytest.approx(1.0e-6)
    assert parsed["fitArgs"].ECM_backgroundShiftRtol == pytest.approx(2.5e-3)
    assert parsed["fitArgs"].ECM_outerNLLRtol == pytest.approx(3.5e-4)


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

    configExplicitYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    uncertaintyCalibrationParams.enabled: false
    uncertaintyCalibrationParams.blockSizeBP: 25000
    uncertaintyCalibrationParams.folds: 3
    uncertaintyCalibrationParams.holdoutFraction: 0.2
    uncertaintyCalibrationParams.maxScores: 1234
    uncertaintyCalibrationParams.targets: [0.5, 0.9]
    uncertaintyCalibrationParams.targetCalibrationDelta: 0.025
    uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: false
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
    assert explicitArgs.targetCalibrationDelta == pytest.approx(0.025)
    assert explicitArgs.scaleUncertaintyByTargetCalibration is False

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
    assert parsedNoSparse["observationArgs"].restrictLocalVarianceToSparseBed is False

    configExplicitSparse = f"""
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.sparseBedFile: {sparseBedPath}
    observationParams.restrictLocalVarianceToSparseBed: true
    observationParams.muncVarianceModel: svarD2
    observationParams.muncTrendBlockSizeBP: 250
    observationParams.muncLocalWindowSizeBP: 500
    observationParams.muncTrendBlockDependenceMultiplier: 1.5
    observationParams.muncLocalWindowDependenceMultiplier: 2.5
    """
    configExplicitSparsePath = writeConfigFile(
        tmp_path,
        "config_restrict_local_ar1_explicit_sparse.yaml",
        configExplicitSparse,
    )
    parsedExplicitSparse = readConfig(str(configExplicitSparsePath))
    explicitObservationArgs = parsedExplicitSparse["observationArgs"]
    assert explicitObservationArgs.restrictLocalAR1ToSparseBed is True
    assert explicitObservationArgs.restrictLocalVarianceToSparseBed is True
    assert explicitObservationArgs.muncVarianceModel == constants.MUNC_VARIANCE_MODEL_SVAR_D2
    assert explicitObservationArgs.muncTrendBlockSizeBP == 250
    assert explicitObservationArgs.muncLocalWindowSizeBP == 500
    assert explicitObservationArgs.muncTrendBlockDependenceMultiplier == 1.5
    assert explicitObservationArgs.muncLocalWindowDependenceMultiplier == 2.5

    configFirstDifferenceModel = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncVarianceModel: svarD1
    """
    configFirstDifferenceModelPath = writeConfigFile(
        tmp_path,
        "config_svar_d1_munc_model.yaml",
        configFirstDifferenceModel,
    )
    parsedFirstDifferenceModel = readConfig(str(configFirstDifferenceModelPath))
    assert (
        parsedFirstDifferenceModel["observationArgs"].muncVarianceModel
        == constants.MUNC_VARIANCE_MODEL_SVAR_D1
    )

    configSecondDifferenceModel = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncVarianceModel: svarD2
    """
    configSecondDifferenceModelPath = writeConfigFile(
        tmp_path,
        "config_svar_d2_munc_model.yaml",
        configSecondDifferenceModel,
    )
    parsedSecondDifferenceModel = readConfig(str(configSecondDifferenceModelPath))
    assert (
        parsedSecondDifferenceModel["observationArgs"].muncVarianceModel
        == constants.MUNC_VARIANCE_MODEL_SVAR_D2
    )

    configInvalidModel = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncVarianceModel: nope
    """
    configInvalidModelPath = writeConfigFile(
        tmp_path,
        "config_invalid_munc_model.yaml",
        configInvalidModel,
    )
    with pytest.raises(ValueError, match="MUNC variance model"):
        readConfig(str(configInvalidModelPath))


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

    indices = consenrich_core._loadSparseIntervalIndices(
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
    caplog.set_level(logging.INFO, logger=consenrich_io.logger.name)
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
    counts = consenrich_core.readSegments(
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
    caplog.set_level(logging.INFO, logger=consenrich_io.logger.name)
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
    treatment, control = consenrich_io._resolveExtendFrom5pBPPairs(
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

    consenrich_io._convertBedGraphToBigWigPyBigWig(
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
        consenrich_io._convertBedGraphToBigWigPyBigWig(
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
        consenrich_io._convertBedGraphToBigWigPyBigWig(
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

    consenrich_io._sortBedGraphInPlace(str(bedGraphPath))

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

    consenrich_io._validateBedGraphSorted(str(bedGraphPath), chromOrder=["chr2", "chr1"])
    with pytest.raises(ValueError, match="chromosome order"):
        consenrich_io._validateBedGraphSorted(
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
    consenrich_io._sortBedGraphInPlace(str(unsortedPath), chromOrder=["chr2", "chr1"])

    assert unsortedPath.read_text(encoding="utf-8").splitlines() == [
        "track type=bedGraph name=toy",
        "browser position chr2:1-20",
        "chr2\t0\t10\t2.0000",
        "chr2\t10\t20\t2.5000",
        "chr1\t0\t10\t1.0000",
        "chr1\t10\t20\t1.5000",
    ]


def _case_resolveFixedDeltaFRequiresPositiveFinite():
    assert consenrich_core._resolveFixedDeltaF(0.25) == pytest.approx(0.25)

    for badDeltaF in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="deltaF"):
            consenrich_core._resolveFixedDeltaF(badDeltaF)


def _caseReplicateDetrendAutoDisabledForControlLogRatios():
    countingArgs = consenrich_core.countingParams(
        intervalSizeBP=25,
        backgroundBlockSizeBP=1000,
        scaleFactors=None,
        scaleFactorsControl=None,
        normMethod="RPKM",
        fragmentsGroupNorm="NONE",
        fixControl=False,
        logOffset=1.0,
        logMult=1.0,
        replicateMedianDetrend=True,
        replicateMedianDetrendWindowMultiplier=5.0,
        gentleDetrendQuantile=0.5,
    )

    enabled, label = consenrich_cli._resolveReplicateDetrendStatus(
        countingArgs,
        controlsPresent=False,
    )
    assert enabled is True
    assert label == "quantile=0.5 x5"

    enabled, label = consenrich_cli._resolveReplicateDetrendStatus(
        countingArgs,
        controlsPresent=True,
    )
    assert enabled is False
    assert label == "no (control log-ratio)"

    disabledArgs = countingArgs._replace(replicateMedianDetrend=False)
    enabled, label = consenrich_cli._resolveReplicateDetrendStatus(
        disabledArgs,
        controlsPresent=True,
    )
    assert enabled is False
    assert label == "no"


def _caseGlobalMedianCenterAutoDisabledForControlLogRatios():
    countingArgs = consenrich_core.countingParams(
        intervalSizeBP=25,
        backgroundBlockSizeBP=1000,
        scaleFactors=None,
        scaleFactorsControl=None,
        normMethod="RPKM",
        fragmentsGroupNorm="NONE",
        fixControl=False,
        logOffset=1.0,
        logMult=1.0,
        subtractGlobalMedian=True,
    )

    enabled, label = consenrich_cli._resolveGlobalMedianCenterStatus(
        countingArgs,
        controlsPresent=False,
    )
    assert enabled is True
    assert label == "yes"

    enabled, label = consenrich_cli._resolveGlobalMedianCenterStatus(
        countingArgs,
        controlsPresent=True,
    )
    assert enabled is False
    assert label == "no (control log-ratio)"

    disabledArgs = countingArgs._replace(subtractGlobalMedian=False)
    enabled, label = consenrich_cli._resolveGlobalMedianCenterStatus(
        disabledArgs,
        controlsPresent=True,
    )
    assert enabled is False
    assert label == "no"


def _run_with_monkeypatch(monkeypatch, func, *args):
    with monkeypatch.context() as mp:
        return func(*args, mp)


def test_config_runtime_logging_and_validation_contracts(
    tmp_path, monkeypatch, caplog, contract_case
):
    contract_case(
        "runtime background span",
        _caseRuntimeBackgroundSpanUsesLengthScaleMultiplier,
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
    contract_case(
        "replicate gain frame",
        _caseReplicateGainFrameShowsIndentedIdFileMeanMedianSdAndIqr,
    )
    contract_case(
        "control log-ratio disables replicate detrend",
        _caseReplicateDetrendAutoDisabledForControlLogRatios,
    )
    contract_case(
        "control log-ratio disables global median centering",
        _caseGlobalMedianCenterAutoDisabledForControlLogRatios,
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
        ("process noise options", _case_readConfigProcessNoiseOptions),
        ("generic profile", _case_readConfigUsesGenericDefaultConfiguration),
        ("centralized runtime defaults", _case_runtime_defaults_are_centralized),
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
        ("observation trend fields", _case_readConfigObservationTrendRemovesLinearEnvelope),
        ("chromosome deduplication", _case_readConfigDeduplicatesChromosomes),
        ("APN disables process precision reweighting", _case_readConfigAPNDisablesProcPrecReweight),
        ("zero-center identifiability fields", _case_readConfigUsesZeroCenterIdentifiabilityFields),
        ("ECM t-nu override", _case_readConfigAllowsEMTNuOverride),
        ("ECM outer-pass tolerance fields", _case_readConfigUsesECMAndOuterPassToleranceFields),
        ("uncertainty calibration fields", _case_readConfigUsesUncertaintyCalibrationFields),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)


def test_optimization_path_output_helpers(tmp_path, monkeypatch):
    diagnostics = {
        "process_noise_warmup_fit": None,
        "post_process_noise_fit": {
            "fixed_background_ecm": [
                {
                    "outer_pass": 1,
                    "outer_objective": 12.5,
                    "outer_objective_per_cell": 0.5,
                    "outer_objective_change_per_cell": None,
                    "outer_objective_threshold_per_cell": 0.01,
                    "outer_objective_stable": False,
                    "background_shift": 0.25,
                    "background_shift_threshold": 0.05,
                    "background_shift_stable": False,
                    "outer_stable_iters": 0,
                    "outer_patience_target": 2,
                    "outer_inner_ecm_converged": False,
                    "optimization_path": [
                        {
                            "iter": 1,
                            "objective_name": "nll",
                            "objective_value": 20.0,
                            "change": None,
                            "threshold": None,
                            "reset_iteration": True,
                            "converged": False,
                        },
                        {
                            "iter": 2,
                            "objective_name": "nll",
                            "objective_value": 19.5,
                            "change": 0.5,
                            "threshold": 0.1,
                            "converged": True,
                        },
                    ],
                }
            ]
        },
    }
    rows = consenrich_cli._flattenOptimizationPathDiagnostics("chrTest", diagnostics)
    assert [row["record_order"] for row in rows] == [0, 1, 2]
    assert rows[0]["path_level"] == "outer"
    assert rows[0]["background_shift"] == 0.25
    assert rows[0]["background_shift_stable"] is False
    assert rows[1]["path_level"] == "inner"
    assert rows[1]["reset_iteration"] is True
    assert rows[1]["change"] is None
    assert rows[-1]["final_solution"] is True

    logPath = tmp_path / "optimization.log"
    consenrich_cli._writeOptimizationPathLog(rows, str(logPath))
    lines = logPath.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == consenrich_cli.OPTIMIZATION_PATH_COLUMNS
    assert len(lines) == 4

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "matplotlib", None)
        assert (
            consenrich_cli._plotOptimizationPathLog(
                rows,
                str(tmp_path / "missing.png"),
            )
            is False
        )

    saveCalls = []
    fakeMatplotlib = types.ModuleType("matplotlib")
    fakePyplot = types.ModuleType("matplotlib.pyplot")

    class FakeFigure:
        def savefig(self, path, dpi=None):
            saveCalls.append((path, dpi))

    class FakeAxis:
        transAxes = object()

        def plot(self, *args, **kwargs):
            return None

        def scatter(self, *args, **kwargs):
            return None

        def annotate(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

        def axhline(self, *args, **kwargs):
            return None

        def set_yscale(self, *args, **kwargs):
            return None

    fakeMatplotlib.use = lambda *args, **kwargs: None
    fakePyplot.rcParams = {}
    fakePyplot.subplots = lambda *args, **kwargs: (
        FakeFigure(),
        FakeAxis(),
    )
    fakePyplot.close = lambda *args, **kwargs: None
    fakeMatplotlib.pyplot = fakePyplot
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "matplotlib", fakeMatplotlib)
        mp.setitem(sys.modules, "matplotlib.pyplot", fakePyplot)
        assert (
            consenrich_cli._plotOptimizationPathLog(
                rows,
                str(tmp_path / "optimization.png"),
            )
            is True
        )
    assert saveCalls == [(str(tmp_path / "optimization.png"), 400)]
    assert (
        consenrich_cli._optimizationPathPrefix("exp name", "chr1/random")
        == f"consenrichOutput_exp_name_chr1_random_optimizationPath.v{consenrich_cli.__version__}"
    )


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
