import textwrap
import io
import json
import logging
import math
import sys
import types
from pathlib import Path

import ccounts
import numpy as np
import pandas as pd
import pytest

from consenrich import _normalization as normalization
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
        return "hg38"

    def fakeGetGenomeResourceFile(genomeLabel: str, resourceName: str) -> str:
        if resourceName == "sizes":
            return str(chromSizesPath)
        return str(tmpPath / f"{genomeLabel}.{resourceName}.bed")

    monkeypatch.setattr(constants, "resolveGenomeName", fakeResolveGenomeName)
    monkeypatch.setattr(constants, "getGenomeResourceFile", fakeGetGenomeResourceFile)
    monkeypatch.setattr(constants, "assemblyChromosomeSizes", {"hg38": {"chrTest": 100000}})
    monkeypatch.setattr(constants, "standardChromosomes", {"hg38": ("chrTest",)})




def _caseReplicateGainSummaryWritesPooledAverageAndStd(tmp_path):
    sources = [
        consenrich_core.inputSource(
            path="/tmp/ENCFF12345_sampleA.bam",
            sourceKind="BAM",
            sampleName="sampleA",
        ),
        consenrich_core.inputSource(
            path="/tmp/sampleB.bam",
            sourceKind="BAM",
            sampleName="sampleB",
        ),
    ]
    controls = [
        consenrich_core.inputSource(
            path="/tmp/controlA.bam",
            sourceKind="BAM",
            sampleName="controlA",
        ),
    ]
    accumulator = consenrich_cli._newReplicateGainAccumulator(2)
    assert (
        consenrich_cli._updateReplicateGainAccumulator(
            accumulator,
            {"mean": [0.125, 0.25], "sd": [0.0125, 0.025], "count": [4, 4]},
        )
        == 2
    )
    assert (
        consenrich_cli._updateReplicateGainAccumulator(
            accumulator,
            {"mean": [0.25, 0.5], "sd": [0.025, 0.05], "count": [6, 6]},
        )
        == 2
    )
    rows = consenrich_cli._replicateGainSummaryRows(
        sources,
        accumulator,
        controlSources=controls,
    )

    expectedAvg = ((0.125 * 4.0) + (0.25 * 6.0)) / 10.0
    expectedSumSq = ((0.0125**2 + 0.125**2) * 4.0) + ((0.025**2 + 0.25**2) * 6.0)
    expectedStd = np.sqrt((expectedSumSq / 10.0) - (expectedAvg**2))
    assert rows[0]["sample_name"] == "sampleA"
    assert rows[0]["sample_file"] == "ENCFF12"
    assert rows[0]["control_path"] == "/tmp/controlA.bam"
    assert rows[0]["chromosome_count"] == 2
    assert rows[0]["finite_interval_count"] == 10
    assert rows[0]["gain_avg"] == pytest.approx(expectedAvg)
    assert rows[0]["gain_std"] == pytest.approx(expectedStd)

    path = tmp_path / "gains.jsonl"
    assert consenrich_cli._writeReplicateGainSummary(rows, str(path)) is True
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["replicate_index"] for record in records] == [1, 2]
    assert records[0]["sample_file"] == "ENCFF12"
    assert records[0]["gain_avg"] == pytest.approx(expectedAvg)
    assert records[0]["gain_std"] == pytest.approx(expectedStd)
    assert "gain_median" not in records[0]
    assert "gain_iqr" not in records[0]


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
        assert (
            False
        ), "Expected ValueError not raised given empty `consenrich_core.inputParams`"


def _caseScaleFactorNormalizationBroadcastsSingletons():
    assert normalization.expandScaleFactors(
        [0.25],
        3,
        "countingParams.scaleFactorsControl",
    ) == [0.25, 0.25, 0.25]
    assert normalization.expandScaleFactors(
        [1.0, 2.0, 3.0],
        3,
        "countingParams.scaleFactors",
    ) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="must contain 1 value or 3 values"):
        normalization.expandScaleFactors(
            [1.0, 2.0],
            3,
            "countingParams.scaleFactorsControl",
        )






def test_count_model_variance_floor_scalar_uses_count_noise_not_munc_minr():
    floor = np.asarray(
        [
            [np.nan, 0.04, 0.01],
            [0.09, np.inf, 0.25],
        ],
        dtype=np.float32,
    )

    expected = np.quantile([0.01, 0.04, 0.09, 0.25], 0.05)
    assert consenrich_cli._countModelVarianceFloorScalar(floor) == pytest.approx(
        expected
    )
    assert consenrich_cli._countModelVarianceFloorScalar(
        np.full((2, 3), np.nan, dtype=np.float32),
        fallback=1.0e-7,
    ) == pytest.approx(1.0e-7)


def test_replicate_exchangeability_diagnostic_helpers(tmp_path):
    nullMatrix = np.zeros((12, 3), dtype=np.float64)
    nullDiagnostic = consenrich_cli._replicateExchangeabilityFromLogSDMatrix(
        nullMatrix,
        sampleNames=("a", "b", "c"),
        seed=11,
        permutationCount=19,
    )
    assert nullDiagnostic["status"] == "ok"
    assert nullDiagnostic["omnibusObserved"] == pytest.approx(0.0)
    assert nullDiagnostic["omnibusPValue"] == pytest.approx(1.0)

    shiftedMatrix = np.zeros((18, 3), dtype=np.float64)
    shiftedMatrix[:, 0] = 0.8
    shiftedDiagnostic = consenrich_cli._replicateExchangeabilityFromLogSDMatrix(
        shiftedMatrix,
        sampleNames=("shifted", "baseA", "baseB"),
        seed=11,
        permutationCount=99,
    )
    assert shiftedDiagnostic["status"] == "ok"
    assert shiftedDiagnostic["omnibusPValue"] <= 0.05
    minPair = shiftedDiagnostic["pairwiseSign"]["minPair"]
    assert minPair is not None
    assert {minPair["replicateA"], minPair["replicateB"]} == {0, 1}
    assert minPair["pValue"] < 0.001

    summaryPath = tmp_path / "exchangeability.txt"
    assert (
        consenrich_cli._writeReplicateExchangeabilitySummary(
            shiftedDiagnostic,
            summaryPath,
        )
        is True
    )
    summaryText = summaryPath.read_text(encoding="utf-8")
    assert "omnibus_p_value" in summaryText
    assert "shifted" in summaryText
    assert consenrich_cli._replicateExchangeabilityPlotPath("exp name").name == (
        "consenrichOutput_exp_name_replicateExchangeability.png"
    )
    assert consenrich_cli._replicateExchangeabilitySummaryPath("exp name").name == (
        "consenrichOutput_exp_name_replicateExchangeability.txt"
    )

    blockCount = 6
    sampleIndex = np.tile(np.arange(2, dtype=np.int64), blockCount)
    chromIndex = np.zeros(sampleIndex.size, dtype=np.int64)
    blockStarts = np.repeat(np.arange(blockCount, dtype=np.int64), 2)
    factors = np.asarray([0.25, 4.0], dtype=np.float64)
    basePrior = np.repeat(np.linspace(1.0, 2.0, blockCount), 2)
    factorPrior = basePrior * factors[sampleIndex]
    factorDiagnostic = consenrich_cli._replicateExchangeabilityFromPooledBlocks(
        factorPrior,
        factorPrior,
        sampleIndex,
        chromIndex,
        blockStarts,
        2,
        sampleNames=("low", "high"),
        seed=17,
    )
    factorDiagnostic["priorVarianceFactorAdjusted"] = True
    factorDiagnostic["replicateVarianceFactors"] = factors
    factorDiagnostic["replicateSDMultipliers"] = np.sqrt(factors)
    factorDiagnostic["rawOmnibusObserved"] = math.log(2.0)
    factorDiagnostic["rawOmnibusPValue"] = 0.01
    assert factorDiagnostic["status"] == "ok"
    assert factorDiagnostic["omnibusObserved"] == pytest.approx(0.0)
    assert factorDiagnostic["omnibusPValue"] == pytest.approx(1.0)

    factorSummaryPath = tmp_path / "exchangeability_factor.txt"
    assert (
        consenrich_cli._writeReplicateExchangeabilitySummary(
            factorDiagnostic,
            factorSummaryPath,
        )
        is True
    )
    factorSummaryText = factorSummaryPath.read_text(encoding="utf-8")
    assert "prior_variance_factor_adjusted: true" in factorSummaryText
    assert "raw_omnibus_p_value: 0.01" in factorSummaryText
    assert "fitted_replicate_munc_factors" in factorSummaryText
    assert "high" in factorSummaryText

    largeBlockCount = consenrich_cli._REPLICATE_EXCHANGEABILITY_MAX_BLOCKS + 7
    largeSampleCount = 3
    largeSampleIndex = np.tile(
        np.arange(largeSampleCount, dtype=np.int64),
        largeBlockCount,
    )
    largeBlockStarts = np.repeat(
        np.arange(largeBlockCount, dtype=np.int64),
        largeSampleCount,
    )
    largeVars = np.ones(largeSampleIndex.size, dtype=np.float64)
    largeDiagnostic = consenrich_cli._replicateExchangeabilityFromPooledBlocks(
        largeVars,
        largeVars,
        largeSampleIndex,
        np.zeros(largeSampleIndex.size, dtype=np.int64),
        largeBlockStarts,
        largeSampleCount,
        seed=17,
    )
    assert largeDiagnostic["blockCount"] == largeBlockCount
    assert (
        largeDiagnostic["diagnosticBlockCount"]
        == consenrich_cli._REPLICATE_EXCHANGEABILITY_MAX_BLOCKS
    )
    assert largeDiagnostic["status"] == "ok"


def test_replicateVarianceHeterogeneityWarningPolicy(tmp_path, caplog):
    diagnosticPath = tmp_path / "replicate_variance_diagnostic.txt"
    caseRows = (
        (
            "rawRatio",
            0.05,
            (1.0, 1.1, 1.5),
            0.5,
            (1.0, 1.1, 1.1),
            (1.0, 1.1, 1.1),
            False,
            True,
        ),
        (
            "fittedRatio",
            0.05,
            (1.0, 1.1, 1.1),
            0.5,
            (1.0, 1.1, 1.1),
            (1.0, 1.1, 1.5),
            True,
            True,
        ),
        (
            "rawPOnly",
            0.05,
            (1.0, 1.1, 1.49),
            0.5,
            (1.0, 1.1, 1.1),
            (1.0, 1.1, 1.49),
            False,
            False,
        ),
        (
            "rawRatioOnly",
            0.051,
            (1.0, 1.1, 1.5),
            0.5,
            (1.0, 1.1, 1.1),
            (1.0, 1.1, 1.1),
            False,
            False,
        ),
        (
            "strongAdjusted",
            0.05,
            (1.0, 1.1, 1.5),
            0.01,
            (1.0, 1.1, 1.25),
            (1.0, 1.1, 1.4),
            True,
            True,
        ),
        (
            "adjustedPOnly",
            0.5,
            (1.0, 1.1, 1.1),
            0.01,
            (1.0, 1.1, 1.24),
            (1.0, 1.1, 1.1),
            True,
            False,
        ),
        (
            "adjustedRatioOnly",
            0.5,
            (1.0, 1.1, 1.1),
            0.011,
            (1.0, 1.1, 1.25),
            (1.0, 1.1, 1.1),
            True,
            False,
        ),
        (
            "adjustmentDisabled",
            0.5,
            (1.0, 1.1, 1.1),
            0.01,
            (1.0, 1.1, 1.25),
            (1.0, 1.1, 1.1),
            False,
            False,
        ),
    )
    strongMessage = ""
    caplog.set_level(logging.WARNING)
    for (
        caseName,
        rawPValue,
        rawSDMultipliers,
        adjustedPValue,
        adjustedSDMultipliers,
        fittedSDMultipliers,
        factorAdjusted,
        shouldWarn,
    ) in caseRows:
        diagnostic = {
            "status": "ok",
            "sampleNames": ("lowReplicate", "middleReplicate", "highReplicate"),
            "rawEffectByReplicate": np.log(
                np.asarray(rawSDMultipliers, dtype=np.float64)
            ),
            "effectByReplicate": np.log(
                np.asarray(adjustedSDMultipliers, dtype=np.float64)
            ),
            "replicateSDMultipliers": np.asarray(
                fittedSDMultipliers,
                dtype=np.float64,
            ),
            "rawOmnibusPValue": rawPValue,
            "omnibusPValue": adjustedPValue,
            "priorVarianceFactorAdjusted": factorAdjusted,
            "pairwiseSign": {
                "minPair": {"pValue": 1.0 if shouldWarn else 1.0e-12}
            },
        }
        caplog.clear()
        warned = consenrich_cli._warnReplicateVarianceHeterogeneity(
            diagnostic,
            diagnosticPath,
        )
        warningMessages = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert warned is shouldWarn, caseName
        assert len(warningMessages) == int(shouldWarn), caseName
        if caseName == "strongAdjusted":
            strongMessage = warningMessages[0]

    assert strongMessage.startswith("Strong modeled heterogeneity warning")
    assert "divergentReplicates='lowReplicate','highReplicate'" in strongMessage
    assert "rawSDRatio=1.5" in strongMessage
    assert "adjustedSDRatio=1.25" in strongMessage
    assert "rawPValue=0.05" in strongMessage
    assert "adjustedPValue=0.01" in strongMessage
    assert f"diagnosticFile={diagnosticPath.resolve()}" in strongMessage
    assert "replicates exhibit blockwise variance heterogeneity" in strongMessage


def test_replicate_exchangeability_plot_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    shiftedMatrix = np.zeros((18, 3), dtype=np.float64)
    shiftedMatrix[:, 0] = 0.8
    shiftedDiagnostic = consenrich_cli._replicateExchangeabilityFromLogSDMatrix(
        shiftedMatrix,
        sampleNames=("shifted", "baseA", "baseB"),
        seed=11,
        permutationCount=99,
    )
    shiftedDiagnostic["rawOmnibusObserved"] = math.log(2.0)
    plotPath = tmp_path / "exchangeability.png"
    assert (
        consenrich_cli._plotReplicateExchangeabilityDiagnostic(
            shiftedDiagnostic,
            plotPath,
        )
        is True
    )
    assert plotPath.stat().st_size > 0




def _case_readConfigOutputDiagnosticTracks(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)

    configPath = writeConfigFile(
        tmp_path,
        "config_output_diagnostic_tracks.yaml",
        """
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        outputParams.diagnosticTracks: [slope, preKappaQLevel, effectiveQTrend, muncTrace, slope]
        """,
    )

    parsed = readConfig(str(configPath))

    assert parsed["outputArgs"].diagnosticTracks == (
        "slope",
        "preKappaQLevel",
        "effectiveQTrend",
        "muncTrace",
    )
    assert normalization.validateDiagnosticTracks("all") == tuple(
        constants.OUTPUT_DIAGNOSTIC_TRACK_NAMES
    )
    with pytest.raises(ValueError, match="outputParams.diagnosticTracks"):
        normalization.validateDiagnosticTracks(["notATrack"])


def _case_readConfigBroadcastsSharedControlScaleFactor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [treatA.bam, treatB.bam]
    inputParams.bamFilesControl: [control.bam]
    genomeParams.name: testGenome
    countingParams.scaleFactorsControl: [0.25]
    """
    configPath = writeConfigFile(tmp_path, "shared_control_scale.yaml", configYaml)

    parsed = readConfig(str(configPath))

    assert parsed["inputArgs"].bamFilesControl == ["control.bam", "control.bam"]
    assert parsed["countingArgs"].scaleFactors is None
    assert parsed["countingArgs"].scaleFactorsControl == [0.25, 0.25]
    configData = consenrich_config.loadConfig(configPath)
    for sampleCount in (1, 3):
        configData["inputParams.bamFiles"] = [
            f"treat{index}.bam" for index in range(sampleCount)
        ]
        configData["countingParams.scaleFactors"] = [0.5]
        countingArgs = readConfig(configData)["countingArgs"]
        assert countingArgs.scaleFactors == [0.5] * sampleCount
        assert countingArgs.scaleFactorsControl == [0.25] * sampleCount


def _case_readConfigProcessNoiseOptions(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    processParams:
      stateModel: level
      processNoiseCalibration: fixedDiagonal
      qSeedPriorLevel: 3.0e-8
      processNoiseWarmupECMIters: 7
      processNoiseWarmupOuterPasses: 5
      precisionMultiplierMin: 0.5
      precisionMultiplierMax: 2.0
    observationParams:
      precisionMultiplierMin: 0.1
      precisionMultiplierMax: 8.0
      useReplicateVarianceScale: false
      useCountNoiseFloor: false
      muncEBPrior:
        gUncertaintyMode: disabled
    """

    configPath = writeConfigFile(tmp_path, "config_process_noise.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    processArgs = configParsed["processArgs"]

    assert processArgs.stateModel == constants.STATE_MODEL_LEVEL
    assert (
        processArgs.processNoiseCalibration
        == constants.PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL
    )
    assert processArgs.qSeedPriorLevel == pytest.approx(3.0e-8)
    assert processArgs.processNoiseWarmupECMIters == 7
    assert processArgs.processNoiseWarmupOuterPasses == 5
    assert processArgs.precisionMultiplierMin == pytest.approx(0.5)
    assert processArgs.precisionMultiplierMax == pytest.approx(2.0)
    assert configParsed["observationArgs"].precisionMultiplierMin == pytest.approx(0.1)
    assert configParsed["observationArgs"].precisionMultiplierMax == pytest.approx(8.0)
    assert configParsed["observationArgs"].useReplicateVarianceScale is False
    assert configParsed["observationArgs"].useCountNoiseFloor is False
    assert (
        configParsed["observationArgs"].muncEBPriorGUncertaintyMode
        == constants.MUNC_EB_PRIOR_G_UNCERTAINTY_MODE_DISABLED
    )
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


def _case_loadConfigValidatesKeys(tmp_path, monkeypatch):
    valid = {
        "countingParams": {"normMethod": "CPM"},
        "observationParams.lengthScaleOptions": {"blockCount": 12},
    }
    assert consenrich_config.loadConfig(valid) is valid
    for declaration in (
        {"other": {}},
        {"countingParams.other": None},
        {"countingParams": {"other": None}},
        {"observationParams": {"lengthScaleOptions.other": None}},
        {"observationParams.lengthScaleOptions": {"other": None}},
    ):
        with pytest.raises(ValueError, match="Unknown configuration key"):
            consenrich_config.loadConfig({**valid, **declaration})
    configPath = writeConfigFile(
        tmp_path, "unknown_key.yaml", "countingParams:\n  other: null",
    )
    with pytest.raises(ValueError, match="countingParams.other"):
        consenrich_config.loadConfig(configPath)


def _case_readConfigUsesGenericDefaultConfiguration(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    configuration: generic
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_generic_defaults.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["defaultConfiguration"] == "generic"
    assert (
        parsed["samArgs"].defaultCountMode
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
    assert (
        parsed["scArgs"].defaultCountMode
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
    assert parsed["processArgs"].stateModel == constants.STATE_MODEL_LEVEL_TREND
    assert (
        parsed["processArgs"].processNoiseWarmupECMIters
        == constants.PROCESS_DEFAULT_WARMUP_ECM_ITERS
    )
    assert (
        parsed["processArgs"].processNoiseWarmupOuterPasses
        == constants.PROCESS_DEFAULT_WARMUP_OUTER_PASSES
    )
    assert (
        parsed["processArgs"].processNoiseCalibration
        == constants.PROCESS_DEFAULT_NOISE_CALIBRATION
    )
    assert (
        parsed["processArgs"].processNoiseCalibration
        == constants.PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL
    )
    assert (
        parsed["observationArgs"].useReplicateVarianceScale
        is constants.OBSERVATION_DEFAULT_USE_REPLICATE_VARIANCE_SCALE
    )
    assert (
        parsed["observationArgs"].useCountNoiseFloor
        is constants.OBSERVATION_DEFAULT_USE_COUNT_NOISE_FLOOR
    )
    assert (
        parsed["observationArgs"].muncEBPriorGUncertaintyMode
        == constants.OBSERVATION_DEFAULT_MUNC_EB_PRIOR_G_UNCERTAINTY_MODE
    )
    assert (
        parsed["outputArgs"].stateShrinkageModel
        == constants.OUTPUT_STATE_SHRINKAGE_MODEL_SPIKE_AND_STUDENT_T
    )
    assert parsed["outputArgs"].stateShrinkageEnabled is True
    assert parsed["outputArgs"].stateShrinkageSpikeOddsMultiplier == pytest.approx(
        2.0
    )
    assert parsed["outputArgs"].stateShrinkageScaleAnchorWeight is None
    assert (
        parsed["outputArgs"].deleteBedGraphsAfterBigWig
        is constants.OUTPUT_DEFAULT_DELETE_BEDGRAPHS_AFTER_BIGWIG
        is True
    )
    assert parsed["matchingArgs"].shrinkChromosomeBudgets is True

    bam = {"path": "smallTest.bam"}
    h5ad = {"path": "cells.h5ad", "matrixRows": "cells"}
    for inputSettings, countingSettings, expectedMethod in (
        ({"inputParams.samples": [bam]}, {}, "RPGC"),
        ({"inputParams.samples": [bam]}, {"normMethod": None}, "RPGC"),
        ({"inputParams.samples": [h5ad]}, {}, "CPM"),
        ({"inputParams.samples": [h5ad]}, {"normMethod": None}, "CPM"),
        ({"inputParams.samples": [bam, {**h5ad, "role": "control"}]}, {}, "CPM"),
        ({"inputParams.samples": [h5ad, {**bam, "role": "control"}]},
         {"normMethod": None}, "CPM"),
        ({"inputParams.samples": [bam, h5ad]}, {}, "CPM"),
        ({"inputParams.samples": [{"path": "counts.bin", "format": "H5AD",
                                   "matrixRows": "pseudobulks"}]}, {}, "CPM"),
        ({"inputParams.samples": [{"path": "sample.fragments.tsv.gz"}],
          "scParams.cellMetadata": {"path": "labels.h5ad"}}, {}, "RPGC"),
        ({"inputParams.samples": [h5ad]}, {"normMethod": "RPKM"}, "RPKM"),
        ({"inputParams.samples": [h5ad, {**bam, "role": "control"}]},
         {"normMethod": "RPGC", "scaleFactors": [0.8],
          "scaleFactorsControl": [1.2]}, "RPGC"),
        ({"inputParams.samples": [h5ad]}, {"scaleFactors": [0.8]}, "CPM"),
    ):
        values = {
            "genomeParams.name": "testGenome", **inputSettings,
            "countingParams": countingSettings,
        }
        countingArgs = readConfig(values)["countingArgs"]
        assert countingArgs.normMethod == expectedMethod
        assert consenrich_config.getCountingArgs(values).normMethod == expectedMethod
        assert countingArgs.scaleFactors == countingSettings.get("scaleFactors")
        assert countingArgs.scaleFactorsControl == countingSettings.get("scaleFactorsControl")
    assert consenrich_config.getCountingArgs({}).normMethod == "RPGC"


def _caseGenericDefaultConfigurationUsesCanonicalUncertaintyKeys():
    defaults = consenrich_config.DEFAULT_CONFIGURATION_VALUES[
        consenrich_config.GENERIC_DEFAULT_CONFIGURATION
    ]

    assert "uncertaintyCalibrationParams.enabled" in defaults
    assert (
        defaults["uncertaintyCalibrationParams.mode"]
        == constants.UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE
    )
    for key in (
        "observationParams.muncEBPrior.tileSizeBP",
        "observationParams.muncEBPrior.tileCount",
        "observationParams.muncEBPrior.strata",
        "observationParams.muncEBPrior.minTilesPerStratum",
        "observationParams.muncEBPrior.seed",
        "observationParams.muncEBPrior.supportMinQ",
        "observationParams.muncEBPrior.supportMaxQ",
        "observationParams.muncEBPrior.maxExtrapolatedFraction",
        "observationParams.muncEBPrior.warmupECMIters",
        "observationParams.muncEBPrior.warmupOuterPasses",
        "observationParams.muncEBPrior.gUncertaintyMode",
        "observationParams.useCountNoiseFloor",
        "observationParams.lengthScaleBP",
        "observationParams.lengthScaleOptions",
        "outputParams.stateShrinkageEnabled",
        "outputParams.stateShrinkageModel",
        "outputParams.stateShrinkageSpikeOddsMultiplier",
        "outputParams.stateShrinkageScaleAnchorWeight",
        "outputParams.stateShrinkageStudentTDF",
        "outputParams.stateShrinkageStudentTQuadratureOrder",
        "outputParams.plotPrecisionReweightingHistograms",
        "outputParams.plotNullCalibrationDiagnostics",
        "outputParams.precisionReweightingHistogramSampleSize",
        "uncertaintyCalibrationParams.deleteBlockVarianceMode",
        "uncertaintyCalibrationParams.deleteBlockUseLambdaInInformation",
        "uncertaintyCalibrationParams.deleteBlockReplicateDependenceRho",
        "uncertaintyCalibrationParams.deleteBlockTargetSignal",
        "uncertaintyCalibrationParams.deleteBlockFactorModel",
        "uncertaintyCalibrationParams.deleteBlockFactorSegmentCount",
        "uncertaintyCalibrationParams.deleteBlockFactorBootstrapReplicates",
        "uncertaintyCalibrationParams.deleteBlockDeletionProbability",
        "uncertaintyCalibrationParams.deleteBlockMinInformationFraction",
        "uncertaintyCalibrationParams.deleteBlockMaxInformationFraction",
        "uncertaintyCalibrationParams.deleteBlockMinDeltaVariance",
        "uncertaintyCalibrationParams.deleteBlockFallbackMinValidFraction",
        "uncertaintyCalibrationParams.deleteBlockScoreWeightMode",
        "uncertaintyCalibrationParams.calibrationOuterIters",
        constants.UNCERTAINTY_CALIBRATION_SCALE_UNCERTAINTY_BY_TARGET_CALIBRATION_CONFIG_KEY,
        "uncertaintyCalibrationParams.deleteBlockApplyTargetCalibration",
    ):
        assert key in defaults
    assert "uncertaintyCalibrationParams.holdoutCount" not in defaults
    assert constants.UNCERTAINTY_CALIBRATION_MODES == (
        constants.UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE,
    )
    assert constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_MODELS == (
        constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_GLOBAL,
        constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_SEG_SHRINK,
    )
    assert constants.OUTPUT_STATE_SHRINKAGE_MODELS == (
        constants.OUTPUT_STATE_SHRINKAGE_MODEL_ADAPTIVE_NORMAL_MIXTURE,
        constants.OUTPUT_STATE_SHRINKAGE_MODEL_SPIKE_AND_NORMAL,
        constants.OUTPUT_STATE_SHRINKAGE_MODEL_SPIKE_AND_STUDENT_T,
    )
    assert constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_ENABLED is True
    assert constants.OUTPUT_DEFAULT_PLOT_PRECISION_REWEIGHTING_HISTOGRAMS is True
    assert constants.OUTPUT_DEFAULT_PLOT_NULL_CALIBRATION_DIAGNOSTICS is True
    assert constants.OUTPUT_DEFAULT_PRECISION_REWEIGHTING_HISTOGRAM_SAMPLE_SIZE == 50_000
    assert constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT is None
    assert constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT_FRACTION == 0.1
    assert constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT_MIN == 1.0e-6
    assert constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT_MAX == 1.0e6
    assert (
        constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_ODDS_MULTIPLIER
        == pytest.approx(2.0)
    )
    assert constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT is None
    assert (
        constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT_FRACTION
        == 0.05
    )
    assert constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT_MIN == 25.0
    assert "predictive_holdout" not in constants.UNCERTAINTY_CALIBRATION_MODES
    assert not any(key.startswith("uncertaintyCalibration.") for key in defaults)


def _case_readConfigRejectsUnsupportedCenterMBMethod(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    countingParams.centerMBMethod: mean
    """
    configPath = writeConfigFile(
        tmp_path,
        "config_bad_center_mb_method.yaml",
        configYaml,
    )

    with pytest.raises(ValueError, match="countingParams.centerMBMethod"):
        readConfig(str(configPath))

    invalidWindowYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    countingParams.centerMBWindowBP: 0
    """
    invalidWindowPath = writeConfigFile(
        tmp_path,
        "config_bad_center_mb_window.yaml",
        invalidWindowYaml,
    )
    with pytest.raises(ValueError, match="countingParams.centerMBWindowBP"):
        readConfig(str(invalidWindowPath))




def _case_observationMuncSeedWeightDefaultAndPrecisionIndependence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    profile = constants.DEFAULT_CONFIGURATION_VALUES[
        constants.GENERIC_DEFAULT_CONFIGURATION
    ]
    seedFieldMap = {
        "muncSeedWeightEnabled": "observationParams.muncSeedWeight.enabled",
        "muncSeedWeightPasses": "observationParams.muncSeedWeight.passes",
        "muncSeedWeightMin": "observationParams.muncSeedWeight.min",
        "muncSeedWeightMax": "observationParams.muncSeedWeight.max",
        "muncSeedWeightStudentT": "observationParams.muncSeedWeight.studentT",
        "muncSeedWeightStudentTdf": "observationParams.muncSeedWeight.studentTdf",
        "muncSeedProcessMinQ": "observationParams.muncSeedProcess.minQ",
        "muncSeedProcessMaxQ": "observationParams.muncSeedProcess.maxQ",
    }

    for defaultKey in constants.OBSERVATION_MUNC_SEED_WEIGHT_CONFIG_KEYS:
        assert defaultKey in profile
    for defaultKey in constants.OBSERVATION_MUNC_SEED_PROCESS_CONFIG_KEYS:
        assert defaultKey in profile

    baseConfig = """
    experimentName: seedWeightDefault
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """
    basePath = writeConfigFile(tmp_path, "config_seed_weight_base.yaml", baseConfig)
    baseParsed = readConfig(str(basePath))
    for attrName, defaultKey in seedFieldMap.items():
        assert getattr(baseParsed["observationArgs"], attrName) == profile[defaultKey]

    precisionConfig = """
    experimentName: seedWeightPrecision
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.precisionMultiplierMin: 0.25
    observationParams.precisionMultiplierMax: 9.0
    processParams.minQ: 1.0e-4
    processParams.maxQ: 1.0e-2
    """
    precisionPath = writeConfigFile(
        tmp_path,
        "config_seed_weight_precision.yaml",
        precisionConfig,
    )
    precisionParsed = readConfig(str(precisionPath))
    assert precisionParsed["observationArgs"].precisionMultiplierMin == pytest.approx(
        0.25
    )
    assert precisionParsed["observationArgs"].precisionMultiplierMax == pytest.approx(
        9.0
    )
    assert precisionParsed["processArgs"].minQ == pytest.approx(1.0e-4)
    assert precisionParsed["processArgs"].maxQ == pytest.approx(1.0e-2)
    for attrName, defaultKey in seedFieldMap.items():
        assert getattr(precisionParsed["observationArgs"], attrName) == profile[defaultKey]

    explicitConfig = """
    experimentName: seedWeightExplicit
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams:
      muncSeedWeight:
        enabled: false
        passes: 3
        min: 0.2
        max: 2.5
        studentT: false
        studentTdf: 3.5
      muncSeedProcess:
        minQ: 2.0e-5
        maxQ: 8.0e-3
    observationParams.precisionMultiplierMin: 0.5
    observationParams.precisionMultiplierMax: 4.0
    """
    explicitPath = writeConfigFile(
        tmp_path,
        "config_seed_weight_explicit.yaml",
        explicitConfig,
    )
    explicitParsed = readConfig(str(explicitPath))
    explicitArgs = explicitParsed["observationArgs"]
    assert explicitArgs.muncSeedWeightEnabled is False
    assert explicitArgs.muncSeedWeightPasses == 3
    assert explicitArgs.muncSeedWeightMin == pytest.approx(0.2)
    assert explicitArgs.muncSeedWeightMax == pytest.approx(2.5)
    assert explicitArgs.muncSeedWeightStudentT is False
    assert explicitArgs.muncSeedWeightStudentTdf == pytest.approx(3.5)
    assert explicitArgs.muncSeedProcessMinQ == pytest.approx(2.0e-5)
    assert explicitArgs.muncSeedProcessMaxQ == pytest.approx(8.0e-3)
    assert explicitParsed["observationArgs"].precisionMultiplierMin == pytest.approx(
        0.5
    )
    assert explicitParsed["observationArgs"].precisionMultiplierMax == pytest.approx(
        4.0
    )

    invalidCases = (
        ("scalar", "observationParams.muncSeedWeight: 0.35", "muncSeedWeight"),
        (
            "enabled string",
            'observationParams.muncSeedWeight.enabled: "false"',
            "enabled",
        ),
        (
            "passes bool",
            "observationParams.muncSeedWeight.passes: true",
            "passes",
        ),
        ("passes low", "observationParams.muncSeedWeight.passes: 1", "passes"),
        ("min high", "observationParams.muncSeedWeight.min: 1.5", "min"),
        ("max low", "observationParams.muncSeedWeight.max: 0.5", "max"),
        (
            "studentT string",
            'observationParams.muncSeedWeight.studentT: "false"',
            "studentT",
        ),
        (
            "studentTdf bool",
            "observationParams.muncSeedWeight.studentTdf: true",
            "studentTdf",
        ),
        ("studentTdf zero", "observationParams.muncSeedWeight.studentTdf: 0", "studentTdf"),
        ("process scalar", "observationParams.muncSeedProcess: 0.35", "muncSeedProcess"),
        (
            "process min bool",
            "observationParams.muncSeedProcess.minQ: true",
            "minQ",
        ),
        ("process min zero", "observationParams.muncSeedProcess.minQ: 0", "minQ"),
        (
            "process max bool",
            "observationParams.muncSeedProcess.maxQ: true",
            "maxQ",
        ),
        (
            "process max low",
            "observationParams.muncSeedProcess.minQ: 2.0e-3\n"
            "        observationParams.muncSeedProcess.maxQ: 1.0e-3",
            "maxQ",
        ),
    )
    for caseName, configLine, message in invalidCases:
        invalidConfig = f"""
        experimentName: seedWeightInvalid{caseName.replace(" ", "")}
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        {configLine}
        """
        invalidPath = writeConfigFile(
            tmp_path,
            f"config_seed_weight_invalid_{caseName.replace(' ', '_')}.yaml",
            invalidConfig,
        )
        with pytest.raises(ValueError, match=message):
            readConfig(str(invalidPath))


def _case_readConfigGenericDefaultsStillAllowExplicitOverrides(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    configuration: generic
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.t_innerIters: 7
    fitParams.ECM_outerIters: 16
    fitParams.ECM_backgroundLengthScaleMultiplier: 2.0
    countingParams.centerMB: false
    countingParams.centerMBMethod: savgol
    processParams.processNoiseCalibration: fixed
    processParams.processNoiseWarmupOuterPasses: 6
    processParams.precisionMultiplierMin: 0.5
    observationParams.lengthScaleBP: 12345
    observationParams.lengthScaleOptions:
      blockCount: 48
      blockWidthBP: 120000
      scaleEndBP: 45000
    observationParams.precisionMultiplierMax: 4.0
    outputParams.saveBackgroundTracks: false
    outputParams.saveGains: false
    outputParams.deleteBedGraphsAfterBigWig: false
    outputParams.stateShrinkageEnabled: false
    outputParams.stateShrinkageModel: spikeAndStudentT
    outputParams.stateShrinkagePriorSpikeProp: 0.33
    outputParams.stateShrinkageSpikeOddsMultiplier: 2.5
    outputParams.stateShrinkageSpikePseudoCount: 2.5
    outputParams.stateShrinkageScaleAnchorWeight: 9
    outputParams.stateShrinkageStudentTDF: 5
    outputParams.stateShrinkageStudentTQuadratureOrder: 32
    outputParams.writeReplicateExchangeabilityDiagnostics: true
    outputParams.plotNullCalibrationDiagnostics: false
    outputParams.plotPrecisionReweightingHistograms: false
    outputParams.precisionReweightingHistogramSampleSize: 12345
    uncertaintyCalibrationParams.enabled: false
    matchingParams.uncertaintyScoreMode: lower_confidence
    matchingParams.uncertaintyScoreZ: 1.75
    matchingParams.shrinkChromosomeBudgets: false
    matchingParams.useLocalBootStrapRadius: false
    """

    configPath = writeConfigFile(tmp_path, "config_generic_override.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["defaultConfiguration"] == "generic"
    assert parsed["fitArgs"].t_innerIters == 7
    assert parsed["fitArgs"].ECM_outerIters == 16
    assert parsed["fitArgs"].ECM_backgroundLengthScaleMultiplier == pytest.approx(2.0)
    assert parsed["countingArgs"].centerMB is False
    assert parsed["countingArgs"].centerMBMethod == "savgol"
    assert parsed["processArgs"].processNoiseCalibration == "fixed"
    assert parsed["processArgs"].processNoiseWarmupOuterPasses == 6
    assert parsed["processArgs"].precisionMultiplierMin == pytest.approx(0.5)
    assert parsed["observationArgs"].lengthScaleBP == 12345
    assert parsed["observationArgs"].lengthScaleOptions["blockCount"] == 48
    assert parsed["observationArgs"].lengthScaleOptions["blockWidthBP"] == 120000
    assert parsed["observationArgs"].lengthScaleOptions["scaleEndBP"] == 45000
    assert parsed["observationArgs"].precisionMultiplierMax == pytest.approx(4.0)
    assert parsed["outputArgs"].saveBackgroundTracks is False
    assert parsed["outputArgs"].saveGains is False
    assert parsed["outputArgs"].deleteBedGraphsAfterBigWig is False
    assert parsed["outputArgs"].stateShrinkageEnabled is False
    assert (
        parsed["outputArgs"].stateShrinkageModel
        == constants.OUTPUT_STATE_SHRINKAGE_MODEL_SPIKE_AND_STUDENT_T
    )
    assert parsed["outputArgs"].stateShrinkagePriorSpikeProp == pytest.approx(0.33)
    assert parsed["outputArgs"].stateShrinkageSpikeOddsMultiplier == pytest.approx(2.5)
    assert parsed["outputArgs"].stateShrinkageSpikePseudoCount == pytest.approx(2.5)
    assert parsed["outputArgs"].stateShrinkageScaleAnchorWeight == pytest.approx(9.0)
    assert parsed["outputArgs"].stateShrinkageStudentTDF == pytest.approx(5.0)
    assert parsed["outputArgs"].stateShrinkageStudentTQuadratureOrder == 32
    assert parsed["outputArgs"].writeReplicateExchangeabilityDiagnostics is True
    assert parsed["outputArgs"].plotNullCalibrationDiagnostics is False
    assert parsed["outputArgs"].plotPrecisionReweightingHistograms is False
    assert parsed["outputArgs"].precisionReweightingHistogramSampleSize == 12345
    assert parsed["uncertaintyCalibrationArgs"].enabled is False
    assert parsed["matchingArgs"].uncertaintyScoreMode == "lower_confidence"
    assert parsed["matchingArgs"].uncertaintyScoreZ == pytest.approx(1.75)
    assert parsed["matchingArgs"].shrinkChromosomeBudgets is False
    assert parsed["matchingArgs"].useLocalBootStrapRadius is False

    for configKey in (
        "outputParams.plotNullCalibrationDiagnostics",
        "matchingParams.shrinkChromosomeBudgets",
        "matchingParams.useLocalBootStrapRadius",
    ):
        for value in ("1", "'true'", "null", "[]", "{}"):
            invalidYaml = configYaml.replace(
                f"{configKey}: false",
                f"{configKey}: {value}",
            )
            invalidPath = writeConfigFile(tmp_path, "invalid.yaml", invalidYaml)
            with pytest.raises(ValueError) as error:
                readConfig(str(invalidPath))
            assert str(error.value) == f"{configKey} must be boolean"

def _case_readConfigRejectsLowStateShrinkageStudentTDF(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    outputParams.stateShrinkageModel: spikeAndStudentT
    outputParams.stateShrinkageStudentTDF: 0.5
    """
    configPath = writeConfigFile(tmp_path, "config_low_student_t_df.yaml", configYaml)

    with pytest.raises(ValueError, match="stateShrinkageStudentTDF"):
        readConfig(str(configPath))


def _case_readConfigRejectsInvalidStateShrinkageSpikeOddsMultiplier(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    for label, value in (
        ("bool", "true"),
        ("zero", "0"),
        ("negative", "-1"),
        ("inf", ".inf"),
    ):
        configYaml = f"""
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        outputParams.stateShrinkageSpikeOddsMultiplier: {value}
        """
        configPath = writeConfigFile(
            tmp_path,
            f"config_bad_state_shrinkage_spike_odds_multiplier_{label}.yaml",
            configYaml,
        )

        with pytest.raises(ValueError, match="stateShrinkageSpikeOddsMultiplier"):
            readConfig(str(configPath))


def _case_readConfigRejectsInvalidStateShrinkageEnabled(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    outputParams.stateShrinkageEnabled: 1
    """
    configPath = writeConfigFile(
        tmp_path,
        "config_bad_state_shrinkage_enabled.yaml",
        configYaml,
    )

    with pytest.raises(ValueError, match="stateShrinkageEnabled"):
        readConfig(str(configPath))


def _case_readConfigRejectsInvalidPrecisionReweightingHistogramSettings(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    for label, key, value in (
        ("plot", "outputParams.plotPrecisionReweightingHistograms", "1"),
        (
            "sample_zero",
            "outputParams.precisionReweightingHistogramSampleSize",
            "0",
        ),
        (
            "sample_negative",
            "outputParams.precisionReweightingHistogramSampleSize",
            "-1",
        ),
        (
            "sample_bool",
            "outputParams.precisionReweightingHistogramSampleSize",
            "true",
        ),
        (
            "sample_float",
            "outputParams.precisionReweightingHistogramSampleSize",
            "12.5",
        ),
        (
            "sample_inf",
            "outputParams.precisionReweightingHistogramSampleSize",
            ".inf",
        ),
    ):
        configYaml = f"""
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        {key}: {value}
        """
        configPath = writeConfigFile(
            tmp_path,
            f"config_bad_precision_histogram_{label}.yaml",
            configYaml,
        )

        with pytest.raises(ValueError, match=key.rsplit(".", 1)[-1]):
            readConfig(str(configPath))


def _case_processNoiseWarmupPassThroughUsesConfiguredKnobs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    processParams:
      processNoiseCalibration: fixed
      qSeedPriorLevel: 4.0e-8
      processNoiseWarmupECMIters: 9
      processNoiseWarmupOuterPasses: 4
      precisionMultiplierMin: 0.25
      precisionMultiplierMax: 9.0
    """

    configPath = writeConfigFile(
        tmp_path, "config_process_noise_passthrough.yaml", configYaml
    )
    processArgs = readConfig(str(configPath))["processArgs"]
    supportedProcessKwargs = {
        "processNoiseWarmupOuterPasses",
        "processNoiseCalibration",
    }
    monkeypatch.setattr(
        consenrich_cli,
        "_coreRunConsenrichSupports",
        lambda name: name in supportedProcessKwargs,
    )
    kwargs = consenrich_cli._processNoiseRunKwargs(processArgs)

    assert kwargs["processNoiseCalibration"] == "fixed"
    assert kwargs["qSeedPriorLevel"] == pytest.approx(4.0e-8)
    assert kwargs["processNoiseWarmupECMIters"] == 9
    assert kwargs["processPrecisionMultiplierMin"] == pytest.approx(0.25)
    assert kwargs["processPrecisionMultiplierMax"] == pytest.approx(9.0)
    assert kwargs["processNoiseWarmupOuterPasses"] == 4

    monkeypatch.setattr(
        consenrich_core,
        "PROCESS_DEFAULT_WARMUP_OUTER_PASSES",
        constants.PROCESS_DEFAULT_WARMUP_OUTER_PASSES,
        raising=False,
    )
    assert consenrich_cli._configureCoreProcessNoiseWarmupDefaults(processArgs) == 4
    expectedDefault = (
        constants.PROCESS_DEFAULT_WARMUP_OUTER_PASSES
        if consenrich_cli._coreRunConsenrichSupports("processNoiseWarmupOuterPasses")
        else 4
    )
    assert consenrich_core.PROCESS_DEFAULT_WARMUP_OUTER_PASSES == expectedDefault


def _case_readConfigRejectsUnknownDefaultConfiguration(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    configuration: narrow
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_unknown_defaults.yaml", configYaml)
    with pytest.raises(ValueError, match="Unsupported default configuration"):
        readConfig(str(configPath))


def _case_readConfigDeduplicatesChromosomes(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.chromosomes: [chr1, chr2, chr1, chr2, chr3]
    """

    configPath = writeConfigFile(tmp_path, "config_dedup.yaml", configYaml)
    configParsed = readConfig(str(configPath))

    assert configParsed["genomeArgs"].chromosomes == ["chr1", "chr2", "chr3"]


def _case_readConfigUsesZeroCenterIdentifiabilityFields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

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
    assert defaultFitArgs.ECM_backgroundLengthScaleMultiplier == pytest.approx(5.0)

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.ECM_zeroCenterBackground: false
    fitParams.useNonnegativeBackground: false
    fitParams.backgroundNegativePenaltyMultiplier: null
    """
    warningCalls = []
    monkeypatch.setattr(
        consenrich_config.logger, "warning", lambda *args: warningCalls.append(args)
    )
    for settingsText, expectedMultiplier, expectedWarnings in (
        ("fitParams.ECM_backgroundLengthScaleMultiplier: 6", 6.0, 0),
        ("fitParams: {backgroundLengthScaleMultiplier: 3}", 3.0, 0),
        ("fitParams.backgroundLengthScaleMultiplier: 4", 4.0, 0),
        ("fitParams: {backgroundLengthScaleMultiplier: 3, "
         "ECM_backgroundLengthScaleMultiplier: 3}", 3.0, 0),
        ("fitParams.backgroundLengthScaleMultiplier: 4\n"
         "fitParams.ECM_backgroundLengthScaleMultiplier: 6", 4.0, 1),
        ("fitParams: {backgroundLengthScaleMultiplier: 2, "
         "ECM_backgroundLengthScaleMultiplier: 6}", 2.0, 1),
    ):
        warningCalls.clear()
        parsedOverride = readConfig(
            writeConfigFile(
                tmp_path, "config_zero_center_override.yaml",
                textwrap.dedent(configOverrideYaml) + settingsText,
            )
        )
        assert parsedOverride["fitArgs"].ECM_zeroCenterBackground is False
        assert parsedOverride["fitArgs"].useNonnegativeBackground is False
        assert parsedOverride["fitArgs"].backgroundNegativePenaltyMultiplier is None
        assert parsedOverride[
            "fitArgs"
        ].ECM_backgroundLengthScaleMultiplier == pytest.approx(expectedMultiplier)
        assert len(warningCalls) == expectedWarnings
        if expectedWarnings:
            assert "fitParams.backgroundLengthScaleMultiplier" in warningCalls[0][0]
            assert warningCalls[0][-1] == expectedMultiplier


def _case_readConfigAllowsEMTNuOverride(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.ECM_robustTNu: 4.0
    fitParams.ECM_processRobustTNu: 6.0
    """
    parsedOverride = readConfig(
        str(
            writeConfigFile(tmp_path, "config_em_tnu_override.yaml", configOverrideYaml)
        )
    )
    assert parsedOverride["fitArgs"].ECM_robustTNu == pytest.approx(4.0)
    assert parsedOverride["fitArgs"].ECM_processRobustTNu == pytest.approx(6.0)

    for rawValue in ("0", ".nan", "true"):
        invalidYaml = f"""
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        fitParams.ECM_processRobustTNu: {rawValue}
        """
        invalidPath = writeConfigFile(
            tmp_path,
            f"config_process_tnu_invalid_{rawValue.replace('.', '_')}.yaml",
            invalidYaml,
        )
        with pytest.raises(ValueError, match="ECM_processRobustTNu"):
            readConfig(str(invalidPath))


def _case_readConfigUsesECMAndOuterPassToleranceFields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

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

    for rawValue in ("0", "1.5"):
        invalidYaml = f"""
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        fitParams.t_innerIters: {rawValue}
        """
        invalidPath = writeConfigFile(
            tmp_path,
            f"config_t_inner_iters_invalid_{rawValue.replace('.', '_')}.yaml",
            invalidYaml,
        )
        with pytest.raises(ValueError, match="fitParams.t_innerIters"):
            readConfig(str(invalidPath))


def _case_readConfigUsesUncertaintyCalibrationFields(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)

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
    defaultArgs = parsedDefault["uncertaintyCalibrationArgs"]
    defaultOutputArgs = parsedDefault["outputArgs"]
    assert defaultArgs.enabled is True
    assert defaultArgs.refitThreads == 2
    assert defaultOutputArgs.plotPrecisionReweightingHistograms is True
    assert defaultArgs.mode == constants.UNCERTAINTY_CALIBRATION_DEFAULT_MODE
    assert (
        defaultArgs.deleteBlockVarianceMode
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_VARIANCE_MODE
    )
    assert (
        defaultArgs.deleteBlockScoreWeightMode
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_SCORE_WEIGHT_MODE
    )
    assert (
        defaultArgs.deleteBlockFactorModel
        == constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_SEG_SHRINK
    )
    assert (
        defaultArgs.calibrationOuterIters
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_CALIBRATION_OUTER_ITERS
    )
    assert (
        defaultArgs.deleteBlockFactorSegmentCount
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_SEGMENT_COUNT
    )
    assert (
        defaultArgs.deleteBlockFactorBootstrapReplicates
        == 100
    )
    assert defaultArgs.deleteBlockDeletionProbability == pytest.approx(0.25)
    assert (
        defaultArgs.deleteBlockReplicateDependenceRho
        == constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_REPLICATE_DEPENDENCE_RHO_AUTO
    )
    assert not hasattr(defaultArgs, "holdoutCount")
    assert defaultArgs.deleteBlockApplyTargetCalibration is None

    configExplicitYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    uncertaintyCalibrationParams.enabled: false
    uncertaintyCalibrationParams.mode: delete_block_state
    uncertaintyCalibrationParams.blockSizeBP: 25000
    uncertaintyCalibrationParams.folds: 3
    uncertaintyCalibrationParams.refitThreads: 1
    uncertaintyCalibrationParams.maxScores: 1234
    uncertaintyCalibrationParams.calibrationOuterIters: 4
    uncertaintyCalibrationParams.targets: [0.5, 0.9]
    uncertaintyCalibrationParams.targetCalibrationDelta: 0.025
    uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: false
    uncertaintyCalibrationParams.deleteBlockVarianceMode: covariance_difference
    uncertaintyCalibrationParams.deleteBlockUseLambdaInInformation: true
    uncertaintyCalibrationParams.deleteBlockReplicateDependenceRho: 0.35
    uncertaintyCalibrationParams.deleteBlockTargetSignal: state_plus_background
    uncertaintyCalibrationParams.deleteBlockFactorModel: segShrink
    uncertaintyCalibrationParams.deleteBlockFactorSegmentCount: 7
    uncertaintyCalibrationParams.deleteBlockFactorBootstrapReplicates: 9
    uncertaintyCalibrationParams.deleteBlockDeletionProbability: 0.4
    uncertaintyCalibrationParams.deleteBlockMinInformationFraction: 0.01
    uncertaintyCalibrationParams.deleteBlockMaxInformationFraction: 0.8
    uncertaintyCalibrationParams.deleteBlockMinDeltaVariance: 1.0e-7
    uncertaintyCalibrationParams.deleteBlockFallbackMinValidFraction: 0.5
    uncertaintyCalibrationParams.deleteBlockScoreWeightMode: sqrt_information_fraction
    uncertaintyCalibrationParams.deleteBlockApplyTargetCalibration: true
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
    assert explicitArgs.mode == "delete_block_state"
    assert explicitArgs.blockSizeBP == 25_000
    assert explicitArgs.folds == 3
    assert explicitArgs.refitThreads == 1
    assert explicitArgs.maxScores == 1234
    assert explicitArgs.calibrationOuterIters == 4
    assert explicitArgs.targets == (0.5, 0.9)
    assert explicitArgs.targetCalibrationDelta == pytest.approx(0.025)
    assert explicitArgs.scaleUncertaintyByTargetCalibration is False
    assert explicitArgs.deleteBlockVarianceMode == "covariance_difference"
    assert explicitArgs.deleteBlockUseLambdaInInformation is True
    assert explicitArgs.deleteBlockReplicateDependenceRho == pytest.approx(0.35)
    assert explicitArgs.deleteBlockTargetSignal == "state_plus_background"
    assert explicitArgs.deleteBlockFactorModel == "segShrink"
    assert explicitArgs.deleteBlockFactorSegmentCount == 7
    assert explicitArgs.deleteBlockFactorBootstrapReplicates == 9
    assert explicitArgs.deleteBlockDeletionProbability == pytest.approx(0.4)
    assert explicitArgs.deleteBlockMinInformationFraction == pytest.approx(0.01)
    assert explicitArgs.deleteBlockMaxInformationFraction == pytest.approx(0.8)
    assert explicitArgs.deleteBlockMinDeltaVariance == pytest.approx(1.0e-7)
    assert explicitArgs.deleteBlockFallbackMinValidFraction == pytest.approx(0.5)
    assert explicitArgs.deleteBlockScoreWeightMode == "sqrt_information_fraction"
    assert explicitArgs.deleteBlockApplyTargetCalibration is True

    configAutoRhoYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    uncertaintyCalibrationParams.deleteBlockReplicateDependenceRho: auto
    """
    parsedAutoRho = readConfig(
        str(
            writeConfigFile(
                tmp_path,
                "config_uncertainty_calibration_auto_rho.yaml",
                configAutoRhoYaml,
            )
        )
    )
    assert (
        parsedAutoRho["uncertaintyCalibrationArgs"].deleteBlockReplicateDependenceRho
        == constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_REPLICATE_DEPENDENCE_RHO_AUTO
    )

    configPredictiveYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    uncertaintyCalibrationParams.mode: predictive_holdout
    """
    configPredictivePath = writeConfigFile(
        tmp_path,
        "config_uncertainty_calibration_predictive_mode.yaml",
        configPredictiveYaml,
    )
    with pytest.raises(ValueError, match="uncertaintyCalibrationParams.mode"):
        readConfig(str(configPredictivePath))

    for value in ("seg-shrink", "seg_shrink", "segshrink", "SegShrink"):
        configAliasYaml = f"""
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        uncertaintyCalibrationParams.deleteBlockFactorModel: {value}
        """
        configAliasPath = writeConfigFile(
            tmp_path,
            f"config_uncertainty_calibration_alias_{value}.yaml",
            configAliasYaml,
        )
        with pytest.raises(ValueError, match="deleteBlockFactorModel"):
            readConfig(str(configAliasPath))

    for key, value in (
        ("refitThreads", 0),
        ("refitThreads", -1),
        ("refitThreads", "false"),
        ("refitThreads", 1.5),
        ("refitThreads", "'2'"),
        ("refitThreads", "null"),
        ("deleteBlockFactorSegmentCount", 0),
        ("deleteBlockFactorBootstrapReplicates", 7),
    ):
        configInvalidYaml = f"""
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        uncertaintyCalibrationParams.{key}: {value}
        """
        configInvalidPath = writeConfigFile(
            tmp_path,
            f"config_uncertainty_calibration_invalid_{key}.yaml",
            configInvalidYaml,
        )
        with pytest.raises(ValueError, match=key):
            readConfig(str(configInvalidPath))

    invalidDeleteBlockScoringCases = (
        (
            "fraction_order",
            (
                "uncertaintyCalibrationParams.deleteBlockMinInformationFraction: 0.8",
                "uncertaintyCalibrationParams.deleteBlockMaxInformationFraction: 0.8",
            ),
            "information fractions",
        ),
        (
            "min_delta_variance",
            ("uncertaintyCalibrationParams.deleteBlockMinDeltaVariance: 0",),
            "deleteBlockMinDeltaVariance",
        ),
        (
            "score_weight_mode",
            (
                "uncertaintyCalibrationParams.deleteBlockScoreWeightMode: sqrtInformationFraction",
            ),
            "deleteBlockScoreWeightMode",
        ),
        (
            "replicate_dependence_rho_negative",
            ("uncertaintyCalibrationParams.deleteBlockReplicateDependenceRho: -0.1",),
            "deleteBlockReplicateDependenceRho",
        ),
        (
            "replicate_dependence_rho_one",
            ("uncertaintyCalibrationParams.deleteBlockReplicateDependenceRho: 1",),
            "deleteBlockReplicateDependenceRho",
        ),
        (
            "replicate_dependence_rho_bool",
            ("uncertaintyCalibrationParams.deleteBlockReplicateDependenceRho: false",),
            "deleteBlockReplicateDependenceRho",
        ),
        (
            "delete_probability",
            ("uncertaintyCalibrationParams.deleteBlockDeletionProbability: 0",),
            "deleteBlockDeletionProbability",
        ),
        (
            "delete_probability_one",
            ("uncertaintyCalibrationParams.deleteBlockDeletionProbability: 1",),
            "deleteBlockDeletionProbability",
        ),
        (
            "targets_empty",
            ("uncertaintyCalibrationParams.targets: []",),
            "targets",
        ),
        (
            "target_delta_zero",
            ("uncertaintyCalibrationParams.targetCalibrationDelta: 0",),
            "targetCalibrationDelta",
        ),
        (
            "target_delta_null_scaling",
            (
                "uncertaintyCalibrationParams.targetCalibrationDelta:",
                "uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: true",
            ),
            "targetCalibrationDelta",
        ),
        (
            "target_scale_bool",
            (
                "uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: 'false'",
            ),
            "scaleUncertaintyByTargetCalibration",
        ),
    )
    for caseName, configLines, message in invalidDeleteBlockScoringCases:
        configLinesYaml = "\n        ".join(configLines)
        configInvalidYaml = f"""
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        {configLinesYaml}
        """
        configInvalidPath = writeConfigFile(
            tmp_path,
            f"config_uncertainty_calibration_invalid_{caseName}.yaml",
            configInvalidYaml,
        )
        with pytest.raises(ValueError, match=message):
            readConfig(str(configInvalidPath))


def _case_readConfigNumNearestRequiresExplicitSparseBed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
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


def _case_readConfigRestrictLocalVarianceToSparseBedRequiresAvailableSparseBed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    sparseBedPath = tmp_path / "explicit_sparse.bed"
    sparseBedPath.write_text("chrTest\t0\t100\n", encoding="utf-8")

    configNoSparse = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.restrictLocalVarianceToSparseBed: true
    """
    configNoSparsePath = writeConfigFile(
        tmp_path,
        "config_restrict_local_variance_no_sparse.yaml",
        configNoSparse,
    )
    parsedNoSparse = readConfig(str(configNoSparsePath))
    assert parsedNoSparse["observationArgs"].restrictLocalVarianceToSparseBed is False

    configExplicitSparse = f"""
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.sparseBedFile: {sparseBedPath}
    observationParams.restrictLocalVarianceToSparseBed: true
    observationParams.muncVarianceModel: kalman
    observationParams.lengthScaleBP: 10000
    observationParams.muncEBPrior.tileSizeBP: 1000
    observationParams.muncEBPrior.tileCount: 17
    observationParams.muncEBPrior.strata: 4
    observationParams.muncEBPrior.minTilesPerStratum: 2
    observationParams.muncEBPrior.seed: 123
    observationParams.muncEBPrior.supportMinQ: 0.05
    observationParams.muncEBPrior.supportMaxQ: 0.95
    observationParams.muncEBPrior.maxExtrapolatedFraction: 0.12
    observationParams.muncEBPrior.warmupECMIters: 9
    observationParams.muncEBPrior.warmupOuterPasses: 2
    observationParams.muncEBPrior.gUncertaintyMode: disabled
    """
    configExplicitSparsePath = writeConfigFile(
        tmp_path,
        "config_restrict_local_variance_explicit_sparse.yaml",
        configExplicitSparse,
    )
    parsedExplicitSparse = readConfig(str(configExplicitSparsePath))
    explicitObservationArgs = parsedExplicitSparse["observationArgs"]
    assert explicitObservationArgs.restrictLocalVarianceToSparseBed is True
    assert (
        explicitObservationArgs.muncVarianceModel == constants.MUNC_VARIANCE_MODEL_KALMAN
    )
    assert explicitObservationArgs.lengthScaleBP == 10000
    assert explicitObservationArgs.muncEBPriorTileSizeBP == 1000
    assert explicitObservationArgs.muncEBPriorTileCount == 17
    assert explicitObservationArgs.muncEBPriorStrata == 4
    assert explicitObservationArgs.muncEBPriorMinTilesPerStratum == 2
    assert explicitObservationArgs.muncEBPriorSeed == 123
    assert explicitObservationArgs.muncEBPriorSupportMinQ == 0.05
    assert explicitObservationArgs.muncEBPriorSupportMaxQ == 0.95
    assert explicitObservationArgs.muncEBPriorMaxExtrapolatedFraction == 0.12
    assert explicitObservationArgs.muncEBPriorWarmupECMIters == 9
    assert explicitObservationArgs.muncEBPriorWarmupOuterPasses == 2
    assert (
        explicitObservationArgs.muncEBPriorGUncertaintyMode
        == constants.MUNC_EB_PRIOR_G_UNCERTAINTY_MODE_DISABLED
    )

    configInvalidModel = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncVarianceModel: ar1
    """
    configInvalidModelPath = writeConfigFile(
        tmp_path,
        "config_invalid_munc_model.yaml",
        configInvalidModel,
    )
    with pytest.raises(ValueError, match="MUNC variance model"):
        readConfig(str(configInvalidModelPath))

    configInvalidGMode = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncEBPrior.gUncertaintyMode: exact
    """
    configInvalidGModePath = writeConfigFile(
        tmp_path,
        "config_invalid_munc_g_mode.yaml",
        configInvalidGMode,
    )
    with pytest.raises(ValueError, match="muncEBPrior.gUncertaintyMode"):
        readConfig(str(configInvalidGModePath))

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






def _case_readConfigSamplesSupportExplicitBedGraphFormat(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    signalPath = tmp_path / "sample.signal"
    signalPath.write_text("chrTest\t0\t50\t3.0\n", encoding="ascii")
    indexedPath = Path(f"{signalPath}.gz")

    configYaml = f"""
    experimentName: sampleExperiment
    inputParams:
      samples:
        - path: {signalPath}
          format: BEDGRAPH
          role: treatment
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_bedgraph_format.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    inputArgs = configParsed["inputArgs"]

    assert inputArgs.treatmentSources is not None
    assert inputArgs.treatmentSources[0].sourceKind == "BEDGRAPH"
    assert inputArgs.bamFiles == [str(signalPath)]
    assert not indexedPath.exists()
    assert not Path(f"{indexedPath}.tbi").exists()






def _case_readConfigRejectsCRAMSources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)

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
    with pytest.raises(ValueError, match="CRAM inputs are unsupported|Unsupported source kind"):
        readConfig(str(configPath))


def _case_convertBedGraphToBigWigPyBigWigWritesExpectedTrack(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    bedGraphPath = tmp_path / "toy.bedGraph"
    chromSizesPath = tmp_path / "toy.chrom.sizes"
    chroms = ["chr1", "chr2", "chr10"]
    rows = [
        ("chr1", 0, 50, 1.000000059604645), ("chr1", 50, 100, -1.0), ("chr1", 100, 150, 2.0),
        ("chr2", 0, 50, 0.5), ("chr2", 60, 110, -1.0), ("chr2", 200, 250, 2.0),
        ("chr10", 0, 25, 0.5), ("chr10", 60, 110, -1.0), ("chr10", 200, 275, 2.0),
    ]
    bedGraphPath.write_text(
        "track type=bedGraph name=toy\nbrowser position chr1:1-150\n"
        + "".join(
            (" " if index % 2 else "\t").join(map(str, row)) + "\n"
            for index, row in enumerate(rows)
        ),
        encoding="ascii",
    )
    chromSizesPath.write_text(
        "".join(f"{chrom}\t1000\n" for chrom in chroms),
        encoding="ascii",
    )
    for sourceIndex, sizeSource in enumerate((
        str(chromSizesPath), types.MappingProxyType(dict.fromkeys(chroms, 1000)),
    )):
        for validated in (False, True):
            for chunkSize in (2, 20):
                output = tmp_path / f"tracks{sourceIndex}{validated}{chunkSize}.bw"
                consenrich_io._convertBedGraphToBigWigPyBigWig(
                    str(bedGraphPath), sizeSource, str(output),
                    chunkSize=chunkSize, validated=validated,
                )
                with pyBigWig.open(str(output)) as track:
                    assert list(track.chroms()) == chroms
                    for chrom in chroms:
                        expected = tuple(
                            (row[1], row[2], float(np.float32(row[3])))
                            for row in rows if row[0] == chrom
                        )
                        assert track.intervals(chrom) == expected
                    header = track.header()
                    assert header["nBasesCovered"] == 450
                    assert header["minVal"] == -1
                    assert header["maxVal"] == 2
                    assert header["sumData"] == 287
                    assert header["sumSquared"] == 918


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


def test_nativeBedGraphChunkValidation(tmp_path):
    path = tmp_path / "tracks.bedGraph"
    chunks = {}
    starts = np.asarray([0, 50], dtype=np.int64)
    ends = starts + 50
    for chrom, values in (("chr22", [0.5, 1.0]), ("chr6", [-1.0, 2.0])):
        consenrich_io._writeBedGraphChunk(
            str(path), chrom, starts, ends, np.asarray(values), chunks,
        )
    consenrich_io._reorderBedGraphChunks(str(path), chunks, ["chr6", "chr22"])
    consenrich_io._validateBedGraphSorted(str(path), chromOrder=["chr6", "chr22"])
    assert path.read_text().splitlines() == [
        "chr6\t0\t50\t-1.00000", "chr6\t50\t100\t2.00000",
        "chr22\t0\t50\t0.50000", "chr22\t50\t100\t1.00000",
    ]
    for invalidStarts, invalidEnds, invalidValues in (
        ([-1, 50], [50, 100], [0.0, 1.0]),
        ([0, 50], [0, 100], [0.0, 1.0]),
        ([50, 0], [100, 50], [0.0, 1.0]),
        ([0, 25], [50, 75], [0.0, 1.0]),
        ([0, 50], [50, 100], [0.0, np.nan]),
    ):
        with pytest.raises(ValueError):
            consenrich_io._writeBedGraphChunk(
                str(path), "chr11", np.asarray(invalidStarts, dtype=np.int64),
                np.asarray(invalidEnds, dtype=np.int64), np.asarray(invalidValues), chunks,
            )
        assert "chr11" not in chunks
        assert len(path.read_text().splitlines()) == 4


@pytest.mark.parametrize("sizeSourceKind", ("file", "mapping"))
def test_convertBedGraphToBigWigSkipsValidatedBedGraphScan(tmp_path, monkeypatch, sizeSourceKind):
    experimentName = "toy"
    version = consenrich_io.__version__
    chromSizesPath = tmp_path / "toy.chrom.sizes"
    if sizeSourceKind == "file":
        chromSizesPath.write_text("chr1\t100\n", encoding="ascii")
    sizeSource = (str(chromSizesPath) if sizeSourceKind == "file"
                  else types.MappingProxyType({"chr1": 100}))
    suffixes = ["state", "uncertainty"]
    bedGraphPaths = {
        suffix: (
            tmp_path
            / f"consenrichOutput_{experimentName}_{suffix}.v{version}.bedGraph"
        )
        for suffix in suffixes
    }
    for bedGraphPath in bedGraphPaths.values():
        bedGraphPath.write_text("chr1\t0\t10\t1.0\n", encoding="ascii")

    validateCalls = []
    readCalls = []
    convertCalls = []
    realReadChromSizes = misc_util.counts.readChromSizes

    def recordReadChromSizes(path):
        readCalls.append(path)
        return realReadChromSizes(path)

    def recordValidate(path, chromOrder=None):
        validateCalls.append((path, tuple(chromOrder or ())))

    def recordConvert(
        bedGraphPath,
        chromSizesFile,
        bigWigPath,
        chunkSize=200_000,
        *,
        chromSizes=None,
        validated=False,
    ):
        convertCalls.append((bedGraphPath, chromSizesFile, bigWigPath, chromSizes, validated))
        Path(bigWigPath).write_bytes(b"x" * 128)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(misc_util.counts, "readChromSizes", recordReadChromSizes)
    monkeypatch.setattr(consenrich_io, "_validateBedGraphSorted", recordValidate)
    monkeypatch.setattr(
        consenrich_io,
        "_convertBedGraphToBigWigPyBigWig",
        recordConvert,
    )

    consenrich_io.convertBedGraphToBigWig(
        experimentName,
        sizeSource,
        suffixes=suffixes,
        validatedBedGraphs={str(bedGraphPaths["state"])},
        deleteBedGraphsAfterBigWig=True,
    )

    assert readCalls == ([str(chromSizesPath)] if sizeSourceKind == "file" else [])
    assert validateCalls == [
        (
            f"consenrichOutput_{experimentName}_uncertainty.v{version}.bedGraph",
            ("chr1",),
        )
    ]
    assert [call[0] for call in convertCalls] == [
        f"consenrichOutput_{experimentName}_state.v{version}.bedGraph",
        f"consenrichOutput_{experimentName}_uncertainty.v{version}.bedGraph",
    ]
    assert all(call[3] == [("chr1", 100)] for call in convertCalls)
    assert [call[4] for call in convertCalls] == [True, False]
    assert all(not bedGraphPath.exists() for bedGraphPath in bedGraphPaths.values())


def test_convertBedGraphToBigWigValidatedSkipStillRejectsUnsortedTrack(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("pyBigWig")
    experimentName = "toy"
    version = consenrich_io.__version__
    chromSizesPath = tmp_path / "toy.chrom.sizes"
    chromSizesPath.write_text("chr1\t100\n", encoding="ascii")
    bedGraphPath = (
        tmp_path / f"consenrichOutput_{experimentName}_state.v{version}.bedGraph"
    )
    bedGraphPath.write_text(
        "\n".join(
            [
                "chr1\t10\t20\t1.0",
                "chr1\t0\t5\t2.0",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    validateCalls = []
    realValidate = consenrich_io._validateBedGraphSorted

    def recordValidate(path, chromOrder=None):
        validateCalls.append(path)
        return realValidate(path, chromOrder=chromOrder)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(consenrich_io, "_validateBedGraphSorted", recordValidate)

    consenrich_io.convertBedGraphToBigWig(
        experimentName,
        str(chromSizesPath),
        suffixes=["state"],
        validatedBedGraphs={str(bedGraphPath)},
        deleteBedGraphsAfterBigWig=True,
    )

    assert validateCalls == []
    assert not (tmp_path / f"{experimentName}_consenrich_state.v{version}.bw").exists()
    assert bedGraphPath.exists()


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

    consenrich_io._validateBedGraphSorted(
        str(bedGraphPath), chromOrder=["chr2", "chr1"]
    )
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
                "chr1\t10\t20\t1e-06",
                "chr2\t10\t20\t2.50001",
                "chr1\t0\t10\t1.00001",
                "chr2\t0\t10\t2.00001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    consenrich_io._sortBedGraphInPlace(str(unsortedPath), chromOrder=["chr2", "chr1"])

    assert unsortedPath.read_text(encoding="utf-8").splitlines() == [
        "track type=bedGraph name=toy",
        "browser position chr2:1-20",
        "chr2\t0\t10\t2.00001",
        "chr2\t10\t20\t2.50001",
        "chr1\t0\t10\t1.00001",
        "chr1\t10\t20\t1e-06",
    ]


def _case_sortBedGraphUsesSizesOrderFilteredToRequestedChromosomes(tmp_path):
    bedGraphPath = tmp_path / "planned_order.bedGraph"
    chromSizesPath = tmp_path / "planned_order.chrom.sizes"
    chromSizesPath.write_text(
        "chr1\t100\nchr2\t100\nchr10\t100\nchrM\t100\n",
        encoding="ascii",
    )
    requestedChromosomes = ["chr10", "chr1", "chr10"]
    chromSizes = {
        chrom: int(size)
        for chrom, size in (
            line.split("\t")
            for line in chromSizesPath.read_text(encoding="ascii").splitlines()
        )
    }
    chromOrder = consenrich_cli._chromSizesOrderForChromosomes(
        chromSizes,
        requestedChromosomes,
    )
    bedGraphPath.write_text(
        "\n".join(
            [
                "chr10\t20\t30\t10.0",
                "chr1\t10\t20\t1.0",
                "chr10\t0\t10\t8.0",
                "chr1\t0\t10\t0.5",
                "chr10\t10\t20\t9.0",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    consenrich_io._sortBedGraphInPlace(str(bedGraphPath), chromOrder=chromOrder)
    consenrich_io._validateBedGraphSorted(str(bedGraphPath), chromOrder=chromOrder)

    rows = [
        line.split("\t")
        for line in bedGraphPath.read_text(encoding="utf-8").splitlines()
    ]
    emittedChromosomes = [row[0] for row in rows]
    chromosomeRuns = [
        chrom
        for index, chrom in enumerate(emittedChromosomes)
        if index == 0 or chrom != emittedChromosomes[index - 1]
    ]
    startsByChromosome = {
        chrom: [int(row[1]) for row in rows if row[0] == chrom]
        for chrom in chromosomeRuns
    }

    assert chromOrder == ["chr1", "chr10"]
    assert chromosomeRuns == ["chr1", "chr10"]
    assert "chr2" not in emittedChromosomes
    assert chromosomeRuns.count("chr10") == 1
    assert startsByChromosome == {
        chrom: sorted(starts) for chrom, starts in startsByChromosome.items()
    }


def _case_resolveFixedDeltaFRequiresPositiveFinite():
    assert consenrich_core._resolveFixedDeltaF(0.25) == pytest.approx(0.25)

    for badDeltaF in [0.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError, match="deltaF"):
            consenrich_core._resolveFixedDeltaF(badDeltaF)




def _run_with_monkeypatch(monkeypatch, func, *args):
    with monkeypatch.context() as mp:
        return func(*args, mp)


def test_config_runtime_validation_contracts(tmp_path, contract_case):
    contract_case(
        "replicate gain summary",
        _caseReplicateGainSummaryWritesPooledAverageAndStd,
        tmp_path,
    )
    contract_case(
        "fixed deltaF validation", _case_resolveFixedDeltaFRequiresPositiveFinite
    )


def test_config_worker_and_input_helper_contracts(monkeypatch, contract_case):
    for label, func in (
        (
            "unknown memory worker cap",
            _case_munc_worker_count_unknown_memory_uses_cpu_cap,
        ),
        (
            "low memory keeps one worker",
            _case_munc_worker_count_low_memory_keeps_one_worker,
        ),
        (
            "moderate memory caps workers",
            _case_munc_worker_count_moderate_memory_caps_below_cpu,
        ),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func)
    contract_case("input presence validation", _case_ensureInput)
    contract_case(
        "scale factor singleton broadcasting",
        _caseScaleFactorNormalizationBroadcastsSingletons,
    )


@pytest.mark.parametrize(
    "scaleObs,scaleProcess",
    [(True, False), (False, True)],
)
def test_precision_median_scaling_config_options(
    tmp_path,
    monkeypatch,
    scaleObs,
    scaleProcess,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    defaultPath = writeConfigFile(
        tmp_path,
        "config_precision_median_defaults.yaml",
        """
        experimentName: precisionMedianDefaults
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        """,
    )
    enabledPath = writeConfigFile(
        tmp_path,
        "config_precision_median_enabled.yaml",
        f"""
        experimentName: precisionMedianEnabled
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        fitParams:
          ECM_scaleObsPrecisionToMedian: {str(scaleObs).lower()}
          ECM_scaleProcessPrecisionToMedian: {str(scaleProcess).lower()}
        """,
    )
    defaultFitArgs = readConfig(str(defaultPath))["fitArgs"]
    enabledFitArgs = readConfig(str(enabledPath))["fitArgs"]
    profile = constants.DEFAULT_CONFIGURATION_VALUES[
        constants.GENERIC_DEFAULT_CONFIGURATION
    ]

    assert constants.FIT_DEFAULT_SCALE_OBS_PRECISION_TO_MEDIAN is False
    assert constants.FIT_DEFAULT_SCALE_PROCESS_PRECISION_TO_MEDIAN is True
    for fieldName, defaultValue in (
        (
            "ECM_scaleObsPrecisionToMedian",
            constants.FIT_DEFAULT_SCALE_OBS_PRECISION_TO_MEDIAN,
        ),
        (
            "ECM_scaleProcessPrecisionToMedian",
            constants.FIT_DEFAULT_SCALE_PROCESS_PRECISION_TO_MEDIAN,
        ),
    ):
        assert getattr(defaultFitArgs, fieldName) is defaultValue
        assert getattr(consenrich_core.fitParams(), fieldName) is defaultValue
        assert profile[f"fitParams.{fieldName}"] is defaultValue
    assert enabledFitArgs.ECM_scaleObsPrecisionToMedian is scaleObs
    assert enabledFitArgs.ECM_scaleProcessPrecisionToMedian is scaleProcess


def test_config_parser_defaults_and_override_contracts(
    tmp_path, monkeypatch, contract_case
):
    for label, func in (
        ("output diagnostic tracks", _case_readConfigOutputDiagnosticTracks),
        (
            "shared control scale factor broadcasting",
            _case_readConfigBroadcastsSharedControlScaleFactor,
        ),
        ("process noise options", _case_readConfigProcessNoiseOptions),
        ("configuration keys", _case_loadConfigValidatesKeys),
        ("generic profile", _case_readConfigUsesGenericDefaultConfiguration),
        (
            "MUNC seed weight defaults",
            _case_observationMuncSeedWeightDefaultAndPrecisionIndependence,
        ),
        (
            "generic overrides",
            _case_readConfigGenericDefaultsStillAllowExplicitOverrides,
        ),
        (
            "state shrinkage Student-t df validation",
            _case_readConfigRejectsLowStateShrinkageStudentTDF,
        ),
        (
            "state shrinkage spike odds multiplier validation",
            _case_readConfigRejectsInvalidStateShrinkageSpikeOddsMultiplier,
        ),
        (
            "state shrinkage enabled validation",
            _case_readConfigRejectsInvalidStateShrinkageEnabled,
        ),
        (
            "precision histogram validation",
            _case_readConfigRejectsInvalidPrecisionReweightingHistogramSettings,
        ),
        (
            "process noise warmup pass-through",
            _case_processNoiseWarmupPassThroughUsesConfiguredKnobs,
        ),
        (
            "unknown default profile rejected",
            _case_readConfigRejectsUnknownDefaultConfiguration,
        ),
        (
            "unsupported centerMB method rejected",
            _case_readConfigRejectsUnsupportedCenterMBMethod,
        ),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)
    contract_case(
        "canonical uncertainty default keys",
        _caseGenericDefaultConfigurationUsesCanonicalUncertaintyKeys,
    )


def test_config_model_parameter_field_contracts(tmp_path, monkeypatch, contract_case):
    for label, func in (
        ("chromosome deduplication", _case_readConfigDeduplicatesChromosomes),
        (
            "zero-center identifiability fields",
            _case_readConfigUsesZeroCenterIdentifiabilityFields,
        ),
        ("ECM t-nu override", _case_readConfigAllowsEMTNuOverride),
        (
            "ECM outer-pass tolerance fields",
            _case_readConfigUsesECMAndOuterPassToleranceFields,
        ),
        (
            "uncertainty calibration fields",
            _case_readConfigUsesUncertaintyCalibrationFields,
        ),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)


def test_optimization_path_output_helpers(tmp_path, monkeypatch):
    diagnostics = {
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

    precisionPath = tmp_path / "precision.jsonl"
    convergencePath = tmp_path / "convergence.jsonl"
    consenrich_cli._initializeDiagnosticLogs(
        consenrich_cli.DiagnosticLogPaths(
            precision=precisionPath,
            convergence=convergencePath,
            delete_block_calibration=tmp_path / "delete.jsonl",
        )
    )
    consenrich_cli._appendConvergenceDiagnostics(rows, convergencePath)
    records = [
        json.loads(line)
        for line in convergencePath.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record_type"] for record in records] == ["trace"] * 3
    assert [record["record_order"] for record in records] == [0, 1, 2]
    assert records[0]["objective_value"] == pytest.approx(12.5)

    precisionDiagnostics = {
        "lambdaExp": np.asarray([0.5, 2.0], dtype=np.float64),
        "processPrecExp": np.asarray([1.5, 0.75], dtype=np.float64),
        "matrixQ0": np.diag([0.2, 0.05]),
        "observationPrecisionMultiplierMin": 0.5,
        "observationPrecisionMultiplierMax": 2.0,
        "processPrecisionMultiplierMin": 0.75,
        "processPrecisionMultiplierMax": 1.5,
        "outputTracks": {
            "muncTrace": np.asarray([0.4, 0.2], dtype=np.float64),
            "sumGain0": np.asarray([0.1, 0.2], dtype=np.float64),
            "sumGain1": np.asarray([0.3, 0.4], dtype=np.float64),
        },
    }
    precisionFrame = consenrich_cli._precisionDiagnosticsFrame(
        chromosome="chrTest",
        intervals=np.asarray([0, 50], dtype=np.int64),
        intervalSizeBP=50,
        matrixMunc=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        pad=0.01,
        precisionDiagnostics=precisionDiagnostics,
    )
    assert (
        consenrich_cli._appendMuncLambdaDiagnostics(
            precisionFrame,
            precisionPath,
            chromosome="chrTest",
            precisionDiagnostics=precisionDiagnostics,
            detail="full",
        ),
        consenrich_cli._appendProcessKappaDiagnostics(
            precisionFrame,
            precisionPath,
            chromosome="chrTest",
            precisionDiagnostics=precisionDiagnostics,
            runDiagnostics={
                "process_noise_calibration": {
                    "processNoiseCalibrationStatus": "ok",
                }
            },
            detail="full",
        ),
    ) == (2, 2)
    precisionRecords = [
        json.loads(line)
        for line in precisionPath.read_text(encoding="utf-8").splitlines()
    ]
    precisionEvents = {record["event"] for record in precisionRecords}
    assert {
        "munc_lambda.interval",
        "munc_lambda.summary",
        "process_kappa.interval",
        "process_kappa.process_noise_calibration",
        "process_kappa.summary",
    } <= precisionEvents
    assert sum(record["record_type"] == "interval" for record in precisionRecords) == 4
    assert precisionRecords[0]["lambda"] == pytest.approx(0.5)
    firstProcess = next(
        record
        for record in precisionRecords
        if record["event"] == "process_kappa.interval"
    )
    assert firstProcess["kappa"] == pytest.approx(1.5)
    histogramSamples = {
        "lambda": np.empty(0, dtype=np.float64),
        "kappa": np.empty(0, dtype=np.float64),
    }
    consenrich_cli._mergePrecisionReweightingHistogramSamples(
        histogramSamples,
        precisionFrame,
        sampleSize=1,
    )
    np.testing.assert_allclose(histogramSamples["lambda"], [0.5])
    np.testing.assert_allclose(histogramSamples["kappa"], [1.5])
    kappaOnlySamples = {
        "lambda": np.empty(0, dtype=np.float64),
        "kappa": np.empty(0, dtype=np.float64),
    }
    consenrich_cli._mergePrecisionReweightingHistogramSamples(
        kappaOnlySamples,
        precisionFrame,
        sampleSize=1,
        columns=("kappa",),
    )
    assert kappaOnlySamples["lambda"].size == 0
    np.testing.assert_allclose(kappaOnlySamples["kappa"], [1.5])
    gainPlotRows = [
        {
            "replicate_index": 1,
            "sample_name": "sampleA",
            "sample_file": "sampleA",
            "treatment_path": "sampleA.bam",
            "control_path": None,
            "chromosome_count": 2,
            "finite_interval_count": 10,
            "gain_avg": 1.2,
            "gain_std": 0.15,
        },
        {
            "replicate_index": 2,
            "sample_name": "sampleB",
            "sample_file": "sampleB",
            "treatment_path": "sampleB.bam",
            "control_path": None,
            "chromosome_count": 2,
            "finite_interval_count": 8,
            "gain_avg": 0.9,
            "gain_std": 0.1,
        },
    ]
    deleteBlockPlotSamples = {"factor": np.asarray([1.0, 1.5, 2.0])}
    deleteBlockCoverageRows = [
        {
            "stratum": "overall",
            "target": 0.8,
            "n": 10,
            "coverage_before": 0.7,
            "coverage_after": 0.79,
        },
        {
            "stratum": "overall",
            "target": 0.9,
            "n": 8,
            "coverage_before": 0.82,
            "coverage_after": 0.88,
        },
    ]
    np.testing.assert_allclose(
        consenrich_cli._deleteBlockBlockFactorValues(
            np.asarray([1.0, 3.0, 5.0], dtype=np.float64),
            {
                "fold_refits": {"block_len_intervals": 2},
                "target_calibration": {"uncertainty_track_scale": 2.0},
            },
        ),
        np.asarray([8.0, 20.0], dtype=np.float64),
    )

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "matplotlib", None)
        assert (
            consenrich_cli._plotOptimizationPathLog(
                rows,
                str(tmp_path / "missing.png"),
            )
            is False
        )
        assert (
            consenrich_cli._plotPrecisionReweightingHistograms(
                histogramSamples,
                str(tmp_path / "missing_hist.png"),
            )
            is False
        )
        assert (
            consenrich_cli._plotReplicateCalibration(
                gainPlotRows,
                str(tmp_path / "missing_replicate.png"),
            )
            is False
        )
        assert (
            consenrich_cli._plotDeleteBlockCalibration(
                deleteBlockPlotSamples,
                deleteBlockCoverageRows,
                str(tmp_path / "missing_delete_block.png"),
            )
            is False
        )

    remainingGap = consenrich_cli._normalizedRemainingObjectiveGap([10.0, 8.0, 7.0])
    np.testing.assert_allclose(remainingGap, np.asarray([1.0, 1.0 / 3.0, 0.0]))
    np.testing.assert_allclose(
        consenrich_cli._normalizedRemainingObjectiveGap([5.0, 5.0]),
        np.asarray([0.0, 0.0]),
    )

    saveCalls = []
    axisText = {"x": [], "y": [], "title": [], "figure": []}
    gainAxisLimits = [(0.8, 1.4)]
    unitLineCalls = []
    fakeMatplotlib = types.ModuleType("matplotlib")
    fakePyplot = types.ModuleType("matplotlib.pyplot")

    class FakeFigure:
        def suptitle(self, *args, **kwargs):
            axisText["figure"].append(args[0])
            return None

        def savefig(self, path, dpi=None):
            saveCalls.append((path, dpi))

    class FakeAxis:
        transAxes = object()

        def plot(self, *args, **kwargs):
            return None

        def scatter(self, *args, **kwargs):
            return None

        def hist(self, *args, **kwargs):
            return None

        def bar(self, *args, **kwargs):
            return None

        def errorbar(self, *args, **kwargs):
            return None

        def axvline(self, *args, **kwargs):
            return None

        def axvspan(self, *args, **kwargs):
            return None

        def annotate(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            axisText["title"].append(args[0])
            return None

        def set_xlabel(self, *args, **kwargs):
            axisText["x"].append(args[0])
            return None

        def set_ylabel(self, *args, **kwargs):
            axisText["y"].append(args[0])
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def get_ylim(self):
            return gainAxisLimits[0]

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

        def fill_between(self, *args, **kwargs):
            return None

        def get_legend_handles_labels(self, *args, **kwargs):
            return (["handle"], ["label"])

        def axhline(self, *args, **kwargs):
            if kwargs.get("label") == "unit":
                unitLineCalls.append(args[0])
            return None

        def set_yscale(self, *args, **kwargs):
            return None

        def set_xscale(self, *args, **kwargs):
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
    with monkeypatch.context() as mp:
        fakePyplot.subplots = lambda *args, **kwargs: (
            FakeFigure(),
            [FakeAxis(), FakeAxis()],
        )
        mp.setitem(sys.modules, "matplotlib", fakeMatplotlib)
        mp.setitem(sys.modules, "matplotlib.pyplot", fakePyplot)
        assert (
            consenrich_cli._plotPrecisionReweightingHistograms(
                histogramSamples,
                str(tmp_path / "precision_hist.png"),
            )
            is True
        )
    assert saveCalls[-1] == (str(tmp_path / "precision_hist.png"), 400)
    histogramSubplotCalls = []

    def fakeHistogramSubplots(*args, **kwargs):
        histogramSubplotCalls.append(args)
        return FakeFigure(), FakeAxis()

    with monkeypatch.context() as mp:
        fakePyplot.subplots = fakeHistogramSubplots
        mp.setitem(sys.modules, "matplotlib", fakeMatplotlib)
        mp.setitem(sys.modules, "matplotlib.pyplot", fakePyplot)
        assert (
            consenrich_cli._plotPrecisionReweightingHistograms(
                kappaOnlySamples,
                str(tmp_path / "kappa_hist.png"),
            )
            is True
        )
    assert histogramSubplotCalls[-1][1] == 1
    assert saveCalls[-1] == (str(tmp_path / "kappa_hist.png"), 400)
    assert consenrich_cli._precisionReweightingHistogramPath("exp name").name == (
        f"consenrichOutput_exp_name_precisionReweightingHistograms.v{consenrich_cli.__version__}.png"
    )
    assert consenrich_cli._replicateCalibrationPlotPath("exp name").name == (
        f"consenrichOutput_exp_name_replicateCalibration.v{consenrich_cli.__version__}.png"
    )
    assert consenrich_cli._deleteBlockCalibrationPlotPath("exp name").name == (
        f"consenrichOutput_exp_name_deleteBlockCalibration.v{consenrich_cli.__version__}.png"
    )
    with monkeypatch.context() as mp:
        fakePyplot.subplots = lambda *args, **kwargs: (
            FakeFigure(),
            [FakeAxis(), FakeAxis()],
        )
        mp.setitem(sys.modules, "matplotlib", fakeMatplotlib)
        mp.setitem(sys.modules, "matplotlib.pyplot", fakePyplot)
        for limits in ((0.8, 1.4), (0.001, 0.003), (4.0, 8.0)):
            gainAxisLimits[0] = limits
            unitLineCalls.clear()
            assert (
                consenrich_cli._plotReplicateCalibration(
                    gainPlotRows,
                    str(tmp_path / "replicate_calibration.png"),
                )
                is True
            )
            assert unitLineCalls == ([1.0] if limits[0] <= 1.0 <= limits[1] else [])
    assert saveCalls[-1] == (str(tmp_path / "replicate_calibration.png"), 400)
    with monkeypatch.context() as mp:
        fakePyplot.subplots = lambda *args, **kwargs: (
            FakeFigure(),
            [[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]],
        )
        mp.setitem(sys.modules, "matplotlib", fakeMatplotlib)
        mp.setitem(sys.modules, "matplotlib.pyplot", fakePyplot)
        assert (
            consenrich_cli._plotDeleteBlockCalibration(
                deleteBlockPlotSamples,
                deleteBlockCoverageRows,
                str(tmp_path / "delete_block_calibration.png"),
            )
            is True
        )
    assert saveCalls[-1] == (str(tmp_path / "delete_block_calibration.png"), 400)
    assert axisText["x"].count("Block RMS SD multiplier") == 2
    assert axisText["figure"][-1] == "Delete-block perturbation recount"
    assert "Delete-block perturbation recount" in axisText["title"]
    assert "Sampled blocks" in axisText["y"]
    assert "Observed perturbation coverage" in axisText["y"]
    genomeRows = rows + [
        {
            **row,
            "chromosome": "chrOther",
            "record_order": row["record_order"] + len(rows),
            "objective_value": (
                row["objective_value"] + 5.0
                if row["objective_value"] is not None
                else None
            ),
        }
        for row in rows
    ]
    with monkeypatch.context() as mp:
        fakePyplot.subplots = lambda *args, **kwargs: (
            FakeFigure(),
            [FakeAxis(), FakeAxis()],
        )
        mp.setitem(sys.modules, "matplotlib", fakeMatplotlib)
        mp.setitem(sys.modules, "matplotlib.pyplot", fakePyplot)
        assert (
            consenrich_cli._plotGenomeOptimizationPathLog(
                genomeRows,
                str(tmp_path / "genome_optimization.png"),
            )
            is True
        )
    assert saveCalls[-1] == (str(tmp_path / "genome_optimization.png"), 400)
    assert (
        consenrich_cli._genomeOptimizationPathPrefix("exp name")
        == f"consenrichOutput_exp_name_genome_optimizationPath.v{consenrich_cli.__version__}"
    )


def test_state_shrinkage_output_helpers_keep_required_tracks():
    assert consenrich_cli._stateShrinkageOutputTracks(False) == [
        ("stateShrunk", "stateShrunk"),
        ("stateShrunkUncertainty", "stateShrunkUncertainty"),
    ]
    assert consenrich_cli._stateShrinkageOutputTracks(True) == [
        ("stateShrunk", "stateShrunk"),
        ("stateShrunkUncertainty", "stateShrunkUncertainty"),
        ("stateSpikeProp", "stateSpikeProp"),
    ]


def test_run_summary_output_helpers(tmp_path):
    paths = consenrich_cli.DiagnosticLogPaths(
        precision=tmp_path / "precision.jsonl",
        convergence=tmp_path / "convergence.jsonl",
        delete_block_calibration=tmp_path / "delete.jsonl",
    )
    row = consenrich_cli._runSummaryRow(
        chromosome="chr1",
        intervals=12,
        samples=3,
        elapsedSeconds=1.25,
        outputTrackCount=2,
        runDiagnostics={
            "final_nll": 42.0,
            "final_forward_nis": 0.75,
            "process_q_policy": "fixedDiagonal",
            "process_q_diagnostics": {
                "effectiveQTraceMin": 0.01,
                "effectiveQTraceMedian": 0.02,
                "effectiveQTraceMax": 0.03,
            },
            "observation_r_trace": {
                "min": 1.0,
                "median": 2.0,
                "max": 3.0,
            },
            "precision_reweighting_boundary_hits": {
                "observation": {"lower": 1, "upper": 2},
                "process": {"lower": 3, "upper": 4},
            },
            "process_noise_calibration": {
                "processNoiseCalibrationStatus": "ok",
                "processNoiseCalibrationReason": "fit",
            },
        },
        stateRoughness={
            "overall_mean_abs_diff": 0.1,
            "block_mean_abs_diff_median": 0.2,
            "block_mean_abs_diff_q90": 0.3,
        },
        calibrationModel={
            "factor_model": "segShrink",
            "global_factor": 1.5,
            "delete_block_factor_distribution": {
                "count": 3,
                "median": 4.0,
                "unscaled_mad": 3.0,
                "q05": 1.3,
                "q95": 8.5,
                "min": 1.0,
                "max": 9.0,
                "sd_multiplier_median": 2.0,
                "quantile_method": "linear",
            },
            "delete_block_deletion_probability": 0.25,
            "delete_block_deleted_blocks": 4,
            "delete_block_deleted_replicate_block_total": 7,
            "delete_block_deleted_observation_interval_total": 70,
            "replicate_dependence": {
                "rho": 0.35,
                "source": "fixed",
                "total_deff_median": 1.4,
            },
            "rows_valid": 10,
            "rows_fit": 5,
            "target_calibration": {
                "uncertainty_track_scale": 1.1,
                "uncertainty_track_scale_reason": "target",
            },
        },
        diagnosticLogPaths=paths,
        countNoiseFloorSummary={
            "enabled": True,
            "model": "rawCountMassDeltaMethod",
            "countModes": ["conservedFractionalOverlap", "cutsite"],
            "sourceKinds": ["BAM", "BEDGRAPH"],
            "bedgraphSkippedTracks": 1,
            "finite": 10,
            "positive": 9,
            "q05": 0.01,
            "median": 0.05,
            "q95": 0.2,
        },
    )
    genome = consenrich_cli._genomeRunSummaryRow(
        [row],
        elapsedSeconds=2.5,
        diagnosticLogPaths=paths,
    )
    summaryPath = tmp_path / "summary.jsonl"

    consenrich_cli._writeRunSummary([row, genome], summaryPath)

    records = [
        json.loads(line)
        for line in summaryPath.read_text(encoding="utf-8").splitlines()
    ]
    frame = pd.DataFrame(records)
    assert set(frame.columns) <= set(consenrich_cli.RUN_SUMMARY_COLUMNS)
    assert frame["record_type"].tolist() == ["chromosome", "genome"]
    assert frame.loc[0, "lambda_lower_bound_hits"] == 1
    assert frame.loc[0, "kappa_upper_bound_hits"] == 4
    assert frame.loc[0, "process_q_trace_median"] == pytest.approx(0.02)
    assert frame.loc[0, "observation_r_trace_median"] == pytest.approx(2.0)
    assert frame.loc[0, "state_roughness_block_median"] == pytest.approx(0.2)
    assert frame.loc[0, "state_roughness_block_q90"] == pytest.approx(0.3)
    assert bool(frame.loc[0, "countNoiseFloorEnabled"]) is True
    assert frame.loc[0, "countNoiseFloorModel"] == "rawCountMassDeltaMethod"
    assert frame.loc[0, "countNoiseFloorBedgraphSkippedTracks"] == 1
    assert frame.loc[0, "countNoiseFloorQ05"] == pytest.approx(0.01)
    assert frame.loc[0, "countNoiseFloorMedian"] == pytest.approx(0.05)
    assert frame.loc[0, "countNoiseFloorQ95"] == pytest.approx(0.2)
    assert "delete_block_global_factor" not in frame.columns
    assert frame.loc[0, "delete_block_factor_model"] == "segShrink"
    assert "delete_block_variance_multiplier_global" not in frame.columns
    assert frame.loc[
        0, "delete_block_variance_multiplier_min"
    ] == pytest.approx(1.0)
    assert frame.loc[
        0, "delete_block_variance_multiplier_q05"
    ] == pytest.approx(1.3)
    assert frame.loc[0, "delete_block_variance_multiplier_median"] == pytest.approx(
        4.0
    )
    assert frame.loc[
        0, "delete_block_variance_multiplier_mad"
    ] == pytest.approx(3.0)
    assert frame.loc[
        0, "delete_block_variance_multiplier_q95"
    ] == pytest.approx(8.5)
    assert frame.loc[
        0, "delete_block_variance_multiplier_max"
    ] == pytest.approx(9.0)
    assert frame.loc[0, "delete_block_sd_multiplier_median"] == pytest.approx(2.0)
    assert frame.loc[0, "delete_block_track_sd_scale"] == pytest.approx(1.1)
    assert frame.loc[0, "delete_block_rows_valid"] == 10
    assert frame.loc[0, "delete_block_rows_fit"] == 5
    assert frame.loc[0, "delete_block_deletion_probability"] == pytest.approx(0.25)
    assert frame.loc[0, "delete_block_deleted_blocks"] == 4
    assert frame.loc[0, "delete_block_deleted_replicate_block_total"] == 7
    assert frame.loc[0, "delete_block_deleted_observation_interval_total"] == 70
    assert frame.loc[0, "delete_block_replicate_dependence_rho"] == pytest.approx(
        0.35
    )
    assert frame.loc[0, "delete_block_replicate_dependence_rho_source"] == "fixed"
    assert frame.loc[0, "delete_block_information_deff_median"] == pytest.approx(1.4)
    assert frame.loc[0, "delete_block_scale"] == pytest.approx(1.1)
    assert frame.loc[0, "delete_block_scale_reason"] == "target"
    assert frame.loc[1, "chromosome"] == "genome"
    assert frame.loc[1, "intervals"] == 12
    assert frame.loc[1, "output_track_count"] == 2
    assert frame.loc[1, "precision_log"].endswith(".jsonl")
    globalDeleteBlockFields = consenrich_cli._deleteBlockFactorSummaryFields(
        {"factor_model": "global", "global_factor": 1.5}
    )
    assert globalDeleteBlockFields["delete_block_global_factor"] == pytest.approx(1.5)
    assert globalDeleteBlockFields[
        "delete_block_variance_multiplier_global"
    ] == pytest.approx(1.5)


def test_run_summary_output_helpers_accept_state_shrinkage_mixture_metadata(tmp_path):
    metadata = {
        "model": "spikeAndStudentT",
        "slab_family": "studentTNormalScaleMixture",
        "scope": "genome",
        "chunk_count": np.int64(2),
        "interval_count": np.int64(12),
        "finite_count": np.int64(10),
        "effective_block_count": np.float64(6.0),
        "block_size_intervals": np.int64(3),
        "prior_spike_prop": np.float64(0.4),
        "spike_odds_multiplier": np.float64(2.5),
        "effective_prior_spike_prop": np.float64(0.625),
        "prior_scale": np.float64(1.25),
        "prior_variance": np.float64(1.5625),
        "prior_variance_defined": np.bool_(True),
        "slab_count": np.int64(2),
        "slab_weight": np.array([0.25, 0.75], dtype=np.float64),
        "slab_variance": [np.float64(0.5), np.float64(2.0)],
        "slab_multiplier": [np.float64(0.32), np.float64(1.28)],
        "component_weights": [
            np.float64(0.4),
            np.float64(0.15),
            np.float64(0.45),
        ],
        "student_t_df": np.float64(3.0),
        "student_t_scale": np.float64(0.7216878364870323),
        "student_t_quadrature_order": np.int64(24),
        "student_t_quadrature_alpha": np.float64(0.5),
        "estimated_prior_spike_prop": np.bool_(True),
        "estimated_prior_scale": np.bool_(True),
        "estimated_slab_weights": np.bool_(True),
        "estimated_slab_scales": np.bool_(True),
        "iterations": np.int64(8),
        "converged": np.bool_(True),
        "stop_reason": "parameterTolerance",
        "objective_change": np.float64(0.00001),
        "max_relative_change": np.float64(0.000002),
        "log_likelihood": np.float64(-12.5),
    }
    row = {
        "record_type": "genome",
        "chromosome": "genome",
        "intervals": 12,
        "samples": 2,
        "elapsed_seconds": 0.25,
        "output_track_count": 3,
    }
    row.update(consenrich_cli._stateShrinkageSummaryFields(metadata))
    summaryPath = tmp_path / "summary.jsonl"

    consenrich_cli._writeRunSummary([row], summaryPath)

    record = json.loads(summaryPath.read_text(encoding="utf-8"))
    assert set(record) <= set(consenrich_cli.RUN_SUMMARY_COLUMNS)
    assert record["state_shrinkage_model"] == "spikeAndStudentT"
    assert record["state_shrinkage_stop_reason"] == "parameterTolerance"
    assert record["state_shrinkage_objective_change"] == pytest.approx(0.00001)
    assert record["state_shrinkage_max_relative_change"] == pytest.approx(0.000002)
    assert record["state_shrinkage_slab_family"] == "studentTNormalScaleMixture"
    assert record["state_shrinkage_slab_count"] == 2
    assert record["state_shrinkage_slab_weight"] == [0.25, 0.75]
    assert record["state_shrinkage_slab_variance"] == [0.5, 2.0]
    assert record["state_shrinkage_slab_multiplier"] == [0.32, 1.28]
    assert record["state_shrinkage_component_weights"] == [0.4, 0.15, 0.45]
    assert record["state_shrinkage_prior_spike_prop"] == pytest.approx(0.4)
    assert record["state_shrinkage_spike_odds_multiplier"] == pytest.approx(2.5)
    assert record["state_shrinkage_effective_prior_spike_prop"] == pytest.approx(
        0.625
    )
    assert record["state_shrinkage_estimated_prior_spike_prop"] is True
    assert record["state_shrinkage_student_t_df"] == pytest.approx(3.0)
    assert record["state_shrinkage_student_t_scale"] == pytest.approx(
        0.7216878364870323
    )
    assert record["state_shrinkage_student_t_quadrature_order"] == 24
    assert record["state_shrinkage_student_t_quadrature_alpha"] == pytest.approx(0.5)
    assert record["state_shrinkage_prior_variance_defined"] is True
    assert record["state_shrinkage_estimated_slab_weights"] is True
    assert record["state_shrinkage_estimated_slab_scales"] is True
    assert (
        consenrich_cli._diagnosticJsonText(metadata["component_weights"])
        == "[0.4,0.15,0.45]"
    )




def test_cli_console_phase_subphase_contract(tmp_path, monkeypatch):
    class ColorStream(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = ColorStream()
    logPath = tmp_path / "events.jsonl"
    packageLogger = logging.getLogger("consenrich")
    originalHandlers = list(packageLogger.handlers)
    originalLevel = packageLogger.level
    originalPropagate = packageLogger.propagate

    resolvedLogPath = consenrich_cli._configureCliLogging(
        logPath,
        verbose=False,
        verbose2=False,
        verbosity="normal",
        consoleStream=stream,
    )
    try:
        assert resolvedLogPath == logPath
        consenrich_cli._logCliPhase("Config", "samples=%d", 2)
        consenrich_cli._logCliSubphase("Diagnostics: %s", "ready")
        consenrich_cli._logCliProgressMilestone("Verbose detail")
        consenrich_cli._logCliMilestone("Milestone")
        consenrich_cli._logCliMilestone("Finished", blue=True)
        for handler in packageLogger.handlers:
            handler.flush()
    finally:
        consenrich_cli._removeCliHandlers(packageLogger)
        packageLogger.setLevel(originalLevel)
        packageLogger.propagate = originalPropagate
        for handler in originalHandlers:
            if handler not in packageLogger.handlers:
                packageLogger.addHandler(handler)

    consoleText = stream.getvalue()
    assert "\033[1;38;2;0;48;96m" in consoleText
    assert "\033[38;2;140;21;21m" in consoleText
    assert "\033[1;38;2;128;24;96m" in consoleText
    assert "\n\n" in consoleText
    assert "  - " in consoleText
    assert "Verbose detail" not in consoleText

    records = [
        json.loads(line)
        for line in logPath.read_text(encoding="utf-8").splitlines()
    ]
    byMessage = {record["message"]: record for record in records}
    assert byMessage["=== Consenrich | Config === samples=2"]["console_phase"]
    assert byMessage["Diagnostics: ready"]["console_subphase"]
    assert byMessage["Verbose detail"]["console_subphase"]
    assert byMessage["Verbose detail"]["console_verbose"]
    assert byMessage["Milestone"]["console_milestone"]
    assert byMessage["Finished"]["console_milestone"]


def test_config_sparse_sample_source_and_matching_contracts(
    tmp_path, monkeypatch, contract_case
):
    for label, func in (
        (
            "numNearest sparse-bed requirement",
            _case_readConfigNumNearestRequiresExplicitSparseBed,
        ),
        (
            "restrict local variance sparse-bed requirement",
            _case_readConfigRestrictLocalVarianceToSparseBedRequiresAvailableSparseBed,
        ),
        ("CRAM sources rejected", _case_readConfigRejectsCRAMSources),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)
    contract_case(
        "sparse interval loading", _case_loadSparseIntervalIndicesUsesBedSpan, tmp_path
    )
    for label, func in (
        (
            "explicit bedGraph sample source",
            _case_readConfigSamplesSupportExplicitBedGraphFormat,
        ),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)


def test_config_bedgraph_bigwig_io_contracts(tmp_path, contract_case):
    for label, func in (
        ("pyBigWig write", _case_convertBedGraphToBigWigPyBigWigWritesExpectedTrack),
        (
            "pyBigWig out-of-bounds rejection",
            _case_convertBedGraphToBigWigPyBigWigRejectsOutOfBounds,
        ),
        (
            "pyBigWig empty rejection",
            _case_convertBedGraphToBigWigPyBigWigRejectsEmptyBedGraph,
        ),
        ("bedGraph sort", _case_sortBedGraphInPlace),
        (
            "bedGraph genome-order validation",
            _case_bedGraphValidationAcceptsGenomeOrderAndSortsFallback,
        ),
        (
            "bedGraph sizes-order planned chromosome filtering",
            _case_sortBedGraphUsesSizesOrderFilteredToRequestedChromosomes,
        ),
    ):
        contract_case(label, func, tmp_path)








@pytest.mark.parametrize(
    ("values", "message"),
    [({"observationParams.lengthScaleBP": value}, "lengthScaleBP")
     for value in (True, 0, -1, 1.5, "1000")]
    + [({"countingParams.backgroundBlockSizeBP": value}, "backgroundBlockSizeBP")
       for value in (True, False, 1.5, "100")]
    + [
        ({"observationParams.lengthScaleOptions": {"blockCount": 0}}, "blockCount"),
        ({"observationParams.lengthScaleOptions": {"filterType": "invalid"}}, "filterType"),
        ({"observationParams.lengthScaleOptions": {"scaleEndBP": 0}}, "scaleEndBP"),
        ({"observationParams.lengthScaleOptions": 5}, "lengthScaleOptions"),
        ({"observationParams.lengthScaleOptions": {"unknown": 1}}, "Unknown|Unsupported"),
    ],
)
def testLengthScaleAndWindowConfigRejectsInvalidValues(tmp_path, monkeypatch, values, message):
    setupGenomeFiles(tmp_path, monkeypatch)
    config = {
        "experimentName": "lengthScaleValidation",
        "inputParams.bamFiles": ["smallTest.bam"],
        "genomeParams.name": "testGenome",
        **values,
    }
    with pytest.raises(ValueError, match=message):
        readConfig(config)


def testLengthScaleConfigPreservesExplicitValuesAndOptions(tmp_path, monkeypatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    config = {
        "experimentName": "lengthScaleOptions",
        "inputParams.bamFiles": ["smallTest.bam"],
        "genomeParams.name": "testGenome",
        "observationParams.lengthScaleOptions": {
            "blockCount": 12, "blockWidthBP": 200000, "scaleEndBP": 50000,
            "filterType": "mean",
        },
        "observationParams.muncTrendBlockSizeBP": 250,
        "observationParams.muncLocalWindowSizeBP": 500,
        "observationParams.muncTrendBlockLengthScaleMultiplier": 1.5,
        "observationParams.muncLocalWindowLengthScaleMultiplier": 2.5,
    }
    for value, spanBP in ((1, 100), (12345, 20000), (1000001, 200000), (12345, 101)):
        config["observationParams.lengthScaleBP"] = value
        config["countingParams.backgroundBlockSizeBP"] = spanBP
        parsed = readConfig(config)
        observation = parsed["observationArgs"]
        assert parsed["countingArgs"].backgroundBlockSizeBP == spanBP
        assert observation.lengthScaleBP == value
        assert observation.lengthScaleOptions["blockCount"] == 12
        assert observation.lengthScaleOptions["blockWidthBP"] == 200000
        assert observation.lengthScaleOptions["scaleEndBP"] == 50000
        assert observation.lengthScaleOptions["binSizeBP"] == 50
        assert observation.lengthScaleOptions["filterType"] == "mean"
        assert observation.muncTrendBlockSizeBP == 250
        assert observation.muncLocalWindowSizeBP == 500
        assert observation.muncTrendBlockLengthScaleMultiplier == 1.5
        assert observation.muncLocalWindowLengthScaleMultiplier == 2.5
    config["countingParams.backgroundBlockSizeBP"] = 0
    config["countingParams.intervalSizeBP"] = 300
    config["observationParams.lengthScaleOptions"] = {}
    options = readConfig(config)["observationArgs"].lengthScaleOptions
    assert options["binSizeBP"] == 300
    assert options["blockWidthBP"] % 300 == 0
    assert options["scaleEndBP"] == options["blockWidthBP"]
    config["countingParams.intervalSizeBP"] = 1000
    for endBP in (None, 200000):
        config["observationParams.lengthScaleOptions"] = {
            "blockWidthBP": 500000, "scaleEndBP": endBP,
        }
        options = readConfig(config)["observationArgs"].lengthScaleOptions
        expectedEndBP = 500000 if endBP is None else endBP
        assert options["scaleEndBP"] == expectedEndBP
        result = ccounts.estimateLengthScaleCounts(
            np.zeros((2, 1, 500), dtype=np.int32), 1000,
            options=ccounts.LengthScaleOptions(**options),
        )
        assert 125000 < max(result["settings"]["filterWidthsBP"]) <= expectedEndBP


@pytest.mark.parametrize(
    ("settingsText", "expectedBinBP", "conflict"),
    [
        ("observationParams: {lengthScaleOptions: {binSizeBP: 25}}", 50, True),
        ("observationParams.lengthScaleOptions: {binSizeBP: 100}", 50, True),
        ("observationParams.lengthScaleOptions.binSizeBP: 10", 50, True),
        ("observationParams.lengthScaleOptions.binSizeBP: 50", 50, False),
        ("observationParams: {lengthScaleOptions: {binSizeBP: true}}", 50, True),
        ("observationParams.lengthScaleOptions: {binSizeBP: 0}", 50, True),
        ("observationParams.lengthScaleOptions.binSizeBP: invalid", 50, True),
        ("countingParams: {intervalSizeBP: 25}\n"
         "observationParams: {lengthScaleOptions: {binSizeBP: 25}}", 25, False),
        ("countingParams.intervalSizeBP: 50\n"
         "observationParams.lengthScaleOptions: {binSizeBP: 25}", 50, True),
        ("countingParams.intervalSizeBP: 100\n"
         "observationParams.lengthScaleOptions.binSizeBP: 25", 100, True),
        ("observationParams: {lengthScaleOptions: {binSizeBP: null}}", 50, False),
        ("countingParams.intervalSizeBP: 25\n"
         "observationParams.lengthScaleOptions: {binSizeBP: null}", 25, False),
    ],
)
def testLengthScaleBinSizeUsesCountingGrid(
    tmp_path, monkeypatch, caplog, settingsText, expectedBinBP, conflict,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    configPath = writeConfigFile(
        tmp_path, "binSize.yaml",
        "inputParams.bamFiles: [smallTest.bam]\n"
        "genomeParams.name: testGenome\n" + settingsText,
    )
    with caplog.at_level(logging.WARNING):
        parsed = readConfig(configPath)
    assert parsed["countingArgs"].intervalSizeBP == expectedBinBP
    assert parsed["observationArgs"].lengthScaleOptions["binSizeBP"] == expectedBinBP
    warnings = [
        record.getMessage() for record in caplog.records
        if "lengthScaleOptions.binSizeBP" in record.getMessage()
    ]
    assert len(warnings) == int(conflict)
    if conflict:
        assert "countingParams.intervalSizeBP" in warnings[0]
        assert str(expectedBinBP) in warnings[0]


@pytest.mark.parametrize(
    "flags",
    [
        ["--match-bedGraph", "signal.bedGraph"],
        ["--match-num-bootstrap", "1"],
        ["--match-no-local-bootstrap-radius"],
    ],
)
def test_cli_rejects_peak_options(flags, capsys):
    with pytest.raises(SystemExit) as error:
        consenrich_cli._buildArgParser().parse_args(flags)
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "unrecognized arguments: " + " ".join(flags) in output
    assert "or see CLI help `consenrich -h`" in output


@pytest.mark.parametrize(
    ("flags", "exitCode"),
    [
        ([], 1),
        (["--config", "missing.yaml"], 1),
        (["--config"], 2),
        (["--verbosity", "invalid"], 2),
        (["--set", "experimentName=example"], 2),
    ],
)
def test_cli_failure_guidance(flags, exitCode, monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(consenrich_cli.sys, "argv", ["consenrich", *flags])
    packageLogger = logging.getLogger("consenrich")
    savedHandlers = packageLogger.handlers
    savedLevel = packageLogger.level
    savedPropagate = packageLogger.propagate
    packageLogger.handlers = []
    try:
        with pytest.raises(SystemExit) as error:
            consenrich_cli.main()
        assert error.value.code == exitCode
        output = capsys.readouterr().err
        assert "  * See documentation: https://nolan-h-hamilton.github.io/Consenrich/" in output
        assert "or see CLI help `consenrich -h`" in output
    finally:
        consenrich_cli._removeCliHandlers(packageLogger)
        packageLogger.handlers = savedHandlers
        packageLogger.setLevel(savedLevel)
        packageLogger.propagate = savedPropagate


@pytest.mark.parametrize("flags", [[], ["--verbose2"]])
def test_cli_config_overrides(tmp_path, monkeypatch, flags):
    setupGenomeFiles(tmp_path, monkeypatch)
    configPath = writeConfigFile(tmp_path, "overrides.yaml", """
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        samParams: {samThreads: 1}
        loggingParams: {verbosity: quiet, progress: on}
    """)
    sourceText = configPath.read_text()
    captured = []
    monkeypatch.setattr(consenrich_cli, "_configureCliLogging", lambda *a, **kw: None)
    monkeypatch.setattr(
        consenrich_cli, "executeFits", lambda *args: captured.append(args)
    )
    argv = ["consenrich", "--config", str(configPath)]
    for assignment in (
        "samParams.samThreads=2",
        "samParams.samThreads=3",
        "observationParams.lengthScaleOptions.binSizeBP=50",
        "countingParams.intervalSizeBP=25",
        "fitParams.backgroundLengthScaleMultiplier=3",
        "loggingParams.verbosity=verbose",
        "loggingParams.progress=off",
    ):
        argv.extend(("--set", assignment))
    monkeypatch.setattr(consenrich_cli.sys, "argv", argv + flags)
    consenrich_cli.main()
    config, args, runConfigured, configureLogging = captured[0]
    assert runConfigured is consenrich_cli._runConfigured
    assert configureLogging is consenrich_cli._configureCliLogging
    assert config["samArgs"].samThreads == 3
    assert config["countingArgs"].intervalSizeBP == 25
    assert config["observationArgs"].lengthScaleOptions["binSizeBP"] == 25
    assert config["fitArgs"].ECM_backgroundLengthScaleMultiplier == 3
    assert args.verbosity == ("debug" if flags else "verbose")
    assert args.progress == "off"
    assert configPath.read_text() == sourceText


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (7, 7),
        (np.int64(9), 9),
        (np.uint64(2**64 - 1), 2**64 - 1),
        (-1, None),
        (True, None),
        (False, None),
        (np.bool_(True), None),
        (2.9, None),
        (-0.5, None),
        (2.0, None),
        (np.float64(2.0), None),
        ("2", None),
    ],
)
def test_nonnegativeIntegerValidation(value, expected):
    configName = "outputParams.maxNonTrackFileBytes"
    if expected is None:
        with pytest.raises(ValueError, match=configName):
            normalization.validateNonnegativeInt(value, configName)
    else:
        result = normalization.validateNonnegativeInt(value, configName)
        assert result == expected
        assert type(result) is int


def test_config_override_errors():
    for assignment in ("samParams.samThread=2", "samParams.samThreads=["):
        with pytest.raises(ValueError):
            consenrich_config.applyConfigOverrides({}, [assignment])
