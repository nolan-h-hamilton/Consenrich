import textwrap
import logging
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
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


def writeGenomeCovariateCache(
    tmpPath,
    *,
    features=("gc", "repeat_frac"),
    chrom="chrTest",
    binSize=50,
):
    cacheLabel = "_".join(str(feature) for feature in features)
    cacheDir = tmpPath / f"genome_covariates_{cacheLabel}"
    arraysDir = cacheDir / "arrays"
    arraysDir.mkdir(parents=True)
    arr = np.zeros((4, len(features)), dtype=np.float32)
    for featureIndex in range(len(features)):
        arr[:, featureIndex] = np.linspace(0.1, 0.4, arr.shape[0])
    np.save(arraysDir / f"{chrom}.npy", arr, allow_pickle=False)
    manifest = {
        "schema": "consenrich-genome-covariates-v1",
        "bin_size_bp": int(binSize),
        "features": list(features),
        "chromosomes": [
            {
                "name": chrom,
                "length": int(arr.shape[0] * binSize),
                "bins": int(arr.shape[0]),
                "array": f"arrays/{chrom}.npy",
            }
        ],
    }
    (cacheDir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cacheDir


def _caseRuntimeBackgroundSpanUsesLengthScaleMultiplier():
    coarseMinSpan, coarseMaxSpan = consenrich_cli._dependenceSpanBoundsFromContextBP(50)
    fineMinSpan, fineMaxSpan = consenrich_cli._dependenceSpanBoundsFromContextBP(25)
    assert fineMinSpan >= coarseMinSpan
    assert fineMaxSpan >= coarseMaxSpan
    assert abs(2 * coarseMaxSpan * 50 - 2 * fineMaxSpan * 25) <= 50
    coarseLen = consenrich_cli._resolveRuntimeBackgroundBlockLen(
        dependenceSpanIntervals=coarseMaxSpan,
        backgroundBlockSizeBP=-1,
        intervalSizeBP=50,
        lengthScaleMultiplier=16.0,
    )
    fineLen = consenrich_cli._resolveRuntimeBackgroundBlockLen(
        dependenceSpanIntervals=fineMaxSpan,
        backgroundBlockSizeBP=-1,
        intervalSizeBP=25,
        lengthScaleMultiplier=16.0,
    )
    assert abs(coarseLen * 50 - fineLen * 25) <= 50

    blockLen = consenrich_cli._resolveRuntimeBackgroundBlockLen(
        dependenceSpanIntervals=5,
        backgroundBlockSizeBP=550,
        intervalSizeBP=50,
        lengthScaleMultiplier=8.0,
    )
    assert blockLen == 89
    assert (
        consenrich_cli._resolveRuntimeBackgroundBlockLen(
            dependenceSpanIntervals=None,
            backgroundBlockSizeBP=250,
            intervalSizeBP=50,
            lengthScaleMultiplier=8.0,
        )
        == 41
    )
    assert (
        consenrich_cli._resolveRuntimeBackgroundBlockLen(
            dependenceSpanIntervals=5,
            backgroundBlockSizeBP=550,
            intervalSizeBP=50,
            lengthScaleMultiplier=4.0,
        )
        == 45
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
    processParams.processNoiseCalibration: tunc
    processParams.tuncMinScale: 0.5
    processParams.tuncMaxScale: 2.0
    processParams.processNoiseWarmupECMIters: 11
    processParams.processNoiseWarmupOuterPasses: 3
    """
    configPath = writeConfigFile(tmp_path, "config_summary.yaml", configYaml)
    parsed = readConfig(str(configPath))

    caplog.set_level(logging.INFO, logger=consenrich_cli.logger.name)
    consenrich_cli._logInitialConfigurationSummary(parsed)

    assert "event=config.initial" in caplog.text
    assert "treatment_inputs=3" in caplog.text
    assert "process_noise_calibration=tunc" in caplog.text
    assert 'tunc_scale_bounds="[0.5, 2]"' in caplog.text
    assert 'process_kappa_bounds="[auto, 10]"' in caplog.text
    assert "3 outer passes x 11 ECM iters" in caplog.text
    assert "inputSource(" not in caplog.text
    assert "'countingArgs':" not in caplog.text


def _caseReplicateGainSummaryWritesPooledAverageAndStd(tmp_path):
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
    assert rows[0]["control_path"] == "/tmp/controlA.bam"
    assert rows[0]["chromosome_count"] == 2
    assert rows[0]["finite_interval_count"] == 10
    assert rows[0]["gain_avg"] == pytest.approx(expectedAvg)
    assert rows[0]["gain_std"] == pytest.approx(expectedStd)

    path = tmp_path / "gains.tsv"
    assert consenrich_cli._writeReplicateGainSummary(rows, str(path)) is True
    text = path.read_text(encoding="utf-8")
    assert "gain_avg\tgain_std" in text
    assert "gain_median" not in text
    assert "gain_iqr" not in text


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
    assert consenrich_io._normalizeScaleFactorList(
        [0.25],
        3,
        "countingParams.scaleFactorsControl",
    ) == [0.25, 0.25, 0.25]
    assert consenrich_io._normalizeScaleFactorList(
        [1.0, 2.0, 3.0],
        3,
        "countingParams.scaleFactors",
    ) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="must contain 1 value or 3 values"):
        consenrich_io._normalizeScaleFactorList(
            [1.0, 2.0],
            3,
            "countingParams.scaleFactorsControl",
        )


def _case_readConfigGenericCountTransform(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: genericTransform
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    countingParams.transformMethod: asinh_sqrt
    countingParams.transformInputOffset: 0.25
    countingParams.transformInputScale: 2.0
    countingParams.transformOutputScale: 2.0
    countingParams.transformOutputOffset: -0.1
    countingParams.transformShape: 0.75
    """
    configPath = writeConfigFile(tmp_path, "config_generic_transform.yaml", configYaml)
    parsed = readConfig(str(configPath))
    countingArgs = parsed["countingArgs"]

    assert countingArgs.transformMethod == "asinhSqrt"
    assert countingArgs.transformInputOffset == pytest.approx(0.25)
    assert countingArgs.transformInputScale == pytest.approx(2.0)
    assert countingArgs.transformOutputScale == pytest.approx(2.0)
    assert countingArgs.transformOutputOffset == pytest.approx(-0.1)
    assert countingArgs.transformShape == pytest.approx(0.75)

    legacyYaml = """
    experimentName: legacyLogTransform
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    countingParams.logOffset: 4.0
    countingParams.logMult: 1.4426950408889634
    """
    legacyPath = writeConfigFile(tmp_path, "config_legacy_transform.yaml", legacyYaml)
    legacyParsed = readConfig(str(legacyPath))
    legacyCountingArgs = legacyParsed["countingArgs"]

    assert legacyCountingArgs.transformMethod == "log"
    assert legacyCountingArgs.transformInputOffset == pytest.approx(4.0)
    assert legacyCountingArgs.transformOutputScale == pytest.approx(1.4426950408889634)

    anscombeYaml = """
    experimentName: anscombeTransform
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    countingParams.transformMethod: anscombe
    """
    anscombePath = writeConfigFile(
        tmp_path, "config_anscombe_transform.yaml", anscombeYaml
    )
    anscombeParsed = readConfig(str(anscombePath))
    anscombeCountingArgs = anscombeParsed["countingArgs"]

    assert anscombeCountingArgs.transformMethod == "anscombe"
    assert anscombeCountingArgs.transformInputOffset == pytest.approx(0.375)
    assert anscombeCountingArgs.transformInputScale == pytest.approx(1.0)
    assert anscombeCountingArgs.transformOutputScale == pytest.approx(2.0)
    assert anscombeCountingArgs.transformOutputOffset == pytest.approx(0.0)

    invalidYaml = """
    experimentName: invalidTransform
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    countingParams.transformMethod: banana
    """
    invalidPath = writeConfigFile(
        tmp_path, "config_invalid_transform.yaml", invalidYaml
    )
    with pytest.raises(ValueError, match="transformMethod"):
        readConfig(str(invalidPath))


def _case_countModelVarianceFloorFollowsPluginPoissonDeltaMethod(
    tmp_path,
    monkeypatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    counts = np.asarray([0.5, 2.0, 8.0], dtype=np.float64)
    scaleFactor = 0.25
    inputOffset = 0.5
    inputScale = 2.0
    outputScale = 1.5
    shape = 0.75
    baseArgs = consenrich_core.countingParams(
        intervalSizeBP=25,
        backgroundBlockSizeBP=1000,
        scaleFactors=None,
        scaleFactorsControl=None,
        normMethod="CPM",
        fragmentsGroupNorm="NONE",
        fixControl=False,
        logOffset=1.0,
        logMult=1.0,
        transformInputOffset=inputOffset,
        transformInputScale=inputScale,
        transformOutputScale=outputScale,
        transformOutputOffset=0.0,
        transformShape=shape,
        subtractGlobalMedian=True,
    )

    momentCounts = counts + 0.5 * scaleFactor
    normalizedVariance = (scaleFactor * counts) + (
        0.5 * scaleFactor * scaleFactor
    )
    z = momentCounts + inputOffset
    u = z / inputScale
    derivativeByMethod = {
        "log": (outputScale * outputScale) / (z * z),
        "sqrt": (outputScale * outputScale) / (4.0 * inputScale * z),
        "anscombe": (outputScale * outputScale) / (4.0 * inputScale * z),
        "asinh": (outputScale * outputScale)
        / ((inputScale * inputScale) * (1.0 + (u * u))),
        "asinhSqrt": (outputScale * outputScale)
        / (
            4.0
            * inputScale
            * inputScale
            * z
            * (1.0 + z / (inputScale * inputScale))
        ),
        "generalizedLog": (outputScale * outputScale)
        / ((inputScale * inputScale) * ((u * u) + (shape * shape))),
        "identity": np.full_like(
            counts,
            (outputScale * outputScale) / (inputScale * inputScale),
        ),
    }

    for method, derivativeSquared in derivativeByMethod.items():
        countingArgs = baseArgs._replace(transformMethod=method)
        floor = consenrich_cli._countModelVarianceFloorForScaledCounts(
            counts,
            scaleFactor,
            countingArgs,
        )
        expected = derivativeSquared * normalizedVariance
        np.testing.assert_allclose(floor, expected, rtol=1.0e-6, atol=1.0e-6)

    logArgs = baseArgs._replace(
        transformMethod="log",
        transformInputOffset=1.0,
        transformInputScale=1.0,
        transformOutputScale=1.0,
    )
    treatmentFloor = consenrich_cli._countModelVarianceFloorForScaledCounts(
        np.asarray([4.0, 9.0], dtype=np.float64),
        0.5,
        logArgs,
    )
    controlFloor = consenrich_cli._countModelVarianceFloorForScaledCounts(
        np.asarray([1.0, 16.0], dtype=np.float64),
        0.25,
        logArgs,
    )
    combined = consenrich_cli._combineCountModelVarianceFloors(
        treatmentFloor,
        controlFloor,
    )
    np.testing.assert_allclose(combined, treatmentFloor + controlFloor)

    bamSource = consenrich_core.inputSource("sample.bam", sourceKind="BAM")
    bedGraphSource = consenrich_core.inputSource(
        "sample.bedGraph",
        sourceKind="BEDGRAPH",
    )
    matrixFloor = consenrich_cli._countModelFloorMatrixForScaledCounts(
        np.vstack([counts, counts]),
        [scaleFactor, scaleFactor],
        [bamSource, bedGraphSource],
        baseArgs._replace(transformMethod="identity"),
    )
    np.testing.assert_allclose(
        matrixFloor[0, :],
        derivativeByMethod["identity"] * normalizedVariance,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert np.all(np.isnan(matrixFloor[1, :]))


def test_count_model_variance_floor_transform_delta_method(tmp_path, monkeypatch):
    _case_countModelVarianceFloorFollowsPluginPoissonDeltaMethod(
        tmp_path,
        monkeypatch,
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


def _case_readConfigDottedAndNestedEquivalent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    dottedYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam, smallTest2.bam]
    genomeParams.name: testGenome
    genomeParams.excludeChroms: [chrM]
    countingParams.intervalSizeBP: 50
    samParams.defaultCountMode: ffp-center
    outputParams.plotOptimizationPath: false
    outputParams.cutoffReport: true
    outputParams.writeRunSummary: false
    outputParams.precisionDiagnosticDetail: sampled
    outputParams.maxPrecisionDiagnosticRowsPerChromosome: 7
    outputParams.maxNonTrackFileBytes: 1024
    matchingParams.uncertaintyScoreMode: lower_confidence
    matchingParams.uncertaintyScoreZ: 1.25
    matchingParams.metadataDetail: full
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
    samParams:
      defaultCountMode: ffp-center
    outputParams:
      plotOptimizationPath: false
      cutoffReport: true
      writeRunSummary: false
      precisionDiagnosticDetail: sampled
      maxPrecisionDiagnosticRowsPerChromosome: 7
      maxNonTrackFileBytes: 1024
    matchingParams:
      uncertaintyScoreMode: lower-confidence
      uncertaintyScoreZ: 1.25
      metadataDetail: full
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
    assert outputDotted.cutoffReport is True
    assert outputNested.cutoffReport is True
    assert outputDotted.writeRunSummary is False
    assert outputNested.writeRunSummary is False
    assert outputDotted.precisionDiagnosticDetail == "sampled"
    assert outputNested.precisionDiagnosticDetail == "sampled"
    assert outputDotted.maxPrecisionDiagnosticRowsPerChromosome == 7
    assert outputNested.maxPrecisionDiagnosticRowsPerChromosome == 7
    assert outputDotted.maxNonTrackFileBytes == 1024
    assert outputNested.maxNonTrackFileBytes == 1024
    assert outputDotted == outputNested

    samDotted = configDotted["samArgs"]
    samNested = configNested["samArgs"]
    matchingDotted = configDotted["matchingArgs"]
    matchingNested = configNested["matchingArgs"]
    assert matchingDotted.metadataDetail == "full"
    assert matchingNested.metadataDetail == "full"

    assert type(samDotted) is type(samNested)
    assert type(matchingDotted) is type(matchingNested)

    assert samDotted.samThreads == samNested.samThreads
    assert samDotted.defaultCountMode == "ffp-center"
    assert samNested.defaultCountMode == "ffp-center"
    assert configDotted["scArgs"].defaultCountMode == "coverage"
    assert configNested["scArgs"].defaultCountMode == "coverage"
    assert matchingDotted.enabled == matchingNested.enabled
    assert matchingDotted.thresholdZ == matchingNested.thresholdZ
    assert matchingDotted.nestedRoccoIters == matchingNested.nestedRoccoIters
    assert (
        matchingDotted.nestedRoccoBudgetScale == matchingNested.nestedRoccoBudgetScale
    )
    assert matchingDotted.uncertaintyScoreMode == "lower_confidence"
    assert matchingNested.uncertaintyScoreMode == "lower_confidence"
    assert matchingDotted.uncertaintyScoreZ == pytest.approx(1.25)
    assert matchingNested.uncertaintyScoreZ == pytest.approx(1.25)


def _case_readConfigOutputDiagnosticTracks(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configPath = writeConfigFile(
        tmp_path,
        "config_output_diagnostic_tracks.yaml",
        """
        experimentName: testExperiment
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        outputParams.diagnosticTracks: [slope, pre-kappa-q-level, effective-q-trend, munc-trace, trend]
        """,
    )

    parsed = readConfig(str(configPath))

    assert parsed["outputArgs"].diagnosticTracks == (
        "slope",
        "preKappaQLevel",
        "effectiveQTrend",
        "muncTrace",
    )
    assert consenrich_config._normalizeOutputDiagnosticTracks("all") == tuple(
        constants.OUTPUT_DIAGNOSTIC_TRACK_NAMES
    )
    with pytest.raises(ValueError, match="Unsupported output diagnostic track"):
        consenrich_config._normalizeOutputDiagnosticTracks(["notATrack"])


def _case_readConfigBroadcastsSharedControlScaleFactor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

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


def _case_readConfigProcessNoiseOptions(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    processParams:
      stateModel: level
      processNoiseCalibration: tunc
      tuncLocalWindowMultiplier: 3.0
      tuncDependenceMultiplier: 1.5
      tuncMinScale: 0.5
      tuncMaxScale: 3.0
      tuncMinWindowWeight: 2.0
      tuncPriorRidge: 0.002
      tuncLevelBufferZ: 1.25
      tuncUseReliabilityWeightedWindows: false
      qSeedPriorLevel: 3.0e-8
      processNoiseWarmupECMIters: 7
      processNoiseWarmupOuterPasses: 5
      precisionMultiplierMin: 0.5
      precisionMultiplierMax: 2.0
    observationParams:
      precisionMultiplierMin: 0.1
      precisionMultiplierMax: 8.0
      useCountNoiseFloor: false
      muncEBPrior:
        gUncertaintyMode: disabled
    """

    configPath = writeConfigFile(tmp_path, "config_process_noise.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    processArgs = configParsed["processArgs"]

    assert processArgs.stateModel == constants.STATE_MODEL_LEVEL
    assert (
        processArgs.processNoiseCalibration == constants.PROCESS_NOISE_CALIBRATION_TUNC
    )
    assert not hasattr(processArgs, "tuncPriorDf")
    assert processArgs.tuncLocalWindowMultiplier == pytest.approx(3.0)
    assert processArgs.tuncDependenceMultiplier == pytest.approx(1.5)
    assert processArgs.tuncMinScale == pytest.approx(0.5)
    assert processArgs.tuncMaxScale == pytest.approx(3.0)
    assert processArgs.tuncMinWindowWeight == pytest.approx(2.0)
    assert processArgs.tuncPriorRidge == pytest.approx(0.002)
    assert processArgs.tuncLevelBufferZ == pytest.approx(1.25)
    assert processArgs.tuncUseReliabilityWeightedWindows is False
    assert processArgs.qSeedPriorLevel == pytest.approx(3.0e-8)
    assert processArgs.processNoiseWarmupECMIters == 7
    assert processArgs.processNoiseWarmupOuterPasses == 5
    assert processArgs.precisionMultiplierMin == pytest.approx(0.5)
    assert processArgs.precisionMultiplierMax == pytest.approx(2.0)
    assert configParsed["observationArgs"].precisionMultiplierMin == pytest.approx(0.1)
    assert configParsed["observationArgs"].precisionMultiplierMax == pytest.approx(8.0)
    assert configParsed["observationArgs"].useReplicateTrends is False
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


def _case_readConfigMuncCovariates(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    cacheDir = writeGenomeCovariateCache(tmp_path)

    configYaml = f"""
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.genomeCovariateCacheDir: {cacheDir}
    countingParams.intervalSizeBP: 50
    observationParams:
      muncCovariates:
        enabled: true
        mode: per-replicate-additive
        features: [gc_dev, repeat_frac]
    """
    configPath = writeConfigFile(tmp_path, "config_munc_covariates.yaml", configYaml)

    parsed = readConfig(str(configPath))

    assert parsed["genomeArgs"].genomeCovariateCacheDir == str(cacheDir)
    assert parsed["observationArgs"].muncCovariatesEnabled is True
    assert (
        parsed["observationArgs"].muncCovariatesMode
        == constants.MUNC_COVARIATES_MODE_PER_REPLICATE_ADDITIVE
    )
    assert parsed["observationArgs"].muncCovariatesFeatures == ("gc", "repeat_frac")
    assert parsed["processArgs"].tuncProcessCovariatesEnabled is False
    assert parsed["processArgs"].tuncProcessCovariatesFeatures == ()


def _case_readConfigMuncCovariatesAcceptsManifestFeatureNames(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    cacheDir = writeGenomeCovariateCache(
        tmp_path,
        features=("gc", "custom_signal_z", "repeat_frac"),
    )

    configYaml = f"""
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.genomeCovariateCacheDir: {cacheDir}
    countingParams.intervalSizeBP: 50
    observationParams:
      muncCovariates:
        enabled: true
        features: [gc_dev, custom_signal_z]
    """
    configPath = writeConfigFile(
        tmp_path,
        "config_munc_covariates_custom_feature.yaml",
        configYaml,
    )

    parsed = readConfig(str(configPath))

    assert parsed["observationArgs"].muncCovariatesFeatures == (
        "gc",
        "custom_signal_z",
    )


def _case_readConfigMuncCovariatesRequireCache(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncCovariates.enabled: true
    """
    configPath = writeConfigFile(
        tmp_path,
        "config_munc_covariates_missing_cache.yaml",
        configYaml,
    )

    with pytest.raises(ValueError, match="genomeCovariateCacheDir"):
        readConfig(str(configPath))


def _case_readConfigMuncCovariatesRejectMissingFeature(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    cacheDir = writeGenomeCovariateCache(tmp_path, features=("gc",))

    configYaml = f"""
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    genomeParams.genomeCovariateCacheDir: {cacheDir}
    countingParams.intervalSizeBP: 50
    observationParams.muncCovariates.enabled: true
    observationParams.muncCovariates.features: repeat_frac
    """
    configPath = writeConfigFile(
        tmp_path,
        "config_munc_covariates_missing_feature.yaml",
        configYaml,
    )

    with pytest.raises(ValueError, match="missing requested MUNC features"):
        readConfig(str(configPath))


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
        == constants.PROCESS_NOISE_CALIBRATION_TUNC
    )
    assert parsed["processArgs"].tuncMinScale == pytest.approx(
        constants.PROCESS_DEFAULT_TUNC_MIN_SCALE
    )
    assert parsed["processArgs"].tuncMaxScale == pytest.approx(
        constants.PROCESS_DEFAULT_TUNC_MAX_SCALE
    )
    assert (
        parsed["processArgs"].tuncUseReliabilityWeightedWindows
        is constants.PROCESS_DEFAULT_TUNC_USE_RELIABILITY_WEIGHTED_WINDOWS
    )
    assert (
        parsed["observationArgs"].useReplicateTrends
        is constants.OBSERVATION_DEFAULT_USE_REPLICATE_TRENDS
    )
    assert (
        parsed["observationArgs"].useCountNoiseFloor
        is constants.OBSERVATION_DEFAULT_USE_COUNT_NOISE_FLOOR
    )
    assert (
        parsed["observationArgs"].muncEBPriorGUncertaintyMode
        == constants.OBSERVATION_DEFAULT_MUNC_EB_PRIOR_G_UNCERTAINTY_MODE
    )


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
        "uncertaintyCalibrationParams.deleteBlockVarianceMode",
        "uncertaintyCalibrationParams.deleteBlockUseLambdaInInformation",
        "uncertaintyCalibrationParams.deleteBlockTargetSignal",
        "uncertaintyCalibrationParams.deleteBlockFactorModel",
        "uncertaintyCalibrationParams.deleteBlockFactorSegmentCount",
        "uncertaintyCalibrationParams.deleteBlockFactorBootstrapReplicates",
        "uncertaintyCalibrationParams.deleteBlockMinInformationFraction",
        "uncertaintyCalibrationParams.deleteBlockMaxInformationFraction",
        "uncertaintyCalibrationParams.deleteBlockMinDeltaVariance",
        "uncertaintyCalibrationParams.deleteBlockFallbackMinValidFraction",
        "uncertaintyCalibrationParams.deleteBlockScoreWeightMode",
        "uncertaintyCalibrationParams.calibrationOuterIters",
        "uncertaintyCalibrationParams.deleteBlockApplyTargetCalibration",
    ):
        assert key in defaults
    assert constants.UNCERTAINTY_CALIBRATION_MODES == (
        constants.UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE,
    )
    assert constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_MODELS == (
        constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_GLOBAL,
        constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_SEG_SHRINK,
    )
    assert "predictive_holdout" not in constants.UNCERTAINTY_CALIBRATION_MODES
    assert not any(key.startswith("uncertaintyCalibration.") for key in defaults)
    assert "observationParams.muncEBPrior.mode" not in defaults
    assert "observationParams.muncAR1VarianceFunctional" not in defaults
    assert "observationParams.muncGUncertaintyMode" not in defaults


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
    configPath = writeConfigFile(
        tmp_path, "config_centralized_defaults.yaml", configYaml
    )
    parsed = readConfig(str(configPath))
    profile = constants.DEFAULT_CONFIGURATION_VALUES[
        constants.GENERIC_DEFAULT_CONFIGURATION
    ]

    assert (
        consenrich_config.DEFAULT_CONFIGURATION_VALUES
        is constants.DEFAULT_CONFIGURATION_VALUES
    )
    assert (
        consenrich_config.GENERIC_DEFAULT_CONFIGURATION
        is constants.GENERIC_DEFAULT_CONFIGURATION
    )
    assert (
        consenrich_config.SUPPORTED_DEFAULT_CONFIGURATIONS
        is constants.SUPPORTED_DEFAULT_CONFIGURATIONS
    )
    assert (
        consenrich_config.DEFAULT_CONFIGURATION_KEYS
        is constants.DEFAULT_CONFIGURATION_KEYS
    )
    assert not hasattr(constants, "PROCESS_DEFAULT_TUNC_PRIOR_DF")

    assert parsed["defaultConfiguration"] == constants.GENERIC_DEFAULT_CONFIGURATION
    assert "processParams.tuncPriorDf" not in profile
    assert parsed["processArgs"].stateModel == profile["processParams.stateModel"]
    assert (
        parsed["processArgs"].processNoiseCalibration
        == profile["processParams.processNoiseCalibration"]
    )
    assert parsed["processArgs"].tuncMinScale == profile["processParams.tuncMinScale"]
    assert parsed["processArgs"].tuncMaxScale == profile["processParams.tuncMaxScale"]
    assert (
        parsed["processArgs"].tuncUseReliabilityWeightedWindows
        == profile["processParams.tuncUseReliabilityWeightedWindows"]
    )
    assert (
        parsed["processArgs"].processNoiseWarmupECMIters
        == profile["processParams.processNoiseWarmupECMIters"]
    )
    assert (
        parsed["processArgs"].processNoiseWarmupOuterPasses
        == profile["processParams.processNoiseWarmupOuterPasses"]
    )
    assert parsed["fitArgs"].ECM_outerIters == profile["fitParams.ECM_outerIters"]
    assert (
        parsed["fitArgs"].ECM_backgroundLengthScaleMultiplier
        == profile["fitParams.ECM_backgroundLengthScaleMultiplier"]
    )
    assert (
        parsed["fitArgs"].useNonnegativeBackground
        == profile["fitParams.useNonnegativeBackground"]
    )
    assert (
        parsed["fitArgs"].backgroundNegativePenaltyMultiplier
        == profile["fitParams.backgroundNegativePenaltyMultiplier"]
    )
    assert (
        parsed["observationArgs"].muncVarianceModel
        == profile["observationParams.muncVarianceModel"]
    )
    assert (
        parsed["observationArgs"].muncTrendBlockSizeBP
        == profile["observationParams.muncTrendBlockSizeBP"]
    )
    assert (
        parsed["observationArgs"].muncLocalWindowSizeBP
        == profile["observationParams.muncLocalWindowSizeBP"]
    )
    assert (
        parsed["observationArgs"].muncTrendBlockDependenceMultiplier
        == profile["observationParams.muncTrendBlockDependenceMultiplier"]
    )
    assert (
        parsed["observationArgs"].muncLocalWindowDependenceMultiplier
        == profile["observationParams.muncLocalWindowDependenceMultiplier"]
    )
    assert (
        parsed["observationArgs"].restrictLocalVarianceToSparseBed
        == profile["observationParams.restrictLocalVarianceToSparseBed"]
    )
    ebPriorFieldMap = {
        "muncEBPriorTileSizeBP": "observationParams.muncEBPrior.tileSizeBP",
        "muncEBPriorTileCount": "observationParams.muncEBPrior.tileCount",
        "muncEBPriorStrata": "observationParams.muncEBPrior.strata",
        "muncEBPriorMinTilesPerStratum": (
            "observationParams.muncEBPrior.minTilesPerStratum"
        ),
        "muncEBPriorSeed": "observationParams.muncEBPrior.seed",
        "muncEBPriorSupportMinQ": "observationParams.muncEBPrior.supportMinQ",
        "muncEBPriorSupportMaxQ": "observationParams.muncEBPrior.supportMaxQ",
        "muncEBPriorMaxExtrapolatedFraction": (
            "observationParams.muncEBPrior.maxExtrapolatedFraction"
        ),
        "muncEBPriorWarmupECMIters": (
            "observationParams.muncEBPrior.warmupECMIters"
        ),
        "muncEBPriorWarmupOuterPasses": (
            "observationParams.muncEBPrior.warmupOuterPasses"
        ),
        "muncEBPriorGUncertaintyMode": (
            "observationParams.muncEBPrior.gUncertaintyMode"
        ),
    }
    for attrName, defaultKey in ebPriorFieldMap.items():
        assert getattr(parsed["observationArgs"], attrName) == profile[defaultKey]
    assert (
        parsed["observationArgs"].muncCovariatesEnabled
        == profile["observationParams.muncCovariates.enabled"]
    )
    assert (
        parsed["observationArgs"].muncCovariatesMode
        == profile["observationParams.muncCovariates.mode"]
    )
    assert (
        parsed["observationArgs"].muncCovariatesFeatures
        == profile["observationParams.muncCovariates.features"]
    )
    assert (
        parsed["observationArgs"].useCountNoiseFloor
        == profile["observationParams.useCountNoiseFloor"]
    )
    assert (
        parsed["countingArgs"].subtractGlobalMedian
        == profile["countingParams.subtractGlobalMedian"]
    )
    assert (
        parsed["outputArgs"].saveBackgroundTracks
        == profile["outputParams.saveBackgroundTracks"]
    )
    assert parsed["outputArgs"].saveGains == profile["outputParams.saveGains"]
    assert (
        parsed["outputArgs"].cutoffReport
        == profile["outputParams.cutoffReport"]
    )
    assert parsed["outputArgs"].cutoffReport is constants.OUTPUT_DEFAULT_CUTOFF_REPORT
    assert (
        parsed["outputArgs"].writeRunSummary
        is constants.OUTPUT_DEFAULT_WRITE_RUN_SUMMARY
    )
    assert (
        parsed["outputArgs"].plotOptimizationPath
        is constants.OUTPUT_DEFAULT_PLOT_OPTIMIZATION_PATH
    )
    assert (
        parsed["outputArgs"].precisionDiagnosticDetail
        == constants.OUTPUT_DEFAULT_PRECISION_DIAGNOSTIC_DETAIL
    )
    assert (
        parsed["outputArgs"].maxPrecisionDiagnosticRowsPerChromosome
        == constants.OUTPUT_DEFAULT_MAX_PRECISION_DIAGNOSTIC_ROWS_PER_CHROMOSOME
    )
    assert (
        parsed["outputArgs"].maxNonTrackFileBytes
        == constants.OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES
    )
    assert (
        parsed["matchingArgs"].metadataDetail
        == constants.MATCHING_DEFAULT_METADATA_DETAIL
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
        consenrich_core.outputParams(
            convertToBigWig=parsed["outputArgs"].convertToBigWig,
            roundDigits=parsed["outputArgs"].roundDigits,
            writeUncertainty=parsed["outputArgs"].writeUncertainty,
        ).saveGains
        == constants.OUTPUT_DEFAULT_SAVE_GAINS
    )
    assert (
        consenrich_core.outputParams(
            convertToBigWig=parsed["outputArgs"].convertToBigWig,
            roundDigits=parsed["outputArgs"].roundDigits,
            writeUncertainty=parsed["outputArgs"].writeUncertainty,
        ).cutoffReport
        == constants.OUTPUT_DEFAULT_CUTOFF_REPORT
    )
    assert (
        consenrich_core.outputParams(
            convertToBigWig=parsed["outputArgs"].convertToBigWig,
            roundDigits=parsed["outputArgs"].roundDigits,
            writeUncertainty=parsed["outputArgs"].writeUncertainty,
        ).writeRunSummary
        == constants.OUTPUT_DEFAULT_WRITE_RUN_SUMMARY
    )
    assert (
        consenrich_core.outputParams(
            convertToBigWig=parsed["outputArgs"].convertToBigWig,
            roundDigits=parsed["outputArgs"].roundDigits,
            writeUncertainty=parsed["outputArgs"].writeUncertainty,
        ).precisionDiagnosticDetail
        == constants.OUTPUT_DEFAULT_PRECISION_DIAGNOSTIC_DETAIL
    )
    assert (
        consenrich_core.fitParams().useNonnegativeBackground
        == constants.FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND
    )
    assert (
        consenrich_core.fitParams().backgroundNegativePenaltyMultiplier
        == constants.FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER
    )
    assert parsed["matchingArgs"].exportFilterUncertaintyMultiplier == (
        constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER
    )
    assert parsed["matchingArgs"].uncertaintyScoreMode == (
        constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE
    )
    assert parsed["matchingArgs"].uncertaintyScoreZ == pytest.approx(
        constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z
    )
    assert consenrich_core.processParams().minQ == constants.PROCESS_DEFAULT_MIN_Q
    assert (
        consenrich_core.processParams().qSeedPriorLevel
        == constants.PROCESS_DEFAULT_Q_SEED_PRIOR_LEVEL
    )
    assert (
        consenrich_core.uncertaintyCalibrationParams().enabled
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_ENABLED
    )
    assert parsed["uncertaintyCalibrationArgs"].mode == (
        profile["uncertaintyCalibrationParams.mode"]
    )
    assert consenrich_core.uncertaintyCalibrationParams().mode == (
        constants.UNCERTAINTY_CALIBRATION_DEFAULT_MODE
    )
    assert (
        consenrich_core.uncertaintyCalibrationParams().deleteBlockVarianceMode
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_VARIANCE_MODE
    )

    cliDefaults = consenrich_cli._buildArgParser().parse_args([])
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
    assert cliDefaults.matchUncertaintyScoreMode == (
        constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE
    )
    assert cliDefaults.matchUncertaintyScoreZ == pytest.approx(
        constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z
    )
    assert cliDefaults.matchRandSeed == constants.MATCHING_DEFAULT_RAND_SEED
    assert cliDefaults.logFile is None


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
    countingParams.subtractGlobalMedian: false
    processParams.processNoiseCalibration: seed
    processParams.tuncMinScale: 0.75
    processParams.tuncMaxScale: 2.5
    processParams.processNoiseWarmupOuterPasses: 6
    processParams.precisionMultiplierMin: 0.5
    observationParams.precisionMultiplierMax: 4.0
    outputParams.saveBackgroundTracks: false
    outputParams.saveGains: false
    uncertaintyCalibrationParams.enabled: false
    matchingParams.uncertaintyScoreMode: lower_confidence
    matchingParams.uncertaintyScoreZ: 1.75
    """

    configPath = writeConfigFile(tmp_path, "config_generic_override.yaml", configYaml)
    parsed = readConfig(str(configPath))

    assert parsed["defaultConfiguration"] == "generic"
    assert parsed["fitArgs"].ECM_outerIters == 16
    assert parsed["fitArgs"].ECM_backgroundLengthScaleMultiplier == pytest.approx(2.0)
    assert parsed["countingArgs"].subtractGlobalMedian is False
    assert parsed["processArgs"].processNoiseCalibration == "seed"
    assert parsed["processArgs"].tuncMinScale == pytest.approx(0.75)
    assert parsed["processArgs"].tuncMaxScale == pytest.approx(2.5)
    assert parsed["processArgs"].processNoiseWarmupOuterPasses == 6
    assert parsed["processArgs"].precisionMultiplierMin == pytest.approx(0.5)
    assert parsed["observationArgs"].precisionMultiplierMax == pytest.approx(4.0)
    assert parsed["outputArgs"].saveBackgroundTracks is False
    assert parsed["outputArgs"].saveGains is False
    assert parsed["uncertaintyCalibrationArgs"].enabled is False
    assert parsed["matchingArgs"].uncertaintyScoreMode == "lower_confidence"
    assert parsed["matchingArgs"].uncertaintyScoreZ == pytest.approx(1.75)


def _case_processNoiseWarmupPassThroughUsesConfiguredKnobs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    processParams:
      processNoiseCalibration: fixed
      tuncLocalWindowMultiplier: 2.5
      tuncDependenceMultiplier: 3.0
      tuncMinScale: 0.5
      tuncMaxScale: 2.5
      tuncMinWindowWeight: 4.0
      tuncPriorRidge: 0.02
      tuncLevelBufferZ: 0.75
      tuncUseReliabilityWeightedWindows: false
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
        "qSeedPriorLevel",
        "tuncLocalWindowMultiplier",
        "tuncDependenceMultiplier",
        "tuncMinScale",
        "tuncMaxScale",
        "tuncMinWindowWeight",
        "tuncPriorRidge",
        "tuncLevelBufferZ",
        "tuncUseReliabilityWeightedWindows",
    }
    monkeypatch.setattr(
        consenrich_cli,
        "_coreRunConsenrichSupports",
        lambda name: name in supportedProcessKwargs,
    )
    kwargs = consenrich_cli._processNoiseRunKwargs(processArgs)

    assert kwargs["processNoiseCalibration"] == "fixed"
    assert "tuncPriorDf" not in kwargs
    assert kwargs["qSeedPriorLevel"] == pytest.approx(4.0e-8)
    assert kwargs["tuncLocalWindowMultiplier"] == pytest.approx(2.5)
    assert kwargs["tuncDependenceMultiplier"] == pytest.approx(3.0)
    assert kwargs["tuncMinScale"] == pytest.approx(0.5)
    assert kwargs["tuncMaxScale"] == pytest.approx(2.5)
    assert kwargs["tuncMinWindowWeight"] == pytest.approx(4.0)
    assert kwargs["tuncPriorRidge"] == pytest.approx(0.02)
    assert kwargs["tuncLevelBufferZ"] == pytest.approx(0.75)
    assert kwargs["tuncUseReliabilityWeightedWindows"] is False
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


def _case_readConfigDeduplicatesChromosomes(tmp_path, monkeypatch: pytest.MonkeyPatch):
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
    assert parsedOverride["fitArgs"].useNonnegativeBackground is False
    assert parsedOverride["fitArgs"].backgroundNegativePenaltyMultiplier is None
    assert parsedOverride[
        "fitArgs"
    ].ECM_backgroundLengthScaleMultiplier == pytest.approx(6.0)


def _case_readConfigAllowsEMTNuOverride(tmp_path, monkeypatch: pytest.MonkeyPatch):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.ECM_robustTNu: 4.0
    """
    parsedOverride = readConfig(
        str(
            writeConfigFile(tmp_path, "config_em_tnu_override.yaml", configOverrideYaml)
        )
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
    defaultArgs = parsedDefault["uncertaintyCalibrationArgs"]
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
        == constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_BOOTSTRAP_REPLICATES
    )
    assert defaultArgs.deleteBlockApplyTargetCalibration is None

    configExplicitYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    uncertaintyCalibrationParams.enabled: false
    uncertaintyCalibrationParams.mode: delete-block-state
    uncertaintyCalibrationParams.blockSizeBP: 25000
    uncertaintyCalibrationParams.folds: 3
    uncertaintyCalibrationParams.maxScores: 1234
    uncertaintyCalibrationParams.calibrationOuterIters: 4
    uncertaintyCalibrationParams.targets: [0.5, 0.9]
    uncertaintyCalibrationParams.targetCalibrationDelta: 0.025
    uncertaintyCalibrationParams.scaleUncertaintyByTargetCalibration: false
    uncertaintyCalibrationParams.deleteBlockVarianceMode: covariance-difference
    uncertaintyCalibrationParams.deleteBlockUseLambdaInInformation: true
    uncertaintyCalibrationParams.deleteBlockTargetSignal: state-plus-background
    uncertaintyCalibrationParams.deleteBlockFactorModel: segShrink
    uncertaintyCalibrationParams.deleteBlockFactorSegmentCount: 7
    uncertaintyCalibrationParams.deleteBlockFactorBootstrapReplicates: 9
    uncertaintyCalibrationParams.deleteBlockMinInformationFraction: 0.01
    uncertaintyCalibrationParams.deleteBlockMaxInformationFraction: 0.8
    uncertaintyCalibrationParams.deleteBlockMinDeltaVariance: 1.0e-7
    uncertaintyCalibrationParams.deleteBlockFallbackMinValidFraction: 0.5
    uncertaintyCalibrationParams.deleteBlockScoreWeightMode: sqrt-information-fraction
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
    assert explicitArgs.maxScores == 1234
    assert explicitArgs.calibrationOuterIters == 4
    assert explicitArgs.targets == (0.5, 0.9)
    assert explicitArgs.targetCalibrationDelta == pytest.approx(0.025)
    assert explicitArgs.scaleUncertaintyByTargetCalibration is False
    assert explicitArgs.deleteBlockVarianceMode == "covariance_difference"
    assert explicitArgs.deleteBlockUseLambdaInInformation is True
    assert explicitArgs.deleteBlockTargetSignal == "state_plus_background"
    assert explicitArgs.deleteBlockFactorModel == "segShrink"
    assert explicitArgs.deleteBlockFactorSegmentCount == 7
    assert explicitArgs.deleteBlockFactorBootstrapReplicates == 9
    assert explicitArgs.deleteBlockMinInformationFraction == pytest.approx(0.01)
    assert explicitArgs.deleteBlockMaxInformationFraction == pytest.approx(0.8)
    assert explicitArgs.deleteBlockMinDeltaVariance == pytest.approx(1.0e-7)
    assert explicitArgs.deleteBlockFallbackMinValidFraction == pytest.approx(0.5)
    assert explicitArgs.deleteBlockScoreWeightMode == "sqrt_information_fraction"
    assert explicitArgs.deleteBlockApplyTargetCalibration is True

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


def _case_readConfigRestrictLocalVarianceToSparseBedRequiresAvailableSparseBed(
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
    observationParams.muncTrendBlockSizeBP: 250
    observationParams.muncLocalWindowSizeBP: 500
    observationParams.muncTrendBlockDependenceMultiplier: 1.5
    observationParams.muncLocalWindowDependenceMultiplier: 2.5
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
    assert not hasattr(explicitObservationArgs, "muncAR1VarianceFunctional")
    assert explicitObservationArgs.muncTrendBlockSizeBP == 250
    assert explicitObservationArgs.muncLocalWindowSizeBP == 500
    assert explicitObservationArgs.muncTrendBlockDependenceMultiplier == 1.5
    assert explicitObservationArgs.muncLocalWindowDependenceMultiplier == 2.5
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

    configInvalidFunctional = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncAR1VarianceFunctional: innovation
    """
    configInvalidFunctionalPath = writeConfigFile(
        tmp_path,
        "config_invalid_munc_ar1_functional.yaml",
        configInvalidFunctional,
    )
    with pytest.raises(ValueError, match="muncAR1VarianceFunctional"):
        readConfig(str(configInvalidFunctionalPath))

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

    configTopLevelGMode = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncGUncertaintyMode: disabled
    """
    configTopLevelGModePath = writeConfigFile(
        tmp_path,
        "config_top_level_munc_g_mode.yaml",
        configTopLevelGMode,
    )
    with pytest.raises(ValueError, match="muncGUncertaintyMode"):
        readConfig(str(configTopLevelGModePath))

    configEBPriorMode = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.muncEBPrior.mode: sampled
    """
    configEBPriorModePath = writeConfigFile(
        tmp_path,
        "config_munc_eb_prior_mode.yaml",
        configEBPriorMode,
    )
    with pytest.raises(ValueError, match="muncEBPrior.mode"):
        readConfig(str(configEBPriorModePath))

    configReplicateTrend = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    observationParams.useReplicateTrends: true
    """
    configReplicateTrendPath = writeConfigFile(
        tmp_path,
        "config_replicate_munc_trend.yaml",
        configReplicateTrend,
    )
    with pytest.raises(ValueError, match="useReplicateTrends"):
        readConfig(str(configReplicateTrendPath))


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
    assert configParsed["countingArgs"].fragmentsGroupNorm == "NONE"


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


def _caseGlobalMedianCenterRespectsUserFlagWithControlInputs():
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
    assert enabled is True
    assert label == "yes"

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
        "replicate gain summary",
        _caseReplicateGainSummaryWritesPooledAverageAndStd,
        tmp_path,
    )
    contract_case(
        "global median centering honors user request with controls",
        _caseGlobalMedianCenterRespectsUserFlagWithControlInputs,
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
    contract_case(
        "5p extension treatment/control compatibility",
        _case_resolveExtendFrom5pBPPairsUsesTreatmentValuesForControls,
    )


def test_config_parser_defaults_and_override_contracts(
    tmp_path, monkeypatch, contract_case
):
    for label, func in (
        (
            "dotted and nested config equivalence",
            _case_readConfigDottedAndNestedEquivalent,
        ),
        ("output diagnostic tracks", _case_readConfigOutputDiagnosticTracks),
        ("generic count transform", _case_readConfigGenericCountTransform),
        (
            "shared control scale factor broadcasting",
            _case_readConfigBroadcastsSharedControlScaleFactor,
        ),
        ("process noise options", _case_readConfigProcessNoiseOptions),
        ("MUNC covariates", _case_readConfigMuncCovariates),
        (
            "MUNC covariates accept manifest feature names",
            _case_readConfigMuncCovariatesAcceptsManifestFeatureNames,
        ),
        (
            "MUNC covariates require cache",
            _case_readConfigMuncCovariatesRequireCache,
        ),
        (
            "MUNC covariates reject missing feature",
            _case_readConfigMuncCovariatesRejectMissingFeature,
        ),
        ("generic profile", _case_readConfigUsesGenericDefaultConfiguration),
        ("centralized runtime defaults", _case_runtime_defaults_are_centralized),
        (
            "generic overrides",
            _case_readConfigGenericDefaultsStillAllowExplicitOverrides,
        ),
        (
            "process noise warmup pass-through",
            _case_processNoiseWarmupPassThroughUsesConfiguredKnobs,
        ),
        (
            "unknown default profile rejected",
            _case_readConfigRejectsUnknownDefaultConfiguration,
        ),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)
    contract_case(
        "canonical uncertainty default keys",
        _caseGenericDefaultConfigurationUsesCanonicalUncertaintyKeys,
    )


def test_config_model_parameter_field_contracts(tmp_path, monkeypatch, contract_case):
    for label, func in (
        (
            "observation trend fields",
            _case_readConfigObservationTrendRemovesLinearEnvelope,
        ),
        ("chromosome deduplication", _case_readConfigDeduplicatesChromosomes),
        (
            "APN disables process precision reweighting",
            _case_readConfigAPNDisablesProcPrecReweight,
        ),
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

    convergencePath = tmp_path / "convergence.log"
    consenrich_cli._initializeDiagnosticLogs(
        consenrich_cli.DiagnosticLogPaths(
            munc_lambda=tmp_path / "munc.log",
            tunc_kappa=tmp_path / "tunc.log",
            convergence=convergencePath,
            delete_block_calibration=tmp_path / "delete.log",
        )
    )
    consenrich_cli._appendConvergenceDiagnostics(rows, convergencePath)
    lines = convergencePath.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == consenrich_cli.CONVERGENCE_LOG_COLUMNS
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

    remainingGap = consenrich_cli._normalizedRemainingObjectiveGap([10.0, 8.0, 7.0])
    np.testing.assert_allclose(remainingGap, np.asarray([1.0, 1.0 / 3.0, 0.0]))
    np.testing.assert_allclose(
        consenrich_cli._normalizedRemainingObjectiveGap([5.0, 5.0]),
        np.asarray([0.0, 0.0]),
    )

    saveCalls = []
    fakeMatplotlib = types.ModuleType("matplotlib")
    fakePyplot = types.ModuleType("matplotlib.pyplot")

    class FakeFigure:
        def suptitle(self, *args, **kwargs):
            return None

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

        def fill_between(self, *args, **kwargs):
            return None

        def get_legend_handles_labels(self, *args, **kwargs):
            return (["handle"], ["label"])

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


def test_run_summary_output_helpers(tmp_path):
    paths = consenrich_cli.DiagnosticLogPaths(
        munc_lambda=tmp_path / "munc.log",
        tunc_kappa=tmp_path / "tunc.log",
        convergence=tmp_path / "convergence.log",
        delete_block_calibration=tmp_path / "delete.log",
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
            "process_q_policy": "tunc",
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
            "global_factor": 1.5,
            "rows_valid": 10,
            "rows_fit": 5,
            "target_calibration": {
                "uncertainty_track_scale": 1.1,
                "uncertainty_track_scale_reason": "target",
            },
        },
        diagnosticLogPaths=paths,
    )
    genome = consenrich_cli._genomeRunSummaryRow(
        [row],
        elapsedSeconds=2.5,
        diagnosticLogPaths=paths,
    )
    summaryPath = tmp_path / "summary.tsv"

    consenrich_cli._writeRunSummary([row, genome], summaryPath)

    raw = summaryPath.read_text(encoding="utf-8")
    frame = pd.read_csv(summaryPath, sep="\t")
    assert list(frame.columns) == consenrich_cli.RUN_SUMMARY_COLUMNS
    assert frame["record_type"].tolist() == ["chromosome", "genome"]
    assert frame.loc[0, "lambda_lower_bound_hits"] == 1
    assert frame.loc[0, "kappa_upper_bound_hits"] == 4
    assert frame.loc[1, "chromosome"] == "genome"
    assert frame.loc[1, "intervals"] == 12
    assert "\tNA\t" in raw


def test_diagnostic_category_log_helpers(tmp_path):
    paths = consenrich_cli.DiagnosticLogPaths(
        munc_lambda=tmp_path / "munc_lambda.log",
        tunc_kappa=tmp_path / "tunc_kappa.log",
        convergence=tmp_path / "convergence.log",
        delete_block_calibration=tmp_path / "delete_block.log",
    )
    consenrich_cli._initializeDiagnosticLogs(paths)
    intervals = np.array([0, 50, 100], dtype=np.int64)
    precisionDiagnostics = {
        "precision_track_diagnostics": True,
        "lambdaExp": np.array([0.9, 1.0, 1.1], dtype=np.float32),
        "processPrecExp": np.array([1.1, 1.0, 0.9], dtype=np.float32),
        "matrixQ0": np.eye(2, dtype=np.float32),
        "process_q_policy": "student_t_kappa",
        "outputTracks": {
            "muncTrace": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "sumGain0": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "sumGain1": np.array([0.01, 0.02, 0.03], dtype=np.float32),
            "preKappaQLevel": np.array([1.0, 1.1, 1.2], dtype=np.float32),
            "preKappaQTrend": np.array([2.0, 2.1, 2.2], dtype=np.float32),
            "effectiveQLevel": np.array([0.9, 1.0, 1.1], dtype=np.float32),
            "effectiveQTrend": np.array([1.9, 2.0, 2.1], dtype=np.float32),
            "tuncQScale": np.array([1.0, 1.1, 1.2], dtype=np.float32),
        },
    }
    frame = consenrich_cli._precisionDiagnosticsFrame(
        chromosome="chr1",
        intervals=intervals,
        intervalSizeBP=50,
        matrixMunc=np.full((2, 3), 0.2, dtype=np.float32),
        pad=1.0e-4,
        precisionDiagnostics=precisionDiagnostics,
    )

    consenrich_cli._appendMuncLambdaDiagnostics(
        frame,
        paths.munc_lambda,
        chromosome="chr1",
        precisionDiagnostics=precisionDiagnostics,
    )
    consenrich_cli._appendTuncKappaDiagnostics(
        frame,
        paths.tunc_kappa,
        chromosome="chr1",
        precisionDiagnostics=precisionDiagnostics,
        runDiagnostics={
            "process_noise_calibration": {
                "processNoiseCalibrationStatus": "ok",
                "preKappaQLevel": 1.2,
            }
        },
    )
    consenrich_cli._appendConvergenceDiagnostics(
        [
            {
                "chromosome": "chr1",
                "phase": "fit",
                "path_level": "outer",
                "record_order": 0,
                "objective_name": "nll",
                "objective_value": 1.0,
            }
        ],
        paths.convergence,
    )

    muncLog = pd.read_csv(paths.munc_lambda, sep="\t")
    tuncLog = pd.read_csv(paths.tunc_kappa, sep="\t")
    convergenceLog = pd.read_csv(paths.convergence, sep="\t")
    deleteLog = pd.read_csv(paths.delete_block_calibration, sep="\t")
    assert list(muncLog.columns) == consenrich_cli.MUNC_LAMBDA_LOG_COLUMNS
    assert list(tuncLog.columns) == consenrich_cli.TUNC_KAPPA_LOG_COLUMNS
    assert list(convergenceLog.columns) == consenrich_cli.CONVERGENCE_LOG_COLUMNS
    assert list(deleteLog.columns) == consenrich_cli.DELETE_BLOCK_CALIBRATION_LOG_COLUMNS
    assert set(muncLog["record_type"]) == {"summary"}
    assert set(tuncLog["record_type"]) == {"summary"}
    assert "interval" not in set(muncLog["record_type"])
    assert "interval" not in set(tuncLog["record_type"])
    assert "rows_omitted" in set(muncLog["key"])
    assert "rows_omitted" in set(tuncLog["key"])
    assert convergenceLog["record_type"].tolist() == ["trace"]
    assert not list(tmp_path.glob("*precisionDiagnostics*"))
    assert not list(tmp_path.glob("*optimizationPath*.log"))

    sampledPaths = consenrich_cli.DiagnosticLogPaths(
        munc_lambda=tmp_path / "munc_lambda_sampled.log",
        tunc_kappa=tmp_path / "tunc_kappa_sampled.log",
        convergence=tmp_path / "convergence_sampled.log",
        delete_block_calibration=tmp_path / "delete_block_sampled.log",
    )
    consenrich_cli._initializeDiagnosticLogs(sampledPaths)
    consenrich_cli._appendMuncLambdaDiagnostics(
        frame,
        sampledPaths.munc_lambda,
        chromosome="chr1",
        precisionDiagnostics=precisionDiagnostics,
        detail="sampled",
        maxRowsPerChromosome=2,
    )
    consenrich_cli._appendTuncKappaDiagnostics(
        frame,
        sampledPaths.tunc_kappa,
        chromosome="chr1",
        precisionDiagnostics=precisionDiagnostics,
        runDiagnostics={},
        detail="sampled",
        maxRowsPerChromosome=2,
    )
    sampledMuncLog = pd.read_csv(sampledPaths.munc_lambda, sep="\t")
    sampledTuncLog = pd.read_csv(sampledPaths.tunc_kappa, sep="\t")
    assert np.sum(sampledMuncLog["record_type"] == "interval") == 2
    assert np.sum(sampledTuncLog["record_type"] == "interval") == 2


def test_cli_logging_contracts(tmp_path, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    packageLogger = logging.getLogger("consenrich")
    previousHandlers = list(packageLogger.handlers)
    previousLevel = packageLogger.level
    previousPropagate = packageLogger.propagate
    stream = io.StringIO()
    logPath = tmp_path / "run.log"
    try:
        resolvedPath = consenrich_cli._configureCliLogging(
            logPath,
            verbose=False,
            verbose2=False,
            consoleStream=stream,
        )
        assert resolvedPath == logPath
        consenrich_cli.logger.info("audit-only detail")
        consenrich_cli._logCliMilestone("live milestone %s", "shown")
        consenrich_cli._logCliPhase("MUNC prior trend", "pairs=%d", 12)
        consenrich_cli._logCliProgressMilestone("chunk %s", "hidden")
        consenrich_cli.logger.warning("visible warning")
        consoleText = stream.getvalue()
        auditText = logPath.read_text(encoding="utf-8")
        assert "live milestone shown" in consoleText
        assert "=== Consenrich | MUNC prior trend ===" in consoleText
        assert "pairs=12" in consoleText
        assert "chunk hidden" not in consoleText
        assert "visible warning" in consoleText
        assert "audit-only detail" not in consoleText
        assert "audit-only detail" not in auditText
        assert "live milestone shown" in auditText
        assert "=== Consenrich | MUNC prior trend ===" in auditText
        assert "chunk hidden" not in auditText
        assert "test_config.test_cli_logging_contracts" in auditText

        streamVerbose = io.StringIO()
        verbosePath = tmp_path / "run.verbose.log"
        consenrich_cli._configureCliLogging(
            verbosePath,
            verbose=True,
            verbose2=False,
            consoleStream=streamVerbose,
        )
        consenrich_cli.logger.info("verbose detail")
        consenrich_cli._logCliProgressMilestone("chunk %s", "shown")
        verboseConsoleText = streamVerbose.getvalue()
        verboseAuditText = verbosePath.read_text(encoding="utf-8")
        assert "verbose detail" in verboseConsoleText
        assert "chunk shown" in verboseConsoleText
        assert "verbose detail" in verboseAuditText
        assert "chunk shown" in verboseAuditText

        colorStream = io.StringIO()
        colorStream.isatty = lambda: True
        colorPath = tmp_path / "run.color.log"
        consenrich_cli._configureCliLogging(
            colorPath,
            verbose=False,
            verbose2=False,
            consoleStream=colorStream,
        )
        consenrich_cli._logCliPhase("Outputs")
        consenrich_cli._logCliMilestone(
            "Final chr1: finalNLL=12 finalForwardNIS=0.9 "
            "calibrationFactor=1.2 signChangePerKB=3.4",
            blue=True,
        )
        colorConsoleText = colorStream.getvalue()
        colorAuditText = colorPath.read_text(encoding="utf-8")
        assert "\033[34m=== Consenrich | Outputs ===\033[0m" in colorConsoleText
        assert "\033[34mFinal chr1:" in colorConsoleText
        assert "\033[" not in colorAuditText
    finally:
        for handler in list(packageLogger.handlers):
            packageLogger.removeHandler(handler)
            if handler not in previousHandlers:
                handler.close()
        for handler in previousHandlers:
            packageLogger.addHandler(handler)
        packageLogger.setLevel(previousLevel)
        packageLogger.propagate = previousPropagate


def test_cli_default_log_paths_use_run_kinds(tmp_path):
    configPath = writeConfigFile(
        tmp_path,
        "demo.yaml",
        """
        experimentName: demo exp
        """,
    )
    assert (
        consenrich_cli._defaultConfigLogPath(str(configPath)).name
        == f"consenrichOutput_demo_exp_run.v{consenrich_cli.__version__}.log"
    )
    statePath = tmp_path / "consenrichOutput_demo_state.v1.bedGraph"
    assert (
        consenrich_cli._defaultMatchLogPath(str(statePath))
        == tmp_path
        / f"consenrichOutput_demo_state.v1_consenrich_run.v{consenrich_cli.__version__}.log"
    )


def test_config_sparse_sample_source_and_matching_contracts(
    tmp_path, monkeypatch, caplog, contract_case
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
        ("structured sample sources", _case_readConfigSampleSources),
        (
            "single-cell fragments defaults",
            _case_readConfigScParamsProvideFragmentsDefaults,
        ),
        ("CRAM sources rejected", _case_readConfigRejectsCRAMSources),
    ):
        contract_case(label, _run_with_monkeypatch, monkeypatch, func, tmp_path)
    contract_case(
        "sparse interval loading", _case_loadSparseIntervalIndicesUsesBedSpan, tmp_path
    )
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
        (
            "pyBigWig empty rejection",
            _case_convertBedGraphToBigWigPyBigWigRejectsEmptyBedGraph,
        ),
        ("bedGraph sort", _case_sortBedGraphInPlace),
        (
            "bedGraph genome-order validation",
            _case_bedGraphValidationAcceptsGenomeOrderAndSortsFallback,
        ),
    ):
        contract_case(label, func, tmp_path)
