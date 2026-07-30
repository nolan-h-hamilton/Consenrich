import textwrap
import io
import json
import logging
import math
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
        centerMB=True,
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

    rawNoiseMass = np.asarray([0.125, 1.5, 3.0], dtype=np.float64)
    pseudoVarianceMass = 0.5 * float(np.sum(rawNoiseMass)) / float(
        np.sum(counts / scaleFactor)
    )
    identityArgs = baseArgs._replace(transformMethod="identity")
    rawMassFloor = consenrich_cli._countModelVarianceFloorForScaledCounts(
        counts,
        scaleFactor,
        identityArgs,
        rawNoiseMass=rawNoiseMass,
    )
    expectedRawMassFloor = (
        derivativeByMethod["identity"] * (rawNoiseMass + 0.5) * scaleFactor * scaleFactor
    )
    np.testing.assert_allclose(
        rawMassFloor,
        expectedRawMassFloor,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    rawPseudoFloor = consenrich_cli._countModelVarianceFloorForScaledCounts(
        counts,
        scaleFactor,
        identityArgs,
        rawNoiseMass=rawNoiseMass,
        countMode=constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
    )
    expectedRawPseudoFloor = (
        derivativeByMethod["identity"]
        * (rawNoiseMass + pseudoVarianceMass)
        * scaleFactor
        * scaleFactor
    )
    np.testing.assert_allclose(
        rawPseudoFloor,
        expectedRawPseudoFloor,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

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

    fractionalSource = consenrich_core.inputSource(
        "fractional.bam",
        sourceKind="BAM",
        countMode=constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
    )
    coverageSource = consenrich_core.inputSource(
        "coverage.bam",
        sourceKind="BAM",
        countMode="coverage",
    )
    bedGraphSource = consenrich_core.inputSource(
        "sample.bedGraph",
        sourceKind="BEDGRAPH",
    )
    matrixFloor = consenrich_cli._countModelFloorMatrixForScaledCounts(
        np.vstack([counts, counts, counts]),
        [scaleFactor, scaleFactor, scaleFactor],
        [fractionalSource, coverageSource, bedGraphSource],
        baseArgs._replace(transformMethod="identity"),
        rawNoiseMassMatrix=np.vstack([rawNoiseMass, rawNoiseMass, rawNoiseMass]),
        countModes=[
            constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
            "coverage",
            "coverage",
        ],
    )
    np.testing.assert_allclose(
        matrixFloor[0, :],
        expectedRawPseudoFloor,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        matrixFloor[1, :],
        expectedRawMassFloor,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert np.all(np.isnan(matrixFloor[2, :]))

    countDerivedMatrixFloor = consenrich_cli._countModelFloorMatrixForScaledCounts(
        counts.reshape(1, -1),
        [scaleFactor],
        [fractionalSource],
        baseArgs._replace(transformMethod="identity"),
        countModes=[constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP],
    )
    np.testing.assert_allclose(
        countDerivedMatrixFloor[0, :],
        derivativeByMethod["identity"] * normalizedVariance,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


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
    assert (
        "does not establish that global biological exchangeability is invalid"
        in strongMessage
    )


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
    countingParams.centerMBMethod: savgol
    samParams.defaultCountMode: ffp-center
    outputParams.plotOptimizationPath: false
    outputParams.plotCorrelationLength: false
    outputParams.cutoffReport: true
    outputParams.writeRunSummary: false
    outputParams.precisionDiagnosticDetail: sampled
    outputParams.maxPrecisionDiagnosticRowsPerChromosome: 7
    outputParams.writeReplicateExchangeabilityDiagnostics: true
    outputParams.plotPrecisionReweightingHistograms: false
    outputParams.precisionReweightingHistogramSampleSize: 123
    outputParams.maxNonTrackFileBytes: 1024
    observationParams.muncTrendBlockDependenceMultiplier: 1.75
    observationParams.muncLocalWindowDependenceMultiplier: 2.25
    observationParams.dependenceWindowCount: 32
    observationParams.dependenceWindowBP: 100000
    observationParams.dependenceMaxLagBP: 40000
    observationParams.dependenceWorkingQuantile: 0.90
    observationParams.dependenceBootstrapDraws: 40
    observationParams.dependenceMinWindowCount: 20
    observationParams.dependenceAcfPointThreshold: 0.03
    observationParams.dependenceAcfSmoothingBP: 350
    observationParams.dependenceCrossingPersistenceBP: 450
    observationParams.dependenceMinFinitePairs: 175
    observationParams.dependenceMinFinitePairCoverage: 0.65
    matchingParams.uncertaintyScoreMode: lower_confidence
    matchingParams.uncertaintyScoreZ: 1.25
    matchingParams.metadataDetail: full
    matchingParams.minPeakScore: 7.5
    matchingParams.peakMode: broad
    matchingParams.broadWeakThresholdZ: 1.1
    matchingParams.broadMaxGapBP: 12000
    loggingParams.verbosity: debug
    loggingParams.progress: off
    loggingParams.logFile: runEvents.jsonl
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
      centerMBMethod: savgol
    samParams:
      defaultCountMode: ffp-center
    outputParams:
      plotOptimizationPath: false
      plotCorrelationLength: false
      cutoffReport: true
      writeRunSummary: false
      precisionDiagnosticDetail: sampled
      maxPrecisionDiagnosticRowsPerChromosome: 7
      writeReplicateExchangeabilityDiagnostics: true
      plotPrecisionReweightingHistograms: false
      precisionReweightingHistogramSampleSize: 123
      maxNonTrackFileBytes: 1024
    observationParams:
      muncTrendBlockDependenceMultiplier: 1.75
      muncLocalWindowDependenceMultiplier: 2.25
      dependenceWindowCount: 32
      dependenceWindowBP: 100000
      dependenceMaxLagBP: 40000
      dependenceWorkingQuantile: 0.90
      dependenceBootstrapDraws: 40
      dependenceMinWindowCount: 20
      dependenceAcfPointThreshold: 0.03
      dependenceAcfSmoothingBP: 350
      dependenceCrossingPersistenceBP: 450
      dependenceMinFinitePairs: 175
      dependenceMinFinitePairCoverage: 0.65
    matchingParams:
      uncertaintyScoreMode: lower-confidence
      uncertaintyScoreZ: 1.25
      metadataDetail: full
      minPeakScore: 7.5
      peakMode: broad
      broadWeakThresholdZ: 1.1
      broadMaxGapBP: 12000
    loggingParams:
      verbosity: debug
      progress: off
      logFile: runEvents.jsonl
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
    assert countingDotted.centerMBMethod == "savgol"
    assert countingNested.centerMBMethod == "savgol"

    observationDotted = configDotted["observationArgs"]
    observationNested = configNested["observationArgs"]
    processDotted = configDotted["processArgs"]
    processNested = configNested["processArgs"]

    assert type(observationDotted) is type(observationNested)
    assert type(processDotted) is type(processNested)
    assert observationDotted == observationNested
    assert processDotted == processNested
    assert observationDotted.muncTrendBlockDependenceMultiplier == pytest.approx(1.75)
    assert observationDotted.muncLocalWindowDependenceMultiplier == pytest.approx(2.25)
    assert observationDotted.dependenceWindowCount == 32
    assert observationDotted.dependenceWindowBP == 100_000
    assert observationDotted.dependenceMaxLagBP == 40_000
    assert observationDotted.dependenceWorkingQuantile == pytest.approx(0.90)
    assert observationDotted.dependenceBootstrapDraws == 40
    assert observationDotted.dependenceMinWindowCount == 20
    assert observationDotted.dependenceAcfPointThreshold == pytest.approx(0.03)
    assert observationDotted.dependenceAcfSmoothingBP == 350
    assert observationDotted.dependenceCrossingPersistenceBP == 450
    assert observationDotted.dependenceMinFinitePairs == 175
    assert observationDotted.dependenceMinFinitePairCoverage == pytest.approx(0.65)

    outputDotted = configDotted["outputArgs"]
    outputNested = configNested["outputArgs"]
    assert outputDotted.plotOptimizationPath is False
    assert outputNested.plotOptimizationPath is False
    assert outputDotted.plotCorrelationLength is False
    assert outputNested.plotCorrelationLength is False
    assert outputDotted.cutoffReport is True
    assert outputNested.cutoffReport is True
    assert outputDotted.writeRunSummary is False
    assert outputNested.writeRunSummary is False
    assert outputDotted.precisionDiagnosticDetail == "sampled"
    assert outputNested.precisionDiagnosticDetail == "sampled"
    assert outputDotted.maxPrecisionDiagnosticRowsPerChromosome == 7
    assert outputNested.maxPrecisionDiagnosticRowsPerChromosome == 7
    assert outputDotted.writeReplicateExchangeabilityDiagnostics is True
    assert outputNested.writeReplicateExchangeabilityDiagnostics is True
    assert outputDotted.plotPrecisionReweightingHistograms is False
    assert outputNested.plotPrecisionReweightingHistograms is False
    assert outputDotted.precisionReweightingHistogramSampleSize == 123
    assert outputNested.precisionReweightingHistogramSampleSize == 123
    assert outputDotted.maxNonTrackFileBytes == 1024
    assert outputNested.maxNonTrackFileBytes == 1024
    assert outputDotted == outputNested
    assert configDotted["loggingArgs"] == configNested["loggingArgs"]
    assert configDotted["loggingArgs"].verbosity == "debug"
    assert configDotted["loggingArgs"].progress == "off"
    assert configDotted["loggingArgs"].logFile == "runEvents.jsonl"

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
    assert (
        configDotted["scArgs"].defaultCountMode
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
    assert (
        configNested["scArgs"].defaultCountMode
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
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
    assert matchingDotted.minPeakScore == pytest.approx(7.5)
    assert matchingNested.minPeakScore == pytest.approx(7.5)
    assert matchingDotted.peakMode == "broad"
    assert matchingNested.peakMode == "broad"
    assert matchingDotted.broadWeakThresholdZ == pytest.approx(1.1)
    assert matchingNested.broadWeakThresholdZ == pytest.approx(1.1)
    assert matchingDotted.broadMaxGapBP == 12000
    assert matchingNested.broadMaxGapBP == 12000


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
        "observationParams.dependenceWindowCount",
        "observationParams.dependenceWindowBP",
        "observationParams.dependenceMaxLagBP",
        "observationParams.dependenceWorkingQuantile",
        "observationParams.dependenceBootstrapDraws",
        "observationParams.dependenceMinWindowCount",
        "observationParams.dependenceAcfPointThreshold",
        "observationParams.dependenceAcfSmoothingBP",
        "observationParams.dependenceCrossingPersistenceBP",
        "observationParams.dependenceMinFinitePairs",
        "observationParams.dependenceMinFinitePairCoverage",
        "outputParams.stateShrinkageEnabled",
        "outputParams.stateShrinkageModel",
        "outputParams.stateShrinkageSpikeOddsMultiplier",
        "outputParams.stateShrinkageScaleAnchorWeight",
        "outputParams.stateShrinkageStudentTDF",
        "outputParams.stateShrinkageStudentTQuadratureOrder",
        "outputParams.plotPrecisionReweightingHistograms",
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
    assert constants.OUTPUT_DEFAULT_PRECISION_REWEIGHTING_HISTOGRAM_SAMPLE_SIZE == 200_000
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
    setupBamHelpers(monkeypatch)

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
    narrowConfigPath = writeConfigFile(
        tmp_path,
        "config_narrow_peak_mode.yaml",
        """
        experimentName: narrowPeakMode
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        matchingParams.peakMode: narrow
        """,
    )
    broadConfigPath = writeConfigFile(
        tmp_path,
        "config_broad_peak_mode.yaml",
        """
        experimentName: broadPeakMode
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        matchingParams.peakMode: broad
        """,
    )
    narrowParsed = readConfig(str(narrowConfigPath))
    broadParsed = readConfig(str(broadConfigPath))
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
    assert parsed["defaultConfiguration"] == constants.GENERIC_DEFAULT_CONFIGURATION
    assert (
        parsed["samArgs"].defaultCountMode
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
    assert (
        parsed["scArgs"].defaultCountMode
        == constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    )
    assert constants.EB_PRIOR_STRENGTH_WINSOR_TAIL == pytest.approx(0.01)
    assert (
        constants.OBSERVATION_DEFAULT_MUNC_EB_PRIOR_STRENGTH_WINSOR_TAIL
        == constants.EB_PRIOR_STRENGTH_WINSOR_TAIL
    )
    assert parsed["processArgs"].stateModel == profile["processParams.stateModel"]
    assert (
        parsed["processArgs"].processNoiseCalibration
        == profile["processParams.processNoiseCalibration"]
    )
    assert (
        parsed["processArgs"].processNoiseWarmupECMIters
        == profile["processParams.processNoiseWarmupECMIters"]
    )
    assert (
        parsed["processArgs"].processNoiseWarmupOuterPasses
        == profile["processParams.processNoiseWarmupOuterPasses"]
    )
    assert parsed["fitArgs"].t_innerIters == profile["fitParams.t_innerIters"]
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
    dependenceDefaults = {
        "dependenceWindowCount": constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_COUNT,
        "dependenceWindowBP": constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_BP,
        "dependenceMaxLagBP": constants.OBSERVATION_DEFAULT_DEPENDENCE_MAX_LAG_BP,
        "dependenceWorkingQuantile": 0.75,
        "dependenceBootstrapDraws": (
            constants.OBSERVATION_DEFAULT_DEPENDENCE_BOOTSTRAP_DRAWS
        ),
        "dependenceMinWindowCount": (
            constants.OBSERVATION_DEFAULT_DEPENDENCE_MIN_WINDOW_COUNT
        ),
        "dependenceAcfSmoothingBP": (
            constants.OBSERVATION_DEFAULT_DEPENDENCE_ACF_SMOOTHING_BP
        ),
        "dependenceCrossingPersistenceBP": (
            constants.OBSERVATION_DEFAULT_DEPENDENCE_CROSSING_PERSISTENCE_BP
        ),
        "dependenceMinFinitePairs": (
            constants.OBSERVATION_DEFAULT_DEPENDENCE_MIN_FINITE_PAIRS
        ),
        "dependenceMinFinitePairCoverage": (
            constants.OBSERVATION_DEFAULT_DEPENDENCE_MIN_FINITE_PAIR_COVERAGE
        ),
    }
    for fieldName, expectedValue in dependenceDefaults.items():
        configKey = f"observationParams.{fieldName}"
        assert getattr(parsed["observationArgs"], fieldName) == expectedValue
        assert profile[configKey] == expectedValue
    assert (
        parsed["observationArgs"].dependenceAcfPointThreshold
        == profile["observationParams.dependenceAcfPointThreshold"]
        == pytest.approx(
            constants.OBSERVATION_DEFAULT_DEPENDENCE_ACF_POINT_THRESHOLD
        )
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
        parsed["countingArgs"].centerMB == profile["countingParams.centerMB"]
    )
    assert (
        parsed["countingArgs"].centerMBMethod
        == profile["countingParams.centerMBMethod"]
    )
    assert (
        parsed["outputArgs"].saveBackgroundTracks
        == profile["outputParams.saveBackgroundTracks"]
    )
    assert (
        parsed["outputArgs"].stateShrinkageModel
        == profile["outputParams.stateShrinkageModel"]
    )
    assert (
        parsed["outputArgs"].stateShrinkageEnabled
        == profile["outputParams.stateShrinkageEnabled"]
    )
    assert (
        parsed["outputArgs"].stateShrinkageStudentTDF
        == profile["outputParams.stateShrinkageStudentTDF"]
    )
    assert (
        parsed["outputArgs"].stateShrinkageStudentTQuadratureOrder
        == profile["outputParams.stateShrinkageStudentTQuadratureOrder"]
    )
    assert (
        parsed["outputArgs"].stateShrinkageSpikeOddsMultiplier
        == profile["outputParams.stateShrinkageSpikeOddsMultiplier"]
    )
    assert (
        parsed["outputArgs"].plotPrecisionReweightingHistograms
        == profile["outputParams.plotPrecisionReweightingHistograms"]
    )
    assert (
        parsed["outputArgs"].precisionReweightingHistogramSampleSize
        == profile["outputParams.precisionReweightingHistogramSampleSize"]
    )
    assert (
        consenrich_core.outputParams(
            convertToBigWig=parsed["outputArgs"].convertToBigWig,
            roundDigits=parsed["outputArgs"].roundDigits,
            writeUncertainty=parsed["outputArgs"].writeUncertainty,
        ).stateShrinkageModel
        == constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_MODEL
    )
    assert (
        consenrich_core.outputParams(
            convertToBigWig=parsed["outputArgs"].convertToBigWig,
            roundDigits=parsed["outputArgs"].roundDigits,
            writeUncertainty=parsed["outputArgs"].writeUncertainty,
        ).plotCorrelationLength
        is constants.OUTPUT_DEFAULT_PLOT_CORRELATION_LENGTH
    )
    assert parsed["outputArgs"].saveGains == profile["outputParams.saveGains"]
    assert (
        parsed["outputArgs"].writeReplicateExchangeabilityDiagnostics
        is constants.OUTPUT_DEFAULT_WRITE_REPLICATE_EXCHANGEABILITY_DIAGNOSTICS
        is True
    )
    assert (
        parsed["outputArgs"].writeReplicateExchangeabilityDiagnostics
        == profile["outputParams.writeReplicateExchangeabilityDiagnostics"]
    )
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
        parsed["outputArgs"].plotCorrelationLength
        is constants.OUTPUT_DEFAULT_PLOT_CORRELATION_LENGTH
    )
    assert parsed["outputArgs"].plotCorrelationLength is True
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
    assert parsed["loggingArgs"].verbosity == constants.LOGGING_DEFAULT_VERBOSITY
    assert parsed["loggingArgs"].progress == constants.LOGGING_DEFAULT_PROGRESS
    assert parsed["loggingArgs"].logFile is None
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
    assert (
        consenrich_core.fitParams().t_innerIters
        == constants.FIT_DEFAULT_T_INNER_ITERS
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
    assert (
        parsed["matchingArgs"].minPeakScore
        == constants.MATCHING_DEFAULT_MIN_PEAK_SCORE
    )
    assert parsed["matchingArgs"].peakMode == constants.MATCHING_DEFAULT_PEAK_MODE
    assert parsed["matchingArgs"].peakMode == "both"
    assert narrowParsed["matchingArgs"].peakMode == "narrow"
    assert broadParsed["matchingArgs"].peakMode == "broad"
    assert parsed["matchingArgs"].broadWeakThresholdZ == pytest.approx(
        constants.MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z
    )
    assert (
        parsed["matchingArgs"].broadMaxGapBP
        == constants.MATCHING_DEFAULT_BROAD_MAX_GAP_BP
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
    assert (
        cliDefaults.matchMinPeakScore
        == constants.MATCHING_DEFAULT_MIN_PEAK_SCORE
    )
    assert cliDefaults.matchPeakMode == constants.MATCHING_DEFAULT_PEAK_MODE
    assert cliDefaults.matchPeakMode == "both"
    cliNarrow = consenrich_cli._buildArgParser().parse_args(
        ["--match-peak-mode", "narrow"]
    )
    cliBroad = consenrich_cli._buildArgParser().parse_args(
        ["--match-peak-mode", "broad"]
    )
    cliBoth = consenrich_cli._buildArgParser().parse_args(
        ["--match-peak-mode", "both"]
    )
    assert cliNarrow.matchPeakMode == "narrow"
    assert cliBroad.matchPeakMode == "broad"
    assert cliBoth.matchPeakMode == "both"
    assert cliDefaults.matchBroadWeakThresholdZ == pytest.approx(
        constants.MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z
    )
    assert (
        cliDefaults.matchBroadMaxGapBP
        == constants.MATCHING_DEFAULT_BROAD_MAX_GAP_BP
    )
    assert cliDefaults.matchRandSeed == constants.MATCHING_DEFAULT_RAND_SEED
    assert cliDefaults.logFile is None
    assert cliDefaults.verbosity is None
    assert cliDefaults.progress is None
    monkeypatch.setattr(
        consenrich_cli.sys,
        "argv",
        ["consenrich", "--verbosity", "debug", "--verbose"],
    )
    with pytest.raises(SystemExit) as conflictExit:
        consenrich_cli.main()
    assert conflictExit.value.code == 2


def _case_observationMuncSeedWeightDefaultAndPrecisionIndependence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
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
    setupBamHelpers(monkeypatch)

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
    observationParams.dependenceWindowCount: 48
    observationParams.dependenceWindowBP: 120000
    observationParams.dependenceMaxLagBP: 45000
    observationParams.dependenceWorkingQuantile: 0.90
    observationParams.dependenceBootstrapDraws: 75
    observationParams.dependenceMinWindowCount: 24
    observationParams.dependenceAcfPointThreshold: 0.02
    observationParams.dependenceAcfSmoothingBP: 300
    observationParams.dependenceCrossingPersistenceBP: 400
    observationParams.dependenceMinFinitePairs: 150
    observationParams.dependenceMinFinitePairCoverage: 0.60
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
    outputParams.plotPrecisionReweightingHistograms: false
    outputParams.precisionReweightingHistogramSampleSize: 12345
    uncertaintyCalibrationParams.enabled: false
    matchingParams.uncertaintyScoreMode: lower_confidence
    matchingParams.uncertaintyScoreZ: 1.75
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
    assert parsed["observationArgs"].dependenceWindowCount == 48
    assert parsed["observationArgs"].dependenceWindowBP == 120_000
    assert parsed["observationArgs"].dependenceMaxLagBP == 45_000
    assert parsed["observationArgs"].dependenceWorkingQuantile == pytest.approx(0.90)
    assert parsed["observationArgs"].dependenceBootstrapDraws == 75
    assert parsed["observationArgs"].dependenceMinWindowCount == 24
    assert parsed["observationArgs"].dependenceAcfPointThreshold == pytest.approx(0.02)
    assert parsed["observationArgs"].dependenceAcfSmoothingBP == 300
    assert parsed["observationArgs"].dependenceCrossingPersistenceBP == 400
    assert parsed["observationArgs"].dependenceMinFinitePairs == 150
    assert parsed["observationArgs"].dependenceMinFinitePairCoverage == pytest.approx(
        0.60
    )
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
    assert parsed["outputArgs"].plotPrecisionReweightingHistograms is False
    assert parsed["outputArgs"].precisionReweightingHistogramSampleSize == 12345
    assert parsed["uncertaintyCalibrationArgs"].enabled is False
    assert parsed["matchingArgs"].uncertaintyScoreMode == "lower_confidence"
    assert parsed["matchingArgs"].uncertaintyScoreZ == pytest.approx(1.75)


def _case_readConfigRejectsLowStateShrinkageStudentTDF(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

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
    setupBamHelpers(monkeypatch)

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
    setupBamHelpers(monkeypatch)

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
    setupBamHelpers(monkeypatch)

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
    setupBamHelpers(monkeypatch)

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
    defaultOutputArgs = parsedDefault["outputArgs"]
    assert defaultArgs.enabled is True
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
    uncertaintyCalibrationParams.deleteBlockReplicateDependenceRho: 0.35
    uncertaintyCalibrationParams.deleteBlockTargetSignal: state-plus-background
    uncertaintyCalibrationParams.deleteBlockFactorModel: segShrink
    uncertaintyCalibrationParams.deleteBlockFactorSegmentCount: 7
    uncertaintyCalibrationParams.deleteBlockFactorBootstrapReplicates: 9
    uncertaintyCalibrationParams.deleteBlockDeletionProbability: 0.4
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
    configParsed = readConfig(str(configPath))
    inputArgs = configParsed["inputArgs"]

    assert inputArgs.treatmentSources is not None
    assert inputArgs.treatmentSources[0].sourceKind == "BEDGRAPH"
    assert inputArgs.bamFiles == [str(indexedPath)]
    assert inputArgs.treatmentSources[0].path == str(indexedPath)
    assert bedGraphPath.exists()
    assert indexedPath.exists()
    assert Path(f"{indexedPath}.tbi").exists()
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
    configParsed = readConfig(str(configPath))
    inputArgs = configParsed["inputArgs"]

    assert inputArgs.treatmentSources is not None
    assert inputArgs.treatmentSources[0].sourceKind == "BEDGRAPH"
    assert inputArgs.bamFiles == [str(indexedPath)]
    assert indexedPath.exists()
    assert Path(f"{indexedPath}.tbi").exists()


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
    with pytest.raises(ValueError, match="CRAM inputs are unsupported"):
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


def test_convertBedGraphToBigWigSkipsValidatedBedGraphScan(tmp_path, monkeypatch):
    experimentName = "toy"
    version = consenrich_io.__version__
    chromSizesPath = tmp_path / "toy.chrom.sizes"
    chromSizesPath.write_text("chr1\t100\n", encoding="ascii")
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
    realReadChromSizes = consenrich_io._readChromSizes

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
    ):
        convertCalls.append((bedGraphPath, chromSizesFile, bigWigPath, chromSizes))
        Path(bigWigPath).write_bytes(b"x" * 128)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(consenrich_io, "_readChromSizes", recordReadChromSizes)
    monkeypatch.setattr(consenrich_io, "_validateBedGraphSorted", recordValidate)
    monkeypatch.setattr(
        consenrich_io,
        "_convertBedGraphToBigWigPyBigWig",
        recordConvert,
    )

    consenrich_io.convertBedGraphToBigWig(
        experimentName,
        str(chromSizesPath),
        suffixes=suffixes,
        validatedBedGraphs={str(bedGraphPaths["state"])},
        deleteBedGraphsAfterBigWig=True,
    )

    assert readCalls == [str(chromSizesPath)]
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


def _case_sortBedGraphUsesSizesOrderFilteredToPlannedChromosomes(tmp_path):
    bedGraphPath = tmp_path / "planned_order.bedGraph"
    chromSizesPath = tmp_path / "planned_order.chrom.sizes"
    chromSizesPath.write_text(
        "chr1\t100\nchr2\t100\nchr10\t100\nchrM\t100\n",
        encoding="ascii",
    )
    plannedChromosomes = ["chr10", "chr1", "chr10"]
    chromSizes = {
        chrom: int(size)
        for chrom, size in (
            line.split("\t")
            for line in chromSizesPath.read_text(encoding="ascii").splitlines()
        )
    }
    chromOrder = consenrich_cli._chromSizesOrderForPlannedChromosomes(
        chromSizes,
        plannedChromosomes,
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


def _caseCenterMBRespectsUserFlagWithControlInputs():
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
        centerMB=True,
    )

    enabled, label = consenrich_cli._resolveCenterMBStatus(
        countingArgs,
        controlsPresent=False,
    )
    assert enabled is True
    assert label == "yes"

    enabled, label = consenrich_cli._resolveCenterMBStatus(
        countingArgs,
        controlsPresent=True,
    )
    assert enabled is True
    assert label == "yes"

    disabledArgs = countingArgs._replace(centerMB=False)
    enabled, label = consenrich_cli._resolveCenterMBStatus(
        disabledArgs,
        controlsPresent=True,
    )
    assert enabled is False
    assert label == "no"


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
        "centerMB honors user request with controls",
        _caseCenterMBRespectsUserFlagWithControlInputs,
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
            "processQScale": np.asarray([1.1, 0.9], dtype=np.float64),
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
            {"fold_refits": {"block_len_intervals": 2}},
        ),
        np.asarray([2.0, 5.0], dtype=np.float64),
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
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
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
        assert (
            consenrich_cli._plotReplicateCalibration(
                gainPlotRows,
                str(tmp_path / "replicate_calibration.png"),
            )
            is True
        )
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


def test_correlation_length_plot_helper_writes_artifact_and_handles_missing_matplotlib(
    tmp_path,
    monkeypatch,
):
    details = {
        "status": "estimated",
        "method": "rankWeightedFinitePairWindowACF",
        "randomSeed": 1729,
        "inferenceScope": "conditionalOnInputTracksAndSelectedWindows",
        "confidenceIntervalMethod": (
            "centralInterquartileSimultaneousLogLogKMSurvivalBand"
        ),
        "survivalBandRegionLower": 0.25,
        "survivalBandRegionUpper": 0.75,
        "survivalBandJumpClosureUsed": True,
        "survivalBandJumpClosureCount": 1,
        "confidenceLevel": 0.95,
        "estimateBP": 400.0,
        "fullSampleMedianRadiusBP": 400.0,
        "lowerBP": 300.0,
        "upperBP": 550.0,
        "workingSpanBP": 800.0,
        "fullSampleWorkingSpanBP": 800.0,
        "workingQuantile": 0.95,
        "chromosomesUsed": ["chr1", "chr2", "chr3", "chr4"],
        "candidateWindowCount": 24,
        "evaluatedCandidateWindowCount": 21,
        "selectedWindowCount": 20,
        "crossedWindowCount": 19,
        "rightCensoredWindowCount": 1,
        "supportCensoredWindowCount": 1,
        "capCensoredWindowCount": 0,
        "selectedAutosomeCount": 4,
        "censorFraction": 0.05,
        "censorTimeBPMinimum": 1000.0,
        "censorTimeBPMedian": 1000.0,
        "censorTimeBPMaximum": 1000.0,
        "inputRowCount": 3,
        "uniqueRowCount": 2,
        "duplicateRowCount": 1,
        "rowDeduplication": "exactBytes",
        "acfThreshold": 0.1,
        "acfSmoothingBP": 250,
        "acfSmoothingBins": 5,
        "crossingPersistenceBP": 250,
        "crossingPersistenceBins": 5,
        "minFinitePairs": 200,
        "minFinitePairCoverage": 0.5,
        "finitePairMinimumUsed": 500.0,
        "finitePairCoverageMinimumUsed": 0.75,
        "validRowsAtCrossingMinimum": 2,
        "dominantACFPeriodBPMedian": None,
        "unresolvedPeriodFraction": 0.0,
        "oscillationStrength": None,
        "postCrossingRevival": None,
        "selectedAdjacencyCount": 12,
        "selectedLongestRun": 5,
        "bootstrapMethod": "hierarchicalAutosomeStationaryWindow",
        "bootstrapBlockLengthWindows": 3,
        "bootstrapDrawsRequested": 4,
        "bootstrapResolvedMedianDraws": 4,
        "bootstrapResolvedWorkingDraws": 4,
        "bootstrapResolvedJointDraws": 4,
        "radiusDistributionBP": [250.0, 400.0, 550.0, 1000.0],
        "radiusCensored": [False, False, False, True],
        "selectedWindows": [
            {
                "chromosome": f"chr{index + 1}",
                "startBP": index * 100_000,
                "endBP": (index + 1) * 100_000,
                "score": 10.0 - index,
                "positiveSignalRank": float(index + 1),
                "rawCrossingLagBP": None if index == 3 else (250.0 + 150.0 * index),
                "censorLagBP": 1000.0 if index == 3 else (250.0 + 150.0 * index),
                "gaussianEquivalentRadiusBP": radiusBP,
                "rightCensored": index == 3,
                "censorReason": "support" if index == 3 else "none",
                "supportCapLagBP": 1000.0,
                "finitePairMinimumUsed": 500.0,
                "finitePairCoverageMinimumUsed": 0.75,
                "validRowCount": 2,
                "validRowsAtCrossing": 2,
                "dominantACFPeriodBP": 200.0,
                "oscillationStrength": 0.2,
                "postCrossingRevival": 0.15,
            }
            for index, radiusBP in enumerate((250.0, 400.0, 550.0, 1000.0))
        ],
        "bootstrapMedianRadiusBP": [350.0, 400.0, 450.0, 425.0],
        "bootstrapWorkingSpanBP": [700.0, 800.0, 900.0, 850.0],
    }
    row = consenrich_cli._correlationLengthRow(
        intervalSizeBP=50,
        pointIntervals=8,
        contextBP=801,
        details=details,
    )
    assert row["correlation_length_bp"] == pytest.approx(400.0)
    assert row["working_span_bp"] == pytest.approx(800.0)
    assert row["random_seed"] == 1729
    assert row["evaluated_candidate_window_count"] == 21
    assert row["survival_band_region_lower"] == pytest.approx(0.25)
    assert row["survival_band_region_upper"] == pytest.approx(0.75)
    assert row["selected_window_count"] == 20
    assert row["duplicate_row_count"] == 1
    assert row["dominant_acf_period_bp_median"] is None
    assert row["oscillation_strength"] is None
    assert row["post_crossing_revival"] is None
    assert consenrich_cli._formatOptionalLogValue(None) == "NA"
    assert row["survival_band_jump_closure_used"] is True
    assert row["survival_band_jump_closure_count"] == 1
    windowRows = consenrich_cli._correlationLengthWindowRows(details)
    assert len(windowRows) == 4
    assert tuple(windowRows[0]) == tuple(
        consenrich_cli.CORRELATION_LENGTH_WINDOW_COLUMNS
    )
    assert windowRows[-1]["censor_reason"] == "support"
    assert windowRows[-1]["positive_signal_rank"] == pytest.approx(4.0)
    assert windowRows[-1]["raw_crossing_lag_bp"] is None
    plotPath = tmp_path / "correlation_length.png"

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "matplotlib", None)
        assert (
            consenrich_cli._plotCorrelationLengthInference(
                row,
                details,
                plotPath,
            )
            is False
        )

    logCalls = []
    saveCalls = []

    def fakeLogFileWritten(loggerArg, *, event, path, fields):
        logCalls.append((event, path, tuple(fields)))

    class FakeColorbar:
        def set_label(self, *args, **kwargs):
            return None

    class FakeFigure:
        def savefig(self, path, dpi=None):
            saveCalls.append((path, dpi))
            Path(path).write_bytes(b"png")

        def colorbar(self, *args, **kwargs):
            return FakeColorbar()

        def suptitle(self, *args, **kwargs):
            return None

    class FakeAxis:
        transAxes = object()

        def axvline(self, *args, **kwargs):
            return None

        def axvspan(self, *args, **kwargs):
            return None

        def hist(self, *args, **kwargs):
            return None

        def fill_between(self, *args, **kwargs):
            return None

        def plot(self, *args, **kwargs):
            return None

        def step(self, *args, **kwargs):
            return None

        def scatter(self, *args, **kwargs):
            return object()

        def text(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def set_yticks(self, *args, **kwargs):
            return None

        def set_xscale(self, *args, **kwargs):
            return None

        def set_xlim(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

    fakeMatplotlib = types.ModuleType("matplotlib")
    fakePyplot = types.ModuleType("matplotlib.pyplot")
    fakeMatplotlib.use = lambda *args, **kwargs: None
    fakePyplot.rcParams = {}
    fakePyplot.subplots = lambda *args, **kwargs: (
        FakeFigure(),
        [FakeAxis(), FakeAxis(), FakeAxis()],
    )
    fakePyplot.close = lambda *args, **kwargs: None
    fakeMatplotlib.pyplot = fakePyplot

    monkeypatch.setattr(
        consenrich_cli.logging_utils,
        "log_file_written",
        fakeLogFileWritten,
    )
    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "matplotlib", fakeMatplotlib)
        mp.setitem(sys.modules, "matplotlib.pyplot", fakePyplot)
        assert (
            consenrich_cli._plotCorrelationLengthInference(
                row,
                details,
                plotPath,
            )
            is True
        )

    assert saveCalls == [(str(plotPath), 400)]
    assert logCalls == [
        (
            "artifact.correlation_length_plot",
            str(plotPath),
            (
                ("format", "png"),
                ("dpi", 400),
                (
                    "confidenceIntervalMethod",
                    "centralInterquartileSimultaneousLogLogKMSurvivalBand",
                ),
                ("survivalBandRegionLower", 0.25),
                ("survivalBandRegionUpper", 0.75),
                ("survivalBandJumpClosureUsed", True),
                ("survivalBandJumpClosureCount", 1),
                ("bootstrapBlockLengthWindows", 3),
            ),
        )
    ]
    assert consenrich_cli._correlationLengthPlotPath("exp name").name == (
        f"consenrichOutput_exp_name_correlationLength.v{consenrich_cli.__version__}.png"
    )


def test_munc_estimation_log_uses_working_span_label(monkeypatch):
    logCalls = []

    def fakeLogAsciiBlock(title, rows, **kwargs):
        logCalls.append((title, tuple(rows)))

    monkeypatch.setattr(consenrich_core, "_logAsciiBlock", fakeLogAsciiBlock)
    sizing = consenrich_core._resolveMuncRuntimeSizing(
        intervalSizeBP=50,
        dependenceSpanIntervals=7,
        muncTrendBlockSizeBP=None,
        muncLocalWindowSizeBP=None,
        muncTrendBlockDependenceMultiplier=2.0,
        muncLocalWindowDependenceMultiplier=3.0,
    )

    consenrich_cli._logMuncEstimationParameters(
        chromosomeCount=1,
        sampleCount=2,
        intervalSizeBP=50,
        sizing=sizing,
        muncVarianceModel="kalman",
        samplingIters=100,
        dependenceContextBP=701,
        dependenceSpanIntervals=7,
        trendMultiplier=2.0,
        localMultiplier=3.0,
        observationArgs=consenrich_core.observationParams(
            minR=constants.OBSERVATION_DEFAULT_MIN_R,
            maxR=constants.OBSERVATION_DEFAULT_MAX_R,
            samplingIters=constants.OBSERVATION_DEFAULT_SAMPLING_ITERS,
            EB_use=constants.OBSERVATION_DEFAULT_EB_USE,
            EB_setNu0=constants.OBSERVATION_DEFAULT_EB_SET_NU0,
            EB_setNuL=constants.OBSERVATION_DEFAULT_EB_SET_NUL,
            trendNumBasis=constants.OBSERVATION_DEFAULT_TREND_NUM_BASIS,
            trendMinObsPerBasis=constants.OBSERVATION_DEFAULT_TREND_MIN_OBS_PER_BASIS,
            trendMinEdf=constants.OBSERVATION_DEFAULT_TREND_MIN_EDF,
            trendMaxEdf=constants.OBSERVATION_DEFAULT_TREND_MAX_EDF,
            trendLambdaMin=constants.OBSERVATION_DEFAULT_TREND_LAMBDA_MIN,
            trendLambdaMax=constants.OBSERVATION_DEFAULT_TREND_LAMBDA_MAX,
            trendLambdaGridSize=constants.OBSERVATION_DEFAULT_TREND_LAMBDA_GRID_SIZE,
            numNearest=constants.OBSERVATION_DEFAULT_NUM_NEAREST,
            sparseSupportScaleBP=constants.OBSERVATION_DEFAULT_SPARSE_SUPPORT_SCALE_BP,
            sparseSupportPrior=constants.OBSERVATION_DEFAULT_SPARSE_SUPPORT_PRIOR,
            pad=constants.OBSERVATION_DEFAULT_PAD,
        ),
        sparseBedEnabled=False,
        varianceFloor=0.01,
        varianceCap=10.0,
        trendNumBasis=25,
        trendMinObsPerBasis=50.0,
        trendMinEdf=2.0,
        trendMaxEdf=10.0,
        trendLambdaMin=1.0e-6,
        trendLambdaMax=1.0e6,
        trendLambdaGridSize=101,
        pooledPairCount=12,
        seedPassCount=3,
    )

    assert len(logCalls) == 1
    title, rows = logCalls[0]
    rowMap = dict(rows)
    assert title == "MUNC estimation parameters"
    assert rowMap["MUNC dependence working span"] == 7
    assert rowMap["MUNC trend block source"] == "dependence working span"
    assert rowMap["MUNC local window source"] == "dependence working span"
    assert rowMap["trend dependence span multiplier"] == pytest.approx(2.0)
    assert rowMap["local dependence span multiplier"] == pytest.approx(3.0)
    assert "MUNC dependence span" not in rowMap
    assert "trend span multiplier" not in rowMap
    assert "local span multiplier" not in rowMap


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
    assert "\033[38;2;191;87;0m" in consoleText
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
            _case_sortBedGraphUsesSizesOrderFilteredToPlannedChromosomes,
        ),
    ):
        contract_case(label, func, tmp_path)


@pytest.mark.parametrize(
    ("caseText", "fieldName"),
    (
        ("observationParams.dependenceWindowCount: 0", "dependenceWindowCount"),
        ("observationParams.dependenceWindowBP: 0", "dependenceWindowBP"),
        ("observationParams.dependenceMaxLagBP: 0", "dependenceMaxLagBP"),
        (
            "observationParams.dependenceWorkingQuantile: 1",
            "dependenceWorkingQuantile",
        ),
        (
            "observationParams.dependenceBootstrapDraws: 0",
            "dependenceBootstrapDraws",
        ),
        (
            "observationParams.dependenceMinWindowCount: 0",
            "dependenceMinWindowCount",
        ),
        (
            "observationParams.dependenceAcfPointThreshold: 1",
            "dependenceAcfPointThreshold",
        ),
        (
            "observationParams.dependenceAcfSmoothingBP: 0",
            "dependenceAcfSmoothingBP",
        ),
        (
            "observationParams.dependenceCrossingPersistenceBP: 0",
            "dependenceCrossingPersistenceBP",
        ),
        (
            "observationParams.dependenceMinFinitePairs: 0",
            "dependenceMinFinitePairs",
        ),
        (
            "observationParams.dependenceMinFinitePairCoverage: 1.1",
            "dependenceMinFinitePairCoverage",
        ),
        (
            "observationParams.dependenceWindowBP: 1000\n"
            "observationParams.dependenceMaxLagBP: 501",
            "dependenceMaxLagBP",
        ),
        (
            "observationParams.dependenceWindowCount: 20\n"
            "observationParams.dependenceMinWindowCount: 21",
            "dependenceMinWindowCount",
        ),
    ),
)
def test_dependence_config_rejects_out_of_range_values(
    tmp_path,
    monkeypatch,
    caseText,
    fieldName,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    indentedCaseText = caseText.replace("\n", "\n        ")
    configPath = writeConfigFile(
        tmp_path,
        f"config_bad_dependence_{fieldName}.yaml",
        f"""
        experimentName: dependenceConfigRange
        inputParams.bamFiles: [smallTest.bam]
        genomeParams.name: testGenome
        {indentedCaseText}
        """,
    )
    with pytest.raises(ValueError, match=fieldName):
        readConfig(str(configPath))
