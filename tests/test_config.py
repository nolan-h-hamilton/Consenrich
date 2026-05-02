import textwrap
import logging
from pathlib import Path

import numpy as np
import pytest

from consenrich.consenrich import readConfig
import consenrich.consenrich as consenrich
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


def test_munc_worker_count_unknown_memory_uses_cpu_cap(monkeypatch):
    monkeypatch.setattr(consenrich.os, "cpu_count", lambda: 8)

    workers = consenrich._getMuncWorkerCount(
        10,
        1000,
        availableMemoryBytes=None,
        logger_=None,
    )

    assert workers == 4


def test_munc_worker_count_low_memory_keeps_one_worker(monkeypatch):
    monkeypatch.setattr(consenrich.os, "cpu_count", lambda: 8)

    workers = consenrich._getMuncWorkerCount(
        10,
        1000,
        availableMemoryBytes=64 * 1024 * 1024,
        logger_=None,
    )

    assert workers == 1


def test_munc_worker_count_moderate_memory_caps_below_cpu(monkeypatch):
    monkeypatch.setattr(consenrich.os, "cpu_count", lambda: 16)

    workers = consenrich._getMuncWorkerCount(
        10,
        1000,
        availableMemoryBytes=1024 * 1024 * 1024,
        logger_=None,
    )

    assert workers == 4


def test_ensureInput():
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


def test_readConfigDottedAndNestedEquivalent(tmp_path, monkeypatch: pytest.MonkeyPatch):
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
    assert observationDotted.minR == observationNested.minR

    samDotted = configDotted["samArgs"]
    samNested = configNested["samArgs"]
    matchingDotted = configDotted["matchingArgs"]
    matchingNested = configNested["matchingArgs"]

    assert type(samDotted) is type(samNested)
    assert type(matchingDotted) is type(matchingNested)

    assert samDotted.samThreads == samNested.samThreads
    assert matchingDotted.enabled == matchingNested.enabled
    assert matchingDotted.thresholdZ == matchingNested.thresholdZ
    assert matchingDotted.nestedRoccoIters == matchingNested.nestedRoccoIters
    assert matchingDotted.nestedRoccoBudgetScale == matchingNested.nestedRoccoBudgetScale


def test_readConfigDeduplicatesChromosomes(
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


def test_readConfigAPNDisablesProcPrecReweight(
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


def test_readConfigUsesEMUseField(
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


def test_readConfigUsesZeroCenterIdentifiabilityFields(
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
    assert parsedDefault["fitArgs"].EM_zeroCenterBackground is True
    assert parsedDefault["fitArgs"].EM_zeroCenterReplicateBias is True

    configOverrideYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    fitParams.EM_zeroCenterBackground: false
    fitParams.EM_zeroCenterReplicateBias: false
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


def test_readConfigDefaultsEMTNuToEightAndAllowsOverride(
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


def test_readConfigUsesInnerAndOuterEMToleranceFields(
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


def test_readConfigUsesEffectiveInfoRescaleField(
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
                tmp_path, "config_effective_info_default.yaml", configDefaultYaml
            )
        )
    )
    assert parsedDefault["stateArgs"].effectiveInfoRescale is True
    assert parsedDefault["stateArgs"].effectiveInfoBlockLengthBP == 50_000
    assert parsedDefault["stateArgs"].effectiveInfoBandwidthBP is None

    configExplicitYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    stateParams.effectiveInfoRescale: false
    stateParams.effectiveInfoBlockLengthBP: 25000
    stateParams.effectiveInfoBandwidthBP: 500
    """
    parsedExplicit = readConfig(
        str(
            writeConfigFile(
                tmp_path, "config_effective_info_false.yaml", configExplicitYaml
            )
        )
    )
    assert parsedExplicit["stateArgs"].effectiveInfoRescale is False
    assert parsedExplicit["stateArgs"].effectiveInfoBlockLengthBP == 25_000
    assert parsedExplicit["stateArgs"].effectiveInfoBandwidthBP == 500

    configAliasYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    stateParams.effectiveInfoBlockLength: 75000
    """
    parsedAlias = readConfig(
        str(
            writeConfigFile(
                tmp_path,
                "config_effective_info_alias.yaml",
                configAliasYaml,
            )
        )
    )
    assert parsedAlias["stateArgs"].effectiveInfoBlockLengthBP == 75_000


def testResolveEffectiveInfoBandwidthIntervals():
    assert consenrich._resolveEffectiveInfoBandwidthIntervals(None, 25) is None
    assert consenrich._resolveEffectiveInfoBandwidthIntervals(0, 25) is None
    assert consenrich._resolveEffectiveInfoBandwidthIntervals(-100, 25) is None
    assert consenrich._resolveEffectiveInfoBandwidthIntervals(250, 25) == 10
    assert consenrich._resolveEffectiveInfoBandwidthIntervals(260, 25) == 11


def testResolveEffectiveInfoBlockLengthIntervals():
    assert (
        consenrich._resolveEffectiveInfoBlockLengthIntervals(50_000, 25, 10_000)
        == 2000
    )
    assert (
        consenrich._resolveEffectiveInfoBlockLengthIntervals(50_000, 100, 200)
        == 200
    )
    assert consenrich._resolveEffectiveInfoBlockLengthIntervals(0, 25, 123) == 123


def test_readConfigNumNearestRequiresExplicitSparseBed(
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


def test_readConfigRestrictLocalAR1ToSparseBedRequiresAvailableSparseBed(
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


def test_loadSparseIntervalIndicesUsesBedSpan(tmp_path):
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

    indices = consenrich._loadSparseIntervalIndices(
        str(sparseBedPath),
        "chrTest",
        intervals,
    )

    assert indices.tolist() == [2, 3, 4]


def test_readConfigSampleSources(tmp_path, monkeypatch: pytest.MonkeyPatch):
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


def test_readConfigSamplesSupportBedGraph(
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


def test_readConfigSamplesSupportExplicitBedGraphFormat(
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


def test_readConfigScParamsProvideFragmentsDefaults(
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


def test_readConfigUsesExplicitBamInputModeRead1(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    configYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam]
    genomeParams.name: testGenome
    samParams.bamInputMode: read1
    """

    configPath = writeConfigFile(tmp_path, "config_macs_like_bam.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    samArgs = configParsed["samArgs"]

    assert samArgs.bamInputMode == "read1"
    assert samArgs.defaultCountMode == "coverage"
    assert samArgs.inferFragmentLength == 0


def test_resolveExtendFrom5pBPPairsUsesTreatmentValuesForControls():
    treatment, control = consenrich._resolveExtendFrom5pBPPairs(
        [150, 180],
        [90, 110],
    )

    assert treatment == [150, 180]
    assert control == [150, 180]


def test_readConfigMatchingDefaultsToROCCO(
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
    assert configParsed["matchingArgs"].gamma == pytest.approx(0.5)
    assert configParsed["matchingArgs"].nestedRoccoIters == 3
    assert configParsed["matchingArgs"].nestedRoccoBudgetScale == pytest.approx(0.5)
    assert not hasattr(configParsed["matchingArgs"], "minMatchLengthBP")
    assert not hasattr(configParsed["matchingArgs"], "merge")
    assert not hasattr(configParsed["matchingArgs"], "mergeGapBP")
    assert configParsed["countingArgs"].intervalSizeBP == 25
    assert not hasattr(configParsed["countingArgs"], "smoothSpanBP")


def test_readConfigRejectsCRAMSources(
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


def test_convertBedGraphToBigWigPyBigWigWritesExpectedTrack(tmp_path):
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
                "chr10 0 5 10.0",
                "chr2\t0\t8\t2.0",
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


def test_convertBedGraphToBigWigPyBigWigRejectsOutOfBounds(tmp_path):
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


def test_convertBedGraphToBigWigPyBigWigRejectsEmptyBedGraph(tmp_path):
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




def test_sortBedGraphInPlace(tmp_path):
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


def test_resolveDeltaFAutoParams():
    sources = [
        consenrich.core.inputSource(
            path="a.bam",
            sourceKind="BAM",
            role="treatment",
        ),
        consenrich.core.inputSource(
            path="b.fragments.tsv.gz",
            sourceKind="FRAGMENTS",
            role="treatment",
        ),
    ]

    deltaFCenter, autoDeltaF, deltaFLow, deltaFHigh = consenrich._resolveDeltaFAutoParams(
        deltaF=-1.0,
        intervalSizeBP=50,
        sources=sources,
        readLengths=[75, 120],
        characteristicLengths=[150, 0],
    )

    assert autoDeltaF is True
    assert deltaFCenter == pytest.approx(0.5 * 50.0 / 135.0)
    assert deltaFLow < deltaFCenter
    assert deltaFHigh > deltaFCenter

    deltaFFixed, autoDeltaFFixed, deltaFLowFixed, deltaFHighFixed = consenrich._resolveDeltaFAutoParams(
        deltaF=0.25,
        intervalSizeBP=50,
        sources=sources,
        readLengths=[75, 120],
        characteristicLengths=[150, 0],
    )

    assert autoDeltaFFixed is False
    assert deltaFFixed == pytest.approx(0.25)
    assert deltaFLowFixed == pytest.approx(0.25)
    assert deltaFHighFixed == pytest.approx(0.25)


def test_prioritizeLargestChromosomePlanMovesLargestFirst():
    chromosomePlans = [
        {"chromosome": "chr2", "numIntervals": 120},
        {"chromosome": "chr1", "numIntervals": 400},
        {"chromosome": "chr3", "numIntervals": 85},
    ]

    prioritized = consenrich._prioritizeLargestChromosomePlan(chromosomePlans)

    assert [plan["chromosome"] for plan in prioritized] == ["chr1", "chr2", "chr3"]
    assert [plan["numIntervals"] for plan in prioritized] == [400, 120, 85]
