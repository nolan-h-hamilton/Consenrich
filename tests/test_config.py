import textwrap
import pytest
from consenrich.consenrich import readConfig
from pathlib import Path
import consenrich.constants as constants
import consenrich.misc_util as misc_util


def writeConfigFile(tmpPath, fileName, yamlText):
    filePath = tmpPath / fileName
    filePath.write_text(
        textwrap.dedent(yamlText).strip() + "\n", encoding="utf-8"
    )
    return filePath


def setupGenomeFiles(
    tmpPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    chromSizesPath = tmpPath / "testGenome.sizes"
    chromSizesPath.write_text("chrTest\t100000\n", encoding="utf-8")

    def fakeResolveGenomeName(genomeName: str) -> str:
        return genomeName

    def fakeGetGenomeResourceFile(
        genomeLabel: str, resourceName: str
    ) -> str:
        if resourceName == "sizes":
            return str(chromSizesPath)
        return str(tmpPath / f"{genomeLabel}.{resourceName}.bed")

    monkeypatch.setattr(
        constants, "resolveGenomeName", fakeResolveGenomeName
    )
    monkeypatch.setattr(
        constants, "getGenomeResourceFile", fakeGetGenomeResourceFile
    )


def setupBamHelpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fakeCheckBamFile(bamPath: str) -> None:
        return None

    def fakeBamsArePairedEnd(bamList: list) -> list:
        return [False] * len(bamList)

    monkeypatch.setattr(misc_util, "checkBamFile", fakeCheckBamFile)
    monkeypatch.setattr(
        misc_util, "bamsArePairedEnd", fakeBamsArePairedEnd
    )


def test_ensureInput():
    configYaml = f"""
    experimentName: testExperiment
    genomeParams.name: hg38
    """

    configPath = writeConfigFile(
        Path("."), "config_no_input.yaml", configYaml
    )
    try:
        readConfig(str(configPath))
    except ValueError as e:
        return
    else:
        assert False, (
            "Expected ValueError not raised given empty `consenrich.core.inputParams`"
        )


def test_readConfigDottedAndNestedEquivalent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)

    dottedYaml = """
    experimentName: testExperiment
    inputParams.bamFiles: [smallTest.bam, smallTest2.bam]
    genomeParams.name: testGenome
    genomeParams.excludeChroms: [chrM]
    countingParams.stepSize: 50
    countingParams.applyAsinh: true
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
      stepSize: 50
      applyAsinh: true
    """

    dottedPath = writeConfigFile(
        tmp_path, "config_dotted.yaml", dottedYaml
    )
    nestedPath = writeConfigFile(
        tmp_path, "config_nested.yaml", nestedYaml
    )

    configDotted = readConfig(str(dottedPath))
    configNested = readConfig(str(nestedPath))

    assert configDotted["experimentName"] == "testExperiment"
    assert configNested["experimentName"] == "testExperiment"
    assert (
        configDotted["experimentName"]
        == configNested["experimentName"]
    )

    inputDotted = configDotted["inputArgs"]
    inputNested = configNested["inputArgs"]

    assert inputDotted.bamFiles == ["smallTest.bam", "smallTest2.bam"]
    assert inputNested.bamFiles == ["smallTest.bam", "smallTest2.bam"]
    assert inputDotted.bamFiles == inputNested.bamFiles

    assert bool(inputDotted.bamFilesControl) is False
    assert bool(inputNested.bamFilesControl) is False
    assert inputDotted.pairedEnd == inputNested.pairedEnd

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

    assert countingDotted.stepSize == 50
    assert countingNested.stepSize == 50
    assert countingDotted.stepSize == countingNested.stepSize

    assert countingDotted.applyAsinh is True
    assert countingNested.applyAsinh is True
    assert countingDotted.applyAsinh == countingNested.applyAsinh

    assert countingDotted.applyLog is False
    assert countingNested.applyLog is False

    observationDotted = configDotted["observationArgs"]
    observationNested = configNested["observationArgs"]
    processDotted = configDotted["processArgs"]
    processNested = configNested["processArgs"]

    assert type(observationDotted) is type(observationNested)
    assert type(processDotted) is type(processNested)
    assert observationDotted.minR == observationNested.minR

    samDotted = configDotted["samArgs"]
    samNested = configNested["samArgs"]
    detrendDotted = configDotted["detrendArgs"]
    detrendNested = configNested["detrendArgs"]
    matchingDotted = configDotted["matchingArgs"]
    matchingNested = configNested["matchingArgs"]

    assert type(samDotted) is type(samNested)
    assert type(detrendDotted) is type(detrendNested)
    assert type(matchingDotted) is type(matchingNested)

    assert samDotted.samThreads == samNested.samThreads
    assert (
        detrendDotted.detrendWindowLengthBP
        == detrendNested.detrendWindowLengthBP
    )
    assert (
        matchingDotted.templateNames == matchingNested.templateNames
    )
