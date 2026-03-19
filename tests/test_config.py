import textwrap
import pytest
from consenrich.consenrich import readConfig
from pathlib import Path
import consenrich.consenrich as consenrich
import consenrich.constants as constants
import consenrich.misc_util as misc_util
import types
import gzip


class stopAfterResolve(Exception):
    pass


class stopOnFirstControlEstimate(Exception):
    pass


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
    monkeypatch.setattr(misc_util, "checkBamFile", fakeCheckAlignmentFile)
    monkeypatch.setattr(misc_util, "bamsArePairedEnd", fakeAlignmentFilesArePairedEnd)


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
    assert matchingDotted.templateNames == matchingNested.templateNames


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
          fragmentPositionsAreOffset: false
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
    assert inputArgs.treatmentSources[1].fragmentPositionsAreOffset is False
    assert configParsed["countingArgs"].normMethod == "CPM"
    assert configParsed["countingArgs"].fragmentsGroupNorm == "CELLS"


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
    scParams.fragmentPositionsAreOffset: false
    scParams.barcodeTag: CR
    """

    configPath = writeConfigFile(tmp_path, "config_sc_defaults.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    source = configParsed["inputArgs"].treatmentSources[0]

    assert source.countMode is None
    assert source.fragmentPositionsAreOffset is False
    assert source.barcodeTag == "CR"
    assert configParsed["scArgs"].defaultCountMode == "center"
    assert configParsed["scArgs"].fragmentsGroupNorm == "CELLS"
    assert configParsed["countingArgs"].fragmentsGroupNorm == "CELLS"


def test_readConfigAcceptsClusterIdAlias(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    setupGenomeFiles(tmp_path, monkeypatch)
    setupBamHelpers(monkeypatch)
    fragmentsPath = tmp_path / "smallTest.fragments.tsv.gz"
    fragmentsPath.write_text("", encoding="utf-8")
    groupMapPath = tmp_path / "groups.tsv"
    groupMapPath.write_text("BC_A\tclusterA\n", encoding="utf-8")

    configYaml = f"""
    experimentName: sampleExperiment
    inputParams:
      samples:
        - path: {fragmentsPath}
          format: fragments
          role: treatment
          barcodeGroupMapFile: {groupMapPath}
          clusterId: clusterA
    genomeParams.name: testGenome
    """

    configPath = writeConfigFile(tmp_path, "config_cluster_id.yaml", configYaml)
    configParsed = readConfig(str(configPath))
    source = configParsed["inputArgs"].treatmentSources[0]

    assert source.selectGroups == ["clusterA"]


def test_readConfigMatchingDefaultsToGlobalEmpiricalNull(
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

    assert configParsed["matchingArgs"].useSplitEmpiricalNull is False


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
        fragmentLengths=[150, 0],
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
        fragmentLengths=[150, 0],
    )

    assert autoDeltaFFixed is False
    assert deltaFFixed == pytest.approx(0.25)
    assert deltaFLowFixed == pytest.approx(0.25)
    assert deltaFHighFixed == pytest.approx(0.25)


def makeFakeArgs(config="dummy.yaml", verbose=False, verbose2=False):
    return types.SimpleNamespace(
        config=config,
        matchBedGraph=None,
        matchTemplate=None,
        matchLevel=None,
        matchAlpha=0.05,
        matchMinMatchLengthBP=-1,
        matchIters=50000,
        matchMinSignalAtMaxima="q:0.75",
        matchMaxNumMatches=1000000,
        matchMergeGapBP=-1,
        matchUseWavelet=False,
        matchRandSeed=42,
        matchExcludeBed=None,
        matchAutoLengthQuantile=0.90,
        matchMethodFDR=None,
        matchIsLogScale=False,
        verbose=verbose,
        verbose2=verbose2,
    )


def makeFakeConfig(useTreatmentFragmentLengths: bool):
    inputArgs = types.SimpleNamespace(
        bamFiles=["t1.bam", "t2.bam"],
        bamFilesControl=["c1.bam", "c2.bam"],
        pairedEnd=False,
    )

    genomeArgs = types.SimpleNamespace(
        genomeName="testGenome",
        chromSizesFile="fake.chrom.sizes",
        excludeChroms=[],
        excludeForNorm=[],
        sparseBedFile=None,
        chromosomes=["chrTest"],
    )

    countingArgs = types.SimpleNamespace(
        intervalSizeBP=25,
        scaleDown=False,
        scaleFactors=None,
        scaleFactorsControl=None,
        numReads=100,
        applyAsinh=False,
        applyLog=False,
        applySqrt=False,
        rescaleToTreatmentCoverage=False,
        normMethod="EGS",
        noTransform=False,
        trimLeftTail=0.0,
        fragmentLengths=None,
        fragmentLengthsControl=None,
        useTreatmentFragmentLengths=useTreatmentFragmentLengths,
        fragmentsGroupNorm="NONE",
    )

    processArgs = types.SimpleNamespace(
        deltaF=-1.0,
        minQ=-1.0,
        maxQ=10000,
        offDiagQ=1.0e-3,
        dStatAlpha=2.0,
        dStatd=1.0,
        dStatPC=1.0,
        dStatUseMean=False,
        scaleResidualsByP11=True,
    )

    observationArgs = types.SimpleNamespace(
        numNearest=25,
        minR=-1.0,
        maxR=10000,
        useALV=True,
        useConstantNoiseLevel=False,
        noGlobal=False,
        localWeight=0.333,
        globalWeight=0.667,
        approximationWindowLengthBP=25000,
        lowPassWindowLengthBP=50000,
        lowPassFilterType="median",
        returnCenter=True,
        shrinkOffset=1 - 0.05,
        kappaALV=50.0,
    )

    stateArgs = types.SimpleNamespace(
        stateInit=0.0,
        stateCovarInit=1000.0,
        boundState=True,
        stateLowerBound=0.0,
        stateUpperBound=10000.0,
    )

    samArgs = types.SimpleNamespace(
        samThreads=1,
        samFlagExclude=3844,
        maxInsertSize=1000,
        oneReadPerBin=0,
        chunkSize=1_000_000,
        offsetStr="0,0",
        pairedEndMode=0,
        inferFragmentLength=1,
        countEndsOnly=False,
        minMappingQuality=0,
        minTemplateLength=-1,
    )

    matchingArgs = types.SimpleNamespace(
        templateNames=[],
        cascadeLevels=[],
        minMatchLengthBP=-1,
        alpha=0.05,
        iters=25000,
        maxNumMatches=100000,
        minSignalAtMaxima="q:0.75",
        merge=True,
        mergeGapBP=-1,
        useScalingFunction=True,
        excludeRegionsBedFile=None,
        randSeed=42,
        penalizeBy=None,
        eps=1.0e-2,
        autoLengthQuantile=0.90,
        methodFDR=None,
    )

    plotArgs = types.SimpleNamespace(
        plotStateEstimatesHistogram=False,
        plotResidualsHistogram=False,
        plotStateStdHistogram=False,
        plotPrefix="test",
        plotDirectory=".",
    )

    outputArgs = types.SimpleNamespace(
        convertToBigWig=False,
        roundDigits=3,
        writeResiduals=False,
        writeMuncTrace=False,
        writeStateStd=False,
    )

    return {
        "experimentName": "testExperiment",
        "genomeArgs": genomeArgs,
        "inputArgs": inputArgs,
        "outputArgs": outputArgs,
        "countingArgs": countingArgs,
        "processArgs": processArgs,
        "observationArgs": observationArgs,
        "stateArgs": stateArgs,
        "samArgs": samArgs,
        "matchingArgs": matchingArgs,
        "plotArgs": plotArgs,
    }


def patchMainEntry(monkeypatch: pytest.MonkeyPatch, useTreatmentFragmentLengths: bool):
    monkeypatch.setattr(
        consenrich.argparse.ArgumentParser,
        "parse_args",
        lambda self: makeFakeArgs(),
    )
    monkeypatch.setattr(consenrich.os.path, "exists", lambda p: True)
    monkeypatch.setattr(
        consenrich,
        "readConfig",
        lambda p: makeFakeConfig(useTreatmentFragmentLengths),
    )
    monkeypatch.setattr(consenrich, "getReadLengths", lambda *a, **k: [50, 50])
    monkeypatch.setattr(
        consenrich,
        "getEffectiveGenomeSizes",
        lambda *a, **k: [1000, 1000],
    )
    monkeypatch.setattr(consenrich.core, "getReadLength", lambda *a, **k: 50)
    monkeypatch.setattr(
        consenrich.constants,
        "getEffectiveGenomeSize",
        lambda *a, **k: 1000,
    )
    monkeypatch.setattr(consenrich, "checkMatchingEnabled", lambda *a, **k: False)
