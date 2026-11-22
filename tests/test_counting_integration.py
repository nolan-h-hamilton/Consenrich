import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import ccounts as counts
import numpy as np
import pandas as pd
import pytest

from consenrich import config, constants, core, countingAdapter, io, sc_runs, consenrich as runtime


@pytest.mark.parametrize("kind,method,paired,factor", [
    ("H5AD", None, False, None), ("H5AD", "RPKM", False, None),
    ("H5AD", "RPGC", False, 0.25), ("BAM", "RPGC", False, None),
    ("BAM", "RPKM", True, None), ("BAM", "RPKM", True, 0.25),
    ("BAM", "SF", False, None), ("BAM", "ReadsInRegions", True, None),
    ("FRAGMENTS", "ReadsInRegions", False, None),
])
def testCountingRouting(tmp_path, monkeypatch, kind, method, paired, factor):
    pysam = pytest.importorskip("pysam")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    sizes, empty, regions = (tmp_path / name for name in ("chrom.sizes", "empty.bed", "regions.bed"))
    sizes.write_text("chr1\t10007\nchr2\t500\n")
    empty.touch()
    regions.write_text("chr1\t500\t1500\nchr1\t1000\t2500\n")
    starts = 20 + np.arange(300) * 30
    treatmentCount = 3 if method == "SF" else 2
    sources, expectedTotals, expectedMass = [], {}, {}
    for index in range(treatmentCount + int(paired)):
        name = f"sample{index}"
        path = tmp_path / f"{name}.{kind.lower()}"
        if kind == "BAM":
            header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10007}]}
            with pysam.AlignmentFile(path, "wb", header=header) as stream:
                for position in starts:
                    record = pysam.AlignedSegment(stream.header)
                    record.query_name, record.query_sequence = str(position), "A" * 50
                    record.flag, record.reference_id, record.reference_start = 0, 0, int(position)
                    record.mapping_quality, record.cigar = 60, [(0, 50)]
                    stream.write(record)
            pysam.index(str(path))
        elif kind == "FRAGMENTS":
            path.write_text("".join(f"chr1\t{start}\t{start + 50}\tcell\t{index + 1}\n" for start in starts))
            compressed = str(path) + ".gz"
            pysam.tabix_compress(str(path), compressed)
            pysam.tabix_index(compressed, preset="bed")
            path = tmp_path / (path.name + ".gz")
        else:
            anndata = pytest.importorskip("anndata")
            bins = np.arange(0, 10007, 50)
            matrix = (np.arange(len(bins)) % 7 + index + 1).astype(np.int32)[None, :]
            variables = pd.DataFrame({"chrom": "chr1", "start": bins, "end": np.minimum(bins + 50, 10007)},
                                     index=[f"bin{i}" for i in range(len(bins))])
            anndata.AnnData(X=matrix, var=variables).write_h5ad(path)
            expectedTotals[name] = float(matrix.sum())
        sources.append({"path": str(path), "format": kind, "name": name,
                        "role": "control" if paired and index == treatmentCount else "treatment"})
        if kind == "H5AD":
            sources[-1]["matrixRows"] = "pseudobulks"
        if kind == "FRAGMENTS":
            sources[-1]["fragmentsUseReadSupport"] = True
        endpoints = np.r_[starts, starts + 49] if kind == "FRAGMENTS" else starts
        expectedMass[name] = float(((endpoints >= 500) & (endpoints < 2500)).sum())
        if kind == "FRAGMENTS":
            expectedMass[name] *= index + 1
    counting = {"centerMB": False}
    if method is not None:
        counting["normMethod"] = method
    if factor is not None:
        counting["scaleFactors"] = [factor]
    if method == "ReadsInRegions":
        counting.update(normRegionsFile=str(regions), normScaleTo=20000 if kind == "FRAGMENTS" else 10000)
    parsed = config.readConfig({
        "experimentName": "routing", "inputParams": {"samples": sources}, "countingParams": counting,
        "genomeParams": {"name": "hg38", "chromSizesFile": str(sizes), "chromosomes": ["chr1"],
                         "blacklistFile": str(empty), "sparseBedFile": str(empty), "effectiveGenomeSize": 10007,
                         "excludeForNorm": ["chr1"] if kind == "H5AD" and factor else []},
        "samParams": {"inferFragmentLength": False,
                      "defaultCountMode": "coverage" if method == "RPGC" else "conservedFractionalOverlap"},
        "observationParams": {"lengthScaleBP": 200}, "matchingParams": {"enabled": False},
        "uncertaintyCalibrationParams": {"enabled": False},
        "fitParams": {"ECM_outerIters": 2, "ECM_minOuterIters": 2},
        "outputParams": {"convertToBigWig": False, "writeRunSummary": False, "stateShrinkageEnabled": False,
                         "writeStateShrinkage": False, "plotOptimizationPath": False,
                         "plotPrecisionReweightingHistograms": False, "plotNullCalibrationDiagnostics": False},
    })
    snapshots = []
    originalRead = counts.Reader.readRegion

    def captureRead(reader, *args, **kwargs):
        block = originalRead(reader, *args, **kwargs)
        snapshots.append((block, block.values.copy(),
                          None if block.rawNoiseMass is None else block.rawNoiseMass.copy()))
        return block

    monkeypatch.setattr(counts.Reader, "readRegion", captureRead)
    args = SimpleNamespace(verbose=False, verbose2=False, config="routing", logFile=None,
                           verbosity="quiet", progress=False)
    result = runtime._runConfigured(parsed, args, sharedReferenceGrid={"chr1": (0, 10007)})
    assert result["method"] == (method or "CPM")
    for block, values, noise in snapshots:
        np.testing.assert_array_equal(block.values, values)
        if noise is not None:
            np.testing.assert_array_equal(block.rawNoiseMass, noise)
    if factor is not None:
        assert all(row["sourceFactor"] == factor and row["denominator"] is None for row in result["treatment"])
    if method == "ReadsInRegions":
        for row in result["treatment"] + (result["control"] if paired else []):
            mass = expectedMass[row["donor"]]
            assert row["method"] == method and row["denominator"]["regionEndpointMass"] == mass
            assert row["sourceFactor"] == pytest.approx(parsed["countingArgs"].normScaleTo / mass)
    elif kind == "H5AD" and factor is None:
        assert result["featureTerritory"]["usedChromosomes"] == ["chr1"]
        for row in result["treatment"]:
            total = expectedTotals[row["donor"]]
            assert row["denominator"]["eventMass"] == total
            assert row["sourceFactor"] == pytest.approx((1e9 if method == "RPKM" else 1e6) / total)
            assert row["divideByBinWidth"] is (method == "RPKM")
    elif paired and factor is not None:
        assert set(result["featureTerritory"]["missingChromosomesBySource"]) == {
            row["sampleID"] for row in result["control"]}
    elif method == "SF":
        assert result["referenceRegion"]["chromosome"] == "chr1"
    if kind == "H5AD" and factor is None:
        parsed["genomeArgs"] = parsed["genomeArgs"]._replace(chromosomes=["chr2"])
        with pytest.raises(ValueError, match="missing requested fitting chromosomes"):
            runtime._runConfigured(parsed, args)


@pytest.mark.parametrize("availableCPUs, slurmAllocation, configuredThreads, expectedThreads", [
    (64, None, None, 4),
    (64, "16", "auto", 2),
    (8, "64", None, 1),
    (2, None, 3, 3),
])
def testBamFragmentLengthRouting(
    tmp_path, monkeypatch, availableCPUs, slurmAllocation, configuredThreads, expectedThreads,
):
    monkeypatch.setattr(countingAdapter.os, "process_cpu_count", lambda: availableCPUs, raising=False)
    if slurmAllocation is None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", slurmAllocation)
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t1000\n")
    samples = []
    for role in ("treatment", "control"):
        for mode in ("reads", "fragments"):
            path = tmp_path / f"{role}-{mode}.bam"
            path.touch()
            samples.append({
                "path": str(path), "format": "BAM", "role": role, "bamInputMode": mode,
            })
    overrides = ["samParams.fragmentLengthOptions.maxLengthBP=500"]
    if configuredThreads is not None:
        overrides.append(f"samParams.samThreads={configuredThreads}")
    parsed = config.readConfig(config.applyConfigOverrides({
        "inputParams": {"samples": samples},
        "genomeParams": {"name": None, "chromSizesFile": str(sizes)},
        "samParams": {
            "inferFragmentLength": True,
            "fragmentLengthOptions": {"minLengthBP": 110, "maxLengthBP": 400},
        },
    }, overrides))

    indexed, inferred, extended = [], [], []

    def prepareIndex(path, kind, *, threads):
        indexed.append((path, kind, threads))

    def estimateFragmentLength(sampleID):
        inferred.append(sampleID)
        return 180

    def openSources(sources, grid, *, threads, fragmentLengthOptions):
        assert type(threads) is int and threads == expectedThreads
        assert fragmentLengthOptions == counts.FragmentLengthOptions(
            minLengthBP=110, maxLengthBP=500,
        )
        options = {source.sampleID: source.countOptions for source in sources}
        assert [sampleID for sampleID, opt in options.items() if opt.inferFragmentLength] == [
            "treatment0",
        ]
        return nullcontext(SimpleNamespace(
            getSourceInfo=lambda sampleID: {"countOptions": options[sampleID]},
            estimateFragmentLength=estimateFragmentLength,
            setReadExtension=lambda sampleID, length: extended.append((sampleID, length)),
        ))

    monkeypatch.setattr(counts, "prepareIndex", prepareIndex)
    monkeypatch.setattr(counts, "openSources", openSources)
    with countingAdapter.openReader(parsed):
        assert indexed == [(sample["path"], "BAM", expectedThreads) for sample in samples]
        assert inferred == ["treatment0"]
        assert extended == [("reference0", 180)]


def _lengthScaleConfig(tmp_path, monkeypatch, kind, binSizeBP):
    pysam = pytest.importorskip("pysam")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))
    sizes, empty = tmp_path / "chrom.sizes", tmp_path / "empty.bed"
    sizes.write_text("chr1\t10007\n")
    empty.touch()
    sources = []
    for index in range(2):
        path = tmp_path / f"sample{index}.{kind.lower()}"
        if kind == "BAM":
            header = {"HD": {"SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 10007}]}
            with pysam.AlignmentFile(path, "wb", header=header) as stream:
                for position in 20 + np.arange(300) * 30:
                    record = pysam.AlignedSegment(stream.header)
                    record.query_name, record.query_sequence = str(position), "A" * 50
                    record.flag, record.reference_id, record.reference_start = 0, 0, int(position)
                    record.mapping_quality, record.cigar = 60, [(0, 50)]
                    stream.write(record)
            pysam.index(str(path))
        else:
            anndata = pytest.importorskip("anndata")
            bins = np.arange(0, 10007, 50)
            matrix = (np.arange(len(bins)) % 7 + index + 1).astype(np.int32)[None, :]
            variables = pd.DataFrame({"chrom": "chr1", "start": bins, "end": np.minimum(bins + 50, 10007)},
                                     index=[f"bin{i}" for i in range(len(bins))])
            anndata.AnnData(X=matrix, var=variables).write_h5ad(path)
        sources.append({"path": str(path), "format": kind, "name": f"sample{index}"})
        if kind == "H5AD":
            sources[-1]["matrixRows"] = "pseudobulks"
    return config.readConfig({
        "experimentName": "lengthScaleBin", "inputParams": {"samples": sources},
        "countingParams": {"centerMB": False},
        "genomeParams": {"name": "hg38", "chromSizesFile": str(sizes), "chromosomes": ["chr1"],
                         "blacklistFile": str(empty), "sparseBedFile": str(empty), "effectiveGenomeSize": 10007},
        "samParams": {"inferFragmentLength": False}, "matchingParams": {"enabled": False},
        "observationParams": {"lengthScaleOptions": {"binSizeBP": binSizeBP}},
        "uncertaintyCalibrationParams": {"enabled": False},
        "fitParams": {"ECM_outerIters": 2, "ECM_minOuterIters": 2},
        "outputParams": {"convertToBigWig": False, "writeRunSummary": False, "stateShrinkageEnabled": False,
                         "writeStateShrinkage": False, "plotOptimizationPath": False,
                         "plotPrecisionReweightingHistograms": False, "plotNullCalibrationDiagnostics": False},
    })


@pytest.mark.parametrize("kind,configuredBinBP,passedBinBP,ignoredBinBP", [
    ("H5AD", None, None, None), ("H5AD", 25, 50, 25), ("H5AD", 50, 50, None),
    ("BAM", None, None, None), ("BAM", 25, 25, None),
])
def testLengthScaleBinSizePassedToCcounts(
    tmp_path, monkeypatch, kind, configuredBinBP, passedBinBP, ignoredBinBP,
):
    # ccounts chooses the bin size when none is passed
    parsed = _lengthScaleConfig(tmp_path, monkeypatch, kind, configuredBinBP)
    assert parsed["countingArgs"].intervalSizeBP == 50
    assert parsed["observationArgs"].lengthScaleOptions["binSizeBP"] == configuredBinBP
    passed = []

    def estimate(reader, *, sampleIDs, options, excludeRanges):
        passed.append((reader.grid.binSizeBP, options.binSizeBP))
        return {"status": "estimated", "scaleBP": 200.0}

    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    monkeypatch.setattr(counts.Reader, "estimateLengthScale", estimate)
    handler = Capture(level=logging.INFO)
    savedLevel = runtime.logger.level
    runtime.logger.addHandler(handler)
    runtime.logger.setLevel(logging.INFO)
    try:
        args = SimpleNamespace(verbose=False, verbose2=False, config="lengthScaleBin", logFile=None,
                               verbosity="quiet", progress=False)
        runtime._runConfigured(parsed, args, sharedReferenceGrid={"chr1": (0, 10007)})
    finally:
        runtime.logger.setLevel(savedLevel)
        runtime.logger.removeHandler(handler)
    assert passed == [(50, passedBinBP)]
    notices = [message for message in messages if message.startswith("Length scale bin")]
    if ignoredBinBP is None:
        assert not notices
    else:
        assert notices == [
            "Length scale bin: H5AD counts cannot be rebinned, so length scale estimation "
            "will use the bin size that was specified for the H5AD input and ignore "
            f"`lengthScaleOptions.binSizeBP={ignoredBinBP}`"
        ]


@pytest.mark.parametrize("kind,expectedBinBP", [("BAM", 150), ("H5AD", 50)])
def testLengthScaleDefaultBinSizeFromCcounts(tmp_path, monkeypatch, kind, expectedBinBP):
    # the default comes from ccounts, not from countingParams.intervalSizeBP
    parsed = _lengthScaleConfig(tmp_path, monkeypatch, kind, None)
    options = counts.LengthScaleOptions(**{
        **parsed["observationArgs"].lengthScaleOptions,
        "blockCount": 2, "blockWidthBP": 1000, "scaleEndBP": None,
    })
    with countingAdapter.openReader(parsed) as reader:
        assert reader.grid.binSizeBP == 50
        result = reader.estimateLengthScale(
            sampleIDs=["treatment0", "treatment1"], options=options, excludeRanges={},
        )
    assert result["settings"]["binSizeBP"] == expectedBinBP


def testGenomeResources(tmp_path):
    for assembly, sizes in counts.assemblyChromosomeSizes.items():
        args = config.getGenomeArgs({"genomeParams": {"name": assembly}})
        assert args.chromosomeSizes == sizes
        assert set(args.chromosomes) == set(counts.standardChromosomes[assembly])
        assert tuple(args.chromosomes) == args.normalizationChromosomes
        assert args.chromSizesFile is None

    sizesPath = tmp_path / "reference.sizes"
    sizesPath.write_text("chr1\t1000\n")
    for assembly in (None, "hs1", "dm3", "hg38"):
        args = config.getGenomeArgs({"genomeParams": {
            "name": assembly, "chromSizesFile": str(sizesPath), "effectiveGenomeSize": 900,
        }})
        assert args.chromosomeSizes == {"chr1": 1000}
        assert args.chromosomes == ["chr1"]
        assert args.normalizationChromosomes == ("chr1",)
        assert args.effectiveGenomeSize == 900


def testEffectiveGenomeTerritory(tmp_path):
    sizesPath = tmp_path / "subset.sizes"
    sizesPath.write_text("chrX\t156040895\nchrY\t57227415\n")
    args = config.getGenomeArgs({"genomeParams": {
        "name": "hg38", "chromSizesFile": str(sizesPath), "excludeForNorm": ["chrY"],
    }})
    fragments = [core.inputSource("one", sourceKind="FRAGMENTS")] * 2
    assert io.getEffectiveGenomeSizes(args, [50, 200], fragments) == [154893029] * 2
    bams = [core.inputSource("one", sourceKind="BAM")] * 2
    with pytest.raises(ValueError, match="effectiveGenomeSize"):
        io.getEffectiveGenomeSizes(args, [50, 200], bams)

    args = args._replace(effectiveGenomeSize=170001, genomeName=None)
    for sources in (fragments, bams):
        assert io.getEffectiveGenomeSizes(args, [0, 200], sources) == [170001] * 2
    for content, message in (("chrUnknown\t100\n", "effectiveGenomeSize"),
                             ("chrX\t100\n", "length differs")):
        sizesPath.write_text(content)
        with pytest.raises(ValueError, match=message):
            constants.getNonNGenomeSize("hg38", str(sizesPath))


@pytest.mark.parametrize("failFit,supplied", [(False, False), (False, True), (True, False)])
def testGroupedSelection(tmp_path, monkeypatch, failFit, supplied):
    anndata = pytest.importorskip("anndata")
    monkeypatch.chdir(tmp_path)
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t100\nchr2\t100\n")
    samples = []
    for index, groups in enumerate((["A"], ["A", "B"])):
        path = tmp_path / f"donor{index}.h5ad"
        obs = pd.DataFrame({"group": groups}, index=[f"cell{i}" for i in range(len(groups))])
        var = pd.DataFrame({"chrom": ["chr2" if index == 0 else "chr1"] * 2,
                            "start": [0, 50], "end": [50, 100]}, index=["bin0", "bin1"])
        anndata.AnnData(X=np.eye(len(groups), 2, dtype=np.int32), obs=obs, var=var).write_h5ad(path)
        samples.append({"name": f"donor{index}", "path": str(path), "format": "H5AD", "matrixRows": "cells",
                        "cellMetadata": {"path": str(path), "groupColumn": "group"}})
    parsed = config.readConfig({
        "experimentName": "groups", "inputParams": {"samples": samples},
        "genomeParams": {"name": None, "chromSizesFile": str(sizes), "chromosomes": ["chr1"]},
        "countingParams": {"scaleFactors": [0.25, 0.5] if supplied else None},
        "scParams": {"fits": [{"name": "groupB", "selectGroups": ["B"]}]},
    })
    args = SimpleNamespace(verbose=False, verbose2=False, logFile=None, config="groups",
                           verbosity="quiet", progress=False)
    called = []

    def fit(selected, args, grid, *, countsReader):
        called.append(selected["countTreatmentIDs"])
        assert called[-1] == ("treatment1",)
        assert selected["countingArgs"].scaleFactors == ([0.5] if supplied else None)
        block = countsReader.readRegion("chr1", 0, 100, sampleIDs=called[-1])
        assert block.values.sum() == 1
        if failFit:
            raise RuntimeError("fit failed")
        return {"method": "CPM"}

    with pytest.raises(RuntimeError, match="fit failed") if failFit else nullcontext():
        summary = sc_runs.executeFits(parsed, args, fit, lambda *args, **kwargs: None)[0]
    if not failFit:
        assert summary["retainedDonors"] == ["donor1"]
        assert summary["omittedDonors"] == [{"donor": "donor0", "reason": "empty cell selection"}]
    assert called == [("treatment1",)]
    assert Path.cwd() == tmp_path
