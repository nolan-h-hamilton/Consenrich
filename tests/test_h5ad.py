from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from consenrich import cellmetadata, config, core, cconsenrich, h5ad, io

anndata = pytest.importorskip("anndata")


def writeMatrix(path, counts, bins, observations=None):
    if observations is None:
        observations = pd.DataFrame(index=[f"row{i}" for i in range(counts.shape[0])])
    anndata.AnnData(X=counts, obs=observations, var=bins).write_h5ad(path)
    return str(path)


def genomeArgs(tmpPath, sizes="chr1\t1200\nchr2\t100\n", excluded=("chr2",)):
    path = tmpPath / "genome.sizes"
    path.write_text(sizes)
    return core.genomeParams(None, str(path), None, None, ["chr1"], [], list(excluded))


def testH5ADInputContract(tmp_path, caplog):
    genome = genomeArgs(tmp_path)
    bins = pd.DataFrame({"chrom": ["chr1"] * 3 + ["chr2"],
                         "start": [0, 500, 1000, 0], "end": [500, 1000, 1200, 100]},
                        index=["b0", "b1", "b2", "b3"])
    counting = config.getCountingArgs({"countingParams": {"intervalSizeBP": 500, "normMethod": "CPM"}})
    for storage in (np.asarray, sparse.csr_matrix, sparse.csc_matrix):
        path = writeMatrix(tmp_path / "counts.h5ad", storage([[1., 0., 3., 19.]]), bins)
        source = io._coerceInputSource({"path": path, "matrixRows": "pseudobulks"}, "treatment")
        assert source.sourceKind == "H5AD"
        assert h5ad.resolveIntervalSizeBP([source], genome) == 500
        prepared = h5ad.prepareSources([source], genome, counting)[0]
        np.testing.assert_array_equal(h5ad.readCounts(prepared, "chr1", 0, 1200, 500), [1, 0, 3])
        assert h5ad.getScaleFactor(prepared, "CPM", 500, 0) == pytest.approx(250000)
        assert h5ad.getScaleFactor(prepared, "RPKM", 500, 0) == pytest.approx(500000)
        for method in ("EGS", "RPGC"):
            with pytest.raises(ValueError):
                h5ad.getScaleFactor(prepared, method, 500, 10000)
    base = {"genomeParams": {"name": None, "chromSizesFile": genome.chromSizesFile,
                            "chromosomes": ["chr1"], "excludeChroms": []},
            "inputParams": {"samples": [{"path": path, "matrixRows": "pseudobulks"}]},
            "countingParams": {"normMethod": "CPM"}}
    assert config.readConfig(base)["countingArgs"].intervalSizeBP == 500
    for width in (500, 50, None):
        nested = deepcopy(base)
        nested["countingParams"]["intervalSizeBP"] = width
        dotted = deepcopy(base)
        dotted["countingParams.intervalSizeBP"] = width
        overridden = config.applyConfigOverrides(base, [f"countingParams.intervalSizeBP={width if width is not None else 'null'}"])
        for values in (nested, dotted, overridden):
            if width == 500:
                assert config.readConfig(values)["countingArgs"].intervalSizeBP == 500
            else:
                with pytest.raises(ValueError):
                    config.readConfig(values)
    for value in (-1., 0.5, np.nan, np.inf, float(np.finfo(np.float32).max) * 2):
        invalid = np.array([[1., 0., 3., 19.]])
        invalid[0, 0] = value
        writeMatrix(tmp_path / "counts.h5ad", invalid, bins)
        with pytest.raises(ValueError):
            h5ad.prepareSources([source], genome, counting)
    badGrids = [bins.drop(columns="chrom")]
    for column, row, value in (("start", 1, 600), ("start", 1, 0),
                               ("end", 0, 400), ("end", 2, 1201), ("chrom", 2, "absent")):
        changed = bins.copy()
        changed.loc[f"b{row}", column] = value
        badGrids.append(changed)
    badGrids.append(bins.iloc[[1, 0, 2, 3]])
    for invalidBins in badGrids:
        writeMatrix(tmp_path / "counts.h5ad", np.ones((1, len(invalidBins))), invalidBins)
        with pytest.raises(ValueError):
            h5ad.resolveIntervalSizeBP([source], genome)
    writeMatrix(tmp_path / "counts.h5ad", np.ones((1, 4)), bins)
    narrowBins = pd.DataFrame({"chrom": ["chr1"] * 4 + ["chr2"],
                               "start": [0, 300, 600, 900, 0], "end": [300, 600, 900, 1200, 100]},
                              index=[f"n{i}" for i in range(5)])
    narrowPath = writeMatrix(tmp_path / "narrow.h5ad", np.ones((1, 5)), narrowBins)
    with pytest.raises(ValueError):
        h5ad.resolveIntervalSizeBP([source, source._replace(path=narrowPath)], genome)
    singlePath = writeMatrix(tmp_path / "single.h5ad", np.ones((1, 1)), bins.iloc[[3]])
    single = source._replace(path=singlePath)
    with pytest.raises(ValueError):
        h5ad.resolveIntervalSizeBP([single], genome)
    assert h5ad.resolveIntervalSizeBP([single], genome, 500) == 500
    for fields in ({"matrixRows": "other"}, {"countMode": "center"}, {"matrixRows": None}):
        with pytest.raises(ValueError):
            io._coerceInputSource({"path": path, "matrixRows": "cells", **fields}, "treatment")
    for field in ("barcodeAllowListFile", "barcodeGroupMapFile"):
        ignored = io._coerceInputSource({"path": path, "matrixRows": "cells", field: "absent.txt"}, "treatment")
        assert getattr(ignored, field) is None
        assert "Ignoring barcodeAllowListFile and barcodeGroupMapFile" in caplog.text


def testH5ADFragmentParity(tmp_path):
    pysam = pytest.importorskip("pysam")
    genome = genomeArgs(tmp_path, "chr1\t210\n", ())
    bins = pd.DataFrame({"chrom": ["chr1"] * 5, "start": [0, 50, 100, 150, 200],
                         "end": [50, 100, 150, 200, 210]}, index=[f"b{i}" for i in range(5)])
    expected = np.array([[1, 2, 0, 1, 1]], dtype=np.float32)
    path = writeMatrix(tmp_path / "bulk.h5ad", sparse.csr_matrix(expected), bins)
    matrixSource = core.inputSource(path, sourceKind="H5AD", matrixRows="pseudobulks")
    counting = config.getCountingArgs({"countingParams": {"intervalSizeBP": 50, "normMethod": "CPM"}})
    matrixSource = h5ad.prepareSources([matrixSource], genome, counting)[0]
    plain = tmp_path / "small.fragments.tsv"
    plain.write_text("chr1\t0\t20\tA\t1\nchr1\t48\t72\tB\t1\n"
                     "chr1\t60\t80\tA\t1\nchr1\t150\t160\tA\t1\nchr1\t200\t210\tB\t1\n")
    fragments = str(plain) + ".gz"
    pysam.tabix_compress(str(plain), fragments, force=True)
    pysam.tabix_index(fragments, preset="bed", force=True)
    fragmentSource = core.inputSource(fragments, sourceKind="FRAGMENTS", countMode="center")
    options = dict(chromosome="chr1", start=0, end=210, intervalSizeBP=50, readLengths=[0],
                   scaleFactors=[2.5], oneReadPerBin=0, samThreads=1, samFlagExclude=0,
                   returnRawNoiseMass=True)
    matrixResult = core.readSegments([matrixSource], **options)
    fragmentResult = core.readSegments([fragmentSource], **options)
    for result in (matrixResult, fragmentResult):
        np.testing.assert_array_equal(result.counts, expected * 2.5)
        np.testing.assert_array_equal(result.rawNoiseMass, expected)
    for method in ("log", "sqrt"):
        np.testing.assert_array_equal(cconsenrich.cTransform(matrixResult.counts, mode=method),
                                      cconsenrich.cTransform(fragmentResult.counts, mode=method))
        matrixFloor = core.transformCountVarianceFloor(matrixResult.counts, [2.5],
            rawNoiseMass=matrixResult.rawNoiseMass, transformMethod=method)
        fragmentFloor = core.transformCountVarianceFloor(fragmentResult.counts, [2.5],
            rawNoiseMass=fragmentResult.rawNoiseMass, transformMethod=method)
        np.testing.assert_array_equal(matrixFloor, fragmentFloor)


def testCellToPseudobulkCounts(tmp_path):
    genome = genomeArgs(tmp_path, "chr1\t200\n", ())
    bins = pd.DataFrame({"chrom": ["chr1"] * 4, "start": [0, 50, 100, 150],
                         "end": [50, 100, 150, 200]}, index=[f"b{i}" for i in range(4)])
    counts = np.array([[1, 0, 3, 0], [2, 1, 0, 0], [9, 9, 9, 9],
                       [4, 0, 1, 0], [0, 5, 2, 0], [8, 8, 8, 8]])
    observations = pd.DataFrame({"barcode": ["A", "B", "X", "A", "B", "X"],
                                 "sample": ["P1"] * 3 + ["P2"] * 3,
                                 "group": ["G", "G", "H"] * 2},
                                index=[f"c{i}" for i in range(6)])
    order = [4, 2, 0, 5, 1, 3]
    path = writeMatrix(tmp_path / "cells.h5ad", sparse.csr_matrix(counts[order]),
                       bins, observations.iloc[order])
    metadata = {"path": path, "barcodeColumn": "barcode", "sampleColumn": "sample", "groupColumn": "group"}
    sources = [core.inputSource(path, sourceKind="H5AD", matrixRows="cells", sampleName=name)
               for name in ("P1", "P2")]
    selected = cellmetadata.MetadataSelections().prepareSources(sources, metadata, ["G"], grouped=True)
    counting = config.getCountingArgs({"countingParams": {"intervalSizeBP": 50, "normMethod": "CPM"}})
    prepared = h5ad.prepareSources(selected, genome, counting)
    expected = np.array([[3, 1, 3, 0], [4, 5, 3, 0]], dtype=np.float32)
    for index, source in enumerate(prepared):
        assert source.selectedCellCount == 2
        np.testing.assert_array_equal(h5ad.readCounts(source, "chr1", 0, 200, 50), expected[index])
        assert h5ad.getScaleFactor(source, "CPM", 50, 0) == pytest.approx(1e6 / expected[index].sum(), rel=1e-9)
    bulkObs = pd.DataFrame({"sample": ["P1", "P2"], "group": ["G", "G"]}, index=["p1", "p2"])
    bulkPath = writeMatrix(tmp_path / "bulk.h5ad", expected, bins, bulkObs)
    bulkSources = [source._replace(path=bulkPath, matrixRows="pseudobulks") for source in sources]
    bulkMetadata = dict(metadata, path=bulkPath, barcodeColumn=None)
    bulkSelected = cellmetadata.MetadataSelections().prepareSources(bulkSources, bulkMetadata, ["G"], grouped=True)
    bulkPrepared = h5ad.prepareSources(bulkSelected, genome, counting)
    options = dict(chromosome="chr1", start=0, end=200, intervalSizeBP=50, readLengths=[0, 0],
                   scaleFactors=[2., .5], oneReadPerBin=0, samThreads=1, samFlagExclude=0,
                   returnRawNoiseMass=True)
    for sourceList in (prepared, bulkPrepared):
        result = core.readSegments(sourceList, **options)
        np.testing.assert_array_equal(result.counts, expected * np.array([[2.], [.5]]))
        np.testing.assert_array_equal(result.rawNoiseMass, expected)
    assert all(source.selectedCellCount is None for source in bulkPrepared)
    duplicatePath = writeMatrix(tmp_path / "duplicate.h5ad", expected, bins)
    with pytest.raises(ValueError):
        h5ad.prepareSources([sources[0]._replace(path=duplicatePath, matrixRows="pseudobulks")], genome, counting)
