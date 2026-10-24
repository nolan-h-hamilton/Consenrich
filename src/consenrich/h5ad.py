from __future__ import annotations

import operator
import logging

import numpy as np
from scipy import sparse

from . import constants

logger = logging.getLogger(__name__)


def _openFile(path):
    try:
        import h5py
    except ImportError as error:
        raise ImportError("H5AD inputs require `consenrich[singlecell]`.") from error
    return h5py.File(path, "r")


def _readGrid(fileHandle, chromSizes, intervalSizeBP=None):
    try:
        from anndata.io import read_elem
    except ImportError as error:
        raise ImportError("H5AD inputs require `consenrich[singlecell]`.") from error
    if "var" not in fileHandle:
        raise ValueError("H5AD input requires `.var` genomic bins.")
    variables = fileHandle["var"]
    if not {"chrom", "start", "end"}.issubset(variables):
        raise ValueError("H5AD `.var` requires nonempty chrom, start, and end columns.")
    starts = np.asarray(read_elem(variables["start"]))
    ends = np.asarray(read_elem(variables["end"]))
    chromosomes = np.asarray(read_elem(variables["chrom"]))
    if starts.ndim != 1 or ends.shape != starts.shape or chromosomes.shape != starts.shape or starts.size == 0:
        raise ValueError("H5AD genomic columns must have equal nonzero lengths.")
    for coordinates in (starts, ends):
        if coordinates.dtype.kind not in "iu" or np.any(coordinates < 0):
            raise ValueError("H5AD bin coordinates must be nonnegative integers.")
    if np.any(ends <= starts):
        raise ValueError("H5AD bins require end > start.")
    slices = {}
    widths = set()
    terminalWidths = []
    for chromosome in dict.fromkeys(chromosomes):
        if not isinstance(chromosome, str) or chromosome not in chromSizes:
            raise ValueError(f"H5AD chromosome absent from chromosome sizes: {chromosome!r}")
        indices = np.flatnonzero(chromosomes == chromosome)
        left, right = int(indices[0]), int(indices[-1]) + 1
        if right - left != len(indices):
            raise ValueError(f"H5AD bins for {chromosome} must be contiguous in `.var`.")
        chromStarts, chromEnds = starts[left:right], ends[left:right]
        if chromStarts[0] != 0 or chromEnds[-1] != int(chromSizes[chromosome]):
            raise ValueError(f"H5AD bins must cover the complete chromosome {chromosome}.")
        if np.any(chromStarts[1:] != chromEnds[:-1]):
            raise ValueError(f"H5AD bins must be ordered, contiguous, and nonoverlapping: {chromosome}")
        widths.update(int(value) for value in np.unique(chromEnds[:-1] - chromStarts[:-1]))
        terminalWidths.append(int(chromEnds[-1] - chromStarts[-1]))
        slices[chromosome] = (left, right)
    if len(widths) > 1:
        raise ValueError("H5AD bins must have one regular interval width.")
    inferredWidth = next(iter(widths), None)
    if intervalSizeBP is not None:
        if inferredWidth is not None and inferredWidth != intervalSizeBP:
            raise ValueError(
                f"H5AD bin width conflict for {fileHandle.filename}: "
                f"inferred {inferredWidth}, requested {intervalSizeBP}."
            )
        if any(width > intervalSizeBP for width in terminalWidths):
            raise ValueError(
                f"H5AD terminal bin exceeds requested width {intervalSizeBP} "
                f"for {fileHandle.filename}."
            )
    return {
        "intervalSizeBP": inferredWidth,
        "chromosomeSlices": slices,
        "chromosomeLengths": {chromosome: int(chromSizes[chromosome]) for chromosome in slices},
        "terminalWidths": terminalWidths,
        "varCount": len(starts),
    }


def _readMatrix(fileHandle):
    if "X" not in fileHandle:
        raise ValueError("H5AD input requires raw counts in `.X`.")
    node = fileHandle["X"]
    if hasattr(node, "shape"):
        matrix = node
    else:
        try:
            from anndata.io import sparse_dataset
        except ImportError as error:
            raise ImportError("H5AD inputs require `consenrich[singlecell]`.") from error
        if node.attrs.get("encoding-type") not in {"csr_matrix", "csc_matrix"}:
            raise ValueError("H5AD `.X` must be dense, CSR, or CSC.")
        matrix = sparse_dataset(node)
    if len(matrix.shape) != 2 or matrix.dtype.kind not in "iuf":
        raise ValueError("H5AD `.X` must be a numeric, two-dimensional count matrix.")
    return matrix


def _selectedBlocks(matrix, rowIndices, left, right):
    if getattr(matrix, "format", None) == "csc":
        for colStart in range(left, right, 4096):
            colEnd = min(colStart + 4096, right)
            yield colStart - left, matrix[:, colStart:colEnd][rowIndices, :]
        return
    for rowStart in range(0, matrix.shape[0], 1024):
        rowEnd = min(rowStart + 1024, matrix.shape[0])
        selectedRows = rowIndices[(rowIndices >= rowStart) & (rowIndices < rowEnd)] - rowStart
        if selectedRows.size == 0:
            continue
        if getattr(matrix, "format", None) == "csr":
            yield 0, matrix[rowStart:rowEnd, :][selectedRows, left:right]
            continue
        for colStart in range(left, right, 4096):
            colEnd = min(colStart + 4096, right)
            yield colStart - left, matrix[rowStart:rowEnd, colStart:colEnd][selectedRows, :]


def _sumRows(matrix, rowIndices, left, right):
    result = np.zeros(right - left, dtype=np.float64)
    if rowIndices.size == 0:
        return result
    for offset, block in _selectedBlocks(matrix, rowIndices, left, right):
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError("H5AD `.X` requires finite, nonnegative integer raw counts.")
        result[offset:offset + block.shape[1]] += np.asarray(
            block.sum(axis=0, dtype=np.float64)
        ).reshape(-1)
    if np.any(result > np.finfo(np.float32).max):
        raise ValueError("H5AD aggregate counts exceed float32 capacity.")
    return result


def resolveIntervalSizeBP(sources, genomeArgs, requestedIntervalSizeBP=None):
    paths = dict.fromkeys(source.path for source in sources if source.sourceKind == "H5AD")
    if not paths:
        return requestedIntervalSizeBP if requestedIntervalSizeBP is not None else constants.COUNTING_DEFAULT_INTERVAL_SIZE_BP
    if requestedIntervalSizeBP is not None:
        if isinstance(requestedIntervalSizeBP, (bool, np.bool_)):
            raise ValueError("countingParams.intervalSizeBP must be a positive integer.")
        try:
            requestedIntervalSizeBP = operator.index(requestedIntervalSizeBP)
        except TypeError as error:
            raise ValueError("countingParams.intervalSizeBP must be a positive integer.") from error
        if requestedIntervalSizeBP <= 0:
            raise ValueError("countingParams.intervalSizeBP must be a positive integer.")
    from .misc_util import getChromSizesDict

    chromSizes = getChromSizesDict(genomeArgs.chromSizesFile)
    grids = []
    for path in paths:
        with _openFile(path) as fileHandle:
            grids.append(_readGrid(fileHandle, chromSizes, requestedIntervalSizeBP))
    widths = {grid["intervalSizeBP"] for grid in grids if grid["intervalSizeBP"] is not None}
    if len(widths) > 1:
        descriptions = ", ".join(
            f"{path}: {grid['intervalSizeBP']}" for path, grid in zip(paths, grids)
        )
        raise ValueError(f"H5AD inputs have conflicting bin widths: {descriptions}")
    width = requestedIntervalSizeBP if requestedIntervalSizeBP is not None else next(iter(widths), None)
    if width is None:
        raise ValueError("Cannot infer H5AD bin width from single terminal bins. Set countingParams.intervalSizeBP.")
    for path, grid in zip(paths, grids):
        if any(terminal > width for terminal in grid["terminalWidths"]):
            raise ValueError(f"H5AD terminal bin for {path} exceeds resolved width {width}.")
    return width


def prepareSources(sources, genomeArgs, countingArgs):
    if not any(source.sourceKind == "H5AD" for source in sources):
        return list(sources)
    try:
        from anndata.io import read_elem
    except ImportError as error:
        raise ImportError("H5AD inputs require `consenrich[singlecell]`.") from error
    from .misc_util import getChromSizesDict

    chromSizes = getChromSizesDict(genomeArgs.chromSizesFile)
    prepared = []
    for source in sources:
        if source.sourceKind != "H5AD":
            prepared.append(source)
            continue
        if source.matrixRows not in ("cells", "pseudobulks"):
            raise ValueError("H5AD input requires matrixRows: cells or pseudobulks.")
        if source.countMode is not None or source.bamInputMode is not None:
            raise ValueError("H5AD raw counts do not accept countMode or bamInputMode.")
        if source.barcodeAllowListFile is not None or source.barcodeGroupMapFile is not None:
            logger.warning("Ignoring barcodeAllowListFile and barcodeGroupMapFile for H5AD input %s", source.path)
            source = source._replace(barcodeAllowListFile=None, barcodeGroupMapFile=None)
        if source.selectGroups is not None and source.cellMetadata is None:
            raise ValueError("H5AD group selection requires cellMetadata.")
        with _openFile(source.path) as fileHandle:
            grid = _readGrid(fileHandle, chromSizes, countingArgs.intervalSizeBP)
            missing = set(genomeArgs.chromosomes) - grid["chromosomeSlices"].keys()
            if missing:
                raise ValueError(f"Requested chromosomes absent from H5AD input: {sorted(missing)}")
            if "obs" not in fileHandle:
                raise ValueError("H5AD input requires `.obs`.")
            frame = read_elem(fileHandle["obs"])
            spec = source.cellMetadata or {}
            barcodeColumn, sampleColumn = spec.get("barcodeColumn"), spec.get("sampleColumn")
            for column in (barcodeColumn, sampleColumn):
                if column is not None and column not in frame.columns:
                    raise ValueError(f"H5AD `.obs` lacks selection column {column!r}.")
            barcodes = frame.index if barcodeColumn is None else frame[barcodeColumn]
            selected = np.ones(len(frame), dtype=bool)
            if sampleColumn is not None:
                sample = source.metadataSample if source.metadataSample is not None else source.sampleName
                selected &= np.asarray(frame[sampleColumn] == sample)
            if source.selectedBarcodes is not None:
                selected &= np.asarray(barcodes.isin(source.selectedBarcodes))
            rowIndices = np.flatnonzero(selected)
            if source.matrixRows == "pseudobulks" and len(rowIndices) > 1:
                raise ValueError("H5AD pseudobulks require exactly one selected row per source.")
            matrix = _readMatrix(fileHandle)
            if matrix.shape != (len(frame), grid["varCount"]):
                raise ValueError("H5AD `.X` shape must match `.obs` and `.var`.")
            totals = {
                chromosome: float(np.sum(_sumRows(matrix, rowIndices, left, right), dtype=np.float64))
                for chromosome, (left, right) in grid["chromosomeSlices"].items()
            }
            info = {
                **grid,
                "intervalSizeBP": int(countingArgs.intervalSizeBP),
                "rowIndices": rowIndices,
                "rowCount": len(rowIndices),
                "shape": tuple(matrix.shape),
                "totalCounts": sum(totals.values()),
                "normalizationCount": sum(value for chromosome, value in totals.items() if chromosome not in (genomeArgs.excludeForNorm or ())),
            }
        logger.info(
            "H5AD normalization source=%s rows=%d chromosomes=%s excluded=%s totalBinCounts=%.12g",
            source.sampleName or source.path, len(rowIndices),
            ",".join(chromosome for chromosome in totals if chromosome not in (genomeArgs.excludeForNorm or ())),
            ",".join(genomeArgs.excludeForNorm or ()), info["normalizationCount"],
        )
        prepared.append(source._replace(
            matrixInfo=info,
            selectedCellCount=len(rowIndices) if source.matrixRows == "cells" else None,
        ))
    return prepared


def readCounts(source, chromosome, start, end, intervalSizeBP):
    info = source.matrixInfo
    if info is None:
        raise ValueError("H5AD source has not been prepared.")
    if intervalSizeBP != info["intervalSizeBP"]:
        raise ValueError("H5AD bin width conflicts with requested intervalSizeBP.")
    if chromosome not in info["chromosomeSlices"]:
        raise ValueError(f"Requested chromosome absent from H5AD input: {chromosome}")
    chromLength = info["chromosomeLengths"][chromosome]
    if start < 0 or start % intervalSizeBP or end <= start or end > chromLength or (end != chromLength and end % intervalSizeBP):
        raise ValueError("H5AD requested range must follow the stored bin grid.")
    left, _ = info["chromosomeSlices"][chromosome]
    colStart = left + start // intervalSizeBP
    colEnd = left + (end + intervalSizeBP - 1) // intervalSizeBP
    with _openFile(source.path) as fileHandle:
        matrix = _readMatrix(fileHandle)
        if tuple(matrix.shape) != info["shape"]:
            raise ValueError("H5AD `.X` shape differs from the prepared source.")
        counts = _sumRows(matrix, info["rowIndices"], colStart, colEnd)
    return counts.astype(np.float32)


def getScaleFactor(source, normMethod, intervalSizeBP, effectiveGenomeSize=None):
    if normMethod not in {"CPM", "RPKM"}:
        raise ValueError("H5AD automatic normalization supports CPM, RPKM, or SF. Supply scaleFactors for EGS/RPGC.")
    if source.matrixInfo is None:
        raise ValueError("H5AD source has not been prepared.")
    count = source.matrixInfo["normalizationCount"]
    if count <= 0:
        raise ValueError("H5AD normalization requires positive total bin counts.")
    factor = 1_000_000.0 / count
    if normMethod == "RPKM":
        factor *= 1000.0 / intervalSizeBP
    return round(factor, 5)
