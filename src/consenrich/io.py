"""Input preparation, bedGraph, and output-conversion helpers."""

from __future__ import annotations

import csv
import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from consenrich import cconsenrich
import consenrich.constants as constants
import consenrich.core as core
import consenrich.misc_util as misc_util
from . import _normalization as normalization
from ._version import __version__
from ._runtime import (
    get_available_memory_bytes as _sharedGetAvailableMemoryBytes,
    get_munc_worker_count as _sharedGetMuncWorkerCount,
    get_small_worker_count as _sharedGetSmallWorkerCount,
    thread_map as _sharedThreadMap,
)

logger = logging.getLogger(__name__)


def _getSmallWorkerCount(taskCount: int, maxWorkers: int = 4) -> int:
    return _sharedGetSmallWorkerCount(taskCount, max_workers=maxWorkers)


_MEMORY_UNSET = object()


def _getMuncWorkerCount(
    numSamples: int,
    numIntervals: int,
    sharedArrays=(),
    availableMemoryBytes=_MEMORY_UNSET,
    logger_=logger,
) -> int:
    if availableMemoryBytes is _MEMORY_UNSET:
        availableMemoryBytes = _sharedGetAvailableMemoryBytes()
    return _sharedGetMuncWorkerCount(
        numSamples,
        numIntervals,
        shared_arrays=sharedArrays,
        available_memory_bytes=availableMemoryBytes,
        logger=logger_,
    )


def _threadMap(
    items,
    func,
    label: str,
    allowThreads: bool = True,
    minItems: int = 4,
    maxWorkers: int = 4,
):
    return _sharedThreadMap(
        items,
        func,
        label,
        logger=logger,
        allow_threads=allowThreads,
        min_items=minItems,
        max_workers=maxWorkers,
    )


@lru_cache(maxsize=1)
def _pyBigWigAvailable() -> bool:
    try:
        import pyBigWig  # noqa: F401
    except Exception:
        return False
    return True


def _listOrEmpty(list_):
    if list_ is None:
        return []
    return list_


def _expandWildCards(pathList: List[str]) -> List[str]:
    expandedList: List[str] = []
    for pathEntry in pathList:
        if "*" in pathEntry or "?" in pathEntry or "[" in pathEntry:
            matchedList = sorted(glob.glob(pathEntry))
            if matchedList:
                expandedList.extend(matchedList)
            else:
                expandedList.append(pathEntry)
        else:
            expandedList.append(pathEntry)
    return expandedList


def _inferSourceKind(path: str) -> str:
    lowerPath = str(path).lower()
    if lowerPath.endswith(".cram"):
        raise ValueError("CRAM inputs are unsupported.")
    if lowerPath.endswith((".bedgraph", ".bedgraph.gz", ".bdg", ".bdg.gz")):
        return core.BEDGRAPH_SOURCE_KIND
    if lowerPath.endswith(".h5ad"):
        return core.H5AD_SOURCE_KIND
    if "fragments" in os.path.basename(lowerPath):
        return "FRAGMENTS"
    return "BAM"


def _resolveSourceKind(sourceKind: Optional[str], path: str) -> str:
    kind = _inferSourceKind(path) if sourceKind is None else sourceKind
    if kind not in core.SUPPORTED_SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind {kind!r}")
    return kind


def _coerceInputSource(
    sourceConfig: Union[str, Mapping[str, Any]],
    defaultRole: str,
    defaultBarcodeTag: str | None = None,
    fragmentsUseReadSupport: bool = constants.SC_DEFAULT_FRAGMENTS_USE_READ_SUPPORT,
) -> core.inputSource:
    if isinstance(sourceConfig, str):
        path = sourceConfig
        sourceKind = None
        sampleName = None
        role = defaultRole
        barcodeTag = defaultBarcodeTag
        barcodeAllowListFile = None
        barcodeGroupMapFile = None
        selectGroups = None
        countMode = None
        bamInputMode = None
        metadataSample = None
        cellMetadata = None
        matrixRows = None
    elif isinstance(sourceConfig, Mapping):
        allowedFields = {
            "path", "format", "name", "role", "barcodeTag", "barcodeAllowListFile",
            "barcodeGroupMapFile", "selectGroups", "metadataSample", "cellMetadata",
            "matrixRows", "countMode", "bamInputMode", "fragmentsUseReadSupport",
        }
        if set(sourceConfig) - allowedFields:
            raise ValueError(f"Unknown input sample field {sorted(set(sourceConfig) - allowedFields)}")
        path = str(sourceConfig.get("path", ""))
        sourceKind = sourceConfig.get("format", None)
        sampleName = sourceConfig.get("name", None)
        role = sourceConfig.get("role", defaultRole)
        barcodeTag = sourceConfig.get("barcodeTag", defaultBarcodeTag)
        barcodeAllowListFile = sourceConfig.get("barcodeAllowListFile", None)
        barcodeGroupMapFile = sourceConfig.get("barcodeGroupMapFile", None)
        selectGroups = sourceConfig.get("selectGroups", None)
        metadataSample = sourceConfig.get("metadataSample", None)
        cellMetadata = sourceConfig.get("cellMetadata", None)
        matrixRows = sourceConfig.get("matrixRows", None)
        countMode = sourceConfig.get("countMode", None)
        bamInputMode = sourceConfig.get("bamInputMode", None)
        fragmentsUseReadSupport = sourceConfig.get(
            "fragmentsUseReadSupport", fragmentsUseReadSupport
        )
    else:
        raise TypeError("Each input source must be a path string or a mapping.")

    if not path:
        raise ValueError("Each input source requires a non-empty `path`.")

    if not isinstance(fragmentsUseReadSupport, bool):
        raise ValueError("`fragmentsUseReadSupport` must be a boolean.")

    if role not in ["treatment", "control"]:
        raise ValueError(f"Unsupported source role `{role}` for `{path}`.")

    if selectGroups is not None and not isinstance(selectGroups, list):
        raise ValueError(f"`selectGroups` must be a list for `{path}`.")

    pathList = _expandWildCards([path])
    if len(pathList) != 1:
        raise ValueError(
            f"Input source `{path}` expanded to {len(pathList)} paths. Use one entry per source."
        )
    normalizedPath = pathList[0]
    normalizedKind = _resolveSourceKind(sourceKind, normalizedPath)
    if normalizedKind == core.H5AD_SOURCE_KIND:
        if matrixRows not in ("cells", "pseudobulks"):
            raise ValueError("H5AD input requires matrixRows: cells or pseudobulks.")
        if countMode is not None or bamInputMode is not None:
            raise ValueError("H5AD raw counts do not accept countMode or bamInputMode.")
    elif matrixRows is not None:
        raise ValueError("matrixRows requires H5AD input.")

    return core.inputSource(
        path=normalizedPath,
        sourceKind=normalizedKind,
        role=role,
        sampleName=sampleName,
        barcodeTag=barcodeTag,
        barcodeAllowListFile=barcodeAllowListFile,
        barcodeGroupMapFile=barcodeGroupMapFile,
        selectGroups=selectGroups,
        countMode=countMode,
        bamInputMode=bamInputMode,
        fragmentsUseReadSupport=fragmentsUseReadSupport,
        metadataSample=metadataSample,
        cellMetadata=cellMetadata,
        matrixRows=matrixRows,
    )


def _buildPathInputSources(
    pathList: List[str],
    role: str,
    fragmentsUseReadSupport: bool = constants.SC_DEFAULT_FRAGMENTS_USE_READ_SUPPORT,
) -> List[core.inputSource]:
    return [
        core.inputSource(
            path=path,
            sourceKind=_resolveSourceKind(None, path),
            role=role,
            fragmentsUseReadSupport=fragmentsUseReadSupport,
        )
        for path in _expandWildCards(pathList)
    ]


def _getSourceCountMode(
    source: core.inputSource,
    defaultBamCountMode: str = constants.SAM_DEFAULT_COUNT_MODE,
    defaultFragmentCountMode: str = constants.SC_DEFAULT_COUNT_MODE,
) -> str:
    if str(source.sourceKind).upper() == core.H5AD_SOURCE_KIND:
        return "rawCounts"
    if str(source.sourceKind).upper() == core.BEDGRAPH_SOURCE_KIND:
        return "coverage"
    countMode = normalization.validateCountMode(
        source.countMode,
        (
            defaultFragmentCountMode
            if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND
            else defaultBamCountMode
        ),
    )
    if source.sourceKind == core.FRAGMENTS_SOURCE_KIND and countMode == "ffp":
        raise ValueError("ffp requires BAM input")
    return countMode


def checkControlsPresent(inputArgs: core.inputParams) -> bool:
    """Check if control BAM files are present in the input arguments.

    :param inputArgs: core.inputParams object
    :return: True if control BAM files are present, False otherwise.
    """
    return (
        bool(inputArgs.bamFilesControl)
        and isinstance(inputArgs.bamFilesControl, list)
        and len(inputArgs.bamFilesControl) > 0
    )


def checkMatchingEnabled(matchingArgs: core.matchingParams) -> bool:
    return bool(getattr(matchingArgs, "enabled", True))


def getEffectiveGenomeSizes(
    genomeArgs: core.genomeParams,
    readLengths: List[int],
    sources: Sequence[core.inputSource] | None = None,
) -> List[int]:
    r"""Get effective genome sizes for the given genome name and read lengths.
    :param genomeArgs: core.genomeParams object
    :param readLengths: List of read lengths for which to get effective genome sizes.
    :return: List of effective genome sizes corresponding to the read lengths.
    """
    if not isinstance(readLengths, list) or len(readLengths) == 0:
        raise ValueError(
            "Read lengths must be a non-empty list. Try calling `getReadLengths` first."
        )
    explicitSize = getattr(genomeArgs, "effectiveGenomeSize", None)
    if explicitSize is not None:
        return [int(explicitSize)] * len(readLengths)
    genomeName = genomeArgs.genomeName
    if not genomeName or not isinstance(genomeName, str):
        raise ValueError("Genome name must be a non-empty string.")
    if sources is not None and len(sources) != len(readLengths):
        raise ValueError("Sources and read lengths must have equal lengths")
    if sources is None or any(source.sourceKind == "BAM" for source in sources):
        assemblySizes = constants.assemblyChromosomeSizes[genomeName]
        expectedTerritory = set(constants.getAssemblyChromosomes(genomeName)) - set(genomeArgs.excludeForNorm or ())
        territory = set(genomeArgs.normalizationChromosomes)
        if territory != expectedTerritory or any(
            genomeArgs.chromosomeSizes[chromosome] != assemblySizes[chromosome]
            for chromosome in territory
        ):
            raise ValueError(
                "Automatic BAM RPGC requires assembly chromosome geometry and normalization territory. "
                "Set genomeParams.effectiveGenomeSize for a custom territory."
            )
    fragmentSize = None
    if sources is not None and any(source.sourceKind == "FRAGMENTS" for source in sources):
        fragmentSize = constants.getNonNGenomeSize(
            genomeName,
            {
                chromosome: genomeArgs.chromosomeSizes[chromosome]
                for chromosome in genomeArgs.normalizationChromosomes
            },
        )
    physicalSizes = genomeArgs.chromosomeSizes
    excludedBases = sum(physicalSizes.get(chromosome, 0)
                        for chromosome in (genomeArgs.excludeForNorm or ()))
    return [
        (
            fragmentSize
            if sources is not None and sources[index].sourceKind == "FRAGMENTS"
            else (
                0
                if int(readLength) <= 0
                else constants.getEffectiveGenomeSize(genomeName, readLength) - excludedBases
            )
        )
        for index, readLength in enumerate(readLengths)
    ]

def convertBedGraphToBigWig(
    experimentName,
    chromSizesFile,
    suffixes: Optional[List[str]] = None,
    *,
    validatedBedGraphs: Optional[Sequence[str]] = None,
    deleteBedGraphsAfterBigWig: bool = False,
):
    if suffixes is None:
        # at least look for `state` bedGraph
        suffixes = ["state"]
    validatedPaths = {
        os.path.abspath(str(path)) for path in (validatedBedGraphs or ())
    }
    chromSizes = None
    logger.info(
        "Attempting to generate bigWig files from bedGraph format via pyBigWig..."
    )
    for suffix in suffixes:
        bedgraph = f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
        if not os.path.exists(bedgraph):
            logger.warning(
                f"bedGraph file {bedgraph} does not exist. Skipping bigWig conversion."
            )
            continue
        if not isinstance(chromSizesFile, Mapping) and not os.path.exists(chromSizesFile):
            logger.warning(
                f"{chromSizesFile} does not exist. Skipping bigWig conversion."
            )
            return
        if chromSizes is None:
            chromSizes = list(misc_util.getChromSizesDict(chromSizesFile).items())
        chromOrder = [chrom for chrom, _size in chromSizes]
        if os.path.abspath(bedgraph) not in validatedPaths:
            try:
                _validateBedGraphSorted(bedgraph, chromOrder=chromOrder)
            except Exception as e:
                logger.warning(
                    "bedGraph %s failed sorted validation before bigWig conversion; "
                    "sorting as a fallback:\n%s",
                    bedgraph,
                    e,
                )
                try:
                    _sortBedGraphInPlace(bedgraph, chromOrder=chromOrder)
                except Exception as sortError:
                    logger.warning(
                        f"Failed to sort {bedgraph} before bigWig conversion:\n{sortError}"
                    )
                    continue
        bigwig = f"{experimentName}_consenrich_{suffix}.v{__version__}.bw"
        logger.info(f"Start: {bedgraph} --> {bigwig}...")
        try:
            _convertBedGraphToBigWigPyBigWig(
                bedgraph,
                chromSizesFile,
                bigwig,
                chromSizes=chromSizes,
                validated=os.path.abspath(bedgraph) in validatedPaths,
            )
        except Exception as e:
            logger.warning(
                f"pyBigWig bedGraph-->bigWig conversion for {bedgraph} raised:\n{e}\n"
            )
            continue
        if os.path.exists(bigwig) and os.path.getsize(bigwig) > 100:
            logger.info(f"Finished: converted {bedgraph} to {bigwig}.")
            if deleteBedGraphsAfterBigWig:
                os.remove(bedgraph)
                logger.info(f"Deleted converted bedGraph {bedgraph}.")


class _BedGraphDataReader:
    def __init__(self, handle):
        self.handle = handle
        self.metadataPattern = re.compile(
            r"(?m)^[^\S\r\n]*(?:#[^\r\n]*|(?:track|browser)(?: [^\r\n]*)?)"
            r"[^\S\r\n]*(?=\r?$)"
        )
        self.whitespacePattern = re.compile(r"[^\S\t\r\n ]")

    def read(self, size=-1):
        text = self.handle.read(size)
        if size >= 0 and text and not text.endswith("\n"):
            text += self.handle.readline()
        if "#" in text or "track" in text or "browser" in text:
            text = self.metadataPattern.sub("", text)
        if not text.isascii() or any(
            character in text for character in "\v\f\x1c\x1d\x1e\x1f"
        ):
            text = self.whitespacePattern.sub(" ", text)
        if "\0" in text:
            raise ValueError("Malformed bedGraph text contains a NUL byte")
        return text


def _convertBedGraphToBigWigPyBigWig(
    bedgraphPath: str,
    chromSizesFile: str | Mapping[str, int],
    bigwigPath: str,
    chunkSize: int = 200_000,
    *,
    chromSizes: Optional[Sequence[Tuple[str, int]]] = None,
    validated: bool = False,
) -> None:
    try:
        import pyBigWig
    except Exception as e:
        raise RuntimeError(
            "pyBigWig is not installed; cannot convert bedGraph files to bigWig"
        ) from e

    chromSizes_ = (
        list(misc_util.getChromSizesDict(chromSizesFile).items())
        if chromSizes is None
        else [(str(chrom), int(size)) for chrom, size in chromSizes]
    )
    if len(chromSizes_) == 0:
        raise ValueError("Chromosome sizes must not be empty")
    chromNames = [chrom for chrom, _size in chromSizes_]
    chromIndex = pd.Index(chromNames, dtype=object)
    sizes = np.asarray([size for _chrom, size in chromSizes_], dtype=np.int64)

    chunkSize_ = max(int(chunkSize), 1)
    outDir = os.path.dirname(os.path.abspath(bigwigPath)) or "."
    tempPath = ""
    bw = None
    seenEntry = False
    lastRank = -1
    lastStart = -1
    lastEnd = -1
    try:
        with tempfile.NamedTemporaryFile(
            prefix="consenrich_bigwig_",
            suffix=".bw",
            delete=False,
            dir=outDir,
        ) as tempHandle:
            tempPath = tempHandle.name
        bw = pyBigWig.open(tempPath, "w")
        bw.addHeader(chromSizes_)
        with (
            open(bedgraphPath, "r", encoding="utf-8") as handle,
            pd.read_csv(
                _BedGraphDataReader(handle),
                sep=r"\s+",
                header=None,
                names=["chrom", "start", "end", "value", "extra"],
                index_col=False,
                dtype=(
                    {"chrom": object, "start": np.int64, "end": np.int64,
                     "value": np.float64, "extra": object}
                    if validated else object
                ),
                chunksize=chunkSize_,
                engine="c",
                float_precision="round_trip",
                na_filter=False,
                quoting=csv.QUOTE_NONE,
                on_bad_lines="error",
            ) as chunks,
        ):
            for frame in chunks:
                if frame.empty:
                    continue
                if np.any(frame["extra"].to_numpy(copy=False) != ""):
                    raise ValueError(
                        f"Malformed bedGraph row in {bedgraphPath}: expected 4 columns"
                    )
                ranks = chromIndex.get_indexer(frame["chrom"])
                unknown = np.flatnonzero(ranks < 0)
                if unknown.size:
                    chrom = frame["chrom"].iloc[int(unknown[0])]
                    raise ValueError(
                        f"Chromosome {chrom} in {bedgraphPath} is not present "
                        "in chromosome sizes"
                    )
                try:
                    starts = frame["start"].to_numpy(dtype=np.int64)
                    ends = frame["end"].to_numpy(dtype=np.int64)
                except (ValueError, OverflowError) as e:
                    raise ValueError(
                        f"Invalid bedGraph coordinates in {bedgraphPath}"
                    ) from e
                try:
                    values = frame["value"].to_numpy(dtype=np.float64)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bedGraph value in {bedgraphPath}"
                    ) from e
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Non-finite bedGraph value in {bedgraphPath}")
                if np.any(starts < 0):
                    raise ValueError(f"Negative start coordinate in {bedgraphPath}")
                if np.any(ends <= starts):
                    raise ValueError(
                        f"End coordinate must be greater than start in {bedgraphPath}"
                    )
                outside = np.flatnonzero(ends > sizes[ranks])
                if outside.size:
                    index = int(outside[0])
                    chrom = chromNames[int(ranks[index])]
                    raise ValueError(
                        f"End coordinate {ends[index]} in {bedgraphPath} exceeds "
                        f"{chrom} size of {sizes[ranks[index]]}"
                    )
                sameChrom = ranks[1:] == ranks[:-1]
                if (
                    ranks[0] < lastRank
                    or (ranks[0] == lastRank and starts[0] < lastStart)
                    or np.any(ranks[1:] < ranks[:-1])
                    or np.any(sameChrom & (starts[1:] < starts[:-1]))
                ):
                    raise ValueError(
                        f"bedGraph input is not sorted in {bedgraphPath}"
                    )
                if (
                    (ranks[0] == lastRank and starts[0] < lastEnd)
                    or np.any(sameChrom & (starts[1:] < ends[:-1]))
                ):
                    raise ValueError(
                        f"Overlapping bedGraph interval in {bedgraphPath}"
                    )
                entryValues = values.astype(np.float32)
                runBounds = np.r_[0, np.flatnonzero(~sameChrom) + 1, len(ranks)]
                for runStart, runStop in zip(runBounds[:-1], runBounds[1:]):
                    chrom = chromNames[int(ranks[runStart])]
                    runStarts = starts[runStart:runStop]
                    runEnds = ends[runStart:runStop]
                    runValues = entryValues[runStart:runStop]
                    span = int(runEnds[0] - runStarts[0])
                    if np.all(runEnds - runStarts == span):
                        step = int(runStarts[1] - runStarts[0]) if len(runStarts) > 1 else span
                        if np.all(np.diff(runStarts) == step):
                            bw.addEntries(
                                chrom, int(runStarts[0]), values=runValues,
                                span=span, step=step, validate=False,
                            )
                        else:
                            bw.addEntries(
                                chrom, runStarts, values=runValues,
                                span=span, validate=False,
                            )
                    else:
                        bw.addEntries(
                            [chrom] * len(runStarts), runStarts,
                            ends=runEnds, values=runValues, validate=False,
                        )
                seenEntry = True
                lastRank = int(ranks[-1])
                lastStart = int(starts[-1])
                lastEnd = int(ends[-1])
        if not seenEntry:
            raise ValueError(f"No bedGraph intervals found in {bedgraphPath}")
        bw.close()
        bw = None
        os.replace(tempPath, bigwigPath)
    finally:
        try:
            if bw is not None:
                bw.close()
        finally:
            if tempPath and os.path.exists(tempPath):
                try:
                    os.remove(tempPath)
                except OSError:
                    pass


def _validateBedGraphSorted(
    bedgraphPath: str,
    chromOrder: Optional[Sequence[str]] = None,
) -> None:
    if not os.path.exists(bedgraphPath) or os.path.getsize(bedgraphPath) == 0:
        return

    chromRankByName = (
        {str(chrom): rank for rank, chrom in enumerate(chromOrder)}
        if chromOrder is not None
        else None
    )
    seenEntry = False
    seenChroms: set[str] = set()
    lastChrom = ""
    lastStart = -1
    lastEnd = -1
    lastRank = -1

    with open(bedgraphPath, "r", encoding="utf-8") as handle:
        for lineNumber, line in enumerate(handle, start=1):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped == "track"
                or stripped.startswith("track ")
                or stripped == "browser"
                or stripped.startswith("browser ")
            ):
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Malformed bedGraph row {lineNumber} in {bedgraphPath}: "
                    "expected 4 columns"
                )
            chrom = str(parts[0])
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError as e:
                raise ValueError(
                    f"Invalid bedGraph coordinates on row {lineNumber} in "
                    f"{bedgraphPath}"
                ) from e
            if start < 0:
                raise ValueError(
                    f"Negative start coordinate on bedGraph row {lineNumber}"
                )
            if end <= start:
                raise ValueError(
                    f"End coordinate must be greater than start on bedGraph row "
                    f"{lineNumber}"
                )
            if chromRankByName is None:
                chromRank = -1
            else:
                if chrom not in chromRankByName:
                    raise ValueError(
                        f"Chromosome {chrom} on bedGraph row {lineNumber} is not "
                        "present in the expected chromosome order"
                    )
                chromRank = chromRankByName[chrom]
            if seenEntry:
                if chrom == lastChrom:
                    if start < lastStart:
                        raise ValueError(
                            f"bedGraph input is not sorted at row {lineNumber}"
                        )
                    if start < lastEnd:
                        raise ValueError(
                            f"Overlapping bedGraph interval at row {lineNumber}"
                        )
                else:
                    if chrom in seenChroms:
                        raise ValueError(
                            f"Chromosome {chrom} reappears on bedGraph row "
                            f"{lineNumber}"
                        )
                    if chromRankByName is None:
                        if chrom < lastChrom:
                            raise ValueError(
                                f"bedGraph input is not sorted at row {lineNumber}"
                            )
                    elif chromRank < lastRank:
                        raise ValueError(
                            f"bedGraph input is not in chromosome order at row "
                            f"{lineNumber}"
                        )
            seenEntry = True
            seenChroms.add(chrom)
            lastChrom = chrom
            lastStart = start
            lastEnd = end
            lastRank = chromRank


def _writeBedGraphChunk(
    bedgraphPath: str,
    chromosome: str,
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    chunkRanges: dict[str, tuple[int, int]],
) -> None:
    if chromosome in chunkRanges:
        raise ValueError(f"Duplicate bedGraph chromosome chunk: {chromosome}")
    offset = os.path.getsize(bedgraphPath) if chunkRanges else 0
    cconsenrich.cWriteBedGraph(
        bedgraphPath,
        chromosome,
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(values),
        append=bool(chunkRanges),
    )
    chunkRanges[chromosome] = (offset, os.path.getsize(bedgraphPath) - offset)


def _reorderBedGraphChunks(
    bedgraphPath: str,
    chunkRanges: dict[str, tuple[int, int]],
    chromOrder: Sequence[str],
) -> None:
    if not chunkRanges:
        raise ValueError("No bedGraph chromosome chunks were written")
    if len(set(chromOrder)) != len(chromOrder):
        raise ValueError("Chromosome order must not contain duplicates")
    orderedChromosomes = [chrom for chrom in chromOrder if chrom in chunkRanges]
    if len(orderedChromosomes) != len(chunkRanges):
        raise ValueError("bedGraph chunks contain chromosomes absent from chromosome order")
    expectedOffset = 0
    for offset, length in chunkRanges.values():
        if offset != expectedOffset or length < 0:
            raise ValueError("bedGraph chromosome byte ranges must be contiguous")
        expectedOffset += length
    if expectedOffset != os.path.getsize(bedgraphPath):
        raise ValueError("bedGraph chromosome byte ranges do not match file size")
    if orderedChromosomes == list(chunkRanges):
        return
    tempPath = ""
    buffer = bytearray(1024 * 1024)
    view = memoryview(buffer)
    try:
        with open(bedgraphPath, "rb") as source, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="consenrich_reorder_",
            suffix=".bedGraph",
            dir=os.path.dirname(os.path.abspath(bedgraphPath)),
            delete=False,
        ) as output:
            tempPath = output.name
            for chromosome in orderedChromosomes:
                offset, remaining = chunkRanges[chromosome]
                source.seek(offset)
                while remaining:
                    count = source.readinto(view[:min(remaining, len(buffer))])
                    if not count:
                        raise ValueError("bedGraph chromosome byte range exceeds file size")
                    if output.write(view[:count]) != count:
                        raise OSError("Incomplete bedGraph chromosome chunk write")
                    remaining -= count
        os.chmod(tempPath, os.stat(bedgraphPath).st_mode & 0o7777)
        os.replace(tempPath, bedgraphPath)
        reorderedRanges = {}
        offset = 0
        for chromosome in orderedChromosomes:
            length = chunkRanges[chromosome][1]
            reorderedRanges[chromosome] = (offset, length)
            offset += length
        chunkRanges.clear()
        chunkRanges.update(reorderedRanges)
    finally:
        if tempPath and os.path.exists(tempPath):
            os.remove(tempPath)


def _sortBedGraphInPlace(
    bedgraphPath: str,
    chromOrder: Optional[Sequence[str]] = None,
) -> None:
    if not os.path.exists(bedgraphPath) or os.path.getsize(bedgraphPath) == 0:
        return

    sortPath = shutil.which("sort")
    if sortPath is not None and chromOrder is None:
        tempPath = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="consenrich_sort_",
                suffix=".bedGraph",
                delete=False,
                dir=os.path.dirname(os.path.abspath(bedgraphPath)) or ".",
            ) as tempHandle:
                tempPath = tempHandle.name
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            with open(tempPath, "w", encoding="utf-8") as outHandle:
                subprocess.run(
                    [
                        sortPath,
                        "-k1,1",
                        "-k2,2n",
                        "-k3,3n",
                        bedgraphPath,
                    ],
                    check=True,
                    stdout=outHandle,
                    env=env,
                )
            os.replace(tempPath, bedgraphPath)
            return
        finally:
            if tempPath and os.path.exists(tempPath):
                try:
                    os.remove(tempPath)
                except OSError:
                    pass

    headerLines: List[str] = []
    recordRows: List[List[str]] = []
    with open(bedgraphPath, "r", encoding="utf-8") as handle:
        for lineNumber, line in enumerate(handle, start=1):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped == "track"
                or stripped.startswith("track ")
                or stripped == "browser"
                or stripped.startswith("browser ")
            ):
                headerLines.append(line.rstrip("\n"))
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Malformed bedGraph row {lineNumber} in {bedgraphPath}: "
                    "expected 4 columns"
                )
            recordRows.append(parts)
    if not recordRows:
        with open(bedgraphPath, "w", encoding="utf-8") as handle:
            for headerLine in headerLines:
                handle.write(f"{headerLine}\n")
        return

    df = pd.DataFrame(
        recordRows,
        columns=["chromosome", "start", "end", "value"],
    )
    df["_valueText"] = df["value"]
    df = df.astype(
        {
            "chromosome": str,
            "start": np.int64,
            "end": np.int64,
            "value": np.float64,
        }
    )
    if chromOrder is not None:
        chromRankByName = {str(chrom): rank for rank, chrom in enumerate(chromOrder)}
        df["_chromRank"] = df["chromosome"].map(chromRankByName)
        unknown = sorted(df.loc[df["_chromRank"].isna(), "chromosome"].unique())
        if unknown:
            raise ValueError(
                "bedGraph contains chromosomes not present in chromosome order: "
                + ", ".join(str(chrom) for chrom in unknown[:5])
            )
        df["_chromRank"] = df["_chromRank"].astype(np.int64)
        df.sort_values(
            by=["_chromRank", "start", "end"],
            kind="mergesort",
            inplace=True,
        )
        df.drop(columns=["_chromRank"], inplace=True)
    else:
        df.sort_values(
            by=["chromosome", "start", "end"],
            kind="mergesort",
            inplace=True,
        )
    with open(bedgraphPath, "w", encoding="utf-8") as handle:
        for headerLine in headerLines:
            handle.write(f"{headerLine}\n")
        df["value"] = df.pop("_valueText")
        df.to_csv(
            handle,
            sep="\t",
            header=False,
            index=False,
            lineterminator="\n",
        )

__all__ = [
    "checkControlsPresent",
    "checkMatchingEnabled",
    "convertBedGraphToBigWig",
    "getEffectiveGenomeSizes",
]
