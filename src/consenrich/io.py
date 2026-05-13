"""Input preparation, bedGraph, and output-conversion helpers."""

from __future__ import annotations

import logging
import os
from multiprocessing.pool import ThreadPool


logger = logging.getLogger(__name__)


def _getSmallWorkerCount(taskCount: int, maxWorkers: int = 4) -> int:
    cpuCount = os.cpu_count() or 1
    return min(taskCount, max(1, cpuCount // 2), maxWorkers)


_MEMORY_UNSET = object()


def _getAvailableMemoryBytes() -> int | None:
    try:
        import psutil  # type: ignore

        available = int(psutil.virtual_memory().available)
        if available > 0:
            return available
    except Exception:
        pass

    try:
        with open("/proc/meminfo", "r", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass

    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        pageSize = int(os.sysconf("SC_PAGE_SIZE"))
        total = pages * pageSize
        if total > 0:
            return total // 4
    except (AttributeError, OSError, ValueError):
        pass

    return None


def _getMuncWorkerCount(
    numSamples: int,
    numIntervals: int,
    sharedArrays=(),
    availableMemoryBytes=_MEMORY_UNSET,
    logger_=logger,
) -> int:
    numSamples = int(numSamples)
    numIntervals = int(numIntervals)
    if numSamples <= 0:
        return 1

    cpuCount = os.cpu_count() or 1
    cpuWorkers = min(numSamples, max(1, cpuCount // 2))
    if cpuWorkers <= 1:
        return 1

    if availableMemoryBytes is _MEMORY_UNSET:
        availableMemoryBytes = _getAvailableMemoryBytes()
    if availableMemoryBytes is None:
        return cpuWorkers

    try:
        availableBytes = int(availableMemoryBytes)
    except (TypeError, ValueError):
        return cpuWorkers
    if availableBytes <= 0:
        return cpuWorkers

    sharedBytes = 0
    for arr in sharedArrays:
        if arr is not None:
            sharedBytes += int(getattr(arr, "nbytes", 0) or 0)

    scratchBytes = max(
        int(64 * max(numIntervals, 0)) + (64 * 1024 * 1024),
        128 * 1024 * 1024,
    )
    memoryBudget = int(max(0, availableBytes - sharedBytes) * 0.5)
    memoryWorkers = max(1, memoryBudget // scratchBytes)
    workers = max(1, min(cpuWorkers, memoryWorkers))
    if workers < cpuWorkers and logger_ is not None:
        logger_.info(
            "munc matrix calculation: reducing workers from %d to %d based on memory estimate "
            "(available=%.2f GiB, shared=%.2f GiB, perWorker=%.2f GiB).",
            int(cpuWorkers),
            int(workers),
            availableBytes / (1024**3),
            sharedBytes / (1024**3),
            scratchBytes / (1024**3),
        )
    return workers


def _threadMap(
    items,
    func,
    label: str,
    allowThreads: bool = True,
    minItems: int = 4,
    maxWorkers: int = 4,
):
    itemList = list(items)
    if len(itemList) == 0:
        return []

    workerCount = _getSmallWorkerCount(len(itemList), maxWorkers=maxWorkers)
    usePool = allowThreads and len(itemList) >= minItems and workerCount > 1
    if usePool:
        logger.info(
            "%s: using ThreadPool with %d workers (n=%d).",
            label,
            int(workerCount),
            int(len(itemList)),
        )
        with ThreadPool(processes=int(workerCount)) as pool:
            return list(pool.map(func, itemList))

    return [func(item) for item in itemList]


def _delegate(name: str, *args, **kwargs):
    from . import consenrich as cli

    return getattr(cli, name)(*args, **kwargs)


def _checkSF(*args, **kwargs):
    return _delegate("_checkSF", *args, **kwargs)


def _expandWildCards(*args, **kwargs):
    return _delegate("_expandWildCards", *args, **kwargs)


def _inferSourceKind(*args, **kwargs):
    return _delegate("_inferSourceKind", *args, **kwargs)


def _prepareBedGraphSources(*args, **kwargs):
    return _delegate("_prepareBedGraphSources", *args, **kwargs)


def _normalizeSourceKind(*args, **kwargs):
    return _delegate("_normalizeSourceKind", *args, **kwargs)


def _coerceInputSource(*args, **kwargs):
    return _delegate("_coerceInputSource", *args, **kwargs)


def _buildPathInputSources(*args, **kwargs):
    return _delegate("_buildPathInputSources", *args, **kwargs)


def _getSourceCountMode(*args, **kwargs):
    return _delegate("_getSourceCountMode", *args, **kwargs)


def _prepareFragmentsNormalizationMetadata(*args, **kwargs):
    return _delegate("_prepareFragmentsNormalizationMetadata", *args, **kwargs)


def _resolveExtendFrom5pBPPairs(*args, **kwargs):
    return _delegate("_resolveExtendFrom5pBPPairs", *args, **kwargs)


def checkControlsPresent(*args, **kwargs):
    return _delegate("checkControlsPresent", *args, **kwargs)


def getReadLengths(*args, **kwargs):
    return _delegate("getReadLengths", *args, **kwargs)


def checkMatchingEnabled(*args, **kwargs):
    return _delegate("checkMatchingEnabled", *args, **kwargs)


def _inferMatchingUncertaintyBedGraph(*args, **kwargs):
    return _delegate("_inferMatchingUncertaintyBedGraph", *args, **kwargs)


def getEffectiveGenomeSizes(*args, **kwargs):
    return _delegate("getEffectiveGenomeSizes", *args, **kwargs)


def convertBedGraphToBigWig(*args, **kwargs):
    return _delegate("convertBedGraphToBigWig", *args, **kwargs)


def _sortBedGraphInPlace(*args, **kwargs):
    return _delegate("_sortBedGraphInPlace", *args, **kwargs)


__all__ = [
    "_buildPathInputSources",
    "_checkSF",
    "_coerceInputSource",
    "_expandWildCards",
    "_getAvailableMemoryBytes",
    "_getMuncWorkerCount",
    "_getSmallWorkerCount",
    "_getSourceCountMode",
    "_inferMatchingUncertaintyBedGraph",
    "_inferSourceKind",
    "_normalizeSourceKind",
    "_prepareBedGraphSources",
    "_prepareFragmentsNormalizationMetadata",
    "_resolveExtendFrom5pBPPairs",
    "_sortBedGraphInPlace",
    "_threadMap",
    "checkControlsPresent",
    "checkMatchingEnabled",
    "convertBedGraphToBigWig",
    "getEffectiveGenomeSizes",
    "getReadLengths",
]
