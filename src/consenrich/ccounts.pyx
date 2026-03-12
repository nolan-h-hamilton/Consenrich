# cython: language_level=3

import numpy as np
cimport numpy as cnp

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from libc.stdlib cimport malloc, free

cnp.import_array()


cdef extern from "native/ccounts_backend.h":
    ctypedef enum ccounts_sourceKind:
        ccounts_sourceKindBAM
        ccounts_sourceKindCRAM
        ccounts_sourceKindFragments

    ctypedef enum ccounts_countMode:
        ccounts_countModeCoverage
        ccounts_countModeCutSite
        ccounts_countModeFivePrime
        ccounts_countModeCenter

    ctypedef struct ccounts_sourceConfig:
        const char* path
        ccounts_sourceKind sourceKind
        const char* referenceFASTA
        const char* barcodeTag
        const char* barcodeAllowListFile
        const char* barcodeGroupMapFile

    ctypedef struct ccounts_region:
        const char* chromosome
        uint32_t start
        uint32_t end
        uint32_t intervalSizeBP

    ctypedef struct ccounts_countOptions:
        uint16_t threadCount
        uint16_t flagExclude
        uint8_t countMode
        uint8_t oneReadPerBin
        int64_t shiftForwardStrand53
        int64_t shiftReverseStrand53
        int64_t readLength
        int64_t extendBP
        int64_t minMappingQuality
        int64_t minTemplateLength
        int64_t maxInsertSize
        int64_t pairedEndMode
        int64_t inferFragmentLength

    ctypedef struct ccounts_result:
        int errorCode
        const char* errorMessage

    ctypedef struct ccounts_sourceHandle:
        pass

    ccounts_result ccounts_checkAlignmentFile(
        const ccounts_sourceConfig* sourceConfig,
        int buildIndex,
        int threadCount,
        int* hasIndexOut
    )

    ccounts_result ccounts_isPairedEnd(
        const ccounts_sourceConfig* sourceConfig,
        int threadCount,
        int maxReads,
        int* isPairedEndOut
    )

    ccounts_result ccounts_getReadLength(
        const ccounts_sourceConfig* sourceConfig,
        int threadCount,
        int minReads,
        int maxIterations,
        int flagExclude,
        uint32_t* readLengthOut
    )

    ccounts_result ccounts_getChromRange(
        const ccounts_sourceConfig* sourceConfig,
        const char* chromosome,
        uint64_t chromLength,
        int threadCount,
        int flagExclude,
        uint64_t* startOut,
        uint64_t* endOut
    )

    ccounts_result ccounts_getMappedReadCount(
        const ccounts_sourceConfig* sourceConfig,
        int threadCount,
        const char* const* excludeChromosomes,
        int excludeChromosomeCount,
        uint8_t countMode,
        uint8_t oneReadPerBin,
        uint64_t* mappedReadCountOut,
        uint64_t* unmappedReadCountOut
    )

    ccounts_result ccounts_getCellCount(
        const ccounts_sourceConfig* sourceConfig,
        uint64_t* cellCountOut
    )

    ccounts_result ccounts_openSource(
        const ccounts_sourceConfig* sourceConfig,
        ccounts_sourceHandle** sourceHandleOut
    )

    void ccounts_closeSource(ccounts_sourceHandle* sourceHandle)

    ccounts_result ccounts_countRegion(
        ccounts_sourceHandle* sourceHandle,
        const ccounts_region* region,
        const ccounts_countOptions* countOptions,
        float* countBuffer,
        size_t countBufferLength
    )


cdef ccounts_sourceKind _getSourceKindCode(str sourceKind):
    cdef str normalizedKind = str(sourceKind).strip().upper()
    if normalizedKind == "BAM":
        return ccounts_sourceKindBAM
    if normalizedKind == "CRAM":
        return ccounts_sourceKindCRAM
    if normalizedKind == "FRAGMENTS":
        return ccounts_sourceKindFragments
    raise ValueError(f"Unsupported source kind `{sourceKind}`")


cdef ccounts_sourceConfig _makeSourceConfig(
    bytes pathBytes,
    str sourceKind,
    bytes referenceBytes,
    bytes barcodeAllowListBytes=b"",
    bytes barcodeGroupMapBytes=b"",
):
    cdef ccounts_sourceConfig sourceConfig
    sourceConfig.path = pathBytes
    sourceConfig.sourceKind = _getSourceKindCode(sourceKind)
    if len(referenceBytes) > 0:
        sourceConfig.referenceFASTA = referenceBytes
    else:
        sourceConfig.referenceFASTA = NULL
    sourceConfig.barcodeTag = NULL
    if len(barcodeAllowListBytes) > 0:
        sourceConfig.barcodeAllowListFile = barcodeAllowListBytes
    else:
        sourceConfig.barcodeAllowListFile = NULL
    if len(barcodeGroupMapBytes) > 0:
        sourceConfig.barcodeGroupMapFile = barcodeGroupMapBytes
    else:
        sourceConfig.barcodeGroupMapFile = NULL
    return sourceConfig


cdef uint8_t _getCountModeCode(str countMode):
    cdef str normalizedMode = str(countMode).strip().lower()
    if normalizedMode in ["coverage", "cov", "0"]:
        return <uint8_t>ccounts_countModeCoverage
    if normalizedMode in ["cutsite", "cut", "cutsites", "1"]:
        return <uint8_t>ccounts_countModeCutSite
    if normalizedMode in ["fiveprime", "5p", "five_prime", "2"]:
        return <uint8_t>ccounts_countModeFivePrime
    if normalizedMode in ["center", "centre", "midpoint", "3"]:
        return <uint8_t>ccounts_countModeCenter
    raise ValueError(f"Unsupported countMode `{countMode}`")


cdef void _raiseIfError(ccounts_result result):
    if result.errorCode == 0:
        return
    if result.errorMessage != NULL:
        raise RuntimeError((<bytes>result.errorMessage).decode("utf-8"))
    raise RuntimeError("native ccounts backend failed")


cpdef bint ccounts_checkAlignmentPath(
    str alignmentPath,
    str sourceKind="BAM",
    str referenceFASTA="",
    bint buildIndex=False,
    int threadCount=0,
):
    cdef bytes pathBytes = alignmentPath.encode("utf-8")
    cdef bytes referenceBytes = referenceFASTA.encode("utf-8")
    cdef ccounts_sourceConfig sourceConfig = _makeSourceConfig(
        pathBytes,
        sourceKind,
        referenceBytes,
    )
    cdef ccounts_result result
    cdef int hasIndex = 0

    result = ccounts_checkAlignmentFile(
        &sourceConfig,
        1 if buildIndex else 0,
        threadCount,
        &hasIndex,
    )
    _raiseIfError(result)
    return hasIndex != 0


cpdef bint ccounts_isAlignmentPairedEnd(
    str alignmentPath,
    int maxReads=1000,
    int threadCount=0,
    str sourceKind="BAM",
    str referenceFASTA="",
):
    cdef bytes pathBytes = alignmentPath.encode("utf-8")
    cdef bytes referenceBytes = referenceFASTA.encode("utf-8")
    cdef ccounts_sourceConfig sourceConfig = _makeSourceConfig(
        pathBytes,
        sourceKind,
        referenceBytes,
    )
    cdef ccounts_result result
    cdef int isPairedEnd = 0

    result = ccounts_isPairedEnd(
        &sourceConfig,
        threadCount,
        maxReads,
        &isPairedEnd,
    )
    _raiseIfError(result)
    return isPairedEnd != 0


cpdef int ccounts_getAlignmentReadLength(
    str alignmentPath,
    int minReads,
    int threadCount,
    int maxIterations,
    int flagExclude,
    str sourceKind="BAM",
    str referenceFASTA="",
):
    cdef bytes pathBytes = alignmentPath.encode("utf-8")
    cdef bytes referenceBytes = referenceFASTA.encode("utf-8")
    cdef ccounts_sourceConfig sourceConfig = _makeSourceConfig(
        pathBytes,
        sourceKind,
        referenceBytes,
    )
    cdef ccounts_result result
    cdef uint32_t readLength = 0

    result = ccounts_getReadLength(
        &sourceConfig,
        threadCount,
        minReads,
        maxIterations,
        flagExclude,
        &readLength,
    )
    _raiseIfError(result)
    return int(readLength)


cpdef tuple ccounts_getAlignmentChromRange(
    str alignmentPath,
    str chromosome,
    unsigned long long chromLength,
    int threadCount,
    int flagExclude,
    str sourceKind="BAM",
    str referenceFASTA="",
):
    cdef bytes pathBytes = alignmentPath.encode("utf-8")
    cdef bytes referenceBytes = referenceFASTA.encode("utf-8")
    cdef bytes chromosomeBytes = chromosome.encode("utf-8")
    cdef ccounts_sourceConfig sourceConfig = _makeSourceConfig(
        pathBytes,
        sourceKind,
        referenceBytes,
    )
    cdef ccounts_result result
    cdef uint64_t startValue = 0
    cdef uint64_t endValue = 0

    result = ccounts_getChromRange(
        &sourceConfig,
        chromosomeBytes,
        chromLength,
        threadCount,
        flagExclude,
        &startValue,
        &endValue,
    )
    _raiseIfError(result)
    return int(startValue), int(endValue)


cpdef tuple ccounts_getAlignmentMappedReadCount(
    str alignmentPath,
    excludeChromosomes=None,
    int threadCount=0,
    str sourceKind="BAM",
    str referenceFASTA="",
    str barcodeAllowListFile="",
    str countMode="coverage",
    int oneReadPerBin=0,
):
    cdef bytes pathBytes = alignmentPath.encode("utf-8")
    cdef bytes referenceBytes = referenceFASTA.encode("utf-8")
    cdef bytes barcodeAllowListBytes = barcodeAllowListFile.encode("utf-8")
    cdef ccounts_sourceConfig sourceConfig = _makeSourceConfig(
        pathBytes,
        sourceKind,
        referenceBytes,
        barcodeAllowListBytes,
    )
    cdef ccounts_result result
    cdef uint64_t mappedReadCount = 0
    cdef uint64_t unmappedReadCount = 0
    cdef const char** excludePointers = NULL
    cdef int excludeCount = 0
    cdef int index
    cdef list excludeBytes = []

    if excludeChromosomes is not None:
        excludeCount = len(excludeChromosomes)
        if excludeCount > 0:
            excludePointers = <const char**>malloc(excludeCount * sizeof(const char*))
            if excludePointers == NULL:
                raise MemoryError("failed to allocate exclude chromosome pointers")
            for index in range(excludeCount):
                excludeBytes.append(str(excludeChromosomes[index]).encode("utf-8"))
                excludePointers[index] = excludeBytes[index]

    try:
        result = ccounts_getMappedReadCount(
            &sourceConfig,
            threadCount,
            excludePointers,
            excludeCount,
            _getCountModeCode(countMode),
            oneReadPerBin,
            &mappedReadCount,
            &unmappedReadCount,
        )
        _raiseIfError(result)
    finally:
        if excludePointers != NULL:
            free(<void*>excludePointers)

    return int(mappedReadCount), int(unmappedReadCount)


cpdef int ccounts_getFragmentCellCount(
    str alignmentPath,
    str barcodeAllowListFile="",
):
    cdef bytes pathBytes = alignmentPath.encode("utf-8")
    cdef bytes barcodeAllowListBytes = barcodeAllowListFile.encode("utf-8")
    cdef ccounts_sourceConfig sourceConfig = _makeSourceConfig(
        pathBytes,
        "FRAGMENTS",
        b"",
        barcodeAllowListBytes,
    )
    cdef ccounts_result result
    cdef uint64_t cellCount = 0

    result = ccounts_getCellCount(&sourceConfig, &cellCount)
    _raiseIfError(result)
    return int(cellCount)


cpdef cnp.ndarray ccounts_countAlignmentRegion(
    str alignmentPath,
    str chromosome,
    int start,
    int end,
    int intervalSizeBP,
    int readLength,
    int oneReadPerBin,
    int threadCount,
    int flagExclude,
    int shiftForwardStrand53=0,
    int shiftReverseStrand53=0,
    int extendBP=0,
    int maxInsertSize=1000,
    int pairedEndMode=0,
    int inferFragmentLength=0,
    int minMappingQuality=0,
    int minTemplateLength=-1,
    str sourceKind="BAM",
    str referenceFASTA="",
    str barcodeAllowListFile="",
    str barcodeGroupMapFile="",
    str countMode="coverage",
):
    cdef int numIntervals
    cdef bytes pathBytes = alignmentPath.encode("utf-8")
    cdef bytes referenceBytes = referenceFASTA.encode("utf-8")
    cdef bytes barcodeAllowListBytes = barcodeAllowListFile.encode("utf-8")
    cdef bytes barcodeGroupMapBytes = barcodeGroupMapFile.encode("utf-8")
    cdef bytes chromosomeBytes = chromosome.encode("utf-8")
    cdef ccounts_sourceConfig sourceConfig = _makeSourceConfig(
        pathBytes,
        sourceKind,
        referenceBytes,
        barcodeAllowListBytes,
        barcodeGroupMapBytes,
    )
    cdef ccounts_sourceHandle* sourceHandle = NULL
    cdef ccounts_region region
    cdef ccounts_countOptions countOptions
    cdef ccounts_result result
    cdef cnp.ndarray[cnp.float32_t, ndim=1] counts

    if intervalSizeBP <= 0 or end <= start:
        raise ValueError("invalid interval size or genomic segment")

    numIntervals = ((end - start - 1) // intervalSizeBP) + 1
    counts = np.zeros(numIntervals, dtype=np.float32)

    region.chromosome = chromosomeBytes
    region.start = <uint32_t>start
    region.end = <uint32_t>end
    region.intervalSizeBP = <uint32_t>intervalSizeBP

    countOptions.threadCount = threadCount
    countOptions.flagExclude = flagExclude
    countOptions.countMode = _getCountModeCode(countMode)
    countOptions.oneReadPerBin = oneReadPerBin
    countOptions.shiftForwardStrand53 = shiftForwardStrand53
    countOptions.shiftReverseStrand53 = shiftReverseStrand53
    countOptions.readLength = readLength
    countOptions.extendBP = extendBP
    countOptions.minMappingQuality = minMappingQuality
    countOptions.minTemplateLength = minTemplateLength
    countOptions.maxInsertSize = maxInsertSize
    countOptions.pairedEndMode = pairedEndMode
    countOptions.inferFragmentLength = inferFragmentLength

    result = ccounts_openSource(&sourceConfig, &sourceHandle)
    _raiseIfError(result)
    try:
        result = ccounts_countRegion(
            sourceHandle,
            &region,
            &countOptions,
            &counts[0],
            numIntervals,
        )
        _raiseIfError(result)
    finally:
        if sourceHandle != NULL:
            ccounts_closeSource(sourceHandle)

    return counts
