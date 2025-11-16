# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
# distutils: language = c
r"""Cython module for Consenrich core functions.

This module contains Cython implementations of core functions used in Consenrich.
"""

cimport cython

import os
import numpy as np
from scipy import ndimage
import pysam

cimport numpy as cnp
from libc.stdint cimport int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from pysam.libcalignmentfile cimport AlignmentFile, AlignedSegment
from libc.float cimport DBL_EPSILON

cnp.import_array()

cpdef int stepAdjustment(int value, int stepSize, int pushForward=0):
    r"""Adjusts a value to the nearest multiple of stepSize, optionally pushing it forward.

    :param value: The value to adjust.
    :type value: int
    :param stepSize: The step size to adjust to.
    :type stepSize: int
    :param pushForward: If non-zero, pushes the value forward by stepSize
    :type pushForward: int
    :return: The adjusted value.
    :rtype: int
    """
    return max(0, (value-(value % stepSize))) + pushForward*stepSize


cpdef uint64_t cgetFirstChromRead(str bamFile, str chromosome, uint64_t chromLength, uint32_t samThreads, int samFlagExclude):
    r"""Get the start position of the first read in a BAM file for a given chromosome.

    :param bamFile: See :func:`consenrich.core.inputParams`.
    :type bamFile: str
    :param chromosome: Chromosome name.
    :type chromosome: str
    :param chromLength: Length of the chromosome in base pairs.
    :type chromLength: uint64_t
    :param samThreads: Number of threads to use for reading the BAM file.
    :type samThreads: uint32_t
    :param samFlagExclude: SAM flags to exclude reads (e.g., unmapped,
    :type samFlagExclude: int
    :return: Start position of the first read in the chromosome, or 0 if no reads are found.
    :rtype: uint64_t
    """

    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    for read in aln.fetch(contig=chromosome, start=0, end=chromLength):
        if not (read.flag & samFlagExclude):
            aln.close()
            return read.reference_start
    aln.close()
    return 0


cpdef uint64_t cgetLastChromRead(str bamFile, str chromosome, uint64_t chromLength, uint32_t samThreads, int samFlagExclude):
    r"""Get the end position of the last read in a BAM file for a given chromosome.

    :param bamFile: See :func:`consenrich.core.inputParams`.
    :type bamFile: str
    :param chromosome: Chromosome name.
    :type chromosome: str
    :param chromLength: Length of the chromosome in base pairs.
    :type chromLength: uint64_t
    :param samThreads: Number of threads to use for reading the BAM file.
    :type samThreads: uint32_t
    :param samFlagExclude: See :class:`consenrich.core.samParams`.
    :type samFlagExclude: int
    :return: End position of the last read in the chromosome, or 0 if no reads are found.
    :rtype: uint64_t
    """

    cdef uint64_t start_ = chromLength - min((chromLength // 2), 1_000_000)
    cdef uint64_t lastPos = 0
    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    for read in aln.fetch(contig=chromosome, start=start_, end=chromLength):
        if not (read.flag & samFlagExclude):
            lastPos = read.reference_end
    aln.close()
    return lastPos



cpdef uint32_t cgetReadLength(str bamFile, uint32_t minReads, uint32_t samThreads, uint32_t maxIterations, int samFlagExclude):
    r"""Get the median read length from a BAM file after fetching a specified number of reads.

    :param bamFile: see :class:`consenrich.core.inputParams`.
    :type bamFile: str
    :param minReads: Minimum number of reads to consider for the median calculation.
    :type minReads: uint32_t
    :param samThreads: See :class:`consenrich.core.samParams`.
    :type samThreads: uint32_t
    :param maxIterations: Maximum number of reads to iterate over.
    :type maxIterations: uint32_t
    :param samFlagExclude: See :class:`consenrich.core.samParams`.
    :type samFlagExclude: int
    :return: Median read length from the BAM file.
    :rtype: uint32_t
    """
    cdef uint32_t observedReads = 0
    cdef uint32_t currentIterations = 0
    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] readLengths = np.zeros(maxIterations, dtype=np.uint32)
    cdef uint32_t i = 0
    if <uint32_t>aln.mapped < minReads:
        aln.close()
        return 0
    for read in aln.fetch():
        if not (observedReads < minReads and currentIterations < maxIterations):
            break
        if not (read.flag & samFlagExclude):
            # meets critera -> add it
            readLengths[i] = read.query_length
            observedReads += 1
            i += 1
        currentIterations += 1
    aln.close()
    if observedReads < minReads:
        return 0
    return <uint32_t>np.median(readLengths[:observedReads])


cdef inline Py_ssize_t floordiv64(int64_t a, int64_t b) nogil:
    if a >= 0:
        return <Py_ssize_t>(a // b)
    else:
        return <Py_ssize_t>(- ((-a + b - 1) // b))


cpdef cnp.uint32_t[:] creadBamSegment(
    str bamFile,
    str chromosome,
    uint32_t start,
    uint32_t end,
    uint32_t stepSize,
    int64_t readLength,
    uint8_t oneReadPerBin,
    uint16_t samThreads,
    uint16_t samFlagExclude,
    int64_t shiftForwardStrand53 = 0,
    int64_t shiftReverseStrand53 = 0,
    int64_t extendBP = 0,
    int64_t maxInsertSize=1000,
    int64_t pairedEndMode=0,
    int64_t inferFragmentLength=0):
    r"""Count reads in a BAM file for a given chromosome"""

    cdef Py_ssize_t numIntervals = <Py_ssize_t>(((end - start) + stepSize - 1) // stepSize)

    cdef cnp.ndarray[cnp.uint32_t, ndim=1] values_np = np.zeros(numIntervals, dtype=np.uint32)
    cdef cnp.uint32_t[::1] values = values_np

    if numIntervals <= 0:
        return values

    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    cdef int64_t start64 = start
    cdef int64_t end64 = end
    cdef int64_t step64 = stepSize
    cdef Py_ssize_t i, index0, index1
    cdef Py_ssize_t lastIndex = numIntervals - 1
    cdef bint readIsForward
    cdef int64_t readStart, readEnd
    cdef int64_t adjStart, adjEnd, fivePrime, mid, midIndex, tlen, atlen
    cdef uint16_t flag
    if inferFragmentLength > 0 and pairedEndMode == 0 and extendBP == 0:
        extendBP = cgetFragmentLength(bamFile,
         chromosome,
         <int64_t>start,
         <int64_t>end,
         samThreads = samThreads,
         samFlagExclude=samFlagExclude,
         maxInsertSize=maxInsertSize,
         minInsertSize=<int64_t>(readLength+1), # xCorr peak > rlen ~~> fraglen
         )
    try:
        with aln:
            for read in aln.fetch(chromosome, start64, end64):
                flag = <uint16_t>read.flag
                if flag & samFlagExclude:
                    continue

                readIsForward = (flag & 16) == 0
                readStart = <int64_t>read.reference_start
                readEnd   = <int64_t>read.reference_end

                if pairedEndMode > 0:
                    if flag & 1 == 0: # not a paired read
                        continue
                    # use first in pair + fragment
                    if flag & 128:
                        continue
                    if (flag & 8) or read.next_reference_id != read.reference_id:
                        continue
                    tlen = <int64_t>read.template_length
                    atlen = tlen if tlen >= 0 else -tlen
                    if atlen == 0 or atlen > maxInsertSize:
                        continue
                    if tlen >= 0:
                        adjStart = readStart
                        adjEnd   = readStart + atlen
                    else:
                        adjEnd   = readEnd
                        adjStart = adjEnd - atlen
                    if shiftForwardStrand53 != 0 or shiftReverseStrand53 != 0:
                        if readIsForward:
                            adjStart += shiftForwardStrand53
                            adjEnd   += shiftForwardStrand53
                        else:
                            adjStart -= shiftReverseStrand53
                            adjEnd   -= shiftReverseStrand53
                else:
                    # SE
                    if readIsForward:
                        fivePrime = readStart + shiftForwardStrand53
                    else:
                        fivePrime = (readEnd - 1) - shiftReverseStrand53

                    if extendBP > 0:
                        # from the cut 5' --> 3'
                        if readIsForward:
                            adjStart = fivePrime
                            adjEnd   = fivePrime + extendBP
                        else:
                            adjEnd   = fivePrime + 1
                            adjStart = adjEnd - extendBP
                    elif shiftForwardStrand53 != 0 or shiftReverseStrand53 != 0:
                        if readIsForward:
                            adjStart = readStart + shiftForwardStrand53
                            adjEnd   = readEnd   + shiftForwardStrand53
                        else:
                            adjStart = readStart - shiftReverseStrand53
                            adjEnd   = readEnd   - shiftReverseStrand53
                    else:
                        adjStart = readStart
                        adjEnd   = readEnd

                if adjEnd <= start64 or adjStart >= end64:
                    continue
                if adjStart < start64:
                    adjStart = start64
                if adjEnd > end64:
                    adjEnd = end64

                if oneReadPerBin:
                    # +1 at midpoint of frag.
                    mid = (adjStart + adjEnd) // 2
                    midIndex = <Py_ssize_t>((mid - start64) // step64)
                    if 0 <= midIndex <= lastIndex:
                        values[midIndex] += <uint32_t>1
                else:
                    # +1 every interval intersecting frag
                    index0 = <Py_ssize_t>((adjStart - start64) // step64)
                    index1 = <Py_ssize_t>(((adjEnd - 1) - start64) // step64)
                    if index0 < 0:
                        index0 = <Py_ssize_t>0
                    if index1 > lastIndex:
                        index1 = lastIndex
                    if index0 > lastIndex or index1 < 0 or index0 > index1:
                        continue
                    for b_ in range(index0, index1 + 1):
                        values[b_] += <uint32_t>1

    finally:
        aln.close()

    return values



cpdef cnp.ndarray[cnp.float32_t, ndim=2] cinvertMatrixE(cnp.ndarray[cnp.float32_t, ndim=1] muncMatrixIter, cnp.float32_t priorCovarianceOO):
    r"""Invert the residual covariance matrix during the forward pass.

    :param muncMatrixIter: The diagonal elements of the covariance matrix at a given genomic interval.
    :type muncMatrixIter: cnp.ndarray[cnp.float32_t, ndim=1]
    :param priorCovarianceOO: The a priori 'primary' state variance :math:`P_{[i|i-1,11]}`.
    :type priorCovarianceOO: cnp.float32_t
    :return: The inverted covariance matrix.
    :rtype: cnp.ndarray[cnp.float32_t, ndim=2]
    """

    cdef int m = muncMatrixIter.size
    # we have to invert a P.D. covariance (diagonal) and rank-one (1*priorCovariance) matrix
    cdef cnp.ndarray[cnp.float32_t, ndim=2] inverse = np.empty((m, m), dtype=np.float32)
    # note, not actually an m-dim matrix, just the diagonal elements taken as input
    cdef cnp.ndarray[cnp.float32_t, ndim=1] muncMatrixInverse = np.empty(m, dtype=np.float32)
    cdef float sqrtPrior = np.sqrt(priorCovarianceOO)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] uVec = np.empty(m, dtype=np.float32)
    cdef float divisor = 1.0
    cdef float scale
    cdef float uVecI
    cdef Py_ssize_t i, j
    for i in range(m):
        # two birds: build up the trace while taking the reciprocals
        muncMatrixInverse[i] = 1.0/(muncMatrixIter[i])
        divisor += priorCovarianceOO*muncMatrixInverse[i]
    # we can combine these two loops, keeping construction
    # of muncMatrixInverse and uVec separate for now in case
    # we want to parallelize this later
    for i in range(m):
        uVec[i] = sqrtPrior*muncMatrixInverse[i]
    scale = 1.0 / divisor
    for i in range(m):
        uVecI = uVec[i]
        inverse[i, i] = muncMatrixInverse[i]-(scale*uVecI*uVecI)
        for j in range(i + 1, m):
            inverse[i, j] = -scale * (uVecI*uVec[j])
            inverse[j, i] = inverse[i, j]
    return inverse


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cgetStateCovarTrace(
    cnp.float32_t[:, :, ::1] stateCovarMatrices
):
    cdef Py_ssize_t n = stateCovarMatrices.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] trace = np.empty(n, dtype=np.float32)
    cdef cnp.float32_t[::1] traceView = trace
    cdef Py_ssize_t i
    for i in range(n):
        traceView[i] = stateCovarMatrices[i, 0, 0] + stateCovarMatrices[i, 1, 1]

    return trace


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cgetPrecisionWeightedResidual(
    cnp.float32_t[:, ::1] postFitResiduals,
    cnp.float32_t[:, ::1] matrixMunc,
):
    cdef Py_ssize_t n = postFitResiduals.shape[0]
    cdef Py_ssize_t m = postFitResiduals.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)
    cdef cnp.float32_t[::1] outv = out
    cdef Py_ssize_t i, j
    cdef float wsum, rwsum, w
    cdef float eps = 1e-12  # guard for zeros

    for i in range(n):
        wsum = 0.0
        rwsum = 0.0
        for j in range(m):
            w = 1.0 / (<float>matrixMunc[j, i] + eps)   # weightsIter[j]
            rwsum += (<float>postFitResiduals[i, j]) * w  # residualsIter[j] * w
            wsum  += w
        outv[i] = <cnp.float32_t>(rwsum / wsum) if wsum > 0.0 else <cnp.float32_t>0.0

    return out



cpdef tuple updateProcessNoiseCovariance(cnp.ndarray[cnp.float32_t, ndim=2] matrixQ,
        cnp.ndarray[cnp.float32_t, ndim=2] matrixQCopy,
        float dStat,
        float dStatAlpha,
        float dStatd,
        float dStatPC,
        bint inflatedQ,
        float maxQ,
        float minQ):
    r"""Adjust process noise covariance matrix :math:`\mathbf{Q}_{[i]}`

    :param matrixQ: Current process noise covariance
    :param matrixQCopy: A copy of the initial original covariance matrix :math:`\mathbf{Q}_{[.]}`
    :param inflatedQ: Flag indicating if the process noise covariance is inflated
    :return: Updated process noise covariance matrix and inflated flag
    :rtype: tuple
    """

    cdef float scaleQ, fac
    if dStat > dStatAlpha:
        scaleQ = np.sqrt(dStatd * np.abs(dStat-dStatAlpha) + dStatPC)
        if matrixQ[0, 0] * scaleQ <= maxQ:
            matrixQ[0, 0] *= scaleQ
            matrixQ[0, 1] *= scaleQ
            matrixQ[1, 0] *= scaleQ
            matrixQ[1, 1] *= scaleQ
        else:
            fac = maxQ / matrixQCopy[0, 0]
            matrixQ[0, 0] = maxQ
            matrixQ[0, 1] = matrixQCopy[0, 1] * fac
            matrixQ[1, 0] = matrixQCopy[1, 0] * fac
            matrixQ[1, 1] = maxQ
        inflatedQ = True

    elif dStat < dStatAlpha and inflatedQ:
        scaleQ = np.sqrt(dStatd * np.abs(dStat-dStatAlpha) + dStatPC)
        if matrixQ[0, 0] / scaleQ >= minQ:
            matrixQ[0, 0] /= scaleQ
            matrixQ[0, 1] /= scaleQ
            matrixQ[1, 0] /= scaleQ
            matrixQ[1, 1] /= scaleQ
        else:
            # we've hit the minimum, no longer 'inflated'
            fac = minQ / matrixQCopy[0, 0]
            matrixQ[0, 0] = minQ
            matrixQ[0, 1] = matrixQCopy[0, 1] * fac
            matrixQ[1, 0] = matrixQCopy[1, 0] * fac
            matrixQ[1, 1] = minQ
            inflatedQ = False
    return matrixQ, inflatedQ


cdef void _blockMax(double[::1] valuesView,
                    Py_ssize_t[::1] blockStartIndices,
                    Py_ssize_t[::1] blockSizes,
                    double[::1] outputView,
                    double eps = 0.0) noexcept:
    cdef Py_ssize_t iterIndex, elementIndex, startIndex, blockLength
    cdef double currentMax, currentValue
    cdef Py_ssize_t firstIdx, lastIdx, centerIdx

    for iterIndex in range(outputView.shape[0]):
        startIndex = blockStartIndices[iterIndex]
        blockLength = blockSizes[iterIndex]

        currentMax = valuesView[startIndex]
        for elementIndex in range(1, blockLength):
            currentValue = valuesView[startIndex + elementIndex]
            if currentValue > currentMax:
                currentMax = currentValue

        firstIdx = -1
        lastIdx = -1
        if eps > 0.0:
            # only run if eps tol is non-zero
            for elementIndex in range(blockLength):
                currentValue = valuesView[startIndex + elementIndex]
                # NOTE: this is intended to mirror the +- eps tol
                if currentValue >= currentMax - eps:
                    if firstIdx == -1:
                        firstIdx = elementIndex
                    lastIdx = elementIndex

        if firstIdx == -1:
            # case: we didn't find a tie or eps == 0
            outputView[iterIndex] = currentMax
        else:
            # case: there's a tie for eps > 0, pick center
            centerIdx = (firstIdx + lastIdx) // 2
            outputView[iterIndex] = valuesView[startIndex + centerIdx]


cpdef double[::1] csampleBlockStats(cnp.ndarray[cnp.uint32_t, ndim=1] intervals,
                        cnp.ndarray[cnp.float64_t, ndim=1] values,
                        int expectedBlockSize,
                        int iters,
                        int randSeed,
                        cnp.ndarray[cnp.uint8_t, ndim=1] excludeIdxMask,
                        double eps = 0.0):
    r"""Sample contiguous blocks in the response sequence (xCorr), record maxima, and repeat.

    Used to build an empirical null distribution and determine significance of response outputs.
    The size of blocks is drawn from a truncated geometric distribution, preserving rough equality
    in expectation but allowing for variability to account for the sampling across different phases
    in the response sequence.

    :param values: The response sequence to sample from.
    :type values: cnp.ndarray[cnp.float64_t, ndim=1]
    :param expectedBlockSize: The expected size (geometric) of the blocks to sample.
    :type expectedBlockSize: int
    :param iters: The number of blocks to sample.
    :type iters: int
    :param randSeed: Random seed for reproducibility.
    :type randSeed: int
    :return: An array of sampled block maxima.
    :rtype: cnp.ndarray[cnp.float64_t, ndim=1]
    :seealso: :func:`consenrich.matching.matchWavelet`
    """
    np.random.seed(randSeed)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] valuesArr = np.ascontiguousarray(values, dtype=np.float64)
    cdef double[::1] valuesView = valuesArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1] sizesArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1] startsArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(iters, dtype=np.float64)
    cdef Py_ssize_t maxBlockLength, maxSize, minSize
    cdef Py_ssize_t n = <Py_ssize_t>intervals.size
    cdef double maxBlockScale = <double>3.0
    cdef double minBlockScale = <double> (1.0 / 3.0)

    minSize = <Py_ssize_t> max(3, expectedBlockSize * minBlockScale)
    maxSize = <Py_ssize_t> min(maxBlockScale * expectedBlockSize, n)
    sizesArr = np.random.geometric(1.0 / expectedBlockSize, size=iters).astype(np.intp, copy=False)
    np.clip(sizesArr, minSize, maxSize, out=sizesArr)
    maxBlockLength = sizesArr.max()
    cdef list support = []
    cdef cnp.intp_t i_ = 0
    while i_ < n-maxBlockLength:
        if excludeIdxMask[i_:i_ + maxBlockLength].any():
            i_ = i_ + maxBlockLength + 1
            continue
        support.append(i_)
        i_ = i_ + 1

    cdef cnp.ndarray[cnp.intp_t, ndim=1] samples = np.random.choice(
        support,
        size=iters,
        replace=True,
        p=None
        ).astype(np.intp)

    cdef Py_ssize_t[::1] startsView = samples
    cdef Py_ssize_t[::1] sizesView = sizesArr
    cdef double[::1] outView = out
    _blockMax(valuesView, startsView, sizesView, outView, eps)
    return out


cpdef cSparseAvg(cnp.float32_t[::1] trackALV, dict sparseMap):
    r"""Fast access and average of `numNearest` sparse elements.

    See :func:`consenrich.core.getMuncTrack`

    :param trackALV: See :func:`consenrich.core.getAverageLocalVarianceTrack`
    :type trackALV: float[::1]
    :param sparseMap: See :func:`consenrich.core.getSparseMap`
    :type sparseMap: dict[int, np.ndarray]
    :return: array of mena('nearest local variances') same length as `trackALV`
    :rtype: cnp.ndarray[cnp.float32_t, ndim=1]
    """
    cdef Py_ssize_t n = <Py_ssize_t>trackALV.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)
    cdef Py_ssize_t i, j, m
    cdef float sumNearestVariances = 0.0
    cdef cnp.ndarray[cnp.intp_t, ndim=1] idxs
    cdef cnp.intp_t[::1] idx_view
    for i in range(n):
        idxs = <cnp.ndarray[cnp.intp_t, ndim=1]> sparseMap[i] # FFR: to avoid the cast, create sparseMap as dict[intp, np.ndarray[intp]]
        idx_view = idxs
        m = idx_view.shape[0] # FFR: maybe enforce strict `m == numNearest` in future releases to avoid extra overhead
        if m == 0:
            # this case probably warrants an exception or np.nan
            out[i] = 0.0
            continue
        sumNearestVariances = 0.0
        with nogil:
            for j in range(m):
                sumNearestVariances += trackALV[idx_view[j]]
        out[i] = sumNearestVariances/m

    return out


cpdef int64_t cgetFragmentLength(
    str bamFile,
    str chromosome,
    int64_t start,
    int64_t end,
    uint16_t samThreads=0,
    uint16_t samFlagExclude=3844,
    int64_t maxInsertSize=1000,
    int64_t minInsertSize=50,
    int64_t iters=1000,
    int64_t blockSize=5000,
    int64_t fallBack=147,
    int64_t rollingChunkSize=250,
    int64_t lagStep=5,
    int64_t earlyExit=100,
    int64_t randSeed=42,
):
    np.random.seed(randSeed)
    cdef int64_t regionLen, numRollSteps, numChunks
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rawArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] medArr
    cdef AlignmentFile aln
    cdef AlignedSegment read_
    cdef list coverageIdxTopK
    cdef list blockCenters
    cdef list bestLags
    cdef int i, j, k, idxVal
    cdef int startIdx, endIdx
    cdef int winSize, takeK
    cdef int blockHalf, readFlag
    cdef int chosenLag, lag, maxValidLag
    cdef int64_t blockStartBP, blockEndBP, readStart, readEnd, med
    cdef double score
    cdef cnp.ndarray[cnp.intp_t, ndim=1] unsortedIdx, sortedIdx, expandedIdx
    cdef cnp.intp_t[::1] expandedIdxView
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unsortedVals
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] seen
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fwd = np.zeros(blockSize, dtype=np.float64, order='C')
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rev = np.zeros(blockSize, dtype=np.float64, order='C')
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fwdDiff = np.zeros(blockSize + 1, dtype=np.float64, order='C')
    cdef cnp.ndarray[cnp.float64_t, ndim=1] revDiff = np.zeros(blockSize + 1, dtype=np.float64, order='C')
    cdef int64_t diffS, diffE = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xCorr
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] bestLagsArr

    earlyExit = min(earlyExit, iters)
    regionLen = end - start

    samThreads_ = <int>samThreads
    numCPUs = os.cpu_count()
    if numCPUs is None:
        numCPUs = 1
    if samThreads < 1:
        samThreads_ = <int>min(max(1,numCPUs // 2), 4)

    if regionLen < blockSize or regionLen <= 0:
        return <int64_t>fallBack

    if maxInsertSize < 1:
        maxInsertSize = 1
    if minInsertSize < 1:
        minInsertSize = 1
    if minInsertSize > maxInsertSize:
        minInsertSize, maxInsertSize = maxInsertSize, minInsertSize

    # first, we build a coarse read coverage track from `start` to `end`
    numRollSteps = regionLen // rollingChunkSize
    if numRollSteps <= 0:
        numRollSteps = 1
    numChunks = <int>numRollSteps

    rawArr = np.zeros(numChunks, dtype=np.float64)
    medArr = np.zeros(numChunks, dtype=np.float64)

    aln = AlignmentFile(bamFile, "rb", threads=samThreads_)
    try:
        for read_ in aln.fetch(chromosome, start, end):
            if (read_.flag & samFlagExclude) != 0:
                continue
            if read_.reference_start < start or read_.reference_end >= end:
                continue
            j = <int>((read_.reference_start - start) // rollingChunkSize)
            if 0 <= j < numChunks:
                rawArr[j] += 1.0

        # second, we apply a rolling/moving/local/weywtci order-statistic filter (median)
        # ...the size of the kernel is based on the blockSize -- we want high-coverage
        # ...blocks as measured by their local median read counts
        winSize = <int>(blockSize // rollingChunkSize)
        if winSize < 1:
            winSize = 1
        if (winSize & 1) == 0:
            winSize += 1
        medArr[:] = ndimage.median_filter(rawArr, size=winSize, mode="nearest")

        # we pick the largest local-medians and form a block around each
        takeK = iters if iters < numChunks else numChunks
        unsortedIdx = np.argpartition(medArr, -takeK)[-takeK:]
        unsortedVals = medArr[unsortedIdx]
        sortedIdx = unsortedIdx[np.argsort(unsortedVals)[::-1]]
        coverageIdxTopK = sortedIdx[:takeK].tolist()
        expandedLen = takeK*winSize
        expandedIdx = np.empty(expandedLen, dtype=np.intp)
        expandedIdxView = expandedIdx
        k = 0
        for i in range(takeK):
            idxVal = coverageIdxTopK[i]
            startIdx = idxVal - (winSize // 2)
            endIdx = startIdx + winSize
            if startIdx < 0:
                startIdx = 0
                endIdx = winSize if winSize < numChunks else numChunks
            if endIdx > numChunks:
                endIdx = numChunks
                startIdx = endIdx - winSize if winSize <= numChunks else 0
            for j in range(startIdx, endIdx):
                expandedIdxView[k] = j
                k += 1
        if k < expandedLen:
            expandedIdx = expandedIdx[:k]
            expandedIdxView = expandedIdx

        seen = np.zeros(numChunks, dtype=np.uint8)
        blockCenters = []
        for i in range(expandedIdx.shape[0]):
            j = <int>expandedIdxView[i]
            if seen[j] == 0:
                seen[j] = 1
                blockCenters.append(j)

        if len(blockCenters) > 1:
            blockCenters = np.random.choice(
                blockCenters,
                size=len(blockCenters),
                replace=False,
                p=None
            ).tolist()

        bestLags = []
        blockHalf = blockSize // 2

        for idxVal in blockCenters:
            # this should map back to genomic coordinates
            blockStartBP = start + idxVal * rollingChunkSize + (rollingChunkSize // 2) - blockHalf
            if blockStartBP < start:
                blockStartBP = start
            blockEndBP = blockStartBP + blockSize
            if blockEndBP > end:
                blockEndBP = end
                blockStartBP = blockEndBP - blockSize
                if blockStartBP < start:
                    continue

            # now we build strand-specific tracks over each block
            fwd.fill(0.0)
            fwdDiff.fill(0.0)
            rev.fill(0.0)
            revDiff.fill(0.0)
            readFlag = -1

            for read_ in aln.fetch(chromosome, blockStartBP, blockEndBP):
                readFlag = read_.flag
                if (readFlag & samFlagExclude) != 0:
                    continue
                readStart = min(read_.reference_start, read_.reference_end)
                readEnd = max(read_.reference_start, read_.reference_end)
                diffS = readStart - blockStartBP
                diffE = readEnd - blockStartBP
                strand = readFlag & 16
                if readStart < blockStartBP or readEnd > blockEndBP:
                    continue
                posInBlock = readStart - blockStartBP
                if strand == 0:
                    # forward
                    fwdDiff[diffS] += 1.0
                    fwdDiff[diffE] -= 1.0
                else:
                    # reverse
                    revDiff[diffS] += 1.0
                    revDiff[diffE] -= 1.0
            fwd = np.cumsum(fwdDiff[:len(fwdDiff)-1], dtype=np.float64)
            rev = np.cumsum(revDiff[:len(revDiff)-1], dtype=np.float64)

            # maximizes the crossCovar(forward, reverse, lag) wrt lag.
            fwd = (fwd - np.mean(fwd))
            rev = (rev - np.mean(rev))
            maxValidLag = maxInsertSize if (maxInsertSize < blockSize) else (blockSize - 1)
            xCorr = np.zeros(maxInsertSize + 1, dtype=np.float64)
            for lag in range(minInsertSize, maxValidLag + 1, lagStep):
                score = 0.0
                for i in range(blockSize - lag):
                    score += fwd[i] * rev[i + lag]
                xCorr[lag] = score

            if maxValidLag < minInsertSize:
                continue
            chosenLag = -1
            chosenLag = int(np.argmax(xCorr[minInsertSize:maxValidLag + 1])) + minInsertSize

            if chosenLag > 0:
                bestLags.append(chosenLag)
            if len(bestLags) >= earlyExit:
                break

    finally:
        aln.close()

    if len(bestLags) < 3:
        return fallBack

    bestLagsArr = np.asarray(bestLags, dtype=np.uint32)
    med = int(np.median(bestLagsArr))
    if med < minInsertSize:
        med = <int>minInsertSize
    elif med > maxInsertSize:
        med = <int>maxInsertSize
    return med


cdef inline Py_ssize_t getInsertion(const uint32_t* array_, Py_ssize_t n, uint32_t x) nogil:
    # helper: binary search to find insertion point into sorted `arrray_`
    cdef Py_ssize_t low = 0
    cdef Py_ssize_t high = n
    cdef Py_ssize_t midpt
    while low < high:
        midpt = low + ((high - low) >> 1)
        if array_[midpt] <= x:
            low = midpt + 1
        else:
            high = midpt
    return low


cdef int maskMembership(const uint32_t* pos, Py_ssize_t numIntervals, const uint32_t* mStarts, const uint32_t* mEnds, Py_ssize_t n, uint8_t* outMask) nogil:
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t k
    cdef uint32_t p
    while i < numIntervals:
        p = pos[i]
        k = getInsertion(mStarts, n, p) - 1
        if k >= 0 and p < mEnds[k]:
            outMask[i] = <uint8_t>1
        else:
            outMask[i] = <uint8_t>0
        i += 1
    return 0


cpdef cnp.ndarray[cnp.uint8_t, ndim=1] cbedMask(
    str chromosome,
    str bedFile,
    cnp.ndarray[cnp.uint32_t, ndim=1] intervals,
    int stepSize
    ):
    r"""Return a 1/0 mask for intervals overlapping a sorted and merged BED file.

    :param chromosome: Chromosome name.
    :type chromosome: str
    :param bedFile: Path to a sorted and merged BED file.
    :type bedFile: str
    :param intervals: Array of sorted, non-overlapping start positions of genomic intervals.
      Each interval is assumed `stepSize`.
    :type intervals: cnp.ndarray[cnp.uint32_t, ndim=1]
    :param stepSize: Step size between genomic positions in `intervals`.
    :type stepSize: int32_t
    :return: A mask s.t. `1` indicates the corresponding interval overlaps a BED region.
    :rtype: cnp.ndarray[cnp.uint8_t, ndim=1]

    """
    cdef list startsList = []
    cdef list endsList = []
    cdef object f = open(bedFile, "r")
    cdef str line
    cdef list cols
    try:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            cols = line.split('\t')
            if not cols or len(cols) < 3:
                continue
            if cols[0] != chromosome:
                continue
            startsList.append(int(cols[1]))
            endsList.append(int(cols[2]))
    finally:
        f.close()
    cdef Py_ssize_t numIntervals = intervals.size
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mask = np.zeros(numIntervals, dtype=np.uint8)
    if not startsList:
        return mask
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] starts = np.asarray(startsList, dtype=np.uint32)
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] ends = np.asarray(endsList, dtype=np.uint32)
    cdef cnp.uint32_t[:] startsView = starts
    cdef cnp.uint32_t[:] endsView = ends
    cdef cnp.uint32_t[:] posView = intervals
    cdef cnp.uint8_t[:] outView = mask
    # for nogil
    cdef uint32_t* svPtr = &startsView[0] if starts.size > 0 else <uint32_t*>NULL
    cdef uint32_t* evPtr = &endsView[0] if ends.size > 0 else <uint32_t*>NULL
    cdef uint32_t* posPtr = &posView[0] if numIntervals > 0 else <uint32_t*>NULL
    cdef uint8_t* outPtr = &outView[0] if numIntervals > 0 else <uint8_t*>NULL
    cdef Py_ssize_t n = starts.size
    with nogil:
        if numIntervals > 0 and n > 0:
            maskMembership(posPtr, numIntervals, svPtr, evPtr, n, outPtr)
    return mask
