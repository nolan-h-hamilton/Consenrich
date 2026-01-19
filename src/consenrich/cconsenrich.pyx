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
cimport numpy as cnp
from libc.stdint cimport int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from pysam.libcalignmentfile cimport AlignmentFile, AlignedSegment
from numpy.random import default_rng
from cython.parallel import prange
from libc.math cimport isfinite, fabs, log1p, log2, log, log2f, logf, asinhf, asinh, fmax, fmaxf, pow, sqrt, sqrtf, fabsf, fminf, fmin, log10, log10f, ceil, floor, floorf, exp, expf, isnan, NAN, INFINITY
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.string cimport memcpy
from libc.stdio cimport printf
cnp.import_array()

# ========
# constants
# ========
cdef const float __INV_LN2_FLOAT = <float>1.44269504
cdef const double __INV_LN2_DOUBLE = <double>1.44269504

# ===============
# inline/helpers
# ===============

cdef inline Py_ssize_t _getInsertion(const uint32_t* array_, Py_ssize_t n, uint32_t x) nogil:
    # CALLERS: `_maskMembership`, `cbedMask`

    cdef Py_ssize_t low = 0
    cdef Py_ssize_t high = n
    cdef Py_ssize_t midpt
    while low < high:
        # [low,x1,x2,x3,...,(high-low)//2,...,xn-2, high]
        # [(high-low)//2 + 1,...,xn-2, high]
        midpt = low + ((high - low) >> 1)
        if array_[midpt] <= x:
            low = midpt + 1
        # [low,x1,x2,x3,...,(high-low)//2,...,xn-2, high]
        # [low,x1,x2,x3,...,(high-low)//2]
        else:
            high = midpt
    # array_[low] <= x* < array_[low+1]
    return low


cdef inline int _maskMembership(const uint32_t* pos, Py_ssize_t numIntervals, const uint32_t* mStarts, const uint32_t* mEnds, Py_ssize_t n, uint8_t* outMask) nogil:
    # CALLERS: `cbedMask`

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t k
    cdef uint32_t p
    while i < numIntervals:
        p = pos[i]
        k = _getInsertion(mStarts, n, p) - 1
        if k >= 0 and p < mEnds[k]:
            outMask[i] = <uint8_t>1
        else:
            outMask[i] = <uint8_t>0
        i += 1
    return 0


cdef inline bint _projectToBox(
    cnp.float32_t[::1] vectorX,
    cnp.float32_t[:, ::1] matrixP,
    cnp.float32_t stateLowerBound,
    cnp.float32_t stateUpperBound,
    cnp.float32_t eps
) nogil:
    # CALLERS: `projectToBox`

    cdef cnp.float32_t initX_i0
    cdef cnp.float32_t projectedX_i0
    cdef cnp.float32_t P00
    cdef cnp.float32_t P10
    cdef cnp.float32_t P11
    cdef cnp.float32_t padded_P00
    cdef cnp.float32_t newP11

    # Note, the following is straightforward algebraically, but some hand-waving here
    # ... for future reference if I forget the intuition/context later on or somebody
    # ... wants to change/debug. Essentially, finding a point in the feasible region
    # ... that minimizes -weighted- distance to the unconstrained/infeasible solution.
    # ... Weighting is determined by inverse state covariance P^{-1}_[i]
    # ... So a WLS-like QP:
    # ...   argmin (x^{*}_[i] - x^{unconstrained}_[i])^T (P^-1_{[i]}) (x^{*}_[i] - x^{unconstrained}_[i])
    # ...   such that: lower <= x^{*}_[i,0] <= upper
    # ... in our case (single-variable in box), solution is a simle truncation
    # ... with a corresponding scaled-update to x_[i,1] based on their covariance
    # ... REFERENCE: Simon, 2006 (IET survey paper on constrained linear filters)

    initX_i0 = vectorX[0]

    if initX_i0 >= stateLowerBound and initX_i0 <= stateUpperBound:
        return <bint>0 # no change if in bounds

    # projection in our case --> truncated box on first state variable
    projectedX_i0 = initX_i0
    if projectedX_i0 < stateLowerBound:
        projectedX_i0 = stateLowerBound
    if projectedX_i0 > stateUpperBound:
        projectedX_i0 = stateUpperBound

    P00 = matrixP[0, 0]
    P10 = matrixP[1, 0]
    P11 = matrixP[1, 1]
    padded_P00 = P00 if P00 > eps else eps

    # FIRST, adjust second state according to its original value + an update
    # ... given the covariance between first,second variables that
    # ... is scaled by the size of projection in the first state
    vectorX[1] = <cnp.float32_t>(vectorX[1] + (P10 / padded_P00)*(projectedX_i0 - initX_i0))

    # SECOND, now we set the projected first state variable
    # ...  and the second state's variance
    vectorX[0] = projectedX_i0
    newP11 = <cnp.float32_t>(P11 - (P10*P10) / padded_P00)

    matrixP[0, 0] = eps
    matrixP[0, 1] = <cnp.float32_t>0.0 # first state fixed --> covar = 0
    matrixP[1, 0] = <cnp.float32_t>0.0
    matrixP[1, 1] = newP11 if newP11 > eps else eps

    return <bint>1


cdef inline void _regionMeanVar(double[::1] valuesView,
                                Py_ssize_t[::1] blockStartIndices,
                                Py_ssize_t[::1] blockSizes,
                                float[::1] meanOutView,
                                float[::1] varOutView,
                                double zeroPenalty,
                                double zeroThresh,
                                bint useInnovationVar,
                                bint useSampleVar,
                                double maxBeta=<double>0.99,
                                double pairsRegLambda=<double>0.1) noexcept nogil:
    # CALLERS: cmeanVarPairs

    cdef Py_ssize_t regionIndex, elementIndex, startIndex, blockLength
    cdef double value
    cdef double sumY
    cdef double sumSqX
    cdef double blockLengthDouble
    cdef double mom1
    cdef double* blockPtr
    cdef double eps
    cdef double nPairsDouble
    cdef double sumXSeq
    cdef double sumYSeq
    cdef double meanX
    cdef double meanYp
    cdef double sumSqXSeq
    cdef double sumSqYSeq
    cdef double sumXYc
    cdef double xDev
    cdef double yDev
    cdef double beta1
    cdef double RSS
    cdef double pairCountDouble
    cdef double oneMinusBetaSq
    cdef double divRSS
    zeroPenalty = zeroPenalty
    zeroThresh = zeroThresh

    for regionIndex in range(meanOutView.shape[0]):
        startIndex = blockStartIndices[regionIndex]
        blockLength = blockSizes[regionIndex]
        blockPtr = &valuesView[startIndex]
        blockLengthDouble = <double>blockLength

        # mean over full block
        sumY = 0.0
        for elementIndex in range(blockLength):
            sumY += blockPtr[elementIndex]
        mom1 = sumY / blockLengthDouble
        meanOutView[regionIndex] = <float>mom1
        if useSampleVar:
            # sample variance over full block around mom1
            sumSqX = 0.0
            for elementIndex in range(blockLength):
                value = blockPtr[elementIndex] - mom1
                sumSqX += value*value
            varOutView[regionIndex] = <float>(sumSqX / (blockLengthDouble - 1.0))
            continue

        # df = n-3
        if blockLength < 4:
            varOutView[regionIndex] = 0.0
            continue

        nPairsDouble = <double>(blockLength - 1)
        sumXSeq = sumY - blockPtr[blockLength - 1] # drop last
        sumYSeq = sumY - blockPtr[0] # drop first

        meanX = sumXSeq / nPairsDouble
        meanYp = sumYSeq / nPairsDouble
        sumSqXSeq = 0.0
        sumSqYSeq = 0.0
        sumXYc = 0.0

        for elementIndex in range(0, blockLength - 1):
            xDev = blockPtr[elementIndex] - meanX
            yDev = blockPtr[elementIndex + 1] - meanYp
            sumSqXSeq += xDev*xDev
            sumSqYSeq += yDev*yDev
            sumXYc += xDev*yDev

        eps = 1.0e-8*(sumSqXSeq + 1.0)
        if sumSqXSeq > eps:
            beta1 = sumXYc / (sumSqXSeq + (pairsRegLambda*nPairsDouble))
        else:
            beta1 = 0.0

        if beta1 > maxBeta:
            beta1 = maxBeta
        elif beta1 < 0.0:
            beta1 = 0.0

        RSS = sumSqYSeq + ((beta1*beta1)*sumSqXSeq) - (2.0*(beta1*sumXYc))
        if RSS < 0.0:
            RSS = 0.0

        pairCountDouble = <double>(blockLength - 3)
        oneMinusBetaSq = 1.0 - (beta1 * beta1)
        if useInnovationVar:
            divRSS = <double>1.0
        else:
            divRSS = <double>oneMinusBetaSq

        if divRSS <= 1.0e-8:
            divRSS = <double>1.0e-8
        varOutView[regionIndex] = <float>(RSS / pairCountDouble / divRSS)



cdef inline float _ctrans_F32(float x, float c0, float c1, float c2, float c3) nogil:
    # CALLERS: `cTransform`

    return c1*asinhf(x*c0)


cdef inline double _ctrans_F64(double x, double c0, double c1, double c2, double c3) nogil:
    # CALLERS: `cTransform`

    return c1*asinh(x*c0)

cdef inline bint _fSwap(float* swapInArray_, Py_ssize_t i, Py_ssize_t j) nogil:
    # CALLERS: `_partitionLt`, `_nthElement`

    cdef float tmp = swapInArray_[i]
    swapInArray_[i] = swapInArray_[j]
    swapInArray_[j] = tmp
    return <bint>0


cdef inline Py_ssize_t _partitionLt(float* vals_, Py_ssize_t left, Py_ssize_t right, Py_ssize_t pivot) nogil:
    # CALLERS: `_nthElement`

    cdef float pv = vals_[pivot]
    cdef Py_ssize_t store = left
    cdef Py_ssize_t i
    _fSwap(vals_, pivot, right)
    for i in range(left, right):
        if vals_[i] < pv:
            _fSwap(vals_, store, i)
            store += 1
    _fSwap(vals_, store, right)
    return store


cdef inline bint _nthElement(float* sortedVals_, Py_ssize_t n, Py_ssize_t k) nogil:
    # CALLERS: `csampleBlockStats`, `_quantileInPlaceF32`

    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = n - 1
    cdef Py_ssize_t pivot, idx
    while left < right:
        # FFR: check whether this avoids a cast
        pivot = (left + right) >> 1
        idx = _partitionLt(sortedVals_, left, right, pivot)
        if k == idx:
            return <bint>0
        elif k < idx:
            right = idx - 1
        else:
            left = idx + 1
    return <bint>0


cdef inline bint _dSwap(double* swapInArray_, Py_ssize_t i, Py_ssize_t j) nogil:
    # CALLERS: `_partitionLt_F64`, `_nthElement_F64`

    cdef double tmp = swapInArray_[i]
    swapInArray_[i] = swapInArray_[j]
    swapInArray_[j] = tmp
    return <bint>0


cdef inline Py_ssize_t _partitionLt_F64(double* vals_, Py_ssize_t left, Py_ssize_t right, Py_ssize_t pivot) nogil:
    # CALLERS: `_nthElement_F64`

    cdef double pv = vals_[pivot]
    cdef Py_ssize_t store = left
    cdef Py_ssize_t i
    _dSwap(vals_, pivot, right)
    for i in range(left, right):
        if vals_[i] < pv:
            _dSwap(vals_, store, i)
            store += 1
    _dSwap(vals_, store, right)
    return store


cdef inline bint _nthElement_F64(double* sortedVals_, Py_ssize_t n, Py_ssize_t k) nogil:
    # CALLERS: `cSF`

    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = n - 1
    cdef Py_ssize_t pivot, idx
    while left < right:
        pivot = (left + right) >> 1
        idx = _partitionLt_F64(sortedVals_, left, right, pivot)
        if k == idx:
            return <bint>0
        elif k < idx:
            right = idx - 1
        else:
            left = idx + 1
    return <bint>0


cdef inline float _quantileInplaceF32(float* vals_, Py_ssize_t n, float q) nogil:
    #CALLERS: ccalibrateStateCovarScalesRobust

    cdef Py_ssize_t k
    if n <= 0:
        return 1.0
    if q <= 0.0:
        k = 0
    elif q >= 1.0:
        k = n - 1
    else:
        k = <Py_ssize_t>floorf(q * <float>(n - 1))
    _nthElement(vals_, n, k)
    return vals_[k]


cdef inline double _U01() nogil:
    # CALLERS: cDenseGlobalBaseline

    return (<double>rand()) / (<double>RAND_MAX + 1.0)


cdef inline Py_ssize_t _rand_int(Py_ssize_t n) nogil:
    # CALLERS: cDenseGlobalBaseline

    return <Py_ssize_t>(rand() % n)


cdef inline Py_ssize_t _geometricDraw(double logq_) nogil:
    # CALLERS: cDenseGlobalBaseline

    cdef double u = _U01()
    if u <= 0.0:
        u = 1.0 / ((<double>RAND_MAX) + 1.0)
    return <Py_ssize_t>(floor(log(u) / logq_) + 1.0)


cpdef int stepAdjustment(int value, int intervalSizeBP, int pushForward=0):
    r"""Adjusts a value to the nearest multiple of intervalSizeBP, optionally pushing it forward.

    .. todo:: refactor caller + this function into one cython func

    :param value: The value to adjust.
    :type value: int
    :param intervalSizeBP: The step size to adjust to.
    :type intervalSizeBP: int
    :param pushForward: If non-zero, pushes the value forward by intervalSizeBP
    :type pushForward: int
    :return: The adjusted value.
    :rtype: int
    """
    return max(0, (value-(value % intervalSizeBP))) + pushForward*intervalSizeBP


cpdef uint64_t cgetFirstChromRead(str bamFile, str chromosome, uint64_t chromLength, uint32_t samThreads, int samFlagExclude):
    r"""Get the start position of the first read in a BAM file for a given chromosome.


    .. todo:: refactor caller + this function into one cython func

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


    .. todo:: refactor caller + this function into one cython func

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


cpdef cnp.float32_t[:] creadBamSegment(
    str bamFile,
    str chromosome,
    uint32_t start,
    uint32_t end,
    uint32_t intervalSizeBP,
    int64_t readLength,
    uint8_t oneReadPerBin,
    uint16_t samThreads,
    uint16_t samFlagExclude,
    int64_t shiftForwardStrand53 = 0,
    int64_t shiftReverseStrand53 = 0,
    int64_t extendBP = 0,
    int64_t maxInsertSize=1000,
    int64_t pairedEndMode=0,
    int64_t inferFragmentLength=0,
    int64_t minMappingQuality=0,
    int64_t minTemplateLength=-1,
    uint8_t weightByOverlap=1,
    uint8_t ignoreTLEN=1,
    ):
    r"""Count reads in a BAM file for a given chromosome"""

    cdef Py_ssize_t numIntervals
    cdef int64_t width = <int64_t>end - <int64_t>start

    if intervalSizeBP <= 0 or width <= 0:
        numIntervals = 0
    else:
        numIntervals = <Py_ssize_t>((width + intervalSizeBP - 1) // intervalSizeBP)

    cdef cnp.ndarray[cnp.float32_t, ndim=1] values_np = np.zeros(numIntervals, dtype=np.float32)
    cdef cnp.float32_t[::1] values = values_np

    if numIntervals <= 0:
        return values

    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    cdef int64_t start64 = start
    cdef int64_t end64 = end
    cdef int64_t step64 = intervalSizeBP
    cdef Py_ssize_t i, index0, index1, b_, midIndex
    cdef Py_ssize_t lastIndex = numIntervals - 1
    cdef bint readIsForward
    cdef int64_t readStart, readEnd
    cdef int64_t binStart, binEnd
    cdef int64_t overlapStart, overlapEnd, overlap
    cdef int64_t adjStart, adjEnd, fivePrime, mid, tlen, atlen
    cdef uint16_t flag
    cdef int64_t minTLEN = minTemplateLength
    cdef int minMapQ = <int>minMappingQuality

    if minTLEN < 0:
        minTLEN = readLength

    if inferFragmentLength > 0 and pairedEndMode <= 0 and extendBP <= 0:
        extendBP = cgetFragmentLength(bamFile,
         samThreads = samThreads,
         samFlagExclude=samFlagExclude,
         )
    try:
        with aln:
            for read in aln.fetch(chromosome, start64, end64):
                flag = <uint16_t>read.flag
                if flag & samFlagExclude or read.mapping_quality < minMapQ:
                    continue

                readIsForward = (flag & 16) == 0
                readStart = <int64_t>read.reference_start
                readEnd = <int64_t>read.reference_end

                if pairedEndMode > 0:
                    if flag & 2 == 0: # not a properly paired read
                        continue
                    # use first in pair + fragment
                    if flag & 128:
                        continue
                    if (flag & 8) or read.next_reference_id != read.reference_id:
                        continue
                    tlen = <int64_t>read.template_length
                    atlen = tlen if tlen >= 0 else -tlen
                    if atlen == 0 or atlen < minTLEN:
                        continue
                    if tlen >= 0:
                        adjStart = readStart
                        adjEnd = readStart + atlen
                    else:
                        adjEnd = readEnd
                        adjStart = adjEnd - atlen
                    if shiftForwardStrand53 != 0 or shiftReverseStrand53 != 0:
                        if readIsForward:
                            adjStart += shiftForwardStrand53
                            adjEnd += shiftForwardStrand53
                        else:
                            adjStart -= shiftReverseStrand53
                            adjEnd -= shiftReverseStrand53
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
                            adjEnd = fivePrime + extendBP
                        else:
                            adjEnd = fivePrime + 1
                            adjStart = adjEnd - extendBP
                    elif shiftForwardStrand53 != 0 or shiftReverseStrand53 != 0:
                        if readIsForward:
                            adjStart = readStart + shiftForwardStrand53
                            adjEnd = readEnd + shiftForwardStrand53
                        else:
                            adjStart = readStart - shiftReverseStrand53
                            adjEnd = readEnd - shiftReverseStrand53
                    else:
                        adjStart = readStart
                        adjEnd = readEnd

                if adjEnd <= start64 or adjStart >= end64:
                    continue
                if adjStart < start64:
                    adjStart = start64
                if adjEnd > end64:
                    adjEnd = end64

                if oneReadPerBin:
                    mid = (adjStart + adjEnd) // 2
                    midIndex = <Py_ssize_t>((mid - start64) // step64)
                    if 0 <= midIndex <= lastIndex:
                        values[midIndex] += <cnp.float32_t>1.0

                else:
                    index0 = <Py_ssize_t>((adjStart - start64) // step64)
                    index1 = <Py_ssize_t>(((adjEnd - 1) - start64) // step64)
                    if index0 < 0:
                        index0 = 0
                    if index1 > lastIndex:
                        index1 = lastIndex
                    if index0 > lastIndex or index1 < 0 or index0 > index1:
                        continue

                    if weightByOverlap:
                        for b_ in range(index0, index1 + 1):
                            binStart = start64 + (<int64_t>b_)*step64
                            binEnd = binStart + step64
                            if binEnd > end64:
                                binEnd = end64

                            overlapStart = adjStart if adjStart > binStart else binStart
                            overlapEnd = adjEnd if adjEnd < binEnd else binEnd
                            overlap = overlapEnd - overlapStart
                            if overlap > 0:
                                values[b_] += (<cnp.float32_t>overlap / <cnp.float32_t>(binEnd - binStart))
                    else:
                        for b_ in range(index0, index1 + 1):
                            values[b_] += <cnp.float32_t>1.0


    finally:
        aln.close()

    return values


cpdef tuple updateProcessNoiseCovariance(cnp.ndarray[cnp.float32_t, ndim=2] matrixQ,
        cnp.ndarray[cnp.float32_t, ndim=2] matrixQCopy,
        float dStat,
        float dStatAlpha,
        float dStatd,
        float dStatPC,
        bint inflatedQ,
        float maxQ,
        float minQ,
        float dStatAlphaLowMult=0.50,
        float maxMult=2.0):

    cdef float scaleQ, fac, dStatAlphaLow
    cdef float baseSlopeToLevelRatio, maxSlopeQ, minSlopeQ
    cdef float baseOffDiagProd, sqrtDiags, maxNoiseCorr
    cdef float newLevelQ, newSlope_Qnoise, newOffDiagQ
    cdef float newLevel_Qnoise # fix (was python-level before)
    cdef float eps = <float>1.0e-8

    if dStatAlphaLowMult <= 0:
        dStatAlphaLow = 1.0
    else:
        dStatAlphaLow = dStatAlpha*dStatAlphaLowMult
    if dStatAlphaLow >= dStatAlpha:
        dStatAlphaLow = dStatAlpha

    if matrixQCopy[0, 0] > eps:
        baseSlopeToLevelRatio = matrixQCopy[1, 1] / matrixQCopy[0, 0]
    else:
        baseSlopeToLevelRatio = 1.0

    # preserve the baseline level:slope ratio
    maxSlopeQ = maxQ * baseSlopeToLevelRatio
    minSlopeQ = minQ * baseSlopeToLevelRatio
    sqrtDiags = sqrtf(fmaxf(matrixQCopy[0, 0] * matrixQCopy[1, 1], eps))
    baseOffDiagProd = matrixQCopy[0, 1] / sqrtDiags
    newLevelQ = matrixQ[0, 0]
    newSlope_Qnoise = matrixQ[1, 1]
    newOffDiagQ = matrixQ[0, 1]

    # ensure SPD wrt off-diagonals
    maxNoiseCorr = <float>0.999
    if baseOffDiagProd > maxNoiseCorr:
        baseOffDiagProd = maxNoiseCorr
    elif baseOffDiagProd < -maxNoiseCorr:
        baseOffDiagProd = -maxNoiseCorr

    if dStat > dStatAlpha:
        scaleQ = fminf(sqrtf(dStatd*fabsf(dStat - dStatAlpha) + dStatPC), maxMult)
        if (matrixQ[0, 0]*scaleQ <= maxQ) and (matrixQ[1, 1]*scaleQ <= maxSlopeQ):
            matrixQ[0, 0] *= scaleQ
            matrixQ[0, 1] *= scaleQ
            matrixQ[1, 0] *= scaleQ
            matrixQ[1, 1] *= scaleQ
        else:
            newLevel_Qnoise = fminf(matrixQ[0, 0]*scaleQ, maxQ)
            newSlope_Qnoise = fminf(matrixQ[1, 1]*scaleQ, maxSlopeQ)
            newOffDiagQ = baseOffDiagProd * sqrtf(fmaxf(newLevel_Qnoise* newSlope_Qnoise, eps))
            matrixQ[0, 0] = newLevel_Qnoise
            matrixQ[0, 1] = newOffDiagQ
            matrixQ[1, 0] = newOffDiagQ
            matrixQ[1, 1] = newSlope_Qnoise
        inflatedQ = <bint>True

    elif dStat <= dStatAlphaLow and inflatedQ:
        scaleQ = fminf(sqrtf(dStatd*fabsf(dStat - dStatAlphaLow) + dStatPC), maxMult)
        if (matrixQ[0, 0] / scaleQ >= minQ) and (matrixQ[1, 1] / scaleQ >= minSlopeQ):
            matrixQ[0, 0] /= scaleQ
            matrixQ[0, 1] /= scaleQ
            matrixQ[1, 0] /= scaleQ
            matrixQ[1, 1] /= scaleQ
        else:
            # we've hit the minimum, no longer 'inflated'
            newLevel_Qnoise = fmaxf(matrixQ[0, 0] / scaleQ, minQ)
            newSlope_Qnoise = fmaxf(matrixQ[1, 1] / scaleQ, minSlopeQ)
            newOffDiagQ = baseOffDiagProd * sqrtf(fmaxf(newLevel_Qnoise* newSlope_Qnoise, eps))
            matrixQ[0, 0] = newLevel_Qnoise
            matrixQ[0, 1] = newOffDiagQ
            matrixQ[1, 0] = newOffDiagQ
            matrixQ[1, 1] = newSlope_Qnoise
            if (newLevel_Qnoise<= minQ + eps) and (newSlope_Qnoise <= minSlopeQ + eps):
                inflatedQ = <bint>False

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
                        double eps = <double>0.0):
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

    minSize = <Py_ssize_t> max(3, expectedBlockSize*minBlockScale)
    maxSize = <Py_ssize_t> min(maxBlockScale*expectedBlockSize, n)
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

    cdef cnp.ndarray[cnp.intp_t, ndim=1] starts_ = np.random.choice(
        support,
        size=iters,
        replace=True,
        p=None
        ).astype(np.intp)

    cdef Py_ssize_t[::1] startsView = starts_
    cdef Py_ssize_t[::1] sizesView = sizesArr
    cdef double[::1] outView = out
    _blockMax(valuesView, startsView, sizesView, outView, eps)
    return out


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cSparseMax(
        cnp.float32_t[::1] vals,
        dict sparseMap,
        double topFrac = <double>0.25):

    cdef Py_ssize_t n = <Py_ssize_t>vals.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)
    cdef float[::1] outView = out
    cdef double sumTop, tmp
    cdef cnp.ndarray[cnp.float32_t, ndim=1] exBuf
    cdef float[::1] exView
    cdef list nearestList
    cdef object k, v
    cdef Py_ssize_t maxNearest = 0
    cdef cnp.ndarray[cnp.intp_t, ndim=1] neighborIdxs
    cdef cnp.intp_t[::1] neighborIdxView
    cdef Py_ssize_t i, j
    cdef Py_ssize_t numNearest, numRetained, startIdx

    nearestList = [None] * n
    for k, v in sparseMap.items():
        i = <Py_ssize_t>k
        if 0 <= i < n:
            nearestList[i] = v
            numNearest = (<cnp.ndarray>v).shape[0]
            if numNearest > maxNearest:
                maxNearest = numNearest
    if maxNearest < 1:
        maxNearest = 1
    exBuf = np.empty(maxNearest, dtype=np.float32)
    exView = exBuf
    cdef float* trackPtr = &vals[0]
    cdef float* exPtr
    cdef cnp.intp_t* nbPtr

    for i in range(n):
        v = nearestList[i]
        if v is None:
            outView[i] = <float>0.0
            continue
        neighborIdxs = <cnp.ndarray[cnp.intp_t, ndim=1]>v
        neighborIdxView = neighborIdxs
        numNearest = neighborIdxView.shape[0]
        if numNearest <= 0:
            outView[i] = <float>0.0
            continue
        if numNearest > exView.shape[0]:
            exBuf = np.empty(numNearest, dtype=np.float32)
            exView = exBuf
        nbPtr = &neighborIdxView[0]
        exPtr = &exView[0]
        with nogil:
            for j in range(numNearest):
                exPtr[j] = trackPtr[nbPtr[j]]
        tmp = topFrac*<double>numNearest
        numRetained = <Py_ssize_t>tmp
        if tmp > <double>numRetained:
            numRetained += 1
        if numRetained < 1:
            numRetained = 1
        elif numRetained > numNearest:
            numRetained = numNearest
        startIdx = numNearest - numRetained

        with nogil:
            _nthElement(exPtr, numNearest, startIdx)
            sumTop = 0.0
            for j in range(startIdx, numNearest):
                sumTop += <double>exPtr[j]
        outView[i] = <float>(sumTop / <double>numRetained)

    return out


cpdef int64_t cgetFragmentLength(
    str bamFile,
    uint16_t samThreads=0,
    uint16_t samFlagExclude=3844,
    int64_t maxInsertSize=1000,
    int64_t iters=1000,
    int64_t blockSize=5000,
    int64_t fallBack=147,
    int64_t rollingChunkSize=250,
    int64_t lagStep=10,
    int64_t earlyExit=250,
    int64_t randSeed=42,
):

    # FFR: this function (as written) has enough python interaction to nearly void benefits of cython
    # ... either rewrite with helpers for median filter, etc. or move to python for readability
    cdef object rng = default_rng(randSeed)
    cdef int64_t regionLen, numRollSteps
    cdef int numChunks
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rawArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] medArr
    cdef AlignmentFile aln
    cdef AlignedSegment readSeg
    cdef list blockCenters
    cdef list bestLags
    cdef int i, j, idxVal
    cdef int startIdx, endIdx
    cdef int winSize, takeK
    cdef int blockHalf, readFlag
    cdef int maxValidLag
    cdef int strand
    cdef int samThreadsInternal
    cdef object cpuCountObj
    cdef int cpuCount
    cdef int64_t blockStartBP, blockEndBP, readStart, readEnd
    cdef int64_t med
    cdef double score
    cdef cnp.ndarray[cnp.intp_t, ndim=1] topContigsIdx
    cdef cnp.ndarray[cnp.intp_t, ndim=1] unsortedIdx, sortedIdx
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unsortedVals
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] seen
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fwd
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rev
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fwdDiff
    cdef cnp.ndarray[cnp.float64_t, ndim=1] revDiff
    cdef int64_t diffS, diffE
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] bestLagsArr
    cdef bint isPairedEnd = <bint>0
    cdef double avgTemplateLen = <double>0.0
    cdef int64_t templateLenSamples = <int64_t>0
    cdef double avgReadLength = <double>0.0
    cdef int64_t numReadLengthSamples = <int64_t>0
    cdef int64_t minInsertSize
    cdef int64_t requiredSamplesPE
    cdef int64_t tlen

    # rather than taking `chromosome`, `start`, `end`
    # ... we will just look at BAM contigs present and use
    # ... the three largest to estimate the fragment length
    cdef tuple contigs
    cdef tuple lengths
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lengthsArr
    cdef Py_ssize_t contigIdx
    cdef str contig
    cdef int64_t contigLen

    cdef double[::1] fwdView
    cdef double[::1] revView
    cdef double[::1] fwdDiffView
    cdef double[::1] revDiffView
    cdef double runningSum
    cdef double fwdSum
    cdef double revSum
    cdef double fwdMean
    cdef double revMean
    cdef double bestScore
    cdef int bestLag
    cdef int blockLen
    cdef int localMinLag
    cdef int localMaxLag
    cdef int localLagStep
    cdef int kTop
    earlyExit = min(earlyExit, iters)
    samThreadsInternal = <int>samThreads
    cpuCountObj = os.cpu_count()
    if cpuCountObj is None:
        cpuCount = 1
    else:
        cpuCount = <int>cpuCountObj
        if cpuCount < 1:
            cpuCount = 1

    if samThreads < 1:
        samThreadsInternal = <int>min(max(1, cpuCount // 2), 4)

    aln = AlignmentFile(bamFile, "rb", threads=samThreadsInternal)
    try:
        contigs = aln.references
        lengths = aln.lengths

        if contigs is None or len(contigs) == 0:
            return <int64_t>fallBack

        lengthsArr = np.asarray(lengths, dtype=np.int64)
        kTop = 2 if len(contigs) >= 2 else 1
        topContigsIdx = np.argpartition(lengthsArr, -kTop)[-kTop:]
        topContigsIdx = topContigsIdx[np.argsort(lengthsArr[topContigsIdx])]
        for contigIdx in topContigsIdx:
            contig = contigs[contigIdx]
            for readSeg in aln.fetch(contig):
                readFlag = readSeg.flag
                if (readFlag & samFlagExclude) != 0:
                    continue
                if not isPairedEnd:
                    if (readFlag & 1) != 0:
                        isPairedEnd = <bint>1
                if numReadLengthSamples < iters:
                    avgReadLength += readSeg.query_length
                    numReadLengthSamples += 1
                else:
                    break

        if numReadLengthSamples <= 0:
            return <int64_t>fallBack
        avgReadLength /= <double>numReadLengthSamples
        minInsertSize = <int64_t>(avgReadLength / 2.0)
        if minInsertSize < 1:
            minInsertSize = 1
        if minInsertSize > maxInsertSize:
            minInsertSize = maxInsertSize

        if isPairedEnd:
            # skip to the paired-end block below (no xCorr --> average template len)
            requiredSamplesPE = max(iters, 1000)

            for contigIdx in topContigsIdx:
                if templateLenSamples >= requiredSamplesPE:
                    break
                contig = contigs[contigIdx]

                for readSeg in aln.fetch(contig):
                    if templateLenSamples >= requiredSamplesPE:
                        break

                    readFlag = readSeg.flag
                    if (readFlag & samFlagExclude) != 0 or (readFlag & 2) == 0:
                        # skip any excluded flags, only count proper pairs
                        continue

                    # read1 only: otherwise each pair contributes to the mean twice
                    # ... which might reduce breadth of the estimate
                    if (readFlag & 64) == 0:
                        continue
                    tlen = <int64_t>readSeg.template_length
                    if tlen > 0:
                        avgTemplateLen += <double>tlen
                        templateLenSamples += 1
                    elif tlen < 0:
                        avgTemplateLen += <double>(-tlen)
                        templateLenSamples += 1

            if templateLenSamples < requiredSamplesPE:
                return <int64_t> fallBack

            avgTemplateLen /= <double>templateLenSamples

            if avgTemplateLen >= minInsertSize and avgTemplateLen <= maxInsertSize:
                return <int64_t>(avgTemplateLen + 0.5)
            else:
                return <int64_t> fallBack

        bestLags = []
        blockHalf = blockSize // 2

        fwd = np.zeros(blockSize, dtype=np.float64, order='C')
        rev = np.zeros(blockSize, dtype=np.float64, order='C')
        fwdDiff = np.zeros(blockSize+1, dtype=np.float64, order='C')
        revDiff = np.zeros(blockSize+1, dtype=np.float64, order='C')

        fwdView = fwd
        revView = rev
        fwdDiffView = fwdDiff
        revDiffView = revDiff

        for contigIdx in topContigsIdx:
            contig = contigs[contigIdx]
            contigLen = <int64_t>lengthsArr[contigIdx]
            regionLen = contigLen

            if regionLen < blockSize or regionLen <= 0:
                continue

            if maxInsertSize < 1:
                maxInsertSize = 1

            # first, we build a coarse read coverage track from `start` to `end`
            numRollSteps = regionLen // rollingChunkSize
            if numRollSteps <= 0:
                numRollSteps = 1
            numChunks = <int>numRollSteps

            rawArr = np.zeros(numChunks, dtype=np.float64)
            medArr = np.zeros(numChunks, dtype=np.float64)

            for readSeg in aln.fetch(contig):
                readFlag = readSeg.flag
                if (readFlag & samFlagExclude) != 0:
                    continue
                j = <int>(readSeg.reference_start // rollingChunkSize)
                if 0 <= j < numChunks:
                    rawArr[j] += 1.0

            # second, we apply a rolling/moving/local/weywtci order-statistic filter (median)
            # ...the size of the kernel is based on the blockSize -- we want high-coverage
            # ...blocks as measured by their local median read count
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

            # expand each top-K center in-place into a "seen" mask,
            # then gather unique block centers once.
            seen = np.zeros(numChunks, dtype=np.uint8)
            blockCenters = []
            for i in range(takeK):
                idxVal = <int>sortedIdx[i]
                startIdx = idxVal - (winSize // 2)
                endIdx = startIdx + winSize
                if startIdx < 0:
                    startIdx = 0
                    endIdx = winSize if winSize < numChunks else numChunks
                if endIdx > numChunks:
                    endIdx = numChunks
                    startIdx = endIdx - winSize if winSize <= numChunks else 0
                for j in range(startIdx, endIdx):
                    if seen[j] == 0:
                        seen[j] = 1
                        blockCenters.append(j)

            if len(blockCenters) > 1:
                rng.shuffle(blockCenters)

            for idxVal in blockCenters:
                # this should map back to genomic coordinates
                blockStartBP = idxVal*rollingChunkSize + (rollingChunkSize // 2) - blockHalf
                if blockStartBP < 0:
                    blockStartBP = 0
                blockEndBP = blockStartBP + blockSize
                if blockEndBP > contigLen:
                    blockEndBP = contigLen
                    blockStartBP = blockEndBP - blockSize
                    if blockStartBP < 0:
                        continue

                # now we build strand-specific tracks
                # ...avoid forward/reverse strand for loops in each block w/ a cumsum
                fwd.fill(0.0)
                fwdDiff.fill(0.0)
                rev.fill(0.0)
                revDiff.fill(0.0)
                readFlag = -1

                for readSeg in aln.fetch(contig, blockStartBP, blockEndBP):
                    readFlag = readSeg.flag
                    if (readFlag & samFlagExclude) != 0:
                        continue
                    readStart = <int64_t>readSeg.reference_start
                    readEnd = <int64_t>readSeg.reference_end
                    if readStart < blockStartBP or readEnd > blockEndBP:
                        continue

                    diffS = readStart - blockStartBP
                    diffE = readEnd - blockStartBP
                    strand = readFlag & 16
                    if strand == 0:
                        # forward
                        # just mark offsets from block start/end
                        fwdDiffView[<int>diffS] += 1.0
                        fwdDiffView[<int>diffE] -= 1.0
                    else:
                        # reverse
                        # ditto
                        revDiffView[<int>diffS] += 1.0
                        revDiffView[<int>diffE] -= 1.0

                maxValidLag = maxInsertSize if (maxInsertSize < blockSize) else (blockSize - 1)
                localMinLag = <int>minInsertSize
                localMaxLag = <int>maxValidLag
                if localMaxLag < localMinLag:
                    continue
                localLagStep = <int>lagStep
                if localLagStep < 1:
                    localLagStep = 1

                # now we can get coverage track by summing over diffs
                # maximizes the crossCovar(forward, reverse, lag) wrt lag.
                with nogil:
                    runningSum = 0.0
                    for i from 0 <= i < blockSize:
                        runningSum += fwdDiffView[i]
                        fwdView[i] = runningSum

                    runningSum = 0.0
                    for i from 0 <= i < blockSize:
                        runningSum += revDiffView[i]
                        revView[i] = runningSum

                    fwdSum = 0.0
                    revSum = 0.0
                    for i from 0 <= i < blockSize:
                        fwdSum += fwdView[i]
                        revSum += revView[i]

                    fwdMean = fwdSum / blockSize
                    revMean = revSum / blockSize

                    for i from 0 <= i < blockSize:
                        fwdView[i] = fwdView[i] - fwdMean
                        revView[i] = revView[i] - revMean

                    bestScore = -1e308
                    bestLag = -1
                    for lag from localMinLag <= lag <= localMaxLag by localLagStep:
                        score = 0.0
                        blockLen = blockSize - lag
                        for i from 0 <= i < blockLen:
                            score += fwdView[i] * revView[i + lag]
                        if score > bestScore:
                            bestScore = score
                            bestLag = lag

                if bestLag > 0 and bestScore != 0.0:
                    bestLags.append(bestLag)
                if len(bestLags) >= earlyExit:
                    break

    finally:
        aln.close()

    if len(bestLags) < 3:
        return fallBack

    bestLagsArr = np.asarray(bestLags, dtype=np.uint32)
    med = int(np.median(bestLagsArr) + avgReadLength + 0.5)
    if med < minInsertSize:
        med = <int>minInsertSize
    elif med > maxInsertSize:
        med = <int>maxInsertSize
    return <int64_t>med


cpdef cnp.ndarray[cnp.uint8_t, ndim=1] cbedMask(
    str chromosome,
    str bedFile,
    cnp.ndarray[cnp.uint32_t, ndim=1] intervals,
    int intervalSizeBP
    ):
    r"""Return a 1/0 mask for intervals overlapping a sorted and merged BED file.

    :param chromosome: Chromosome name.
    :type chromosome: str
    :param bedFile: Path to a sorted and merged BED file.
    :type bedFile: str
    :param intervals: Array of sorted, non-overlapping start positions of genomic intervals.
      Each interval is assumed `intervalSizeBP`.
    :type intervals: cnp.ndarray[cnp.uint32_t, ndim=1]
    :param intervalSizeBP: Step size between genomic positions in `intervals`.
    :type intervalSizeBP: int32_t
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
    cdef uint32_t* svPtr
    cdef uint32_t* evPtr
    cdef uint32_t* posPtr

    cdef uint8_t* outPtr
    cdef Py_ssize_t n = starts.size
    if starts.size > 0:
        svPtr = &startsView[0]
    else:
        svPtr = <uint32_t*>NULL

    if ends.size > 0:
        evPtr = &endsView[0]
    else:
        evPtr = <uint32_t*>NULL

    if numIntervals > 0:
        posPtr = &posView[0]
        outPtr = &outView[0]
    else:
        posPtr = <uint32_t*>NULL
        outPtr = <uint8_t*>NULL

    with nogil:
        if numIntervals > 0 and n > 0:
            _maskMembership(posPtr, numIntervals, svPtr, evPtr, n, outPtr)
    return mask


cpdef void projectToBox(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] vectorX,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixP,
    cnp.float32_t stateLowerBound,
    cnp.float32_t stateUpperBound,
    cnp.float32_t eps
):
    _projectToBox(vectorX, matrixP, stateLowerBound, stateUpperBound, eps)


cpdef tuple cmeanVarPairs(cnp.ndarray[cnp.uint32_t, ndim=1] intervals,
                          cnp.ndarray[cnp.float32_t, ndim=1] values,
                          int blockSize,
                          int iters,
                          int randSeed,
                          cnp.ndarray[cnp.uint8_t, ndim=1] excludeIdxMask,
                          double zeroPenalty=0.0,
                          double zeroThresh=0.0,
                          bint useInnovationVar = <bint>True,
                          bint useSampleVar = <bint>False):

    cdef cnp.ndarray[cnp.float64_t, ndim=1] valuesArray
    cdef double[::1] valuesView
    cdef cnp.ndarray[cnp.intp_t, ndim=1] sizesArray
    cdef cnp.ndarray[cnp.float32_t, ndim=1] outMeans
    cdef cnp.ndarray[cnp.float32_t, ndim=1] outVars
    cdef Py_ssize_t valuesLength
    cdef Py_ssize_t maxBlockLength
    cdef list supportList
    cdef cnp.intp_t scanIndex
    cdef cnp.ndarray[cnp.intp_t, ndim=1] supportArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1] starts_
    cdef cnp.ndarray[cnp.intp_t, ndim=1] ends
    cdef Py_ssize_t[::1] startsView
    cdef Py_ssize_t[::1] sizesView
    cdef float[::1] meansView
    cdef float[::1] varsView
    cdef cnp.ndarray[cnp.intp_t, ndim=1] emptyStarts
    cdef cnp.ndarray[cnp.intp_t, ndim=1] emptyEnds
    cdef double geomProb


    rng = default_rng(randSeed)
    valuesArray = np.ascontiguousarray(values, dtype=np.float64)
    valuesView = valuesArray
    sizesArray = np.full(iters, blockSize, dtype=np.intp)
    outMeans = np.empty(iters, dtype=np.float32)
    outVars = np.empty(iters, dtype=np.float32)
    valuesLength = <Py_ssize_t>valuesArray.size
    maxBlockLength = <Py_ssize_t>blockSize
    geomProb = 1.0 / (<double>maxBlockLength)
    supportList = []
    scanIndex = 0

    while scanIndex <= valuesLength - maxBlockLength:
        if excludeIdxMask[scanIndex:scanIndex + maxBlockLength].any():
            scanIndex = scanIndex + maxBlockLength + 1
            continue
        supportList.append(scanIndex)
        scanIndex = scanIndex + 1

    if len(supportList) == 0:
        outMeans[:] = 0.0
        outVars[:] = 0.0
        emptyStarts = np.empty(0, dtype=np.intp)
        emptyEnds = np.empty(0, dtype=np.intp)
        return outMeans, outVars, emptyStarts, emptyEnds

    supportArr = np.asarray(supportList, dtype=np.intp)
    starts_ = rng.choice(supportArr, size=iters, replace=True).astype(np.intp)
    sizesArray = rng.geometric(geomProb, size=<int>starts_.size).astype(np.intp, copy=False)
    np.maximum(sizesArray, <cnp.intp_t>5, out=sizesArray)
    np.minimum(sizesArray, 5*maxBlockLength, out=sizesArray)
    ends = starts_ + sizesArray

    startsView = starts_
    sizesView = sizesArray
    meansView = outMeans
    varsView = outVars

    _regionMeanVar(valuesView, startsView, sizesView, meansView, varsView, zeroPenalty, zeroThresh, useInnovationVar, useSampleVar)

    return outMeans, outVars, starts_, ends


cdef bint _cEMA(const double* xPtr, double* outPtr,
                    Py_ssize_t n, double alpha) nogil:
    cdef Py_ssize_t i
    if alpha > 1.0 or alpha < 0.0:
        return <bint>1

    outPtr[0] = xPtr[0]

    # forward
    for i in range(1, n):
        outPtr[i] = alpha*xPtr[i] + (1.0 - alpha)*outPtr[i - 1]

    # back
    for i in range(n - 2, -1, -1):
        outPtr[i] = alpha*outPtr[i] + (1.0 - alpha)*outPtr[i + 1]

    return <bint>0


cpdef cEMA(cnp.ndarray x, double alpha):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x1 = np.ascontiguousarray(x, dtype=np.float64)
    cdef Py_ssize_t n = x1.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    _cEMA(<const double*>x1.data, <double*>out.data, n, alpha)
    return out


cpdef tuple momVST(object x, double a=<double>(-1.0), double offset=<double>(3.0/8.0), bint useAsymptoticLog2=<bint>True):

    cdef cnp.ndarray[cnp.float64_t, ndim=1] x_F64 = np.ascontiguousarray(x, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_F64 = np.empty(<Py_ssize_t>(x_F64.size), dtype=np.float64)
    cdef Py_ssize_t n = x_F64.shape[0]
    cdef double mom1 = 0.0
    cdef double mom2 = 0.0
    cdef double welf1, welf2
    cdef double muHat, varHat
    cdef double a_ = a
    cdef double offset_ = offset
    cdef double scale_
    cdef double multLog2 = 2.0 * __INV_LN2_DOUBLE
    cdef double xValue
    cdef Py_ssize_t i

    if offset_ <= 0.0:
        offset_ = 0.375

    if not (useAsymptoticLog2):
        multLog2 = <double>(1.0)

    if a_ <= 0.0:
        for i in range(n):
            xValue = x_F64[i]
            welf1 = xValue - mom1
            mom1 += welf1 / (i + 1.0)
            welf2 = xValue - mom1
            mom2 += welf1 * welf2

        muHat = mom1
        varHat = (mom2 / (n - 1.0)) if n > 1 else 0.0

        if muHat > 0.0 and varHat > muHat:
            a_ = (muHat * muHat) / (varHat - muHat)
        else:
            # push to poisson when 1/a --> 0, and return before asinh
            for i in range(n):
                xValue = x_F64[i]
                out_F64[i] = multLog2 * sqrt(xValue + offset_)
            return (out_F64, a_)


    scale_ = <double>(a_ - 0.75)
    if scale_ <= 0.0 or (not isfinite(scale_)):
        a_ = <double>(0.75 + 1.0e-4)
        scale_ = <double>(a_ - 0.75)
    cdef double valAtZero = multLog2 * asinh(sqrt(offset_ / scale_))
    for i in range(n):
        xValue = x_F64[i]
        out_F64[i] = multLog2 * asinh(sqrt((xValue + offset_) / scale_)) - valAtZero

    return (out_F64, a_)


cpdef object cTransform(
    object x,
    Py_ssize_t blockLength,
    bint disableLocalBackground=<bint>False,
    double denseMeanQuantile=<double>0.50,
    double liftLower=<double>1.0,
    double c0=<double>(3.0/8.0),
    double c1=<double>1/log(2.0),
    double c2=<double>0.0,
    double c3=<double>0.0,
    double w_local=<double>1.0,
    double w_global=<double>3.0,
    bint verbose=<bint>False,
    uint64_t rseed=<uint64_t>0,
):
    cdef Py_ssize_t bootBlockSize, n, i
    cdef double wLocal, wGlobal, weightSum, invWeightSum
    cdef object momRes
    cdef cnp.ndarray outArr
    cdef cnp.ndarray zArr_F32, localBaseArr_F32
    cdef float[::1] zView_F32, localBaseView_F32, outView_F32
    cdef float globalBaselineF32, combinedBaselineF32
    cdef float wLocalF32, wGlobalF32, invWeightSumF32
    cdef cnp.ndarray zArr_F64, localBaseArr_F64
    cdef double[::1] zView_F64, localBaseView_F64, outView_F64
    cdef double globalBaselineF64, combinedBaselineF64
    cdef double wLocalF64, wGlobalF64, invWeightSumF64

    if disableLocalBackground:
        wLocal = 0.0
        wGlobal = 1.0
    else:
        wLocal = w_local
        wGlobal = w_global

    # If user explicitly sets w_local=w_global=0, skip both local+global subtraction
    if not disableLocalBackground and wLocal == 0.0 and wGlobal == 0.0:
        momRes = momVST(x)
        if (<cnp.ndarray>x).dtype == np.float32:
            return np.ascontiguousarray(momRes[0], dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(momRes[0], dtype=np.float64).reshape(-1)

    bootBlockSize = <Py_ssize_t>max(min(blockLength, 10000), 3)
    if bootBlockSize % 2 == 0:
        bootBlockSize += 1
        if bootBlockSize > 10000:
            bootBlockSize -= 2

    weightSum = wLocal + wGlobal
    if weightSum <= 0.0:
        wLocal = 0.0
        wGlobal = 1.0
        weightSum = 1.0
    invWeightSum = 1.0 / weightSum

    n = (<cnp.ndarray>x).size
    # F32
    if (<cnp.ndarray>x).dtype == np.float32:
        # I) ~VST~ with a approximated from data
        momRes = momVST(x)
        zArr_F32 = np.ascontiguousarray(momRes[0], dtype=np.float32).reshape(-1)
        zView_F32 = zArr_F32
        n = zArr_F32.shape[0]
        outArr = np.empty(n, dtype=np.float32)
        outView_F32 = outArr

        # II) baseline estimation (weighted(local, global)) on transformed data
        if wGlobal > 0.0:
            # right-tail biased circular resampling
            globalBaselineF32 = <float>cDenseGlobalBaseline(
                zArr_F32,
                bootBlockSize=bootBlockSize,
                denseMeanQuantile=denseMeanQuantile,
                verbose=verbose,
                seed=rseed,
            )
        else:
            globalBaselineF32 = 0.0

        if wLocal > 0.0:
            # lower envelope baseline
            localBaseArr_F32 = clocalBaseline(zArr_F32, <int>bootBlockSize, liftLower=<float>liftLower)
            localBaseView_F32 = localBaseArr_F32

        # III) remove weighted combination of local,global baselines
        wLocalF32 = <float>wLocal
        wGlobalF32 = <float>wGlobal
        invWeightSumF32 = <float>invWeightSum

        with nogil:
            for i in range(n):
                combinedBaselineF32 = (
                    (wLocalF32 * (localBaseView_F32[i] if wLocalF32 > 0.0 else 0.0)) +
                    (wGlobalF32 * globalBaselineF32)
                ) * invWeightSumF32
                outView_F32[i] = zView_F32[i] - combinedBaselineF32
        return outArr

    # F64
    momRes = momVST(x)
    zArr_F64 = np.ascontiguousarray(momRes[0], dtype=np.float64).reshape(-1)
    zView_F64 = zArr_F64
    n = zArr_F64.shape[0]

    outArr = np.empty(n, dtype=np.float64)
    outView_F64 = outArr

    if wGlobal > 0.0:
        globalBaselineF64 = <double>cDenseGlobalBaseline(
            zArr_F64,
            bootBlockSize=bootBlockSize,
            denseMeanQuantile=denseMeanQuantile,
            verbose=verbose,
            seed=rseed,
        )
    else:
        globalBaselineF64 = 0.0

    if wLocal > 0.0:
        localBaseArr_F64 = clocalBaseline(zArr_F64, <int>bootBlockSize, liftLower=<double>liftLower)
        localBaseView_F64 = localBaseArr_F64

    wLocalF64 = <double>wLocal
    wGlobalF64 = <double>wGlobal
    invWeightSumF64 = <double>invWeightSum

    with nogil:
        for i in range(n):
            combinedBaselineF64 = (
                (wLocalF64 * (localBaseView_F64[i] if wLocalF64 > 0.0 else 0.0)) +
                (wGlobalF64 * globalBaselineF64)
            ) * invWeightSumF64
            outView_F64[i] = zView_F64[i] - combinedBaselineF64
    return outArr


cpdef protectCovariance22(object A, double eigFloor=1.0e-4):
    cdef cnp.ndarray arr
    cdef double a_, b_, c_
    cdef double TRACE, DET, EIG1, EIG2
    cdef float TRACE_F32, DET_F32, EIG1_F32, EIG2_F32, LAM_F32
    cdef double eigvecFirstComponent, eigvecSecondComponent, invn, eigvecFirstSquared, eigvecSecondSquared, eigvecProd, LAM
    cdef double* ptr_F64
    cdef float* ptr_F32
    arr = <cnp.ndarray>A

    # F64
    if arr.dtype == np.float64:
        ptr_F64 = <double*>arr.data
        with nogil:
            a_ = ptr_F64[0]
            c_ = ptr_F64[3]
            b_ = 0.5*(ptr_F64[1] + ptr_F64[2])

            if b_ == 0.0:
                if a_ < eigFloor: a_ = eigFloor
                if c_ < eigFloor: c_ = eigFloor
                ptr_F64[0] = a_
                ptr_F64[1] = 0.0
                ptr_F64[2] = 0.0
                ptr_F64[3] = c_
            else:
                TRACE = a_ + c_
                DET = sqrt(0.25*(a_ - c_)*(a_ - c_) + (b_*b_))
                EIG1 = <double>(TRACE + 2*DET)/2.0
                EIG2 = <double>(TRACE - 2*DET)/2.0

                if EIG1 < eigFloor: EIG1 = eigFloor
                if EIG2 < eigFloor: EIG2 = eigFloor

                if fabs(EIG1 - c_) > fabs(EIG1 - a_):
                    eigvecFirstComponent = EIG1 - c_
                    eigvecSecondComponent = b_
                else:
                    eigvecFirstComponent = b_
                    eigvecSecondComponent = EIG1 - a_

                if eigvecFirstComponent == 0.0 and eigvecSecondComponent == 0.0:
                    eigvecFirstComponent = <double>1.0
                    eigvecSecondComponent = <double>0.0

                invn = 1.0 / sqrt((eigvecFirstComponent*eigvecFirstComponent) + (eigvecSecondComponent*eigvecSecondComponent))
                eigvecFirstComponent *= invn
                eigvecSecondComponent *= invn

                eigvecFirstSquared = (eigvecFirstComponent*eigvecFirstComponent)
                eigvecSecondSquared = (eigvecSecondComponent*eigvecSecondComponent)
                eigvecProd = eigvecFirstComponent*eigvecSecondComponent
                LAM = EIG1 - EIG2


                # rewrite/padViewgiven 2x2 + SPD (and pad):
                # A = λ_2*(I) + (λ_1 - λ_2)*(vv^T), where v <--> λ_1
                ptr_F64[0] = EIG2 + LAM*eigvecFirstSquared
                ptr_F64[3] = EIG2 + LAM*eigvecSecondSquared
                ptr_F64[1] = LAM*eigvecProd
                ptr_F64[2] = ptr_F64[1]
        return A

    # F32
    if arr.dtype == np.float32:
        ptr_F32 = <float*>arr.data
        with nogil:
            a_ = <double>ptr_F32[0]
            c_ = <double>ptr_F32[3]
            b_ = 0.5*((<double>ptr_F32[1]) + (<double>ptr_F32[2]))

            if b_ == 0.0:
                if a_ < eigFloor: a_ = eigFloor
                if c_ < eigFloor: c_ = eigFloor
                ptr_F32[0] = <float>a_
                ptr_F32[1] = <float>0.0
                ptr_F32[2] = <float>0.0
                ptr_F32[3] = <float>c_
            else:
                TRACE_F32 = <float>(a_ + c_)
                DET_F32 = <float>(sqrt(0.25*(a_ - c_)*(a_-c_) + (b_*b_)))
                EIG1_F32 = <float>((TRACE_F32 + 2*DET_F32) / 2.0)
                EIG2_F32 = <float>(TRACE_F32 - 2*DET_F32) / 2.0

                if EIG1_F32 < eigFloor: EIG1_F32 = eigFloor
                if EIG2_F32 < eigFloor: EIG2_F32 = eigFloor

                if fabs(EIG1_F32 - c_) > fabs(EIG1_F32 - a_):
                    eigvecFirstComponent = EIG1_F32 - c_
                    eigvecSecondComponent = b_
                else:
                    eigvecFirstComponent = b_
                    eigvecSecondComponent = EIG1_F32 - a_

                if eigvecFirstComponent == 0.0 and eigvecSecondComponent == 0.0:
                    eigvecFirstComponent = 1.0
                    eigvecSecondComponent = 0.0

                invn = 1.0 / sqrt(eigvecFirstComponent*eigvecFirstComponent + eigvecSecondComponent*eigvecSecondComponent)
                eigvecFirstComponent *= invn
                eigvecSecondComponent *= invn

                eigvecFirstSquared = eigvecFirstComponent*eigvecFirstComponent
                eigvecSecondSquared = eigvecSecondComponent*eigvecSecondComponent
                eigvecProd = eigvecFirstComponent*eigvecSecondComponent
                LAM_F32 = EIG1_F32 - EIG2_F32

                ptr_F32[0] = <float>(EIG2_F32 + LAM_F32*eigvecFirstSquared)
                ptr_F32[3] = <float>(EIG2_F32 + LAM_F32*eigvecSecondSquared)
                ptr_F32[1] = <float>(LAM_F32*eigvecProd)
                ptr_F32[2] = ptr_F32[1]
        return A


cpdef tuple cforwardPass(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixMunc,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQ,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQCopy,
    float dStatAlpha,
    float dStatd,
    float dStatPC,
    float maxQ,
    float minQ,
    float stateInit,
    float stateCovarInit,
    object coefficientsH=None,
    float covarClip=3.0,
    float pad=1.0e-2,
    bint projectStateDuringFiltering=False,
    float stateLowerBound=0.0,
    float stateUpperBound=0.0,
    Py_ssize_t chunkSize=1000000,
    object stateForward=None,
    object stateCovarForward=None,
    object pNoiseForward=None,
    object vectorD=None,
    object progressBar=None,
    Py_ssize_t progressIter=25000,
    bint returnNLL=False,
    bint storeNLLInD=False,
    object intervalToBlockMap=None,
    object blockGradLogScale=None,
    object blockGradCount=None,
):
    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] muncView = matrixMunc
    cdef cnp.float32_t[:, ::1] stateTransitionView = matrixF
    cdef Py_ssize_t trackCount = dataView.shape[0]
    cdef Py_ssize_t intervalCount = dataView.shape[1]
    cdef Py_ssize_t intervalIndex, trackIndex

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] dStatVectorArr
    cdef cnp.float32_t[::1] dStatVector

    cdef bint doStore = (stateForward is not None)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForwardArr
    cdef cnp.float32_t[:, ::1] stateForwardView
    cdef cnp.float32_t[:, :, ::1] stateCovarForwardView
    cdef cnp.float32_t[:, :, ::1] pNoiseForwardView

    cdef bint doFlush = False
    cdef bint doProgress = False
    cdef Py_ssize_t stepsDone = 0
    cdef Py_ssize_t progressRemainder = 0

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] stateVector = np.array([stateInit, 0.0], dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateCovar = (np.eye(2, dtype=np.float32)*np.float32(stateCovarInit))
    cdef cnp.float32_t[::1] stateVectorView = stateVector
    cdef cnp.float32_t[:, ::1] stateCovarView = stateCovar

    cdef double clipSmall = pow(10.0, -covarClip)
    cdef double clipBig = pow(10.0, covarClip)

    cdef bint inflatedQ = False
    cdef int adjustmentCount = 0
    cdef float phiHat = 1.0

    cdef double stateTransition00, stateTransition01, stateTransition10, stateTransition11
    cdef double sumWeightUU, sumWeightUY, sumWeightYY, sumResidualUU
    cdef double innovationValue, measurementVariance, paddedVariance, invVariance
    cdef double addP00Trace, weightRank1, quadraticForm, dStatValue
    cdef double posteriorP00, posteriorP01, posteriorP10, posteriorP11
    cdef double priorP00, priorP01, priorP10, priorP11
    cdef double priorState0, priorState1
    cdef double tmp00, tmp01, tmp10, tmp11
    cdef double intermediate_, gainG, gainH, IKH00, IKH10
    cdef double posteriorNew00, posteriorNew01, posteriorNew11
    cdef double sumLogR = 0.0
    cdef double sumNLL = 0.0
    cdef double intervalNLL = 0.0
    cdef double sumDStat = 0.0
    cdef bint doBlockGrad = (intervalToBlockMap is not None and blockGradLogScale is not None and blockGradCount is not None)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] intervalToBlockMapArr
    cdef cnp.int32_t[::1] intervalToBlockMapView
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] blockGradLogScaleArr
    cdef cnp.float32_t[::1] blockGradLogScaleView
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] blockGradCountArr
    cdef cnp.float32_t[::1] blockGradCountView
    cdef cnp.int32_t blockId = 0

    cdef double gradSumLogR, gradSumWUU, gradSumWUY, gradSumWYY
    cdef double dAddTraceLog, dWeightRank1, dQuad, intervalGrad
    cdef double dVar

    if vectorD is None:
        dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
        vectorD = dStatVectorArr
    else:
        dStatVectorArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> vectorD

    dStatVector = dStatVectorArr

    if doStore:
        stateForwardArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateForward
        stateCovarForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarForward
        pNoiseForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> pNoiseForward
        stateForwardView = stateForwardArr
        stateCovarForwardView = stateCovarForwardArr
        pNoiseForwardView = pNoiseForwardArr

    if doBlockGrad:
        intervalToBlockMapArr = <cnp.ndarray[cnp.int32_t, ndim=1, mode="c"]> intervalToBlockMap
        intervalToBlockMapView = intervalToBlockMapArr
        blockGradLogScaleArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> blockGradLogScale
        blockGradLogScaleView = blockGradLogScaleArr
        blockGradCountArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> blockGradCount
        blockGradCountView = blockGradCountArr

    doFlush = (doStore and chunkSize > 0)
    doProgress = (progressBar is not None and progressIter > 0)

    stateTransition00 = <double>stateTransitionView[0,0]
    stateTransition01 = <double>stateTransitionView[0,1]
    stateTransition10 = <double>stateTransitionView[1,0]
    stateTransition11 = <double>stateTransitionView[1,1]

    # in case we refit Q across multiple passes
    matrixQ[0,0] = matrixQCopy[0,0]
    matrixQ[0,1] = matrixQCopy[0,1]
    matrixQ[1,0] = matrixQCopy[1,0]
    matrixQ[1,1] = matrixQCopy[1,1]

    for intervalIndex in range(intervalCount):
        # 'PREDICT' (prior, state transition model)
        priorState0 = stateTransition00*(<double>stateVectorView[0]) + stateTransition01*(<double>stateVectorView[1])
        priorState1 = stateTransition10*(<double>stateVectorView[0]) + stateTransition11*(<double>stateVectorView[1])
        stateVectorView[0] = <cnp.float32_t>priorState0
        stateVectorView[1] = <cnp.float32_t>priorState1

        posteriorP00 = <double>stateCovarView[0,0]
        posteriorP01 = <double>stateCovarView[0,1]
        posteriorP10 = <double>stateCovarView[1,0]
        posteriorP11 = <double>stateCovarView[1,1]

        tmp00 = stateTransition00*posteriorP00 + stateTransition01*posteriorP10
        tmp01 = stateTransition00*posteriorP01 + stateTransition01*posteriorP11
        tmp10 = stateTransition10*posteriorP00 + stateTransition11*posteriorP10
        tmp11 = stateTransition10*posteriorP01 + stateTransition11*posteriorP11

        priorP00 = tmp00*stateTransition00 + tmp01*stateTransition01 + (<double>matrixQ[0,0])
        priorP01 = tmp00*stateTransition10 + tmp01*stateTransition11 + (<double>matrixQ[0,1])
        priorP10 = tmp10*stateTransition00 + tmp11*stateTransition01 + (<double>matrixQ[1,0])
        priorP11 = tmp10*stateTransition10 + tmp11*stateTransition11 + (<double>matrixQ[1,1])

        stateCovarView[0,0] = <cnp.float32_t>priorP00
        stateCovarView[0,1] = <cnp.float32_t>priorP01
        stateCovarView[1,0] = <cnp.float32_t>priorP10
        stateCovarView[1,1] = <cnp.float32_t>priorP11

        sumWeightUU = 0.0
        sumWeightUY = 0.0
        sumWeightYY = 0.0
        sumResidualUU = 0.0
        if returnNLL:
            sumLogR = 0.0

        if doBlockGrad:
            gradSumLogR = 0.0
            gradSumWUU = 0.0
            gradSumWUY = 0.0
            gradSumWYY = 0.0

        for trackIndex in range(trackCount):
            innovationValue = (<double>dataView[trackIndex, intervalIndex]) - (<double>stateVectorView[0])
            measurementVariance = (<double>muncView[trackIndex, intervalIndex])
            paddedVariance = measurementVariance + (<double>pad)
            if paddedVariance < clipSmall:
                paddedVariance = clipSmall
            invVariance = 1.0 / paddedVariance
            if returnNLL:
                sumLogR += log(paddedVariance)
            sumWeightYY += invVariance*(innovationValue*innovationValue)
            sumWeightUY += invVariance*innovationValue
            sumResidualUU += measurementVariance*(invVariance*invVariance)
            sumWeightUU += invVariance

            if doBlockGrad:
                if paddedVariance > clipSmall:
                    dVar = measurementVariance
                else:
                    dVar = 0.0
                gradSumLogR += dVar / paddedVariance
                gradSumWUU += dVar / (paddedVariance*paddedVariance)
                gradSumWUY += dVar * innovationValue / (paddedVariance*paddedVariance)
                gradSumWYY += dVar * (innovationValue*innovationValue) / (paddedVariance*paddedVariance)

        addP00Trace = 1.0 + (<double>stateCovarView[0,0])*sumWeightUU
        if addP00Trace < clipSmall:
            addP00Trace = clipSmall
        weightRank1 = (<double>stateCovarView[0,0]) / addP00Trace
        quadraticForm = sumWeightYY - weightRank1*(sumWeightUY*sumWeightUY)
        if quadraticForm < 0.0:
            quadraticForm = 0.0

        if returnNLL:
        # Quadratic term rewards fit and log-determinant penalizes undue variance inflation.
            intervalNLL = (0.5 *(sumLogR + log(addP00Trace) + quadraticForm))
            sumNLL += intervalNLL

        if doBlockGrad and returnNLL:
            dAddTraceLog = -((<double>stateCovarView[0,0]) * gradSumWUU) / addP00Trace
            dWeightRank1 = (weightRank1*weightRank1) * gradSumWUU
            dQuad = (-gradSumWYY) - (dWeightRank1*(sumWeightUY*sumWeightUY)) + (2.0*weightRank1*sumWeightUY*gradSumWUY)
            intervalGrad = 0.5 * (gradSumLogR + dAddTraceLog + dQuad)

            if isfinite(intervalGrad):
                blockId = intervalToBlockMapView[intervalIndex]
                blockGradLogScaleView[blockId] += <cnp.float32_t>intervalGrad
                blockGradCountView[blockId] += <cnp.float32_t>1.0

        # D stat ~=~ NIS
        dStatValue = quadraticForm / (<double>trackCount)
        sumDStat += dStatValue
        if returnNLL and storeNLLInD:
            dStatVector[intervalIndex] = <cnp.float32_t>intervalNLL
        else:
            dStatVector[intervalIndex] = <cnp.float32_t>dStatValue

        adjustmentCount += <int>(dStatValue > (<double>dStatAlpha))
        if dStatAlpha < 1.0e6:
            matrixQ, inflatedQ = updateProcessNoiseCovariance(
                matrixQ,
                matrixQCopy,
                <float>dStatValue,
                <float>dStatAlpha,
                <float>dStatd,
                <float>dStatPC,
                inflatedQ,
                <float>maxQ,
                <float>minQ
            )

        if matrixQ[0,0] < <cnp.float32_t>clipSmall: matrixQ[0,0] = <cnp.float32_t>clipSmall
        elif matrixQ[0,0] > <cnp.float32_t>clipBig: matrixQ[0,0] = <cnp.float32_t>clipBig
        if matrixQ[1,1] < <cnp.float32_t>clipSmall: matrixQ[1,1] = <cnp.float32_t>clipSmall
        elif matrixQ[1,1] > <cnp.float32_t>clipBig: matrixQ[1,1] = <cnp.float32_t>clipBig

        # 'UPDATE' (posterior, measurement update)
        intermediate_ = sumWeightUY / addP00Trace
        stateVectorView[0] = <cnp.float32_t>((<double>stateVectorView[0]) + (<double>stateCovarView[0,0])*intermediate_)
        stateVectorView[1] = <cnp.float32_t>((<double>stateVectorView[1]) + (<double>stateCovarView[1,0])*intermediate_)
        gainG = sumWeightUU / addP00Trace
        gainH = sumResidualUU / (addP00Trace*addP00Trace)
        IKH00 = 1.0 - ((<double>stateCovarView[0,0])*gainG)
        IKH10 = -((<double>stateCovarView[1,0])*gainG)

        posteriorP00 = <double>stateCovarView[0,0]
        posteriorP01 = <double>stateCovarView[0,1]
        posteriorP10 = <double>stateCovarView[1,0]
        posteriorP11 = <double>stateCovarView[1,1]

        posteriorNew00 = (IKH00*IKH00*posteriorP00) + (gainH*(posteriorP00*posteriorP00))
        posteriorNew01 = (IKH00*(IKH10*posteriorP00 + posteriorP01)) + (gainH*(posteriorP00*posteriorP10))
        posteriorNew11 = ((IKH10*IKH10*posteriorP00) + 2.0*IKH10*posteriorP10 + posteriorP11) + (gainH*(posteriorP10*posteriorP10))

        if posteriorNew00 < clipSmall: posteriorNew00 = clipSmall
        elif posteriorNew00 > clipBig: posteriorNew00 = clipBig
        if posteriorNew01 < clipSmall: posteriorNew01 = clipSmall
        elif posteriorNew01 > clipBig: posteriorNew01 = clipBig
        if posteriorNew11 < clipSmall: posteriorNew11 = clipSmall
        elif posteriorNew11 > clipBig: posteriorNew11 = clipBig

        stateCovarView[0,0] = <cnp.float32_t>posteriorNew00 # next iter's prior
        stateCovarView[0,1] = <cnp.float32_t>posteriorNew01
        stateCovarView[1,0] = <cnp.float32_t>posteriorNew01
        stateCovarView[1,1] = <cnp.float32_t>posteriorNew11

        if projectStateDuringFiltering:
            projectToBox(
                stateVector,
                stateCovar,
                <cnp.float32_t>stateLowerBound,
                <cnp.float32_t>stateUpperBound,
                <cnp.float32_t>clipSmall
            )

        protectCovariance22(stateCovar)

        if doStore:
            stateForwardView[intervalIndex,0] = stateVectorView[0]
            stateForwardView[intervalIndex,1] = stateVectorView[1]
            stateCovarForwardView[intervalIndex,0,0] = stateCovarView[0,0]
            stateCovarForwardView[intervalIndex,0,1] = stateCovarView[0,1]
            stateCovarForwardView[intervalIndex,1,0] = stateCovarView[1,0]
            stateCovarForwardView[intervalIndex,1,1] = stateCovarView[1,1]
            pNoiseForwardView[intervalIndex,0,0] = matrixQ[0,0]
            pNoiseForwardView[intervalIndex,0,1] = matrixQ[0,1]
            pNoiseForwardView[intervalIndex,1,0] = matrixQ[1,0]
            pNoiseForwardView[intervalIndex,1,1] = matrixQ[1,1]
            # memmap flush every chunkSize intervals
            if doFlush and (intervalIndex % chunkSize == 0) and (intervalIndex > 0):
                stateForwardArr.flush()
                stateCovarForwardArr.flush()
                pNoiseForwardArr.flush()

        if doProgress:
            stepsDone += 1
            if (stepsDone % progressIter) == 0:
                progressBar.update(progressIter)

    if doStore and doFlush:
        stateForwardArr.flush()
        stateCovarForwardArr.flush()
        pNoiseForwardArr.flush()

    if doProgress:
        progressRemainder = intervalCount % progressIter
        if progressRemainder != 0:
            progressBar.update(progressRemainder)

    phiHat = <float>(sumDStat / (<double>intervalCount))

    if returnNLL:
        return (phiHat, adjustmentCount, vectorD, sumNLL)

    return (phiHat, adjustmentCount, vectorD)


cpdef tuple cbackwardPass(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForward,
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForward,
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForward,
    object coefficientsH=None,
    bint projectStateDuringFiltering=False,
    float stateLowerBound=0.0,
    float stateUpperBound=0.0,
    float covarClip=3.0,
    Py_ssize_t chunkSize=1000000,
    object stateSmoothed=None,
    object stateCovarSmoothed=None,
    object postFitResiduals=None,
    object progressBar=None,
    Py_ssize_t progressIter=10000
):
    cdef Py_ssize_t stepsDone = 0
    cdef Py_ssize_t progressRemainder = 0
    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] stateTransitionView = matrixF
    cdef cnp.float32_t[:, ::1] stateForwardView = stateForward
    cdef cnp.float32_t[:, :, ::1] stateCovarForwardView = stateCovarForward
    cdef cnp.float32_t[:, :, ::1] pNoiseForwardView = pNoiseForward
    cdef Py_ssize_t trackCount = dataView.shape[0]
    cdef Py_ssize_t intervalCount = dataView.shape[1]
    cdef Py_ssize_t intervalIndex, trackIndex
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateSmoothedArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarSmoothedArr
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] postFitResidualsArr
    cdef cnp.float32_t[:, ::1] stateSmoothedView
    cdef cnp.float32_t[:, :, ::1] stateCovarSmoothedView
    cdef cnp.float32_t[:, ::1] postFitResidualsView
    cdef bint doProgress = False
    cdef bint doFlush = False
    cdef double clipSmall = pow(10.0, -covarClip)
    cdef double clipBig = pow(10.0, covarClip)
    cdef double stateTransition00, stateTransition01, stateTransition10, stateTransition11
    cdef double priorState0, priorState1
    cdef double deltaState0, deltaState1
    cdef double smoothedState0, smoothedState1
    cdef double forwardP00, forwardP01, forwardP10, forwardP11
    cdef double processNoise00, processNoise01, processNoise10, processNoise11
    cdef double priorP00, priorP01, priorP10, priorP11, detPrior, invPrior00, invPrior01, invPrior10, invPrior11, tmp00, tmp01, tmp10, tmp11
    cdef double cross00, cross01, cross10, cross11
    cdef double smootherGain00, smootherGain01, smootherGain10, smootherGain11
    cdef double deltaP00, deltaP01, deltaP10, deltaP11, retrCorrection00, retrCorrection01, retrCorrection10, retrCorrection11
    cdef double smoothedP00, smoothedP01, smoothedP11
    cdef double innovationValue


    if stateSmoothed is not None:
        stateSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateSmoothed
    else:
        stateSmoothedArr = np.empty((intervalCount, 2), dtype=np.float32)

    if stateCovarSmoothed is not None:
        stateCovarSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarSmoothed
    else:
        stateCovarSmoothedArr = np.empty((intervalCount, 2, 2), dtype=np.float32)

    if postFitResiduals is not None:
        postFitResidualsArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> postFitResiduals
    else:
        postFitResidualsArr = np.empty((intervalCount, trackCount), dtype=np.float32)

    stateSmoothedView = stateSmoothedArr
    stateCovarSmoothedView = stateCovarSmoothedArr
    postFitResidualsView = postFitResidualsArr

    doFlush = (chunkSize > 0 and stateSmoothed is not None and stateCovarSmoothed is not None and postFitResiduals is not None)
    doProgress = (progressBar is not None and progressIter > 0)

    stateTransition00 = <double>stateTransitionView[0,0]
    stateTransition01 = <double>stateTransitionView[0,1]
    stateTransition10 = <double>stateTransitionView[1,0]
    stateTransition11 = <double>stateTransitionView[1,1]
    stateSmoothedView[intervalCount - 1, 0] = stateForwardView[intervalCount - 1, 0]
    stateSmoothedView[intervalCount - 1, 1] = stateForwardView[intervalCount - 1, 1]
    stateCovarSmoothedView[intervalCount - 1, 0, 0] = stateCovarForwardView[intervalCount - 1, 0, 0]
    stateCovarSmoothedView[intervalCount - 1, 0, 1] = stateCovarForwardView[intervalCount - 1, 0, 1]
    stateCovarSmoothedView[intervalCount - 1, 1, 0] = stateCovarForwardView[intervalCount - 1, 1, 0]
    stateCovarSmoothedView[intervalCount - 1, 1, 1] = stateCovarForwardView[intervalCount - 1, 1, 1]

    for trackIndex in range(trackCount):
        postFitResidualsView[intervalCount - 1, trackIndex] = <cnp.float32_t>(
            (<double>dataView[trackIndex, intervalCount - 1]) - (<double>stateSmoothedView[intervalCount - 1, 0])
        )

    for intervalIndex in range(intervalCount - 2, -1, -1):

        # from the forward pass
        forwardP00 = <double>stateCovarForwardView[intervalIndex, 0, 0]
        forwardP01 = <double>stateCovarForwardView[intervalIndex, 0, 1]
        forwardP10 = <double>stateCovarForwardView[intervalIndex, 1, 0]
        forwardP11 = <double>stateCovarForwardView[intervalIndex, 1, 1]

        # x[k+1|k]
        priorState0 = stateTransition00*(<double>stateForwardView[intervalIndex,0]) + stateTransition01*(<double>stateForwardView[intervalIndex,1])
        priorState1 = stateTransition10*(<double>stateForwardView[intervalIndex,0]) + stateTransition11*(<double>stateForwardView[intervalIndex,1])
        processNoise00 = <double>pNoiseForwardView[intervalIndex,0,0]
        processNoise01 = <double>pNoiseForwardView[intervalIndex,0,1]
        processNoise10 = <double>pNoiseForwardView[intervalIndex,1,0]
        processNoise11 = <double>pNoiseForwardView[intervalIndex,1,1]

        # intermediates
        tmp00 = stateTransition00*forwardP00 + stateTransition01*forwardP10
        tmp01 = stateTransition00*forwardP01 + stateTransition01*forwardP11
        tmp10 = stateTransition10*forwardP00 + stateTransition11*forwardP10
        tmp11 = stateTransition10*forwardP01 + stateTransition11*forwardP11

        # P[k+1|k]
        priorP00 = tmp00*stateTransition00 + tmp01*stateTransition01 + processNoise00
        priorP01 = tmp00*stateTransition10 + tmp01*stateTransition11 + processNoise01
        priorP10 = tmp10*stateTransition00 + tmp11*stateTransition01 + processNoise10
        priorP11 = tmp10*stateTransition10 + tmp11*stateTransition11 + processNoise11

        detPrior = (priorP00*priorP11) - (priorP01*priorP10)
        if detPrior == 0.0:
            detPrior = clipSmall

        invPrior00 = priorP11/detPrior
        invPrior01 = -priorP01 / detPrior
        invPrior10 = -priorP10 / detPrior
        invPrior11 = priorP00/detPrior
        cross00 = forwardP00*stateTransition00 + forwardP01*stateTransition01
        cross01 = forwardP00*stateTransition10 + forwardP01*stateTransition11
        cross10 = forwardP10*stateTransition00 + forwardP11*stateTransition01
        cross11 = forwardP10*stateTransition10 + forwardP11*stateTransition11

        smootherGain00 = cross00*invPrior00 + cross01*invPrior10
        smootherGain01 = cross00*invPrior01 + cross01*invPrior11
        smootherGain10 = cross10*invPrior00 + cross11*invPrior10
        smootherGain11 = cross10*invPrior01 + cross11*invPrior11

        deltaState0 = (<double>stateSmoothedView[intervalIndex + 1, 0]) - priorState0
        deltaState1 = (<double>stateSmoothedView[intervalIndex + 1, 1]) - priorState1
        smoothedState0 = (<double>stateForwardView[intervalIndex, 0]) + (smootherGain00*deltaState0 + smootherGain01*deltaState1)
        smoothedState1 = (<double>stateForwardView[intervalIndex, 1]) + (smootherGain10*deltaState0 + smootherGain11*deltaState1)
        stateSmoothedView[intervalIndex, 0] = <cnp.float32_t>smoothedState0
        stateSmoothedView[intervalIndex, 1] = <cnp.float32_t>smoothedState1
        deltaP00 = (<double>stateCovarSmoothedView[intervalIndex + 1, 0, 0]) - priorP00
        deltaP01 = (<double>stateCovarSmoothedView[intervalIndex + 1, 0, 1]) - priorP01
        deltaP10 = (<double>stateCovarSmoothedView[intervalIndex + 1, 1, 0]) - priorP10
        deltaP11 = (<double>stateCovarSmoothedView[intervalIndex + 1, 1, 1]) - priorP11
        retrCorrection00 = deltaP00*smootherGain00 + deltaP01*smootherGain01
        retrCorrection01 = deltaP00*smootherGain10 + deltaP01*smootherGain11
        retrCorrection10 = deltaP10*smootherGain00 + deltaP11*smootherGain01
        retrCorrection11 = deltaP10*smootherGain10 + deltaP11*smootherGain11
        smoothedP00 = forwardP00 + (smootherGain00*retrCorrection00 + smootherGain01*retrCorrection10)
        smoothedP01 = forwardP01 + (smootherGain00*retrCorrection01 + smootherGain01*retrCorrection11)
        smoothedP11 = forwardP11 + (smootherGain10*retrCorrection01 + smootherGain11*retrCorrection11)

        if smoothedP00 < clipSmall: smoothedP00 = clipSmall
        elif smoothedP00 > clipBig: smoothedP00 = clipBig
        if smoothedP01 < clipSmall: smoothedP01 = clipSmall
        elif smoothedP01 > clipBig: smoothedP01 = clipBig
        if smoothedP11 < clipSmall: smoothedP11 = clipSmall
        elif smoothedP11 > clipBig: smoothedP11 = clipBig

        stateCovarSmoothedView[intervalIndex, 0, 0] = <cnp.float32_t>smoothedP00
        stateCovarSmoothedView[intervalIndex, 0, 1] = <cnp.float32_t>smoothedP01
        stateCovarSmoothedView[intervalIndex, 1, 0] = <cnp.float32_t>smoothedP01
        stateCovarSmoothedView[intervalIndex, 1, 1] = <cnp.float32_t>smoothedP11

        if projectStateDuringFiltering:
            projectToBox(
                <object>stateSmoothedArr[intervalIndex],
                <object>stateCovarSmoothedArr[intervalIndex],
                <cnp.float32_t>stateLowerBound,
                <cnp.float32_t>stateUpperBound,
                <cnp.float32_t>clipSmall
            )

        protectCovariance22(<object>stateCovarSmoothedArr[intervalIndex])

        for trackIndex in range(trackCount):
            innovationValue = (<double>dataView[trackIndex, intervalIndex]) - (<double>stateSmoothedView[intervalIndex, 0])
            postFitResidualsView[intervalIndex, trackIndex] = <cnp.float32_t>innovationValue

        if doFlush and (intervalIndex % chunkSize == 0) and (intervalIndex > 0):
            stateSmoothedArr.flush()
            stateCovarSmoothedArr.flush()
            postFitResidualsArr.flush()

        if doProgress:
            stepsDone += 1
            if (stepsDone % progressIter) == 0:
                progressBar.update(progressIter)

    if doFlush:
        stateSmoothedArr.flush()
        stateCovarSmoothedArr.flush()
        postFitResidualsArr.flush()

    if doProgress:
        progressRemainder = (intervalCount - 1) % progressIter
        if progressRemainder != 0:
            progressBar.update(progressRemainder)

    return (stateSmoothedArr, stateCovarSmoothedArr, postFitResidualsArr)


cpdef double cDenseGlobalBaseline(
    object x,
    Py_ssize_t bootBlockSize=250,
    Py_ssize_t numBoots=1000,
    double denseMeanQuantile=<double>0.50,
    uint64_t seed=0,
    bint verbose = <bint>False,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] raw, values
    cdef cnp.float32_t[::1] rawView
    cdef cnp.float32_t[::1] valuesView
    cdef Py_ssize_t numValues
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] prefixSums
    cdef cnp.float64_t[::1] prefixView
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] bootstrapMeans
    cdef cnp.float64_t[::1] bootView
    cdef double p, logq_
    cdef Py_ssize_t i, b
    cdef Py_ssize_t remaining, L, start, end
    cdef double sumInReplicate
    cdef double lower__, upper__
    cdef double upperBound
    cdef Py_ssize_t blockSizeUBound
    cdef Py_ssize_t blockSizeLBound = <Py_ssize_t>5
    cdef bint useGeom
    cdef bint trackVerbose = verbose
    cdef long acceptCt = <long>0
    cdef long proposalCt = <long>0
    cdef double lowEPS
    cdef double scale

    if bootBlockSize <= 0:
        bootBlockSize = 1
    if numBoots <= 0:
        return 0.0

    if denseMeanQuantile <= 0.0:
        denseMeanQuantile = 0.0
    elif denseMeanQuantile >= 1.0:
        denseMeanQuantile = 1.0

    raw = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    rawView = raw
    # these values only ~make sense~ if momVST has been called!!!
    values = np.clip(raw, 0.25, 4.0)
    valuesView = values
    numValues = values.size
    prefixSums = np.empty(numValues + 1, dtype=np.float64)
    prefixView = prefixSums
    bootstrapMeans = np.empty(numBoots, dtype=np.float64)
    bootView = bootstrapMeans
    blockSizeUBound = <Py_ssize_t>(5 * bootBlockSize)

    # length L ~ Geometric(p) [truncated to [blockSizeLBound, blockSizeUBound]!]
    useGeom = bootBlockSize > 1
    if useGeom:
        p = 1.0 / (<double>bootBlockSize)
        logq_ = log(1.0 - p)
    else:
        logq_ = 0.0

    # ---------DO NOT CALL THIS FUNCTION IN PARALLEL-------
    srand(seed)
    # ------------------------------------------------------

    with nogil:
        # first pass through to build prefix sums and count zeros
        prefixView[0] = 0.0
        for i in range(numValues):
            prefixView[i + 1] = prefixView[i] + (<double>valuesView[i])

    # Now, start drawing bootstrap replicates
    with nogil:
        for b in range(numBoots):
            # each bootstrap rep: draw random blocks until
            # the total drawn length reaches `numValues`.
            remaining = numValues
            sumInReplicate = 0.0

            while remaining > 0:
                # pick a random start index and block length
                if useGeom:
                    L = _geometricDraw(logq_)
                    if L < blockSizeLBound:
                        L = blockSizeLBound
                    elif L > blockSizeUBound:
                        L = blockSizeUBound
                else:
                    L = 1
                if L > remaining:
                    L = remaining
                if L <= 0:
                    L = 1

                start = _rand_int(numValues)
                if trackVerbose:
                    proposalCt += 1
                    acceptCt += 1

                end = start + L
                if end <= numValues:
                    # No wraparound: sum values[start:end].
                    sumInReplicate += prefixView[end] - prefixView[start]
                else:
                    # circular: sum both (values[start:numValues]) and (values[0:(end - numValues)])
                    end -= numValues
                    sumInReplicate += (prefixView[numValues] - prefixView[start]) + prefixView[end]

                # update the number of remaining values to draw.
                remaining -= L
            bootView[b] = sumInReplicate / (<double>numValues)

    # finally, compute the upper bound quantile from the bootstrap means
    # we pick from the right tail of *bootstrap replicates* (not the original values)
    upperBound = <double>np.quantile(bootstrapMeans, denseMeanQuantile)
    if verbose:
        printf(b"cconsenrich.cDenseGlobalBaseline: Dense baseline = %.4f\n", upperBound)
    return upperBound


cpdef cnp.ndarray[cnp.float32_t, ndim=1] crolling_AR1_IVar(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    int blockLength,
    cnp.ndarray[cnp.uint8_t, ndim=1] excludeMask,
    double maxBeta=0.99,
    double pairsRegLambda = 0.1,
):
    cdef Py_ssize_t numIntervals=values.shape[0]
    cdef Py_ssize_t regionIndex, elementIndex, startIndex,  maxStartIndex
    cdef int halfBlockLength, maskSum
    cdef cnp.ndarray[cnp.float32_t, ndim=1] varAtStartIndex
    cdef cnp.ndarray[cnp.float32_t, ndim=1] varOut
    cdef float[::1] valuesView=values
    cdef cnp.uint8_t[::1] maskView=excludeMask
    cdef double sumY
    cdef double sumSqY
    cdef double sumLagProd
    cdef double nPairsDouble
    cdef double sumXSeq
    cdef double sumYSeq
    cdef double meanX
    cdef double meanYp
    cdef double sumSqXSeq
    cdef double sumSqYSeq
    cdef double sumXYc
    cdef double previousValue
    cdef double currentValue
    cdef double beta1, eps
    cdef double RSS
    cdef double pairCountDouble
    cdef double sumFOD, sumSqFOD, FODLeaving, FODEntering
    cdef double nFODDouble, varFOD
    varOut = np.empty(numIntervals,dtype=np.float32)

    if blockLength > numIntervals:
        blockLength = <int>numIntervals

    if blockLength < 4:
        varOut[:] = 0.0
        return varOut

    halfBlockLength = (blockLength//2)
    maxStartIndex = (numIntervals - blockLength)
    varAtStartIndex = np.empty((maxStartIndex + 1),dtype=np.float32)

    sumY=0.0
    sumSqY=0.0
    sumLagProd=0.0
    maskSum=0
    sumFOD = 0.0
    sumSqFOD = 0.0

    # initialize first
    for elementIndex in range(blockLength):
        currentValue=valuesView[elementIndex]
        sumY += currentValue
        sumSqY += (currentValue*currentValue)
        maskSum += <int>maskView[elementIndex]
        if elementIndex < (blockLength - 1):
            sumLagProd += (currentValue*valuesView[(elementIndex + 1)])

    for elementIndex in range(blockLength - 1):
        FODEntering = valuesView[elementIndex + 1] - valuesView[elementIndex]
        sumFOD += FODEntering
        sumSqFOD += FODEntering * FODEntering

    # sliding window until last block's start
    for startIndex in range(maxStartIndex + 1):
        if maskSum != 0:
            varAtStartIndex[startIndex]=<cnp.float32_t>-1.0
        else:
            nPairsDouble = <double>(blockLength - 1)
            previousValue = valuesView[startIndex]
            currentValue = valuesView[(startIndex + blockLength - 1)]

            # x[i] = values[startIndex+i] i=0,1,...n-2
            # y[i] = values[startIndex+i+1] i=0,1,...n-2
            sumXSeq = sumY - currentValue
            sumYSeq = sumY - previousValue

            # these are distinct now, so means must be computed separately
            meanX = sumXSeq / nPairsDouble
            meanYp = sumYSeq / nPairsDouble
            sumSqXSeq = (sumSqY - (currentValue*currentValue)) - (nPairsDouble*meanX*meanX)
            sumSqYSeq = (sumSqY - (previousValue*previousValue)) - (nPairsDouble*meanYp*meanYp)
            if sumSqXSeq < 0.0:
                sumSqXSeq = 0.0
            if sumSqYSeq < 0.0:
                sumSqYSeq = 0.0

            # sum (x[i] - meanX)*(y[i] - meanYp) i = 0..n-2
            sumXYc = (sumLagProd - (meanYp*sumXSeq) - (meanX*sumYSeq) + (nPairsDouble*meanX*meanYp))
            eps = 1.0e-8*(sumSqXSeq + 1.0)
            if sumSqXSeq > eps:
                # reg. AR(1) coefficient estimate
                beta1 = (sumXYc / (sumSqXSeq + (pairsRegLambda*nPairsDouble)))
            else:
                beta1 = 0.0

            if beta1 > maxBeta:
                beta1 = maxBeta

            # AR(1) negative autocorrelation hides noise here
            elif beta1 < 0.0:
                beta1 = 0.0
            RSS = sumSqYSeq + ((beta1*beta1)*sumSqXSeq) - (2.0*(beta1*sumXYc))
            if RSS < 0.0:
                RSS = 0.0

            # n-1 pairs, slope and intercept estimated --> use df = n-3
            pairCountDouble = <double>(blockLength - 3)
            varAtStartIndex[startIndex]=<cnp.float32_t>(RSS/pairCountDouble)


            nFODDouble = <double>(blockLength - 1)
            # rolling first-order differences variance
            varFOD = sumSqFOD - (sumFOD * sumFOD) / nFODDouble
            if varFOD < 0.0:
                varFOD = 0.0
            if nFODDouble > 1.0:
                varFOD = varFOD / (nFODDouble - 1.0)
            else:
                varFOD = 0.0

            varAtStartIndex[startIndex] = <cnp.float32_t>(
                (<double>varAtStartIndex[startIndex]))
            if varFOD > varAtStartIndex[startIndex]:
                varAtStartIndex[startIndex] = <cnp.float32_t>(
                    varAtStartIndex[startIndex] + 0.5*(varFOD - varAtStartIndex[startIndex])
                )
        if startIndex < maxStartIndex:
            # slide window forward --> (previousSum - leavingValue) + enteringValue
            sumY = (sumY-valuesView[startIndex]) + (valuesView[(startIndex + blockLength)])
            sumSqY = sumSqY + (-(valuesView[startIndex]*valuesView[startIndex]) + (valuesView[(startIndex + blockLength)]*valuesView[(startIndex + blockLength)]))
            sumLagProd = sumLagProd + (-(valuesView[startIndex]*valuesView[(startIndex + 1)]) + (valuesView[(startIndex + blockLength - 1)]*valuesView[(startIndex + blockLength)]))
            maskSum = maskSum + (-<int>maskView[startIndex] + <int>maskView[(startIndex + blockLength)])
            FODLeaving = valuesView[startIndex + 1] - valuesView[startIndex]
            FODEntering = valuesView[startIndex + blockLength] - valuesView[startIndex + blockLength - 1]
            sumFOD += (FODEntering - FODLeaving)
            sumSqFOD += (FODEntering * FODEntering) - (FODLeaving * FODLeaving)

    for regionIndex in range(numIntervals):
        startIndex = regionIndex - blockLength + 1
        if startIndex < 0:
            # flag as invalid (i.e., divert to prior model until full window)
            varOut[regionIndex] = <cnp.float32_t>-1.0
            continue
        if startIndex > maxStartIndex:
            startIndex = maxStartIndex
        varOut[regionIndex] = varAtStartIndex[startIndex]

    return varOut


cpdef cnp.ndarray[cnp.float64_t, ndim=1] cPAVA(
    cnp.ndarray x,
    cnp.ndarray w):
    r"""PAVA for isotonic regression

    This code aims for the notation and algorithm of Busing 2022 (JSS, DOI: 10.18637/jss.v102.c01).

    Key algorithmic insight:

        > Observe that the violation 8 = x3 > x4 = 2 is solved by combining two values, 8 and 2, resulting
        > in a (new) block value of 5, i.e., (8 + 2)/2 = 5. Instead of immediately turning
        > around and start solving down block violation, we may first look ahead for the next
        > value in the sequence, k-up, for if this element is smaller than or equal to 5, the
        > next value can immediately be pooled into the current block, i.e., (8 + 2 + 2)/3 = 4.
        > Looking ahead can be continued until the next element is larger than the current block
        > value or if we reach the end of the sequence.

    :param x: 1D array to be fitted as nondecreasing
    :type x: cnp.ndarray, (either f32 or f64)
    :param w: 1D array of weights corresponding to each observed value.
      These are the number of 'observations' associated to each 'unique' value in `x`. Intuition: more weight to values with more observations.
    :type w: cnp.ndarray
    :return: PAVA-fitted values
    :rtype: cnp.ndarray, (either f32 or f64)
    """

    cdef cnp.ndarray[cnp.float64_t, ndim=1] xArr = np.ascontiguousarray(x, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wArr = np.ascontiguousarray(w, dtype=np.float64).ravel()
    cdef Py_ssize_t n = xArr.shape[0]

    cdef cnp.ndarray[cnp.float64_t, ndim=1] xBlock = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wBlock = np.empty(n, dtype=np.float64)
    # right boundaries for each block
    cdef cnp.ndarray[cnp.int64_t,  ndim=1] rBlock = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] predicted = np.empty(n, dtype=np.float64)
    cdef double[:] xV = xArr
    cdef double[:] wV = wArr
    cdef double[:] xB = xBlock
    cdef double[:] wB = wBlock
    cdef long[:] rB = rBlock
    cdef double[:] predicted_ = predicted
    cdef Py_ssize_t i, k, j, f, t, b
    cdef double xCur, W, S

    b = 1
    xB[0] = xV[0]
    wB[0] = wV[0]
    rB[0] = 0

    i = 1 # index over elements
    while i < n:
        # proceed assuming monotonic+unique: new block at each index
        b += 1
        xCur = xV[i]
        W = wV[i]

        # not monotonic -- discard 'new' block, element goes to a previously existing block
        if xB[b - 2] > xCur:
            # reset
            b -= 1
            S = wB[b - 1]*xB[b - 1] + W*xCur
            W = W + wB[b - 1]
            # update the level/weighted average
            xCur = S / W

            # Busing: until the current pooled level does not break monotonicity, keep merging elements into the block
            while i < n - 1 and xCur >= xV[i + 1]:
                i += 1
                S = S + (wV[i]*xV[i])
                W = W + wV[i]
                xCur = S / W

            # if the now-current block level may break monotonicity with previous block(s) merge backwards
            # ... note that this should only happen once, as we have already ensured monotonicity when creating previous blocks
            while b > 1 and xB[b - 2] > xCur:
                b -= 1
                S = S + (wB[b - 1]*xB[b - 1])
                W = W + wB[b - 1]
                xCur = S / W

        # update block-level stats, boundaries
        xB[b - 1] = xCur
        wB[b - 1] = W
        rB[b - 1] = i
        i += 1

    # We have monotonicity at the --block level-- and right boundaries, xB stored
    # ... now we expand blocks back to get predicted values for all original elements
    f = n - 1
    for k in range(b - 1, -1, -1):
        # case: we hit the first block
        if k == 0:
            # ... so 'next' block starts at index 0
            t = 0
        else:
            # current block's first element is previous block's right boundary + 1
            t = rB[k - 1] + 1
        for j in range(f, t - 1, -1):
            predicted_[j] = xB[k]
        f = t - 1

    return predicted


cpdef tuple cscaleStateCovar(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] postFitResiduals,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixMunc,
    cnp.ndarray[cnp.float32_t, ndim=1] stateVar0,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] intervalToBlockMap,
    Py_ssize_t blockCount,
    float pad=1.0e-3,
    float scaleFactorLow_=1.0,
    float scaleFactorUpper_=10.0,
    float sqResClip=100.0,
    bint allowDeflate=False,
    float clipSmall=1.0e-8,
    double smoothAlpha=0.1,
):
    cdef cnp.float32_t[:, ::1] resView = postFitResiduals
    cdef cnp.float32_t[:, ::1] muncView = matrixMunc
    cdef cnp.float32_t[:] stateVarView = stateVar0
    cdef cnp.int32_t[::1] blockMapView = intervalToBlockMap
    cdef Py_ssize_t n = resView.shape[0]
    cdef Py_ssize_t m = resView.shape[1]
    cdef Py_ssize_t i, t
    cdef Py_ssize_t currBlock = 0
    cdef Py_ssize_t nextBlock
    cdef Py_ssize_t numScalesInBlock = 0
    cdef Py_ssize_t maxBlockLength = 0
    cdef Py_ssize_t blockIntervals = 0
    cdef double chi2Interval
    cdef double residual
    cdef double mVariance
    cdef double totalVariance
    cdef double invVar
    cdef double studentizedResidual
    cdef double sqRes
    cdef float scaleFactor
    cdef float median_

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] blockscaleFactorArr = np.ones(blockCount, dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] blockNArr = np.zeros(blockCount, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] blockChi2Arr = np.zeros(blockCount, dtype=np.float32)
    cdef cnp.float32_t[::1] blockscaleFactorView = blockscaleFactorArr
    cdef cnp.int32_t[::1] blockNView = blockNArr
    cdef cnp.float32_t[::1] blockChi2View = blockChi2Arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] logScale = np.empty(blockCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] logScaleSm = np.empty(blockCount, dtype=np.float64)
    cdef double[::1] logScaleView = logScale
    cdef double[::1] logScaleSmView = logScaleSm
    cdef double s_

    maxBlockLength = 0
    currBlock = <Py_ssize_t>blockMapView[0] if n > 0 else 0
    blockIntervals = 0
    for i in range(n):
        nextBlock = <Py_ssize_t>blockMapView[i]
        if nextBlock != currBlock:
            if blockIntervals > maxBlockLength:
                maxBlockLength = blockIntervals
            blockIntervals = 0
            currBlock = nextBlock
        blockIntervals += 1
    if blockIntervals > maxBlockLength:
        maxBlockLength = blockIntervals
    if maxBlockLength <= 0:
        maxBlockLength = 1

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] calcBuffer = np.empty(maxBlockLength, dtype=np.float32)
    cdef float[::1] bufferView = calcBuffer

    currBlock = <Py_ssize_t>blockMapView[0] if n > 0 else 0


    # numScalesInBlock is both the number of interval-level scale estimates in the current block
    # ... and the index into bufferView to store them before computing the median
    numScalesInBlock = 0

    with nogil:
        for i in range(n):
            nextBlock = <Py_ssize_t>blockMapView[i]
            if nextBlock != currBlock:
                if numScalesInBlock > 0 and currBlock >= 0 and currBlock < blockCount:
                    median_ = _quantileInplaceF32(&bufferView[0], numScalesInBlock, <float>0.5)
                    # block-level scale factor is the median of interval-level scale factors (over all tracks)
                    scaleFactor = median_
                    if (not allowDeflate) and (scaleFactor < 1.0):
                        scaleFactor = 1.0
                    if scaleFactor < scaleFactorLow_:
                        scaleFactor = scaleFactorLow_
                    elif scaleFactor > scaleFactorUpper_:
                        scaleFactor = scaleFactorUpper_
                    blockscaleFactorView[currBlock] = scaleFactor
                    blockNView[currBlock] = <cnp.int32_t>numScalesInBlock
                numScalesInBlock = 0
                currBlock = nextBlock


            chi2Interval = 0.0
            for t in range(m):
                residual = <double>resView[i, t]
                mVariance = <double>muncView[t, i]
                # P_[i,00] + R_[i,jj] + pad
                totalVariance = mVariance + (<double>stateVarView[i]) + (<double>pad)
                if totalVariance < (<double>clipSmall):
                    totalVariance = <double>clipSmall
                invVar = 1.0 / totalVariance
                studentizedResidual = residual*sqrt(invVar)
                sqRes = studentizedResidual*studentizedResidual
                if sqRes > (<double>sqResClip):
                    sqRes = <double>sqResClip
                chi2Interval = chi2Interval + sqRes

            #  check bounds in case interval-->block map from optimization ends up here
            if currBlock >= 0 and currBlock < blockCount:
                scaleFactor = <float>(chi2Interval / (<double>m))
                if numScalesInBlock >= maxBlockLength:
                    with gil:
                        raise RuntimeError("Overflow: is intervalToBlockMap correct?")
                bufferView[numScalesInBlock] = scaleFactor
                numScalesInBlock += 1
                blockChi2View[currBlock] += <cnp.float32_t>chi2Interval

        if numScalesInBlock > 0 and currBlock >= 0 and currBlock < blockCount:
            median_ = _quantileInplaceF32(&bufferView[0], numScalesInBlock, <float>0.5)
            scaleFactor = median_
            if (not allowDeflate) and (scaleFactor < 1.0):
                scaleFactor = 1.0
            if scaleFactor < scaleFactorLow_:
                scaleFactor = scaleFactorLow_
            elif scaleFactor > scaleFactorUpper_:
                scaleFactor = scaleFactorUpper_
            blockscaleFactorView[currBlock] = scaleFactor
            blockNView[currBlock] = <cnp.int32_t>numScalesInBlock

    if blockCount > 1 and smoothAlpha > 0.0:
        # smooth over blocks in log-space to prevent kinks
        if smoothAlpha > 1.0:
            smoothAlpha = 1.0

        with nogil:
            # log-scale
            for i in range(blockCount):
                scaleFactor = blockscaleFactorView[i]
                if (not allowDeflate) and (scaleFactor < 1.0):
                    scaleFactor = 1.0
                if scaleFactor < scaleFactorLow_:
                    scaleFactor = scaleFactorLow_
                elif scaleFactor > scaleFactorUpper_:
                    scaleFactor = scaleFactorUpper_
                if scaleFactor < clipSmall:
                    scaleFactor = clipSmall
                logScaleView[i] = log(<double>scaleFactor)

            _cEMA(<const double*>&logScaleView[0], <double*>&logScaleSmView[0], blockCount, smoothAlpha)

            for i in range(blockCount):
                # make sure our bounds are still in check AFTER cEMA
                s_ = exp(logScaleSmView[i])
                scaleFactor = <float>s_
                if (not allowDeflate) and (scaleFactor < 1.0):
                    scaleFactor = 1.0
                if scaleFactor < scaleFactorLow_:
                    scaleFactor = scaleFactorLow_
                elif scaleFactor > scaleFactorUpper_:
                    scaleFactor = scaleFactorUpper_
                blockscaleFactorView[i] = scaleFactor

    return (blockscaleFactorArr, blockNArr, blockChi2Arr)


cdef void _rmin_F64(double[::1] x, int blockSize, double[::1] out, int[::1] sliding) noexcept nogil:
    cdef int m = <int>x.shape[0]
    cdef int i, first_ = 0, end_ = 0
    cdef int idxDrop
    cdef double xi
    for i in range(m):
        xi = x[i]
        while end_ > first_ and xi <= x[sliding[end_ - 1]]:
            end_ -= 1
        sliding[end_] = i
        end_ += 1
        idxDrop = i - blockSize
        if sliding[first_] <= idxDrop:
            first_ += 1
        if i >= blockSize - 1:
            out[i - (blockSize - 1)] = x[sliding[first_]]


cdef void _rmax_F64(double[::1] x, int blockSize, double[::1] out, int[::1] sliding) noexcept nogil:
    cdef int m = <int>x.shape[0]
    cdef int i, first_ = 0, end_ = 0
    cdef int idxDrop
    cdef double xi
    for i in range(m):
        xi = x[i]
        while end_ > first_ and xi >= x[sliding[end_ - 1]]:
            end_ -= 1
        sliding[end_] = i
        end_ += 1
        idxDrop = i - blockSize
        if sliding[first_] <= idxDrop:
            first_ += 1
        if i >= blockSize - 1:
            out[i - (blockSize - 1)] = x[sliding[first_]]


cdef void _rmin_F32(float[::1] x, int blockSize, float[::1] out, int[::1] sliding) noexcept nogil:
    cdef int m = <int>x.shape[0]
    cdef int i, first_ = 0, end_ = 0
    cdef int idxDrop
    cdef float xi
    for i in range(m):
        xi = x[i]
        while end_ > first_ and xi <= x[sliding[end_ - 1]]:
            end_ -= 1
        sliding[end_] = i
        end_ += 1
        idxDrop = i - blockSize
        if sliding[first_] <= idxDrop:
            first_ += 1
        if i >= blockSize - 1:
            out[i - (blockSize - 1)] = x[sliding[first_]]


cdef void _rmax_F32(float[::1] x, int blockSize, float[::1] out, int[::1] sliding) noexcept nogil:
    cdef int m = <int>x.shape[0]
    cdef int i, first_ = 0, end_ = 0
    cdef int idxDrop
    cdef float xi
    for i in range(m):
        xi = x[i]
        while end_ > first_ and xi >= x[sliding[end_ - 1]]:
            end_ -= 1
        sliding[end_] = i
        end_ += 1
        idxDrop = i - blockSize
        if sliding[first_] <= idxDrop:
            first_ += 1
        if i >= blockSize - 1:
            out[i - (blockSize - 1)] = x[sliding[first_]]


cpdef cnp.ndarray clocalBaseline_F64(cnp.ndarray data, int blockSize, double liftLower=<double>1.0):
    if blockSize < 3 or (blockSize & 1) == 0:
        raise ValueError("need an odd-length block")

    cdef cnp.ndarray y_vec = np.ascontiguousarray(data, dtype=np.float64)
    cdef double[::1] y = y_vec
    cdef int n = <int>y.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.float64)

    cdef int radius = blockSize // 2
    cdef int m = n + 2 * radius
    cdef cnp.ndarray pad_vec = np.empty(m, dtype=np.float64)
    cdef double[::1] padView= pad_vec
    cdef cnp.ndarray sw__vec = np.empty(n, dtype=np.float64)
    cdef double[::1] sw_ = sw__vec
    cdef cnp.ndarray pad2_vec = np.empty(m, dtype=np.float64)
    cdef double[::1] pad2View = pad2_vec
    cdef cnp.ndarray baseline_vec = np.empty(n, dtype=np.float64)
    cdef double[::1] baselineView = baseline_vec
    cdef cnp.ndarray sliding1_vec = np.empty(m, dtype=np.int32)
    cdef int[::1] sliding1 = sliding1_vec
    cdef cnp.ndarray sliding2_vec = np.empty(m, dtype=np.int32)
    cdef int[::1] sliding2View = sliding2_vec
    cdef double med, lift, mad
    cdef cnp.ndarray residuals

    padView[0:radius] = y[0]
    padView[radius:radius + n] = y
    padView[radius + n:m] = y[n - 1]
    with nogil:
        _rmin_F64(padView, blockSize, sw_, sliding1)

    pad2View[0:radius] = sw_[0]
    pad2View[radius:radius + n] = sw_
    pad2View[radius + n:m] = sw_[n - 1]
    with nogil:
        _rmax_F64(pad2View, blockSize, baselineView, sliding2View)
    residuals = y_vec - baseline_vec
    np.maximum(residuals, 0.0, out=residuals)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    lift = liftLower * (mad / 0.6744897501960817)
    np.add(lift, baseline_vec, out=baseline_vec)
    ndimage.median_filter(baseline_vec, size=4*blockSize + 1, mode='nearest', output=baseline_vec)
    return baseline_vec


cpdef cnp.ndarray clocalBaseline_F32(cnp.ndarray data, int blockSize, float liftLower=<float>1.0):
    if blockSize < 3 or (blockSize & 1) == 0:
        raise ValueError("need an odd-length block")

    cdef cnp.ndarray y_vec = np.ascontiguousarray(data, dtype=np.float32)
    cdef float[::1] y = y_vec
    cdef int n = <int>y.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.float32)

    cdef int radius = blockSize // 2
    cdef int m = n + 2 * radius
    cdef cnp.ndarray pad_vec = np.empty(m, dtype=np.float32)
    cdef float[::1] padView = pad_vec
    cdef cnp.ndarray sw__vec = np.empty(n, dtype=np.float32)
    cdef float[::1] sw_ = sw__vec
    cdef cnp.ndarray pad2_vec = np.empty(m, dtype=np.float32)
    cdef float[::1] pad2View = pad2_vec
    cdef cnp.ndarray baseline_vec = np.empty(n, dtype=np.float32)
    cdef float[::1] baselineView = baseline_vec
    cdef cnp.ndarray sliding1_vec = np.empty(m, dtype=np.int32)
    cdef int[::1] sliding1 = sliding1_vec
    cdef cnp.ndarray sliding2_vec = np.empty(m, dtype=np.int32)
    cdef int[::1] sliding2View = sliding2_vec

    padView[0:radius] = y[0]
    padView[radius:radius + n] = y
    padView[radius + n:m] = y[n - 1]
    with nogil:
        _rmin_F32(padView, blockSize, sw_, sliding1)

    pad2View[0:radius] = sw_[0]
    pad2View[radius:radius + n] = sw_
    pad2View[radius + n:m] = sw_[n - 1]
    with nogil:
        _rmax_F32(pad2View, blockSize, baselineView, sliding2View)
    cdef float med, mad, lift
    cdef cnp.ndarray residuals = np.empty(n, dtype=np.float32)
    # MAD of points' residuals *above* the lower envelope
    residuals = y_vec - baseline_vec
    np.maximum(residuals, 0.0, out=residuals)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    lift = liftLower * (mad / 0.6744897501960817)
    np.add(lift, baseline_vec, out=baseline_vec)
    ndimage.median_filter(baseline_vec, size=4*blockSize + 1, mode='nearest', output=baseline_vec)
    return baseline_vec


cpdef clocalBaseline(object x, int blockSize=101, double liftLower=<double>(1.0)):
    arr = np.asarray(x)
    if arr.dtype == np.float64:
        return clocalBaseline_F64(arr, <int>blockSize, liftLower=<double>liftLower)
    if arr.dtype == np.float32:
        return clocalBaseline_F32(arr, <int>blockSize, liftLower=<float>liftLower)
    raise TypeError("x must be np.float32 or np.float64")


cpdef cnp.ndarray[cnp.float64_t, ndim=1] cSF(
    object chromMat,
    float minCount=<float>1.0,
    double nonzeroFrac=0.25,
    bint centerGeoMean=True,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] chromMat_ = np.ascontiguousarray(chromMat, dtype=np.float32)
    cdef Py_ssize_t m = chromMat_.shape[0]
    cdef Py_ssize_t n = chromMat_.shape[1]
    cdef cnp.float32_t[:, ::1] chromMatView = chromMat_

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] refLog = np.empty(n, dtype=np.float64)
    cdef double[::1] refLogView = refLog

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] scaleFactors = np.empty(m, dtype=np.float64)
    cdef double[::1] scaleFactorsView = scaleFactors

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logRatioBuf = np.empty(n, dtype=np.float64)
    cdef double[::1] logRatioBufView = logRatioBuf

    cdef Py_ssize_t s, i, k, needNonzero
    cdef Py_ssize_t presentCount
    cdef double sumLog, v, medLog, geoMean, eps
    eps = 1e-8

    needNonzero = <Py_ssize_t>(nonzeroFrac * (<double>m) + (1.0 - 1e-8))
    if needNonzero < 1:
        needNonzero = 1
    elif needNonzero > m:
        needNonzero = m

    with nogil:
        for i in range(n):
            sumLog = 0.0
            presentCount = 0
            for s in range(m):
                v = <double>chromMatView[s, i]
                if v > <double>minCount:
                    sumLog += log(v)
                    presentCount += 1
            refLogView[i] = (sumLog / (<double>presentCount)) if presentCount >= needNonzero else NAN

        for s in range(m):
            k = 0
            for i in range(n):
                if not isnan(refLogView[i]):
                    v = <double>chromMatView[s, i]
                    if v > <double>minCount:
                        logRatioBufView[k] = log(v) - refLogView[i]
                        k += 1

            if k == 0:
                scaleFactorsView[s] = 1.0
            else:
                _nthElement_F64(&logRatioBufView[0], k, k >> 1)
                medLog = logRatioBufView[k >> 1]
                scaleFactorsView[s] = exp(medLog)

        if centerGeoMean and m > 0:
            sumLog = 0.0
            for s in range(m):
                sumLog += log(scaleFactorsView[s] + eps)
            geoMean = exp(sumLog / (<double>m))
            for s in range(m):
                # center around geometric mean (i.e.,  SF_1 * SF_2 * SF... = 1)
                scaleFactorsView[s] /= geoMean

    return 1/scaleFactors
