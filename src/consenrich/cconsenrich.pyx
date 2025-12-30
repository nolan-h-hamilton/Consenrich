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
from libc.math cimport isfinite, fabs, asinh, sinh, log, asinhf, logf, fmax, fmaxf, pow, sqrt, sqrtf, fabsf, fminf, fmin, log10, log10f, ceil, floor, exp, expf
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
        # ... --> [(high-low)//2 + 1,...,xn-2, high]
        # ... FFR: hard to read, check if bitshift is indeed faster
        midpt = low + ((high - low) >> 1)
        if array_[midpt] <= x:
            low = midpt + 1
        # [low,x1,x2,x3,...,(high-low)//2,...,xn-2, high]
        # ... --> [low,x1,x2,x3,...,(high-low)//2]
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
                                bint useSampleVar) noexcept nogil:
    # CALLERS: cmeanVarPairs
    cdef Py_ssize_t regionIndex, elementIndex, startIndex, blockLength
    cdef double value
    cdef double sumX, sumSqX, sumY, sumXY
    cdef double beta0, beta1
    cdef double fitted, residual, RSS
    cdef double zeroCount, zeroProp, scaleFactor
    cdef double blockLengthDouble
    cdef double previousValue, currentValue
    cdef double scaleFac
    cdef double meanValue
    cdef double pairCountDouble
    cdef double centeredPrev, centeredCurr, oneMinusBetaSq
    cdef double divRSS
    cdef double* blockPtr
    cdef double oneMinusZeroProp
    cdef double maxBeta = <double>0.99
    cdef double sampleVar = <double>0.0

    for regionIndex in range(meanOutView.shape[0]):
        startIndex = blockStartIndices[regionIndex]
        blockLength = blockSizes[regionIndex]
        blockPtr = &valuesView[startIndex]
        blockLengthDouble = <double>blockLength

        sumY = 0.0
        for elementIndex in range(blockLength):
            value = blockPtr[elementIndex]
            sumY += value

        meanValue = sumY / blockLengthDouble
        meanOutView[regionIndex] = <float>meanValue

        if blockLength <= 1:
            varOutView[regionIndex] = 0.0
            continue

        sumX = 0.0
        sumSqX = 0.0
        sumXY = 0.0

        previousValue = blockPtr[0]
        centeredPrev = previousValue - meanValue
        for elementIndex in range(1, blockLength):
            currentValue = blockPtr[elementIndex]
            centeredCurr = currentValue - meanValue

            sumX += centeredPrev
            sumSqX += centeredPrev*centeredPrev
            sumXY += centeredPrev*centeredCurr

            centeredPrev = centeredCurr
            previousValue = currentValue
        pairCountDouble = <double>(blockLength - 1)

        if useSampleVar:
            sumSqX += centeredPrev * centeredPrev # add last element now
            varOutView[regionIndex] = <float>(sumSqX / (blockLengthDouble - 1.0))
            continue

        scaleFac = sumSqX
        if fabs(scaleFac) > 1.0e-2:
            beta1 = sumXY / scaleFac
        else:
            beta1 = 0.0
        beta0 = 0.0

        if beta1 > maxBeta:
            beta1 = maxBeta
        if beta1 < -maxBeta:
            beta1 = -maxBeta

        RSS = 0.0
        previousValue = blockPtr[0]
        centeredPrev = previousValue - meanValue

        for elementIndex in range(1, blockLength):
            currentValue = blockPtr[elementIndex]
            centeredCurr = currentValue - meanValue
            fitted = beta1 * centeredPrev
            residual = centeredCurr - fitted
            RSS += residual * residual

            centeredPrev = centeredCurr
            previousValue = currentValue

        oneMinusBetaSq = 1.0 - (beta1 * beta1)
        if useInnovationVar:
            divRSS = 1.0
        else:
            divRSS = oneMinusBetaSq

        if divRSS <= 1.0e-8:
            varOutView[regionIndex] = 0.0
        else:
            varOutView[regionIndex] = <float>(RSS / pairCountDouble / divRSS)


cdef inline float _carsinh_F32(float x) nogil:
    # CALLERS: `carsinhRatio`

    # arsinh(x / 2) / ln(2) ~~> sign(x) * log2(|x|)
    return asinhf(x/2.0) * __INV_LN2_FLOAT


cdef inline double _carsinh_F64(double x) nogil:
    # CALLERS: `carsinhRatio`

    # arsinh(x / 2) / ln(2) ~~> sign(x) * log2(|x|)
    return asinh(x/2.0) * __INV_LN2_DOUBLE


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
    # CALLERS: `csampleBlockStats`

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


cdef inline double _lossAbsQuadIntercept(
    double absSlope,
    double quadSlope,
    double intercept,
    double sumAbsMu,
    double sumMuSq,
    double sumAbsMuSq,
    double sumAbsMuMuSq,
    double sumMuSqSq,
    double sumZ,
    double sumAbsMuZ,
    double sumMuSqZ,
    double sumSqZ,
    double numSamples,
) nogil:
    # CALLERS: cmonotonicFit

    # (Z - (absSlope*(|mu|) + quadSlope*(mu^2) + intercept*(1)))^2
    cdef double lossVal
    lossVal = sumSqZ
    lossVal -= 2.0 * (absSlope*sumAbsMuZ + quadSlope*sumMuSqZ + intercept*sumZ)
    lossVal += absSlope*absSlope*sumAbsMuSq
    lossVal += 2.0*absSlope*quadSlope*sumAbsMuMuSq
    lossVal += 2.0*absSlope*intercept*sumAbsMu
    lossVal += quadSlope*quadSlope*sumMuSqSq
    lossVal += 2.0*quadSlope*intercept*sumMuSq
    lossVal += intercept*intercept*numSamples
    return lossVal


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


cpdef cnp.ndarray[cnp.float32_t, ndim=2] cinvertMatrixE(
        cnp.ndarray[cnp.float32_t, ndim=1] muncMatrixIter,
        cnp.float32_t priorCovarianceOO,
        cnp.float32_t innovationCovariancePadding=1.0e-2):
    r"""Invert the residual covariance matrix during the forward pass.

    .. todo:: REMOVE (no longer used in the filter iteration)

    :param muncMatrixIter: The diagonal elements of the covariance matrix at a given genomic interval.
    :type muncMatrixIter: cnp.ndarray[cnp.float32_t, ndim=1]
    :param priorCovarianceOO: The a priori 'primary' state variance :math:`P_{[i|i-1,00]} = \left(\mathbf{F}\mathbf{P}_{[i-1\,|\,i-1]}\mathbf{F}^{\top} + Q_[i]\right)_{[00]}`.
    :type priorCovarianceOO: cnp.float32_t
    :param innovationCovariancePadding: Small value added to the diagonal for numerical stability.
    :type innovationCovariancePadding: cnp.float32_t
    :return: The inverted covariance matrix.
    :rtype: cnp.ndarray[cnp.float32_t, ndim=2]
    """

    cdef int m = muncMatrixIter.size
    # we have to invert a P.D. covariance (diagonal) and rank-one (1*priorCovariance) matrix
    cdef cnp.ndarray[cnp.float32_t, ndim=2] inverse = np.empty((m, m), dtype=np.float32)
    # note, not actually an m-dim matrix, just the diagonal elements taken as input
    cdef cnp.ndarray[cnp.float32_t, ndim=1] muncMatrixInverse = np.empty(m, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] muncArr = np.ascontiguousarray(muncMatrixIter, dtype=np.float32, )

    # (numpy) memoryviews for faster indexing + nogil safety
    cdef cnp.float32_t[::1] munc = muncArr
    cdef cnp.float32_t[::1] muncInv = muncMatrixInverse
    cdef cnp.float32_t[:, ::1] inv = inverse


    cdef float divisor = 1.0
    cdef float scale, scaleTimesPrior
    cdef float prior = priorCovarianceOO
    cdef float pad = innovationCovariancePadding
    cdef float inv_i
    cdef float val
    cdef Py_ssize_t i, j

    for i in range(m):
        # two birds: build up the trace while taking the reciprocals
        muncInv[i] = 1.0/(munc[i] + pad)
        divisor += prior*muncInv[i]

    # precompute both scale, scale*prior
    scale = 1.0/divisor
    scaleTimesPrior = scale*prior

    # ----
    # FFR (I): explore prange(...) options to quickly invoke openMP for both cases
    # FFR (II: add nogil block for prange-less case, too?
    # FFR (III): run prange(m, schedule='static', nogil=True)?
    # ----

    # unless sample size warrants it, no OMP here
    if m < 512:
        for i in range(m):
            inv_i = muncInv[i]
            inv[i, i] = inv_i-(scaleTimesPrior*inv_i*inv_i)
            for j in range(i + 1, m):
                val = -scaleTimesPrior*inv_i*muncInv[j]
                inv[i, j] = val
                inv[j, i] = val

    # very large sample size --> prange
    else:
        with nogil:
            for i in prange(m, schedule='static'):
                inv_i = muncInv[i]
                inv[i, i] = inv_i-(scaleTimesPrior*inv_i*inv_i)
                for j in range(i + 1, m):
                    val = -scaleTimesPrior*inv_i*muncInv[j]
                    inv[i, j] = val
                    inv[j, i] = val

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
    cdef float eps = 1e-6  # guard for zeros

    for i in range(n):
        wsum = 0.0
        rwsum = 0.0
        for j in range(m):
            w = 1.0 / (<float>matrixMunc[j, i] + eps)   # weightsIter[j]
            rwsum += (<float>postFitResiduals[i, j])*w  # residualsIter[j]*w
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
        float minQ,
        float dStatAlphaLowMult=0.50,
        float maxMult=2.0):

    cdef float scaleQ, fac, dStatAlphaLow
    cdef float baseSlopeToLevelRatio, maxSlopeQ, minSlopeQ
    cdef float baseOffDiagProd, sqrtDiags, maxNoiseCorr
    cdef float newLevelQ, newSlope_Qnoise, newOffDiagQ
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
            newLevel_Qnoise= fminf(matrixQ[0, 0]*scaleQ, maxQ)
            newSlope_Qnoise = fminf(matrixQ[1, 1]*scaleQ, maxSlopeQ)
            newOffDiagQ = baseOffDiagProd * sqrtf(fmaxf(newLevel_Qnoise* newSlope_Qnoise, eps))
            matrixQ[0, 0] = newLevelQ
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
            newLevel_Qnoise= fmaxf(matrixQ[0, 0] / scaleQ, minQ)
            newSlope_Qnoise = fmaxf(matrixQ[1, 1] / scaleQ, minSlopeQ)
            newOffDiagQ = baseOffDiagProd * sqrtf(fmaxf(newLevel_Qnoise* newSlope_Qnoise, eps))
            matrixQ[0, 0] = newLevelQ
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

    # FFR: can we inline/nogil this?

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

    np.random.seed(randSeed)
    valuesArray = np.ascontiguousarray(values, dtype=np.float64)
    valuesView = valuesArray
    sizesArray = np.full(iters, blockSize, dtype=np.intp)
    outMeans = np.empty(iters, dtype=np.float32)
    outVars = np.empty(iters, dtype=np.float32)
    valuesLength = <Py_ssize_t>valuesArray.size
    maxBlockLength = <Py_ssize_t>blockSize

    supportList = []
    scanIndex = 0

    while scanIndex <= valuesLength - maxBlockLength:
        if excludeIdxMask[scanIndex:scanIndex + maxBlockLength].any():
            scanIndex = scanIndex + maxBlockLength + 1
            continue
        supportList.append(scanIndex)
        scanIndex = scanIndex + 1

    # in case we want to put a distribution on block sizes later,
    # ... e.g., `_blockMax`
    if len(supportList) == 0:
        outMeans[:] = 0.0
        outVars[:] = 0.0
        emptyStarts = np.empty(0, dtype=np.intp)
        emptyEnds = np.empty(0, dtype=np.intp)
        return outMeans, outVars, emptyStarts, emptyEnds

    supportArr = np.asarray(supportList, dtype=np.intp)
    starts_ = np.random.choice(supportArr, size=iters, replace=True).astype(np.intp)
    ends = starts_ + maxBlockLength

    startsView = starts_
    sizesView = sizesArray
    meansView = outMeans
    varsView = outVars

    _regionMeanVar(valuesView, startsView, sizesView, meansView, varsView, zeroPenalty, zeroThresh, useInnovationVar, useSampleVar)

    return outMeans, outVars, starts_, ends


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cmonotonicFit(
    jointlySortedMeans,
    jointlySortedVariances,
    double floorTarget = <double>0.01,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] meanArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1] varArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out
    cdef Py_ssize_t i,n
    cdef double mu
    cdef double absMu
    cdef double muSq
    cdef double varVal
    cdef double zVal
    cdef double sumAbsMu
    cdef double sumMuSq
    cdef double sumAbsMuSq
    cdef double sumAbsMuMuSq
    cdef double sumMuSqSq
    cdef double sumZ
    cdef double sumAbsMuZ
    cdef double sumMuSqZ
    cdef double sumSqZ
    cdef double numSamples
    cdef double candAbsSlope, candQuadSlope, candIntercept
    cdef double bestAbsSlope, bestQuadSlope, bestIntercept
    cdef double bestObjective, currentObjective
    cdef double solvedAbsSlope, solvedQuadSlope, solvedIntercept
    cdef bint solved_
    cdef double minAbsMu, maxAbsMu
    cdef double muCut
    cdef double reqIntercept
    cdef double candReq
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat22
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vec_b2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] solveVec2
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mat33
    cdef cnp.ndarray[cnp.float64_t, ndim=1] vec_b3
    cdef cnp.ndarray[cnp.float64_t, ndim=1] solveVec3
    cdef double det22, det33

    meanArr = np.ascontiguousarray(jointlySortedMeans, dtype=np.float32).ravel()
    varArr  = np.ascontiguousarray(jointlySortedVariances, dtype=np.float32).ravel()
    out = np.empty(3, dtype=np.float32)
    n = meanArr.shape[0]

    if n <= 0:
        out[0] = <cnp.float32_t>0.0
        out[1] = <cnp.float32_t>0.0
        out[2] = <cnp.float32_t>fmax(0.0, floorTarget)
        return out

    sumAbsMu = 0.0
    sumMuSq = 0.0
    sumAbsMuSq = 0.0
    sumAbsMuMuSq = 0.0
    sumMuSqSq = 0.0
    sumZ = 0.0
    sumAbsMuZ = 0.0
    sumMuSqZ = 0.0
    sumSqZ = 0.0
    minAbsMu = 1.0e8
    maxAbsMu = 0.0

    for i in range(n):
        mu = <double>meanArr[i]
        absMu = fabs(mu)
        muSq = mu*mu

        varVal = <double>varArr[i]
        if varVal < 0.0:
            zVal = 0.0
        else:
            zVal = varVal

        if absMu < minAbsMu:
            minAbsMu = absMu
        if absMu > maxAbsMu:
            maxAbsMu = absMu

        sumAbsMu += absMu
        sumMuSq += muSq
        sumAbsMuSq += absMu*absMu
        sumAbsMuMuSq += absMu*muSq
        sumMuSqSq += muSq*muSq

        sumZ += zVal
        sumAbsMuZ += absMu*zVal
        sumMuSqZ += muSq*zVal
        sumSqZ += zVal*zVal

    numSamples = <double>n
    bestAbsSlope = 0.0
    bestQuadSlope = 0.0
    bestIntercept = 0.0

    mat22 = np.empty((2, 2), dtype=np.float64)
    vec_b2 = np.empty(2, dtype=np.float64)

    bestObjective = _lossAbsQuadIntercept(
        0.0, 0.0, 0.0,
        sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
        sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
    )

    candAbsSlope = 0.0
    candQuadSlope = 0.0
    candIntercept = sumZ / numSamples
    if candIntercept < 0.0:
        candIntercept = 0.0

    currentObjective = _lossAbsQuadIntercept(
        candAbsSlope, candQuadSlope, candIntercept,
        sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
        sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
    )
    if currentObjective < bestObjective:
        bestObjective = currentObjective
        bestAbsSlope = candAbsSlope
        bestQuadSlope = candQuadSlope
        bestIntercept = candIntercept

    if sumAbsMuSq > 0.0:
        candAbsSlope = sumAbsMuZ / sumAbsMuSq
    else:
        candAbsSlope = 0.0
    if candAbsSlope < 0.0:
        candAbsSlope = 0.0

    candQuadSlope = 0.0
    candIntercept = 0.0
    currentObjective = _lossAbsQuadIntercept(
        candAbsSlope, candQuadSlope, candIntercept,
        sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
        sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
    )
    if currentObjective < bestObjective:
        bestObjective = currentObjective
        bestAbsSlope = candAbsSlope
        bestQuadSlope = 0.0
        bestIntercept = 0.0

    if sumMuSqSq > 0.0:
        candQuadSlope = sumMuSqZ / sumMuSqSq
    else:
        candQuadSlope = 0.0
    if candQuadSlope < 0.0:
        candQuadSlope = 0.0

    candAbsSlope = 0.0
    candIntercept = 0.0
    currentObjective = _lossAbsQuadIntercept(
        candAbsSlope, candQuadSlope, candIntercept,
        sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
        sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
    )
    if currentObjective < bestObjective:
        bestObjective = currentObjective
        bestAbsSlope = 0.0
        bestQuadSlope = candQuadSlope
        bestIntercept = 0.0

    solvedAbsSlope = 0.0
    solvedQuadSlope = 0.0

    mat22[0, 0] = sumAbsMuSq
    mat22[0, 1] = sumAbsMuMuSq
    mat22[1, 0] = sumAbsMuMuSq
    mat22[1, 1] = sumMuSqSq
    det22 = (mat22[0, 0]*mat22[1, 1]) - (mat22[0, 1]*mat22[1, 0])

    if fabs(det22) < 1.0e-4:
        solvedAbsSlope = 0.0
        solvedQuadSlope = (sumMuSqZ / sumMuSqSq) if fabs(sumMuSqSq) > 1.0e-4 else 0.0
    else:
        vec_b2[0] = sumAbsMuZ
        vec_b2[1] = sumMuSqZ
        solveVec2 = np.linalg.solve(mat22, vec_b2)
        solvedAbsSlope = <double>solveVec2[0]
        solvedQuadSlope = <double>solveVec2[1]

    if solvedAbsSlope >= 0.0 and solvedQuadSlope >= 0.0:
        currentObjective = _lossAbsQuadIntercept(
            solvedAbsSlope, solvedQuadSlope, 0.0,
            sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
            sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
        )
        if currentObjective < bestObjective:
            bestObjective = currentObjective
            bestAbsSlope = solvedAbsSlope
            bestQuadSlope = solvedQuadSlope
            bestIntercept = 0.0

    solvedAbsSlope = 0.0
    solvedIntercept = 0.0

    mat22[0, 0] = sumAbsMuSq
    mat22[0, 1] = sumAbsMu
    mat22[1, 0] = sumAbsMu
    mat22[1, 1] = numSamples
    det22 = (mat22[0, 0]*mat22[1, 1]) - (mat22[0, 1]*mat22[1, 0])

    if fabs(det22) < 1.0e-4:
        solvedAbsSlope = 0.0
        solvedIntercept = (sumZ / numSamples) if fabs(numSamples) > 1.0e-8 else 0.0
    else:
        vec_b2[0] = sumAbsMuZ
        vec_b2[1] = sumZ
        solveVec2 = np.linalg.solve(mat22, vec_b2)
        solvedAbsSlope = <double>solveVec2[0]
        solvedIntercept = <double>solveVec2[1]

    if solvedAbsSlope >= 0.0 and solvedIntercept >= 0.0:
        currentObjective = _lossAbsQuadIntercept(
            solvedAbsSlope, 0.0, solvedIntercept,
            sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
            sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
        )
        if currentObjective < bestObjective:
            bestObjective = currentObjective
            bestAbsSlope = solvedAbsSlope
            bestQuadSlope = 0.0
            bestIntercept = solvedIntercept

    solvedQuadSlope = 0.0
    solvedIntercept = 0.0

    mat22[0, 0] = sumMuSqSq
    mat22[0, 1] = sumMuSq
    mat22[1, 0] = sumMuSq
    mat22[1, 1] = numSamples
    det22 = (mat22[0, 0]*mat22[1, 1]) - (mat22[0, 1]*mat22[1, 0])

    if fabs(det22) < 1.0e-4:
        solvedQuadSlope = 0.0
        solvedIntercept = (sumZ / numSamples) if fabs(numSamples) > 1.0e-8 else 0.0
    else:
        vec_b2[0] = sumMuSqZ
        vec_b2[1] = sumZ
        solveVec2 = np.linalg.solve(mat22, vec_b2)
        solvedQuadSlope = <double>solveVec2[0]
        solvedIntercept = <double>solveVec2[1]

    if solvedQuadSlope >= 0.0 and solvedIntercept >= 0.0:
        currentObjective = _lossAbsQuadIntercept(
            0.0, solvedQuadSlope, solvedIntercept,
            sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
            sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
        )
        if currentObjective < bestObjective:
            bestObjective = currentObjective
            bestAbsSlope = 0.0
            bestQuadSlope = solvedQuadSlope
            bestIntercept = solvedIntercept

    solved_ = <bint>0
    solvedAbsSlope = 0.0
    solvedQuadSlope = 0.0
    solvedIntercept = 0.0

    mat33 = np.empty((3, 3), dtype=np.float64)
    vec_b3 = np.empty(3, dtype=np.float64)
    mat33[0, 0] = sumAbsMuSq
    mat33[0, 1] = sumAbsMuMuSq
    mat33[0, 2] = sumAbsMu
    mat33[1, 0] = sumAbsMuMuSq
    mat33[1, 1] = sumMuSqSq
    mat33[1, 2] = sumMuSq
    mat33[2, 0] = sumAbsMu
    mat33[2, 1] = sumMuSq
    mat33[2, 2] = numSamples

    det33 = (
        mat33[0, 0]*(mat33[1, 1]*mat33[2, 2] - mat33[1, 2]*mat33[2, 1])
        - mat33[0, 1]*(mat33[1, 0]*mat33[2, 2] - mat33[1, 2]*mat33[2, 0])
        + mat33[0, 2]*(mat33[1, 0]*mat33[2, 1] - mat33[1, 1]*mat33[2, 0])
    )

    if fabs(det33) >= 1.0e-4:
        vec_b3[0] = sumAbsMuZ
        vec_b3[1] = sumMuSqZ
        vec_b3[2] = sumZ
        solveVec3 = np.linalg.solve(mat33, vec_b3)
        solvedAbsSlope = <double>solveVec3[0]
        solvedQuadSlope = <double>solveVec3[1]
        solvedIntercept = <double>solveVec3[2]
        solved_ = <bint>1
    else:
        solved_ = <bint>0

    if solved_ and solvedAbsSlope >= 0.0 and solvedQuadSlope >= 0.0 and solvedIntercept >= 0.0:
        currentObjective = _lossAbsQuadIntercept(
            solvedAbsSlope, solvedQuadSlope, solvedIntercept,
            sumAbsMu, sumMuSq, sumAbsMuSq, sumAbsMuMuSq, sumMuSqSq,
            sumZ, sumAbsMuZ, sumMuSqZ, sumSqZ, numSamples,
        )
        if currentObjective < bestObjective:
            bestObjective = currentObjective
            bestAbsSlope = solvedAbsSlope
            bestQuadSlope = solvedQuadSlope
            bestIntercept = solvedIntercept

    if n > 0:
        muCut = <double>floorTarget
        if muCut < minAbsMu:
            muCut = minAbsMu
        if muCut > maxAbsMu:
            muCut = maxAbsMu

        reqIntercept = bestIntercept
        for i in range(n):
            mu = <double>meanArr[i]
            absMu = fabs(mu)
            if absMu > muCut:
                continue

            varVal = <double>varArr[i]
            if varVal < 0.0:
                varVal = 0.0

            muSq = mu*mu
            candReq = varVal - bestAbsSlope*absMu - bestQuadSlope*muSq
            if candReq > reqIntercept:
                reqIntercept = candReq

        if reqIntercept > bestIntercept:
            bestIntercept = reqIntercept

        if bestIntercept < 0.0:
            bestIntercept = 0.0

    out[0] = <cnp.float32_t>bestAbsSlope
    out[1] = <cnp.float32_t>bestQuadSlope
    out[2] = <cnp.float32_t>fmax(bestIntercept, floorTarget)
    return out


cpdef cmonotonicFitEval(
    cnp.ndarray coeffs,
    cnp.ndarray meanTrack,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] meanArr
    cdef Py_ssize_t i,n
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out
    cdef float[:] outView
    cdef float[:] meanView
    cdef cnp.ndarray[cnp.float32_t, ndim=1] coeffArr
    cdef double absSlope
    cdef double quadSlope
    cdef double intercept
    cdef double mu
    cdef double zVal

    meanArr = np.ascontiguousarray(meanTrack, dtype=np.float32).ravel()
    n = meanArr.shape[0]
    out = np.empty(n, dtype=np.float32)
    outView = out
    meanView = meanArr
    coeffArr = np.ascontiguousarray(coeffs, dtype=np.float32).ravel()
    absSlope = <double>coeffArr[0]
    quadSlope = <double>coeffArr[1]
    intercept = <double>coeffArr[2]

    if absSlope < 0.0:
        absSlope = 0.0
    if quadSlope < 0.0:
        quadSlope = 0.0
    if intercept < 0.0:
        intercept = 0.0

    for i in range(n):
        mu = <double>meanView[i]
        zVal = absSlope*fabs(mu) + quadSlope*(mu*mu) + intercept
        if zVal < 0.0:
            zVal = 0.0
        outView[i] = <float>zVal

    return out


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


cpdef object carsinhRatio(object x, Py_ssize_t blockLength,
                          float boundaryEps = <float>0.1,
                          bint disableLocalBackground = <bint>False,
                          bint disableBackground = <bint>False,
                          double scaleCB = 3.0):
    r"""Compute log-scale enrichment versus locally computed backgrounds

    'blocks' are comprised of multiple, contiguous genomic intervals.

    The local background at each genomic interval is obtained by linearly interpolating blocks' mean values between.

    Note that a two-way exponential moving average defines these block means such that autocorrelation
    is tempered between neighboring blocks. Still, we can get a reasonably smooth + locally-informative background.
    """

    cdef cnp.ndarray finalArr__
    cdef Py_ssize_t valuesLength, blockCount, blockIndex, startIndex, endIndex, i, k, blockSize_F32, blockSize_F64, centerIndex
    cdef cnp.ndarray valuesArr_F32
    cdef float[::1] valuesView_F32
    cdef float[::1] outputView_F32
    cdef float[::1] blockMeans_F32
    cdef float[::1] emaView_F32
    cdef float* valuesPtr_F32
    cdef float* outputPtr_F32
    cdef float* blockPtr_F32
    cdef float* emaPtr_F32
    cdef float interpolatedBackground_F32
    cdef float logDiff_F32
    cdef cnp.ndarray valuesArr_F64
    cdef double[::1] valuesView_F64
    cdef double[::1] outputView_F64
    cdef double[::1] blockMeans_F64
    cdef double[::1] emaView_F64
    cdef double* valuesPtr_F64
    cdef double* outputPtr_F64
    cdef double* blockPtr_F64
    cdef double* emaPtr_F64
    cdef double interpolatedBackground_F64
    cdef double logDiff_F64
    cdef float trackWideOffset_F32
    cdef double trackWideOffset_F64
    cdef Py_ssize_t n, tw__, tw_
    cdef double blockCenterCurr,blockCenterNext,lastCenter, edgeWeight
    cdef double carryOver, bgroundEstimate
    cdef double boundaryEps_F64 = <double>boundaryEps
    cdef float boundaryEps_F32 = <float>boundaryEps
    if blockLength <= 0:
        return None
    if <bint>disableBackground and not <bint>disableLocalBackground:
        disableLocalBackground = <bint>True

    if isinstance(x, np.ndarray):
        # F32 case
        if (<cnp.ndarray>x).dtype == np.float32:
            valuesArr_F32 = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
            valuesView_F32 = valuesArr_F32
            valuesLength = valuesView_F32.shape[0]
            finalArr__ = np.zeros(valuesLength, dtype=np.float32)
            outputView_F32 = finalArr__

            if valuesLength == 0:
                return None

            # note last block may have length < blockLength
            blockCount = (valuesLength + blockLength - 1) // blockLength
            if blockCount < 2:
                return None

            # run a two-way exponential moving average weighted such that,
            # ... at each block's --center--, the influence from values at the
            # ... edges has decayed wrt ~boundaryEps~. Assign the mean for *block* `k`
            # ... as the EMA-value at the center of *block* `k`. Then, interpolate
            # ... background estimates for each *interval* `i` within the larger blocks.

            edgeWeight = 1.0 - pow(<double>boundaryEps_F32, 2.0 / (<double>(blockLength + 1)))
            emaView_F32 = cEMA(valuesArr_F32, edgeWeight).astype(np.float32)
            blockMeans_F32 = np.empty(blockCount, dtype=np.float32)
            valuesPtr_F32 = &valuesView_F32[0]
            outputPtr_F32 = &outputView_F32[0]
            emaPtr_F32 = &emaView_F32[0]
            blockPtr_F32  = &blockMeans_F32[0]

            with nogil:
                for blockIndex in range(blockCount):
                    startIndex = blockIndex*blockLength
                    endIndex = startIndex + blockLength
                    if endIndex > valuesLength:
                        endIndex = valuesLength
                    blockSize_F32 = endIndex - startIndex
                    centerIndex = startIndex + (blockSize_F32 // 2)
                    # local-only block mean (clamped)
                    blockPtr_F32[blockIndex] = emaPtr_F32[centerIndex]

            # can't call from nogil
            trackWideOffset_F32 = <float>cgetGlobalBaseline(valuesArr_F32, scaleCB=scaleCB)
            with nogil:
                # 'disable' local background --> use global baseline everywhere
                if <bint>disableLocalBackground == <bint>True:
                    for blockIndex in range(blockCount):
                        blockPtr_F32[blockIndex] = trackWideOffset_F32
                else:
                    for blockIndex in range(blockCount):
                        blockPtr_F32[blockIndex] = fmaxf(trackWideOffset_F32, blockPtr_F32[blockIndex])

                k = <Py_ssize_t>0
                blockCenterCurr = (<double>blockLength)*(<double>k + 0.5)
                blockCenterNext = (<double>blockLength)*(<double>(k + 1) + 0.5)
                lastCenter = (<double>blockLength)*(<double>(blockCount - 1) + 0.5)

                for i in range(valuesLength):
                    if (<double>i) <= blockCenterCurr:
                        interpolatedBackground_F32 = blockPtr_F32[0]
                    elif (<double>i) >= lastCenter:
                        interpolatedBackground_F32 = blockPtr_F32[blockCount - 1]
                    else:
                        while (<double>i) > blockCenterNext and k < blockCount - 2:
                            k += 1
                            blockCenterCurr = blockCenterNext
                            blockCenterNext = (<double>blockLength)*(<double>(k + 1) + 0.5)
                        # interpolate based on position of the interval relative to subsequent block's midpoints
                        # [---------block_k---------|---------block_k+1---------]
                        #                <----------|---------->
                        #                  (c < 1/2)|(c > 1/2)
                        # where c is `carryOver` to the subsequent block mean (see below)
                        carryOver = ((<double>i) - blockCenterCurr) / (blockCenterNext - blockCenterCurr)
                        bgroundEstimate = ((1.0 - carryOver)*(<double>blockPtr_F32[k])) + (carryOver*(<double>blockPtr_F32[k+1]))
                        interpolatedBackground_F32 = <float>(fmax(bgroundEstimate,0.0))

                    if interpolatedBackground_F32 < 0.0:
                        interpolatedBackground_F32 = 0.0


                    # finally, we take ~log-scale~ difference currentValue - background
                    if not <bint>disableBackground:
                        logDiff_F32 = _carsinh_F32(valuesPtr_F32[i]) - _carsinh_F32(interpolatedBackground_F32)
                    else:
                        # case: ChIP w/ input, etc.
                        logDiff_F32 = _carsinh_F32(valuesPtr_F32[i])
                    outputPtr_F32[i] = logDiff_F32

            return finalArr__


        # F64 case, same logic as above
        valuesArr_F64 = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        valuesView_F64 = valuesArr_F64
        valuesLength = valuesView_F64.shape[0]
        finalArr__ = np.zeros(valuesLength, dtype=np.float64)
        outputView_F64 = finalArr__

        if valuesLength == 0:
            return None

        blockCount = (valuesLength + blockLength - 1) // blockLength
        if blockCount < 2:
            return None
        edgeWeight = 1.0 - pow(<double>boundaryEps_F64, 2.0 / (<double>(blockLength + 1)))
        emaView_F64 = cEMA(valuesArr_F64, edgeWeight).astype(np.float64)
        blockMeans_F64 = np.empty(blockCount, dtype=np.float64)

        valuesPtr_F64 = &valuesView_F64[0]
        outputPtr_F64 = &outputView_F64[0]
        emaPtr_F64 = &emaView_F64[0]
        blockPtr_F64  = &blockMeans_F64[0]

        with nogil:
            for blockIndex in range(blockCount):
                startIndex = blockIndex*blockLength
                endIndex = startIndex + blockLength
                if endIndex > valuesLength:
                    endIndex = valuesLength
                blockSize_F64 = endIndex - startIndex
                centerIndex = startIndex + (blockSize_F64 // 2)
                blockPtr_F64[blockIndex] = emaPtr_F64[centerIndex]


        trackWideOffset_F64 = <double>cgetGlobalBaseline(valuesArr_F64, scaleCB=scaleCB)

        with nogil:
            if <bint>disableLocalBackground == <bint>True:
                for blockIndex in range(blockCount):
                    blockPtr_F64[blockIndex] = trackWideOffset_F64
            else:
                for blockIndex in range(blockCount):
                    blockPtr_F64[blockIndex] = fmax(blockPtr_F64[blockIndex], trackWideOffset_F64)

            k = 0
            blockCenterCurr = (<double>blockLength)*(<double>k + 0.5)
            blockCenterNext = (<double>blockLength)*(<double>(k + 1) + 0.5)
            lastCenter = (<double>blockLength)*(<double>(blockCount - 1) + 0.5)
            for i in range(valuesLength):
                if (<double>i) <= blockCenterCurr:
                    interpolatedBackground_F64 = blockPtr_F64[0]
                elif (<double>i) >= lastCenter:
                    interpolatedBackground_F64 = blockPtr_F64[blockCount - 1]
                else:
                    while (<double>i) > blockCenterNext and k < blockCount - 2:
                        k += 1
                        blockCenterCurr = blockCenterNext
                        blockCenterNext = (<double>blockLength)*(<double>(k + 1) + 0.5)

                    carryOver = ((<double>i) - blockCenterCurr) / (blockCenterNext - blockCenterCurr)
                    bgroundEstimate = ((1.0 - carryOver)*blockPtr_F64[k]) + (carryOver*blockPtr_F64[k+1])
                    interpolatedBackground_F64 = <double>(fmax(bgroundEstimate, 0.0))

                if interpolatedBackground_F64 < 0.0:
                    interpolatedBackground_F64 = 0.0

                if not <bint>disableBackground:
                    logDiff_F64 = _carsinh_F64(valuesPtr_F64[i]) - _carsinh_F64(interpolatedBackground_F64)
                else:
                    logDiff_F64 = _carsinh_F64(valuesPtr_F64[i])

                outputPtr_F64[i] = logDiff_F64


        return finalArr__

    return None


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


                # rewrite/pad given 2x2 + SPD (and pad):
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
        if matrixQ[0,1] < <cnp.float32_t>clipSmall: matrixQ[0,1] = <cnp.float32_t>clipSmall
        elif matrixQ[0,1] > <cnp.float32_t>clipBig: matrixQ[0,1] = <cnp.float32_t>clipBig
        if matrixQ[1,0] < <cnp.float32_t>clipSmall: matrixQ[1,0] = <cnp.float32_t>clipSmall
        elif matrixQ[1,0] > <cnp.float32_t>clipBig: matrixQ[1,0] = <cnp.float32_t>clipBig
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


cpdef double cgetGlobalBaseline(
    object x,
    Py_ssize_t bootBlockSize=250,
    Py_ssize_t numBoots=5000,
    double scaleCB=3.0,
    uint64_t seed=0,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] values
    cdef cnp.float32_t[::1] valuesView
    cdef Py_ssize_t numValues
    cdef object rng
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] blockSizes
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] blockStarts
    cdef cnp.intp_t[::1] blockSizesView
    cdef cnp.intp_t[::1] blockStartsView
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] prefixSums
    cdef cnp.float64_t[::1] prefixView
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] bootstrapMeans
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] posMeans
    cdef double bootMean, bootStdErr
    cdef double geomProb, upperBound
    cdef double maxVal
    cdef Py_ssize_t i
    cdef Py_ssize_t numPos

    if bootBlockSize <= 0:
        bootBlockSize = 1
    if numBoots <= 0:
        return 0.0


    values = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    valuesView = values
    numValues = values.size
    if numValues <= 0:
        return 0.0

    rng = default_rng(seed)
    geomProb = 1.0 / (<double>bootBlockSize)
    blockSizes = rng.geometric(geomProb, size=numBoots).astype(np.intp, copy=False)
    np.minimum(blockSizes, numValues, out=blockSizes)
    blockSizesView = blockSizes
    blockStarts = (rng.random(numBoots) * (numValues - blockSizes + 1)).astype(np.intp, copy=False)
    blockStartsView = blockStarts
    prefixSums = np.empty(numValues + 1, dtype=np.float64)
    prefixView = prefixSums
    prefixView[0] = 0.0
    for i in range(numValues):
        prefixView[i + 1] = prefixView[i] + (<double>valuesView[i])

    # prefix[start + size] - prefix[start] yields block's sum
    bootstrapMeans = (prefixSums[blockStarts + blockSizes] - prefixSums[blockStarts]) / blockSizes
    posMeans = bootstrapMeans
    numPos = posMeans.size
    if numPos <= 0:
        return 0.0

    bootMean = <double>np.mean(posMeans)
    bootStdErr = (<double>np.std(posMeans, ddof=1) / np.sqrt(<double>numPos))
    maxVal = <double>np.max(values)
    upperBound = fmin(bootMean + (scaleCB * bootStdErr), maxVal * 0.995)
    return upperBound


cpdef cnp.ndarray[cnp.float32_t, ndim=1] crolling_AR1_IVar(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    int blockLength,
    cnp.ndarray[cnp.uint8_t, ndim=1] excludeMask,
    double maxBeta=0.99,
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
    cdef double meanValue
    cdef double sumSqCenteredTotal
    cdef double previousValue
    cdef double currentValue
    cdef double centeredFirst
    cdef double centeredLast
    cdef double sumSqX
    cdef double sumSqNext
    cdef double sumXY
    cdef double beta1
    cdef double RSS
    cdef double pairCountDouble
    varOut = np.empty(numIntervals,dtype=np.float32)

    if numIntervals <= 0:
        return varOut
    if blockLength < 2:
        varOut[:] = 0.0
        return varOut
    if blockLength > numIntervals:
        blockLength = <int>numIntervals

    halfBlockLength = (blockLength//2)
    maxStartIndex = (numIntervals - blockLength)
    varAtStartIndex = np.empty((maxStartIndex + 1),dtype=np.float32)

    sumY=0.0
    sumSqY=0.0
    sumLagProd=0.0
    maskSum=0

    # initialize first block
    for elementIndex in range(blockLength):
        currentValue=valuesView[elementIndex]
        sumY += currentValue
        sumSqY += (currentValue*currentValue)
        maskSum += <int>maskView[elementIndex]
        if elementIndex < (blockLength - 1):
            sumLagProd += (currentValue*valuesView[(elementIndex + 1)])
    pairCountDouble=<double>(blockLength - 1)


    # sliding window until last block's start
    for startIndex in range(maxStartIndex + 1):
        if maskSum != 0:
            varAtStartIndex[startIndex]=<cnp.float32_t>-1.0
        else:
            previousValue = valuesView[startIndex]
            currentValue = valuesView[(startIndex + blockLength - 1)]
            meanValue = (sumY/(<double>blockLength))
            sumSqCenteredTotal = sumSqY - ((<double>blockLength)*meanValue*meanValue)
            centeredFirst = (previousValue - meanValue)
            centeredLast = (currentValue - meanValue)
            sumSqNext = (sumSqCenteredTotal - (centeredFirst*centeredFirst))
            sumSqX = sumSqCenteredTotal - (centeredLast*centeredLast)
            sumXY=(
                sumLagProd
                - meanValue*(2.0*sumY - previousValue - currentValue)
                + (<double>(blockLength - 1))*meanValue*meanValue
            )

            if fabs(sumSqX) > 1.0e-4:
                # within-block AR(1) term
                # centered x: (x[i+1]*x[i]) / x[i]^2
                beta1=(sumXY/sumSqX)
            else:
                beta1=0.0

            if beta1 > maxBeta:
                beta1=maxBeta
            elif beta1 < -maxBeta:
                beta1=-maxBeta

            RSS = sumSqNext + ((beta1*beta1)*sumSqX) - (2.0*(beta1*sumXY))
            if RSS < 0.0:
                RSS=0.0
            varAtStartIndex[startIndex]=<cnp.float32_t>(RSS/pairCountDouble)

        if startIndex < maxStartIndex:
            # slide window forward --> (previousSum - leavingValue) + enteringValue
            sumY = (sumY-valuesView[startIndex]) + (valuesView[(startIndex + blockLength)])
            sumSqY = sumSqY + (-(valuesView[startIndex]*valuesView[startIndex]) + (valuesView[(startIndex + blockLength)]*valuesView[(startIndex + blockLength)]))
            sumLagProd = sumLagProd + (-(valuesView[startIndex]*valuesView[(startIndex + 1)]) + (valuesView[(startIndex + blockLength - 1)]*valuesView[(startIndex + blockLength)]))
            maskSum = maskSum + (-<int>maskView[startIndex] + <int>maskView[(startIndex + blockLength)])

    for regionIndex in range(numIntervals):
        startIndex=(regionIndex - halfBlockLength)
        if startIndex < 0:
            startIndex=0
        elif startIndex > maxStartIndex:
            startIndex=maxStartIndex
        varOut[regionIndex]=varAtStartIndex[startIndex]

    return varOut
