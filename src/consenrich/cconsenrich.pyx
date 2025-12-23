# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
# distutils: language = c
r"""Cython module for Consenrich core functions.

This module contains Cython implementations of core functions used in Consenrich.
"""

cimport cython
import os
import numpy as np
from numpy._core.multiarray import ndarray
from scipy import ndimage
cimport numpy as cnp
from libc.stdint cimport int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from pysam.libcalignmentfile cimport AlignmentFile, AlignedSegment
from libc.float cimport DBL_EPSILON
from numpy.random import default_rng
from cython.parallel import prange
from libc.math cimport isfinite, fabs, asinh, sinh, log, asinhf, logf, fmax, fmaxf, pow, sqrt, sqrtf, fabsf, fminf, fmin, log10, log10f
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


cdef inline bint _solveSystem22(
    double a00, double a01, double a10, double a11,
    double b0, double b1,
    double* x0, double* x1
) noexcept nogil:
    # CALLERS: `cmonotonicFit`

    cdef double det = (a00*a11) - (a01*a10)
    if fabs(det) < 1.0e-4:
        x0[0] = 0.0
        x1[0] = b1 / a11 if fabs(a11) > 1.0e-8 else 0.0
        return <bint>1
    x0[0] = ( b0*a11 - a01*b1) / det
    x1[0] = (-b0*a10 + a00*b1) / det
    return <bint>0


cdef inline double _loss(
    double slope, double intercept,
    double sumX, double sumSqX,
    double sumZ, double sumXZ,
    double sumSqZ, double numSamples,
) noexcept:
    # CALLERS: `cmonotonicFit`

    cdef double loss = 0.0

    loss = (
        sumSqZ
        - 2.0*slope*sumXZ
        - 2.0*intercept*sumZ
        + (slope*slope)*sumSqX
        + 2.0*slope*intercept*sumX
        + (intercept*intercept)*numSamples
    )
    return loss


cdef inline void _regionMeanVar(double[::1] valuesView,
                                Py_ssize_t[::1] blockStartIndices,
                                Py_ssize_t[::1] blockSizes,
                                float[::1] meanOutView,
                                float[::1] varOutView,
                                double zeroPenalty,
                                double zeroThresh) noexcept nogil:
    # CALLERS: `cmeanVarPairs`

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
    cdef double* blockPtr # FFR: try to use this convention to avoid indexing as [start + idx]
    cdef double oneMinusZeroProp
    cdef double maxBeta = <double>0.99

    # fit AR(1) to each block, take RSS/(N-1)/(1-beta^2) as var estimate
    # ... block sizes should be sized such that AR(1) assumptions are plausible
    # ... on the average...See caller
    for regionIndex in range(meanOutView.shape[0]):
        startIndex = blockStartIndices[regionIndex]
        blockLength = blockSizes[regionIndex]
        blockPtr = &valuesView[startIndex]
        blockLengthDouble = <double>blockLength

        sumY = 0.0
        zeroCount = 0.0
        for elementIndex in range(blockLength):
            value = blockPtr[elementIndex]
            sumY += value
            if fabs(value) < zeroThresh:
                zeroCount += 1.0

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

        pairCountDouble = <double>(blockLength-1)
        scaleFac = sumSqX

        if fabs(scaleFac) > 1.0e-2:
            beta1 = sumXY / scaleFac
        else:
            beta1 = 0.0
        beta0 = 0.0

        if beta1 >  maxBeta:
            beta1 =  maxBeta
        if beta1 < -maxBeta:
            beta1 = -maxBeta

        RSS = 0.0
        previousValue = blockPtr[0]
        centeredPrev = previousValue - meanValue

        for elementIndex in range(1, blockLength):
            currentValue = blockPtr[elementIndex]
            centeredCurr = currentValue - meanValue
            fitted = beta1*centeredPrev
            residual = centeredCurr - fitted
            RSS += residual*residual

            centeredPrev = centeredCurr
            previousValue = currentValue

        oneMinusBetaSq = (1.0 - (beta1*beta1))
        varOutView[regionIndex] = <float>((RSS/pairCountDouble/oneMinusBetaSq))


cdef inline float _carsinh_F32(float x) nogil:
    # CALLERS: `carsinhRatio`

    # arsinh(x / 2) / ln(2) ~~> sign(x) * log2(|x|)
    return asinhf(x/2.0) * __INV_LN2_FLOAT


cdef inline double _carsinh_F64(double x) nogil:
    # CALLERS: `carsinhRatio`

    # arsinh(x / 2) / ln(2) ~~> sign(x) * log2(|x|)
    return asinh(x/2.0) * __INV_LN2_DOUBLE


cdef inline double _interpolateQuantile_F32(float[::1] sortedValues, Py_ssize_t valueCount, double quantile) nogil:
    # CALLERS: `_kneeGrad_F32`

    cdef double scaledIndex = quantile * (<double>(valueCount - 1))
    cdef Py_ssize_t leftIndex = <Py_ssize_t>scaledIndex
    cdef Py_ssize_t rightIndex = leftIndex + 1
    cdef double fraction
    if rightIndex >= valueCount:
        rightIndex = leftIndex
    fraction = scaledIndex - <double>leftIndex
    return (<double>sortedValues[leftIndex]) * (1.0 - fraction) + (<double>sortedValues[rightIndex]) * fraction


cdef inline double _interpolateQuantile_F64(double[::1] sortedValues, Py_ssize_t valueCount, double quantile) nogil:
    # CALLERS: `_kneeGrad_F64`

    cdef double scaledIndex = quantile * (<double>(valueCount - 1))
    cdef Py_ssize_t leftIndex = <Py_ssize_t>scaledIndex
    cdef Py_ssize_t rightIndex = leftIndex + 1
    cdef double fraction
    if rightIndex >= valueCount:
        rightIndex = leftIndex
    fraction = scaledIndex - <double>leftIndex
    return sortedValues[leftIndex] * (1.0 - fraction) + sortedValues[rightIndex] * fraction


cdef inline double _kneeGrad_F32(float[::1] sortedValues,
                                    Py_ssize_t valueCount,
                                    double quantile,
                                    double inverseQuantileSpan,
                                    double inverseValueSpan) nogil:
    # CALLERS: `cgetGlobalBaseline`

    cdef double quantileStep
    cdef double halfWindow
    cdef double leftQuantile
    cdef double rightQuantile
    cdef double leftValue
    cdef double rightValue
    cdef double valueSlope

    if valueCount < 2:
        return -inverseQuantileSpan

    quantileStep = 1.0 / (<double>(valueCount - 1))
    halfWindow = 4.0 * quantileStep
    if halfWindow < 1e-4:
        halfWindow = 1e-4
    if halfWindow > 1e-2:
        halfWindow = 1e-2

    leftQuantile = quantile - halfWindow
    rightQuantile = quantile + halfWindow
    if leftQuantile < 0.0:
        leftQuantile = 0.0
    if rightQuantile > 1.0:
        rightQuantile = 1.0
    if rightQuantile <= leftQuantile:
        return -inverseQuantileSpan

    leftValue = _interpolateQuantile_F32(sortedValues, valueCount, leftQuantile)
    rightValue = _interpolateQuantile_F32(sortedValues, valueCount, rightQuantile)
    valueSlope = (rightValue - leftValue) / (rightQuantile - leftQuantile)

    return (valueSlope * inverseValueSpan) - inverseQuantileSpan


cdef inline double _kneeGrad_F64(double[::1] sortedValues,
                                    Py_ssize_t valueCount,
                                    double quantile,
                                    double inverseQuantileSpan,
                                    double inverseValueSpan) nogil:
    # CALLERS: `cgetGlobalBaseline`

    cdef double quantileStep
    cdef double halfWindow
    cdef double leftQuantile
    cdef double rightQuantile
    cdef double leftValue
    cdef double rightValue
    cdef double valueSlope

    if valueCount < 2:
        return -inverseQuantileSpan

    quantileStep = 1.0 / (<double>(valueCount - 1))
    halfWindow = 4.0 * quantileStep
    if halfWindow < 1e-4:
        halfWindow = 1e-4
    if halfWindow > 1e-2:
        halfWindow = 1e-2

    leftQuantile = quantile - halfWindow
    rightQuantile = quantile + halfWindow
    if leftQuantile < 0.0:
        leftQuantile = 0.0
    if rightQuantile > 1.0:
        rightQuantile = 1.0
    if rightQuantile <= leftQuantile:
        return -inverseQuantileSpan

    leftValue = _interpolateQuantile_F64(sortedValues, valueCount, leftQuantile)
    rightValue = _interpolateQuantile_F64(sortedValues, valueCount, rightQuantile)
    valueSlope = (rightValue - leftValue) / (rightQuantile - leftQuantile)
    return (valueSlope * inverseValueSpan) - inverseQuantileSpan



cpdef int stepAdjustment(int value, int stepSize, int pushForward=0):
    r"""Adjusts a value to the nearest multiple of stepSize, optionally pushing it forward.

    .. todo:: refactor caller + this function into one cython func

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
    int64_t inferFragmentLength=0,
    int64_t minMappingQuality=0,
    int64_t minTemplateLength=-1,
    uint8_t weightByOverlap=1,
    ):
    r"""Count reads in a BAM file for a given chromosome"""

    cdef Py_ssize_t numIntervals
    cdef int64_t width = <int64_t>end - <int64_t>start

    if stepSize <= 0 or width <= 0:
        numIntervals = 0
    else:
        numIntervals = <Py_ssize_t>((width + stepSize - 1) // stepSize)

    cdef cnp.ndarray[cnp.float32_t, ndim=1] values_np = np.zeros(numIntervals, dtype=np.float32)
    cdef cnp.float32_t[::1] values = values_np

    if numIntervals <= 0:
        return values

    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    cdef int64_t start64 = start
    cdef int64_t end64 = end
    cdef int64_t step64 = stepSize
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
    if dStatAlphaLowMult <= 0:
        dStatAlphaLow = 1.0
    else:
        dStatAlphaLow = dStatAlpha*dStatAlphaLowMult
    if dStatAlphaLow >= dStatAlpha:
        dStatAlphaLow = dStatAlpha

    if dStat > dStatAlpha:
        scaleQ = fminf(sqrtf(dStatd*fabsf(dStat - dStatAlpha) + dStatPC), maxMult)
        if matrixQ[0, 0]*scaleQ <= maxQ:
            matrixQ[0, 0] *= scaleQ
            matrixQ[0, 1] *= scaleQ
            matrixQ[1, 0] *= scaleQ
            matrixQ[1, 1] *= scaleQ
        else:
            fac = maxQ/matrixQCopy[0, 0]
            matrixQ[0, 0] = maxQ
            matrixQ[0, 1] = matrixQCopy[0, 1]*fac
            matrixQ[1, 0] = matrixQCopy[1, 0]*fac
            matrixQ[1, 1] = maxQ
        inflatedQ = True

    elif dStat <= dStatAlphaLow and inflatedQ:
        scaleQ = fminf(sqrtf(dStatd*fabsf(dStat - dStatAlphaLow) + dStatPC), maxMult)
        if matrixQ[0, 0] / scaleQ >= minQ:
            matrixQ[0, 0] /= scaleQ
            matrixQ[0, 1] /= scaleQ
            matrixQ[1, 0] /= scaleQ
            matrixQ[1, 1] /= scaleQ
        else:
            # we've hit the minimum, no longer 'inflated'
            fac = minQ / matrixQCopy[0, 0]
            matrixQ[0, 0] = minQ
            matrixQ[0, 1] = matrixQCopy[0, 1]*fac
            matrixQ[1, 0] = matrixQCopy[1, 0]*fac
            matrixQ[1, 1] = minQ
            inflatedQ = False

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


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cSparseMax(cnp.float32_t[::1] trackALV, dict sparseMap):
    r"""Fast access and max of `numNearest` sparse elements.

    See :func:`consenrich.core.getMuncTrack`

    :param sparseMap: See :func:`consenrich.core.getSparseMap`
    :type sparseMap: dict[int, np.ndarray]
    :return: array of mena('nearest local variances') same length as `trackALV`
    :rtype: cnp.ndarray[cnp.float32_t, ndim=1]
    """
    cdef Py_ssize_t n = <Py_ssize_t>trackALV.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)
    cdef Py_ssize_t i, j, m
    cdef float maxNearestVariances = 0.0
    cdef cnp.ndarray[cnp.intp_t, ndim=1] idxs
    cdef cnp.intp_t[::1] idx_view
    for i in range(n):
        idxs = <cnp.ndarray[cnp.intp_t, ndim=1]> sparseMap[i]
        idx_view = idxs
        m = idx_view.shape[0]
        maxNearestVariances = 0.0
        with nogil:
            # find max in numNearest sparse regions
            for j in range(m):
                if trackALV[idx_view[j]] > maxNearestVariances:
                    maxNearestVariances = trackALV[idx_view[j]]
        out[i] = maxNearestVariances

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

    # FFR: standardize, across codebase, random seeding (e.g., np.random.seed vs default_rng)
    cdef object rng = default_rng(randSeed)
    cdef int64_t regionLen, numRollSteps
    cdef int numChunks
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rawArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] medArr
    cdef AlignmentFile aln
    cdef AlignedSegment readSeg
    cdef list coverageIdxTopK
    cdef list blockCenters
    cdef list bestLags
    cdef int i, j, k, idxVal
    cdef int startIdx, endIdx
    cdef int winSize, takeK
    cdef int blockHalf, readFlag
    cdef int chosenLag, lag, maxValidLag
    cdef int strand
    cdef int expandedLen
    cdef int samThreadsInternal
    cdef int cpuCount
    cdef int64_t blockStartBP, blockEndBP, readStart, readEnd
    cdef int64_t med
    cdef double score
    cdef cnp.ndarray[cnp.intp_t, ndim=1] unsortedIdx, sortedIdx, expandedIdx
    cdef cnp.intp_t[::1] expandedIdxView
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

    # rather than taking `chromosome`, `start`, `end`
    # ... we will just look at BAM contigs present and use
    # ... the three largest to estimate the fragment length
    cdef tuple contigs
    cdef tuple lengths
    cdef Py_ssize_t contigIdx
    cdef str contig
    cdef int64_t contigLen
    cdef object top2ContigsIdx

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

    earlyExit = min(earlyExit, iters)

    samThreadsInternal = <int>samThreads
    cpuCount = <uint32_t>os.cpu_count()
    if cpuCount is None:
        cpuCount = 1
    if samThreads < 1:
        samThreadsInternal = <int>min(max(1,cpuCount // 2), 4)

    aln = AlignmentFile(bamFile, "rb", threads=samThreadsInternal)
    try:
        contigs = aln.references
        lengths = aln.lengths

        if contigs is None or len(contigs) == 0:
            return <int64_t>fallBack

        top2ContigsIdx = np.argsort(lengths)[-min(2, len(contigs)):]

        for contigIdx in top2ContigsIdx:
            contig = contigs[contigIdx]
            for readSeg in aln.fetch(contig):
                if (readSeg.flag & samFlagExclude) != 0:
                    continue
                if numReadLengthSamples < iters:
                    avgReadLength += readSeg.query_length
                    numReadLengthSamples += 1
                else:
                    break

        avgReadLength /= numReadLengthSamples if numReadLengthSamples > 0 else 1
        minInsertSize = <int64_t>(avgReadLength + 0.5)
        if minInsertSize < 1:
            minInsertSize = 1
        if minInsertSize > maxInsertSize:
            minInsertSize = maxInsertSize

        for contigIdx in top2ContigsIdx:
            contig = contigs[contigIdx]
            for readSeg in aln.fetch(contig):
                if (readSeg.flag & samFlagExclude) != 0:
                    continue
                if readSeg.is_paired:
                    # skip to the paired-end block below (no xCorr --> average template len)
                    isPairedEnd = <bint>1
                    break
            if isPairedEnd:
                break

        if isPairedEnd:
            requiredSamplesPE = max(iters, 1000)

            for contigIdx in top2ContigsIdx:
                if templateLenSamples >= requiredSamplesPE:
                    break
                contig = contigs[contigIdx]

                for readSeg in aln.fetch(contig):
                    if templateLenSamples >= requiredSamplesPE:
                        break
                    if (readSeg.flag & samFlagExclude) != 0 or (readSeg.flag & 2) == 0:
                        # skip any excluded flags, only count proper pairs
                        continue
                    if readSeg.template_length > 0 and readSeg.is_read1:
                        # read1 only: otherwise each pair contributes to the mean twice
                        # ...which might reduce breadth of the estimate
                        avgTemplateLen += abs(readSeg.template_length)
                        templateLenSamples += 1

            if templateLenSamples < requiredSamplesPE:
                return <int64_t> fallBack

            avgTemplateLen /= <double>templateLenSamples

            if avgTemplateLen >= minInsertSize and avgTemplateLen <= maxInsertSize:
                return <int64_t> (avgTemplateLen + 0.5)
            else:
                return <int64_t> fallBack

        top2ContigsIdx = np.argsort(lengths)[-min(2, len(contigs)):]
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

        for contigIdx in top2ContigsIdx:
            contig = contigs[contigIdx]
            contigLen = <int64_t>lengths[contigIdx]
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
                if (readSeg.flag & samFlagExclude) != 0:
                    continue
                j = <int>((readSeg.reference_start) // rollingChunkSize)
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
                    readStart = <int64_t>readSeg.reference_start
                    readEnd = <int64_t>readSeg.reference_end
                    if (readFlag & samFlagExclude) != 0:
                        continue
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
                            score += fwdView[i]*revView[i + lag]
                        if score > bestScore:
                            bestScore = score
                            bestLag = lag

                chosenLag = bestLag

                if chosenLag > 0 and bestScore != 0.0:
                    bestLags.append(chosenLag)
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
                          double zeroThresh=0.0):

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

    _regionMeanVar(valuesView, startsView, sizesView, meansView, varsView, zeroPenalty, zeroThresh)

    return outMeans, outVars, starts_, ends


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cmonotonicFit(jointlySortedMeans, jointlySortedVariances, double refitWeight = 1.0, double floorTarget = 0.1):

    cdef cnp.ndarray[cnp.float32_t, ndim=1] xArr = np.ascontiguousarray(jointlySortedMeans, dtype=np.float32).ravel()
    cdef cnp.ndarray[cnp.float32_t, ndim=1] yArr = np.ascontiguousarray(jointlySortedVariances, dtype=np.float32).ravel()
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(2, dtype=np.float32)
    cdef Py_ssize_t n = xArr.shape[0]
    cdef Py_ssize_t i
    cdef double xMin
    cdef double x, z, yVal
    cdef double sumX = 0.0
    cdef double sumSqX = 0.0
    cdef double sumZ = 0.0
    cdef double sumXZ = 0.0
    cdef double sumSqZ = 0.0
    cdef double numSamples
    cdef double slopeUnconstrained, interceptUnconstrained
    cdef double slopeZero, interceptZero
    cdef double slopeBound, interceptBound
    cdef double optimalSlope, optimalIntercept
    cdef double optimalObjective, currentObjective
    cdef double sumSqShiftX = 0.0
    cdef double sumShiftXZ = 0.0
    cdef double xShift
    cdef double sumW, sumWX, sumWXX, sumWZ, sumWXZ
    cdef double penSlope, penIntercept
    cdef double initialVar, penTarget, penLoss
    cdef double finalErr
    cdef int it, maxIter
    cdef double tol = 1.0e-4
    xMin = <double>xArr[0] + 1.0e-4
    maxIter = 0

    for i in range(n):
        x = <double>xArr[i]
        yVal = <double>yArr[i]
        if yVal < 0.0:
            z = 0.0
        else:
            z = yVal

        sumX += x
        sumSqX += x*x
        sumZ += z
        sumXZ += x*z
        sumSqZ += z*z

    numSamples = <double>n
    _solveSystem22(sumSqX, sumX, sumX, numSamples,
        sumXZ,
        sumZ,
        &slopeUnconstrained,
        &interceptUnconstrained)

    optimalSlope = <double>0.0
    optimalIntercept = <double>0.0
    optimalObjective = _loss(
        optimalSlope, optimalIntercept,
        sumX, sumSqX, sumZ, sumXZ, sumSqZ, numSamples,
    )

    if slopeUnconstrained >= 0.0 and (slopeUnconstrained*xMin + interceptUnconstrained) >= 0.0:
        currentObjective = _loss(
            slopeUnconstrained, interceptUnconstrained,
            sumX, sumSqX, sumZ, sumXZ, sumSqZ, numSamples,
        )
        if currentObjective < optimalObjective:
            optimalObjective = currentObjective
            optimalSlope = slopeUnconstrained
            optimalIntercept = interceptUnconstrained

    slopeZero = 0.0
    interceptZero = sumZ / numSamples
    if interceptZero < 0.0:
        interceptZero = 0.0

    currentObjective = _loss(
        slopeZero, interceptZero,
        sumX, sumSqX, sumZ, sumXZ, sumSqZ, numSamples,
    )
    if currentObjective < optimalObjective:
        optimalObjective = currentObjective
        optimalSlope = slopeZero
        optimalIntercept = interceptZero

    for i in range(n):
        xShift = (<double>xArr[i]) - xMin
        yVal = <double>yArr[i]
        if yVal < 0.0:
            yVal = 0.0
            z = 0.0
        z = yVal
        sumSqShiftX += (xShift*xShift)
        sumShiftXZ += (xShift*z) # z*x - z*xMin

    if sumSqShiftX > 0.0:
        # regularized slope fitting the offset vals
        slopeBound = sumShiftXZ/(sumSqShiftX)
    else:
        slopeBound = 0.0
    if slopeBound < 0.0:
        slopeBound = 0.0

    # since 0 <= slope <= slopeBound = sumShiftXZ/(sumSqShiftX),
    # ... intercept >= -slopeBound*xMin --> nonnegative estimates
    interceptBound = -slopeBound*xMin
    currentObjective = _loss(
        slopeBound, interceptBound,
        sumX, sumSqX, sumZ, sumXZ, sumSqZ, numSamples,
    )
    if currentObjective < optimalObjective:
        optimalObjective = currentObjective
        optimalSlope = slopeBound
        optimalIntercept = interceptBound

    # FFR: revisit this
    # ... Our dynamic ranges are practically restricted to 3-10. The following
    # ... mimics (Abu-mostafa, 1992 NeurIPS) and refits data with a
    # ... --one-sided-- penalty (svm/hinge) via a conservative inequality
    # ... 'hint' --> floor value

    # set maxIters as number of points below the target variance
    # ... this doesn't confer any strong guarantee but is natural
    # ... and avoids a hyperparameter.
    maxIter = <int>0
    i = <Py_ssize_t>0
    for i in range(n):
        x = <double>xArr[i]
        # original estimate B0 + B1*x
        initialVar = optimalIntercept + optimalSlope*x
        penTarget = <double>floorTarget
        if initialVar < penTarget:
            maxIter += 1

    maxIter = <int>fmin(<double>100.0, <double>maxIter)
    if refitWeight > 0.0:
        for it in range(maxIter):
            sumW = numSamples
            sumWX = sumX
            sumWXX = sumSqX
            sumWZ = sumZ
            sumWXZ = sumXZ


            for i in range(n):
                x = <double>xArr[i]
                initialVar = optimalIntercept + optimalSlope*x
                penTarget = <double>floorTarget
                penLoss = penTarget - initialVar
                if penLoss > 0.0:
                    sumW += refitWeight
                    sumWX += refitWeight*x
                    sumWXX += refitWeight*x*x
                    sumWZ += refitWeight*penTarget
                    sumWXZ += refitWeight*x*penTarget

            # new WLS fit
            _solveSystem22(sumWXX, sumWX, sumWX, sumW,
                sumWXZ,
                sumWZ,
                &penSlope,
                &penIntercept)

            if penSlope < 0.0:
                penSlope = 0.0
                penIntercept = sumWZ / sumW
                if penIntercept < 0.0:
                    penIntercept = 0.0
            if (penSlope*xMin + penIntercept) < 0.0:
                penIntercept = -penSlope*xMin
            optimalSlope = penSlope
            optimalIntercept = penIntercept

    out[0] = <cnp.float32_t>optimalSlope
    out[1] = <cnp.float32_t>optimalIntercept
    return out


cpdef cmonotonicFitEval(
    cnp.ndarray coeffs,
    cnp.ndarray meanTrack,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] xArr = np.ascontiguousarray(meanTrack, dtype=np.float32).ravel()
    cdef Py_ssize_t n = xArr.shape[0]
    cdef Py_ssize_t i
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)

    cdef float[:] outView = out
    cdef float[:] xView = xArr

    cdef cnp.ndarray[cnp.float32_t, ndim=1] cArr = np.ascontiguousarray(coeffs, dtype=np.float32).ravel()
    cdef double slope = <double>cArr[0]
    cdef double intercept = <double>cArr[1]
    cdef double z

    if slope < 0.0:
        slope = 0.0
    if intercept < 0.0:
        intercept = 0.0

    for i in range(n):
        z = slope*(<double>xView[i]) + intercept
        if z < 0.0:
            z = 0.0
        outView[i] = <float>z

    return out


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cgetPosteriorMunc(
        float[::1] priorMeanVariances,
        float[::1] localMeanSquaredSOD,
        Py_ssize_t windowLength,
        double prior_Nu):

    cdef cnp.ndarray[cnp.float32_t, ndim=1] outputArray
    cdef float[::1] outputView

    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef double localDF = 0.0
    cdef double S_theta
    cdef double S_0
    cdef double posteriorDF # the 'posterior sample size'
    cdef double posteriorScale
    cdef double priorMean
    cdef double posteriorMean

    n = priorMeanVariances.shape[0]
    outputArray = np.zeros(n, dtype=np.float32)
    outputView = outputArray

    with nogil:
        for i in range(n):
            if i < 2:
                localDF = <double>0.0
            else:
                localDF = <double>(windowLength if i >= windowLength + 1 else i - 1)
            # Hoff's notation (chapter 7.3)
            # ... S_theta is based on local mean-squared second order differences
            # ... at proximal 'sparse' genomic regions, and S_0 is based the global/prior
            # ... mean-variance fit
            S_theta = (<double>localMeanSquaredSOD[i])*localDF
            priorMean = <double>priorMeanVariances[i]
            S_0 = (prior_Nu-2) * priorMean
            posteriorDF = prior_Nu + localDF
            posteriorScale = fmax(posteriorDF - 2.0, 1.0)
            posteriorMean = (S_0 + S_theta) / posteriorScale
            outputView[i] = <float>posteriorMean

    return outputArray


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
                          float eps_F32=<float>1.0e-4, double eps_F64=<double>1.0e-4,
                          float boundaryEps = <float>0.1,
                          double leftQ_ = <double>0.75,
                          double rightQ_ = <double>0.99):
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
    cdef float trackWideMean_F32
    cdef double trackWideMean_F64
    cdef Py_ssize_t n, tw__, tw_
    cdef double blockCenterCurr,blockCenterNext,lastCenter, edgeWeight
    cdef double carryOver, bgroundEstimate
    cdef double boundaryEps_F64 = <double>boundaryEps
    cdef float boundaryEps_F32 = <float>boundaryEps
    if blockLength <= 0:
        return None

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
            tw__ = <Py_ssize_t>0
            # (bounded) kneedle on the quantile curve
            # FFR: bound also by fraction of dynamic range
            trackWideMean_F32 = <float>cgetGlobalBaseline(valuesArr_F32, leftQ=leftQ_, rightQ=rightQ_)
            for tw__ in range(valuesLength):
                valuesArr_F32[tw__] = valuesArr_F32[tw__] - trackWideMean_F32

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
                    blockPtr_F32[blockIndex] = fmaxf(emaPtr_F32[centerIndex], eps_F32)

                k = 0
                blockCenterCurr = (<double>blockLength)*(<double>k + 0.5)
                blockCenterNext = (<double>blockLength)*(<double>(k + 1) + 0.5)
                lastCenter = (<double>blockLength)*(<double>(blockCount - 1) + 0.5)

                for i in range(valuesLength):
                    # (literal) edge cases
                    if (<double>i) <= blockCenterCurr:
                        interpolatedBackground_F32 = blockPtr_F32[0]
                    elif (<double>i) >= lastCenter:
                        interpolatedBackground_F32 = blockPtr_F32[blockCount - 1]
                    else:
                        while (<double>i) > blockCenterNext and k < blockCount - 2:
                            k += 1
                            blockCenterCurr = blockCenterNext
                            blockCenterNext = (<double>blockLength)*(<double>(k + 1) + 0.5)
                        # interpolate based on position of the interval relative to neighboring blocks' midpoints
                        # [---------block_k---------|---------block_k+1---------]
                        #                <----------|---------->
                        #                  (c < 1/2)|(c > 1/2)
                        # where c is `carryOver` to the subsequent block mean (see below)
                        carryOver = ((<double>i) - blockCenterCurr) / (blockCenterNext - blockCenterCurr)
                        bgroundEstimate = ((1.0 - carryOver)*(<double>blockPtr_F32[k])) + (carryOver*(<double>blockPtr_F32[k+1]))
                        interpolatedBackground_F32 = <float>(bgroundEstimate)

                    # finally, we take ~log-scale~ difference currentValue - background
                    logDiff_F32 = _carsinh_F32(valuesPtr_F32[i]) - _carsinh_F32(interpolatedBackground_F32)
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
        tw__ = <Py_ssize_t>0
        trackWideMean_F64 = <double>cgetGlobalBaseline(valuesArr_F64, leftQ=leftQ_, rightQ=rightQ_)
        for tw__ in range(valuesLength):
            valuesArr_F64[tw__] = valuesArr_F64[tw__] - trackWideMean_F64
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
                blockPtr_F64[blockIndex] = fmax(emaPtr_F64[centerIndex],  eps_F64)

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
                    interpolatedBackground_F64 = <double>(bgroundEstimate)
                logDiff_F64 = _carsinh_F64(valuesPtr_F64[i]) - _carsinh_F64(interpolatedBackground_F64)
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

    raise TypeError("A.dtype must be float32 or float64")



cpdef tuple cforwardPass(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixMunc,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQ,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQCopy,
    float phiScale,
    float dStatAlpha,
    float dStatd,
    float dStatPC,
    float maxQ,
    float minQ,
    float stateInit,
    float stateCovarInit,
    bint collectD=True,
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
    Py_ssize_t progressIter=10000
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


    if collectD and vectorD is None:
        dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
        vectorD = dStatVectorArr
    elif collectD:
        dStatVectorArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> vectorD

    if collectD:
        dStatVector = dStatVectorArr
    if doStore:
        stateForwardArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateForward
        stateCovarForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarForward
        pNoiseForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> pNoiseForward
        stateForwardView = stateForwardArr
        stateCovarForwardView = stateCovarForwardArr
        pNoiseForwardView = pNoiseForwardArr

    doFlush = (doStore and chunkSize > 0)
    # for tqdm
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
        # Fx
        priorState0 = stateTransition00*(<double>stateVectorView[0]) + stateTransition01*(<double>stateVectorView[1])
        priorState1 = stateTransition10*(<double>stateVectorView[0]) + stateTransition11*(<double>stateVectorView[1])
        stateVectorView[0] = <cnp.float32_t>priorState0
        stateVectorView[1] = <cnp.float32_t>priorState1
        # confusing, but here, 'posterior' is wrt the last iteration and is the current prior
        posteriorP00 = <double>stateCovarView[0,0]
        posteriorP01 = <double>stateCovarView[0,1]
        posteriorP10 = <double>stateCovarView[1,0]
        posteriorP11 = <double>stateCovarView[1,1]
        # intermediates, apparently faster this way per Simon
        tmp00 = stateTransition00*posteriorP00 + stateTransition01*posteriorP10
        tmp01 = stateTransition00*posteriorP01 + stateTransition01*posteriorP11
        tmp10 = stateTransition10*posteriorP00 + stateTransition11*posteriorP10
        tmp11 = stateTransition10*posteriorP01 + stateTransition11*posteriorP11
        # the actual prior state covariance at the current iteration, FPF^t + Q
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

        for trackIndex in range(trackCount):
            # H is ones vec, zij + v, v ~ (0, Rij) with Ri diagonal
            innovationValue = (<double>dataView[trackIndex, intervalIndex]) - (<double>stateVectorView[0])
            measurementVariance = (<double>muncView[trackIndex, intervalIndex])*(<double>phiScale)
            paddedVariance = measurementVariance + (<double>pad)
            invVariance = 1.0 / paddedVariance
            sumWeightYY += invVariance*(innovationValue*innovationValue)
            sumWeightUY += invVariance*innovationValue
            sumResidualUU += measurementVariance*(invVariance*invVariance)
            sumWeightUU += invVariance

        # sherman-morrison-based calculations [(H*P*H^T + R)^-1], R is diagonal PD, H is ones
        addP00Trace = 1.0 + (<double>stateCovarView[0,0])*sumWeightUU
        weightRank1 = (<double>stateCovarView[0,0]) / addP00Trace
        quadraticForm = sumWeightYY - weightRank1*(sumWeightUY*sumWeightUY)
        # D stat ~=~ NIS, but possibly defined as median, etc.
        dStatValue = quadraticForm / (<double>trackCount)
        if collectD:
            dStatVector[intervalIndex] = <cnp.float32_t>dStatValue

        # record number of process noise covariance updates
        adjustmentCount += <int>(dStatValue > (<double>dStatAlpha))
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
        # inlining hell -- FFR: its really just the eigenvals assuming default offDiag
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
        # FFR: confusing, again, but 'posterior' here is i|i-1
        posteriorP00 = <double>stateCovarView[0,0]
        posteriorP01 = <double>stateCovarView[0,1]
        posteriorP10 = <double>stateCovarView[1,0]
        posteriorP11 = <double>stateCovarView[1,1]
        # ... *this* is the updated posterior through sumResidualUU / (addP00Trace*addP00Trace)
        posteriorNew00 = (IKH00*IKH00*posteriorP00) + (gainH*(posteriorP00*posteriorP00))
        posteriorNew01 = (IKH00*(IKH10*posteriorP00 + posteriorP01)) + (gainH*(posteriorP00*posteriorP10))
        posteriorNew11 = ((IKH10*IKH10*posteriorP00) + 2.0*IKH10*posteriorP10 + posteriorP11) + (gainH*(posteriorP10*posteriorP10))

        if posteriorNew00 < clipSmall: posteriorNew00 = clipSmall
        elif posteriorNew00 > clipBig: posteriorNew00 = clipBig
        if posteriorNew01 < clipSmall: posteriorNew01 = clipSmall
        elif posteriorNew01 > clipBig: posteriorNew01 = clipBig
        if posteriorNew11 < clipSmall: posteriorNew11 = clipSmall
        elif posteriorNew11 > clipBig: posteriorNew11 = clipBig


        stateCovarView[0,0] = <cnp.float32_t>posteriorNew00 # next prior
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
            # memmaps -- flush every `chunkSize`
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

    if collectD:
        # currently unused (12162025) but potentially useful
        # ... for covariance matching, EM-style updates based on
        # ... NIS stats
        phiHat = float(np.mean(dStatVectorArr))

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


cpdef object cgetGlobalBaseline(object x, double leftQ=<double>0.75, double rightQ=<double>0.99):
    cdef cnp.ndarray vals_F32
    cdef cnp.ndarray vals_F64
    cdef cnp.ndarray posVals_
    cdef cnp.ndarray sortedPositive
    cdef float[::1] sortedViewF32
    cdef double[::1] sortedViewF64
    cdef Py_ssize_t valueCount
    cdef double minValue_
    cdef double maxValue_
    cdef double spanVals_
    cdef double spanQuantile_
    cdef double inverseQuantileSpan
    cdef double inverseValueSpan
    cdef double leftQuantile
    cdef double rightQuantile
    cdef double midQuantile
    cdef double leftGrad
    cdef double rightGrad
    cdef double midGrad
    cdef double elbowQuantile
    cdef double elbow_
    cdef Py_ssize_t iteration

    if isinstance(x, np.ndarray) and (<cnp.ndarray>x).dtype == np.float32:
        vals_F32 = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        posVals_ = vals_F32[vals_F32 > 0]
        if posVals_.size == 0:
            return <float>0.0

        sortedPositive = np.ascontiguousarray(np.sort(posVals_), dtype=np.float32)
        sortedViewF32 = sortedPositive
        valueCount = sortedViewF32.shape[0]
        minValue_ = _interpolateQuantile_F32(sortedViewF32, valueCount, leftQ)
        maxValue_ = _interpolateQuantile_F32(sortedViewF32, valueCount, rightQ)
        spanVals_ = maxValue_ - minValue_
        if spanVals_ <= 0.0: return <float>minValue_
        spanQuantile_ = rightQ - leftQ
        inverseQuantileSpan = 1.0 / spanQuantile_
        inverseValueSpan = 1.0 / spanVals_
        leftQuantile = leftQ
        rightQuantile = rightQ

        with nogil:
            leftGrad = _kneeGrad_F32(sortedViewF32, valueCount, leftQuantile, inverseQuantileSpan, inverseValueSpan)
            rightGrad = _kneeGrad_F32(sortedViewF32, valueCount, rightQuantile, inverseQuantileSpan, inverseValueSpan)

            if leftGrad <= 0.0:
                elbowQuantile = leftQuantile
            elif rightGrad >= 0.0:
                elbowQuantile = rightQuantile
            else:
            # bisect + kneedle: find a maximum in [leftQ, rightQ] in the quantile curve above y=x
                for iteration in range(128):
                    midQuantile = 0.5 * (leftQuantile + rightQuantile)
                    midGrad = _kneeGrad_F32(sortedViewF32, valueCount, midQuantile, inverseQuantileSpan, inverseValueSpan)
                    if midGrad > 1.0e-8:
                        leftQuantile = midQuantile
                    elif midGrad < -1.0e-8:
                        rightQuantile = midQuantile
                    else:
                        break
                elbowQuantile = 0.5 * (leftQuantile + rightQuantile)

            elbow_ = _interpolateQuantile_F32(sortedViewF32, valueCount, elbowQuantile)

        return <float>elbow_

    vals_F64 = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    posVals_ = vals_F64[vals_F64 > 0]
    sortedPositive = np.ascontiguousarray(np.sort(posVals_), dtype=np.float64)
    sortedViewF64 = sortedPositive
    valueCount = sortedViewF64.shape[0]

    minValue_ = _interpolateQuantile_F64(sortedViewF64, valueCount, leftQ)
    maxValue_ = _interpolateQuantile_F64(sortedViewF64, valueCount, rightQ)
    spanVals_ = maxValue_ - minValue_
    spanQuantile_ = rightQ - leftQ
    inverseQuantileSpan = 1.0 / spanQuantile_
    inverseValueSpan = 1.0 / spanVals_
    leftQuantile = leftQ
    rightQuantile = rightQ

    with nogil:
        leftGrad = _kneeGrad_F64(sortedViewF64, valueCount, leftQuantile, inverseQuantileSpan, inverseValueSpan)
        rightGrad = _kneeGrad_F64(sortedViewF64, valueCount, rightQuantile, inverseQuantileSpan, inverseValueSpan)
        if leftGrad <= 0.0:
            elbowQuantile = leftQuantile
        elif rightGrad >= 0.0:
            elbowQuantile = rightQuantile
        else:
            for iteration in range(128):
                midQuantile = 0.5 * (leftQuantile + rightQuantile)
                midGrad = _kneeGrad_F32(sortedViewF32, valueCount, midQuantile, inverseQuantileSpan, inverseValueSpan)
                if midGrad > 1.0e-8:
                    leftQuantile = midQuantile
                elif midGrad < -1.0e-8:
                    rightQuantile = midQuantile
                else:
                    break
            elbowQuantile = 0.5 * (leftQuantile + rightQuantile)

        elbow_ = _interpolateQuantile_F64(sortedViewF64, valueCount, elbowQuantile)

    return <double>elbow_


cpdef cnp.ndarray[cnp.float32_t, ndim=1] clocalAR1Var(
        object x,
        Py_ssize_t windowLength):

    cdef cnp.ndarray valuesArr64
    cdef double[::1] valuesView
    cdef Py_ssize_t n
    cdef cnp.ndarray sum_accumArray
    cdef cnp.ndarray sumSq_accumArray
    cdef cnp.ndarray prodWithLag_accumArray
    cdef double[::1] sumCum
    cdef double[::1] sumSqCum
    cdef double[::1] prodWithLag
    cdef cnp.ndarray[cnp.float32_t, ndim=1] varOut
    cdef float[::1] varOutView
    cdef Py_ssize_t i, k
    cdef Py_ssize_t leftIndex, rightIndex, halfWindow
    cdef Py_ssize_t valueCount, pairCount
    cdef double sumValues, meanValue
    cdef double sumPrev, sumCurr
    cdef double sumPrevSq, sumCurrSq
    cdef double sumLagProd
    cdef double sumSqX, sumXY
    cdef double beta, maxBeta
    cdef double tmp0, tmp1, oneMinusBeta
    cdef double RSS
    cdef double oneMinusBetaSq
    cdef double localVariance

    valuesArr64 = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    valuesView = valuesArr64
    n = valuesView.shape[0]
    varOut = np.zeros(n, dtype=np.float32)
    varOutView = varOut
    sum_accumArray = np.empty(n + 1, dtype=np.float64)
    sumSq_accumArray = np.empty(n + 1, dtype=np.float64)
    prodWithLag_accumArray = np.empty(n + 1, dtype=np.float64)
    # point memviews
    sumCum = sum_accumArray
    sumSqCum = sumSq_accumArray
    prodWithLag = prodWithLag_accumArray
    halfWindow = windowLength // 2
    if maxBeta < 0.0:
        maxBeta = -maxBeta
    if maxBeta > 0.99:
        maxBeta = 0.99

    with nogil:
        sumCum[0] = 0.0
        sumSqCum[0] = 0.0
        prodWithLag[0] = 0.0
        prodWithLag[1] = 0.0

        for k in range(n):
            sumCum[k + 1] = sumCum[k] + valuesView[k]
            sumSqCum[k + 1] = sumSqCum[k] + (valuesView[k]*valuesView[k])
        for k in range(1, n):
            prodWithLag[k + 1] = prodWithLag[k] + (valuesView[k - 1] * valuesView[k])
        for i in range(n):
            # no padding
            leftIndex = i - halfWindow
            if leftIndex < 0:
                leftIndex = 0
            rightIndex = i + halfWindow
            if rightIndex >= n:
                rightIndex = n - 1

            valueCount = rightIndex - leftIndex + 1
            if valueCount < 2:
                varOutView[i] = <float>0.0
                continue

            pairCount = valueCount - 1
            sumValues = sumCum[rightIndex + 1] - sumCum[leftIndex]
            meanValue = sumValues / (<double>valueCount)
            sumPrev = sumCum[rightIndex] - sumCum[leftIndex]
            sumCurr = sumCum[rightIndex + 1] - sumCum[leftIndex + 1]
            sumPrevSq = sumSqCum[rightIndex] - sumSqCum[leftIndex]
            sumCurrSq = sumSqCum[rightIndex + 1] - sumSqCum[leftIndex + 1]
            sumLagProd = prodWithLag[rightIndex + 1] - prodWithLag[leftIndex + 1]
            sumSqX = sumPrevSq - 2.0 * meanValue * sumPrev + (<double>pairCount) * (meanValue*meanValue)
            sumXY = sumLagProd - meanValue * sumPrev - meanValue * sumCurr + (<double>pairCount) * meanValue * meanValue

            if fabs(sumSqX) > 1.0e-4:
                beta = sumXY / sumSqX
            else:
                beta = 0.0

            if beta >  maxBeta:
                beta =  maxBeta
            elif beta < -maxBeta:
                beta = -maxBeta


            tmp0 = sumCurrSq - 2.0*(beta*sumLagProd) + (beta*beta)*sumPrevSq
            tmp1 = sumCurr - beta * sumPrev
            oneMinusBeta = 1.0 - beta
            RSS = tmp0 - (2.0*(oneMinusBeta * meanValue)*tmp1 + (<double>pairCount)*(oneMinusBeta*oneMinusBeta)*(meanValue*meanValue))

            if RSS < 0.0:
                RSS = 0.0

            oneMinusBetaSq = 1.0 - (beta*beta)
            if oneMinusBetaSq <= 1.0e-4:
                varOutView[i] = <float>0.0
            else:
                localVariance = (RSS / (<double>pairCount)) / oneMinusBetaSq
                if localVariance < 0.0:
                    localVariance = 0.0
                varOutView[i] = <float>localVariance

    return varOut
