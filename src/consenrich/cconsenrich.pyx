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
from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf, fprintf, stderr
cnp.import_array()

# ========
# constants
# ========
cdef const float __INV_LN2_FLOAT = <float>1.44269504
cdef const double __INV_LN2_DOUBLE = <double>1.44269504088896340

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
                                double pairsRegLambda=<double>1.0) noexcept nogil:
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

    cdef double lambdaEff
    cdef double Scale
    cdef double scaleFloor

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

        eps = 1.0e-6*(sumSqXSeq + 1.0)

        # scale-aware ridge
        if nPairsDouble > 0.0:
            lambdaEff = pairsRegLambda / (nPairsDouble + 1.0)
        else:
            lambdaEff = pairsRegLambda

        scaleFloor = 1.0e-4*(sumSqXSeq + 1.0)

        if sumSqXSeq > eps:
            Scale = (sumSqXSeq * (1.0 + lambdaEff)) + scaleFloor
            beta1 = sumXYc / Scale
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
    # CALLERS: cDenseMean

    return (<double>rand()) / (<double>RAND_MAX + 1.0)


cdef inline Py_ssize_t _rand_int(Py_ssize_t n) nogil:
    # CALLERS: cDenseMean

    return <Py_ssize_t>(rand() % n)


cdef inline Py_ssize_t _geometricDraw(double logq_) nogil:
    # CALLERS: cDenseMean

    cdef double u = _U01()
    if u <= 0.0:
        u = 1.0 / ((<double>RAND_MAX) + 1.0)
    return <Py_ssize_t>(floor(log(u) / logq_) + 1.0)


cdef inline double _median_F64(const double* src, Py_ssize_t n) noexcept nogil:
    # CALLERS: `_MADSigma_F64`, `clocalBaseline`
    # FFR: tidy this up, create F32 version and use consitently
    # FFR: explicit malloc/free is inconsistent with the rest of codebase,
    # ... but this may be fastest/safest for immutable inputs _REVISIT_

    cdef double* buf
    cdef Py_ssize_t k
    cdef double upper, lower

    if n <= 0:
        return <double>(0.0)

    buf = <double*>malloc(n * sizeof(double))
    if buf == NULL:
        return <double>(0.0)
    memcpy(buf, src, n * sizeof(double))

    k = n >> 1
    _nthElement_F64(buf, n, k)
    upper = buf[k]

    if (n & 1) == 0:
        _nthElement_F64(buf, n, k - 1)
        lower = buf[k - 1]
        upper = 0.5 * (upper + lower)

    free(buf)
    return upper


cdef inline double _MADSigma_F64(const double* src, Py_ssize_t n, double* medOut) noexcept nogil:
    # CALLERS: clocalBaseline
    cdef double med
    cdef double* dev
    cdef Py_ssize_t i
    cdef double mad

    if n <= 0:
        if medOut != NULL:
            medOut[0] = <double>(0.0)
        return <double>(0.0)

    med = _median_F64(src, n)
    if medOut != NULL:
        medOut[0] = med

    dev = <double*>malloc(n * sizeof(double))
    if dev == NULL:
        return <double>(0.0)

    for i in range(n):
        dev[i] = fabs(src[i] - med)

    mad = _median_F64(dev, n)
    free(dev)
    return 1.4826 * mad


cdef inline double _log_norm_pdf(double y, double mu, double var) nogil:
    # CALLERS: cDenseMean

    cdef double z = y - mu
    return -0.5 * log(6.28318 * var) - 0.5 * (z*z) / var


cdef inline float _medianCopy_F32(const float* src, Py_ssize_t n) noexcept nogil:
    cdef float* buf
    cdef float med

    if n <= 0:
        return <float>0.0

    buf = <float*>malloc(n * sizeof(float))
    if buf == NULL:
        return <float>0.0

    memcpy(buf, src, n * sizeof(float))
    med = _quantileInplaceF32(buf, n, <float>0.5)
    free(buf)
    return med


# ===========================
# --- MAT2: for readability/nogil inlining in the filter implementations ---
ctypedef struct MAT2:
    double a00
    double a01
    double a10
    double a11


cdef inline MAT2 MAT2_make(double a00, double a01, double a10, double a11) noexcept nogil:
    cdef MAT2 M
    M.a00 = a00
    M.a01 = a01
    M.a10 = a10
    M.a11 = a11
    return M


cdef inline MAT2 MAT2_add(MAT2 A, MAT2 B) noexcept nogil:
    return MAT2_make(A.a00 + B.a00, A.a01 + B.a01,
                     A.a10 + B.a10, A.a11 + B.a11)


cdef inline MAT2 MAT2_sub(MAT2 A, MAT2 B) noexcept nogil:
    return MAT2_make(A.a00 - B.a00, A.a01 - B.a01,
                     A.a10 - B.a10, A.a11 - B.a11)


cdef inline MAT2 MAT2_mul(MAT2 A, MAT2 B) noexcept nogil:
    return MAT2_make(
        A.a00*B.a00 + A.a01*B.a10,
        A.a00*B.a01 + A.a01*B.a11,
        A.a10*B.a00 + A.a11*B.a10,
        A.a10*B.a01 + A.a11*B.a11
    )


cdef inline MAT2 MAT2_transpose(MAT2 A) noexcept nogil:
    return MAT2_make(A.a00, A.a10, A.a01, A.a11)


cdef inline MAT2 MAT2_outer(double x0, double x1) noexcept nogil:
    return MAT2_make(x0*x0, x0*x1, x1*x0, x1*x1)


cdef inline MAT2 MAT2_clipDiagNonneg(MAT2 A) noexcept nogil:
    if A.a00 < 0.0:
        A.a00 = 0.0
    if A.a11 < 0.0:
        A.a11 = 0.0
    return A


cdef inline double MAT2_traceProd(MAT2 A, MAT2 B) noexcept nogil:
    return A.a00*B.a00 + A.a01*B.a10 + A.a10*B.a01 + A.a11*B.a11


# ===========================

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
    int64_t lagStep=5,
    int64_t earlyExit=250,
    int64_t randSeed=42,
):

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
    cdef bint isPairedEnd = <bint>0
    cdef double avgReadLength = <double>0.0
    cdef int64_t numReadLengthSamples = <int64_t>0
    cdef int64_t minInsertSize
    cdef int64_t requiredSamplesPE
    cdef int64_t tlen

    # rather than taking chromosome start end
    # look at contigs present and use the largest few
    cdef tuple contigs
    cdef tuple lengths
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lengthsArr
    cdef Py_ssize_t contigIdx
    cdef str contig
    cdef int64_t contigLen
    cdef int kTop

    # Single end tracks
    # fwd[i] is forward 5prime end count at offset i
    # rev[i] is reverse 5prime end count at offset i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fwd
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rev
    cdef double[::1] fwdView
    cdef double[::1] revView
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

    # FFT buffers
    cdef int nFFT
    cdef object Ff
    cdef object Fr
    cdef object corr

    # paired end sample buffer for median
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tlenArr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tlenWork
    cdef cnp.int32_t[::1] tlenWorkView
    cdef int tlenN
    cdef int midIdx
    cdef cnp.int32_t medPE

    # avoid out of bounds access in tight loops
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

    if maxInsertSize < 1:
        maxInsertSize = 1
    if iters < 1:
        return <int64_t>fallBack
    if blockSize < 64:
        blockSize = 64
    if rollingChunkSize < 1:
        rollingChunkSize = 1
    if lagStep < 1:
        lagStep = 1

    aln = AlignmentFile(bamFile, "rb", threads=samThreadsInternal)
    try:
        contigs = aln.references
        lengths = aln.lengths

        if contigs is None or len(contigs) == 0:
            return <int64_t>fallBack

        lengthsArr = np.asarray(lengths, dtype=np.int64)
        kTop = 3 if len(contigs) >= 3 else (2 if len(contigs) >= 2 else 1)
        topContigsIdx = np.argpartition(lengthsArr, -kTop)[-kTop:]
        topContigsIdx = topContigsIdx[np.argsort(lengthsArr[topContigsIdx])[::-1]]

        # detect paired end and estimate read length
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
        minInsertSize = <int64_t>(avgReadLength)
        if minInsertSize < 1:
            minInsertSize = 1
        if minInsertSize > maxInsertSize:
            minInsertSize = maxInsertSize

        # paired end: abs(TLEN) median over proper read1 pairs
        if isPairedEnd:
            requiredSamplesPE = max(iters, 2000)

            tlenArr = np.empty(requiredSamplesPE, dtype=np.int32)
            tlenN = 0

            for contigIdx in topContigsIdx:
                if tlenN >= requiredSamplesPE:
                    break
                contig = contigs[contigIdx]

                for readSeg in aln.fetch(contig):
                    if tlenN >= requiredSamplesPE:
                        break

                    readFlag = readSeg.flag
                    if (readFlag & samFlagExclude) != 0:
                        continue
                    if (readFlag & 2) == 0:
                        continue
                    if (readFlag & 64) == 0:
                        continue

                    tlen = <int64_t>readSeg.template_length
                    if tlen == 0:
                        continue
                    if tlen < 0:
                        tlen = -tlen

                    if tlen < minInsertSize or tlen > maxInsertSize:
                        continue

                    tlenArr[tlenN] = <cnp.int32_t>tlen
                    tlenN += 1

            if tlenN < max(500, requiredSamplesPE // 5):
                return <int64_t>fallBack

            midIdx = tlenN // 2
            tlenWork = tlenArr[:tlenN].copy()
            tlenWork = np.partition(tlenWork, midIdx)
            tlenWorkView = tlenWork
            medPE = <cnp.int32_t>tlenWorkView[midIdx]

            if medPE < <cnp.int32_t>minInsertSize:
                return <int64_t>minInsertSize
            if medPE > <cnp.int32_t>maxInsertSize:
                return <int64_t>fallBack
            return <int64_t>medPE

        # single end
        # - coarse bins select blocks with high local signal
        # - build fwd and rev 5prime end tracks in each block
        # - score lag by cross covariance via FFT correlation
        #
        # corr[lag] = sum_i fwd0[i] * rev0[i + lag]
        # fwd0 = fwd - mean(fwd)
        # rev0 = rev - mean(rev)
        #
        # fragmentLen ~= lag + 1
        # reverse 5prime end is reference_end - 1

        bestLags = []
        blockHalf = blockSize // 2

        fwd = np.zeros(blockSize, dtype=np.float64, order="C")
        rev = np.zeros(blockSize, dtype=np.float64, order="C")
        fwdView = fwd
        revView = rev

        nFFT = 1
        while nFFT < (2 * blockSize):
            nFFT <<= 1

        for contigIdx in topContigsIdx:
            contig = contigs[contigIdx]
            contigLen = <int64_t>lengthsArr[contigIdx]
            regionLen = contigLen

            if regionLen < blockSize or regionLen <= 0:
                continue

            # coarse bin replicate
            # rawArr[j] counts reads with reference_start in bin j
            # medArr is a local median envelope
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

            # rolling local median filter
            # kernel size tied to blockSize
            winSize = <int>(blockSize // rollingChunkSize)
            if winSize < 1:
                winSize = 1
            if (winSize & 1) == 0:
                winSize += 1
            medArr[:] = ndimage.median_filter(rawArr, size=winSize, mode="nearest")

            # pick top bins then thin by a seen mask
            takeK = iters if iters < numChunks else numChunks
            unsortedIdx = np.argpartition(medArr, -takeK)[-takeK:]
            unsortedVals = medArr[unsortedIdx]
            sortedIdx = unsortedIdx[np.argsort(unsortedVals)[::-1]]

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
                # map bin center to genomic coordinates
                blockStartBP = idxVal*rollingChunkSize + (rollingChunkSize // 2) - blockHalf
                if blockStartBP < 0:
                    blockStartBP = 0
                blockEndBP = blockStartBP + blockSize
                if blockEndBP > contigLen:
                    blockEndBP = contigLen
                    blockStartBP = blockEndBP - blockSize
                    if blockStartBP < 0:
                        continue

                # 5prime end tracks
                # forward uses reference_start
                # reverse uses reference_end - 1
                fwd.fill(0.0)
                rev.fill(0.0)

                for readSeg in aln.fetch(contig, blockStartBP, blockEndBP):
                    readFlag = readSeg.flag
                    if (readFlag & samFlagExclude) != 0:
                        continue

                    readStart = <int64_t>readSeg.reference_start
                    readEnd = <int64_t>readSeg.reference_end
                    if readStart < blockStartBP or readEnd > blockEndBP:
                        continue
                    if readEnd <= readStart:
                        continue

                    strand = readFlag & 16
                    if strand == 0:
                        i = <int>(readStart - blockStartBP)
                        if 0 <= i < blockSize:
                            fwdView[i] += 1.0
                    else:
                        i = <int>((readEnd - 1) - blockStartBP)
                        if 0 <= i < blockSize:
                            revView[i] += 1.0

                maxValidLag = maxInsertSize if (maxInsertSize < blockSize) else (blockSize - 1)
                localMinLag = <int>minInsertSize
                localMaxLag = <int>maxValidLag
                if localMaxLag < localMinLag:
                    continue
                localLagStep = <int>lagStep

                # low count blocks contribute mostly noise
                fwdSum = 0.0
                revSum = 0.0
                for i in range(blockSize):
                    fwdSum += fwdView[i]
                    revSum += revView[i]

                if fwdSum < 10.0 or revSum < 10.0:
                    continue

                fwdMean = fwdSum / (<double>blockSize)
                revMean = revSum / (<double>blockSize)

                for i in range(blockSize):
                    fwdView[i] = fwdView[i] - fwdMean
                    revView[i] = revView[i] - revMean

                # corr[lag] = sum_i fwd0[i] * rev0[i + lag]
                # zero padding to nFFT >= 2*blockSize makes this linear for lag in [0, blockSize-1]
                Ff = np.fft.rfft(fwd, nFFT)
                Fr = np.fft.rfft(rev, nFFT)
                corr = np.fft.irfft(np.conj(Ff) * Fr, nFFT)

                bestScore = -1e308
                bestLag = -1

                for lag in range(localMinLag, localMaxLag + 1, localLagStep):
                    blockLen = blockSize - lag
                    if blockLen <= 0:
                        continue

                    score = <double>corr[lag]
                    if score > bestScore:
                        bestScore = score
                        bestLag = lag

                if bestLag > 0 and bestScore != 0.0:
                    bestLags.append(bestLag)
                if len(bestLags) >= earlyExit:
                    break

            if len(bestLags) >= earlyExit:
                break

    finally:
        aln.close()

    if len(bestLags) < 3:
        return fallBack

    bestLagsArr = np.asarray(bestLags, dtype=np.uint32)

    med = <int64_t>(np.median(bestLagsArr) + 1.0 + 0.5)

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


cpdef tuple monoFunc(object x, double offset=<double>(1.0), double scale=<double>(1.0)):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = np.ascontiguousarray(x, dtype=np.float64)
    cdef Py_ssize_t n = arr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[::1] arrView = arr
    cdef double[::1] outView = out
    cdef double offset_ = offset
    cdef double scale_ = scale
    cdef Py_ssize_t i
    cdef double xval, u

    # scale * log(x + offset)
    if offset_ <= 0.0:
        offset_ = 1.0

    with nogil:
        for i in range(n):
            xval = arrView[i]
            u = xval + offset_
            if u <= 0.0:
                u = offset_ # keep defined if x has negatives
            outView[i] = scale_ * log(u)

    return (out, -1.0)


cpdef object cTransform(
    object x,
    Py_ssize_t blockLength,
    bint disableLocalBackground=<bint>False,
    double w_local=<double>1.0,
    double w_global=<double>2.0,
    bint verbose=<bint>False,
    uint64_t rseed=<uint64_t>0,
    bint useIRLS=<bint>True,
    double asymPos=<double>(2.0/5.0),
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0)
):
    cdef Py_ssize_t blockLenTarget, n, i
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
    cdef float meanVal_F32 = 0.0
    cdef double meanVal_F64 = 0.0
    cdef double asymPos_ = asymPos
    if disableLocalBackground:
        wLocal = 0.0
        wGlobal = 1.0
    else:
        wLocal = w_local
        wGlobal = w_global

    # 0,0 --> skip both local+global baseline subtraction
    if not disableLocalBackground and wLocal == 0.0 and wGlobal == 0.0:

        momRes = monoFunc(x)
        if (<cnp.ndarray>x).dtype == np.float32:
            return np.ascontiguousarray(momRes[0], dtype=np.float32).reshape(-1)
        else:
            return np.ascontiguousarray(momRes[0], dtype=np.float64).reshape(-1)

        raise RuntimeError("Unreachable code executed in cTransform")

    blockLenTarget = <Py_ssize_t>max(min(blockLength, 10000), 3)
    if blockLenTarget % 2 == 0:
        blockLenTarget += 1
        if blockLenTarget > 10000:
            blockLenTarget -= 2

    weightSum = wLocal + wGlobal
    if weightSum <= 0.0:
        wLocal = 0.0
        wGlobal = 1.0
        weightSum = 1.0
    invWeightSum = 1.0 / weightSum


    n = (<cnp.ndarray>x).size
    # F32
    if (<cnp.ndarray>x).dtype == np.float32:
        momRes = monoFunc(x, offset=logOffset, scale=logMult)
        zArr_F32 = np.ascontiguousarray(momRes[0], dtype=np.float32).reshape(-1)
        zView_F32 = zArr_F32
        n = zArr_F32.shape[0]
        outArr = np.empty(n, dtype=np.float32)
        outView_F32 = outArr
        if wGlobal > 0.0:
            globalBaselineF32 = <float>cDenseMean(
            zArr_F32,
            blockLenTarget=blockLenTarget,
            verbose=verbose,
            seed=rseed,
            )
        else:
            globalBaselineF32 = 0.0

        if wLocal > 0.0:
            localBaseArr_F32 = clocalBaseline(zArr_F32, <int>blockLenTarget, useIRLS=useIRLS, asymPos=asymPos_)
            localBaseView_F32 = localBaseArr_F32

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
    momRes = monoFunc(x, offset=logOffset, scale=logMult)
    zArr_F64 = np.ascontiguousarray(momRes[0], dtype=np.float64).reshape(-1)
    zView_F64 = zArr_F64
    n = zArr_F64.shape[0]
    outArr = np.empty(n, dtype=np.float64)
    outView_F64 = outArr
    if wGlobal > 0.0:
        globalBaselineF64 = <double>cDenseMean(
            zArr_F64,
            blockLenTarget=blockLenTarget,
            verbose=verbose,
            seed=rseed,
        )
    else:
        globalBaselineF64 = 0.0

    if wLocal > 0.0:
        localBaseArr_F64 = clocalBaseline(zArr_F64, <int>blockLenTarget, useIRLS=useIRLS, asymPos=asymPos_)
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
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixPluginMuncInit,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQ0,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] intervalToBlockMap,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] rScale,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] qScale,
    Py_ssize_t blockCount,
    float stateInit,
    float stateCovarInit,
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
    object lambdaExp=None,
):
    r"""Run the forward pass (filter) step in the forward-backward state estimation phase

    :seealso: :func:`consenrich.cconsenrich.cbackwardPass`, :func:`consenrich.cconsenrich.cblockScaleEM`, :func:`consenrich.core.runConsenrich`, :class:`consenrich.core.processParams`, :class:`consenrich.core.observationParams`

    """

    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] muncMatView = matrixPluginMuncInit
    cdef cnp.float32_t[:, ::1] fView = matrixF
    cdef cnp.float32_t[:, ::1] q0View = matrixQ0
    cdef cnp.int32_t[::1] blockMapView = intervalToBlockMap
    cdef cnp.float32_t[::1] rScaleView = rScale
    cdef cnp.float32_t[::1] qScaleView = qScale
    cdef Py_ssize_t trackCount = dataView.shape[0]
    cdef Py_ssize_t intervalCount = dataView.shape[1]
    cdef Py_ssize_t k, j
    cdef Py_ssize_t blockId

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] dStatVectorArr
    cdef cnp.float32_t[::1] dStatVector

    cdef bint doStore = (stateForward is not None)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForwardArr
    cdef cnp.float32_t[:, ::1] stateForwardView
    cdef cnp.float32_t[:, :, ::1] stateCovarForwardView
    cdef cnp.float32_t[:, :, ::1] pNoiseForwardView

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] lambdaExpArr
    cdef cnp.float32_t[:, ::1] lambdaExpView
    cdef bint useLambda = (lambdaExp is not None)

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] stateVector = np.array([stateInit, 0.0], dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateCovar = (np.eye(2, dtype=np.float32) * np.float32(stateCovarInit))
    cdef cnp.float32_t[::1] stateVectorView = stateVector
    cdef cnp.float32_t[:, ::1] stateCovarView = stateCovar
    cdef double clipSmall = pow(10.0, -covarClip)
    cdef double clipBig = pow(10.0, covarClip)
    cdef float phiHat = 1.0

    # inlining to reduce small matrix indexing cost
    cdef double F00, F01, F10, F11
    cdef double Q00, Q01, Q10, Q11
    cdef double xPred0, xPred1
    cdef double P00, P01, P10, P11
    cdef double PPred00, PPred01, PPred10, PPred11
    cdef double tmp00, tmp01, tmp10, tmp11

    cdef double innov
    cdef double baseVar, measVar, invMeasVar

    cdef double sumInvR, sumInvRInnov, sumInvRInnov2
    cdef double gainLike, quadForm
    cdef double innovScale
    cdef double delta0

    cdef double IKH00, IKH10
    cdef double gainG, gainH
    cdef double PNew00, PNew01, PNew11

    cdef double sumLogR = 0.0
    cdef double sumNLL = 0.0
    cdef double intervalNLL = 0.0
    cdef double sumDStat = 0.0
    cdef float rScaleB, qScaleB

    cdef double w
    cdef double wMin = 1.0e-6
    cdef double wMax = 1.0e6
    cdef double LOG2PI = log(6.2831853071795864769)

    if useLambda:
        lambdaExpArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> lambdaExp
        lambdaExpView = lambdaExpArr

    # ------------------------------------------------------------
    # Check edge cases here once before loops
    # ------------------------------------------------------------
    if intervalCount <= 0 or trackCount <= 0:
        if vectorD is None:
            dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
        else:
            dStatVectorArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> vectorD
        if returnNLL:
            return (np.float32(0.0), 0, dStatVectorArr, 0.0)
        return (np.float32(0.0), 0, dStatVectorArr)

    if blockCount <= 0:
        raise ValueError("blockCount must be positive")
    if rScale.shape[0] < blockCount or qScale.shape[0] < blockCount:
        raise ValueError("rScale or qScale length must be >= blockCount")
    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")

    if useLambda:
        if lambdaExpArr.shape[0] != trackCount or lambdaExpArr.shape[1] != intervalCount:
            raise ValueError("lambdaExp shape must match (trackCount, intervalCount)")

    # dStatVector[k]: diagnostic scalar statistic for interval k that can optionally store NLL[k]
    # If storeNLLInD and returnNLL then dStatVector[k] stores NLL[k]
    # Else dStatVector[k] stores quadForm / trackCount
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

    # F constants
    F00 = <double>fView[0, 0]
    F01 = <double>fView[0, 1]
    F10 = <double>fView[1, 0]
    F11 = <double>fView[1, 1]

    # ============================================================
    # Outer loop genomic intervals k
    # ============================================================
    for k in range(intervalCount):
        blockId = <Py_ssize_t>blockMapView[k]
        if blockId < 0 or blockId >= blockCount:
            raise ValueError("intervalToBlockMap has out-of-range block id")

        rScaleB = rScaleView[blockId]
        qScaleB = qScaleView[blockId]
        if rScaleB < <float>clipSmall:
            rScaleB = <float>clipSmall
        if qScaleB < <float>clipSmall:
            qScaleB = <float>clipSmall

        # ========================================================
        # Predict step transition
        # ========================================================
        xPred0 = F00*(<double>stateVectorView[0]) + F01*(<double>stateVectorView[1])
        xPred1 = F10*(<double>stateVectorView[0]) + F11*(<double>stateVectorView[1])
        stateVectorView[0] = <cnp.float32_t>xPred0
        stateVectorView[1] = <cnp.float32_t>xPred1

        # Q[k,* , *] = qScale[block(k)] * Q0[* , *]
        Q00 = (<double>qScaleB) * (<double>q0View[0, 0])
        Q01 = (<double>qScaleB) * (<double>q0View[0, 1])
        Q10 = (<double>qScaleB) * (<double>q0View[1, 0])
        Q11 = (<double>qScaleB) * (<double>q0View[1, 1])

        # P[k | k-1,* , *] = F P F^T + Q
        P00 = <double>stateCovarView[0, 0]
        P01 = <double>stateCovarView[0, 1]
        P10 = <double>stateCovarView[1, 0]
        P11 = <double>stateCovarView[1, 1]

        tmp00 = F00*P00 + F01*P10
        tmp01 = F00*P01 + F01*P11
        tmp10 = F10*P00 + F11*P10
        tmp11 = F10*P01 + F11*P11

        PPred00 = tmp00*F00 + tmp01*F01 + Q00
        PPred01 = tmp00*F10 + tmp01*F11 + Q01
        PPred10 = tmp10*F00 + tmp11*F01 + Q10
        PPred11 = tmp10*F10 + tmp11*F11 + Q11

        stateCovarView[0, 0] = <cnp.float32_t>PPred00
        stateCovarView[0, 1] = <cnp.float32_t>PPred01
        stateCovarView[1, 0] = <cnp.float32_t>PPred10
        stateCovarView[1, 1] = <cnp.float32_t>PPred11

        # ========================================================
        # t weight plugin precision via lambda[j,k]
        # ========================================================
        sumInvR = 0.0
        sumInvRInnov = 0.0
        sumInvRInnov2 = 0.0
        if returnNLL:
            sumLogR = 0.0
            intervalNLL = 0.0

        # Inner loop tracks j at interval k
        for j in range(trackCount):
            innov = (<double>dataView[j, k]) - (<double>stateVectorView[0])

            baseVar = (<double>muncMatView[j, k]) + (<double>pad)
            if baseVar < clipSmall:
                baseVar = clipSmall

            measVar = (<double>rScaleB) * baseVar
            if measVar < clipSmall:
                measVar = clipSmall

            if useLambda:
                w = <double>lambdaExpView[j, k]
                if w < wMin:
                    w = wMin
                elif w > wMax:
                    w = wMax
            else:
                w = 1.0

            invMeasVar = w / measVar

            if returnNLL:
                # Gaussian objective given lambda
                # log Var[j,k] = log(R[j,k]) - log(lambda[j,k])
                sumLogR += (log(measVar) - log(w))

            sumInvRInnov2 += invMeasVar * (innov * innov)
            sumInvRInnov += invMeasVar * innov
            sumInvR += invMeasVar

        # innovScale = 1 + P[k | k-1,0,0] * sumInvR
        innovScale = 1.0 + (<double>stateCovarView[0, 0]) * sumInvR
        if innovScale < clipSmall:
            innovScale = clipSmall

        gainLike = (<double>stateCovarView[0, 0]) / innovScale
        quadForm = sumInvRInnov2 - gainLike * (sumInvRInnov * sumInvRInnov)
        if quadForm < 0.0:
            quadForm = 0.0

        if returnNLL:
            # NLL[k] is the conditional Gaussian NLL _given_ lambdaExp
            # Note that an observed data t-NLL would actually integrate over lambda
            intervalNLL = 0.5 * (sumLogR + log(innovScale) + quadForm + (<double>trackCount) * LOG2PI)
            sumNLL += intervalNLL

        dStatVector[k] = <cnp.float32_t>(
            intervalNLL if (returnNLL and storeNLLInD) else (quadForm / (<double>trackCount))
        )
        sumDStat += (<double>dStatVector[k])

        # delta0 = sumInvRInnov / innovScale
        delta0 = sumInvRInnov / innovScale

        stateVectorView[0] = <cnp.float32_t>((<double>stateVectorView[0]) + (<double>stateCovarView[0, 0]) * delta0)
        stateVectorView[1] = <cnp.float32_t>((<double>stateVectorView[1]) + (<double>stateCovarView[1, 0]) * delta0)

        gainG = sumInvR / innovScale
        gainH = sumInvR / (innovScale * innovScale)

        IKH00 = 1.0 - ((<double>stateCovarView[0, 0]) * gainG)
        IKH10 = -((<double>stateCovarView[1, 0]) * gainG)

        P00 = <double>stateCovarView[0, 0]
        P01 = <double>stateCovarView[0, 1]
        P10 = <double>stateCovarView[1, 0]
        P11 = <double>stateCovarView[1, 1]

        PNew00 = (IKH00*IKH00*P00) + (gainH*(P00*P00))
        PNew01 = (IKH00*(IKH10*P00 + P01)) + (gainH*(P00*P10))
        PNew11 = ((IKH10*IKH10*P00) + 2.0*IKH10*P10 + P11) + (gainH*(P10*P10))

        if PNew00 < clipSmall:
            PNew00 = clipSmall
        elif PNew00 > clipBig:
            PNew00 = clipBig

        if PNew01 < clipSmall:
            PNew01 = clipSmall
        elif PNew01 > clipBig:
            PNew01 = clipBig

        if PNew11 < clipSmall:
            PNew11 = clipSmall
        elif PNew11 > clipBig:
            PNew11 = clipBig

        stateCovarView[0, 0] = <cnp.float32_t>PNew00
        stateCovarView[0, 1] = <cnp.float32_t>PNew01
        stateCovarView[1, 0] = <cnp.float32_t>PNew01
        stateCovarView[1, 1] = <cnp.float32_t>PNew11

        # store for smoothing and EM logistics
        if doStore:
            stateForwardView[k, 0] = stateVectorView[0]
            stateForwardView[k, 1] = stateVectorView[1]
            stateCovarForwardView[k, 0, 0] = stateCovarView[0, 0]
            stateCovarForwardView[k, 0, 1] = stateCovarView[0, 1]
            stateCovarForwardView[k, 1, 0] = stateCovarView[1, 0]
            stateCovarForwardView[k, 1, 1] = stateCovarView[1, 1]
            pNoiseForwardView[k, 0, 0] = <cnp.float32_t>Q00
            pNoiseForwardView[k, 0, 1] = <cnp.float32_t>Q01
            pNoiseForwardView[k, 1, 0] = <cnp.float32_t>Q10
            pNoiseForwardView[k, 1, 1] = <cnp.float32_t>Q11

    phiHat = <float>(sumDStat / (<double>intervalCount))

    if returnNLL:
        return (phiHat, 0, vectorD, sumNLL)

    return (phiHat, 0, vectorD)


cpdef tuple cbackwardPass(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForward,
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForward,
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForward,
    float covarClip=3.0,
    Py_ssize_t chunkSize=1000000,
    object stateSmoothed=None,
    object stateCovarSmoothed=None,
    object lagCovSmoothed=None,
    object postFitResiduals=None,
    object progressBar=None,
    Py_ssize_t progressIter=10000
):
    r"""Run the backward pass (RTS smoother) step in the forward-backward state estimation phase


    Inputs from the forward pass
    ----------------------------

    The smoother uses:

    .. math::

        \mathbf{x}_{[k|k]}, \qquad \mathbf{P}_{[k|k]}, \qquad \mathbf{Q}_{[k]},

    from the forward pass ``stateForward``, ``stateCovarForward``, and ``pNoiseForward``.


    RTS recursion
    -------------

    One-step prediction from the filtered state:


    .. math::

        \mathbf{x}_{[k+1|k]} = \mathbf{F}\,\mathbf{x}_{[k|k]}

        \\
        \mathbf{P}_{[k+1|k]} = \mathbf{F}\,\mathbf{P}_{[k|k]}\,\mathbf{F}^\top + \mathbf{Q}_{[k]}.


    Smoother gain:

    .. math::

        \mathbf{J}_{[k]} = \mathbf{P}_{[k|k]}\mathbf{F}^\top \mathbf{P}_{[k+1|k]}^{-1}.


    Smoothed mean and covariance:


    .. math::

        \widetilde{\mathbf{x}}_{[k]} =
        \mathbf{x}_{[k|k]} + \mathbf{J}_{[k]}\left(\widetilde{\mathbf{x}}_{[k+1]} - \mathbf{x}_{[k+1|k]}\right)

        \\
        \widetilde{\mathbf{P}}_{[k]} =
        \mathbf{P}_{[k|k]} + \mathbf{J}_{[k]}\left(\widetilde{\mathbf{P}}_{[k+1]} - \mathbf{P}_{[k+1|k]}\right)\mathbf{J}_{[k]}^\top.


    Lag-one covariance
    ------------------

    The smoother also returns an estimate of the lag covariance:

    .. math::

        \mathbf{C}_{[k,k+1]} \approx \mathrm{Cov}(\mathbf{x}_{[k]}, \mathbf{x}_{[k+1]} \mid \text{all data}),

    stored in ``lagCovSmoothed[k]``, which is used during the EM step (:func:`consenrich.cconsenrich.cblockScaleEM`).


    Post-fit residuals
    ------------------

    Residuals against the smoothed level component are:

    .. math::

        r_{j,k} = z_{j,k} - \widetilde{x}_{[k]},

    stored as ``postFitResiduals[k, j]``.

    :param matrixF: State transition matrix :math:`\mathbf{F}`.
    :type  matrixF: numpy.ndarray[float32], shape (2, 2)
    :param stateForward: Filtered state :math:`\mathbf{x}_{[k|k]}` from the forward pass.
    :type  stateForward: numpy.ndarray[float32], shape (n, 2)
    :param stateCovarForward: Filtered state covariances :math:`\mathbf{P}_{[k|k]}` from the forward pass.
    :type  stateCovarForward: numpy.ndarray[float32], shape (n, 2, 2)
    :param pNoiseForward: Process noise covariances :math:`\mathbf{Q}_{[k]}` used at each interval.
    :type  pNoiseForward: numpy.ndarray[float32], shape (n, 2, 2)
    :param stateSmoothed: Output buffer for smoothed means :math:`\widetilde{\mathbf{x}}_{[k]}`.
    :type  stateSmoothed: numpy.ndarray[float32], shape (n, 2) or None
    :param stateCovarSmoothed: Optional output buffer for smoothed covariances :math:`\widetilde{\mathbf{P}}_{[k]}`.
    :type  stateCovarSmoothed: numpy.ndarray[float32], shape (n, 2, 2) or None
    :param lagCovSmoothed: Optional output buffer for lag-one covariances :math:`\mathbf{C}_{[k,k+1]}`.
    :type  lagCovSmoothed: numpy.ndarray[float32], shape (max(n-1, 1), 2, 2) or None
    :param postFitResiduals: Optional output buffer for residuals :math:`r_{j,k}`.
    :type  postFitResiduals: numpy.ndarray[float32], shape (n, m) or None
    :returns: ``(stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals)``.
    :rtype: tuple

    Related References
    ---------------------

    * RTS smoother: `Rauch, Tung & Striebel (1965), DOI:10.2514/3.3166 <https://doi.org/10.2514/3.3166>`_

    :seealso: :func:`consenrich.cconsenrich.cforwardPass`, :func:`consenrich.cconsenrich.cbackwardPass`, :func:`consenrich.core.runConsenrich`

    """

    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] fView = matrixF
    cdef cnp.float32_t[:, ::1] stateForwardView = stateForward
    cdef cnp.float32_t[:, :, ::1] stateCovarForwardView = stateCovarForward
    cdef cnp.float32_t[:, :, ::1] pNoiseForwardView = pNoiseForward

    cdef Py_ssize_t trackCount = dataView.shape[0]
    cdef Py_ssize_t intervalCount = dataView.shape[1]
    cdef Py_ssize_t k, j

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateSmoothedArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarSmoothedArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] lagCovSmoothedArr
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] postFitResidualsArr

    cdef cnp.float32_t[:, ::1] stateSmoothedView
    cdef cnp.float32_t[:, :, ::1] stateCovarSmoothedView
    cdef cnp.float32_t[:, :, ::1] lagCovSmoothedView
    cdef cnp.float32_t[:, ::1] postFitResidualsView

    cdef double clipSmall = pow(10.0, -covarClip)
    cdef double clipBig = pow(10.0, covarClip)
    cdef double F00, F01, F10, F11
    cdef double xPred0, xPred1
    cdef double Q00, Q01, Q10, Q11
    cdef double PPred00, PPred01, PPred10, PPred11
    cdef double Pf00, Pf01, Pf10, Pf11
    cdef double detPred
    cdef double invPred00, invPred01, invPred10, invPred11
    cdef double cross00, cross01, cross10, cross11
    cdef double J00, J01, J10, J11
    cdef double dx0, dx1
    cdef double xs0, xs1
    cdef double dP00, dP01, dP10, dP11
    cdef double corr00, corr01, corr10, corr11
    cdef double Ps00, Ps01, Ps11
    cdef double C00, C01, C10, C11
    cdef double JD00, JD01, JD10, JD11

    cdef double innov

    if stateSmoothed is not None:
        stateSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateSmoothed
    else:
        stateSmoothedArr = np.empty((intervalCount, 2), dtype=np.float32)

    if stateCovarSmoothed is not None:
        stateCovarSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarSmoothed
    else:
        stateCovarSmoothedArr = np.empty((intervalCount, 2, 2), dtype=np.float32)

    if lagCovSmoothed is not None:
        lagCovSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> lagCovSmoothed
    else:
        lagCovSmoothedArr = np.empty((max(intervalCount - 1, 1), 2, 2), dtype=np.float32)

    if postFitResiduals is not None:
        postFitResidualsArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> postFitResiduals
    else:
        postFitResidualsArr = np.empty((intervalCount, trackCount), dtype=np.float32)

    stateSmoothedView = stateSmoothedArr
    stateCovarSmoothedView = stateCovarSmoothedArr
    lagCovSmoothedView = lagCovSmoothedArr
    postFitResidualsView = postFitResidualsArr

    F00 = <double>fView[0, 0]
    F01 = <double>fView[0, 1]
    F10 = <double>fView[1, 0]
    F11 = <double>fView[1, 1]

    if intervalCount <= 0:
        return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)

    with nogil:
        # ========================================================
        # initialize with the final forward pass estimates at k = intervalCount - 1
        # ========================================================
        stateSmoothedView[intervalCount - 1, 0] = stateForwardView[intervalCount - 1, 0]
        stateSmoothedView[intervalCount - 1, 1] = stateForwardView[intervalCount - 1, 1]

        stateCovarSmoothedView[intervalCount - 1, 0, 0] = stateCovarForwardView[intervalCount - 1, 0, 0]
        stateCovarSmoothedView[intervalCount - 1, 0, 1] = stateCovarForwardView[intervalCount - 1, 0, 1]
        stateCovarSmoothedView[intervalCount - 1, 1, 0] = stateCovarForwardView[intervalCount - 1, 1, 0]
        stateCovarSmoothedView[intervalCount - 1, 1, 1] = stateCovarForwardView[intervalCount - 1, 1, 1]

        for j in range(trackCount):
            postFitResidualsView[intervalCount - 1, j] = <cnp.float32_t>(
                (<double>dataView[j, intervalCount - 1]) - (<double>stateSmoothedView[intervalCount - 1, 0])
            )

        # ========================================================
        # backward recursion k = intervalCount - 2 down to 0
        # ========================================================
        for k in range(intervalCount - 2, -1, -1):
            Pf00 = <double>stateCovarForwardView[k, 0, 0]
            Pf01 = <double>stateCovarForwardView[k, 0, 1]
            Pf10 = <double>stateCovarForwardView[k, 1, 0]
            Pf11 = <double>stateCovarForwardView[k, 1, 1]

            xPred0 = F00*(<double>stateForwardView[k, 0]) + F01*(<double>stateForwardView[k, 1])
            xPred1 = F10*(<double>stateForwardView[k, 0]) + F11*(<double>stateForwardView[k, 1])

            Q00 = <double>pNoiseForwardView[k, 0, 0]
            Q01 = <double>pNoiseForwardView[k, 0, 1]
            Q10 = <double>pNoiseForwardView[k, 1, 0]
            Q11 = <double>pNoiseForwardView[k, 1, 1]

            cross00 = F00*Pf00 + F01*Pf10
            cross01 = F00*Pf01 + F01*Pf11
            cross10 = F10*Pf00 + F11*Pf10
            cross11 = F10*Pf01 + F11*Pf11

            PPred00 = cross00*F00 + cross01*F01 + Q00
            PPred01 = cross00*F10 + cross01*F11 + Q01
            PPred10 = cross10*F00 + cross11*F01 + Q10
            PPred11 = cross10*F10 + cross11*F11 + Q11

            # 2x2 inverse for PPred
            detPred = (PPred00*PPred11) - (PPred01*PPred10)
            if detPred == 0.0:
                detPred = clipSmall

            invPred00 = PPred11 / detPred
            invPred01 = -PPred01 / detPred
            invPred10 = -PPred10 / detPred
            invPred11 = PPred00 / detPred

            # J[k] = P[k|k] F^T inv(PPred[k+1|k])
            cross00 = Pf00*F00 + Pf01*F01
            cross01 = Pf00*F10 + Pf01*F11
            cross10 = Pf10*F00 + Pf11*F01
            cross11 = Pf10*F10 + Pf11*F11

            J00 = cross00*invPred00 + cross01*invPred10
            J01 = cross00*invPred01 + cross01*invPred11
            J10 = cross10*invPred00 + cross11*invPred10
            J11 = cross10*invPred01 + cross11*invPred11

            dx0 = (<double>stateSmoothedView[k + 1, 0]) - xPred0
            dx1 = (<double>stateSmoothedView[k + 1, 1]) - xPred1

            xs0 = (<double>stateForwardView[k, 0]) + (J00*dx0 + J01*dx1)
            xs1 = (<double>stateForwardView[k, 1]) + (J10*dx0 + J11*dx1)

            stateSmoothedView[k, 0] = <cnp.float32_t>xs0
            stateSmoothedView[k, 1] = <cnp.float32_t>xs1

            dP00 = (<double>stateCovarSmoothedView[k + 1, 0, 0]) - PPred00
            dP01 = (<double>stateCovarSmoothedView[k + 1, 0, 1]) - PPred01
            dP10 = (<double>stateCovarSmoothedView[k + 1, 1, 0]) - PPred10
            dP11 = (<double>stateCovarSmoothedView[k + 1, 1, 1]) - PPred11

            corr00 = dP00*J00 + dP01*J01
            corr01 = dP00*J10 + dP01*J11
            corr10 = dP10*J00 + dP11*J01
            corr11 = dP10*J10 + dP11*J11

            Ps00 = Pf00 + (J00*corr00 + J01*corr10)
            Ps01 = Pf01 + (J00*corr01 + J01*corr11)
            Ps11 = Pf11 + (J10*corr01 + J11*corr11)

            if Ps00 < clipSmall:
                Ps00 = clipSmall
            elif Ps00 > clipBig:
                Ps00 = clipBig

            if Ps01 < clipSmall:
                Ps01 = clipSmall
            elif Ps01 > clipBig:
                Ps01 = clipBig

            if Ps11 < clipSmall:
                Ps11 = clipSmall
            elif Ps11 > clipBig:
                Ps11 = clipBig

            stateCovarSmoothedView[k, 0, 0] = <cnp.float32_t>Ps00
            stateCovarSmoothedView[k, 0, 1] = <cnp.float32_t>Ps01
            stateCovarSmoothedView[k, 1, 0] = <cnp.float32_t>Ps01
            stateCovarSmoothedView[k, 1, 1] = <cnp.float32_t>Ps11

            # C[k] = P[k|k] F^T + J[k] (PS[k+1] - PPred[k+1|k])
            C00 = Pf00*F00 + Pf01*F01
            C01 = Pf00*F10 + Pf01*F11
            C10 = Pf10*F00 + Pf11*F01
            C11 = Pf10*F10 + Pf11*F11

            JD00 = J00*dP00 + J01*dP10
            JD01 = J00*dP01 + J01*dP11
            JD10 = J10*dP00 + J11*dP10
            JD11 = J10*dP01 + J11*dP11

            C00 += JD00
            C01 += JD01
            C10 += JD10
            C11 += JD11

            if k < lagCovSmoothedArr.shape[0]:
                lagCovSmoothedView[k, 0, 0] = <cnp.float32_t>C00
                lagCovSmoothedView[k, 0, 1] = <cnp.float32_t>C01
                lagCovSmoothedView[k, 1, 0] = <cnp.float32_t>C10
                lagCovSmoothedView[k, 1, 1] = <cnp.float32_t>C11

            for j in range(trackCount):
                innov = (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                postFitResidualsView[k, j] = <cnp.float32_t>innov

    return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)


cpdef tuple cblockScaleEM(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixPluginMuncInit,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQ0,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] intervalToBlockMap,
    Py_ssize_t blockCount,
    float stateInit,
    float stateCovarInit,
    Py_ssize_t EM_maxIters=50,
    float EM_rtol=1.0e-3,
    float covarClip=3.0,
    float pad=1.0e-2,
    float EM_multiplierLow=0.1,
    float EM_multiplierHigh=10.0,
    float EM_alphaEMA=0.2,
    bint EM_scaleToMedian=True,
    bint returnIntermediates=False,
    float EM_tNu=10.0,
    Py_ssize_t t_innerIters=3,
):
    r"""Calibrate blockwise measurement and process noise scales while inferring the epigenomic consensus signal


    Multisample epigenomic HTS data shows strong heteroskedasticity (replicate- and interval-specific noise) and region-dependent dynamics.
    This routine estimates *blockwise* multipliers of *given* noise scales (both observation noise and process noise)

    .. math::

        R_{[j,k]} = r_{\mathrm{scale}}[b(k)]\,(\mathrm{pluginMuncInit}_{j,k}+\mathrm{pad}),
        \qquad
        \mathbf{Q}_{[k]} = q_{\mathrm{scale}}[b(k)]\,\mathbf{Q}_0,

    while inferring the consensus signal trajectory. Note that each :math:`R_{[j,k]}` is, in fact, estimated
    from the data in :func:`consenrich.core.getMuncTrack`.

    We aim for robustness by treating residuals with a Gaussian **scale**-mixture using
    *auxiliary* per-observation precision weight :math:`\lambda_{j,k}` to moderate
    the influence of outliers. Concretely, *given* :math:`\lambda_{j,k}`, the residual
    is treated as Gaussian with variance scaled by :math:`1/\lambda_{j,k}`:

    .. math::

      e_{j,k} \mid \lambda_{j,k} \sim \mathcal{N}\!\left(0,\; R_{j,k}/\lambda_{j,k}\right),
      \qquad
      \lambda_{j,k} \sim \mathrm{Gamma}\!\left(\nu/2,\; \nu/2\right).

    Large-magnitude (studentized) residuals yield smaller
    :math:`\mathbb{E}[\lambda_{j,k}\mid e_{j,k}]`. Because the mixture variance is
    :math:`R_{j,k}/\lambda_{j,k}`, this smaller :math:`\lambda_{j,k}` inflates the variance for that
    observation and thus downweights its contribution.


    .. note::

      To be sure, this routine uses an empirical *plug-in* approximation, rather than full EM over a joint model for :math:`\lambda_{j,k}` *and* the consensus signal.

      So, the heavy-tailed model is used to derive per-observation reweighting, but the M-step
      proceeds with a weighted-Gaussian criterion using these plug-in weights.


      Notice also that this routine can deviate from standard EM: It alternates E-step-style updates of
      :math:`\lambda_{j,k}^{\mathrm{exp}}=\mathbb{E}[\lambda_{j,k}\mid e_{j,k}]` with M-step closed-form updates of
      :math:`(r_{\mathrm{scale}}, q_{\mathrm{scale}})` and RTS smoothing for :math:`\mathbf{x}`. But
      (optional) log-EMA/median-normalization heuristics can void traditional EM monotonicity guarantees.

    E-step
    -------------------------------------

    Using the current :math:`r_{\mathrm{scale}}` and :math:`q_{\mathrm{scale}}`, run a forward-backward pass:

    * Forward (filter) pass to obtain :math:`\mathbf{x}_{[k\mid k]}` and :math:`\mathbf{P}_{[k\mid k]}`.

    * Backward (RTS smoother) pass to obtain :math:`\widetilde{\mathbf{x}}_{[k]}`,
    :math:`\widetilde{\mathbf{P}}_{[k]}`, and lag covariances.


    Approximate the expected squared error of each observation as the squared residual
    against the smoothed level plus the smoothed level variance:

    .. math::

        \mathbb{E}[e_{j,k}^2] \approx (z_{j,k} - \widetilde{x}_{[k]})^2 + \widetilde{P}_{00,[k]}.


    Define the squared studentized residual and the expected precision weight:

    .. math::

        u_{j,k}^2 = \frac{\mathbb{E}[e_{j,k}^2]}{R_{j,k}},
        \qquad
        \lambda_{j,k}^{\mathrm{exp}} = \frac{\nu+1}{\nu+u_{j,k}^2}.

    M-step
    --------------------------------

    **Measurement noise scale** per block :math:`b` (with :math:`\mathcal{I}_b=\{(j,k): b(k)=b\}`):

    .. math::

        r_{\mathrm{scale}}[b] \leftarrow
        \frac{1}{|\mathcal{I}_b|}
        \sum_{(j,k)\in \mathcal{I}_b}
        \lambda_{j,k}^{\mathrm{exp}}
        \frac{\mathbb{E}[e_{j,k}^2]}{\mathrm{pluginMuncInit}_{j,k}+\mathrm{pad}}.

    **Process noise scale** per block :math:`b` using expected innovations

    .. math::

        \mathbf{w}_{[k]} = \widetilde{\mathbf{x}}_{[k+1]} - \mathbf{F}\widetilde{\mathbf{x}}_{[k]},

    and smoothed second moments to approximate :math:`\mathbb{E}[\mathbf{w}\mathbf{w}^\top]`, then

    .. math::

        q_{\mathrm{scale}}[b] \leftarrow
        \frac{1}{2}\,\mathrm{tr}\!\left(\mathbf{Q}_0^{-1}\,\mathbb{E}[\mathbf{w}\mathbf{w}^\top]\right),

    where :math:`\mathbf{Q}_0` is the given process noise (co)variance template.

    :param intervalToBlockMap: Mapping :math:`b(k)` from interval index :math:`k` to block id.
    :type  intervalToBlockMap: numpy.ndarray[int32], shape (n,)
    :param blockCount: Number of blocks :math:`B`.
    :type  blockCount: int
    :param stateInit: Initial level used in :math:`\mathbf{x}_{[0| -1]}=[\mathrm{stateInit},0]^\top`.
    :type  stateInit: float
    :param stateCovarInit: Initial covariance scale; initializes :math:`\mathbf{P}` to ``stateCovarInit * I``.
    :type  stateCovarInit: float
    :param EM_maxIters: Maximum number of outer EM iterations.
    :type  EM_maxIters: int
    :param EM_rtol: Relative tolerance for stopping based on NLL improvement.
    :type  EM_rtol: float
    :param EM_multiplierLow: Lower bound for ``rScale`` and ``qScale`` during EM.
    :type  EM_multiplierLow: float
    :param EM_multiplierHigh: Upper bound for ``rScale`` and ``qScale`` during EM.
    :type  EM_multiplierHigh: float
    :param EM_alphaEMA: EMA coefficient (in log space) applied to scale updates, in (0,1]
    :type  EM_alphaEMA: float
    :param EM_scaleToMedian: If True, divide ``rScale`` and ``qScale`` by their blockwise medians each iteration.
    :type  EM_scaleToMedian: bool
    :param returnIntermediates: If True, also return smoothed states, covariances, residuals, and ``lambdaExp``.
    :type  returnIntermediates: bool
    :param EM_tNu: Student-t df :math:`\nu` for robust reweighting
    :type  EM_tNu: float
    :param t_innerIters: Number of inner iterations updating ``lambdaExp`` before each M-step
    :type  t_innerIters: int
    :returns: ``(rScale, qScale, itersDone, finalNLL, stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals, lambdaExp)``.
    :rtype: tuple


    Related References
    ------------------------------------------------

    * Gaussian scale-mixtures (e.g., :math:`Y` s.t. :math:`Y|(Z=z) \sim \mathcal{N}(\mu, z^{-1}\Sigma)`): `West (1987), <doi.org/10.1093/biomet/74.3.646>`_

    * Student-t KF: `Aravkin et al. (2014), DOI:10.1137/130918861 <https://doi.org/10.1137/130918861>`_


    :seealso: :func:`consenrich.cconsenrich.cforwardPass`, :func:`consenrich.cconsenrich.cbackwardPass`, :func:`consenrich.core.runConsenrich`
    """

    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef Py_ssize_t i, k, j, inner
    cdef Py_ssize_t b
    cdef cnp.int32_t[::1] blockMapView = intervalToBlockMap
    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] muncMatView = matrixPluginMuncInit
    cdef cnp.float32_t[:, ::1] fView = matrixF
    cdef cnp.float32_t[:, ::1] q0View = matrixQ0
    cdef double clipSmall = pow(10.0, -covarClip)

    # per block scale factors
    # rScale[b] rescales measurement variance
    # qScale[b] rescales process variance
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] rScaleArr = np.ones(blockCount, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] qScaleArr = np.ones(blockCount, dtype=np.float32)
    cdef cnp.float32_t[::1] rScaleView = rScaleArr
    cdef cnp.float32_t[::1] qScaleView = qScaleArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rScaleLogTmp = np.zeros(blockCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qScaleLogTmp = np.zeros(blockCount, dtype=np.float64)
    cdef double[::1] rLogView = rScaleLogTmp
    cdef double[::1] qLogView = qScaleLogTmp
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rScaleLogSm = np.zeros(blockCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qScaleLogSm = np.zeros(blockCount, dtype=np.float64)
    cdef double[::1] rLogSmView = rScaleLogSm
    cdef double[::1] qLogSmView = qScaleLogSm
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rStatSum = np.zeros(blockCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] rWeightCount = np.zeros(blockCount, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qStatSum = np.zeros(blockCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] qStatCount = np.zeros(blockCount, dtype=np.int32)
    cdef double[::1] rStatSumView = rStatSum
    cdef cnp.int32_t[::1] rWeightCountView = rWeightCount
    cdef double[::1] qStatSumView = qStatSum
    cdef cnp.int32_t[::1] qStatCountView = qStatCount

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForward = np.empty((intervalCount, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForward = np.empty((intervalCount, 2, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForward = np.empty((intervalCount, 2, 2), dtype=np.float32)

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateSmoothed = np.empty((intervalCount, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarSmoothed = np.empty((intervalCount, 2, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] lagCovSmoothed = np.empty((max(intervalCount - 1, 1), 2, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] postFitResiduals = np.empty((intervalCount, trackCount), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] lambdaExp = np.ones((trackCount, intervalCount), dtype=np.float32)
    cdef cnp.float32_t[:, ::1] lambdaExpView = lambdaExp

    cdef cnp.float32_t[:, ::1] stateForwardView = stateForward
    cdef cnp.float32_t[:, :, ::1] stateCovarSmoothedView = stateCovarSmoothed
    cdef cnp.float32_t[:, ::1] residualView = postFitResiduals

    # inlined small matrices
    cdef double f00 = <double>fView[0, 0]
    cdef double f01 = <double>fView[0, 1]
    cdef double f10 = <double>fView[1, 0]
    cdef double f11 = <double>fView[1, 1]

    cdef double q0_00 = <double>q0View[0, 0]
    cdef double q0_01 = <double>q0View[0, 1]
    cdef double q0_10 = <double>q0View[1, 0]
    cdef double q0_11 = <double>q0View[1, 1]
    cdef double detQ0 = (q0_00*q0_11 - q0_01*q0_10) # precomputed (used throughout EM for qScale update)
    cdef double q0Inv00
    cdef double q0Inv01
    cdef double q0Inv10
    cdef double q0Inv11
    cdef MAT2 F
    cdef MAT2 Ft
    cdef MAT2 Q0inv
    cdef double previousNLL = 1.0e300
    cdef double currentNLL = 0.0
    cdef double relImprovement = 0.0
    cdef Py_ssize_t itersDone = 0
    cdef double res
    cdef double muncPlusPad
    cdef double p00k
    cdef double Rkj
    cdef double rMed = 1.0
    cdef double qMed = 1.0
    cdef double x0, x1, y0, y1
    cdef MAT2 Pk, Pk1, Ck_k1
    cdef MAT2 expec_xx, expec_yy, expec_xy, expec_yx, expec_ww
    cdef double trVal
    cdef double u2
    cdef double w
    cdef double wMin = 1.0e-6
    cdef double wMax = 1.0e6
    cdef double tmpVal
    cdef double sumU2
    cdef double sumLam
    cdef double sumLamU2
    cdef Py_ssize_t nObs
    cdef Py_ssize_t nTail4
    cdef Py_ssize_t nTail9
    cdef Py_ssize_t nTail16
    cdef Py_ssize_t nClipMin
    cdef Py_ssize_t nClipMax
    cdef double meanU2
    cdef double meanLam
    cdef double meanLamU2
    cdef double fracTail4
    cdef double fracTail9
    cdef double fracTail16
    cdef double refU2
    cdef double rHat
    cdef double qHat
    cdef double rRatio
    cdef double qRatio
    cdef double rRatioMin
    cdef double rRatioMax
    cdef double qRatioMin
    cdef double qRatioMax
    cdef double sumAbsLogRRatio
    cdef double sumAbsLogQRatio
    cdef Py_ssize_t nRRatio
    cdef Py_ssize_t nQRatio
    cdef Py_ssize_t nClipRLow
    cdef Py_ssize_t nClipRHigh
    cdef Py_ssize_t nClipQLow
    cdef Py_ssize_t nClipQHigh
    cdef double meanAbsLogRRatio
    cdef double meanAbsLogQRatio
    cdef double rMin
    cdef double rMax
    cdef double qMin
    cdef double qMax

    # check edge cases here once before loops
    if intervalCount <= 5:
        if returnIntermediates:
            return (
                rScaleArr, qScaleArr, 0, float(previousNLL),
                stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals, lambdaExp
            )
        return (rScaleArr, qScaleArr, 0, float(previousNLL))

    # validate dimensions and invertibility
    if blockCount <= 0:
        raise ValueError("blockCount must be positive")
    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")
    if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
        raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
    if detQ0 == 0.0:
        raise ValueError("matrixQ0 is singular")

    # Q0 inverse used in qScale update via trace Q0inv times E wwT
    q0Inv00 = q0_11 / detQ0
    q0Inv01 = -q0_01 / detQ0
    q0Inv10 = -q0_10 / detQ0
    q0Inv11 = q0_00 / detQ0

    F = MAT2_make(f00, f01, f10, f11)
    Ft = MAT2_transpose(F)
    Q0inv = MAT2_make(q0Inv00, q0Inv01, q0Inv10, q0Inv11)

    # reference mean u2 under integrated-t for nu > 2
    refU2 = (<double>EM_tNu) / ((<double>EM_tNu) - 2.0) if EM_tNu > 2.0 else 1.0e300

    for i in range(EM_maxIters):
        itersDone = i + 1
        fprintf(stderr, "\n\t[cblockScaleEM] iter=%zd\n", itersDone)

        # inner iterations update lambdaExp and then rerun the filter and smoother steps
        for inner in range(t_innerIters):
            cforwardPass(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixPluginMuncInit,
                matrixF=matrixF,
                matrixQ0=matrixQ0,
                intervalToBlockMap=intervalToBlockMap,
                rScale=rScaleArr,
                qScale=qScaleArr,
                blockCount=blockCount,
                stateInit=stateInit,
                stateCovarInit=stateCovarInit,
                covarClip=covarClip,
                pad=pad,
                projectStateDuringFiltering=False,
                stateLowerBound=0.0,
                stateUpperBound=0.0,
                chunkSize=0,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                vectorD=None,
                progressBar=None,
                progressIter=0,
                returnNLL=False,
                storeNLLInD=False,
                lambdaExp=lambdaExp,
            )

            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals = cbackwardPass(
                matrixData=matrixData,
                matrixF=matrixF,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                covarClip=covarClip,
                chunkSize=0,
                stateSmoothed=stateSmoothed,
                stateCovarSmoothed=stateCovarSmoothed,
                lagCovSmoothed=lagCovSmoothed,
                postFitResiduals=postFitResiduals,
                progressBar=None,
                progressIter=0,
            )

            # u2 is squared (studentized) residual
            sumU2 = 0.0
            sumLam = 0.0
            sumLamU2 = 0.0
            nObs = 0
            nTail4 = 0
            nTail9 = 0
            nTail16 = 0
            nClipMin = 0
            nClipMax = 0

            with nogil:

                for k in range(intervalCount):
                    b = <Py_ssize_t>blockMapView[k]
                    if b < 0 or b >= blockCount:
                        continue

                    p00k = <double>stateCovarSmoothedView[k, 0, 0]
                    if p00k < 0.0:
                        p00k = 0.0

                    for j in range(trackCount):
                        muncPlusPad = (<double>muncMatView[j, k]) + (<double>pad)
                        if muncPlusPad < clipSmall:
                            muncPlusPad = clipSmall

                        Rkj = (<double>rScaleView[b]) * muncPlusPad
                        if Rkj < clipSmall:
                            Rkj = clipSmall

                        res = (<double>dataView[j, k]) - (<double>stateSmoothed[k, 0])
                        tmpVal = (res*res + p00k)
                        u2 = tmpVal / Rkj

                        # w is the expected t-weight for the observation
                        w = ((<double>EM_tNu) + 1.0) / ((<double>EM_tNu) + u2)
                        if w < wMin:
                            w = wMin
                            if inner == (t_innerIters - 1):
                                nClipMin += 1
                        elif w > wMax:
                            w = wMax
                            if inner == (t_innerIters - 1):
                                nClipMax += 1

                        lambdaExpView[j, k] = <cnp.float32_t>w

                        if inner == (t_innerIters - 1):
                            # count up stats for tail probabilities
                            sumU2 += u2
                            sumLam += w
                            sumLamU2 += (w * u2)
                            nObs += 1
                            if u2 > 4.0:
                                nTail4 += 1
                            if u2 > 9.0:
                                nTail9 += 1
                            if u2 > 16.0:
                                nTail16 += 1

            if inner == (t_innerIters - 1):
                if nObs > 0:
                    meanU2 = sumU2 / (<double>nObs)
                    meanLam = sumLam / (<double>nObs)
                    meanLamU2 = sumLamU2 / (<double>nObs)
                    fracTail4 = (<double>nTail4) / (<double>nObs)
                    fracTail9 = (<double>nTail9) / (<double>nObs)
                    fracTail16 = (<double>nTail16) / (<double>nObs)
                else:
                    meanU2 = 0.0
                    meanLam = 0.0
                    meanLamU2 = 0.0
                    fracTail4 = 0.0
                    fracTail9 = 0.0
                    fracTail16 = 0.0


                fprintf(
                    stderr,
                    "\t[cblockScaleEM] studentized residual tails (nu=%.1f): "
                    "P[|u|>2] ~=~ %.6f,  P[|u|>3] ~=~ %.6f,  P[|u|>4] ~=~ %.6f\n",
                    EM_tNu,
                    fracTail4,
                    fracTail9,
                    fracTail16
                )

        # Gaussian NLL _given_ current lambdaExp
        currentNLL = (<double>cforwardPass(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixPluginMuncInit,
            matrixF=matrixF,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            rScale=rScaleArr,
            qScale=qScaleArr,
            blockCount=blockCount,
            stateInit=stateInit,
            stateCovarInit=stateCovarInit,
            covarClip=covarClip,
            pad=pad,
            projectStateDuringFiltering=False,
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            chunkSize=0,
            stateForward=None,
            stateCovarForward=None,
            pNoiseForward=None,
            vectorD=None,
            progressBar=None,
            progressIter=0,
            returnNLL=True,
            storeNLLInD=False,
            lambdaExp=lambdaExp,
        )[3])

        relImprovement = (previousNLL - currentNLL) / (fabs(previousNLL) + 1.0)
        previousNLL = currentNLL

        fprintf(stderr, "\t[cblockScaleEM] nll=%.6f  rel_improve=%.6f\n", currentNLL, relImprovement)

        if i > 0 and relImprovement >= 0.0 and relImprovement < <double>EM_rtol:
            fprintf(stderr, "\t[cblockScaleEM] CONVERGED  iter=%zd  rel_improve_below_rtol\n", itersDone)
            break

        # ---M step: update rScale[b] and qScale[b] per block b using smoothed moments and lambdaExp---
        with nogil:
            for b in range(blockCount):
                rStatSumView[b] = 0.0
                rWeightCountView[b] = 0
                qStatSumView[b] = 0.0
                qStatCountView[b] = 0

            # rScale stats
            for k in range(intervalCount):
                b = <Py_ssize_t>blockMapView[k]
                if b < 0 or b >= blockCount:
                    continue

                p00k = <double>stateCovarSmoothedView[k, 0, 0]
                if p00k < 0.0:
                    p00k = 0.0

                for j in range(trackCount):
                    res = <double>residualView[k, j]

                    muncPlusPad = (<double>muncMatView[j, k]) + (<double>pad)
                    if muncPlusPad < clipSmall:
                        muncPlusPad = clipSmall

                    tmpVal = (res*res + p00k)
                    w = <double>lambdaExpView[j, k]

                    rStatSumView[b] += w * (tmpVal / muncPlusPad)
                    rWeightCountView[b] += 1

            # qScale stats
            for k in range(intervalCount - 1):
                b = <Py_ssize_t>blockMapView[k]
                if b < 0 or b >= blockCount:
                    continue

                x0 = <double>stateSmoothed[k, 0]
                x1 = <double>stateSmoothed[k, 1]
                y0 = <double>stateSmoothed[k + 1, 0]
                y1 = <double>stateSmoothed[k + 1, 1]

                Pk = MAT2_make(
                    <double>stateCovarSmoothed[k, 0, 0],
                    <double>stateCovarSmoothed[k, 0, 1],
                    <double>stateCovarSmoothed[k, 1, 0],
                    <double>stateCovarSmoothed[k, 1, 1],
                )

                Pk1 = MAT2_make(
                    <double>stateCovarSmoothed[k + 1, 0, 0],
                    <double>stateCovarSmoothed[k + 1, 0, 1],
                    <double>stateCovarSmoothed[k + 1, 1, 0],
                    <double>stateCovarSmoothed[k + 1, 1, 1],
                )
                # Cross covariance between x[k] and x[k+1] from RTS smoother
                # NOTE: this is not necessarily equal to P[k] * F^T (due to lag-one RTS correction term)
                #       and so is preferred for a correct expected innovation covariance in the M-step
                Ck_k1 = MAT2_make(
                    <double>lagCovSmoothed[k, 0, 0],
                    <double>lagCovSmoothed[k, 0, 1],
                    <double>lagCovSmoothed[k, 1, 0],
                    <double>lagCovSmoothed[k, 1, 1],
                )

                expec_xx = MAT2_add(Pk, MAT2_outer(x0, x1))
                expec_yy = MAT2_add(Pk1, MAT2_outer(y0, y1))
                expec_xy = MAT2_add(Ck_k1, MAT2_make(x0*y0, x0*y1, x1*y0, x1*y1))
                expec_yx = MAT2_transpose(expec_xy)

                expec_ww = expec_yy
                expec_ww = MAT2_sub(expec_ww, MAT2_mul(expec_yx, Ft))
                expec_ww = MAT2_sub(expec_ww, MAT2_mul(F, expec_xy))
                expec_ww = MAT2_add(expec_ww, MAT2_mul(MAT2_mul(F, expec_xx), Ft))
                expec_ww = MAT2_clipDiagNonneg(expec_ww)

                trVal = MAT2_traceProd(Q0inv, expec_ww) / 2.0
                if trVal < 0.0:
                    trVal = 0.0

                qStatSumView[b] += trVal
                qStatCountView[b] += 1

            nClipRLow = 0
            nClipRHigh = 0
            nClipQLow = 0
            nClipQHigh = 0

            # update scales then clip to multiplier bounds
            for b in range(blockCount):
                if rWeightCountView[b] > 0:
                    rScaleView[b] = <cnp.float32_t>(rStatSumView[b] / (<double>rWeightCountView[b]))

                if qStatCountView[b] > 0:
                    qScaleView[b] = <cnp.float32_t>(qStatSumView[b] / (<double>qStatCountView[b]))

                if rScaleView[b] < <cnp.float32_t>EM_multiplierLow:
                    rScaleView[b] = <cnp.float32_t>EM_multiplierLow
                    nClipRLow += 1
                elif rScaleView[b] > <cnp.float32_t>EM_multiplierHigh:
                    rScaleView[b] = <cnp.float32_t>EM_multiplierHigh
                    nClipRHigh += 1

                if qScaleView[b] < <cnp.float32_t>EM_multiplierLow:
                    qScaleView[b] = <cnp.float32_t>EM_multiplierLow
                    nClipQLow += 1
                elif qScaleView[b] > <cnp.float32_t>EM_multiplierHigh:
                    qScaleView[b] = <cnp.float32_t>EM_multiplierHigh
                    nClipQHigh += 1

            # smooth scales across blocks in log-space (EMA)
            if EM_alphaEMA > 0.0 and EM_alphaEMA <= 1.0:
                for b in range(blockCount):
                    tmpVal = <double>rScaleView[b]
                    if tmpVal < clipSmall:
                        tmpVal = clipSmall
                    rLogView[b] = log(tmpVal)

                    tmpVal = <double>qScaleView[b]
                    if tmpVal < clipSmall:
                        tmpVal = clipSmall
                    qLogView[b] = log(tmpVal)

                if i == 0:
                    for b in range(blockCount):
                        rLogSmView[b] = rLogView[b]
                        qLogSmView[b] = qLogView[b]
                else:
                    for b in range(blockCount):
                        rLogSmView[b] = (1.0 - <double>EM_alphaEMA) * rLogSmView[b] + (<double>EM_alphaEMA) * rLogView[b]
                        qLogSmView[b] = (1.0 - <double>EM_alphaEMA) * qLogSmView[b] + (<double>EM_alphaEMA) * qLogView[b]

                for b in range(blockCount):
                    rScaleView[b] = <cnp.float32_t>exp(rLogSmView[b])
                    qScaleView[b] = <cnp.float32_t>exp(qLogSmView[b])

        # summarize current scales
        rMin = 1.0e300
        rMax = -1.0e300
        qMin = 1.0e300
        qMax = -1.0e300

        for b in range(blockCount):
            if (<double>rScaleView[b]) < rMin:
                rMin = <double>rScaleView[b]
            if (<double>rScaleView[b]) > rMax:
                rMax = <double>rScaleView[b]
            if (<double>qScaleView[b]) < qMin:
                qMin = <double>qScaleView[b]
            if (<double>qScaleView[b]) > qMax:
                qMax = <double>qScaleView[b]

        rRatioMin = 1.0e300
        rRatioMax = -1.0e300
        qRatioMin = 1.0e300
        qRatioMax = -1.0e300
        sumAbsLogRRatio = 0.0
        sumAbsLogQRatio = 0.0
        nRRatio = 0
        nQRatio = 0

        for b in range(blockCount):
            if rWeightCountView[b] > 0 and (<double>rScaleView[b]) > 0.0:
                rHat = rStatSumView[b] / (<double>rWeightCountView[b])
                rRatio = rHat / (<double>rScaleView[b])
                if rRatio < rRatioMin:
                    rRatioMin = rRatio
                if rRatio > rRatioMax:
                    rRatioMax = rRatio
                if rRatio > 0.0:
                    sumAbsLogRRatio += fabs(log(rRatio))
                nRRatio += 1

            if qStatCountView[b] > 0 and (<double>qScaleView[b]) > 0.0:
                qHat = qStatSumView[b] / (<double>qStatCountView[b])
                qRatio = qHat / (<double>qScaleView[b])
                if qRatio < qRatioMin:
                    qRatioMin = qRatio
                if qRatio > qRatioMax:
                    qRatioMax = qRatio
                if qRatio > 0.0:
                    sumAbsLogQRatio += fabs(log(qRatio))
                nQRatio += 1

        if nRRatio > 0:
            meanAbsLogRRatio = sumAbsLogRRatio / (<double>nRRatio)
        else:
            meanAbsLogRRatio = 0.0

        if nQRatio > 0:
            meanAbsLogQRatio = sumAbsLogQRatio / (<double>nQRatio)
        else:
            meanAbsLogQRatio = 0.0

        # divide by block median so typical scale is near one
        # voids EM guarantees but can improve interpretation
        # of block-to-block scale comparisons
        if EM_scaleToMedian:
            rMed = <double>_medianCopy_F32(<float*>&rScaleView[0], blockCount)
            qMed = <double>_medianCopy_F32(<float*>&qScaleView[0], blockCount)

            if rMed > 0.0 or qMed > 0.0:
                with nogil:
                    if rMed > 0.0:
                        for b in range(blockCount):
                            rScaleView[b] = <cnp.float32_t>((<double>rScaleView[b]) / rMed)

                    if qMed > 0.0:
                        for b in range(blockCount):
                            qScaleView[b] = <cnp.float32_t>((<double>qScaleView[b]) / qMed)

    if returnIntermediates:
        return (
            rScaleArr, qScaleArr, itersDone, float(previousNLL),
            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals, lambdaExp
        )

    return (rScaleArr, qScaleArr, itersDone, float(previousNLL))


cdef double cOtsu(
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] raw,
    double loQuantile=0.001,
    double hiQuantile=0.999,
) except? -1:
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] finiteValues
    cdef cnp.float64_t[::1] finiteView
    cdef Py_ssize_t n
    cdef Py_ssize_t i, binIdx, bestBinIdx
    cdef Py_ssize_t numBins
    cdef double lower_, higher_, binWidth, invBinWidth
    cdef double x

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] histCounts
    cdef cnp.float64_t[::1] histView
    cdef double totalCount, sumBinIndexAll
    cdef double weightBelow, weightAbove
    cdef double sumBinIndexBelow
    cdef double meanBelow, meanAbove
    cdef double betweenClassVar, bestBetweenClassVar
    cdef double avgCount, initPerBin, sumIdxInit, q25, q75, iqr, fdBinWidth, m_d, range_
    cdef Py_ssize_t mInRange, fdBins


    finiteValues = np.ascontiguousarray(raw[np.isfinite(raw)], dtype=np.float64).ravel()
    finiteView = finiteValues
    n = finiteValues.size
    if n <= 0:
        return NAN

    lower_ = <double>np.quantile(finiteValues, loQuantile)
    higher_ = <double>np.quantile(finiteValues, hiQuantile)
    if (not isfinite(lower_)) or (not isfinite(higher_)) or higher_ <= lower_:
        return NAN

    # Freedman-Diaconis bin counts
    q25 = <double>np.quantile(finiteValues, 0.25)
    q75 = <double>np.quantile(finiteValues, 0.75)
    iqr = q75 - q25
    range_ = higher_ - lower_

    if (not isfinite(iqr)) or iqr <= 0.0 or (not isfinite(range_)) or range_ <= 0.0:
        numBins = 512
    else:
        mInRange = 0
        with nogil:
            for i in range(n):
                x = finiteView[i]
                if x < lower_ or x > higher_:
                    continue
                mInRange += 1

        if mInRange <= 1:
            numBins = 64
        else:
            m_d = <double>mInRange
            fdBinWidth = (2.0 * iqr) / pow(m_d, 1.0 / 3.0)
            if (not isfinite(fdBinWidth)) or fdBinWidth <= 0.0:
                numBins = 256
            else:
                fdBins = <Py_ssize_t>ceil(range_ / fdBinWidth)
                if fdBins < 64:
                    numBins = 64
                elif fdBins > 256:
                    numBins = 256
                else:
                    numBins = fdBins

    binWidth = range_ / (<double>numBins)
    if (not isfinite(binWidth)) or binWidth <= 0.0:
        return NAN

    invBinWidth = 1.0 / binWidth
    histCounts = np.zeros(numBins, dtype=np.float64)
    histView = histCounts

    with nogil:
        for i in range(n):
            x = finiteView[i]
            if x < lower_ or x > higher_:
                continue
            binIdx = <Py_ssize_t>((x - lower_) * invBinWidth)
            if binIdx < 0:
                binIdx = 0
            elif binIdx >= numBins:
                binIdx = numBins - 1
            histView[binIdx] += 1.0

    totalCount = 0.0
    sumBinIndexAll = 0.0
    with nogil:
        for binIdx in range(numBins):
            totalCount += histView[binIdx]
            sumBinIndexAll += histView[binIdx] * (<double>binIdx)

    if (not isfinite(totalCount)) or totalCount <= 0.0:
        return NAN

    avgCount = totalCount / (<double>numBins)
    initPerBin = 1.0e-4 * avgCount # smooth: small uniform const added across bins
    if initPerBin > 0.0:
        # add
        totalCount += initPerBin * (<double>numBins)
        sumIdxInit = (<double>numBins) * (<double>(numBins - 1)) * 0.5
        sumBinIndexAll += initPerBin * sumIdxInit
        with nogil:
            for binIdx in range(numBins):
                histView[binIdx] += initPerBin

    weightBelow = 0.0
    sumBinIndexBelow = 0.0
    bestBetweenClassVar = -1.0
    bestBinIdx = numBins // 2

    with nogil:
        for binIdx in range(numBins):
            weightBelow += histView[binIdx]
            sumBinIndexBelow += histView[binIdx] * (<double>binIdx)
            weightAbove = totalCount - weightBelow
            if weightBelow <= 0.0 or weightAbove <= 0.0:
                continue

            meanBelow = sumBinIndexBelow / weightBelow
            meanAbove = (sumBinIndexAll - sumBinIndexBelow) / weightAbove
            betweenClassVar = meanBelow - meanAbove
            betweenClassVar = weightBelow * weightAbove * betweenClassVar * betweenClassVar

            if betweenClassVar > bestBetweenClassVar:
                bestBetweenClassVar = betweenClassVar
                bestBinIdx = binIdx

    return lower_ + ((<double>bestBinIdx) + 0.5) * binWidth


cpdef double cDenseMean(
    object x,
    Py_ssize_t blockLenTarget=250,
    Py_ssize_t itersEM=1000,
    uint64_t seed=0,
    bint verbose = <bint>False,
):
    r"""Estimate a 'dense' baseline to subtract from each replicate's transformed count track

    If calling this function outside of the default implementation, note that the input is
    assumed to have been normalized by sequencing depth / library size and transformed to
    log-scale.

    Data is modeled with a simple Gaussian mixture:

    .. math::

      p(y) = \pi \cdot \mathcal{N}(y;\,\mu_1,\sigma_1^2) + (1-\pi) \cdot \mathcal{N}(y;\,\mu_2,\sigma_2^2).

    We maximize the likelihood with respect to :math:`\mu_2 \geq \mu_1` using an expectation-maximization routine.

    **E-step**: For each observation :math:`y_i`, define

    .. math::

      \ell_{i1} = \log \pi + \log \mathcal{N}(y_i;\mu_1,\sigma_1^2),
      \qquad
      \ell_{i2} = \log (1-\pi) + \log \mathcal{N}(y_i;\mu_2,\sigma_2^2).

    .. math::

      \ell_i = \log\!\left(\exp(\ell_{i1}) + \exp(\ell_{i2})\right),

    and the component-1 weight

    .. math::

      w_i = \exp(\ell_{i1} - \ell_i),
          \qquad
          1-w_i = \exp(\ell_{i2} - \ell_i).


    **M-step**

    Let

    .. math::

      W = \sum_{i=1}^n w_i, \qquad W' = \sum_{i=1}^n (1-w_i) = n - W.

    *Update*

    .. math::

      \pi \leftarrow \frac{W}{n},

    .. math::

      \mu_1 \leftarrow \frac{\sum_{i=1}^n w_i y_i}{W},
      \qquad
      \mu_2 \leftarrow \frac{\sum_{i=1}^n (1-w_i) y_i}{W'},

    (the order of the two components is swapped if :math:`\mu_1 > \mu_2` ) and

    .. math::

      \sigma_1^2 \leftarrow \frac{\sum_{i=1}^n w_i y_i^2}{W} - \mu_1^2,
      \qquad
      \sigma_2^2 \leftarrow \frac{\sum_{i=1}^n (1-w_i) y_i^2}{W'} - \mu_2^2.

    We enforce a variance floor and a minimum / maximum value for :math:`\pi` for stability.

    The returned value is the estimated 'dense' baseline :math:`\mu_2 \geq \mu_1`. This value is used
    in a weighted average with :func:`consenrich.cconsenrich.clocalBaseline` for centering replicates' transformed
    count tracks.

    """

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] y
    cdef cnp.float64_t[::1] yView
    cdef Py_ssize_t n, i, it, maxIters
    cdef double q
    cdef double pi, mu1, mu2, var1, var2
    cdef double r, sum_r, sum_1mr
    cdef double sum_r_y, sum_1mr_y
    cdef double sum_r_y2, sum_1mr_y2
    cdef double logp1, logp2, mlog, tll_
    cdef double loglik, prev_loglik, diff
    cdef double dense_mu, dense_sd
    cdef double tol = 1e-8
    cdef double varFloor = 1e-6
    cdef double piFloor = 1e-6

    y = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    yView = y
    n = y.size

    maxIters = itersEM if itersEM > 0 else 50
    if maxIters < 10:
        maxIters = 10
    elif maxIters > 500:
        maxIters = 500

    cdef double q10 = <double>np.quantile(y, 0.10)
    cdef double q90 = <double>np.quantile(y, 0.90)
    cdef double ymean = <double>np.mean(y)
    cdef double ysd = <double>np.std(y)
    if ysd < 1e-6:
        ysd = 1.0

    mu1 = q10
    mu2 = q90
    if mu1 == mu2:
        mu1 = ymean - 0.5 * ysd
        mu2 = ymean + 0.5 * ysd

    var1 = ysd * ysd
    var2 = ysd * ysd
    if var1 < varFloor:
        var1 = varFloor
    if var2 < varFloor:
        var2 = varFloor

    pi = 0.9
    prev_loglik = -INFINITY

    for it in range(maxIters):
        sum_r = 0.0
        sum_1mr = 0.0
        sum_r_y = 0.0
        sum_1mr_y = 0.0
        sum_r_y2 = 0.0
        sum_1mr_y2 = 0.0
        loglik = 0.0

        # E-step
        with nogil:
            # fix mu1,mu2,var1,var2,pi, collect stats
            for i in range(n):
                logp1 = log(pi) + _log_norm_pdf(yView[i], mu1, var1)
                logp2 = log(1.0 - pi) + _log_norm_pdf(yView[i], mu2, var2)

                mlog = logp1 if logp1 > logp2 else logp2
                tll_ = mlog + log(exp(logp1 - mlog) + exp(logp2 - mlog))
                r = exp(logp1 - tll_)

                sum_r += r
                sum_1mr += (1.0 - r)
                sum_r_y += r * yView[i]
                sum_1mr_y += (1.0 - r) * yView[i]
                sum_r_y2 += r * yView[i] * yView[i]
                sum_1mr_y2 += (1.0 - r) * yView[i] * yView[i]
                loglik += tll_

        if sum_r <= 0.0 or sum_1mr <= 0.0:
            break

        # M-step: sum_r,sum_1mr,... --> update pi, mu1, mu2, var1, var2
        pi = sum_r / (<double>n)
        if pi < piFloor:
            pi = piFloor
        elif pi > 1.0 - piFloor:
            pi = 1.0 - piFloor

        mu1 = sum_r_y / sum_r
        mu2 = sum_1mr_y / sum_1mr

        # E[w*y^2]/E[w] - mu^2
        var1 = (sum_r_y2 / sum_r) - (mu1 * mu1)
        var2 = (sum_1mr_y2 / sum_1mr) - (mu2 * mu2)

        if (not isfinite(var1)) or var1 < varFloor:
            var1 = varFloor
        if (not isfinite(var2)) or var2 < varFloor:
            var2 = varFloor

        # enforce mu1 <= mu2
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            var1, var2 = var2, var1
            pi = 1.0 - pi

        diff = loglik - prev_loglik
        if diff < 0.0:
            diff = -diff

        if it > 0 and diff < tol * (1.0 + fabs(prev_loglik)):
            break

        if it >= maxIters-1 and diff >= tol * (1.0 + fabs(prev_loglik)):
            if verbose:
                printf(b"\tcconsenrich.cDenseMean: EM did not converge after %ld EM iters, diff=%.6f\n",
                       <long>it + 1, diff)

        prev_loglik = loglik

    dense_mu = mu2
    dense_sd = sqrt(var2)

    if verbose:
        printf(b"\tcconsenrich.cDenseMean(GMM): baseline=%.4f\n",
               dense_mu)

    return dense_mu


cpdef cnp.ndarray[cnp.float32_t, ndim=1] crolling_AR1_IVar(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    int blockLength,
    cnp.ndarray[cnp.uint8_t, ndim=1] excludeMask,
    double maxBeta=0.99,
    double pairsRegLambda = 1.0,
):
    cdef Py_ssize_t numIntervals=values.shape[0]
    cdef Py_ssize_t regionIndex, elementIndex, startIndex,  maxStartIndex
    cdef int halfBlockLength, maskSum
    cdef cnp.ndarray[cnp.float32_t, ndim=1] varAtStartIndex
    cdef cnp.ndarray[cnp.float32_t, ndim=1] varOut
    cdef float[::1] valuesView=values
    cdef cnp.uint8_t[::1] maskView=excludeMask
    cdef float[::1] varAtView
    cdef float[::1] varOutView
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
    cdef double lambdaEff
    cdef double Scale
    cdef double scaleFloor

    varOut = np.empty(numIntervals,dtype=np.float32)

    if blockLength > numIntervals:
        blockLength = <int>numIntervals

    if blockLength < 4:
        varOut[:] = 0.0
        return varOut

    halfBlockLength = (blockLength//2)
    maxStartIndex = (numIntervals - blockLength)
    varAtStartIndex = np.empty((maxStartIndex + 1),dtype=np.float32)
    varAtView = varAtStartIndex
    varOutView = varOut

    sumY=0.0
    sumSqY=0.0
    sumLagProd=0.0
    maskSum=0

    with nogil:
        # initialize first
        for elementIndex in range(blockLength):
            currentValue=valuesView[elementIndex]
            sumY += currentValue
            sumSqY += (currentValue*currentValue)
            maskSum += <int>maskView[elementIndex]
            if elementIndex < (blockLength - 1):
                sumLagProd += (currentValue*valuesView[(elementIndex + 1)])

        # sliding window until last block's start
        for startIndex in range(maxStartIndex + 1):
            if maskSum != 0:
                varAtView[startIndex]=<cnp.float32_t>-1.0
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
                eps = 1.0e-6*(sumSqXSeq + 1.0)
                if sumSqXSeq > eps:
                    # reg. AR(1) coefficient estimate
                    # scale-aware ridge: keep regularizer in same units as sumSqXSeq
                    lambdaEff = pairsRegLambda / (nPairsDouble + 1.0)
                    scaleFloor = 1.0e-4*(sumSqXSeq + 1.0)
                    Scale = (sumSqXSeq * (1.0 + lambdaEff)) + scaleFloor
                    beta1 = (sumXYc / Scale)
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
                varAtView[startIndex]=<cnp.float32_t>(RSS/pairCountDouble)

            if startIndex < maxStartIndex:
                # slide window forward --> (previousSum - leavingValue) + enteringValue
                sumY = (sumY-valuesView[startIndex]) + (valuesView[(startIndex + blockLength)])
                sumSqY = sumSqY + (-(valuesView[startIndex]*valuesView[startIndex]) + (valuesView[(startIndex + blockLength)]*valuesView[(startIndex + blockLength)]))
                sumLagProd = sumLagProd + (-(valuesView[startIndex]*valuesView[(startIndex + 1)]) + (valuesView[(startIndex + blockLength - 1)]*valuesView[(startIndex + blockLength)]))
                maskSum = maskSum + (-<int>maskView[startIndex] + <int>maskView[(startIndex + blockLength)])

        for regionIndex in range(numIntervals):
            startIndex = regionIndex - blockLength + 1
            if startIndex < 0:
                # flag as invalid (i.e., divert to prior model until full window)
                varOutView[regionIndex] = <cnp.float32_t>-1.0
                continue
            if startIndex > maxStartIndex:
                startIndex = maxStartIndex
            varOutView[regionIndex] = varAtView[startIndex]

    return varOut


cpdef cnp.ndarray[cnp.float64_t, ndim=1] cPAVA(
    cnp.ndarray x,
    cnp.ndarray postWeight):
    r"""PAVA for isotonic regression

    This code aims for the notation and algorithm of Busing 2022 (JSS, ``DOI: 10.18637/jss.v102.c01``).

    From Busing:

        > Observe that the violation 8 = x3 > x4 = 2 is solved by combining two values, 8 and 2, resulting
        > in a (new) block value of 5, i.e., (8 + 2)/2 = 5. Instead of immediately turning
        > around and start solving down block violation, we may first look ahead for the next
        > value in the sequence, k-up, for if this element is smaller than or equal to 5, the
        > next value can immediately be pooled into the current block, i.e., (8 + 2 + 2)/3 = 4.
        > Looking ahead can be continued until the next element is larger than the current block
        > value or if we reach the end of the sequence.

    :param x: 1D array to be fitted as nondecreasing
    :type x: cnp.ndarray, (either f32 or f64)
    :param postWeight: 1D array of weights corresponding to each observed value.
      These are the number of 'observations' associated to each 'unique' value in `x`. Intuition: more weight to values with more observations.
    :type postWeight: cnp.ndarray
    :return: PAVA-fitted values
    :rtype: cnp.ndarray, (either f32 or f64)
    """

    cdef cnp.ndarray[cnp.float64_t, ndim=1] xArr = np.ascontiguousarray(x, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wArr = np.ascontiguousarray(postWeight, dtype=np.float64).ravel()
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


cpdef cnp.ndarray[cnp.float64_t, ndim=1] cSF(
    object chromMat,
    bint centerGeoMean=<bint>(True),  # FFR: in fact, we use the _MEDIAN_ for centering!, change in next 0.x+1.0 release
    Py_ssize_t minRefDist=<Py_ssize_t>(10),
):
    #FFR: revisit this, may want to offer guidance given correlation structure...
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

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logSFBuf = np.empty(m, dtype=np.float64)
    cdef double[::1] logSFBufView = logSFBuf

    cdef Py_ssize_t s, i, k
    cdef Py_ssize_t presentCount
    cdef double sumLog, v, medLog, geoMean, eps
    cdef double centerLog
    cdef Py_ssize_t kLow, kHigh
    cdef double low, high
    cdef Py_ssize_t validCols
    cdef double minSF, maxSF
    eps = 1e-8

    # bound scale factors for extreme cases, even if centering is not applied
    minSF = 0.2
    maxSF = 5.0

    # reference uses geometric mean over *positive counts*, less than x% nonzero counts --> NAN (ignored later)
    validCols = 0
    cdef Py_ssize_t requiredNonZeroSamples_ssize = <Py_ssize_t>(1.0 * (<double>m) + (1.0 - 1e-8))  # 100% for consistency w/ other implementations

    # enforce _minimum distance_ between selected reference columns!
    # ... since we're working with adjacent genomic intervals, best
    # ... to avoid local correlations skewing the SF calculation.
    # FFR: consider coupling minRefDist with `getContextSize`
    cdef Py_ssize_t lastSelected = -minRefDist
    cdef Py_ssize_t prevSelected = -1
    cdef Py_ssize_t selectedCount = 0
    cdef double sumGaps = 0.0
    cdef double avgGap = NAN

    with nogil:
        for i in range(n):
            sumLog = 0.0
            presentCount = 0
            for s in range(m):
                v = <double>chromMatView[s, i]
                if v >= 1.0:
                    sumLog += log(v)
                    presentCount += 1

            refLogView[i] = (sumLog / (<double>presentCount)) if presentCount >= requiredNonZeroSamples_ssize else NAN

            if not isnan(refLogView[i]):
                # enforce _minimum distance_ between selected reference columns
                if (i - lastSelected) < minRefDist:
                    refLogView[i] = NAN
                else:
                    validCols += 1
                    lastSelected = i

                    if selectedCount > 0:
                        sumGaps += <double>(i - prevSelected)
                    prevSelected = i
                    selectedCount += 1

        if selectedCount > 1:
            avgGap = (1.0*sumGaps) / <double>(selectedCount - 1)

    printf(
        b"\tcconsenrich.cSF: minRefDist=%zd, selected=%zd, AVG distance between reference cols=%.2f indices\n",
        minRefDist, selectedCount, avgGap
    )

    # ensure there are enough usable columns for the SF calculation
    if validCols < fmax(fmin(<double>(500.0), np.sqrt(<double>(n*0.5))), 10.0):
        raise ValueError(
            f"insufficient valid/dense columns for `countingParams.normMethod: SF`, (need >= 500, got {validCols})... "
            f"If this is expected, consider using `countingParams.normMethod: EGS` or RPKM instead."
        )

    printf(b"\tcconsenrich.cSF: using %zd valid/dense columns for scale factor calculation.\n", validCols)

    with nogil:
        for s in range(m):
            k = 0
            for i in range(n):
                if not isnan(refLogView[i]):
                    v = <double>chromMatView[s, i]
                    if v > 0.0:
                        logRatioBufView[k] = log(v) - refLogView[i]
                        k += 1

            if k == 0:
                scaleFactorsView[s] = 1.0
            else:
                # quickselect for median
                if k & 1:  # case: ODD, just take middle element
                    _nthElement_F64(&logRatioBufView[0], k, k >> 1)
                    medLog = logRatioBufView[k >> 1]
                else:      # case: EVEN, average two middle elements
                    kHigh = k >> 1
                    kLow = kHigh - 1

                    _nthElement_F64(&logRatioBufView[0], k, kHigh)
                    high = logRatioBufView[kHigh]

                    _nthElement_F64(&logRatioBufView[0], k, kLow)
                    low = logRatioBufView[kLow]
                    medLog = 0.5 * (low + high)

                scaleFactorsView[s] = exp(medLog)

            # note that inflated/deflated SFs should be fine after clipping here given later global/local corrections and UQ
            if scaleFactorsView[s] < minSF:
                printf(b"\tWarning: sample scale factor %.4f below min %.4f, clipping to lower.\n", scaleFactorsView[s], minSF)
                scaleFactorsView[s] = minSF
            elif scaleFactorsView[s] > maxSF:
                printf(b"\tWarning: sample scale factor %.4f above max %.4f, clipping to upper.\n", scaleFactorsView[s], maxSF)
                scaleFactorsView[s] = maxSF

        if centerGeoMean and m > 0:
            # robust centering around --median-- log(SF)
            # ... this, and the bounds on SFs should prevent extreme scale factors
            # ... or centering based on pathological samples
            for s in range(m):
                logSFBufView[s] = log(scaleFactorsView[s] + eps)

            # quickselect for median on SFs
            if m & 1:  # case: ODD, just take middle element
                _nthElement_F64(&logSFBufView[0], m, m >> 1)
                centerLog = logSFBufView[m >> 1]
            else:      # case: EVEN, average two middle elements
                kHigh = m >> 1
                kLow = kHigh - 1

                _nthElement_F64(&logSFBufView[0], m, kHigh)
                high = logSFBufView[kHigh]

                _nthElement_F64(&logSFBufView[0], m, kLow)
                low = logSFBufView[kLow]
                centerLog = 0.5 * (low + high)

            geoMean = exp(centerLog)  #  _MEDIAN_
            for s in range(m):
                # center around ~~geometric median~~
                scaleFactorsView[s] /= geoMean

                # make sure bounds still hold
                if scaleFactorsView[s] < minSF:
                    printf(b"\tWarning: sample scale factor %.4f below min %.4f after centering, clipping to lower.\n", scaleFactorsView[s], minSF)
                    scaleFactorsView[s] = minSF
                elif scaleFactorsView[s] > maxSF:
                    printf(b"\tWarning: sample scale factor %.4f above max %.4f after centering, clipping to upper.\n", scaleFactorsView[s], maxSF)
                    scaleFactorsView[s] = maxSF

    return 1 / scaleFactors


cdef void solveSOD_LDL_F64(
    const double[::1] A0_F64,
    const double[::1] A1_F64,
    const double[::1] A2_F64,
    const double[::1] b_F64,
    double[::1] x_F64,
    double[::1] D_F64,
    double[::1] L1_F64,
    double[::1] L2_F64,
    double[::1] y_tmp_F64,
    double[::1] z_tmp_F64) noexcept nogil:
    r"""Solve ``Ax = b`` for a symmetric 5-diagonal matrix ``A``

    Matrix :math:`A \in \mathbb{R}^{n \times n}` is represented in three 1D arrays:

        A0[i] = A[i,i]
        A1[i] = A[i,i+1] = A[i+1,i]
        A2[i] = A[i,i+2] = A[i+2,i]

    i.e., a 5-diagonal matrix with the following structure::

            c0      c1      c2      c3      c4      c5
        r0  A0_0    A1_0    A2_0     .       .       .
        r1  A1_0    A0_1    A1_1    A2_1     .       .
        r2  A2_0    A1_1    A0_2    A1_2    A2_2     .
        r3   .      A2_1    A1_2    A0_3    A1_3    A2_3
        r4   .       .      A2_2    A1_3    A0_4    A1_4
        r5   .       .       .      A2_3    A1_4    A0_5

    """

    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef double t1
    cdef double t2

    # Ax=b, A=LD(L^T), Ly=b, Dz=y, L^Tx=z
    n = A0_F64.shape[0]
    if n == 0:
        return
    if n == 1:
        x_F64[0] = (b_F64[0] / A0_F64[0])
        return

    # LDL decomposition of A
    D_F64[0] = A0_F64[0]
    L1_F64[0] = (A1_F64[0] / D_F64[0])
    if n > 2:
        L2_F64[0] = (A2_F64[0] / D_F64[0])

    D_F64[1] = (A0_F64[1] - ((L1_F64[0] * L1_F64[0]) * D_F64[0]))
    if n > 2:
        t1 = ((L2_F64[0] * D_F64[0]) * L1_F64[0])
        L1_F64[1] = ((A1_F64[1] - t1) / D_F64[1])
    if n > 3:
        L2_F64[1] = (A2_F64[1] / D_F64[1])

    for i in range(2, n):
        t1 = ((L1_F64[i - 1] * L1_F64[i - 1]) * D_F64[i - 1])
        t2 = ((L2_F64[i - 2] * L2_F64[i - 2]) * D_F64[i - 2])
        D_F64[i] = (A0_F64[i] - t1 - t2)

        if i <= (n - 2):
            t1 = ((L2_F64[i - 1] * D_F64[i - 1]) * L1_F64[i - 1])
            L1_F64[i] = ((A1_F64[i] - t1) / D_F64[i])

        if i <= (n - 3):
            L2_F64[i] = (A2_F64[i] / D_F64[i])

    # now solve Ly=b
    y_tmp_F64[0] = b_F64[0]
    y_tmp_F64[1] = (b_F64[1] - (L1_F64[0] * y_tmp_F64[0]))
    for i in range(2, n):
        t1 = (L1_F64[i - 1] * y_tmp_F64[i - 1])
        t2 = (L2_F64[i - 2] * y_tmp_F64[i - 2])
        y_tmp_F64[i] = (b_F64[i] - t1 - t2)

    # ... Dz=y
    for i in range(n):
        z_tmp_F64[i] = (y_tmp_F64[i] / D_F64[i])

    # ... L^Tx=z
    x_F64[n - 1] = z_tmp_F64[n - 1]
    x_F64[n - 2] = (z_tmp_F64[n - 2] - (L1_F64[n - 2] * x_F64[n - 1]))
    for i in range(n - 3, -1, -1):
        t1 = (L1_F64[i] * x_F64[i + 1])
        t2 = (L2_F64[i] * x_F64[i + 2])
        x_F64[i] = (z_tmp_F64[i] - t1 - t2)


cdef void solveSOD_LDL_F32(
    const float[::1] A0_F32,
    const float[::1] A1_F32,
    const float[::1] A2_F32,
    const float[::1] b_F32,
    float[::1] x_F32,
    float[::1] D_F32,
    float[::1] L1_F32,
    float[::1] L2_F32,
    float[::1] y_tmp_F32,
    float[::1] z_tmp_F32
) noexcept nogil:
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef float t1
    cdef float t2

    # Ax=b, A=LD(L^T), Ly=b, Dz=y, L^Tx=z
    n = A0_F32.shape[0]
    if n == 0:
        return
    if n == 1:
        x_F32[0] = (b_F32[0] / A0_F32[0])
        return

    # LDL decomposition of A
    D_F32[0] = A0_F32[0]
    L1_F32[0] = (A1_F32[0] / D_F32[0])
    if n > 2:
        L2_F32[0] = (A2_F32[0] / D_F32[0])

    D_F32[1] = (A0_F32[1] - ((L1_F32[0] * L1_F32[0]) * D_F32[0]))
    if n > 2:
        t1 = ((L2_F32[0] * D_F32[0]) * L1_F32[0])
        L1_F32[1] = ((A1_F32[1] - t1) / D_F32[1])
    if n > 3:
        L2_F32[1] = (A2_F32[1] / D_F32[1])

    for i in range(2, n):
        t1 = ((L1_F32[i - 1] * L1_F32[i - 1]) * D_F32[i - 1])
        t2 = ((L2_F32[i - 2] * L2_F32[i - 2]) * D_F32[i - 2])
        D_F32[i] = (A0_F32[i] - t1 - t2)

        if i <= (n - 2):
            t1 = ((L2_F32[i - 1] * D_F32[i - 1]) * L1_F32[i - 1])
            L1_F32[i] = ((A1_F32[i] - t1) / D_F32[i])

        if i <= (n - 3):
            L2_F32[i] = (A2_F32[i] / D_F32[i])

    # now solve Ly=b
    y_tmp_F32[0] = b_F32[0]
    y_tmp_F32[1] = (b_F32[1] - (L1_F32[0] * y_tmp_F32[0]))
    for i in range(2, n):
        t1 = (L1_F32[i - 1] * y_tmp_F32[i - 1])
        t2 = (L2_F32[i - 2] * y_tmp_F32[i - 2])
        y_tmp_F32[i] = (b_F32[i] - t1 - t2)

    # ... Dz=y
    for i in range(n):
        z_tmp_F32[i] = (y_tmp_F32[i] / D_F32[i])

    # ... L^Tx=z
    x_F32[n - 1] = z_tmp_F32[n - 1]
    x_F32[n - 2] = (z_tmp_F32[n - 2] - (L1_F32[n - 2] * x_F32[n - 1]))
    for i in range(n - 3, -1, -1):
        t1 = (L1_F32[i] * x_F32[i + 1])
        t2 = (L2_F32[i] * x_F32[i + 2])
        x_F32[i] = (z_tmp_F32[i] - t1 - t2)


cpdef cnp.ndarray locBaselineWeighted_F64(cnp.ndarray in_, cnp.ndarray wIn, double lambda_F64):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] b
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] D
    cdef cnp.ndarray[cnp.float64_t, ndim=1] L1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] L2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] yTMP
    cdef cnp.ndarray[cnp.float64_t, ndim=1] zTMP
    cdef double[::1] y_F64
    cdef double[::1] w_F64
    cdef double[::1] b_F64
    cdef double[::1] A0_F64
    cdef double[::1] A1_F64
    cdef double[::1] A2_F64
    cdef double[::1] D_F64
    cdef double[::1] L1_F64
    cdef double[::1] L2_F64
    cdef double[::1] y_tmp_F64
    cdef double[::1] z_tmp_F64
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef double wi, dtd0
    cdef double lam = lambda_F64
    # FFR: revisit, but using pointers directly has proven faster in this function (and F64)
    cdef double* yPtr
    cdef double* wPtr
    cdef double* bPtr
    cdef double* a0Ptr
    cdef double* a1Ptr
    cdef double* a2Ptr

    y = np.ascontiguousarray(in_, dtype=np.float64)
    wArr = np.ascontiguousarray(wIn, dtype=np.float64)
    y_F64 = y
    w_F64 = wArr
    n = y_F64.shape[0]
    if n < 3:
        return y.copy()
    if w_F64.shape[0] != n:
        raise ValueError("weight length mismatch")

    b = np.empty(n, dtype=np.float64)
    A0 = np.empty(n, dtype=np.float64)
    A1 = np.empty(n - 1, dtype=np.float64)
    A2 = np.empty(n - 2, dtype=np.float64)
    D = np.empty(n, dtype=np.float64)
    L1 = np.empty(n - 1, dtype=np.float64)
    L2 = np.empty(n - 2, dtype=np.float64)
    yTMP = np.empty(n, dtype=np.float64)
    zTMP = np.empty(n, dtype=np.float64)

    b_F64 = b
    A0_F64 = A0
    A1_F64 = A1
    A2_F64 = A2
    D_F64 = D
    L1_F64 = L1
    L2_F64 = L2
    y_tmp_F64 = yTMP
    z_tmp_F64 = zTMP
    yPtr = <double*>y.data
    wPtr = <double*>wArr.data
    bPtr = <double*>b.data
    a0Ptr = <double*>A0.data
    a1Ptr = <double*>A1.data
    a2Ptr = <double*>A2.data

    with nogil:
        # D^T D bands are fixed for 2nd-difference penalty:
        #   DTD0 = [1, 5, 6, ..., 6, 5, 1]
        #   DTD1 = [-2, -4, ..., -4, -2]
        #   DTD2 = [1, 1, ..., 1]
        #
        # (W + lam D^T D)b = Wy

        # fill A2 (second off-diagonal): +lam everywhere
        for i in range(n - 2):
            a2Ptr[i] = lam

        # fill A1 (first off-diagonal): -2lam at ends, -4lam in middle
        a1Ptr[0] = -2.0 * lam
        for i in range(1, n - 2):
            a1Ptr[i] = -4.0 * lam
        a1Ptr[n - 2] = -2.0 * lam

        # fill A0 (diagonal) and rhs b
        for i in range(n):
            wi = wPtr[i]
            if wi < 0.0:
                wi = 0.0

            # diag of D^T D
            if i == 0 or i == (n - 1):
                dtd0 = 1.0
            elif i == 1 or i == (n - 2):
                dtd0 = 5.0
            else:
                dtd0 = 6.0

            a0Ptr[i] = wi + lam * dtd0
            bPtr[i] = wi * yPtr[i]

        solveSOD_LDL_F64(A0_F64, A1_F64, A2_F64, b_F64, b_F64,
                         D_F64, L1_F64, L2_F64, y_tmp_F64, z_tmp_F64)

    return b


cpdef cnp.ndarray locBaselineWeighted_F32(cnp.ndarray in_, cnp.ndarray wIn, float lambda_F32):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] y
    cdef cnp.ndarray[cnp.float32_t, ndim=1] wArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1] b
    cdef cnp.ndarray[cnp.float32_t, ndim=1] A0
    cdef cnp.ndarray[cnp.float32_t, ndim=1] A1
    cdef cnp.ndarray[cnp.float32_t, ndim=1] A2
    cdef cnp.ndarray[cnp.float32_t, ndim=1] D
    cdef cnp.ndarray[cnp.float32_t, ndim=1] L1
    cdef cnp.ndarray[cnp.float32_t, ndim=1] L2
    cdef cnp.ndarray[cnp.float32_t, ndim=1] yTMP
    cdef cnp.ndarray[cnp.float32_t, ndim=1] zTMP

    cdef float[::1] y_F32
    cdef float[::1] w_F32
    cdef float[::1] b_F32
    cdef float[::1] A0_F32
    cdef float[::1] A1_F32
    cdef float[::1] A2_F32
    cdef float[::1] D_F32
    cdef float[::1] L1_F32
    cdef float[::1] L2_F32
    cdef float[::1] y_tmp_F32
    cdef float[::1] z_tmp_F32

    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef float wi, dtd0
    cdef float lam = lambda_F32

    # FFR: revisit, but using pointers directly has proven faster in this function (and F64)
    cdef float* yPtr
    cdef float* wPtr
    cdef float* bPtr
    cdef float* a0Ptr
    cdef float* a1Ptr
    cdef float* a2Ptr

    y = np.ascontiguousarray(in_, dtype=np.float32)
    wArr = np.ascontiguousarray(wIn, dtype=np.float32)
    y_F32 = y
    w_F32 = wArr
    n = y_F32.shape[0]
    if n < 3:
        return y.copy()
    if w_F32.shape[0] != n:
        raise ValueError("weight length mismatch")

    b = np.empty(n, dtype=np.float32)
    A0 = np.empty(n, dtype=np.float32)
    A1 = np.empty(n - 1, dtype=np.float32)
    A2 = np.empty(n - 2, dtype=np.float32)
    D = np.empty(n, dtype=np.float32)
    L1 = np.empty(n - 1, dtype=np.float32)
    L2 = np.empty(n - 2, dtype=np.float32)
    yTMP = np.empty(n, dtype=np.float32)
    zTMP = np.empty(n, dtype=np.float32)

    b_F32 = b
    A0_F32 = A0
    A1_F32 = A1
    A2_F32 = A2
    D_F32 = D
    L1_F32 = L1
    L2_F32 = L2
    y_tmp_F32 = yTMP
    z_tmp_F32 = zTMP

    # pointer setup (safe: contiguous arrays)
    yPtr = <float*>y.data
    wPtr = <float*>wArr.data
    bPtr = <float*>b.data
    a0Ptr = <float*>A0.data
    a1Ptr = <float*>A1.data
    a2Ptr = <float*>A2.data

    with nogil:
        # D^T D bands are fixed for 2nd-difference penalty:
        #   DTD0 = [1, 5, 6, ..., 6, 5, 1]
        #   DTD1 = [-2, -4, ..., -4, -2]
        #   DTD2 = [1, 1, ..., 1]
        #
        # (W + lam D^T D)b = Wy

        # fill A2 (second off-diagonal): +lam everywhere
        for i in range(n - 2):
            a2Ptr[i] = lam

        # fill A1 (first off-diagonal): -2lam at ends, -4lam in middle
        a1Ptr[0] = <float>(-2.0) * lam
        for i in range(1, n - 2):
            a1Ptr[i] = <float>(-4.0) * lam
        a1Ptr[n - 2] = <float>(-2.0) * lam

        # fill A0 (diagonal) and rhs b
        for i in range(n):
            wi = wPtr[i]
            if wi < <float>(0.0):
                wi = <float>(0.0)

            # diag of D^T D
            if i == 0 or i == (n - 1):
                dtd0 = <float>(1.0)
            elif i == 1 or i == (n - 2):
                dtd0 = <float>(5.0)
            else:
                dtd0 = <float>(6.0)

            a0Ptr[i] = wi + lam * dtd0
            bPtr[i] = wi * yPtr[i]

        solveSOD_LDL_F32(A0_F32, A1_F32, A2_F32, b_F32, b_F32,
                         D_F32, L1_F32, L2_F32, y_tmp_F32, z_tmp_F32)

    return b


cpdef cnp.ndarray locBaselineCrossfit2w_F64(cnp.ndarray yIn, cnp.ndarray wIn, double lambda_F64):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wEven
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wOdd
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bEven
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bOdd
    cdef cnp.ndarray[cnp.float64_t, ndim=1] b
    cdef Py_ssize_t n, i

    cdef double[::1] wView
    cdef double[::1] wEvenView
    cdef double[::1] wOddView
    cdef double[::1] bView
    cdef double[::1] bEvenView
    cdef double[::1] bOddView

    y = np.ascontiguousarray(yIn, dtype=np.float64)
    w = np.ascontiguousarray(wIn, dtype=np.float64)
    n = y.shape[0]
    if n < 3:
        return y.copy()
    if w.shape[0] != n:
        raise ValueError("weight length mismatch")

    wEven = np.empty(n, dtype=np.float64)
    wOdd  = np.empty(n, dtype=np.float64)
    wView = w
    wEvenView = wEven
    wOddView = wOdd

    with nogil:
        for i in range(n):
            if (i & 1) == 0:
                wEvenView[i] = wView[i]
                wOddView[i]  = 0.0
            else:
                wEvenView[i] = 0.0
                wOddView[i]  = wView[i]

    bEven = locBaselineWeighted_F64(y, wEven, lambda_F64)
    bOdd  = locBaselineWeighted_F64(y, wOdd,  lambda_F64)

    b = np.empty(n, dtype=np.float64)
    bView = b
    bEvenView = bEven
    bOddView  = bOdd
    with nogil:
        for i in range(n):
            bView[i] = 0.5 * (bEvenView[i] + bOddView[i])
    return b


cpdef cnp.ndarray locBaselineCrossfit2w_F32(cnp.ndarray yIn, cnp.ndarray wIn, float lambda_F32):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] y
    cdef cnp.ndarray[cnp.float32_t, ndim=1] w
    cdef cnp.ndarray[cnp.float32_t, ndim=1] wEven
    cdef cnp.ndarray[cnp.float32_t, ndim=1] wOdd
    cdef cnp.ndarray[cnp.float32_t, ndim=1] bEven
    cdef cnp.ndarray[cnp.float32_t, ndim=1] bOdd
    cdef cnp.ndarray[cnp.float32_t, ndim=1] b
    cdef Py_ssize_t n, i

    cdef float[::1] wView
    cdef float[::1] wEvenView
    cdef float[::1] wOddView
    cdef float[::1] bView
    cdef float[::1] bEvenView
    cdef float[::1] bOddView

    y = np.ascontiguousarray(yIn, dtype=np.float32)
    w = np.ascontiguousarray(wIn, dtype=np.float32)
    n = y.shape[0]
    if n < 3:
        return y.copy()
    if w.shape[0] != n:
        raise ValueError("weight length mismatch")

    wEven = np.empty(n, dtype=np.float32)
    wOdd  = np.empty(n, dtype=np.float32)
    wView = w
    wEvenView = wEven
    wOddView = wOdd

    with nogil:
        for i in range(n):
            if (i & 1) == 0:
                wEvenView[i] = wView[i]
                wOddView[i]  = 0.0
            else:
                wEvenView[i] = 0.0
                wOddView[i]  = wView[i]

    bEven = locBaselineWeighted_F32(y, wEven, lambda_F32)
    bOdd  = locBaselineWeighted_F32(y, wOdd,  lambda_F32)

    b = np.empty(n, dtype=np.float32)
    bView = b
    bEvenView = bEven
    bOddView  = bOdd
    with nogil:
        for i in range(n):
            bView[i] = <cnp.float32_t>(0.5) * (bEvenView[i] + bOddView[i])
    return b


cpdef cnp.ndarray locBaselineMasked_F64(cnp.ndarray in_, cnp.ndarray fitMaskIn, double lambda_F64):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] fitMask
    cdef cnp.ndarray[cnp.float64_t, ndim=1] b
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] A2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] DTD0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] DTD1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] DTD2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] D
    cdef cnp.ndarray[cnp.float64_t, ndim=1] L1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] L2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] yTMP
    cdef cnp.ndarray[cnp.float64_t, ndim=1] zTMP

    cdef double[::1] y_F64
    cdef cnp.uint8_t[::1] fitMask_U8
    cdef double[::1] b_F64
    cdef double[::1] A0_F64
    cdef double[::1] A1_F64
    cdef double[::1] A2_F64
    cdef double[::1] DTD0_F64
    cdef double[::1] DTD1_F64
    cdef double[::1] DTD2_F64
    cdef double[::1] D_F64
    cdef double[::1] L1_F64
    cdef double[::1] L2_F64
    cdef double[::1] y_tmp_F64
    cdef double[::1] z_tmp_F64

    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef double wi_F64

    y = np.ascontiguousarray(in_, dtype=np.float64)
    fitMask = np.ascontiguousarray(fitMaskIn, dtype=np.uint8)
    y_F64 = y
    fitMask_U8 = fitMask
    n = y_F64.shape[0]
    if n < 3:
        return y.copy()
    if fitMask_U8.shape[0] != n:
        raise ValueError("fitMask length mismatch")

    b = np.empty(n, dtype=np.float64)
    A0 = np.empty(n, dtype=np.float64)
    A1 = np.empty(n - 1, dtype=np.float64)
    A2 = np.empty(n - 2, dtype=np.float64)
    DTD0 = np.empty(n, dtype=np.float64)
    DTD1 = np.empty(n - 1, dtype=np.float64)
    DTD2 = np.empty(n - 2, dtype=np.float64)
    D = np.empty(n, dtype=np.float64)
    L1 = np.empty(n - 1, dtype=np.float64)
    L2 = np.empty(n - 2, dtype=np.float64)
    yTMP = np.empty(n, dtype=np.float64)
    zTMP = np.empty(n, dtype=np.float64)

    b_F64 = b
    A0_F64 = A0
    A1_F64 = A1
    A2_F64 = A2
    DTD0_F64 = DTD0
    DTD1_F64 = DTD1
    DTD2_F64 = DTD2
    D_F64 = D
    L1_F64 = L1
    L2_F64 = L2
    y_tmp_F64 = yTMP
    z_tmp_F64 = zTMP

    with nogil:
        # D^T D
        DTD0_F64[0] = 1.0
        DTD0_F64[1] = 5.0
        for i in range(2, n - 2):
            DTD0_F64[i] = 6.0
        DTD0_F64[n - 2] = 5.0
        DTD0_F64[n - 1] = 1.0

        DTD1_F64[0] = -2.0
        for i in range(1, n - 2):
            DTD1_F64[i] = -4.0
        DTD1_F64[n - 2] = -2.0

        for i in range(n - 2):
            DTD2_F64[i] = 1.0

        # (W + lam D^T D)b = Wy
        for i in range(n):
            wi_F64 = 1.0 if fitMask_U8[i] else 0.0 # crossfit mask
            A0_F64[i] = wi_F64 + lambda_F64 * DTD0_F64[i]
            b_F64[i] = wi_F64 * y_F64[i]
        for i in range(n - 1):
            A1_F64[i] = lambda_F64 * DTD1_F64[i]
        for i in range(n - 2):
            A2_F64[i] = lambda_F64 * DTD2_F64[i]

        solveSOD_LDL_F64(A0_F64, A1_F64, A2_F64, b_F64, b_F64,
                         D_F64, L1_F64, L2_F64, y_tmp_F64, z_tmp_F64)

    return b


cpdef cnp.ndarray locBaselineMasked_F32(cnp.ndarray in_, cnp.ndarray fitMaskIn, float lambda_F32):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] y
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] fitMask
    cdef cnp.ndarray[cnp.float32_t, ndim=1] b
    cdef cnp.ndarray[cnp.float32_t, ndim=1] A0
    cdef cnp.ndarray[cnp.float32_t, ndim=1] A1
    cdef cnp.ndarray[cnp.float32_t, ndim=1] A2
    cdef cnp.ndarray[cnp.float32_t, ndim=1] DTD0
    cdef cnp.ndarray[cnp.float32_t, ndim=1] DTD1
    cdef cnp.ndarray[cnp.float32_t, ndim=1] DTD2
    cdef cnp.ndarray[cnp.float32_t, ndim=1] D
    cdef cnp.ndarray[cnp.float32_t, ndim=1] L1
    cdef cnp.ndarray[cnp.float32_t, ndim=1] L2
    cdef cnp.ndarray[cnp.float32_t, ndim=1] yTMP
    cdef cnp.ndarray[cnp.float32_t, ndim=1] zTMP

    cdef float[::1] y_F32
    cdef cnp.uint8_t[::1] fitMask_U8
    cdef float[::1] b_F32
    cdef float[::1] A0_F32
    cdef float[::1] A1_F32
    cdef float[::1] A2_F32
    cdef float[::1] DTD0_F32
    cdef float[::1] DTD1_F32
    cdef float[::1] DTD2_F32
    cdef float[::1] D_F32
    cdef float[::1] L1_F32
    cdef float[::1] L2_F32
    cdef float[::1] y_tmp_F32
    cdef float[::1] z_tmp_F32
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef float wi_F32

    y = np.ascontiguousarray(in_, dtype=np.float32)
    fitMask = np.ascontiguousarray(fitMaskIn, dtype=np.uint8)
    y_F32 = y
    fitMask_U8 = fitMask
    n = y_F32.shape[0]
    if n < 3:
        return y.copy()
    if fitMask_U8.shape[0] != n:
        raise ValueError("fitMask length mismatch")

    b = np.empty(n, dtype=np.float32)
    A0 = np.empty(n, dtype=np.float32)
    A1 = np.empty(n - 1, dtype=np.float32)
    A2 = np.empty(n - 2, dtype=np.float32)
    DTD0 = np.empty(n, dtype=np.float32)
    DTD1 = np.empty(n - 1, dtype=np.float32)
    DTD2 = np.empty(n - 2, dtype=np.float32)
    D = np.empty(n, dtype=np.float32)
    L1 = np.empty(n - 1, dtype=np.float32)
    L2 = np.empty(n - 2, dtype=np.float32)
    yTMP = np.empty(n, dtype=np.float32)
    zTMP = np.empty(n, dtype=np.float32)

    b_F32 = b
    A0_F32 = A0
    A1_F32 = A1
    A2_F32 = A2
    DTD0_F32 = DTD0
    DTD1_F32 = DTD1
    DTD2_F32 = DTD2
    D_F32 = D
    L1_F32 = L1
    L2_F32 = L2
    y_tmp_F32 = yTMP
    z_tmp_F32 = zTMP

    with nogil:
        # D^T D
        DTD0_F32[0] = 1.0
        DTD0_F32[1] = 5.0
        for i in range(2, n - 2):
            DTD0_F32[i] = 6.0
        DTD0_F32[n - 2] = 5.0
        DTD0_F32[n - 1] = 1.0

        DTD1_F32[0] = -2.0
        for i in range(1, n - 2):
            DTD1_F32[i] = -4.0
        DTD1_F32[n - 2] = -2.0

        for i in range(n - 2):
            DTD2_F32[i] = 1.0

        # (W + lam D^T D)b = Wy
        for i in range(n):
            wi_F32 = 1.0 if fitMask_U8[i] else 0.0 # crossfit mask
            A0_F32[i] = wi_F32 + lambda_F32 * DTD0_F32[i]
            b_F32[i] = wi_F32 * y_F32[i]
        for i in range(n - 1):
            A1_F32[i] = lambda_F32 * DTD1_F32[i]
        for i in range(n - 2):
            A2_F32[i] = lambda_F32 * DTD2_F32[i]

        solveSOD_LDL_F32(A0_F32, A1_F32, A2_F32, b_F32, b_F32,
                         D_F32, L1_F32, L2_F32, y_tmp_F32, z_tmp_F32)

    return b


cpdef cnp.ndarray locBaselineCrossfit2_F32(cnp.ndarray in_, float lambda_F32):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] y
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mEven
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mOdd
    cdef cnp.ndarray[cnp.float32_t, ndim=1] bEven
    cdef cnp.ndarray[cnp.float32_t, ndim=1] bOdd
    cdef cnp.ndarray[cnp.float32_t, ndim=1] b
    cdef Py_ssize_t n
    cdef Py_ssize_t i

    # for nogil
    cdef cnp.uint8_t[::1] mEven_U8
    cdef cnp.uint8_t[::1] mOdd_U8
    cdef cnp.float32_t[::1] bView
    cdef cnp.float32_t[::1] bEvenView
    cdef cnp.float32_t[::1] bOddView

    y = np.ascontiguousarray(in_, dtype=np.float32)
    n = y.shape[0]
    if n < 3:
        return y.copy()

    mEven = np.empty(n, dtype=np.uint8)
    mOdd = np.empty(n, dtype=np.uint8)
    mEven_U8 = mEven
    mOdd_U8 = mOdd

    with nogil:
        for i in range(n):
            mEven_U8[i] = 1 if (i & 1) == 0 else 0
            mOdd_U8[i] = 1 if (i & 1) == 1 else 0

    # solve for even cols and odd cols separately, then average
    bEven = locBaselineMasked_F32(y, mEven, lambda_F32)
    bOdd = locBaselineMasked_F32(y, mOdd, lambda_F32)

    b = np.empty(n, dtype=np.float32)
    bView = b
    bEvenView = bEven
    bOddView = bOdd

    with nogil:
        for i in range(n):
            bView[i] = 0.5 * (bEvenView[i] + bOddView[i])
    return b


cpdef cnp.ndarray locBaselineCrossfit2_F64(cnp.ndarray yIn, double lambda_F64):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] yArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mEvenArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mOddArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bEvenArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bOddArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bArr
    cdef Py_ssize_t n
    cdef Py_ssize_t i

    # for nogil
    cdef cnp.uint8_t[::1] mEven_U8
    cdef cnp.uint8_t[::1] mOdd_U8
    cdef cnp.float64_t[::1] bView
    cdef cnp.float64_t[::1] bEvenView
    cdef cnp.float64_t[::1] bOddView

    yArr = np.ascontiguousarray(yIn, dtype=np.float64)
    n = yArr.shape[0]
    if n < 3:
        return yArr.copy()

    mEvenArr = np.empty(n, dtype=np.uint8)
    mOddArr = np.empty(n, dtype=np.uint8)
    mEven_U8 = mEvenArr
    mOdd_U8 = mOddArr

    with nogil:
        for i in range(n):
            mEven_U8[i] = 1 if (i & 1) == 0 else 0
            mOdd_U8[i] = 1 if (i & 1) == 1 else 0

    # solve even/odd separately (akin to 2-fold CV)
    # ... so that self-fit bias is mitigated
    bEvenArr = locBaselineMasked_F64(yArr, mEvenArr, lambda_F64)
    bOddArr = locBaselineMasked_F64(yArr, mOddArr, lambda_F64)

    bArr = np.empty(n, dtype=np.float64)
    bView = bArr
    bEvenView = bEvenArr
    bOddView = bOddArr

    with nogil:
        for i in range(n):
            bView[i] = 0.5 * (bEvenView[i] + bOddView[i])
    return bArr


cdef void _initPenaltyBandsF64(
        Py_ssize_t n,
        double lambda_,
        double[::1] pen0, # lambda * diag(D^T D)   (n,)
        double[::1] a1, # lambda * off1(D^T D)   (n-1,)
        double[::1] a2  # lambda * off2(D^T D)   (n-2,)
) noexcept nogil:
    cdef Py_ssize_t i

    # Note, D^T D bands are fixed for 2nd-difference penalty:
    #   diag: [1, 5, 6, ..., 6, 5, 1]
    #   off1: [-2, -4, ..., -4, -2]
    #   off2: [1, 1, ..., 1]
    #
    # so we can store as pre-multiplied by lambda_ st the per-iter update is just A0 = w + pen0
    if n <= 0:
        return

    # pen0 (diag)
    pen0[0] = lambda_ * 1.0
    if n > 1:
        pen0[1] = lambda_ * 5.0
    for i in range(2, n - 2):
        pen0[i] = lambda_ * 6.0
    if n > 2:
        pen0[n - 2] = lambda_ * 5.0
        pen0[n - 1] = lambda_ * 1.0

    # a1 (off1)
    if n > 1:
        a1[0] = lambda_ * (-2.0)
        for i in range(1, n - 2):
            a1[i] = lambda_ * (-4.0)
        a1[n - 2] = lambda_ * (-2.0)

    # a2 (off2)
    for i in range(n - 2):
        a2[i] = lambda_


cdef void _solveBaselineWeightedInplaceF64(
        const double* yPtr,
        const double* wPtr,
        Py_ssize_t n,
        const double[::1] pen0,
        const double[::1] a1,
        const double[::1] a2,
        double[::1] a0,
        double[::1] rhs,
        double[::1] D,
        double[::1] L1,
        double[::1] L2,
        double[::1] yTmp,
        double[::1] zTmp
) noexcept nogil:
    cdef Py_ssize_t i
    cdef double wi

    # (W + lam D^T D)b = Wy
    for i in range(n):
        wi = wPtr[i]
        if wi < 0.0:
            wi = 0.0
        a0[i] = wi + pen0[i]
        rhs[i] = wi * yPtr[i]

    solveSOD_LDL_F64(a0, a1, a2, rhs, rhs, D, L1, L2, yTmp, zTmp)


cpdef cnp.ndarray[cnp.float32_t, ndim=1] clocalBaseline(
        object x,
        int blockSize=101,
        bint useIRLS=<bint>(True),
        double asymPos=<double>(2.0/5.0),
        double noiseMult=<double>(1.0),
        double cauchyScale=<double>(3.0),
        int maxIters=<int>(25),
        double minWeight=<double>(1.0e-6),
        double tol=<double>(1.0e-3)):
    r"""Estimate a *local* baseline on `x` with a lower/smooth envelope via IRLS

    Compute a locally smooth baseline :math:`\hat{b}` for an input signal :math:`y`,
    using a second-order penalized smoother (Whittaker) with *asymmetric* iteratively reweighted
    least squares (IRLS) to reduce influence from peaks.

    :param x: Signal measurements over fixed-length genomic intervals
    :type x: np.ndarray
    :param asymPos: *Relative* weight assigned to positive residuals to induce asymmetry in reweighting. Used
      during IRLS for the local baseline computation. Smaller values will downweight peaks more and pose less
      risk of removing true signal. Typical range is ``(0, 0.75]``.
    :type asymPos: float
    :param cauchyScale: Controls how quickly weights decay with normalized residual magnitude. Smaller values downweight
      outliers strongly.
    :type cauchyScale: float
    :return: Baseline estimate as a float32 NumPy array of shape ``(n,)``.
    :rtype: numpy.ndarray

    """

    cdef cnp.ndarray[cnp.float64_t, ndim=1] y
    cdef cnp.ndarray[cnp.float64_t, ndim=1] wArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] resid
    cdef cnp.ndarray[cnp.float64_t, ndim=1] baseArr # smoother solution
    cdef cnp.ndarray[cnp.float64_t, ndim=1] a0Arr # diag (n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] a1Arr # off1 (n-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] a2Arr # off2 (n-2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pen0Arr # lambda*diag(D^T D)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] dArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] l1Arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] l2Arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] yTmpArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] zTmpArr
    # pointers to memviews
    cdef double* yPtr
    cdef double* basePtr
    cdef double* residPtr
    cdef double* wPtr

    cdef Py_ssize_t n, i
    cdef int it
    cdef double lambda_ = <double>(0.0)
    cdef double w_
    cdef double r, t
    cdef double wPrev, wUpdated, newW, dW, maxDW
    cdef double noiseScale
    cdef double invN
    cdef double absSum, absMean, clipThr
    cdef double r2Sum, clipR
    cdef double eps_ = 1.0e-12

    # memviews for nogil calls into solver
    cdef double[::1] baseView
    cdef double[::1] a0View
    cdef double[::1] a1View
    cdef double[::1] a2View
    cdef double[::1] pen0View
    cdef double[::1] dView
    cdef double[::1] l1View
    cdef double[::1] l2View
    cdef double[::1] yTmpView
    cdef double[::1] zTmpView

    y = np.ascontiguousarray(x, dtype=np.float64)
    n = y.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)

    if blockSize < 3:
        blockSize = 3
    if (blockSize & 1) == 0:
        blockSize += 1

    # lambda_ is set so the gain is about 1/8 (and decreasing) for frequencies greater than 1/blockSize,
    # ... so that the baseline is smooth at scales smaller than blockSize but can follow larger-scale trends
    w_ = blockSize * 0.15915494
    lambda_ = (w_ * w_ * w_ * w_)*7.0

    if useIRLS == False:
        return np.zeros(n, dtype=np.float32)

    if n < 25:
        printf("\tcconsenrich.clocalBaseline: input_len < 25  -->  returning zeros\n")
        return np.zeros(n, dtype=np.float32)

    # weights and residuals
    wArr = np.empty(n, dtype=np.float64)
    resid = np.empty(n, dtype=np.float64)

    # allocate everything once here and avoid malloc/free inside IRLS
    baseArr = np.empty(n, dtype=np.float64)
    a0Arr = np.empty(n, dtype=np.float64)
    a1Arr = np.empty(n - 1, dtype=np.float64)
    a2Arr = np.empty(n - 2, dtype=np.float64)
    pen0Arr = np.empty(n, dtype=np.float64)
    dArr = np.empty(n, dtype=np.float64)
    l1Arr = np.empty(n - 1, dtype=np.float64)
    l2Arr = np.empty(n - 2, dtype=np.float64)
    yTmpArr = np.empty(n, dtype=np.float64)
    zTmpArr = np.empty(n, dtype=np.float64)
    yPtr = <double*>y.data
    basePtr = <double*>baseArr.data
    residPtr = <double*>resid.data
    wPtr = <double*>wArr.data

    # memviews for nogil solver call
    baseView = baseArr
    a0View = a0Arr
    a1View = a1Arr
    a2View = a2Arr
    pen0View = pen0Arr
    dView = dArr
    l1View = l1Arr
    l2View = l2Arr
    yTmpView = yTmpArr
    zTmpView = zTmpArr

    with nogil:
        # initialize with uniform weights
        for i in range(n):
            wPtr[i] = 1.0

        # precompute _constant_ penalty bands for the solver: lambda * diag(D^T D), lambda * off1(D^T D), lambda * off2(D^T D)
        _initPenaltyBandsF64(n, lambda_, pen0View, a1View, a2View)

    invN = 1.0 / <double>(n) # precompute once
    maxDW = 0.0
    for it in range(maxIters):

        # For current weights, W, solve (LDL) for a baseline `b` to minimize:
        # [(W^(1/2)(y-b))^T (W^(1/2)(y-b))] + [lambda (D b)^T (D b)]
        # (D is the second-difference operator, so the second term penalizes ~jumps~ in `b`)
        with nogil:
            _solveBaselineWeightedInplaceF64(
                yPtr, wPtr, n,
                pen0View, a1View, a2View,
                a0View, baseView,
                dView, l1View, l2View,
                yTmpView, zTmpView
            )

            for i in range(n):
                # residual = obs - baseline
                residPtr[i] = yPtr[i] - basePtr[i]

            # estimate noiseScale each pass via clipped _RMS_ of residuals:
            #   noiseScale = noiseMult * sqrt(mean(clip(r, +/-clipThr)^2))
            absSum = 0.0
            for i in range(n):
                absSum += fabs(residPtr[i])

            absMean = absSum * invN
            clipThr = 6.0 * absMean
            if clipThr < 1.0e-12 or not isfinite(clipThr):
                clipThr = 1.0e-12

            r2Sum = 0.0
            for i in range(n):
                clipR = residPtr[i]
                if clipR > clipThr:
                    clipR = clipThr
                elif clipR < -clipThr:
                    clipR = -clipThr
                r2Sum += clipR * clipR

            noiseScale = noiseMult * sqrt(r2Sum * invN)
            if noiseScale <= eps_ or not isfinite(noiseScale):
                noiseScale = eps_

            # initialized per pass
            maxDW = 0.0
            for i in range(n):
                wPrev = wPtr[i]
                r = residPtr[i]

                if r > 0.0:
                    # case: data > estimated baseline  -->  weigh by `asymPos`
                    t = r / noiseScale
                    wUpdated = (asymPos) / (1.0 + (t / cauchyScale) * (t / cauchyScale))
                else:
                    # case: data <= estimated baseline  --> weigh by 1-asymPos
                    t = (-r) / noiseScale
                    wUpdated = (1.0 - asymPos) / (1.0 + (t / cauchyScale) * (t / cauchyScale))

                if wUpdated < minWeight:
                    wUpdated = minWeight

                # smooth update
                newW = 0.5 * wPrev + 0.5 * wUpdated
                wPtr[i] = newW

                # measure change vs previous for convergence
                dW = fabs(newW - wPrev)
                if dW > maxDW:
                    # record new max change
                    maxDW = dW

        if maxDW < tol:
            printf("\tcconsenrich.clocalBaseline: converged at iteration %d with max weight change %.6f\n", it, maxDW)
            break

    with nogil:
        _solveBaselineWeightedInplaceF64(
            yPtr, wPtr, n,
            pen0View, a1View, a2View,
            a0View, baseView,
            dView, l1View, l2View,
            yTmpView, zTmpView
        )

    return baseArr.astype(np.float32)
