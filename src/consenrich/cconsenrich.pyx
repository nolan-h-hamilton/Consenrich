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
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from numpy.random import default_rng
from libc.math cimport isfinite, fabs, log1p, log2, log, log2f, logf, asinhf, asinh, fmax, fmaxf, pow, sqrt, sqrtf, fabsf, fminf, fmin, log10, log10f, ceil, floor, floorf, exp, expf, isnan, NAN, INFINITY
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf, fprintf, fflush, stdout, stderr

cdef extern from "htslib/hts.h":
    ctypedef struct htsFile
    ctypedef struct hts_idx_t
    ctypedef struct hts_itr_t
    ctypedef long long hts_pos_t

    int hts_set_threads(htsFile* fp, int n)
    void hts_idx_destroy(hts_idx_t* idx)
    void hts_itr_destroy(hts_itr_t* itr)


cdef extern from "htslib/sam.h":
    ctypedef struct samFile

    ctypedef struct bam1_core_t:
        int32_t tid
        int32_t pos
        uint16_t bin
        uint8_t qual
        uint16_t l_qname
        uint16_t n_cigar
        uint16_t flag
        int32_t l_qseq
        int32_t mtid
        int32_t mpos
        int64_t isize

    ctypedef struct bam1_t:
        bam1_core_t core

    ctypedef struct sam_hdr_t:
        int32_t n_targets
        uint32_t* target_len
        char** target_name

    samFile* sam_open(const char* fn, const char* mode)
    int sam_close(samFile* fp)
    sam_hdr_t* sam_hdr_read(samFile* fp)
    void sam_hdr_destroy(sam_hdr_t* h)
    bam1_t* bam_init1()
    void bam_destroy1(bam1_t* b)
    int sam_read1(samFile* fp, sam_hdr_t* h, bam1_t* b)
    hts_idx_t* sam_index_load(htsFile* fp, const char* fn)
    int sam_hdr_name2tid(sam_hdr_t* h, const char* ref)
    hts_itr_t* sam_itr_queryi(hts_idx_t* idx, int tid, hts_pos_t beg, hts_pos_t end)
    int sam_itr_next(htsFile* htsfp, hts_itr_t* itr, bam1_t* r)
    hts_pos_t bam_endpos(bam1_t* b)
    uint32_t* bam_get_cigar(bam1_t* b)
    hts_pos_t bam_cigar2qlen(int n_cigar, uint32_t* cigar)

cnp.import_array()

# ========
# constants
# ========
cdef const float __INV_LN2_FLOAT = <float>1.44269504
cdef const double __INV_LN2_DOUBLE = <double>1.44269504088896340
ctypedef fused real_t:
    float
    double

# ===============
# inline/helpers
# ===============

cpdef tuple cExpectedTransitionResidualSums(
    cnp.ndarray[cnp.float64_t, ndim=2] stateSmoothed,
    cnp.ndarray[cnp.float64_t, ndim=3] stateCovarSmoothed,
    cnp.ndarray[cnp.float64_t, ndim=3] lagCovSmoothed,
    cnp.ndarray[cnp.float64_t, ndim=2] matrixF,
):
    cdef Py_ssize_t n = stateSmoothed.shape[0]
    cdef Py_ssize_t transitionCount = n - 1
    cdef Py_ssize_t k
    cdef double f00
    cdef double f01
    cdef double f10
    cdef double f11
    cdef double x00
    cdef double x01
    cdef double x10
    cdef double x11
    cdef double y0
    cdef double y1
    cdef double exx0_00
    cdef double exx0_01
    cdef double exx0_10
    cdef double exx0_11
    cdef double exx1_00
    cdef double exx1_11
    cdef double ex0x1_00
    cdef double ex0x1_01
    cdef double ex0x1_10
    cdef double ex0x1_11
    cdef double levelMoment
    cdef double trendMoment
    cdef double sumLevel = 0.0
    cdef double sumTrend = 0.0
    cdef Py_ssize_t requiredLagCount

    requiredLagCount = transitionCount if transitionCount > 0 else 0
    if stateSmoothed.shape[1] != 2:
        raise ValueError("stateSmoothed must have shape (n, 2)")
    if (
        stateCovarSmoothed.shape[0] != n
        or stateCovarSmoothed.shape[1] != 2
        or stateCovarSmoothed.shape[2] != 2
    ):
        raise ValueError("stateCovarSmoothed must have shape (n, 2, 2)")
    if (
        lagCovSmoothed.shape[0] < requiredLagCount
        or lagCovSmoothed.shape[1] != 2
        or lagCovSmoothed.shape[2] != 2
    ):
        raise ValueError("lagCovSmoothed must have shape (n - 1, 2, 2)")
    if matrixF.shape[0] != 2 or matrixF.shape[1] != 2:
        raise ValueError("matrixF must have shape (2, 2)")
    if transitionCount <= 0:
        return 0.0, 0.0, 0

    f00 = matrixF[0, 0]
    f01 = matrixF[0, 1]
    f10 = matrixF[1, 0]
    f11 = matrixF[1, 1]

    for k in range(transitionCount):
        x00 = stateSmoothed[k, 0]
        x01 = stateSmoothed[k, 1]
        x10 = stateSmoothed[k + 1, 0]
        x11 = stateSmoothed[k + 1, 1]

        exx0_00 = stateCovarSmoothed[k, 0, 0] + (x00 * x00)
        exx0_01 = stateCovarSmoothed[k, 0, 1] + (x00 * x01)
        exx0_10 = stateCovarSmoothed[k, 1, 0] + (x01 * x00)
        exx0_11 = stateCovarSmoothed[k, 1, 1] + (x01 * x01)

        y0 = x10
        y1 = x11
        exx1_00 = stateCovarSmoothed[k + 1, 0, 0] + (y0 * y0)
        exx1_11 = stateCovarSmoothed[k + 1, 1, 1] + (y1 * y1)

        ex0x1_00 = lagCovSmoothed[k, 0, 0] + (x00 * y0)
        ex0x1_01 = lagCovSmoothed[k, 0, 1] + (x00 * y1)
        ex0x1_10 = lagCovSmoothed[k, 1, 0] + (x01 * y0)
        ex0x1_11 = lagCovSmoothed[k, 1, 1] + (x01 * y1)

        levelMoment = (
            exx1_00
            - (2.0 * ((f00 * ex0x1_00) + (f01 * ex0x1_10)))
            + (f00 * f00 * exx0_00)
            + (f00 * f01 * exx0_01)
            + (f01 * f00 * exx0_10)
            + (f01 * f01 * exx0_11)
        )
        trendMoment = (
            exx1_11
            - (2.0 * ((f10 * ex0x1_01) + (f11 * ex0x1_11)))
            + (f10 * f10 * exx0_00)
            + (f10 * f11 * exx0_01)
            + (f11 * f10 * exx0_10)
            + (f11 * f11 * exx0_11)
        )

        if levelMoment < 0.0:
            levelMoment = 0.0
        if trendMoment < 0.0:
            trendMoment = 0.0
        sumLevel += levelMoment
        sumTrend += trendMoment

    return sumLevel, sumTrend, transitionCount

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


cdef inline int _maskMembership(const uint32_t* pos, Py_ssize_t numIntervals, const uint32_t* mStarts, const uint32_t* mEnds, Py_ssize_t n, uint32_t intervalSizeBP, uint8_t* outMask) nogil:
    # CALLERS: `cbedMask`

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t k
    cdef uint32_t p
    cdef uint32_t intervalEnd
    while i < numIntervals:
        p = pos[i]
        intervalEnd = p + intervalSizeBP
        k = _getInsertion(mStarts, n, intervalEnd - 1) - 1
        if k >= 0 and mEnds[k] > p:
            outMask[i] = <uint8_t>1
        else:
            outMask[i] = <uint8_t>0
        i += 1
    return 0


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
    cdef double ScaleX
    cdef double ScaleY
    cdef double denomSym

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

        eps = 1.0e-6*(sumSqXSeq + sumSqYSeq + 1.0)

        # scale-aware ridge
        if nPairsDouble > 0.0:
            lambdaEff = pairsRegLambda / (nPairsDouble + 1.0)
        else:
            lambdaEff = pairsRegLambda

        scaleFloor = 1.0e-4*(sumSqXSeq + 1.0)

        # _symmetry_: lag-1 correlation for beta
        ScaleX = (sumSqXSeq * (1.0 + lambdaEff)) + scaleFloor
        ScaleY = (sumSqYSeq * (1.0 + lambdaEff)) + scaleFloor
        # both directions accounted for
        denomSym = sqrt(ScaleX * ScaleY)
        if denomSym > eps:
            beta1 = sumXYc / denomSym
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


cdef inline double _secondDiffPenaltyDiag(Py_ssize_t n, Py_ssize_t i, double lam) noexcept nogil:
    if n < 3 or lam <= 0.0:
        return 0.0
    if n == 3:
        if i == 1:
            return 4.0 * lam
        return lam
    if i == 0 or i == n - 1:
        return lam
    if i == 1 or i == n - 2:
        return 5.0 * lam
    return 6.0 * lam


cdef inline double _secondDiffPenaltyOff1(Py_ssize_t n, Py_ssize_t i, double lam) noexcept nogil:
    if n < 3 or lam <= 0.0:
        return 0.0
    if n == 3:
        return -2.0 * lam
    if i == 0 or i == n - 2:
        return -2.0 * lam
    return -4.0 * lam


cdef inline double _firstDiffPenaltyDiag(Py_ssize_t n, Py_ssize_t i, double lam) noexcept nogil:
    if n < 2 or lam <= 0.0:
        return 0.0
    if i == 0 or i == n - 1:
        return lam
    return 2.0 * lam


cdef inline double _firstDiffPenaltyOff1(Py_ssize_t n, double lam) noexcept nogil:
    if n < 2 or lam <= 0.0:
        return 0.0
    return -lam


cpdef cnp.ndarray[cnp.float32_t, ndim=1] csolveZeroCenteredBackground(
    cnp.ndarray[cnp.float64_t, ndim=1] weightTrack,
    cnp.ndarray[cnp.float64_t, ndim=1] rhsTrack,
    double lam,
    bint zeroCenter=True,
    double lamFirst=<double>0.0,
):
    r"""Solve the roughness-penalized background update.

    Solves ``(diag(weightTrack) + lamFirst * D1.T @ D1 + lam * D2.T @ D2) x =
    rhsTrack`` using a pentadiagonal LDL' factorization. If ``zeroCenter`` is
    true, also applies the identifiability constraint ``sum(x) = 0`` via a
    Lagrange multiplier.
    """

    cdef Py_ssize_t n = weightTrack.shape[0]
    cdef Py_ssize_t i
    cdef double minPivot = 1.0e-12
    cdef double offVal
    cdef double l2Val
    cdef double sumRhs = 0.0
    cdef double sumConstraint = 0.0
    cdef double mu = 0.0
    cdef double denomOne
    cdef cnp.ndarray[cnp.float64_t, ndim=1] diag
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rhs
    cdef cnp.ndarray[cnp.float64_t, ndim=1] constraintSolve
    cdef cnp.ndarray[cnp.float64_t, ndim=1] firstLower
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out
    cdef double[::1] diagView
    cdef double[::1] rhsView
    cdef double[::1] constraintView
    cdef double[::1] firstLowerView
    cdef cnp.float32_t[::1] outView

    if rhsTrack.shape[0] != n:
        raise ValueError("weightTrack and rhsTrack must have the same length")

    out = np.zeros(n, dtype=np.float32)
    if n <= 0:
        return out
    if n == 1:
        if not zeroCenter:
            denomOne = <double>weightTrack[0]
            if denomOne < minPivot:
                denomOne = minPivot
            out[0] = <cnp.float32_t>((<double>rhsTrack[0]) / denomOne)
        return out

    diag = np.ascontiguousarray(weightTrack, dtype=np.float64)
    rhs = np.ascontiguousarray(rhsTrack, dtype=np.float64)
    constraintSolve = np.ones(n, dtype=np.float64)
    firstLower = np.zeros(n, dtype=np.float64)

    diagView = diag
    rhsView = rhs
    constraintView = constraintSolve
    firstLowerView = firstLower
    outView = out

    with nogil:
        for i in range(n):
            diagView[i] = (
                diagView[i]
                + _firstDiffPenaltyDiag(n, i, lamFirst)
                + _secondDiffPenaltyDiag(n, i, lam)
            )
            if diagView[i] < minPivot:
                diagView[i] = minPivot

        # Pentadiagonal LDL' factorization. The second lower diagonal is
        # lam / diag[i - 2] and can be recomputed, so only the first lower
        # diagonal needs storage.
        offVal = _firstDiffPenaltyOff1(n, lamFirst) + _secondDiffPenaltyOff1(n, 0, lam)
        firstLowerView[1] = offVal / diagView[0]
        diagView[1] = diagView[1] - firstLowerView[1] * firstLowerView[1] * diagView[0]
        if diagView[1] < minPivot:
            diagView[1] = minPivot

        for i in range(2, n):
            offVal = _firstDiffPenaltyOff1(n, lamFirst) + _secondDiffPenaltyOff1(n, i - 1, lam)
            firstLowerView[i] = (offVal - lam * firstLowerView[i - 1]) / diagView[i - 1]
            diagView[i] = (
                diagView[i]
                - firstLowerView[i] * firstLowerView[i] * diagView[i - 1]
                - (lam * lam) / diagView[i - 2]
            )
            if diagView[i] < minPivot:
                diagView[i] = minPivot

        # Forward solve for two RHS vectors: the data RHS and A^{-1}1 for
        # the zero-sum Lagrange multiplier.
        rhsView[1] = rhsView[1] - firstLowerView[1] * rhsView[0]
        constraintView[1] = constraintView[1] - firstLowerView[1] * constraintView[0]
        for i in range(2, n):
            l2Val = lam / diagView[i - 2]
            rhsView[i] = rhsView[i] - firstLowerView[i] * rhsView[i - 1] - l2Val * rhsView[i - 2]
            constraintView[i] = constraintView[i] - firstLowerView[i] * constraintView[i - 1] - l2Val * constraintView[i - 2]

        for i in range(n):
            rhsView[i] = rhsView[i] / diagView[i]
            constraintView[i] = constraintView[i] / diagView[i]

        # Backward solve.
        rhsView[n - 2] = rhsView[n - 2] - firstLowerView[n - 1] * rhsView[n - 1]
        constraintView[n - 2] = constraintView[n - 2] - firstLowerView[n - 1] * constraintView[n - 1]
        for i in range(n - 3, -1, -1):
            l2Val = lam / diagView[i]
            rhsView[i] = rhsView[i] - firstLowerView[i + 1] * rhsView[i + 1] - l2Val * rhsView[i + 2]
            constraintView[i] = constraintView[i] - firstLowerView[i + 1] * constraintView[i + 1] - l2Val * constraintView[i + 2]

        if zeroCenter:
            for i in range(n):
                sumRhs += rhsView[i]
                sumConstraint += constraintView[i]
            if fabs(sumConstraint) > minPivot:
                mu = sumRhs / sumConstraint
            else:
                mu = sumRhs / <double>n

            for i in range(n):
                outView[i] = <cnp.float32_t>(rhsView[i] - mu * constraintView[i])
        else:
            for i in range(n):
                outView[i] = <cnp.float32_t>rhsView[i]

    return out


cdef inline bint _swapReal(real_t* swapInArray_, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef real_t tmp = swapInArray_[i]
    swapInArray_[i] = swapInArray_[j]
    swapInArray_[j] = tmp
    return <bint>0


cdef inline Py_ssize_t _partitionLtReal(real_t* vals_, Py_ssize_t left, Py_ssize_t right, Py_ssize_t pivot) noexcept nogil:
    cdef real_t pv = vals_[pivot]
    cdef Py_ssize_t store = left
    cdef Py_ssize_t i
    _swapReal(vals_, pivot, right)
    for i in range(left, right):
        if vals_[i] < pv:
            _swapReal(vals_, store, i)
            store += 1
    _swapReal(vals_, store, right)
    return store


cdef inline bint _nthElementReal(real_t* sortedVals_, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = n - 1
    cdef Py_ssize_t pivot, idx
    while left < right:
        pivot = (left + right) >> 1
        idx = _partitionLtReal(sortedVals_, left, right, pivot)
        if k == idx:
            return <bint>0
        elif k < idx:
            right = idx - 1
        else:
            left = idx + 1
    return <bint>0


cdef inline bint _nthElement(float* sortedVals_, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    # CALLERS: `_quantileInplaceF32`

    return _nthElementReal(sortedVals_, n, k)


cdef inline bint _nthElement_F64(double* sortedVals_, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    # CALLERS: `cSF`, `_quantileInplaceF64`

    return _nthElementReal(sortedVals_, n, k)


cdef inline real_t _quantileInplaceReal(real_t* vals_, Py_ssize_t n, real_t q, real_t emptyValue) noexcept nogil:
    cdef Py_ssize_t k
    if n <= 0:
        return emptyValue
    if q <= <real_t>0.0:
        k = 0
    elif q >= <real_t>1.0:
        k = n - 1
    else:
        k = <Py_ssize_t>floor(<double>(q * <real_t>(n - 1)))
    _nthElementReal(vals_, n, k)
    return vals_[k]


cdef inline double _quantileInplaceF64(double* vals_, Py_ssize_t n, double q) noexcept nogil:
    return <double>_quantileInplaceReal(vals_, n, q, <double>0.0)


cdef inline float _quantileInplaceF32(float* vals_, Py_ssize_t n, float q) noexcept nogil:
    # CALLERS: `_medianCopy_F32`

    return <float>_quantileInplaceReal(vals_, n, q, <float>1.0)


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


cdef inline void _insertionSortF64(double* vals_, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double key
    for i in range(1, n):
        key = vals_[i]
        j = i
        while j > 0 and vals_[j - 1] > key:
            vals_[j] = vals_[j - 1]
            j -= 1
        vals_[j] = key


cpdef cnp.ndarray[cnp.float64_t, ndim=1] ctrimMeanAxis0(
    cnp.ndarray values,
    double trim=0.10,
):
    r"""Column-wise finite trimmed mean for replicate-by-interval matrices."""

    cdef cnp.ndarray[cnp.float64_t, ndim=2] values2d
    cdef cnp.ndarray[cnp.float64_t, ndim=1] values1d
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    cdef double[:, ::1] valuesView
    cdef double[::1] values1dView
    cdef double[::1] outView
    cdef Py_ssize_t rowCount, colCount, rowIndex, colIndex
    cdef Py_ssize_t validCount, trimCount, loIndex, hiIndex, workIndex
    cdef double* work
    cdef double value
    cdef double sumValue

    if values.ndim == 1:
        values1d = np.ascontiguousarray(values, dtype=np.float64)
        out = np.empty(values1d.shape[0], dtype=np.float64)
        values1dView = values1d
        outView = out
        with nogil:
            for colIndex in range(values1dView.shape[0]):
                value = values1dView[colIndex]
                if isfinite(value):
                    outView[colIndex] = value
                else:
                    outView[colIndex] = NAN
        return out

    if values.ndim != 2:
        raise ValueError("values must be one- or two-dimensional")

    values2d = np.ascontiguousarray(values, dtype=np.float64)
    rowCount = <Py_ssize_t>values2d.shape[0]
    colCount = <Py_ssize_t>values2d.shape[1]
    out = np.empty(colCount, dtype=np.float64)
    if colCount <= 0:
        return out
    if rowCount <= 0:
        out[:] = np.nan
        return out

    if trim < 0.0:
        trim = 0.0
    elif trim >= 0.5:
        trim = 0.499999

    work = <double*>malloc(rowCount * sizeof(double))
    if work == NULL:
        raise MemoryError("failed to allocate trimmed-mean work buffer")

    valuesView = values2d
    outView = out
    try:
        with nogil:
            for colIndex in range(colCount):
                validCount = 0
                for rowIndex in range(rowCount):
                    value = valuesView[rowIndex, colIndex]
                    if isfinite(value):
                        work[validCount] = value
                        validCount += 1

                if validCount <= 0:
                    outView[colIndex] = NAN
                    continue

                _insertionSortF64(work, validCount)
                trimCount = <Py_ssize_t>floor(trim * <double>validCount)
                loIndex = trimCount
                hiIndex = validCount - trimCount
                if hiIndex <= loIndex:
                    loIndex = 0
                    hiIndex = validCount

                sumValue = 0.0
                for workIndex in range(loIndex, hiIndex):
                    sumValue += work[workIndex]
                outView[colIndex] = sumValue / <double>(hiIndex - loIndex)
    finally:
        free(work)

    return out


cpdef tuple cdependenceLengthStats(
    cnp.ndarray[cnp.float64_t, ndim=1] centeredTrack,
    int maxSpan,
):
    r"""Autocorrelation and normalized increment variance over finite runs."""

    cdef cnp.ndarray[cnp.float64_t, ndim=1] trackArr = np.ascontiguousarray(
        centeredTrack,
        dtype=np.float64,
    )
    cdef Py_ssize_t n = <Py_ssize_t>trackArr.shape[0]
    cdef Py_ssize_t maxSpan_ = <Py_ssize_t>max(maxSpan, 0)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] acfNum = np.zeros(maxSpan_, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] incNum = np.zeros(maxSpan_, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] acf = np.zeros(maxSpan_, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] inc = np.zeros(maxSpan_, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] pairCounts = np.zeros(maxSpan_, dtype=np.int64)
    cdef double[::1] trackView = trackArr
    cdef double[::1] acfNumView = acfNum
    cdef double[::1] incNumView = incNum
    cdef double[::1] acfView = acf
    cdef double[::1] incView = inc
    cdef int64_t[::1] pairCountView = pairCounts
    cdef Py_ssize_t i, lag, runStart, runEnd, j
    cdef Py_ssize_t finiteCount = 0
    cdef double value, nextValue, diff
    cdef double gamma0Sum = 0.0
    cdef double gamma0 = 0.0

    if n <= 0 or maxSpan_ <= 0:
        return acf, inc, pairCounts, float(gamma0), int(finiteCount)

    with nogil:
        for i in range(n):
            value = trackView[i]
            if isfinite(value):
                finiteCount += 1
                gamma0Sum += value * value

        if finiteCount > 0:
            gamma0 = gamma0Sum / <double>finiteCount

        if gamma0 > 0.0 and isfinite(gamma0):
            i = 0
            while i < n:
                while i < n and not isfinite(trackView[i]):
                    i += 1
                runStart = i
                while i < n and isfinite(trackView[i]):
                    i += 1
                runEnd = i
                if runEnd - runStart > 1:
                    for lag in range(1, maxSpan_ + 1):
                        if runEnd - runStart <= lag:
                            break
                        for j in range(runStart, runEnd - lag):
                            value = trackView[j]
                            nextValue = trackView[j + lag]
                            acfNumView[lag - 1] += value * nextValue
                            diff = nextValue - value
                            incNumView[lag - 1] += diff * diff
                            pairCountView[lag - 1] += 1

            for lag in range(maxSpan_):
                if pairCountView[lag] > 0:
                    acfView[lag] = (
                        acfNumView[lag] / <double>pairCountView[lag]
                    ) / gamma0
                    incView[lag] = (
                        0.5 * incNumView[lag] / <double>pairCountView[lag]
                    ) / gamma0
                else:
                    acfView[lag] = NAN
                    incView[lag] = NAN
        else:
            for lag in range(maxSpan_):
                acfView[lag] = NAN
                incView[lag] = NAN

    return acf, inc, pairCounts, float(gamma0), int(finiteCount)


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


cpdef bint cisAlignmentPairedEnd(
    str bamFile,
    int64_t maxReads=1000,
    uint16_t samThreads=0,
    uint16_t samFlagExclude=3844,
):
    r"""Return True when sampled alignment records carry the paired-end flag."""
    cdef bytes bamFileBytes = bamFile.encode("utf-8")
    cdef samFile* fileHandle = NULL
    cdef sam_hdr_t* header = NULL
    cdef bam1_t* record = NULL
    cdef int64_t sampled = 0
    cdef int readFlag
    cdef bint isPairedEnd = <bint>0

    if maxReads < 1:
        maxReads = 1

    fileHandle = sam_open(bamFileBytes, "r")
    if fileHandle == NULL:
        raise FileNotFoundError(f"Could not open alignment file `{bamFile}`")

    try:
        if samThreads > 1:
            hts_set_threads(<htsFile*>fileHandle, <int>samThreads)

        header = sam_hdr_read(fileHandle)
        if header == NULL:
            raise OSError(f"Could not read alignment header for `{bamFile}`")

        record = bam_init1()
        if record == NULL:
            raise MemoryError("failed to allocate BAM record")

        while sampled < maxReads and sam_read1(fileHandle, header, record) >= 0:
            readFlag = <int>record.core.flag
            if (readFlag & samFlagExclude) != 0:
                continue
            sampled += 1
            if (readFlag & 1) != 0:
                isPairedEnd = <bint>1
                break

        return isPairedEnd
    finally:
        if record != NULL:
            bam_destroy1(record)
        if header != NULL:
            sam_hdr_destroy(header)
        if fileHandle != NULL:
            sam_close(fileHandle)


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
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lengthsArr
    cdef Py_ssize_t contigIdx
    cdef int contigTid
    cdef int64_t contigLen
    cdef int kTop
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
    cdef int nFFT
    cdef object Ff
    cdef object Fr
    cdef object corr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tlenArr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tlenWork
    cdef cnp.int32_t[::1] tlenWorkView
    cdef int tlenN
    cdef int midIdx
    cdef cnp.int32_t medPE
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] bestLagsArr
    cdef bytes bamFileBytes = bamFile.encode("utf-8")
    cdef samFile* fileHandle = NULL
    cdef sam_hdr_t* header = NULL
    cdef hts_idx_t* indexHandle = NULL
    cdef hts_itr_t* iteratorHandle = NULL
    cdef bam1_t* record = NULL
    cdef hts_pos_t queryLength = 0

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

    fileHandle = sam_open(bamFileBytes, "r")
    if fileHandle == NULL:
        return <int64_t>fallBack

    try:
        if samThreadsInternal > 1:
            hts_set_threads(<htsFile*>fileHandle, samThreadsInternal)

        header = sam_hdr_read(fileHandle)
        if header == NULL or header.n_targets <= 0:
            return <int64_t>fallBack

        indexHandle = sam_index_load(<htsFile*>fileHandle, bamFileBytes)
        if indexHandle == NULL:
            return <int64_t>fallBack

        record = bam_init1()
        if record == NULL:
            return <int64_t>fallBack

        lengthsArr = np.empty(header.n_targets, dtype=np.int64)
        for contigIdx in range(header.n_targets):
            lengthsArr[contigIdx] = <int64_t>header.target_len[contigIdx]

        kTop = 3 if header.n_targets >= 3 else (2 if header.n_targets >= 2 else 1)
        topContigsIdx = np.argpartition(lengthsArr, -kTop)[-kTop:]
        topContigsIdx = topContigsIdx[np.argsort(lengthsArr[topContigsIdx])[::-1]]

        for contigIdx in topContigsIdx:
            contigTid = <int>contigIdx
            contigLen = <int64_t>lengthsArr[contigTid]
            if contigLen <= 0:
                continue

            iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, <hts_pos_t>contigLen)
            if iteratorHandle == NULL:
                continue

            while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                readFlag = <int>record.core.flag
                if (readFlag & samFlagExclude) != 0:
                    continue

                if not isPairedEnd and (readFlag & 1) != 0:
                    isPairedEnd = <bint>1

                if numReadLengthSamples >= iters:
                    break

                queryLength = record.core.l_qseq
                if queryLength <= 0 and record.core.n_cigar > 0:
                    queryLength = bam_cigar2qlen(record.core.n_cigar, bam_get_cigar(record))
                if queryLength <= 0:
                    continue

                avgReadLength += <double>queryLength
                numReadLengthSamples += 1

            hts_itr_destroy(iteratorHandle)
            iteratorHandle = NULL

            if numReadLengthSamples >= iters:
                break

        if numReadLengthSamples <= 0:
            return <int64_t>fallBack

        avgReadLength /= <double>numReadLengthSamples
        minInsertSize = <int64_t>(avgReadLength)
        if minInsertSize < 1:
            minInsertSize = 1
        if minInsertSize > maxInsertSize:
            minInsertSize = maxInsertSize

        if isPairedEnd:
            requiredSamplesPE = max(iters, 2000)
            tlenArr = np.empty(requiredSamplesPE, dtype=np.int32)
            tlenN = 0

            for contigIdx in topContigsIdx:
                if tlenN >= requiredSamplesPE:
                    break

                contigTid = <int>contigIdx
                contigLen = <int64_t>lengthsArr[contigTid]
                if contigLen <= 0:
                    continue

                iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, <hts_pos_t>contigLen)
                if iteratorHandle == NULL:
                    continue

                while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                    if tlenN >= requiredSamplesPE:
                        break

                    readFlag = <int>record.core.flag
                    if (readFlag & samFlagExclude) != 0:
                        continue
                    if (readFlag & 2) == 0:
                        continue
                    if (readFlag & 64) == 0:
                        continue

                    tlen = <int64_t>record.core.isize
                    if tlen == 0:
                        continue
                    if tlen < 0:
                        tlen = -tlen

                    if tlen < minInsertSize or tlen > maxInsertSize:
                        continue

                    tlenArr[tlenN] = <cnp.int32_t>tlen
                    tlenN += 1

                hts_itr_destroy(iteratorHandle)
                iteratorHandle = NULL

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
            contigTid = <int>contigIdx
            contigLen = <int64_t>lengthsArr[contigTid]
            regionLen = contigLen

            if regionLen < blockSize or regionLen <= 0:
                continue

            numRollSteps = regionLen // rollingChunkSize
            if numRollSteps <= 0:
                numRollSteps = 1
            numChunks = <int>numRollSteps

            rawArr = np.zeros(numChunks, dtype=np.float64)
            medArr = np.zeros(numChunks, dtype=np.float64)

            iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, <hts_pos_t>contigLen)
            if iteratorHandle == NULL:
                continue

            while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                readFlag = <int>record.core.flag
                if (readFlag & samFlagExclude) != 0:
                    continue
                j = <int>(record.core.pos // rollingChunkSize)
                if 0 <= j < numChunks:
                    rawArr[j] += 1.0

            hts_itr_destroy(iteratorHandle)
            iteratorHandle = NULL

            winSize = <int>(blockSize // rollingChunkSize)
            if winSize < 1:
                winSize = 1
            if (winSize & 1) == 0:
                winSize += 1
            medArr[:] = ndimage.median_filter(rawArr, size=winSize, mode="nearest")

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
                blockStartBP = idxVal*rollingChunkSize + (rollingChunkSize // 2) - blockHalf
                if blockStartBP < 0:
                    blockStartBP = 0
                blockEndBP = blockStartBP + blockSize
                if blockEndBP > contigLen:
                    blockEndBP = contigLen
                    blockStartBP = blockEndBP - blockSize
                    if blockStartBP < 0:
                        continue

                fwd.fill(0.0)
                rev.fill(0.0)

                iteratorHandle = sam_itr_queryi(
                    indexHandle,
                    contigTid,
                    <hts_pos_t>blockStartBP,
                    <hts_pos_t>blockEndBP,
                )
                if iteratorHandle == NULL:
                    continue

                while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                    readFlag = <int>record.core.flag
                    if (readFlag & samFlagExclude) != 0:
                        continue

                    readStart = <int64_t>record.core.pos
                    readEnd = <int64_t>bam_endpos(record)
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

                hts_itr_destroy(iteratorHandle)
                iteratorHandle = NULL

                maxValidLag = maxInsertSize if (maxInsertSize < blockSize) else (blockSize - 1)
                localMinLag = <int>minInsertSize
                localMaxLag = <int>maxValidLag
                if localMaxLag < localMinLag:
                    continue
                localLagStep = <int>lagStep

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
        if iteratorHandle != NULL:
            hts_itr_destroy(iteratorHandle)
        if record != NULL:
            bam_destroy1(record)
        if indexHandle != NULL:
            hts_idx_destroy(indexHandle)
        if header != NULL:
            sam_hdr_destroy(header)
        if fileHandle != NULL:
            sam_close(fileHandle)

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
            _maskMembership(posPtr, numIntervals, svPtr, evPtr, n, <uint32_t>intervalSizeBP, outPtr)
    return mask


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
    outMeans = np.empty(iters, dtype=np.float32)
    outVars = np.empty(iters, dtype=np.float32)
    valuesLength = <Py_ssize_t>valuesArray.size
    maxBlockLength = <Py_ssize_t>blockSize
    if valuesLength <= 0 or maxBlockLength <= 0 or iters <= 0:
        outMeans[:] = 0.0
        outVars[:] = 0.0
        emptyStarts = np.empty(0, dtype=np.intp)
        emptyEnds = np.empty(0, dtype=np.intp)
        return outMeans, outVars, emptyStarts, emptyEnds

    geomProb = 1.0 / (<double>maxBlockLength)
    sizesArray = rng.geometric(geomProb, size=iters).astype(np.intp, copy=False)
    np.maximum(sizesArray, <cnp.intp_t>min(5, valuesLength), out=sizesArray)
    np.minimum(sizesArray, <cnp.intp_t>min(5*maxBlockLength, valuesLength), out=sizesArray)
    maxBlockLength = <Py_ssize_t>sizesArray.max()
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
    ends = starts_ + sizesArray

    startsView = starts_
    sizesView = sizesArray
    meansView = outMeans
    varsView = outVars

    _regionMeanVar(
        valuesView,
        startsView,
        sizesView,
        meansView,
        varsView,
        zeroPenalty,
        zeroThresh,
        useInnovationVar,
        useSampleVar,
    )

    return outMeans, outVars, starts_, ends


cpdef tuple cSparseNearestMeanVarTrack(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    cnp.ndarray[cnp.intp_t, ndim=1] sparseCenters,
    cnp.ndarray[cnp.intp_t, ndim=1] blockStarts,
    cnp.ndarray[cnp.intp_t, ndim=1] blockSizes,
    int numNearest,
    double zeroPenalty=0.0,
    double zeroThresh=0.0,
    bint useInnovationVar=True,
    bint useSampleVar=False,
    bint aggregateMeanAbs=True,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] valuesArray
    cdef double[::1] valuesView
    cdef Py_ssize_t sparseCount
    cdef Py_ssize_t intervalCount
    cdef cnp.ndarray[cnp.float32_t, ndim=1] sparseMeans
    cdef cnp.ndarray[cnp.float32_t, ndim=1] sparseVars
    cdef cnp.ndarray[cnp.float32_t, ndim=1] outMeans
    cdef cnp.ndarray[cnp.float32_t, ndim=1] outVars
    cdef float[::1] sparseMeansView
    cdef float[::1] sparseVarsView
    cdef float[::1] outMeansView
    cdef float[::1] outVarsView
    cdef Py_ssize_t[::1] sparseCentersView = sparseCenters
    cdef Py_ssize_t[::1] blockStartsView = blockStarts
    cdef Py_ssize_t[::1] blockSizesView = blockSizes
    cdef Py_ssize_t i, left, right, chosenIdx, usedCount
    cdef Py_ssize_t nearestTarget
    cdef Py_ssize_t lo, hi, mid, insertPos
    cdef double sumMean, sumVar
    cdef double leftDist, rightDist

    intervalCount = <Py_ssize_t>values.shape[0]
    sparseCount = <Py_ssize_t>sparseCenters.shape[0]
    outMeans = np.empty(intervalCount, dtype=np.float32)
    outVars = np.empty(intervalCount, dtype=np.float32)

    if intervalCount <= 0:
        return outMeans, outVars

    if (
        sparseCount <= 0
        or blockStarts.shape[0] != sparseCount
        or blockSizes.shape[0] != sparseCount
        or numNearest <= 0
    ):
        outMeans.fill(np.float32(0.0))
        outVars.fill(np.float32(np.nan))
        return outMeans, outVars

    valuesArray = np.ascontiguousarray(values, dtype=np.float64)
    valuesView = valuesArray
    sparseMeans = np.empty(sparseCount, dtype=np.float32)
    sparseVars = np.empty(sparseCount, dtype=np.float32)
    sparseMeansView = sparseMeans
    sparseVarsView = sparseVars

    _regionMeanVar(
        valuesView,
        blockStartsView,
        blockSizesView,
        sparseMeansView,
        sparseVarsView,
        zeroPenalty,
        zeroThresh,
        useInnovationVar,
        useSampleVar,
    )

    outMeansView = outMeans
    outVarsView = outVars

    for i in range(intervalCount):
        lo = 0
        hi = sparseCount
        while lo < hi:
            mid = lo + ((hi - lo) // 2)
            if sparseCentersView[mid] < i:
                lo = mid + 1
            else:
                hi = mid
        insertPos = lo
        left = insertPos - 1
        right = insertPos
        nearestTarget = numNearest
        if nearestTarget > sparseCount:
            nearestTarget = sparseCount

        usedCount = 0
        sumMean = 0.0
        sumVar = 0.0

        while usedCount < nearestTarget and (left >= 0 or right < sparseCount):
            if left < 0:
                chosenIdx = right
                right += 1
            elif right >= sparseCount:
                chosenIdx = left
                left -= 1
            else:
                leftDist = <double>(i - sparseCentersView[left])
                if leftDist < 0.0:
                    leftDist = -leftDist
                rightDist = <double>(sparseCentersView[right] - i)
                if rightDist < 0.0:
                    rightDist = -rightDist
                if leftDist <= rightDist:
                    chosenIdx = left
                    left -= 1
                else:
                    chosenIdx = right
                    right += 1

            if aggregateMeanAbs:
                sumMean += fabs(<double>sparseMeansView[chosenIdx])
            else:
                sumMean += <double>sparseMeansView[chosenIdx]
            sumVar += <double>sparseVarsView[chosenIdx]
            usedCount += 1

        if usedCount > 0:
            outMeansView[i] = <cnp.float32_t>(sumMean / <double>usedCount)
            outVarsView[i] = <cnp.float32_t>(sumVar / <double>usedCount)
        else:
            outMeansView[i] = <cnp.float32_t>0.0
            outVarsView[i] = <cnp.float32_t>np.nan

    return outMeans, outVars


cdef bint _cEMA(const real_t* xPtr, real_t* outPtr,
                Py_ssize_t n, real_t alpha) noexcept nogil:
    cdef Py_ssize_t i
    if alpha > <real_t>1.0 or alpha < <real_t>0.0:
        return <bint>1

    outPtr[0] = xPtr[0]

    for i in range(1, n):
        outPtr[i] = alpha*xPtr[i] + (1.0 - alpha)*outPtr[i - 1]

    for i in range(n - 2, -1, -1):
        outPtr[i] = alpha*outPtr[i] + (1.0 - alpha)*outPtr[i + 1]

    return <bint>0


cdef inline Py_ssize_t _bsplineSpan(
    const double* knotsPtr,
    Py_ssize_t nBasis,
    int degree,
    double x,
) noexcept nogil:
    cdef Py_ssize_t low = degree
    cdef Py_ssize_t high = nBasis
    cdef Py_ssize_t mid

    if x <= knotsPtr[degree]:
        return degree
    if x >= knotsPtr[nBasis]:
        return nBasis - 1

    while low < high:
        mid = low + ((high - low) >> 1)
        if x < knotsPtr[mid]:
            high = mid
        elif x >= knotsPtr[mid + 1]:
            low = mid + 1
        else:
            return mid
    return nBasis - 1


cdef inline double _deBoorValue(
    const double* knotsPtr,
    const double* betaPtr,
    Py_ssize_t nBasis,
    int degree,
    double x,
    double* work,
) noexcept nogil:
    cdef Py_ssize_t span = _bsplineSpan(knotsPtr, nBasis, degree, x)
    cdef int j, r
    cdef Py_ssize_t idx
    cdef double denom, alpha

    for j in range(degree + 1):
        idx = span - degree + j
        if idx < 0:
            idx = 0
        elif idx >= nBasis:
            idx = nBasis - 1
        work[j] = betaPtr[idx]

    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            idx = span - degree + j
            denom = knotsPtr[idx + degree - r + 1] - knotsPtr[idx]
            if denom == 0.0:
                alpha = 0.0
            else:
                alpha = (x - knotsPtr[idx]) / denom
            work[j] = ((1.0 - alpha) * work[j - 1]) + (alpha * work[j])

    return work[degree]


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cEvalPSplineLogVarianceTrend(
    object predictorTrack,
    object knots,
    object beta,
    int degree,
    double xMin,
    double xMax,
    double logFloor,
    double logCap,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] predictorArr = np.ascontiguousarray(predictorTrack, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] knotsArr = np.ascontiguousarray(knots, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] betaArr = np.ascontiguousarray(beta, dtype=np.float64).ravel()
    cdef Py_ssize_t n = predictorArr.shape[0]
    cdef Py_ssize_t nBasis = betaArr.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)
    cdef const double* predictorPtr = <const double*>predictorArr.data
    cdef const double* knotsPtr = <const double*>knotsArr.data
    cdef const double* betaPtr = <const double*>betaArr.data
    cdef cnp.float32_t* outPtr = <cnp.float32_t*>out.data
    cdef double* work = NULL
    cdef Py_ssize_t i
    cdef double x, logOut

    if n == 0:
        return out

    if degree < 0 or knotsArr.shape[0] == 0 or nBasis == 0:
        logOut = betaPtr[0] if nBasis > 0 else logFloor
        if not isfinite(logOut):
            logOut = logCap if logOut > 0.0 else logFloor
        if logOut < logFloor:
            logOut = logFloor
        elif logOut > logCap:
            logOut = logCap
        for i in range(n):
            outPtr[i] = <cnp.float32_t>exp(logOut)
        return out

    work = <double*>malloc((degree + 1) * sizeof(double))
    if work == NULL:
        raise MemoryError("failed to allocate P-spline work buffer")

    try:
        with nogil:
            for i in range(n):
                x = predictorPtr[i]
                if not isfinite(x):
                    logOut = logFloor
                else:
                    if x < xMin:
                        x = xMin
                    elif x > xMax:
                        x = xMax
                    logOut = _deBoorValue(
                        knotsPtr,
                        betaPtr,
                        nBasis,
                        degree,
                        x,
                        work,
                    )
                    if not isfinite(logOut):
                        logOut = logCap if logOut > 0.0 else logFloor

                if logOut < logFloor:
                    logOut = logFloor
                elif logOut > logCap:
                    logOut = logCap
                outPtr[i] = <cnp.float32_t>exp(logOut)
    finally:
        free(work)

    return out


cpdef cEMA(cnp.ndarray x, double alpha):
    cdef Py_ssize_t n
    cdef cnp.ndarray[cnp.float32_t, ndim=1] x1_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out_F32
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x1_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_F64

    if isinstance(x, np.ndarray) and (<cnp.ndarray>x).dtype == np.float32:
        x1_F32 = np.ascontiguousarray(x, dtype=np.float32)
        n = x1_F32.shape[0]
        out_F32 = np.empty(n, dtype=np.float32)
        _cEMA(<const float*>x1_F32.data, <float*>out_F32.data, n, <float>alpha)
        return out_F32

    x1_F64 = np.ascontiguousarray(x, dtype=np.float64)
    n = x1_F64.shape[0]
    out_F64 = np.empty(n, dtype=np.float64)
    _cEMA(<const double*>x1_F64.data, <double*>out_F64.data, n, alpha)
    return out_F64


cdef void _monoLog(
    const real_t* arrPtr,
    real_t* outPtr,
    Py_ssize_t n,
    real_t offset,
    real_t scale,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef real_t xval, u

    for i in range(n):
        xval = arrPtr[i]
        u = xval + offset
        if u <= <real_t>0.0:
            u = offset
        outPtr[i] = scale * <real_t>log(<double>u)


cdef void _logRatio(
    const real_t* treatmentPtr,
    const real_t* controlPtr,
    real_t* outPtr,
    Py_ssize_t n,
    real_t offset,
    real_t scale,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef real_t t, c

    for i in range(n):
        t = treatmentPtr[i] + offset
        c = controlPtr[i] + offset
        if t <= <real_t>0.0:
            t = offset
        if c <= <real_t>0.0:
            c = offset
        outPtr[i] = scale * <real_t>(log(<double>t) - log(<double>c))


cpdef tuple monoFunc(object x, double offset=<double>(1.0), double scale=<double>(1.0)):
    cdef Py_ssize_t n
    cdef double offset_ = offset
    cdef double scale_ = scale
    cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out_F32
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_F64

    if offset_ <= 0.0:
        offset_ = 1.0

    if isinstance(x, np.ndarray) and (<cnp.ndarray>x).dtype == np.float32:
        arr_F32 = np.ascontiguousarray(x, dtype=np.float32)
        n = arr_F32.shape[0]
        out_F32 = np.empty(n, dtype=np.float32)
        with nogil:
            _monoLog(
                <const float*>arr_F32.data,
                <float*>out_F32.data,
                n,
                <float>offset_,
                <float>scale_,
            )
        return (out_F32, -1.0)

    arr_F64 = np.ascontiguousarray(x, dtype=np.float64)
    n = arr_F64.shape[0]
    out_F64 = np.empty(n, dtype=np.float64)
    with nogil:
        _monoLog(
            <const double*>arr_F64.data,
            <double*>out_F64.data,
            n,
            offset_,
            scale_,
        )

    return (out_F64, -1.0)


cpdef object cTransformWithInput(
    object treatment,
    object control,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
):
    r"""Return the log-ratio transform for treatment/control tracks.

    The transform is ``logMult * (log(treatment + logOffset) -
    log(control + logOffset))``. Negative values are retained as depletion
    relative to control.
    """
    cdef Py_ssize_t n
    cdef cnp.ndarray[cnp.float32_t, ndim=1] treat_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] control_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out_F32
    cdef cnp.ndarray[cnp.float64_t, ndim=1] treat_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] control_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_F64

    if (
        isinstance(treatment, np.ndarray)
        and isinstance(control, np.ndarray)
        and (<cnp.ndarray>treatment).dtype == np.float32
        and (<cnp.ndarray>control).dtype == np.float32
    ):
        treat_F32 = np.ascontiguousarray(treatment, dtype=np.float32).reshape(-1)
        control_F32 = np.ascontiguousarray(control, dtype=np.float32).reshape(-1)
        if treat_F32.size != control_F32.size:
            raise ValueError("treatment and control must have the same length")
        n = treat_F32.shape[0]
        out_F32 = np.empty(n, dtype=np.float32)
        return cTransformWithInputInto(
            treat_F32,
            control_F32,
            out_F32,
            logOffset=logOffset,
            logMult=logMult,
        )

    treat_F64 = np.ascontiguousarray(treatment, dtype=np.float64).reshape(-1)
    control_F64 = np.ascontiguousarray(control, dtype=np.float64).reshape(-1)
    if treat_F64.size != control_F64.size:
        raise ValueError("treatment and control must have the same length")
    n = treat_F64.shape[0]
    out_F64 = np.empty(n, dtype=np.float64)
    return cTransformWithInputInto(
        treat_F64,
        control_F64,
        out_F64,
        logOffset=logOffset,
        logMult=logMult,
    )


cpdef object cTransformWithInputInto(
    object treatment,
    object control,
    object out,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
):
    r"""Write the treatment/control log-ratio transform into ``out``."""
    cdef Py_ssize_t n
    cdef double offset_ = logOffset
    cdef double scale_ = logMult
    cdef object outObj = out
    cdef cnp.ndarray[cnp.float32_t, ndim=1] treat_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] control_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out_F32
    cdef cnp.ndarray[cnp.float64_t, ndim=1] treat_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] control_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_F64

    if offset_ <= 0.0:
        offset_ = 1.0

    if not isinstance(outObj, np.ndarray):
        raise TypeError("out must be a NumPy ndarray")
    if outObj.ndim != 1:
        raise ValueError("out must be one-dimensional")
    if not outObj.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")

    if (<cnp.ndarray>outObj).dtype == np.float32:
        treat_F32 = np.ascontiguousarray(treatment, dtype=np.float32).reshape(-1)
        control_F32 = np.ascontiguousarray(control, dtype=np.float32).reshape(-1)
        out_F32 = outObj
        if treat_F32.size != control_F32.size:
            raise ValueError("treatment and control must have the same length")
        if out_F32.size != treat_F32.size:
            raise ValueError("out must have the same length as treatment and control")
        n = treat_F32.shape[0]
        with nogil:
            _logRatio(
                <const float*>treat_F32.data,
                <const float*>control_F32.data,
                <float*>out_F32.data,
                n,
                <float>offset_,
                <float>scale_,
            )
        return out

    if (<cnp.ndarray>outObj).dtype == np.float64:
        treat_F64 = np.ascontiguousarray(treatment, dtype=np.float64).reshape(-1)
        control_F64 = np.ascontiguousarray(control, dtype=np.float64).reshape(-1)
        out_F64 = outObj
        if treat_F64.size != control_F64.size:
            raise ValueError("treatment and control must have the same length")
        if out_F64.size != treat_F64.size:
            raise ValueError("out must have the same length as treatment and control")
        n = treat_F64.shape[0]
        with nogil:
            _logRatio(
                <const double*>treat_F64.data,
                <const double*>control_F64.data,
                <double*>out_F64.data,
                n,
                offset_,
                scale_,
            )
        return out

    raise TypeError("out dtype must be float32 or float64")


cpdef object cTransformInPlace(
    object x,
    bint verbose=<bint>False,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
):
    r"""Log-transform a contiguous coverage track in-place."""
    cdef Py_ssize_t n
    cdef double offset_ = logOffset
    cdef double scale_ = logMult
    cdef object arrObj = x
    cdef cnp.ndarray zArr_F32
    cdef cnp.ndarray zArr_F64

    if offset_ <= 0.0:
        offset_ = 1.0

    if not isinstance(arrObj, np.ndarray):
        raise TypeError("x must be a NumPy ndarray")
    if arrObj.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not arrObj.flags.c_contiguous:
        raise ValueError("x must be C-contiguous")

    if (<cnp.ndarray>arrObj).dtype == np.float32:
        zArr_F32 = arrObj
        n = zArr_F32.shape[0]
        with nogil:
            _monoLog(
                <const float*>zArr_F32.data,
                <float*>zArr_F32.data,
                n,
                <float>offset_,
                <float>scale_,
            )
        return x

    if (<cnp.ndarray>arrObj).dtype != np.float64:
        raise TypeError("x dtype must be float32 or float64")
    zArr_F64 = arrObj
    n = zArr_F64.shape[0]
    with nogil:
        _monoLog(
            <const double*>zArr_F64.data,
            <double*>zArr_F64.data,
            n,
            offset_,
            scale_,
        )

    return x


cpdef object cTransform(
    object x,
    bint verbose=<bint>False,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
):
    r"""Log-transform a coverage track."""
    cdef object outArr

    if isinstance(x, np.ndarray) and (<cnp.ndarray>x).dtype == np.float32:
        outArr = np.array(x, dtype=np.float32, copy=True, order="C").reshape(-1)
    else:
        outArr = np.array(x, dtype=np.float64, copy=True, order="C").reshape(-1)

    return cTransformInPlace(
        outArr,
        verbose=verbose,
        logOffset=logOffset,
        logMult=logMult,
    )


cpdef tuple cforwardPass(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixPluginMuncInit,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQ0,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] intervalToBlockMap,
    Py_ssize_t blockCount,
    float stateInit,
    float stateCovarInit,
    float pad=1.0e-4,
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
    object processPrecExp=None,
    object replicateBias=None,
    bint ECM_useObsPrecisionReweighting=True,
    bint ECM_useProcessPrecisionReweighting=True,
    bint ECM_useAPN=False,
    float obsPrecisionMultiplierMin=0.25,
    float obsPrecisionMultiplierMax=4.0,
    float procPrecisionMultiplierMin=0.25,
    float procPrecisionMultiplierMax=4.0,
    float APN_minQ=1.0e-4,
    float APN_maxQ=1000.0,
    float APN_dStatThresh=5.0,
    float APN_dStatScale=10.0,
    float APN_dStatPC=2.0,
):
    r"""Run the forward pass (filter) for state estimation

    See :func:`consenrich.cconsenrich.cfixedBackgroundECM`, where this routine is applied
    within the filter, smooth, update loop.


    :seealso: :func:`consenrich.cconsenrich.cbackwardPass`,
            :func:`consenrich.cconsenrich.cfixedBackgroundECM`,
            :func:`consenrich.core.runConsenrich`
    """

    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] muncMatView = matrixPluginMuncInit
    cdef cnp.float32_t[:, ::1] fView = matrixF
    cdef cnp.float32_t[:, ::1] q0View = matrixQ0
    cdef cnp.int32_t[::1] blockMapView = intervalToBlockMap
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
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] lambdaExpArr
    cdef cnp.float32_t[::1] lambdaExpView
    cdef bint useLambda = (ECM_useObsPrecisionReweighting and (lambdaExp is not None))
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processPrecExpArr
    cdef cnp.float32_t[::1] processPrecExpView
    cdef bint useProcPrec = (
        ECM_useProcessPrecisionReweighting and (processPrecExp is not None) and (not ECM_useAPN)
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] replicateBiasArr
    cdef cnp.float32_t[::1] replicateBiasView
    cdef bint useReplicateBias = (replicateBias is not None)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] stateVector = np.array([stateInit, 0.0], dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateCovar = (np.eye(2, dtype=np.float32) * np.float32(stateCovarInit))
    cdef cnp.float32_t[::1] stateVectorView = stateVector
    cdef cnp.float32_t[:, ::1] stateCovarView = stateCovar
    cdef float phiHat = 1.0

    # inlining to reduce small matrix indexing cost
    cdef double F00, F01, F10, F11
    cdef double Q00, Q01, Q10, Q11
    cdef double xPred0, xPred1
    cdef double P00, P01, P10, P11
    cdef double PPred00, PPred01, PPred10, PPred11
    cdef double tmp00, tmp01, tmp10, tmp11, rhoMax, offCap
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
    cdef double w
    cdef double wMin = <double>obsPrecisionMultiplierMin
    cdef double wMax = <double>obsPrecisionMultiplierMax
    cdef double procPrec
    cdef double obsPrec
    cdef double procPrecMin = <double>procPrecisionMultiplierMin
    cdef double procPrecMax = <double>procPrecisionMultiplierMax
    cdef double biasJ
    cdef double qDiagBase
    cdef double apnScale = 1.0
    cdef double currentProcNoise
    cdef double adaptiveMult
    cdef double apnMinQ = <double>APN_minQ
    cdef double apnMaxQ = <double>APN_maxQ
    cdef double apnThresh = <double>APN_dStatThresh
    cdef double apnScaleCoef = <double>APN_dStatScale
    cdef double apnPC = <double>APN_dStatPC

    cdef double LOG2PI = log(6.2831853071795864769)

    if useLambda:
        lambdaExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> lambdaExp
        lambdaExpView = lambdaExpArr

    if useProcPrec:
        processPrecExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> processPrecExp
        processPrecExpView = processPrecExpArr

    if useReplicateBias:
        replicateBiasArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> replicateBias
        replicateBiasView = replicateBiasArr

    # Check edge cases here once before loops
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

    if wMin <= 0.0 or wMax <= 0.0 or wMax < wMin:
        raise ValueError("observation precision multiplier bounds must satisfy 0 < min <= max")

    if procPrecMin <= 0.0 or procPrecMax <= 0.0 or procPrecMax < procPrecMin:
        raise ValueError("process precision multiplier bounds must satisfy 0 < min <= max")

    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")

    if useLambda:
        if lambdaExpArr.shape[0] != intervalCount:
            raise ValueError("lambdaExp length must match intervalCount")

    if useProcPrec:
        if processPrecExpArr.shape[0] != intervalCount:
            raise ValueError("processPrecExp length must match intervalCount")

    if useReplicateBias:
        if replicateBiasArr.shape[0] != trackCount:
            raise ValueError("replicateBias length must match trackCount")

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
    qDiagBase = 0.5 * ((<double>q0View[0, 0]) + (<double>q0View[1, 1]))
    if qDiagBase <= 1.0e-12:
        ECM_useAPN = False

    for k in range(intervalCount):
        blockId = <Py_ssize_t>blockMapView[k]
        if blockId < 0 or blockId >= blockCount:
            raise ValueError("intervalToBlockMap has out-of-range block id")

        # --------------------------------------------------------
        # Robust _process_ precision multiplier at _interval_ k
        # --------------------------------------------------------
        if useProcPrec:
            procPrec = <double>processPrecExpView[k]
            if procPrec < procPrecMin:
                procPrec = procPrecMin
            elif procPrec > procPrecMax:
                procPrec = procPrecMax
        else:
            procPrec = 1.0

        # ========================================================
        # Predict step transition
        # ========================================================
        xPred0 = F00*(<double>stateVectorView[0]) + F01*(<double>stateVectorView[1])
        xPred1 = F10*(<double>stateVectorView[0]) + F11*(<double>stateVectorView[1])
        stateVectorView[0] = <cnp.float32_t>xPred0
        stateVectorView[1] = <cnp.float32_t>xPred1

        # Q[k,* , *] = (apnScale[k] / procPrec[k]) * Q0[* , *]
        Q00 = (apnScale / procPrec) * (<double>q0View[0, 0])
        Q01 = (apnScale / procPrec) * (<double>q0View[0, 1])
        Q10 = (apnScale / procPrec) * (<double>q0View[1, 0])
        Q11 = (apnScale / procPrec) * (<double>q0View[1, 1])

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
        # Robust observation precision multiplier via lambda[k].
        # This scales the full diagonal observation covariance at interval k.
        # ========================================================
        if useLambda:
            obsPrec = <double>lambdaExpView[k]
            if obsPrec < wMin:
                obsPrec = wMin
            elif obsPrec > wMax:
                obsPrec = wMax
        else:
            obsPrec = 1.0

        sumInvR = 0.0
        sumInvRInnov = 0.0
        sumInvRInnov2 = 0.0
        if returnNLL:
            sumLogR = 0.0
            intervalNLL = 0.0

        for j in range(trackCount):
            if useReplicateBias:
                biasJ = <double>replicateBiasView[j]
            else:
                biasJ = 0.0

            innov = (<double>dataView[j, k]) - biasJ - (<double>stateVectorView[0])

            baseVar = (<double>muncMatView[j, k]) + (<double>pad)
            measVar = baseVar
            if measVar < 1.0e-12:
                measVar = 1.0e-12

            invMeasVar = obsPrec / measVar

            if returnNLL:
                sumLogR += (log(measVar) - log(obsPrec))

            sumInvRInnov2 += invMeasVar * (innov * innov)
            sumInvRInnov += invMeasVar * innov
            sumInvR += invMeasVar

        innovScale = 1.0 + (<double>stateCovarView[0, 0]) * sumInvR

        gainLike = (<double>stateCovarView[0, 0]) / innovScale
        quadForm = sumInvRInnov2 - gainLike * (sumInvRInnov * sumInvRInnov)
        if quadForm < 0.0:
            quadForm = 0.0

        if returnNLL:
            intervalNLL = 0.5 * (sumLogR + log(innovScale) + quadForm + (<double>trackCount) * LOG2PI)
            sumNLL += intervalNLL

        dStatVector[k] = <cnp.float32_t>(
            intervalNLL if (returnNLL and storeNLLInD) else (quadForm / (<double>trackCount))
        )
        sumDStat += (<double>dStatVector[k])

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

        stateCovarView[0, 0] = <cnp.float32_t>PNew00
        stateCovarView[0, 1] = <cnp.float32_t>PNew01
        stateCovarView[1, 0] = <cnp.float32_t>PNew01
        stateCovarView[1, 1] = <cnp.float32_t>PNew11

        # store for smoothing and ECM logistics
        if doStore:
            stateForwardView[k, 0] = stateVectorView[0]
            stateForwardView[k, 1] = stateVectorView[1]
            stateCovarForwardView[k, 0, 0] = stateCovarView[0, 0]
            stateCovarForwardView[k, 0, 1] = stateCovarView[0, 1]
            stateCovarForwardView[k, 1, 0] = stateCovarView[1, 0]
            stateCovarForwardView[k, 1, 1] = stateCovarView[1, 1]
            # RTS expects pNoiseForward[t] to correspond to transition t -> t+1,
            # so store into (k-1).
            if k > 0:
                pNoiseForwardView[k - 1, 0, 0] = <cnp.float32_t>Q00
                pNoiseForwardView[k - 1, 0, 1] = <cnp.float32_t>Q01
                pNoiseForwardView[k - 1, 1, 0] = <cnp.float32_t>Q10
                pNoiseForwardView[k - 1, 1, 1] = <cnp.float32_t>Q11

        if ECM_useAPN:
            currentProcNoise = 0.5 * (Q00 + Q11)
            if dStatVector[k] > apnThresh and currentProcNoise < apnMaxQ:
                adaptiveMult = sqrt(
                    apnScaleCoef * ((<double>dStatVector[k]) - apnThresh) + apnPC
                )
                apnScale *= adaptiveMult
            elif dStatVector[k] <= apnThresh and currentProcNoise > apnMinQ:
                adaptiveMult = 1.0 / sqrt(
                    apnScaleCoef * (apnThresh - (<double>dStatVector[k])) + apnPC
                )
                apnScale *= adaptiveMult

            currentProcNoise = apnScale * qDiagBase
            if currentProcNoise < apnMinQ:
                apnScale = apnMinQ / qDiagBase
            elif currentProcNoise > apnMaxQ:
                apnScale = apnMaxQ / qDiagBase

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
    Py_ssize_t chunkSize=1000000,
    object stateSmoothed=None,
    object stateCovarSmoothed=None,
    object lagCovSmoothed=None,
    object postFitResiduals=None,
    object replicateBias=None,
    object progressBar=None,
    Py_ssize_t progressIter=10000,
):
    r"""Run the backward pass (smoother)

    This function executes the smoothing phase of Consenrich's forward-backward state estimation. It operates on
    outputs from the *forward-filtered* outputs (those returned by :func:`consenrich.cconsenrich.cforwardPass`).

    That is, given the forward-pass, filtered estimates over genomic intervals :math:`i = 1, \dots, n`,

    .. math::

        \mathbf{x}_{[i|i]}, \qquad \mathbf{P}_{[i|i]}, \qquad \mathbf{Q}_{[i]},

    this routine computes the *backward-smoothed* state estimates :math:`\widetilde{\mathbf{x}}_{[i]}`
    and the *backward-smoothed* covariances :math:`\widetilde{\mathbf{P}}_{[i]}`.


    :seealso: :func:`consenrich.cconsenrich.cforwardPass`,
            :func:`consenrich.cconsenrich.cfixedBackgroundECM`,
            :func:`consenrich.core.runConsenrich`

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
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] replicateBiasArr

    cdef cnp.float32_t[:, ::1] stateSmoothedView
    cdef cnp.float32_t[:, :, ::1] stateCovarSmoothedView
    cdef cnp.float32_t[:, :, ::1] lagCovSmoothedView
    cdef cnp.float32_t[:, ::1] postFitResidualsView
    cdef cnp.float32_t[::1] replicateBiasView
    cdef bint useReplicateBias = (replicateBias is not None)
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
    cdef double biasJ

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

    if useReplicateBias:
        replicateBiasArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> replicateBias
        replicateBiasView = replicateBiasArr
        if replicateBiasArr.shape[0] != trackCount:
            raise ValueError("replicateBias length must match trackCount")

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
            if useReplicateBias:
                biasJ = <double>replicateBiasView[j]
            else:
                biasJ = 0.0
            postFitResidualsView[intervalCount - 1, j] = <cnp.float32_t>(
                (<double>dataView[j, intervalCount - 1]) - biasJ - (<double>stateSmoothedView[intervalCount - 1, 0])
            )

        #  `k = intervalCount - 2`...`k=0`
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
                if useReplicateBias:
                    biasJ = <double>replicateBiasView[j]
                else:
                    biasJ = 0.0
                innov = (<double>dataView[j, k]) - biasJ - (<double>stateSmoothedView[k, 0])
                postFitResidualsView[k, j] = <cnp.float32_t>innov

    return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)


cpdef tuple cfixedBackgroundECM(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixPluginMuncInit,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixF,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQ0,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] intervalToBlockMap,
    Py_ssize_t blockCount,
    float stateInit,
    float stateCovarInit,
    Py_ssize_t ECM_fixedBackgroundIters=50,
    float ECM_fixedBackgroundRtol=1.0e-4,
    float pad=1.0e-4,
    float ECM_robustTNu=8.0,
    float obsPrecisionMultiplierMin=0.25,
    float obsPrecisionMultiplierMax=4.0,
    float procPrecisionMultiplierMin=0.25,
    float procPrecisionMultiplierMax=4.0,
    bint ECM_useObsPrecisionReweighting=True,
    bint ECM_useProcessPrecisionReweighting=True,
    bint ECM_useAPN=False,
    float APN_minQ=1.0e-4,
    float APN_maxQ=1000.0,
    float APN_dStatThresh=5.0,
    float APN_dStatScale=10.0,
    float APN_dStatPC=2.0,
    Py_ssize_t t_innerIters=3,
    bint returnIntermediates=False,
    bint returnDiagnostics=False,
    object lambdaExpInit=None,
    object processPrecExpInit=None,
    object replicateBiasInit=None,
):
    r"""Run the fixed-background Consenrich ECM loop with iteratively updated observation and process noise covariances.

    This routine is the fixed-background fit used by
    :func:`consenrich.core.runConsenrich`. Any shared interval background has
    already been removed from ``matrixData`` before this step.

    Take observation and process noise [co]variances:

    .. math::

        \widetilde{R}_{[i]}=\frac{1}{\lambda_{[i]}}
          \operatorname{diag}(v_{[1,i]},\ldots,v_{[m,i]}),
        \qquad
        \widetilde{\mathbf{Q}}_{[i]}=\frac{\mathbf{Q}_0}{\kappa_{[i]}}.

    We also include replicate-level additive offsets in the observation mean:

    .. math::

        z_{[j,i]} = x_{[i,0]} + b_j + \epsilon_{[j,i]}.

    Here :math:`b_j` is a replicate-level offset, and
    :math:`\lambda_{[i]}` and :math:`\kappa_{[i]}` are Student-t precision multipliers.


    Estimation loop
    ---------------

    Repeat until convergence:

    #. **Filter-Smoother estimation**

    Run the forward filter and backward smoother under the current (given)
    effective noises :math:`\widetilde{R}` and :math:`\widetilde{\mathbf{Q}}`. This yields smoothed moments
    :math:`\widetilde{\mathbf{x}}_{[i]}`, :math:`\widetilde{\mathbf{P}}_{[i]}`, and lag-one covariances
    :math:`\widetilde{\mathbf{C}}_{[i,i+1]}`.


    #. **Studentized precision reweighting**:

    *Observation weights* :math:`\lambda_{[i]}` (``ECM_useObsPrecisionReweighting``):

    .. math::

        u^2_{[i]}=\sum_{j=1}^m
          \frac{(z_{[j,i]}-b_j-\widetilde{x}_{[i,0]})^2+\widetilde{P}_{[i,0,0]}}
               {v_{[j,i]}+\mathrm{pad}}
        \quad\Rightarrow\quad
        \lambda_{[i]} \leftarrow \frac{\nu_R+m}{\nu_R+u^2_{[i]}}.

    In code, ``ECM_robustTNu`` corresponds to :math:`\nu_R`.

    *Process weights* :math:`\kappa_{[i]}`:

    Let :math:`\mathbf{w}_{[i]}=\mathbf{x}_{[i]}-\mathbf{F}\mathbf{x}_{[i-1]}` and define

    .. math::

        \Delta_{[i]}=\textsf{Trace}\!\left(\mathbf{Q}_0^{-1}\,\mathbb{E}\left[\mathbf{w}_{[i]}\mathbf{w}_{[i]}^\top\right]\right).

    Then

    .. math::

        \kappa_{[i]} \leftarrow \frac{\nu_Q+d}{\nu_Q+\Delta_{[i]}},

    where :math:`d=2`.

    3. **Replicate-level updates**: update :math:`b_j` from weighted residual moments at
    replicate resolution while holding block and interval-level terms fixed.


    Objective Function
    ----------------------------------

    Let :math:`x_{1:n}=\{\mathbf{x}_{[i]}\}_{i=1}^n`, :math:`\lambda=\{\lambda_{[i]}\}`, and
    :math:`\kappa=\{\kappa_{[i]}\}`. Collecting process and observation terms and mixing penalties yields:

    .. math::
      :nowrap:

        \begin{align}
        \mathcal{J}(x,\Lambda,\kappa,b)
        &=
        \frac12\sum_{i=2}^{n}
        \left[
        \log\left|\frac{1}{\kappa_{[i]}}\mathbf{Q}_0\right|
        +
        (\mathbf{x}_{[i]}-\mathbf{F}\mathbf{x}_{[i-1]})^\top
        \left(\kappa_{[i]}\mathbf{Q}_0^{-1}\right)
        (\mathbf{x}_{[i]}-\mathbf{F}\mathbf{x}_{[i-1]})
        \right] \\
        &\quad+
        \frac12\sum_{i=1}^{n}\sum_{j=1}^m
        \left[
        \log\!\left(\frac{v_{[j,i]}}{\lambda_{[i]}}\right)
        +
        (z_{[j,i]}-b_j-x_{[i,0]})^2\,\frac{\lambda_{[i]}}{v_{[j,i]}}
        \right] \\
        &\quad+
        \sum_{i=1}^{n}
        \left[
        -\frac{\nu_R}{2}\log\lambda_{[i]}
        +\frac{\nu_R}{2}\lambda_{[i]}
        \right] \\
        &\quad+
        \sum_{i=2}^{n}
        \left[
        -\left(\frac{\nu_Q+d}{2}-1\right)\log\kappa_{[i]}
        +\frac{\nu_Q+d}{2}\kappa_{[i]}
        \right].
        \end{align}


    So the estimation loop maximizing our objective function may be viewed as a coordinate ascent where the filter-smoother
    solves the quadratic subproblem *conditional* on the current estimates of :math:`\lambda`, :math:`\kappa`, and :math:`b`,
    and reweighting plus offset updates optimize over :math:`\lambda`, :math:`\kappa`, and :math:`b`.

    :param matrixData: Replicate observed track values :math:`z_{[j,i]}` (rows:
        replicates, columns: genomic intervals).
    :type matrixData: numpy.ndarray[numpy.float32]
    :param matrixPluginMuncInit: Data-derived observation noise variances :math:`v_{[j,i]}`. Same per-replicate/per-interval shape as ``matrixData``.
    :type matrixPluginMuncInit: numpy.ndarray[numpy.float32]
    :param matrixF: Transition matrix :math:`\mathbf{F}`, shape ``(2, 2)``.
    :type matrixF: numpy.ndarray[numpy.float32]
    :param matrixQ0: Base process noise covariance: :math:`\mathbf{Q}_0 \in \mathbb{R}^{2 \times 2}`
    :type matrixQ0: numpy.ndarray[numpy.float32]
    :param intervalToBlockMap: Mapping from interval index :math:`i` to block index :math:`b(i)`
    :type intervalToBlockMap: numpy.ndarray[numpy.int32]
    :param blockCount: Number of interval blocks.
    :type blockCount: int
    :param stateInit: Initial state value for the signal-level (first component) of the state vector :math:`\mathbf{x}_{[0]}`
    :type stateInit: float
    :param stateCovarInit: Initial state covariance scale
    :type stateCovarInit: float
    :param ECM_fixedBackgroundIters: Maximum fixed-background ECM iterations.
    :type ECM_fixedBackgroundIters: int
    :param ECM_fixedBackgroundRtol: Relative tolerance used for the inner NLL stabilization test.
        The inner loop is considered stable when
        ``abs(NLL_k - NLL_{k-1}) <= ECM_fixedBackgroundRtol * max(abs(NLL_k), abs(NLL_{k-1}), 1)``
        for two consecutive iterations.
    :type ECM_fixedBackgroundRtol: float
    :param ECM_robustTNu: Student-t df for reweighting strengths (smaller = stronger reweighting)
    :type ECM_robustTNu: float
    :param obsPrecisionMultiplierMin: Lower clamp for observation precision multipliers :math:`\lambda_{[i]}`.
    :type obsPrecisionMultiplierMin: float
    :param obsPrecisionMultiplierMax: Upper clamp for observation precision multipliers :math:`\lambda_{[i]}`.
    :type obsPrecisionMultiplierMax: float
    :param procPrecisionMultiplierMin: Lower clamp for process precision multipliers :math:`\kappa_{[i]}`.
    :type procPrecisionMultiplierMin: float
    :param procPrecisionMultiplierMax: Upper clamp for process precision multipliers :math:`\kappa_{[i]}`.
    :type procPrecisionMultiplierMax: float
    :param ECM_useObsPrecisionReweighting: If True, update observation precision multipliers :math:`\lambda_{[i]}` (Student-t reweighting); otherwise :math:`\lambda\equiv 1`.
    :type ECM_useObsPrecisionReweighting: bool
    :param ECM_useProcessPrecisionReweighting: If True, update process precision multipliers :math:`\kappa_{[i]}` (Student-t reweighting); otherwise :math:`\kappa\equiv 1`.
    :type ECM_useProcessPrecisionReweighting: bool
    :param t_innerIters: Number of filter/smoother + reweighting updates per ECM iteration.
    :type t_innerIters: int
    :param returnIntermediates: If True, also return smoothed states/covariances, residuals, and (if enabled) precision multipliers.
    :type returnIntermediates: bool
    :param returnDiagnostics: If True, append a dictionary with iteration,
        convergence, and NLL-change diagnostics to the returned tuple.
    :type returnDiagnostics: bool
    :param lambdaExpInit: Optional warm-start observation precision multipliers.
        If supplied and observation reweighting is enabled, length must match
        the number of intervals.
    :type lambdaExpInit: numpy.ndarray | None
    :param processPrecExpInit: Optional warm-start process precision multipliers.
        If supplied and process reweighting is enabled, length must match the
        number of intervals.
    :type processPrecExpInit: numpy.ndarray | None
    :param replicateBiasInit: Optional warm-start replicate offsets.
    :type replicateBiasInit: numpy.ndarray | None

    :returns: A tuple ``(itersDone, finalNLL)``. If
            ``returnIntermediates=True``, additionally returns
            ``(stateSmoothed, stateCovarSmoothed, lagCovSmoothed,
            postFitResiduals, lambdaExp, processPrecExp, replicateBias)``.
            If ``returnDiagnostics=True``, a diagnostics dictionary is appended.
    :rtype: tuple


    References
    ----------

    * Shumway, R. H. & Stoffer, D. S. (1982): *An approach to time series smoothing and forecasting using the EM algorithm*. DOI: ``10.1111/j.1467-9892.1982.tb00349.x``

    * West, M. (1987): *On scale mixtures of normal distributions*. DOI: ``10.1093/biomet/74.3.646``

    See Also
    --------

    :func:`consenrich.cconsenrich.cforwardPass`
    :func:`consenrich.cconsenrich.cbackwardPass`
    :func:`consenrich.core.runConsenrich`
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

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] replicateBiasArr
    if replicateBiasInit is None:
        replicateBiasArr = np.zeros(trackCount, dtype=np.float32)
    else:
        replicateBiasArr = np.array(replicateBiasInit, dtype=np.float32, copy=True, order="C").reshape(-1)
        if replicateBiasArr.shape[0] != trackCount:
            raise ValueError("replicateBiasInit length must match trackCount")
        if not np.all(np.isfinite(replicateBiasArr)):
            raise ValueError("replicateBiasInit must contain only finite values")
    cdef cnp.float32_t[::1] replicateBiasView = replicateBiasArr

    # Allocate latent precision multipliers only if enabled
    cdef object lambdaExp = None
    cdef object processPrecExp = None
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] lambdaExpArr
    cdef cnp.float32_t[::1] lambdaExpView
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processPrecExpArr
    cdef cnp.float32_t[::1] processPrecExpView

    if ECM_useObsPrecisionReweighting:
        if lambdaExpInit is None:
            lambdaExpArr = np.ones(intervalCount, dtype=np.float32)
        else:
            lambdaExpArr = np.array(lambdaExpInit, dtype=np.float32, copy=True, order="C")
            if lambdaExpArr.shape[0] != intervalCount:
                raise ValueError("lambdaExpInit length must match intervalCount")
            if not np.all(np.isfinite(lambdaExpArr)):
                raise ValueError("lambdaExpInit must contain only finite values")
            np.clip(lambdaExpArr, obsPrecisionMultiplierMin, obsPrecisionMultiplierMax, out=lambdaExpArr)
        lambdaExp = lambdaExpArr
        lambdaExpView = lambdaExpArr

    if ECM_useProcessPrecisionReweighting and (not ECM_useAPN):
        if processPrecExpInit is None:
            processPrecExpArr = np.ones(intervalCount, dtype=np.float32)
        else:
            processPrecExpArr = np.array(processPrecExpInit, dtype=np.float32, copy=True, order="C").reshape(-1)
            if processPrecExpArr.shape[0] != intervalCount:
                raise ValueError("processPrecExpInit length must match intervalCount")
            if not np.all(np.isfinite(processPrecExpArr)):
                raise ValueError("processPrecExpInit must contain only finite values")
            np.clip(processPrecExpArr, procPrecisionMultiplierMin, procPrecisionMultiplierMax, out=processPrecExpArr)
        processPrecExp = processPrecExpArr
        processPrecExpView = processPrecExpArr

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForward = np.empty((intervalCount, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForward = np.empty((intervalCount, 2, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForward = np.empty((intervalCount, 2, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateSmoothed = np.empty((intervalCount, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarSmoothed = np.empty((intervalCount, 2, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] lagCovSmoothed = np.empty((max(intervalCount - 1, 1), 2, 2), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] postFitResiduals = np.empty((intervalCount, trackCount), dtype=np.float32)
    cdef cnp.float32_t[:, ::1] stateSmoothedView = stateSmoothed
    cdef cnp.float32_t[:, :, ::1] stateCovarSmoothedView = stateCovarSmoothed
    cdef cnp.float32_t[:, :, ::1] lagCovSmoothedView = lagCovSmoothed
    cdef cnp.float32_t[:, ::1] residualView = postFitResiduals

    cdef double f00 = <double>fView[0, 0]
    cdef double f01 = <double>fView[0, 1]
    cdef double f10 = <double>fView[1, 0]
    cdef double f11 = <double>fView[1, 1]
    cdef double q0_00 = <double>q0View[0, 0]
    cdef double q0_01 = <double>q0View[0, 1]
    cdef double q0_10 = <double>q0View[1, 0]
    cdef double q0_11 = <double>q0View[1, 1]
    cdef double detQ0 = (q0_00*q0_11 - q0_01*q0_10)
    cdef double q0Inv00
    cdef double q0Inv01
    cdef double q0Inv10
    cdef double q0Inv11
    cdef MAT2 F
    cdef MAT2 Ft
    cdef MAT2 Q0inv
    cdef double previousNLL = 1.0e16
    cdef double currentNLL = 0.0
    cdef double initialNLL = 0.0
    cdef double nllDelta = 0.0
    cdef double nllScale = 1.0
    cdef double nllTol = 0.0
    cdef double relImprovement = 0.0
    cdef double absRelChange = 0.0
    cdef Py_ssize_t itersDone = 0
    cdef Py_ssize_t nllIncreaseCount = 0
    cdef bint hasInitialNLL = False
    cdef bint converged = False
    cdef double res
    cdef double muncPlusPad
    cdef double p00k
    cdef double Rkj
    cdef double x0, x1, y0, y1
    cdef MAT2 Pk, Pk1, Ck_k1
    cdef MAT2 expec_xx, expec_yy, expec_xy, expec_yx, expec_ww
    cdef double delta
    cdef double u2
    cdef double w
    cdef double obsU2
    cdef double wMin = <double>obsPrecisionMultiplierMin
    cdef double wMax = <double>obsPrecisionMultiplierMax
    cdef double kappa_
    cdef double kappaMin_ = <double>procPrecisionMultiplierMin
    cdef double kappaMax_ = <double>procPrecisionMultiplierMax
    cdef double dState = 2.0
    cdef double tmpVal
    cdef double procNu = ECM_robustTNu
    cdef cnp.ndarray[cnp.float64_t, ndim=1] repBiasNum = np.zeros(trackCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] repBiasDen = np.zeros(trackCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] repBiasCenterWeight = np.zeros(trackCount, dtype=np.float64)
    cdef double[::1] repBiasNumView = repBiasNum
    cdef double[::1] repBiasDenView = repBiasDen
    cdef double[::1] repBiasCenterWeightView = repBiasCenterWeight
    cdef Py_ssize_t stableIters = 0
    cdef Py_ssize_t patienceTarget = 2
    cdef double repBiasAlpha
    cdef double repBiasProjectionNum
    cdef double repBiasProjectionDen
    cdef double repBiasCenterWeightJ
    cdef double invVar
    cdef double denomNoRep
    cdef double yMinusState

    if intervalCount <= 5:
        diagnostics = {
            "iters_done": int(0),
            "max_iters": int(ECM_fixedBackgroundIters),
            "converged": False,
            "stable_iters": int(0),
            "patience_target": int(patienceTarget),
            "initial_nll": None,
            "final_nll": float(previousNLL),
            "final_abs_rel_change": None,
            "final_rel_improvement": None,
            "nll_increase_count": int(0),
        }
        if returnIntermediates:
            if returnDiagnostics:
                return (
                    0, float(previousNLL),
                    stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                    lambdaExp, processPrecExp, replicateBiasArr, diagnostics
                )
            return (
                0, float(previousNLL),
                stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                lambdaExp, processPrecExp, replicateBiasArr
            )
        if returnDiagnostics:
            return (0, float(previousNLL), diagnostics)
        return (0, float(previousNLL))

    if blockCount <= 0:
        raise ValueError("blockCount must be positive")
    if wMin <= 0.0 or wMax <= 0.0 or wMax < wMin:
        raise ValueError("observation precision multiplier bounds must satisfy 0 < min <= max")
    if kappaMin_ <= 0.0 or kappaMax_ <= 0.0 or kappaMax_ < kappaMin_:
        raise ValueError("process precision multiplier bounds must satisfy 0 < min <= max")
    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")
    if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
        raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
    if detQ0 == 0.0:
        raise ValueError("matrixQ0 is singular")

    q0Inv00 = q0_11 / detQ0
    q0Inv01 = -q0_01 / detQ0
    q0Inv10 = -q0_10 / detQ0
    q0Inv11 = q0_00 / detQ0

    F = MAT2_make(f00, f01, f10, f11)
    Ft = MAT2_transpose(F)
    Q0inv = MAT2_make(q0Inv00, q0Inv01, q0Inv10, q0Inv11)

    with nogil:
        for j in range(trackCount):
            repBiasCenterWeightView[j] = 0.0
        for k in range(intervalCount):
            b = <Py_ssize_t>blockMapView[k]
            if b < 0 or b >= blockCount:
                continue
            for j in range(trackCount):
                muncPlusPad = (<double>muncMatView[j, k]) + (<double>pad)
                if muncPlusPad < 1.0e-12:
                    muncPlusPad = 1.0e-12
                repBiasCenterWeightView[j] += 1.0 / muncPlusPad

    for i in range(ECM_fixedBackgroundIters):
        itersDone = i + 1
        fprintf(stderr, "\n\t[cfixedBackgroundECM] iter=%zd\n", itersDone)

        for inner in range(t_innerIters):
            cforwardPass(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixPluginMuncInit,
                matrixF=matrixF,
                matrixQ0=matrixQ0,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=blockCount,
                stateInit=stateInit,
                stateCovarInit=stateCovarInit,
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
                processPrecExp=processPrecExp,
                replicateBias=replicateBiasArr,
                ECM_useObsPrecisionReweighting=ECM_useObsPrecisionReweighting,
                ECM_useProcessPrecisionReweighting=ECM_useProcessPrecisionReweighting,
                ECM_useAPN=ECM_useAPN,
                obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
                obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
                procPrecisionMultiplierMin=procPrecisionMultiplierMin,
                procPrecisionMultiplierMax=procPrecisionMultiplierMax,
                APN_minQ=APN_minQ,
                APN_maxQ=APN_maxQ,
                APN_dStatThresh=APN_dStatThresh,
                APN_dStatScale=APN_dStatScale,
                APN_dStatPC=APN_dStatPC,
            )

            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals = cbackwardPass(
                matrixData=matrixData,
                matrixF=matrixF,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                chunkSize=0,
                stateSmoothed=stateSmoothed,
                stateCovarSmoothed=stateCovarSmoothed,
                lagCovSmoothed=lagCovSmoothed,
                postFitResiduals=postFitResiduals,
                replicateBias=replicateBiasArr,
                progressBar=None,
                progressIter=0,
            )

            # -----------------------------
            # E-step: update interval-level lambdaExp (optional)
            # -----------------------------
            if ECM_useObsPrecisionReweighting:
                with nogil:
                    for k in range(intervalCount):
                        b = <Py_ssize_t>blockMapView[k]
                        if b < 0 or b >= blockCount:
                            lambdaExpView[k] = <cnp.float32_t>1.0
                            continue

                        p00k = <double>stateCovarSmoothedView[k, 0, 0]
                        if p00k < 0.0:
                            p00k = 0.0

                        obsU2 = 0.0
                        for j in range(trackCount):
                            muncPlusPad = (<double>muncMatView[j, k]) + (<double>pad)
                            if muncPlusPad < 1.0e-12:
                                muncPlusPad = 1.0e-12
                            Rkj = muncPlusPad

                            res = (<double>dataView[j, k]) - (<double>replicateBiasView[j]) - (<double>stateSmoothedView[k, 0])
                            tmpVal = (res*res + p00k)
                            obsU2 += tmpVal / Rkj

                        w = ((<double>ECM_robustTNu) + (<double>trackCount)) / ((<double>ECM_robustTNu) + obsU2)
                        if w < wMin:
                            w = wMin
                        elif w > wMax:
                            w = wMax

                        lambdaExpView[k] = <cnp.float32_t>w

            # -----------------------------
            # update process precision multipliers kappa_ and store in processPrecExp
            # -----------------------------
            if ECM_useProcessPrecisionReweighting and (not ECM_useAPN):
                processPrecExpView[0] = <cnp.float32_t>1.0
                for k in range(intervalCount - 1):
                    b = <Py_ssize_t>blockMapView[k]
                    if b < 0 or b >= blockCount:
                        processPrecExpView[k + 1] = <cnp.float32_t>1.0
                        continue

                    x0 = <double>stateSmoothedView[k, 0]
                    x1 = <double>stateSmoothedView[k, 1]
                    y0 = <double>stateSmoothedView[k + 1, 0]
                    y1 = <double>stateSmoothedView[k + 1, 1]

                    Pk = MAT2_make(
                        <double>stateCovarSmoothedView[k, 0, 0],
                        <double>stateCovarSmoothedView[k, 0, 1],
                        <double>stateCovarSmoothedView[k, 1, 0],
                        <double>stateCovarSmoothedView[k, 1, 1],
                    )

                    Pk1 = MAT2_make(
                        <double>stateCovarSmoothedView[k + 1, 0, 0],
                        <double>stateCovarSmoothedView[k + 1, 0, 1],
                        <double>stateCovarSmoothedView[k + 1, 1, 0],
                        <double>stateCovarSmoothedView[k + 1, 1, 1],
                    )

                    Ck_k1 = MAT2_make(
                        <double>lagCovSmoothedView[k, 0, 0],
                        <double>lagCovSmoothedView[k, 0, 1],
                        <double>lagCovSmoothedView[k, 1, 0],
                        <double>lagCovSmoothedView[k, 1, 1],
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
                    delta = MAT2_traceProd(Q0inv, expec_ww)
                    if delta < 0.0:
                        delta = 0.0

                    kappa_ = ((<double>procNu) + dState) / ((<double>procNu) + delta)
                    if kappa_ < kappaMin_:
                        kappa_ = kappaMin_
                    elif kappa_ > kappaMax_:
                        kappa_ = kappaMax_
                    processPrecExpView[k + 1] = <cnp.float32_t>kappa_

        with nogil:
            # -----------------------------
            # sufficient stats for replicate-level bias
            #   z[j,k] = x[k] + bias[j] + e[j,k]
            #   Var[e[j,k]] = (munc[j,k] + pad) / lambda[k]
            # -----------------------------
            for j in range(trackCount):
                repBiasNumView[j] = 0.0
                repBiasDenView[j] = 0.0
            for k in range(intervalCount):
                b = <Py_ssize_t>blockMapView[k]
                if b < 0 or b >= blockCount:
                    continue

                for j in range(trackCount):
                    muncPlusPad = (<double>muncMatView[j, k]) + (<double>pad)
                    if muncPlusPad < 1.0e-12:
                        muncPlusPad = 1.0e-12

                    denomNoRep = muncPlusPad
                    if denomNoRep < 1.0e-12:
                        denomNoRep = 1.0e-12

                    if ECM_useObsPrecisionReweighting:
                        w = <double>lambdaExpView[k]
                        if w < wMin:
                            w = wMin
                        elif w > wMax:
                            w = wMax
                    else:
                        w = 1.0

                    invVar = w / denomNoRep
                    yMinusState = (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                    repBiasNumView[j] += invVar * yMinusState
                    repBiasDenView[j] += invVar

            repBiasProjectionNum = 0.0
            repBiasProjectionDen = 0.0
            for j in range(trackCount):
                if repBiasDenView[j] > 0.0:
                    tmpVal = repBiasNumView[j] / repBiasDenView[j]
                    replicateBiasView[j] = <cnp.float32_t>tmpVal
                    repBiasCenterWeightJ = repBiasCenterWeightView[j]
                    repBiasProjectionNum += repBiasCenterWeightJ * tmpVal
                    repBiasProjectionDen += (
                        repBiasCenterWeightJ * repBiasCenterWeightJ
                    ) / repBiasDenView[j]
                else:
                    replicateBiasView[j] = <cnp.float32_t>0.0

            if repBiasProjectionDen > 0.0:
                repBiasAlpha = repBiasProjectionNum / repBiasProjectionDen
            else:
                repBiasAlpha = 0.0

            for j in range(trackCount):
                tmpVal = <double>replicateBiasView[j]
                if repBiasDenView[j] > 0.0:
                    tmpVal = (
                        tmpVal
                        - repBiasAlpha * repBiasCenterWeightView[j] / repBiasDenView[j]
                    )
                replicateBiasView[j] = <cnp.float32_t>tmpVal

        currentNLL = (<double>cforwardPass(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixPluginMuncInit,
            matrixF=matrixF,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=blockCount,
            stateInit=stateInit,
            stateCovarInit=stateCovarInit,
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
            processPrecExp=processPrecExp,
            replicateBias=replicateBiasArr,
            ECM_useObsPrecisionReweighting=ECM_useObsPrecisionReweighting,
            ECM_useProcessPrecisionReweighting=ECM_useProcessPrecisionReweighting,
            ECM_useAPN=ECM_useAPN,
            obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
            obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
            procPrecisionMultiplierMin=procPrecisionMultiplierMin,
            procPrecisionMultiplierMax=procPrecisionMultiplierMax,
            APN_minQ=APN_minQ,
            APN_maxQ=APN_maxQ,
            APN_dStatThresh=APN_dStatThresh,
            APN_dStatScale=APN_dStatScale,
            APN_dStatPC=APN_dStatPC,
        )[3])

        if not hasInitialNLL:
            initialNLL = currentNLL
            hasInitialNLL = True
        elif currentNLL > previousNLL + (1.0e-12 * fmax(fabs(previousNLL), 1.0)):
            nllIncreaseCount += 1

        nllDelta = fabs(currentNLL - previousNLL)
        nllScale = fabs(previousNLL)
        if fabs(currentNLL) > nllScale:
            nllScale = fabs(currentNLL)
        if nllScale < 1.0:
            nllScale = 1.0
        relImprovement = (previousNLL - currentNLL) / nllScale
        absRelChange = nllDelta / nllScale
        nllTol = (<double>ECM_fixedBackgroundRtol) * nllScale
        previousNLL = currentNLL
        fprintf(
            stderr,
            "\t[cfixedBackgroundECM] NLL=%.6f  REL=%+.6e  ABSREL=%.6e  THRESH=%.6e\n",
            currentNLL,
            relImprovement,
            absRelChange,
            nllTol,
        )

        if nllDelta <= nllTol:
            stableIters += 1
        else:
            stableIters = 0

        fprintf(
            stderr,
            "\t[cfixedBackgroundECM] stable=%zd/%zd\n",
            stableIters, patienceTarget
        )

        if stableIters >= patienceTarget:
            converged = True
            fprintf(stderr, "\t[cfixedBackgroundECM] CONVERGED (ECM) iter=%zd \n", itersDone)
            break

    diagnostics = {
        "iters_done": int(itersDone),
        "max_iters": int(ECM_fixedBackgroundIters),
        "converged": bool(converged),
        "stable_iters": int(stableIters),
        "patience_target": int(patienceTarget),
        "initial_nll": float(initialNLL) if hasInitialNLL else None,
        "final_nll": float(previousNLL),
        "final_abs_rel_change": float(absRelChange) if hasInitialNLL else None,
        "final_rel_improvement": float(relImprovement) if hasInitialNLL else None,
        "nll_increase_count": int(nllIncreaseCount),
    }

    if returnIntermediates:
        if returnDiagnostics:
            return (
                itersDone, float(previousNLL),
                stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                lambdaExp, processPrecExp, replicateBiasArr, diagnostics
            )
        return (
            itersDone, float(previousNLL),
            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
            lambdaExp, processPrecExp, replicateBiasArr
        )

    if returnDiagnostics:
        return (itersDone, float(previousNLL), diagnostics)
    return (itersDone, float(previousNLL))


cpdef cnp.ndarray[cnp.float32_t, ndim=1] crolling_AR1_IVar(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    int blockLength,
    cnp.ndarray[cnp.uint8_t, ndim=1] excludeMask,
    double maxBeta=0.99,
    double pairsRegLambda = 1.0,
    bint useInnovationVar = <bint>True,
):
    r"""Estimate a rolling AR(1)-based variance track for a 1D array of values

    If ``useInnovationVar`` is true, returns the one-step AR(1) innovation variance.
    If false, returns the local stationary/marginal AR(1) variance. MUNC uses the
    marginal target because it is consumed as a diagonal observation variance by the
    downstream state-space smoother.
    """

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
    cdef double ScaleX
    cdef double ScaleY
    cdef double denomSym
    cdef double scaleFloorX
    cdef double scaleFloorY
    cdef double blockLengthDouble
    cdef double meanAll
    cdef double gamma0_num
    cdef double gamma1_num
    cdef double scale_
    cdef double gamma0
    cdef double oneMinusBetaSq
    cdef double innoVar

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

        blockLengthDouble = <double>blockLength

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

                meanAll = sumY / blockLengthDouble
                gamma0_num = sumSqY - (blockLengthDouble * meanAll * meanAll)
                if gamma0_num < 0.0:
                    gamma0_num = 0.0

                # Yule--Walker (time-reversal symmetric): beta from lag-1 autocovariance
                gamma1_num = sumLagProd - (meanAll * sumXSeq) - (meanAll * sumYSeq) + (nPairsDouble * meanAll * meanAll)
                lambdaEff = pairsRegLambda / (blockLengthDouble + 1.0)
                scaleFloor = 1.0e-4*(gamma0_num + 1.0)
                scale_ = (gamma0_num * (1.0 + lambdaEff)) + scaleFloor
                eps = 1.0e-12*(gamma0_num + 1.0)
                if scale_ > eps:
                    beta1 = gamma1_num / scale_
                else:
                    beta1 = 0.0

                if beta1 > maxBeta:
                    beta1 = maxBeta
                elif beta1 < 0.0:
                    beta1 = 0.0

                gamma0 = gamma0_num / blockLengthDouble
                oneMinusBetaSq = 1.0 - (beta1 * beta1)
                if oneMinusBetaSq < 0.0:
                    oneMinusBetaSq = 0.0

                if useInnovationVar:
                    innoVar = gamma0 * oneMinusBetaSq
                    if innoVar < 0.0:
                        innoVar = 0.0
                    varAtView[startIndex]=<cnp.float32_t>innoVar
                else:
                    varAtView[startIndex]=<cnp.float32_t>gamma0

            if startIndex < maxStartIndex:
                # slide window forward --> (previousSum - leavingValue) + enteringValue
                sumY = (sumY-valuesView[startIndex]) + (valuesView[(startIndex + blockLength)])
                sumSqY = sumSqY + (-(valuesView[startIndex]*valuesView[startIndex]) + (valuesView[(startIndex + blockLength)]*valuesView[(startIndex + blockLength)]))
                sumLagProd = sumLagProd + (-(valuesView[startIndex]*valuesView[(startIndex + 1)]) + (valuesView[(startIndex + blockLength - 1)]*valuesView[(startIndex + blockLength)]))
                maskSum = maskSum + (-<int>maskView[startIndex] + <int>maskView[(startIndex + blockLength)])

        for regionIndex in range(numIntervals):
            # assign to center of block around regionIndex (as close as possible if near edges)
            startIndex = regionIndex - halfBlockLength
            if startIndex < 0:
                startIndex = 0
            elif startIndex > maxStartIndex:
                startIndex = maxStartIndex
            varOutView[regionIndex] = varAtView[startIndex]

    return varOut

cpdef cnp.ndarray[cnp.float64_t, ndim=1] cSF(
    object chromMat,
    bint centerMedian=<bint>(True),  # FFR: in fact, we use the _MEDIAN_ for centering!, change in next 0.x+1.0 release
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
    # FFR: consider coupling minRefDist with `chooseDependenceLength`
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

        if centerMedian and m > 0:
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


cdef tuple _solvePenalizedChainROCCO_F64(
    double[::1] scoresView,
    double[::1] switchCostsView,
    double selectionPenalty,
):
    cdef Py_ssize_t n = scoresView.shape[0]
    cdef cnp.ndarray[uint8_t, ndim=1] solutionArr
    cdef cnp.ndarray[uint8_t, ndim=1] bt0Arr
    cdef cnp.ndarray[uint8_t, ndim=1] bt1Arr
    cdef uint8_t[::1] solutionView
    cdef uint8_t[::1] bt0View
    cdef uint8_t[::1] bt1View
    cdef Py_ssize_t i
    cdef int state
    cdef double penalty_ = selectionPenalty
    cdef double prev0Val
    cdef double prev1Val
    cdef double stay0Val
    cdef double stay1Val
    cdef double switch0Val
    cdef double switch1Val
    cdef double new0Val
    cdef double new1Val
    cdef double bestVal
    cdef double switchCost
    cdef Py_ssize_t prev0Count
    cdef Py_ssize_t prev1Count
    cdef Py_ssize_t stay0Count
    cdef Py_ssize_t stay1Count
    cdef Py_ssize_t switch0Count
    cdef Py_ssize_t switch1Count
    cdef Py_ssize_t new0Count
    cdef Py_ssize_t new1Count
    cdef Py_ssize_t bestCount
    cdef double selectVal

    if n == 0:
        raise ValueError("`scores` cannot be empty")
    if n > 1 and switchCostsView.shape[0] != n - 1:
        raise ValueError("`switchCosts` must have length len(scores) - 1")
    if n == 1:
        selectVal = scoresView[0] - penalty_
        if selectVal > 0.0:
            return np.asarray([1], dtype=np.uint8), float(selectVal), 1
        return np.asarray([0], dtype=np.uint8), 0.0, 0

    bt0Arr = np.zeros(n, dtype=np.uint8)
    bt1Arr = np.zeros(n, dtype=np.uint8)
    bt0View = bt0Arr
    bt1View = bt1Arr

    prev0Val = 0.0
    prev0Count = 0
    prev1Val = scoresView[0] - penalty_
    prev1Count = 1

    for i in range(1, n):
        switchCost = switchCostsView[i - 1]

        stay0Val = prev0Val
        stay0Count = prev0Count
        switch0Val = prev1Val - switchCost
        switch0Count = prev1Count
        if switch0Val > stay0Val or (
            switch0Val == stay0Val and switch0Count < stay0Count
        ):
            new0Val = switch0Val
            new0Count = switch0Count
            bt0View[i] = <uint8_t>1
        else:
            new0Val = stay0Val
            new0Count = stay0Count
            bt0View[i] = <uint8_t>0

        stay1Val = prev1Val + scoresView[i] - penalty_
        stay1Count = prev1Count + 1
        switch1Val = prev0Val - switchCost + scoresView[i] - penalty_
        switch1Count = prev0Count + 1
        if switch1Val > stay1Val or (
            switch1Val == stay1Val and switch1Count < stay1Count
        ):
            new1Val = switch1Val
            new1Count = switch1Count
            bt1View[i] = <uint8_t>0
        else:
            new1Val = stay1Val
            new1Count = stay1Count
            bt1View[i] = <uint8_t>1

        prev0Val = new0Val
        prev0Count = new0Count
        prev1Val = new1Val
        prev1Count = new1Count

    if prev1Val > prev0Val or (prev1Val == prev0Val and prev1Count < prev0Count):
        bestVal = prev1Val
        bestCount = prev1Count
        state = 1
    else:
        bestVal = prev0Val
        bestCount = prev0Count
        state = 0

    solutionArr = np.zeros(n, dtype=np.uint8)
    solutionView = solutionArr
    solutionView[n - 1] = <uint8_t>state
    for i in range(n - 1, 0, -1):
        if state == 0:
            state = <int>bt0View[i]
        else:
            state = <int>bt1View[i]
        solutionView[i - 1] = <uint8_t>state

    return solutionArr, float(bestVal), int(bestCount)


cpdef tuple csolvePenalizedChainROCCO(
    object scores,
    object switchCosts,
    double selectionPenalty,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scoresArr = np.ascontiguousarray(
        np.asarray(scores, dtype=np.float64).ravel(),
        dtype=np.float64,
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] switchCostsArr = np.ascontiguousarray(
        np.asarray(switchCosts, dtype=np.float64).ravel(),
        dtype=np.float64,
    )
    if scoresArr.size == 0:
        raise ValueError("`scores` cannot be empty")
    if not np.all(np.isfinite(scoresArr)):
        raise ValueError("`scores` contains non-finite values")
    if not np.all(np.isfinite(switchCostsArr)):
        raise ValueError("`switchCosts` contains non-finite values")
    if scoresArr.size > 1 and switchCostsArr.size != scoresArr.size - 1:
        raise ValueError("`switchCosts` must have length len(scores) - 1")
    return _solvePenalizedChainROCCO_F64(scoresArr, switchCostsArr, selectionPenalty)


cdef tuple _calibrateSelectionPenaltyROCCO_F64(
    double[::1] scoresView,
    double[::1] switchCostsView,
    Py_ssize_t targetCount,
    int maxIter,
):
    cdef Py_ssize_t n = scoresView.shape[0]
    cdef Py_ssize_t i
    cdef Py_ssize_t targetCount_
    cdef double scoreMin
    cdef double scoreMax
    cdef double switchSum = 0.0
    cdef double lower
    cdef double upper
    cdef double midpoint
    cdef cnp.ndarray[uint8_t, ndim=1] solutionArr
    cdef cnp.ndarray[uint8_t, ndim=1] bestSolutionArr
    cdef cnp.ndarray[uint8_t, ndim=1] lowerSolutionArr
    cdef double penalizedObjective
    cdef double bestValue
    cdef double lowerValue
    cdef Py_ssize_t selectedCount
    cdef Py_ssize_t bestCount
    cdef Py_ssize_t lowerCount

    if n == 0:
        raise ValueError("`scores` cannot be empty")
    if n > 1 and switchCostsView.shape[0] != n - 1:
        raise ValueError("`switchCosts` must have length len(scores) - 1")

    targetCount_ = targetCount
    if targetCount_ < 0:
        targetCount_ = 0
    elif targetCount_ > n:
        targetCount_ = n

    if targetCount_ == n:
        solutionArr, penalizedObjective, selectedCount = _solvePenalizedChainROCCO_F64(
            scoresView,
            switchCostsView,
            0.0,
        )
        return 0.0, solutionArr, float(penalizedObjective), int(selectedCount)

    scoreMin = scoresView[0]
    scoreMax = scoresView[0]
    for i in range(n):
        if scoresView[i] < scoreMin:
            scoreMin = scoresView[i]
        if scoresView[i] > scoreMax:
            scoreMax = scoresView[i]
        if i < n - 1:
            switchSum += switchCostsView[i]

    lower = scoreMin - switchSum - 1.0
    upper = scoreMax + switchSum + 1.0

    lowerSolutionArr, lowerValue, lowerCount = _solvePenalizedChainROCCO_F64(
        scoresView,
        switchCostsView,
        lower,
    )
    while lowerCount <= targetCount_:
        lower -= fmax(1.0, fabs(lower))
        lowerSolutionArr, lowerValue, lowerCount = _solvePenalizedChainROCCO_F64(
            scoresView,
            switchCostsView,
            lower,
        )

    bestSolutionArr, bestValue, bestCount = _solvePenalizedChainROCCO_F64(
        scoresView,
        switchCostsView,
        upper,
    )
    while bestCount > targetCount_:
        upper += fmax(1.0, fabs(upper))
        bestSolutionArr, bestValue, bestCount = _solvePenalizedChainROCCO_F64(
            scoresView,
            switchCostsView,
            upper,
        )

    for i in range(max(maxIter, 1)):
        midpoint = (lower + upper) / 2.0
        solutionArr, penalizedObjective, selectedCount = _solvePenalizedChainROCCO_F64(
            scoresView,
            switchCostsView,
            midpoint,
        )
        if selectedCount > targetCount_:
            lower = midpoint
            lowerSolutionArr = solutionArr
            lowerValue = penalizedObjective
            lowerCount = selectedCount
        else:
            upper = midpoint
            bestSolutionArr = solutionArr
            bestValue = penalizedObjective
            bestCount = selectedCount

    return float(upper), bestSolutionArr, float(bestValue), int(bestCount)


cpdef tuple ccalibrateSelectionPenaltyROCCO(
    object scores,
    object switchCosts,
    int targetCount,
    int maxIter=60,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scoresArr = np.ascontiguousarray(
        np.asarray(scores, dtype=np.float64).ravel(),
        dtype=np.float64,
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] switchCostsArr = np.ascontiguousarray(
        np.asarray(switchCosts, dtype=np.float64).ravel(),
        dtype=np.float64,
    )
    if scoresArr.size == 0:
        raise ValueError("`scores` cannot be empty")
    if not np.all(np.isfinite(scoresArr)):
        raise ValueError("`scores` contains non-finite values")
    if not np.all(np.isfinite(switchCostsArr)):
        raise ValueError("`switchCosts` contains non-finite values")
    if scoresArr.size > 1 and switchCostsArr.size != scoresArr.size - 1:
        raise ValueError("`switchCosts` must have length len(scores) - 1")
    return _calibrateSelectionPenaltyROCCO_F64(
        scoresArr,
        switchCostsArr,
        <Py_ssize_t>targetCount,
        maxIter,
    )


cpdef tuple csolveChromROCCOExact(
    object scores,
    object budget=None,
    double gamma=0.5,
    object selectionPenalty=None,
    int maxIter=60,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scoresArr = np.ascontiguousarray(
        np.asarray(scores, dtype=np.float64).ravel(),
        dtype=np.float64,
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] switchCostsArr
    cdef double[::1] scoresView
    cdef double[::1] switchCostsView
    cdef cnp.ndarray[uint8_t, ndim=1] solutionArr
    cdef uint8_t[::1] solutionView
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef Py_ssize_t selectedCount
    cdef double objective = 0.0
    cdef double penalizedObjective
    cdef double selectionPenalty_
    cdef double budget_
    cdef Py_ssize_t targetCount

    if scoresArr.size == 0:
        raise ValueError("`scores` cannot be empty")
    if not np.all(np.isfinite(scoresArr)):
        raise ValueError("`scores` contains non-finite values")
    if (not isfinite(gamma)) or gamma < 0.0:
        raise ValueError("`gamma` must be finite and non-negative")

    n = scoresArr.shape[0]
    if n <= 1:
        switchCostsArr = np.zeros(0, dtype=np.float64)
    else:
        switchCostsArr = np.full(n - 1, gamma, dtype=np.float64)
    scoresView = scoresArr
    switchCostsView = switchCostsArr

    if selectionPenalty is None:
        if budget is None:
            selectionPenalty_ = 0.0
            solutionArr, penalizedObjective, selectedCount = _solvePenalizedChainROCCO_F64(
                scoresView,
                switchCostsView,
                selectionPenalty_,
            )
        else:
            budget_ = float(budget)
            if not isfinite(budget_):
                raise ValueError("`budget` must be finite")
            targetCount = <Py_ssize_t>floor(n * budget_)
            selectionPenalty_, solutionArr, penalizedObjective, selectedCount = (
                _calibrateSelectionPenaltyROCCO_F64(
                    scoresView,
                    switchCostsView,
                    targetCount,
                    maxIter,
                )
            )
    else:
        selectionPenalty_ = float(selectionPenalty)
        solutionArr, penalizedObjective, selectedCount = _solvePenalizedChainROCCO_F64(
            scoresView,
            switchCostsView,
            selectionPenalty_,
        )

    solutionView = solutionArr
    for i in range(n):
        objective += scoresView[i] * <double>solutionView[i]
        if i < n - 1 and solutionView[i] != solutionView[i + 1]:
            objective -= switchCostsView[i]

    return (
        solutionArr,
        float(objective),
        float(penalizedObjective),
        int(selectedCount),
        float(selectionPenalty_),
    )
