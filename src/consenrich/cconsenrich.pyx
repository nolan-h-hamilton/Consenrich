# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
# distutils: language = c
r"""Cython module for Consenrich core functions.

This module contains Cython implementations of core functions used in Consenrich.
"""

cimport cython
import os
import numpy as np
from . import misc_util
from scipy import ndimage
cimport numpy as cnp
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from numpy.random import default_rng
from libc.math cimport isfinite, fabs, log1p, log2, log, log2f, logf, asinhf, asinh, fmax, fmaxf, pow, sqrt, sqrtf, fabsf, fminf, fmin, log10, log10f, ceil, floor, floorf, exp, expf, cos, sin, erf, isnan, NAN, INFINITY
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
cdef const double __PI_DOUBLE = <double>3.14159265358979323846264338327950288
cdef const double __MASKED_OBSERVATION_VARIANCE_CUTOFF = <double>5.0e29
cdef const int __TRANSFORM_MODE_LOG = 0
cdef const int __TRANSFORM_MODE_SQRT = 1
cdef const int __TRANSFORM_MODE_ASINH = 2
cdef const int __TRANSFORM_MODE_ASINH_SQRT = 3
cdef const int __TRANSFORM_MODE_GENERALIZED_LOG = 4
cdef const int __TRANSFORM_MODE_IDENTITY = 5
cdef const int __TRANSFORM_MODE_ANSCOMBE = 6
ctypedef fused real_t:
    float
    double

# ===============
# inline/helpers
# ===============

cdef object _coerceProcessQScale(
    object processQScale,
    Py_ssize_t intervalCount,
):
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] scaleArr
    cdef cnp.float32_t[::1] scaleView
    cdef Py_ssize_t scaleLen
    cdef Py_ssize_t i
    cdef Py_ssize_t invalidIndex = -1
    cdef bint firstIsOne = True
    cdef double value

    scaleArr = np.ascontiguousarray(processQScale, dtype=np.float32).reshape(-1)
    scaleLen = scaleArr.shape[0]
    if scaleLen != intervalCount:
        raise ValueError("processQScale length must match intervalCount")

    scaleView = scaleArr
    with nogil:
        for i in range(scaleLen):
            value = <double>scaleView[i]
            if (not isfinite(value)) or value <= 0.0:
                invalidIndex = i
                break
        if scaleLen > 0 and fabs((<double>scaleView[0]) - 1.0) > 1.0e-6:
            firstIsOne = False
    if invalidIndex >= 0:
        raise ValueError("processQScale must contain only positive finite values")
    if not firstIsOne:
        raise ValueError("processQScale[0] must be 1.0")
    return scaleArr



cdef inline double _clampMultiplierValue(double value, double lower, double upper) noexcept nogil:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


cdef inline void _validateMultiplierBounds(
    double lower,
    double upper,
    bint isObservation,
) except *:
    if lower <= 0.0 or upper <= 0.0 or upper < lower:
        if isObservation:
            raise ValueError("observation precision multiplier bounds must satisfy 0 < min <= max")
        raise ValueError("process precision multiplier bounds must satisfy 0 < min <= max")


cdef int _parseTransformMode(object mode) except -1:
    cdef str modeStr

    if mode is None:
        return __TRANSFORM_MODE_LOG

    modeStr = (
        str(mode)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )
    if modeStr == "" or modeStr == "log":
        return __TRANSFORM_MODE_LOG
    if modeStr == "ln" or modeStr == "naturallog":
        return __TRANSFORM_MODE_LOG
    if modeStr == "sqrt" or modeStr == "squareroot":
        return __TRANSFORM_MODE_SQRT
    if modeStr == "anscombe" or modeStr == "anscombetransform":
        return __TRANSFORM_MODE_ANSCOMBE
    if modeStr == "asinh" or modeStr == "arcsinh":
        return __TRANSFORM_MODE_ASINH
    if modeStr == "asinhx" or modeStr == "arcsinhx":
        return __TRANSFORM_MODE_ASINH
    if modeStr == "asinhsqrt" or modeStr == "arcsinhsqrt":
        return __TRANSFORM_MODE_ASINH_SQRT
    if modeStr == "sqrtasinh":
        return __TRANSFORM_MODE_ASINH_SQRT
    if modeStr == "generalizedlog" or modeStr == "generalisedlog":
        return __TRANSFORM_MODE_GENERALIZED_LOG
    if modeStr == "glog" or modeStr == "softlog":
        return __TRANSFORM_MODE_GENERALIZED_LOG
    if modeStr == "identity" or modeStr == "linear" or modeStr == "raw" or modeStr == "none":
        return __TRANSFORM_MODE_IDENTITY
    raise ValueError(
        "mode must be one of 'log', 'sqrt', 'asinh', "
        "'anscombe', 'asinh_sqrt', 'generalized_log', or 'identity'"
    )


cdef double _coerceTransformDouble(object value, double defaultValue, str name) except *:
    cdef double out
    if value is None:
        return defaultValue
    out = float(value)
    if not isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


cdef tuple _resolveTransformParameters(
    int modeCode,
    double logOffset,
    double logMult,
    object offset,
    object scale,
    object inputOffset,
    object inputScale,
    object outputScale,
    object outputOffset,
    object shape,
):
    cdef double defaultInputOffset = 1.0 if modeCode == __TRANSFORM_MODE_LOG else 0.0
    cdef double inputOffset_
    cdef double inputScale_
    cdef double outputScale_
    cdef double outputOffset_
    cdef double shape_

    if modeCode == __TRANSFORM_MODE_ANSCOMBE:
        defaultInputOffset = 0.375

    if inputOffset is None and offset is not None:
        inputOffset = offset
    if inputOffset is None and modeCode == __TRANSFORM_MODE_LOG:
        inputOffset = logOffset

    if outputScale is None and scale is not None:
        outputScale = scale
    if outputScale is None and modeCode == __TRANSFORM_MODE_LOG:
        outputScale = logMult
    if outputScale is None and modeCode == __TRANSFORM_MODE_ANSCOMBE:
        outputScale = 2.0

    inputOffset_ = _coerceTransformDouble(inputOffset, defaultInputOffset, "inputOffset")
    if modeCode == __TRANSFORM_MODE_LOG and inputOffset_ <= 0.0:
        inputOffset_ = 1.0

    inputScale_ = _coerceTransformDouble(inputScale, 1.0, "inputScale")
    if inputScale_ <= 0.0:
        raise ValueError("inputScale must be positive")

    outputScale_ = _coerceTransformDouble(outputScale, 1.0, "outputScale")
    outputOffset_ = _coerceTransformDouble(outputOffset, 0.0, "outputOffset")
    shape_ = _coerceTransformDouble(shape, 1.0, "shape")
    if shape_ <= 0.0:
        raise ValueError("shape must be positive")
    return (inputOffset_, inputScale_, outputScale_, outputOffset_, shape_)


cdef inline void _accumulateObservationValue(
    double observed,
    double stateLevel,
    double baseVariance,
    double pad,
    double obsPrecision,
    bint returnNLL,
    double* sumInvR,
    double* sumInvRInnov,
    double* sumInvRInnov2,
    double* sumLogR,
) noexcept nogil:
    cdef double innov = observed - stateLevel
    cdef double measVar = baseVariance + pad
    cdef double invMeasVar

    if measVar < 1.0e-12:
        measVar = 1.0e-12
    invMeasVar = obsPrecision / measVar
    if returnNLL:
        sumLogR[0] += (log(measVar) - log(obsPrecision))
    sumInvRInnov2[0] += invMeasVar * (innov * innov)
    sumInvRInnov[0] += invMeasVar * innov
    sumInvR[0] += invMeasVar


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


cpdef tuple cExpectedTransitionResidualSumsLevel(
    cnp.ndarray[cnp.float64_t, ndim=2] stateSmoothed,
    cnp.ndarray[cnp.float64_t, ndim=3] stateCovarSmoothed,
    cnp.ndarray[cnp.float64_t, ndim=3] lagCovSmoothed,
):
    cdef Py_ssize_t n = stateSmoothed.shape[0]
    cdef Py_ssize_t transitionCount = n - 1
    cdef Py_ssize_t requiredLagCount = transitionCount if transitionCount > 0 else 0
    cdef Py_ssize_t k
    cdef double x0
    cdef double y0
    cdef double exx0
    cdef double exx1
    cdef double ex0x1
    cdef double levelMoment
    cdef double sumLevel = 0.0

    if stateSmoothed.shape[1] != 1:
        raise ValueError("stateSmoothed must have shape (n, 1)")
    if (
        stateCovarSmoothed.shape[0] != n
        or stateCovarSmoothed.shape[1] != 1
        or stateCovarSmoothed.shape[2] != 1
    ):
        raise ValueError("stateCovarSmoothed must have shape (n, 1, 1)")
    if (
        lagCovSmoothed.shape[0] < requiredLagCount
        or lagCovSmoothed.shape[1] != 1
        or lagCovSmoothed.shape[2] != 1
    ):
        raise ValueError("lagCovSmoothed must have shape (n - 1, 1, 1)")
    if transitionCount <= 0:
        return 0.0, 0.0, 0

    for k in range(transitionCount):
        x0 = stateSmoothed[k, 0]
        y0 = stateSmoothed[k + 1, 0]
        exx0 = stateCovarSmoothed[k, 0, 0] + (x0 * x0)
        exx1 = stateCovarSmoothed[k + 1, 0, 0] + (y0 * y0)
        ex0x1 = lagCovSmoothed[k, 0, 0] + (x0 * y0)
        levelMoment = exx1 - (2.0 * ex0x1) + exx0
        if levelMoment < 0.0:
            levelMoment = 0.0
        sumLevel += levelMoment

    return sumLevel, 0.0, transitionCount


cdef void _expectedTransitionEvidenceLevelLoop(
    double[:, ::1] stateSmoothed,
    double[:, :, ::1] stateCovarSmoothed,
    double[:, :, ::1] lagCovSmoothed,
    double seedQInv,
    double[::1] evidence,
    Py_ssize_t transitionCount,
    Py_ssize_t* finiteCount,
    Py_ssize_t* invalidCount,
    Py_ssize_t* negativeCount,
    double* finiteSum,
    double* finiteMin,
    double* finiteMax,
) noexcept nogil:
    cdef Py_ssize_t k
    cdef Py_ssize_t finiteLocal = 0
    cdef Py_ssize_t invalidLocal = 0
    cdef Py_ssize_t negativeLocal = 0
    cdef double sumLocal = 0.0
    cdef double minLocal = INFINITY
    cdef double maxLocal = -INFINITY
    cdef double x0
    cdef double y0
    cdef double exx0
    cdef double exx1
    cdef double ex0x1
    cdef double moment
    cdef double u

    for k in range(transitionCount):
        x0 = stateSmoothed[k, 0]
        y0 = stateSmoothed[k + 1, 0]
        exx0 = stateCovarSmoothed[k, 0, 0] + (x0 * x0)
        exx1 = stateCovarSmoothed[k + 1, 0, 0] + (y0 * y0)
        ex0x1 = lagCovSmoothed[k, 0, 0] + (x0 * y0)
        moment = exx1 - (2.0 * ex0x1) + exx0
        u = moment * seedQInv
        if not isfinite(u):
            evidence[k] = NAN
            invalidLocal += 1
            continue
        if u < 0.0:
            u = 0.0
            negativeLocal += 1
        evidence[k] = u
        finiteLocal += 1
        sumLocal += u
        if u < minLocal:
            minLocal = u
        if u > maxLocal:
            maxLocal = u

    finiteCount[0] = finiteLocal
    invalidCount[0] = invalidLocal
    negativeCount[0] = negativeLocal
    finiteSum[0] = sumLocal
    finiteMin[0] = minLocal
    finiteMax[0] = maxLocal


cdef void _expectedTransitionEvidenceLevelTrendLoop(
    double[:, ::1] stateSmoothed,
    double[:, :, ::1] stateCovarSmoothed,
    double[:, :, ::1] lagCovSmoothed,
    double[:, ::1] matrixF,
    double qInv00,
    double qInv01,
    double qInv10,
    double qInv11,
    double[::1] evidence,
    Py_ssize_t transitionCount,
    Py_ssize_t* finiteCount,
    Py_ssize_t* invalidCount,
    Py_ssize_t* negativeCount,
    double* finiteSum,
    double* finiteMin,
    double* finiteMax,
) noexcept nogil:
    cdef Py_ssize_t k
    cdef Py_ssize_t finiteLocal = 0
    cdef Py_ssize_t invalidLocal = 0
    cdef Py_ssize_t negativeLocal = 0
    cdef double sumLocal = 0.0
    cdef double minLocal = INFINITY
    cdef double maxLocal = -INFINITY
    cdef double f00 = matrixF[0, 0]
    cdef double f01 = matrixF[0, 1]
    cdef double f10 = matrixF[1, 0]
    cdef double f11 = matrixF[1, 1]
    cdef double x0, x1, y0, y1
    cdef double exx00, exx01, exx10, exx11
    cdef double eyy00, eyy01, eyy10, eyy11
    cdef double exy00, exy01, exy10, exy11
    cdef double eyxFt00, eyxFt01, eyxFt10, eyxFt11
    cdef double fExy00, fExy01, fExy10, fExy11
    cdef double tmp00, tmp01, tmp10, tmp11
    cdef double fExxFt00, fExxFt01, fExxFt10, fExxFt11
    cdef double eww00, eww01, eww10, eww11
    cdef double u

    for k in range(transitionCount):
        x0 = stateSmoothed[k, 0]
        x1 = stateSmoothed[k, 1]
        y0 = stateSmoothed[k + 1, 0]
        y1 = stateSmoothed[k + 1, 1]

        exx00 = stateCovarSmoothed[k, 0, 0] + (x0 * x0)
        exx01 = stateCovarSmoothed[k, 0, 1] + (x0 * x1)
        exx10 = stateCovarSmoothed[k, 1, 0] + (x1 * x0)
        exx11 = stateCovarSmoothed[k, 1, 1] + (x1 * x1)

        eyy00 = stateCovarSmoothed[k + 1, 0, 0] + (y0 * y0)
        eyy01 = stateCovarSmoothed[k + 1, 0, 1] + (y0 * y1)
        eyy10 = stateCovarSmoothed[k + 1, 1, 0] + (y1 * y0)
        eyy11 = stateCovarSmoothed[k + 1, 1, 1] + (y1 * y1)

        # lagCovSmoothed[k] is Cov(x_k, x_{k+1} | y), so exy is E[x_k x_{k+1}^T].
        exy00 = lagCovSmoothed[k, 0, 0] + (x0 * y0)
        exy01 = lagCovSmoothed[k, 0, 1] + (x0 * y1)
        exy10 = lagCovSmoothed[k, 1, 0] + (x1 * y0)
        exy11 = lagCovSmoothed[k, 1, 1] + (x1 * y1)

        eyxFt00 = exy00 * f00 + exy10 * f01
        eyxFt01 = exy00 * f10 + exy10 * f11
        eyxFt10 = exy01 * f00 + exy11 * f01
        eyxFt11 = exy01 * f10 + exy11 * f11

        fExy00 = f00 * exy00 + f01 * exy10
        fExy01 = f00 * exy01 + f01 * exy11
        fExy10 = f10 * exy00 + f11 * exy10
        fExy11 = f10 * exy01 + f11 * exy11

        tmp00 = f00 * exx00 + f01 * exx10
        tmp01 = f00 * exx01 + f01 * exx11
        tmp10 = f10 * exx00 + f11 * exx10
        tmp11 = f10 * exx01 + f11 * exx11

        fExxFt00 = tmp00 * f00 + tmp01 * f01
        fExxFt01 = tmp00 * f10 + tmp01 * f11
        fExxFt10 = tmp10 * f00 + tmp11 * f01
        fExxFt11 = tmp10 * f10 + tmp11 * f11

        eww00 = eyy00 - eyxFt00 - fExy00 + fExxFt00
        eww01 = eyy01 - eyxFt01 - fExy01 + fExxFt01
        eww10 = eyy10 - eyxFt10 - fExy10 + fExxFt10
        eww11 = eyy11 - eyxFt11 - fExy11 + fExxFt11

        u = (
            qInv00 * eww00
            + qInv01 * eww10
            + qInv10 * eww01
            + qInv11 * eww11
        ) * 0.5
        if not isfinite(u):
            evidence[k] = NAN
            invalidLocal += 1
            continue
        if u < 0.0:
            u = 0.0
            negativeLocal += 1
        evidence[k] = u
        finiteLocal += 1
        sumLocal += u
        if u < minLocal:
            minLocal = u
        if u > maxLocal:
            maxLocal = u

    finiteCount[0] = finiteLocal
    invalidCount[0] = invalidLocal
    negativeCount[0] = negativeLocal
    finiteSum[0] = sumLocal
    finiteMin[0] = minLocal
    finiteMax[0] = maxLocal


cpdef tuple cExpectedTransitionProcessEvidence(
    object stateSmoothed,
    object stateCovarSmoothed,
    object lagCovSmoothed,
    object seedQ,
    object matrixF=None,
):
    r"""Return per-transition standardized process evidence for TUNC.

    For transition ``i``, the scalar is
    ``trace(inv(seedQ) * E[w_i w_i.T]) / stateDim``.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] stateArr
    cdef cnp.ndarray[cnp.float64_t, ndim=3, mode="c"] covArr
    cdef cnp.ndarray[cnp.float64_t, ndim=3, mode="c"] lagArr
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] seedQArr
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] fArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] evidenceArr
    cdef double[:, ::1] stateView
    cdef double[:, :, ::1] covView
    cdef double[:, :, ::1] lagView
    cdef double[:, ::1] seedQView
    cdef double[:, ::1] fView
    cdef double[::1] evidenceView
    cdef Py_ssize_t n
    cdef Py_ssize_t transitionCount
    cdef Py_ssize_t requiredLagCount
    cdef Py_ssize_t stateDim
    cdef Py_ssize_t finiteCount = 0
    cdef Py_ssize_t invalidCount = 0
    cdef Py_ssize_t negativeCount = 0
    cdef double finiteSum = 0.0
    cdef double finiteMin = INFINITY
    cdef double finiteMax = -INFINITY
    cdef double q00
    cdef double q01
    cdef double q10
    cdef double q11
    cdef double detQ
    cdef double qInv00
    cdef double qInv01
    cdef double qInv10
    cdef double qInv11
    cdef dict diagnostics

    stateArr = np.ascontiguousarray(stateSmoothed, dtype=np.float64)
    covArr = np.ascontiguousarray(stateCovarSmoothed, dtype=np.float64)
    lagArr = np.ascontiguousarray(lagCovSmoothed, dtype=np.float64)
    seedQArr = np.ascontiguousarray(seedQ, dtype=np.float64)

    if stateArr.ndim != 2:
        raise ValueError("stateSmoothed must be a 2D array")
    n = stateArr.shape[0]
    stateDim = stateArr.shape[1]
    if stateDim != 1 and stateDim != 2:
        raise ValueError("stateSmoothed must have one or two state columns")
    if (
        covArr.shape[0] != n
        or covArr.shape[1] != stateDim
        or covArr.shape[2] != stateDim
    ):
        raise ValueError("stateCovarSmoothed shape must match stateSmoothed state dimension")
    transitionCount = n - 1
    requiredLagCount = transitionCount if transitionCount > 0 else 0
    if (
        lagArr.shape[0] < requiredLagCount
        or lagArr.shape[1] != stateDim
        or lagArr.shape[2] != stateDim
    ):
        raise ValueError("lagCovSmoothed shape must match transition count and state dimension")
    if seedQArr.shape[0] != stateDim or seedQArr.shape[1] != stateDim:
        raise ValueError("seedQ shape must match state dimension")

    evidenceArr = np.empty(requiredLagCount, dtype=np.float64)
    if requiredLagCount <= 0:
        diagnostics = {
            "state_dim": int(stateDim),
            "transition_count": int(0),
            "finite_count": int(0),
            "invalid_count": int(0),
            "negative_clamped_count": int(0),
            "finite_fraction": 1.0,
            "mean": None,
            "min": None,
            "max": None,
        }
        return evidenceArr, diagnostics

    stateView = stateArr
    covView = covArr
    lagView = lagArr
    seedQView = seedQArr
    evidenceView = evidenceArr

    if stateDim == 1:
        q00 = seedQView[0, 0]
        if (not isfinite(q00)) or q00 <= 0.0:
            raise ValueError("seedQ[0, 0] must be positive and finite")
        with nogil:
            _expectedTransitionEvidenceLevelLoop(
                stateView,
                covView,
                lagView,
                1.0 / q00,
                evidenceView,
                transitionCount,
                &finiteCount,
                &invalidCount,
                &negativeCount,
                &finiteSum,
                &finiteMin,
                &finiteMax,
            )
    else:
        if matrixF is None:
            raise ValueError("matrixF is required for two-state process evidence")
        fArr = np.ascontiguousarray(matrixF, dtype=np.float64)
        if fArr.shape[0] != 2 or fArr.shape[1] != 2:
            raise ValueError("matrixF must have shape (2, 2)")
        fView = fArr
        q00 = seedQView[0, 0]
        q01 = seedQView[0, 1]
        q10 = seedQView[1, 0]
        q11 = seedQView[1, 1]
        detQ = q00 * q11 - q01 * q10
        if (
            (not isfinite(q00))
            or (not isfinite(q01))
            or (not isfinite(q10))
            or (not isfinite(q11))
            or (not isfinite(detQ))
            or q00 <= 0.0
            or q11 <= 0.0
            or detQ <= 0.0
        ):
            raise ValueError("seedQ must be finite and positive definite")
        qInv00 = q11 / detQ
        qInv01 = -q01 / detQ
        qInv10 = -q10 / detQ
        qInv11 = q00 / detQ
        with nogil:
            _expectedTransitionEvidenceLevelTrendLoop(
                stateView,
                covView,
                lagView,
                fView,
                qInv00,
                qInv01,
                qInv10,
                qInv11,
                evidenceView,
                transitionCount,
                &finiteCount,
                &invalidCount,
                &negativeCount,
                &finiteSum,
                &finiteMin,
                &finiteMax,
            )

    diagnostics = {
        "state_dim": int(stateDim),
        "transition_count": int(transitionCount),
        "finite_count": int(finiteCount),
        "invalid_count": int(invalidCount),
        "negative_clamped_count": int(negativeCount),
        "finite_fraction": (
            float(finiteCount) / float(transitionCount)
            if transitionCount > 0
            else 1.0
        ),
        "mean": float(finiteSum / <double>finiteCount) if finiteCount > 0 else None,
        "min": float(finiteMin) if finiteCount > 0 else None,
        "max": float(finiteMax) if finiteCount > 0 else None,
    }
    return evidenceArr, diagnostics


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


cdef inline double _canonicalAR1BetaFromPairStats(double sumSqXSeq,
                                                  double sumSqYSeq,
                                                  double sumXYc,
                                                  Py_ssize_t blockLength,
                                                  double maxBeta,
                                                  double pairsRegLambda) noexcept nogil:
    cdef double eps
    cdef double nPairsDouble
    cdef double beta1
    cdef double lambdaEff
    cdef double scaleFloor
    cdef double ScaleX
    cdef double ScaleY
    cdef double denomSym

    if blockLength < 4:
        return 0.0
    if sumSqXSeq < 0.0:
        sumSqXSeq = 0.0
    if sumSqYSeq < 0.0:
        sumSqYSeq = 0.0

    nPairsDouble = <double>(blockLength - 1)
    eps = 1.0e-6 * (sumSqXSeq + sumSqYSeq + 1.0)
    lambdaEff = pairsRegLambda / (nPairsDouble + 1.0) # pairsRegLambda is effectively the number of pseudo-observations added
    scaleFloor = 1.0e-4 * (sumSqXSeq + 1.0)

    ScaleX = (sumSqXSeq * (1.0 + lambdaEff)) + scaleFloor
    ScaleY = (sumSqYSeq * (1.0 + lambdaEff)) + scaleFloor
    denomSym = sqrt(ScaleX * ScaleY)
    if denomSym > eps:
        beta1 = sumXYc / denomSym
    else:
        beta1 = 0.0

    if beta1 > maxBeta:
        beta1 = maxBeta
    elif beta1 < 0.0:
        beta1 = 0.0
    return beta1


cdef inline double _canonicalAR1VarianceFromPairStats(double sumSqXSeq,
                                                      double sumSqYSeq,
                                                      double sumXYc,
                                                      Py_ssize_t blockLength,
                                                      bint useInnovationVar,
                                                      double maxBeta,
                                                      double pairsRegLambda) noexcept nogil:
    cdef double beta1
    cdef double RSS
    cdef double pairCountDouble
    cdef double oneMinusBetaSq
    cdef double divRSS

    if blockLength < 4:
        return 0.0
    if sumSqXSeq < 0.0:
        sumSqXSeq = 0.0
    if sumSqYSeq < 0.0:
        sumSqYSeq = 0.0

    beta1 = _canonicalAR1BetaFromPairStats(
        sumSqXSeq,
        sumSqYSeq,
        sumXYc,
        blockLength,
        maxBeta,
        pairsRegLambda,
    )
    RSS = sumSqYSeq + ((beta1 * beta1) * sumSqXSeq) - (2.0 * (beta1 * sumXYc))
    if RSS < 0.0:
        RSS = 0.0

    pairCountDouble = <double>(blockLength - 3)
    oneMinusBetaSq = 1.0 - (beta1 * beta1)
    if useInnovationVar:
        divRSS = 1.0
    else:
        divRSS = oneMinusBetaSq

    if divRSS <= 1.0e-8:
        divRSS = 1.0e-8
    return RSS / pairCountDouble / divRSS


cdef inline void _regionMeanVar(double[::1] valuesView,
                                Py_ssize_t[::1] blockStartIndices,
                                Py_ssize_t[::1] blockSizes,
                                float[::1] meanOutView,
                                float[::1] varOutView,
                                double zeroPenalty,
                                double zeroThresh,
                                bint useInnovationVar,
                                double maxBeta=<double>0.95,
                                double pairsRegLambda=<double>1.0) noexcept nogil:
    # CALLERS: cmeanVarPairs

    cdef Py_ssize_t regionIndex, elementIndex, startIndex, blockLength
    cdef double sumY
    cdef double blockLengthDouble
    cdef double mom1
    cdef double* blockPtr
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

        varOutView[regionIndex] = <float>_canonicalAR1VarianceFromPairStats(
            sumSqXSeq,
            sumSqYSeq,
            sumXYc,
            blockLength,
            useInnovationVar,
            maxBeta,
            pairsRegLambda,
        )


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


cpdef cnp.ndarray[cnp.float64_t, ndim=1] csolveZeroCenteredBackground(
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    cdef double[::1] diagView
    cdef double[::1] rhsView
    cdef double[::1] constraintView
    cdef double[::1] firstLowerView
    cdef double[::1] outView

    if rhsTrack.shape[0] != n:
        raise ValueError("weightTrack and rhsTrack must have the same length")

    out = np.zeros(n, dtype=np.float64)
    if n <= 0:
        return out
    if n == 1:
        if not zeroCenter:
            denomOne = <double>weightTrack[0]
            if denomOne < minPivot:
                denomOne = minPivot
            out[0] = (<double>rhsTrack[0]) / denomOne
        return out

    diag = np.ascontiguousarray(weightTrack, dtype=np.float64).copy()
    rhs = np.ascontiguousarray(rhsTrack, dtype=np.float64).copy()
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
                outView[i] = rhsView[i] - mu * constraintView[i]
        else:
            for i in range(n):
                outView[i] = rhsView[i]

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


cdef tuple _dependenceAcfStats(
    cnp.ndarray[cnp.float64_t, ndim=1] centeredTrack,
    int maxSpan,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] trackArr = np.ascontiguousarray(
        centeredTrack,
        dtype=np.float64,
    )
    cdef Py_ssize_t n = <Py_ssize_t>trackArr.shape[0]
    cdef Py_ssize_t maxSpan_ = <Py_ssize_t>max(maxSpan, 0)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] acfNum = np.zeros(maxSpan_, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] acf = np.zeros(maxSpan_, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] pairCounts = np.zeros(maxSpan_, dtype=np.int64)
    cdef double[::1] trackView = trackArr
    cdef double[::1] acfNumView = acfNum
    cdef double[::1] acfView = acf
    cdef int64_t[::1] pairCountView = pairCounts
    cdef Py_ssize_t i, lag, runStart, runEnd, j
    cdef Py_ssize_t finiteCount = 0
    cdef double value, nextValue
    cdef double gamma0Sum = 0.0
    cdef double gamma0 = 0.0

    if n <= 0 or maxSpan_ <= 0:
        return acf, pairCounts, float(gamma0), int(finiteCount)

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
                            pairCountView[lag - 1] += 1

            for lag in range(maxSpan_):
                if pairCountView[lag] > 0:
                    acfView[lag] = (
                        acfNumView[lag] / <double>pairCountView[lag]
                    ) / gamma0
                else:
                    acfView[lag] = NAN
        else:
            for lag in range(maxSpan_):
                acfView[lag] = NAN

    return acf, pairCounts, float(gamma0), int(finiteCount)


cdef bint _isStandardAutosomeName(object chromosome):
    return <bint>misc_util.isStandardAutosomalChromosome(chromosome)


cdef tuple _normalizeDependenceSpanBounds(
    Py_ssize_t n,
    int minSpan,
    int maxSpan,
):
    cdef int minSpan_ = max(1, int(minSpan))
    cdef int maxSpan_ = max(minSpan_, int(maxSpan))
    if n > 0:
        maxSpan_ = min(maxSpan_, max(minSpan_, int(n) - 1))
    return int(minSpan_), int(maxSpan_)


cdef int _acfCrossingLag(cnp.ndarray[cnp.float64_t, ndim=1] acf, double threshold):
    cdef double[::1] acfView = acf
    cdef Py_ssize_t i
    if acfView.shape[0] < 3:
        return -1
    for i in range(acfView.shape[0] - 2):
        if (
            isfinite(acfView[i])
            and isfinite(acfView[i + 1])
            and isfinite(acfView[i + 2])
            and fabs(acfView[i]) < threshold
            and fabs(acfView[i + 1]) < threshold
            and fabs(acfView[i + 2]) < threshold
        ):
            return int(i + 1)
    return -1


cdef inline double _normalCdf(double x) noexcept nogil:
    if not isfinite(x):
        if x == INFINITY:
            return 1.0
        if x == -INFINITY:
            return 0.0
        return NAN
    if x <= -8.0:
        return 0.0
    if x >= 8.0:
        return 1.0
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


cdef double _dependenceAcfCrossingLogVariance(
    cnp.ndarray[cnp.float64_t, ndim=1] acf,
    cnp.ndarray[cnp.int64_t, ndim=1] pairCounts,
    int minSpan,
    int maxSpan,
    double threshold,
    int consecutive,
):
    cdef Py_ssize_t maxLag = <Py_ssize_t>acf.shape[0]
    cdef Py_ssize_t pairLag = <Py_ssize_t>pairCounts.shape[0]
    cdef int consecutive_ = int(consecutive)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] crossingProb
    cdef double[::1] acfView = acf
    cdef int64_t[::1] pairCountView = pairCounts
    cdef double[::1] crossingProbView
    cdef Py_ssize_t lagIndex
    cdef Py_ssize_t windowIndex
    cdef Py_ssize_t maxStart
    cdef int effectiveLag
    cdef int tailLag
    cdef int64_t nPairs
    cdef double rho
    cdef double prefixSq = 0.0
    cdef double varRho
    cdef double se
    cdef double p
    cdef double q
    cdef double mass
    cdef double survival = 1.0
    cdef double logSpan
    cdef double sumWeight = 0.0
    cdef double sumLog = 0.0
    cdef double sumLogSq = 0.0
    cdef double meanLog
    cdef double variance

    if consecutive_ < 1:
        consecutive_ = 1
    if pairLag < maxLag:
        maxLag = pairLag
    if maxSpan > 0 and <Py_ssize_t>maxSpan < maxLag:
        maxLag = <Py_ssize_t>maxSpan
    if maxLag < consecutive_:
        return NAN

    crossingProb = np.zeros(maxLag, dtype=np.float64)
    crossingProbView = crossingProb

    for lagIndex in range(maxLag):
        rho = acfView[lagIndex]
        nPairs = pairCountView[lagIndex]
        p = 0.0
        if isfinite(rho) and nPairs > 0:
            varRho = (1.0 + (2.0 * prefixSq)) / <double>nPairs
            if isfinite(varRho) and varRho > 0.0:
                se = sqrt(varRho)
                p = (
                    _normalCdf((threshold - rho) / se)
                    - _normalCdf((-threshold - rho) / se)
                )
                if not isfinite(p):
                    p = 0.0
                elif p < 0.0:
                    p = 0.0
                elif p > 1.0:
                    p = 1.0
        crossingProbView[lagIndex] = p
        if isfinite(rho):
            prefixSq += rho * rho

    maxStart = maxLag - <Py_ssize_t>consecutive_ + 1
    for lagIndex in range(maxStart):
        q = 1.0
        for windowIndex in range(consecutive_):
            q *= crossingProbView[lagIndex + windowIndex]
        if not isfinite(q):
            q = 0.0
        elif q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        mass = survival * q
        if mass > 0.0:
            effectiveLag = int(lagIndex + 1)
            if effectiveLag < minSpan:
                effectiveLag = int(minSpan)
            logSpan = log(<double>max(effectiveLag, 1))
            sumWeight += mass
            sumLog += mass * logSpan
            sumLogSq += mass * logSpan * logSpan

        survival *= (1.0 - q)
        if survival < 1.0e-15:
            survival = 0.0
            break

    if survival > 0.0:
        tailLag = int(max(maxSpan, minSpan))
        logSpan = log(<double>max(tailLag, 1))
        sumWeight += survival
        sumLog += survival * logSpan
        sumLogSq += survival * logSpan * logSpan

    if sumWeight <= 0.0:
        return NAN

    meanLog = sumLog / sumWeight
    variance = (sumLogSq / sumWeight) - (meanLog * meanLog)
    if not isfinite(variance):
        return NAN
    if variance < 0.0:
        variance = 0.0
    return max(0.01, variance)


cdef tuple _fallbackDependenceSpanResult(
    Py_ssize_t n,
    int minSpan,
    int maxSpan,
    int intervalSizeBP,
    str reason,
):
    cdef int point = int(max(minSpan, min(maxSpan, max(minSpan, int(round(sqrt(max(float(n), 1.0))))))))
    cdef int contextSizeBP = int(point * (2 * max(int(intervalSizeBP), 1)) + 1)
    return (
        int(point),
        int(point),
        int(point),
        {
            "method": "sampled_block_fallback",
            "fallback": True,
            "fallback_reason": reason,
            "point_span": int(point),
            "lower_span": int(point),
            "upper_span": int(point),
            "min_span": int(minSpan),
            "max_span": int(maxSpan),
            "finite_count": int(n),
            "context_size_bp": int(contextSizeBP),
            "estimand": "acf_abs_three_lag_crossing",
            "right_censored": False,
        },
    )


cdef tuple _estimateDependenceSpanForBlock(
    object blockMat,
    int intervalSizeBP,
    int minContextBP,
    int maxContextBP,
    double trim,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] contextTrack
    cdef cnp.ndarray[cnp.float64_t, ndim=1] finiteVals
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y
    cdef cnp.ndarray[cnp.float64_t, ndim=1] acf
    cdef cnp.ndarray[cnp.int64_t, ndim=1] pairCounts
    cdef Py_ssize_t n
    cdef Py_ssize_t finiteCount
    cdef int minSpan
    cdef int maxSpan
    cdef int crossingLag
    cdef int lowerCrossing
    cdef int upperCrossing
    cdef int pointSpan
    cdef int lowerSpan
    cdef int upperSpan
    cdef int contextSizeBP
    cdef double intervalSizeBP_ = <double>max(int(intervalSizeBP), 1)
    cdef double center
    cdef double scale
    cdef double lo
    cdef double hi
    cdef double gamma0
    cdef double acfCrossingLogVariance
    cdef bint rightCensored

    arr = np.ascontiguousarray(blockMat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("block matrices must be two-dimensional")
    n = <Py_ssize_t>arr.shape[1]
    minSpan = max(3, int(ceil(float(minContextBP) / (2.0 * intervalSizeBP_))))
    maxSpan = max(minSpan, int(ceil(float(maxContextBP) / (2.0 * intervalSizeBP_))))
    maxSpan = min(maxSpan, max(minSpan, max(3, int(n // 3))))
    minSpan, maxSpan = _normalizeDependenceSpanBounds(n, minSpan, maxSpan)

    if n < max(8, minSpan + 3):
        return _fallbackDependenceSpanResult(n, minSpan, maxSpan, intervalSizeBP, "too_few_intervals")

    contextTrack = np.asarray(ctrimMeanAxis0(arr, trim), dtype=np.float64)
    finiteVals = np.asarray(contextTrack[np.isfinite(contextTrack)], dtype=np.float64)
    finiteCount = <Py_ssize_t>finiteVals.size
    if finiteCount < max(20, minSpan + 3):
        return _fallbackDependenceSpanResult(finiteCount, minSpan, maxSpan, intervalSizeBP, "too_few_finite_values")

    center = float(np.median(finiteVals))
    scale = 1.4826 * float(np.median(np.abs(finiteVals - center)))
    if (not isfinite(scale)) or scale <= 0.0:
        scale = float(np.std(finiteVals, ddof=1)) if finiteCount >= 2 else 0.0
    if (not isfinite(scale)) or scale <= 0.0:
        scale = 1.0

    lo = float(np.quantile(finiteVals, 0.005))
    hi = float(np.quantile(finiteVals, 0.995))
    lo = max(lo, center - (8.0 * scale))
    hi = min(hi, center + (8.0 * scale))
    if (not isfinite(lo)) or (not isfinite(hi)) or hi <= lo:
        lo = center - (8.0 * scale)
        hi = center + (8.0 * scale)

    y = np.full(contextTrack.shape[0], np.nan, dtype=np.float64)
    y[np.isfinite(contextTrack)] = np.clip(contextTrack[np.isfinite(contextTrack)], lo, hi) - center
    acf, pairCounts, gamma0, finiteCount = _dependenceAcfStats(
        np.ascontiguousarray(y, dtype=np.float64),
        int(maxSpan),
    )
    if (not isfinite(gamma0)) or gamma0 <= 0.0:
        return _fallbackDependenceSpanResult(finiteCount, minSpan, maxSpan, intervalSizeBP, "zero_or_invalid_gamma0")

    crossingLag = _acfCrossingLag(acf, 0.10)
    rightCensored = crossingLag < 0
    pointSpan = int(maxSpan) if rightCensored else int(crossingLag)
    pointSpan = int(max(minSpan, min(maxSpan, pointSpan)))

    lowerCrossing = _acfCrossingLag(acf, 0.20)
    upperCrossing = _acfCrossingLag(acf, 0.05)
    lowerSpan = int(lowerCrossing) if lowerCrossing >= 0 else pointSpan
    upperSpan = int(upperCrossing) if upperCrossing >= 0 else pointSpan
    lowerSpan = int(max(minSpan, min(maxSpan, min(lowerSpan, pointSpan))))
    upperSpan = int(max(minSpan, min(maxSpan, max(upperSpan, pointSpan))))
    contextSizeBP = int(pointSpan * (2 * max(int(intervalSizeBP), 1)) + 1)
    acfCrossingLogVariance = _dependenceAcfCrossingLogVariance(
        acf,
        pairCounts,
        int(minSpan),
        int(maxSpan),
        0.10,
        3,
    )
    return (
        int(pointSpan),
        int(lowerSpan),
        int(upperSpan),
        {
            "method": "dependence_acf_sampled_block",
            "fallback": False,
            "point_span": int(pointSpan),
            "lower_span": int(lowerSpan),
            "upper_span": int(upperSpan),
            "context_size_bp": int(contextSizeBP),
            "estimand": "acf_abs_three_lag_crossing",
            "interval_size_bp": int(intervalSizeBP),
            "min_span": int(minSpan),
            "max_span": int(maxSpan),
            "trim": float(trim),
            "finite_count": int(finiteCount),
            "crossing_lag": None if crossingLag < 0 else int(crossingLag),
            "relaxed_crossing_lag": None if lowerCrossing < 0 else int(lowerCrossing),
            "strict_crossing_lag": None if upperCrossing < 0 else int(upperCrossing),
            "point_threshold": 0.10,
            "lower_threshold": 0.20,
            "upper_threshold": 0.05,
            "right_censored": bool(rightCensored),
            "log_span_variance": float(acfCrossingLogVariance),
            "log_span_variance_method": "acf_bartlett_crossing_distribution",
        },
    )


cdef double _dependenceLogVarianceFromDiagnostics(dict diagnostics, Py_ssize_t nBins):
    cdef object directVarianceObj = diagnostics.get("log_span_variance", None)
    cdef double directVariance = NAN
    cdef double point = max(1.0, float(diagnostics.get("point_span", 1.0)))
    cdef double lower = max(1.0, float(diagnostics.get("lower_span", point)))
    cdef double upper = max(lower, float(diagnostics.get("upper_span", point)))
    cdef double bracketWidth = log((upper + 1.0) / (lower + 1.0))
    cdef double finiteCount = float(diagnostics.get("finite_count", 0.0) or 0.0)
    cdef double finiteFraction = min(1.0, max(0.0, finiteCount / max(1.0, float(nBins))))
    cdef int minSpan = int(diagnostics.get("min_span", 0) or 0)
    cdef int maxSpan = int(diagnostics.get("max_span", 0) or 0)
    cdef double boundaryPenalty = 0.0
    cdef double missingCrossingPenalty = 0.0
    cdef double censoringPenalty = 0.0
    cdef double variance

    if int(round(point)) == minSpan or int(round(point)) == maxSpan:
        boundaryPenalty = 0.75
    if diagnostics.get("crossing_lag") is None:
        missingCrossingPenalty += 0.35
    if diagnostics.get("relaxed_crossing_lag") is None:
        missingCrossingPenalty += 0.20
    if diagnostics.get("strict_crossing_lag") is None:
        missingCrossingPenalty += 0.20
    if bool(diagnostics.get("right_censored", False)):
        censoringPenalty = 1.0

    if directVarianceObj is not None:
        directVariance = float(directVarianceObj)
        if isfinite(directVariance) and directVariance > 0.0:
            return max(
                0.01,
                directVariance
                + boundaryPenalty
                + missingCrossingPenalty
                + censoringPenalty,
            )

    variance = (
        0.04
        + bracketWidth * bracketWidth
        + 1.5 * (1.0 - finiteFraction)
        + boundaryPenalty
        + missingCrossingPenalty
        + censoringPenalty
    )
    return max(0.01, variance)


cpdef tuple cchooseDependenceSpan(
    object chromosomeNames,
    object chromosomeMatrices,
    int intervalSizeBP,
    int numBlocks=100,
    int randSeed=1729,
    double blockMedianBP=50000.0,
    double blockSigma=1.0,
    int blockMinBP=1000,
    int blockMaxBP=1000000,
    int minContextBP=500,
    int maxContextBP=100000,
    double priorMedianSpan=80.0,
    double priorLogSd=1.0,
    double trim=0.10,
):
    r"""Sample blocks across autosomes and choose a pooled dependence span."""

    cdef list names = list(chromosomeNames)
    cdef list matrices = list(chromosomeMatrices)
    cdef list eligibleNames = []
    cdef list eligibleMatrices = []
    cdef list eligibleBins = []
    cdef list excludedNames = []
    cdef list sampledChromosomes = []
    cdef list sampledWidths = []
    cdef list sampledPointSpans = []
    cdef list logSpans = []
    cdef list logVariances = []
    cdef Py_ssize_t i
    cdef Py_ssize_t selected
    cdef Py_ssize_t nBins
    cdef int intervalSizeBP_ = max(int(intervalSizeBP), 1)
    cdef int minSpan = max(3, int(ceil(float(minContextBP) / (2.0 * float(intervalSizeBP_)))))
    cdef int maxSpan = max(minSpan, int(ceil(float(maxContextBP) / (2.0 * float(intervalSizeBP_)))))
    cdef int blocksRequested = max(0, int(numBlocks))
    cdef int validBlocks = 0
    cdef int fallbackBlocks = 0
    cdef int rightCensoredBlocks = 0
    cdef int widthBP
    cdef int blockBins
    cdef int startBin
    cdef int endBin
    cdef int point
    cdef int lower
    cdef int upper
    cdef int pointSpan
    cdef int lowerSpan
    cdef int upperSpan
    cdef int contextSizeBP
    cdef double logWidth
    cdef double postMean
    cdef double postSd
    cdef double tau2 = 0.0
    cdef double rawTau2
    cdef double priorMu
    cdef double priorVar
    cdef double priorPrec
    cdef double postPrec
    cdef double obsPrec
    cdef double obsPrecSum = 0.0
    cdef double obsWeightedSum = 0.0
    cdef double robustLogMedian = NAN
    cdef double robustLogMad = NAN
    cdef double robustBlend = 0.0
    cdef double robustSdFloor = 0.0
    cdef double censorFraction = 0.0
    cdef double censorBlend = 0.0
    cdef dict blockDiagnostics
    cdef dict diagnostics
    cdef object rng
    cdef object matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=1] placementWeights
    cdef cnp.ndarray[cnp.float64_t, ndim=1] logSpanArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] logVarArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] eligibleForBlock
    cdef list eligibleForBlockList
    cdef list placementList

    if len(names) != len(matrices):
        raise ValueError("chromosomeNames and chromosomeMatrices must have the same length")
    if blockMedianBP <= 0.0 or blockSigma <= 0.0 or blockMinBP <= 0 or blockMaxBP < blockMinBP:
        raise ValueError("invalid dependence block-size distribution parameters")
    if priorMedianSpan <= 0.0 or priorLogSd <= 0.0:
        raise ValueError("invalid dependence span prior parameters")

    for i in range(len(names)):
        matrix = matrices[i]
        if _isStandardAutosomeName(names[i]):
            nBins = <Py_ssize_t>matrix.shape[1]
            if nBins > 1:
                eligibleNames.append(str(names[i]))
                eligibleMatrices.append(matrix)
                eligibleBins.append(int(nBins))
            else:
                excludedNames.append(str(names[i]))
        else:
            excludedNames.append(str(names[i]))

    rng = default_rng(randSeed)
    if len(eligibleNames) > 0 and blocksRequested > 0:
        while len(sampledWidths) < blocksRequested:
            drawn = rng.lognormal(mean=log(blockMedianBP), sigma=blockSigma)
            if drawn < blockMinBP or drawn > blockMaxBP:
                continue
            widthBP = int(round(float(drawn)))
            widthBP = max(blockMinBP, min(blockMaxBP, widthBP))
            eligibleForBlockList = []
            placementList = []
            for i in range(len(eligibleNames)):
                nBins = int(eligibleBins[i])
                blockBins = max(1, int(ceil(float(widthBP) / float(intervalSizeBP_))))
                blockBins = min(blockBins, nBins)
                if nBins >= blockBins:
                    eligibleForBlockList.append(i)
                    placementList.append(max(1.0, float(nBins - blockBins + 1)))
            if not eligibleForBlockList:
                fallbackBlocks += 1
                continue
            placementWeights = np.asarray(placementList, dtype=np.float64)
            placementWeights = placementWeights / float(np.sum(placementWeights))
            selected = int(rng.choice(np.asarray(eligibleForBlockList, dtype=np.int64), p=placementWeights))
            nBins = int(eligibleBins[selected])
            blockBins = max(1, int(ceil(float(widthBP) / float(intervalSizeBP_))))
            blockBins = min(blockBins, nBins)
            startBin = int(rng.integers(0, max(1, nBins - blockBins + 1)))
            endBin = min(nBins, startBin + blockBins)
            if endBin <= startBin:
                fallbackBlocks += 1
                continue
            point, lower, upper, blockDiagnostics = _estimateDependenceSpanForBlock(
                np.asarray(eligibleMatrices[selected])[:, startBin:endBin],
                intervalSizeBP_,
                minContextBP,
                maxContextBP,
                trim,
            )
            sampledChromosomes.append(str(eligibleNames[selected]))
            sampledWidths.append(int(widthBP))
            sampledPointSpans.append(int(point))
            if bool(blockDiagnostics.get("fallback", False)):
                fallbackBlocks += 1
                continue
            if bool(blockDiagnostics.get("right_censored", False)):
                rightCensoredBlocks += 1
            logSpans.append(log(max(1.0, float(point))))
            logVariances.append(_dependenceLogVarianceFromDiagnostics(blockDiagnostics, endBin - startBin))
            validBlocks += 1

    priorMu = log(float(priorMedianSpan))
    priorVar = priorLogSd * priorLogSd
    if validBlocks > 0:
        logSpanArr = np.asarray(logSpans, dtype=np.float64)
        logVarArr = np.asarray(logVariances, dtype=np.float64)
        rawTau2 = float(np.var(logSpanArr, ddof=1) - np.mean(logVarArr)) if logSpanArr.size >= 2 else 0.0
        tau2 = max(0.0, rawTau2)
        for i in range(logSpanArr.size):
            obsPrec = 1.0 / (float(logVarArr[i]) + tau2)
            obsPrecSum += obsPrec
            obsWeightedSum += obsPrec * float(logSpanArr[i])
        priorPrec = 1.0 / priorVar
        postPrec = priorPrec + obsPrecSum
        postMean = ((priorMu * priorPrec) + obsWeightedSum) / postPrec
        postSd = sqrt(1.0 / postPrec)
        if logSpanArr.size >= 8:
            robustLogMedian = float(np.median(logSpanArr))
            robustLogMad = 1.4826 * float(np.median(np.abs(logSpanArr - robustLogMedian)))
            if isfinite(robustLogMad) and robustLogMad > 0.25:
                robustBlend = min(0.85, max(0.0, (robustLogMad - 0.25) / (robustLogMad + 0.25)))
            if rightCensoredBlocks > 0:
                censorFraction = min(1.0, max(0.0, float(rightCensoredBlocks) / float(validBlocks)))
                censorBlend = min(0.85, 1.5 * censorFraction)
                if censorBlend > robustBlend:
                    robustBlend = censorBlend
            if robustBlend > 0.0:
                postMean = ((1.0 - robustBlend) * postMean) + (robustBlend * robustLogMedian)
                robustSdFloor = 0.25 * robustBlend * max(robustLogMad, censorFraction)
                if robustSdFloor > 0.50:
                    robustSdFloor = 0.50
                if robustSdFloor > postSd:
                    postSd = robustSdFloor
        pointSpan = int(round(exp(postMean)))
        lowerSpan = int(floor(exp(postMean - 1.96 * postSd)))
        upperSpan = int(ceil(exp(postMean + 1.96 * postSd)))
        fallback = False
    else:
        postMean = priorMu
        postSd = priorLogSd
        pointSpan = int(round(priorMedianSpan))
        lowerSpan = pointSpan
        upperSpan = pointSpan
        fallback = True

    pointSpan = int(max(minSpan, min(maxSpan, pointSpan)))
    lowerSpan = int(max(minSpan, min(pointSpan, lowerSpan)))
    upperSpan = int(max(pointSpan, min(maxSpan, upperSpan)))
    contextSizeBP = int(pointSpan * (2 * intervalSizeBP_) + 1)
    diagnostics = {
        "method": "sampled_block_lognormal_map",
        "num_blocks": int(blocksRequested),
        "blocks_requested": int(blocksRequested),
        "blocks_valid": int(validBlocks),
        "fallback_blocks": int(fallbackBlocks),
        "right_censored_blocks": int(rightCensoredBlocks),
        "fallback": bool(fallback),
        "point_span": int(pointSpan),
        "lower_span": int(lowerSpan),
        "upper_span": int(upperSpan),
        "context_size_bp": int(contextSizeBP),
        "estimand": "acf_abs_three_lag_crossing",
        "point_threshold": 0.10,
        "lower_threshold": 0.20,
        "upper_threshold": 0.05,
        "interval_size_bp": int(intervalSizeBP_),
        "min_span": int(minSpan),
        "max_span": int(maxSpan),
        "chromosomes_used": sorted(set(sampledChromosomes)),
        "chromosomes_excluded": sorted(set(excludedNames)),
        "excluded_nonstandard_chromosomes": sorted(set(excludedNames)),
        "sampled_chromosomes": list(sampledChromosomes),
        "sampled_width_bp": [int(v) for v in sampledWidths],
        "sampled_point_span": [int(v) for v in sampledPointSpans],
        "sampled_width_median_bp": (
            float(np.median(np.asarray(sampledWidths, dtype=np.float64)))
            if len(sampledWidths) > 0
            else float("nan")
        ),
        "posterior_log_span_mean": float(postMean),
        "posterior_log_span_sd": float(postSd),
        "tau2": float(tau2),
        "robust_log_span_median": float(robustLogMedian),
        "robust_log_span_mad": float(robustLogMad),
        "robust_aggregation_blend": float(robustBlend),
        "right_censored_fraction": float(censorFraction),
        "block_log_span_variance_method": "acf_bartlett_crossing_distribution",
        "block_lognormal_median_bp": float(blockMedianBP),
        "block_lognormal_sigma": float(blockSigma),
        "block_min_bp": int(blockMinBP),
        "block_max_bp": int(blockMaxBP),
        "prior_median_span": float(priorMedianSpan),
        "prior_log_sd": float(priorLogSd),
    }
    return int(pointSpan), int(lowerSpan), int(upperSpan), diagnostics


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
                          double maxBeta = 0.95,
                          double pairsRegLambda = <double>1.0):

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
        maxBeta,
        pairsRegLambda,
    )

    return outMeans, outVars, starts_, ends


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cblockAR1Beta(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    cnp.ndarray[cnp.intp_t, ndim=1] blockStarts,
    cnp.ndarray[cnp.intp_t, ndim=1] blockSizes,
    double maxBeta=0.95,
    double pairsRegLambda=1.0,
):
    r"""Estimate the clipped AR(1) beta for each explicit block."""

    cdef cnp.ndarray[cnp.float64_t, ndim=1] valuesArray
    cdef double[::1] valuesView
    cdef Py_ssize_t[::1] startsView = blockStarts
    cdef Py_ssize_t[::1] sizesView = blockSizes
    cdef cnp.ndarray[cnp.float32_t, ndim=1] outBeta
    cdef float[::1] outView
    cdef Py_ssize_t nBlocks
    cdef Py_ssize_t valuesLength
    cdef Py_ssize_t regionIndex
    cdef Py_ssize_t elementIndex
    cdef Py_ssize_t startIndex
    cdef Py_ssize_t blockLength
    cdef double* blockPtr
    cdef double sumY
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

    if blockStarts.shape[0] != blockSizes.shape[0]:
        raise ValueError("blockStarts and blockSizes must align")

    nBlocks = <Py_ssize_t>blockStarts.shape[0]
    outBeta = np.empty(nBlocks, dtype=np.float32)
    outView = outBeta
    valuesArray = np.ascontiguousarray(values, dtype=np.float64)
    valuesView = valuesArray
    valuesLength = <Py_ssize_t>valuesArray.shape[0]

    with nogil:
        for regionIndex in range(nBlocks):
            startIndex = startsView[regionIndex]
            blockLength = sizesView[regionIndex]
            if (
                blockLength < 4
                or startIndex < 0
                or startIndex + blockLength > valuesLength
            ):
                outView[regionIndex] = <cnp.float32_t>-1.0
                continue

            blockPtr = &valuesView[startIndex]
            sumY = 0.0
            for elementIndex in range(blockLength):
                sumY += blockPtr[elementIndex]

            nPairsDouble = <double>(blockLength - 1)
            sumXSeq = sumY - blockPtr[blockLength - 1]
            sumYSeq = sumY - blockPtr[0]
            meanX = sumXSeq / nPairsDouble
            meanYp = sumYSeq / nPairsDouble

            sumSqXSeq = 0.0
            sumSqYSeq = 0.0
            sumXYc = 0.0
            for elementIndex in range(blockLength - 1):
                xDev = blockPtr[elementIndex] - meanX
                yDev = blockPtr[elementIndex + 1] - meanYp
                sumSqXSeq += xDev * xDev
                sumSqYSeq += yDev * yDev
                sumXYc += xDev * yDev

            outView[regionIndex] = <cnp.float32_t>(
                _canonicalAR1BetaFromPairStats(
                    sumSqXSeq,
                    sumSqYSeq,
                    sumXYc,
                    blockLength,
                    maxBeta,
                    pairsRegLambda,
                )
            )

    return outBeta


cpdef tuple cSparseNearestMeanVarTrack(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    cnp.ndarray[cnp.intp_t, ndim=1] sparseCenters,
    cnp.ndarray[cnp.intp_t, ndim=1] blockStarts,
    cnp.ndarray[cnp.intp_t, ndim=1] blockSizes,
    int numNearest,
    double zeroPenalty=0.0,
    double zeroThresh=0.0,
    bint useInnovationVar=True,
    bint aggregateMeanAbs=True,
    double maxBeta=0.95,
    double pairsRegLambda=1.0,
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
        maxBeta,
        pairsRegLambda,
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


cdef inline real_t _transformValue(
    real_t xval,
    int mode,
    real_t inputOffset,
    real_t inputScale,
    real_t outputScale,
    real_t outputOffset,
    real_t shape,
) noexcept nogil:
    cdef real_t u
    cdef double ud
    cdef double shapeD

    if mode == __TRANSFORM_MODE_LOG:
        u = xval + inputOffset
        if u <= <real_t>0.0:
            u = inputOffset
        u = u / inputScale
        if u <= <real_t>0.0:
            u = <real_t>1.0
        return outputOffset + outputScale * <real_t>log(<double>u)

    if mode == __TRANSFORM_MODE_SQRT or mode == __TRANSFORM_MODE_ANSCOMBE:
        u = (xval + inputOffset) / inputScale
        if u < <real_t>0.0:
            u = <real_t>0.0
        return outputOffset + outputScale * <real_t>sqrt(<double>u)

    if mode == __TRANSFORM_MODE_ASINH:
        u = (xval + inputOffset) / inputScale
        return outputOffset + outputScale * <real_t>asinh(<double>u)

    if mode == __TRANSFORM_MODE_ASINH_SQRT:
        u = xval + inputOffset
        if u < <real_t>0.0:
            u = <real_t>0.0
        return outputOffset + outputScale * <real_t>asinh(
            sqrt(<double>u) / <double>inputScale
        )

    if mode == __TRANSFORM_MODE_GENERALIZED_LOG:
        ud = <double>((xval + inputOffset) / inputScale)
        shapeD = <double>shape
        return outputOffset + outputScale * <real_t>(
            log((ud + sqrt((ud * ud) + (shapeD * shapeD))) / shapeD)
        )

    return outputOffset + outputScale * (
        (xval + inputOffset) / inputScale
    )


cdef void _monoTransform(
    const real_t* arrPtr,
    real_t* outPtr,
    Py_ssize_t n,
    real_t inputOffset,
    real_t inputScale,
    real_t outputScale,
    real_t outputOffset,
    real_t shape,
    int mode,
) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        outPtr[i] = _transformValue(
            arrPtr[i],
            mode,
            inputOffset,
            inputScale,
            outputScale,
            outputOffset,
            shape,
        )


cdef void _transformDiff(
    const real_t* treatmentPtr,
    const real_t* controlPtr,
    real_t* outPtr,
    Py_ssize_t n,
    real_t inputOffset,
    real_t inputScale,
    real_t outputScale,
    real_t shape,
    int mode,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef real_t t
    cdef real_t c

    if mode == __TRANSFORM_MODE_LOG:
        for i in range(n):
            t = treatmentPtr[i] + inputOffset
            c = controlPtr[i] + inputOffset
            if t <= <real_t>0.0:
                t = inputOffset
            if c <= <real_t>0.0:
                c = inputOffset
            outPtr[i] = outputScale * <real_t>(
                log(<double>t) - log(<double>c)
            )
        return

    for i in range(n):
        outPtr[i] = (
            _transformValue(
                treatmentPtr[i],
                mode,
                inputOffset,
                inputScale,
                outputScale,
                <real_t>0.0,
                shape,
            )
            - _transformValue(
                controlPtr[i],
                mode,
                inputOffset,
                inputScale,
                outputScale,
                <real_t>0.0,
                shape,
            )
        )


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
            _monoTransform(
                <const float*>arr_F32.data,
                <float*>out_F32.data,
                n,
                <float>offset_,
                <float>1.0,
                <float>scale_,
                <float>0.0,
                <float>1.0,
                __TRANSFORM_MODE_LOG,
            )
        return (out_F32, -1.0)

    arr_F64 = np.ascontiguousarray(x, dtype=np.float64)
    n = arr_F64.shape[0]
    out_F64 = np.empty(n, dtype=np.float64)
    with nogil:
        _monoTransform(
            <const double*>arr_F64.data,
            <double*>out_F64.data,
            n,
            offset_,
            1.0,
            scale_,
            0.0,
            1.0,
            __TRANSFORM_MODE_LOG,
        )

    return (out_F64, -1.0)


cpdef object cTransformWithInput(
    object treatment,
    object control,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
    object mode=None,
    object offset=None,
    object scale=None,
    object inputOffset=None,
    object inputScale=None,
    object outputScale=None,
    object outputOffset=None,
    object shape=None,
):
    r"""Return the treatment/control transform difference.

    The default is the historical log-ratio transform
    ``logMult * (log(treatment + logOffset) - log(control + logOffset))``.
    Non-log modes return ``f(treatment) - f(control)`` so depletion remains
    signed relative to control.
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
            mode=mode,
            offset=offset,
            scale=scale,
            inputOffset=inputOffset,
            inputScale=inputScale,
            outputScale=outputScale,
            outputOffset=outputOffset,
            shape=shape,
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
        mode=mode,
        offset=offset,
        scale=scale,
        inputOffset=inputOffset,
        inputScale=inputScale,
        outputScale=outputScale,
        outputOffset=outputOffset,
        shape=shape,
    )


cpdef object cTransformWithInputInto(
    object treatment,
    object control,
    object out,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
    object mode=None,
    object offset=None,
    object scale=None,
    object inputOffset=None,
    object inputScale=None,
    object outputScale=None,
    object outputOffset=None,
    object shape=None,
):
    r"""Write a treatment/control transform difference into ``out``."""
    cdef Py_ssize_t n
    cdef int modeCode = _parseTransformMode(mode)
    cdef tuple transformParams = _resolveTransformParameters(
        modeCode,
        logOffset,
        logMult,
        offset,
        scale,
        inputOffset,
        inputScale,
        outputScale,
        outputOffset,
        shape,
    )
    cdef double inputOffset_ = <double>transformParams[0]
    cdef double inputScale_ = <double>transformParams[1]
    cdef double outputScale_ = <double>transformParams[2]
    cdef double shape_ = <double>transformParams[4]
    cdef object outObj = out
    cdef cnp.ndarray[cnp.float32_t, ndim=1] treat_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] control_F32
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out_F32
    cdef cnp.ndarray[cnp.float64_t, ndim=1] treat_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] control_F64
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_F64

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
            _transformDiff(
                <const float*>treat_F32.data,
                <const float*>control_F32.data,
                <float*>out_F32.data,
                n,
                <float>inputOffset_,
                <float>inputScale_,
                <float>outputScale_,
                <float>shape_,
                modeCode,
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
            _transformDiff(
                <const double*>treat_F64.data,
                <const double*>control_F64.data,
                <double*>out_F64.data,
                n,
                inputOffset_,
                inputScale_,
                outputScale_,
                shape_,
                modeCode,
            )
        return out

    raise TypeError("out dtype must be float32 or float64")


cpdef object cTransformInPlace(
    object x,
    bint verbose=<bint>False,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
    object mode=None,
    object offset=None,
    object scale=None,
    object inputOffset=None,
    object inputScale=None,
    object outputScale=None,
    object outputOffset=None,
    object shape=None,
):
    r"""Transform a contiguous coverage track in-place."""
    cdef Py_ssize_t n
    cdef int modeCode = _parseTransformMode(mode)
    cdef tuple transformParams = _resolveTransformParameters(
        modeCode,
        logOffset,
        logMult,
        offset,
        scale,
        inputOffset,
        inputScale,
        outputScale,
        outputOffset,
        shape,
    )
    cdef double inputOffset_ = <double>transformParams[0]
    cdef double inputScale_ = <double>transformParams[1]
    cdef double outputScale_ = <double>transformParams[2]
    cdef double outputOffset_ = <double>transformParams[3]
    cdef double shape_ = <double>transformParams[4]
    cdef object arrObj = x
    cdef cnp.ndarray zArr_F32
    cdef cnp.ndarray zArr_F64

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
            _monoTransform(
                <const float*>zArr_F32.data,
                <float*>zArr_F32.data,
                n,
                <float>inputOffset_,
                <float>inputScale_,
                <float>outputScale_,
                <float>outputOffset_,
                <float>shape_,
                modeCode,
            )
        return x

    if (<cnp.ndarray>arrObj).dtype != np.float64:
        raise TypeError("x dtype must be float32 or float64")
    zArr_F64 = arrObj
    n = zArr_F64.shape[0]
    with nogil:
        _monoTransform(
            <const double*>zArr_F64.data,
            <double*>zArr_F64.data,
            n,
            inputOffset_,
            inputScale_,
            outputScale_,
            outputOffset_,
            shape_,
            modeCode,
        )

    return x


cpdef object cTransform(
    object x,
    bint verbose=<bint>False,
    double logOffset=<double>(1.0),
    double logMult=<double>(1.0),
    object mode=None,
    object offset=None,
    object scale=None,
    object inputOffset=None,
    object inputScale=None,
    object outputScale=None,
    object outputOffset=None,
    object shape=None,
):
    r"""Transform a coverage track."""
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
        mode=mode,
        offset=offset,
        scale=scale,
        inputOffset=inputOffset,
        inputScale=inputScale,
        outputScale=outputScale,
        outputOffset=outputOffset,
        shape=shape,
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
    object processQScale=None,
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
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processQScaleArr
    cdef cnp.float32_t[::1] processQScaleView
    cdef bint useProcessQScale = (processQScale is not None)
    cdef bint useProcPrec = (
        ECM_useProcessPrecisionReweighting
        and (processPrecExp is not None)
        and ((not ECM_useAPN) or useProcessQScale)
    )
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
    cdef double qDiagBase
    cdef double apnScale = 1.0
    cdef double qScale
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

    if useProcessQScale:
        processQScaleArr = _coerceProcessQScale(processQScale, intervalCount)
        processQScaleView = processQScaleArr

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

    _validateMultiplierBounds(wMin, wMax, True)
    _validateMultiplierBounds(procPrecMin, procPrecMax, False)

    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")

    if useLambda:
        if lambdaExpArr.shape[0] != intervalCount:
            raise ValueError("lambdaExp length must match intervalCount")

    if useProcPrec:
        if processPrecExpArr.shape[0] != intervalCount:
            raise ValueError("processPrecExp length must match intervalCount")

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
            procPrec = _clampMultiplierValue(<double>processPrecExpView[k], procPrecMin, procPrecMax)
        else:
            procPrec = 1.0

        # ========================================================
        # Predict step transition
        # ========================================================
        xPred0 = F00*(<double>stateVectorView[0]) + F01*(<double>stateVectorView[1])
        xPred1 = F10*(<double>stateVectorView[0]) + F11*(<double>stateVectorView[1])
        stateVectorView[0] = <cnp.float32_t>xPred0
        stateVectorView[1] = <cnp.float32_t>xPred1

        # processQScale is interval-indexed: k=0 is fixed at 1, transition k-1 -> k uses processQScale[k].
        if useProcessQScale:
            qScale = <double>processQScaleView[k]
            Q00 = (qScale / procPrec) * (<double>q0View[0, 0])
            Q01 = (qScale / procPrec) * (<double>q0View[0, 1])
            Q10 = (qScale / procPrec) * (<double>q0View[1, 0])
            Q11 = (qScale / procPrec) * (<double>q0View[1, 1])
        else:
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
            obsPrec = _clampMultiplierValue(<double>lambdaExpView[k], wMin, wMax)
        else:
            obsPrec = 1.0

        sumInvR = 0.0
        sumInvRInnov = 0.0
        sumInvRInnov2 = 0.0
        if returnNLL:
            sumLogR = 0.0
            intervalNLL = 0.0

        for j in range(trackCount):
            _accumulateObservationValue(
                <double>dataView[j, k],
                <double>stateVectorView[0],
                <double>muncMatView[j, k],
                <double>pad,
                obsPrec,
                returnNLL,
                &sumInvR,
                &sumInvRInnov,
                &sumInvRInnov2,
                &sumLogR,
            )

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

        if ECM_useAPN and (not useProcessQScale):
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

    cdef cnp.float32_t[:, ::1] stateSmoothedView
    cdef cnp.float32_t[:, :, ::1] stateCovarSmoothedView
    cdef cnp.float32_t[:, :, ::1] lagCovSmoothedView
    cdef cnp.float32_t[:, ::1] postFitResidualsView
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
                innov = (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                postFitResidualsView[k, j] = <cnp.float32_t>innov

    return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)


cpdef tuple cforwardPassLevel(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixPluginMuncInit,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixQ0,
    cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] intervalToBlockMap,
    Py_ssize_t blockCount,
    float stateInit,
    float stateCovarInit,
    float pad=1.0e-4,
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
    object processQScale=None,
):
    r"""Run the scalar level-only forward pass."""

    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] muncMatView = matrixPluginMuncInit
    cdef cnp.float32_t[:, ::1] q0View = matrixQ0
    cdef cnp.int32_t[::1] blockMapView = intervalToBlockMap
    cdef Py_ssize_t trackCount = dataView.shape[0]
    cdef Py_ssize_t intervalCount = dataView.shape[1]
    cdef Py_ssize_t k, j
    cdef Py_ssize_t blockId
    cdef bint doStore = (stateForward is not None)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] dStatVectorArr
    cdef cnp.float32_t[::1] dStatVector
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
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processQScaleArr
    cdef cnp.float32_t[::1] processQScaleView
    cdef bint useProcessQScale = (processQScale is not None)
    cdef bint useProcPrec = (
        ECM_useProcessPrecisionReweighting
        and (processPrecExp is not None)
        and ((not ECM_useAPN) or useProcessQScale)
    )
    cdef double stateValue = <double>stateInit
    cdef double stateVar = <double>stateCovarInit
    cdef double q0 = <double>q0View[0, 0]
    cdef double Q
    cdef double baseVar, measVar, invMeasVar
    cdef double sumInvR, sumInvRInnov, sumInvRInnov2
    cdef double sumLogR = 0.0
    cdef double sumNLL = 0.0
    cdef double intervalNLL = 0.0
    cdef double sumDStat = 0.0
    cdef double innov
    cdef double innovScale
    cdef double gainLike
    cdef double quadForm
    cdef double delta0
    cdef double gainG
    cdef double gainH
    cdef double IKH
    cdef double PNew
    cdef double obsPrec
    cdef double procPrec
    cdef double wMin = <double>obsPrecisionMultiplierMin
    cdef double wMax = <double>obsPrecisionMultiplierMax
    cdef double procPrecMin = <double>procPrecisionMultiplierMin
    cdef double procPrecMax = <double>procPrecisionMultiplierMax
    cdef double phiHat = 1.0
    cdef double apnScale = 1.0
    cdef double qScale
    cdef double currentProcNoise
    cdef double adaptiveMult
    cdef double apnMinQ = <double>APN_minQ
    cdef double apnMaxQ = <double>APN_maxQ
    cdef double apnThresh = <double>APN_dStatThresh
    cdef double apnScaleCoef = <double>APN_dStatScale
    cdef double apnPC = <double>APN_dStatPC
    cdef double LOG2PI = log(6.2831853071795864769)

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
    if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
        raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
    if matrixQ0.shape[0] < 1 or matrixQ0.shape[1] < 1:
        raise ValueError("matrixQ0 must have at least shape (1, 1)")
    if q0 <= 0.0:
        raise ValueError("matrixQ0[0, 0] must be positive")
    _validateMultiplierBounds(wMin, wMax, True)
    _validateMultiplierBounds(procPrecMin, procPrecMax, False)
    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")

    if useLambda:
        lambdaExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> lambdaExp
        if lambdaExpArr.shape[0] != intervalCount:
            raise ValueError("lambdaExp length must match intervalCount")
        lambdaExpView = lambdaExpArr
    if useProcPrec:
        processPrecExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> processPrecExp
        if processPrecExpArr.shape[0] != intervalCount:
            raise ValueError("processPrecExp length must match intervalCount")
        processPrecExpView = processPrecExpArr
    if useProcessQScale:
        processQScaleArr = _coerceProcessQScale(processQScale, intervalCount)
        processQScaleView = processQScaleArr
    if vectorD is None:
        dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
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

    if q0 <= 1.0e-12:
        ECM_useAPN = False

    for k in range(intervalCount):
        blockId = <Py_ssize_t>blockMapView[k]
        if blockId < 0 or blockId >= blockCount:
            raise ValueError("intervalToBlockMap has out-of-range block id")

        if useProcPrec:
            procPrec = _clampMultiplierValue(<double>processPrecExpView[k], procPrecMin, procPrecMax)
        else:
            procPrec = 1.0

        if useProcessQScale:
            qScale = <double>processQScaleView[k]
            Q = (qScale / procPrec) * q0
        else:
            Q = (apnScale / procPrec) * q0
        stateVar = stateVar + Q

        if useLambda:
            obsPrec = _clampMultiplierValue(<double>lambdaExpView[k], wMin, wMax)
        else:
            obsPrec = 1.0

        sumInvR = 0.0
        sumInvRInnov = 0.0
        sumInvRInnov2 = 0.0
        if returnNLL:
            sumLogR = 0.0
            intervalNLL = 0.0

        for j in range(trackCount):
            _accumulateObservationValue(
                <double>dataView[j, k],
                stateValue,
                <double>muncMatView[j, k],
                <double>pad,
                obsPrec,
                returnNLL,
                &sumInvR,
                &sumInvRInnov,
                &sumInvRInnov2,
                &sumLogR,
            )

        innovScale = 1.0 + stateVar * sumInvR
        gainLike = stateVar / innovScale
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
        stateValue = stateValue + stateVar * delta0

        gainG = sumInvR / innovScale
        gainH = sumInvR / (innovScale * innovScale)
        IKH = 1.0 - stateVar * gainG
        PNew = (IKH * IKH * stateVar) + (gainH * (stateVar * stateVar))
        stateVar = PNew

        if doStore:
            stateForwardView[k, 0] = <cnp.float32_t>stateValue
            stateCovarForwardView[k, 0, 0] = <cnp.float32_t>stateVar
            if k > 0:
                pNoiseForwardView[k - 1, 0, 0] = <cnp.float32_t>Q

        if ECM_useAPN and (not useProcessQScale):
            currentProcNoise = apnScale * q0
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

            currentProcNoise = apnScale * q0
            if currentProcNoise < apnMinQ:
                apnScale = apnMinQ / q0
            elif currentProcNoise > apnMaxQ:
                apnScale = apnMaxQ / q0

    phiHat = sumDStat / (<double>intervalCount)
    if returnNLL:
        return (<float>phiHat, 0, vectorD, sumNLL)
    return (<float>phiHat, 0, vectorD)


cpdef tuple cbackwardPassLevel(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForward,
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForward,
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForward,
    Py_ssize_t chunkSize=1000000,
    object stateSmoothed=None,
    object stateCovarSmoothed=None,
    object lagCovSmoothed=None,
    object postFitResiduals=None,
    object progressBar=None,
    Py_ssize_t progressIter=10000,
):
    r"""Run the scalar level-only backward smoother."""

    cdef cnp.float32_t[:, ::1] dataView = matrixData
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
    cdef double Pf
    cdef double Q
    cdef double PPred
    cdef double J
    cdef double dx
    cdef double xs
    cdef double dP
    cdef double Ps
    cdef double C
    cdef double innov

    if stateSmoothed is not None:
        stateSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateSmoothed
    else:
        stateSmoothedArr = np.empty((intervalCount, 1), dtype=np.float32)
    if stateCovarSmoothed is not None:
        stateCovarSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarSmoothed
    else:
        stateCovarSmoothedArr = np.empty((intervalCount, 1, 1), dtype=np.float32)
    if lagCovSmoothed is not None:
        lagCovSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> lagCovSmoothed
    else:
        lagCovSmoothedArr = np.empty((max(intervalCount - 1, 1), 1, 1), dtype=np.float32)
    if postFitResiduals is not None:
        postFitResidualsArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> postFitResiduals
    else:
        postFitResidualsArr = np.empty((intervalCount, trackCount), dtype=np.float32)

    stateSmoothedView = stateSmoothedArr
    stateCovarSmoothedView = stateCovarSmoothedArr
    lagCovSmoothedView = lagCovSmoothedArr
    postFitResidualsView = postFitResidualsArr

    if intervalCount <= 0:
        return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)

    with nogil:
        stateSmoothedView[intervalCount - 1, 0] = stateForwardView[intervalCount - 1, 0]
        stateCovarSmoothedView[intervalCount - 1, 0, 0] = stateCovarForwardView[intervalCount - 1, 0, 0]

        for j in range(trackCount):
            postFitResidualsView[intervalCount - 1, j] = <cnp.float32_t>(
                (<double>dataView[j, intervalCount - 1]) - (<double>stateSmoothedView[intervalCount - 1, 0])
            )

        for k in range(intervalCount - 2, -1, -1):
            Pf = <double>stateCovarForwardView[k, 0, 0]
            Q = <double>pNoiseForwardView[k, 0, 0]
            PPred = Pf + Q
            if PPred < 1.0e-12:
                PPred = 1.0e-12
            J = Pf / PPred
            dx = (<double>stateSmoothedView[k + 1, 0]) - (<double>stateForwardView[k, 0])
            xs = (<double>stateForwardView[k, 0]) + J * dx
            stateSmoothedView[k, 0] = <cnp.float32_t>xs

            dP = (<double>stateCovarSmoothedView[k + 1, 0, 0]) - PPred
            Ps = Pf + (J * J * dP)
            if Ps < 0.0:
                Ps = 0.0
            stateCovarSmoothedView[k, 0, 0] = <cnp.float32_t>Ps

            C = Pf + (J * dP)
            if k < lagCovSmoothedArr.shape[0]:
                lagCovSmoothedView[k, 0, 0] = <cnp.float32_t>C

            for j in range(trackCount):
                innov = (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                postFitResidualsView[k, j] = <cnp.float32_t>innov

    return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)


cpdef tuple cfixedBackgroundECMLevel(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixPluginMuncInit,
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
    Py_ssize_t t_innerIters=5,
    bint returnIntermediates=False,
    bint returnDiagnostics=False,
    object lambdaExpInit=None,
    object processPrecExpInit=None,
    bint trackOptimizationPath=False,
    object progressBar=None,
    bint logIterations=True,
    object processQScale=None,
):
    r"""Run fixed-background ECM for the scalar level-only process model."""

    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef Py_ssize_t i, k, j, inner
    cdef Py_ssize_t b
    cdef cnp.int32_t[::1] blockMapView = intervalToBlockMap
    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] muncMatView = matrixPluginMuncInit
    cdef cnp.float32_t[:, ::1] q0View = matrixQ0
    cdef object lambdaExp = None
    cdef object processPrecExp = None
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] lambdaExpArr
    cdef cnp.float32_t[::1] lambdaExpView
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processPrecExpArr
    cdef cnp.float32_t[::1] processPrecExpView
    cdef object processQScaleArg = None
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processQScaleArr
    cdef cnp.float32_t[::1] processQScaleView
    cdef bint useProcessQScale = (processQScale is not None)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForward = np.empty((intervalCount, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForward = np.empty((intervalCount, 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForward = np.empty((intervalCount, 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateSmoothed = np.empty((intervalCount, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarSmoothed = np.empty((intervalCount, 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] lagCovSmoothed = np.empty((max(intervalCount - 1, 1), 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] postFitResiduals = np.empty((intervalCount, trackCount), dtype=np.float32)
    cdef cnp.float32_t[:, ::1] stateSmoothedView = stateSmoothed
    cdef cnp.float32_t[:, :, ::1] stateCovarSmoothedView = stateCovarSmoothed
    cdef cnp.float32_t[:, :, ::1] lagCovSmoothedView = lagCovSmoothed
    cdef double q0 = <double>q0View[0, 0]
    cdef double q0Inv
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
    cdef bint hasPreviousNLL = False
    cdef bint converged = False
    cdef double res
    cdef double muncPlusPad
    cdef double p00k
    cdef double Rkj
    cdef double x0, y0
    cdef double Pk, Pk1, Ck_k1
    cdef double delta
    cdef double obsU2
    cdef double w
    cdef double wMin = <double>obsPrecisionMultiplierMin
    cdef double wMax = <double>obsPrecisionMultiplierMax
    cdef double kappa_
    cdef double kappaMin_ = <double>procPrecisionMultiplierMin
    cdef double kappaMax_ = <double>procPrecisionMultiplierMax
    cdef double dState = 1.0
    cdef double procNu = ECM_robustTNu
    cdef Py_ssize_t stableIters = 0
    cdef Py_ssize_t patienceTarget = 2
    cdef bint iterationConverged = False
    cdef object optimizationPath = None

    if trackOptimizationPath:
        optimizationPath = []

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

    if ECM_useProcessPrecisionReweighting and ((not ECM_useAPN) or useProcessQScale):
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

    if useProcessQScale:
        processQScaleArr = _coerceProcessQScale(processQScale, intervalCount)
        processQScaleView = processQScaleArr
        processQScaleArg = processQScaleArr

    if intervalCount <= 5:
        if intervalCount <= 0 or trackCount <= 0:
            currentNLL = 0.0
        else:
            if blockCount <= 0:
                raise ValueError("blockCount must be positive")
            if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
                raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
            if q0 <= 0.0:
                raise ValueError("matrixQ0[0, 0] must be positive")
            _validateMultiplierBounds(wMin, wMax, True)
            _validateMultiplierBounds(kappaMin_, kappaMax_, False)
            if intervalToBlockMap.shape[0] < intervalCount:
                raise ValueError("intervalToBlockMap length must match intervalCount")

            cforwardPassLevel(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixPluginMuncInit,
                matrixQ0=matrixQ0,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=blockCount,
                stateInit=stateInit,
                stateCovarInit=stateCovarInit,
                pad=pad,
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
                processQScale=processQScaleArg,
            )
            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals = cbackwardPassLevel(
                matrixData=matrixData,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                chunkSize=0,
                stateSmoothed=stateSmoothed,
                stateCovarSmoothed=stateCovarSmoothed,
                lagCovSmoothed=lagCovSmoothed,
                postFitResiduals=postFitResiduals,
                progressBar=None,
                progressIter=0,
            )
            currentNLL = (<double>cforwardPassLevel(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixPluginMuncInit,
                matrixQ0=matrixQ0,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=blockCount,
                stateInit=stateInit,
                stateCovarInit=stateCovarInit,
                pad=pad,
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
                processQScale=processQScaleArg,
            )[3])
        previousNLL = currentNLL
        diagnostics = {
            "iters_done": int(0),
            "max_iters": int(ECM_fixedBackgroundIters),
            "converged": False,
            "skipped": True,
            "skip_reason": "too_few_intervals" if intervalCount > 0 else "empty_input",
            "fallback": "filter_smoother_only",
            "stable_iters": int(0),
            "patience_target": int(patienceTarget),
            "initial_nll": float(previousNLL),
            "final_nll": float(previousNLL),
            "final_abs_rel_change": None,
            "final_rel_improvement": None,
            "nll_increase_count": int(0),
        }
        if trackOptimizationPath:
            diagnostics["optimization_path"] = optimizationPath
        if returnIntermediates:
            if returnDiagnostics:
                return (
                    0, float(previousNLL),
                    stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                    lambdaExp, processPrecExp, diagnostics
                )
            return (
                0, float(previousNLL),
                stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                lambdaExp, processPrecExp
            )
        if returnDiagnostics:
            return (0, float(previousNLL), diagnostics)
        return (0, float(previousNLL))

    if blockCount <= 0:
        raise ValueError("blockCount must be positive")
    if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
        raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
    if q0 <= 0.0:
        raise ValueError("matrixQ0[0, 0] must be positive")
    _validateMultiplierBounds(wMin, wMax, True)
    _validateMultiplierBounds(kappaMin_, kappaMax_, False)
    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")

    q0Inv = 1.0 / q0

    for i in range(ECM_fixedBackgroundIters):
        itersDone = i + 1
        if logIterations:
            fprintf(stderr, "\n\t[cfixedBackgroundECMLevel] iter=%zd\n", itersDone)

        for inner in range(t_innerIters):
            cforwardPassLevel(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixPluginMuncInit,
                matrixQ0=matrixQ0,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=blockCount,
                stateInit=stateInit,
                stateCovarInit=stateCovarInit,
                pad=pad,
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
                processQScale=processQScaleArg,
            )

            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals = cbackwardPassLevel(
                matrixData=matrixData,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                chunkSize=0,
                stateSmoothed=stateSmoothed,
                stateCovarSmoothed=stateCovarSmoothed,
                lagCovSmoothed=lagCovSmoothed,
                postFitResiduals=postFitResiduals,
                progressBar=None,
                progressIter=0,
            )

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
                            res = (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                            obsU2 += (res * res + p00k) / Rkj
                        w = ((<double>ECM_robustTNu) + (<double>trackCount)) / ((<double>ECM_robustTNu) + obsU2)
                        if w < wMin:
                            w = wMin
                        elif w > wMax:
                            w = wMax
                        lambdaExpView[k] = <cnp.float32_t>w

            if ECM_useProcessPrecisionReweighting and ((not ECM_useAPN) or useProcessQScale):
                processPrecExpView[0] = <cnp.float32_t>1.0
                for k in range(intervalCount - 1):
                    b = <Py_ssize_t>blockMapView[k]
                    if b < 0 or b >= blockCount:
                        processPrecExpView[k + 1] = <cnp.float32_t>1.0
                        continue
                    x0 = <double>stateSmoothedView[k, 0]
                    y0 = <double>stateSmoothedView[k + 1, 0]
                    Pk = <double>stateCovarSmoothedView[k, 0, 0]
                    Pk1 = <double>stateCovarSmoothedView[k + 1, 0, 0]
                    Ck_k1 = <double>lagCovSmoothedView[k, 0, 0]
                    delta = ((Pk1 + y0 * y0) - (2.0 * (Ck_k1 + x0 * y0)) + (Pk + x0 * x0)) * q0Inv
                    if useProcessQScale:
                        delta = delta / (<double>processQScaleView[k + 1])
                    if delta < 0.0:
                        delta = 0.0
                    kappa_ = ((<double>procNu) + dState) / ((<double>procNu) + delta)
                    if kappa_ < kappaMin_:
                        kappa_ = kappaMin_
                    elif kappa_ > kappaMax_:
                        kappa_ = kappaMax_
                    processPrecExpView[k + 1] = <cnp.float32_t>kappa_

        currentNLL = (<double>cforwardPassLevel(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixPluginMuncInit,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=blockCount,
            stateInit=stateInit,
            stateCovarInit=stateCovarInit,
            pad=pad,
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
            processQScale=processQScaleArg,
        )[3])

        hasPreviousNLL = hasInitialNLL
        if not hasPreviousNLL:
            initialNLL = currentNLL
            hasInitialNLL = True
        elif currentNLL > previousNLL + (1.0e-12 * fmax(fabs(previousNLL), 1.0)):
            nllIncreaseCount += 1

        if hasPreviousNLL:
            nllDelta = fabs(currentNLL - previousNLL)
            nllScale = fabs(previousNLL)
        else:
            nllDelta = 0.0
            nllScale = fabs(currentNLL)
        if fabs(currentNLL) > nllScale:
            nllScale = fabs(currentNLL)
        if nllScale < 1.0:
            nllScale = 1.0
        if hasPreviousNLL:
            relImprovement = (previousNLL - currentNLL) / nllScale
            absRelChange = nllDelta / nllScale
        else:
            relImprovement = 0.0
            absRelChange = 0.0
        nllTol = (<double>ECM_fixedBackgroundRtol) * nllScale
        previousNLL = currentNLL
        if logIterations:
            fprintf(
                stderr,
                "\t[cfixedBackgroundECMLevel] NLL=%.6f  REL=%+.6e  ABSREL=%.6e  THRESH=%.6e\n",
                currentNLL,
                relImprovement,
                absRelChange,
                nllTol,
            )
        if hasPreviousNLL and nllDelta <= nllTol:
            stableIters += 1
        else:
            stableIters = 0
        if logIterations:
            fprintf(
                stderr,
                "\t[cfixedBackgroundECMLevel] stable=%zd/%zd\n",
                stableIters, patienceTarget
            )
        iterationConverged = stableIters >= patienceTarget
        if progressBar is not None:
            progressBar.set_postfix_str(
                (
                    f"NLL={currentNLL:.6g} rel={relImprovement:+.2e} "
                    f"stable={int(stableIters)}/{int(patienceTarget)}"
                ),
                refresh=False,
            )
            progressBar.update(1)
        if trackOptimizationPath:
            optimizationPath.append({
                "iter": int(itersDone),
                "objective_name": "nll",
                "objective_value": float(currentNLL),
                "change": float(nllDelta) if hasPreviousNLL else None,
                "relative_improvement": (
                    float(relImprovement) if hasPreviousNLL else None
                ),
                "abs_relative_change": (
                    float(absRelChange) if hasPreviousNLL else None
                ),
                "threshold": float(nllTol) if hasPreviousNLL else None,
                "stable_iters": int(stableIters),
                "patience_target": int(patienceTarget),
                "reset_iteration": bool(not hasPreviousNLL),
                "converged": bool(iterationConverged),
            })
        if iterationConverged:
            converged = True
            if logIterations:
                fprintf(stderr, "\t[cfixedBackgroundECMLevel] CONVERGED (ECM) iter=%zd \n", itersDone)
            if progressBar is not None:
                progressBar.set_postfix_str(f"converged iter={int(itersDone)}")
            break

    diagnostics = {
        "iters_done": int(itersDone),
        "max_iters": int(ECM_fixedBackgroundIters),
        "converged": bool(converged),
        "skipped": False,
        "skip_reason": None,
        "fallback": None,
        "stable_iters": int(stableIters),
        "patience_target": int(patienceTarget),
        "initial_nll": float(initialNLL) if hasInitialNLL else None,
        "final_nll": float(previousNLL),
        "final_abs_rel_change": float(absRelChange) if hasInitialNLL else None,
        "final_rel_improvement": float(relImprovement) if hasInitialNLL else None,
        "nll_increase_count": int(nllIncreaseCount),
    }
    if trackOptimizationPath:
        diagnostics["optimization_path"] = optimizationPath

    if returnIntermediates:
        if returnDiagnostics:
            return (
                itersDone, float(previousNLL),
                stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                lambdaExp, processPrecExp, diagnostics
            )
        return (
            itersDone, float(previousNLL),
            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
            lambdaExp, processPrecExp
        )
    if returnDiagnostics:
        return (itersDone, float(previousNLL), diagnostics)
    return (itersDone, float(previousNLL))


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
    Py_ssize_t t_innerIters=5,
    bint returnIntermediates=False,
    bint returnDiagnostics=False,
    object lambdaExpInit=None,
    object processPrecExpInit=None,
    bint trackOptimizationPath=False,
    object progressBar=None,
    bint logIterations=True,
    object processQScale=None,
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

    Here :math:`\lambda_{[i]}` and :math:`\kappa_{[i]}` are Student-t precision multipliers.


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
          \frac{(z_{[j,i]}-\widetilde{x}_{[i,0]})^2+\widetilde{P}_{[i,0,0]}}
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

    Objective Function
    ----------------------------------

    Let :math:`x_{1:n}=\{\mathbf{x}_{[i]}\}_{i=1}^n`, :math:`\lambda=\{\lambda_{[i]}\}`, and
    :math:`\kappa=\{\kappa_{[i]}\}`. Collecting process and observation terms and mixing penalties yields:

    .. math::
      :nowrap:

        \begin{align}
        \mathcal{J}(x,\Lambda,\kappa)
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
        (z_{[j,i]}-x_{[i,0]})^2\,\frac{\lambda_{[i]}}{v_{[j,i]}}
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
    solves the quadratic subproblem *conditional* on the current estimates of :math:`\lambda` and :math:`\kappa`,
    and reweighting optimizes over :math:`\lambda` and :math:`\kappa`.

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
    :returns: A tuple ``(itersDone, finalNLL)``. If
            ``returnIntermediates=True``, additionally returns
            ``(stateSmoothed, stateCovarSmoothed, lagCovSmoothed,
            postFitResiduals, lambdaExp, processPrecExp)``.
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

    # Allocate latent precision multipliers only if enabled
    cdef object lambdaExp = None
    cdef object processPrecExp = None
    cdef object processQScaleArg = None
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] lambdaExpArr
    cdef cnp.float32_t[::1] lambdaExpView
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processPrecExpArr
    cdef cnp.float32_t[::1] processPrecExpView
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processQScaleArr
    cdef cnp.float32_t[::1] processQScaleView
    cdef bint useProcessQScale = (processQScale is not None)

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

    if ECM_useProcessPrecisionReweighting and ((not ECM_useAPN) or useProcessQScale):
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

    if useProcessQScale:
        processQScaleArr = _coerceProcessQScale(processQScale, intervalCount)
        processQScaleView = processQScaleArr
        processQScaleArg = processQScaleArr

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
    cdef bint hasPreviousNLL = False
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
    cdef Py_ssize_t stableIters = 0
    cdef Py_ssize_t patienceTarget = 2
    cdef bint iterationConverged = False
    cdef object optimizationPath = None

    if trackOptimizationPath:
        optimizationPath = []

    if intervalCount <= 5:
        if intervalCount <= 0 or trackCount <= 0:
            currentNLL = 0.0
        else:
            if blockCount <= 0:
                raise ValueError("blockCount must be positive")
            _validateMultiplierBounds(wMin, wMax, True)
            _validateMultiplierBounds(kappaMin_, kappaMax_, False)
            if intervalToBlockMap.shape[0] < intervalCount:
                raise ValueError("intervalToBlockMap length must match intervalCount")
            if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
                raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
            if detQ0 == 0.0:
                raise ValueError("matrixQ0 is singular")

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
                processQScale=processQScaleArg,
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
                progressBar=None,
                progressIter=0,
            )
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
                processQScale=processQScaleArg,
            )[3])
        previousNLL = currentNLL
        diagnostics = {
            "iters_done": int(0),
            "max_iters": int(ECM_fixedBackgroundIters),
            "converged": False,
            "skipped": True,
            "skip_reason": "too_few_intervals" if intervalCount > 0 else "empty_input",
            "fallback": "filter_smoother_only",
            "stable_iters": int(0),
            "patience_target": int(patienceTarget),
            "initial_nll": float(previousNLL),
            "final_nll": float(previousNLL),
            "final_abs_rel_change": None,
            "final_rel_improvement": None,
            "nll_increase_count": int(0),
        }
        if trackOptimizationPath:
            diagnostics["optimization_path"] = optimizationPath
        if returnIntermediates:
            if returnDiagnostics:
                return (
                    0, float(previousNLL),
                    stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                    lambdaExp, processPrecExp, diagnostics
                )
            return (
                0, float(previousNLL),
                stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                lambdaExp, processPrecExp
            )
        if returnDiagnostics:
            return (0, float(previousNLL), diagnostics)
        return (0, float(previousNLL))

    if blockCount <= 0:
        raise ValueError("blockCount must be positive")
    _validateMultiplierBounds(wMin, wMax, True)
    _validateMultiplierBounds(kappaMin_, kappaMax_, False)
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

    for i in range(ECM_fixedBackgroundIters):
        itersDone = i + 1
        if logIterations:
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
                processQScale=processQScaleArg,
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

                            res = (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
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
            if ECM_useProcessPrecisionReweighting and ((not ECM_useAPN) or useProcessQScale):
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
                    if useProcessQScale:
                        delta = delta / (<double>processQScaleView[k + 1])
                    if delta < 0.0:
                        delta = 0.0

                    kappa_ = ((<double>procNu) + dState) / ((<double>procNu) + delta)
                    if kappa_ < kappaMin_:
                        kappa_ = kappaMin_
                    elif kappa_ > kappaMax_:
                        kappa_ = kappaMax_
                    processPrecExpView[k + 1] = <cnp.float32_t>kappa_

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
            processQScale=processQScaleArg,
        )[3])

        hasPreviousNLL = hasInitialNLL
        if not hasPreviousNLL:
            initialNLL = currentNLL
            hasInitialNLL = True
        elif currentNLL > previousNLL + (1.0e-12 * fmax(fabs(previousNLL), 1.0)):
            nllIncreaseCount += 1

        if hasPreviousNLL:
            nllDelta = fabs(currentNLL - previousNLL)
            nllScale = fabs(previousNLL)
        else:
            nllDelta = 0.0
            nllScale = fabs(currentNLL)
        if fabs(currentNLL) > nllScale:
            nllScale = fabs(currentNLL)
        if nllScale < 1.0:
            nllScale = 1.0
        if hasPreviousNLL:
            relImprovement = (previousNLL - currentNLL) / nllScale
            absRelChange = nllDelta / nllScale
        else:
            relImprovement = 0.0
            absRelChange = 0.0
        nllTol = (<double>ECM_fixedBackgroundRtol) * nllScale
        previousNLL = currentNLL
        if logIterations:
            fprintf(
                stderr,
                "\t[cfixedBackgroundECM] NLL=%.6f  REL=%+.6e  ABSREL=%.6e  THRESH=%.6e\n",
                currentNLL,
                relImprovement,
                absRelChange,
                nllTol,
            )

        if hasPreviousNLL and nllDelta <= nllTol:
            stableIters += 1
        else:
            stableIters = 0

        if logIterations:
            fprintf(
                stderr,
                "\t[cfixedBackgroundECM] stable=%zd/%zd\n",
                stableIters, patienceTarget
            )

        iterationConverged = stableIters >= patienceTarget
        if progressBar is not None:
            progressBar.set_postfix_str(
                (
                    f"NLL={currentNLL:.6g} rel={relImprovement:+.2e} "
                    f"stable={int(stableIters)}/{int(patienceTarget)}"
                ),
                refresh=False,
            )
            progressBar.update(1)
        if trackOptimizationPath:
            optimizationPath.append({
                "iter": int(itersDone),
                "objective_name": "nll",
                "objective_value": float(currentNLL),
                "change": float(nllDelta) if hasPreviousNLL else None,
                "relative_improvement": (
                    float(relImprovement) if hasPreviousNLL else None
                ),
                "abs_relative_change": (
                    float(absRelChange) if hasPreviousNLL else None
                ),
                "threshold": float(nllTol) if hasPreviousNLL else None,
                "stable_iters": int(stableIters),
                "patience_target": int(patienceTarget),
                "reset_iteration": bool(not hasPreviousNLL),
                "converged": bool(iterationConverged),
            })
        if iterationConverged:
            converged = True
            if logIterations:
                fprintf(stderr, "\t[cfixedBackgroundECM] CONVERGED (ECM) iter=%zd \n", itersDone)
            if progressBar is not None:
                progressBar.set_postfix_str(f"converged iter={int(itersDone)}")
            break

    diagnostics = {
        "iters_done": int(itersDone),
        "max_iters": int(ECM_fixedBackgroundIters),
        "converged": bool(converged),
        "skipped": False,
        "skip_reason": None,
        "fallback": None,
        "stable_iters": int(stableIters),
        "patience_target": int(patienceTarget),
        "initial_nll": float(initialNLL) if hasInitialNLL else None,
        "final_nll": float(previousNLL),
        "final_abs_rel_change": float(absRelChange) if hasInitialNLL else None,
        "final_rel_improvement": float(relImprovement) if hasInitialNLL else None,
        "nll_increase_count": int(nllIncreaseCount),
    }
    if trackOptimizationPath:
        diagnostics["optimization_path"] = optimizationPath

    if returnIntermediates:
        if returnDiagnostics:
            return (
                itersDone, float(previousNLL),
                stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
                lambdaExp, processPrecExp, diagnostics
            )
        return (
            itersDone, float(previousNLL),
            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals,
            lambdaExp, processPrecExp
        )

    if returnDiagnostics:
        return (itersDone, float(previousNLL), diagnostics)
    return (itersDone, float(previousNLL))


cpdef cnp.ndarray[cnp.float32_t, ndim=1] crollingMuncVariance(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    int blockLength,
    cnp.ndarray[cnp.uint8_t, ndim=1] excludeMask,
    double maxBeta=0.95,
    double pairsRegLambda = 1.0,
    bint useInnovationVar = <bint>True,
):
    r"""Estimate a rolling AR(1) MUNC variance track for a 1D array of values."""

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
    cdef double sumSqXSeq
    cdef double sumSqYSeq
    cdef double sumXYc
    cdef double previousValue
    cdef double currentValue
    cdef double leavingValue
    cdef double enteringValue

    varOut = np.empty(numIntervals,dtype=np.float32)

    if blockLength > numIntervals:
        blockLength = <int>numIntervals

    if blockLength < 2:
        varOut[:] = 0.0
        return varOut

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

                sumSqXSeq = (
                    (sumSqY - (currentValue * currentValue))
                    - ((sumXSeq * sumXSeq) / nPairsDouble)
                )
                sumSqYSeq = (
                    (sumSqY - (previousValue * previousValue))
                    - ((sumYSeq * sumYSeq) / nPairsDouble)
                )
                sumXYc = sumLagProd - ((sumXSeq * sumYSeq) / nPairsDouble)
                varAtView[startIndex] = <cnp.float32_t>(
                    _canonicalAR1VarianceFromPairStats(
                        sumSqXSeq,
                        sumSqYSeq,
                        sumXYc,
                        blockLength,
                        useInnovationVar,
                        maxBeta,
                        pairsRegLambda,
                    )
                )

            if startIndex < maxStartIndex:
                # slide window forward --> (previousSum - leavingValue) + enteringValue
                leavingValue = valuesView[startIndex]
                enteringValue = valuesView[(startIndex + blockLength)]
                sumY = (sumY-leavingValue) + enteringValue
                sumSqY = sumSqY + (-(leavingValue*leavingValue) + (enteringValue*enteringValue))
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


cpdef cnp.ndarray[cnp.float32_t, ndim=1] crollingMuncAR1Beta(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    int blockLength,
    cnp.ndarray[cnp.uint8_t, ndim=1] excludeMask,
    double maxBeta=0.95,
    double pairsRegLambda = 1.0,
):
    r"""Estimate the clipped AR(1) beta used by rolling MUNC windows."""

    cdef Py_ssize_t numIntervals=values.shape[0]
    cdef Py_ssize_t regionIndex, elementIndex, startIndex, maxStartIndex
    cdef int halfBlockLength, maskSum
    cdef cnp.ndarray[cnp.float32_t, ndim=1] betaAtStartIndex
    cdef cnp.ndarray[cnp.float32_t, ndim=1] betaOut
    cdef float[::1] valuesView=values
    cdef cnp.uint8_t[::1] maskView=excludeMask
    cdef float[::1] betaAtView
    cdef float[::1] betaOutView
    cdef double sumY
    cdef double sumSqY
    cdef double sumLagProd
    cdef double nPairsDouble
    cdef double sumXSeq
    cdef double sumYSeq
    cdef double sumSqXSeq
    cdef double sumSqYSeq
    cdef double sumXYc
    cdef double previousValue
    cdef double currentValue
    cdef double leavingValue
    cdef double enteringValue

    betaOut = np.empty(numIntervals, dtype=np.float32)

    if blockLength > numIntervals:
        blockLength = <int>numIntervals
    if blockLength < 4:
        betaOut[:] = 0.0
        return betaOut

    halfBlockLength = blockLength // 2
    maxStartIndex = numIntervals - blockLength
    betaAtStartIndex = np.empty((maxStartIndex + 1), dtype=np.float32)
    betaAtView = betaAtStartIndex
    betaOutView = betaOut

    sumY = 0.0
    sumSqY = 0.0
    sumLagProd = 0.0
    maskSum = 0

    with nogil:
        for elementIndex in range(blockLength):
            currentValue = valuesView[elementIndex]
            sumY += currentValue
            sumSqY += currentValue * currentValue
            maskSum += <int>maskView[elementIndex]
            if elementIndex < (blockLength - 1):
                sumLagProd += currentValue * valuesView[elementIndex + 1]

        for startIndex in range(maxStartIndex + 1):
            if maskSum != 0:
                betaAtView[startIndex] = <cnp.float32_t>-1.0
            else:
                nPairsDouble = <double>(blockLength - 1)
                previousValue = valuesView[startIndex]
                currentValue = valuesView[startIndex + blockLength - 1]
                sumXSeq = sumY - currentValue
                sumYSeq = sumY - previousValue
                sumSqXSeq = (
                    (sumSqY - (currentValue * currentValue))
                    - ((sumXSeq * sumXSeq) / nPairsDouble)
                )
                sumSqYSeq = (
                    (sumSqY - (previousValue * previousValue))
                    - ((sumYSeq * sumYSeq) / nPairsDouble)
                )
                sumXYc = sumLagProd - ((sumXSeq * sumYSeq) / nPairsDouble)
                betaAtView[startIndex] = <cnp.float32_t>(
                    _canonicalAR1BetaFromPairStats(
                        sumSqXSeq,
                        sumSqYSeq,
                        sumXYc,
                        blockLength,
                        maxBeta,
                        pairsRegLambda,
                    )
                )

            if startIndex < maxStartIndex:
                leavingValue = valuesView[startIndex]
                enteringValue = valuesView[startIndex + blockLength]
                sumY = (sumY - leavingValue) + enteringValue
                sumSqY = sumSqY + (-(leavingValue * leavingValue) + (enteringValue * enteringValue))
                sumLagProd = sumLagProd + (
                    -(valuesView[startIndex] * valuesView[startIndex + 1])
                    + (valuesView[startIndex + blockLength - 1] * valuesView[startIndex + blockLength])
                )
                maskSum = maskSum + (-<int>maskView[startIndex] + <int>maskView[startIndex + blockLength])

        for regionIndex in range(numIntervals):
            startIndex = regionIndex - halfBlockLength
            if startIndex < 0:
                startIndex = 0
            elif startIndex > maxStartIndex:
                startIndex = maxStartIndex
            betaOutView[regionIndex] = betaAtView[startIndex]

    return betaOut


cpdef cnp.ndarray[cnp.float32_t, ndim=1] crolling_AR1_IVar(
    cnp.ndarray[cnp.float32_t, ndim=1] values,
    int blockLength,
    cnp.ndarray[cnp.uint8_t, ndim=1] excludeMask,
    double maxBeta=0.95,
    double pairsRegLambda = 1.0,
    bint useInnovationVar = <bint>True,
):
    r"""Compatibility wrapper for the AR(1) rolling MUNC variance model."""
    return crollingMuncVariance(
        values,
        blockLength,
        excludeMask,
        maxBeta=maxBeta,
        pairsRegLambda=pairsRegLambda,
        useInnovationVar=useInnovationVar,
    )

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
    # FFR: consider coupling minRefDist with sampled dependence span sizing
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


# ---------------------------------------------------------------------------
# Optional fast helper kernels used by Python compatibility wrappers.
# ---------------------------------------------------------------------------

cdef inline double _transformDerivativeAtMean_F64(
    double x,
    int mode,
    double inputOffset,
    double inputScale,
    double outputScale,
    double shape,
) noexcept nogil:
    cdef double tiny = 2.2250738585072014e-308
    cdef double shifted = x + inputOffset
    cdef double u
    cdef double root
    if shifted < tiny:
        shifted = tiny
    if mode == __TRANSFORM_MODE_LOG:
        return outputScale / shifted
    if mode == __TRANSFORM_MODE_SQRT or mode == __TRANSFORM_MODE_ANSCOMBE:
        return outputScale / (2.0 * inputScale * sqrt(fmax(shifted / inputScale, tiny)))
    if mode == __TRANSFORM_MODE_ASINH:
        u = shifted / inputScale
        return outputScale / (inputScale * sqrt(1.0 + u * u))
    if mode == __TRANSFORM_MODE_ASINH_SQRT:
        root = sqrt(shifted)
        u = root / inputScale
        return outputScale / (2.0 * inputScale * root * sqrt(1.0 + u * u))
    if mode == __TRANSFORM_MODE_GENERALIZED_LOG:
        u = shifted / inputScale
        return outputScale / (inputScale * sqrt(u * u + shape * shape))
    return outputScale / inputScale


def cTransformCountVarianceFloor(
    object normalizedCounts,
    object scaleFactors,
    object mode=None,
    object transformMethod=None,
    object logOffset=1.0,
    object logMult=1.0,
    object inputOffset=None,
    object inputScale=None,
    object outputScale=None,
    object shape=None,
    object transformInputOffset=None,
    object transformInputScale=None,
    object transformOutputScale=None,
    object transformShape=None,
):
    r"""Delta-method count-transform variance floor with a nogil inner loop."""
    cdef object countsObj = np.asarray(normalizedCounts, dtype=np.float64)
    cdef bint squeeze = False
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] counts2
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] scales
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out
    cdef object selectedMode = mode if mode is not None else transformMethod
    cdef object selectedInputOffset = inputOffset if inputOffset is not None else transformInputOffset
    cdef object selectedInputScale = inputScale if inputScale is not None else transformInputScale
    cdef object selectedOutputScale = outputScale if outputScale is not None else transformOutputScale
    cdef object selectedShape = shape if shape is not None else transformShape
    cdef int modeCode = _parseTransformMode(selectedMode)
    cdef tuple params = _resolveTransformParameters(
        modeCode,
        float(logOffset),
        float(logMult),
        None,
        None,
        selectedInputOffset,
        selectedInputScale,
        selectedOutputScale,
        None,
        selectedShape,
    )
    cdef double inputOffset_ = <double>params[0]
    cdef double inputScale_ = <double>params[1]
    cdef double outputScale_ = <double>params[2]
    cdef double shape_ = <double>params[4]
    cdef Py_ssize_t m, n, i, j
    cdef double count, sf, rawMean, normalizedMean, normalizedVariance, deriv, floorValue

    if countsObj.ndim == 1:
        squeeze = True
        counts2 = np.ascontiguousarray(np.asarray(countsObj, dtype=np.float64).reshape(1, -1), dtype=np.float64)
    elif countsObj.ndim == 2:
        counts2 = np.ascontiguousarray(countsObj, dtype=np.float64)
    else:
        raise ValueError("normalizedCounts must be a 1D or 2D array")

    m = counts2.shape[0]
    n = counts2.shape[1]
    scales = np.ascontiguousarray(np.asarray(scaleFactors, dtype=np.float64).reshape(-1), dtype=np.float64)
    if scales.shape[0] == 1 and m != 1:
        scales = np.ascontiguousarray(np.full(m, float(scales[0]), dtype=np.float64), dtype=np.float64)
    if scales.shape[0] != m:
        raise ValueError("scaleFactors must contain one value per count track")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("scaleFactors must be finite positive values")

    out = np.empty((m, n), dtype=np.float32)
    with nogil:
        for i in range(m):
            sf = <double>scales[i]
            for j in range(n):
                count = <double>counts2[i, j]
                if not isfinite(count):
                    out[i, j] = <float>NAN
                    continue
                if count < 0.0:
                    count = 0.0
                rawMean = (count / sf) + 0.5
                normalizedMean = rawMean * sf
                normalizedVariance = rawMean * sf * sf
                deriv = _transformDerivativeAtMean_F64(
                    normalizedMean,
                    modeCode,
                    inputOffset_,
                    inputScale_,
                    outputScale_,
                    shape_,
                )
                floorValue = deriv * deriv * normalizedVariance
                if isfinite(floorValue) and floorValue > 0.0:
                    out[i, j] = <float>floorValue
                else:
                    out[i, j] = <float>NAN
    if squeeze:
        return np.asarray(out[0, :], dtype=np.float32)
    return out


def cTuncObservationInformation(
    object matrixMunc,
    double pad,
    object lambdaExp=None,
    double observationPrecisionMultiplierMin=1.0,
    double observationPrecisionMultiplierMax=1.0,
):
    r"""Interval information reduction for TUNC with a nogil inner loop."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] munc = np.ascontiguousarray(matrixMunc, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] lam
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] info
    cdef Py_ssize_t m, n, i, j
    cdef double v, sumInfo, lamj
    if munc.ndim != 2:
        raise ValueError("matrixMunc must be a 2D array")
    m = munc.shape[0]
    n = munc.shape[1]
    if lambdaExp is None:
        lam = np.ascontiguousarray(np.ones(n, dtype=np.float64), dtype=np.float64)
    else:
        lam = np.ascontiguousarray(np.asarray(lambdaExp, dtype=np.float64).reshape(-1), dtype=np.float64)
        if lam.shape[0] != n:
            raise ValueError("lambdaExp length must match interval count")
        lam = np.ascontiguousarray(np.nan_to_num(lam, nan=1.0, posinf=1.0, neginf=1.0), dtype=np.float64)
        lam = np.ascontiguousarray(np.clip(lam, observationPrecisionMultiplierMin, observationPrecisionMultiplierMax), dtype=np.float64)
    info = np.zeros(n, dtype=np.float64)
    with nogil:
        for j in range(n):
            lamj = <double>lam[j]
            sumInfo = 0.0
            for i in range(m):
                v = <double>munc[i, j]
                if isfinite(v):
                    v = v + pad
                    if isfinite(v) and v > 0.0:
                        if v < 1.0e-12:
                            v = 1.0e-12
                        sumInfo += lamj / v
            if sumInfo > 0.0 and isfinite(sumInfo):
                info[j] = sumInfo
            else:
                info[j] = 0.0
    return info


def cMovingAverageSame(object values, int window):
    r"""Same-length moving average using cumulative sums."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1), dtype=np.float64)
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t w = max(int(window), 1)
    cdef Py_ssize_t leftPad, rightPad, paddedN, i
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] padded
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] csum
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out
    if w <= 1 or n <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    if w > n:
        w = n
    leftPad = (w - 1) // 2
    rightPad = w - 1 - leftPad
    paddedN = n + leftPad + rightPad
    padded = np.zeros(paddedN, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        padded[leftPad + i] = x[i]
    csum = np.empty(paddedN + 1, dtype=np.float64)
    csum[0] = 0.0
    with nogil:
        for i in range(paddedN):
            csum[i + 1] = csum[i] + padded[i]
        for i in range(n):
            out[i] = (csum[i + w] - csum[i]) / <double>w
    return out


def cEstimateEffectiveSampleSize(object values, int maxLag):
    r"""Positive-autocorrelation effective sample size scan."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1), dtype=np.float64)
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i, lag, maxLag_, lagsUsed = 0
    cdef double mean = 0.0
    cdef double var = 0.0
    cdef double cov, rho, tau = 1.0
    if n < 2:
        return float(n), 1.0, 0
    for i in range(n):
        mean += x[i]
    mean /= <double>n
    with nogil:
        for i in range(n):
            x[i] = x[i] - mean
            var += x[i] * x[i]
    var /= <double>max(n, 1)
    if (not isfinite(var)) or var <= 2.2250738585072014e-308:
        return float(n), 1.0, 0
    maxLag_ = max(1, min(int(maxLag), n - 1))
    with nogil:
        for lag in range(1, maxLag_ + 1):
            cov = 0.0
            for i in range(n - lag):
                cov += x[i] * x[i + lag]
            cov /= <double>max(n - lag, 1)
            rho = cov / var
            if (not isfinite(rho)) or rho <= 0.0:
                break
            tau += 2.0 * rho
            lagsUsed = lag
    if tau < 1.0:
        tau = 1.0
    return float(n / tau), float(tau), int(lagsUsed)

# Additional ROCCO/DWB helper kernels added during the runtime cleanup pass.

cdef int _dwbKernelCodeRefactor(object kernel) except -1:
    cdef str name = str(kernel).strip().lower().replace("-", "_")
    if name == "bartlett" or name == "triangle" or name == "triangular":
        return 0
    if name == "parzen":
        return 1
    if name == "qs" or name == "quadratic_spectral" or name == "quadraticspectral":
        return 2
    raise ValueError(f"Unknown DWB kernel: {kernel}")


cdef inline int _dwbMaxLagRefactor(int bandwidth, int kernelCode) noexcept nogil:
    cdef int bw = bandwidth if bandwidth >= 2 else 2
    cdef int lag
    if kernelCode == 2:
        lag = 8 * bw
        if lag < 32:
            lag = 32
        return lag
    return bw


cdef inline double _dwbKernelValueRefactor(int kernelCode, long lag, int bandwidth) noexcept nogil:
    cdef double bw = <double>(bandwidth if bandwidth >= 1 else 1)
    cdef double ax = fabs(<double>lag) / bw
    cdef double y
    if kernelCode == 0:
        if ax <= 1.0:
            return 1.0 - ax
        return 0.0
    if kernelCode == 1:
        if ax <= 0.5:
            return 1.0 - 6.0 * ax * ax + 6.0 * ax * ax * ax
        if ax <= 1.0:
            return 2.0 * (1.0 - ax) * (1.0 - ax) * (1.0 - ax)
        return 0.0
    if ax < 1.0e-12:
        return 1.0
    y = (6.0 * __PI_DOUBLE * ax) / 5.0
    return (25.0 / (12.0 * __PI_DOUBLE * __PI_DOUBLE * ax * ax)) * ((sin(y) / fmax(y, 1.0e-12)) - cos(y))


cpdef object cGenerateDWBMultipliersFromNoise(object noise, int bandwidth, object kernel="bartlett"):
    """Generate standardized dependent wild-bootstrap multipliers from supplied Gaussian noise."""
    cdef int bw = bandwidth if bandwidth >= 2 else 2
    cdef int kernelCode = _dwbKernelCodeRefactor(kernel)
    cdef int maxLag = _dwbMaxLagRefactor(bw, kernelCode)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] noiseArr = np.ascontiguousarray(
        np.asarray(noise, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef Py_ssize_t n = noiseArr.shape[0] - 2 * maxLag
    cdef Py_ssize_t weightCount = 2 * maxLag + 1
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weights
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out
    cdef double[::1] noiseView
    cdef double[::1] weightView
    cdef double[::1] outView
    cdef Py_ssize_t i, j
    cdef long lag
    cdef double normSq = 0.0
    cdef double norm, value, meanValue = 0.0, varSum = 0.0, sd
    if n <= 0:
        raise ValueError("noise length is too short for the requested DWB bandwidth")
    weights = np.empty(weightCount, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    noiseView = noiseArr
    weightView = weights
    outView = out
    with nogil:
        for j in range(weightCount):
            lag = <long>j - <long>maxLag
            value = _dwbKernelValueRefactor(kernelCode, lag, bw)
            weightView[j] = value
            normSq += value * value
        norm = sqrt(fmax(normSq, 2.2250738585072014e-308))
        for j in range(weightCount):
            weightView[j] = weightView[j] / norm
        for i in range(n):
            value = 0.0
            for j in range(weightCount):
                value += noiseView[i + j] * weightView[j]
            outView[i] = value
            meanValue += value
        meanValue = meanValue / <double>n
        if n >= 2:
            for i in range(n):
                value = outView[i] - meanValue
                varSum += value * value
            sd = sqrt(varSum / <double>(n - 1))
        else:
            sd = 0.0
        if (not isfinite(sd)) or sd <= 2.2250738585072014e-308:
            for i in range(n):
                outView[i] = 1.0
        else:
            for i in range(n):
                outView[i] = (outView[i] - meanValue) / sd
    return out


cpdef object cApplyStationaryNullDWB(object template, object multipliers):
    """Apply DWB multipliers to a template and subtract the draw mean."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] templateArr = np.ascontiguousarray(
        np.asarray(template, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] multArr = np.ascontiguousarray(
        np.asarray(multipliers, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef Py_ssize_t n = templateArr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out
    cdef double[::1] templateView
    cdef double[::1] multView
    cdef double[::1] outView
    cdef Py_ssize_t i
    cdef double meanValue = 0.0
    if multArr.shape[0] != n:
        raise ValueError("template and multipliers must have the same length")
    out = np.empty(n, dtype=np.float64)
    templateView = templateArr
    multView = multArr
    outView = out
    with nogil:
        for i in range(n):
            outView[i] = templateView[i] * multView[i]
            meanValue += outView[i]
        if n > 0:
            meanValue = meanValue / <double>n
            for i in range(n):
                outView[i] = outView[i] - meanValue
    return out


cpdef tuple cBooleanRunBounds(object above, int maxGapBins=0):
    """Return start/end arrays for true-runs, optionally bridging small false gaps."""
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] arr = np.ascontiguousarray(
        np.asarray(above, dtype=np.uint8).reshape(-1), dtype=np.uint8
    )
    cdef Py_ssize_t n = arr.shape[0]
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] starts = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] ends = np.empty(n, dtype=np.int64)
    cdef uint8_t[::1] arrView = arr
    cdef int64_t[::1] startsView = starts
    cdef int64_t[::1] endsView = ends
    cdef Py_ssize_t i, outCount = 0
    cdef Py_ssize_t runStart = -1
    cdef Py_ssize_t lastTrue = -1
    cdef int maxGap = maxGapBins if maxGapBins > 0 else 0
    with nogil:
        for i in range(n):
            if arrView[i] != 0:
                if runStart < 0:
                    runStart = i
                elif i - lastTrue > maxGap + 1:
                    startsView[outCount] = runStart
                    endsView[outCount] = lastTrue
                    outCount += 1
                    runStart = i
                lastTrue = i
        if runStart >= 0:
            startsView[outCount] = runStart
            endsView[outCount] = lastTrue
            outCount += 1
    return starts[:outCount].copy(), ends[:outCount].copy()

# ---------------------------------------------------------------------------
# Additional lowercase compatibility kernels for Python runtime fast paths.
# ---------------------------------------------------------------------------

def cbackgroundWeightedStats(object residualMatrix, object invVarMatrix):
    r"""Column-wise background sufficient statistics with a nogil inner loop."""
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] residualArr = np.ascontiguousarray(residualMatrix, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] invArr = np.ascontiguousarray(invVarMatrix, dtype=np.float32)
    if residualArr.ndim != 2 or invArr.shape[0] != residualArr.shape[0] or invArr.shape[1] != residualArr.shape[1]:
        raise ValueError("residualMatrix and invVarMatrix must have identical 2D shapes")
    cdef Py_ssize_t m = residualArr.shape[0]
    cdef Py_ssize_t n = residualArr.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weightArr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rhsArr = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i, j
    cdef double wsum, rsum, w
    with nogil:
        for i in range(n):
            wsum = 0.0
            rsum = 0.0
            for j in range(m):
                w = <double>invArr[j, i]
                wsum += w
                rsum += w * <double>residualArr[j, i]
            weightArr[i] = wsum
            rhsArr[i] = rsum
    return weightArr, rhsArr


def ctuncObservationInformation(
    object matrixMunc,
    double pad,
    object lambdaExp=None,
    double observationPrecisionMultiplierMin=1.0,
    double observationPrecisionMultiplierMax=1.0,
):
    return cTuncObservationInformation(
        matrixMunc,
        pad,
        lambdaExp=lambdaExp,
        observationPrecisionMultiplierMin=observationPrecisionMultiplierMin,
        observationPrecisionMultiplierMax=observationPrecisionMultiplierMax,
    )


def crebaseTuncIntervalScales(object seedQ, object baseQ, object rawScale, object stateModel):
    r"""Rebase transition scalar Q scales from seed-Q units to final base-Q units."""
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] seedArr = np.ascontiguousarray(seedQ, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] baseArr = np.ascontiguousarray(baseQ, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rawArr = np.ascontiguousarray(np.asarray(rawScale, dtype=np.float64).reshape(-1), dtype=np.float64)
    cdef int dim = 1 if str(stateModel) == "level" else 2
    if seedArr.shape[0] < dim or seedArr.shape[1] < dim or baseArr.shape[0] < dim or baseArr.shape[1] < dim:
        raise ValueError("seedQ/baseQ must cover the active state dimension")
    cdef Py_ssize_t n = rawArr.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] scaleArr = np.ones(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] errArr = np.empty(max(n * dim, 1), dtype=np.float64)
    cdef Py_ssize_t i, errCount = 0
    cdef int d, validCount
    cdef double scalar, targetDiag, baseDiag, ratio, logSum, localScale, recomposed, absErr, maxErr = 0.0
    with nogil:
        for i in range(n):
            scalar = rawArr[i]
            if (not isfinite(scalar)) or scalar < 1.0e-12:
                scalar = 1.0e-12
            logSum = 0.0
            validCount = 0
            for d in range(dim):
                targetDiag = seedArr[d, d] * scalar
                baseDiag = baseArr[d, d]
                if isfinite(targetDiag) and isfinite(baseDiag) and targetDiag > 0.0 and baseDiag > 0.0:
                    ratio = targetDiag / baseDiag
                    if ratio < 1.0e-300:
                        ratio = 1.0e-300
                    logSum += log(ratio)
                    validCount += 1
            localScale = 1.0 if validCount <= 0 else exp(logSum / <double>validCount)
            scaleArr[i] = <float>localScale
            for d in range(dim):
                targetDiag = seedArr[d, d] * scalar
                baseDiag = baseArr[d, d]
                if isfinite(targetDiag) and isfinite(baseDiag) and targetDiag > 0.0 and baseDiag > 0.0:
                    recomposed = baseDiag * localScale
                    if recomposed < 1.0e-300:
                        recomposed = 1.0e-300
                    if targetDiag < 1.0e-300:
                        targetDiag = 1.0e-300
                    absErr = fabs(log(recomposed / targetDiag))
                    errArr[errCount] = absErr
                    errCount += 1
                    if absErr > maxErr:
                        maxErr = absErr
    if errCount == 0:
        return scaleArr, 0.0, 0.0
    return scaleArr, float(maxErr), float(np.median(np.asarray(errArr[:errCount], dtype=np.float64)))


def cmovingAverageSame(object values, int window):
    # Preserve the original NumPy centering for small windows, and use the Cython
    # cumulative-sum kernel for large windows where Python overhead dominated.
    cdef int window_ = max(int(window), 1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1), dtype=np.float64)
    if window_ <= 1 or x.shape[0] <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    if window_ > x.shape[0]:
        window_ = <int>x.shape[0]
    if window_ <= 256:
        return np.ascontiguousarray(np.convolve(x, np.full(window_, 1.0 / float(window_), dtype=np.float64), mode="same"), dtype=np.float64)
    return cMovingAverageSame(x, window_)


def cbooleanRunBounds(object above, int maxGapBins=0):
    r"""Run bounds for boolean threshold tracks, optionally bridging short gaps."""
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] flagsArr = np.ascontiguousarray(np.asarray(above, dtype=np.uint8).reshape(-1), dtype=np.uint8)
    cdef Py_ssize_t n = flagsArr.shape[0]
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] startsArr = np.empty(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] endsArr = np.empty(n, dtype=np.int64)
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t start
    cdef Py_ssize_t lastTrue
    cdef int gap = max(int(maxGapBins), 0)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    with nogil:
        while i < n:
            while i < n and flagsArr[i] == 0:
                i += 1
            if i >= n:
                break
            start = i
            lastTrue = i
            i += 1
            while i < n:
                if flagsArr[i] != 0:
                    if i - lastTrue > gap + 1:
                        break
                    lastTrue = i
                elif gap == 0:
                    break
                i += 1
            startsArr[count] = start
            endsArr[count] = lastTrue
            count += 1
            if i <= lastTrue:
                i = lastTrue + 1
    return startsArr[:count].copy(), endsArr[:count].copy()
