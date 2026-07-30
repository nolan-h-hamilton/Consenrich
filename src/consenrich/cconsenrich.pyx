# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
# distutils: language = c
r"""Cython module for Consenrich core functions.

This module contains Cython implementations of core functions used in Consenrich.
"""

cimport cython
import hashlib
import numbers
import os
import numpy as np
from . import misc_util
from scipy import ndimage, signal
cimport numpy as cnp
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from numpy.random import default_rng
from libc.math cimport isfinite, fabs, log1p, log2, log, log2f, logf, asinhf, asinh, fmax, fmaxf, pow, sqrt, sqrtf, fabsf, fminf, fmin, log10, log10f, ceil, floor, floorf, exp, expf, cos, sin, erf, isnan, lgamma, NAN, INFINITY
from libc.float cimport DBL_MIN
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdio cimport printf, fprintf, fflush, stdout, stderr
from cython.parallel cimport prange

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
cdef dict __dependenceDPSSCache = {}
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


ctypedef struct LevelTrendForwardLoopResult:
    double sumDStat
    double sumNLL
    Py_ssize_t invalidBlockIndex


cdef LevelTrendForwardLoopResult _levelTrendForwardPassLoop(
    const cnp.float32_t* dataPtr,
    const cnp.float32_t* muncPtr,
    const cnp.int32_t* blockMapPtr,
    const cnp.float32_t* lambdaExpPtr,
    const cnp.float32_t* processPrecExpPtr,
    const cnp.float32_t* processQScalePtr,
    cnp.float32_t* dStatPtr,
    cnp.float32_t* stateForwardPtr,
    cnp.float32_t* stateCovarForwardPtr,
    cnp.float32_t* pNoiseForwardPtr,
    Py_ssize_t trackCount,
    Py_ssize_t intervalCount,
    Py_ssize_t blockCount,
    double stateInitValue,
    double stateCovarInitValue,
    double padValue,
    double F00,
    double F01,
    double F10,
    double F11,
    double qBase00,
    double qBase01,
    double qBase10,
    double qBase11,
    double qDiagBase,
    double log2PI,
    double wMin,
    double wMax,
    double procPrecMin,
    double procPrecMax,
    double apnMinQ,
    double apnMaxQ,
    double apnThresh,
    double apnScaleCoef,
    double apnPC,
    bint doStore,
    bint useLambda,
    bint useProcPrec,
    bint useProcessQScale,
    bint useAPN,
    bint returnNLL,
    bint storeNLLInD,
) noexcept nogil:
    cdef LevelTrendForwardLoopResult result
    cdef Py_ssize_t k
    cdef Py_ssize_t j
    cdef Py_ssize_t idx
    cdef Py_ssize_t blockId
    cdef double state0 = <double><cnp.float32_t>stateInitValue
    cdef double state1 = 0.0
    cdef double cov00 = <double><cnp.float32_t>stateCovarInitValue
    cdef double cov01 = 0.0
    cdef double cov10 = 0.0
    cdef double cov11 = <double><cnp.float32_t>stateCovarInitValue
    cdef double xPred0
    cdef double xPred1
    cdef double Q00
    cdef double Q01
    cdef double Q10
    cdef double Q11
    cdef double qScale
    cdef double procPrec
    cdef double obsPrec
    cdef double apnScale = 1.0
    cdef double tmp00
    cdef double tmp01
    cdef double tmp10
    cdef double tmp11
    cdef double pred00
    cdef double pred01
    cdef double pred10
    cdef double pred11
    cdef double sumInvR
    cdef double sumInvRInnov
    cdef double sumInvRInnov2
    cdef double sumLogR
    cdef double intervalNLL
    cdef double innovScale
    cdef double gainLike
    cdef double quadForm
    cdef double statValue
    cdef double delta0
    cdef double gainG
    cdef double gainH
    cdef double IKH00
    cdef double IKH10
    cdef double new00
    cdef double new01
    cdef double new11
    cdef double procNoiseValue
    cdef double adaptiveMult

    result.sumDStat = 0.0
    result.sumNLL = 0.0
    result.invalidBlockIndex = -1

    for k in range(intervalCount):
        blockId = <Py_ssize_t>blockMapPtr[k]
        if blockId < 0 or blockId >= blockCount:
            result.invalidBlockIndex = k
            return result

        if useProcPrec:
            procPrec = _clampMultiplierValue(
                <double>processPrecExpPtr[k],
                procPrecMin,
                procPrecMax,
            )
        else:
            procPrec = 1.0

        xPred0 = F00 * state0 + F01 * state1
        xPred1 = F10 * state0 + F11 * state1
        state0 = <double><cnp.float32_t>xPred0
        state1 = <double><cnp.float32_t>xPred1

        if useProcessQScale:
            qScale = <double>processQScalePtr[k]
        else:
            qScale = apnScale
        Q00 = (qScale / procPrec) * qBase00
        Q01 = (qScale / procPrec) * qBase01
        Q10 = (qScale / procPrec) * qBase10
        Q11 = (qScale / procPrec) * qBase11

        tmp00 = F00 * cov00 + F01 * cov10
        tmp01 = F00 * cov01 + F01 * cov11
        tmp10 = F10 * cov00 + F11 * cov10
        tmp11 = F10 * cov01 + F11 * cov11

        pred00 = tmp00 * F00 + tmp01 * F01 + Q00
        pred01 = tmp00 * F10 + tmp01 * F11 + Q01
        pred10 = tmp10 * F00 + tmp11 * F01 + Q10
        pred11 = tmp10 * F10 + tmp11 * F11 + Q11

        cov00 = <double><cnp.float32_t>pred00
        cov01 = <double><cnp.float32_t>pred01
        cov10 = <double><cnp.float32_t>pred10
        cov11 = <double><cnp.float32_t>pred11

        if useLambda:
            obsPrec = _clampMultiplierValue(<double>lambdaExpPtr[k], wMin, wMax)
        else:
            obsPrec = 1.0

        sumInvR = 0.0
        sumInvRInnov = 0.0
        sumInvRInnov2 = 0.0
        sumLogR = 0.0
        intervalNLL = 0.0

        for j in range(trackCount):
            idx = j * intervalCount + k
            _accumulateObservationValue(
                <double>dataPtr[idx],
                state0,
                <double>muncPtr[idx],
                padValue,
                obsPrec,
                returnNLL,
                &sumInvR,
                &sumInvRInnov,
                &sumInvRInnov2,
                &sumLogR,
            )

        innovScale = 1.0 + cov00 * sumInvR
        gainLike = cov00 / innovScale
        quadForm = sumInvRInnov2 - gainLike * (sumInvRInnov * sumInvRInnov)
        if quadForm < 0.0:
            quadForm = 0.0

        if returnNLL:
            intervalNLL = 0.5 * (
                sumLogR + log(innovScale) + quadForm + (<double>trackCount) * log2PI
            )
            result.sumNLL += intervalNLL

        if returnNLL and storeNLLInD:
            statValue = intervalNLL
        else:
            statValue = quadForm / (<double>trackCount)
        dStatPtr[k] = <cnp.float32_t>statValue
        result.sumDStat += <double>dStatPtr[k]

        delta0 = sumInvRInnov / innovScale
        state0 = <double><cnp.float32_t>(state0 + cov00 * delta0)
        state1 = <double><cnp.float32_t>(state1 + cov10 * delta0)

        gainG = sumInvR / innovScale
        gainH = sumInvR / (innovScale * innovScale)
        IKH00 = 1.0 - (cov00 * gainG)
        IKH10 = -(cov10 * gainG)

        new00 = (IKH00 * IKH00 * cov00) + (gainH * (cov00 * cov00))
        new01 = (IKH00 * (IKH10 * cov00 + cov01)) + (gainH * (cov00 * cov10))
        new11 = (
            (IKH10 * IKH10 * cov00) + 2.0 * IKH10 * cov10 + cov11
        ) + (gainH * (cov10 * cov10))

        cov00 = <double><cnp.float32_t>new00
        cov01 = <double><cnp.float32_t>new01
        cov10 = cov01
        cov11 = <double><cnp.float32_t>new11

        if doStore:
            stateForwardPtr[k * 2] = <cnp.float32_t>state0
            stateForwardPtr[k * 2 + 1] = <cnp.float32_t>state1
            stateCovarForwardPtr[k * 4] = <cnp.float32_t>cov00
            stateCovarForwardPtr[k * 4 + 1] = <cnp.float32_t>cov01
            stateCovarForwardPtr[k * 4 + 2] = <cnp.float32_t>cov10
            stateCovarForwardPtr[k * 4 + 3] = <cnp.float32_t>cov11
            if k > 0:
                pNoiseForwardPtr[(k - 1) * 4] = <cnp.float32_t>Q00
                pNoiseForwardPtr[(k - 1) * 4 + 1] = <cnp.float32_t>Q01
                pNoiseForwardPtr[(k - 1) * 4 + 2] = <cnp.float32_t>Q10
                pNoiseForwardPtr[(k - 1) * 4 + 3] = <cnp.float32_t>Q11

        if useAPN and (not useProcessQScale):
            procNoiseValue = 0.5 * (Q00 + Q11)
            if dStatPtr[k] > apnThresh and procNoiseValue < apnMaxQ:
                adaptiveMult = sqrt(
                    apnScaleCoef * ((<double>dStatPtr[k]) - apnThresh) + apnPC
                )
                apnScale *= adaptiveMult
            elif dStatPtr[k] <= apnThresh and procNoiseValue > apnMinQ:
                adaptiveMult = 1.0 / sqrt(
                    apnScaleCoef * (apnThresh - (<double>dStatPtr[k])) + apnPC
                )
                apnScale *= adaptiveMult

            procNoiseValue = apnScale * qDiagBase
            if procNoiseValue < apnMinQ:
                apnScale = apnMinQ / qDiagBase
            elif procNoiseValue > apnMaxQ:
                apnScale = apnMaxQ / qDiagBase

    return result


ctypedef struct LevelForwardLoopResult:
    double sumDStat
    double sumNLL
    Py_ssize_t invalidBlockIndex


cdef LevelForwardLoopResult _levelForwardPassLoop(
    const cnp.float32_t* dataPtr,
    const cnp.float32_t* muncPtr,
    const cnp.int32_t* blockMapPtr,
    const cnp.float32_t* lambdaExpPtr,
    const cnp.float32_t* processPrecExpPtr,
    const cnp.float32_t* processQScalePtr,
    cnp.float32_t* dStatPtr,
    cnp.float32_t* stateForwardPtr,
    cnp.float32_t* stateCovarForwardPtr,
    cnp.float32_t* pNoiseForwardPtr,
    Py_ssize_t trackCount,
    Py_ssize_t intervalCount,
    Py_ssize_t blockCount,
    double stateInitValue,
    double stateCovarInitValue,
    double padValue,
    double q0,
    double log2PI,
    double wMin,
    double wMax,
    double procPrecMin,
    double procPrecMax,
    double apnMinQ,
    double apnMaxQ,
    double apnThresh,
    double apnScaleCoef,
    double apnPC,
    bint doStore,
    bint useLambda,
    bint useProcPrec,
    bint useProcessQScale,
    bint useAPN,
    bint returnNLL,
    bint storeNLLInD,
) noexcept nogil:
    cdef LevelForwardLoopResult result
    cdef Py_ssize_t k
    cdef Py_ssize_t j
    cdef Py_ssize_t idx
    cdef Py_ssize_t blockId
    cdef double stateValue = stateInitValue
    cdef double stateVar = stateCovarInitValue
    cdef double Q
    cdef double sumInvR
    cdef double sumInvRInnov
    cdef double sumInvRInnov2
    cdef double sumLogR
    cdef double intervalNLL
    cdef double innovScale
    cdef double gainLike
    cdef double quadForm
    cdef double statValue
    cdef double delta0
    cdef double gainG
    cdef double gainH
    cdef double IKH
    cdef double newVar
    cdef double obsPrec
    cdef double procPrec
    cdef double qScale
    cdef double procNoiseValue
    cdef double adaptiveMult
    cdef double apnScale = 1.0

    result.sumDStat = 0.0
    result.sumNLL = 0.0
    result.invalidBlockIndex = -1

    for k in range(intervalCount):
        blockId = <Py_ssize_t>blockMapPtr[k]
        if blockId < 0 or blockId >= blockCount:
            result.invalidBlockIndex = k
            return result

        if useProcPrec:
            procPrec = _clampMultiplierValue(
                <double>processPrecExpPtr[k],
                procPrecMin,
                procPrecMax,
            )
        else:
            procPrec = 1.0

        if useProcessQScale:
            qScale = <double>processQScalePtr[k]
        else:
            qScale = apnScale
        Q = (qScale / procPrec) * q0
        stateVar += Q

        if useLambda:
            obsPrec = _clampMultiplierValue(<double>lambdaExpPtr[k], wMin, wMax)
        else:
            obsPrec = 1.0

        sumInvR = 0.0
        sumInvRInnov = 0.0
        sumInvRInnov2 = 0.0
        sumLogR = 0.0
        intervalNLL = 0.0

        for j in range(trackCount):
            idx = j * intervalCount + k
            _accumulateObservationValue(
                <double>dataPtr[idx],
                stateValue,
                <double>muncPtr[idx],
                padValue,
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
            intervalNLL = 0.5 * (
                sumLogR + log(innovScale) + quadForm + (<double>trackCount) * log2PI
            )
            result.sumNLL += intervalNLL

        if returnNLL and storeNLLInD:
            statValue = intervalNLL
        else:
            statValue = quadForm / (<double>trackCount)
        dStatPtr[k] = <cnp.float32_t>statValue
        result.sumDStat += <double>dStatPtr[k]

        delta0 = sumInvRInnov / innovScale
        stateValue += stateVar * delta0

        gainG = sumInvR / innovScale
        gainH = sumInvR / (innovScale * innovScale)
        IKH = 1.0 - stateVar * gainG
        newVar = (IKH * IKH * stateVar) + (gainH * (stateVar * stateVar))
        stateVar = newVar

        if doStore:
            stateForwardPtr[k] = <cnp.float32_t>stateValue
            stateCovarForwardPtr[k] = <cnp.float32_t>stateVar
            if k > 0:
                pNoiseForwardPtr[k - 1] = <cnp.float32_t>Q

        if useAPN and (not useProcessQScale):
            procNoiseValue = apnScale * q0
            if dStatPtr[k] > apnThresh and procNoiseValue < apnMaxQ:
                adaptiveMult = sqrt(
                    apnScaleCoef * ((<double>dStatPtr[k]) - apnThresh) + apnPC
                )
                apnScale *= adaptiveMult
            elif dStatPtr[k] <= apnThresh and procNoiseValue > apnMinQ:
                adaptiveMult = 1.0 / sqrt(
                    apnScaleCoef * (apnThresh - (<double>dStatPtr[k])) + apnPC
                )
                apnScale *= adaptiveMult

            procNoiseValue = apnScale * q0
            if procNoiseValue < apnMinQ:
                apnScale = apnMinQ / q0
            elif procNoiseValue > apnMaxQ:
                apnScale = apnMaxQ / q0

    return result


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
    cdef Py_ssize_t firstBadPivot = -1
    cdef double minPivot = 1.0e-12
    cdef double badPivotValue = 0.0
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
    if not isfinite(lamFirst) or lamFirst < 0.0:
        raise ValueError("lamFirst must be finite and nonnegative")
    if not isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be finite and nonnegative")

    out = np.zeros(n, dtype=np.float64)
    if n <= 0:
        return out
    if n == 1:
        if not zeroCenter:
            denomOne = <double>weightTrack[0]
            if denomOne < minPivot:
                raise RuntimeError(
                    "roughness-penalized LDL factorization required pivot "
                    f"modification at index 0 (pivot={denomOne:.6g}, "
                    f"floor={minPivot:.6g})."
                )
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
                if firstBadPivot < 0:
                    firstBadPivot = i
                    badPivotValue = diagView[i]
                diagView[i] = minPivot

        # Pentadiagonal LDL' factorization. The second lower diagonal is
        # lam / diag[i - 2] and can be recomputed, so only the first lower
        # diagonal needs storage.
        offVal = _firstDiffPenaltyOff1(n, lamFirst) + _secondDiffPenaltyOff1(n, 0, lam)
        firstLowerView[1] = offVal / diagView[0]
        diagView[1] = diagView[1] - firstLowerView[1] * firstLowerView[1] * diagView[0]
        if diagView[1] < minPivot:
            if firstBadPivot < 0:
                firstBadPivot = 1
                badPivotValue = diagView[1]
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
                if firstBadPivot < 0:
                    firstBadPivot = i
                    badPivotValue = diagView[i]
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

    if firstBadPivot >= 0:
        raise RuntimeError(
            "roughness-penalized LDL factorization required pivot "
            f"modification at index {firstBadPivot} "
            f"(pivot={badPivotValue:.6g}, floor={minPivot:.6g})."
        )

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


cdef inline void _swapF64(double* vals, Py_ssize_t i, Py_ssize_t j) noexcept nogil:
    cdef double tmp = vals[i]
    vals[i] = vals[j]
    vals[j] = tmp


cdef inline void _nthElementF64ThreeWay(double* vals, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = n - 1
    cdef Py_ssize_t lt
    cdef Py_ssize_t i
    cdef Py_ssize_t gt
    cdef double pivot

    if n <= 1:
        return
    while left < right:
        pivot = vals[(left + right) >> 1]
        lt = left
        i = left
        gt = right
        while i <= gt:
            if vals[i] < pivot:
                _swapF64(vals, lt, i)
                lt += 1
                i += 1
            elif vals[i] > pivot:
                _swapF64(vals, i, gt)
                gt -= 1
            else:
                i += 1
        if k < lt:
            right = lt - 1
        elif k > gt:
            left = gt + 1
        else:
            return


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


cdef inline double _linearQuantileInplaceF64(double* values, Py_ssize_t n, double q) noexcept nogil:
    cdef double pos
    cdef double frac
    cdef double lowVal
    cdef double highVal
    cdef Py_ssize_t lowIndex
    cdef Py_ssize_t highIndex

    if n <= 0:
        return NAN
    if q <= 0.0:
        pos = 0.0
    elif q >= 1.0:
        pos = <double>(n - 1)
    else:
        pos = q * <double>(n - 1)
    lowIndex = <Py_ssize_t>floor(pos)
    highIndex = lowIndex + 1
    if highIndex >= n:
        highIndex = n - 1
    frac = pos - <double>lowIndex
    _nthElementF64ThreeWay(values, n, lowIndex)
    lowVal = values[lowIndex]
    if highIndex == lowIndex:
        return lowVal
    _nthElementF64ThreeWay(values, n, highIndex)
    highVal = values[highIndex]
    return lowVal + frac * (highVal - lowVal)


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


cdef double _linearQuantileCopyF64(const double* values, Py_ssize_t n, double q) except *:
    cdef double* buf
    cdef double pos
    cdef double frac
    cdef double lowVal
    cdef double highVal
    cdef Py_ssize_t lowIndex
    cdef Py_ssize_t highIndex

    if n <= 0:
        return NAN
    if q <= 0.0:
        pos = 0.0
    elif q >= 1.0:
        pos = <double>(n - 1)
    else:
        pos = q * <double>(n - 1)
    lowIndex = <Py_ssize_t>floor(pos)
    highIndex = lowIndex + 1
    if highIndex >= n:
        highIndex = n - 1
    frac = pos - <double>lowIndex
    buf = <double*>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    memcpy(buf, values, n * sizeof(double))
    _nthElementF64ThreeWay(buf, n, lowIndex)
    lowVal = buf[lowIndex]
    if highIndex == lowIndex:
        free(buf)
        return lowVal
    _nthElementF64ThreeWay(buf, n, highIndex)
    highVal = buf[highIndex]
    free(buf)
    return lowVal + frac * (highVal - lowVal)


cdef double _weightedQuantileInterpolatedF64(
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] valuesArr,
    cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weightsArr,
    double quantile,
) except *:
    cdef object order
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sortedValuesArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sortedWeightsArr
    cdef double[::1] valueView
    cdef double[::1] weightView
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef double total = 0.0
    cdef double target
    cdef double cum = 0.0
    cdef double prevCum = 0.0
    cdef double prevValue = 0.0
    cdef double denom

    n = valuesArr.shape[0]
    if n != weightsArr.shape[0]:
        raise ValueError("values and weights must have the same length")
    if n <= 0:
        return NAN
    order = np.argsort(valuesArr, kind="mergesort")
    sortedValuesArr = np.ascontiguousarray(valuesArr[order], dtype=np.float64)
    sortedWeightsArr = np.ascontiguousarray(weightsArr[order], dtype=np.float64)
    valueView = sortedValuesArr
    weightView = sortedWeightsArr
    for i in range(n):
        total += weightView[i]
    if total <= 0.0:
        return NAN
    if quantile <= 0.0:
        target = 0.0
    elif quantile >= 1.0:
        target = total
    else:
        target = quantile * total
    for i in range(n):
        cum += weightView[i]
        if target <= cum:
            if i == 0:
                return valueView[0]
            denom = cum - prevCum
            if denom <= 0.0:
                return valueView[i]
            return prevValue + ((target - prevCum) / denom) * (valueView[i] - prevValue)
        prevCum = cum
        prevValue = valueView[i]
    return valueView[n - 1]


cdef double _robustLocationF64(double* values, double* weights, Py_ssize_t n) except *:
    cdef double loc
    cdef double scale
    cdef double c = 1.345
    cdef double resid
    cdef double huber
    cdef double eff
    cdef double denom
    cdef double numer
    cdef double nextLoc
    cdef double* absDev
    cdef Py_ssize_t i
    cdef Py_ssize_t iterIndex

    if n <= 0:
        return NAN
    if n == 1:
        return values[0]
    loc = _linearQuantileCopyF64(values, n, 0.5)
    absDev = <double*>malloc(n * sizeof(double))
    if absDev == NULL:
        raise MemoryError()
    for i in range(n):
        absDev[i] = fabs(values[i] - loc)
    scale = 1.4826 * _linearQuantileCopyF64(absDev, n, 0.5)
    free(absDev)
    if scale <= 1.0e-12:
        return loc
    for iterIndex in range(4):
        denom = 0.0
        numer = 0.0
        for i in range(n):
            resid = values[i] - loc
            huber = (c * scale) / fmax(fabs(resid), 1.0e-12)
            if huber > 1.0:
                huber = 1.0
            eff = weights[i] * huber
            denom += eff
            numer += eff * values[i]
        if denom <= 0.0:
            break
        nextLoc = numer / denom
        if fabs(nextLoc - loc) <= 1.0e-10 * fmax(1.0, fabs(loc)):
            loc = nextLoc
            break
        loc = nextLoc
    return loc


cdef double _cdfQuantileF64(
    double[::1] gridView,
    double[::1] posteriorView,
    Py_ssize_t n,
    double prob,
) except *:
    cdef double target
    cdef double cum = 0.0
    cdef double prevCum = 0.0
    cdef double denom
    cdef Py_ssize_t i

    if n <= 0:
        return NAN
    if prob <= 0.0:
        target = 0.0
    elif prob >= 1.0:
        target = 1.0
    else:
        target = prob
    for i in range(n):
        cum += posteriorView[i]
        if target <= cum:
            if i == 0:
                return gridView[0]
            denom = cum - prevCum
            if denom <= 0.0:
                return gridView[i]
            return gridView[i - 1] + ((target - prevCum) / denom) * (
                gridView[i] - gridView[i - 1]
            )
        prevCum = cum
    return gridView[n - 1]


cdef inline Py_ssize_t _qSeedSampleIndex(
    Py_ssize_t sampleIndex,
    Py_ssize_t itemCount,
    Py_ssize_t sampleCount,
) noexcept nogil:
    return <Py_ssize_t>floor(
        ((<double>sampleIndex + 0.5) * <double>itemCount) / <double>sampleCount
    )


cpdef tuple cEstimateSameTrackProcessNoiseTransitions(
    object matrixData,
    object obsVar,
    object activeObservation,
    double precisionCapQuantile,
    double precisionCapMultiplier,
    Py_ssize_t maxTransitionSamples=0,
    Py_ssize_t precisionSampleCap=32000,
    Py_ssize_t signalPanelSize=0,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] dataArr
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] obsArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] activeArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rawPrecisionArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] deltasArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] samplingVariancesArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] transitionWeightsArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] signalLevelsArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] localDeltaArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] localLevelArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] localPrecisionArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] selectedDeltasArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] selectedSamplingVariancesArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] selectedTransitionWeightsArr
    cdef double[:, ::1] dataView
    cdef double[:, ::1] obsView
    cdef cnp.uint8_t[:, ::1] activeView
    cdef double[::1] rawPrecisionView
    cdef double[::1] deltasView
    cdef double[::1] samplingVariancesView
    cdef double[::1] transitionWeightsView
    cdef double[::1] signalLevelsView
    cdef double[::1] localDeltaView
    cdef double[::1] localLevelView
    cdef double[::1] localPrecisionView
    cdef double[::1] selectedDeltasView
    cdef double[::1] selectedSamplingVariancesView
    cdef double[::1] selectedTransitionWeightsView
    cdef Py_ssize_t trackCount
    cdef Py_ssize_t intervalCount
    cdef Py_ssize_t maxTransitionCount
    cdef Py_ssize_t transitionScanCount
    cdef Py_ssize_t maxPairCount
    cdef Py_ssize_t pairCount = 0
    cdef Py_ssize_t sampledPairCount = 0
    cdef Py_ssize_t precisionSampleCount = 0
    cdef Py_ssize_t cappedPairCount = 0
    cdef Py_ssize_t outCount = 0
    cdef Py_ssize_t candidateTransitionCount = 0
    cdef Py_ssize_t selectedTransitionCount = 0
    cdef Py_ssize_t localCount
    cdef Py_ssize_t scanIndex
    cdef Py_ssize_t panelIndex
    cdef Py_ssize_t selectedRank
    cdef Py_ssize_t candidateIndex
    cdef Py_ssize_t pairOrdinal
    cdef Py_ssize_t sampleSlot
    cdef Py_ssize_t targetPairIndex
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef double obsLeft
    cdef double obsRight
    cdef double rd
    cdef double rawPrecision
    cdef double precision
    cdef double medianPrecision
    cdef double qPrecision
    cdef double cap = NAN
    cdef double capFraction = 0.0
    cdef double diff
    cdef double loc
    cdef double signalLevel
    cdef double sumP
    cdef double sumP2
    cdef double effPairs
    cdef double transitionSampleFraction = 1.0
    cdef bint cappedMode = False
    cdef dict diagnostics
    cdef object signalOrder
    cdef object sampledTransitionIndices

    if signalPanelSize < 0:
        raise ValueError("signalPanelSize must be nonnegative")
    if (not isfinite(precisionCapQuantile)) or precisionCapQuantile < 0.0 or precisionCapQuantile > 1.0:
        raise ValueError("precisionCapQuantile must be in [0, 1]")
    if (not isfinite(precisionCapMultiplier)) or precisionCapMultiplier <= 0.0:
        raise ValueError("precisionCapMultiplier must be positive")
    dataArr = np.ascontiguousarray(matrixData, dtype=np.float64)
    obsArr = np.ascontiguousarray(obsVar, dtype=np.float64)
    activeArr = np.ascontiguousarray(activeObservation, dtype=np.uint8)
    if dataArr.ndim != 2:
        raise ValueError("matrixData must be a 2D array")
    if obsArr.shape[0] != dataArr.shape[0] or obsArr.shape[1] != dataArr.shape[1]:
        raise ValueError("obsVar shape must match matrixData")
    if activeArr.shape[0] != dataArr.shape[0] or activeArr.shape[1] != dataArr.shape[1]:
        raise ValueError("activeObservation shape must match matrixData")
    trackCount = dataArr.shape[0]
    intervalCount = dataArr.shape[1]
    if intervalCount < 2 or trackCount <= 0:
        diagnostics = {
            "pairCount": int(0),
            "precisionCap": float(cap),
            "precisionCapFraction": float(capFraction),
            "candidateTransitionCount": int(0),
            "selectedTransitionCount": int(0),
        }
        if cappedMode:
            diagnostics.update(
                {
                    "sampledPairCount": int(0),
                    "sampledTransitionCount": int(0),
                    "transitionSampleFraction": float(0.0),
                    "precisionSampleCap": int(precisionSampleCap),
                    "maxTransitionSamples": int(maxTransitionSamples),
                }
            )
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            diagnostics,
        )
    dataView = dataArr
    obsView = obsArr
    activeView = activeArr
    maxTransitionCount = intervalCount - 1
    transitionScanCount = maxTransitionCount
    if maxTransitionSamples > 0 and maxTransitionSamples < maxTransitionCount:
        cappedMode = True
        transitionScanCount = maxTransitionSamples
        transitionSampleFraction = <double>transitionScanCount / <double>maxTransitionCount
        if precisionSampleCap <= 0:
            raise ValueError("precisionSampleCap must be positive")

    if cappedMode:
        for scanIndex in range(transitionScanCount):
            k = _qSeedSampleIndex(scanIndex, maxTransitionCount, transitionScanCount)
            for j in range(trackCount):
                if activeView[j, k] != 0 and activeView[j, k + 1] != 0:
                    if (not isfinite(dataView[j, k])) or (not isfinite(dataView[j, k + 1])):
                        raise ValueError("active matrixData values must be finite")
                    obsLeft = obsView[j, k]
                    obsRight = obsView[j, k + 1]
                    if (
                        (not isfinite(obsLeft))
                        or (not isfinite(obsRight))
                        or obsLeft <= 0.0
                        or obsRight <= 0.0
                    ):
                        raise ValueError("active obsVar values must be positive finite")
                    diff = dataView[j, k + 1] - dataView[j, k]
                    rd = obsLeft + obsRight
                    if (not isfinite(diff)) or (not isfinite(rd)) or rd <= 0.0:
                        raise ValueError("active transition values must be finite")
                    rawPrecision = 1.0 / rd
                    if (not isfinite(rawPrecision)) or rawPrecision <= 0.0:
                        raise ValueError("active transition precision must be positive finite")
                    pairCount += 1
        precisionSampleCount = pairCount
        if precisionSampleCount > precisionSampleCap:
            precisionSampleCount = precisionSampleCap
        sampledPairCount = pairCount
        if precisionSampleCount > 0:
            rawPrecisionArr = np.empty(precisionSampleCount, dtype=np.float64)
            rawPrecisionView = rawPrecisionArr
            pairOrdinal = 0
            sampleSlot = 0
            targetPairIndex = _qSeedSampleIndex(sampleSlot, pairCount, precisionSampleCount)
            for scanIndex in range(transitionScanCount):
                k = _qSeedSampleIndex(scanIndex, maxTransitionCount, transitionScanCount)
                for j in range(trackCount):
                    if activeView[j, k] != 0 and activeView[j, k + 1] != 0:
                        if pairOrdinal == targetPairIndex:
                            rawPrecisionView[sampleSlot] = 1.0 / (obsView[j, k] + obsView[j, k + 1])
                            sampleSlot += 1
                            if sampleSlot < precisionSampleCount:
                                targetPairIndex = _qSeedSampleIndex(
                                    sampleSlot,
                                    pairCount,
                                    precisionSampleCount,
                                )
                        pairOrdinal += 1
    else:
        maxPairCount = trackCount * maxTransitionCount
        rawPrecisionArr = np.empty(maxPairCount, dtype=np.float64)
        rawPrecisionView = rawPrecisionArr
        for k in range(maxTransitionCount):
            for j in range(trackCount):
                if activeView[j, k] != 0 and activeView[j, k + 1] != 0:
                    if (not isfinite(dataView[j, k])) or (not isfinite(dataView[j, k + 1])):
                        raise ValueError("active matrixData values must be finite")
                    obsLeft = obsView[j, k]
                    obsRight = obsView[j, k + 1]
                    if (
                        (not isfinite(obsLeft))
                        or (not isfinite(obsRight))
                        or obsLeft <= 0.0
                        or obsRight <= 0.0
                    ):
                        raise ValueError("active obsVar values must be positive finite")
                    diff = dataView[j, k + 1] - dataView[j, k]
                    rd = obsLeft + obsRight
                    if (not isfinite(diff)) or (not isfinite(rd)) or rd <= 0.0:
                        raise ValueError("active transition values must be finite")
                    rawPrecision = 1.0 / rd
                    if (not isfinite(rawPrecision)) or rawPrecision <= 0.0:
                        raise ValueError("active transition precision must be positive finite")
                    rawPrecisionView[pairCount] = rawPrecision
                    pairCount += 1
        sampledPairCount = pairCount
        precisionSampleCount = pairCount
    if precisionSampleCount > 0:
        medianPrecision = _linearQuantileCopyF64(&rawPrecisionView[0], precisionSampleCount, 0.5)
        qPrecision = _linearQuantileCopyF64(
            &rawPrecisionView[0], precisionSampleCount, precisionCapQuantile
        )
        cap = fmin(qPrecision, precisionCapMultiplier * medianPrecision)
        if cap > 0.0 and not cappedMode:
            for j in range(pairCount):
                if rawPrecisionView[j] > cap:
                    cappedPairCount += 1
            capFraction = <double>cappedPairCount / <double>pairCount
    deltasArr = np.empty(transitionScanCount, dtype=np.float64)
    samplingVariancesArr = np.empty(transitionScanCount, dtype=np.float64)
    transitionWeightsArr = np.empty(transitionScanCount, dtype=np.float64)
    signalLevelsArr = np.empty(transitionScanCount, dtype=np.float64)
    localDeltaArr = np.empty(trackCount, dtype=np.float64)
    localLevelArr = np.empty(trackCount, dtype=np.float64)
    localPrecisionArr = np.empty(trackCount, dtype=np.float64)
    deltasView = deltasArr
    samplingVariancesView = samplingVariancesArr
    transitionWeightsView = transitionWeightsArr
    signalLevelsView = signalLevelsArr
    localDeltaView = localDeltaArr
    localLevelView = localLevelArr
    localPrecisionView = localPrecisionArr
    for scanIndex in range(transitionScanCount):
        if cappedMode:
            k = _qSeedSampleIndex(scanIndex, maxTransitionCount, transitionScanCount)
        else:
            k = scanIndex
        localCount = 0
        for j in range(trackCount):
            if activeView[j, k] != 0 and activeView[j, k + 1] != 0:
                rawPrecision = 1.0 / (obsView[j, k] + obsView[j, k + 1])
                if cappedMode and cap > 0.0 and rawPrecision > cap:
                    cappedPairCount += 1
                precision = rawPrecision
                if cap > 0.0 and precision > cap:
                    precision = cap
                localDeltaView[localCount] = dataView[j, k + 1] - dataView[j, k]
                rd = obsView[j, k] + obsView[j, k + 1]
                localLevelView[localCount] = (
                    (obsView[j, k + 1] / rd) * dataView[j, k]
                    + (obsView[j, k] / rd) * dataView[j, k + 1]
                )
                localPrecisionView[localCount] = precision
                localCount += 1
        if localCount <= 0:
            continue
        loc = _robustLocationF64(&localDeltaView[0], &localPrecisionView[0], localCount)
        signalLevel = _robustLocationF64(
            &localLevelView[0],
            &localPrecisionView[0],
            localCount,
        )
        sumP = 0.0
        sumP2 = 0.0
        for j in range(localCount):
            sumP += localPrecisionView[j]
            sumP2 += localPrecisionView[j] * localPrecisionView[j]
        deltasView[outCount] = loc
        samplingVariancesView[outCount] = 1.0 / sumP
        if sumP2 > 0.0:
            effPairs = (sumP * sumP) / sumP2
        else:
            effPairs = 1.0
        if effPairs < 1.0:
            effPairs = 1.0
        transitionWeightsView[outCount] = effPairs
        signalLevelsView[outCount] = signalLevel
        outCount += 1
    candidateTransitionCount = outCount
    selectedTransitionCount = candidateTransitionCount
    if (
        signalPanelSize > 0
        and candidateTransitionCount > signalPanelSize
    ):
        signalOrder = np.argsort(
            signalLevelsArr[:candidateTransitionCount],
            kind="mergesort",
        )
        selectedTransitionCount = signalPanelSize
        selectedDeltasArr = np.empty(selectedTransitionCount, dtype=np.float64)
        selectedSamplingVariancesArr = np.empty(
            selectedTransitionCount,
            dtype=np.float64,
        )
        selectedTransitionWeightsArr = np.empty(
            selectedTransitionCount,
            dtype=np.float64,
        )
        selectedDeltasView = selectedDeltasArr
        selectedSamplingVariancesView = selectedSamplingVariancesArr
        selectedTransitionWeightsView = selectedTransitionWeightsArr
        for panelIndex in range(selectedTransitionCount):
            selectedRank = _qSeedSampleIndex(
                panelIndex,
                candidateTransitionCount,
                selectedTransitionCount,
            )
            candidateIndex = <Py_ssize_t>signalOrder[selectedRank]
            selectedDeltasView[panelIndex] = deltasView[candidateIndex]
            selectedSamplingVariancesView[panelIndex] = (
                samplingVariancesView[candidateIndex]
            )
            selectedTransitionWeightsView[panelIndex] = (
                transitionWeightsView[candidateIndex]
            )
        deltasArr = selectedDeltasArr
        samplingVariancesArr = selectedSamplingVariancesArr
        transitionWeightsArr = selectedTransitionWeightsArr
        outCount = selectedTransitionCount
    if cappedMode and pairCount > 0:
        capFraction = <double>cappedPairCount / <double>pairCount
    diagnostics = {
        "pairCount": int(pairCount),
        "precisionCap": float(cap),
        "precisionCapFraction": float(capFraction),
        "candidateTransitionCount": int(candidateTransitionCount),
        "selectedTransitionCount": int(selectedTransitionCount),
    }
    if cappedMode:
        if transitionScanCount <= 1024:
            sampledTransitionIndices = [
                int(_qSeedSampleIndex(scanIndex, maxTransitionCount, transitionScanCount))
                for scanIndex in range(transitionScanCount)
            ]
        else:
            sampledTransitionIndices = None
        diagnostics.update(
            {
                "sampledPairCount": int(sampledPairCount),
                "precisionSamplePairCount": int(precisionSampleCount),
                "sampledTransitionCount": int(transitionScanCount),
                "transitionSampleFraction": float(transitionSampleFraction),
                "precisionSampleCap": int(precisionSampleCap),
                "maxTransitionSamples": int(maxTransitionSamples),
                "sampledTransitionIndices": sampledTransitionIndices,
            }
        )
    return (
        deltasArr[:outCount],
        samplingVariancesArr[:outCount],
        transitionWeightsArr[:outCount],
        diagnostics,
    )


cpdef tuple cEstimatePooledProcessNoiseTransitions(
    object matrixData,
    object obsVar,
    object activeObservation,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] dataArr
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] obsArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] activeArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pooledMeanArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pooledVarArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] deltasArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] samplingVariancesArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] transitionWeightsArr
    cdef double[:, ::1] dataView
    cdef double[:, ::1] obsView
    cdef cnp.uint8_t[:, ::1] activeView
    cdef double[::1] pooledMeanView
    cdef double[::1] pooledVarView
    cdef double[::1] deltasView
    cdef double[::1] samplingVariancesView
    cdef double[::1] transitionWeightsView
    cdef Py_ssize_t trackCount
    cdef Py_ssize_t intervalCount
    cdef Py_ssize_t maxTransitionCount
    cdef Py_ssize_t outCount = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double obs
    cdef double value
    cdef double weight
    cdef double weightSum
    cdef double weightedSum
    cdef double s2

    dataArr = np.ascontiguousarray(matrixData, dtype=np.float64)
    obsArr = np.ascontiguousarray(obsVar, dtype=np.float64)
    activeArr = np.ascontiguousarray(activeObservation, dtype=np.uint8)
    if dataArr.ndim != 2:
        raise ValueError("matrixData must be a 2D array")
    if obsArr.shape[0] != dataArr.shape[0] or obsArr.shape[1] != dataArr.shape[1]:
        raise ValueError("obsVar shape must match matrixData")
    if activeArr.shape[0] != dataArr.shape[0] or activeArr.shape[1] != dataArr.shape[1]:
        raise ValueError("activeObservation shape must match matrixData")
    trackCount = dataArr.shape[0]
    intervalCount = dataArr.shape[1]
    if intervalCount < 2 or trackCount <= 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    dataView = dataArr
    obsView = obsArr
    activeView = activeArr
    pooledMeanArr = np.empty(intervalCount, dtype=np.float64)
    pooledVarArr = np.empty(intervalCount, dtype=np.float64)
    pooledMeanView = pooledMeanArr
    pooledVarView = pooledVarArr
    for i in range(intervalCount):
        weightSum = 0.0
        weightedSum = 0.0
        for j in range(trackCount):
            if activeView[j, i] != 0:
                value = dataView[j, i]
                obs = obsView[j, i]
                if (not isfinite(value)) or (not isfinite(obs)) or obs <= 0.0:
                    raise ValueError("active pooled observations must be finite with positive variance")
                weight = 1.0 / obs
                weightSum += weight
                weightedSum += value * weight
        if weightSum > 0.0:
            pooledMeanView[i] = weightedSum / weightSum
            pooledVarView[i] = 1.0 / weightSum
        else:
            pooledMeanView[i] = NAN
            pooledVarView[i] = NAN
    maxTransitionCount = intervalCount - 1
    deltasArr = np.empty(maxTransitionCount, dtype=np.float64)
    samplingVariancesArr = np.empty(maxTransitionCount, dtype=np.float64)
    transitionWeightsArr = np.empty(maxTransitionCount, dtype=np.float64)
    deltasView = deltasArr
    samplingVariancesView = samplingVariancesArr
    transitionWeightsView = transitionWeightsArr
    for i in range(maxTransitionCount):
        if (
            isfinite(pooledMeanView[i])
            and isfinite(pooledMeanView[i + 1])
            and isfinite(pooledVarView[i])
            and isfinite(pooledVarView[i + 1])
        ):
            s2 = pooledVarView[i] + pooledVarView[i + 1]
            deltasView[outCount] = pooledMeanView[i + 1] - pooledMeanView[i]
            samplingVariancesView[outCount] = s2
            if s2 > 0.0:
                transitionWeightsView[outCount] = 1.0 / fmax(s2, DBL_MIN)
            else:
                transitionWeightsView[outCount] = 1.0
            outCount += 1
    return (
        deltasArr[:outCount],
        samplingVariancesArr[:outCount],
        transitionWeightsArr[:outCount],
    )


cpdef dict cQSeedPosteriorFromTransitions(
    object deltas,
    object samplingVariances,
    object transitionWeights,
    double qFloor,
    double qCap,
    double robustTNu,
    object source,
    double qSeedPriorLevel,
    int minTransitions,
    double priorLogSd,
    double defaultTNu,
    Py_ssize_t gridSize,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] deltaArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] s2Arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weightsArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] absDevArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] deconvolvedArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] gridArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logPostArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] posteriorArr
    cdef double[::1] deltaView
    cdef double[::1] s2View
    cdef double[::1] weightsView
    cdef double[::1] absDevView
    cdef double[::1] deconvolvedView
    cdef double[::1] gridView
    cdef double[::1] logPostView
    cdef double[::1] posteriorView
    cdef Py_ssize_t transitionCount
    cdef Py_ssize_t actualGridSize
    cdef Py_ssize_t i
    cdef Py_ssize_t g
    cdef Py_ssize_t modeIndex = 0
    cdef double sumW = 0.0
    cdef double sumW2 = 0.0
    cdef double effectiveCount = 0.0
    cdef double center
    cdef double robustScale
    cdef double medianS2
    cdef double qPrior
    cdef double qTransition90
    cdef double maxDeltaSq = 0.0
    cdef double lower
    cdef double upper
    cdef double candidate
    cdef double logLower
    cdef double logUpper
    cdef double step
    cdef double nu
    cdef double medianWeight
    cdef double logPriorCenter
    cdef double logPriorSd
    cdef double logNorm
    cdef double q
    cdef double var
    cdef double weightNorm
    cdef double logLikeSum
    cdef double logPrior
    cdef double logPostValue
    cdef double maxLogPost = -INFINITY
    cdef double totalPosterior = 0.0
    cdef str sourceText

    deltaArr = np.ascontiguousarray(deltas, dtype=np.float64).reshape(-1)
    s2Arr = np.ascontiguousarray(samplingVariances, dtype=np.float64).reshape(-1)
    weightsArr = np.ascontiguousarray(transitionWeights, dtype=np.float64).reshape(-1)
    if deltaArr.shape[0] != s2Arr.shape[0] or deltaArr.shape[0] != weightsArr.shape[0]:
        raise ValueError("transition arrays must have the same length")
    if (not isfinite(qFloor)) or qFloor <= 0.0:
        raise ValueError("qFloor must be positive finite")
    if isfinite(qCap) and qCap <= 0.0:
        raise ValueError("qCap must be positive or infinite")
    if (not isfinite(qSeedPriorLevel)) or qSeedPriorLevel <= 0.0:
        raise ValueError("qSeedPriorLevel must be positive finite")
    if isfinite(qCap) and qSeedPriorLevel > qCap:
        raise ValueError("`qSeedPriorLevel` must not exceed `maxQ`")
    if minTransitions <= 0:
        raise ValueError("minTransitions must be positive")
    if (not isfinite(priorLogSd)) or priorLogSd <= 0.0:
        raise ValueError("priorLogSd must be positive finite")
    if (not isfinite(defaultTNu)) or defaultTNu <= 0.0:
        raise ValueError("defaultTNu must be positive finite")
    if gridSize <= 0:
        raise ValueError("gridSize must be positive")
    sourceText = str(source)
    transitionCount = deltaArr.shape[0]
    deltaView = deltaArr
    s2View = s2Arr
    weightsView = weightsArr
    for i in range(transitionCount):
        if not isfinite(deltaView[i]):
            raise ValueError("deltas must be finite")
        if (not isfinite(s2View[i])) or s2View[i] < 0.0:
            raise ValueError("samplingVariances must be nonnegative finite")
        if (not isfinite(weightsView[i])) or weightsView[i] <= 0.0:
            raise ValueError("transitionWeights must be positive finite")
        sumW += weightsView[i]
        sumW2 += weightsView[i] * weightsView[i]
    if sumW2 > 0.0:
        effectiveCount = (sumW * sumW) / sumW2
    if transitionCount < minTransitions or effectiveCount < <double>minTransitions:
        return {
            "ok": False,
            "source": sourceText,
            "reason": "insufficient_transition_support",
            "transitionCount": int(transitionCount),
            "effectiveTransitionCount": float(effectiveCount),
        }
    center = _weightedQuantileInterpolatedF64(deltaArr, weightsArr, 0.5)
    absDevArr = np.empty(transitionCount, dtype=np.float64)
    absDevView = absDevArr
    for i in range(transitionCount):
        absDevView[i] = fabs(deltaView[i] - center)
    robustScale = 1.4826 * _weightedQuantileInterpolatedF64(absDevArr, weightsArr, 0.5)
    medianS2 = _weightedQuantileInterpolatedF64(s2Arr, weightsArr, 0.5)
    qPrior = robustScale * robustScale - medianS2
    if qPrior < qFloor:
        qPrior = qFloor
    if qPrior < qSeedPriorLevel:
        qPrior = qSeedPriorLevel
    deconvolvedArr = np.empty(transitionCount, dtype=np.float64)
    deconvolvedView = deconvolvedArr
    for i in range(transitionCount):
        candidate = deltaView[i] * deltaView[i]
        if candidate > maxDeltaSq:
            maxDeltaSq = candidate
        candidate -= s2View[i]
        if candidate < 0.0:
            candidate = 0.0
        deconvolvedView[i] = candidate
    qTransition90 = _weightedQuantileInterpolatedF64(deconvolvedArr, weightsArr, 0.9)
    lower = qFloor
    if isfinite(qCap):
        upper = fmax(qCap, lower)
    else:
        upper = lower * 10.0
        candidate = qPrior * 1.0e4
        if candidate > upper and candidate > lower:
            upper = candidate
        candidate = qTransition90 * 100.0
        if candidate > upper and candidate > lower:
            upper = candidate
        candidate = medianS2 * 100.0
        if candidate > upper and candidate > lower:
            upper = candidate
        candidate = maxDeltaSq * 10.0
        if candidate > upper and candidate > lower:
            upper = candidate
        candidate = lower * 1.0e6
        if candidate > upper and candidate > lower:
            upper = candidate
    if upper <= lower * (1.0 + 1.0e-10):
        actualGridSize = 1
    else:
        actualGridSize = gridSize
    gridArr = np.empty(actualGridSize, dtype=np.float64)
    logPostArr = np.empty(actualGridSize, dtype=np.float64)
    posteriorArr = np.empty(actualGridSize, dtype=np.float64)
    gridView = gridArr
    logPostView = logPostArr
    posteriorView = posteriorArr
    if actualGridSize == 1:
        gridView[0] = lower
    else:
        logLower = log(lower)
        logUpper = log(upper)
        step = (logUpper - logLower) / <double>(actualGridSize - 1)
        for g in range(actualGridSize):
            gridView[g] = exp(logLower + step * <double>g)
    nu = robustTNu
    if (not isfinite(nu)) or nu <= 0.0:
        nu = defaultTNu
    if nu < 4.0:
        nu = 4.0
    medianWeight = _weightedQuantileInterpolatedF64(weightsArr, weightsArr, 0.5)
    if medianWeight < DBL_MIN:
        medianWeight = DBL_MIN
    logPriorCenter = log(fmax(qPrior, lower))
    logPriorSd = fmax(priorLogSd, 1.0e-6)
    logNorm = (
        lgamma((nu + 1.0) * 0.5)
        - lgamma(nu * 0.5)
        - 0.5 * (log(nu) + log(__PI_DOUBLE))
    )
    for g in range(actualGridSize):
        q = gridView[g]
        logLikeSum = 0.0
        for i in range(transitionCount):
            var = q + s2View[i]
            if var < DBL_MIN:
                var = DBL_MIN
            weightNorm = weightsView[i] / medianWeight
            if weightNorm < 0.25:
                weightNorm = 0.25
            elif weightNorm > 4.0:
                weightNorm = 4.0
            logLikeSum += weightNorm * (
                logNorm
                - 0.5 * log(var)
                - 0.5 * (nu + 1.0) * log1p(
                    (deltaView[i] * deltaView[i]) / (nu * var)
                )
            )
        logPrior = -0.5 * ((log(q) - logPriorCenter) / logPriorSd) * (
            (log(q) - logPriorCenter) / logPriorSd
        )
        logPostValue = logLikeSum + logPrior
        if not isfinite(logPostValue):
            raise ValueError("q seed posterior produced a nonfinite score")
        logPostView[g] = logPostValue
        if logPostValue > maxLogPost:
            maxLogPost = logPostValue
            modeIndex = g
    for g in range(actualGridSize):
        posteriorView[g] = exp(logPostView[g] - maxLogPost)
        totalPosterior += posteriorView[g]
    if (not isfinite(totalPosterior)) or totalPosterior <= 0.0:
        raise ValueError("q seed posterior normalization failed")
    for g in range(actualGridSize):
        posteriorView[g] = posteriorView[g] / totalPosterior
    return {
        "ok": True,
        "source": sourceText,
        "reason": "ok",
        "transitionCount": int(transitionCount),
        "effectiveTransitionCount": float(effectiveCount),
        "medianSamplingVariance": float(medianS2),
        "priorLevel": float(qPrior),
        "posteriorModeLevel": float(gridView[modeIndex]),
        "posteriorMedianLevel": float(
            _cdfQuantileF64(gridView, posteriorView, actualGridSize, 0.5)
        ),
        "posteriorQ05Level": float(
            _cdfQuantileF64(gridView, posteriorView, actualGridSize, 0.05)
        ),
        "posteriorQ95Level": float(
            _cdfQuantileF64(gridView, posteriorView, actualGridSize, 0.95)
        ),
        "transitionQ90": float(qTransition90),
    }


cdef inline uint64_t _mixEBPriorKey(uint64_t value) noexcept nogil:
    value ^= value >> 33
    value *= <uint64_t>0xff51afd7ed558ccd
    value ^= value >> 33
    value *= <uint64_t>0xc4ceb9fe1a85ec53
    value ^= value >> 33
    return value


cdef inline uint64_t _hashEBPriorKey(
    int64_t sampleId,
    int64_t chromosomeId,
    int64_t binId,
) noexcept nogil:
    cdef uint64_t value

    value = _mixEBPriorKey(<uint64_t>sampleId)
    value ^= _mixEBPriorKey((<uint64_t>chromosomeId) + <uint64_t>0x9e3779b97f4a7c15)
    value ^= _mixEBPriorKey((<uint64_t>binId) + <uint64_t>0xbf58476d1ce4e5b9)
    return _mixEBPriorKey(value)


cdef inline bint _insertEBPriorKey(
    uint8_t* usedPtr,
    int64_t* samplePtr,
    int64_t* chromosomePtr,
    int64_t* binPtr,
    Py_ssize_t capacity,
    int64_t sampleId,
    int64_t chromosomeId,
    int64_t binId,
) noexcept nogil:
    cdef Py_ssize_t slot = <Py_ssize_t>(
        _hashEBPriorKey(sampleId, chromosomeId, binId) & <uint64_t>(capacity - 1)
    )

    while usedPtr[slot] != 0:
        if (
            samplePtr[slot] == sampleId
            and chromosomePtr[slot] == chromosomeId
            and binPtr[slot] == binId
        ):
            return False
        slot = (slot + 1) & (capacity - 1)

    usedPtr[slot] = <uint8_t>1
    samplePtr[slot] = sampleId
    chromosomePtr[slot] = chromosomeId
    binPtr[slot] = binId
    return True


cpdef tuple cEBPriorStrengthCandidateIdx(
    object localModelVariances,
    object globalModelVariances,
    object localLogVarianceNoise=None,
    object candidateMask=None,
    Py_ssize_t thinStride=1,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] localArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] globalArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] noiseArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] maskArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] phaseCountArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] candidateArr
    cdef double[::1] localView
    cdef double[::1] globalView
    cdef double[::1] noiseView
    cdef cnp.uint8_t[::1] maskView
    cdef cnp.intp_t[::1] phaseCountView
    cdef cnp.intp_t[::1] candidateView
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef Py_ssize_t stride = thinStride
    cdef Py_ssize_t phase
    cdef Py_ssize_t bestPhase = 0
    cdef Py_ssize_t bestCount = 0
    cdef Py_ssize_t candidateCount = 0
    cdef Py_ssize_t outCount = 0
    cdef bint useNoise = localLogVarianceNoise is not None
    cdef bint useMask = candidateMask is not None
    cdef double localValue
    cdef double globalValue
    cdef double noiseValue

    localArr = np.ascontiguousarray(
        np.asarray(localModelVariances, dtype=np.float64).ravel()
    )
    globalArr = np.ascontiguousarray(
        np.asarray(globalModelVariances, dtype=np.float64).ravel()
    )
    if localArr.shape[0] != globalArr.shape[0]:
        raise ValueError("localModelVariances and globalModelVariances must have the same shape")
    n = localArr.shape[0]
    localView = localArr
    globalView = globalArr
    if useNoise:
        noiseArr = np.ascontiguousarray(
            np.asarray(localLogVarianceNoise, dtype=np.float64).ravel()
        )
        if noiseArr.shape[0] != n:
            raise ValueError("localLogVarianceNoise must match localModelVariances")
        noiseView = noiseArr
    if useMask:
        maskArr = np.ascontiguousarray(
            np.asarray(candidateMask, dtype=np.uint8).ravel()
        )
        if maskArr.shape[0] != n:
            raise ValueError("candidateMask must match localModelVariances")
        maskView = maskArr
    if stride < 1:
        stride = 1
    if stride > 1:
        phaseCountArr = np.zeros(stride, dtype=np.intp)
        phaseCountView = phaseCountArr

    for i in range(n):
        if useMask and maskView[i] == 0:
            continue
        localValue = localView[i]
        if (not isfinite(localValue)) or localValue <= 0.0:
            raise ValueError(f"localModelVariances must contain finite positive values at index {i}")
        globalValue = globalView[i]
        if (not isfinite(globalValue)) or globalValue <= 0.0:
            raise ValueError(f"globalModelVariances must contain finite positive values at index {i}")
        if useNoise:
            noiseValue = noiseView[i]
            if (not isfinite(noiseValue)) or noiseValue <= 0.0:
                raise ValueError(f"localLogVarianceNoise must contain finite positive values at index {i}")
        candidateCount += 1
        if stride > 1:
            phase = i % stride
            phaseCountView[phase] += 1
        else:
            outCount += 1

    if stride > 1:
        for phase in range(stride):
            if phaseCountView[phase] > bestCount:
                bestCount = phaseCountView[phase]
                bestPhase = phase
        outCount = bestCount

    candidateArr = np.empty(outCount, dtype=np.intp)
    candidateView = candidateArr
    outCount = 0
    for i in range(n):
        if useMask and maskView[i] == 0:
            continue
        if stride == 1 or (i % stride) == bestPhase:
            candidateView[outCount] = i
            outCount += 1

    return candidateArr, int(candidateCount)


cpdef tuple cEBPooledPriorStrengthCandidateIdx(
    object localModelVariances,
    object globalModelVariances,
    object localLogVarianceNoise=None,
    object sampleIndex=None,
    object chromosomeIndex=None,
    object blockStarts=None,
    Py_ssize_t thinBinSize=1,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] localArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] globalArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] noiseArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] sampleArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] chromosomeArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] startArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] candidateArr
    cdef double[::1] localView
    cdef double[::1] globalView
    cdef double[::1] noiseView
    cdef int64_t[::1] sampleView
    cdef int64_t[::1] chromosomeView
    cdef int64_t[::1] startView
    cdef cnp.intp_t[::1] candidateView
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef Py_ssize_t tableIdx
    cdef Py_ssize_t capacity
    cdef Py_ssize_t binSize = thinBinSize
    cdef Py_ssize_t candidateCount = 0
    cdef Py_ssize_t outCount = 0
    cdef bint useNoise = localLogVarianceNoise is not None
    cdef bint hasSample = sampleIndex is not None
    cdef bint hasChromosome = chromosomeIndex is not None
    cdef bint hasStart = blockStarts is not None
    cdef bint usePooledThin = hasSample or hasChromosome or hasStart
    cdef double localValue
    cdef double globalValue
    cdef double noiseValue
    cdef int64_t sampleId
    cdef int64_t chromosomeId
    cdef int64_t startValue
    cdef int64_t binId
    cdef uint8_t* usedPtr = <uint8_t*>NULL
    cdef int64_t* sampleKeyPtr = <int64_t*>NULL
    cdef int64_t* chromosomeKeyPtr = <int64_t*>NULL
    cdef int64_t* binKeyPtr = <int64_t*>NULL

    if usePooledThin and not (hasSample and hasChromosome and hasStart):
        raise ValueError("sampleIndex, chromosomeIndex, and blockStarts must be provided together")

    localArr = np.ascontiguousarray(
        np.asarray(localModelVariances, dtype=np.float64).ravel()
    )
    globalArr = np.ascontiguousarray(
        np.asarray(globalModelVariances, dtype=np.float64).ravel()
    )
    if localArr.shape[0] != globalArr.shape[0]:
        raise ValueError("localModelVariances and globalModelVariances must have the same shape")
    n = localArr.shape[0]
    localView = localArr
    globalView = globalArr
    if useNoise:
        noiseArr = np.ascontiguousarray(
            np.asarray(localLogVarianceNoise, dtype=np.float64).ravel()
        )
        if noiseArr.shape[0] != n:
            raise ValueError("localLogVarianceNoise must match localModelVariances")
        noiseView = noiseArr
    if usePooledThin:
        sampleArr = np.ascontiguousarray(
            np.asarray(sampleIndex, dtype=np.int64).ravel()
        )
        chromosomeArr = np.ascontiguousarray(
            np.asarray(chromosomeIndex, dtype=np.int64).ravel()
        )
        startArr = np.ascontiguousarray(
            np.asarray(blockStarts, dtype=np.int64).ravel()
        )
        if sampleArr.shape[0] != n or chromosomeArr.shape[0] != n or startArr.shape[0] != n:
            raise ValueError("sampleIndex, chromosomeIndex, and blockStarts must match localModelVariances")
        sampleView = sampleArr
        chromosomeView = chromosomeArr
        startView = startArr
    if binSize < 1:
        binSize = 1

    for i in range(n):
        localValue = localView[i]
        if (not isfinite(localValue)) or localValue <= 0.0:
            raise ValueError(f"localModelVariances must contain finite positive values at index {i}")
        globalValue = globalView[i]
        if (not isfinite(globalValue)) or globalValue <= 0.0:
            raise ValueError(f"globalModelVariances must contain finite positive values at index {i}")
        if useNoise:
            noiseValue = noiseView[i]
            if (not isfinite(noiseValue)) or noiseValue <= 0.0:
                raise ValueError(f"localLogVarianceNoise must contain finite positive values at index {i}")
        if usePooledThin:
            if sampleView[i] < 0:
                raise ValueError(f"sampleIndex must contain nonnegative values at index {i}")
            if chromosomeView[i] < 0:
                raise ValueError(f"chromosomeIndex must contain nonnegative values at index {i}")
            if startView[i] < 0:
                raise ValueError(f"blockStarts must contain nonnegative values at index {i}")
        candidateCount += 1

    candidateArr = np.empty(candidateCount, dtype=np.intp)
    candidateView = candidateArr
    if usePooledThin and n > 0:
        capacity = 8
        while capacity < n:
            capacity <<= 1
        capacity <<= 1
        usedPtr = <uint8_t*>malloc(capacity * sizeof(uint8_t))
        sampleKeyPtr = <int64_t*>malloc(capacity * sizeof(int64_t))
        chromosomeKeyPtr = <int64_t*>malloc(capacity * sizeof(int64_t))
        binKeyPtr = <int64_t*>malloc(capacity * sizeof(int64_t))
        if usedPtr == NULL or sampleKeyPtr == NULL or chromosomeKeyPtr == NULL or binKeyPtr == NULL:
            if usedPtr != NULL:
                free(usedPtr)
            if sampleKeyPtr != NULL:
                free(sampleKeyPtr)
            if chromosomeKeyPtr != NULL:
                free(chromosomeKeyPtr)
            if binKeyPtr != NULL:
                free(binKeyPtr)
            raise MemoryError()
        for tableIdx in range(capacity):
            usedPtr[tableIdx] = <uint8_t>0
        for i in range(n):
            sampleId = sampleView[i]
            chromosomeId = chromosomeView[i]
            startValue = startView[i]
            binId = startValue // binSize
            if _insertEBPriorKey(
                usedPtr,
                sampleKeyPtr,
                chromosomeKeyPtr,
                binKeyPtr,
                capacity,
                sampleId,
                chromosomeId,
                binId,
            ):
                candidateView[outCount] = i
                outCount += 1
        free(usedPtr)
        free(sampleKeyPtr)
        free(chromosomeKeyPtr)
        free(binKeyPtr)
    else:
        for i in range(n):
            candidateView[outCount] = i
            outCount += 1

    return candidateArr[:outCount], int(candidateCount)


cpdef tuple cEBPriorStrengthLogRatiosFromCandidateIdx(
    object localModelVariances,
    object globalModelVariances,
    object candidateIdx,
    object localLogVarianceNoise=None,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] localArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] globalArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] noiseArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] candidateArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logRatioArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] noiseOutArr
    cdef double[::1] localView
    cdef double[::1] globalView
    cdef double[::1] noiseView
    cdef cnp.intp_t[::1] candidateView
    cdef double[::1] logRatioView
    cdef double[::1] noiseOutView
    cdef Py_ssize_t n
    cdef Py_ssize_t candidateCount
    cdef Py_ssize_t i
    cdef Py_ssize_t idx
    cdef bint useNoise = localLogVarianceNoise is not None
    cdef double localValue
    cdef double globalValue
    cdef double noiseValue

    localArr = np.ascontiguousarray(
        np.asarray(localModelVariances, dtype=np.float64).ravel()
    )
    globalArr = np.ascontiguousarray(
        np.asarray(globalModelVariances, dtype=np.float64).ravel()
    )
    if localArr.shape[0] != globalArr.shape[0]:
        raise ValueError("localModelVariances and globalModelVariances must have the same shape")
    n = localArr.shape[0]
    localView = localArr
    globalView = globalArr
    if useNoise:
        noiseArr = np.ascontiguousarray(
            np.asarray(localLogVarianceNoise, dtype=np.float64).ravel()
        )
        if noiseArr.shape[0] != n:
            raise ValueError("localLogVarianceNoise must match localModelVariances")
        noiseView = noiseArr
    candidateArr = np.ascontiguousarray(
        np.asarray(candidateIdx, dtype=np.intp).ravel()
    )
    candidateCount = candidateArr.shape[0]
    candidateView = candidateArr
    logRatioArr = np.empty(candidateCount, dtype=np.float64)
    logRatioView = logRatioArr
    if useNoise:
        noiseOutArr = np.empty(candidateCount, dtype=np.float64)
        noiseOutView = noiseOutArr

    for i in range(candidateCount):
        idx = candidateView[i]
        if idx < 0 or idx >= n:
            raise IndexError("candidateIdx contains an out-of-bounds index")
        localValue = localView[idx]
        if (not isfinite(localValue)) or localValue <= 0.0:
            raise ValueError(f"localModelVariances must contain finite positive values at index {idx}")
        globalValue = globalView[idx]
        if (not isfinite(globalValue)) or globalValue <= 0.0:
            raise ValueError(f"globalModelVariances must contain finite positive values at index {idx}")
        logRatioView[i] = log(localValue / globalValue)
        if useNoise:
            noiseValue = noiseView[idx]
            if (not isfinite(noiseValue)) or noiseValue <= 0.0:
                raise ValueError(f"localLogVarianceNoise must contain finite positive values at index {idx}")
            noiseOutView[i] = noiseValue

    if useNoise:
        return logRatioArr, noiseOutArr
    return logRatioArr, None


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


cdef bint _isStandardAutosomeName(object chromosome):
    return <bint>misc_util.isStandardAutosomalChromosome(chromosome)


cdef long long _dependenceValidatedInteger(object value, object name):
    cdef long long integerValue
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError("%s must be an integer" % str(name))
    integerValue = int(value)
    if integerValue < -2147483648 or integerValue > 2147483647:
        raise ValueError("%s is outside the supported integer range" % str(name))
    return integerValue


cdef double _dependenceValidatedReal(object value, object name):
    cdef double realValue
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError("%s must be a real scalar" % str(name))
    realValue = float(value)
    if not isfinite(realValue):
        raise ValueError("%s must be finite" % str(name))
    return realValue


cdef int _dependenceAutosomeOrdinal(object chromosome):
    cdef str name = str(chromosome).strip()
    if name.lower().startswith("chr"):
        name = name[3:]
    return int(name)


cdef list _dependenceSortedAutosomeNames(object names):
    cdef list keyedNames = []
    cdef object name
    for name in names:
        keyedNames.append(
            (_dependenceAutosomeOrdinal(name), "chr%d" % _dependenceAutosomeOrdinal(name))
        )
    keyedNames.sort()
    return [str(value[1]) for value in keyedNames]


cdef int _nextPowerOfTwoInt(int value):
    if value <= 1:
        return 1
    return 1 << int(ceil(log2(<double>value)))


cdef double _dependenceGaussianRadiusCorrection(double threshold):
    return 3.0 / (2.0 * sqrt(-log(threshold)))


cdef int _dependenceNearestOddBins(int targetBP, int intervalSizeBP):
    cdef int lowerOdd
    cdef int upperOdd
    cdef double targetBins = <double>targetBP / <double>intervalSizeBP
    lowerOdd = max(1, int(floor(targetBins)))
    if lowerOdd % 2 == 0:
        lowerOdd -= 1
    lowerOdd = max(1, lowerOdd)
    upperOdd = lowerOdd + 2
    if fabs((<double>upperOdd * intervalSizeBP) - targetBP) <= fabs(
        (<double>lowerOdd * intervalSizeBP) - targetBP
    ):
        return upperOdd
    return lowerOdd


cdef list _dependenceUniqueRows(list matrices, int rowCount):
    cdef dict digestRows = {}
    cdef list retainedRows = []
    cdef object digest
    cdef object digestKey
    cdef object matrix
    cdef object rowValues
    cdef object matchedRow
    cdef list candidateRows
    cdef int rowIndex
    cdef bint equalRows

    for rowIndex in range(rowCount):
        digest = hashlib.sha256()
        for matrix in matrices:
            digest.update(str(matrix.dtype.str).encode("utf-8"))
            digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
            rowValues = np.asarray(matrix[rowIndex])
            digest.update(rowValues.tobytes(order="C"))
        digestKey = digest.digest()
        candidateRows = digestRows.get(digestKey, [])
        equalRows = False
        for matchedRow in candidateRows:
            equalRows = True
            for matrix in matrices:
                if not np.array_equal(
                    np.asarray(matrix[rowIndex]),
                    np.asarray(matrix[int(matchedRow)]),
                    equal_nan=True,
                ):
                    equalRows = False
                    break
            if equalRows:
                break
        if not equalRows:
            retainedRows.append(int(rowIndex))
            candidateRows.append(int(rowIndex))
            digestRows[digestKey] = candidateRows
    return retainedRows


cdef object _dependenceKMQuantile(object values, object censored, double quantile):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] valueArray = np.ascontiguousarray(
        np.asarray(values, dtype=np.float64).ravel(),
        dtype=np.float64,
    )
    cdef object censorArray = np.asarray(censored, dtype=np.bool_).ravel()
    cdef object order
    cdef object sortedValues
    cdef object sortedCensored
    cdef Py_ssize_t sampleCount = valueArray.size
    cdef Py_ssize_t start = 0
    cdef Py_ssize_t stop
    cdef Py_ssize_t atRisk = sampleCount
    cdef Py_ssize_t eventCount
    cdef double survival = 1.0
    cdef double timeValue

    if sampleCount <= 0 or censorArray.size != sampleCount:
        return None
    order = np.argsort(valueArray, kind="mergesort")
    sortedValues = valueArray[order]
    sortedCensored = censorArray[order]
    for timeValue in np.unique(sortedValues):
        stop = int(np.searchsorted(sortedValues, timeValue, side="right"))
        eventCount = int(np.count_nonzero(~sortedCensored[start:stop]))
        if eventCount > 0:
            survival *= 1.0 - (<double>eventCount / <double>atRisk)
            if (1.0 - survival) + 1.0e-15 >= quantile:
                return float(timeValue)
        atRisk -= stop - start
        start = stop
    return None


cdef object _dependenceKMSurvivalAt(
    object values,
    object censored,
    object evaluationTimes,
):
    cdef object valueArray = np.asarray(values, dtype=np.float64).ravel()
    cdef object censorArray = np.asarray(censored, dtype=np.bool_).ravel()
    cdef object grid = np.asarray(evaluationTimes, dtype=np.float64).ravel()
    cdef object order
    cdef object sortedValues
    cdef object sortedCensored
    cdef object uniqueTimes
    cdef object survivalSteps
    cdef object positions
    cdef object output
    cdef object validPositions
    cdef Py_ssize_t start = 0
    cdef Py_ssize_t stop
    cdef Py_ssize_t atRisk
    cdef Py_ssize_t eventCount
    cdef Py_ssize_t timeIndex
    cdef double survival = 1.0
    cdef double timeValue

    if valueArray.size <= 0 or censorArray.size != valueArray.size:
        raise ValueError("Kaplan-Meier inputs must have equal positive lengths")
    order = np.argsort(valueArray, kind="mergesort")
    sortedValues = valueArray[order]
    sortedCensored = censorArray[order]
    uniqueTimes = np.unique(sortedValues)
    survivalSteps = np.ones(uniqueTimes.size, dtype=np.float64)
    atRisk = int(valueArray.size)
    for timeIndex in range(int(uniqueTimes.size)):
        timeValue = float(uniqueTimes[timeIndex])
        stop = int(np.searchsorted(sortedValues, timeValue, side="right"))
        eventCount = int(np.count_nonzero(~sortedCensored[start:stop]))
        if eventCount > 0:
            survival *= 1.0 - (<double>eventCount / <double>atRisk)
        survivalSteps[timeIndex] = survival
        atRisk -= stop - start
        start = stop
    positions = np.searchsorted(uniqueTimes, grid, side="right") - 1
    output = np.ones(grid.size, dtype=np.float64)
    validPositions = positions >= 0
    output[validPositions] = survivalSteps[positions[validPositions]]
    return output


cdef tuple _dependencePeriodDiagnostics(
    cnp.ndarray[cnp.float64_t, ndim=1] acf,
    int intervalSizeBP,
    int crossingLag,
    int crossingPersistenceBins,
):
    cdef int acfCount = int(acf.size)
    cdef int envelopeBins
    cdef object envelope
    cdef object residual
    cdef object tapers
    cdef object spectra
    cdef object power
    cdef object frequencies
    cdef object periods
    cdef object bandMask
    cdef object bandIndices
    cdef object revivalValues
    cdef int frequencyIndex
    cdef int revivalStart
    cdef double periodMinimumBP = max(2.0 * intervalSizeBP, 150.0)
    cdef double periodMaximumBP = 500.0
    cdef double totalPower
    cdef object dominantPeriodBP = None
    cdef object oscillationStrength = None
    cdef object postCrossingRevival = None

    if acfCount >= 3 and periodMinimumBP <= periodMaximumBP:
        envelopeBins = _dependenceNearestOddBins(2000, intervalSizeBP)
        if envelopeBins > acfCount:
            envelopeBins = acfCount if acfCount % 2 == 1 else acfCount - 1
        if envelopeBins >= 3:
            envelope = signal.savgol_filter(
                np.asarray(acf, dtype=np.float64),
                envelopeBins,
                2,
                mode="interp",
            )
            residual = np.asarray(acf, dtype=np.float64) - envelope
            tapers = __dependenceDPSSCache.get(acfCount)
            if tapers is None:
                tapers = np.asarray(
                    signal.windows.dpss(
                        acfCount,
                        2.5,
                        Kmax=3,
                        sym=False,
                        norm=2,
                    ),
                    dtype=np.float64,
                )
                __dependenceDPSSCache[acfCount] = tapers
            spectra = np.fft.rfft(tapers * residual[np.newaxis, :], axis=1)
            power = np.mean(np.square(np.abs(spectra)), axis=0)
            frequencies = np.fft.rfftfreq(acfCount, d=intervalSizeBP)
            periods = np.full(frequencies.size, np.inf, dtype=np.float64)
            periods[frequencies > 0.0] = 1.0 / frequencies[frequencies > 0.0]
            bandMask = (
                (periods >= periodMinimumBP)
                & (periods <= periodMaximumBP)
                & ((acfCount * intervalSizeBP) >= (4.0 * periods))
            )
            bandIndices = np.flatnonzero(bandMask)
            if bandIndices.size > 0:
                frequencyIndex = int(
                    bandIndices[int(np.argmax(power[bandIndices]))]
                )
                totalPower = float(np.sum(power[bandIndices]))
                if totalPower > 0.0:
                    dominantPeriodBP = float(periods[frequencyIndex])
                    oscillationStrength = float(power[frequencyIndex]) / totalPower
    if crossingLag > 0:
        revivalStart = crossingLag + crossingPersistenceBins - 1
        if revivalStart < acf.size:
            revivalValues = np.abs(acf[revivalStart:])
            if revivalValues.size > 0:
                postCrossingRevival = float(np.max(revivalValues))
    return (
        dominantPeriodBP,
        oscillationStrength,
        postCrossingRevival,
    )


cdef object _dependenceFinitePairWindow(
    object windowMatrix,
    int intervalSizeBP,
    int maxLagBins,
    double acfThreshold,
    int acfSmoothingBins,
    int crossingPersistenceBins,
    int minFinitePairs,
    double minFinitePairCoverage,
    double gaussianRadiusCorrection,
):
    cdef object matrix = np.asarray(windowMatrix)
    cdef list rowACFs = []
    cdef list rowPairCounts = []
    cdef list rowPairCoverages = []
    cdef object values
    cdef object finiteMask
    cdef object finiteValues
    cdef object clipped
    cdef object centered
    cdef object maskFloat
    cdef object valueFFT
    cdef object maskFFT
    cdef object autoSums
    cdef object pairCounts
    cdef object pairCoverage
    cdef object admissible
    cdef object covariance
    cdef object acf
    cdef object acfMatrix
    cdef object pairMatrix
    cdef object coverageMatrix
    cdef object contributingMask
    cdef object contributingCounts
    cdef object usedPairCounts
    cdef object usedPairCoverages
    cdef object absoluteACF
    cdef object prefixSums
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pooledACF
    cdef cnp.ndarray[cnp.float64_t, ndim=1] crossingACF
    cdef int rowIndex
    cdef int fftSize = _nextPowerOfTwoInt((2 * int(matrix.shape[1])) - 1)
    cdef int start
    cdef int lagIndex
    cdef int lowerIndex
    cdef int upperIndex
    cdef int crossingLag = -1
    cdef int supportCapLag = 0
    cdef int lastCrossingStart
    cdef int smoothingHalfWidth = (acfSmoothingBins - 1) // 2
    cdef int useEndLag
    cdef int stencilStartLag
    cdef int stencilEndLag
    cdef int validRowCount
    cdef int quorum
    cdef int validRowsAtCrossing
    cdef bint crossingFound
    cdef double lowerClip
    cdef double upperClip
    cdef double lagZero
    cdef double finitePairMinimumUsed
    cdef double finitePairCoverageMinimumUsed
    cdef double rawCrossingLagBP
    cdef double censorLagBP
    cdef double radiusBP
    cdef object dominantPeriodBP
    cdef object oscillationStrength
    cdef object postCrossingRevival
    cdef object censorReason

    for rowIndex in range(int(matrix.shape[0])):
        values = np.asarray(matrix[rowIndex], dtype=np.float64)
        finiteMask = np.isfinite(values)
        if int(np.count_nonzero(finiteMask)) < 2:
            continue
        finiteValues = np.asarray(values[finiteMask], dtype=np.float64)
        lowerClip, upperClip = np.quantile(finiteValues, [0.005, 0.995])
        clipped = np.zeros(values.size, dtype=np.float64)
        clipped[finiteMask] = np.clip(
            finiteValues,
            float(lowerClip),
            float(upperClip),
        )
        clipped[finiteMask] -= float(np.mean(clipped[finiteMask]))
        maskFloat = np.asarray(finiteMask, dtype=np.float64)
        valueFFT = np.fft.rfft(clipped, n=fftSize)
        maskFFT = np.fft.rfft(maskFloat, n=fftSize)
        autoSums = np.asarray(
            np.fft.irfft(valueFFT * np.conjugate(valueFFT), n=fftSize)[
                :maxLagBins + 1
            ],
            dtype=np.float64,
        )
        pairCounts = np.asarray(
            np.rint(
                np.fft.irfft(maskFFT * np.conjugate(maskFFT), n=fftSize)[
                    :maxLagBins + 1
                ]
            ),
            dtype=np.float64,
        )
        pairCoverage = pairCounts / np.arange(
            int(matrix.shape[1]),
            int(matrix.shape[1]) - maxLagBins - 1,
            -1,
            dtype=np.float64,
        )
        covariance = np.full(maxLagBins + 1, np.nan, dtype=np.float64)
        admissible = (
            (pairCounts >= minFinitePairs)
            & (pairCoverage >= minFinitePairCoverage)
        )
        covariance[admissible] = autoSums[admissible] / pairCounts[admissible]
        lagZero = float(covariance[0])
        if (not isfinite(lagZero)) or lagZero <= 0.0:
            continue
        acf = np.asarray(covariance[1:] / lagZero, dtype=np.float64)
        rowACFs.append(acf)
        rowPairCounts.append(np.asarray(pairCounts[1:], dtype=np.float64))
        rowPairCoverages.append(np.asarray(pairCoverage[1:], dtype=np.float64))

    validRowCount = len(rowACFs)
    if validRowCount <= 0:
        return None
    quorum = max(1, int(ceil(<double>validRowCount / 2.0)))
    acfMatrix = np.asarray(rowACFs, dtype=np.float64)
    pairMatrix = np.asarray(rowPairCounts, dtype=np.float64)
    coverageMatrix = np.asarray(rowPairCoverages, dtype=np.float64)
    contributingMask = np.isfinite(acfMatrix)
    contributingCounts = np.sum(contributingMask, axis=0)
    pooledACF = np.full(maxLagBins, np.nan, dtype=np.float64)
    finiteMask = np.flatnonzero(contributingCounts < quorum)
    supportCapLag = (
        int(finiteMask[0]) if finiteMask.size > 0 else maxLagBins
    )
    if supportCapLag <= 0:
        return None
    if validRowCount == 1:
        pooledACF[:supportCapLag] = acfMatrix[0, :supportCapLag]
    else:
        pooledACF[:supportCapLag] = np.nanmedian(
            acfMatrix[:, :supportCapLag],
            axis=0,
        )

    crossingACF = np.full(supportCapLag + 1, np.nan, dtype=np.float64)
    absoluteACF = np.abs(np.asarray(pooledACF[:supportCapLag], dtype=np.float64))
    prefixSums = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(absoluteACF))
    )
    for lagIndex in range(
        1 + smoothingHalfWidth,
        supportCapLag - smoothingHalfWidth + 1,
    ):
        lowerIndex = lagIndex - smoothingHalfWidth - 1
        upperIndex = lagIndex + smoothingHalfWidth
        crossingACF[lagIndex] = (
            float(prefixSums[upperIndex] - prefixSums[lowerIndex])
            / <double>acfSmoothingBins
        )

    lastCrossingStart = (
        supportCapLag - smoothingHalfWidth - crossingPersistenceBins + 1
    )
    if lastCrossingStart < 1 + smoothingHalfWidth:
        return None
    for start in range(1 + smoothingHalfWidth, lastCrossingStart + 1):
        crossingFound = True
        for lagIndex in range(crossingPersistenceBins):
            if (
                (not isfinite(crossingACF[start + lagIndex]))
                or crossingACF[start + lagIndex] >= acfThreshold
            ):
                crossingFound = False
                break
        if crossingFound:
            crossingLag = start
            break

    if crossingLag > 0:
        useEndLag = crossingLag + crossingPersistenceBins - 1 + smoothingHalfWidth
        stencilStartLag = crossingLag - smoothingHalfWidth
        stencilEndLag = useEndLag
    else:
        useEndLag = supportCapLag
        stencilStartLag = lastCrossingStart - smoothingHalfWidth
        stencilEndLag = supportCapLag
    usedPairCounts = pairMatrix[:, :useEndLag][contributingMask[:, :useEndLag]]
    usedPairCoverages = coverageMatrix[:, :useEndLag][contributingMask[:, :useEndLag]]
    if usedPairCounts.size <= 0 or usedPairCoverages.size <= 0:
        return None
    finitePairMinimumUsed = float(np.min(usedPairCounts))
    finitePairCoverageMinimumUsed = float(np.min(usedPairCoverages))
    validRowsAtCrossing = int(
        np.min(contributingCounts[stencilStartLag - 1:stencilEndLag])
    )

    if crossingLag > 0:
        rawCrossingLagBP = <double>crossingLag * <double>intervalSizeBP
        censorLagBP = NAN
        censorReason = "none"
    else:
        rawCrossingLagBP = NAN
        censorLagBP = <double>lastCrossingStart * <double>intervalSizeBP
        censorReason = "maxLag" if supportCapLag >= maxLagBins else "support"
    radiusBP = (
        rawCrossingLagBP if crossingLag > 0 else censorLagBP
    ) * gaussianRadiusCorrection
    (
        dominantPeriodBP,
        oscillationStrength,
        postCrossingRevival,
    ) = _dependencePeriodDiagnostics(
        np.ascontiguousarray(pooledACF[:supportCapLag], dtype=np.float64),
        intervalSizeBP,
        crossingLag,
        crossingPersistenceBins,
    )
    return {
        "rawCrossingLagBP": None if crossingLag < 0 else float(rawCrossingLagBP),
        "censorLagBP": None if crossingLag > 0 else float(censorLagBP),
        "gaussianEquivalentRadiusBP": float(radiusBP),
        "rightCensored": bool(crossingLag < 0),
        "censorReason": str(censorReason),
        "supportCapLagBP": int(supportCapLag * intervalSizeBP),
        "finitePairMinimumUsed": float(finitePairMinimumUsed),
        "finitePairCoverageMinimumUsed": float(finitePairCoverageMinimumUsed),
        "validRowCount": int(validRowCount),
        "validRowsAtCrossing": int(validRowsAtCrossing),
        "dominantACFPeriodBP": dominantPeriodBP,
        "oscillationStrength": oscillationStrength,
        "postCrossingRevival": postCrossingRevival,
    }


cdef tuple _dependenceCoordinateRuns(
    list selectedWindows,
    dict selectedByChromosome,
):
    cdef list runs = []
    cdef list keyedChromosomes = []
    cdef list keyedIndices
    cdef list chromosomeIndices
    cdef list run
    cdef object chromosome
    cdef int windowIndex
    cdef int leftIndex
    cdef int adjacencyCount = 0
    cdef int longestRun = 1

    for chromosome in selectedByChromosome:
        keyedChromosomes.append(
            (_dependenceAutosomeOrdinal(chromosome), str(chromosome))
        )
    keyedChromosomes.sort()
    for chromosomeOrdinal, chromosome in keyedChromosomes:
        keyedIndices = []
        for windowIndex in selectedByChromosome[str(chromosome)]:
            keyedIndices.append(
                (int(selectedWindows[int(windowIndex)]["startBP"]), int(windowIndex))
            )
        keyedIndices.sort()
        chromosomeIndices = []
        for value in keyedIndices:
            chromosomeIndices.append(int(value[1]))
        if len(chromosomeIndices) <= 0:
            continue
        run = [int(chromosomeIndices[0])]
        leftIndex = int(chromosomeIndices[0])
        for windowIndex in chromosomeIndices[1:]:
            windowIndex = int(windowIndex)
            if int(selectedWindows[windowIndex]["startBP"]) == int(
                selectedWindows[leftIndex]["endBP"]
            ):
                run.append(windowIndex)
                adjacencyCount += 1
            else:
                runs.append(run)
                longestRun = max(longestRun, len(run))
                run = [windowIndex]
            leftIndex = windowIndex
        runs.append(run)
        longestRun = max(longestRun, len(run))
    return runs, int(adjacencyCount), int(longestRun)


cdef double _dependencePolitisWhiteComponent(
    object componentValues,
    list coordinateRuns,
    int longestRun,
):
    cdef object values = np.asarray(componentValues, dtype=np.float64).ravel()
    cdef object standardized
    cdef object autocovariance
    cdef object runValues
    cdef object run
    cdef int sampleCount = int(values.size)
    cdef int kN
    cdef int maxLag
    cdef int lag
    cdef int cutoffLag
    cdef int candidateStart
    cdef int mHat
    cdef int mValue
    cdef int pairCount
    cdef double standardDeviation
    cdef double criticalValue
    cdef double productSum
    cdef double weight
    cdef double gValue = 0.0
    cdef double longRunVariance
    cdef double denominator
    cdef double blockLength
    cdef double blockMaximum

    if sampleCount <= 1 or longestRun <= 1:
        return 1.0
    standardDeviation = float(np.std(values))
    if (not isfinite(standardDeviation)) or standardDeviation <= 1.0e-12:
        return 1.0
    standardized = (values - float(np.mean(values))) / standardDeviation
    kN = max(5, int(ceil(log10(<double>sampleCount))))
    maxLag = min(
        longestRun - 1,
        int(ceil(sqrt(<double>sampleCount))) + kN,
    )
    if maxLag <= 0:
        return 1.0
    autocovariance = np.zeros(maxLag + 1, dtype=np.float64)
    autocovariance[0] = float(np.mean(np.square(standardized)))
    if float(autocovariance[0]) <= 0.0:
        return 1.0
    for lag in range(1, maxLag + 1):
        productSum = 0.0
        pairCount = 0
        for run in coordinateRuns:
            if len(run) <= lag:
                continue
            runValues = standardized[np.asarray(run, dtype=np.int64)]
            productSum += float(np.dot(runValues[:-lag], runValues[lag:]))
            pairCount += len(run) - lag
        if pairCount > 0:
            autocovariance[lag] = productSum / <double>pairCount

    criticalValue = 2.0 * sqrt(log10(<double>sampleCount) / <double>sampleCount)
    mHat = 0
    cutoffLag = min(kN, maxLag)
    for lag in range(cutoffLag, maxLag + 1):
        candidateStart = lag - cutoffLag + 1
        if bool(
            np.all(
                np.abs(
                    autocovariance[candidateStart:lag + 1]
                    / float(autocovariance[0])
                ) < criticalValue
            )
        ):
            mHat = candidateStart
            break
    if mHat <= 0:
        for lag in range(maxLag, 0, -1):
            if fabs(
                float(autocovariance[lag]) / float(autocovariance[0])
            ) > criticalValue:
                mHat = lag
                break
        if mHat <= 0:
            mHat = 1
    mValue = min(maxLag, 2 * mHat)
    longRunVariance = float(autocovariance[0])
    for lag in range(1, mValue + 1):
        weight = 1.0 if (<double>lag / <double>mValue) <= 0.5 else (
            2.0 * (1.0 - (<double>lag / <double>mValue))
        )
        gValue += 2.0 * weight * lag * float(autocovariance[lag])
        longRunVariance += 2.0 * weight * float(autocovariance[lag])
    denominator = 2.0 * longRunVariance * longRunVariance
    if denominator <= 0.0 or (not isfinite(denominator)):
        return 1.0
    blockLength = pow(
        (2.0 * gValue * gValue) / denominator,
        1.0 / 3.0,
    ) * pow(<double>sampleCount, 1.0 / 3.0)
    blockMaximum = max(
        1.0,
        min(3.0 * sqrt(<double>sampleCount), <double>sampleCount / 3.0),
    )
    if not isfinite(blockLength):
        return 1.0
    return max(1.0, min(blockLength, blockMaximum, <double>longestRun))


cdef tuple _dependenceBootstrapGeometry(
    list selectedWindows,
    dict selectedByChromosome,
    object radiusValues,
    object radiusCensored,
):
    cdef list coordinateRuns
    cdef int adjacencyCount
    cdef int longestRun
    cdef object logRadius
    cdef object eventValues
    cdef object scoreValues
    cdef double radiusBlock
    cdef double eventBlock
    cdef double scoreBlock
    cdef int blockLength

    coordinateRuns, adjacencyCount, longestRun = _dependenceCoordinateRuns(
        selectedWindows,
        selectedByChromosome,
    )
    logRadius = np.log(np.asarray(radiusValues, dtype=np.float64))
    eventValues = (~np.asarray(radiusCensored, dtype=np.bool_)).astype(np.float64)
    scoreValues = np.asarray(
        [float(window["score"]) for window in selectedWindows],
        dtype=np.float64,
    )
    radiusBlock = _dependencePolitisWhiteComponent(
        logRadius,
        coordinateRuns,
        longestRun,
    )
    eventBlock = _dependencePolitisWhiteComponent(
        eventValues,
        coordinateRuns,
        longestRun,
    )
    scoreBlock = _dependencePolitisWhiteComponent(
        scoreValues,
        coordinateRuns,
        longestRun,
    )
    blockLength = max(
        1,
        min(
            longestRun,
            int(ceil(max(radiusBlock, eventBlock, scoreBlock))),
        ),
    )
    return int(blockLength), int(adjacencyCount), int(longestRun)


cpdef tuple cchooseDependenceSpan(
    object chromosomeNames,
    object chromosomeMatrices,
    object intervalSizeBP,
    object windowBP=100000,
    object windowCount=256,
    object maxLagBP=50000,
    object workingQuantile=0.75,
    object bootstrapDraws=500,
    object randSeed=1729,
    object minWindowCount=20,
    object acfThreshold=0.1,
    object acfSmoothingBP=250,
    object crossingPersistenceBP=250,
    object minFinitePairs=200,
    object minFinitePairCoverage=0.5,
):
    cdef list names
    cdef list matrices
    cdef list matrixArrays = []
    cdef list eligibleNames = []
    cdef list eligibleMatrices = []
    cdef list eligibleRecords = []
    cdef list excludedNames = []
    cdef list retainedRows
    cdef list candidateWindows = []
    cdef list selectedWindows = []
    cdef list radiusValues = []
    cdef list radiusCensored = []
    cdef list finitePairMinimaUsed = []
    cdef list finitePairCoverageMinimaUsed = []
    cdef list validRowsAtCrossingValues = []
    cdef list censorTimesBP = []
    cdef list dominantPeriods = []
    cdef list oscillationStrengths = []
    cdef list revivalValues = []
    cdef list chromosomesUsed
    cdef list bootstrapMedianRadiusBP = []
    cdef list bootstrapWorkingSpanBP = []
    cdef list bootstrapDistances = []
    cdef list jumpClosureIndices = []
    cdef dict selectedByChromosome = {}
    cdef dict selectedCountsByChromosome = {}
    cdef object seenAutosomeOrdinals = set()
    cdef object matrix
    cdef object windowMatrix
    cdef object values
    cdef object finiteMask
    cdef object result
    cdef object candidate
    cdef object record
    cdef object window
    cdef object chromosome
    cdef object selectionSeed
    cdef object bootstrapSeed
    cdef object selectionRNG
    cdef object bootstrapRNG
    cdef object sampledChromosomeIndices
    cdef object fullGrid
    cdef object fullSurvival
    cdef object fullTransformed
    cdef object bandDomainMask
    cdef object drawSurvival
    cdef object drawTransformed
    cdef object lowerSurvival
    cdef object upperSurvival
    cdef object confidenceMask
    cdef object fullMedian
    cdef object fullWorkingSpan
    cdef object drawMedian
    cdef object drawWorkingSpan
    cdef list drawValues
    cdef list drawCensored
    cdef list chromosomeWindows
    cdef list chromosomePositions
    cdef list rankingScores
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] exponentialKeys
    cdef cnp.float64_t[::1] exponentialKeyView
    cdef Py_ssize_t i
    cdef Py_ssize_t candidateCount
    cdef Py_ssize_t rankStart
    cdef Py_ssize_t rankEnd
    cdef int rowCount = -1
    cdef int windowBins
    cdef int maxLagBins
    cdef int acfSmoothingBins
    cdef int crossingPersistenceBins
    cdef int intervalSizeBPValue
    cdef int windowBPValue
    cdef int windowCountValue
    cdef int maxLagBPValue
    cdef int bootstrapDrawsValue
    cdef int randSeedValue
    cdef int minWindowCountValue
    cdef int acfSmoothingBPValue
    cdef int crossingPersistenceBPValue
    cdef int minFinitePairsValue
    cdef int chromosomeIndex
    cdef int chromosomeOrdinal
    cdef int fullWindowCount
    cdef int windowIndex
    cdef int startBin
    cdef int endBin
    cdef int startBP
    cdef int endBP
    cdef int retainedRow
    cdef int selectedAutosomeCount
    cdef int rightCensoredWindowCount
    cdef int crossedWindowCount
    cdef int supportCensoredWindowCount
    cdef int capCensoredWindowCount
    cdef int selectedAdjacencyCount
    cdef int selectedLongestRun
    cdef int bootstrapBlockLengthWindows
    cdef int resolvedMedianDraws = 0
    cdef int resolvedWorkingDraws = 0
    cdef int resolvedJointDraws = 0
    cdef int requiredResolvedDraws
    cdef int drawIndex
    cdef int sampledChromosomeIndex
    cdef int sampledWindowIndex
    cdef int currentPosition
    cdef int nextPosition
    cdef int positionIndex
    cdef int finiteCount
    cdef int bandIndex
    cdef int evaluatedCandidateWindowCount = 0
    cdef double score
    cdef double rowScore
    cdef double positiveSignalRank
    cdef double rankWeight
    cdef double selectionKey
    cdef double rankCoverageMinimum
    cdef double workingQuantileValue
    cdef double acfThresholdValue
    cdef double minFinitePairCoverageValue
    cdef double gaussianRadiusCorrection
    cdef double estimateBP
    cdef double lowerBP
    cdef double upperBP
    cdef double workingSpanBP
    cdef double censorFraction
    cdef object dominantACFPeriodBPMedian = None
    cdef object oscillationStrength = None
    cdef object postCrossingRevival = None
    cdef double epsilonValue
    cdef double bandCriticalValue
    cdef double restartProbability
    cdef double leftBandTransform
    cdef double leftLowerSurvival
    cdef double leftUpperSurvival
    cdef bint survivalBandJumpClosureUsed = False
    cdef dict diagnostics

    try:
        names = list(chromosomeNames)
        matrices = list(chromosomeMatrices)
    except Exception as exc:
        raise ValueError("chromosome inputs must be finite sequences") from exc
    if len(names) <= 0 or len(names) != len(matrices):
        raise ValueError("chromosome inputs must be nonempty and have equal lengths")
    intervalSizeBPValue = <int>_dependenceValidatedInteger(
        intervalSizeBP,
        "intervalSizeBP",
    )
    windowBPValue = <int>_dependenceValidatedInteger(windowBP, "windowBP")
    windowCountValue = <int>_dependenceValidatedInteger(
        windowCount,
        "windowCount",
    )
    maxLagBPValue = <int>_dependenceValidatedInteger(maxLagBP, "maxLagBP")
    bootstrapDrawsValue = <int>_dependenceValidatedInteger(
        bootstrapDraws,
        "bootstrapDraws",
    )
    randSeedValue = <int>_dependenceValidatedInteger(randSeed, "randSeed")
    minWindowCountValue = <int>_dependenceValidatedInteger(
        minWindowCount,
        "minWindowCount",
    )
    acfSmoothingBPValue = <int>_dependenceValidatedInteger(
        acfSmoothingBP,
        "acfSmoothingBP",
    )
    crossingPersistenceBPValue = <int>_dependenceValidatedInteger(
        crossingPersistenceBP,
        "crossingPersistenceBP",
    )
    minFinitePairsValue = <int>_dependenceValidatedInteger(
        minFinitePairs,
        "minFinitePairs",
    )
    workingQuantileValue = _dependenceValidatedReal(
        workingQuantile,
        "workingQuantile",
    )
    acfThresholdValue = _dependenceValidatedReal(acfThreshold, "acfThreshold")
    minFinitePairCoverageValue = _dependenceValidatedReal(
        minFinitePairCoverage,
        "minFinitePairCoverage",
    )
    if intervalSizeBPValue <= 0:
        raise ValueError("intervalSizeBP must be positive")
    if windowBPValue <= 0:
        raise ValueError("windowBP must be positive")
    if maxLagBPValue <= 0 or maxLagBPValue > windowBPValue // 2:
        raise ValueError("maxLagBP must satisfy 0 < maxLagBP <= windowBP / 2")
    if windowCountValue <= 0 or bootstrapDrawsValue <= 0:
        raise ValueError("windowCount and bootstrapDraws must be positive")
    if randSeedValue < 0:
        raise ValueError("randSeed must be nonnegative")
    if minWindowCountValue <= 0 or minWindowCountValue > windowCountValue:
        raise ValueError("minWindowCount must satisfy 0 < minWindowCount <= windowCount")
    if workingQuantileValue <= 0.0 or workingQuantileValue >= 1.0:
        raise ValueError("workingQuantile must satisfy 0 < q < 1")
    if acfThresholdValue <= 0.0 or acfThresholdValue >= 1.0:
        raise ValueError("acfThreshold must satisfy 0 < x < 1")
    if acfSmoothingBPValue <= 0 or crossingPersistenceBPValue <= 0:
        raise ValueError("ACF smoothing and crossing persistence must be positive")
    if (
        acfSmoothingBPValue > maxLagBPValue
        or crossingPersistenceBPValue > maxLagBPValue
    ):
        raise ValueError("ACF smoothing and crossing persistence cannot exceed maxLagBP")
    if minFinitePairsValue <= 0:
        raise ValueError("minFinitePairs must be positive")
    if (
        minFinitePairCoverageValue <= 0.0
        or minFinitePairCoverageValue > 1.0
    ):
        raise ValueError("minFinitePairCoverage must satisfy 0 < x <= 1")
    if (
        windowBPValue % intervalSizeBPValue != 0
        or maxLagBPValue % intervalSizeBPValue != 0
    ):
        raise ValueError("windowBP and maxLagBP must be integer multiples of intervalSizeBP")
    windowBins = windowBPValue // intervalSizeBPValue
    maxLagBins = maxLagBPValue // intervalSizeBPValue
    acfSmoothingBins = _dependenceNearestOddBins(
        acfSmoothingBPValue,
        intervalSizeBPValue,
    )
    crossingPersistenceBins = max(
        1,
        int(
            ceil(
                <double>crossingPersistenceBPValue
                / <double>intervalSizeBPValue
            )
        ),
    )
    if minFinitePairsValue > windowBins:
        raise ValueError("minFinitePairs cannot exceed the window bin count")
    if (
        windowBins < 2
        or maxLagBins < crossingPersistenceBins
        or maxLagBins < acfSmoothingBins
    ):
        raise ValueError("physical window and lag settings are too short for the bin width")
    if maxLagBins > windowBins // 2:
        raise ValueError("binned maxLagBP must not exceed half of the binned windowBP")

    for matrix in matrices:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] < 2:
            raise ValueError("each chromosome matrix must be two-dimensional and nonempty")
        if matrix.dtype.kind not in "biuf":
            raise ValueError("chromosome matrices must contain real numeric values")
        if rowCount < 0:
            rowCount = int(matrix.shape[0])
        elif int(matrix.shape[0]) != rowCount:
            raise ValueError("chromosome matrices must have one shared row count")
        matrixArrays.append(matrix)

    for i in range(len(names)):
        matrix = matrixArrays[i]
        if _isStandardAutosomeName(names[i]):
            chromosomeOrdinal = _dependenceAutosomeOrdinal(names[i])
            if chromosomeOrdinal in seenAutosomeOrdinals:
                raise ValueError(
                    "duplicate canonical autosome chr%d" % chromosomeOrdinal
                )
            seenAutosomeOrdinals.add(chromosomeOrdinal)
            if int(matrix.shape[1]) >= windowBins:
                eligibleRecords.append(
                    (chromosomeOrdinal, "chr%d" % chromosomeOrdinal, matrix)
                )
            else:
                excludedNames.append("chr%d" % chromosomeOrdinal)
        else:
            excludedNames.append(str(names[i]))
    eligibleRecords.sort()
    for record in eligibleRecords:
        eligibleNames.append(str(record[1]))
        eligibleMatrices.append(record[2])
    if len(eligibleNames) <= 0:
        raise ValueError("dependence estimator found no eligible autosomes")

    retainedRows = _dependenceUniqueRows(eligibleMatrices, rowCount)
    if len(retainedRows) <= 0:
        raise RuntimeError("dependence estimator found no unique rows")
    gaussianRadiusCorrection = _dependenceGaussianRadiusCorrection(acfThresholdValue)

    rankCoverageMinimum = sqrt(minFinitePairCoverageValue)
    for chromosomeIndex in range(len(eligibleNames)):
        chromosome = eligibleNames[chromosomeIndex]
        chromosomeOrdinal = _dependenceAutosomeOrdinal(chromosome)
        matrix = eligibleMatrices[chromosomeIndex]
        fullWindowCount = int(matrix.shape[1]) // windowBins
        for windowIndex in range(fullWindowCount):
            startBin = windowIndex * windowBins
            endBin = startBin + windowBins
            rankingScores = []
            for retainedRow in retainedRows:
                values = np.asarray(matrix[retainedRow, startBin:endBin], dtype=np.float64)
                finiteMask = np.isfinite(values)
                finiteCount = int(np.count_nonzero(finiteMask))
                if (
                    finiteCount > 0
                    and (<double>finiteCount / <double>windowBins)
                    >= rankCoverageMinimum
                ):
                    rowScore = (
                        <double>windowBins
                        / <double>finiteCount
                        * float(np.sum(np.maximum(values[finiteMask], 0.0)))
                    )
                    rankingScores.append(float(rowScore))
            if len(rankingScores) <= 0:
                continue
            score = float(np.median(np.asarray(rankingScores, dtype=np.float64)))
            startBP = startBin * intervalSizeBPValue
            endBP = endBin * intervalSizeBPValue
            candidateWindows.append(
                (
                    -float(score),
                    int(chromosomeOrdinal),
                    int(startBP),
                    int(chromosomeIndex),
                    int(startBin),
                    int(endBin),
                    float(score),
                )
            )
    candidateWindows.sort()
    candidateCount = len(candidateWindows)
    selectionSeed, bootstrapSeed = np.random.SeedSequence(randSeedValue).spawn(2)
    selectionRNG = default_rng(selectionSeed)
    bootstrapRNG = default_rng(bootstrapSeed)
    exponentialKeys = np.ascontiguousarray(
        selectionRNG.exponential(size=candidateCount),
        dtype=np.float64,
    )
    exponentialKeyView = exponentialKeys
    rankStart = 0
    while rankStart < candidateCount:
        rankEnd = rankStart + 1
        while (
            rankEnd < candidateCount
            and candidateWindows[rankEnd][0] == candidateWindows[rankStart][0]
        ):
            rankEnd += 1
        positiveSignalRank = 0.5 * (
            <double>(rankStart + 1) + <double>rankEnd
        )
        rankWeight = <double>candidateCount - positiveSignalRank + 1.0
        for i in range(rankStart, rankEnd):
            candidate = candidateWindows[i]
            selectionKey = exponentialKeyView[i] / rankWeight
            candidateWindows[i] = (
                float(selectionKey),
                int(candidate[1]),
                int(candidate[2]),
                int(candidate[3]),
                int(candidate[4]),
                int(candidate[5]),
                float(candidate[6]),
                float(positiveSignalRank),
            )
        rankStart = rankEnd
    candidateWindows.sort()

    for candidate in candidateWindows:
        chromosomeIndex = int(candidate[3])
        startBin = int(candidate[4])
        endBin = int(candidate[5])
        matrix = eligibleMatrices[chromosomeIndex]
        windowMatrix = np.asarray(matrix[retainedRows, startBin:endBin])
        evaluatedCandidateWindowCount += 1
        result = _dependenceFinitePairWindow(
            windowMatrix,
            intervalSizeBPValue,
            maxLagBins,
            acfThresholdValue,
            acfSmoothingBins,
            crossingPersistenceBins,
            minFinitePairsValue,
            minFinitePairCoverageValue,
            gaussianRadiusCorrection,
        )
        if result is None:
            continue
        chromosome = eligibleNames[chromosomeIndex]
        startBP = startBin * intervalSizeBPValue
        endBP = endBin * intervalSizeBPValue
        window = {
            "chromosome": str(chromosome),
            "startBP": int(startBP),
            "endBP": int(endBP),
            "score": float(candidate[6]),
            "positiveSignalRank": float(candidate[7]),
            "rawCrossingLagBP": result["rawCrossingLagBP"],
            "censorLagBP": result["censorLagBP"],
            "gaussianEquivalentRadiusBP": float(result["gaussianEquivalentRadiusBP"]),
            "rightCensored": bool(result["rightCensored"]),
            "censorReason": str(result["censorReason"]),
            "supportCapLagBP": int(result["supportCapLagBP"]),
            "finitePairMinimumUsed": float(result["finitePairMinimumUsed"]),
            "finitePairCoverageMinimumUsed": float(
                result["finitePairCoverageMinimumUsed"]
            ),
            "validRowCount": int(result["validRowCount"]),
            "validRowsAtCrossing": int(result["validRowsAtCrossing"]),
            "dominantACFPeriodBP": result["dominantACFPeriodBP"],
            "oscillationStrength": result["oscillationStrength"],
            "postCrossingRevival": result["postCrossingRevival"],
        }
        selectedWindows.append(window)
        radiusValues.append(float(result["gaussianEquivalentRadiusBP"]))
        radiusCensored.append(bool(result["rightCensored"]))
        finitePairMinimaUsed.append(float(result["finitePairMinimumUsed"]))
        finitePairCoverageMinimaUsed.append(
            float(result["finitePairCoverageMinimumUsed"])
        )
        validRowsAtCrossingValues.append(int(result["validRowsAtCrossing"]))
        if bool(result["rightCensored"]):
            censorTimesBP.append(float(result["gaussianEquivalentRadiusBP"]))
        if result["dominantACFPeriodBP"] is not None:
            dominantPeriods.append(float(result["dominantACFPeriodBP"]))
        if result["oscillationStrength"] is not None:
            oscillationStrengths.append(float(result["oscillationStrength"]))
        if result["postCrossingRevival"] is not None:
            revivalValues.append(float(result["postCrossingRevival"]))
        chromosomeWindows = selectedByChromosome.get(str(chromosome), [])
        chromosomeWindows.append(len(selectedWindows) - 1)
        selectedByChromosome[str(chromosome)] = chromosomeWindows
        if len(selectedWindows) >= windowCountValue:
            break

    chromosomesUsed = _dependenceSortedAutosomeNames(selectedByChromosome.keys())
    selectedAutosomeCount = len(chromosomesUsed)
    rightCensoredWindowCount = int(np.count_nonzero(radiusCensored))
    crossedWindowCount = len(selectedWindows) - rightCensoredWindowCount
    supportCensoredWindowCount = 0
    capCensoredWindowCount = 0
    for window in selectedWindows:
        if str(window["censorReason"]) == "support":
            supportCensoredWindowCount += 1
        elif str(window["censorReason"]) == "maxLag":
            capCensoredWindowCount += 1
    for chromosome in chromosomesUsed:
        selectedCountsByChromosome[str(chromosome)] = int(
            len(selectedByChromosome[str(chromosome)])
        )
    censorFraction = (
        <double>rightCensoredWindowCount / <double>len(selectedWindows)
        if len(selectedWindows) > 0
        else 0.0
    )
    if len(selectedWindows) < minWindowCountValue:
        raise RuntimeError(
            "dependence estimator has %d valid windows from %d autosomes, "
            "needs at least %d windows, censor fraction %.6f"
            % (
                len(selectedWindows),
                selectedAutosomeCount,
                minWindowCountValue,
                censorFraction,
            )
        )

    fullMedian = _dependenceKMQuantile(radiusValues, radiusCensored, 0.5)
    fullWorkingSpan = _dependenceKMQuantile(
        radiusValues,
        radiusCensored,
        workingQuantileValue,
    )
    if fullMedian is None or fullWorkingSpan is None:
        raise RuntimeError(
            "dependence estimator Kaplan-Meier quantiles are unresolved for %d valid "
            "windows from %d autosomes; censor fraction %.6f"
            % (len(selectedWindows), selectedAutosomeCount, censorFraction)
        )

    (
        bootstrapBlockLengthWindows,
        selectedAdjacencyCount,
        selectedLongestRun,
    ) = _dependenceBootstrapGeometry(
        selectedWindows,
        selectedByChromosome,
        radiusValues,
        radiusCensored,
    )

    fullGrid = np.unique(np.asarray(radiusValues, dtype=np.float64))
    fullSurvival = _dependenceKMSurvivalAt(
        radiusValues,
        radiusCensored,
        fullGrid,
    )
    epsilonValue = 1.0 / (2.0 * <double>len(selectedWindows))
    fullTransformed = np.log(
        -np.log(np.clip(fullSurvival, epsilonValue, 1.0 - epsilonValue))
    )
    bandDomainMask = (fullSurvival >= 0.25) & (fullSurvival <= 0.75)
    if int(np.count_nonzero(bandDomainMask)) <= 0:
        bandDomainMask[int(np.argmin(np.abs(fullSurvival - 0.5)))] = True
    restartProbability = 1.0 / <double>bootstrapBlockLengthWindows
    for drawIndex in range(bootstrapDrawsValue):
        drawValues = []
        drawCensored = []
        sampledChromosomeIndices = bootstrapRNG.integers(
            0,
            selectedAutosomeCount,
            size=selectedAutosomeCount,
        )
        for sampledChromosomeIndex in sampledChromosomeIndices:
            chromosome = chromosomesUsed[int(sampledChromosomeIndex)]
            chromosomePositions = []
            for windowIndex in selectedByChromosome[str(chromosome)]:
                chromosomePositions.append(
                    (
                        int(selectedWindows[int(windowIndex)]["startBP"]),
                        int(windowIndex),
                    )
                )
            chromosomePositions.sort()
            chromosomeWindows = []
            for record in chromosomePositions:
                chromosomeWindows.append(int(record[1]))
            currentPosition = int(
                bootstrapRNG.integers(0, len(chromosomeWindows))
            )
            for positionIndex in range(len(chromosomeWindows)):
                windowIndex = int(chromosomeWindows[currentPosition])
                drawValues.append(float(radiusValues[windowIndex]))
                drawCensored.append(bool(radiusCensored[windowIndex]))
                nextPosition = currentPosition + 1
                if (
                    float(bootstrapRNG.random()) < restartProbability
                    or nextPosition >= len(chromosomeWindows)
                    or int(
                        selectedWindows[int(chromosomeWindows[nextPosition])]["startBP"]
                    )
                    != int(selectedWindows[windowIndex]["endBP"])
                ):
                    currentPosition = int(
                        bootstrapRNG.integers(0, len(chromosomeWindows))
                    )
                else:
                    currentPosition = nextPosition
        drawMedian = _dependenceKMQuantile(drawValues, drawCensored, 0.5)
        drawWorkingSpan = _dependenceKMQuantile(
            drawValues,
            drawCensored,
            workingQuantileValue,
        )
        if drawMedian is not None:
            resolvedMedianDraws += 1
            bootstrapMedianRadiusBP.append(float(drawMedian))
        if drawWorkingSpan is not None:
            resolvedWorkingDraws += 1
            bootstrapWorkingSpanBP.append(float(drawWorkingSpan))
        if drawMedian is not None and drawWorkingSpan is not None:
            resolvedJointDraws += 1
        drawSurvival = _dependenceKMSurvivalAt(
            drawValues,
            drawCensored,
            fullGrid,
        )
        drawTransformed = np.log(
            -np.log(np.clip(drawSurvival, epsilonValue, 1.0 - epsilonValue))
        )
        bootstrapDistances.append(
            float(
                np.max(
                    np.abs(drawTransformed - fullTransformed)[bandDomainMask]
                )
            )
        )

    requiredResolvedDraws = int(ceil(0.95 * <double>bootstrapDrawsValue))
    if (
        resolvedJointDraws < requiredResolvedDraws
        or resolvedMedianDraws < requiredResolvedDraws
        or resolvedWorkingDraws < requiredResolvedDraws
    ):
        raise RuntimeError(
            "dependence estimator resolved %d of %d joint bootstrap draws; needs %d"
            % (resolvedJointDraws, bootstrapDrawsValue, requiredResolvedDraws)
        )

    estimateBP = float(fullMedian)
    workingSpanBP = float(fullWorkingSpan)
    bandCriticalValue = float(
        np.quantile(np.asarray(bootstrapDistances, dtype=np.float64), 0.95)
    )
    lowerSurvival = np.exp(-np.exp(fullTransformed + bandCriticalValue))
    upperSurvival = np.exp(-np.exp(fullTransformed - bandCriticalValue))
    confidenceMask = (lowerSurvival <= 0.5) & (upperSurvival >= 0.5)
    if int(np.count_nonzero(confidenceMask)) <= 0:
        leftBandTransform = log(-log(1.0 - epsilonValue))
        leftLowerSurvival = exp(
            -exp(leftBandTransform + bandCriticalValue)
        )
        leftUpperSurvival = exp(
            -exp(leftBandTransform - bandCriticalValue)
        )
        for bandIndex in range(int(fullGrid.size)):
            if (
                leftLowerSurvival > 0.5
                and leftUpperSurvival > 0.5
                and float(lowerSurvival[bandIndex]) < 0.5
                and float(upperSurvival[bandIndex]) < 0.5
            ):
                jumpClosureIndices.append(int(bandIndex))
            leftLowerSurvival = float(lowerSurvival[bandIndex])
            leftUpperSurvival = float(upperSurvival[bandIndex])
        if len(jumpClosureIndices) != 1:
            raise RuntimeError(
                "dependence estimator could not invert its simultaneous survival band"
            )
        bandIndex = int(jumpClosureIndices[0])
        lowerBP = float(fullGrid[bandIndex])
        upperBP = float(fullGrid[bandIndex])
        survivalBandJumpClosureUsed = True
    else:
        lowerBP = float(np.min(fullGrid[confidenceMask]))
        upperBP = float(np.max(fullGrid[confidenceMask]))
    if (not isfinite(lowerBP)) or (not isfinite(upperBP)):
        raise RuntimeError(
            "dependence estimator produced nonfinite survival-band endpoints"
        )
    lowerBP = min(lowerBP, estimateBP)
    upperBP = max(upperBP, estimateBP)
    if len(dominantPeriods) > 0:
        dominantACFPeriodBPMedian = float(
            np.median(np.asarray(dominantPeriods, dtype=np.float64))
        )
    if len(oscillationStrengths) > 0:
        oscillationStrength = float(
            np.median(np.asarray(oscillationStrengths, dtype=np.float64))
        )
    if len(revivalValues) > 0:
        postCrossingRevival = float(
            np.median(np.asarray(revivalValues, dtype=np.float64))
        )

    diagnostics = {
        "status": "estimated",
        "method": "rankWeightedFinitePairWindowACF",
        "randomSeed": int(randSeedValue),
        "estimateBP": float(estimateBP),
        "lowerBP": float(lowerBP),
        "upperBP": float(upperBP),
        "fullSampleMedianRadiusBP": float(fullMedian),
        "fullSampleWorkingSpanBP": float(fullWorkingSpan),
        "workingSpanBP": float(workingSpanBP),
        "bootstrapMedianRadiusBP": [
            float(value) for value in bootstrapMedianRadiusBP
        ],
        "bootstrapWorkingSpanBP": [
            float(value) for value in bootstrapWorkingSpanBP
        ],
        "workingQuantile": float(workingQuantileValue),
        "inferenceScope": "conditionalOnInputTracksAndSelectedWindows",
        "confidenceIntervalMethod": (
            "centralInterquartileSimultaneousLogLogKMSurvivalBand"
        ),
        "survivalBandRegionLower": 0.25,
        "survivalBandRegionUpper": 0.75,
        "survivalBandJumpClosureUsed": bool(survivalBandJumpClosureUsed),
        "survivalBandJumpClosureCount": int(
            1 if survivalBandJumpClosureUsed else 0
        ),
        "confidenceLevel": 0.95,
        "intervalSizeBP": int(intervalSizeBPValue),
        "windowBP": int(windowBPValue),
        "windowCountRequested": int(windowCountValue),
        "candidateWindowCount": int(len(candidateWindows)),
        "evaluatedCandidateWindowCount": int(evaluatedCandidateWindowCount),
        "selectedWindowCount": int(len(selectedWindows)),
        "minWindowCount": int(minWindowCountValue),
        "selectedAutosomeCount": int(selectedAutosomeCount),
        "chromosomesUsed": [str(value) for value in chromosomesUsed],
        "chromosomesExcluded": sorted(set(excludedNames)),
        "selectedWindows": selectedWindows,
        "inputRowCount": int(rowCount),
        "uniqueRowCount": int(len(retainedRows)),
        "duplicateRowCount": int(rowCount - len(retainedRows)),
        "rowDeduplication": "exactBytes",
        "acfThreshold": float(acfThresholdValue),
        "acfSmoothingBP": int(acfSmoothingBPValue),
        "acfSmoothingBins": int(acfSmoothingBins),
        "crossingPersistenceBP": int(crossingPersistenceBPValue),
        "crossingPersistenceBins": int(crossingPersistenceBins),
        "minFinitePairs": int(minFinitePairsValue),
        "minFinitePairCoverage": float(minFinitePairCoverageValue),
        "maxLagBP": int(maxLagBins * intervalSizeBPValue),
        "gaussianRadiusCorrection": float(gaussianRadiusCorrection),
        "censorFraction": float(censorFraction),
        "crossedWindowCount": int(crossedWindowCount),
        "rightCensoredWindowCount": int(rightCensoredWindowCount),
        "supportCensoredWindowCount": int(supportCensoredWindowCount),
        "capCensoredWindowCount": int(capCensoredWindowCount),
        "censorTimeBPMinimum": (
            None if len(censorTimesBP) <= 0 else float(np.min(censorTimesBP))
        ),
        "censorTimeBPMedian": (
            None if len(censorTimesBP) <= 0 else float(np.median(censorTimesBP))
        ),
        "censorTimeBPMaximum": (
            None if len(censorTimesBP) <= 0 else float(np.max(censorTimesBP))
        ),
        "finitePairMinimumUsed": float(
            np.min(np.asarray(finitePairMinimaUsed, dtype=np.float64))
        ),
        "finitePairCoverageMinimumUsed": float(
            np.min(np.asarray(finitePairCoverageMinimaUsed, dtype=np.float64))
        ),
        "validRowsAtCrossingMinimum": int(
            np.min(np.asarray(validRowsAtCrossingValues, dtype=np.int64))
        ),
        "selectedCountsByChromosome": selectedCountsByChromosome,
        "selectedAdjacencyCount": int(selectedAdjacencyCount),
        "selectedLongestRun": int(selectedLongestRun),
        "radiusDistributionBP": [float(value) for value in radiusValues],
        "radiusCensored": [bool(value) for value in radiusCensored],
        "dominantACFPeriodBPMedian": dominantACFPeriodBPMedian,
        "oscillationStrength": oscillationStrength,
        "postCrossingRevival": postCrossingRevival,
        "unresolvedPeriodFraction": float(
            1.0 - (<double>len(dominantPeriods) / <double>len(selectedWindows))
        ),
        "periodicitySearchMinBP": int(max(2 * intervalSizeBPValue, 150)),
        "periodicitySearchMaxBP": 500,
        "bootstrapMethod": "hierarchicalAutosomeStationaryWindow",
        "bootstrapBlockLengthWindows": int(bootstrapBlockLengthWindows),
        "bootstrapRestartProbability": float(restartProbability),
        "bootstrapDrawsRequested": int(bootstrapDrawsValue),
        "bootstrapResolvedMedianDraws": int(resolvedMedianDraws),
        "bootstrapResolvedWorkingDraws": int(resolvedWorkingDraws),
        "bootstrapResolvedJointDraws": int(resolvedJointDraws),
    }
    return (
        int(ceil(estimateBP / <double>intervalSizeBPValue)),
        int(ceil(lowerBP / <double>intervalSizeBPValue)),
        int(ceil(upperBP / <double>intervalSizeBPValue)),
        diagnostics,
    )
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


cdef inline bint _muncSeedMaskAllowsCell(
    const uint8_t* maskPtr,
    int maskMode,
    Py_ssize_t intervalCount,
    Py_ssize_t j,
    Py_ssize_t k,
    bint nonzeroMeansActive,
) noexcept nogil:
    cdef uint8_t value

    if maskMode == 0:
        return True
    if maskMode == 1:
        value = maskPtr[k]
    else:
        value = maskPtr[j * intervalCount + k]
    if nonzeroMeansActive:
        return value != 0
    return value == 0


cdef Py_ssize_t _muncObservationMomentSeedInvalidIndex(
    const cnp.float32_t* dataPtr,
    const cnp.float32_t* muncPtr,
    const cnp.float32_t* stateMeanPtr,
    const cnp.float32_t* stateVariancePtr,
    const cnp.float32_t* backgroundPtr,
    const cnp.float32_t* gVariancePtr,
    const cnp.float32_t* countFloorPtr,
    const cnp.float32_t* omegaInPtr,
    const cnp.float32_t* rhoInPtr,
    const uint8_t* activePtr,
    Py_ssize_t trackCount,
    Py_ssize_t intervalCount,
    int omegaInMode,
    int activeMode,
    double pad,
    bint useBackground,
    bint useGVariance,
    bint useCountFloor,
    bint useWeights,
    bint studentT,
    bint updateWeights,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t idx
    cdef double value

    for j in range(trackCount):
        for k in range(intervalCount):
            if not _muncSeedMaskAllowsCell(
                activePtr,
                activeMode,
                intervalCount,
                j,
                k,
                True,
            ):
                continue
            idx = j * intervalCount + k
            value = <double>stateMeanPtr[k]
            if not isfinite(value):
                return idx
            value = <double>stateVariancePtr[k]
            if not isfinite(value):
                return idx
            if useBackground:
                value = <double>backgroundPtr[k]
                if not isfinite(value):
                    return idx
            if useGVariance:
                value = <double>gVariancePtr[k]
                if not isfinite(value):
                    return idx
            value = <double>dataPtr[idx]
            if not isfinite(value):
                return idx
            value = (<double>muncPtr[idx]) + pad
            if (not isfinite(value)) or value <= 0.0:
                return idx
            if useCountFloor:
                value = <double>countFloorPtr[idx]
                if (not isfinite(value)) or value < 0.0:
                    return idx
            if useWeights and studentT:
                if omegaInMode == 1:
                    value = <double>omegaInPtr[k]
                    if (not isfinite(value)) or value <= 0.0:
                        return idx
                if not updateWeights:
                    value = <double>rhoInPtr[idx]
                    if (not isfinite(value)) or value <= 0.0:
                        return idx
    return -1


cdef inline void _muncObservationMomentSeedPassInterval(
    const cnp.float32_t* dataPtr,
    const cnp.float32_t* muncPtr,
    const cnp.float32_t* stateMeanPtr,
    const cnp.float32_t* stateVariancePtr,
    const cnp.float32_t* backgroundPtr,
    const cnp.float32_t* gVariancePtr,
    const cnp.float32_t* countFloorPtr,
    const cnp.float32_t* omegaInPtr,
    const cnp.float32_t* rhoInPtr,
    const uint8_t* activePtr,
    cnp.float32_t* momentPtr,
    cnp.float32_t* rhoOutPtr,
    cnp.float32_t* omegaRawPtr,
    cnp.float32_t* omegaOutPtr,
    cnp.float32_t* localPtr,
    cnp.float32_t* variancePtr,
    Py_ssize_t trackCount,
    Py_ssize_t intervalCount,
    Py_ssize_t intervalIndex,
    int omegaInMode,
    int activeMode,
    double pad,
    double dS,
    double dOmega,
    double omegaMin,
    double omegaMax,
    double varianceFloor,
    double varianceCap,
    bint useWeights,
    bint studentT,
    bint updateWeights,
    bint useBackground,
    bint useGVariance,
    bint useCountFloor,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t idx
    cdef Py_ssize_t activeCount = 0
    cdef double stateValue = <double>stateMeanPtr[intervalIndex]
    cdef double momentVarianceBase = <double>stateVariancePtr[intervalIndex]
    cdef double momentVariance
    cdef double backgroundValue = 0.0
    cdef double countValue = 0.0
    cdef double baseVariance
    cdef double residual
    cdef double momentValue
    cdef double rhoValue = 1.0
    cdef double omegaInValue = 1.0
    cdef double omegaRawValue
    cdef double omegaValue = 1.0
    cdef double dbar
    cdef double localValue
    cdef double totalValue

    if useBackground:
        backgroundValue = <double>backgroundPtr[intervalIndex]
    if useGVariance:
        momentVarianceBase += <double>gVariancePtr[intervalIndex]
    if momentVarianceBase < 0.0:
        momentVarianceBase = 0.0

    if useWeights and studentT:
        if omegaInMode == 1:
            omegaInValue = <double>omegaInPtr[intervalIndex]
        else:
            omegaInValue = 1.0
        if updateWeights:
            dbar = 0.0
            for j in range(trackCount):
                idx = j * intervalCount + intervalIndex
                if not _muncSeedMaskAllowsCell(
                    activePtr,
                    activeMode,
                    intervalCount,
                    j,
                    intervalIndex,
                    True,
                ):
                    momentPtr[idx] = <cnp.float32_t>0.0
                    rhoOutPtr[idx] = <cnp.float32_t>1.0
                    continue
                baseVariance = (<double>muncPtr[idx]) + pad
                if baseVariance < varianceFloor:
                    baseVariance = varianceFloor
                residual = (<double>dataPtr[idx]) - backgroundValue - stateValue
                momentValue = residual * residual + momentVarianceBase
                rhoValue = (dS + 1.0) / (
                    dS + omegaInValue * momentValue / baseVariance
                )
                momentPtr[idx] = <cnp.float32_t>momentValue
                rhoOutPtr[idx] = <cnp.float32_t>rhoValue
                dbar += momentValue / baseVariance
                activeCount += 1
            if activeCount > 0:
                dbar = dbar / (<double>activeCount)
                omegaRawValue = (dOmega + 1.0) / (dOmega + dbar)
                omegaValue = _clampMultiplierValue(
                    omegaRawValue,
                    omegaMin,
                    omegaMax,
                )
            else:
                omegaRawValue = 1.0
                omegaValue = 1.0
        else:
            omegaRawValue = omegaInValue
            omegaValue = _clampMultiplierValue(
                omegaRawValue,
                omegaMin,
                omegaMax,
            )
            for j in range(trackCount):
                idx = j * intervalCount + intervalIndex
                if not _muncSeedMaskAllowsCell(
                    activePtr,
                    activeMode,
                    intervalCount,
                    j,
                    intervalIndex,
                    True,
                ):
                    momentPtr[idx] = <cnp.float32_t>0.0
                    rhoOutPtr[idx] = <cnp.float32_t>1.0
                    continue
                residual = (<double>dataPtr[idx]) - backgroundValue - stateValue
                momentValue = residual * residual + momentVarianceBase
                rhoValue = <double>rhoInPtr[idx]
                momentPtr[idx] = <cnp.float32_t>momentValue
                rhoOutPtr[idx] = <cnp.float32_t>rhoValue
        omegaRawPtr[intervalIndex] = <cnp.float32_t>omegaRawValue
        omegaOutPtr[intervalIndex] = <cnp.float32_t>omegaValue
    else:
        omegaRawPtr[intervalIndex] = <cnp.float32_t>1.0
        omegaOutPtr[intervalIndex] = <cnp.float32_t>1.0

    for j in range(trackCount):
        idx = j * intervalCount + intervalIndex
        if _muncSeedMaskAllowsCell(
            activePtr,
            activeMode,
            intervalCount,
            j,
            intervalIndex,
            True,
        ):
            if not (useWeights and studentT):
                residual = (<double>dataPtr[idx]) - backgroundValue - stateValue
                momentValue = residual * residual + momentVarianceBase
                momentPtr[idx] = <cnp.float32_t>momentValue
                rhoOutPtr[idx] = <cnp.float32_t>1.0
            else:
                momentValue = <double>momentPtr[idx]
                rhoValue = <double>rhoOutPtr[idx]
            if useCountFloor:
                countValue = <double>countFloorPtr[idx]
            else:
                countValue = 0.0
            if useWeights and studentT:
                localValue = (
                    omegaValue * rhoValue * momentValue
                    - pad
                    - countValue
                )
            else:
                localValue = momentValue - pad - countValue
            totalValue = localValue + countValue
            if localValue < varianceFloor:
                localValue = varianceFloor
                totalValue = localValue + countValue
            if totalValue > varianceCap:
                totalValue = varianceCap
                localValue = totalValue - countValue
                if localValue < varianceFloor:
                    localValue = varianceFloor
                    totalValue = localValue + countValue
            localPtr[idx] = <cnp.float32_t>localValue
            variancePtr[idx] = <cnp.float32_t>totalValue
        else:
            if useCountFloor:
                countValue = <double>countFloorPtr[idx]
            else:
                countValue = 0.0
            localValue = (<double>muncPtr[idx]) - countValue
            if localValue < varianceFloor:
                localValue = varianceFloor
            totalValue = localValue + countValue
            if totalValue > varianceCap:
                totalValue = varianceCap
                localValue = totalValue - countValue
                if localValue < varianceFloor:
                    localValue = varianceFloor
                    totalValue = localValue + countValue
            momentPtr[idx] = <cnp.float32_t>0.0
            rhoOutPtr[idx] = <cnp.float32_t>1.0
            localPtr[idx] = <cnp.float32_t>localValue
            variancePtr[idx] = <cnp.float32_t>totalValue


cpdef tuple cMuncObservationMomentSeedPass(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixMunc,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] stateMean,
    cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] stateVariance,
    object background=None,
    object gVariance=None,
    object countFloor=None,
    object omegaIn=None,
    object rhoIn=None,
    float pad=1.0e-4,
    float studentTdf=8.0,
    bint useSeedWeights=True,
    bint updateWeights=True,
    float omegaMin=0.01,
    float omegaMax=100.0,
    float varianceFloor=1.0e-12,
    float varianceCap=3.4028234663852886e38,
    bint enabled=True,
    bint studentT=True,
    float dOmega=8.0,
    object activeMask=None,
):
    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef Py_ssize_t intervalIndex
    cdef Py_ssize_t invalidIndex
    cdef bint useBackground = background is not None
    cdef bint useGVariance = gVariance is not None
    cdef bint useCountFloor = countFloor is not None
    cdef bint useWeights = enabled and useSeedWeights
    cdef double padValue = <double>pad
    cdef double studentTdfValue = <double>studentTdf
    cdef double dOmegaValue = <double>dOmega
    cdef double omegaMinValue = <double>omegaMin
    cdef double omegaMaxValue = <double>omegaMax
    cdef double varianceFloorValue = <double>varianceFloor
    cdef double varianceCapValue = <double>varianceCap
    cdef object omegaObj
    cdef object activeObj
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] backgroundArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] gVarianceArr
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] countFloorArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] omegaInArr1d
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] rhoInArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] activeArr1d
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] activeArr2d
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] momentArr
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] rhoOutArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] omegaRawArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] omegaOutArr
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] localArr
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] varianceArr
    cdef cnp.float32_t* backgroundPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* gVariancePtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* countFloorPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* omegaInPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* rhoInPtr = <cnp.float32_t*>NULL
    cdef uint8_t* activePtr = <uint8_t*>NULL
    cdef cnp.float32_t* momentPtr
    cdef cnp.float32_t* rhoOutPtr
    cdef cnp.float32_t* omegaRawPtr
    cdef cnp.float32_t* omegaOutPtr
    cdef cnp.float32_t* localPtr
    cdef cnp.float32_t* variancePtr
    cdef int omegaInMode = 0
    cdef int activeMode = 0

    if matrixMunc.shape[0] != trackCount or matrixMunc.shape[1] != intervalCount:
        raise ValueError("matrixMunc shape must match matrixData shape")
    if stateMean.shape[0] != intervalCount:
        raise ValueError("stateMean length must match interval count")
    if stateVariance.shape[0] != intervalCount:
        raise ValueError("stateVariance length must match interval count")
    if padValue < 0.0 or not isfinite(padValue):
        raise ValueError("pad must be finite and nonnegative")
    if varianceFloorValue <= 0.0 or not isfinite(varianceFloorValue):
        raise ValueError("varianceFloor must be positive and finite")
    if (not isfinite(varianceCapValue)) or varianceCapValue < varianceFloorValue:
        raise ValueError("varianceCap must be greater than or equal to varianceFloor")
    if useWeights and studentT and (
        studentTdfValue <= 0.0
        or dOmegaValue <= 0.0
        or not isfinite(studentTdfValue)
        or not isfinite(dOmegaValue)
        or omegaMinValue <= 0.0
        or omegaMaxValue < omegaMinValue
        or not isfinite(omegaMinValue)
        or not isfinite(omegaMaxValue)
    ):
        raise ValueError("seed weight parameters are invalid")

    if useBackground:
        backgroundArr = np.ascontiguousarray(background, dtype=np.float32).reshape(-1)
        if backgroundArr.shape[0] != intervalCount:
            raise ValueError("background length must match interval count")
        backgroundPtr = <cnp.float32_t*>backgroundArr.data
    if useGVariance:
        gVarianceArr = np.ascontiguousarray(gVariance, dtype=np.float32).reshape(-1)
        if gVarianceArr.shape[0] != intervalCount:
            raise ValueError("gVariance length must match interval count")
        gVariancePtr = <cnp.float32_t*>gVarianceArr.data
    if useCountFloor:
        countFloorArr = np.ascontiguousarray(countFloor, dtype=np.float32)
        if (
            countFloorArr.shape[0] != trackCount
            or countFloorArr.shape[1] != intervalCount
        ):
            raise ValueError("countFloor shape must match matrixData shape")
        countFloorPtr = <cnp.float32_t*>countFloorArr.data
    if omegaIn is not None:
        omegaObj = np.ascontiguousarray(omegaIn, dtype=np.float32)
        if omegaObj.ndim == 1:
            omegaInArr1d = np.ascontiguousarray(omegaObj.reshape(-1), dtype=np.float32)
            if omegaInArr1d.shape[0] != intervalCount:
                raise ValueError("omegaIn length must match interval count")
            omegaInPtr = <cnp.float32_t*>omegaInArr1d.data
            omegaInMode = 1
        else:
            raise ValueError("omegaIn must be one-dimensional")
    if rhoIn is None and useWeights and studentT and not updateWeights:
        rhoInArr = np.ones((trackCount, intervalCount), dtype=np.float32)
        rhoInPtr = <cnp.float32_t*>rhoInArr.data
    elif rhoIn is not None:
        rhoInArr = np.ascontiguousarray(rhoIn, dtype=np.float32)
        if (
            rhoInArr.shape[0] != trackCount
            or rhoInArr.shape[1] != intervalCount
        ):
            raise ValueError("rhoIn shape must match matrixData shape")
        rhoInPtr = <cnp.float32_t*>rhoInArr.data
    if activeMask is not None:
        activeObj = np.ascontiguousarray(activeMask, dtype=np.uint8)
        if activeObj.ndim == 1:
            activeArr1d = np.ascontiguousarray(activeObj.reshape(-1), dtype=np.uint8)
            if activeArr1d.shape[0] != intervalCount:
                raise ValueError("activeMask length must match interval count")
            activePtr = <uint8_t*>activeArr1d.data
            activeMode = 1
        elif activeObj.ndim == 2:
            activeArr2d = <cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"]>activeObj
            if (
                activeArr2d.shape[0] != trackCount
                or activeArr2d.shape[1] != intervalCount
            ):
                raise ValueError("activeMask shape must match matrixData shape")
            activePtr = <uint8_t*>activeArr2d.data
            activeMode = 2
        else:
            raise ValueError("activeMask must be one- or two-dimensional")

    invalidIndex = _muncObservationMomentSeedInvalidIndex(
        <cnp.float32_t*>matrixData.data,
        <cnp.float32_t*>matrixMunc.data,
        <cnp.float32_t*>stateMean.data,
        <cnp.float32_t*>stateVariance.data,
        backgroundPtr,
        gVariancePtr,
        countFloorPtr,
        omegaInPtr,
        rhoInPtr,
        activePtr,
        trackCount,
        intervalCount,
        omegaInMode,
        activeMode,
        padValue,
        useBackground,
        useGVariance,
        useCountFloor,
        useWeights,
        studentT,
        updateWeights,
    )
    if invalidIndex >= 0:
        raise ValueError("active MUNC seed cells must be finite with positive denominators")

    momentArr = np.empty((trackCount, intervalCount), dtype=np.float32)
    rhoOutArr = np.empty((trackCount, intervalCount), dtype=np.float32)
    omegaRawArr = np.empty(intervalCount, dtype=np.float32)
    omegaOutArr = np.empty(intervalCount, dtype=np.float32)
    localArr = np.empty((trackCount, intervalCount), dtype=np.float32)
    varianceArr = np.empty((trackCount, intervalCount), dtype=np.float32)

    momentPtr = <cnp.float32_t*>momentArr.data
    rhoOutPtr = <cnp.float32_t*>rhoOutArr.data
    omegaRawPtr = <cnp.float32_t*>omegaRawArr.data
    omegaOutPtr = <cnp.float32_t*>omegaOutArr.data
    localPtr = <cnp.float32_t*>localArr.data
    variancePtr = <cnp.float32_t*>varianceArr.data

    if USE_OPENMP:
        if intervalCount >= OPENMP_APPLY_MIN_ROWS:
            for intervalIndex in prange(intervalCount, nogil=True, schedule="static"):
                _muncObservationMomentSeedPassInterval(
                    <cnp.float32_t*>matrixData.data,
                    <cnp.float32_t*>matrixMunc.data,
                    <cnp.float32_t*>stateMean.data,
                    <cnp.float32_t*>stateVariance.data,
                    backgroundPtr,
                    gVariancePtr,
                    countFloorPtr,
                    omegaInPtr,
                    rhoInPtr,
                    activePtr,
                    momentPtr,
                    rhoOutPtr,
                    omegaRawPtr,
                    omegaOutPtr,
                    localPtr,
                    variancePtr,
                    trackCount,
                    intervalCount,
                    intervalIndex,
                    omegaInMode,
                    activeMode,
                    padValue,
                    studentTdfValue,
                    dOmegaValue,
                    omegaMinValue,
                    omegaMaxValue,
                    varianceFloorValue,
                    varianceCapValue,
                    useWeights,
                    studentT,
                    updateWeights,
                    useBackground,
                    useGVariance,
                    useCountFloor,
                )
        else:
            with nogil:
                for intervalIndex in range(intervalCount):
                    _muncObservationMomentSeedPassInterval(
                        <cnp.float32_t*>matrixData.data,
                        <cnp.float32_t*>matrixMunc.data,
                        <cnp.float32_t*>stateMean.data,
                        <cnp.float32_t*>stateVariance.data,
                        backgroundPtr,
                        gVariancePtr,
                        countFloorPtr,
                        omegaInPtr,
                        rhoInPtr,
                        activePtr,
                        momentPtr,
                        rhoOutPtr,
                        omegaRawPtr,
                        omegaOutPtr,
                        localPtr,
                        variancePtr,
                        trackCount,
                        intervalCount,
                        intervalIndex,
                        omegaInMode,
                        activeMode,
                        padValue,
                        studentTdfValue,
                        dOmegaValue,
                        omegaMinValue,
                        omegaMaxValue,
                        varianceFloorValue,
                        varianceCapValue,
                        useWeights,
                        studentT,
                        updateWeights,
                        useBackground,
                        useGVariance,
                        useCountFloor,
                    )
    else:
        with nogil:
            for intervalIndex in range(intervalCount):
                _muncObservationMomentSeedPassInterval(
                    <cnp.float32_t*>matrixData.data,
                    <cnp.float32_t*>matrixMunc.data,
                    <cnp.float32_t*>stateMean.data,
                    <cnp.float32_t*>stateVariance.data,
                    backgroundPtr,
                    gVariancePtr,
                    countFloorPtr,
                    omegaInPtr,
                    rhoInPtr,
                    activePtr,
                    momentPtr,
                    rhoOutPtr,
                    omegaRawPtr,
                    omegaOutPtr,
                    localPtr,
                    variancePtr,
                    trackCount,
                    intervalCount,
                    intervalIndex,
                    omegaInMode,
                    activeMode,
                    padValue,
                    studentTdfValue,
                    dOmegaValue,
                    omegaMinValue,
                    omegaMaxValue,
                    varianceFloorValue,
                    varianceCapValue,
                    useWeights,
                    studentT,
                    updateWeights,
                    useBackground,
                    useGVariance,
                    useCountFloor,
                )

    return momentArr, rhoOutArr, omegaRawArr, omegaOutArr, localArr, varianceArr


ctypedef struct MuncEBFinalizeResult:
    Py_ssize_t invalidLocalIndex
    Py_ssize_t invalidPriorIndex
    Py_ssize_t invalidCountFloorIndex
    Py_ssize_t supportCount
    Py_ssize_t countFloorFiniteCount
    Py_ssize_t countFloorAddedCount
    Py_ssize_t countFloorMissingCount


cdef MuncEBFinalizeResult _finalizeMuncEBTrackLoop(
    const cnp.float32_t* localPtr,
    const cnp.float32_t* priorPtr,
    const cnp.float32_t* countFloorPtr,
    cnp.float32_t* outPtr,
    Py_ssize_t intervalCount,
    double nuLocal,
    double nuPrior,
    double posteriorSampleSize,
    double varianceFloor,
    double varianceCap,
    bint useEB,
    bint useCountFloor,
) noexcept nogil:
    cdef MuncEBFinalizeResult result
    cdef Py_ssize_t i
    cdef double localValue
    cdef double priorValue
    cdef double outValue
    cdef double countFloorValue

    result.invalidLocalIndex = -1
    result.invalidPriorIndex = -1
    result.invalidCountFloorIndex = -1
    result.supportCount = 0
    result.countFloorFiniteCount = 0
    result.countFloorAddedCount = 0
    result.countFloorMissingCount = 0

    for i in range(intervalCount):
        localValue = <double>localPtr[i]
        if (not isfinite(localValue)) or localValue <= 0.0:
            result.invalidLocalIndex = i
            return result
        if localValue > varianceFloor:
            result.supportCount += 1
        if localValue < varianceFloor:
            localValue = varianceFloor
        elif localValue > varianceCap:
            localValue = varianceCap

        if useEB:
            priorValue = <double>priorPtr[i]
            if (not isfinite(priorValue)) or priorValue <= 0.0:
                result.invalidPriorIndex = i
                return result
            if priorValue < varianceFloor:
                priorValue = varianceFloor
            elif priorValue > varianceCap:
                priorValue = varianceCap
            outValue = ((nuLocal * localValue) + (nuPrior * priorValue)) / posteriorSampleSize
        else:
            outValue = localValue

        if outValue < varianceFloor:
            outValue = varianceFloor
        elif outValue > varianceCap:
            outValue = varianceCap

        if useCountFloor:
            countFloorValue = <double>countFloorPtr[i]
            if countFloorValue == countFloorValue:
                if (not isfinite(countFloorValue)) or countFloorValue < 0.0:
                    result.invalidCountFloorIndex = i
                    return result
                result.countFloorFiniteCount += 1
                outValue += countFloorValue
                if countFloorValue > 0.0:
                    result.countFloorAddedCount += 1
                if outValue < varianceFloor:
                    outValue = varianceFloor
                elif outValue > varianceCap:
                    outValue = varianceCap
            else:
                result.countFloorMissingCount += 1

        outPtr[i] = <cnp.float32_t>outValue

    return result


cpdef tuple cFinalizeMuncEBTrack(
    object localVarianceTrack,
    object priorVarianceTrack=None,
    object countFloor=None,
    float nuLocal=0.0,
    float nuPrior=0.0,
    float varianceFloor=1.0e-12,
    float varianceCap=3.4028234663852886e38,
    bint useEB=True,
):
    cdef Py_ssize_t intervalCount
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] localArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] priorArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] countFloorArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] outArr
    cdef cnp.float32_t* priorPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* countFloorPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* outPtr
    cdef double nuLocalValue = <double>nuLocal
    cdef double nuPriorValue = <double>nuPrior
    cdef double posteriorSampleSize = nuLocalValue + nuPriorValue
    cdef double varianceFloorValue = <double>varianceFloor
    cdef double varianceCapValue = <double>varianceCap
    cdef bint useCountFloor = countFloor is not None
    cdef MuncEBFinalizeResult result
    cdef double supportFraction

    localArr = np.ascontiguousarray(localVarianceTrack, dtype=np.float32).reshape(-1)
    intervalCount = localArr.shape[0]

    if varianceFloorValue <= 0.0 or not isfinite(varianceFloorValue):
        raise ValueError("varianceFloor must be positive and finite")
    if varianceCapValue < varianceFloorValue or not isfinite(varianceCapValue):
        raise ValueError("varianceCap must be finite and at least varianceFloor")
    if useEB:
        if priorVarianceTrack is None:
            raise ValueError("priorVarianceTrack is required for MUNC EB finalization")
        if (not isfinite(nuLocalValue)) or nuLocalValue <= 0.0:
            raise ValueError("nuLocal must be positive and finite")
        if (not isfinite(nuPriorValue)) or nuPriorValue <= 0.0:
            raise ValueError("nuPrior must be positive and finite")
        if (not isfinite(posteriorSampleSize)) or posteriorSampleSize <= 0.0:
            raise ValueError("posterior sample size must be positive and finite")
        priorArr = np.ascontiguousarray(priorVarianceTrack, dtype=np.float32).reshape(-1)
        if priorArr.shape[0] != intervalCount:
            raise ValueError("priorVarianceTrack length must match localVarianceTrack length")
        priorPtr = <cnp.float32_t*>priorArr.data

    if useCountFloor:
        countFloorArr = np.ascontiguousarray(countFloor, dtype=np.float32).reshape(-1)
        if countFloorArr.shape[0] != intervalCount:
            raise ValueError("countFloor length must match localVarianceTrack length")
        countFloorPtr = <cnp.float32_t*>countFloorArr.data

    outArr = np.empty(intervalCount, dtype=np.float32)
    outPtr = <cnp.float32_t*>outArr.data

    with nogil:
        result = _finalizeMuncEBTrackLoop(
            <cnp.float32_t*>localArr.data,
            priorPtr,
            countFloorPtr,
            outPtr,
            intervalCount,
            nuLocalValue,
            nuPriorValue,
            posteriorSampleSize,
            varianceFloorValue,
            varianceCapValue,
            useEB,
            useCountFloor,
        )

    if result.invalidLocalIndex >= 0:
        raise ValueError(
            f"localVarianceTrack must contain finite positive values at index {result.invalidLocalIndex}"
        )
    if result.invalidPriorIndex >= 0:
        raise ValueError(
            f"priorVarianceTrack must contain finite positive values at index {result.invalidPriorIndex}"
        )
    if result.invalidCountFloorIndex >= 0:
        raise ValueError(
            f"countFloor must be nonnegative where finite at index {result.invalidCountFloorIndex}"
        )

    supportFraction = (
        (<double>result.supportCount) / (<double>intervalCount)
        if intervalCount > 0
        else 0.0
    )
    return outArr, {
        "supportCount": result.supportCount,
        "supportFraction": supportFraction,
        "countFloorFiniteCount": result.countFloorFiniteCount,
        "countFloorAddedCount": result.countFloorAddedCount,
        "countFloorMissingCount": result.countFloorMissingCount,
        "finalShrinkagePairCount": intervalCount if useEB else 0,
        "finalShrinkagePairFraction": 1.0 if useEB and intervalCount > 0 else 0.0,
    }


cdef Py_ssize_t _muncSmoothDenseLocalEvidenceInvalidIndex(
    const cnp.float32_t* localPtr,
    const uint8_t* excludePtr,
    Py_ssize_t trackCount,
    Py_ssize_t intervalCount,
    int excludeMode,
) noexcept nogil:
    cdef Py_ssize_t j
    cdef Py_ssize_t i
    cdef Py_ssize_t idx
    cdef double value

    for j in range(trackCount):
        for i in range(intervalCount):
            if not _muncSeedMaskAllowsCell(
                excludePtr,
                excludeMode,
                intervalCount,
                j,
                i,
                False,
            ):
                continue
            idx = j * intervalCount + i
            value = <double>localPtr[idx]
            if (not isfinite(value)) or value <= 0.0:
                return idx
    return -1


cdef inline void _muncSmoothDenseLocalEvidenceRow(
    const cnp.float32_t* localPtr,
    const uint8_t* excludePtr,
    cnp.float32_t* outPtr,
    Py_ssize_t intervalCount,
    Py_ssize_t rowIndex,
    Py_ssize_t windowIntervals,
    int excludeMode,
    double epsValue,
) noexcept nogil:
    cdef Py_ssize_t rowStart = rowIndex * intervalCount
    cdef Py_ssize_t i
    cdef Py_ssize_t left
    cdef Py_ssize_t right
    cdef Py_ssize_t targetLeft
    cdef Py_ssize_t targetRight
    cdef Py_ssize_t half = windowIntervals // 2
    cdef Py_ssize_t count = 0
    cdef double value
    cdef double rollingSum = 0.0

    left = 0
    right = 0
    for i in range(intervalCount):
        targetLeft = i - half if i >= half else 0
        targetRight = targetLeft + windowIntervals
        if targetRight > intervalCount:
            targetRight = intervalCount
            if targetRight >= windowIntervals:
                targetLeft = targetRight - windowIntervals
            else:
                targetLeft = 0
        while right < targetRight:
            if _muncSeedMaskAllowsCell(
                excludePtr,
                excludeMode,
                intervalCount,
                rowIndex,
                right,
                False,
            ):
                rollingSum += <double>localPtr[rowStart + right]
                count += 1
            right += 1
        while left < targetLeft:
            if _muncSeedMaskAllowsCell(
                excludePtr,
                excludeMode,
                intervalCount,
                rowIndex,
                left,
                False,
            ):
                rollingSum -= <double>localPtr[rowStart + left]
                count -= 1
            left += 1
        if count > 0:
            value = rollingSum / (<double>count)
        else:
            value = <double>localPtr[rowStart + i]
        if value < epsValue:
            value = epsValue
        outPtr[rowStart + i] = <cnp.float32_t>value


cpdef cnp.ndarray cMuncSmoothDenseLocalEvidence(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] localEvidence,
    Py_ssize_t windowIntervals,
    object excludeMask=None,
    float eps=1.0e-12,
):
    cdef Py_ssize_t trackCount = localEvidence.shape[0]
    cdef Py_ssize_t intervalCount = localEvidence.shape[1]
    cdef Py_ssize_t rowIndex
    cdef Py_ssize_t invalidIndex
    cdef Py_ssize_t cellCount = trackCount * intervalCount
    cdef double epsValue = <double>eps
    cdef object excludeObj
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] outArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] excludeArr1d
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] excludeArr2d
    cdef cnp.float32_t* localPtr = <cnp.float32_t*>localEvidence.data
    cdef cnp.float32_t* outPtr
    cdef uint8_t* excludePtr = <uint8_t*>NULL
    cdef int excludeMode = 0
    cdef bint useExclude = excludeMask is not None

    if windowIntervals < 1:
        raise ValueError("windowIntervals must be positive")
    if epsValue <= 0.0 or not isfinite(epsValue):
        raise ValueError("eps must be positive and finite")
    if useExclude:
        excludeObj = np.ascontiguousarray(excludeMask, dtype=np.uint8)
        if excludeObj.ndim == 1:
            excludeArr1d = np.ascontiguousarray(excludeObj.reshape(-1), dtype=np.uint8)
            if excludeArr1d.shape[0] != intervalCount:
                raise ValueError("excludeMask length must match interval count")
            excludePtr = <uint8_t*>excludeArr1d.data
            excludeMode = 1
        elif excludeObj.ndim == 2:
            excludeArr2d = <cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"]>excludeObj
            if (
                excludeArr2d.shape[0] != trackCount
                or excludeArr2d.shape[1] != intervalCount
            ):
                raise ValueError("excludeMask shape must match localEvidence shape")
            excludePtr = <uint8_t*>excludeArr2d.data
            excludeMode = 2
        else:
            raise ValueError("excludeMask must be one- or two-dimensional")

    invalidIndex = _muncSmoothDenseLocalEvidenceInvalidIndex(
        localPtr,
        excludePtr,
        trackCount,
        intervalCount,
        excludeMode,
    )
    if invalidIndex >= 0:
        raise ValueError("active local evidence cells must be positive and finite")

    outArr = np.empty((trackCount, intervalCount), dtype=np.float32)
    outPtr = <cnp.float32_t*>outArr.data

    if USE_OPENMP:
        if cellCount >= OPENMP_APPLY_MIN_ROWS:
            for rowIndex in prange(trackCount, nogil=True, schedule="static"):
                _muncSmoothDenseLocalEvidenceRow(
                    localPtr,
                    excludePtr,
                    outPtr,
                    intervalCount,
                    rowIndex,
                    windowIntervals,
                    excludeMode,
                    epsValue,
                )
        else:
            with nogil:
                for rowIndex in range(trackCount):
                    _muncSmoothDenseLocalEvidenceRow(
                        localPtr,
                        excludePtr,
                        outPtr,
                        intervalCount,
                        rowIndex,
                        windowIntervals,
                        excludeMode,
                        epsValue,
                    )
    else:
        with nogil:
            for rowIndex in range(trackCount):
                _muncSmoothDenseLocalEvidenceRow(
                    localPtr,
                    excludePtr,
                    outPtr,
                    intervalCount,
                    rowIndex,
                    windowIntervals,
                    excludeMode,
                    epsValue,
                )

    return outArr


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

    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef LevelTrendForwardLoopResult loopResult
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] dStatVectorArr
    cdef bint doStore = (stateForward is not None)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] lambdaExpArr
    cdef bint useLambda = (ECM_useObsPrecisionReweighting and (lambdaExp is not None))
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processPrecExpArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processQScaleArr
    cdef bint useProcessQScale = (processQScale is not None)
    cdef bint useProcPrec = (
        ECM_useProcessPrecisionReweighting
        and (processPrecExp is not None)
        and ((not ECM_useAPN) or useProcessQScale)
    )
    cdef cnp.float32_t* dataPtr = <cnp.float32_t*>matrixData.data
    cdef cnp.float32_t* muncPtr = <cnp.float32_t*>matrixPluginMuncInit.data
    cdef cnp.int32_t* blockMapPtr = <cnp.int32_t*>intervalToBlockMap.data
    cdef cnp.float32_t* dStatPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* lambdaExpPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* processPrecExpPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* processQScalePtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* stateForwardPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* stateCovarForwardPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* pNoiseForwardPtr = <cnp.float32_t*>NULL
    cdef float phiHat = 1.0

    cdef double F00, F01, F10, F11
    cdef double qBase00, qBase01, qBase10, qBase11
    cdef double wMin = <double>obsPrecisionMultiplierMin
    cdef double wMax = <double>obsPrecisionMultiplierMax
    cdef double procPrecMin = <double>procPrecisionMultiplierMin
    cdef double procPrecMax = <double>procPrecisionMultiplierMax
    cdef double qDiagBase
    cdef double apnMinQ = <double>APN_minQ
    cdef double apnMaxQ = <double>APN_maxQ
    cdef double apnThresh = <double>APN_dStatThresh
    cdef double apnScaleCoef = <double>APN_dStatScale
    cdef double apnPC = <double>APN_dStatPC

    cdef double LOG2PI = log(6.2831853071795864769)

    if useLambda:
        lambdaExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> lambdaExp

    if useProcPrec:
        processPrecExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> processPrecExp

    if useProcessQScale:
        processQScaleArr = _coerceProcessQScale(processQScale, intervalCount)

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
    if matrixF.shape[0] < 2 or matrixF.shape[1] < 2:
        raise ValueError("matrixF must have at least shape (2, 2)")
    if matrixQ0.shape[0] < 2 or matrixQ0.shape[1] < 2:
        raise ValueError("matrixQ0 must have at least shape (2, 2)")

    _validateMultiplierBounds(wMin, wMax, True)
    _validateMultiplierBounds(procPrecMin, procPrecMax, False)

    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")

    if useLambda:
        if lambdaExpArr.shape[0] != intervalCount:
            raise ValueError("lambdaExp length must match intervalCount")
        lambdaExpPtr = <cnp.float32_t*>lambdaExpArr.data

    if useProcPrec:
        if processPrecExpArr.shape[0] != intervalCount:
            raise ValueError("processPrecExp length must match intervalCount")
        processPrecExpPtr = <cnp.float32_t*>processPrecExpArr.data

    if useProcessQScale:
        processQScalePtr = <cnp.float32_t*>processQScaleArr.data

    if vectorD is None:
        dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
        vectorD = dStatVectorArr
    else:
        dStatVectorArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> vectorD
        if dStatVectorArr.shape[0] < intervalCount:
            raise ValueError("vectorD length must match intervalCount")
    dStatPtr = <cnp.float32_t*>dStatVectorArr.data

    if doStore:
        stateForwardArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateForward
        stateCovarForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarForward
        pNoiseForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> pNoiseForward
        if stateForwardArr.shape[0] < intervalCount or stateForwardArr.shape[1] < 2:
            raise ValueError("stateForward shape must match intervalCount by 2")
        if (
            stateCovarForwardArr.shape[0] < intervalCount
            or stateCovarForwardArr.shape[1] < 2
            or stateCovarForwardArr.shape[2] < 2
        ):
            raise ValueError("stateCovarForward shape must match intervalCount by 2 by 2")
        if (
            intervalCount > 1
            and (
                pNoiseForwardArr.shape[0] < intervalCount - 1
                or pNoiseForwardArr.shape[1] < 2
                or pNoiseForwardArr.shape[2] < 2
            )
        ):
            raise ValueError("pNoiseForward shape must permit intervalCount minus one 2 by 2 entries")
        stateForwardPtr = <cnp.float32_t*>stateForwardArr.data
        stateCovarForwardPtr = <cnp.float32_t*>stateCovarForwardArr.data
        pNoiseForwardPtr = <cnp.float32_t*>pNoiseForwardArr.data

    F00 = <double>matrixF[0, 0]
    F01 = <double>matrixF[0, 1]
    F10 = <double>matrixF[1, 0]
    F11 = <double>matrixF[1, 1]
    qBase00 = <double>matrixQ0[0, 0]
    qBase01 = <double>matrixQ0[0, 1]
    qBase10 = <double>matrixQ0[1, 0]
    qBase11 = <double>matrixQ0[1, 1]
    qDiagBase = 0.5 * (qBase00 + qBase11)
    if qDiagBase <= 1.0e-12:
        ECM_useAPN = False

    with nogil:
        loopResult = _levelTrendForwardPassLoop(
            dataPtr,
            muncPtr,
            blockMapPtr,
            lambdaExpPtr,
            processPrecExpPtr,
            processQScalePtr,
            dStatPtr,
            stateForwardPtr,
            stateCovarForwardPtr,
            pNoiseForwardPtr,
            trackCount,
            intervalCount,
            blockCount,
            <double>stateInit,
            <double>stateCovarInit,
            <double>pad,
            F00,
            F01,
            F10,
            F11,
            qBase00,
            qBase01,
            qBase10,
            qBase11,
            qDiagBase,
            LOG2PI,
            wMin,
            wMax,
            procPrecMin,
            procPrecMax,
            apnMinQ,
            apnMaxQ,
            apnThresh,
            apnScaleCoef,
            apnPC,
            doStore,
            useLambda,
            useProcPrec,
            useProcessQScale,
            ECM_useAPN,
            returnNLL,
            storeNLLInD,
        )

    if loopResult.invalidBlockIndex >= 0:
        raise ValueError("intervalToBlockMap has out-of-range block id")

    phiHat = <float>(loopResult.sumDStat / (<double>intervalCount))

    if returnNLL:
        return (phiHat, 0, vectorD, loopResult.sumNLL)

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

    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef bint doStore = (stateForward is not None)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] dStatVectorArr
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForwardArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] lambdaExpArr
    cdef bint useLambda = (ECM_useObsPrecisionReweighting and (lambdaExp is not None))
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processPrecExpArr
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processQScaleArr
    cdef bint useProcessQScale = (processQScale is not None)
    cdef bint useProcPrec = (
        ECM_useProcessPrecisionReweighting
        and (processPrecExp is not None)
        and ((not ECM_useAPN) or useProcessQScale)
    )
    cdef const cnp.float32_t* dataPtr = NULL
    cdef const cnp.float32_t* muncPtr = NULL
    cdef const cnp.int32_t* blockMapPtr = NULL
    cdef const cnp.float32_t* lambdaExpPtr = NULL
    cdef const cnp.float32_t* processPrecExpPtr = NULL
    cdef const cnp.float32_t* processQScalePtr = NULL
    cdef cnp.float32_t* dStatPtr = NULL
    cdef cnp.float32_t* stateForwardPtr = NULL
    cdef cnp.float32_t* stateCovarForwardPtr = NULL
    cdef cnp.float32_t* pNoiseForwardPtr = NULL
    cdef LevelForwardLoopResult loopResult
    cdef double q0
    cdef double sumNLL = 0.0
    cdef double wMin = <double>obsPrecisionMultiplierMin
    cdef double wMax = <double>obsPrecisionMultiplierMax
    cdef double procPrecMin = <double>procPrecisionMultiplierMin
    cdef double procPrecMax = <double>procPrecisionMultiplierMax
    cdef double phiHat = 1.0
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
    q0 = <double>matrixQ0[0, 0]
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
        lambdaExpPtr = <const cnp.float32_t*>lambdaExpArr.data
    if useProcPrec:
        processPrecExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> processPrecExp
        if processPrecExpArr.shape[0] != intervalCount:
            raise ValueError("processPrecExp length must match intervalCount")
        processPrecExpPtr = <const cnp.float32_t*>processPrecExpArr.data
    if useProcessQScale:
        processQScaleArr = _coerceProcessQScale(processQScale, intervalCount)
        processQScalePtr = <const cnp.float32_t*>processQScaleArr.data
    if vectorD is None:
        dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
    else:
        dStatVectorArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> vectorD
        if dStatVectorArr.shape[0] < intervalCount:
            raise ValueError("vectorD length must match intervalCount")

    if doStore:
        stateForwardArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateForward
        stateCovarForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarForward
        pNoiseForwardArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> pNoiseForward
        if stateForwardArr.shape[0] < intervalCount or stateForwardArr.shape[1] < 1:
            raise ValueError("stateForward shape must match intervalCount by 1")
        if (
            stateCovarForwardArr.shape[0] < intervalCount
            or stateCovarForwardArr.shape[1] < 1
            or stateCovarForwardArr.shape[2] < 1
        ):
            raise ValueError("stateCovarForward shape must match intervalCount by 1 by 1")
        if (
            intervalCount > 1
            and (
                pNoiseForwardArr.shape[0] < intervalCount - 1
                or pNoiseForwardArr.shape[1] < 1
                or pNoiseForwardArr.shape[2] < 1
            )
        ):
            raise ValueError("pNoiseForward shape must permit intervalCount minus one 1 by 1 entries")
        stateForwardPtr = <cnp.float32_t*>stateForwardArr.data
        stateCovarForwardPtr = <cnp.float32_t*>stateCovarForwardArr.data
        pNoiseForwardPtr = <cnp.float32_t*>pNoiseForwardArr.data

    if q0 <= 1.0e-12:
        ECM_useAPN = False

    dataPtr = <const cnp.float32_t*>matrixData.data
    muncPtr = <const cnp.float32_t*>matrixPluginMuncInit.data
    blockMapPtr = <const cnp.int32_t*>intervalToBlockMap.data
    dStatPtr = <cnp.float32_t*>dStatVectorArr.data

    with nogil:
        loopResult = _levelForwardPassLoop(
            dataPtr,
            muncPtr,
            blockMapPtr,
            lambdaExpPtr,
            processPrecExpPtr,
            processQScalePtr,
            dStatPtr,
            stateForwardPtr,
            stateCovarForwardPtr,
            pNoiseForwardPtr,
            trackCount,
            intervalCount,
            blockCount,
            <double>stateInit,
            <double>stateCovarInit,
            <double>pad,
            q0,
            LOG2PI,
            wMin,
            wMax,
            procPrecMin,
            procPrecMax,
            apnMinQ,
            apnMaxQ,
            apnThresh,
            apnScaleCoef,
            apnPC,
            doStore,
            useLambda,
            useProcPrec,
            useProcessQScale,
            ECM_useAPN,
            returnNLL,
            storeNLLInD,
        )
    if loopResult.invalidBlockIndex >= 0:
        raise ValueError("intervalToBlockMap has out-of-range block id")

    phiHat = loopResult.sumDStat / (<double>intervalCount)
    sumNLL = loopResult.sumNLL
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

    # ensure there are enough usable columns for the SF calculation
    if validCols < fmax(fmin(<double>(500.0), np.sqrt(<double>(n*0.5))), 10.0):
        raise ValueError(
            f"insufficient valid/dense columns for `countingParams.normMethod: SF`, (need >= 500, got {validCols})... "
            f"If this is expected, consider using `countingParams.normMethod: EGS` or RPKM instead."
        )

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
                scaleFactorsView[s] = minSF
            elif scaleFactorsView[s] > maxSF:
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
                    scaleFactorsView[s] = minSF
                elif scaleFactorsView[s] > maxSF:
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
    object rawNoiseMass=None,
    object countNoisePseudoMeanMass=0.5,
    object countNoisePseudoVarianceMass=0.5,
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
    r"""Conditional Poisson delta-method count-transform variance floor."""
    cdef object countsObj = np.asarray(normalizedCounts, dtype=np.float64)
    cdef object rawNoiseObj = None
    cdef bint squeeze = False
    cdef bint hasRawNoise = rawNoiseMass is not None
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] counts2
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] rawNoise2
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
    cdef double pseudoMeanMass = float(countNoisePseudoMeanMass)
    cdef double pseudoVarianceMass = float(countNoisePseudoVarianceMass)
    cdef Py_ssize_t m, n, i, j
    cdef double count, sf, normalizedMean, normalizedVariance, deriv, floorValue
    cdef double rawNoiseValue, rawCount, noiseMass

    if pseudoMeanMass <= 0.0 or not isfinite(pseudoMeanMass):
        raise ValueError("countNoisePseudoMeanMass must be positive and finite")
    if pseudoVarianceMass <= 0.0 or not isfinite(pseudoVarianceMass):
        raise ValueError("countNoisePseudoVarianceMass must be positive and finite")

    if countsObj.ndim == 1:
        squeeze = True
        counts2 = np.ascontiguousarray(np.asarray(countsObj, dtype=np.float64).reshape(1, -1), dtype=np.float64)
    elif countsObj.ndim == 2:
        counts2 = np.ascontiguousarray(countsObj, dtype=np.float64)
    else:
        raise ValueError("normalizedCounts must be a 1D or 2D array")

    m = counts2.shape[0]
    n = counts2.shape[1]
    if hasRawNoise:
        rawNoiseObj = np.asarray(rawNoiseMass, dtype=np.float64)
        if rawNoiseObj.ndim == 1:
            rawNoise2 = np.ascontiguousarray(np.asarray(rawNoiseObj, dtype=np.float64).reshape(1, -1), dtype=np.float64)
        elif rawNoiseObj.ndim == 2:
            rawNoise2 = np.ascontiguousarray(rawNoiseObj, dtype=np.float64)
        else:
            raise ValueError("rawNoiseMass must be a 1D or 2D array")
        if rawNoise2.shape[0] != m or rawNoise2.shape[1] != n:
            raise ValueError("rawNoiseMass must match normalizedCounts shape")
        if np.any(np.isfinite(rawNoise2) & (rawNoise2 < 0.0)):
            raise ValueError("rawNoiseMass must be nonnegative where finite")
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
                rawCount = count / sf
                if rawCount < 0.0:
                    rawCount = 0.0
                if hasRawNoise:
                    rawNoiseValue = <double>rawNoise2[i, j]
                    if not isfinite(rawNoiseValue):
                        out[i, j] = <float>NAN
                        continue
                    noiseMass = rawNoiseValue
                else:
                    noiseMass = rawCount
                normalizedMean = (rawCount + pseudoMeanMass) * sf
                normalizedVariance = (noiseMass + pseudoVarianceMass) * sf * sf
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


def cEstimateEffectiveSampleSize(
    object values,
    int maxLag,
    object activeMask=None,
    bint logPositive=False,
    int windowIntervals=0,
):
    r"""Positive-autocorrelation effective sample size scan."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1), dtype=np.float64)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] activeArr
    cdef cnp.uint8_t[::1] activeView = None
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t activeCount = 0
    cdef Py_ssize_t pairCount
    cdef Py_ssize_t i, lag, maxLag_, lagsUsed = 0
    cdef double mean = 0.0
    cdef double var = 0.0
    cdef double cov, rho, value, taper, tau = 1.0
    cdef bint useActiveMask = activeMask is not None
    cdef bint badRho = False
    cdef int windowIntervals_ = windowIntervals
    if windowIntervals_ < 0:
        raise ValueError("windowIntervals must be nonnegative")
    if not useActiveMask and not logPositive and windowIntervals_ == 0:
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
    if useActiveMask:
        activeArr = np.ascontiguousarray(np.asarray(activeMask, dtype=np.uint8).reshape(-1), dtype=np.uint8)
        if activeArr.shape[0] != n:
            raise ValueError("activeMask length must match values length")
        activeView = activeArr
    for i in range(n):
        if useActiveMask and activeView[i] == 0:
            continue
        value = x[i]
        if logPositive:
            if (not isfinite(value)) or value <= 0.0:
                raise ValueError("active values must be positive finite")
            value = log(value)
        elif not isfinite(value):
            raise ValueError("active values must be finite")
        x[i] = value
        mean += value
        activeCount += 1
    if activeCount < 2:
        return float(activeCount), 1.0, 0
    mean /= <double>activeCount
    with nogil:
        for i in range(n):
            if useActiveMask and activeView[i] == 0:
                continue
            x[i] = x[i] - mean
            var += x[i] * x[i]
    var /= <double>activeCount
    if not isfinite(var):
        raise ValueError("variance estimate must be finite")
    if var <= 2.2250738585072014e-308:
        return float(activeCount), 1.0, 0
    maxLag_ = max(1, min(int(maxLag), n - 1))
    if windowIntervals_ > 0 and windowIntervals_ - 1 < maxLag_:
        maxLag_ = windowIntervals_ - 1
    with nogil:
        for lag in range(1, maxLag_ + 1):
            cov = 0.0
            pairCount = 0
            for i in range(n - lag):
                if useActiveMask and (activeView[i] == 0 or activeView[i + lag] == 0):
                    continue
                cov += x[i] * x[i + lag]
                pairCount += 1
            if pairCount <= 0:
                continue
            cov /= <double>pairCount
            rho = cov / var
            if not isfinite(rho):
                badRho = True
                break
            if windowIntervals_ <= 0:
                if rho <= 0.0:
                    break
                tau += 2.0 * rho
                lagsUsed = lag
            else:
                if rho > 0.0:
                    taper = 1.0 - (<double>lag / <double>windowIntervals_)
                    if taper > 0.0:
                        tau += 2.0 * taper * rho
                        lagsUsed = lag
    if badRho:
        raise ValueError("autocorrelation estimate must be finite")
    if tau < 1.0:
        tau = 1.0
    if windowIntervals_ > 0 and tau > <double>windowIntervals_:
        tau = <double>windowIntervals_
    return float(activeCount / tau), float(tau), int(lagsUsed)

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


cpdef object cStationaryNullDWBDraw(object template, int bandwidth, object rng, object kernel="bartlett"):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] templateArr = np.ascontiguousarray(
        np.asarray(template, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef int bw = bandwidth if bandwidth >= 2 else 2
    cdef int kernelCode = _dwbKernelCodeRefactor(kernel)
    cdef int maxLag = _dwbMaxLagRefactor(bw, kernelCode)
    cdef object noise = rng.standard_normal(int(templateArr.shape[0] + 2 * maxLag))
    cdef object multipliers = cGenerateDWBMultipliersFromNoise(noise, bw, kernel)
    return cApplyStationaryNullDWB(templateArr, multipliers)


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


def cMultiscaleCandidateSegmentStats(
    object scores,
    object scales,
    object thresholds,
    object nullScales,
    int minRunBins=1,
    int maxGapBins=0,
    int maxSegmentsPerView=0,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] scoreArr = np.ascontiguousarray(
        np.asarray(scores, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] scaleArr = np.ascontiguousarray(
        np.asarray(scales, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] thresholdArr = np.ascontiguousarray(
        np.asarray(thresholds, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] nullScaleArr = np.ascontiguousarray(
        np.asarray(nullScales, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    if thresholdArr.shape[0] != nullScaleArr.shape[0]:
        raise ValueError("thresholds and nullScales must have the same length")

    cdef Py_ssize_t n = scoreArr.shape[0]
    cdef Py_ssize_t scaleCount = scaleArr.shape[0]
    cdef Py_ssize_t viewCount = thresholdArr.shape[0]
    cdef int minRun = minRunBins if minRunBins > 1 else 1
    cdef int gap = maxGapBins if maxGapBins > 0 else 0
    cdef int cap = maxSegmentsPerView if maxSegmentsPerView > 0 else 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] smoothArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] prefixArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] excessArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] excessPrefixArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] runStarts
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] runEnds
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] candStarts
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] candEnds
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] candScores
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] candIntegrated
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] candMean
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] candMax
    cdef object scoreSlice
    cdef object startSlice
    cdef object selected
    cdef Py_ssize_t si, vi, i, j, r, k
    cdef Py_ssize_t w, leftPad, rightPad, startIdx, endIdx
    cdef Py_ssize_t runStart, lastTrue, runCount, keepCount
    cdef double threshold, nullScale, value, integrated, runLength
    cdef double bestScore, maxValue
    cdef Py_ssize_t bestIdx
    cdef Py_ssize_t eligibleCount = 0
    cdef Py_ssize_t perViewCapHitCount = 0
    cdef Py_ssize_t perViewDiscardedCount = 0
    cdef list startOut = []
    cdef list endOut = []
    cdef list scaleOut = []
    cdef list viewOut = []
    cdef list scoreOut = []
    cdef list integratedOut = []
    cdef list meanOut = []
    cdef list maxOut = []

    if n <= 0 or scaleCount <= 0 or viewCount <= 0:
        return (
            np.asarray(startOut, dtype=np.int64),
            np.asarray(endOut, dtype=np.int64),
            np.asarray(scaleOut, dtype=np.int64),
            np.asarray(viewOut, dtype=np.int64),
            np.asarray(scoreOut, dtype=np.float64),
            np.asarray(integratedOut, dtype=np.float64),
            np.asarray(meanOut, dtype=np.float64),
            np.asarray(maxOut, dtype=np.float64),
            int(0),
            int(0),
            int(0),
        )

    runStarts = np.empty(n, dtype=np.int64)
    runEnds = np.empty(n, dtype=np.int64)
    candStarts = np.empty(n, dtype=np.int64)
    candEnds = np.empty(n, dtype=np.int64)
    candScores = np.empty(n, dtype=np.float64)
    candIntegrated = np.empty(n, dtype=np.float64)
    candMean = np.empty(n, dtype=np.float64)
    candMax = np.empty(n, dtype=np.float64)
    prefixArr = np.empty(n + 1, dtype=np.float64)
    smoothArr = np.empty(n, dtype=np.float64)
    excessArr = np.empty(n, dtype=np.float64)
    excessPrefixArr = np.empty(n + 1, dtype=np.float64)

    for si in range(scaleCount):
        w = scaleArr[si]
        if w < 1:
            w = 1
        if w > n:
            w = n
        prefixArr[0] = 0.0
        for i in range(n):
            prefixArr[i + 1] = prefixArr[i] + scoreArr[i]
        if w <= 1 or n <= 1:
            for i in range(n):
                smoothArr[i] = scoreArr[i]
        else:
            leftPad = (w - 1) // 2
            rightPad = w - 1 - leftPad
            for i in range(n):
                startIdx = i - leftPad
                if startIdx < 0:
                    startIdx = 0
                endIdx = i + rightPad + 1
                if endIdx > n:
                    endIdx = n
                smoothArr[i] = (prefixArr[endIdx] - prefixArr[startIdx]) / <double>w

        for vi in range(viewCount):
            threshold = thresholdArr[vi]
            nullScale = nullScaleArr[vi]
            if nullScale < DBL_MIN:
                nullScale = DBL_MIN
            excessPrefixArr[0] = 0.0
            for i in range(n):
                value = (scoreArr[i] - threshold) / nullScale
                if value < 0.0:
                    value = 0.0
                excessArr[i] = value
                excessPrefixArr[i + 1] = excessPrefixArr[i] + value

            runCount = 0
            runStart = -1
            lastTrue = -1
            for i in range(n):
                if smoothArr[i] > threshold:
                    if runStart < 0:
                        runStart = i
                    elif i - lastTrue > gap + 1:
                        runStarts[runCount] = runStart
                        runEnds[runCount] = lastTrue
                        runCount += 1
                        runStart = i
                    lastTrue = i
            if runStart >= 0:
                runStarts[runCount] = runStart
                runEnds[runCount] = lastTrue
                runCount += 1
            if runCount <= 0:
                continue

            keepCount = 0
            for r in range(runCount):
                runLength = <double>(runEnds[r] - runStarts[r] + 1)
                if runLength < <double>minRun:
                    continue
                integrated = excessPrefixArr[runEnds[r] + 1] - excessPrefixArr[runStarts[r]]
                maxValue = 0.0
                for j in range(runStarts[r], runEnds[r] + 1):
                    if excessArr[j] > maxValue:
                        maxValue = excessArr[j]
                candStarts[keepCount] = runStarts[r]
                candEnds[keepCount] = runEnds[r]
                candIntegrated[keepCount] = integrated
                candMean[keepCount] = integrated / runLength
                candScores[keepCount] = integrated / sqrt(fmax(runLength, 1.0))
                candMax[keepCount] = maxValue
                keepCount += 1
            if keepCount <= 0:
                continue
            eligibleCount += keepCount

            if cap > 0 and keepCount > cap:
                perViewCapHitCount += 1
                perViewDiscardedCount += keepCount - cap
                scoreSlice = np.asarray(candScores[:keepCount], dtype=np.float64)
                startSlice = np.asarray(candStarts[:keepCount], dtype=np.int64)
                selected = np.argpartition(-scoreSlice, cap - 1)[:cap]
                selected = selected[np.argsort(startSlice[selected], kind="mergesort")]
                for k in selected:
                    r = <Py_ssize_t>k
                    startOut.append(int(candStarts[r]))
                    endOut.append(int(candEnds[r]))
                    scaleOut.append(int(w))
                    viewOut.append(int(vi))
                    scoreOut.append(float(candScores[r]))
                    integratedOut.append(float(candIntegrated[r]))
                    meanOut.append(float(candMean[r]))
                    maxOut.append(float(candMax[r]))
            else:
                for r in range(keepCount):
                    startOut.append(int(candStarts[r]))
                    endOut.append(int(candEnds[r]))
                    scaleOut.append(int(w))
                    viewOut.append(int(vi))
                    scoreOut.append(float(candScores[r]))
                    integratedOut.append(float(candIntegrated[r]))
                    meanOut.append(float(candMean[r]))
                    maxOut.append(float(candMax[r]))

    return (
        np.asarray(startOut, dtype=np.int64),
        np.asarray(endOut, dtype=np.int64),
        np.asarray(scaleOut, dtype=np.int64),
        np.asarray(viewOut, dtype=np.int64),
        np.asarray(scoreOut, dtype=np.float64),
        np.asarray(integratedOut, dtype=np.float64),
        np.asarray(meanOut, dtype=np.float64),
        np.asarray(maxOut, dtype=np.float64),
        int(eligibleCount),
        int(perViewCapHitCount),
        int(perViewDiscardedCount),
    )

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


def cbackgroundWeightedStatsWithSupport(object residualMatrix, object invVarMatrix):
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] residualArr = np.ascontiguousarray(residualMatrix, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] invArr = np.ascontiguousarray(invVarMatrix, dtype=np.float32)
    if residualArr.ndim != 2 or invArr.shape[0] != residualArr.shape[0] or invArr.shape[1] != residualArr.shape[1]:
        raise ValueError("residualMatrix and invVarMatrix must have identical 2D shapes")
    cdef Py_ssize_t m = residualArr.shape[0]
    cdef Py_ssize_t n = residualArr.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weightArr = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] rhsArr = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i, j
    cdef Py_ssize_t supportCount = 0
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
            if wsum > 0.0:
                supportCount += 1
    return weightArr, rhsArr, int(supportCount)



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

# ==============================================
# Post-fit state-shrinkage helpers
# ==============================================

cdef inline double _state_shrink_safe_variance(double v) noexcept nogil:
    if (not isfinite(v)) or v <= 1.0e-12:
        return 1.0e-12
    return v

cdef inline bint _state_shrink_valid(double x, double v) noexcept nogil:
    return isfinite(x) and isfinite(v) and v > 0.0

cpdef tuple cstateShrinkInitialSums(
    object state,
    object variance,
    double nullZ,
    int blockSize=1,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] xArr = np.ascontiguousarray(
        np.asarray(state, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vArr = np.ascontiguousarray(
        np.asarray(variance, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef Py_ssize_t n = xArr.shape[0]
    if vArr.shape[0] != n:
        raise ValueError("state and variance must have the same length")
    cdef double[::1] xView = xArr
    cdef double[::1] vView = vArr
    cdef Py_ssize_t start, end, i
    cdef Py_ssize_t validInBlock
    cdef int block = blockSize if blockSize > 0 else 1
    cdef double x, v, z, weight
    cdef double totalWeight = 0.0
    cdef double centralWeight = 0.0
    cdef double excessMomentSum = 0.0
    cdef double varianceSum = 0.0
    cdef double x2
    cdef Py_ssize_t finiteCount = 0
    cdef double nullZSafe = nullZ if nullZ > 1.0e-12 else 1.0e-12

    with nogil:
        start = 0
        while start < n:
            end = start + block
            if end > n:
                end = n
            validInBlock = 0
            for i in range(start, end):
                if _state_shrink_valid(xView[i], vView[i]):
                    validInBlock += 1
            if validInBlock > 0:
                weight = 1.0 / <double>validInBlock
                for i in range(start, end):
                    x = xView[i]
                    v = vView[i]
                    if _state_shrink_valid(x, v):
                        v = _state_shrink_safe_variance(v)
                        x2 = x * x
                        z = fabs(x) / sqrt(v)
                        totalWeight += weight
                        if z <= nullZSafe:
                            centralWeight += weight
                        excessMomentSum += weight * (x2 - v)
                        varianceSum += weight * v
                        finiteCount += 1
            start = end
    return (
        totalWeight,
        centralWeight,
        excessMomentSum,
        varianceSum,
        int(finiteCount),
    )

cpdef tuple cstateShrinkMixtureEMStep(
    object state,
    object variance,
    double priorSpikeProp,
    object slabVariance,
    object slabWeight,
    int blockSize=1,
):
    cdef object tauObj = np.asarray(slabVariance, dtype=np.float64)
    cdef object slabWeightObj = np.asarray(slabWeight, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] tauArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] slabPriorWeightArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logSlabPriorArr
    cdef Py_ssize_t slabCount
    cdef Py_ssize_t j
    cdef double pi0 = priorSpikeProp
    cdef double slabWeightTotal = 0.0
    cdef double logPriorScale

    if getattr(tauObj, "ndim", 0) != 1:
        raise ValueError("slabVariance must be one-dimensional")
    if getattr(slabWeightObj, "ndim", 0) != 1:
        raise ValueError("slabWeight must be one-dimensional")
    tauArr = np.ascontiguousarray(tauObj, dtype=np.float64)
    slabPriorWeightArr = np.ascontiguousarray(slabWeightObj, dtype=np.float64)
    slabCount = tauArr.shape[0]
    if slabCount <= 0:
        raise ValueError("slab arrays must be nonempty")
    if slabPriorWeightArr.shape[0] != slabCount:
        raise ValueError("slabVariance and slabWeight must have the same length")
    if (not isfinite(pi0)) or pi0 <= 0.0 or pi0 >= 1.0:
        raise ValueError("priorSpikeProp must be finite with 0 < priorSpikeProp < 1")

    cdef double[::1] tauView = tauArr
    cdef double[::1] slabPriorWeightView = slabPriorWeightArr

    with nogil:
        for j in range(slabCount):
            if (not isfinite(tauView[j])) or tauView[j] <= 0.0:
                slabWeightTotal = -1.0
                break
            if (not isfinite(slabPriorWeightView[j])) or slabPriorWeightView[j] < 0.0:
                slabWeightTotal = -2.0
                break
            slabWeightTotal += slabPriorWeightView[j]
    if slabWeightTotal == -1.0:
        raise ValueError("slabVariance must contain only positive finite values")
    if slabWeightTotal == -2.0 or slabWeightTotal <= 0.0 or not isfinite(slabWeightTotal):
        raise ValueError("slabWeight must contain only finite nonnegative values with positive sum")

    logPriorScale = log(1.0 - pi0) - log(slabWeightTotal)
    logSlabPriorArr = np.empty(slabCount, dtype=np.float64)
    cdef double[::1] logSlabPriorView = logSlabPriorArr
    with nogil:
        for j in range(slabCount):
            if slabPriorWeightView[j] > 0.0:
                logSlabPriorView[j] = logPriorScale + log(slabPriorWeightView[j])
            else:
                logSlabPriorView[j] = -INFINITY
    return cstateShrinkMixtureEMStepPrepared(
        state,
        variance,
        pi0,
        tauArr,
        logSlabPriorArr,
        blockSize,
    )

cpdef tuple cstateShrinkMixtureEMStepPrepared(
    object state,
    object variance,
    double priorSpikeProp,
    object slabVariance,
    object logSlabPrior,
    int blockSize=1,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] xArr = np.ascontiguousarray(
        np.asarray(state, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vArr = np.ascontiguousarray(
        np.asarray(variance, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef object tauObj = np.asarray(slabVariance, dtype=np.float64)
    cdef object logSlabPriorObj = np.asarray(logSlabPrior, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] tauArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logSlabPriorArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] slabMassArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] slabSecondArr
    cdef Py_ssize_t n = xArr.shape[0]
    cdef Py_ssize_t slabCount
    cdef Py_ssize_t start, end, i, j, blockIndex, blockCount
    cdef Py_ssize_t validInBlock
    cdef int block = blockSize if blockSize > 0 else 1
    cdef double pi0 = priorSpikeProp
    cdef double x, v, weight, logNull, logDenom
    cdef double maxLog, denomSum, resp, tau2, slabShrinkage, slabMean
    cdef double slabPosteriorVariance, logNullPrior, logValue, x2, vPlusTau
    cdef double totalWeight = 0.0
    cdef double nullMass = 0.0
    cdef double logLikelihood = 0.0
    cdef Py_ssize_t finiteCount = 0
    cdef double* logSlabScratch = NULL
    cdef double* respScratch = NULL

    if vArr.shape[0] != n:
        raise ValueError("state and variance must have the same length")
    if getattr(tauObj, "ndim", 0) != 1:
        raise ValueError("slabVariance must be one-dimensional")
    if getattr(logSlabPriorObj, "ndim", 0) != 1:
        raise ValueError("logSlabPrior must be one-dimensional")
    tauArr = np.ascontiguousarray(tauObj, dtype=np.float64)
    logSlabPriorArr = np.ascontiguousarray(logSlabPriorObj, dtype=np.float64)
    slabCount = tauArr.shape[0]
    if slabCount <= 0:
        raise ValueError("slab arrays must be nonempty")
    if logSlabPriorArr.shape[0] != slabCount:
        raise ValueError("slabVariance and logSlabPrior must have the same length")
    if (not isfinite(pi0)) or pi0 <= 0.0 or pi0 >= 1.0:
        raise ValueError("priorSpikeProp must be finite with 0 < priorSpikeProp < 1")

    cdef double[::1] xView = xArr
    cdef double[::1] vView = vArr
    cdef double[::1] tauView = tauArr
    cdef double[::1] logSlabPriorView = logSlabPriorArr

    with nogil:
        for j in range(slabCount):
            if (not isfinite(tauView[j])) or tauView[j] <= 0.0:
                totalWeight = -1.0
                break
    if totalWeight == -1.0:
        raise ValueError("slabVariance must contain only positive finite values")

    logNullPrior = log(pi0)
    slabMassArr = np.zeros(slabCount, dtype=np.float64)
    slabSecondArr = np.zeros(slabCount, dtype=np.float64)
    cdef double[::1] slabMassView = slabMassArr
    cdef double[::1] slabSecondView = slabSecondArr

    logSlabScratch = <double*>malloc(slabCount * sizeof(double))
    if logSlabScratch == NULL:
        raise MemoryError()
    respScratch = <double*>malloc(slabCount * sizeof(double))
    if respScratch == NULL:
        free(logSlabScratch)
        raise MemoryError()
    try:
        blockCount = (n + block - 1) // block
        with nogil:
            for blockIndex in range(blockCount):
                start = blockIndex * block
                end = start + block
                if end > n:
                    end = n
                validInBlock = 0
                for i in range(start, end):
                    if _state_shrink_valid(xView[i], vView[i]):
                        validInBlock += 1
                if validInBlock > 0:
                    weight = 1.0 / <double>validInBlock
                    for i in range(start, end):
                        x = xView[i]
                        v = vView[i]
                        if _state_shrink_valid(x, v):
                            v = _state_shrink_safe_variance(v)
                            x2 = x * x
                            logNull = logNullPrior - 0.5 * (
                                log(2.0 * __PI_DOUBLE * v) + x2 / v
                            )
                            maxLog = logNull
                            for j in range(slabCount):
                                vPlusTau = v + tauView[j]
                                logValue = logSlabPriorView[j] - 0.5 * (
                                    log(2.0 * __PI_DOUBLE * vPlusTau)
                                    + x2 / vPlusTau
                                )
                                logSlabScratch[j] = logValue
                                if logValue > maxLog:
                                    maxLog = logValue
                            denomSum = exp(logNull - maxLog)
                            for j in range(slabCount):
                                respScratch[j] = exp(logSlabScratch[j] - maxLog)
                                denomSum += respScratch[j]
                            logDenom = maxLog + log(denomSum)
                            nullMass += weight * (exp(logNull - maxLog) / denomSum)
                            for j in range(slabCount):
                                resp = respScratch[j] / denomSum
                                tau2 = tauView[j]
                                slabShrinkage = tau2 / (tau2 + v)
                                slabMean = slabShrinkage * x
                                slabPosteriorVariance = slabShrinkage * v
                                slabMassView[j] += weight * resp
                                slabSecondView[j] += weight * resp * (
                                    slabPosteriorVariance + slabMean * slabMean
                                )
                            totalWeight += weight
                            logLikelihood += weight * logDenom
                            finiteCount += 1
    finally:
        free(respScratch)
        free(logSlabScratch)
    return (
        totalWeight,
        nullMass,
        slabMassArr,
        slabSecondArr,
        logLikelihood,
        int(finiteCount),
    )

cpdef tuple cstateShrinkMixturePosterior(
    object state,
    object variance,
    double priorSpikeProp,
    object slabVariance,
    object slabWeight,
):
    cdef object tauObj = np.asarray(slabVariance, dtype=np.float64)
    cdef object slabWeightObj = np.asarray(slabWeight, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] tauArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] slabPriorWeightArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logSlabPriorArr
    cdef Py_ssize_t slabCount
    cdef Py_ssize_t j
    cdef double pi0 = priorSpikeProp
    cdef double slabWeightTotal = 0.0
    cdef double logPriorScale

    if getattr(tauObj, "ndim", 0) != 1:
        raise ValueError("slabVariance must be one-dimensional")
    if getattr(slabWeightObj, "ndim", 0) != 1:
        raise ValueError("slabWeight must be one-dimensional")
    tauArr = np.ascontiguousarray(tauObj, dtype=np.float64)
    slabPriorWeightArr = np.ascontiguousarray(slabWeightObj, dtype=np.float64)
    slabCount = tauArr.shape[0]
    if slabCount <= 0:
        raise ValueError("slab arrays must be nonempty")
    if slabPriorWeightArr.shape[0] != slabCount:
        raise ValueError("slabVariance and slabWeight must have the same length")
    if (not isfinite(pi0)) or pi0 <= 0.0 or pi0 >= 1.0:
        raise ValueError("priorSpikeProp must be finite with 0 < priorSpikeProp < 1")

    cdef double[::1] tauView = tauArr
    cdef double[::1] slabPriorWeightView = slabPriorWeightArr

    with nogil:
        for j in range(slabCount):
            if (not isfinite(tauView[j])) or tauView[j] <= 0.0:
                slabWeightTotal = -1.0
                break
            if (not isfinite(slabPriorWeightView[j])) or slabPriorWeightView[j] < 0.0:
                slabWeightTotal = -2.0
                break
            slabWeightTotal += slabPriorWeightView[j]
    if slabWeightTotal == -1.0:
        raise ValueError("slabVariance must contain only positive finite values")
    if slabWeightTotal == -2.0 or slabWeightTotal <= 0.0 or not isfinite(slabWeightTotal):
        raise ValueError("slabWeight must contain only finite nonnegative values with positive sum")

    logPriorScale = log(1.0 - pi0) - log(slabWeightTotal)
    logSlabPriorArr = np.empty(slabCount, dtype=np.float64)
    cdef double[::1] logSlabPriorView = logSlabPriorArr
    with nogil:
        for j in range(slabCount):
            if slabPriorWeightView[j] > 0.0:
                logSlabPriorView[j] = logPriorScale + log(slabPriorWeightView[j])
            else:
                logSlabPriorView[j] = -INFINITY
    return cstateShrinkMixturePosteriorPrepared(
        state,
        variance,
        pi0,
        tauArr,
        logSlabPriorArr,
    )

cpdef tuple cstateShrinkMixturePosteriorPrepared(
    object state,
    object variance,
    double priorSpikeProp,
    object slabVariance,
    object logSlabPrior,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] xArr = np.ascontiguousarray(
        np.asarray(state, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vArr = np.ascontiguousarray(
        np.asarray(variance, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef object tauObj = np.asarray(slabVariance, dtype=np.float64)
    cdef object logSlabPriorObj = np.asarray(logSlabPrior, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] tauArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logSlabPriorArr
    cdef Py_ssize_t n = xArr.shape[0]
    cdef Py_ssize_t slabCount
    cdef double pi0 = priorSpikeProp
    cdef double logNullPrior, x, v, logNull, maxLog, denomSum, resp
    cdef double logValue, nullProb, tau2, slabShrinkage, slabMean
    cdef double slabPosteriorVariance, shrunk, postSecond, postVariance, posteriorSd
    cdef double slabPosteriorWeight
    cdef double x2, vPlusTau
    cdef double badTau = 0.0
    cdef Py_ssize_t i, j
    cdef double* logSlabScratch = NULL
    cdef double* respScratch = NULL

    if vArr.shape[0] != n:
        raise ValueError("state and variance must have the same length")
    if getattr(tauObj, "ndim", 0) != 1:
        raise ValueError("slabVariance must be one-dimensional")
    if getattr(logSlabPriorObj, "ndim", 0) != 1:
        raise ValueError("logSlabPrior must be one-dimensional")
    tauArr = np.ascontiguousarray(tauObj, dtype=np.float64)
    logSlabPriorArr = np.ascontiguousarray(logSlabPriorObj, dtype=np.float64)
    slabCount = tauArr.shape[0]
    if slabCount <= 0:
        raise ValueError("slab arrays must be nonempty")
    if logSlabPriorArr.shape[0] != slabCount:
        raise ValueError("slabVariance and logSlabPrior must have the same length")
    if (not isfinite(pi0)) or pi0 <= 0.0 or pi0 >= 1.0:
        raise ValueError("priorSpikeProp must be finite with 0 < priorSpikeProp < 1")

    cdef double[::1] xView = xArr
    cdef double[::1] vView = vArr
    cdef double[::1] tauView = tauArr
    cdef double[::1] logSlabPriorView = logSlabPriorArr

    with nogil:
        for j in range(slabCount):
            if (not isfinite(tauView[j])) or tauView[j] <= 0.0:
                badTau = 1.0
                break
    if badTau > 0.0:
        raise ValueError("slabVariance must contain only positive finite values")

    logNullPrior = log(pi0)

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] shrunkArr = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] posteriorSdArr = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] spikePropArr = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] slabMeanArr = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] slabWeightArr = np.empty(n, dtype=np.float32)
    cdef cnp.float32_t[::1] shrunkView = shrunkArr
    cdef cnp.float32_t[::1] posteriorSdView = posteriorSdArr
    cdef cnp.float32_t[::1] spikePropView = spikePropArr
    cdef cnp.float32_t[::1] slabMeanView = slabMeanArr
    cdef cnp.float32_t[::1] slabWeightView = slabWeightArr

    logSlabScratch = <double*>malloc(slabCount * sizeof(double))
    if logSlabScratch == NULL:
        raise MemoryError()
    respScratch = <double*>malloc(slabCount * sizeof(double))
    if respScratch == NULL:
        free(logSlabScratch)
        raise MemoryError()
    try:
        with nogil:
            for i in range(n):
                x = xView[i]
                v = vView[i]
                if _state_shrink_valid(x, v):
                    v = _state_shrink_safe_variance(v)
                    x2 = x * x
                    logNull = logNullPrior - 0.5 * (
                        log(2.0 * __PI_DOUBLE * v) + x2 / v
                    )
                    maxLog = logNull
                    for j in range(slabCount):
                        vPlusTau = v + tauView[j]
                        logValue = logSlabPriorView[j] - 0.5 * (
                            log(2.0 * __PI_DOUBLE * vPlusTau) + x2 / vPlusTau
                        )
                        logSlabScratch[j] = logValue
                        if logValue > maxLog:
                            maxLog = logValue
                    denomSum = exp(logNull - maxLog)
                    for j in range(slabCount):
                        respScratch[j] = exp(logSlabScratch[j] - maxLog)
                        denomSum += respScratch[j]
                    nullProb = exp(logNull - maxLog) / denomSum
                    shrunk = 0.0
                    postSecond = 0.0
                    slabPosteriorWeight = 0.0
                    for j in range(slabCount):
                        resp = respScratch[j] / denomSum
                        tau2 = tauView[j]
                        slabShrinkage = tau2 / (tau2 + v)
                        slabMean = slabShrinkage * x
                        slabPosteriorVariance = slabShrinkage * v
                        slabPosteriorWeight += resp
                        shrunk += resp * slabMean
                        postSecond += resp * (
                            slabPosteriorVariance + slabMean * slabMean
                        )
                    postVariance = postSecond - shrunk * shrunk
                    if (not isfinite(postVariance)) or postVariance <= 1.0e-12:
                        postVariance = 1.0e-12
                    posteriorSd = sqrt(postVariance)
                    shrunkView[i] = <cnp.float32_t>shrunk
                    posteriorSdView[i] = <cnp.float32_t>posteriorSd
                    spikePropView[i] = <cnp.float32_t>nullProb
                    if slabPosteriorWeight > 1.0e-12:
                        slabMeanView[i] = <cnp.float32_t>(shrunk / slabPosteriorWeight)
                    else:
                        slabMeanView[i] = <cnp.float32_t>0.0
                    slabWeightView[i] = <cnp.float32_t>slabPosteriorWeight
                else:
                    shrunkView[i] = <cnp.float32_t>x
                    posteriorSdView[i] = <cnp.float32_t>NAN
                    spikePropView[i] = <cnp.float32_t>NAN
                    slabMeanView[i] = <cnp.float32_t>NAN
                    slabWeightView[i] = <cnp.float32_t>NAN
    finally:
        free(respScratch)
        free(logSlabScratch)
    return shrunkArr, posteriorSdArr, spikePropArr, slabMeanArr, slabWeightArr
