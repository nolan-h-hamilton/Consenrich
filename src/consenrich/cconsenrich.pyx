# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
# distutils: language = c
r"""Cython module for Consenrich core functions.

This module contains Cython implementations of core functions used in Consenrich.
"""

cimport cython
import numbers
import os
import numpy as np
from . import constants, misc_util
from scipy import ndimage, signal
from scipy.fft import irfft, next_fast_len, rfft
cimport numpy as cnp
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from numpy.random import default_rng
from numpy.random cimport bitgen_t
from libc.math cimport isfinite, fabs, fma, hypot, log1p, log2, log, log2f, logf, asinhf, asinh, fmax, fmaxf, pow, sqrt, sqrtf, fabsf, fminf, fmin, log10, log10f, ceil, floor, floorf, exp, expf, erf, isnan, lgamma, nextafterf, NAN, INFINITY
from libc.float cimport DBL_MIN
from libc.math cimport frexp, ldexp
from libc.errno cimport EIO, EOVERFLOW, errno
from libc.locale cimport localeconv
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdio cimport FILE, fclose, fopen, fwrite, snprintf, printf, fprintf, fflush, stdout, stderr
from cython.parallel cimport prange
from .numericMoments cimport centeredMomentsAdd, compensatedSumAdd

cdef extern from "fenv.h" nogil:
    int FE_TONEAREST
    int fegetround()

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
cdef dict dependenceTaperCache = {}
ctypedef fused real_t:
    float
    double

# ===============
# inline/helpers
# ===============

cdef inline double _clampMultiplierValue(double value, double lower, double upper) noexcept nogil:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


cdef inline bint _projectCovariance2x2(
    double* cov00,
    double* cov01,
    double* cov11,
) noexcept nogil:
    cdef double a = cov00[0]
    cdef double b = cov01[0]
    cdef double c = cov11[0]
    cdef double halfTrace
    cdef double radius
    cdef double eigLarge
    cdef double eigSmall
    cdef double gap
    cdef double multiplier
    cdef double determinant
    cdef double trace

    if not isfinite(a) or not isfinite(b) or not isfinite(c):
        return False

    trace = a + c
    determinant = a * c - b * b
    # det / trace^2 <= lambdaMin / lambdaMax for a positive definite matrix.
    if (
        a > 0.0 and c > 0.0
        and determinant > DBL_MIN
        and determinant > 1.0e-12 * trace * trace
    ):
        return True

    halfTrace = 0.5 * trace
    radius = hypot(0.5 * (a - c), b)
    eigLarge = halfTrace + radius
    eigSmall = halfTrace - radius
    if eigSmall >= 0.0:
        return True
    if eigLarge < 0.0:
        return False
    if eigLarge == 0.0:
        if eigSmall < 0.0:
            return False
        cov00[0] = 0.0
        cov01[0] = 0.0
        cov11[0] = 0.0
        return True

    gap = eigLarge - eigSmall
    multiplier = eigLarge / gap
    cov00[0] = multiplier * (a - eigSmall)
    cov01[0] = multiplier * b
    cov11[0] = multiplier * (c - eigSmall)
    return True


cdef inline bint _storeCovariance2x2F32(
    double cov00,
    double cov01,
    double cov11,
    cnp.float32_t* destination,
) noexcept nogil:
    cdef cnp.float32_t stored00 = <cnp.float32_t>cov00
    cdef cnp.float32_t stored01 = <cnp.float32_t>cov01
    cdef cnp.float32_t stored11 = <cnp.float32_t>cov11
    cdef cnp.float32_t storedAbs
    cdef double diagonalProduct
    cdef double offDiagonalSquare
    cdef double maxOffDiagonal

    if (
        not isfinite(<double>stored00)
        or not isfinite(<double>stored01)
        or not isfinite(<double>stored11)
        or stored00 < 0.0
        or stored11 < 0.0
    ):
        return False

    diagonalProduct = (<double>stored00) * (<double>stored11)
    offDiagonalSquare = (<double>stored01) * (<double>stored01)
    if offDiagonalSquare > diagonalProduct:
        maxOffDiagonal = sqrt(diagonalProduct)
        storedAbs = <cnp.float32_t>maxOffDiagonal
        if (<double>storedAbs) * (<double>storedAbs) > diagonalProduct:
            storedAbs = nextafterf(storedAbs, <cnp.float32_t>0.0)
        if stored01 < 0.0:
            stored01 = -storedAbs
        else:
            stored01 = storedAbs

    destination[0] = stored00
    destination[1] = stored01
    destination[2] = stored01
    destination[3] = stored11
    return True


cdef inline bint _inverseSymmetricPSD2x2(
    double cov00,
    double cov01,
    double cov11,
    double* inv00,
    double* inv01,
    double* inv11,
    bint* rankDeficient,
) noexcept nogil:
    cdef double halfTrace
    cdef double radius
    cdef double eigLarge
    cdef double eigSmall
    cdef double eigGap
    cdef double pivot
    cdef double factor
    cdef double schur
    cdef double invSchur
    cdef double denominator
    cdef double determinant
    cdef double trace
    cdef bint isPosDef

    rankDeficient[0] = True
    if not isfinite(cov00) or not isfinite(cov01) or not isfinite(cov11):
        return False

    trace = cov00 + cov11
    halfTrace = 0.5 * trace
    determinant = cov00 * cov11 - cov01 * cov01
    # The determinant bound certifies the spectral cutoff with a roundoff margin.
    isPosDef = (
        cov00 > 0.0 and cov11 > 0.0
        and determinant > DBL_MIN
        and determinant > 1.0e-12 * trace * trace
    )
    if not isPosDef:
        radius = hypot(0.5 * (cov00 - cov11), cov01)
        eigLarge = halfTrace + radius
        eigSmall = halfTrace - radius
        if eigLarge < 0.0:
            return False
        if eigLarge == 0.0:
            if eigSmall < 0.0:
                return False
            inv00[0] = 0.0
            inv01[0] = 0.0
            inv11[0] = 0.0
            return True

    if isPosDef or eigSmall > 64.0 * 2.2204460492503131e-16 * eigLarge:
        if cov00 >= cov11:
            pivot = cov00
            factor = cov01 / pivot
            schur = fma(-factor, cov01, cov11)
            if schur > 0.0:
                invSchur = 1.0 / schur
                inv00[0] = (1.0 / pivot) + factor * factor * invSchur
                inv01[0] = -factor * invSchur
                inv11[0] = invSchur
                rankDeficient[0] = False
                return True
        else:
            pivot = cov11
            factor = cov01 / pivot
            schur = fma(-factor, cov01, cov00)
            if schur > 0.0:
                invSchur = 1.0 / schur
                inv00[0] = invSchur
                inv01[0] = -factor * invSchur
                inv11[0] = (1.0 / pivot) + factor * factor * invSchur
                rankDeficient[0] = False
                return True

    if isPosDef:
        radius = hypot(0.5 * (cov00 - cov11), cov01)
        eigLarge = halfTrace + radius
        eigSmall = halfTrace - radius
    eigGap = eigLarge - eigSmall
    if eigGap <= 0.0:
        return False
    denominator = eigGap * eigLarge
    inv00[0] = (cov00 - eigSmall) / denominator
    inv01[0] = cov01 / denominator
    inv11[0] = (cov11 - eigSmall) / denominator
    return True


cdef inline void _symmetricCongruence2x2(
    double a00, double a01, double a10, double a11,
    double p00, double p01, double p11,
    double* out00, double* out01, double* out11,
) noexcept nogil:
    cdef double ap00 = a00 * p00 + a01 * p01
    cdef double ap01 = a00 * p01 + a01 * p11
    cdef double ap10 = a10 * p00 + a11 * p01
    cdef double ap11 = a10 * p01 + a11 * p11
    out00[0] = ap00 * a00 + ap01 * a01
    out01[0] = 0.5 * (ap00 * a10 + ap01 * a11 + ap10 * a00 + ap11 * a01)
    out11[0] = ap10 * a10 + ap11 * a11


cdef inline void _validateMultiplierBounds(
    double lower,
    double upper,
    bint isObservation,
) except *:
    if lower <= 0.0 or upper <= 0.0 or upper < lower:
        if isObservation:
            raise ValueError("observation precision multiplier bounds must satisfy 0 < min <= max")
        raise ValueError("process precision multiplier bounds must satisfy 0 < min <= max")


cdef inline void _validateMedianScaling(
    bint enabled,
    bint reweightingEnabled,
    double lower,
    double upper,
    bint isObservation,
) except *:
    if not enabled:
        return
    if not reweightingEnabled:
        if isObservation:
            raise ValueError(
                "ECM_scaleObsPrecisionToMedian requires "
                "ECM_useObsPrecisionReweighting=True"
            )
        raise ValueError(
            "ECM_scaleProcessPrecisionToMedian requires active process "
            "precision reweighting"
        )
    if not (lower <= 1.0 and 1.0 <= upper):
        if isObservation:
            raise ValueError(
                "observation precision multiplier bounds must contain 1 when "
                "ECM_scaleObsPrecisionToMedian=True"
            )
        raise ValueError(
            "process precision multiplier bounds must contain 1 when "
            "ECM_scaleProcessPrecisionToMedian=True"
        )


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
    double logObsPrecision,
    bint returnNLL,
    double* sumInvR,
    double* meanInnov,
    double* innovM2,
    double* sumLogR,
) noexcept nogil:
    cdef double innov = observed - stateLevel
    cdef double measVar = baseVariance + pad
    cdef double invMeasVar

    if measVar < 1.0e-12:
        measVar = 1.0e-12
    invMeasVar = obsPrecision / measVar
    if returnNLL:
        sumLogR[0] += (log(measVar) - logObsPrecision)
    centeredMomentsAdd(innov, invMeasVar, sumInvR, meanInnov, innovM2)


cdef cnp.ndarray _collapseObservationStats(
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixData,
    cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] matrixMunc,
    double pad):
    r"""Get fast column-wise summaries of the observation stats"""

    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] stats = np.zeros(
        (intervalCount, 5), dtype=np.float64
    )
    cdef const cnp.float32_t* data = <const cnp.float32_t*>matrixData.data
    cdef const cnp.float32_t* munc = <const cnp.float32_t*>matrixMunc.data
    cdef double* values = <double*>stats.data
    cdef Py_ssize_t chunkStart, chunkStop, j, k, index, offset
    cdef double variance, weight
    with nogil:
        for chunkStart in range(0, intervalCount, 4096):
            chunkStop = min(chunkStart + 4096, intervalCount)
            for j in range(trackCount):
                for k in range(chunkStart, chunkStop):
                    index = j * intervalCount + k
                    variance = <double>munc[index]
                    if variance >= __MASKED_OBSERVATION_VARIANCE_CUTOFF:
                        continue
                    variance += pad
                    if variance < 1.0e-12:
                        variance = 1.0e-12
                    weight = 1.0 / variance
                    offset = k * 5
                    centeredMomentsAdd(
                        <double>data[index], weight,
                        values + offset, values + offset + 1, values + offset + 2,
                    )
                    values[offset + 3] += log(variance)
                    values[offset + 4] += 1.0
    return stats


ctypedef struct LevelTrendForwardLoopResult:
    double sumDStat
    double sumNLL
    Py_ssize_t invalidBlockIndex
    Py_ssize_t invalidCovarianceIndex


cdef LevelTrendForwardLoopResult _levelTrendForwardPassLoop(
    const cnp.float32_t* dataPtr,
    const cnp.float32_t* muncPtr,
    const double* observationStatsPtr,
    const cnp.int32_t* blockMapPtr,
    const cnp.float32_t* lambdaExpPtr,
    const cnp.float32_t* processPrecExpPtr,
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
    double log2PI,
    double wMin,
    double wMax,
    double procPrecMin,
    double procPrecMax,
    bint doStore,
    bint useLambda,
    bint useProcPrec,
    bint returnNLL,
    bint storeNLLInD,
) noexcept nogil:
    cdef LevelTrendForwardLoopResult result
    cdef Py_ssize_t k
    cdef Py_ssize_t j
    cdef Py_ssize_t idx
    cdef Py_ssize_t blockId
    cdef Py_ssize_t activeTrackCount
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
    cdef double procPrec
    cdef double obsPrec
    cdef double logObsPrec
    cdef double tmp00
    cdef double tmp01
    cdef double tmp10
    cdef double tmp11
    cdef double pred00
    cdef double pred01
    cdef double pred10
    cdef double pred11
    cdef double sumInvR
    cdef double meanInnov
    cdef double innovM2
    cdef double sumLogR
    cdef double intervalNLL
    cdef double innovScale
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
    cdef double baseVariance

    result.sumDStat = 0.0
    result.sumNLL = 0.0
    result.invalidBlockIndex = -1
    result.invalidCovarianceIndex = -1

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

        Q00 = qBase00 / procPrec
        Q01 = qBase01 / procPrec
        Q10 = qBase10 / procPrec
        Q11 = qBase11 / procPrec

        tmp00 = F00 * cov00 + F01 * cov10
        tmp01 = F00 * cov01 + F01 * cov11
        tmp10 = F10 * cov00 + F11 * cov10
        tmp11 = F10 * cov01 + F11 * cov11

        pred00 = tmp00 * F00 + tmp01 * F01 + Q00
        pred01 = tmp00 * F10 + tmp01 * F11 + Q01
        pred10 = tmp10 * F00 + tmp11 * F01 + Q10
        pred11 = tmp10 * F10 + tmp11 * F11 + Q11

        cov00 = pred00
        cov01 = 0.5 * (pred01 + pred10)
        cov10 = cov01
        cov11 = pred11

        if useLambda:
            obsPrec = _clampMultiplierValue(<double>lambdaExpPtr[k], wMin, wMax)
        else:
            obsPrec = 1.0

        logObsPrec = log(obsPrec) if returnNLL else 0.0
        sumInvR = 0.0
        meanInnov = 0.0
        innovM2 = 0.0
        sumLogR = 0.0
        intervalNLL = 0.0
        activeTrackCount = 0

        if observationStatsPtr != NULL:
            sumInvR = obsPrec * observationStatsPtr[k * 5]
            meanInnov = observationStatsPtr[k * 5 + 1] - state0
            innovM2 = obsPrec * observationStatsPtr[k * 5 + 2]
            sumLogR = observationStatsPtr[k * 5 + 3] - observationStatsPtr[k * 5 + 4] * logObsPrec
            activeTrackCount = <Py_ssize_t>observationStatsPtr[k * 5 + 4]
        else:
            for j in range(trackCount):
                idx = j * intervalCount + k
                baseVariance = <double>muncPtr[idx]
                if baseVariance >= __MASKED_OBSERVATION_VARIANCE_CUTOFF:
                    continue
                activeTrackCount += 1
                _accumulateObservationValue(
                    <double>dataPtr[idx],
                    state0,
                    baseVariance,
                    padValue,
                    obsPrec,
                    logObsPrec,
                    returnNLL,
                    &sumInvR,
                    &meanInnov,
                    &innovM2,
                    &sumLogR,
                )

        innovScale = 1.0 + cov00 * sumInvR
        if returnNLL or dStatPtr != NULL:
            quadForm = innovM2 + (sumInvR / innovScale) * (meanInnov * meanInnov)

            if returnNLL:
                intervalNLL = 0.5 * (
                    sumLogR + log(innovScale) + quadForm + (<double>activeTrackCount) * log2PI
                )
                result.sumNLL += intervalNLL

        if dStatPtr != NULL:
            if returnNLL and storeNLLInD:
                statValue = intervalNLL
            elif activeTrackCount > 0:
                statValue = quadForm / (<double>activeTrackCount)
            else:
                statValue = 0.0
            dStatPtr[k] = <cnp.float32_t>statValue
            result.sumDStat += <double>dStatPtr[k]

        delta0 = (sumInvR / innovScale) * meanInnov
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

        cov00 = new00
        cov01 = new01
        cov10 = cov01
        cov11 = new11
        if not _projectCovariance2x2(&cov00, &cov01, &cov11):
            result.invalidCovarianceIndex = k
            return result
        cov10 = cov01

        if doStore:
            stateForwardPtr[k * 2] = <cnp.float32_t>state0
            stateForwardPtr[k * 2 + 1] = <cnp.float32_t>state1
            if not _storeCovariance2x2F32(
                cov00,
                cov01,
                cov11,
                stateCovarForwardPtr + k * 4,
            ):
                result.invalidCovarianceIndex = k
                return result
            if k > 0:
                if not _storeCovariance2x2F32(
                    Q00,
                    0.5 * (Q01 + Q10),
                    Q11,
                    pNoiseForwardPtr + (k - 1) * 4,
                ):
                    result.invalidCovarianceIndex = k
                    return result

    return result


ctypedef struct LevelForwardLoopResult:
    double sumDStat
    double sumNLL
    Py_ssize_t invalidBlockIndex


cdef LevelForwardLoopResult _levelForwardPassLoop(
    const cnp.float32_t* dataPtr,
    const cnp.float32_t* muncPtr,
    const double* observationStatsPtr,
    const cnp.int32_t* blockMapPtr,
    const cnp.float32_t* lambdaExpPtr,
    const cnp.float32_t* processPrecExpPtr,
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
    bint doStore,
    bint useLambda,
    bint useProcPrec,
    bint returnNLL,
    bint storeNLLInD,
) noexcept nogil:
    cdef LevelForwardLoopResult result
    cdef Py_ssize_t k
    cdef Py_ssize_t j
    cdef Py_ssize_t idx
    cdef Py_ssize_t blockId
    cdef Py_ssize_t activeTrackCount
    cdef double stateValue = stateInitValue
    cdef double stateVar = stateCovarInitValue
    cdef double Q
    cdef double sumInvR
    cdef double meanInnov
    cdef double innovM2
    cdef double sumLogR
    cdef double intervalNLL
    cdef double innovScale
    cdef double quadForm
    cdef double statValue
    cdef double delta0
    cdef double gainG
    cdef double gainH
    cdef double IKH
    cdef double newVar
    cdef double obsPrec
    cdef double logObsPrec
    cdef double procPrec
    cdef double baseVariance

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

        Q = q0 / procPrec
        stateVar += Q

        if useLambda:
            obsPrec = _clampMultiplierValue(<double>lambdaExpPtr[k], wMin, wMax)
        else:
            obsPrec = 1.0

        logObsPrec = log(obsPrec) if returnNLL else 0.0
        sumInvR = 0.0
        meanInnov = 0.0
        innovM2 = 0.0
        sumLogR = 0.0
        intervalNLL = 0.0
        activeTrackCount = 0

        if observationStatsPtr != NULL:
            sumInvR = obsPrec * observationStatsPtr[k * 5]
            meanInnov = observationStatsPtr[k * 5 + 1] - stateValue
            innovM2 = obsPrec * observationStatsPtr[k * 5 + 2]
            sumLogR = observationStatsPtr[k * 5 + 3] - observationStatsPtr[k * 5 + 4] * logObsPrec
            activeTrackCount = <Py_ssize_t>observationStatsPtr[k * 5 + 4]
        else:
            for j in range(trackCount):
                idx = j * intervalCount + k
                baseVariance = <double>muncPtr[idx]
                if baseVariance >= __MASKED_OBSERVATION_VARIANCE_CUTOFF:
                    continue
                activeTrackCount += 1
                _accumulateObservationValue(
                    <double>dataPtr[idx],
                    stateValue,
                    baseVariance,
                    padValue,
                    obsPrec,
                    logObsPrec,
                    returnNLL,
                    &sumInvR,
                    &meanInnov,
                    &innovM2,
                    &sumLogR,
                )

        innovScale = 1.0 + stateVar * sumInvR
        if returnNLL or dStatPtr != NULL:
            quadForm = innovM2 + (sumInvR / innovScale) * (meanInnov * meanInnov)
            if returnNLL:
                intervalNLL = 0.5 * (
                    sumLogR + log(innovScale) + quadForm + (<double>activeTrackCount) * log2PI
                )
                result.sumNLL += intervalNLL

        if dStatPtr != NULL:
            if returnNLL and storeNLLInD:
                statValue = intervalNLL
            elif activeTrackCount > 0:
                statValue = quadForm / (<double>activeTrackCount)
            else:
                statValue = 0.0
            dStatPtr[k] = <cnp.float32_t>statValue
            result.sumDStat += <double>dStatPtr[k]

        delta0 = (sumInvR / innovScale) * meanInnov
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

    return result


cdef extern from "native/barcode_selection.h":
    ctypedef struct crBarcodeSelection:
        int active
    crBarcodeSelection* crBarcodeSelectionCreate(const char* tag, const char* path, const char** error)
    void crBarcodeSelectionDestroy(crBarcodeSelection* selection)
    int crBarcodeSelected(const crBarcodeSelection* selection, const bam1_t* record)


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
    cdef double residualLevel
    cdef double residualTrend
    cdef double cov00, cov01, cov10, cov11
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
        residualLevel = (
            stateSmoothed[k + 1, 0]
            - (f00 * stateSmoothed[k, 0] + f01 * stateSmoothed[k, 1])
        )
        residualTrend = (
            stateSmoothed[k + 1, 1]
            - (f10 * stateSmoothed[k, 0] + f11 * stateSmoothed[k, 1])
        )
        cov00 = stateCovarSmoothed[k, 0, 0]
        cov01 = stateCovarSmoothed[k, 0, 1]
        cov10 = stateCovarSmoothed[k, 1, 0]
        cov11 = stateCovarSmoothed[k, 1, 1]

        levelMoment = (
            residualLevel * residualLevel + stateCovarSmoothed[k + 1, 0, 0]
            - 2.0 * (f00 * lagCovSmoothed[k, 0, 0] + f01 * lagCovSmoothed[k, 1, 0])
            + f00 * (f00 * cov00 + f01 * cov01)
            + f01 * (f00 * cov10 + f01 * cov11)
        )
        trendMoment = (
            residualTrend * residualTrend + stateCovarSmoothed[k + 1, 1, 1]
            - 2.0 * (f10 * lagCovSmoothed[k, 0, 1] + f11 * lagCovSmoothed[k, 1, 1])
            + f10 * (f10 * cov00 + f11 * cov01)
            + f11 * (f10 * cov10 + f11 * cov11)
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
    cdef double residual
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
        residual = stateSmoothed[k + 1, 0] - stateSmoothed[k, 0]
        levelMoment = (
            residual * residual
            + stateCovarSmoothed[k + 1, 0, 0]
            - 2.0 * lagCovSmoothed[k, 0, 0]
            + stateCovarSmoothed[k, 0, 0]
        )
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


cdef struct _doubleDouble:
    double high
    double low


cdef inline _doubleDouble _ddValue(double value) noexcept nogil:
    cdef _doubleDouble result
    result.high = value
    result.low = 0.0
    return result


cdef inline _doubleDouble _ddAdd(_doubleDouble left, _doubleDouble right) noexcept nogil:
    cdef _doubleDouble result
    cdef double total = left.high + right.high
    cdef double part = total - left.high
    cdef double error = (left.high - (total - part)) + (right.high - part)
    error += left.low + right.low
    result.high = total + error
    part = result.high - total
    result.low = (total - (result.high - part)) + (error - part)
    return result


cdef inline _doubleDouble _ddSubtract(_doubleDouble left, _doubleDouble right) noexcept nogil:
    right.high = -right.high
    right.low = -right.low
    return _ddAdd(left, right)


cdef inline _doubleDouble _ddMultiply(_doubleDouble left, _doubleDouble right) noexcept nogil:
    cdef _doubleDouble result
    cdef double product = left.high * right.high
    cdef double error = fma(left.high, right.high, -product)
    error += left.high * right.low + left.low * right.high
    error += left.low * right.low
    result.high = product + error
    result.low = error - (result.high - product)
    return result


cdef inline _doubleDouble _ddDivide(_doubleDouble numerator, _doubleDouble denominator) noexcept nogil:
    cdef _doubleDouble quotient = _ddValue(numerator.high / denominator.high)
    cdef _doubleDouble remainder = _ddSubtract(numerator, _ddMultiply(denominator, quotient))
    cdef _doubleDouble correction = _ddValue(remainder.high / denominator.high)
    quotient = _ddAdd(quotient, correction)
    remainder = _ddSubtract(remainder, _ddMultiply(denominator, correction))
    return _ddAdd(quotient, _ddValue(remainder.high / denominator.high))


cdef struct _windowSumState:
    Py_ssize_t blockStart
    Py_ssize_t blockEnd
    Py_ssize_t prefixRight
    Py_ssize_t prefixCount
    Py_ssize_t count
    _doubleDouble prefixSum


cdef inline _doubleDouble _blockedWindowSum(
    const real_t* values,
    const uint8_t* excluded,
    Py_ssize_t size,
    Py_ssize_t width,
    Py_ssize_t left,
    Py_ssize_t right,
    _windowSumState* state,
    _doubleDouble* suffixSums,
    Py_ssize_t* suffixCounts,
) noexcept nogil:
    r"""Perform a 'rolling' window sum without subtracting large outgoing values, totals, etc.,
    as subtraction can reduce precision of small values and has previously made several downstream
    operations degenerate (e.g., munc trend) in special cases.

    - Uses local _blocked_ cumulative sums as below, avoiding subtraction but still giving window total:

      ```
      blocks: | a b c d | e f g h |
      window:      [c d | e f]
      total:       (c+d) + (e+f)
                    suffix  prefix
      ```

      as opposed to `total = total - outgoing + incoming`
      (e.g., `total = total - c + g` for `[c d | e f] -> [d | e f g]`).
    """
    cdef Py_ssize_t j, count = 0
    cdef _doubleDouble total = _ddValue(0.0)

    # build [0, right) directly until left advances, adding each included value once.
    if left == 0 and right <= width and state.blockEnd == 0:
        while state.prefixRight < right:
            j = state.prefixRight
            if excluded == NULL or excluded[j] == 0:
                # accumulate
                state.prefixSum = _ddAdd(state.prefixSum, _ddValue(<double>values[j]))
                state.prefixCount += 1
            state.prefixRight += 1
        state.count = state.prefixCount
        return state.prefixSum

    # accumulate in double-double precision to retain small contributions.
    if left >= state.blockEnd:
        # block | a b c d | -> cache suffixes [a+b+c+d, b+c+d, c+d, d]
        state.blockStart = (left // width) * width
        state.blockEnd = min(state.blockStart + width, size)
        # scan right --> left to extend suffix sum over [j, blockEnd)
        for j in range(state.blockEnd - 1, state.blockStart - 1, -1):
            if excluded == NULL or excluded[j] == 0:
                total = _ddAdd(total, _ddValue(<double>values[j]))
                count += 1
            suffixSums[j - state.blockStart] = total
            if excluded != NULL:
                suffixCounts[j - state.blockStart] = count
        state.prefixSum = _ddValue(0.0)
        state.prefixCount = 0
        state.prefixRight = state.blockEnd
    while state.prefixRight < right:
        j = state.prefixRight
        if excluded == NULL or excluded[j] == 0:
            state.prefixSum = _ddAdd(state.prefixSum, _ddValue(<double>values[j]))
            state.prefixCount += 1
        state.prefixRight += 1
    # combine the cached [left, blockEnd) suffix and growing [blockEnd, right) prefix
    # to get the count and total for included values in [left, right).
    state.count = (right - left if excluded == NULL else
                   suffixCounts[left - state.blockStart] + state.prefixCount)
    return _ddAdd(suffixSums[left - state.blockStart], state.prefixSum)


cdef void _windowSums(
    const double* values,
    double* out,
    Py_ssize_t size,
    Py_ssize_t width,
    Py_ssize_t leftOffset,
    bint shiftEdges,
    _doubleDouble* suffixSums,
) noexcept nogil:
    cdef Py_ssize_t i, left, right
    cdef _doubleDouble total
    cdef _windowSumState state = _windowSumState(0, 0, 0, 0, 0, _ddValue(0.0))
    if width == 1:
        for i in range(size):
            out[i] = values[i]
        return
    for i in range(size):
        left = i - leftOffset
        if shiftEdges:
            left = max(0, min(left, size - width))
            right = left + width
        else:
            right = min(size, left + width)
            left = max(0, left)
        total = _blockedWindowSum(
            values, NULL, size, width, left, right, &state, suffixSums, NULL,
        )
        out[i] = total.high + total.low


def cWindowSums(
    object values,
    Py_ssize_t windowIntervals,
    bint shiftEdges=False,
    Py_ssize_t leftOffset=-1,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] valuesArr = np.ascontiguousarray(values, dtype=np.float64)
    cdef Py_ssize_t size = valuesArr.shape[0]
    cdef Py_ssize_t width
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(size, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] suffixArr
    if windowIntervals < 1:
        raise ValueError("windowIntervals must be positive")
    if size == 0:
        return out
    width = min(windowIntervals, size)
    if leftOffset == -1:
        leftOffset = width // 2 if shiftEdges else (width - 1) // 2
    if leftOffset < 0 or leftOffset >= width:
        raise ValueError("leftOffset must be nonnegative and less than the window width")
    suffixArr = np.empty((0 if shiftEdges and width == size else width, 2), dtype=np.float64)
    with nogil:
        _windowSums(
            <double*>valuesArr.data, <double*>out.data, size, width,
            leftOffset, shiftEdges, <_doubleDouble*>suffixArr.data,
        )
    return out


cdef inline _doubleDouble _rangeSum(
    const double* values,
    Py_ssize_t start,
    Py_ssize_t end,
    double* maximum,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef _doubleDouble total = _ddValue(0.0)
    for i in range(start, end):
        total = _ddAdd(total, _ddValue(values[i]))
        if maximum != NULL and values[i] > maximum[0]:
            maximum[0] = values[i]
    return total


def cRangeSums(object values, object starts, object ends):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] valuesArr = np.ascontiguousarray(values, dtype=np.float64)
    cdef object startsObj = np.asarray(starts)
    cdef object endsObj = np.asarray(ends)
    if startsObj.size and not np.issubdtype(startsObj.dtype, np.integer):
        raise ValueError("starts must contain integers")
    if endsObj.size and not np.issubdtype(endsObj.dtype, np.integer):
        raise ValueError("ends must contain integers")
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] startsArr = np.ascontiguousarray(startsObj, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] endsArr = np.ascontiguousarray(endsObj, dtype=np.int64)
    cdef Py_ssize_t i, size = valuesArr.shape[0], count = startsArr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(count, dtype=np.float64)
    cdef _doubleDouble total
    if endsArr.shape[0] != count:
        raise ValueError("starts and ends must have the same length")
    for i in range(count):
        if startsArr[i] < 0 or endsArr[i] < startsArr[i] or endsArr[i] > size:
            raise ValueError("range indices must satisfy 0 <= start <= end <= size")
    with nogil:
        for i in range(count):
            total = _rangeSum(<double*>valuesArr.data, startsArr[i], endsArr[i], NULL)
            out[i] = total.high + total.low
    return out


def _weightedQuantiles(
    const double[::1] values,
    const double[::1] weights,
    const double[::1] quantiles,
    bint interpolate,
):
    cdef Py_ssize_t size = values.shape[0], count = quantiles.shape[0]
    cdef Py_ssize_t i, left, right, middle, index
    cdef int exponent
    cdef double maximum = 0.0, q, fraction
    cdef _doubleDouble total = _ddValue(0.0)
    cdef _doubleDouble target, delta, span
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] cumulative = np.empty((size, 2), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out = np.empty(count, dtype=np.float64)
    cdef _doubleDouble* cdf = <_doubleDouble*>cumulative.data
    if size == 0 or weights.shape[0] != size:
        raise ValueError("values and weights must have matching nonempty lengths")
    with nogil:
        for i in range(size):
            maximum = fmax(maximum, weights[i])
        frexp(maximum, &exponent)
        # Scale by a power of two to avoid overflow without rounding normal weights.
        # Retain low parts in CDF comparisons so large weights cannot erase tail mass.
        for i in range(size):
            total = _ddAdd(total, _ddValue(ldexp(weights[i], -exponent)))
            cdf[i] = total
        for i in range(count):
            q = quantiles[i]
            if isnan(q):
                out[i] = NAN if interpolate else values[size - 1]
                continue
            if q <= 0.0:
                out[i] = values[0]
                continue
            if q >= 1.0:
                out[i] = values[size - 1]
                continue
            target = _ddMultiply(total, _ddValue(q))
            left = 0
            right = size
            while left < right:
                middle = left + (right - left) // 2
                delta = _ddSubtract(cdf[middle], target)
                if delta.high < 0.0 or (interpolate and delta.high == 0.0 and delta.low == 0.0):
                    left = middle + 1
                else:
                    right = middle
            index = min(left, size - 1)
            if not interpolate or index == 0:
                out[i] = values[index]
            else:
                delta = _ddSubtract(target, cdf[index - 1])
                span = _ddSubtract(cdf[index], cdf[index - 1])
                fraction = (delta.high + delta.low) / (span.high + span.low)
                out[i] = (1.0 - fraction) * values[index - 1] + fraction * values[index]
    return out


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
    cdef Py_ssize_t i, column, positiveCount = 0
    cdef Py_ssize_t firstBadPivot = -1
    cdef bint invalidConstraint = False
    cdef bint invalidRange = False
    cdef double scale = 0.0
    cdef _doubleDouble scaleDD, firstDD, secondDD, offVal, pivot, value
    cdef _doubleDouble mu = _ddValue(0.0)
    cdef _doubleDouble sumRhs = _ddValue(0.0)
    cdef _doubleDouble sumConstraint = _ddValue(0.0)
    cdef _doubleDouble one = _ddValue(1.0)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] factors
    cdef cnp.ndarray[cnp.float64_t, ndim=3] solutions
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    cdef _doubleDouble *inverseDiag
    cdef _doubleDouble *firstLower
    cdef _doubleDouble *secondLower
    cdef _doubleDouble *rhs
    cdef _doubleDouble *constraint

    if rhsTrack.shape[0] != n:
        raise ValueError("weightTrack and rhsTrack must have the same length")
    if not isfinite(lamFirst) or lamFirst < 0.0:
        raise ValueError("lamFirst must be finite and nonnegative")
    if not isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be finite and nonnegative")

    out = np.zeros(n, dtype=np.float64)
    if n <= 0:
        return out
    for i in range(n):
        if not isfinite(weightTrack[i]) or weightTrack[i] < 0.0:
            raise ValueError("weightTrack must be finite and nonnegative")
        if not isfinite(rhsTrack[i]):
            raise ValueError("rhsTrack must be finite")
        positiveCount += weightTrack[i] > 0.0
        scale = fmax(scale, weightTrack[i])
    if n == 1:
        if not zeroCenter:
            if positiveCount == 0:
                raise RuntimeError("roughness-penalized system is rank deficient")
            out[0] = rhsTrack[0] / weightTrack[0]
            if not isfinite(out[0]):
                raise FloatingPointError("background solution exceeds float64 exponent range")
        return out
    if n < 3:
        lam = 0.0
    if (
        (lamFirst > 0.0 and positiveCount == 0)
        or (lamFirst == 0.0 and lam > 0.0 and positiveCount < 2)
        or (lamFirst == 0.0 and lam == 0.0 and positiveCount < n)
    ):
        raise RuntimeError("roughness-penalized system is rank deficient")

    scaleDD = _ddValue(fmax(scale, fmax(lamFirst, lam)))
    firstDD = _ddDivide(_ddValue(lamFirst), scaleDD)
    secondDD = _ddDivide(_ddValue(lam), scaleDD)
    factors = np.zeros((3, n, 2), dtype=np.float64)
    solutions = np.zeros((1 + <int>zeroCenter, n, 2), dtype=np.float64)
    inverseDiag = <_doubleDouble *>factors.data
    firstLower = inverseDiag + n
    secondLower = firstLower + n
    constraint = <_doubleDouble *>solutions.data + n

    with nogil:
        for i in range(n):
            pivot = _ddAdd(
                _ddDivide(_ddValue(weightTrack[i]), scaleDD),
                _ddAdd(
                    _ddMultiply(firstDD, _ddValue(_firstDiffPenaltyDiag(n, i, 1.0))),
                    _ddMultiply(secondDD, _ddValue(_secondDiffPenaltyDiag(n, i, 1.0))),
                ),
            )
            if i > 0:
                offVal = _ddAdd(
                    _ddMultiply(firstDD, _ddValue(_firstDiffPenaltyOff1(n, 1.0))),
                    _ddMultiply(secondDD, _ddValue(_secondDiffPenaltyOff1(n, i - 1, 1.0))),
                )
                if i > 1:
                    secondLower[i] = _ddMultiply(secondDD, inverseDiag[i - 2])
                    offVal = _ddSubtract(offVal, _ddMultiply(secondDD, firstLower[i - 1]))
                    pivot = _ddSubtract(pivot, _ddMultiply(secondDD, secondLower[i]))
                firstLower[i] = _ddMultiply(offVal, inverseDiag[i - 1])
                pivot = _ddSubtract(pivot, _ddMultiply(offVal, firstLower[i]))
            if not isfinite(pivot.high) or pivot.high <= 0.0:
                firstBadPivot = i
                break
            inverseDiag[i] = _ddDivide(one, pivot)
            if not isfinite(inverseDiag[i].high):
                firstBadPivot = i
                invalidRange = True
                break

        if firstBadPivot < 0:
            for column in range(1 + <int>zeroCenter):
                rhs = <_doubleDouble *>solutions.data + column * n
                for i in range(n):
                    value = _ddDivide(_ddValue(rhsTrack[i]), scaleDD) if column == 0 else one
                    if i > 0:
                        value = _ddSubtract(value, _ddMultiply(firstLower[i], rhs[i - 1]))
                    if i > 1:
                        value = _ddSubtract(value, _ddMultiply(secondLower[i], rhs[i - 2]))
                    rhs[i] = value
                for i in range(n):
                    rhs[i] = _ddMultiply(rhs[i], inverseDiag[i])
                for i in range(n - 2, -1, -1):
                    rhs[i] = _ddSubtract(rhs[i], _ddMultiply(firstLower[i + 1], rhs[i + 1]))
                    if i < n - 2:
                        rhs[i] = _ddSubtract(rhs[i], _ddMultiply(secondLower[i + 2], rhs[i + 2]))

            rhs = <_doubleDouble *>solutions.data
            if zeroCenter:
                for i in range(n):
                    sumRhs = _ddAdd(sumRhs, rhs[i])
                    sumConstraint = _ddAdd(sumConstraint, constraint[i])
                invalidConstraint = not isfinite(sumConstraint.high) or sumConstraint.high <= 0.0
                if not invalidConstraint:
                    mu = _ddDivide(sumRhs, sumConstraint)
            if not invalidConstraint:
                for i in range(n):
                    value = _ddSubtract(rhs[i], _ddMultiply(mu, constraint[i])) if zeroCenter else rhs[i]
                    out[i] = value.high + value.low
                    if not isfinite(out[i]):
                        invalidRange = True
                        break

    if invalidRange:
        raise FloatingPointError("compensated LDL arithmetic exceeds float64 exponent range")
    if firstBadPivot >= 0:
        raise RuntimeError(
            "compensated LDL factorization has a nonpositive or nonfinite "
            f"pivot at index {firstBadPivot}"
        )
    if invalidConstraint:
        raise RuntimeError("compensated LDL constraint solve has invalid precision")
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


cdef inline void _nthElementThreeWay(real_t* vals, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = n - 1
    cdef Py_ssize_t lt
    cdef Py_ssize_t i
    cdef Py_ssize_t gt
    cdef real_t pivot

    if n <= 1:
        return
    while left < right:
        pivot = vals[(left + right) >> 1]
        lt = left
        i = left
        gt = right
        while i <= gt:
            if vals[i] < pivot:
                _swapReal(vals, lt, i)
                lt += 1
                i += 1
            elif vals[i] > pivot:
                _swapReal(vals, i, gt)
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


cdef inline double _linearQuantileInplaceReal(real_t* values, Py_ssize_t n, double q) noexcept nogil:
    cdef double pos
    cdef double frac
    cdef double lowVal
    cdef double highVal
    cdef Py_ssize_t lowIndex
    cdef Py_ssize_t highIndex
    cdef Py_ssize_t index

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
    _nthElementThreeWay(values, n, lowIndex)
    lowVal = values[lowIndex]
    if highIndex == lowIndex or frac == 0.0:
        return lowVal
    highVal = values[highIndex]
    for index in range(highIndex + 1, n):
        if values[index] < highVal:
            highVal = values[index]
    return lowVal + frac * (highVal - lowVal)


cdef void _initializeProcessPrecisionFromPointwiseChanges(
    cnp.float32_t[:, ::1] data,
    cnp.float32_t[:, ::1] munc,
    cnp.int32_t[::1] intervalToBlock,
    Py_ssize_t blockCount,
    cnp.float32_t[:, ::1] matrixQ0,
    double pad,
    double robustTNu,
    cnp.float32_t[::1] processPrecision,
) except *:
    cdef Py_ssize_t trackCount = data.shape[0]
    cdef Py_ssize_t intervalCount = data.shape[1]
    cdef Py_ssize_t trackIndex
    cdef Py_ssize_t intervalIndex
    cdef Py_ssize_t blockIndex
    cdef double leftVariance
    cdef double rightVariance
    cdef double leftWeight
    cdef double rightWeight
    cdef double leftWeightSum
    cdef double rightWeightSum
    cdef double leftWeightedLevel
    cdef double rightWeightedLevel
    cdef double leftPointwiseLevel
    cdef double rightPointwiseLevel
    cdef double pointwiseChange
    cdef double samplingVariance
    cdef double transitionStatistic
    cdef double qLevel

    if munc.shape[0] != trackCount or munc.shape[1] != intervalCount:
        raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
    if processPrecision.shape[0] != intervalCount:
        raise ValueError("process precision length must match intervalCount")
    if intervalCount <= 0:
        return
    processPrecision[0] = <cnp.float32_t>1.0
    if trackCount <= 0:
        return
    if intervalToBlock.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")
    if blockCount <= 0:
        raise ValueError("blockCount must be positive")
    if matrixQ0.shape[0] < 1 or matrixQ0.shape[1] < 1 or matrixQ0[0, 0] <= 0.0:
        raise ValueError("matrixQ0[0, 0] must be positive")
    qLevel = <double>matrixQ0[0, 0]

    with nogil:
        for intervalIndex in range(intervalCount - 1):
            blockIndex = <Py_ssize_t>intervalToBlock[intervalIndex]
            if blockIndex < 0 or blockIndex >= blockCount:
                processPrecision[intervalIndex + 1] = <cnp.float32_t>1.0
                continue
            leftWeightSum = 0.0
            rightWeightSum = 0.0
            leftWeightedLevel = 0.0
            rightWeightedLevel = 0.0
            for trackIndex in range(trackCount):
                if (
                    (<double>munc[trackIndex, intervalIndex])
                    < __MASKED_OBSERVATION_VARIANCE_CUTOFF
                ):
                    leftVariance = (
                        (<double>munc[trackIndex, intervalIndex]) + pad
                    )
                    if leftVariance < 1.0e-12:
                        leftVariance = 1.0e-12
                    leftWeight = 1.0 / leftVariance
                    leftWeightSum += leftWeight
                    leftWeightedLevel += (
                        leftWeight * <double>data[trackIndex, intervalIndex]
                    )
                if (
                    (<double>munc[trackIndex, intervalIndex + 1])
                    < __MASKED_OBSERVATION_VARIANCE_CUTOFF
                ):
                    rightVariance = (
                        (<double>munc[trackIndex, intervalIndex + 1]) + pad
                    )
                    if rightVariance < 1.0e-12:
                        rightVariance = 1.0e-12
                    rightWeight = 1.0 / rightVariance
                    rightWeightSum += rightWeight
                    rightWeightedLevel += (
                        rightWeight * <double>data[trackIndex, intervalIndex + 1]
                    )
            if leftWeightSum <= 0.0 or rightWeightSum <= 0.0:
                processPrecision[intervalIndex + 1] = <cnp.float32_t>1.0
                continue
            leftPointwiseLevel = leftWeightedLevel / leftWeightSum
            rightPointwiseLevel = rightWeightedLevel / rightWeightSum
            pointwiseChange = rightPointwiseLevel - leftPointwiseLevel
            samplingVariance = (1.0 / leftWeightSum) + (1.0 / rightWeightSum)
            transitionStatistic = (
                pointwiseChange * pointwiseChange - samplingVariance
            )
            if transitionStatistic < 0.0:
                transitionStatistic = 0.0
            transitionStatistic /= qLevel
            processPrecision[intervalIndex + 1] = <cnp.float32_t>(
                (robustTNu + 1.0)
                / (robustTNu + transitionStatistic)
            )


cdef void _scalePrecisionMultipliersToMedian(
    cnp.float32_t[::1] multipliers,
    Py_ssize_t start,
    Py_ssize_t stop,
    double lower,
    double upper,
    cnp.float32_t[::1] workspace,
) except *:
    cdef Py_ssize_t count = stop - start
    cdef Py_ssize_t index
    cdef Py_ssize_t invalidIndex = -1
    cdef double value
    cdef double median = 1.0

    if count <= 0:
        return
    with nogil:
        for index in range(count):
            value = <double>multipliers[start + index]
            if not isfinite(value) or value <= 0.0:
                invalidIndex = start + index
                break
            workspace[index] = multipliers[start + index]
        if invalidIndex < 0:
            median = _linearQuantileInplaceReal(&workspace[0], count, 0.5)
    if invalidIndex >= 0:
        raise ValueError(
            f"precision multiplier at index {invalidIndex} must be positive and finite"
        )
    if not isfinite(median) or median <= 0.0:
        raise ValueError("precision multiplier median must be positive and finite")
    with nogil:
        for index in range(start, stop):
            multipliers[index] = <cnp.float32_t>_clampMultiplierValue(
                (<double>multipliers[index]) / median,
                lower,
                upper,
            )


cdef void _validateMedianScaledPrecisionWarmStart(
    cnp.float32_t[::1] multipliers,
    Py_ssize_t start,
    Py_ssize_t stop,
    double lower,
    double upper,
) except *:
    cdef Py_ssize_t index
    cdef Py_ssize_t invalidIndex = -1
    cdef Py_ssize_t boundIndex = -1
    cdef double value
    cdef double effectiveLower = <double><cnp.float32_t>lower
    cdef double effectiveUpper = <double><cnp.float32_t>upper

    with nogil:
        for index in range(start, stop):
            value = <double>multipliers[index]
            if not isfinite(value) or value <= 0.0:
                invalidIndex = index
                break
            if value < effectiveLower or value > effectiveUpper:
                boundIndex = index
                break
    if invalidIndex >= 0:
        raise ValueError(
            f"precision multiplier at index {invalidIndex} must be positive and finite"
        )
    if boundIndex >= 0:
        raise ValueError(
            f"median-scaled precision multiplier at index {boundIndex} is outside its bounds"
        )


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
    cdef double result
    if n <= 0:
        return NAN
    buf = <double*>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError()
    memcpy(buf, values, n * sizeof(double))
    result = _linearQuantileInplaceReal(buf, n, q)
    free(buf)
    return result


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
    cdef double rd
    cdef double rawPrecision
    cdef double precision
    cdef double medianPrecision
    cdef double qPrecision
    cdef double cap = NAN
    cdef double capFraction = 0.0
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
                    rawPrecision = 1.0 / (
                        obsView[j, k] + obsView[j, k + 1]
                    )
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

    localArr = np.ascontiguousarray(localModelVariances, dtype=np.float64).reshape(-1)
    globalArr = np.ascontiguousarray(globalModelVariances, dtype=np.float64).reshape(-1)
    if localArr.shape[0] != globalArr.shape[0]:
        raise ValueError("localModelVariances and globalModelVariances must have the same shape")
    n = localArr.shape[0]
    localView = localArr
    globalView = globalArr
    if useNoise:
        noiseArr = np.ascontiguousarray(localLogVarianceNoise, dtype=np.float64).reshape(-1)
        if noiseArr.shape[0] != n:
            raise ValueError("localLogVarianceNoise must match localModelVariances")
        noiseView = noiseArr
    if useMask:
        maskArr = np.ascontiguousarray(candidateMask, dtype=np.uint8).reshape(-1)
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

    localArr = np.ascontiguousarray(localModelVariances, dtype=np.float64).reshape(-1)
    globalArr = np.ascontiguousarray(globalModelVariances, dtype=np.float64).reshape(-1)
    if localArr.shape[0] != globalArr.shape[0]:
        raise ValueError("localModelVariances and globalModelVariances must have the same shape")
    n = localArr.shape[0]
    localView = localArr
    globalView = globalArr
    if useNoise:
        noiseArr = np.ascontiguousarray(localLogVarianceNoise, dtype=np.float64).reshape(-1)
        if noiseArr.shape[0] != n:
            raise ValueError("localLogVarianceNoise must match localModelVariances")
        noiseView = noiseArr
    if usePooledThin:
        sampleArr = np.ascontiguousarray(sampleIndex, dtype=np.int64).reshape(-1)
        chromosomeArr = np.ascontiguousarray(chromosomeIndex, dtype=np.int64).reshape(-1)
        startArr = np.ascontiguousarray(blockStarts, dtype=np.int64).reshape(-1)
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

    localArr = np.ascontiguousarray(localModelVariances, dtype=np.float64).reshape(-1)
    globalArr = np.ascontiguousarray(globalModelVariances, dtype=np.float64).reshape(-1)
    if localArr.shape[0] != globalArr.shape[0]:
        raise ValueError("localModelVariances and globalModelVariances must have the same shape")
    n = localArr.shape[0]
    localView = localArr
    globalView = globalArr
    if useNoise:
        noiseArr = np.ascontiguousarray(localLogVarianceNoise, dtype=np.float64).reshape(-1)
        if noiseArr.shape[0] != n:
            raise ValueError("localLogVarianceNoise must match localModelVariances")
        noiseView = noiseArr
    candidateArr = np.ascontiguousarray(candidateIdx, dtype=np.intp).reshape(-1)
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


cdef int _integer(object value, str name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not -2147483648 <= value <= 2147483647:
        raise ValueError(f"{name} is outside the supported integer range")
    return int(value)


cdef double _real(object value, str name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real scalar")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


cdef int _oddBins(int bp, int binBP):
    cdef double target = bp / <double>binBP
    cdef int lower = max(1, int(target))
    lower -= 1 - lower % 2
    lower = max(1, lower)
    return lower + 2 if fabs(lower + 2 - target) <= fabs(lower - target) else lower


cdef double _select(double[::1] values, int count, int rank) noexcept nogil:
    cdef int left = 0, right = count - 1, lower, upper, index
    cdef double pivot, temporary
    while left < right:
        pivot = values[(left + right) // 2]
        lower, upper, index = left, right, left
        while index <= upper:
            if values[index] < pivot:
                temporary = values[lower]
                values[lower] = values[index]
                values[index] = temporary
                lower += 1
                index += 1
            elif values[index] > pivot:
                temporary = values[upper]
                values[upper] = values[index]
                values[index] = temporary
                upper -= 1
            else:
                index += 1
        if rank < lower:
            right = lower - 1
        elif rank > upper:
            left = upper + 1
        else:
            return pivot
    return values[left]


cdef object _replicateMean(object matrix, list retained):
    cdef Py_ssize_t column, length = matrix.shape[1]
    cdef int row, minimum = (len(retained) + 1) // 2
    result = np.zeros((1, length), dtype=np.float64)
    countArray = np.zeros(length, dtype=np.int32)
    cdef double[:, ::1] mean = result
    cdef int[::1] counts = countArray
    cdef const double[::1] values
    for row in retained:
        values = np.ascontiguousarray(matrix[row], dtype=np.float64)
        with nogil:
            for column in range(length):
                if isfinite(values[column]):
                    mean[0, column] += values[column]
                    counts[column] += 1
    with nogil:
        for column in range(length):
            if counts[column] >= minimum:
                mean[0, column] /= counts[column]
            else:
                mean[0, column] = NAN
    return result


cpdef dict cclippedCovarianceReplicates(object matrix, int maxLagBins,
                                      int minFinitePairs=200, double minFinitePairCoverage=0.5,
                                      object referenceMean=None):
    cdef const double[:, ::1] values = np.ascontiguousarray(matrix, dtype=np.float64)
    cdef Py_ssize_t rowCount = values.shape[0], binCount = values.shape[1]
    cdef int group, row, index, lag, left, right, count, contributorCount, stop, rank
    cdef double low, high, value, mean, meanCorrection, pairCount, coverage, total, location, delta, populationTotal
    cdef bint restoreMean = referenceMean is not None
    if rowCount < 1 or binCount < 2 or not 0 <= maxLagBins < binCount:
        raise ValueError("Invalid covariance dimensions or lag")
    if minFinitePairs < 1 or not 0 < minFinitePairCoverage <= 1:
        raise ValueError("Invalid finite-pair requirements")
    centeredArray = np.zeros((9, rowCount, binCount), dtype=np.float64)
    availableArray = np.zeros((9, rowCount, binCount), dtype=np.float64)
    meanArray = np.zeros((9, rowCount), dtype=np.float64)
    referenceArray = np.zeros(rowCount) if not restoreMean else np.ascontiguousarray(referenceMean, dtype=np.float64)
    if referenceArray.shape != (rowCount,) or not np.all(np.isfinite(referenceArray)):
        raise ValueError("referenceMean must contain one finite value per replicate")
    cdef double[:, ::1] means = meanArray
    cdef double[:, ::1] meanDeltas = np.zeros((9, rowCount), dtype=np.float64)
    cdef const double[::1] reference = referenceArray
    cdef double[:, :, ::1] centered = centeredArray, available = availableArray
    retainedArray = np.empty(binCount, dtype=np.float64)
    cdef double[::1] retained = retainedArray
    for group in range(9):
        left = (group - 1) * binCount // 8 if group else binCount
        right = group * binCount // 8 if group else binCount
        for row in range(rowCount):
            count = 0
            with nogil:
                for index in range(binCount):
                    value = values[row, index]
                    if not left <= index < right and isfinite(value):
                        retained[count] = value
                        count += 1
            if count < 2:
                continue
            mean = 0.0
            meanCorrection = 0.0
            with nogil:
                location = 0.005 * (count - 1)
                rank = <int>location
                low = _select(retained, count, rank)
                low += (_select(retained, count, rank + 1) - low) * (location - rank)
                location = 0.995 * (count - 1)
                rank = <int>location
                high = _select(retained, count, rank)
                high += (_select(retained, count, rank + 1) - high) * (location - rank)
                for index in range(binCount):
                    value = values[row, index]
                    if not left <= index < right and isfinite(value):
                        value = min(high, max(low, value))
                        centered[group, row, index] = value - low
                        available[group, row, index] = 1.0
                        compensatedSumAdd(value - low, &mean, &meanCorrection)
                mean = (mean + meanCorrection) / count
                means[group, row] = low + mean
                if restoreMean:
                    meanDeltas[group, row] = (low - reference[row]) + mean
                for index in range(binCount):
                    if available[group, row, index] > 0:
                        centered[group, row, index] -= mean
    cdef Py_ssize_t fftSize = next_fast_len(2 * binCount - 1)
    spectrum = rfft(centeredArray, n=fftSize, axis=2)
    cdef bint commonMask = bool(np.all(availableArray == availableArray[:, :1]))
    maskSpectrum = rfft(availableArray[:, :1] if commonMask else availableArray, n=fftSize, axis=2)
    sumArray = irfft(spectrum * spectrum.conj(), n=fftSize, axis=2)[:, :, :maxLagBins + 1].copy()
    pairArray = np.rint(irfft(maskSpectrum * maskSpectrum.conj(), n=fftSize, axis=2)[:, :, :maxLagBins + 1]).copy()
    endpointArray = (irfft(spectrum * maskSpectrum.conj(), n=fftSize, axis=2)
                     if restoreMean else np.empty((1, 1, 1), dtype=np.float64))
    curveArray = np.zeros((9, maxLagBins + 1), dtype=np.float64)
    populationArray = np.zeros_like(curveArray)
    minimumPairs = np.full(maxLagBins + 1, np.inf)
    minimumCoverage = np.ones(maxLagBins + 1)
    contributorArray = np.zeros(maxLagBins + 1, dtype=np.int32)
    cdef double[:, :, ::1] sums = sumArray
    cdef const double[:, :, ::1] pairs = pairArray
    cdef double[:, ::1] curves = curveArray
    cdef double[:, ::1] population = populationArray
    cdef double[:, :, ::1] endpoints = endpointArray
    cdef double[::1] pairMinimum = minimumPairs, coverageMinimum = minimumCoverage
    cdef int[::1] contributors = contributorArray
    cdef int quorum, validRowCount = 0
    usableRows = np.zeros(rowCount, dtype=np.uint8)
    cdef cnp.uint8_t[::1] usable = usableRows
    for row in range(rowCount):
        pairCount = pairs[0, 0 if commonMask else row, 0]
        if pairCount >= minFinitePairs and pairCount / binCount >= minFinitePairCoverage and sums[0, row, 0] > 0:
            usable[row] = 1
            validRowCount += 1
    quorum = max(1, (validRowCount + 1) // 2)
    stop = maxLagBins + 1
    with nogil:
        for lag in range(maxLagBins + 1):
            for group in range(9):
                total = 0.0
                populationTotal = 0.0
                contributorCount = 0
                for row in range(rowCount):
                    pairCount = pairs[group, 0 if commonMask else row, lag]
                    coverage = pairCount / (binCount - lag)
                    if usable[row] and pairCount >= minFinitePairs and coverage >= minFinitePairCoverage and sums[group, row, 0] > 0:
                        total += sums[group, row, lag] / pairCount
                        if restoreMean:
                            delta = meanDeltas[group, row]
                            value = endpoints[group, row, lag] + endpoints[group, row, fftSize - lag if lag else 0]
                            populationTotal += (sums[group, row, lag] + delta * value) / pairCount + delta * delta
                        contributorCount += 1
                        if group == 0:
                            pairMinimum[lag] = min(pairMinimum[lag], pairCount)
                            coverageMinimum[lag] = min(coverageMinimum[lag], coverage)
                if contributorCount < quorum:
                    stop = min(stop, lag)
                else:
                    curves[group, lag] = total / contributorCount
                    if restoreMean:
                        population[group, lag] = populationTotal / contributorCount
                if group == 0:
                    contributors[lag] = contributorCount
                    if lag == 0:
                        validRowCount = contributorCount
            if stop <= lag:
                break
    result = dict(curves=curveArray[:, :stop].copy(), pairs=minimumPairs[:stop],
                  coverage=minimumCoverage[:stop], contributors=contributorArray[:stop],
                  validRowCount=validRowCount, means=meanArray)
    if restoreMean:
        result["populationCurves"] = populationArray[:, :stop].copy()
    return result


cdef object _referenceMeans(object matrix, list retainedRows, int width):
    cdef int row, index, start, block, count, rank
    cdef double low, high, location, mean
    cdef double[::1] values
    cdef const double[::1] rawValues
    output = np.zeros(len(retainedRows), dtype=np.float64)
    for row, source in enumerate(retainedRows):
        raw = np.ascontiguousarray(matrix[source], dtype=np.float64)
        rawValues = raw
        values = np.empty(raw.size, dtype=np.float64)
        count = 0
        with nogil:
            for block in range(rawValues.shape[0] // width):
                start = block * width
                low, high = INFINITY, -INFINITY
                for index in range(start, start + width):
                    if isfinite(rawValues[index]):
                        low = min(low, rawValues[index])
                        high = max(high, rawValues[index])
                if high > low:
                    for index in range(start, start + width):
                        if isfinite(rawValues[index]):
                            values[count] = rawValues[index]
                            count += 1
        if count == 0:
            continue
        with nogil:
            location = 0.005 * (count - 1)
            rank = <int>location
            low = _select(values, count, rank)
            if rank + 1 < count:
                low += (_select(values, count, rank + 1) - low) * (location - rank)
            location = 0.995 * (count - 1)
            rank = <int>location
            high = _select(values, count, rank)
            if rank + 1 < count:
                high += (_select(values, count, rank + 1) - high) * (location - rank)
            mean = 0
            for index in range(count):
                mean += min(high, max(low, values[index])) - low
            mean = low + mean / count
        output[row] = mean
    return output


cdef list _shrink(list records):
    cdef Py_ssize_t count = len(records), index, length
    cdef double meanNoise, priorVariance, localVariance, heterogeneity, alpha, value
    if count < 5:
        raise RuntimeError("Shrinkage requires four eligible donors on the chromosome")
    length = min(record["stats"]["curves"].shape[1] for record in records)
    tensor = np.ascontiguousarray([record["stats"]["curves"][:, :length] for record in records])
    deviations = tensor - tensor[:1]
    meanDelta = deviations.mean(axis=0)
    centered = deviations - meanDelta
    m2 = np.sum(centered * centered, axis=0)
    jack = tensor[:, 1:, :] - tensor[:, 1:2, :]
    jack -= jack.mean(axis=1, keepdims=True)
    variance = 7.0 / 8.0 * np.sum(jack * jack, axis=1).mean(axis=1)
    noiseSum = float(np.sum(variance))
    output = []
    for index in range(count):
        localVariance = variance[index]
        meanNoise = max(0.0, (noiseSum - localVariance) / (count - 1))
        priorVariance = meanNoise / (count - 1)
        donorMean = tensor[0, 0] + (count * meanDelta[0] - deviations[index, 0]) / (count - 1)
        donorM2 = np.maximum(0.0, m2[0] - count / <double>(count - 1) * centered[index, 0] ** 2)
        heterogeneity = max(0.0, float(np.mean(donorM2 / (count - 2))) - meanNoise)
        value = localVariance + heterogeneity + priorVariance
        alpha = localVariance / value if value > 0 else 0.0
        curve = (1 - alpha) * tensor[index, 0] + alpha * donorMean
        output.append(dict(curve=curve, borrowingWeight=alpha, donorCount=count - 1,
                           localVariance=localVariance, residualVariance=heterogeneity,
                           priorVariance=priorVariance, poolDominated=bool(alpha >= 0.8)))
    return output


cdef tuple _period(double[::1] acf, int binBP, int crossingLag, int persistence):
    cdef int size = acf.shape[0], width, frequencyIndex, start
    cdef double total
    period = strength = revival = None
    if size >= 3 and max(2 * binBP, 150) <= 500:
        width = _oddBins(2000, binBP)
        width = min(width, size if size % 2 else size - 1)
        if width >= 3:
            frequencies = np.fft.rfftfreq(size, d=binBP)
            indices = np.flatnonzero((frequencies > 0) & (frequencies >= 1.0 / 500)
                                     & (frequencies <= 1.0 / max(2 * binBP, 150))
                                     & (frequencies * size * binBP >= 4))
            if indices.size:
                residual = np.asarray(acf) - signal.savgol_filter(np.asarray(acf), width, 2, mode="interp")
                tapers = dependenceTaperCache.get(size)
                if tapers is None:
                    tapers = signal.windows.dpss(size, 2.5, Kmax=3, sym=False, norm=2)
                    dependenceTaperCache[size] = tapers
                power = np.mean(np.abs(rfft(tapers * residual, axis=1)) ** 2, axis=0)
                frequencyIndex = indices[np.argmax(power[indices])]
                total = np.sum(power[indices])
                if total > 0:
                    period = float(1.0 / frequencies[frequencyIndex])
                    strength = float(power[frequencyIndex] / total)
    if crossingLag > 0:
        start = crossingLag + persistence - 1
        if start < size:
            revival = float(np.max(np.abs(np.asarray(acf)[start:])))
    return period, strength, revival


cdef int _crossing(double[::1] covariance, double threshold, int smooth, int required) noexcept nogil:
    cdef int cap = covariance.shape[0] - 1, half = smooth // 2, lag, index, run = 0
    cdef double rolling = 0, scale = covariance[0]
    if scale <= 0 or cap < smooth + required - 1:
        return -1
    for index in range(1, smooth + 1):
        rolling += fabs(covariance[index]) / scale
    for lag in range(1 + half, cap - half + 1):
        if rolling / smooth < threshold:
            run += 1
            if run >= required:
                return lag - required + 1
        else:
            run = 0
        if lag < cap - half:
            rolling += (fabs(covariance[lag + half + 1]) - fabs(covariance[lag - half])) / scale
    return -1


cdef void _supportFailure(str message, dict diagnostics) except *:
    error = RuntimeError(
        f"{message}. Consider setting observationParams.dependenceSpanBP "
        "to a positive length in bp appropriate for the mark to bypass "
        "dependence estimation for background fitting, MUNC, and peak calling. "
        "Aim for a fitted background span comfortably wider than the mark's "
        "expected enrichment length scale. Account for "
        "fitParams.ECM_backgroundLengthScaleMultiplier and the "
        f"{constants.FIT_BACKGROUND_LENGTH_SCALE_CAP_BP:g} bp fitting-span cap, "
        "and check representative regions for loss of broad enrichment."
    )
    error.diagnostics = diagnostics
    raise error


cdef tuple _pooledRadius(double[:, ::1] numerator, double[:, ::1] denominator,
                         int[::1] indices, int binBP, double threshold, int smooth, int required):
    cdef int lag, index, row, count, stop = numerator.shape[1]
    cdef int quorum = max(1, (indices.shape[0] + 1) // 2), crossing
    cdef double total, weight, correction = 3.0 / (2.0 * sqrt(-log(threshold)))
    curveArray = np.zeros(stop, dtype=np.float64)
    cdef double[::1] curve = curveArray
    with nogil:
        for lag in range(stop):
            total, weight, count = 0, 0, 0
            for index in range(indices.shape[0]):
                row = indices[index]
                if denominator[row, lag] > 0:
                    total += numerator[row, lag]
                    weight += denominator[row, lag]
                    count += 1
            if count < quorum:
                stop = lag
                break
            curve[lag] = total / weight
        crossing = _crossing(curve[:stop], threshold, smooth, required) if stop else -1
    radius = (crossing if crossing > 0 else max(0, stop - 1 - smooth // 2 - required + 1)) * binBP * correction
    return float(radius), bool(crossing < 0), int(max(0, stop - 1) * binBP), curveArray[:stop]


cdef object _radius(dict record, dict shrunk, int binBP, double threshold,
                    int smooth, int persistence, int crossingCount, int maxLag):
    curve = np.ascontiguousarray(shrunk["curve"], dtype=np.float64)
    cdef double[::1] covariance = curve
    cdef int cap = curve.size - 1, half = smooth // 2, crossing
    cdef int required = persistence * crossingCount, last = cap - half - required + 1
    cdef double correction = 3.0 / (2.0 * sqrt(-log(threshold)))
    if last < 1 + half or covariance[0] <= 0:
        return None
    crossing = _crossing(covariance, threshold, smooth, required)
    correlation = curve[1:] / covariance[0]
    cdef double[::1] acf = np.ascontiguousarray(correlation)
    end = crossing + required - 1 + half if crossing > 0 else cap
    start = crossing - half if crossing > 0 else last - half
    period, strength, revival = _period(acf, binBP, crossing, required)
    stats = record["stats"]
    radius = (crossing if crossing > 0 else last) * binBP * correction
    return dict(rawCrossingLagBP=crossing * binBP if crossing > 0 else None,
                censorLagBP=last * binBP if crossing < 0 else None,
                gaussianEquivalentRadiusBP=radius, rightCensored=bool(crossing < 0),
                censorReason="none" if crossing > 0 else "maxLag" if cap >= maxLag else "support",
                supportCapLagBP=cap * binBP, finitePairMinimumUsed=float(np.min(stats["pairs"][1:end + 1])),
                finitePairCoverageMinimumUsed=float(np.min(stats["coverage"][1:end + 1])),
                validRowCount=stats["validRowCount"], validRowsAtCrossing=int(np.min(stats["contributors"][start:end + 1])),
                dominantACFPeriodBP=period, oscillationStrength=strength, postCrossingRevival=revival,
                crossingCount=crossingCount, localRadiusLowerBP=radius,
                localRadiusUpperBP=None if crossing < 0 else radius,
                **{key: value for key, value in shrunk.items() if key != "curve"})


cdef tuple _KMQuantiles(object values, object censored, double firstQuantile, double secondQuantile):
    order = np.argsort(values, kind="mergesort")
    cdef double[::1] times = np.ascontiguousarray(np.asarray(values)[order], dtype=np.float64)
    cdef cnp.uint8_t[::1] flags = np.ascontiguousarray(np.asarray(censored)[order], dtype=np.uint8)
    cdef Py_ssize_t n = times.shape[0], start = 0, stop, deaths
    cdef double probability = 1.0, time
    first = second = None
    while start < n:
        time = times[start]
        stop = start
        deaths = 0
        while stop < n and times[stop] == time:
            deaths += 1 - flags[stop]
            stop += 1
        if deaths:
            probability *= 1.0 - deaths / <double>(n - start)
            if first is None and probability <= 1.0 - firstQuantile + 1e-15:
                first = float(time)
            if second is None and probability <= 1.0 - secondQuantile + 1e-15:
                second = float(time)
            if first is not None and second is not None:
                break
        start = stop
    return first, second


cdef tuple _geometry(list windows, list chromosomes, object radii, object censored):
    groups = [sorted([index for index, window in enumerate(windows) if window["chromosome"] == chromosome],
                     key=lambda index: windows[index]["startBP"]) for chromosome in chromosomes]
    runs = []
    cdef int index, left, channel, lag, count = len(windows), adjacency = 0, longest = 1
    cdef int maxLag, kN = max(5, int(ceil(log10(count)))), cutoff, mHat, m, pairCount
    cdef double critical, weight, g, longVariance, denominator, block = 1, estimate
    for indices in groups:
        run = [indices[0]]
        for left, index in zip(indices, indices[1:]):
            if windows[index]["startBP"] == windows[left]["endBP"]:
                run.append(index)
                adjacency += 1
            else:
                runs.append(run)
                longest = max(longest, len(run))
                run = [index]
        runs.append(run)
        longest = max(longest, len(run))
    channels = np.ascontiguousarray([np.log(radii), 1.0 - np.asarray(censored, dtype=float),
                                    [window["score"] for window in windows]], dtype=np.float64)
    deviations = channels.std(axis=1)
    for channel in range(3):
        if deviations[channel] > 1e-12:
            channels[channel] = (channels[channel] - channels[channel].mean()) / deviations[channel]
        else:
            channels[channel] = 0
    cdef double[:, ::1] data = channels
    cdef int[::1] indicesView
    maxLag = min(longest - 1, int(ceil(sqrt(count))) + kN)
    acovArray = np.zeros((3, maxLag + 1), dtype=np.float64)
    cdef double[:, ::1] acov = acovArray
    for run in runs:
        indicesView = np.asarray(run, dtype=np.int32)
        with nogil:
            for lag in range(min(maxLag + 1, indicesView.shape[0])):
                for channel in range(3):
                    for index in range(indicesView.shape[0] - lag):
                        acov[channel, lag] += data[channel, indicesView[index]] * data[channel, indicesView[index + lag]]
    for lag in range(maxLag + 1):
        pairCount = sum(max(0, len(run) - lag) for run in runs)
        acovArray[:, lag] /= pairCount
    critical = 2 * sqrt(log10(count) / count)
    for channel in range(3):
        if acov[channel, 0] <= 0 or maxLag == 0:
            continue
        cutoff = min(kN, maxLag)
        mHat = 0
        for lag in range(cutoff, maxLag + 1):
            if np.all(np.abs(acovArray[channel, lag - cutoff + 1:lag + 1] / acov[channel, 0]) < critical):
                mHat = lag - cutoff + 1
                break
        if mHat == 0:
            selected = np.flatnonzero(np.abs(acovArray[channel, 1:] / acov[channel, 0]) > critical)
            mHat = int(selected[selected.size - 1]) + 1 if selected.size else 1
        m = min(maxLag, 2 * mHat)
        g = 0
        longVariance = acov[channel, 0]
        for lag in range(1, m + 1):
            weight = 1.0 if lag <= m / 2.0 else 2.0 * (1 - lag / <double>m)
            g += 2 * weight * lag * acov[channel, lag]
            longVariance += 2 * weight * acov[channel, lag]
        denominator = 2 * longVariance * longVariance
        if denominator > 0:
            estimate = pow(2 * g * g / denominator, 1.0 / 3) * pow(count, 1.0 / 3)
            block = max(block, min(estimate, 3 * sqrt(count), count / 3.0, longest))
    return max(1, min(longest, int(ceil(block)))), adjacency, longest, groups


cdef dict _inference(list windows, list chromosomes, object rng, int draws, double quantile,
                     list populationRecords, int binBP, double threshold, int smooth, int required):
    radii = np.asarray([window["gaussianEquivalentRadiusBP"] for window in windows], dtype=np.float64)
    censored = np.asarray([window["rightCensored"] for window in windows], dtype=np.uint8)
    median, localQuantileBP = _KMQuantiles(radii, censored, 0.5, quantile)
    cdef int length = max(record["stats"]["populationCurves"].shape[1] for record in populationRecords)
    numeratorArray = np.zeros((len(windows), length), dtype=np.float64)
    denominatorArray = np.zeros_like(numeratorArray)
    for recordIndex, record in enumerate(populationRecords):
        stats = record["stats"]
        stop = stats["populationCurves"].shape[1]
        denominatorArray[recordIndex, :stop] = stats["pairs"]
        numeratorArray[recordIndex, :stop] = stats["populationCurves"][0] * stats["pairs"]
    cdef double[:, ::1] numerator = numeratorArray, denominator = denominatorArray
    cdef int[::1] fullIndices = np.arange(len(windows), dtype=np.int32)
    pooledRadius, pooledCensored, pooledCap, pooledCurve = _pooledRadius(
        numerator, denominator, fullIndices, binBP, threshold, smooth, required)
    if pooledCensored:
        _supportFailure(f"Pooled covariance radius is right-censored above {pooledRadius:.6g} bp",
                        dict(estimateBP=None, radiusLowerBP=pooledRadius, radiusUpperBP=None,
                             rightCensored=True, supportCapLagBP=pooledCap, localMedianRadiusBP=median,
                             estimationTarget="pooledChromosomeCenteredCovariance"))
    block, adjacency, longest, groups = _geometry(windows, chromosomes, radii, censored)
    cdef int chromosomeCount = len(groups), maximum = max(len(group) for group in groups)
    groupArray = np.full((chromosomeCount, maximum), -1, dtype=np.int32)
    nextArray = np.full((chromosomeCount, maximum), -1, dtype=np.int32)
    lengthsArray = np.asarray([len(group) for group in groups], dtype=np.int32)
    cdef int[:, ::1] groupIndex = groupArray, nextIndex = nextArray
    cdef int[::1] lengths = lengthsArray
    cdef int chromosome, index, position, sampled, draw, size, nextPosition, drawPosition
    cdef double probability = 1.0 / block
    for chromosome in range(chromosomeCount):
        for position, index in enumerate(groups[chromosome]):
            groupIndex[chromosome, position] = index
            if position + 1 < lengths[chromosome]:
                nextPosition = groups[chromosome][position + 1]
                if windows[nextPosition]["startBP"] == windows[index]["endBP"]:
                    nextIndex[chromosome, position] = position + 1
    cdef int[::1] drawIndices = np.empty(chromosomeCount * maximum, dtype=np.int32)
    cdef double[:, ::1] uniforms
    pooledDraws, pooledFlags = [], []
    for draw in range(draws):
        uniforms = rng.random((4, chromosomeCount * maximum + chromosomeCount))
        size = 0
        with nogil:
            for index in range(chromosomeCount):
                sampled = <int>(uniforms[0, index] * chromosomeCount)
                position = <int>(uniforms[1, index] * lengths[sampled])
                for drawPosition in range(lengths[sampled]):
                    drawIndices[size] = groupIndex[sampled, position]
                    nextPosition = nextIndex[sampled, position]
                    if nextPosition < 0 or uniforms[2, size] < probability:
                        position = <int>(uniforms[3, size] * lengths[sampled])
                    else:
                        position = nextPosition
                    size += 1
        drawRadius, drawCensored, _, _ = _pooledRadius(
            numerator, denominator, drawIndices[:size], binBP, threshold, smooth, required)
        pooledDraws.append(drawRadius)
        pooledFlags.append(drawCensored)
    pooledLower, pooledUpper = _KMQuantiles(pooledDraws, pooledFlags, 0.025, 0.975)
    pooledResolved = draws - sum(pooledFlags)
    if pooledLower is None or pooledUpper is None or pooledResolved < int(ceil(0.95 * draws)):
        _supportFailure("Pooled covariance bootstrap support is insufficient for a resolved interval",
                        dict(estimateBP=pooledRadius, radiusLowerBP=pooledLower, radiusUpperBP=None,
                             rightCensored=False, supportCapLagBP=pooledCap, localMedianRadiusBP=median,
                             estimationTarget="pooledChromosomeCenteredCovariance"))
    return dict(estimateBP=pooledRadius, lowerBP=min(pooledLower, pooledRadius),
                upperBP=max(pooledUpper, pooledRadius), fullSampleWorkingSpanBP=pooledRadius,
                workingSpanBP=pooledRadius, workingSpanRule="pooledRadius",
                estimationTarget="pooledChromosomeCenteredCovariance", pooledCovariance=pooledCurve.tolist(),
                selectedAdjacencyCount=adjacency, selectedLongestRun=longest,
                bootstrapMethod="hierarchicalAutosomeStationaryWindow", bootstrapBlockLengthWindows=block,
                bootstrapRestartProbability=probability, bootstrapDrawsRequested=draws,
                bootstrapPooledRadiusBP=pooledDraws, bootstrapPooledRadiusCensored=pooledFlags,
                bootstrapResolvedPooledDraws=pooledResolved, confidenceLevel=0.95,
                confidenceIntervalMethod="hierarchicalStationaryPooledCovariancePercentile",
                fullSampleMedianRadiusBP=median,
                localRadiusSummary=dict(fullSampleMedianRadiusBP=median,
                                        localQuantile=quantile, localQuantileBP=localQuantileBP))


def _chooseDependenceSpan(chromosomeNames, chromosomeMatrices, intervalSizeBP, dict replicateData,
                               windowBP=constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_BP,
                               windowCount=constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_COUNT,
                               maxLagBP=constants.OBSERVATION_DEFAULT_DEPENDENCE_MAX_LAG_BP,
                               workingQuantile=0.9, bootstrapDraws=500, randSeed=1729,
                               minWindowCount=32, acfThreshold=0.1, acfSmoothingBP=500,
                               crossingPersistenceBP=500, minFinitePairs=200,
                               minFinitePairCoverage=0.5, crossingCount=5):
    integers = {name: _integer(value, name) for name, value in (
        ("intervalSizeBP", intervalSizeBP), ("windowBP", windowBP), ("windowCount", windowCount),
        ("maxLagBP", maxLagBP), ("bootstrapDraws", bootstrapDraws), ("randSeed", randSeed),
        ("minWindowCount", minWindowCount), ("acfSmoothingBP", acfSmoothingBP),
        ("crossingPersistenceBP", crossingPersistenceBP), ("minFinitePairs", minFinitePairs),
        ("crossingCount", crossingCount))}
    cdef int binBP = integers["intervalSizeBP"], widthBP = integers["windowBP"]
    cdef int width, lagCount, smooth, persistence, crossings = integers["crossingCount"]
    cdef int requested = integers["windowCount"], minimum = integers["minWindowCount"]
    cdef int pairMinimum = integers["minFinitePairs"], draws = integers["bootstrapDraws"]
    cdef double quantile = _real(workingQuantile, "workingQuantile")
    cdef double threshold = _real(acfThreshold, "acfThreshold")
    cdef double coverage = _real(minFinitePairCoverage, "minFinitePairCoverage")
    if binBP <= 0 or widthBP <= 0 or not 0 < integers["maxLagBP"] <= widthBP // 2:
        raise ValueError("Positive window/bin sizes and maxLagBP <= windowBP/2 are required")
    if requested <= 0 or draws <= 0 or not 0 < minimum <= requested or integers["randSeed"] < 0:
        raise ValueError("Invalid window/bootstrap counts or random seed")
    if not 0 < quantile < 1 or not 0 < threshold < 1 or not 0 < coverage <= 1:
        raise ValueError("Invalid quantile, ACF threshold, or pair coverage")
    if min(integers["acfSmoothingBP"], integers["crossingPersistenceBP"], pairMinimum, crossings) <= 0:
        raise ValueError("Smoothing, persistence, crossing count, and minimum pairs must be positive")
    if widthBP % binBP or integers["maxLagBP"] % binBP:
        raise ValueError("windowBP and maxLagBP must be integer multiples of intervalSizeBP")
    if max(integers["acfSmoothingBP"], integers["crossingPersistenceBP"]) > integers["maxLagBP"]:
        raise ValueError("ACF smoothing and crossing persistence cannot exceed maxLagBP")
    width, lagCount = widthBP // binBP, integers["maxLagBP"] // binBP
    smooth = _oddBins(integers["acfSmoothingBP"], binBP)
    persistence = int(ceil(integers["crossingPersistenceBP"] / <double>binBP))
    if pairMinimum > width or width < 2 or crossings > (lagCount - smooth + 1) // persistence:
        raise ValueError("Physical window and lag settings are too short for the support requirements")
    names, matrices = list(chromosomeNames), list(chromosomeMatrices)
    if not names or len(names) != len(matrices):
        raise ValueError("Chromosome inputs must be nonempty and have equal lengths")
    cdef int rows = -1, ordinal, row, other, chromosomeIndex, start, end, index
    cdef int rankStart, rankEnd, evaluated = 0, supportedCount = 0
    cdef double score, rank, rankWeight, rowScore
    eligible, excluded, seen = [], [], set()
    for name, matrix in zip(names, matrices):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] < 2 or matrix.dtype.kind not in "biuf":
            raise ValueError("Chromosome matrices must be nonempty real numeric 2D arrays")
        if rows < 0:
            rows = matrix.shape[0]
        elif rows != matrix.shape[0]:
            raise ValueError("Chromosome matrices must share one row count")
        if misc_util.isStandardAutosomalChromosome(name):
            ordinal = int(str(name).strip()[3:] if str(name).strip().lower().startswith("chr") else str(name).strip())
            if ordinal in seen:
                raise ValueError(f"duplicate canonical autosome chr{ordinal}")
            seen.add(ordinal)
            if matrix.shape[1] >= width:
                eligible.append((ordinal, f"chr{ordinal}", matrix))
            else:
                excluded.append(f"chr{ordinal}")
        else:
            excluded.append(str(name))
    eligible.sort(key=lambda record: record[0])
    if not eligible:
        raise ValueError("dependence estimator found no eligible autosomes")
    if not replicateData:
        retained = []
        for row in range(rows):
            if not any(all(np.array_equal(record[2][row], record[2][other], equal_nan=True)
                           for record in eligible) for other in retained):
                retained.append(row)
        replicateData.update(matrices={record[0]: _replicateMean(record[2], retained) for record in eligible},
                             inputRowCount=rows, uniqueRowCount=len(retained), duplicateRowCount=rows - len(retained),
                             replicateAggregation="pointwiseMean", covarianceRowCount=1,
                             minimumReplicateContributors=(len(retained) + 1) // 2,
                             finitePairQualityScope="pointwiseMeanTrack", bootstrapRefitsReplicateAggregation=False)
    eligible = [(record[0], record[1], replicateData["matrices"][record[0]]) for record in eligible]
    retained = [0]
    referenceMeans = [_referenceMeans(record[2], retained, width) for record in eligible]
    candidates = []
    for chromosomeIndex, record in enumerate(eligible):
        matrix = record[2]
        for start in range(0, matrix.shape[1] - width + 1, width):
            scores = []
            for row in retained:
                values = np.asarray(matrix[row, start:start + width], dtype=np.float64)
                finite = values[np.isfinite(values)]
                if finite.size and finite.size / <double>width >= sqrt(coverage):
                    scores.append(float(np.maximum(finite, 0).sum()) * width / finite.size)
            if scores:
                score = float(np.median(scores))
                candidates.append((-score, record[0], start * binBP, chromosomeIndex, start, score))
    candidates.sort()
    selectionSeed, bootstrapSeed = np.random.SeedSequence(integers["randSeed"]).spawn(2)
    keys = np.random.default_rng(selectionSeed).exponential(size=len(candidates))
    rankStart = 0
    while rankStart < len(candidates):
        rankEnd = rankStart + 1
        while rankEnd < len(candidates) and candidates[rankEnd][0] == candidates[rankStart][0]:
            rankEnd += 1
        rank = 0.5 * (rankStart + 1 + rankEnd)
        rankWeight = len(candidates) - rank + 1
        for index in range(rankStart, rankEnd):
            candidate = candidates[index]
            candidates[index] = (keys[index] / rankWeight, *candidate[1:], rank)
        rankStart = rankEnd
    candidates.sort()
    pools = {record[1]: [] for record in eligible}
    records = []
    for candidate in candidates:
        chromosomeIndex, start = candidate[3], candidate[4]
        stats = cclippedCovarianceReplicates(eligible[chromosomeIndex][2][retained, start:start + width],
                                           lagCount, pairMinimum, coverage, referenceMeans[chromosomeIndex])
        evaluated += 1
        if stats["curves"].shape[1] < smooth + persistence * crossings or stats["curves"][0, 0] <= 0:
            continue
        chromosome = eligible[chromosomeIndex][1]
        record = dict(chromosome=chromosome, startBP=start * binBP, endBP=(start + width) * binBP,
                      score=candidate[5], positiveSignalRank=candidate[6], stats=stats)
        records.append(record)
        pools[chromosome].append(record)
        supportedCount = sum(len(pool) for pool in pools.values() if len(pool) >= 5)
        if supportedCount >= requested:
            break
    if supportedCount < minimum:
        raise RuntimeError(f"dependence estimator has {supportedCount} valid windows, needs at least {minimum} windows with four donors")
    selected = []
    for record in records:
        if len(pools[record["chromosome"]]) >= 5:
            selected.append(record)
            if len(selected) == requested:
                break
    selectedPools = {record[1]: [item for item in selected if item["chromosome"] == record[1]] for record in eligible}
    # Extra evaluated windows support a chromosome with fewer than five selected targets.
    for chromosome, pool in selectedPools.items():
        if pool and len(pool) < 5:
            selectedPools[chromosome] = pools[chromosome]
    for chromosome, pool in selectedPools.items():
        if not pool:
            continue
        for record, shrunk in zip(pool, _shrink(pool)):
            record["result"] = _radius(record, shrunk, binBP, threshold, smooth, persistence, crossings, lagCount)
    windows, populationRecords = [], []
    for record in selected:
        if record["result"] is not None:
            windows.append({key: record[key] for key in ("chromosome", "startBP", "endBP", "score", "positiveSignalRank")}
                           | record["result"])
            populationRecords.append(record)
    if len(windows) < minimum:
        raise RuntimeError(f"dependence estimator has {len(windows)} valid windows, needs at least {minimum} windows")
    chromosomes = [record[1] for record in eligible if any(window["chromosome"] == record[1] for window in windows)]
    diagnostics = _inference(windows, chromosomes, np.random.default_rng(bootstrapSeed), draws, quantile,
                             populationRecords, binBP, threshold, smooth, persistence * crossings)
    censored = [window["gaussianEquivalentRadiusBP"] for window in windows if window["rightCensored"]]
    distributions = {key: [window[key] for window in windows if window[key] is not None]
                     for key in ("dominantACFPeriodBP", "oscillationStrength", "postCrossingRevival")}
    diagnostics.update(status="estimated", method="clippedCovarianceShrinkage", randomSeed=integers["randSeed"],
        inferenceScope="conditionalOnInputTracksSelectedWindowsAndChromosomeReferenceMeans", bootstrapRefitsShrinkage=False,
        bootstrapRefitsReferenceMeans=False, bootstrapRefitsCovarianceAggregation=True,
        intervalSizeBP=binBP, windowBP=widthBP, windowCountRequested=requested, candidateWindowCount=len(candidates),
        evaluatedCandidateWindowCount=evaluated, selectedWindowCount=len(windows), minWindowCount=minimum,
        selectedAutosomeCount=len(chromosomes), chromosomesUsed=chromosomes, chromosomesExcluded=sorted(set(excluded)),
        selectedWindows=windows,
        rowDeduplication="exactEquality", acfThreshold=threshold, acfSmoothingBP=integers["acfSmoothingBP"],
        acfSmoothingBins=smooth, crossingPersistenceBP=integers["crossingPersistenceBP"], crossingPersistenceBins=persistence,
        crossingCount=crossings, totalPersistenceBP=crossings * persistence * binBP, minFinitePairs=pairMinimum,
        minFinitePairCoverage=coverage, maxLagBP=lagCount * binBP,
        gaussianRadiusCorrection=3.0 / (2.0 * sqrt(-log(threshold))), censorFraction=len(censored) / len(windows),
        crossedWindowCount=len(windows) - len(censored), rightCensoredWindowCount=len(censored),
        supportCensoredWindowCount=sum(window["censorReason"] == "support" for window in windows),
        capCensoredWindowCount=sum(window["censorReason"] == "maxLag" for window in windows),
        censorTimeBPMinimum=min(censored) if censored else None, censorTimeBPMedian=float(np.median(censored)) if censored else None,
        censorTimeBPMaximum=max(censored) if censored else None,
        finitePairMinimumUsed=min(window["finitePairMinimumUsed"] for window in windows),
        finitePairCoverageMinimumUsed=min(window["finitePairCoverageMinimumUsed"] for window in windows),
        validRowsAtCrossingMinimum=min(window["validRowsAtCrossing"] for window in windows),
        selectedCountsByChromosome={name: sum(window["chromosome"] == name for window in windows) for name in chromosomes},
        radiusDistributionBP=[window["gaussianEquivalentRadiusBP"] for window in windows],
        radiusCensored=[window["rightCensored"] for window in windows],
        dominantACFPeriodBPMedian=float(np.median(distributions["dominantACFPeriodBP"])) if distributions["dominantACFPeriodBP"] else None,
        oscillationStrength=float(np.median(distributions["oscillationStrength"])) if distributions["oscillationStrength"] else None,
        postCrossingRevival=float(np.median(distributions["postCrossingRevival"])) if distributions["postCrossingRevival"] else None,
        unresolvedPeriodFraction=1 - len(distributions["dominantACFPeriodBP"]) / len(windows),
        periodicitySearchMinBP=max(2 * binBP, 150), periodicitySearchMaxBP=500)
    diagnostics.update({key: value for key, value in replicateData.items() if key != "matrices"})
    return (*(int(ceil(diagnostics[key] / binBP)) for key in ("estimateBP", "lowerBP", "upperBP")), diagnostics)


cdef void _boundDependenceRadii(dict diagnostics, int binBP) except *:
    cdef int minimumBP = constants.OBSERVATION_MIN_DEPENDENCE_RADIUS_BP
    cdef int maximumBP = constants.OBSERVATION_MAX_DEPENDENCE_RADIUS_BP
    cdef int reportMinimumBP = <int>ceil(minimumBP / <double>binBP) * binBP
    cdef int reportMaximumBP = (maximumBP // binBP) * binBP
    cdef double value
    diagnostics["unboundedRadiusSummary"] = dict(diagnostics)
    boundedFields = []
    for key, unboundedKey in (("estimateBP", "unboundedEstimateBP"),
                              ("lowerBP", "unboundedLowerBP"),
                              ("upperBP", "unboundedUpperBP"),
                              ("workingSpanBP", "unboundedWorkingSpanBP")):
        rawValue = diagnostics.get(key)
        diagnostics[unboundedKey] = rawValue
        if rawValue is not None:
            value = rawValue
            if isfinite(value):
                diagnostics[key] = float(min(reportMaximumBP, max(reportMinimumBP, value)))
                if diagnostics[key] != rawValue:
                    boundedFields.append(key)
    diagnostics.update(unboundedFullSampleWorkingSpanBP=diagnostics.get("fullSampleWorkingSpanBP"),
                       unboundedRadiusLowerBP=diagnostics.get("radiusLowerBP"),
                       unboundedRadiusUpperBP=diagnostics.get("radiusUpperBP"),
                       radiusMinimumBP=minimumBP, radiusMaximumBP=maximumBP,
                       reportingRadiusMinimumBP=reportMinimumBP, reportingRadiusMaximumBP=reportMaximumBP,
                       radiusBoundsApplied=bool(boundedFields), boundedRadiusFields=boundedFields)
    if "workingSpanRule" in diagnostics:
        diagnostics["workingSpanRule"] = "clip(pooledRadius,minimumRadius,maximumRadius)"


cpdef tuple cchooseDependenceSpan(chromosomeNames, chromosomeMatrices, intervalSizeBP,
                               windowBP=constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_BP,
                               windowCount=constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_COUNT,
                               maxLagBP=constants.OBSERVATION_DEFAULT_DEPENDENCE_MAX_LAG_BP,
                               workingQuantile=0.9, bootstrapDraws=500, randSeed=1729,
                               minWindowCount=32, acfThreshold=0.1, acfSmoothingBP=500,
                               crossingPersistenceBP=500, minFinitePairs=200,
                               minFinitePairCoverage=0.5, crossingCount=5, adaptiveSupport=True):
    if not isinstance(adaptiveSupport, (bool, np.bool_)):
        raise ValueError("adaptiveSupport must be boolean")
    cdef int binBP = _integer(intervalSizeBP, "intervalSizeBP")
    cdef int requestedWidthBP = _integer(windowBP, "windowBP")
    cdef int requestedLagBP = _integer(maxLagBP, "maxLagBP")
    if binBP <= 0:
        raise ValueError("intervalSizeBP must be positive")
    if requestedWidthBP <= 0 or not 0 < requestedLagBP <= requestedWidthBP // 2:
        raise ValueError("Positive window/bin sizes and maxLagBP <= windowBP/2 are required")
    if ceil(constants.OBSERVATION_MIN_DEPENDENCE_RADIUS_BP / <double>binBP) > constants.OBSERVATION_MAX_DEPENDENCE_RADIUS_BP // binBP:
        raise ValueError("intervalSizeBP admits no radius inside the reporting bounds")
    try:
        names, matrices = list(chromosomeNames), list(chromosomeMatrices)
        for index in range(len(matrices)):
            matrices[index] = np.asarray(matrices[index])
    except Exception as error:
        raise ValueError("Chromosome inputs must be finite sequences") from error
    arguments = dict(windowBP=(requestedWidthBP // binBP) * binBP, windowCount=windowCount,
                     maxLagBP=(requestedLagBP // binBP) * binBP,
                     workingQuantile=workingQuantile, bootstrapDraws=bootstrapDraws, randSeed=randSeed,
                     minWindowCount=minWindowCount, acfThreshold=acfThreshold, acfSmoothingBP=acfSmoothingBP,
                     crossingPersistenceBP=crossingPersistenceBP, minFinitePairs=minFinitePairs,
                     minFinitePairCoverage=minFinitePairCoverage, crossingCount=crossingCount)
    attempts, replicateData = [], {}
    while True:
        try:
            result = _chooseDependenceSpan(names, matrices, intervalSizeBP, replicateData, **arguments)
            result[3].update(requestedWindowBP=windowBP, requestedMaxLagBP=maxLagBP,
                             supportAttempts=attempts, adaptiveSupport=bool(adaptiveSupport))
            _boundDependenceRadii(result[3], binBP)
            return (int(ceil(result[3]["estimateBP"] / binBP)),
                    int(ceil(result[3]["lowerBP"] / binBP)),
                    int(ceil(result[3]["upperBP"] / binBP)), result[3])
        except RuntimeError as error:
            if not hasattr(error, "diagnostics"):
                raise
            details = error.diagnostics
            for key, value in replicateData.items():
                if key != "matrices":
                    details[key] = value
            details.update(windowBP=arguments["windowBP"], maxLagBP=arguments["maxLagBP"])
            _boundDependenceRadii(details, binBP)
            attempts.append(dict(details))
            details["supportAttempts"] = list(attempts)
            nextWidth = 2 * arguments["windowBP"]
            available = 0
            for name, matrix in zip(names, matrices):
                if misc_util.isStandardAutosomalChromosome(name):
                    count = matrix.shape[1] * binBP // nextWidth
                    if count >= 5:
                        available += count
            if not adaptiveSupport:
                raise
            if available < minWindowCount or nextWidth > 2147483647:
                lower = arguments["windowBP"] // intervalSizeBP
                upper = min(nextWidth, 2147483647) // intervalSizeBP
                while upper - lower > 1:
                    middle = (lower + upper) // 2
                    available = 0
                    for name, matrix in zip(names, matrices):
                        if misc_util.isStandardAutosomalChromosome(name):
                            count = matrix.shape[1] // middle
                            if count >= 5:
                                available += count
                    if available >= minWindowCount:
                        lower = middle
                    else:
                        upper = middle
                nextWidth = lower * intervalSizeBP
            if nextWidth <= arguments["windowBP"]:
                raise
            nextLag = (arguments["maxLagBP"] * nextWidth // arguments["windowBP"] // intervalSizeBP) * intervalSizeBP
            arguments.update(windowBP=nextWidth, maxLagBP=nextLag)


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
    str barcodeAllowListFile="",
    str barcodeTag="CB",
):
    r"""Return True when sampled alignment records carry the paired-end flag."""
    cdef bytes bamFileBytes = bamFile.encode("utf-8")
    cdef samFile* fileHandle = NULL
    cdef sam_hdr_t* header = NULL
    cdef bam1_t* record = NULL
    cdef int64_t sampled = 0
    cdef int readFlag
    cdef bint isPairedEnd = <bint>0
    cdef bytes allowListBytes = barcodeAllowListFile.encode("utf-8")
    cdef bytes tagBytes = barcodeTag.encode("utf-8")
    cdef const char* selectionError = NULL
    cdef crBarcodeSelection* selection = NULL
    cdef int selected

    if maxReads < 1:
        maxReads = 1

    fileHandle = sam_open(bamFileBytes, "r")
    if fileHandle == NULL:
        raise FileNotFoundError(f"Could not open alignment file `{bamFile}`")

    try:
        selection = crBarcodeSelectionCreate(tagBytes, allowListBytes, &selectionError)
        if selection == NULL:
            raise ValueError((<bytes>selectionError).decode("utf-8"))
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
            if selection.active and (readFlag & 1) != 0 and (readFlag & 64) == 0:
                continue
            selected = crBarcodeSelected(selection, record)
            if selected < 0:
                raise ValueError("barcodeTag requires SAM type Z")
            if not selected:
                continue
            sampled += 1
            if (readFlag & 1) != 0:
                isPairedEnd = <bint>1
                break

        return isPairedEnd
    finally:
        crBarcodeSelectionDestroy(selection)
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
    str barcodeAllowListFile="",
    str barcodeTag="CB",
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
    cdef bytes allowListBytes = barcodeAllowListFile.encode("utf-8")
    cdef bytes tagBytes = barcodeTag.encode("utf-8")
    cdef const char* selectionError = NULL
    cdef crBarcodeSelection* selection = NULL
    cdef int selected
    cdef bint hasSelection = bool(barcodeAllowListFile)

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
        if hasSelection:
            raise ValueError("could not estimate fragment length from selected BAM records")
        return <int64_t>fallBack
    if blockSize < 64:
        blockSize = 64
    if rollingChunkSize < 1:
        rollingChunkSize = 1
    if lagStep < 1:
        lagStep = 1

    fileHandle = sam_open(bamFileBytes, "r")
    if fileHandle == NULL:
        if hasSelection:
            raise ValueError("could not estimate fragment length from selected BAM records")
        return <int64_t>fallBack

    try:
        selection = crBarcodeSelectionCreate(tagBytes, allowListBytes, &selectionError)
        if selection == NULL:
            raise ValueError((<bytes>selectionError).decode("utf-8"))
        if samThreadsInternal > 1:
            hts_set_threads(<htsFile*>fileHandle, samThreadsInternal)

        header = sam_hdr_read(fileHandle)
        if header == NULL or header.n_targets <= 0:
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
            return <int64_t>fallBack

        indexHandle = sam_index_load(<htsFile*>fileHandle, bamFileBytes)
        if indexHandle == NULL:
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
            return <int64_t>fallBack

        record = bam_init1()
        if record == NULL:
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
            return <int64_t>fallBack

        lengthsArr = np.empty(header.n_targets, dtype=np.int64)
        for contigIdx in range(header.n_targets):
            lengthsArr[contigIdx] = <int64_t>header.target_len[contigIdx]

        kTop = header.n_targets if hasSelection else (3 if header.n_targets >= 3 else (2 if header.n_targets >= 2 else 1))
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
                if hasSelection and (readFlag & 1) != 0 and (readFlag & 64) == 0:
                    continue
                selected = crBarcodeSelected(selection, record)
                if selected < 0:
                    raise ValueError("barcodeTag requires SAM type Z")
                if not selected:
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
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
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
                    if (readFlag & 64) == 0:
                        continue
                    selected = crBarcodeSelected(selection, record)
                    if selected < 0:
                        raise ValueError("barcodeTag requires SAM type Z")
                    if not selected:
                        continue
                    if (readFlag & 2) == 0:
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

            if tlenN < (1 if hasSelection else max(500, requiredSamplesPE // 5)):
                if hasSelection:
                    raise ValueError("could not estimate fragment length from selected BAM records")
                return <int64_t>fallBack

            midIdx = tlenN // 2
            tlenWork = tlenArr[:tlenN].copy()
            tlenWork = np.partition(tlenWork, midIdx)
            tlenWorkView = tlenWork
            medPE = <cnp.int32_t>tlenWorkView[midIdx]

            if medPE < <cnp.int32_t>minInsertSize:
                return <int64_t>minInsertSize
            if medPE > <cnp.int32_t>maxInsertSize:
                if hasSelection:
                    raise ValueError("could not estimate fragment length from selected BAM records")
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
                selected = crBarcodeSelected(selection, record)
                if selected < 0:
                    raise ValueError("barcodeTag requires SAM type Z")
                if not selected:
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
                    selected = crBarcodeSelected(selection, record)
                    if selected < 0:
                        raise ValueError("barcodeTag requires SAM type Z")
                    if not selected:
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
        crBarcodeSelectionDestroy(selection)
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
        if hasSelection:
            raise ValueError("could not estimate fragment length from selected BAM records")
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
    cdef double dbarCorrection
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
            dbarCorrection = 0.0
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
                    continue
                baseVariance = (<double>muncPtr[idx]) + pad
                if baseVariance < varianceFloor:
                    baseVariance = varianceFloor
                residual = (<double>dataPtr[idx]) - backgroundValue - stateValue
                momentValue = residual * residual + momentVarianceBase
                compensatedSumAdd(momentValue / baseVariance, &dbar, &dbarCorrection)
                activeCount += 1
            if activeCount > 0:
                dbar = (dbar + dbarCorrection) / (<double>activeCount)
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
            residual = (<double>dataPtr[idx]) - backgroundValue - stateValue
            momentValue = residual * residual + momentVarianceBase
            rhoValue = 1.0
            if useWeights and studentT:
                if updateWeights:
                    baseVariance = max((<double>muncPtr[idx]) + pad, varianceFloor)
                    rhoValue = (dS + 1.0) / (
                        dS + omegaInValue * momentValue / baseVariance
                    )
                else:
                    rhoValue = <double>rhoInPtr[idx]
            momentPtr[idx] = <cnp.float32_t>momentValue
            rhoOutPtr[idx] = <cnp.float32_t>rhoValue
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
    _doubleDouble* suffixSums,
    Py_ssize_t* suffixCounts,
) noexcept nogil:
    cdef Py_ssize_t rowStart = rowIndex * intervalCount
    cdef Py_ssize_t i, left, right
    cdef Py_ssize_t width = min(windowIntervals, intervalCount)
    cdef Py_ssize_t half = width // 2
    cdef const uint8_t* rowExcluded = excludePtr
    cdef _windowSumState state = _windowSumState(0, 0, 0, 0, 0, _ddValue(0.0))
    cdef _doubleDouble total
    cdef double value
    if excludeMode == 2:
        rowExcluded += rowStart
    for i in range(intervalCount):
        left = max(0, min(i - half, intervalCount - width))
        right = left + width
        total = _blockedWindowSum(
            localPtr + rowStart, rowExcluded, intervalCount, width, left, right,
            &state, suffixSums, suffixCounts,
        )
        value = ((total.high + total.low) / <double>state.count
                 if state.count > 0 else <double>localPtr[rowStart + i])
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
    cdef Py_ssize_t scratchWidth = windowIntervals if windowIntervals < intervalCount else 0
    cdef Py_ssize_t scratchRows = 1 if trackCount > 0 else 0
    cdef double epsValue = <double>eps
    cdef object excludeObj
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] outArr
    cdef cnp.ndarray[cnp.float64_t, ndim=3, mode="c"] suffixSumArr
    cdef cnp.ndarray[cnp.intp_t, ndim=2, mode="c"] suffixCountArr
    cdef _doubleDouble* suffixSumPtr
    cdef Py_ssize_t* suffixCountPtr
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
    if USE_OPENMP and cellCount >= OPENMP_APPLY_MIN_ROWS:
        scratchRows = trackCount
    suffixSumArr = np.empty((scratchRows, scratchWidth, 2), dtype=np.float64)
    suffixCountArr = np.empty((scratchRows, scratchWidth), dtype=np.intp)
    suffixSumPtr = <_doubleDouble*>suffixSumArr.data
    suffixCountPtr = <Py_ssize_t*>suffixCountArr.data

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
                    suffixSumPtr + rowIndex * scratchWidth,
                    suffixCountPtr + rowIndex * scratchWidth,
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
                        suffixSumPtr,
                        suffixCountPtr,
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
                    suffixSumPtr,
                    suffixCountPtr,
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


cdef inline double _transformValue(
    double xval,
    int mode,
    double inputOffset,
    double inputScale,
    double outputScale,
    double outputOffset,
    double shape,
) noexcept nogil:
    cdef double u

    if mode == __TRANSFORM_MODE_LOG:
        u = xval + inputOffset
        if u <= 0.0:
            u = inputOffset
        u = u / inputScale
        if u <= 0.0:
            u = 1.0
        return outputOffset + outputScale * log(u)

    if mode == __TRANSFORM_MODE_SQRT or mode == __TRANSFORM_MODE_ANSCOMBE:
        u = (xval + inputOffset) / inputScale
        if u < 0.0:
            u = 0.0
        return outputOffset + outputScale * sqrt(u)

    if mode == __TRANSFORM_MODE_ASINH:
        u = (xval + inputOffset) / inputScale
        return outputOffset + outputScale * asinh(u)

    if mode == __TRANSFORM_MODE_ASINH_SQRT:
        u = xval + inputOffset
        if u < 0.0:
            u = 0.0
        return outputOffset + outputScale * asinh(
            sqrt(u) / inputScale
        )

    if mode == __TRANSFORM_MODE_GENERALIZED_LOG:
        # Use the inverse-hyperbolic identity to avoid cancellation for negative input.
        # NIST DLMF, Eq. 4.37.16, https://dlmf.nist.gov/4.37.E16.
        u = (xval + inputOffset) / inputScale
        return outputOffset + outputScale * asinh(u / shape)

    return outputOffset + outputScale * (
        (xval + inputOffset) / inputScale
    )


cdef void _monoTransform(
    const real_t* arrPtr,
    real_t* outPtr,
    Py_ssize_t n,
    double inputOffset,
    double inputScale,
    double outputScale,
    double outputOffset,
    double shape,
    int mode,
) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        outPtr[i] = <real_t>_transformValue(
            arrPtr[i],
            mode,
            inputOffset,
            inputScale,
            outputScale,
            outputOffset,
            shape,
        )


cdef inline double _asinhDiff(double t, double c, double difference) noexcept nogil:
    cdef double sign = 1.0
    cdef double rootT, rootC, ratio
    if t < 0.0 and c < 0.0:
        t = -t
        c = -c
        difference = -difference
        sign = -1.0
    if t >= 0.0 and c >= 0.0 and fabs(difference) <= 0.5 * fmax(t, c):
        rootT = hypot(t, 1.0)
        rootC = hypot(c, 1.0)
        # Rationalize the root difference before forming the log ratio.
        ratio = (0.5 * t + 0.5 * c) / (0.5 * rootT + 0.5 * rootC)
        ratio = (difference / (0.5 * c + 0.5 * rootC)) * (0.5 + 0.5 * ratio)
        return sign * log1p(ratio)
    return sign * (asinh(t) - asinh(c))


cdef void _transformDiff(
    const real_t* treatmentPtr,
    const real_t* controlPtr,
    real_t* outPtr,
    Py_ssize_t n,
    double inputOffset,
    double inputScale,
    double outputScale,
    double shape,
    int mode,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef double t, c, difference, rootT, rootC

    if mode == __TRANSFORM_MODE_LOG:
        for i in range(n):
            t = <double>treatmentPtr[i] + inputOffset
            c = <double>controlPtr[i] + inputOffset
            difference = <double>treatmentPtr[i] - <double>controlPtr[i]
            if t <= 0.0 or c <= 0.0:
                if t <= 0.0:
                    t = inputOffset
                if c <= 0.0:
                    c = inputOffset
                difference = t - c
            if c > 0.0 and fabs(difference) <= 0.5 * c:
                outPtr[i] = <real_t>(outputScale * log1p(difference / c))
            else:
                outPtr[i] = <real_t>(outputScale * (log(t) - log(c)))
        return

    for i in range(n):
        t = <double>treatmentPtr[i] + inputOffset
        c = <double>controlPtr[i] + inputOffset
        difference = <double>treatmentPtr[i] - <double>controlPtr[i]
        if mode == __TRANSFORM_MODE_SQRT or mode == __TRANSFORM_MODE_ANSCOMBE or mode == __TRANSFORM_MODE_ASINH_SQRT:
            if t < 0.0 or c < 0.0:
                t = fmax(t, 0.0)
                c = fmax(c, 0.0)
                difference = t - c
            if mode == __TRANSFORM_MODE_ASINH_SQRT:
                rootT = sqrt(t)
                rootC = sqrt(c)
                difference = difference / (rootT + rootC) if rootT + rootC > 0.0 else 0.0
                outPtr[i] = <real_t>(outputScale * _asinhDiff(
                    rootT / inputScale, rootC / inputScale, difference / inputScale,
                ))
            else:
                rootT = sqrt(t / inputScale)
                rootC = sqrt(c / inputScale)
                outPtr[i] = <real_t>(outputScale * ((difference / inputScale) / (rootT + rootC))) if rootT + rootC > 0.0 else <real_t>0.0
        elif mode == __TRANSFORM_MODE_ASINH or mode == __TRANSFORM_MODE_GENERALIZED_LOG:
            t /= inputScale
            c /= inputScale
            difference /= inputScale
            if mode == __TRANSFORM_MODE_GENERALIZED_LOG:
                t /= shape
                c /= shape
                difference /= shape
            outPtr[i] = <real_t>(outputScale * _asinhDiff(t, c, difference))
        else:
            outPtr[i] = <real_t>(outputScale * (difference / inputScale))


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
                offset_,
                <float>1.0,
                scale_,
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
                inputOffset_,
                inputScale_,
                outputScale_,
                shape_,
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
                inputOffset_,
                inputScale_,
                outputScale_,
                outputOffset_,
                shape_,
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
    float obsPrecisionMultiplierMin=0.25,
    float obsPrecisionMultiplierMax=4.0,
    float procPrecisionMultiplierMin=0.0001,
    float procPrecisionMultiplierMax=10.0,
    object observationStats=None,
    bint computeStatistic=True,
):
    r"""Run the forward pass (filter) for state estimation

    See :func:`consenrich.cconsenrich.cfixedBackgroundECM`, where this routine is applied
    within the filter, smooth, update loop.


    :seealso: :func:`consenrich.cconsenrich.cbackwardPass`,
            :func:`consenrich.cconsenrich.cfixedBackgroundECM`,
            :func:`consenrich.core.runConsenrich`
    """

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] observationStatsArr
    cdef const double* observationStatsPtr = NULL
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
    cdef bint useProcPrec = (
        ECM_useProcessPrecisionReweighting
        and (processPrecExp is not None)
    )
    cdef cnp.float32_t* dataPtr = <cnp.float32_t*>matrixData.data
    cdef cnp.float32_t* muncPtr = <cnp.float32_t*>matrixPluginMuncInit.data
    cdef cnp.int32_t* blockMapPtr = <cnp.int32_t*>intervalToBlockMap.data
    cdef cnp.float32_t* dStatPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* lambdaExpPtr = <cnp.float32_t*>NULL
    cdef cnp.float32_t* processPrecExpPtr = <cnp.float32_t*>NULL
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

    cdef double LOG2PI = log(6.2831853071795864769)

    if observationStats is not None:
        observationStatsArr = observationStats
        if observationStatsArr.shape[0] != intervalCount or observationStatsArr.shape[1] != 5:
            raise ValueError("observationStats must have shape (intervalCount, 5)")
        observationStatsPtr = <const double*>observationStatsArr.data

    if useLambda:
        lambdaExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> lambdaExp

    if useProcPrec:
        processPrecExpArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> processPrecExp

    if intervalCount <= 0 or trackCount <= 0:
        if not computeStatistic:
            if returnNLL:
                return (np.float32(0.0), 0, None, 0.0)
            return (np.float32(0.0), 0, None)
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

    if computeStatistic:
        if vectorD is None:
            dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
            vectorD = dStatVectorArr
        else:
            dStatVectorArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> vectorD
            if dStatVectorArr.shape[0] < intervalCount:
                raise ValueError("vectorD length must match intervalCount")
        dStatPtr = <cnp.float32_t*>dStatVectorArr.data
    else:
        vectorD = None

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
    with nogil:
        loopResult = _levelTrendForwardPassLoop(
            dataPtr,
            muncPtr,
            observationStatsPtr,
            blockMapPtr,
            lambdaExpPtr,
            processPrecExpPtr,
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
            LOG2PI,
            wMin,
            wMax,
            procPrecMin,
            procPrecMax,
            doStore,
            useLambda,
            useProcPrec,
            returnNLL,
            storeNLLInD,
        )

    if loopResult.invalidBlockIndex >= 0:
        raise ValueError("intervalToBlockMap has out-of-range block id")
    if loopResult.invalidCovarianceIndex >= 0:
        raise ArithmeticError("state covariance became non-finite")

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
    bint computeResiduals=True,
    bint computeLagCovariance=True,
):
    r"""Run the backward pass (smoother)

    This function executes the smoothing phase of Consenrich's forward-backward state estimation. It operates on
    outputs from the *forward-filtered* outputs (those returned by :func:`consenrich.cconsenrich.cforwardPass`).

    That is, given the forward-pass, filtered estimates over genomic intervals :math:`i = 1, \dots, n`,

    .. math::

        \mathbf{x}_{[i|i]}, \qquad \mathbf{P}_{[i|i]}, \qquad \mathbf{Q}^{\mathrm{eff}}_{[i]},

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
    cdef double invPred00, invPred01, invPred10, invPred11
    cdef double cross00, cross01, cross10, cross11
    cdef double J00, J01, J10, J11
    cdef double dx0, dx1
    cdef double xs0, xs1
    cdef double nextP00, nextP01, nextP11
    cdef double A00, A01, A10, A11
    cdef double conditional00, conditional01, conditional11
    cdef double Ps00, Ps01, Ps11
    cdef double C00, C01, C10, C11
    cdef bint rankDeficient
    cdef Py_ssize_t invalidCovarianceIndex = -1
    cdef cnp.float32_t* stateCovarSmoothedPtr = NULL

    cdef double innov

    if stateSmoothed is not None:
        stateSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> stateSmoothed
    else:
        stateSmoothedArr = np.empty((intervalCount, 2), dtype=np.float32)

    if stateCovarSmoothed is not None:
        stateCovarSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> stateCovarSmoothed
    else:
        stateCovarSmoothedArr = np.empty((intervalCount, 2, 2), dtype=np.float32)

    if computeLagCovariance:
        if lagCovSmoothed is not None:
            lagCovSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> lagCovSmoothed
        else:
            lagCovSmoothedArr = np.empty((max(intervalCount - 1, 1), 2, 2), dtype=np.float32)

    else:
        lagCovSmoothedArr = None

    if computeResiduals:
        if postFitResiduals is not None:
            postFitResidualsArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> postFitResiduals
        else:
            postFitResidualsArr = np.empty((intervalCount, trackCount), dtype=np.float32)
    else:
        postFitResidualsArr = None

    stateSmoothedView = stateSmoothedArr
    stateCovarSmoothedView = stateCovarSmoothedArr
    stateCovarSmoothedPtr = <cnp.float32_t*>stateCovarSmoothedArr.data
    if computeLagCovariance:
        lagCovSmoothedView = lagCovSmoothedArr
    if computeResiduals:
        postFitResidualsView = postFitResidualsArr

    F00 = <double>fView[0, 0]
    F01 = <double>fView[0, 1]
    F10 = <double>fView[1, 0]
    F11 = <double>fView[1, 1]

    if intervalCount <= 0:
        return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)

    Ps00 = <double>stateCovarForwardView[intervalCount - 1, 0, 0]
    Ps01 = 0.5 * (
        (<double>stateCovarForwardView[intervalCount - 1, 0, 1])
        + (<double>stateCovarForwardView[intervalCount - 1, 1, 0])
    )
    Ps11 = <double>stateCovarForwardView[intervalCount - 1, 1, 1]
    if not _projectCovariance2x2(&Ps00, &Ps01, &Ps11):
        raise ArithmeticError("state covariance is not positive semidefinite")

    with nogil:
        # ========================================================
        # initialize with the final forward pass estimates at k = intervalCount - 1
        # ========================================================
        stateSmoothedView[intervalCount - 1, 0] = stateForwardView[intervalCount - 1, 0]
        stateSmoothedView[intervalCount - 1, 1] = stateForwardView[intervalCount - 1, 1]

        if not _storeCovariance2x2F32(
            Ps00,
            Ps01,
            Ps11,
            stateCovarSmoothedPtr + (intervalCount - 1) * 4,
        ):
            invalidCovarianceIndex = intervalCount - 1

        if computeResiduals and invalidCovarianceIndex < 0:
            for j in range(trackCount):
                postFitResidualsView[intervalCount - 1, j] = <cnp.float32_t>(
                    (<double>dataView[j, intervalCount - 1]) - (<double>stateSmoothedView[intervalCount - 1, 0])
                )

        #  `k = intervalCount - 2`...`k=0`
        for k in range(intervalCount - 2, -1, -1):
            if invalidCovarianceIndex >= 0:
                break
            Pf00 = <double>stateCovarForwardView[k, 0, 0]
            Pf01 = 0.5 * (
                (<double>stateCovarForwardView[k, 0, 1])
                + (<double>stateCovarForwardView[k, 1, 0])
            )
            Pf10 = Pf01
            Pf11 = <double>stateCovarForwardView[k, 1, 1]
            xPred0 = F00*(<double>stateForwardView[k, 0]) + F01*(<double>stateForwardView[k, 1])
            xPred1 = F10*(<double>stateForwardView[k, 0]) + F11*(<double>stateForwardView[k, 1])
            Q00 = <double>pNoiseForwardView[k, 0, 0]
            Q01 = 0.5 * (
                (<double>pNoiseForwardView[k, 0, 1])
                + (<double>pNoiseForwardView[k, 1, 0])
            )
            Q10 = Q01
            Q11 = <double>pNoiseForwardView[k, 1, 1]
            cross00 = F00*Pf00 + F01*Pf10
            cross01 = F00*Pf01 + F01*Pf11
            cross10 = F10*Pf00 + F11*Pf10
            cross11 = F10*Pf01 + F11*Pf11

            PPred00 = cross00*F00 + cross01*F01 + Q00
            PPred01 = cross00*F10 + cross01*F11 + Q01
            PPred10 = cross10*F00 + cross11*F01 + Q10
            PPred11 = cross10*F10 + cross11*F11 + Q11

            PPred01 = 0.5 * (PPred01 + PPred10)
            PPred10 = PPred01
            if not _inverseSymmetricPSD2x2(
                PPred00,
                PPred01,
                PPred11,
                &invPred00,
                &invPred01,
                &invPred11,
                &rankDeficient,
            ):
                invalidCovarianceIndex = k
                break
            invPred10 = invPred01

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

            nextP00 = <double>stateCovarSmoothedView[k + 1, 0, 0]
            nextP01 = <double>stateCovarSmoothedView[k + 1, 0, 1]
            nextP11 = <double>stateCovarSmoothedView[k + 1, 1, 1]
            A00 = 1.0 - (J00 * F00 + J01 * F10)
            A01 = -(J00 * F01 + J01 * F11)
            A10 = -(J10 * F00 + J11 * F10)
            A11 = 1.0 - (J10 * F01 + J11 * F11)
            # Conditional covariance plus propagated smoothed covariance,
            # using (I-JF) Pf (I-JF)' + J Q J' for the conditional term.
            # Rasmussen and Williams (2006), Eq. (A.6), gaussianprocess.org/gpml/chapters/RWA.pdf.
            _symmetricCongruence2x2(
                A00, A01, A10, A11, Pf00, Pf01, Pf11,
                &conditional00, &conditional01, &conditional11,
            )
            _symmetricCongruence2x2(
                J00, J01, J10, J11,
                Q00 + nextP00, Q01 + nextP01, Q11 + nextP11,
                &Ps00, &Ps01, &Ps11,
            )
            Ps00 += conditional00
            Ps01 += conditional01
            Ps11 += conditional11
            if not _projectCovariance2x2(&Ps00, &Ps01, &Ps11):
                invalidCovarianceIndex = k
                break
            if not _storeCovariance2x2F32(
                Ps00,
                Ps01,
                Ps11,
                stateCovarSmoothedPtr + k * 4,
            ):
                invalidCovarianceIndex = k
                break

            if computeLagCovariance:
                C00 = J00 * nextP00 + J01 * nextP01
                C01 = J00 * nextP01 + J01 * nextP11
                C10 = J10 * nextP00 + J11 * nextP01
                C11 = J10 * nextP01 + J11 * nextP11
                if rankDeficient:
                    # Retain the cross covariance in discarded prediction modes.
                    C00 += cross00 - (J00 * PPred00 + J01 * PPred10)
                    C01 += cross01 - (J00 * PPred01 + J01 * PPred11)
                    C10 += cross10 - (J10 * PPred00 + J11 * PPred10)
                    C11 += cross11 - (J10 * PPred01 + J11 * PPred11)

                if k < lagCovSmoothedArr.shape[0]:
                    lagCovSmoothedView[k, 0, 0] = <cnp.float32_t>C00
                    lagCovSmoothedView[k, 0, 1] = <cnp.float32_t>C01
                    lagCovSmoothedView[k, 1, 0] = <cnp.float32_t>C10
                    lagCovSmoothedView[k, 1, 1] = <cnp.float32_t>C11

            if computeResiduals:
                for j in range(trackCount):
                    innov = (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                    postFitResidualsView[k, j] = <cnp.float32_t>innov

    if invalidCovarianceIndex >= 0:
        raise ArithmeticError("predicted covariance is not positive semidefinite")

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
    float obsPrecisionMultiplierMin=0.25,
    float obsPrecisionMultiplierMax=4.0,
    float procPrecisionMultiplierMin=0.0001,
    float procPrecisionMultiplierMax=10.0,
    object observationStats=None,
    bint computeStatistic=True,
):
    r"""Run the scalar level-only forward pass."""

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] observationStatsArr
    cdef const double* observationStatsPtr = NULL
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
    cdef bint useProcPrec = (
        ECM_useProcessPrecisionReweighting
        and (processPrecExp is not None)
    )
    cdef const cnp.float32_t* dataPtr = NULL
    cdef const cnp.float32_t* muncPtr = NULL
    cdef const cnp.int32_t* blockMapPtr = NULL
    cdef const cnp.float32_t* lambdaExpPtr = NULL
    cdef const cnp.float32_t* processPrecExpPtr = NULL
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
    cdef double LOG2PI = log(6.2831853071795864769)

    if intervalCount <= 0 or trackCount <= 0:
        if not computeStatistic:
            if returnNLL:
                return (np.float32(0.0), 0, None, 0.0)
            return (np.float32(0.0), 0, None)
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

    if observationStats is not None:
        observationStatsArr = observationStats
        if observationStatsArr.shape[0] != intervalCount or observationStatsArr.shape[1] != 5:
            raise ValueError("observationStats must have shape (intervalCount, 5)")
        observationStatsPtr = <const double*>observationStatsArr.data

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
    if computeStatistic:
        if vectorD is None:
            dStatVectorArr = np.empty(intervalCount, dtype=np.float32)
        else:
            dStatVectorArr = <cnp.ndarray[cnp.float32_t, ndim=1, mode="c"]> vectorD
            if dStatVectorArr.shape[0] < intervalCount:
                raise ValueError("vectorD length must match intervalCount")
        dStatPtr = <cnp.float32_t*>dStatVectorArr.data
    else:
        vectorD = None

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

    dataPtr = <const cnp.float32_t*>matrixData.data
    muncPtr = <const cnp.float32_t*>matrixPluginMuncInit.data
    blockMapPtr = <const cnp.int32_t*>intervalToBlockMap.data

    with nogil:
        loopResult = _levelForwardPassLoop(
            dataPtr,
            muncPtr,
            observationStatsPtr,
            blockMapPtr,
            lambdaExpPtr,
            processPrecExpPtr,
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
            doStore,
            useLambda,
            useProcPrec,
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
    bint computeResiduals=True,
    bint computeLagCovariance=True,
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
    cdef double nextP
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
    if computeLagCovariance:
        if lagCovSmoothed is not None:
            lagCovSmoothedArr = <cnp.ndarray[cnp.float32_t, ndim=3, mode="c"]> lagCovSmoothed
        else:
            lagCovSmoothedArr = np.empty((max(intervalCount - 1, 1), 1, 1), dtype=np.float32)
    else:
        lagCovSmoothedArr = None

    if computeResiduals:
        if postFitResiduals is not None:
            postFitResidualsArr = <cnp.ndarray[cnp.float32_t, ndim=2, mode="c"]> postFitResiduals
        else:
            postFitResidualsArr = np.empty((intervalCount, trackCount), dtype=np.float32)
    else:
        postFitResidualsArr = None

    stateSmoothedView = stateSmoothedArr
    stateCovarSmoothedView = stateCovarSmoothedArr
    if computeLagCovariance:
        lagCovSmoothedView = lagCovSmoothedArr
    if computeResiduals:
        postFitResidualsView = postFitResidualsArr

    if intervalCount <= 0:
        return (stateSmoothedArr, stateCovarSmoothedArr, lagCovSmoothedArr, postFitResidualsArr)

    with nogil:
        stateSmoothedView[intervalCount - 1, 0] = stateForwardView[intervalCount - 1, 0]
        stateCovarSmoothedView[intervalCount - 1, 0, 0] = stateCovarForwardView[intervalCount - 1, 0, 0]

        if computeResiduals:
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
                Q = PPred - Pf
            J = Pf / PPred
            dx = (<double>stateSmoothedView[k + 1, 0]) - (<double>stateForwardView[k, 0])
            xs = (<double>stateForwardView[k, 0]) + J * dx
            stateSmoothedView[k, 0] = <cnp.float32_t>xs

            nextP = <double>stateCovarSmoothedView[k + 1, 0, 0]
            Ps = J * Q + J * J * nextP
            if Ps < 0.0:
                Ps = 0.0
            stateCovarSmoothedView[k, 0, 0] = <cnp.float32_t>Ps

            if computeLagCovariance:
                C = J * nextP
                if k < lagCovSmoothedArr.shape[0]:
                    lagCovSmoothedView[k, 0, 0] = <cnp.float32_t>C

            if computeResiduals:
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
    float procPrecisionMultiplierMin=0.0001,
    float procPrecisionMultiplierMax=10.0,
    bint ECM_useObsPrecisionReweighting=True,
    bint ECM_useProcessPrecisionReweighting=True,
    Py_ssize_t t_innerIters=5,
    bint returnIntermediates=False,
    bint returnDiagnostics=False,
    object lambdaExpInit=None,
    object processPrecExpInit=None,
    bint trackOptimizationPath=False,
    bint logIterations=True,
    bint ECM_scaleObsPrecisionToMedian=False,
    bint ECM_scaleProcessPrecisionToMedian=True,
    bint obsPrecisionWarmStartIsMedianScaled=False,
    bint processPrecisionWarmStartIsMedianScaled=False,
    float ECM_processRobustTNu=8.0):
    r"""Run the fixed-background ECM for the scalar level-only process model

    :param matrixData: Offset-adjusted observations, shape ``(m, n)``.
    :param matrixPluginMuncInit: Base observation variances
        :math:`v_{[j,i]}`, shape ``(m, n)``.
    :param ECM_fixedBackgroundIters: Maximum number of ECM iterations.
    :param ECM_fixedBackgroundRtol: Relative negative-log-likelihood
        tolerance for convergence.
    :param t_innerIters: Filter--smoother and reweighting cycles per
        ECM iteration.
    :param ECM_robustTNu: Observation degrees of freedom
        :math:`\nu_{\mathrm{obs}}`.
    :param ECM_processRobustTNu: Process degrees of freedom
        :math:`\nu_{\mathrm{proc}}`.
    :return: Iteration count and final obj.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] observationStats = None
    cdef double[:, ::1] observationStatsView
    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef Py_ssize_t i, k, j, inner
    cdef Py_ssize_t b
    cdef Py_ssize_t activeTrackCount
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
    cdef bint processPrecisionActive = ECM_useProcessPrecisionReweighting
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] medianWorkspaceArr
    cdef cnp.float32_t[::1] medianWorkspaceView
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateForward = np.empty((intervalCount, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarForward = np.empty((intervalCount, 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] pNoiseForward = np.empty((intervalCount, 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] stateSmoothed = np.empty((intervalCount, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] stateCovarSmoothed = np.empty((intervalCount, 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] lagCovSmoothed = np.empty((max(intervalCount - 1, 1), 1, 1), dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] postFitResiduals = np.empty((intervalCount, trackCount), dtype=np.float32)
    cdef cnp.float32_t[:, ::1] residualView = postFitResiduals
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
    cdef double p00k
    cdef double x0, y0
    cdef double delta
    cdef double obsU2
    cdef double w
    cdef double wMin = <double>obsPrecisionMultiplierMin
    cdef double wMax = <double>obsPrecisionMultiplierMax
    cdef double kappa_
    cdef double kappaMin_ = <double>procPrecisionMultiplierMin
    cdef double kappaMax_ = <double>procPrecisionMultiplierMax
    cdef double dState = 1.0
    cdef double procNu = ECM_processRobustTNu
    cdef Py_ssize_t stableIters = 0
    cdef Py_ssize_t patienceTarget = 2
    cdef bint iterationConverged = False
    cdef object optimizationPath = None

    _validateMultiplierBounds(wMin, wMax, True)
    _validateMultiplierBounds(kappaMin_, kappaMax_, False)
    if not isfinite(procNu) or procNu <= 0.0:
        raise ValueError("ECM_processRobustTNu must be positive and finite")
    _validateMedianScaling(
        ECM_scaleObsPrecisionToMedian,
        ECM_useObsPrecisionReweighting,
        wMin,
        wMax,
        True,
    )
    _validateMedianScaling(
        ECM_scaleProcessPrecisionToMedian,
        processPrecisionActive,
        kappaMin_,
        kappaMax_,
        False,
    )
    if ECM_scaleObsPrecisionToMedian or ECM_scaleProcessPrecisionToMedian:
        medianWorkspaceArr = np.empty(max(intervalCount, 1), dtype=np.float32)
        medianWorkspaceView = medianWorkspaceArr

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
            if not ECM_scaleObsPrecisionToMedian:
                np.clip(lambdaExpArr, obsPrecisionMultiplierMin, obsPrecisionMultiplierMax, out=lambdaExpArr)
        lambdaExp = lambdaExpArr
        lambdaExpView = lambdaExpArr
        if ECM_scaleObsPrecisionToMedian:
            if obsPrecisionWarmStartIsMedianScaled:
                _validateMedianScaledPrecisionWarmStart(
                    lambdaExpView,
                    0,
                    intervalCount,
                    wMin,
                    wMax,
                )
            else:
                _scalePrecisionMultipliersToMedian(
                    lambdaExpView,
                    0,
                    intervalCount,
                    wMin,
                    wMax,
                    medianWorkspaceView,
                )

    if processPrecisionActive:
        if processPrecExpInit is None:
            processPrecExpArr = np.ones(intervalCount, dtype=np.float32)
            _initializeProcessPrecisionFromPointwiseChanges(
                dataView,
                muncMatView,
                blockMapView,
                blockCount,
                q0View,
                <double>pad,
                <double>procNu,
                processPrecExpArr,
            )
        else:
            processPrecExpArr = np.array(processPrecExpInit, dtype=np.float32, copy=True, order="C").reshape(-1)
            if processPrecExpArr.shape[0] != intervalCount:
                raise ValueError("processPrecExpInit length must match intervalCount")
            if not np.all(np.isfinite(processPrecExpArr)):
                raise ValueError("processPrecExpInit must contain only finite values")
        if not ECM_scaleProcessPrecisionToMedian:
            np.clip(processPrecExpArr, procPrecisionMultiplierMin, procPrecisionMultiplierMax, out=processPrecExpArr)
        processPrecExp = processPrecExpArr
        processPrecExpView = processPrecExpArr
        if ECM_scaleProcessPrecisionToMedian:
            if intervalCount > 0:
                processPrecExpView[0] = <cnp.float32_t>1.0
            if processPrecisionWarmStartIsMedianScaled and processPrecExpInit is not None:
                _validateMedianScaledPrecisionWarmStart(
                    processPrecExpView,
                    1,
                    intervalCount,
                    kappaMin_,
                    kappaMax_,
                )
            else:
                _scalePrecisionMultipliersToMedian(
                    processPrecExpView,
                    1,
                    intervalCount,
                    kappaMin_,
                    kappaMax_,
                    medianWorkspaceView,
                )

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
            if intervalToBlockMap.shape[0] < intervalCount:
                raise ValueError("intervalToBlockMap length must match intervalCount")

            cforwardPassLevel(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixPluginMuncInit,
                observationStats=observationStats,
                computeStatistic=False,
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
                obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
                obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
                procPrecisionMultiplierMin=procPrecisionMultiplierMin,
                procPrecisionMultiplierMax=procPrecisionMultiplierMax,
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
                observationStats=observationStats,
                computeStatistic=False,
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
                obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
                obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
                procPrecisionMultiplierMin=procPrecisionMultiplierMin,
                procPrecisionMultiplierMax=procPrecisionMultiplierMax,
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
            "scale_obs_precision_to_median": bool(
                ECM_scaleObsPrecisionToMedian
            ),
            "scale_process_precision_to_median": bool(
                ECM_scaleProcessPrecisionToMedian
            ),
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
    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")

    q0Inv = 1.0 / q0

    observationStats = _collapseObservationStats(matrixData, matrixPluginMuncInit, <double>pad)
    observationStatsView = observationStats

    for i in range(ECM_fixedBackgroundIters):
        itersDone = i + 1
        if logIterations:
            fprintf(stderr, "\n\t[cfixedBackgroundECMLevel] iter=%zd\n", itersDone)

        for inner in range(t_innerIters):
            if i == 0 or inner > 0:
                cforwardPassLevel(
                    matrixData=matrixData,
                    matrixPluginMuncInit=matrixPluginMuncInit,
                    observationStats=observationStats,
                    computeStatistic=False,
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
                    obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
                    obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
                    procPrecisionMultiplierMin=procPrecisionMultiplierMin,
                    procPrecisionMultiplierMax=procPrecisionMultiplierMax,
                )

            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, _unusedResiduals = cbackwardPassLevel(
                matrixData=matrixData,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                chunkSize=0,
                stateSmoothed=stateSmoothed,
                stateCovarSmoothed=stateCovarSmoothed,
                lagCovSmoothed=lagCovSmoothed,
                computeResiduals=False,
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
                        activeTrackCount = <Py_ssize_t>observationStatsView[k, 4]
                        res = observationStatsView[k, 1] - <double>stateSmoothedView[k, 0]
                        obsU2 = observationStatsView[k, 2] + observationStatsView[k, 0] * (res * res + p00k)

                        w = ((<double>ECM_robustTNu) + (<double>activeTrackCount)) / ((<double>ECM_robustTNu) + obsU2)
                        if not ECM_scaleObsPrecisionToMedian:
                            w = _clampMultiplierValue(w, wMin, wMax)
                        lambdaExpView[k] = <cnp.float32_t>w
                if ECM_scaleObsPrecisionToMedian:
                    _scalePrecisionMultipliersToMedian(
                        lambdaExpView,
                        0,
                        intervalCount,
                        wMin,
                        wMax,
                        medianWorkspaceView,
                    )

            if processPrecisionActive:
                processPrecExpView[0] = <cnp.float32_t>1.0
                for k in range(intervalCount - 1):
                    b = <Py_ssize_t>blockMapView[k]
                    if b < 0 or b >= blockCount:
                        processPrecExpView[k + 1] = <cnp.float32_t>1.0
                        continue
                    x0 = <double>stateSmoothedView[k, 0]
                    y0 = <double>stateSmoothedView[k + 1, 0]
                    # note, precision reweighting is through the forward-backward mean transition only
                    # (i.e., does not consider lag-covariance or P_i). See Aravkin '13
                    delta = ((y0 - x0) * (y0 - x0)) * q0Inv
                    if delta < 0.0:
                        delta = 0.0
                    kappa_ = ((<double>procNu) + dState) / ((<double>procNu) + delta)
                    if not ECM_scaleProcessPrecisionToMedian:
                        kappa_ = _clampMultiplierValue(
                            kappa_,
                            kappaMin_,
                            kappaMax_,
                        )
                    processPrecExpView[k + 1] = <cnp.float32_t>kappa_
                if ECM_scaleProcessPrecisionToMedian:
                    _scalePrecisionMultipliersToMedian(
                        processPrecExpView,
                        1,
                        intervalCount,
                        kappaMin_,
                        kappaMax_,
                        medianWorkspaceView,
                    )

        currentNLL = (<double>cforwardPassLevel(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixPluginMuncInit,
            observationStats=observationStats,
            computeStatistic=False,
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
            returnNLL=True,
            storeNLLInD=False,
            lambdaExp=lambdaExp,
            processPrecExp=processPrecExp,
            ECM_useObsPrecisionReweighting=ECM_useObsPrecisionReweighting,
            ECM_useProcessPrecisionReweighting=ECM_useProcessPrecisionReweighting,
            obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
            obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
            procPrecisionMultiplierMin=procPrecisionMultiplierMin,
            procPrecisionMultiplierMax=procPrecisionMultiplierMax,
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
        "scale_obs_precision_to_median": bool(ECM_scaleObsPrecisionToMedian),
        "scale_process_precision_to_median": bool(
            ECM_scaleProcessPrecisionToMedian
        ),
    }
    if trackOptimizationPath:
        diagnostics["optimization_path"] = optimizationPath

    if returnIntermediates:
        with nogil:
            for k in range(intervalCount):
                for j in range(trackCount):
                    residualView[k, j] = <cnp.float32_t>(
                        (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                    )
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
    float procPrecisionMultiplierMin=0.0001,
    float procPrecisionMultiplierMax=10.0,
    bint ECM_useObsPrecisionReweighting=True,
    bint ECM_useProcessPrecisionReweighting=True,
    Py_ssize_t t_innerIters=5,
    bint returnIntermediates=False,
    bint returnDiagnostics=False,
    object lambdaExpInit=None,
    object processPrecExpInit=None,
    bint trackOptimizationPath=False,
    bint logIterations=True,
    bint ECM_scaleObsPrecisionToMedian=False,
    bint ECM_scaleProcessPrecisionToMedian=True,
    bint obsPrecisionWarmStartIsMedianScaled=False,
    bint processPrecisionWarmStartIsMedianScaled=False,
    float ECM_processRobustTNu=8.0,
):
    r"""Run the fixed-background Consenrich ECM loop with iteratively updated observation and process noise covariances.

    This routine is the fixed-background fit used by
    :func:`consenrich.core.runConsenrich`. Note, the shared background track
    has already been removed from ``matrixData`` before this step
    """

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] observationStats = None
    cdef double[:, ::1] observationStatsView
    cdef Py_ssize_t trackCount = matrixData.shape[0]
    cdef Py_ssize_t intervalCount = matrixData.shape[1]
    cdef Py_ssize_t i, k, j, inner
    cdef Py_ssize_t b
    cdef Py_ssize_t activeTrackCount
    cdef cnp.int32_t[::1] blockMapView = intervalToBlockMap
    cdef cnp.float32_t[:, ::1] dataView = matrixData
    cdef cnp.float32_t[:, ::1] muncMatView = matrixPluginMuncInit
    cdef cnp.float32_t[:, ::1] fView = matrixF
    cdef cnp.float32_t[:, ::1] q0View = matrixQ0

    # Allocate latent precision multipliers only if enabled
    cdef object lambdaExp = None
    cdef object processPrecExp = None
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] lambdaExpArr
    cdef cnp.float32_t[::1] lambdaExpView
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] processPrecExpArr
    cdef cnp.float32_t[::1] processPrecExpView
    cdef bint processPrecisionActive = ECM_useProcessPrecisionReweighting
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] medianWorkspaceArr
    cdef cnp.float32_t[::1] medianWorkspaceView

    _validateMultiplierBounds(
        <double>obsPrecisionMultiplierMin,
        <double>obsPrecisionMultiplierMax,
        True,
    )
    _validateMultiplierBounds(
        <double>procPrecisionMultiplierMin,
        <double>procPrecisionMultiplierMax,
        False,
    )
    if not isfinite(ECM_processRobustTNu) or ECM_processRobustTNu <= 0.0:
        raise ValueError("ECM_processRobustTNu must be positive and finite")
    _validateMedianScaling(
        ECM_scaleObsPrecisionToMedian,
        ECM_useObsPrecisionReweighting,
        <double>obsPrecisionMultiplierMin,
        <double>obsPrecisionMultiplierMax,
        True,
    )
    _validateMedianScaling(
        ECM_scaleProcessPrecisionToMedian,
        processPrecisionActive,
        <double>procPrecisionMultiplierMin,
        <double>procPrecisionMultiplierMax,
        False,
    )
    if ECM_scaleObsPrecisionToMedian or ECM_scaleProcessPrecisionToMedian:
        medianWorkspaceArr = np.empty(max(intervalCount, 1), dtype=np.float32)
        medianWorkspaceView = medianWorkspaceArr

    if ECM_useObsPrecisionReweighting:
        if lambdaExpInit is None:
            lambdaExpArr = np.ones(intervalCount, dtype=np.float32)
        else:
            lambdaExpArr = np.array(lambdaExpInit, dtype=np.float32, copy=True, order="C")
            if lambdaExpArr.shape[0] != intervalCount:
                raise ValueError("lambdaExpInit length must match intervalCount")
            if not np.all(np.isfinite(lambdaExpArr)):
                raise ValueError("lambdaExpInit must contain only finite values")
            if not ECM_scaleObsPrecisionToMedian:
                np.clip(lambdaExpArr, obsPrecisionMultiplierMin, obsPrecisionMultiplierMax, out=lambdaExpArr)
        lambdaExp = lambdaExpArr
        lambdaExpView = lambdaExpArr
        if ECM_scaleObsPrecisionToMedian:
            if obsPrecisionWarmStartIsMedianScaled:
                _validateMedianScaledPrecisionWarmStart(
                    lambdaExpView,
                    0,
                    intervalCount,
                    <double>obsPrecisionMultiplierMin,
                    <double>obsPrecisionMultiplierMax,
                )
            else:
                _scalePrecisionMultipliersToMedian(
                    lambdaExpView,
                    0,
                    intervalCount,
                    <double>obsPrecisionMultiplierMin,
                    <double>obsPrecisionMultiplierMax,
                    medianWorkspaceView,
                )

    if processPrecisionActive:
        if processPrecExpInit is None:
            processPrecExpArr = np.ones(intervalCount, dtype=np.float32)
            _initializeProcessPrecisionFromPointwiseChanges(
                dataView,
                muncMatView,
                blockMapView,
                blockCount,
                q0View,
                <double>pad,
                <double>ECM_processRobustTNu,
                processPrecExpArr,
            )
        else:
            processPrecExpArr = np.array(processPrecExpInit, dtype=np.float32, copy=True, order="C").reshape(-1)
            if processPrecExpArr.shape[0] != intervalCount:
                raise ValueError("processPrecExpInit length must match intervalCount")
            if not np.all(np.isfinite(processPrecExpArr)):
                raise ValueError("processPrecExpInit must contain only finite values")
        if not ECM_scaleProcessPrecisionToMedian:
            np.clip(processPrecExpArr, procPrecisionMultiplierMin, procPrecisionMultiplierMax, out=processPrecExpArr)
        processPrecExp = processPrecExpArr
        processPrecExpView = processPrecExpArr
        if ECM_scaleProcessPrecisionToMedian:
            if intervalCount > 0:
                processPrecExpView[0] = <cnp.float32_t>1.0
            if processPrecisionWarmStartIsMedianScaled and processPrecExpInit is not None:
                _validateMedianScaledPrecisionWarmStart(
                    processPrecExpView,
                    1,
                    intervalCount,
                    <double>procPrecisionMultiplierMin,
                    <double>procPrecisionMultiplierMax,
                )
            else:
                _scalePrecisionMultipliersToMedian(
                    processPrecExpView,
                    1,
                    intervalCount,
                    <double>procPrecisionMultiplierMin,
                    <double>procPrecisionMultiplierMax,
                    medianWorkspaceView,
                )

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
    cdef double p00k
    cdef double x0, x1, y0, y1
    cdef double r0, r1
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
    cdef double procNu = ECM_processRobustTNu
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
            if intervalToBlockMap.shape[0] < intervalCount:
                raise ValueError("intervalToBlockMap length must match intervalCount")
            if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
                raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
            if detQ0 == 0.0:
                raise ValueError("matrixQ0 is singular")

            cforwardPass(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixPluginMuncInit,
                observationStats=observationStats,
                computeStatistic=False,
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
                obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
                obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
                procPrecisionMultiplierMin=procPrecisionMultiplierMin,
                procPrecisionMultiplierMax=procPrecisionMultiplierMax,
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
                observationStats=observationStats,
                computeStatistic=False,
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
                obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
                obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
                procPrecisionMultiplierMin=procPrecisionMultiplierMin,
                procPrecisionMultiplierMax=procPrecisionMultiplierMax,
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
            "scale_obs_precision_to_median": bool(
                ECM_scaleObsPrecisionToMedian
            ),
            "scale_process_precision_to_median": bool(
                ECM_scaleProcessPrecisionToMedian
            ),
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
    if intervalToBlockMap.shape[0] < intervalCount:
        raise ValueError("intervalToBlockMap length must match intervalCount")
    if matrixPluginMuncInit.shape[0] != trackCount or matrixPluginMuncInit.shape[1] != intervalCount:
        raise ValueError("matrixPluginMuncInit shape must match matrixData shape")
    if detQ0 == 0.0:
        raise ValueError("matrixQ0 is singular")

    # 2x2 inverse of process noise covariance matrix
    q0Inv00 = q0_11 / detQ0
    q0Inv01 = -q0_01 / detQ0
    q0Inv10 = -q0_10 / detQ0
    q0Inv11 = q0_00 / detQ0

    observationStats = _collapseObservationStats(matrixData, matrixPluginMuncInit, <double>pad)
    observationStatsView = observationStats

    for i in range(ECM_fixedBackgroundIters):
        itersDone = i + 1
        if logIterations:
            fprintf(stderr, "\n\t[cfixedBackgroundECM] iter=%zd\n", itersDone)

        for inner in range(t_innerIters):
            # the inner block fixes g(i) and runs the forward-backward filter-smoother
            if i == 0 or inner > 0:
                cforwardPass(
                    matrixData=matrixData,
                    matrixPluginMuncInit=matrixPluginMuncInit,
                    observationStats=observationStats,
                    computeStatistic=False,
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
                    obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
                    obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
                    procPrecisionMultiplierMin=procPrecisionMultiplierMin,
                    procPrecisionMultiplierMax=procPrecisionMultiplierMax,
                )

            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, _unusedResiduals = cbackwardPass(
                matrixData=matrixData,
                matrixF=matrixF,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                chunkSize=0,
                stateSmoothed=stateSmoothed,
                stateCovarSmoothed=stateCovarSmoothed,
                lagCovSmoothed=lagCovSmoothed,
                computeResiduals=False,
            )

            # after forward-backward smoothing, update the interval-level precision multipliers (lambdaExp)
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

                        activeTrackCount = <Py_ssize_t>observationStatsView[k, 4]
                        res = observationStatsView[k, 1] - <double>stateSmoothedView[k, 0]
                        obsU2 = observationStatsView[k, 2] + observationStatsView[k, 0] * (res * res + p00k)

                        w = ((<double>ECM_robustTNu) + (<double>activeTrackCount)) / ((<double>ECM_robustTNu) + obsU2)
                        if not ECM_scaleObsPrecisionToMedian:
                            w = _clampMultiplierValue(w, wMin, wMax)

                        lambdaExpView[k] = <cnp.float32_t>w
                if ECM_scaleObsPrecisionToMedian:
                    _scalePrecisionMultipliersToMedian(
                        lambdaExpView,
                        0,
                        intervalCount,
                        wMin,
                        wMax,
                        medianWorkspaceView,
                    )

            # -----------------------------
            # update process precision multipliers kappa_ and store in processPrecExp
            # -----------------------------
            if processPrecisionActive:
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

                    r0 = y0 - ((f00 * x0) + (f01 * x1))
                    r1 = y1 - ((f10 * x0) + (f11 * x1))
                    # note, precision reweighting is through the forward-backward mean transition only
                    # (i.e., does not consider lag-covariance or P_i). See Aravkin '13
                    delta = (
                        r0 * ((q0Inv00 * r0) + (q0Inv01 * r1))
                        + r1 * ((q0Inv10 * r0) + (q0Inv11 * r1))
                    )
                    if delta < 0.0:
                        delta = 0.0

                    kappa_ = ((<double>procNu) + dState) / ((<double>procNu) + delta)
                    if not ECM_scaleProcessPrecisionToMedian:
                        kappa_ = _clampMultiplierValue(
                            kappa_,
                            kappaMin_,
                            kappaMax_,
                        )
                    processPrecExpView[k + 1] = <cnp.float32_t>kappa_
                if ECM_scaleProcessPrecisionToMedian:
                    _scalePrecisionMultipliersToMedian(
                        processPrecExpView,
                        1,
                        intervalCount,
                        kappaMin_,
                        kappaMax_,
                        medianWorkspaceView,
                    )

        currentNLL = (<double>cforwardPass(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixPluginMuncInit,
            observationStats=observationStats,
            computeStatistic=False,
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
            returnNLL=True,
            storeNLLInD=False,
            lambdaExp=lambdaExp,
            processPrecExp=processPrecExp,
            ECM_useObsPrecisionReweighting=ECM_useObsPrecisionReweighting,
            ECM_useProcessPrecisionReweighting=ECM_useProcessPrecisionReweighting,
            obsPrecisionMultiplierMin=obsPrecisionMultiplierMin,
            obsPrecisionMultiplierMax=obsPrecisionMultiplierMax,
            procPrecisionMultiplierMin=procPrecisionMultiplierMin,
            procPrecisionMultiplierMax=procPrecisionMultiplierMax,
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
        "scale_obs_precision_to_median": bool(ECM_scaleObsPrecisionToMedian),
        "scale_process_precision_to_median": bool(
            ECM_scaleProcessPrecisionToMedian
        ),
    }
    if trackOptimizationPath:
        diagnostics["optimization_path"] = optimizationPath

    if returnIntermediates:
        with nogil:
            for k in range(intervalCount):
                for j in range(trackCount):
                    residualView[k, j] = <cnp.float32_t>(
                        (<double>dataView[j, k]) - (<double>stateSmoothedView[k, 0])
                    )
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
    r"""Solve the (soft)-ROCCO optimization using a binary chain with selection and switching penalties.

    .. math::

       \max_{\ell\in\{0,1\}^{n}}
       \sum_{i=1}^{n}(s_{[i]}-\tau)\ell_{[i]}
       -\sum_{i=1}^{n-1}\gamma_{[i]}|\ell_{[i+1]}-\ell_{[i]}|.

    Here :math:`s` is ``scoresView``, :math:`\tau` is ``selectionPenalty``,
    and :math:`\gamma` is ``switchCostsView``.

    The criterion is equivalent to weighted chain fused lasso with binary labels (e.g., [MadridPadilla2018]_).

    **Intuition**

    If two 'prefixes' (bit strings up to length :math:`i`) end in the same label, any fixed continuation involves adding the same objective to both.
    *Optimal substructure*: every prefix of an optimal solution must also be optimal among prefixes with the same ending label. Otherwise, we could replace it with some other prefix to improve the objective (contradicting optimality).
    The :math:`\{0,1\}`-prefix with the lower score is suboptimal, so we only need to track the best prefix for each ending label.

    Appealing to dynamic programming gives an :math:`O(n)` solution method.

    Set :math:`D_{[1]}(0)=0` and :math:`D_{[1]}(1)=s_{[1]}-\tau`.
    For :math:`i=2,\ldots,n`,

    .. math::

       \begin{aligned}
       D_{[i]}(0) &= \max\{D_{[i-1]}(0),\,
                           D_{[i-1]}(1)-\gamma_{[i-1]}\},\\
       D_{[i]}(1) &= s_{[i]}-\tau+
                     \max\{D_{[i-1]}(1),\,
                            D_{[i-1]}(0)-\gamma_{[i-1]}\}.
       \end{aligned}

    Each maximum compares keeping the label with switching it and paying
    the :math:`\gamma_{[i-1]}` cost. Selecting interval :math:`i` also adds
    :math:`s_{[i]}-\tau`. Choose the better ending label at :math:`i=n`, then follow the stored
    predecessors backward to recover the full sequence.

    References
    ----------

    .. [HamiltonFurey2023] Hamilton, N. H. and Furey, T. S. (2023).
       ROCCO: a robust method for detection of open chromatin via convex
       optimization. Bioinformatics, 39(12), btad725.
       https://doi.org/10.1093/bioinformatics/btad725

    .. [MadridPadilla2018] Madrid Padilla, O. H., Sharpnack, J., Scott, J. G.,
       and Tibshirani, R. J. (2018). The DFS Fused Lasso: Linear-Time Denoising
       over General Graphs. Journal of Machine Learning Research, 18(176), 1--36.
       https://jmlr.org/papers/v18/16-532.html
    """

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

    # Keep one best prefix per ending label
    # First interval: D_[1](0)=0 and D_[1](1)=s_[1]-tau
    prev0Val = 0.0
    prev0Count = 0
    prev1Val = scoresView[0] - penalty_
    prev1Count = 1

    for i in range(1, n):
        switchCost = switchCostsView[i - 1]

        # Case end=0: stay at 0, or switch from 1 and pay the boundary cost.
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

        # Case end = 1: staying or switching also adds scoresView[i] - penalty_.
        stay1Val = prev1Val + scoresView[i] - penalty_
        stay1Count = prev1Count + 1
        switch1Val = prev0Val - switchCost + scoresView[i] - penalty_
        switch1Count = prev0Count + 1

        # choose whichever option gives the better value, breaking ties by fewer selected intervals
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

    # Choose the best ending label, resolving score ties by fewer selected intervals.
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
    # Following the saved winning predecessors yields the optimal sequence
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scoresArr = np.ascontiguousarray(scores, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] switchCostsArr = np.ascontiguousarray(switchCosts, dtype=np.float64).reshape(-1)
    if scoresArr.size == 0:
        raise ValueError("`scores` cannot be empty")
    if not np.all(np.isfinite(scoresArr)):
        raise ValueError("`scores` contains non-finite values")
    if not np.all(np.isfinite(switchCostsArr)):
        raise ValueError("`switchCosts` contains non-finite values")
    if scoresArr.size > 1 and switchCostsArr.size != scoresArr.size - 1:
        raise ValueError("`switchCosts` must have length len(scores) - 1")
    return _solvePenalizedChainROCCO_F64(scoresArr, switchCostsArr, selectionPenalty)


cdef inline void _updateParentConditionedState(
    double* values,
    cnp.intp_t* counts,
    int32_t* backState,
    Py_ssize_t newState,
    double value,
    Py_ssize_t count,
    Py_ssize_t sourceState,
) noexcept nogil:
    if value > values[newState] + 1.0e-12 or (
        fabs(value - values[newState]) <= 1.0e-12 and count < counts[newState]
    ):
        values[newState] = value
        counts[newState] = count
        backState[newState] = sourceState


cpdef object cSolveParentConditionedSubpeaks(
    object scores,
    object boundaryCosts,
    double selectionPenalty,
    object minRunBins,
    object requiredIndex=None,
    double runPenalty=0.0,
):
    cdef object scoresArr = np.asarray(scores, dtype=np.float64)
    cdef object costsArr = np.asarray(boundaryCosts, dtype=np.float64)
    cdef const double[:] scoresView
    cdef const double[:] costsView
    cdef cnp.ndarray valueBuffersArr
    cdef cnp.ndarray countBuffersArr
    cdef cnp.ndarray backStateArr
    cdef cnp.ndarray maskArr
    cdef double[:, ::1] valueBuffersView
    cdef cnp.intp_t[:, ::1] countBuffersView
    cdef int32_t[:, ::1] backStateView
    cdef cnp.npy_bool[::1] maskView
    cdef double* prevValues
    cdef double* newValues
    cdef double* valueSwap
    cdef cnp.intp_t* prevCounts
    cdef cnp.intp_t* newCounts
    cdef cnp.intp_t* countSwap
    cdef int32_t* backRow
    cdef Py_ssize_t n
    cdef Py_ssize_t minRunBins_
    cdef Py_ssize_t numStates
    cdef Py_ssize_t largeCount
    cdef Py_ssize_t requiredBin = -1
    cdef Py_ssize_t maxSize = np.iinfo(np.intp).max
    cdef Py_ssize_t maxBackState = np.iinfo(np.int32).max
    cdef Py_ssize_t i, state, sourceState
    cdef object requiredBinValue
    cdef double adjustedScore, transitionCost, closedValue, bestValue

    if scoresArr.ndim != 1:
        raise ValueError("`scores` must be one-dimensional")
    if costsArr.ndim != 1:
        raise ValueError("`boundaryCosts` must be one-dimensional")
    n = scoresArr.shape[0]
    if n == maxSize:
        raise OverflowError("`scores` is too large for subpeak DP storage")
    if costsArr.shape[0] != n + 1:
        raise ValueError("`boundaryCosts` must have length len(scores) + 1")
    if requiredIndex is not None:
        requiredBinValue = int(requiredIndex)
        if requiredBinValue < 0 or requiredBinValue >= n:
            raise ValueError("`requiredIndex` is outside `scores`")
        requiredBin = requiredBinValue
    minRunBins_ = min(max(int(minRunBins), 1), n)
    if n == 0:
        return np.zeros(0, dtype=bool)
    numStates = minRunBins_ + 1
    if (
        minRunBins_ > maxBackState
        or numStates > maxSize // (2 * sizeof(double))
        or n > maxSize // sizeof(int32_t) // numStates
    ):
        raise OverflowError("subpeak DP storage exceeds the addressable size")
    largeCount = n + 1
    scoresView = scoresArr
    costsView = costsArr
    valueBuffersArr = np.empty((2, numStates), dtype=np.float64)
    countBuffersArr = np.empty((2, numStates), dtype=np.intp)
    backStateArr = np.full((n, numStates), -1, dtype=np.int32)
    maskArr = np.zeros(n, dtype=bool)
    valueBuffersView = valueBuffersArr
    countBuffersView = countBuffersArr
    backStateView = backStateArr
    maskView = maskArr
    prevValues = &valueBuffersView[0, 0]
    newValues = &valueBuffersView[1, 0]
    prevCounts = &countBuffersView[0, 0]
    newCounts = &countBuffersView[1, 0]

    with nogil:
        for state in range(numStates):
            prevValues[state] = -INFINITY
            prevCounts[state] = largeCount
        prevValues[0] = 0.0
        prevCounts[0] = 0
        for i in range(n):
            adjustedScore = scoresView[i] - selectionPenalty
            transitionCost = costsView[i]
            backRow = &backStateView[i, 0]
            for state in range(numStates):
                newValues[state] = -INFINITY
                newCounts[state] = largeCount

            if i != requiredBin:
                if isfinite(prevValues[0]):
                    _updateParentConditionedState(
                        newValues, newCounts, backRow,
                        0, prevValues[0], prevCounts[0], 0,
                    )
                if isfinite(prevValues[minRunBins_]):
                    _updateParentConditionedState(
                        newValues, newCounts, backRow,
                        0, prevValues[minRunBins_] - transitionCost,
                        prevCounts[minRunBins_], minRunBins_,
                    )

            if isfinite(prevValues[0]):
                _updateParentConditionedState(
                    newValues, newCounts, backRow,
                    1,
                    prevValues[0] - transitionCost - runPenalty + adjustedScore,
                    prevCounts[0] + 1, 0,
                )
            for state in range(1, minRunBins_):
                if isfinite(prevValues[state]):
                    _updateParentConditionedState(
                        newValues, newCounts, backRow,
                        state + 1, prevValues[state] + adjustedScore,
                        prevCounts[state] + 1, state,
                    )
            if isfinite(prevValues[minRunBins_]):
                _updateParentConditionedState(
                    newValues, newCounts, backRow,
                    minRunBins_, prevValues[minRunBins_] + adjustedScore,
                    prevCounts[minRunBins_] + 1, minRunBins_,
                )

            valueSwap = prevValues
            prevValues = newValues
            newValues = valueSwap
            countSwap = prevCounts
            prevCounts = newCounts
            newCounts = countSwap

        bestValue = prevValues[0]
        closedValue = prevValues[minRunBins_] - costsView[n]
        state = 0
        if closedValue > bestValue or (
            closedValue == bestValue and prevCounts[minRunBins_] < prevCounts[0]
        ):
            bestValue = closedValue
            state = minRunBins_

    if not isfinite(bestValue):
        raise RuntimeError("parent-conditioned subpeak DP found no feasible path")
    with nogil:
        for i in range(n - 1, -1, -1):
            if state > 0:
                maskView[i] = True
            sourceState = backStateView[i, state]
            if sourceState < 0:
                break
            state = sourceState
    if sourceState < 0:
        raise RuntimeError("parent-conditioned subpeak DP found an incomplete path")
    return maskArr


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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scoresArr = np.ascontiguousarray(scores, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] switchCostsArr = np.ascontiguousarray(switchCosts, dtype=np.float64).reshape(-1)
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scoresArr = np.ascontiguousarray(scores, dtype=np.float64).reshape(-1)
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
    if shifted < tiny and mode != __TRANSFORM_MODE_ASINH and mode != __TRANSFORM_MODE_GENERALIZED_LOG:
        shifted = tiny
    if mode == __TRANSFORM_MODE_LOG:
        return outputScale / shifted
    if mode == __TRANSFORM_MODE_SQRT or mode == __TRANSFORM_MODE_ANSCOMBE:
        return outputScale / (2.0 * inputScale * sqrt(fmax(shifted / inputScale, tiny)))
    if mode == __TRANSFORM_MODE_ASINH:
        return outputScale / hypot(inputScale, shifted)
    if mode == __TRANSFORM_MODE_ASINH_SQRT:
        root = sqrt(shifted)
        return (0.5 * outputScale / root) / hypot(inputScale, root)
    if mode == __TRANSFORM_MODE_GENERALIZED_LOG:
        u = shifted / inputScale
        return (outputScale / hypot(u, shape)) / inputScale
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
    scales = np.ascontiguousarray(scaleFactors, dtype=np.float64).reshape(-1)
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t width = min(max(window, 1), x.shape[0])
    if width <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    return cWindowSums(x, width) / <double>width


def cEstimateEffectiveSampleSize(
    object values,
    int maxLag,
    object activeMask=None,
    bint logPositive=False,
    int windowIntervals=0,
):
    r"""Positive-autocorrelation effective sample size scan."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = np.array(values, dtype=np.float64, order="C", copy=True).reshape(-1)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] activeArr
    cdef const cnp.uint8_t[::1] activeView = None
    cdef double[::1] covarianceSums, pairCounts
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t activeCount = 0
    cdef Py_ssize_t pairCount
    cdef Py_ssize_t i, lag, maxLag_, fftSize, lagsUsed = 0
    cdef double origin = 0.0
    cdef double mean = 0.0
    cdef double correction = 0.0
    cdef double var = 0.0
    cdef double cov, rho, value, taper, fftTolerance = 0.0, tau = 1.0
    cdef bint useActiveMask = activeMask is not None
    cdef bint strictValues = useActiveMask or logPositive or windowIntervals != 0
    cdef bint useFFT = False
    cdef bint badRho = False
    cdef int windowIntervals_ = windowIntervals
    if windowIntervals_ < 0:
        raise ValueError("windowIntervals must be nonnegative")
    if useActiveMask:
        activeArr = np.ascontiguousarray(activeMask, dtype=np.uint8).reshape(-1)
        if activeArr.shape[0] != n:
            raise ValueError("activeMask length must match values length")
        activeView = activeArr
    for i in range(n):
        if useActiveMask and activeView[i] == 0:
            x[i] = 0.0
            continue
        value = x[i]
        if logPositive:
            if (not isfinite(value)) or value <= 0.0:
                raise ValueError("active values must be positive finite")
            value = log(value)
        elif strictValues and not isfinite(value):
            raise ValueError("active values must be finite")
        if activeCount == 0:
            origin = value
        x[i] = value - origin
        compensatedSumAdd(x[i], &mean, &correction)
        activeCount += 1
    if activeCount < 2:
        return float(activeCount), 1.0, 0
    mean = (mean + correction) / <double>activeCount
    correction = 0.0
    with nogil:
        for i in range(n):
            if useActiveMask and activeView[i] == 0:
                continue
            x[i] -= mean
            compensatedSumAdd(x[i] * x[i], &var, &correction)
    var = (var + correction) / <double>activeCount
    if not isfinite(var):
        if strictValues:
            raise ValueError("variance estimate must be finite")
        return float(activeCount), 1.0, 0
    if var <= 2.2250738585072014e-308:
        return float(activeCount), 1.0, 0
    maxLag_ = max(1, min(int(maxLag), n - 1))
    if windowIntervals_ > 0 and windowIntervals_ - 1 < maxLag_:
        maxLag_ = windowIntervals_ - 1
    if windowIntervals_ > 0 and maxLag_ >= 64:
        fftSize = next_fast_len(2 * n - 1)
        useFFT = <double>n * maxLag_ >= 8.0 * fftSize * log2(<double>fftSize)
        if useFFT:
            spectrum = rfft(x, n=fftSize)
            covarianceSums = irfft(spectrum * spectrum.conj(), n=fftSize)[:maxLag_ + 1].copy()
            if useActiveMask:
                spectrum = rfft(activeArr != 0, n=fftSize)
                pairCounts = np.rint(irfft(spectrum * spectrum.conj(), n=fftSize)[:maxLag_ + 1]).copy()
            fftTolerance = 32.0 * 2.2204460492503131e-16 * var * n * log2(<double>fftSize)
    with nogil:
        for lag in range(1, maxLag_ + 1):
            if useFFT:
                cov = covarianceSums[lag]
                pairCount = <Py_ssize_t>pairCounts[lag] if useActiveMask else n - lag
                if pairCount <= 0:
                    continue
            if not useFFT or fabs(cov) <= fftTolerance:
                cov = 0.0
                correction = 0.0
                pairCount = 0
                for i in range(n - lag):
                    if useActiveMask and (activeView[i] == 0 or activeView[i + lag] == 0):
                        continue
                    compensatedSumAdd(x[i] * x[i + lag], &cov, &correction)
                    pairCount += 1
                cov += correction
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
        if strictValues:
            raise ValueError("autocorrelation estimate must be finite")
    if tau < 1.0:
        tau = 1.0
    if windowIntervals_ > 0 and tau > <double>windowIntervals_:
        tau = <double>windowIntervals_
    return float(activeCount / tau), float(tau), int(lagsUsed)


cdef inline uint64_t _stationaryBootstrapBoundedIndex(
    bitgen_t* bitGenerator,
    uint64_t bound,
) noexcept nogil:
    cdef uint64_t threshold = (<uint64_t>0 - bound) % bound
    cdef uint64_t value

    while True:
        value = bitGenerator.next_uint64(bitGenerator.state)
        if value >= threshold:
            return value % bound


cdef inline Py_ssize_t _stationaryBootstrapSegmentIndex(
    const int64_t* segmentOffsetsPtr,
    Py_ssize_t segmentCount,
    Py_ssize_t templateIndex,
) noexcept nogil:
    cdef Py_ssize_t lower = 0
    cdef Py_ssize_t upper = segmentCount
    cdef Py_ssize_t middle

    while lower < upper:
        middle = lower + (upper - lower) // 2
        if templateIndex < segmentOffsetsPtr[middle + 1]:
            upper = middle
        else:
            lower = middle + 1
    return lower


cdef inline Py_ssize_t _stationaryBootstrapLocalIndex(
    bitgen_t* bitGenerator,
    Py_ssize_t outputIndex,
    Py_ssize_t segmentStart,
    Py_ssize_t segmentStop,
    Py_ssize_t radius,
) noexcept nogil:
    cdef Py_ssize_t leftRadius = outputIndex - segmentStart
    cdef Py_ssize_t rightRadius = segmentStop - outputIndex - 1
    cdef Py_ssize_t lower
    cdef Py_ssize_t upper

    if leftRadius > radius:
        leftRadius = radius
    if rightRadius > radius:
        rightRadius = radius
    lower = outputIndex - leftRadius
    upper = outputIndex + rightRadius + 1
    return lower + <Py_ssize_t>_stationaryBootstrapBoundedIndex(
        bitGenerator,
        <uint64_t>(upper - lower),
    )


cdef void _stationaryNullBootstrapKernel(
    const double* templatePtr,
    const int64_t* segmentOffsetsPtr,
    const double* coverageWeightsPtr,
    double* outPtr,
    Py_ssize_t n,
    Py_ssize_t segmentCount,
    double meanBlockLength,
    double restartProbability,
    Py_ssize_t maxLocalRadiusIntervals,
    bitgen_t* bitGenerator,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t outputSegment
    cdef Py_ssize_t outputStart
    cdef Py_ssize_t outputStop
    cdef Py_ssize_t sourceSegment
    cdef Py_ssize_t sourceStart
    cdef Py_ssize_t sourceStop
    cdef Py_ssize_t templateIndex
    cdef Py_ssize_t segmentLength
    cdef Py_ssize_t localRadius
    cdef bint useLocalRadius = maxLocalRadiusIntervals >= 0
    cdef double localRadiusValue
    cdef double coverageTotal = 0.0
    cdef double drawWeightedSum = 0.0
    cdef double drawMean

    for outputSegment in range(segmentCount):
        outputStart = segmentOffsetsPtr[outputSegment]
        outputStop = segmentOffsetsPtr[outputSegment + 1]
        if useLocalRadius:
            sourceStart = outputStart
            sourceStop = outputStop
            segmentLength = outputStop - outputStart
            if segmentLength <= 1:
                localRadius = 0
            elif meanBlockLength >= <double>segmentLength:
                localRadius = segmentLength - 1
            else:
                localRadiusValue = ceil(
                    sqrt(meanBlockLength * <double>segmentLength)
                )
                if localRadiusValue >= <double>(segmentLength - 1):
                    localRadius = segmentLength - 1
                else:
                    localRadius = <Py_ssize_t>localRadiusValue
            if localRadius > maxLocalRadiusIntervals:
                localRadius = maxLocalRadiusIntervals
            templateIndex = _stationaryBootstrapLocalIndex(
                bitGenerator,
                outputStart,
                sourceStart,
                sourceStop,
                localRadius,
            )
        else:
            templateIndex = <Py_ssize_t>_stationaryBootstrapBoundedIndex(
                bitGenerator,
                <uint64_t>n,
            )
            sourceSegment = _stationaryBootstrapSegmentIndex(
                segmentOffsetsPtr,
                segmentCount,
                templateIndex,
            )
            sourceStart = segmentOffsetsPtr[sourceSegment]
            sourceStop = segmentOffsetsPtr[sourceSegment + 1]
        outPtr[outputStart] = templatePtr[templateIndex]
        drawWeightedSum += coverageWeightsPtr[outputStart] * outPtr[outputStart]
        coverageTotal += coverageWeightsPtr[outputStart]
        for i in range(outputStart + 1, outputStop):
            if bitGenerator.next_double(bitGenerator.state) < restartProbability:
                if useLocalRadius:
                    templateIndex = _stationaryBootstrapLocalIndex(
                        bitGenerator,
                        i,
                        sourceStart,
                        sourceStop,
                        localRadius,
                    )
                else:
                    templateIndex = <Py_ssize_t>_stationaryBootstrapBoundedIndex(
                        bitGenerator,
                        <uint64_t>n,
                    )
                    sourceSegment = _stationaryBootstrapSegmentIndex(
                        segmentOffsetsPtr,
                        segmentCount,
                        templateIndex,
                    )
                    sourceStart = segmentOffsetsPtr[sourceSegment]
                    sourceStop = segmentOffsetsPtr[sourceSegment + 1]
            else:
                templateIndex += 1
                if templateIndex == sourceStop:
                    templateIndex = sourceStart
            outPtr[i] = templatePtr[templateIndex]
            drawWeightedSum += coverageWeightsPtr[i] * outPtr[i]
            coverageTotal += coverageWeightsPtr[i]

    drawMean = drawWeightedSum / coverageTotal
    for i in range(n):
        outPtr[i] -= drawMean


cpdef object cStationaryNullBootstrapDraw(
    object template,
    object segmentOffsets,
    object coverageWeights,
    double meanBlockLength,
    object rng,
    Py_ssize_t maxLocalRadiusIntervals=-1,
):
    cdef object templateObject = np.asarray(template, dtype=np.float64)
    cdef object segmentOffsetsObject = np.asarray(segmentOffsets, dtype=np.int64)
    cdef object coverageWeightsObject = np.asarray(coverageWeights, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] templateArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] segmentOffsetsArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] coverageWeightsArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] out
    cdef const double[::1] templateView
    cdef const int64_t[::1] segmentOffsetsView
    cdef const double[::1] coverageWeightsView
    cdef double[::1] outView
    cdef Py_ssize_t n
    cdef Py_ssize_t segmentCount
    cdef Py_ssize_t i
    cdef double restartProbability
    cdef object bitGeneratorObject
    cdef object capsule
    cdef bitgen_t* bitGenerator

    if templateObject.ndim != 1:
        raise ValueError("template must be one-dimensional")
    if segmentOffsetsObject.ndim != 1:
        raise ValueError("segmentOffsets must be one-dimensional")
    if coverageWeightsObject.ndim != 1:
        raise ValueError("coverageWeights must be one-dimensional")
    templateArr = np.ascontiguousarray(templateObject, dtype=np.float64)
    segmentOffsetsArr = np.ascontiguousarray(segmentOffsetsObject, dtype=np.int64)
    coverageWeightsArr = np.ascontiguousarray(
        coverageWeightsObject,
        dtype=np.float64,
    )
    n = templateArr.shape[0]
    if n == 0:
        raise ValueError("template must be non-empty")
    if coverageWeightsArr.shape[0] != n:
        raise ValueError("template and coverageWeights must have the same length")
    if segmentOffsetsArr.shape[0] < 2:
        raise ValueError("segmentOffsets must define at least one segment")
    segmentCount = segmentOffsetsArr.shape[0] - 1
    templateView = templateArr
    segmentOffsetsView = segmentOffsetsArr
    coverageWeightsView = coverageWeightsArr
    if segmentOffsetsView[0] != 0 or segmentOffsetsView[segmentCount] != n:
        raise ValueError("segmentOffsets must have 0 and template length as endpoints")
    for i in range(segmentCount):
        if segmentOffsetsView[i] >= segmentOffsetsView[i + 1]:
            raise ValueError("segmentOffsets must be strictly increasing")
    for i in range(n):
        if not isfinite(templateView[i]):
            raise ValueError("template values must be finite")
        if not isfinite(coverageWeightsView[i]) or coverageWeightsView[i] <= 0.0:
            raise ValueError("coverageWeights must be finite and positive")
    if not isfinite(meanBlockLength) or meanBlockLength < 1.0:
        raise ValueError("meanBlockLength must be finite and at least 1")
    if maxLocalRadiusIntervals < -1:
        raise ValueError("maxLocalRadiusIntervals must be -1 or non-negative")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    bitGeneratorObject = rng.bit_generator
    capsule = bitGeneratorObject.capsule
    if not PyCapsule_IsValid(capsule, "BitGenerator"):
        raise ValueError("rng has an invalid BitGenerator capsule")
    bitGenerator = <bitgen_t*>PyCapsule_GetPointer(capsule, "BitGenerator")
    if bitGenerator == NULL:
        raise ValueError("rng has an invalid BitGenerator capsule")

    restartProbability = 1.0 / meanBlockLength
    out = np.empty(n, dtype=np.float64)
    outView = out
    with bitGeneratorObject.lock:
        with nogil:
            _stationaryNullBootstrapKernel(
                &templateView[0],
                &segmentOffsetsView[0],
                &coverageWeightsView[0],
                &outView[0],
                n,
                segmentCount,
                meanBlockLength,
                restartProbability,
                maxLocalRadiusIntervals,
                bitGenerator,
            )
    return out


def cSelectedCoordinateRunBounds(
    const cnp.npy_bool[:] mask not None,
    const int64_t[:] intervals not None,
    const int64_t[:] ends not None):
    r"""Fast finder: runs of selected genomic intervals"""

    cdef Py_ssize_t n = mask.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t runStart
    cdef list runs = []
    if intervals.shape[0] != n or ends.shape[0] != n:
        raise ValueError("`intervals`, `ends`, and `mask` must match length")
    while i < n:
        if mask[i] == 0:
            i += 1
            continue
        runStart = i
        while (
            i + 1 < n
            and mask[i + 1] != 0
            and ends[i] == intervals[i + 1]
        ):
            i += 1
        runs.append((runStart, i))
        i += 1
    return runs


cpdef tuple cBooleanRunBounds(object above, int maxGapBins=0):
    """Return start/end arrays for true-runs, optionally bridging small false gaps."""
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] arr = np.ascontiguousarray(above, dtype=np.uint8).reshape(-1)
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] scoreArr = np.ascontiguousarray(scores, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] scaleArr = np.ascontiguousarray(scales, dtype=np.int64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] thresholdArr = np.ascontiguousarray(thresholds, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] nullScaleArr = np.ascontiguousarray(nullScales, dtype=np.float64).reshape(-1)
    if thresholdArr.shape[0] != nullScaleArr.shape[0]:
        raise ValueError("thresholds and nullScales must have the same length")

    cdef Py_ssize_t n = scoreArr.shape[0]
    cdef Py_ssize_t scaleCount = scaleArr.shape[0]
    cdef Py_ssize_t viewCount = thresholdArr.shape[0]
    cdef int minRun = minRunBins if minRunBins > 1 else 1
    cdef int gap = maxGapBins if maxGapBins > 0 else 0
    cdef int cap = maxSegmentsPerView if maxSegmentsPerView > 0 else 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] smoothArr
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] suffixArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] excessArr
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
    cdef Py_ssize_t w, scratchWidth
    cdef _doubleDouble integratedDD
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
    scratchWidth = min(n, max(1, int(np.max(scaleArr))))
    suffixArr = np.empty((scratchWidth, 2), dtype=np.float64)
    smoothArr = np.empty(n, dtype=np.float64)
    excessArr = np.empty(n, dtype=np.float64)

    for si in range(scaleCount):
        w = scaleArr[si]
        if w < 1:
            w = 1
        if w > n:
            w = n
        if w <= 1:
            smoothArr[:] = scoreArr
        else:
            with nogil:
                _windowSums(
                    <double*>scoreArr.data, <double*>smoothArr.data,
                    n, w, (w - 1) // 2, False, <_doubleDouble*>suffixArr.data,
                )
                for i in range(n):
                    smoothArr[i] /= <double>w

        for vi in range(viewCount):
            threshold = thresholdArr[vi]
            nullScale = nullScaleArr[vi]
            if nullScale < DBL_MIN:
                nullScale = DBL_MIN
            for i in range(n):
                value = (scoreArr[i] - threshold) / nullScale
                if value < 0.0:
                    value = 0.0
                excessArr[i] = value

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
                maxValue = 0.0
                integratedDD = _rangeSum(
                    <double*>excessArr.data, runStarts[r], runEnds[r] + 1, &maxValue,
                )
                integrated = integratedDD.high + integratedDD.low
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

def cbackgroundWeightedStats(object residualMatrix, object invVarMatrix):
    r"""Column-wise background sufficient statistics with a nogil inner loop."""
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] residualArr = np.ascontiguousarray(residualMatrix, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] invArr = np.ascontiguousarray(invVarMatrix, dtype=np.float32)
    if invArr.shape[0] != residualArr.shape[0] or invArr.shape[1] != residualArr.shape[1]:
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
    if invArr.shape[0] != residualArr.shape[0] or invArr.shape[1] != residualArr.shape[1]:
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] x = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    cdef Py_ssize_t width = min(max(window, 1), x.shape[0])
    if width <= 1:
        return np.asarray(x, dtype=np.float64).copy()
    cdef Py_ssize_t leftOffset = width // 2 if width <= 256 else (width - 1) // 2
    return cWindowSums(x, width, leftOffset=leftOffset) / <double>width


def cbooleanRunBounds(object above, int maxGapBins=0):
    r"""Run bounds for boolean threshold tracks, optionally bridging short gaps."""
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] flagsArr = np.ascontiguousarray(above, dtype=np.uint8).reshape(-1)
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] xArr = np.ascontiguousarray(state, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vArr = np.ascontiguousarray(variance, dtype=np.float64).reshape(-1)
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

cdef inline double stateShrinkResponsibilities(
    double x,
    double v,
    double logNullPrior,
    const double* tau,
    const double* logSlabPrior,
    Py_ssize_t slabCount,
    double* responsibility,
    double* nullProbability,
    bint likelihoodRequired,
    Py_ssize_t* dominantSlab,
) noexcept nogil:
    cdef Py_ssize_t j, maxIndex = -1
    cdef double x2 = x * x
    cdef double logNull = logNullPrior - 0.5 * (
        log(2.0 * __PI_DOUBLE) + log(v) + x2 / v
    )
    cdef double maxLog = logNull
    cdef double vPlusTau, logValue, denomSum, inverseDenom, nullWeight
    for j in range(slabCount):
        vPlusTau = v + tau[j]
        logValue = logSlabPrior[j] - 0.5 * (
            log(2.0 * __PI_DOUBLE) + log(vPlusTau) + x2 / vPlusTau
        )
        responsibility[j] = logValue
        if logValue > maxLog:
            maxLog = logValue
            maxIndex = j
    nullWeight = exp(logNull - maxLog)
    denomSum = nullWeight
    for j in range(slabCount):
        responsibility[j] = exp(responsibility[j] - maxLog)
        denomSum += responsibility[j]
    inverseDenom = 1.0 / denomSum
    nullProbability[0] = nullWeight * inverseDenom
    for j in range(slabCount):
        responsibility[j] *= inverseDenom
    if dominantSlab != NULL:
        dominantSlab[0] = maxIndex
    if likelihoodRequired:
        return maxLog + log(denomSum)
    return 0.0


cpdef tuple cstateShrinkMixtureEMStepPrepared(
    object state,
    object variance,
    double priorSpikeProp,
    object slabVariance,
    object logSlabPrior,
    int blockSize=1,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] xArr = np.ascontiguousarray(state, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vArr = np.ascontiguousarray(variance, dtype=np.float64).reshape(-1)
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
    cdef double x, v, weight, logDenom, nullProbability
    cdef double resp, tau2, slabShrinkage, slabMean
    cdef double slabPosteriorVariance, logNullPrior
    cdef double totalWeight = 0.0
    cdef double nullMass = 0.0
    cdef double logLikelihood = 0.0
    cdef Py_ssize_t finiteCount = 0
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

    respScratch = <double*>malloc(slabCount * sizeof(double))
    if respScratch == NULL:
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
                            logDenom = stateShrinkResponsibilities(
                                x, v, logNullPrior, &tauView[0],
                                &logSlabPriorView[0], slabCount, respScratch,
                                &nullProbability, True, NULL,
                            )
                            nullMass += weight * nullProbability
                            for j in range(slabCount):
                                resp = respScratch[j]
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
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] xArr = np.ascontiguousarray(state, dtype=np.float64).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] vArr = np.ascontiguousarray(variance, dtype=np.float64).reshape(-1)
    cdef object tauObj = np.asarray(slabVariance, dtype=np.float64)
    cdef object logSlabPriorObj = np.asarray(logSlabPrior, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] tauArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logSlabPriorArr
    cdef Py_ssize_t n = xArr.shape[0]
    cdef Py_ssize_t slabCount
    cdef double pi0 = priorSpikeProp
    cdef double logNullPrior, x, v, resp
    cdef double nullProb, tau2, slabShrinkage, slabMean
    cdef double slabPosteriorVariance, shrunk, postVariance, posteriorSd
    cdef double slabPosteriorWeight, posteriorWeight, centeredMean, meanOrigin, meanOffset
    cdef double badTau = 0.0
    cdef Py_ssize_t i, j, dominantSlab
    cdef double* respScratch = NULL
    cdef double* componentMeanScratch = NULL

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

    respScratch = <double*>malloc(2 * slabCount * sizeof(double))
    if respScratch == NULL:
        raise MemoryError()
    componentMeanScratch = respScratch + slabCount
    try:
        with nogil:
            for i in range(n):
                x = xView[i]
                v = vView[i]
                if _state_shrink_valid(x, v):
                    v = _state_shrink_safe_variance(v)
                    stateShrinkResponsibilities(
                        x, v, logNullPrior, &tauView[0], &logSlabPriorView[0],
                        slabCount, respScratch, &nullProb, False, &dominantSlab,
                    )
                    meanOrigin = 0.0
                    if dominantSlab >= 0:
                        meanOrigin = tauView[dominantSlab] / (tauView[dominantSlab] + v) * x
                    meanOffset = -nullProb * meanOrigin
                    postVariance = 0.0
                    slabPosteriorWeight = 0.0
                    for j in range(slabCount):
                        resp = respScratch[j]
                        tau2 = tauView[j]
                        slabShrinkage = tau2 / (tau2 + v)
                        slabMean = slabShrinkage * x
                        slabPosteriorVariance = slabShrinkage * v
                        slabPosteriorWeight += resp
                        componentMeanScratch[j] = slabMean - meanOrigin
                        meanOffset += resp * componentMeanScratch[j]
                        postVariance += resp * slabPosteriorVariance
                    posteriorWeight = nullProb + slabPosteriorWeight
                    meanOffset /= posteriorWeight
                    shrunk = meanOrigin + meanOffset
                    postVariance += nullProb * shrunk * shrunk
                    for j in range(slabCount):
                        centeredMean = componentMeanScratch[j] - meanOffset
                        postVariance += respScratch[j] * centeredMean * centeredMean
                    postVariance /= posteriorWeight
                    if (not isfinite(postVariance)) or postVariance <= 1.0e-12:
                        postVariance = 1.0e-12
                    posteriorSd = sqrt(postVariance)
                    shrunkView[i] = <cnp.float32_t>shrunk
                    posteriorSdView[i] = <cnp.float32_t>posteriorSd
                    spikePropView[i] = <cnp.float32_t>nullProb
                    if slabPosteriorWeight > 1.0e-12:
                        slabMeanView[i] = <cnp.float32_t>(shrunk * posteriorWeight / slabPosteriorWeight)
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
    return shrunkArr, posteriorSdArr, spikePropArr, slabMeanArr, slabWeightArr


def cWriteBedGraph(
    object path,
    str chromosome not None,
    const int64_t[:] starts not None,
    const int64_t[:] ends not None,
    const real_t[:] values not None,
    bint append=False,
):
    cdef Py_ssize_t n = starts.shape[0]
    cdef Py_ssize_t i
    cdef bytes pathBytes = os.fsencode(path)
    cdef bytes chromosomeBytes = chromosome.encode("utf-8")
    cdef const char* pathData = pathBytes
    cdef const char* chromosomeData = chromosomeBytes
    cdef const char* fileMode = "ab" if append else "wb"
    cdef size_t chromosomeLength = len(chromosomeBytes)
    cdef size_t rowCapacity = chromosomeLength + 400
    cdef size_t bufferCapacity = max(<size_t>1048576, rowCapacity + 1)
    cdef size_t used = 0
    cdef size_t available
    cdef int formattedLength
    cdef int errorNumber = 0
    cdef int closeStatus
    cdef char* buffer = NULL
    cdef FILE* handle = NULL

    if ends.shape[0] != n or values.shape[0] != n:
        raise ValueError("starts, ends, and values must have equal lengths")
    if not chromosomeBytes:
        raise ValueError("chromosome must be nonempty without quotes or row delimiters")
    for i in range(chromosomeLength):
        if chromosomeData[i] in (9, 13, 10, 34, 0):
            raise ValueError("chromosome must be nonempty without quotes or row delimiters")
    for i in range(len(pathBytes)):
        if pathData[i] == 0:
            raise ValueError("path must not contain a null byte")
    if localeconv().decimal_point[0] != 46 or localeconv().decimal_point[1] != 0:
        raise ValueError("native formatting requires a period decimal separator")
    if fegetround() != FE_TONEAREST:
        raise ValueError("native formatting requires round-to-nearest arithmetic")
    with nogil:
        for i in range(n):
            if starts[i] < 0 or ends[i] <= starts[i]:
                with gil:
                    raise ValueError(f"invalid interval at row {i}")
            if i > 0 and starts[i] < ends[i - 1]:
                with gil:
                    raise ValueError(f"unsorted or overlapping interval at row {i}")
            if not isfinite(values[i]):
                with gil:
                    raise ValueError(f"non-finite value at row {i}")
    buffer = <char*>malloc(bufferCapacity)
    if buffer == NULL:
        raise MemoryError()
    try:
        with nogil:
            handle = fopen(pathData, fileMode)
            if handle == NULL:
                errorNumber = errno
        if handle == NULL:
            raise OSError(errorNumber, "could not open bedGraph output", path)
        with nogil:
            for i in range(n):
                if bufferCapacity - used <= rowCapacity:
                    if fwrite(buffer, 1, used, handle) != used:
                        errorNumber = errno if errno != 0 else EIO
                        break
                    used = 0
                memcpy(buffer + used, chromosomeData, chromosomeLength)
                available = bufferCapacity - used - chromosomeLength
                formattedLength = snprintf(
                    buffer + used + chromosomeLength,
                    available,
                    "\t%lld\t%lld\t%.5f\n",
                    <long long>starts[i],
                    <long long>ends[i],
                    <double>values[i],
                )
                if formattedLength < 0:
                    errorNumber = errno if errno != 0 else EIO
                    break
                if <size_t>formattedLength >= available:
                    errorNumber = EOVERFLOW
                    break
                used += chromosomeLength + <size_t>formattedLength
            if errorNumber == 0 and used > 0:
                if fwrite(buffer, 1, used, handle) != used:
                    errorNumber = errno if errno != 0 else EIO
            closeStatus = fclose(handle)
            handle = NULL
            if closeStatus != 0 and errorNumber == 0:
                errorNumber = errno if errno != 0 else EIO
        if errorNumber != 0:
            raise OSError(errorNumber, "could not write bedGraph output", path)
    finally:
        if handle != NULL:
            fclose(handle)
        free(buffer)
