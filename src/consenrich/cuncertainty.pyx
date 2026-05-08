# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3

"""Cython kernels for state-uncertainty calibration."""

import numpy as np
cimport cython
cimport numpy as cnp

from libc.math cimport fabs, exp, log, sqrt, isfinite, NAN

IF USE_OPENMP:
    from cython.parallel cimport prange

cnp.import_array()


ctypedef fused real_t:
    cnp.float32_t
    cnp.float64_t


cdef inline double _clip_double(double value, double lower, double upper) noexcept nogil:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


cdef inline double _max_floor(double value, double floorValue) noexcept nogil:
    if value != value:
        return value
    if value < floorValue:
        return floorValue
    return value


cpdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] cmakeFoldMasks(
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t blockLen,
    Py_ssize_t folds,
    Py_ssize_t holdoutCount,
    long seed,
):
    if folds < 2:
        raise ValueError("uncertainty calibration requires at least two folds")
    if holdoutCount < 1:
        raise ValueError("uncertainty calibration requires at least one held-out replicate")
    if m < 1 or n < 1 or blockLen < 1:
        raise ValueError("invalid uncertainty calibration mask dimensions")

    cdef Py_ssize_t blockCount = (n + blockLen - 1) // blockLen
    rng = np.random.default_rng(int(seed))
    blockOrder = rng.permutation(blockCount).astype(np.int32, copy=False)
    blockFoldArr = np.empty(blockCount, dtype=np.int32)
    blockFoldArr[blockOrder] = np.arange(blockCount, dtype=np.int32) % int(folds)
    repsByBlockArr = np.empty((blockCount, holdoutCount), dtype=np.intp)
    cdef Py_ssize_t block
    for block in range(blockCount):
        repsByBlockArr[block, :] = rng.choice(m, size=holdoutCount, replace=False)

    cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] masks = np.ones(
        (folds, m, n), dtype=np.uint8
    )
    cdef cnp.uint8_t[:, :, ::1] masksView = masks
    cdef cnp.int32_t[::1] blockFold = blockFoldArr
    cdef Py_ssize_t[:, ::1] repsByBlock = repsByBlockArr
    cdef Py_ssize_t start, end, i, h, rep, fold

    with nogil:
        for block in range(blockCount):
            start = block * blockLen
            end = start + blockLen
            if end > n:
                end = n
            fold = blockFold[block]
            for h in range(holdoutCount):
                rep = repsByBlock[block, h]
                for i in range(start, end):
                    masksView[fold, rep, i] = <cnp.uint8_t>0
    return masks


cpdef tuple cfeatureMatrix(
    real_t[::1] state,
    real_t[::1] stateVar,
    real_t[:, ::1] matrixMunc,
    double highSignalQuantile,
    double positiveFloor,
    double madNormalScale,
    double featureScaleFloor,
):
    cdef Py_ssize_t n = state.shape[0]
    cdef Py_ssize_t m = matrixMunc.shape[0]
    if stateVar.shape[0] != n or matrixMunc.shape[1] != n:
        raise ValueError("feature inputs have inconsistent dimensions")

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] raw = np.empty((n, 5), dtype=np.float64)
    cdef double[:, ::1] rawView = raw
    cdef Py_ssize_t i, j
    cdef double obsSum, obsCount, obsMean, sv, st, prev, slope, absState
    cdef double highThresh = float(
        np.nanquantile(np.abs(np.asarray(state)), highSignalQuantile)
    )

    with nogil:
        for i in range(n):
            obsSum = 0.0
            obsCount = 0.0
            for j in range(m):
                obsMean = <double>matrixMunc[j, i]
                if obsMean == obsMean:
                    obsSum += obsMean
                    obsCount += 1.0
            if obsCount > 0.0:
                obsMean = obsSum / obsCount
            else:
                obsMean = NAN
            obsMean = _max_floor(obsMean, positiveFloor)

            sv = _max_floor(<double>stateVar[i], positiveFloor)
            st = <double>state[i]
            if i == 0:
                slope = 0.0
            else:
                prev = <double>state[i - 1]
                slope = st - prev

            absState = fabs(st)
            rawView[i, 0] = log(sv)
            rawView[i, 1] = log(obsMean)
            rawView[i, 2] = absState
            rawView[i, 3] = fabs(slope)
            rawView[i, 4] = 1.0 if absState > highThresh else 0.0

    center = np.nanmedian(raw, axis=0)
    scale = np.nanmedian(np.abs(raw - center[None, :]), axis=0) * madNormalScale
    scale = np.where(np.isfinite(scale) & (scale > featureScaleFloor), scale, 1.0)

    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] X = np.empty((n, 6), dtype=np.float64)
    cdef double[:, ::1] xView = X
    cdef double[::1] centerView = center
    cdef double[::1] scaleView = scale
    cdef double value

    with nogil:
        for i in range(n):
            xView[i, 0] = 1.0
            for j in range(5):
                value = (rawView[i, j] - centerView[j]) / scaleView[j]
                if not isfinite(value):
                    value = 0.0
                xView[i, j + 1] = value
    return X, center, scale


cpdef tuple cextractHeldoutScores(
    real_t[:, ::1] matrixData,
    real_t[:, ::1] matrixMunc,
    real_t[::1] state,
    real_t[::1] stateVar,
    real_t[::1] replicateBias,
    cnp.uint8_t[:, ::1] mask,
    int fold,
    double pad,
    double varianceFloor,
):
    cdef Py_ssize_t m = matrixData.shape[0]
    cdef Py_ssize_t n = matrixData.shape[1]
    if matrixMunc.shape[0] != m or matrixMunc.shape[1] != n or mask.shape[0] != m or mask.shape[1] != n:
        raise ValueError("held-out score inputs have inconsistent dimensions")
    if state.shape[0] != n or stateVar.shape[0] != n or replicateBias.shape[0] != m:
        raise ValueError("held-out score state inputs have inconsistent dimensions")

    cdef Py_ssize_t j, i, count = 0, k
    with nogil:
        for j in range(m):
            for i in range(n):
                if mask[j, i] == 0:
                    count += 1

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] residual = np.empty(count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pState = np.empty(count, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] obsVar = np.empty(count, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] intervalIndex = np.empty(count, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] repIndex = np.empty(count, dtype=np.int64)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] foldIndex = np.empty(count, dtype=np.int32)

    cdef double[::1] residualView = residual
    cdef double[::1] pStateView = pState
    cdef double[::1] obsVarView = obsVar
    cdef cnp.int64_t[::1] intervalView = intervalIndex
    cdef cnp.int64_t[::1] repView = repIndex
    cdef cnp.int32_t[::1] foldView = foldIndex
    cdef double pVal, rVal

    k = 0
    with nogil:
        for j in range(m):
            for i in range(n):
                if mask[j, i] == 0:
                    residualView[k] = (
                        <double>matrixData[j, i]
                        - <double>state[i]
                        - <double>replicateBias[j]
                    )
                    pVal = <double>stateVar[i]
                    if pVal < varianceFloor or pVal != pVal:
                        pVal = varianceFloor
                    rVal = <double>matrixMunc[j, i] + pad
                    if rVal < varianceFloor or rVal != rVal:
                        rVal = varianceFloor
                    pStateView[k] = pVal
                    obsVarView[k] = rVal
                    intervalView[k] = i
                    repView[k] = j
                    foldView[k] = fold
                    k += 1

    return residual, pState, obsVar, intervalIndex, repIndex, foldIndex


cpdef double cfactorObjective(
    real_t[::1] theta,
    real_t[::1] residual,
    real_t[::1] pState,
    real_t[::1] obsVar,
    real_t[:, ::1] featureByInterval,
    cnp.int64_t[::1] intervalIndex,
    real_t[::1] targets,
    real_t[::1] targetZ,
    double factorMin,
    double factorMax,
    double ridge,
    double aObsPenalty,
    double scaleWIS,
    double aObsMin,
    double aObsMax,
    double targetAlphaFloor,
    double wisWeight,
    double varianceFloor,
):
    cdef Py_ssize_t n = residual.shape[0]
    cdef Py_ssize_t p = featureByInterval.shape[1]
    if theta.shape[0] != p + 1:
        raise ValueError("theta length does not match feature matrix")
    if pState.shape[0] != n or obsVar.shape[0] != n or intervalIndex.shape[0] != n:
        raise ValueError("objective inputs have inconsistent dimensions")

    cdef double logFactorMin = log(factorMin)
    cdef double logFactorMax = log(factorMax)
    cdef double logAObs = _clip_double(<double>theta[p], log(aObsMin), log(aObsMax))
    cdef double aObs = exp(logAObs)
    cdef double nllSum = 0.0
    cdef double wisSum = 0.0
    cdef double ridgeSum = 0.0
    cdef double eta, factor, variance, sd, r, lower, upper, below, above, alpha
    cdef Py_ssize_t k, col, targetIndex, interval

    with nogil:
        for k in range(n):
            interval = intervalIndex[k]
            eta = 0.0
            for col in range(p):
                eta += <double>featureByInterval[interval, col] * <double>theta[col]
            eta = _clip_double(eta, logFactorMin, logFactorMax)
            factor = exp(eta)
            variance = factor * <double>pState[k] + aObs * <double>obsVar[k]
            if variance < varianceFloor:
                variance = varianceFloor
            r = <double>residual[k]
            nllSum += log(variance) + (r * r) / variance
            sd = sqrt(variance)
            for targetIndex in range(targets.shape[0]):
                alpha = 1.0 - <double>targets[targetIndex]
                if alpha < targetAlphaFloor:
                    alpha = targetAlphaFloor
                lower = -(<double>targetZ[targetIndex]) * sd
                upper = (<double>targetZ[targetIndex]) * sd
                below = lower - r
                if below < 0.0:
                    below = 0.0
                above = r - upper
                if above < 0.0:
                    above = 0.0
                wisSum += (upper - lower) + (2.0 / alpha) * below + (2.0 / alpha) * above

        for col in range(1, p):
            ridgeSum += <double>theta[col] * <double>theta[col]

    if n > 0:
        nllSum = 0.5 * nllSum / <double>n
        wisSum = wisSum / (<double>n * <double>max(targets.shape[0], 1))
    return nllSum + wisWeight * wisSum / scaleWIS + ridge * ridgeSum + aObsPenalty * logAObs * logAObs


cpdef tuple cfactorObjectiveAndGradient(
    real_t[::1] theta,
    real_t[::1] residual,
    real_t[::1] pState,
    real_t[::1] obsVar,
    real_t[:, ::1] featureByInterval,
    cnp.int64_t[::1] intervalIndex,
    real_t[::1] targets,
    real_t[::1] targetZ,
    double factorMin,
    double factorMax,
    double ridge,
    double aObsPenalty,
    double scaleWIS,
    double aObsMin,
    double aObsMax,
    double targetAlphaFloor,
    double wisWeight,
    double varianceFloor,
):
    cdef Py_ssize_t n = residual.shape[0]
    cdef Py_ssize_t p = featureByInterval.shape[1]
    if theta.shape[0] != p + 1:
        raise ValueError("theta length does not match feature matrix")
    if pState.shape[0] != n or obsVar.shape[0] != n or intervalIndex.shape[0] != n:
        raise ValueError("objective inputs have inconsistent dimensions")

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] grad = np.zeros(p + 1, dtype=np.float64)
    cdef double[::1] gradView = grad
    cdef double logFactorMin = log(factorMin)
    cdef double logFactorMax = log(factorMax)
    cdef double rawLogAObs = <double>theta[p]
    cdef double logAObs = _clip_double(rawLogAObs, log(aObsMin), log(aObsMax))
    cdef double aObs = exp(logAObs)
    cdef bint aObsActive = rawLogAObs >= log(aObsMin) and rawLogAObs <= log(aObsMax)
    cdef double nllSum = 0.0
    cdef double wisSum = 0.0
    cdef double ridgeSum = 0.0
    cdef double etaRaw, eta, factor, variance, sd, r, lower, upper, below, above
    cdef double alpha, dvarCommon, dWisDsd, dWisDvar, coef, dvarEta, dvarObs
    cdef Py_ssize_t k, col, targetIndex, interval
    cdef bint etaActive

    with nogil:
        for k in range(n):
            interval = intervalIndex[k]
            etaRaw = 0.0
            for col in range(p):
                etaRaw += <double>featureByInterval[interval, col] * <double>theta[col]
            eta = _clip_double(etaRaw, logFactorMin, logFactorMax)
            etaActive = etaRaw >= logFactorMin and etaRaw <= logFactorMax
            factor = exp(eta)
            variance = factor * <double>pState[k] + aObs * <double>obsVar[k]
            if variance < varianceFloor:
                variance = varianceFloor
            r = <double>residual[k]
            nllSum += log(variance) + (r * r) / variance
            dvarCommon = 0.5 * (1.0 / variance - (r * r) / (variance * variance))
            sd = sqrt(variance)
            dWisDsd = 0.0
            for targetIndex in range(targets.shape[0]):
                alpha = 1.0 - <double>targets[targetIndex]
                if alpha < targetAlphaFloor:
                    alpha = targetAlphaFloor
                lower = -(<double>targetZ[targetIndex]) * sd
                upper = (<double>targetZ[targetIndex]) * sd
                below = lower - r
                if below < 0.0:
                    below = 0.0
                above = r - upper
                if above < 0.0:
                    above = 0.0
                wisSum += (upper - lower) + (2.0 / alpha) * below + (2.0 / alpha) * above
                dWisDsd += 2.0 * <double>targetZ[targetIndex]
                if below > 0.0:
                    dWisDsd -= (2.0 / alpha) * <double>targetZ[targetIndex]
                if above > 0.0:
                    dWisDsd -= (2.0 / alpha) * <double>targetZ[targetIndex]

            coef = dvarCommon / <double>n
            if targets.shape[0] > 0:
                dWisDvar = dWisDsd * 0.5 / sd
                coef += wisWeight * dWisDvar / (
                    <double>n * <double>targets.shape[0] * scaleWIS
                )
            dvarEta = factor * <double>pState[k]
            dvarObs = aObs * <double>obsVar[k]
            if etaActive:
                for col in range(p):
                    gradView[col] += coef * dvarEta * <double>featureByInterval[interval, col]
            if aObsActive:
                gradView[p] += coef * dvarObs

        for col in range(1, p):
            ridgeSum += <double>theta[col] * <double>theta[col]
            gradView[col] += 2.0 * ridge * <double>theta[col]
        if aObsActive:
            gradView[p] += 2.0 * aObsPenalty * logAObs

    if n > 0:
        nllSum = 0.5 * nllSum / <double>n
        wisSum = wisSum / (<double>n * <double>max(targets.shape[0], 1))
    return (
        nllSum + wisWeight * wisSum / scaleWIS + ridge * ridgeSum + aObsPenalty * logAObs * logAObs,
        grad,
    )


cpdef tuple cevaluateFactor(
    real_t[:, ::1] featureByInterval,
    real_t[::1] beta,
    real_t[::1] fullP,
    double factorMin,
    double factorMax,
):
    cdef Py_ssize_t n = featureByInterval.shape[0]
    cdef Py_ssize_t p = featureByInterval.shape[1]
    if beta.shape[0] != p or fullP.shape[0] != n:
        raise ValueError("factor evaluation inputs have inconsistent dimensions")
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] factor = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] calibrated = np.empty(n, dtype=np.float32)
    cdef cnp.float32_t[::1] factorView = factor
    cdef cnp.float32_t[::1] calView = calibrated
    cdef double logFactorMin = log(factorMin)
    cdef double logFactorMax = log(factorMax)
    cdef Py_ssize_t i, col
    cdef double eta, fac, pVal

    IF USE_OPENMP:
        for i in prange(n, nogil=True, schedule="static"):
            eta = 0.0
            for col in range(p):
                eta += <double>featureByInterval[i, col] * <double>beta[col]
            eta = _clip_double(eta, logFactorMin, logFactorMax)
            fac = exp(eta)
            pVal = <double>fullP[i]
            if pVal < 0.0 or pVal != pVal:
                pVal = 0.0
            factorView[i] = <cnp.float32_t>fac
            calView[i] = <cnp.float32_t>sqrt(fac * pVal)
    ELSE:
        with nogil:
            for i in range(n):
                eta = 0.0
                for col in range(p):
                    eta += <double>featureByInterval[i, col] * <double>beta[col]
                eta = _clip_double(eta, logFactorMin, logFactorMax)
                fac = exp(eta)
                pVal = <double>fullP[i]
                if pVal < 0.0 or pVal != pVal:
                    pVal = 0.0
                factorView[i] = <cnp.float32_t>fac
                calView[i] = <cnp.float32_t>sqrt(fac * pVal)

    return factor, calibrated


cpdef dict csummarizeCoverageWidths(
    real_t[::1] residual,
    real_t[::1] sdBefore,
    real_t[::1] sdAfter,
    cnp.int32_t[::1] decile,
    real_t[::1] targets,
    real_t[::1] targetZ,
    double medianQuantile,
    double highWidthQuantile,
):
    cdef Py_ssize_t n = residual.shape[0]
    if sdBefore.shape[0] != n or sdAfter.shape[0] != n or decile.shape[0] != n:
        raise ValueError("summary inputs have inconsistent dimensions")
    if targets.shape[0] != targetZ.shape[0]:
        raise ValueError("target inputs have inconsistent dimensions")

    decileArr = np.asarray(decile)
    groupCodes = np.concatenate(
        [np.array([-1], dtype=np.int32), np.unique(decileArr[decileArr >= 0]).astype(np.int32)]
    )
    cdef Py_ssize_t groupCount = groupCodes.shape[0]
    cdef Py_ssize_t targetCount = targets.shape[0]
    cdef Py_ssize_t rowCount = groupCount * targetCount
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] groupOut = np.empty(rowCount, dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] targetOut = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] nOut = np.empty(rowCount, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] covBefore = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] covAfter = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] meanBefore = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] meanAfter = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] medianBefore = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] medianAfter = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q90Before = np.empty(rowCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] q90After = np.empty(rowCount, dtype=np.float64)

    cdef cnp.int32_t[::1] groupView = groupCodes
    cdef Py_ssize_t g, t, k, row, count
    cdef int groupCode
    cdef double z, widthScale, absResidual
    cdef double sumBefore, sumAfter, hitBefore, hitAfter

    for g in range(groupCount):
        groupCode = groupView[g]
        if groupCode < 0:
            idx = np.ones(n, dtype=bool)
        else:
            idx = decileArr == groupCode
        count = int(np.sum(idx))
        if count > 0:
            sdBeforeGroup = np.asarray(sdBefore)[idx]
            sdAfterGroup = np.asarray(sdAfter)[idx]
            residualGroup = np.asarray(residual)[idx]
            medBeforeBase = float(np.quantile(sdBeforeGroup, medianQuantile))
            medAfterBase = float(np.quantile(sdAfterGroup, medianQuantile))
            q90BeforeBase = float(np.quantile(sdBeforeGroup, highWidthQuantile))
            q90AfterBase = float(np.quantile(sdAfterGroup, highWidthQuantile))
            meanBeforeBase = float(np.mean(sdBeforeGroup))
            meanAfterBase = float(np.mean(sdAfterGroup))
        else:
            sdBeforeGroup = np.asarray(sdBefore)[idx]
            sdAfterGroup = np.asarray(sdAfter)[idx]
            residualGroup = np.asarray(residual)[idx]
            medBeforeBase = np.nan
            medAfterBase = np.nan
            q90BeforeBase = np.nan
            q90AfterBase = np.nan
            meanBeforeBase = np.nan
            meanAfterBase = np.nan
        for t in range(targetCount):
            row = t * groupCount + g
            z = <double>targetZ[t]
            widthScale = 2.0 * z
            hitBefore = 0.0
            hitAfter = 0.0
            for k in range(count):
                absResidual = fabs(<double>residualGroup[k])
                if absResidual <= z * <double>sdBeforeGroup[k]:
                    hitBefore += 1.0
                if absResidual <= z * <double>sdAfterGroup[k]:
                    hitAfter += 1.0
            groupOut[row] = groupCode
            targetOut[row] = <double>targets[t]
            nOut[row] = count
            covBefore[row] = hitBefore / count if count > 0 else np.nan
            covAfter[row] = hitAfter / count if count > 0 else np.nan
            meanBefore[row] = widthScale * meanBeforeBase
            meanAfter[row] = widthScale * meanAfterBase
            medianBefore[row] = widthScale * medBeforeBase
            medianAfter[row] = widthScale * medAfterBase
            q90Before[row] = widthScale * q90BeforeBase
            q90After[row] = widthScale * q90AfterBase

    return {
        "group": groupOut,
        "target": targetOut,
        "n": nOut,
        "coverage_before": covBefore,
        "coverage_after": covAfter,
        "mean_width_before": meanBefore,
        "mean_width_after": meanAfter,
        "median_width_before": medianBefore,
        "median_width_after": medianAfter,
        "q90_width_before": q90Before,
        "q90_width_after": q90After,
    }
