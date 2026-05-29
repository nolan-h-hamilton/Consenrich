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


def cextractDeletedStateScores(
    fullState,
    deletedState,
    deletedStateVar,
    activeMask,
    foldMask,
    int fold,
    double varianceFloor,
):
    """Extract interval-level full-vs-delete-state calibration residuals."""
    cdef object fullStateObj
    cdef object deletedStateObj
    cdef object deletedStateVarObj
    cdef object activeMaskObj
    cdef object foldMaskObj
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] fullStateArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] deletedStateArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] deletedStateVarArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] activeMaskArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] foldMaskArr
    cdef double[::1] fullStateView
    cdef double[::1] deletedStateView
    cdef double[::1] deletedStateVarView
    cdef cnp.uint8_t[:, ::1] activeMaskView
    cdef cnp.uint8_t[:, ::1] foldMaskView
    cdef Py_ssize_t m, n, i, j, count, k, heldout, kept
    cdef double pVal

    fullStateObj = np.ascontiguousarray(fullState, dtype=np.float64)
    deletedStateObj = np.ascontiguousarray(deletedState, dtype=np.float64)
    deletedStateVarObj = np.ascontiguousarray(deletedStateVar, dtype=np.float64)
    activeMaskObj = np.ascontiguousarray(activeMask, dtype=np.uint8)
    foldMaskObj = np.ascontiguousarray(foldMask, dtype=np.uint8)

    if fullStateObj.ndim != 1 or deletedStateObj.ndim != 1 or deletedStateVarObj.ndim != 1:
        raise ValueError("deleted-state score state inputs must be one-dimensional")
    if activeMaskObj.ndim != 2 or foldMaskObj.ndim != 2:
        raise ValueError("deleted-state score masks must be two-dimensional")

    fullStateArr = fullStateObj
    deletedStateArr = deletedStateObj
    deletedStateVarArr = deletedStateVarObj
    activeMaskArr = activeMaskObj
    foldMaskArr = foldMaskObj

    n = fullStateArr.shape[0]
    m = activeMaskArr.shape[0]
    if (
        deletedStateArr.shape[0] != n
        or deletedStateVarArr.shape[0] != n
        or activeMaskArr.shape[1] != n
        or foldMaskArr.shape[0] != m
        or foldMaskArr.shape[1] != n
    ):
        raise ValueError("deleted-state score inputs have inconsistent dimensions")

    fullStateView = fullStateArr
    deletedStateView = deletedStateArr
    deletedStateVarView = deletedStateVarArr
    activeMaskView = activeMaskArr
    foldMaskView = foldMaskArr

    count = 0
    with nogil:
        for i in range(n):
            heldout = 0
            for j in range(m):
                if activeMaskView[j, i] != 0 and foldMaskView[j, i] == 0:
                    heldout += 1
            if heldout > 0:
                count += 1

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] residual = np.empty(
        count, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pState = np.empty(
        count, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] intervalIndex = np.empty(
        count, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] foldIndex = np.empty(
        count, dtype=np.int32
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] heldoutCount = np.empty(
        count, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] keptCount = np.empty(
        count, dtype=np.int64
    )
    cdef double[::1] residualView = residual
    cdef double[::1] pStateView = pState
    cdef cnp.int64_t[::1] intervalView = intervalIndex
    cdef cnp.int32_t[::1] foldView = foldIndex
    cdef cnp.int64_t[::1] heldoutCountView = heldoutCount
    cdef cnp.int64_t[::1] keptCountView = keptCount

    k = 0
    with nogil:
        for i in range(n):
            heldout = 0
            kept = 0
            for j in range(m):
                if activeMaskView[j, i] != 0:
                    if foldMaskView[j, i] == 0:
                        heldout += 1
                    else:
                        kept += 1
            if heldout > 0:
                residualView[k] = fullStateView[i] - deletedStateView[i]
                pVal = deletedStateVarView[i]
                if pVal < varianceFloor or pVal != pVal:
                    pVal = varianceFloor
                pStateView[k] = pVal
                intervalView[k] = i
                foldView[k] = fold
                heldoutCountView[k] = heldout
                keptCountView[k] = kept
                k += 1

    return residual, pState, intervalIndex, foldIndex, heldoutCount, keptCount


def cdeleteBlockBlockScores(
    residual,
    pState,
    factorByInterval,
    intervalIndex,
    blockIndex,
    targetBlockMask,
    *args,
    **kwargs,
):
    """Return max delete-state z-score per selected block."""
    cdef object residualObj
    cdef object pStateObj
    cdef object factorObj
    cdef object intervalObj
    cdef object blockObj
    cdef object targetMaskObj
    cdef object heldoutObj = None
    cdef object heldoutCountObj
    cdef object kwargsDict
    cdef object varianceFloorObj = None
    cdef double varianceFloor
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] residualArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pStateArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] factorArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] intervalArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockArr
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] targetMaskArr
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] heldoutCountArr
    cdef double[::1] residualView
    cdef double[::1] pStateView
    cdef double[::1] factorView
    cdef cnp.int64_t[::1] intervalView
    cdef cnp.int64_t[::1] blockView
    cdef cnp.uint8_t[::1] targetMaskView
    cdef cnp.int64_t[::1] heldoutCountView
    cdef bint useHeldoutCounts = False
    cdef Py_ssize_t n, blockCount, k, interval, block, outCount, outIndex
    cdef double var, score, r
    cdef cnp.int64_t cellCount

    kwargsDict = dict(kwargs)
    if "heldoutCounts" in kwargsDict and "heldoutCount" not in kwargsDict:
        kwargsDict["heldoutCount"] = kwargsDict.pop("heldoutCounts")
    if "heldoutCount" in kwargsDict:
        heldoutObj = kwargsDict.pop("heldoutCount")
        useHeldoutCounts = True
    if "varianceFloor" in kwargsDict:
        varianceFloorObj = kwargsDict.pop("varianceFloor")
    if len(kwargsDict) != 0:
        raise TypeError(
            "cdeleteBlockBlockScores got unexpected keyword argument(s): "
            + ", ".join(sorted(kwargsDict))
        )

    if len(args) == 1:
        if varianceFloorObj is not None:
            raise TypeError("cdeleteBlockBlockScores got multiple varianceFloor values")
        varianceFloorObj = args[0]
    elif len(args) == 2:
        if heldoutObj is not None or varianceFloorObj is not None:
            raise TypeError("cdeleteBlockBlockScores got multiple optional argument values")
        heldoutObj = args[0]
        varianceFloorObj = args[1]
        useHeldoutCounts = True
    elif len(args) != 0:
        raise TypeError(
            "cdeleteBlockBlockScores expects varianceFloor, optionally preceded by heldoutCount"
        )
    if varianceFloorObj is None:
        raise TypeError("cdeleteBlockBlockScores missing required varianceFloor")
    varianceFloor = float(varianceFloorObj)

    residualObj = np.ascontiguousarray(residual, dtype=np.float64)
    pStateObj = np.ascontiguousarray(pState, dtype=np.float64)
    factorObj = np.ascontiguousarray(factorByInterval, dtype=np.float64)
    intervalObj = np.ascontiguousarray(intervalIndex, dtype=np.int64)
    blockObj = np.ascontiguousarray(blockIndex, dtype=np.int64)
    targetMaskObj = np.ascontiguousarray(targetBlockMask, dtype=np.uint8)
    heldoutCountObj = np.empty(0, dtype=np.int64)
    if useHeldoutCounts:
        heldoutCountObj = np.ascontiguousarray(heldoutObj, dtype=np.int64)

    if (
        residualObj.ndim != 1
        or pStateObj.ndim != 1
        or factorObj.ndim != 1
        or intervalObj.ndim != 1
        or blockObj.ndim != 1
        or targetMaskObj.ndim != 1
        or heldoutCountObj.ndim != 1
    ):
        raise ValueError("delete-block score inputs must be one-dimensional")

    residualArr = residualObj
    pStateArr = pStateObj
    factorArr = factorObj
    intervalArr = intervalObj
    blockArr = blockObj
    targetMaskArr = targetMaskObj
    heldoutCountArr = heldoutCountObj

    n = residualArr.shape[0]
    blockCount = targetMaskArr.shape[0]
    if (
        pStateArr.shape[0] != n
        or intervalArr.shape[0] != n
        or blockArr.shape[0] != n
        or (useHeldoutCounts and heldoutCountArr.shape[0] != n)
    ):
        raise ValueError("delete-block score inputs have inconsistent dimensions")

    residualView = residualArr
    pStateView = pStateArr
    factorView = factorArr
    intervalView = intervalArr
    blockView = blockArr
    targetMaskView = targetMaskArr
    heldoutCountView = heldoutCountArr

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] blockScore = np.full(
        blockCount, -1.0, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockCellCount = np.zeros(
        blockCount, dtype=np.int64
    )
    cdef double[::1] blockScoreView = blockScore
    cdef cnp.int64_t[::1] blockCellCountView = blockCellCount

    with nogil:
        for k in range(n):
            block = blockView[k]
            if block < 0 or block >= blockCount:
                continue
            if targetMaskView[block] == 0:
                continue
            interval = intervalView[k]
            if interval < 0 or interval >= factorView.shape[0]:
                continue
            r = residualView[k]
            if not isfinite(r):
                continue
            var = factorView[interval] * pStateView[k]
            if not isfinite(var) or var <= varianceFloor:
                continue
            score = fabs(r) / sqrt(var)
            if not isfinite(score):
                continue
            if score > blockScoreView[block]:
                blockScoreView[block] = score
            if useHeldoutCounts:
                cellCount = heldoutCountView[k]
                if cellCount < 1:
                    cellCount = 1
                blockCellCountView[block] += cellCount
            else:
                blockCellCountView[block] += 1

    outCount = 0
    for block in range(blockCount):
        if blockCellCountView[block] > 0 and blockScoreView[block] >= 0.0:
            outCount += 1

    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockOut = np.empty(
        outCount, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] scoreOut = np.empty(
        outCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] countOut = np.empty(
        outCount, dtype=np.int64
    )
    cdef cnp.int64_t[::1] blockOutView = blockOut
    cdef double[::1] scoreOutView = scoreOut
    cdef cnp.int64_t[::1] countOutView = countOut

    outIndex = 0
    for block in range(blockCount):
        if blockCellCountView[block] > 0 and blockScoreView[block] >= 0.0:
            blockOutView[outIndex] = block
            scoreOutView[outIndex] = blockScoreView[block]
            countOutView[outIndex] = blockCellCountView[block]
            outIndex += 1

    return blockOut, scoreOut, countOut


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
