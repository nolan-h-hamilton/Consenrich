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


cpdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] csegShrinkSegmentCodes(
    Py_ssize_t n,
    Py_ssize_t segmentCount,
):
    if n < 1:
        raise ValueError("n must be positive")
    if segmentCount < 1:
        raise ValueError("segmentCount must be positive")
    cdef Py_ssize_t effectiveCount = segmentCount if segmentCount < n else n
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.int32)
    cdef cnp.int32_t[::1] outView = out
    cdef Py_ssize_t i
    with nogil:
        for i in range(n):
            outView[i] = <cnp.int32_t>((i * effectiveCount) // n)
    return out


cpdef tuple csegShrinkScopeCodes(
    long contigOrdinal,
    object segmentByInterval,
    object intervalIndex,
):
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] segmentArr = np.ascontiguousarray(
        np.asarray(segmentByInterval, dtype=np.int32).reshape(-1), dtype=np.int32
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] intervalArr = np.ascontiguousarray(
        np.asarray(intervalIndex, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef Py_ssize_t n = intervalArr.shape[0]
    cdef Py_ssize_t intervalCount = segmentArr.shape[0]
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] contigScope = np.empty(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] segmentScope = np.empty(n, dtype=np.int32)
    cdef cnp.int32_t[::1] segmentView = segmentArr
    cdef cnp.int64_t[::1] intervalView = intervalArr
    cdef cnp.int32_t[::1] contigScopeView = contigScope
    cdef cnp.int32_t[::1] segmentScopeView = segmentScope
    cdef Py_ssize_t i
    cdef cnp.int32_t maxSegment = -1
    cdef cnp.int32_t segment
    cdef cnp.int64_t interval
    with nogil:
        for i in range(intervalCount):
            if segmentView[i] > maxSegment:
                maxSegment = segmentView[i]
        for i in range(n):
            interval = intervalView[i]
            contigScopeView[i] = <cnp.int32_t>contigOrdinal
            if interval < 0 or interval >= intervalCount or maxSegment < 0:
                segmentScopeView[i] = <cnp.int32_t>-1
            else:
                segment = segmentView[interval]
                segmentScopeView[i] = <cnp.int32_t>(
                    contigOrdinal * (<long>maxSegment + 1) + segment
                )
    return contigScope, segmentScope


cpdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] csegShrinkGroupCodes(
    long contigOrdinal,
    object foldIndex,
    object blockIDX,
):
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] foldArr = np.ascontiguousarray(
        np.asarray(foldIndex, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockArr = np.ascontiguousarray(
        np.asarray(blockIDX, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef Py_ssize_t n = foldArr.shape[0]
    if blockArr.shape[0] != n:
        raise ValueError("foldIndex and blockIDX must have the same length")
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] out = np.empty(n, dtype=np.int64)
    cdef cnp.int64_t[::1] foldView = foldArr
    cdef cnp.int64_t[::1] blockView = blockArr
    cdef cnp.int64_t[::1] outView = out
    cdef Py_ssize_t i
    cdef cnp.int64_t maxFold = 0
    cdef cnp.int64_t maxBlock = 0
    cdef cnp.int64_t fold
    cdef cnp.int64_t block
    cdef cnp.int64_t foldStride
    cdef cnp.int64_t blockStride
    with nogil:
        for i in range(n):
            if foldView[i] > maxFold:
                maxFold = foldView[i]
            if blockView[i] > maxBlock:
                maxBlock = blockView[i]
        foldStride = maxFold + 1
        blockStride = maxBlock + 1
        for i in range(n):
            fold = foldView[i]
            block = blockView[i]
            if fold < 0 or block < 0:
                outView[i] = <cnp.int64_t>-1
            else:
                outView[i] = (
                    (<cnp.int64_t>contigOrdinal * foldStride + fold) * blockStride
                    + block
                )
    return out


cpdef tuple csegShrinkBootstrapLogFactors(
    object ratio,
    object rowWeight,
    object scopeCode,
    object groupCode,
    object bootstrapMultiplier,
    Py_ssize_t scopeCount,
    double target,
    double z,
    double factorMin,
    double factorMax,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] ratioArr = np.ascontiguousarray(
        np.asarray(ratio, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] weightArr = np.ascontiguousarray(
        np.asarray(rowWeight, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] scopeArr = np.ascontiguousarray(
        np.asarray(scopeCode, dtype=np.int32).reshape(-1), dtype=np.int32
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] groupArr = np.ascontiguousarray(
        np.asarray(groupCode, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] multArr = np.ascontiguousarray(
        np.asarray(bootstrapMultiplier, dtype=np.float64), dtype=np.float64
    )
    cdef Py_ssize_t n = ratioArr.shape[0]
    if weightArr.shape[0] != n or scopeArr.shape[0] != n or groupArr.shape[0] != n:
        raise ValueError("segShrink bootstrap inputs must have the same length")
    if scopeCount < 1:
        raise ValueError("scopeCount must be positive")
    cdef Py_ssize_t replicateCount = multArr.shape[0]
    cdef Py_ssize_t groupCount = multArr.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] baseLog = np.full(scopeCount, np.nan, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] bootLog = np.full((scopeCount, replicateCount), np.nan, dtype=np.float64)
    cdef double[::1] ratioView = ratioArr
    cdef double[::1] weightView = weightArr
    cdef cnp.int32_t[::1] scopeView = scopeArr
    cdef cnp.int64_t[::1] groupView = groupArr
    cdef double[:, ::1] multView = multArr
    cdef double[::1] baseView = baseLog
    cdef double[:, ::1] bootView = bootLog
    cdef Py_ssize_t s, r, i
    cdef double total, threshold, cumulative, w, qValue, factor
    cdef cnp.int64_t group
    with nogil:
        for s in range(scopeCount):
            total = 0.0
            for i in range(n):
                if scopeView[i] == s and isfinite(ratioView[i]) and isfinite(weightView[i]) and weightView[i] > 0.0:
                    total += weightView[i]
            if total > 0.0 and isfinite(total) and z > 0.0:
                threshold = target * total
                cumulative = 0.0
                qValue = NAN
                for i in range(n):
                    if scopeView[i] == s and isfinite(ratioView[i]) and isfinite(weightView[i]) and weightView[i] > 0.0:
                        cumulative += weightView[i]
                        if cumulative >= threshold:
                            qValue = ratioView[i]
                            break
                if isfinite(qValue):
                    factor = (qValue / z) * (qValue / z)
                    factor = _clip_double(factor, factorMin, factorMax)
                    if factor > 0.0 and isfinite(factor):
                        baseView[s] = log(factor)
            for r in range(replicateCount):
                total = 0.0
                for i in range(n):
                    group = groupView[i]
                    if (
                        scopeView[i] == s
                        and group >= 0
                        and group < groupCount
                        and isfinite(ratioView[i])
                        and isfinite(weightView[i])
                        and weightView[i] > 0.0
                    ):
                        w = weightView[i] * multView[r, group]
                        if isfinite(w) and w > 0.0:
                            total += w
                if not (total > 0.0 and isfinite(total) and z > 0.0):
                    continue
                threshold = target * total
                cumulative = 0.0
                qValue = NAN
                for i in range(n):
                    group = groupView[i]
                    if (
                        scopeView[i] == s
                        and group >= 0
                        and group < groupCount
                        and isfinite(ratioView[i])
                        and isfinite(weightView[i])
                        and weightView[i] > 0.0
                    ):
                        w = weightView[i] * multView[r, group]
                        if isfinite(w) and w > 0.0:
                            cumulative += w
                            if cumulative >= threshold:
                                qValue = ratioView[i]
                                break
                if isfinite(qValue):
                    factor = (qValue / z) * (qValue / z)
                    factor = _clip_double(factor, factorMin, factorMax)
                    if factor > 0.0 and isfinite(factor):
                        bootView[s, r] = log(factor)
    return baseLog, bootLog


cpdef dict csegShrinkEmpiricalBayes(
    double genomeLogFactor,
    object contigLogFactor,
    object contigVariance,
    object segmentLogFactor,
    object segmentVariance,
    object segmentContigIndex,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] contigLogArr = np.ascontiguousarray(
        np.asarray(contigLogFactor, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] contigVarArr = np.ascontiguousarray(
        np.asarray(contigVariance, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] segmentLogArr = np.ascontiguousarray(
        np.asarray(segmentLogFactor, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] segmentVarArr = np.ascontiguousarray(
        np.asarray(segmentVariance, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] segmentContigArr = np.ascontiguousarray(
        np.asarray(segmentContigIndex, dtype=np.int32).reshape(-1), dtype=np.int32
    )
    cdef Py_ssize_t contigCount = contigLogArr.shape[0]
    cdef Py_ssize_t segmentCount = segmentLogArr.shape[0]
    if contigVarArr.shape[0] != contigCount:
        raise ValueError("contig inputs must have the same length")
    if segmentVarArr.shape[0] != segmentCount or segmentContigArr.shape[0] != segmentCount:
        raise ValueError("segment inputs must have the same length")
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] contigTheta = np.empty(contigCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] contigAlpha = np.empty(contigCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] segmentTheta = np.empty(segmentCount, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] segmentAlpha = np.empty(segmentCount, dtype=np.float64)
    cdef double[::1] contigLog = contigLogArr
    cdef double[::1] contigVar = contigVarArr
    cdef double[::1] segmentLog = segmentLogArr
    cdef double[::1] segmentVar = segmentVarArr
    cdef cnp.int32_t[::1] segmentContig = segmentContigArr
    cdef double[::1] contigThetaView = contigTheta
    cdef double[::1] contigAlphaView = contigAlpha
    cdef double[::1] segmentThetaView = segmentTheta
    cdef double[::1] segmentAlphaView = segmentAlpha
    cdef Py_ssize_t i
    cdef double weightSum = 0.0
    cdef double valueSum = 0.0
    cdef double w, v, y, diff, value, denom, parent
    cdef double tauContigSq = 0.0
    cdef double tauSegmentSq = 0.0
    cdef cnp.int32_t contig
    with nogil:
        for i in range(contigCount):
            y = contigLog[i]
            v = contigVar[i]
            if isfinite(y) and isfinite(v) and v >= 0.0 and isfinite(genomeLogFactor):
                w = 1.0 / _max_floor(v, 1.0e-12)
                diff = y - genomeLogFactor
                value = diff * diff - v
                if isfinite(value) and isfinite(w) and w > 0.0:
                    valueSum += w * value
                    weightSum += w
        if weightSum > 0.0 and valueSum > 0.0:
            tauContigSq = valueSum / weightSum
        for i in range(contigCount):
            y = contigLog[i]
            v = contigVar[i]
            if isfinite(y) and isfinite(v) and v >= 0.0 and isfinite(genomeLogFactor):
                denom = tauContigSq + v
                if denom > 0.0 and isfinite(denom):
                    contigAlphaView[i] = tauContigSq / denom
                else:
                    contigAlphaView[i] = 0.0
                contigThetaView[i] = contigAlphaView[i] * y + (1.0 - contigAlphaView[i]) * genomeLogFactor
            else:
                contigAlphaView[i] = 0.0
                contigThetaView[i] = genomeLogFactor
        weightSum = 0.0
        valueSum = 0.0
        for i in range(segmentCount):
            contig = segmentContig[i]
            y = segmentLog[i]
            v = segmentVar[i]
            if contig >= 0 and contig < contigCount and isfinite(y) and isfinite(v) and v >= 0.0:
                parent = contigThetaView[contig]
                w = 1.0 / _max_floor(v, 1.0e-12)
                diff = y - parent
                value = diff * diff - v
                if isfinite(value) and isfinite(w) and w > 0.0:
                    valueSum += w * value
                    weightSum += w
        if weightSum > 0.0 and valueSum > 0.0:
            tauSegmentSq = valueSum / weightSum
        for i in range(segmentCount):
            contig = segmentContig[i]
            y = segmentLog[i]
            v = segmentVar[i]
            if contig >= 0 and contig < contigCount:
                parent = contigThetaView[contig]
            else:
                parent = genomeLogFactor
            if isfinite(y) and isfinite(v) and v >= 0.0 and isfinite(parent):
                denom = tauSegmentSq + v
                if denom > 0.0 and isfinite(denom):
                    segmentAlphaView[i] = tauSegmentSq / denom
                else:
                    segmentAlphaView[i] = 0.0
                segmentThetaView[i] = segmentAlphaView[i] * y + (1.0 - segmentAlphaView[i]) * parent
            else:
                segmentAlphaView[i] = 0.0
                segmentThetaView[i] = parent
    return {
        "tauContigSq": float(tauContigSq),
        "tauSegmentSq": float(tauSegmentSq),
        "contigTheta": contigTheta,
        "contigAlpha": contigAlpha,
        "segmentTheta": segmentTheta,
        "segmentAlpha": segmentAlpha,
    }


cpdef tuple csegShrinkApplyFactors(
    object segmentByInterval,
    object segmentLogFactor,
    object fullP,
    double positiveFloor,
):
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] segmentArr = np.ascontiguousarray(
        np.asarray(segmentByInterval, dtype=np.int32).reshape(-1), dtype=np.int32
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] logArr = np.ascontiguousarray(
        np.asarray(segmentLogFactor, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pArr = np.ascontiguousarray(
        np.asarray(fullP, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef Py_ssize_t n = segmentArr.shape[0]
    if pArr.shape[0] != n:
        raise ValueError("segmentByInterval and fullP must have the same length")
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] factor = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] calibrated = np.empty(n, dtype=np.float32)
    cdef cnp.int32_t[::1] segmentView = segmentArr
    cdef double[::1] logView = logArr
    cdef double[::1] pView = pArr
    cdef double[::1] factorView = factor
    cdef cnp.float32_t[::1] calView = calibrated
    cdef Py_ssize_t i
    cdef cnp.int32_t segment
    cdef double f, variance
    with nogil:
        for i in range(n):
            segment = segmentView[i]
            if segment >= 0 and segment < logArr.shape[0] and isfinite(logView[segment]):
                f = exp(logView[segment])
            else:
                f = 1.0
            factorView[i] = f
            variance = f * pView[i]
            if variance < positiveFloor or not isfinite(variance):
                variance = positiveFloor
            calView[i] = <cnp.float32_t>sqrt(variance)
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

# Dense delete-block target-block scoring moved from Python loops.
cpdef tuple cdeleteBlockTargetBlockScores(
    object residual,
    object pDelta,
    object factorByInterval,
    object intervalIndex,
    object blockIndex,
    object targetBlockMask,
    double positiveFloor,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] residualArr = np.ascontiguousarray(
        np.asarray(residual, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pDeltaArr = np.ascontiguousarray(
        np.asarray(pDelta, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] factorArr = np.ascontiguousarray(
        np.asarray(factorByInterval, dtype=np.float64).reshape(-1), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] intervalArr = np.ascontiguousarray(
        np.asarray(intervalIndex, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockArr = np.ascontiguousarray(
        np.asarray(blockIndex, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] targetMaskArr = np.ascontiguousarray(
        np.asarray(targetBlockMask, dtype=np.uint8).reshape(-1), dtype=np.uint8
    )
    cdef Py_ssize_t n = residualArr.shape[0]
    if pDeltaArr.shape[0] != n or intervalArr.shape[0] != n or blockArr.shape[0] != n:
        raise ValueError("residual, pDelta, intervalIndex, and blockIndex must have the same length")
    cdef Py_ssize_t blockCount = targetMaskArr.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] blockScoreArr = np.full(blockCount, -1.0, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockCellCountArr = np.zeros(blockCount, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] validBlocksArr = np.empty(blockCount, dtype=np.int64)
    cdef double[::1] residualView = residualArr
    cdef double[::1] pDeltaView = pDeltaArr
    cdef double[::1] factorView = factorArr
    cdef cnp.int64_t[::1] intervalView = intervalArr
    cdef cnp.int64_t[::1] blockView = blockArr
    cdef cnp.uint8_t[::1] targetMaskView = targetMaskArr
    cdef double[::1] blockScore = blockScoreArr
    cdef cnp.int64_t[::1] blockCellCount = blockCellCountArr
    cdef cnp.int64_t[::1] validBlocks = validBlocksArr
    cdef Py_ssize_t k, validCount
    cdef cnp.int64_t block, interval
    cdef double variance, score

    with nogil:
        for k in range(n):
            block = blockView[k]
            if block < 0 or block >= blockCount or targetMaskView[block] == 0:
                continue
            interval = intervalView[k]
            if interval < 0 or interval >= factorArr.shape[0]:
                continue
            variance = factorView[interval] * pDeltaView[k]
            if not (isfinite(residualView[k]) and isfinite(variance) and variance > positiveFloor):
                continue
            score = fabs(residualView[k]) / sqrt(variance)
            if isfinite(score):
                if score > blockScore[block]:
                    blockScore[block] = score
                blockCellCount[block] += 1
        validCount = 0
        for block in range(blockCount):
            if blockCellCount[block] > 0 and blockScore[block] >= 0.0:
                validBlocks[validCount] = block
                validCount += 1
    return (
        validBlocksArr[:validCount].copy(),
        blockScoreArr[validBlocksArr[:validCount]].copy(),
        blockCellCountArr[validBlocksArr[:validCount]].copy(),
    )
