# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3

"""Cython kernels for state-uncertainty calibration."""

import numpy as np
cimport cython
cimport numpy as cnp

from libc.math cimport fabs, exp, log, sqrt, isfinite, NAN

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


cdef inline double _exchangeable_information(
    double sumWeight,
    double sumSqrtWeight,
    Py_ssize_t count,
    double rho,
) noexcept nogil:
    cdef double oneMinusRho
    cdef double denom
    cdef double adjusted
    if count <= 0 or sumWeight <= 0.0:
        return 0.0
    if rho <= 0.0:
        return sumWeight
    oneMinusRho = 1.0 - rho
    denom = oneMinusRho + rho * <double>count
    adjusted = (
        sumWeight / oneMinusRho
        - rho * sumSqrtWeight * sumSqrtWeight / (oneMinusRho * denom)
    )
    if adjusted > sumWeight:
        return sumWeight
    return adjusted


cpdef tuple cmakeFoldSpec(
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t blockLen,
    Py_ssize_t folds,
    double deletionProbability,
    long seed,
):
    if folds < 2:
        raise ValueError("uncertainty calibration requires at least two folds")
    if m < 1 or n < 1 or blockLen < 1:
        raise ValueError("invalid uncertainty calibration mask dimensions")
    if (
        not isfinite(deletionProbability)
        or deletionProbability <= 0.0
        or deletionProbability >= 1.0
    ):
        raise ValueError("delete-block deletion probability must be in (0, 1)")

    cdef Py_ssize_t blockCount = (n + blockLen - 1) // blockLen
    rng = np.random.default_rng(int(seed))
    blockOrder = rng.permutation(blockCount).astype(np.int32, copy=False)
    blockFoldArr = np.empty(blockCount, dtype=np.int32)
    blockFoldArr[blockOrder] = np.arange(blockCount, dtype=np.int32) % int(folds)
    repsByBlockCountArr = np.empty(blockCount, dtype=np.intp)
    repsByBlockArr = np.full((blockCount, m), -1, dtype=np.intp)
    cdef Py_ssize_t block, deleteCount
    for block in range(blockCount):
        deleteCount = <Py_ssize_t>rng.binomial(m, deletionProbability)
        while deleteCount < 1 or (m > 1 and deleteCount >= m):
            deleteCount = <Py_ssize_t>rng.binomial(m, deletionProbability)
        repsByBlockCountArr[block] = deleteCount
        repsByBlockArr[block, :deleteCount] = rng.choice(
            m, size=deleteCount, replace=False
        )

    return blockFoldArr, repsByBlockCountArr, repsByBlockArr


cpdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] cobservationTotalInformation(
    real_t[:, ::1] matrixMunc,
    cnp.uint8_t[:, ::1] activeMask,
    double[::1] lambdaExp,
    bint useLambda,
    double pad,
    double replicateDependenceRho=0.0,
):
    cdef Py_ssize_t m = matrixMunc.shape[0]
    cdef Py_ssize_t n = matrixMunc.shape[1]
    if activeMask.shape[0] != m or activeMask.shape[1] != n:
        raise ValueError("activeMask must match matrixMunc shape")
    if useLambda and lambdaExp.shape[0] != n:
        raise ValueError("fullObservationPrecision must match interval count")
    if not isfinite(pad):
        raise ValueError("observation information pad must be finite")
    if (
        not isfinite(replicateDependenceRho)
        or replicateDependenceRho < 0.0
        or replicateDependenceRho >= 1.0
    ):
        raise ValueError("replicate dependence rho must be in [0, 1)")

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] totalInfo = np.zeros(
        n, dtype=np.float64
    )
    cdef double[::1] totalView = totalInfo
    cdef Py_ssize_t i, j
    cdef Py_ssize_t count
    cdef double total, sumSqrt, value, lamValue

    with nogil:
        for i in range(n):
            total = 0.0
            sumSqrt = 0.0
            count = 0
            if useLambda:
                lamValue = lambdaExp[i]
            else:
                lamValue = 1.0
            for j in range(m):
                if activeMask[j, i] != 0:
                    value = lamValue / (<double>matrixMunc[j, i] + pad)
                    total += value
                    if replicateDependenceRho > 0.0:
                        sumSqrt += sqrt(value)
                        count += 1
            if replicateDependenceRho > 0.0:
                totalView[i] = _exchangeable_information(
                    total,
                    sumSqrt,
                    count,
                    replicateDependenceRho,
                )
            else:
                totalView[i] = total
    return totalInfo


cpdef tuple cmakeFoldMaskAndInformation(
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t blockLen,
    Py_ssize_t fold,
    cnp.int32_t[::1] blockFold,
    Py_ssize_t[::1] repsByBlockCount,
    Py_ssize_t[:, ::1] repsByBlock,
    real_t[:, ::1] matrixMunc,
    cnp.uint8_t[:, ::1] activeMask,
    double[::1] totalInfo,
    double[::1] lambdaExp,
    bint useLambda,
    double pad,
    double replicateDependenceRho=0.0,
    bint returnNominalHeldout=False,
):
    if m < 1 or n < 1 or blockLen < 1:
        raise ValueError("invalid uncertainty calibration mask dimensions")
    if fold < 0:
        raise ValueError("fold must be nonnegative")
    if matrixMunc.shape[0] != m or matrixMunc.shape[1] != n:
        raise ValueError("matrixMunc shape does not match fold spec")
    if activeMask.shape[0] != m or activeMask.shape[1] != n:
        raise ValueError("activeMask must match matrixMunc shape")
    if totalInfo.shape[0] != n:
        raise ValueError("total information must match interval count")
    if useLambda and lambdaExp.shape[0] != n:
        raise ValueError("fullObservationPrecision must match interval count")
    if not isfinite(pad):
        raise ValueError("observation information pad must be finite")
    if (
        not isfinite(replicateDependenceRho)
        or replicateDependenceRho < 0.0
        or replicateDependenceRho >= 1.0
    ):
        raise ValueError("replicate dependence rho must be in [0, 1)")

    cdef Py_ssize_t blockCount = (n + blockLen - 1) // blockLen
    if (
        blockFold.shape[0] != blockCount
        or repsByBlockCount.shape[0] != blockCount
        or repsByBlock.shape[0] != blockCount
    ):
        raise ValueError("fold spec has inconsistent block count")
    cdef Py_ssize_t deleteSlotCount = repsByBlock.shape[1]
    if deleteSlotCount < m:
        raise ValueError("fold spec replicate matrix must allow every sample")

    cdef Py_ssize_t block, h, h2, rep, deleteCount
    for block in range(blockCount):
        if blockFold[block] < 0:
            raise ValueError("fold spec contains negative fold id")
        deleteCount = repsByBlockCount[block]
        if deleteCount < 1 or deleteCount > m or deleteCount > deleteSlotCount:
            raise ValueError("fold spec deleted-replicate count is out of bounds")
        for h in range(deleteCount):
            rep = repsByBlock[block, h]
            if rep < 0 or rep >= m:
                raise ValueError("fold spec replicate is out of bounds")
            for h2 in range(h):
                if repsByBlock[block, h2] == rep:
                    raise ValueError("fold spec contains a duplicate replicate")

    cdef cnp.ndarray[cnp.uint8_t, ndim=2, mode="c"] mask = np.ones(
        (m, n), dtype=np.uint8
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] keptInfo = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] heldoutInfo = np.zeros(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] hFraction = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] nominalHeldoutInfo
    if returnNominalHeldout:
        nominalHeldoutInfo = np.zeros(n, dtype=np.float64)
    else:
        nominalHeldoutInfo = np.empty(0, dtype=np.float64)
    cdef cnp.uint8_t[:, ::1] maskView = mask
    cdef double[::1] totalView = totalInfo
    cdef double[::1] keptView = keptInfo
    cdef double[::1] heldoutView = heldoutInfo
    cdef double[::1] hView = hFraction
    cdef double[::1] nominalHeldoutView = nominalHeldoutInfo
    cdef Py_ssize_t start, end, i, j, count
    cdef double value, total, kept, sumSqrt, lamValue

    with nogil:
        for block in range(blockCount):
            if blockFold[block] != fold:
                continue
            start = block * blockLen
            end = start + blockLen
            if end > n:
                end = n
            deleteCount = repsByBlockCount[block]
            for h in range(deleteCount):
                rep = repsByBlock[block, h]
                for i in range(start, end):
                    maskView[rep, i] = <cnp.uint8_t>0
                    if activeMask[rep, i] != 0:
                        value = 1.0 / (<double>matrixMunc[rep, i] + pad)
                        if useLambda:
                            value *= lambdaExp[i]
                        if replicateDependenceRho <= 0.0:
                            heldoutView[i] += value
                        if returnNominalHeldout:
                            nominalHeldoutView[i] += value
        for i in range(n):
            total = totalView[i]
            if replicateDependenceRho > 0.0:
                kept = 0.0
                sumSqrt = 0.0
                count = 0
                if useLambda:
                    lamValue = lambdaExp[i]
                else:
                    lamValue = 1.0
                for j in range(m):
                    if activeMask[j, i] != 0 and maskView[j, i] != 0:
                        value = lamValue / (<double>matrixMunc[j, i] + pad)
                        kept += value
                        sumSqrt += sqrt(value)
                        count += 1
                kept = _exchangeable_information(
                    kept,
                    sumSqrt,
                    count,
                    replicateDependenceRho,
                )
                keptView[i] = kept
                heldoutView[i] = total - kept
            else:
                keptView[i] = total - heldoutView[i]
            if total > 0.0:
                hView[i] = heldoutView[i] / total
            else:
                hView[i] = NAN

    if returnNominalHeldout:
        return mask, keptInfo, heldoutInfo, hFraction, nominalHeldoutInfo
    return mask, keptInfo, heldoutInfo, hFraction


cpdef dict cdeleteBlockReplicateDependenceRhoEvidence(
    real_t[:, ::1] matrixData,
    real_t[:, ::1] matrixMunc,
    cnp.uint8_t[:, ::1] activeMask,
    cnp.int32_t[::1] blockFold,
    Py_ssize_t[::1] repsByBlockCount,
    Py_ssize_t[:, ::1] repsByBlock,
    double[::1] signal,
    double[::1] lambdaExp,
    bint useLambda,
    double pad,
    Py_ssize_t blockLen,
    Py_ssize_t fold,
):
    cdef Py_ssize_t m = matrixData.shape[0]
    cdef Py_ssize_t n = matrixData.shape[1]
    if matrixMunc.shape[0] != m or matrixMunc.shape[1] != n:
        raise ValueError("matrixMunc must match matrixData shape")
    if activeMask.shape[0] != m or activeMask.shape[1] != n:
        raise ValueError("activeMask must match matrixData shape")
    if signal.shape[0] != n:
        raise ValueError("signal must match interval count")
    if useLambda and lambdaExp.shape[0] != n:
        raise ValueError("fullObservationPrecision must match interval count")
    if blockLen < 1:
        raise ValueError("replicate-dependence block length must be positive")
    if fold < 0:
        raise ValueError("fold must be nonnegative")
    if not isfinite(pad):
        raise ValueError("observation information pad must be finite")
    if m < 2 or n < 1:
        return {
            "fisher_z_weighted_sum": 0.0,
            "weight_sum": 0.0,
            "block_count": 0,
            "pair_count": 0,
            "rho_upper_bound": 0.25,
        }

    cdef Py_ssize_t blockCount = (n + blockLen - 1) // blockLen
    if (
        blockFold.shape[0] != blockCount
        or repsByBlockCount.shape[0] != blockCount
        or repsByBlock.shape[0] != blockCount
    ):
        raise ValueError("fold spec has inconsistent block count")
    cdef Py_ssize_t deleteSlotCount = repsByBlock.shape[1]
    if deleteSlotCount < m:
        raise ValueError("fold spec replicate matrix must allow every sample")

    cdef Py_ssize_t block, start, end, i, j, k, h, h2
    cdef Py_ssize_t pairCountTotal = 0
    cdef Py_ssize_t validBlockCount = 0
    cdef Py_ssize_t count, deleteCount
    cdef bint blockHasPair
    cdef double denomJ, denomK, lamValue, residualJ, residualK
    cdef double sumJ, sumK, sumJJ, sumKK, sumJK
    cdef double cov, varJ, varK, corr, zValue, weight
    cdef double zSum = 0.0
    cdef double weightSum = 0.0
    cdef double corrBound = 0.25

    cdef Py_ssize_t rep
    for block in range(blockCount):
        if blockFold[block] < 0:
            raise ValueError("fold spec contains negative fold id")
        deleteCount = repsByBlockCount[block]
        if deleteCount < 1 or deleteCount > m or deleteCount > deleteSlotCount:
            raise ValueError("fold spec deleted-replicate count is out of bounds")
        for h in range(deleteCount):
            rep = repsByBlock[block, h]
            if rep < 0 or rep >= m:
                raise ValueError("fold spec replicate is out of bounds")
            for h2 in range(h):
                if repsByBlock[block, h2] == rep:
                    raise ValueError("fold spec contains a duplicate replicate")

    with nogil:
        for block in range(blockCount):
            if blockFold[block] != fold:
                continue
            deleteCount = repsByBlockCount[block]
            if deleteCount < 2:
                continue
            start = block * blockLen
            end = start + blockLen
            if end > n:
                end = n
            blockHasPair = False
            for h in range(deleteCount - 1):
                j = repsByBlock[block, h]
                for h2 in range(h + 1, deleteCount):
                    k = repsByBlock[block, h2]
                    count = 0
                    sumJ = 0.0
                    sumK = 0.0
                    sumJJ = 0.0
                    sumKK = 0.0
                    sumJK = 0.0
                    for i in range(start, end):
                        if activeMask[j, i] != 0 and activeMask[k, i] != 0:
                            if useLambda:
                                lamValue = lambdaExp[i]
                            else:
                                lamValue = 1.0
                            denomJ = <double>matrixMunc[j, i] + pad
                            denomK = <double>matrixMunc[k, i] + pad
                            if denomJ > 0.0 and denomK > 0.0 and lamValue > 0.0:
                                residualJ = (
                                    (<double>matrixData[j, i] - signal[i])
                                    * sqrt(lamValue / denomJ)
                                )
                                residualK = (
                                    (<double>matrixData[k, i] - signal[i])
                                    * sqrt(lamValue / denomK)
                                )
                                if isfinite(residualJ) and isfinite(residualK):
                                    sumJ += residualJ
                                    sumK += residualK
                                    sumJJ += residualJ * residualJ
                                    sumKK += residualK * residualK
                                    sumJK += residualJ * residualK
                                    count += 1
                    if count >= 4:
                        cov = sumJK - sumJ * sumK / <double>count
                        varJ = sumJJ - sumJ * sumJ / <double>count
                        varK = sumKK - sumK * sumK / <double>count
                        if varJ > 0.0 and varK > 0.0:
                            corr = cov / sqrt(varJ * varK)
                            corr = _clip_double(corr, -corrBound, corrBound)
                            zValue = 0.5 * log((1.0 + corr) / (1.0 - corr))
                            weight = <double>count - 3.0
                            if weight < 1.0:
                                weight = 1.0
                            zSum += weight * zValue
                            weightSum += weight
                            pairCountTotal += 1
                            blockHasPair = True
            if blockHasPair:
                validBlockCount += 1

    return {
        "fisher_z_weighted_sum": float(zSum),
        "weight_sum": float(weightSum),
        "block_count": int(validBlockCount),
        "pair_count": int(pairCountTotal),
        "rho_upper_bound": float(corrBound),
    }


cpdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] cmakeFoldMasks(
    Py_ssize_t m,
    Py_ssize_t n,
    Py_ssize_t blockLen,
    Py_ssize_t folds,
    double deletionProbability,
    long seed,
):
    cdef object foldSpec = cmakeFoldSpec(
        m, n, blockLen, folds, deletionProbability, seed
    )
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] blockFoldArr = foldSpec[0]
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] repsByBlockCountArr = foldSpec[1]
    cdef cnp.ndarray[cnp.intp_t, ndim=2, mode="c"] repsByBlockArr = foldSpec[2]
    cdef Py_ssize_t blockCount = blockFoldArr.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=3, mode="c"] masks = np.ones(
        (folds, m, n), dtype=np.uint8
    )
    cdef cnp.uint8_t[:, :, ::1] masksView = masks
    cdef cnp.int32_t[::1] blockFold = blockFoldArr
    cdef Py_ssize_t[::1] repsByBlockCount = repsByBlockCountArr
    cdef Py_ssize_t[:, ::1] repsByBlock = repsByBlockArr
    cdef Py_ssize_t block, start, end, i, h, rep, fold, deleteCount

    with nogil:
        for block in range(blockCount):
            start = block * blockLen
            end = start + blockLen
            if end > n:
                end = n
            fold = blockFold[block]
            deleteCount = repsByBlockCount[block]
            for h in range(deleteCount):
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


cpdef dict cdeleteBlockPostFitDiagnostics(
    object residual,
    object pDelta,
    object factorByInterval,
    object intervalIndex,
    object blockIndex,
    object targetBlockMask,
    object fitRows,
    double varianceFloor,
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] residualArr = np.ascontiguousarray(
        residual, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] pDeltaArr = np.ascontiguousarray(
        pDelta, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] factorArr = np.ascontiguousarray(
        factorByInterval, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] intervalArr = np.ascontiguousarray(
        intervalIndex, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockArr = np.ascontiguousarray(
        blockIndex, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, mode="c"] targetMaskArr = np.ascontiguousarray(
        targetBlockMask, dtype=np.uint8
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] fitRowsArr = np.ascontiguousarray(
        fitRows, dtype=np.int64
    )
    if (
        residualArr.ndim != 1
        or pDeltaArr.ndim != 1
        or factorArr.ndim != 1
        or intervalArr.ndim != 1
        or blockArr.ndim != 1
        or targetMaskArr.ndim != 1
        or fitRowsArr.ndim != 1
    ):
        raise ValueError("delete-block post-fit inputs must be one-dimensional")
    cdef Py_ssize_t n = residualArr.shape[0]
    cdef Py_ssize_t fitCount = fitRowsArr.shape[0]
    cdef Py_ssize_t factorCount = factorArr.shape[0]
    cdef Py_ssize_t blockCount = targetMaskArr.shape[0]
    if pDeltaArr.shape[0] != n or intervalArr.shape[0] != n or blockArr.shape[0] != n:
        raise ValueError("delete-block post-fit inputs have inconsistent dimensions")

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sdBeforeAll = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sdAfterAll = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sdBeforeFit = np.empty(
        fitCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sdAfterFit = np.empty(
        fitCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] heldFactorFit = np.empty(
        fitCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] blockScoreArr = np.full(
        blockCount, -1.0, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] blockCellCountArr = np.zeros(
        blockCount, dtype=np.int64
    )
    cdef double[::1] residualView = residualArr
    cdef double[::1] pDeltaView = pDeltaArr
    cdef double[::1] factorView = factorArr
    cdef cnp.int64_t[::1] intervalView = intervalArr
    cdef cnp.int64_t[::1] blockView = blockArr
    cdef cnp.uint8_t[::1] targetMaskView = targetMaskArr
    cdef cnp.int64_t[::1] fitRowsView = fitRowsArr
    cdef double[::1] sdBeforeAllView = sdBeforeAll
    cdef double[::1] sdAfterAllView = sdAfterAll
    cdef double[::1] sdBeforeFitView = sdBeforeFit
    cdef double[::1] sdAfterFitView = sdAfterFit
    cdef double[::1] heldFactorFitView = heldFactorFit
    cdef double[::1] blockScoreView = blockScoreArr
    cdef cnp.int64_t[::1] blockCellCountView = blockCellCountArr
    cdef Py_ssize_t k, f, outCount, outIndex
    cdef cnp.int64_t interval, block, row
    cdef double beforeVar, afterVar, score, factorValue
    cdef Py_ssize_t badRow = -1
    cdef Py_ssize_t badFitRow = -1

    with nogil:
        for k in range(n):
            interval = intervalView[k]
            if interval < 0 or interval >= factorCount:
                badRow = k
                break
            factorValue = factorView[interval]
            beforeVar = pDeltaView[k]
            if beforeVar < varianceFloor:
                beforeVar = varianceFloor
            sdBeforeAllView[k] = sqrt(beforeVar)
            afterVar = factorValue * pDeltaView[k]
            if afterVar < varianceFloor:
                afterVar = varianceFloor
            sdAfterAllView[k] = sqrt(afterVar)
            block = blockView[k]
            if block >= 0 and block < blockCount and targetMaskView[block] != 0:
                afterVar = factorValue * pDeltaView[k]
                if afterVar > varianceFloor:
                    score = fabs(residualView[k]) / sqrt(afterVar)
                    if score > blockScoreView[block]:
                        blockScoreView[block] = score
                    blockCellCountView[block] += 1
    if badRow >= 0:
        raise ValueError("delete-block post-fit interval index is out of bounds")

    with nogil:
        for f in range(fitCount):
            row = fitRowsView[f]
            if row < 0 or row >= n:
                badFitRow = f
                break
            interval = intervalView[row]
            sdBeforeFitView[f] = sdBeforeAllView[row]
            sdAfterFitView[f] = sdAfterAllView[row]
            heldFactorFitView[f] = factorView[interval]
    if badFitRow >= 0:
        raise ValueError("delete-block post-fit fit row is out of bounds")

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

    return {
        "sd_before_all": sdBeforeAll,
        "sd_after_all": sdAfterAll,
        "sd_before_fit": sdBeforeFit,
        "sd_after_fit": sdAfterFit,
        "held_factor_fit": heldFactorFit,
        "target_block_ids": blockOut,
        "target_block_scores": scoreOut,
        "target_block_cell_counts": countOut,
    }


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

    if USE_OPENMP:
        if n >= OPENMP_FACTOR_MIN_ROWS:
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
        else:
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
    else:
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


cpdef tuple csegShrinkBootstrapLogFactorsCompact(
    object ratio,
    object rowWeight,
    object groupCode,
    object bootstrapMultiplier,
    object scopeRowIndex,
    object scopeOffset,
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
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] groupArr = np.ascontiguousarray(
        np.asarray(groupCode, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] multArr = np.ascontiguousarray(
        np.asarray(bootstrapMultiplier, dtype=np.float64), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] indexArr = np.ascontiguousarray(
        np.asarray(scopeRowIndex, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] offsetArr = np.ascontiguousarray(
        np.asarray(scopeOffset, dtype=np.int64).reshape(-1), dtype=np.int64
    )
    cdef Py_ssize_t n = ratioArr.shape[0]
    if weightArr.shape[0] != n or groupArr.shape[0] != n:
        raise ValueError("segShrink compact bootstrap inputs must have the same length")
    if offsetArr.shape[0] < 2:
        raise ValueError("segShrink compact bootstrap offsets must define at least one scope")
    if not (target > 0.0 and target <= 1.0):
        raise ValueError("segShrink compact bootstrap target must be in (0, 1]")
    if not (z > 0.0):
        raise ValueError("segShrink compact bootstrap z must be positive")
    if not (factorMin > 0.0 and factorMax >= factorMin):
        raise ValueError("segShrink compact bootstrap factor bounds are invalid")
    cdef Py_ssize_t replicateCount = multArr.shape[0]
    cdef Py_ssize_t groupCount = multArr.shape[1]
    cdef Py_ssize_t indexCount = indexArr.shape[0]
    cdef Py_ssize_t scopeCount = offsetArr.shape[0] - 1
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] baseLog = np.full(scopeCount, np.nan, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] bootLog = np.full((scopeCount, replicateCount), np.nan, dtype=np.float64)
    cdef double[::1] ratioView = ratioArr
    cdef double[::1] weightView = weightArr
    cdef cnp.int64_t[::1] groupView = groupArr
    cdef double[:, ::1] multView = multArr
    cdef cnp.int64_t[::1] indexView = indexArr
    cdef cnp.int64_t[::1] offsetView = offsetArr
    cdef double[::1] baseView = baseLog
    cdef double[:, ::1] bootView = bootLog
    cdef Py_ssize_t s, r, i, j, start, stop, g
    cdef double total, threshold, cumulative, w, qValue, factor
    cdef cnp.int64_t group
    if offsetView[0] != 0 or offsetView[scopeCount] != indexCount:
        raise ValueError("segShrink compact bootstrap offsets do not span row indexes")
    for s in range(scopeCount):
        if offsetView[s] > offsetView[s + 1]:
            raise ValueError("segShrink compact bootstrap offsets must be sorted")
    for i in range(n):
        if not isfinite(ratioView[i]):
            raise ValueError("segShrink compact bootstrap ratios must be finite")
        if not isfinite(weightView[i]) or weightView[i] <= 0.0:
            raise ValueError("segShrink compact bootstrap weights must be finite and positive")
        group = groupView[i]
        if group < 0 or group >= groupCount:
            raise ValueError("segShrink compact bootstrap groups are out of range")
    for j in range(indexCount):
        i = indexView[j]
        if i < 0 or i >= n:
            raise ValueError("segShrink compact bootstrap row indexes are out of range")
    for r in range(replicateCount):
        for g in range(groupCount):
            if not isfinite(multView[r, g]) or multView[r, g] < 0.0:
                raise ValueError("segShrink compact bootstrap multipliers must be finite and nonnegative")
    with nogil:
        for s in range(scopeCount):
            start = offsetView[s]
            stop = offsetView[s + 1]
            total = 0.0
            for j in range(start, stop):
                i = indexView[j]
                total += weightView[i]
            if total > 0.0:
                threshold = target * total
                cumulative = 0.0
                qValue = NAN
                for j in range(start, stop):
                    i = indexView[j]
                    cumulative += weightView[i]
                    if cumulative >= threshold:
                        qValue = ratioView[i]
                        break
                factor = (qValue / z) * (qValue / z)
                factor = _clip_double(factor, factorMin, factorMax)
                if factor > 0.0:
                    baseView[s] = log(factor)
            for r in range(replicateCount):
                total = 0.0
                for j in range(start, stop):
                    i = indexView[j]
                    group = groupView[i]
                    w = weightView[i] * multView[r, group]
                    if w > 0.0:
                        total += w
                if not (total > 0.0):
                    continue
                threshold = target * total
                cumulative = 0.0
                qValue = NAN
                for j in range(start, stop):
                    i = indexView[j]
                    group = groupView[i]
                    w = weightView[i] * multView[r, group]
                    if w > 0.0:
                        cumulative += w
                        if cumulative >= threshold:
                            qValue = ratioView[i]
                            break
                factor = (qValue / z) * (qValue / z)
                factor = _clip_double(factor, factorMin, factorMax)
                if factor > 0.0:
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
    cdef Py_ssize_t logCount = logArr.shape[0]
    cdef cnp.int32_t segment
    cdef double f, variance
    if USE_OPENMP:
        if n >= OPENMP_APPLY_MIN_ROWS:
            for i in prange(n, nogil=True, schedule="static"):
                segment = segmentView[i]
                if segment >= 0 and segment < logCount and isfinite(logView[segment]):
                    f = exp(logView[segment])
                else:
                    f = 1.0
                factorView[i] = f
                variance = f * pView[i]
                if variance < positiveFloor or not isfinite(variance):
                    variance = positiveFloor
                calView[i] = <cnp.float32_t>sqrt(variance)
        else:
            with nogil:
                for i in range(n):
                    segment = segmentView[i]
                    if segment >= 0 and segment < logCount and isfinite(logView[segment]):
                        f = exp(logView[segment])
                    else:
                        f = 1.0
                    factorView[i] = f
                    variance = f * pView[i]
                    if variance < positiveFloor or not isfinite(variance):
                        variance = positiveFloor
                    calView[i] = <cnp.float32_t>sqrt(variance)
    else:
        with nogil:
            for i in range(n):
                segment = segmentView[i]
                if segment >= 0 and segment < logCount and isfinite(logView[segment]):
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
    cdef cnp.ndarray[cnp.int32_t, ndim=1, mode="c"] rowGroup = np.zeros(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] groupCountArr = np.zeros(
        groupCount, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] groupOffsetArr = np.zeros(
        groupCount, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] groupCursorArr = np.zeros(
        groupCount, dtype=np.int64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sumBeforeArr = np.zeros(
        groupCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] sumAfterArr = np.zeros(
        groupCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] hitBeforeArr = np.zeros(
        rowCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] hitAfterArr = np.zeros(
        rowCount, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] beforeOverall = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] afterOverall = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] beforeByGroup = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] afterByGroup = np.empty(
        n, dtype=np.float64
    )
    cdef cnp.int32_t[::1] rowGroupView = rowGroup
    cdef cnp.int32_t[::1] decileView = decile
    cdef cnp.int64_t[::1] groupCountView = groupCountArr
    cdef cnp.int64_t[::1] groupOffsetView = groupOffsetArr
    cdef cnp.int64_t[::1] groupCursorView = groupCursorArr
    cdef double[::1] sumBeforeView = sumBeforeArr
    cdef double[::1] sumAfterView = sumAfterArr
    cdef double[::1] hitBeforeView = hitBeforeArr
    cdef double[::1] hitAfterView = hitAfterArr
    cdef double[::1] beforeOverallView = beforeOverall
    cdef double[::1] afterOverallView = afterOverall
    cdef double[::1] beforeByGroupView = beforeByGroup
    cdef double[::1] afterByGroupView = afterByGroup
    cdef Py_ssize_t g, t, k, row, slot, totalGrouped
    cdef int groupCode, code
    cdef double z, widthScale, absResidual, beforeValue, afterValue
    cdef double medBeforeBase, medAfterBase, q90BeforeBase, q90AfterBase
    cdef double meanBeforeBase, meanAfterBase

    with nogil:
        groupCountView[0] = n
        for k in range(n):
            code = decileView[k]
            slot = 0
            if code >= 0:
                for g in range(1, groupCount):
                    if groupView[g] == code:
                        slot = g
                        break
            rowGroupView[k] = <cnp.int32_t>slot
            if slot > 0:
                groupCountView[slot] += 1

    totalGrouped = 0
    for g in range(1, groupCount):
        groupOffsetView[g] = totalGrouped
        totalGrouped += groupCountView[g]

    with nogil:
        for k in range(n):
            beforeValue = <double>sdBefore[k]
            afterValue = <double>sdAfter[k]
            beforeOverallView[k] = beforeValue
            afterOverallView[k] = afterValue
            sumBeforeView[0] += beforeValue
            sumAfterView[0] += afterValue
            slot = rowGroupView[k]
            if slot > 0:
                row = groupOffsetView[slot] + groupCursorView[slot]
                beforeByGroupView[row] = beforeValue
                afterByGroupView[row] = afterValue
                groupCursorView[slot] += 1
                sumBeforeView[slot] += beforeValue
                sumAfterView[slot] += afterValue
            absResidual = fabs(<double>residual[k])
            for t in range(targetCount):
                z = <double>targetZ[t]
                row = t * groupCount
                if absResidual <= z * beforeValue:
                    hitBeforeView[row] += 1.0
                if absResidual <= z * afterValue:
                    hitAfterView[row] += 1.0
                if slot > 0:
                    row = t * groupCount + slot
                    if absResidual <= z * beforeValue:
                        hitBeforeView[row] += 1.0
                    if absResidual <= z * afterValue:
                        hitAfterView[row] += 1.0

    for g in range(groupCount):
        groupCode = groupView[g]
        if groupCountView[g] > 0:
            if g == 0:
                medBeforeBase = float(np.quantile(beforeOverall, medianQuantile))
                medAfterBase = float(np.quantile(afterOverall, medianQuantile))
                q90BeforeBase = float(np.quantile(beforeOverall, highWidthQuantile))
                q90AfterBase = float(np.quantile(afterOverall, highWidthQuantile))
            else:
                slot = groupOffsetView[g]
                row = slot + groupCountView[g]
                medBeforeBase = float(np.quantile(beforeByGroup[slot:row], medianQuantile))
                medAfterBase = float(np.quantile(afterByGroup[slot:row], medianQuantile))
                q90BeforeBase = float(np.quantile(beforeByGroup[slot:row], highWidthQuantile))
                q90AfterBase = float(np.quantile(afterByGroup[slot:row], highWidthQuantile))
            meanBeforeBase = sumBeforeView[g] / groupCountView[g]
            meanAfterBase = sumAfterView[g] / groupCountView[g]
        else:
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
            groupOut[row] = groupCode
            targetOut[row] = <double>targets[t]
            nOut[row] = groupCountView[g]
            covBefore[row] = (
                hitBeforeView[row] / groupCountView[g] if groupCountView[g] > 0 else np.nan
            )
            covAfter[row] = (
                hitAfterView[row] / groupCountView[g] if groupCountView[g] > 0 else np.nan
            )
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

cpdef tuple cdeleteBlockTargetBlockScores(
    object residual,
    object pDelta,
    object factorByInterval,
    object intervalIndex,
    object blockIndex,
    object targetBlockMask,
    double positiveFloor,
):
    return cdeleteBlockBlockScores(
        residual,
        pDelta,
        factorByInterval,
        intervalIndex,
        blockIndex,
        targetBlockMask,
        varianceFloor=positiveFloor,
    )
