from __future__ import annotations

from typing import Any

import numpy as np

from . import cuncertainty as _cuncertainty


SEGSHRINK_MODEL = "segShrink"


def bootstrapMultipliers(
    *,
    groupCount: int,
    replicateCount: int,
    seed: int,
) -> np.ndarray:
    if groupCount < 1:
        return np.zeros((int(replicateCount), 0), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    return rng.poisson(
        1.0,
        size=(int(replicateCount), int(groupCount)),
    ).astype(np.float64, copy=False)


def _bootstrapVariance(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("inf")
    variance = float(np.var(finite, ddof=1))
    if not np.isfinite(variance) or variance <= 0.0:
        return float("inf")
    return variance


def _denseGroupCodes(groupCode: np.ndarray) -> tuple[np.ndarray, int]:
    groupCode = np.asarray(groupCode, dtype=np.int64).reshape(-1)
    valid = groupCode >= 0
    dense = np.full(groupCode.shape[0], -1, dtype=np.int64)
    if not np.any(valid):
        return dense, 0
    unique, inverse = np.unique(groupCode[valid], return_inverse=True)
    dense[valid] = inverse.astype(np.int64, copy=False)
    return dense, int(unique.size)


def _compactScopeRows(
    *,
    ratio: np.ndarray,
    rowWeight: np.ndarray,
    rowSegment: np.ndarray,
    groupCode: np.ndarray,
    segmentCount: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    ratio = np.asarray(ratio, dtype=np.float64).reshape(-1)
    rowWeight = np.asarray(rowWeight, dtype=np.float64).reshape(-1)
    rowSegment = np.asarray(rowSegment, dtype=np.int32).reshape(-1)
    groupCode = np.asarray(groupCode, dtype=np.int64).reshape(-1)
    if not (
        ratio.shape[0]
        == rowWeight.shape[0]
        == rowSegment.shape[0]
        == groupCode.shape[0]
    ):
        raise ValueError("segShrink compact score inputs must have the same length")
    if ratio.size == 0:
        scopeCount = int(segmentCount) + 2
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.zeros(scopeCount + 1, dtype=np.int64),
            scopeCount,
        )
    if np.any((rowSegment < 0) | (rowSegment >= int(segmentCount))):
        raise ValueError("segShrink compact score segments are out of range")
    if np.any(groupCode < 0):
        raise ValueError("segShrink compact score groups are out of range")
    scopeCount = int(segmentCount) + 2
    rowCount = int(ratio.size)
    rowIndexAll = np.tile(np.arange(rowCount, dtype=np.int64), 3)
    scopeCode = np.concatenate(
        [
            np.zeros(rowCount, dtype=np.int32),
            np.ones(rowCount, dtype=np.int32),
            (rowSegment + 2).astype(np.int32, copy=False),
        ]
    )
    order = np.lexsort((ratio[rowIndexAll], scopeCode))
    scopeSorted = scopeCode[order]
    scopeOffset = np.searchsorted(
        scopeSorted,
        np.arange(scopeCount + 1, dtype=np.int32),
        side="left",
    ).astype(np.int64, copy=False)
    rowIndex = rowIndexAll[order]
    return (
        ratio,
        rowWeight,
        groupCode,
        rowIndex,
        scopeOffset,
        scopeCount,
    )


def fitSingleContig(
    *,
    residual: np.ndarray,
    pDelta: np.ndarray,
    rowWeight: np.ndarray,
    intervalIndex: np.ndarray,
    foldIndex: np.ndarray,
    blockIDX: np.ndarray,
    fullP: np.ndarray,
    target: float,
    targetZ: float,
    factorMin: float,
    factorMax: float,
    segmentCount: int,
    bootstrapReplicates: int,
    seed: int,
    positiveFloor: float,
) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    pDelta = np.asarray(pDelta, dtype=np.float64).reshape(-1)
    rowWeight = np.asarray(rowWeight, dtype=np.float64).reshape(-1)
    intervalIndex = np.asarray(intervalIndex, dtype=np.int64).reshape(-1)
    foldIndex = np.asarray(foldIndex, dtype=np.int64).reshape(-1)
    blockIDX = np.asarray(blockIDX, dtype=np.int64).reshape(-1)
    fullP = np.asarray(fullP, dtype=np.float64).reshape(-1)
    if not (
        residual.shape[0]
        == pDelta.shape[0]
        == rowWeight.shape[0]
        == intervalIndex.shape[0]
        == foldIndex.shape[0]
        == blockIDX.shape[0]
    ):
        raise ValueError("segShrink score inputs must have the same length")
    segmentByInterval = _cuncertainty.csegShrinkSegmentCodes(
        int(fullP.shape[0]),
        int(segmentCount),
    )
    segmentCountEffective = int(np.max(segmentByInterval)) + 1
    groupCodeRaw = _cuncertainty.csegShrinkGroupCodes(0, foldIndex, blockIDX)
    groupCode, groupCount = _denseGroupCodes(groupCodeRaw)
    validVariance = (
        np.isfinite(residual)
        & np.isfinite(pDelta)
        & (pDelta > float(positiveFloor))
        & np.isfinite(rowWeight)
        & (rowWeight > 0.0)
        & (intervalIndex >= 0)
        & (intervalIndex < fullP.shape[0])
    )
    if not np.any(validVariance):
        raise ValueError("segShrink factor fit has no valid score rows")
    validScore = validVariance & (groupCode >= 0)
    validSegment = segmentByInterval[intervalIndex[validVariance]].astype(
        np.int32,
        copy=False,
    )
    segmentRows = np.bincount(
        validSegment,
        minlength=segmentCountEffective,
    ).astype(np.int64, copy=False)
    scoreSegment = segmentByInterval[intervalIndex[validScore]].astype(
        np.int32,
        copy=False,
    )
    (
        ratioCompact,
        weightCompact,
        groupCompact,
        rowIndex,
        scopeOffset,
        scopeCount,
    ) = _compactScopeRows(
        ratio=np.abs(residual[validScore]) / np.sqrt(pDelta[validScore]),
        rowWeight=rowWeight[validScore],
        rowSegment=scoreSegment,
        groupCode=groupCode[validScore],
        segmentCount=segmentCountEffective,
    )
    if ratioCompact.size == 0:
        raise ValueError("segShrink factor fit has no finite weighted score rows")
    multipliers = bootstrapMultipliers(
        groupCount=groupCount,
        replicateCount=int(bootstrapReplicates),
        seed=int(seed),
    )
    baseLog, bootLog = _cuncertainty.csegShrinkBootstrapLogFactorsCompact(
        ratioCompact,
        weightCompact,
        groupCompact,
        multipliers,
        rowIndex,
        scopeOffset,
        float(target),
        float(targetZ),
        float(factorMin),
        float(factorMax),
    )
    baseLog = np.asarray(baseLog, dtype=np.float64)
    bootLog = np.asarray(bootLog, dtype=np.float64)
    scopeVariance = np.array(
        [_bootstrapVariance(bootLog[idx, :]) for idx in range(scopeCount)],
        dtype=np.float64,
    )
    genomeLog = float(baseLog[0])
    if not np.isfinite(genomeLog):
        raise ValueError("segShrink processed-genome factor is not finite")
    contigLog = np.asarray([baseLog[1]], dtype=np.float64)
    contigVariance = np.asarray([scopeVariance[1]], dtype=np.float64)
    segmentLog = np.asarray(baseLog[2:], dtype=np.float64)
    segmentVariance = np.asarray(scopeVariance[2:], dtype=np.float64)
    segmentContigIndex = np.zeros(segmentCountEffective, dtype=np.int32)
    empiricalBayes = _cuncertainty.csegShrinkEmpiricalBayes(
        genomeLog,
        contigLog,
        contigVariance,
        segmentLog,
        segmentVariance,
        segmentContigIndex,
    )
    segmentTheta = np.asarray(empiricalBayes["segmentTheta"], dtype=np.float64)
    factor, _calibrated = _cuncertainty.csegShrinkApplyFactors(
        segmentByInterval,
        segmentTheta,
        fullP,
        float(positiveFloor),
    )
    factor = np.maximum(np.asarray(factor, dtype=np.float64), 1.0)
    calibrated = np.sqrt(np.maximum(factor * fullP, positiveFloor)).astype(np.float32)
    segmentShrinkage = []
    for idx in range(segmentCountEffective):
        rawLog = float(segmentLog[idx]) if idx < segmentLog.size else float("nan")
        rawFactor = float(np.exp(rawLog)) if np.isfinite(rawLog) else None
        variance = float(segmentVariance[idx]) if idx < segmentVariance.size else float("inf")
        alpha = float(np.asarray(empiricalBayes["segmentAlpha"], dtype=np.float64)[idx])
        theta = float(segmentTheta[idx])
        reason = "none"
        if rawFactor is None:
            reason = "missingRawFactor"
        elif not np.isfinite(variance):
            reason = "invalidBootstrapVariance"
        elif alpha <= 0.0:
            reason = "collapsedToContig"
        segmentShrinkage.append(
            {
                "segment": int(idx),
                "rows": int(segmentRows[idx]),
                "rawFactor": rawFactor,
                "bootstrapVariance": None if not np.isfinite(variance) else variance,
                "shrinkageWeight": alpha,
                "factor": float(np.exp(theta)) if np.isfinite(theta) else None,
                "fallbackReason": reason,
            }
        )
    contigTheta = np.asarray(empiricalBayes["contigTheta"], dtype=np.float64)
    contigAlpha = np.asarray(empiricalBayes["contigAlpha"], dtype=np.float64)
    contigFactor = float(np.exp(contigTheta[0])) if contigTheta.size else float(np.exp(genomeLog))
    genomeFactor = float(np.exp(genomeLog))
    modelMeta = {
        "success": True,
        "factor_model": SEGSHRINK_MODEL,
        "factorModel": SEGSHRINK_MODEL,
        "global_factor": contigFactor,
        "global_sd_multiplier": float(np.sqrt(contigFactor)),
        "global_factor_target": float(target),
        "global_factor_target_z": float(targetZ),
        "hierarchyScope": "singleProcessedContig",
        "processedContigCount": 1,
        "segmentCount": int(segmentCountEffective),
        "bootstrapReplicates": int(bootstrapReplicates),
        "blockIDXUnitCount": int(groupCount),
        "genomeFactor": genomeFactor,
        "tauContigSq": float(empiricalBayes["tauContigSq"]),
        "tauSegmentSq": float(empiricalBayes["tauSegmentSq"]),
        "contigShrinkage": [
            {
                "contigOrdinal": 0,
                "rawFactor": float(np.exp(contigLog[0])),
                "bootstrapVariance": (
                    None
                    if not np.isfinite(contigVariance[0])
                    else float(contigVariance[0])
                ),
                "shrinkageWeight": float(contigAlpha[0]) if contigAlpha.size else 0.0,
                "factor": contigFactor,
            }
        ],
        "segmentShrinkage": segmentShrinkage,
    }
    return {
        "factor": factor,
        "calibrated": calibrated,
        "modelMeta": modelMeta,
        "segmentByInterval": np.asarray(segmentByInterval, dtype=np.int32),
        "segmentRawLogFactor": segmentLog,
        "segmentBootstrapVariance": segmentVariance,
        "segmentShrinkageWeight": np.asarray(empiricalBayes["segmentAlpha"], dtype=np.float64),
        "refitPolicy": {},
    }


def _finiteLogFactor(value: Any) -> float:
    try:
        valueFloat = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(valueFloat) or valueFloat <= 0.0:
        return float("nan")
    return float(np.log(valueFloat))


def _finiteVariance(value: Any) -> float:
    try:
        valueFloat = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if not np.isfinite(valueFloat) or valueFloat < 0.0:
        return float("inf")
    return valueFloat


def _processedGenomeLog(contigLog: np.ndarray, contigVariance: np.ndarray) -> float:
    finite = np.isfinite(contigLog)
    finiteVar = finite & np.isfinite(contigVariance) & (contigVariance > 0.0)
    if np.any(finiteVar):
        weights = 1.0 / np.maximum(contigVariance[finiteVar], 1.0e-12)
        return float(np.sum(weights * contigLog[finiteVar]) / np.sum(weights))
    if np.any(finite):
        return float(np.mean(contigLog[finite]))
    raise ValueError("segShrink processed-genome factor is not finite")


def combinePreparedContigs(
    prepared: list[dict[str, Any]],
    *,
    positiveFloor: float,
) -> list[dict[str, Any]]:
    if not prepared:
        raise ValueError("segShrink uncertainty calibration has no processed contigs")
    contigCount = int(len(prepared))
    if contigCount == 1:
        item = dict(prepared[0])
        model = dict(item["model"])
        model["hierarchyScope"] = "singleProcessedContig"
        model["processedContigCount"] = 1
        item["model"] = model
        return [item]

    contigLog = np.empty(contigCount, dtype=np.float64)
    contigVariance = np.empty(contigCount, dtype=np.float64)
    segmentLogPieces: list[np.ndarray] = []
    segmentVariancePieces: list[np.ndarray] = []
    segmentContigPieces: list[np.ndarray] = []
    segmentRowsByContig: list[list[dict[str, Any]]] = []

    for contigOrdinal, item in enumerate(prepared):
        model = item["model"]
        contigRows = list(model.get("contigShrinkage", ()))
        contigRow = contigRows[0] if contigRows else {}
        contigLog[contigOrdinal] = _finiteLogFactor(contigRow.get("rawFactor"))
        contigVariance[contigOrdinal] = _finiteVariance(
            contigRow.get("bootstrapVariance")
        )
        segmentRows = list(model.get("segmentShrinkage", ()))
        segmentRowsByContig.append(segmentRows)
        segmentLogPieces.append(
            np.asarray(
                [_finiteLogFactor(row.get("rawFactor")) for row in segmentRows],
                dtype=np.float64,
            )
        )
        segmentVariancePieces.append(
            np.asarray(
                [_finiteVariance(row.get("bootstrapVariance")) for row in segmentRows],
                dtype=np.float64,
            )
        )
        segmentContigPieces.append(
            np.full(len(segmentRows), contigOrdinal, dtype=np.int32)
        )

    genomeLog = _processedGenomeLog(contigLog, contigVariance)
    segmentLog = (
        np.concatenate(segmentLogPieces)
        if segmentLogPieces
        else np.empty(0, dtype=np.float64)
    )
    segmentVariance = (
        np.concatenate(segmentVariancePieces)
        if segmentVariancePieces
        else np.empty(0, dtype=np.float64)
    )
    segmentContigIndex = (
        np.concatenate(segmentContigPieces)
        if segmentContigPieces
        else np.empty(0, dtype=np.int32)
    )
    empiricalBayes = _cuncertainty.csegShrinkEmpiricalBayes(
        genomeLog,
        contigLog,
        contigVariance,
        segmentLog,
        segmentVariance,
        segmentContigIndex,
    )
    contigTheta = np.asarray(empiricalBayes["contigTheta"], dtype=np.float64)
    contigAlpha = np.asarray(empiricalBayes["contigAlpha"], dtype=np.float64)
    segmentTheta = np.asarray(empiricalBayes["segmentTheta"], dtype=np.float64)
    segmentAlpha = np.asarray(empiricalBayes["segmentAlpha"], dtype=np.float64)
    genomeFactor = float(np.exp(genomeLog))
    contigTable = []
    for contigOrdinal in range(contigCount):
        rawFactor = (
            float(np.exp(contigLog[contigOrdinal]))
            if np.isfinite(contigLog[contigOrdinal])
            else None
        )
        variance = contigVariance[contigOrdinal]
        contigTable.append(
            {
                "contigOrdinal": int(contigOrdinal),
                "chromosome": str(prepared[contigOrdinal].get("chromosome", "")),
                "rawFactor": rawFactor,
                "bootstrapVariance": None if not np.isfinite(variance) else float(variance),
                "shrinkageWeight": float(contigAlpha[contigOrdinal]),
                "factor": float(np.exp(contigTheta[contigOrdinal])),
            }
        )

    out: list[dict[str, Any]] = []
    offset = 0
    for contigOrdinal, item in enumerate(prepared):
        model = dict(item["model"])
        fullP = np.asarray(item["fullP"], dtype=np.float64).reshape(-1)
        segmentRows = segmentRowsByContig[contigOrdinal]
        localCount = len(segmentRows)
        localTheta = segmentTheta[offset:offset + localCount]
        localAlpha = segmentAlpha[offset:offset + localCount]
        segmentByInterval = _cuncertainty.csegShrinkSegmentCodes(
            int(fullP.shape[0]),
            max(localCount, 1),
        )
        factor, _calibrated = _cuncertainty.csegShrinkApplyFactors(
            segmentByInterval,
            localTheta,
            fullP,
            float(positiveFloor),
        )
        factor = np.maximum(np.asarray(factor, dtype=np.float64), 1.0)
        calibrated = np.sqrt(np.maximum(factor * fullP, positiveFloor)).astype(np.float32)
        targetCalibration = model.get("target_calibration")
        uncertaintyTrackScale = 1.0
        if isinstance(targetCalibration, dict) and bool(
            targetCalibration.get("uncertainty_track_scaled", False)
        ):
            uncertaintyTrackScale = float(
                targetCalibration.get("uncertainty_track_scale", 1.0)
            )
            if not (np.isfinite(uncertaintyTrackScale) and uncertaintyTrackScale > 0.0):
                raise ValueError("segShrink target uncertainty scale is not positive")
            calibrated = (
                np.asarray(calibrated, dtype=np.float32)
                * np.float32(uncertaintyTrackScale)
            )
        calibrated = np.maximum(
            np.asarray(calibrated, dtype=np.float32),
            np.sqrt(fullP).astype(np.float32),
        )
        segmentTable = []
        for localIDX, row in enumerate(segmentRows):
            rawLog = segmentLog[offset + localIDX]
            variance = segmentVariance[offset + localIDX]
            theta = localTheta[localIDX]
            alpha = localAlpha[localIDX]
            reason = "none"
            if not np.isfinite(rawLog):
                reason = "missingRawFactor"
            elif not np.isfinite(variance):
                reason = "invalidBootstrapVariance"
            elif alpha <= 0.0:
                reason = "collapsedToContig"
            segmentTable.append(
                {
                    **dict(row),
                    "rawFactor": float(np.exp(rawLog)) if np.isfinite(rawLog) else None,
                    "bootstrapVariance": (
                        None if not np.isfinite(variance) else float(variance)
                    ),
                    "shrinkageWeight": float(alpha),
                    "factor": float(np.exp(theta)) if np.isfinite(theta) else None,
                    "fallbackReason": reason,
                }
            )
        model.update(
            {
                "hierarchyScope": "processedGenome",
                "processedContigCount": contigCount,
                "genomeFactor": genomeFactor,
                "global_factor": float(np.exp(contigTheta[contigOrdinal])),
                "global_sd_multiplier": float(
                    np.sqrt(np.exp(contigTheta[contigOrdinal]))
                ),
                "tauContigSq": float(empiricalBayes["tauContigSq"]),
                "tauSegmentSq": float(empiricalBayes["tauSegmentSq"]),
                "contigShrinkage": contigTable,
                "segmentShrinkage": segmentTable,
            }
        )
        out.append(
            {
                **item,
                "factor": np.asarray(factor, dtype=np.float64),
                "calibrated": np.asarray(calibrated, dtype=np.float32),
                "model": model,
            }
        )
        offset += localCount
    return out
