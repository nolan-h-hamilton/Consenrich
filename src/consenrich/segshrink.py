"""(Experimental) Pooled delete-block uncertainty calibration"""

from __future__ import annotations

from collections.abc import Mapping
import os
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
        if "calibrationReplayPath" in prepared[0]:
            raise ValueError(
                "segShrink replay pooling requires at least two processed contigs"
            )
        item = dict(prepared[0])
        model = dict(item["model"])
        model["hierarchyScope"] = "singleProcessedContig"
        model["processedContigCount"] = 1
        item["model"] = model
        return [item]

    positiveFloor = float(positiveFloor)
    if not np.isfinite(positiveFloor) or positiveFloor <= 0.0:
        raise ValueError("segShrink positive floor must be finite and positive")
    expectedItemKeys = {
        "chromosome",
        "intervals",
        "fullP",
        "model",
        "calibrationReplayPath",
        "summaryRowIndex",
    }
    expectedReplayKeys = {
        "residual",
        "pDelta",
        "intervalIndex",
        "fitRows",
        "targetBlockMask",
        "deletedObservationAll",
        "coverageCodeAll",
        "coverageCodeFit",
        "summaryDecile",
    }
    replayDtypes = {
        "residual": np.dtype(np.float64),
        "pDelta": np.dtype(np.float64),
        "intervalIndex": np.dtype(np.int64),
        "fitRows": np.dtype(np.int64),
        "targetBlockMask": np.dtype(np.uint8),
        "deletedObservationAll": np.dtype(np.int64),
        "coverageCodeAll": np.dtype(np.int32),
        "coverageCodeFit": np.dtype(np.int32),
        "summaryDecile": np.dtype(np.int32),
    }
    preparedCopies: list[dict[str, Any]] = []
    seenChromosomes: set[str] = set()
    seenSummaryRows: set[int] = set()
    sharedSelectedTarget: float | None = None
    sharedTargetSignal: str | None = None
    sharedScaleFlag: bool | None = None
    sharedTargetEnabled: bool | None = None
    sharedDelta: float | None = None
    for itemOrdinal, item in enumerate(prepared):
        if not isinstance(item, Mapping):
            raise ValueError("segShrink prepared contigs must be mappings")
        itemKeys = set(item)
        if itemKeys != expectedItemKeys:
            missing = sorted(expectedItemKeys - itemKeys)
            extra = sorted(itemKeys - expectedItemKeys)
            raise ValueError(
                "segShrink prepared contig keys do not match the replay contract: "
                f"missing={missing} extra={extra}"
            )
        chromosome = item["chromosome"]
        if not isinstance(chromosome, str) or not chromosome.strip():
            raise ValueError("segShrink prepared chromosome must be a nonempty string")
        if chromosome in seenChromosomes:
            raise ValueError("segShrink prepared contig chromosomes must be unique")
        seenChromosomes.add(chromosome)
        if not isinstance(item["intervals"], np.ndarray) or not isinstance(
            item["fullP"],
            np.ndarray,
        ):
            raise ValueError("segShrink prepared intervals and fullP must be arrays")
        intervals = np.asarray(item["intervals"])
        fullP = np.asarray(item["fullP"])
        if intervals.dtype != np.dtype(np.int64):
            raise ValueError("segShrink prepared intervals dtype must be int64")
        if fullP.dtype != np.dtype(np.float64):
            raise ValueError("segShrink prepared fullP dtype must be float64")
        if intervals.ndim != 1 or fullP.ndim != 1 or intervals.shape != fullP.shape:
            raise ValueError("segShrink intervals and fullP must be aligned vectors")
        if intervals.size == 0 or np.any(intervals < 0):
            raise ValueError("segShrink intervals must be nonempty and nonnegative")
        if intervals.size > 1 and np.any(np.diff(intervals) <= 0):
            raise ValueError("segShrink intervals must be strictly increasing")
        if fullP.size == 0 or not np.all(np.isfinite(fullP)) or np.any(fullP <= 0.0):
            raise ValueError("segShrink fullP must be nonempty, finite, and positive")
        model = item["model"]
        if not isinstance(model, Mapping):
            raise ValueError("segShrink calibration model must be a mapping")
        if model.get("factor_model") != SEGSHRINK_MODEL:
            raise ValueError("segShrink prepared model must have the segShrink tag")
        if "factorModel" in model and model["factorModel"] != SEGSHRINK_MODEL:
            raise ValueError("segShrink prepared factor-model tags disagree")
        foldRefits = model.get("fold_refits")
        if not isinstance(foldRefits, Mapping):
            raise ValueError("segShrink prepared model requires fold_refits")
        blockLenValue = foldRefits.get("block_len_intervals")
        if isinstance(blockLenValue, (bool, np.bool_)) or not isinstance(
            blockLenValue,
            (int, np.integer),
        ):
            raise ValueError("segShrink prepared block length must be an integer")
        blockLenIntervals = int(blockLenValue)
        if blockLenIntervals < 1:
            raise ValueError("segShrink prepared block length must be positive")
        topBlockLen = model.get("block_len_intervals")
        if topBlockLen is not None and (
            isinstance(topBlockLen, (bool, np.bool_))
            or not isinstance(topBlockLen, (int, np.integer))
            or int(topBlockLen) != blockLenIntervals
        ):
            raise ValueError("segShrink prepared block-length metadata disagree")
        targetsValue = model.get("targets")
        if not isinstance(targetsValue, (list, tuple)) or not targetsValue:
            raise ValueError("segShrink prepared model requires targets")
        targets: list[float] = []
        for targetValue in targetsValue:
            if isinstance(targetValue, (bool, np.bool_)) or not isinstance(
                targetValue,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError("segShrink prepared targets must be numeric")
            target = float(targetValue)
            if not np.isfinite(target) or not 0.0 < target < 1.0:
                raise ValueError("segShrink prepared targets must be probabilities")
            targets.append(target)
        selectedTarget = max(targets)
        globalTargetValue = model.get("global_factor_target")
        if globalTargetValue is not None:
            if isinstance(globalTargetValue, (bool, np.bool_)) or not isinstance(
                globalTargetValue,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError("segShrink selected target must be numeric")
            if float(globalTargetValue) != selectedTarget:
                raise ValueError("segShrink selected target must be the maximum target")
        targetSignal = model.get("target_signal")
        if not isinstance(targetSignal, str) or not targetSignal:
            raise ValueError("segShrink prepared target signal must be a nonempty string")
        targetCalibration = model.get("target_calibration")
        if not isinstance(targetCalibration, Mapping):
            raise ValueError("segShrink prepared model requires target calibration")
        for key in (
            "enabled",
            "delta",
            "scale_uncertainty_by_target_calibration",
        ):
            if key not in targetCalibration:
                raise ValueError(
                    f"segShrink prepared target calibration requires {key}"
                )
        targetEnabledValue = targetCalibration["enabled"]
        scaleFlagValue = targetCalibration[
            "scale_uncertainty_by_target_calibration"
        ]
        if not isinstance(targetEnabledValue, (bool, np.bool_)):
            raise ValueError("segShrink target-calibration enabled flag must be boolean")
        if not isinstance(scaleFlagValue, (bool, np.bool_)):
            raise ValueError("segShrink target-calibration scale flag must be boolean")
        targetEnabled = bool(targetEnabledValue)
        scaleFlag = bool(scaleFlagValue)
        deltaValue = targetCalibration["delta"]
        if targetEnabled:
            if isinstance(deltaValue, (bool, np.bool_)) or not isinstance(
                deltaValue,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError("segShrink enabled target calibration requires delta")
            targetDelta = float(deltaValue)
            if not np.isfinite(targetDelta) or not 0.0 < targetDelta < 1.0:
                raise ValueError("segShrink target delta must be a probability")
        else:
            if deltaValue is not None:
                raise ValueError("segShrink disabled target calibration requires null delta")
            targetDelta = None
        contigRowsValue = model.get("contigShrinkage")
        if not isinstance(contigRowsValue, (list, tuple)) or len(contigRowsValue) != 1:
            raise ValueError("segShrink prepared contig table must have one row")
        contigRow = contigRowsValue[0]
        if not isinstance(contigRow, Mapping) or not {
            "rawFactor",
            "bootstrapVariance",
        } <= set(contigRow):
            raise ValueError("segShrink prepared contig table row is malformed")
        segmentRowsValue = model.get("segmentShrinkage")
        if not isinstance(segmentRowsValue, (list, tuple)) or not segmentRowsValue:
            raise ValueError("segShrink prepared segment table must be nonempty")
        segmentRowTotal = 0
        for segmentOrdinal, segmentRow in enumerate(segmentRowsValue):
            if not isinstance(segmentRow, Mapping) or not {
                "segment",
                "rows",
                "rawFactor",
                "bootstrapVariance",
            } <= set(segmentRow):
                raise ValueError("segShrink prepared segment table row is malformed")
            segmentValue = segmentRow["segment"]
            rowsValue = segmentRow["rows"]
            if (
                isinstance(segmentValue, (bool, np.bool_))
                or not isinstance(segmentValue, (int, np.integer))
                or int(segmentValue) != segmentOrdinal
            ):
                raise ValueError("segShrink prepared segment indices must be consecutive")
            if (
                isinstance(rowsValue, (bool, np.bool_))
                or not isinstance(rowsValue, (int, np.integer))
                or int(rowsValue) < 0
            ):
                raise ValueError(
                    "segShrink prepared segment row counts must be nonnegative"
                )
            segmentRowTotal += int(rowsValue)
        if len(segmentRowsValue) > fullP.size:
            raise ValueError("segShrink prepared segment table exceeds fullP")
        rowsFitValue = model.get("rows_fit")
        if rowsFitValue is not None and (
            isinstance(rowsFitValue, (bool, np.bool_))
            or not isinstance(rowsFitValue, (int, np.integer))
            or int(rowsFitValue) != segmentRowTotal
        ):
            raise ValueError("segShrink prepared segment rows do not match rows_fit")
        segmentCountValue = model.get("segmentCount")
        if segmentCountValue is not None and (
            isinstance(segmentCountValue, (bool, np.bool_))
            or not isinstance(segmentCountValue, (int, np.integer))
            or int(segmentCountValue) != len(segmentRowsValue)
        ):
            raise ValueError("segShrink prepared segment-count metadata disagree")
        summaryRowIndex = item["summaryRowIndex"]
        if (
            isinstance(summaryRowIndex, (bool, np.bool_))
            or not isinstance(summaryRowIndex, (int, np.integer))
            or int(summaryRowIndex) < 0
        ):
            raise ValueError("segShrink summary row index must be a nonnegative integer")
        summaryRowIndex = int(summaryRowIndex)
        if summaryRowIndex in seenSummaryRows:
            raise ValueError("segShrink summary row indices must be unique")
        seenSummaryRows.add(summaryRowIndex)
        if itemOrdinal == 0:
            sharedSelectedTarget = selectedTarget
            sharedTargetSignal = targetSignal
            sharedScaleFlag = scaleFlag
            sharedTargetEnabled = targetEnabled
            sharedDelta = targetDelta
        else:
            if selectedTarget != sharedSelectedTarget:
                raise ValueError(
                    "segShrink prepared contigs must share the selected maximum target"
                )
            if targetSignal != sharedTargetSignal:
                raise ValueError(
                    "segShrink prepared contigs must share the target signal"
                )
            if scaleFlag != sharedScaleFlag:
                raise ValueError(
                    "segShrink prepared contigs must share the target-calibration scale flag"
                )
            if targetEnabled != sharedTargetEnabled:
                raise ValueError(
                    "segShrink prepared contigs must share the target-calibration enabled flag"
                )
            if targetDelta != sharedDelta:
                raise ValueError(
                    "segShrink prepared contigs must share the target-calibration delta"
                )
        try:
            replayPath = os.fspath(item["calibrationReplayPath"])
        except TypeError as exc:
            raise ValueError("segShrink calibration replay path is invalid") from exc
        if not isinstance(replayPath, str) or not replayPath:
            raise ValueError("segShrink calibration replay path must be a nonempty string")
        if not os.path.isfile(replayPath):
            raise ValueError(f"segShrink calibration replay does not exist: {replayPath}")
        with np.load(replayPath, allow_pickle=False) as replay:
            replayKeys = set(replay.files)
            if replayKeys != expectedReplayKeys:
                missing = sorted(expectedReplayKeys - replayKeys)
                extra = sorted(replayKeys - expectedReplayKeys)
                raise ValueError(
                    "segShrink calibration replay keys do not match the contract: "
                    f"missing={missing} extra={extra}"
                )
            replayArrays = {key: np.asarray(replay[key]) for key in expectedReplayKeys}
            if any(array.ndim != 1 for array in replayArrays.values()):
                raise ValueError("segShrink calibration replay arrays must be vectors")
            for key, dtype in replayDtypes.items():
                if replayArrays[key].dtype != dtype:
                    raise ValueError(
                        f"segShrink calibration replay {key} dtype must be {dtype}"
                    )
            rowCount = int(replayArrays["residual"].size)
            if rowCount < 1:
                raise ValueError("segShrink calibration replay has no perturbation rows")
            for key in (
                "pDelta",
                "intervalIndex",
                "deletedObservationAll",
                "coverageCodeAll",
            ):
                if replayArrays[key].size != rowCount:
                    raise ValueError(
                        f"segShrink calibration replay {key} does not match residual rows"
                    )
            fitCount = int(replayArrays["fitRows"].size)
            if fitCount < 1:
                raise ValueError("segShrink calibration replay has no factor-fit rows")
            for key in ("coverageCodeFit", "summaryDecile"):
                if replayArrays[key].size != fitCount:
                    raise ValueError(
                        f"segShrink calibration replay {key} does not match fit rows"
                    )
            residual = replayArrays["residual"]
            pDelta = replayArrays["pDelta"]
            intervalIndex = replayArrays["intervalIndex"]
            fitRows = replayArrays["fitRows"]
            targetBlockMask = replayArrays["targetBlockMask"]
            deletedObservationAll = replayArrays["deletedObservationAll"]
            coverageCodeAll = replayArrays["coverageCodeAll"]
            coverageCodeFit = replayArrays["coverageCodeFit"]
            summaryDecile = replayArrays["summaryDecile"]
            if not np.all(np.isfinite(residual)):
                raise ValueError("segShrink calibration replay residual is not finite")
            if not np.all(np.isfinite(pDelta)) or np.any(pDelta <= positiveFloor):
                raise ValueError(
                    "segShrink calibration replay pDelta must exceed the positive floor"
                )
            if np.any(intervalIndex < 0) or np.any(intervalIndex >= fullP.size):
                raise ValueError("segShrink calibration replay interval index is out of bounds")
            if np.any(fitRows < 0) or np.any(fitRows >= rowCount):
                raise ValueError("segShrink calibration replay fit row is out of bounds")
            if fitRows.size > 1 and np.any(np.diff(fitRows) <= 0):
                raise ValueError(
                    "segShrink calibration replay fit rows must be strictly increasing"
                )
            blockIndex = intervalIndex // blockLenIntervals
            expectedBlockCount = int(np.max(blockIndex)) + 1
            if targetBlockMask.size != expectedBlockCount:
                raise ValueError(
                    "segShrink calibration replay target mask does not match rebuilt blocks"
                )
            if np.any((targetBlockMask != 0) & (targetBlockMask != 1)):
                raise ValueError("segShrink calibration replay target mask must be binary")
            presentBlockMask = np.zeros(expectedBlockCount, dtype=bool)
            presentBlockMask[np.unique(blockIndex)] = True
            if np.any(targetBlockMask[~presentBlockMask] != 0):
                raise ValueError(
                    "segShrink calibration replay selects a block without perturbation rows"
                )
            if not targetEnabled and np.any(targetBlockMask != 0):
                raise ValueError(
                    "segShrink disabled target calibration selects target blocks"
                )
            if np.any(deletedObservationAll < 1):
                raise ValueError(
                    "segShrink calibration replay deleted-observation counts must be positive"
                )
            if np.any((coverageCodeAll < 0) | (coverageCodeAll > 4)):
                raise ValueError(
                    "segShrink calibration replay all-row coverage code is invalid"
                )
            if np.any((coverageCodeFit < 0) | (coverageCodeFit > 4)):
                raise ValueError(
                    "segShrink calibration replay fit-row coverage code is invalid"
                )
            if np.any((summaryDecile < -1) | (summaryDecile > 9)):
                raise ValueError(
                    "segShrink calibration replay summary decile is invalid"
                )
            del (
                blockIndex,
                coverageCodeAll,
                coverageCodeFit,
                deletedObservationAll,
                fitRows,
                intervalIndex,
                pDelta,
                presentBlockMask,
                replayArrays,
                residual,
                summaryDecile,
                targetBlockMask,
            )
        intervalsView = intervals.view()
        intervalsView.flags.writeable = False
        fullPView = fullP.view()
        fullPView.flags.writeable = False
        preparedCopies.append(
            {
                "chromosome": chromosome,
                "intervals": intervalsView,
                "fullP": fullPView,
                "model": model,
                "calibrationReplayPath": replayPath,
                "summaryRowIndex": summaryRowIndex,
            }
        )

    from . import uncertainty as _uncertainty

    replayEvaluator = getattr(_uncertainty, "_evaluateSegShrinkReplay", None)
    if not callable(replayEvaluator):
        raise RuntimeError("segShrink calibration replay evaluator is unavailable")

    contigLog = np.empty(contigCount, dtype=np.float64)
    contigVariance = np.empty(contigCount, dtype=np.float64)
    segmentLogPieces: list[np.ndarray] = []
    segmentVariancePieces: list[np.ndarray] = []
    segmentContigPieces: list[np.ndarray] = []
    segmentRowsByContig: list[list[dict[str, Any]]] = []

    for contigOrdinal, item in enumerate(preparedCopies):
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
                "chromosome": str(preparedCopies[contigOrdinal]["chromosome"]),
                "rawFactor": rawFactor,
                "bootstrapVariance": None if not np.isfinite(variance) else float(variance),
                "shrinkageWeight": float(contigAlpha[contigOrdinal]),
                "factor": float(np.exp(contigTheta[contigOrdinal])),
            }
        )

    out: list[dict[str, Any]] = []
    offset = 0
    for contigOrdinal, item in enumerate(preparedCopies):
        model = dict(item["model"])
        fullP = item["fullP"]
        segmentRows = segmentRowsByContig[contigOrdinal]
        localCount = len(segmentRows)
        localTheta = segmentTheta[offset:offset + localCount]
        localAlpha = segmentAlpha[offset:offset + localCount]
        segmentByInterval = _cuncertainty.csegShrinkSegmentCodes(
            int(fullP.shape[0]),
            max(localCount, 1),
        )
        fullPWork = np.require(
            fullP,
            dtype=np.float64,
            requirements=("C", "W"),
        )
        factorRaw, _calibrated = _cuncertainty.csegShrinkApplyFactors(
            segmentByInterval,
            localTheta,
            fullPWork,
            float(positiveFloor),
        )
        del _calibrated, fullPWork
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
        replayResult = replayEvaluator(
            factorRaw=factorRaw,
            fullP=fullP,
            calibrationModel=model,
            replayPath=item["calibrationReplayPath"],
            positiveFloor=float(positiveFloor),
        )
        if not isinstance(replayResult, Mapping) or set(replayResult) != {
            "factor",
            "calibrated",
            "summary",
            "model",
        }:
            raise RuntimeError("segShrink replay result does not match the interface")
        factor = replayResult["factor"]
        calibrated = replayResult["calibrated"]
        summary = replayResult["summary"]
        replayedModel = replayResult["model"]
        factor = np.asarray(factor)
        calibrated = np.asarray(calibrated)
        if factor.ndim != 1 or calibrated.ndim != 1:
            raise RuntimeError("segShrink replay outputs must be vectors")
        if factor.shape != fullP.shape or calibrated.shape != fullP.shape:
            raise RuntimeError("segShrink replay output does not match fullP")
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0.0):
            raise RuntimeError("segShrink replay factor must be positive and finite")
        if not np.all(np.isfinite(calibrated)) or np.any(calibrated < 0.0):
            raise RuntimeError("segShrink replay uncertainty must be finite and nonnegative")
        if not isinstance(replayedModel, Mapping):
            raise RuntimeError("segShrink replay model must be a mapping")
        if replayedModel.get("hierarchyScope") != "processedGenome" or int(
            replayedModel.get("processedContigCount", 0)
        ) != contigCount:
            raise RuntimeError("segShrink replay discarded processed-genome metadata")
        targetCalibration = replayedModel.get("target_calibration")
        if not isinstance(targetCalibration, Mapping):
            raise RuntimeError("segShrink replay target calibration must be a mapping")
        uncertaintyTrackScale = float(
            targetCalibration.get("uncertainty_track_scale", 1.0)
        )
        if not np.isfinite(uncertaintyTrackScale) or uncertaintyTrackScale <= 0.0:
            raise RuntimeError("segShrink replay target uncertainty scale is not positive")
        rawSD = np.sqrt(np.maximum(fullP, positiveFloor))
        baseSD = np.sqrt(np.maximum(factor * fullP, positiveFloor))
        expectedCalibrated = np.maximum(
            baseSD * uncertaintyTrackScale,
            rawSD,
        ).astype(np.float32)
        if not np.allclose(
            calibrated,
            expectedCalibrated,
            rtol=2.0e-6,
            atol=2.0e-7,
        ):
            raise RuntimeError(
                "segShrink replay uncertainty does not match factor and target scale"
            )
        if not hasattr(summary, "copy") or not hasattr(summary, "to_dict"):
            raise RuntimeError("segShrink replay summary must be tabular")
        out.append(
            {
                "chromosome": item["chromosome"],
                "intervals": item["intervals"],
                "fullP": fullP,
                "factor": factor,
                "calibrated": calibrated,
                "summary": summary,
                "model": replayedModel,
                "summaryRowIndex": item["summaryRowIndex"],
            }
        )
        offset += localCount
    return out
