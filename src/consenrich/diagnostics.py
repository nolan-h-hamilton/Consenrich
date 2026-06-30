# -*- coding: utf-8 -*-
"""Small model diagnostics used in logs and metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .constants import (
    UNCERTAINTY_CALIBRATION_AUTO_BLOCK_INTERVAL_MULTIPLIER,
    UNCERTAINTY_CALIBRATION_AUTO_BLOCK_MIN_BP,
    UNCERTAINTY_CALIBRATION_MIN_BLOCK_INTERVALS,
    UNCERTAINTY_CALIBRATION_MIN_FOLDS,
)


def metadataFloat(value: float) -> float | None:
    value_ = float(value)
    if not np.isfinite(value_):
        return None
    return value_


def resolveUncertaintyBlockSizeIntervals(
    blockSizeBP: int | str | None,
    intervalSizeBP: int,
    n: int,
    folds: int | None = None,
) -> int:
    r"""Resolve the interval block length used by block-holdout calibration."""

    intervalCount = int(n)
    intervalBP = max(int(intervalSizeBP), 1)
    minBlockIntervals = UNCERTAINTY_CALIBRATION_MIN_BLOCK_INTERVALS
    if blockSizeBP is None or str(blockSizeBP).lower() == "auto":
        targetIntervals = round(
            max(
                UNCERTAINTY_CALIBRATION_AUTO_BLOCK_MIN_BP,
                UNCERTAINTY_CALIBRATION_AUTO_BLOCK_INTERVAL_MULTIPLIER
                * intervalBP,
            )
            / intervalBP
        )
        foldCount = max(
            int(folds) if folds is not None else UNCERTAINTY_CALIBRATION_MIN_FOLDS,
            UNCERTAINTY_CALIBRATION_MIN_FOLDS,
        )
        if intervalCount > 0:
            maxAutoIntervals = max(
                (intervalCount + foldCount - 1) // foldCount,
                minBlockIntervals,
            )
            targetIntervals = min(targetIntervals, maxAutoIntervals)
    else:
        targetIntervals = round(int(blockSizeBP) / intervalBP)
    return int(
        np.clip(
            targetIntervals,
            minBlockIntervals,
            max(intervalCount, minBlockIntervals),
        )
    )


def summarizeStateRoughness(
    state: npt.ArrayLike,
    *,
    blockLenIntervals: int,
    intervalSizeBP: int | None = None,
) -> dict[str, Any]:
    r"""Summarize final-state mean AFDs wrt held-out blocks.

    Roughness is the mean absolute first difference within each block
    """

    stateArr = np.asarray(state, dtype=np.float64)
    if stateArr.ndim == 2:
        stateArr = stateArr[:, 0]
    stateArr = stateArr.reshape(-1)
    n = int(stateArr.size)
    blockLen = int(max(1, blockLenIntervals))
    blockCount = int(np.ceil(n / float(blockLen))) if n > 0 else 0

    blockMeanDiff = np.full(blockCount, np.nan, dtype=np.float64)
    blockDiffCount = np.zeros(blockCount, dtype=np.int64)
    blockDiffSum = np.zeros(blockCount, dtype=np.float64)
    blockSignal = np.full(blockCount, np.nan, dtype=np.float64)

    for blockIdx in range(blockCount):
        start = blockIdx * blockLen
        end = min(start + blockLen, n)
        values = stateArr[start:end]
        finiteValues = np.isfinite(values)
        if np.any(finiteValues):
            blockSignal[blockIdx] = float(np.nanmedian(np.abs(values[finiteValues])))
        if values.size < 2:
            continue
        diffMask = np.isfinite(values[:-1]) & np.isfinite(values[1:])
        if not np.any(diffMask):
            continue
        absDiff = np.abs(np.diff(values)[diffMask])
        blockDiffCount[blockIdx] = int(absDiff.size)
        blockDiffSum[blockIdx] = float(np.sum(absDiff))
        blockMeanDiff[blockIdx] = float(np.mean(absDiff))

    totalDiffCount = int(np.sum(blockDiffCount))
    totalDiffSum = float(np.sum(blockDiffSum))
    finiteBlockMean = blockMeanDiff[np.isfinite(blockMeanDiff)]
    summary: dict[str, Any] = {
        "method": "mean_abs_first_difference_by_holdout_block",
        "block_len_intervals": int(blockLen),
        "block_len_bp": (
            None if intervalSizeBP is None else int(blockLen * int(intervalSizeBP))
        ),
        "n_intervals": int(n),
        "n_blocks": int(blockCount),
        "n_differences": int(totalDiffCount),
        "overall_mean_abs_diff": (
            metadataFloat(totalDiffSum / float(totalDiffCount))
            if totalDiffCount > 0
            else None
        ),
        "block_mean_abs_diff_median": (
            metadataFloat(np.nanmedian(finiteBlockMean))
            if finiteBlockMean.size
            else None
        ),
        "block_mean_abs_diff_q90": (
            metadataFloat(np.nanquantile(finiteBlockMean, 0.9))
            if finiteBlockMean.size
            else None
        ),
        "signal_strata_basis": "block_median_abs_state",
        "signal_strata": [],
    }

    validStrataBlocks = np.isfinite(blockSignal) & np.isfinite(blockMeanDiff)
    if np.any(validStrataBlocks):
        quantiles = np.asarray([0.0, 0.5, 0.9, 1.0], dtype=np.float64)
        cuts = np.nanquantile(blockSignal[validStrataBlocks], quantiles)
        strataRows: list[dict[str, Any]] = []
        for idx in range(len(quantiles) - 1):
            lo = float(cuts[idx])
            hi = float(cuts[idx + 1])
            if idx == 0:
                mask = validStrataBlocks & (blockSignal <= hi)
            else:
                mask = validStrataBlocks & (blockSignal > lo) & (blockSignal <= hi)
            if not np.any(mask):
                continue
            diffCount = int(np.sum(blockDiffCount[mask]))
            diffSum = float(np.sum(blockDiffSum[mask]))
            means = blockMeanDiff[mask]
            strataRows.append(
                {
                    "stratum": (
                        f"signal_abs_q{int(quantiles[idx] * 100):02d}_"
                        f"{int(quantiles[idx + 1] * 100):02d}"
                    ),
                    "n_blocks": int(np.sum(mask)),
                    "n_differences": int(diffCount),
                    "signal_min": metadataFloat(lo),
                    "signal_max": metadataFloat(hi),
                    "mean_abs_diff": (
                        metadataFloat(diffSum / float(diffCount))
                        if diffCount > 0
                        else None
                    ),
                    "median_block_mean_abs_diff": (
                        metadataFloat(np.nanmedian(means)) if means.size else None
                    ),
                }
            )
        summary["signal_strata"] = strataRows

    return summary


def summarizePrecisionBoundaryHits(
    *,
    observationPrecision: npt.ArrayLike | None,
    observationPrecisionMin: float,
    observationPrecisionMax: float,
    processPrecision: npt.ArrayLike | None,
    processPrecisionMin: float,
    processPrecisionMax: float,
) -> dict[str, Any]:
    r"""Count final precision multipliers that are pinned to their boundaries."""

    def _summarize(
        values: npt.ArrayLike | None,
        lower: float,
        upper: float,
        *,
        skipFirst: bool = False,
    ) -> dict[str, Any]:
        if values is None:
            return {
                "enabled": False,
                "total": 0,
                "lower": 0,
                "upper": 0,
                "lower_fraction": None,
                "upper_fraction": None,
            }
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if skipFirst and arr.size > 0:
            arr = arr[1:]
        finite = arr[np.isfinite(arr)]
        total = int(finite.size)
        atolLower = 1.0e-6 * max(abs(float(lower)), 1.0)
        atolUpper = 1.0e-6 * max(abs(float(upper)), 1.0)
        lowerHits = int(
            np.sum(np.isclose(finite, float(lower), rtol=0.0, atol=atolLower))
        )
        upperHits = int(
            np.sum(np.isclose(finite, float(upper), rtol=0.0, atol=atolUpper))
        )
        return {
            "enabled": True,
            "total": total,
            "lower": lowerHits,
            "upper": upperHits,
            "lower_fraction": (
                metadataFloat(lowerHits / float(total)) if total > 0 else None
            ),
            "upper_fraction": (
                metadataFloat(upperHits / float(total)) if total > 0 else None
            ),
        }

    return {
        "observation": _summarize(
            observationPrecision,
            observationPrecisionMin,
            observationPrecisionMax,
        ),
        "process": _summarize(
            processPrecision,
            processPrecisionMin,
            processPrecisionMax,
            skipFirst=True,
        ),
        "bounds": {
            "observation": [
                float(observationPrecisionMin),
                float(observationPrecisionMax),
            ],
            "process": [float(processPrecisionMin), float(processPrecisionMax)],
        },
    }
