"""Shared normalization, enum parsing, and small numeric helpers.

This module is intentionally dependency-light so it can be imported by config,
core, IO, peak-calling, and uncertainty code without creating circular imports.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import constants


def validateChoice(
    value: Any, *, default: str, supported: Sequence[str], paramName: str
) -> str:
    raw = default if value is None else value
    if not isinstance(raw, str) or raw not in supported:
        choices = ", ".join(supported)
        raise ValueError(f"Unsupported {paramName} {raw!r}. Supported values: {choices}.")
    return raw


def validateDefaultConfigurationName(value: Any) -> str:
    if isinstance(value, str) and value in constants.SUPPORTED_COUNTING_PRESETS:
        raise ValueError(
            f"Unsupported default configuration {value!r}. "
            f"Set countingPreset: {value} for assay counting."
        )
    return validateChoice(
        value,
        default=constants.GENERIC_DEFAULT_CONFIGURATION,
        supported=constants.SUPPORTED_DEFAULT_CONFIGURATIONS,
        paramName="default configuration",
    )


def validateCountMode(value: str | None, defaultMode: str) -> str:
    mode = validateChoice(
        value, default=defaultMode,
        supported=constants.SUPPORTED_COUNT_MODES, paramName="countMode",
    )
    return "center" if mode == "midpoint" else mode


def nativeCountModeForPreset(countMode: str) -> str:
    return "center" if countMode == "ffp-center" else countMode


def validateBamInputMode(
    value: str | None, *, default: str = "auto", autoAsReads: bool = False
) -> str:
    mode = validateChoice(
        value, default=default,
        supported=constants.SUPPORTED_BAM_INPUT_MODES, paramName="bamInputMode",
    )
    return "reads" if autoAsReads and mode == "auto" else mode


def validateMuncVarianceModel(value: str | int | None) -> str:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if value == constants.MUNC_VARIANCE_MODEL_CODE_KALMAN:
            return constants.MUNC_VARIANCE_MODEL_KALMAN
    return validateChoice(
        value,
        default=constants.OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL,
        supported=constants.MUNC_SUPPORTED_VARIANCE_MODELS,
        paramName="MUNC variance model",
    )


def validateDiagnosticTracks(value: Any) -> tuple[str, ...]:
    allTracks = constants.OUTPUT_DIAGNOSTIC_TRACK_NAMES
    if value is None or value is False:
        return ()
    if value is True:
        return tuple(allTracks)
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        raise ValueError(
            "outputParams.diagnosticTracks must be a list, comma-separated string, "
            "boolean, or null."
        )
    tracks = []
    for item in items:
        if item == "all":
            choices = allTracks
        else:
            choices = (validateChoice(
                item, default="", supported=allTracks,
                paramName="outputParams.diagnosticTracks",
            ),)
        for choice in choices:
            if choice not in tracks:
                tracks.append(choice)
    return tuple(tracks)


def validateStudentTDF(value: Any, paramName: str = "studentTDF") -> float:
    message = f"{paramName} must be numeric with 1 <= value <= 30"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not np.isfinite(result) or result < 1.0 or result > 30.0:
        raise ValueError(message)
    return result


def validateStudentTQuadratureOrder(
    value: Any, paramName: str = "studentTQuadratureOrder"
) -> int:
    message = f"{paramName} must be an integer with 8 <= order <= 96"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not np.isfinite(result) or not result.is_integer() or result < 8 or result > 96:
        raise ValueError(message)
    return int(result)


def validatePositiveFloat(value: Any, paramName: str) -> float:
    message = f"{paramName} must be finite and positive"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(message)
    return result


def validateFiniteNonnegativeFloat(
    value: Any, *, paramName: str
) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"`{paramName}` must be finite and non-negative.")
    return result


def validateNonnegativeInt(value: Any, paramName: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{paramName} must be a non-negative integer")

    try:
        out = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{paramName} must be a non-negative integer") from exc
    if out < 0:
        raise ValueError(f"{paramName} must be a non-negative integer")
    return out


def validateOptionalPositiveInteger(value: Any, paramName: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{paramName} must be a positive integer or null")
    try:
        out = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{paramName} must be a positive integer or null") from exc
    if out <= 0:
        raise ValueError(f"{paramName} must be a positive integer or null")
    return int(out)


def validateOptionalOpenUnitInterval(value: Any, paramName: str) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out) or out <= 0.0 or out >= 1.0:
        raise ValueError(f"{paramName} must be finite and strictly between 0 and 1")
    return out


def validateOptionalPositiveFloat(value: Any, paramName: str) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{paramName} must be finite and positive")
    return out


def validateNonnegativeFloat(value: Any, paramName: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{paramName} must be finite and non-negative")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{paramName} must be finite and non-negative") from exc
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{paramName} must be finite and non-negative")
    return out


def validateOptionalNonnegativeFloat(value: Any, paramName: str) -> float | None:
    if value is None:
        return None
    return validateNonnegativeFloat(value, paramName)


def validateTransformFloat(
    value: Any,
    *,
    paramName: str,
    default: float | None = None,
    positive: bool = False,
) -> float | None:
    if value is None:
        return default
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{paramName} must be finite.")
    if positive and out <= 0.0:
        raise ValueError(f"{paramName} must be positive.")
    return out


def expandScaleFactors(
    scaleFactors: Sequence[float] | None,
    expectedCount: int,
    paramName: str,
) -> list[float] | None:
    if scaleFactors is None:
        return None
    if expectedCount < 1:
        raise ValueError(f"`{paramName}` was provided but no matching inputs exist.")
    expanded = [float(scaleFactor) for scaleFactor in scaleFactors]
    if len(expanded) == expectedCount:
        return expanded
    if len(expanded) == 1:
        return expanded * expectedCount
    raise ValueError(
        f"`{paramName}` must contain 1 value or {expectedCount} values; "
        f"got {len(expanded)}."
    )


def weightedQuantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Weighted empirical quantile with deterministic stable sorting."""

    valuesArray = np.asarray(values, dtype=np.float64).reshape(-1)
    weightsArray = np.asarray(weights, dtype=np.float64).reshape(-1)
    if valuesArray.shape != weightsArray.shape:
        raise ValueError("values and weights must have the same shape")
    valid = np.isfinite(valuesArray) & np.isfinite(weightsArray) & (weightsArray > 0.0)
    if not np.any(valid):
        raise ValueError("weighted quantile requires at least one finite positive-weight value")
    valuesArray = valuesArray[valid]
    weightsArray = weightsArray[valid]
    order = np.argsort(valuesArray, kind="mergesort")
    valuesArray = valuesArray[order]
    weightsArray = weightsArray[order]
    total = float(np.sum(weightsArray))
    if not np.isfinite(total):
        raise ValueError("weighted quantile requires positive total weight")
    cumulative = np.cumsum(weightsArray) / total

    quantileArray = np.asarray(q, dtype=np.float64)
    clipped = np.clip(quantileArray, 0.0, 1.0)
    idx = np.searchsorted(cumulative, clipped, side="left")
    idx = np.clip(idx, 0, valuesArray.size - 1)
    out = valuesArray[idx]
    if quantileArray.ndim == 0:
        return float(np.asarray(out).reshape(()))
    return np.asarray(out, dtype=np.float64)


def weightedQuantileInterpolated(
    values: np.ndarray,
    weights: np.ndarray,
    q: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Weighted quantile using linear interpolation on the weighted CDF.

    This preserves the historical MUNC/P-spline knot behavior, while
    ``weighted_quantile`` preserves the empirical order-statistic behavior used
    by delete-block uncertainty calibration.
    """

    valuesArray = np.asarray(values, dtype=np.float64).reshape(-1)
    weightsArray = np.asarray(weights, dtype=np.float64).reshape(-1)
    quantileArray = np.asarray(q, dtype=np.float64)
    if valuesArray.shape != weightsArray.shape:
        raise ValueError("values and weights must have the same shape")
    valid = np.isfinite(valuesArray) & np.isfinite(weightsArray) & (weightsArray > 0.0)
    if not np.any(valid):
        out = np.full(quantileArray.shape, np.nan, dtype=np.float64)
        if quantileArray.ndim == 0:
            return float(np.asarray(out).reshape(()))
        return out
    valuesArray = valuesArray[valid]
    weightsArray = weightsArray[valid]
    order = np.argsort(valuesArray, kind="mergesort")
    valuesArray = valuesArray[order]
    weightsArray = weightsArray[order]
    total = float(np.sum(weightsArray))
    cdf = np.cumsum(weightsArray)
    out = np.interp(np.clip(quantileArray, 0.0, 1.0) * total, cdf, valuesArray)
    if quantileArray.ndim == 0:
        return float(np.asarray(out).reshape(()))
    return np.asarray(out, dtype=np.float64)


__all__ = [
    "validateChoice",
    "validateDefaultConfigurationName",
    "validateCountMode",
    "nativeCountModeForPreset",
    "validateBamInputMode",
    "validateMuncVarianceModel",
    "validateDiagnosticTracks",
    "validateStudentTDF",
    "validateStudentTQuadratureOrder",
    "validatePositiveFloat",
    "validateFiniteNonnegativeFloat",
    "validateNonnegativeInt",
    "validateOptionalPositiveInteger",
    "validateOptionalOpenUnitInterval",
    "validateOptionalPositiveFloat",
    "validateNonnegativeFloat",
    "validateOptionalNonnegativeFloat",
    "validateTransformFloat",
    "expandScaleFactors",
    "weightedQuantile",
    "weightedQuantileInterpolated",
]
