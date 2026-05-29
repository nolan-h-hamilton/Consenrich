"""Shared normalization, enum parsing, and small numeric helpers.

This module is intentionally dependency-light so it can be imported by config,
core, IO, peak-calling, and uncertainty code without creating circular imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import constants


def enum_token_key(value: Any) -> str:
    """Return a stable, punctuation-insensitive enum lookup key."""

    text = str(value).strip().replace("-", "_").replace(" ", "_").lower()
    return "_".join(part for part in text.split("_") if part)


def compact_token_key(value: Any) -> str:
    """Return an aggressive enum key for historical aliases."""

    return (
        str(value)
        .strip()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def normalize_config_enum(
    value: Any,
    *,
    default: str,
    supported: Sequence[str],
    config_name: str,
) -> str:
    """Normalize a config enum against an explicit supported set."""

    raw = default if value is None else value
    canonical_by_key = {enum_token_key(item): item for item in supported}
    key = enum_token_key(raw)
    if key not in canonical_by_key:
        supported_text = ", ".join(supported)
        raise ValueError(
            f"Unsupported {config_name} {raw!r}. Supported values: {supported_text}."
        )
    return str(canonical_by_key[key])


def normalize_count_transform_method(
    value: Any,
    *,
    config_name: str = "countingParams.transformMethod",
) -> str:
    """Normalize count-transform names used by config and variance floors."""

    raw = constants.COUNTING_DEFAULT_TRANSFORM_METHOD if value is None else value
    canonical_by_key = {
        "log": "log",
        "ln": "log",
        "naturallog": "log",
        "sqrt": "sqrt",
        "squareroot": "sqrt",
        "anscombe": "anscombe",
        "anscombetransform": "anscombe",
        "asinh": "asinh",
        "arcsinh": "asinh",
        "asinhx": "asinh",
        "arcsinhx": "asinh",
        "asinhsqrt": "asinhSqrt",
        "arcsinhsqrt": "asinhSqrt",
        "sqrtasinh": "asinhSqrt",
        "generalizedlog": "generalizedLog",
        "generalisedlog": "generalizedLog",
        "glog": "generalizedLog",
        "softlog": "generalizedLog",
        "identity": "identity",
        "linear": "identity",
        "raw": "identity",
        "none": "identity",
    }
    key = compact_token_key(raw)
    if key not in canonical_by_key:
        supported = ", ".join(constants.COUNTING_SUPPORTED_TRANSFORM_METHODS)
        raise ValueError(
            f"Unsupported {config_name} {raw!r}. Supported methods: {supported}."
        )
    return canonical_by_key[key]


def normalize_count_mode(count_mode: str | None, default_mode: str) -> str:
    """Normalize BAM/fragments count modes to native counting labels."""

    normalized = str(count_mode or default_mode).strip().lower()
    if normalized not in constants.SUPPORTED_COUNT_MODES:
        raise ValueError(f"Unsupported countMode `{count_mode}`")
    if normalized == "midpoint":
        return "center"
    return normalized


def native_count_mode_for_preset(count_mode: str) -> str:
    """Return the native C-counting label for a higher-level count preset."""

    return "center" if str(count_mode) == "ffp-center" else str(count_mode)


def normalize_bam_input_mode(
    bam_input_mode: str | None,
    *,
    default: str = "auto",
    auto_as_reads: bool = False,
) -> str:
    """Normalize BAM interpretation mode.

    ``auto_as_reads`` is used by legacy detrorm normalization, where ``auto``
    historically resolved to per-read counting rather than inspecting the file.
    """

    normalized = str(bam_input_mode or default).strip().lower()
    if normalized == "auto" and auto_as_reads:
        return "reads"
    if normalized not in constants.SUPPORTED_BAM_INPUT_MODES:
        raise ValueError(f"Unsupported bamInputMode `{bam_input_mode}`")
    return normalized


def normalize_fragment_position_mode(fragment_position_mode: str | None) -> str:
    """Normalize 10x fragments endpoint-position mode."""

    normalized = (
        str(fragment_position_mode or constants.SC_DEFAULT_FRAGMENT_POSITION_MODE)
        .strip()
        .replace("_", "")
        .replace("-", "")
        .lower()
    )
    if normalized not in constants.SUPPORTED_FRAGMENT_POSITION_MODES:
        raise ValueError(
            f"Unsupported fragmentPositionMode `{fragment_position_mode}`"
        )
    return normalized


def normalize_matching_uncertainty_score_mode(
    value: Any,
    *,
    config_name: str = "matchingParams.uncertaintyScoreMode",
    allow_consenrich_state_alias: bool = True,
) -> str:
    """Normalize peak-calling uncertainty score modes."""

    text = (
        str(constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE)
        if value is None
        else str(value)
    )
    mode = text.strip().lower().replace("-", "_")
    if allow_consenrich_state_alias and mode == "consenrich_state":
        mode = "state"
    if mode not in constants.MATCHING_SUPPORTED_UNCERTAINTY_SCORE_MODES:
        supported = ", ".join(constants.MATCHING_SUPPORTED_UNCERTAINTY_SCORE_MODES)
        raise ValueError(
            f"Unsupported {config_name} {value!r}. Supported modes: {supported}."
        )
    return mode


def validate_uncertainty_score_z(
    value: Any,
    *,
    config_name: str = "matchingParams.uncertaintyScoreZ",
) -> float:
    """Validate a non-negative normal-score multiplier."""

    z = float(value)
    if not np.isfinite(z) or z < 0.0:
        raise ValueError(f"`{config_name}` must be finite and non-negative.")
    return z


def normalize_process_noise_calibration(value: Any) -> str:
    """Normalize process-noise calibration mode and historical aliases."""

    raw = constants.PROCESS_DEFAULT_NOISE_CALIBRATION if value is None else value
    key = str(raw).strip().replace("-", "_").lower()
    aliases = {
        "tunc": constants.PROCESS_NOISE_CALIBRATION_TUNC,
        "seed": constants.PROCESS_NOISE_CALIBRATION_SEED,
        "fixed": constants.PROCESS_NOISE_CALIBRATION_FIXED,
        "none": constants.PROCESS_NOISE_CALIBRATION_SEED,
        "warm_start": constants.PROCESS_NOISE_CALIBRATION_FIXED,
    }
    normalized = aliases.get(key, key)
    if normalized not in constants.PROCESS_NOISE_CALIBRATION_MODES:
        supported = ", ".join(constants.PROCESS_NOISE_CALIBRATION_MODES)
        raise ValueError(
            f"Unsupported processNoiseCalibration {raw!r}. Supported modes: {supported}."
        )
    return str(normalized)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    q: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Weighted empirical quantile with deterministic stable sorting."""

    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values_arr.shape != weights_arr.shape:
        raise ValueError("values and weights must have the same shape")
    valid = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0.0)
    if not np.any(valid):
        raise ValueError("weighted quantile requires at least one finite positive-weight value")
    values_arr = values_arr[valid]
    weights_arr = weights_arr[valid]
    order = np.argsort(values_arr, kind="mergesort")
    values_arr = values_arr[order]
    weights_arr = weights_arr[order]
    total = float(np.sum(weights_arr))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weighted quantile requires positive total weight")
    cumulative = np.cumsum(weights_arr) / total

    q_arr = np.asarray(q, dtype=np.float64)
    clipped = np.clip(q_arr, 0.0, 1.0)
    idx = np.searchsorted(cumulative, clipped, side="left")
    idx = np.clip(idx, 0, values_arr.size - 1)
    out = values_arr[idx]
    if q_arr.ndim == 0:
        return float(np.asarray(out).reshape(()))
    return np.asarray(out, dtype=np.float64)


def weighted_quantile_interpolated(
    values: np.ndarray,
    weights: np.ndarray,
    q: float | Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Weighted quantile using linear interpolation on the weighted CDF.

    This preserves the historical MUNC/P-spline knot behavior, while
    ``weighted_quantile`` preserves the empirical order-statistic behavior used
    by delete-block uncertainty calibration.
    """

    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    q_arr = np.asarray(q, dtype=np.float64)
    if values_arr.shape != weights_arr.shape:
        raise ValueError("values and weights must have the same shape")
    valid = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0.0)
    if not np.any(valid):
        out = np.full(q_arr.shape, np.nan, dtype=np.float64)
        if q_arr.ndim == 0:
            return float(np.asarray(out).reshape(()))
        return out
    values_arr = values_arr[valid]
    weights_arr = weights_arr[valid]
    order = np.argsort(values_arr, kind="mergesort")
    values_arr = values_arr[order]
    weights_arr = np.maximum(weights_arr[order], 0.0)
    total = float(np.sum(weights_arr))
    if total <= 0.0 or values_arr.size == 0:
        out = np.full(q_arr.shape, np.nan, dtype=np.float64)
        if q_arr.ndim == 0:
            return float(np.asarray(out).reshape(()))
        return out
    cdf = np.cumsum(weights_arr)
    out = np.interp(np.clip(q_arr, 0.0, 1.0) * total, cdf, values_arr)
    if q_arr.ndim == 0:
        return float(np.asarray(out).reshape(()))
    return np.asarray(out, dtype=np.float64)


__all__ = [
    "compact_token_key",
    "enum_token_key",
    "native_count_mode_for_preset",
    "normalize_bam_input_mode",
    "normalize_config_enum",
    "normalize_count_mode",
    "normalize_count_transform_method",
    "normalize_fragment_position_mode",
    "normalize_matching_uncertainty_score_mode",
    "normalize_process_noise_calibration",
    "validate_uncertainty_score_z",
    "weighted_quantile",
    "weighted_quantile_interpolated",
]
