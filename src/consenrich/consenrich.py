#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import logging
import math
from multiprocessing.pool import ThreadPool
import os
import tempfile
import time
from pathlib import Path
from collections.abc import Mapping
from typing import List, Optional, Tuple, Dict, Any, Union, Sequence, NamedTuple
import sys
import numpy as np
import pandas as pd
from scipy import stats

import consenrich.core as core
import consenrich.diagnostics as diagnostics
import consenrich.detrorm as detrorm
import consenrich.peaks as peaks
import consenrich.shrinkState as shrinkState
from . import cconsenrich
from . import constants
from . import _logging as logging_utils
from . import misc_util
from ._version import __version__
from .genome_covariates import (
    ConsenrichGenomeCovariateCache,
    resolve_genome_covariate_feature_config,
)
from . import io as io_helpers
from .config import getLoggingArgs, loadConfig, readConfig
from .io import (
    _buildPathInputSources,
    _checkSF,
    _getSourceCountMode,
    _inferMatchingUncertaintyBedGraph,
    _listOrEmpty,
    _prepareFragmentsNormalizationMetadata,
    _resolveExtendFrom5pBPPairs,
    _sortBedGraphInPlace,
    _validateBedGraphSorted,
    checkControlsPresent,
    checkMatchingEnabled,
    convertBedGraphToBigWig,
    getEffectiveGenomeSizes,
    getReadLengths,
)

logger = logging.getLogger("consenrich.consenrich")

_CLI_HANDLER_ATTR = "_consenrich_cli_handler"
_CONSOLE_EVENT_ATTR = "consenrich_console"
_CONSOLE_VERBOSE_EVENT_ATTR = "consenrich_console_verbose"
_CONSOLE_PHASE_ATTR = "consenrich_console_phase"
_CONSOLE_SUBPHASE_ATTR = "consenrich_console_subphase"
_CONSOLE_BLUE_ATTR = "consenrich_console_blue"
_CONSOLE_RICH_NAVY_TEXT = "\033[1;38;2;0;48;96m"
_CONSOLE_BURNT_ORANGE_TEXT = "\033[38;2;191;87;0m"
_CONSOLE_DARK_MAGENTA_TEXT = "\033[1;38;2;128;24;96m"
_CONSOLE_PHASE_TEXT = _CONSOLE_RICH_NAVY_TEXT
_CONSOLE_SUBPHASE_TEXT = _CONSOLE_BURNT_ORANGE_TEXT
_CONSOLE_MILESTONE_TEXT = _CONSOLE_DARK_MAGENTA_TEXT
_CONSOLE_BLUE_TEXT = _CONSOLE_RICH_NAVY_TEXT
_CONSOLE_WARNING_TEXT = _CONSOLE_BURNT_ORANGE_TEXT
_CONSOLE_ERROR_TEXT = _CONSOLE_DARK_MAGENTA_TEXT
_CONSOLE_STYLE_RESET = "\033[0m"
_JSON_LOG_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_CONSOLE_VERBOSITY_ORDER = {
    "quiet": 0,
    "normal": 1,
    "verbose": 2,
    "debug": 3,
}

OPTIMIZATION_PATH_COLUMNS = [
    "chromosome",
    "phase",
    "path_level",
    "outer_pass",
    "inner_iter",
    "record_order",
    "objective_name",
    "objective_value",
    "objective_per_cell",
    "change",
    "threshold",
    "objective_stable",
    "background_shift",
    "background_shift_threshold",
    "background_shift_stable",
    "outer_stable_iters",
    "outer_patience_target",
    "outer_inner_ecm_converged",
    "reset_iteration",
    "converged",
    "final_solution",
]

PRECISION_DIAGNOSTIC_COLUMNS = [
    "Chromosome",
    "Start",
    "End",
    "Interval",
    "lambda",
    "kappa",
    "Q00",
    "Q11",
    "baseQ00",
    "baseQ11",
    "effectiveQ00",
    "effectiveQ11",
    "process_q_policy",
    "apn_enabled",
    "process_precision_reweighting_requested",
    "process_precision_reweighting_effective",
    "process_precision_reweighting_disabled_by_apn",
    "median_diag_R",
    "median_effective_diag_R",
]

_OUTPUT_TRACK_FALLBACK_NAMES: Dict[str, Tuple[str, ...]] = {
    "preKappaQLevel": ("baseQLevel",),
    "preKappaQTrend": ("baseQTrend",),
}

_REPLICATE_EXCHANGEABILITY_PERMUTATIONS = 199
_REPLICATE_EXCHANGEABILITY_MIN_BLOCKS = 6
_REPLICATE_EXCHANGEABILITY_TOP_EFFECTS = 10
_REPLICATE_EXCHANGEABILITY_PAIR_CHUNK_ELEMENTS = 2_000_000
_REPLICATE_EXCHANGEABILITY_MAX_BLOCKS = 5_000

GAIN_SUMMARY_COLUMNS = [
    "replicate_index",
    "sample_name",
    "sample_file",
    "treatment_path",
    "control_path",
    "chromosome_count",
    "finite_interval_count",
    "gain_avg",
    "gain_std",
]

CORRELATION_LENGTH_COLUMNS = [
    "status",
    "method",
    "random_seed",
    "inference_scope",
    "confidence_interval_method",
    "survival_band_region_lower",
    "survival_band_region_upper",
    "survival_band_jump_closure_used",
    "survival_band_jump_closure_count",
    "confidence_level",
    "interval_size_bp",
    "correlation_length_intervals",
    "correlation_length_bp",
    "full_sample_median_radius_bp",
    "lower_bp",
    "upper_bp",
    "working_span_bp",
    "full_sample_working_span_bp",
    "working_quantile",
    "context_bp",
    "chromosomes_used",
    "candidate_window_count",
    "evaluated_candidate_window_count",
    "selected_window_count",
    "crossed_window_count",
    "right_censored_window_count",
    "support_censored_window_count",
    "cap_censored_window_count",
    "censor_fraction",
    "censor_time_bp_minimum",
    "censor_time_bp_median",
    "censor_time_bp_maximum",
    "input_row_count",
    "unique_row_count",
    "duplicate_row_count",
    "row_deduplication",
    "acf_threshold",
    "acf_smoothing_bp",
    "acf_smoothing_bins",
    "crossing_persistence_bp",
    "crossing_persistence_bins",
    "min_finite_pairs",
    "min_finite_pair_coverage",
    "finite_pair_minimum_used",
    "finite_pair_coverage_minimum_used",
    "valid_rows_at_crossing_minimum",
    "dominant_acf_period_bp_median",
    "unresolved_period_fraction",
    "oscillation_strength",
    "post_crossing_revival",
    "selected_adjacency_count",
    "selected_longest_run",
    "bootstrap_method",
    "bootstrap_block_length_windows",
    "bootstrap_draws_requested",
    "bootstrap_resolved_median_draws",
    "bootstrap_resolved_working_draws",
    "bootstrap_resolved_joint_draws",
]

CORRELATION_LENGTH_WINDOW_COLUMNS = [
    "chromosome",
    "start_bp",
    "end_bp",
    "score",
    "positive_signal_rank",
    "raw_crossing_lag_bp",
    "censor_lag_bp",
    "gaussian_equivalent_radius_bp",
    "right_censored",
    "censor_reason",
    "support_cap_lag_bp",
    "finite_pair_minimum_used",
    "finite_pair_coverage_minimum_used",
    "valid_row_count",
    "valid_rows_at_crossing",
    "dominant_acf_period_bp",
    "oscillation_strength",
    "post_crossing_revival",
]

MUNC_LAMBDA_LOG_COLUMNS = [
    "record_type",
    "event",
    "chromosome",
    "start",
    "end",
    "interval",
    "replicate_index",
    "lambda",
    "lambda_lower_bound_hit",
    "lambda_upper_bound_hit",
    "median_diag_R",
    "median_effective_diag_R",
    "muncTrace",
    "sumGain0",
    "sumGain1",
    "key",
    "value",
]

PROCESS_KAPPA_LOG_COLUMNS = [
    "record_type",
    "event",
    "chromosome",
    "start",
    "end",
    "interval",
    "kappa",
    "kappa_lower_bound_hit",
    "kappa_upper_bound_hit",
    "process_q_policy",
    "apn_enabled",
    "process_precision_reweighting_requested",
    "process_precision_reweighting_effective",
    "process_precision_reweighting_disabled_by_apn",
    "baseQ00",
    "baseQ11",
    "preKappaQLevel",
    "preKappaQTrend",
    "effectiveQLevel",
    "effectiveQTrend",
    "processQScale",
    "key",
    "value",
]

DELETE_BLOCK_CALIBRATION_LOG_COLUMNS = [
    "record_type",
    "event",
    "chromosome",
    "fold",
    "interval_index",
    "block_index",
    "blockIDX",
    "chrom_start",
    "uncertainty_decile",
    "high_signal",
    "stratum",
    "target",
    "alpha",
    "delta",
    "q",
    "q_source",
    "k",
    "tail_probability",
    "finite_bound",
    "certified",
    "reason",
    "n",
    "coverage_before",
    "coverage_after",
    "mean_width_before",
    "mean_width_after",
    "median_width_before",
    "median_width_after",
    "q90_width_before",
    "q90_width_after",
    "residual",
    "deleted_state_delta",
    "state_full",
    "state_masked",
    "P00_full",
    "P00_masked",
    "covariance_delta",
    "total_information",
    "kept_information",
    "heldout_information",
    "heldout_information_fraction",
    "delta_variance",
    "delta_variance_source",
    "row_weight",
    "sd_before",
    "sd_after",
    "a_state",
    "factor_segment",
    "segment_raw_factor",
    "segment_bootstrap_variance",
    "segment_shrinkage_weight",
    "contig_shrinkage_weight",
    "key",
    "value",
]

CONVERGENCE_LOG_COLUMNS = ["record_type", *OPTIMIZATION_PATH_COLUMNS]

RUN_SUMMARY_COLUMNS = [
    "record_type",
    "chromosome",
    "intervals",
    "samples",
    "elapsed_seconds",
    "output_track_count",
    "final_nll",
    "final_forward_nis",
    "process_q_policy",
    "process_q_trace_min",
    "process_q_trace_median",
    "process_q_trace_max",
    "observation_r_trace_min",
    "observation_r_trace_median",
    "observation_r_trace_max",
    "process_noise_status",
    "process_noise_reason",
    "countNoiseFloorEnabled",
    "countNoiseFloorModel",
    "countNoiseFloorCountModes",
    "countNoiseFloorSourceKinds",
    "countNoiseFloorBedgraphSkippedTracks",
    "countNoiseFloorFinite",
    "countNoiseFloorPositive",
    "countNoiseFloorQ05",
    "countNoiseFloorMedian",
    "countNoiseFloorQ95",
    "lambda_lower_bound_hits",
    "lambda_upper_bound_hits",
    "kappa_lower_bound_hits",
    "kappa_upper_bound_hits",
    "state_roughness_mean_abs_diff",
    "state_roughness_block_median",
    "state_roughness_block_q90",
    "state_shrinkage_model",
    "state_shrinkage_slab_family",
    "state_shrinkage_scope",
    "state_shrinkage_chunk_count",
    "state_shrinkage_interval_count",
    "state_shrinkage_finite_count",
    "state_shrinkage_effective_block_count",
    "state_shrinkage_block_size_intervals",
    "state_shrinkage_prior_spike_prop",
    "state_shrinkage_spike_odds_multiplier",
    "state_shrinkage_effective_prior_spike_prop",
    "state_shrinkage_prior_scale",
    "state_shrinkage_prior_variance",
    "state_shrinkage_prior_variance_defined",
    "state_shrinkage_slab_count",
    "state_shrinkage_slab_weight",
    "state_shrinkage_slab_variance",
    "state_shrinkage_slab_multiplier",
    "state_shrinkage_component_weights",
    "state_shrinkage_student_t_df",
    "state_shrinkage_student_t_scale",
    "state_shrinkage_student_t_quadrature_order",
    "state_shrinkage_student_t_quadrature_alpha",
    "state_shrinkage_estimated_prior_spike_prop",
    "state_shrinkage_estimated_prior_scale",
    "state_shrinkage_estimated_slab_weights",
    "state_shrinkage_estimated_slab_scales",
    "state_shrinkage_iterations",
    "state_shrinkage_converged",
    "state_shrinkage_log_likelihood",
    "state_shrinkage_state_abs_median_before",
    "state_shrinkage_state_abs_median_after",
    "state_shrinkage_spike_prop_median",
    "state_shrinkage_posterior_sd_median",
    "delete_block_global_factor",
    "delete_block_factor_model",
    "delete_block_variance_multiplier_global",
    "delete_block_variance_multiplier_min",
    "delete_block_variance_multiplier_q05",
    "delete_block_variance_multiplier_median",
    "delete_block_variance_multiplier_mad",
    "delete_block_variance_multiplier_q95",
    "delete_block_variance_multiplier_max",
    "delete_block_sd_multiplier_median",
    "delete_block_track_sd_scale",
    "delete_block_rows_valid",
    "delete_block_rows_fit",
    "delete_block_deletion_probability",
    "delete_block_deleted_blocks",
    "delete_block_deleted_replicate_block_total",
    "delete_block_deleted_observation_interval_total",
    "delete_block_replicate_dependence_rho",
    "delete_block_replicate_dependence_rho_source",
    "delete_block_information_deff_median",
    "delete_block_scale",
    "delete_block_scale_reason",
    "precision_log",
    "convergence_log",
    "delete_block_calibration_log",
]


class DiagnosticLogPaths(NamedTuple):
    precision: Path
    convergence: Path
    delete_block_calibration: Path


def _countTransformVarianceFloorKwargs(
    countingArgs: core.countingParams,
) -> dict[str, Any]:
    return {
        "transformMethod": countingArgs.transformMethod,
        "logOffset": countingArgs.logOffset,
        "logMult": countingArgs.logMult,
        "transformInputOffset": countingArgs.transformInputOffset,
        "transformInputScale": countingArgs.transformInputScale,
        "transformOutputScale": countingArgs.transformOutputScale,
        "transformShape": countingArgs.transformShape,
    }


_MUNC_NUMERIC_VARIANCE_FLOOR = 1.0e-12


def _countNoisePseudoVarianceMassForScaledCounts(
    scaledCounts: np.ndarray,
    scaleFactor: float,
    rawNoiseMass: np.ndarray | None,
    countMode: str | None,
) -> float:
    if (
        rawNoiseMass is None
        or countMode != constants.COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
    ):
        return 0.5
    counts = np.asarray(scaledCounts, dtype=np.float64)
    rawNoise = np.asarray(rawNoiseMass, dtype=np.float64)
    if rawNoise.shape != counts.shape:
        raise ValueError("rawNoiseMass must match scaledCounts shape")
    rawCounts = counts / float(scaleFactor)
    finite = np.isfinite(rawNoise) & np.isfinite(rawCounts)
    if np.any(finite & ((rawNoise < 0.0) | (rawCounts < 0.0))):
        raise ValueError("count and noise masses must be nonnegative where finite")
    positive = finite & (rawCounts > 0.0)
    rawNoiseSum = float(np.sum(rawNoise[positive], dtype=np.float64))
    rawCountSum = float(np.sum(rawCounts[positive], dtype=np.float64))
    if (
        not np.isfinite(rawNoiseSum)
        or rawNoiseSum <= 0.0
        or not np.isfinite(rawCountSum)
        or rawCountSum <= 0.0
    ):
        raise ValueError("count noise pseudo variance masses must be positive finite")
    pseudoVarianceMass = 0.5 * rawNoiseSum / rawCountSum
    if not np.isfinite(pseudoVarianceMass) or pseudoVarianceMass <= 0.0:
        raise ValueError("countNoisePseudoVarianceMass must be positive and finite")
    return float(pseudoVarianceMass)


def _countModelVarianceFloorForScaledCounts(
    scaledCounts: np.ndarray,
    scaleFactor: float,
    countingArgs: core.countingParams,
    *,
    rawNoiseMass: np.ndarray | None = None,
    countMode: str | None = None,
    countModelSource: bool = True,
) -> np.ndarray:
    counts = np.asarray(scaledCounts, dtype=np.float64)
    floor = np.full(counts.shape, np.nan, dtype=np.float64)
    if not countModelSource:
        return floor
    try:
        scaleFactor_ = float(scaleFactor)
    except (TypeError, ValueError):
        return floor
    if not np.isfinite(scaleFactor_) or scaleFactor_ <= 0.0:
        return floor
    countNoisePseudoVarianceMass = _countNoisePseudoVarianceMassForScaledCounts(
        counts,
        scaleFactor_,
        rawNoiseMass,
        countMode,
    )
    return np.asarray(
        core.transformCountVarianceFloor(
            counts,
            [scaleFactor_],
            rawNoiseMass=rawNoiseMass,
            countNoisePseudoVarianceMass=countNoisePseudoVarianceMass,
            **_countTransformVarianceFloorKwargs(countingArgs),
        ),
        dtype=np.float64,
    )


def _combineCountModelVarianceFloors(*floors: np.ndarray) -> np.ndarray:
    if not floors:
        return np.empty(0, dtype=np.float64)
    arrays = [np.asarray(floor, dtype=np.float64) for floor in floors]
    out = np.full(arrays[0].shape, np.nan, dtype=np.float64)
    anyFinite = np.zeros(arrays[0].shape, dtype=bool)
    for arr in arrays:
        finite = np.isfinite(arr)
        out[finite & ~anyFinite] = 0.0
        out[finite] += arr[finite]
        anyFinite |= finite
    return out


def _sourceUsesCountModelFloor(source: core.inputSource) -> bool:
    return (
        str(getattr(source, "sourceKind", "")).upper() != constants.BEDGRAPH_SOURCE_KIND
    )


def _countModelFloorMatrixForScaledCounts(
    scaledCountMatrix: np.ndarray,
    scaleFactors: Sequence[float] | np.ndarray | None,
    sources: Sequence[core.inputSource],
    countingArgs: core.countingParams,
    *,
    rawNoiseMassMatrix: np.ndarray | None = None,
    countModes: Sequence[str] | None = None,
) -> np.ndarray:
    counts = np.asarray(scaledCountMatrix, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("scaledCountMatrix must be two-dimensional")
    if scaleFactors is None:
        scaleArr = np.ones(int(counts.shape[0]), dtype=np.float64)
    else:
        scaleArr = np.asarray(scaleFactors, dtype=np.float64).reshape(-1)
    if scaleArr.size != int(counts.shape[0]):
        raise ValueError("scaleFactors must match scaledCountMatrix rows")
    if countModes is not None and len(countModes) != int(counts.shape[0]):
        raise ValueError("countModes must match scaledCountMatrix rows")
    rawNoise = None
    if rawNoiseMassMatrix is not None:
        rawNoise = np.asarray(rawNoiseMassMatrix, dtype=np.float64)
        if rawNoise.shape != counts.shape:
            raise ValueError("rawNoiseMassMatrix must match scaledCountMatrix shape")
    floor = np.full(counts.shape, np.nan, dtype=np.float64)
    for j in range(int(counts.shape[0])):
        source = sources[j] if j < len(sources) else None
        rowRawNoise = None if rawNoise is None else rawNoise[j, :]
        rowCountMode = None if countModes is None else countModes[j]
        floor[j, :] = _countModelVarianceFloorForScaledCounts(
            counts[j, :],
            float(scaleArr[j]),
            countingArgs,
            rawNoiseMass=rowRawNoise,
            countMode=rowCountMode,
            countModelSource=(source is None or _sourceUsesCountModelFloor(source)),
        )
    return np.ascontiguousarray(floor.astype(np.float32), dtype=np.float32)


def _countModelVarianceFloorScalar(
    floorMatrix: np.ndarray | None,
    *,
    quantile: float = 0.05,
    fallback: float = _MUNC_NUMERIC_VARIANCE_FLOOR,
) -> float:
    if floorMatrix is None:
        return float(fallback)
    arr = np.asarray(floorMatrix, dtype=np.float64)
    values = arr[np.isfinite(arr) & (arr > 0.0)]
    if values.size == 0:
        return float(fallback)
    q = float(np.clip(float(quantile), 0.0, 1.0))
    value = float(np.quantile(values, q))
    if not np.isfinite(value) or value <= 0.0:
        return float(fallback)
    return float(max(value, float(fallback)))


def _summarizeCountModelVarianceFloor(
    floorMatrix: np.ndarray | None,
) -> dict[str, float | int]:
    if floorMatrix is None:
        return {"finite": 0, "positive": 0}
    arr = np.asarray(floorMatrix, dtype=np.float64)
    finite = np.isfinite(arr)
    positive = finite & (arr > 0.0)
    out: dict[str, float | int] = {
        "finite": int(np.count_nonzero(finite)),
        "positive": int(np.count_nonzero(positive)),
    }
    if np.any(positive):
        vals = arr[positive]
        out.update(
            {
                "min": float(np.min(vals)),
                "q05": float(np.quantile(vals, 0.05)),
                "median": float(np.median(vals)),
                "p95": float(np.quantile(vals, 0.95)),
                "q95": float(np.quantile(vals, 0.95)),
                "max": float(np.max(vals)),
            }
        )
    return out


def _fmtDiagnosticFloat(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value_ = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value_):
        return "NA"
    return f"{value_:.6g}"


def _diagnosticJsonText(value: Any) -> str:
    jsonValue = _jsonDiagnosticValue(value)
    if jsonValue is None:
        return "NA"
    return json.dumps(jsonValue, sort_keys=True, separators=(",", ":"))


def _finiteOptimizationValue(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_ = float(value)
    except (TypeError, ValueError):
        return None
    return value_ if np.isfinite(value_) else None


def _intOptimizationValue(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flattenOptimizationPathDiagnostics(
    chromosome: str,
    runDiagnostics: Mapping[str, Any],
    *,
    startOrder: int = 0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    phases = (
        ("post_process_noise_fit", "post_process_noise_fit"),
    )
    for fitKey, phaseLabel in phases:
        fitDiagnostics = runDiagnostics.get(fitKey)
        if not isinstance(fitDiagnostics, Mapping):
            continue
        ecmDiagnostics = fitDiagnostics.get("fixed_background_ecm", [])
        if not isinstance(ecmDiagnostics, list):
            continue
        lastOuterIndex = len(ecmDiagnostics) - 1
        for ecmIndex, ecmPass in enumerate(ecmDiagnostics):
            if not isinstance(ecmPass, Mapping):
                continue
            outerPass = int(ecmPass.get("outer_pass") or (ecmIndex + 1))
            outerObjective = _finiteOptimizationValue(ecmPass.get("outer_objective"))
            if outerObjective is not None:
                outerObjectiveStable = bool(
                    ecmPass.get("outer_objective_stable", False)
                )
                backgroundShiftStable = bool(
                    ecmPass.get("background_shift_stable", False)
                )
                innerECMConverged = bool(
                    ecmPass.get("outer_inner_ecm_converged", False)
                )
                rows.append(
                    {
                        "chromosome": str(chromosome),
                        "phase": phaseLabel,
                        "path_level": "outer",
                        "outer_pass": outerPass,
                        "inner_iter": None,
                        "record_order": int(startOrder + len(rows)),
                        "objective_name": "penalized_objective",
                        "objective_value": outerObjective,
                        "objective_per_cell": _finiteOptimizationValue(
                            ecmPass.get("outer_objective_per_cell")
                        ),
                        "change": _finiteOptimizationValue(
                            ecmPass.get("outer_objective_change_per_cell")
                        ),
                        "threshold": _finiteOptimizationValue(
                            ecmPass.get("outer_objective_threshold_per_cell")
                        ),
                        "objective_stable": outerObjectiveStable,
                        "background_shift": _finiteOptimizationValue(
                            ecmPass.get("background_shift")
                        ),
                        "background_shift_threshold": _finiteOptimizationValue(
                            ecmPass.get("background_shift_threshold")
                        ),
                        "background_shift_stable": backgroundShiftStable,
                        "outer_stable_iters": _intOptimizationValue(
                            ecmPass.get("outer_stable_iters")
                        ),
                        "outer_patience_target": _intOptimizationValue(
                            ecmPass.get("outer_patience_target")
                        ),
                        "outer_inner_ecm_converged": innerECMConverged,
                        "reset_iteration": False,
                        "converged": bool(
                            outerObjectiveStable
                            and backgroundShiftStable
                            and innerECMConverged
                        ),
                        "final_solution": bool(ecmIndex == lastOuterIndex),
                    }
                )
            innerPath = ecmPass.get("optimization_path", [])
            if not isinstance(innerPath, list):
                continue
            for innerIndex, innerStep in enumerate(innerPath):
                if not isinstance(innerStep, Mapping):
                    continue
                objectiveValue = _finiteOptimizationValue(
                    innerStep.get("objective_value")
                )
                if objectiveValue is None:
                    continue
                rows.append(
                    {
                        "chromosome": str(chromosome),
                        "phase": phaseLabel,
                        "path_level": "inner",
                        "outer_pass": outerPass,
                        "inner_iter": int(innerStep.get("iter") or (innerIndex + 1)),
                        "record_order": int(startOrder + len(rows)),
                        "objective_name": str(innerStep.get("objective_name") or "nll"),
                        "objective_value": objectiveValue,
                        "objective_per_cell": None,
                        "change": _finiteOptimizationValue(innerStep.get("change")),
                        "threshold": _finiteOptimizationValue(
                            innerStep.get("threshold")
                        ),
                        "objective_stable": None,
                        "background_shift": None,
                        "background_shift_threshold": None,
                        "background_shift_stable": None,
                        "outer_stable_iters": None,
                        "outer_patience_target": None,
                        "outer_inner_ecm_converged": None,
                        "reset_iteration": bool(
                            innerStep.get("reset_iteration", False)
                            or int(innerStep.get("iter") or (innerIndex + 1)) <= 1
                        ),
                        "converged": bool(innerStep.get("converged", False)),
                        "final_solution": bool(
                            ecmIndex == lastOuterIndex
                            and innerIndex == len(innerPath) - 1
                        ),
                    }
                )
    return rows


def _safeOutputToken(value: Any, *, fallback: str) -> str:
    token = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(value).strip()
    ).strip("._-")
    return token or str(fallback)


def _diagnosticLogPaths(experimentName: str) -> DiagnosticLogPaths:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    prefix = f"consenrichOutput_{experimentToken}"
    return DiagnosticLogPaths(
        precision=Path(f"{prefix}_precision.v{__version__}.jsonl.gz"),
        convergence=Path(f"{prefix}_convergence.v{__version__}.jsonl"),
        delete_block_calibration=Path(
            f"{prefix}_delete_block_calibration.v{__version__}.jsonl.gz"
        ),
    )


def _runSummaryPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(f"consenrichOutput_{experimentToken}_summary.v{__version__}.jsonl")


def _correlationLengthPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(
        f"consenrichOutput_{experimentToken}_correlationLength.v{__version__}.tsv"
    )


def _correlationLengthPlotPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(
        f"consenrichOutput_{experimentToken}_correlationLength.v{__version__}.png"
    )


def _correlationLengthWindowsPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(
        f"consenrichOutput_{experimentToken}_correlationLengthWindows.v{__version__}.tsv"
    )


def _chromSizesOrderForPlannedChromosomes(
    chromSizes: Mapping[str, Any],
    plannedChromosomes: Sequence[str],
) -> List[str]:
    plannedChromosomeSet = {str(chromosome) for chromosome in plannedChromosomes}
    return [
        str(chromosome)
        for chromosome in chromSizes.keys()
        if str(chromosome) in plannedChromosomeSet
    ]


def _initializeDiagnosticLogs(paths: DiagnosticLogPaths) -> None:
    for path in paths:
        logging_utils.init_jsonl_log(path)


def _jsonDiagnosticValue(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_jsonDiagnosticValue(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonDiagnosticValue(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonDiagnosticValue(item) for item in value]
    if value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        valueFloat = float(value)
        return valueFloat if math.isfinite(valueFloat) else None
    return value


def _jsonlRecords(
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(rows, pd.DataFrame):
        frame = rows if columns is None else rows.reindex(columns=list(columns))
        rowRecords = [
            dict(zip(frame.columns, values))
            for values in frame.itertuples(index=False, name=None)
        ]
    else:
        rowRecords = []
        for row in rows:
            rowDict = dict(row)
            if columns is not None:
                rowDict = {str(column): rowDict.get(column) for column in columns}
            rowRecords.append(rowDict)
    records: list[dict[str, Any]] = []
    for row in rowRecords:
        converted = {str(key): _jsonDiagnosticValue(value) for key, value in row.items()}
        records.append({key: value for key, value in converted.items() if value is not None})
    return records


def _writeJsonlRecords(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> int:
    records = _jsonlRecords(rows, columns)
    logging_utils.init_jsonl_log(path)
    return logging_utils.append_jsonl_log(path, records)


def _appendJsonlRecords(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> int:
    records = _jsonlRecords(rows, columns)
    if not records:
        return 0
    return logging_utils.append_jsonl_log(path, records)


def _appendMappingDiagnostics(
    path: Path,
    *,
    recordType: str,
    event: str,
    chromosome: str | None,
    values: Mapping[str, Any],
) -> int:
    record = {
        "record_type": recordType,
        "event": event,
        "chromosome": chromosome,
        **{str(key): value for key, value in values.items()},
    }
    return _appendJsonlRecords(path, [record])


def _selectPrecisionDiagnosticIntervalRows(
    frame: pd.DataFrame,
    *,
    detail: str,
    maxRowsPerChromosome: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    totalRows = int(len(frame))
    detail_ = str(detail or constants.OUTPUT_DEFAULT_PRECISION_DIAGNOSTIC_DETAIL)
    if detail_ == "full":
        positions = np.arange(totalRows, dtype=np.int64)
    elif detail_ == "sampled":
        maxRows = max(int(maxRowsPerChromosome), 0)
        if totalRows <= maxRows:
            positions = np.arange(totalRows, dtype=np.int64)
        elif maxRows == 0:
            positions = np.empty(0, dtype=np.int64)
        else:
            positions = np.unique(
                np.linspace(0, totalRows - 1, num=maxRows, dtype=np.int64)
            )
    elif detail_ == "summary":
        positions = np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported precision diagnostic detail {detail!r}")
    selected = frame.iloc[positions].copy() if positions.size else frame.iloc[[]].copy()
    return (
        selected,
        positions,
        {
            "detail": detail_,
            "rows_total": totalRows,
            "rows_written": int(positions.size),
            "rows_omitted": int(max(totalRows - int(positions.size), 0)),
            "max_rows_per_chromosome": int(maxRowsPerChromosome),
            "sampled": bool(int(positions.size) < totalRows),
        },
    )


def _genomeOptimizationPathPrefix(experimentName: str) -> str:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return f"consenrichOutput_{experimentToken}_genome_optimizationPath.v{__version__}"


def _precisionReweightingHistogramPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(
        f"consenrichOutput_{experimentToken}_precisionReweightingHistograms.v{__version__}.png"
    )


def _replicateCalibrationPlotPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(
        f"consenrichOutput_{experimentToken}_replicateCalibration.v{__version__}.png"
    )


def _deleteBlockCalibrationPlotPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(
        f"consenrichOutput_{experimentToken}_deleteBlockCalibration.v{__version__}.png"
    )


def _replicateExchangeabilityPlotPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(f"consenrichOutput_{experimentToken}_replicateExchangeability.png")


def _replicateExchangeabilitySummaryPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(f"consenrichOutput_{experimentToken}_replicateExchangeability.txt")


def _coerceOptimizationPathFrame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=OPTIMIZATION_PATH_COLUMNS)
    if frame.empty:
        return frame
    for numericColumn in (
        "objective_value",
        "record_order",
        "outer_pass",
        "inner_iter",
        "objective_per_cell",
        "change",
        "threshold",
        "background_shift",
        "background_shift_threshold",
    ):
        frame[numericColumn] = pd.to_numeric(frame[numericColumn], errors="coerce")
    for boolColumn in (
        "objective_stable",
        "background_shift_stable",
        "outer_inner_ecm_converged",
        "reset_iteration",
        "converged",
        "final_solution",
    ):
        frame[boolColumn] = frame[boolColumn].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    frame.loc[
        (frame["path_level"] == "inner") & (frame["inner_iter"] <= 1),
        "reset_iteration",
    ] = True
    return frame.dropna(subset=["record_order", "objective_value"])


def _plotOptimizationPathLog(
    rows: Sequence[Mapping[str, Any]],
    path: str,
    *,
    dpi: int = 400,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "outputParams.plotOptimizationPath=True but matplotlib is not installed; "
            "wrote the optimization .log only."
        )
        return False

    frame = _coerceOptimizationPathFrame(rows)
    if frame.empty:
        logger.warning(
            "optimizationPath.plot skipped because no trace rows were recorded."
        )
        return False
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    innerFrame = frame[
        (frame["path_level"] == "inner") & (frame["phase"] == "post_process_noise_fit")
    ].copy()
    if innerFrame.empty:
        logger.warning(
            "optimizationPath.plot skipped because no final-stage inner trace rows "
            "were recorded."
        )
        return False
    innerFrame = innerFrame.sort_values("record_order").reset_index(drop=True)
    innerFrame["plot_order"] = np.arange(1, len(innerFrame) + 1, dtype=np.int64)
    initialObjective = float(innerFrame["objective_value"].iloc[0])
    innerFrame["nll_improvement"] = initialObjective - innerFrame["objective_value"]

    fig, innerAx = plt.subplots(
        1,
        1,
        figsize=(10.0, 4.4),
        constrained_layout=True,
    )

    def _legend(axis: Any) -> None:
        try:
            handles, _labels = axis.get_legend_handles_labels()
        except AttributeError:
            axis.legend(loc="best", fontsize=8, frameon=False)
            return
        if handles:
            axis.legend(loc="best", fontsize=8, frameon=False)

    navyBlue = "#003B73"
    burntOrange = "#C65A1E"
    darkBlack = "#050505"
    pathColor = navyBlue
    startColor = darkBlack
    finalColor = burntOrange
    innerAx.plot(
        innerFrame["plot_order"],
        innerFrame["nll_improvement"],
        marker=".",
        linewidth=1.25,
        markersize=4.2,
        alpha=0.96,
        color=pathColor,
    )
    startRows = innerFrame[
        innerFrame["reset_iteration"].astype(bool) | (innerFrame["inner_iter"] <= 1)
    ].copy()
    if not startRows.empty:
        innerAx.scatter(
            startRows["plot_order"],
            startRows["nll_improvement"],
            s=44,
            marker="o",
            facecolors="none",
            edgecolors=startColor,
            linewidths=1.0,
            zorder=4,
            label="outer pass start",
        )
    finalInner = innerFrame[innerFrame["final_solution"].astype(bool)]
    if not finalInner.empty:
        innerAx.scatter(
            finalInner["plot_order"],
            finalInner["nll_improvement"],
            s=76,
            marker="o",
            linewidths=1.0,
            edgecolors=darkBlack,
            c=finalColor,
            zorder=5,
            label="final solution",
        )
        for _, row in finalInner.iterrows():
            innerAx.annotate(
                "final",
                xy=(row["plot_order"], row["nll_improvement"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=darkBlack,
            )
    chromosomes = [str(value) for value in innerFrame["chromosome"].dropna().unique()]
    chromosomeLabel = (
        chromosomes[0] if len(chromosomes) == 1 else ", ".join(chromosomes)
    )
    innerAx.set_title(
        f"Consenrich ECM Objective Path: {chromosomeLabel}",
        color=darkBlack,
    )
    innerAx.set_xlabel("Total Iterations", color=darkBlack)
    innerAx.set_ylabel("NLL improvement", color=darkBlack)
    innerAx.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    _legend(innerAx)

    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    logger.info("optimizationPath.output wrote %s dpi=%d", path, int(dpi))
    return True


def _mergePrecisionReweightingHistogramSamples(
    samples: dict[str, Any],
    frame: pd.DataFrame,
    *,
    sampleSize: int,
    columns: Sequence[str] = ("lambda", "kappa"),
) -> None:
    if isinstance(sampleSize, (bool, np.bool_)):
        raise ValueError(
            "precision reweighting histogram sample size must be positive"
        )
    sampleSize_ = int(sampleSize)
    if sampleSize_ <= 0:
        raise ValueError(
            "precision reweighting histogram sample size must be positive"
        )
    for column in columns:
        if column not in frame:
            continue
        nextValues = pd.to_numeric(frame[column]).to_numpy(dtype=np.float64)
        if nextValues.size == 0:
            continue
        priorityKey = f"_{column}_priority"
        seenKey = f"_{column}_seen"
        rngKey = f"_{column}_rng"
        rng = samples.get(rngKey)
        if rng is None:
            rng = np.random.default_rng(1)
            samples[rngKey] = rng
        priorValues = np.asarray(samples.get(column, []), dtype=np.float64).reshape(
            -1
        )
        priorPriority = np.asarray(
            samples.get(priorityKey, []),
            dtype=np.float64,
        ).reshape(-1)
        if priorValues.size != priorPriority.size:
            priorPriority = rng.random(priorValues.size, dtype=np.float64)
        nextPriority = rng.random(nextValues.size, dtype=np.float64)
        mergedValues = np.concatenate((priorValues, nextValues))
        mergedPriority = np.concatenate((priorPriority, nextPriority))
        if mergedValues.size > sampleSize_:
            keep = np.argpartition(mergedPriority, sampleSize_ - 1)[:sampleSize_]
            mergedValues = mergedValues[keep]
            mergedPriority = mergedPriority[keep]
        samples[column] = mergedValues
        samples[priorityKey] = mergedPriority
        samples[seenKey] = int(samples.get(seenKey, 0)) + int(nextValues.size)


def _histogramBins(values: np.ndarray) -> np.ndarray | int:
    if values.size <= 1:
        return 10
    low = float(np.min(values))
    high = float(np.max(values))
    if low <= 0.0:
        return 40
    if high <= low:
        return np.geomspace(low / 1.25, high * 1.25, 21)
    return np.geomspace(low, high, 41)


def _plotPrecisionReweightingHistograms(
    samples: Mapping[str, Any],
    path: str | Path,
    *,
    dpi: int = 400,
) -> bool:
    lambdaValues = np.asarray(samples.get("lambda", []), dtype=np.float64).reshape(
        -1
    )
    kappaValues = np.asarray(samples.get("kappa", []), dtype=np.float64).reshape(-1)
    lambdaValues = lambdaValues[lambdaValues > 0.0]
    kappaValues = kappaValues[kappaValues > 0.0]
    plotItems = []
    if lambdaValues.size:
        plotItems.append(
            (
                lambdaValues,
                "Observation precision weights",
                r"$\lambda_i$",
                "#003B73",
            )
        )
    if kappaValues.size:
        plotItems.append(
            (
                kappaValues,
                "Process precision weights",
                r"$\kappa_i$",
                "#C65A1E",
            )
        )
    if not plotItems:
        logger.info(
            "precisionReweightingHistograms.plot skipped because no enabled "
            "precision reweighting values were sampled."
        )
        return False

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "outputParams.plotPrecisionReweightingHistograms=True but matplotlib "
            "is not installed. Skipped precision reweighting histogram plot."
        )
        return False

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    darkBlack = "#050505"
    gridColor = "#D8D8D8"
    fig, axes = plt.subplots(
        1,
        len(plotItems),
        figsize=(5.8 * len(plotItems), 4.8),
        constrained_layout=True,
    )
    axesSeq = list(np.ravel(axes))
    for axis, (values, title, label, color) in zip(axesSeq, plotItems):
        axis.set_title(title, color=darkBlack)
        axis.set_xlabel(label, color=darkBlack)
        axis.set_ylabel("Sampled intervals", color=darkBlack)
        axis.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
        axis.hist(
            values,
            bins=_histogramBins(values),
            color=color,
            alpha=0.84,
            edgecolor=darkBlack,
            linewidth=0.35,
        )
        axis.axvline(
            float(np.median(values)),
            color=darkBlack,
            linewidth=1.2,
            alpha=0.9,
            label="median",
        )
        if float(np.min(values)) > 0.0 and float(np.max(values)) > float(
            np.min(values)
        ):
            axis.set_xscale("log")
        axis.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle("Precision Reweighting Histograms", color=darkBlack)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    logger.info(
        "precisionReweightingHistograms.output wrote %s dpi=%d",
        path,
        int(dpi),
    )
    return True


def _plotReplicateCalibration(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    dpi: int = 400,
) -> bool:
    frame = pd.DataFrame(list(rows), columns=GAIN_SUMMARY_COLUMNS)
    if frame.empty:
        logger.info("replicateCalibration.plot skipped because no gain rows exist.")
        return False
    gain = pd.to_numeric(frame["gain_avg"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    gainSD = pd.to_numeric(frame["gain_std"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finiteCounts = pd.to_numeric(
        frame["finite_interval_count"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    valid = (finiteCounts > 0.0) & np.isfinite(gain)
    if not np.any(valid):
        logger.info(
            "replicateCalibration.plot skipped because no finite gains were recorded."
        )
        return False

    plotFrame = frame.loc[valid].copy()
    plotGain = gain[valid]
    plotGainSD = np.nan_to_num(gainSD[valid], nan=0.0, posinf=0.0, neginf=0.0)
    plotGainSD = np.maximum(plotGainSD, 0.0)
    replicateIndex = pd.to_numeric(
        plotFrame["replicate_index"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    missingIndex = ~np.isfinite(replicateIndex)
    if np.any(missingIndex):
        replicateIndex[missingIndex] = np.arange(
            1,
            plotGain.size + 1,
            dtype=np.float64,
        )[missingIndex]
    labels = []
    for idx, (_rowIndex, row) in enumerate(plotFrame.iterrows()):
        label = str(row.get("sample_name") or f"replicate_{idx + 1}")
        if len(label) > 18:
            label = f"{label[:15]}..."
        labels.append(label)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "replicateCalibration.plot skipped because matplotlib is not installed."
        )
        return False

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    darkBlack = "#050505"
    navyBlue = "#003B73"
    burntOrange = "#C65A1E"
    gridColor = "#D8D8D8"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.8),
        constrained_layout=True,
    )
    histAx, gainAx = list(np.ravel(axes))

    histAx.set_title("Replicate Calibration Factors", color=darkBlack)
    histAx.set_xlabel("Gain", color=darkBlack)
    histAx.set_ylabel("Replicates", color=darkBlack)
    histAx.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    histAx.hist(
        plotGain,
        bins=_histogramBins(plotGain),
        color=navyBlue,
        alpha=0.84,
        edgecolor=darkBlack,
        linewidth=0.35,
    )
    histAx.axvline(
        float(np.median(plotGain)),
        color=darkBlack,
        linewidth=1.2,
        alpha=0.9,
        label="median",
    )
    if float(np.min(plotGain)) > 0.0 and float(np.max(plotGain)) > float(
        np.min(plotGain)
    ):
        histAx.set_xscale("log")
    histAx.legend(loc="best", fontsize=8, frameon=False)

    x = np.arange(plotGain.size, dtype=np.float64)
    gainAx.set_title("Replicate Gain by Sample", color=darkBlack)
    gainAx.set_xlabel("Replicate", color=darkBlack)
    gainAx.set_ylabel("Gain", color=darkBlack)
    gainAx.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    gainAx.errorbar(
        x,
        plotGain,
        yerr=plotGainSD,
        fmt="o",
        color=burntOrange,
        ecolor=navyBlue,
        elinewidth=1.0,
        capsize=3.0,
        markersize=5.0,
        label="replicate",
    )
    gainAx.axhline(
        1.0,
        color=darkBlack,
        linewidth=1.0,
        alpha=0.75,
        linestyle="--",
        label="unit",
    )
    if plotGain.size <= 24:
        gainAx.set_xticks(x)
        gainAx.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    else:
        gainAx.set_xticks(x)
        gainAx.set_xticklabels(
            [str(int(value)) for value in replicateIndex],
            rotation=90,
            fontsize=6,
        )
    gainAx.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle("Replicate Calibration", color=darkBlack)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    logger.info("replicateCalibration.output wrote %s dpi=%d", path, int(dpi))
    return True


def _deleteBlockBlockFactorValues(
    factor: Any,
    calibrationModel: Mapping[str, Any] | None,
) -> np.ndarray:
    factorArr = np.asarray(factor, dtype=np.float64).reshape(-1)
    if factorArr.size == 0:
        return np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(factorArr)) or np.any(factorArr <= 0.0):
        raise RuntimeError("delete-block calibration factors must be positive finite")
    calibration = _summaryMapping(calibrationModel)
    foldRefits = _summaryMapping(calibration.get("fold_refits"))
    blockLen = _summaryInt(foldRefits.get("block_len_intervals"))
    if blockLen is None:
        blockLen = _summaryInt(calibration.get("block_len_intervals"))
    blockLen = max(1, int(blockLen or 1))
    starts = np.arange(0, factorArr.size, blockLen, dtype=np.int64)
    counts = np.minimum(blockLen, factorArr.size - starts).astype(np.float64)
    return np.add.reduceat(factorArr, starts) / counts


def _deleteBlockCoverageRowsForPlot(
    *,
    chromosome: str,
    calibrationModel: Mapping[str, Any] | None,
    summary: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    calibration = _summaryMapping(calibrationModel)
    rawRows = calibration.get("state_uncertainty_coverage")
    rows: list[Mapping[str, Any]]
    if isinstance(rawRows, Sequence) and not isinstance(rawRows, (str, bytes)):
        rows = [row for row in rawRows if isinstance(row, Mapping)]
    elif summary is not None and not summary.empty:
        rows = [
            row
            for row in summary.to_dict(orient="records")
            if isinstance(row, Mapping)
        ]
    else:
        rows = []
    return [{"chromosome": str(chromosome), **dict(row)} for row in rows]


def _plotDeleteBlockCalibration(
    samples: Mapping[str, Any],
    coverageRows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    dpi: int = 400,
) -> bool:
    factorValues = np.asarray(samples.get("factor", []), dtype=np.float64).reshape(-1)
    factorValues = factorValues[np.isfinite(factorValues) & (factorValues > 0.0)]
    sdFactorValues = np.sqrt(factorValues)
    coverageFrame = pd.DataFrame(list(coverageRows))
    if factorValues.size == 0 and coverageFrame.empty:
        logger.info(
            "deleteBlockCalibration.plot skipped because no calibration values exist."
        )
        return False

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "deleteBlockCalibration.plot skipped because matplotlib is not installed."
        )
        return False

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    darkBlack = "#050505"
    burntOrange = "#C65A1E"
    green = "#207A4A"
    violet = "#6A4C93"
    gridColor = "#D8D8D8"
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.6, 8.6),
        constrained_layout=True,
    )
    sdFactorAx, sdCDFax, coverageAx, detailAx = list(np.ravel(axes))

    sdFactorAx.set_title("Block-Level SD Multipliers", color=darkBlack)
    sdFactorAx.set_xlabel("Block mean SD multiplier", color=darkBlack)
    sdFactorAx.set_ylabel("Blocks", color=darkBlack)
    sdFactorAx.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    if sdFactorValues.size:
        sdFactorAx.hist(
            sdFactorValues,
            bins=_histogramBins(sdFactorValues),
            color=violet,
            alpha=0.84,
            edgecolor=darkBlack,
            linewidth=0.35,
        )
        sdFactorAx.axvline(
            float(np.median(sdFactorValues)),
            color=darkBlack,
            linewidth=1.2,
            alpha=0.9,
            label="median",
        )
        if float(np.min(sdFactorValues)) > 0.0 and float(
            np.max(sdFactorValues)
        ) > float(np.min(sdFactorValues)):
            sdFactorAx.set_xscale("log")
        sdFactorAx.legend(loc="best", fontsize=8, frameon=False)
    else:
        sdFactorAx.text(
            0.5,
            0.5,
            "No factor values",
            transform=sdFactorAx.transAxes,
            ha="center",
            va="center",
            color=darkBlack,
        )

    sdCDFax.set_title("SD Multiplier ECDF", color=darkBlack)
    sdCDFax.set_xlabel("Block mean SD multiplier", color=darkBlack)
    sdCDFax.set_ylabel("Cumulative fraction", color=darkBlack)
    sdCDFax.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    if sdFactorValues.size:
        sortedSD = np.sort(sdFactorValues)
        cumulative = (np.arange(sortedSD.size, dtype=np.float64) + 1.0) / float(
            sortedSD.size
        )
        sdCDFax.plot(
            sortedSD,
            cumulative,
            color=violet,
            linewidth=1.4,
            label="ECDF",
        )
        sdCDFax.axvline(
            float(np.median(sdFactorValues)),
            color=darkBlack,
            linewidth=1.2,
            alpha=0.9,
            label="median",
        )
        if float(np.min(sdFactorValues)) > 0.0 and float(
            np.max(sdFactorValues)
        ) > float(np.min(sdFactorValues)):
            sdCDFax.set_xscale("log")
        sdCDFax.legend(loc="best", fontsize=8, frameon=False)
    else:
        sdCDFax.text(
            0.5,
            0.5,
            "No factor values",
            transform=sdCDFax.transAxes,
            ha="center",
            va="center",
            color=darkBlack,
        )

    def weightedRows(
        frame: pd.DataFrame,
        groupColumn: str,
        valueColumns: Sequence[str],
    ) -> pd.DataFrame:
        rows = []
        for groupValue, group in frame.groupby(groupColumn, sort=True):
            weight = group["n"].to_numpy(dtype=np.float64)
            weightTotal = float(np.sum(weight))
            if weightTotal <= 0.0:
                continue
            row: dict[str, Any] = {groupColumn: groupValue}
            for column in valueColumns:
                values = group[column].to_numpy(dtype=np.float64)
                row[column] = float(np.sum(values * weight) / weightTotal)
            rows.append(row)
        return pd.DataFrame(rows)

    requiredCoverageColumns = {
        "stratum",
        "target",
        "n",
        "coverage_before",
        "coverage_after",
    }
    if requiredCoverageColumns <= set(coverageFrame.columns):
        coveragePlotFrame = coverageFrame.copy()
        for column in (
            "target",
            "n",
            "coverage_before",
            "coverage_after",
            "median_width_before",
            "median_width_after",
        ):
            if column in coveragePlotFrame:
                coveragePlotFrame[column] = pd.to_numeric(
                    coveragePlotFrame[column],
                    errors="coerce",
                )
        coveragePlotFrame = coveragePlotFrame.dropna(
            subset=["target", "n", "coverage_before", "coverage_after"]
        )
        coveragePlotFrame = coveragePlotFrame[coveragePlotFrame["n"] > 0.0]
        overallFrame = coveragePlotFrame[
            coveragePlotFrame["stratum"].astype(str) == "overall"
        ].copy()
    else:
        coveragePlotFrame = pd.DataFrame()
        overallFrame = pd.DataFrame()

    coverageAx.set_title("Coverage Calibration", color=darkBlack)
    coverageAx.set_xlabel("Nominal coverage", color=darkBlack)
    coverageAx.set_ylabel("Observed coverage", color=darkBlack)
    coverageAx.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    if overallFrame.empty:
        coverageAx.text(
            0.5,
            0.5,
            "No coverage rows",
            transform=coverageAx.transAxes,
            ha="center",
            va="center",
            color=darkBlack,
        )
    else:
        groupedFrame = weightedRows(
            overallFrame,
            "target",
            ("coverage_before", "coverage_after"),
        ).sort_values(
            "target",
            kind="mergesort",
        )
        targetValues = groupedFrame["target"].to_numpy(dtype=np.float64)
        rawValues = groupedFrame["coverage_before"].to_numpy(dtype=np.float64)
        calibratedValues = groupedFrame["coverage_after"].to_numpy(dtype=np.float64)
        plotValues = np.concatenate((targetValues, rawValues, calibratedValues))
        axisMin = max(0.0, float(np.min(plotValues)) - 0.02)
        axisMax = min(1.0, float(np.max(plotValues)) + 0.02)
        coverageAx.plot(
            [axisMin, axisMax],
            [axisMin, axisMax],
            color=darkBlack,
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
            label="target",
        )
        coverageAx.plot(
            targetValues,
            rawValues,
            marker="o",
            color=burntOrange,
            linewidth=1.2,
            label="raw",
        )
        coverageAx.plot(
            targetValues,
            calibratedValues,
            marker="o",
            color=green,
            linewidth=1.2,
            label="calibrated",
        )
        coverageAx.set_xlim(axisMin, axisMax)
        coverageAx.set_ylim(axisMin, axisMax)
        coverageAx.legend(loc="best", fontsize=8, frameon=False)

    detailAx.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    strataFrame = pd.DataFrame()
    if not coveragePlotFrame.empty:
        strataFrame = coveragePlotFrame[
            coveragePlotFrame["stratum"].astype(str) != "overall"
        ].copy()
    if not strataFrame.empty:
        targetValue = float(np.max(strataFrame["target"].to_numpy(dtype=np.float64)))
        targetFrame = strataFrame[
            np.isclose(strataFrame["target"].to_numpy(dtype=np.float64), targetValue)
        ].copy()
        strataPlotFrame = weightedRows(
            targetFrame,
            "stratum",
            ("coverage_before", "coverage_after"),
        )
        if not strataPlotFrame.empty:
            x = np.arange(len(strataPlotFrame), dtype=np.float64)
            width = 0.38
            detailAx.set_title(
                f"Strata Coverage at Target {targetValue:.2f}",
                color=darkBlack,
            )
            detailAx.set_ylabel("Observed coverage", color=darkBlack)
            detailAx.bar(
                x - width / 2.0,
                strataPlotFrame["coverage_before"].to_numpy(dtype=np.float64),
                width=width,
                color=burntOrange,
                alpha=0.84,
                label="raw",
            )
            detailAx.bar(
                x + width / 2.0,
                strataPlotFrame["coverage_after"].to_numpy(dtype=np.float64),
                width=width,
                color=green,
                alpha=0.84,
                label="calibrated",
            )
            detailAx.axhline(
                targetValue,
                color=darkBlack,
                linewidth=1.0,
                linestyle="--",
                alpha=0.65,
                label="target",
            )
            detailAx.set_ylim(0.0, 1.0)
            detailAx.set_xticks(x)
            detailAx.set_xticklabels(
                strataPlotFrame["stratum"].astype(str),
                rotation=45,
                ha="right",
                fontsize=8,
            )
            detailAx.legend(loc="best", fontsize=8, frameon=False)
        else:
            strataFrame = pd.DataFrame()
    if strataFrame.empty:
        widthColumns = {
            "target",
            "median_width_before",
            "median_width_after",
        }
        widthFrame = (
            overallFrame.dropna(
                subset=["target", "median_width_before", "median_width_after"]
            ).copy()
            if widthColumns <= set(overallFrame.columns)
            else pd.DataFrame()
        )
        if widthFrame.empty:
            detailAx.set_title("Coverage Detail", color=darkBlack)
            detailAx.text(
                0.5,
                0.5,
                "No strata rows",
                transform=detailAx.transAxes,
                ha="center",
                va="center",
                color=darkBlack,
            )
        else:
            widthPlotFrame = weightedRows(
                widthFrame,
                "target",
                ("median_width_before", "median_width_after"),
            ).sort_values(
                "target",
                kind="mergesort",
            )
            detailAx.set_title("Median Width by Coverage Target", color=darkBlack)
            detailAx.set_xlabel("Nominal coverage", color=darkBlack)
            detailAx.set_ylabel("Median interval width", color=darkBlack)
            detailAx.plot(
                widthPlotFrame["target"].to_numpy(dtype=np.float64),
                widthPlotFrame["median_width_before"].to_numpy(dtype=np.float64),
                marker="o",
                color=burntOrange,
                linewidth=1.2,
                label="raw",
            )
            detailAx.plot(
                widthPlotFrame["target"].to_numpy(dtype=np.float64),
                widthPlotFrame["median_width_after"].to_numpy(dtype=np.float64),
                marker="o",
                color=green,
                linewidth=1.2,
                label="calibrated",
            )
            detailAx.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle("Delete-Block Calibration", color=darkBlack)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    logger.info("deleteBlockCalibration.output wrote %s dpi=%d", path, int(dpi))
    return True


def _replicateExchangeabilityDenseMatrix(
    pooledBlockVars: np.ndarray,
    pooledPriorVariance: np.ndarray,
    pooledSampleIndex: np.ndarray,
    pooledChromIndex: np.ndarray,
    pooledBlockStarts: np.ndarray,
    sampleCount: int,
    *,
    maxBlocks: int = _REPLICATE_EXCHANGEABILITY_MAX_BLOCKS,
) -> tuple[np.ndarray, int, int]:
    blockVars = np.asarray(pooledBlockVars, dtype=np.float64).reshape(-1)
    priorVariance = np.asarray(pooledPriorVariance, dtype=np.float64).reshape(-1)
    sampleIndex = np.asarray(pooledSampleIndex, dtype=np.int64).reshape(-1)
    chromIndex = np.asarray(pooledChromIndex, dtype=np.int64).reshape(-1)
    blockStarts = np.asarray(pooledBlockStarts, dtype=np.int64).reshape(-1)
    rowCount = int(blockVars.size)
    if not (
        priorVariance.size
        == sampleIndex.size
        == chromIndex.size
        == blockStarts.size
        == rowCount
    ):
        raise RuntimeError("replicate exchangeability block arrays are misaligned")
    if sampleCount < 0:
        raise RuntimeError("replicate exchangeability sample count is negative")
    if rowCount == 0:
        return np.empty((0, int(sampleCount)), dtype=np.float64), 0, 0
    if np.any(sampleIndex < 0) or np.any(sampleIndex >= int(sampleCount)):
        raise RuntimeError("replicate exchangeability sample index is out of range")
    validInternal = (
        np.isfinite(blockVars)
        & np.isfinite(priorVariance)
        & (blockVars > 0.0)
        & (priorVariance > 0.0)
    )
    if not np.all(validInternal):
        raise RuntimeError("replicate exchangeability variances must be positive finite")
    order = np.lexsort((sampleIndex, blockStarts, chromIndex))
    sortedChrom = chromIndex[order]
    sortedStarts = blockStarts[order]
    sortedSamples = sampleIndex[order]
    keyBreaks = np.empty(rowCount, dtype=bool)
    keyBreaks[0] = True
    keyBreaks[1:] = (sortedChrom[1:] != sortedChrom[:-1]) | (
        sortedStarts[1:] != sortedStarts[:-1]
    )
    blockIds = np.cumsum(keyBreaks, dtype=np.int64) - 1
    duplicateRows = (
        (blockIds[1:] == blockIds[:-1])
        & (sortedSamples[1:] == sortedSamples[:-1])
        if rowCount > 1
        else np.zeros(0, dtype=bool)
    )
    if np.any(duplicateRows):
        raise RuntimeError(
            "replicate exchangeability found duplicate replicate-block rows"
        )
    blockCount = int(blockIds[-1] + 1)
    maxBlocks_ = int(max(_REPLICATE_EXCHANGEABILITY_MIN_BLOCKS, maxBlocks))
    if blockCount > maxBlocks_:
        selectedBlockIds = np.linspace(
            0,
            blockCount - 1,
            maxBlocks_,
            dtype=np.int64,
        )
        keepRows = np.isin(blockIds, selectedBlockIds)
        selectedOrder = order[keepRows]
        selectedSamples = sortedSamples[keepRows]
        selectedRows = np.searchsorted(selectedBlockIds, blockIds[keepRows])
        matrixRows = int(selectedBlockIds.size)
    else:
        selectedOrder = order
        selectedSamples = sortedSamples
        selectedRows = blockIds
        matrixRows = blockCount
    matrix = np.full((matrixRows, int(sampleCount)), np.nan, dtype=np.float64)
    logSdMultiplier = 0.5 * (
        np.log(blockVars[selectedOrder]) - np.log(priorVariance[selectedOrder])
    )
    matrix[selectedRows, selectedSamples] = logSdMultiplier
    return matrix, blockCount, matrixRows


def _replicateExchangeabilityPairwiseSign(
    centeredMatrix: np.ndarray,
    *,
    minBlocks: int = _REPLICATE_EXCHANGEABILITY_MIN_BLOCKS,
) -> dict[str, Any]:
    matrix = np.asarray(centeredMatrix, dtype=np.float64)
    replicateCount = int(matrix.shape[1])
    pValues = np.full((replicateCount, replicateCount), np.nan, dtype=np.float64)
    wins = np.zeros((replicateCount, replicateCount), dtype=np.int64)
    signBlocks = np.zeros((replicateCount, replicateCount), dtype=np.int64)
    sharedBlocks = np.zeros((replicateCount, replicateCount), dtype=np.int64)
    if replicateCount < 2:
        return {
            "pValues": pValues,
            "wins": wins,
            "signBlocks": signBlocks,
            "sharedBlocks": sharedBlocks,
            "minPair": None,
        }
    pairA, pairB = np.triu_indices(replicateCount, k=1)
    blockCount = max(1, int(matrix.shape[0]))
    chunkSize = max(
        1,
        min(
            pairA.size,
            _REPLICATE_EXCHANGEABILITY_PAIR_CHUNK_ELEMENTS // blockCount,
        ),
    )
    for chunkStart in range(0, int(pairA.size), int(chunkSize)):
        chunkEnd = min(chunkStart + int(chunkSize), int(pairA.size))
        a = pairA[chunkStart:chunkEnd]
        b = pairB[chunkStart:chunkEnd]
        left = matrix[:, a]
        right = matrix[:, b]
        valid = np.isfinite(left) & np.isfinite(right)
        diff = left - right
        positive = valid & (diff > 0.0)
        negative = valid & (diff < 0.0)
        chunkWins = np.count_nonzero(positive, axis=0).astype(np.int64, copy=False)
        chunkLosses = np.count_nonzero(negative, axis=0).astype(np.int64, copy=False)
        chunkSignBlocks = chunkWins + chunkLosses
        chunkSharedBlocks = np.count_nonzero(valid, axis=0).astype(
            np.int64,
            copy=False,
        )
        chunkMin = np.minimum(chunkWins, chunkLosses)
        chunkP = np.ones(chunkWins.size, dtype=np.float64)
        informative = chunkSignBlocks >= int(minBlocks)
        if np.any(informative):
            chunkP[informative] = np.minimum(
                1.0,
                2.0
                * stats.binom.cdf(
                    chunkMin[informative],
                    chunkSignBlocks[informative],
                    0.5,
                ),
            )
            chunkP[informative] = np.maximum(
                chunkP[informative],
                np.nextafter(0.0, 1.0),
            )
        pValues[a, b] = chunkP
        pValues[b, a] = chunkP
        wins[a, b] = chunkWins
        wins[b, a] = chunkLosses
        signBlocks[a, b] = chunkSignBlocks
        signBlocks[b, a] = chunkSignBlocks
        sharedBlocks[a, b] = chunkSharedBlocks
        sharedBlocks[b, a] = chunkSharedBlocks
    validPairs = (
        np.triu(np.ones((replicateCount, replicateCount), dtype=bool), k=1)
        & (signBlocks >= int(minBlocks))
        & np.isfinite(pValues)
    )
    if not np.any(validPairs):
        minPair = None
    else:
        candidate = np.where(validPairs, pValues, np.inf)
        flatIndex = int(np.argmin(candidate))
        i, j = np.unravel_index(flatIndex, candidate.shape)
        minPair = {
            "replicateA": int(i),
            "replicateB": int(j),
            "pValue": float(pValues[i, j]),
            "winsA": int(wins[i, j]),
            "winsB": int(signBlocks[i, j] - wins[i, j]),
            "signBlocks": int(signBlocks[i, j]),
            "sharedBlocks": int(sharedBlocks[i, j]),
        }
    return {
        "pValues": pValues,
        "wins": wins,
        "signBlocks": signBlocks,
        "sharedBlocks": sharedBlocks,
        "minPair": minPair,
    }


def _replicateExchangeabilityFromLogSDMatrix(
    logSDMatrix: np.ndarray,
    *,
    sampleNames: Sequence[str] | None = None,
    seed: int = constants.UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
    permutationCount: int = _REPLICATE_EXCHANGEABILITY_PERMUTATIONS,
    minBlocks: int = _REPLICATE_EXCHANGEABILITY_MIN_BLOCKS,
) -> dict[str, Any]:
    matrix = np.asarray(logSDMatrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise RuntimeError("replicate exchangeability matrix must be two-dimensional")
    blockCount, replicateCount = (int(matrix.shape[0]), int(matrix.shape[1]))
    names = list(sampleNames or [])
    if len(names) < replicateCount:
        names.extend(
            f"replicate_{i + 1}" for i in range(len(names), replicateCount)
        )
    names = [str(name) for name in names[:replicateCount]]
    result: dict[str, Any] = {
        "status": "skipped",
        "reason": "",
        "replicateCount": replicateCount,
        "blockCount": blockCount,
        "completeBlockCount": 0,
        "sampleNames": names,
    }
    if replicateCount < 2:
        result["reason"] = "fewer than two replicates"
        return result
    if blockCount < minBlocks:
        result["reason"] = f"fewer than {int(minBlocks)} block rows"
        return result
    completeBlocks = np.all(np.isfinite(matrix), axis=1)
    completeMatrix = matrix[completeBlocks, :]
    completeBlockCount = int(completeMatrix.shape[0])
    result["completeBlockCount"] = completeBlockCount
    if completeBlockCount < minBlocks:
        result["reason"] = f"fewer than {int(minBlocks)} complete shared blocks"
        centeredForPairs = matrix - np.nanmean(matrix, axis=1, keepdims=True)
        result["pairwiseSign"] = _replicateExchangeabilityPairwiseSign(
            centeredForPairs,
            minBlocks=minBlocks,
        )
        return result
    centeredComplete = completeMatrix - np.mean(completeMatrix, axis=1, keepdims=True)
    effectByReplicate = np.mean(centeredComplete, axis=0)
    observedStatistic = float(np.max(np.abs(effectByReplicate)))
    permutationStats = np.empty(int(permutationCount), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for permutationIndex in range(int(permutationCount)):
        permuted = rng.permuted(centeredComplete, axis=1)
        permutationStats[permutationIndex] = float(
            np.max(np.abs(np.mean(permuted, axis=0)))
        )
    exceedanceCount = int(np.count_nonzero(permutationStats >= observedStatistic))
    pValue = float((exceedanceCount + 1.0) / (float(permutationCount) + 1.0))
    centeredForPairs = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    result.update(
        {
            "status": "ok",
            "reason": "ok",
            "effectByReplicate": effectByReplicate,
            "omnibusObserved": observedStatistic,
            "omnibusPValue": pValue,
            "omnibusPermutationStats": permutationStats,
            "permutationCount": int(permutationCount),
            "seed": int(seed),
            "pairwiseSign": _replicateExchangeabilityPairwiseSign(
                centeredForPairs,
                minBlocks=minBlocks,
            ),
        }
    )
    return result


def _replicateExchangeabilityFromPooledBlocks(
    pooledBlockVars: np.ndarray,
    pooledPriorVariance: np.ndarray,
    pooledSampleIndex: np.ndarray,
    pooledChromIndex: np.ndarray,
    pooledBlockStarts: np.ndarray,
    sampleCount: int,
    *,
    sampleNames: Sequence[str] | None = None,
    seed: int = constants.UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
) -> dict[str, Any]:
    matrix, blockCount, diagnosticBlockCount = _replicateExchangeabilityDenseMatrix(
        pooledBlockVars,
        pooledPriorVariance,
        pooledSampleIndex,
        pooledChromIndex,
        pooledBlockStarts,
        sampleCount,
    )
    result = _replicateExchangeabilityFromLogSDMatrix(
        matrix,
        sampleNames=sampleNames,
        seed=seed,
    )
    result["blockCount"] = int(blockCount)
    result["diagnosticBlockCount"] = int(diagnosticBlockCount)
    return result


def _warnReplicateVarianceHeterogeneity(
    diagnostic: Mapping[str, Any],
    diagnosticPath: str | Path,
) -> bool:
    if str(diagnostic.get("status", "skipped")) != "ok":
        return False
    requiredKeys = (
        "effectByReplicate",
        "rawEffectByReplicate",
        "omnibusPValue",
        "rawOmnibusPValue",
    )
    if any(key not in diagnostic for key in requiredKeys):
        return False

    adjustedEffect = np.asarray(
        diagnostic["effectByReplicate"],
        dtype=np.float64,
    ).reshape(-1)
    rawEffect = np.asarray(
        diagnostic["rawEffectByReplicate"],
        dtype=np.float64,
    ).reshape(-1)
    sampleNames = [str(name) for name in diagnostic.get("sampleNames", [])]
    if (
        adjustedEffect.size < 2
        or rawEffect.shape != adjustedEffect.shape
        or len(sampleNames) < int(adjustedEffect.size)
    ):
        return False

    rawSDRatio = float(np.exp(np.max(rawEffect) - np.min(rawEffect)))
    adjustedSDRatio = float(
        np.exp(np.max(adjustedEffect) - np.min(adjustedEffect))
    )
    fittedSDMultipliers = np.asarray(
        diagnostic.get("replicateSDMultipliers", []),
        dtype=np.float64,
    ).reshape(-1)
    fittedSDRatio = (
        float(np.max(fittedSDMultipliers) / np.min(fittedSDMultipliers))
        if fittedSDMultipliers.shape == adjustedEffect.shape
        else None
    )
    rawPValue = float(diagnostic["rawOmnibusPValue"])
    adjustedPValue = float(diagnostic["omnibusPValue"])
    rawWarning = rawPValue <= 0.05 and (
        rawSDRatio >= 1.5
        or (fittedSDRatio is not None and fittedSDRatio >= 1.5)
    )
    strongWarning = (
        diagnostic.get("priorVarianceFactorAdjusted") is True
        and adjustedPValue <= 0.01
        and adjustedSDRatio >= 1.25
    )
    if not (rawWarning or strongWarning):
        return False

    if strongWarning:
        pairValues = adjustedEffect
        pairBasis = "adjusted"
        warningLead = "Strong modeled heterogeneity warning"
    elif fittedSDRatio is not None and fittedSDRatio > rawSDRatio:
        pairValues = fittedSDMultipliers
        pairBasis = "fitted"
        warningLead = "Substantial modeled heterogeneity warning"
    else:
        pairValues = rawEffect
        pairBasis = "raw"
        warningLead = "Substantial modeled heterogeneity warning"
    lowIndex = int(np.argmin(pairValues))
    highIndex = int(np.argmax(pairValues))
    fittedRatioText = (
        f"{fittedSDRatio:.6g}" if fittedSDRatio is not None else "unavailable"
    )
    logger.warning(
        "%s: replicates exhibit blockwise variance heterogeneity. "
        "divergentReplicates=%r,%r pairBasis=%s rawSDRatio=%.6g "
        "adjustedSDRatio=%.6g fittedSDRatio=%s rawPValue=%.6g "
        "adjustedPValue=%.6g diagnosticFile=%s. The result is confined to "
        "blockwise variance and does not establish that global biological "
        "exchangeability is invalid.",
        warningLead,
        sampleNames[lowIndex],
        sampleNames[highIndex],
        pairBasis,
        rawSDRatio,
        adjustedSDRatio,
        fittedRatioText,
        rawPValue,
        adjustedPValue,
        str(Path(diagnosticPath).resolve()),
    )
    return True


def _writeReplicateExchangeabilitySummary(
    diagnostic: Mapping[str, Any],
    path: str | Path,
) -> bool:
    status = str(diagnostic.get("status", "skipped"))
    replicateCount = int(diagnostic.get("replicateCount", 0))
    blockCount = int(diagnostic.get("blockCount", 0))
    diagnosticBlockCount = int(diagnostic.get("diagnosticBlockCount", blockCount))
    completeBlockCount = int(diagnostic.get("completeBlockCount", 0))
    if status != "ok":
        reason = str(diagnostic.get("reason", "not enough evidence"))
        factorAdjusted = diagnostic.get("priorVarianceFactorAdjusted")
        factorAdjustedText = (
            ""
            if factorAdjusted is None
            else (
                " prior_variance_factor_adjusted="
                f"{'true' if bool(factorAdjusted) else 'false'}"
            )
        )
        text = (
            "replicate exchangeability diagnostics skipped: "
            f"reason={reason} replicate_count={replicateCount} "
            f"block_count={blockCount} complete_block_count={completeBlockCount}"
            f"{factorAdjustedText}\n"
        )
        Path(path).write_text(text, encoding="utf-8")
        logger.info("replicateExchangeability.output wrote %s", path)
        return True
    sampleNames = list(diagnostic.get("sampleNames", []))
    effect = np.asarray(diagnostic["effectByReplicate"], dtype=np.float64)
    topOrder = np.argsort(-np.abs(effect))[
        : min(_REPLICATE_EXCHANGEABILITY_TOP_EFFECTS, effect.size)
    ]
    pairwise = diagnostic.get("pairwiseSign", {})
    minPair = pairwise.get("minPair") if isinstance(pairwise, Mapping) else None
    lines = [
        "replicate exchangeability diagnostics",
        f"replicate_count: {replicateCount}",
        f"block_count: {blockCount}",
        f"complete_block_count: {completeBlockCount}",
        f"omnibus_statistic: {float(diagnostic['omnibusObserved']):.8g}",
        f"omnibus_p_value: {float(diagnostic['omnibusPValue']):.8g}",
    ]
    if diagnosticBlockCount != blockCount:
        lines.append(f"tested_block_count: {diagnosticBlockCount}")
    if "priorVarianceFactorAdjusted" in diagnostic:
        factorAdjusted = bool(diagnostic.get("priorVarianceFactorAdjusted", False))
        lines.append(
            f"prior_variance_factor_adjusted: {'true' if factorAdjusted else 'false'}"
        )
    if "rawOmnibusObserved" in diagnostic and "rawOmnibusPValue" in diagnostic:
        lines.extend(
            [
                (
                    "raw_omnibus_statistic: "
                    f"{float(diagnostic['rawOmnibusObserved']):.8g}"
                ),
                f"raw_omnibus_p_value: {float(diagnostic['rawOmnibusPValue']):.8g}",
            ]
        )
    if minPair is None:
        lines.append("most_imbalanced_pair: none")
    else:
        pairA = int(minPair["replicateA"])
        pairB = int(minPair["replicateB"])
        nameA = sampleNames[pairA] if pairA < len(sampleNames) else f"replicate_{pairA + 1}"
        nameB = sampleNames[pairB] if pairB < len(sampleNames) else f"replicate_{pairB + 1}"
        lines.extend(
            [
                f"most_imbalanced_pair: {pairA + 1} {nameA} vs {pairB + 1} {nameB}",
                f"most_imbalanced_pair_p_value: {float(minPair['pValue']):.8g}",
                (
                    "most_imbalanced_pair_sign_blocks: "
                    f"{int(minPair['signBlocks'])}"
                ),
                (
                    "most_imbalanced_pair_wins: "
                    f"{int(minPair['winsA'])} vs {int(minPair['winsB'])}"
                ),
            ]
        )
    replicateVarianceFactors = np.asarray(
        diagnostic.get("replicateVarianceFactors", []),
        dtype=np.float64,
    ).reshape(-1)
    replicateSDMultipliers = np.asarray(
        diagnostic.get("replicateSDMultipliers", []),
        dtype=np.float64,
    ).reshape(-1)
    if replicateVarianceFactors.size:
        lines.append("fitted_replicate_munc_factors:")
        lines.append("replicate_index\tsample_name\tvariance_factor\tsd_multiplier")
        for index in range(int(replicateVarianceFactors.size)):
            name = sampleNames[index] if index < len(sampleNames) else (
                f"replicate_{index + 1}"
            )
            sdMultiplier = (
                float(replicateSDMultipliers[index])
                if index < int(replicateSDMultipliers.size)
                else float(math.sqrt(replicateVarianceFactors[index]))
            )
            lines.append(
                f"{index + 1}\t{name}\t{float(replicateVarianceFactors[index]):.8g}"
                f"\t{sdMultiplier:.8g}"
            )
    lines.append("top_replicate_effects:")
    lines.append("replicate_index\tsample_name\tlog_sd_shift\tsd_multiplier")
    for index in topOrder:
        name = sampleNames[int(index)] if int(index) < len(sampleNames) else (
            f"replicate_{int(index) + 1}"
        )
        lines.append(
            f"{int(index) + 1}\t{name}\t{float(effect[index]):.8g}\t"
            f"{float(np.exp(effect[index])):.8g}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("replicateExchangeability.output wrote %s", path)
    return True


def _plotReplicateExchangeabilityDiagnostic(
    diagnostic: Mapping[str, Any],
    path: str | Path,
    *,
    dpi: int = 400,
) -> bool:
    if str(diagnostic.get("status", "skipped")) != "ok":
        logger.info(
            "replicateExchangeability.plot skipped because diagnostics were not fit."
        )
        return False
    permutationStats = np.asarray(
        diagnostic.get("omnibusPermutationStats", []),
        dtype=np.float64,
    ).reshape(-1)
    pairwise = diagnostic.get("pairwiseSign", {})
    wins = np.asarray(pairwise.get("wins", []), dtype=np.float64)
    signBlocks = np.asarray(pairwise.get("signBlocks", []), dtype=np.float64)
    if permutationStats.size == 0 or wins.ndim != 2 or signBlocks.shape != wins.shape:
        logger.info(
            "replicateExchangeability.plot skipped because plot arrays are empty."
        )
        return False
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "replicateExchangeability.plot skipped because matplotlib is not installed."
        )
        return False

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    darkBlack = "#050505"
    navyBlue = "#003B73"
    burntOrange = "#C65A1E"
    gridColor = "#D8D8D8"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.8),
        constrained_layout=True,
    )
    nullAx, heatAx = list(np.ravel(axes))
    observed = float(diagnostic["omnibusObserved"])
    plotPermutationStats = np.exp(permutationStats)
    plotObserved = float(np.exp(observed))
    nullAx.set_title("Block-Label Permutation Omnibus", color=darkBlack)
    nullAx.set_xlabel("Max replicate mean SD multiplier", color=darkBlack)
    nullAx.set_ylabel("Permutations", color=darkBlack)
    nullAx.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
    nullAx.hist(
        plotPermutationStats,
        bins=_histogramBins(plotPermutationStats),
        color=navyBlue,
        alpha=0.84,
        edgecolor=darkBlack,
        linewidth=0.35,
    )
    nullAx.axvline(
        plotObserved,
        color=burntOrange,
        linewidth=1.6,
        alpha=0.95,
        label="observed",
    )
    if "rawOmnibusObserved" in diagnostic:
        nullAx.axvline(
            float(np.exp(float(diagnostic["rawOmnibusObserved"]))),
            color=darkBlack,
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
            label="raw",
        )
    nullAx.legend(loc="best", fontsize=8, frameon=False)

    effect = np.asarray(diagnostic["effectByReplicate"], dtype=np.float64)
    order = np.argsort(effect)
    heat = np.full(wins.shape, np.nan, dtype=np.float64)
    signReady = signBlocks > 0.0
    heat[signReady] = (2.0 * (wins[signReady] / signBlocks[signReady])) - 1.0
    heat = heat[np.ix_(order, order)]
    image = heatAx.imshow(
        heat,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        interpolation="nearest",
        aspect="auto",
    )
    heatAx.set_title("Pairwise Block Sign Test", color=darkBlack)
    heatAx.set_xlabel("Replicate ordered by effect", color=darkBlack)
    heatAx.set_ylabel("Replicate ordered by effect", color=darkBlack)
    replicateCount = int(effect.size)
    if replicateCount <= 24:
        labels = [str(int(index) + 1) for index in order]
        ticks = np.arange(replicateCount)
        heatAx.set_xticks(ticks)
        heatAx.set_yticks(ticks)
        heatAx.set_xticklabels(labels, rotation=90, fontsize=6)
        heatAx.set_yticklabels(labels, fontsize=6)
    else:
        heatAx.set_xticks([])
        heatAx.set_yticks([])
    colorbar = fig.colorbar(image, ax=heatAx, fraction=0.046, pad=0.04)
    colorbar.set_label("Block-win balance vs 50:50", color=darkBlack)
    fig.suptitle("Replicate Exchangeability Diagnostics", color=darkBlack)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    logger.info("replicateExchangeability.output wrote %s dpi=%d", path, int(dpi))
    return True


def _plotCorrelationLengthInference(
    row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    path: str | Path,
    *,
    dpi: int = 400,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "outputParams.plotCorrelationLength=True but matplotlib is not installed. "
            "wrote the correlation-length TSV only."
        )
        return False

    def numberValue(*keys: str) -> float | None:
        for key in keys:
            value = row.get(key)
            if value is None:
                value = diagnostics.get(key)
            out = _summaryNumber(value)
            if out is not None:
                return float(out)
        return None

    pointBP = numberValue("correlation_length_bp", "estimateBP")
    lowerBP = numberValue("lower_bp", "lowerBP")
    upperBP = numberValue("upper_bp", "upperBP")
    workingSpanBP = numberValue("working_span_bp", "workingSpanBP")
    workingQuantile = numberValue("working_quantile", "workingQuantile")
    survivalBandRegionLower = numberValue("survivalBandRegionLower")
    survivalBandRegionUpper = numberValue("survivalBandRegionUpper")
    survivalBandJumpClosureUsed = bool(diagnostics["survivalBandJumpClosureUsed"])
    survivalBandJumpClosureCount = _summaryInt(
        diagnostics["survivalBandJumpClosureCount"]
    )
    contextBP = numberValue("context_bp")
    radiusDistributionBP = np.asarray(
        diagnostics["radiusDistributionBP"], dtype=np.float64
    )
    radiusCensored = np.asarray(diagnostics["radiusCensored"], dtype=np.bool_)
    selectedWindows = list(diagnostics["selectedWindows"])
    censorReasons = np.asarray(
        [str(record["censorReason"]) for record in selectedWindows], dtype=object
    )
    bootstrapMedianBP = np.asarray(
        diagnostics["bootstrapMedianRadiusBP"], dtype=np.float64
    )
    bootstrapWorkingBP = np.asarray(
        diagnostics["bootstrapWorkingSpanBP"], dtype=np.float64
    )
    if not (
        radiusDistributionBP.size
        == radiusCensored.size
        == censorReasons.size
        == len(selectedWindows)
    ):
        raise RuntimeError(
            "correlation-length window diagnostics have inconsistent lengths"
        )
    radiusMask = np.isfinite(radiusDistributionBP) & (radiusDistributionBP > 0.0)
    medianMask = np.isfinite(bootstrapMedianBP) & (bootstrapMedianBP > 0.0)
    workingMask = np.isfinite(bootstrapWorkingBP) & (bootstrapWorkingBP > 0.0)
    radiusDistributionBP = radiusDistributionBP[radiusMask]
    radiusCensored = radiusCensored[radiusMask]
    censorReasons = censorReasons[radiusMask]
    bootstrapMedianBP = bootstrapMedianBP[medianMask]
    bootstrapWorkingBP = bootstrapWorkingBP[workingMask]
    if (
        pointBP is None
        or pointBP <= 0.0
        or lowerBP is None
        or lowerBP <= 0.0
        or upperBP is None
        or upperBP <= 0.0
        or workingSpanBP is None
        or workingSpanBP <= 0.0
        or survivalBandRegionLower is None
        or survivalBandRegionUpper is None
        or survivalBandRegionLower < 0.0
        or survivalBandRegionUpper > 1.0
        or survivalBandRegionLower >= survivalBandRegionUpper
        or survivalBandJumpClosureCount not in (0, 1)
        or survivalBandJumpClosureCount != int(survivalBandJumpClosureUsed)
        or radiusDistributionBP.size == 0
        or bootstrapMedianBP.size == 0
        or bootstrapWorkingBP.size == 0
    ):
        raise RuntimeError(
            "correlation-length plot requires finite window and bootstrap distributions"
        )
    validCensorReasons = np.isin(censorReasons, ("none", "support", "maxLag"))
    if not np.all(validCensorReasons):
        raise RuntimeError("correlation-length diagnostics contain an unknown censor reason")
    if np.any((~radiusCensored) & (censorReasons != "none")) or np.any(
        radiusCensored & (censorReasons == "none")
    ):
        raise RuntimeError("correlation-length censor flags and reasons disagree")

    uniqueTimes = np.unique(radiusDistributionBP)
    atRisk = int(radiusDistributionBP.size)
    survival = 1.0
    kmTimes = [0.0]
    kmSurvival = [1.0]
    survivalAtTime: dict[float, float] = {}
    for observedTime in uniqueTimes:
        atTime = radiusDistributionBP == observedTime
        eventCount = int(np.sum(atTime & ~radiusCensored))
        if eventCount:
            survival *= 1.0 - float(eventCount) / float(atRisk)
        kmTimes.append(float(observedTime))
        kmSurvival.append(float(survival))
        survivalAtTime[float(observedTime)] = float(survival)
        atRisk -= int(np.sum(atTime))

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)
    windowAx, kmAx, bootstrapAx = list(np.ravel(axes))
    navyBlue = "#003B73"
    burntOrange = "#C65A1E"
    darkBlack = "#050505"
    softGray = "#A9A9A9"
    mutedPurple = "#665191"
    crossedRadii = radiusDistributionBP[~radiusCensored]
    supportCensoredRadii = radiusDistributionBP[censorReasons == "support"]
    capCensoredRadii = radiusDistributionBP[censorReasons == "maxLag"]
    if crossedRadii.size:
        windowAx.hist(
            crossedRadii,
            bins=_histogramBins(crossedRadii),
            color=navyBlue,
            alpha=0.78,
            edgecolor=darkBlack,
            linewidth=0.35,
            label="crossed window",
        )
    if supportCensoredRadii.size:
        windowAx.hist(
            supportCensoredRadii,
            bins=_histogramBins(supportCensoredRadii),
            color=softGray,
            alpha=0.7,
            edgecolor=darkBlack,
            linewidth=0.35,
            hatch="//",
            label="support-censored window",
        )
    if capCensoredRadii.size:
        windowAx.hist(
            capCensoredRadii,
            bins=_histogramBins(capCensoredRadii),
            color=mutedPurple,
            alpha=0.62,
            edgecolor=darkBlack,
            linewidth=0.35,
            hatch="xx",
            label="maximum-lag-censored window",
        )
    windowAx.set_title("Observed Window Times", color=darkBlack)
    windowAx.set_xlabel("event or censor time (bp)", color=darkBlack)
    windowAx.set_ylabel("windows", color=darkBlack)
    windowAx.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    windowAx.legend(loc="best", fontsize=8, frameon=False)

    kmAx.step(kmTimes, kmSurvival, where="post", color=navyBlue, linewidth=1.8)
    for reason, color, marker, label in (
        ("support", softGray, "x", "support censor"),
        ("maxLag", mutedPurple, "+", "maximum-lag censor"),
    ):
        reasonTimes = radiusDistributionBP[censorReasons == reason]
        if reasonTimes.size:
            kmAx.scatter(
                reasonTimes,
                [survivalAtTime[float(value)] for value in reasonTimes],
                color=color,
                marker=marker,
                s=26,
                linewidths=1.1,
                label=label,
                zorder=3,
            )
    kmAx.axvspan(
        float(lowerBP),
        float(upperBP),
        color=softGray,
        alpha=0.28,
        label=(
            "95% KM-band median interval "
            f"({survivalBandRegionLower:.0%}–{survivalBandRegionUpper:.0%} region)"
        ),
    )
    kmAx.axvline(
        float(pointBP),
        color=burntOrange,
        linewidth=1.8,
        linestyle="--",
        label="KM median radius",
    )
    kmAx.axvline(
        float(workingSpanBP),
        color=darkBlack,
        linewidth=1.4,
        linestyle=":",
        label=f"KM Q{100.0 * float(workingQuantile or 0.0):.0f} working span",
    )
    kmAx.set_ylim(0.0, 1.03)
    kmAx.set_title("Kaplan–Meier Survival", color=darkBlack)
    kmAx.set_xlabel("radius (bp)", color=darkBlack)
    kmAx.set_ylabel("survival probability", color=darkBlack)
    kmAx.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    kmAx.legend(loc="best", fontsize=7, frameon=False)

    bootstrapAx.hist(
        bootstrapMedianBP,
        bins=_histogramBins(bootstrapMedianBP),
        color=navyBlue,
        alpha=0.72,
        edgecolor=darkBlack,
        linewidth=0.35,
        label="bootstrap median radius",
    )
    bootstrapAx.hist(
        bootstrapWorkingBP,
        bins=_histogramBins(bootstrapWorkingBP),
        color=burntOrange,
        alpha=0.58,
        edgecolor=darkBlack,
        linewidth=0.35,
        label="bootstrap working span",
    )
    bootstrapAx.set_title("Dependent Bootstrap", color=darkBlack)
    bootstrapAx.set_xlabel("radius or span (bp)", color=darkBlack)
    bootstrapAx.set_ylabel("draws", color=darkBlack)
    bootstrapAx.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    bootstrapAx.legend(loc="best", fontsize=8, frameon=False)

    selectedWindowCount = _summaryInt(diagnostics.get("selectedWindowCount"))
    selectedAutosomeCount = _summaryInt(diagnostics.get("selectedAutosomeCount"))
    censorFraction = numberValue("censor_fraction", "censorFraction")
    supportCensoredCount = _summaryInt(diagnostics.get("supportCensoredWindowCount"))
    capCensoredCount = _summaryInt(diagnostics.get("capCensoredWindowCount"))
    unresolvedPeriodFraction = numberValue("unresolvedPeriodFraction")
    blockLength = _summaryInt(diagnostics.get("bootstrapBlockLengthWindows"))
    captionParts = [
        f"windows={selectedWindowCount if selectedWindowCount is not None else 'NA'}",
        f"autosomes={selectedAutosomeCount if selectedAutosomeCount is not None else 'NA'}",
        f"support-censored={supportCensoredCount if supportCensoredCount is not None else 'NA'}",
        f"cap-censored={capCensoredCount if capCensoredCount is not None else 'NA'}",
        f"block={blockLength if blockLength is not None else 'NA'} windows",
        (
            f"band-region={survivalBandRegionLower:.2f}–"
            f"{survivalBandRegionUpper:.2f}"
        ),
        (
            "jump-closure=used"
            if survivalBandJumpClosureUsed
            else "jump-closure=not-used"
        ),
    ]
    if censorFraction is not None:
        captionParts.append(f"right-censored={censorFraction:.3g}")
    if contextBP is not None:
        captionParts.append(f"context={int(round(float(contextBP)))} bp")
    if unresolvedPeriodFraction is not None:
        captionParts.append(f"unresolved-period={unresolvedPeriodFraction:.3g}")
    fig.suptitle(
        "Correlation-Length Inference\n" + "   ".join(captionParts),
        color=darkBlack,
        fontsize=8,
    )

    fig.savefig(str(path), dpi=int(dpi))
    plt.close(fig)
    logging_utils.log_file_written(
        logger,
        event="artifact.correlation_length_plot",
        path=str(path),
        fields=(
            ("format", "png"),
            ("dpi", int(dpi)),
            ("confidenceIntervalMethod", diagnostics["confidenceIntervalMethod"]),
            ("survivalBandRegionLower", diagnostics["survivalBandRegionLower"]),
            ("survivalBandRegionUpper", diagnostics["survivalBandRegionUpper"]),
            (
                "survivalBandJumpClosureUsed",
                diagnostics["survivalBandJumpClosureUsed"],
            ),
            (
                "survivalBandJumpClosureCount",
                diagnostics["survivalBandJumpClosureCount"],
            ),
            ("bootstrapBlockLengthWindows", diagnostics["bootstrapBlockLengthWindows"]),
        ),
    )
    return True


def _normalizedRemainingObjectiveGap(objective: Sequence[float]) -> np.ndarray:
    objectiveArray = np.asarray(objective, dtype=np.float64)
    if objectiveArray.size == 0:
        return np.asarray([], dtype=np.float64)
    finalObjective = float(objectiveArray[-1])
    initialGap = float(objectiveArray[0] - finalObjective)
    if not np.isfinite(initialGap) or abs(initialGap) <= 1.0e-12:
        return np.zeros_like(objectiveArray, dtype=np.float64)
    return (objectiveArray - finalObjective) / initialGap


def _plotGenomeOptimizationPathLog(
    rows: Sequence[Mapping[str, Any]],
    path: str,
    *,
    dpi: int = 400,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "outputParams.plotOptimizationPath=True but matplotlib is not installed; "
            "wrote the genome-wide optimization .log only."
        )
        return False

    frame = _coerceOptimizationPathFrame(rows)
    if frame.empty:
        logger.warning(
            "genomeOptimizationPath.plot skipped because no trace rows were recorded."
        )
        return False
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    innerFrame = frame[
        (frame["path_level"] == "inner") & (frame["phase"] == "post_process_noise_fit")
    ].copy()
    if innerFrame.empty:
        logger.warning(
            "genomeOptimizationPath.plot skipped because no final-stage inner trace "
            "rows were recorded."
        )
        return False

    navyBlue = "#003B73"
    burntOrange = "#C65A1E"
    darkBlack = "#050505"
    gridColor = "#D8D8D8"
    bandColor = "#F2B078"
    chromosomes = [str(value) for value in innerFrame["chromosome"].dropna().unique()]
    if len(chromosomes) < 2:
        logger.info(
            "genomeOptimizationPath.plot skipped because only one chromosome was recorded."
        )
        return False

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        constrained_layout=True,
    )
    rawAx, normAx = axes
    interpolationGrid = np.linspace(0.0, 1.0, 101)
    normalizedCurves: List[np.ndarray] = []

    firstTrace = True
    firstFinal = True
    for chromosome in chromosomes:
        chromFrame = (
            innerFrame[innerFrame["chromosome"].astype(str) == chromosome]
            .sort_values("record_order")
            .reset_index(drop=True)
        )
        if chromFrame.empty:
            continue
        plotOrder = np.arange(1, len(chromFrame) + 1, dtype=np.float64)
        objective = chromFrame["objective_value"].to_numpy(dtype=np.float64)
        improvement = objective[0] - objective
        remainingGap = _normalizedRemainingObjectiveGap(objective)
        if len(plotOrder) == 1:
            normalizedX = np.array([0.0], dtype=np.float64)
        else:
            normalizedX = (plotOrder - 1.0) / float(len(plotOrder) - 1)
        if len(normalizedX) == 1:
            interpolated = np.full_like(
                interpolationGrid,
                float(remainingGap[0]),
                dtype=np.float64,
            )
        else:
            interpolated = np.interp(
                interpolationGrid,
                normalizedX,
                remainingGap,
            )
        normalizedCurves.append(interpolated)

        label = "chromosome traces" if firstTrace else None
        rawAx.plot(
            plotOrder,
            improvement,
            color=navyBlue,
            alpha=0.24,
            linewidth=1.0,
            label=label,
        )
        normAx.plot(
            normalizedX,
            remainingGap,
            color=navyBlue,
            alpha=0.24,
            linewidth=1.0,
            label=label,
        )
        finalLabel = "final solution" if firstFinal else None
        rawAx.scatter(
            [plotOrder[-1]],
            [improvement[-1]],
            s=18,
            marker="o",
            linewidths=0.6,
            edgecolors=darkBlack,
            c=burntOrange,
            alpha=0.84,
            zorder=4,
            label=finalLabel,
        )
        normAx.scatter(
            [normalizedX[-1]],
            [remainingGap[-1]],
            s=18,
            marker="o",
            linewidths=0.6,
            edgecolors=darkBlack,
            c=burntOrange,
            alpha=0.84,
            zorder=4,
            label=finalLabel,
        )
        firstTrace = False
        firstFinal = False

    if normalizedCurves:
        curveMat = np.vstack(normalizedCurves)
        q25 = np.nanquantile(curveMat, 0.25, axis=0)
        median = np.nanmedian(curveMat, axis=0)
        q75 = np.nanquantile(curveMat, 0.75, axis=0)
        normAx.fill_between(
            interpolationGrid,
            q25,
            q75,
            color=bandColor,
            alpha=0.22,
            linewidth=0,
            label="IQR",
        )
        normAx.plot(
            interpolationGrid,
            median,
            color=burntOrange,
            alpha=0.98,
            linewidth=2.2,
            label="median",
            zorder=5,
        )

    fig.suptitle(
        f"Consenrich ECM Genome-wide Objective Path ({len(chromosomes)} chromosomes)",
        color=darkBlack,
    )
    rawAx.set_title("Raw iteration scale", color=darkBlack)
    rawAx.set_xlabel("Inner iterations", color=darkBlack)
    rawAx.set_ylabel("NLL improvement", color=darkBlack)
    normAx.set_title("Normalized chromosome scale", color=darkBlack)
    normAx.set_xlabel("Fraction of chromosome iterations", color=darkBlack)
    normAx.set_ylabel("Remaining NLL gap to final (fraction of start)", color=darkBlack)
    normAx.axhline(0.0, color=darkBlack, linewidth=0.8, alpha=0.5)
    for axis in (rawAx, normAx):
        axis.grid(True, color=gridColor, linewidth=0.7, alpha=0.75)
        handles, _labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=8, frameon=False)

    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    logger.info("genomeOptimizationPath.output wrote %s dpi=%d", path, int(dpi))
    return True


def _precisionDiagnosticsFrame(
    *,
    chromosome: str,
    intervals: np.ndarray,
    intervalSizeBP: int,
    matrixMunc: np.ndarray,
    pad: float,
    precisionDiagnostics: Mapping[str, Any],
) -> pd.DataFrame:
    lambdaExp = precisionDiagnostics.get("lambdaExp")
    processPrecExp = precisionDiagnostics.get("processPrecExp")
    matrixQ0 = precisionDiagnostics.get("matrixQ0")
    outputTracks = precisionDiagnostics.get("outputTracks", {})
    if not isinstance(outputTracks, Mapping):
        outputTracks = {}
    n = int(len(intervals))
    if n <= 0:
        logger.warning(
            "precisionDiagnostics.output skipped %s because no intervals were available.",
            chromosome,
        )
        return pd.DataFrame()

    lambdaArr = np.ones(n, dtype=np.float64)
    if lambdaExp is not None:
        lambdaArr = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
        if lambdaArr.size != n:
            raise RuntimeError(
                f"lambdaExp length mismatch for {chromosome}: expected {n}, got {lambdaArr.size}"
            )
    if not np.all(np.isfinite(lambdaArr)):
        raise RuntimeError(f"lambdaExp contains non-finite values for {chromosome}")
    lambdaArr = np.maximum(lambdaArr, np.finfo(np.float64).tiny)

    kappaArr = np.ones(n, dtype=np.float64)
    if processPrecExp is not None:
        kappaArr = np.asarray(processPrecExp, dtype=np.float64).reshape(-1)
        if kappaArr.size != n:
            raise RuntimeError(
                f"processPrecExp length mismatch for {chromosome}: expected {n}, got {kappaArr.size}"
            )
    if not np.all(np.isfinite(kappaArr)):
        raise RuntimeError(
            f"processPrecExp contains non-finite values for {chromosome}"
        )
    kappaArr = np.maximum(kappaArr, np.finfo(np.float64).tiny)

    q = np.asarray(matrixQ0, dtype=np.float64)
    if q.shape != (2, 2):
        raise RuntimeError(
            f"matrixQ0 shape mismatch for {chromosome}: expected (2, 2), got {q.shape}"
        )

    def _coerceOutputTrack(trackName: str) -> np.ndarray | None:
        sourceName = trackName
        if sourceName not in outputTracks:
            for fallbackName in _OUTPUT_TRACK_FALLBACK_NAMES.get(trackName, ()):
                if fallbackName in outputTracks:
                    sourceName = fallbackName
                    break
            else:
                return None
        arr = np.asarray(outputTracks[sourceName], dtype=np.float64).reshape(-1)
        if arr.size != n:
            raise RuntimeError(
                f"diagnostic output track {sourceName!r} length mismatch for "
                f"{chromosome}: expected {n}, got {arr.size}"
            )
        return arr

    processQPolicy = str(precisionDiagnostics.get("process_q_policy") or "")
    apnEnabled = bool(
        precisionDiagnostics.get(
            "ECM_useAPN",
            processQPolicy == "adaptive_process_noise",
        )
    )
    processPrecisionRequested = bool(
        precisionDiagnostics.get(
            "process_precision_reweighting_requested",
            processPrecExp is not None,
        )
    )
    processPrecisionEffective = bool(
        precisionDiagnostics.get(
            "process_precision_reweighting_effective",
            processPrecExp is not None,
        )
    )
    processPrecisionDisabledByAPN = bool(
        precisionDiagnostics.get(
            "process_precision_reweighting_disabled_by_apn",
            processPrecisionRequested and apnEnabled and not processPrecisionEffective,
        )
    )
    if not processQPolicy:
        if apnEnabled:
            processQPolicy = "adaptive_process_noise"
        elif processPrecisionEffective and processPrecExp is not None:
            processQPolicy = "student_t_kappa"
        else:
            processQPolicy = "base"

    stateModel = str(precisionDiagnostics.get("state_model") or "").strip().lower()
    baseQ00 = _coerceOutputTrack("preKappaQLevel")
    if baseQ00 is None:
        baseQ00 = np.full(n, float(q[0, 0]), dtype=np.float64)
    baseQ11 = _coerceOutputTrack("preKappaQTrend")
    if baseQ11 is None:
        baseQ11 = np.full(
            n,
            0.0 if stateModel == "level" else float(q[1, 1]),
            dtype=np.float64,
        )

    effectiveQ00 = _coerceOutputTrack("effectiveQLevel")
    effectiveQ11 = _coerceOutputTrack("effectiveQTrend")
    usedEffectiveQFallback = effectiveQ00 is None or effectiveQ11 is None
    if effectiveQ00 is None:
        effectiveQ00 = (
            baseQ00 / kappaArr
            if processPrecisionEffective and processPrecExp is not None
            else baseQ00.copy()
        )
    if effectiveQ11 is None:
        effectiveQ11 = (
            baseQ11 / kappaArr
            if processPrecisionEffective and processPrecExp is not None
            else baseQ11.copy()
        )
    if apnEnabled and usedEffectiveQFallback:
        logger.warning(
            "precisionDiagnostics.output %s has APN enabled but no effective process-Q "
            "tracks; falling back to base Q for missing precision TSV columns.",
            chromosome,
        )
    q00 = effectiveQ00
    q11 = effectiveQ11

    munc = np.asarray(matrixMunc)
    if munc.ndim != 2 or munc.shape[1] != n:
        raise RuntimeError(
            f"matrixMunc shape mismatch for {chromosome}: expected second dimension {n}, got {munc.shape}"
        )
    medianDiagR = np.asarray(np.nanmedian(munc, axis=0), dtype=np.float64)
    medianDiagR += float(pad)
    medianEffectiveDiagR = medianDiagR / lambdaArr
    starts = np.asarray(intervals, dtype=np.int64).reshape(-1)

    frame = pd.DataFrame(
        {
            "Chromosome": chromosome,
            "Start": starts,
            "End": starts + int(intervalSizeBP),
            "Interval": np.arange(1, n + 1, dtype=np.int64),
            "lambda": lambdaArr,
            "kappa": kappaArr,
            "Q00": q00,
            "Q11": q11,
            "baseQ00": baseQ00,
            "baseQ11": baseQ11,
            "effectiveQ00": effectiveQ00,
            "effectiveQ11": effectiveQ11,
            "process_q_policy": processQPolicy,
            "apn_enabled": apnEnabled,
            "process_precision_reweighting_requested": processPrecisionRequested,
            "process_precision_reweighting_effective": processPrecisionEffective,
            "process_precision_reweighting_disabled_by_apn": (
                processPrecisionDisabledByAPN
            ),
            "median_diag_R": medianDiagR,
            "median_effective_diag_R": medianEffectiveDiagR,
        }
    )
    return frame


def _appendMuncLambdaDiagnostics(
    frame: pd.DataFrame,
    path: Path,
    *,
    chromosome: str,
    precisionDiagnostics: Mapping[str, Any],
    detail: str = constants.OUTPUT_DEFAULT_PRECISION_DIAGNOSTIC_DETAIL,
    maxRowsPerChromosome: int = (
        constants.OUTPUT_DEFAULT_MAX_PRECISION_DIAGNOSTIC_ROWS_PER_CHROMOSOME
    ),
) -> int:
    if frame.empty:
        return 0
    outFrame, rowPositions, sampling = _selectPrecisionDiagnosticIntervalRows(
        frame,
        detail=detail,
        maxRowsPerChromosome=maxRowsPerChromosome,
    )
    lambdaValues = pd.to_numeric(outFrame["lambda"]).to_numpy(dtype=np.float64)
    lambdaMin = _summaryNumber(
        precisionDiagnostics.get("observationPrecisionMultiplierMin")
    )
    lambdaMax = _summaryNumber(
        precisionDiagnostics.get("observationPrecisionMultiplierMax")
    )
    lambdaLowerHit = (
        np.zeros(lambdaValues.shape, dtype=bool)
        if lambdaMin is None
        else lambdaValues <= (float(lambdaMin) * (1.0 + 1.0e-6))
    )
    lambdaUpperHit = (
        np.zeros(lambdaValues.shape, dtype=bool)
        if lambdaMax is None
        else lambdaValues >= (float(lambdaMax) * (1.0 - 1.0e-6))
    )
    out = pd.DataFrame(
        {
            "record_type": "interval",
            "event": "munc_lambda.interval",
            "chromosome": outFrame["Chromosome"],
            "start": outFrame["Start"],
            "end": outFrame["End"],
            "interval": outFrame["Interval"],
            "lambda": outFrame["lambda"],
            "lambda_lower_bound_hit": lambdaLowerHit,
            "lambda_upper_bound_hit": lambdaUpperHit,
            "median_diag_R": outFrame["median_diag_R"],
            "median_effective_diag_R": outFrame["median_effective_diag_R"],
        }
    )
    outputTracks = precisionDiagnostics.get("outputTracks", {})
    if isinstance(outputTracks, Mapping):
        for name in ("muncTrace", "sumGain0", "sumGain1"):
            if name in outputTracks:
                track = np.asarray(outputTracks[name], dtype=np.float64).reshape(-1)
                if track.shape[0] == len(frame):
                    out[name] = track[rowPositions]
                elif track.shape[0] == len(outFrame):
                    out[name] = track
    rowsWritten = _appendJsonlRecords(path, out)
    _appendMappingDiagnostics(
        path,
        recordType="summary",
        event="munc_lambda.summary",
        chromosome=chromosome,
        values={
            **sampling,
            "rows": int(rowsWritten),
            "lambda_median": float(pd.to_numeric(frame["lambda"]).median()),
            "median_diag_R": float(pd.to_numeric(frame["median_diag_R"]).median()),
            "median_effective_diag_R": float(
                pd.to_numeric(frame["median_effective_diag_R"]).median()
            ),
        },
    )
    return rowsWritten


def _appendProcessKappaDiagnostics(
    frame: pd.DataFrame,
    path: Path,
    *,
    chromosome: str,
    precisionDiagnostics: Mapping[str, Any],
    runDiagnostics: Mapping[str, Any],
    detail: str = constants.OUTPUT_DEFAULT_PRECISION_DIAGNOSTIC_DETAIL,
    maxRowsPerChromosome: int = (
        constants.OUTPUT_DEFAULT_MAX_PRECISION_DIAGNOSTIC_ROWS_PER_CHROMOSOME
    ),
) -> int:
    if frame.empty:
        return 0
    outFrame, rowPositions, sampling = _selectPrecisionDiagnosticIntervalRows(
        frame,
        detail=detail,
        maxRowsPerChromosome=maxRowsPerChromosome,
    )
    kappaValues = pd.to_numeric(outFrame["kappa"]).to_numpy(dtype=np.float64)
    kappaMin = _summaryNumber(precisionDiagnostics.get("processPrecisionMultiplierMin"))
    kappaMax = _summaryNumber(precisionDiagnostics.get("processPrecisionMultiplierMax"))
    kappaLowerHit = (
        np.zeros(kappaValues.shape, dtype=bool)
        if kappaMin is None
        else kappaValues <= (float(kappaMin) * (1.0 + 1.0e-6))
    )
    kappaUpperHit = (
        np.zeros(kappaValues.shape, dtype=bool)
        if kappaMax is None
        else kappaValues >= (float(kappaMax) * (1.0 - 1.0e-6))
    )
    out = pd.DataFrame(
        {
            "record_type": "interval",
            "event": "process_kappa.interval",
            "chromosome": outFrame["Chromosome"],
            "start": outFrame["Start"],
            "end": outFrame["End"],
            "interval": outFrame["Interval"],
            "kappa": outFrame["kappa"],
            "kappa_lower_bound_hit": kappaLowerHit,
            "kappa_upper_bound_hit": kappaUpperHit,
            "process_q_policy": outFrame["process_q_policy"],
            "apn_enabled": outFrame["apn_enabled"],
            "process_precision_reweighting_requested": outFrame[
                "process_precision_reweighting_requested"
            ],
            "process_precision_reweighting_effective": outFrame[
                "process_precision_reweighting_effective"
            ],
            "process_precision_reweighting_disabled_by_apn": outFrame[
                "process_precision_reweighting_disabled_by_apn"
            ],
            "baseQ00": outFrame["baseQ00"],
            "baseQ11": outFrame["baseQ11"],
            "preKappaQLevel": outFrame["baseQ00"],
            "preKappaQTrend": outFrame["baseQ11"],
            "effectiveQLevel": outFrame["effectiveQ00"],
            "effectiveQTrend": outFrame["effectiveQ11"],
        }
    )
    outputTracks = precisionDiagnostics.get("outputTracks", {})
    if isinstance(outputTracks, Mapping) and "processQScale" in outputTracks:
        track = np.asarray(outputTracks["processQScale"], dtype=np.float64).reshape(-1)
        if track.shape[0] == len(frame):
            out["processQScale"] = track[rowPositions]
        elif track.shape[0] == len(outFrame):
            out["processQScale"] = track
    rowsWritten = _appendJsonlRecords(path, out)
    processNoise = runDiagnostics.get("process_noise_calibration")
    if isinstance(processNoise, Mapping):
        _appendMappingDiagnostics(
            path,
            recordType="summary",
            event="process_kappa.process_noise_calibration",
            chromosome=chromosome,
            values=processNoise,
        )
    _appendMappingDiagnostics(
        path,
        recordType="summary",
        event="process_kappa.summary",
        chromosome=chromosome,
        values={
            **sampling,
            "rows": int(rowsWritten),
            "kappa_median": float(pd.to_numeric(frame["kappa"]).median()),
            "effective_q_level_median": float(
                pd.to_numeric(frame["effectiveQ00"]).median()
            ),
            "effective_q_trend_median": float(
                pd.to_numeric(frame["effectiveQ11"]).median()
            ),
        },
    )
    return rowsWritten


def _appendConvergenceDiagnostics(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> int:
    if not rows:
        return 0
    records = ({"record_type": "trace", **dict(row)} for row in rows)
    rowsWritten = _appendJsonlRecords(path, records)
    core._logEvent(
        "artifact.convergence",
        (("path", str(path)), ("rows", int(rowsWritten))),
        logger_=logger,
    )
    return rowsWritten


def _summaryNumber(value: Any) -> float | int | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        valueFloat = float(value)
    except (TypeError, ValueError):
        return None
    return valueFloat if np.isfinite(valueFloat) else None


def _summaryInt(value: Any) -> int | None:
    valueNumber = _summaryNumber(value)
    if valueNumber is None:
        return None
    return int(valueNumber)


def _summaryMapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stateShrinkageOutputTracks(
    writeStateShrinkageTracks: bool,
) -> list[tuple[str, str]]:
    tracks = [
        ("stateShrunk", "stateShrunk"),
        ("stateShrunkUncertainty", "stateShrunkUncertainty"),
    ]
    if writeStateShrinkageTracks:
        tracks.extend(
            [
                ("stateSpikeProp", "stateSpikeProp"),
            ]
        )
    return tracks


def _stateShrinkageSummaryFields(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stateShrinkage = _summaryMapping(metadata)
    fields = {
        "state_shrinkage_model": stateShrinkage.get("model"),
        "state_shrinkage_slab_family": stateShrinkage.get("slab_family"),
        "state_shrinkage_scope": stateShrinkage.get("scope"),
        "state_shrinkage_chunk_count": _summaryInt(
            stateShrinkage.get("chunk_count")
        ),
        "state_shrinkage_interval_count": _summaryInt(
            stateShrinkage.get("interval_count")
        ),
        "state_shrinkage_finite_count": _summaryInt(
            stateShrinkage.get("finite_count")
        ),
        "state_shrinkage_effective_block_count": _summaryNumber(
            stateShrinkage.get("effective_block_count")
        ),
        "state_shrinkage_block_size_intervals": _summaryInt(
            stateShrinkage.get("block_size_intervals")
        ),
        "state_shrinkage_prior_spike_prop": _summaryNumber(
            stateShrinkage.get("prior_spike_prop")
        ),
        "state_shrinkage_spike_odds_multiplier": _summaryNumber(
            stateShrinkage.get("spike_odds_multiplier")
        ),
        "state_shrinkage_effective_prior_spike_prop": _summaryNumber(
            stateShrinkage.get("effective_prior_spike_prop")
        ),
        "state_shrinkage_prior_scale": _summaryNumber(
            stateShrinkage.get("prior_scale")
        ),
        "state_shrinkage_prior_variance": _summaryNumber(
            stateShrinkage.get("prior_variance")
        ),
        "state_shrinkage_prior_variance_defined": _jsonDiagnosticValue(
            stateShrinkage.get("prior_variance_defined")
        ),
        "state_shrinkage_slab_count": _summaryInt(
            stateShrinkage.get("slab_count")
        ),
        "state_shrinkage_slab_weight": _jsonDiagnosticValue(
            stateShrinkage.get("slab_weight")
        ),
        "state_shrinkage_slab_variance": _jsonDiagnosticValue(
            stateShrinkage.get("slab_variance")
        ),
        "state_shrinkage_slab_multiplier": _jsonDiagnosticValue(
            stateShrinkage.get("slab_multiplier")
        ),
        "state_shrinkage_component_weights": _jsonDiagnosticValue(
            stateShrinkage.get("component_weights")
        ),
        "state_shrinkage_student_t_df": _summaryNumber(
            stateShrinkage.get("student_t_df")
        ),
        "state_shrinkage_student_t_scale": _summaryNumber(
            stateShrinkage.get("student_t_scale")
        ),
        "state_shrinkage_student_t_quadrature_order": _summaryInt(
            stateShrinkage.get("student_t_quadrature_order")
        ),
        "state_shrinkage_student_t_quadrature_alpha": _summaryNumber(
            stateShrinkage.get("student_t_quadrature_alpha")
        ),
        "state_shrinkage_estimated_prior_spike_prop": _jsonDiagnosticValue(
            stateShrinkage.get("estimated_prior_spike_prop")
        ),
        "state_shrinkage_estimated_prior_scale": _jsonDiagnosticValue(
            stateShrinkage.get("estimated_prior_scale")
        ),
        "state_shrinkage_estimated_slab_weights": _jsonDiagnosticValue(
            stateShrinkage.get("estimated_slab_weights")
        ),
        "state_shrinkage_estimated_slab_scales": _jsonDiagnosticValue(
            stateShrinkage.get("estimated_slab_scales")
        ),
        "state_shrinkage_iterations": _summaryInt(stateShrinkage.get("iterations")),
        "state_shrinkage_converged": _jsonDiagnosticValue(
            stateShrinkage.get("converged")
        ),
        "state_shrinkage_log_likelihood": _summaryNumber(
            stateShrinkage.get("log_likelihood")
        ),
        "state_shrinkage_state_abs_median_before": _summaryNumber(
            stateShrinkage.get("state_abs_median_before")
        ),
        "state_shrinkage_state_abs_median_after": _summaryNumber(
            stateShrinkage.get("state_abs_median_after")
        ),
        "state_shrinkage_spike_prop_median": _summaryNumber(
            stateShrinkage.get("spike_prop_median")
        ),
        "state_shrinkage_posterior_sd_median": _summaryNumber(
            stateShrinkage.get("posterior_sd_median")
        ),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _deleteBlockFactorDistributionFromArray(factor: Any) -> dict[str, Any]:
    factorArr = np.asarray(factor, dtype=np.float64).reshape(-1)
    if factorArr.size == 0:
        raise RuntimeError("delete-block factor array is empty")
    if not np.all(np.isfinite(factorArr)):
        raise RuntimeError("delete-block factor array contains non-finite values")
    if np.any(factorArr <= 0.0):
        raise RuntimeError("delete-block factor array contains non-positive values")
    factorMedian = float(np.median(factorArr))
    sdFactorArr = np.sqrt(factorArr)
    sdFactorMedian = float(np.median(sdFactorArr))
    factorQuantileMethod = "linear"
    factorQ05, factorQ95 = np.quantile(
        factorArr,
        [0.05, 0.95],
        method=factorQuantileMethod,
    )
    sdFactorQ05, sdFactorQ95 = np.quantile(
        sdFactorArr,
        [0.05, 0.95],
        method=factorQuantileMethod,
    )
    return {
        "count": int(factorArr.size),
        "median": factorMedian,
        "unscaled_mad": float(np.median(np.abs(factorArr - factorMedian))),
        "q05": float(factorQ05),
        "q95": float(factorQ95),
        "min": float(np.min(factorArr)),
        "max": float(np.max(factorArr)),
        "sd_multiplier_median": sdFactorMedian,
        "sd_multiplier_unscaled_mad": float(
            np.median(np.abs(sdFactorArr - sdFactorMedian))
        ),
        "sd_multiplier_q05": float(sdFactorQ05),
        "sd_multiplier_q95": float(sdFactorQ95),
        "sd_multiplier_min": float(np.min(sdFactorArr)),
        "sd_multiplier_max": float(np.max(sdFactorArr)),
        "quantile_method": factorQuantileMethod,
    }


def _deleteBlockFactorSummaryFields(
    calibrationModel: Mapping[str, Any] | None,
) -> dict[str, Any]:
    calibration = _summaryMapping(calibrationModel)
    targetCalibration = _summaryMapping(calibration.get("target_calibration"))
    replicateDependence = _summaryMapping(calibration.get("replicate_dependence"))
    factorDistribution = _summaryMapping(
        calibration.get("delete_block_factor_distribution")
    )
    factorModel = calibration.get("factor_model")
    globalFactor = (
        _summaryNumber(calibration.get("global_factor"))
        if factorModel == "global"
        else None
    )
    return {
        "delete_block_global_factor": globalFactor,
        "delete_block_factor_model": factorModel,
        "delete_block_variance_multiplier_global": globalFactor,
        "delete_block_variance_multiplier_min": _summaryNumber(
            factorDistribution.get("min")
        ),
        "delete_block_variance_multiplier_q05": _summaryNumber(
            factorDistribution.get("q05")
        ),
        "delete_block_variance_multiplier_median": _summaryNumber(
            factorDistribution.get("median")
        ),
        "delete_block_variance_multiplier_mad": _summaryNumber(
            factorDistribution.get("unscaled_mad")
        ),
        "delete_block_variance_multiplier_q95": _summaryNumber(
            factorDistribution.get("q95")
        ),
        "delete_block_variance_multiplier_max": _summaryNumber(
            factorDistribution.get("max")
        ),
        "delete_block_sd_multiplier_median": _summaryNumber(
            factorDistribution.get("sd_multiplier_median")
        ),
        "delete_block_track_sd_scale": _summaryNumber(
            targetCalibration.get("uncertainty_track_scale")
        ),
        "delete_block_rows_valid": _summaryInt(calibration.get("rows_valid")),
        "delete_block_rows_fit": _summaryInt(calibration.get("rows_fit")),
        "delete_block_deletion_probability": _summaryNumber(
            calibration.get("delete_block_deletion_probability")
        ),
        "delete_block_deleted_blocks": _summaryInt(
            calibration.get("delete_block_deleted_blocks")
        ),
        "delete_block_deleted_replicate_block_total": _summaryInt(
            calibration.get("delete_block_deleted_replicate_block_total")
        ),
        "delete_block_deleted_observation_interval_total": _summaryInt(
            calibration.get("delete_block_deleted_observation_interval_total")
        ),
        "delete_block_replicate_dependence_rho": _summaryNumber(
            replicateDependence.get("rho")
        ),
        "delete_block_replicate_dependence_rho_source": replicateDependence.get(
            "source"
        ),
        "delete_block_information_deff_median": _summaryNumber(
            replicateDependence.get("total_deff_median")
        ),
        "delete_block_scale": _summaryNumber(
            targetCalibration.get("uncertainty_track_scale")
        ),
        "delete_block_scale_reason": targetCalibration.get(
            "uncertainty_track_scale_reason"
        ),
    }


def _countNoiseFloorSummaryFields(
    summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    countNoise = _summaryMapping(summary)
    return {
        "countNoiseFloorEnabled": _jsonDiagnosticValue(countNoise.get("enabled")),
        "countNoiseFloorModel": countNoise.get("model"),
        "countNoiseFloorCountModes": _jsonDiagnosticValue(countNoise.get("countModes")),
        "countNoiseFloorSourceKinds": _jsonDiagnosticValue(
            countNoise.get("sourceKinds")
        ),
        "countNoiseFloorBedgraphSkippedTracks": _summaryInt(
            countNoise.get("bedgraphSkippedTracks")
        ),
        "countNoiseFloorFinite": _summaryInt(countNoise.get("finite")),
        "countNoiseFloorPositive": _summaryInt(countNoise.get("positive")),
        "countNoiseFloorQ05": _summaryNumber(countNoise.get("q05")),
        "countNoiseFloorMedian": _summaryNumber(countNoise.get("median")),
        "countNoiseFloorQ95": _summaryNumber(countNoise.get("q95")),
    }


def _deleteBlockFactorLogFields(
    calibrationModel: Mapping[str, Any] | None,
) -> dict[str, Any]:
    calibration = _summaryMapping(calibrationModel)
    targetCalibration = _summaryMapping(calibration.get("target_calibration"))
    factorDistribution = _summaryMapping(
        calibration.get("delete_block_factor_distribution")
    )
    factorModel = calibration.get("factor_model")
    globalSDMultiplier = None
    if factorModel == "global":
        globalSDMultiplier = _summaryNumber(calibration.get("global_sd_multiplier"))
        if globalSDMultiplier is None:
            globalFactor = _summaryNumber(calibration.get("global_factor"))
            if globalFactor is not None and globalFactor >= 0.0:
                globalSDMultiplier = float(math.sqrt(float(globalFactor)))
    return {
        "delete_block_factor_model": factorModel,
        "delete_block_sd_global": globalSDMultiplier,
        "delete_block_sd_median": _summaryNumber(
            factorDistribution.get("sd_multiplier_median")
        ),
        "delete_block_sd_mad": _summaryNumber(
            factorDistribution.get("sd_multiplier_unscaled_mad")
        ),
        "delete_block_sd_q05": _summaryNumber(
            factorDistribution.get("sd_multiplier_q05")
        ),
        "delete_block_sd_q95": _summaryNumber(
            factorDistribution.get("sd_multiplier_q95")
        ),
        "delete_block_track_sd_scale": _summaryNumber(
            targetCalibration.get("uncertainty_track_scale")
        ),
        "delete_block_deletion_probability": _summaryNumber(
            calibration.get("delete_block_deletion_probability")
        ),
        "delete_block_deleted_blocks": _summaryInt(
            calibration.get("delete_block_deleted_blocks")
        ),
        "delete_block_deleted_observation_interval_total": _summaryInt(
            calibration.get("delete_block_deleted_observation_interval_total")
        ),
    }


def _runSummaryRow(
    *,
    chromosome: str,
    intervals: int,
    samples: int,
    elapsedSeconds: float,
    outputTrackCount: int,
    runDiagnostics: Mapping[str, Any],
    stateRoughness: Mapping[str, Any],
    calibrationModel: Mapping[str, Any] | None,
    diagnosticLogPaths: DiagnosticLogPaths,
    countNoiseFloorSummary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    precisionHits = _summaryMapping(
        runDiagnostics.get("precision_reweighting_boundary_hits")
    )
    obsHits = _summaryMapping(precisionHits.get("observation"))
    procHits = _summaryMapping(precisionHits.get("process"))
    processQ = _summaryMapping(runDiagnostics.get("process_q_diagnostics"))
    observationRTrace = _summaryMapping(runDiagnostics.get("observation_r_trace"))
    processNoise = _summaryMapping(runDiagnostics.get("process_noise_calibration"))
    deleteBlockFields = _deleteBlockFactorSummaryFields(calibrationModel)
    countNoiseFloorFields = _countNoiseFloorSummaryFields(countNoiseFloorSummary)
    return {
        "record_type": "chromosome",
        "chromosome": chromosome,
        "intervals": int(intervals),
        "samples": int(samples),
        "elapsed_seconds": float(elapsedSeconds),
        "output_track_count": int(outputTrackCount),
        "final_nll": _summaryNumber(runDiagnostics.get("final_nll")),
        "final_forward_nis": _summaryNumber(runDiagnostics.get("final_forward_nis")),
        "process_q_policy": runDiagnostics.get("process_q_policy"),
        "process_q_trace_min": _summaryNumber(processQ.get("effectiveQTraceMin")),
        "process_q_trace_median": _summaryNumber(processQ.get("effectiveQTraceMedian")),
        "process_q_trace_max": _summaryNumber(processQ.get("effectiveQTraceMax")),
        "observation_r_trace_min": _summaryNumber(observationRTrace.get("min")),
        "observation_r_trace_median": _summaryNumber(observationRTrace.get("median")),
        "observation_r_trace_max": _summaryNumber(observationRTrace.get("max")),
        "process_noise_status": processNoise.get("processNoiseCalibrationStatus"),
        "process_noise_reason": processNoise.get("processNoiseCalibrationReason"),
        **countNoiseFloorFields,
        "lambda_lower_bound_hits": _summaryInt(obsHits.get("lower")),
        "lambda_upper_bound_hits": _summaryInt(obsHits.get("upper")),
        "kappa_lower_bound_hits": _summaryInt(procHits.get("lower")),
        "kappa_upper_bound_hits": _summaryInt(procHits.get("upper")),
        "state_roughness_mean_abs_diff": _summaryNumber(
            stateRoughness.get("overall_mean_abs_diff")
        ),
        "state_roughness_block_median": _summaryNumber(
            stateRoughness.get("block_mean_abs_diff_median")
        ),
        "state_roughness_block_q90": _summaryNumber(
            stateRoughness.get("block_mean_abs_diff_q90")
        ),
        **deleteBlockFields,
        "precision_log": str(diagnosticLogPaths.precision),
        "convergence_log": str(diagnosticLogPaths.convergence),
        "delete_block_calibration_log": str(
            diagnosticLogPaths.delete_block_calibration
        ),
    }


def _genomeRunSummaryRow(
    chromosomeRows: Sequence[Mapping[str, Any]],
    *,
    elapsedSeconds: float,
    diagnosticLogPaths: DiagnosticLogPaths,
) -> dict[str, Any]:
    intervals = sum(int(row.get("intervals") or 0) for row in chromosomeRows)
    samples = max((int(row.get("samples") or 0) for row in chromosomeRows), default=0)
    outputTrackCount = max(
        (int(row.get("output_track_count") or 0) for row in chromosomeRows),
        default=0,
    )
    return {
        "record_type": "genome",
        "chromosome": "genome",
        "intervals": int(intervals),
        "samples": int(samples),
        "elapsed_seconds": float(elapsedSeconds),
        "output_track_count": int(outputTrackCount),
        "precision_log": str(diagnosticLogPaths.precision),
        "convergence_log": str(diagnosticLogPaths.convergence),
        "delete_block_calibration_log": str(
            diagnosticLogPaths.delete_block_calibration
        ),
    }


def _writeRunSummary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    rowsList = list(rows)
    _writeJsonlRecords(path, rowsList)
    logging_utils.log_file_written(
        logger,
        event="artifact.run_summary",
        path=str(path),
        fields=(("rows", int(len(rowsList))),),
    )


def _correlationLengthRow(
    *,
    intervalSizeBP: int,
    pointIntervals: int,
    contextBP: int,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    intervalSize = int(intervalSizeBP)

    def intValue(key: str) -> int | None:
        return _summaryInt(details.get(key))

    def numberValue(key: str) -> float | int | None:
        return _summaryNumber(details.get(key))

    usedChromosomes = details["chromosomesUsed"]
    return {
        "status": str(details["status"]),
        "method": str(details["method"]),
        "random_seed": intValue("randomSeed"),
        "inference_scope": str(details["inferenceScope"]),
        "confidence_interval_method": str(details["confidenceIntervalMethod"]),
        "survival_band_region_lower": numberValue("survivalBandRegionLower"),
        "survival_band_region_upper": numberValue("survivalBandRegionUpper"),
        "survival_band_jump_closure_used": bool(
            details["survivalBandJumpClosureUsed"]
        ),
        "survival_band_jump_closure_count": intValue(
            "survivalBandJumpClosureCount"
        ),
        "confidence_level": numberValue("confidenceLevel"),
        "interval_size_bp": intervalSize,
        "correlation_length_intervals": int(pointIntervals),
        "correlation_length_bp": numberValue("estimateBP"),
        "full_sample_median_radius_bp": numberValue("fullSampleMedianRadiusBP"),
        "lower_bp": numberValue("lowerBP"),
        "upper_bp": numberValue("upperBP"),
        "working_span_bp": numberValue("workingSpanBP"),
        "full_sample_working_span_bp": numberValue("fullSampleWorkingSpanBP"),
        "working_quantile": numberValue("workingQuantile"),
        "context_bp": int(contextBP),
        "chromosomes_used": int(len(usedChromosomes)),
        "candidate_window_count": intValue("candidateWindowCount"),
        "evaluated_candidate_window_count": intValue(
            "evaluatedCandidateWindowCount"
        ),
        "selected_window_count": intValue("selectedWindowCount"),
        "crossed_window_count": intValue("crossedWindowCount"),
        "right_censored_window_count": intValue("rightCensoredWindowCount"),
        "support_censored_window_count": intValue("supportCensoredWindowCount"),
        "cap_censored_window_count": intValue("capCensoredWindowCount"),
        "censor_fraction": numberValue("censorFraction"),
        "censor_time_bp_minimum": numberValue("censorTimeBPMinimum"),
        "censor_time_bp_median": numberValue("censorTimeBPMedian"),
        "censor_time_bp_maximum": numberValue("censorTimeBPMaximum"),
        "input_row_count": intValue("inputRowCount"),
        "unique_row_count": intValue("uniqueRowCount"),
        "duplicate_row_count": intValue("duplicateRowCount"),
        "row_deduplication": str(details["rowDeduplication"]),
        "acf_threshold": numberValue("acfThreshold"),
        "acf_smoothing_bp": intValue("acfSmoothingBP"),
        "acf_smoothing_bins": intValue("acfSmoothingBins"),
        "crossing_persistence_bp": intValue("crossingPersistenceBP"),
        "crossing_persistence_bins": intValue("crossingPersistenceBins"),
        "min_finite_pairs": intValue("minFinitePairs"),
        "min_finite_pair_coverage": numberValue("minFinitePairCoverage"),
        "finite_pair_minimum_used": numberValue("finitePairMinimumUsed"),
        "finite_pair_coverage_minimum_used": numberValue(
            "finitePairCoverageMinimumUsed"
        ),
        "valid_rows_at_crossing_minimum": intValue(
            "validRowsAtCrossingMinimum"
        ),
        "dominant_acf_period_bp_median": numberValue(
            "dominantACFPeriodBPMedian"
        ),
        "unresolved_period_fraction": numberValue("unresolvedPeriodFraction"),
        "oscillation_strength": numberValue("oscillationStrength"),
        "post_crossing_revival": numberValue("postCrossingRevival"),
        "selected_adjacency_count": intValue("selectedAdjacencyCount"),
        "selected_longest_run": intValue("selectedLongestRun"),
        "bootstrap_method": str(details["bootstrapMethod"]),
        "bootstrap_block_length_windows": intValue(
            "bootstrapBlockLengthWindows"
        ),
        "bootstrap_draws_requested": intValue("bootstrapDrawsRequested"),
        "bootstrap_resolved_median_draws": intValue(
            "bootstrapResolvedMedianDraws"
        ),
        "bootstrap_resolved_working_draws": intValue(
            "bootstrapResolvedWorkingDraws"
        ),
        "bootstrap_resolved_joint_draws": intValue(
            "bootstrapResolvedJointDraws"
        ),
    }


def _writeCorrelationLengthSummary(
    row: Mapping[str, Any],
    path: Path,
) -> None:
    frame = pd.DataFrame([dict(row)], columns=CORRELATION_LENGTH_COLUMNS)
    frame.to_csv(path, sep="\t", header=True, index=False, lineterminator="\n")
    logging_utils.log_file_written(
        logger,
        event="artifact.correlation_length",
        path=str(path),
        fields=(("rows", int(len(frame))),),
    )


def _correlationLengthWindowRows(
    details: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in details["selectedWindows"]:
        rows.append(
            {
                "chromosome": str(record["chromosome"]),
                "start_bp": _summaryInt(record.get("startBP")),
                "end_bp": _summaryInt(record.get("endBP")),
                "score": _summaryNumber(record.get("score")),
                "positive_signal_rank": _summaryNumber(
                    record.get("positiveSignalRank")
                ),
                "raw_crossing_lag_bp": _summaryNumber(
                    record.get("rawCrossingLagBP")
                ),
                "censor_lag_bp": _summaryNumber(record.get("censorLagBP")),
                "gaussian_equivalent_radius_bp": _summaryNumber(
                    record.get("gaussianEquivalentRadiusBP")
                ),
                "right_censored": bool(record["rightCensored"]),
                "censor_reason": str(record["censorReason"]),
                "support_cap_lag_bp": _summaryNumber(
                    record.get("supportCapLagBP")
                ),
                "finite_pair_minimum_used": _summaryNumber(
                    record.get("finitePairMinimumUsed")
                ),
                "finite_pair_coverage_minimum_used": _summaryNumber(
                    record.get("finitePairCoverageMinimumUsed")
                ),
                "valid_row_count": _summaryInt(record.get("validRowCount")),
                "valid_rows_at_crossing": _summaryInt(
                    record.get("validRowsAtCrossing")
                ),
                "dominant_acf_period_bp": _summaryNumber(
                    record.get("dominantACFPeriodBP")
                ),
                "oscillation_strength": _summaryNumber(
                    record.get("oscillationStrength")
                ),
                "post_crossing_revival": _summaryNumber(
                    record.get("postCrossingRevival")
                ),
            }
        )
    return rows


def _writeCorrelationLengthWindows(
    details: Mapping[str, Any],
    path: Path,
) -> None:
    frame = pd.DataFrame(
        _correlationLengthWindowRows(details),
        columns=CORRELATION_LENGTH_WINDOW_COLUMNS,
    )
    frame.to_csv(path, sep="\t", header=True, index=False, lineterminator="\n")
    logging_utils.log_file_written(
        logger,
        event="artifact.correlation_length_windows",
        path=str(path),
        fields=(
            ("rows", int(len(frame))),
            ("supportCensoredWindowCount", details["supportCensoredWindowCount"]),
            ("capCensoredWindowCount", details["capCensoredWindowCount"]),
        ),
    )


def _replicateGainSummaryPath(experimentName: str) -> str:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return f"consenrichOutput_{experimentToken}_replicateGains.v{__version__}.jsonl"


def _newReplicateGainAccumulator(replicateCount: int) -> dict[str, np.ndarray]:
    count = max(0, int(replicateCount))
    return {
        "chromosome_count": np.zeros(count, dtype=np.int64),
        "finite_interval_count": np.zeros(count, dtype=np.int64),
        "sum": np.zeros(count, dtype=np.float64),
        "sum_sq": np.zeros(count, dtype=np.float64),
    }


def _coerceGainSummaryVector(
    values: Any,
    replicateCount: int,
    *,
    dtype: Any,
) -> np.ndarray:
    out = np.zeros(int(replicateCount), dtype=dtype)
    if np.issubdtype(np.dtype(dtype), np.floating):
        out.fill(np.nan)
    if values is None:
        return out
    arr = np.asarray(list(values), dtype=dtype).reshape(-1)
    limit = min(int(replicateCount), int(arr.size))
    if limit > 0:
        out[:limit] = arr[:limit]
    return out


def _updateReplicateGainAccumulator(
    accumulator: dict[str, np.ndarray],
    gainSummary: Mapping[str, Any],
) -> int:
    replicateCount = int(accumulator["finite_interval_count"].size)
    means = _coerceGainSummaryVector(
        gainSummary.get("mean"), replicateCount, dtype=np.float64
    )
    sds = _coerceGainSummaryVector(
        gainSummary.get("sd"), replicateCount, dtype=np.float64
    )
    counts = _coerceGainSummaryVector(
        gainSummary.get("count"), replicateCount, dtype=np.int64
    )
    valid = (counts > 0) & np.isfinite(means) & np.isfinite(sds)
    if not np.any(valid):
        return 0
    validCounts = counts[valid].astype(np.float64, copy=False)
    accumulator["chromosome_count"][valid] += 1
    accumulator["finite_interval_count"][valid] += counts[valid]
    accumulator["sum"][valid] += means[valid] * validCounts
    accumulator["sum_sq"][valid] += (
        (sds[valid] ** 2) + (means[valid] ** 2)
    ) * validCounts
    return int(np.count_nonzero(valid))


def _replicateGainSummaryRows(
    treatmentSources: Sequence[core.inputSource],
    accumulator: Mapping[str, np.ndarray],
    *,
    controlSources: Sequence[core.inputSource] | None = None,
) -> list[dict[str, Any]]:
    finiteCounts = np.asarray(accumulator["finite_interval_count"], dtype=np.int64)
    sums = np.asarray(accumulator["sum"], dtype=np.float64)
    sumSqs = np.asarray(accumulator["sum_sq"], dtype=np.float64)
    chromosomeCounts = np.asarray(accumulator["chromosome_count"], dtype=np.int64)
    controlSources_ = list(controlSources or [])
    rows: list[dict[str, Any]] = []
    for i in range(int(finiteCounts.size)):
        source = treatmentSources[i] if i < len(treatmentSources) else None
        if source is None:
            sourceId = f"replicate_{i + 1}"
            treatmentPath = "unknown"
        else:
            sourceId = str(
                source.sampleName
                or os.path.basename(source.path)
                or f"replicate_{i + 1}"
            )
            treatmentPath = str(source.path)
        sampleFile = os.path.basename(treatmentPath)[:7]
        controlPath = str(controlSources_[i].path) if i < len(controlSources_) else None
        count = int(finiteCounts[i])
        if count > 0:
            avg = float(sums[i] / float(count))
            variance = max(float(sumSqs[i] / float(count)) - (avg * avg), 0.0)
            std = float(math.sqrt(variance))
        else:
            avg = None
            std = None
        rows.append(
            {
                "replicate_index": int(i + 1),
                "sample_name": sourceId,
                "sample_file": sampleFile,
                "treatment_path": treatmentPath,
                "control_path": controlPath,
                "chromosome_count": int(chromosomeCounts[i]),
                "finite_interval_count": count,
                "gain_avg": avg,
                "gain_std": std,
            }
        )
    return rows


def _writeReplicateGainSummary(rows: Sequence[Mapping[str, Any]], path: str) -> bool:
    frame = pd.DataFrame(list(rows), columns=GAIN_SUMMARY_COLUMNS)
    finiteIntervals = (
        int(
            pd.to_numeric(frame["finite_interval_count"], errors="coerce")
            .fillna(0)
            .sum()
        )
        if not frame.empty
        else 0
    )
    if frame.empty or finiteIntervals <= 0:
        core._logEvent(
            "artifact.replicate_gains.skipped",
            (("path", path), ("reason", "no_finite_gains")),
            logger_=logger,
            level=logging.WARNING,
        )
        return False
    _writeJsonlRecords(path, frame, GAIN_SUMMARY_COLUMNS)
    core._logEvent(
        "artifact.replicate_gains",
        (
            ("path", path),
            ("rows", int(len(frame))),
            ("finite_intervals", finiteIntervals),
        ),
        logger_=logger,
    )
    return True


def _getProcessNoiseWarmupOuterPasses(processArgs: Any) -> int:
    return max(
        1,
        int(
            getattr(
                processArgs,
                "processNoiseWarmupOuterPasses",
                constants.PROCESS_DEFAULT_WARMUP_OUTER_PASSES,
            )
        ),
    )


def _coreRunConsenrichSupports(parameterName: str) -> bool:
    try:
        return parameterName in inspect.signature(core.runConsenrich).parameters
    except (TypeError, ValueError):
        return False


def _configureCoreProcessNoiseWarmupDefaults(processArgs: Any) -> int:
    warmupOuterPasses = _getProcessNoiseWarmupOuterPasses(processArgs)
    if not _coreRunConsenrichSupports("processNoiseWarmupOuterPasses") and hasattr(
        core, "PROCESS_DEFAULT_WARMUP_OUTER_PASSES"
    ):
        core.PROCESS_DEFAULT_WARMUP_OUTER_PASSES = warmupOuterPasses
    return warmupOuterPasses


def _processNoiseRunKwargs(processArgs: Any) -> Dict[str, Any]:
    warmupOuterPasses = _getProcessNoiseWarmupOuterPasses(processArgs)
    kwargs: Dict[str, Any] = {
        "stateModel": processArgs.stateModel,
        "qSeedPriorLevel": processArgs.qSeedPriorLevel,
        "processNoiseWarmupECMIters": processArgs.processNoiseWarmupECMIters,
        "processPrecisionMultiplierMin": processArgs.precisionMultiplierMin,
        "processPrecisionMultiplierMax": processArgs.precisionMultiplierMax,
    }
    if _coreRunConsenrichSupports("processNoiseWarmupOuterPasses"):
        kwargs["processNoiseWarmupOuterPasses"] = warmupOuterPasses
    for parameterName in (
        "processNoiseCalibration",
    ):
        if hasattr(processArgs, parameterName) and _coreRunConsenrichSupports(
            parameterName
        ):
            kwargs[parameterName] = getattr(processArgs, parameterName)
    return kwargs


def _logInitialConfigurationSummary(config: Mapping[str, Any]) -> None:
    inputArgs = config["inputArgs"]
    outputArgs = config["outputArgs"]
    genomeArgs = config["genomeArgs"]
    countingArgs = config["countingArgs"]
    processArgs = config["processArgs"]
    observationArgs = config["observationArgs"]
    uncertaintyArgs = config["uncertaintyCalibrationArgs"]
    matchingArgs = config["matchingArgs"]
    fitArgs = config["fitArgs"]
    processKappaMin = float(processArgs.precisionMultiplierMin)
    processKappaMinLabel = "auto" if processKappaMin < 0.0 else f"{processKappaMin:.6g}"

    def yn(value: Any) -> str:
        return "yes" if bool(value) else "no"

    controlInputCount = len(getattr(inputArgs, "controlSources", []) or [])
    controlsPresent = checkControlsPresent(inputArgs)

    rows = (
        ("version", __version__),
        ("experiment", config.get("experimentName", "")),
        ("defaults", config.get("defaultConfiguration", "generic")),
        ("genome", getattr(genomeArgs, "genomeName", "")),
        ("chromosome count", len(getattr(genomeArgs, "chromosomes", []) or [])),
        ("treatment inputs", len(getattr(inputArgs, "treatmentSources", []) or [])),
        ("control inputs", controlInputCount),
        ("interval bp", int(countingArgs.intervalSizeBP)),
        ("normalization", countingArgs.normMethod),
        (
            "centerMB",
            _resolveCenterMBStatus(
                countingArgs,
                controlsPresent=controlsPresent,
            )[1],
        ),
        ("centerMB method", countingArgs.centerMBMethod),
        ("MUNC variance model", observationArgs.muncVarianceModel),
        ("MUNC variance EB", yn(observationArgs.EB_use)),
        (
            "MUNC count model floor",
            yn(
                getattr(
                    observationArgs,
                    "useCountNoiseFloor",
                    True,
                )
            ),
        ),
        ("MUNC sampling iters", int(observationArgs.samplingIters)),
        ("ECM max iters", int(fitArgs.ECM_fixedBackgroundIters)),
        ("outer passes", int(fitArgs.ECM_outerIters)),
        ("background model", yn(fitArgs.fitBackground)),
        ("nonnegative background", yn(fitArgs.useNonnegativeBackground)),
        ("state model", processArgs.stateModel),
        ("deltaF", float(processArgs.deltaF)),
        (
            "process Q bounds",
            f"[{float(processArgs.minQ):.6g}, {float(processArgs.maxQ):.6g}]",
        ),
        (
            "MUNC seed process Q bounds",
            (
                f"[{float(observationArgs.muncSeedProcessMinQ):.6g}, "
                f"{float(observationArgs.muncSeedProcessMaxQ):.6g}]"
            ),
        ),
        (
            "q seed prior level",
            float(
                getattr(
                    processArgs,
                    "qSeedPriorLevel",
                    constants.PROCESS_DEFAULT_Q_SEED_PRIOR_LEVEL,
                )
            ),
        ),
        (
            "process noise calibration",
            getattr(
                processArgs,
                "processNoiseCalibration",
                constants.PROCESS_DEFAULT_NOISE_CALIBRATION,
            ),
        ),
        (
            "process kappa bounds",
            (
                f"[{processKappaMinLabel}, "
                f"{float(processArgs.precisionMultiplierMax):.6g}]"
            ),
        ),
        ("uncertainty calib", yn(uncertaintyArgs.enabled)),
        ("ROCCO peaks", yn(matchingArgs.enabled)),
        (
            "ROCCO shrunk-state export signal",
            yn(
                getattr(
                    matchingArgs,
                    "useShrunkStateScores",
                    constants.MATCHING_DEFAULT_USE_SHRUNK_STATE_SCORES,
                )
            ),
        ),
    )
    core._logEvent("config.initial", rows, logger_=logger)


def _resolveCenterMBStatus(
    countingArgs: core.countingParams,
    controlsPresent: bool,
) -> tuple[bool, str]:
    if not bool(countingArgs.centerMB):
        return False, "no"
    return True, "yes"


def _oddIntervalsFromBP(
    windowBP: float,
    intervalSizeBP: int,
    *,
    minIntervals: int = 3,
) -> int:
    intervalSizeBP_ = max(1, int(intervalSizeBP))
    window = int(math.ceil(float(windowBP) / float(intervalSizeBP_)))
    window = max(int(minIntervals), window)
    if window % 2 == 0:
        window += 1
    return int(window)


def _resolveRuntimeBackgroundBlockLen(
    dependenceSpanIntervals: Optional[int],
    backgroundBlockSizeBP: int,
    intervalSizeBP: int,
    lengthScaleMultiplier: float,
) -> int:
    multiplier = float(lengthScaleMultiplier)
    intervalSizeBP_ = max(1, int(intervalSizeBP))
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            "fitParams.ECM_backgroundLengthScaleMultiplier must be positive"
        )
    if int(backgroundBlockSizeBP) > 0:
        baseBP = max(float(backgroundBlockSizeBP), float(intervalSizeBP_))
    elif dependenceSpanIntervals is not None and int(dependenceSpanIntervals) > 0:
        baseBP = float(dependenceSpanIntervals) * float(intervalSizeBP_)
    else:
        baseBP = max(float(backgroundBlockSizeBP), float(intervalSizeBP_))
    windowBP = min(
        multiplier * baseBP,
        float(constants.FIT_BACKGROUND_LENGTH_SCALE_CAP_BP),
    )
    resolvedIntervals = _oddIntervalsFromBP(
        windowBP,
        intervalSizeBP_,
        minIntervals=1,
    )
    capIntervals = max(
        1,
        int(constants.FIT_BACKGROUND_LENGTH_SCALE_CAP_BP) // intervalSizeBP_,
    )
    if capIntervals > 1 and capIntervals % 2 == 0:
        capIntervals -= 1
    return min(int(resolvedIntervals), int(capIntervals))


def _formatOptionalLogValue(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, float) and not np.isfinite(value):
        return "NA"
    return value


def _logMuncEstimationParameters(
    *,
    chromosomeCount: int,
    sampleCount: int,
    intervalSizeBP: int,
    sizing: core._MuncRuntimeSizing,
    muncVarianceModel: str,
    samplingIters: int,
    dependenceContextBP: int | None,
    dependenceSpanIntervals: int | None,
    trendMultiplier: float,
    localMultiplier: float,
    observationArgs: core.observationParams,
    sparseBedEnabled: bool,
    varianceFloor: float | None,
    varianceCap: float | None,
    trendNumBasis: int,
    trendMinObsPerBasis: float,
    trendMinEdf: float,
    trendMaxEdf: float | None,
    trendLambdaMin: float,
    trendLambdaMax: float,
    trendLambdaGridSize: int,
    pooledPairCount: int,
    seedPassCount: int,
    logger_: logging.Logger = logger,
) -> None:
    restrictLocalVariance = bool(
        getattr(observationArgs, "restrictLocalVarianceToSparseBed", False)
    )
    core._logAsciiBlock(
        "MUNC estimation parameters",
        (
            ("MUNC variance model", muncVarianceModel),
            ("chromosomes", int(chromosomeCount)),
            ("samples", int(sampleCount)),
            ("interval bp", int(intervalSizeBP)),
            ("MUNC sampling iterations", int(samplingIters)),
            ("MUNC trend block bp", int(sizing.trendBlockSizeBP)),
            ("sampling block intervals", int(sizing.trendBlockIntervals)),
            ("MUNC trend block source", sizing.trendBlockSource),
            ("MUNC local window bp", int(sizing.localWindowSizeBP)),
            ("local window intervals", int(sizing.localWindowIntervals)),
            ("MUNC local window source", sizing.localWindowSource),
            (
                "MUNC dependence working span",
                _formatOptionalLogValue(dependenceSpanIntervals),
            ),
            (
                "MUNC derived context bp",
                _formatOptionalLogValue(dependenceContextBP),
            ),
            ("trend dependence span multiplier", float(trendMultiplier)),
            ("local dependence span multiplier", float(localMultiplier)),
            ("MUNC trend mode", "pooled"),
            ("MUNC pooled trend pairs", int(pooledPairCount)),
            ("MUNC seed passes", int(seedPassCount)),
            (
                "fragment smoothing",
                (
                    "enabled"
                    if bool(
                        getattr(
                            observationArgs,
                            "smoothToFraglen",
                            constants.OBSERVATION_DEFAULT_SMOOTH_TO_FRAGLEN,
                        )
                    )
                    else "disabled"
                ),
            ),
            (
                "MUNC seed process Q bounds",
                (
                    f"[{float(observationArgs.muncSeedProcessMinQ):.6g}, "
                    f"{float(observationArgs.muncSeedProcessMaxQ):.6g}]"
                ),
            ),
            (
                "MUNC variance EB",
                "enabled" if bool(observationArgs.EB_use) else "disabled",
            ),
            (
                "MUNC count model floor",
                (
                    "enabled"
                    if bool(
                        getattr(
                            observationArgs,
                            "useCountNoiseFloor",
                            True,
                        )
                    )
                    else "disabled"
                ),
            ),
            (
                "MUNC genomic covariates",
                (
                    "enabled"
                    if bool(
                        getattr(
                            observationArgs,
                            "muncCovariatesEnabled",
                            constants.OBSERVATION_DEFAULT_MUNC_COVARIATES_ENABLED,
                        )
                    )
                    else "disabled"
                ),
            ),
            ("EB Nu0 override", _formatOptionalLogValue(observationArgs.EB_setNu0)),
            ("EB NuL override", _formatOptionalLogValue(observationArgs.EB_setNuL)),
            ("sparse nearest regions", int(observationArgs.numNearest or 0)),
            (
                "restrict local sparse",
                "yes" if restrictLocalVariance and sparseBedEnabled else "no",
            ),
            (
                "sparse support bp",
                _formatOptionalLogValue(observationArgs.sparseSupportScaleBP),
            ),
            (
                "sparse support prior",
                _formatOptionalLogValue(observationArgs.sparseSupportPrior),
            ),
            ("variance floor", _formatOptionalLogValue(varianceFloor)),
            ("variance cap", _formatOptionalLogValue(varianceCap)),
            ("trend basis", int(trendNumBasis)),
            ("trend min obs/basis", float(trendMinObsPerBasis)),
            (
                "trend EDF range",
                f"{float(trendMinEdf):.6g}-{_formatOptionalLogValue(trendMaxEdf)}",
            ),
            (
                "trend lambda range",
                f"{float(trendLambdaMin):.6g}-{float(trendLambdaMax):.6g}",
            ),
            ("trend lambda grid", int(trendLambdaGridSize)),
        ),
        logger_=logger_,
    )


class _ConsoleLogFilter(logging.Filter):
    def __init__(
        self,
        *,
        verbose: bool,
        verbose2: bool,
        verbosity: str | None = None,
    ):
        super().__init__()
        if verbosity is None:
            if verbose2:
                verbosity = "debug"
            elif verbose:
                verbosity = "verbose"
            else:
                verbosity = constants.LOGGING_DEFAULT_VERBOSITY
        self.verbosity = str(verbosity)

    def filter(self, record: logging.LogRecord) -> bool:
        levelRank = _CONSOLE_VERBOSITY_ORDER.get(
            self.verbosity,
            _CONSOLE_VERBOSITY_ORDER[constants.LOGGING_DEFAULT_VERBOSITY],
        )
        if record.levelno >= logging.WARNING:
            return True
        if levelRank >= _CONSOLE_VERBOSITY_ORDER["debug"]:
            return True
        if (
            levelRank >= _CONSOLE_VERBOSITY_ORDER["verbose"]
            and record.levelno >= logging.INFO
        ):
            return True
        if getattr(record, _CONSOLE_VERBOSE_EVENT_ATTR, False):
            return levelRank >= _CONSOLE_VERBOSITY_ORDER["verbose"]
        return (
            levelRank >= _CONSOLE_VERBOSITY_ORDER["normal"]
            and bool(getattr(record, _CONSOLE_EVENT_ATTR, False))
        )


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, *, colorPhaseHeaders: bool = False):
        super().__init__()
        self.colorPhaseHeaders = bool(colorPhaseHeaders)

    def _color(self, message: str, colorCode: str) -> str:
        if not self.colorPhaseHeaders:
            return message
        return f"{colorCode}{message}{_CONSOLE_STYLE_RESET}"

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if getattr(record, _CONSOLE_PHASE_ATTR, False):
            message = self._color(message, _CONSOLE_PHASE_TEXT)
            message = "\n\n" + message
        elif getattr(record, _CONSOLE_SUBPHASE_ATTR, False):
            message = "  - " + self._color(message, _CONSOLE_SUBPHASE_TEXT)
        elif getattr(record, _CONSOLE_BLUE_ATTR, False):
            message = self._color(message, _CONSOLE_BLUE_TEXT)
        elif getattr(record, _CONSOLE_EVENT_ATTR, False):
            message = "  * " + self._color(message, _CONSOLE_MILESTONE_TEXT)
        if record.levelno >= logging.WARNING:
            colorCode = (
                _CONSOLE_ERROR_TEXT
                if record.levelno >= logging.ERROR
                else _CONSOLE_WARNING_TEXT
            )
            message = self._color(f"{record.levelname}: {message}", colorCode)
        if record.exc_info:
            message = message + "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            message = message + "\n" + self.formatStack(record.stack_info)
        return message


class _JSONLFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "record_type": "event",
            "event": getattr(record, "consenrich_event", None) or "log.message",
            "function": record.funcName,
            "level": record.levelname.lower(),
            "line": int(record.lineno),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "time": self.formatTime(record, _JSON_LOG_TIME_FORMAT),
        }
        fields = getattr(record, "consenrich_fields", None)
        if isinstance(fields, Mapping) and fields:
            payload["fields"] = dict(fields)
        if getattr(record, _CONSOLE_PHASE_ATTR, False):
            payload["console_phase"] = True
        if getattr(record, _CONSOLE_SUBPHASE_ATTR, False):
            payload["console_subphase"] = True
        if getattr(record, _CONSOLE_EVENT_ATTR, False):
            payload["console_milestone"] = True
        if getattr(record, _CONSOLE_VERBOSE_EVENT_ATTR, False):
            payload["console_verbose"] = True
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return logging_utils.strictJsonDumps(payload)


def _removeCliHandlers(packageLogger: logging.Logger) -> None:
    for handler in list(packageLogger.handlers):
        packageLogger.removeHandler(handler)
        if getattr(handler, _CLI_HANDLER_ATTR, False):
            try:
                handler.close()
            except Exception:
                pass


def _defaultConfigLogPath(configPath: str) -> Path:
    experimentName = constants.EXPERIMENT_DEFAULT_NAME
    try:
        configData = loadConfig(configPath)
        if isinstance(configData, Mapping):
            experimentName = configData.get(
                "experimentName",
                constants.EXPERIMENT_DEFAULT_NAME,
            )
    except Exception:
        pass
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(f"consenrichOutput_{experimentToken}_run.v{__version__}.jsonl")


def _defaultMatchLogPath(stateBedGraphPath: str) -> Path:
    stateBedGraph = Path(stateBedGraphPath)
    return stateBedGraph.with_name(
        f"{stateBedGraph.stem}_consenrich_run.v{__version__}.jsonl"
    )


def _configureCliLogging(
    logFile: str | Path | None,
    *,
    verbose: bool,
    verbose2: bool,
    verbosity: str | None = None,
    consoleStream=None,
) -> Path | None:
    verbosityLabel = (
        str(verbosity)
        if verbosity is not None
        else ("debug" if verbose2 else "verbose" if verbose else "normal")
    )
    if verbosityLabel not in _CONSOLE_VERBOSITY_ORDER:
        raise ValueError(f"Unsupported verbosity {verbosityLabel!r}")
    packageLogger = logging.getLogger("consenrich")
    _removeCliHandlers(packageLogger)
    packageLogger.setLevel(logging.DEBUG)
    packageLogger.propagate = False

    consoleTarget = sys.stderr if consoleStream is None else consoleStream
    consoleHandler = logging.StreamHandler(consoleTarget)
    setattr(consoleHandler, _CLI_HANDLER_ATTR, True)
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.addFilter(
        _ConsoleLogFilter(
            verbose=bool(verbose),
            verbose2=bool(verbose2),
            verbosity=verbosityLabel,
        )
    )
    consoleColor = (
        getattr(consoleTarget, "isatty", lambda: False)()
        and os.environ.get("TERM", "") != "dumb"
        and "NO_COLOR" not in os.environ
    )
    consoleHandler.setFormatter(_ConsoleFormatter(colorPhaseHeaders=consoleColor))
    packageLogger.addHandler(consoleHandler)

    if logFile is None:
        return None

    logPath = Path(logFile)
    try:
        if logPath.parent != Path(""):
            logPath.parent.mkdir(parents=True, exist_ok=True)
        fileHandler = logging.FileHandler(logPath, mode="w", encoding="utf-8")
        setattr(fileHandler, _CLI_HANDLER_ATTR, True)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(_JSONLFormatter())
        packageLogger.addHandler(fileHandler)
        return logPath
    except Exception as exc:
        logger.warning(
            "Failed to configure JSONL log file %s: %s.",
            logPath,
            exc,
        )
        return None


def _logCliMilestone(message: str, *args: Any, blue: bool = False) -> None:
    extra = {_CONSOLE_EVENT_ATTR: True}
    if blue:
        extra[_CONSOLE_BLUE_ATTR] = True
    logger.info(message, *args, extra=extra, stacklevel=2)


def _logCliSubphase(
    message: str,
    *args: Any,
    verboseOnly: bool = False,
) -> None:
    extra = {_CONSOLE_SUBPHASE_ATTR: True}
    if verboseOnly:
        extra[_CONSOLE_VERBOSE_EVENT_ATTR] = True
    else:
        extra[_CONSOLE_EVENT_ATTR] = True
    logger.info(message, *args, extra=extra, stacklevel=2)


def _logCliPhase(phaseLabel: str, message: str | None = None, *args: Any) -> None:
    extra = {_CONSOLE_EVENT_ATTR: True, _CONSOLE_PHASE_ATTR: True}
    if message is None:
        logger.info(
            "=== Consenrich | %s ===",
            phaseLabel,
            extra=extra,
            stacklevel=2,
        )
        return
    detail = message % args if args else message
    logger.info(
        "=== Consenrich | %s === %s",
        phaseLabel,
        detail,
        extra=extra,
        stacklevel=2,
    )


def _logCliProgressMilestone(message: str, *args: Any) -> None:
    _logCliSubphase(message, *args, verboseOnly=True)


def _buildArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consenrich CLI")
    parser.add_argument(
        "--config",
        type=str,
        dest="config",
        help="Path to a YAML config file with parameters + arguments defined in `consenrich.core`",
    )

    # --- Post hoc ROCCO peak-calling arguments ---
    parser.add_argument(
        "--match-bedGraph",
        type=str,
        dest="matchBedGraph",
        help="Path to a Consenrich state bedGraph file",
    )
    parser.add_argument(
        "--match-uncertainty-bedGraph",
        type=str,
        default=None,
        dest="matchUncertaintyBedGraph",
        help="Optional uncertainty bedGraph paired with `--match-bedGraph`. If omitted, Consenrich looks for a sibling `_uncertainty` bedGraph.",
    )
    parser.add_argument(
        "--match-blacklist-bed",
        type=str,
        default=None,
        dest="matchBlacklistBed",
        help="Optional BED blacklist applied to post hoc ROCCO peak export.",
    )
    parser.add_argument(
        "--match-num-bootstrap",
        type=int,
        default=constants.MATCHING_DEFAULT_NUM_BOOTSTRAP,
        dest="matchNumBootstrap",
        help="Number of dependent wild-bootstrap null draws used for budget calibration.",
    )
    parser.add_argument(
        "--match-threshold-z",
        type=float,
        default=constants.MATCHING_DEFAULT_THRESHOLD_Z,
        dest="matchThresholdZ",
        help="One-sided Gaussian z-threshold used when calibrating null tail occupancy.",
    )
    parser.add_argument(
        "--match-nested-rocco-iters",
        type=int,
        default=constants.MATCHING_DEFAULT_NESTED_ROCCO_ITERS,
        dest="matchNestedRoccoIters",
        help="Number of monotone nested ROCCO refinement iterations within first-pass peaks. Set to 0 to disable.",
    )
    parser.add_argument(
        "--match-nested-rocco-budget-scale",
        type=float,
        default=constants.MATCHING_DEFAULT_NESTED_ROCCO_BUDGET_SCALE,
        dest="matchNestedRoccoBudgetScale",
        help=(
            "Soft budget scale for nested ROCCO refinement; values below 1 "
            "increase the local selection penalty but do not impose a hard quota."
        ),
    )
    parser.add_argument(
        "--match-export-filter-c",
        type=float,
        default=constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER,
        dest="matchExportFilterUncertaintyMultiplier",
        help=(
            "Multiplier c in the final ROCCO export filter "
            "`medianState < -c * median(local uncertainty)`. "
            f"Default: {constants.MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER:g}. "
            "Setting c=0 requires exported peaks to have positive median signal."
        ),
    )
    parser.add_argument(
        "--match-min-peak-score",
        type=float,
        default=constants.MATCHING_DEFAULT_MIN_PEAK_SCORE,
        dest="matchMinPeakScore",
        help=(
            "Minimum ROCCO signalValue required to keep a peak in the exported result."
        ),
    )
    parser.add_argument(
        "--match-peak-mode",
        type=str,
        choices=constants.MATCHING_PEAK_MODES,
        default=constants.MATCHING_DEFAULT_PEAK_MODE,
        dest="matchPeakMode",
        help="ROCCO peak export mode.",
    )
    parser.add_argument(
        "--match-broad-weak-threshold-z",
        type=float,
        default=constants.MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z,
        dest="matchBroadWeakThresholdZ",
        help="Weak one-sided Gaussian z-threshold used by broad ROCCO mode.",
    )
    parser.add_argument(
        "--match-broad-max-gap-bp",
        type=int,
        default=constants.MATCHING_DEFAULT_BROAD_MAX_GAP_BP,
        dest="matchBroadMaxGapBP",
        help="Maximum gap in bp considered for broad ROCCO parent merging.",
    )
    parser.add_argument(
        "--match-uncertainty-score-mode",
        type=str,
        choices=constants.MATCHING_SUPPORTED_UNCERTAINTY_SCORE_MODES,
        default=constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
        dest="matchUncertaintyScoreMode",
        help=(
            "ROCCO score construction mode. `state` uses the fitted state directly; "
            "`lower_confidence` uses state - z * uncertainty and requires an "
            "uncertainty bedGraph."
        ),
    )
    parser.add_argument(
        "--match-uncertainty-score-z",
        type=float,
        default=constants.MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
        dest="matchUncertaintyScoreZ",
        help="Multiplier z used by --match-uncertainty-score-mode lower_confidence.",
    )
    parser.add_argument(
        "--match-seed",
        type=int,
        default=constants.MATCHING_DEFAULT_RAND_SEED,
        dest="matchRandSeed",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        dest="logFile",
        help="Path for the canonical Consenrich JSONL event log.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=constants.LOGGING_VERBOSITY_LEVELS,
        default=None,
        dest="verbosity",
        help="Console logging level.",
    )
    parser.add_argument(
        "--progress",
        type=str,
        choices=constants.LOGGING_PROGRESS_MODES,
        default=None,
        dest="progress",
        help="Progress display mode. Progress bars are not used by this CLI.",
    )
    parser.add_argument("--verbose", action="store_true", help="Use verbose logging.")
    parser.add_argument(
        "--verbose2",
        action="store_true",
        help="Use debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Consenrich v{__version__}",
    )
    return parser


def main():
    parser = _buildArgParser()
    args = parser.parse_args()
    cliVerbositySet = args.verbosity is not None or args.verbose or args.verbose2
    cliProgressSet = args.progress is not None
    if args.verbosity is not None and (args.verbose or args.verbose2):
        parser.error("--verbosity cannot be combined with --verbose or --verbose2")
    if args.verbosity is None:
        if args.verbose2:
            args.verbosity = "debug"
        elif args.verbose:
            args.verbosity = "verbose"
        else:
            args.verbosity = constants.LOGGING_DEFAULT_VERBOSITY
    configLoggingArgs = None
    if not args.matchBedGraph and args.config and os.path.exists(args.config):
        configLoggingArgs = getLoggingArgs(args.config)
        if not cliVerbositySet:
            args.verbosity = configLoggingArgs.verbosity
        if not cliProgressSet:
            args.progress = configLoggingArgs.progress
        if args.logFile is None and configLoggingArgs.logFile is not None:
            args.logFile = configLoggingArgs.logFile
    if args.progress is None:
        args.progress = constants.LOGGING_DEFAULT_PROGRESS
    args.verbose = args.verbosity in {"verbose", "debug"}
    args.verbose2 = args.verbosity == "debug"

    if args.matchBedGraph:
        if not os.path.exists(args.matchBedGraph):
            raise FileNotFoundError(
                f"bedGraph file {args.matchBedGraph} couldn't be found."
            )
        uncertaintyBedGraph = args.matchUncertaintyBedGraph
        if uncertaintyBedGraph is not None and not os.path.exists(uncertaintyBedGraph):
            raise FileNotFoundError(
                f"uncertainty bedGraph file {uncertaintyBedGraph} couldn't be found."
            )
        if args.matchBlacklistBed is not None and not os.path.exists(
            args.matchBlacklistBed
        ):
            raise FileNotFoundError(
                f"blacklist BED file {args.matchBlacklistBed} couldn't be found."
            )
        resolvedLogPath = _configureCliLogging(
            args.logFile or _defaultMatchLogPath(args.matchBedGraph),
            verbose=bool(args.verbose),
            verbose2=bool(args.verbose2),
            verbosity=args.verbosity,
        )
        if resolvedLogPath is not None:
            _logCliMilestone("Canonical log: %s", resolvedLogPath)
        if uncertaintyBedGraph is None:
            uncertaintyBedGraph = _inferMatchingUncertaintyBedGraph(args.matchBedGraph)
        matchStart = time.perf_counter()
        _logCliPhase("Post-hoc ROCCO")
        _logCliSubphase(
            "State input: %s",
            args.matchBedGraph,
        )
        logger.info(
            "Running post hoc ROCCO peak caller using state bedGraph %s...",
            args.matchBedGraph,
        )
        outName = peaks.solveRocco(
            args.matchBedGraph,
            uncertaintyBedGraphFile=uncertaintyBedGraph,
            numBootstrap=args.matchNumBootstrap,
            thresholdZ=args.matchThresholdZ,
            nestedRoccoIters=args.matchNestedRoccoIters,
            nestedRoccoBudgetScale=args.matchNestedRoccoBudgetScale,
            exportFilterUncertaintyMultiplier=(
                args.matchExportFilterUncertaintyMultiplier
            ),
            minPeakScore=args.matchMinPeakScore,
            peakMode=args.matchPeakMode,
            broadWeakThresholdZ=args.matchBroadWeakThresholdZ,
            broadMaxGapBP=args.matchBroadMaxGapBP,
            uncertaintyScoreMode=args.matchUncertaintyScoreMode,
            uncertaintyScoreZ=args.matchUncertaintyScoreZ,
            blacklistBedFile=args.matchBlacklistBed,
            randSeed=args.matchRandSeed,
            verbose=bool(args.verbose or args.verbose2),
        )
        logger.info("Finished post hoc ROCCO peak calling. Written to %s", outName)
        _logCliMilestone(
            "Consenrich post-hoc ROCCO done: output=%s elapsed=%.1fs",
            outName,
            time.perf_counter() - matchStart,
        )
        sys.exit(0)

    if not args.config:
        _configureCliLogging(
            args.logFile,
            verbose=bool(args.verbose),
            verbose2=bool(args.verbose2),
            verbosity=args.verbosity,
        )
        _logCliMilestone(
            "No config file provided, run with `--config <path_to_config.yaml>`"
        )
        _logCliMilestone(
            "See documentation: https://nolan-h-hamilton.github.io/Consenrich/"
        )
        sys.exit(1)

    if not os.path.exists(args.config):
        _configureCliLogging(
            args.logFile,
            verbose=bool(args.verbose),
            verbose2=bool(args.verbose2),
            verbosity=args.verbosity,
        )
        _logCliMilestone("Config file %s does not exist.", args.config)
        _logCliMilestone(
            "See documentation: https://nolan-h-hamilton.github.io/Consenrich/"
        )
        sys.exit(1)

    resolvedLogPath = _configureCliLogging(
        args.logFile
        or (
            configLoggingArgs.logFile
            if configLoggingArgs is not None
            else constants.LOGGING_DEFAULT_LOG_FILE
        )
        or _defaultConfigLogPath(args.config),
        verbose=bool(args.verbose),
        verbose2=bool(args.verbose2),
        verbosity=args.verbosity,
    )
    if resolvedLogPath is not None:
        _logCliMilestone("Canonical log: %s", resolvedLogPath)
    _logCliPhase("Config")
    cliRunStart = time.perf_counter()
    config = readConfig(args.config)
    experimentName = config["experimentName"]
    diagnosticLogPaths = _diagnosticLogPaths(str(experimentName))
    correlationLengthPath = _correlationLengthPath(str(experimentName))
    correlationLengthPlotPath = _correlationLengthPlotPath(str(experimentName))
    correlationLengthWindowsPath = _correlationLengthWindowsPath(str(experimentName))
    _initializeDiagnosticLogs(diagnosticLogPaths)
    genomeArgs = config["genomeArgs"]
    inputArgs = config["inputArgs"]
    outputArgs = config["outputArgs"]
    countingArgs = config["countingArgs"]
    scArgs = config["scArgs"]
    processArgs = config["processArgs"]
    observationArgs = config["observationArgs"]
    stateArgs = config["stateArgs"]
    uncertaintyCalibrationArgs = config["uncertaintyCalibrationArgs"]
    samArgs = config["samArgs"]
    matchingArgs = config["matchingArgs"]
    fitArgs = config["fitArgs"]
    _configureCoreProcessNoiseWarmupDefaults(processArgs)
    treatmentSources = _listOrEmpty(getattr(inputArgs, "treatmentSources", None))
    controlSources = _listOrEmpty(getattr(inputArgs, "controlSources", None))
    if not treatmentSources:
        treatmentSources = _buildPathInputSources(inputArgs.bamFiles, role="treatment")
    if not controlSources:
        controlSources = _buildPathInputSources(
            _listOrEmpty(inputArgs.bamFilesControl),
            role="control",
        )
    bamFiles = core.getSourcePaths(treatmentSources)
    bamFilesControl = core.getSourcePaths(controlSources)
    numSamples = len(bamFiles)
    sampleFileByIndex = [
        os.path.basename(str(source.path))[:7] for source in treatmentSources
    ]
    if len(sampleFileByIndex) != numSamples or any(
        not sampleFile for sampleFile in sampleFileByIndex
    ):
        raise ValueError("treatment source sampleFile must be non-empty")
    intervalSizeBP = countingArgs.intervalSizeBP
    excludeForNorm = genomeArgs.excludeForNorm
    chromSizes = genomeArgs.chromSizesFile
    deltaF_ = processArgs.deltaF
    maxR_ = observationArgs.maxR
    minQ_ = processArgs.minQ
    maxQ_ = processArgs.maxQ
    configuredMinR_ = observationArgs.minR
    if configuredMinR_ is not None and not np.isfinite(float(configuredMinR_)):
        raise ValueError("observationParams.minR must be finite when provided")
    if configuredMinR_ is not None and float(configuredMinR_) > 0.0:
        logger.info(
            "observationParams.minR is ignored; transformed-scale count-noise "
            "floors now provide the observation variance floor.",
        )
    muncTrendBlockSizeBP_ = getattr(
        observationArgs,
        "muncTrendBlockSizeBP",
        constants.OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_SIZE_BP,
    )
    muncLocalWindowSizeBP_ = getattr(
        observationArgs,
        "muncLocalWindowSizeBP",
        constants.OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_SIZE_BP,
    )
    dependenceWindowCount_ = int(
        getattr(
            observationArgs,
            "dependenceWindowCount",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_COUNT,
        )
    )
    dependenceWindowBP_ = int(
        getattr(
            observationArgs,
            "dependenceWindowBP",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_WINDOW_BP,
        )
    )
    dependenceMaxLagBP_ = int(
        getattr(
            observationArgs,
            "dependenceMaxLagBP",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_MAX_LAG_BP,
        )
    )
    dependenceWorkingQuantile_ = float(
        getattr(
            observationArgs,
            "dependenceWorkingQuantile",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_WORKING_QUANTILE,
        )
    )
    dependenceBootstrapDraws_ = int(
        getattr(
            observationArgs,
            "dependenceBootstrapDraws",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_BOOTSTRAP_DRAWS,
        )
    )
    dependenceMinWindowCount_ = int(
        getattr(
            observationArgs,
            "dependenceMinWindowCount",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_MIN_WINDOW_COUNT,
        )
    )
    if (
        dependenceWindowCount_ <= 0
        or dependenceWindowBP_ <= 0
        or dependenceMaxLagBP_ <= 0
        or dependenceMaxLagBP_ > dependenceWindowBP_ // 2
        or not np.isfinite(dependenceWorkingQuantile_)
        or dependenceWorkingQuantile_ <= 0.0
        or dependenceWorkingQuantile_ >= 1.0
        or dependenceBootstrapDraws_ <= 0
        or dependenceMinWindowCount_ <= 0
        or dependenceMinWindowCount_ > dependenceWindowCount_
    ):
        raise ValueError("observationParams dependence window settings are invalid")
    dependenceAcfPointThreshold_ = float(
        getattr(
            observationArgs,
            "dependenceAcfPointThreshold",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_ACF_POINT_THRESHOLD,
        )
    )
    if (
        not np.isfinite(dependenceAcfPointThreshold_)
        or dependenceAcfPointThreshold_ <= 0.0
        or dependenceAcfPointThreshold_ >= 1.0
    ):
        raise ValueError(
            "observationParams.dependenceAcfPointThreshold must satisfy 0 < x < 1"
        )
    dependenceAcfSmoothingBP_ = int(
        getattr(
            observationArgs,
            "dependenceAcfSmoothingBP",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_ACF_SMOOTHING_BP,
        )
    )
    dependenceCrossingPersistenceBP_ = int(
        getattr(
            observationArgs,
            "dependenceCrossingPersistenceBP",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_CROSSING_PERSISTENCE_BP,
        )
    )
    dependenceMinFinitePairs_ = int(
        getattr(
            observationArgs,
            "dependenceMinFinitePairs",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_MIN_FINITE_PAIRS,
        )
    )
    dependenceMinFinitePairCoverage_ = float(
        getattr(
            observationArgs,
            "dependenceMinFinitePairCoverage",
            constants.OBSERVATION_DEFAULT_DEPENDENCE_MIN_FINITE_PAIR_COVERAGE,
        )
    )
    if (
        dependenceAcfSmoothingBP_ <= 0
        or dependenceCrossingPersistenceBP_ <= 0
        or dependenceMinFinitePairs_ <= 0
        or not np.isfinite(dependenceMinFinitePairCoverage_)
        or dependenceMinFinitePairCoverage_ <= 0.0
        or dependenceMinFinitePairCoverage_ > 1.0
    ):
        raise ValueError(
            "observationParams dependence smoothing and finite-pair settings are invalid"
        )
    muncTrendBlockDependenceMultiplier_ = float(
        getattr(
            observationArgs,
            "muncTrendBlockDependenceMultiplier",
            constants.OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER,
        )
    )
    muncLocalWindowDependenceMultiplier_ = float(
        getattr(
            observationArgs,
            "muncLocalWindowDependenceMultiplier",
            constants.OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER,
        )
    )
    smoothToFraglenRaw = getattr(
        observationArgs,
        "smoothToFraglen",
        constants.OBSERVATION_DEFAULT_SMOOTH_TO_FRAGLEN,
    )
    if not isinstance(smoothToFraglenRaw, (bool, np.bool_)):
        raise ValueError("observationParams.smoothToFraglen must be boolean")
    smoothToFraglen_ = bool(smoothToFraglenRaw)
    muncVarianceModel_ = core._normalizeMuncVarianceModel(
        getattr(
            observationArgs,
            "muncVarianceModel",
            constants.OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL,
        )
    )
    backgroundBlockSizeBP_ = countingArgs.backgroundBlockSizeBP
    dependenceContextBP_: Optional[int] = None
    dependenceSpanIntervals_: Optional[int] = None
    correlationLengthRow_: dict[str, Any] | None = None
    correlationLengthDiagnostics_: dict[str, Any] | None = None
    waitForMatrix: bool = False
    normMethod_: Optional[str] = countingArgs.normMethod.upper()
    pad_ = observationArgs.pad if hasattr(observationArgs, "pad") else 1.0e-4
    _logCliSubphase(
        "Diagnostic logs: precision=%s convergence=%s delete-block=%s",
        diagnosticLogPaths.precision,
        diagnosticLogPaths.convergence,
        diagnosticLogPaths.delete_block_calibration,
    )
    _logCliSubphase("Correlation length TSV: %s", correlationLengthPath)
    _logCliSubphase("Correlation length windows TSV: %s", correlationLengthWindowsPath)
    if bool(
        getattr(
            outputArgs,
            "plotCorrelationLength",
            constants.OUTPUT_DEFAULT_PLOT_CORRELATION_LENGTH,
        )
    ):
        _logCliSubphase("Correlation length plot: %s", correlationLengthPlotPath)
    _logCliSubphase(
        "Output file policy: nonTrackCapBytes=%d precisionDiagnostics=%s maxRowsPerChromosome=%d roccoMetadata=%s",
        int(outputArgs.maxNonTrackFileBytes),
        outputArgs.precisionDiagnosticDetail,
        int(outputArgs.maxPrecisionDiagnosticRowsPerChromosome),
        matchingArgs.metadataDetail,
    )
    _logCliSubphase(
        "Run config: experiment=%s version=%s config=%s chromosomes=%d samples=%d",
        experimentName,
        __version__,
        args.config,
        len(genomeArgs.chromosomes),
        int(numSamples),
    )

    if args.verbose:
        try:
            _logInitialConfigurationSummary(config)
        except Exception as e:
            logger.warning(f"Failed to print parsed config:\n{e}\n")

    if normMethod_ in ["SF"] and (len(bamFilesControl) > 0 or numSamples < 3):
        logger.warning(
            "`countingParams.normMethod` `SF` is not available when control inputs are present OR if < 3 treatment samples are given."
            "  --> using CPM/RPKM ..."
        )
        normMethod_ = "RPKM"

    controlsPresent = checkControlsPresent(inputArgs)
    if args.verbose:
        logger.info(f"controlsPresent: {controlsPresent}")
    anyFragments = any(
        str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND
        for source in treatmentSources + controlSources
    )
    anyBedGraph = any(
        str(source.sourceKind).upper() == core.BEDGRAPH_SOURCE_KIND
        for source in treatmentSources + controlSources
    )
    allBedGraph = all(
        str(source.sourceKind).upper() == core.BEDGRAPH_SOURCE_KIND
        for source in treatmentSources + controlSources
    )
    if anyFragments and normMethod_ in ["EGS", "RPGC"]:
        logger.warning(
            "Fragments inputs use insertion-based depth normalization not EGS/RPGC"
            "  --> using CPM/RPKM ..."
        )
        normMethod_ = "CPM"
    if (
        anyBedGraph
        and not allBedGraph
        and (
            countingArgs.scaleFactors is None
            or (controlsPresent and countingArgs.scaleFactorsControl is None)
        )
    ):
        raise ValueError(
            "Mixed BEDGRAPH and read-count sources require explicit "
            "`countingParams.scaleFactors`"
            + (" and `countingParams.scaleFactorsControl`." if controlsPresent else ".")
        )
    if allBedGraph and normMethod_ in ["EGS", "RPGC", "RPKM", "CPM"]:
        logger.info(
            "BEDGRAPH inputs are treated as already scaled tracks; using unit "
            "scale factors unless explicit scale factors are provided."
        )
    for source in treatmentSources + controlSources:
        if str(source.sourceKind).upper() == core.FRAGMENTS_SOURCE_KIND:
            core._normalizeFragmentPositionMode(
                source.fragmentPositionMode or scArgs.defaultFragmentPositionMode
            )
    readLengthsBamFiles = getReadLengths(inputArgs, countingArgs, samArgs)
    effectiveGenomeSizes = getEffectiveGenomeSizes(genomeArgs, readLengthsBamFiles)
    treatmentBamInputModes = [
        (
            core._resolveSourceBamInputMode(
                source,
                str(samArgs.bamInputMode or "auto"),
            )
            if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS
            else "reads"
        )
        for source in treatmentSources
    ]
    controlBamInputModes = [
        (
            core._resolveSourceBamInputMode(
                source,
                str(samArgs.bamInputMode or "auto"),
            )
            if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS
            else "reads"
        )
        for source in controlSources
    ]
    treatmentCountModes = [
        _getSourceCountMode(
            source,
            str(samArgs.defaultCountMode or constants.SAM_DEFAULT_COUNT_MODE),
            str(scArgs.defaultCountMode or constants.SC_DEFAULT_COUNT_MODE),
        )
        for source in treatmentSources
    ]
    controlCountModes = [
        _getSourceCountMode(
            source,
            str(samArgs.defaultCountMode or constants.SAM_DEFAULT_COUNT_MODE),
            str(scArgs.defaultCountMode or constants.SC_DEFAULT_COUNT_MODE),
        )
        for source in controlSources
    ]
    singleBinCountModes = {"cutsite", "fiveprime", "ffp", "ffp-center", "center"}
    singleBinActiveCountModes = sorted(
        {
            str(countMode)
            for countMode in treatmentCountModes + controlCountModes
            if str(countMode) in singleBinCountModes
        }
    )
    if bool(smoothToFraglen_) and singleBinActiveCountModes:
        singleBinDisplayModes = [
            "center/midpoint" if mode == "center" else mode
            for mode in singleBinActiveCountModes
        ]
        logger.warning(
            "observationParams.smoothToFraglen is enabled, but "
            "single-bin count mode(s) are active (%s). Disabling fragment "
            "smoothing because these modes intentionally collapse each event to "
            "one interval.",
            ",".join(singleBinDisplayModes),
        )
        smoothToFraglen_ = False
    treatmentBamInputModes = [
        (
            core._resolveSourceBamInputModeForCountMode(
                source,
                str(samArgs.bamInputMode or "auto"),
                countMode,
            )
            if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS
            else sourceBamInputMode
        )
        for source, sourceBamInputMode, countMode in zip(
            treatmentSources,
            treatmentBamInputModes,
            treatmentCountModes,
        )
    ]
    controlBamInputModes = [
        (
            core._resolveSourceBamInputModeForCountMode(
                source,
                str(samArgs.bamInputMode or "auto"),
                countMode,
            )
            if str(source.sourceKind).upper() in core.ALIGNMENT_SOURCE_KINDS
            else sourceBamInputMode
        )
        for source, sourceBamInputMode, countMode in zip(
            controlSources,
            controlBamInputModes,
            controlCountModes,
        )
    ]
    treatmentNativeCountModes = [
        core._nativeCountModeForPreset(countMode) for countMode in treatmentCountModes
    ]
    controlNativeCountModes = [
        core._nativeCountModeForPreset(countMode) for countMode in controlCountModes
    ]
    autoInferFragmentLength = (
        samArgs.inferFragmentLength is None
        and core._normalizeBamInputMode(samArgs.bamInputMode) == "auto"
    )
    inferFragmentLengthRequested = int(samArgs.inferFragmentLength or 0) > 0
    if autoInferFragmentLength and any(
        sourceBamInputMode in ("reads", "read1")
        for sourceBamInputMode in treatmentBamInputModes + controlBamInputModes
    ):
        logger.info(
            "samParams.bamInputMode=auto and samParams.inferFragmentLength omitted: "
            "single-end BAM sources will be extended by inferred fragment length."
        )
    treatmentAllowLists, treatmentSelectedCellCounts, treatmentNormTempPaths = (
        _prepareFragmentsNormalizationMetadata(treatmentSources)
    )
    controlAllowLists, controlSelectedCellCounts, controlNormTempPaths = (
        _prepareFragmentsNormalizationMetadata(controlSources)
    )

    peakCallingEnabled = checkMatchingEnabled(matchingArgs)
    if args.verbose:
        logger.info(f"peakCallingEnabled: {peakCallingEnabled}")
    scaleFactors = io_helpers._normalizeScaleFactorList(
        countingArgs.scaleFactors,
        len(treatmentSources),
        "countingParams.scaleFactors",
    )
    scaleFactorsControl = countingArgs.scaleFactorsControl
    if controlsPresent:
        scaleFactorsControl = io_helpers._normalizeScaleFactorList(
            countingArgs.scaleFactorsControl,
            len(controlSources),
            "countingParams.scaleFactorsControl",
        )
    characteristicFragmentLengthsTreatment: List[int] = []
    characteristicFragmentLengthsControl: List[int] = []
    countExtendFrom5pBPTreatment: List[int] = []
    countExtendFrom5pBPControl: List[int] = []
    setupAllowThreads = int(samArgs.samThreads) <= 1
    sf: np.ndarray = np.empty((numSamples,), dtype=float)
    configuredExtendFrom5pBPTreatment = core._resolveExtendFrom5pBP(
        samArgs.extendFrom5pBP,
        treatmentSources,
    )
    configuredExtendFrom5pBPControl = (
        list(configuredExtendFrom5pBPTreatment[: len(controlSources)])
        if controlsPresent
        else []
    )

    def _estimateFragmentLengthForSource(
        source: core.inputSource,
        sourceFlagExclude: int,
    ) -> int:
        if str(source.sourceKind).upper() not in core.ALIGNMENT_SOURCE_KINDS:
            return 0
        fragmentLength = int(
            cconsenrich.cgetFragmentLength(
                source.path,
                samThreads=samArgs.samThreads,
                samFlagExclude=sourceFlagExclude,
                maxInsertSize=samArgs.maxInsertSize,
            )
        )
        logger.info(
            "Estimated fragment length for %s: %d",
            source.path,
            fragmentLength,
        )
        return fragmentLength

    def _resolveCharacteristicFragmentLength(task) -> int:
        source, sourceBamInputMode, configuredExtendBP = task
        if str(source.sourceKind).upper() not in core.ALIGNMENT_SOURCE_KINDS:
            return 0
        if sourceBamInputMode == "fragments" and (
            int(configuredExtendBP) > 0 or inferFragmentLengthRequested
        ):
            raise ValueError(
                "`samParams.extendFrom5pBP` and `samParams.inferFragmentLength` "
                "require `bamInputMode` `reads` or `read1`."
            )
        if int(configuredExtendBP) > 0:
            return int(configuredExtendBP)
        return _estimateFragmentLengthForSource(
            source,
            core._resolveSourceFlagExclude(
                samArgs.samFlagExclude,
                sourceBamInputMode,
            ),
        )

    characteristicFragmentLengthsTreatment = io_helpers._threadMap(
        zip(
            treatmentSources,
            treatmentBamInputModes,
            configuredExtendFrom5pBPTreatment,
        ),
        _resolveCharacteristicFragmentLength,
        "characteristic lengths",
        allowThreads=setupAllowThreads,
    )
    if controlsPresent:
        logger.info(
            "Using treatment-derived extension lengths for control BAM sources."
        )
        (
            characteristicFragmentLengthsTreatment,
            characteristicFragmentLengthsControl,
        ) = _resolveExtendFrom5pBPPairs(
            characteristicFragmentLengthsTreatment,
            characteristicFragmentLengthsControl,
        )

    def _resolveCountExtendFrom5pBP(
        source: core.inputSource,
        sourceBamInputMode: str,
        countMode: str,
        configuredExtendBP: int,
        characteristicFragmentLength: int,
    ) -> int:
        if str(source.sourceKind).upper() not in core.ALIGNMENT_SOURCE_KINDS:
            return 0
        if sourceBamInputMode == "fragments":
            return 0
        if int(configuredExtendBP) > 0:
            return int(configuredExtendBP)
        if countMode == "ffp-center":
            return int(characteristicFragmentLength)
        if inferFragmentLengthRequested or (
            autoInferFragmentLength and sourceBamInputMode in ("reads", "read1")
        ):
            return int(characteristicFragmentLength)
        return 0

    countExtendFrom5pBPTreatment = [
        _resolveCountExtendFrom5pBP(
            source,
            sourceBamInputMode,
            countMode,
            configuredExtendBP,
            characteristicFragmentLength,
        )
        for (
            source,
            sourceBamInputMode,
            countMode,
            configuredExtendBP,
            characteristicFragmentLength,
        ) in zip(
            treatmentSources,
            treatmentBamInputModes,
            treatmentCountModes,
            configuredExtendFrom5pBPTreatment,
            characteristicFragmentLengthsTreatment,
        )
    ]
    if controlsPresent:
        countExtendFrom5pBPControl = [
            _resolveCountExtendFrom5pBP(
                source,
                sourceBamInputMode,
                countMode,
                configuredExtendBP,
                characteristicFragmentLength,
            )
            for (
                source,
                sourceBamInputMode,
                countMode,
                configuredExtendBP,
                characteristicFragmentLength,
            ) in zip(
                controlSources,
                controlBamInputModes,
                controlCountModes,
                configuredExtendFrom5pBPControl,
                characteristicFragmentLengthsControl,
            )
        ]

    try:
        if controlsPresent:

            def _getReadLengthForSource(source: core.inputSource) -> int:
                return core.getReadLength(
                    source.path,
                    100,
                    1000,
                    samArgs.samThreads,
                    samArgs.samFlagExclude,
                    sourceKind=str(source.sourceKind).upper(),
                )

            readLengthsControlBamFiles = io_helpers._threadMap(
                controlSources,
                _getReadLengthForSource,
                "control read lengths",
                allowThreads=setupAllowThreads,
            )
            effectiveGenomeSizesControl = getEffectiveGenomeSizes(
                genomeArgs,
                readLengthsControlBamFiles,
            )

            if scaleFactors is not None and scaleFactorsControl is not None:
                treatScaleFactors = scaleFactors
                controlScaleFactors = scaleFactorsControl
            elif allBedGraph:
                treatScaleFactors = scaleFactors or [1.0] * len(treatmentSources)
                controlScaleFactors = scaleFactorsControl or [1.0] * len(controlSources)
            else:

                def _getPairScaleFactors(task):
                    (
                        sourceA,
                        sourceB,
                        effectiveGenomeSizeA,
                        effectiveGenomeSizeB,
                        readLengthA,
                        readLengthB,
                        barcodeAllowListPathA,
                        barcodeAllowListPathB,
                        countModeA,
                        countModeB,
                        groupCellCountA,
                        groupCellCountB,
                        bamInputModeA,
                        bamInputModeB,
                        extendBPA,
                        extendBPB,
                        countReadLengthA,
                        countReadLengthB,
                    ) = task
                    return detrorm.getPairScaleFactors(
                        sourceA.path,
                        sourceB.path,
                        effectiveGenomeSizeA,
                        effectiveGenomeSizeB,
                        readLengthA,
                        readLengthB,
                        excludeForNorm,
                        chromSizes,
                        samArgs.samThreads,
                        intervalSizeBP,
                        normMethod=normMethod_,
                        fixControl=countingArgs.fixControl,
                        sourceKindA=str(sourceA.sourceKind).upper(),
                        sourceKindB=str(sourceB.sourceKind).upper(),
                        barcodeAllowListFileA=barcodeAllowListPathA,
                        barcodeAllowListFileB=barcodeAllowListPathB,
                        countModeA=countModeA,
                        countModeB=countModeB,
                        oneReadPerBin=samArgs.oneReadPerBin,
                        groupCellCountA=groupCellCountA,
                        groupCellCountB=groupCellCountB,
                        fragmentsGroupNorm=scArgs.fragmentsGroupNorm,
                        bamInputModeA=bamInputModeA,
                        bamInputModeB=bamInputModeB,
                        samFlagExclude=samArgs.samFlagExclude,
                        minMappingQuality=samArgs.minMappingQuality,
                        minTemplateLength=samArgs.minTemplateLength,
                        maxInsertSize=samArgs.maxInsertSize,
                        extendBPA=extendBPA,
                        extendBPB=extendBPB,
                        countReadLengthA=countReadLengthA,
                        countReadLengthB=countReadLengthB,
                    )

                pairScalingFactors = io_helpers._threadMap(
                    zip(
                        treatmentSources,
                        controlSources,
                        effectiveGenomeSizes,
                        effectiveGenomeSizesControl,
                        characteristicFragmentLengthsTreatment,
                        characteristicFragmentLengthsControl,
                        treatmentAllowLists,
                        controlAllowLists,
                        treatmentNativeCountModes,
                        controlNativeCountModes,
                        treatmentSelectedCellCounts,
                        controlSelectedCellCounts,
                        treatmentBamInputModes,
                        controlBamInputModes,
                        countExtendFrom5pBPTreatment,
                        countExtendFrom5pBPControl,
                        readLengthsBamFiles,
                        readLengthsControlBamFiles,
                    ),
                    _getPairScaleFactors,
                    "pair scale factors",
                    allowThreads=setupAllowThreads,
                )
                treatScaleFactors = []
                controlScaleFactors = []
                for scaleFactorA, scaleFactorB in pairScalingFactors:
                    treatScaleFactors.append(scaleFactorA)
                    controlScaleFactors.append(scaleFactorB)

        else:
            treatScaleFactors = scaleFactors
            controlScaleFactors = scaleFactorsControl

        if scaleFactors is None and not controlsPresent:
            if allBedGraph and normMethod_ != "SF":
                scaleFactors = [1.0] * len(treatmentSources)
            elif normMethod_ in ["RPKM", "CPM"]:

                def _getScaleFactorPerMillion(task) -> float:
                    (
                        source,
                        barcodeAllowListPath,
                        countMode,
                        groupCellCount,
                        bamInputMode,
                        extendBP,
                        readLength,
                    ) = task
                    return detrorm.getScaleFactorPerMillion(
                        source.path,
                        excludeForNorm,
                        intervalSizeBP,
                        normMethod=normMethod_,
                        sourceKind=str(source.sourceKind).upper(),
                        barcodeAllowListFile=barcodeAllowListPath,
                        countMode=countMode,
                        oneReadPerBin=samArgs.oneReadPerBin,
                        samThreads=samArgs.samThreads,
                        groupCellCount=groupCellCount,
                        fragmentsGroupNorm=scArgs.fragmentsGroupNorm,
                        bamInputMode=bamInputMode,
                        samFlagExclude=samArgs.samFlagExclude,
                        minMappingQuality=samArgs.minMappingQuality,
                        minTemplateLength=samArgs.minTemplateLength,
                        maxInsertSize=samArgs.maxInsertSize,
                        readLength=readLength,
                        extendBP=extendBP,
                    )

                scaleFactors = io_helpers._threadMap(
                    zip(
                        treatmentSources,
                        treatmentAllowLists,
                        treatmentNativeCountModes,
                        treatmentSelectedCellCounts,
                        treatmentBamInputModes,
                        countExtendFrom5pBPTreatment,
                        readLengthsBamFiles,
                    ),
                    _getScaleFactorPerMillion,
                    "scale factors",
                    allowThreads=setupAllowThreads,
                )
            elif normMethod_ in ["EGS", "RPGC"]:

                def _getScaleFactor1x(task) -> float:
                    (
                        source,
                        effectiveGenomeSize,
                        readLength,
                        barcodeAllowListPath,
                        countMode,
                        groupCellCount,
                        bamInputMode,
                        extendBP,
                        countReadLength,
                    ) = task
                    return detrorm.getScaleFactor1x(
                        source.path,
                        effectiveGenomeSize,
                        readLength,
                        excludeForNorm,
                        genomeArgs.chromSizesFile,
                        samArgs.samThreads,
                        sourceKind=str(source.sourceKind).upper(),
                        barcodeAllowListFile=barcodeAllowListPath,
                        countMode=countMode,
                        oneReadPerBin=samArgs.oneReadPerBin,
                        groupCellCount=groupCellCount,
                        fragmentsGroupNorm=scArgs.fragmentsGroupNorm,
                        bamInputMode=bamInputMode,
                        samFlagExclude=samArgs.samFlagExclude,
                        minMappingQuality=samArgs.minMappingQuality,
                        minTemplateLength=samArgs.minTemplateLength,
                        maxInsertSize=samArgs.maxInsertSize,
                        extendBP=extendBP,
                        intervalSizeBP=intervalSizeBP,
                        countReadLength=countReadLength,
                    )

                scaleFactors = io_helpers._threadMap(
                    zip(
                        treatmentSources,
                        effectiveGenomeSizes,
                        characteristicFragmentLengthsTreatment,
                        treatmentAllowLists,
                        treatmentNativeCountModes,
                        treatmentSelectedCellCounts,
                        treatmentBamInputModes,
                        countExtendFrom5pBPTreatment,
                        readLengthsBamFiles,
                    ),
                    _getScaleFactor1x,
                    "scale factors",
                    allowThreads=setupAllowThreads,
                )
            elif normMethod_ in ["SF"]:
                waitForMatrix = True
    finally:
        for tempPath in treatmentNormTempPaths + controlNormTempPaths:
            try:
                os.remove(tempPath)
            except OSError:
                pass

    deltaF_ = core._resolveFixedDeltaF(deltaF_)
    logger.info("Using fixed deltaF=%.6f", deltaF_)

    _logCliPhase(
        "Chromosome planning",
        "requested=%d interval_bp=%d",
        len(genomeArgs.chromosomes),
        int(intervalSizeBP),
    )
    chromSizesDict = misc_util.getChromSizesDict(
        genomeArgs.chromSizesFile,
        excludeChroms=genomeArgs.excludeChroms,
    )
    chromosomes = genomeArgs.chromosomes
    skippedChromosomes = [
        str(chromosome)
        for chromosome in chromosomes
        if str(chromosome) not in chromSizesDict
    ]
    if skippedChromosomes:
        logger.info(
            "chromosome.skip missing_or_excluded names=%s",
            ",".join(skippedChromosomes),
        )
    treatmentSourceKinds = [
        str(source.sourceKind).upper() for source in treatmentSources
    ]
    chromosomePlans: List[Dict[str, Any]] = []
    for chromosome in chromosomes:
        if str(chromosome) not in chromSizesDict:
            continue
        chromosomeStart, chromosomeEnd = core.getChromRangesJoint(
            bamFiles,
            chromosome,
            chromSizesDict[chromosome],
            samArgs.samThreads,
            samArgs.samFlagExclude,
            sourceKinds=treatmentSourceKinds,
        )
        chromosomeStart = max(0, (chromosomeStart - (chromosomeStart % intervalSizeBP)))
        chromosomeEnd = max(0, (chromosomeEnd - (chromosomeEnd % intervalSizeBP)))
        numIntervals = (
            ((chromosomeEnd - chromosomeStart) + intervalSizeBP) - 1
        ) // intervalSizeBP
        chromosomePlans.append(
            {
                "chromosome": str(chromosome),
                "start": int(chromosomeStart),
                "end": int(chromosomeEnd),
                "numIntervals": int(numIntervals),
            }
        )

    if chromosomePlans:
        for file_ in os.listdir("."):
            if file_.startswith(f"consenrichOutput_{experimentName}") and (
                file_.endswith(".bedGraph")
                or file_.endswith(".narrowPeak")
                or file_.endswith(".broadPeak")
                or file_.endswith(".gappedPeak")
                or file_.endswith(".narrowPeak.json")
                or file_.endswith(".gappedPeak.json")
                or file_.endswith(".narrowPeak.nested_rocco_subproblems.jsonl")
                or file_.endswith(".gappedPeak.nested_rocco_subproblems.jsonl")
            ):
                logger.warning(f"Overwriting: {file_}")
                os.remove(file_)

    trendNumBasis_ = (
        60
        if observationArgs.trendNumBasis is None
        else int(observationArgs.trendNumBasis)
    )
    trendMinObsPerBasis_ = (
        25.0
        if observationArgs.trendMinObsPerBasis is None
        else float(observationArgs.trendMinObsPerBasis)
    )
    trendMinEdf_ = (
        3.0
        if observationArgs.trendMinEdf is None
        else float(observationArgs.trendMinEdf)
    )
    trendMaxEdf_ = (
        None
        if observationArgs.trendMaxEdf is None
        else float(observationArgs.trendMaxEdf)
    )
    trendLambdaMin_ = (
        1.0e-6
        if observationArgs.trendLambdaMin is None
        else float(observationArgs.trendLambdaMin)
    )
    trendLambdaMax_ = (
        1.0e6
        if observationArgs.trendLambdaMax is None
        else float(observationArgs.trendLambdaMax)
    )
    trendLambdaGridSize_ = (
        41
        if observationArgs.trendLambdaGridSize is None
        else int(observationArgs.trendLambdaGridSize)
    )
    samplingIters_ = (
        25_000
        if observationArgs.samplingIters is None
        else int(observationArgs.samplingIters)
    )

    # Negative process bounds are resolved after MUNC construction.
    if observationArgs.maxR < 0.0:
        maxR_ = 1e4
    if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
        minQ_ = 0.0
        maxQ_ = 1e4

    sf: np.ndarray | None = None
    pooledMuncCache = tempfile.TemporaryDirectory(
        prefix=f"consenrich_{experimentName}_munc_"
    )
    transformedMatrixCachePaths: Dict[str, str] = {}
    countModelVarianceFloorCachePaths: Dict[str, str] = {}
    countNoiseFloorSummaryByChromosome: Dict[str, dict[str, Any]] = {}
    muncResidualBackgroundCachePaths: Dict[str, str] = {}
    muncPriorMeanCachePaths: Dict[str, str] = {}
    muncLocalEvidenceCachePaths: Dict[str, str] = {}
    pooledBlockMeansParts: list[np.ndarray] = []
    pooledBlockVarsParts: list[np.ndarray] = []
    pooledBlockCovariatesParts: list[np.ndarray] = []
    pooledBlockLogVarianceNoiseParts: list[np.ndarray] = []
    pooledSampleIndexParts: list[np.ndarray] = []
    pooledChromIndexParts: list[np.ndarray] = []
    pooledBlockStartsParts: list[np.ndarray] = []
    pooledTrendWeightsParts: list[np.ndarray] = []
    pooledNuLTauWeightSum = 0.0
    pooledNuLWeightSum = 0.0
    useReplicateVarianceScale = bool(
        getattr(
            observationArgs,
            "useReplicateVarianceScale",
            constants.OBSERVATION_DEFAULT_USE_REPLICATE_VARIANCE_SCALE,
        )
    )
    useCountNoiseFloor = bool(
        getattr(
            observationArgs,
            "useCountNoiseFloor",
            constants.OBSERVATION_DEFAULT_USE_COUNT_NOISE_FLOOR,
        )
    )
    countNoiseFloorSourceKinds = [
        str(source.sourceKind).upper()
        for source in (
            treatmentSources + (controlSources if controlsPresent else [])
        )
    ]
    countNoiseFloorCountModes = list(
        treatmentCountModes + (controlCountModes if controlsPresent else [])
    )
    countNoiseFloorBedgraphSkippedTracks = sum(
        1
        for sourceKind in countNoiseFloorSourceKinds
        if sourceKind == constants.BEDGRAPH_SOURCE_KIND
    )
    countNoiseFloorStaticSummary: dict[str, Any] = {
        "enabled": bool(useCountNoiseFloor),
        "model": "rawCountMassDeltaMethod" if useCountNoiseFloor else "disabled",
        "countModes": countNoiseFloorCountModes,
        "sourceKinds": countNoiseFloorSourceKinds,
        "bedgraphSkippedTracks": int(countNoiseFloorBedgraphSkippedTracks),
    }

    def _smoothRowsByFragmentLength(
        chromosome: str,
        matrix: np.ndarray,
        *,
        varianceLike: bool = False,
        logSmoothing: bool = True,
    ) -> np.ndarray:
        inputArr = np.ascontiguousarray(matrix, dtype=np.float32)
        if not bool(smoothToFraglen_):
            return inputArr
        if len(characteristicFragmentLengthsTreatment) != int(inputArr.shape[0]):
            raise RuntimeError(
                "fragment-length smoothing requires one characteristic fragment "
                "length per treatment track"
            )
        intervalCount = int(inputArr.shape[1])
        smoothedArr = np.empty_like(inputArr)
        centerIndex = np.arange(intervalCount, dtype=np.int64)
        smoothWindowCache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        smoothingWindows: list[int] = []
        for sampleIndex, fragmentLengthBP in enumerate(
            characteristicFragmentLengthsTreatment
        ):
            fragmentIntervals = int(float(fragmentLengthBP) // float(intervalSizeBP))
            windowIntervals = max(1, fragmentIntervals + 1)
            smoothingWindows.append(windowIntervals)
            rowArr = inputArr[sampleIndex, :]
            if bool(varianceLike):
                rowNaN = np.isnan(rowArr)
                if np.all(rowNaN):
                    smoothedArr[sampleIndex, :] = rowArr
                    continue
                if np.any(rowNaN):
                    raise RuntimeError(
                        "fragment smoothing for variance floors requires finite "
                        "or all-NaN rows"
                    )
            if windowIntervals <= 1:
                smoothedArr[sampleIndex, :] = rowArr
                continue
            cachedWindow = smoothWindowCache.get(windowIntervals)
            if cachedWindow is None:
                halfWindow = windowIntervals // 2
                startsArrSmooth = np.maximum(0, centerIndex - halfWindow)
                endsArrSmooth = np.minimum(
                    intervalCount,
                    startsArrSmooth + windowIntervals,
                )
                startsArrSmooth = np.maximum(
                    0,
                    endsArrSmooth - windowIntervals,
                )
                cachedWindow = (
                    startsArrSmooth.astype(np.int64, copy=False),
                    endsArrSmooth.astype(np.int64, copy=False),
                )
                smoothWindowCache[windowIntervals] = cachedWindow
            startsArrSmooth, endsArrSmooth = cachedWindow
            prefixArr = np.empty(intervalCount + 1, dtype=np.float64)
            prefixArr[0] = 0.0
            prefixArr[1:] = np.cumsum(
                rowArr,
                dtype=np.float64,
            )
            denomArr = np.maximum(1, endsArrSmooth - startsArrSmooth)
            divisorArr = (
                denomArr * denomArr
                if bool(varianceLike)
                else denomArr
            )
            smoothedArr[sampleIndex, :] = (
                (prefixArr[endsArrSmooth] - prefixArr[startsArrSmooth]) / divisorArr
            ).astype(np.float32)
        positiveWindows = [int(value) for value in smoothingWindows if int(value) > 1]
        if bool(logSmoothing) and positiveWindows:
            logger.info(
                "fragment smoothing %s windowIntervals[min,median,max]="
                "[%d,%.6g,%d]",
                chromosome,
                int(min(positiveWindows)),
                float(np.median(np.asarray(positiveWindows, dtype=np.float64))),
                int(max(positiveWindows)),
            )
        return np.ascontiguousarray(smoothedArr, dtype=np.float32)

    muncEBPriorWarmupECMIters = int(
        getattr(
            observationArgs,
            "muncEBPriorWarmupECMIters",
            constants.OBSERVATION_DEFAULT_MUNC_EB_PRIOR_WARMUP_ECM_ITERS,
        )
    )
    if muncEBPriorWarmupECMIters <= 0:
        raise ValueError(
            "observationParams.muncEBPrior.warmupECMIters must be positive"
        )
    muncEBPriorWarmupOuterPasses = int(
        getattr(
            observationArgs,
            "muncEBPriorWarmupOuterPasses",
            constants.OBSERVATION_DEFAULT_MUNC_EB_PRIOR_WARMUP_OUTER_PASSES,
        )
    )
    if muncEBPriorWarmupOuterPasses <= 0:
        raise ValueError(
            "observationParams.muncEBPrior.warmupOuterPasses must be positive"
        )
    muncSeedWeightPasses_ = int(
        getattr(
            observationArgs,
            "muncSeedWeightPasses",
            constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_PASSES,
        )
    )
    if muncSeedWeightPasses_ < constants.OBSERVATION_MIN_MUNC_SEED_WEIGHT_PASSES:
        raise ValueError(
            "observationParams.muncSeedWeight.passes must be at least "
            f"{constants.OBSERVATION_MIN_MUNC_SEED_WEIGHT_PASSES}"
        )
    muncSeedProcessMinQ_ = float(
        getattr(
            observationArgs,
            "muncSeedProcessMinQ",
            constants.OBSERVATION_DEFAULT_MUNC_SEED_PROCESS_MIN_Q,
        )
    )
    if not np.isfinite(muncSeedProcessMinQ_) or muncSeedProcessMinQ_ <= 0.0:
        raise ValueError(
            "observationParams.muncSeedProcess.minQ must be finite and positive"
        )
    muncSeedProcessMaxQ_ = float(
        getattr(
            observationArgs,
            "muncSeedProcessMaxQ",
            constants.OBSERVATION_DEFAULT_MUNC_SEED_PROCESS_MAX_Q,
        )
    )
    if np.isnan(muncSeedProcessMaxQ_):
        raise ValueError("observationParams.muncSeedProcess.maxQ must not be NaN")
    if muncSeedProcessMaxQ_ >= 0.0 and muncSeedProcessMaxQ_ < muncSeedProcessMinQ_:
        raise ValueError(
            "observationParams.muncSeedProcess.maxQ must be negative or at least minQ"
        )
    muncEBPriorGUncertaintyMode = core._normalizeMuncEBPriorGUncertaintyMode(
        getattr(
            observationArgs,
            "muncEBPriorGUncertaintyMode",
            constants.OBSERVATION_DEFAULT_MUNC_EB_PRIOR_G_UNCERTAINTY_MODE,
        )
    )
    muncCovariatesEnabled = bool(
        getattr(
            observationArgs,
            "muncCovariatesEnabled",
            constants.OBSERVATION_DEFAULT_MUNC_COVARIATES_ENABLED,
        )
    )
    muncCovariatesMode = getattr(
        observationArgs,
        "muncCovariatesMode",
        constants.OBSERVATION_DEFAULT_MUNC_COVARIATES_MODE,
    )
    if muncCovariatesEnabled:
        covariateModeKey = (
            str(muncCovariatesMode).strip().replace("-", "").replace("_", "").lower()
        )
        supportedCovariateModeByKey = {
            mode.replace("-", "").replace("_", "").lower(): mode
            for mode in constants.MUNC_SUPPORTED_COVARIATE_MODES
        }
        if covariateModeKey not in supportedCovariateModeByKey:
            raise ValueError(
                "Unsupported observationParams.muncCovariates.mode "
                f"{muncCovariatesMode!r}."
            )
        muncCovariatesMode = supportedCovariateModeByKey[covariateModeKey]
    muncCovariateFeatureConfig = getattr(
        observationArgs,
        "muncCovariatesFeatures",
        constants.OBSERVATION_DEFAULT_MUNC_COVARIATE_FEATURES,
    )
    muncCovariateRawFeatures = resolve_genome_covariate_feature_config(
        muncCovariateFeatureConfig,
        default_features=constants.OBSERVATION_DEFAULT_MUNC_COVARIATE_FEATURES,
        config_name="observationParams.muncCovariates.features",
    )
    genomeCovariateCache = None
    if muncCovariatesEnabled:
        if not getattr(genomeArgs, "genomeCovariateCacheDir", None):
            raise ValueError(
                "`genomeParams.genomeCovariateCacheDir` is required when "
                "`observationParams.muncCovariates.enabled` is true."
            )
        if not muncCovariateRawFeatures:
            raise ValueError(
                "`observationParams.muncCovariates.features` must select at least "
                "one feature when MUNC covariates are enabled."
            )
        genomeCovariateCache = ConsenrichGenomeCovariateCache(
            genomeArgs.genomeCovariateCacheDir,
            interval_size_bp=int(intervalSizeBP),
            requested_chromosomes=tuple(
                str(chromPlan["chromosome"]) for chromPlan in chromosomePlans
            ),
        )
        muncCovariateRawFeatures = resolve_genome_covariate_feature_config(
            muncCovariateFeatureConfig,
            default_features=constants.OBSERVATION_DEFAULT_MUNC_COVARIATE_FEATURES,
            available_features=genomeCovariateCache.features,
            config_name="observationParams.muncCovariates.features",
        )
        if not muncCovariateRawFeatures:
            raise ValueError(
                "`observationParams.muncCovariates.features` must select at least "
                "one feature when MUNC covariates are enabled."
            )
        genomeCovariateCache.validate_request(
            required_features=muncCovariateRawFeatures,
            interval_size_bp=int(intervalSizeBP),
            required_features_label="requested MUNC features",
        )
    muncCovariateFeatureNames = tuple(
        "gc_dev" if str(name) == "gc" else str(name)
        for name in muncCovariateRawFeatures
    )
    if muncCovariatesEnabled:
        logger.info(
            "MUNC genomic covariates enabled: mode=%s features=%s cache=%s",
            muncCovariatesMode,
            ",".join(muncCovariateFeatureNames),
            genomeArgs.genomeCovariateCacheDir,
        )

    def _prepareMuncCovariateTrack(rawCovariates: np.ndarray) -> np.ndarray:
        prepared = np.asarray(rawCovariates, dtype=np.float32).copy()
        if prepared.ndim != 2 or prepared.shape[1] != len(muncCovariateRawFeatures):
            raise ValueError("genome covariate track has an unexpected shape")
        for featureIndex, featureName in enumerate(muncCovariateRawFeatures):
            col = prepared[:, featureIndex].astype(np.float64, copy=False)
            if str(featureName) == "gc":
                finite = col[np.isfinite(col)]
                medianGc = float(np.median(finite)) if finite.size else 0.0
                col = np.abs(col - medianGc)
            else:
                col = np.maximum(col, 0.0)
            prepared[:, featureIndex] = col.astype(np.float32, copy=False)
        return prepared

    def _getChromMuncCovariates(
        chromosome: str,
        chromosomeStart: int,
        chromosomeEnd: int,
        numIntervals: int,
    ) -> np.ndarray | None:
        if not muncCovariatesEnabled or genomeCovariateCache is None:
            return None
        try:
            raw = genomeCovariateCache.fetch(
                chromosome,
                start=int(chromosomeStart),
                end=int(chromosomeEnd),
                feature_names=muncCovariateRawFeatures,
                interval_size_bp=int(intervalSizeBP),
            )
        except KeyError as exc:
            raise ValueError(
                f"MUNC genomic covariates: chromosome {chromosome} is missing from "
                "the cache; "
                "disable observationParams.muncCovariates or rebuild the cache with "
                "all requested chromosomes."
            ) from exc
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[0] != int(numIntervals):
            raise ValueError(
                "MUNC genomic covariates: cache for "
                f"{chromosome} returned {int(raw.shape[0])} intervals, "
                f"expected {int(numIntervals)}"
            )
        return _prepareMuncCovariateTrack(raw)

    def _blockCovariateMeans(
        covariates: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        covArr = np.asarray(covariates, dtype=np.float64)
        startsArr = np.asarray(starts, dtype=np.int64).ravel()
        endsArr = np.asarray(ends, dtype=np.int64).ravel()
        if startsArr.shape != endsArr.shape:
            raise ValueError("block starts and ends must align")
        nBlocks = int(startsArr.size)
        nFeatures = int(covArr.shape[1])
        if nBlocks == 0:
            return np.empty((0, nFeatures), dtype=np.float32)
        startsClip = np.clip(startsArr, 0, covArr.shape[0]).astype(np.int64)
        endsClip = np.clip(endsArr, 0, covArr.shape[0]).astype(np.int64)
        validSpan = endsClip > startsClip
        finite = np.isfinite(covArr)
        values = np.where(finite, covArr, 0.0)
        cumValues = np.vstack(
            (np.zeros((1, nFeatures), dtype=np.float64), np.cumsum(values, axis=0))
        )
        cumCounts = np.vstack(
            (
                np.zeros((1, nFeatures), dtype=np.float64),
                np.cumsum(finite.astype(np.float64), axis=0),
            )
        )
        sums = cumValues[endsClip, :] - cumValues[startsClip, :]
        counts = cumCounts[endsClip, :] - cumCounts[startsClip, :]
        out = np.full((nBlocks, nFeatures), np.nan, dtype=np.float64)
        np.divide(sums, counts, out=out, where=(counts > 0.0) & validSpan[:, None])
        finiteOut = np.isfinite(out)
        out[finiteOut & (out < 0.0)] = 0.0
        return out.astype(np.float32, copy=False)

    def _inverseMuncTrendPredictor(predictor: np.ndarray) -> np.ndarray:
        arr = np.asarray(predictor, dtype=np.float64)
        return np.sign(arr) * np.expm1(np.abs(arr))

    def _getChromBlacklistMask(chromosome: str, intervals: np.ndarray) -> np.ndarray:
        if not genomeArgs.blacklistFile or len(intervals) < 2:
            return np.zeros(len(intervals), dtype=np.uint8)
        mask = core.getBedMask(chromosome, genomeArgs.blacklistFile, intervals)
        return np.asarray(mask, dtype=np.uint8)

    def _countAndTransformChromosomeMatrix(
        c_: int,
        chromPlan: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        nonlocal backgroundBlockSizeBP_
        nonlocal dependenceContextBP_, dependenceSpanIntervals_, sf

        chromosome = str(chromPlan["chromosome"])
        chromosomeStart = int(chromPlan["start"])
        chromosomeEnd = int(chromPlan["end"])
        numIntervals = int(chromPlan["numIntervals"])
        intervals = np.arange(chromosomeStart, chromosomeEnd, intervalSizeBP)
        chromMat: np.ndarray = np.empty((numSamples, numIntervals), dtype=np.float32)
        rawNoiseMassMat: np.ndarray | None = None
        countModelVarianceFloorMat = (
            np.full(
                (numSamples, numIntervals),
                np.nan,
                dtype=np.float32,
            )
            if useCountNoiseFloor
            else None
        )

        if controlsPresent:
            for j_, (bamA, bamB) in enumerate(zip(bamFiles, bamFilesControl)):
                countStart = time.perf_counter()
                logger.info(
                    "counting.start %s sample=%d/%d treatment=%s control=%s",
                    chromosome,
                    int(j_ + 1),
                    int(numSamples),
                    bamA,
                    bamB,
                )
                pairReadResult = core.readSegments(
                    [
                        treatmentSources[j_],
                        controlSources[j_],
                    ],
                    chromosome,
                    chromosomeStart,
                    chromosomeEnd,
                    intervalSizeBP,
                    [
                        readLengthsBamFiles[j_],
                        readLengthsControlBamFiles[j_],
                    ],
                    [treatScaleFactors[j_], controlScaleFactors[j_]],
                    samArgs.oneReadPerBin,
                    samArgs.samThreads,
                    samArgs.samFlagExclude,
                    bamInputMode=samArgs.bamInputMode,
                    defaultCountMode=samArgs.defaultCountMode,
                    defaultFragmentCountMode=scArgs.defaultCountMode,
                    shiftForward5p=samArgs.shiftForward5p,
                    shiftReverse5p=samArgs.shiftReverse5p,
                    extendFrom5pBP=[
                        countExtendFrom5pBPTreatment[j_],
                        countExtendFrom5pBPControl[j_],
                    ],
                    maxInsertSize=samArgs.maxInsertSize,
                    inferFragmentLength=samArgs.inferFragmentLength,
                    minMappingQuality=samArgs.minMappingQuality,
                    minTemplateLength=samArgs.minTemplateLength,
                    returnRawNoiseMass=useCountNoiseFloor,
                )
                pairRawNoiseMass = None
                if useCountNoiseFloor:
                    pairMatrix = pairReadResult.counts
                    pairRawNoiseMass = pairReadResult.rawNoiseMass
                else:
                    pairMatrix = pairReadResult
                if useCountNoiseFloor:
                    if countModelVarianceFloorMat is None:
                        raise RuntimeError("count floor matrix missing")
                    if pairRawNoiseMass is None:
                        raise RuntimeError("raw noise mass matrix missing")
                    countModelVarianceFloorMat[j_, :] = (
                        _combineCountModelVarianceFloors(
                            _countModelVarianceFloorForScaledCounts(
                                pairMatrix[0, :],
                                treatScaleFactors[j_],
                                countingArgs,
                                rawNoiseMass=pairRawNoiseMass[0, :],
                                countMode=treatmentCountModes[j_],
                                countModelSource=_sourceUsesCountModelFloor(
                                    treatmentSources[j_]
                                ),
                            ),
                            _countModelVarianceFloorForScaledCounts(
                                pairMatrix[1, :],
                                controlScaleFactors[j_],
                                countingArgs,
                                rawNoiseMass=pairRawNoiseMass[1, :],
                                countMode=controlCountModes[j_],
                                countModelSource=_sourceUsesCountModelFloor(
                                    controlSources[j_]
                                ),
                            ),
                        ).astype(np.float32, copy=False)
                    )
                cconsenrich.cTransformWithInputInto(
                    pairMatrix[0, :],
                    pairMatrix[1, :],
                    chromMat[j_, :],
                    logOffset=countingArgs.logOffset,
                    logMult=countingArgs.logMult,
                    mode=countingArgs.transformMethod,
                    inputOffset=countingArgs.transformInputOffset,
                    inputScale=countingArgs.transformInputScale,
                    outputScale=countingArgs.transformOutputScale,
                    outputOffset=countingArgs.transformOutputOffset,
                    shape=countingArgs.transformShape,
                )
                logger.info(
                    "counting.done %s sample=%d/%d elapsed=%.3fs",
                    chromosome,
                    int(j_ + 1),
                    int(numSamples),
                    time.perf_counter() - countStart,
                )
        else:
            countStart = time.perf_counter()
            logger.info(
                "counting.start %s samples=%d intervals=%d samThreads=%d",
                chromosome,
                int(numSamples),
                int(numIntervals),
                int(samArgs.samThreads),
            )
            readResult = core.readSegments(
                treatmentSources,
                chromosome,
                chromosomeStart,
                chromosomeEnd,
                intervalSizeBP,
                readLengthsBamFiles,
                np.ones(numSamples) if waitForMatrix else scaleFactors,
                samArgs.oneReadPerBin,
                samArgs.samThreads,
                samArgs.samFlagExclude,
                bamInputMode=samArgs.bamInputMode,
                defaultCountMode=samArgs.defaultCountMode,
                defaultFragmentCountMode=scArgs.defaultCountMode,
                shiftForward5p=samArgs.shiftForward5p,
                shiftReverse5p=samArgs.shiftReverse5p,
                extendFrom5pBP=countExtendFrom5pBPTreatment,
                maxInsertSize=samArgs.maxInsertSize,
                inferFragmentLength=samArgs.inferFragmentLength,
                minMappingQuality=samArgs.minMappingQuality,
                minTemplateLength=samArgs.minTemplateLength,
                returnRawNoiseMass=useCountNoiseFloor,
            )
            if useCountNoiseFloor:
                chromMat = readResult.counts
                rawNoiseMassMat = readResult.rawNoiseMass
            else:
                chromMat = readResult
            logger.info(
                "counting.done %s samples=%d elapsed=%.3fs",
                chromosome,
                int(numSamples),
                time.perf_counter() - countStart,
            )

        if waitForMatrix:
            if sf is None:
                sf = cconsenrich.cSF(chromMat)
                logger.info(
                    "`countingParams.normMethod=SF` --> calculating scaling factors\n%s\n",
                    sf,
                )
                _checkSF(sf, logger)
            np.multiply(chromMat, sf[:, None], out=chromMat)

        if useCountNoiseFloor and not controlsPresent:
            countModelScaleFactors = sf if waitForMatrix else scaleFactors
            countModelVarianceFloorMat = _countModelFloorMatrixForScaledCounts(
                chromMat,
                countModelScaleFactors,
                treatmentSources,
                countingArgs,
                rawNoiseMassMatrix=rawNoiseMassMat,
                countModes=treatmentCountModes,
            )

        if useCountNoiseFloor:
            if countModelVarianceFloorMat is None:
                raise RuntimeError("count floor matrix missing")
            floorSummary = _summarizeCountModelVarianceFloor(countModelVarianceFloorMat)
            countNoiseFloorSummaryByChromosome[chromosome] = {
                **countNoiseFloorStaticSummary,
                **floorSummary,
            }
            countNoiseDerivedVarianceFloor = _countModelVarianceFloorScalar(
                countModelVarianceFloorMat,
            )
            logger.info(
                "count noise floor-derived variance floor %s value=%s finite=%d "
                "positive=%d q05=%s min=%s median=%s p95=%s max=%s",
                chromosome,
                _fmtDiagnosticFloat(countNoiseDerivedVarianceFloor),
                int(floorSummary.get("finite", 0)),
                int(floorSummary.get("positive", 0)),
                _fmtDiagnosticFloat(floorSummary.get("q05")),
                _fmtDiagnosticFloat(floorSummary.get("min")),
                _fmtDiagnosticFloat(floorSummary.get("median")),
                _fmtDiagnosticFloat(floorSummary.get("p95")),
                _fmtDiagnosticFloat(floorSummary.get("max")),
            )
        else:
            countNoiseFloorSummaryByChromosome[chromosome] = dict(
                countNoiseFloorStaticSummary
            )
            logger.info("count noise floor disabled %s", chromosome)

        def _transformTrack(j: int) -> int:
            cconsenrich.cTransformInPlace(
                chromMat[j, :],
                verbose=args.verbose2,
                logOffset=countingArgs.logOffset,
                logMult=countingArgs.logMult,
                mode=countingArgs.transformMethod,
                inputOffset=countingArgs.transformInputOffset,
                inputScale=countingArgs.transformInputScale,
                outputScale=countingArgs.transformOutputScale,
                outputOffset=countingArgs.transformOutputOffset,
                shape=countingArgs.transformShape,
            )
            return j

        if controlsPresent:
            logger.info(
                "Skipping ordinary count transform: treatment/control tracks "
                "were already transformed as log-ratios.",
            )
        else:
            transformStart = time.perf_counter()
            transformWorkers = io_helpers._getSmallWorkerCount(
                numSamples,
                maxWorkers=4,
            )
            useParallelTransform = (
                numSamples >= 4 and chromMat.shape[1] >= 5000 and transformWorkers > 1
            )
            if useParallelTransform:
                logger.info(
                    "transform matrix: using ThreadPool with %d workers (numSamples=%d, numIntervals=%d).",
                    int(transformWorkers),
                    int(numSamples),
                    int(chromMat.shape[1]),
                )
                with ThreadPool(processes=int(transformWorkers)) as pool:
                    for _ in pool.imap(_transformTrack, range(numSamples)):
                        pass
            else:
                for j in range(numSamples):
                    _transformTrack(j)
            logger.info(
                "transform.done %s samples=%d elapsed=%.3fs",
                chromosome,
                int(numSamples),
                time.perf_counter() - transformStart,
            )

        centerMBEnabled, _ = _resolveCenterMBStatus(
            countingArgs,
            controlsPresent=controlsPresent,
        )
        if centerMBEnabled:
            centerStart = time.perf_counter()
            centerStats = core.centerMBInPlace(
                chromMat,
                intervalSizeBP=intervalSizeBP,
                centerMBMethod=countingArgs.centerMBMethod,
            )
            logger.info(
                "centerMB.done\t%s\tsamples=%d\tapplied=%d\t"
                "meanTrackValBeforeCenterMB=%.6g\t"
                "meanTrackValAfterCenterMB=%.6g\t"
                "stdTrackValBeforeCenterMB=%.6g\t"
                "stdTrackValAfterCenterMB=%.6g\telapsed=%.3fs",
                chromosome,
                int(numSamples),
                int(centerStats.get("applied_tracks", 0)),
                float(centerStats["meanTrackValBeforeCenterMB"]),
                float(centerStats["meanTrackValAfterCenterMB"]),
                float(centerStats["stdTrackValBeforeCenterMB"]),
                float(centerStats["stdTrackValAfterCenterMB"]),
                time.perf_counter() - centerStart,
            )

        return (
            intervals,
            np.ascontiguousarray(chromMat, dtype=np.float32),
            (
                None
                if countModelVarianceFloorMat is None
                else np.ascontiguousarray(countModelVarianceFloorMat, dtype=np.float32)
            ),
        )

    def _summarizeFiniteArray(values: np.ndarray) -> dict[str, float | int]:
        arr = np.asarray(values)
        finiteMask = np.isfinite(arr)
        finiteCount = int(np.count_nonzero(finiteMask))
        if finiteCount == 0:
            return {
                "count": 0,
                "min": float("nan"),
                "p05": float("nan"),
                "median": float("nan"),
                "mean": float("nan"),
                "p95": float("nan"),
                "max": float("nan"),
                "frac_negative": float("nan"),
                "frac_zero": float("nan"),
                "frac_positive": float("nan"),
            }
        fracNegative = float(np.count_nonzero(finiteMask & (arr < 0.0))) / float(
            finiteCount
        )
        fracZero = float(np.count_nonzero(finiteMask & (arr == 0.0))) / float(
            finiteCount
        )
        fracPositive = float(np.count_nonzero(finiteMask & (arr > 0.0))) / float(
            finiteCount
        )
        flat = arr.reshape(-1)
        maxSummaryValues = 1_000_000
        if flat.size > maxSummaryValues:
            idx = np.linspace(0, flat.size - 1, maxSummaryValues, dtype=np.int64)
            summary = np.asarray(flat[idx], dtype=np.float64)
            summary = summary[np.isfinite(summary)]
            if summary.size == 0:
                summary = np.asarray(flat[finiteMask.reshape(-1)], dtype=np.float64)
        else:
            summary = np.asarray(flat[finiteMask.reshape(-1)], dtype=np.float64)
        q05, q50, q95 = np.quantile(summary, [0.05, 0.5, 0.95])
        return {
            "count": finiteCount,
            "min": float(np.min(summary)),
            "p05": float(q05),
            "median": float(q50),
            "mean": float(np.mean(summary, dtype=np.float64)),
            "p95": float(q95),
            "max": float(np.max(summary)),
            "frac_negative": fracNegative,
            "frac_zero": fracZero,
            "frac_positive": fracPositive,
        }

    def _coarseMuncResidualizationInvVar(chromMat: np.ndarray) -> np.ndarray:
        arr = np.asarray(chromMat, dtype=np.float32)
        finiteMask = np.isfinite(arr)
        sampleVariances = np.empty(int(arr.shape[0]), dtype=np.float64)
        for j in range(int(arr.shape[0])):
            row = np.asarray(arr[j, finiteMask[j, :]], dtype=np.float64)
            if row.size >= 2:
                rowMedian = float(np.median(row))
                rowMad = float(np.median(np.abs(row - rowMedian)))
                rowVariance = float((1.4826 * rowMad) ** 2)
                if not np.isfinite(rowVariance) or rowVariance <= 0.0:
                    rowVariance = float(np.var(row, dtype=np.float64))
            else:
                rowVariance = float("nan")
            sampleVariances[j] = rowVariance

        finiteVariances = sampleVariances[
            np.isfinite(sampleVariances) & (sampleVariances > 0.0)
        ]
        fallbackVariance = (
            float(np.median(finiteVariances)) if finiteVariances.size else 1.0
        )
        fallbackVariance = max(fallbackVariance, 1.0e-4)
        sampleVariances = np.where(
            np.isfinite(sampleVariances) & (sampleVariances > 0.0),
            sampleVariances,
            fallbackVariance,
        )
        invSampleVariances = 1.0 / np.maximum(sampleVariances, 1.0e-4)
        invVarMatrix = np.broadcast_to(
            invSampleVariances[:, None],
            arr.shape,
        ).astype(np.float32, copy=True)
        invVarMatrix[~finiteMask] = np.float32(0.0)
        return np.ascontiguousarray(invVarMatrix, dtype=np.float32)

    def _coarseMuncResidualizationMunc(
        chromMat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        invVarMatrix = _coarseMuncResidualizationInvVar(chromMat)
        finiteMask = invVarMatrix > 0.0
        coarseMunc = np.full(invVarMatrix.shape, 1.0, dtype=np.float32)
        coarseMunc[finiteMask] = np.maximum(
            (1.0 / invVarMatrix[finiteMask]) - float(pad_),
            1.0e-4,
        ).astype(np.float32, copy=False)
        return np.ascontiguousarray(coarseMunc, dtype=np.float32), finiteMask

    def _estimateMuncResidualizationBackground(
        chromosome: str,
        chromMat: np.ndarray,
    ) -> tuple[np.ndarray, str, int]:
        intervalCount = int(chromMat.shape[1])
        if not bool(fitArgs.fitBackground):
            return (
                np.zeros(intervalCount, dtype=np.float32),
                "disabled_fitBackground_false",
                0,
            )

        blockLenIntervals = _resolveRuntimeBackgroundBlockLen(
            dependenceSpanIntervals_,
            int(backgroundBlockSizeBP_),
            intervalSizeBP,
            fitArgs.ECM_backgroundLengthScaleMultiplier,
        )
        helper = getattr(core, "estimateProvisionalBackground", None)
        if callable(helper):
            arr = np.asarray(chromMat, dtype=np.float32)
            finiteValueMask = np.isfinite(arr)
            matrixMunc, observationMask = _coarseMuncResidualizationMunc(arr)
            matrixData = np.where(finiteValueMask, arr, 0.0).astype(
                np.float32,
                copy=False,
            )
            result = helper(
                np.ascontiguousarray(matrixData, dtype=np.float32),
                matrixMunc,
                blockLenIntervals=int(blockLenIntervals),
                pad=float(pad_),
                observationMask=observationMask,
                observationPrecisionMultiplierMin=(
                    observationArgs.precisionMultiplierMin
                ),
                observationPrecisionMultiplierMax=(
                    observationArgs.precisionMultiplierMax
                ),
                backgroundSmoothness=float(fitArgs.ECM_backgroundSmoothness),
                zeroCenterBackground=bool(fitArgs.ECM_zeroCenterBackground),
                useNonnegativeBackground=bool(fitArgs.useNonnegativeBackground),
                backgroundNegativePenaltyMultiplier=(
                    fitArgs.backgroundNegativePenaltyMultiplier
                ),
                returnDiagnostics=True,
            )
            if isinstance(result, tuple) and len(result) == 2:
                background, helperDiagnostics = result
                helperSource = str(
                    dict(helperDiagnostics).get(
                        "source", "estimateProvisionalBackground"
                    )
                )
            else:
                background = result
                helperSource = "estimateProvisionalBackground"
            backgroundArr = np.asarray(background, dtype=np.float32).reshape(-1)
            if backgroundArr.shape != (intervalCount,):
                raise RuntimeError(
                    "MUNC residualization helper returned background shape "
                    f"{backgroundArr.shape} for {chromosome}; expected {(intervalCount,)}"
                )
            return (
                np.ascontiguousarray(backgroundArr, dtype=np.float32),
                f"core.estimateProvisionalBackground:{helperSource}",
                int(blockLenIntervals),
            )

        arr = np.asarray(chromMat, dtype=np.float32)
        finiteMask = np.isfinite(arr)
        residualMatrix = np.where(
            finiteMask,
            arr,
            0.0,
        ).astype(np.float32, copy=False)
        invVarMatrix = _coarseMuncResidualizationInvVar(arr)
        backgroundArr = core.solveZeroCenteredBackground(
            residualMatrix=np.ascontiguousarray(residualMatrix, dtype=np.float32),
            invVarMatrix=invVarMatrix,
            blockLenIntervals=int(blockLenIntervals),
            backgroundSmoothness=float(fitArgs.ECM_backgroundSmoothness),
            zeroCenter=bool(fitArgs.ECM_zeroCenterBackground),
            useNonnegative=bool(fitArgs.useNonnegativeBackground),
            backgroundNegativePenaltyMultiplier=(
                fitArgs.backgroundNegativePenaltyMultiplier
            ),
        )
        backgroundArr = np.asarray(backgroundArr, dtype=np.float32).reshape(-1)
        if backgroundArr.shape != (intervalCount,):
            raise RuntimeError(
                "MUNC residualization background shape mismatch for "
                f"{chromosome}: expected {(intervalCount,)}, got {backgroundArr.shape}"
            )
        return (
            np.ascontiguousarray(backgroundArr, dtype=np.float32),
            "core.solveZeroCenteredBackground_coarse_weights",
            int(blockLenIntervals),
        )

    def _residualizeMuncInput(
        chromosome: str,
        chromMat: np.ndarray,
        background: np.ndarray,
        *,
        logDiagnostics: bool,
        source: str = "",
        blockLenIntervals: int = 0,
    ) -> np.ndarray:
        backgroundArr = np.asarray(background, dtype=np.float32).reshape(-1)
        if backgroundArr.shape != (int(chromMat.shape[1]),):
            raise RuntimeError(
                "MUNC residualization background shape mismatch for "
                f"{chromosome}: expected {(int(chromMat.shape[1]),)}, got {backgroundArr.shape}"
            )
        residMat = np.ascontiguousarray(
            np.asarray(chromMat, dtype=np.float32) - backgroundArr[None, :],
            dtype=np.float32,
        )
        if not logDiagnostics:
            return residMat

        backgroundStats = _summarizeFiniteArray(backgroundArr)
        rawStats = _summarizeFiniteArray(chromMat)
        residStats = _summarizeFiniteArray(residMat)
        logger.info(
            "MUNC residualization prepass %s source=%s blockLen=%d "
            "g0[min,p05,median,mean,p95,max]=[%s,%s,%s,%s,%s,%s] "
            "g0_fracPositive=%s g0_fracAbsLe1e-3=%s",
            chromosome,
            source,
            int(blockLenIntervals),
            _fmtDiagnosticFloat(backgroundStats["min"]),
            _fmtDiagnosticFloat(backgroundStats["p05"]),
            _fmtDiagnosticFloat(backgroundStats["median"]),
            _fmtDiagnosticFloat(backgroundStats["mean"]),
            _fmtDiagnosticFloat(backgroundStats["p95"]),
            _fmtDiagnosticFloat(backgroundStats["max"]),
            _fmtDiagnosticFloat(backgroundStats["frac_positive"]),
            _fmtDiagnosticFloat(
                float(
                    np.mean(np.abs(backgroundArr[np.isfinite(backgroundArr)]) <= 1.0e-3)
                )
                if int(backgroundStats["count"]) > 0
                else float("nan")
            ),
        )
        logger.info(
            "MUNC residualization value signs %s raw(n=%d,neg=%.4f,zero=%.4f,pos=%.4f,median=%s) "
            "residual(n=%d,neg=%.4f,zero=%.4f,pos=%.4f,median=%s)",
            chromosome,
            int(rawStats["count"]),
            float(rawStats["frac_negative"]),
            float(rawStats["frac_zero"]),
            float(rawStats["frac_positive"]),
            _fmtDiagnosticFloat(rawStats["median"]),
            int(residStats["count"]),
            float(residStats["frac_negative"]),
            float(residStats["frac_zero"]),
            float(residStats["frac_positive"]),
            _fmtDiagnosticFloat(residStats["median"]),
        )
        return residMat

    def _loadMuncResidualizationBackground(
        chromosome: str,
        intervalCount: int,
    ) -> np.ndarray:
        path = muncResidualBackgroundCachePaths.get(chromosome)
        if path is None:
            logger.warning(
                "Missing MUNC residualization background cache for %s; using zero g0.",
                chromosome,
            )
            return np.zeros(int(intervalCount), dtype=np.float32)
        backgroundArr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        backgroundArr = backgroundArr.reshape(-1)
        if backgroundArr.shape != (int(intervalCount),):
            raise RuntimeError(
                "MUNC residualization background cache shape mismatch for "
                f"{chromosome}: expected {(int(intervalCount),)}, got {backgroundArr.shape}"
            )
        return np.ascontiguousarray(backgroundArr, dtype=np.float32)

    def _loadMuncPriorMeanTrack(
        chromosome: str,
        intervalCount: int,
    ) -> np.ndarray:
        path = muncPriorMeanCachePaths.get(chromosome)
        if path is None:
            raise RuntimeError(f"Missing MUNC prior mean cache for {chromosome}")
        meanArr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        meanArr = meanArr.reshape(-1)
        if meanArr.shape != (int(intervalCount),):
            raise RuntimeError(
                "MUNC prior mean cache shape mismatch for "
                f"{chromosome}: expected {(int(intervalCount),)}, got {meanArr.shape}"
            )
        return np.ascontiguousarray(meanArr, dtype=np.float32)

    def _loadMuncLocalEvidenceMatrix(
        chromosome: str,
        intervalCount: int,
    ) -> np.ndarray:
        path = muncLocalEvidenceCachePaths.get(chromosome)
        if path is None:
            raise RuntimeError(f"Missing MUNC local evidence cache for {chromosome}")
        evidenceArr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if evidenceArr.shape != (numSamples, int(intervalCount)):
            raise RuntimeError(
                "MUNC local evidence cache shape mismatch for "
                f"{chromosome}: expected {(numSamples, int(intervalCount))}, "
                f"got {evidenceArr.shape}"
            )
        return np.ascontiguousarray(evidenceArr, dtype=np.float32)

    def _loadCountModelVarianceFloor(
        chromosome: str,
        intervalCount: int,
    ) -> np.ndarray:
        path = countModelVarianceFloorCachePaths.get(chromosome)
        if path is None:
            return np.full((numSamples, int(intervalCount)), np.nan, dtype=np.float32)
        floorArr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if floorArr.shape != (numSamples, int(intervalCount)):
            raise RuntimeError(
                "count-model variance floor cache shape mismatch for "
                f"{chromosome}: expected {(numSamples, int(intervalCount))}, "
                f"got {floorArr.shape}"
            )
        return np.ascontiguousarray(floorArr, dtype=np.float32)

    def _rawMeansForSampledBlocks(
        values: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        valuesArr = np.asarray(values, dtype=np.float64)
        startsArr = np.asarray(starts, dtype=np.int64)
        endsArr = np.asarray(ends, dtype=np.int64)
        valid = (
            np.isfinite(startsArr)
            & np.isfinite(endsArr)
            & (startsArr >= 0)
            & (endsArr > startsArr)
            & (endsArr <= valuesArr.size)
        )
        out = np.full(startsArr.shape, np.nan, dtype=np.float64)
        if not np.any(valid):
            return out
        finiteValues = np.where(np.isfinite(valuesArr), valuesArr, 0.0)
        finiteCounts = np.asarray(np.isfinite(valuesArr), dtype=np.int64)
        prefixSum = np.empty(valuesArr.size + 1, dtype=np.float64)
        prefixCount = np.empty(valuesArr.size + 1, dtype=np.int64)
        prefixSum[0] = 0.0
        prefixCount[0] = 0
        prefixSum[1:] = np.cumsum(finiteValues, dtype=np.float64)
        prefixCount[1:] = np.cumsum(finiteCounts, dtype=np.int64)
        startsValid = startsArr[valid]
        endsValid = endsArr[valid]
        counts = prefixCount[endsValid] - prefixCount[startsValid]
        sums = prefixSum[endsValid] - prefixSum[startsValid]
        localMeans = np.full(startsValid.shape, np.nan, dtype=np.float64)
        nonzero = counts > 0
        localMeans[nonzero] = sums[nonzero] / counts[nonzero]
        out[valid] = localMeans
        return out

    def _logMuncTrendInputSummary(
        chromosome: str,
        label: str,
        blockMeans: np.ndarray,
    ) -> np.ndarray:
        stats = _summarizeFiniteArray(blockMeans)
        logger.info(
            "MUNC trend input %s %s blocks=%d "
            "neg=%.4f zero=%.4f pos=%.4f "
            "mean=%s p05=%s median=%s p95=%s",
            label,
            chromosome,
            int(stats["count"]),
            float(stats["frac_negative"]),
            float(stats["frac_zero"]),
            float(stats["frac_positive"]),
            _fmtDiagnosticFloat(stats["mean"]),
            _fmtDiagnosticFloat(stats["p05"]),
            _fmtDiagnosticFloat(stats["median"]),
            _fmtDiagnosticFloat(stats["p95"]),
        )

    def _ensureSampledDependenceSpan(
        cachedMatrixPaths: Mapping[str, str],
    ) -> None:
        nonlocal correlationLengthDiagnostics_
        nonlocal correlationLengthRow_, dependenceContextBP_, dependenceSpanIntervals_

        if dependenceSpanIntervals_ is not None and dependenceContextBP_ is not None:
            return

        chromNames: list[str] = []
        chromMatrices: list[np.ndarray] = []
        for chromPlan in chromosomePlans:
            chromosome = str(chromPlan["chromosome"])
            cachePath = cachedMatrixPaths.get(chromosome)
            if cachePath is None:
                continue
            chromNames.append(chromosome)
            chromMatrices.append(np.load(cachePath, mmap_mode="r"))

        _logCliPhase(
            "Correlation length",
            "chromosomes=%d windows=%d",
            int(len(chromNames)),
            int(dependenceWindowCount_),
        )
        correlationLengthStart = time.perf_counter()
        depPoint, depLower, depUpper, depDiagnostics = (
            cconsenrich.cchooseDependenceSpan(
                chromNames,
                chromMatrices,
                intervalSizeBP,
                windowBP=int(dependenceWindowBP_),
                windowCount=int(dependenceWindowCount_),
                maxLagBP=int(dependenceMaxLagBP_),
                workingQuantile=float(dependenceWorkingQuantile_),
                bootstrapDraws=int(dependenceBootstrapDraws_),
                randSeed=int(constants.UNCERTAINTY_CALIBRATION_DEFAULT_SEED),
                minWindowCount=int(dependenceMinWindowCount_),
                acfThreshold=float(dependenceAcfPointThreshold_),
                acfSmoothingBP=int(dependenceAcfSmoothingBP_),
                crossingPersistenceBP=int(dependenceCrossingPersistenceBP_),
                minFinitePairs=int(dependenceMinFinitePairs_),
                minFinitePairCoverage=float(dependenceMinFinitePairCoverage_),
            )
        )
        depDiagnostics = dict(depDiagnostics)
        workingSpanBP = float(depDiagnostics["workingSpanBP"])
        if not math.isclose(
            workingSpanBP,
            float(depDiagnostics["fullSampleWorkingSpanBP"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError("workingSpanBP must equal the full-sample KM quantile")
        expectedPoint = int(
            math.ceil(float(depDiagnostics["estimateBP"]) / float(intervalSizeBP))
        )
        if int(depPoint) != expectedPoint:
            raise RuntimeError("correlation-radius tuple and BP diagnostic disagree")
        dependenceSpanIntervals_ = int(
            math.ceil(workingSpanBP / float(intervalSizeBP))
        )
        dependenceContextBP_ = int(
            2 * int(dependenceSpanIntervals_) * int(intervalSizeBP) + 1
        )
        correlationLengthDiagnostics_ = depDiagnostics
        correlationLengthRow_ = _correlationLengthRow(
            intervalSizeBP=intervalSizeBP,
            pointIntervals=int(depPoint),
            contextBP=int(dependenceContextBP_),
            details=depDiagnostics,
        )

        excluded = [str(value) for value in depDiagnostics["chromosomesExcluded"]]
        excludedLabel = "none" if not excluded else ",".join(excluded)
        logger.info(
            "chooseCorrelationLength.finitePairWindows status=%s method=%s "
            "random_seed=%d "
            "inference_scope=%s confidence_interval_method=%s "
            "survival_band_region=[%.6g,%.6g] "
            "survival_band_jump_closure_used=%s "
            "survival_band_jump_closure_count=%d "
            "chromosomes_used=%d chromosomes_excluded=%s "
            "candidate_windows=%d evaluated_candidate_windows=%d "
            "selected_windows=%d requested_windows=%d "
            "selected_adjacencies=%d selected_longest_run=%d "
            "window_bp=%d max_lag_bp=%d threshold=%.6g "
            "acf_smoothing_bp=%d acf_smoothing_bins=%d "
            "crossing_persistence_bp=%d crossing_persistence_bins=%d "
            "min_finite_pairs=%d min_finite_pair_coverage=%.6g "
            "input_rows=%d unique_rows=%d duplicate_rows=%d row_deduplication=%s "
            "finite_pair_minimum_used=%.6g finite_pair_coverage_minimum_used=%.6g "
            "valid_rows_at_crossing_minimum=%s "
            "crossed_windows=%d right_censored_windows=%d "
            "support_censored_windows=%d cap_censored_windows=%d "
            "censor_fraction=%.6g censor_time_bp_minimum=%s "
            "censor_time_bp_median=%s censor_time_bp_maximum=%s "
            "estimate_bp=%.6g lower_bp=%.6g upper_bp=%.6g "
            "working_span_bp=%.6g working_quantile=%.6g "
            "estimate_intervals=%d lower_intervals=%d upper_intervals=%d "
            "working_span_intervals=%d context_bp=%d "
            "dominant_acf_period_bp_median=%s unresolved_period_fraction=%.6g "
            "oscillation_strength=%s post_crossing_revival=%s "
            "bootstrap_method=%s bootstrap_block_length_windows=%d "
            "bootstrap_draws=%d "
            "bootstrap_resolved_median=%d bootstrap_resolved_working=%d "
            "bootstrap_resolved_joint=%d confidence_level=%.6g",
            str(depDiagnostics["status"]),
            str(depDiagnostics["method"]),
            int(depDiagnostics["randomSeed"]),
            str(depDiagnostics["inferenceScope"]),
            str(depDiagnostics["confidenceIntervalMethod"]),
            float(depDiagnostics["survivalBandRegionLower"]),
            float(depDiagnostics["survivalBandRegionUpper"]),
            (
                "true"
                if bool(depDiagnostics["survivalBandJumpClosureUsed"])
                else "false"
            ),
            int(depDiagnostics["survivalBandJumpClosureCount"]),
            int(len(depDiagnostics["chromosomesUsed"])),
            excludedLabel,
            int(depDiagnostics["candidateWindowCount"]),
            int(depDiagnostics["evaluatedCandidateWindowCount"]),
            int(depDiagnostics["selectedWindowCount"]),
            int(depDiagnostics["windowCountRequested"]),
            int(depDiagnostics["selectedAdjacencyCount"]),
            int(depDiagnostics["selectedLongestRun"]),
            int(depDiagnostics["windowBP"]),
            int(depDiagnostics["maxLagBP"]),
            float(depDiagnostics["acfThreshold"]),
            int(depDiagnostics["acfSmoothingBP"]),
            int(depDiagnostics["acfSmoothingBins"]),
            int(depDiagnostics["crossingPersistenceBP"]),
            int(depDiagnostics["crossingPersistenceBins"]),
            int(depDiagnostics["minFinitePairs"]),
            float(depDiagnostics["minFinitePairCoverage"]),
            int(depDiagnostics["inputRowCount"]),
            int(depDiagnostics["uniqueRowCount"]),
            int(depDiagnostics["duplicateRowCount"]),
            str(depDiagnostics["rowDeduplication"]),
            float(depDiagnostics["finitePairMinimumUsed"]),
            float(depDiagnostics["finitePairCoverageMinimumUsed"]),
            _formatOptionalLogValue(depDiagnostics["validRowsAtCrossingMinimum"]),
            int(depDiagnostics["crossedWindowCount"]),
            int(depDiagnostics["rightCensoredWindowCount"]),
            int(depDiagnostics["supportCensoredWindowCount"]),
            int(depDiagnostics["capCensoredWindowCount"]),
            float(depDiagnostics["censorFraction"]),
            _formatOptionalLogValue(depDiagnostics["censorTimeBPMinimum"]),
            _formatOptionalLogValue(depDiagnostics["censorTimeBPMedian"]),
            _formatOptionalLogValue(depDiagnostics["censorTimeBPMaximum"]),
            float(depDiagnostics["estimateBP"]),
            float(depDiagnostics["lowerBP"]),
            float(depDiagnostics["upperBP"]),
            workingSpanBP,
            float(depDiagnostics["workingQuantile"]),
            int(depPoint),
            int(depLower),
            int(depUpper),
            int(dependenceSpanIntervals_),
            int(dependenceContextBP_),
            _formatOptionalLogValue(depDiagnostics["dominantACFPeriodBPMedian"]),
            float(depDiagnostics["unresolvedPeriodFraction"]),
            _formatOptionalLogValue(depDiagnostics["oscillationStrength"]),
            _formatOptionalLogValue(depDiagnostics["postCrossingRevival"]),
            str(depDiagnostics["bootstrapMethod"]),
            int(depDiagnostics["bootstrapBlockLengthWindows"]),
            int(depDiagnostics["bootstrapDrawsRequested"]),
            int(depDiagnostics["bootstrapResolvedMedianDraws"]),
            int(depDiagnostics["bootstrapResolvedWorkingDraws"]),
            int(depDiagnostics["bootstrapResolvedJointDraws"]),
            float(depDiagnostics["confidenceLevel"]),
        )
        censorFraction = float(depDiagnostics["censorFraction"])
        workingQuantile = float(depDiagnostics["workingQuantile"])
        maxLagBP = float(depDiagnostics["maxLagBP"])
        censorThreshold = 1.0 - workingQuantile
        maxLagFraction = workingSpanBP / maxLagBP
        if (
            censorFraction >= censorThreshold
            or maxLagFraction >= 0.8
        ):
            logger.warning(
                "dependence working span has weak upper-tail support: "
                "censorFraction=%.6g censorThreshold=%.6g "
                "workingQuantile=%.6g workingSpanBP=%.6g maxLagBP=%.6g "
                "maxLagFraction=%.6g. Tail-derived span inference may be "
                "weakly supported.",
                censorFraction,
                censorThreshold,
                workingQuantile,
                workingSpanBP,
                maxLagBP,
                maxLagFraction,
            )

        if bool(fitArgs.fitBackground) and int(backgroundBlockSizeBP_) <= 0:
            inferredSpanIntervals = _resolveRuntimeBackgroundBlockLen(
                dependenceSpanIntervals_,
                int(backgroundBlockSizeBP_),
                intervalSizeBP,
                fitArgs.ECM_backgroundLengthScaleMultiplier,
            )
            inferredLambdaFirst, inferredLambdaSecond = (
                core._backgroundPenaltyWeightsFromSpan(
                    blockLenIntervals=int(inferredSpanIntervals),
                    backgroundSmoothness=float(fitArgs.ECM_backgroundSmoothness),
                )
            )
            nominalRoundoffIndex = float(
                np.finfo(np.float64).eps
                * (
                    1.0
                    + 4.0 * float(inferredLambdaFirst)
                    + 16.0 * float(inferredLambdaSecond)
                )
            )
            if nominalRoundoffIndex >= 1.0:
                logger.warning(
                    "inferred roughness span is numerically wide: "
                    "dependenceSpanIntervals=%d lengthScaleMultiplier=%.6g "
                    "resolvedSpanIntervals=%d resolvedSpanBP=%d "
                    "smoothness=%.6g nominalRoundoffIndex=%.6g. "
                    "The penalty scale reaches float64 resolution at unit "
                    "observation precision.",
                    int(dependenceSpanIntervals_),
                    float(fitArgs.ECM_backgroundLengthScaleMultiplier),
                    int(inferredSpanIntervals),
                    int(inferredSpanIntervals * intervalSizeBP),
                    float(fitArgs.ECM_backgroundSmoothness),
                    nominalRoundoffIndex,
                )
        _logCliProgressMilestone(
            "Correlation length: radius_intervals=%d radius_bp=%.6g "
            "working_span_intervals=%d working_span_bp=%.6g elapsed=%.1fs",
            int(depPoint),
            float(depDiagnostics["estimateBP"]),
            int(dependenceSpanIntervals_),
            workingSpanBP,
            time.perf_counter() - correlationLengthStart,
        )

    def _collectPooledMuncBlocks(
        c_: int,
        intervals: np.ndarray,
        chromMat: np.ndarray,
        seedMuncMat: np.ndarray,
        *,
        inputLabel: str = "raw",
        diagnosticRawMat: np.ndarray | None = None,
        chromCovariates: np.ndarray | None = None,
        countModelVarianceFloorMat: np.ndarray | None = None,
    ) -> np.ndarray:
        nonlocal pooledNuLTauWeightSum, pooledNuLWeightSum
        chromosomeName = str(chromosomePlans[c_]["chromosome"])
        muncSizing = core._resolveMuncRuntimeSizing(
            intervalSizeBP=intervalSizeBP,
            dependenceSpanIntervals=dependenceSpanIntervals_,
            muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
            muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
            muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier_,
            muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier_,
        )
        blacklistExcludeMask = _getChromBlacklistMask(chromosomeName, intervals)
        chromArr = np.asarray(chromMat, dtype=np.float32)
        seedMuncArr = np.array(seedMuncMat, dtype=np.float32, copy=True)
        residArr = np.ascontiguousarray(chromArr, dtype=np.float32)
        if chromArr.shape != seedMuncArr.shape:
            raise RuntimeError("seed smoother matrices must align")
        intervalCount = int(residArr.shape[1])
        if intervalCount < 2:
            raise RuntimeError(
                f"MUNC seed ECM needs at least two intervals for {chromosomeName}"
            )
        if not np.all(np.isfinite(seedMuncArr)):
            raise RuntimeError(
                f"seed MUNC contains non-finite entries for {chromosomeName}"
            )
        if blacklistExcludeMask.shape[0] != intervalCount:
            raise RuntimeError("MUNC blacklist mask length mismatch")
        countFloorWork = None
        if countModelVarianceFloorMat is not None:
            countFloorWork = np.asarray(
                countModelVarianceFloorMat,
                dtype=np.float32,
            ).copy()
            if countFloorWork.shape != seedMuncArr.shape:
                raise RuntimeError(
                    "count-model floor matrix must align with MUNC evidence"
                )
            finiteFloor = np.isfinite(countFloorWork)
            if np.any(finiteFloor & (countFloorWork < 0.0)):
                raise RuntimeError("count-model floor must be nonnegative")
            if np.any((~finiteFloor) & (~np.isnan(countFloorWork))):
                raise RuntimeError("count-model floor must be finite or NaN")
            countFloorWork[~finiteFloor] = 0.0

        localEvidenceMatrix = np.empty(seedMuncArr.shape, dtype=np.float32)
        responseWeightMatrix = np.ones(seedMuncArr.shape, dtype=np.float32)

        blockIntervals = max(2, int(muncSizing.trendBlockIntervals))
        startsArr = np.arange(0, intervalCount, blockIntervals, dtype=np.intp)
        endsArr = np.minimum(startsArr + blockIntervals, intervalCount).astype(
            np.intp
        )
        blockLengthsArr = endsArr - startsArr
        keepBlocks = blockLengthsArr >= 2
        startsArr = startsArr[keepBlocks]
        endsArr = endsArr[keepBlocks]
        blockLengthsArr = blockLengthsArr[keepBlocks]
        if startsArr.size == 0:
            raise RuntimeError(
                f"MUNC has no deterministic blocks for {chromosomeName}"
            )

        priorBlockLenIntervals = _resolveRuntimeBackgroundBlockLen(
            dependenceSpanIntervals_,
            backgroundBlockSizeBP_,
            intervalSizeBP,
            fitArgs.ECM_backgroundLengthScaleMultiplier,
        )
        seedStateModel = core._normalizeStateModel(processArgs.stateModel)
        seedDeltaF = (
            1.0
            if seedStateModel == core.STATE_MODEL_LEVEL
            else core._resolveFixedDeltaF(deltaF_)
        )
        seedMinQ = float(muncSeedProcessMinQ_)
        seedMaxQ = float(muncSeedProcessMaxQ_)
        seedQStart = time.perf_counter()
        seedProcessQ, seedQDiagnostics = core._estimateInitialProcessNoiseFromData(
            matrixData=np.ascontiguousarray(residArr, dtype=np.float32),
            matrixMunc=np.ascontiguousarray(seedMuncArr, dtype=np.float32),
            pad=float(pad_),
            stateModel=seedStateModel,
            minQ=float(seedMinQ),
            maxQ=float(seedMaxQ),
            deltaF=float(seedDeltaF),
            robustTNu=float(
                getattr(
                    observationArgs,
                    "muncSeedWeightStudentTdf",
                    constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_TDF,
                )
            ),
            qSeedPriorLevel=float(processArgs.qSeedPriorLevel),
        )
        seedProcessQ = np.ascontiguousarray(seedProcessQ, dtype=np.float32)
        logger.info(
            "MUNC seed Q %s source=%s reason=%s q_level=%.6g "
            "q_trend=%.6g minQ=%.6g maxQ=%.6g elapsed=%.3fs",
            chromosomeName,
            seedQDiagnostics["qSeedSource"],
            seedQDiagnostics["qSeedReason"],
            float(seedQDiagnostics["qSeedLevelFinal"]),
            float(seedQDiagnostics["qSeedTrendFinal"]),
            seedMinQ,
            seedMaxQ,
            time.perf_counter() - seedQStart,
        )

        seedPassCount = int(muncSeedWeightPasses_)
        seedWeightEnabled = bool(
            getattr(
                observationArgs,
                "muncSeedWeightEnabled",
                constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_ENABLED,
            )
        )
        seedWeightStudentT = bool(
            getattr(
                observationArgs,
                "muncSeedWeightStudentT",
                constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_T,
            )
        )
        seedUseWeights = bool(seedWeightEnabled and seedWeightStudentT)
        seedStudentTdf = float(
            getattr(
                observationArgs,
                "muncSeedWeightStudentTdf",
                constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_TDF,
            )
        )
        seedOmegaMin = float(
            getattr(
                observationArgs,
                "muncSeedWeightMin",
                constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_MIN,
            )
        )
        seedOmegaMax = float(
            getattr(
                observationArgs,
                "muncSeedWeightMax",
                constants.OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_MAX,
            )
        )
        if seedUseWeights:
            if (
                seedStudentTdf <= 0.0
                or seedOmegaMin <= 0.0
                or seedOmegaMax < seedOmegaMin
                or not np.isfinite(seedStudentTdf)
                or not np.isfinite(seedOmegaMin)
                or not np.isfinite(seedOmegaMax)
            ):
                raise ValueError("MUNC seed Student-t weight parameters are invalid")
        seedBlockMap = np.zeros(intervalCount, dtype=np.int32)
        seedStateDim = 1 if seedStateModel == core.STATE_MODEL_LEVEL else 2
        seedMatrixF = np.ascontiguousarray(
            core.constructMatrixF(float(seedDeltaF)),
            dtype=np.float32,
        )
        seedBackground = np.zeros(intervalCount, dtype=np.float32)
        seedGVariance = np.zeros(intervalCount, dtype=np.float32)
        omegaTrack = np.ones(intervalCount, dtype=np.float32)
        rhoTrack = np.ones(seedMuncArr.shape, dtype=np.float32)
        def _seedWorkingMunc(
            totalVariance: np.ndarray,
            omega: np.ndarray,
            rho: np.ndarray,
            gVariance: np.ndarray,
        ) -> np.ndarray:
            base = np.asarray(totalVariance, dtype=np.float64) + float(pad_)
            if seedUseWeights:
                denom = np.maximum(
                    np.asarray(omega, dtype=np.float64).reshape(1, -1)
                    * np.asarray(rho, dtype=np.float64),
                    1.0e-12,
                )
                working = base / denom
            else:
                working = base
            working += np.asarray(gVariance, dtype=np.float64).reshape(1, -1)
            working -= float(pad_)
            return np.ascontiguousarray(
                np.maximum(working, _MUNC_NUMERIC_VARIANCE_FLOOR),
                dtype=np.float32,
            )

        def _muncObservationMomentSeedPass(
            matrixData: np.ndarray,
            matrixMunc: np.ndarray,
            stateMean: np.ndarray,
            stateVariance: np.ndarray,
            *,
            background: np.ndarray,
            gVariance: np.ndarray,
            countFloor: np.ndarray | None,
            omegaIn: np.ndarray,
            rhoIn: np.ndarray,
            updateWeights: bool,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            cythonSeedPass = cconsenrich.cMuncObservationMomentSeedPass
            return cythonSeedPass(
                np.ascontiguousarray(matrixData, dtype=np.float32),
                np.ascontiguousarray(matrixMunc, dtype=np.float32),
                np.ascontiguousarray(stateMean, dtype=np.float32),
                np.ascontiguousarray(stateVariance, dtype=np.float32),
                background=np.ascontiguousarray(background, dtype=np.float32),
                gVariance=np.ascontiguousarray(gVariance, dtype=np.float32),
                countFloor=countFloor,
                omegaIn=np.ascontiguousarray(omegaIn, dtype=np.float32),
                rhoIn=np.ascontiguousarray(rhoIn, dtype=np.float32),
                pad=float(pad_),
                studentTdf=float(seedStudentTdf),
                useSeedWeights=bool(seedUseWeights),
                updateWeights=bool(updateWeights),
                omegaMin=float(seedOmegaMin),
                omegaMax=float(seedOmegaMax),
                varianceFloor=float(_MUNC_NUMERIC_VARIANCE_FLOOR),
                varianceCap=(
                    float(maxR_)
                    if maxR_ is not None and maxR_ > 0.0
                    else np.finfo(np.float32).max
                ),
            )

        def _smoothDenseLocalEvidence(
            localEvidence: np.ndarray,
        ) -> np.ndarray:
            cythonSmooth = cconsenrich.cMuncSmoothDenseLocalEvidence
            return np.ascontiguousarray(
                cythonSmooth(
                    np.ascontiguousarray(localEvidence, dtype=np.float32),
                    int(muncSizing.localWindowIntervals),
                    excludeMask=blacklistExcludeMask,
                    eps=float(_MUNC_NUMERIC_VARIANCE_FLOOR),
                ),
                dtype=np.float32,
            )

        def _localEvidenceToTotalVariance(localEvidence: np.ndarray) -> np.ndarray:
            localArr = np.asarray(localEvidence, dtype=np.float32)
            if countFloorWork is None:
                totalArr = localArr.copy()
            else:
                totalArr = np.ascontiguousarray(
                    localArr + np.asarray(countFloorWork, dtype=np.float32),
                    dtype=np.float32,
                )
            varianceCap = (
                float(maxR_)
                if maxR_ is not None and maxR_ > 0.0
                else None
            )
            if varianceCap is not None:
                totalArr = np.minimum(totalArr, np.float32(varianceCap))
            return np.ascontiguousarray(
                np.maximum(totalArr, np.float32(_MUNC_NUMERIC_VARIANCE_FLOOR)),
                dtype=np.float32,
            )

        def _runSeedSmoother(
            totalVariance: np.ndarray,
            omega: np.ndarray,
            rho: np.ndarray,
            background: np.ndarray,
            gVariance: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            seedData = np.ascontiguousarray(
                residArr - np.asarray(background, dtype=np.float32).reshape(1, -1),
                dtype=np.float32,
            )
            workingMunc = _seedWorkingMunc(totalVariance, omega, rho, gVariance)
            stateForward = np.empty((intervalCount, seedStateDim), dtype=np.float32)
            stateCovarForward = np.empty(
                (intervalCount, seedStateDim, seedStateDim),
                dtype=np.float32,
            )
            pNoiseForward = np.empty(
                (intervalCount, seedStateDim, seedStateDim),
                dtype=np.float32,
            )
            vectorD = np.empty(intervalCount, dtype=np.float32)
            if seedStateModel == core.STATE_MODEL_LEVEL:
                cconsenrich.cforwardPassLevel(
                    matrixData=seedData,
                    matrixPluginMuncInit=workingMunc,
                    matrixQ0=np.ascontiguousarray(
                        seedProcessQ[:1, :1],
                        dtype=np.float32,
                    ),
                    intervalToBlockMap=seedBlockMap,
                    blockCount=1,
                    stateInit=float(stateArgs.stateInit),
                    stateCovarInit=float(stateArgs.stateCovarInit),
                    pad=float(pad_),
                    chunkSize=0,
                    stateForward=stateForward,
                    stateCovarForward=stateCovarForward,
                    pNoiseForward=pNoiseForward,
                    vectorD=vectorD,
                    returnNLL=True,
                    storeNLLInD=False,
                    lambdaExp=None,
                    processPrecExp=None,
                    ECM_useObsPrecisionReweighting=False,
                    ECM_useProcessPrecisionReweighting=False,
                    ECM_useAPN=False,
                )
                stateSmoothed, stateCovarSmoothed, _lagCov, _postResiduals = (
                    cconsenrich.cbackwardPassLevel(
                        matrixData=seedData,
                        stateForward=stateForward,
                        stateCovarForward=stateCovarForward,
                        pNoiseForward=pNoiseForward,
                        chunkSize=0,
                        stateSmoothed=None,
                        stateCovarSmoothed=None,
                        lagCovSmoothed=None,
                        postFitResiduals=None,
                    )
                )
            else:
                cconsenrich.cforwardPass(
                    matrixData=seedData,
                    matrixPluginMuncInit=workingMunc,
                    matrixF=seedMatrixF,
                    matrixQ0=np.ascontiguousarray(
                        seedProcessQ[:2, :2],
                        dtype=np.float32,
                    ),
                    intervalToBlockMap=seedBlockMap,
                    blockCount=1,
                    stateInit=float(stateArgs.stateInit),
                    stateCovarInit=float(stateArgs.stateCovarInit),
                    pad=float(pad_),
                    projectStateDuringFiltering=bool(stateArgs.boundState),
                    stateLowerBound=float(stateArgs.stateLowerBound),
                    stateUpperBound=float(stateArgs.stateUpperBound),
                    chunkSize=0,
                    stateForward=stateForward,
                    stateCovarForward=stateCovarForward,
                    pNoiseForward=pNoiseForward,
                    vectorD=vectorD,
                    returnNLL=True,
                    storeNLLInD=False,
                    lambdaExp=None,
                    processPrecExp=None,
                    ECM_useObsPrecisionReweighting=False,
                    ECM_useProcessPrecisionReweighting=False,
                    ECM_useAPN=False,
                )
                stateSmoothed, stateCovarSmoothed, _lagCov, _postResiduals = (
                    cconsenrich.cbackwardPass(
                        matrixData=seedData,
                        matrixF=seedMatrixF,
                        stateForward=stateForward,
                        stateCovarForward=stateCovarForward,
                        pNoiseForward=pNoiseForward,
                        chunkSize=0,
                        stateSmoothed=None,
                        stateCovarSmoothed=None,
                        lagCovSmoothed=None,
                        postFitResiduals=None,
                    )
                )
            return (
                np.asarray(stateSmoothed, dtype=np.float32),
                np.asarray(stateCovarSmoothed, dtype=np.float32),
            )

        def _updateSeedBackground(
            stateMean: np.ndarray,
            totalVariance: np.ndarray,
            omega: np.ndarray,
            rho: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not bool(fitArgs.fitBackground):
                return (
                    np.zeros(intervalCount, dtype=np.float32),
                    np.zeros(intervalCount, dtype=np.float32),
                )
            base = np.asarray(totalVariance, dtype=np.float64) + float(pad_)
            if seedUseWeights:
                invVar = (
                    np.asarray(omega, dtype=np.float64).reshape(1, -1)
                    * np.asarray(rho, dtype=np.float64)
                    / np.maximum(base, 1.0e-12)
                )
            else:
                invVar = 1.0 / np.maximum(base, 1.0e-12)
            residualForG = (
                np.asarray(residArr, dtype=np.float32)
                - np.asarray(stateMean, dtype=np.float32).reshape(1, -1)
            )
            nextBackground = core.solveZeroCenteredBackground(
                residualMatrix=np.ascontiguousarray(residualForG, dtype=np.float32),
                invVarMatrix=np.ascontiguousarray(invVar, dtype=np.float32),
                blockLenIntervals=int(priorBlockLenIntervals),
                backgroundSmoothness=float(fitArgs.ECM_backgroundSmoothness),
                zeroCenter=bool(fitArgs.ECM_zeroCenterBackground),
                useNonnegative=bool(fitArgs.useNonnegativeBackground),
                backgroundNegativePenaltyMultiplier=(
                    fitArgs.backgroundNegativePenaltyMultiplier
                ),
            )
            if (
                muncEBPriorGUncertaintyMode
                == constants.MUNC_EB_PRIOR_G_UNCERTAINTY_MODE_DISABLED
            ):
                nextGVariance = np.zeros(intervalCount, dtype=np.float32)
            elif (
                muncEBPriorGUncertaintyMode
                == constants.MUNC_EB_PRIOR_G_UNCERTAINTY_MODE_PROXY
            ):
                weightTrack = np.sum(invVar, axis=0, dtype=np.float64)
                lamFirst, lamSecond = core._backgroundPenaltyWeightsFromSpan(
                    blockLenIntervals=int(priorBlockLenIntervals),
                    backgroundSmoothness=float(fitArgs.ECM_backgroundSmoothness),
                )
                diagonal = weightTrack + core._backgroundPenaltyDiagonal(
                    intervalCount,
                    float(lamFirst),
                    float(lamSecond),
                )
                negativePenaltyMultiplier = (
                    fitArgs.backgroundNegativePenaltyMultiplier
                    if bool(fitArgs.useNonnegativeBackground)
                    else None
                )
                if negativePenaltyMultiplier is not None:
                    negativePenaltyValue = float(negativePenaltyMultiplier)
                    if negativePenaltyValue > 0.0:
                        positiveWeights = weightTrack[weightTrack > 0.0]
                        weightScale = (
                            float(np.median(positiveWeights))
                            if positiveWeights.size
                            else 1.0
                        )
                        diagonal[np.asarray(nextBackground) < 0.0] += (
                            negativePenaltyValue * max(weightScale, 1.0e-12)
                        )
                nextGVariance = (
                    1.0 / np.maximum(diagonal, 1.0e-12)
                ).astype(np.float32)
            else:
                raise RuntimeError(
                    "unsupported MUNC EB prior g uncertainty mode "
                    f"{muncEBPriorGUncertaintyMode}"
                )
            maxGVariance = float(np.quantile(np.asarray(totalVariance, dtype=np.float64), 0.99))
            if not np.isfinite(maxGVariance) or maxGVariance <= 0.0:
                maxGVariance = 1.0
            clippedGVariance = np.minimum(
                np.maximum(np.asarray(nextGVariance, dtype=np.float32), 0.0),
                np.float32(maxGVariance),
            )
            return (
                np.ascontiguousarray(nextBackground, dtype=np.float32),
                np.ascontiguousarray(clippedGVariance, dtype=np.float32),
            )

        stateScore = np.zeros((intervalCount, seedStateDim), dtype=np.float32)
        stateCovScore = np.zeros(
            (intervalCount, seedStateDim, seedStateDim),
            dtype=np.float32,
        )
        for seedPassIndex in range(seedPassCount):
            stateArr, stateCovArr = _runSeedSmoother(
                seedMuncArr,
                omegaTrack,
                rhoTrack,
                seedBackground,
                seedGVariance,
            )
            stateVariance = np.maximum(
                np.asarray(stateCovArr[:, 0, 0], dtype=np.float32),
                0.0,
            )
            (
                momentArr,
                rhoNext,
                omegaRaw,
                omegaNext,
                localNext,
                seedMuncNext,
            ) = _muncObservationMomentSeedPass(
                residArr,
                seedMuncArr,
                stateArr[:, 0],
                stateVariance,
                background=np.ascontiguousarray(seedBackground, dtype=np.float32),
                gVariance=np.ascontiguousarray(seedGVariance, dtype=np.float32),
                countFloor=countFloorWork,
                omegaIn=np.ascontiguousarray(omegaTrack, dtype=np.float32),
                rhoIn=np.ascontiguousarray(rhoTrack, dtype=np.float32),
                updateWeights=True,
            )
            seedBackground, seedGVariance = _updateSeedBackground(
                np.asarray(stateArr[:, 0], dtype=np.float32),
                seedMuncArr,
                omegaTrack,
                rhoTrack,
            )
            seedMuncArr = np.ascontiguousarray(seedMuncNext, dtype=np.float32)
            localEvidenceMatrix = np.ascontiguousarray(localNext, dtype=np.float32)
            omegaTrack = np.ascontiguousarray(omegaNext, dtype=np.float32)
            rhoTrack = np.ascontiguousarray(rhoNext, dtype=np.float32)
            if not seedUseWeights:
                omegaTrack.fill(1.0)
                rhoTrack.fill(1.0)
            omegaStats = _summarizeFiniteArray(omegaTrack)
            rhoStats = _summarizeFiniteArray(rhoTrack)
            momentStats = _summarizeFiniteArray(momentArr)
            gStats = _summarizeFiniteArray(seedBackground)
            GStats = _summarizeFiniteArray(seedGVariance)
            baseRatio = np.asarray(momentArr, dtype=np.float64) / np.maximum(
                np.asarray(seedMuncArr, dtype=np.float64) + float(pad_),
                1.0e-12,
            )
            logger.info(
                "MUNC dense seed %s pass=%d/%d mode=%s "
                "seed_omega[q05,q50,q95]=[%s,%s,%s] "
                "seed_rho[q05,q50,q95]=[%s,%s,%s] "
                "M[q05,q50,q95]=[%s,%s,%s] mean_M_over_B=%.6g "
                "g[q05,q50,q95]=[%s,%s,%s] G[q05,q50,q95]=[%s,%s,%s]",
                chromosomeName,
                int(seedPassIndex + 1),
                int(seedPassCount),
                (
                    "student_t"
                    if seedUseWeights
                    else ("gaussian" if seedWeightEnabled else "disabled")
                ),
                _fmtDiagnosticFloat(omegaStats["p05"]),
                _fmtDiagnosticFloat(omegaStats["median"]),
                _fmtDiagnosticFloat(omegaStats["p95"]),
                _fmtDiagnosticFloat(rhoStats["p05"]),
                _fmtDiagnosticFloat(rhoStats["median"]),
                _fmtDiagnosticFloat(rhoStats["p95"]),
                _fmtDiagnosticFloat(momentStats["p05"]),
                _fmtDiagnosticFloat(momentStats["median"]),
                _fmtDiagnosticFloat(momentStats["p95"]),
                float(np.mean(baseRatio, dtype=np.float64)),
                _fmtDiagnosticFloat(gStats["p05"]),
                _fmtDiagnosticFloat(gStats["median"]),
                _fmtDiagnosticFloat(gStats["p95"]),
                _fmtDiagnosticFloat(GStats["p05"]),
                _fmtDiagnosticFloat(GStats["median"]),
                _fmtDiagnosticFloat(GStats["p95"]),
            )
            stateScore = stateArr
            stateCovScore = stateCovArr

        stateScore, stateCovScore = _runSeedSmoother(
            seedMuncArr,
            omegaTrack,
            rhoTrack,
            seedBackground,
            seedGVariance,
        )
        scoreVariance = np.maximum(
            np.asarray(stateCovScore[:, 0, 0], dtype=np.float32),
            0.0,
        )
        (
            momentScore,
            rhoScore,
            omegaRawScore,
            omegaScore,
            localPointwise,
            seedMuncPointwise,
        ) = _muncObservationMomentSeedPass(
            residArr,
            seedMuncArr,
            stateScore[:, 0],
            scoreVariance,
            background=np.ascontiguousarray(seedBackground, dtype=np.float32),
            gVariance=np.ascontiguousarray(seedGVariance, dtype=np.float32),
            countFloor=countFloorWork,
            omegaIn=np.ascontiguousarray(omegaTrack, dtype=np.float32),
            rhoIn=np.ascontiguousarray(rhoTrack, dtype=np.float32),
            updateWeights=False,
        )
        if not (
            observationArgs.EB_setNuL is not None and observationArgs.EB_setNuL > 3
        ):
            nuLSpanIntervals = muncSizing.dependenceSpanIntervals
            if nuLSpanIntervals is None:
                raise RuntimeError("MUNC ESS Nu_L requires a dependence working span")
            nuLHorizon = min(
                int(nuLSpanIntervals),
                int(muncSizing.localWindowIntervals) - 1,
            )
            if nuLHorizon < 1:
                raise RuntimeError(
                    "MUNC ESS Nu_L requires a positive truncation horizon"
                )
            nuLActiveMask = np.ascontiguousarray(
                blacklistExcludeMask == 0,
                dtype=np.uint8,
            )
            nuLActiveCount = int(np.count_nonzero(nuLActiveMask))
            if nuLActiveCount >= 2:
                for sampleIndex in range(int(localPointwise.shape[0])):
                    _ess, tau, _lagsUsed = cconsenrich.cEstimateEffectiveSampleSize(
                        localPointwise[sampleIndex, :],
                        int(nuLHorizon),
                        activeMask=nuLActiveMask,
                        logPositive=True,
                        windowIntervals=int(muncSizing.localWindowIntervals),
                    )
                    if not np.isfinite(tau) or tau <= 0.0:
                        raise RuntimeError(
                            "MUNC ESS Nu_L design effect is invalid for "
                            f"{chromosomeName}"
                        )
                    pooledNuLTauWeightSum += float(tau) * float(nuLActiveCount)
                    pooledNuLWeightSum += float(nuLActiveCount)
        localEvidenceMatrix = _smoothDenseLocalEvidence(localPointwise)
        seedMuncArr = _localEvidenceToTotalVariance(localEvidenceMatrix)
        if seedUseWeights:
            responseWeightMatrix = np.ascontiguousarray(
                np.asarray(rhoScore, dtype=np.float32)
                * np.asarray(omegaScore, dtype=np.float32).reshape(1, -1),
                dtype=np.float32,
            )
        else:
            responseWeightMatrix = np.ones(seedMuncArr.shape, dtype=np.float32)
        seedMeanTrack = np.asarray(stateScore[:, 0], dtype=np.float64)
        localStats = _summarizeFiniteArray(localEvidenceMatrix)
        totalStats = _summarizeFiniteArray(seedMuncArr)
        qStats = _summarizeFiniteArray(responseWeightMatrix)
        logger.info(
            "MUNC dense score %s L_pt[q05,q50,q95]=[%s,%s,%s] "
            "V_pt[q05,q50,q95]=[%s,%s,%s] q_ji[q05,q50,q95]=[%s,%s,%s]",
            chromosomeName,
            _fmtDiagnosticFloat(localStats["p05"]),
            _fmtDiagnosticFloat(localStats["median"]),
            _fmtDiagnosticFloat(localStats["p95"]),
            _fmtDiagnosticFloat(totalStats["p05"]),
            _fmtDiagnosticFloat(totalStats["median"]),
            _fmtDiagnosticFloat(totalStats["p95"]),
            _fmtDiagnosticFloat(qStats["p05"]),
            _fmtDiagnosticFloat(qStats["median"]),
            _fmtDiagnosticFloat(qStats["p95"]),
        )

        coveredMask = np.zeros(intervalCount, dtype=bool)
        for blockStart, blockEnd in zip(startsArr, endsArr):
            coveredMask[int(blockStart) : int(blockEnd)] = True
        includedMask = blacklistExcludeMask == 0
        includedCount = int(np.count_nonzero(includedMask))
        coveredCount = int(np.count_nonzero(coveredMask & includedMask))
        coverageFraction = (
            float(coveredCount) / float(includedCount) if includedCount else 0.0
        )
        localEvidenceCachePath = os.path.join(
            pooledMuncCache.name,
            f"chrom_{c_:05d}_munc_local_evidence.npy",
        )
        np.save(localEvidenceCachePath, localEvidenceMatrix, allow_pickle=False)
        muncLocalEvidenceCachePaths[chromosomeName] = localEvidenceCachePath

        blockCount = int(startsArr.size)
        blockPredictorMatrix = np.full(
            (numSamples, blockCount),
            np.nan,
            dtype=np.float64,
        )
        blockEvidenceArr = np.full((numSamples, blockCount), np.nan, dtype=np.float64)
        blockWeightArr = np.zeros((numSamples, blockCount), dtype=np.float64)
        blockEffectiveSupport = np.zeros((numSamples, blockCount), dtype=np.float64)
        blockIncludedCounts = np.zeros(blockCount, dtype=np.int64)
        predictorTrack = core._muncTrendPredictor(seedMeanTrack)
        responseWeightArr = np.ascontiguousarray(
            responseWeightMatrix,
            dtype=np.float64,
        )
        responseWeightArr[:, blacklistExcludeMask != 0] = 0.0
        dependenceDivisor = float(max(1.0, float(muncTrendBlockDependenceMultiplier_)))
        for blockIndex, (blockStart, blockEnd) in enumerate(zip(startsArr, endsArr)):
            blockSlice = slice(int(blockStart), int(blockEnd))
            blockAllowed = np.asarray(blacklistExcludeMask[blockSlice]) == 0
            blockIncludedCounts[blockIndex] = int(np.count_nonzero(blockAllowed))
            predictorBlock = np.asarray(predictorTrack[blockSlice], dtype=np.float64)
            for sampleIndex in range(numSamples):
                evidenceBlock = np.asarray(
                    localEvidenceMatrix[sampleIndex, blockSlice],
                    dtype=np.float64,
                )
                qBlock = np.asarray(
                    responseWeightArr[sampleIndex, blockSlice],
                    dtype=np.float64,
                )
                validResponse = (
                    blockAllowed
                    & np.isfinite(predictorBlock)
                    & np.isfinite(evidenceBlock)
                    & (evidenceBlock > 0.0)
                    & np.isfinite(qBlock)
                    & (qBlock > 0.0)
                )
                if not np.any(validResponse):
                    continue
                qValid = qBlock[validResponse]
                evidenceValid = evidenceBlock[validResponse]
                predictorValid = predictorBlock[validResponse]
                qSum = float(np.sum(qValid, dtype=np.float64))
                if qSum <= 0.0:
                    continue
                blockPredictorMatrix[sampleIndex, blockIndex] = float(
                    np.sum(qValid * predictorValid, dtype=np.float64) / qSum
                )
                blockEvidenceArr[sampleIndex, blockIndex] = float(
                    np.sum(qValid * evidenceValid, dtype=np.float64) / qSum
                )
                blockWeightArr[sampleIndex, blockIndex] = qSum
                qSquareSum = float(np.sum(qValid * qValid, dtype=np.float64))
                if qSquareSum > 0.0:
                    blockEffectiveSupport[sampleIndex, blockIndex] = max(
                        1.0,
                        (qSum * qSum / qSquareSum) / dependenceDivisor,
                    )
        blockMeansForFitMatrix = _inverseMuncTrendPredictor(blockPredictorMatrix)
        collectedMeansParts: list[np.ndarray] = []
        rawDiagnosticMeansParts: list[np.ndarray] = []
        blockCovariates = (
            _blockCovariateMeans(
                chromCovariates,
                startsArr,
                endsArr,
            )
            if chromCovariates is not None
            else None
        )
        chromPooledRowCount = 0
        for j in range(numSamples):
            blockMeansArr = blockMeansForFitMatrix[j, :]
            blockVarsForPooled = blockEvidenceArr[j, :]
            blockKernelWeights = blockWeightArr[j, :]
            blockWeightsForPooled = blockEffectiveSupport[j, :]
            blockLogVarianceNoise = np.asarray(
                core.trigamma(np.maximum(blockWeightsForPooled, 4.0) / 2.0),
                dtype=np.float64,
            )
            blockTrendWeights = 1.0 / blockLogVarianceNoise
            valid = (
                np.isfinite(blockMeansArr)
                & np.isfinite(blockVarsForPooled)
                & np.isfinite(blockWeightsForPooled)
                & np.isfinite(blockKernelWeights)
                & (blockVarsForPooled >= _MUNC_NUMERIC_VARIANCE_FLOOR)
                & (blockWeightsForPooled > 0.0)
                & (blockKernelWeights > 0.0)
            )
            if not np.any(valid):
                continue
            count = int(np.count_nonzero(valid))
            chromPooledRowCount += count
            collectedMeansParts.append(
                np.asarray(blockMeansArr[valid], dtype=np.float64)
            )
            if diagnosticRawMat is not None and endsArr.size == blockMeansArr.size:
                rawMeans = _rawMeansForSampledBlocks(
                    diagnosticRawMat[j, :],
                    startsArr,
                    endsArr,
                )
                rawValid = valid & np.isfinite(rawMeans)
                if np.any(rawValid):
                    rawDiagnosticMeansParts.append(
                        np.asarray(rawMeans[rawValid], dtype=np.float64)
                    )
            pooledBlockMeansParts.append(
                np.asarray(blockMeansArr[valid], dtype=np.float64)
            )
            pooledBlockVarsParts.append(
                np.asarray(blockVarsForPooled[valid], dtype=np.float64)
            )
            pooledBlockLogVarianceNoiseParts.append(
                np.asarray(blockLogVarianceNoise[valid], dtype=np.float64)
            )
            if blockCovariates is not None:
                pooledBlockCovariatesParts.append(
                    np.asarray(blockCovariates[valid, :], dtype=np.float32)
                )
            pooledSampleIndexParts.append(np.full(count, int(j), dtype=np.int64))
            pooledChromIndexParts.append(np.full(count, int(c_), dtype=np.int64))
            pooledBlockStartsParts.append(
                np.asarray(startsArr[valid], dtype=np.int64)
            )
            pooledTrendWeightsParts.append(
                np.asarray(blockTrendWeights[valid], dtype=np.float64)
            )
        effectiveSupportPositive = blockEffectiveSupport[blockEffectiveSupport > 0.0]
        blockNoisePositive = core.trigamma(
            np.maximum(effectiveSupportPositive, 4.0) / 2.0
        )
        blockTrendWeightPositive = (
            1.0 / blockNoisePositive
            if blockNoisePositive.size
            else np.empty(0, dtype=np.float64)
        )
        logger.info(
            "MUNC deterministic block summary %s blocks=%d valid_blocks=%d "
            "rows=%d coverage_fraction=%.6g block_intervals_median=%.4g "
            "n_eff_median=%.4g tau2_median=%.4g trend_weight_median=%.4g",
            chromosomeName,
            int(startsArr.size),
            int(np.count_nonzero(blockIncludedCounts > 0)),
            int(chromPooledRowCount),
            float(coverageFraction),
            (
                float(np.median(blockIncludedCounts))
                if blockIncludedCounts.size
                else 0.0
            ),
            (
                float(np.median(effectiveSupportPositive))
                if effectiveSupportPositive.size
                else 0.0
            ),
            (
                float(np.median(blockNoisePositive))
                if blockNoisePositive.size
                else 0.0
            ),
            (
                float(np.median(blockTrendWeightPositive))
                if blockTrendWeightPositive.size
                else 0.0
            ),
        )
        if diagnosticRawMat is not None and rawDiagnosticMeansParts:
            _logMuncTrendInputSummary(
                chromosomeName,
                "raw",
                np.concatenate(rawDiagnosticMeansParts),
            )
        if collectedMeansParts:
            _logMuncTrendInputSummary(
                chromosomeName,
                inputLabel,
                np.concatenate(collectedMeansParts),
            )
        return np.ascontiguousarray(seedMeanTrack, dtype=np.float32)

    _logCliPhase(
        "Count and transform",
        "chromosomes=%d samples=%d",
        int(len(chromosomePlans)),
        int(numSamples),
    )
    cachedIntervalsByChromosome: dict[str, np.ndarray] = {}
    for c_, chromPlan in enumerate(chromosomePlans):
        chromosome = str(chromPlan["chromosome"])
        intervals, chromMat, countModelVarianceFloorMat = (
            _countAndTransformChromosomeMatrix(c_, chromPlan)
        )
        chromMat = _smoothRowsByFragmentLength(chromosome, chromMat)
        if countModelVarianceFloorMat is not None:
            countModelVarianceFloorMat = _smoothRowsByFragmentLength(
                chromosome,
                countModelVarianceFloorMat,
                varianceLike=True,
                logSmoothing=False,
            )
        cachePath = os.path.join(pooledMuncCache.name, f"chrom_{c_:05d}.npy")
        np.save(cachePath, chromMat, allow_pickle=False)
        transformedMatrixCachePaths[chromosome] = cachePath
        floorCachePath = os.path.join(
            pooledMuncCache.name,
            f"chrom_{c_:05d}_count_model_floor.npy",
        )
        if useCountNoiseFloor:
            if countModelVarianceFloorMat is None:
                raise RuntimeError("count floor matrix missing")
            np.save(floorCachePath, countModelVarianceFloorMat, allow_pickle=False)
            countModelVarianceFloorCachePaths[chromosome] = floorCachePath
        cachedIntervalsByChromosome[chromosome] = np.asarray(intervals, dtype=np.uint32)

    muncSizingNeedsDependence = core._muncSizingNeedsDependence(
        muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
    )
    if backgroundBlockSizeBP_ < 0 or muncSizingNeedsDependence:
        _ensureSampledDependenceSpan(transformedMatrixCachePaths)

    _logCliPhase(
        "MUNC prior setup",
        "chromosomes=%d seed_passes=%d EB=%s",
        int(len(chromosomePlans)),
        int(muncSeedWeightPasses_),
        "yes" if bool(observationArgs.EB_use) else "no",
    )
    for c_, chromPlan in enumerate(chromosomePlans):
        chromosome = str(chromPlan["chromosome"])
        cachePath = transformedMatrixCachePaths[chromosome]
        chromMat = np.ascontiguousarray(np.load(cachePath), dtype=np.float32)
        intervals = cachedIntervalsByChromosome[chromosome]
        countModelVarianceFloorMat = (
            _loadCountModelVarianceFloor(
                chromosome,
                int(chromMat.shape[1]),
            )
            if useCountNoiseFloor
            else None
        )
        muncResidualBackground, muncResidualSource, muncResidualBlockLen = (
            _estimateMuncResidualizationBackground(chromosome, chromMat)
        )
        backgroundCachePath = os.path.join(
            pooledMuncCache.name,
            f"chrom_{c_:05d}_munc_g0.npy",
        )
        np.save(backgroundCachePath, muncResidualBackground, allow_pickle=False)
        muncResidualBackgroundCachePaths[chromosome] = backgroundCachePath
        residMat = _residualizeMuncInput(
            chromosome,
            chromMat,
            muncResidualBackground,
            logDiagnostics=True,
            source=muncResidualSource,
            blockLenIntervals=muncResidualBlockLen,
        )
        seedMuncMat, _seedObservationMask = _coarseMuncResidualizationMunc(chromMat)
        for j in range(numSamples):
            seedCountFloor = None
            if countModelVarianceFloorMat is not None:
                seedCountFloor = np.ascontiguousarray(
                    countModelVarianceFloorMat[j, :],
                    dtype=np.float32,
                ).reshape(-1)
                if seedCountFloor.shape[0] != chromMat.shape[1]:
                    raise RuntimeError(
                        "seed MUNC count-model floor track length mismatch for "
                        f"{chromosome} sample {j}: expected {chromMat.shape[1]}, "
                        f"got {seedCountFloor.shape[0]}"
                    )
            seedMuncMat[j, :] = core.applyMuncCountModelVarianceFloor(
                seedMuncMat[j, :],
                seedCountFloor,
                varianceFloor=_MUNC_NUMERIC_VARIANCE_FLOOR,
                varianceCap=maxR_ if maxR_ is not None and maxR_ > 0.0 else None,
            )
        muncTrendInputLabel = "dense_seed_state"
        chromCovariates = _getChromMuncCovariates(
            chromosome,
            int(chromPlan["start"]),
            int(chromPlan["end"]),
            int(chromPlan["numIntervals"]),
        )
        priorMeanTrack = _collectPooledMuncBlocks(
            c_,
            intervals,
            chromMat,
            seedMuncMat,
            inputLabel=muncTrendInputLabel,
            diagnosticRawMat=chromMat if bool(fitArgs.fitBackground) else None,
            chromCovariates=chromCovariates,
            countModelVarianceFloorMat=countModelVarianceFloorMat,
        )
        priorMeanCachePath = os.path.join(
            pooledMuncCache.name,
            f"chrom_{c_:05d}_munc_prior_mean.npy",
        )
        np.save(priorMeanCachePath, priorMeanTrack, allow_pickle=False)
        muncPriorMeanCachePaths[chromosome] = priorMeanCachePath

    if pooledBlockMeansParts:
        pooledBlockMeans = np.concatenate(pooledBlockMeansParts)
        pooledBlockVars = np.concatenate(pooledBlockVarsParts)
        if pooledBlockLogVarianceNoiseParts:
            pooledBlockLogVarianceNoise = np.concatenate(
                pooledBlockLogVarianceNoiseParts
            )
        else:
            pooledBlockLogVarianceNoise = None
        pooledSampleIndex = np.concatenate(pooledSampleIndexParts)
        pooledChromIndex = np.concatenate(pooledChromIndexParts)
        pooledBlockStarts = np.concatenate(pooledBlockStartsParts)
        pooledTrendWeights = np.concatenate(pooledTrendWeightsParts)
        if pooledBlockCovariatesParts:
            pooledBlockCovariates = np.concatenate(pooledBlockCovariatesParts)
        elif muncCovariatesEnabled:
            pooledBlockCovariates = np.zeros(
                (pooledBlockMeans.size, len(muncCovariateRawFeatures)),
                dtype=np.float32,
            )
        else:
            pooledBlockCovariates = None
    else:
        pooledBlockMeans = np.empty(0, dtype=np.float64)
        pooledBlockVars = np.empty(0, dtype=np.float64)
        pooledBlockCovariates = (
            np.empty((0, len(muncCovariateRawFeatures)), dtype=np.float32)
            if muncCovariatesEnabled
            else None
        )
        pooledBlockLogVarianceNoise = None
        pooledSampleIndex = np.empty(0, dtype=np.int64)
        pooledChromIndex = np.empty(0, dtype=np.int64)
        pooledBlockStarts = np.empty(0, dtype=np.int64)
        pooledTrendWeights = np.empty(0, dtype=np.float64)

    pooledBlockMeansParts.clear()
    pooledBlockVarsParts.clear()
    pooledBlockCovariatesParts.clear()
    pooledBlockLogVarianceNoiseParts.clear()
    pooledSampleIndexParts.clear()
    pooledChromIndexParts.clear()
    pooledBlockStartsParts.clear()
    pooledTrendWeightsParts.clear()

    pooledMuncSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceSpanIntervals=dependenceSpanIntervals_,
        muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
        muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier_,
        muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier_,
    )
    if pooledBlockLogVarianceNoise is not None:
        blockNoise = pooledBlockLogVarianceNoise[
            np.isfinite(pooledBlockLogVarianceNoise)
            & (pooledBlockLogVarianceNoise > 0.0)
        ]
        if blockNoise.size:
            blockEffNu = 2.0 * core.itrigamma(blockNoise)
            blockEffNu = blockEffNu[np.isfinite(blockEffNu)]
            logger.info(
                "pooled MUNC deterministic block log-variance noise: pairs=%d "
                "logvar_noise_median=%.4g trend_weight_median=%.4g "
                "block_nu_L_effective_median=%.2f",
                int(blockNoise.size),
                float(np.median(blockNoise)),
                float(np.median(1.0 / blockNoise)),
                float(np.median(blockEffNu)) if blockEffNu.size else float("nan"),
    )
    pooledBlockSizeIntervals = int(pooledMuncSizing.trendBlockIntervals)
    pooledLocalWindowIntervals = int(pooledMuncSizing.localWindowIntervals)
    pooledNuLHorizon = (
        0
        if pooledMuncSizing.dependenceSpanIntervals is None
        else min(
            int(pooledMuncSizing.dependenceSpanIntervals),
            int(pooledLocalWindowIntervals) - 1,
        )
    )
    pooledNuLEta = 1.0
    if observationArgs.EB_setNuL is not None and observationArgs.EB_setNuL > 3:
        pooledNuL = float(observationArgs.EB_setNuL)
    else:
        if pooledMuncSizing.dependenceSpanIntervals is None:
            raise RuntimeError("MUNC ESS Nu_L requires a dependence working span")
        if pooledNuLHorizon < 1:
            raise RuntimeError(
                "MUNC ESS Nu_L requires a positive truncation horizon"
            )
        if pooledNuLWeightSum <= 0.0:
            raise RuntimeError("MUNC Nu_L ESS has no active pointwise evidence")
        pooledNuLEta = float(pooledNuLTauWeightSum / pooledNuLWeightSum)
        if not np.isfinite(pooledNuLEta) or pooledNuLEta <= 0.0:
            raise RuntimeError(f"MUNC Nu_L ESS eta is invalid: {pooledNuLEta}")
        pooledNuL = float(
            max(4, (float(pooledLocalWindowIntervals) / pooledNuLEta) - 3.0)
        )
    pooledNu0Cap = 100.0 * float(pooledNuL)

    varianceFloorForTrend = _MUNC_NUMERIC_VARIANCE_FLOOR
    varianceCapForTrend = maxR_ if maxR_ is not None and maxR_ > 0.0 else None
    _logCliPhase(
        "MUNC trend fit",
        "pairs=%d samples=%d",
        int(pooledBlockMeans.size),
        int(numSamples),
    )
    _logMuncEstimationParameters(
        chromosomeCount=len(chromosomePlans),
        sampleCount=numSamples,
        intervalSizeBP=intervalSizeBP,
        sizing=pooledMuncSizing,
        muncVarianceModel=muncVarianceModel_,
        samplingIters=samplingIters_,
        dependenceContextBP=dependenceContextBP_,
        dependenceSpanIntervals=dependenceSpanIntervals_,
        trendMultiplier=muncTrendBlockDependenceMultiplier_,
        localMultiplier=muncLocalWindowDependenceMultiplier_,
        observationArgs=observationArgs,
        sparseBedEnabled=bool(genomeArgs.sparseBedFile),
        varianceFloor=varianceFloorForTrend,
        varianceCap=varianceCapForTrend,
        trendNumBasis=trendNumBasis_,
        trendMinObsPerBasis=trendMinObsPerBasis_,
        trendMinEdf=trendMinEdf_,
        trendMaxEdf=trendMaxEdf_,
        trendLambdaMin=trendLambdaMin_,
        trendLambdaMax=trendLambdaMax_,
        trendLambdaGridSize=trendLambdaGridSize_,
        pooledPairCount=int(pooledBlockMeans.size),
        seedPassCount=int(muncSeedWeightPasses_),
    )
    pooledMuncFit = core.fitPooledMuncVarianceTrend(
        pooledBlockMeans,
        pooledBlockVars,
        pooledSampleIndex,
        weights=pooledTrendWeights,
        eps=varianceFloorForTrend,
        trendNumBasis=trendNumBasis_,
        trendMinObsPerBasis=trendMinObsPerBasis_,
        trendMinEdf=trendMinEdf_,
        trendMaxEdf=trendMaxEdf_,
        trendLambdaMin=trendLambdaMin_,
        trendLambdaMax=trendLambdaMax_,
        trendLambdaGridSize=trendLambdaGridSize_,
        sampleCount=int(numSamples),
    )
    pooledPriorVarianceBase = (
        core.evalPSplineLogVarianceTrend(
            pooledMuncFit.trend,
            pooledBlockMeans,
            eps=varianceFloorForTrend,
            maxVariance=varianceCapForTrend,
        ).astype(np.float64, copy=False)
        if pooledBlockMeans.size
        else np.empty(0, dtype=np.float64)
    )
    pooledReplicateVarianceFactors = np.asarray(
        pooledMuncFit.replicateVarianceFactors,
        dtype=np.float64,
    ).reshape(-1)
    fittedPooledReplicateVarianceFactors = pooledReplicateVarianceFactors.copy()
    if pooledReplicateVarianceFactors.shape != (int(numSamples),):
        raise RuntimeError(
            "pooled MUNC replicate variance factors do not match sample count"
        )
    if (
        not np.all(np.isfinite(pooledReplicateVarianceFactors))
        or np.any(pooledReplicateVarianceFactors <= 0.0)
    ):
        raise RuntimeError("pooled MUNC replicate variance factors are invalid")
    if pooledSampleIndex.size and np.any(
        (pooledSampleIndex < 0) | (pooledSampleIndex >= int(numSamples))
    ):
        raise RuntimeError("pooled MUNC sample indices are invalid")
    if not useReplicateVarianceScale:
        pooledReplicateVarianceFactors = np.ones_like(pooledReplicateVarianceFactors)
    pooledPriorVariance = (
        core._clipVarianceTrack(
            pooledPriorVarianceBase * pooledReplicateVarianceFactors[pooledSampleIndex],
            floor=varianceFloorForTrend,
            cap=varianceCapForTrend,
        ).astype(np.float64, copy=False)
        if pooledPriorVarianceBase.size
        else pooledPriorVarianceBase
    )
    if bool(
        getattr(
            outputArgs,
            "writeReplicateExchangeabilityDiagnostics",
            constants.OUTPUT_DEFAULT_WRITE_REPLICATE_EXCHANGEABILITY_DIAGNOSTICS,
        )
    ):
        replicateExchangeabilitySampleNames: list[str] = []
        for sampleIndex in range(int(numSamples)):
            source = (
                treatmentSources[sampleIndex]
                if sampleIndex < len(treatmentSources)
                else None
            )
            if source is None:
                sampleName = f"replicate_{sampleIndex + 1}"
            else:
                sampleName = str(
                    source.sampleName
                    or os.path.basename(source.path)
                    or f"replicate_{sampleIndex + 1}"
                )
            replicateExchangeabilitySampleNames.append(sampleName)
        rawReplicateExchangeabilityDiagnostic = (
            _replicateExchangeabilityFromPooledBlocks(
                pooledBlockVars,
                pooledPriorVarianceBase,
                pooledSampleIndex,
                pooledChromIndex,
                pooledBlockStarts,
                int(numSamples),
                sampleNames=replicateExchangeabilitySampleNames,
            )
        )
        replicateExchangeabilityDiagnostic = _replicateExchangeabilityFromPooledBlocks(
            pooledBlockVars,
            pooledPriorVariance,
            pooledSampleIndex,
            pooledChromIndex,
            pooledBlockStarts,
            int(numSamples),
            sampleNames=replicateExchangeabilitySampleNames,
        )
        replicateExchangeabilityDiagnostic["replicateVarianceFactors"] = (
            fittedPooledReplicateVarianceFactors.copy()
        )
        replicateExchangeabilityDiagnostic["priorVarianceFactorAdjusted"] = bool(
            useReplicateVarianceScale
        )
        replicateExchangeabilityDiagnostic["replicateSDMultipliers"] = np.sqrt(
            fittedPooledReplicateVarianceFactors
        )
        if str(rawReplicateExchangeabilityDiagnostic.get("status", "skipped")) == "ok":
            replicateExchangeabilityDiagnostic["rawOmnibusObserved"] = (
                rawReplicateExchangeabilityDiagnostic["omnibusObserved"]
            )
            replicateExchangeabilityDiagnostic["rawOmnibusPValue"] = (
                rawReplicateExchangeabilityDiagnostic["omnibusPValue"]
            )
            replicateExchangeabilityDiagnostic["rawEffectByReplicate"] = (
                rawReplicateExchangeabilityDiagnostic["effectByReplicate"]
            )
        replicateExchangeabilitySummaryPath = (
            _replicateExchangeabilitySummaryPath(str(experimentName))
        )
        _writeReplicateExchangeabilitySummary(
            replicateExchangeabilityDiagnostic,
            replicateExchangeabilitySummaryPath,
        )
        _warnReplicateVarianceHeterogeneity(
            replicateExchangeabilityDiagnostic,
            replicateExchangeabilitySummaryPath,
        )
        if str(replicateExchangeabilityDiagnostic.get("status", "skipped")) == "ok":
            _plotReplicateExchangeabilityDiagnostic(
                replicateExchangeabilityDiagnostic,
                _replicateExchangeabilityPlotPath(str(experimentName)),
                dpi=400,
            )

    specifiedNu0 = core._coerceEBPriorStrength(observationArgs.EB_setNu0)
    pooledMuncNu0BySample = np.empty(numSamples, dtype=np.float64)
    weakNu0Samples: list[int] = []
    for j in range(numSamples):
        if specifiedNu0 is not None:
            repNu0 = specifiedNu0
        else:
            repMask = pooledSampleIndex == j
            if np.count_nonzero(repMask) < 4:
                repNu0 = 4.0
                weakNu0Samples.append(int(j))
            else:
                repNu0 = core.EB_computePooledPriorStrength(
                    pooledBlockVars[repMask],
                    pooledPriorVariance[repMask],
                    pooledNuL,
                    sampleIndex=pooledSampleIndex[repMask],
                    chromosomeIndex=pooledChromIndex[repMask],
                    blockStarts=pooledBlockStarts[repMask],
                    thinBinSize=pooledLocalWindowIntervals,
                    localLogVarianceNoise=(
                        None
                        if pooledBlockLogVarianceNoise is None
                        else pooledBlockLogVarianceNoise[repMask]
                    ),
                )
                if repNu0 == 4.0:
                    weakNu0Samples.append(int(j))
        if not np.isfinite(repNu0):
            raise RuntimeError(f"pooled MUNC Nu_0 is non-finite for sample {j}")
        if repNu0 < 4.0:
            raise RuntimeError(f"pooled MUNC Nu_0 is below 4.0 for sample {j}")
        pooledMuncNu0BySample[j] = min(float(repNu0), float(pooledNu0Cap))

    logger.info(
        "pooled MUNC deterministic block trend: pairs=%d samples=%d "
        "nu_L=%.2f nu_L_W=%d nu_L_H=%d nu_L_eta=%.4g "
        "sampleFile=%s sampleNu0=%s weakNu0Samples=%s diagnostics=%s",
        int(pooledBlockMeans.size),
        int(numSamples),
        float(pooledNuL),
        int(pooledLocalWindowIntervals),
        int(pooledNuLHorizon),
        float(pooledNuLEta),
        sampleFileByIndex,
        np.array2string(
            np.asarray(pooledMuncNu0BySample, dtype=np.float64),
            precision=2,
            floatmode="fixed",
            separator=", ",
        ),
        weakNu0Samples,
        pooledMuncFit.diagnostics,
    )

    additiveCovariateModel: core.MuncAdditiveCovariateModel | None = None
    if muncCovariatesEnabled:
        if pooledBlockCovariates is None:
            logger.warning(
                "MUNC genomic covariates were enabled, but no block-level covariates "
                "were collected; continuing with the baseline MUNC trend only."
            )
        elif pooledBlockCovariates.shape[0] != pooledBlockMeans.size:
            raise RuntimeError(
                "pooled MUNC block covariates are not aligned with block means"
            )
        elif pooledBlockMeans.size == 0:
            logger.warning(
                "MUNC genomic covariates were enabled, but no pooled MUNC blocks "
                "were available; continuing with the baseline MUNC trend only."
            )
        else:
            baselinePooledVariance = core._clipVarianceTrack(
                pooledPriorVarianceBase,
                floor=varianceFloorForTrend,
                cap=varianceCapForTrend,
            ).astype(np.float64, copy=False)
            factorAdjustedBlockVars = (
                np.asarray(pooledBlockVars, dtype=np.float64)
                / pooledReplicateVarianceFactors[pooledSampleIndex]
            )
            additiveCovariateModel = core.fitMuncAdditiveCovariateModel(
                pooledBlockMeans,
                factorAdjustedBlockVars,
                baselinePooledVariance,
                pooledBlockCovariates,
                pooledSampleIndex,
                featureNames=muncCovariateFeatureNames,
                weights=pooledTrendWeights,
                sampleCount=numSamples,
                eps=varianceFloorForTrend,
            )
            pooledCoefficients = np.asarray(
                additiveCovariateModel.pooledCoefficients,
                dtype=np.float64,
            )
            additiveCovariateModel = additiveCovariateModel._replace(
                perReplicateCoefficients=np.repeat(
                    pooledCoefficients[None, :, :],
                    int(numSamples),
                    axis=0,
                ),
                replicateUsesPooled=np.ones(int(numSamples), dtype=bool),
                diagnostics={
                    **dict(additiveCovariateModel.diagnostics),
                    "replicate_covariate_fit": "disabled",
                    "replicate_fallback_count": int(numSamples),
                },
            )
            logger.info(
                "MUNC additive genomic covariate model: features=%s valid_pairs=%d "
                "basis=%d shared_replicates=%d pooled_coef_sum=%.4g",
                ",".join(additiveCovariateModel.featureNames),
                int(additiveCovariateModel.diagnostics.get("valid_pairs", 0)),
                int(additiveCovariateModel.diagnostics.get("basis_count", 0)),
                int(
                    additiveCovariateModel.diagnostics.get(
                        "replicate_fallback_count",
                        0,
                    )
                ),
                float(
                    additiveCovariateModel.diagnostics.get(
                        "pooled_coefficient_sum",
                        0.0,
                    )
                ),
            )

    del (
        pooledBlockMeans,
        pooledBlockVars,
        pooledBlockCovariates,
        pooledBlockLogVarianceNoise,
        pooledSampleIndex,
        pooledChromIndex,
        pooledBlockStarts,
        pooledTrendWeights,
        pooledPriorVarianceBase,
        pooledPriorVariance,
    )

    stateDiagnosticsByChromosome: Dict[str, Any] = {}
    stateShrinkageEnabled = bool(
        getattr(
            outputArgs,
            "stateShrinkageEnabled",
            constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_ENABLED,
        )
    )
    writeStateShrinkageTracks = bool(
        stateShrinkageEnabled and getattr(outputArgs, "writeStateShrinkage", False)
    )
    useShrunkStateScores = bool(
        stateShrinkageEnabled
        and peakCallingEnabled
        and getattr(
            matchingArgs,
            "useShrunkStateScores",
            constants.MATCHING_DEFAULT_USE_SHRUNK_STATE_SCORES,
        )
    )
    writeStateShrunkTrack = bool(writeStateShrinkageTracks or useShrunkStateScores)
    bedGraphTracks: List[Tuple[str, str]] = [("State", "state")]
    if outputArgs.writeUncertainty:
        bedGraphTracks.append(("uncertainty", "uncertainty"))
    if writeStateShrunkTrack:
        bedGraphTracks.extend(_stateShrinkageOutputTracks(writeStateShrinkageTracks))
    diagnosticTrackNames = tuple(getattr(outputArgs, "diagnosticTracks", ()) or ())
    stateDiagnosticTrackNames = tuple(
        trackName for trackName in diagnosticTrackNames if trackName == "slope"
    )
    if any(trackName != "slope" for trackName in diagnosticTrackNames):
        logger.info(
            "MUNC/omega and process/kappa diagnostic tracks are written to category logs. "
            "only slope remains a bedGraph diagnostic track."
        )
    for trackName in stateDiagnosticTrackNames:
        bedGraphTracks.append((trackName, trackName))
    saveBackgroundTracks = bool(
        outputArgs.saveBackgroundTracks and fitArgs.fitBackground
    )
    if outputArgs.saveBackgroundTracks and not fitArgs.fitBackground:
        logger.info(
            "outputParams.saveBackgroundTracks=True but fitParams.fitBackground=False; "
            "skipping background track export because no fitted g(i) is available."
        )
    if saveBackgroundTracks:
        bedGraphTracks.append(("background", "background"))
    suffixes = [suffix for _column, suffix in bedGraphTracks]
    bedGraphPlanChromOrder = [
        str(chromPlan["chromosome"]) for chromPlan in chromosomePlans
    ]
    bedGraphChromOrder = _chromSizesOrderForPlannedChromosomes(
        chromSizesDict,
        bedGraphPlanChromOrder,
    )
    bedGraphPlanOrderDiffers = bedGraphChromOrder != bedGraphPlanChromOrder
    if bedGraphPlanOrderDiffers:
        logger.info(
            "bedGraph track order uses chrom.sizes order because execution plan "
            "order differs."
        )
    genomeOptimizationPathRows: List[Mapping[str, Any]] = []
    precisionReweightingHistogramSamples: dict[str, Any] = {
        "lambda": np.empty(0, dtype=np.float64),
        "kappa": np.empty(0, dtype=np.float64),
    }
    deleteBlockCalibrationPlotSamples: dict[str, Any] = {
        "factor": np.empty(0, dtype=np.float64),
    }
    deleteBlockCalibrationCoverageRows: List[Mapping[str, Any]] = []
    runSummaryRows: List[Dict[str, Any]] = []
    segShrinkGenomeRequested = bool(
        outputArgs.writeUncertainty
        and uncertaintyCalibrationArgs.enabled
        and getattr(
            uncertaintyCalibrationArgs,
            "deleteBlockFactorModel",
            constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_MODEL,
        )
        == constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_SEG_SHRINK
    )
    segShrinkDeferredUncertainty: List[Dict[str, Any]] = []
    stateShrinkDeferred: List[Dict[str, Any]] = []
    stateShrinkDeferredByChromosome: Dict[str, Dict[str, Any]] = {}
    stateShrinkDeferredSuffixes = {
        suffix for _column, suffix in _stateShrinkageOutputTracks(True)
    }
    stateShrinkGenomeSummaryFields: Dict[str, Any] = {}
    if writeStateShrunkTrack:
        maxShrinkIntervals = max(
            (int(chromPlan["numIntervals"]) for chromPlan in chromosomePlans),
            default=1,
        )
        stateShrinkBlockIntervals = diagnostics.resolveUncertaintyBlockSizeIntervals(
            getattr(uncertaintyCalibrationArgs, "blockSizeBP", None),
            intervalSizeBP,
            maxShrinkIntervals,
        )
    else:
        stateShrinkBlockIntervals = 1
    saveGains = bool(
        getattr(outputArgs, "saveGains", constants.OUTPUT_DEFAULT_SAVE_GAINS)
    )
    replicateGainAccumulator = (
        _newReplicateGainAccumulator(len(treatmentSources)) if saveGains else None
    )

    _logCliPhase(
        "Contig final fits",
        "chromosomes=%d samples=%d",
        int(len(chromosomePlans)),
        int(numSamples),
    )
    for c_, chromPlan in enumerate(chromosomePlans):
        chromosomeStartTime = time.perf_counter()
        chromosome = str(chromPlan["chromosome"])
        chromosomeStart = int(chromPlan["start"])
        chromosomeEnd = int(chromPlan["end"])
        numIntervals = int(chromPlan["numIntervals"])
        _logCliProgressMilestone(
            "Contig %d/%d: %s intervals=%d",
            int(c_ + 1),
            int(len(chromosomePlans)),
            chromosome,
            int(numIntervals),
        )
        maxR_ = observationArgs.maxR
        minQ_ = processArgs.minQ
        maxQ_ = processArgs.maxQ
        # Negative process bounds are data-based and must be resolved independently
        # for each chromosome.
        if maxR_ is not None and maxR_ < 0.0:
            maxR_ = 1e4
        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            minQ_ = 0.0
            maxQ_ = 1e4
        logger.info(
            "chromosome.start %s intervals=%d samples=%d",
            chromosome,
            int(numIntervals),
            int(numSamples),
        )
        intervals = np.arange(chromosomeStart, chromosomeEnd, intervalSizeBP)
        cachePath = transformedMatrixCachePaths.get(chromosome)
        if cachePath is None:
            raise RuntimeError(f"Missing transformed matrix cache for {chromosome}")
        chromMat = np.ascontiguousarray(
            np.load(cachePath, allow_pickle=False),
            dtype=np.float32,
        )
        if chromMat.shape != (numSamples, numIntervals):
            raise RuntimeError(
                "Transformed matrix cache shape mismatch for "
                f"{chromosome}: expected {(numSamples, numIntervals)}, got {chromMat.shape}"
            )
        countModelVarianceFloorMat = (
            _loadCountModelVarianceFloor(
                chromosome,
                int(numIntervals),
            )
            if useCountNoiseFloor
            else None
        )
        countModelFloorQ05 = (
            _countModelVarianceFloorScalar(
                countModelVarianceFloorMat,
            )
            if countModelVarianceFloorMat is not None
            else None
        )
        muncResidualBackground = _loadMuncResidualizationBackground(
            chromosome,
            int(numIntervals),
        )
        muncPriorMeanTrack = _loadMuncPriorMeanTrack(
            chromosome,
            int(numIntervals),
        )
        pooledPriorVarianceTrack = (
            core.evalPSplineLogVarianceTrend(
                pooledMuncFit.trend,
                muncPriorMeanTrack,
                eps=varianceFloorForTrend,
                maxVariance=maxR_ if maxR_ is not None and maxR_ > 0.0 else None,
            )
            if bool(observationArgs.EB_use)
            else None
        )
        muncLocalEvidenceMat = _loadMuncLocalEvidenceMatrix(
            chromosome,
            int(numIntervals),
        )
        residMat = _residualizeMuncInput(
            chromosome,
            chromMat,
            muncResidualBackground,
            logDiagnostics=False,
        )
        chromMuncCovariates = _getChromMuncCovariates(
            chromosome,
            chromosomeStart,
            chromosomeEnd,
            numIntervals,
        )
        muncMat: np.ndarray = np.empty_like(chromMat, dtype=np.float32)
        logger.info(
            "loaded transformed matrix cache %s samples=%d intervals=%d",
            chromosome,
            int(numSamples),
            int(numIntervals),
        )

        useSparseNearest = bool(
            observationArgs.numNearest is not None
            and int(observationArgs.numNearest) > 0
            and genomeArgs.sparseBedFile
        )
        useSparseRestrictedLocalVariance = bool(
            getattr(observationArgs, "restrictLocalVarianceToSparseBed", False)
            and genomeArgs.sparseBedFile
        )

        sparseIntervalIndices = None
        sparseRegionMask = None
        muncIntervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32)
        muncExcludeMask = _getChromBlacklistMask(chromosome, intervals)
        blacklistedIntervals = int(np.count_nonzero(muncExcludeMask))
        if blacklistedIntervals:
            logger.info(
                "munc matrix: excluding blacklist intervals (chrom=%s, intervals=%d).",
                chromosome,
                blacklistedIntervals,
            )
        if useSparseNearest:
            sparseIntervalIndices = core._loadSparseIntervalIndices(
                genomeArgs.sparseBedFile,
                chromosome,
                intervals,
            )
            logger.info(
                "munc matrix: using explicit sparse-bed nearest-neighbor local variance "
                "(chrom=%s, numNearest=%d, sparseIntervals=%d).",
                chromosome,
                int(observationArgs.numNearest),
                int(sparseIntervalIndices.size),
            )
        if useSparseRestrictedLocalVariance:
            sparseRegionMask = core.getBedMask(
                chromosome,
                genomeArgs.sparseBedFile,
                intervals,
            )
            logger.info(
                "munc matrix: restricting local observation variance to "
                "sparse-bed regions (chrom=%s, model=%s, sparseIntervals=%d).",
                chromosome,
                muncVarianceModel_,
                int(np.count_nonzero(sparseRegionMask)),
            )

        def _fitMuncTrack(j: int) -> tuple[int, np.ndarray]:
            pooledTrend = pooledMuncFit.trend
            pooledNu0 = float(pooledMuncNu0BySample[j])
            replicateVarianceFactor = float(pooledReplicateVarianceFactors[j])
            finalMuncFloorKwargs: dict[str, np.ndarray] = {}
            if countModelVarianceFloorMat is not None:
                finalMuncFloorKwargs["countModelVarianceFloor"] = (
                    countModelVarianceFloorMat[j, :]
                )
            muncTrack, _ = core.getMuncTrack(
                chromosome,
                intervals,
                residMat[j, :],
                intervalSizeBP,
                samplingIters=samplingIters_,
                muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
                muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
                dependenceSpanIntervals=dependenceSpanIntervals_,
                muncTrendBlockDependenceMultiplier=(
                    muncTrendBlockDependenceMultiplier_
                ),
                muncLocalWindowDependenceMultiplier=(
                    muncLocalWindowDependenceMultiplier_
                ),
                muncVarianceModel=muncVarianceModel_,
                randomSeed=42 + j,
                EB_use=observationArgs.EB_use,
                EB_setNuL=observationArgs.EB_setNuL,
                EB_effectiveNuL=pooledNuL,
                trendNumBasis=trendNumBasis_,
                trendMinObsPerBasis=trendMinObsPerBasis_,
                trendMinEdf=trendMinEdf_,
                trendMaxEdf=trendMaxEdf_,
                trendLambdaMin=trendLambdaMin_,
                trendLambdaMax=trendLambdaMax_,
                trendLambdaGridSize=trendLambdaGridSize_,
                sparseIntervalIndices=sparseIntervalIndices,
                sparseRegionMask=sparseRegionMask,
                numNearest=int(observationArgs.numNearest or 0),
                sparseSupportScaleBP=observationArgs.sparseSupportScaleBP,
                sparseSupportPrior=float(observationArgs.sparseSupportPrior or 0.0),
                restrictLocalVarianceToSparseBed=bool(
                    getattr(observationArgs, "restrictLocalVarianceToSparseBed", False)
                ),
                verbose=args.verbose2,
                eps=varianceFloorForTrend,
                varianceFloor=varianceFloorForTrend,
                varianceCap=maxR_ if maxR_ is not None and maxR_ > 0.0 else None,
                intervalsArr=muncIntervalsArr,
                excludeMaskArr=muncExcludeMask,
                pooledTrend=pooledTrend,
                priorMeanTrack=muncPriorMeanTrack,
                priorVarianceTrack=pooledPriorVarianceTrack,
                replicateVarianceFactor=replicateVarianceFactor,
                EB_pooledNu0=pooledNu0,
                covariateTrack=chromMuncCovariates,
                additiveCovariateModel=additiveCovariateModel,
                replicateIndex=j,
                sampleFile=sampleFileByIndex[j],
                localVarianceTrack=muncLocalEvidenceMat[j, :],
                **finalMuncFloorKwargs,
            )
            return j, muncTrack

        # this has become a bottleneck, so gentle multiprocessing
        muncProgressDesc = "Fitting pooled signed MUNC variance"
        muncWorkers = io_helpers._getMuncWorkerCount(
            numSamples,
            chromMat.shape[1],
            sharedArrays=(
                chromMat,
                residMat,
                muncMat,
                muncIntervalsArr,
                muncExcludeMask,
                chromMuncCovariates,
                countModelVarianceFloorMat,
                sparseIntervalIndices,
                sparseRegionMask,
            ),
        )
        useParallelMunc = (
            numSamples >= 4 and chromMat.shape[1] >= 5000 and muncWorkers > 1
        )
        muncStart = time.perf_counter()
        logger.info(
            "munc.start %s samples=%d intervals=%d workers=%d",
            chromosome,
            int(numSamples),
            int(chromMat.shape[1]),
            int(muncWorkers if useParallelMunc else 1),
        )
        if useParallelMunc:
            logger.info(
                "munc matrix: using ThreadPool with %d workers (numSamples=%d, numIntervals=%d).",
                int(muncWorkers),
                int(numSamples),
                int(chromMat.shape[1]),
            )
            with ThreadPool(processes=int(muncWorkers)) as pool:
                for j, muncTrack in pool.imap(_fitMuncTrack, range(numSamples)):
                    muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)
        else:
            for j in range(numSamples):
                _, muncTrack = _fitMuncTrack(j)
                muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)
        logger.info(
            "munc.done %s samples=%d elapsed=%.3fs",
            chromosome,
            int(numSamples),
            time.perf_counter() - muncStart,
        )
        del residMat, muncLocalEvidenceMat, chromMuncCovariates
        if countModelVarianceFloorMat is not None:
            del countModelVarianceFloorMat

        blockLenIntervals_ = _resolveRuntimeBackgroundBlockLen(
            dependenceSpanIntervals_,
            int(backgroundBlockSizeBP_),
            intervalSizeBP,
            fitArgs.ECM_backgroundLengthScaleMultiplier,
        )

        if blacklistedIntervals:
            floors = core.applyBlacklistMuncFloor(
                muncMat,
                muncExcludeMask,
                float(_MUNC_NUMERIC_VARIANCE_FLOOR),
            )
            logger.info(
                "munc matrix: applied blacklist floors (chrom=%s, min=%.4g, median=%.4g, max=%.4g).",
                chromosome,
                float(np.min(floors)),
                float(np.median(floors)),
                float(np.max(floors)),
            )
        maxRFinite_ = (
            maxR_ is not None
            and np.isfinite(float(maxR_))
            and float(maxR_) > _MUNC_NUMERIC_VARIANCE_FLOOR
        )
        maxRForRepair_ = (
            float(maxR_) if maxRFinite_ else float(np.finfo(np.float32).max)
        )
        if not np.all(np.isfinite(muncMat)):
            invalidMuncCount = int(
                np.size(muncMat) - np.count_nonzero(np.isfinite(muncMat))
            )
            raise RuntimeError(
                f"MUNC matrix for {chromosome} has {invalidMuncCount} non-finite entries"
            )
        np.maximum(
            muncMat,
            np.float32(_MUNC_NUMERIC_VARIANCE_FLOOR),
            out=muncMat,
        )
        if maxRFinite_:
            np.minimum(muncMat, np.float32(maxRForRepair_), out=muncMat)
        minQ_ = processArgs.minQ
        maxQ_ = processArgs.maxQ

        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            autoMinQ = np.float32(1.0e-6)
            logger.info(
                "processParams.minQ < 0 or processParams.maxQ < 0 --> "
                "applying count-noise-independent numeric process bounds "
                "for conditioning minQ=%s",
                _formatOptionalLogValue(autoMinQ),
            )
            if processArgs.minQ < 0.0:
                minQ_ = autoMinQ
            else:
                minQ_ = np.float32(processArgs.minQ)
            if processArgs.maxQ < 0.0:
                maxQ_ = np.float32(np.inf)
            else:
                maxQ_ = np.float32(max(processArgs.maxQ, minQ_))
        else:
            maxQ_ = np.float32(max(maxQ_, minQ_))
        logger.info(
            "count noise floor summary %s status=%s q05=%s "
            "numericRFloor=%s maxR=%s minQ=%s maxQ=%s",
            chromosome,
            "enabled" if useCountNoiseFloor else "disabled",
            _formatOptionalLogValue(countModelFloorQ05),
            _formatOptionalLogValue(_MUNC_NUMERIC_VARIANCE_FLOOR),
            _formatOptionalLogValue(maxR_),
            _formatOptionalLogValue(minQ_),
            _formatOptionalLogValue(maxQ_),
        )
        core._logAsciiBlock(
            "chromosome fit",
            (
                ("chromosome", chromosome),
                ("intervals", int(numIntervals)),
                ("samples", int(numSamples)),
                (
                    "dependence derived context bp",
                    (
                        int(dependenceContextBP_)
                        if dependenceContextBP_ is not None
                        else "configured"
                    ),
                ),
                (
                    "background configured bp",
                    (
                        int(backgroundBlockSizeBP_)
                        if int(backgroundBlockSizeBP_) > 0
                        else "auto"
                    ),
                ),
                ("background window intervals", int(blockLenIntervals_)),
                (
                    "count noise floor q05",
                    float(countModelFloorQ05) if useCountNoiseFloor else "disabled",
                ),
                ("numeric R floor", float(_MUNC_NUMERIC_VARIANCE_FLOOR)),
                ("maxR", float(maxR_) if maxR_ is not None else "NA"),
                ("minQ", float(minQ_)),
                ("maxQ", float(maxQ_)),
                ("peak calling", bool(peakCallingEnabled)),
            ),
            logger_=logger,
        )
        logger.info(f">>>  Running consenrich: {chromosome}  <<<")
        runStart = time.perf_counter()
        logger.info(
            "runConsenrich.start %s intervals=%d samples=%d blocks=%d",
            chromosome,
            int(numIntervals),
            int(numSamples),
            int(np.ceil(numIntervals / float(blockLenIntervals_))),
        )
        useUncertaintyCalibration = bool(
            outputArgs.writeUncertainty and uncertaintyCalibrationArgs.enabled
        )
        calibrationNeedsBackground = bool(
            useUncertaintyCalibration
            and fitArgs.fitBackground
            and getattr(
                uncertaintyCalibrationArgs,
                "deleteBlockTargetSignal",
                constants.UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_TARGET_SIGNAL,
            )
            == constants.UNCERTAINTY_CALIBRATION_DELETE_BLOCK_TARGET_STATE_PLUS_BACKGROUND
        )
        returnBackgroundTrack = bool(saveBackgroundTracks or calibrationNeedsBackground)
        runResult = core.runConsenrich(
            chromMat,
            muncMat,
            deltaF=deltaF_,
            minQ=minQ_,
            maxQ=maxQ_,
            stateInit=stateArgs.stateInit,
            stateCovarInit=stateArgs.stateCovarInit,
            boundState=stateArgs.boundState,
            stateLowerBound=stateArgs.stateLowerBound,
            stateUpperBound=stateArgs.stateUpperBound,
            blockLenIntervals=blockLenIntervals_,
            intervalSizeBP=intervalSizeBP,
            returnScales=True,
            returnBackground=returnBackgroundTrack,
            pad=pad_,
            ECM_fixedBackgroundIters=fitArgs.ECM_fixedBackgroundIters,
            ECM_fixedBackgroundRtol=fitArgs.ECM_fixedBackgroundRtol,
            t_innerIters=fitArgs.t_innerIters,
            ECM_robustTNu=fitArgs.ECM_robustTNu,
            ECM_useObsPrecisionReweighting=fitArgs.ECM_useObsPrecisionReweighting,
            ECM_useProcessPrecisionReweighting=fitArgs.ECM_useProcessPrecisionReweighting,
            ECM_useAPN=fitArgs.ECM_useAPN,
            fitBackground=fitArgs.fitBackground,
            useNonnegativeBackground=fitArgs.useNonnegativeBackground,
            backgroundNegativePenaltyMultiplier=(
                fitArgs.backgroundNegativePenaltyMultiplier
            ),
            ECM_zeroCenterBackground=fitArgs.ECM_zeroCenterBackground,
            ECM_outerIters=fitArgs.ECM_outerIters,
            ECM_minOuterIters=fitArgs.ECM_minOuterIters,
            ECM_backgroundShiftRtol=fitArgs.ECM_backgroundShiftRtol,
            ECM_outerNLLRtol=fitArgs.ECM_outerNLLRtol,
            ECM_backgroundSmoothness=fitArgs.ECM_backgroundSmoothness,
            **_processNoiseRunKwargs(processArgs),
            observationPrecisionMultiplierMin=observationArgs.precisionMultiplierMin,
            observationPrecisionMultiplierMax=observationArgs.precisionMultiplierMax,
            trackOptimizationPath=outputArgs.plotOptimizationPath,
            returnPrecisionDiagnostics=True,
            returnDiagnostics=True,
            initialBackground=(
                muncResidualBackground if bool(fitArgs.fitBackground) else None
            ),
            logIndentLevel=1,
            logRunRole="primary chromosome" if args.verbose2 else "main chromosome",
        )
        x, P, postFitResiduals, _NISVec, intervalToBlockMap = runResult[:5]
        runResultIndex = 5
        backgroundTrack = None
        if returnBackgroundTrack:
            if len(runResult) <= runResultIndex:
                raise RuntimeError("runConsenrich did not return a background track.")
            backgroundTrack = np.asarray(runResult[runResultIndex], dtype=np.float32)
            runResultIndex += 1
        precisionDiagnostics = {}
        if (
            len(runResult) > runResultIndex
            and isinstance(runResult[runResultIndex], Mapping)
            and runResult[runResultIndex].get("precision_track_diagnostics") is True
        ):
            precisionDiagnostics = runResult[runResultIndex]
            runResultIndex += 1
        runDiagnostics = (
            runResult[runResultIndex]
            if len(runResult) > runResultIndex
            and isinstance(runResult[runResultIndex], Mapping)
            else {}
        )
        if outputArgs.plotOptimizationPath:
            optimizationPathRows = _flattenOptimizationPathDiagnostics(
                chromosome,
                runDiagnostics,
                startOrder=0,
            )
            genomeRecordOffset = len(genomeOptimizationPathRows)
            genomeOptimizationPathRows.extend(
                {
                    **row,
                    "record_order": int(row.get("record_order") or 0)
                    + genomeRecordOffset,
                }
                for row in optimizationPathRows
            )
        if precisionDiagnostics:
            precisionFrame = _precisionDiagnosticsFrame(
                chromosome=chromosome,
                intervals=intervals,
                intervalSizeBP=intervalSizeBP,
                matrixMunc=muncMat,
                pad=pad_,
                precisionDiagnostics=precisionDiagnostics,
            )
            if bool(
                getattr(
                    outputArgs,
                    "plotPrecisionReweightingHistograms",
                    constants.OUTPUT_DEFAULT_PLOT_PRECISION_REWEIGHTING_HISTOGRAMS,
                )
            ):
                histogramColumns = []
                if precisionDiagnostics.get("lambdaExp") is not None:
                    histogramColumns.append("lambda")
                if (
                    precisionDiagnostics.get("processPrecExp") is not None
                    and bool(
                        precisionDiagnostics.get(
                            "process_precision_reweighting_effective",
                            True,
                        )
                    )
                ):
                    histogramColumns.append("kappa")
                if histogramColumns:
                    _mergePrecisionReweightingHistogramSamples(
                        precisionReweightingHistogramSamples,
                        precisionFrame,
                        sampleSize=getattr(
                            outputArgs,
                            "precisionReweightingHistogramSampleSize",
                            constants.OUTPUT_DEFAULT_PRECISION_REWEIGHTING_HISTOGRAM_SAMPLE_SIZE,
                        ),
                        columns=histogramColumns,
                    )
            _appendMuncLambdaDiagnostics(
                precisionFrame,
                diagnosticLogPaths.precision,
                chromosome=chromosome,
                precisionDiagnostics=precisionDiagnostics,
                detail=outputArgs.precisionDiagnosticDetail,
                maxRowsPerChromosome=outputArgs.maxPrecisionDiagnosticRowsPerChromosome,
            )
            _appendProcessKappaDiagnostics(
                precisionFrame,
                diagnosticLogPaths.precision,
                chromosome=chromosome,
                precisionDiagnostics=precisionDiagnostics,
                runDiagnostics=runDiagnostics,
                detail=outputArgs.precisionDiagnosticDetail,
                maxRowsPerChromosome=outputArgs.maxPrecisionDiagnosticRowsPerChromosome,
            )
        logger.info(
            "runConsenrich.done %s elapsed=%.3fs",
            chromosome,
            time.perf_counter() - runStart,
        )
        finalForwardGainSummary = runDiagnostics.get(
            "final_forward_gain_contig_summary"
        )
        if isinstance(finalForwardGainSummary, Mapping):
            updatedReplicates = (
                _updateReplicateGainAccumulator(
                    replicateGainAccumulator,
                    finalForwardGainSummary,
                )
                if replicateGainAccumulator is not None
                else 0
            )
            core._logEvent(
                "replicate.gain.contig",
                (
                    ("chromosome", chromosome),
                    ("updated_replicates", int(updatedReplicates)),
                ),
                logger_=logger,
                level=logging.DEBUG,
            )
        backgroundWarmStart = None
        if bool(fitArgs.fitBackground):
            rawByInterval = np.asarray(chromMat, dtype=np.float32).T
            residualWarmStart = np.asarray(postFitResiduals, dtype=np.float32)
            stateLevelWarmStart = np.asarray(x[:, 0], dtype=np.float32)
            if rawByInterval.shape == residualWarmStart.shape:
                backgroundWarmStart = np.mean(
                    rawByInterval - stateLevelWarmStart[:, None] - residualWarmStart,
                    axis=1,
                    dtype=np.float64,
                ).astype(np.float32)
                if not np.all(np.isfinite(backgroundWarmStart)):
                    backgroundWarmStart = None
        if backgroundWarmStart is None:
            backgroundWarmStart = np.zeros(int(numIntervals), dtype=np.float32)

        initialProcessQWarmStart = None
        processNoiseDiagnostics = runDiagnostics.get("process_noise_calibration")
        if isinstance(processNoiseDiagnostics, Mapping):
            preKappaLevelWarmStart = processNoiseDiagnostics.get("preKappaQLevel")
            preKappaTrendWarmStart = processNoiseDiagnostics.get("preKappaQTrend")
            if (
                core._normalizeStateModel(processArgs.stateModel)
                == core.STATE_MODEL_LEVEL
                and preKappaLevelWarmStart is not None
            ):
                initialProcessQWarmStart = core.constructMatrixQ(
                    minDiagQ=float(minQ_),
                    Q00=float(preKappaLevelWarmStart),
                    Q01=0.0,
                    Q10=0.0,
                    Q11=max(float(preKappaLevelWarmStart), float(minQ_)),
                )
            elif (
                preKappaLevelWarmStart is not None
                and preKappaTrendWarmStart is not None
            ):
                initialProcessQWarmStart = core.constructMatrixQ(
                    minDiagQ=float(minQ_),
                    Q00=float(preKappaLevelWarmStart),
                    Q01=0.0,
                    Q10=0.0,
                    Q11=float(preKappaTrendWarmStart),
                )

        x_ = core.getPrimaryState(
            x,
            stateLowerBound=stateArgs.stateLowerBound,
            stateUpperBound=stateArgs.stateUpperBound,
            boundState=stateArgs.boundState,
        )
        roughnessBlockLen = diagnostics.resolveUncertaintyBlockSizeIntervals(
            uncertaintyCalibrationArgs.blockSizeBP,
            intervalSizeBP,
            len(x_),
            folds=uncertaintyCalibrationArgs.folds,
        )
        stateRoughness = diagnostics.summarizeStateRoughness(
            x_,
            blockLenIntervals=roughnessBlockLen,
            intervalSizeBP=intervalSizeBP,
        )
        roughnessStrata = {
            str(row.get("stratum", "")): row
            for row in stateRoughness.get("signal_strata", [])
            if isinstance(row, Mapping)
        }
        logger.info(
            "stateRoughness[%s]: block=%d intervals (%s bp) meanAbsDiff=%s "
            "blockMedian=%s blockQ90=%s signalLow/Mid/High=%s/%s/%s",
            chromosome,
            int(stateRoughness["block_len_intervals"]),
            _fmtDiagnosticFloat(stateRoughness.get("block_len_bp")),
            _fmtDiagnosticFloat(stateRoughness.get("overall_mean_abs_diff")),
            _fmtDiagnosticFloat(stateRoughness.get("block_mean_abs_diff_median")),
            _fmtDiagnosticFloat(stateRoughness.get("block_mean_abs_diff_q90")),
            _fmtDiagnosticFloat(
                roughnessStrata.get("signal_abs_q00_50", {}).get("mean_abs_diff")
            ),
            _fmtDiagnosticFloat(
                roughnessStrata.get("signal_abs_q50_90", {}).get("mean_abs_diff")
            ),
            _fmtDiagnosticFloat(
                roughnessStrata.get("signal_abs_q90_100", {}).get("mean_abs_diff")
            ),
        )
        precisionBoundaryHits = dict(
            runDiagnostics.get("precision_reweighting_boundary_hits", {})
        )
        obsBoundaryHits = dict(precisionBoundaryHits.get("observation", {}))
        procBoundaryHits = dict(precisionBoundaryHits.get("process", {}))
        logger.info(
            "precisionReweight.boundaryHits[%s]: obs lower=%d upper=%d total=%d; "
            "proc lower=%d upper=%d total=%d",
            chromosome,
            int(obsBoundaryHits.get("lower", 0)),
            int(obsBoundaryHits.get("upper", 0)),
            int(obsBoundaryHits.get("total", 0)),
            int(procBoundaryHits.get("lower", 0)),
            int(procBoundaryHits.get("upper", 0)),
            int(procBoundaryHits.get("total", 0)),
        )
        stateDiagnosticsByChromosome[chromosome] = {
            "state_roughness": stateRoughness,
            "precision_reweighting_boundary_hits": precisionBoundaryHits,
        }
        P00_ = (P[:, 0, 0]).astype(np.float32, copy=False)
        uncertaintyTrack = np.sqrt(P00_).astype(np.float32, copy=False)
        calibrationResult = None
        calibrationModel = None
        deleteBlockFactor = None

        if useUncertaintyCalibration:
            core._logAsciiBlock(
                "uncertainty calibration",
                (
                    ("chromosome", chromosome),
                    ("mode", "delete-block state"),
                    ("intervals", int(len(x_))),
                    ("samples", int(numSamples)),
                    ("block intervals", int(roughnessBlockLen)),
                ),
                logger_=logger,
                indentLevel=1,
            )
            from consenrich import uncertainty as uncertainty_module

            calibrationRunKwargs = dict(
                deltaF=deltaF_,
                minQ=minQ_,
                maxQ=maxQ_,
                stateInit=stateArgs.stateInit,
                stateCovarInit=stateArgs.stateCovarInit,
                boundState=stateArgs.boundState,
                stateLowerBound=stateArgs.stateLowerBound,
                stateUpperBound=stateArgs.stateUpperBound,
                blockLenIntervals=blockLenIntervals_,
                intervalSizeBP=intervalSizeBP,
                returnScales=True,
                pad=pad_,
                ECM_fixedBackgroundIters=fitArgs.ECM_fixedBackgroundIters,
                ECM_fixedBackgroundRtol=fitArgs.ECM_fixedBackgroundRtol,
                t_innerIters=fitArgs.t_innerIters,
                ECM_robustTNu=fitArgs.ECM_robustTNu,
                ECM_useObsPrecisionReweighting=fitArgs.ECM_useObsPrecisionReweighting,
                ECM_useProcessPrecisionReweighting=fitArgs.ECM_useProcessPrecisionReweighting,
                ECM_useAPN=fitArgs.ECM_useAPN,
                fitBackground=fitArgs.fitBackground,
                useNonnegativeBackground=fitArgs.useNonnegativeBackground,
                backgroundNegativePenaltyMultiplier=(
                    fitArgs.backgroundNegativePenaltyMultiplier
                ),
                ECM_zeroCenterBackground=fitArgs.ECM_zeroCenterBackground,
                ECM_outerIters=fitArgs.ECM_outerIters,
                ECM_minOuterIters=fitArgs.ECM_minOuterIters,
                ECM_backgroundShiftRtol=fitArgs.ECM_backgroundShiftRtol,
                ECM_outerNLLRtol=fitArgs.ECM_outerNLLRtol,
                ECM_backgroundSmoothness=fitArgs.ECM_backgroundSmoothness,
                **_processNoiseRunKwargs(processArgs),
                observationPrecisionMultiplierMin=observationArgs.precisionMultiplierMin,
                observationPrecisionMultiplierMax=observationArgs.precisionMultiplierMax,
                initialBackground=backgroundWarmStart,
                initialProcessQ=initialProcessQWarmStart,
                logIndentLevel=2,
                logRunRole="delete-block state calibration fold",
            )
            fullObservationPrecision = None
            if isinstance(precisionDiagnostics, Mapping):
                fullObservationPrecision = precisionDiagnostics.get("lambdaExp")
            calibrationPrefix = (
                f"consenrichOutput_{experimentName}_uncertaintyCalibration"
                f".v{__version__}"
            )
            calibrationResult = uncertainty_module.calibrateChromosomeStateUncertainty(
                matrixData=chromMat,
                matrixMunc=muncMat,
                fullState=x,
                fullCovar=P,
                fullBackground=backgroundTrack,
                fullObservationPrecision=fullObservationPrecision,
                intervals=intervals,
                intervalSizeBP=intervalSizeBP,
                params=uncertaintyCalibrationArgs,
                runKwargs=calibrationRunKwargs,
                outPrefix=calibrationPrefix,
                diagnosticsLogPath=str(diagnosticLogPaths.delete_block_calibration),
                chromosome=chromosome,
            )
            calibrationModel = calibrationResult.model
            deleteBlockFactor = np.asarray(
                calibrationResult.factor,
                dtype=np.float64,
            )
            calibrationModel["delete_block_factor_distribution"] = (
                _deleteBlockFactorDistributionFromArray(deleteBlockFactor)
            )
            deleteBlockLogFields = _deleteBlockFactorLogFields(calibrationModel)
            uncertaintyTrack = np.asarray(
                calibrationResult.calibratedUncertainty,
                dtype=np.float32,
            )
            logger.info(
                "Delete-block state uncertainty calibration applied for %s: "
                "factorModel=%s deletionProbability=%s deletedObservations=%s "
                "sdGlobal=%s sdMedian=%s sdMAD=%s",
                chromosome,
                str(deleteBlockLogFields.get("delete_block_factor_model") or "NA"),
                _fmtDiagnosticFloat(
                    deleteBlockLogFields.get("delete_block_deletion_probability")
                ),
                str(
                    deleteBlockLogFields.get(
                        "delete_block_deleted_observation_interval_total"
                    )
                    or "NA"
                ),
                _fmtDiagnosticFloat(
                    deleteBlockLogFields.get("delete_block_sd_global")
                ),
                _fmtDiagnosticFloat(
                    deleteBlockLogFields.get("delete_block_sd_median")
                ),
                _fmtDiagnosticFloat(deleteBlockLogFields.get("delete_block_sd_mad")),
            )
            if segShrinkGenomeRequested:
                segShrinkDeferredUncertainty.append(
                    {
                        "chromosome": chromosome,
                        "intervals": np.asarray(intervals, dtype=np.int64).copy(),
                        "fullP": np.asarray(P00_, dtype=np.float64).copy(),
                        "model": dict(calibrationResult.model),
                        "factor": deleteBlockFactor.copy(),
                        "calibrated": np.asarray(
                            calibrationResult.calibratedUncertainty,
                            dtype=np.float32,
                        ),
                        "summary": calibrationResult.summary.copy(),
                        "summaryRowIndex": len(runSummaryRows),
                    }
                )
            else:
                blockFactors = _deleteBlockBlockFactorValues(
                    deleteBlockFactor,
                    calibrationModel,
                )
                if blockFactors.size:
                    _mergePrecisionReweightingHistogramSamples(
                        deleteBlockCalibrationPlotSamples,
                        pd.DataFrame({"factor": blockFactors}),
                        sampleSize=getattr(
                            outputArgs,
                            "precisionReweightingHistogramSampleSize",
                            constants.OUTPUT_DEFAULT_PRECISION_REWEIGHTING_HISTOGRAM_SAMPLE_SIZE,
                        ),
                        columns=("factor",),
                    )
                deleteBlockCalibrationCoverageRows.extend(
                    _deleteBlockCoverageRowsForPlot(
                        chromosome=chromosome,
                        calibrationModel=calibrationModel,
                        summary=calibrationResult.summary,
                    )
                )

        if writeStateShrunkTrack:
            shrinkVariance = np.ascontiguousarray(uncertaintyTrack, dtype=np.float32)
            np.multiply(shrinkVariance, shrinkVariance, out=shrinkVariance)
            np.maximum(
                shrinkVariance,
                np.float32(constants.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
                out=shrinkVariance,
            )
            stateShrinkItem = {
                "chromosome": chromosome,
                "start": int(chromosomeStart),
                "intervals": int(len(x_)),
                "state": np.ascontiguousarray(x_, dtype=np.float32),
                "variance": shrinkVariance,
                "summaryRowIndex": len(runSummaryRows),
            }
            stateShrinkDeferred.append(stateShrinkItem)
            stateShrinkDeferredByChromosome[chromosome] = stateShrinkItem

        postFitDiagnostics = _summaryMapping(
            runDiagnostics.get("post_process_noise_fit")
        )
        processQDiagnostics = _summaryMapping(
            runDiagnostics.get("process_q_diagnostics")
        )
        observationRTrace = _summaryMapping(runDiagnostics.get("observation_r_trace"))
        deleteBlockLogFields = _deleteBlockFactorLogFields(calibrationModel)
        _logCliMilestone(
            "Final %s: finalNLL=%s finalForwardNIS=%s "
            "deleteBlockFactorModel=%s deleteBlockSDGlobal=%s "
            "deleteBlockSDMedian=%s deleteBlockSDMAD=%s "
            "deleteBlockSDQ05=%s deleteBlockSDQ95=%s "
            "deleteBlockTrackSDScale=%s "
            "deleteBlockDeletionProbability=%s "
            "deleteBlockDeletedObservations=%s "
            "processQTraceMin=%s processQTraceMax=%s "
            "observationRTraceMin=%s "
            "observationRTraceMax=%s signChangePerKB=%s",
            chromosome,
            _fmtDiagnosticFloat(runDiagnostics.get("final_nll")),
            _fmtDiagnosticFloat(runDiagnostics.get("final_forward_nis")),
            str(deleteBlockLogFields.get("delete_block_factor_model") or "NA"),
            _fmtDiagnosticFloat(
                deleteBlockLogFields.get("delete_block_sd_global")
            ),
            _fmtDiagnosticFloat(deleteBlockLogFields.get("delete_block_sd_median")),
            _fmtDiagnosticFloat(deleteBlockLogFields.get("delete_block_sd_mad")),
            _fmtDiagnosticFloat(deleteBlockLogFields.get("delete_block_sd_q05")),
            _fmtDiagnosticFloat(deleteBlockLogFields.get("delete_block_sd_q95")),
            _fmtDiagnosticFloat(
                deleteBlockLogFields.get("delete_block_track_sd_scale")
            ),
            _fmtDiagnosticFloat(
                deleteBlockLogFields.get("delete_block_deletion_probability")
            ),
            str(
                deleteBlockLogFields.get(
                    "delete_block_deleted_observation_interval_total"
                )
                or "NA"
            ),
            _fmtDiagnosticFloat(processQDiagnostics.get("effectiveQTraceMin")),
            _fmtDiagnosticFloat(processQDiagnostics.get("effectiveQTraceMax")),
            _fmtDiagnosticFloat(observationRTrace.get("min")),
            _fmtDiagnosticFloat(observationRTrace.get("max")),
            _fmtDiagnosticFloat(postFitDiagnostics.get("relative_sign_change_per_kb")),
            blue=True,
        )

        df = pd.DataFrame(
            {
                "Chromosome": chromosome,
                "Start": intervals,
                "End": intervals + intervalSizeBP,
                "State": x_,
            }
        )

        if outputArgs.writeUncertainty:
            df["uncertainty"] = uncertaintyTrack
        if stateDiagnosticTrackNames:
            outputTracks = (
                precisionDiagnostics.get("outputTracks", {})
                if isinstance(precisionDiagnostics, Mapping)
                else {}
            )
            for trackName in stateDiagnosticTrackNames:
                if trackName == "slope":
                    trackValues = np.asarray(x[:, 1], dtype=np.float32)
                else:
                    sourceTrackName = trackName
                    if (
                        not isinstance(outputTracks, Mapping)
                        or sourceTrackName not in outputTracks
                    ):
                        for fallbackName in _OUTPUT_TRACK_FALLBACK_NAMES.get(
                            trackName,
                            (),
                        ):
                            if (
                                isinstance(outputTracks, Mapping)
                                and fallbackName in outputTracks
                            ):
                                sourceTrackName = fallbackName
                                break
                        else:
                            raise RuntimeError(
                                f"Requested diagnostic track {trackName!r} was not returned "
                                f"for {chromosome}."
                            )
                    if not isinstance(outputTracks, Mapping):
                        raise RuntimeError(
                            f"Requested diagnostic track {trackName!r} was not returned "
                            f"for {chromosome}."
                        )
                    trackValues = np.asarray(
                        outputTracks[sourceTrackName],
                        dtype=np.float32,
                    )
                if trackValues.shape != (len(intervals),):
                    raise RuntimeError(
                        f"Diagnostic track {trackName!r} shape mismatch for "
                        f"{chromosome}: expected {(len(intervals),)}, got "
                        f"{trackValues.shape}"
                    )
                df[trackName] = trackValues
        if saveBackgroundTracks:
            if backgroundTrack is None or backgroundTrack.shape != (len(intervals),):
                raise RuntimeError(
                    "Background track shape mismatch for "
                    f"{chromosome}: expected {(len(intervals),)}, got "
                    f"{None if backgroundTrack is None else backgroundTrack.shape}"
                )
            df["background"] = backgroundTrack

        immediateBedGraphTracks = [
            (col, suffix)
            for col, suffix in bedGraphTracks
            if suffix not in stateShrinkDeferredSuffixes
        ]
        cols_ = ["Chromosome", "Start", "End"] + [
            column for column, _suffix in immediateBedGraphTracks
        ]
        df = df[cols_].sort_values(
            by=["Start", "End"],
            kind="mergesort",
        )

        writeStart = time.perf_counter()
        tracksForChromosome = [
            (col, suffix)
            for col, suffix in immediateBedGraphTracks
            if not (segShrinkGenomeRequested and suffix == "uncertainty")
        ]
        for col, suffix in tracksForChromosome:
            bedgraphPath = (
                f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
            )
            logger.info(
                "%s: writing genome-ordered chunk to: %s",
                chromosome,
                bedgraphPath,
            )
            df[["Chromosome", "Start", "End", col]].to_csv(
                bedgraphPath,
                sep="\t",
                header=False,
                index=False,
                mode="w" if c_ == 0 else "a",
                float_format="%.4f",
                lineterminator="\n",
            )
        chromosomeElapsed = time.perf_counter() - chromosomeStartTime
        outputElapsed = time.perf_counter() - writeStart
        logger.info(
            "chromosome.done %s elapsed=%.3fs outputElapsed=%.3fs",
            chromosome,
            chromosomeElapsed,
            outputElapsed,
        )
        runSummaryRows.append(
            _runSummaryRow(
                chromosome=chromosome,
                intervals=int(numIntervals),
                samples=int(numSamples),
                elapsedSeconds=float(chromosomeElapsed),
                outputTrackCount=int(len(bedGraphTracks)),
                runDiagnostics=runDiagnostics,
                stateRoughness=stateRoughness,
                calibrationModel=calibrationModel,
                diagnosticLogPaths=diagnosticLogPaths,
                countNoiseFloorSummary=countNoiseFloorSummaryByChromosome.get(
                    chromosome
                ),
            )
        )
        _logCliProgressMilestone(
            "Chromosome %s done: elapsed=%.1fs outputs=%d",
            chromosome,
            chromosomeElapsed,
            int(len(bedGraphTracks)),
        )

    _logCliPhase(
        "Outputs",
        "tracks=%d",
        int(len(suffixes)),
    )

    if segShrinkGenomeRequested:
        if not segShrinkDeferredUncertainty:
            raise ValueError(
                "segShrink uncertainty calibration has no processed contigs"
            )
        from consenrich import segshrink as segshrink_module

        _logCliProgressMilestone(
            "Uncertainty output: contigs=%d",
            int(len(segShrinkDeferredUncertainty)),
        )
        finalizedSegShrink = segshrink_module.combinePreparedContigs(
            segShrinkDeferredUncertainty,
            positiveFloor=float(constants.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
        )
        uncertaintyBedGraphPath = (
            f"consenrichOutput_{experimentName}_uncertainty.v{__version__}.bedGraph"
        )
        for idx, item in enumerate(finalizedSegShrink):
            chromosome = str(item["chromosome"])
            intervals = np.asarray(item["intervals"], dtype=np.int64)
            calibrated = np.asarray(item["calibrated"], dtype=np.float32)
            if writeStateShrunkTrack:
                shrinkItem = stateShrinkDeferredByChromosome.get(chromosome)
                if shrinkItem is not None:
                    finalShrinkVariance = np.ascontiguousarray(
                        calibrated,
                        dtype=np.float32,
                    )
                    np.multiply(
                        finalShrinkVariance,
                        finalShrinkVariance,
                        out=finalShrinkVariance,
                    )
                    np.maximum(
                        finalShrinkVariance,
                        np.float32(constants.UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR),
                        out=finalShrinkVariance,
                    )
                    shrinkItem["variance"] = finalShrinkVariance
            itemModel = item["model"]
            itemModel["delete_block_factor_distribution"] = (
                _deleteBlockFactorDistributionFromArray(item["factor"])
            )
            itemDeleteBlockFields = _deleteBlockFactorSummaryFields(itemModel)
            itemDeleteBlockLogFields = _deleteBlockFactorLogFields(itemModel)
            dfUncertainty = pd.DataFrame(
                {
                    "Chromosome": chromosome,
                    "Start": intervals,
                    "End": intervals + intervalSizeBP,
                    "uncertainty": calibrated,
                }
            ).sort_values(by=["Start", "End"], kind="mergesort")
            dfUncertainty.to_csv(
                uncertaintyBedGraphPath,
                sep="\t",
                header=False,
                index=False,
                mode="w" if idx == 0 else "a",
                float_format="%.4f",
                lineterminator="\n",
            )
            summaryRowIndex = item.get("summaryRowIndex")
            if isinstance(summaryRowIndex, int) and 0 <= summaryRowIndex < len(
                runSummaryRows
            ):
                runSummaryRows[summaryRowIndex].update(itemDeleteBlockFields)
            logger.info(
                "deleteBlockFactor.finalized %s factorModel=%s "
                "deletionProbability=%s deletedObservations=%s "
                "sdMedian=%s sdMAD=%s sdQ05=%s sdQ95=%s",
                chromosome,
                str(
                    itemDeleteBlockLogFields.get("delete_block_factor_model") or "NA"
                ),
                _fmtDiagnosticFloat(
                    itemDeleteBlockLogFields.get("delete_block_deletion_probability")
                ),
                str(
                    itemDeleteBlockLogFields.get(
                        "delete_block_deleted_observation_interval_total"
                    )
                    or "NA"
                ),
                _fmtDiagnosticFloat(
                    itemDeleteBlockLogFields.get("delete_block_sd_median")
                ),
                _fmtDiagnosticFloat(itemDeleteBlockLogFields.get("delete_block_sd_mad")),
                _fmtDiagnosticFloat(
                    itemDeleteBlockLogFields.get("delete_block_sd_q05")
                ),
                _fmtDiagnosticFloat(
                    itemDeleteBlockLogFields.get("delete_block_sd_q95")
                ),
            )
            if bool(getattr(uncertaintyCalibrationArgs, "writeDiagnostics", True)):
                _appendMappingDiagnostics(
                    diagnosticLogPaths.delete_block_calibration,
                    recordType="model",
                    event="delete_block_calibration.segShrink.processed_genome_model",
                    chromosome=chromosome,
                    values=itemModel,
                )
            blockFactors = _deleteBlockBlockFactorValues(item["factor"], itemModel)
            if blockFactors.size:
                _mergePrecisionReweightingHistogramSamples(
                    deleteBlockCalibrationPlotSamples,
                    pd.DataFrame({"factor": blockFactors}),
                    sampleSize=getattr(
                        outputArgs,
                        "precisionReweightingHistogramSampleSize",
                        constants.OUTPUT_DEFAULT_PRECISION_REWEIGHTING_HISTOGRAM_SAMPLE_SIZE,
                    ),
                    columns=("factor",),
                )
            itemSummary = item.get("summary")
            deleteBlockCalibrationCoverageRows.extend(
                _deleteBlockCoverageRowsForPlot(
                    chromosome=chromosome,
                    calibrationModel=itemModel,
                    summary=(
                        itemSummary
                        if isinstance(itemSummary, pd.DataFrame)
                        else None
                    ),
                )
            )
        logger.info(
            "segShrink processed-genome finalization wrote uncertainty bedGraph for %d contigs: %s",
            int(len(finalizedSegShrink)),
            uncertaintyBedGraphPath,
        )

    if writeStateShrunkTrack:
        if not stateShrinkDeferred:
            raise ValueError("state shrinkage requested but no contigs were processed")
        stateShrinkPrior = shrinkState.fitStateShrinkagePrior(
            stateShrinkDeferred,
            model=getattr(outputArgs, "stateShrinkageModel", None),
            priorSpikeProp=getattr(outputArgs, "stateShrinkagePriorSpikeProp", None),
            priorScale=getattr(outputArgs, "stateShrinkagePriorScale", None),
            stateShrinkageSpikePseudoCount=outputArgs.stateShrinkageSpikePseudoCount,
            stateShrinkageScaleAnchorWeight=getattr(
                outputArgs,
                "stateShrinkageScaleAnchorWeight",
                constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT,
            ),
            studentTDF=getattr(
                outputArgs,
                "stateShrinkageStudentTDF",
                constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_STUDENT_T_DF,
            ),
            studentTQuadratureOrder=getattr(
                outputArgs,
                "stateShrinkageStudentTQuadratureOrder",
                constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_STUDENT_T_QUADRATURE_ORDER,
            ),
            blockSize=int(stateShrinkBlockIntervals),
        )
        logger.info(
            "stateShrinkage.prior: model=%s scope=%s blockIntervals=%d "
            "effectiveBlocks=%s priorSpikeProp=%s priorScale=%s iterations=%d "
            "converged=%s slabCount=%s slabWeight=%s slabVariance=%s "
            "componentWeights=%s",
            str(stateShrinkPrior.metadata.get("model") or "NA"),
            str(stateShrinkPrior.metadata.get("scope") or "NA"),
            int(stateShrinkPrior.metadata.get("block_size_intervals") or 1),
            _fmtDiagnosticFloat(stateShrinkPrior.metadata.get("effective_block_count")),
            _fmtDiagnosticFloat(stateShrinkPrior.metadata.get("prior_spike_prop")),
            _fmtDiagnosticFloat(stateShrinkPrior.metadata.get("prior_scale")),
            int(stateShrinkPrior.metadata.get("iterations") or 0),
            bool(stateShrinkPrior.metadata.get("converged")),
            _diagnosticJsonText(stateShrinkPrior.metadata.get("slab_count")),
            _diagnosticJsonText(stateShrinkPrior.metadata.get("slab_weight")),
            _diagnosticJsonText(stateShrinkPrior.metadata.get("slab_variance")),
            _diagnosticJsonText(stateShrinkPrior.metadata.get("component_weights")),
        )
        stateShrinkGenomeSummaryFields = _stateShrinkageSummaryFields(
            stateShrinkPrior.metadata
        )
        shrinkOutputTracks = _stateShrinkageOutputTracks(writeStateShrinkageTracks)
        for idx, item in enumerate(stateShrinkDeferred):
            chromosome = str(item["chromosome"])
            intervalCount = int(item["intervals"])
            intervals = np.arange(
                int(item["start"]),
                int(item["start"]) + intervalCount * intervalSizeBP,
                intervalSizeBP,
                dtype=np.int64,
            )
            stateShrinkageResult = shrinkState.applyStateShrinkagePrior(
                item["state"],
                item["variance"],
                stateShrinkPrior,
                stateShrinkageSpikeOddsMultiplier=getattr(
                    outputArgs,
                    "stateShrinkageSpikeOddsMultiplier",
                    constants.OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_ODDS_MULTIPLIER,
                ),
            )
            stateDiagnosticsByChromosome.setdefault(chromosome, {})[
                "state_shrinkage"
            ] = _jsonDiagnosticValue(dict(stateShrinkageResult.metadata))
            summaryRowIndex = item.get("summaryRowIndex")
            if isinstance(summaryRowIndex, int) and 0 <= summaryRowIndex < len(
                runSummaryRows
            ):
                runSummaryRows[summaryRowIndex].update(
                    _stateShrinkageSummaryFields(stateShrinkageResult.metadata)
                )
            dfShrink = pd.DataFrame(
                {
                    "Chromosome": chromosome,
                    "Start": intervals,
                    "End": intervals + intervalSizeBP,
                    "stateShrunk": stateShrinkageResult.shrunkState,
                    "stateShrunkUncertainty": stateShrinkageResult.posteriorSd,
                }
            )
            if writeStateShrinkageTracks:
                dfShrink["stateSpikeProp"] = stateShrinkageResult.spikeProp
            dfShrink = dfShrink.sort_values(
                by=["Start", "End"],
                kind="mergesort",
            )
            for col, suffix in shrinkOutputTracks:
                bedgraphPath = (
                    f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
                )
                logger.info(
                    "%s: writing genome-ordered state-shrinkage chunk to: %s",
                    chromosome,
                    bedgraphPath,
                )
                dfShrink[["Chromosome", "Start", "End", col]].to_csv(
                    bedgraphPath,
                    sep="\t",
                    header=False,
                    index=False,
                    mode="w" if idx == 0 else "a",
                    float_format="%.4f",
                    lineterminator="\n",
                )

    if replicateGainAccumulator is not None:
        gainRows = _replicateGainSummaryRows(
            treatmentSources,
            replicateGainAccumulator,
            controlSources=(controlSources if controlsPresent else None),
        )
        _writeReplicateGainSummary(
            gainRows,
            _replicateGainSummaryPath(str(experimentName)),
        )
        _plotReplicateCalibration(
            gainRows,
            _replicateCalibrationPlotPath(str(experimentName)),
            dpi=400,
        )

    if correlationLengthRow_ is not None:
        _writeCorrelationLengthSummary(
            correlationLengthRow_,
            correlationLengthPath,
        )
        if correlationLengthDiagnostics_ is None:
            raise RuntimeError("correlation-length diagnostics are unavailable")
        _writeCorrelationLengthWindows(
            correlationLengthDiagnostics_,
            correlationLengthWindowsPath,
        )
        if (
            bool(
                getattr(
                    outputArgs,
                    "plotCorrelationLength",
                    constants.OUTPUT_DEFAULT_PLOT_CORRELATION_LENGTH,
                )
            )
        ):
            _plotCorrelationLengthInference(
                correlationLengthRow_,
                correlationLengthDiagnostics_,
                correlationLengthPlotPath,
                dpi=400,
            )

    if outputArgs.plotOptimizationPath and genomeOptimizationPathRows:
        genomeOptimizationPathPrefix = _genomeOptimizationPathPrefix(
            str(experimentName)
        )
        _appendConvergenceDiagnostics(
            genomeOptimizationPathRows,
            diagnosticLogPaths.convergence,
        )
        _plotGenomeOptimizationPathLog(
            genomeOptimizationPathRows,
            f"{genomeOptimizationPathPrefix}.png",
            dpi=400,
        )

    if (
        bool(
            getattr(
                outputArgs,
                "plotPrecisionReweightingHistograms",
                constants.OUTPUT_DEFAULT_PLOT_PRECISION_REWEIGHTING_HISTOGRAMS,
            )
        )
        and (
            precisionReweightingHistogramSamples["lambda"].size
            or precisionReweightingHistogramSamples["kappa"].size
        )
    ):
        _plotPrecisionReweightingHistograms(
            precisionReweightingHistogramSamples,
            _precisionReweightingHistogramPath(str(experimentName)),
            dpi=400,
        )

    if (
        bool(outputArgs.writeUncertainty and uncertaintyCalibrationArgs.enabled)
        and (
            deleteBlockCalibrationPlotSamples["factor"].size
            or deleteBlockCalibrationCoverageRows
        )
    ):
        _plotDeleteBlockCalibration(
            deleteBlockCalibrationPlotSamples,
            deleteBlockCalibrationCoverageRows,
            _deleteBlockCalibrationPlotPath(str(experimentName)),
            dpi=400,
        )

    if bool(getattr(outputArgs, "writeRunSummary", True)):
        summaryRows = list(runSummaryRows)
        if summaryRows:
            genomeSummaryRow = _genomeRunSummaryRow(
                summaryRows,
                elapsedSeconds=time.perf_counter() - cliRunStart,
                diagnosticLogPaths=diagnosticLogPaths,
            )
            genomeSummaryRow.update(stateShrinkGenomeSummaryFields)
            summaryRows.append(genomeSummaryRow)
        _writeRunSummary(summaryRows, _runSummaryPath(str(experimentName)))

    _logCliProgressMilestone(
        "Validating outputs: tracks=%d",
        int(len(suffixes)),
    )

    validatedBedGraphs = set()
    for suffix in suffixes:
        bedgraphPath = (
            f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
        )
        try:
            if bedGraphPlanOrderDiffers:
                _sortBedGraphInPlace(bedgraphPath, chromOrder=bedGraphChromOrder)
            _validateBedGraphSorted(bedgraphPath, chromOrder=bedGraphChromOrder)
            validatedBedGraphs.add(os.path.abspath(bedgraphPath))
        except Exception as ex:
            logger.warning(
                "bedGraph %s failed genome-order validation; sorting as fallback:\n%s",
                bedgraphPath,
                ex,
            )
            try:
                _sortBedGraphInPlace(bedgraphPath, chromOrder=bedGraphChromOrder)
                _validateBedGraphSorted(bedgraphPath, chromOrder=bedGraphChromOrder)
                validatedBedGraphs.add(os.path.abspath(bedgraphPath))
            except Exception as sortEx:
                logger.warning(f"Failed to sort {bedgraphPath}:\n{sortEx}")

    if peakCallingEnabled:
        try:
            stateScoreSuffix = "state"
            uncertaintyScoreSuffix = "uncertainty"
            exportSignalSuffix = "stateShrunk" if useShrunkStateScores else "state"
            stateBedGraphPath = (
                f"consenrichOutput_{experimentName}_{stateScoreSuffix}.v{__version__}.bedGraph"
            )
            uncertaintyBedGraphPath = (
                f"consenrichOutput_{experimentName}_{uncertaintyScoreSuffix}.v{__version__}.bedGraph"
            )
            exportSignalBedGraphPath = (
                f"consenrichOutput_{experimentName}_{exportSignalSuffix}.v{__version__}.bedGraph"
                if useShrunkStateScores
                else None
            )
            _logCliSubphase(
                "ROCCO peaks: scoreTrack=%s path=%s uncertaintyTrack=%s exportSignalTrack=%s",
                stateScoreSuffix,
                stateBedGraphPath,
                uncertaintyScoreSuffix,
                exportSignalSuffix,
            )
            if not os.path.exists(uncertaintyBedGraphPath):
                if matchingArgs.uncertaintyScoreMode == "lower_confidence":
                    raise FileNotFoundError(
                        "matchingParams.uncertaintyScoreMode='lower_confidence' "
                        f"requires uncertainty bedGraph {uncertaintyBedGraphPath}."
                    )
                logger.warning(
                    "Uncertainty bedGraph %s was not found; proceeding without model-based uncertainty.",
                    uncertaintyBedGraphPath,
                )
                uncertaintyBedGraphPath = None
            if exportSignalBedGraphPath is not None and not os.path.exists(
                exportSignalBedGraphPath
            ):
                raise FileNotFoundError(
                    "matchingParams.useShrunkStateScores requires "
                    f"stateShrunk bedGraph {exportSignalBedGraphPath}."
                )
            outName, roccoSummary = peaks.solveRocco(
                stateBedGraphPath,
                uncertaintyBedGraphFile=uncertaintyBedGraphPath,
                exportSignalBedGraphFile=exportSignalBedGraphPath,
                numBootstrap=int(matchingArgs.numBootstrap),
                thresholdZ=float(matchingArgs.thresholdZ),
                dependenceSpan=matchingArgs.dependenceSpan,
                gamma=matchingArgs.gamma,
                selectionPenalty=matchingArgs.selectionPenalty,
                gammaScale=float(matchingArgs.gammaScale),
                nestedRoccoIters=int(matchingArgs.nestedRoccoIters),
                nestedRoccoBudgetScale=float(matchingArgs.nestedRoccoBudgetScale),
                exportFilterUncertaintyMultiplier=float(
                    matchingArgs.exportFilterUncertaintyMultiplier
                ),
                minPeakScore=matchingArgs.minPeakScore,
                peakMode=matchingArgs.peakMode,
                broadWeakThresholdZ=float(matchingArgs.broadWeakThresholdZ),
                broadMaxGapBP=matchingArgs.broadMaxGapBP,
                uncertaintyScoreMode=matchingArgs.uncertaintyScoreMode,
                uncertaintyScoreZ=float(matchingArgs.uncertaintyScoreZ),
                blacklistBedFile=genomeArgs.blacklistFile,
                randSeed=matchingArgs.randSeed,
                verbose=bool(args.verbose),
                metadataDetail=matchingArgs.metadataDetail,
                maxNonTrackFileBytes=outputArgs.maxNonTrackFileBytes,
                stateDiagnosticsByChromosome=stateDiagnosticsByChromosome,
                returnSummary=True,
            )

            logger.info("Finished ROCCO peak calling. Written to %s", outName)
            if bool(outputArgs.cutoffReport):
                try:
                    roccoFormatSummaries = roccoSummary.get("per_format", {})
                    if not isinstance(roccoFormatSummaries, Mapping):
                        roccoFormatSummaries = {}
                    roccoNarrowSummary = (
                        roccoFormatSummaries.get("narrowPeak")
                        if matchingArgs.peakMode == "both"
                        else roccoSummary
                        if matchingArgs.peakMode == "narrow"
                        else None
                    )
                    roccoNarrowPeakPath = roccoSummary.get("narrowPeak_path")
                    cutoffReportDir = peaks.solveRoccoCutoffReport(
                        stateBedGraphPath,
                        uncertaintyBedGraphFile=uncertaintyBedGraphPath,
                        exportSignalBedGraphFile=exportSignalBedGraphPath,
                        numBootstrap=int(matchingArgs.numBootstrap),
                        thresholdZ=float(matchingArgs.thresholdZ),
                        dependenceSpan=matchingArgs.dependenceSpan,
                        gamma=matchingArgs.gamma,
                        selectionPenalty=matchingArgs.selectionPenalty,
                        gammaScale=float(matchingArgs.gammaScale),
                        nestedRoccoIters=int(matchingArgs.nestedRoccoIters),
                        nestedRoccoBudgetScale=float(
                            matchingArgs.nestedRoccoBudgetScale
                        ),
                        exportFilterUncertaintyMultiplier=float(
                            matchingArgs.exportFilterUncertaintyMultiplier
                        ),
                        minPeakScore=matchingArgs.minPeakScore,
                        uncertaintyScoreMode=matchingArgs.uncertaintyScoreMode,
                        uncertaintyScoreZ=float(matchingArgs.uncertaintyScoreZ),
                        blacklistBedFile=genomeArgs.blacklistFile,
                        randSeed=matchingArgs.randSeed,
                        baselineNarrowPeakFile=(
                            None
                            if roccoNarrowPeakPath is None
                            else str(roccoNarrowPeakPath)
                        ),
                        baselineSummary=roccoNarrowSummary,
                    )
                    logger.info(
                        "Finished ROCCO cutoff report. Written to %s",
                        cutoffReportDir,
                    )
                except Exception as cutoffEx:
                    logger.warning(
                        "ROCCO cutoff report raised an exception:\n\n\t%s\n"
                        "Skipping cutoff-report step...",
                        cutoffEx,
                    )
        except Exception as ex_:
            logger.warning(
                f"ROCCO peak calling raised an exception:\n\n\t{ex_}\n"
                f"Skipping peak-calling step...try running post hoc via `consenrich --match-bedGraph <bedGraphFile>`\n"
                f"\tSee ``consenrich -h`` for more details.\n"
            )

    if outputArgs.convertToBigWig:
        convertBedGraphToBigWig(
            experimentName,
            genomeArgs.chromSizesFile,
            suffixes=suffixes,
            validatedBedGraphs=validatedBedGraphs,
            deleteBedGraphsAfterBigWig=outputArgs.deleteBedGraphsAfterBigWig,
        )

    _logCliMilestone(
        "Consenrich run done: experiment=%s elapsed=%.1fs",
        experimentName,
        time.perf_counter() - cliRunStart,
    )


if __name__ == "__main__":
    main()
