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
from tqdm import tqdm

import consenrich.core as core
import consenrich.diagnostics as diagnostics
import consenrich.detrorm as detrorm
import consenrich.peaks as peaks
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
from .config import loadConfig, readConfig
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
_AUDIT_LOG_FORMAT = (
    "%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s"
)

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

GAIN_SUMMARY_COLUMNS = [
    "replicate_index",
    "sample_name",
    "treatment_path",
    "control_path",
    "chromosome_count",
    "finite_interval_count",
    "gain_avg",
    "gain_std",
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

TUNC_KAPPA_LOG_COLUMNS = [
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
    "tuncQScale",
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
    "process_noise_status",
    "process_noise_reason",
    "lambda_lower_bound_hits",
    "lambda_upper_bound_hits",
    "kappa_lower_bound_hits",
    "kappa_upper_bound_hits",
    "state_roughness_mean_abs_diff",
    "state_roughness_block_median",
    "state_roughness_block_q90",
    "delete_block_global_factor",
    "delete_block_rows_valid",
    "delete_block_rows_fit",
    "delete_block_scale",
    "delete_block_scale_reason",
    "munc_lambda_log",
    "tunc_kappa_log",
    "convergence_log",
    "delete_block_calibration_log",
]


class DiagnosticLogPaths(NamedTuple):
    munc_lambda: Path
    tunc_kappa: Path
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


def _countModelVarianceFloorForScaledCounts(
    scaledCounts: np.ndarray,
    scaleFactor: float,
    countingArgs: core.countingParams,
    *,
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
    return np.asarray(
        core.transformCountVarianceFloor(
            counts,
            [scaleFactor_],
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
    floor = np.full(counts.shape, np.nan, dtype=np.float64)
    for j in range(int(counts.shape[0])):
        source = sources[j] if j < len(sources) else None
        floor[j, :] = _countModelVarianceFloorForScaledCounts(
            counts[j, :],
            float(scaleArr[j]),
            countingArgs,
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
        ("process_noise_warmup_fit", "process_noise_warmup"),
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
        munc_lambda=Path(f"{prefix}_munc_lambda.v{__version__}.log"),
        tunc_kappa=Path(f"{prefix}_tunc_kappa.v{__version__}.log"),
        convergence=Path(f"{prefix}_convergence.v{__version__}.log"),
        delete_block_calibration=Path(
            f"{prefix}_delete_block_calibration.v{__version__}.log"
        ),
    )


def _runSummaryPath(experimentName: str) -> Path:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return Path(f"consenrichOutput_{experimentToken}_summary.v{__version__}.tsv")


def _initializeDiagnosticLogs(paths: DiagnosticLogPaths) -> None:
    logging_utils.init_tsv_log(paths.munc_lambda, MUNC_LAMBDA_LOG_COLUMNS)
    logging_utils.init_tsv_log(paths.tunc_kappa, TUNC_KAPPA_LOG_COLUMNS)
    logging_utils.init_tsv_log(paths.convergence, CONVERGENCE_LOG_COLUMNS)
    logging_utils.init_tsv_log(
        paths.delete_block_calibration,
        DELETE_BLOCK_CALIBRATION_LOG_COLUMNS,
    )


def _jsonDiagnosticValue(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonDiagnosticValue(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonDiagnosticValue(item) for item in value]
    if isinstance(value, (float, np.floating)):
        value_ = float(value)
        return value_ if np.isfinite(value_) else None
    return value


def _appendKeyValueDiagnostics(
    path: Path,
    columns: Sequence[str],
    *,
    recordType: str,
    event: str,
    chromosome: str | None,
    values: Mapping[str, Any],
) -> int:
    rows = []
    for key, value in values.items():
        rows.append(
            {
                "record_type": recordType,
                "event": event,
                "chromosome": chromosome,
                "key": str(key),
                "value": (
                    json.dumps(_jsonDiagnosticValue(value), sort_keys=True)
                    if isinstance(value, (Mapping, list, tuple, np.ndarray))
                    else _jsonDiagnosticValue(value)
                ),
            }
        )
    return logging_utils.append_tsv_log(path, rows, columns)


def _genomeOptimizationPathPrefix(experimentName: str) -> str:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return f"consenrichOutput_{experimentToken}_genome_optimizationPath.v{__version__}"


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
        if len(plotOrder) == 1:
            normalizedX = np.array([0.0], dtype=np.float64)
        else:
            normalizedX = (plotOrder - 1.0) / float(len(plotOrder) - 1)
        finalImprovement = float(improvement[-1])
        denominator = finalImprovement if abs(finalImprovement) > 1.0e-12 else 1.0
        normalizedImprovement = improvement / denominator
        if len(normalizedX) == 1:
            interpolated = np.full_like(
                interpolationGrid,
                float(normalizedImprovement[0]),
                dtype=np.float64,
            )
        else:
            interpolated = np.interp(
                interpolationGrid,
                normalizedX,
                normalizedImprovement,
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
            normalizedImprovement,
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
            [normalizedImprovement[-1]],
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
    normAx.set_ylabel("Fraction of final NLL improvement", color=darkBlack)
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
    lambdaArr = np.nan_to_num(lambdaArr, nan=1.0, posinf=1.0, neginf=1.0)
    lambdaArr = np.maximum(lambdaArr, np.finfo(np.float64).tiny)

    kappaArr = np.ones(n, dtype=np.float64)
    if processPrecExp is not None:
        kappaArr = np.asarray(processPrecExp, dtype=np.float64).reshape(-1)
        if kappaArr.size != n:
            raise RuntimeError(
                f"processPrecExp length mismatch for {chromosome}: expected {n}, got {kappaArr.size}"
            )
    kappaArr = np.nan_to_num(kappaArr, nan=1.0, posinf=1.0, neginf=1.0)
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

    munc = np.asarray(matrixMunc, dtype=np.float64)
    if munc.ndim != 2 or munc.shape[1] != n:
        raise RuntimeError(
            f"matrixMunc shape mismatch for {chromosome}: expected second dimension {n}, got {munc.shape}"
        )
    medianDiagR = np.nanmedian(munc + float(pad), axis=0)
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
) -> int:
    if frame.empty:
        return 0
    lambdaValues = pd.to_numeric(frame["lambda"]).to_numpy(dtype=np.float64)
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
            "chromosome": frame["Chromosome"],
            "start": frame["Start"],
            "end": frame["End"],
            "interval": frame["Interval"],
            "lambda": frame["lambda"],
            "lambda_lower_bound_hit": lambdaLowerHit,
            "lambda_upper_bound_hit": lambdaUpperHit,
            "median_diag_R": frame["median_diag_R"],
            "median_effective_diag_R": frame["median_effective_diag_R"],
        }
    )
    outputTracks = precisionDiagnostics.get("outputTracks", {})
    if isinstance(outputTracks, Mapping):
        for name in ("muncTrace", "sumGain0", "sumGain1"):
            if name in outputTracks:
                out[name] = np.asarray(outputTracks[name], dtype=np.float64).reshape(-1)
    rowsWritten = logging_utils.append_tsv_log(path, out, MUNC_LAMBDA_LOG_COLUMNS)
    _appendKeyValueDiagnostics(
        path,
        MUNC_LAMBDA_LOG_COLUMNS,
        recordType="summary",
        event="munc_lambda.summary",
        chromosome=chromosome,
        values={
            "rows": int(rowsWritten),
            "lambda_median": float(pd.to_numeric(frame["lambda"]).median()),
            "median_diag_R": float(pd.to_numeric(frame["median_diag_R"]).median()),
            "median_effective_diag_R": float(
                pd.to_numeric(frame["median_effective_diag_R"]).median()
            ),
        },
    )
    return rowsWritten


def _appendTuncKappaDiagnostics(
    frame: pd.DataFrame,
    path: Path,
    *,
    chromosome: str,
    precisionDiagnostics: Mapping[str, Any],
    runDiagnostics: Mapping[str, Any],
) -> int:
    if frame.empty:
        return 0
    kappaValues = pd.to_numeric(frame["kappa"]).to_numpy(dtype=np.float64)
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
            "event": "tunc_kappa.interval",
            "chromosome": frame["Chromosome"],
            "start": frame["Start"],
            "end": frame["End"],
            "interval": frame["Interval"],
            "kappa": frame["kappa"],
            "kappa_lower_bound_hit": kappaLowerHit,
            "kappa_upper_bound_hit": kappaUpperHit,
            "process_q_policy": frame["process_q_policy"],
            "apn_enabled": frame["apn_enabled"],
            "process_precision_reweighting_requested": frame[
                "process_precision_reweighting_requested"
            ],
            "process_precision_reweighting_effective": frame[
                "process_precision_reweighting_effective"
            ],
            "process_precision_reweighting_disabled_by_apn": frame[
                "process_precision_reweighting_disabled_by_apn"
            ],
            "baseQ00": frame["baseQ00"],
            "baseQ11": frame["baseQ11"],
            "preKappaQLevel": frame["baseQ00"],
            "preKappaQTrend": frame["baseQ11"],
            "effectiveQLevel": frame["effectiveQ00"],
            "effectiveQTrend": frame["effectiveQ11"],
        }
    )
    outputTracks = precisionDiagnostics.get("outputTracks", {})
    if isinstance(outputTracks, Mapping) and "tuncQScale" in outputTracks:
        out["tuncQScale"] = np.asarray(
            outputTracks["tuncQScale"],
            dtype=np.float64,
        ).reshape(-1)
    rowsWritten = logging_utils.append_tsv_log(path, out, TUNC_KAPPA_LOG_COLUMNS)
    processNoise = runDiagnostics.get("process_noise_calibration")
    if isinstance(processNoise, Mapping):
        _appendKeyValueDiagnostics(
            path,
            TUNC_KAPPA_LOG_COLUMNS,
            recordType="summary",
            event="tunc_kappa.process_noise_calibration",
            chromosome=chromosome,
            values=processNoise,
        )
    _appendKeyValueDiagnostics(
        path,
        TUNC_KAPPA_LOG_COLUMNS,
        recordType="summary",
        event="tunc_kappa.summary",
        chromosome=chromosome,
        values={
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
    frame = pd.DataFrame(
        ({"record_type": "trace", **dict(row)} for row in rows),
        columns=CONVERGENCE_LOG_COLUMNS,
    )
    rowsWritten = logging_utils.append_tsv_log(path, frame, CONVERGENCE_LOG_COLUMNS)
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
) -> dict[str, Any]:
    precisionHits = _summaryMapping(
        runDiagnostics.get("precision_reweighting_boundary_hits")
    )
    obsHits = _summaryMapping(precisionHits.get("observation"))
    procHits = _summaryMapping(precisionHits.get("process"))
    processNoise = _summaryMapping(runDiagnostics.get("process_noise_calibration"))
    calibration = _summaryMapping(calibrationModel)
    targetCalibration = _summaryMapping(calibration.get("target_calibration"))
    return {
        "record_type": "chromosome",
        "chromosome": chromosome,
        "intervals": int(intervals),
        "samples": int(samples),
        "elapsed_seconds": float(elapsedSeconds),
        "output_track_count": int(outputTrackCount),
        "final_nll": _summaryNumber(runDiagnostics.get("final_nll")),
        "final_forward_nis": _summaryNumber(
            runDiagnostics.get("final_forward_nis")
        ),
        "process_q_policy": runDiagnostics.get("process_q_policy"),
        "process_noise_status": processNoise.get("processNoiseCalibrationStatus"),
        "process_noise_reason": processNoise.get("processNoiseCalibrationReason"),
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
        "delete_block_global_factor": _summaryNumber(
            calibration.get("global_factor")
        ),
        "delete_block_rows_valid": _summaryInt(calibration.get("rows_valid")),
        "delete_block_rows_fit": _summaryInt(calibration.get("rows_fit")),
        "delete_block_scale": _summaryNumber(
            targetCalibration.get("uncertainty_track_scale")
        ),
        "delete_block_scale_reason": targetCalibration.get(
            "uncertainty_track_scale_reason"
        ),
        "munc_lambda_log": str(diagnosticLogPaths.munc_lambda),
        "tunc_kappa_log": str(diagnosticLogPaths.tunc_kappa),
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
        "munc_lambda_log": str(diagnosticLogPaths.munc_lambda),
        "tunc_kappa_log": str(diagnosticLogPaths.tunc_kappa),
        "convergence_log": str(diagnosticLogPaths.convergence),
        "delete_block_calibration_log": str(
            diagnosticLogPaths.delete_block_calibration
        ),
    }


def _writeRunSummary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(list(rows), columns=RUN_SUMMARY_COLUMNS)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", na_rep="NA")
    logging_utils.log_file_written(
        logger,
        event="artifact.run_summary",
        path=str(path),
        fields=(("rows", int(len(frame))),),
    )


def _replicateGainSummaryPath(experimentName: str) -> str:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    return f"consenrichOutput_{experimentToken}_replicateGains.v{__version__}.tsv"


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
        controlPath = str(controlSources_[i].path) if i < len(controlSources_) else None
        count = int(finiteCounts[i])
        if count > 0:
            avg = float(sums[i] / float(count))
            variance = max(float(sumSqs[i] / float(count)) - (avg * avg), 0.0)
            std = float(math.sqrt(variance))
        else:
            avg = float("nan")
            std = float("nan")
        rows.append(
            {
                "replicate_index": int(i + 1),
                "sample_name": sourceId,
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
    frame.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        float_format="%.7g",
        lineterminator="\n",
        na_rep="NA",
    )
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
        "processNoiseWarmupECMIters": processArgs.processNoiseWarmupECMIters,
        "processPrecisionMultiplierMin": processArgs.precisionMultiplierMin,
        "processPrecisionMultiplierMax": processArgs.precisionMultiplierMax,
    }
    if _coreRunConsenrichSupports("processNoiseWarmupOuterPasses"):
        kwargs["processNoiseWarmupOuterPasses"] = warmupOuterPasses
    for parameterName in (
        "processNoiseCalibration",
        "tuncLocalWindowMultiplier",
        "tuncDependenceMultiplier",
        "tuncMinScale",
        "tuncMaxScale",
        "tuncMinWindowWeight",
        "tuncPriorRidge",
        "tuncLevelBufferZ",
        "tuncProcessCovariatesEnabled",
        "tuncProcessCovariatesMode",
        "tuncProcessCovariatesFeatures",
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
            "global median center",
            _resolveGlobalMedianCenterStatus(
                countingArgs,
                controlsPresent=controlsPresent,
            )[1],
        ),
        ("MUNC variance model", observationArgs.muncVarianceModel),
        (
            "MUNC AR1 variance functional",
            getattr(
                observationArgs,
                "muncAR1VarianceFunctional",
                constants.OBSERVATION_DEFAULT_MUNC_AR1_VARIANCE_FUNCTIONAL,
            ),
        ),
        (
            "MUNC AR1 max beta",
            constants.MUNC_AR1_MAX_BETA_DEFAULT,
        ),
        (
            "MUNC AR1 pairs reg lambda",
            constants.MUNC_AR1_PAIRS_REG_LAMBDA_DEFAULT,
        ),
        ("MUNC variance EB", yn(observationArgs.EB_use)),
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
            "process noise calibration",
            getattr(
                processArgs,
                "processNoiseCalibration",
                constants.PROCESS_DEFAULT_NOISE_CALIBRATION,
            ),
        ),
        (
            "TUNC scale bounds",
            (
                f"[{float(getattr(processArgs, 'tuncMinScale', constants.PROCESS_DEFAULT_TUNC_MIN_SCALE)):.6g}, "
                f"{float(getattr(processArgs, 'tuncMaxScale', constants.PROCESS_DEFAULT_TUNC_MAX_SCALE)):.6g}]"
            ),
        ),
        (
            "TUNC level buffer z",
            float(
                getattr(
                    processArgs,
                    "tuncLevelBufferZ",
                    constants.PROCESS_DEFAULT_TUNC_LEVEL_BUFFER_Z,
                )
            ),
        ),
        (
            "process kappa bounds",
            (
                f"[{float(processArgs.precisionMultiplierMin):.6g}, "
                f"{float(processArgs.precisionMultiplierMax):.6g}]"
            ),
        ),
        (
            "process noise warmup",
            f"{_getProcessNoiseWarmupOuterPasses(processArgs)} outer passes x "
            f"{int(processArgs.processNoiseWarmupECMIters)} ECM iters",
        ),
        ("uncertainty calib", yn(uncertaintyArgs.enabled)),
        ("ROCCO peaks", yn(matchingArgs.enabled)),
    )
    core._logEvent("config.initial", rows, logger_=logger)


def _resolveGlobalMedianCenterStatus(
    countingArgs: core.countingParams,
    controlsPresent: bool,
) -> tuple[bool, str]:
    if not bool(countingArgs.subtractGlobalMedian):
        return False, "no"
    return True, "yes"


_DEPENDENCE_MIN_CONTEXT_BP = 500
_DEPENDENCE_MAX_CONTEXT_BP = 100_000


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


def _dependenceSpanBoundsFromContextBP(
    intervalSizeBP: int,
    *,
    minContextBP: int = _DEPENDENCE_MIN_CONTEXT_BP,
    maxContextBP: int = _DEPENDENCE_MAX_CONTEXT_BP,
) -> tuple[int, int]:
    intervalSizeBP_ = max(1, int(intervalSizeBP))
    minContextBP_ = max(1, int(minContextBP))
    maxContextBP_ = max(minContextBP_, int(maxContextBP))
    minSpan = max(3, int(math.ceil(minContextBP_ / float(2 * intervalSizeBP_))))
    maxSpan = max(minSpan, int(math.ceil(maxContextBP_ / float(2 * intervalSizeBP_))))
    return int(minSpan), int(maxSpan)


def _resolveRuntimeBackgroundBlockLen(
    dependenceSpanIntervals: Optional[int],
    backgroundBlockSizeBP: int,
    intervalSizeBP: int,
    lengthScaleMultiplier: float,
) -> int:
    multiplier = float(lengthScaleMultiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            "fitParams.ECM_backgroundLengthScaleMultiplier must be positive"
        )
    if int(backgroundBlockSizeBP) > 0:
        windowBP = multiplier * max(float(backgroundBlockSizeBP), float(intervalSizeBP))
    elif dependenceSpanIntervals is not None and int(dependenceSpanIntervals) > 0:
        windowBP = multiplier * float(dependenceSpanIntervals) * float(intervalSizeBP)
    else:
        windowBP = multiplier * max(float(backgroundBlockSizeBP), float(intervalSizeBP))
    return _oddIntervalsFromBP(windowBP, intervalSizeBP, minIntervals=1)


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
    useReplicateTrends: bool,
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
    logger_: logging.Logger = logger,
) -> None:
    restrictLocalVariance = bool(
        getattr(observationArgs, "restrictLocalVarianceToSparseBed", False)
    )
    core._logAsciiBlock(
        "MUNC estimation parameters",
        (
            ("MUNC variance model", muncVarianceModel),
            (
                "MUNC AR1 variance functional",
                getattr(
                    observationArgs,
                    "muncAR1VarianceFunctional",
                    constants.OBSERVATION_DEFAULT_MUNC_AR1_VARIANCE_FUNCTIONAL,
                ),
            ),
            (
                "MUNC AR1 max beta",
                float(constants.MUNC_AR1_MAX_BETA_DEFAULT),
            ),
            (
                "MUNC AR1 pairs reg lambda",
                float(constants.MUNC_AR1_PAIRS_REG_LAMBDA_DEFAULT),
            ),
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
                "MUNC dependence span",
                _formatOptionalLogValue(dependenceSpanIntervals),
            ),
            (
                "MUNC derived context bp",
                _formatOptionalLogValue(dependenceContextBP),
            ),
            ("trend span multiplier", float(trendMultiplier)),
            ("local span multiplier", float(localMultiplier)),
            (
                "MUNC trend mode",
                "replicate-specific" if bool(useReplicateTrends) else "pooled",
            ),
            ("MUNC pooled trend pairs", int(pooledPairCount)),
            (
                "MUNC variance EB",
                "enabled" if bool(observationArgs.EB_use) else "disabled",
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
            (
                "MUNC delta-method variance",
                (
                    "disabled"
                    if bool(getattr(observationArgs, "noDMVar", False))
                    else "enabled"
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


def _progress(iterable, **kwargs):
    disable = kwargs.pop("disable", False)
    if disable or not logging_utils.progress_enabled():
        return iterable
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, disable=False, **kwargs)


class _ConsoleLogFilter(logging.Filter):
    def __init__(self, *, verbose: bool, verbose2: bool):
        super().__init__()
        self.verbose = bool(verbose)
        self.verbose2 = bool(verbose2)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if self.verbose2:
            return True
        if self.verbose and record.levelno >= logging.INFO:
            return True
        if getattr(record, _CONSOLE_VERBOSE_EVENT_ATTR, False):
            return self.verbose
        return bool(getattr(record, _CONSOLE_EVENT_ATTR, False))


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            message = f"{record.levelname}: {message}"
        if record.exc_info:
            message = message + "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            message = message + "\n" + self.formatStack(record.stack_info)
        return message


class _TqdmConsoleHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            stream = self.stream
            if stream is sys.stderr and getattr(stream, "isatty", lambda: False)():
                tqdm.write(message, file=stream, end=self.terminator)
            else:
                stream.write(message + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


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
    return Path(f"consenrichOutput_{experimentToken}_run.v{__version__}.log")


def _defaultMatchLogPath(stateBedGraphPath: str) -> Path:
    statePath = Path(stateBedGraphPath)
    return statePath.with_name(f"{statePath.stem}_consenrich_run.v{__version__}.log")


def _configureCliLogging(
    logFile: str | Path | None,
    *,
    verbose: bool,
    verbose2: bool,
    consoleStream=None,
) -> Path | None:
    packageLogger = logging.getLogger("consenrich")
    _removeCliHandlers(packageLogger)
    packageLogger.setLevel(logging.DEBUG if verbose2 else logging.INFO)
    packageLogger.propagate = False

    consoleHandler = _TqdmConsoleHandler(
        sys.stderr if consoleStream is None else consoleStream
    )
    setattr(consoleHandler, _CLI_HANDLER_ATTR, True)
    consoleHandler.setLevel(logging.DEBUG if verbose2 else logging.INFO)
    consoleHandler.addFilter(
        _ConsoleLogFilter(verbose=bool(verbose), verbose2=bool(verbose2))
    )
    consoleHandler.setFormatter(_ConsoleFormatter())
    packageLogger.addHandler(consoleHandler)

    if logFile is None:
        return None

    logPath = Path(logFile)
    try:
        if logPath.parent != Path(""):
            logPath.parent.mkdir(parents=True, exist_ok=True)
        fileHandler = logging.FileHandler(logPath, mode="w", encoding="utf-8")
        setattr(fileHandler, _CLI_HANDLER_ATTR, True)
        fileHandler.setLevel(logging.DEBUG if verbose2 else logging.INFO)
        fileHandler.addFilter(
            _ConsoleLogFilter(verbose=bool(verbose), verbose2=bool(verbose2))
        )
        fileHandler.setFormatter(logging.Formatter(_AUDIT_LOG_FORMAT))
        packageLogger.addHandler(fileHandler)
        return logPath
    except Exception as exc:
        logger.warning(
            "Failed to configure canonical log file %s: %s. Continuing with console logging.",
            logPath,
            exc,
        )
        return None


def _logCliMilestone(message: str, *args: Any) -> None:
    logger.info(message, *args, extra={_CONSOLE_EVENT_ATTR: True}, stacklevel=2)


def _logCliProgressMilestone(message: str, *args: Any) -> None:
    logger.info(
        message,
        *args,
        extra={_CONSOLE_VERBOSE_EVENT_ATTR: True},
        stacklevel=2,
    )


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
        help="Path for the canonical Consenrich audit log.",
    )
    parser.add_argument("--verbose", action="store_true", help="If set, logs config")
    parser.add_argument(
        "--verbose2",
        action="store_true",
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
    if args.verbose2:
        args.verbose = True
    logging_utils.set_progress_enabled(bool(args.verbose or args.verbose2))

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
        )
        if resolvedLogPath is not None:
            _logCliMilestone("Canonical log: %s", resolvedLogPath)
        if uncertaintyBedGraph is None:
            uncertaintyBedGraph = _inferMatchingUncertaintyBedGraph(args.matchBedGraph)
        matchStart = time.perf_counter()
        _logCliMilestone(
            "Consenrich post-hoc ROCCO start: state=%s",
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
        )
        _logCliMilestone("Config file %s does not exist.", args.config)
        _logCliMilestone(
            "See documentation: https://nolan-h-hamilton.github.io/Consenrich/"
        )
        sys.exit(1)

    resolvedLogPath = _configureCliLogging(
        args.logFile or _defaultConfigLogPath(args.config),
        verbose=bool(args.verbose),
        verbose2=bool(args.verbose2),
    )
    if resolvedLogPath is not None:
        _logCliMilestone("Canonical log: %s", resolvedLogPath)
    cliRunStart = time.perf_counter()
    config = readConfig(args.config)
    experimentName = config["experimentName"]
    diagnosticLogPaths = _diagnosticLogPaths(str(experimentName))
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
    muncVarianceModel_ = core._normalizeMuncVarianceModel(
        getattr(
            observationArgs,
            "muncVarianceModel",
            constants.OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL,
        )
    )
    muncAR1VarianceFunctional_ = core._normalizeMuncAR1VarianceFunctional(
        getattr(
            observationArgs,
            "muncAR1VarianceFunctional",
            constants.OBSERVATION_DEFAULT_MUNC_AR1_VARIANCE_FUNCTIONAL,
        )
    )
    muncAR1UseInnovationVariance_ = (
        muncAR1VarianceFunctional_ == constants.MUNC_AR1_VARIANCE_FUNCTIONAL_INNOVATION
    )
    backgroundBlockSizeBP_ = countingArgs.backgroundBlockSizeBP
    dependenceContextBP_: Optional[int] = None
    dependenceSpanIntervals_: Optional[int] = None
    waitForMatrix: bool = False
    normMethod_: Optional[str] = countingArgs.normMethod.upper()
    pad_ = observationArgs.pad if hasattr(observationArgs, "pad") else 1.0e-4
    _logCliMilestone(
        "Diagnostic logs: munc/lambda=%s tunc/kappa=%s convergence=%s delete-block=%s",
        diagnosticLogPaths.munc_lambda,
        diagnosticLogPaths.tunc_kappa,
        diagnosticLogPaths.convergence,
        diagnosticLogPaths.delete_block_calibration,
    )
    _logCliMilestone(
        "Consenrich run start: experiment=%s version=%s config=%s chromosomes=%d samples=%d",
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
            str(samArgs.defaultCountMode or "coverage"),
            str(scArgs.defaultCountMode or "coverage"),
        )
        for source in treatmentSources
    ]
    controlCountModes = [
        _getSourceCountMode(
            source,
            str(samArgs.defaultCountMode or "coverage"),
            str(scArgs.defaultCountMode or "coverage"),
        )
        for source in controlSources
    ]
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
    for chromosome in _progress(
        chromosomes,
        total=len(chromosomes),
        desc="Planning chromosomes",
        unit="chrom",
    ):
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
                file_.endswith(".bedGraph") or file_.endswith(".narrowPeak")
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
    muncResidualBackgroundCachePaths: Dict[str, str] = {}
    pooledBlockMeansParts: list[np.ndarray] = []
    pooledBlockVarsParts: list[np.ndarray] = []
    pooledBlockCovariatesParts: list[np.ndarray] = []
    pooledBlockLogVarianceNoiseParts: list[np.ndarray] = []
    pooledSampleIndexParts: list[np.ndarray] = []
    pooledChromIndexParts: list[np.ndarray] = []
    pooledBlockStartsParts: list[np.ndarray] = []
    pooledWeightsParts: list[np.ndarray] = []
    useReplicateTrends = bool(getattr(observationArgs, "useReplicateTrends", False))
    noDMVar = bool(getattr(observationArgs, "noDMVar", False))
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
        if raw.shape[0] < int(numIntervals):
            logger.warning(
                "MUNC genomic covariates: cache for %s covers %d/%d requested "
                "intervals; missing intervals will be excluded from additive "
                "covariate fitting and will use the baseline MUNC trend.",
                chromosome,
                int(raw.shape[0]),
                int(numIntervals),
            )
            padded = np.full(
                (int(numIntervals), raw.shape[1]),
                np.nan,
                dtype=np.float32,
            )
            padded[: raw.shape[0], :] = raw
            raw = padded
        elif raw.shape[0] > int(numIntervals):
            raw = raw[: int(numIntervals), :]
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

    def _getChromBlacklistMask(chromosome: str, intervals: np.ndarray) -> np.ndarray:
        if not genomeArgs.blacklistFile or len(intervals) < 2:
            return np.zeros(len(intervals), dtype=np.uint8)
        mask = core.getBedMask(chromosome, genomeArgs.blacklistFile, intervals)
        return np.asarray(mask, dtype=np.uint8)

    def _countAndTransformChromosomeMatrix(
        c_: int,
        chromPlan: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal backgroundBlockSizeBP_
        nonlocal dependenceContextBP_, dependenceSpanIntervals_, sf

        chromosome = str(chromPlan["chromosome"])
        chromosomeStart = int(chromPlan["start"])
        chromosomeEnd = int(chromPlan["end"])
        numIntervals = int(chromPlan["numIntervals"])
        intervals = np.arange(chromosomeStart, chromosomeEnd, intervalSizeBP)
        chromMat: np.ndarray = np.empty((numSamples, numIntervals), dtype=np.float32)
        countModelVarianceFloorMat = np.full(
            (numSamples, numIntervals),
            np.nan,
            dtype=np.float32,
        )

        if controlsPresent:
            for j_, (bamA, bamB) in enumerate(
                _progress(
                    zip(bamFiles, bamFilesControl),
                    total=numSamples,
                    desc=f"Counting {chromosome}",
                    unit="sample",
                )
            ):
                countStart = time.perf_counter()
                logger.info(
                    "counting.start %s sample=%d/%d treatment=%s control=%s",
                    chromosome,
                    int(j_ + 1),
                    int(numSamples),
                    bamA,
                    bamB,
                )
                pairMatrix: np.ndarray = core.readSegments(
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
                )
                # compute the observation noise floor _before_ transforming counts
                countModelVarianceFloorMat[j_, :] = _combineCountModelVarianceFloors(
                    _countModelVarianceFloorForScaledCounts(
                        pairMatrix[0, :],
                        treatScaleFactors[j_],
                        countingArgs,
                        countModelSource=_sourceUsesCountModelFloor(
                            treatmentSources[j_]
                        ),
                    ),
                    _countModelVarianceFloorForScaledCounts(
                        pairMatrix[1, :],
                        controlScaleFactors[j_],
                        countingArgs,
                        countModelSource=_sourceUsesCountModelFloor(controlSources[j_]),
                    ),
                ).astype(np.float32, copy=False)
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
            chromMat = core.readSegments(
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
            )
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

        if not controlsPresent:
            countModelScaleFactors = sf if waitForMatrix else scaleFactors
            # Treatment-only floors use the scaled count matrix just before transformation.
            countModelVarianceFloorMat = _countModelFloorMatrixForScaledCounts(
                chromMat,
                countModelScaleFactors,
                treatmentSources,
                countingArgs,
        )

        floorSummary = _summarizeCountModelVarianceFloor(countModelVarianceFloorMat)
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
                    for _ in _progress(
                        pool.imap(_transformTrack, range(numSamples)),
                        total=numSamples,
                        desc="Transforming data",
                        unit="sample",
                    ):
                        pass
            else:
                for j in _progress(
                    range(numSamples),
                    desc="Transforming data",
                    unit="sample",
                ):
                    _transformTrack(j)
            logger.info(
                "transform.done %s samples=%d elapsed=%.3fs",
                chromosome,
                int(numSamples),
                time.perf_counter() - transformStart,
            )

        centerEnabled, _ = _resolveGlobalMedianCenterStatus(
            countingArgs,
            controlsPresent=controlsPresent,
        )
        if centerEnabled:
            centerStart = time.perf_counter()
            centerStats = core.subtractGlobalMedianInPlace(chromMat)
            trackMedians = np.asarray(
                centerStats.get("track_medians", []),
                dtype=np.float64,
            )
            finiteTrackMedians = trackMedians[np.isfinite(trackMedians)]
            medianRange = (
                "NA"
                if finiteTrackMedians.size == 0
                else (
                    f"[{float(np.min(finiteTrackMedians)):.4g}, "
                    f"{float(np.max(finiteTrackMedians)):.4g}]"
                )
            )
            logger.info(
                "global median center.done %s samples=%d applied=%d "
                "medianRange=%s elapsed=%.3fs",
                chromosome,
                int(numSamples),
                int(centerStats.get("applied_tracks", 0)),
                medianRange,
                time.perf_counter() - centerStart,
            )

        return (
            intervals,
            np.ascontiguousarray(chromMat, dtype=np.float32),
            np.ascontiguousarray(countModelVarianceFloorMat, dtype=np.float32),
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

        if not hasattr(core, "_solveZeroCenteredBackground"):
            logger.warning(
                "MUNC residualization prepass: no core background helper is available; "
                "using zero g0 for %s.",
                chromosome,
            )
            return (
                np.zeros(intervalCount, dtype=np.float32),
                "zero_no_core_background_helper",
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
        backgroundArr = core._solveZeroCenteredBackground(
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
            "core._solveZeroCenteredBackground_coarse_weights",
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
    ) -> None:
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
        nonlocal dependenceContextBP_, dependenceSpanIntervals_

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

        depPoint, depLower, depUpper, depDiagnostics = (
            cconsenrich.cchooseDependenceSpan(
                chromNames,
                chromMatrices,
                intervalSizeBP,
                numBlocks=100,
                randSeed=int(constants.UNCERTAINTY_CALIBRATION_DEFAULT_SEED),
                blockMedianBP=50_000.0,
                blockSigma=1.0,
                blockMinBP=1_000,
                blockMaxBP=1_000_000,
                minContextBP=int(_DEPENDENCE_MIN_CONTEXT_BP),
                maxContextBP=int(_DEPENDENCE_MAX_CONTEXT_BP),
                priorMedianSpan=80.0,
                priorLogSd=1.0,
            )
        )
        dependenceSpanIntervals_ = int(depPoint)
        dependenceContextBP_ = int(
            2 * int(dependenceSpanIntervals_) * int(intervalSizeBP) + 1
        )

        excluded = [
            str(value) for value in depDiagnostics.get("chromosomes_excluded", [])
        ]
        excludedLabel = "none" if not excluded else ",".join(excluded)
        sampledWidthMedian = float(
            depDiagnostics.get("sampled_width_median_bp", float("nan"))
        )
        sampledWidthLabel = (
            "nan"
            if not np.isfinite(sampledWidthMedian)
            else str(int(round(sampledWidthMedian)))
        )
        logger.info(
            "chooseDependenceSpan.sampledBlocks chromosomes_used=%d "
            "chromosomes_excluded=%s blocks_requested=%d blocks_valid=%d "
            "block_lognormal_median_bp=%d block_lognormal_sigma=%.1f "
            "block_min_bp=%d block_max_bp=%d sampled_width_median_bp=%s "
            "span=%d lower=%d upper=%d context_bp=%d right_censored_blocks=%d "
            "posterior_log_sd=%.6g tau2=%.6g fallback=%s",
            int(len(depDiagnostics.get("chromosomes_used", []))),
            excludedLabel,
            int(depDiagnostics.get("blocks_requested", 100)),
            int(depDiagnostics.get("blocks_valid", 0)),
            int(depDiagnostics.get("block_lognormal_median_bp", 50_000)),
            float(depDiagnostics.get("block_lognormal_sigma", 1.0)),
            int(depDiagnostics.get("block_min_bp", 1_000)),
            int(depDiagnostics.get("block_max_bp", 1_000_000)),
            sampledWidthLabel,
            int(depPoint),
            int(depLower),
            int(depUpper),
            int(dependenceContextBP_),
            int(depDiagnostics.get("right_censored_blocks", 0)),
            float(depDiagnostics.get("posterior_log_span_sd", float("nan"))),
            float(depDiagnostics.get("tau2", float("nan"))),
            "true" if bool(depDiagnostics.get("fallback", False)) else "false",
        )

    def _collectPooledMuncBlocks(
        c_: int,
        intervals: np.ndarray,
        chromMat: np.ndarray,
        *,
        inputLabel: str = "raw",
        diagnosticRawMat: np.ndarray | None = None,
        chromCovariates: np.ndarray | None = None,
    ) -> None:
        chromosomeName = str(chromosomePlans[c_]["chromosome"])
        muncSizing = core._resolveMuncRuntimeSizing(
            intervalSizeBP=intervalSizeBP,
            dependenceSpanIntervals=dependenceSpanIntervals_,
            muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
            muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
            muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier_,
            muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier_,
        )
        blockSizeIntervals = int(muncSizing.trendBlockIntervals)
        blacklistExcludeMask = _getChromBlacklistMask(chromosomeName, intervals)
        countModelVarianceFloorMat = _loadCountModelVarianceFloor(
            chromosomeName,
            int(chromMat.shape[1]),
        )
        intervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32)
        collectedMeansParts: list[np.ndarray] = []
        rawDiagnosticMeansParts: list[np.ndarray] = []
        for j in range(numSamples):
            blockMeans, blockVars, starts, ends = cconsenrich.cmeanVarPairs(
                intervalsArr,
                np.ascontiguousarray(chromMat[j, :], dtype=np.float32),
                blockSizeIntervals,
                samplingIters_,
                42 + j,
                blacklistExcludeMask,
                useInnovationVar=muncAR1UseInnovationVariance_,
                maxBeta=constants.MUNC_AR1_MAX_BETA_DEFAULT,
                pairsRegLambda=constants.MUNC_AR1_PAIRS_REG_LAMBDA_DEFAULT,
            )
            blockMeansArr = np.asarray(blockMeans)
            blockVarsArr = np.asarray(blockVars)
            startsArr = np.asarray(starts)
            endsArr = np.asarray(ends)
            if startsArr.size != blockMeansArr.size:
                continue
            blockLengthsArr = endsArr - startsArr
            blockVarsForPooled = np.asarray(blockVarsArr, dtype=np.float64)
            hasCountNoise = (
                countModelVarianceFloorMat is not None
                and np.any(np.isfinite(countModelVarianceFloorMat[j, :]))
            )
            if hasCountNoise:
                blockNoise = core._finiteIntervalMeans(
                    countModelVarianceFloorMat[j, :],
                    startsArr,
                    endsArr,
                )
                blockVarsForPooled = core._subtractMuncCountNoise(
                    blockVarsForPooled,
                    blockNoise,
                    varianceFloor=_MUNC_NUMERIC_VARIANCE_FLOOR,
                    varianceCap=maxR_ if maxR_ is not None and maxR_ > 0.0 else None,
                    fillNaN=False,
                ).astype(np.float64, copy=False)
            if not noDMVar:
                blockBetas = cconsenrich.cblockAR1Beta(
                    np.ascontiguousarray(chromMat[j, :], dtype=np.float32),
                    np.ascontiguousarray(startsArr, dtype=np.intp),
                    np.ascontiguousarray(blockLengthsArr, dtype=np.intp),
                    maxBeta=constants.MUNC_AR1_MAX_BETA_DEFAULT,
                    pairsRegLambda=constants.MUNC_AR1_PAIRS_REG_LAMBDA_DEFAULT,
                )
                blockLogVarianceNoise = core._computeDeltaMethodAR1LogVarianceNoise(
                    blockBetas,
                    blockLengthsArr,
                    maxBeta=constants.MUNC_AR1_MAX_BETA_DEFAULT,
                    muncAR1VarianceFunctional=muncAR1VarianceFunctional_,
                )
            else:
                blockLogVarianceNoise = None
            valid = (
                np.isfinite(blockMeansArr)
                & np.isfinite(blockVarsForPooled)
                & (
                    blockVarsForPooled
                    >= (_MUNC_NUMERIC_VARIANCE_FLOOR if hasCountNoise else 1.0e-3)
                )
            )
            if not np.any(valid):
                continue
            count = int(np.count_nonzero(valid))
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
            if blockLogVarianceNoise is not None:
                pooledBlockLogVarianceNoiseParts.append(
                    np.asarray(blockLogVarianceNoise[valid], dtype=np.float64)
                )
            if chromCovariates is not None:
                blockCovariates = _blockCovariateMeans(
                    chromCovariates,
                    startsArr,
                    endsArr,
                )
                pooledBlockCovariatesParts.append(
                    np.asarray(blockCovariates[valid, :], dtype=np.float32)
                )
            pooledSampleIndexParts.append(np.full(count, int(j), dtype=np.int64))
            pooledChromIndexParts.append(np.full(count, int(c_), dtype=np.int64))
            pooledBlockStartsParts.append(np.asarray(startsArr[valid], dtype=np.int64))
            pooledWeightsParts.append(
                np.full(
                    count,
                    max(float(chromMat.shape[1]) / float(max(count, 1)), 1.0),
                    dtype=np.float64,
                )
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

    cachedIntervalsByChromosome: dict[str, np.ndarray] = {}
    for c_, chromPlan in enumerate(
        _progress(
            chromosomePlans,
            total=len(chromosomePlans),
            desc="Counting/transformation pass",
            unit="chrom",
        )
    ):
        chromosome = str(chromPlan["chromosome"])
        intervals, chromMat, countModelVarianceFloorMat = (
            _countAndTransformChromosomeMatrix(c_, chromPlan)
        )
        cachePath = os.path.join(pooledMuncCache.name, f"chrom_{c_:05d}.npy")
        np.save(cachePath, chromMat, allow_pickle=False)
        transformedMatrixCachePaths[chromosome] = cachePath
        floorCachePath = os.path.join(
            pooledMuncCache.name,
            f"chrom_{c_:05d}_count_model_floor.npy",
        )
        np.save(floorCachePath, countModelVarianceFloorMat, allow_pickle=False)
        countModelVarianceFloorCachePaths[chromosome] = floorCachePath
        cachedIntervalsByChromosome[chromosome] = np.asarray(intervals, dtype=np.uint32)

    muncSizingNeedsDependence = core._muncSizingNeedsDependence(
        muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
    )
    if backgroundBlockSizeBP_ < 0 or muncSizingNeedsDependence:
        _ensureSampledDependenceSpan(transformedMatrixCachePaths)

    for c_, chromPlan in enumerate(
        _progress(
            chromosomePlans,
            total=len(chromosomePlans),
            desc="Preparing pooled MUNC trend",
            unit="chrom",
        )
    ):
        chromosome = str(chromPlan["chromosome"])
        cachePath = transformedMatrixCachePaths[chromosome]
        chromMat = np.ascontiguousarray(np.load(cachePath), dtype=np.float32)
        np.save(cachePath, chromMat, allow_pickle=False)
        intervals = cachedIntervalsByChromosome[chromosome]
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
        muncTrendInputLabel = "residualized" if bool(fitArgs.fitBackground) else "raw"
        chromCovariates = _getChromMuncCovariates(
            chromosome,
            int(chromPlan["start"]),
            int(chromPlan["end"]),
            int(chromPlan["numIntervals"]),
        )
        _collectPooledMuncBlocks(
            c_,
            intervals,
            residMat,
            inputLabel=muncTrendInputLabel,
            diagnosticRawMat=chromMat if bool(fitArgs.fitBackground) else None,
            chromCovariates=chromCovariates,
        )

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
        pooledWeights = np.concatenate(pooledWeightsParts)
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
        pooledWeights = np.empty(0, dtype=np.float64)

    pooledMuncSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceSpanIntervals=dependenceSpanIntervals_,
        muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
        muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier_,
        muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier_,
    )
    if pooledBlockLogVarianceNoise is not None:
        finiteBlockNoise = pooledBlockLogVarianceNoise[
            np.isfinite(pooledBlockLogVarianceNoise)
            & (pooledBlockLogVarianceNoise > 0.0)
        ]
        if finiteBlockNoise.size:
            blockEffNu = 2.0 * core.itrigamma(finiteBlockNoise)
            blockEffNu = blockEffNu[np.isfinite(blockEffNu)]
            logger.info(
                "pooled MUNC block delta-method noise: pairs=%d "
                "logvar_noise_median=%.4g block_Nu_L_effective_median=%.2f "
                "ar1_variance_functional=%s ar1_max_beta=%.4g "
                "ar1_pairs_reg_lambda=%.4g",
                int(finiteBlockNoise.size),
                float(np.median(finiteBlockNoise)),
                float(np.median(blockEffNu)) if blockEffNu.size else float("nan"),
                muncAR1VarianceFunctional_,
                float(constants.MUNC_AR1_MAX_BETA_DEFAULT),
                float(constants.MUNC_AR1_PAIRS_REG_LAMBDA_DEFAULT),
            )
    pooledBlockSizeIntervals = int(pooledMuncSizing.trendBlockIntervals)
    pooledLocalWindowIntervals = int(pooledMuncSizing.localWindowIntervals)
    if observationArgs.EB_setNuL is not None and observationArgs.EB_setNuL > 3:
        pooledNuL = float(observationArgs.EB_setNuL)
    else:
        pooledNuL = float(max(4, pooledLocalWindowIntervals - 3))
    pooledNu0Cap = 100.0 * float(pooledNuL)

    varianceFloorForTrend = _MUNC_NUMERIC_VARIANCE_FLOOR
    varianceCapForTrend = maxR_ if maxR_ is not None and maxR_ > 0.0 else None
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
        useReplicateTrends=useReplicateTrends,
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
    )
    pooledMuncFit: core.PooledMuncVarianceTrend | None = None
    replicateMuncPriors: list[core.ReplicateMuncVariancePrior] | None = None
    pooledReplicateVarianceFactors = np.ones(numSamples, dtype=np.float64)
    pooledMuncNu0 = float("nan")

    if useReplicateTrends:
        replicateTrendWorkers = io_helpers._getSmallWorkerCount(
            numSamples,
            maxWorkers=min(4, max(numSamples, 1)),
        )
        replicateTrendWorkers = (
            replicateTrendWorkers
            if numSamples >= 2 and pooledBlockMeans.size >= 256
            else 1
        )
        replicateMuncPriors = core.fitReplicateMuncVariancePriors(
            pooledBlockMeans,
            pooledBlockVars,
            pooledSampleIndex,
            chromosomeIndex=pooledChromIndex,
            blockStarts=pooledBlockStarts,
            weights=pooledWeights,
            localLogVarianceNoise=pooledBlockLogVarianceNoise,
            sampleCount=numSamples,
            eps=varianceFloorForTrend,
            maxVariance=varianceCapForTrend,
            trendNumBasis=trendNumBasis_,
            trendMinObsPerBasis=trendMinObsPerBasis_,
            trendMinEdf=trendMinEdf_,
            trendMaxEdf=trendMaxEdf_,
            trendLambdaMin=trendLambdaMin_,
            trendLambdaMax=trendLambdaMax_,
            trendLambdaGridSize=trendLambdaGridSize_,
            EB_setNu0=observationArgs.EB_setNu0,
            EB_setNuL=observationArgs.EB_setNuL,
            localWindowIntervals=pooledLocalWindowIntervals,
            thinBinSize=pooledLocalWindowIntervals,
            workers=replicateTrendWorkers,
            memmapDir=pooledMuncCache.name,
            muncAR1VarianceFunctional=muncAR1VarianceFunctional_,
        )
        logger.info(
            "replicate MUNC signed trends: pairs=%d samples=%d workers=%d Nu_0=%s diagnostics=%s",
            int(pooledBlockMeans.size),
            int(numSamples),
            int(replicateTrendWorkers),
            np.array2string(
                np.asarray(
                    [prior.Nu_0 for prior in replicateMuncPriors],
                    dtype=np.float64,
                ),
                precision=2,
                floatmode="fixed",
                separator=", ",
            ),
            [dict(prior.diagnostics) for prior in replicateMuncPriors],
        )
    else:
        pooledMuncFit = core.fitPooledMuncVarianceTrend(
            pooledBlockMeans,
            pooledBlockVars,
            pooledSampleIndex,
            weights=pooledWeights,
            eps=varianceFloorForTrend,
            trendNumBasis=trendNumBasis_,
            trendMinObsPerBasis=trendMinObsPerBasis_,
            trendMinEdf=trendMinEdf_,
            trendMaxEdf=trendMaxEdf_,
            trendLambdaMin=trendLambdaMin_,
            trendLambdaMax=trendLambdaMax_,
            trendLambdaGridSize=trendLambdaGridSize_,
        )
        pooledReplicateVarianceFactors = np.asarray(
            pooledMuncFit.replicateVarianceFactors,
            dtype=np.float64,
        )
        if pooledReplicateVarianceFactors.size < numSamples:
            pooledReplicateVarianceFactors = np.pad(
                pooledReplicateVarianceFactors,
                (0, int(numSamples - pooledReplicateVarianceFactors.size)),
                constant_values=1.0,
            )
        pooledReplicateVarianceFactors = pooledReplicateVarianceFactors[:numSamples]

        if pooledBlockMeans.size:
            pooledPriorVariance = core.evalPSplineLogVarianceTrend(
                pooledMuncFit.trend,
                pooledBlockMeans,
                eps=varianceFloorForTrend,
                maxVariance=varianceCapForTrend,
            ).astype(np.float64, copy=False)
            factorByPair = pooledReplicateVarianceFactors[
                np.clip(pooledSampleIndex, 0, numSamples - 1)
            ]
            pooledPriorVariance *= factorByPair
        else:
            pooledPriorVariance = np.empty(0, dtype=np.float64)

        specifiedNu0 = core._coerceEBPriorStrength(observationArgs.EB_setNu0)
        if specifiedNu0 is not None:
            pooledMuncNu0 = specifiedNu0
            logger.info("Using fixed/specified pooled Nu_0=%.2f", pooledMuncNu0)
        else:
            pooledMuncNu0 = core.EB_computePooledPriorStrength(
                pooledBlockVars,
                pooledPriorVariance,
                pooledNuL,
                sampleIndex=pooledSampleIndex,
                chromosomeIndex=pooledChromIndex,
                blockStarts=pooledBlockStarts,
                thinBinSize=pooledLocalWindowIntervals,
                localLogVarianceNoise=pooledBlockLogVarianceNoise,
            )
        if not np.isfinite(pooledMuncNu0) or pooledMuncNu0 < 4.0:
            pooledMuncNu0 = pooledNu0Cap
        if pooledMuncNu0 > pooledNu0Cap:
            logger.info(
                "Capping pooled Nu_0=%.2f at 50*Nu_L=%.2f",
                float(pooledMuncNu0),
                float(pooledNu0Cap),
            )
            pooledMuncNu0 = pooledNu0Cap

        replicateNu0Diagnostics: list[float] = []
        for j in range(numSamples):
            repMask = pooledSampleIndex == j
            if np.count_nonzero(repMask) < 4:
                replicateNu0Diagnostics.append(float("nan"))
                continue
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
            if np.isfinite(repNu0):
                repNu0 = min(float(repNu0), pooledNu0Cap)
            replicateNu0Diagnostics.append(float(repNu0))

        logger.info(
            "pooled MUNC signed trend: pairs=%d samples=%d factors=%s Nu_0=%.2f Nu_L=%.2f perRepNu0=%s diagnostics=%s",
            int(pooledBlockMeans.size),
            int(numSamples),
            np.array2string(
                pooledReplicateVarianceFactors,
                precision=4,
                floatmode="fixed",
                separator=", ",
            ),
            float(pooledMuncNu0),
            float(pooledNuL),
            np.array2string(
                np.asarray(replicateNu0Diagnostics, dtype=np.float64),
                precision=2,
                floatmode="fixed",
                separator=", ",
            ),
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
            baselinePooledVariance = np.empty_like(pooledBlockMeans, dtype=np.float64)
            if replicateMuncPriors is not None:
                for j in range(numSamples):
                    repMask = pooledSampleIndex == int(j)
                    if not np.any(repMask):
                        continue
                    prior = replicateMuncPriors[j]
                    baselinePooledVariance[repMask] = core.evalPSplineLogVarianceTrend(
                        prior.trend,
                        pooledBlockMeans[repMask],
                        eps=varianceFloorForTrend,
                        maxVariance=varianceCapForTrend,
                    ).astype(np.float64, copy=False)
            else:
                if pooledMuncFit is None:
                    raise RuntimeError("pooled MUNC trend was not initialized")
                baselinePooledVariance = core.evalPSplineLogVarianceTrend(
                    pooledMuncFit.trend,
                    pooledBlockMeans,
                    eps=varianceFloorForTrend,
                    maxVariance=varianceCapForTrend,
                ).astype(np.float64, copy=False)
                factorByPair = pooledReplicateVarianceFactors[
                    np.clip(pooledSampleIndex, 0, numSamples - 1)
                ]
                baselinePooledVariance *= factorByPair
            baselinePooledVariance = core._clipVarianceTrack(
                baselinePooledVariance,
                floor=varianceFloorForTrend,
                cap=varianceCapForTrend,
            ).astype(np.float64, copy=False)
            additiveCovariateModel = core.fitMuncAdditiveCovariateModel(
                pooledBlockMeans,
                pooledBlockVars,
                baselinePooledVariance,
                pooledBlockCovariates,
                pooledSampleIndex,
                featureNames=muncCovariateFeatureNames,
                weights=pooledWeights,
                sampleCount=numSamples,
                eps=varianceFloorForTrend,
            )
            logger.info(
                "MUNC additive genomic covariate model: features=%s valid_pairs=%d "
                "basis=%d fallback_replicates=%d pooled_coef_sum=%.4g",
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

    stateDiagnosticsByChromosome: Dict[str, Any] = {}
    bedGraphTracks: List[Tuple[str, str]] = [("State", "state")]
    if outputArgs.writeUncertainty:
        bedGraphTracks.append(("uncertainty", "uncertainty"))
    diagnosticTrackNames = tuple(getattr(outputArgs, "diagnosticTracks", ()) or ())
    stateDiagnosticTrackNames = tuple(
        trackName for trackName in diagnosticTrackNames if trackName == "slope"
    )
    if any(trackName != "slope" for trackName in diagnosticTrackNames):
        logger.info(
            "MUNC/lambda and TUNC/kappa diagnostic tracks are written to category logs; "
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
    bedGraphChromOrder = [str(chromPlan["chromosome"]) for chromPlan in chromosomePlans]
    genomeOptimizationPathRows: List[Mapping[str, Any]] = []
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
    saveGains = bool(
        getattr(outputArgs, "saveGains", constants.OUTPUT_DEFAULT_SAVE_GAINS)
    )
    replicateGainAccumulator = (
        _newReplicateGainAccumulator(len(treatmentSources)) if saveGains else None
    )

    for c_, chromPlan in enumerate(
        _progress(
            chromosomePlans,
            total=len(chromosomePlans),
            desc="Processing chromosomes",
            unit="chrom",
        )
    ):
        chromosomeStartTime = time.perf_counter()
        chromosome = str(chromPlan["chromosome"])
        chromosomeStart = int(chromPlan["start"])
        chromosomeEnd = int(chromPlan["end"])
        numIntervals = int(chromPlan["numIntervals"])
        _logCliProgressMilestone(
            "Chromosome %d/%d start: %s intervals=%d",
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
        countModelVarianceFloorMat = _loadCountModelVarianceFloor(
            chromosome,
            int(numIntervals),
        )
        countModelFloorQ05 = _countModelVarianceFloorScalar(
            countModelVarianceFloorMat,
        )
        muncResidualBackground = _loadMuncResidualizationBackground(
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
        useSparseRestrictedLocalAR1 = bool(
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
        if useSparseRestrictedLocalAR1:
            sparseRegionMask = core.getBedMask(
                chromosome,
                genomeArgs.sparseBedFile,
                intervals,
            )
            logger.info(
                "munc matrix: restricting rolling local observation variance to "
                "sparse-bed regions (chrom=%s, model=%s, sparseIntervals=%d).",
                chromosome,
                muncVarianceModel_,
                int(np.count_nonzero(sparseRegionMask)),
            )

        def _fitMuncTrack(j: int) -> tuple[int, np.ndarray]:
            if replicateMuncPriors is not None:
                prior = replicateMuncPriors[j]
                pooledTrend = prior.trend
                replicateVarianceFactor = 1.0
                pooledNu0 = prior.Nu_0
            else:
                if pooledMuncFit is None:
                    raise RuntimeError("pooled MUNC trend was not initialized")
                pooledTrend = pooledMuncFit.trend
                replicateVarianceFactor = float(pooledReplicateVarianceFactors[j])
                pooledNu0 = float(pooledMuncNu0)
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
                muncAR1VarianceFunctional=muncAR1VarianceFunctional_,
                randomSeed=42 + j,
                EB_use=observationArgs.EB_use,
                EB_setNu0=observationArgs.EB_setNu0,
                EB_setNuL=observationArgs.EB_setNuL,
                noDMVar=noDMVar,
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
                replicateVarianceFactor=replicateVarianceFactor,
                EB_pooledNu0=pooledNu0,
                covariateTrack=chromMuncCovariates,
                additiveCovariateModel=additiveCovariateModel,
                replicateIndex=j,
                countModelVarianceFloor=countModelVarianceFloorMat[j, :],
            )
            return j, muncTrack

        # this has become a bottleneck, so gentle multiprocessing
        muncProgressDesc = (
            "Fitting replicate signed MUNC variance"
            if replicateMuncPriors is not None
            else "Fitting pooled signed MUNC variance"
        )
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
                for j, muncTrack in _progress(
                    pool.imap(_fitMuncTrack, range(numSamples)),
                    total=numSamples,
                    desc=muncProgressDesc,
                    unit="sample",
                ):
                    muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)
        else:
            for j in _progress(
                range(numSamples),
                desc=muncProgressDesc,
                unit="sample",
            ):
                _, muncTrack = _fitMuncTrack(j)
                muncMat[j, :] = np.asarray(muncTrack, dtype=np.float32)
        logger.info(
            "munc.done %s samples=%d elapsed=%.3fs",
            chromosome,
            int(numSamples),
            time.perf_counter() - muncStart,
        )

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
        muncMat = np.nan_to_num(
            muncMat.astype(np.float32, copy=False),
            nan=np.float32(_MUNC_NUMERIC_VARIANCE_FLOOR),
            posinf=np.float32(maxRForRepair_),
            neginf=np.float32(_MUNC_NUMERIC_VARIANCE_FLOOR),
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
            autoMinQ = (1.0e-2 * countModelFloorQ05) + 1.0e-6
            logger.info(
                "processParams.minQ < 0 or processParams.maxQ < 0 --> applying minimal numerically stable bounds for conditioning",
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
            "count noise floor-derived variance floor %s value=%s "
            "numericRFloor=%s maxR=%s minQ=%s maxQ=%s",
            chromosome,
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
                ("count noise floor q05", float(countModelFloorQ05)),
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
            _appendMuncLambdaDiagnostics(
                precisionFrame,
                diagnosticLogPaths.munc_lambda,
                chromosome=chromosome,
                precisionDiagnostics=precisionDiagnostics,
            )
            _appendTuncKappaDiagnostics(
                precisionFrame,
                diagnosticLogPaths.tunc_kappa,
                chromosome=chromosome,
                precisionDiagnostics=precisionDiagnostics,
                runDiagnostics=runDiagnostics,
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
            try:
                from consenrich import uncertainty as uncertainty_module
            except ImportError as exc:
                raise RuntimeError(
                    "Delete-block state uncertainty calibration requires the optional "
                    "`consenrich.uncertainty` module and `consenrich.cuncertainty` "
                    "extension. Build/install Consenrich with uncertainty support, "
                    "or set `uncertaintyCalibrationParams.enabled: false`."
                ) from exc

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
            uncertaintyTrack = np.asarray(
                calibrationResult.calibratedUncertainty,
                dtype=np.float32,
            )
            logger.info(
                "Delete-block state uncertainty calibration applied for %s: "
                "globalFactor=%.6g rowsValid=%d",
                chromosome,
                float(calibrationResult.model.get("global_factor", np.nan)),
                int(calibrationResult.model.get("rows_valid", 0)),
            )
            if segShrinkGenomeRequested:
                segShrinkDeferredUncertainty.append(
                    {
                        "chromosome": chromosome,
                        "intervals": np.asarray(intervals, dtype=np.int64).copy(),
                        "fullP": np.asarray(P00_, dtype=np.float64).copy(),
                        "model": dict(calibrationResult.model),
                        "factor": np.asarray(calibrationResult.factor, dtype=np.float64),
                        "calibrated": np.asarray(
                            calibrationResult.calibratedUncertainty,
                            dtype=np.float32,
                        ),
                        "summaryRowIndex": len(runSummaryRows),
                    }
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

        cols_ = ["Chromosome", "Start", "End"] + [
            column for column, _suffix in bedGraphTracks
        ]
        df = df[cols_].sort_values(
            by=["Start", "End"],
            kind="mergesort",
        )

        writeStart = time.perf_counter()
        tracksForChromosome = [
            (col, suffix)
            for col, suffix in bedGraphTracks
            if not (segShrinkGenomeRequested and suffix == "uncertainty")
        ]
        for col, suffix in _progress(
            tracksForChromosome,
            total=len(tracksForChromosome),
            desc=f"Writing {chromosome}",
            unit="track",
        ):
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
            )
        )
        _logCliProgressMilestone(
            "Chromosome %s done: elapsed=%.1fs outputs=%d",
            chromosome,
            chromosomeElapsed,
            int(len(bedGraphTracks)),
        )

    if segShrinkGenomeRequested:
        if not segShrinkDeferredUncertainty:
            raise ValueError("segShrink uncertainty calibration has no processed contigs")
        from consenrich import segshrink as segshrink_module

        finalizedSegShrink = segshrink_module.combine_prepared_contigs(
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
            if isinstance(summaryRowIndex, int) and 0 <= summaryRowIndex < len(runSummaryRows):
                runSummaryRows[summaryRowIndex]["delete_block_global_factor"] = (
                    _summaryNumber(item["model"].get("global_factor"))
                )
                runSummaryRows[summaryRowIndex]["delete_block_rows_valid"] = (
                    _summaryInt(item["model"].get("rows_valid"))
                )
                runSummaryRows[summaryRowIndex]["delete_block_rows_fit"] = (
                    _summaryInt(item["model"].get("rows_fit"))
                )
            if bool(getattr(uncertaintyCalibrationArgs, "writeDiagnostics", True)):
                _appendKeyValueDiagnostics(
                    diagnosticLogPaths.delete_block_calibration,
                    DELETE_BLOCK_CALIBRATION_LOG_COLUMNS,
                    recordType="model",
                    event="delete_block_calibration.segShrink.processed_genome_model",
                    chromosome=chromosome,
                    values=item["model"],
                )
        logger.info(
            "segShrink processed-genome finalization wrote uncertainty bedGraph for %d contigs: %s",
            int(len(finalizedSegShrink)),
            uncertaintyBedGraphPath,
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

    if bool(getattr(outputArgs, "writeRunSummary", True)):
        summaryRows = list(runSummaryRows)
        if summaryRows:
            summaryRows.append(
                _genomeRunSummaryRow(
                    summaryRows,
                    elapsedSeconds=time.perf_counter() - cliRunStart,
                    diagnosticLogPaths=diagnosticLogPaths,
                )
            )
        _writeRunSummary(summaryRows, _runSummaryPath(str(experimentName)))

    logger.info("Finished: output in human-readable format")

    for suffix in _progress(
        suffixes,
        total=len(suffixes),
        desc="Validating bedGraphs",
        unit="track",
    ):
        bedgraphPath = (
            f"consenrichOutput_{experimentName}_{suffix}.v{__version__}.bedGraph"
        )
        try:
            _validateBedGraphSorted(bedgraphPath, chromOrder=bedGraphChromOrder)
        except Exception as ex:
            logger.warning(
                "bedGraph %s failed genome-order validation; sorting as fallback:\n%s",
                bedgraphPath,
                ex,
            )
            try:
                _sortBedGraphInPlace(bedgraphPath, chromOrder=bedGraphChromOrder)
            except Exception as sortEx:
                logger.warning(f"Failed to sort {bedgraphPath}:\n{sortEx}")

    if outputArgs.convertToBigWig:
        convertBedGraphToBigWig(
            experimentName,
            genomeArgs.chromSizesFile,
            suffixes=suffixes,
        )

    if peakCallingEnabled:
        try:
            logger.info("running Consenrich+ROCCO for peaks...")
            stateBedGraphPath = (
                f"consenrichOutput_{experimentName}_state.v{__version__}.bedGraph"
            )
            uncertaintyBedGraphPath = (
                f"consenrichOutput_{experimentName}_uncertainty.v{__version__}.bedGraph"
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
            outName, roccoSummary = peaks.solveRocco(
                stateBedGraphPath,
                uncertaintyBedGraphFile=uncertaintyBedGraphPath,
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
                uncertaintyScoreMode=matchingArgs.uncertaintyScoreMode,
                uncertaintyScoreZ=float(matchingArgs.uncertaintyScoreZ),
                blacklistBedFile=genomeArgs.blacklistFile,
                randSeed=matchingArgs.randSeed,
                verbose=bool(args.verbose),
                stateDiagnosticsByChromosome=stateDiagnosticsByChromosome,
                returnSummary=True,
            )

            logger.info("Finished ROCCO peak calling. Written to %s", outName)
            if bool(outputArgs.cutoffReport):
                try:
                    cutoffReportDir = peaks.solveRoccoCutoffReport(
                        stateBedGraphPath,
                        uncertaintyBedGraphFile=uncertaintyBedGraphPath,
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
                        uncertaintyScoreMode=matchingArgs.uncertaintyScoreMode,
                        uncertaintyScoreZ=float(matchingArgs.uncertaintyScoreZ),
                        blacklistBedFile=genomeArgs.blacklistFile,
                        randSeed=matchingArgs.randSeed,
                        baselineNarrowPeakFile=outName,
                        baselineSummary=roccoSummary,
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

    _logCliMilestone(
        "Consenrich run done: experiment=%s elapsed=%.1fs",
        experimentName,
        time.perf_counter() - cliRunStart,
    )


if __name__ == "__main__":
    main()
