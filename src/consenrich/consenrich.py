#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect
import logging
import math
from multiprocessing.pool import ThreadPool
import pprint
import os
import tempfile
import time
from pathlib import Path
from collections.abc import Mapping
from typing import List, Optional, Tuple, Dict, Any, Union, Sequence
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
from . import misc_util
from ._version import __version__
from . import io as io_helpers
from .config import readConfig
from .io import (
    _buildPathInputSources,
    _checkSF,
    _getSourceCountMode,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

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
    "median_diag_R",
    "median_effective_diag_R",
]


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


def _writeOptimizationPathLog(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    frame = pd.DataFrame(list(rows), columns=OPTIMIZATION_PATH_COLUMNS)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", na_rep="NA")
    logger.info("optimizationPath.output wrote %s rows=%d", path, int(len(frame)))


def _safeOutputToken(value: Any, *, fallback: str) -> str:
    token = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(value).strip()
    ).strip("._-")
    return token or str(fallback)


def _optimizationPathPrefix(experimentName: str, chromosome: str) -> str:
    experimentToken = _safeOutputToken(experimentName, fallback="experiment")
    chromosomeToken = _safeOutputToken(chromosome, fallback="contig")
    return (
        f"consenrichOutput_{experimentToken}_{chromosomeToken}"
        f"_optimizationPath.v{__version__}"
    )


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

    frame = pd.DataFrame(list(rows), columns=OPTIMIZATION_PATH_COLUMNS)
    if frame.empty:
        logger.warning(
            "optimizationPath.plot skipped because no trace rows were recorded."
        )
        return False
    frame["objective_value"] = pd.to_numeric(
        frame["objective_value"],
        errors="coerce",
    )
    frame["record_order"] = pd.to_numeric(frame["record_order"], errors="coerce")
    frame["outer_pass"] = pd.to_numeric(frame["outer_pass"], errors="coerce")
    frame["inner_iter"] = pd.to_numeric(frame["inner_iter"], errors="coerce")
    frame["objective_per_cell"] = pd.to_numeric(
        frame["objective_per_cell"],
        errors="coerce",
    )
    frame["change"] = pd.to_numeric(frame["change"], errors="coerce")
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    frame["background_shift"] = pd.to_numeric(
        frame["background_shift"],
        errors="coerce",
    )
    frame["background_shift_threshold"] = pd.to_numeric(
        frame["background_shift_threshold"],
        errors="coerce",
    )
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
    frame = frame.dropna(subset=["record_order", "objective_value"])
    if frame.empty:
        logger.warning(
            "optimizationPath.plot skipped because all objective values were NA."
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


def _writePrecisionDiagnostics(
    *,
    chromosome: str,
    intervals: np.ndarray,
    intervalSizeBP: int,
    matrixMunc: np.ndarray,
    pad: float,
    precisionDiagnostics: Mapping[str, Any],
    path: str,
) -> bool:
    lambdaExp = precisionDiagnostics.get("lambdaExp")
    processPrecExp = precisionDiagnostics.get("processPrecExp")
    matrixQ0 = precisionDiagnostics.get("matrixQ0")
    n = int(len(intervals))
    if n <= 0:
        logger.warning(
            "precisionDiagnostics.output skipped %s because no intervals were available.",
            path,
        )
        return False

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
    q00 = float(q[0, 0]) / kappaArr
    q11 = float(q[1, 1]) / kappaArr

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
            "median_diag_R": medianDiagR,
            "median_effective_diag_R": medianEffectiveDiagR,
        }
    )
    frame.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        float_format="%.7g",
        lineterminator="\n",
        compression="gzip" if str(path).endswith(".gz") else None,
    )
    logger.info(
        "precisionDiagnostics.output wrote %s rows=%d lambdaMedian=%.6g kappaMedian=%.6g",
        path,
        int(len(frame)),
        float(np.median(lambdaArr)),
        float(np.median(kappaArr)),
    )
    return True


def _truncateMiddle(text: Any, width: int) -> str:
    text_ = str(text)
    width_ = max(0, int(width))
    if len(text_) <= width_:
        return text_
    if width_ <= 3:
        return text_[:width_]
    head = (width_ - 3) // 2
    tail = width_ - 3 - head
    return f"{text_[:head]}...{text_[-tail:]}"


def _formatReplicateGainFrame(
    chromosome: str,
    treatmentSources: Sequence[core.inputSource],
    gainMeans: Sequence[Any],
    gainMedians: Sequence[Any],
    gainSds: Sequence[Any] = (),
    gainIqrs: Sequence[Any] = (),
    *,
    controlSources: Sequence[core.inputSource] | None = None,
    indentLevel: int = 0,
) -> str:
    columns = (
        ("rep", 5),
        ("id", 18),
        ("file", 46),
        ("mean", 12),
        ("median", 12),
        ("sd", 12),
        ("IQR", 12),
    )
    border = "+" + "+".join("-" * (width + 2) for _name, width in columns) + "+"
    titleWidth = len(border) - 4
    title = _truncateMiddle(f"FINAL FORWARD-PASS GAINS [{chromosome}]", titleWidth)
    header = "| " + " | ".join(f"{name:<{width}}" for name, width in columns) + " |"
    lines = [border, f"| {title:<{titleWidth}} |", border, header, border]
    controlSources_ = list(controlSources or [])
    rowCount = max(len(gainMeans), len(gainMedians), len(gainSds), len(gainIqrs))
    for i in range(rowCount):
        source = treatmentSources[i] if i < len(treatmentSources) else None
        if source is None:
            sourceId = f"replicate_{i + 1}"
            fileLabel = "unknown"
        else:
            sourceId = str(
                source.sampleName
                or os.path.basename(source.path)
                or f"replicate_{i + 1}"
            )
            fileLabel = str(source.path)
        if i < len(controlSources_):
            fileLabel = f"{fileLabel} | control={controlSources_[i].path}"
        values = (
            str(i + 1),
            _truncateMiddle(sourceId, columns[1][1]),
            _truncateMiddle(fileLabel, columns[2][1]),
            _truncateMiddle(
                _fmtDiagnosticFloat(gainMeans[i] if i < len(gainMeans) else None),
                columns[3][1],
            ),
            _truncateMiddle(
                _fmtDiagnosticFloat(gainMedians[i] if i < len(gainMedians) else None),
                columns[4][1],
            ),
            _truncateMiddle(
                _fmtDiagnosticFloat(gainSds[i] if i < len(gainSds) else None),
                columns[5][1],
            ),
            _truncateMiddle(
                _fmtDiagnosticFloat(gainIqrs[i] if i < len(gainIqrs) else None),
                columns[6][1],
            ),
        )
        lines.append(
            "| "
            + " | ".join(
                f"{value:<{width}}" for value, (_name, width) in zip(values, columns)
            )
            + " |"
        )
    lines.append(border)
    indent = " " * (max(0, int(indentLevel)) * getattr(core, "_LOG_INDENT_WIDTH", 6))
    if indent:
        return "\n".join(f"{indent}{line}" for line in lines)
    return "\n".join(lines)


def _getProcessNoiseMAPRoughnessPenalty(processArgs: Any) -> float:
    value = getattr(processArgs, "processNoiseMapRoughnessPenalty", None)
    if value is None:
        value = getattr(processArgs, "processNoiseMAPRoughnessPenalty", None)
    if value is None:
        value = getattr(processArgs, "regularizationStrength")
    return float(value)


def _getProcessNoiseWarmupOuterPasses(processArgs: Any) -> int:
    return max(
        1,
        int(
            getattr(
                processArgs,
                "processNoiseWarmupOuterPasses",
                constants.PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES,
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
    if (
        not _coreRunConsenrichSupports("processNoiseWarmupOuterPasses")
        and hasattr(core, "PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES")
    ):
        core.PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES = warmupOuterPasses
    return warmupOuterPasses


def _processNoiseRunKwargs(processArgs: Any) -> Dict[str, Any]:
    warmupOuterPasses = _getProcessNoiseWarmupOuterPasses(processArgs)
    kwargs: Dict[str, Any] = {
        "stateModel": processArgs.stateModel,
        "regularizationStrength": processArgs.regularizationStrength,
        "regularizationRatio": processArgs.regularizationRatio,
        "processNoiseWarmupECMIters": processArgs.processNoiseWarmupECMIters,
        "processPrecisionMultiplierMin": processArgs.precisionMultiplierMin,
        "processPrecisionMultiplierMax": processArgs.precisionMultiplierMax,
    }
    if _coreRunConsenrichSupports("processNoiseMapRoughnessPenalty"):
        kwargs["processNoiseMapRoughnessPenalty"] = (
            _getProcessNoiseMAPRoughnessPenalty(processArgs)
        )
    if _coreRunConsenrichSupports("processNoiseWarmupOuterPasses"):
        kwargs["processNoiseWarmupOuterPasses"] = warmupOuterPasses
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
    replicateDetrendEnabled, replicateDetrendLabel = _resolveReplicateDetrendStatus(
        countingArgs,
        controlsPresent=controlsPresent,
    )

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
        ("replicate detrend", replicateDetrendLabel),
        ("MUNC variance model", observationArgs.muncVarianceModel),
        ("MUNC variance EB", yn(observationArgs.EB_use)),
        ("MUNC sampling iters", int(observationArgs.samplingIters)),
        ("ECM max iters", int(fitArgs.ECM_fixedBackgroundIters)),
        ("outer passes", int(fitArgs.ECM_outerIters)),
        ("background model", yn(fitArgs.fitBackground)),
        ("nonnegative background", yn(fitArgs.useNonnegativeBackground)),
        ("state model", processArgs.stateModel),
        (
            "MAP roughness penalty",
            _getProcessNoiseMAPRoughnessPenalty(processArgs),
        ),
        ("trend/level ratio", float(processArgs.regularizationRatio)),
        (
            "process noise warmup",
            f"{_getProcessNoiseWarmupOuterPasses(processArgs)} outer passes x "
            f"{int(processArgs.processNoiseWarmupECMIters)} ECM iters",
        ),
        ("uncertainty calib", yn(uncertaintyArgs.enabled)),
        ("ROCCO peaks", yn(matchingArgs.enabled)),
    )
    logger.info(
        "\n%s\n",
        core._formatAsciiLogBlock("initial configuration", rows),
        stacklevel=2,
    )
    if bool(countingArgs.replicateMedianDetrend) and not replicateDetrendEnabled:
        logger.info(
            "replicate quantile detrend disabled because control inputs are present; "
            "treatment/control tracks are already log-ratios."
        )
    if bool(countingArgs.subtractGlobalMedian) and controlsPresent:
        logger.info(
            "global median center disabled because control inputs are present; "
            "treatment/control tracks are already log-ratios."
        )


def _resolveGlobalMedianCenterStatus(
    countingArgs: core.countingParams,
    controlsPresent: bool,
) -> tuple[bool, str]:
    if not bool(countingArgs.subtractGlobalMedian):
        return False, "no"
    if bool(controlsPresent):
        return False, "no (control log-ratio)"
    return True, "yes"


def _resolveReplicateDetrendStatus(
    countingArgs: core.countingParams,
    controlsPresent: bool,
) -> tuple[bool, str]:
    if not bool(countingArgs.replicateMedianDetrend):
        return False, "no"
    if bool(controlsPresent):
        return False, "no (control log-ratio)"
    return (
        True,
        "quantile="
        f"{float(countingArgs.gentleDetrendQuantile):.3g} x"
        f"{float(countingArgs.replicateMedianDetrendWindowMultiplier):.3g}",
    )


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


def _resolveRuntimeReplicateDetrendWindow(
    dependenceSpanIntervals: Optional[int],
    backgroundBlockSizeBP: int,
    intervalSizeBP: int,
    lengthScaleMultiplier: float,
    windowMultiplier: float,
) -> int:
    multiplier = float(lengthScaleMultiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            "fitParams.ECM_backgroundLengthScaleMultiplier must be positive"
        )
    windowMultiplier_ = float(windowMultiplier)
    if not np.isfinite(windowMultiplier_) or windowMultiplier_ <= 0.0:
        raise ValueError(
            "countingParams.replicateMedianDetrendWindowMultiplier must be positive"
        )
    if dependenceSpanIntervals is not None and int(dependenceSpanIntervals) > 0:
        windowBP = (
            windowMultiplier_
            * multiplier
            * float(dependenceSpanIntervals)
            * float(intervalSizeBP)
        )
    else:
        windowBP = (
            windowMultiplier_
            * multiplier
            * max(float(backgroundBlockSizeBP), float(intervalSizeBP))
        )
    return _oddIntervalsFromBP(windowBP, intervalSizeBP, minIntervals=3)


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
    samplingBlockSizeBP: int | None,
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
        getattr(
            observationArgs,
            "restrictLocalVarianceToSparseBed",
            getattr(observationArgs, "restrictLocalAR1ToSparseBed", False),
        )
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
                "MUNC dependence span",
                _formatOptionalLogValue(dependenceSpanIntervals),
            ),
            (
                "MUNC derived context bp",
                _formatOptionalLogValue(dependenceContextBP),
            ),
            ("trend span multiplier", float(trendMultiplier)),
            ("local span multiplier", float(localMultiplier)),
            ("legacy sampling block bp", _formatOptionalLogValue(samplingBlockSizeBP)),
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
    disable = kwargs.pop("disable", not sys.stderr.isatty())
    if disable:
        return iterable
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, disable=False, **kwargs)


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
        "--match-tau0",
        type=float,
        default=constants.MATCHING_DEFAULT_TAU0,
        dest="matchTau0",
        help="Shrinkage-score pseudovariance parameter; direct ROCCO scoring uses the fitted state values.",
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
        help="Optional fraction of each eligible first-pass peak region available to nested ROCCO refinement.",
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
        "--match-seed",
        type=int,
        default=constants.MATCHING_DEFAULT_RAND_SEED,
        dest="matchRandSeed",
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
        if uncertaintyBedGraph is None:
            uncertaintyBedGraph = _inferMatchingUncertaintyBedGraph(args.matchBedGraph)
        logger.info(
            "Running post hoc ROCCO peak caller using state bedGraph %s...",
            args.matchBedGraph,
        )
        outName = peaks.solveRocco(
            args.matchBedGraph,
            uncertaintyBedGraphFile=uncertaintyBedGraph,
            tau0=args.matchTau0,
            numBootstrap=args.matchNumBootstrap,
            thresholdZ=args.matchThresholdZ,
            nestedRoccoIters=args.matchNestedRoccoIters,
            nestedRoccoBudgetScale=args.matchNestedRoccoBudgetScale,
            exportFilterUncertaintyMultiplier=(
                args.matchExportFilterUncertaintyMultiplier
            ),
            blacklistBedFile=args.matchBlacklistBed,
            randSeed=args.matchRandSeed,
            verbose=bool(args.verbose or args.verbose2),
        )
        logger.info("Finished post hoc ROCCO peak calling. Written to %s", outName)
        sys.exit(0)

    if not args.config:
        logger.info(
            "No config file provided, run with `--config <path_to_config.yaml>`"
        )
        logger.info("See documentation: https://nolan-h-hamilton.github.io/Consenrich/")
        sys.exit(1)

    if not os.path.exists(args.config):
        logger.info(f"Config file {args.config} does not exist.")
        logger.info("See documentation: https://nolan-h-hamilton.github.io/Consenrich/")
        sys.exit(1)

    config = readConfig(args.config)
    experimentName = config["experimentName"]
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
    minR_ = observationArgs.minR
    maxR_ = observationArgs.maxR
    minQ_ = processArgs.minQ
    maxQ_ = processArgs.maxQ
    samplingBlockSizeBP_ = observationArgs.samplingBlockSizeBP
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
    muncVarianceModelCode_ = core._muncVarianceModelCode(muncVarianceModel_)
    backgroundBlockSizeBP_ = countingArgs.backgroundBlockSizeBP
    dependenceContextBP_: Optional[int] = None
    dependenceSpanIntervals_: Optional[int] = None
    waitForMatrix: bool = False
    normMethod_: Optional[str] = countingArgs.normMethod.upper()
    pad_ = observationArgs.pad if hasattr(observationArgs, "pad") else 1.0e-4
    if args.verbose2:
        args.verbose = True

    if args.verbose:
        try:
            _logInitialConfigurationSummary(config)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "raw parsed config:\n%s",
                    pprint.pformat(
                        config,
                        indent=2,
                        width=120,
                        sort_dicts=True,
                        compact=False,
                    ),
                )
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
    treatmentAllowLists, treatmentSelectedCellCounts, treatmentNormTempPaths = (
        _prepareFragmentsNormalizationMetadata(treatmentSources)
    )
    controlAllowLists, controlSelectedCellCounts, controlNormTempPaths = (
        _prepareFragmentsNormalizationMetadata(controlSources)
    )

    peakCallingEnabled = checkMatchingEnabled(matchingArgs)
    if args.verbose:
        logger.info(f"peakCallingEnabled: {peakCallingEnabled}")
    scaleFactors = countingArgs.scaleFactors
    scaleFactorsControl = countingArgs.scaleFactorsControl
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
        configuredExtendBP: int,
        characteristicFragmentLength: int,
    ) -> int:
        if str(source.sourceKind).upper() not in core.ALIGNMENT_SOURCE_KINDS:
            return 0
        if sourceBamInputMode == "fragments":
            return 0
        if int(configuredExtendBP) > 0:
            return int(configuredExtendBP)
        if inferFragmentLengthRequested or (
            autoInferFragmentLength and sourceBamInputMode in ("reads", "read1")
        ):
            return int(characteristicFragmentLength)
        return 0

    countExtendFrom5pBPTreatment = [
        _resolveCountExtendFrom5pBP(
            source,
            sourceBamInputMode,
            configuredExtendBP,
            characteristicFragmentLength,
        )
        for source, sourceBamInputMode, configuredExtendBP, characteristicFragmentLength in zip(
            treatmentSources,
            treatmentBamInputModes,
            configuredExtendFrom5pBPTreatment,
            characteristicFragmentLengthsTreatment,
        )
    ]
    if controlsPresent:
        countExtendFrom5pBPControl = [
            _resolveCountExtendFrom5pBP(
                source,
                sourceBamInputMode,
                configuredExtendBP,
                characteristicFragmentLength,
            )
            for source, sourceBamInputMode, configuredExtendBP, characteristicFragmentLength in zip(
                controlSources,
                controlBamInputModes,
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
                        treatmentCountModes,
                        controlCountModes,
                        treatmentSelectedCellCounts,
                        controlSelectedCellCounts,
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
                    source, barcodeAllowListPath, countMode, groupCellCount = task
                    return detrorm.getScaleFactorPerMillion(
                        source.path,
                        excludeForNorm,
                        intervalSizeBP,
                        sourceKind=str(source.sourceKind).upper(),
                        barcodeAllowListFile=barcodeAllowListPath,
                        countMode=countMode,
                        oneReadPerBin=samArgs.oneReadPerBin,
                        groupCellCount=groupCellCount,
                        fragmentsGroupNorm=scArgs.fragmentsGroupNorm,
                    )

                scaleFactors = io_helpers._threadMap(
                    zip(
                        treatmentSources,
                        treatmentAllowLists,
                        treatmentCountModes,
                        treatmentSelectedCellCounts,
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
                    )

                scaleFactors = io_helpers._threadMap(
                    zip(
                        treatmentSources,
                        effectiveGenomeSizes,
                        characteristicFragmentLengthsTreatment,
                        treatmentAllowLists,
                        treatmentCountModes,
                        treatmentSelectedCellCounts,
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

    # negative --> data-based bounds resolved after MUNC construction
    if observationArgs.minR < 0.0:
        minR_ = 0.0
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
    muncResidualBackgroundCachePaths: Dict[str, str] = {}
    pooledBlockMeansParts: list[np.ndarray] = []
    pooledBlockVarsParts: list[np.ndarray] = []
    pooledBlockLogVarianceNoiseParts: list[np.ndarray] = []
    pooledSampleIndexParts: list[np.ndarray] = []
    pooledChromIndexParts: list[np.ndarray] = []
    pooledBlockStartsParts: list[np.ndarray] = []
    pooledWeightsParts: list[np.ndarray] = []
    useReplicateTrends = bool(getattr(observationArgs, "useReplicateTrends", False))
    noDMVar = bool(getattr(observationArgs, "noDMVar", False))

    def _getChromBlacklistMask(chromosome: str, intervals: np.ndarray) -> np.ndarray:
        if not genomeArgs.blacklistFile or len(intervals) < 2:
            return np.zeros(len(intervals), dtype=np.uint8)
        mask = core.getBedMask(chromosome, genomeArgs.blacklistFile, intervals)
        return np.asarray(mask, dtype=np.uint8)

    def _countAndTransformChromosomeMatrix(
        c_: int,
        chromPlan: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal backgroundBlockSizeBP_
        nonlocal dependenceContextBP_, dependenceSpanIntervals_, sf

        chromosome = str(chromPlan["chromosome"])
        chromosomeStart = int(chromPlan["start"])
        chromosomeEnd = int(chromPlan["end"])
        numIntervals = int(chromPlan["numIntervals"])
        intervals = np.arange(chromosomeStart, chromosomeEnd, intervalSizeBP)
        chromMat: np.ndarray = np.empty((numSamples, numIntervals), dtype=np.float32)

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
                cconsenrich.cTransformWithInputInto(
                    pairMatrix[0, :],
                    pairMatrix[1, :],
                    chromMat[j_, :],
                    logOffset=countingArgs.logOffset,
                    logMult=countingArgs.logMult,
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

        def _transformTrack(j: int) -> int:
            cconsenrich.cTransformInPlace(
                chromMat[j, :],
                verbose=args.verbose2,
                logOffset=countingArgs.logOffset,
                logMult=countingArgs.logMult,
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
        elif bool(countingArgs.subtractGlobalMedian) and controlsPresent:
            logger.info(
                "global median center.skip %s samples=%d reason=control-log-ratio",
                chromosome,
                int(numSamples),
            )

        return intervals, np.ascontiguousarray(chromMat, dtype=np.float32)

    def _applyReplicateMedianDetrend(
        chromosome: str,
        chromMat: np.ndarray,
    ) -> np.ndarray:
        replicateDetrendEnabled, _ = _resolveReplicateDetrendStatus(
            countingArgs,
            controlsPresent=controlsPresent,
        )
        if not replicateDetrendEnabled:
            if bool(countingArgs.replicateMedianDetrend) and controlsPresent:
                logger.info(
                    "replicate quantile detrend.skip %s samples=%d reason=control-log-ratio",
                    chromosome,
                    int(numSamples),
                )
            return np.ascontiguousarray(chromMat, dtype=np.float32)

        detrendWindowIntervals = _resolveRuntimeReplicateDetrendWindow(
            dependenceSpanIntervals_,
            int(backgroundBlockSizeBP_),
            intervalSizeBP,
            fitArgs.ECM_backgroundLengthScaleMultiplier,
            countingArgs.replicateMedianDetrendWindowMultiplier,
        )

        def _detrendTrack(j: int) -> dict[str, Any]:
            stats_ = core.quantileFilterDetrendInPlace(
                chromMat[j, :],
                detrendWindowIntervals,
                countingArgs.gentleDetrendQuantile,
            )
            stats_["sample_index"] = int(j)
            return stats_

        detrendStart = time.perf_counter()
        detrendWorkers = io_helpers._getSmallWorkerCount(
            numSamples,
            maxWorkers=4,
        )
        useParallelDetrend = (
            numSamples >= 4 and chromMat.shape[1] >= 5000 and detrendWorkers > 1
        )
        detrendStats: list[dict[str, Any]] = []
        if useParallelDetrend:
            logger.info(
                "replicate quantile detrend: using ThreadPool with %d workers "
                "(numSamples=%d, numIntervals=%d, quantile=%.3g, window=%d).",
                int(detrendWorkers),
                int(numSamples),
                int(chromMat.shape[1]),
                float(countingArgs.gentleDetrendQuantile),
                int(detrendWindowIntervals),
            )
            with ThreadPool(processes=int(detrendWorkers)) as pool:
                for stats_ in _progress(
                    pool.imap(_detrendTrack, range(numSamples)),
                    total=numSamples,
                    desc="Quantile-detrending replicates",
                    unit="sample",
                ):
                    detrendStats.append(stats_)
        else:
            for j in _progress(
                range(numSamples),
                desc="Quantile-detrending replicates",
                unit="sample",
            ):
                detrendStats.append(_detrendTrack(j))

        appliedStats = [s for s in detrendStats if bool(s.get("applied", False))]
        trendMedians = np.asarray(
            [float(s.get("trend_median", 0.0)) for s in appliedStats],
            dtype=np.float64,
        )
        trendMads = np.asarray(
            [float(s.get("trend_mad", 0.0)) for s in appliedStats],
            dtype=np.float64,
        )
        medianRange = (
            "NA"
            if trendMedians.size == 0
            else (
                f"[{float(np.min(trendMedians)):.4g}, "
                f"{float(np.max(trendMedians)):.4g}]"
            )
        )
        madMedian = float("nan") if trendMads.size == 0 else float(np.median(trendMads))
        logger.info(
            "replicate quantile detrend.done %s samples=%d applied=%d "
            "quantile=%.3g window=%d trendMedianRange=%s "
            "trendMADMedian=%.4g elapsed=%.3fs",
            chromosome,
            int(numSamples),
            int(len(appliedStats)),
            float(countingArgs.gentleDetrendQuantile),
            int(detrendWindowIntervals),
            medianRange,
            madMedian,
            time.perf_counter() - detrendStart,
        )
        return np.ascontiguousarray(chromMat, dtype=np.float32)

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

    def _coarseMuncResidualizationReplicateBias(chromMat: np.ndarray) -> np.ndarray:
        arr = np.asarray(chromMat, dtype=np.float32)
        sampleMedians = np.zeros(int(arr.shape[0]), dtype=np.float64)
        for j in range(int(arr.shape[0])):
            row = np.asarray(arr[j, np.isfinite(arr[j, :])], dtype=np.float64)
            sampleMedians[j] = float(np.median(row)) if row.size else 0.0
        finite = np.isfinite(sampleMedians)
        if np.any(finite):
            sampleMedians[~finite] = float(np.mean(sampleMedians[finite]))
        else:
            sampleMedians[:] = 0.0
        sampleMedians -= float(np.mean(sampleMedians, dtype=np.float64))
        return np.ascontiguousarray(sampleMedians.astype(np.float32), dtype=np.float32)

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
                initialReplicateBias=_coarseMuncResidualizationReplicateBias(arr),
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
                    dict(helperDiagnostics).get("source", "estimateProvisionalBackground")
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
        replicateBias = _coarseMuncResidualizationReplicateBias(arr)
        residualMatrix = np.where(
            finiteMask,
            arr - replicateBias[:, None],
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
                float(np.mean(np.abs(backgroundArr[np.isfinite(backgroundArr)]) <= 1.0e-3))
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
        dependenceContextBP_ = int(2 * int(dependenceSpanIntervals_) * int(intervalSizeBP) + 1)

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
    ) -> None:
        muncSizing = core._resolveMuncRuntimeSizing(
            intervalSizeBP=intervalSizeBP,
            dependenceSpanIntervals=dependenceSpanIntervals_,
            samplingBlockSizeBP=samplingBlockSizeBP_,
            muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
            muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
            muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier_,
            muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier_,
        )
        blockSizeIntervals = int(muncSizing.trendBlockIntervals)
        blacklistExcludeMask = _getChromBlacklistMask(
            str(chromosomePlans[c_]["chromosome"]), intervals
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
                useInnovationVar=False,
                modelCode=int(muncVarianceModelCode_),
            )
            blockMeansArr = np.asarray(blockMeans)
            blockVarsArr = np.asarray(blockVars)
            startsArr = np.asarray(starts)
            endsArr = np.asarray(ends)
            if startsArr.size != blockMeansArr.size:
                continue
            blockLengthsArr = endsArr - startsArr
            if (
                not noDMVar
                and int(muncVarianceModelCode_) == int(core.MUNC_VARIANCE_MODEL_CODE_AR1)
            ):
                blockBetas = cconsenrich.cblockAR1Beta(
                    np.ascontiguousarray(chromMat[j, :], dtype=np.float32),
                    np.ascontiguousarray(startsArr, dtype=np.intp),
                    np.ascontiguousarray(blockLengthsArr, dtype=np.intp),
                )
                blockLogVarianceNoise = core._computeDeltaMethodAR1LogVarianceNoise(
                    blockBetas,
                    blockLengthsArr,
                )
            else:
                blockLogVarianceNoise = None
            valid = (
                np.isfinite(blockMeansArr)
                & np.isfinite(blockVarsArr)
                & (blockVarsArr >= 1.0e-3)
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
                np.asarray(blockVarsArr[valid], dtype=np.float64)
            )
            if blockLogVarianceNoise is not None:
                pooledBlockLogVarianceNoiseParts.append(
                    np.asarray(blockLogVarianceNoise[valid], dtype=np.float64)
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
                str(chromosomePlans[c_]["chromosome"]),
                "raw",
                np.concatenate(rawDiagnosticMeansParts),
            )
        if collectedMeansParts:
            _logMuncTrendInputSummary(
                str(chromosomePlans[c_]["chromosome"]),
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
        intervals, chromMat = _countAndTransformChromosomeMatrix(c_, chromPlan)
        cachePath = os.path.join(pooledMuncCache.name, f"chrom_{c_:05d}.npy")
        np.save(cachePath, chromMat, allow_pickle=False)
        transformedMatrixCachePaths[chromosome] = cachePath
        cachedIntervalsByChromosome[chromosome] = np.asarray(intervals, dtype=np.uint32)

    muncSizingNeedsDependence = core._muncSizingNeedsDependence(
        samplingBlockSizeBP=samplingBlockSizeBP_,
        muncTrendBlockSizeBP=muncTrendBlockSizeBP_,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP_,
    )
    replicateDetrendEnabled, _ = _resolveReplicateDetrendStatus(
        countingArgs,
        controlsPresent=controlsPresent,
    )
    if (
        backgroundBlockSizeBP_ < 0
        or muncSizingNeedsDependence
        or replicateDetrendEnabled
    ):
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
        chromMat = _applyReplicateMedianDetrend(chromosome, chromMat)
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
        _collectPooledMuncBlocks(
            c_,
            intervals,
            residMat,
            inputLabel=muncTrendInputLabel,
            diagnosticRawMat=chromMat if bool(fitArgs.fitBackground) else None,
        )

    if pooledBlockMeansParts:
        pooledBlockMeans = np.concatenate(pooledBlockMeansParts)
        pooledBlockVars = np.concatenate(pooledBlockVarsParts)
        if pooledBlockLogVarianceNoiseParts:
            pooledBlockLogVarianceNoise = np.concatenate(pooledBlockLogVarianceNoiseParts)
        else:
            pooledBlockLogVarianceNoise = None
        pooledSampleIndex = np.concatenate(pooledSampleIndexParts)
        pooledChromIndex = np.concatenate(pooledChromIndexParts)
        pooledBlockStarts = np.concatenate(pooledBlockStartsParts)
        pooledWeights = np.concatenate(pooledWeightsParts)
    else:
        pooledBlockMeans = np.empty(0, dtype=np.float64)
        pooledBlockVars = np.empty(0, dtype=np.float64)
        pooledBlockLogVarianceNoise = None
        pooledSampleIndex = np.empty(0, dtype=np.int64)
        pooledChromIndex = np.empty(0, dtype=np.int64)
        pooledBlockStarts = np.empty(0, dtype=np.int64)
        pooledWeights = np.empty(0, dtype=np.float64)

    pooledMuncSizing = core._resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceSpanIntervals=dependenceSpanIntervals_,
        samplingBlockSizeBP=samplingBlockSizeBP_,
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
                "pooled MUNC block delta-method noise: pairs=%d logvar_noise_median=%.4g block_Nu_L_effective_median=%.2f",
                int(finiteBlockNoise.size),
                float(np.median(finiteBlockNoise)),
                float(np.median(blockEffNu)) if blockEffNu.size else float("nan"),
            )
    pooledBlockSizeIntervals = int(pooledMuncSizing.trendBlockIntervals)
    pooledLocalWindowIntervals = int(pooledMuncSizing.localWindowIntervals)
    if observationArgs.EB_setNuL is not None and observationArgs.EB_setNuL > 3:
        pooledNuL = float(observationArgs.EB_setNuL)
    else:
        pooledNuL = float(max(4, pooledLocalWindowIntervals - 3))
    pooledNu0Cap = 100.0 * float(pooledNuL)

    varianceFloorForTrend = minR_ if minR_ is not None and minR_ > 0.0 else 1.0e-4
    varianceCapForTrend = maxR_ if maxR_ is not None and maxR_ > 0.0 else None
    _logMuncEstimationParameters(
        chromosomeCount=len(chromosomePlans),
        sampleCount=numSamples,
        intervalSizeBP=intervalSizeBP,
        sizing=pooledMuncSizing,
        muncVarianceModel=muncVarianceModel_,
        samplingIters=samplingIters_,
        samplingBlockSizeBP=samplingBlockSizeBP_,
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

    stateDiagnosticsByChromosome: Dict[str, Any] = {}
    bedGraphTracks: List[Tuple[str, str]] = [("State", "state")]
    if outputArgs.writeUncertainty:
        bedGraphTracks.append(("uncertainty", "uncertainty"))
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
        minR_ = observationArgs.minR
        maxR_ = observationArgs.maxR
        minQ_ = processArgs.minQ
        maxQ_ = processArgs.maxQ
        # Negative bounds are data-based and must be resolved independently
        # for each chromosome; do not let an auto floor from a previous
        # chromosome seed the MUNC fit for the next chromosome.
        if observationArgs.minR < 0.0:
            minR_ = 0.0
        if observationArgs.maxR < 0.0:
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
            getattr(
                observationArgs,
                "restrictLocalVarianceToSparseBed",
                getattr(observationArgs, "restrictLocalAR1ToSparseBed", False),
            )
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
                samplingBlockSizeBP=samplingBlockSizeBP_,
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
                    getattr(
                        observationArgs,
                        "restrictLocalVarianceToSparseBed",
                        getattr(observationArgs, "restrictLocalAR1ToSparseBed", False),
                    )
                ),
                verbose=args.verbose2,
                varianceFloor=varianceFloorForTrend,
                varianceCap=maxR_ if maxR_ is not None and maxR_ > 0.0 else None,
                intervalsArr=muncIntervalsArr,
                excludeMaskArr=muncExcludeMask,
                pooledTrend=pooledTrend,
                replicateVarianceFactor=replicateVarianceFactor,
                EB_pooledNu0=pooledNu0,
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

        if observationArgs.minR < 0.0:
            finiteMask = np.isfinite(muncMat)
            if blacklistedIntervals and blacklistedIntervals < numIntervals:
                finiteMask[:, np.asarray(muncExcludeMask, dtype=bool)] = False
            finiteMunc = muncMat[finiteMask]
            fallbackMinR = (
                max(float(np.quantile(finiteMunc, 0.05)) + 1.0e-3, 1.0e-4)
                if finiteMunc.size
                else 1.0e-4
            )
            muncMat = np.nan_to_num(
                muncMat.astype(np.float32, copy=False),
                nan=np.float32(0.0),
                posinf=np.float32(maxR_),
                neginf=np.float32(0.0),
            )
            np.clip(
                muncMat,
                np.float32(0.0),
                np.float32(maxR_),
                out=muncMat,
            )
            logger.info(
                "observationParams.minR < 0 --> estimating empirical observation-variance floor from held-out innovations",
            )
            pilotMinQ = processArgs.minQ
            pilotMaxQ = processArgs.maxQ
            if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
                pilotAutoMinQ = (1.0e-2 * fallbackMinR) + 1.0e-6
                pilotMinQ = (
                    np.float32(pilotAutoMinQ)
                    if processArgs.minQ < 0.0
                    else np.float32(processArgs.minQ)
                )
                pilotMaxQ = (
                    np.float32(np.inf)
                    if processArgs.maxQ < 0.0
                    else np.float32(max(processArgs.maxQ, pilotMinQ))
                )
            try:
                from consenrich import uncertainty as uncertainty_module

                floorRunKwargs = dict(
                    deltaF=deltaF_,
                    minQ=pilotMinQ,
                    maxQ=pilotMaxQ,
                    stateInit=stateArgs.stateInit,
                    stateCovarInit=stateArgs.stateCovarInit,
                    boundState=stateArgs.boundState,
                    stateLowerBound=stateArgs.stateLowerBound,
                    stateUpperBound=stateArgs.stateUpperBound,
                    blockLenIntervals=blockLenIntervals_,
                    intervalSizeBP=intervalSizeBP,
                    returnScales=True,
                    returnReplicateOffsets=True,
                    pad=pad_,
                    ECM_fixedBackgroundIters=fitArgs.ECM_fixedBackgroundIters,
                    ECM_fixedBackgroundRtol=fitArgs.ECM_fixedBackgroundRtol,
                    ECM_robustTNu=fitArgs.ECM_robustTNu,
                    ECM_useObsPrecisionReweighting=(
                        fitArgs.ECM_useObsPrecisionReweighting
                    ),
                    ECM_useProcessPrecisionReweighting=(
                        fitArgs.ECM_useProcessPrecisionReweighting
                    ),
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
                    observationPrecisionMultiplierMin=(
                        observationArgs.precisionMultiplierMin
                    ),
                    observationPrecisionMultiplierMax=(
                        observationArgs.precisionMultiplierMax
                    ),
                    initialBackground=(
                        muncResidualBackground if bool(fitArgs.fitBackground) else None
                    ),
                    logIndentLevel=1,
                    logRunRole="observation-floor pilot",
                )
                floorResult = (
                    uncertainty_module.estimateObservationVarianceFloorFromHeldout(
                        matrixData=chromMat,
                        matrixMunc=muncMat,
                        intervalSizeBP=intervalSizeBP,
                        params=uncertaintyCalibrationArgs,
                        runKwargs=floorRunKwargs,
                        pad=pad_,
                        maxR=float(maxR_),
                        fallbackMinR=float(fallbackMinR),
                        excludeIntervals=np.asarray(muncExcludeMask, dtype=bool),
                        chromosome=chromosome,
                    )
                )
                minR_ = np.float32(floorResult.minR)
                logger.info(
                    "Empirical observation floor applied for %s: minR=%.6g "
                    "trimmedMean=%.6g heldoutCells=%d fitCells=%d usedLambda=%s",
                    chromosome,
                    float(floorResult.minR),
                    float(floorResult.trimmedMean),
                    int(floorResult.heldoutCells),
                    int(floorResult.fitCells),
                    bool(floorResult.usedLambda),
                )
            except Exception as exc:
                minR_ = np.float32(fallbackMinR)
                logger.warning(
                    "Empirical observation floor calibration failed for %s; "
                    "using fallback minR=%.6g. Error: %s",
                    chromosome,
                    float(minR_),
                    exc,
                )
        if blacklistedIntervals:
            floors = core.applyBlacklistMuncFloor(
                muncMat, muncExcludeMask, float(minR_)
            )
            logger.info(
                "munc matrix: applied blacklist floors (chrom=%s, min=%.4g, median=%.4g, max=%.4g).",
                chromosome,
                float(np.min(floors)),
                float(np.median(floors)),
                float(np.max(floors)),
            )
        muncMat = np.nan_to_num(
            muncMat.astype(np.float32, copy=False),
            nan=np.float32(minR_),
            posinf=np.float32(maxR_),
            neginf=np.float32(minR_),
        )
        np.clip(
            muncMat,
            np.float32(minR_),
            np.float32(maxR_),
            out=muncMat,
        )
        minQ_ = processArgs.minQ
        maxQ_ = processArgs.maxQ

        if processArgs.minQ < 0.0 or processArgs.maxQ < 0.0:
            if minR_ is None:
                minR_ = np.float32(np.quantile(muncMat, 0.05) + 1.0e-3)
            effectiveDeltaFForMinQ = (
                1.0
                if core._normalizeStateModel(processArgs.stateModel)
                == core.STATE_MODEL_LEVEL
                else deltaF_
            )
            autoMinQ = (1.0e-2 * minR_) + 1.0e-6
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
        logger.info(f"minR={minR_}, maxR={maxR_}, minQ={minQ_}, maxQ={maxQ_}")
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
                    int(backgroundBlockSizeBP_) if int(backgroundBlockSizeBP_) > 0 else "auto",
                ),
                ("background window intervals", int(blockLenIntervals_)),
                ("minR", float(minR_)),
                ("maxR", float(maxR_)),
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
        useCrossFitUncertainty = bool(
            outputArgs.writeUncertainty and uncertaintyCalibrationArgs.enabled
        )
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
            returnReplicateOffsets=True,
            returnBackground=saveBackgroundTracks,
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
            ECM_zeroCenterReplicateBias=fitArgs.ECM_zeroCenterReplicateBias,
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
            logRunRole="primary chromosome",
        )
        x, P, postFitResiduals, _NISVec, replicateBias, intervalToBlockMap = runResult[
            :6
        ]
        runResultIndex = 6
        backgroundTrack = None
        if saveBackgroundTracks:
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
            optimizationPathPrefix = _optimizationPathPrefix(
                str(experimentName),
                chromosome,
            )
            _writeOptimizationPathLog(
                optimizationPathRows,
                f"{optimizationPathPrefix}.log",
            )
            _plotOptimizationPathLog(
                optimizationPathRows,
                f"{optimizationPathPrefix}.png",
                dpi=400,
            )
        if precisionDiagnostics:
            precisionPath = (
                f"consenrichOutput_{experimentName}_{chromosome}"
                f"_precisionDiagnostics.v{__version__}.tsv.gz"
            )
            _writePrecisionDiagnostics(
                chromosome=chromosome,
                intervals=intervals,
                intervalSizeBP=intervalSizeBP,
                matrixMunc=muncMat,
                pad=pad_,
                precisionDiagnostics=precisionDiagnostics,
                path=precisionPath,
            )
        logger.info(
            "runConsenrich.done %s elapsed=%.3fs",
            chromosome,
            time.perf_counter() - runStart,
        )
        replicateBias = np.asarray(replicateBias, dtype=np.float32)
        logger.info(
            "finalReplicateBias[%s]=%s",
            chromosome,
            np.array2string(
                replicateBias,
                precision=6,
                floatmode="fixed",
                separator=", ",
            ),
        )
        finalForwardGainSummary = runDiagnostics.get(
            "final_forward_gain_contig_summary"
        )
        if isinstance(finalForwardGainSummary, Mapping):
            gainMeans = finalForwardGainSummary.get("mean", [])
            gainMedians = finalForwardGainSummary.get("median", [])
            gainSds = finalForwardGainSummary.get("sd", [])
            gainIqrs = finalForwardGainSummary.get("iqr", [])
            logger.info(
                "\n%s\n",
                _formatReplicateGainFrame(
                    chromosome,
                    treatmentSources,
                    list(gainMeans if gainMeans is not None else []),
                    list(gainMedians if gainMedians is not None else []),
                    list(gainSds if gainSds is not None else []),
                    list(gainIqrs if gainIqrs is not None else []),
                    controlSources=(controlSources if controlsPresent else None),
                    indentLevel=1,
                ),
            )
        backgroundWarmStart = None
        if bool(fitArgs.fitBackground):
            rawByInterval = np.asarray(chromMat, dtype=np.float32).T
            residualWarmStart = np.asarray(postFitResiduals, dtype=np.float32)
            stateLevelWarmStart = np.asarray(x[:, 0], dtype=np.float32)
            if rawByInterval.shape == residualWarmStart.shape:
                backgroundWarmStart = np.mean(
                    rawByInterval
                    - stateLevelWarmStart[:, None]
                    - replicateBias[None, :]
                    - residualWarmStart,
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
            qLevelWarmStart = processNoiseDiagnostics.get("qLevel")
            qTrendWarmStart = processNoiseDiagnostics.get("qTrend")
            if (
                core._normalizeStateModel(processArgs.stateModel)
                == core.STATE_MODEL_LEVEL
                and qLevelWarmStart is not None
            ):
                initialProcessQWarmStart = core.constructMatrixQ(
                    minDiagQ=float(minQ_),
                    Q00=float(qLevelWarmStart),
                    Q01=0.0,
                    Q10=0.0,
                    Q11=max(float(qLevelWarmStart), float(minQ_)),
                )
            elif qLevelWarmStart is not None and qTrendWarmStart is not None:
                initialProcessQWarmStart = core.constructMatrixQ(
                    minDiagQ=float(minQ_),
                    Q00=float(qLevelWarmStart),
                    Q01=0.0,
                    Q10=0.0,
                    Q11=float(qTrendWarmStart),
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

        if useCrossFitUncertainty:
            core._logAsciiBlock(
                "uncertainty calibration",
                (
                    ("chromosome", chromosome),
                    ("mode", "cross-fit"),
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
                    "Cross-fit uncertainty calibration requires the optional "
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
                returnReplicateOffsets=True,
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
                initialReplicateBias=replicateBias,
                initialProcessQ=initialProcessQWarmStart,
                logIndentLevel=2,
                logRunRole="held-out fold",
            )
            calibrationPrefix = (
                f"consenrichOutput_{experimentName}_uncertaintyCalibration"
                f".v{__version__}"
            )
            calibrationResult = uncertainty_module.calibrateChromosomeStateUncertainty(
                matrixData=chromMat,
                matrixMunc=muncMat,
                fullState=x,
                fullCovar=P,
                fullReplicateBias=replicateBias,
                intervals=intervals,
                intervalSizeBP=intervalSizeBP,
                params=uncertaintyCalibrationArgs,
                runKwargs=calibrationRunKwargs,
                pad=pad_,
                outPrefix=calibrationPrefix,
                chromosome=chromosome,
            )
            uncertaintyTrack = np.asarray(
                calibrationResult.calibratedUncertainty,
                dtype=np.float32,
            )
            logger.info(
                "Cross-fit uncertainty calibration applied for %s: "
                "aObs=%.6g heldoutCells=%d",
                chromosome,
                float(calibrationResult.model.get("a_obs_factor", np.nan)),
                int(calibrationResult.model.get("heldout_cells", 0)),
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
        for col, suffix in _progress(
            bedGraphTracks,
            total=len(suffixes),
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
        logger.info(
            "chromosome.done %s elapsed=%.3fs outputElapsed=%.3fs",
            chromosome,
            time.perf_counter() - chromosomeStartTime,
            time.perf_counter() - writeStart,
        )

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
                logger.warning(
                    "Uncertainty bedGraph %s was not found; proceeding without model-based uncertainty.",
                    uncertaintyBedGraphPath,
                )
                uncertaintyBedGraphPath = None
            outName = peaks.solveRocco(
                stateBedGraphPath,
                uncertaintyBedGraphFile=uncertaintyBedGraphPath,
                tau0=float(matchingArgs.tau0),
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
                blacklistBedFile=genomeArgs.blacklistFile,
                randSeed=matchingArgs.randSeed,
                verbose=bool(args.verbose),
                stateDiagnosticsByChromosome=stateDiagnosticsByChromosome,
            )

            logger.info("Finished ROCCO peak calling. Written to %s", outName)
        except Exception as ex_:
            logger.warning(
                f"ROCCO peak calling raised an exception:\n\n\t{ex_}\n"
                f"Skipping peak-calling step...try running post hoc via `consenrich --match-bedGraph <bedGraphFile>`\n"
                f"\tSee ``consenrich -h`` for more details.\n"
            )


if __name__ == "__main__":
    main()
