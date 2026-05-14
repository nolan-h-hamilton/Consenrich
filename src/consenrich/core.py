# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
import os
import time
import warnings
from functools import lru_cache
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from scipy import ndimage, signal, stats, optimize, special, sparse, interpolate
from scipy.sparse import linalg as sparse_linalg
from tqdm import tqdm
from itrigamma import itrigamma
from . import cconsenrich
from . import ccounts
from .constants import (
    ALIGNMENT_SOURCE_KINDS,
    BEDGRAPH_SOURCE_KIND,
    FRAGMENTS_SOURCE_KIND,
    PROCESS_Q_CALIBRATION_MODES,
    PROCESS_Q_CALIBRATION_DEFAULT_OUTER_ITERS,
    PROCESS_Q_CALIBRATION_NONE,
    PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL,
    PROCESS_Q_DEFAULT_TREND_TARGET_RATIO,
    PROCESS_Q_NUMERICAL_FLOOR,
    PROCESS_Q_TREND_FLOOR_RATIO,
    SUPPORTED_BAM_INPUT_MODES,
    SUPPORTED_COUNT_MODES,
    SUPPORTED_FRAGMENT_POSITION_MODES,
    SUPPORTED_SOURCE_KINDS,
    UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX,
    UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN,
    UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY,
    UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH,
    UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
    UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
    UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MAX,
    UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MIN,
    UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS,
    UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES,
    UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS,
    UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE,
    UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
    UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS,
    UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT,
    UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET,
    UNCERTAINTY_CALIBRATION_FACTOR_MAX_MIN_RATIO,
    UNCERTAINTY_CALIBRATION_FACTOR_MIN_FLOOR,
    UNCERTAINTY_CALIBRATION_FEATURE_HIGH_SIGNAL_QUANTILE,
    UNCERTAINTY_CALIBRATION_FEATURE_MAD_NORMAL_SCALE,
    UNCERTAINTY_CALIBRATION_FEATURE_NAMES,
    UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR,
    UNCERTAINTY_CALIBRATION_FEATURE_SCALE_FLOOR,
    UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE,
    UNCERTAINTY_CALIBRATION_MIN_CALIBRATION_ECM_ITERS,
    UNCERTAINTY_CALIBRATION_MIN_FOLDS,
    UNCERTAINTY_CALIBRATION_MIN_HOLDOUT_REPLICATES,
    UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
    UNCERTAINTY_CALIBRATION_REFIT_PROCESS_Q_WARMUP_OUTER_ITERS,
    UNCERTAINTY_CALIBRATION_REFIT_PROCESS_Q_WARMUP_ECM_ITERS,
    UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE,
    UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES,
    UNCERTAINTY_CALIBRATION_SCORE_REPLICATE_CODE_STRIDE,
    UNCERTAINTY_CALIBRATION_SCORE_STATE_ABS_QUANTILE,
    UNCERTAINTY_CALIBRATION_SUMMARY_MEDIAN_QUANTILE,
    UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE,
    UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
    UNCERTAINTY_CALIBRATION_WIS_SCALE_MULTIPLIER,
    UNCERTAINTY_CALIBRATION_WIS_WEIGHT,
)
from .diagnostics import metadataFloat, summarizePrecisionBoundaryHits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_LOG_BLOCK_WIDTH = 72
_LOG_KEY_WIDTH = 24
_LOG_INDENT_WIDTH = 6


def _formatLogValue(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}" if np.isfinite(value) else str(value)
    return str(value)


def _formatAsciiLogBlock(
    title: str,
    rows: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] = (),
    *,
    indentLevel: int = 0,
) -> str:
    width = _LOG_BLOCK_WIDTH
    titleWidth = width - 4
    keyWidth = _LOG_KEY_WIDTH
    valueWidth = width - keyWidth - 7
    indent = " " * (max(0, int(indentLevel)) * _LOG_INDENT_WIDTH)
    titleLine = f"PHASE: {str(title).upper()}"
    if len(titleLine) > titleWidth:
        titleLine = titleLine[: titleWidth - 1] + "~"
    outerBorder = "+" + ("=" * (width - 2)) + "+"
    lines = [outerBorder, f"| {titleLine:<{titleWidth}} |"]
    if rows:
        rowBorder = "+" + ("-" * (keyWidth + 2)) + "+" + ("-" * (valueWidth + 2)) + "+"
        lines.append(rowBorder)
        for key, value in rows:
            keyText = str(key)
            valueText = _formatLogValue(value)
            if (
                "\n" in keyText
                or "\n" in valueText
                or len(keyText) > keyWidth
                or len(valueText) > valueWidth
            ):
                continue
            lines.append(f"| {keyText:<{keyWidth}} | {valueText:<{valueWidth}} |")
            lines.append(rowBorder)
    if indent:
        return "\n".join(f"{indent}{line}" for line in lines)
    return "\n".join(lines)


def _logAsciiBlock(
    title: str,
    rows: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] = (),
    *,
    logger_: logging.Logger = logger,
    level: int = logging.INFO,
    indentLevel: int = 0,
) -> None:
    logger_.log(
        level,
        "\n%s\n",
        _formatAsciiLogBlock(title, rows, indentLevel=indentLevel),
        stacklevel=2,
    )


class processParams(NamedTuple):
    r"""Parameters related to the process model of Consenrich.

    :param deltaF: Fixed positive integration step size in the two-state
        transition :math:`x_{[i+1,0]} = x_{[i,0]} + \delta_F x_{[i,1]}`.
    :type deltaF: float
    :param minQ: Minimum process noise scale (diagonal in :math:`\mathbf{Q}_{[i]}`)
        on the primary state variable (signal level). If ``minQ < 0``, a small
        value scales the minimum observation noise level (``observationParams.minR``) and is used
        for numerical stability.
    :type minQ: float
    :param maxQ: Maximum process noise scale. If ``maxQ < 0``, no effective upper bound is enforced.
    :type maxQ: float
    :param offDiagQ: Off-diagonal value in the process noise covariance :math:`\mathbf{Q}_{[i,01]}`
    :type offDiagQ: float
    :param processQCalibration: Process-noise covariance setup mode. ``"regularizedDiagonal"``
        runs a short pre-fit calibration pass to estimate a fixed diagonal base
        covariance :math:`\mathbf{Q}_0 = \mathrm{diag}(q_\mathrm{level}, q_\mathrm{trend})`.
        ``"none"`` preserves the legacy scalar process covariance.
    :type processQCalibration: str
    :param processQWarmupECMIters: Maximum fixed-background ECM iterations used by
        the process-Q warm-up fit.
    :type processQWarmupECMIters: int
    :param processQWarmupOuterIters: Maximum outer alternation passes used by
        the process-Q warm-up calibration fit.
    :type processQWarmupOuterIters: int
    :param processQLevelTarget: Optional shrinkage target for level innovation variance.
        If unset, the resolved ``minQ`` value is used.
    :type processQLevelTarget: float | None
    :param processQTrendTarget: Optional shrinkage target for trend innovation variance.
        If unset, ``PROCESS_Q_DEFAULT_TREND_TARGET_RATIO * processQLevelTarget`` is used.
    :type processQTrendTarget: float | None
    :param processQLevelPriorWeight: Shrinkage weight toward ``processQLevelTarget``.
    :type processQLevelPriorWeight: float
    :param processQTrendPriorWeight: Shrinkage weight toward ``processQTrendTarget``.
    :type processQTrendPriorWeight: float
    :param precisionMultiplierMin: Lower clamp for process precision multipliers
        :math:`\kappa_{[i]}` during robust ECM reweighting.
    :type precisionMultiplierMin: float
    :param precisionMultiplierMax: Upper clamp for process precision multipliers
        :math:`\kappa_{[i]}` during robust ECM reweighting.
    :type precisionMultiplierMax: float
    :seealso: :func:`consenrich.core.runConsenrich`

    """

    deltaF: float = 1.0
    minQ: float = 2.5e-4
    maxQ: float = 1000.0
    offDiagQ: float = 0.0
    processQCalibration: str = "regularizedDiagonal"
    processQWarmupECMIters: int = 3
    processQWarmupOuterIters: int = PROCESS_Q_CALIBRATION_DEFAULT_OUTER_ITERS
    processQLevelTarget: float | None = None
    processQTrendTarget: float | None = None
    processQLevelPriorWeight: float = 0.05
    processQTrendPriorWeight: float = 1.0
    precisionMultiplierMin: float = 0.5
    precisionMultiplierMax: float = 2.0


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich

    :param minR: Genome-wide lower bound for replicate-specific observation noise levels.
    :type minR: float | None
    :param maxR: Genome-wide upper bound for the replicate-specific observation noise levels.
    :type maxR: float | None
    :param samplingIters: Number of blocks (within-contig) to sample while building the empirical signed-mean variance trend in :func:`consenrich.core.fitPSplineLogVarianceTrend`.
    :type samplingIters: int | None
    :param samplingBlockSizeBP: Expected size (in bp) of contiguous blocks that are sampled when fitting AR1 parameters to estimate signed :math:`(\mu_b, \sigma^2_b)` pairs.
      Note, during sampling, each block's size (unit: genomic intervals) is drawn from truncated :math:`\textsf{Geometric}(p=1/\textsf{samplingBlockSize})` to reduce artifacts from fixed-size blocks.
      If `None` or ` < 1`, then this value is inferred using :func:`consenrich.core.chooseDependenceLength`.
    :type samplingBlockSizeBP: int | None
    :param EB_use: If True, shrink 'local' noise estimates to a prior trend dependent on amplitude. See  :func:`consenrich.core.getMuncTrack`.
    :type EB_use: bool | None
    :param EB_setNu0: If provided, manually set :math:`\nu_0` to this value (rather than computing via :func:`consenrich.core.EB_computePriorStrength`).
    :type EB_setNu0: int | None
    :param EB_setNuL: If provided, manually set local model df, :math:`\nu_L`, to this value.
    :type EB_setNuL: int | None
    :param trendNumBasis: Upper bound on P-spline basis functions for the global log-variance trend.
    :type trendNumBasis: int | None
    :param trendMinObsPerBasis: Minimum effective trend observations per fitted spline basis function.
    :type trendMinObsPerBasis: float | None
    :param trendMinEdf: Minimum effective degrees of freedom accepted during guarded GCV selection.
    :type trendMinEdf: float | None
    :param trendMaxEdf: Maximum effective degrees of freedom accepted during guarded GCV selection. If unset, a conservative default cap is used.
    :type trendMaxEdf: float | None
    :param trendLambdaMin: Lower endpoint for the guarded GCV smoothing-parameter grid.
    :type trendLambdaMin: float | None
    :param trendLambdaMax: Upper endpoint for the guarded GCV smoothing-parameter grid.
    :type trendLambdaMax: float | None
    :param trendLambdaGridSize: Number of points in the guarded GCV smoothing-parameter grid.
    :type trendLambdaGridSize: int | None
    :param numNearest: If ``> 0`` and an explicit sparse BED is supplied, estimate the local observation variance from the nearest sparse regions instead of the default rolling AR(1) local variance.
      In this sparse-nearest mode, the same nearest sparse blocks also define a signed local intercept track that is subtracted before fitting and evaluating the global mean-variance prior.
    :type numNearest: int | None
    :param sparseSupportScaleBP: Exponential length scale :math:`\ell`, in bp, used to soften sparse-nearest estimates by support density. If unset or non-positive, defaults to the local observation window scale.
    :type sparseSupportScaleBP: float | None
    :param sparseSupportPrior: Positive pseudo-count :math:`k` in ``n_eff / (n_eff + k)``. Smaller values make sparse evidence dominate more aggressively; values ``<= 0`` disable soft blending where sparse support exists.
    :type sparseSupportPrior: float | None
    :param restrictLocalAR1ToSparseBed: If True, and a sparse BED mask is supplied to :func:`consenrich.core.getMuncTrack`, restrict the default rolling AR(1) local observation noise level estimates to windows fully contained in sparse BED regions.
      This only affects the local rolling AR(1) model, the global prior fit and sparse-nearest mode are unchanged.
    :type restrictLocalAR1ToSparseBed: bool | None
    :param pad: A small constant added to the observation noise variance estimates for conditioning
    :type pad: float | None
    :param precisionMultiplierMin: Lower clamp for observation precision multipliers
        :math:`\lambda_{[i]}` during robust ECM reweighting.
    :type precisionMultiplierMin: float | None
    :param precisionMultiplierMax: Upper clamp for observation precision multipliers
        :math:`\lambda_{[i]}` during robust ECM reweighting.
    :type precisionMultiplierMax: float | None
    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitPSplineLogVarianceTrend`, :func:`consenrich.core.EB_computePriorStrength`, :func:`consenrich.cconsenrich.cfixedBackgroundECM`

    """

    minR: float | None
    maxR: float | None
    samplingIters: int | None
    samplingBlockSizeBP: int | None
    EB_use: bool | None
    EB_setNu0: int | None
    EB_setNuL: int | None
    trendNumBasis: int | None
    trendMinObsPerBasis: float | None
    trendMinEdf: float | None
    trendMaxEdf: float | None
    trendLambdaMin: float | None
    trendLambdaMax: float | None
    trendLambdaGridSize: int | None
    numNearest: int | None
    sparseSupportScaleBP: float | None
    sparseSupportPrior: float | None
    restrictLocalAR1ToSparseBed: bool | None
    pad: float | None
    precisionMultiplierMin: float | None = 0.25
    precisionMultiplierMax: float | None = 4.0


class stateParams(NamedTuple):
    r"""Parameters related to state variables and covariances.

    :param stateInit: Initial value of the 'primary' state/signal at the first genomic interval: :math:`x_{[1]}`
    :type stateInit: float
    :param stateCovarInit: Initial state covariance (covariance) scale. Note, the *initial* state uncertainty :math:`\mathbf{P}_{[1]}` is a multiple of the identity matrix :math:`\mathbf{I}`. Final results are typically insensitive to this parameter, since the filter effectively 'forgets' its initialization after processing a moderate number of intervals and backward smoothing.
    :type stateCovarInit: float
    :param boundState: If True, the primary state estimate for :math:`x_{[i]}` is reported within `stateLowerBound` and `stateUpperBound`. Note that the internal filtering is unaffected.
    :type boundState: bool
    :param stateLowerBound: Lower bound for the state estimate.
    :type stateLowerBound: float
    :param stateUpperBound: Upper bound for the state estimate.
    :type stateUpperBound: float
    """

    stateInit: float
    stateCovarInit: float
    boundState: bool
    stateLowerBound: float
    stateUpperBound: float


class uncertaintyCalibrationParams(NamedTuple):
    r"""Parameters for cross-fit chromosome state-uncertainty calibration."""

    enabled: bool = False
    folds: int = 3
    blockSizeBP: int | str | None = None
    holdoutFraction: float | None = None
    heldoutReplicateFraction: float | None = None
    maxScores: int = UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES
    maxHeldoutCells: int | None = None
    maxDiagnosticRows: int = UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS
    minHeldoutCells: int = UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS
    targets: tuple[float, ...] = UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS
    minFactor: float = UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN
    maxFactor: float = UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX
    factorMin: float | None = None
    factorMax: float | None = None
    ridge: float = UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE
    wisWeight: float = UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT
    aObsPenalty: float = UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY
    aObsPriorStrength: float | None = None
    calibrationECMIters: int = 2
    seed: int = UNCERTAINTY_CALIBRATION_DEFAULT_SEED
    pad: float | None = None
    writeDiagnostics: bool = False


def checkStateUncertaintyCoverage(
    residual: npt.ArrayLike,
    uncertaintyBefore: npt.ArrayLike,
    uncertaintyAfter: npt.ArrayLike | None = None,
    targets: tuple[float, ...] = UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS,
    *,
    strata: dict[str, npt.ArrayLike] | None = None,
    minUncertainty: float = UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
) -> list[dict[str, float | int | str | None]]:
    r"""Empirical coverage of held-out residuals under state uncertainty.

    ``uncertaintyBefore`` and ``uncertaintyAfter`` are standard deviations for
    the same held-out residuals, typically the predictive state-plus-observation
    SD before and after the block-holdout uncertainty calibration.
    """

    residualArr = np.asarray(residual, dtype=np.float64).reshape(-1)
    beforeArr = np.asarray(uncertaintyBefore, dtype=np.float64).reshape(-1)
    if residualArr.shape != beforeArr.shape:
        raise ValueError("residual and uncertaintyBefore must have the same shape")
    afterArr = None
    if uncertaintyAfter is not None:
        afterArr = np.asarray(uncertaintyAfter, dtype=np.float64).reshape(-1)
        if afterArr.shape != residualArr.shape:
            raise ValueError("uncertaintyAfter must have the same shape as residual")

    valid = np.isfinite(residualArr) & np.isfinite(beforeArr) & (beforeArr > 0.0)
    if afterArr is not None:
        valid &= np.isfinite(afterArr) & (afterArr > 0.0)

    stratumMasks: list[tuple[str, np.ndarray]] = [("overall", valid)]
    if strata:
        for name, mask in strata.items():
            maskArr = np.asarray(mask, dtype=bool).reshape(-1)
            if maskArr.shape != residualArr.shape:
                raise ValueError(f"stratum {name!r} must match residual shape")
            stratumMasks.append((str(name), valid & maskArr))

    rows: list[dict[str, float | int | str | None]] = []
    for stratum, mask in stratumMasks:
        resid = np.abs(residualArr[mask])
        before = np.maximum(beforeArr[mask], float(minUncertainty))
        after = (
            np.maximum(afterArr[mask], float(minUncertainty))
            if afterArr is not None
            else None
        )
        n = int(resid.size)
        for target in tuple(float(x) for x in targets):
            targetClipped = float(
                np.clip(
                    target,
                    UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
                    1.0 - UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
                )
            )
            z = float(stats.norm.ppf(0.5 + 0.5 * targetClipped))
            if n == 0:
                coverageBefore = np.nan
                meanWidthBefore = np.nan
                medianWidthBefore = np.nan
                coverageAfter = None if after is None else np.nan
                meanWidthAfter = None if after is None else np.nan
                medianWidthAfter = None if after is None else np.nan
            else:
                widthBefore = 2.0 * z * before
                coverageBefore = float(np.mean(resid <= z * before))
                meanWidthBefore = float(np.mean(widthBefore))
                medianWidthBefore = float(np.median(widthBefore))
                if after is None:
                    coverageAfter = None
                    meanWidthAfter = None
                    medianWidthAfter = None
                else:
                    widthAfter = 2.0 * z * after
                    coverageAfter = float(np.mean(resid <= z * after))
                    meanWidthAfter = float(np.mean(widthAfter))
                    medianWidthAfter = float(np.median(widthAfter))
            rows.append(
                {
                    "stratum": stratum,
                    "target": float(target),
                    "z": z,
                    "n": n,
                    "coverage_before": coverageBefore,
                    "coverage_after": coverageAfter,
                    "mean_width_before": meanWidthBefore,
                    "mean_width_after": meanWidthAfter,
                    "median_width_before": medianWidthBefore,
                    "median_width_after": medianWidthAfter,
                }
            )
    return rows


class samParams(NamedTuple):
    r"""Parameters related to reading BAM files

    :param samThreads: The number of threads to use for reading BAM files.
    :type samThreads: int
    :param samFlagExclude: The SAM flag to exclude certain reads.
    :type samFlagExclude: int
    :param oneReadPerBin: If 1, only the interval with the greatest read overlap is incremented.
    :type oneReadPerBin: int
    :param chunkSize: maximum number of intervals' data to hold in memory before flushing to disk.
    :type chunkSize: int
    :param bamInputMode: Default interpretation for BAM inputs.
        ``auto`` resolves to fragment spans for paired-end BAM and per-read tags for single-end BAM.
        ``fragments`` uses proper template spans, ``reads`` counts each alignment independently,
        and ``read1`` uses only the first mate from paired-end BAM in a MACS3 ``-f BAM`` mode.
    :type bamInputMode: str
    :param defaultCountMode: Default count mode for BAM inputs when a source does not set ``countMode``.
    :type defaultCountMode: str
    :param shiftForward5p: 5' shift applied to forward-strand alignments.
    :type shiftForward5p: int
    :param shiftReverse5p: 5' shift applied to reverse-strand alignments.
    :type shiftReverse5p: int
    :param extendFrom5pBP: Optional extension length or list of extension lengths for BAM inputs operating
        in ``reads`` or ``read1`` mode. When omitted and fragment extension is requested, treatment BAMs
        are inferred and paired controls reuse those inferred lengths.
    :type extendFrom5pBP: List[int] | int | None
    :param maxInsertSize: Maximum frag length/insert to consider when estimating fragment length.
    :type maxInsertSize: int
    :param inferFragmentLength: Intended for single-end data: if > 0, the maximum correlation lag
       (avg.) between *strand-specific* read tracks is taken as the fragment length estimate and used to
       extend reads from shifted 5' ends when ``bamInputMode`` resolves to ``reads`` or ``read1``.
       When omitted in CLI configs and ``bamInputMode`` is ``auto``, single-end BAM inputs are extended
       by their inferred fragment lengths while paired-end BAM inputs keep fragment spans.
       This is often important when targeting broader marks (e.g., ChIP-seq H3K27me3).
    :type inferFragmentLength: Optional[int]
    :param minMappingQuality: Minimum mapping quality (MAPQ) for reads to be counted.
    :type minMappingQuality: Optional[int]

    .. tip::

        For an overview of SAM flags, see https://broadinstitute.github.io/picard/explain-flags.html

    """

    samThreads: int
    samFlagExclude: int
    oneReadPerBin: int
    chunkSize: int
    bamInputMode: str | None = "auto"
    defaultCountMode: str | None = "coverage"
    shiftForward5p: int | None = 0
    shiftReverse5p: int | None = 0
    extendFrom5pBP: List[int] | int | None = None
    maxInsertSize: Optional[int] = 1000
    inferFragmentLength: Optional[int] = None
    minMappingQuality: Optional[int] = 0
    minTemplateLength: Optional[int] = -1


class inputSource(NamedTuple):
    r"""Describes one input source for counting

    :param path: Path to an alignment, fragments, or bedGraph file
    :type path: str
    :param sourceKind: One of ``BAM``, ``FRAGMENTS``, or ``BEDGRAPH``
    :type sourceKind: str
    :param role: ``treatment`` or ``control``
    :type role: str
    :param sampleName: Optional sample label
    :type sampleName: str | None
    :param barcodeTag: Optional alignment tag used to read cell barcodes
    :type barcodeTag: str | None
    :param barcodeAllowListFile: Optional barcode allowlist path
    :type barcodeAllowListFile: str | None
    :param barcodeGroupMapFile: Optional barcode to group map path
    :type barcodeGroupMapFile: str | None
    :param selectGroups: Optional subset of barcode groups to keep
    :type selectGroups: List[str] | None
    :param countMode: Optional counting mode label.
      Inputs default to `coverage`; set `cutsite`, `fiveprime`, or `center`
      explicitly for endpoint counting.
    :type countMode: str | None
    :param bamInputMode: Optional BAM interpretation override for this source
    :type bamInputMode: str | None
    :param fragmentPositionMode: Optional fragments endpoint interpretation label
    :type fragmentPositionMode: str | None
    """

    path: str
    sourceKind: str = "BAM"
    role: str = "treatment"
    sampleName: str | None = None
    barcodeTag: str | None = None
    barcodeAllowListFile: str | None = None
    barcodeGroupMapFile: str | None = None
    selectGroups: List[str] | None = None
    countMode: str | None = None
    bamInputMode: str | None = None
    fragmentPositionMode: str | None = None


class inputParams(NamedTuple):
    r"""Parameters related to the input data for Consenrich.

    :param bamFiles: A list of paths to distinct coordinate-sorted and indexed BAM files.
    :type bamFiles: List[str]

    :param bamFilesControl: A list of paths to distinct coordinate-sorted and
        indexed control BAM files. e.g., IgG control inputs for ChIP-seq.
    :type bamFilesControl: List[str], optional
    :param treatmentSources: Parsed treatment input sources
    :type treatmentSources: List[inputSource] | None
    :param controlSources: Parsed control input sources
    :type controlSources: List[inputSource] | None
    """

    bamFiles: List[str]
    bamFilesControl: Optional[List[str]]
    treatmentSources: List[inputSource] | None = None
    controlSources: List[inputSource] | None = None


class genomeParams(NamedTuple):
    r"""Specify assembly-specific resources, parameters.

    :param genomeName: If supplied, default resources for the assembly (sizes file, blacklist, and 'sparse' regions) in `src/consenrich/data` are used.
      ``ce10, ce11, dm6, hg19, hg38, mm10, mm39`` have default resources available.
    :type genomeName: str
    :param chromSizesFile: A two-column TSV file with chromosome names and sizes (in base pairs).
    :type chromSizesFile: str
    :param blacklistFile: A BED file with regions to exclude.
    :type blacklistFile: str, optional
    :param sparseBedFile: A BED file with 'sparse regions' that are mutually exclusive with or devoid of the targeted signal. Used to estimate noise levels. See :func:`getMuncTrack`.
    :type sparseBedFile: str, optional
    :param chromosomes: A list of chromosome names to analyze. If None, all chromosomes in `chromSizesFile` are used.
    :type chromosomes: List[str]
    :param excludeChroms: A list of chromosome names to *exclude* from analysis.
    :type excludeChroms: List[str]
    :param excludeForNorm: A list of chromosome names to *exclude* when summing up the 'effective genome size' during normalization. This can be useful to avoid bias from poorly assembled, highly repetitive, and/or sex-specific chromosomes (e.g., chrM, chrUn, etc.). For reference, see `effective genome size <https://deeptools.readthedocs.io/en/latest/content/feature/effectiveGenomeSize.html>`_.
    :type excludeForNorm: List[str]
    """

    genomeName: str
    chromSizesFile: str
    blacklistFile: Optional[str]
    sparseBedFile: Optional[str]
    chromosomes: List[str]
    excludeChroms: List[str]
    excludeForNorm: List[str]


class countingParams(NamedTuple):
    r"""Parameters related to counting aligned reads

    :param intervalSizeBP: Length (bp) of each genomic interval :math:`i=1\ldots n` that comprise the larger genomic region (contig, chromosome, etc.)
        The default value is generally robust, but users may consider increasing this value when expected feature size is large and/or sequencing depth
        is low (less than :math:`\approx 5 \textsf{million}`, depending on assay).
    :type intervalSizeBP: int
    :param backgroundBlockSizeBP: Length (bp) of blocks used to estimate local statistics (background, noise, etc.). If a negative value is provided (default), this value is inferred from the data using :func:`consenrich.core.chooseDependenceLength`.
    :type backgroundBlockSizeBP: int
    :param normMethod: Method used to normalize read counts for sequencing depth / library size.

        - ``EGS``: Effective Genome Size normalization (see :func:`consenrich.detrorm.getScaleFactor1x`)
          only appropriate for alignment coverage.

        - ``SF``: Median of ratios scale factors (see :func:`consenrich.cconsenrich.cSF`). Restricted to analyses with ``>= 3`` samples (no input control).

        - ``RPKM`` / ``CPM``: Scale factors based on emitted counts per million mapped units
          fragments pseudobulks use emitted insertions rather than raw fragment totals

    :type normMethod: str
    :param fragmentsGroupNorm: Optional extra normalization for fragments pseudobulks
      `NONE` keeps library-size scaling only and `CELLS` additionally divides by selected cell count
    :type fragmentsGroupNorm: str | None
    :param fixControl: If True, treatment samples are not upscaled, and control
        samples are not downscaled. If False, treatment/control pairs use
        MACS normalization: the deeper sample is downscaled to the
        shallower sample.
    :type fixControl: bool, optional
    :param logOffset: A small constant added to read normalized counts before log-transforming (pseudocount). For example,  :math:`\log(x + 1)` for ``logOffset = 1``. Default is ``1.0``.
    :type logOffset: float, optional
    :param logMult: Multiplicative factor applied to log-scaled and normalized counts. For example, setting ``logMult = 1 / \log(2)`` will yield log2-scaled counts after transformation, and setting ``logMult = 1.0`` yields natural log-scaled counts.
    :type logMult: float, optional
    :param replicateMedianDetrend: If True, subtract a broad per-replicate
        quantile-filter trend after log/log-ratio transformation and before MUNC
        and state estimation.
    :type replicateMedianDetrend: bool | None
    :param replicateMedianDetrendWindowMultiplier: Multiplier applied to the
        effective background span when choosing the quantile-filter detrend
        window. The production default uses a doubly-wide window.
    :type replicateMedianDetrendWindowMultiplier: float | None
    :param gentleDetrendQuantile: Quantile used for broad per-replicate
        detrending. The default ``0.5`` reproduces the previous median filter.
    :type gentleDetrendQuantile: float | None
    :seealso: :func:`consenrich.cconsenrich.cTransform`

    .. admonition:: Treatment vs. Control Extension Lengths in Single-End Data
      :class: tip
      :collapsible: closed

      For single-end data, cross-correlation-based estimates for fragment length
      in control inputs can be biased due to a comparative lack of structure in
      strand-specific coverage tracks.

      This can create artifacts during counting, so it is common to use the estimated treatment
      fragment length for both treatment and control samples. Consenrich resolves inferred
      control extension lengths from their paired treatments.

    """

    intervalSizeBP: int | None
    backgroundBlockSizeBP: int | None
    scaleFactors: List[float] | None
    scaleFactorsControl: List[float] | None
    normMethod: str | None
    fragmentsGroupNorm: str | None
    fixControl: bool | None
    logOffset: float | None
    logMult: float | None
    replicateMedianDetrend: bool | None = True
    replicateMedianDetrendWindowMultiplier: float | None = 2.0
    gentleDetrendQuantile: float | None = 0.5


class scParams(NamedTuple):
    r"""Parameters related to single-cell and fragments inputs

    :param barcodeTag: Default alignment tag used to read cell barcodes from single-cell BAM inputs
    :type barcodeTag: str | None
    :param defaultCountMode: Default count mode for fragments inputs when a source does not set `countMode`
    :type defaultCountMode: str | None
    :param fragmentsGroupNorm: Optional extra normalization for fragments pseudobulks
      `NONE` keeps library-size scaling only and `CELLS` additionally divides by selected cell count
    :type fragmentsGroupNorm: str | None
    :param defaultFragmentPositionMode: Default fragments endpoint interpretation mode.
    :type defaultFragmentPositionMode: str | None
    """

    barcodeTag: str | None = "CB"
    defaultCountMode: str | None = "coverage"
    fragmentsGroupNorm: str | None = "NONE"
    defaultFragmentPositionMode: str | None = "insertionEndpoints"


class matchingParams(NamedTuple):
    r"""Parameters related to post-fit peak calling.

    Consenrich uses the within-package dynamic-programming peak caller based on ROCCO.

    :param enabled: If True, run post-fit ROCCO peak calling on the emitted state bedGraph.
    :type enabled: bool
    :param randSeed: Random seed used for bootstrap calibration and any stochastic tie-breaking.
    :type randSeed: Optional[int]
    :param tau0: Shrinkage-score pseudovariance parameter; direct ROCCO
        scoring uses the fitted Consenrich state values.
    :type tau0: Optional[float]
    :param numBootstrap: Number of dependent wild-bootstrap null draws used for budget calibration.
    :type numBootstrap: Optional[int]
    :param thresholdZ: One-sided null tail threshold on the ROCCO score, on a Gaussian ``z`` scale.
    :type thresholdZ: Optional[float]
    :param dependenceSpan: Optional fixed dependence span in intervals for bootstrap calibration.
        If not provided, it is estimated from the score track.
    :type dependenceSpan: Optional[int]
    :param gamma: ROCCO boundary penalty. Non-negative values are used directly
        (default ``0.5``); negative values request data-driven estimation from
        local score scale and context size.
    :type gamma: Optional[float]
    :param selectionPenalty: Optional direct per-bin selection penalty override.
    :type selectionPenalty: Optional[float]
    :param gammaScale: Multiplicative scale used when converting estimated context size into
        a ROCCO boundary penalty.
    :type gammaScale: Optional[float]
    :param nestedRoccoIters: Number of nested, monotone-shrinking local ROCCO refinement
        iterations to run within first-pass peak regions. Set to ``0`` to disable.
    :type nestedRoccoIters: Optional[int]
    :param nestedRoccoBudgetScale: Optional fraction of each eligible parent peak region
        available to the nested local ROCCO subproblem. Values below ``1`` make the local
        pass stricter. The default is ``0.5``.
    :type nestedRoccoBudgetScale: Optional[float]
    :param exportFilterUncertaintyMultiplier: Non-negative multiplier ``c`` in the
        final export filter ``medianState < -c * median(local uncertainty)``.
        The default is ``2.5``. Setting ``c=0`` requires exported peaks to have
        positive median signal.
    :type exportFilterUncertaintyMultiplier: Optional[float]
    :seealso: :mod:`consenrich.peaks`, :class:`outputParams`.
    """

    enabled: bool
    randSeed: Optional[int]
    tau0: Optional[float]
    numBootstrap: Optional[int]
    thresholdZ: Optional[float]
    dependenceSpan: Optional[int]
    gamma: Optional[float]
    selectionPenalty: Optional[float]
    gammaScale: Optional[float]
    nestedRoccoIters: Optional[int]
    nestedRoccoBudgetScale: Optional[float]
    exportFilterUncertaintyMultiplier: Optional[float]


class outputParams(NamedTuple):
    r"""Parameters related to output files.

    :param convertToBigWig: If True, output bedGraph files are converted to bigWig format.
    :type convertToBigWig: bool
    :param roundDigits: Number of decimal places to round output values (bedGraph)
    :type roundDigits: int
    :param writeUncertainty: If True, write the state uncertainty track to bedGraph.
        The default uncalibrated track is :math:`\sqrt{\widetilde{P}_{[i,0,0]}}`;
        when uncertainty calibration is enabled, the caller may replace it with the
        cross-fit calibrated state-variance track.
    :type writeUncertainty: bool

    """

    convertToBigWig: bool
    roundDigits: int
    writeUncertainty: bool


class fitParams(NamedTuple):
    r"""Parameters controlling the optimization/fitting procedures.

    These arguments control both the fixed-background ECM routine in
    :func:`consenrich.cconsenrich.cfixedBackgroundECM` and the outer
    fit/background alternation loop in :func:`consenrich.core.runConsenrich`.

    Fixed-background ECM:

    1. Filter-smoother state estimation *given* current noise scales
    2. Interval-level Student-t precision reweighting at: \(\lambda_{[i]}\) and \(\kappa_{[i]}\)
    3. Replicate-level observation offset updates: \(b_j\)

    Outer alternation:

    1. run the fixed-background ECM path against the current shared background
    2. optionally update a shared low-frequency background track \(g_{[i]}\),
       optionally constrained to have mean zero

    Replicate-level bias calibration, replicate-bias centering, and robust
    precision reweighting are fixed parts of the ECM path.


    :param ECM_fixedBackgroundIters: Maximum fixed-background ECM iterations.
    :type ECM_fixedBackgroundIters: int
    :param ECM_fixedBackgroundRtol: Relative tolerance used for the fixed-background NLL stabilization test.
      The fixed-background ECM loop is treated as stable once
      ``abs(NLL_k - NLL_{k-1}) <= ECM_fixedBackgroundRtol * max(abs(NLL_k), abs(NLL_{k-1}), 1)``
      for two consecutive iterations.
    :type ECM_fixedBackgroundRtol: float
    :param ECM_robustTNu: Student-t df for reweighting strengths (smaller = stronger reweighting)
    :type ECM_robustTNu: float
    :param ECM_useObsPrecisionReweighting: If True, update observation noise precision multipliers \(\lambda_{[i]}\) (Student-\(t\) reweighting); otherwise \(\lambda\equiv 1\).
    :type ECM_useObsPrecisionReweighting: bool
    :param ECM_useProcessPrecisionReweighting: If True, update process noise precision multipliers \(\kappa_{[i]}\) (Student-\(t\) reweighting); otherwise \(\kappa\equiv 1\).
    :type ECM_useProcessPrecisionReweighting: bool
    :param ECM_useAPN: If True, use the adaptive-process-noise (APN)
      D-statistic update during filtering. This option disables
      ``ECM_useProcessPrecisionReweighting`` and technically voids guarantees
      of monotonic descent.
    :type ECM_useAPN: bool
    :param fitBackground: If True, estimate the shared low-frequency background
      track \(g_{[i]}\) in the outer loop. If False, keep \(g_{[i]} \equiv 0\).
    :type fitBackground: bool
    :param ECM_zeroCenterBackground: If True, enforce the identifiability
      constraint that the shared background has mean zero.
    :type ECM_zeroCenterBackground: bool
    :param ECM_outerIters: Number of alternations between the fixed-background ECM fit and shared background update.
    :type ECM_outerIters: int
    :param ECM_minOuterIters: Optional lower bound on the number
      of outer alternation passes. The default keeps the production minimum
      of three; calibration refits can set this to one explicitly.
    :type ECM_minOuterIters: int | None
    :param ECM_backgroundShiftRtol: Relative tolerance used by the outer loop's background-shift criterion.
      The background-shift criterion is stable once the maximum pointwise background update is at most
      ``ECM_backgroundShiftRtol * max(max(abs(g_next)), max(abs(g_cur)), 1)``.
    :type ECM_backgroundShiftRtol: float
    :param ECM_outerNLLRtol: Relative tolerance used by the outer loop's NLL-change criterion.
      The NLL criterion is stable once consecutive fixed-background ECM NLLs differ by at most
      ``ECM_outerNLLRtol * max(abs(NLL_next), abs(NLL_prev), 1)``.
    :type ECM_outerNLLRtol: float
    :param ECM_backgroundSmoothness: Multiplier applied to the first- and second-difference roughness penalties used for the shared background update.
    :type ECM_backgroundSmoothness: float
    :param ECM_backgroundLengthScaleMultiplier: Runtime multiplier converting the inferred or configured background dependence scale to the background fitting span; e.g. ``8`` gives ``8 * baseIntervals + 1``.
    :type ECM_backgroundLengthScaleMultiplier: float


    :seealso: :func:`consenrich.cconsenrich.cfixedBackgroundECM`, :func:`consenrich.core.runConsenrich`, :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitPSplineLogVarianceTrend`
    """

    ECM_fixedBackgroundIters: int | None = 50
    ECM_fixedBackgroundRtol: float | None = 1.0e-4
    ECM_robustTNu: float | None = 8.0
    ECM_useObsPrecisionReweighting: bool | None = True
    ECM_useProcessPrecisionReweighting: bool | None = True
    ECM_useAPN: bool | None = False
    ECM_zeroCenterBackground: bool | None = False
    ECM_outerIters: int | None = 3
    ECM_minOuterIters: int | None = None
    ECM_backgroundShiftRtol: float | None = 1.0e-3
    ECM_outerNLLRtol: float | None = 1.0e-4
    ECM_backgroundSmoothness: float | None = 1.0
    ECM_backgroundLengthScaleMultiplier: float | None = 8.0
    fitBackground: bool | None = True


def _inferAlignmentSourceKind(path: str) -> str:
    lowerPath = str(path).lower()
    if lowerPath.endswith(".cram"):
        raise ValueError("CRAM inputs are no longer supported.")
    if lowerPath.endswith((".bedgraph", ".bedgraph.gz", ".bdg", ".bdg.gz")):
        return BEDGRAPH_SOURCE_KIND
    return "BAM"


def getChromRanges(
    bamFile: str,
    chromosome: str,
    chromLength: int,
    samThreads: int,
    samFlagExclude: int,
    sourceKind: str = "BAM",
) -> Tuple[int, int]:
    r"""Get the start and end positions of reads in a chromosome from a BAM file.

    :param bamFile: See :class:`inputParams`.
    :type bamFile: str
    :param chromosome: the chromosome to read in `bamFile`.
    :type chromosome: str
    :param chromLength: Base pair length of the chromosome.
    :type chromLength: int
    :param samThreads: See :class:`samParams`.
    :type samThreads: int
    :param samFlagExclude: See :class:`samParams`.
    :type samFlagExclude: int
    :return: Tuple of start and end positions (nucleotide coordinates) in the chromosome.
    :rtype: Tuple[int, int]

    :seealso: :func:`getChromRangesJoint`, :func:`consenrich.ccounts.ccounts_getAlignmentChromRange`
    """
    start, end = ccounts.ccounts_getAlignmentChromRange(
        bamFile,
        chromosome,
        chromLength,
        samThreads,
        samFlagExclude,
        sourceKind=sourceKind,
    )
    return int(start), int(end)


def getChromRangesJoint(
    bamFiles: List[str],
    chromosome: str,
    chromSize: int,
    samThreads: int,
    samFlagExclude: int,
    sourceKinds: List[str] | None = None,
) -> Tuple[int, int]:
    r"""For multiple BAM files, reconcile a single start and end position over which to count reads,
    where the start and end positions are defined by the first and last reads across all BAM files.

    :param bamFiles: List of BAM files to read.
    :type bamFiles: List[str]
    :param chromosome: Chromosome to read.
    :type chromosome: str
    :param chromSize: Size of the chromosome.
    :type chromSize: int
    :param samThreads: Number of threads to use for reading the BAM files.
    :type samThreads: int
    :param samFlagExclude: SAM flag to exclude certain reads.
    :type samFlagExclude: int
    :return: Tuple of start and end positions.
    :rtype: Tuple[int, int]

    :seealso: :func:`getChromRanges`
    """
    starts = []
    ends = []
    if sourceKinds is None:
        sourceKinds = [_inferAlignmentSourceKind(path) for path in bamFiles]
    for bam_, sourceKind in zip(
        bamFiles,
        sourceKinds,
    ):
        start, end = getChromRanges(
            bam_,
            chromosome,
            chromLength=chromSize,
            samThreads=samThreads,
            samFlagExclude=samFlagExclude,
            sourceKind=sourceKind,
        )
        starts.append(start)
        ends.append(end)
    return min(starts), max(ends)


def getReadLength(
    bamFile: str,
    numReads: int,
    maxIterations: int,
    samThreads: int,
    samFlagExclude: int,
    sourceKind: str = "BAM",
) -> int:
    r"""Infer read length from mapped reads in a BAM file.

    Samples at least `numReads` reads passing criteria given by `samFlagExclude`
    and returns the median read length.

    :param bamFile: See :class:`inputParams`.
    :type bamFile: str
    :param numReads: Number of reads to sample.
    :type numReads: int
    :param maxIterations: Maximum number of iterations to perform.
    :type maxIterations: int
    :param samThreads: See :class:`samParams`.
    :type samThreads: int
    :param samFlagExclude: See :class:`samParams`.
    :type samFlagExclude: int
    :return: The median read length.
    :rtype: int

    :raises ValueError: If the read length cannot be determined after scanning `maxIterations` reads.

    :seealso: :func:`consenrich.ccounts.ccounts_getAlignmentReadLength`
    """
    if str(sourceKind).upper() == BEDGRAPH_SOURCE_KIND:
        return 0
    init_rlen = ccounts.ccounts_getAlignmentReadLength(
        bamFile,
        numReads,
        samThreads,
        maxIterations,
        samFlagExclude,
        sourceKind=sourceKind,
    )
    if init_rlen == 0:
        raise ValueError(
            f"Failed to determine read length in {bamFile}. Revise `numReads`, and/or `samFlagExclude` parameters?"
        )
    return init_rlen


def getReadLengths(
    bamFiles: List[str],
    numReads: int,
    maxIterations: int,
    samThreads: int,
    samFlagExclude: int,
    sourceKinds: List[str] | None = None,
) -> List[int]:
    r"""Get read lengths for a list of BAM files.

    :seealso: :func:`getReadLength`
    """
    if sourceKinds is None:
        sourceKinds = [_inferAlignmentSourceKind(path) for path in bamFiles]
    return [
        getReadLength(
            bamFile,
            numReads=numReads,
            maxIterations=maxIterations,
            samThreads=samThreads,
            samFlagExclude=samFlagExclude,
            sourceKind=sourceKind,
        )
        for bamFile, sourceKind in zip(
            bamFiles,
            sourceKinds,
        )
    ]


def getSourcePaths(sources: List[inputSource]) -> List[str]:
    r"""Return input source paths in order"""

    return [source.path for source in sources]


def getSourceKinds(sources: List[inputSource]) -> List[str]:
    r"""Return normalized source kinds in order"""

    return [str(source.sourceKind).upper() for source in sources]


def _loadBarcodeAllowSet(path: str | None) -> set[str]:
    barcodeSet: set[str] = set()
    if path is None or not str(path).strip():
        return barcodeSet
    with open(path, "r", encoding="utf-8") as fileHandle:
        for line in fileHandle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            barcodeSet.add(line.split()[0])
    return barcodeSet


def _resolveFragmentsBarcodeAllowSet(source: inputSource) -> set[str] | None:
    allowListPath = (
        str(source.barcodeAllowListFile)
        if source.barcodeAllowListFile is not None and str(source.barcodeAllowListFile)
        else None
    )
    groupMapPath = (
        str(source.barcodeGroupMapFile)
        if source.barcodeGroupMapFile is not None and str(source.barcodeGroupMapFile)
        else None
    )
    selectGroups = set(source.selectGroups or [])

    if allowListPath is None and (groupMapPath is None or len(selectGroups) == 0):
        return None

    allowSet = _loadBarcodeAllowSet(allowListPath)
    if groupMapPath is not None:
        groupSet: set[str] = set()
        with open(groupMapPath, "r", encoding="utf-8") as fileHandle:
            for line in fileHandle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", "\t").split()
                if len(parts) < 2:
                    continue
                barcode = parts[0]
                groupName = parts[1]
                if len(selectGroups) == 0 or groupName in selectGroups:
                    groupSet.add(barcode)
        if len(allowSet) > 0:
            allowSet &= groupSet
        else:
            allowSet = groupSet

    return allowSet


def _writeFragmentsAllowList(
    source: inputSource,
) -> tuple[str | None, str | None]:
    allowSet = _resolveFragmentsBarcodeAllowSet(source)
    if allowSet is None:
        return None, None

    if len(allowSet) == 0:
        raise ValueError(f"No barcodes selected for fragments source `{source.path}`")

    with NamedTemporaryFile(
        mode="w",
        prefix="consenrich_fragments_allow_",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as tempHandle:
        for barcode in sorted(allowSet):
            tempHandle.write(f"{barcode}\n")
        return tempHandle.name, tempHandle.name


def getFragmentsSelectedBarcodeCount(source: inputSource) -> int | None:
    allowSet = _resolveFragmentsBarcodeAllowSet(source)
    if allowSet is None:
        return None
    return len(allowSet)


def _normalizeCountMode(
    countMode: str | None,
    defaultMode: str,
) -> str:
    normalizedMode = str(countMode or defaultMode).strip().lower()
    if normalizedMode not in SUPPORTED_COUNT_MODES:
        raise ValueError(f"Unsupported countMode `{countMode}`")
    if normalizedMode == "coverage":
        return "coverage"
    if normalizedMode == "cutsite":
        return "cutsite"
    if normalizedMode == "fiveprime":
        return "fiveprime"
    return "center"


def _normalizeBamInputMode(bamInputMode: str | None) -> str:
    normalizedMode = str(bamInputMode or "auto").strip().lower()
    if normalizedMode not in SUPPORTED_BAM_INPUT_MODES:
        raise ValueError(f"Unsupported bamInputMode `{bamInputMode}`")
    return normalizedMode


def _normalizeFragmentPositionMode(fragmentPositionMode: str | None) -> str:
    normalizedMode = (
        str(fragmentPositionMode or "insertionEndpoints")
        .strip()
        .replace("_", "")
        .replace("-", "")
        .lower()
    )
    if normalizedMode not in SUPPORTED_FRAGMENT_POSITION_MODES:
        raise ValueError(f"Unsupported fragmentPositionMode `{fragmentPositionMode}`")
    return normalizedMode


@lru_cache(maxsize=None)
def _isAlignmentSourcePairedEnd(alignmentPath: str) -> bool:
    try:
        return bool(
            cconsenrich.cisAlignmentPairedEnd(
                alignmentPath,
                maxReads=1_000,
            )
        )
    except AttributeError:
        return bool(
            ccounts.ccounts_isAlignmentPairedEnd(
                alignmentPath,
                maxReads=1_000,
                sourceKind="BAM",
            )
        )


def _resolveSourceBamInputMode(
    source: inputSource,
    defaultBamInputMode: str | None,
) -> str:
    normalizedMode = _normalizeBamInputMode(source.bamInputMode or defaultBamInputMode)
    if str(source.sourceKind).upper() not in ALIGNMENT_SOURCE_KINDS:
        return "reads"
    if normalizedMode != "auto":
        return normalizedMode
    return "fragments" if _isAlignmentSourcePairedEnd(source.path) else "reads"


def _resolveSourceFlagExclude(
    samFlagExclude: int,
    sourceBamInputMode: str,
) -> int:
    if sourceBamInputMode == "read1":
        return int(samFlagExclude) | 8 | 128
    return int(samFlagExclude)


def _resolveExtendFrom5pBP(
    extendFrom5pBP: List[int] | int | None,
    sources: List[inputSource],
) -> list[int]:
    alignmentIndices = [
        index
        for index, source in enumerate(sources)
        if str(source.sourceKind).upper() in ALIGNMENT_SOURCE_KINDS
    ]
    resolvedValues = [0] * len(sources)

    if extendFrom5pBP is None:
        return resolvedValues

    if isinstance(extendFrom5pBP, int):
        configuredValues = [int(extendFrom5pBP)]
    else:
        configuredValues = [int(value) for value in extendFrom5pBP]

    if len(configuredValues) == 0:
        return resolvedValues

    if len(configuredValues) == 1:
        configuredValues = configuredValues * len(alignmentIndices)
    elif len(configuredValues) == len(alignmentIndices):
        pass
    elif len(configuredValues) == len(sources):
        return configuredValues
    else:
        raise ValueError(
            "`extendFrom5pBP` length must match BAM sources length, "
            f"all sources length, or 1: {len(configuredValues)}"
        )

    for sourceIndex, extendBP in zip(alignmentIndices, configuredValues):
        resolvedValues[sourceIndex] = int(extendBP)
    return resolvedValues


def readSegments(
    sources: List[inputSource],
    chromosome: str,
    start: int,
    end: int,
    intervalSizeBP: int,
    readLengths: List[int],
    scaleFactors: List[float],
    oneReadPerBin: int,
    samThreads: int,
    samFlagExclude: int,
    bamInputMode: str | None = "auto",
    defaultCountMode: str | None = "coverage",
    shiftForward5p: int | None = 0,
    shiftReverse5p: int | None = 0,
    extendFrom5pBP: List[int] | int | None = None,
    maxInsertSize: Optional[int] = 1000,
    inferFragmentLength: Optional[int] = 0,
    minMappingQuality: Optional[int] = 0,
    minTemplateLength: Optional[int] = -1,
) -> npt.NDArray[np.float32]:
    r"""Read binned tracks from generic input sources

    this is the source-agnostic entry point for counting.

    For BAM inputs, ``bamInputMode`` controls whether we count template spans, per-read alignments,
    or only read1 tags from paired-end BAM. Combine ``shiftForward5p`` / ``shiftReverse5p`` with
    ``extendFrom5pBP`` or ``inferFragmentLength`` to emulate MACS ``--shift`` and ``--extsize``
    behavior.
    """

    if len(sources) == 0:
        raise ValueError("sources list is empty")

    if len(readLengths) != len(sources) or len(scaleFactors) != len(sources):
        raise ValueError("readLengths and scaleFactors must match sources length")

    sourcePaths = getSourcePaths(sources)
    sourceKinds = getSourceKinds(sources)
    numIntervals = ((end - start - 1) // intervalSizeBP) + 1
    counts = np.empty((len(sources), numIntervals), dtype=np.float32)
    tempPaths: list[str] = []
    defaultBamInputMode = _normalizeBamInputMode(bamInputMode)
    defaultBamCountMode = _normalizeCountMode(defaultCountMode, "coverage")
    resolvedExtendValues = _resolveExtendFrom5pBP(extendFrom5pBP, sources)
    totalStart = time.perf_counter()
    logger.info(
        "readSegments.start chromosome=%s sources=%d intervals=%d samThreads=%d",
        chromosome,
        int(len(sources)),
        int(numIntervals),
        int(samThreads),
    )

    try:
        for sourceIndex, sourcePath in enumerate(sourcePaths):
            sourceStart = time.perf_counter()
            sourceKind = sourceKinds[sourceIndex]
            source = sources[sourceIndex]
            logger.info(
                "readSegments.source.start chromosome=%s source=%d/%d kind=%s path=%s",
                chromosome,
                int(sourceIndex + 1),
                int(len(sources)),
                sourceKind,
                sourcePath,
            )
            barcodeAllowListFile, tempPath = _writeFragmentsAllowList(source)
            if tempPath is not None:
                tempPaths.append(tempPath)

            if sourceKind == BEDGRAPH_SOURCE_KIND:
                counts[sourceIndex, :] = ccounts.ccounts_countAlignmentRegion(
                    sourcePath,
                    chromosome,
                    start,
                    end,
                    intervalSizeBP,
                    0,
                    0,
                    samThreads,
                    0,
                    shiftForwardStrand53=0,
                    shiftReverseStrand53=0,
                    extendBP=0,
                    maxInsertSize=0,
                    pairedEndMode=0,
                    inferFragmentLength=0,
                    minMappingQuality=0,
                    minTemplateLength=0,
                    sourceKind=sourceKind,
                    barcodeAllowListFile="",
                    barcodeGroupMapFile="",
                    countMode="coverage",
                )
            elif sourceKind == FRAGMENTS_SOURCE_KIND:
                countMode = _normalizeCountMode(source.countMode, "coverage")
                _normalizeFragmentPositionMode(source.fragmentPositionMode)
                counts[sourceIndex, :] = ccounts.ccounts_countAlignmentRegion(
                    sourcePath,
                    chromosome,
                    start,
                    end,
                    intervalSizeBP,
                    0,
                    oneReadPerBin,
                    samThreads,
                    0,
                    shiftForwardStrand53=0,
                    shiftReverseStrand53=0,
                    extendBP=0,
                    maxInsertSize=0,
                    pairedEndMode=0,
                    inferFragmentLength=0,
                    minMappingQuality=0,
                    minTemplateLength=0,
                    sourceKind=sourceKind,
                    barcodeAllowListFile=barcodeAllowListFile or "",
                    barcodeGroupMapFile="",
                    countMode=countMode,
                )
            else:
                countMode = _normalizeCountMode(
                    source.countMode,
                    defaultBamCountMode,
                )
                sourceBamInputMode = _resolveSourceBamInputMode(
                    source,
                    defaultBamInputMode,
                )
                sourceExtendBP = int(resolvedExtendValues[sourceIndex])
                sourceFlagExclude = _resolveSourceFlagExclude(
                    samFlagExclude,
                    sourceBamInputMode,
                )
                sourceInferFragmentLength = int(inferFragmentLength or 0)
                sourcePairedEndMode = 1 if sourceBamInputMode == "fragments" else 0

                if sourcePairedEndMode > 0:
                    if sourceExtendBP > 0 or sourceInferFragmentLength > 0:
                        raise ValueError(
                            "`extendFrom5pBP` and `inferFragmentLength` require "
                            "`bamInputMode` `reads` or `read1`."
                        )
                    sourceInferFragmentLength = 0
                elif sourceInferFragmentLength > 0 and sourceExtendBP <= 0:
                    sourceExtendBP = int(
                        cconsenrich.cgetFragmentLength(
                            sourcePath,
                            samThreads=samThreads,
                            samFlagExclude=sourceFlagExclude,
                            maxInsertSize=maxInsertSize,
                        )
                    )
                sourceInferFragmentLength = 0
                counts[sourceIndex, :] = ccounts.ccounts_countAlignmentRegion(
                    sourcePath,
                    chromosome,
                    start,
                    end,
                    intervalSizeBP,
                    readLengths[sourceIndex],
                    oneReadPerBin,
                    samThreads,
                    sourceFlagExclude,
                    shiftForwardStrand53=int(shiftForward5p or 0),
                    shiftReverseStrand53=int(shiftReverse5p or 0),
                    extendBP=sourceExtendBP,
                    maxInsertSize=maxInsertSize,
                    pairedEndMode=sourcePairedEndMode,
                    inferFragmentLength=sourceInferFragmentLength,
                    minMappingQuality=minMappingQuality,
                    minTemplateLength=minTemplateLength,
                    sourceKind=sourceKind,
                    barcodeAllowListFile="",
                    barcodeGroupMapFile="",
                    countMode=countMode,
                )
            np.multiply(
                counts[sourceIndex, :],
                np.float32(scaleFactors[sourceIndex]),
                out=counts[sourceIndex, :],
            )
            logger.info(
                "readSegments.source.done chromosome=%s source=%d/%d elapsed=%.3fs",
                chromosome,
                int(sourceIndex + 1),
                int(len(sources)),
                time.perf_counter() - sourceStart,
            )
    finally:
        for tempPath in tempPaths:
            try:
                os.remove(tempPath)
            except Exception:
                pass

    logger.info(
        "readSegments.done chromosome=%s sources=%d elapsed=%.3fs",
        chromosome,
        int(len(sources)),
        time.perf_counter() - totalStart,
    )
    return counts


def readBamSegments(
    bamFiles: List[str],
    chromosome: str,
    start: int,
    end: int,
    intervalSizeBP: int,
    readLengths: List[int],
    scaleFactors: List[float],
    oneReadPerBin: int,
    samThreads: int,
    samFlagExclude: int,
    bamInputMode: str | None = "auto",
    defaultCountMode: str | None = "coverage",
    shiftForward5p: int | None = 0,
    shiftReverse5p: int | None = 0,
    extendFrom5pBP: List[int] | int | None = None,
    maxInsertSize: Optional[int] = 1000,
    inferFragmentLength: Optional[int] = 0,
    minMappingQuality: Optional[int] = 0,
    minTemplateLength: Optional[int] = -1,
) -> npt.NDArray[np.float32]:
    r"""Calculate coverage tracks for each BAM file.

    :param bamFiles: See :class:`inputParams`.
    :type bamFiles: List[str]
    :param chromosome: Chromosome to read.
    :type chromosome: str
    :param start: Start position of the genomic segment.
    :type start: int
    :param end: End position of the genomic segment.
    :type end: int
    :param readLengths: List of read lengths for each BAM file.
    :type readLengths: List[int]
    :param scaleFactors: List of scale factors for each BAM file.
    :type scaleFactors: List[float]
    :param intervalSizeBP: See :class:`countingParams`.
    :type intervalSizeBP: int
    :param oneReadPerBin: See :class:`samParams`.
    :type oneReadPerBin: int
    :param samThreads: See :class:`samParams`.
    :type samThreads: int
    :param samFlagExclude: See :class:`samParams`.
    :type samFlagExclude: int
    :param bamInputMode: See :class:`samParams`.
    :type bamInputMode: str
    :param defaultCountMode: See :class:`samParams`.
    :type defaultCountMode: str
    :param shiftForward5p: See :class:`samParams`.
    :type shiftForward5p: int
    :param shiftReverse5p: See :class:`samParams`.
    :type shiftReverse5p: int
    :param extendFrom5pBP: See :class:`samParams`.
    :type extendFrom5pBP: Optional[List[int] | int]
    :param maxInsertSize: See :class:`samParams`.
    :type maxInsertSize: int
    :param inferFragmentLength: See :class:`samParams`.
    :type inferFragmentLength: int
    :param minMappingQuality: See :class:`samParams`.
    :type minMappingQuality: int
    :param minTemplateLength: See :class:`samParams`.
    :type minTemplateLength: Optional[int]

    This is a BAM-only convenience wrapper over :func:`readSegments`.
    """

    segmentSize_ = end - start
    if intervalSizeBP <= 0 or segmentSize_ <= 0:
        raise ValueError(
            "Invalid intervalSizeBP or genomic segment specified (end <= start)"
        )

    if len(bamFiles) == 0:
        raise ValueError("bamFiles list is empty")

    if len(readLengths) != len(bamFiles) or len(scaleFactors) != len(bamFiles):
        raise ValueError("readLengths and scaleFactors must match bamFiles length")

    return readSegments(
        [
            inputSource(
                path=bamPath,
                sourceKind="BAM",
            )
            for bamPath in bamFiles
        ],
        chromosome,
        start,
        end,
        intervalSizeBP,
        readLengths,
        scaleFactors,
        oneReadPerBin,
        samThreads,
        samFlagExclude,
        bamInputMode=bamInputMode,
        defaultCountMode=defaultCountMode,
        shiftForward5p=shiftForward5p,
        shiftReverse5p=shiftReverse5p,
        extendFrom5pBP=extendFrom5pBP,
        maxInsertSize=maxInsertSize,
        inferFragmentLength=inferFragmentLength,
        minMappingQuality=minMappingQuality,
        minTemplateLength=minTemplateLength,
    )


def constructMatrixF(deltaF: float) -> npt.NDArray[np.float32]:
    r"""Build the state transition matrix for the process model

    :param deltaF: See :class:`processParams`.
    :type deltaF: float
    :return: The state transition matrix :math:`\mathbf{F}`
    :rtype: npt.NDArray[np.float32]

    :seealso: :class:`processParams`
    """
    initMatrixF: npt.NDArray[np.float32] = np.eye(2, dtype=np.float32)
    initMatrixF[0, 1] = np.float32(deltaF)
    return initMatrixF


def _resolveFixedDeltaF(deltaF: float) -> float:
    deltaF_ = float(deltaF)
    if (not np.isfinite(deltaF_)) or deltaF_ <= 0.0:
        raise ValueError("deltaF must be a positive finite fixed step size")
    return deltaF_


def _normalizeProcessQCalibration(processQCalibration: str | None) -> str:
    if processQCalibration is None:
        return PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL

    mode = str(processQCalibration).strip()
    key = mode.lower().replace("_", "").replace("-", "")
    if key in {"none", "off", "false", "disabled", "disable"}:
        return PROCESS_Q_CALIBRATION_NONE
    if key in {
        "regularizeddiagonal",
        "regularizeddiag",
        "regularized",
        "diagonal",
        "diag",
    }:
        return PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL

    raise ValueError(
        "`processQCalibration` must be one of "
        + ", ".join(repr(mode_) for mode_ in PROCESS_Q_CALIBRATION_MODES)
    )


def _checkFinitePositive(name: str, value: float) -> float:
    value_ = float(value)
    if (not np.isfinite(value_)) or value_ <= 0.0:
        raise ValueError(f"`{name}` must be positive and finite")
    return value_


def _checkFiniteNonnegative(name: str, value: float) -> float:
    value_ = float(value)
    if (not np.isfinite(value_)) or value_ < 0.0:
        raise ValueError(f"`{name}` must be nonnegative and finite")
    return value_


def _checkPrecisionMultiplierBounds(
    prefix: str,
    minValue: float,
    maxValue: float,
) -> tuple[float, float]:
    minValue_ = _checkFinitePositive(f"{prefix}PrecisionMultiplierMin", minValue)
    maxValue_ = _checkFinitePositive(f"{prefix}PrecisionMultiplierMax", maxValue)
    if maxValue_ < minValue_:
        raise ValueError(
            f"`{prefix}PrecisionMultiplierMax` must be >= "
            f"`{prefix}PrecisionMultiplierMin`"
        )
    return minValue_, maxValue_


def _diagnosticScalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value_ = float(value)
        return value_ if np.isfinite(value_) else None
    if value is None:
        return None
    return value


def _formatMaybeFloat(value: Any) -> str:
    try:
        value_ = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{value_:.6g}" if np.isfinite(value_) else "NA"


def _observationLambdaSummary(
    lambdaExp: np.ndarray | None,
    *,
    lower: float,
    upper: float,
) -> tuple[Any, Any]:
    if lambdaExp is None:
        return None, None
    arr = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    clipped = np.clip(finite, float(lower), float(upper))
    return (
        metadataFloat(float(np.mean(clipped))),
        metadataFloat(float(np.median(clipped))),
    )


def _coerceOptionalVector(
    name: str,
    value: np.ndarray | None,
    length: int,
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (int(length),):
        raise ValueError(f"`{name}` must have length {int(length)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _coerceOptionalProcessQ(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (2, 2):
        raise ValueError("`initialProcessQ` must have shape (2, 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("`initialProcessQ` must contain only finite values")
    if arr[0, 0] <= 0.0 or arr[1, 1] <= 0.0:
        raise ValueError("`initialProcessQ` diagonal entries must be positive")
    if not np.allclose(arr, arr.T, rtol=1.0e-5, atol=1.0e-8):
        raise ValueError("`initialProcessQ` must be symmetric")
    try:
        np.linalg.cholesky(arr.astype(np.float64, copy=False) + 1.0e-8 * np.eye(2))
    except Exception as exc:
        raise ValueError("`initialProcessQ` must be positive definite") from exc
    return np.ascontiguousarray(arr, dtype=np.float32)


def _effectiveObservationCount(matrixMunc: np.ndarray) -> int:
    munc = np.asarray(matrixMunc, dtype=np.float64)
    active = np.isfinite(munc) & (
        munc < 0.5 * float(UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE)
    )
    return int(max(1, np.count_nonzero(active)))


def _robustPrecisionPenalty(
    *,
    lambdaExp: np.ndarray | None,
    processPrecExp: np.ndarray | None,
    robustTNu: float,
) -> tuple[float, float]:
    nu = float(robustTNu)
    obsPenalty = 0.0
    procPenalty = 0.0
    tiny = float(np.finfo(np.float64).tiny)
    if lambdaExp is not None:
        lam = np.maximum(np.asarray(lambdaExp, dtype=np.float64), tiny)
        obsPenalty = float(0.5 * nu * np.sum(lam - np.log(lam)))
    if processPrecExp is not None:
        kappa = np.maximum(np.asarray(processPrecExp, dtype=np.float64), tiny)
        if kappa.size > 1:
            kappa = kappa[1:]
        procPenalty = float(0.5 * nu * np.sum(kappa - np.log(kappa)))
    return obsPenalty, procPenalty


def _backgroundObjectivePenalty(
    *,
    background: np.ndarray,
    blockLenIntervals: int,
    backgroundSmoothness: float,
) -> tuple[float, float, float]:
    bg = np.asarray(background, dtype=np.float64).reshape(-1)
    lamFirst, lamSecond = _backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=int(blockLenIntervals),
        backgroundSmoothness=float(backgroundSmoothness),
    )
    if bg.size >= 2:
        d1 = np.diff(bg)
        firstPenalty = 0.5 * float(lamFirst) * float(np.dot(d1, d1))
    else:
        firstPenalty = 0.0
    if bg.size >= 3:
        d2 = np.diff(bg, n=2)
        secondPenalty = 0.5 * float(lamSecond) * float(np.dot(d2, d2))
    else:
        secondPenalty = 0.0

    return firstPenalty + secondPenalty, firstPenalty, secondPenalty


def _legacyFixedBackgroundECMDiagnostics(
    *,
    itersDone: int,
    finalNLL: float,
    maxIters: int,
) -> dict[str, Any]:
    finalNLL_ = float(finalNLL)
    return {
        "iters_done": int(itersDone),
        "max_iters": int(maxIters),
        "converged": None,
        "stable_iters": None,
        "patience_target": None,
        "initial_nll": None,
        "final_nll": metadataFloat(finalNLL_),
        "final_abs_rel_change": None,
        "final_rel_improvement": None,
        "nll_increase_count": None,
        "diagnostics_source": "legacy_tuple",
    }


def _normalizeFixedBackgroundECMDiagnostics(
    diagnostics: Mapping[str, Any] | None,
    *,
    itersDone: int,
    finalNLL: float,
    maxIters: int,
    outerPass: int,
) -> dict[str, Any]:
    if diagnostics is None:
        normalized = _legacyFixedBackgroundECMDiagnostics(
            itersDone=itersDone,
            finalNLL=finalNLL,
            maxIters=maxIters,
        )
    else:
        normalized = {
            str(key): _diagnosticScalar(value) for key, value in diagnostics.items()
        }
        normalized.setdefault("iters_done", int(itersDone))
        normalized.setdefault("max_iters", int(maxIters))
        normalized.setdefault("final_nll", metadataFloat(float(finalNLL)))
        normalized.setdefault("diagnostics_source", "cfixedBackgroundECM")
    normalized["outer_pass"] = int(outerPass)
    return normalized


def _fitDiagnosticsMetadata(fit: Mapping[str, Any]) -> dict[str, Any]:
    ecmDiagnostics = [
        {str(key): _diagnosticScalar(value) for key, value in dict(item).items()}
        for item in fit.get("fixedBackgroundECMDiagnostics", [])
        if isinstance(item, Mapping)
    ]
    convergedValues = [
        bool(item.get("converged"))
        for item in ecmDiagnostics
        if item.get("converged") is not None
    ]
    increaseValues = [
        int(item.get("nll_increase_count"))
        for item in ecmDiagnostics
        if item.get("nll_increase_count") is not None
    ]
    return {
        "requested_outer_passes": int(fit.get("requestedOuterIters", 0) or 0),
        "min_outer_passes": int(fit.get("minOuterIters", 0) or 0),
        "planned_outer_passes": int(fit.get("plannedOuterPasses", 0) or 0),
        "actual_outer_passes": int(fit.get("actualOuterPasses", 0) or 0),
        "outer_converged": bool(fit.get("outerConverged", False)),
        "outer_stop_reason": str(fit.get("outerStopReason") or "unknown"),
        "background_shift": metadataFloat(float(fit.get("backgroundShift", np.nan))),
        "background_shift_threshold": fit.get("backgroundShiftThreshold"),
        "outer_nll": fit.get("outerNLL"),
        "outer_nll_change": fit.get("outerNLLChange"),
        "outer_nll_threshold": fit.get("outerNLLThreshold"),
        "outer_nll_stable": bool(fit.get("outerNLLStable", False)),
        "outer_objective": fit.get("outerObjective"),
        "outer_objective_per_cell": fit.get("outerObjectivePerCell"),
        "outer_objective_change_per_cell": fit.get("outerObjectiveChangePerCell"),
        "outer_objective_threshold_per_cell": fit.get("outerObjectiveThresholdPerCell"),
        "outer_objective_stable": bool(fit.get("outerObjectiveStable", False)),
        "outer_effective_observation_count": int(
            fit.get("outerEffectiveObservationCount", 0) or 0
        ),
        "outer_stable_iters": int(fit.get("outerStableIters", 0) or 0),
        "outer_patience_target": int(fit.get("outerPatienceTarget", 0) or 0),
        "inner_ecm_converged": bool(fit.get("innerECMConverged", False)),
        "warm_start": dict(fit.get("warmStart", {}) or {}),
        "all_ecm_converged": (
            bool(convergedValues) and all(convergedValues) if convergedValues else None
        ),
        "max_nll_increase_count": max(increaseValues) if increaseValues else None,
        "fixed_background_ecm": ecmDiagnostics,
    }


def _computeExpectedTransitionResidualSums(
    stateSmoothed: np.ndarray,
    stateCovarSmoothed: np.ndarray,
    lagCovSmoothed: np.ndarray,
    matrixF: np.ndarray,
) -> tuple[float, float, int]:
    r"""Compute expected squared state-transition residual sums.

    ``lagCovSmoothed[k]`` is interpreted as
    :math:`\mathrm{Cov}(x_k, x_{k+1}\mid y)`, matching
    :func:`consenrich.cconsenrich.cbackwardPass`.
    """

    m = np.asarray(stateSmoothed, dtype=np.float64)
    P = np.asarray(stateCovarSmoothed, dtype=np.float64)
    C = np.asarray(lagCovSmoothed, dtype=np.float64)
    F = np.asarray(matrixF, dtype=np.float64)

    if m.ndim != 2 or m.shape[1] != 2:
        raise ValueError("stateSmoothed must have shape (n, 2)")
    if P.shape != (m.shape[0], 2, 2):
        raise ValueError("stateCovarSmoothed must have shape (n, 2, 2)")
    if C.shape[0] < max(m.shape[0] - 1, 0) or C.shape[1:] != (2, 2):
        raise ValueError("lagCovSmoothed must have shape (n - 1, 2, 2)")
    if F.shape != (2, 2):
        raise ValueError("matrixF must have shape (2, 2)")

    transitionCount = int(m.shape[0] - 1)
    if transitionCount <= 0:
        return 0.0, 0.0, 0

    sumLevel, sumTrend, cTransitionCount = cconsenrich.cExpectedTransitionResidualSums(
        np.ascontiguousarray(m, dtype=np.float64),
        np.ascontiguousarray(P, dtype=np.float64),
        np.ascontiguousarray(C, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.float64),
    )
    return float(sumLevel), float(sumTrend), int(cTransitionCount)


def _estimateRegularizedDiagonalProcessQ(
    *,
    stateSmoothed: np.ndarray,
    stateCovarSmoothed: np.ndarray,
    lagCovSmoothed: np.ndarray,
    matrixF: np.ndarray,
    minQ: float,
    maxQ: float,
    processQLevelTarget: float | None,
    processQTrendTarget: float | None,
    processQLevelPriorWeight: float,
    processQTrendPriorWeight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    qLevelTarget = (
        max(float(minQ), PROCESS_Q_NUMERICAL_FLOOR)
        if processQLevelTarget is None
        else _checkFinitePositive("processQLevelTarget", processQLevelTarget)
    )
    qTrendTarget = (
        qLevelTarget * PROCESS_Q_DEFAULT_TREND_TARGET_RATIO
        if processQTrendTarget is None
        else _checkFinitePositive("processQTrendTarget", processQTrendTarget)
    )
    levelWeight = _checkFiniteNonnegative(
        "processQLevelPriorWeight",
        processQLevelPriorWeight,
    )
    trendWeight = _checkFiniteNonnegative(
        "processQTrendPriorWeight",
        processQTrendPriorWeight,
    )

    sumLevel, sumTrend, transitionCount = _computeExpectedTransitionResidualSums(
        stateSmoothed=stateSmoothed,
        stateCovarSmoothed=stateCovarSmoothed,
        lagCovSmoothed=lagCovSmoothed,
        matrixF=matrixF,
    )
    if transitionCount <= 0:
        raise ValueError("need at least one transition for process-Q calibration")

    rawLevel = sumLevel / float(transitionCount)
    rawTrend = sumTrend / float(transitionCount)
    qLevelShrunk = (rawLevel + levelWeight * qLevelTarget) / (1.0 + levelWeight)
    qTrendShrunk = (rawTrend + trendWeight * qTrendTarget) / (1.0 + trendWeight)

    levelFloor = max(float(minQ), PROCESS_Q_NUMERICAL_FLOOR)
    trendFloor = max(
        qTrendTarget * PROCESS_Q_TREND_FLOOR_RATIO,
        PROCESS_Q_NUMERICAL_FLOOR,
    )
    qLevel = float(qLevelShrunk)
    qTrend = float(qTrendShrunk)
    levelFloorHit = bool(qLevel <= levelFloor)
    trendFloorHit = bool(qTrend <= trendFloor)
    qLevel = max(float(qLevel), levelFloor)
    qTrend = max(float(qTrend), trendFloor)

    levelCapHit = False
    trendCapHit = False
    if np.isfinite(float(maxQ)) and float(maxQ) > 0.0:
        qLevelCap = max(float(maxQ), levelFloor)
        qTrendCap = max(float(maxQ), trendFloor)
        levelCapHit = bool(qLevel >= qLevelCap)
        trendCapHit = bool(qTrend >= qTrendCap)
        qLevel = min(qLevel, qLevelCap)
        qTrend = min(qTrend, qTrendCap)
    rawLevelDenom = max(float(rawLevel), PROCESS_Q_NUMERICAL_FLOOR)
    rawTrendDenom = max(float(rawTrend), PROCESS_Q_NUMERICAL_FLOOR)

    matrixQ = constructMatrixQ(
        minDiagQ=qLevel,
        offDiagQ=0.0,
        Q00=qLevel,
        Q11=qTrend,
    ).astype(np.float32, copy=False)
    return matrixQ, {
        "q_level": float(qLevel),
        "q_trend": float(qTrend),
        "raw_q_level": float(rawLevel),
        "raw_q_trend": float(rawTrend),
        "q_level_target": float(qLevelTarget),
        "q_trend_target": float(qTrendTarget),
        "q_level_prior_weight": float(levelWeight),
        "q_trend_prior_weight": float(trendWeight),
        "q_level_shrunk": float(qLevelShrunk),
        "q_trend_shrunk": float(qTrendShrunk),
        "q_level_floor": float(levelFloor),
        "q_trend_floor": float(trendFloor),
        "q_level_floor_hit": float(levelFloorHit),
        "q_trend_floor_hit": float(trendFloorHit),
        "q_level_cap_hit": float(levelCapHit),
        "q_trend_cap_hit": float(trendCapHit),
        "q_level_final_raw_ratio": float(qLevel / rawLevelDenom),
        "q_trend_final_raw_ratio": float(qTrend / rawTrendDenom),
        "transition_count": float(transitionCount),
    }


def _warnIfProcessQCalibrationForced(info: Mapping[str, float]) -> None:
    warnings_: list[str] = []
    for component in ("level", "trend"):
        if float(info.get(f"q_{component}_floor_hit", 0.0) or 0.0) > 0.0:
            warnings_.append(f"{component} hit floor")
        if float(info.get(f"q_{component}_cap_hit", 0.0) or 0.0) > 0.0:
            warnings_.append(f"{component} hit cap")
        ratio = float(info.get(f"q_{component}_final_raw_ratio", np.nan))
        if np.isfinite(ratio) and (ratio < 0.1 or ratio > 10.0):
            warnings_.append(f"{component} final/raw ratio={ratio:.6g}")
    if warnings_:
        logger.warning(
            "processQCalibration.forced %s; q_level=%.6g raw_level=%.6g "
            "q_trend=%.6g raw_trend=%.6g",
            "; ".join(warnings_),
            float(info.get("q_level", np.nan)),
            float(info.get("raw_q_level", np.nan)),
            float(info.get("q_trend", np.nan)),
            float(info.get("raw_q_trend", np.nan)),
        )


def constructMatrixQ(
    minDiagQ: float,
    offDiagQ: float = 0.0,
    Q00: Optional[float] = None,
    Q01: Optional[float] = None,
    Q10: Optional[float] = None,
    Q11: Optional[float] = None,
    useIdentity: float = -1.0,
    tol: float = 1.0e-8,  # conservative
) -> npt.NDArray[np.float32]:
    r"""Build the (base) process noise covariance matrix :math:`\mathbf{Q}`.

    :param minDiagQ: Minimum value for diagonal entries of :math:`\mathbf{Q}`.
    :type minDiagQ: float
    :param offDiagQ: Value for off-diagonal entries of :math:`\mathbf{Q}`.
    :type offDiagQ: float
    :param Q00: Optional value for entry (0,0) of :math:`\mathbf{Q}`.
    :type Q00: Optional[float]
    :param Q01: Optional value for entry (0,1) of :math:`\mathbf{Q}`.
    :type Q01: Optional[float]
    :param Q10: Optional value for entry (1,0) of :math:`\mathbf{Q}`.
    :type Q10: Optional[float]
    :param Q11: Optional value for entry (1,1) of :math:`\mathbf{Q}`.
    :type Q11: Optional[float]
    :param useIdentity: If > 0.0, use a scaled identity matrix for :math:`\mathbf{Q}`.
        Overrides other parameters.
    :type useIdentity: float
    :param tol: Tolerance for positive definiteness check.
    :type tol: float
    :return: The process noise covariance matrix :math:`\mathbf{Q}`.
    :rtype: npt.NDArray[np.float32]

    :seealso: :class:`processParams`
    """

    if useIdentity > 0.0:
        return np.eye(2, dtype=np.float32) * np.float32(useIdentity)

    Q = np.empty((2, 2), dtype=np.float32)

    Q[0, 0] = np.float32(minDiagQ if Q00 is None else Q00)
    Q[1, 1] = np.float32(minDiagQ if Q11 is None else Q11)

    if Q11 is None:
        Q[1, 1] = Q[0, 0]

    if Q01 is not None and Q10 is None:
        Q10 = Q01
    elif Q10 is not None and Q01 is None:
        Q01 = Q10

    Q[0, 1] = np.float32(offDiagQ if Q01 is None else Q01)
    Q[1, 0] = np.float32(offDiagQ if Q10 is None else Q10)

    if not np.allclose(Q[0, 1], Q[1, 0], rtol=0.0, atol=1e-4):
        raise ValueError(f"Matrix is not symmetric: Q=\n{Q}")

    maxNoiseCorr = np.float32(0.99)
    maxOffDiag = maxNoiseCorr * np.sqrt(Q[0, 0] * Q[1, 1]).astype(np.float32)
    Q[0, 1] = np.clip(Q[0, 1], -maxOffDiag, maxOffDiag)
    Q[1, 0] = Q[0, 1]

    try:
        np.linalg.cholesky(Q.astype(np.float64, copy=False) + tol * np.eye(2))
    except Exception as ex:
        raise ValueError(
            f"Process noise covariance Q is not positive definite:\n{Q}"
        ) from ex
    return Q


def runConsenrich(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    deltaF: float,
    minQ: float,
    maxQ: float,
    offDiagQ: float,
    stateInit: float,
    stateCovarInit: float,
    boundState: bool,
    stateLowerBound: float,
    stateUpperBound: float,
    blockLenIntervals: int,
    projectStateDuringFiltering: bool = False,
    pad: float = 1.0e-4,
    ECM_fixedBackgroundIters: int = 50,
    ECM_fixedBackgroundRtol: float = 1.0e-4,
    ECM_robustTNu: float = 8.0,
    ECM_useObsPrecisionReweighting: bool = True,
    ECM_useProcessPrecisionReweighting: bool = True,
    ECM_useAPN: bool = False,
    ECM_zeroCenterBackground: bool = False,
    ECM_outerIters: int = 3,
    ECM_minOuterIters: int | None = None,
    ECM_backgroundShiftRtol: float = 1.0e-3,
    ECM_outerNLLRtol: float = 1.0e-4,
    ECM_backgroundSmoothness: float = 1.0,
    fitBackground: bool = True,
    returnScales: bool = True,
    returnReplicateOffsets: bool = False,
    processQCalibration: str | None = PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL,
    processQWarmupECMIters: int = 3,
    processQWarmupOuterIters: int | None = PROCESS_Q_CALIBRATION_DEFAULT_OUTER_ITERS,
    processQLevelTarget: float | None = None,
    processQTrendTarget: float | None = None,
    processQLevelPriorWeight: float = 0.05,
    processQTrendPriorWeight: float = 1.0,
    observationPrecisionMultiplierMin: float = 0.25,
    observationPrecisionMultiplierMax: float = 4.0,
    processPrecisionMultiplierMin: float = 0.5,
    processPrecisionMultiplierMax: float = 2.0,
    observationMask: np.ndarray | None = None,
    initialBackground: np.ndarray | None = None,
    initialReplicateBias: np.ndarray | None = None,
    initialObservationPrecision: np.ndarray | None = None,
    initialProcessPrecision: np.ndarray | None = None,
    initialProcessQ: np.ndarray | None = None,
    returnDiagnostics: bool = False,
    logIndentLevel: int = 0,
    logRunRole: str | None = None,
):
    r"""Run Consenrich over a contiguous genomic region

    Consenrich estimates a shared signal level from multiple replicate tracks using a two-state
    linear smoother plus fixed-background ECM and an outer fit/background
    alternation loop.

    The observation model is

    .. math::

      z_{[j,i]} = g_{[i]} + x_{[i,0]} + b_j + \epsilon_{[j,i]},
      \qquad
      \mathrm{Var}(\epsilon_{[j,i]}) =
      \frac{v_{[j,i]} + \mathrm{pad}}{\lambda_{[i]}}.

    Here :math:`z_{[j,i]}` is the observed track value, :math:`g_{[i]}` is an
    optional low-frequency background shared across replicates, :math:`b_j` is
    a replicate-level bias term, and :math:`v_{[j,i]}` is the plugin
    observation variance supplied by ``matrixMunc``. Note, by default, replicate offsets
    are centered to zero for identifiability while the shared background is allowed to
    carry a contig-wide level.

    The latent state follows

    .. math::

      \mathbf{x}_{[i+1]} = \mathbf{F}(\delta_F)\mathbf{x}_{[i]} + \eta_{[i]},
      \qquad
      \mathrm{Var}(\eta_{[i]}) = \frac{\mathbf{Q}_0}{\kappa_{[i]}}.

    If ``ECM_useAPN=True``, the forward filter instead uses the
    adaptive-process-noise D-statistic update to scale
    :math:`\mathbf{Q}_0`; process-precision reweighting is disabled, and the
    monotonic-descent guarantee no longer applies.

    This wrapper ties together several fundamental routines written in Cython:

    #. :func:`consenrich.cconsenrich.cforwardPass`: Forward filter (predict, update)
    #. :func:`consenrich.cconsenrich.cbackwardPass`: Backward fixed-interval smoother
    #. :func:`consenrich.cconsenrich.cfixedBackgroundECM`: Run ECM to convergence wrt a fixed :math:`g`.

    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.cconsenrich.cTransform`, :func:`consenrich.cconsenrich.cforwardPass`, :func:`consenrich.cconsenrich.cbackwardPass`, :func:`consenrich.cconsenrich.cfixedBackgroundECM`
    """

    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.ascontiguousarray(matrixMunc, dtype=np.float32)

    if matrixData.ndim == 1:
        matrixData = matrixData[None, :]
    elif matrixData.ndim != 2:
        raise ValueError(f"matrixData must be 1D or 2D (got ndim={matrixData.ndim})")

    if matrixMunc.ndim == 1:
        matrixMunc = matrixMunc[None, :]
    elif matrixMunc.ndim != 2:
        raise ValueError(f"matrixMunc must be 1D or 2D (got ndim={matrixMunc.ndim})")

    if matrixData.shape != matrixMunc.shape:
        raise ValueError("matrixData and matrixMunc must have identical shapes")

    if observationMask is not None:
        observationMaskArr = np.asarray(observationMask, dtype=bool)
        if observationMaskArr.ndim == 1:
            observationMaskArr = np.broadcast_to(
                observationMaskArr[None, :],
                matrixData.shape,
            )
        if observationMaskArr.shape != matrixData.shape:
            raise ValueError("observationMask must match matrixData shape")
        matrixMunc = np.asarray(matrixMunc, dtype=np.float32).copy(order="C")
        matrixMunc[~observationMaskArr] = (
            UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE
        )
        matrixMunc = np.ascontiguousarray(matrixMunc, dtype=np.float32)

    trackCount, intervalCount = matrixData.shape
    if intervalCount < 2:
        raise ValueError("need at least 2 intervals for smoothing")

    ECM_useAPN = bool(ECM_useAPN)
    if ECM_useAPN:
        ECM_useProcessPrecisionReweighting = False
    (
        observationPrecisionMultiplierMin,
        observationPrecisionMultiplierMax,
    ) = _checkPrecisionMultiplierBounds(
        "observation",
        observationPrecisionMultiplierMin,
        observationPrecisionMultiplierMax,
    )
    (
        processPrecisionMultiplierMin,
        processPrecisionMultiplierMax,
    ) = _checkPrecisionMultiplierBounds(
        "process",
        processPrecisionMultiplierMin,
        processPrecisionMultiplierMax,
    )
    initialBackgroundArr = _coerceOptionalVector(
        "initialBackground",
        initialBackground,
        intervalCount,
    )
    initialReplicateBiasArr = _coerceOptionalVector(
        "initialReplicateBias",
        initialReplicateBias,
        trackCount,
    )
    initialObservationPrecisionArr = _coerceOptionalVector(
        "initialObservationPrecision",
        initialObservationPrecision,
        intervalCount,
    )
    if initialObservationPrecisionArr is not None:
        initialObservationPrecisionArr = np.ascontiguousarray(
            np.clip(
                initialObservationPrecisionArr,
                float(observationPrecisionMultiplierMin),
                float(observationPrecisionMultiplierMax),
            ),
            dtype=np.float32,
        )
    initialProcessPrecisionArr = _coerceOptionalVector(
        "initialProcessPrecision",
        initialProcessPrecision,
        intervalCount,
    )
    if initialProcessPrecisionArr is not None:
        initialProcessPrecisionArr = np.ascontiguousarray(
            np.clip(
                initialProcessPrecisionArr,
                float(processPrecisionMultiplierMin),
                float(processPrecisionMultiplierMax),
            ),
            dtype=np.float32,
        )
    initialProcessQArr = _coerceOptionalProcessQ(initialProcessQ)
    processQCalibrationMode = _normalizeProcessQCalibration(processQCalibration)
    ECM_outerIters = max(1, int(ECM_outerIters))
    ECM_minOuterIters = (
        3 if ECM_minOuterIters is None else max(1, int(ECM_minOuterIters))
    )
    ECM_backgroundShiftRtol = float(max(ECM_backgroundShiftRtol, 0.0))
    ECM_outerNLLRtol = float(max(ECM_outerNLLRtol, 0.0))
    logIndentLevel = max(0, int(logIndentLevel or 0))
    logRunRole = str(logRunRole or "").strip()

    blockCount = int(np.ceil(intervalCount / float(blockLenIntervals)))
    intervalToBlockMap = (
        np.arange(intervalCount, dtype=np.int32) // blockLenIntervals
    ).astype(np.int32)
    intervalToBlockMap[intervalToBlockMap >= blockCount] = blockCount - 1
    totalStart = time.perf_counter()
    _logAsciiBlock(
        "core start",
        (
            ("tracks", int(trackCount)),
            ("intervals", int(intervalCount)),
            ("blocks", int(blockCount)),
            ("ECM max iterations", int(ECM_fixedBackgroundIters)),
            ("outer passes", int(ECM_outerIters)),
            ("process Q calibration", processQCalibrationMode),
            ("background model fit", bool(fitBackground)),
        ),
        indentLevel=logIndentLevel,
    )
    logger.info(
        "runConsenrich.core.start tracks=%d intervals=%d blocks=%d ECM_fixedBackgroundIters=%d outerIters=%d processQCalibration=%s",
        int(trackCount),
        int(intervalCount),
        int(blockCount),
        int(ECM_fixedBackgroundIters),
        int(ECM_outerIters),
        processQCalibrationMode,
    )
    logger.info(
        "precisionMultiplierBounds: obs=[%.6g, %.6g] proc=[%.6g, %.6g]",
        observationPrecisionMultiplierMin,
        observationPrecisionMultiplierMax,
        processPrecisionMultiplierMin,
        processPrecisionMultiplierMax,
    )

    # keep the transition matrix step-dependent while using an explicit base Q
    def buildMatrixF(deltaFLocal: float) -> np.ndarray:
        return constructMatrixF(float(deltaFLocal)).astype(np.float32, copy=False)

    def buildMatrixQ0(_deltaFLocal: float) -> np.ndarray:
        return constructMatrixQ(
            minDiagQ=float(minQ),
            offDiagQ=float(offDiagQ),
        ).astype(np.float32, copy=False)

    def _runForwardBackward(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        lambdaExp: np.ndarray | None,
        processPrecExp: np.ndarray | None,
        replicateBias: np.ndarray | None,
        useProcPrecReweightLocal: bool,
        useAPNLocal: bool,
    ):
        stateForward = np.empty((intervalCount, 2), dtype=np.float32)
        stateCovarForward = np.empty((intervalCount, 2, 2), dtype=np.float32)
        pNoiseForward = np.empty((intervalCount, 2, 2), dtype=np.float32)
        vectorD = np.empty(intervalCount, dtype=np.float32)

        phiHat, _, vectorD, sumNLL = cconsenrich.cforwardPass(
            matrixData=matrixDataLocal,
            matrixPluginMuncInit=matrixMuncLocal,
            matrixF=matrixFLocal,
            matrixQ0=matrixQ0Local,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=int(blockCount),
            stateInit=float(stateInit),
            stateCovarInit=float(stateCovarInit),
            pad=float(pad),
            projectStateDuringFiltering=bool(projectStateDuringFiltering),
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            chunkSize=0,
            stateForward=stateForward,
            stateCovarForward=stateCovarForward,
            pNoiseForward=pNoiseForward,
            vectorD=vectorD,
            progressBar=None,
            progressIter=0,
            returnNLL=True,
            storeNLLInD=False,
            lambdaExp=lambdaExp,
            processPrecExp=processPrecExp,
            replicateBias=replicateBias,
            ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
            ECM_useProcessPrecisionReweighting=bool(useProcPrecReweightLocal),
            ECM_useAPN=bool(useAPNLocal),
            obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
            obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
            procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
            procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
            APN_minQ=float(minQ),
            APN_maxQ=float(maxQ),
        )

        stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals = (
            cconsenrich.cbackwardPass(
                matrixData=matrixDataLocal,
                matrixF=matrixFLocal,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                chunkSize=0,
                stateSmoothed=None,
                stateCovarSmoothed=None,
                lagCovSmoothed=None,
                postFitResiduals=None,
                replicateBias=replicateBias,
                progressBar=None,
                progressIter=0,
            )
        )

        NIS = vectorD.astype(np.float32, copy=False)
        return (
            phiHat,
            sumNLL,
            stateForward,
            stateCovarForward,
            pNoiseForward,
            stateSmoothed,
            stateCovarSmoothed,
            lagCovSmoothed,
            postFitResiduals,
            NIS,
        )

    def _scoreForwardNLL(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        lambdaExp: np.ndarray | None,
        processPrecExp: np.ndarray | None,
        replicateBias: np.ndarray | None,
        useProcPrecReweightLocal: bool,
        useAPNLocal: bool,
    ) -> float:
        _phiHat, _unused, _vectorD, sumNLL = cconsenrich.cforwardPass(
            matrixData=matrixDataLocal,
            matrixPluginMuncInit=matrixMuncLocal,
            matrixF=matrixFLocal,
            matrixQ0=matrixQ0Local,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=int(blockCount),
            stateInit=float(stateInit),
            stateCovarInit=float(stateCovarInit),
            pad=float(pad),
            projectStateDuringFiltering=bool(projectStateDuringFiltering),
            stateLowerBound=0.0,
            stateUpperBound=0.0,
            chunkSize=0,
            stateForward=None,
            stateCovarForward=None,
            pNoiseForward=None,
            vectorD=None,
            progressBar=None,
            progressIter=0,
            returnNLL=True,
            storeNLLInD=False,
            lambdaExp=lambdaExp,
            processPrecExp=processPrecExp,
            replicateBias=replicateBias,
            ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
            ECM_useProcessPrecisionReweighting=bool(useProcPrecReweightLocal),
            ECM_useAPN=bool(useAPNLocal),
            obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
            obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
            procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
            procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
            APN_minQ=float(minQ),
            APN_maxQ=float(maxQ),
        )
        return float(sumNLL)

    def _scorePenalizedObjective(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        background: np.ndarray,
        lambdaExp: np.ndarray | None,
        processPrecExp: np.ndarray | None,
        replicateBias: np.ndarray | None,
        useProcPrecReweightLocal: bool,
        useAPNLocal: bool,
    ) -> dict[str, float]:
        dataAdjusted = np.ascontiguousarray(
            matrixDataLocal - np.asarray(background, dtype=np.float32)[None, :],
            dtype=np.float32,
        )
        forwardNLL = _scoreForwardNLL(
            matrixDataLocal=dataAdjusted,
            matrixMuncLocal=matrixMuncLocal,
            matrixFLocal=matrixFLocal,
            matrixQ0Local=matrixQ0Local,
            lambdaExp=lambdaExp,
            processPrecExp=(
                processPrecExp
                if bool(useProcPrecReweightLocal) and not bool(useAPNLocal)
                else None
            ),
            replicateBias=replicateBias,
            useProcPrecReweightLocal=useProcPrecReweightLocal,
            useAPNLocal=useAPNLocal,
        )
        obsPenalty, procPenalty = _robustPrecisionPenalty(
            lambdaExp=lambdaExp if bool(ECM_useObsPrecisionReweighting) else None,
            processPrecExp=(
                processPrecExp
                if bool(useProcPrecReweightLocal) and not bool(useAPNLocal)
                else None
            ),
            robustTNu=float(ECM_robustTNu),
        )
        smoothPenalty, firstDiffPenalty, secondDiffPenalty = _backgroundObjectivePenalty(
            background=background,
            blockLenIntervals=int(blockLenIntervals),
            backgroundSmoothness=float(ECM_backgroundSmoothness),
        )
        objective = float(
            forwardNLL + obsPenalty + procPenalty + smoothPenalty
        )
        effectiveCount = float(_effectiveObservationCount(matrixMuncLocal))
        return {
            "forward_nll": float(forwardNLL),
            "robust_observation_penalty": float(obsPenalty),
            "robust_process_penalty": float(procPenalty),
            "background_smoothness_penalty": float(smoothPenalty),
            "background_first_difference_penalty": float(firstDiffPenalty),
            "background_second_difference_penalty": float(secondDiffPenalty),
            "penalized_objective": float(objective),
            "penalized_objective_per_cell": float(objective / effectiveCount),
            "effective_observation_count": float(effectiveCount),
        }

    def _fitOuter(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        ecmItersLocal: int,
        ecmRtolLocal: float,
        outerItersLocal: int | None = None,
        minOuterItersLocal: int | None = None,
        useProcPrecReweightOverride: bool | None = None,
        useAPNOverride: bool | None = None,
        initialBackgroundLocal: np.ndarray | None = None,
        initialReplicateBiasLocal: np.ndarray | None = None,
        initialLambdaLocal: np.ndarray | None = None,
        initialProcessPrecLocal: np.ndarray | None = None,
        phaseLabel: str = "fit",
        phaseIndentLevel: int = 0,
    ) -> dict[str, np.ndarray | float | None]:
        mLocal = int(matrixDataLocal.shape[0])
        nLocal = int(matrixDataLocal.shape[1])
        useAPNLocal = bool(ECM_useAPN if useAPNOverride is None else useAPNOverride)
        useProcPrecLocal = bool(
            ECM_useProcessPrecisionReweighting
            if useProcPrecReweightOverride is None
            else useProcPrecReweightOverride
        )
        if useAPNLocal:
            useProcPrecLocal = False

        currentBackground = (
            np.zeros(nLocal, dtype=np.float32)
            if initialBackgroundLocal is None
            else np.ascontiguousarray(initialBackgroundLocal, dtype=np.float32).copy()
        )
        currentMunc = np.ascontiguousarray(matrixMuncLocal, dtype=np.float32)

        lambdaExpLocal = (
            np.ascontiguousarray(initialLambdaLocal, dtype=np.float32).copy()
            if initialLambdaLocal is not None and bool(ECM_useObsPrecisionReweighting)
            else None
        )
        processPrecExpLocal = (
            np.ascontiguousarray(initialProcessPrecLocal, dtype=np.float32).copy()
            if initialProcessPrecLocal is not None
            and bool(useProcPrecLocal)
            and not bool(useAPNLocal)
            else None
        )
        replicateBiasLocal = (
            np.zeros(mLocal, dtype=np.float32)
            if initialReplicateBiasLocal is None
            else np.ascontiguousarray(
                initialReplicateBiasLocal, dtype=np.float32
            ).copy()
        )
        warmStartSummaryLocal = {
            "background": bool(initialBackgroundLocal is not None),
            "replicate_bias": bool(initialReplicateBiasLocal is not None),
            "observation_precision": bool(lambdaExpLocal is not None),
            "process_precision": bool(processPrecExpLocal is not None),
        }
        stateSmoothedLocal = None
        stateCovarSmoothedLocal = None
        lagCovSmoothedLocal = None
        postFitResidualsLocal = None
        NISLocal = None
        sumNLLLocal = np.nan
        lastBackgroundShiftLocal = 0.0

        fitBackgroundLocal = bool(fitBackground)
        if fitBackgroundLocal:
            requestedOuterIters = max(
                1,
                int(ECM_outerIters if outerItersLocal is None else outerItersLocal),
            )
            minOuterIters = (
                int(ECM_minOuterIters)
                if minOuterItersLocal is None
                else max(1, int(minOuterItersLocal))
            )
            outerPassCount = max(
                minOuterIters,
                requestedOuterIters,
            )
        else:
            outerPassCount = 1
            minOuterIters = 1
            requestedOuterIters = 1
        backgroundShiftTolMultiplier = float(ECM_backgroundShiftRtol)
        outerObjectiveTolMultiplier = float(ECM_outerNLLRtol)
        actualOuterPasses = 0
        previousOuterNLLLocal = np.nan
        previousOuterObjectivePerCellLocal = np.nan
        lastOuterNLLLocal = np.nan
        lastOuterNLLChangeLocal = np.nan
        lastOuterNLLTolLocal = np.nan
        lastOuterNLLStableLocal = False
        lastOuterObjectiveLocal = np.nan
        lastOuterObjectivePerCellLocal = np.nan
        lastOuterObjectiveChangePerCellLocal = np.nan
        lastOuterObjectiveTolPerCellLocal = np.nan
        lastOuterObjectiveStableLocal = False
        lastObjectiveDiagnosticsLocal: dict[str, float] = {}
        lastInnerECMConvergedLocal = False
        outerStableItersLocal = 0
        outerPatienceTargetLocal = 2
        lastBackgroundShiftTolLocal = np.nan
        outerConvergedLocal = False
        outerStopReasonLocal = "max_outer_passes"
        fixedBackgroundECMDiagnostics: list[dict[str, Any]] = []

        def _recordOuterObjective(
            ecmDiagnosticsNormalizedLocal: dict[str, Any],
            *,
            ecmFitNLL: float,
        ) -> None:
            nonlocal previousOuterNLLLocal
            nonlocal previousOuterObjectivePerCellLocal
            nonlocal lastOuterNLLLocal
            nonlocal lastOuterNLLChangeLocal
            nonlocal lastOuterNLLTolLocal
            nonlocal lastOuterNLLStableLocal
            nonlocal lastOuterObjectiveLocal
            nonlocal lastOuterObjectivePerCellLocal
            nonlocal lastOuterObjectiveChangePerCellLocal
            nonlocal lastOuterObjectiveTolPerCellLocal
            nonlocal lastOuterObjectiveStableLocal
            nonlocal lastObjectiveDiagnosticsLocal

            lastObjectiveDiagnosticsLocal = _scorePenalizedObjective(
                matrixDataLocal=matrixDataLocal,
                matrixMuncLocal=currentMunc,
                matrixFLocal=matrixFLocal,
                matrixQ0Local=matrixQ0Local,
                background=currentBackground,
                lambdaExp=lambdaExpLocal,
                processPrecExp=processPrecExpLocal,
                replicateBias=replicateBiasLocal,
                useProcPrecReweightLocal=useProcPrecLocal,
                useAPNLocal=useAPNLocal,
            )

            currentForwardNLL = float(lastObjectiveDiagnosticsLocal["forward_nll"])
            if np.isfinite(previousOuterNLLLocal) and np.isfinite(currentForwardNLL):
                lastOuterNLLChangeLocal = abs(currentForwardNLL - previousOuterNLLLocal)
                lastOuterNLLTolLocal = outerObjectiveTolMultiplier * max(
                    abs(currentForwardNLL),
                    abs(previousOuterNLLLocal),
                    1.0,
                )
                lastOuterNLLStableLocal = bool(
                    lastOuterNLLChangeLocal <= lastOuterNLLTolLocal
                )
            else:
                lastOuterNLLChangeLocal = np.nan
                lastOuterNLLTolLocal = np.nan
                lastOuterNLLStableLocal = False
            previousOuterNLLLocal = currentForwardNLL
            lastOuterNLLLocal = currentForwardNLL

            currentObjective = float(
                lastObjectiveDiagnosticsLocal["penalized_objective"]
            )
            currentObjectivePerCell = float(
                lastObjectiveDiagnosticsLocal["penalized_objective_per_cell"]
            )
            if np.isfinite(previousOuterObjectivePerCellLocal) and np.isfinite(
                currentObjectivePerCell
            ):
                lastOuterObjectiveChangePerCellLocal = abs(
                    currentObjectivePerCell - previousOuterObjectivePerCellLocal
                )
                lastOuterObjectiveTolPerCellLocal = outerObjectiveTolMultiplier * max(
                    abs(currentObjectivePerCell),
                    abs(previousOuterObjectivePerCellLocal),
                    1.0,
                )
                lastOuterObjectiveStableLocal = bool(
                    lastOuterObjectiveChangePerCellLocal
                    <= lastOuterObjectiveTolPerCellLocal
                )
            else:
                lastOuterObjectiveChangePerCellLocal = np.nan
                lastOuterObjectiveTolPerCellLocal = np.nan
                lastOuterObjectiveStableLocal = False
            previousOuterObjectivePerCellLocal = currentObjectivePerCell
            lastOuterObjectiveLocal = currentObjective
            lastOuterObjectivePerCellLocal = currentObjectivePerCell

            ecmDiagnosticsNormalizedLocal.update(
                {
                    "outer_ecm_fit_nll": metadataFloat(float(ecmFitNLL)),
                    "outer_forward_nll": metadataFloat(lastOuterNLLLocal),
                    "outer_nll_change": metadataFloat(lastOuterNLLChangeLocal),
                    "outer_nll_threshold": metadataFloat(lastOuterNLLTolLocal),
                    "outer_nll_stable": bool(lastOuterNLLStableLocal),
                    "outer_objective": metadataFloat(lastOuterObjectiveLocal),
                    "outer_objective_per_cell": metadataFloat(
                        lastOuterObjectivePerCellLocal
                    ),
                    "outer_objective_change_per_cell": metadataFloat(
                        lastOuterObjectiveChangePerCellLocal
                    ),
                    "outer_objective_threshold_per_cell": metadataFloat(
                        lastOuterObjectiveTolPerCellLocal
                    ),
                    "outer_objective_stable": bool(lastOuterObjectiveStableLocal),
                    "outer_effective_observation_count": int(
                        lastObjectiveDiagnosticsLocal["effective_observation_count"]
                    ),
                    "outer_robust_observation_penalty": metadataFloat(
                        lastObjectiveDiagnosticsLocal["robust_observation_penalty"]
                    ),
                    "outer_robust_process_penalty": metadataFloat(
                        lastObjectiveDiagnosticsLocal["robust_process_penalty"]
                    ),
                    "outer_background_smoothness_penalty": metadataFloat(
                        lastObjectiveDiagnosticsLocal["background_smoothness_penalty"]
                    ),
                    "outer_background_first_difference_penalty": metadataFloat(
                        lastObjectiveDiagnosticsLocal[
                            "background_first_difference_penalty"
                        ]
                    ),
                    "outer_background_second_difference_penalty": metadataFloat(
                        lastObjectiveDiagnosticsLocal[
                            "background_second_difference_penalty"
                        ]
                    ),
                }
            )

        for outerPassIndex in range(outerPassCount):
            _logAsciiBlock(
                f"{phaseLabel} / fixed-background ECM",
                (
                    (
                        "outer pass",
                        f"{int(outerPassIndex + 1)}/{int(outerPassCount)}",
                    ),
                    ("tracks", int(mLocal)),
                    ("intervals", int(nLocal)),
                    ("ECM max iterations", int(ecmItersLocal)),
                    ("ECM rtol", float(ecmRtolLocal)),
                    ("background model fit", bool(fitBackgroundLocal)),
                    ("adaptive process noise", bool(useAPNLocal)),
                    ("obs precision weights", bool(ECM_useObsPrecisionReweighting)),
                    ("process precision weights", bool(useProcPrecLocal)),
                ),
                indentLevel=phaseIndentLevel + 1,
            )
            dataAdjusted = np.ascontiguousarray(
                matrixDataLocal - currentBackground[None, :],
                dtype=np.float32,
            )
            ecmKwargs = dict(
                matrixData=dataAdjusted,
                matrixPluginMuncInit=currentMunc,
                matrixF=matrixFLocal,
                matrixQ0=matrixQ0Local,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                ECM_fixedBackgroundIters=int(ecmItersLocal),
                ECM_fixedBackgroundRtol=float(ecmRtolLocal),
                pad=float(pad),
                ECM_robustTNu=float(ECM_robustTNu),
                returnIntermediates=True,
                ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
                ECM_useProcessPrecisionReweighting=bool(useProcPrecLocal),
                ECM_useAPN=bool(useAPNLocal),
                obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
                obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
                procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                APN_minQ=float(minQ),
                APN_maxQ=float(maxQ),
                lambdaExpInit=lambdaExpLocal,
                processPrecExpInit=processPrecExpLocal,
                replicateBiasInit=replicateBiasLocal,
            )
            try:
                ecmOutLocal = cconsenrich.cfixedBackgroundECM(
                    **ecmKwargs,
                    returnDiagnostics=True,
                )
            except TypeError as exc:
                if "returnDiagnostics" not in str(exc):
                    raise
                ecmOutLocal = cconsenrich.cfixedBackgroundECM(**ecmKwargs)

            ecmDiagnosticsLocal = None
            if len(ecmOutLocal) == 10 and isinstance(ecmOutLocal[-1], Mapping):
                ecmDiagnosticsLocal = ecmOutLocal[-1]
                ecmOutLocal = ecmOutLocal[:-1]
            if len(ecmOutLocal) != 9:
                raise ValueError(
                    "Expected cfixedBackgroundECM(..., returnIntermediates=True) to return 9 values "
                    f"(got {len(ecmOutLocal)})."
                )

            (
                ecmItersDoneLocal,
                nllECMLocal,
                stateSmoothedLocal,
                stateCovarSmoothedLocal,
                lagCovSmoothedLocal,
                postFitResidualsLocal,
                lambdaExpLocal,
                processPrecExpLocal,
                replicateBiasLocal,
            ) = ecmOutLocal
            actualOuterPasses = int(outerPassIndex + 1)
            currentECMNLLLocal = float(nllECMLocal)
            ecmDiagnosticsNormalized = _normalizeFixedBackgroundECMDiagnostics(
                ecmDiagnosticsLocal,
                itersDone=int(ecmItersDoneLocal),
                finalNLL=currentECMNLLLocal,
                maxIters=int(ecmItersLocal),
                outerPass=actualOuterPasses,
            )
            lastInnerECMConvergedLocal = bool(
                ecmDiagnosticsNormalized.get("converged") is True
            )

            if lambdaExpLocal is not None:
                lambdaExpLocal = np.asarray(lambdaExpLocal, dtype=np.float32)
            if processPrecExpLocal is not None:
                processPrecExpLocal = np.asarray(processPrecExpLocal, dtype=np.float32)
            lambdaMeanLocal, lambdaMedianLocal = _observationLambdaSummary(
                lambdaExpLocal,
                lower=float(observationPrecisionMultiplierMin),
                upper=float(observationPrecisionMultiplierMax),
            )
            replicateBiasLocal = np.asarray(replicateBiasLocal, dtype=np.float32)
            stateSmoothedLocal = np.asarray(stateSmoothedLocal, dtype=np.float32)
            stateCovarSmoothedLocal = np.asarray(
                stateCovarSmoothedLocal, dtype=np.float32
            )
            lagCovSmoothedLocal = np.asarray(lagCovSmoothedLocal, dtype=np.float32)
            postFitResidualsLocal = np.asarray(postFitResidualsLocal, dtype=np.float32)

            if not fitBackgroundLocal:
                lastBackgroundShiftLocal = 0.0
                lastBackgroundShiftTolLocal = 0.0
                _recordOuterObjective(
                    ecmDiagnosticsNormalized,
                    ecmFitNLL=currentECMNLLLocal,
                )
                ecmDiagnosticsNormalized["background_shift"] = metadataFloat(
                    lastBackgroundShiftLocal
                )
                ecmDiagnosticsNormalized["background_shift_threshold"] = metadataFloat(
                    lastBackgroundShiftTolLocal
                )
                ecmDiagnosticsNormalized["background_shift_stable"] = True
                ecmDiagnosticsNormalized["observation_lambda_mean"] = lambdaMeanLocal
                ecmDiagnosticsNormalized["observation_lambda_median"] = lambdaMedianLocal
                ecmDiagnosticsNormalized["outer_inner_ecm_converged"] = bool(
                    lastInnerECMConvergedLocal
                )
                ecmDiagnosticsNormalized["outer_stable_iters"] = int(
                    outerStableItersLocal
                )
                ecmDiagnosticsNormalized["outer_patience_target"] = int(
                    outerPatienceTargetLocal
                )
                fixedBackgroundECMDiagnostics.append(ecmDiagnosticsNormalized)
                outerConvergedLocal = True
                outerStopReasonLocal = "fit_background_false"
                logger.info(
                    "outerPass[1/1]:\n\tfitBackground=False\n\tbackgroundShift=0\n\tlambdaMean=%s\n\tlambdaMedian=%s\n\touterObjectiveChangePerCell=%s",
                    _formatMaybeFloat(lambdaMeanLocal),
                    _formatMaybeFloat(lambdaMedianLocal),
                    _formatMaybeFloat(lastOuterObjectiveChangePerCellLocal),
                )
                break

            invVarMatrix = 1.0 / np.maximum(currentMunc + float(pad), 1.0e-8)
            if lambdaExpLocal is not None:
                obsPrecision = np.clip(
                    np.asarray(lambdaExpLocal, dtype=np.float32).reshape(
                        1, intervalCount
                    ),
                    float(observationPrecisionMultiplierMin),
                    float(observationPrecisionMultiplierMax),
                )
                invVarMatrix *= obsPrecision
            residualMatrix = (
                np.asarray(matrixDataLocal, dtype=np.float32)
                - np.asarray(replicateBiasLocal[:, None], dtype=np.float32)
                - np.asarray(stateSmoothedLocal[:, 0][None, :], dtype=np.float32)
            )
            nextBackground = _solveZeroCenteredBackground(
                residualMatrix=residualMatrix,
                invVarMatrix=invVarMatrix,
                blockLenIntervals=int(blockLenIntervals),
                backgroundSmoothness=float(ECM_backgroundSmoothness),
                zeroCenter=bool(ECM_zeroCenterBackground),
            )

            bgChange = float(
                np.max(
                    np.abs(np.asarray(nextBackground) - np.asarray(currentBackground))
                )
            )
            bgScale = float(
                max(
                    np.max(np.abs(np.asarray(nextBackground))),
                    np.max(np.abs(np.asarray(currentBackground))),
                    1.0,
                )
            )
            bgTol = float(backgroundShiftTolMultiplier * bgScale)
            currentBackground = np.asarray(nextBackground, dtype=np.float32)
            lastBackgroundShiftLocal = float(bgChange)
            lastBackgroundShiftTolLocal = float(bgTol)
            backgroundShiftStable = bool(bgChange <= bgTol)
            _recordOuterObjective(
                ecmDiagnosticsNormalized,
                ecmFitNLL=currentECMNLLLocal,
            )
            if (
                backgroundShiftStable
                and lastOuterObjectiveStableLocal
                and lastInnerECMConvergedLocal
            ):
                outerStableItersLocal += 1
            else:
                outerStableItersLocal = 0
            ecmDiagnosticsNormalized["background_shift"] = metadataFloat(
                lastBackgroundShiftLocal
            )
            ecmDiagnosticsNormalized["background_shift_threshold"] = metadataFloat(
                lastBackgroundShiftTolLocal
            )
            ecmDiagnosticsNormalized["background_shift_stable"] = bool(
                backgroundShiftStable
            )
            ecmDiagnosticsNormalized["observation_lambda_mean"] = lambdaMeanLocal
            ecmDiagnosticsNormalized["observation_lambda_median"] = lambdaMedianLocal
            ecmDiagnosticsNormalized["outer_inner_ecm_converged"] = bool(
                lastInnerECMConvergedLocal
            )
            ecmDiagnosticsNormalized["outer_stable_iters"] = int(outerStableItersLocal)
            ecmDiagnosticsNormalized["outer_patience_target"] = int(
                outerPatienceTargetLocal
            )
            fixedBackgroundECMDiagnostics.append(ecmDiagnosticsNormalized)
            logger.info(
                "outerPass[%d/%d]:\n\tbackgroundShift=%.6g\n\tbackgroundShiftThreshold=%.6g\n\tlambdaMean=%s\n\tlambdaMedian=%s\n\touterObjectivePerCell=%s\n\touterObjectiveChangePerCell=%s\n\touterObjectiveThresholdPerCell=%s\n\touterStable=%d/%d\n\tinnerECMConverged=%s",
                int(outerPassIndex + 1),
                int(outerPassCount),
                float(bgChange),
                float(bgTol),
                _formatMaybeFloat(lambdaMeanLocal),
                _formatMaybeFloat(lambdaMedianLocal),
                _formatMaybeFloat(lastOuterObjectivePerCellLocal),
                _formatMaybeFloat(lastOuterObjectiveChangePerCellLocal),
                _formatMaybeFloat(lastOuterObjectiveTolPerCellLocal),
                int(outerStableItersLocal),
                int(outerPatienceTargetLocal),
                str(bool(lastInnerECMConvergedLocal)),
            )
            if (
                outerPassIndex + 1
            ) >= minOuterIters and outerStableItersLocal >= outerPatienceTargetLocal:
                outerConvergedLocal = True
                outerStopReasonLocal = "background_objective_inner_stable"
                break

        if fitBackgroundLocal and not outerConvergedLocal:
            if not lastInnerECMConvergedLocal:
                outerStopReasonLocal = "max_outer_passes_inner_ecm_unconverged"
            elif not lastOuterObjectiveStableLocal:
                outerStopReasonLocal = "max_outer_passes_objective"
            elif outerStableItersLocal < outerPatienceTargetLocal:
                outerStopReasonLocal = "max_outer_passes_patience"

        dataAdjusted = np.ascontiguousarray(
            matrixDataLocal - currentBackground[None, :],
            dtype=np.float32,
        )
        _logAsciiBlock(
            f"{phaseLabel} / forward-backward scoring",
            (
                ("tracks", int(mLocal)),
                ("intervals", int(nLocal)),
                ("background model", bool(fitBackgroundLocal)),
                ("obs precision weights", bool(lambdaExpLocal is not None)),
                ("process precision weights", bool(processPrecExpLocal is not None)),
            ),
            indentLevel=phaseIndentLevel + 1,
        )
        (
            _phiHatLocal,
            sumNLLLocal,
            stateForwardLocal,
            stateCovarForwardLocal,
            pNoiseForwardLocal,
            stateSmoothedLocal,
            stateCovarSmoothedLocal,
            lagCovSmoothedLocal,
            postFitResidualsLocal,
            NISLocal,
        ) = _runForwardBackward(
            matrixDataLocal=dataAdjusted,
            matrixMuncLocal=currentMunc,
            matrixFLocal=matrixFLocal,
            matrixQ0Local=matrixQ0Local,
            lambdaExp=lambdaExpLocal,
            processPrecExp=processPrecExpLocal,
            replicateBias=replicateBiasLocal,
            useProcPrecReweightLocal=useProcPrecLocal,
            useAPNLocal=useAPNLocal,
        )
        return {
            "matrixMunc": currentMunc,
            "background": currentBackground,
            "lambdaExp": lambdaExpLocal,
            "processPrecExp": processPrecExpLocal,
            "replicateBias": np.asarray(replicateBiasLocal, dtype=np.float32),
            "stateForward": np.asarray(stateForwardLocal, dtype=np.float32),
            "stateCovarForward": np.asarray(stateCovarForwardLocal, dtype=np.float32),
            "pNoiseForward": np.asarray(pNoiseForwardLocal, dtype=np.float32),
            "stateSmoothed": np.asarray(stateSmoothedLocal, dtype=np.float32),
            "stateCovarSmoothed": np.asarray(stateCovarSmoothedLocal, dtype=np.float32),
            "lagCovSmoothed": np.asarray(lagCovSmoothedLocal, dtype=np.float32),
            "postFitResiduals": np.asarray(postFitResidualsLocal, dtype=np.float32),
            "NIS": np.asarray(NISLocal, dtype=np.float32),
            "sumNLL": float(sumNLLLocal),
            "backgroundShift": float(lastBackgroundShiftLocal),
            "backgroundShiftThreshold": metadataFloat(lastBackgroundShiftTolLocal),
            "outerNLL": metadataFloat(lastOuterNLLLocal),
            "outerNLLChange": metadataFloat(lastOuterNLLChangeLocal),
            "outerNLLThreshold": metadataFloat(lastOuterNLLTolLocal),
            "outerNLLStable": bool(lastOuterNLLStableLocal),
            "outerObjective": metadataFloat(lastOuterObjectiveLocal),
            "outerObjectivePerCell": metadataFloat(lastOuterObjectivePerCellLocal),
            "outerObjectiveChangePerCell": metadataFloat(
                lastOuterObjectiveChangePerCellLocal
            ),
            "outerObjectiveThresholdPerCell": metadataFloat(
                lastOuterObjectiveTolPerCellLocal
            ),
            "outerObjectiveStable": bool(lastOuterObjectiveStableLocal),
            "outerEffectiveObservationCount": (
                int(lastObjectiveDiagnosticsLocal["effective_observation_count"])
                if lastObjectiveDiagnosticsLocal
                else 0
            ),
            "outerStableIters": int(outerStableItersLocal),
            "outerPatienceTarget": int(outerPatienceTargetLocal),
            "innerECMConverged": bool(lastInnerECMConvergedLocal),
            "outerConverged": bool(outerConvergedLocal),
            "outerStopReason": str(outerStopReasonLocal),
            "requestedOuterIters": int(requestedOuterIters),
            "minOuterIters": int(minOuterIters),
            "plannedOuterPasses": int(outerPassCount),
            "actualOuterPasses": int(actualOuterPasses),
            "warmStart": warmStartSummaryLocal,
            "fixedBackgroundECMDiagnostics": fixedBackgroundECMDiagnostics,
        }

    deltaF_fit = _resolveFixedDeltaF(deltaF)

    matrixF = buildMatrixF(float(deltaF_fit))
    matrixQ0 = (
        np.ascontiguousarray(initialProcessQArr, dtype=np.float32).copy()
        if initialProcessQArr is not None
        else buildMatrixQ0(float(deltaF_fit))
    )
    fitProcessQWarmup: Mapping[str, Any] | None = None
    processQCalibrationInfo: dict[str, float] | None = None
    postQInitialBackground = initialBackgroundArr
    postQInitialReplicateBias = initialReplicateBiasArr
    postQInitialLambda = initialObservationPrecisionArr
    postQInitialProcessPrec = initialProcessPrecisionArr

    if initialProcessQArr is not None:
        processQCalibrationInfo = {
            "q_level": float(matrixQ0[0, 0]),
            "q_trend": float(matrixQ0[1, 1]),
            "raw_q_level": float(matrixQ0[0, 0]),
            "raw_q_trend": float(matrixQ0[1, 1]),
            "q_level_target": float(matrixQ0[0, 0]),
            "q_trend_target": float(matrixQ0[1, 1]),
            "q_level_prior_weight": 0.0,
            "q_trend_prior_weight": 0.0,
            "q_level_shrunk": float(matrixQ0[0, 0]),
            "q_trend_shrunk": float(matrixQ0[1, 1]),
            "q_level_floor": float(minQ),
            "q_trend_floor": float(minQ),
            "q_level_floor_hit": 0.0,
            "q_trend_floor_hit": 0.0,
            "q_level_cap_hit": 0.0,
            "q_trend_cap_hit": 0.0,
            "q_level_final_raw_ratio": 1.0,
            "q_trend_final_raw_ratio": 1.0,
            "transition_count": float(max(intervalCount - 1, 0)),
            "warm_start_process_q": 1.0,
        }
        logger.info(
            "processQCalibration=warm-start: using initialProcessQ q_level=%.6g q_trend=%.6g",
            float(matrixQ0[0, 0]),
            float(matrixQ0[1, 1]),
        )
    elif processQCalibrationMode == PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL:
        warmupIters = max(1, int(processQWarmupECMIters))
        warmupOuterIters = max(
            1,
            int(
                PROCESS_Q_CALIBRATION_DEFAULT_OUTER_ITERS
                if processQWarmupOuterIters is None
                else processQWarmupOuterIters
            ),
        )
        stageStart = time.perf_counter()
        _logAsciiBlock(
            "process Q warmup",
            (
                ("purpose", "estimate diagonal process covariance"),
                ("tracks", int(trackCount)),
                ("intervals", int(intervalCount)),
                ("outer passes", int(warmupOuterIters)),
                ("ECM max iterations", int(warmupIters)),
                ("process precision weights", False),
                ("adaptive process noise", False),
            ),
            indentLevel=logIndentLevel + 1,
        )
        logger.info(
            "runConsenrich.processQWarmup.start tracks=%d intervals=%d ECM_fixedBackgroundIters=%d outerIters=%d",
            int(trackCount),
            int(intervalCount),
            int(warmupIters),
            int(warmupOuterIters),
        )
        fitProcessQWarmup = _fitOuter(
            matrixDataLocal=matrixData,
            matrixMuncLocal=matrixMunc,
            matrixFLocal=matrixF,
            matrixQ0Local=matrixQ0,
            ecmItersLocal=warmupIters,
            ecmRtolLocal=float(ECM_fixedBackgroundRtol),
            outerItersLocal=warmupOuterIters,
            minOuterItersLocal=1,
            useProcPrecReweightOverride=False,
            useAPNOverride=False,
            phaseLabel="process Q warmup",
            phaseIndentLevel=logIndentLevel + 1,
        )
        logger.info(
            "runConsenrich.processQWarmup.done elapsed=%.3fs",
            time.perf_counter() - stageStart,
        )
        matrixQ0, processQCalibrationInfo = _estimateRegularizedDiagonalProcessQ(
            stateSmoothed=np.asarray(fitProcessQWarmup["stateSmoothed"]),
            stateCovarSmoothed=np.asarray(fitProcessQWarmup["stateCovarSmoothed"]),
            lagCovSmoothed=np.asarray(fitProcessQWarmup["lagCovSmoothed"]),
            matrixF=matrixF,
            minQ=float(minQ),
            maxQ=float(maxQ),
            processQLevelTarget=processQLevelTarget,
            processQTrendTarget=processQTrendTarget,
            processQLevelPriorWeight=float(processQLevelPriorWeight),
            processQTrendPriorWeight=float(processQTrendPriorWeight),
        )
        processQCalibrationInfo["warmup_ecm_iters"] = float(warmupIters)
        processQCalibrationInfo["warmup_outer_passes"] = float(warmupOuterIters)
        processQCalibrationInfo["warm_start_process_q"] = 0.0
        postQInitialBackground = np.asarray(
            fitProcessQWarmup["background"],
            dtype=np.float32,
        )
        postQInitialReplicateBias = np.asarray(
            fitProcessQWarmup["replicateBias"],
            dtype=np.float32,
        )
        postQInitialLambda = (
            np.asarray(fitProcessQWarmup["lambdaExp"], dtype=np.float32)
            if fitProcessQWarmup.get("lambdaExp") is not None
            else initialObservationPrecisionArr
        )
        _warnIfProcessQCalibrationForced(processQCalibrationInfo)
        logger.info(
            "processQCalibration=%s:\n"
            "\tq_level=%.6g\tq_trend=%.6g\n"
            "\traw_level=%.6g\traw_trend=%.6g\n"
            "\ttarget_level=%.6g\ttarget_trend=%.6g\n"
            "\tweight_level=%.6g\tweight_trend=%.6g\n"
            "\twarmup_outer_passes=%d\twarmup_ecm_iters=%d",
            PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL,
            processQCalibrationInfo["q_level"],
            processQCalibrationInfo["q_trend"],
            processQCalibrationInfo["raw_q_level"],
            processQCalibrationInfo["raw_q_trend"],
            processQCalibrationInfo["q_level_target"],
            processQCalibrationInfo["q_trend_target"],
            processQCalibrationInfo["q_level_prior_weight"],
            processQCalibrationInfo["q_trend_prior_weight"],
            int(warmupOuterIters),
            int(warmupIters),
        )
    else:
        logger.info("processQCalibration=none: using legacy scalar process Q")

    baseFitPhaseLabel = (
        "post-Q fit"
        if processQCalibrationMode == PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL
        else "model fit"
    )
    fitPhaseLabel = (
        f"{logRunRole} {baseFitPhaseLabel}" if logRunRole else baseFitPhaseLabel
    )
    fitLogEvent = (
        "postQFit"
        if processQCalibrationMode == PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL
        else "modelFit"
    )

    stageStart = time.perf_counter()
    _logAsciiBlock(
        fitPhaseLabel,
        (
            ("tracks", int(trackCount)),
            ("intervals", int(intervalCount)),
            ("ECM max iterations", int(ECM_fixedBackgroundIters)),
            ("outer passes", int(ECM_outerIters)),
            ("background model fit", bool(fitBackground)),
            ("obs precision weights", bool(ECM_useObsPrecisionReweighting)),
            ("process precision weights", bool(ECM_useProcessPrecisionReweighting)),
            ("adaptive process noise", bool(ECM_useAPN)),
        ),
        indentLevel=logIndentLevel + 1,
    )
    logger.info(
        "runConsenrich.%s.start tracks=%d intervals=%d ECM_fixedBackgroundIters=%d outerIters=%d",
        fitLogEvent,
        int(trackCount),
        int(intervalCount),
        int(ECM_fixedBackgroundIters),
        int(ECM_outerIters),
    )
    fitFinal = _fitOuter(
        matrixDataLocal=matrixData,
        matrixMuncLocal=matrixMunc,
        matrixFLocal=matrixF,
        matrixQ0Local=matrixQ0,
        ecmItersLocal=int(ECM_fixedBackgroundIters),
        ecmRtolLocal=float(ECM_fixedBackgroundRtol),
        initialBackgroundLocal=postQInitialBackground,
        initialReplicateBiasLocal=postQInitialReplicateBias,
        initialLambdaLocal=postQInitialLambda,
        initialProcessPrecLocal=postQInitialProcessPrec,
        phaseLabel=fitPhaseLabel,
        phaseIndentLevel=logIndentLevel + 1,
    )
    logger.info(
        "runConsenrich.%s.done elapsed=%.3fs",
        fitLogEvent,
        time.perf_counter() - stageStart,
    )
    replicateBias_final = np.asarray(fitFinal["replicateBias"], dtype=np.float32)
    stateSmoothed = np.asarray(fitFinal["stateSmoothed"], dtype=np.float32)
    stateCovarSmoothed = np.asarray(fitFinal["stateCovarSmoothed"], dtype=np.float32)
    postFitResiduals = np.asarray(fitFinal["postFitResiduals"], dtype=np.float32)
    NIS = np.asarray(fitFinal["NIS"], dtype=np.float32)
    finalForwardNIS = _finalForwardNIS(NIS)
    finalForwardGainContigSummary = _finalForwardReplicateGainContigSummary(
        stateCovarForward=np.asarray(fitFinal["stateCovarForward"], dtype=np.float32),
        matrixMunc=np.asarray(fitFinal["matrixMunc"], dtype=np.float32),
        lambdaExp=(
            fitFinal.get("lambdaExp")
            if bool(ECM_useObsPrecisionReweighting)
            else None
        ),
        pad=float(pad),
        obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
        obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
    )
    _logAsciiBlock(
        f"{fitPhaseLabel} summary",
        (
            ("fit NLL", float(fitFinal.get("sumNLL", np.nan))),
            ("standardized forward innovation", float(finalForwardNIS)),
            ("backgroundShift at stop", float(fitFinal.get("backgroundShift", np.nan))),
            (
                "outer objective/cell change at stop",
                fitFinal.get("outerObjectiveChangePerCell"),
            ),
            ("outer stop reason", fitFinal.get("outerStopReason", "unknown")),
            ("background max abs", float(np.max(np.abs(fitFinal["background"])))),
            ("elapsed seconds", time.perf_counter() - stageStart),
        ),
        indentLevel=logIndentLevel + 1,
    )
    logger.info("Standardized forward innovation: %.6g", finalForwardNIS)
    processQCalibrationMetadata = (
        None
        if processQCalibrationInfo is None
        else {
            key: metadataFloat(value) for key, value in processQCalibrationInfo.items()
        }
    )
    runDiagnostics = {
        "final_nll": metadataFloat(float(fitFinal.get("sumNLL", np.nan))),
        "final_forward_nis": metadataFloat(finalForwardNIS),
        "final_forward_gain_contig_summary": {
            "mean": [
                metadataFloat(float(value))
                for value in finalForwardGainContigSummary["mean"]
            ],
            "median": [
                metadataFloat(float(value))
                for value in finalForwardGainContigSummary["median"]
            ],
        },
        "precision_reweighting_boundary_hits": summarizePrecisionBoundaryHits(
            observationPrecision=(
                fitFinal.get("lambdaExp")
                if bool(ECM_useObsPrecisionReweighting)
                else None
            ),
            observationPrecisionMin=float(observationPrecisionMultiplierMin),
            observationPrecisionMax=float(observationPrecisionMultiplierMax),
            processPrecision=(
                fitFinal.get("processPrecExp")
                if bool(ECM_useProcessPrecisionReweighting) and not bool(ECM_useAPN)
                else None
            ),
            processPrecisionMin=float(processPrecisionMultiplierMin),
            processPrecisionMax=float(processPrecisionMultiplierMax),
        ),
        "process_q_calibration": processQCalibrationMetadata,
        "process_q_warmup_fit": (
            _fitDiagnosticsMetadata(fitProcessQWarmup)
            if fitProcessQWarmup is not None
            else None
        ),
        "post_q_fit": _fitDiagnosticsMetadata(fitFinal),
    }

    outStateSmoothed = np.asarray(stateSmoothed, dtype=np.float32)
    outStateCovarSmoothed = np.asarray(stateCovarSmoothed, dtype=np.float32)
    outPostFitResiduals = np.asarray(postFitResiduals, dtype=np.float32)
    outNIS = np.asarray(NIS, dtype=np.float32)

    if boundState:
        np.clip(
            outStateSmoothed[:, 0],
            np.float32(stateLowerBound),
            np.float32(stateUpperBound),
            out=outStateSmoothed[:, 0],
        )

    totalElapsed = time.perf_counter() - totalStart
    _logAsciiBlock(
        "core done",
        (
            ("tracks", int(trackCount)),
            ("intervals", int(intervalCount)),
            ("elapsed seconds", float(totalElapsed)),
        ),
        indentLevel=logIndentLevel,
    )
    logger.info(
        "runConsenrich.core.done tracks=%d intervals=%d elapsed=%.3fs",
        int(trackCount),
        int(intervalCount),
        totalElapsed,
    )

    def _maybeAddDiagnostics(result: tuple[Any, ...]) -> tuple[Any, ...]:
        if returnDiagnostics:
            return (*result, runDiagnostics)
        return result

    if returnScales:
        if returnReplicateOffsets:
            result = (
                outStateSmoothed,
                outStateCovarSmoothed,
                outPostFitResiduals,
                outNIS,
                np.asarray(replicateBias_final, dtype=np.float32),
                intervalToBlockMap,
            )
            return _maybeAddDiagnostics(result)
        result = (
            outStateSmoothed,
            outStateCovarSmoothed,
            outPostFitResiduals,
            outNIS,
            intervalToBlockMap,
        )
        return _maybeAddDiagnostics(result)

    result = (
        outStateSmoothed,
        outStateCovarSmoothed,
        outPostFitResiduals,
        outNIS,
    )
    return _maybeAddDiagnostics(result)


def getPrimaryState(
    stateVectors: np.ndarray,
    roundPrecision: int = 4,
    stateLowerBound: Optional[float] = None,
    stateUpperBound: Optional[float] = None,
    boundState: bool = False,
) -> npt.NDArray[np.float32]:
    r"""Get the primary state variable (*signal level*) from each estimated state vector after running Consenrich.

    :param stateVectors: State vectors from :func:`runConsenrich`.
    :type stateVectors: npt.NDArray[np.float32]
    :return: A one-dimensional numpy array of the primary state estimates ( signal level, :math:`\widetilde{x}_{[i,0]}`).
    :rtype: npt.NDArray[np.float32]
    """
    out_ = np.ascontiguousarray(stateVectors[:, 0], dtype=np.float32)
    if boundState:
        if stateLowerBound is not None:
            np.maximum(out_, np.float32(stateLowerBound), out=out_)
        if stateUpperBound is not None:
            np.minimum(out_, np.float32(stateUpperBound), out=out_)
    np.round(out_, decimals=roundPrecision, out=out_)
    return out_


def getBedMask(
    chromosome: str,
    bedFile: str,
    intervals: np.ndarray,
) -> np.ndarray:
    r"""Return a 1/0 mask for intervals overlapping a sorted and merged BED file.

    This function is a wrapper for :func:`cconsenrich.cbedMask`.

    :param chromosome: The chromosome name.
    :type chromosome: str
    :param intervals: chromosome-specific, sorted, non-overlapping start positions of genomic intervals.
      Each interval is assumed `intervalSizeBP`.
    :type intervals: np.ndarray
    :param bedFile: Path to a sorted and merged BED file
    :type bedFile: str
    :return: An `intervals`-length mask s.t. True indicates the interval overlaps a feature in the BED file.
    :rtype: np.ndarray
    """
    if not os.path.exists(bedFile):
        raise ValueError(f"Could not find {bedFile}")
    if len(intervals) < 2:
        raise ValueError("intervals must contain at least two positions")
    bedFile_ = str(bedFile)

    # + quick check for constant steps
    intervals_ = np.asarray(intervals, dtype=np.uint32)
    if (intervals_[1] - intervals_[0]) != (intervals_[-1] - intervals_[-2]):
        raise ValueError("Intervals are not fixed in size")

    stepSize_: int = intervals[1] - intervals[0]
    return cconsenrich.cbedMask(
        chromosome,
        bedFile_,
        intervals_,
        stepSize_,
    ).astype(np.bool_)


class PSplineLogVarianceTrend(NamedTuple):
    r"""Guarded-GCV P-spline fit for a log-variance trend."""

    knots: np.ndarray
    degree: int
    beta: np.ndarray
    xMin: float
    xMax: float
    lambdaHat: float
    edf: float
    gcv: float
    lambdaAtBoundary: bool
    finiteCount: int
    diagnostics: dict[str, Any]


class PooledMuncVarianceTrend(NamedTuple):
    r"""Genome-wide MUNC variance trend with replicate-specific scale factors."""

    trend: PSplineLogVarianceTrend
    replicateVarianceFactors: np.ndarray
    diagnostics: dict[str, Any]


def _muncTrendPredictor(values: np.ndarray) -> np.ndarray:
    r"""Signed MUNC trend predictor: ``sign(mean) * log1p(abs(mean))``."""

    arr = np.asarray(values, dtype=np.float64)
    out = np.sign(arr) * np.log1p(np.abs(arr))
    out[~np.isfinite(out)] = np.nan
    return out


def _weightedQuantile(
    values: np.ndarray,
    weights: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    valuesArr = np.asarray(values, dtype=np.float64).ravel()
    weightsArr = np.asarray(weights, dtype=np.float64).ravel()
    probsArr = np.asarray(probs, dtype=np.float64).ravel()
    order = np.argsort(valuesArr)
    valuesArr = valuesArr[order]
    weightsArr = weightsArr[order]
    weightsArr = np.maximum(weightsArr, 0.0)
    totalWeight = float(np.sum(weightsArr))
    if totalWeight <= 0.0 or valuesArr.size == 0:
        return np.full(probsArr.shape, np.nan, dtype=np.float64)
    cdf = np.cumsum(weightsArr)
    return np.interp(
        np.clip(probsArr, 0.0, 1.0) * totalWeight,
        cdf,
        valuesArr,
    )


def _psplineKnots(
    xMin: float,
    xMax: float,
    numBasis: int,
    degree: int,
    x: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    degree = int(max(0, degree))
    numBasis = int(max(numBasis, degree + 1))
    if (not np.isfinite(xMin)) or (not np.isfinite(xMax)) or xMax <= xMin:
        xMin = float(xMin) if np.isfinite(xMin) else 0.0
        xMax = xMin + 1.0
    internalCount = max(0, int(numBasis) - int(degree) - 1)
    if internalCount > 0:
        probs = np.linspace(0.0, 1.0, internalCount + 2, dtype=np.float64)[1:-1]
        if x is not None and weights is not None:
            internal = _weightedQuantile(
                np.asarray(x, dtype=np.float64),
                np.asarray(weights, dtype=np.float64),
                probs,
            )
            internal = internal[np.isfinite(internal)]
        else:
            internal = np.linspace(xMin, xMax, internalCount + 2, dtype=np.float64)[
                1:-1
            ]
        minGap = max((xMax - xMin) * 1.0e-10, 1.0e-12)
        internal = internal[(internal > xMin + minGap) & (internal < xMax - minGap)]
        internal = np.unique(internal)
    else:
        internal = np.empty(0, dtype=np.float64)
    return np.concatenate(
        (
            np.full(degree + 1, xMin, dtype=np.float64),
            internal,
            np.full(degree + 1, xMax, dtype=np.float64),
        )
    )


def _supportLimitedBasisCount(
    x: np.ndarray,
    weights: np.ndarray,
    requestedBasis: int,
    degree: int,
    minObsPerBasis: float,
) -> tuple[int, float, int]:
    minBasis = int(max(1, degree + 1))
    requested = int(max(requestedBasis, minBasis))
    weightsArr = np.asarray(weights, dtype=np.float64).ravel()
    totalWeight = float(np.sum(weightsArr))
    weightSq = float(np.sum(np.square(weightsArr)))
    nEff = (totalWeight * totalWeight / weightSq) if weightSq > 0.0 else 0.0
    minObs = float(max(minObsPerBasis, 1.0))
    basisByObs = int(np.floor(nEff / minObs))
    uniqueCount = int(np.unique(np.asarray(x, dtype=np.float64).ravel()).size)
    basisBySupport = max(minBasis, min(uniqueCount, max(minBasis, basisByObs)))
    return int(max(minBasis, min(requested, basisBySupport))), float(nEff), uniqueCount


def _bsplineDesign(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    xArr = np.asarray(x, dtype=np.float64).ravel()
    return np.asarray(
        interpolate.BSpline.design_matrix(
            xArr,
            np.asarray(knots, dtype=np.float64),
            int(degree),
            extrapolate=False,
        ).toarray(),
        dtype=np.float64,
    )


def _coefficientDiffPenalty(numBasis: int, order: int) -> np.ndarray:
    numBasis = int(numBasis)
    order = int(max(order, 0))
    if order == 0:
        return np.eye(numBasis, dtype=np.float64)
    if numBasis <= order:
        return np.zeros((numBasis, numBasis), dtype=np.float64)
    diff = np.diff(np.eye(numBasis, dtype=np.float64), n=order, axis=0)
    return diff.T @ diff


def fitPSplineLogVarianceTrend(
    blockMeans: np.ndarray,
    blockVariances: np.ndarray,
    weights: np.ndarray | None = None,
    eps: float = 1.0e-2,
    trendNumBasis: int = 60,
    trendMinObsPerBasis: float = 25.0,
    trendSplineDegree: int = 2,
    trendPenaltyOrder: int = 2,
    trendLambdaMin: float = 1.0e-6,
    trendLambdaMax: float = 1.0e6,
    trendLambdaGridSize: int = 41,
    trendMinEdf: float = 3.0,
    trendMaxEdf: float | None = 30.0,
) -> PSplineLogVarianceTrend:
    r"""Fit a P-spline trend to ``log(variance)`` versus the signed MUNC predictor.

    Smoothing-parameter selection is guarded GCV only. The fit intentionally
    imposes no monotonicity constraint and no signal-dependent linear floor.
    """

    means = np.asarray(blockMeans, dtype=np.float64).ravel()
    variances = np.asarray(blockVariances, dtype=np.float64).ravel()
    if weights is None:
        weightsArr = np.ones_like(means, dtype=np.float64)
    else:
        weightsArr = np.asarray(weights, dtype=np.float64).ravel()
        if weightsArr.size != means.size:
            raise ValueError("weights must have the same length as blockMeans")

    if variances.size != means.size:
        raise ValueError("blockMeans and blockVariances must have the same length")

    floor = float(max(float(eps), 1.0e-12))
    x = _muncTrendPredictor(means)
    y = np.log(np.maximum(variances, floor))
    mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(weightsArr) & (weightsArr > 0.0)
    )
    x = x[mask]
    y = y[mask]
    weightsArr = weightsArr[mask]

    if x.size == 0:
        y0 = float(np.log(floor))
        return PSplineLogVarianceTrend(
            knots=np.empty(0, dtype=np.float64),
            degree=-1,
            beta=np.array([y0], dtype=np.float64),
            xMin=0.0,
            xMax=0.0,
            lambdaHat=0.0,
            edf=1.0,
            gcv=0.0,
            lambdaAtBoundary=False,
            finiteCount=0,
            diagnostics={"fallback": "no_finite_pairs"},
        )

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    weightsArr = weightsArr[order]

    xMin = float(x[0])
    xMax = float(x[-1])
    if x.size < max(4, int(trendSplineDegree) + 2) or xMax <= xMin:
        y0 = float(np.average(y, weights=weightsArr))
        return PSplineLogVarianceTrend(
            knots=np.empty(0, dtype=np.float64),
            degree=-1,
            beta=np.array([y0], dtype=np.float64),
            xMin=xMin,
            xMax=xMax,
            lambdaHat=0.0,
            edf=1.0,
            gcv=0.0,
            lambdaAtBoundary=False,
            finiteCount=int(x.size),
            diagnostics={"fallback": "constant_trend"},
        )

    degree = int(max(0, trendSplineDegree))
    requestedBasis = int(max(int(trendNumBasis), degree + 1))
    numBasis, nEff, uniqueXCount = _supportLimitedBasisCount(
        x=x,
        weights=weightsArr,
        requestedBasis=requestedBasis,
        degree=degree,
        minObsPerBasis=trendMinObsPerBasis,
    )
    knots = _psplineKnots(
        xMin,
        xMax,
        numBasis,
        degree,
        x=x,
        weights=weightsArr,
    )
    B = _bsplineDesign(np.clip(x, xMin, xMax), knots, degree)
    numBasis = int(B.shape[1])
    penalty = _coefficientDiffPenalty(numBasis, int(trendPenaltyOrder))

    sqrtW = np.sqrt(weightsArr)
    BW = B * sqrtW[:, None]
    yW = y * sqrtW
    gram = BW.T @ BW
    rhs = BW.T @ yW

    lamMin = float(trendLambdaMin)
    lamMax = float(trendLambdaMax)
    if (not np.isfinite(lamMin)) or lamMin <= 0.0:
        lamMin = 1.0e-6
    if (not np.isfinite(lamMax)) or lamMax <= lamMin:
        lamMax = 1.0e6
    gridSize = int(max(3, trendLambdaGridSize))
    lambdaGrid = np.logspace(np.log10(lamMin), np.log10(lamMax), gridSize)

    minEdf = float(min(max(1.0, trendMinEdf), max(float(numBasis), 1.0)))
    if trendMaxEdf is None or not np.isfinite(float(trendMaxEdf)):
        maxEdf = min(float(numBasis - 1), 30.0)
    else:
        maxEdf = float(trendMaxEdf)
    maxEdf = max(minEdf, min(maxEdf, float(numBasis)))
    best: tuple[float, float, float, np.ndarray] | None = None
    bestRejected: tuple[float, float, float, np.ndarray] | None = None
    ridge = 1.0e-10 * max(float(np.trace(gram)) / max(numBasis, 1), 1.0)

    for lam in lambdaGrid:
        A = gram + (float(lam) * penalty)
        try:
            beta = np.linalg.solve(A + ridge * np.eye(numBasis), rhs)
            edf = float(np.trace(np.linalg.solve(A + ridge * np.eye(numBasis), gram)))
        except np.linalg.LinAlgError:
            continue

        fitted = B @ beta
        residSS = float(np.sum(weightsArr * np.square(y - fitted)))
        R = float(max(x.size, 1))
        denom = 1.0 - (edf / R)
        if abs(denom) < 1.0e-8:
            gcv = float("inf")
        else:
            gcv = float((residSS / R) / (denom * denom))
        if not np.isfinite(gcv):
            continue

        candidate = (gcv, float(lam), edf, beta)
        if bestRejected is None or gcv < bestRejected[0]:
            bestRejected = candidate

        if edf < minEdf:
            continue
        if maxEdf is not None and edf > maxEdf:
            continue
        if best is None or gcv < best[0]:
            best = candidate

    if best is None:
        if bestRejected is None:
            y0 = float(np.average(y, weights=weightsArr))
            return PSplineLogVarianceTrend(
                knots=np.empty(0, dtype=np.float64),
                degree=-1,
                beta=np.array([y0], dtype=np.float64),
                xMin=xMin,
                xMax=xMax,
                lambdaHat=0.0,
                edf=1.0,
                gcv=0.0,
                lambdaAtBoundary=False,
                finiteCount=int(x.size),
                diagnostics={"fallback": "constant_after_solve_failure"},
            )
        best = bestRejected

    gcvHat, lambdaHat, edfHat, betaHat = best
    lambdaAtBoundary = bool(
        np.isclose(lambdaHat, lambdaGrid[0]) or np.isclose(lambdaHat, lambdaGrid[-1])
    )
    return PSplineLogVarianceTrend(
        knots=knots.astype(np.float64, copy=False),
        degree=degree,
        beta=np.asarray(betaHat, dtype=np.float64),
        xMin=xMin,
        xMax=xMax,
        lambdaHat=float(lambdaHat),
        edf=float(edfHat),
        gcv=float(gcvHat),
        lambdaAtBoundary=lambdaAtBoundary,
        finiteCount=int(x.size),
        diagnostics={
            "lambda_grid_min": float(lambdaGrid[0]),
            "lambda_grid_max": float(lambdaGrid[-1]),
            "lambda_at_boundary": lambdaAtBoundary,
            "num_basis": int(numBasis),
            "requested_num_basis": int(requestedBasis),
            "support_limited_num_basis": int(numBasis),
            "trend_n_eff": float(nEff),
            "trend_unique_x": int(uniqueXCount),
            "trend_min_obs_per_basis": float(max(trendMinObsPerBasis, 1.0)),
            "trend_min_edf": float(minEdf),
            "trend_max_edf": float(maxEdf),
            "knot_mode": "weighted_quantile",
            "degree": int(degree),
            "penalty_order": int(trendPenaltyOrder),
        },
    )


def evalPSplineLogVarianceTrend(
    trend: PSplineLogVarianceTrend,
    meanTrack: np.ndarray,
    eps: float = 1.0e-2,
    maxVariance: float | None = None,
) -> np.ndarray:
    floor = float(max(eps, 1.0e-12))
    if (
        maxVariance is None
        or not np.isfinite(float(maxVariance))
        or maxVariance <= floor
    ):
        cap = float(np.finfo(np.float32).max)
    else:
        cap = float(maxVariance)
    logFloor = float(np.log(floor))
    logCap = float(np.log(cap))
    predictorTrack = _muncTrendPredictor(meanTrack)
    return cconsenrich.cEvalPSplineLogVarianceTrend(
        predictorTrack,
        np.asarray(trend.knots, dtype=np.float64),
        np.asarray(trend.beta, dtype=np.float64),
        int(trend.degree),
        float(trend.xMin),
        float(trend.xMax),
        logFloor,
        logCap,
    )


def _winsorizedMedian(values: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if arr.size >= 20:
        qLo, qHi = np.quantile(arr, [lo, hi])
        arr = np.clip(arr, qLo, qHi)
    return float(np.median(arr))


def fitPooledMuncVarianceTrend(
    blockMeans: np.ndarray,
    blockVariances: np.ndarray,
    sampleIndex: np.ndarray,
    weights: np.ndarray | None = None,
    eps: float = 1.0e-2,
    trendNumBasis: int = 60,
    trendMinObsPerBasis: float = 25.0,
    trendMinEdf: float = 3.0,
    trendMaxEdf: float | None = 30.0,
    trendLambdaMin: float = 1.0e-6,
    trendLambdaMax: float = 1.0e6,
    trendLambdaGridSize: int = 41,
    maxIters: int = 3,
    tol: float = 0.02,
) -> PooledMuncVarianceTrend:
    r"""Fit a pooled signed MUNC trend plus replicate variance factors."""

    means = np.asarray(blockMeans, dtype=np.float64).ravel()
    variances = np.asarray(blockVariances, dtype=np.float64).ravel()
    samples = np.asarray(sampleIndex, dtype=np.intp).ravel()
    if means.shape != variances.shape or means.shape != samples.shape:
        raise ValueError("blockMeans, blockVariances, and sampleIndex must align")
    if weights is None:
        weightsArr = np.ones_like(means, dtype=np.float64)
    else:
        weightsArr = np.asarray(weights, dtype=np.float64).ravel()
        if weightsArr.shape != means.shape:
            raise ValueError("weights must align with blockMeans")

    mask = (
        np.isfinite(means)
        & np.isfinite(variances)
        & np.isfinite(weightsArr)
        & (variances > max(float(eps), 1.0e-12))
        & (weightsArr > 0.0)
        & (samples >= 0)
    )
    means = means[mask]
    variances = variances[mask]
    samples = samples[mask]
    weightsArr = weightsArr[mask]

    if samples.size == 0:
        trend = fitPSplineLogVarianceTrend(
            np.array([0.0], dtype=np.float64),
            np.array([max(float(eps), 1.0e-12)], dtype=np.float64),
            weights=np.array([1.0], dtype=np.float64),
            eps=eps,
            trendNumBasis=trendNumBasis,
            trendMinObsPerBasis=trendMinObsPerBasis,
            trendMinEdf=trendMinEdf,
            trendMaxEdf=trendMaxEdf,
            trendLambdaMin=trendLambdaMin,
            trendLambdaMax=trendLambdaMax,
            trendLambdaGridSize=trendLambdaGridSize,
        )
        return PooledMuncVarianceTrend(
            trend=trend,
            replicateVarianceFactors=np.ones(0, dtype=np.float64),
            diagnostics={"fallback": "no_valid_pairs"},
        )

    sampleCount = int(np.max(samples)) + 1
    logFactors = np.zeros(sampleCount, dtype=np.float64)
    trend: PSplineLogVarianceTrend | None = None
    maxChange = float("inf")
    itersUsed = 0
    for iterIndex in range(int(max(1, maxIters))):
        factors = np.exp(logFactors)
        adjustedVariances = variances / np.maximum(factors[samples], 1.0e-12)
        trend = fitPSplineLogVarianceTrend(
            means,
            adjustedVariances,
            weights=weightsArr,
            eps=eps,
            trendNumBasis=trendNumBasis,
            trendMinObsPerBasis=trendMinObsPerBasis,
            trendMinEdf=trendMinEdf,
            trendMaxEdf=trendMaxEdf,
            trendLambdaMin=trendLambdaMin,
            trendLambdaMax=trendLambdaMax,
            trendLambdaGridSize=trendLambdaGridSize,
        )
        predicted = evalPSplineLogVarianceTrend(trend, means, eps=eps)
        residual = np.log(np.maximum(variances, eps)) - np.log(
            np.maximum(predicted, eps)
        )
        nextLogFactors = np.zeros_like(logFactors)
        for sample in range(sampleCount):
            sampleMask = samples == sample
            nextLogFactors[sample] = _winsorizedMedian(residual[sampleMask])
        nextLogFactors -= float(np.mean(nextLogFactors))
        maxChange = float(np.max(np.abs(nextLogFactors - logFactors)))
        logFactors = nextLogFactors
        itersUsed = iterIndex + 1
        if maxChange < float(tol):
            break

    factors = np.exp(logFactors)
    trend = fitPSplineLogVarianceTrend(
        means,
        variances / np.maximum(factors[samples], 1.0e-12),
        weights=weightsArr,
        eps=eps,
        trendNumBasis=trendNumBasis,
        trendMinObsPerBasis=trendMinObsPerBasis,
        trendMinEdf=trendMinEdf,
        trendMaxEdf=trendMaxEdf,
        trendLambdaMin=trendLambdaMin,
        trendLambdaMax=trendLambdaMax,
        trendLambdaGridSize=trendLambdaGridSize,
    )
    diagnostics = {
        "pooled_pairs": int(means.size),
        "replicate_count": int(sampleCount),
        "factor_min": float(np.min(factors)) if factors.size else 1.0,
        "factor_median": float(np.median(factors)) if factors.size else 1.0,
        "factor_max": float(np.max(factors)) if factors.size else 1.0,
        "iterations": int(itersUsed),
        "max_log_factor_change": float(maxChange),
        "predictor": "signed_log1p",
    }
    return PooledMuncVarianceTrend(
        trend=trend,
        replicateVarianceFactors=factors.astype(np.float64, copy=False),
        diagnostics=diagnostics,
    )


def _formatPSplineTrendSummary(
    trend: PSplineLogVarianceTrend,
    supportSignedMeans: np.ndarray,
    eps: float,
    maxVariance: float | None = None,
    pointCount: int = 9,
) -> str:
    support = np.asarray(supportSignedMeans, dtype=np.float64).ravel()
    support = support[np.isfinite(support)]
    if support.size == 0:
        predictor = np.linspace(float(trend.xMin), float(trend.xMax), int(pointCount))
        support = np.sign(predictor) * np.expm1(
            np.abs(predictor),
        )
    probs = np.linspace(0.0, 1.0, int(max(pointCount, 2)), dtype=np.float64)
    amp = np.unique(np.quantile(support, probs))
    pred = evalPSplineLogVarianceTrend(
        trend,
        amp,
        eps=eps,
        maxVariance=maxVariance,
    )
    predSd = np.sqrt(np.maximum(np.asarray(pred, dtype=np.float64), 0.0))
    pairs = ", ".join(f"{float(a):.4g}->{float(sd):.4g}" for a, sd in zip(amp, predSd))
    diagnostics = getattr(trend, "diagnostics", {})
    beta = getattr(trend, "beta", np.empty(0, dtype=np.float64))
    basisCount = diagnostics.get("num_basis", len(beta))
    requestedBasis = diagnostics.get("requested_num_basis", len(beta))
    return (
        "MUNC P-spline signed-mean-SD trend:\n"
        f"\tlambda={getattr(trend, 'lambdaHat', float('nan')):.4g}\t"
        f"edf={getattr(trend, 'edf', float('nan')):.3g}\t"
        f"basis={basisCount}/{requestedBasis}\n"
        f"\tn_eff={diagnostics.get('trend_n_eff', float('nan')):.1f}\t"
        f"unique_x={diagnostics.get('trend_unique_x', 0)}\t"
        f"edf_cap={diagnostics.get('trend_max_edf', float('nan')):.3g}\t"
        f"lambda_at_boundary={getattr(trend, 'lambdaAtBoundary', False)}\n"
        f"\tsigned_mean->sd[{pairs}]"
    )


def _formatMuncVarianceDiagnostics(
    localVarianceTrack: np.ndarray,
    globalVarianceTrack: np.ndarray,
    finalVarianceTrack: np.ndarray,
    supportAbsMeans: np.ndarray,
) -> str:
    probs = np.asarray([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float64)
    labels = ("p05", "p25", "p50", "p75", "p95")

    def _sdQuantiles(name: str, values: np.ndarray) -> str:
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr) & (arr >= 0.0)]
        if arr.size == 0:
            return f"{name}[empty]"
        sd = np.sqrt(arr)
        quantiles = np.quantile(sd, probs)
        pairs = ",".join(
            f"{label}={float(value):.4g}" for label, value in zip(labels, quantiles)
        )
        return f"{name}[n={arr.size},{pairs}]"

    support = np.asarray(supportAbsMeans, dtype=np.float64).ravel()
    support = support[np.isfinite(support) & (support >= 0.0)]
    if support.size == 0:
        tailSummary = "tail_support(abs_signed_mean)[empty]"
    else:
        tailProbs = np.asarray([0.90, 0.95, 0.99], dtype=np.float64)
        tailVals = np.quantile(support, tailProbs)
        tailParts = [
            f"q{int(prob * 100):02d}={float(value):.4g}:n>={int(np.count_nonzero(support >= value))}"
            for prob, value in zip(tailProbs, tailVals)
        ]
        tailParts.append(f"max={float(np.max(support)):.4g}")
        tailSummary = (
            f"tail_support(abs_signed_mean)[n={support.size},{','.join(tailParts)}]"
        )

    return (
        "MUNC variance SD diagnostics:\n"
        f"\t{_sdQuantiles('L', localVarianceTrack)}\n"
        f"\t{_sdQuantiles('G', globalVarianceTrack)}\n"
        f"\t{_sdQuantiles('V0', finalVarianceTrack)}\n"
        f"\t{tailSummary}"
    )


def _clipVarianceTrack(
    values: np.ndarray,
    floor: float,
    cap: float | None = None,
    fillNaN: bool = True,
) -> np.ndarray:
    floor_ = float(max(float(floor), 1.0e-12))
    cap_ = (
        float(np.finfo(np.float32).max)
        if cap is None or (not np.isfinite(float(cap))) or float(cap) <= floor_
        else float(cap)
    )
    out = np.asarray(values, dtype=np.float64)
    if fillNaN:
        out = np.nan_to_num(out, nan=floor_, posinf=cap_, neginf=floor_)
        np.clip(out, floor_, cap_, out=out)
    else:
        posInfMask = np.isposinf(out)
        negInfMask = np.isneginf(out)
        out[posInfMask] = cap_
        out[negInfMask] = np.nan
        finiteMask = np.isfinite(out)
        out[finiteMask] = np.clip(out[finiteMask], floor_, cap_)
    return out.astype(np.float32)


def _buildSecondDiffPenalty(intervalCount: int) -> sparse.csr_matrix:
    if intervalCount <= 2:
        return sparse.csr_matrix((intervalCount, intervalCount), dtype=np.float64)

    rowCount = intervalCount - 2
    diffMat = sparse.diags(
        [
            np.ones(rowCount, dtype=np.float64),
            -2.0 * np.ones(rowCount, dtype=np.float64),
            np.ones(rowCount, dtype=np.float64),
        ],
        offsets=[0, 1, 2],
        shape=(rowCount, intervalCount),
        format="csr",
        dtype=np.float64,
    )
    return (diffMat.T @ diffMat).tocsr()


def _buildFirstDiffPenalty(intervalCount: int) -> sparse.csr_matrix:
    if intervalCount <= 1:
        return sparse.csr_matrix((intervalCount, intervalCount), dtype=np.float64)

    rowCount = intervalCount - 1
    diffMat = sparse.diags(
        [
            -1.0 * np.ones(rowCount, dtype=np.float64),
            np.ones(rowCount, dtype=np.float64),
        ],
        offsets=[0, 1],
        shape=(rowCount, intervalCount),
        format="csr",
        dtype=np.float64,
    )
    return (diffMat.T @ diffMat).tocsr()


def _backgroundPenaltyWeightsFromSpan(
    blockLenIntervals: int,
    backgroundSmoothness: float = 1.0,
) -> tuple[float, float]:
    spanIntervals = max(2.0, float(blockLenIntervals))
    smoothness = float(backgroundSmoothness)
    firstPenalty = (spanIntervals * spanIntervals) / 4.0
    secondPenalty = (
        spanIntervals * spanIntervals * spanIntervals * spanIntervals
    ) / 16.0
    return (
        float(max(1.0, smoothness * firstPenalty)),
        float(max(1.0, smoothness * secondPenalty)),
    )


def _backgroundPenaltyFromSpan(
    blockLenIntervals: int,
    backgroundSmoothness: float = 1.0,
) -> float:
    return _backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=blockLenIntervals,
        backgroundSmoothness=backgroundSmoothness,
    )[1]


def _coerceOddFilterWindow(windowIntervals: int | float, length: int) -> int:
    r"""Resolve an odd filter window bounded by a one-dimensional track."""

    if length < 3:
        return 0
    try:
        window = int(np.ceil(float(windowIntervals)))
    except (TypeError, ValueError, OverflowError):
        return 0
    if window < 3:
        return 0
    if window % 2 == 0:
        window += 1
    maxWindow = int(length)
    if maxWindow % 2 == 0:
        maxWindow -= 1
    if maxWindow < 3:
        return 0
    return int(min(window, maxWindow))


def quantileFilterDetrendInPlace(
    values: npt.NDArray[np.floating],
    windowIntervals: int | float,
    quantile: float = 0.5,
) -> dict[str, Any]:
    r"""Subtract a broad quantile-filter trend from one transformed replicate track.

    The removed trend is not zero-centered: the operation is exactly
    ``values -= quantile_filter(values, q, window)`` after odd-window coercion.
    With ``quantile=0.5`` this uses ``median_filter`` for the standard median
    trend.
    """

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("values dtype must be floating point")

    quantile_ = float(quantile)
    if not np.isfinite(quantile_) or quantile_ < 0.0 or quantile_ > 1.0:
        raise ValueError("quantile must be between 0 and 1")

    window = _coerceOddFilterWindow(windowIntervals, arr.size)
    if window <= 0:
        return {
            "applied": False,
            "window_intervals": int(window),
            "detrend_quantile": float(quantile_),
            "trend_median": 0.0,
            "trend_quantile": 0.0,
            "trend_mad": 0.0,
        }

    if quantile_ == 0.5:
        trend = ndimage.median_filter(arr, size=int(window), mode="nearest")
    else:
        trend = ndimage.percentile_filter(
            arr,
            percentile=float(100.0 * quantile_),
            size=int(window),
            mode="nearest",
        )
    trend = np.asarray(trend, dtype=arr.dtype)
    if not np.all(np.isfinite(trend)):
        trend = np.nan_to_num(trend, nan=0.0, posinf=0.0, neginf=0.0)

    finite = np.asarray(trend[np.isfinite(trend)], dtype=np.float64)
    if finite.size:
        trendMedian = float(np.median(finite))
        trendQuantile = float(np.quantile(finite, quantile_))
        trendMad = float(np.median(np.abs(finite - trendMedian)))
    else:
        trendMedian = 0.0
        trendQuantile = 0.0
        trendMad = 0.0

    np.subtract(arr, trend, out=arr, casting="unsafe")
    return {
        "applied": True,
        "window_intervals": int(window),
        "detrend_quantile": float(quantile_),
        "trend_median": trendMedian,
        "trend_quantile": trendQuantile,
        "trend_mad": trendMad,
    }

def _finalForwardNIS(NIS: np.ndarray) -> float:
    nisTrack = np.asarray(NIS, dtype=np.float64)
    if nisTrack.ndim != 1:
        raise ValueError("NIS must be one-dimensional")
    finite = nisTrack[np.isfinite(nisTrack)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _finalForwardReplicateGainContigSummary(
    *,
    stateCovarForward: np.ndarray,
    matrixMunc: np.ndarray,
    lambdaExp: np.ndarray | None,
    pad: float,
    obsPrecisionMultiplierMin: float,
    obsPrecisionMultiplierMax: float,
) -> dict[str, np.ndarray]:
    covar = np.asarray(stateCovarForward, dtype=np.float64)
    munc = np.asarray(matrixMunc, dtype=np.float64)
    if covar.ndim != 3 or covar.shape[1:] != (2, 2):
        raise ValueError("stateCovarForward must have shape (n, 2, 2)")
    if munc.ndim != 2:
        raise ValueError("matrixMunc must be two-dimensional")
    if covar.shape[0] != munc.shape[1]:
        raise ValueError("stateCovarForward and matrixMunc interval counts must match")

    p00Forward = np.maximum(covar[:, 0, 0], 0.0)
    obsVariance = np.maximum(munc + float(pad), 1.0e-12)
    if lambdaExp is None:
        obsPrecision = np.ones(covar.shape[0], dtype=np.float64)
    else:
        obsPrecision = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
        if obsPrecision.shape != (covar.shape[0],):
            raise ValueError("lambdaExp length must match interval count")
        obsPrecision = np.clip(
            obsPrecision,
            float(obsPrecisionMultiplierMin),
            float(obsPrecisionMultiplierMax),
        )

    # For row j at interval k, K[j,k] = P[k|k]00 * precision[j,k].
    gains = (p00Forward[None, :] * obsPrecision[None, :]) / obsVariance
    finite = np.isfinite(gains)
    gainSums = np.sum(np.where(finite, gains, 0.0), axis=1, dtype=np.float64)
    gainCounts = np.sum(finite, axis=1, dtype=np.int64)
    gainMeans = np.full(munc.shape[0], np.nan, dtype=np.float64)
    gainMedians = np.full(munc.shape[0], np.nan, dtype=np.float64)
    np.divide(gainSums, gainCounts, out=gainMeans, where=gainCounts > 0)
    for rowIdx in range(munc.shape[0]):
        row = gains[rowIdx, finite[rowIdx, :]]
        if row.size:
            gainMedians[rowIdx] = float(np.median(row))
    return {"mean": gainMeans, "median": gainMedians}


def _sparseSupportWeights(
    sparseIntervalIndices: np.ndarray,
    intervalCount: int,
    ellIntervals: float,
    supportPrior: float,
) -> np.ndarray:
    r"""Compute soft sparse support weights from exponential distance decay.

    ``n_eff[i] = sum_j exp(-abs(i - sparse_j) / ell)`` is evaluated exactly in
    linear time using left/right recurrences over the sorted sparse interval
    indices, then converted to ``n_eff / (n_eff + supportPrior)``.
    """
    intervalCount = int(intervalCount)
    if intervalCount <= 0:
        return np.empty(0, dtype=np.float32)

    sparseIdx = np.asarray(sparseIntervalIndices, dtype=np.intp).ravel()
    sparseIdx = sparseIdx[(sparseIdx >= 0) & (sparseIdx < intervalCount)]
    if sparseIdx.size == 0:
        return np.zeros(intervalCount, dtype=np.float32)
    sparseIdx = np.unique(sparseIdx)

    ellIntervals = float(ellIntervals)
    if (not np.isfinite(ellIntervals)) or ellIntervals <= 0.0:
        weights = np.zeros(intervalCount, dtype=np.float32)
        weights[sparseIdx] = 1.0
        return weights

    counts = np.zeros(intervalCount, dtype=np.float32)
    counts[sparseIdx] = 1.0

    decay = float(np.exp(-1.0 / ellIntervals))
    left = np.empty(intervalCount, dtype=np.float32)
    running = 0.0
    for i in range(intervalCount):
        running = (running * decay) + counts[i]
        left[i] = running

    right = np.empty(intervalCount, dtype=np.float32)
    running = 0.0
    for i in range(intervalCount - 1, -1, -1):
        running = (running * decay) + counts[i]
        right[i] = running

    nEff = left + right - counts
    supportPrior = float(supportPrior)
    if (not np.isfinite(supportPrior)) or supportPrior <= 0.0:
        weights = np.zeros(intervalCount, dtype=np.float32)
        weights[nEff > 0.0] = 1.0
        return weights

    weights = nEff / (nEff + supportPrior)
    weights[~np.isfinite(weights)] = 0.0
    return weights.astype(np.float32, copy=False)


def _solveZeroCenteredBackground(
    residualMatrix: np.ndarray,
    invVarMatrix: np.ndarray,
    blockLenIntervals: int,
    backgroundSmoothness: float = 1.0,
    zeroCenter: bool = True,
) -> npt.NDArray[np.float32]:
    residualArr = np.asarray(residualMatrix, dtype=np.float32)
    invVarArr = np.asarray(invVarMatrix, dtype=np.float32)
    if residualArr.ndim != 2 or invVarArr.shape != residualArr.shape:
        raise ValueError(
            "residualMatrix and invVarMatrix must have identical 2D shapes"
        )

    intervalCount = int(residualArr.shape[1])
    if intervalCount < 1:
        return np.zeros(0, dtype=np.float32)

    weightTrack = np.sum(invVarArr, axis=0, dtype=np.float64)
    rhsTrack = np.einsum(
        "ij,ij->j",
        invVarArr,
        residualArr,
        dtype=np.float64,
    )
    if not np.any(weightTrack > 0.0):
        return np.zeros(intervalCount, dtype=np.float32)

    lamFirst, lamSecond = _backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=blockLenIntervals,
        backgroundSmoothness=backgroundSmoothness,
    )
    return cconsenrich.csolveZeroCenteredBackground(
        np.ascontiguousarray(weightTrack, dtype=np.float64),
        np.ascontiguousarray(rhsTrack, dtype=np.float64),
        float(lamSecond),
        bool(zeroCenter),
        lamFirst=float(lamFirst),
    )


def _coerceEBPriorStrength(value: float | int | None) -> float | None:
    if value is None:
        return None
    nu0 = float(value)
    if not np.isfinite(nu0) or nu0 < 4.0:
        return None
    return nu0


def getMuncTrack(
    chromosome: str,
    intervals: np.ndarray,
    values: np.ndarray,
    intervalSizeBP: int,
    samplingBlockSizeBP: int | None = None,
    samplingIters: int = 25_000,
    randomSeed: int = 42,
    excludeMask: Optional[np.ndarray] = None,
    useEMA: Optional[bool] = True,
    excludeFitCoefs: Optional[Tuple[int, ...]] = None,
    EB_use: bool = True,
    EB_setNu0: int | None = None,
    EB_setNuL: int | None = None,
    trendNumBasis: int = 60,
    trendMinObsPerBasis: float = 25.0,
    trendMinEdf: float = 3.0,
    trendMaxEdf: float | None = 30.0,
    trendLambdaMin: float = 1.0e-6,
    trendLambdaMax: float = 1.0e6,
    trendLambdaGridSize: int = 41,
    sparseIntervalIndices: Optional[np.ndarray] = None,
    sparseRegionMask: Optional[np.ndarray] = None,
    numNearest: int = 0,
    sparseSupportScaleBP: Optional[float] = None,
    sparseSupportPrior: float = 1.0,
    restrictLocalAR1ToSparseBed: bool = False,
    EB_localQuantile: float = 0.0,
    verbose: bool = False,
    eps: float = 1.0e-2,
    varianceFloor: float | None = None,
    varianceCap: float | None = None,
    intervalsArr: Optional[np.ndarray] = None,
    excludeMaskArr: Optional[np.ndarray] = None,
    pooledTrend: Optional[PSplineLogVarianceTrend] = None,
    replicateVarianceFactor: float = 1.0,
    EB_pooledNu0: float | None = None,
) -> tuple[npt.NDArray[np.float32], float]:
    r"""Approximate initial sample-specific (**M**)easurement (**unc**)ertainty tracks

    For an individual experimental sample (replicate), quantify *positional* observation noise levels over genomic intervals :math:`i=1,2,\ldots n` spanning ``chromosome``.
    These tracks (per-sample) comprise the ``matrixMunc`` input to :func:`runConsenrich`, :math:`\mathbf{R}[:,:] \in \mathbb{R}^{m \times n}`.

    Variance is modeled as a function of a signed mean signal predictor. For ``EB_use=True``, local variance estimates are shrunk toward a signal level dependent global variance fit.

    :param chromosome: chromosome/contig name
    :type chromosome: str
    :param values: normalized/transformed signal measurements over genomic intervals (e.g., :func:`consenrich.cconsenrich.cTransform` output)
    :type values: np.ndarray
    :param intervals: genomic intervals positions (start positions)
    :type intervals: np.ndarray

    See :class:`consenrich.core.observationParams` for other parameters.

    """

    # Retained for compatibility with older callers; current fitting uses masks directly.
    _ = excludeFitCoefs

    AR1_PARAMCT = 3  # intercept, AR(1) coefficient, variance
    varianceFloor_ = float(max(eps, varianceFloor or eps, 1.0e-12))
    varianceCap_ = (
        None
        if varianceCap is None
        or (not np.isfinite(float(varianceCap)))
        or float(varianceCap) <= varianceFloor_
        else float(varianceCap)
    )
    if samplingBlockSizeBP is None:
        samplingBlockSizeBP = intervalSizeBP * (11 * (AR1_PARAMCT))
    blockSizeIntervals = int(samplingBlockSizeBP / intervalSizeBP)

    localWindowIntervals = max(4, (blockSizeIntervals + 1))
    if intervalsArr is None:
        intervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32).reshape(-1)
    else:
        intervalsArr = np.ascontiguousarray(intervalsArr, dtype=np.uint32).reshape(-1)
    valuesArr = np.ascontiguousarray(values, dtype=np.float32)
    if intervalsArr.shape[0] != valuesArr.size:
        raise ValueError("intervalsArr must match values length")

    if excludeMask is None:
        if excludeMaskArr is None:
            excludeMaskArr = np.zeros_like(intervalsArr, dtype=np.uint8)
        else:
            excludeMaskArr = np.ascontiguousarray(
                excludeMaskArr,
                dtype=np.uint8,
            ).reshape(-1)
    else:
        excludeMaskArr = np.ascontiguousarray(excludeMask, dtype=np.uint8).reshape(-1)
    if excludeMaskArr.shape != intervalsArr.shape:
        raise ValueError("excludeMaskArr must match intervals/values length")

    _logAsciiBlock(
        "MUNC track start",
        (
            ("chromosome", chromosome),
            ("intervals", int(valuesArr.size)),
            ("interval size bp", int(intervalSizeBP)),
            ("sampling block bp", int(samplingBlockSizeBP)),
            ("sampling block intervals", int(blockSizeIntervals)),
            ("local window intervals", int(localWindowIntervals)),
            ("MUNC variance EB", "enabled" if EB_use else "disabled"),
            (
                "MUNC trend source",
                (
                    "pooled signed trend"
                    if pooledTrend is not None
                    else "fit sample trend"
                ),
            ),
        ),
    )

    localObsExcludeMaskArr = excludeMaskArr
    if restrictLocalAR1ToSparseBed:
        if sparseRegionMask is None:
            logger.warning(
                "restrictLocalAR1ToSparseBed=True but sparseRegionMask is not provided; "
                "using the unrestricted rolling AR(1) local observation-variance estimate.",
            )
        else:
            sparseRegionMaskArr = np.ascontiguousarray(
                sparseRegionMask,
                dtype=np.uint8,
            ).ravel()
            if sparseRegionMaskArr.shape != excludeMaskArr.shape:
                raise ValueError(
                    "sparseRegionMask must match the shape of intervals/values.",
                )
            localObsExcludeMaskArr = np.ascontiguousarray(
                np.logical_or(
                    excludeMaskArr != 0,
                    sparseRegionMaskArr == 0,
                ),
                dtype=np.uint8,
            )

    def _estimateSparseNearestObsTracks() -> (
        tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ):
        if sparseIntervalIndices is None or int(numNearest) <= 0:
            return None

        sparseIdx = np.asarray(sparseIntervalIndices, dtype=np.intp).ravel()
        if sparseIdx.size == 0:
            return None

        sparseIdx = np.unique(sparseIdx)
        sparseIdx = sparseIdx[(sparseIdx >= 0) & (sparseIdx < valuesArr.size)]
        if sparseIdx.size == 0:
            return None

        blockLen = int(max(4, min(localWindowIntervals, valuesArr.size)))
        if blockLen < 2:
            return None

        blockStarts = np.empty(sparseIdx.shape, dtype=np.intp)
        blockSizes = np.empty(sparseIdx.shape, dtype=np.intp)
        retainedCenters = np.empty(sparseIdx.shape, dtype=np.intp)
        retainedCount = 0
        runBreaks = np.concatenate(
            (
                np.array([0], dtype=np.intp),
                np.flatnonzero(np.diff(sparseIdx) > 1).astype(np.intp) + 1,
                np.array([sparseIdx.size], dtype=np.intp),
            )
        )
        for runPos in range(runBreaks.size - 1):
            runLo = int(runBreaks[runPos])
            runHi = int(runBreaks[runPos + 1])
            runStart = int(sparseIdx[runLo])
            runEnd = int(sparseIdx[runHi - 1]) + 1
            runLen = runEnd - runStart
            if runLen < 4:
                continue
            runBlockLen = min(blockLen, runLen)
            halfLen = runBlockLen // 2
            for center in sparseIdx[runLo:runHi]:
                blockStart = int(center) - halfLen
                if blockStart < runStart:
                    blockStart = runStart
                if blockStart + runBlockLen > runEnd:
                    blockStart = runEnd - runBlockLen
                retainedCenters[retainedCount] = int(center)
                blockStarts[retainedCount] = blockStart
                blockSizes[retainedCount] = runBlockLen
                retainedCount += 1

        if retainedCount == 0:
            return None
        sparseIdx = retainedCenters[:retainedCount]
        blockStarts = blockStarts[:retainedCount]
        blockSizes = blockSizes[:retainedCount]

        if np.any(excludeMaskArr):
            excludeCum = np.concatenate(
                (
                    np.array([0], dtype=np.int64),
                    np.cumsum(excludeMaskArr.astype(np.int64, copy=False)),
                )
            )
            blockEnds = blockStarts + blockSizes
            validMask = (excludeCum[blockEnds] - excludeCum[blockStarts]) == 0
            blockStarts = blockStarts[validMask]
            blockSizes = blockSizes[validMask]
            sparseIdx = sparseIdx[validMask]
            if sparseIdx.size == 0:
                return None

        sparseMeanTrack, sparseVarTrack = cconsenrich.cSparseNearestMeanVarTrack(
            valuesArr,
            np.ascontiguousarray(sparseIdx, dtype=np.intp),
            np.ascontiguousarray(blockStarts, dtype=np.intp),
            np.ascontiguousarray(blockSizes, dtype=np.intp),
            int(numNearest),
            useInnovationVar=False,
            aggregateMeanAbs=False,
        )
        sparseMeanTrack = np.asarray(sparseMeanTrack, dtype=np.float32)
        sparseVarTrack = np.asarray(sparseVarTrack, dtype=np.float32)
        sparseMeanTrack[~np.isfinite(sparseMeanTrack)] = 0.0
        sparseVarTrack[~np.isfinite(sparseVarTrack)] = np.nan
        return sparseMeanTrack, sparseVarTrack, sparseIdx

    sparseInterceptTrack: np.ndarray | None = None
    sparseObsVarTrack: np.ndarray | None = None
    sparseSupportWeightTrack: np.ndarray | None = None
    valuesForPriorFitArr = valuesArr
    sparseObsTracks = _estimateSparseNearestObsTracks()
    if sparseObsTracks is not None:
        sparseInterceptTrack, sparseObsVarTrack, sparseSupportIdx = sparseObsTracks
        if sparseSupportScaleBP is None or float(sparseSupportScaleBP) <= 0.0:
            ellIntervals = float(localWindowIntervals)
        else:
            ellIntervals = max(
                1.0,
                float(sparseSupportScaleBP) / float(intervalSizeBP),
            )
        sparseSupportWeightTrack = _sparseSupportWeights(
            sparseSupportIdx,
            valuesArr.size,
            ellIntervals,
            float(sparseSupportPrior),
        )
        sparseInterceptTrack = sparseInterceptTrack * sparseSupportWeightTrack
        valuesForPriorFitArr = valuesArr - sparseInterceptTrack
        if verbose:
            logger.info(
                "Sparse-nearest support: ell=%.2f intervals, median weight=%.4f, max weight=%.4f",
                ellIntervals,
                float(np.median(sparseSupportWeightTrack)),
                float(np.max(sparseSupportWeightTrack)),
            )

    supportFraction = 1.0
    if pooledTrend is None:
        _logAsciiBlock(
            "MUNC trend fit",
            (
                ("chromosome", chromosome),
                ("sampling iterations", int(samplingIters)),
                ("basis requested", int(trendNumBasis)),
                ("lambda grid size", int(trendLambdaGridSize)),
                ("excluded intervals", int(np.count_nonzero(excludeMaskArr))),
            ),
        )
        # Global:
        # ... Variance as a function of signed mean, globally, as observed in
        # ... distinct, randomly drawn genomic blocks. Within fixed-size blocks,
        # ... an AR(1) fit captures local autocorrelation, while the
        # ... stationary/marginal AR(1) variance is used as the diagonal
        # ... observation-variance target.
        blockMeans, blockVars, _starts, _ends = cconsenrich.cmeanVarPairs(
            intervalsArr,
            np.ascontiguousarray(valuesForPriorFitArr, dtype=np.float32),
            blockSizeIntervals,
            samplingIters,
            randomSeed,
            excludeMaskArr,
            useInnovationVar=False,
        )
        mask = np.isfinite(blockMeans) & np.isfinite(blockVars) & (blockVars >= 1.0e-3)
        supportFraction = (
            float(np.sum(mask)) / float(len(blockMeans)) if len(blockMeans) else 0.0
        )
        means_Masked = blockMeans[mask]
        var_Masked = blockVars[mask]
        order = np.argsort(_muncTrendPredictor(means_Masked))
        means_Sorted = means_Masked[order]
        var_Sorted = var_Masked[order]
        opt = fitPSplineLogVarianceTrend(
            means_Sorted,
            var_Sorted,
            eps=eps,
            trendNumBasis=trendNumBasis,
            trendMinObsPerBasis=trendMinObsPerBasis,
            trendMinEdf=trendMinEdf,
            trendMaxEdf=trendMaxEdf,
            trendLambdaMin=trendLambdaMin,
            trendLambdaMax=trendLambdaMax,
            trendLambdaGridSize=trendLambdaGridSize,
        )
    else:
        _logAsciiBlock(
            "MUNC pooled trend reuse",
            (
                ("chromosome", chromosome),
                ("intervals", int(valuesArr.size)),
                ("replicate variance factor", float(replicateVarianceFactor)),
            ),
        )
        opt = pooledTrend
        finiteValues = np.asarray(valuesForPriorFitArr, dtype=np.float64).ravel()
        means_Sorted = finiteValues[np.isfinite(finiteValues)]
        if means_Sorted.size == 0:
            means_Sorted = np.array([0.0], dtype=np.float64)
    logger.info(
        _formatPSplineTrendSummary(
            opt,
            means_Sorted,
            eps=varianceFloor_,
            maxVariance=varianceCap_,
        )
    )

    meanTrack = np.ascontiguousarray(valuesForPriorFitArr, dtype=np.float32)
    if useEMA:
        meanTrack = cconsenrich.cEMA(meanTrack, 2 / (localWindowIntervals + 1))
    priorTrack = evalPSplineLogVarianceTrend(
        opt,
        meanTrack,
        eps=varianceFloor_,
        maxVariance=varianceCap_,
    )
    priorFactor = float(replicateVarianceFactor)
    if not np.isfinite(priorFactor) or priorFactor <= 0.0:
        priorFactor = 1.0
    priorTrack = priorTrack * np.float32(priorFactor)
    priorTrack = _clipVarianceTrack(
        priorTrack,
        floor=varianceFloor_,
        cap=varianceCap_,
    )

    if not EB_use:
        _logAsciiBlock(
            "MUNC EB shrinkage skipped",
            (
                ("chromosome", chromosome),
                ("reason", "MUNC variance EB disabled"),
                ("support fraction", float(supportFraction)),
            ),
        )
        return priorTrack.astype(np.float32, copy=False), float(supportFraction)

    # Local:
    # ... default: rolling AR(1) marginal variance over a sliding window
    # ... optional sparse-bed restriction: invalidate any local window leaving sparse regions
    # ... sparse-nearest mode: aggregate region mean/variance stats at the nearest sparse blocks
    fallbackObsVarTrack = cconsenrich.crolling_AR1_IVar(
        valuesArr,
        localWindowIntervals,
        localObsExcludeMaskArr,
        useInnovationVar=False,
    ).astype(np.float32, copy=False)
    fallbackObsVarTrack[fallbackObsVarTrack < 0.0] = np.nan
    fallbackObsVarTrack = _clipVarianceTrack(
        fallbackObsVarTrack,
        floor=varianceFloor_,
        cap=varianceCap_,
        fillNaN=False,
    )

    if sparseObsVarTrack is not None:
        sparseObsVarTrack = sparseObsVarTrack.astype(np.float32, copy=False)
        sparseObsVarTrack[sparseObsVarTrack < 0.0] = np.nan
        sparseObsVarTrack = _clipVarianceTrack(
            sparseObsVarTrack,
            floor=varianceFloor_,
            cap=varianceCap_,
            fillNaN=False,
        )
        if sparseSupportWeightTrack is None:
            sparseSupportWeightTrack = np.ones_like(sparseObsVarTrack, dtype=np.float32)
        supportWeight = np.asarray(sparseSupportWeightTrack, dtype=np.float32)
        supportWeight = np.clip(supportWeight, 0.0, 1.0)

        obsVarTrack = np.array(fallbackObsVarTrack, dtype=np.float32, copy=True)
        finSparse = np.isfinite(sparseObsVarTrack)
        finFallback = np.isfinite(fallbackObsVarTrack)
        finBoth = finSparse & finFallback
        obsVarTrack[finBoth] = (
            supportWeight[finBoth] * sparseObsVarTrack[finBoth]
            + (1.0 - supportWeight[finBoth]) * fallbackObsVarTrack[finBoth]
        )
        sparseOnly = finSparse & ~finFallback
        obsVarTrack[sparseOnly] = sparseObsVarTrack[sparseOnly]
    else:
        obsVarTrack = fallbackObsVarTrack

    # Note, negative values are a flag from `cconsenrich.crolling_AR1_IVar`
    # ... -- set as _NaN_ -- and handle later during shrinkage
    obsVarTrack[obsVarTrack < 0.0] = np.nan

    # Optionally, run a quantile filter over the local variance track.
    # ...     EB_localQuantile < 0 --> disable
    # ...     EB_localQuantile == 0 --> median filter
    # ...     EB_localQuantile > 0 --> use supplied quantile value (x100)
    # ... NOTE: Useful heuristic for tempering spurious measurements in sparse genomic
    # ...    regions where estimated noise levels are often artificially deflated.
    if EB_localQuantile >= 0.0:
        quantile_ = 0.5 if EB_localQuantile == 0.0 else float(EB_localQuantile)
        if quantile_ < 0.0:
            quantile_ = 0.0
        elif quantile_ > 1.0:
            quantile_ = 1.0
        pct = 100.0 * quantile_
        win = int(localWindowIntervals)
        if win < 1:
            win = 1
        if (win & 1) == 0:
            win += 1

        # inf sentinel for NaN positions
        fillVal = np.inf if quantile_ >= 0.5 else -np.inf
        nanMask = ~np.isfinite(obsVarTrack)
        if np.any(nanMask):
            tmp = obsVarTrack.copy()
            tmp[nanMask] = fillVal
            ndimage.percentile_filter(
                tmp,
                size=win + 2,
                percentile=pct,
                mode="nearest",
                output=tmp,
            )
            # immediately after, replace sentinel inf --> NaN
            tmp[nanMask] = np.nan
            tmp[~np.isfinite(tmp)] = np.nan
            obsVarTrack = _clipVarianceTrack(
                tmp + varianceFloor_,
                floor=varianceFloor_,
                cap=varianceCap_,
                fillNaN=False,
            )
        else:
            ndimage.percentile_filter(
                obsVarTrack + varianceFloor_,
                size=win + 2,
                percentile=pct,
                mode="nearest",
                output=obsVarTrack,
            )
            obsVarTrack = _clipVarianceTrack(
                obsVarTrack,
                floor=varianceFloor_,
                cap=varianceCap_,
                fillNaN=False,
            )

    # df / effective sample size for local variance
    if EB_setNuL is not None and EB_setNuL > 3:
        Nu_L = float(EB_setNuL)
        logger.info(f"Using fixed/specified Nu_L={Nu_L:.2f}")
    else:
        Nu_L = float(max(4, localWindowIntervals - 3))

    # --- Determine prior strength ---
    minScale_prior: float | None = None
    minScale_obs: float | None = None
    finMask_obs2: Optional[np.ndarray] = None
    finMask_prior2: Optional[np.ndarray] = None
    finMask_both2: Optional[np.ndarray] = None

    specifiedNu0 = _coerceEBPriorStrength(EB_setNu0)
    pooledNu0 = _coerceEBPriorStrength(EB_pooledNu0)

    if specifiedNu0 is not None:
        # check if Nu_0 is specified before computing
        Nu_0 = specifiedNu0
        logger.info(f"Using fixed/specified Nu_0={Nu_0:.2f}")
    elif pooledNu0 is not None:
        Nu_0 = pooledNu0
        logger.info(f"Using pooled Nu_0={Nu_0:.2f}")
    else:
        # finite/non-zero mask _BEFORE_ Nu_0 fit
        priorFinite = priorTrack[np.isfinite(priorTrack)]
        obsFinite = obsVarTrack[np.isfinite(obsVarTrack)]
        medPrior = float(np.median(priorFinite)) if priorFinite.size else 0.0
        medObs = float(np.median(obsFinite)) if obsFinite.size else 0.0

        minScale_prior = (1.0e-2 * medPrior) + 1.0e-4
        minScale_obs = (1.0e-2 * medObs) + 1.0e-4

        finMask_obs = np.isfinite(obsVarTrack) & (obsVarTrack > minScale_obs)
        finMask_prior = np.isfinite(priorTrack) & (priorTrack > minScale_prior)
        finMask_both = finMask_obs & finMask_prior

        # only pass matched finite pairs into EB_computePriorStrength
        if np.count_nonzero(finMask_both) < 4:
            logger.warning(
                f"Insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
            )
            Nu_0 = float(1.0e6)
        else:
            Nu_0 = EB_computePriorStrength(
                obsVarTrack,
                priorTrack,
                Nu_L,
                thinStride=max(localWindowIntervals, blockSizeIntervals, 1),
            )

        # reuse masks during shrinkage (no need to recompute)
        finMask_obs2 = finMask_obs
        finMask_prior2 = finMask_prior
        finMask_both2 = finMask_both

    Nu_0_cap = 50.0 * float(Nu_L)
    if np.isfinite(Nu_0_cap) and Nu_0 > Nu_0_cap:
        logger.info(
            "Capping Nu_0=%.2f at 50*Nu_L=%.2f",
            float(Nu_0),
            float(Nu_0_cap),
        )
        Nu_0 = float(Nu_0_cap)

    _logAsciiBlock(
        "MUNC EB shrinkage",
        (
            ("chromosome", chromosome),
            ("Nu_0", float(Nu_0)),
            ("Nu_L", float(Nu_L)),
            ("posterior sample size", float(Nu_L + Nu_0)),
            ("support fraction", float(supportFraction)),
            ("local quantile", float(EB_localQuantile)),
        ),
    )
    logger.info("MUNC EB shrinkage:\n\tNu_0=%.2f\n\tNu_L=%.2f", Nu_0, Nu_L)
    posteriorSampleSize: float = Nu_L + Nu_0

    # --- Shrinkage ---
    posteriorVarTrack = np.array(priorTrack, dtype=np.float32, copy=True)

    # check if bounds/masks already exist (i.e., computed during Nu_0 fit), reuse them
    # ... otherwise compute them for the first time here
    if finMask_both2 is None:
        if minScale_prior is None or minScale_obs is None:
            priorFinite2 = posteriorVarTrack[np.isfinite(posteriorVarTrack)]
            obsFinite2 = obsVarTrack[np.isfinite(obsVarTrack)]
            medPrior2 = float(np.median(priorFinite2)) if priorFinite2.size else 0.0
            medObs2 = float(np.median(obsFinite2)) if obsFinite2.size else 0.0

            minScale_prior = (1.0e-2 * medPrior2) + 1.0e-4
            minScale_obs = (1.0e-2 * medObs2) + 1.0e-4

        finMask_obs2 = np.isfinite(obsVarTrack) & (obsVarTrack > minScale_obs)
        finMask_prior2 = np.isfinite(posteriorVarTrack) & (
            posteriorVarTrack > minScale_prior
        )
        finMask_both2 = finMask_obs2 & finMask_prior2

    # Case: both prior and obs yield meaningful estimates --> proper shrinkage
    posteriorVarTrack[finMask_both2] = (
        (
            Nu_L * obsVarTrack[finMask_both2].astype(np.float64)
            + Nu_0 * posteriorVarTrack[finMask_both2].astype(np.float64)
        )
        / posteriorSampleSize
    ).astype(np.float32)

    # Case: prior is missing but obs value is valid --> use the local estimate
    # ... (shouldn't really happen, but JIC for completeness)
    finMask_onlyObs2 = finMask_obs2 & ~finMask_prior2
    if np.count_nonzero(finMask_onlyObs2) > 0:
        logger.warning(
            f"{np.count_nonzero(finMask_onlyObs2)} intervals with _only_ local variance information...using local estimate.",
        )
        posteriorVarTrack[finMask_onlyObs2] = obsVarTrack[finMask_onlyObs2]

    # Case: Neither present --> assign NaN
    # ... again, shouldn't happen
    finMask_neither2 = ~finMask_obs2 & ~finMask_prior2
    if np.count_nonzero(finMask_neither2) > 0:
        logger.warning(
            f"{np.count_nonzero(finMask_neither2)} intervals with _neither_ local nor prior variance information...setting as NaN (!!!)",
        )
        posteriorVarTrack[finMask_neither2] = np.nan

    posteriorVarTrack = _clipVarianceTrack(
        posteriorVarTrack,
        floor=varianceFloor_,
        cap=varianceCap_,
    )

    logger.info(
        _formatMuncVarianceDiagnostics(
            obsVarTrack,
            priorTrack,
            posteriorVarTrack,
            np.abs(means_Sorted),
        )
    )

    if verbose:
        logger.info(
            f"Median variance after shrinkage: {float(np.nanmedian(posteriorVarTrack)):.4f}",
        )

    return posteriorVarTrack.astype(np.float32, copy=False), float(supportFraction)


def _computePriorStrengthFromCandidateIdx(
    localModelVariancesArr: np.ndarray,
    globalModelVariancesArr: np.ndarray,
    Nu_local: float,
    candidateIdx: np.ndarray,
) -> float:
    varRatioArr = (
        localModelVariancesArr[candidateIdx] / globalModelVariancesArr[candidateIdx]
    )
    varRatioArr = varRatioArr[np.isfinite(varRatioArr) & (varRatioArr > 0.0)]
    if varRatioArr.size < 4:
        logger.warning(
            f"After masking, insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
        )
        return float(1.0e6)

    logVarRatioArr = np.log(varRatioArr)
    if logVarRatioArr.size >= 20:
        clipSmall = np.quantile(logVarRatioArr, 0.01)
        clipBig = np.quantile(logVarRatioArr, 0.99)
        np.clip(logVarRatioArr, clipSmall, clipBig, out=logVarRatioArr)

    varLogVarRatio = float(np.var(logVarRatioArr, ddof=1))
    trigammaLocal = float(special.polygamma(1, float(Nu_local) / 2.0))
    # inverse trigamma --> inf near 0
    gap = max(varLogVarRatio - trigammaLocal, 1.0e-6)
    Nu_0 = 2.0 * itrigamma(gap)
    if Nu_0 < 4.0:
        Nu_0 = 4.0

    return float(Nu_0)


def EB_computePriorStrength(
    localModelVariances: np.ndarray,
    globalModelVariances: np.ndarray,
    Nu_local: float,
    thinStride: int = 1,
) -> float:
    r"""Compute :math:`\nu_0` to determine 'prior strength'

    The prior model strength is determined by 'excess' dispersion beyond sampling noise at the local level.

    :param localModelVariances: Local model variance estimates (e.g., rolling AR(1) marginal variances :func:`consenrich.cconsenrich.crolling_AR1_IVar`).
    :type localModelVariances: np.ndarray
    :param globalModelVariances: Global model variance estimates from the signed mean-variance trend fit (:func:`consenrich.core.fitPSplineLogVarianceTrend`).
    :type globalModelVariances: np.ndarray
    :param Nu_local: Effective sample size/degrees of freedom for the local model.
    :type Nu_local: float
    :param thinStride: Deterministic interval stride used to thin serially dependent local/global pairs before moment matching.
    :type thinStride: int
    :return: Estimated prior strength :math:`\nu_{0}`.
    :rtype: float

    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitPSplineLogVarianceTrend`
    """

    localModelVariancesArr = np.asarray(localModelVariances, dtype=np.float64).ravel()
    globalModelVariancesArr = np.asarray(globalModelVariances, dtype=np.float64).ravel()
    if localModelVariancesArr.shape != globalModelVariancesArr.shape:
        raise ValueError(
            "localModelVariances and globalModelVariances must have the same shape"
        )

    ratioMask = (
        np.isfinite(localModelVariancesArr)
        & np.isfinite(globalModelVariancesArr)
        & (localModelVariancesArr > 0.0)
        & (globalModelVariancesArr > 0.0)
    )
    candidateIdx = np.flatnonzero(ratioMask)
    if candidateIdx.size < max(4, int(np.ceil((0.10) * localModelVariancesArr.size))):
        logger.warning(
            f"Insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
        )
        return float(1.0e6)

    stride = max(int(thinStride or 1), 1)
    if stride > 1:
        phases = candidateIdx % stride
        phaseCounts = np.bincount(phases, minlength=stride)
        bestPhase = int(np.argmax(phaseCounts))
        candidateIdx = candidateIdx[phases == bestPhase]

    if candidateIdx.size < 4:
        logger.warning(
            f"After thinning, insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
        )
        return float(1.0e6)

    return _computePriorStrengthFromCandidateIdx(
        localModelVariancesArr,
        globalModelVariancesArr,
        Nu_local,
        candidateIdx,
    )


def EB_computePooledPriorStrength(
    localModelVariances: np.ndarray,
    globalModelVariances: np.ndarray,
    Nu_local: float,
    sampleIndex: np.ndarray | None = None,
    chromosomeIndex: np.ndarray | None = None,
    blockStarts: np.ndarray | None = None,
    thinBinSize: int = 1,
) -> float:
    r"""Compute pooled :math:`\nu_0` using deterministic sample/chromosome/bin thinning."""

    localArr = np.asarray(localModelVariances, dtype=np.float64).ravel()
    globalArr = np.asarray(globalModelVariances, dtype=np.float64).ravel()
    if localArr.shape != globalArr.shape:
        raise ValueError("localModelVariances and globalModelVariances must align")
    ratioMask = (
        np.isfinite(localArr)
        & np.isfinite(globalArr)
        & (localArr > 0.0)
        & (globalArr > 0.0)
    )
    candidateIdx = np.flatnonzero(ratioMask)
    if candidateIdx.size < max(4, int(np.ceil(0.10 * localArr.size))):
        logger.warning(
            "Insufficient pooled prior/local variance pairs...setting Nu_0 = 1.0e6"
        )
        return float(1.0e6)

    if (
        sampleIndex is not None
        and chromosomeIndex is not None
        and blockStarts is not None
    ):
        samples = np.asarray(sampleIndex, dtype=np.int64).ravel()
        chromosomes = np.asarray(chromosomeIndex, dtype=np.int64).ravel()
        starts = np.asarray(blockStarts, dtype=np.int64).ravel()
        if (
            samples.shape != localArr.shape
            or chromosomes.shape != localArr.shape
            or starts.shape != localArr.shape
        ):
            raise ValueError("sampleIndex, chromosomeIndex, and blockStarts must align")
        binSize = max(int(thinBinSize or 1), 1)
        seen: set[tuple[int, int, int]] = set()
        thinned: list[int] = []
        for idx in candidateIdx:
            key = (
                int(samples[idx]),
                int(chromosomes[idx]),
                int(starts[idx] // binSize),
            )
            if key in seen:
                continue
            seen.add(key)
            thinned.append(int(idx))
        candidateIdx = np.asarray(thinned, dtype=np.intp)

    if candidateIdx.size < 4:
        logger.warning(
            "After pooled thinning, insufficient pairs...setting Nu_0 = 1.0e6"
        )
        return float(1.0e6)

    return _computePriorStrengthFromCandidateIdx(
        localArr,
        globalArr,
        Nu_local,
        candidateIdx,
    )


def _estimateLocalPeakWidth(
    yVals: np.ndarray,
    peakIdx: int,
    peakSearchRadius: int = 2,
) -> float | None:
    n = int(yVals.size)
    if n < 3:
        return None

    leftIdx = max(0, int(peakIdx) - int(max(0, peakSearchRadius)))
    rightIdx = min(n, int(peakIdx) + int(max(0, peakSearchRadius)) + 1)
    if rightIdx - leftIdx < 1:
        return None

    localPeakIdx = leftIdx + int(np.argmax(yVals[leftIdx:rightIdx]))
    if localPeakIdx <= 0 or localPeakIdx >= (n - 1):
        return None

    peakHeight = float(yVals[localPeakIdx])
    if not np.isfinite(peakHeight):
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        widths, _, _, _ = signal.peak_widths(
            yVals,
            np.array([localPeakIdx], dtype=np.int64),
            rel_height=0.5,
        )

    if widths.size == 0:
        return None

    widthHat = float(widths[0])
    if not np.isfinite(widthHat) or widthHat <= 0.0:
        return None

    return float(max(1.0, widthHat))


def _bootstrapLocalPeakLogWidthVariance(
    yVals: np.ndarray,
    peakIdx: int,
    halfWindow: int,
    numBoot: int = 16,
) -> tuple[float | None, float | None]:
    n = int(yVals.size)
    if n < 8:
        return None, None

    localLeft = max(0, int(peakIdx) - int(max(2, halfWindow)))
    localRight = min(n, int(peakIdx) + int(max(2, halfWindow)) + 1)
    localY = np.asarray(yVals[localLeft:localRight], dtype=np.float64)
    localPeakIdx = int(peakIdx) - localLeft
    widthHat = _estimateLocalPeakWidth(localY, localPeakIdx, peakSearchRadius=2)
    if widthHat is None:
        return None, None

    smoothSize = int(max(3, min(localY.size - (1 - (localY.size % 2)), 9)))
    if smoothSize % 2 == 0:
        smoothSize -= 1
    smoothSize = max(3, smoothSize)
    if smoothSize >= localY.size:
        smoothSize = localY.size - 1 if localY.size % 2 == 0 else localY.size
    smoothSize = max(3, smoothSize)
    localFit = signal.savgol_filter(
        localY,
        window_length=smoothSize,
        polyorder=min(2, smoothSize - 1),
        mode="interp",
    )
    residuals = np.asarray(localY - localFit, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size < 4:
        return float(np.log(widthHat)), 1.0e-6

    residuals = residuals - float(np.mean(residuals))
    rng = np.random.default_rng(
        int((int(peakIdx) + 1) * 104729 + (halfWindow + 1) * 1009 + n)
    )
    bootLogWidths: list[float] = []
    for _ in range(int(max(4, numBoot))):
        bootResiduals = rng.choice(residuals, size=localY.size, replace=True)
        bootResiduals = bootResiduals - float(np.mean(bootResiduals))
        localYBoot = localFit + bootResiduals
        widthBoot = _estimateLocalPeakWidth(
            localYBoot, localPeakIdx, peakSearchRadius=2
        )
        if widthBoot is None:
            continue
        bootLogWidths.append(float(np.log(widthBoot)))

    logWidth = float(np.log(widthHat))
    if len(bootLogWidths) < 2:
        return logWidth, 1.0e-6

    return logWidth, float(max(1.0e-6, np.var(bootLogWidths, ddof=1)))


def _normalizeSpanBounds(
    n: int,
    minSpan: int | None,
    maxSpan: int | None,
) -> tuple[int, int]:
    minSpan_ = 3 if minSpan is None else int(minSpan)
    minSpan_ = max(1, minSpan_)
    maxSpan_ = (
        int(max(10, min(50, np.floor(np.log2(max(int(n), 1) + 1) * 2))))
        if maxSpan is None
        else int(maxSpan)
    )
    if maxSpan_ <= 0:
        raise ValueError("`maxSpan` must be positive.")
    if maxSpan_ < minSpan_:
        maxSpan_ = minSpan_
    return minSpan_, maxSpan_


def _fallbackLengthResult(
    n: int,
    minSpan: int,
    maxSpan: int,
    method: str,
    intervalSizeBP: int | None = None,
    reason: str | None = None,
) -> tuple[int, int, int, dict[str, Any]]:
    point = int(np.clip(int(round(np.sqrt(max(int(n), 1)))), minSpan, maxSpan))
    contextSizeBP = (
        None if intervalSizeBP is None else int(point * (2 * int(intervalSizeBP)) + 1)
    )
    diagnostics: dict[str, Any] = {
        "method": str(method),
        "fallback": True,
        "fallback_reason": reason,
        "point_span": int(point),
        "lower_span": int(point),
        "upper_span": int(point),
        "min_span": int(minSpan),
        "max_span": int(maxSpan),
        "finite_count": int(n),
        "context_size_bp": contextSizeBP,
    }
    return int(point), int(point), int(point), diagnostics


def _acfCrossingLag(acf: np.ndarray, threshold: float) -> int | None:
    acfArr = np.asarray(acf, dtype=np.float64)
    if acfArr.size < 3:
        return None
    threshold_ = float(threshold)
    for i in range(0, acfArr.size - 2):
        window = acfArr[i : i + 3]
        if np.all(np.isfinite(window)) and np.all(np.abs(window) < threshold_):
            return int(i + 1)
    return None


def chooseDependenceLength(
    chromMat: np.ndarray,
    intervalSizeBP: int,
    minSpan: int | None = 3,
    maxSpan: int | None = 64,
    trim: float = 0.10,
) -> tuple[int, int, int, dict[str, Any]]:
    r"""Choose local dependence length for model-fitting context sizes.

    This estimator targets autocorrelation scale in the transformed signal/background
    track. It is intended for background, MUNC, local variance, and sampling block
    sizes.
    """

    arr = np.asarray(chromMat)
    n = int(arr.shape[-1]) if arr.ndim > 0 else 0
    minSpan_, maxSpan_ = _normalizeSpanBounds(n, minSpan, maxSpan)
    intervalSizeBP_ = max(int(intervalSizeBP), 1)
    if n < max(8, minSpan_ + 3):
        return _fallbackLengthResult(
            n,
            minSpan_,
            maxSpan_,
            "sqrt_fallback",
            intervalSizeBP_,
            "too_few_intervals",
        )

    contextTrack = np.asarray(
        cconsenrich.ctrimMeanAxis0(arr, float(trim)),
        dtype=np.float64,
    )
    finiteMask = np.isfinite(contextTrack)
    finiteVals = np.asarray(contextTrack[finiteMask], dtype=np.float64)
    finiteCount = int(finiteVals.size)
    if finiteCount < max(20, minSpan_ + 3):
        return _fallbackLengthResult(
            finiteCount,
            minSpan_,
            maxSpan_,
            "sqrt_fallback",
            intervalSizeBP_,
            "too_few_finite_values",
        )

    center = float(np.median(finiteVals))
    scale = 1.4826 * float(np.median(np.abs(finiteVals - center)))
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = float(np.std(finiteVals, ddof=1)) if finiteCount >= 2 else 0.0
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = 1.0

    lo, hi = np.quantile(finiteVals, [0.005, 0.995])
    lo = float(max(float(lo), center - (8.0 * scale)))
    hi = float(min(float(hi), center + (8.0 * scale)))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        lo = center - (8.0 * scale)
        hi = center + (8.0 * scale)

    y = np.full(contextTrack.shape, np.nan, dtype=np.float64)
    y[finiteMask] = np.clip(contextTrack[finiteMask], lo, hi) - center
    acf, incrementVariance, pairCounts, gamma0, finiteCountKernel = (
        cconsenrich.cdependenceLengthStats(
            np.ascontiguousarray(y, dtype=np.float64),
            int(maxSpan_),
        )
    )
    acf = np.asarray(acf, dtype=np.float64)
    incrementVariance = np.asarray(incrementVariance, dtype=np.float64)
    pairCounts = np.asarray(pairCounts, dtype=np.int64)
    gamma0 = float(gamma0)
    if (not np.isfinite(gamma0)) or gamma0 <= 0.0:
        return _fallbackLengthResult(
            finiteCount,
            minSpan_,
            maxSpan_,
            "sqrt_fallback",
            intervalSizeBP_,
            "zero_or_invalid_gamma0",
        )

    crossingLag = _acfCrossingLag(acf, 0.10)
    if crossingLag is None:
        acfForIAT = acf
    else:
        acfForIAT = acf[:crossingLag]
    positiveAcf = np.clip(acfForIAT[np.isfinite(acfForIAT)], 0.0, None)
    iatSpan = int(np.ceil(0.5 + float(np.sum(positiveAcf))))
    iatSpan = int(np.clip(iatSpan, minSpan_, maxSpan_))

    validInc = incrementVariance[np.isfinite(incrementVariance) & (pairCounts > 0)]
    plateau = float("nan")
    incrementElbow: int | None = None
    if validInc.size > 0:
        tailStart = int(np.floor(0.75 * validInc.size))
        plateauVals = validInc[tailStart:]
        if plateauVals.size == 0:
            plateauVals = validInc
        plateau = float(np.median(plateauVals))
        if np.isfinite(plateau) and plateau > 0.0:
            threshold = 0.90 * plateau
            for idx, value in enumerate(incrementVariance, start=1):
                if np.isfinite(value) and value >= threshold:
                    incrementElbow = int(idx)
                    break

    candidates = [int(iatSpan)]
    if crossingLag is not None:
        candidates.append(int(crossingLag))
    if incrementElbow is not None:
        candidates.append(int(incrementElbow))
    pointSpan = int(round(float(np.median(np.asarray(candidates, dtype=np.float64)))))
    pointSpan = int(np.clip(pointSpan, minSpan_, maxSpan_))

    lowerCrossing = _acfCrossingLag(acf, 0.20)
    upperCrossing = _acfCrossingLag(acf, 0.05)
    lowerSpan = int(lowerCrossing) if lowerCrossing is not None else pointSpan
    upperSpan = int(upperCrossing) if upperCrossing is not None else pointSpan
    lowerSpan = int(np.clip(min(lowerSpan, pointSpan), minSpan_, maxSpan_))
    upperSpan = int(np.clip(max(upperSpan, pointSpan), minSpan_, maxSpan_))
    contextSizeBP = int(pointSpan * (2 * intervalSizeBP_) + 1)

    diagnostics: dict[str, Any] = {
        "method": "dependence_acf_increment",
        "fallback": False,
        "point_span": int(pointSpan),
        "lower_span": int(lowerSpan),
        "upper_span": int(upperSpan),
        "context_size_bp": int(contextSizeBP),
        "interval_size_bp": int(intervalSizeBP_),
        "min_span": int(minSpan_),
        "max_span": int(maxSpan_),
        "trim": float(trim),
        "finite_count": int(finiteCountKernel),
        "center": float(center),
        "scale": float(scale),
        "clip_lo": float(lo),
        "clip_hi": float(hi),
        "gamma0": float(gamma0),
        "crossing_lag": None if crossingLag is None else int(crossingLag),
        "relaxed_crossing_lag": (None if lowerCrossing is None else int(lowerCrossing)),
        "strict_crossing_lag": None if upperCrossing is None else int(upperCrossing),
        "iat_span": int(iatSpan),
        "increment_plateau": float(plateau),
        "increment_elbow": None if incrementElbow is None else int(incrementElbow),
        "candidate_spans": [int(v) for v in candidates],
        "acf": [float(v) if np.isfinite(v) else None for v in acf],
        "increment_variance": [
            float(v) if np.isfinite(v) else None for v in incrementVariance
        ],
        "pair_counts": [int(v) for v in pairCounts],
    }
    return int(pointSpan), int(lowerSpan), int(upperSpan), diagnostics


def chooseFeatureLength(
    vals: np.ndarray,
    minSpan: int | None = 3,
    maxSpan: int | None = 64,
    bandZ: float = 1.0,
    maxOrder: int = 5,
) -> tuple[int, int, int, dict[str, Any]]:
    r"""Choose typical feature/peak length from local peak morphology.

    Candidate features are detected on a smoothed log-scale track, half-height
    widths are measured locally, and width uncertainty is estimated by a local
    residual bootstrap before EB shrinkage on the log-width scale
    """
    y = np.asarray(vals, dtype=np.float64)
    n = int(y.size)
    minSpan_, maxSpan_ = _normalizeSpanBounds(n, minSpan, maxSpan)
    finiteMask = np.isfinite(y)
    finiteCount = int(np.count_nonzero(finiteMask))
    if finiteCount < 100:
        return _fallbackLengthResult(
            finiteCount,
            minSpan_,
            maxSpan_,
            "feature_sqrt_fallback",
            reason="too_few_finite_values",
        )
    y = np.where(finiteMask, y, 0.0)

    yPos = np.clip(y, 0.0, None)
    yLog = np.log1p(yPos)
    posLog = yLog[y > 0]
    if posLog.size <= max(1, int(maxOrder)):
        return _fallbackLengthResult(
            int(posLog.size),
            minSpan_,
            maxSpan_,
            "feature_sqrt_fallback",
            reason="too_few_positive_values",
        )

    smoothSize = min(int(max(1, minSpan_)), int(maxSpan_ / 2))
    yLogSmooth = ndimage.uniform_filter1d(yLog, size=smoothSize, mode="nearest")
    kMinFeatures = int(max(1, (2 * np.log2(n + 1))))
    thrLog = float(np.mean(posLog))
    startOrder = int(max(1, maxOrder))
    bestOrder = 1
    bestFeatures = np.array([], dtype=np.int64)
    bestScore = -1.0

    # first, choose between-feature 'distance' threshold based on score that is increasing wrt number/prominence of features
    # search space is over [1, maxOrder]
    for o in range(startOrder, 0, -1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            features, props = signal.find_peaks(
                yLogSmooth,
                distance=int(max(1, o)),
                prominence=1e-4,
            )

        prominences = props.get("prominences")
        promMasked: np.ndarray | None = None
        if features.size:
            featureMask = yLog[features] > thrLog
            if prominences is not None:
                featureMask &= prominences > 0.0
                promMasked = prominences[featureMask]
            features = features[featureMask]

        numFeatures = int(features.size)

        # score by sum of top-K prominences, prevent small/noisy peaks dominating the score.
        if numFeatures == 0:
            score = -np.inf
        else:
            if promMasked is None:
                logger.warning(
                    "Prominences missing for context size estimation...using number of features of distinct features as score.",
                )
                score = float(numFeatures)
            else:
                K = int(min(1000, promMasked.size))
                if K <= 0:
                    score = float(numFeatures)
                else:
                    top = np.partition(promMasked, -K)[-K:]
                    score = float(np.sum(np.log1p(top)))

        if score > bestScore:
            bestScore = score
            bestOrder = o
            bestFeatures = features.astype(np.int64, copy=False)

    chosenFeatures = bestFeatures
    logger.info(
        f"Chose order={bestOrder} with numFeatures={int(chosenFeatures.size)} (score={bestScore:.3f})."
    )

    if chosenFeatures.size == 0:
        return _fallbackLengthResult(
            n,
            minSpan_,
            maxSpan_,
            "feature_sqrt_fallback",
            reason="no_candidate_peaks",
        )

    chosenFeatures = np.unique(chosenFeatures.astype(np.int64))
    chosenFeatures.sort()

    # We compute a 'baseline' around each feature -- these are used to
    # compute per-feature 'feature scores' (height above baseline).
    baseQ = 0.05
    featureBaselines = np.empty(chosenFeatures.size, dtype=np.float64)
    for i, idx in enumerate(chosenFeatures):
        # FFR: this might be redundant in practice...perhaps just take quantile over full window?
        left = max(0, idx - maxSpan_)
        right = min(n - 1, idx + maxSpan_)

        leftQ = float(np.quantile(yLog[left : idx + 1], baseQ))
        rightQ = float(np.quantile(yLog[idx : right + 1], baseQ))
        featureBaselines[i] = max(leftQ, rightQ)

    featureScores = yLog[chosenFeatures] - featureBaselines
    keepMask = featureScores > 0.0
    if np.any(keepMask):
        chosenFeatures = chosenFeatures[keepMask]
        featureScores = featureScores[keepMask]

    kKeep = int(min(1000, featureScores.size, max(kMinFeatures, n // max(8, maxSpan_))))
    if kKeep <= 0:
        return _fallbackLengthResult(
            n,
            minSpan_,
            maxSpan_,
            "feature_sqrt_fallback",
            reason="no_scored_peaks",
        )

    # we use the top-scoring kKeep features for estimation
    keep = np.argpartition(-featureScores, kKeep - 1)[:kKeep]
    featureIndexArray = np.unique(chosenFeatures[keep].astype(np.int64))
    featureIndexArray.sort()

    # for each feature, estimate width on log scale and use a local bootstrap
    # to quantify width uncertainty
    noiseWindow = int(min(maxSpan_, 32))
    sHatList: list[float] = []
    sigma2List: list[float] = []

    for peakIdx in featureIndexArray:
        logWidth, sigmaS2 = _bootstrapLocalPeakLogWidthVariance(
            yVals=yLog,
            peakIdx=int(peakIdx),
            halfWindow=noiseWindow,
            numBoot=16,
        )
        if logWidth is None or sigmaS2 is None:
            continue
        sHatList.append(logWidth)
        sigma2List.append(sigmaS2)

    if len(sHatList) == 0:
        return _fallbackLengthResult(
            n,
            minSpan_,
            maxSpan_,
            "feature_sqrt_fallback",
            reason="no_valid_widths",
        )

    # Method-of-moments estimate given observed variance of log-widths
    sHatArr = np.asarray(sHatList, dtype=np.float64)
    sigma2Arr = np.asarray(sigma2List, dtype=np.float64)
    nFeat = int(sHatArr.size)

    #   varS := sample variance of log-widths across all features
    varS = float(np.var(sHatArr, ddof=1)) if nFeat > 1 else 0.0

    #   meanSigma2 := average of per-feature variance estimates
    meanSigma2 = float(np.mean(sigma2Arr))

    #   Given observed variance of log-widths (varS) and average per-feature variance estimates (meanSigma2),
    #   the MoM estimate of the 'between-feature' variance component is from E[x^2] - E[x]^2 = Var[x],
    #   ... i.e., tau^2_MoM = varS - meanSigma2
    tau2Mom = float(max(0.0, varS - meanSigma2))
    tau2Max = float(max(1e-6, varS + meanSigma2) * 10.0)

    def _LL(betweenFeatureVar: float) -> float:
        # the negative LL here is computed assuming independent Gaussian features with
        # ... per-feature variance = sigma2[i] + tau^2 (where sigma2[i] is the per-feature variance estimate from above)
        tauSq = float(max(0.0, betweenFeatureVar))
        totalVar = sigma2Arr + tauSq
        totalVar = np.maximum(totalVar, 1e-12)
        invTotalVar = 1.0 / totalVar
        muHat = float(np.sum(invTotalVar * sHatArr) / np.sum(invTotalVar))
        residuals = sHatArr - muHat
        # note several constant terms dropped from the expression, but order is preserved
        return 0.5 * float(
            np.sum(np.log(totalVar)) + np.sum((residuals * residuals) * invTotalVar)
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        # maximize marginal likelihood over 0 <= tau^2 <= tau2Max
        # FFR: there may be a cleaner approach -- this is still negligible in runtime for now
        res = optimize.minimize_scalar(
            _LL,
            bounds=(0.0, tau2Max),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 50},
        )

    # Take the posterior at each feature given fixed tauSqHat
    # ... (or backup MoM estimate if optimization failed)
    tauSqHat: float = 0.0
    if getattr(res, "success", False):
        tauSqHat = float(res.x)
        logger.info(f"Between-feature variance (tau^2): {tauSqHat:.6f}")
    else:
        tauSqHat = tau2Mom
        logger.warning(
            f"Failed to solve for tau^2...using MoM estimate tau^2 = {tau2Mom:.6f}.",
        )

    #   Posterior variance sigma^2[i] + tau^2 used to compute weights
    vHat = sigma2Arr + tauSqHat
    vHat = np.maximum(vHat, 1e-12)
    wHat = 1.0 / vHat

    # Get point estimate and bounds on natural scale
    muHat = float(np.sum(wHat * sHatArr) / np.sum(wHat))
    muVar = float(1.0 / np.sum(wHat))
    predStd = float(np.sqrt(max(0.0, tauSqHat + muVar)))
    logLower = float(muHat - bandZ * predStd)
    logUpper = float(muHat + bandZ * predStd)
    pointEstimate = float(np.exp(muHat))
    widthLower = max(1.0, float(np.exp(logLower)))
    widthUpper = max(1.0, float(np.exp(logUpper)), widthLower)

    pointEstimate = float(np.clip(pointEstimate, minSpan_, maxSpan_))
    widthLower = float(np.clip(widthLower, minSpan_, maxSpan_))
    widthUpper = float(np.clip(max(widthUpper, widthLower), minSpan_, maxSpan_))

    point = int(round(pointEstimate))
    lower = int(round(min(widthLower, point)))
    upper = int(round(max(widthUpper, point)))
    diagnostics: dict[str, Any] = {
        "method": "feature_peak_width_random_effects",
        "fallback": False,
        "point_span": int(point),
        "lower_span": int(lower),
        "upper_span": int(upper),
        "min_span": int(minSpan_),
        "max_span": int(maxSpan_),
        "num_intervals": int(n),
        "finite_count": int(finiteCount),
        "num_candidate_peaks": int(chosenFeatures.size),
        "num_retained_peaks": int(featureIndexArray.size),
        "num_valid_widths": int(nFeat),
        "best_order": int(bestOrder),
        "best_peak_score": float(bestScore),
        "threshold_log": float(thrLog),
        "smooth_size": int(smoothSize),
        "mean_log_width": float(muHat),
        "between_feature_variance": float(tauSqHat),
        "predictive_log_width_sd": float(predStd),
    }
    return int(point), int(lower), int(upper), diagnostics
