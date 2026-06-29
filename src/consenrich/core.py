# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
import math
import operator
import os
import sys
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
from scipy import ndimage, signal, stats, optimize, sparse, interpolate, special
from scipy.sparse import linalg as sparse_linalg
from itrigamma import itrigamma, trigamma
from . import cconsenrich
from . import ccounts
from .constants import (
    ALIGNMENT_SOURCE_KINDS,
    BEDGRAPH_SOURCE_KIND,
    COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP,
    COUNTING_DEFAULT_LOG_MULT,
    COUNTING_DEFAULT_LOG_OFFSET,
    COUNTING_DEFAULT_CENTER_MB,
    COUNTING_DEFAULT_CENTER_MB_METHOD,
    COUNTING_CENTER_MB_METHOD_MEDFILT,
    COUNTING_CENTER_MB_METHOD_SAVGOL,
    COUNTING_DEFAULT_TRANSFORM_INPUT_OFFSET,
    COUNTING_DEFAULT_TRANSFORM_INPUT_SCALE,
    COUNTING_DEFAULT_TRANSFORM_METHOD,
    COUNTING_DEFAULT_TRANSFORM_OUTPUT_OFFSET,
    COUNTING_DEFAULT_TRANSFORM_OUTPUT_SCALE,
    COUNTING_DEFAULT_TRANSFORM_SHAPE,
    COUNTING_SUPPORTED_CENTER_MB_METHODS,
    FRAGMENTS_SOURCE_KIND,
    FIT_DEFAULT_BACKGROUND,
    FIT_DEFAULT_BACKGROUND_LENGTH_SCALE_MULTIPLIER,
    FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER,
    FIT_DEFAULT_BACKGROUND_SHIFT_RTOL,
    FIT_DEFAULT_BACKGROUND_SMOOTHNESS,
    FIT_DEFAULT_FIXED_BACKGROUND_ITERS,
    FIT_DEFAULT_FIXED_BACKGROUND_RTOL,
    FIT_DEFAULT_MIN_OUTER_ITERS,
    FIT_DEFAULT_OUTER_ITERS,
    FIT_DEFAULT_OUTER_NLL_RTOL,
    FIT_DEFAULT_ROBUST_T_NU,
    FIT_DEFAULT_T_INNER_ITERS,
    FIT_DEFAULT_USE_APN,
    FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND,
    FIT_DEFAULT_USE_OBS_PRECISION_REWEIGHTING,
    FIT_DEFAULT_USE_PROCESS_PRECISION_REWEIGHTING,
    FIT_DEFAULT_ZERO_CENTER_BACKGROUND,
    INPUT_DEFAULT_ROLE,
    MATCHING_DEFAULT_METADATA_DETAIL,
    MATCHING_DEFAULT_MIN_PEAK_SCORE,
    MATCHING_DEFAULT_BROAD_MAX_GAP_BP,
    MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z,
    MATCHING_DEFAULT_PEAK_MODE,
    MATCHING_DEFAULT_USE_SHRUNK_STATE_SCORES,
    MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE,
    MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z,
    MUNC_EB_PRIOR_G_UNCERTAINTY_MODE_DISABLED,
    MUNC_EB_PRIOR_G_UNCERTAINTY_MODE_PROXY,
    MUNC_SUPPORTED_EB_PRIOR_G_UNCERTAINTY_MODES,
    MUNC_SUPPORTED_VARIANCE_MODELS,
    MUNC_VARIANCE_MODEL_CODE_KALMAN,
    MUNC_VARIANCE_MODEL_KALMAN,
    OBSERVATION_DEFAULT_DEPENDENCE_ACF_MIN_EVIDENCE_NATS,
    OBSERVATION_DEFAULT_DEPENDENCE_ACF_POINT_THRESHOLD,
    OBSERVATION_DEFAULT_DEPENDENCE_ACF_REQUIRED_CROSSINGS,
    OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER,
    OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_SIZE_BP,
    OBSERVATION_DEFAULT_MUNC_SEED_PROCESS_MAX_Q,
    OBSERVATION_DEFAULT_MUNC_SEED_PROCESS_MIN_Q,
    OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_ENABLED,
    OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_MAX,
    OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_MIN,
    OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_PASSES,
    OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_T,
    OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_TDF,
    OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER,
    OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_SIZE_BP,
    OBSERVATION_DEFAULT_MUNC_COVARIATE_FEATURES,
    OBSERVATION_DEFAULT_MUNC_COVARIATES_ENABLED,
    OBSERVATION_DEFAULT_MUNC_COVARIATES_MODE,
    OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_MAX_BP,
    OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_MEDIAN_BP,
    OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_MIN_BP,
    OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_SIGMA,
    OBSERVATION_DEFAULT_DEPENDENCE_MAX_CONTEXT_SIZE_BP,
    OBSERVATION_DEFAULT_DEPENDENCE_NUM_BLOCKS,
    OBSERVATION_DEFAULT_DEPENDENCE_PRIOR_LOG_SD,
    OBSERVATION_DEFAULT_DEPENDENCE_PRIOR_MEDIAN_SPAN,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_G_UNCERTAINTY_MODE,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_MAX_EXTRAPOLATED_FRACTION,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_MIN_TILES_PER_STRATUM,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_SEED,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_STRATA,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_SUPPORT_MAX_Q,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_SUPPORT_MIN_Q,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_TILE_COUNT,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_TILE_SIZE_BP,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_STRENGTH_WINSOR_TAIL,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_WARMUP_ECM_ITERS,
    OBSERVATION_DEFAULT_MUNC_EB_PRIOR_WARMUP_OUTER_PASSES,
    OBSERVATION_DEFAULT_MIN_R,
    OBSERVATION_DEFAULT_MUNC_DEPENDENCE_MIN_CONTEXT_SIZE_BP,
    OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL,
    OBSERVATION_DEFAULT_USE_COUNT_NOISE_FLOOR,
    OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MAX,
    OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MIN,
    OBSERVATION_DEFAULT_RESTRICT_LOCAL_VARIANCE_TO_SPARSE_BED,
    OUTPUT_DEFAULT_DIAGNOSTIC_TRACKS,
    OUTPUT_DEFAULT_CUTOFF_REPORT,
    OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES,
    OUTPUT_DEFAULT_MAX_PRECISION_DIAGNOSTIC_ROWS_PER_CHROMOSOME,
    OUTPUT_DEFAULT_PLOT_CORRELATION_LENGTH,
    OUTPUT_DEFAULT_PLOT_OPTIMIZATION_PATH,
    OUTPUT_DEFAULT_PRECISION_DIAGNOSTIC_DETAIL,
    OUTPUT_DEFAULT_SAVE_BACKGROUND_TRACKS,
    OUTPUT_DEFAULT_SAVE_GAINS,
    OUTPUT_DEFAULT_STATE_SHRINKAGE_MODEL,
    OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT,
    OUTPUT_DEFAULT_STATE_SHRINKAGE_PRIOR_SPIKE_PROP,
    OUTPUT_DEFAULT_STATE_SHRINKAGE_PRIOR_SCALE,
    OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT,
    OUTPUT_DEFAULT_STATE_SHRINKAGE_STUDENT_T_DF,
    OUTPUT_DEFAULT_STATE_SHRINKAGE_STUDENT_T_QUADRATURE_ORDER,
    OUTPUT_DEFAULT_WRITE_RUN_SUMMARY,
    OUTPUT_DEFAULT_WRITE_STATE_SHRINKAGE,
    LOGGING_DEFAULT_LOG_FILE,
    LOGGING_DEFAULT_PROGRESS,
    LOGGING_DEFAULT_VERBOSITY,
    PROCESS_DEFAULT_DELTA_F,
    PROCESS_DEFAULT_NOISE_CALIBRATION,
    PROCESS_DEFAULT_Q_PRIOR_LEVEL,
    PROCESS_DEFAULT_Q_PRIOR_TREND,
    PROCESS_DEFAULT_Q_SEED_PRIOR_LEVEL,
    PROCESS_DEFAULT_PUNC_DEPENDENCE_MULTIPLIER,
    PROCESS_DEFAULT_PUNC_DEADBAND_PRIOR_WEIGHT,
    PROCESS_DEFAULT_PUNC_LOCAL_WINDOW_MULTIPLIER,
    PROCESS_DEFAULT_PUNC_MAX_SCALE,
    PROCESS_DEFAULT_PUNC_MIN_SCALE,
    PROCESS_DEFAULT_PUNC_MIN_WINDOW_WEIGHT,
    PROCESS_DEFAULT_PUNC_LEVEL_BUFFER_Z,
    PROCESS_DEFAULT_PUNC_PRIOR_DF,
    PROCESS_DEFAULT_PUNC_PRIOR_DF_MOMENTS_MIN_WINDOWS,
    PROCESS_DEFAULT_PUNC_PRIOR_DF_MOMENTS_WINSOR_TAIL,
    PROCESS_DEFAULT_PUNC_PRIOR_RIDGE,
    PROCESS_DEFAULT_PUNC_USE_BOUNDARY_CLAMPS,
    PROCESS_DEFAULT_PUNC_USE_GLOBAL_SCALE,
    PROCESS_DEFAULT_PUNC_USE_PRIOR_DF_MOMENTS,
    PROCESS_DEFAULT_PUNC_USE_PRIOR_SHRINKAGE,
    PROCESS_DEFAULT_PUNC_USE_RELIABILITY_WEIGHTED_WINDOWS,
    PROCESS_DEFAULT_PUNC_USE_SCALE_REBASE,
    PROCESS_DEFAULT_PUNC_USE_TRANSITION_EVIDENCE,
    PROCESS_DEFAULT_PUNC_USE_WARMUP_FIT,
    PROCESS_DEFAULT_PUNC_TREND_SEED_RATIO,
    PROCESS_DEFAULT_WARMUP_ECM_ITERS,
    PROCESS_DEFAULT_WARMUP_OUTER_PASSES,
    PROCESS_DEFAULT_MIN_Q,
    PROCESS_DEFAULT_MAX_Q,
    PROCESS_DEFAULT_PRECISION_MULTIPLIER_MAX,
    PROCESS_DEFAULT_PRECISION_MULTIPLIER_MIN,
    PROCESS_NOISE_CALIBRATION_FIXED,
    PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL,
    PROCESS_NOISE_CALIBRATION_MODES,
    PROCESS_NOISE_CALIBRATION_SEED,
    PROCESS_NOISE_CALIBRATION_PUNC,
    PROCESS_DEFAULT_STATE_MODEL,
    SAM_DEFAULT_BAM_INPUT_MODE,
    SAM_DEFAULT_COUNT_MODE,
    SAM_DEFAULT_EXTEND_FROM_5P_BP,
    SAM_DEFAULT_INFER_FRAGMENT_LENGTH,
    SAM_DEFAULT_MAX_INSERT_SIZE,
    SAM_DEFAULT_MIN_MAPPING_QUALITY,
    SAM_DEFAULT_MIN_TEMPLATE_LENGTH,
    SAM_DEFAULT_SHIFT_FORWARD_5P,
    SAM_DEFAULT_SHIFT_REVERSE_5P,
    SC_DEFAULT_BARCODE_TAG,
    SC_DEFAULT_COUNT_MODE,
    SC_DEFAULT_FRAGMENTS_GROUP_NORM,
    SC_DEFAULT_FRAGMENT_POSITION_MODE,
    STATE_MODEL_LEVEL,
    STATE_MODEL_LEVEL_TREND,
    STATE_MODEL_MODES,
    SUPPORTED_BAM_INPUT_MODES,
    SUPPORTED_COUNT_MODES,
    SUPPORTED_FRAGMENT_POSITION_MODES,
    SUPPORTED_SOURCE_KINDS,
    UNCERTAINTY_CALIBRATION_DEFAULT_BLOCK_SIZE_BP,
    UNCERTAINTY_CALIBRATION_DEFAULT_CALIBRATION_ECM_ITERS,
    UNCERTAINTY_CALIBRATION_DEFAULT_CALIBRATION_OUTER_ITERS,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_APPLY_TARGET_CALIBRATION,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_BOOTSTRAP_REPLICATES,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_MODEL,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_SEGMENT_COUNT,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FALLBACK_MIN_VALID_FRACTION,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_MAX_INFORMATION_FRACTION,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_MIN_INFORMATION_FRACTION,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_MIN_DELTA_VARIANCE,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_SCORE_WEIGHT_MODE,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_TARGET_SIGNAL,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_USE_LAMBDA_IN_INFORMATION,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_VARIANCE_MODE,
    UNCERTAINTY_CALIBRATION_DEFAULT_ENABLED,
    UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX,
    UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX_OVERRIDE,
    UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN,
    UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN_OVERRIDE,
    UNCERTAINTY_CALIBRATION_DEFAULT_FOLDS,
    UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_DELETION_PROBABILITY,
    UNCERTAINTY_CALIBRATION_DEFAULT_MAX_HELDOUT_CELLS,
    UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS,
    UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES,
    UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS,
    UNCERTAINTY_CALIBRATION_DEFAULT_MODE,
    UNCERTAINTY_CALIBRATION_DEFAULT_PAD,
    UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE,
    UNCERTAINTY_CALIBRATION_DEFAULT_SCALE_UNCERTAINTY_BY_TARGET_CALIBRATION,
    UNCERTAINTY_CALIBRATION_DEFAULT_SEED,
    UNCERTAINTY_CALIBRATION_DEFAULT_TARGET_CALIBRATION_DELTA,
    UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS,
    UNCERTAINTY_CALIBRATION_DEFAULT_WRITE_DIAGNOSTICS,
    UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET,
    UNCERTAINTY_CALIBRATION_DELETE_BLOCK_FACTOR_MODELS,
    UNCERTAINTY_CALIBRATION_DELETE_BLOCK_SCORE_WEIGHT_MODES,
    UNCERTAINTY_CALIBRATION_DELETE_BLOCK_TARGET_SIGNALS,
    UNCERTAINTY_CALIBRATION_DELETE_BLOCK_VARIANCE_MODES,
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
    UNCERTAINTY_CALIBRATION_MODE_DELETE_BLOCK_STATE,
    UNCERTAINTY_CALIBRATION_MODES,
    UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR,
    UNCERTAINTY_CALIBRATION_REFIT_PROCESS_NOISE_WARMUP_ECM_ITERS,
    UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE,
    UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES,
    UNCERTAINTY_CALIBRATION_SCORE_REPLICATE_CODE_STRIDE,
    UNCERTAINTY_CALIBRATION_SCORE_STATE_ABS_QUANTILE,
    UNCERTAINTY_CALIBRATION_SUMMARY_MEDIAN_QUANTILE,
    UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE,
    UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR,
)
from .diagnostics import metadataFloat, summarizePrecisionBoundaryHits
from ._logging import (
    format_log_event as _formatLogEvent,
    format_log_value as _formatLogValue,
    log_event as _sharedLogEvent,
    log_event_name as _logEventName,
    log_field_name as _logFieldName,
    quote_log_string as _quoteLogString,
)
from ._normalization import (
    native_count_mode_for_preset as _sharedNativeCountModeForPreset,
    normalize_bam_input_mode as _sharedNormalizeBamInputMode,
    normalize_count_mode as _sharedNormalizeCountMode,
    normalize_count_transform_method as _sharedNormalizeCountTransformMethod,
    normalize_fragment_position_mode as _sharedNormalizeFragmentPositionMode,
    normalize_process_noise_calibration as _sharedNormalizeProcessNoiseCalibration,
    weighted_quantile_interpolated as _sharedWeightedQuantileInterpolated,
)

logger = logging.getLogger(__name__)

_PROCESS_NOISE_WARMUP_Q_LOG_CHANGE_RTOL = 1.0e-2
_QINIT_MIN_TRANSITIONS = 8
_QINIT_MAX_TRANSITIONS = 32_000
_QINIT_GRID_SIZE = 64
_QINIT_PRECISION_SAMPLE_CAP = 32_000
_QINIT_PRECISION_CAP_QUANTILE = 0.95
_QINIT_PRECISION_CAP_MULTIPLIER = 20.0
_QINIT_PRIOR_LOG_SD = math.log(4.0)
_QINIT_DEFAULT_T_NU = 8.0
_PUNC_DEADBAND_HIGH_PROBABILITY = 0.8
_PUNC_STAGE_TOGGLE_KEYS = (
    "puncUseWarmupFit",
    "puncUseTransitionEvidence",
    "puncUseScaleRebase",
    "puncUseGlobalScale",
    "puncUseBoundaryClamps",
    "puncUsePriorDfMoments",
    "puncUsePriorShrinkage",
)


def _logEvent(
    event: str,
    fields: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] = (),
    *,
    logger_: logging.Logger = logger,
    level: int = logging.INFO,
    stacklevel: int = 2,
) -> None:
    _sharedLogEvent(
        logger_,
        event,
        fields,
        level=level,
        stacklevel=int(stacklevel) + 1,
    )


def _logAsciiBlock(
    title: str,
    rows: list[tuple[str, Any]] | tuple[tuple[str, Any], ...] = (),
    *,
    logger_: logging.Logger = logger,
    level: int = logging.INFO,
    indentLevel: int = 0,
) -> None:
    indentLevel = max(0, int(indentLevel))
    if indentLevel:
        rows = (("phase depth", int(indentLevel)), *tuple(rows))
    _logEvent(title, rows, logger_=logger_, level=level, stacklevel=3)


class processParams(NamedTuple):
    r"""Parameters related to the process model of Consenrich.

    :param deltaF: Fixed positive integration step size in the two-state
        transition :math:`x_{[i+1,0]} = x_{[i,0]} + \delta_F x_{[i,1]}`.
        Ignored when ``stateModel="level"``.
    :type deltaF: float
    :param stateModel: Latent process model. ``"levelTrend"`` uses the existing
        two-state level/slope model. ``"level"`` uses only the signal level
        state and pads public state arrays for compatibility.
    :type stateModel: str
    :param minQ: Lower floor for calibrated base process-noise diagonal entries.
        The same floor is used for process-noise calibration bounds, warm-start
        conditioning, and adaptive process-noise bounds.
    :type minQ: float
    :param maxQ: Maximum process noise scale. If ``maxQ < 0``, no effective upper bound is enforced.
    :type maxQ: float
    :param processNoiseCalibration: Process-noise calibration mode:
        ``"punc"``, ``"seed"``, or ``"fixed"``.
    :type processNoiseCalibration: str
    :param puncLocalWindowMultiplier: Multiplier converting ``blockLenIntervals``
        into PUNC transition-window length.
    :type puncLocalWindowMultiplier: float
    :param puncDependenceMultiplier: Effective-sample-size divisor for
        overlapping/dependent transition evidence.
    :type puncDependenceMultiplier: float
    :param puncMinScale: Lower clamp for local PUNC process-Q scales.
    :type puncMinScale: float
    :param puncMaxScale: Upper clamp for local PUNC process-Q scales.
    :type puncMaxScale: float
    :param puncMinWindowWeight: Minimum total reliability weight for a PUNC
        window to contribute local evidence.
    :type puncMinWindowWeight: float
    :param puncPriorRidge: Ridge penalty for the PUNC process prior fit.
    :type puncPriorRidge: float
    :param puncLevelBufferZ: Posterior-SD buffer applied to the PUNC
        state-level prior covariate. Values near zero flatten the prior trend;
        ``0`` recovers the unbuffered prior.
    :type puncLevelBufferZ: float
    :param processNoiseWarmupECMIters: Maximum fixed-background ECM iterations
        per nuisance pass used by process-noise warm-up calibration.
    :type processNoiseWarmupECMIters: int
    :param processNoiseWarmupOuterPasses: Total outer warm-up pass budget used
        by the internal nuisance/Q alternation before the final fit.
    :type processNoiseWarmupOuterPasses: int
    :param precisionMultiplierMin: Lower clamp for process precision multipliers
        :math:`\kappa_{[i]}` during robust ECM reweighting. If negative, it is
        resolved at fit time to the most permissive strict
        convexity-preserving lower clamp for the active state dimension and
        ``ECM_robustTNu``.
    :type precisionMultiplierMin: float
    :param precisionMultiplierMax: Upper clamp for process precision multipliers
        :math:`\kappa_{[i]}` during robust ECM reweighting.
    :type precisionMultiplierMax: float
    :seealso: :func:`consenrich.core.runConsenrich`

    """

    deltaF: float = PROCESS_DEFAULT_DELTA_F
    minQ: float = PROCESS_DEFAULT_MIN_Q
    maxQ: float = PROCESS_DEFAULT_MAX_Q
    qPriorLevel: float = PROCESS_DEFAULT_Q_PRIOR_LEVEL
    qPriorTrend: float = PROCESS_DEFAULT_Q_PRIOR_TREND
    qSeedPriorLevel: float = PROCESS_DEFAULT_Q_SEED_PRIOR_LEVEL
    processNoiseCalibration: str = PROCESS_DEFAULT_NOISE_CALIBRATION
    puncLocalWindowMultiplier: float = PROCESS_DEFAULT_PUNC_LOCAL_WINDOW_MULTIPLIER
    puncDependenceMultiplier: float = PROCESS_DEFAULT_PUNC_DEPENDENCE_MULTIPLIER
    puncMinScale: float = PROCESS_DEFAULT_PUNC_MIN_SCALE
    puncMaxScale: float = PROCESS_DEFAULT_PUNC_MAX_SCALE
    puncMinWindowWeight: float = PROCESS_DEFAULT_PUNC_MIN_WINDOW_WEIGHT
    puncPriorDf: float = PROCESS_DEFAULT_PUNC_PRIOR_DF
    puncPriorRidge: float = PROCESS_DEFAULT_PUNC_PRIOR_RIDGE
    puncLevelBufferZ: float = PROCESS_DEFAULT_PUNC_LEVEL_BUFFER_Z
    puncUseReliabilityWeightedWindows: bool = (
        PROCESS_DEFAULT_PUNC_USE_RELIABILITY_WEIGHTED_WINDOWS
    )
    puncUseWarmupFit: bool = PROCESS_DEFAULT_PUNC_USE_WARMUP_FIT
    puncUseTransitionEvidence: bool = PROCESS_DEFAULT_PUNC_USE_TRANSITION_EVIDENCE
    puncUseScaleRebase: bool = PROCESS_DEFAULT_PUNC_USE_SCALE_REBASE
    puncUseGlobalScale: bool = PROCESS_DEFAULT_PUNC_USE_GLOBAL_SCALE
    puncUseBoundaryClamps: bool = PROCESS_DEFAULT_PUNC_USE_BOUNDARY_CLAMPS
    puncUsePriorDfMoments: bool = PROCESS_DEFAULT_PUNC_USE_PRIOR_DF_MOMENTS
    puncUsePriorShrinkage: bool = PROCESS_DEFAULT_PUNC_USE_PRIOR_SHRINKAGE
    puncProcessCovariatesEnabled: bool = False
    puncProcessCovariatesMode: str = "transition"
    puncProcessCovariatesFeatures: tuple[str, ...] = ()
    processNoiseWarmupECMIters: int = PROCESS_DEFAULT_WARMUP_ECM_ITERS
    processNoiseWarmupOuterPasses: int = PROCESS_DEFAULT_WARMUP_OUTER_PASSES
    precisionMultiplierMin: float = PROCESS_DEFAULT_PRECISION_MULTIPLIER_MIN
    precisionMultiplierMax: float = PROCESS_DEFAULT_PRECISION_MULTIPLIER_MAX
    stateModel: str = PROCESS_DEFAULT_STATE_MODEL


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich

    :param maxR: Genome-wide upper bound for the replicate-specific observation noise levels.
    :type maxR: float | None
    :param samplingIters: Number of blocks (within-contig) to sample while building the empirical signed-mean variance trend in :func:`consenrich.core.fitPSplineLogVarianceTrend`.
    :type samplingIters: int | None
    :param muncVarianceModel: MUNC variance evidence model. The only
        supported value is ``"kalman"``.
    :type muncVarianceModel: str | None
    :param muncTrendBlockSizeBP: Expected sampled block size for fitting the MUNC signed mean-variance trend. If unset, it is inferred at runtime.
    :type muncTrendBlockSizeBP: int | None
    :param muncLocalWindowSizeBP: Sliding-window size for local MUNC variance tracks. If unset, it is inferred at runtime.
    :type muncLocalWindowSizeBP: int | None
    :param muncTrendBlockDependenceMultiplier: Multiplier applied to the inferred dependence span when auto-sizing MUNC trend sampling blocks.
    :type muncTrendBlockDependenceMultiplier: float | None
    :param muncLocalWindowDependenceMultiplier: Multiplier applied to the inferred dependence span when auto-sizing local MUNC variance windows.
    :type muncLocalWindowDependenceMultiplier: float | None
    :param EB_use: If True, shrink 'local' noise estimates to a prior trend dependent on amplitude. See  :func:`consenrich.core.getMuncTrack`.
    :type EB_use: bool | None
    :param useReplicateTrends: If True, fit the empirical Bayes MUNC mean/variance prior separately for each replicate instead of using one pooled trend with replicate scale factors.
    :type useReplicateTrends: bool | None
    :param muncCovariatesEnabled: If True, add a nonnegative per-replicate genomic covariate variance component to the MUNC prior before empirical-Bayes shrinkage.
    :type muncCovariatesEnabled: bool | None
    :param muncCovariatesMode: Genomic covariate MUNC mode. Currently ``"perReplicateAdditive"``.
    :type muncCovariatesMode: str | None
    :param muncCovariatesFeatures: Genomic covariate feature names to read from the cache. ``"gc"`` is transformed to chromosome-median absolute deviation at runtime.
    :type muncCovariatesFeatures: tuple[str, ...] | None
    :param EB_setNu0: If provided, manually set :math:`\nu_0` to this value (rather than computing via :func:`consenrich.core.EB_computePriorStrength`).
    :type EB_setNu0: int | None
    :param EB_setNuL: If provided, manually sets local model df, :math:`\nu_L`, to this value.
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
    :param restrictLocalVarianceToSparseBed: If True, and a sparse BED mask is supplied to :func:`consenrich.core.getMuncTrack`, restrict local MUNC variance windows to sparse BED regions.
    :type restrictLocalVarianceToSparseBed: bool | None
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
    pad: float | None
    precisionMultiplierMin: float | None = 0.25
    precisionMultiplierMax: float | None = 4.0
    useReplicateTrends: bool | None = False
    muncVarianceModel: str | None = OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL
    muncTrendBlockSizeBP: int | None = OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_SIZE_BP
    muncLocalWindowSizeBP: int | None = OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_SIZE_BP
    muncDependenceMinContextSizeBP: int | None = (
        OBSERVATION_DEFAULT_MUNC_DEPENDENCE_MIN_CONTEXT_SIZE_BP
    )
    dependenceMaxContextSizeBP: int | None = (
        OBSERVATION_DEFAULT_DEPENDENCE_MAX_CONTEXT_SIZE_BP
    )
    dependenceNumBlocks: int | None = OBSERVATION_DEFAULT_DEPENDENCE_NUM_BLOCKS
    dependenceBlockMedianBP: float | None = (
        OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_MEDIAN_BP
    )
    dependenceBlockSigma: float | None = OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_SIGMA
    dependenceBlockMinBP: int | None = OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_MIN_BP
    dependenceBlockMaxBP: int | None = OBSERVATION_DEFAULT_DEPENDENCE_BLOCK_MAX_BP
    dependencePriorMedianSpan: float | None = (
        OBSERVATION_DEFAULT_DEPENDENCE_PRIOR_MEDIAN_SPAN
    )
    dependencePriorLogSd: float | None = OBSERVATION_DEFAULT_DEPENDENCE_PRIOR_LOG_SD
    dependenceAcfPointThreshold: float | None = (
        OBSERVATION_DEFAULT_DEPENDENCE_ACF_POINT_THRESHOLD
    )
    dependenceAcfRequiredCrossings: int | None = (
        OBSERVATION_DEFAULT_DEPENDENCE_ACF_REQUIRED_CROSSINGS
    )
    dependenceAcfMinEvidenceNats: float | None = (
        OBSERVATION_DEFAULT_DEPENDENCE_ACF_MIN_EVIDENCE_NATS
    )
    muncTrendBlockDependenceMultiplier: float | None = (
        OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER
    )
    muncLocalWindowDependenceMultiplier: float | None = (
        OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER
    )
    muncSeedWeightEnabled: bool | None = OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_ENABLED
    muncSeedWeightPasses: int | None = OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_PASSES
    muncSeedWeightMin: float | None = OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_MIN
    muncSeedWeightMax: float | None = OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_MAX
    muncSeedWeightStudentT: bool | None = OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_T
    muncSeedWeightStudentTdf: float | None = (
        OBSERVATION_DEFAULT_MUNC_SEED_WEIGHT_STUDENT_TDF
    )
    muncSeedProcessMinQ: float | None = OBSERVATION_DEFAULT_MUNC_SEED_PROCESS_MIN_Q
    muncSeedProcessMaxQ: float | None = OBSERVATION_DEFAULT_MUNC_SEED_PROCESS_MAX_Q
    restrictLocalVarianceToSparseBed: bool | None = (
        OBSERVATION_DEFAULT_RESTRICT_LOCAL_VARIANCE_TO_SPARSE_BED
    )
    useCountNoiseFloor: bool | None = OBSERVATION_DEFAULT_USE_COUNT_NOISE_FLOOR
    muncEBPriorTileSizeBP: int | None = OBSERVATION_DEFAULT_MUNC_EB_PRIOR_TILE_SIZE_BP
    muncEBPriorTileCount: int = OBSERVATION_DEFAULT_MUNC_EB_PRIOR_TILE_COUNT
    muncEBPriorStrata: int | None = OBSERVATION_DEFAULT_MUNC_EB_PRIOR_STRATA
    muncEBPriorMinTilesPerStratum: int = (
        OBSERVATION_DEFAULT_MUNC_EB_PRIOR_MIN_TILES_PER_STRATUM
    )
    muncEBPriorSeed: int = OBSERVATION_DEFAULT_MUNC_EB_PRIOR_SEED
    muncEBPriorSupportMinQ: float = OBSERVATION_DEFAULT_MUNC_EB_PRIOR_SUPPORT_MIN_Q
    muncEBPriorSupportMaxQ: float = OBSERVATION_DEFAULT_MUNC_EB_PRIOR_SUPPORT_MAX_Q
    muncEBPriorMaxExtrapolatedFraction: float = (
        OBSERVATION_DEFAULT_MUNC_EB_PRIOR_MAX_EXTRAPOLATED_FRACTION
    )
    muncEBPriorWarmupECMIters: int = OBSERVATION_DEFAULT_MUNC_EB_PRIOR_WARMUP_ECM_ITERS
    muncEBPriorWarmupOuterPasses: int = (
        OBSERVATION_DEFAULT_MUNC_EB_PRIOR_WARMUP_OUTER_PASSES
    )
    muncEBPriorGUncertaintyMode: str | None = (
        OBSERVATION_DEFAULT_MUNC_EB_PRIOR_G_UNCERTAINTY_MODE
    )
    muncCovariatesEnabled: bool | None = OBSERVATION_DEFAULT_MUNC_COVARIATES_ENABLED
    muncCovariatesMode: str | None = OBSERVATION_DEFAULT_MUNC_COVARIATES_MODE
    muncCovariatesFeatures: tuple[str, ...] | None = (
        OBSERVATION_DEFAULT_MUNC_COVARIATE_FEATURES
    )


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
    r"""Parameters for delete-block chromosome state-uncertainty calibration."""

    enabled: bool = UNCERTAINTY_CALIBRATION_DEFAULT_ENABLED
    mode: str = UNCERTAINTY_CALIBRATION_DEFAULT_MODE
    folds: int = UNCERTAINTY_CALIBRATION_DEFAULT_FOLDS
    blockSizeBP: int | str | None = UNCERTAINTY_CALIBRATION_DEFAULT_BLOCK_SIZE_BP
    deleteBlockDeletionProbability: float = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_DELETION_PROBABILITY
    )
    maxScores: int = UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES
    maxHeldoutCells: int | None = UNCERTAINTY_CALIBRATION_DEFAULT_MAX_HELDOUT_CELLS
    maxDiagnosticRows: int = UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS
    minHeldoutCells: int = UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS
    targets: tuple[float, ...] = UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS
    minFactor: float = UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN
    maxFactor: float = UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX
    factorMin: float | None = UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN_OVERRIDE
    factorMax: float | None = UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX_OVERRIDE
    ridge: float = UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE
    calibrationECMIters: int = UNCERTAINTY_CALIBRATION_DEFAULT_CALIBRATION_ECM_ITERS
    calibrationOuterIters: int = UNCERTAINTY_CALIBRATION_DEFAULT_CALIBRATION_OUTER_ITERS
    targetCalibrationDelta: float | None = (
        UNCERTAINTY_CALIBRATION_DEFAULT_TARGET_CALIBRATION_DELTA
    )
    scaleUncertaintyByTargetCalibration: bool = (
        UNCERTAINTY_CALIBRATION_DEFAULT_SCALE_UNCERTAINTY_BY_TARGET_CALIBRATION
    )
    deleteBlockVarianceMode: str = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_VARIANCE_MODE
    )
    deleteBlockUseLambdaInInformation: bool = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_USE_LAMBDA_IN_INFORMATION
    )
    deleteBlockTargetSignal: str = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_TARGET_SIGNAL
    )
    deleteBlockFactorModel: str = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_MODEL
    )
    deleteBlockMinInformationFraction: float = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_MIN_INFORMATION_FRACTION
    )
    deleteBlockMaxInformationFraction: float = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_MAX_INFORMATION_FRACTION
    )
    deleteBlockMinDeltaVariance: float = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_MIN_DELTA_VARIANCE
    )
    deleteBlockFallbackMinValidFraction: float = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FALLBACK_MIN_VALID_FRACTION
    )
    deleteBlockScoreWeightMode: str = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_SCORE_WEIGHT_MODE
    )
    deleteBlockApplyTargetCalibration: bool | None = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_APPLY_TARGET_CALIBRATION
    )
    seed: int = UNCERTAINTY_CALIBRATION_DEFAULT_SEED
    writeDiagnostics: bool = UNCERTAINTY_CALIBRATION_DEFAULT_WRITE_DIAGNOSTICS
    deleteBlockFactorSegmentCount: int = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_SEGMENT_COUNT
    )
    deleteBlockFactorBootstrapReplicates: int = (
        UNCERTAINTY_CALIBRATION_DEFAULT_DELETE_BLOCK_FACTOR_BOOTSTRAP_REPLICATES
    )


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
    the same delete-block state residuals before and after calibration.
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
        ``ffp`` ("first five prime") is BAM-only; for paired-end BAM fragment
        mode it emits exactly one strand-aware event from the first read's
        5-prime end. ``ffp-center`` is a BAM-only preset that resolves to
        read1 input, centered counting, and a per-source 5-prime extension.
        For single-end or per-read BAM mode, ``ffp`` is equivalent to one
        5-prime event per retained alignment.
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
    bamInputMode: str | None = SAM_DEFAULT_BAM_INPUT_MODE
    defaultCountMode: str | None = SAM_DEFAULT_COUNT_MODE
    shiftForward5p: int | None = SAM_DEFAULT_SHIFT_FORWARD_5P
    shiftReverse5p: int | None = SAM_DEFAULT_SHIFT_REVERSE_5P
    extendFrom5pBP: List[int] | int | None = SAM_DEFAULT_EXTEND_FROM_5P_BP
    maxInsertSize: Optional[int] = SAM_DEFAULT_MAX_INSERT_SIZE
    inferFragmentLength: Optional[int] = SAM_DEFAULT_INFER_FRAGMENT_LENGTH
    minMappingQuality: Optional[int] = SAM_DEFAULT_MIN_MAPPING_QUALITY
    minTemplateLength: Optional[int] = SAM_DEFAULT_MIN_TEMPLATE_LENGTH


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
      Inputs default to `coverage`; set `cutsite`, `fiveprime`,
      `ffp`, `ffp-center`, or `center` explicitly for endpoint counting.
    :type countMode: str | None
    :param bamInputMode: Optional BAM interpretation override for this source
    :type bamInputMode: str | None
    :param fragmentPositionMode: Optional fragments endpoint interpretation label
    :type fragmentPositionMode: str | None
    """

    path: str
    sourceKind: str = ALIGNMENT_SOURCE_KINDS[0]
    role: str = INPUT_DEFAULT_ROLE
    sampleName: str | None = None
    barcodeTag: str | None = None
    barcodeAllowListFile: str | None = None
    barcodeGroupMapFile: str | None = None
    selectGroups: List[str] | None = None
    countMode: str | None = None
    bamInputMode: str | None = None
    fragmentPositionMode: str | None = None


class readSegmentsResult(NamedTuple):
    counts: npt.NDArray[np.float32]
    rawNoiseMass: npt.NDArray[np.float32]


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
    :param genomeCovariateCacheDir: Optional directory containing a Consenrich genome covariate cache used by additive genomic MUNC covariance.
    :type genomeCovariateCacheDir: str, optional
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
    genomeCovariateCacheDir: Optional[str]
    chromosomes: List[str]
    excludeChroms: List[str]
    excludeForNorm: List[str]


class countingParams(NamedTuple):
    r"""Parameters related to counting aligned reads

    :param intervalSizeBP: Length (bp) of each genomic interval :math:`i=1\ldots n` that comprise the larger genomic region (contig, chromosome, etc.)
        The default value is generally robust, but users may consider increasing this value when expected feature size is large and/or sequencing depth
        is low (less than :math:`\approx 5 \textsf{million}`, depending on assay).
    :type intervalSizeBP: int
    :param backgroundBlockSizeBP: Length (bp) of blocks used to estimate local statistics (background, noise, etc.). If a negative value is provided (default), this value is inferred from sampled autosomal dependence blocks.
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
    :param logOffset: Log transform input offset. For example,
        :math:`\log(x + 1)` for ``logOffset = 1``.
    :type logOffset: float, optional
    :param logMult: Log transform output scale. For example, setting
        ``logMult = 1 / \log(2)`` yields log2-scaled counts.
    :type logMult: float, optional
    :param transformMethod: Count transform family. Supported values are
        ``log``, ``sqrt``, ``anscombe``, ``asinh``, ``asinhSqrt``,
        ``generalizedLog``, and ``identity``. ``anscombe`` is a preset for
        :math:`2\sqrt{x + 3/8}`. Treatment/control inputs are transformed as
        ``f(treatment) - f(control)``.
    :type transformMethod: str, optional
    :param transformInputOffset: Constant added before transformation. If unset,
        the default is ``logOffset`` for ``log``, ``3/8`` for ``anscombe``,
        and ``0`` for other methods.
    :type transformInputOffset: float, optional
    :param transformInputScale: Positive scale applied before transformation.
        For ``asinhSqrt``, this divides the square-rooted count, so
        ``transformOutputScale = 2`` and ``transformInputScale = s`` gives
        ``2 * asinh(sqrt(x + offset) / s)``.
    :type transformInputScale: float, optional
    :param transformOutputScale: Multiplicative factor applied after
        transformation. If unset for ``log``, ``logMult`` is used; if unset
        for ``anscombe``, ``2`` is used.
    :type transformOutputScale: float, optional
    :param transformOutputOffset: Additive constant applied after unary
        transformation. This offset cancels in treatment/control differences.
    :type transformOutputOffset: float, optional
    :param transformShape: Positive shape/softening constant for
        ``generalizedLog``.
    :type transformShape: float, optional
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
    transformMethod: str | None = COUNTING_DEFAULT_TRANSFORM_METHOD
    transformInputOffset: float | None = COUNTING_DEFAULT_TRANSFORM_INPUT_OFFSET
    transformInputScale: float | None = COUNTING_DEFAULT_TRANSFORM_INPUT_SCALE
    transformOutputScale: float | None = COUNTING_DEFAULT_TRANSFORM_OUTPUT_SCALE
    transformOutputOffset: float | None = COUNTING_DEFAULT_TRANSFORM_OUTPUT_OFFSET
    transformShape: float | None = COUNTING_DEFAULT_TRANSFORM_SHAPE
    centerMB: bool | None = COUNTING_DEFAULT_CENTER_MB
    centerMBMethod: str | None = COUNTING_DEFAULT_CENTER_MB_METHOD


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

    barcodeTag: str | None = SC_DEFAULT_BARCODE_TAG
    defaultCountMode: str | None = SC_DEFAULT_COUNT_MODE
    fragmentsGroupNorm: str | None = SC_DEFAULT_FRAGMENTS_GROUP_NORM
    defaultFragmentPositionMode: str | None = SC_DEFAULT_FRAGMENT_POSITION_MODE


class matchingParams(NamedTuple):
    r"""Parameters related to post-fit peak calling.

    Consenrich uses the within-package dynamic-programming peak caller based on ROCCO.

    :param enabled: If True, run post-fit ROCCO peak calling on the emitted state bedGraph.
    :type enabled: bool
    :param randSeed: Random seed used for bootstrap calibration and any stochastic tie-breaking.
    :type randSeed: Optional[int]
    :param numBootstrap: Number of dependent wild-bootstrap null draws used for budget calibration.
    :type numBootstrap: Optional[int]
    :param thresholdZ: One-sided null tail threshold on the ROCCO score, on a Gaussian ``z`` scale.
    :type thresholdZ: Optional[float]
    :param dependenceSpan: Optional fixed dependence span in intervals for bootstrap calibration.
        If not provided, it is estimated from the score track.
    :type dependenceSpan: Optional[int]
    :param gamma: ROCCO boundary penalty. Non-negative values are used directly
        (default ``0.25``); negative values request data-driven estimation from
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
    :param nestedRoccoBudgetScale: Soft budget scale for each eligible parent peak
        region in nested ROCCO. Values below ``1`` increase the local per-bin
        selection penalty and bias the child solution smaller, but they are not
        a hard quota. The default is ``0.75``.
    :type nestedRoccoBudgetScale: Optional[float]
    :param exportFilterUncertaintyMultiplier: Non-negative multiplier ``c`` in the
        final export filter ``medianState < -c * median(local uncertainty)``.
        The default follows the package matching defaults in
        :mod:`consenrich.constants`. Setting ``c=0`` requires exported peaks to
        have positive median signal.
    :type exportFilterUncertaintyMultiplier: Optional[float]
    :param uncertaintyScoreMode: ROCCO score construction mode. ``"state"`` uses
        the fitted state directly. ``"lower_confidence"`` uses
        ``state - uncertaintyScoreZ * uncertainty`` and requires an uncertainty track.
    :type uncertaintyScoreMode: str
    :param uncertaintyScoreZ: Non-negative multiplier for ``"lower_confidence"``
        score construction.
    :type uncertaintyScoreZ: float
    :param metadataDetail: Detail level for ROCCO metadata JSON. ``"compact"``
        keeps summary diagnostics, while ``"full"`` also includes per-peak and
        candidate-detail arrays.
    :type metadataDetail: str
    :param useShrunkStateScores: If True, integrated ROCCO peak calling uses the
        post-fit EB-shrunken state track as its score input and the shrinkage
        posterior standard deviation as the paired uncertainty track. This only
        affects ROCCO scoring/export and does not alter the Kalman/ECM fit.
    :type useShrunkStateScores: bool
    :seealso: :mod:`consenrich.peaks`, :class:`outputParams`.
    """

    enabled: bool
    randSeed: Optional[int]
    numBootstrap: Optional[int]
    thresholdZ: Optional[float]
    dependenceSpan: Optional[int]
    gamma: Optional[float]
    selectionPenalty: Optional[float]
    gammaScale: Optional[float]
    nestedRoccoIters: Optional[int]
    nestedRoccoBudgetScale: Optional[float]
    exportFilterUncertaintyMultiplier: Optional[float]
    uncertaintyScoreMode: str = MATCHING_DEFAULT_UNCERTAINTY_SCORE_MODE
    uncertaintyScoreZ: float = MATCHING_DEFAULT_UNCERTAINTY_SCORE_Z
    metadataDetail: str = MATCHING_DEFAULT_METADATA_DETAIL
    minPeakScore: Optional[float] = MATCHING_DEFAULT_MIN_PEAK_SCORE
    useShrunkStateScores: bool = MATCHING_DEFAULT_USE_SHRUNK_STATE_SCORES
    peakMode: str = MATCHING_DEFAULT_PEAK_MODE
    broadWeakThresholdZ: float = MATCHING_DEFAULT_BROAD_WEAK_THRESHOLD_Z
    broadMaxGapBP: Optional[int] = MATCHING_DEFAULT_BROAD_MAX_GAP_BP


class outputParams(NamedTuple):
    r"""Parameters related to output files.

    :param convertToBigWig: If True, output bedGraph files are converted to bigWig format.
    :type convertToBigWig: bool
    :param roundDigits: Number of decimal places to round output values (bedGraph)
    :type roundDigits: int
    :param writeUncertainty: If True, write the state uncertainty track to bedGraph.
        The default uncalibrated track is :math:`\sqrt{\widetilde{P}_{[i,0,0]}}`;
        when uncertainty calibration is enabled, the caller may replace it with the
        delete-block calibrated state-variance track.
    :type writeUncertainty: bool
    :param writeStateShrinkage: If True, write post-fit empirical-Bayes state
        shrinkage tracks: ``stateShrunk``, ``stateShrinkageFactor``, and
        ``stateNullProbability``. Shrinkage is applied only to reported tracks and
        does not affect the Kalman/ECM fit.
    :type writeStateShrinkage: bool
    :param stateShrinkageModel: Post-fit state shrinkage model. The default
        ``"adaptiveNormalMixture"`` uses a point mass at zero plus several
        zero-centered Normal slabs, with genome-level EB hyperparameters.
    :type stateShrinkageModel: str
    :param stateShrinkagePriorNull: Optional fixed point-null prior probability
        for state shrinkage. If unset, it is estimated by empirical Bayes.
    :type stateShrinkagePriorNull: Optional[float]
    :param stateShrinkagePriorScale: Optional fixed prior scale. For
        ``"spikeAndNormal"`` this fixes the slab standard deviation.
    :type stateShrinkagePriorScale: Optional[float]
    :param saveBackgroundTracks: If True, write the fitted shared background
        track :math:`g_{[i]}` to bedGraph and optional bigWig output.
    :type saveBackgroundTracks: bool
    :param saveGains: If True, write a genome-wide per-replicate final
        forward-pass gain summary TSV.
    :type saveGains: bool
    :param plotOptimizationPath: If True, write and optionally plot the objective
        trace for outer/background passes and fixed-background ECM iterations.
    :type plotOptimizationPath: bool
    :param diagnosticTracks: Extra per-interval diagnostic tracks to write as
        bedGraph and optional bigWig outputs. Supported names include ``slope``,
        ``baseQLevel``, ``baseQTrend``, ``preKappaQLevel``,
        ``preKappaQTrend``, ``effectiveQLevel``, ``effectiveQTrend``,
        ``puncQScale``, ``muncTrace``, ``sumGain0``, and ``sumGain1``.
    :type diagnosticTracks: tuple[str, ...]
    :param writeRunSummary: If True, write one high-level run summary TSV.
    :type writeRunSummary: bool
    :param precisionDiagnosticDetail: Detail level for dense precision diagnostic
        diagnostic logs. ``"summary"`` writes only compact summaries, ``"sampled"``
        writes summaries plus deterministic interval samples, and ``"full"`` writes
        all interval rows.
    :type precisionDiagnosticDetail: str
    :param maxPrecisionDiagnosticRowsPerChromosome: Maximum interval rows to write
        per chromosome when ``precisionDiagnosticDetail`` is ``"sampled"``.
    :type maxPrecisionDiagnosticRowsPerChromosome: int
    :param maxNonTrackFileBytes: Default size budget for non-track files.
        Exact ``.bw``, ``.bedGraph``, and ``.narrowPeak`` track outputs are exempt.
        Set to 0 to disable the cap.
    :type maxNonTrackFileBytes: int

    """

    convertToBigWig: bool
    roundDigits: int
    writeUncertainty: bool
    writeStateShrinkage: bool = OUTPUT_DEFAULT_WRITE_STATE_SHRINKAGE
    stateShrinkageModel: str = OUTPUT_DEFAULT_STATE_SHRINKAGE_MODEL
    stateShrinkagePriorSpikeProp: Optional[float] = (
        OUTPUT_DEFAULT_STATE_SHRINKAGE_PRIOR_SPIKE_PROP
    )
    stateShrinkagePriorScale: Optional[float] = (
        OUTPUT_DEFAULT_STATE_SHRINKAGE_PRIOR_SCALE
    )
    stateShrinkageSpikePseudoCount: Optional[float] = (
        OUTPUT_DEFAULT_STATE_SHRINKAGE_SPIKE_PSEUDO_COUNT
    )
    stateShrinkageScaleAnchorWeight: Optional[float] = (
        OUTPUT_DEFAULT_STATE_SHRINKAGE_SCALE_ANCHOR_WEIGHT
    )
    saveBackgroundTracks: bool = OUTPUT_DEFAULT_SAVE_BACKGROUND_TRACKS
    plotOptimizationPath: bool = OUTPUT_DEFAULT_PLOT_OPTIMIZATION_PATH
    plotCorrelationLength: bool = OUTPUT_DEFAULT_PLOT_CORRELATION_LENGTH
    diagnosticTracks: Tuple[str, ...] = OUTPUT_DEFAULT_DIAGNOSTIC_TRACKS
    saveGains: bool = OUTPUT_DEFAULT_SAVE_GAINS
    cutoffReport: bool = OUTPUT_DEFAULT_CUTOFF_REPORT
    writeRunSummary: bool = OUTPUT_DEFAULT_WRITE_RUN_SUMMARY
    precisionDiagnosticDetail: str = OUTPUT_DEFAULT_PRECISION_DIAGNOSTIC_DETAIL
    maxPrecisionDiagnosticRowsPerChromosome: int = (
        OUTPUT_DEFAULT_MAX_PRECISION_DIAGNOSTIC_ROWS_PER_CHROMOSOME
    )
    maxNonTrackFileBytes: int = OUTPUT_DEFAULT_MAX_NON_TRACK_FILE_BYTES
    stateShrinkageStudentTDF: float = OUTPUT_DEFAULT_STATE_SHRINKAGE_STUDENT_T_DF
    stateShrinkageStudentTQuadratureOrder: int = (
        OUTPUT_DEFAULT_STATE_SHRINKAGE_STUDENT_T_QUADRATURE_ORDER
    )


class loggingParams(NamedTuple):
    verbosity: str = LOGGING_DEFAULT_VERBOSITY
    progress: str = LOGGING_DEFAULT_PROGRESS
    logFile: str | None = LOGGING_DEFAULT_LOG_FILE


class fitParams(NamedTuple):
    r"""Parameters controlling the optimization/fitting procedures.

    These arguments control both the fixed-background ECM routine in
    :func:`consenrich.cconsenrich.cfixedBackgroundECM` and the outer
    fit/background alternation loop in :func:`consenrich.core.runConsenrich`.

    Fixed-background ECM:

    1. Filter-smoother state estimation *given* current noise scales
    2. Interval-level Student-t precision reweighting at: \(\lambda_{[i]}\) and \(\kappa_{[i]}\)

    Outer alternation:

    1. run the fixed-background ECM path against the current shared background
    2. optionally update a shared low-frequency background track \(g_{[i]}\),
       optionally constrained to have mean zero

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
    :param useNonnegativeBackground: If True, discourage negative shared
      background values with a soft asymmetric quadratic penalty.
    :type useNonnegativeBackground: bool
    :param backgroundNegativePenaltyMultiplier: Multiplier for the soft
      negative-background penalty. ``None`` disables the penalty.
    :type backgroundNegativePenaltyMultiplier: float | None
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

    ECM_fixedBackgroundIters: int | None = FIT_DEFAULT_FIXED_BACKGROUND_ITERS
    ECM_fixedBackgroundRtol: float | None = FIT_DEFAULT_FIXED_BACKGROUND_RTOL
    t_innerIters: int | None = FIT_DEFAULT_T_INNER_ITERS
    ECM_robustTNu: float | None = FIT_DEFAULT_ROBUST_T_NU
    ECM_useObsPrecisionReweighting: bool | None = (
        FIT_DEFAULT_USE_OBS_PRECISION_REWEIGHTING
    )
    ECM_useProcessPrecisionReweighting: bool | None = (
        FIT_DEFAULT_USE_PROCESS_PRECISION_REWEIGHTING
    )
    ECM_useAPN: bool | None = FIT_DEFAULT_USE_APN
    ECM_zeroCenterBackground: bool | None = FIT_DEFAULT_ZERO_CENTER_BACKGROUND
    ECM_outerIters: int | None = FIT_DEFAULT_OUTER_ITERS
    ECM_minOuterIters: int | None = FIT_DEFAULT_MIN_OUTER_ITERS
    ECM_backgroundShiftRtol: float | None = FIT_DEFAULT_BACKGROUND_SHIFT_RTOL
    ECM_outerNLLRtol: float | None = FIT_DEFAULT_OUTER_NLL_RTOL
    ECM_backgroundSmoothness: float | None = FIT_DEFAULT_BACKGROUND_SMOOTHNESS
    ECM_backgroundLengthScaleMultiplier: float | None = (
        FIT_DEFAULT_BACKGROUND_LENGTH_SCALE_MULTIPLIER
    )
    fitBackground: bool | None = FIT_DEFAULT_BACKGROUND
    useNonnegativeBackground: bool | None = FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND
    backgroundNegativePenaltyMultiplier: float | None = (
        FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER
    )


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
    return _stableChromRange(
        start,
        end,
        chromLength,
        chromosome=chromosome,
        sourcePath=bamFile,
        sourceKind=sourceKind,
    )


def _stableChromRange(
    start: int,
    end: int,
    chromLength: int,
    *,
    chromosome: str = "",
    sourcePath: str = "",
    sourceKind: str = "",
) -> Tuple[int, int]:
    """Clamp native chromosome ranges and guard against sparse-tail misses."""

    chromLength_ = max(0, int(chromLength))
    if chromLength_ == 0:
        return 0, 0
    start_ = min(max(0, int(start)), chromLength_)
    end_ = min(max(0, int(end)), chromLength_)
    if end_ > start_:
        return start_, end_
    if start_ > 0 or end_ > 0:
        logger.warning(
            "chromosome range fallback: source=%s kind=%s chromosome=%s "
            "native_range=(%d,%d) chrom_length=%d; using full chromosome",
            sourcePath,
            sourceKind,
            chromosome,
            int(start),
            int(end),
            chromLength_,
        )
        return 0, chromLength_
    return 0, 0


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
    return _sharedNormalizeCountMode(countMode, defaultMode)


def _nativeCountModeForPreset(countMode: str) -> str:
    return _sharedNativeCountModeForPreset(countMode)


def _resolveSourceBamInputModeForCountMode(
    source: inputSource,
    defaultBamInputMode: str | None,
    countMode: str,
) -> str:
    sourceBamInputMode = _resolveSourceBamInputMode(source, defaultBamInputMode)
    if countMode != "ffp-center":
        return sourceBamInputMode
    configuredMode = _normalizeBamInputMode(source.bamInputMode or defaultBamInputMode)
    if configuredMode not in {"auto", "read1"}:
        raise ValueError(
            "countMode `ffp-center` resolves to `bamInputMode=read1`; "
            "`bamInputMode` must be `auto` or `read1` when this preset is used"
        )
    return "read1"


def _normalizeBamInputMode(bamInputMode: str | None) -> str:
    return _sharedNormalizeBamInputMode(bamInputMode, default="auto")


def _normalizeFragmentPositionMode(fragmentPositionMode: str | None) -> str:
    return _sharedNormalizeFragmentPositionMode(fragmentPositionMode)


@lru_cache(maxsize=None)
def _isAlignmentSourcePairedEnd(alignmentPath: str) -> bool:
    return bool(
        cconsenrich.cisAlignmentPairedEnd(
            alignmentPath,
            maxReads=1_000,
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


def transformCountVarianceFloor(
    normalizedCounts: np.ndarray,
    scaleFactors: npt.ArrayLike,
    *,
    rawNoiseMass: np.ndarray | None = None,
    countNoisePseudoMeanMass: float = 0.5,
    countNoisePseudoVarianceMass: float = 0.5,
    transformMethod: str | None = COUNTING_DEFAULT_TRANSFORM_METHOD,
    logOffset: float | None = COUNTING_DEFAULT_LOG_OFFSET,
    logMult: float | None = COUNTING_DEFAULT_LOG_MULT,
    transformInputOffset: float | None = COUNTING_DEFAULT_TRANSFORM_INPUT_OFFSET,
    transformInputScale: float | None = COUNTING_DEFAULT_TRANSFORM_INPUT_SCALE,
    transformOutputScale: float | None = COUNTING_DEFAULT_TRANSFORM_OUTPUT_SCALE,
    transformShape: float | None = COUNTING_DEFAULT_TRANSFORM_SHAPE,
) -> np.ndarray:
    r"""Approximate transformation-induced count variance for MUNC.

    The floor is the delta-method variance of the active count transform under
    a conditional Poisson count model with Jeffreys-rate smoothing. For scaled
    count ``c`` and scale factor ``s``, ``lambdaHat = c / s + 1/2`` and
    ``Var(sY | lambdaHat) = s^2 * lambdaHat``.
    """

    return cconsenrich.cTransformCountVarianceFloor(
        normalizedCounts,
        scaleFactors,
        rawNoiseMass=rawNoiseMass,
        countNoisePseudoMeanMass=countNoisePseudoMeanMass,
        countNoisePseudoVarianceMass=countNoisePseudoVarianceMass,
        mode=transformMethod,
        logOffset=logOffset,
        logMult=logMult,
        inputOffset=transformInputOffset,
        inputScale=transformInputScale,
        outputScale=transformOutputScale,
        shape=transformShape,
    )


def transformCountDifferenceVarianceFloor(
    treatmentCounts: np.ndarray,
    controlCounts: np.ndarray,
    *,
    treatmentScaleFactor: float,
    controlScaleFactor: float,
    treatmentRawNoiseMass: np.ndarray | None = None,
    controlRawNoiseMass: np.ndarray | None = None,
    countNoisePseudoMeanMass: float = 0.5,
    countNoisePseudoVarianceMass: float = 0.5,
    transformMethod: str | None = COUNTING_DEFAULT_TRANSFORM_METHOD,
    logOffset: float | None = COUNTING_DEFAULT_LOG_OFFSET,
    logMult: float | None = COUNTING_DEFAULT_LOG_MULT,
    transformInputOffset: float | None = COUNTING_DEFAULT_TRANSFORM_INPUT_OFFSET,
    transformInputScale: float | None = COUNTING_DEFAULT_TRANSFORM_INPUT_SCALE,
    transformOutputScale: float | None = COUNTING_DEFAULT_TRANSFORM_OUTPUT_SCALE,
    transformShape: float | None = COUNTING_DEFAULT_TRANSFORM_SHAPE,
) -> np.ndarray:
    r"""Approximate variance floor for independent treatment-control differences."""

    treatmentFloor = transformCountVarianceFloor(
        treatmentCounts,
        [float(treatmentScaleFactor)],
        rawNoiseMass=treatmentRawNoiseMass,
        countNoisePseudoMeanMass=countNoisePseudoMeanMass,
        countNoisePseudoVarianceMass=countNoisePseudoVarianceMass,
        transformMethod=transformMethod,
        logOffset=logOffset,
        logMult=logMult,
        transformInputOffset=transformInputOffset,
        transformInputScale=transformInputScale,
        transformOutputScale=transformOutputScale,
        transformShape=transformShape,
    )
    controlFloor = transformCountVarianceFloor(
        controlCounts,
        [float(controlScaleFactor)],
        rawNoiseMass=controlRawNoiseMass,
        countNoisePseudoMeanMass=countNoisePseudoMeanMass,
        countNoisePseudoVarianceMass=countNoisePseudoVarianceMass,
        transformMethod=transformMethod,
        logOffset=logOffset,
        logMult=logMult,
        transformInputOffset=transformInputOffset,
        transformInputScale=transformInputScale,
        transformOutputScale=transformOutputScale,
        transformShape=transformShape,
    )
    treat = np.asarray(treatmentFloor, dtype=np.float64)
    control = np.asarray(controlFloor, dtype=np.float64)
    finiteTreat = np.isfinite(treat)
    finiteControl = np.isfinite(control)
    out = np.full(
        np.broadcast_shapes(treat.shape, control.shape), np.nan, dtype=np.float64
    )
    treatB = np.broadcast_to(treat, out.shape)
    controlB = np.broadcast_to(control, out.shape)
    finiteTreatB = np.broadcast_to(finiteTreat, out.shape)
    finiteControlB = np.broadcast_to(finiteControl, out.shape)
    out[finiteTreatB] = treatB[finiteTreatB]
    out[finiteControlB] = np.where(
        np.isfinite(out[finiteControlB]),
        out[finiteControlB] + controlB[finiteControlB],
        controlB[finiteControlB],
    )
    return np.asarray(out, dtype=np.float32)


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
    defaultCountMode: str | None = SAM_DEFAULT_COUNT_MODE,
    defaultFragmentCountMode: str | None = SC_DEFAULT_COUNT_MODE,
    shiftForward5p: int | None = 0,
    shiftReverse5p: int | None = 0,
    extendFrom5pBP: List[int] | int | None = None,
    maxInsertSize: Optional[int] = 1000,
    inferFragmentLength: Optional[int] = 0,
    minMappingQuality: Optional[int] = 0,
    minTemplateLength: Optional[int] = -1,
    returnRawNoiseMass: bool = False,
) -> npt.NDArray[np.float32] | readSegmentsResult:
    r"""Read binned tracks from generic input sources

    this is the source-agnostic entry point for counting.

    For BAM inputs, ``bamInputMode`` controls whether we count template spans, per-read alignments,
    or only read1 tags from paired-end BAM. ``countMode="ffp"`` counts exactly one
    strand-aware first-read 5-prime event for paired-end template-span inputs, while ``fiveprime``
    retains endpoint semantics. ``countMode="ffp-center"`` is a BAM-only preset that counts
    read1 centers after extending from the read1 5-prime end by the configured or estimated
    per-source fragment length. ``defaultFragmentCountMode`` provides the corresponding default
    for fragments sources when they do not set a source-level ``countMode``. Combine
    ``shiftForward5p`` / ``shiftReverse5p`` with
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
    rawNoiseMass = (
        np.full((len(sources), numIntervals), np.nan, dtype=np.float32)
        if returnRawNoiseMass
        else None
    )
    tempPaths: list[str] = []
    defaultBamInputMode = _normalizeBamInputMode(bamInputMode)
    defaultBamCountMode = _normalizeCountMode(
        defaultCountMode,
        SAM_DEFAULT_COUNT_MODE,
    )
    defaultFragmentCountMode = _normalizeCountMode(
        defaultFragmentCountMode,
        SC_DEFAULT_COUNT_MODE,
    )
    if defaultFragmentCountMode in {"ffp", "ffp-center"}:
        raise ValueError(
            f"defaultFragmentCountMode `{defaultFragmentCountMode}` requires BAM input"
        )
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
                countMode = _normalizeCountMode(
                    source.countMode, defaultFragmentCountMode
                )
                if countMode in {"ffp", "ffp-center"}:
                    raise ValueError(f"countMode `{countMode}` requires BAM input")
                if (
                    countMode == COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
                    and int(oneReadPerBin) != 0
                ):
                    raise ValueError(
                        "conservedFractionalOverlap count mode does not support "
                        "oneReadPerBin"
                    )
                _normalizeFragmentPositionMode(source.fragmentPositionMode)
                if returnRawNoiseMass:
                    countResult = ccounts.ccounts_countAlignmentRegionMass(
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
                    counts[sourceIndex, :] = countResult.counts
                    if rawNoiseMass is None:
                        raise RuntimeError("raw noise mass matrix missing")
                    rawNoiseMass[sourceIndex, :] = countResult.rawNoiseMass
                else:
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
                if (
                    countMode == COUNT_MODE_CONSERVED_FRACTIONAL_OVERLAP
                    and int(oneReadPerBin) != 0
                ):
                    raise ValueError(
                        "conservedFractionalOverlap count mode does not support "
                        "oneReadPerBin"
                    )
                ffpCenterPreset = countMode == "ffp-center"
                sourceBamInputMode = _resolveSourceBamInputModeForCountMode(
                    source,
                    defaultBamInputMode,
                    countMode,
                )
                if ffpCenterPreset:
                    countMode = _nativeCountModeForPreset(countMode)
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
                elif (
                    ffpCenterPreset or sourceInferFragmentLength > 0
                ) and sourceExtendBP <= 0:
                    sourceExtendBP = int(
                        cconsenrich.cgetFragmentLength(
                            sourcePath,
                            samThreads=samThreads,
                            samFlagExclude=sourceFlagExclude,
                            maxInsertSize=maxInsertSize,
                        )
                    )
                if ffpCenterPreset and sourceExtendBP <= 0:
                    raise ValueError(
                        "countMode `ffp-center` requires a positive "
                        "`extendFrom5pBP` or estimable fragment length"
                    )
                sourceInferFragmentLength = 0
                if returnRawNoiseMass:
                    countResult = ccounts.ccounts_countAlignmentRegionMass(
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
                    counts[sourceIndex, :] = countResult.counts
                    if rawNoiseMass is None:
                        raise RuntimeError("raw noise mass matrix missing")
                    rawNoiseMass[sourceIndex, :] = countResult.rawNoiseMass
                else:
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
    if returnRawNoiseMass:
        if rawNoiseMass is None:
            raise RuntimeError("raw noise mass matrix missing")
        return readSegmentsResult(counts, rawNoiseMass)
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
    defaultCountMode: str | None = SAM_DEFAULT_COUNT_MODE,
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


def _normalizeStateModel(stateModel: str | None) -> str:
    if stateModel is None:
        return STATE_MODEL_LEVEL_TREND

    mode = str(stateModel).strip()
    if mode == STATE_MODEL_LEVEL_TREND:
        return STATE_MODEL_LEVEL_TREND
    if mode == STATE_MODEL_LEVEL:
        return STATE_MODEL_LEVEL

    raise ValueError(
        "`stateModel` must be one of "
        + ", ".join(repr(mode_) for mode_ in STATE_MODEL_MODES)
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


def _processKappaConvexityLowerBound(
    *,
    robustTNu: float | None,
    stateDim: int,
) -> float:
    if robustTNu is None:
        raise ValueError(
            "`fitParams.ECM_robustTNu` must be positive and finite when "
            "`processParams.precisionMultiplierMin` is negative."
        )
    nu = _checkFinitePositive("ECM_robustTNu", robustTNu)
    stateDim_ = int(stateDim)
    if stateDim_ <= 0:
        raise ValueError("state dimension must be positive")
    return (nu + float(stateDim_)) / (2.0 * nu) + 1.0e-4


def _checkProcessPrecisionMultiplierBounds(
    *,
    minValue: float,
    maxValue: float,
    robustTNu: float | None,
    stateDim: int,
) -> tuple[float, float]:
    maxValue_ = _checkFinitePositive("processPrecisionMultiplierMax", maxValue)
    minValueRaw = float(minValue)
    if not np.isfinite(minValueRaw):
        raise ValueError("`processPrecisionMultiplierMin` must be finite")
    if minValueRaw < 0.0:
        convexityMin = _processKappaConvexityLowerBound(
            robustTNu=robustTNu,
            stateDim=stateDim,
        )
        if maxValue_ < convexityMin:
            raise ValueError(
                "`processPrecisionMultiplierMax` must be >= the auto "
                "`processPrecisionMultiplierMin` convexity-preserving lower "
                f"bound {float(convexityMin):.6g}"
            )
        minValue_ = float(convexityMin)
        logger.info(
            "processParams.precisionMultiplierMin=auto resolved to %.6g "
            "using strict convexity-preserving bound %.6g "
            "(ECM_robustTNu=%.6g, stateDim=%d, precisionMultiplierMax=%.6g)",
            float(minValue_),
            float(convexityMin),
            float(robustTNu),
            int(stateDim),
            float(maxValue_),
        )
    else:
        minValue_ = _checkFinitePositive(
            "processPrecisionMultiplierMin",
            minValueRaw,
        )
    if maxValue_ < minValue_:
        raise ValueError(
            "`processPrecisionMultiplierMax` must be >= "
            "`processPrecisionMultiplierMin`"
        )
    return minValue_, maxValue_


def _warnIfProcessKappaMinAllowsProfiledNonconvexity(
    *,
    kappaMin: float,
    kappaMax: float,
    robustTNu: float | None,
    stateDim: int,
    processReweightingEnabled: bool,
) -> None:
    if not processReweightingEnabled:
        return
    if robustTNu is None:
        return
    try:
        threshold = _processKappaConvexityLowerBound(
            robustTNu=robustTNu,
            stateDim=stateDim,
        )
    except ValueError:
        return


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


def _processKappaSummary(
    processPrecExp: np.ndarray | None,
    *,
    lower: float,
    upper: float,
) -> tuple[Any, Any]:
    if processPrecExp is None:
        return None, None
    arr = np.asarray(processPrecExp, dtype=np.float64).reshape(-1)
    if arr.size > 1:
        arr = arr[1:]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    clipped = np.clip(finite, float(lower), float(upper))
    return (
        metadataFloat(float(np.mean(clipped))),
        metadataFloat(float(np.median(clipped))),
    )


def _processQPolicy(
    *,
    useAPN: bool,
    processPrecisionEffective: bool,
) -> str:
    if bool(useAPN):
        return "adaptive_process_noise"
    if bool(processPrecisionEffective):
        return "student_t_kappa"
    return "base"


def _metadataTrackSummary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "min": None,
            "q05": None,
            "median": None,
            "mean": None,
            "q95": None,
            "max": None,
        }
    return {
        "min": metadataFloat(float(np.min(finite))),
        "q05": metadataFloat(float(np.quantile(finite, 0.05))),
        "median": metadataFloat(float(np.median(finite))),
        "mean": metadataFloat(float(np.mean(finite))),
        "q95": metadataFloat(float(np.quantile(finite, 0.95))),
        "max": metadataFloat(float(np.max(finite))),
    }


def _metadataTrackSummaryWhere(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    maskArr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.shape != maskArr.shape:
        return _metadataTrackSummary(np.asarray([], dtype=np.float64))
    return _metadataTrackSummary(arr[maskArr])


def _processQTrackArrays(
    *,
    matrixQ0: np.ndarray,
    intervalCount: int,
    stateModel: str,
    processPrecExp: np.ndarray | None,
    processQScale: np.ndarray | None,
    pNoiseForward: np.ndarray | None,
    procPrecisionMultiplierMin: float,
    procPrecisionMultiplierMax: float,
    returnFullQ: bool = True,
) -> dict[str, np.ndarray]:
    intervalCount_ = int(intervalCount)
    if intervalCount_ < 0:
        raise ValueError("intervalCount must be nonnegative")

    stateModelMode = _normalizeStateModel(stateModel)
    stateDim = 1 if stateModelMode == STATE_MODEL_LEVEL else 2
    q0 = np.asarray(matrixQ0, dtype=np.float64)
    if q0.ndim != 2 or q0.shape[0] < stateDim or q0.shape[1] < stateDim:
        raise ValueError("matrixQ0 shape does not match stateModel")
    baseQ = q0[:stateDim, :stateDim]

    qScale = np.ones(intervalCount_, dtype=np.float64)
    if processQScale is not None:
        qScale = np.asarray(processQScale, dtype=np.float64).reshape(-1)
        if qScale.shape != (intervalCount_,):
            raise ValueError("processQScale length must match interval count")
        if not np.all(np.isfinite(qScale)):
            raise ValueError("processQScale contains non-finite values")
        qScale = np.maximum(qScale, np.finfo(np.float64).tiny)
        if intervalCount_:
            qScale = qScale.copy()
            qScale[0] = 1.0

    baseQLevel = np.full(intervalCount_, float(baseQ[0, 0]), dtype=np.float64)
    baseQTrend = np.zeros(intervalCount_, dtype=np.float64)
    if stateDim == 2:
        baseQTrend.fill(float(baseQ[1, 1]))
    preKappaQLevel = baseQLevel * qScale
    preKappaQTrend = baseQTrend * qScale
    effectiveQLevel = preKappaQLevel.copy()
    effectiveQTrend = preKappaQTrend.copy()

    preKappaQ = None
    effectiveQ = None
    if bool(returnFullQ):
        preKappaQ = np.empty((intervalCount_, stateDim, stateDim), dtype=np.float64)
        if intervalCount_:
            preKappaQ[:, :, :] = baseQ[None, :, :] * qScale[:, None, None]
        effectiveQ = preKappaQ.copy()

    if processPrecExp is not None:
        procPrecision = np.asarray(processPrecExp, dtype=np.float64).reshape(-1)
        if procPrecision.shape != (intervalCount_,):
            raise ValueError("processPrecExp length must match interval count")
        if not np.all(np.isfinite(procPrecision)):
            raise ValueError("processPrecExp contains non-finite values")
        procPrecision = np.clip(
            procPrecision,
            float(procPrecisionMultiplierMin),
            float(procPrecisionMultiplierMax),
        )
        procPrecision = np.maximum(procPrecision, np.finfo(np.float64).tiny)
        if intervalCount_:
            effectiveQLevel = preKappaQLevel / procPrecision
            effectiveQTrend = preKappaQTrend / procPrecision
            if effectiveQ is not None:
                effectiveQ[:, :, :] = preKappaQ / procPrecision[:, None, None]
    elif pNoiseForward is not None:
        pNoise = np.asarray(pNoiseForward)
        if (
            pNoise.ndim != 3
            or pNoise.shape[0] < max(intervalCount_ - 1, 0)
            or pNoise.shape[1] < stateDim
            or pNoise.shape[2] < stateDim
        ):
            raise ValueError("pNoiseForward shape does not match stateModel")
        for k in range(1, intervalCount_):
            qEff = pNoise[k - 1, :stateDim, :stateDim]
            if np.all(np.isfinite(qEff)):
                effectiveQLevel[k] = float(qEff[0, 0])
                if stateDim == 2:
                    effectiveQTrend[k] = float(qEff[1, 1])
                if effectiveQ is not None:
                    effectiveQ[k, :, :] = qEff

    tracks = {
        "baseQLevel": baseQLevel,
        "baseQTrend": baseQTrend,
        "preKappaQLevel": preKappaQLevel,
        "preKappaQTrend": preKappaQTrend,
        "effectiveQLevel": effectiveQLevel,
        "effectiveQTrend": effectiveQTrend,
        "puncQScale": qScale,
    }
    if bool(returnFullQ):
        tracks["preKappaQ"] = preKappaQ
        tracks["effectiveQ"] = effectiveQ
    return tracks


def _processQDiagnosticsMetadata(
    *,
    matrixQ0: np.ndarray,
    intervalCount: int,
    stateModel: str,
    processPrecExp: np.ndarray | None,
    processQScale: np.ndarray | None,
    pNoiseForward: np.ndarray | None,
    useAPN: bool,
    processPrecisionRequested: bool,
    processPrecisionEffective: bool,
    procPrecisionMultiplierMin: float,
    procPrecisionMultiplierMax: float,
) -> dict[str, Any]:
    qTracks = _processQTrackArrays(
        matrixQ0=matrixQ0,
        intervalCount=int(intervalCount),
        stateModel=stateModel,
        processPrecExp=processPrecExp,
        processQScale=processQScale,
        pNoiseForward=pNoiseForward,
        procPrecisionMultiplierMin=float(procPrecisionMultiplierMin),
        procPrecisionMultiplierMax=float(procPrecisionMultiplierMax),
        returnFullQ=False,
    )
    baseLevel = float(np.asarray(matrixQ0, dtype=np.float64)[0, 0])
    baseTrend = (
        0.0
        if _normalizeStateModel(stateModel) == STATE_MODEL_LEVEL
        else float(np.asarray(matrixQ0, dtype=np.float64)[1, 1])
    )
    preLevelSummary = _metadataTrackSummary(qTracks["preKappaQLevel"])
    preTrendSummary = _metadataTrackSummary(qTracks["preKappaQTrend"])
    levelSummary = _metadataTrackSummary(qTracks["effectiveQLevel"])
    trendSummary = _metadataTrackSummary(qTracks["effectiveQTrend"])
    qTraceSummary = _metadataTrackSummary(
        qTracks["effectiveQLevel"] + qTracks["effectiveQTrend"]
    )
    return {
        "policy": _processQPolicy(
            useAPN=bool(useAPN),
            processPrecisionEffective=bool(processPrecisionEffective),
        ),
        "apn_enabled": bool(useAPN),
        "process_precision_reweighting_requested": bool(processPrecisionRequested),
        "process_precision_reweighting_effective": bool(processPrecisionEffective),
        "process_precision_reweighting_disabled_by_apn": bool(
            processPrecisionRequested and useAPN and not processPrecisionEffective
        ),
        "baseQLevel": metadataFloat(baseLevel),
        "baseQTrend": metadataFloat(baseTrend),
        "preKappaQLevel": preLevelSummary,
        "preKappaQTrend": preTrendSummary,
        "effectiveQLevel": levelSummary,
        "effectiveQTrend": trendSummary,
        "effectiveQTrace": qTraceSummary,
        "puncQScale": _metadataTrackSummary(qTracks["puncQScale"]),
        "effectiveQLevelMedian": levelSummary["median"],
        "effectiveQTrendMedian": trendSummary["median"],
        "effectiveQTraceMedian": qTraceSummary["median"],
        "effectiveQLevelMin": levelSummary["min"],
        "effectiveQTrendMin": trendSummary["min"],
        "effectiveQTraceMin": qTraceSummary["min"],
        "effectiveQLevelMax": levelSummary["max"],
        "effectiveQTrendMax": trendSummary["max"],
        "effectiveQTraceMax": qTraceSummary["max"],
    }


def _precisionBoundHits(
    values: np.ndarray | None,
    *,
    lower: float,
    upper: float,
    skipFirst: bool = False,
) -> tuple[Any, Any]:
    if values is None:
        return None, None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if bool(skipFirst) and arr.size > 1:
        arr = arr[1:]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    lower_ = float(lower)
    upper_ = float(upper)
    lowerHits = int(np.count_nonzero(finite <= lower_)) if np.isfinite(lower_) else 0
    upperHits = int(np.count_nonzero(finite >= upper_)) if np.isfinite(upper_) else 0
    total = float(finite.size)
    return metadataFloat(lowerHits / total), metadataFloat(upperHits / total)


def _signChangePerKB(
    values: np.ndarray | None,
    *,
    intervalSizeBP: int | None,
) -> Any:
    if values is None or intervalSizeBP is None:
        return None
    intervalSizeBP_ = int(intervalSizeBP)
    if intervalSizeBP_ <= 0:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    meanAbs = float(np.mean(np.abs(finite), dtype=np.float64))
    if not np.isfinite(meanAbs):
        return None
    minSignMagnitude = 0.01 * meanAbs
    if minSignMagnitude > 0.0:
        finite = finite[np.abs(finite) >= minSignMagnitude]
    signs = np.sign(finite)
    signs = signs[signs != 0.0]
    signChangeCount = (
        int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0)) if signs.size >= 2 else 0
    )
    spanKB = float(arr.size) * float(intervalSizeBP_) / 1000.0
    if not np.isfinite(spanKB) or spanKB <= 0.0:
        return None
    return metadataFloat(float(signChangeCount) / spanKB)


def _relativeSignChangePerKB(
    stateValues: np.ndarray | None,
    matrixData: np.ndarray | None,
    matrixMunc: np.ndarray | None,
    *,
    intervalSizeBP: int | None,
    background: np.ndarray | None = None,
    pad: float = 0.0,
) -> Any:
    r"""Computes a proxy for sign-change density as the average per-KB residual between estimated states and weighted means of observations"""
    if stateValues is None or matrixData is None or matrixMunc is None:
        return None
    stateArr = np.asarray(stateValues, dtype=np.float64).reshape(-1)
    dataArr = np.asarray(matrixData)
    muncArr = np.asarray(matrixMunc)
    if dataArr.ndim != 2 or muncArr.shape != dataArr.shape:
        return None
    if dataArr.shape[1] != stateArr.size:
        return None

    if background is not None:
        backgroundArr = np.asarray(background, dtype=np.float64).reshape(-1)
        if backgroundArr.size != stateArr.size:
            return None
    else:
        backgroundArr = np.zeros(stateArr.size, dtype=np.float64)
    weightedTotal = np.zeros(stateArr.shape, dtype=np.float64)
    weightSum = np.zeros(stateArr.shape, dtype=np.float64)
    stateFinite = np.isfinite(stateArr)
    pad_ = float(pad)
    for j in range(dataArr.shape[0]):
        dataRow = np.asarray(dataArr[j, :], dtype=np.float64)
        denomRow = np.asarray(muncArr[j, :], dtype=np.float64) + pad_
        valid = (
            stateFinite
            & np.isfinite(dataRow)
            & np.isfinite(denomRow)
            & (denomRow > 0.0)
        )
        if not np.any(valid):
            continue
        rowWeights = 1.0 / np.maximum(denomRow[valid], 1.0e-12)
        adjustedRow = dataRow[valid] - backgroundArr[valid]
        weightedTotal[valid] += adjustedRow * rowWeights
        weightSum[valid] += rowWeights

    weightedMean = np.full(stateArr.shape, np.nan, dtype=np.float64)
    hasWeight = weightSum > 0.0
    if np.any(hasWeight):
        weightedMean[hasWeight] = weightedTotal[hasWeight] / weightSum[hasWeight]
    return _signChangePerKB(
        stateArr - weightedMean,
        intervalSizeBP=intervalSizeBP,
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


def _coerceOptionalProcessNoiseMatrix(value: np.ndarray | None) -> np.ndarray | None:
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


def _coerceMatrixDataMuncPair(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrixDataArr = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMuncArr = np.ascontiguousarray(matrixMunc, dtype=np.float32)

    if matrixDataArr.ndim == 1:
        matrixDataArr = matrixDataArr[None, :]
    elif matrixDataArr.ndim != 2:
        raise ValueError(f"matrixData must be 1D or 2D (got ndim={matrixDataArr.ndim})")

    if matrixMuncArr.ndim == 1:
        matrixMuncArr = matrixMuncArr[None, :]
    elif matrixMuncArr.ndim != 2:
        raise ValueError(f"matrixMunc must be 1D or 2D (got ndim={matrixMuncArr.ndim})")

    if matrixDataArr.shape != matrixMuncArr.shape:
        raise ValueError("matrixData and matrixMunc must have identical shapes")
    return matrixDataArr, matrixMuncArr


def _applyObservationMaskToMunc(
    matrixMunc: np.ndarray,
    observationMask: np.ndarray | None,
) -> np.ndarray:
    matrixMuncArr = np.ascontiguousarray(matrixMunc, dtype=np.float32)
    if observationMask is None:
        return matrixMuncArr

    observationMaskArr = np.asarray(observationMask, dtype=bool)
    if observationMaskArr.ndim == 1:
        observationMaskArr = np.broadcast_to(
            observationMaskArr[None, :],
            matrixMuncArr.shape,
        )
    if observationMaskArr.shape != matrixMuncArr.shape:
        raise ValueError("observationMask must match matrixData shape")

    matrixMuncArr = matrixMuncArr.copy(order="C")
    matrixMuncArr[~observationMaskArr] = (
        UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE
    )
    return np.ascontiguousarray(matrixMuncArr, dtype=np.float32)


def _backgroundWarmStartSummary(background: np.ndarray) -> dict[str, float]:
    backgroundArr = np.asarray(background, dtype=np.float64).reshape(-1)
    if backgroundArr.size == 0:
        return {
            "min": 0.0,
            "p05": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "frac_positive": 0.0,
            "frac_abs_le_1e_3": 0.0,
        }
    quantiles = np.quantile(backgroundArr, [0.05, 0.5, 0.95])
    return {
        "min": float(np.min(backgroundArr)),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(np.mean(backgroundArr, dtype=np.float64)),
        "p95": float(quantiles[2]),
        "max": float(np.max(backgroundArr)),
        "frac_positive": float(np.mean(backgroundArr > 0.0)),
        "frac_abs_le_1e_3": float(np.mean(np.abs(backgroundArr) <= 1.0e-3)),
    }


def _estimateBackgroundWarmStart(
    *,
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    blockLenIntervals: int,
    pad: float,
    observationPrecision: np.ndarray | None,
    observationPrecisionMultiplierMin: float,
    observationPrecisionMultiplierMax: float,
    backgroundSmoothness: float,
    zeroCenterBackground: bool,
    useNonnegativeBackground: bool,
    backgroundNegativePenaltyMultiplier: float | None,
    phaseLabel: str | None = None,
    logSummary: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrixDataArr, matrixMuncArr = _coerceMatrixDataMuncPair(matrixData, matrixMunc)
    trackCount, intervalCount = matrixDataArr.shape
    if intervalCount < 1:
        background = np.zeros(0, dtype=np.float32)
        diagnostics = {
            "applied": True,
            "source": "empty",
            **_backgroundWarmStartSummary(background),
        }
        return background, diagnostics

    blockLenIntervals = max(1, int(blockLenIntervals))
    observationPrecisionArr = _coerceOptionalVector(
        "observationPrecision",
        observationPrecision,
        intervalCount,
    )
    warmInvVarMatrix = 1.0 / np.maximum(
        matrixMuncArr + float(pad),
        np.float32(1.0e-8),
    )
    if observationPrecisionArr is not None:
        warmObsPrecision = np.clip(
            np.asarray(observationPrecisionArr, dtype=np.float32).reshape(
                1,
                intervalCount,
            ),
            float(observationPrecisionMultiplierMin),
            float(observationPrecisionMultiplierMax),
        )
        warmInvVarMatrix *= warmObsPrecision

    warmResidualMatrix = np.asarray(matrixDataArr, dtype=np.float32)
    if useNonnegativeBackground:
        background = solveZeroCenteredBackground(
            residualMatrix=warmResidualMatrix,
            invVarMatrix=warmInvVarMatrix,
            blockLenIntervals=int(blockLenIntervals),
            backgroundSmoothness=float(backgroundSmoothness),
            zeroCenter=bool(zeroCenterBackground),
            useNonnegative=True,
            backgroundNegativePenaltyMultiplier=backgroundNegativePenaltyMultiplier,
        )
        source = (
            "asymmetric_irls_zero_centered_weighted_data"
            if bool(zeroCenterBackground)
            else "asymmetric_irls_weighted_data"
        )
    else:
        background = solveZeroCenteredBackground(
            residualMatrix=warmResidualMatrix,
            invVarMatrix=warmInvVarMatrix,
            blockLenIntervals=int(blockLenIntervals),
            backgroundSmoothness=float(backgroundSmoothness),
            zeroCenter=bool(zeroCenterBackground),
            useNonnegative=False,
        )
        source = (
            "zero_centered_banded_weighted_data"
            if bool(zeroCenterBackground)
            else "banded_weighted_data"
        )

    background = np.ascontiguousarray(background, dtype=np.float32)
    diagnostics = {
        "applied": True,
        "source": source,
        **_backgroundWarmStartSummary(background),
    }
    if logSummary:
        logger.debug(
            "backgroundWarmStart[%s]: source=%s min=%.6g "
            "p05=%.6g median=%.6g mean=%.6g p95=%.6g max=%.6g "
            "fracPositive=%.4f fracAbsLe1e-3=%.4f",
            phaseLabel or "provisional",
            diagnostics["source"],
            float(diagnostics["min"]),
            float(diagnostics["p05"]),
            float(diagnostics["median"]),
            float(diagnostics["mean"]),
            float(diagnostics["p95"]),
            float(diagnostics["max"]),
            float(diagnostics["frac_positive"]),
            float(diagnostics["frac_abs_le_1e_3"]),
        )
    return background, diagnostics


def estimateProvisionalBackground(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    *,
    blockLenIntervals: int,
    pad: float = 1.0e-4,
    observationPrecision: np.ndarray | None = None,
    observationMask: np.ndarray | None = None,
    observationPrecisionMultiplierMin: float = (
        OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MIN
    ),
    observationPrecisionMultiplierMax: float = (
        OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MAX
    ),
    backgroundSmoothness: float = FIT_DEFAULT_BACKGROUND_SMOOTHNESS,
    zeroCenterBackground: bool = FIT_DEFAULT_ZERO_CENTER_BACKGROUND,
    useNonnegativeBackground: bool = FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND,
    backgroundNegativePenaltyMultiplier: float | None = (
        FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER
    ),
    returnDiagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    r"""Estimate a provisional shared background from weighted observations.

    This exposes the same weighted background warm-start solve used internally by
    :func:`runConsenrich`. It is intended for nuisance-background prepasses such
    as MUNC residualization; the returned track is not a full latent-state fit.
    """

    matrixDataArr, matrixMuncArr = _coerceMatrixDataMuncPair(matrixData, matrixMunc)
    matrixMuncArr = _applyObservationMaskToMunc(matrixMuncArr, observationMask)
    (
        observationPrecisionMultiplierMin,
        observationPrecisionMultiplierMax,
    ) = _checkPrecisionMultiplierBounds(
        "observation",
        observationPrecisionMultiplierMin,
        observationPrecisionMultiplierMax,
    )
    if backgroundNegativePenaltyMultiplier is None:
        backgroundNegativePenaltyMultiplierLocal = None
    else:
        backgroundNegativePenaltyMultiplierLocal = float(
            backgroundNegativePenaltyMultiplier
        )
        if not np.isfinite(backgroundNegativePenaltyMultiplierLocal):
            raise ValueError(
                "`backgroundNegativePenaltyMultiplier` must be finite or None"
            )

    background, diagnostics = _estimateBackgroundWarmStart(
        matrixData=matrixDataArr,
        matrixMunc=matrixMuncArr,
        blockLenIntervals=int(blockLenIntervals),
        pad=float(pad),
        observationPrecision=observationPrecision,
        observationPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
        observationPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
        backgroundSmoothness=float(backgroundSmoothness),
        zeroCenterBackground=bool(zeroCenterBackground),
        useNonnegativeBackground=bool(useNonnegativeBackground),
        backgroundNegativePenaltyMultiplier=backgroundNegativePenaltyMultiplierLocal,
    )
    if returnDiagnostics:
        return background, diagnostics
    return background


def _effectiveObservationCount(matrixMunc: np.ndarray) -> int:
    munc = np.asarray(matrixMunc, dtype=np.float64)
    active = np.isfinite(munc) & (
        munc < 0.5 * float(UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE)
    )
    return int(max(1, np.count_nonzero(active)))


def _activeProcessNoiseObservationMask(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(matrixData, dtype=np.float64)
    munc = np.asarray(matrixMunc, dtype=np.float64)
    if data.shape != munc.shape:
        raise ValueError("matrixData and matrixMunc must have matching shapes")
    obsVariance = munc + float(pad)
    unmasked = np.isfinite(munc) & (
        munc < 0.5 * float(UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE)
    )
    positiveObsVariance = np.isfinite(obsVariance) & (obsVariance > 0.0)
    active = np.isfinite(data) & unmasked & positiveObsVariance
    return active, obsVariance


def _processNoiseCalibrationSupport(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    pad: float,
) -> dict[str, Any]:
    data = np.asarray(matrixData, dtype=np.float64)
    munc = np.asarray(matrixMunc, dtype=np.float64)
    active, obsVariance = _activeProcessNoiseObservationMask(data, munc, pad)
    unmaskedPositiveObsVar = (
        np.isfinite(munc)
        & (munc < 0.5 * float(UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE))
        & np.isfinite(obsVariance)
        & (obsVariance > 0.0)
    )
    activeIntervals = np.any(active, axis=0) if active.ndim == 2 else np.zeros(0, bool)
    activeAdjacent = (
        activeIntervals[1:] & activeIntervals[:-1]
        if activeIntervals.size >= 2
        else np.zeros(0, dtype=bool)
    )
    sameTrackAdjacent = (
        np.any(active[:, 1:] & active[:, :-1], axis=0)
        if active.ndim == 2 and active.shape[1] >= 2
        else np.zeros(0, dtype=bool)
    )
    finiteDataCount = int(np.count_nonzero(np.isfinite(data)))
    positiveObservationVarianceCount = int(np.count_nonzero(unmaskedPositiveObsVar))
    activeObservationCount = int(np.count_nonzero(active))
    activeAdjacentTransitionCount = int(np.count_nonzero(activeAdjacent))
    skipReason: str | None = None
    if finiteDataCount <= 0:
        skipReason = "no_finite_data"
    elif positiveObservationVarianceCount <= 0:
        skipReason = "no_positive_observation_variance"
    elif activeObservationCount <= 0:
        skipReason = "no_active_observations"
    elif activeAdjacentTransitionCount <= 0:
        skipReason = "no_active_adjacent_transitions"
    return {
        "finiteDataCount": finiteDataCount,
        "positiveObservationVarianceCount": positiveObservationVarianceCount,
        "activeObservationCount": activeObservationCount,
        "activeIntervalCount": int(np.count_nonzero(activeIntervals)),
        "intervalTransitionCount": int(max(data.shape[1] - 1, 0)),
        "activeAdjacentTransitionCount": activeAdjacentTransitionCount,
        "sameTrackAdjacentTransitionCount": int(np.count_nonzero(sameTrackAdjacent)),
        "processNoiseCalibrationCanRun": bool(skipReason is None),
        "processNoiseCalibrationSkipReason": skipReason,
    }


def _hasFiniteTransitionVariation(matrixData: np.ndarray) -> bool:
    data = np.asarray(matrixData, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] <= 1:
        return False
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    diffs = np.diff(data, axis=1)
    finiteDiffs = diffs[np.isfinite(diffs)]
    if finiteDiffs.size == 0:
        return False
    tolerance = 1.0e-8 * max(1.0, float(np.max(np.abs(finite))))
    return bool(np.max(np.abs(finiteDiffs)) > tolerance)


def _processNoiseQBoundaryDiagnostics(
    matrixQ0: np.ndarray,
    stateModel: str,
    minQ: float,
    maxQ: float,
) -> dict[str, Any]:
    stateModelMode = _normalizeStateModel(stateModel)
    dim = 1 if stateModelMode == STATE_MODEL_LEVEL else 2
    qFloor = _resolveProcessNoiseFloor(minQ)
    qCap = _resolveProcessNoiseCap(maxQ, minQ=qFloor)
    q0 = np.asarray(matrixQ0, dtype=np.float64)
    levelQ = _clampProcessNoise(float(q0[0, 0]), qFloor=qFloor, qCap=qCap)
    trendQ = (
        0.0
        if dim == 1
        else _clampProcessNoise(float(q0[1, 1]), qFloor=qFloor, qCap=qCap)
    )
    hitLevelFloor = bool(levelQ <= 1.0001 * qFloor)
    hitTrendFloor = bool(dim == 2 and trendQ <= 1.0001 * qFloor)
    hitLevelCap = bool(np.isfinite(qCap) and levelQ >= 0.9999 * qCap)
    hitTrendCap = bool(dim == 2 and np.isfinite(qCap) and trendQ >= 0.9999 * qCap)
    hitFloor = bool(hitLevelFloor or hitTrendFloor)
    hitCap = bool(hitLevelCap or hitTrendCap)
    if hitFloor and hitCap:
        boundaryStatus = "floor_and_cap"
    elif hitFloor:
        boundaryStatus = "floor"
    elif hitCap:
        boundaryStatus = "cap"
    else:
        boundaryStatus = "interior"
    return {
        "preKappaQLevel": float(levelQ),
        "preKappaQTrend": float(trendQ),
        "qFloor": float(qFloor),
        "qCap": float(qCap),
        "hitQLevelFloor": hitLevelFloor,
        "hitQTrendFloor": hitTrendFloor,
        "hitQLevelCap": hitLevelCap,
        "hitQTrendCap": hitTrendCap,
        "hitQFloor": hitFloor,
        "hitQCap": hitCap,
        "qBoundaryStatus": boundaryStatus,
    }


def _staticProcessNoiseCalibrationDiagnostics(
    *,
    processNoisePolicy: str,
    status: str,
    reason: str,
    matrixQ0: np.ndarray,
    stateModel: str,
    minQ: float,
    maxQ: float,
    support: Mapping[str, Any],
    warmStartProcessNoise: float,
) -> dict[str, Any]:
    stateModelMode = _normalizeStateModel(stateModel)
    dim = 1 if stateModelMode == STATE_MODEL_LEVEL else 2
    boundary = _processNoiseQBoundaryDiagnostics(
        matrixQ0,
        stateModelMode,
        minQ=minQ,
        maxQ=maxQ,
    )
    matrixQ0Final = _clampProcessNoiseMatrix(
        matrixQ0,
        stateModel=stateModelMode,
        minQ=minQ,
        maxQ=maxQ,
    )
    levelQ = float(boundary["preKappaQLevel"])
    trendQ = float(boundary["preKappaQTrend"])
    ratio = 0.0 if dim == 1 else trendQ / max(levelQ, float(boundary["qFloor"]))
    diagnostics = {
        "processNoisePolicy": processNoisePolicy,
        "processNoiseCalibrationStatus": status,
        "processNoiseCalibrationReason": reason,
        "stateModel": stateModelMode,
        "preKappaQLevel": levelQ,
        "preKappaQTrend": trendQ,
        "rawTrendLevelRatio": float(ratio),
        "effectiveTrendLevelRatio": float(ratio),
        "logQLevel": float(np.log(max(levelQ, float(boundary["qFloor"])))),
        "logQTrend": (
            0.0 if dim == 1 else float(np.log(max(trendQ, float(boundary["qFloor"]))))
        ),
        "usedInitialProcessQFallback": bool(
            status != "estimated" and float(warmStartProcessNoise) <= 0.0
        ),
        "matrixQ0Final": matrixQ0Final.astype(float).tolist(),
        "warmStartProcessNoise": float(warmStartProcessNoise),
        "globalScale": 1.0,
        "windowCount": 0,
        "validTransitionCount": 0,
        "qScaleClampFraction": 0.0,
    }
    diagnostics.update({key: False for key in _PUNC_STAGE_TOGGLE_KEYS})
    diagnostics["puncStagesActive"] = False
    diagnostics.update(boundary)
    diagnostics.update(dict(support))
    return diagnostics


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


class _FixedBackgroundECMResult(NamedTuple):
    iters_done: int
    nll: float
    state_smoothed: np.ndarray
    state_covar_smoothed: np.ndarray
    lag_covar_smoothed: np.ndarray
    post_fit_residuals: np.ndarray
    lambda_exp: np.ndarray | None
    process_prec_exp: np.ndarray | None
    diagnostics: Mapping[str, Any]


def _runFixedBackgroundECMPhase(
    *,
    matrixDataLocal: np.ndarray,
    currentBackground: np.ndarray,
    currentMunc: np.ndarray,
    matrixQ0Local: np.ndarray,
    intervalToBlockMap: np.ndarray,
    blockCount: int,
    stateInit: float,
    stateCovarInit: float,
    ecmItersLocal: int,
    ecmRtolLocal: float,
    t_innerItersLocal: int,
    pad: float,
    ECM_robustTNu: float,
    ECM_useObsPrecisionReweighting: bool,
    useProcPrecLocal: bool,
    useAPNLocal: bool,
    observationPrecisionMultiplierMin: float,
    observationPrecisionMultiplierMax: float,
    processPrecisionMultiplierMin: float,
    processPrecisionMultiplierMax: float,
    minQ: float,
    maxQForAPN: float,
    lambdaExpLocal: np.ndarray | None,
    processPrecExpLocal: np.ndarray | None,
    processQScaleLocal: np.ndarray | None,
    trackOptimizationPath: bool,
    logIterations: bool,
    stateModelMode: str,
    matrixFLocal: np.ndarray | None,
) -> _FixedBackgroundECMResult:
    """Run the Cython fixed-background ECM kernel and normalize its return shape."""

    dataAdjusted = np.ascontiguousarray(
        matrixDataLocal - currentBackground[None, :],
        dtype=np.float32,
    )
    ecmKwargs = dict(
        matrixData=dataAdjusted,
        matrixPluginMuncInit=currentMunc,
        matrixQ0=matrixQ0Local,
        intervalToBlockMap=intervalToBlockMap,
        blockCount=int(blockCount),
        stateInit=float(stateInit),
        stateCovarInit=float(stateCovarInit),
        ECM_fixedBackgroundIters=int(ecmItersLocal),
        ECM_fixedBackgroundRtol=float(ecmRtolLocal),
        t_innerIters=int(t_innerItersLocal),
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
        APN_maxQ=float(maxQForAPN),
        lambdaExpInit=lambdaExpLocal,
        processPrecExpInit=processPrecExpLocal,
        processQScale=processQScaleLocal,
        trackOptimizationPath=bool(trackOptimizationPath),
        logIterations=bool(logIterations),
    )
    ecmFunction = cconsenrich.cfixedBackgroundECMLevel
    if stateModelMode == STATE_MODEL_LEVEL_TREND:
        ecmFunction = cconsenrich.cfixedBackgroundECM
        ecmKwargs["matrixF"] = matrixFLocal
    ecmOutLocal = ecmFunction(**ecmKwargs, returnDiagnostics=True)

    if len(ecmOutLocal) == 9 and isinstance(ecmOutLocal[-1], Mapping):
        ecmDiagnosticsLocal = ecmOutLocal[-1]
        ecmOutLocal = ecmOutLocal[:-1]
    else:
        raise ValueError(
            "Expected cfixedBackgroundECM(..., returnDiagnostics=True) "
            "to return diagnostics as the final value."
        )
    if len(ecmOutLocal) != 8:
        raise ValueError(
            "Expected cfixedBackgroundECM(..., returnIntermediates=True) to return 8 values "
            f"(got {len(ecmOutLocal)})."
        )
    (
        ecmItersDoneLocal,
        nllECMLocal,
        stateSmoothedLocal,
        stateCovarSmoothedLocal,
        lagCovSmoothedLocal,
        postFitResidualsLocal,
        lambdaExpLocalOut,
        processPrecExpLocalOut,
    ) = ecmOutLocal
    return _FixedBackgroundECMResult(
        iters_done=int(ecmItersDoneLocal),
        nll=float(nllECMLocal),
        state_smoothed=np.asarray(stateSmoothedLocal, dtype=np.float32),
        state_covar_smoothed=np.asarray(stateCovarSmoothedLocal, dtype=np.float32),
        lag_covar_smoothed=np.asarray(lagCovSmoothedLocal, dtype=np.float32),
        post_fit_residuals=np.asarray(postFitResidualsLocal, dtype=np.float32),
        lambda_exp=(
            None
            if lambdaExpLocalOut is None
            else np.asarray(lambdaExpLocalOut, dtype=np.float32)
        ),
        process_prec_exp=(
            None
            if processPrecExpLocalOut is None
            else np.asarray(processPrecExpLocalOut, dtype=np.float32)
        ),
        diagnostics=ecmDiagnosticsLocal,
    )


def _normalizeFixedBackgroundECMDiagnostics(
    diagnostics: Mapping[str, Any],
    *,
    itersDone: int,
    finalNLL: float,
    maxIters: int,
    outerPass: int,
) -> dict[str, Any]:
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
        "background_objective": fit.get("backgroundObjective"),
        "background_objective_per_cell": fit.get("backgroundObjectivePerCell"),
        "background_objective_change_per_cell": fit.get(
            "backgroundObjectiveChangePerCell"
        ),
        "background_objective_threshold_per_cell": fit.get(
            "backgroundObjectiveThresholdPerCell"
        ),
        "background_objective_stable": bool(
            fit.get("backgroundObjectiveStable", False)
        ),
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
        "observation_lambda_lower_bound_hits": fit.get("lambdaLowerBoundHits"),
        "observation_lambda_upper_bound_hits": fit.get("lambdaUpperBoundHits"),
        "process_kappa_lower_bound_hits": fit.get("kappaLowerBoundHits"),
        "process_kappa_upper_bound_hits": fit.get("kappaUpperBoundHits"),
        "relative_sign_change_per_kb": fit.get("relativeSignChangePerKB"),
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


def _computeExpectedLevelTransitionResidualSums(
    stateSmoothed: np.ndarray,
    stateCovarSmoothed: np.ndarray,
    lagCovSmoothed: np.ndarray,
) -> tuple[float, float, int]:
    r"""Compute expected scalar level-transition residual sums."""

    m = np.asarray(stateSmoothed, dtype=np.float64)
    P = np.asarray(stateCovarSmoothed, dtype=np.float64)
    C = np.asarray(lagCovSmoothed, dtype=np.float64)

    if m.ndim != 2 or m.shape[1] != 1:
        raise ValueError("stateSmoothed must have shape (n, 1)")
    if P.shape != (m.shape[0], 1, 1):
        raise ValueError("stateCovarSmoothed must have shape (n, 1, 1)")
    if C.shape[0] < max(m.shape[0] - 1, 0) or C.shape[1:] != (1, 1):
        raise ValueError("lagCovSmoothed must have shape (n - 1, 1, 1)")

    transitionCount = int(m.shape[0] - 1)
    if transitionCount <= 0:
        return 0.0, 0.0, 0

    sumLevel, sumTrend, cTransitionCount = (
        cconsenrich.cExpectedTransitionResidualSumsLevel(
            np.ascontiguousarray(m, dtype=np.float64),
            np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(C, dtype=np.float64),
        )
    )
    return float(sumLevel), float(sumTrend), int(cTransitionCount)


def _normalizeProcessNoiseCalibrationMode(value: str | None) -> str:
    return _sharedNormalizeProcessNoiseCalibration(value)


def _coerceOptionalProcessCovariates(
    processCovariates: np.ndarray | None,
    *,
    intervalCount: int,
) -> np.ndarray | None:
    if processCovariates is None:
        return None
    arr = np.asarray(processCovariates, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("processCovariates must be a 2D array")
    transitionCount = max(int(intervalCount) - 1, 0)
    if arr.shape[0] == transitionCount:
        out = arr
    elif arr.shape[0] == int(intervalCount):
        out = 0.5 * (arr[:-1, :] + arr[1:, :])
    else:
        raise ValueError(
            "processCovariates must be transition-aligned with intervalCount - 1 "
            "rows or interval-aligned with intervalCount rows"
        )
    if out.shape[0] != transitionCount:
        raise ValueError("processCovariates do not align with transitions")
    if out.size and not np.all(np.isfinite(out)):
        raise ValueError("processCovariates must contain only finite values")
    return np.ascontiguousarray(out, dtype=np.float64)


def _puncObservationInformation(
    *,
    matrixMunc: np.ndarray,
    pad: float,
    lambdaExp: np.ndarray | None,
    observationPrecisionMultiplierMin: float,
    observationPrecisionMultiplierMax: float,
) -> np.ndarray:
    return np.asarray(
        cconsenrich.cPuncObservationInformation(
            matrixMunc,
            float(pad),
            lambdaExp,
            float(observationPrecisionMultiplierMin),
            float(observationPrecisionMultiplierMax),
        ),
        dtype=np.float64,
    )


def _weightedGeometricMean(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    floor: float,
) -> float:
    v = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), float(floor))
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(mask):
        return 1.0
    return float(np.exp(np.sum(w[mask] * np.log(v[mask])) / np.sum(w[mask])))


def _weightedSampleVariance(values: np.ndarray, weights: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    x = x[mask]
    w = w[mask]
    sumW = float(np.sum(w))
    if sumW <= 0.0:
        return float("nan")
    mean = float(np.sum(w * x) / sumW)
    sumW2 = float(np.sum(w * w))
    denom = sumW - (sumW2 / sumW)
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(w * np.square(x - mean)) / denom)


def _estimatePuncPriorDfMethodOfMoments(
    localEvidence: np.ndarray,
    localPrior: np.ndarray,
    nuLocal: np.ndarray,
    weights: np.ndarray,
    *,
    minPriorDf: float = 4.0,
    maxPriorDf: float = 1.0e6,
    minWindows: int = PROCESS_DEFAULT_PUNC_PRIOR_DF_MOMENTS_MIN_WINDOWS,
    winsorTail: float = PROCESS_DEFAULT_PUNC_PRIOR_DF_MOMENTS_WINSOR_TAIL,
    minScale: float = PROCESS_DEFAULT_PUNC_MIN_SCALE,
    maxScale: float = PROCESS_DEFAULT_PUNC_MAX_SCALE,
) -> tuple[float, float, dict[str, Any]]:
    evidence = np.asarray(localEvidence, dtype=np.float64).reshape(-1)
    prior = np.asarray(localPrior, dtype=np.float64).reshape(-1)
    nu = np.asarray(nuLocal, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (evidence.shape == prior.shape == nu.shape == w.shape):
        raise ValueError(
            "localEvidence, localPrior, nuLocal, and weights must have the same shape"
        )

    valid = (
        np.isfinite(evidence)
        & np.isfinite(prior)
        & np.isfinite(nu)
        & np.isfinite(w)
        & (evidence > 0.0)
        & (prior > 0.0)
        & (nu > 0.0)
        & (w > 0.0)
    )
    diagnostics: dict[str, Any] = {
        "puncPriorDfMomentWindowCount": int(np.count_nonzero(valid)),
        "puncPriorDfMomentEffectiveWindowCount": 0.0,
        "puncPriorDfMomentLogRatioVariance": float("nan"),
        "puncPriorDfMomentSamplingVariance": float("nan"),
        "puncPriorDfMomentExcessVariance": float("nan"),
        "puncPriorDfMomentScale": 1.0,
        "puncPriorDfMomentWinsorLower": float("nan"),
        "puncPriorDfMomentWinsorUpper": float("nan"),
        "puncPriorDfMomentReason": "ok",
    }
    if np.count_nonzero(valid) < int(minWindows):
        diagnostics["puncPriorDfMomentReason"] = "insufficient_windows"
        return float(maxPriorDf), 1.0, diagnostics

    evidence = evidence[valid]
    prior = prior[valid]
    nu = nu[valid]
    w = w[valid]
    sumW = float(np.sum(w))
    sumW2 = float(np.sum(w * w))
    effectiveWindowCount = (sumW * sumW / sumW2) if sumW > 0.0 and sumW2 > 0.0 else 0.0
    diagnostics["puncPriorDfMomentEffectiveWindowCount"] = float(effectiveWindowCount)
    if effectiveWindowCount < float(minWindows):
        diagnostics["puncPriorDfMomentReason"] = "insufficient_effective_windows"
        return float(maxPriorDf), 1.0, diagnostics

    logRatio = np.log(np.maximum(evidence, np.finfo(np.float64).tiny)) - np.log(
        np.maximum(prior, np.finfo(np.float64).tiny)
    )
    localBias = special.digamma(nu / 2.0) - np.log(nu / 2.0)
    centeredLogRatio = logRatio - localBias

    tail = float(max(0.0, min(0.25, winsorTail)))
    if centeredLogRatio.size >= 20 and tail > 0.0:
        lo, hi = _weightedQuantile(
            centeredLogRatio,
            w,
            np.asarray([tail, 1.0 - tail], dtype=np.float64),
        )
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            np.clip(centeredLogRatio, float(lo), float(hi), out=centeredLogRatio)
            diagnostics["puncPriorDfMomentWinsorLower"] = float(lo)
            diagnostics["puncPriorDfMomentWinsorUpper"] = float(hi)

    observedVariance = _weightedSampleVariance(centeredLogRatio, w)
    samplingVarianceArr = np.asarray(trigamma(nu / 2.0), dtype=np.float64)
    samplingMask = np.isfinite(samplingVarianceArr) & (samplingVarianceArr >= 0.0)
    if not np.any(samplingMask):
        diagnostics["puncPriorDfMomentReason"] = "nonfinite_sampling_variance"
        return float(maxPriorDf), 1.0, diagnostics
    samplingVariance = float(
        np.sum(w[samplingMask] * samplingVarianceArr[samplingMask])
        / np.sum(w[samplingMask])
    )
    excessVariance = float(observedVariance - samplingVariance)
    diagnostics["puncPriorDfMomentLogRatioVariance"] = float(observedVariance)
    diagnostics["puncPriorDfMomentSamplingVariance"] = float(samplingVariance)
    diagnostics["puncPriorDfMomentExcessVariance"] = float(excessVariance)

    if not np.isfinite(observedVariance) or not np.isfinite(excessVariance):
        diagnostics["puncPriorDfMomentReason"] = "nonfinite_log_ratio_variance"
        return float(maxPriorDf), 1.0, diagnostics
    if excessVariance <= 1.0e-12:
        priorDf = float(maxPriorDf)
        priorBias = 0.0
        diagnostics["puncPriorDfMomentReason"] = "no_excess_dispersion"
    else:
        priorDf = float(2.0 * itrigamma(max(excessVariance, 1.0e-12)))
        if not np.isfinite(priorDf) or priorDf <= 0.0:
            priorDf = float(maxPriorDf)
            priorBias = 0.0
            diagnostics["puncPriorDfMomentReason"] = "invalid_inverse_trigamma"
        else:
            priorDf = float(np.clip(priorDf, float(minPriorDf), float(maxPriorDf)))
            priorBias = (
                0.0
                if priorDf >= 0.999 * float(maxPriorDf)
                else float(special.digamma(priorDf / 2.0) - math.log(priorDf / 2.0))
            )

    logScale = float(np.sum(w * (logRatio - localBias + priorBias)) / sumW)
    scale = float(
        np.exp(np.clip(logScale, math.log(float(minScale)), math.log(float(maxScale))))
    )
    diagnostics["puncPriorDfMomentScale"] = float(scale)
    return float(priorDf), float(scale), diagnostics


def _activeProcessQDiagonal(
    matrixQ: np.ndarray,
    *,
    stateModel: str,
) -> np.ndarray:
    stateModelMode = _normalizeStateModel(stateModel)
    dim = 1 if stateModelMode == STATE_MODEL_LEVEL else 2
    q = np.asarray(matrixQ, dtype=np.float64)
    return np.diag(q[:dim, :dim]).astype(np.float64, copy=True)


def _fitPuncProcessNoise(
    *,
    warmupFit: Mapping[str, Any],
    matrixMunc: np.ndarray,
    matrixF: np.ndarray,
    seedQ: np.ndarray,
    stateModel: str,
    pad: float,
    minQ: float,
    maxQ: float,
    blockLenIntervals: int,
    processCovariates: np.ndarray | None,
    puncLocalWindowMultiplier: float,
    puncDependenceMultiplier: float,
    puncMinScale: float,
    puncMaxScale: float,
    puncMinWindowWeight: float,
    puncPriorDf: float = PROCESS_DEFAULT_PUNC_PRIOR_DF,
    puncPriorRidge: float = PROCESS_DEFAULT_PUNC_PRIOR_RIDGE,
    puncLevelBufferZ: float = PROCESS_DEFAULT_PUNC_LEVEL_BUFFER_Z,
    puncUseReliabilityWeightedWindows: bool = (
        PROCESS_DEFAULT_PUNC_USE_RELIABILITY_WEIGHTED_WINDOWS
    ),
    puncUseWarmupFit: bool = PROCESS_DEFAULT_PUNC_USE_WARMUP_FIT,
    puncUseTransitionEvidence: bool = PROCESS_DEFAULT_PUNC_USE_TRANSITION_EVIDENCE,
    puncUseScaleRebase: bool = PROCESS_DEFAULT_PUNC_USE_SCALE_REBASE,
    puncUseGlobalScale: bool = PROCESS_DEFAULT_PUNC_USE_GLOBAL_SCALE,
    puncUseBoundaryClamps: bool = PROCESS_DEFAULT_PUNC_USE_BOUNDARY_CLAMPS,
    puncUsePriorDfMoments: bool = PROCESS_DEFAULT_PUNC_USE_PRIOR_DF_MOMENTS,
    puncUsePriorShrinkage: bool = PROCESS_DEFAULT_PUNC_USE_PRIOR_SHRINKAGE,
    observationPrecisionMultiplierMin: float = 1.0,
    observationPrecisionMultiplierMax: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not isinstance(puncUseReliabilityWeightedWindows, (bool, np.bool_)):
        raise ValueError("puncUseReliabilityWeightedWindows must be boolean")
    puncUseReliabilityWeightedWindows = bool(puncUseReliabilityWeightedWindows)
    puncToggleValues = {
        "puncUseWarmupFit": puncUseWarmupFit,
        "puncUseTransitionEvidence": puncUseTransitionEvidence,
        "puncUseScaleRebase": puncUseScaleRebase,
        "puncUseGlobalScale": puncUseGlobalScale,
        "puncUseBoundaryClamps": puncUseBoundaryClamps,
        "puncUsePriorDfMoments": puncUsePriorDfMoments,
        "puncUsePriorShrinkage": puncUsePriorShrinkage,
    }
    for puncToggleName, puncToggleValue in puncToggleValues.items():
        if not isinstance(puncToggleValue, (bool, np.bool_)):
            raise ValueError(f"{puncToggleName} must be boolean")
    puncToggles = {key: bool(value) for key, value in puncToggleValues.items()}
    puncPriorDf = _checkFinitePositive("puncPriorDf", puncPriorDf)
    stateModelMode = _normalizeStateModel(stateModel)
    intervalCount = int(np.asarray(matrixMunc).shape[1])
    transitionCount = max(intervalCount - 1, 0)
    qFloor = _resolveProcessNoiseFloor(minQ)
    qCap = _resolveProcessNoiseCap(maxQ, minQ=qFloor)
    seedQClamped = _clampProcessNoiseMatrix(
        seedQ,
        stateModel=stateModelMode,
        minQ=float(minQ),
        maxQ=float(maxQ),
    )
    processQScale = np.ones(intervalCount, dtype=np.float32)
    if transitionCount <= 0:
        info = {
            "processNoisePolicy": PROCESS_NOISE_CALIBRATION_PUNC,
            "processNoiseCalibrationStatus": "skipped",
            "processNoiseCalibrationReason": "too_few_intervals",
            "globalScale": 1.0,
            "validTransitionCount": 0,
            "windowCount": 0,
            "qScaleClampFraction": 0.0,
            "processQScaleSummary": _metadataTrackSummary(processQScale),
            "priorDesignColumnCount": 0,
            "processCovariateCount": 0,
            "puncUseReliabilityWeightedWindows": bool(
                puncUseReliabilityWeightedWindows
            ),
            **puncToggles,
            "puncStagesActive": True,
            "baseQClampChanged": False,
            "baseQClampMaxRelativeChange": 0.0,
            "qScaleDecompositionMaxLogError": 0.0,
            "qScaleDecompositionMedianLogError": 0.0,
        }
        info.update(
            _processNoiseQBoundaryDiagnostics(seedQClamped, stateModelMode, minQ, maxQ)
        )
        info["matrixQ0Final"] = seedQClamped.astype(float).tolist()
        return seedQClamped, processQScale, info

    evidence, _evidenceDiagnostics = cconsenrich.cExpectedTransitionProcessEvidence(
        np.asarray(warmupFit["stateSmoothed"]),
        np.asarray(warmupFit["stateCovarSmoothed"]),
        np.asarray(warmupFit["lagCovSmoothed"]),
        seedQClamped,
        matrixF=(None if stateModelMode == STATE_MODEL_LEVEL else matrixF),
    )
    evidence = np.asarray(evidence, dtype=np.float64)
    infoByInterval = _puncObservationInformation(
        matrixMunc=np.asarray(warmupFit.get("matrixMunc", matrixMunc)),
        pad=float(pad),
        lambdaExp=warmupFit.get("lambdaExp"),
        observationPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
        observationPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
    )
    transitionWeights = np.sqrt(
        np.maximum(infoByInterval[:-1], 0.0) * np.maximum(infoByInterval[1:], 0.0)
    )
    transitionWeights = np.nan_to_num(
        transitionWeights,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    valid = (
        np.isfinite(evidence)
        & (evidence > 0.0)
        & np.isfinite(transitionWeights)
        & (transitionWeights > 0.0)
    )
    if not puncToggles["puncUseTransitionEvidence"]:
        evidence = np.ones_like(evidence, dtype=np.float64)
        valid = np.isfinite(transitionWeights) & (transitionWeights > 0.0)
    if not np.any(valid):
        info = {
            "processNoisePolicy": PROCESS_NOISE_CALIBRATION_PUNC,
            "processNoiseCalibrationStatus": "skipped",
            "processNoiseCalibrationReason": "no_valid_transition_evidence",
            "globalScale": 1.0,
            "validTransitionCount": 0,
            "windowCount": 0,
            "qScaleClampFraction": 0.0,
            "processQScaleSummary": _metadataTrackSummary(processQScale),
            "priorDesignColumnCount": 0,
            "processCovariateCount": 0,
            "puncUseReliabilityWeightedWindows": bool(
                puncUseReliabilityWeightedWindows
            ),
            **puncToggles,
            "puncStagesActive": True,
            "baseQClampChanged": False,
            "baseQClampMaxRelativeChange": 0.0,
            "qScaleDecompositionMaxLogError": 0.0,
            "qScaleDecompositionMedianLogError": 0.0,
        }
        info.update(
            _processNoiseQBoundaryDiagnostics(seedQClamped, stateModelMode, minQ, maxQ)
        )
        info["matrixQ0Final"] = seedQClamped.astype(float).tolist()
        return seedQClamped, processQScale, info

    tiny = 1.0e-12
    if puncToggles["puncUseBoundaryClamps"]:
        minScale = float(max(puncMinScale, tiny))
        maxScale = float(max(puncMaxScale, minScale))
    else:
        minScale = tiny
        maxScale = float(1.0 / tiny)
    y = np.log(np.maximum(evidence, tiny))
    state = np.asarray(warmupFit["stateSmoothed"])
    levelMidRaw = 0.5 * np.add(state[:-1, 0], state[1:, 0], dtype=np.float64)
    levelMid = levelMidRaw
    levelBufferZ = float(max(puncLevelBufferZ, 0.0))
    levelMidSd = np.zeros(transitionCount, dtype=np.float64)
    if levelBufferZ > 0.0:
        stateCov = np.asarray(warmupFit["stateCovarSmoothed"])
        lagCov = np.asarray(warmupFit["lagCovSmoothed"])
        if (
            stateCov.ndim == 3
            and lagCov.ndim == 3
            and stateCov.shape[0] >= intervalCount
            and lagCov.shape[0] >= transitionCount
            and stateCov.shape[1] >= 1
            and stateCov.shape[2] >= 1
            and lagCov.shape[1] >= 1
            and lagCov.shape[2] >= 1
        ):
            levelMidVar = np.add(
                stateCov[:transitionCount, 0, 0],
                stateCov[1 : transitionCount + 1, 0, 0],
                dtype=np.float64,
            )
            np.add(
                levelMidVar,
                lagCov[:transitionCount, 0, 0],
                out=levelMidVar,
                casting="unsafe",
            )
            np.add(
                levelMidVar,
                lagCov[:transitionCount, 0, 0],
                out=levelMidVar,
                casting="unsafe",
            )
            levelMidVar *= 0.25
            levelMidSd = np.sqrt(np.maximum(levelMidVar, 0.0))
        buffer = levelBufferZ * levelMidSd
        levelMid = np.sign(levelMidRaw) * np.maximum(np.abs(levelMidRaw) - buffer, 0.0)
    columns = [
        np.ones(transitionCount, dtype=np.float64),
        levelMid,
    ]
    priorDesignColumns = [
        "intercept",
        "stateLevelMidpointBuffered" if levelBufferZ > 0.0 else "stateLevelMidpoint",
    ]
    covariates = _coerceOptionalProcessCovariates(
        processCovariates,
        intervalCount=intervalCount,
    )
    if covariates is not None:
        for j in range(covariates.shape[1]):
            columns.append(covariates[:, j])
            priorDesignColumns.append(f"processCovariate{j}")
    designColumns: list[np.ndarray] = []
    for idx, col in enumerate(columns):
        arr = np.asarray(col, dtype=np.float64).reshape(-1)
        if idx == 0:
            designColumns.append(np.ones_like(arr))
            continue
        mask = valid & np.isfinite(arr)
        if not np.any(mask):
            designColumns.append(np.zeros_like(arr))
            continue
        center = float(np.average(arr[mask], weights=transitionWeights[mask]))
        spread = float(
            np.sqrt(
                np.average(
                    (arr[mask] - center) * (arr[mask] - center),
                    weights=transitionWeights[mask],
                )
            )
        )
        if not np.isfinite(spread) or spread <= 0.0:
            spread = 1.0
        designColumns.append((arr - center) / spread)
    X = np.column_stack(designColumns)
    fitMask = valid & np.all(np.isfinite(X), axis=1)
    sqrtW = np.sqrt(transitionWeights[fitMask])
    Xw = X[fitMask, :] * sqrtW[:, None]
    yw = y[fitMask] * sqrtW
    ridge = np.eye(X.shape[1], dtype=np.float64) * float(puncPriorRidge)
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(Xw.T @ Xw + ridge, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw.T @ Xw + ridge, Xw.T @ yw, rcond=None)[0]
    priorScale = np.exp(np.clip(X @ beta, -30.0, 30.0))
    priorGeom = _weightedGeometricMean(
        priorScale[valid],
        transitionWeights[valid],
        floor=tiny,
    )
    priorScale = priorScale / max(priorGeom, tiny)
    priorScaleBeforeDeadband = priorScale.copy()
    deadbandProbability = np.zeros(transitionCount, dtype=np.float64)
    deadbandWeight = np.zeros(transitionCount, dtype=np.float64)
    deadbandEnabled = bool(levelBufferZ > 0.0)
    if deadbandEnabled:
        denom = np.maximum(levelMidSd, tiny)
        r = np.divide(
            levelMidRaw,
            denom,
            out=np.zeros_like(levelMidRaw, dtype=np.float64),
            where=np.isfinite(denom) & (denom > 0.0),
        )
        deadbandProbability = stats.norm.cdf(levelBufferZ - r) - stats.norm.cdf(
            -levelBufferZ - r
        )
        deadbandProbability = np.clip(
            np.nan_to_num(deadbandProbability, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1.0,
        )
        deadbandWeight = np.clip(
            float(PROCESS_DEFAULT_PUNC_DEADBAND_PRIOR_WEIGHT) * deadbandProbability,
            0.0,
            1.0,
        )
        logPrior = np.log(np.maximum(priorScale, tiny))
        logNull = math.log(max(minScale, tiny))
        priorScale = np.exp(
            np.clip(
                (1.0 - deadbandWeight) * logPrior + deadbandWeight * logNull,
                -30.0,
                30.0,
            )
        )
        deadbandGeom = _weightedGeometricMean(
            priorScale[valid],
            transitionWeights[valid],
            floor=tiny,
        )
        priorScale = priorScale / max(deadbandGeom, tiny)
    levelBufferDiagnostics: dict[str, Any] = {
        "puncLevelBufferZ": float(levelBufferZ),
        "puncLevelBufferEnabled": bool(levelBufferZ > 0.0),
        "puncBufferedLevelZeroFraction": 0.0,
        "puncBufferedLevelMedianShrinkage": 1.0,
        "puncLevelMidpointRawSummary": _metadataTrackSummary(levelMidRaw),
        "puncLevelMidpointBufferedSummary": _metadataTrackSummary(levelMid),
        "puncLevelMidpointSdSummary": _metadataTrackSummary(levelMidSd),
        "puncDeadbandPriorEnabled": bool(deadbandEnabled),
        "puncDeadbandZ": float(levelBufferZ),
        "puncDeadbandMeanProbability": 0.0,
        "puncDeadbandHighProbabilityFraction": 0.0,
        "puncDeadbandNullScale": float(minScale),
        "puncPriorScaleBeforeDeadbandSummary": _metadataTrackSummary(
            priorScaleBeforeDeadband
        ),
        "puncPriorScaleAfterDeadbandSummary": _metadataTrackSummary(priorScale),
    }
    if np.any(valid):
        validWeights = transitionWeights[valid]
        rawAbs = np.abs(levelMidRaw[valid])
        bufferedAbs = np.abs(levelMid[valid])
        zeroIndicator = (bufferedAbs <= tiny).astype(np.float64)
        levelBufferDiagnostics["puncBufferedLevelZeroFraction"] = float(
            np.average(zeroIndicator, weights=validWeights)
        )
        shrinkMask = rawAbs > tiny
        if np.any(shrinkMask):
            shrink = bufferedAbs[shrinkMask] / rawAbs[shrinkMask]
            shrinkWeights = validWeights[shrinkMask]
            levelBufferDiagnostics["puncBufferedLevelMedianShrinkage"] = float(
                _weightedQuantile(shrink, shrinkWeights, np.asarray([0.5]))[0]
            )
        if deadbandEnabled:
            deadbandValid = deadbandProbability[valid]
            levelBufferDiagnostics["puncDeadbandMeanProbability"] = float(
                np.average(deadbandValid, weights=validWeights)
            )
            levelBufferDiagnostics["puncDeadbandHighProbabilityFraction"] = float(
                np.average(
                    (deadbandValid >= float(_PUNC_DEADBAND_HIGH_PROBABILITY)).astype(
                        np.float64
                    ),
                    weights=validWeights,
                )
            )
    _logEvent(
        "process_noise.punc_level_buffer",
        (
            ("z", float(levelBufferZ)),
            ("enabled", bool(levelBufferZ > 0.0)),
            (
                "zero_fraction",
                levelBufferDiagnostics["puncBufferedLevelZeroFraction"],
            ),
            (
                "median_shrinkage",
                levelBufferDiagnostics["puncBufferedLevelMedianShrinkage"],
            ),
        ),
    )
    _logEvent(
        "process_noise.punc_deadband_prior",
        (
            ("enabled", bool(deadbandEnabled)),
            ("z", float(levelBufferZ)),
            ("mean_probability", levelBufferDiagnostics["puncDeadbandMeanProbability"]),
            (
                "high_probability_fraction",
                levelBufferDiagnostics["puncDeadbandHighProbabilityFraction"],
            ),
            ("null_scale", float(minScale)),
        ),
    )

    windowLength = max(
        3,
        int(round(max(1, int(blockLenIntervals)) * float(puncLocalWindowMultiplier))),
    )
    windowLength = min(windowLength, transitionCount)
    halfWindow = max(1, windowLength // 2)
    stride = max(1, halfWindow)
    centers = list(range(0, transitionCount, stride))
    if centers[-1] != transitionCount - 1:
        centers.append(transitionCount - 1)
    if puncUseReliabilityWeightedWindows:
        w = np.where(valid, transitionWeights, 0.0)
    else:
        w = valid.astype(np.float64)
    wu = w * np.where(valid, evidence, 0.0)
    wLogG = w * np.log(np.maximum(priorScale, tiny))
    w2 = w * w
    cumW = np.concatenate(([0.0], np.cumsum(w)))
    cumWU = np.concatenate(([0.0], np.cumsum(wu)))
    cumWLogG = np.concatenate(([0.0], np.cumsum(wLogG)))
    cumW2 = np.concatenate(([0.0], np.cumsum(w2)))
    diffWeight = np.zeros(transitionCount + 1, dtype=np.float64)
    diffLogScale = np.zeros(transitionCount + 1, dtype=np.float64)
    priorDf = 1.0e6
    priorDfScale = 1.0
    priorDfDiagnostics: dict[str, Any] = {}
    dependence = float(max(puncDependenceMultiplier, tiny))
    minWindowWeight = float(max(puncMinWindowWeight, 0.0))
    windowStats: list[tuple[int, int, float, float, float, float]] = []
    for center in centers:
        start = max(0, int(center) - halfWindow)
        end = min(transitionCount, int(center) + halfWindow + 1)
        sumW = float(cumW[end] - cumW[start])
        if sumW < minWindowWeight:
            continue
        sumWU = float(cumWU[end] - cumWU[start])
        sumW2 = float(cumW2[end] - cumW2[start])
        if sumWU <= 0.0 or sumW2 <= 0.0:
            continue
        localEvidence = sumWU / sumW
        localPrior = math.exp(float(cumWLogG[end] - cumWLogG[start]) / sumW)
        effN = (sumW * sumW) / sumW2
        stateDimForNu = 1 if stateModelMode == STATE_MODEL_LEVEL else 2
        nuLocal = max(0.0, float(stateDimForNu) * effN / dependence)
        windowStats.append((start, end, sumW, localEvidence, localPrior, nuLocal))

    if puncToggles["puncUsePriorDfMoments"] and windowStats:
        priorDf, priorDfScale, priorDfDiagnostics = _estimatePuncPriorDfMethodOfMoments(
            np.asarray([row[3] for row in windowStats], dtype=np.float64),
            np.asarray([row[4] for row in windowStats], dtype=np.float64),
            np.asarray([row[5] for row in windowStats], dtype=np.float64),
            np.asarray([row[2] for row in windowStats], dtype=np.float64),
            minScale=minScale,
            maxScale=maxScale,
        )
    else:
        priorDf = float(puncPriorDf)
        priorDfScale = 1.0
        priorDfDiagnostics = {
            "puncPriorDfMomentWindowCount": 0,
            "puncPriorDfMomentEffectiveWindowCount": 0.0,
            "puncPriorDfMomentLogRatioVariance": float("nan"),
            "puncPriorDfMomentSamplingVariance": float("nan"),
            "puncPriorDfMomentExcessVariance": float("nan"),
            "puncPriorDfMomentScale": 1.0,
            "puncPriorDfMomentWinsorLower": float("nan"),
            "puncPriorDfMomentWinsorUpper": float("nan"),
            "puncPriorDfMomentReason": (
                "disabled"
                if not puncToggles["puncUsePriorDfMoments"]
                else "insufficient_windows"
            ),
        }

    _logEvent(
        "process_noise.punc_prior_df_moments",
        (
            ("prior_df", float(priorDf)),
            ("prior_scale", float(priorDfScale)),
            ("reason", priorDfDiagnostics.get("puncPriorDfMomentReason", "ok")),
            (
                "windows",
                int(priorDfDiagnostics.get("puncPriorDfMomentWindowCount", 0)),
            ),
            (
                "effective_windows",
                float(
                    priorDfDiagnostics.get(
                        "puncPriorDfMomentEffectiveWindowCount",
                        0.0,
                    )
                ),
            ),
            (
                "log_ratio_var",
                priorDfDiagnostics.get("puncPriorDfMomentLogRatioVariance", np.nan),
            ),
            (
                "sampling_var",
                priorDfDiagnostics.get("puncPriorDfMomentSamplingVariance", np.nan),
            ),
            (
                "excess_var",
                priorDfDiagnostics.get("puncPriorDfMomentExcessVariance", np.nan),
            ),
        ),
    )

    windowCount = 0
    for start, end, sumW, localEvidence, localPrior, nuLocal in windowStats:
        targetPrior = float(priorDfScale) * float(localPrior)
        if puncToggles["puncUsePriorShrinkage"]:
            scale = (nuLocal * localEvidence + priorDf * targetPrior) / max(
                nuLocal + priorDf,
                tiny,
            )
        else:
            scale = float(localEvidence)
        if puncToggles["puncUseBoundaryClamps"]:
            scale = float(np.clip(scale, minScale, maxScale))
        elif not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "PUNC local scale must stay finite and positive when "
                "puncUseBoundaryClamps=False"
            )
        paintWeight = max(sumW, tiny)
        diffWeight[start] += paintWeight
        diffWeight[end] -= paintWeight
        diffLogScale[start] += paintWeight * math.log(max(scale, tiny))
        diffLogScale[end] -= paintWeight * math.log(max(scale, tiny))
        windowCount += 1
    paintWeightByTransition = np.cumsum(diffWeight[:-1])
    paintLogScaleByTransition = np.cumsum(diffLogScale[:-1])
    painted = np.divide(
        paintLogScaleByTransition,
        paintWeightByTransition,
        out=np.log(np.maximum(priorScale, tiny)),
        where=paintWeightByTransition > 0.0,
    )
    if puncToggles["puncUseBoundaryClamps"]:
        rawScale = np.exp(np.clip(painted, math.log(minScale), math.log(maxScale)))
    else:
        rawScale = np.exp(np.clip(painted, -30.0, 30.0))
    if puncToggles["puncUseGlobalScale"]:
        globalScale = _weightedGeometricMean(
            rawScale[valid], transitionWeights[valid], floor=tiny
        )
        if puncToggles["puncUseBoundaryClamps"]:
            globalScale = float(np.clip(globalScale, minScale, maxScale))
        else:
            globalScale = float(globalScale)
    else:
        globalScale = 1.0
    transitionScale = rawScale / max(globalScale, tiny)
    if puncToggles["puncUseBoundaryClamps"]:
        transitionScale = np.clip(transitionScale, minScale, maxScale)
    unclampedBaseQ0 = np.asarray(seedQClamped, dtype=np.float64) * globalScale
    if puncToggles["puncUseBoundaryClamps"]:
        matrixQ0Punc = _clampProcessNoiseMatrix(
            unclampedBaseQ0,
            stateModel=stateModelMode,
            minQ=float(minQ),
            maxQ=float(maxQ),
        )
    else:
        activeDiag = _activeProcessQDiagonal(
            unclampedBaseQ0,
            stateModel=stateModelMode,
        )
        if not np.all(np.isfinite(activeDiag)) or np.any(activeDiag <= 0.0):
            raise ValueError(
                "PUNC base process Q must stay finite and positive when "
                "puncUseBoundaryClamps=False"
            )
        matrixQ0Punc = np.ascontiguousarray(unclampedBaseQ0, dtype=np.float32)
    if puncToggles["puncUseScaleRebase"]:
        (
            rebasedTransitionScale,
            decompMaxLogError,
            decompMedianLogError,
        ) = cconsenrich.crebasePuncIntervalScales(
            seedQClamped,
            matrixQ0Punc,
            rawScale,
            stateModelMode,
        )
        rebasedTransitionScale = np.asarray(rebasedTransitionScale, dtype=np.float32)
    else:
        rebasedTransitionScale = np.asarray(transitionScale, dtype=np.float32)
        decompMaxLogError = 0.0
        decompMedianLogError = 0.0
    processQScale[1:] = rebasedTransitionScale
    processQScale[0] = 1.0
    deadbandHighMask = (
        valid
        & deadbandEnabled
        & (deadbandProbability >= float(_PUNC_DEADBAND_HIGH_PROBABILITY))
    )
    highTransitionCount = int(np.count_nonzero(deadbandHighMask))
    validTransitionCount = int(np.count_nonzero(valid))
    highTransitionFraction = (
        float(highTransitionCount / validTransitionCount)
        if validTransitionCount
        else 0.0
    )
    highTransitionWeight = (
        float(np.sum(transitionWeights[deadbandHighMask]))
        if highTransitionCount
        else 0.0
    )
    highIntervalMask = np.zeros(intervalCount, dtype=bool)
    highIntervalMask[1:] = deadbandHighMask
    levelQTrack = float(matrixQ0Punc[0, 0]) * rebasedTransitionScale
    if stateModelMode == STATE_MODEL_LEVEL:
        trendQTrack = np.zeros_like(levelQTrack, dtype=np.float64)
    else:
        trendQTrack = float(matrixQ0Punc[1, 1]) * rebasedTransitionScale
    baseDiagTarget = _activeProcessQDiagonal(unclampedBaseQ0, stateModel=stateModelMode)
    baseDiagActual = _activeProcessQDiagonal(matrixQ0Punc, stateModel=stateModelMode)
    baseMask = (
        np.isfinite(baseDiagTarget)
        & np.isfinite(baseDiagActual)
        & (baseDiagTarget > 0.0)
        & (baseDiagActual > 0.0)
    )
    if np.any(baseMask):
        baseRel = np.abs(baseDiagActual[baseMask] / baseDiagTarget[baseMask] - 1.0)
        baseClampMaxRelativeChange = float(np.max(baseRel))
    else:
        baseClampMaxRelativeChange = 0.0
    baseQClampChanged = bool(baseClampMaxRelativeChange > 1.0e-6)
    clampFraction = float(
        np.mean(
            (transitionScale <= minScale * 1.0001)
            | (transitionScale >= maxScale * 0.9999)
        )
    )
    boundary = _processNoiseQBoundaryDiagnostics(
        matrixQ0Punc,
        stateModelMode,
        minQ=float(minQ),
        maxQ=float(maxQ),
    )
    levelQ = float(boundary["preKappaQLevel"])
    trendQ = float(boundary["preKappaQTrend"])
    trendLevelRatio = (
        0.0
        if stateModelMode == STATE_MODEL_LEVEL
        else trendQ / max(levelQ, float(boundary["qFloor"]))
    )
    diagnostics = {
        "processNoisePolicy": PROCESS_NOISE_CALIBRATION_PUNC,
        "processNoiseCalibrationStatus": "estimated",
        "processNoiseCalibrationReason": "ok",
        "globalScale": float(globalScale),
        "validTransitionCount": int(np.count_nonzero(valid)),
        "transitionCount": int(transitionCount),
        "windowCount": int(windowCount),
        "windowLength": int(windowLength),
        "qScaleClampFraction": float(clampFraction),
        "puncPriorDf": float(priorDf),
        "puncPriorDfSource": (
            "method_of_moments"
            if puncToggles["puncUsePriorDfMoments"] and windowStats
            else "configured"
        ),
        "puncPriorScale": float(priorDfScale),
        "puncLevelBufferZ": float(levelBufferZ),
        "puncDependenceMultiplier": float(puncDependenceMultiplier),
        "puncLocalWindowMultiplier": float(puncLocalWindowMultiplier),
        "puncUseReliabilityWeightedWindows": bool(puncUseReliabilityWeightedWindows),
        **puncToggles,
        "puncStagesActive": True,
        "processCovariateCount": int(0 if covariates is None else covariates.shape[1]),
        "priorDesignColumnCount": int(len(priorDesignColumns)),
        "priorDesignColumns": tuple(priorDesignColumns),
        "rawTrendLevelRatio": float(trendLevelRatio),
        "effectiveTrendLevelRatio": float(trendLevelRatio),
        "logQLevel": float(np.log(max(levelQ, float(boundary["qFloor"])))),
        "logQTrend": (
            0.0
            if stateModelMode == STATE_MODEL_LEVEL
            else float(np.log(max(trendQ, float(boundary["qFloor"]))))
        ),
        "baseQClampChanged": baseQClampChanged,
        "baseQClampMaxRelativeChange": float(baseClampMaxRelativeChange),
        "qScaleDecompositionMaxLogError": float(decompMaxLogError),
        "qScaleDecompositionMedianLogError": float(decompMedianLogError),
        "processQScaleSummary": _metadataTrackSummary(processQScale),
        "puncDeadbandHighProbabilityThreshold": float(_PUNC_DEADBAND_HIGH_PROBABILITY),
        "puncDeadbandHighTransitionCount": int(highTransitionCount),
        "puncDeadbandHighTransitionFraction": float(highTransitionFraction),
        "puncDeadbandHighTransitionWeight": float(highTransitionWeight),
        "puncDeadbandHighRawScaleSummary": _metadataTrackSummaryWhere(
            rawScale,
            deadbandHighMask,
        ),
        "puncDeadbandHighTransitionScaleSummary": _metadataTrackSummaryWhere(
            transitionScale,
            deadbandHighMask,
        ),
        "puncDeadbandHighRebasedProcessQScaleSummary": _metadataTrackSummaryWhere(
            rebasedTransitionScale,
            deadbandHighMask,
        ),
        "puncDeadbandHighQLevelSummary": _metadataTrackSummaryWhere(
            levelQTrack,
            deadbandHighMask,
        ),
        "puncDeadbandHighQTrendSummary": _metadataTrackSummaryWhere(
            trendQTrack,
            deadbandHighMask,
        ),
        "matrixQ0Final": matrixQ0Punc.astype(float).tolist(),
        "_puncDeadbandHighIntervalMask": highIntervalMask,
    }
    diagnostics.update(levelBufferDiagnostics)
    diagnostics.update(priorDfDiagnostics)
    diagnostics.update(boundary)
    return matrixQ0Punc, processQScale, diagnostics


def _resolveProcessNoiseFloor(minQ: float) -> float:
    return _checkFinitePositive("minQ", minQ)


def _resolveProcessNoiseCap(
    maxQ: float,
    *,
    minQ: float,
) -> float:
    qFloor = _resolveProcessNoiseFloor(minQ)
    maxQValue = float(maxQ)
    if maxQValue < 0.0:
        return float(np.inf)
    if np.isfinite(maxQValue):
        return float(max(maxQValue, qFloor))
    return float(np.inf)


def _clampProcessNoise(value: float, *, qFloor: float, qCap: float) -> float:
    qFloor_ = _resolveProcessNoiseFloor(qFloor)
    value_ = float(value)
    if not np.isfinite(value_):
        value_ = qFloor_
    value_ = max(value_, qFloor_)
    if np.isfinite(qCap):
        value_ = min(value_, float(qCap))
    return float(value_)


def _clampProcessNoiseMatrix(
    matrixQ0: np.ndarray,
    *,
    stateModel: str,
    minQ: float,
    maxQ: float,
) -> np.ndarray:
    stateModelMode = _normalizeStateModel(stateModel)
    qFloor = _resolveProcessNoiseFloor(minQ)
    qCap = _resolveProcessNoiseCap(maxQ, minQ=qFloor)
    q0 = np.asarray(matrixQ0, dtype=np.float64)
    levelVariance = _clampProcessNoise(float(q0[0, 0]), qFloor=qFloor, qCap=qCap)
    if stateModelMode == STATE_MODEL_LEVEL:
        return np.asarray([[levelVariance]], dtype=np.float32)
    trendVariance = _clampProcessNoise(float(q0[1, 1]), qFloor=qFloor, qCap=qCap)
    return constructMatrixQ(
        minDiagQ=qFloor,
        Q00=levelVariance,
        Q01=float(q0[0, 1]) if q0.shape[0] > 1 and q0.shape[1] > 1 else 0.0,
        Q10=float(q0[1, 0]) if q0.shape[0] > 1 and q0.shape[1] > 1 else 0.0,
        Q11=trendVariance,
    ).astype(np.float32, copy=False)


def _activeObservationMaskForProcessNoise(
    *,
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    pad: float,
) -> np.ndarray:
    active, _obsVariance = _activeProcessNoiseObservationMask(
        matrixData,
        matrixMunc,
        float(pad),
    )
    return active


def _countActiveAdjacentProcessNoiseTransitions(
    *,
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    pad: float,
) -> int:
    active = _activeObservationMaskForProcessNoise(
        matrixData=matrixData,
        matrixMunc=matrixMunc,
        pad=float(pad),
    )
    if active.ndim != 2 or active.shape[1] < 2:
        return 0
    intervalActive = np.any(active, axis=0)
    return int(np.count_nonzero(intervalActive[1:] & intervalActive[:-1]))


def _qSeedPosteriorFromTransitions(
    *,
    deltas: np.ndarray,
    samplingVariances: np.ndarray,
    transitionWeights: np.ndarray,
    qFloor: float,
    qCap: float,
    robustTNu: float,
    source: str,
    qSeedPriorLevel: float,
) -> dict[str, Any]:
    return cconsenrich.cQSeedPosteriorFromTransitions(
        deltas,
        samplingVariances,
        transitionWeights,
        float(qFloor),
        float(qCap),
        float(robustTNu),
        str(source),
        float(qSeedPriorLevel),
        int(_QINIT_MIN_TRANSITIONS),
        float(_QINIT_PRIOR_LOG_SD),
        float(_QINIT_DEFAULT_T_NU),
        int(_QINIT_GRID_SIZE),
    )


def _estimateSameTrackProcessNoiseTransitions(
    *,
    matrixData: np.ndarray,
    obsVar: np.ndarray,
    finiteMask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    return cconsenrich.cEstimateSameTrackProcessNoiseTransitions(
        matrixData,
        obsVar,
        finiteMask,
        float(_QINIT_PRECISION_CAP_QUANTILE),
        float(_QINIT_PRECISION_CAP_MULTIPLIER),
        int(_QINIT_MAX_TRANSITIONS),
        int(_QINIT_PRECISION_SAMPLE_CAP),
    )


def _estimatePooledProcessNoiseTransitions(
    *,
    matrixData: np.ndarray,
    obsVar: np.ndarray,
    finiteMask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return cconsenrich.cEstimatePooledProcessNoiseTransitions(
        matrixData,
        obsVar,
        finiteMask,
    )


def _estimateInitialProcessNoiseFromData(
    *,
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    pad: float,
    stateModel: str,
    minQ: float,
    maxQ: float,
    deltaF: float,
    puncMaxScale: float,
    processNoiseCalibration: str,
    robustTNu: float | None,
    qSeedPriorLevel: float = PROCESS_DEFAULT_Q_SEED_PRIOR_LEVEL,
) -> tuple[np.ndarray, dict[str, Any]]:
    qFloor = _resolveProcessNoiseFloor(minQ)
    qCap = _resolveProcessNoiseCap(maxQ, minQ=qFloor)
    qSeedPriorFloor = _resolveProcessNoiseFloor(qSeedPriorLevel)
    if np.isfinite(qCap) and qSeedPriorFloor > qCap:
        raise ValueError("`qSeedPriorLevel` must not exceed `maxQ`")
    data = np.asarray(matrixData, dtype=np.float64)
    munc = np.asarray(matrixMunc, dtype=np.float64)
    if data.shape != munc.shape:
        raise ValueError("matrixData and matrixMunc must have matching shapes")
    obsVarRaw = munc + float(pad)
    finiteMask = _activeObservationMaskForProcessNoise(
        matrixData=data,
        matrixMunc=munc,
        pad=float(pad),
    )
    obsVar = np.maximum(obsVarRaw, 1.0e-12)

    robustNu = (
        _QINIT_DEFAULT_T_NU
        if robustTNu is None or not np.isfinite(float(robustTNu))
        else float(robustTNu)
    )
    deltas, samplingVariances, transitionWeights, sameTrackDiagnostics = (
        _estimateSameTrackProcessNoiseTransitions(
            matrixData=data,
            obsVar=obsVar,
            finiteMask=finiteMask,
        )
    )
    estimate = _qSeedPosteriorFromTransitions(
        deltas=deltas,
        samplingVariances=samplingVariances,
        transitionWeights=transitionWeights,
        qFloor=qFloor,
        qCap=qCap,
        robustTNu=robustNu,
        source="sameTrackEB",
        qSeedPriorLevel=qSeedPriorFloor,
    )
    if not bool(estimate.get("ok", False)):
        pooledDeltas, pooledSamplingVariances, pooledTransitionWeights = (
            _estimatePooledProcessNoiseTransitions(
                matrixData=data,
                obsVar=obsVar,
                finiteMask=finiteMask,
            )
        )
        pooledEstimate = _qSeedPosteriorFromTransitions(
            deltas=pooledDeltas,
            samplingVariances=pooledSamplingVariances,
            transitionWeights=pooledTransitionWeights,
            qFloor=qFloor,
            qCap=qCap,
            robustTNu=robustNu,
            source="pooledEB",
            qSeedPriorLevel=qSeedPriorFloor,
        )
        if bool(pooledEstimate.get("ok", False)):
            estimate = pooledEstimate

    source = str(estimate.get("source", "fallback"))
    reason = str(estimate.get("reason", "ok"))
    qPosteriorMedian = float(estimate.get("posteriorMedianLevel", np.nan))
    qTransition90 = float(estimate.get("transitionQ90", np.nan))
    qBeforeGuardrail = qPosteriorMedian
    guardrailApplied = False
    calibrationMode = _normalizeProcessNoiseCalibrationMode(processNoiseCalibration)
    puncScale = float(puncMaxScale)
    if (
        calibrationMode == PROCESS_NOISE_CALIBRATION_PUNC
        and np.isfinite(puncScale)
        and puncScale > 0.0
        and np.isfinite(qTransition90)
        and qTransition90 > 0.0
    ):
        guarded = max(qPosteriorMedian, qTransition90 / puncScale)
        guardrailApplied = bool(
            np.isfinite(qPosteriorMedian) and guarded > qPosteriorMedian * 1.000001
        )
        qBeforeGuardrail = guarded

    if not np.isfinite(qBeforeGuardrail) or qBeforeGuardrail <= 0.0:
        fallbackPool = obsVar[finiteMask]
        fallbackPool = fallbackPool[np.isfinite(fallbackPool) & (fallbackPool > 0.0)]
        fallbackVar = (
            float(np.median(fallbackPool)) if fallbackPool.size else float("nan")
        )
        qBeforeGuardrail = (
            1.0e-4 * fallbackVar
            if np.isfinite(fallbackVar) and fallbackVar > 0.0
            else qFloor
        )
        source = "observationVarianceFloor" if np.isfinite(fallbackVar) else "minQ"
        reason = (
            "fallback_observation_variance"
            if np.isfinite(fallbackVar)
            else "fallback_min_q"
        )
    qInit = _clampProcessNoise(qBeforeGuardrail, qFloor=qFloor, qCap=qCap)
    stateModelMode = _normalizeStateModel(stateModel)
    if stateModelMode == STATE_MODEL_LEVEL_TREND:
        deltaF_ = max(float(deltaF), 1.0e-12)
        qTrendRaw = (
            qInit * float(PROCESS_DEFAULT_PUNC_TREND_SEED_RATIO) / (deltaF_ * deltaF_)
        )
        qTrendInit = _clampProcessNoise(
            qTrendRaw,
            qFloor=qFloor,
            qCap=qCap,
        )
    else:
        qTrendInit = qInit
        qTrendRaw = qTrendInit
    matrixQ = constructMatrixQ(
        minDiagQ=qFloor,
        Q00=qInit,
        Q01=0.0,
        Q10=0.0,
        Q11=qTrendInit,
    ).astype(np.float32, copy=False)
    qSeedClampChanged = bool(
        abs(qInit / max(qBeforeGuardrail, qFloor) - 1.0) > 1.0e-6
        if np.isfinite(qBeforeGuardrail) and qBeforeGuardrail > 0.0
        else False
    )
    qSeedPuncCoverageQ90 = (
        qTransition90 / max(qInit * puncScale, np.finfo(np.float64).tiny)
        if calibrationMode == PROCESS_NOISE_CALIBRATION_PUNC
        and np.isfinite(qTransition90)
        and np.isfinite(puncScale)
        and puncScale > 0.0
        else float("nan")
    )
    diagnostics = {
        "qSeedSource": source,
        "qSeedReason": reason,
        "qSeedTransitionCount": int(estimate.get("transitionCount", 0)),
        "qSeedEffectiveTransitionCount": float(
            estimate.get("effectiveTransitionCount", 0.0)
        ),
        "qSeedPairCount": int(sameTrackDiagnostics.get("pairCount", 0)),
        "qSeedPrecisionCapFraction": float(
            sameTrackDiagnostics.get("precisionCapFraction", 0.0)
        ),
        "qSeedPriorLevel": float(estimate.get("priorLevel", np.nan)),
        "qSeedPosteriorMedianLevel": float(
            estimate.get("posteriorMedianLevel", np.nan)
        ),
        "qSeedPosteriorModeLevel": float(estimate.get("posteriorModeLevel", np.nan)),
        "qSeedPosteriorQ05Level": float(estimate.get("posteriorQ05Level", np.nan)),
        "qSeedPosteriorQ95Level": float(estimate.get("posteriorQ95Level", np.nan)),
        "qSeedTransitionQ90": float(qTransition90),
        "qSeedPuncCoverageQ90": float(qSeedPuncCoverageQ90),
        "qSeedGuardrailApplied": bool(guardrailApplied),
        "qSeedLevelPreClamp": float(qBeforeGuardrail),
        "qSeedTrendPreClamp": float(qTrendRaw),
        "qSeedLevelFinal": float(qInit),
        "qSeedTrendFinal": float(qTrendInit),
        "qSeedClampChanged": bool(qSeedClampChanged),
        "qSeedTrendLevelRatio": float(qTrendInit / max(qInit, qFloor)),
        "qSeedMedianSamplingVariance": float(
            estimate.get("medianSamplingVariance", np.nan)
        ),
    }
    return matrixQ, diagnostics


def constructMatrixQ(
    minDiagQ: float,
    Q00: Optional[float] = None,
    Q01: Optional[float] = 0.0,
    Q10: Optional[float] = 0.0,
    Q11: Optional[float] = None,
    useIdentity: float = -1.0,
    tol: float = 1.0e-8,  # conservative
) -> npt.NDArray[np.float32]:
    r"""Build the (base) process noise covariance matrix :math:`\mathbf{Q}`.

    :param minDiagQ: Minimum value for diagonal entries of :math:`\mathbf{Q}`.
    :type minDiagQ: float
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

    minDiagQ_ = _checkFinitePositive("minDiagQ", minDiagQ)

    if useIdentity > 0.0:
        return np.eye(2, dtype=np.float32) * np.float32(
            max(float(useIdentity), minDiagQ_)
        )

    def _diagOrFloor(value: Optional[float]) -> float:
        if value is None:
            return minDiagQ_
        value_ = float(value)
        if not np.isfinite(value_):
            return minDiagQ_
        return max(value_, minDiagQ_)

    Q = np.empty((2, 2), dtype=np.float32)

    Q[0, 0] = np.float32(_diagOrFloor(Q00))
    Q[1, 1] = np.float32(_diagOrFloor(Q11))

    if Q11 is None:
        Q[1, 1] = Q[0, 0]

    if Q01 is not None and Q10 is None:
        Q10 = Q01
    elif Q10 is not None and Q01 is None:
        Q01 = Q10

    Q[0, 1] = np.float32(0.0 if Q01 is None else Q01)
    Q[1, 0] = np.float32(0.0 if Q10 is None else Q10)

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
    *,
    stateInit: float,
    stateCovarInit: float,
    boundState: bool,
    stateLowerBound: float,
    stateUpperBound: float,
    blockLenIntervals: int,
    intervalSizeBP: int | None = None,
    projectStateDuringFiltering: bool = False,
    pad: float = 1.0e-4,
    ECM_fixedBackgroundIters: int = 50,
    ECM_fixedBackgroundRtol: float = 1.0e-4,
    t_innerIters: int = FIT_DEFAULT_T_INNER_ITERS,
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
    useNonnegativeBackground: bool = FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND,
    backgroundNegativePenaltyMultiplier: float | None = (
        FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER
    ),
    returnScales: bool = True,
    returnBackground: bool = False,
    stateModel: str | None = STATE_MODEL_LEVEL_TREND,
    processNoiseCalibration: str = PROCESS_DEFAULT_NOISE_CALIBRATION,
    qPriorLevel: float = PROCESS_DEFAULT_Q_PRIOR_LEVEL,
    qPriorTrend: float = PROCESS_DEFAULT_Q_PRIOR_TREND,
    qSeedPriorLevel: float = PROCESS_DEFAULT_Q_SEED_PRIOR_LEVEL,
    puncLocalWindowMultiplier: float = PROCESS_DEFAULT_PUNC_LOCAL_WINDOW_MULTIPLIER,
    puncDependenceMultiplier: float = PROCESS_DEFAULT_PUNC_DEPENDENCE_MULTIPLIER,
    puncMinScale: float = PROCESS_DEFAULT_PUNC_MIN_SCALE,
    puncMaxScale: float = PROCESS_DEFAULT_PUNC_MAX_SCALE,
    puncMinWindowWeight: float = PROCESS_DEFAULT_PUNC_MIN_WINDOW_WEIGHT,
    puncPriorDf: float = PROCESS_DEFAULT_PUNC_PRIOR_DF,
    puncPriorRidge: float = PROCESS_DEFAULT_PUNC_PRIOR_RIDGE,
    puncLevelBufferZ: float = PROCESS_DEFAULT_PUNC_LEVEL_BUFFER_Z,
    puncUseReliabilityWeightedWindows: bool = (
        PROCESS_DEFAULT_PUNC_USE_RELIABILITY_WEIGHTED_WINDOWS
    ),
    puncUseWarmupFit: bool = PROCESS_DEFAULT_PUNC_USE_WARMUP_FIT,
    puncUseTransitionEvidence: bool = PROCESS_DEFAULT_PUNC_USE_TRANSITION_EVIDENCE,
    puncUseScaleRebase: bool = PROCESS_DEFAULT_PUNC_USE_SCALE_REBASE,
    puncUseGlobalScale: bool = PROCESS_DEFAULT_PUNC_USE_GLOBAL_SCALE,
    puncUseBoundaryClamps: bool = PROCESS_DEFAULT_PUNC_USE_BOUNDARY_CLAMPS,
    puncUsePriorDfMoments: bool = PROCESS_DEFAULT_PUNC_USE_PRIOR_DF_MOMENTS,
    puncUsePriorShrinkage: bool = PROCESS_DEFAULT_PUNC_USE_PRIOR_SHRINKAGE,
    processNoiseWarmupECMIters: int = PROCESS_DEFAULT_WARMUP_ECM_ITERS,
    processNoiseWarmupOuterPasses: int = PROCESS_DEFAULT_WARMUP_OUTER_PASSES,
    processCovariates: np.ndarray | None = None,
    observationPrecisionMultiplierMin: float = 0.25,
    observationPrecisionMultiplierMax: float = 4.0,
    processPrecisionMultiplierMin: float = PROCESS_DEFAULT_PRECISION_MULTIPLIER_MIN,
    processPrecisionMultiplierMax: float = PROCESS_DEFAULT_PRECISION_MULTIPLIER_MAX,
    observationMask: np.ndarray | None = None,
    initialBackground: np.ndarray | None = None,
    initialObservationPrecision: np.ndarray | None = None,
    initialProcessPrecision: np.ndarray | None = None,
    initialProcessQ: np.ndarray | None = None,
    trackOptimizationPath: bool = False,
    returnPrecisionDiagnostics: bool = False,
    returnDiagnostics: bool = False,
    logIndentLevel: int = 0,
    logRunRole: str | None = None,
):
    r"""Run Consenrich over a contiguous genomic region

    Consenrich estimates a shared signal level from multiple replicate tracks using a
    selectable latent smoother plus fixed-background ECM and an outer
    fit/background alternation loop.

    The observation model is

    .. math::

      z_{[j,i]} = g_{[i]} + x_{[i,0]} + \epsilon_{[j,i]},
      \qquad
      \mathrm{Var}(\epsilon_{[j,i]}) =
      \frac{v_{[j,i]} + \mathrm{pad}}{\lambda_{[i]}}.

    Here :math:`z_{[j,i]}` is the observed track value, :math:`g_{[i]}` is an
    optional low-frequency background shared across replicates, and
    :math:`v_{[j,i]}` is the plugin observation variance supplied by
    ``matrixMunc``.

    By default, the latent state follows the two-state level/trend model

    .. math::

      \mathbf{x}_{[i+1]} = \mathbf{F}(\delta_F)\mathbf{x}_{[i]} + \eta_{[i]},
      \qquad
      \mathrm{Var}(\eta_{[i]}) = \frac{\mathbf{Q}_0}{\kappa_{[i]}}.

    With ``stateModel="level"``, the model is scalar:

    .. math::

      x_{[i+1]} = x_{[i]} + \eta_{[i]}.

    This wrapper ties together several fundamental routines written in Cython:

    #. :func:`consenrich.cconsenrich.cforwardPass`: Forward filter (predict, update)
    #. :func:`consenrich.cconsenrich.cbackwardPass`: Backward fixed-interval smoother
    #. :func:`consenrich.cconsenrich.cfixedBackgroundECM`: Run ECM to convergence wrt a fixed :math:`g`.

    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.cconsenrich.cTransform`, :func:`consenrich.cconsenrich.cforwardPass`, :func:`consenrich.cconsenrich.cbackwardPass`, :func:`consenrich.cconsenrich.cfixedBackgroundECM`
    """

    matrixData, matrixMunc = _coerceMatrixDataMuncPair(matrixData, matrixMunc)
    matrixMunc = _applyObservationMaskToMunc(matrixMunc, observationMask)
    pad = _checkFiniteNonnegative("pad", pad)

    trackCount, intervalCount = matrixData.shape
    if intervalCount < 2:
        raise ValueError("need at least 2 intervals for smoothing")
    intervalSizeBPLocal = None
    if intervalSizeBP is not None:
        intervalSizeBPLocal = int(intervalSizeBP)
        if intervalSizeBPLocal <= 0:
            raise ValueError("intervalSizeBP must be positive when provided")

    requestedProcessPrecisionReweighting = bool(ECM_useProcessPrecisionReweighting)
    ECM_useAPN = bool(ECM_useAPN)
    if ECM_useAPN:
        ECM_useProcessPrecisionReweighting = False
    stateModelMode = _normalizeStateModel(stateModel)
    stateDim = 1 if stateModelMode == STATE_MODEL_LEVEL else 2
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
    ) = _checkProcessPrecisionMultiplierBounds(
        minValue=processPrecisionMultiplierMin,
        maxValue=processPrecisionMultiplierMax,
        robustTNu=ECM_robustTNu,
        stateDim=int(stateDim),
    )
    initialBackgroundArr = _coerceOptionalVector(
        "initialBackground",
        initialBackground,
        intervalCount,
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
    initialProcessQArr = _coerceOptionalProcessNoiseMatrix(initialProcessQ)
    minQ = _checkFinitePositive("minQ", minQ)
    maxQ = float(maxQ)
    if np.isnan(maxQ):
        raise ValueError("`maxQ` must not be NaN")
    maxQForAPN = np.inf if maxQ < 0.0 else max(maxQ, minQ)
    processNoiseCalibrationMode = _normalizeProcessNoiseCalibrationMode(
        processNoiseCalibration
    )

    def _checkPriorProcessQ(name: str, value: float) -> float:
        out = _checkFinitePositive(name, value)
        if out < float(minQ):
            raise ValueError(f"`{name}` must be greater than or equal to `minQ`")
        if np.isfinite(maxQForAPN) and out > maxQForAPN:
            raise ValueError(f"`{name}` must not exceed `maxQ`")
        return float(out)

    qPriorLevel = _checkPriorProcessQ("qPriorLevel", qPriorLevel)
    qPriorTrend = _checkPriorProcessQ("qPriorTrend", qPriorTrend)
    qSeedPriorLevel = _checkFinitePositive("qSeedPriorLevel", qSeedPriorLevel)
    if np.isfinite(maxQForAPN) and qSeedPriorLevel > maxQForAPN:
        raise ValueError("`qSeedPriorLevel` must not exceed `maxQ`")
    if processNoiseCalibrationMode == PROCESS_NOISE_CALIBRATION_PUNC and ECM_useAPN:
        raise ValueError(
            "processNoiseCalibration='punc' is mutually exclusive with ECM_useAPN=True"
        )
    puncLocalWindowMultiplier = _checkFinitePositive(
        "puncLocalWindowMultiplier",
        puncLocalWindowMultiplier,
    )
    puncDependenceMultiplier = _checkFinitePositive(
        "puncDependenceMultiplier",
        puncDependenceMultiplier,
    )
    puncMinScale = _checkFinitePositive("puncMinScale", puncMinScale)
    puncMaxScale = _checkFinitePositive("puncMaxScale", puncMaxScale)
    if float(puncMaxScale) < float(puncMinScale):
        raise ValueError("puncMaxScale must be greater than or equal to puncMinScale")
    puncMinWindowWeight = _checkFiniteNonnegative(
        "puncMinWindowWeight",
        puncMinWindowWeight,
    )
    puncPriorDf = _checkFinitePositive("puncPriorDf", puncPriorDf)
    puncPriorRidge = _checkFiniteNonnegative("puncPriorRidge", puncPriorRidge)
    puncLevelBufferZ = _checkFiniteNonnegative(
        "puncLevelBufferZ",
        puncLevelBufferZ,
    )
    if not isinstance(puncUseReliabilityWeightedWindows, (bool, np.bool_)):
        raise ValueError("puncUseReliabilityWeightedWindows must be boolean")
    puncUseReliabilityWeightedWindows = bool(puncUseReliabilityWeightedWindows)
    puncToggleValues = {
        "puncUseWarmupFit": puncUseWarmupFit,
        "puncUseTransitionEvidence": puncUseTransitionEvidence,
        "puncUseScaleRebase": puncUseScaleRebase,
        "puncUseGlobalScale": puncUseGlobalScale,
        "puncUseBoundaryClamps": puncUseBoundaryClamps,
        "puncUsePriorDfMoments": puncUsePriorDfMoments,
        "puncUsePriorShrinkage": puncUsePriorShrinkage,
    }
    for puncToggleName, puncToggleValue in puncToggleValues.items():
        if not isinstance(puncToggleValue, (bool, np.bool_)):
            raise ValueError(f"{puncToggleName} must be boolean")
    puncUseWarmupFit = bool(puncUseWarmupFit)
    puncUseTransitionEvidence = bool(puncUseTransitionEvidence)
    puncUseScaleRebase = bool(puncUseScaleRebase)
    puncUseGlobalScale = bool(puncUseGlobalScale)
    puncUseBoundaryClamps = bool(puncUseBoundaryClamps)
    puncUsePriorDfMoments = bool(puncUsePriorDfMoments)
    puncUsePriorShrinkage = bool(puncUsePriorShrinkage)
    processCovariatesArr = _coerceOptionalProcessCovariates(
        processCovariates,
        intervalCount=intervalCount,
    )
    qCalibrationSupport = _processNoiseCalibrationSupport(matrixData, matrixMunc, pad)
    processCalibrationSkipReason = (
        None
        if initialProcessQArr is not None
        or processNoiseCalibrationMode == PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL
        else qCalibrationSupport.get("processNoiseCalibrationSkipReason")
    )
    processNoiseWarmupECMIters = max(1, int(processNoiseWarmupECMIters))
    processNoiseWarmupOuterPasses = max(1, int(processNoiseWarmupOuterPasses))
    if isinstance(t_innerIters, (bool, np.bool_)):
        raise ValueError("t_innerIters must be a positive integer")
    try:
        t_innerIters = operator.index(t_innerIters)
    except TypeError as ex:
        raise ValueError("t_innerIters must be a positive integer") from ex
    if t_innerIters <= 0:
        raise ValueError("t_innerIters must be a positive integer")
    ECM_outerIters = max(1, int(ECM_outerIters))
    ECM_minOuterIters = (
        3 if ECM_minOuterIters is None else max(1, int(ECM_minOuterIters))
    )
    ECM_backgroundShiftRtol = float(max(ECM_backgroundShiftRtol, 0.0))
    ECM_outerNLLRtol = float(max(ECM_outerNLLRtol, 0.0))
    trackOptimizationPath = bool(trackOptimizationPath)
    useNonnegativeBackground = bool(useNonnegativeBackground)
    if backgroundNegativePenaltyMultiplier is None:
        backgroundNegativePenaltyMultiplierLocal = None
    else:
        backgroundNegativePenaltyMultiplierLocal = float(
            backgroundNegativePenaltyMultiplier
        )
        if not np.isfinite(backgroundNegativePenaltyMultiplierLocal):
            raise ValueError(
                "`backgroundNegativePenaltyMultiplier` must be finite or None"
            )
    negativeBackgroundPenaltyActive = bool(
        useNonnegativeBackground
        and backgroundNegativePenaltyMultiplierLocal is not None
        and backgroundNegativePenaltyMultiplierLocal > 0.0
    )
    if (
        bool(fitBackground)
        and negativeBackgroundPenaltyActive
        and bool(ECM_zeroCenterBackground)
    ):
        logger.warning(
            "fitParams.useNonnegativeBackground=True with a nonzero "
            "fitParams.backgroundNegativePenaltyMultiplier conflicts with "
            "fitParams.ECM_zeroCenterBackground=True: positive background mass "
            "must be balanced by negative background mass, so penalizing "
            "negative values may pull the shared background toward zero."
        )
    logIndentLevel = max(0, int(logIndentLevel or 0))
    logRunRole = str(logRunRole or "").strip()
    logRunLabel = logRunRole if logRunRole else "primary chromosome"
    logRunRoleLower = logRunRole.lower()
    logPrimaryRole = not logRunRoleLower or logRunRoleLower.startswith("primary")
    logDeepDetails = logger.isEnabledFor(logging.DEBUG)
    logCoreBlockLevel = logging.INFO if logPrimaryRole else logging.DEBUG
    logSummaryBlockLevel = logging.DEBUG
    logMainAlternatingECMIterations = bool(logPrimaryRole and logDeepDetails)

    blockCount = int(np.ceil(intervalCount / float(blockLenIntervals)))
    processNoiseCalibrationPolicy = (
        PROCESS_NOISE_CALIBRATION_FIXED
        if initialProcessQArr is not None
        else processNoiseCalibrationMode
    )
    intervalToBlockMap = (
        np.arange(intervalCount, dtype=np.int32) // blockLenIntervals
    ).astype(np.int32)
    intervalToBlockMap[intervalToBlockMap >= blockCount] = blockCount - 1
    totalStart = time.perf_counter()
    _logAsciiBlock(
        "core start",
        (
            ("run label", logRunLabel),
            ("tracks", int(trackCount)),
            ("intervals", int(intervalCount)),
            ("blocks", int(blockCount)),
            ("ECM max iterations", int(ECM_fixedBackgroundIters)),
            ("outer passes", int(ECM_outerIters)),
            ("state model", stateModelMode),
            ("process noise calibration", processNoiseCalibrationPolicy),
            (
                "process active observations",
                int(qCalibrationSupport["activeObservationCount"]),
            ),
            (
                "process active adjacent transitions",
                int(qCalibrationSupport["activeAdjacentTransitionCount"]),
            ),
            ("process calibration skip reason", processCalibrationSkipReason or "none"),
            (
                "PUNC prior df",
                "method_of_moments" if puncUsePriorDfMoments else float(puncPriorDf),
            ),
            ("PUNC local window multiplier", float(puncLocalWindowMultiplier)),
            (
                "PUNC reliability weighted windows",
                bool(puncUseReliabilityWeightedWindows),
            ),
            (
                "PUNC process covariates",
                0 if processCovariatesArr is None else processCovariatesArr.shape[1],
            ),
            ("background model fit", bool(fitBackground)),
            ("nonnegative background", bool(useNonnegativeBackground)),
            (
                "negative background penalty",
                (
                    "disabled"
                    if backgroundNegativePenaltyMultiplierLocal is None
                    else float(backgroundNegativePenaltyMultiplierLocal)
                ),
            ),
        ),
        indentLevel=logIndentLevel,
        level=logCoreBlockLevel,
    )
    logger.info(
        "runConsenrich.core.start runLabel=%s tracks=%d intervals=%d blocks=%d "
        "ECM_fixedBackgroundIters=%d outerIters=%d stateModel=%s "
        "processNoiseCalibration=%s",
        logRunLabel,
        int(trackCount),
        int(intervalCount),
        int(blockCount),
        int(ECM_fixedBackgroundIters),
        int(ECM_outerIters),
        stateModelMode,
        processNoiseCalibrationPolicy,
    )
    logger.info(
        "precisionMultiplierBounds: obs=[%.6g, %.6g] proc=[%.6g, %.6g]",
        observationPrecisionMultiplierMin,
        observationPrecisionMultiplierMax,
        processPrecisionMultiplierMin,
        processPrecisionMultiplierMax,
    )

    _warnIfProcessKappaMinAllowsProfiledNonconvexity(
        kappaMin=float(processPrecisionMultiplierMin),
        kappaMax=float(processPrecisionMultiplierMax),
        robustTNu=ECM_robustTNu,
        stateDim=int(stateDim),
        processReweightingEnabled=bool(ECM_useProcessPrecisionReweighting),
    )

    def _padLevelStateArray(arr: np.ndarray) -> np.ndarray:
        arr_ = np.asarray(arr, dtype=np.float32)
        if arr_.ndim != 2 or arr_.shape[1] != 1:
            return arr_
        out = np.zeros((arr_.shape[0], 2), dtype=np.float32)
        out[:, 0] = arr_[:, 0]
        return out

    def _padLevelCovarArray(arr: np.ndarray) -> np.ndarray:
        arr_ = np.asarray(arr, dtype=np.float32)
        if arr_.ndim != 3 or arr_.shape[1:] != (1, 1):
            return arr_
        out = np.zeros((arr_.shape[0], 2, 2), dtype=np.float32)
        out[:, 0, 0] = arr_[:, 0, 0]
        return out

    # keep the transition matrix step-dependent while using an explicit base Q
    def buildMatrixF(deltaFLocal: float) -> np.ndarray:
        return constructMatrixF(float(deltaFLocal)).astype(np.float32, copy=False)

    qSeedDiagnostics: dict[str, Any] = {}

    def buildMatrixQ0(deltaFLocal: float) -> np.ndarray:
        nonlocal qSeedDiagnostics
        matrixQ0Estimated, qSeedDiagnostics = _estimateInitialProcessNoiseFromData(
            matrixData=matrixData,
            matrixMunc=matrixMunc,
            pad=float(pad),
            stateModel=stateModelMode,
            minQ=float(minQ),
            maxQ=float(maxQ),
            deltaF=float(deltaFLocal),
            puncMaxScale=float(puncMaxScale),
            processNoiseCalibration=processNoiseCalibrationMode,
            robustTNu=ECM_robustTNu,
            qSeedPriorLevel=float(qSeedPriorLevel),
        )
        return matrixQ0Estimated

    def buildFixedDiagonalMatrixQ0() -> np.ndarray:
        return constructMatrixQ(
            minDiagQ=float(minQ),
            Q00=float(qPriorLevel),
            Q01=0.0,
            Q10=0.0,
            Q11=float(qPriorTrend),
        ).astype(np.float32, copy=False)

    def _runForwardBackward(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        lambdaExp: np.ndarray | None,
        processPrecExp: np.ndarray | None,
        processQScaleLocal: np.ndarray | None,
        useProcPrecReweightLocal: bool,
        useAPNLocal: bool,
    ):
        stateForward = np.empty((intervalCount, stateDim), dtype=np.float32)
        stateCovarForward = np.empty(
            (intervalCount, stateDim, stateDim),
            dtype=np.float32,
        )
        pNoiseForward = np.empty(
            (intervalCount, stateDim, stateDim),
            dtype=np.float32,
        )
        vectorD = np.empty(intervalCount, dtype=np.float32)

        if stateModelMode == STATE_MODEL_LEVEL:
            phiHat, _, vectorD, sumNLL = cconsenrich.cforwardPassLevel(
                matrixData=matrixDataLocal,
                matrixPluginMuncInit=matrixMuncLocal,
                matrixQ0=matrixQ0Local,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                pad=float(pad),
                chunkSize=0,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                vectorD=vectorD,
                returnNLL=True,
                storeNLLInD=False,
                lambdaExp=lambdaExp,
                processPrecExp=processPrecExp,
                processQScale=processQScaleLocal,
                ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
                ECM_useProcessPrecisionReweighting=bool(useProcPrecReweightLocal),
                ECM_useAPN=bool(useAPNLocal),
                obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
                obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
                procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                APN_minQ=float(minQ),
                APN_maxQ=float(maxQForAPN),
            )
            stateSmoothed, stateCovarSmoothed, lagCovSmoothed, postFitResiduals = (
                cconsenrich.cbackwardPassLevel(
                    matrixData=matrixDataLocal,
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
                returnNLL=True,
                storeNLLInD=False,
                lambdaExp=lambdaExp,
                processPrecExp=processPrecExp,
                processQScale=processQScaleLocal,
                ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
                ECM_useProcessPrecisionReweighting=bool(useProcPrecReweightLocal),
                ECM_useAPN=bool(useAPNLocal),
                obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
                obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
                procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                APN_minQ=float(minQ),
                APN_maxQ=float(maxQForAPN),
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
        processQScaleLocal: np.ndarray | None,
        useProcPrecReweightLocal: bool,
        useAPNLocal: bool,
        storeNLLInD: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        if stateModelMode == STATE_MODEL_LEVEL:
            _phiHat, _unused, _vectorD, sumNLL = cconsenrich.cforwardPassLevel(
                matrixData=matrixDataLocal,
                matrixPluginMuncInit=matrixMuncLocal,
                matrixQ0=matrixQ0Local,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                pad=float(pad),
                chunkSize=0,
                stateForward=None,
                stateCovarForward=None,
                pNoiseForward=None,
                vectorD=None,
                returnNLL=True,
                storeNLLInD=bool(storeNLLInD),
                lambdaExp=lambdaExp,
                processPrecExp=processPrecExp,
                processQScale=processQScaleLocal,
                ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
                ECM_useProcessPrecisionReweighting=bool(useProcPrecReweightLocal),
                ECM_useAPN=bool(useAPNLocal),
                obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
                obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
                procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                APN_minQ=float(minQ),
                APN_maxQ=float(maxQForAPN),
            )
        else:
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
                returnNLL=True,
                storeNLLInD=bool(storeNLLInD),
                lambdaExp=lambdaExp,
                processPrecExp=processPrecExp,
                processQScale=processQScaleLocal,
                ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
                ECM_useProcessPrecisionReweighting=bool(useProcPrecReweightLocal),
                ECM_useAPN=bool(useAPNLocal),
                obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
                obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
                procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                APN_minQ=float(minQ),
                APN_maxQ=float(maxQForAPN),
            )
        if storeNLLInD:
            return float(sumNLL), np.asarray(_vectorD, dtype=np.float64).copy()
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
        processQScaleLocal: np.ndarray | None,
        useProcPrecReweightLocal: bool,
        useAPNLocal: bool,
    ) -> dict[str, float]:
        def _backgroundNegativePenaltyForObjective(
            backgroundLocal: np.ndarray,
            objectiveWeightTrack: np.ndarray,
        ) -> float:
            if not negativeBackgroundPenaltyActive:
                return 0.0
            objectivePositiveWeights = objectiveWeightTrack[
                np.isfinite(objectiveWeightTrack) & (objectiveWeightTrack > 0.0)
            ]
            objectiveWeightScale = (
                float(np.median(objectivePositiveWeights))
                if objectivePositiveWeights.size
                else 1.0
            )
            if not np.isfinite(objectiveWeightScale) or objectiveWeightScale <= 0.0:
                objectiveWeightScale = 1.0
            objectivePenaltyWeight = float(
                backgroundNegativePenaltyMultiplierLocal * objectiveWeightScale
            )
            return (
                0.5
                * objectivePenaltyWeight
                * float(
                    np.sum(
                        np.minimum(
                            np.asarray(backgroundLocal, dtype=np.float64),
                            0.0,
                        )
                        ** 2,
                        dtype=np.float64,
                    )
                )
            )

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
            processQScaleLocal=processQScaleLocal,
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
        objectiveWeightTrack = np.zeros(intervalCount, dtype=np.float64)
        if lambdaExp is not None:
            objectiveObsPrecision = np.clip(
                np.asarray(lambdaExp, dtype=np.float64).reshape(-1),
                float(observationPrecisionMultiplierMin),
                float(observationPrecisionMultiplierMax),
            )
            if objectiveObsPrecision.shape != (intervalCount,):
                raise ValueError("lambdaExp length must match interval count")
        else:
            objectiveObsPrecision = None
        for rowIndex in range(int(matrixMuncLocal.shape[0])):
            invVarRow = 1.0 / np.maximum(
                np.asarray(matrixMuncLocal[rowIndex, :], dtype=np.float64) + float(pad),
                1.0e-8,
            )
            if objectiveObsPrecision is not None:
                invVarRow *= objectiveObsPrecision
            objectiveWeightTrack += invVarRow
        smoothPenalty, firstDiffPenalty, secondDiffPenalty = (
            _backgroundObjectivePenalty(
                background=background,
                blockLenIntervals=int(blockLenIntervals),
                backgroundSmoothness=float(ECM_backgroundSmoothness),
            )
        )
        negativePenalty = _backgroundNegativePenaltyForObjective(
            background,
            objectiveWeightTrack,
        )
        objective = float(
            forwardNLL + obsPenalty + procPenalty + smoothPenalty + negativePenalty
        )
        effectiveCount = float(_effectiveObservationCount(matrixMuncLocal))
        return {
            "forward_nll": float(forwardNLL),
            "robust_observation_penalty": float(obsPenalty),
            "robust_process_penalty": float(procPenalty),
            "background_smoothness_penalty": float(smoothPenalty),
            "background_first_difference_penalty": float(firstDiffPenalty),
            "background_second_difference_penalty": float(secondDiffPenalty),
            "background_negative_penalty": float(negativePenalty),
            "penalized_objective": float(objective),
            "penalized_objective_per_cell": float(objective / effectiveCount),
            "effective_observation_count": float(effectiveCount),
        }

    def _scoreBackgroundFitObjective(
        *,
        residualMatrix: np.ndarray,
        invVarMatrix: np.ndarray,
        background: np.ndarray,
    ) -> dict[str, float]:
        residualArr = np.asarray(residualMatrix, dtype=np.float64)
        invVarArr = np.asarray(invVarMatrix, dtype=np.float64)
        backgroundArr = np.asarray(background, dtype=np.float64).reshape(-1)
        fitResidual = residualArr - backgroundArr[None, :]
        weightedResidualObjective = 0.5 * float(
            np.sum(invVarArr * fitResidual * fitResidual, dtype=np.float64)
        )
        smoothPenalty, firstDiffPenalty, secondDiffPenalty = (
            _backgroundObjectivePenalty(
                background=backgroundArr,
                blockLenIntervals=int(blockLenIntervals),
                backgroundSmoothness=float(ECM_backgroundSmoothness),
            )
        )
        negativePenalty = 0.0
        if negativeBackgroundPenaltyActive:
            objectiveWeightTrack = np.sum(invVarArr, axis=0, dtype=np.float64)
            objectivePositiveWeights = objectiveWeightTrack[
                np.isfinite(objectiveWeightTrack) & (objectiveWeightTrack > 0.0)
            ]
            objectiveWeightScale = (
                float(np.median(objectivePositiveWeights))
                if objectivePositiveWeights.size
                else 1.0
            )
            if not np.isfinite(objectiveWeightScale) or objectiveWeightScale <= 0.0:
                objectiveWeightScale = 1.0
            objectivePenaltyWeight = float(
                backgroundNegativePenaltyMultiplierLocal * objectiveWeightScale
            )
            negativePenalty = (
                0.5
                * objectivePenaltyWeight
                * float(
                    np.sum(
                        np.minimum(backgroundArr, 0.0) ** 2,
                        dtype=np.float64,
                    )
                )
            )
        objective = float(weightedResidualObjective + smoothPenalty + negativePenalty)
        effectiveCount = float(
            max(
                1,
                np.count_nonzero(
                    np.isfinite(residualArr)
                    & np.isfinite(invVarArr)
                    & (invVarArr > 0.0)
                ),
            )
        )
        return {
            "background_weighted_residual_objective": weightedResidualObjective,
            "background_smoothness_penalty": float(smoothPenalty),
            "background_first_difference_penalty": float(firstDiffPenalty),
            "background_second_difference_penalty": float(secondDiffPenalty),
            "background_negative_penalty": float(negativePenalty),
            "background_objective": float(objective),
            "background_objective_per_cell": float(objective / effectiveCount),
            "background_effective_observation_count": float(effectiveCount),
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
        initialLambdaLocal: np.ndarray | None = None,
        initialProcessPrecLocal: np.ndarray | None = None,
        processQScaleLocal: np.ndarray | None = None,
        phaseLabel: str = "fit",
        phaseIndentLevel: int = 0,
        logAlternatingECMIterations: bool = False,
        showNonAlternatingECMProgress: bool = True,
    ) -> dict[str, np.ndarray | float | None]:
        mLocal = int(matrixDataLocal.shape[0])
        nLocal = int(matrixDataLocal.shape[1])
        fitBackgroundLocal = bool(fitBackground)
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
        processQScaleLocal = (
            None
            if processQScaleLocal is None
            else np.ascontiguousarray(processQScaleLocal, dtype=np.float32).reshape(-1)
        )
        if processQScaleLocal is not None and processQScaleLocal.shape[0] != nLocal:
            raise ValueError("processQScale length must match interval count")
        if processQScaleLocal is not None and nLocal:
            processQScaleLocal = processQScaleLocal.copy()
            processQScaleLocal[0] = 1.0
        backgroundPrepassApplied = False
        backgroundPrepassSource = ""
        if fitBackgroundLocal and initialBackgroundLocal is None:
            currentBackground, backgroundPrepassDiagnostics = (
                _estimateBackgroundWarmStart(
                    matrixData=matrixDataLocal,
                    matrixMunc=currentMunc,
                    blockLenIntervals=int(blockLenIntervals),
                    pad=float(pad),
                    observationPrecision=lambdaExpLocal,
                    observationPrecisionMultiplierMin=float(
                        observationPrecisionMultiplierMin
                    ),
                    observationPrecisionMultiplierMax=float(
                        observationPrecisionMultiplierMax
                    ),
                    backgroundSmoothness=float(ECM_backgroundSmoothness),
                    zeroCenterBackground=bool(ECM_zeroCenterBackground),
                    useNonnegativeBackground=bool(useNonnegativeBackground),
                    backgroundNegativePenaltyMultiplier=(
                        backgroundNegativePenaltyMultiplierLocal
                    ),
                    phaseLabel=phaseLabel,
                    logSummary=True,
                )
            )
            backgroundPrepassSource = str(backgroundPrepassDiagnostics["source"])
            backgroundPrepassApplied = True
        warmStartSummaryLocal = {
            "background": bool(initialBackgroundLocal is not None),
            "background_prepass": bool(backgroundPrepassApplied),
            "background_prepass_source": backgroundPrepassSource,
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
        previousBackgroundObjectivePerCellLocal = np.nan
        lastBackgroundObjectiveLocal = np.nan
        lastBackgroundObjectivePerCellLocal = np.nan
        lastBackgroundObjectiveChangePerCellLocal = np.nan
        lastBackgroundObjectiveTolPerCellLocal = np.nan
        lastBackgroundObjectiveStableLocal = False
        lastBackgroundObjectiveDiagnosticsLocal: dict[str, float] = {}
        lastObjectiveDiagnosticsLocal: dict[str, float] = {}
        lastInnerECMConvergedLocal = False
        outerStableItersLocal = 0
        outerPatienceTargetLocal = 2
        lastBackgroundShiftTolLocal = np.nan
        lambdaLowerBoundHitsLocal = None
        lambdaUpperBoundHitsLocal = None
        kappaLowerBoundHitsLocal = None
        kappaUpperBoundHitsLocal = None
        lastRelativeSignChangePerKBLocal = None
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
                processQScaleLocal=processQScaleLocal,
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
                    "outer_background_negative_penalty": metadataFloat(
                        lastObjectiveDiagnosticsLocal["background_negative_penalty"]
                    ),
                }
            )

        for outerPassIndex in range(outerPassCount):
            _logAsciiBlock(
                f"{phaseLabel} / fixed-g ECM pass",
                (
                    ("run label", logRunLabel),
                    ("fit phase", phaseLabel),
                    (
                        "outer pass",
                        f"{int(outerPassIndex + 1)}/{int(outerPassCount)}",
                    ),
                    ("tracks", int(mLocal)),
                    ("intervals", int(nLocal)),
                    ("ECM max iterations", int(ecmItersLocal)),
                    ("ECM rtol", float(ecmRtolLocal)),
                    ("background model fit", bool(fitBackgroundLocal)),
                    ("APN enabled", bool(useAPNLocal)),
                    ("obs precision weights", bool(ECM_useObsPrecisionReweighting)),
                    ("proc precision weights", bool(useProcPrecLocal)),
                ),
                indentLevel=phaseIndentLevel + 1,
                level=logging.DEBUG,
            )
            ecmLogIterations = bool(logAlternatingECMIterations)
            ecmResultLocal = _runFixedBackgroundECMPhase(
                matrixDataLocal=matrixDataLocal,
                currentBackground=currentBackground,
                currentMunc=currentMunc,
                matrixQ0Local=matrixQ0Local,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                ecmItersLocal=int(ecmItersLocal),
                ecmRtolLocal=float(ecmRtolLocal),
                t_innerItersLocal=int(t_innerIters),
                pad=float(pad),
                ECM_robustTNu=float(ECM_robustTNu),
                ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
                useProcPrecLocal=bool(useProcPrecLocal),
                useAPNLocal=bool(useAPNLocal),
                observationPrecisionMultiplierMin=float(
                    observationPrecisionMultiplierMin
                ),
                observationPrecisionMultiplierMax=float(
                    observationPrecisionMultiplierMax
                ),
                processPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                processPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                minQ=float(minQ),
                maxQForAPN=float(maxQForAPN),
                lambdaExpLocal=lambdaExpLocal,
                processPrecExpLocal=processPrecExpLocal,
                processQScaleLocal=processQScaleLocal,
                trackOptimizationPath=bool(trackOptimizationPath),
                logIterations=ecmLogIterations,
                stateModelMode=stateModelMode,
                matrixFLocal=matrixFLocal,
            )
            ecmDiagnosticsLocal = ecmResultLocal.diagnostics
            ecmItersDoneLocal = ecmResultLocal.iters_done
            nllECMLocal = ecmResultLocal.nll
            stateSmoothedLocal = ecmResultLocal.state_smoothed
            stateCovarSmoothedLocal = ecmResultLocal.state_covar_smoothed
            lagCovSmoothedLocal = ecmResultLocal.lag_covar_smoothed
            postFitResidualsLocal = ecmResultLocal.post_fit_residuals
            lambdaExpLocal = ecmResultLocal.lambda_exp
            processPrecExpLocal = ecmResultLocal.process_prec_exp
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
            kappaMeanLocal, kappaMedianLocal = _processKappaSummary(
                processPrecExpLocal,
                lower=float(processPrecisionMultiplierMin),
                upper=float(processPrecisionMultiplierMax),
            )
            lambdaLowerBoundHitsLocal, lambdaUpperBoundHitsLocal = _precisionBoundHits(
                lambdaExpLocal,
                lower=float(observationPrecisionMultiplierMin),
                upper=float(observationPrecisionMultiplierMax),
            )
            kappaLowerBoundHitsLocal, kappaUpperBoundHitsLocal = _precisionBoundHits(
                processPrecExpLocal,
                lower=float(processPrecisionMultiplierMin),
                upper=float(processPrecisionMultiplierMax),
                skipFirst=True,
            )
            stateSmoothedLocal = np.asarray(stateSmoothedLocal, dtype=np.float32)
            lastRelativeSignChangePerKBLocal = _relativeSignChangePerKB(
                stateSmoothedLocal[:, 0],
                matrixDataLocal,
                currentMunc,
                intervalSizeBP=intervalSizeBPLocal,
                background=currentBackground,
                pad=float(pad),
            )
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
                ecmDiagnosticsNormalized["observation_lambda_median"] = (
                    lambdaMedianLocal
                )
                ecmDiagnosticsNormalized["process_kappa_mean"] = kappaMeanLocal
                ecmDiagnosticsNormalized["process_kappa_median"] = kappaMedianLocal
                ecmDiagnosticsNormalized["observation_lambda_lower_bound_hits"] = (
                    lambdaLowerBoundHitsLocal
                )
                ecmDiagnosticsNormalized["observation_lambda_upper_bound_hits"] = (
                    lambdaUpperBoundHitsLocal
                )
                ecmDiagnosticsNormalized["process_kappa_lower_bound_hits"] = (
                    kappaLowerBoundHitsLocal
                )
                ecmDiagnosticsNormalized["process_kappa_upper_bound_hits"] = (
                    kappaUpperBoundHitsLocal
                )
                ecmDiagnosticsNormalized["relative_sign_change_per_kb"] = (
                    lastRelativeSignChangePerKBLocal
                )
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
                logger.log(
                    logging.INFO if logPrimaryRole else logging.DEBUG,
                    "outerPass[1/1] phase=%s runLabel=%s: "
                    "fitBackground=False backgroundShift=0 "
                    "lambdaMean=%s lambdaMedian=%s relativeSignChangePerKB=%s "
                    "lambdaLowerBoundHits=%s lambdaUpperBoundHits=%s "
                    "kappaMean=%s kappaMedian=%s kappaLowerBoundHits=%s "
                    "kappaUpperBoundHits=%s outerObjectiveChangePerCell=%s",
                    phaseLabel,
                    logRunLabel,
                    _formatMaybeFloat(lambdaMeanLocal),
                    _formatMaybeFloat(lambdaMedianLocal),
                    _formatMaybeFloat(lastRelativeSignChangePerKBLocal),
                    _formatMaybeFloat(lambdaLowerBoundHitsLocal),
                    _formatMaybeFloat(lambdaUpperBoundHitsLocal),
                    _formatMaybeFloat(kappaMeanLocal),
                    _formatMaybeFloat(kappaMedianLocal),
                    _formatMaybeFloat(kappaLowerBoundHitsLocal),
                    _formatMaybeFloat(kappaUpperBoundHitsLocal),
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
            residualMatrix = np.asarray(matrixDataLocal, dtype=np.float32) - np.asarray(
                stateSmoothedLocal[:, 0][None, :], dtype=np.float32
            )
            backgroundWeightTrack = np.sum(invVarMatrix, axis=0, dtype=np.float64)
            backgroundRhsTrack = np.einsum(
                "ij,ij->j",
                invVarMatrix,
                residualMatrix,
                dtype=np.float64,
            )
            backgroundValidTarget = backgroundWeightTrack > 0.0
            if np.any(backgroundValidTarget):
                backgroundTarget = (
                    backgroundRhsTrack[backgroundValidTarget]
                    / backgroundWeightTrack[backgroundValidTarget]
                )
                backgroundTarget = backgroundTarget[np.isfinite(backgroundTarget)]
            else:
                backgroundTarget = np.zeros(0, dtype=np.float64)
            if backgroundTarget.size > 0:
                targetQuantiles = np.quantile(
                    backgroundTarget,
                    [0.05, 0.5, 0.95],
                )
                logger.debug(
                    "backgroundTarget[%s pass=%d/%d]: valid=%d/%d "
                    "min=%.6g p05=%.6g median=%.6g mean=%.6g "
                    "p95=%.6g max=%.6g fracPositive=%.4f fracAbsLe1e-3=%.4f",
                    phaseLabel,
                    int(outerPassIndex + 1),
                    int(outerPassCount),
                    int(backgroundTarget.size),
                    int(intervalCount),
                    float(np.min(backgroundTarget)),
                    float(targetQuantiles[0]),
                    float(targetQuantiles[1]),
                    float(np.mean(backgroundTarget, dtype=np.float64)),
                    float(targetQuantiles[2]),
                    float(np.max(backgroundTarget)),
                    float(np.mean(backgroundTarget > 0.0)),
                    float(np.mean(np.abs(backgroundTarget) <= 1.0e-3)),
                )
            else:
                logger.debug(
                    "backgroundTarget[%s pass=%d/%d]: valid=0/%d",
                    phaseLabel,
                    int(outerPassIndex + 1),
                    int(outerPassCount),
                    int(intervalCount),
                )
            nextBackground = solveZeroCenteredBackground(
                residualMatrix=residualMatrix,
                invVarMatrix=invVarMatrix,
                blockLenIntervals=int(blockLenIntervals),
                backgroundSmoothness=float(ECM_backgroundSmoothness),
                zeroCenter=bool(ECM_zeroCenterBackground),
                useNonnegative=bool(useNonnegativeBackground),
                backgroundNegativePenaltyMultiplier=(
                    backgroundNegativePenaltyMultiplierLocal
                ),
                initialBackground=currentBackground,
                weightTrack=backgroundWeightTrack,
                rhsTrack=backgroundRhsTrack,
            )
            nextBackgroundSummary = np.asarray(nextBackground, dtype=np.float64)
            if nextBackgroundSummary.size > 0:
                nextBackgroundQuantiles = np.quantile(
                    nextBackgroundSummary,
                    [0.05, 0.5, 0.95],
                )
                logger.debug(
                    "backgroundSolve[%s pass=%d/%d]: "
                    "min=%.6g p05=%.6g median=%.6g mean=%.6g "
                    "p95=%.6g max=%.6g fracPositive=%.4f fracAbsLe1e-3=%.4f",
                    phaseLabel,
                    int(outerPassIndex + 1),
                    int(outerPassCount),
                    float(np.min(nextBackgroundSummary)),
                    float(nextBackgroundQuantiles[0]),
                    float(nextBackgroundQuantiles[1]),
                    float(np.mean(nextBackgroundSummary, dtype=np.float64)),
                    float(nextBackgroundQuantiles[2]),
                    float(np.max(nextBackgroundSummary)),
                    float(np.mean(nextBackgroundSummary > 0.0)),
                    float(np.mean(np.abs(nextBackgroundSummary) <= 1.0e-3)),
                )

            lastBackgroundObjectiveDiagnosticsLocal = _scoreBackgroundFitObjective(
                residualMatrix=residualMatrix,
                invVarMatrix=invVarMatrix,
                background=nextBackground,
            )
            currentBackgroundObjective = float(
                lastBackgroundObjectiveDiagnosticsLocal["background_objective"]
            )
            currentBackgroundObjectivePerCell = float(
                lastBackgroundObjectiveDiagnosticsLocal["background_objective_per_cell"]
            )
            if np.isfinite(previousBackgroundObjectivePerCellLocal) and np.isfinite(
                currentBackgroundObjectivePerCell
            ):
                lastBackgroundObjectiveChangePerCellLocal = abs(
                    currentBackgroundObjectivePerCell
                    - previousBackgroundObjectivePerCellLocal
                )
                lastBackgroundObjectiveTolPerCellLocal = (
                    outerObjectiveTolMultiplier
                    * max(
                        abs(currentBackgroundObjectivePerCell),
                        abs(previousBackgroundObjectivePerCellLocal),
                        1.0,
                    )
                )
                lastBackgroundObjectiveStableLocal = bool(
                    lastBackgroundObjectiveChangePerCellLocal
                    <= lastBackgroundObjectiveTolPerCellLocal
                )
            else:
                lastBackgroundObjectiveChangePerCellLocal = np.nan
                lastBackgroundObjectiveTolPerCellLocal = np.nan
                lastBackgroundObjectiveStableLocal = False
            previousBackgroundObjectivePerCellLocal = currentBackgroundObjectivePerCell
            lastBackgroundObjectiveLocal = currentBackgroundObjective
            lastBackgroundObjectivePerCellLocal = currentBackgroundObjectivePerCell

            shiftWeights = np.asarray(
                backgroundWeightTrack,
                dtype=np.float64,
            )
            shiftWeightSum = float(np.sum(shiftWeights, dtype=np.float64))
            if shiftWeightSum <= 0.0:
                raise ValueError("shift RMS requires positive weights")
            proposalG = np.asarray(nextBackground, dtype=np.float64)
            referenceG = np.asarray(currentBackground, dtype=np.float64)
            shiftDelta = proposalG - referenceG
            bgChange = float(
                np.sqrt(
                    float(
                        np.dot(
                            shiftWeights,
                            shiftDelta * shiftDelta,
                        )
                    )
                    / shiftWeightSum
                )
            )
            proposalRMS = float(
                np.sqrt(
                    float(
                        np.dot(
                            shiftWeights,
                            proposalG * proposalG,
                        )
                    )
                    / shiftWeightSum
                )
            )
            referenceRMS = float(
                np.sqrt(
                    float(
                        np.dot(
                            shiftWeights,
                            referenceG * referenceG,
                        )
                    )
                    / shiftWeightSum
                )
            )
            bgScale = float(max(proposalRMS, referenceRMS, 1.0))
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
            ecmDiagnosticsNormalized["background_objective"] = metadataFloat(
                lastBackgroundObjectiveLocal
            )
            ecmDiagnosticsNormalized["background_objective_per_cell"] = metadataFloat(
                lastBackgroundObjectivePerCellLocal
            )
            ecmDiagnosticsNormalized["background_objective_change_per_cell"] = (
                metadataFloat(lastBackgroundObjectiveChangePerCellLocal)
            )
            ecmDiagnosticsNormalized["background_objective_threshold_per_cell"] = (
                metadataFloat(lastBackgroundObjectiveTolPerCellLocal)
            )
            ecmDiagnosticsNormalized["background_objective_stable"] = bool(
                lastBackgroundObjectiveStableLocal
            )
            ecmDiagnosticsNormalized["background_weighted_residual_objective"] = (
                metadataFloat(
                    lastBackgroundObjectiveDiagnosticsLocal.get(
                        "background_weighted_residual_objective",
                        np.nan,
                    )
                )
            )
            ecmDiagnosticsNormalized["background_fit_effective_observation_count"] = (
                int(
                    lastBackgroundObjectiveDiagnosticsLocal.get(
                        "background_effective_observation_count",
                        0,
                    )
                    or 0
                )
            )
            ecmDiagnosticsNormalized["observation_lambda_mean"] = lambdaMeanLocal
            ecmDiagnosticsNormalized["observation_lambda_median"] = lambdaMedianLocal
            ecmDiagnosticsNormalized["process_kappa_mean"] = kappaMeanLocal
            ecmDiagnosticsNormalized["process_kappa_median"] = kappaMedianLocal
            ecmDiagnosticsNormalized["observation_lambda_lower_bound_hits"] = (
                lambdaLowerBoundHitsLocal
            )
            ecmDiagnosticsNormalized["observation_lambda_upper_bound_hits"] = (
                lambdaUpperBoundHitsLocal
            )
            ecmDiagnosticsNormalized["process_kappa_lower_bound_hits"] = (
                kappaLowerBoundHitsLocal
            )
            ecmDiagnosticsNormalized["process_kappa_upper_bound_hits"] = (
                kappaUpperBoundHitsLocal
            )
            ecmDiagnosticsNormalized["relative_sign_change_per_kb"] = (
                lastRelativeSignChangePerKBLocal
            )
            ecmDiagnosticsNormalized["outer_inner_ecm_converged"] = bool(
                lastInnerECMConvergedLocal
            )
            ecmDiagnosticsNormalized["outer_stable_iters"] = int(outerStableItersLocal)
            ecmDiagnosticsNormalized["outer_patience_target"] = int(
                outerPatienceTargetLocal
            )
            fixedBackgroundECMDiagnostics.append(ecmDiagnosticsNormalized)
            logger.log(
                (
                    logging.INFO
                    if logPrimaryRole and "post-process-noise fit" in phaseLabel
                    else logging.DEBUG
                ),
                "outerPass[%d/%d] phase=%s runLabel=%s: backgroundShift=%.6g "
                "backgroundShiftThreshold=%.6g backgroundObjectivePerCell=%s "
                "backgroundObjectiveChangePerCell=%s "
                "backgroundObjectiveThresholdPerCell=%s lambdaMean=%s "
                "lambdaMedian=%s relativeSignChangePerKB=%s "
                "lambdaLowerBoundHits=%s "
                "lambdaUpperBoundHits=%s kappaMean=%s kappaMedian=%s "
                "kappaLowerBoundHits=%s kappaUpperBoundHits=%s "
                "outerObjectivePerCell=%s outerObjectiveChangePerCell=%s "
                "outerObjectiveThresholdPerCell=%s outerStable=%d/%d "
                "innerECMConverged=%s",
                int(outerPassIndex + 1),
                int(outerPassCount),
                phaseLabel,
                logRunLabel,
                float(bgChange),
                float(bgTol),
                _formatMaybeFloat(lastBackgroundObjectivePerCellLocal),
                _formatMaybeFloat(lastBackgroundObjectiveChangePerCellLocal),
                _formatMaybeFloat(lastBackgroundObjectiveTolPerCellLocal),
                _formatMaybeFloat(lambdaMeanLocal),
                _formatMaybeFloat(lambdaMedianLocal),
                _formatMaybeFloat(lastRelativeSignChangePerKBLocal),
                _formatMaybeFloat(lambdaLowerBoundHitsLocal),
                _formatMaybeFloat(lambdaUpperBoundHitsLocal),
                _formatMaybeFloat(kappaMeanLocal),
                _formatMaybeFloat(kappaMedianLocal),
                _formatMaybeFloat(kappaLowerBoundHitsLocal),
                _formatMaybeFloat(kappaUpperBoundHitsLocal),
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

        if fitBackgroundLocal:
            _logAsciiBlock(
                f"{phaseLabel} / final fixed-g ECM",
                (
                    ("run label", logRunLabel),
                    ("fit phase", phaseLabel),
                    ("tracks", int(mLocal)),
                    ("intervals", int(nLocal)),
                    ("ECM max iterations", int(ecmItersLocal)),
                    ("ECM rtol", float(ecmRtolLocal)),
                    ("background model fit", False),
                    ("APN enabled", bool(useAPNLocal)),
                    ("obs precision weights", bool(ECM_useObsPrecisionReweighting)),
                    ("proc precision weights", bool(useProcPrecLocal)),
                ),
                indentLevel=phaseIndentLevel + 1,
                level=logging.DEBUG,
            )
            ecmResultLocal = _runFixedBackgroundECMPhase(
                matrixDataLocal=matrixDataLocal,
                currentBackground=currentBackground,
                currentMunc=currentMunc,
                matrixQ0Local=matrixQ0Local,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                ecmItersLocal=int(ecmItersLocal),
                ecmRtolLocal=float(ecmRtolLocal),
                t_innerItersLocal=int(t_innerIters),
                pad=float(pad),
                ECM_robustTNu=float(ECM_robustTNu),
                ECM_useObsPrecisionReweighting=bool(ECM_useObsPrecisionReweighting),
                useProcPrecLocal=bool(useProcPrecLocal),
                useAPNLocal=bool(useAPNLocal),
                observationPrecisionMultiplierMin=float(
                    observationPrecisionMultiplierMin
                ),
                observationPrecisionMultiplierMax=float(
                    observationPrecisionMultiplierMax
                ),
                processPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                processPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                minQ=float(minQ),
                maxQForAPN=float(maxQForAPN),
                lambdaExpLocal=lambdaExpLocal,
                processPrecExpLocal=processPrecExpLocal,
                processQScaleLocal=processQScaleLocal,
                trackOptimizationPath=bool(trackOptimizationPath),
                logIterations=False,
                stateModelMode=stateModelMode,
                matrixFLocal=matrixFLocal,
            )
            ecmDiagnosticsLocal = ecmResultLocal.diagnostics
            ecmItersDoneLocal = ecmResultLocal.iters_done
            nllECMLocal = ecmResultLocal.nll
            stateSmoothedLocal = ecmResultLocal.state_smoothed
            stateCovarSmoothedLocal = ecmResultLocal.state_covar_smoothed
            lagCovSmoothedLocal = ecmResultLocal.lag_covar_smoothed
            postFitResidualsLocal = ecmResultLocal.post_fit_residuals
            lambdaExpLocal = ecmResultLocal.lambda_exp
            processPrecExpLocal = ecmResultLocal.process_prec_exp
            currentECMNLLLocal = float(nllECMLocal)
            finalECMDiagnosticsNormalized = _normalizeFixedBackgroundECMDiagnostics(
                ecmDiagnosticsLocal,
                itersDone=int(ecmItersDoneLocal),
                finalNLL=currentECMNLLLocal,
                maxIters=int(ecmItersLocal),
                outerPass=int(actualOuterPasses + 1),
            )
            finalECMDiagnosticsNormalized["final_fixed_background_ecm"] = True
            lastInnerECMConvergedLocal = bool(
                finalECMDiagnosticsNormalized.get("converged") is True
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
            kappaMeanLocal, kappaMedianLocal = _processKappaSummary(
                processPrecExpLocal,
                lower=float(processPrecisionMultiplierMin),
                upper=float(processPrecisionMultiplierMax),
            )
            lambdaLowerBoundHitsLocal, lambdaUpperBoundHitsLocal = _precisionBoundHits(
                lambdaExpLocal,
                lower=float(observationPrecisionMultiplierMin),
                upper=float(observationPrecisionMultiplierMax),
            )
            kappaLowerBoundHitsLocal, kappaUpperBoundHitsLocal = _precisionBoundHits(
                processPrecExpLocal,
                lower=float(processPrecisionMultiplierMin),
                upper=float(processPrecisionMultiplierMax),
                skipFirst=True,
            )
            stateSmoothedLocal = np.asarray(stateSmoothedLocal, dtype=np.float32)
            lastRelativeSignChangePerKBLocal = _relativeSignChangePerKB(
                stateSmoothedLocal[:, 0],
                matrixDataLocal,
                currentMunc,
                intervalSizeBP=intervalSizeBPLocal,
                background=currentBackground,
                pad=float(pad),
            )
            stateCovarSmoothedLocal = np.asarray(
                stateCovarSmoothedLocal,
                dtype=np.float32,
            )
            lagCovSmoothedLocal = np.asarray(lagCovSmoothedLocal, dtype=np.float32)
            postFitResidualsLocal = np.asarray(postFitResidualsLocal, dtype=np.float32)
            finalECMDiagnosticsNormalized["observation_lambda_mean"] = lambdaMeanLocal
            finalECMDiagnosticsNormalized["observation_lambda_median"] = (
                lambdaMedianLocal
            )
            finalECMDiagnosticsNormalized["process_kappa_mean"] = kappaMeanLocal
            finalECMDiagnosticsNormalized["process_kappa_median"] = kappaMedianLocal
            finalECMDiagnosticsNormalized["observation_lambda_lower_bound_hits"] = (
                lambdaLowerBoundHitsLocal
            )
            finalECMDiagnosticsNormalized["observation_lambda_upper_bound_hits"] = (
                lambdaUpperBoundHitsLocal
            )
            finalECMDiagnosticsNormalized["process_kappa_lower_bound_hits"] = (
                kappaLowerBoundHitsLocal
            )
            finalECMDiagnosticsNormalized["process_kappa_upper_bound_hits"] = (
                kappaUpperBoundHitsLocal
            )
            finalECMDiagnosticsNormalized["relative_sign_change_per_kb"] = (
                lastRelativeSignChangePerKBLocal
            )
            fixedBackgroundECMDiagnostics.append(finalECMDiagnosticsNormalized)
            logger.log(
                (
                    logging.INFO
                    if logPrimaryRole and "post-process-noise fit" in phaseLabel
                    else logging.DEBUG
                ),
                "finalFixedBackgroundECM[%s] runLabel=%s: iters=%d converged=%s "
                "stable=%d/%d fixedBackgroundAbsRelChange=%s "
                "fixedBackgroundRtol=%s relativeSignChangePerKB=%s "
                "lambdaMean=%s lambdaMedian=%s lambdaLowerBoundHits=%s "
                "lambdaUpperBoundHits=%s kappaMean=%s kappaMedian=%s "
                "kappaLowerBoundHits=%s kappaUpperBoundHits=%s",
                phaseLabel,
                logRunLabel,
                int(ecmItersDoneLocal),
                str(bool(lastInnerECMConvergedLocal)),
                int(finalECMDiagnosticsNormalized.get("stable_iters", 0) or 0),
                int(finalECMDiagnosticsNormalized.get("patience_target", 0) or 0),
                _formatMaybeFloat(
                    finalECMDiagnosticsNormalized.get("final_abs_rel_change")
                ),
                _formatMaybeFloat(ecmRtolLocal),
                _formatMaybeFloat(lastRelativeSignChangePerKBLocal),
                _formatMaybeFloat(lambdaMeanLocal),
                _formatMaybeFloat(lambdaMedianLocal),
                _formatMaybeFloat(lambdaLowerBoundHitsLocal),
                _formatMaybeFloat(lambdaUpperBoundHitsLocal),
                _formatMaybeFloat(kappaMeanLocal),
                _formatMaybeFloat(kappaMedianLocal),
                _formatMaybeFloat(kappaLowerBoundHitsLocal),
                _formatMaybeFloat(kappaUpperBoundHitsLocal),
            )

        dataAdjusted = np.ascontiguousarray(
            matrixDataLocal - currentBackground[None, :],
            dtype=np.float32,
        )
        _logAsciiBlock(
            f"{phaseLabel} / forward-backward scoring",
            (
                ("run label", logRunLabel),
                ("fit phase", phaseLabel),
                ("tracks", int(mLocal)),
                ("intervals", int(nLocal)),
                ("background model", bool(fitBackgroundLocal)),
                ("obs precision weights", bool(lambdaExpLocal is not None)),
                ("proc precision weights", bool(processPrecExpLocal is not None)),
            ),
            indentLevel=phaseIndentLevel + 1,
            level=logging.DEBUG,
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
            processQScaleLocal=processQScaleLocal,
            useProcPrecReweightLocal=useProcPrecLocal,
            useAPNLocal=useAPNLocal,
        )
        return {
            "matrixMunc": currentMunc,
            "background": currentBackground,
            "lambdaExp": lambdaExpLocal,
            "processPrecExp": processPrecExpLocal,
            "processQScale": processQScaleLocal,
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
            "backgroundObjective": metadataFloat(lastBackgroundObjectiveLocal),
            "backgroundObjectivePerCell": metadataFloat(
                lastBackgroundObjectivePerCellLocal
            ),
            "backgroundObjectiveChangePerCell": metadataFloat(
                lastBackgroundObjectiveChangePerCellLocal
            ),
            "backgroundObjectiveThresholdPerCell": metadataFloat(
                lastBackgroundObjectiveTolPerCellLocal
            ),
            "backgroundObjectiveStable": bool(lastBackgroundObjectiveStableLocal),
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
            "lambdaLowerBoundHits": lambdaLowerBoundHitsLocal,
            "lambdaUpperBoundHits": lambdaUpperBoundHitsLocal,
            "kappaLowerBoundHits": kappaLowerBoundHitsLocal,
            "kappaUpperBoundHits": kappaUpperBoundHitsLocal,
            "relativeSignChangePerKB": lastRelativeSignChangePerKBLocal,
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

    deltaF_fit = (
        1.0 if stateModelMode == STATE_MODEL_LEVEL else _resolveFixedDeltaF(deltaF)
    )

    matrixF = buildMatrixF(float(deltaF_fit))
    if initialProcessQArr is not None:
        matrixQ0 = np.ascontiguousarray(initialProcessQArr, dtype=np.float32).copy()
    elif processNoiseCalibrationMode == PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL:
        matrixQ0 = buildFixedDiagonalMatrixQ0()
    else:
        matrixQ0 = buildMatrixQ0(float(deltaF_fit))
    matrixQ0 = _clampProcessNoiseMatrix(
        matrixQ0,
        stateModel=stateModelMode,
        minQ=float(minQ),
        maxQ=float(maxQ),
    )
    fitProcessNoiseWarmup: Mapping[str, Any] | None = None
    fitProcessNoiseWarmupMetadata: dict[str, Any] | None = None
    processNoiseCalibrationInfo: dict[str, Any] | None = None
    postQInitialBackground = initialBackgroundArr
    postQInitialLambda = initialObservationPrecisionArr
    postQInitialProcessPrec = initialProcessPrecisionArr

    processQScaleFinal = np.ones(intervalCount, dtype=np.float32)
    if initialProcessQArr is not None:
        processNoiseCalibrationInfo = _staticProcessNoiseCalibrationDiagnostics(
            processNoisePolicy=PROCESS_NOISE_CALIBRATION_FIXED,
            status="skipped",
            reason="initial_process_q",
            matrixQ0=matrixQ0,
            stateModel=stateModelMode,
            minQ=float(minQ),
            maxQ=float(maxQ),
            support=qCalibrationSupport,
            warmStartProcessNoise=1.0,
        )
    elif (
        processNoiseCalibrationMode
        in {
            PROCESS_NOISE_CALIBRATION_SEED,
            PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL,
            PROCESS_NOISE_CALIBRATION_FIXED,
        }
        or processCalibrationSkipReason is not None
    ):
        skipReason = (
            str(processCalibrationSkipReason)
            if processCalibrationSkipReason is not None
            else (
                "fixed_diagonal"
                if processNoiseCalibrationMode
                == PROCESS_NOISE_CALIBRATION_FIXED_DIAGONAL
                else processNoiseCalibrationMode
            )
        )
        processNoiseCalibrationInfo = _staticProcessNoiseCalibrationDiagnostics(
            processNoisePolicy=processNoiseCalibrationMode,
            status="skipped",
            reason=skipReason,
            matrixQ0=matrixQ0,
            stateModel=stateModelMode,
            minQ=float(minQ),
            maxQ=float(maxQ),
            support=qCalibrationSupport,
            warmStartProcessNoise=0.0,
        )
    else:
        warmupIters = int(processNoiseWarmupECMIters)
        warmupOuterIters = int(processNoiseWarmupOuterPasses)
        stageStart = time.perf_counter()
        if puncUseWarmupFit:
            _logAsciiBlock(
                "process noise warmup",
                (
                    ("run label", logRunLabel),
                    ("fit phase", "PUNC warmup fit"),
                    ("purpose", "PUNC transition evidence warmup"),
                    ("tracks", int(trackCount)),
                    ("intervals", int(intervalCount)),
                    ("outer passes", int(warmupOuterIters)),
                    ("ECM max iterations", int(warmupIters)),
                    ("proc precision weights", False),
                    ("APN enabled", False),
                ),
                indentLevel=logIndentLevel + 1,
                level=logCoreBlockLevel,
            )
            fitProcessNoiseWarmup = _fitOuter(
                matrixDataLocal=matrixData,
                matrixMuncLocal=matrixMunc,
                matrixFLocal=matrixF,
                matrixQ0Local=matrixQ0,
                ecmItersLocal=warmupIters,
                ecmRtolLocal=float(ECM_fixedBackgroundRtol),
                outerItersLocal=int(warmupOuterIters),
                minOuterItersLocal=1,
                useProcPrecReweightOverride=False,
                useAPNOverride=False,
                initialBackgroundLocal=postQInitialBackground,
                initialLambdaLocal=postQInitialLambda,
                phaseLabel="PUNC warmup fit",
                phaseIndentLevel=logIndentLevel + 1,
                logAlternatingECMIterations=False,
            )
            postQInitialBackground = np.asarray(
                fitProcessNoiseWarmup["background"],
                dtype=np.float32,
            )
            postQInitialLambda = (
                np.asarray(fitProcessNoiseWarmup["lambdaExp"], dtype=np.float32)
                if fitProcessNoiseWarmup.get("lambdaExp") is not None
                else initialObservationPrecisionArr
            )
        else:
            evidenceBackground = (
                np.zeros(intervalCount, dtype=np.float32)
                if postQInitialBackground is None
                else np.asarray(postQInitialBackground, dtype=np.float32)
            )
            adjustedEvidenceData = np.ascontiguousarray(
                matrixData - evidenceBackground[None, :],
                dtype=np.float32,
            )
            (
                _phiHatWarmup,
                warmupNLL,
                stateForwardWarmup,
                stateCovarForwardWarmup,
                pNoiseForwardWarmup,
                stateSmoothedWarmup,
                stateCovarSmoothedWarmup,
                lagCovSmoothedWarmup,
                postFitResidualsWarmup,
                warmupNIS,
            ) = _runForwardBackward(
                matrixDataLocal=adjustedEvidenceData,
                matrixMuncLocal=matrixMunc,
                matrixFLocal=matrixF,
                matrixQ0Local=matrixQ0,
                lambdaExp=postQInitialLambda,
                processPrecExp=None,
                processQScaleLocal=None,
                useProcPrecReweightLocal=False,
                useAPNLocal=False,
            )
            fitProcessNoiseWarmup = {
                "matrixMunc": matrixMunc,
                "background": evidenceBackground,
                "lambdaExp": postQInitialLambda,
                "processPrecExp": None,
                "stateForward": stateForwardWarmup,
                "stateCovarForward": stateCovarForwardWarmup,
                "pNoiseForward": pNoiseForwardWarmup,
                "stateSmoothed": stateSmoothedWarmup,
                "stateCovarSmoothed": stateCovarSmoothedWarmup,
                "lagCovSmoothed": lagCovSmoothedWarmup,
                "postFitResiduals": postFitResidualsWarmup,
                "NIS": warmupNIS,
                "sumNLL": float(warmupNLL),
            }
        matrixQ0, processQScaleFinal, processNoiseCalibrationInfo = (
            _fitPuncProcessNoise(
                warmupFit=fitProcessNoiseWarmup,
                matrixMunc=matrixMunc,
                matrixF=matrixF,
                seedQ=matrixQ0,
                stateModel=stateModelMode,
                pad=float(pad),
                minQ=float(minQ),
                maxQ=float(maxQ),
                blockLenIntervals=int(blockLenIntervals),
                processCovariates=processCovariatesArr,
                puncLocalWindowMultiplier=float(puncLocalWindowMultiplier),
                puncDependenceMultiplier=float(puncDependenceMultiplier),
                puncMinScale=float(puncMinScale),
                puncMaxScale=float(puncMaxScale),
                puncMinWindowWeight=float(puncMinWindowWeight),
                puncPriorDf=float(puncPriorDf),
                puncPriorRidge=float(puncPriorRidge),
                puncLevelBufferZ=float(puncLevelBufferZ),
                puncUseReliabilityWeightedWindows=bool(
                    puncUseReliabilityWeightedWindows
                ),
                puncUseWarmupFit=bool(puncUseWarmupFit),
                puncUseTransitionEvidence=bool(puncUseTransitionEvidence),
                puncUseScaleRebase=bool(puncUseScaleRebase),
                puncUseGlobalScale=bool(puncUseGlobalScale),
                puncUseBoundaryClamps=bool(puncUseBoundaryClamps),
                puncUsePriorDfMoments=bool(puncUsePriorDfMoments),
                puncUsePriorShrinkage=bool(puncUsePriorShrinkage),
                observationPrecisionMultiplierMin=float(
                    observationPrecisionMultiplierMin
                ),
                observationPrecisionMultiplierMax=float(
                    observationPrecisionMultiplierMax
                ),
            )
        )
        processNoiseCalibrationInfo["warmupElapsedSeconds"] = float(
            time.perf_counter() - stageStart
        )
        processNoiseCalibrationInfo["warmupECMIters"] = (
            float(warmupIters) if puncUseWarmupFit else 0.0
        )
        processNoiseCalibrationInfo["warmupOuterPasses"] = (
            float(warmupOuterIters) if puncUseWarmupFit else 0.0
        )
        fitProcessNoiseWarmupMetadata = _fitDiagnosticsMetadata(fitProcessNoiseWarmup)
        fitProcessNoiseWarmup = None
    if processNoiseCalibrationInfo is None:
        raise RuntimeError("process-noise calibration did not produce diagnostics")
    if qSeedDiagnostics:
        processNoiseCalibrationInfo.update(qSeedDiagnostics)
    processNoiseCalibrationInfo.update(qCalibrationSupport)
    processNoiseCalibrationInfo["resolvedMinQ"] = float(minQ)
    processNoiseCalibrationInfo["resolvedMaxQ"] = float(maxQForAPN)
    processNoiseCalibrationInfo["transitionCount"] = float(max(intervalCount - 1, 0))
    processNoiseCalibrationInfo["processQScaleSummary"] = _metadataTrackSummary(
        processQScaleFinal
    )
    _logEvent(
        "process_noise.calibration",
        (
            ("run label", logRunLabel),
            ("mode", processNoiseCalibrationInfo["processNoisePolicy"]),
            ("status", processNoiseCalibrationInfo["processNoiseCalibrationStatus"]),
            ("reason", processNoiseCalibrationInfo["processNoiseCalibrationReason"]),
            ("pre_kappa_level", processNoiseCalibrationInfo["preKappaQLevel"]),
            ("pre_kappa_trend", processNoiseCalibrationInfo["preKappaQTrend"]),
            ("global_scale", processNoiseCalibrationInfo.get("globalScale", 1.0)),
            (
                "valid_transitions",
                int(processNoiseCalibrationInfo.get("validTransitionCount", 0)),
            ),
            ("windows", int(processNoiseCalibrationInfo.get("windowCount", 0))),
            (
                "qscale_clamp_fraction",
                processNoiseCalibrationInfo.get("qScaleClampFraction", 0.0),
            ),
            (
                "base_q_clamp_changed",
                bool(processNoiseCalibrationInfo.get("baseQClampChanged", False)),
            ),
            (
                "qscale_decomp_max_log_error",
                processNoiseCalibrationInfo.get("qScaleDecompositionMaxLogError", 0.0),
            ),
        ),
    )
    baseFitPhaseLabel = "post-process-noise fit"
    fitPhaseLabel = (
        f"{logRunRole} {baseFitPhaseLabel}" if logRunRole else baseFitPhaseLabel
    )
    fitLogEvent = "postProcessNoiseFit"

    stageStart = time.perf_counter()
    _logAsciiBlock(
        fitPhaseLabel,
        (
            ("run label", logRunLabel),
            ("fit phase", fitPhaseLabel),
            ("tracks", int(trackCount)),
            ("intervals", int(intervalCount)),
            ("ECM max iterations", int(ECM_fixedBackgroundIters)),
            ("outer passes", int(ECM_outerIters)),
            ("background model fit", bool(fitBackground)),
            ("obs precision weights", bool(ECM_useObsPrecisionReweighting)),
            ("proc precision weights", bool(ECM_useProcessPrecisionReweighting)),
            ("APN enabled", bool(ECM_useAPN)),
        ),
        indentLevel=logIndentLevel + 1,
        level=logCoreBlockLevel,
    )
    logger.info(
        "runConsenrich.%s.start runLabel=%s tracks=%d intervals=%d "
        "ECM_fixedBackgroundIters=%d outerIters=%d",
        fitLogEvent,
        logRunLabel,
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
        initialLambdaLocal=postQInitialLambda,
        initialProcessPrecLocal=postQInitialProcessPrec,
        processQScaleLocal=processQScaleFinal,
        phaseLabel=fitPhaseLabel,
        phaseIndentLevel=logIndentLevel + 1,
        logAlternatingECMIterations=bool(logMainAlternatingECMIterations),
    )
    logger.info(
        "runConsenrich.%s.done runLabel=%s elapsed=%.3fs",
        fitLogEvent,
        logRunLabel,
        time.perf_counter() - stageStart,
    )
    stateSmoothed = np.asarray(fitFinal["stateSmoothed"], dtype=np.float32)
    stateCovarSmoothed = np.asarray(fitFinal["stateCovarSmoothed"], dtype=np.float32)
    postFitResiduals = np.asarray(fitFinal["postFitResiduals"], dtype=np.float32)
    NIS = np.asarray(fitFinal["NIS"], dtype=np.float32)
    finalForwardNIS = _finalForwardNIS(NIS)
    finalForwardGainContigSummary = _finalForwardReplicateGainContigSummary(
        stateCovarForward=np.asarray(fitFinal["stateCovarForward"], dtype=np.float32),
        matrixMunc=np.asarray(fitFinal["matrixMunc"], dtype=np.float32),
        lambdaExp=(
            fitFinal.get("lambdaExp") if bool(ECM_useObsPrecisionReweighting) else None
        ),
        pad=float(pad),
        obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
        obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
    )
    if bool(ECM_useObsPrecisionReweighting) and fitFinal.get("lambdaExp") is not None:
        finalObsPrecision = np.asarray(fitFinal["lambdaExp"]).reshape(-1)
        if finalObsPrecision.shape != (intervalCount,):
            raise ValueError("lambdaExp length must match interval count")
    finalObservationRTrace = np.zeros(intervalCount, dtype=np.float64)
    finalMunc = np.asarray(fitFinal["matrixMunc"])
    for rowIndex in range(int(finalMunc.shape[0])):
        finalObservationRTrace += np.maximum(
            np.asarray(finalMunc[rowIndex, :], dtype=np.float64) + float(pad),
            1.0e-12,
        )
    observationRTraceSummary = _metadataTrackSummary(finalObservationRTrace)
    processQDiagnostics = _processQDiagnosticsMetadata(
        matrixQ0=np.asarray(matrixQ0, dtype=np.float32),
        intervalCount=int(intervalCount),
        stateModel=stateModelMode,
        processPrecExp=(
            fitFinal.get("processPrecExp")
            if bool(ECM_useProcessPrecisionReweighting)
            else None
        ),
        processQScale=fitFinal.get("processQScale"),
        pNoiseForward=np.asarray(fitFinal["pNoiseForward"], dtype=np.float32),
        useAPN=bool(ECM_useAPN),
        processPrecisionRequested=bool(requestedProcessPrecisionReweighting),
        processPrecisionEffective=bool(ECM_useProcessPrecisionReweighting),
        procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
        procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
    )
    highIntervalMaskSource = processNoiseCalibrationInfo.get(
        "_puncDeadbandHighIntervalMask"
    )
    if highIntervalMaskSource is not None:
        highIntervalMask = np.asarray(highIntervalMaskSource, dtype=bool).reshape(-1)
        if highIntervalMask.shape == (intervalCount,):
            finalQTracks = _processQTrackArrays(
                matrixQ0=np.asarray(matrixQ0, dtype=np.float64),
                intervalCount=int(intervalCount),
                stateModel=stateModelMode,
                processPrecExp=(
                    fitFinal.get("processPrecExp")
                    if bool(ECM_useProcessPrecisionReweighting)
                    else None
                ),
                processQScale=fitFinal.get("processQScale"),
                pNoiseForward=np.asarray(fitFinal["pNoiseForward"], dtype=np.float32),
                procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
                returnFullQ=False,
            )
            if (
                bool(ECM_useProcessPrecisionReweighting)
                and fitFinal.get("processPrecExp") is not None
            ):
                kappaTrack = np.clip(
                    np.asarray(
                        fitFinal["processPrecExp"],
                        dtype=np.float64,
                    ).reshape(-1),
                    float(processPrecisionMultiplierMin),
                    float(processPrecisionMultiplierMax),
                )
            else:
                kappaTrack = np.ones(intervalCount, dtype=np.float64)
            if kappaTrack.shape == (intervalCount,):
                processNoiseCalibrationInfo["puncDeadbandHighKappaSummary"] = (
                    _metadataTrackSummaryWhere(kappaTrack, highIntervalMask)
                )
            processNoiseCalibrationInfo["puncDeadbandHighEffectiveQLevelSummary"] = (
                _metadataTrackSummaryWhere(
                    finalQTracks["effectiveQLevel"],
                    highIntervalMask,
                )
            )
            processNoiseCalibrationInfo["puncDeadbandHighEffectiveQTrendSummary"] = (
                _metadataTrackSummaryWhere(
                    finalQTracks["effectiveQTrend"],
                    highIntervalMask,
                )
            )
    logger.info(
        "processQ.finalPolicy runLabel=%s policy=%s APN=%s procPrecisionRequested=%s "
        "procPrecisionEffective=%s baseQLevel=%s baseQTrend=%s "
        "effectiveQLevelMedian=%s effectiveQTrendMedian=%s",
        logRunLabel,
        str(processQDiagnostics["policy"]),
        str(bool(processQDiagnostics["apn_enabled"])).lower(),
        str(
            bool(processQDiagnostics["process_precision_reweighting_requested"])
        ).lower(),
        str(
            bool(processQDiagnostics["process_precision_reweighting_effective"])
        ).lower(),
        _formatMaybeFloat(processQDiagnostics.get("baseQLevel")),
        _formatMaybeFloat(processQDiagnostics.get("baseQTrend")),
        _formatMaybeFloat(processQDiagnostics.get("effectiveQLevelMedian")),
        _formatMaybeFloat(processQDiagnostics.get("effectiveQTrendMedian")),
    )
    _logAsciiBlock(
        f"{fitPhaseLabel} summary",
        (
            ("run label", logRunLabel),
            ("fit phase", fitPhaseLabel),
            ("fit NLL", float(fitFinal.get("sumNLL", np.nan))),
            ("standardized forward innovation", float(finalForwardNIS)),
            ("backgroundShift at stop", float(fitFinal.get("backgroundShift", np.nan))),
            (
                "background objective/cell change at stop",
                fitFinal.get("backgroundObjectiveChangePerCell"),
            ),
            (
                "outer objective/cell change at stop",
                fitFinal.get("outerObjectiveChangePerCell"),
            ),
            ("outer stop reason", fitFinal.get("outerStopReason", "unknown")),
            ("background max abs", float(np.max(np.abs(fitFinal["background"])))),
            ("elapsed seconds", time.perf_counter() - stageStart),
        ),
        indentLevel=logIndentLevel + 1,
        level=logSummaryBlockLevel,
    )
    logger.info(
        "runConsenrich.%s.summary runLabel=%s finalNLL=%s finalForwardNIS=%.6g "
        "backgroundShift=%s backgroundObjectiveChangePerCell=%s "
        "outerObjectiveChangePerCell=%s outerStopReason=%s backgroundMaxAbs=%.6g",
        fitLogEvent,
        logRunLabel,
        _formatMaybeFloat(fitFinal.get("sumNLL", np.nan)),
        finalForwardNIS,
        _formatMaybeFloat(fitFinal.get("backgroundShift", np.nan)),
        _formatMaybeFloat(fitFinal.get("backgroundObjectiveChangePerCell")),
        _formatMaybeFloat(fitFinal.get("outerObjectiveChangePerCell")),
        str(fitFinal.get("outerStopReason", "unknown")),
        float(np.max(np.abs(fitFinal["background"]))),
    )

    def _processNoiseCalibrationMetadataValue(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        if isinstance(value, Mapping):
            return {
                str(key): _processNoiseCalibrationMetadataValue(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_processNoiseCalibrationMetadataValue(item) for item in value]
        return _diagnosticScalar(value)

    processNoiseCalibrationMetadata = (
        None
        if processNoiseCalibrationInfo is None
        else {
            key: _processNoiseCalibrationMetadataValue(value)
            for key, value in processNoiseCalibrationInfo.items()
            if not str(key).startswith("_")
        }
    )
    runDiagnostics = {
        "state_model": stateModelMode,
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
            "sd": [
                metadataFloat(float(value))
                for value in finalForwardGainContigSummary["sd"]
            ],
            "iqr": [
                metadataFloat(float(value))
                for value in finalForwardGainContigSummary["iqr"]
            ],
            "count": [int(value) for value in finalForwardGainContigSummary["count"]],
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
        "process_noise_calibration": processNoiseCalibrationMetadata,
        "process_noise_warmup_fit": fitProcessNoiseWarmupMetadata,
        "post_process_noise_fit": _fitDiagnosticsMetadata(fitFinal),
        "optimization_path_tracked": bool(trackOptimizationPath),
        "process_precision_reweighting_requested": bool(
            requestedProcessPrecisionReweighting
        ),
        "process_precision_reweighting_effective": bool(
            ECM_useProcessPrecisionReweighting
        ),
        "process_precision_reweighting_disabled_by_apn": bool(
            requestedProcessPrecisionReweighting
            and ECM_useAPN
            and not ECM_useProcessPrecisionReweighting
        ),
        "adaptive_process_noise_effective": bool(ECM_useAPN),
        "process_q_policy": processQDiagnostics["policy"],
        "process_q_diagnostics": processQDiagnostics,
        "observation_r_trace": observationRTraceSummary,
    }

    outStateSmoothed = np.asarray(stateSmoothed, dtype=np.float32)
    outStateCovarSmoothed = np.asarray(stateCovarSmoothed, dtype=np.float32)
    if stateModelMode == STATE_MODEL_LEVEL:
        outStateSmoothed = _padLevelStateArray(outStateSmoothed)
        outStateCovarSmoothed = _padLevelCovarArray(outStateCovarSmoothed)
    outPostFitResiduals = np.asarray(postFitResiduals, dtype=np.float32)
    outNIS = np.asarray(NIS, dtype=np.float32)
    outBackground = np.asarray(fitFinal["background"], dtype=np.float32)

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
            ("run label", logRunLabel),
            ("tracks", int(trackCount)),
            ("intervals", int(intervalCount)),
            ("elapsed seconds", float(totalElapsed)),
        ),
        indentLevel=logIndentLevel,
        level=logCoreBlockLevel,
    )
    logger.info(
        "runConsenrich.core.done runLabel=%s tracks=%d intervals=%d elapsed=%.3fs",
        logRunLabel,
        int(trackCount),
        int(intervalCount),
        totalElapsed,
    )

    def _maybeAddRequestedOutputs(result: tuple[Any, ...]) -> tuple[Any, ...]:
        if returnBackground:
            result = (*result, outBackground)
        if returnPrecisionDiagnostics:
            perIntervalOutputDiagnostics = _perIntervalOutputDiagnosticTracks(
                stateCovarForward=np.asarray(
                    fitFinal["stateCovarForward"],
                    dtype=np.float32,
                ),
                matrixMunc=np.asarray(fitFinal["matrixMunc"], dtype=np.float32),
                matrixQ0=np.asarray(matrixQ0, dtype=np.float32),
                matrixF=np.asarray(matrixF, dtype=np.float32),
                stateCovarInit=float(stateCovarInit),
                stateModel=stateModelMode,
                lambdaExp=(
                    fitFinal.get("lambdaExp")
                    if bool(ECM_useObsPrecisionReweighting)
                    else None
                ),
                processPrecExp=(
                    fitFinal.get("processPrecExp")
                    if (
                        bool(ECM_useProcessPrecisionReweighting)
                        and not bool(ECM_useAPN)
                    )
                    else None
                ),
                processQScale=fitFinal.get("processQScale"),
                pNoiseForward=np.asarray(fitFinal["pNoiseForward"], dtype=np.float32),
                pad=float(pad),
                obsPrecisionMultiplierMin=float(observationPrecisionMultiplierMin),
                obsPrecisionMultiplierMax=float(observationPrecisionMultiplierMax),
                procPrecisionMultiplierMin=float(processPrecisionMultiplierMin),
                procPrecisionMultiplierMax=float(processPrecisionMultiplierMax),
            )
            result = (
                *result,
                {
                    "precision_track_diagnostics": True,
                    "state_model": stateModelMode,
                    "ECM_useAPN": bool(ECM_useAPN),
                    "process_precision_reweighting_requested": bool(
                        requestedProcessPrecisionReweighting
                    ),
                    "process_precision_reweighting_effective": bool(
                        ECM_useProcessPrecisionReweighting
                    ),
                    "process_precision_reweighting_disabled_by_apn": bool(
                        requestedProcessPrecisionReweighting
                        and ECM_useAPN
                        and not ECM_useProcessPrecisionReweighting
                    ),
                    "process_q_policy": processQDiagnostics["policy"],
                    "process_q_diagnostics": processQDiagnostics,
                    "observationPrecisionMultiplierMin": float(
                        observationPrecisionMultiplierMin
                    ),
                    "observationPrecisionMultiplierMax": float(
                        observationPrecisionMultiplierMax
                    ),
                    "processPrecisionMultiplierMin": float(
                        processPrecisionMultiplierMin
                    ),
                    "processPrecisionMultiplierMax": float(
                        processPrecisionMultiplierMax
                    ),
                    "lambdaExp": (
                        None
                        if fitFinal.get("lambdaExp") is None
                        else np.asarray(fitFinal["lambdaExp"], dtype=np.float32)
                    ),
                    "processPrecExp": (
                        None
                        if fitFinal.get("processPrecExp") is None
                        else np.asarray(fitFinal["processPrecExp"], dtype=np.float32)
                    ),
                    "matrixQ0": np.asarray(matrixQ0, dtype=np.float32),
                    "outputTracks": {
                        name: np.asarray(values, dtype=np.float32)
                        for name, values in perIntervalOutputDiagnostics.items()
                    },
                },
            )
        if returnDiagnostics:
            return (*result, runDiagnostics)
        return result

    if returnScales:
        result = (
            outStateSmoothed,
            outStateCovarSmoothed,
            outPostFitResiduals,
            outNIS,
            intervalToBlockMap,
        )
        return _maybeAddRequestedOutputs(result)

    result = (
        outStateSmoothed,
        outStateCovarSmoothed,
        outPostFitResiduals,
        outNIS,
    )
    return _maybeAddRequestedOutputs(result)


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


@lru_cache(maxsize=8)
def _readSparseRegionsByChrom(sparseBedFile: str) -> dict[str, np.ndarray]:
    sparseRegionsByChrom: dict[str, list[tuple[int, int]]] = {}
    with open(sparseBedFile, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chromName = str(parts[0])
            bedStart = int(parts[1])
            bedEnd = int(parts[2])
            if bedEnd <= bedStart:
                continue
            sparseRegionsByChrom.setdefault(chromName, []).append((bedStart, bedEnd))

    out: dict[str, np.ndarray] = {}
    for chromName, regions in sparseRegionsByChrom.items():
        chromRegions = np.asarray(regions, dtype=np.int64)
        order = np.argsort(chromRegions[:, 0], kind="mergesort")
        out[chromName] = chromRegions[order, :]
    return out


def _loadSparseIntervalIndices(
    sparseBedFile: str,
    chromosome: str,
    intervals: np.ndarray,
) -> np.ndarray:
    sparseRegions = _readSparseRegionsByChrom(str(sparseBedFile)).get(
        str(chromosome),
        np.empty((0, 2), dtype=np.int64),
    )
    if sparseRegions.size == 0:
        return np.empty(0, dtype=np.intp)

    intervalStarts = np.asarray(intervals, dtype=np.int64)
    if intervalStarts.size == 0:
        return np.empty(0, dtype=np.intp)
    if intervalStarts.size == 1:
        intervalSize = 1
    else:
        intervalSize = int(intervalStarts[1] - intervalStarts[0])
        if intervalSize <= 0:
            raise ValueError("intervals must be strictly increasing")
    intervalEnds = intervalStarts + int(intervalSize)

    sparseMask = np.zeros(intervalStarts.size, dtype=bool)
    for bedStart, bedEnd in sparseRegions:
        firstIdx = int(np.searchsorted(intervalEnds, int(bedStart), side="right"))
        lastIdx = int(np.searchsorted(intervalStarts, int(bedEnd), side="left"))
        if firstIdx < 0:
            firstIdx = 0
        if lastIdx > intervalStarts.size:
            lastIdx = intervalStarts.size
        if lastIdx > firstIdx:
            sparseMask[firstIdx:lastIdx] = True

    sparseIdx = np.flatnonzero(sparseMask)
    if sparseIdx.size == 0:
        return np.empty(0, dtype=np.intp)
    return sparseIdx.astype(np.intp, copy=False)


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


class MuncAdditiveCovariateModel(NamedTuple):
    r"""Nonnegative additive genomic-covariate variance model for MUNC."""

    featureNames: tuple[str, ...]
    basisEdges: np.ndarray
    basisMetadata: dict[str, Any]
    pooledCoefficients: np.ndarray
    perReplicateCoefficients: np.ndarray
    replicateUsesPooled: np.ndarray
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
    quantiles: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        _sharedWeightedQuantileInterpolated(values, weights, quantiles),
        dtype=np.float64,
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
    if variances.size and (
        (not np.all(np.isfinite(variances))) or np.any(variances <= 0.0)
    ):
        raise ValueError("blockVariances must contain only finite positive values")

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


def _sanitizeMuncCovariateMatrix(
    covariates: np.ndarray,
    expectedFeatures: int | None = None,
) -> np.ndarray:
    covArr = np.asarray(covariates, dtype=np.float64)
    if covArr.ndim == 1:
        covArr = covArr.reshape(-1, 1)
    elif covArr.ndim != 2:
        raise ValueError("MUNC covariates must be a 2D array")
    if expectedFeatures is not None and covArr.shape[1] != int(expectedFeatures):
        raise ValueError(
            "MUNC covariate feature count mismatch: "
            f"expected {int(expectedFeatures)}, got {int(covArr.shape[1])}"
        )
    covArr = np.array(covArr, dtype=np.float64, copy=True)
    finite = np.isfinite(covArr)
    covArr[finite & (covArr < 0.0)] = 0.0
    return covArr


def _muncAdditiveBasisEdges(
    means: np.ndarray,
    weights: np.ndarray | None = None,
    basisCount: int = 4,
) -> np.ndarray:
    predictor = _muncTrendPredictor(means).ravel()
    if weights is None:
        weightsArr = np.ones_like(predictor, dtype=np.float64)
    else:
        weightsArr = np.asarray(weights, dtype=np.float64).ravel()
        if weightsArr.shape != predictor.shape:
            raise ValueError("weights must align with blockMeans")
    valid = np.isfinite(predictor) & np.isfinite(weightsArr) & (weightsArr > 0.0)
    x = predictor[valid]
    w = weightsArr[valid]
    if x.size < 2 or np.nanmax(x) <= np.nanmin(x):
        return np.array([-np.inf, np.inf], dtype=np.float64)
    requested = int(max(1, min(int(basisCount), 16)))
    if requested <= 1:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    probs = np.linspace(0.0, 1.0, requested + 1, dtype=np.float64)[1:-1]
    internal = _weightedQuantile(x, w, probs)
    internal = np.unique(internal[np.isfinite(internal)])
    if internal.size:
        span = float(np.max(x) - np.min(x))
        minGap = max(span * 1.0e-8, 1.0e-10)
        internal = internal[
            (internal > float(np.min(x)) + minGap)
            & (internal < float(np.max(x)) - minGap)
        ]
        internal = np.unique(internal)
    if internal.size == 0:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    return np.concatenate(
        (
            np.array([-np.inf], dtype=np.float64),
            internal.astype(np.float64, copy=False),
            np.array([np.inf], dtype=np.float64),
        )
    )


def _muncAdditiveBasisIndex(means: np.ndarray, basisEdges: np.ndarray) -> np.ndarray:
    edges = np.asarray(basisEdges, dtype=np.float64).ravel()
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf], dtype=np.float64)
    predictor = _muncTrendPredictor(means).ravel()
    bins = np.searchsorted(edges[1:-1], predictor, side="right").astype(np.intp)
    bins[~np.isfinite(predictor)] = 0
    return np.clip(bins, 0, int(edges.size) - 2)


def _muncAdditiveDesign(
    means: np.ndarray,
    covariates: np.ndarray,
    basisEdges: np.ndarray,
) -> np.ndarray:
    covArr = _sanitizeMuncCovariateMatrix(covariates)
    bins = _muncAdditiveBasisIndex(means, basisEdges)
    if covArr.shape[0] != bins.size:
        raise ValueError("MUNC covariates must align with blockMeans")
    featureCount = int(covArr.shape[1])
    basisCount = int(max(1, np.asarray(basisEdges).size - 1))
    design = np.zeros((bins.size, featureCount * basisCount), dtype=np.float64)
    rows = np.arange(bins.size, dtype=np.intp)
    if np.any(~np.isfinite(covArr)):
        raise ValueError("MUNC covariate design cannot contain missing covariates")
    for feature in range(featureCount):
        design[rows, feature * basisCount + bins] = covArr[:, feature]
    return design


def _fitNonnegativeRidge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    X = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != y.size or y.size != w.size:
        raise ValueError("nonnegative ridge design, target, and weights must align")
    if X.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return np.zeros(X.shape[1], dtype=np.float64)
    X = X[valid, :]
    y = np.maximum(y[valid], 0.0)
    w = w[valid]
    if not np.any(np.isfinite(X)) or float(np.max(np.abs(X))) <= 0.0:
        return np.zeros(X.shape[1], dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    sqrtW = np.sqrt(w)
    Xw = X * sqrtW[:, None]
    yw = y * sqrtW
    ridge_ = float(ridge)
    if np.isfinite(ridge_) and ridge_ > 0.0:
        Xw = np.vstack(
            (
                Xw,
                math.sqrt(ridge_) * np.eye(X.shape[1], dtype=np.float64),
            )
        )
        yw = np.concatenate((yw, np.zeros(X.shape[1], dtype=np.float64)))
    try:
        beta, _ = optimize.nnls(Xw, yw, maxiter=max(3 * Xw.shape[1], 1))
    except RuntimeError as exc:
        raise RuntimeError("nonnegative ridge NNLS failed to converge") from exc
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (X.shape[1],):
        raise RuntimeError(
            "nonnegative ridge NNLS returned an invalid coefficient count"
        )
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError(
            "nonnegative ridge NNLS returned non-finite coefficients"
        )
    if np.any(beta < 0.0):
        raise FloatingPointError(
            "nonnegative ridge NNLS returned negative coefficients"
        )
    return beta


def fitMuncAdditiveCovariateModel(
    blockMeans: np.ndarray,
    blockVariances: np.ndarray,
    baselineVariances: np.ndarray,
    blockCovariates: np.ndarray,
    sampleIndex: np.ndarray,
    *,
    featureNames: Tuple[str, ...] | tuple[str, ...] | None = None,
    weights: np.ndarray | None = None,
    sampleCount: int | None = None,
    minBlocksPerReplicate: int = 250,
    basisCount: int = 4,
    ridge: float = 1.0,
    eps: float = 1.0e-12,
) -> MuncAdditiveCovariateModel:
    r"""Fit per-replicate nonnegative additive genomic MUNC variance."""

    means = np.asarray(blockMeans, dtype=np.float64).ravel()
    variances = np.asarray(blockVariances, dtype=np.float64).ravel()
    baseline = np.asarray(baselineVariances, dtype=np.float64).ravel()
    samples = np.asarray(sampleIndex, dtype=np.int64).ravel()
    if means.shape != variances.shape or means.shape != baseline.shape:
        raise ValueError("blockMeans, blockVariances, and baselineVariances must align")
    if samples.shape != means.shape:
        raise ValueError("sampleIndex must align with blockMeans")
    covArr = _sanitizeMuncCovariateMatrix(blockCovariates)
    if covArr.shape[0] != means.size:
        raise ValueError("blockCovariates must align with blockMeans")
    featureCount = int(covArr.shape[1])
    if featureNames is None:
        names = tuple(f"feature_{idx}" for idx in range(featureCount))
    else:
        names = tuple(str(name) for name in featureNames)
    if len(names) != featureCount:
        raise ValueError("featureNames must match blockCovariates columns")
    if weights is None:
        weightsArr = np.ones_like(means, dtype=np.float64)
    else:
        weightsArr = np.asarray(weights, dtype=np.float64).ravel()
        if weightsArr.shape != means.shape:
            raise ValueError("weights must align with blockMeans")
    if sampleCount is None:
        sampleCount_ = int(np.max(samples)) + 1 if samples.size else 0
    else:
        sampleCount_ = int(max(sampleCount, 0))

    valid = (
        np.isfinite(means)
        & np.isfinite(variances)
        & np.isfinite(baseline)
        & np.isfinite(weightsArr)
        & np.all(np.isfinite(covArr), axis=1)
        & (weightsArr > 0.0)
        & (variances > max(float(eps), 0.0))
        & (baseline > 0.0)
        & (samples >= 0)
        & (samples < sampleCount_)
    )
    validCount = int(np.count_nonzero(valid))
    basisEdges = _muncAdditiveBasisEdges(
        means[valid],
        weights=weightsArr[valid],
        basisCount=basisCount,
    )
    basisCount_ = int(max(1, basisEdges.size - 1))
    coefShape = (featureCount, basisCount_)
    pooledCoefficients = np.zeros(coefShape, dtype=np.float64)
    perReplicateCoefficients = np.zeros(
        (sampleCount_, featureCount, basisCount_),
        dtype=np.float64,
    )
    replicateUsesPooled = np.ones(sampleCount_, dtype=bool)
    replicateValidCounts = np.zeros(sampleCount_, dtype=np.int64)

    if validCount > 0 and featureCount > 0:
        excess = np.maximum(variances[valid] - baseline[valid], 0.0)
        fitMeans = means[valid]
        fitCov = covArr[valid, :]
        fitWeights = weightsArr[valid]
        fitSamples = samples[valid]
        pooledDesign = _muncAdditiveDesign(fitMeans, fitCov, basisEdges)
        pooledBeta = _fitNonnegativeRidge(
            pooledDesign,
            excess,
            fitWeights,
            ridge,
        )
        pooledCoefficients = pooledBeta.reshape(coefShape)
        perReplicateCoefficients[:] = pooledCoefficients[None, :, :]

        minBlocks = int(max(1, minBlocksPerReplicate))
        for sample in range(sampleCount_):
            sampleMask = fitSamples == int(sample)
            replicateValidCounts[sample] = int(np.count_nonzero(sampleMask))
            if replicateValidCounts[sample] < minBlocks:
                continue
            repBeta = _fitNonnegativeRidge(
                pooledDesign[sampleMask, :],
                excess[sampleMask],
                fitWeights[sampleMask],
                ridge,
            )
            perReplicateCoefficients[sample, :, :] = repBeta.reshape(coefShape)
            replicateUsesPooled[sample] = False

    diagnostics = {
        "valid_pairs": validCount,
        "feature_count": featureCount,
        "basis_count": basisCount_,
        "basis_edges": basisEdges.tolist(),
        "basis_predictor": "signed_log1p",
        "ridge": float(ridge),
        "min_blocks_per_replicate": int(max(1, minBlocksPerReplicate)),
        "pooled_coefficient_sum": float(np.sum(pooledCoefficients)),
        "per_replicate_coefficient_sum": [
            float(value)
            for value in np.sum(perReplicateCoefficients, axis=(1, 2)).tolist()
        ],
        "replicate_valid_counts": [
            int(value) for value in replicateValidCounts.tolist()
        ],
        "replicate_fallback_count": int(np.count_nonzero(replicateUsesPooled)),
    }
    return MuncAdditiveCovariateModel(
        featureNames=names,
        basisEdges=basisEdges.astype(np.float64, copy=False),
        basisMetadata={
            "type": "quantile_indicator",
            "predictor": "signed_log1p",
            "basis_count": basisCount_,
        },
        pooledCoefficients=pooledCoefficients.astype(np.float64, copy=False),
        perReplicateCoefficients=perReplicateCoefficients.astype(
            np.float64,
            copy=False,
        ),
        replicateUsesPooled=replicateUsesPooled,
        diagnostics=diagnostics,
    )


def evalMuncAdditiveCovariateModel(
    model: MuncAdditiveCovariateModel | None,
    meanTrack: np.ndarray,
    covariateTrack: np.ndarray | None,
    replicateIndex: int | None = None,
) -> np.ndarray:
    r"""Evaluate a nonnegative additive MUNC covariate variance track."""

    means = np.asarray(meanTrack, dtype=np.float64).ravel()
    if model is None or covariateTrack is None:
        return np.zeros_like(means, dtype=np.float32)
    featureCount = int(len(model.featureNames))
    if featureCount == 0:
        return np.zeros_like(means, dtype=np.float32)
    covArr = _sanitizeMuncCovariateMatrix(covariateTrack, featureCount)
    if covArr.shape[0] != means.size:
        raise ValueError("covariateTrack must align with meanTrack")
    finiteRows = np.all(np.isfinite(covArr), axis=1)
    if not np.any(finiteRows):
        return np.zeros_like(means, dtype=np.float32)
    basisEdges = np.asarray(model.basisEdges, dtype=np.float64)
    bins = _muncAdditiveBasisIndex(means, basisEdges)
    if replicateIndex is None:
        coefficients = np.asarray(model.pooledCoefficients, dtype=np.float64)
    else:
        rep = int(replicateIndex)
        perRep = np.asarray(model.perReplicateCoefficients, dtype=np.float64)
        usesPooled = np.asarray(model.replicateUsesPooled, dtype=bool).ravel()
        if (
            rep < 0
            or rep >= perRep.shape[0]
            or (rep < usesPooled.size and bool(usesPooled[rep]))
        ):
            coefficients = np.asarray(model.pooledCoefficients, dtype=np.float64)
        else:
            coefficients = perRep[rep, :, :]
    out = np.zeros(means.size, dtype=np.float64)
    for feature in range(featureCount):
        out[finiteRows] += (
            covArr[finiteRows, feature] * coefficients[feature, bins[finiteRows]]
        )
    out[~np.isfinite(out)] = 0.0
    out[out < 0.0] = 0.0
    return out.astype(np.float32, copy=False)


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

    _ = maxIters, tol
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
    if variances.size and (
        (not np.all(np.isfinite(variances))) or np.any(variances <= 0.0)
    ):
        raise ValueError("blockVariances must contain only finite positive values")

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

    trend = fitPSplineLogVarianceTrend(
        means,
        variances,
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
    sampleCount = int(np.max(samples)) + 1
    factors = np.ones(sampleCount, dtype=np.float64)
    diagnostics = {
        "pooled_pairs": int(means.size),
        "replicate_count": int(sampleCount),
        "factor_min": 1.0,
        "factor_median": 1.0,
        "factor_max": 1.0,
        "iterations": 0,
        "max_log_factor_change": 0.0,
        "predictor": "signed_log1p",
        "replicate_factor_fit": "disabled",
    }
    return PooledMuncVarianceTrend(
        trend=trend,
        replicateVarianceFactors=factors.astype(np.float64, copy=False),
        diagnostics=diagnostics,
    )


def applyBlacklistMuncFloor(
    muncMatrix: np.ndarray,
    blacklistMask: np.ndarray,
    minR: float,
    quantile: float = 0.05,
) -> np.ndarray:
    r"""Enforce a robust per-sample MUNC floor inside blacklisted bins."""

    arr = np.asarray(muncMatrix)
    if arr.ndim == 1:
        arr2 = arr.reshape(1, -1)
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise ValueError("muncMatrix must be one- or two-dimensional")
    mask = np.asarray(blacklistMask, dtype=bool).ravel()
    if mask.size != arr2.shape[1]:
        raise ValueError("blacklistMask length must match MUNC track length")
    baseFloor = resolveMuncMinRFloor(arr2, minR, fallback=0.0)
    floors = np.full(arr2.shape[0], baseFloor, dtype=np.float64)
    if not np.any(mask):
        return floors

    q = float(np.clip(float(quantile), 0.0, 1.0))
    nonBlacklist = ~mask
    for sample in range(arr2.shape[0]):
        row = arr2[sample, :]
        finiteReference = row[nonBlacklist & np.isfinite(row)]
        if finiteReference.size:
            floor = max(baseFloor, float(np.quantile(finiteReference, q)))
        else:
            floor = baseFloor
        floors[sample] = floor
        blackValues = row[mask]
        row[mask] = np.where(
            np.isfinite(blackValues),
            np.maximum(blackValues, floor),
            floor,
        )
    return floors


def resolveMuncMinRFloor(
    muncMatrix: np.ndarray,
    minR: float,
    quantile: float = 0.05,
    fallback: float = OBSERVATION_DEFAULT_MIN_R,
) -> float:
    r"""Resolve a negative MUNC variance floor from the MUNC matrix quantile."""

    requested = float(minR)
    if not np.isfinite(requested):
        raise ValueError("minR must be finite")
    if requested >= 0.0:
        return requested

    q = float(np.clip(float(quantile), 0.0, 1.0))
    arr = np.asarray(muncMatrix)
    finiteValues = np.asarray(
        arr[np.isfinite(arr) & (arr >= 0.0)],
        dtype=np.float64,
    )
    if finiteValues.size == 0:
        return float(max(float(fallback), 0.0))

    resolved = float(np.quantile(finiteValues, q))
    if not np.isfinite(resolved) or resolved < 0.0:
        return float(max(float(fallback), 0.0))
    return resolved


def _formatPSplineTrendSummary(
    trend: PSplineLogVarianceTrend,
    supportSignedMeans: np.ndarray,
    eps: float,
    maxVariance: float | None = None,
    pointCount: int = 9,
    sampleFile: str | None = None,
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
    sampleText = (
        "" if sampleFile is None else f" sampleFile={_quoteLogString(sampleFile)}"
    )
    return (
        f"MUNC P-spline signed-mean-SD trend{sampleText}:\n"
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
    countModelVarianceFloorTrack: np.ndarray | None = None,
    sampleFile: str | None = None,
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

    floorLine = ""
    if countModelVarianceFloorTrack is not None:
        floorLine = f"\n\t{_sdQuantiles('count_floor', countModelVarianceFloorTrack)}"
        localName = "L_excess"
        globalName = "G_excess"
        finalName = "V_total"
    else:
        localName = "L"
        globalName = "G"
        finalName = "V0"

    sampleText = (
        "" if sampleFile is None else f" sampleFile={_quoteLogString(sampleFile)}"
    )
    return (
        f"MUNC variance SD diagnostics{sampleText}:\n"
        f"\t{_sdQuantiles(localName, localVarianceTrack)}\n"
        f"\t{_sdQuantiles(globalName, globalVarianceTrack)}\n"
        f"\t{_sdQuantiles(finalName, finalVarianceTrack)}"
        f"{floorLine}\n"
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


def _coerceMuncCountModelVarianceFloor(
    countModelVarianceFloor: np.ndarray | None,
    intervalCount: int,
) -> np.ndarray | None:
    if countModelVarianceFloor is None:
        return None

    floorTrack = np.asarray(countModelVarianceFloor, dtype=np.float64).reshape(-1)
    if floorTrack.size != int(intervalCount):
        raise ValueError(
            "countModelVarianceFloor must be a one-dimensional track matching "
            "the MUNC interval count",
        )
    finite = np.isfinite(floorTrack)
    if np.any(finite & (floorTrack < 0.0)):
        raise ValueError("countModelVarianceFloor must be nonnegative where finite")
    if not np.any(finite):
        return None
    out = np.full(floorTrack.shape, np.nan, dtype=np.float64)
    out[finite] = floorTrack[finite]
    return out


def applyMuncCountModelVarianceFloor(
    muncVarianceTrack: np.ndarray,
    countModelVarianceFloor: np.ndarray | None,
    *,
    varianceFloor: float,
    varianceCap: float | None = None,
) -> npt.NDArray[np.float32]:
    r"""Apply transformed-scale count-model variance as a MUNC lower bound.

    ``countModelVarianceFloor`` is intentionally already on the same scale as
    the transformed observation track. Raw-count Poisson/NB or
    treatment-control delta-method calculations belong upstream, where the raw
    counts, scale factors, and transform parameters are still available.
    """

    out = _clipVarianceTrack(
        muncVarianceTrack,
        floor=varianceFloor,
        cap=varianceCap,
    ).astype(np.float64, copy=False)
    floorTrack = _coerceMuncCountModelVarianceFloor(
        countModelVarianceFloor,
        out.size,
    )
    if floorTrack is None:
        return out.astype(np.float32, copy=False)

    finite = np.isfinite(floorTrack)
    out[finite] = np.maximum(out[finite], floorTrack[finite])
    return _clipVarianceTrack(out, floor=varianceFloor, cap=varianceCap)


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


def _buildBackgroundSparseSystem(
    weightTrack: np.ndarray,
    lamFirst: float,
    lamSecond: float,
) -> sparse.csr_matrix:
    n = int(weightTrack.shape[0])
    systemMat = sparse.diags(weightTrack, offsets=0, format="csr")
    systemMat = systemMat + (float(lamFirst) * _buildFirstDiffPenalty(n))
    systemMat = systemMat + (float(lamSecond) * _buildSecondDiffPenalty(n))
    return systemMat.tocsr()


def _buildBackgroundBandedSystem(
    weightTrack: np.ndarray,
    lamFirst: float,
    lamSecond: float,
) -> np.ndarray:
    weightArr = np.asarray(weightTrack, dtype=np.float64).reshape(-1)
    n = int(weightArr.shape[0])
    banded = np.zeros((3, n), dtype=np.float64)
    if n <= 0:
        return banded

    diag = weightArr.copy()
    if n >= 2 and lamFirst > 0.0:
        diag[0] += lamFirst
        diag[-1] += lamFirst
        if n > 2:
            diag[1:-1] += 2.0 * lamFirst
        banded[1, : n - 1] -= lamFirst

    if n >= 3 and lamSecond > 0.0:
        if n == 3:
            diag[0] += lamSecond
            diag[1] += 4.0 * lamSecond
            diag[2] += lamSecond
            banded[1, :2] += -2.0 * lamSecond
        else:
            diag[0] += lamSecond
            diag[-1] += lamSecond
            diag[1] += 5.0 * lamSecond
            diag[-2] += 5.0 * lamSecond
            if n > 4:
                diag[2:-2] += 6.0 * lamSecond
            banded[1, 0] += -2.0 * lamSecond
            banded[1, n - 2] += -2.0 * lamSecond
            if n > 3:
                banded[1, 1 : n - 2] += -4.0 * lamSecond
        banded[2, : n - 2] = lamSecond

    banded[0, :] = diag
    return banded


def _backgroundPenaltyDiagonal(
    intervalCount: int,
    lamFirst: float,
    lamSecond: float,
) -> np.ndarray:
    n = int(intervalCount)
    diag = np.zeros(n, dtype=np.float64)
    if n >= 2 and lamFirst > 0.0:
        diag[0] += lamFirst
        diag[-1] += lamFirst
        if n > 2:
            diag[1:-1] += 2.0 * lamFirst
    if n >= 3 and lamSecond > 0.0:
        if n == 3:
            diag += np.asarray([1.0, 4.0, 1.0], dtype=np.float64) * lamSecond
        else:
            diag[0] += lamSecond
            diag[-1] += lamSecond
            diag[1] += 5.0 * lamSecond
            diag[-2] += 5.0 * lamSecond
            if n > 4:
                diag[2:-2] += 6.0 * lamSecond
    return diag


def _diagonalBackgroundUncertainty(
    matrixMunc: np.ndarray,
    *,
    blockLenIntervals: int,
    pad: float,
    lambdaExp: np.ndarray | None,
    backgroundSmoothness: float,
    fitBackground: bool,
    backgroundTrack: np.ndarray | None = None,
    useNonnegativeBackground: bool = False,
    backgroundNegativePenaltyMultiplier: float | None = None,
) -> np.ndarray:
    munc = np.asarray(matrixMunc)
    if munc.ndim != 2:
        raise ValueError("matrixMunc must be two-dimensional")
    intervalCount = int(munc.shape[1])
    if not bool(fitBackground):
        return np.zeros(intervalCount, dtype=np.float32)
    weightTrack = np.zeros(intervalCount, dtype=np.float64)
    if lambdaExp is not None:
        lam = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
        if lam.shape[0] != intervalCount:
            raise ValueError("lambdaExp length must match matrixMunc interval count")
        lam = np.maximum(lam, 1.0e-12)
    else:
        lam = None
    for rowIndex in range(int(munc.shape[0])):
        denom = np.maximum(
            np.asarray(munc[rowIndex, :], dtype=np.float64) + float(pad),
            1.0e-12,
        )
        invVar = 1.0 / denom
        if lam is not None:
            invVar *= lam
        weightTrack += invVar
    lamFirst, lamSecond = _backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=int(blockLenIntervals),
        backgroundSmoothness=float(backgroundSmoothness),
    )
    diagonal = weightTrack + _backgroundPenaltyDiagonal(
        intervalCount,
        float(lamFirst),
        float(lamSecond),
    )
    if (
        bool(useNonnegativeBackground)
        and backgroundTrack is not None
        and backgroundNegativePenaltyMultiplier is not None
    ):
        negativePenaltyMultiplier = float(backgroundNegativePenaltyMultiplier)
        if negativePenaltyMultiplier > 0.0:
            bg = np.asarray(backgroundTrack, dtype=np.float64).reshape(-1)
            if bg.shape[0] != intervalCount:
                raise ValueError("backgroundTrack length must match matrixMunc")
            positiveWeights = weightTrack[weightTrack > 0.0]
            weightScale = (
                float(np.median(positiveWeights)) if positiveWeights.size else 1.0
            )
            diagonal[bg < 0.0] += negativePenaltyMultiplier * max(weightScale, 1.0e-12)
    return (1.0 / np.maximum(diagonal, 1.0e-12)).astype(np.float32)


def _solveBackgroundLinearSystem(
    weightTrack: np.ndarray,
    rhsTrack: np.ndarray,
    lamFirst: float,
    lamSecond: float,
) -> np.ndarray:
    rhs = np.asarray(rhsTrack, dtype=np.float64)
    if rhs.size == 0:
        return np.zeros(0, dtype=np.float64)
    weight = np.ascontiguousarray(weightTrack, dtype=np.float64)
    if rhs.ndim == 1:
        return np.asarray(
            cconsenrich.csolveZeroCenteredBackground(
                weight,
                np.ascontiguousarray(rhs, dtype=np.float64),
                float(lamSecond),
                False,
                lamFirst=float(lamFirst),
            ),
            dtype=np.float64,
        )
    if rhs.ndim == 2:
        return np.column_stack(
            [
                np.asarray(
                    cconsenrich.csolveZeroCenteredBackground(
                        weight,
                        np.ascontiguousarray(rhs[:, colIndex], dtype=np.float64),
                        float(lamSecond),
                        False,
                        lamFirst=float(lamFirst),
                    ),
                    dtype=np.float64,
                )
                for colIndex in range(rhs.shape[1])
            ]
        )
    raise ValueError("rhsTrack must be one- or two-dimensional")


def solveZeroCenteredBackgroundLinearSystem(
    weightTrack: np.ndarray,
    rhsTrack: np.ndarray,
    lamFirst: float,
    lamSecond: float,
) -> np.ndarray:
    rhs = np.asarray(rhsTrack, dtype=np.float64)
    if rhs.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        cconsenrich.csolveZeroCenteredBackground(
            np.ascontiguousarray(weightTrack, dtype=np.float64),
            np.ascontiguousarray(rhs, dtype=np.float64),
            float(lamSecond),
            True,
            lamFirst=float(lamFirst),
        ),
        dtype=np.float64,
    )


def _backgroundBandedMatvec(
    systemBanded: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    product = systemBanded[0, :n] * x
    if n >= 2:
        off1 = systemBanded[1, : n - 1]
        product[:-1] += off1 * x[1:]
        product[1:] += off1 * x[:-1]
    if n >= 3:
        off2 = systemBanded[2, : n - 2]
        product[:-2] += off2 * x[2:]
        product[2:] += off2 * x[:-2]
    return product


def _nonnegativeBackgroundKKTDiagnostics(
    *,
    systemBanded: np.ndarray,
    rhsTrack: np.ndarray,
    candidate: np.ndarray,
    primalTol: float,
) -> dict[str, float | int | bool]:
    x = np.asarray(candidate, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhsTrack, dtype=np.float64).reshape(-1)
    matX = _backgroundBandedMatvec(systemBanded, x)
    gradient = matX - rhs

    finiteX = x[np.isfinite(x)]
    finiteRhs = rhs[np.isfinite(rhs)]
    finiteMatX = matX[np.isfinite(matX)]
    signalScale = float(max(np.max(np.abs(finiteX)) if finiteX.size else 0.0, 1.0))
    gradientScale = float(
        max(
            np.max(np.abs(finiteRhs)) if finiteRhs.size else 0.0,
            np.max(np.abs(finiteMatX)) if finiteMatX.size else 0.0,
            1.0,
        )
    )
    freeThreshold = float(max(primalTol, 1.0e-8 * signalScale, 1.0e-12))
    freeMask = x > freeThreshold
    activeMask = ~freeMask

    primalViolation = float(max(0.0, -float(np.min(x)) if x.size else 0.0))
    freeStationarityAbs = (
        float(np.max(np.abs(gradient[freeMask]))) if np.any(freeMask) else 0.0
    )
    activeDualViolationAbs = (
        float(np.max(np.maximum(-gradient[activeMask], 0.0)))
        if np.any(activeMask)
        else 0.0
    )
    complementarityAbs = (
        float(np.max(np.abs(x * gradient))) if x.size and gradient.size else 0.0
    )
    freeStationarityRel = freeStationarityAbs / gradientScale
    activeDualViolationRel = activeDualViolationAbs / gradientScale
    complementarityRel = complementarityAbs / (signalScale * gradientScale)
    kktRelMax = float(
        max(
            freeStationarityRel,
            activeDualViolationRel,
            complementarityRel,
        )
    )
    return {
        "kkt_ok": bool(primalViolation <= freeThreshold and kktRelMax <= 0.25),
        "primal_min": float(np.min(x)) if x.size else 0.0,
        "primal_violation": primalViolation,
        "free_count": int(np.sum(freeMask)),
        "active_count": int(np.sum(activeMask)),
        "free_stationarity_abs": freeStationarityAbs,
        "free_stationarity_rel": float(freeStationarityRel),
        "active_dual_violation_abs": activeDualViolationAbs,
        "active_dual_violation_rel": float(activeDualViolationRel),
        "complementarity_abs": complementarityAbs,
        "complementarity_rel": float(complementarityRel),
        "kkt_rel_max": kktRelMax,
        "gradient_scale": gradientScale,
        "free_threshold": freeThreshold,
    }


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


def centerMBInPlace(
    values: npt.NDArray[np.floating],
    *,
    intervalSizeBP: int,
    filterWindowBP: int = 5_000_000,
    centerMBMethod: str = COUNTING_DEFAULT_CENTER_MB_METHOD,
) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        tracks = arr.reshape(1, -1)
    elif arr.ndim == 2:
        tracks = arr
    else:
        raise ValueError("values must be a one- or two-dimensional array")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("values dtype must be floating point")
    if tracks.shape[0] == 0 or tracks.shape[1] == 0:
        raise ValueError("values must include at least one track and interval")

    intervalSize = int(intervalSizeBP)
    if intervalSize <= 0:
        raise ValueError("intervalSizeBP must be positive")
    filterWindowBP_ = int(filterWindowBP)
    if filterWindowBP_ <= 0:
        raise ValueError("filterWindowBP must be positive")
    filterWindowIntervals = int(math.ceil(filterWindowBP_ / float(intervalSize)))
    if filterWindowIntervals % 2 == 0:
        filterWindowIntervals += 1
    if centerMBMethod not in COUNTING_SUPPORTED_CENTER_MB_METHODS:
        supported = ", ".join(COUNTING_SUPPORTED_CENTER_MB_METHODS)
        raise ValueError(f"centerMBMethod must be one of: {supported}")

    meanTrackValBeforeCenterMB = float(np.mean(tracks))
    stdTrackValBeforeCenterMB = float(np.std(tracks))
    appliedCount = 0
    for trackIndex in range(tracks.shape[0]):
        track = tracks[trackIndex]
        if centerMBMethod == COUNTING_CENTER_MB_METHOD_MEDFILT:
            filtered = ndimage.median_filter(
                track,
                size=filterWindowIntervals,
                mode="nearest",
            )
        elif centerMBMethod == COUNTING_CENTER_MB_METHOD_SAVGOL:
            halfWindow = filterWindowIntervals // 2
            paddedTrack = np.pad(track, (halfWindow, halfWindow), mode="edge")
            cumulative = np.empty(paddedTrack.size + 1, dtype=np.float64)
            cumulative[0] = 0.0
            np.cumsum(paddedTrack, dtype=np.float64, out=cumulative[1:])
            filtered = (
                cumulative[filterWindowIntervals:] - cumulative[:-filterWindowIntervals]
            ) / float(filterWindowIntervals)
        else:
            raise ValueError(f"centerMBMethod must be one of: {supported}")
        np.subtract(track, filtered, out=track, casting="unsafe")
        appliedCount += 1
    meanTrackValAfterCenterMB = float(np.mean(tracks))
    stdTrackValAfterCenterMB = float(np.std(tracks))

    return {
        "applied": bool(appliedCount > 0),
        "applied_tracks": int(appliedCount),
        "meanTrackValBeforeCenterMB": meanTrackValBeforeCenterMB,
        "meanTrackValAfterCenterMB": meanTrackValAfterCenterMB,
        "stdTrackValBeforeCenterMB": stdTrackValBeforeCenterMB,
        "stdTrackValAfterCenterMB": stdTrackValAfterCenterMB,
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
    munc = np.asarray(matrixMunc)
    if covar.ndim != 3 or covar.shape[1] < 1 or covar.shape[2] < 1:
        raise ValueError("stateCovarForward must have shape (n, d, d)")
    if munc.ndim != 2:
        raise ValueError("matrixMunc must be two-dimensional")
    if covar.shape[0] != munc.shape[1]:
        raise ValueError("stateCovarForward and matrixMunc interval counts must match")

    p00Forward = np.maximum(covar[:, 0, 0], 0.0)
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

    gainMeans = np.full(munc.shape[0], np.nan, dtype=np.float64)
    gainMedians = np.full(munc.shape[0], np.nan, dtype=np.float64)
    gainSds = np.full(munc.shape[0], np.nan, dtype=np.float64)
    gainIqrs = np.full(munc.shape[0], np.nan, dtype=np.float64)
    gainCounts = np.zeros(munc.shape[0], dtype=np.int64)
    numerator = p00Forward * obsPrecision
    for rowIdx in range(munc.shape[0]):
        obsVariance = np.maximum(
            np.asarray(munc[rowIdx, :], dtype=np.float64) + float(pad),
            1.0e-12,
        )
        rowGains = numerator / obsVariance
        finite = np.isfinite(rowGains)
        gainCounts[rowIdx] = int(np.count_nonzero(finite))
        if gainCounts[rowIdx] == 0:
            continue
        row = rowGains[finite]
        gainMeans[rowIdx] = float(np.mean(row, dtype=np.float64))
        if row.size:
            gainMedians[rowIdx] = float(np.median(row))
            gainSds[rowIdx] = float(np.std(row))
            q25, q75 = np.quantile(row, [0.25, 0.75])
            gainIqrs[rowIdx] = float(q75 - q25)
    return {
        "mean": gainMeans,
        "median": gainMedians,
        "sd": gainSds,
        "iqr": gainIqrs,
        "count": gainCounts,
    }


def _perIntervalOutputDiagnosticTracks(
    *,
    stateCovarForward: np.ndarray,
    matrixMunc: np.ndarray,
    matrixQ0: np.ndarray,
    matrixF: np.ndarray,
    stateCovarInit: float,
    stateModel: str,
    lambdaExp: np.ndarray | None,
    processPrecExp: np.ndarray | None,
    processQScale: np.ndarray | None,
    pNoiseForward: np.ndarray | None,
    pad: float,
    obsPrecisionMultiplierMin: float,
    obsPrecisionMultiplierMax: float,
    procPrecisionMultiplierMin: float,
    procPrecisionMultiplierMax: float,
) -> dict[str, np.ndarray]:
    r"""Build per-interval diagnostic tracks aligned to emitted bedGraph rows."""

    covar = np.asarray(stateCovarForward, dtype=np.float64)
    munc = np.asarray(matrixMunc)
    q0 = np.asarray(matrixQ0, dtype=np.float64)
    f = np.asarray(matrixF, dtype=np.float64)
    stateModelMode = _normalizeStateModel(stateModel)
    stateDim = 1 if stateModelMode == STATE_MODEL_LEVEL else 2

    if covar.ndim != 3 or covar.shape[1] < stateDim or covar.shape[2] < stateDim:
        raise ValueError("stateCovarForward shape does not match stateModel")
    intervalCount = int(covar.shape[0])
    if munc.ndim != 2 or int(munc.shape[1]) != intervalCount:
        raise ValueError("matrixMunc must have shape (trackCount, intervalCount)")
    if q0.ndim != 2 or q0.shape[0] < stateDim or q0.shape[1] < stateDim:
        raise ValueError("matrixQ0 shape does not match stateModel")
    if stateDim == 2 and f.shape != (2, 2):
        raise ValueError("matrixF must have shape (2, 2) for level-trend tracks")

    obsVariance = np.maximum(munc + float(pad), 1.0e-12)
    if lambdaExp is None:
        obsPrecision = np.ones(intervalCount, dtype=np.float64)
    else:
        obsPrecision = np.asarray(lambdaExp, dtype=np.float64).reshape(-1)
        if obsPrecision.shape != (intervalCount,):
            raise ValueError("lambdaExp length must match interval count")
        obsPrecision = np.clip(
            obsPrecision,
            float(obsPrecisionMultiplierMin),
            float(obsPrecisionMultiplierMax),
        )
    obsPrecision = np.maximum(obsPrecision, np.finfo(np.float64).tiny)

    muncTrace = np.zeros(intervalCount, dtype=np.float64)
    sumInvR = np.zeros(intervalCount, dtype=np.float64)
    for rowIndex in range(int(munc.shape[0])):
        obsVariance = np.maximum(
            np.asarray(munc[rowIndex, :], dtype=np.float64) + float(pad),
            1.0e-12,
        )
        effectiveObsVariance = obsVariance / obsPrecision
        finiteEffectiveObsVariance = np.isfinite(effectiveObsVariance)
        muncTrace[finiteEffectiveObsVariance] += effectiveObsVariance[
            finiteEffectiveObsVariance
        ]
        invObsVariance = obsPrecision / obsVariance
        finiteInvObsVariance = np.isfinite(invObsVariance)
        sumInvR[finiteInvObsVariance] += invObsVariance[finiteInvObsVariance]

    qTracks = _processQTrackArrays(
        matrixQ0=q0,
        intervalCount=intervalCount,
        stateModel=stateModelMode,
        processPrecExp=processPrecExp,
        processQScale=processQScale,
        pNoiseForward=pNoiseForward,
        procPrecisionMultiplierMin=float(procPrecisionMultiplierMin),
        procPrecisionMultiplierMax=float(procPrecisionMultiplierMax),
        returnFullQ=False,
    )
    preKappaQLevel = qTracks["preKappaQLevel"]
    preKappaQTrend = qTracks["preKappaQTrend"]
    effectiveQLevel = qTracks["effectiveQLevel"]
    effectiveQTrend = qTracks["effectiveQTrend"]
    sumGain0 = np.zeros(intervalCount, dtype=np.float64)
    sumGain1 = np.zeros(intervalCount, dtype=np.float64)
    previousCovar = np.eye(stateDim, dtype=np.float64) * float(stateCovarInit)
    baseQ = q0[:stateDim, :stateDim]
    qScale = qTracks["puncQScale"]
    procPrecision = None
    if processPrecExp is not None:
        procPrecision = np.asarray(processPrecExp, dtype=np.float64).reshape(-1)
        procPrecision = np.clip(
            procPrecision,
            float(procPrecisionMultiplierMin),
            float(procPrecisionMultiplierMax),
        )
        procPrecision = np.maximum(procPrecision, np.finfo(np.float64).tiny)
    pNoise = (
        None
        if pNoiseForward is None or procPrecision is not None
        else np.asarray(pNoiseForward)
    )
    qEff = np.empty((stateDim, stateDim), dtype=np.float64)

    for k in range(intervalCount):
        if pNoise is not None and k > 0:
            pNoiseEff = pNoise[k - 1, :stateDim, :stateDim]
            if np.all(np.isfinite(pNoiseEff)):
                qEff[:, :] = pNoiseEff
            else:
                qEff[:, :] = baseQ * float(qScale[k])
        else:
            qEff[:, :] = baseQ * float(qScale[k])
            if procPrecision is not None:
                qEff[:, :] /= float(procPrecision[k])
        if stateDim == 2:
            predCovar = f @ previousCovar @ f.T + qEff
            pred10 = float(predCovar[1, 0])
        else:
            predCovar = previousCovar + qEff
            pred10 = 0.0

        pred00 = max(float(predCovar[0, 0]), 0.0)
        denom = 1.0 + pred00 * float(sumInvR[k])
        if np.isfinite(denom) and denom > 0.0:
            gainScale = float(sumInvR[k]) / denom
            sumGain0[k] = pred00 * gainScale
            sumGain1[k] = pred10 * gainScale

        previousCovar = np.asarray(
            covar[k, :stateDim, :stateDim],
            dtype=np.float64,
        )

    return {
        "baseQLevel": qTracks["baseQLevel"].astype(np.float32, copy=False),
        "baseQTrend": qTracks["baseQTrend"].astype(np.float32, copy=False),
        "preKappaQLevel": preKappaQLevel.astype(np.float32, copy=False),
        "preKappaQTrend": preKappaQTrend.astype(np.float32, copy=False),
        "effectiveQLevel": effectiveQLevel.astype(np.float32, copy=False),
        "effectiveQTrend": effectiveQTrend.astype(np.float32, copy=False),
        "puncQScale": qTracks["puncQScale"].astype(np.float32, copy=False),
        "muncTrace": muncTrace.astype(np.float32, copy=False),
        "sumGain0": sumGain0.astype(np.float32, copy=False),
        "sumGain1": sumGain1.astype(np.float32, copy=False),
    }


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


class _MuncRuntimeSizing(NamedTuple):
    trendBlockSizeBP: int
    trendBlockIntervals: int
    trendBlockSource: str
    localWindowSizeBP: int
    localWindowIntervals: int
    localWindowSource: str
    usedDependenceSpan: bool
    dependenceSpanIntervals: int | None


def _normalizeMuncVarianceModel(model: str | int | None) -> str:
    if model is None:
        return OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL
    if isinstance(model, (int, np.integer)):
        modelCode = int(model)
        if modelCode == MUNC_VARIANCE_MODEL_CODE_KALMAN:
            return MUNC_VARIANCE_MODEL_KALMAN
    modelName = str(model).strip().lower().replace("-", "").replace("_", "")
    if modelName in {"kalman", "smoother", "kf"}:
        return MUNC_VARIANCE_MODEL_KALMAN
    supportedModels = ", ".join(MUNC_SUPPORTED_VARIANCE_MODELS)
    raise ValueError(
        f"unsupported MUNC variance model {model!r}; expected {supportedModels}"
    )


def _normalizeMuncEBPriorGUncertaintyMode(mode: str | None) -> str:
    if mode is None:
        return OBSERVATION_DEFAULT_MUNC_EB_PRIOR_G_UNCERTAINTY_MODE
    key = str(mode).strip().lower().replace("-", "").replace("_", "")
    if key == "diagonal":
        return MUNC_EB_PRIOR_G_UNCERTAINTY_MODE_PROXY
    canonicalByKey = {
        item.replace("-", "").replace("_", "").lower(): item
        for item in MUNC_SUPPORTED_EB_PRIOR_G_UNCERTAINTY_MODES
    }
    if key not in canonicalByKey:
        supported = ", ".join(MUNC_SUPPORTED_EB_PRIOR_G_UNCERTAINTY_MODES)
        raise ValueError(
            f"unsupported MUNC EB prior g uncertainty mode {mode!r}; expected {supported}"
        )
    return canonicalByKey[key]


def _resolveMuncRuntimeSizing(
    *,
    intervalSizeBP: int,
    dependenceSpanIntervals: int | None = None,
    muncTrendBlockSizeBP: int | None = None,
    muncLocalWindowSizeBP: int | None = None,
    muncTrendBlockDependenceMultiplier: float | None = (
        OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER
    ),
    muncLocalWindowDependenceMultiplier: float | None = (
        OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER
    ),
) -> _MuncRuntimeSizing:
    intervalSizeBP_ = max(1, int(intervalSizeBP))
    dependenceIntervals = (
        None if dependenceSpanIntervals is None else int(dependenceSpanIntervals)
    )
    if dependenceIntervals is not None and dependenceIntervals <= 0:
        dependenceIntervals = None

    defaultTrendIntervals = 11 * 3
    defaultLocalIntervals = max(4, defaultTrendIntervals + 1)

    def resolveMultiplier(value: float | None, defaultValue: float) -> float:
        multiplier = defaultValue if value is None else float(value)
        if (not np.isfinite(multiplier)) or multiplier <= 0.0:
            raise ValueError(
                "MUNC dependence multipliers must be positive finite values"
            )
        return float(multiplier)

    trendMultiplier = resolveMultiplier(
        muncTrendBlockDependenceMultiplier,
        OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER,
    )
    localMultiplier = resolveMultiplier(
        muncLocalWindowDependenceMultiplier,
        OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER,
    )

    def resolveDependenceIntervals(multiplier: float, minIntervals: int) -> int | None:
        if dependenceIntervals is None:
            return None
        return max(
            int(minIntervals),
            int(math.ceil(float(multiplier) * float(dependenceIntervals))),
        )

    def resolveExplicitIntervals(value: int | None) -> int | None:
        if value is None:
            return None
        valueBP = int(value)
        if valueBP <= 0:
            return None
        return max(1, int(valueBP / intervalSizeBP_))

    trendIntervals = resolveExplicitIntervals(muncTrendBlockSizeBP)
    usedDependenceSpan = False
    trendSource = "explicit bp"
    if trendIntervals is None:
        trendIntervals = resolveDependenceIntervals(trendMultiplier, 1)
        usedDependenceSpan = trendIntervals is not None
        if trendIntervals is not None:
            trendSource = "correlation length"
        if trendIntervals is None:
            trendIntervals = int(defaultTrendIntervals)
            trendSource = "fallback default"

    localIntervals = resolveExplicitIntervals(muncLocalWindowSizeBP)
    localSource = "explicit bp"
    if localIntervals is None:
        localIntervals = resolveDependenceIntervals(localMultiplier, 4)
        usedDependenceSpan = usedDependenceSpan or localIntervals is not None
        if localIntervals is not None:
            localSource = "correlation length"
        if localIntervals is None:
            localIntervals = int(defaultLocalIntervals)
            localSource = "fallback default"

    trendIntervals = max(1, int(trendIntervals))
    localIntervals = max(4, int(localIntervals))
    return _MuncRuntimeSizing(
        trendBlockSizeBP=int(trendIntervals * intervalSizeBP_),
        trendBlockIntervals=int(trendIntervals),
        trendBlockSource=trendSource,
        localWindowSizeBP=int(localIntervals * intervalSizeBP_),
        localWindowIntervals=int(localIntervals),
        localWindowSource=localSource,
        usedDependenceSpan=bool(usedDependenceSpan),
        dependenceSpanIntervals=dependenceIntervals,
    )


def _muncSizingNeedsDependence(
    *,
    muncTrendBlockSizeBP: int | None,
    muncLocalWindowSizeBP: int | None,
) -> bool:
    trendNeedsAuto = muncTrendBlockSizeBP is None or int(muncTrendBlockSizeBP) <= 0
    localNeedsAuto = muncLocalWindowSizeBP is None or int(muncLocalWindowSizeBP) <= 0
    return bool(trendNeedsAuto or localNeedsAuto)


def solveZeroCenteredBackground(
    residualMatrix: np.ndarray,
    invVarMatrix: np.ndarray,
    blockLenIntervals: int,
    backgroundSmoothness: float = 1.0,
    zeroCenter: bool = False,
    useNonnegative: bool = True,
    backgroundNegativePenaltyMultiplier: float | None = (
        FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER
    ),
    initialBackground: np.ndarray | None = None,
    weightTrack: np.ndarray | None = None,
    rhsTrack: np.ndarray | None = None,
) -> npt.NDArray[np.float32]:
    r"""Estimate a positional background with asymmetric IRLS and strong first/second-order difference penalties (to prevent conflation with signal).


    Parameters
    ----------
    residualMatrix : np.ndarray
        The residual matrix after the inner/fixed-background phase of the ECM
    invVarMatrix : np.ndarray
        Precision matrix corresponding to the residual matrix
    blockLenIntervals : int
        The block length in intervals.
    backgroundSmoothness : float, optional
        Linear multiplier for first/second-order penalties.
    zeroCenter : bool, optional
        Whether to zero-center the background, by default False. This feature is an artifact and should maybe not be used, ever. It does not affect signal-background conflation appreciably.
    useNonnegative : bool
        Use an asymmetric penalty on negative background values (note this is, in fact, a soft, not hard constraint)

    Returns
    -------
    npt.NDArray[np.float32]
        The estimated background :math:`g(i)^k, i=1,\dots,n`.
    """
    residualArr = np.asarray(residualMatrix, dtype=np.float32)
    invVarArr = np.asarray(invVarMatrix, dtype=np.float32)
    if residualArr.ndim != 2 or invVarArr.shape != residualArr.shape:
        raise ValueError(
            "residualMatrix and invVarMatrix must have identical 2D shapes"
        )

    intervalCount = int(residualArr.shape[1])
    if intervalCount < 1:
        return np.zeros(0, dtype=np.float32)

    if weightTrack is not None or rhsTrack is not None:
        if weightTrack is None or rhsTrack is None:
            raise ValueError("weightTrack and rhsTrack must be supplied together")
        weightTrack = np.asarray(weightTrack, dtype=np.float64).reshape(-1)
        rhsTrack = np.asarray(rhsTrack, dtype=np.float64).reshape(-1)
        if weightTrack.shape[0] != intervalCount or rhsTrack.shape[0] != intervalCount:
            raise ValueError(
                "weightTrack and rhsTrack length must match interval count"
            )
        positiveSupportCount = int(np.count_nonzero(weightTrack > 0.0))
    else:
        weightTrack, rhsTrack, positiveSupportCount = (
            cconsenrich.cbackgroundWeightedStatsWithSupport(
                residualArr,
                invVarArr,
            )
        )
    if positiveSupportCount <= 0:
        return np.zeros(intervalCount, dtype=np.float32)

    # set penalties based on calculated 'span'/correlation-length
    lamFirst, lamSecond = _backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=blockLenIntervals,
        backgroundSmoothness=backgroundSmoothness,
    )
    weightTrack = np.ascontiguousarray(weightTrack, dtype=np.float64)
    rhsTrack = np.ascontiguousarray(rhsTrack, dtype=np.float64)

    if useNonnegative:
        return _solveNonnegativeBackground(
            weightTrack=weightTrack,
            rhsTrack=rhsTrack,
            lamFirst=float(lamFirst),
            lamSecond=float(lamSecond),
            zeroCenter=bool(zeroCenter),
            backgroundNegativePenaltyMultiplier=backgroundNegativePenaltyMultiplier,
            initialBackground=initialBackground,
        )

    if zeroCenter:
        if intervalCount == 1:
            return np.zeros(1, dtype=np.float32)
        return solveZeroCenteredBackgroundLinearSystem(
            weightTrack,
            rhsTrack,
            float(lamFirst),
            float(lamSecond),
        ).astype(np.float32)

    return np.asarray(
        _solveBackgroundLinearSystem(
            weightTrack,
            rhsTrack,
            float(lamFirst),
            float(lamSecond),
        ),
        dtype=np.float32,
    )


def _solveClippedBackgroundHeuristic(
    residualMatrix: np.ndarray,
    invVarMatrix: np.ndarray,
    blockLenIntervals: int,
    backgroundSmoothness: float = 1.0,
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

    weightTrack, rhsTrack, positiveSupportCount = (
        cconsenrich.cbackgroundWeightedStatsWithSupport(
            residualArr,
            invVarArr,
        )
    )
    if int(positiveSupportCount) <= 0:
        return np.zeros(intervalCount, dtype=np.float32)

    lamFirst, lamSecond = _backgroundPenaltyWeightsFromSpan(
        blockLenIntervals=blockLenIntervals,
        backgroundSmoothness=backgroundSmoothness,
    )
    background = _solveBackgroundLinearSystem(
        np.ascontiguousarray(weightTrack, dtype=np.float64),
        np.ascontiguousarray(rhsTrack, dtype=np.float64),
        float(lamFirst),
        float(lamSecond),
    )
    return np.maximum(background, 0.0).astype(np.float32)


def _solveNonnegativeBackground(
    *,
    weightTrack: np.ndarray,
    rhsTrack: np.ndarray,
    lamFirst: float,
    lamSecond: float,
    zeroCenter: bool,
    backgroundNegativePenaltyMultiplier: float | None,
    initialBackground: np.ndarray | None = None,
) -> npt.NDArray[np.float32]:
    n = int(rhsTrack.shape[0])
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    def solveWithWeights(weightTrackLocal: np.ndarray) -> np.ndarray:
        if bool(zeroCenter):
            if n == 1:
                return np.zeros(1, dtype=np.float64)
            return solveZeroCenteredBackgroundLinearSystem(
                np.ascontiguousarray(weightTrackLocal, dtype=np.float64),
                rhsTrack,
                float(lamFirst),
                float(lamSecond),
            )
        return np.asarray(
            _solveBackgroundLinearSystem(
                np.ascontiguousarray(weightTrackLocal, dtype=np.float64),
                rhsTrack,
                float(lamFirst),
                float(lamSecond),
            ),
            dtype=np.float64,
        )

    if backgroundNegativePenaltyMultiplier is None:
        background = np.asarray(
            solveWithWeights(weightTrack),
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(background)):
            raise RuntimeError("solver returned non-finite values")
        return background.astype(np.float32)

    negativePenaltyMultiplier = float(backgroundNegativePenaltyMultiplier)
    if not np.isfinite(negativePenaltyMultiplier) or negativePenaltyMultiplier <= 0.0:
        background = np.asarray(
            solveWithWeights(weightTrack),
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(background)):
            raise RuntimeError("solver returned non-finite values")
        return background.astype(np.float32)

    positiveWeightTrack = np.asarray(weightTrack, dtype=np.float64)
    positiveWeightTrack = positiveWeightTrack[
        np.isfinite(positiveWeightTrack) & (positiveWeightTrack > 0.0)
    ]
    weightScale = (
        float(np.median(positiveWeightTrack)) if positiveWeightTrack.size else 1.0
    )
    if not np.isfinite(weightScale) or weightScale <= 0.0:
        weightScale = 1.0
    negativePenaltyWeight = float(negativePenaltyMultiplier * weightScale)
    if not np.isfinite(negativePenaltyWeight) or negativePenaltyWeight <= 0.0:
        background = np.asarray(
            solveWithWeights(weightTrack),
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(background)):
            raise RuntimeError("solver returned non-finite values")
        return background.astype(np.float32)

    previousNegativeMask: np.ndarray | None = None
    if initialBackground is not None:
        initialBackgroundArr = np.asarray(
            initialBackground,
            dtype=np.float64,
        ).reshape(-1)
        if initialBackgroundArr.shape[0] != n:
            raise ValueError("initialBackground length must match interval count")
        previousNegativeMask = np.asarray(initialBackgroundArr < 0.0, dtype=bool)
        adjustedWeightTrack = np.asarray(weightTrack, dtype=np.float64).copy()
        adjustedWeightTrack[previousNegativeMask] += negativePenaltyWeight
        background = np.asarray(
            solveWithWeights(adjustedWeightTrack),
            dtype=np.float64,
        ).reshape(-1)
    else:
        background = np.asarray(
            solveWithWeights(weightTrack),
            dtype=np.float64,
        ).reshape(-1)
    if not np.all(np.isfinite(background)):
        raise RuntimeError("solver returned non-finite values")

    initialNegativeFraction = float(np.mean(background < 0.0)) if n else 0.0
    passCount = 0
    maxPasses = 5
    solverStart = time.perf_counter()
    for passIndex in range(maxPasses):
        negativeMask = np.asarray(background < 0.0, dtype=bool)
        if previousNegativeMask is not None and np.array_equal(
            negativeMask,
            previousNegativeMask,
        ):
            break
        if not np.any(negativeMask):
            break
        previousNegativeMask = negativeMask.copy()
        adjustedWeightTrack = np.asarray(weightTrack, dtype=np.float64).copy()
        adjustedWeightTrack[negativeMask] += negativePenaltyWeight
        background = np.asarray(
            solveWithWeights(adjustedWeightTrack),
            dtype=np.float64,
        ).reshape(-1)
        if not np.all(np.isfinite(background)):
            raise RuntimeError("solver returned non-finite values")
        passCount = int(passIndex + 1)
    elapsed = time.perf_counter() - solverStart
    finalNegativeFraction = float(np.mean(background < 0.0)) if n else 0.0
    negativePenaltyValue = (
        0.5
        * negativePenaltyWeight
        * float(np.sum(np.minimum(background, 0.0) ** 2, dtype=np.float64))
    )
    logger.debug(
        "backgroundIRLS: solver=asymmetric_pentadiagonal intervals=%d "
        "penaltyMultiplier=%.6g penaltyWeight=%.6g passes=%d/%d elapsed=%.3fs "
        "min=%.6g negativeFractionBefore=%.4f negativeFractionAfter=%.4f "
        "negativePenalty=%.6g zeroCenter=%s",
        int(n),
        float(negativePenaltyMultiplier),
        float(negativePenaltyWeight),
        int(passCount),
        int(maxPasses),
        float(elapsed),
        float(np.min(background)) if background.size else 0.0,
        float(initialNegativeFraction),
        float(finalNegativeFraction),
        float(negativePenaltyValue),
        str(bool(zeroCenter)),
    )
    return background.astype(np.float32)


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
    muncTrendBlockSizeBP: int | None = None,
    muncLocalWindowSizeBP: int | None = None,
    dependenceSpanIntervals: int | None = None,
    muncTrendBlockDependenceMultiplier: float | None = (
        OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER
    ),
    muncLocalWindowDependenceMultiplier: float | None = (
        OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER
    ),
    muncVarianceModel: str | int | None = OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL,
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
    restrictLocalVarianceToSparseBed: bool = (
        OBSERVATION_DEFAULT_RESTRICT_LOCAL_VARIANCE_TO_SPARSE_BED
    ),
    EB_localQuantile: float = 0.0,
    verbose: bool = False,
    eps: float = 1.0e-6,
    varianceFloor: float | None = None,
    varianceCap: float | None = None,
    intervalsArr: Optional[np.ndarray] = None,
    excludeMaskArr: Optional[np.ndarray] = None,
    pooledTrend: Optional[PSplineLogVarianceTrend] = None,
    priorMeanTrack: Optional[np.ndarray] = None,
    replicateVarianceFactor: float = 1.0,
    EB_pooledNu0: float | None = None,
    covariateTrack: Optional[np.ndarray] = None,
    additiveCovariateModel: Optional[MuncAdditiveCovariateModel] = None,
    replicateIndex: int | None = None,
    sampleFile: str | None = None,
    countModelVarianceFloor: Optional[np.ndarray] = None,
    localVarianceTrack: Optional[np.ndarray] = None,
    priorVarianceTrack: Optional[np.ndarray] = None,
    EB_effectiveNuL: float | None = None,
) -> tuple[npt.NDArray[np.float32], float]:
    r"""Approximate initial sample-specific (**M**)easurement (**unc**)ertainty tracks

    For an individual experimental sample (replicate), quantify *positional* observation noise levels over genomic intervals :math:`i=1,2,\ldots n` spanning ``chromosome``.
    These tracks (per-sample) comprise the ``matrixMunc`` input to :func:`runConsenrich`, :math:`\mathbf{R}[:,:] \in \mathbb{R}^{m \times n}`.

    Variance is modeled as a function of a signed mean signal predictor. For ``EB_use=True``, local variance estimates are shrunk toward a signal level dependent global variance fit.
    If ``countModelVarianceFloor`` is supplied, it must be a precomputed
    per-interval transformed-scale count-noise variance for this replicate.
    The fitted MUNC track is bounded below by finite count-noise entries.

    :param chromosome: chromosome/contig name
    :type chromosome: str
    :param values: normalized/transformed signal measurements over genomic intervals (e.g., :func:`consenrich.cconsenrich.cTransform` output)
    :type values: np.ndarray
    :param intervals: genomic intervals positions (start positions)
    :type intervals: np.ndarray
    :param countModelVarianceFloor: Optional transformed-scale observation
        count-noise variance for this replicate. Finite entries bound the
        final fitted MUNC track from below.
    :type countModelVarianceFloor: np.ndarray | None

    See :class:`consenrich.core.observationParams` for other parameters.

    """

    _ = excludeFitCoefs

    varianceFloor_ = float(max(eps, varianceFloor or eps, 1.0e-12))
    varianceCap_ = (
        None
        if varianceCap is None
        or (not np.isfinite(float(varianceCap)))
        or float(varianceCap) <= varianceFloor_
        else float(varianceCap)
    )
    varianceCapForKernel = (
        float(varianceCap_)
        if varianceCap_ is not None
        else float(np.finfo(np.float32).max)
    )
    muncVarianceModelName = _normalizeMuncVarianceModel(muncVarianceModel)
    replicateFactor = float(replicateVarianceFactor)
    if not np.isfinite(replicateFactor) or abs(replicateFactor - 1.0) > 1.0e-8:
        raise ValueError("replicateVarianceFactor is not supported for MUNC priors")
    sizing = _resolveMuncRuntimeSizing(
        intervalSizeBP=intervalSizeBP,
        dependenceSpanIntervals=dependenceSpanIntervals,
        muncTrendBlockSizeBP=muncTrendBlockSizeBP,
        muncLocalWindowSizeBP=muncLocalWindowSizeBP,
        muncTrendBlockDependenceMultiplier=muncTrendBlockDependenceMultiplier,
        muncLocalWindowDependenceMultiplier=muncLocalWindowDependenceMultiplier,
    )
    blockSizeIntervals = int(sizing.trendBlockIntervals)
    localWindowIntervals = int(sizing.localWindowIntervals)
    restrictLocalVariance = bool(restrictLocalVarianceToSparseBed)
    if intervalsArr is None:
        intervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32).reshape(-1)
    else:
        intervalsArr = np.ascontiguousarray(intervalsArr, dtype=np.uint32).reshape(-1)
    valuesArr = np.ascontiguousarray(values, dtype=np.float32)
    if intervalsArr.shape[0] != valuesArr.size:
        raise ValueError("intervalsArr must match values length")
    sampleFileText = None if sampleFile is None else str(sampleFile)[:7]
    if sampleFileText == "":
        raise ValueError("sampleFile must be non-empty when provided")
    sampleFileRows = (
        (("sample file", sampleFileText),) if sampleFileText is not None else ()
    )
    sampleFileLog = "NA" if sampleFileText is None else sampleFileText
    countModelVarianceFloorArr = _coerceMuncCountModelVarianceFloor(
        countModelVarianceFloor,
        valuesArr.size,
    )
    if localVarianceTrack is None:
        raise ValueError("kalman MUNC requires localVarianceTrack")
    obsVarTrack = np.ascontiguousarray(localVarianceTrack, dtype=np.float32).reshape(-1)
    if obsVarTrack.shape[0] != valuesArr.size:
        raise ValueError("localVarianceTrack must match values length")
    obsVarTrack, localFinalizeDiagnostics = cconsenrich.cFinalizeMuncEBTrack(
        obsVarTrack,
        useEB=False,
        varianceFloor=float(varianceFloor_),
        varianceCap=float(varianceCapForKernel),
    )
    supportFraction = float(localFinalizeDiagnostics["supportFraction"])

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
        "MUNC track parameters",
        (
            ("chromosome", chromosome),
            *sampleFileRows,
            ("intervals", int(valuesArr.size)),
            ("interval size bp", int(intervalSizeBP)),
            ("MUNC variance model", muncVarianceModelName),
            ("MUNC trend block bp", int(sizing.trendBlockSizeBP)),
            ("sampling block intervals", int(blockSizeIntervals)),
            ("MUNC local window bp", int(sizing.localWindowSizeBP)),
            ("local window intervals", int(localWindowIntervals)),
            (
                "MUNC correlation length",
                (
                    "NA"
                    if sizing.dependenceSpanIntervals is None
                    else int(sizing.dependenceSpanIntervals)
                ),
            ),
            (
                "trend correlation length multiplier",
                float(
                    OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER
                    if muncTrendBlockDependenceMultiplier is None
                    else muncTrendBlockDependenceMultiplier
                ),
            ),
            (
                "local correlation length multiplier",
                float(
                    OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER
                    if muncLocalWindowDependenceMultiplier is None
                    else muncLocalWindowDependenceMultiplier
                ),
            ),
            ("MUNC variance EB", "enabled" if EB_use else "disabled"),
            (
                "MUNC delta-method variance",
                "not used",
            ),
            (
                "MUNC trend source",
                (
                    "pooled signed trend"
                    if pooledTrend is not None
                    else "fit sample trend"
                ),
            ),
            (
                "MUNC genomic covariates",
                (
                    "enabled"
                    if additiveCovariateModel is not None and covariateTrack is not None
                    else "disabled"
                ),
            ),
            (
                "MUNC count model floor",
                "enabled" if countModelVarianceFloorArr is not None else "disabled",
            ),
        ),
        level=logging.DEBUG,
    )

    if int(numNearest) > 0 or sparseIntervalIndices is not None:
        raise ValueError("sparse-nearest MUNC is not supported by kalman MUNC")
    if bool(restrictLocalVariance):
        raise ValueError(
            "restrictLocalVarianceToSparseBed is not supported by kalman MUNC"
        )

    if pooledTrend is None:
        if EB_use:
            raise ValueError("kalman MUNC EB requires a pooled MUNC trend")
        means_Sorted = np.asarray(valuesArr, dtype=np.float64).ravel()
        opt = None
    else:
        _logAsciiBlock(
            "MUNC pooled trend reuse",
            (
                ("chromosome", chromosome),
                *sampleFileRows,
                ("intervals", int(valuesArr.size)),
            ),
        )
        opt = pooledTrend
        finiteValues = np.asarray(valuesArr, dtype=np.float64).ravel()
        means_Sorted = finiteValues[np.isfinite(finiteValues)]
        if means_Sorted.size == 0:
            means_Sorted = np.array([0.0], dtype=np.float64)
        if priorVarianceTrack is None:
            logger.info(
                _formatPSplineTrendSummary(
                    opt,
                    means_Sorted,
                    eps=varianceFloor_,
                    maxVariance=varianceCap_,
                    sampleFile=sampleFileText,
                )
            )

    if priorMeanTrack is None:
        meanTrack = np.ascontiguousarray(valuesArr, dtype=np.float32)
    else:
        meanTrack = np.ascontiguousarray(priorMeanTrack, dtype=np.float32).reshape(-1)
        if meanTrack.shape[0] != valuesArr.size:
            raise ValueError("priorMeanTrack must match values length")
    if useEMA and priorMeanTrack is None:
        meanTrack = cconsenrich.cEMA(meanTrack, 2 / (localWindowIntervals + 1))
    if not EB_use:
        _logAsciiBlock(
            "MUNC EB shrinkage skipped",
            (
                ("chromosome", chromosome),
                *sampleFileRows,
                ("reason", "MUNC variance EB disabled"),
                ("support fraction", float(supportFraction)),
            ),
        )
        posteriorVarTrack, finalizeDiagnostics = cconsenrich.cFinalizeMuncEBTrack(
            obsVarTrack,
            countFloor=countModelVarianceFloorArr,
            useEB=False,
            varianceFloor=float(varianceFloor_),
            varianceCap=float(varianceCapForKernel),
        )
        return posteriorVarTrack.astype(np.float32, copy=False), float(
            finalizeDiagnostics["supportFraction"]
        )
    if opt is None:
        raise ValueError("kalman MUNC EB requires a pooled MUNC trend")
    if priorVarianceTrack is None:
        priorTrack = evalPSplineLogVarianceTrend(
            opt,
            meanTrack,
            eps=varianceFloor_,
            maxVariance=varianceCap_,
        )
    else:
        priorTrack = np.ascontiguousarray(
            priorVarianceTrack,
            dtype=np.float32,
        ).reshape(-1)
        if priorTrack.shape[0] != valuesArr.size:
            raise ValueError("priorVarianceTrack must match values length")
    if additiveCovariateModel is not None and covariateTrack is not None:
        additionalTrack = evalMuncAdditiveCovariateModel(
            additiveCovariateModel,
            meanTrack,
            covariateTrack,
            replicateIndex,
        ).astype(np.float64, copy=False)
        priorTrack = (
            np.asarray(
                priorTrack,
                dtype=np.float64,
            ).reshape(-1)
            + additionalTrack
        )
        finiteAdditional = additionalTrack[np.isfinite(additionalTrack)]
        if finiteAdditional.size:
            logger.info(
                "MUNC additive genomic covariate variance: replicate=%s "
                "sampleFile=%s active_fraction=%.4f median=%.4g q95=%.4g",
                "pooled" if replicateIndex is None else int(replicateIndex),
                sampleFileLog,
                float(np.count_nonzero(finiteAdditional > 0.0))
                / float(finiteAdditional.size),
                float(np.median(finiteAdditional)),
                float(np.quantile(finiteAdditional, 0.95)),
            )
    priorTrack, _ = cconsenrich.cFinalizeMuncEBTrack(
        priorTrack,
        useEB=False,
        varianceFloor=float(varianceFloor_),
        varianceCap=float(varianceCapForKernel),
    )

    if EB_setNuL is not None and EB_setNuL > 3:
        Nu_L = float(EB_setNuL)
        logger.info(
            "Using fixed/specified Nu_L=%.2f sampleFile=%s",
            Nu_L,
            sampleFileLog,
        )
    elif EB_effectiveNuL is not None:
        Nu_L = float(EB_effectiveNuL)
        if not np.isfinite(Nu_L) or Nu_L < 4.0:
            raise ValueError("EB_effectiveNuL must be finite and at least 4.0")
    else:
        Nu_L = float(max(4, localWindowIntervals - 3))

    # --- Determine prior strength ---
    specifiedNu0 = _coerceEBPriorStrength(EB_setNu0)
    pooledNu0 = _coerceEBPriorStrength(EB_pooledNu0)
    medPrior = float(np.median(priorTrack)) if priorTrack.size else 0.0
    medObs = float(np.median(obsVarTrack)) if obsVarTrack.size else 0.0

    nu0MinPrior = (1.0e-2 * medPrior) + 1.0e-4
    nu0MinObs = (1.0e-2 * medObs) + 1.0e-4

    nu0LocalEvidence = obsVarTrack > nu0MinObs
    nu0PriorEvidence = priorTrack > nu0MinPrior
    nu0Evidence = nu0LocalEvidence & nu0PriorEvidence

    if specifiedNu0 is not None:
        # check if Nu_0 is specified before computing
        Nu_0 = specifiedNu0
        logger.info(
            "Using fixed/specified Nu_0=%.2f sampleFile=%s",
            Nu_0,
            sampleFileLog,
        )
    elif pooledNu0 is not None:
        Nu_0 = pooledNu0
        logger.info(
            "Using pooled Nu_0=%.2f sampleFile=%s",
            Nu_0,
            sampleFileLog,
        )
    else:
        # only pass matched finite pairs into EB_computePriorStrength
        if np.count_nonzero(nu0Evidence) < 4:
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
                candidateMask=nu0Evidence,
            )

    Nu_0_cap = 50.0 * float(Nu_L)
    if np.isfinite(Nu_0_cap) and Nu_0 > Nu_0_cap:
        logger.info(
            "Capping Nu_0=%.2f at 50*Nu_L=%.2f sampleFile=%s",
            float(Nu_0),
            float(Nu_0_cap),
            sampleFileLog,
        )
        Nu_0 = float(Nu_0_cap)

    _logAsciiBlock(
        "MUNC EB shrinkage",
        (
            ("chromosome", chromosome),
            *sampleFileRows,
            ("Nu_0", float(Nu_0)),
            ("Nu_L", float(Nu_L)),
            ("posterior sample size", float(Nu_L + Nu_0)),
            ("support fraction", float(supportFraction)),
            ("MUNC variance model", muncVarianceModelName),
        ),
    )
    logger.info(
        "MUNC EB shrinkage: sampleFile=%s\n\tNu_0=%.2f\n\tNu_L=%.2f",
        sampleFileLog,
        Nu_0,
        Nu_L,
    )
    posteriorSampleSize: float = Nu_L + Nu_0
    if not np.isfinite(posteriorSampleSize) or posteriorSampleSize <= 0.0:
        raise ValueError(
            f"MUNC EB posterior sample size is invalid on {chromosome}: {posteriorSampleSize}"
        )

    posteriorVarTrack, finalizeDiagnostics = cconsenrich.cFinalizeMuncEBTrack(
        obsVarTrack,
        priorVarianceTrack=priorTrack,
        countFloor=countModelVarianceFloorArr,
        nuLocal=float(Nu_L),
        nuPrior=float(Nu_0),
        useEB=True,
        varianceFloor=float(varianceFloor_),
        varianceCap=float(varianceCapForKernel),
    )
    supportFraction = float(finalizeDiagnostics["supportFraction"])
    finalShrinkagePairFraction = float(
        finalizeDiagnostics["finalShrinkagePairFraction"]
    )
    countFloorAddedCount = int(finalizeDiagnostics["countFloorAddedCount"])

    logger.info(
        "MUNC EB evidence: chromosome=%s sampleFile=%s localAboveFloorFraction=%.6g "
        "nu0PairFraction=%.6g finalShrinkagePairFraction=%.6g "
        "countFloorAddedCount=%d",
        chromosome,
        sampleFileLog,
        float(supportFraction),
        (
            float(np.count_nonzero(nu0Evidence) / nu0Evidence.size)
            if nu0Evidence.size
            else 0.0
        ),
        float(finalShrinkagePairFraction),
        countFloorAddedCount,
    )

    logger.info(
        _formatMuncVarianceDiagnostics(
            obsVarTrack,
            priorTrack,
            posteriorVarTrack,
            np.abs(means_Sorted),
            countModelVarianceFloorArr,
            sampleFile=sampleFileText,
        )
    )

    if verbose:
        logger.info(
            "Median variance after shrinkage: %.4f sampleFile=%s",
            float(np.nanmedian(posteriorVarTrack)),
            sampleFileLog,
        )

    return posteriorVarTrack.astype(np.float32, copy=False), float(supportFraction)


def _computePriorStrengthFromCandidateIdx(
    localModelVariancesArr: np.ndarray,
    globalModelVariancesArr: np.ndarray,
    Nu_local: float,
    candidateIdx: np.ndarray,
    localLogVarianceNoiseArr: np.ndarray | None = None,
) -> float:
    (
        logVarRatioArr,
        noiseSelected,
    ) = cconsenrich.cEBPriorStrengthLogRatiosFromCandidateIdx(
        localModelVariancesArr,
        globalModelVariancesArr,
        candidateIdx,
        localLogVarianceNoiseArr,
    )
    if logVarRatioArr.size < 4:
        logger.warning(
            f"After masking, insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
        )
        return float(1.0e6)

    if logVarRatioArr.size >= 20:
        tail = float(OBSERVATION_DEFAULT_MUNC_EB_PRIOR_STRENGTH_WINSOR_TAIL)
        clipSmall = np.quantile(logVarRatioArr, tail)
        clipBig = np.quantile(logVarRatioArr, 1.0 - tail)
        np.clip(logVarRatioArr, clipSmall, clipBig, out=logVarRatioArr)

    varLogVarRatio = float(np.var(logVarRatioArr, ddof=1))
    if localLogVarianceNoiseArr is None:
        localLogVarianceNoise = float(trigamma(float(Nu_local) / 2.0))
    else:
        localLogVarianceNoise = float(np.mean(noiseSelected, dtype=np.float64))
    # inverse trigamma --> inf near 0
    gap = max(varLogVarRatio - localLogVarianceNoise, 1.0e-6)
    Nu_0 = 2.0 * itrigamma(gap)
    if Nu_0 < 4.0:
        Nu_0 = 4.0

    return float(Nu_0)


def EB_computePriorStrength(
    localModelVariances: np.ndarray,
    globalModelVariances: np.ndarray,
    Nu_local: float,
    thinStride: int = 1,
    localLogVarianceNoise: np.ndarray | None = None,
    candidateMask: np.ndarray | None = None,
) -> float:
    r"""Compute :math:`\nu_0` to determine 'prior strength'

    The prior model strength is determined by 'excess' dispersion beyond sampling noise at the local level.

    :param localModelVariances: Local model variance estimates.
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
    if localLogVarianceNoise is None:
        localLogVarianceNoiseArr = None
    else:
        localLogVarianceNoiseArr = np.asarray(
            localLogVarianceNoise,
            dtype=np.float64,
        ).ravel()
        if localLogVarianceNoiseArr.shape != localModelVariancesArr.shape:
            raise ValueError(
                "localLogVarianceNoise must align with localModelVariances"
            )

    if candidateMask is not None:
        candidateMaskArr = np.asarray(candidateMask, dtype=bool).ravel()
        if candidateMaskArr.shape != localModelVariancesArr.shape:
            raise ValueError("candidateMask must align with localModelVariances")
    else:
        candidateMaskArr = None

    stride = max(int(thinStride or 1), 1)
    candidateIdx, candidateCount = cconsenrich.cEBPriorStrengthCandidateIdx(
        localModelVariancesArr,
        globalModelVariancesArr,
        localLogVarianceNoiseArr,
        candidateMaskArr,
        stride,
    )
    minPoints = max(4, int(np.ceil((0.10) * localModelVariancesArr.size)))
    if candidateCount < minPoints:
        logger.warning(
            f"Insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
        )
        return float(1.0e6)

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
        localLogVarianceNoiseArr,
    )


def EB_computePooledPriorStrength(
    localModelVariances: np.ndarray,
    globalModelVariances: np.ndarray,
    Nu_local: float,
    sampleIndex: np.ndarray | None = None,
    chromosomeIndex: np.ndarray | None = None,
    blockStarts: np.ndarray | None = None,
    thinBinSize: int = 1,
    localLogVarianceNoise: np.ndarray | None = None,
) -> float:
    r"""Compute pooled :math:`\nu_0` using deterministic sample/chromosome/bin thinning."""

    localArr = np.asarray(localModelVariances, dtype=np.float64).ravel()
    globalArr = np.asarray(globalModelVariances, dtype=np.float64).ravel()
    if localArr.shape != globalArr.shape:
        raise ValueError("localModelVariances and globalModelVariances must align")
    if localLogVarianceNoise is None:
        localLogVarianceNoiseArr = None
    else:
        localLogVarianceNoiseArr = np.asarray(
            localLogVarianceNoise,
            dtype=np.float64,
        ).ravel()
        if localLogVarianceNoiseArr.shape != localArr.shape:
            raise ValueError(
                "localLogVarianceNoise must align with localModelVariances"
            )
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
    else:
        samples = None
        chromosomes = None
        starts = None
        binSize = max(int(thinBinSize or 1), 1)

    candidateIdx, candidateCount = cconsenrich.cEBPooledPriorStrengthCandidateIdx(
        localArr,
        globalArr,
        localLogVarianceNoiseArr,
        samples,
        chromosomes,
        starts,
        binSize,
    )
    minPoints = max(4, int(np.ceil(0.10 * localArr.size)))
    if candidateCount < minPoints:
        logger.warning(
            "Insufficient pooled prior/local variance pairs...setting Nu_0 = 4.0"
        )
        return float(4.0)

    if candidateIdx.size < 4:
        logger.warning("After pooled thinning, insufficient pairs...setting Nu_0 = 4.0")
        return float(4.0)

    return _computePriorStrengthFromCandidateIdx(
        localArr,
        globalArr,
        Nu_local,
        candidateIdx,
        localLogVarianceNoiseArr,
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
