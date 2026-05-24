# -*- coding: utf-8 -*-
r"""Various constants and genome resources used in Consenrich."""

import logging
import os
from typing import Final, TypeAlias

from numpy import float32 as _float32

ConfigurationDefaultValue: TypeAlias = bool | int | float | str | None
ConfigurationProfileMap: TypeAlias = dict[str, dict[str, ConfigurationDefaultValue]]
GenomeEffectiveSizeMap: TypeAlias = dict[str, dict[int, int]]
StrTuple: TypeAlias = tuple[str, ...]
FloatTuple: TypeAlias = tuple[float, ...]

logger: Final[logging.Logger] = logging.getLogger(__name__)

ALIGNMENT_SOURCE_KINDS: Final[StrTuple] = ("BAM",)
FRAGMENTS_SOURCE_KIND: Final[str] = "FRAGMENTS"  # 10x
BEDGRAPH_SOURCE_KIND: Final[str] = "BEDGRAPH"
SUPPORTED_SOURCE_KINDS: Final[StrTuple] = ALIGNMENT_SOURCE_KINDS + (
    FRAGMENTS_SOURCE_KIND,
    BEDGRAPH_SOURCE_KIND,
)
SUPPORTED_BAM_INPUT_MODES: Final[StrTuple] = ("auto", "fragments", "reads", "read1")
SUPPORTED_FRAGMENT_POSITION_MODES: Final[StrTuple] = (
    "insertionendpoints",
    "fragmentendpoints",
)
SUPPORTED_COUNT_MODES: Final[StrTuple] = (
    "coverage",
    "cutsite",
    "fiveprime",
    "center",
)

GENERIC_DEFAULT_CONFIGURATION: Final[str] = "generic"
SUPPORTED_DEFAULT_CONFIGURATIONS: Final[StrTuple] = (GENERIC_DEFAULT_CONFIGURATION,)
DEFAULT_CONFIGURATION_KEYS: Final[StrTuple] = ("configuration",)

UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR: Final[float] = 1.0e-12
UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR: Final[float] = 1.0e-12
UNCERTAINTY_CALIBRATION_FEATURE_SCALE_FLOOR: Final[float] = 1.0e-8
UNCERTAINTY_CALIBRATION_FEATURE_MAD_NORMAL_SCALE: Final[float] = 1.4826
UNCERTAINTY_CALIBRATION_FEATURE_HIGH_SIGNAL_QUANTILE: Final[float] = 0.90
UNCERTAINTY_CALIBRATION_FEATURE_NAMES: Final[StrTuple] = (
    "intercept",
    "log_state_variance",
    "log_mean_observation_variance",
    "abs_state",
    "abs_state_delta",
    "high_signal",
)
UNCERTAINTY_CALIBRATION_AUTO_BLOCK_MIN_BP: Final[int] = 50_000
UNCERTAINTY_CALIBRATION_AUTO_BLOCK_INTERVAL_MULTIPLIER: Final[int] = 100
UNCERTAINTY_CALIBRATION_MIN_BLOCK_INTERVALS: Final[int] = 8
UNCERTAINTY_CALIBRATION_MIN_FOLDS: Final[int] = 2
UNCERTAINTY_CALIBRATION_MIN_HOLDOUT_REPLICATES: Final[int] = 1
UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MIN: Final[float] = 0.10
UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MAX: Final[float] = 0.30
UNCERTAINTY_CALIBRATION_MIN_CALIBRATION_ECM_ITERS: Final[int] = 2
UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS: Final[FloatTuple] = (
    0.50,
    0.80,
    0.90,
    0.95,
)
UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR: Final[float] = 1.0e-6
UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN: Final[float] = 0.05
UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX: Final[float] = 25.0
UNCERTAINTY_CALIBRATION_FACTOR_MIN_FLOOR: Final[float] = 1.0e-6
UNCERTAINTY_CALIBRATION_FACTOR_MAX_MIN_RATIO: Final[float] = 1.01
UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN: Final[float] = 0.25
UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX: Final[float] = 4.0
UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE: Final[float] = 1.0e-2
UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT: Final[float] = 0.05
UNCERTAINTY_CALIBRATION_WIS_WEIGHT: Final[float] = (
    UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT
)
UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY: Final[float] = 1.0e-2
UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH: Final[float] = (
    UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY
)
UNCERTAINTY_CALIBRATION_WIS_SCALE_MULTIPLIER: Final[float] = 2.0
UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES: Final[int] = 10
UNCERTAINTY_CALIBRATION_SCORE_SD_DECILES: Final[int] = (
    UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES
)
UNCERTAINTY_CALIBRATION_SCORE_STATE_ABS_QUANTILE: Final[float] = 0.90
UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE: Final[int] = 4096
UNCERTAINTY_CALIBRATION_SCORE_REPLICATE_CODE_STRIDE: Final[int] = 32
UNCERTAINTY_CALIBRATION_SUMMARY_MEDIAN_QUANTILE: Final[float] = 0.50
UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE: Final[float] = 0.90
UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES: Final[int] = 200_000
UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS: Final[int] = 50_000
UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS: Final[int] = 1_000
UNCERTAINTY_CALIBRATION_DEFAULT_SEED: Final[int] = 1729
UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET: Final[int] = 10_000
UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE: Final[_float32] = _float32(1.0e30)

PROCESS_NOISE_DEFAULT_REGULARIZATION_STRENGTH: Final[float] = 1.0  ###
PROCESS_NOISE_DEFAULT_MAP_ROUGHNESS_PENALTY: Final[float] = (  ###
    PROCESS_NOISE_DEFAULT_REGULARIZATION_STRENGTH
)
PROCESS_NOISE_DEFAULT_REGULARIZATION_RATIO: Final[float] = 1.0e-8  ###
PROCESS_NOISE_DEFAULT_WARMUP_ECM_ITERS: Final[int] = 50
PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES: Final[int] = 8
PROCESS_NOISE_NUMERICAL_FLOOR: Final[float] = 1.0e-10
UNCERTAINTY_CALIBRATION_REFIT_PROCESS_NOISE_WARMUP_ECM_ITERS: Final[int] = (
    PROCESS_NOISE_DEFAULT_WARMUP_ECM_ITERS
)

STATE_MODEL_LEVEL: Final[str] = "level"
STATE_MODEL_LEVEL_TREND: Final[str] = "levelTrend"
STATE_MODEL_MODES: Final[StrTuple] = (
    STATE_MODEL_LEVEL,
    STATE_MODEL_LEVEL_TREND,
)

EXPERIMENT_DEFAULT_NAME: Final[str] = "consenrichExperiment"

INPUT_DEFAULT_SAMPLES: Final[None] = None
INPUT_DEFAULT_BAM_FILES: Final[StrTuple] = ()
INPUT_DEFAULT_BAM_FILES_CONTROL: Final[StrTuple] = ()
INPUT_DEFAULT_ROLE: Final[str] = "treatment"

OUTPUT_DEFAULT_ROUND_DIGITS: Final[int] = 3
OUTPUT_DEFAULT_WRITE_UNCERTAINTY: Final[bool] = True
OUTPUT_DEFAULT_SAVE_BACKGROUND_TRACKS: Final[bool] = True
OUTPUT_DEFAULT_PLOT_OPTIMIZATION_PATH: Final[bool] = True

GENOME_DEFAULT_NAME: Final[str | None] = None
GENOME_DEFAULT_CHROM_SIZES_FILE: Final[str | None] = None
GENOME_DEFAULT_BLACKLIST_FILE: Final[str | None] = None
GENOME_DEFAULT_SPARSE_BED_FILE: Final[str | None] = None
GENOME_DEFAULT_CHROMOSOMES: Final[StrTuple | None] = None
GENOME_DEFAULT_EXCLUDE_CHROMS: Final[StrTuple] = ()
GENOME_DEFAULT_EXCLUDE_FOR_NORM: Final[StrTuple] = ()
NONSTANDARD_CHROMOSOME_NAMES: Final[StrTuple] = (
    "chrX",
    "chrY",
    "chrM",
    "chrMT",
    "X",
    "Y",
    "M",
    "MT",
    "_",
)

STATE_DEFAULT_INIT: Final[float] = 0.0
STATE_DEFAULT_COVAR_INIT: Final[float] = 1000.0
STATE_DEFAULT_BOUND_STATE: Final[bool] = False
STATE_DEFAULT_LOWER_BOUND: Final[float] = 0.0
STATE_DEFAULT_UPPER_BOUND: Final[float] = 10000.0

COUNTING_DEFAULT_INTERVAL_SIZE_BP: Final[int] = 50
COUNTING_DEFAULT_BACKGROUND_BLOCK_SIZE_BP: Final[int] = -1
COUNTING_DEFAULT_SCALE_FACTORS: Final[tuple[float, ...] | None] = None
COUNTING_DEFAULT_SCALE_FACTORS_CONTROL: Final[tuple[float, ...] | None] = None
COUNTING_DEFAULT_NORM_METHOD: Final[str] = "EGS"
COUNTING_DEFAULT_FRAGMENTS_GROUP_NORM: Final[str] = "NONE"
COUNTING_SUPPORTED_NORM_METHODS: Final[StrTuple] = ("EGS", "RPGC", "RPKM", "CPM", "SF")
COUNTING_SUPPORTED_FRAGMENTS_GROUP_NORMS: Final[StrTuple] = ("NONE", "CELLS")
COUNTING_DEFAULT_FIX_CONTROL: Final[bool] = False
COUNTING_DEFAULT_LOG_OFFSET: Final[float] = 1.0
COUNTING_DEFAULT_LOG_MULT: Final[float] = 1.0
COUNTING_DEFAULT_SUBTRACT_GLOBAL_MEDIAN: Final[bool] = False  ###
COUNTING_DEFAULT_REPLICATE_MEDIAN_DETREND: Final[bool] = False  ###
COUNTING_DEFAULT_REPLICATE_MEDIAN_DETREND_WINDOW_MULTIPLIER: Final[float] = 5.0
COUNTING_DEFAULT_GENTLE_DETREND_QUANTILE: Final[float] = 0.5

SC_DEFAULT_BARCODE_TAG: Final[str] = "CB"
SC_DEFAULT_COUNT_MODE: Final[str] = "coverage"
SC_DEFAULT_FRAGMENTS_GROUP_NORM: Final[str] = COUNTING_DEFAULT_FRAGMENTS_GROUP_NORM
SC_DEFAULT_FRAGMENT_POSITION_MODE: Final[str] = "insertionEndpoints"
SC_SUPPORTED_COUNT_MODES: Final[StrTuple] = SUPPORTED_COUNT_MODES + ("midpoint",)

PROCESS_DEFAULT_DELTA_F: Final[float] = 1.0
PROCESS_DEFAULT_MIN_Q: Final[float] = 1.0e-8  ###
PROCESS_DEFAULT_MAX_Q: Final[float] = 1000.0
PROCESS_DEFAULT_PRECISION_MULTIPLIER_MIN: Final[float] = 0.75  ### *
PROCESS_DEFAULT_PRECISION_MULTIPLIER_MAX: Final[float] = 1.25  ### *
PROCESS_DEFAULT_STATE_MODEL: Final[str] = STATE_MODEL_LEVEL_TREND  ###

OBSERVATION_DEFAULT_MIN_R: Final[float] = -1.0
OBSERVATION_DEFAULT_MAX_R: Final[float] = 1000.0
OBSERVATION_DEFAULT_SAMPLING_ITERS: Final[int] = 10_000
OBSERVATION_DEFAULT_SAMPLING_BLOCK_SIZE_BP: Final[int] = -1
MUNC_VARIANCE_MODEL_AR1: Final[str] = "ar1"
MUNC_VARIANCE_MODEL_SVAR: Final[str] = "svar"
MUNC_VARIANCE_MODEL_SVAR_D1: Final[str] = "svarD1"
MUNC_VARIANCE_MODEL_SVAR_D2: Final[str] = "svarD2"
MUNC_VARIANCE_MODEL_CODE_AR1: Final[int] = 0
MUNC_VARIANCE_MODEL_CODE_SVAR: Final[int] = 2
MUNC_VARIANCE_MODEL_CODE_SVAR_D1: Final[int] = 4
MUNC_VARIANCE_MODEL_CODE_SVAR_D2: Final[int] = 5
MUNC_SUPPORTED_VARIANCE_MODELS: Final[StrTuple] = (
    MUNC_VARIANCE_MODEL_AR1,
    MUNC_VARIANCE_MODEL_SVAR,
    MUNC_VARIANCE_MODEL_SVAR_D1,
    MUNC_VARIANCE_MODEL_SVAR_D2,
)
OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL: Final[str] = (
    MUNC_VARIANCE_MODEL_AR1  ### * alt: svar
)
OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_SIZE_BP: Final[int | None] = None
OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_SIZE_BP: Final[int | None] = None
OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER: Final[float] = 2.0  ###
OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER: Final[float] = 2.0  ###
OBSERVATION_DEFAULT_EB_USE: Final[bool] = True
OBSERVATION_DEFAULT_EB_SET_NU0: Final[float | None] = None
OBSERVATION_DEFAULT_EB_SET_NUL: Final[float | None] = None
OBSERVATION_DEFAULT_NO_DM_VAR: Final[bool] = False
OBSERVATION_DEFAULT_TREND_NUM_BASIS: Final[int] = 60
OBSERVATION_DEFAULT_TREND_MIN_OBS_PER_BASIS: Final[float] = 25.0
OBSERVATION_DEFAULT_TREND_MIN_EDF: Final[float] = 3.0
OBSERVATION_DEFAULT_TREND_MAX_EDF: Final[float] = 30.0
OBSERVATION_DEFAULT_TREND_LAMBDA_MIN: Final[float] = 1.0e-6
OBSERVATION_DEFAULT_TREND_LAMBDA_MAX: Final[float] = 1.0e6
OBSERVATION_DEFAULT_TREND_LAMBDA_GRID_SIZE: Final[int] = 41
OBSERVATION_DEFAULT_NUM_NEAREST: Final[int] = 0
OBSERVATION_DEFAULT_RESTRICT_LOCAL_AR1_TO_SPARSE_BED: Final[bool] = False
OBSERVATION_DEFAULT_RESTRICT_LOCAL_VARIANCE_TO_SPARSE_BED: Final[bool] = (
    OBSERVATION_DEFAULT_RESTRICT_LOCAL_AR1_TO_SPARSE_BED
)
OBSERVATION_DEFAULT_SPARSE_SUPPORT_SCALE_BP: Final[float] = -1.0
OBSERVATION_DEFAULT_SPARSE_SUPPORT_PRIOR: Final[float] = 1.0
OBSERVATION_DEFAULT_PAD: Final[float] = 1.0e-4  ###
OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MIN: Final[float] = 1.0  ###
OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MAX: Final[float] = 1.0  ###
OBSERVATION_DEFAULT_USE_REPLICATE_TRENDS: Final[bool] = True  ###

FIT_DEFAULT_FIXED_BACKGROUND_ITERS: Final[int] = 50
FIT_DEFAULT_FIXED_BACKGROUND_RTOL: Final[float] = 1.0e-6  ###
FIT_DEFAULT_ROBUST_T_NU: Final[float] = 10.0
FIT_DEFAULT_USE_OBS_PRECISION_REWEIGHTING: Final[bool] = False  ###
FIT_DEFAULT_USE_PROCESS_PRECISION_REWEIGHTING: Final[bool] = True  ###
FIT_DEFAULT_USE_APN: Final[bool] = False
FIT_DEFAULT_BACKGROUND: Final[bool] = True  ###
FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND: Final[bool] = True  ###
FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER: Final[float] = 1.0  ###
FIT_DEFAULT_ZERO_CENTER_BACKGROUND: Final[bool] = False
FIT_DEFAULT_ZERO_CENTER_REPLICATE_BIAS: Final[bool] = True
FIT_DEFAULT_OUTER_ITERS: Final[int] = 32
FIT_DEFAULT_MIN_OUTER_ITERS: Final[int] = 3
FIT_DEFAULT_BACKGROUND_SHIFT_RTOL: Final[float] = 0.005  ###
FIT_DEFAULT_OUTER_NLL_RTOL: Final[float] = 1.0e-4  ###
FIT_DEFAULT_BACKGROUND_SMOOTHNESS: Final[float] = 10.0
FIT_DEFAULT_BACKGROUND_LENGTH_SCALE_MULTIPLIER: Final[float] = 16.0  ###

SAM_DEFAULT_THREADS: Final[int] = 2
SAM_DEFAULT_FLAG_EXCLUDE: Final[int] = 3844
SAM_DEFAULT_MIN_MAPPING_QUALITY: Final[int] = 10
SAM_DEFAULT_ONE_READ_PER_BIN: Final[int] = 0
SAM_DEFAULT_CHUNK_SIZE: Final[int] = 500_000
SAM_DEFAULT_BAM_INPUT_MODE: Final[str] = "auto"
SAM_DEFAULT_COUNT_MODE: Final[str] = "coverage"
SAM_DEFAULT_SHIFT_FORWARD_5P: Final[int] = 0
SAM_DEFAULT_SHIFT_REVERSE_5P: Final[int] = 0
SAM_DEFAULT_EXTEND_FROM_5P_BP: Final[int | None] = None
SAM_DEFAULT_MAX_INSERT_SIZE: Final[int] = 1000
SAM_DEFAULT_INFER_FRAGMENT_LENGTH: Final[int | None] = None
SAM_DEFAULT_MIN_TEMPLATE_LENGTH: Final[int] = -1

MATCHING_DEFAULT_ENABLED: Final[bool] = True
MATCHING_DEFAULT_RAND_SEED: Final[int] = 42
MATCHING_DEFAULT_NUM_BOOTSTRAP: Final[int] = 128
MATCHING_DEFAULT_THRESHOLD_Z: Final[float] = 2.0  ###
MATCHING_DEFAULT_DEPENDENCE_SPAN: Final[int | None] = None
MATCHING_DEFAULT_GAMMA: Final[float] = 0.25  ###
MATCHING_DEFAULT_SELECTION_PENALTY: Final[float | None] = None
MATCHING_DEFAULT_GAMMA_SCALE: Final[float] = 0.5
MATCHING_DEFAULT_NESTED_ROCCO_ITERS: Final[int] = 3
MATCHING_DEFAULT_NESTED_ROCCO_BUDGET_SCALE: Final[float] = 0.9  ###
MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER: Final[float] = 2.0

UNCERTAINTY_CALIBRATION_DEFAULT_ENABLED: Final[bool] = True
UNCERTAINTY_CALIBRATION_DEFAULT_FOLDS: Final[int] = 3
UNCERTAINTY_CALIBRATION_DEFAULT_BLOCK_SIZE_BP: Final[int | None] = None
UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION: Final[float | None] = None
UNCERTAINTY_CALIBRATION_DEFAULT_HELDOUT_REPLICATE_FRACTION: Final[float | None] = None
UNCERTAINTY_CALIBRATION_DEFAULT_MAX_HELDOUT_CELLS: Final[int | None] = None
UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN_OVERRIDE: Final[float | None] = None
UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX_OVERRIDE: Final[float | None] = None
UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH_OVERRIDE: Final[float | None] = (
    None
)
UNCERTAINTY_CALIBRATION_DEFAULT_CALIBRATION_ECM_ITERS: Final[int] = 10  ### *
UNCERTAINTY_CALIBRATION_DEFAULT_TARGET_CALIBRATION_DELTA: Final[float] = 0.05
UNCERTAINTY_CALIBRATION_DEFAULT_SCALE_UNCERTAINTY_BY_TARGET_CALIBRATION: Final[bool] = (
    True
)
UNCERTAINTY_CALIBRATION_DEFAULT_PAD: Final[float] = 1.0e-4
UNCERTAINTY_CALIBRATION_DEFAULT_WRITE_DIAGNOSTICS: Final[bool] = False

ROCCO_BUDGET_MIN: Final[float] = 0.001  ###
ROCCO_BUDGET_MAX: Final[float] = 0.25  ###
ROCCO_NULL_QUANTILE: Final[float] = 0.8
ROCCO_THRESHOLD_Z_DEFAULT: Final[float] = MATCHING_DEFAULT_THRESHOLD_Z
ROCCO_NUM_BOOTSTRAP_DEFAULT: Final[int] = MATCHING_DEFAULT_NUM_BOOTSTRAP
ROCCO_BUDGET_Z_GRID: Final[FloatTuple] = (1.5, 2.0, 2.5, 3.0)
ROCCO_MAX_ITER_DEFAULT: Final[int] = 60
NESTED_ROCCO_ITERS_DEFAULT: Final[int] = MATCHING_DEFAULT_NESTED_ROCCO_ITERS
NESTED_ROCCO_JACCARD_DEFAULT: Final[float] = 0.999
NESTED_ROCCO_MIN_PARENT_STEPS: Final[int] = 5
NESTED_ROCCO_MIN_CHILD_STEPS: Final[int] = NESTED_ROCCO_MIN_PARENT_STEPS
NESTED_ROCCO_BUDGET_SCALE_DEFAULT: Final[float] = (
    MATCHING_DEFAULT_NESTED_ROCCO_BUDGET_SCALE
)
NESTED_ROCCO_SUBTASK_MAX_ITER: Final[int] = 5
EXPORT_MEDIAN_SIGNAL_LOCAL_UNCERTAINTY_MULTIPLIER: Final[float] = (
    MATCHING_DEFAULT_EXPORT_FILTER_UNCERTAINTY_MULTIPLIER
)
MASSIVE_SUBPEAK_CLEANUP_DEFAULT: Final[bool] = True
MASSIVE_SUBPEAK_MIN_BP: Final[int] = 10000
MASSIVE_SUBPEAK_WIDTH_ALPHA: Final[float] = 0.1
MASSIVE_SUBPEAK_WIDTH_BULK_QUANTILE: Final[float] = 0.95
MASSIVE_SUBPEAK_MAX_FRACTION: Final[float] = 0.005
MASSIVE_SUBPEAK_MIN_LOG_GAP: Final[float] = 0.10
MASSIVE_SUBPEAK_MIN_PEAKS: Final[int] = 25
MASSIVE_SUBPEAK_SPLIT_QUANTILE: Final[float] = 0.25
MASSIVE_SUBPEAK_SPLIT_Z: Final[float] = 2.0
MASSIVE_SUBPEAK_MAX_DEPTH: Final[int] = 8
MASSIVE_SUBPEAK_MIN_CHILD_BP: Final[int] = 500
MASSIVE_SUBPEAK_MIN_CHILD_FRACTION: Final[float] = 0.1

DEFAULT_CONFIGURATION_VALUES: Final[ConfigurationProfileMap] = {
    GENERIC_DEFAULT_CONFIGURATION: {
        "fitParams.ECM_fixedBackgroundIters": FIT_DEFAULT_FIXED_BACKGROUND_ITERS,
        "fitParams.ECM_fixedBackgroundRtol": FIT_DEFAULT_FIXED_BACKGROUND_RTOL,
        "fitParams.ECM_outerIters": FIT_DEFAULT_OUTER_ITERS,
        "fitParams.ECM_minOuterIters": FIT_DEFAULT_MIN_OUTER_ITERS,
        "fitParams.ECM_backgroundShiftRtol": FIT_DEFAULT_BACKGROUND_SHIFT_RTOL,
        "fitParams.ECM_outerNLLRtol": FIT_DEFAULT_OUTER_NLL_RTOL,
        "fitParams.ECM_backgroundSmoothness": FIT_DEFAULT_BACKGROUND_SMOOTHNESS,
        "fitParams.ECM_backgroundLengthScaleMultiplier": (
            FIT_DEFAULT_BACKGROUND_LENGTH_SCALE_MULTIPLIER
        ),
        "fitParams.useNonnegativeBackground": FIT_DEFAULT_USE_NONNEGATIVE_BACKGROUND,
        "fitParams.backgroundNegativePenaltyMultiplier": (
            FIT_DEFAULT_BACKGROUND_NEGATIVE_PENALTY_MULTIPLIER
        ),
        "processParams.regularizationStrength": (
            PROCESS_NOISE_DEFAULT_REGULARIZATION_STRENGTH
        ),
        "processParams.processNoiseMapRoughnessPenalty": (
            PROCESS_NOISE_DEFAULT_MAP_ROUGHNESS_PENALTY
        ),
        "processParams.processNoiseMAPRoughnessPenalty": (
            PROCESS_NOISE_DEFAULT_MAP_ROUGHNESS_PENALTY
        ),
        "processParams.regularizationRatio": PROCESS_NOISE_DEFAULT_REGULARIZATION_RATIO,
        "processParams.processNoiseWarmupECMIters": (
            PROCESS_NOISE_DEFAULT_WARMUP_ECM_ITERS
        ),
        "processParams.processNoiseWarmupOuterPasses": (
            PROCESS_NOISE_DEFAULT_WARMUP_OUTER_PASSES
        ),
        "processParams.stateModel": PROCESS_DEFAULT_STATE_MODEL,
        "processParams.precisionMultiplierMin": (
            PROCESS_DEFAULT_PRECISION_MULTIPLIER_MIN
        ),
        "processParams.precisionMultiplierMax": (
            PROCESS_DEFAULT_PRECISION_MULTIPLIER_MAX
        ),
        "observationParams.precisionMultiplierMin": (
            OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MIN
        ),
        "observationParams.precisionMultiplierMax": (
            OBSERVATION_DEFAULT_PRECISION_MULTIPLIER_MAX
        ),
        "observationParams.muncVarianceModel": (
            OBSERVATION_DEFAULT_MUNC_VARIANCE_MODEL
        ),
        "observationParams.muncTrendBlockSizeBP": (
            OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_SIZE_BP
        ),
        "observationParams.muncLocalWindowSizeBP": (
            OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_SIZE_BP
        ),
        "observationParams.muncTrendBlockDependenceMultiplier": (
            OBSERVATION_DEFAULT_MUNC_TREND_BLOCK_DEPENDENCE_MULTIPLIER
        ),
        "observationParams.muncLocalWindowDependenceMultiplier": (
            OBSERVATION_DEFAULT_MUNC_LOCAL_WINDOW_DEPENDENCE_MULTIPLIER
        ),
        "observationParams.noDMVar": OBSERVATION_DEFAULT_NO_DM_VAR,
        "observationParams.restrictLocalVarianceToSparseBed": (
            OBSERVATION_DEFAULT_RESTRICT_LOCAL_VARIANCE_TO_SPARSE_BED
        ),
        "countingParams.gentleDetrendQuantile": (
            COUNTING_DEFAULT_GENTLE_DETREND_QUANTILE
        ),
        "countingParams.subtractGlobalMedian": (
            COUNTING_DEFAULT_SUBTRACT_GLOBAL_MEDIAN
        ),
        "outputParams.saveBackgroundTracks": OUTPUT_DEFAULT_SAVE_BACKGROUND_TRACKS,
        "uncertaintyCalibrationParams.enabled": (
            UNCERTAINTY_CALIBRATION_DEFAULT_ENABLED
        ),
    }
}

EFFECTIVE_GENOME_SIZES: Final[GenomeEffectiveSizeMap] = {
    "hg19": {
        50: 2685511454,
        75: 2736124898,
        100: 2776919708,
        150: 2827436883,
        200: 2855463800,
        250: 2855044784,
    },
    "hg38": {
        50: 2701495711,
        75: 2747877702,
        100: 2805636231,
        150: 2862010428,
        200: 2887553103,
        250: 2898802627,
    },
    "t2t": {
        50: 2725240337,
        75: 2786136059,
        100: 2814334875,
        150: 2931551487,
        200: 2936403235,
        250: 2960856300,
    },
    "mm10": {
        50: 2308125299,
        75: 2407883243,
        100: 2467481008,
        150: 2494787038,
        200: 2520868989,
        250: 2538590322,
    },
    "mm39": {
        50: 2309746861,
        75: 2410055689,
        100: 2468088461,
        150: 2495461690,
        200: 2521902382,
        250: 2538633971,
    },
    "dm3": {
        50: 130428510,
        75: 135004387,
        100: 139647132,
        150: 144307658,
        200: 148523810,
        250: 151901455,
    },
    "dm6": {
        50: 125464678,
        75: 127324557,
        100: 129789773,
        150: 129940985,
        200: 132508963,
        250: 132900923,
    },
    "ce11": {
        50: 95159402,
        75: 96945370,
        100: 98259898,
        150: 98721103,
        200: 98672558,
        250: 101271756,
    },
}


def resolveGenomeName(genome: str) -> str:
    r"""Standardize the genome name for consistency
    :param genome: Name of the genome. See :class:`consenrich.core.genomeParams`.
    :type genome: str
    :return: Standardized genome name.
    :rtype: str
    :raises ValueError: If the genome is not recognized.
    """
    genome_ = genome.lower()
    if genome_ in ["hg19", "grch37"]:
        return "hg19"
    elif genome_ in ["hg38", "grch38"]:
        return "hg38"
    elif genome_ in ["t2t", "chm13", "t2t-chm13"]:
        return "t2t"
    elif genome_ in ["mm10", "grcm38"]:
        return "mm10"
    elif genome_ in ["mm39", "grcm39"]:
        return "mm39"
    elif genome_ in ["dm3"]:
        return "dm3"
    elif genome_ in ["dm6"]:
        return "dm6"
    elif genome_ in ["ce10", "ws220"]:
        return "ce10"
    elif genome_ in ["ce11", "wbcel235"]:
        return "ce11"
    raise ValueError(
        f"Genome {genome} is not recognized. Please provide a valid genome name or manually specify resources"
    )


def getEffectiveGenomeSize(genome: str, readLength: int) -> int:
    r"""Get the effective genome size for a given genome and read length.

    :param genome: Name of the genome. See :func:`consenrich.constants.resolveGenomeName` and :class:`consenrich.core.genomeParams`.
    :type genome: str
    :param readLength: Length of the reads. See :func:`consenrich.core.getReadLength`.
    :type readLength: int
    :raises ValueError: If the genome is not recognized or if the read length is not available for the genome.
    :return: Effective genome size in base pairs.
    :rtype: int
    """

    global EFFECTIVE_GENOME_SIZES
    genome_: str = resolveGenomeName(genome)
    if genome_ in EFFECTIVE_GENOME_SIZES:
        if readLength not in EFFECTIVE_GENOME_SIZES[genome_]:
            nearestReadLength: int = int(
                min(
                    EFFECTIVE_GENOME_SIZES[genome_].keys(),
                    key=lambda x: abs(x - readLength),
                )
            )
            return EFFECTIVE_GENOME_SIZES[genome_][nearestReadLength]
        return EFFECTIVE_GENOME_SIZES[genome_][readLength]
    raise ValueError(f"Defaults not available for {genome}")


def getGenomeResourceFile(genome: str, fileType: str, dir_: str = "data") -> str:
    r"""Get the path to a genome resource file.

    :param genome: the genome assembly. See :func:`consenrich.constants.resolveGenomeName` and :class:`consenrich.core.genomeParams`.
    :type genome: str
    :param fileType: One of 'sizes', 'blacklist', 'sparse'.
    :type fileType: str
    :return: Path to the resource file.
    :rtype: str
    :raises ValueError: If not a sizes, blacklist, or sparse file.
    :raises FileNotFoundError: If the resource file does not exist.
    """
    if fileType.lower() in ["sizes"]:
        fileName = f"{genome}.sizes"
    elif fileType.lower() in ["blacklist"]:
        fileName = f"{genome}_blacklist.bed"
    elif fileType.lower() in ["sparse"]:
        fileName = f"{genome}_sparse.bed"
    filePath = os.path.join(os.path.dirname(__file__), os.path.join(dir_, fileName))
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Resource file {filePath} does not exist.")
    return filePath
