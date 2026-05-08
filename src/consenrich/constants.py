# -*- coding: utf-8 -*-
r"""Various constants and genome resources used in Consenrich."""

import logging
import os
from typing import List, Optional

from numpy import float32 as _float32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALIGNMENT_SOURCE_KINDS = ("BAM",)
FRAGMENTS_SOURCE_KIND = "FRAGMENTS"  # 10x
BEDGRAPH_SOURCE_KIND = "BEDGRAPH"
SUPPORTED_SOURCE_KINDS = ALIGNMENT_SOURCE_KINDS + (
    FRAGMENTS_SOURCE_KIND,
    BEDGRAPH_SOURCE_KIND,
)
SUPPORTED_BAM_INPUT_MODES = ("auto", "fragments", "reads", "read1")
SUPPORTED_FRAGMENT_POSITION_MODES = ("insertionendpoints", "fragmentendpoints")
SUPPORTED_COUNT_MODES = (
    "coverage",
    "cutsite",
    "fiveprime",
    "center",
)

UNCERTAINTY_CALIBRATION_POSITIVE_FLOOR = 1.0e-12
UNCERTAINTY_CALIBRATION_FEATURE_POSITIVE_FLOOR = 1.0e-12
UNCERTAINTY_CALIBRATION_FEATURE_SCALE_FLOOR = 1.0e-8
UNCERTAINTY_CALIBRATION_FEATURE_MAD_NORMAL_SCALE = 1.4826
UNCERTAINTY_CALIBRATION_FEATURE_HIGH_SIGNAL_QUANTILE = 0.90
UNCERTAINTY_CALIBRATION_FEATURE_NAMES = (
    "intercept",
    "log_state_variance",
    "log_mean_observation_variance",
    "abs_state",
    "abs_state_delta",
    "high_signal",
)
UNCERTAINTY_CALIBRATION_AUTO_BLOCK_MIN_BP = 50_000
UNCERTAINTY_CALIBRATION_AUTO_BLOCK_INTERVAL_MULTIPLIER = 100
UNCERTAINTY_CALIBRATION_MIN_BLOCK_INTERVALS = 8
UNCERTAINTY_CALIBRATION_MIN_FOLDS = 2
UNCERTAINTY_CALIBRATION_MIN_HOLDOUT_REPLICATES = 1
UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MIN = 0.10
UNCERTAINTY_CALIBRATION_DEFAULT_HOLDOUT_FRACTION_MAX = 0.30
UNCERTAINTY_CALIBRATION_MIN_CALIBRATION_EM_ITERS = 1
UNCERTAINTY_CALIBRATION_DEFAULT_TARGETS = (0.50, 0.80, 0.90, 0.95)
UNCERTAINTY_CALIBRATION_TARGET_ALPHA_FLOOR = 1.0e-6
UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MIN = 0.05
UNCERTAINTY_CALIBRATION_DEFAULT_FACTOR_MAX = 25.0
UNCERTAINTY_CALIBRATION_FACTOR_MIN_FLOOR = 1.0e-6
UNCERTAINTY_CALIBRATION_FACTOR_MAX_MIN_RATIO = 1.01
UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MIN = 0.25
UNCERTAINTY_CALIBRATION_A_OBS_FACTOR_MAX = 4.0
UNCERTAINTY_CALIBRATION_DEFAULT_RIDGE = 1.0e-2
UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT = 0.05
UNCERTAINTY_CALIBRATION_WIS_WEIGHT = UNCERTAINTY_CALIBRATION_DEFAULT_WIS_WEIGHT
UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY = 1.0e-2
UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PRIOR_STRENGTH = (
    UNCERTAINTY_CALIBRATION_DEFAULT_A_OBS_PENALTY
)
UNCERTAINTY_CALIBRATION_WIS_SCALE_MULTIPLIER = 2.0
UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES = 10
UNCERTAINTY_CALIBRATION_SCORE_SD_DECILES = UNCERTAINTY_CALIBRATION_SCORE_PSTATE_DECILES
UNCERTAINTY_CALIBRATION_SCORE_STATE_ABS_QUANTILE = 0.90
UNCERTAINTY_CALIBRATION_SCORE_FOLD_CODE_STRIDE = 4096
UNCERTAINTY_CALIBRATION_SCORE_REPLICATE_CODE_STRIDE = 32
UNCERTAINTY_CALIBRATION_SUMMARY_MEDIAN_QUANTILE = 0.50
UNCERTAINTY_CALIBRATION_SUMMARY_Q90_QUANTILE = 0.90
UNCERTAINTY_CALIBRATION_DEFAULT_MAX_SCORES = 200_000
UNCERTAINTY_CALIBRATION_DEFAULT_MAX_DIAGNOSTIC_ROWS = 50_000
UNCERTAINTY_CALIBRATION_DEFAULT_MIN_HELDOUT_CELLS = 1_000
UNCERTAINTY_CALIBRATION_DEFAULT_SEED = 1729
UNCERTAINTY_CALIBRATION_DIAGNOSTIC_SEED_OFFSET = 10_000
UNCERTAINTY_CALIBRATION_MASKED_OBSERVATION_VARIANCE = _float32(1.0e30)

PROCESS_Q_CALIBRATION_NONE = "none"
PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL = "regularizedDiagonal"
PROCESS_Q_CALIBRATION_MODES = (
    PROCESS_Q_CALIBRATION_NONE,
    PROCESS_Q_CALIBRATION_REGULARIZED_DIAGONAL,
)
PROCESS_Q_DEFAULT_TREND_TARGET_RATIO = 0.01
PROCESS_Q_TREND_FLOOR_RATIO = 1.0e-6
PROCESS_Q_NUMERICAL_FLOOR = 1.0e-8

EFFECTIVE_GENOME_SIZES = {
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


def getGenomeResourceFile(genome: str, fileType: str, dir_: str = "data"):
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
