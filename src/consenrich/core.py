# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
import os
import warnings
from functools import lru_cache
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from scipy import ndimage, signal, stats, optimize, special, sparse
from scipy.sparse import linalg as sparse_linalg
from tqdm import tqdm
from itrigamma import itrigamma
from . import cconsenrich
from . import ccounts

logging.basicConfig(
    level=logging.INFO,
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


class processParams(NamedTuple):
    r"""Parameters related to the process model of Consenrich.

    :param deltaF: Integration step size in the two-state transition
        :math:`x_{[i+1,0]} = x_{[i,0]} + \delta_F x_{[i,1]}`. If ``deltaF < 0``, the CLI centers a narrow
        search around ``0.5 * intervalSizeBP / medianFragmentLength``.
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
    :seealso: :func:`consenrich.core.runConsenrich`

    """

    deltaF: float = -1.0
    minQ: float = 2.5e-4
    maxQ: float = 1000.0
    offDiagQ: float = 0.0


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich.

    The observation model supplies the plugin variance track used by the inner filter-smoother.


    :param minR: Genome-wide lower bound for replicate-specific observation noise levels.
    :type minR: float | None
    :param maxR: Genome-wide upper bound for the replicate-specific observation noise levels.
    :type maxR: float | None
    :param samplingIters: Number of blocks (within-contig) to sample while building the empirical absMean-variance trend in :func:`consenrich.core.fitVarianceFunction`.
    :type samplingIters: int | None
    :param samplingBlockSizeBP: Expected size (in bp) of contiguous blocks that are sampled when fitting AR1 parameters to estimate :math:`(\lvert \mu_b \rvert, \sigma^2_b)` pairs.
      Note, during sampling, each block's size (unit: genomic intervals) is drawn from truncated :math:`\textsf{Geometric}(p=1/\textsf{samplingBlockSize})` to reduce artifacts from fixed-size blocks.
      If `None` or ` < 1`, then this value is inferred using :func:`consenrich.core.getContextSize`.
    :type samplingBlockSizeBP: int | None
    :param binQuantileCutoff: When fitting the variance function, pairs :math:`(\lvert \mu_b \rvert, \sigma^2_b)` are binned by their (absolute) means. This parameter sets the quantile of variances within each bin to use when fitting the global mean-variance trend.
      Increasing this value toward `1.0` can raise the prior trend for observation noise levels and therefore yield stiffer signal estimates overall.
    :type binQuantileCutoff: float | None
    :param EB_minLin: Require that the fitted trend in :func:`consenrich.core.getMuncTrack` satisfy: :math:`\textsf{variance} \geq \textsf{minLin} \cdot |\textsf{mean}|`. See :func:`fitVarianceFunction`.
    :type EB_minLin: float | None
    :param EB_use: If True, shrink 'local' noise estimates to a prior trend dependent on amplitude. See  :func:`consenrich.core.getMuncTrack`.
    :type EB_use: bool | None
    :param EB_setNu0: If provided, manually set :math:`\nu_0` to this value (rather than computing via :func:`consenrich.core.EB_computePriorStrength`).
    :type EB_setNu0: int | None
    :param EB_setNuL: If provided, manually set local model df, :math:`\nu_L`, to this value.
    :type EB_setNuL: int | None
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
    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitVarianceFunction`, :func:`consenrich.core.EB_computePriorStrength`, :func:`consenrich.cconsenrich.cinnerEM`

    """

    minR: float | None
    maxR: float | None
    samplingIters: int | None
    samplingBlockSizeBP: int | None
    binQuantileCutoff: float | None
    EB_minLin: float | None
    EB_use: bool | None
    EB_setNu0: int | None
    EB_setNuL: int | None
    numNearest: int | None
    sparseSupportScaleBP: float | None
    sparseSupportPrior: float | None
    restrictLocalAR1ToSparseBed: bool | None
    pad: float | None


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
    :param effectiveInfoRescale: If True, inflate the default model-based uncertainty track by
        block-level effective-information correction factors estimated from standardized
        one-step-ahead innovations via Bartlett/Newey-West long-run variance.
    :type effectiveInfoRescale: bool | None
    :param effectiveInfoBlockLengthBP: Genomic block length, in bp, for HAC
        uncertainty scaling. Non-positive values fall back to chromosome-wide scaling.
    :type effectiveInfoBlockLengthBP: int | None
    """

    stateInit: float
    stateCovarInit: float
    boundState: bool
    stateLowerBound: float
    stateUpperBound: float
    effectiveInfoRescale: bool | None
    effectiveInfoBlockLengthBP: int | None


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
        in ``reads`` or ``read1`` mode. When omitted and ``inferFragmentLength > 0``, treatment BAMs are
        inferred and paired controls reuse those inferred lengths.
    :type extendFrom5pBP: List[int] | int | None
    :param maxInsertSize: Maximum frag length/insert to consider when estimating fragment length.
    :type maxInsertSize: int
    :param inferFragmentLength: Intended for single-end data: if > 0, the maximum correlation lag
       (avg.) between *strand-specific* read tracks is taken as the fragment length estimate and used to
       extend reads from shifted 5' ends when ``bamInputMode`` resolves to ``reads`` or ``read1``.
       This is often important when targeting broader marks (e.g., ChIP-seq H3K27me3).
    :type inferFragmentLength: int
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
    inferFragmentLength: Optional[int] = 0
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
    :param countMode: Optional counting mode label
      BAM inputs default to `coverage`; fragments inputs default to `cutsite`
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
    :param backgroundBlockSizeBP: Length (bp) of blocks used to estimate local statistics (background, noise, etc.). If a negative value is provided (default), this value is inferred from the data using :func:`consenrich.core.getContextSize`.
    :type backgroundBlockSizeBP: int
    :param normMethod: Method used to normalize read counts for sequencing depth / library size.

        - ``EGS``: Effective Genome Size normalization (see :func:`consenrich.detrorm.getScaleFactor1x`)
          only appropriate for alignment coverage, not fragments pseudobulks

        - ``SF``: Median of ratios scale factors (see :func:`consenrich.cconsenrich.cSF`). Restricted to analyses with ``>= 3`` samples (no input control).

        - ``RPKM`` / ``CPM``: Scale factors based on emitted counts per million mapped units
          fragments pseudobulks use emitted insertions rather than raw fragment totals

    :type normMethod: str
    :param fragmentsGroupNorm: Optional extra normalization for fragments pseudobulks
      `NONE` keeps library-size scaling only and `CELLS` additionally divides by selected cell count
    :type fragmentsGroupNorm: str | None
    :param fixControl: If True, treatment samples are not upscaled, and control samples are not downscaled.
    :type fixControl: bool, optional
    :param globalWeight: Preprocessing centering weight. Any positive value applies subtraction of the dense centering offset estimated from the transformed coverage track, while non-positive values skip preprocessing centering entirely.
    :type globalWeight: float, optional
    :param logOffset: A small constant added to read normalized counts before log-transforming (pseudocount). For example,  :math:`\log(x + 1)` for ``logOffset = 1``. Default is ``1.0``.
    :type logOffset: float, optional
    :param logMult: Multiplicative factor applied to log-scaled and normalized counts. For example, setting ``logMult = 1 / \log(2)`` will yield log2-scaled counts after transformation, and setting ``logMult = 1.0`` yields natural log-scaled counts.
    :type logMult: float, optional
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
    globalWeight: float | None
    logOffset: float | None
    logMult: float | None


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
    defaultCountMode: str | None = "cutsite"
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
    :param writeUncertainty: If True, write the state uncertainty track to bedGraph. By default this is
        :math:`\sqrt{c_{\mathrm{eff},i}\,\widetilde{P}_{[i,0,0]}}` when
        ``stateParams.effectiveInfoRescale=True``, where :math:`c_{\mathrm{eff},i} \ge 1`
        is a correction factor estimated from
        standardized innovations. Otherwise it is :math:`\sqrt{\widetilde{P}_{[i,0,0]}}`.
    :type writeUncertainty: bool
    :param writeJackknifeSE: If True, write the standard error of the signal level estimates across jackknife replicates to bedGraph. This is only relevant if `applyJackknife` is True.
    :type writeJackknifeSE: bool
    :param applyJackknife: If True, estimate replicate-level sampling variability in the signal level estimates with the jackknife
    :type applyJackknife: bool

    """

    convertToBigWig: bool
    roundDigits: int
    writeUncertainty: bool
    writeJackknifeSE: bool
    applyJackknife: bool


class fitParams(NamedTuple):
    r"""Parameters controlling the optimization/fitting procedures.

    These arguments control both the inner routine in :func:`consenrich.cconsenrich.cinnerEM`
    and the outer calibration loop in :func:`consenrich.core.runConsenrich`.

    Inner loop:

    1. Filter-smoother state estimation *given* current noise scales
    2. Interval-level Student-t precision reweighting at: \(\lambda_{[j,i]}\) and \(\kappa_{[i]}\)
    3. Replicate-level observation offset updates: \(b_j\)

    Outer loop:

    1. update a shared background track \(g_i\), optionally constrained to have mean zero

    The default fit keeps replicate-level bias calibration and robust precision reweighting on.


    :param EM_maxIters: Maximum inner EM iterations.
    :type EM_maxIters: int
    :param EM_use: If False, skip the iterative EM / outer calibration updates entirely and treat the input observation variance track as a fixed plugin.
      This is compatible with ``EM_useAPN=True`` for adaptive process-noise filtering without iterative refitting.
    :type EM_use: bool
    :param EM_innerRtol: Relative tolerance used for the inner NLL stabilization test.
      The inner loop is treated as stable once
      ``abs(NLL_k - NLL_{k-1}) <= EM_innerRtol * max(abs(NLL_k), abs(NLL_{k-1}), 1)``
      for two consecutive iterations.
    :type EM_innerRtol: float
    :param EM_tNu: Student-t df for reweighting strengths (smaller = stronger reweighting)
    :type EM_tNu: float
    :param EM_useObsPrecReweight: If True, update observation noise precision multipliers \(\lambda_{[j,i]}\) (Student-\(t\) reweighting); otherwise \(\lambda\equiv 1\).
    :type EM_useObsPrecReweight: bool
    :param EM_useProcPrecReweight: If True, update process noise precision multipliers \(\kappa_{[i]}\) (Student-\(t\) reweighting); otherwise \(\kappa\equiv 1\).
    :type EM_useProcPrecReweight: bool
    :param EM_useAPN: If True, use the adaptive-process-noise (APN) D-statistic update during filtering.
      This option disables ``EM_useProcPrecReweight``.
    :type EM_useAPN: bool
    :param EM_useReplicateBias: If True, estimate additive replicate offsets \(b_j\) in the observation equation.
    :type EM_useReplicateBias: bool
    :param EM_zeroCenterBackground: If True, enforce the identifiability constraint that the shared smooth background has mean zero.
    :type EM_zeroCenterBackground: bool
    :param EM_zeroCenterReplicateBias: If True, enforce the identifiability constraint that replicate offsets have weighted mean zero.
    :type EM_zeroCenterReplicateBias: bool
    :param EM_repBiasShrink: Non-negative shrinkage applied to replicate bias estimates after optional centering.
    :type EM_repBiasShrink: float
    :param EM_outerIters: Number of outer alternations between the inner Kalman-EM fit and shared background update.
    :type EM_outerIters: int
    :param EM_outerRtol: Relative tolerance used to stop the outer background loop early.
      The outer loop stops once the maximum pointwise background update is at most
      ``EM_outerRtol * max(max(abs(g_next)), max(abs(g_cur)), 1)``.
    :type EM_outerRtol: float
    :param EM_backgroundSmoothness: Multiplier applied to the second-difference roughness penalty used for the shared background update.
    :type EM_backgroundSmoothness: float


    :seealso: :func:`consenrich.cconsenrich.cinnerEM`, :func:`consenrich.core.runConsenrich`, :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitVarianceFunction`
    """

    EM_maxIters: int | None = 50
    EM_use: bool | None = True
    EM_innerRtol: float | None = 1.0e-4
    EM_tNu: float | None = 8.0
    EM_useObsPrecReweight: bool | None = True
    EM_useProcPrecReweight: bool | None = True
    EM_useAPN: bool | None = False
    EM_useReplicateBias: bool | None = True
    EM_zeroCenterBackground: bool | None = True
    EM_zeroCenterReplicateBias: bool | None = True
    EM_repBiasShrink: float | None = 0.0
    EM_outerIters: int | None = 3
    EM_outerRtol: float | None = 1.0e-3
    EM_backgroundSmoothness: float | None = 1.0


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

    :seealso: :func:`getChromRanges`, :func:`cconsenrich.cgetFirstChromRead`, :func:`cconsenrich.cgetLastChromRead`
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

    try:
        for sourceIndex, sourcePath in enumerate(sourcePaths):
            sourceKind = sourceKinds[sourceIndex]
            source = sources[sourceIndex]
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
                countMode = _normalizeCountMode(source.countMode, "cutsite")
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
    finally:
        for tempPath in tempPaths:
            try:
                os.remove(tempPath)
            except Exception:
                pass

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


def constructMatrixQ(
    minDiagQ: float,
    offDiagQ: float = 0.0,
    Q00: Optional[float] = None,
    Q01: Optional[float] = None,
    Q10: Optional[float] = None,
    Q11: Optional[float] = None,
    useIdentity: float = -1.0,
    tol: float = 1.0e-8,  # conservative
    useWhiteAccel: bool = False,
    useDiscreteConstAccel: bool = False,
    deltaF: Optional[float] = None,
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

    if useWhiteAccel and useDiscreteConstAccel:
        raise ValueError(
            "Only one of `useWhiteAccel` or `useDiscreteConstAccel` can be True."
        )

    Q = np.empty((2, 2), dtype=np.float32)

    if useWhiteAccel or useDiscreteConstAccel:
        d = float(offDiagQ) if deltaF is None else float(deltaF)
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError(
                "`deltaF` (or fallback `offDiagQ`) must be a positive finite step size."
            )

        qa = float(minDiagQ)
        if not np.isfinite(qa) or qa <= 0.0:
            raise ValueError(
                "`minDiagQ` must be positive and finite in accel-based Q overrides."
            )

        if useWhiteAccel:
            Q[0, 0] = np.float32(qa * (d**3) / 3.0)
            Q[0, 1] = np.float32(qa * (d**2) / 2.0)
            Q[1, 0] = Q[0, 1]
            Q[1, 1] = np.float32(qa * d)
        else:
            Q[0, 0] = np.float32(qa * (d**4) / 4.0)
            Q[0, 1] = np.float32(qa * (d**3) / 2.0)
            Q[1, 0] = Q[0, 1]
            Q[1, 1] = np.float32(qa * (d**2))

        try:
            np.linalg.cholesky(Q.astype(np.float64, copy=False) + tol * np.eye(2))
        except Exception as ex:
            raise ValueError(
                f"Process noise covariance Q is not positive definite:\n{Q}"
            ) from ex
        return Q

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


def estimateAutoDeltaF(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    minQ: float,
    offDiagQ: float,
    stateInit: float,
    stateCovarInit: float,
    blockLenIntervals: int,
    pad: float = 1.0e-4,
    autoDeltaF_low: float = 1.0e-4,
    autoDeltaF_high: float = 2.0,
    autoDeltaF_init: float = 0.01,
    autoDeltaF_maxEvals: int = 25,
    useWhiteAccel: bool = False,
    useDiscreteConstAccel: bool = False,
) -> float:
    r"""Estimate ``deltaF`` from one transformed chromosome matrix."""
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

    trackCount, intervalCount = matrixData.shape
    if intervalCount < 2:
        raise ValueError("need at least 2 intervals for autoDeltaF estimation")

    blockLenIntervals_ = max(int(blockLenIntervals), 1)
    blockCount = int(np.ceil(intervalCount / float(blockLenIntervals_)))
    intervalToBlockMap = (
        np.arange(intervalCount, dtype=np.int32) // blockLenIntervals_
    ).astype(np.int32)
    intervalToBlockMap[intervalToBlockMap >= blockCount] = blockCount - 1

    def buildMatrixF(deltaFLocal: float) -> np.ndarray:
        return constructMatrixF(float(deltaFLocal)).astype(np.float32, copy=False)

    def buildMatrixQ0(deltaFLocal: float) -> np.ndarray:
        return constructMatrixQ(
            minDiagQ=float(minQ),
            offDiagQ=float(offDiagQ),
            useWhiteAccel=bool(useWhiteAccel),
            useDiscreteConstAccel=bool(useDiscreteConstAccel),
            deltaF=float(deltaFLocal),
        ).astype(np.float32, copy=False)

    deltaFMin = float(autoDeltaF_low)
    deltaFMax = float(autoDeltaF_high)
    if (
        (not np.isfinite(deltaFMin))
        or (not np.isfinite(deltaFMax))
        or deltaFMin <= 0.0
        or deltaFMax <= deltaFMin
    ):
        deltaFMin, deltaFMax = 1.0e-4, 1.0

    deltaFInit = float(autoDeltaF_init)
    if (not np.isfinite(deltaFInit)) or deltaFInit <= 0.0:
        deltaFInit = float(np.sqrt(deltaFMin * deltaFMax))
    deltaFInit = float(np.clip(deltaFInit, deltaFMin, deltaFMax))

    qScaleUnity = np.ones(blockCount, dtype=np.float32)

    stateForward = np.empty((intervalCount, 2), dtype=np.float32)
    stateCovarForward = np.empty((intervalCount, 2, 2), dtype=np.float32)
    pNoiseForward = np.empty((intervalCount, 2, 2), dtype=np.float32)
    vectorD = np.empty(intervalCount, dtype=np.float32)

    stateSmoothed = np.empty((intervalCount, 2), dtype=np.float32)
    stateCovarSmoothed = np.empty((intervalCount, 2, 2), dtype=np.float32)
    lagCovSmoothed = np.empty((max(intervalCount - 1, 1), 2, 2), dtype=np.float32)
    postFitResiduals = np.empty((intervalCount, trackCount), dtype=np.float32)

    def _penNLL(deltaF_candidate: float) -> tuple[float, float]:
        deltaF_candidate = float(deltaF_candidate)
        if (not np.isfinite(deltaF_candidate)) or deltaF_candidate <= 0.0:
            return float(1.0e16), float(1.0e16)

        try:
            matrixF_candidate = buildMatrixF(deltaF_candidate)
            matrixQ0_candidate = buildMatrixQ0(deltaF_candidate)
        except Exception:
            return float(1.0e16), float(1.0e16)

        try:
            out = cconsenrich.cforwardPass(
                matrixData=matrixData,
                matrixPluginMuncInit=matrixMunc,
                matrixF=matrixF_candidate,
                matrixQ0=matrixQ0_candidate,
                intervalToBlockMap=intervalToBlockMap,
                qScale=qScaleUnity,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                pad=float(pad),
                projectStateDuringFiltering=False,
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
                lambdaExp=None,
                processPrecExp=None,
                EM_useObsPrecReweight=False,
                EM_useProcPrecReweight=False,
                EM_useAPN=False,
            )
            sumNLL = float(out[3])

            cconsenrich.cbackwardPass(
                matrixData=matrixData,
                matrixF=matrixF_candidate,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                chunkSize=0,
                stateSmoothed=stateSmoothed,
                stateCovarSmoothed=stateCovarSmoothed,
                lagCovSmoothed=lagCovSmoothed,
                postFitResiduals=postFitResiduals,
                progressBar=None,
                progressIter=0,
            )

            if intervalCount <= 1:
                return float(sumNLL), 0.0

            mu = stateSmoothed.astype(np.float64, copy=False)
            P = stateCovarSmoothed.astype(np.float64, copy=False)
            C = lagCovSmoothed.astype(np.float64, copy=False)

            L = intervalCount - 1
            deltaMu0 = mu[1:, 0] - mu[:-1, 0]
            expDelta2 = (
                (deltaMu0 * deltaMu0) + P[1:, 0, 0] + P[:-1, 0, 0] - 2.0 * C[:L, 0, 0]
            )
            expDelta2 = np.maximum(expDelta2, 0.0)
            roughnessMean = float(np.mean(expDelta2))
            return float(sumNLL), float(roughnessMean)
        except Exception:
            return float(1.0e16), float(1.0e16)

    tLOW = float(np.log(deltaFMin))
    tHIGH = float(np.log(deltaFMax))

    gridDeltaF = np.exp(np.linspace(tLOW, tHIGH, num=16, dtype=np.float64))
    gridTerms = []
    for d in gridDeltaF:
        sumNLL_g, rough_g = _penNLL(float(d))
        if sumNLL_g >= 1.0e16 or rough_g >= 1.0e16:
            continue

        nll_per_obs = sumNLL_g / (float(trackCount) * float(intervalCount))
        rough_log = float(np.log1p(rough_g))
        if np.isfinite(nll_per_obs) and np.isfinite(rough_log):
            gridTerms.append((float(nll_per_obs), float(rough_log)))

    if gridTerms:
        nll_ref = float(np.median([t[0] for t in gridTerms]))
        rough_ref = float(np.median([t[1] for t in gridTerms]))
    else:
        nll_ref = 1.0
        rough_ref = 1.0
    nll_ref = float(max(nll_ref, 1.0e-12))
    rough_ref = float(max(rough_ref, 1.0e-12))

    def deltaF_score(deltaF_candidate: float, w1=0.95, w2=0.05) -> float:
        sumNLL, roughnessMean = _penNLL(deltaF_candidate)
        nll_term = (sumNLL / (float(trackCount) * float(intervalCount))) / nll_ref
        rough_term = float(np.log1p(roughnessMean)) / rough_ref
        return float((w1 * nll_term) + (w2 * rough_term))

    def obj(t: float) -> float:
        return float(deltaF_score(float(np.exp(float(t)))))

    try:
        res = optimize.minimize_scalar(
            obj,
            bounds=(tLOW, tHIGH),
            method="bounded",
            options={"maxiter": int(autoDeltaF_maxEvals), "xatol": 1.0e-4},
        )
        if (not res.success) or (not np.isfinite(res.x)):
            bestDeltaF = deltaFInit
        else:
            bestDeltaF = float(np.exp(float(res.x)))
    except Exception:
        bestDeltaF = deltaFInit

    bestDeltaF = float(np.clip(bestDeltaF, deltaFMin, deltaFMax))
    bestScore = float(deltaF_score(bestDeltaF))
    logger.info(
        "autoDeltaF search completed: bestDeltaF=%s\tbestScore=%.4e",
        bestDeltaF,
        bestScore,
    )
    return float(bestDeltaF)


def _bartlettEffectiveInfoCorrection(
    standardizedInnovations: np.ndarray,
    bandwidth: int,
    eps: float = 1.0e-12,
) -> dict[str, float | int]:
    r"""Estimate a Bartlett/Newey-West effective correction factor for the state variance.

    The input should already be a matrix of standardized innovations with rows corresponding
    to replicates and columns to intervals.
    """

    z = np.asarray(standardizedInnovations, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    elif z.ndim != 2:
        raise ValueError("`standardizedInnovations` must be 1D or 2D")

    bw = max(int(bandwidth), 0)
    if z.size == 0 or bw <= 0:
        return {
            "bandwidth": int(bw),
            "gamma0": 0.0,
            "lrv": 0.0,
            "correctionFactor": 1.0,
            "effectiveInfoFraction": 1.0,
            "effectiveSampleSizeFraction": 1.0,
            "numSeries": 0,
            "seriesLength": 0,
        }

    centeredSeries: list[np.ndarray] = []
    for j in range(int(z.shape[0])):
        zj = np.asarray(z[j], dtype=np.float64)
        zj = zj[np.isfinite(zj)]
        if zj.size < 2:
            continue
        zj = zj - float(np.mean(zj))
        centeredSeries.append(zj)

    if not centeredSeries:
        return {
            "bandwidth": int(bw),
            "gamma0": 0.0,
            "lrv": 0.0,
            "correctionFactor": 1.0,
            "effectiveInfoFraction": 1.0,
            "effectiveSampleSizeFraction": 1.0,
            "numSeries": 0,
            "seriesLength": 0,
        }

    maxUsableLag = min(int(bw), max(int(arr.size) - 1 for arr in centeredSeries))
    if maxUsableLag <= 0:
        gamma0List = [
            float(np.dot(arr, arr) / max(int(arr.size), 1)) for arr in centeredSeries
        ]
        gamma0 = float(np.mean(gamma0List)) if gamma0List else 0.0
        correctionFactor = float(max(1.0, gamma0))
        effectiveInfoFraction = float(1.0 / correctionFactor)
        return {
            "bandwidth": int(maxUsableLag),
            "gamma0": float(max(gamma0, 0.0)),
            "lrv": float(max(gamma0, 0.0)),
            "correctionFactor": float(correctionFactor),
            "effectiveInfoFraction": float(effectiveInfoFraction),
            "effectiveSampleSizeFraction": float(effectiveInfoFraction),
            "numSeries": int(len(centeredSeries)),
            "seriesLength": int(max(arr.size for arr in centeredSeries)),
        }

    gamma = np.zeros(maxUsableLag + 1, dtype=np.float64)
    for h in range(maxUsableLag + 1):
        gammaHList: list[float] = []
        for arr in centeredSeries:
            t = int(arr.size)
            if t <= h:
                continue
            if h == 0:
                gammaH = float(np.dot(arr, arr) / float(t))
            else:
                gammaH = float(np.dot(arr[h:], arr[:-h]) / float(t - h))
            if np.isfinite(gammaH):
                gammaHList.append(gammaH)
        gamma[h] = float(np.mean(gammaHList)) if gammaHList else np.nan

    gamma0 = float(gamma[0]) if np.isfinite(gamma[0]) else 0.0
    if gamma0 <= float(eps):
        return {
            "bandwidth": int(maxUsableLag),
            "gamma0": float(max(gamma0, 0.0)),
            "lrv": float(max(gamma0, 0.0)),
            "correctionFactor": 1.0,
            "effectiveInfoFraction": 1.0,
            "effectiveSampleSizeFraction": 1.0,
            "numSeries": int(len(centeredSeries)),
            "seriesLength": int(max(arr.size for arr in centeredSeries)),
        }

    lrv = gamma0
    for h in range(1, maxUsableLag + 1):
        gammaH = float(gamma[h]) if np.isfinite(gamma[h]) else 0.0
        weight = 1.0 - (float(h) / float(maxUsableLag + 1))
        lrv += 2.0 * weight * gammaH

    correctionFactor = float(max(1.0, lrv))
    effectiveInfoFraction = float(1.0 / correctionFactor)
    return {
        "bandwidth": int(maxUsableLag),
        "gamma0": float(max(gamma0, 0.0)),
        "lrv": float(max(lrv, 0.0)),
        "correctionFactor": float(correctionFactor),
        "effectiveInfoFraction": float(effectiveInfoFraction),
        "effectiveSampleSizeFraction": float(effectiveInfoFraction),
        "numSeries": int(len(centeredSeries)),
        "seriesLength": int(max(arr.size for arr in centeredSeries)),
    }


def _shrunkenBlockEffectiveInfoCorrection(
    standardizedInnovations: np.ndarray,
    bandwidth: int,
    blockLengthIntervals: int | None,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    r"""Estimate blockwise HAC factors shrunk toward the chromosome-wide HAC factor"""

    z = np.asarray(standardizedInnovations, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    elif z.ndim != 2:
        raise ValueError("`standardizedInnovations` must be 1D or 2D")

    n = int(z.shape[1])
    bw = max(int(bandwidth), 0)
    chrDiag = _bartlettEffectiveInfoCorrection(z, bandwidth=int(bw), eps=float(eps))
    chrFactor = float(chrDiag.get("correctionFactor", 1.0))
    if (not np.isfinite(chrFactor)) or chrFactor <= 0.0:
        chrFactor = 1.0
    chrFactor = float(max(1.0, chrFactor))

    if n <= 0:
        out = dict(chrDiag)
        out.update(
            {
                "chromosomeFactor": chrFactor,
                "blockLengthIntervals": 0,
                "numBlocks": 0,
                "intervalFactors": np.ones(0, dtype=np.float32),
                "blockFactorMin": chrFactor,
                "blockFactorMedian": chrFactor,
                "blockFactorMax": chrFactor,
                "shrinkageWeightMean": 0.0,
            }
        )
        return out

    if blockLengthIntervals is None:
        # case: whole-chromosome blocks
        blockLen = n
    else:
        blockLen = int(blockLengthIntervals)
        if blockLen <= 0:
            blockLen = n
    blockLen = max(1, min(int(blockLen), n))

    factors = np.full(n, chrFactor, dtype=np.float64)
    rawBlockFactors: list[float] = []
    shrunkBlockFactors: list[float] = []
    shrinkageWeights: list[float] = []
    for start in range(0, n, blockLen):
        end = min(start + blockLen, n)
        blockZ = z[:, start:end]
        blockBw = min(int(bw), max(int(end - start) - 1, 0))
        # raw block factors are from HAC alone
        blockDiag = _bartlettEffectiveInfoCorrection(
            blockZ,
            bandwidth=int(blockBw),
            eps=float(eps),
        )
        rawFactor = float(blockDiag.get("correctionFactor", chrFactor))
        if (
            int(blockDiag.get("numSeries", 0)) <= 0
            or int(blockDiag.get("seriesLength", 0)) <= 1
            or (not np.isfinite(rawFactor))
            or rawFactor <= 0.0
        ):
            rawFactor = chrFactor
            weight = 0.0
        else:
            rawFactor = float(max(1.0, rawFactor))
            usedBw = max(int(blockDiag.get("bandwidth", 0)), 0)
            seriesLength = max(int(blockDiag.get("seriesLength", 0)), 0)
            numSeries = max(int(blockDiag.get("numSeries", 0)), 0)

            # shrink wrt the effective number of observations in the block
            effObs = float(numSeries) * (
                float(seriesLength) / float(max(2 * usedBw + 1, 1))
            )
            weight = float(effObs / (effObs + 1.0)) if effObs > 0.0 else 0.0
            weight = float(np.clip(weight, 0.0, 1.0))

        # shrink block factors toward chromosome-level
        shrunkFactor = float(
            np.exp(
                weight * np.log(max(rawFactor, float(eps)))
                + (1.0 - weight) * np.log(max(chrFactor, float(eps)))
            )
        )
        if (not np.isfinite(shrunkFactor)) or shrunkFactor <= 0.0:
            shrunkFactor = chrFactor
        shrunkFactor = float(max(1.0, shrunkFactor))
        factors[start:end] = shrunkFactor
        rawBlockFactors.append(float(rawFactor))
        shrunkBlockFactors.append(float(shrunkFactor))
        shrinkageWeights.append(float(weight))

    out = dict(chrDiag)
    out.update(
        {
            "chromosomeFactor": chrFactor,
            "blockLengthIntervals": int(blockLen),
            "numBlocks": int(len(shrunkBlockFactors)),
            "intervalFactors": factors.astype(np.float32, copy=False),
            "blockFactorMin": float(np.min(shrunkBlockFactors)),
            "blockFactorMedian": float(np.median(shrunkBlockFactors)),
            "blockFactorMax": float(np.max(shrunkBlockFactors)),
            "rawBlockFactorMin": float(np.min(rawBlockFactors)),
            "rawBlockFactorMedian": float(np.median(rawBlockFactors)),
            "rawBlockFactorMax": float(np.max(rawBlockFactors)),
            "shrinkageWeightMean": float(np.mean(shrinkageWeights)),
        }
    )
    return out


def _computeInnovationEffectiveInfoDiagnostics(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    stateForward: np.ndarray,
    stateCovarForward: np.ndarray,
    pNoiseForward: np.ndarray,
    matrixF: np.ndarray,
    replicateBias: np.ndarray | None,
    lambdaExp: np.ndarray | None,
    pad: float,
    effectiveInfoBandwidthIntervals: int | None,
    effectiveInfoBlockLengthIntervals: int | None = None,
) -> dict[str, Any]:
    r"""Estimate the variance correction factor from one-step forward innovations."""

    n = int(matrixData.shape[1])
    if effectiveInfoBandwidthIntervals is None:
        bw = max(1, min(32, n // 8))
    else:
        bw = max(int(effectiveInfoBandwidthIntervals), 0)

    if n < 3 or bw <= 0:
        return {
            "bandwidth": int(max(bw, 0)),
            "gamma0": 0.0,
            "lrv": 0.0,
            "correctionFactor": 1.0,
            "effectiveInfoFraction": 1.0,
            "effectiveSampleSizeFraction": 1.0,
            "numSeries": 0,
            "seriesLength": max(0, n - 1),
            "blockLengthIntervals": 0,
            "numBlocks": 0,
            "intervalFactors": np.ones(n, dtype=np.float32),
        }

    dataArr = np.asarray(matrixData, dtype=np.float64)
    muncArr = np.asarray(matrixMunc, dtype=np.float64)
    stateForwardArr = np.asarray(stateForward, dtype=np.float64)
    stateCovarForwardArr = np.asarray(stateCovarForward, dtype=np.float64)
    pNoiseForwardArr = np.asarray(pNoiseForward, dtype=np.float64)
    matrixFArr = np.asarray(matrixF, dtype=np.float64)

    biasArr = np.zeros((dataArr.shape[0], 1), dtype=np.float64)
    if replicateBias is not None:
        biasArr = np.asarray(replicateBias, dtype=np.float64)[:, None]
    # state-transition
    f00 = float(matrixFArr[0, 0])
    f01 = float(matrixFArr[0, 1])

    # get one-step predictions and innovation variances from the forward pass
    xPred = f00 * stateForwardArr[:-1, 0].astype(
        np.float64, copy=False
    ) + f01 * stateForwardArr[:-1, 1].astype(np.float64, copy=False)

    pPred00 = (
        (f00 * f00) * stateCovarForwardArr[:-1, 0, 0].astype(np.float64, copy=False)
        + (f00 * f01)
        * (
            stateCovarForwardArr[:-1, 0, 1].astype(np.float64, copy=False)
            + stateCovarForwardArr[:-1, 1, 0].astype(np.float64, copy=False)
        )
        + (f01 * f01) * stateCovarForwardArr[:-1, 1, 1].astype(np.float64, copy=False)
        + pNoiseForwardArr[:-1, 0, 0].astype(np.float64, copy=False)
    )
    pPred00 = np.maximum(pPred00, 0.0)
    # plugin observation noise level
    obsVar = muncArr[:, 1:] + float(pad)
    if lambdaExp is not None:
        lambdaArr = np.asarray(lambdaExp, dtype=np.float64)[:, 1:]
        lambdaArr = np.maximum(lambdaArr, 1.0e-8)
        obsVar = obsVar / lambdaArr
    obsVar = np.maximum(obsVar, 1.0e-12)

    totalVar = np.sqrt(np.maximum(obsVar + pPred00[None, :], 1.0e-12))
    z = (dataArr[:, 1:] - biasArr - xPred[None, :]) / totalVar
    diag = _shrunkenBlockEffectiveInfoCorrection(
        z,
        bandwidth=int(bw),
        blockLengthIntervals=effectiveInfoBlockLengthIntervals,
    )
    innovationFactors = np.asarray(
        diag.get("intervalFactors", np.ones(max(0, n - 1), dtype=np.float32)),
        dtype=np.float32,
    )
    intervalFactors = np.ones(n, dtype=np.float32)
    if innovationFactors.size:
        intervalFactors[0] = innovationFactors[0]
        intervalFactors[1:] = innovationFactors[: n - 1]
    correctionFactor = (
        float(np.median(intervalFactors)) if intervalFactors.size else 1.0
    )
    diag["intervalFactors"] = intervalFactors
    diag["correctionFactor"] = float(max(1.0, correctionFactor))
    diag["effectiveInfoFraction"] = float(
        1.0 / max(float(diag["correctionFactor"]), 1.0e-12)
    )
    diag["effectiveSampleSizeFraction"] = float(diag["effectiveInfoFraction"])
    return diag


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
    disableCalibration: bool = False,
    EM_maxIters: int = 50,
    EM_innerRtol: float = 1.0e-4,
    EM_tNu: float = 8.0,
    EM_useObsPrecReweight: bool = True,
    EM_useProcPrecReweight: bool = True,
    EM_useAPN: bool = False,
    EM_useReplicateBias: bool = True,
    EM_zeroCenterBackground: bool = True,
    EM_zeroCenterReplicateBias: bool = True,
    EM_repBiasShrink: float = 0.0,
    EM_outerIters: int = 3,
    EM_outerRtol: float = 1.0e-3,
    EM_backgroundSmoothness: float = 1.0,
    returnScales: bool = True,
    returnReplicateOffsets: bool = False,
    applyJackknife: bool = False,
    jackknifeEM_maxIters: int = 5,
    jackknifeEM_innerRtol: float = 1.0e-2,
    useWhiteAccel: bool = False,
    useDiscreteConstAccel: bool = False,
    autoDeltaF: bool = True,
    autoDeltaF_low: float = 1.0e-4,
    autoDeltaF_high: float = 2.0,
    autoDeltaF_init: float = 0.01,
    autoDeltaF_maxEvals: int = 25,
    effectiveInfoRescale: bool = True,
    effectiveInfoBandwidthIntervals: int | None = None,
    effectiveInfoBlockLengthIntervals: int | None = None,
):
    r"""Run Consenrich over a contiguous genomic region

    Consenrich estimates a shared signal level from multiple replicate tracks using a two-state
    linear smoother plus an outer calibration loop.

    The observation model is

    .. math::

      y_{[j,i]} = g_{[i]} + x_{[i,0]} + b_j + \epsilon_{[j,i]},
      \qquad
      \mathrm{Var}(\epsilon_{[j,i]}) =
      \frac{v_{[j,i]} + \mathrm{pad}}{\lambda_{[j,i]}}.

    Here :math:`g_{[i]}` is a shared smooth background, :math:`b_j` is a
    replicate-level bias term, and :math:`v_{[j,i]}` is the plugin observation
    variance supplied by ``matrixMunc``. By default, identifiability is enforced
    by zero-centering the shared background and replicate offsets; those constraints
    can be disabled with ``EM_zeroCenterBackground=False`` and
    ``EM_zeroCenterReplicateBias=False``.

    The latent state follows

    .. math::

      \mathbf{x}_{[i+1]} = \mathbf{F}(\delta_F)\mathbf{x}_{[i]} + \eta_{[i]},
      \qquad
      \mathrm{Var}(\eta_{[i]}) = \frac{\mathbf{Q}_0}{\kappa_{[i]}}.

    If ``EM_useAPN=True``, the forward filter instead uses the adaptive-process-noise
    D-statistic update to scale :math:`\mathbf{Q}_0` and process-precision reweighting is disabled.

    If ``effectiveInfoRescale=True``, the default state uncertainty output is
    multiplied by effective-information correction factors estimated from
    standardized one-step-ahead innovations via Bartlett/Newey-West long-run variance.

    This wrapper ties together several fundamental routines written in Cython:

    #. :func:`consenrich.cconsenrich.cforwardPass`: Forward filter (predict, update)
    #. :func:`consenrich.cconsenrich.cbackwardPass`: Backward fixed-interval smoother
    #. :func:`consenrich.cconsenrich.cinnerEM`: Joint optimization of robust precision reweighting and replicate-level observation calibration

    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.cconsenrich.cTransform`, :func:`consenrich.cconsenrich.cforwardPass`, :func:`consenrich.cconsenrich.cbackwardPass`, :func:`consenrich.cconsenrich.cinnerEM`
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

    trackCount, intervalCount = matrixData.shape
    if intervalCount < 2:
        raise ValueError("need at least 2 intervals for smoothing")

    if applyJackknife and trackCount < 3:
        raise ValueError("`applyJackknife` requires at least 3 replicates")

    EM_useAPN = bool(EM_useAPN)
    if EM_useAPN:
        EM_useProcPrecReweight = False

    blockCount = int(np.ceil(intervalCount / float(blockLenIntervals)))
    intervalToBlockMap = (
        np.arange(intervalCount, dtype=np.int32) // blockLenIntervals
    ).astype(np.int32)
    intervalToBlockMap[intervalToBlockMap >= blockCount] = blockCount - 1
    if effectiveInfoBandwidthIntervals is None:
        effectiveInfoBandwidthIntervals = max(int((int(blockLenIntervals) - 1) // 4), 1)
    else:
        effectiveInfoBandwidthIntervals = max(int(effectiveInfoBandwidthIntervals), 1)
    if effectiveInfoBlockLengthIntervals is None:
        effectiveInfoBlockLengthIntervals = int(intervalCount)
    else:
        effectiveInfoBlockLengthIntervals = int(effectiveInfoBlockLengthIntervals)
        if effectiveInfoBlockLengthIntervals <= 0:
            effectiveInfoBlockLengthIntervals = int(intervalCount)
    effectiveInfoBlockLengthIntervals = max(
        1,
        min(int(effectiveInfoBlockLengthIntervals), int(intervalCount)),
    )

    # some pnoise/Q templates can depend on deltaF, hence the wrappers
    def buildMatrixF(deltaFLocal: float) -> np.ndarray:
        return constructMatrixF(float(deltaFLocal)).astype(np.float32, copy=False)

    def buildMatrixQ0(deltaFLocal: float) -> np.ndarray:
        # when deltaF determines Q, pnoise covariance can become ill-conditioned
        # for extreme step sizes. constructMatrixQ handles validity checks.
        return constructMatrixQ(
            minDiagQ=float(minQ),
            offDiagQ=float(offDiagQ),
            useWhiteAccel=bool(useWhiteAccel),
            useDiscreteConstAccel=bool(useDiscreteConstAccel),
            deltaF=float(deltaFLocal),
        ).astype(np.float32, copy=False)

    def _autoDeltaF(matrixDataLocal: np.ndarray, matrixMuncLocal: np.ndarray) -> float:
        return estimateAutoDeltaF(
            matrixData=matrixDataLocal,
            matrixMunc=matrixMuncLocal,
            minQ=float(minQ),
            offDiagQ=float(offDiagQ),
            stateInit=float(stateInit),
            stateCovarInit=float(stateCovarInit),
            blockLenIntervals=int(blockLenIntervals),
            pad=float(pad),
            autoDeltaF_low=float(autoDeltaF_low),
            autoDeltaF_high=float(autoDeltaF_high),
            autoDeltaF_init=float(autoDeltaF_init),
            autoDeltaF_maxEvals=int(autoDeltaF_maxEvals),
            useWhiteAccel=bool(useWhiteAccel),
            useDiscreteConstAccel=bool(useDiscreteConstAccel),
        )

    inferDeltaF = bool(autoDeltaF) or (deltaF < 0.0)

    def _runForwardBackward(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        qScale: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        lambdaExp: np.ndarray | None,
        processPrecExp: np.ndarray | None,
        replicateBias: np.ndarray | None,
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
            qScale=qScale,
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
            EM_useObsPrecReweight=bool(EM_useObsPrecReweight),
            EM_useProcPrecReweight=bool(EM_useProcPrecReweight),
            EM_useAPN=bool(EM_useAPN),
            EM_useReplicateBias=bool(EM_useReplicateBias),
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
                EM_useReplicateBias=bool(EM_useReplicateBias),
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

    def _fitOuter(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        emMaxItersLocal: int,
        emInnerRtolLocal: float,
    ) -> dict[str, np.ndarray | float | None]:
        mLocal = int(matrixDataLocal.shape[0])
        nLocal = int(matrixDataLocal.shape[1])

        if disableCalibration or mLocal < 2:
            currentBackground = np.zeros(nLocal, dtype=np.float32)
            currentMunc = np.ascontiguousarray(matrixMuncLocal, dtype=np.float32)
            qScaleLocal = np.ones(blockCount, dtype=np.float32)
            lambdaExpLocal = None
            processPrecExpLocal = None
            replicateBiasLocal = np.zeros(mLocal, dtype=np.float32)
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
                matrixDataLocal=matrixDataLocal,
                matrixMuncLocal=currentMunc,
                qScale=qScaleLocal,
                matrixFLocal=matrixFLocal,
                matrixQ0Local=matrixQ0Local,
                lambdaExp=lambdaExpLocal,
                processPrecExp=processPrecExpLocal,
                replicateBias=replicateBiasLocal,
            )
            return {
                "matrixMunc": currentMunc,
                "background": currentBackground,
                "qScale": qScaleLocal,
                "lambdaExp": lambdaExpLocal,
                "processPrecExp": processPrecExpLocal,
                "replicateBias": replicateBiasLocal,
                "stateForward": np.asarray(stateForwardLocal, dtype=np.float32),
                "stateCovarForward": np.asarray(
                    stateCovarForwardLocal, dtype=np.float32
                ),
                "pNoiseForward": np.asarray(pNoiseForwardLocal, dtype=np.float32),
                "stateSmoothed": np.asarray(stateSmoothedLocal, dtype=np.float32),
                "stateCovarSmoothed": np.asarray(
                    stateCovarSmoothedLocal, dtype=np.float32
                ),
                "lagCovSmoothed": np.asarray(lagCovSmoothedLocal, dtype=np.float32),
                "postFitResiduals": np.asarray(postFitResidualsLocal, dtype=np.float32),
                "NIS": np.asarray(NISLocal, dtype=np.float32),
                "sumNLL": float(sumNLLLocal),
            }

        currentBackground = np.zeros(nLocal, dtype=np.float32)
        currentMunc = np.ascontiguousarray(matrixMuncLocal, dtype=np.float32)

        lambdaExpLocal = None
        processPrecExpLocal = None
        qScaleLocal = np.ones(blockCount, dtype=np.float32)
        replicateBiasLocal = np.zeros(mLocal, dtype=np.float32)
        stateSmoothedLocal = None
        stateCovarSmoothedLocal = None
        lagCovSmoothedLocal = None
        postFitResidualsLocal = None
        NISLocal = None
        sumNLLLocal = np.nan

        outerIters = max(1, int(EM_outerIters))
        outerTol = float(max(EM_outerRtol, 0.0))

        for outerIter in range(outerIters):
            dataAdjusted = np.ascontiguousarray(
                matrixDataLocal - currentBackground[None, :],
                dtype=np.float32,
            )
            EM_out_local = cconsenrich.cinnerEM(
                matrixData=dataAdjusted,
                matrixPluginMuncInit=currentMunc,
                matrixF=matrixFLocal,
                matrixQ0=matrixQ0Local,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                EM_maxIters=int(emMaxItersLocal),
                EM_innerRtol=float(emInnerRtolLocal),
                pad=float(pad),
                EM_tNu=float(EM_tNu),
                returnIntermediates=True,
                EM_useObsPrecReweight=bool(EM_useObsPrecReweight),
                EM_useProcPrecReweight=bool(EM_useProcPrecReweight),
                EM_useAPN=bool(EM_useAPN),
                EM_useReplicateBias=bool(EM_useReplicateBias),
                EM_zeroCenterReplicateBias=bool(EM_zeroCenterReplicateBias),
                EM_repBiasShrink=float(EM_repBiasShrink),
                APN_minQ=float(minQ),
                APN_maxQ=float(maxQ),
            )
            if len(EM_out_local) != 10:
                raise ValueError(
                    "Expected cinnerEM(..., returnIntermediates=True) to return 10 values "
                    f"(got {len(EM_out_local)})."
                )

            (
                qScaleLocal,
                _itersEMDoneLocal,
                _nllEMLocal,
                stateSmoothedLocal,
                stateCovarSmoothedLocal,
                lagCovSmoothedLocal,
                postFitResidualsLocal,
                lambdaExpLocal,
                processPrecExpLocal,
                replicateBiasLocal,
            ) = EM_out_local

            qScaleLocal = np.asarray(qScaleLocal, dtype=np.float32)
            if lambdaExpLocal is not None:
                lambdaExpLocal = np.asarray(lambdaExpLocal, dtype=np.float32)
            if processPrecExpLocal is not None:
                processPrecExpLocal = np.asarray(processPrecExpLocal, dtype=np.float32)
            replicateBiasLocal = np.asarray(replicateBiasLocal, dtype=np.float32)
            stateSmoothedLocal = np.asarray(stateSmoothedLocal, dtype=np.float32)
            stateCovarSmoothedLocal = np.asarray(
                stateCovarSmoothedLocal, dtype=np.float32
            )
            lagCovSmoothedLocal = np.asarray(lagCovSmoothedLocal, dtype=np.float32)
            postFitResidualsLocal = np.asarray(postFitResidualsLocal, dtype=np.float32)

            invVarMatrix = 1.0 / np.maximum(currentMunc + float(pad), 1.0e-8)
            if lambdaExpLocal is not None:
                invVarMatrix *= np.asarray(lambdaExpLocal, dtype=np.float32)
            residualMatrix = (
                np.asarray(matrixDataLocal, dtype=np.float32)
                - np.asarray(replicateBiasLocal[:, None], dtype=np.float32)
                - np.asarray(stateSmoothedLocal[:, 0][None, :], dtype=np.float32)
            )
            nextBackground = _solveZeroCenteredBackground(
                residualMatrix=residualMatrix,
                invVarMatrix=invVarMatrix,
                blockLenIntervals=int(blockLenIntervals),
                backgroundSmoothness=float(EM_backgroundSmoothness),
                zeroCenter=bool(EM_zeroCenterBackground),
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
            bgTol = float(outerTol * bgScale)
            currentBackground = np.asarray(nextBackground, dtype=np.float32)
            logger.info(
                "outerEM[%d/%d]: backgroundShift=%.6g\tthreshold=%.6g",
                int(outerIter + 1),
                int(outerIters),
                float(bgChange),
                float(bgTol),
            )
            if bgChange <= bgTol:
                break

        dataAdjusted = np.ascontiguousarray(
            matrixDataLocal - currentBackground[None, :],
            dtype=np.float32,
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
            qScale=qScaleLocal,
            matrixFLocal=matrixFLocal,
            matrixQ0Local=matrixQ0Local,
            lambdaExp=lambdaExpLocal,
            processPrecExp=processPrecExpLocal,
            replicateBias=replicateBiasLocal,
        )
        return {
            "matrixMunc": currentMunc,
            "background": currentBackground,
            "qScale": np.asarray(qScaleLocal, dtype=np.float32),
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
        }

    if inferDeltaF:
        deltaF = _autoDeltaF(matrixData, matrixMunc)
    deltaF_fit = float(deltaF)

    matrixF = buildMatrixF(float(deltaF_fit))
    matrixQ0 = buildMatrixQ0(float(deltaF_fit))

    matrixDataFit = matrixData
    matrixMuncFit = matrixMunc

    fitFinal = _fitOuter(
        matrixDataLocal=matrixDataFit,
        matrixMuncLocal=matrixMuncFit,
        matrixFLocal=matrixF,
        matrixQ0Local=matrixQ0,
        emMaxItersLocal=int(EM_maxIters),
        emInnerRtolLocal=float(EM_innerRtol),
    )
    lambdaExp_final = fitFinal["lambdaExp"]
    replicateBias_final = np.asarray(fitFinal["replicateBias"], dtype=np.float32)
    qScale = np.asarray(fitFinal["qScale"], dtype=np.float32)
    matrixMuncFit = np.asarray(fitFinal["matrixMunc"], dtype=np.float32)
    sumNLL = float(fitFinal["sumNLL"])
    stateForward = np.asarray(fitFinal["stateForward"], dtype=np.float32)
    stateCovarForward = np.asarray(fitFinal["stateCovarForward"], dtype=np.float32)
    pNoiseForward = np.asarray(fitFinal["pNoiseForward"], dtype=np.float32)
    stateSmoothed = np.asarray(fitFinal["stateSmoothed"], dtype=np.float32)
    stateCovarSmoothed = np.asarray(fitFinal["stateCovarSmoothed"], dtype=np.float32)
    postFitResiduals = np.asarray(fitFinal["postFitResiduals"], dtype=np.float32)
    NIS = np.asarray(fitFinal["NIS"], dtype=np.float32)

    outStateSmoothed = np.asarray(stateSmoothed, dtype=np.float32)
    outStateCovarSmoothed = np.asarray(stateCovarSmoothed, dtype=np.float32)
    outPostFitResiduals = np.asarray(postFitResiduals, dtype=np.float32)

    outTrack4 = NIS

    # Jackknife:
    # If deltaF was inferred, each leave-one-out fit re-runs the same deltaF search
    # on the reduced replicate set. If deltaF was provided, deltaF stays fixed.

    if applyJackknife:
        meanLOO_x0 = np.zeros(intervalCount, dtype=np.float64)
        M2LOO_x0 = np.zeros(intervalCount, dtype=np.float64)

        for repIdx in range(trackCount):
            keepMask = np.ones(trackCount, dtype=bool)
            keepMask[repIdx] = False

            matrixData_LOO = np.ascontiguousarray(
                matrixData[keepMask, :], dtype=np.float32
            )
            matrixMunc_LOO = np.ascontiguousarray(
                matrixMunc[keepMask, :], dtype=np.float32
            )

            if inferDeltaF:
                deltaF_LOO = _autoDeltaF(matrixData_LOO, matrixMunc_LOO)
                matrixF_LOO = buildMatrixF(float(deltaF_LOO))
                matrixQ0_LOO = buildMatrixQ0(float(deltaF_LOO))
            else:
                matrixF_LOO = matrixF
                matrixQ0_LOO = matrixQ0

            fitLOO = _fitOuter(
                matrixDataLocal=matrixData_LOO,
                matrixMuncLocal=matrixMunc_LOO,
                matrixFLocal=matrixF_LOO,
                matrixQ0Local=matrixQ0_LOO,
                emMaxItersLocal=int(jackknifeEM_maxIters),
                emInnerRtolLocal=float(jackknifeEM_innerRtol),
            )
            x0_LOO = np.asarray(fitLOO["stateSmoothed"], dtype=np.float32)[:, 0].astype(
                np.float64, copy=False
            )

            kk = float(repIdx + 1)
            deltaVec = x0_LOO - meanLOO_x0
            meanLOO_x0 += deltaVec / kk
            deltaVec2 = x0_LOO - meanLOO_x0
            M2LOO_x0 += deltaVec * deltaVec2

        jackknifeVar0 = ((trackCount - 1.0) / float(trackCount)) * M2LOO_x0
        jackknifeVar0 = jackknifeVar0.astype(np.float32, copy=False)
        outTrack4 = np.sqrt(jackknifeVar0, dtype=np.float32)

    if effectiveInfoRescale:
        dataAdjustedFit = np.ascontiguousarray(
            matrixDataFit
            - np.asarray(fitFinal["background"], dtype=np.float32)[None, :],
            dtype=np.float32,
        )
        effectiveInfoDiag = _computeInnovationEffectiveInfoDiagnostics(
            matrixData=dataAdjustedFit,
            matrixMunc=matrixMuncFit,
            stateForward=stateForward,
            stateCovarForward=stateCovarForward,
            pNoiseForward=pNoiseForward,
            matrixF=matrixF,
            replicateBias=replicateBias_final,
            lambdaExp=lambdaExp_final,
            pad=float(pad),
            effectiveInfoBandwidthIntervals=effectiveInfoBandwidthIntervals,
            effectiveInfoBlockLengthIntervals=effectiveInfoBlockLengthIntervals,
        )
        correctionFactors = np.asarray(
            effectiveInfoDiag.get(
                "intervalFactors",
                np.full(intervalCount, effectiveInfoDiag["correctionFactor"]),
            ),
            dtype=np.float32,
        )
        if correctionFactors.size != intervalCount:
            correctionFactors = np.full(
                intervalCount,
                float(effectiveInfoDiag["correctionFactor"]),
                dtype=np.float32,
            )
        correctionFactors = np.maximum(correctionFactors, np.float32(1.0))
        outStateCovarSmoothed[:, 0, 0] = (
            outStateCovarSmoothed[:, 0, 0].astype(np.float32, copy=False)
            * correctionFactors
        )
        logger.info(
            "Effective-information uncertainty correction applied: "
            "bandwidth=%d blockLengthIntervals=%d blocks=%d "
            "gamma0=%.6g lrv=%.6g chrFactor=%.6g factorMedian=%.6g "
            "factorRange=[%.6g, %.6g] shrinkageWeightMean=%.6g",
            int(effectiveInfoDiag["bandwidth"]),
            int(effectiveInfoDiag.get("blockLengthIntervals", intervalCount)),
            int(effectiveInfoDiag.get("numBlocks", 1)),
            float(effectiveInfoDiag["gamma0"]),
            float(effectiveInfoDiag["lrv"]),
            float(effectiveInfoDiag.get("chromosomeFactor", 1.0)),
            float(effectiveInfoDiag["correctionFactor"]),
            float(
                effectiveInfoDiag.get(
                    "blockFactorMin",
                    effectiveInfoDiag["correctionFactor"],
                )
            ),
            float(
                effectiveInfoDiag.get(
                    "blockFactorMax",
                    effectiveInfoDiag["correctionFactor"],
                )
            ),
            float(effectiveInfoDiag.get("shrinkageWeightMean", 0.0)),
        )
    else:
        logger.info(
            "Effective-information uncertainty correction bypassed: effectiveInfoRescale=False "
            "(effectiveInfoBandwidthIntervals=%s, effectiveInfoBlockLengthIntervals=%s)",
            (
                "auto"
                if effectiveInfoBandwidthIntervals is None
                else int(effectiveInfoBandwidthIntervals)
            ),
            (
                "chromosome"
                if effectiveInfoBlockLengthIntervals is None
                else int(effectiveInfoBlockLengthIntervals)
            ),
        )

    if boundState:
        np.clip(
            outStateSmoothed[:, 0],
            np.float32(stateLowerBound),
            np.float32(stateUpperBound),
            out=outStateSmoothed[:, 0],
        )

    if returnScales:
        if returnReplicateOffsets:
            return (
                outStateSmoothed,
                outStateCovarSmoothed,
                outPostFitResiduals,
                outTrack4,
                np.asarray(qScale, dtype=np.float32),
                np.asarray(replicateBias_final, dtype=np.float32),
                intervalToBlockMap,
            )
        return (
            outStateSmoothed,
            outStateCovarSmoothed,
            outPostFitResiduals,
            outTrack4,
            np.asarray(qScale, dtype=np.float32),
            intervalToBlockMap,
        )

    return (
        outStateSmoothed,
        outStateCovarSmoothed,
        outPostFitResiduals,
        outTrack4,
    )


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


def fitVarianceFunction(
    jointlySortedMeans: np.ndarray,
    jointlySortedVariances: np.ndarray,
    eps: float = 1.0e-2,
    binQuantileCutoff: float = 0.5,
    EB_minLin: float = 0.0,
) -> np.ndarray:
    means = np.asarray(jointlySortedMeans, dtype=np.float64).ravel()
    variances = np.asarray(jointlySortedVariances, dtype=np.float64).ravel()
    absMeans = np.abs(means)
    n = absMeans.size

    sortIdx = np.argsort(absMeans)
    absMeans = absMeans[sortIdx]
    variances = variances[sortIdx]
    variances = np.maximum(variances, EB_minLin * absMeans) + eps

    # --- determine bins for isotonic regression ---
    binCount = int(1 + np.log2(n + 1, dtype=np.float64))
    binCount = max(4, binCount)
    binEdges = np.linspace(0, n, binCount + 1, dtype=np.int64)
    binEdges = np.unique(binEdges)
    if binEdges.size < 2:
        binEdges = np.array([0, n], dtype=np.int64)

    binnedAbsMeans = []
    binnedVariances = []
    binWeights = []
    for k in range(binEdges.size - 1):
        i = int(binEdges[k])
        j = int(binEdges[k + 1])
        if j <= i:
            continue
        # - mean of abs means defines x-axis for isotonic regression
        # - quantile of variances defines y-axis
        # - bin weight is number of points in bin
        binnedAbsMeans.append(np.median(absMeans[i:j]))
        binnedVariances.append(np.quantile(variances[i:j], binQuantileCutoff))
        binWeights.append(float(j - i))

    try:
        counts = [int(w) for w in binWeights]
        if counts:
            msg = (
                f"{len(counts)} bins; n/bin: "
                f"min binSize={min(counts)}, median binSize={int(np.median(counts))}, max binSize={max(counts)}"
            )
        else:
            msg = "0 bins; n/bin: []"
        logger.info(f"Bins: {msg}")
    except Exception:
        pass

    absMeans = np.asarray(binnedAbsMeans, dtype=np.float64)
    variances = np.asarray(binnedVariances, dtype=np.float64)
    weights = np.asarray(binWeights, dtype=np.float64)

    # one bin --> skip PAVA
    if absMeans.size < 2:
        logger.warning(
            "Skipping PAVA (isotonic regression) since only one bin was determined..."
        )
        m0 = (
            float(absMeans[0])
            if absMeans.size == 1
            else float(np.median(np.abs(means)))
        )
        v0 = (
            float(variances[0])
            if variances.size == 1
            else float(
                np.quantile(
                    np.maximum(
                        np.asarray(jointlySortedVariances, float),
                        EB_minLin * np.abs(np.asarray(jointlySortedMeans, float)),
                    )
                    + eps,
                    binQuantileCutoff,
                )
            )
        )
        v0 = max(v0, EB_minLin * m0)
        return np.vstack([np.array([m0], np.float32), np.array([v0], np.float32)])

    # isotonic regression via PAVA
    varsFit = cconsenrich.cPAVA(variances, weights)
    breaks = np.empty(varsFit.size, dtype=bool)
    breaks[0] = True
    breaks[1:] = varsFit[1:] != varsFit[:-1]

    coefAMu = absMeans[breaks]
    coefVar = varsFit[breaks]

    # lower envelope
    coefVar = np.maximum(coefVar, EB_minLin * coefAMu)
    return np.vstack([coefAMu.astype(np.float32), coefVar.astype(np.float32)])


def evalVarianceFunction(
    coeffs: np.ndarray,
    meanTrack: np.ndarray,
    eps: float = 1.0e-2,
    EB_minLin: float = 0.0,
) -> np.ndarray:
    absMeans = np.abs(np.asarray(meanTrack, dtype=np.float64).ravel())
    if coeffs is None or np.asarray(coeffs).size == 0:
        return np.full(absMeans.shape, np.nan, dtype=np.float32)

    coefAMu = np.asarray(coeffs[0], dtype=np.float64).ravel()
    coefVar = np.asarray(coeffs[1], dtype=np.float64).ravel()
    if coefAMu.size == 0:
        return np.full(absMeans.shape, np.nan, dtype=np.float32)

    # keep in range used to fit
    x = np.clip(absMeans, coefAMu[0], coefAMu[-1])
    varsEval = np.interp(x, coefAMu, coefVar)
    return varsEval.astype(np.float32)


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


def _backgroundPenaltyFromSpan(
    blockLenIntervals: int,
    backgroundSmoothness: float = 1.0,
) -> float:
    spanIntervals = max(2.0, float(blockLenIntervals))
    penalty = (spanIntervals * spanIntervals * spanIntervals * spanIntervals) / 16.0
    return float(max(1.0, float(backgroundSmoothness) * penalty))


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
        return np.empty(0, dtype=np.float64)

    sparseIdx = np.asarray(sparseIntervalIndices, dtype=np.intp).ravel()
    sparseIdx = sparseIdx[(sparseIdx >= 0) & (sparseIdx < intervalCount)]
    if sparseIdx.size == 0:
        return np.zeros(intervalCount, dtype=np.float64)
    sparseIdx = np.unique(sparseIdx)

    ellIntervals = float(ellIntervals)
    if (not np.isfinite(ellIntervals)) or ellIntervals <= 0.0:
        weights = np.zeros(intervalCount, dtype=np.float64)
        weights[sparseIdx] = 1.0
        return weights

    counts = np.zeros(intervalCount, dtype=np.float64)
    counts[sparseIdx] = 1.0

    decay = float(np.exp(-1.0 / ellIntervals))
    left = np.empty(intervalCount, dtype=np.float64)
    running = 0.0
    for i in range(intervalCount):
        running = (running * decay) + counts[i]
        left[i] = running

    right = np.empty(intervalCount, dtype=np.float64)
    running = 0.0
    for i in range(intervalCount - 1, -1, -1):
        running = (running * decay) + counts[i]
        right[i] = running

    nEff = left + right - counts
    supportPrior = float(supportPrior)
    if (not np.isfinite(supportPrior)) or supportPrior <= 0.0:
        weights = np.zeros(intervalCount, dtype=np.float64)
        weights[nEff > 0.0] = 1.0
        return weights

    weights = nEff / (nEff + supportPrior)
    weights[~np.isfinite(weights)] = 0.0
    return weights


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

    lam = _backgroundPenaltyFromSpan(
        blockLenIntervals=blockLenIntervals,
        backgroundSmoothness=backgroundSmoothness,
    )
    return cconsenrich.csolveZeroCenteredBackground(
        np.ascontiguousarray(weightTrack, dtype=np.float64),
        np.ascontiguousarray(rhsTrack, dtype=np.float64),
        float(lam),
        bool(zeroCenter),
    )


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
    binQuantileCutoff: float = 0.5,
    EB_minLin: float = 0.0,
    EB_use: bool = True,
    EB_setNu0: int | None = None,
    EB_setNuL: int | None = None,
    sparseIntervalIndices: Optional[np.ndarray] = None,
    sparseRegionMask: Optional[np.ndarray] = None,
    numNearest: int = 0,
    sparseSupportScaleBP: Optional[float] = None,
    sparseSupportPrior: float = 1.0,
    restrictLocalAR1ToSparseBed: bool = False,
    EB_localQuantile: float = 0.0,
    verbose: bool = False,
    eps: float = 1.0e-2,
) -> tuple[npt.NDArray[np.float32], float]:
    r"""Approximate initial sample-specific (**M**)easurement (**unc**)ertainty tracks

    For an individual experimental sample (replicate), quantify *positional* observation noise levels over genomic intervals :math:`i=1,2,\ldots n` spanning ``chromosome``.
    These tracks (per-sample) comprise the ``matrixMunc`` input to :func:`runConsenrich`, :math:`\mathbf{R}[:,:] \in \mathbb{R}^{m \times n}`.

    Variance is modeled as a function of the absolute mean signal level. For ``EB_use=True``, local variance estimates are shrunk toward a signal level dependent global variance fit.

    :param chromosome: chromosome/contig name
    :type chromosome: str
    :param values: normalized/transformed signal measurements over genomic intervals (e.g., :func:`consenrich.cconsenrich.cTransform` output)
    :type values: np.ndarray
    :param intervals: genomic intervals positions (start positions)
    :type intervals: np.ndarray

    See :class:`consenrich.core.observationParams` for other parameters.

    """

    AR1_PARAMCT = 3  # intercept, AR(1) coefficient, innovation variance
    if samplingBlockSizeBP is None:
        samplingBlockSizeBP = intervalSizeBP * (11 * (AR1_PARAMCT))
    blockSizeIntervals = int(samplingBlockSizeBP / intervalSizeBP)

    localWindowIntervals = max(4, (blockSizeIntervals + 1))
    intervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32)
    valuesArr = np.ascontiguousarray(values, dtype=np.float32)

    if excludeMask is None:
        excludeMaskArr = np.zeros_like(intervalsArr, dtype=np.uint8)
    else:
        excludeMaskArr = np.ascontiguousarray(excludeMask, dtype=np.uint8)

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

    def _estimateSparseNearestObsTracks() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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
            validMask = (
                excludeCum[blockStarts + blockLen] - excludeCum[blockStarts]
            ) == 0
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
            useInnovationVar=True,
            aggregateMeanAbs=False,
        )
        sparseMeanTrack = np.asarray(sparseMeanTrack, dtype=np.float64)
        sparseVarTrack = np.asarray(sparseVarTrack, dtype=np.float64)
        sparseMeanTrack[~np.isfinite(sparseMeanTrack)] = 0.0
        sparseVarTrack[~np.isfinite(sparseVarTrack)] = np.nan
        return sparseMeanTrack, sparseVarTrack, sparseIdx

    sparseInterceptTrack: np.ndarray | None = None
    sparseObsVarTrack: np.ndarray | None = None
    sparseSupportWeightTrack: np.ndarray | None = None
    valuesForPriorFitArr = valuesArr.astype(np.float64, copy=False)
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
        valuesForPriorFitArr = (
            valuesArr.astype(np.float64, copy=False) - sparseInterceptTrack
        )
        if verbose:
            logger.info(
                "Sparse-nearest support: ell=%.2f intervals, median weight=%.4f, max weight=%.4f",
                ellIntervals,
                float(np.median(sparseSupportWeightTrack)),
                float(np.max(sparseSupportWeightTrack)),
            )

    # Global:
    # ... Variance as function of |mean|, globally, as observed in distinct, randomly drawn genomic
    # ... blocks. Within fixed-size blocks, it's assumed that an AR(1) process can, on average,
    # ... account for a large fraction of desired signal, and the (residual) innovation variance
    # ... reflects noise
    blockMeans, blockVars, starts, ends = cconsenrich.cmeanVarPairs(
        intervalsArr,
        np.ascontiguousarray(valuesForPriorFitArr, dtype=np.float32),
        blockSizeIntervals,
        samplingIters,
        randomSeed,
        excludeMaskArr,
        useInnovationVar=True,
    )

    meanAbs = np.abs(blockMeans)
    mask = np.isfinite(meanAbs) & np.isfinite(blockVars) & (blockVars >= 1.0e-3)

    meanAbs_Masked = meanAbs[mask]
    var_Masked = blockVars[mask]
    order = np.argsort(meanAbs_Masked)
    meanAbs_Sorted = meanAbs_Masked[order]
    var_Sorted = var_Masked[order]
    opt = fitVarianceFunction(
        meanAbs_Sorted,
        var_Sorted,
        binQuantileCutoff=binQuantileCutoff,
        EB_minLin=EB_minLin,
        eps=eps,
    )

    meanTrack = np.ascontiguousarray(valuesForPriorFitArr, dtype=np.float32)
    if useEMA:
        meanTrack = cconsenrich.cEMA(meanTrack, 2 / (localWindowIntervals + 1))
    meanTrack = np.abs(meanTrack)
    priorTrack = evalVarianceFunction(opt, meanTrack, EB_minLin=EB_minLin).astype(
        np.float32, copy=False
    )

    if not EB_use:
        return priorTrack.astype(np.float32), np.sum(mask) / float(len(blockMeans))

    # Local:
    # ... default: rolling AR(1) innovation variance over a sliding window
    # ... optional sparse-bed restriction: invalidate any local window leaving sparse regions
    # ... sparse-nearest mode: aggregate region mean/variance stats at the nearest sparse blocks
    fallbackObsVarTrack = cconsenrich.crolling_AR1_IVar(
        valuesArr,
        localWindowIntervals,
        localObsExcludeMaskArr,
    ).astype(np.float64, copy=False)
    fallbackObsVarTrack[fallbackObsVarTrack < 0.0] = np.nan

    if sparseObsVarTrack is not None:
        sparseObsVarTrack = sparseObsVarTrack.astype(np.float64, copy=False)
        sparseObsVarTrack[sparseObsVarTrack < 0.0] = np.nan
        if sparseSupportWeightTrack is None:
            sparseSupportWeightTrack = np.ones_like(sparseObsVarTrack, dtype=np.float64)
        supportWeight = np.asarray(sparseSupportWeightTrack, dtype=np.float64)
        supportWeight = np.clip(supportWeight, 0.0, 1.0)

        obsVarTrack = np.array(fallbackObsVarTrack, dtype=np.float64, copy=True)
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

    # ~Corresponds~ to `binQuantileCutoff` that is applied in the global/prior fit:
    # ... Optionally, run a quantile filter over the local variance track
    # ...     EB_localQuantile < 0 --> disable
    # ...     EB_localQuantile == 0 --> use binQuantileCutoff
    # ...     EB_localQuantile > 0 --> use supplied quantile value (x100)
    # ... NOTE: Useful heuristic for parity with the global model and tempering effects of
    # ...    spurious measurements in sparse genomic regions where estimated noise levels
    # ...    are often artificially deflated. Note that the quantile filter _centered_,
    # ...    unlike innovations
    if EB_localQuantile >= 0.0:
        quantile_ = (
            float(binQuantileCutoff)
            if EB_localQuantile == 0.0
            else float(EB_localQuantile)
        )
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
            obsVarTrack = tmp + eps
        else:
            ndimage.percentile_filter(
                obsVarTrack + eps,
                size=win + 2,
                percentile=pct,
                mode="nearest",
                output=obsVarTrack,
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

    priorTrackF64 = priorTrack.astype(np.float64, copy=False)

    if EB_setNu0 is not None and EB_setNu0 > 4:
        # check if Nu_0 is specified before computing
        Nu_0 = float(EB_setNu0)
        logger.info(f"Using fixed/specified Nu_0={Nu_0:.2f}")
    else:
        # finite/non-zero mask _BEFORE_ Nu_0 fit
        priorFinite = priorTrackF64[np.isfinite(priorTrackF64)]
        obsFinite = obsVarTrack[np.isfinite(obsVarTrack)]
        medPrior = float(np.median(priorFinite)) if priorFinite.size else 0.0
        medObs = float(np.median(obsFinite)) if obsFinite.size else 0.0

        minScale_prior = (1.0e-2 * medPrior) + 1.0e-4
        minScale_obs = (1.0e-2 * medObs) + 1.0e-4

        finMask_obs = np.isfinite(obsVarTrack) & (obsVarTrack > minScale_obs)
        finMask_prior = np.isfinite(priorTrackF64) & (priorTrackF64 > minScale_prior)
        finMask_both = finMask_obs & finMask_prior

        # only pass matched finite pairs into EB_computePriorStrength
        if np.count_nonzero(finMask_both) < 4:
            logger.warning(
                f"Insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
            )
            Nu_0 = float(1.0e6)
        else:
            Nu_0 = EB_computePriorStrength(
                obsVarTrack[finMask_both],
                priorTrackF64[finMask_both],
                Nu_L,
            )

        # reuse masks during shrinkage (no need to recompute)
        finMask_obs2 = finMask_obs
        finMask_prior2 = finMask_prior
        finMask_both2 = finMask_both

    logger.info(f"Nu_0={Nu_0:.2f}, Nu_L={Nu_L:.2f}")
    posteriorSampleSize: float = Nu_L + Nu_0

    # --- Shrinkage ---
    posteriorVarTrack = np.array(priorTrackF64, dtype=np.float64, copy=True)

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
        Nu_L * obsVarTrack[finMask_both2] + Nu_0 * posteriorVarTrack[finMask_both2]
    ) / posteriorSampleSize

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

    if verbose:
        logger.info(
            f"Median variance after shrinkage: {float(np.nanmedian(posteriorVarTrack)):.4f}",
        )

    return posteriorVarTrack.astype(np.float32, copy=False), np.sum(mask) / float(
        len(blockMeans)
    )


def EB_computePriorStrength(
    localModelVariances: np.ndarray, globalModelVariances: np.ndarray, Nu_local: float
) -> float:
    r"""Compute :math:`\nu_0` to determine 'prior strength'

    The prior model strength is determined by 'excess' dispersion beyond sampling noise at the local level.

    :param localModelVariances: Local model variance estimates (e.g., rolling AR(1) innovation variances :func:`consenrich.cconsenrich.crolling_AR1_IVar`).
    :type localModelVariances: np.ndarray
    :param globalModelVariances: Global model variance estimates from the absMean-variance trend fit (:func:`consenrich.core.fitVarianceFunction`).
    :type globalModelVariances: np.ndarray
    :param Nu_local: Effective sample size/degrees of freedom for the local model.
    :type Nu_local: float
    :return: Estimated prior strength :math:`\nu_{0}`.
    :rtype: float

    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitVarianceFunction`
    """

    localModelVariancesArr = np.asarray(localModelVariances, dtype=np.float64)
    globalModelVariancesArr = np.asarray(globalModelVariances, dtype=np.float64)

    ratioMask = (localModelVariancesArr > 0.0) & (globalModelVariancesArr > 0.0)
    if np.count_nonzero(ratioMask) < (0.10) * localModelVariancesArr.size:
        logger.warning(
            f"Insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
        )
        return float(1.0e6)

    varRatioArr = localModelVariancesArr[ratioMask] / globalModelVariancesArr[ratioMask]
    varRatioArr = varRatioArr[np.isfinite(varRatioArr) & (varRatioArr > 0.0)]
    if varRatioArr.size < (0.10) * localModelVariancesArr.size:
        logger.warning(
            f"After masking, insufficient prior/local variance pairs...setting Nu_0 = 1.0e6",
        )
        return float(1.0e6)

    logVarRatioArr = np.log(varRatioArr)
    clipSmall = np.quantile(logVarRatioArr, 0.001)
    clipBig = np.quantile(logVarRatioArr, 0.999)
    np.clip(logVarRatioArr, clipSmall, clipBig, out=logVarRatioArr)

    varLogVarRatio = float(np.var(logVarRatioArr, ddof=1))
    trigammaLocal = float(special.polygamma(1, float(Nu_local) / 2.0))
    # inverse trigamma --> inf near 0
    gap = max(varLogVarRatio - trigammaLocal, 1.0e-6)
    Nu_0 = 2.0 * itrigamma(gap)
    if Nu_0 < 4.0:
        Nu_0 = 4.0

    return float(Nu_0)


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


def getContextSize(
    vals: np.ndarray,
    minSpan: int | None = 3,
    maxSpan: int | None = 64,
    bandZ: float = 1.0,
    maxOrder: int = 5,
) -> tuple[int, int, int]:
    r"""(Experimental) Heuristic estimator for characteristic feature width from local peak widths

    Candidate features are detected on a smoothed log-scale track, half-height
    widths are measured locally, and width uncertainty is estimated by a local
    residual bootstrap before EB shrinkage on the log-width scale
    """
    y = np.asarray(vals, dtype=np.float64)
    n = int(y.size)
    if n < 100:
        raise ValueError(
            "input `vals` is too small for context size estimation...set `countingParams.backgroundBlockSizeBP` manually."
        )

    minSpan = 3 if minSpan is None else int(minSpan)
    maxSpan = (
        int(max(10, min(50, np.floor(np.log2(n + 1) * 2))))
        if maxSpan is None
        else int(maxSpan)
    )
    if maxSpan <= 0:
        raise ValueError("`maxSpan` must be positive.")

    yPos = np.clip(y, 0.0, None)
    yLog = np.log1p(yPos)
    posLog = yLog[y > 0]
    if posLog.size <= max(1, int(maxOrder)):
        raise ValueError(
            "Insufficient positive elements found...set `countingParams.backgroundBlockSizeBP` manually."
        )

    smoothSize = min(int(max(1, minSpan)), int(maxSpan / 2))
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
        raise ValueError(
            "Could not identify features for context size estimation...using largest values as features... "
            "Consider setting `countingParams.backgroundBlockSizeBP` manually..."
        )

    chosenFeatures = np.unique(chosenFeatures.astype(np.int64))
    chosenFeatures.sort()

    # We compute a 'baseline' around each feature -- these are used to
    # compute per-feature 'feature scores' (height above baseline).
    baseQ = 0.05
    featureBaselines = np.empty(chosenFeatures.size, dtype=np.float64)
    for i, idx in enumerate(chosenFeatures):
        # FFR: this might be redundant in practice...perhaps just take quantile over full window?
        left = max(0, idx - maxSpan)
        right = min(n - 1, idx + maxSpan)

        leftQ = float(np.quantile(yLog[left : idx + 1], baseQ))
        rightQ = float(np.quantile(yLog[idx : right + 1], baseQ))
        featureBaselines[i] = max(leftQ, rightQ)

    featureScores = yLog[chosenFeatures] - featureBaselines
    keepMask = featureScores > 0.0
    if np.any(keepMask):
        chosenFeatures = chosenFeatures[keepMask]
        featureScores = featureScores[keepMask]

    kKeep = int(min(1000, featureScores.size, max(kMinFeatures, n // max(8, maxSpan))))
    if kKeep <= 0:
        raise ValueError(
            "No features found for context size estimation...supply `countingParams.backgroundBlockSizeBP` manually"
        )

    # we use the top-scoring kKeep features for estimation
    keep = np.argpartition(-featureScores, kKeep - 1)[:kKeep]
    featureIndexArray = np.unique(chosenFeatures[keep].astype(np.int64))
    featureIndexArray.sort()

    # for each feature, estimate width on log scale and use a local bootstrap
    # to quantify width uncertainty
    noiseWindow = int(min(maxSpan, 32))
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
        raise ValueError(
            "Failed to estimate context size from feature widths due to insufficient valid features...set `countingParams.backgroundBlockSizeBP` manually.",
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

    if maxSpan > 0:
        pointEstimate = min(pointEstimate, float(maxSpan))
        widthLower = min(widthLower, float(maxSpan))
        widthUpper = min(widthUpper, float(maxSpan))

    return int(pointEstimate), int(widthLower), int(widthUpper)
