# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import (
    Any,
    Callable,
    DefaultDict,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

from importlib.util import find_spec
import numpy as np
import numpy.typing as npt
import pybedtools as bed
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage, signal
from scipy.stats.mstats import trimtail
from tqdm import tqdm
from . import cconsenrich

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class plotParams(NamedTuple):
    r"""(Experimental) Parameters related to plotting filter results and diagnostics.

    :param plotPrefix: Prefix for output plot filenames.
    :type plotPrefix: str or None
    :param plotStateEstimatesHistogram: If True, plot a histogram of post-fit primary state estimates
    :type plotStateEstimatesHistogram: bool
    :param plotResidualsHistogram: If True, plot a histogram of post-fit residuals
    :type plotResidualsHistogram: bool
    :param plotStateStdHistogram: If True, plot a histogram of the posterior state standard deviations.
    :type plotStateStdHistogram: bool
    :param plotHeightInches: Height of output plots in inches.
    :type plotHeightInches: float
    :param plotWidthInches: Width of output plots in inches.
    :type plotWidthInches: float
    :param plotDPI: DPI of output plots (png)
    :type plotDPI: int
    :param plotDirectory: Directory where plots will be written.
    :type plotDirectory: str or None

    :seealso: :func:`plotStateEstimatesHistogram`, :func:`plotResidualsHistogram`, :func:`plotStateStdHistogram`
    """

    plotPrefix: str | None = None
    plotStateEstimatesHistogram: bool = False
    plotResidualsHistogram: bool = False
    plotStateStdHistogram: bool = False
    plotHeightInches: float = 6.0
    plotWidthInches: float = 8.0
    plotDPI: int = 300
    plotDirectory: str | None = None


class processParams(NamedTuple):
    r"""Parameters related to the process model of Consenrich.

    The process model governs the signal and variance propagation
    through the state transition :math:`\mathbf{F} \in \mathbb{R}^{2 \times 2}`
    and process noise covariance :math:`\mathbf{Q}_{[i]} \in \mathbb{R}^{2 \times 2}`
    matrices.

    :param deltaF: Scales the signal and variance propagation between adjacent genomic intervals. If ``< 0`` (default), determined based on stepSize:fragmentLength ratio.
    :type deltaF: float
    :param minQ: Minimum process noise level (diagonal in :math:`\mathbf{Q}_{[i]}`)
        for each state variable. If `minQ < 0` (default), a value based on
        the minimum observation noise level (``observationParams.minR``) is used that
        enforces numerical stability and a worst-case balance between process and observation models
        for the given number of samples.
    :param maxQ: Maximum process noise level.
    :type minQ: float
    :param dStatAlpha: Threshold on the (normalized) deviation between the data and estimated signal -- determines whether the process noise is scaled up.
    :type dStatAlpha: float
    :param dStatd: Constant :math:`d` in the scaling expression :math:`\sqrt{d|D_{[i]} - \alpha_D| + c}`
        that is used to up/down-scale the process noise covariance in the event of a model mismatch.
    :type dStatd: float
    :param dStatPC: Constant :math:`c` in the scaling expression :math:`\sqrt{d|D_{[i]} - \alpha_D| + c}`
        that is used to up/down-scale the process noise covariance in the event of a model mismatch.
    :type dStatPC: float
    :param dStatUseMean: If `True`, the mean of squared, diagonal-standardized residuals (rather than the median) is used to compute the :math:`D_{[i]}` statistic at each interval :math:`i`.
    :param scaleResidualsByP11: If `True`, the primary state variances (posterior) :math:`\widetilde{P}_{[i], (11)}, i=1\ldots n` are included in the inverse-variance (precision) weighting of residuals :math:`\widetilde{\mathbf{y}}_{[i]}, i=1\ldots n`.
        If `False`, only the per-sample *observation noise levels* will be used in the precision-weighting. Note that this does not affect `raw` residuals output (i.e., ``postFitResiduals`` from :func:`consenrich.consenrich.runConsenrich`).
    :type scaleResidualsByP11: Optional[bool]
    """

    deltaF: float
    minQ: float
    maxQ: float
    offDiagQ: float
    dStatAlpha: float
    dStatd: float
    dStatPC: float
    dStatUseMean: bool
    scaleResidualsByP11: Optional[bool]


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich.

    The observation model is used to integrate sequence alignment count
    data from the multiple input samples and account for region-and-sample-specific
    noise processes corrupting data. The observation model matrix
    :math:`\mathbf{H} \in \mathbb{R}^{m \times 2}` maps from the state dimension (2)
    to the dimension of measurements/data (:math:`m`).

    :param minR: Genome-wide lower bound for sample-specific measurement uncertainty levels.
    :type minR: float
    :param maxR: Genome-wide upper bound for the sample-specific measurement uncertainty levels.
    :param numNearest: Optional. The number of nearest 'sparse' features in ``consenrich.core.genomeParams.sparseBedFile``
      to use at each interval during the ALV/local measurement uncertainty calculation. See :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.getAverageLocalVarianceTrack`.
    :type numNearest: int
    :param localWeight: Weight for the 'local' model used to approximate genome-wide sample/region-level measurement uncertainty (see `consenrich.core.getAverageLocalVarianceTrack`, `consenrich.core.getMuncTrack`).
    :type localWeight: float
    :param approximationWindowLengthBP: The size of the moving window used in the local model (:func:`consenrich.core.getMuncTrack`).
    :type approximationWindowLengthBP: int
    :param sparseBedFile: The path to a BED file of 'sparse' regions. For genomes with default resources in `src/consenrich/data`, this may be left as `None`,
      and a default annotation that is devoid/exclusive of/with putative regulatory elements (ENCODE cCREs) will be used. Users can instead supply a custom BED file annotation
      or rely exclusively on the ALV heuristic for the *local* component.
    :type sparseBedFile: str, optional
    :param lowPassFilterType: The type of low-pass filter to use (e.g., 'median', 'mean') in the ALV calculation (:func:`consenrich.core.getAverageLocalVarianceTrack`).
    :type lowPassFilterType: Optional[str]
    :param shrinkOffset: (*Experimental*) An offset applied to local lag-1 autocorrelation, :math:`A_{[i,1]}`, such that the structure-based shrinkage factor in :func:`consenrich.core.getAverageLocalVarianceTrack`, :math:`1 - A_{[i,1]}^2`, does not deplete the ALV variance estimates. Consider setting near `1.0` if data has been preprocessed to remove local trends.
        To disable, set to `>= 1`.
    :type shrinkOffset: Optional[float]
    :param zeroPenalty: Inflate variance estimates in genomic regions with a larger proportion of zeros.
    :type zeroPenalty: Optional[float]
    :param kappaALV: Applicable if ``minR < 0``. Prevent ill-conditioning by bounding the ratios :math:`\frac{R_{[i,j_{\max}]}}{R_{[i,j_{\min}]}}` at each interval :math:`i=1\ldots n`. Values up to `100` will typically retain most of the initial dynamic range while improving stability and mitigating outliers.
    """

    minR: float
    maxR: float
    useConstantNoiseLevel: bool
    noGlobal: bool  # deprecated
    numNearest: int
    localWeight: float
    approximationWindowLengthBP: int
    lowPassWindowLengthBP: int
    lowPassFilterType: Optional[str]
    returnCenter: bool  # deprecated
    shrinkOffset: Optional[float]
    kappaALV: Optional[float]
    zeroPenalty: Optional[float]


class stateParams(NamedTuple):
    r"""Parameters related to state and uncertainty bounds and initialization.

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
    :param offsetStr: A string of two comma-separated integers -- first for the 5' shift on forward strand, second for the 5' shift on reverse strand.
    :type offsetStr: str
    :param maxInsertSize: Maximum frag length/insert to consider when estimating fragment length.
    :type maxInsertSize: int
    :param pairedEndMode: If > 0, use TLEN attribute to determine span of (proper) read pairs and extend reads accordingly.
    :type pairedEndMode: int
    :param inferFragmentLength: Intended for single-end data: if > 0, the maximum correlation lag
       (avg.) between *strand-specific* read tracks is taken as the fragment length estimate and used to
       extend reads from 5'. Ignored if `pairedEndMode > 0`, `countEndsOnly`, or `fragmentLengths` is provided.
       important when targeting broader marks (e.g., ChIP-seq H3K27me3).
    :type inferFragmentLength: int
    :param countEndsOnly: If True, only the 5' read lengths contribute to counting. Overrides `inferFragmentLength` and `pairedEndMode`.
    :type countEndsOnly: Optional[bool]
    :param minMappingQuality: Minimum mapping quality (MAPQ) for reads to be counted.
    :type minMappingQuality: Optional[int]
    :param fragmentLengths:

    .. tip::

        For an overview of SAM flags, see https://broadinstitute.github.io/picard/explain-flags.html

    """

    samThreads: int
    samFlagExclude: int
    oneReadPerBin: int
    chunkSize: int
    offsetStr: Optional[str] = "0,0"
    maxInsertSize: Optional[int] = 1000
    pairedEndMode: Optional[int] = 0
    inferFragmentLength: Optional[int] = 0
    countEndsOnly: Optional[bool] = False
    minMappingQuality: Optional[int] = 0
    minTemplateLength: Optional[int] = -1
    fragmentLengths: Optional[List[int]] = None


class detrendParams(NamedTuple):
    r"""Parameters related detrending and background-removal after normalizing by sequencing depth.

    :param useOrderStatFilter: Whether to use a local/moving order statistic (percentile filter) to model and remove trends in the read density data.
    :type useOrderStatFilter: bool
    :param usePolyFilter: Whether to use a low-degree polynomial fit to model and remove trends in the read density data.
    :type usePolyFilter: bool
    :param detrendSavitzkyGolayDegree: The polynomial degree of the Savitzky-Golay filter to use for detrending
    :type detrendSavitzkyGolayDegree: int
    :param detrendTrackPercentile: The percentile to use for the local/moving order-statistic filter.
      Decrease for broad marks + sparse data if `useOrderStatFilter` is True.
    :type detrendTrackPercentile: float
    :param detrendWindowLengthBP: The length of the window in base pairs for detrending.
      Increase for broader marks + sparse data.
    :type detrendWindowLengthBP: int
    """

    useOrderStatFilter: bool
    usePolyFilter: bool
    detrendTrackPercentile: float
    detrendSavitzkyGolayDegree: int
    detrendWindowLengthBP: int


class inputParams(NamedTuple):
    r"""Parameters related to the input data for Consenrich.

    :param bamFiles: A list of paths to distinct coordinate-sorted and indexed BAM files.
    :type bamFiles: List[str]

    :param bamFilesControl: A list of paths to distinct coordinate-sorted and
        indexed control BAM files. e.g., IgG control inputs for ChIP-seq.
    :type bamFilesControl: List[str], optional
    :param pairedEnd: Deprecated: Paired-end/Single-end is inferred automatically from the alignment flags in input BAM files.
    :type pairedEnd: Optional[bool]
    """

    bamFiles: List[str]
    bamFilesControl: Optional[List[str]]
    pairedEnd: Optional[bool]


class genomeParams(NamedTuple):
    r"""Specify assembly-specific resources, parameters.

    :param genomeName: If supplied, default resources for the assembly (sizes file, blacklist, and 'sparse' regions) in `src/consenrich/data` are used.
      ``ce10, ce11, dm6, hg19, hg38, mm10, mm39`` have default resources available.
    :type genomeName: str
    :param chromSizesFile: A two-column TSV-like file with chromosome names and sizes (in base pairs).
    :type chromSizesFile: str
    :param blacklistFile: A BED file with regions to exclude.
    :type blacklistFile: str, optional
    :param sparseBedFile: A BED file with 'sparse regions' that are mutually exclusive with or devoid of the targeted signal. Used to estimate noise levels. See :func:`getMuncTrack`.
    :type sparseBedFile: str, optional
    :param chromosomes: A list of chromosome names to analyze. If None, all chromosomes in `chromSizesFile` are used.
    :type chromosomes: List[str]
    """

    genomeName: str
    chromSizesFile: str
    blacklistFile: Optional[str]
    sparseBedFile: Optional[str]
    chromosomes: List[str]
    excludeChroms: List[str]
    excludeForNorm: List[str]


class countingParams(NamedTuple):
    r"""Parameters related to counting reads in genomic intervals.

    :param stepSize: Size (bp) of genomic intervals (AKA bin size, interval length, width, etc.).
        ``consenrich.py`` defaults to 25 bp, but users may adjust this based on expected sequencing
        depth and expected feature sizes. Lower sequencing depth and/or broader features may warrant
        larger step sizes (e.g., 50-100bp or more).
    :type stepSize: int
    :param scaleDown: If using paired treatment and control BAM files, whether to
        scale down the larger of the two before computing the difference/ratio
    :type scaleDown: bool, optional
    :param scaleFactors: Scale factors for the read counts.
    :type scaleFactors: List[float], optional
    :param scaleFactorsControl: Scale factors for the control read counts.
    :type scaleFactorsControl: List[float], optional
    :param numReads: Number of reads to sample.
    :type numReads: int
    :param applyAsinh: If true, :math:`\textsf{arsinh}(x)` applied to counts :math:`x` for each supplied BAM file (log-like for large values and linear near the origin).
    :type applyAsinh: bool, optional
    :param applyLog: If true, :math:`\textsf{log}(x + 1)` applied to counts :math:`x` for each supplied BAM file.
    :type applyLog: bool, optional
    :param applySqrt: If true, :math:`\sqrt{x}` applied to counts :math:`x` for each supplied BAM file.
    :type applySqrt: bool, optional
    :param noTransform: Disable all transformations.
    :type noTransform: bool, optional
    :param rescaleToTreatmentCoverage: Deprecated: no effect.
    :type rescaleToTreatmentCoverage: bool, optional
    :param trimLeftTail: If > 0, quantile of scaled counts to trim from the left tail before computing transformations.
    :type trimLeftTail: float, optional
    :param fragmentLengths: List of fragment lengths (bp) to use for extending reads from 5' ends when counting single-end data.
    :type fragmentLengths: List[int], optional
    :param fragmentLengthsControl: List of fragment lengths (bp) to use for extending reads from 5' ends when counting single-end with control data.
    :type fragmentLengthsControl: List[int], optional
    :param useTreatmentFragmentLengths: If True, use fragment lengths estimated from treatment BAM files for control BAM files, too.
    :type useTreatmentFragmentLengths: bool, optional


    .. admonition:: Treatment vs. Control Fragment Lengths in Single-End Data
    :class: tip
    :collapsible: closed

        For single-end data, cross-correlation-based estimates for fragment length
        in control inputs can be biased due to a comparative lack of structure in
        strand-specific coverage tracks.

        This can create artifacts during counting, so it is common to use the estimated treatment
        fragment length for both treatment and control samples. The argument
        ``observationParams.useTreatmentFragmentLengths`` enables this behavior.

    :seealso: :ref:`calibration`, :class:`samParams`.
    """

    stepSize: int
    scaleDown: Optional[bool]
    scaleFactors: Optional[List[float]]
    scaleFactorsControl: Optional[List[float]]
    numReads: int
    applyAsinh: Optional[bool]
    applyLog: Optional[bool]
    applySqrt: Optional[bool]
    noTransform: Optional[bool]
    rescaleToTreatmentCoverage: Optional[bool]
    normMethod: Optional[str]
    trimLeftTail: Optional[float]
    fragmentLengths: Optional[List[int]]
    fragmentLengthsControl: Optional[List[int]]
    useTreatmentFragmentLengths: Optional[bool]


class matchingParams(NamedTuple):
    r"""Parameters related to the matching algorithm.

    See :ref:`matching` for an overview of the approach.

    :param templateNames: A list of str values -- each entry references a mother wavelet (or its corresponding scaling function). e.g., `[haar, db2]`
    :type templateNames: List[str]
    :param cascadeLevels: Number of cascade iterations, or 'levels', used to define wavelet-based templates
        Must have the same length as `templateNames`, with each entry aligned to the
        corresponding template. e.g., given templateNames `[haar, db2]`, then `[2,2]` would use 2 cascade levels for both templates.
    :type cascadeLevels: List[int]
    :param iters: Number of random blocks to sample in the response sequence while building
        an empirical null to test significance. See :func:`cconsenrich.csampleBlockStats`.
    :type iters: int
    :param alpha: Primary significance threshold on detected matches. Specifically, the
        minimum corrected empirical p-value approximated from randomly sampled blocks in the
        response sequence.
    :type alpha: float
    :param minMatchLengthBP: Within a window of `minMatchLengthBP` length (bp), relative maxima in
        the signal-template convolution :math:`\mathcal{R}_{[\ast]}` must be greater in value than
        others to qualify as matches. If set to a value less than 1, the minimum length is determined
        via :func:`consenrich.matching.autoMinLengthIntervals` (default behavior).
    :type minMatchLengthBP: Optional[int]
    :param minSignalAtMaxima: Secondary/optional threshold coupled with ``alpha``. Requires the *signal value*, :math:`\widetilde{x}_{[i^*]}`,
        at relative maxima in the response sequence, :math:`\mathcal{R}_{[i^*]}`, to be greater than this threshold.
        If a ``str`` value is provided, looks for 'q:quantileValue', e.g., 'q:0.90'. The threshold is then set to the
        corresponding quantile of the non-zero signal estimates in the distribution of transformed values.
    :type minSignalAtMaxima: Optional[str | float]
    :param useScalingFunction: If True, use (only) the scaling function to build the matching template.
        If False, use (only) the wavelet function.
    :type useScalingFunction: bool
    :param excludeRegionsBedFile: A BED file with regions to exclude from matching
    :type excludeRegionsBedFile: Optional[str]
    :param penalizeBy: Specify a positional metric to scale signal estimate values by when matching.
      For example, ``stateUncertainty`` divides signal values by the square root of the primary state
      variance :math:`\sqrt{\widetilde{P}_{i,(11)}}` at each position :math:`i`,
      thereby down-weighting positions where the posterior state uncertainty is
      high during matching.
    :type penalizeBy: Optional[str]
    :param eps: Tolerance parameter for relative maxima detection in the response sequence. Set to zero to enforce strict
        inequalities when identifying discrete relative maxima.
    :type eps: float
    :param autoLengthQuantile: If `minMatchLengthBP < 1`, the minimum match length (``minMatchLengthBP / stepSize``) is determined
        by the quantile in the distribution of non-zero segment lengths (i.e., consecutive intervals with non-zero signal estimates).
        after local standardization.
    :type autoLengthQuantile: float
    :param methodFDR: Method for genome-wide multiple hypothesis testing correction. Can specify either Benjamini-Hochberg ('BH'), the more conservative Benjamini-Yekutieli ('BY') to account for arbitrary dependencies between tests, or None.
    :type methodFDR: str
    :param massQuantileCutoff: Quantile cutoff for filtering initial (unmerged) matches based on their 'mass' ``((avgSignal*length)/intervalLength)``. To diable, set ``< 0``.
    :type massQuantileCutoff: float
    :seealso: :func:`cconsenrich.csampleBlockStats`, :ref:`matching`, :class:`outputParams`.
    """

    templateNames: List[str]
    cascadeLevels: List[int]
    iters: int
    alpha: float
    useScalingFunction: Optional[bool]
    minMatchLengthBP: Optional[int]
    maxNumMatches: Optional[int]
    minSignalAtMaxima: Optional[str | float]
    merge: Optional[bool]
    mergeGapBP: Optional[int]
    excludeRegionsBedFile: Optional[str]
    penalizeBy: Optional[str]
    randSeed: Optional[int]
    eps: Optional[float]
    autoLengthQuantile: Optional[float]
    methodFDR: Optional[str]
    massQuantileCutoff: Optional[float]


class outputParams(NamedTuple):
    r"""Parameters related to output files.

    :param convertToBigWig: If True, output bedGraph files are converted to bigWig format.
    :type convertToBigWig: bool
    :param roundDigits: Number of decimal places to round output values (bedGraph)
    :type roundDigits: int
    :param writeResiduals: If True, write to a separate bedGraph the pointwise avg. of precision-weighted residuals at each interval. These may be interpreted as
        a measure of model mismatch. Where these quantities are larger (+-), there may be more unexplained deviation between the data and fitted model.
    :type writeResiduals: bool
    :param writeMuncTrace: If True, write to a separate bedGraph :math:`\sqrt{\frac{\textsf{Trace}\left(\mathbf{R}_{[i]}\right)}{m}}` -- that is, square root of the 'average' measurement uncertainty level at each interval :math:`i=1\ldots n`, where :math:`m` is the number of samples/tracks.
    :type writeMuncTrace: bool
    :param writeStateStd: If True, write to a separate bedGraph the estimated pointwise uncertainty in the primary state, :math:`\sqrt{\widetilde{P}_{i,(11)}}`, on a scale comparable to the estimated signal.
    :type writeStateStd: bool
    """

    convertToBigWig: bool
    roundDigits: int
    writeResiduals: bool
    writeMuncTrace: bool
    writeStateStd: bool


def _checkMod(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except Exception:
        return False


def _numIntervals(start: int, end: int, step: int) -> int:
    # helper for consistency
    length = max(0, end - start)
    return (length + step) // step


def getChromRanges(
    bamFile: str,
    chromosome: str,
    chromLength: int,
    samThreads: int,
    samFlagExclude: int,
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

    :seealso: :func:`getChromRangesJoint`, :func:`cconsenrich.cgetFirstChromRead`, :func:`cconsenrich.cgetLastChromRead`
    """
    start: int = cconsenrich.cgetFirstChromRead(
        bamFile, chromosome, chromLength, samThreads, samFlagExclude
    )
    end: int = cconsenrich.cgetLastChromRead(
        bamFile, chromosome, chromLength, samThreads, samFlagExclude
    )
    return start, end


def getChromRangesJoint(
    bamFiles: List[str],
    chromosome: str,
    chromSize: int,
    samThreads: int,
    samFlagExclude: int,
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
    for bam_ in bamFiles:
        start, end = getChromRanges(
            bam_,
            chromosome,
            chromLength=chromSize,
            samThreads=samThreads,
            samFlagExclude=samFlagExclude,
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

    :seealso: :func:`cconsenrich.cgetReadLength`
    """
    init_rlen = cconsenrich.cgetReadLength(
        bamFile, numReads, samThreads, maxIterations, samFlagExclude
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
) -> List[int]:
    r"""Get read lengths for a list of BAM files.

    :seealso: :func:`getReadLength`
    """
    return [
        getReadLength(
            bamFile,
            numReads=numReads,
            maxIterations=maxIterations,
            samThreads=samThreads,
            samFlagExclude=samFlagExclude,
        )
        for bamFile in bamFiles
    ]


def readBamSegments(
    bamFiles: List[str],
    chromosome: str,
    start: int,
    end: int,
    stepSize: int,
    readLengths: List[int],
    scaleFactors: List[float],
    oneReadPerBin: int,
    samThreads: int,
    samFlagExclude: int,
    offsetStr: Optional[str] = "0,0",
    applyAsinh: Optional[bool] = False,
    applyLog: Optional[bool] = False,
    applySqrt: Optional[bool] = False,
    maxInsertSize: Optional[int] = 1000,
    pairedEndMode: Optional[int] = 0,
    inferFragmentLength: Optional[int] = 0,
    countEndsOnly: Optional[bool] = False,
    minMappingQuality: Optional[int] = 0,
    minTemplateLength: Optional[int] = -1,
    trimLeftTail: Optional[float] = 0.0,
    fragmentLengths: Optional[List[int]] = None,
) -> npt.NDArray[np.float32]:
    r"""Calculate tracks of read counts (or a function thereof) for each BAM file.

    See :func:`cconsenrich.creadBamSegment` for the underlying implementation in Cython.
    Note that read counts are scaled by `scaleFactors` and possibly transformed if
    any of `applyAsinh`, `applyLog`, `applySqrt`. Note that these transformations are mutually
    exclusive and may affect interpretation of results.

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
    :param stepSize: See :class:`countingParams`.
    :type stepSize: int
    :param oneReadPerBin: See :class:`samParams`.
    :type oneReadPerBin: int
    :param samThreads: See :class:`samParams`.
    :type samThreads: int
    :param samFlagExclude: See :class:`samParams`.
    :type samFlagExclude: int
    :param offsetStr: See :class:`samParams`.
    :type offsetStr: str
    :param maxInsertSize: See :class:`samParams`.
    :type maxInsertSize: int
    :param pairedEndMode: See :class:`samParams`.
    :type pairedEndMode: int
    :param inferFragmentLength: See :class:`samParams`.
    :type inferFragmentLength: int
    :param minMappingQuality: See :class:`samParams`.
    :type minMappingQuality: int
    :param minTemplateLength: See :class:`samParams`.
    :type minTemplateLength: Optional[int]
    :param fragmentLengths: If supplied, a list of estimated fragment lengths for each BAM file.
        In single-end mode, these are values are used to extend reads. They are ignored in paired-end
        mode, where each proper pair `TLEN` is counted.
    :type fragmentLengths: Optional[List[int]]
    """

    segmentSize_ = end - start
    if stepSize <= 0 or segmentSize_ <= 0:
        raise ValueError(
            "Invalid stepSize or genomic segment specified (end <= start)"
        )

    if len(bamFiles) == 0:
        raise ValueError("bamFiles list is empty")

    if len(readLengths) != len(bamFiles) or len(scaleFactors) != len(
        bamFiles
    ):
        raise ValueError(
            "readLengths and scaleFactors must match bamFiles length"
        )

    offsetStr = ((str(offsetStr) or "0,0").replace(" ", "")).split(
        ","
    )

    numIntervals = ((end - start) + stepSize - 1) // stepSize
    counts = np.empty((len(bamFiles), numIntervals), dtype=np.float32)

    if pairedEndMode:
        fragmentLengths = [0] * len(bamFiles)
        inferFragmentLength = 0
    if not pairedEndMode and (
        fragmentLengths is None or len(fragmentLengths) == 0
    ):
        inferFragmentLength = 1
        fragmentLengths = [-1] * len(bamFiles)

    if isinstance(countEndsOnly, bool) and countEndsOnly:
        # note: setting this option ignores inferFragmentLength, pairedEndMode
        inferFragmentLength = 0
        pairedEndMode = 0
        fragmentLengths = [0] * len(bamFiles)

    for j, bam in enumerate(bamFiles):
        logger.info(f"Reading {chromosome}: {bam}")
        arr = cconsenrich.creadBamSegment(
            bam,
            chromosome,
            start,
            end,
            stepSize,
            readLengths[j],
            oneReadPerBin,
            samThreads,
            samFlagExclude,
            int(offsetStr[0]),
            int(offsetStr[1]),
            fragmentLengths[j],
            maxInsertSize,
            pairedEndMode,
            inferFragmentLength,
            minMappingQuality,
            minTemplateLength,
        )

        counts[j, :] = arr
        if trimLeftTail > 0.0:
            counts[j, :] = trimtail(
                counts[j, :], trimLeftTail, tail="left"
            )
        counts[j, :] *= np.float32(scaleFactors[j])
    return counts


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
    minDiagQ: float, offDiagQ: float = 0.0
) -> npt.NDArray[np.float32]:
    r"""Build the initial process noise covariance matrix :math:`\mathbf{Q}_{[1]}`.

    :param minDiagQ: See :class:`processParams`.
    :type minDiagQ: float
    :param offDiagQ: See :class:`processParams`.
    :type offDiagQ: float
    :return: The initial process noise covariance matrix :math:`\mathbf{Q}_{[1]}`.
    :rtype: npt.NDArray[np.float32]

    :seealso: :class:`processParams`
    """
    minDiagQ = np.float32(minDiagQ)
    offDiagQ = np.float32(offDiagQ)
    initMatrixQ: npt.NDArray[np.float32] = np.zeros(
        (2, 2), dtype=np.float32
    )
    initMatrixQ[0, 0] = minDiagQ
    initMatrixQ[1, 1] = minDiagQ
    initMatrixQ[0, 1] = 0.0
    initMatrixQ[1, 0] = 0.0
    return initMatrixQ


def constructMatrixH(
    m: int, coefficients: Optional[np.ndarray] = None
) -> npt.NDArray[np.float32]:
    r"""Build the observation model matrix :math:`\mathbf{H}`.

    :param m: Number of observations.
    :type m: int
    :param coefficients: Optional coefficients for the observation model,
        which can be used to weight the observations manually.
    :type coefficients: Optional[np.ndarray]
    :return: The observation model matrix :math:`\mathbf{H}`.
    :rtype: npt.NDArray[np.float32]

    :seealso: :class:`observationParams`, class:`inputParams`
    """
    if coefficients is None:
        coefficients = np.ones(m, dtype=np.float32)
    elif isinstance(coefficients, list):
        coefficients = np.array(coefficients, dtype=np.float32)
    initMatrixH = np.empty((m, 2), dtype=np.float32)
    initMatrixH[:, 0] = coefficients.astype(np.float32)
    initMatrixH[:, 1] = np.zeros(m, dtype=np.float32)
    return initMatrixH


def runConsenrich(
    matrixData: np.ndarray,
    matrixMunc: np.ndarray,
    deltaF: float,
    minQ: float,
    maxQ: float,
    offDiagQ: float,
    dStatAlpha: float,
    dStatd: float,
    dStatPC: float,
    dStatUseMean: bool,
    stateInit: float,
    stateCovarInit: float,
    boundState: bool,
    stateLowerBound: float,
    stateUpperBound: float,
    chunkSize: int,
    progressIter: int,
    coefficientsH: Optional[np.ndarray] = None,
    residualCovarInversionFunc: Optional[Callable] = None,
    adjustProcessNoiseFunc: Optional[Callable] = None,
    covarClip: float = 3.0,
    projectStateDuringFiltering: bool = False,
    textPlotDstatHistogram: bool = False,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    r"""Run consenrich on a contiguous segment (e.g. a chromosome) of read-density-based data from multiple samples.
    Completes the forward and backward passes given data :math:`\mathbf{Z}^{m \times n}` and
    corresponding uncertainty tracks :math:`\mathbf{R}_{[1:n, (11:mm)]}` (see :func:`getMuncTrack`).

    This is the primary function implementing the core Consenrich algorithm. Users requiring specialized
    preprocessing may prefer to call this function programmatically on their own preprocessed data rather
    than using the command-line interface.

    :param deltaF: See :class:`processParams`.
    :type deltaF: float
    :param minQ: See :class:`processParams`.
    :type minQ: float
    :param maxQ: See :class:`processParams`.
    :type maxQ: float
    :param offDiagQ: See :class:`processParams`.
    :type offDiagQ: float
    :param dStatAlpha: See :class:`processParams`.
    :type dStatAlpha: float
    :param dStatd: See :class:`processParams`.
    :type dStatd: float
    :param dStatPC: See :class:`processParams`.
    :type dStatPC: float
    :param dStatUseMean: See :class:`processParams`.
    :type dStatUseMean: bool
    :param stateInit: See :class:`stateParams`.
    :type stateInit: float
    :param stateCovarInit: See :class:`stateParams`.
    :type stateCovarInit: float
    :param chunkSize: Number of genomic intervals' data to keep in memory before flushing to disk.
    :type chunkSize: int
    :param progressIter: The number of iterations after which to log progress.
    :type progressIter: int
    :param coefficientsH: Optional coefficients for the observation model matrix :math:`\mathbf{H}`.
        If None, the coefficients are set to 1.0 for all samples.
    :type coefficientsH: Optional[np.ndarray]
    :param residualCovarInversionFunc: Callable function to invert the residual (innovation) covariance matrix at each interval, :math:`\mathbf{E}_{[i]}`.
        If None, defaults to :func:`cconsenrich.cinvertMatrixE`.
    :type residualCovarInversionFunc: Optional[Callable]
    :param adjustProcessNoiseFunc: Function to adjust the process noise covariance matrix :math:`\mathbf{Q}_{[i]}`.
        If None, defaults to :func:`cconsenrich.updateProcessNoiseCovariance`.
    :type adjustProcessNoiseFunc: Optional[Callable]
    :param covarClip: For numerical stability, truncate state/process noise covariances
        to :math:`[10^{-\textsf{covarClip}}, 10^{\textsf{covarClip}}]`.
    :type covarClip: float
    :param projectStateDuringFiltering: If `True`, the posterior state estimates are projected to the feasible region defined by `stateLowerBound`, `stateUpperBound` *during* iteration.
    See the constrained+weighted least-squares problem solved in :func:`consenrich.cconsenrich._projectToBox` and Simon, 2010.
    :type projectStateDuringFiltering: bool:
    :param boundState: If `True` final state estimates are clipped to [stateLowerBound, stateUpperBound] post-hoc.
    :type boundState: bool
    :return: Tuple of three numpy arrays:
        - post-fit (forward/backward-smoothed) state estimates :math:`\widetilde{\mathbf{x}}_{[i]}` of shape :math:`n \times 2`
        - post-fit (forward/backward-smoothed) state covariance estimates :math:`\widetilde{\mathbf{P}}_{[i]}` of shape :math:`n \times 2 \times 2`
        - post-fit residuals (after forward/backward smoothing) :math:`\widetilde{\mathbf{y}}_{[i]}` of shape :math:`n \times m`
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    :raises ValueError: If the number of samples in `matrixData` is not equal to the number of samples in `matrixMunc`.
    :seealso: :class:`observationParams`, :class:`processParams`, :class:`stateParams`
    """

    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.ascontiguousarray(matrixMunc, dtype=np.float32)

    # -------
    # check edge cases
    if matrixData.ndim == 1:
        matrixData = matrixData[None, :]
    elif matrixData.ndim != 2:
        raise ValueError(
            "`matrixData` must be 1D or 2D (got ndim = "
            f"{matrixData.ndim})"
        )
    if matrixMunc.ndim == 1:
        matrixMunc = matrixMunc[None, :]
    elif matrixMunc.ndim != 2:
        raise ValueError(
            "`matrixMunc` must be 1D or 2D (got ndim = "
            f"{matrixMunc.ndim})"
        )
    if matrixMunc.shape != matrixData.shape:
        raise ValueError(
            f"`matrixMunc` shape {matrixMunc.shape} not equal to `matrixData` shape {matrixData.shape}"
        )

    m, n = matrixData.shape
    if m < 1 or n < 1:
        raise ValueError(
            f"`matrixData` and `matrixMunc` need positive m x n, shape={matrixData.shape})"
        )

    if n <= 100:
        logger.warning(
            f"`matrixData` and `matrixMunc` span very fer genomic intervals (n={n})...is this correct?"
        )

    if chunkSize < 1:
        logger.warning(
            f"`chunkSize` must be positive, setting to 1000000"
        )
        chunkSize = 1_000_000

    if chunkSize > n:
        logger.warning(
            f"`chunkSize` of {chunkSize} is greater than the number of intervals (n={n}), setting to {n}"
        )
        chunkSize = n
    # -------
    phiHat__: float = np.float32(1.0)
    inflatedQ: bool = False
    dStat: float = np.float32(0.0)
    y64 = np.empty(m, dtype=np.float64)
    sq64 = np.empty(m, dtype=np.float64)
    tmp64 = np.empty(m, dtype=np.float64)
    vectorD = np.zeros(n, dtype=np.float32)
    countAdjustments: int = 0

    matrixF: np.ndarray = constructMatrixF(deltaF)
    matrixQ: np.ndarray = constructMatrixQ(minQ, offDiagQ=offDiagQ)
    matrixQCopy: np.ndarray = matrixQ.copy()
    matrixP: np.ndarray = np.eye(2, dtype=np.float32) * np.float32(
        stateCovarInit
    )

    u64 = None
    u2_64 = None
    if coefficientsH is not None:
        u64 = np.ascontiguousarray(
            coefficientsH, dtype=np.float64
        ).ravel()
        u2_64 = u64 * u64

    vectorX: np.ndarray = np.array([stateInit, 0.0], dtype=np.float32)
    vectorY: np.ndarray = np.zeros(m, dtype=np.float32)

    clipSmall: float = 10 ** (-covarClip)
    clipBig: float = 10 ** (covarClip)

    if adjustProcessNoiseFunc is None:
        adjustProcessNoiseFunc = (
            cconsenrich.updateProcessNoiseCovariance
        )

    with TemporaryDirectory() as tempDir_:
        stateForwardPathMM = os.path.join(
            tempDir_, "stateForward.dat"
        )
        stateCovarForwardPathMM = os.path.join(
            tempDir_, "stateCovarForward.dat"
        )
        pNoiseForwardPathMM = os.path.join(
            tempDir_, "pNoiseForward.dat"
        )
        stateBackwardPathMM = os.path.join(
            tempDir_, "stateSmoothed.dat"
        )
        stateCovarBackwardPathMM = os.path.join(
            tempDir_, "stateCovarSmoothed.dat"
        )
        postFitResidualsPathMM = os.path.join(
            tempDir_, "postFitResiduals.dat"
        )

        # ==========================
        # forward: 0,1,2,...,n-1
        # ==========================
        stateForward = np.memmap(
            stateForwardPathMM,
            dtype=np.float32,
            mode="w+",
            shape=(n, 2),
        )
        stateCovarForward = np.memmap(
            stateCovarForwardPathMM,
            dtype=np.float32,
            mode="w+",
            shape=(n, 2, 2),
        )
        pNoiseForward = np.memmap(
            pNoiseForwardPathMM,
            dtype=np.float32,
            mode="w+",
            shape=(n, 2, 2),
        )

        def _forwardPass(
            phiScale: float,
            collectD_: bool = True,
            isInitialPass: bool = False,
        ) -> tuple[float, int]:
            nonlocal inflatedQ, matrixQ, matrixQCopy, matrixP, vectorX, vectorY, countAdjustments, vectorD
            inflatedQ = False
            countAdjustments = 0
            matrixQ = constructMatrixQ(minQ, offDiagQ=offDiagQ)
            matrixQCopy = matrixQ.copy()
            matrixP = np.eye(2, dtype=np.float32) * np.float32(
                stateCovarInit
            )
            vectorX = np.array([stateInit, 0.0], dtype=np.float32)
            vectorY = np.zeros(m, dtype=np.float32)

            if isInitialPass:
                stateForwardOut = None
                stateCovarForwardOut = None
                pNoiseForwardOut = None
            else:
                stateForwardOut = stateForward
                stateCovarForwardOut = stateCovarForward
                pNoiseForwardOut = pNoiseForward

            progressBar = None
            if (
                (not isInitialPass)
                and (progressIter is not None)
                and (progressIter > 0)
            ):
                progressBar = tqdm(total=n, unit="intervals")

            try:
                phiHat__, countAdjustments__, vectorDOut = (
                    cconsenrich.cforwardPass(
                        matrixData=matrixData,
                        matrixMunc=matrixMunc,
                        matrixF=matrixF,
                        matrixQ=matrixQ,
                        matrixQCopy=matrixQCopy,
                        phiScale=float(phiScale),
                        dStatAlpha=float(dStatAlpha),
                        dStatd=float(dStatd),
                        dStatPC=float(dStatPC),
                        maxQ=float(maxQ),
                        minQ=float(minQ),
                        stateInit=float(stateInit),
                        stateCovarInit=float(stateCovarInit),
                        collectD=bool(collectD_),
                        coefficientsH=coefficientsH,
                        covarClip=float(covarClip),
                        pad=float(1e-3),
                        projectStateDuringFiltering=bool(
                            projectStateDuringFiltering
                        ),
                        stateLowerBound=float(stateLowerBound),
                        stateUpperBound=float(stateUpperBound),
                        chunkSize=int(chunkSize),
                        stateForward=stateForwardOut,
                        stateCovarForward=stateCovarForwardOut,
                        pNoiseForward=pNoiseForwardOut,
                        vectorD=vectorD,
                        progressBar=progressBar,
                        progressIter=int(progressIter),
                    )
                )
            finally:
                if progressBar is not None:
                    progressBar.close()

            vectorD = vectorDOut
            return float(phiHat__), int(countAdjustments__)

        logger.info("Running forward pass...\n")
        phiHat__, countAdjustments__ = _forwardPass(
            phiScale=1.0, collectD_=False, isInitialPass=False
        )

        stateForwardArr = stateForward
        stateCovarForwardArr = stateCovarForward
        pNoiseForwardArr = pNoiseForward
        stateForward.flush()
        stateCovarForward.flush()
        pNoiseForward.flush()

        logger.info(
            f"`D_[i] > α_D` triggered adjustments to Q_[i] at "
            f"[{round(((1.0 * countAdjustments__) / n) * 100.0, 4)}%]"
        )

        _textplotDstatHistogram(
            vectorD,
            n=vectorD.size,
            enabled=textPlotDstatHistogram,
        )

        # ==========================
        # backward: n-1,n-2,...,0
        # ==========================
        stateSmoothed = np.memmap(
            stateBackwardPathMM,
            dtype=np.float32,
            mode="w+",
            shape=(n, 2),
        )
        stateCovarSmoothed = np.memmap(
            stateCovarBackwardPathMM,
            dtype=np.float32,
            mode="w+",
            shape=(n, 2, 2),
        )
        postFitResiduals = np.memmap(
            postFitResidualsPathMM,
            dtype=np.float32,
            mode="w+",
            shape=(n, m),
        )

        progressBarBack = None
        if (progressIter is not None) and (progressIter > 0):
            progressBarBack = tqdm(total=(n - 1), unit="intervals")

            stateSmoothedArr, stateCovarSmoothedArr, postFitResidualsArr = cconsenrich.cbackwardPass(
            matrixData=matrixData,
            matrixF=matrixF,
            stateForward=stateForwardArr,
            stateCovarForward=stateCovarForwardArr,
            pNoiseForward=pNoiseForwardArr,
            coefficientsH=coefficientsH,
            projectStateDuringFiltering=bool(
                projectStateDuringFiltering
            ),
            stateLowerBound=float(stateLowerBound),
            stateUpperBound=float(stateUpperBound),
            covarClip=float(covarClip),
            chunkSize=int(chunkSize),
            stateSmoothed=stateSmoothed,
            stateCovarSmoothed=stateCovarSmoothed,
            postFitResiduals=postFitResiduals,
            progressBar=progressBarBack,
            progressIter=int(progressIter),
        )

        if progressBarBack is not None:
            progressBarBack.close()

        stateSmoothedArr.flush()
        stateCovarSmoothedArr.flush()
        postFitResidualsArr.flush()

        outStateSmoothed = np.array(stateSmoothedArr, copy=True)
        outPostFitResiduals = np.array(postFitResidualsArr, copy=True)
        outStateCovarSmoothed = np.array(
            stateCovarSmoothedArr, copy=True
        )

    if boundState:
        np.clip(
            outStateSmoothed[:, 0],
            np.float32(stateLowerBound),
            np.float32(stateUpperBound),
            out=outStateSmoothed[:, 0],
        )

    return (
        outStateSmoothed,
        outStateCovarSmoothed,
        outPostFitResiduals,
    )


def getPrimaryState(
    stateVectors: np.ndarray,
    roundPrecision: int = 4,
    stateLowerBound: Optional[float] = None,
    stateUpperBound: Optional[float] = None,
    boundState: bool = False,
) -> npt.NDArray[np.float32]:
    r"""Get the primary state estimate from each vector after running Consenrich.

    :param stateVectors: State vectors from :func:`runConsenrich`.
    :type stateVectors: npt.NDArray[np.float32]
    :return: A one-dimensional numpy array of the primary state estimates.
    :rtype: npt.NDArray[np.float32]
    """
    out_ = np.ascontiguousarray(stateVectors[:, 0], dtype=np.float32)
    if boundState:
        if stateLowerBound is not None:
            np.maximum(
                out_, np.float32(stateLowerBound), out=out_)
        if stateUpperBound is not None:
            np.minimum(
                out_, np.float32(stateUpperBound), out=out_)
    np.round(out_, decimals=roundPrecision, out=out_)
    return out_


def getStateCovarTrace(
    stateCovarMatrices: np.ndarray,
    roundPrecision: int = 4,
) -> npt.NDArray[np.float32]:
    r"""Get a one-dimensional array of state covariance traces after running Consenrich

    :param stateCovarMatrices: Estimated state covariance matrices :math:`\widetilde{\mathbf{P}}_{[i]}`
    :type stateCovarMatrices: np.ndarray
    :return: A one-dimensional numpy array of the traces of the state covariance matrices.
    :rtype: npt.NDArray[np.float32]
    """
    stateCovarMatrices = np.ascontiguousarray(
        stateCovarMatrices, dtype=np.float32
    )
    out_ = cconsenrich.cgetStateCovarTrace(stateCovarMatrices)
    np.round(out_, decimals=roundPrecision, out=out_)
    return out_


def getPrecisionWeightedResidual(
    postFitResiduals: np.ndarray,
    matrixMunc: Optional[np.ndarray] = None,
    stateCovarSmoothed: Optional[np.ndarray] = None,
    roundPrecision: int = 4,
) -> npt.NDArray[np.float32]:
    r"""Get a one-dimensional representation of average residuals after running Consenrich.

    Optionally, if `matrixMunc` is provided, residuals are precision-weighted by the respective covariances.

    :param postFitResiduals: Post-fit residuals :math:`\widetilde{\mathbf{y}}_{[i]}` from :func:`runConsenrich`.
    :type postFitResiduals: np.ndarray
    :param matrixMunc: An :math:`m \times n` sample-by-interval matrix -- At genomic intervals :math:`i = 1,2,\ldots,n`, the respective length-:math:`m` column is :math:`\mathbf{R}_{[i,11:mm]}`.
        That is, the observation noise levels for each sample :math:`j=1,2,\ldots,m` at interval :math:`i`. To keep memory usage minimal `matrixMunc` is not returned in full or computed in
        in :func:`runConsenrich`. If using Consenrich programmatically, run :func:`consenrich.core.getMuncTrack` for each sample's count data (rows in the matrix output of :func:`readBamSegments`).
    :type matrixMunc: np.ndarray
    :param stateCovarSmoothed: Post-fit (forward/backward-smoothed) state covariance matrices :math:`\widetilde{\mathbf{P}}_{[i]}` from :func:`runConsenrich`.
    :type stateCovarSmoothed: Optional[np.ndarray]
    :return: A one-dimensional array of "precision-weighted residuals"
    :rtype: npt.NDArray[np.float32]
    """

    n, m = postFitResiduals.shape
    postFitResiduals_CContig = np.ascontiguousarray(
        postFitResiduals, dtype=np.float32)

    if matrixMunc is None:
        return np.mean(postFitResiduals_CContig, axis=1)

    else:
        if matrixMunc.shape != (m, n):
            raise ValueError(
                f"matrixMunc should be (m,n)=({m}, {n}): observed {matrixMunc.shape}"
            )
    if stateCovarSmoothed is not None and (
        stateCovarSmoothed.ndim < 3 or len(stateCovarSmoothed) != n
    ):
        raise ValueError(
            "stateCovarSmoothed must be shape (n) x (2,2) (if provided)"
        )
    needsCopy = (
        (stateCovarSmoothed is not None)
        and len(stateCovarSmoothed) == n
    ) or (not matrixMunc.flags.writeable)

    matrixMunc_CContig = np.array(
        matrixMunc, dtype=np.float32, order="C", copy=needsCopy
    )

    if needsCopy:
        stateCovarArr00 = np.asarray(
            stateCovarSmoothed[:, 0, 0], dtype=np.float32
        )
        matrixMunc_CContig += stateCovarArr00

    np.maximum(
        matrixMunc_CContig, np.float32(1e-8), out=matrixMunc_CContig
    )
    out = cconsenrich.cgetPrecisionWeightedResidual(
        postFitResiduals_CContig, matrixMunc_CContig
    )
    np.round(out, decimals=roundPrecision, out=out)
    return out


def getMuncTrack(
    chromosome: str,
    intervals: np.ndarray,
    values: np.ndarray,
    stepSize: int,
    minR: float,
    maxR: float,
    sparseMap: Optional[dict] = None,
    blockSizeBP: Optional[int] = 1000,
    samplingIters: int = 25_000,
    randomSeed: int = 42,
    localWeight: float = 0.50,
    zeroPenalty: float = 1.0,
    approximationWindowLengthBP: int = 25_000,
    lowPassWindowLengthBP: int = 25_000,
    fitFunc: Optional[Callable] = None,
    fitFuncArgs: Optional[dict] = None,
    evalFunc: Optional[Callable] = None,
    excludeMask: Optional[np.ndarray] = None,
    binQuantile: float = 0.50,
    textPlotMeanVarianceTrend: bool = False,
    isTransformed: bool = True,
) -> npt.NDArray[np.float32]:
    r"""Approximate region- and sample-specific (**M**)easurement (**unc**)ertainty tracks

    Compute sample- and region-specific measurement uncertainty track :math:`{R}_{[i:n]}` as a
    weighted combination of (i) a global mean-variance trend and (ii) a rolling average of squared, first (or second)-order differences.

    * The global model is based on a mean-variance trend :math:`\hat{f}`, fit to pairs :math:`(\hat{\mu}_k, \hat{\sigma}^2_k)`
        for each of :math:`k=1,2,\ldots,K` (``samplingIters``) randomly sampled contiguous genomic blocks.
        Note, :math:`(\hat{\mu}_k, \hat{sigma}^2_k)` is computed by taking the average of squared residuals
        from an AR(1) model fit with a correction for ρ (see :func:`consenrich.cconsenrich.cmeanVarPairs`).

    * The local model, :math:`\hat{f}_{\textsf{local}}(i)`, is based rolling-window stats at each genomic
        *interval* :math:`i=1,2,\ldots,n`. Specifically, the local squared, first (or second)-order differences
        is computed over a window of size ``approximationWindowLengthBP``. See :func:`consenrich.cconsenrich.csumSquaredFOD`.

        Optionally, if the ``dict`` mapping ``sparseMap`` is provided (built from ``genomeParams.sparseBedFile``),
        the local model restricts the calculation to the nearest 'sparse' genomic regions at each interval :math:`i=1,2,\ldots,n`
        defined as regions devoid of or mutually exclusive with the targeted signal (:func:`consenrich.cconsenrich.cSparseAvg`).
        For instance, if targeting the histone PTM `H3K27ac`, sparse regions could be defined with a
        BED annotation of broad `H3K27me3` domains.


    The final quantity is a weighted combination from both models, controlled by ``localWeight``,

    .. math::

       R_{[i]} = \textsf{localWeight}\,\hat{f}_{\textsf{local}}(i)
                + (1-\textsf{localWeight})\,\hat{f}_{\textsf{global}}(\mu_i)

    :param chromosome: chromosome/contig name
    :type chromosome: str
    :param intervals: genomic intervals positions (start positions)
    :type intervals: npt.NDArray[np.uint32]
    :param minR: Minimum allowable uncertainty.
    :type minR: float
    :param maxR: Maximum allowable uncertainty.
    :type maxR: float
    :param sparseMap: Optional mapping of genomic intervals to sparse regions.
        If provided, the local model estimates are averaged over these sparse regions.
    :type sparseMap: Optional[dict]
    :param blockSizeBP: Size (in bp) of contiguous blocks to sample when estimating global mean-variance trend.
    :type blockSizeBP: int
    :param samplingIters: Number of contiguous blocks to sample when estimating global mean-variance trend.
    :type samplingIters: int
    :param randomSeed: Random seed for the sampling during global mean-variance trend estimation
    :type randomSeed: int
    :param localWeight: Weight of local model in mixed uncertainty estimate.
        ``--> 1.0 ignore global (mean-variance) model``, ``--> 0.0 ignore local rolling mean/var model``.
    :type localWeight: float
    :param zeroPenalty: Inflate variance at data points in the left tail of the mean-variance trend
        (i.e., low mean values) by this amount during global model fitting.
    :type zeroPenalty: float
    :param approximationWindowLengthBP: Window length (in bp) for local variance approximation. See :func:`getAverageLocalVarianceTrack`.
    :type approximationWindowLengthBP: int
    :param lowPassWindowLengthBP: Deprecated -- no effect.
    :type lowPassWindowLengthBP: int
    :param fitFunc: A *callable* function accepting input ``(arrayOfMeans,arrayOfVariances, **kwargs)``. Used to fit the global mean-variance model
    given sampled blocks from :func:``consenrich.cconsenrich.cmeanVarPairs``. Defaults to `cconsenrich.cmonotonicFit`, ridge-penalized, positive regression.
    :type fitFunc: Optional[Callable]
    :param fitFuncArgs: Additional keyword arguments to pass to `fitFunc`.
    :type fitFuncArgs: Optional[dict]
    :param evalFunc: A *callable* function with input (``outputFromFitFunc, arrayLengthN``) that evaluates the fitted :math:`\hat{f}(array[i])` at each genomic interval :math:`i=1,2,\ldots,n`.
    :type evalFunc: Optional[Callable]
    :return: An uncertainty track with same length as input
    :rtype: npt.NDArray[np.float32]
    """

    if fitFunc is None:
        fitFunc = cconsenrich.cmonotonicFit
    if evalFunc is None:
        evalFunc = cconsenrich.cmonotonicFitEval
    if fitFuncArgs is None:
        fitFuncArgs = {
            "ridge": 1.0e-4,
            "isTransformed": isTransformed,
        }

    if blockSizeBP is None:
        blockSizeBP = stepSize * 11
    blockSizeIntervals = int(blockSizeBP / stepSize)
    if blockSizeIntervals < 10:
        logger.warning(
            f"`blockSizeBP` is small for sampling (mean, variance) pairs...trying 11*stepSize"
        )
        blockSizeIntervals = 11

    intervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32)
    valuesArr = np.ascontiguousarray(values, dtype=np.float32)

    if excludeMask is None:
        excludeMaskArr = np.zeros_like(intervalsArr, dtype=np.uint8)
    else:
        excludeMaskArr = np.ascontiguousarray(
            excludeMask, dtype=np.uint8
        )

    cpy = localWeight
    localWeight = np.clip(localWeight, 0.0, 1.0)
    globalWeight = 1.0 - localWeight  # force sum-to-one
    if localWeight != cpy:
        logger.warning(
            f"`localWeight` clipped to [{localWeight}] to ensure in [0.0,1.0] and local+global=1.0"
        )
    logger.info(
        f"localWeight={localWeight}, globalWeight={globalWeight}",
    )

    # I: Global model (variance = f(mean))
    # ...  Variance as function of mean globally, as observed in contiguous blocks
    # ... in cmeanVarPairs, variance is measured in each block as AR(1) RSS/n-1 after
    # ... correcting for ρ.
    # ... For each block, we get a (blockMean, blockVar) pair, and `samplingIters`
    # ...  such pairs are used are used to fit the global trend.


    blockMeans, blockVars, starts, ends = cconsenrich.cmeanVarPairs(
        intervalsArr,
        valuesArr,
        blockSizeIntervals,
        samplingIters,
        randomSeed,
        excludeMaskArr,
    )

    #  (i) Fit mean ~ variance relationship as \hat{f}
    # ... over sampled blocks' stats (mean_k, var_k) k=1,..,samplingIters
    sortIdx = np.argsort(blockMeans)  # jointly sorted by blockMean
    blockMeansSorted = blockMeans[sortIdx]
    blockVarsSorted = blockVars[sortIdx]

    cleanBinMeans, cleanBinVars = _extractUpperTail(
        blockMeansSorted, blockVarsSorted, q=binQuantile,
    )

    blockMeansSorted = cleanBinMeans.astype(np.float32)
    blockVarsSorted = cleanBinVars.astype(np.float32)

    keep = np.isfinite(blockMeansSorted) & np.isfinite(
        blockVarsSorted
    )
    blockMeansSorted = blockMeansSorted[keep]
    blockVarsSorted = blockVarsSorted[keep]

    opt = fitFunc(
        blockMeansSorted,
        blockVarsSorted,
        **fitFuncArgs,
    ).astype(np.float32)

    globalModelVariances = evalFunc(
        opt, cconsenrich.cEMA(valuesArr, 2/(blockSizeIntervals+1)), isTransformed,
    ).astype(np.float32)

    _textplotMeanVarianceTrend(
        blockMeans=blockMeans,
        blockVars=blockVars,
        blockMeansSorted=blockMeansSorted,
        opt=opt,
        evalFunc=evalFunc,
        isTransformed=isTransformed,
        enabled=textPlotMeanVarianceTrend,
    )

    # II: Local model (local moment-based variance via sliding windows)
    # ... (a) At each genomic interval i = 1,2,...,n,
    # ... apply local/moment-based heuristic on first/second order differences
    # ... (b) `sparseMap` is an optional mapping (implemented as a dictionary)
    # ...    sparseMap(i) --> {F_i1,F_i2,...,F_i{numNearest}}
    # ... where each F_ij is a 'sparse' genomic region *devoid* of previously-annotated
    # ... regulatory elements. If provided,
    # ...  `trackALV_{new}(i) = average(trackALV_{initial}(F_{i,1}, F_{i,2},..., F_{i,numNearest})`
    # ... is the track from (a) are aggregated over sparseMap(i) to get each localModelVariance(i)
    localModelVariances = np.asarray(
        cconsenrich.csumSquaredSOD(
            valuesArr,
            max(2, int(approximationWindowLengthBP / stepSize)),
        ),
        dtype=np.float32,
    )

    if sparseMap is not None:
        localModelVariances = cconsenrich.cSparseAvg(
            localModelVariances.copy(), sparseMap
        )

    # III: mix local and global models, weight sum to one
    muncTrack = (localWeight * localModelVariances) + (
        globalWeight * globalModelVariances
    )

    return np.clip(muncTrack, minR, maxR).astype(np.float32)


def sparseIntersection(
    chromosome: str, intervals: np.ndarray, sparseBedFile: str
) -> npt.NDArray[np.int64]:
    r"""Returns intervals in the chromosome that overlap with the 'sparse' features.

    :param chromosome: The chromosome name.
    :type chromosome: str
    :param intervals: The genomic intervals to consider.
    :type intervals: np.ndarray
    :param sparseBedFile: Path to the sparse BED file.
    :type sparseBedFile: str
    :return: A numpy array of start positions of the sparse features that overlap with the intervals
    :rtype: np.ndarray[Tuple[Any], np.dtype[Any]]
    :seealso: :func:`getSparseMap`, :class:`consenrich.core.observationParams`
    """

    stepSize: int = intervals[1] - intervals[0]
    chromFeatures: bed.BedTool = (
        bed.BedTool(sparseBedFile)
        .sort()
        .merge()
        .filter(
            lambda b: (
                b.chrom == chromosome
                and b.start > intervals[0]
                and b.end < intervals[-1]
                and (b.end - b.start) >= stepSize
            )
        )
    )
    centeredFeatures: bed.BedTool = chromFeatures.each(
        adjustFeatureBounds, stepSize=stepSize
    )

    start0: int = int(intervals[0])
    last: int = int(intervals[-1])
    chromFeatures: bed.BedTool = (
        bed.BedTool(sparseBedFile)
        .sort()
        .merge()
        .filter(
            lambda b: (
                b.chrom == chromosome
                and b.start > start0
                and b.end < last
                and (b.end - b.start) >= stepSize
            )
        )
    )
    centeredFeatures: bed.BedTool = chromFeatures.each(
        adjustFeatureBounds, stepSize=stepSize
    )
    centeredStarts = []
    for f in centeredFeatures:
        s = int(f.start)
        if start0 <= s <= last and (s - start0) % stepSize == 0:
            centeredStarts.append(s)
    return np.asarray(centeredStarts, dtype=np.int64)


def adjustFeatureBounds(
    feature: bed.Interval, stepSize: int
) -> bed.Interval:
    r"""Adjust the start and end positions of a BED feature to be centered around a step."""
    feature.start = cconsenrich.stepAdjustment(
        (feature.start + feature.end) // 2, stepSize
    )
    feature.end = feature.start + stepSize
    return feature


def getSparseMap(
    chromosome: str,
    intervals: np.ndarray,
    numNearest: int,
    sparseBedFile: str,
) -> dict:
    r"""Build a map between each genomic interval and numNearest sparse features

    :param chromosome: The chromosome name. Note, this function only needs to be run once per chromosome.
    :type chromosome: str
    :param intervals: The genomic intervals to map.
    :type intervals: np.ndarray
    :param numNearest: The number of nearest sparse features to consider
    :type numNearest: int
    :param sparseBedFile: path to the sparse BED file.
    :type sparseBedFile: str
    :return: A dictionary mapping each interval index to the indices of the nearest sparse regions.
    :rtype: dict[int, np.ndarray]

    """
    numNearest = numNearest
    sparseStarts = sparseIntersection(
        chromosome, intervals, sparseBedFile
    )
    idxSparseInIntervals = np.searchsorted(
        intervals, sparseStarts, side="left"
    )
    centers = np.searchsorted(sparseStarts, intervals, side="left")
    sparseMap: dict = {}
    for i, (interval, center) in enumerate(zip(intervals, centers)):
        left = max(0, center - numNearest)
        right = min(len(sparseStarts), center + numNearest)
        candidates = np.arange(left, right)
        dists = np.abs(sparseStarts[candidates] - interval)
        take = np.argsort(dists)[:numNearest]
        sparseMap[i] = idxSparseInIntervals[candidates[take]]
    return sparseMap


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
      Each interval is assumed `stepSize`.
    :type intervals: np.ndarray
    :param bedFile: Path to a sorted and merged BED file
    :type bedFile: str
    :return: An `intervals`-length mask s.t. True indicates the interval overlaps a feature in the BED file.
    :rtype: np.ndarray
    """
    if not os.path.exists(bedFile):
        raise ValueError(f"Could not find {bedFile}")
    if len(intervals) < 2:
        raise ValueError(
            "intervals must contain at least two positions"
        )
    bedFile_ = str(bedFile)

    # (possibly redundant) creation of uint32 version
    # + quick check for constant steps
    intervals_ = np.asarray(intervals, dtype=np.uint32)
    if (intervals_[1] - intervals_[0]) != (
        intervals_[-1] - intervals_[-2]
    ):
        raise ValueError("Intervals are not fixed in size")

    stepSize_: int = intervals[1] - intervals[0]
    return cconsenrich.cbedMask(
        chromosome,
        bedFile_,
        intervals_,
        stepSize_,
    ).astype(np.bool_)


def autoDeltaF(
    bamFiles: List[str],
    stepSize: int,
    fragmentLengths: Optional[List[int]] = None,
    fallBackFragmentLength: int = 147,
    randomSeed: int = 42,
) -> float:
    r"""(Experimental) Set `deltaF` as the ratio intervalLength:fragmentLength.

    Computes average fragment length across samples and sets `processParams.deltaF = countingArgs.stepSize / medianFragmentLength`.

    Where `stepSize` is small, adjacent genomic intervals may share information from the same fragments. This motivates
    a smaller `deltaF` (i.e., less state change between neighboring intervals).

    :param stepSize: Length of genomic intervals/bins. See :class:`countingParams`.
    :type stepSize: int
    :param bamFiles: List of sorted/indexed BAM files to estimate fragment lengths from if they are not provided directly.
    :type bamFiles: List[str]
    :param fragmentLengths: Optional list of fragment lengths (in bp) for each sample. If provided, these values are used directly instead of estimating from `bamFiles`.
    :type fragmentLengths: Optional[List[int]]
    :param fallBackFragmentLength: If fragment length estimation from a BAM file fails, this value is used instead.
    :type fallBackFragmentLength: int
    :param randomSeed: Random seed for fragment length estimation.
    :type randomSeed: int
    :return: Estimated `deltaF` value.
    :rtype: float
    :seealso: :func:`cconsenrich.cgetFragmentLength`, :class:`processParams`, :class:`countingParams`
    """

    avgFragmentLength: float = 0.0
    if (
        fragmentLengths is not None
        and len(fragmentLengths) > 0
        and all(isinstance(x, (int, float)) for x in fragmentLengths)
    ):
        avgFragmentLength = np.median(fragmentLengths)
    elif bamFiles is not None and len(bamFiles) > 0:
        fragmentLengths_ = []
        for bamFile in bamFiles:
            fLen = cconsenrich.cgetFragmentLength(
                bamFile,
                fallBack=fallBackFragmentLength,
                randSeed=randomSeed,
            )
            fragmentLengths_.append(fLen)
        avgFragmentLength = np.median(fragmentLengths_)
    else:
        raise ValueError(
            "One of `fragmentLengths` or `bamFiles` is required..."
        )
    if avgFragmentLength > 0:
        deltaF = round(stepSize / float(avgFragmentLength), 4)
        logger.info(f"Setting `processParams.deltaF`={deltaF}")
        return np.float32(deltaF)
    else:
        raise ValueError(
            "Average cross-sample fraglen estimation failed"
        )


def _forPlotsSampleBlockStats(
    values_: npt.NDArray[np.float32],
    blockSize_: int,
    numBlocks_: int,
    statFunction_: Callable = np.mean,
    randomSeed_: int = 42,
):
    r"""Pure python helper for plotting distributions of block-sampled statistics.

    Intended for use in the plotting functions, not as an alternative to
    the Cython ``cconsenrich.csampleBlockStats`` function used in the
    `matching` module. Call on 32bit numpy arrays so that copies are not made.

    :param values: One-dimensional array of values to sample blocks from.
    :type values: np.ndarray
    :param blockSize: Length of each block to sample.
    :type blockSize: int
    :param numBlocks: Number of blocks to sample.
    :type numBlocks: int
    """
    np.random.seed(randomSeed_)

    if type(values_) == npt.NDArray[np.float32]:
        x = values_
    else:
        x = np.ascontiguousarray(values_, dtype=np.float32)
    n = x.shape[0]
    if blockSize_ > n:
        logger.warning(
            f"`blockSize>values.size`...setting `blockSize` = {max(n // 2, 1)}."
        )
        blockSize_ = int(max(n // 2, 1))

    maxStart = n - blockSize_ + 1

    # avoid copies
    blockView = as_strided(
        x,
        shape=(maxStart, blockSize_),
        strides=(x.strides[0], x.strides[0]),
    )
    starts = np.random.randint(0, maxStart, size=numBlocks_)
    return statFunction_(blockView[starts], axis=1)


def plotStateEstimatesHistogram(
    chromosome: str,
    plotPrefix: str,
    primaryStateValues: npt.NDArray[np.float32],
    blockSize: int = 10,
    numBlocks: int = 10_000,
    statFunction: Callable = np.mean,
    randomSeed: int = 42,
    roundPrecision: int = 4,
    plotHeightInches: float = 8.0,
    plotWidthInches: float = 10.0,
    plotDPI: int = 300,
    plotDirectory: str | None = None,
) -> str | None:
    r"""(Experimental) Plot a histogram of block-sampled (within-chromosome) primary state estimates.

    :param plotPrefix: Prefixes the output filename
    :type plotPrefix: str
    :param primaryStateValues: 1D 32bit float array of primary state estimates, i.e., :math:`\widetilde{\mathbf{x}}_{[i,1]}`,
        that is, ``stateSmoothed[0,:]`` from :func:`runConsenrich`. See also :func:`getPrimaryState`.
    :type primaryStateValues: npt.NDArray[np.float32]
    :param blockSize: Number of contiguous intervals to sample per block.
    :type blockSize: int
    :param numBlocks: Number of samples to draw
    :type numBlocks: int
    :param statFunction: Numpy callable function to compute on each sampled block (e.g., `np.mean`, `np.median`).
    :type statFunction: Callable
    :param plotDirectory: If provided, saves the plot to this directory. The directory should exist.
    :type plotDirectory: str | None
    """

    if _checkMod("matplotlib"):
        import matplotlib.pyplot as plt
    else:
        logger.warning("matplotlib not found...returning None")
        return None

    if plotDirectory is None:
        plotDirectory = os.getcwd()
    elif not os.path.exists(plotDirectory):
        raise ValueError(
            f"`plotDirectory` {plotDirectory} does not exist"
        )
    elif not os.path.isdir(plotDirectory):
        raise ValueError(
            f"`plotDirectory` {plotDirectory} is not a directory"
        )

    plotFileName = os.path.join(
        plotDirectory,
        f"consenrichPlot_hist_{chromosome}_{plotPrefix}_state.png",
    )
    binnedStateEstimates = _forPlotsSampleBlockStats(
        values_=primaryStateValues,
        blockSize_=blockSize,
        numBlocks_=numBlocks,
        statFunction_=statFunction,
        randomSeed_=randomSeed,
    )
    plt.figure(
        figsize=(plotWidthInches, plotHeightInches), dpi=plotDPI
    )
    plt.hist(
        binnedStateEstimates,
        bins="doane",
        color="blue",
        alpha=0.85,
        edgecolor="black",
        fill=True,
    )
    plt.title(
        rf"Histogram: {numBlocks} sampled blocks ({blockSize} contiguous intervals each): Posterior Signal Estimates $\widetilde{{x}}_{{[1 : n]}}$",
    )
    plt.savefig(plotFileName, dpi=plotDPI)
    plt.close()
    if os.path.exists(plotFileName):
        logger.info(
            f"Wrote state estimate histogram to {plotFileName}"
        )
        return plotFileName
    logger.warning(
        f"Failed to create histogram. {plotFileName} not written."
    )
    return None


def plotResidualsHistogram(
    chromosome: str,
    plotPrefix: str,
    residuals: npt.NDArray[np.float32],
    blockSize: int = 10,
    numBlocks: int = 10_000,
    statFunction: Callable = np.mean,
    randomSeed: int = 42,
    roundPrecision: int = 4,
    plotHeightInches: float = 8.0,
    plotWidthInches: float = 10.0,
    plotDPI: int = 300,
    flattenResiduals: bool = False,
    plotDirectory: str | None = None,
) -> str | None:
    r"""(Experimental) Plot a histogram of within-chromosome post-fit residuals.

    .. note::

      To economically represent residuals across multiple samples, at each genomic interval :math:`i`,
      we randomly select a single sample's residual in vector :math:`\mathbf{y}_{[i]} = \mathbf{Z}_{[:,i]} - \mathbf{H}\widetilde{\mathbf{x}}_{[i]}`
      to obtain a 1D array, :math:`\mathbf{a} \in \mathbb{R}^{1 \times n}`. Then, contiguous blocks :math:`\mathbf{a}_{[k:k+blockSize]}` are sampled
      to compute the desired statistic (e.g., mean, median). These block statistics comprise the empirical distribution plotted in the histogram.

    :param plotPrefix: Prefixes the output filename
    :type plotPrefix: str
    :param residuals: :math:`m \times n` (sample-by-interval) 32bit float array of post-fit residuals.
    :type residuals: npt.NDArray[np.float32]
    :param blockSize: Number of contiguous intervals to sample per block.
    :type blockSize: int
    :param numBlocks: Number of samples to draw
    :type numBlocks: int
    :param statFunction: Numpy callable function to compute on each sampled block (e.g., `np.mean`, `np.median`).
    :type statFunction: Callable
    :param flattenResiduals: If True, flattens the :math:`m \times n` (sample-by-interval) residuals
        array to 1D (via `np.flatten`) before sampling blocks. If False, a random row (sample) is
        selected for each column (interval) prior to the block sampling.
        in each iteration.
    :type flattenResiduals: bool
    :param plotDirectory: If provided, saves the plot to this directory. The directory should exist.
    :type plotDirectory: str | None
    """

    if _checkMod("matplotlib"):
        import matplotlib.pyplot as plt
    else:
        logger.warning("matplotlib not found...returning None")
        return None

    if plotDirectory is None:
        plotDirectory = os.getcwd()
    elif not os.path.exists(plotDirectory):
        raise ValueError(
            f"`plotDirectory` {plotDirectory} does not exist"
        )
    elif not os.path.isdir(plotDirectory):
        raise ValueError(
            f"`plotDirectory` {plotDirectory} is not a directory"
        )

    plotFileName = os.path.join(
        plotDirectory,
        f"consenrichPlot_hist_{chromosome}_{plotPrefix}_residuals.png",
    )

    x = np.ascontiguousarray(residuals, dtype=np.float32)

    if not flattenResiduals:
        n, m = x.shape
        rng = np.random.default_rng(randomSeed)
        sample_idx = rng.integers(0, m, size=n)
        x = x[np.arange(n), sample_idx]
    else:
        x = x.ravel()

    binnedResiduals = _forPlotsSampleBlockStats(
        values_=x,
        blockSize_=blockSize,
        numBlocks_=numBlocks,
        statFunction_=statFunction,
        randomSeed_=randomSeed,
    )
    plt.figure(
        figsize=(plotWidthInches, plotHeightInches), dpi=plotDPI
    )
    plt.hist(
        binnedResiduals,
        bins="doane",
        color="blue",
        alpha=0.85,
        edgecolor="black",
        fill=True,
    )
    plt.title(
        rf"Histogram: {numBlocks} sampled blocks ({blockSize} contiguous intervals each): Post-Fit Residuals $\widetilde{{y}}_{{[1 : m,  1 : n]}}$",
    )
    plt.savefig(plotFileName, dpi=plotDPI)
    plt.close()
    if os.path.exists(plotFileName):
        logger.info(f"Wrote residuals histogram to {plotFileName}")
        return plotFileName
    logger.warning(
        f"Failed to create histogram. {plotFileName} not written."
    )
    return None


def plotStateStdHistogram(
    chromosome: str,
    plotPrefix: str,
    stateStd: npt.NDArray[np.float32],
    blockSize: int = 10,
    numBlocks: int = 10_000,
    statFunction: Callable = np.mean,
    randomSeed: int = 42,
    roundPrecision: int = 4,
    plotHeightInches: float = 8.0,
    plotWidthInches: float = 10.0,
    plotDPI: int = 300,
    plotDirectory: str | None = None,
) -> str | None:
    r"""(Experimental) Plot a histogram of block-sampled (within-chromosome) primary state standard deviations, i.e., :math:`\sqrt{\widetilde{\mathbf{P}}_{[i,11]}}`.

    :param plotPrefix: Prefixes the output filename
    :type plotPrefix: str
    :param stateStd: 1D numpy 32bit float array of primary state standard deviations, i.e., :math:`\sqrt{\widetilde{\mathbf{P}}_{[i,11]}}`,
        that is, the first diagonal elements in the :math:`n \times (2 \times 2)` numpy array `stateCovarSmoothed`. Access as ``(stateCovarSmoothed[:, 0, 0]``.
    :type stateStd: npt.NDArray[np.float32]
    :param blockSize: Number of contiguous intervals to sample per block.
    :type blockSize: int
    :param numBlocks: Number of samples to draw
    :type numBlocks: int
    :param statFunction: Numpy callable function to compute on each sampled block (e.g., `np.mean`, `np.median`).
    :type statFunction: Callable
    :param plotDirectory: If provided, saves the plot to this directory. The directory should exist.
    :type plotDirectory: str | None
    """

    if _checkMod("matplotlib"):
        import matplotlib.pyplot as plt
    else:
        logger.warning("matplotlib not found...returning None")
        return None

    if plotDirectory is None:
        plotDirectory = os.getcwd()
    elif not os.path.exists(plotDirectory):
        raise ValueError(
            f"`plotDirectory` {plotDirectory} does not exist"
        )
    elif not os.path.isdir(plotDirectory):
        raise ValueError(
            f"`plotDirectory` {plotDirectory} is not a directory"
        )

    plotFileName = os.path.join(
        plotDirectory,
        f"consenrichPlot_hist_{chromosome}_{plotPrefix}_stateStd.png",
    )

    binnedStateStdEstimates = _forPlotsSampleBlockStats(
        values_=stateStd,
        blockSize_=blockSize,
        numBlocks_=numBlocks,
        statFunction_=statFunction,
        randomSeed_=randomSeed,
    )
    plt.figure(
        figsize=(plotWidthInches, plotHeightInches),
        dpi=plotDPI,
    )
    plt.hist(
        binnedStateStdEstimates,
        bins="doane",
        color="blue",
        alpha=0.85,
        edgecolor="black",
        fill=True,
    )
    plt.title(
        rf"Histogram: {numBlocks} sampled blocks ({blockSize} contiguous intervals each): Posterior State StdDev $\sqrt{{\widetilde{{P}}_{{[1:n,11]}}}}$",
    )
    plt.savefig(plotFileName, dpi=plotDPI)
    plt.close()
    if os.path.exists(plotFileName):
        logger.info(f"Wrote state std histogram to {plotFileName}")
        return plotFileName
    logger.warning(
        f"Failed to create histogram. {plotFileName} not written."
    )
    return None


def _extractUpperTail(
    blockMeans,
    blockVars,
    q=0.50,
    binsPerUnit=10,
    minBins=50,
    maxBins=250,
    transformFunc=None,
):
    means = np.asarray(blockMeans, dtype=np.float64)
    vars_ = np.asarray(blockVars, dtype=np.float64)
    transformedBlockMeans = means if transformFunc is None else transformFunc(means)
    transformedMin = float(np.quantile(transformedBlockMeans, 0.005))
    transformedMax = float(np.quantile(transformedBlockMeans, 0.995))
    dynamicRange = transformedMax - transformedMin

    numBins = int(
        np.clip(
            np.ceil(binsPerUnit * max(dynamicRange, 1e-4)),
            minBins,
            maxBins,
        )
    )

    edges = np.linspace(
        transformedMin,
        transformedMax,
        num=numBins + 1,
        dtype=np.float64,
    )
    binIdx = np.clip(
        np.searchsorted(edges, transformedBlockMeans, side="right")
        - 1,
        0,
        numBins - 1,
    )

    outMeans = np.empty(numBins, dtype=np.float64)
    outVars = np.empty(numBins, dtype=np.float64)

    for k in range(numBins):
        mask = binIdx == k
        if not np.any(mask):
            outMeans[k] = np.nan
            outVars[k] = np.nan
            continue
        # median of means within bin
        outMeans[k] = np.median(means[mask])
        outVars[k] = np.quantile(vars_[mask], q)

    keep = np.isfinite(outMeans) & np.isfinite(outVars)
    return outMeans[keep], outVars[keep]


def getAverageLocalVarianceTrack(
    values: np.ndarray,
    stepSize: int,
    approximationWindowLengthBP: int,
    lowPassWindowLengthBP: int,
    minR: float,
    maxR: float,
    lowPassFilterType: Optional[str] = "median",
    shrinkOffset: float = 1.0,
) -> npt.NDArray[np.float32]:
    r"""A moment-based local variance estimator with autocorrelation-based shrinkage for genome-wide sample-specific noise level approximation.

    First, computes a moving average of ``values`` using a bp-length window
    ``approximationWindowLengthBP`` and a moving average of ``values**2`` over the
    same window. Their difference is used to approximate the *initial* 'local variance' before
    autocorrelation-based shrinkage. Finally, a broad/low-pass filter (``median`` or ``mean``)
    with window ``lowPassWindowLengthBP`` then smooths the variance track.

    (Retained for backward compatibility).

    :param stepSize: see :class:`countingParams`.
    :type stepSize: int
    :param approximationWindowLengthBP: Window (bp) for local mean and second-moment. See :class:`observationParams`.
    :type approximationWindowLengthBP: int
    :param lowPassWindowLengthBP: Window (bp) for the low-pass filter on the variance track. See :class:`observationParams`.
    :type lowPassWindowLengthBP: int
    :param minR: Lower bound for the returned noise level. See :class:`observationParams`.
    :type minR: float
    :param maxR: Upper bound for the returned noise level. See :class:`observationParams`.
    :type maxR: float
    :param lowPassFilterType: ``"median"`` (default) or ``"mean"``. Type of low-pass filter to use for smoothing the final noise level track. See :class:`observationParams`.
    :type lowPassFilterType: Optional[str]
    :param shrinkOffset: Offset applied to lag-1 autocorrelation when shrinking local variance estimates. See :class:`observationParams`.
    :type shrinkOffset: float
    :return: Local noise level per interval.
    :rtype: npt.NDArray[np.float32]

    :seealso: :class:`observationParams`
    """
    values = np.asarray(values, dtype=np.float32)
    windowLength = int(approximationWindowLengthBP / stepSize)
    if windowLength % 2 == 0:
        windowLength += 1

    if len(values) < 3:
        constVar = np.var(values)
        if constVar < minR:
            return np.full_like(values, minR, dtype=np.float32)
        return np.full_like(values, constVar, dtype=np.float32)

    # get local mean (simple moving average)
    localMeanTrack: npt.NDArray[np.float32] = ndimage.uniform_filter(
        values, size=windowLength, mode="nearest"
    )

    # apply V[X] ~=~ E[X^2] - (E[X])^2 locally to approximate local variance
    totalVarTrack: npt.NDArray[np.float32] = (
        ndimage.uniform_filter(
            values**2, size=windowLength, mode="nearest"
        )
        - localMeanTrack**2
    )

    np.maximum(totalVarTrack, 0.0, out=totalVarTrack)  # JIC

    noiseLevel: npt.NDArray[np.float32]
    localVarTrack: npt.NDArray[np.float32]

    if abs(shrinkOffset) < 1:
        # Aim is to shrink the local noise variance estimates
        # ...where there's evidence of structure (signal) in the data
        # ...autocorr small --> retain more of the variance estimate
        # ...autocorr large --> more shrinkage

        # shift idx +1
        valuesLag = np.roll(values, 1)
        valuesLag[0] = valuesLag[1]

        # get smooth `x_{[i]} * x_{[i-1]}` and standardize
        localMeanLag: npt.NDArray[np.float32] = (
            ndimage.uniform_filter(
                valuesLag, size=windowLength, mode="nearest"
            )
        )
        smoothProd: npt.NDArray[np.float32] = ndimage.uniform_filter(
            values * valuesLag, size=windowLength, mode="nearest"
        )
        covLag1: npt.NDArray[np.float32] = (
            smoothProd - localMeanTrack * localMeanLag
        )
        rho1: npt.NDArray[np.float32] = np.clip(
            covLag1 / (totalVarTrack + 1.0e-4),
            -1.0 + shrinkOffset,
            1 - shrinkOffset,
        )

        noiseFracEstimate: npt.NDArray[np.float32] = 1.0 - rho1**2
        localVarTrack = totalVarTrack * noiseFracEstimate

    else:
        localVarTrack = totalVarTrack

    np.maximum(localVarTrack, 0.0, out=localVarTrack)
    lpassWindowLength = int(lowPassWindowLengthBP / stepSize)
    if lpassWindowLength % 2 == 0:
        lpassWindowLength += 1

    # FFR: consider making this step optional
    if lowPassFilterType is None or (
        isinstance(lowPassFilterType, str)
        and lowPassFilterType.lower() == "median"
    ):
        noiseLevel: npt.NDArray[np.float32] = ndimage.median_filter(
            localVarTrack,
            size=lpassWindowLength,
        )
    elif (
        isinstance(lowPassFilterType, str)
        and lowPassFilterType.lower() == "mean"
    ):
        noiseLevel = ndimage.uniform_filter(
            localVarTrack,
            size=lpassWindowLength,
        )
    else:
        logger.warning(
            f"Unknown lowPassFilterType, expected `median` or `mean`, defaulting to `median`..."
        )
        noiseLevel = ndimage.median_filter(
            localVarTrack,
            size=lpassWindowLength,
        )

    return np.clip(noiseLevel, minR, maxR).astype(np.float32)


def _textplotDstatHistogram(
    vectorD: np.ndarray,
    n: int,
    enabled: bool = True,
    checkMod: Callable[[str], bool] = _checkMod,
    maxPoints: int = 10_000,
) -> None:
    if not enabled:
        return
    try:
        if not checkMod("plotext"):
            return
        import plotext as textplt

        if n > maxPoints:
            indices = np.random.choice(
                n, size=maxPoints, replace=True
            )
            dvals = vectorD[indices]
        else:
            dvals = vectorD
        iqr = np.quantile(vectorD, 0.75) - np.quantile(vectorD, 0.25)
        binWidth = (2 * iqr) / (vectorD.size ** (1 / 3))
        if binWidth > 0.0:
            bins = int(
                np.ceil(
                    (np.max(vectorD) - np.min(vectorD)) / binWidth
                )
            )
        else:
            bins = 10
        logger.info(
            "\tPlotting NIS histogram via `plotext`..."
            "(To disable these plots in the default CLI implementation, do not invoke `--verbose2` flag)"
        )
        textplt.hist(dvals, bins=bins, label="(Forward) NIS/D [:]")
        textplt.limit_size(True, True)
        textplt.show()
        textplt.clf()
    except Exception as e:
        logger.warning(f"Ignoring `textPlotDstatHistogram`:\n{e}\n")


def _textplotMeanVarianceTrend(
    blockMeans: np.ndarray,
    blockVars: np.ndarray,
    blockMeansSorted: np.ndarray,
    opt: np.ndarray,
    evalFunc: Callable = cconsenrich.cmonotonicFitEval,
    isTransformed: bool = True,
    enabled: bool = True,
    checkMod: Callable[[str], bool] = _checkMod,
    maxPoints: int = 10_000,
) -> None:
    if not enabled:
        return
    try:
        if not checkMod("plotext"):
            return
        import plotext as textplt

        n = int(blockMeans.size)
        if n > maxPoints:
            idx = np.random.choice(n, size=maxPoints, replace=True)
            x = blockMeans[idx]
            y = blockVars[idx]
        else:
            x = blockMeans
            y = blockVars

        logger.info(
            "\tPlotting mean-variance trend via `plotext`..."
            "(To disable these plots in the default CLI implementation, do not invoke `--verbose2` flag)"
        )
        textplt.scatter(x, y, label="(Block Mean, Block Var)")
        textplt.scatter(
            blockMeansSorted,
            evalFunc(opt, blockMeansSorted, isTransformed),
            label=f"`opt`: {opt}",
            color="red",
        )
        textplt.limit_size(True, True)
        textplt.show()
        textplt.clf()
    except Exception as e:
        logger.warning(
            f"Ignoring `textPlotMeanVarianceTrend`:\n{e}\n"
        )
