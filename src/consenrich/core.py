# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
import os
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Callable,
    DefaultDict,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pybedtools as bed
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage, signal
from scipy.stats.mstats import trimtail
from . import cconsenrich

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def resolveExtendBP(extendBP, bamFiles: List[str]) -> List[int]:
    numFiles = len(bamFiles)

    if isinstance(extendBP, str):
        stringValue = extendBP.replace(" ", "")
        try:
            extendBP = (
                [int(x) for x in stringValue.split(",")]
                if stringValue
                else []
            )
        except ValueError:
            raise ValueError(
                "`extendBP` string must be comma-separated values (castable to integers)"
            )
    if extendBP is None:
        return [0] * numFiles
    elif isinstance(extendBP, list):
        valuesList = [int(x) for x in extendBP]
        valuesLen = len(valuesList)
        if valuesLen == 0:
            return [0] * numFiles
        if valuesLen == 1:
            return [valuesList[0]] * numFiles
        if valuesLen == numFiles:
            return valuesList
        raise ValueError(
            f"extendBP length {valuesLen} does not match number of bamFiles {numFiles}; "
            f"provide 0, 1, or {numFiles} values."
        )
    elif isinstance(extendBP, int) or isinstance(extendBP, float):
        return [int(extendBP)] * numFiles
    raise TypeError(
        f"Invalid extendBP type: {type(extendBP).__name__}. "
        "Expecting a single number (broadcast), a list of numbers matching `bamFiles`."
    )


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

    :param deltaF: Scales the signal and variance propagation between adjacent genomic intervals. If ``< 0`` (default), determined based on stepSize:fragment-length ratio.
    :type deltaF: float
    :param minQ: Minimum process noise level (diagonal in :math:`\mathbf{Q}_{[i]}`)
        for each state variable. If `minQ < 0` (default), a value based on
        the minimum observation noise level (``observationParams.minR``) is used that
        enforces numerical stability and a worst-case balance between process and observation models
        for the given number of samples.
    :param maxQ: Maximum process noise level.
    :type minQ: float
    :param dStatAlpha: Threshold on the deviation between the data and estimated signal -- used to determine whether the process noise is scaled up.
    :type dStatAlpha: float
    :param dStatd: Constant :math:`d` in the scaling expression :math:`\sqrt{d|D_{[i]} - \alpha_D| + c}`
        that is used to up/down-scale the process noise covariance in the event of a model mismatch.
    :type dStatd: float
    :param dStatPC: Constant :math:`c` in the scaling expression :math:`\sqrt{d|D_{[i]} - \alpha_D| + c}`
        that is used to up/down-scale the process noise covariance in the event of a model mismatch.
    :type dStatPC: float
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
    scaleResidualsByP11: Optional[bool]
    adjustPmatByInnovationAC: Optional[bool]


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich.

    The observation model is used to integrate sequence alignment count
    data from the multiple input samples and account for region-and-sample-specific
    noise processes corrupting data. The observation model matrix
    :math:`\mathbf{H} \in \mathbb{R}^{m \times 2}` maps from the state dimension (2)
    to the dimension of measurements/data (:math:`m`).

    :param minR: Genome-wide lower bound for the local/sample-specific observation noise levels.
        If ``minR < 0`` (default), the minimum noise level is set based on the data as in the left
        tail of empirical noise level estimates.
    :type minR: float
    :param maxR: Genome-wide upper bound for the local/sample-specific observation noise levels
    :param numNearest: The number of nearest nearby 'sparse' features to use for local
        variance calculation. Ignored if `useALV` is True.
    :type numNearest: int
    :param localWeight: The coefficient for the local noise level (based on the local surrounding window / `numNearest` features) used in the weighted sum measuring sample-specific noise level at the current interval.
    :type localWeight: float
    :param globalWeight: The coefficient for the global noise level (based on all genomic intervals :math:`i=1\ldots n`) used in the weighted sum measuring sample-specific noise level at the current interval.
    :type globalWeight: float
    :param approximationWindowLengthBP: The length of the local variance approximation window in base pairs (BP)
        for the local variance calculation.
    :type approximationWindowLengthBP: int
    :param sparseBedFile: The path to a BED file of 'sparse' regions for the local variance calculation. For genomes with default resources in `src/consenrich/data`, this may be left as `None`,
      and a default annotation that is devoid of putative regulatory elements (ENCODE cCREs) will be used. Users can instead supply a custom BED file or set `observationParams.useALV` to `True`
      to avoid predefined annotations.
    :type sparseBedFile: str, optional
    :param noGlobal: If True, only the 'local' variances are used to approximate observation noise
        covariance :math:`\mathbf{R}_{[:, (11:mm)]}`.
    :type noGlobal: bool
    :param useALV: Whether to use average local variance (ALV) heuristic *exclusively* to approximate observation noise
        covariances per-sample, per-interval. Note that unrestricted ALV (i.e., without masking previously annotated high-signal regions) is comparatively vulnerable to inflated noise estimates in large enriched genomic domains.
    :type useALV: bool
    :param lowPassFilterType: The type of low-pass filter to use (e.g., 'median', 'mean') in the ALV calculation (:func:`consenrich.core.getAverageLocalVarianceTrack`).
    :type lowPassFilterType: Optional[str]
    :param shrinkOffset: An offset applied to local lag-1 autocorrelation, :math:`A_{[i,1]}`, such that the shrinkage factor in :func:`consenrich.core.getAverageLocalVarianceTrack`, :math:`1 - A_{[i,1]}^2`, does not reduce the local variance estimate near zero.
        Setting to >= `1` disables shrinkage.
    :type shrinkOffset: Optional[float]
    :param kappaALV: For ``minR < 0``, bounds the ratio of :math:`\frac{R_{[i,j_{\max}]}}{R_{[i,j_{\min}]}}` at each interval :math:`i=1\ldots n. Promotes numerical stability.
    """

    minR: float
    maxR: float
    useALV: bool
    useConstantNoiseLevel: bool
    noGlobal: bool
    numNearest: int
    localWeight: float
    globalWeight: float
    approximationWindowLengthBP: int
    lowPassWindowLengthBP: int
    lowPassFilterType: Optional[str]
    returnCenter: bool
    shrinkOffset: Optional[float]
    kappaALV: Optional[float]


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
    :param extendBP: A list of integers specifying the number of base pairs to extend reads for each BAM file after shifting per `offsetStr`.
        If all BAM files share the same expected frag. length, can supply a single numeric value to be broadcasted. Ignored for PE reads.
    :type extendBP: List[int]
    :param maxInsertSize: Maximum frag length/insert to consider when estimating fragment length.
    :type maxInsertSize: int
    :param pairedEndMode: If > 0, use TLEN attribute to determine span of (proper) read pairs and extend reads accordingly.
    :type pairedEndMode: int
    :param inferFragmentLength: Intended for single-end data: if > 0, the maximum correlation lag
       (avg.) between *strand-specific* read tracks is taken as the fragment length estimate and used to
       extend reads from 5'. Ignored if `pairedEndMode > 0` or `extendBP` set. This parameter is particularly
       important when targeting broader marks (e.g., ChIP-seq H3K27me3).
    :type inferFragmentLength: int
    :param countEndsOnly: If True, only the 5' read lengths contribute to counting. Overrides `inferFragmentLength` and `pairedEndMode`.
    :type countEndsOnly: Optional[bool]
    :param minMappingQuality: Minimum mapping quality (MAPQ) for reads to be counted.
    :type minMappingQuality: Optional[int]

    .. tip::

        For an overview of SAM flags, see https://broadinstitute.github.io/picard/explain-flags.html

    """

    samThreads: int
    samFlagExclude: int
    oneReadPerBin: int
    chunkSize: int
    offsetStr: Optional[str] = "0,0"
    extendBP: Optional[List[int]] = []
    maxInsertSize: Optional[int] = 1000
    pairedEndMode: Optional[int] = 0
    inferFragmentLength: Optional[int] = 0
    countEndsOnly: Optional[bool] = False
    minMappingQuality: Optional[int] = 0
    minTemplateLength: Optional[int] = -1


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
    :param sparseBedFile: A BED file with 'sparse regions' used to estimate noise levels -- ignored if `observationParams.useALV` is True. 'Sparse regions' broadly refers to genomic intervals devoid of the targeted signal, based on prior annotations.
      Users may supply a custom BED file and/or set `observationParams.useALV` to `True` to avoid relying on predefined annotations.
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
    :param applySqrt: If true (default), :math:`\sqrt{x}` applied to counts :math:`x` for each supplied BAM file.
    :type applySqrt: bool, optional
    :param noTransform: Disable all transformations.
    :type noTransform: bool, optional
    :param rescaleToTreatmentCoverage: Deprecated: no effect.
    :type rescaleToTreatmentCoverage: bool, optional
    :param trimLeftTail: If > 0, quantile of scaled counts to trim from the left tail before computing transformations.

    :seealso: :ref:`calibration` for details on transformations and their effects on interpretation of state estimates and uncertainty quantification.
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
    rescaleToTreatmentCoverage: Optional[bool] = False  # deprecated
    normMethod: Optional[str] = "EGS"
    trimLeftTail: Optional[float] = 0.0


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
    :param methodFDR: Method for genome-wide multiple hypothesis testing correction. Can use Benjamini-Hochberg ('bh') or the more conservative Benjamini-Yekutieli ('by') to account for arbitrary dependencies between tests.
    :type methodFDR: str
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
    randSeed: Optional[int] = 42
    eps: Optional[float] = 1.0e-2
    autoLengthQuantile: Optional[float] = 0.90
    methodFDR: Optional[str] = "bh"


class outputParams(NamedTuple):
    r"""Parameters related to output files.

    :param convertToBigWig: If True, output bedGraph files are converted to bigWig format.
    :type convertToBigWig: bool
    :param roundDigits: Number of decimal places to round output values (bedGraph)
    :type roundDigits: int
    :param writeResiduals: If True, write to a separate bedGraph the pointwise avg. of precision-weighted residuals at each interval. These may be interpreted as
        a measure of model mismatch. Where these quantities are larger (+-), there may be more unexplained deviation between the data and fitted model.
    :type writeResiduals: bool
    :param writeMuncTrace: If True, write to a separate bedGraph :math:`\sqrt{\frac{\textsf{Trace}\left(\mathbf{R}_{[i]}\right)}{m}}` -- that is, square root of the 'average' observation noise level at each interval :math:`i=1\ldots n`, where :math:`m` is the number of samples/tracks.
    :type writeMuncTrace: bool
    :param writeStateStd: If True, write to a separate bedGraph the estimated pointwise uncertainty in the primary state, :math:`\sqrt{\widetilde{P}_{i,(11)}}`, on a scale comparable to the estimated signal.
    :type writeStateStd: bool
    """

    convertToBigWig: bool
    roundDigits: int
    writeResiduals: bool
    writeMuncTrace: bool
    writeStateStd: bool


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
    extendBP: Optional[List[int]] = None,
    maxInsertSize: Optional[int] = 1000,
    pairedEndMode: Optional[int] = 0,
    inferFragmentLength: Optional[int] = 0,
    countEndsOnly: Optional[bool] = False,
    minMappingQuality: Optional[int] = 0,
    minTemplateLength: Optional[int] = -1,
    trimLeftTail: Optional[float] = 0.0,
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
    :param extendBP: See :class:`samParams`.
    :type extendBP: int
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

    extendBP = resolveExtendBP(extendBP or [], bamFiles)
    offsetStr = ((str(offsetStr) or "0,0").replace(" ", "")).split(
        ","
    )
    numIntervals = ((end - start) + stepSize - 1) // stepSize
    counts = np.empty((len(bamFiles), numIntervals), dtype=np.float32)

    if isinstance(countEndsOnly, bool) and countEndsOnly:
        # note: setting this option ignores inferFragmentLength, pairedEndMode
        inferFragmentLength = 0
        pairedEndMode = 0

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
            extendBP[j],
            maxInsertSize,
            pairedEndMode,
            inferFragmentLength,
            minMappingQuality,
            minTemplateLength,
        )

        counts[j, :] = arr
        if trimLeftTail > 0.0:
            trimtail(counts[j, :], trimLeftTail, tail="left")
        counts[j, :] *= np.float32(scaleFactors[j])
        if applyAsinh:
            np.asinh(counts[j, :], out=counts[j, :])
        elif applyLog:
            np.log1p(counts[j, :], out=counts[j, :])
        elif applySqrt:
            np.sqrt(counts[j, :], out=counts[j, :])
    return counts


def getAverageLocalVarianceTrack(
    values: np.ndarray,
    stepSize: int,
    approximationWindowLengthBP: int,
    lowPassWindowLengthBP: int,
    minR: float,
    maxR: float,
    lowPassFilterType: Optional[str] = "median",
    shrinkOffset: float = 0.5,
) -> npt.NDArray[np.float32]:
    r"""A moment-based local variance estimator with autocorrelation-based shrinkage for genome-wide sample-specific noise level approximation.

    First, computes a moving average of ``values`` using a bp-length window
    ``approximationWindowLengthBP`` and a moving average of ``values**2`` over the
    same window. Their difference is used to approximate the *initial* 'local variance' before
    autocorrelation-based shrinkage. Finally, a broad/low-pass filter (``median`` or ``mean``)
    with window ``lowPassWindowLengthBP`` then smooths the variance track.

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
    initMatrixQ[0, 1] = offDiagQ
    initMatrixQ[1, 0] = offDiagQ
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
    adjustPmatByInnovationAC: Optional[bool] = False,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    r"""Run consenrich on a contiguous segment (e.g. a chromosome) of read-density-based data.
    Completes the forward and backward passes given data and approximated observation noise
    covariance matrices :math:`\mathbf{R}_{[1:n, (11:mm)]}`.

    This is the primary function implementing the core Consenrich algorithm. Users requiring specialized
    preprocessing may prefer to call this function programmatically on their own preprocessed data rather
    than using the command-line interface.


    :param matrixData: Read density data for a single chromosome or general contiguous segment,
      possibly preprocessed. Two-dimensional array of shape :math:`m \times n` where :math:`m`
      is the number of samples/tracks and :math:`n` the number of genomic intervals.
    :type matrixData: np.ndarray
    :param matrixMunc: Uncertainty estimates for the read coverage data.
        Two-dimensional array of shape :math:`m \times n` where :math:`m` is the number of samples/tracks
        and :math:`n` the number of genomic intervals. See :func:`getMuncTrack`.
    :type matrixMunc: np.ndarray
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
    :param residualCovarInversionFunc: Callable function to invert the observation covariance matrix :math:`\mathbf{E}_{[i]}`.
        If None, defaults to :func:`cconsenrich.cinvertMatrixE`.
    :type residualCovarInversionFunc: Optional[Callable]
    :param adjustProcessNoiseFunc: Function to adjust the process noise covariance matrix :math:`\mathbf{Q}_{[i]}`.
        If None, defaults to :func:`cconsenrich.updateProcessNoiseCovariance`.
    :type adjustProcessNoiseFunc: Optional[Callable]
    :param adjustPmatByInnovationAC: See :class:`processParams`.
    :type adjustPmatByInnovationAC: Optional[bool]
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
        # ideally, we don't get here, but JIC
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

    inflatedQ: bool = False
    dStat: float = np.float32(0.0)
    countAdjustments: int = 0

    IKH: np.ndarray = np.zeros(shape=(2, 2), dtype=np.float32)
    matrixEInverse: np.ndarray = np.zeros(
        shape=(m, m), dtype=np.float32
    )
    matrixF: np.ndarray = constructMatrixF(deltaF)
    matrixQ: np.ndarray = constructMatrixQ(minQ, offDiagQ=offDiagQ)
    matrixQCopy: np.ndarray = matrixQ.copy()
    matrixP: np.ndarray = np.eye(2, dtype=np.float32) * np.float32(
        stateCovarInit
    )
    matrixH: np.ndarray = constructMatrixH(
        m, coefficients=coefficientsH
    )
    matrixK: np.ndarray = np.zeros((2, m), dtype=np.float32)
    vectorX: np.ndarray = np.array([stateInit, 0.0], dtype=np.float32)
    vectorY: np.ndarray = np.zeros(m, dtype=np.float32)
    matrixI2: np.ndarray = np.eye(2, dtype=np.float32)

    if residualCovarInversionFunc is None:
        residualCovarInversionFunc = cconsenrich.cinvertMatrixE
    if adjustProcessNoiseFunc is None:
        adjustProcessNoiseFunc = (
            cconsenrich.updateProcessNoiseCovariance
        )

    innovationProd = np.zeros(m, dtype=np.float64)
    innovSq = np.zeros(m, dtype=np.float64)
    innovPrev = np.zeros(m, dtype=np.float64)
    innovSum = np.zeros(m, dtype=np.float64)
    innovFirst = np.zeros(m, dtype=np.float64)
    innovLast = np.zeros(m, dtype=np.float64)
    innov = np.zeros(m, dtype=np.float64)
    haveInnovPrev = False

    # ==========================
    # forward: 0,1,2,...,n-1
    # ==========================
    stateForward = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, 2),
    )
    stateCovarForward = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, 2, 2),
    )
    pNoiseForward = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, 2, 2),
    )
    progressIter = max(1, progressIter)
    avgDstat: float = 0.0
    for i in range(n):
        if i % progressIter == 0:
            logger.info(f"\tForward pass interval: {i + 1}/{n}")
        vectorZ = matrixData[:, i]
        vectorX = matrixF @ vectorX
        matrixP = matrixF @ matrixP @ matrixF.T + matrixQ
        vectorY = vectorZ - (matrixH @ vectorX)
        if adjustPmatByInnovationAC:
            innov = vectorY.astype(np.float64)

            if i == 0:
                innovFirst[:] = innov
            innovSum += innov
            innovLast[:] = innov
            if haveInnovPrev:
                innovationProd += innovPrev * innov
            innovSq += innov * innov
            innovPrev = innov
            haveInnovPrev = True

        matrixEInverse = residualCovarInversionFunc(
            matrixMunc[:, i],
            np.float32(matrixP[0, 0]),
        )
        Einv_diag = np.diag(matrixEInverse)
        dStat = np.median((vectorY**2) * Einv_diag)
        avgDstat += float(dStat)
        countAdjustments = countAdjustments + int(dStat > dStatAlpha)
        matrixQ, inflatedQ = adjustProcessNoiseFunc(
            matrixQ,
            matrixQCopy,
            dStat,
            dStatAlpha,
            dStatd,
            dStatPC,
            inflatedQ,
            maxQ,
            minQ,
        )
        matrixK = (matrixP @ matrixH.T) @ matrixEInverse
        IKH = matrixI2 - (matrixK @ matrixH)

        vectorX = vectorX + (matrixK @ vectorY)
        matrixP = (IKH) @ matrixP @ (IKH).T + (
            matrixK * matrixMunc[:, i]
        ) @ matrixK.T
        stateForward[i] = vectorX.astype(np.float32)
        stateCovarForward[i] = matrixP.astype(np.float32)
        pNoiseForward[i] = matrixQ.astype(np.float32)

        if i % chunkSize == 0 and i > 0:
            stateForward.flush()
            stateCovarForward.flush()
            pNoiseForward.flush()

    stateForward.flush()
    stateCovarForward.flush()
    pNoiseForward.flush()
    stateForwardArr = stateForward
    stateCovarForwardArr = stateCovarForward
    pNoiseForwardArr = pNoiseForward
    avgDstat /= n

    logger.info(f"Average D_[i] statistic over `n` intervals: {avgDstat:.3f}")
    logger.info(
        f"`D_[i] > α_D` triggered adjustments to Q_[i] at [{round(((1.0 * countAdjustments) / n) * 100.0, 4)}%] of intervals"
    )

    # ==========================
    # backward: n,n-1,n-2,...,0
    # ==========================
    stateSmoothed = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, 2),
    )
    stateCovarSmoothed = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, 2, 2),
    )
    postFitResiduals = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, m),
    )

    stateSmoothed[-1] = np.float32(stateForwardArr[-1])
    stateCovarSmoothed[-1] = np.float32(stateCovarForwardArr[-1])
    postFitResiduals[-1] = np.float32(
        matrixData[:, -1] - (matrixH @ stateSmoothed[-1])
    )

    for k in range(n - 2, -1, -1):
        if k % progressIter == 0:
            logger.info(f"\tBackward pass interval: {k + 1}/{n}")
        forwardStatePosterior = stateForwardArr[k]
        forwardCovariancePosterior = stateCovarForwardArr[k]
        backwardInitialState = matrixF @ forwardStatePosterior
        backwardInitialCovariance = (
            matrixF @ forwardCovariancePosterior @ matrixF.T
            + pNoiseForwardArr[k + 1]
        )

        smootherGain = np.linalg.solve(
            backwardInitialCovariance.T,
            (forwardCovariancePosterior @ matrixF.T).T,
        ).T
        stateSmoothed[k] = (
            forwardStatePosterior
            + smootherGain
            @ (stateSmoothed[k + 1] - backwardInitialState)
        ).astype(np.float32)

        stateCovarSmoothed[k] = (
            forwardCovariancePosterior
            + smootherGain
            @ (stateCovarSmoothed[k + 1] - backwardInitialCovariance)
            @ smootherGain.T
        ).astype(np.float32)
        postFitResiduals[k] = np.float32(
            matrixData[:, k] - matrixH @ stateSmoothed[k]
        )

        if k % chunkSize == 0 and k > 0:
            stateSmoothed.flush()
            stateCovarSmoothed.flush()
            postFitResiduals.flush()

    if adjustPmatByInnovationAC and n > 1:
        rhoValues = []
        biasTol = 1e-3
        for j in range(m):
            totalSquaredInnovation = float(innovSq[j])
            totalInnovation = float(innovSum[j])
            if totalSquaredInnovation <= 0.0:
                continue
            meanInnovation = totalInnovation / n
            if abs(meanInnovation) < biasTol:
                rhoValues.append(
                    float(innovationProd[j]) / totalSquaredInnovation
                )
                continue
            innovationVariance = (
                totalSquaredInnovation
                - (totalInnovation * totalInnovation) / n
            )
            if innovationVariance <= 0.0:
                continue
            prodLag1 = float(innovationProd[j])
            firstInnovation = float(innovFirst[j])
            lastInnovation = float(innovLast[j])

            centeredLag1Sum = (
                prodLag1
                - meanInnovation
                * (
                    (totalInnovation - firstInnovation)
                    + (totalInnovation - lastInnovation)
                )
                + meanInnovation * meanInnovation * (n - 1.0)
            )

            rhoValues.append(centeredLag1Sum / innovationVariance)

        if rhoValues:
            rhoHat = float(np.median(rhoValues))
            varInflation = 1.0 + max(0.0, 2.0 * rhoHat)
            varInflation = float(np.clip(varInflation, 1.0, 10.0))
            stateCovarSmoothed[:] = stateCovarSmoothed[
                :
            ] * np.float32(varInflation)
            stateCovarSmoothed.flush()

    stateSmoothed.flush()
    stateCovarSmoothed.flush()
    postFitResiduals.flush()

    if boundState:
        stateSmoothed[:, 0] = np.clip(
            stateSmoothed[:, 0], stateLowerBound, stateUpperBound
        ).astype(np.float32)

    return (
        stateSmoothed[:],
        stateCovarSmoothed[:],
        postFitResiduals[:],
    )


def getPrimaryState(
    stateVectors: np.ndarray,
    roundPrecision: int = 4,
) -> npt.NDArray[np.float32]:
    r"""Get the primary state estimate from each vector after running Consenrich.

    :param stateVectors: State vectors from :func:`runConsenrich`.
    :type stateVectors: npt.NDArray[np.float32]
    :return: A one-dimensional numpy array of the primary state estimates.
    :rtype: npt.NDArray[np.float32]
    """
    out_ = np.ascontiguousarray(stateVectors[:, 0], dtype=np.float32)
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
    matrixMunc: np.ndarray,
    roundPrecision: int = 4,
    stateCovarSmoothed: Optional[np.ndarray] = None,
) -> npt.NDArray[np.float32]:
    r"""Get a one-dimensional precision-weighted array residuals after running Consenrich.

    Post-fit residuals weighted by the inverse of the observation noise covariance and primary state uncertainty.

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

    postFitResiduals_CContig = np.ascontiguousarray(
        postFitResiduals, dtype=np.float32
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
    stepSize: int,
    rowValues: np.ndarray,
    minR: float,
    maxR: float,
    useALV: bool,
    useConstantNoiseLevel: bool,
    noGlobal: bool,
    localWeight: float,
    globalWeight: float,
    approximationWindowLengthBP: int,
    lowPassWindowLengthBP: int,
    returnCenter: bool,
    sparseMap: Optional[dict[int, int]] = None,
    lowPassFilterType: Optional[str] = "median",
    shrinkOffset: float = 0.5,
) -> npt.NDArray[np.float32]:
    r"""Get observation noise variance :math:`R_{[:,jj]}` for the sample :math:`j`.

    Combines a local ALV estimate (see :func:`getAverageLocalVarianceTrack`) with an
    optional global component. If ``useALV`` is True, *only* the ALV is used. If
    ``useConstantNoiseLevel`` is True, a constant track set to the global mean is used.
    When a ``sparseMap`` is provided, local values are aggregated over nearby 'sparse'
    regions before mixing with the global component.

    :param chromosome: Tracks are approximated for this chromosome.
    :type chromosome: str
    :param intervals: Genomic intervals for which to compute the noise track.
    :param stepSize: See :class:`countingParams`.
    :type stepSize: int
    :param rowValues: Read-density-based values for the sample :math:`j` at the genomic intervals :math:`i=1,2,\ldots,n`.
    :type rowValues: np.ndarray
    :param minR: See :class:`observationParams`.
    :type minR: float
    :param maxR: See :class:`observationParams`.
    :type maxR: float
    :param useALV: See :class:`observationParams`.
    :type useALV: bool
    :param useConstantNoiseLevel: See :class:`observationParams`.
    :type useConstantNoiseLevel: bool
    :param noGlobal: See :class:`observationParams`.
    :type noGlobal: bool
    :param localWeight: See :class:`observationParams`.
    :type localWeight: float
    :param globalWeight: See :class:`observationParams`.
    :type globalWeight: float
    :param approximationWindowLengthBP: See :class:`observationParams` and/or :func:`getAverageLocalVarianceTrack`.
    :type approximationWindowLengthBP: int
    :param lowPassWindowLengthBP: See :class:`observationParams` and/or :func:`getAverageLocalVarianceTrack`.
    :type lowPassWindowLengthBP: int
    :param sparseMap: Optional mapping (dictionary) of interval indices to the nearest sparse regions. See :func:`getSparseMap`.
    :type sparseMap: Optional[dict[int, int]]
    :param lowPassFilterType: The type of low-pass filter to use in average local variance track (e.g., 'median', 'mean').
    :type lowPassFilterType: Optional[str]
    :param shrinkOffset: See :func:`getAverageLocalVarianceTrack`.
    :type shrinkOffset: float
    :return: A one-dimensional numpy array of the observation noise track for the sample :math:`j`.
    :rtype: npt.NDArray[np.float32]
    """

    # FFR: we should consider whether to apply bounds only after mixing local/global
    trackALV = getAverageLocalVarianceTrack(
        rowValues,
        stepSize,
        approximationWindowLengthBP,
        lowPassWindowLengthBP,
        minR,
        maxR,
        lowPassFilterType,
        shrinkOffset=shrinkOffset,
    ).astype(np.float32)

    globalNoise: float = np.float32(np.mean(trackALV))
    if noGlobal or globalWeight == 0 or useALV:
        return np.clip(trackALV, minR, maxR).astype(np.float32)

    if (
        useConstantNoiseLevel
        or localWeight == 0
        and sparseMap is None
    ):
        return np.clip(
            globalNoise * np.ones_like(rowValues), minR, maxR
        ).astype(np.float32)

    if sparseMap is not None:
        trackALV = cconsenrich.cSparseAvg(trackALV, sparseMap)

    return np.clip(
        trackALV * localWeight + np.mean(trackALV) * globalWeight,
        minR,
        maxR,
    ).astype(np.float32)


def sparseIntersection(
    chromosome: str, intervals: np.ndarray, sparseBedFile: str
) -> npt.NDArray[np.int64]:
    r"""Returns intervals in the chromosome that overlap with the sparse features.

    Not relevant if `observationParams.useALV` is True.

    :param chromosome: The chromosome name.
    :type chromosome: str
    :param intervals: The genomic intervals to consider.
    :type intervals: np.ndarray
    :param sparseBedFile: Path to the sparse BED file.
    :type sparseBedFile: str
    :return: A numpy array of start positions of the sparse features that overlap with the intervals
    :rtype: np.ndarray[Tuple[Any], np.dtype[Any]]
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
    chromosome: str,
    start: int,
    end: int,
    stepSize: int,
    fragmentLengths: Optional[List[int]] = None,
    bamFiles: Optional[List[str]] = None,
    fallBackFragmentLength: int = 147,
    randomSeed: int = 42,
) -> float:
    r"""(Experimental) Set `deltaF` as the ratio intervalLength:fragmentLength.

    Computes average fragment length across samples and sets `deltaF = stepSize / avgFragmentLength`.
    Assuming fragments contribute +1 to each genomic interval they overlap, a small intervalLength:fragmentLength
    motivates a small `deltaF` (i.e., less state change between intervals).

    :param chromosome: Chromosome name.
    :type chromosome: str
    :param start: Start position of a region (e.g., chromosome)
    :type start: int
    :param end: End position of a region (e.g., chromosome)
    :type end: int
    :param stepSize: Length of genomic intervals/bins. See :class:`countingParams`.
    :type stepSize: int
    :param bamFiles: List of sorted/indexed BAM files to estimate fragment lengths from.
    :type bamFiles: Optional[List[str]]
    """

    avgFragmentLength: float = 0.0
    if (
        fragmentLengths is not None
        and len(fragmentLengths) > 0
        and all(isinstance(x, (int, float)) for x in fragmentLengths)
    ):
        avgFragmentLength = np.mean(fragmentLengths)
    elif bamFiles is not None and len(bamFiles) > 0:
        for bamFile in bamFiles:
            fLen = cconsenrich.cgetFragmentLength(
                bamFile,
                chromosome,
                start,
                end,
                fallBack=fallBackFragmentLength,
                randSeed=randomSeed,
            )
            avgFragmentLength += fLen
            logger.info(
                f"Estimated fragment length for {bamFile}: {fLen} bp"
            )
        avgFragmentLength /= len(bamFiles)
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
