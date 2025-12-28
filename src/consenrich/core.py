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
from scipy import ndimage, signal, optimize, stats
from tqdm import tqdm
from . import cconsenrich
from . import __version__

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
    dStatUseMean: Optional[bool]  # deprecated
    scaleResidualsByP11: Optional[bool]


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich.

    The observation model is used to integrate measured sequence alignment count-based
    data from the multiple input samples while accounting for region- and sample-specific
    uncertainty arising from biological and/or technical sources of noise.


    :param minR: Genome-wide lower bound for sample-specific measurement uncertainty levels.
    :type minR: float
    :param maxR: Genome-wide upper bound for the sample-specific measurement uncertainty levels.
    :param numNearest: Optional. The number of nearest 'sparse' features in ``consenrich.core.genomeParams.sparseBedFile``
      to use at each interval during the ALV/local measurement uncertainty calculation. See :func:`consenrich.core.getMuncTrack`
    :type numNearest: int
    """

    minR: float | None
    maxR: float | None
    numNearest: int | None


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
    r"""Parameters related to counting aligned reads

    :param stepSize: Length (bp) of each genomic interval :math:`i=1\ldots n` used to index/partition contigs.
        In the default implementation, 50bp is used, and this is generally robust. Sequencing depth and expected feature size may warrant
        tuning, however. For very broad marks and/or low tag counts, consider increasing to 75, 100bp, etc.
    :type stepSize: int
    :param backgroundWindowSizeBP: Size of windows (bp) used for estimating+interpolating between-block background estimates.
        Per-interval autocorrelation in the background estimates grows roughly as :math:`\frac{stepSize}{\textsf{backgroundWindowBP}}`.
        Note that this parameter is inconsequential for datasets with treatment+control samples, where background is estimated from control inputs.
        See :func:`consenrich.cconsenrich.carsinhRatio`.
    :param fragmentLengths: List of fragment lengths (bp) to use for extending reads from 5' ends when counting single-end data.
    :type fragmentLengths: List[int], optional
    :param fragmentLengthsControl: List of fragment lengths (bp) to use for extending reads from 5' ends when counting single-end with control data.
    :type fragmentLengthsControl: List[int], optional
    :param useTreatmentFragmentLengths: If True, use fragment lengths estimated from treatment BAM files for control BAM files, too.
    :type useTreatmentFragmentLengths: bool, optional
    :param fixControl: If True, treatment samples are not upscaled, and control samples are not downscaled.


    .. admonition:: Treatment vs. Control Fragment Lengths in Single-End Data
    :class: tip
    :collapsible: closed

      For single-end data, cross-correlation-based estimates for fragment length
      in control inputs can be biased due to a comparative lack of structure in
      strand-specific coverage tracks.

      This can create artifacts during counting, so it is common to use the estimated treatment
      fragment length for both treatment and control samples. The argument
      ``observationParams.useTreatmentFragmentLengths`` enables this behavior.

    """

    stepSize: int | None
    backgroundWindowSizeBP: int | None
    scaleFactors: List[float] | None
    scaleFactorsControl: List[float] | None
    normMethod: str | None
    fragmentLengths: List[int] | None
    fragmentLengthsControl: List[int] | None
    useTreatmentFragmentLengths: bool | None
    fixControl: bool | None


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
    maxInsertSize: Optional[int] = 1000,
    pairedEndMode: Optional[int] = 0,
    inferFragmentLength: Optional[int] = 0,
    countEndsOnly: Optional[bool] = False,
    minMappingQuality: Optional[int] = 0,
    minTemplateLength: Optional[int] = -1,
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

    if len(readLengths) != len(bamFiles) or len(scaleFactors) != len(bamFiles):
        raise ValueError(
            "readLengths and scaleFactors must match bamFiles length"
        )

    offsetStr = ((str(offsetStr) or "0,0").replace(" ", "")).split(",")

    numIntervals = ((end - start - 1) // stepSize) + 1
    counts = np.empty((len(bamFiles), numIntervals), dtype=np.float32)

    if pairedEndMode:
        # paired end --> use TLEN attribute for each properly paired read
        fragmentLengths = [0] * len(bamFiles)
        inferFragmentLength = 0

    if not pairedEndMode and (
        fragmentLengths is None or len(fragmentLengths) == 0
    ):
        # single-end without user-supplied fragment length -->
        # ... estimate fragment lengths as the peak lag_k in
        # ... cross-correlation(forwardReadsTrack,backwardReadsTrack, lag_k)
        inferFragmentLength = 1
        fragmentLengths = [-1] * len(bamFiles)

    if isinstance(countEndsOnly, bool) and countEndsOnly:
        # No fragment length extension, just count 5' ends
        # ... May be preferred for high-resolution analyses in deeply-sequenced HTS
        # ...  data but note the drift in interpretation for processParams.deltaF,
        # ... consider setting deltaF \propto (readLength / stepSize)
        inferFragmentLength = 0
        pairedEndMode = 0
        fragmentLengths = [0] * len(bamFiles)

    for j, bam in tqdm(
        enumerate(bamFiles),
        desc="Building count matrix",
        unit=" bam files",
        total=len(bamFiles),
    ):
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
        np.multiply(
            counts[j, :],
            np.float32(scaleFactors[j]),
            out=counts[j, :],
        )
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
    """

    if useIdentity > 0.0:
        return np.eye(2, dtype=np.float32) * np.float32(useIdentity)

    Q = np.empty((2, 2), dtype=np.float32)
    Q[0, 0] = np.float32(minDiagQ if Q00 is None else Q00)
    Q[1, 1] = np.float32(minDiagQ if Q11 is None else Q11)

    if Q01 is not None and Q10 is None:
        Q10 = Q01
    elif Q10 is not None and Q01 is None:
        Q01 = Q10

    Q[0, 1] = np.float32(offDiagQ if Q01 is None else Q01)
    Q[1, 0] = np.float32(offDiagQ if Q10 is None else Q10)

    if not np.allclose(Q[0, 1], Q[1, 0], rtol=0.0, atol=1e-4):
        raise ValueError(f"Matrix is not symmetric: Q=\n{Q}")

    # raise if poorly-conditioned/non-SPD
    try:
        np.linalg.cholesky(Q.astype(np.float64, copy=False) + tol * np.eye(2))
    except Exception as ex:
        raise ValueError(
            f"Process noise covariance Q is not positive definite:\n{Q}"
        ) from ex
    return Q


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
    covarClip: float = 3.0,
    projectStateDuringFiltering: bool = False,
    pad: float = 1.0e-3,
    calibration_kwargs: Optional[dict[str, Any]] = None,
    disableCalibration: bool = False,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    r"""Run consenrich on a contiguous segment (e.g. a chromosome) of read-density-based data from multiple samples.
    Completes the forward and backward passes given data :math:`\mathbf{Z}^{m \times n}` and
    corresponding uncertainty tracks :math:`\mathbf{R}_{[1:n, (11:mm)]}` (see :func:`getMuncTrack`).

    :seealso: :class:`processParams`, :class:`observationParams`, :class:`inputParams`, :class:`outputParams`, :class:`countingParams`
    """
    pad_ = np.float32(pad)
    matrixData = np.ascontiguousarray(matrixData, dtype=np.float32)
    matrixMunc = np.ascontiguousarray(matrixMunc, dtype=np.float32)
    if calibration_kwargs is None:
        calibration_kwargs = {}
    # -------
    # check edge cases
    if matrixData.ndim == 1:
        matrixData = matrixData[None, :]
    elif matrixData.ndim != 2:
        raise ValueError(
            f"`matrixData` must be 1D or 2D (got ndim = {matrixData.ndim})"
        )
    if matrixMunc.ndim == 1:
        matrixMunc = matrixMunc[None, :]
    elif matrixMunc.ndim != 2:
        raise ValueError(
            f"`matrixMunc` must be 1D or 2D (got ndim = {matrixMunc.ndim})"
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
            f"`matrixData` and `matrixMunc` span very few genomic intervals (n={n})...is this correct?"
        )

    if chunkSize < 1:
        logger.warning(f"`chunkSize` must be positive, setting to 1000000")
        chunkSize = 1_000_000

    if chunkSize > n:
        logger.warning(
            f"`chunkSize` of {chunkSize} is greater than the number of intervals (n={n}), setting to {n}"
        )
        chunkSize = n

    # -------
    vectorD = np.zeros(n, dtype=np.float32)
    countAdjustments: int = 0
    LN2: np.float32 = np.log(2.0, dtype=np.float32)

    matrixF: np.ndarray = constructMatrixF(deltaF)
    matrixQ0: np.ndarray = constructMatrixQ(
        minQ,
        offDiagQ=offDiagQ,
    ).astype(np.float32, copy=False)

    with TemporaryDirectory() as tempDir_:
        stateForwardPathMM = os.path.join(tempDir_, "stateForward.dat")
        stateCovarForwardPathMM = os.path.join(
            tempDir_, "stateCovarForward.dat"
        )
        pNoiseForwardPathMM = os.path.join(tempDir_, "pNoiseForward.dat")
        stateBackwardPathMM = os.path.join(tempDir_, "stateSmoothed.dat")
        stateCovarBackwardPathMM = os.path.join(
            tempDir_, "stateCovarSmoothed.dat"
        )
        postFitResidualsPathMM = os.path.join(tempDir_, "postFitResiduals.dat")

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

        fwdPassArgs = dict(
            matrixData=matrixData,
            matrixMunc=matrixMunc,
            matrixF=matrixF,
            matrixQCopy=matrixQ0,
            dStatd=float(dStatd),
            dStatPC=float(dStatPC),
            maxQ=float(maxQ),
            minQ=float(minQ),
            stateInit=float(stateInit),
            stateCovarInit=float(stateCovarInit),
            covarClip=float(covarClip),
            pad=float(pad_),
            projectStateDuringFiltering=bool(projectStateDuringFiltering),
            stateLowerBound=float(stateLowerBound),
            stateUpperBound=float(stateUpperBound),
            chunkSize=int(chunkSize),
            vectorD=vectorD,
            progressIter=int(progressIter),
        )

        def _forwardPass(
            isInitialPass: bool = False,
            returnNLL_: bool = False,
            storeNLLInD_: bool = False,
            stateForwardOut=None,
            stateCovarForwardOut=None,
            pNoiseForwardOut=None,
            intervalToBlockMapOut=None,
            blockGradLogScalesOut=None,
            blockGradCountOut=None,
        ):
            nonlocal vectorD, countAdjustments

            matrixQWork = matrixQ0.copy()

            if blockGradLogScalesOut is not None:
                blockGradLogScalesOut.fill(np.float32(0.0))
            if blockGradCountOut is not None:
                blockGradCountOut.fill(np.float32(0.0))

            progressBar = None
            if (
                (not isInitialPass)
                and (progressIter is not None)
                and (progressIter > 0)
            ):
                progressBar = tqdm(total=n, unit=" intervals ")

            try:
                out = cconsenrich.cforwardPass(
                    **fwdPassArgs,
                    matrixQ=matrixQWork,
                    dStatAlpha=float(1.0e6 if isInitialPass else dStatAlpha),
                    stateForward=stateForwardOut,
                    stateCovarForward=stateCovarForwardOut,
                    pNoiseForward=pNoiseForwardOut,
                    progressBar=progressBar,
                    returnNLL=bool(returnNLL_),
                    storeNLLInD=bool(storeNLLInD_),
                    intervalToBlockMap=intervalToBlockMapOut,
                    blockGradLogScale=blockGradLogScalesOut,
                    blockGradCount=blockGradCountOut,
                )
            finally:
                if progressBar is not None:
                    progressBar.close()

            if returnNLL_:
                phiHatOut, countAdjustmentsOut, vectorDOut, NLLOut = out
                vectorD = vectorDOut
                countAdjustments = int(countAdjustmentsOut)
                fwdPassArgs["vectorD"] = vectorD
                return (
                    float(phiHatOut),
                    int(countAdjustmentsOut),
                    vectorD,
                    float(NLLOut),
                )

            phiHatOut, countAdjustmentsOut, vectorDOut = out
            vectorD = vectorDOut
            countAdjustments = int(countAdjustmentsOut)
            fwdPassArgs["vectorD"] = vectorD
            return float(phiHatOut), int(countAdjustmentsOut), vectorD

        if not disableCalibration:
            initialMuncBaseline = matrixMunc.copy()
            calibration_maxIters = int(
                calibration_kwargs.get("calibration_maxIters", 25)
            )
            calibration_numTotalBlocks = int(
                calibration_kwargs.get(
                    "calibration_numTotalBlocks",
                    max(min(50, n // 100), 1),
                )
            )
            calibration_activeSetLogStepSize = np.float32(
                calibration_kwargs.get("calibration_activeSetLogStepSize", LN2)
            )
            calibration_activeSetMaxLogStep = np.float32(
                calibration_kwargs.get("calibration_activeSetMaxLogStep", LN2)
            )
            calibration_BGDMaxBacktracks = int(
                calibration_kwargs.get("calibration_BGDMaxBacktracks", 5)
            )
            calibration_BGDBacktrackFactor = np.float32(
                calibration_kwargs.get("calibration_BGDBacktrackFactor", 0.5)
            )
            calibration_earlyStopZ = np.float32(
                calibration_kwargs.get("calibration_earlyStopZ", 1.0)
            )
            calibration_earlyStopCriteria = int(
                calibration_kwargs.get(
                    "calibration_earlyStopCriteria",
                    2,
                )
            )
            calibration_earlyStopMinValidIntervals = int(
                calibration_kwargs.get(
                    "calibration_earlyStopMinValidIntervals",
                    max(n // 20, 256),
                )
            )

            earlyStopNoSigIters = 0
            prevAcceptedIntervalNLL = None

            # size of the active set (those being updated in each iter) << total blocks
            activeSetSize = (calibration_numTotalBlocks // 10) + 1
            numIntervalsPerBlock = int(np.ceil(n / calibration_numTotalBlocks))

            # map intervals to their respective blocks through intervalIdx // numIntervalsPerBlock
            intervalToBlockMap = (
                np.arange(n, dtype=np.int32) // numIntervalsPerBlock
            ).astype(np.int32)

            # depending on divisibility, remaining intervals mapped to last block
            intervalToBlockMap[
                intervalToBlockMap >= calibration_numTotalBlocks
            ] = calibration_numTotalBlocks - 1

            # `lowerUpdateLimit` and `upperUpdateLimit` control scale of updates
            # ... Since initial R[:,:] is not random as a 'prior' (getMuncTrack) updates are capped at 10x
            lowerUpdateLimit = np.float32(0.10)
            upperUpdateLimit = np.float32(10.0)

            # 'dispersion factors' scale within-block uncertainty levels during calibration
            BlockDispersionFactors = np.ones(
                calibration_numTotalBlocks, dtype=np.float32
            )
            bestLoss = 1e8
            bestBlockDispersionFactors = BlockDispersionFactors.copy()

            # define block-level gradients as the sum
            # ... of per-interval NLL gradients within each block
            blockGradLogScales = np.zeros(
                calibration_numTotalBlocks, dtype=np.float32
            )
            blockGradCount = np.zeros(
                calibration_numTotalBlocks, dtype=np.float32
            )

            # Each interval within a block gets the same scale factor (`BlockDispersionFactors[b]`)
            intervalDispersionFactors = BlockDispersionFactors[
                intervalToBlockMap
            ]
            matrixMunc[:] = (
                initialMuncBaseline * intervalDispersionFactors[None, :]
            )

            # run the initial forward pass with baseline matrixMunc
            # ... recall, all dispersion factors = 1.0, here
            phiHat, adjustmentCount, perIntervalNLL, loss = _forwardPass(
                isInitialPass=True,
                returnNLL_=True,
                storeNLLInD_=True,
            )

            bestLoss = float(loss)  # initial loss with baseline matrixMunc
            bestBlockDispersionFactors = BlockDispersionFactors.copy()
            prevAcceptedIntervalNLL = np.array(perIntervalNLL, copy=True)
            acceptedLoss = float(loss)

            for iterCt in range(int(calibration_maxIters)):
                # Set interval-level scales, 'update' matrixMunc, and run forward pass
                intervalDispersionFactors = BlockDispersionFactors[
                    intervalToBlockMap
                ]
                matrixMunc[:] = (
                    initialMuncBaseline * intervalDispersionFactors[None, :]
                )
                phiHat, adjustmentCount, perIntervalNLL, loss = _forwardPass(
                    isInitialPass=True,
                    returnNLL_=True,
                    storeNLLInD_=True,
                    intervalToBlockMapOut=intervalToBlockMap,
                    blockGradLogScalesOut=blockGradLogScales,
                    blockGradCountOut=blockGradCount,
                )

                # loss after update
                loss = float(loss)
                if loss < bestLoss:
                    bestLoss = float(loss)
                    logger.info(f" Iter {iterCt}: loss: {bestLoss:.4f} ")
                    bestBlockDispersionFactors = BlockDispersionFactors.copy()

                # We form a smaller 'active set' of blocks to reduce overhead
                # ... The active set is chosen as the top `activeSetSize` blocks ranked by gradient magnitude
                gradDenomAll = np.maximum(blockGradCount, 1.0).astype(
                    np.float32, copy=False
                )
                gradMeansAll = (blockGradLogScales / gradDenomAll).astype(
                    np.float32, copy=False
                )
                gradScore = np.abs(gradMeansAll).astype(np.float32, copy=False)
                priority = np.argsort(gradScore)[::-1]
                activeBlocks = priority[:activeSetSize]

                # Greedy Minimization
                # ... Take a single joint step on  active blocks in log-space (others fixed),
                # ... project to bounds, then do short backtracking line search on the step size. Repeat.
                # ... Minimal guarantees: accepted steps give non-increasing loss, and the active set
                # ... (largest |gradMean|) targets the steepest first-order decrease on the linearized loss.

                accepted = False
                acceptedIntervalNLL = perIntervalNLL
                acceptedLoss = float(loss)

                logLowerUpdateLimit = np.float32(
                    np.log(float(lowerUpdateLimit))
                )
                logUpperUpdateLimit = np.float32(
                    np.log(float(upperUpdateLimit))
                )

                u = np.log(BlockDispersionFactors).astype(
                    np.float32, copy=False
                )
                eta = np.float32(calibration_activeSetLogStepSize)

                deltaU = np.zeros_like(u, dtype=np.float32)
                deltaU[activeBlocks] = (
                    -eta * gradMeansAll[activeBlocks]
                ).astype(np.float32, copy=False)
                np.clip(
                    deltaU,
                    -np.float32(calibration_activeSetMaxLogStep),
                    np.float32(calibration_activeSetMaxLogStep),
                    out=deltaU,
                )

                for retryCt in range(int(calibration_BGDMaxBacktracks)):
                    stepScale = np.float32(
                        calibration_BGDBacktrackFactor
                    ) ** np.float32(retryCt)
                    candidateU = (u + (deltaU * stepScale)).astype(
                        np.float32, copy=False
                    )
                    np.clip(
                        candidateU,
                        logLowerUpdateLimit,
                        logUpperUpdateLimit,
                        out=candidateU,
                    )
                    candidate_BlockDispersionFactors = np.exp(
                        candidateU
                    ).astype(np.float32)

                    intervalDispersionFactors = (
                        candidate_BlockDispersionFactors[intervalToBlockMap]
                    )
                    matrixMunc[:] = (
                        initialMuncBaseline * intervalDispersionFactors[None, :]
                    )

                    (
                        phiHatTry,
                        adjTry,
                        candIntervalNLL,
                        candidateLoss,
                    ) = _forwardPass(
                        isInitialPass=True,
                        returnNLL_=True,
                        storeNLLInD_=True,
                    )
                    candidateLoss = float(candidateLoss)

                    if candidateLoss <= acceptedLoss:
                        BlockDispersionFactors[:] = (
                            candidate_BlockDispersionFactors
                        )
                        acceptedIntervalNLL = candIntervalNLL
                        acceptedLoss = candidateLoss
                        accepted = True

                        if candidateLoss < bestLoss:
                            bestLoss = float(candidateLoss)
                            bestBlockDispersionFactors = (
                                BlockDispersionFactors.copy()
                            )
                        logger.info(
                            f" BGD step accepted, candidate loss: {candidateLoss:.4f}"
                        )
                        break

                if not accepted:
                    acceptedIntervalNLL = perIntervalNLL
                    acceptedLoss = loss

                if prevAcceptedIntervalNLL is not None:
                    isfiniteMask_ = np.isfinite(
                        prevAcceptedIntervalNLL
                    ) & np.isfinite(acceptedIntervalNLL)
                    k = int(isfiniteMask_.sum())
                    if k >= calibration_earlyStopMinValidIntervals:
                        deltaNLL = (
                            prevAcceptedIntervalNLL[isfiniteMask_]
                            - acceptedIntervalNLL[isfiniteMask_]
                        ).astype(np.float64, copy=False)

                        # compute mean and sd of per-interval NLL differences
                        # ... use to assess whether current improvement is negligible
                        # ... --> early stop
                        meanDiff = float(deltaNLL.mean())
                        if k > 1:
                            sd = float(deltaNLL.std(ddof=1))
                        else:
                            sd = 0.0
                        if sd > 0.0:
                            se = sd / float(np.sqrt(k))
                            isNotNegligible = (
                                meanDiff > float(calibration_earlyStopZ) * se
                            )
                        else:
                            isNotNegligible = meanDiff > 0.0

                        if isNotNegligible:
                            earlyStopNoSigIters = 0
                        else:
                            earlyStopNoSigIters += 1

                        if earlyStopNoSigIters >= calibration_earlyStopCriteria:
                            break

                prevAcceptedIntervalNLL = np.array(
                    acceptedIntervalNLL, copy=True
                )

            intervalDispersionFactors = bestBlockDispersionFactors[
                intervalToBlockMap
            ]
            matrixMunc[:] = (
                initialMuncBaseline * intervalDispersionFactors[None, :]
            )

        phiHat, countAdjustments, NIS = _forwardPass(
            isInitialPass=False,
            stateForwardOut=stateForward,
            stateCovarForwardOut=stateCovarForward,
            pNoiseForwardOut=pNoiseForward,
        )
        logger.info(
            f"Forward pass completed: {countAdjustments} process noise adjustments, {phiHat:.4f}"
        )

        stateForwardArr = stateForward
        stateCovarForwardArr = stateCovarForward
        pNoiseForwardArr = pNoiseForward
        stateForward.flush()
        stateCovarForward.flush()
        pNoiseForward.flush()

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
            progressBarBack = tqdm(total=(n - 1), unit=" intervals ")

        try:
            (
                stateSmoothedArr,
                stateCovarSmoothedArr,
                postFitResidualsArr,
            ) = cconsenrich.cbackwardPass(
                matrixData=matrixData,
                matrixF=matrixF,
                stateForward=stateForwardArr,
                stateCovarForward=stateCovarForwardArr,
                pNoiseForward=pNoiseForwardArr,
                projectStateDuringFiltering=bool(projectStateDuringFiltering),
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
        finally:
            # close in any case
            if progressBarBack is not None:
                progressBarBack.close()

        stateSmoothedArr.flush()
        stateCovarSmoothedArr.flush()
        postFitResidualsArr.flush()

        outStateSmoothed = np.array(stateSmoothedArr, copy=True)
        outPostFitResiduals = np.array(postFitResidualsArr, copy=True)
        outStateCovarSmoothed = np.array(stateCovarSmoothedArr, copy=True)

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
        NIS.astype(np.float32, copy=False),
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
            np.maximum(out_, np.float32(stateLowerBound), out=out_)
        if stateUpperBound is not None:
            np.minimum(out_, np.float32(stateUpperBound), out=out_)
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
        That is, the observation noise levels for each sample :math:`j=1,2,\ldots,m` at interval :math:`i`. To mask_ memory usage minimal `matrixMunc` is not returned in full or computed in
        in :func:`runConsenrich`. If using Consenrich programmatically, run :func:`consenrich.core.getMuncTrack` for each sample's count data (rows in the matrix output of :func:`readBamSegments`).
    :type matrixMunc: np.ndarray
    :param stateCovarSmoothed: Post-fit (forward/backward-smoothed) state covariance matrices :math:`\widetilde{\mathbf{P}}_{[i]}` from :func:`runConsenrich`.
    :type stateCovarSmoothed: Optional[np.ndarray]
    :return: A one-dimensional array of "precision-weighted residuals"
    :rtype: npt.NDArray[np.float32]
    """

    n, m = postFitResiduals.shape
    postFitResiduals_CContig = np.ascontiguousarray(
        postFitResiduals, dtype=np.float32
    )

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
        (stateCovarSmoothed is not None) and len(stateCovarSmoothed) == n
    ) or (not matrixMunc.flags.writeable)

    matrixMunc_CContig = np.array(
        matrixMunc, dtype=np.float32, order="C", copy=needsCopy
    )

    if needsCopy:
        stateCovarArr00 = np.asarray(
            stateCovarSmoothed[:, 0, 0], dtype=np.float32
        )
        matrixMunc_CContig += stateCovarArr00

    np.maximum(matrixMunc_CContig, np.float32(1e-8), out=matrixMunc_CContig)
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
    blockSizeBP: Optional[int] = 1000,
    samplingIters: int = 25_000,
    randomSeed: int = 42,
    fitFunc: Optional[Callable] = None,
    fitFuncArgs: Optional[dict] = None,
    evalFunc: Optional[Callable] = None,
    excludeMask: Optional[np.ndarray] = None,
) -> tuple[npt.NDArray[np.float32], float]:
    r"""Approximate region- and sample-specific (**M**)easurement (**unc**)ertainty tracks

    Compute prior values for sample- and region-specific uncertainty :math:`{R}_{[i:n]}` as a function of mean, :math:`\hat{f}\left(\sigma^2 \,\mid\,\mu\right)`, where :math:`\mu` is the local mean

    * We fit an AR(1) process model to randomly-sampled, contiguous genomic blocks. The length of blocks ``blockSizeBP`` is defined such that residual variances from the AR(1) fit reflect unwanted technical and/or biological sources of variation.
    * The mean and variance of the observed values within each block are recorded as a pair :math:`\left(\mu_k, \sigma^2_k\right)`, where :math:`k=1,2,\ldots,\textsf{samplingIters}`.
    * The sampled pairs are then used to estimate the global mean-variance trend, :math:`\hat{f}\left(\sigma^2 \,\mid\,\mu\right)`, using default :func:`consenrich.cconsenrich.cmonotonicFit`.
    * Finally, at each genomic interval :math:`i=1,2,\ldots,n`, the observed value is assigned an uncertainty :math:`R_{[i]} = \hat{f}\left(\sigma^2 \,\mid\,\mu_i\right)`, where :math:`\mu_i` is the local mean estimated using an exponential moving average (EMA) with window size similar to ``blockSizeBP``.

    :param chromosome: chromosome/contig name
    :type chromosome: str
    :param intervals: genomic intervals positions (start positions)
    :type intervals: npt.NDArray[np.uint32]
    :param minR: Minimum allowable uncertainty.
    :type minR: float
    :param maxR: Maximum allowable uncertainty.
    :type maxR: float
    :param blockSizeBP: Size (in bp) of contiguous blocks to sample when estimating global mean-variance trend.
    :type blockSizeBP: int
    :param samplingIters: Number of contiguous blocks to sample when estimating global mean-variance trend.
    :type samplingIters: int
    :param fitFunc: A *callable* function accepting input ``(arrayOfMeans,arrayOfVariances, **kwargs)``. Used to fit the global mean-variance model
    given sampled blocks from :func:``consenrich.cconsenrich.cmeanVarPairs``.
    :type fitFunc: Optional[Callable]
    :param fitFuncArgs: Additional keyword arguments to pass to `fitFunc`.
    :type fitFuncArgs: Optional[dict]
    :param evalFunc: A *callable* function with input (``outputFromFitFunc, arrayLengthN``) that evaluates the fitted :math:`\hat{f}(array[i])` at each genomic interval :math:`i=1,2,\ldots,n`.
    :type evalFunc: Optional[Callable]
    :return: Tuple (uncertainty track, prior strength), where prior strength is a scalar reflecting goodness-of-fit of the global mean-variance trend.
    :rtype: tuple[npt.NDArray[np.float32], float]
    """

    if fitFunc is None:
        fitFunc = cconsenrich.cmonotonicFit
    if evalFunc is None:
        evalFunc = cconsenrich.cmonotonicFitEval
    if fitFuncArgs is None:
        fitFuncArgs = {}
    if blockSizeBP is None:
        blockSizeBP = stepSize * 11
    blockSizeIntervals = int(blockSizeBP / stepSize)
    if blockSizeIntervals < 10:
        logger.warning(
            f"`blockSizeBP` is small for sampling (mean, variance) pairs...trying 11*stepSize"
        )
        blockSizeIntervals = 11
    localWindow = max(2, (blockSizeIntervals + 1))
    intervalsArr = np.ascontiguousarray(intervals, dtype=np.uint32)
    valuesArr = np.ascontiguousarray(values, dtype=np.float32)

    if excludeMask is None:
        excludeMaskArr = np.zeros_like(intervalsArr, dtype=np.uint8)
    else:
        excludeMaskArr = np.ascontiguousarray(excludeMask, dtype=np.uint8)

    # Variance as function of mean, globally, as observed in distinct, randomly drawn genomic
    # ... blocks. Within each block, it is assumed that an AR(1) process can --on the average--
    # ... account for a large fraction of desired signal, and the associated fit RSS
    # ... can be treated as unwanted technical/biological variation.

    # (i) For each block `k`, we get one (blockMean_k, blockVar_k) pair, where
    # ... `k=1,2,...,samplingIters`
    blockMeans, blockVars, starts, ends = cconsenrich.cmeanVarPairs(
        intervalsArr,
        valuesArr,
        blockSizeIntervals,
        samplingIters,
        randomSeed,
        excludeMaskArr,
        useInnovationVar=True,
    )

    # (ii) Fit mean-variance trend to sampled blocks/pairs
    predBlock = blockMeans
    sortIdx = np.argsort(np.abs(predBlock))
    blockMeansSorted = predBlock[sortIdx]
    blockVarsSorted = blockVars[sortIdx]
    opt = fitFunc(blockMeansSorted, blockVarsSorted, **fitFuncArgs)
    fittedBlockVars = evalFunc(opt, blockMeansSorted).astype(
        np.float64, copy=False
    )
    fitResidual = (
        blockVarsSorted.astype(np.float64, copy=False) - fittedBlockVars
    )
    fitRSS = float(np.sum(fitResidual * fitResidual))
    y_ = blockVarsSorted.astype(np.float64, copy=False)
    y_mean = float(np.mean(y_))
    sumSq = float(np.sum((y_ - y_mean) * (y_ - y_mean))) + 1.0e-4
    globalR2 = 1.0 - (fitRSS / sumSq)
    fitR2 = globalR2
    logger.info(
        f"Mean-variance fit over {len(blockMeans)} sampled blocks:"
        f"coefficients = {opt[::-1]}"
    )
    meanTrack = valuesArr.copy()
    globalModelVariances = evalFunc(
        opt, cconsenrich.cEMA(meanTrack, localWindow)
    ).astype(np.float32)

    return (
        np.clip(globalModelVariances, minR, maxR).astype(np.float32),
        float(fitR2),
    )


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


def adjustFeatureBounds(feature: bed.Interval, stepSize: int) -> bed.Interval:
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
    distThreshold: float = 1e6,
) -> dict[int, np.ndarray]:
    r"""Get a mapping of genomic intervals to nearest 'sparse' regions.

    :param chromosome: chromosome/contig ID
    :type chromosome: str
    numNearest: number of nearest sparse regions to consider
    :type numNearest: int
    sparseBedFile: path to sparse BED file. See :class:`observationParams`.
    :type sparseBedFile: str
    distThreshold: maximum distance (in bp) to consider a sparse region 'nearby'
    :type distThreshold: float
    :return: A dictionary mapping each interval index to an array of indices of the nearest sparse regions.
    :rtype: dict[int, np.ndarray]

    .. todo::

        This function could be an easy target for optimization
    """
    sparseStarts = np.asarray(
        sparseIntersection(chromosome, intervals, sparseBedFile)
    ).ravel()
    intervals = np.asarray(intervals).ravel()

    n = intervals.size
    m = sparseStarts.size
    numNearest = int(numNearest)

    if n == 0:
        return {}
    if numNearest <= 0:
        return {i: np.empty(0, dtype=np.uint32) for i in range(n)}
    if m == 0:
        return {i: np.zeros(numNearest, dtype=np.uint32) for i in range(n)}

    idxSparseInIntervals = np.searchsorted(intervals, sparseStarts, side="left")
    idxSparseInIntervals = np.clip(idxSparseInIntervals, 0, n - 1).astype(
        np.uint32, copy=False
    )

    centers = np.searchsorted(sparseStarts, intervals, side="left")
    centers = np.clip(centers, 0, m - 1).astype(np.int64, copy=False)
    offsets = np.arange(-numNearest, numNearest, dtype=np.int64)
    candidates = centers[:, None] + offsets[None, :]
    valid = (candidates >= 0) & (candidates < m)
    candidates = np.clip(candidates, 0, m - 1)

    dists = np.abs(sparseStarts[candidates] - intervals[:, None]).astype(
        np.float32
    )
    dists[~valid] = np.inf

    mask = np.argpartition(
        dists, kth=min(numNearest - 1, dists.shape[1] - 1), axis=1
    )[:, :numNearest]

    chosenCandidates = candidates[np.arange(n)[:, None], mask]
    chosenDists = dists[np.arange(n)[:, None], mask]
    mapped = idxSparseInIntervals[chosenCandidates]
    avgNumKept: float = 0.0
    rows = []
    for i in tqdm(range(n), desc="Building sparse map", unit=" intervals "):
        keep = np.isfinite(chosenDists[i]) & (chosenDists[i] <= distThreshold)
        kept = mapped[i, keep]
        avgNumKept += kept.size
        if kept.size >= numNearest:
            outRow = kept[:numNearest]
        elif kept.size > 0:
            outRow = np.pad(kept, (0, numNearest - kept.size), mode="edge")
        else:
            nearest = mapped[i, int(np.argmin(chosenDists[i]))]
            outRow = np.full(numNearest, nearest, dtype=np.intp)

        rows.append(outRow.astype(np.intp, copy=False))
    avgNumKept /= n
    logger.info(
        f"Avg. number of unique sparse regions within distThreshold={distThreshold} bp = {avgNumKept}"
    )
    return {i: rows[i] for i in range(n)}


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
        raise ValueError("intervals must contain at least two positions")
    bedFile_ = str(bedFile)

    # (possibly redundant) creation of uint32 version
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
        raise ValueError("Average cross-sample fraglen estimation failed")


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
        raise ValueError(f"`plotDirectory` {plotDirectory} does not exist")
    elif not os.path.isdir(plotDirectory):
        raise ValueError(f"`plotDirectory` {plotDirectory} is not a directory")

    plotFileName = os.path.join(
        plotDirectory,
        f"consenrichPlot_hist_{chromosome}_{plotPrefix}_state.v{__version__}.png",
    )
    binnedStateEstimates = _forPlotsSampleBlockStats(
        values_=primaryStateValues,
        blockSize_=blockSize,
        numBlocks_=numBlocks,
        statFunction_=statFunction,
        randomSeed_=randomSeed,
    )
    plt.figure(figsize=(plotWidthInches, plotHeightInches), dpi=plotDPI)
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
        logger.info(f"Wrote state estimate histogram to {plotFileName}")
        return plotFileName
    logger.warning(f"Failed to create histogram. {plotFileName} not written.")
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
        raise ValueError(f"`plotDirectory` {plotDirectory} does not exist")
    elif not os.path.isdir(plotDirectory):
        raise ValueError(f"`plotDirectory` {plotDirectory} is not a directory")

    plotFileName = os.path.join(
        plotDirectory,
        f"consenrichPlot_hist_{chromosome}_{plotPrefix}_residuals.v{__version__}.png",
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
    plt.figure(figsize=(plotWidthInches, plotHeightInches), dpi=plotDPI)
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
    logger.warning(f"Failed to create histogram. {plotFileName} not written.")
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
        raise ValueError(f"`plotDirectory` {plotDirectory} does not exist")
    elif not os.path.isdir(plotDirectory):
        raise ValueError(f"`plotDirectory` {plotDirectory} is not a directory")

    plotFileName = os.path.join(
        plotDirectory,
        f"consenrichPlot_hist_{chromosome}_{plotPrefix}_stateStd.v{__version__}.png",
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
    logger.warning(f"Failed to create histogram. {plotFileName} not written.")
    return None
