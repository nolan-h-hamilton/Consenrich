# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
import os
import warnings
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import (
    Any,
    Callable,
    DefaultDict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Literal,
)

from importlib.util import find_spec
import numpy as np
import numpy.typing as npt
import pybedtools as bed
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage, signal, stats, optimize, special
from tqdm import tqdm
from itrigamma import itrigamma
from . import cconsenrich
from . import __version__

MATHFONT = {
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
}

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
    :param plotMWSRHistogram: If True, plot a histogram of post-fit weighted squared residuals (MWSR).
    :type plotMWSRHistogram: bool
    :param plotHeightInches: Height of output plots in inches.
    :type plotHeightInches: float
    :param plotWidthInches: Width of output plots in inches.
    :type plotWidthInches: float
    :param plotDPI: DPI of output plots (png)
    :type plotDPI: int
    :param plotDirectory: Directory where plots will be written.
    :type plotDirectory: str or None

    :seealso: :func:`plotStateEstimatesHistogram`, :func:`plotMWSRHistogram`, :class:`outputParams`
    """

    plotPrefix: str | None = None
    plotStateEstimatesHistogram: bool = False
    plotMWSRHistogram: bool = False
    plotHeightInches: float = 6.0
    plotWidthInches: float = 8.0
    plotDPI: int = 300
    plotDirectory: str | None = None


class processParams(NamedTuple):
    r"""Parameters related to the process model of Consenrich.

    The consensus epigenomic signal is modeled explicitly with a simple 'level + slope' *process*.

    :param deltaF: Controls the integration step size between the signal 'slope' :math:`\dot{x}_{[i]}`
            and the signal 'level' :math:`x_{[i]}`.

            - If set ``< 0``, ``deltaF`` is determined automatically from :func:`consenrich.core.autoDeltaF` based on the data.

    :type deltaF: float
    :param minQ: Minimum process noise scale (diagonal in :math:`\mathbf{Q}_{[i]}`)
        on the primary state variable (signal level). If ``minQ < 0`` (default), a small
        value scales the minimum observation noise level (``observationParams.minR``) and is used
        for numerical stability.
    :type minQ: float
    :param maxQ: Maximum process noise scale. If ``maxQ < 0`` (default), no effective upper bound is enforced.
    :type maxQ: float
    :param offDiagQ: Off-diagonal value in the process noise covariance :math:`\mathbf{Q}_{[i,01]}`
    :type offDiagQ: float
    :seealso: :func:`consenrich.core.autoDeltaF`, :func:`consenrich.core.runConsenrich`

    """

    deltaF: float
    minQ: float
    maxQ: float
    offDiagQ: float


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich.

    The observation model is used to integrate sequence alignment data from the multiple replicates
    while accounting for both region- and replicate-specific noise.


    :param minR: Genome-wide lower bound for replicate-specific observation noise scales. In the default implementation, this clip is computed as a small fraction of values in the left-tail of :math:`\mathbf{R} \in \mathbb{R}^{m \times n}` with :func:`consenrich.core.getMuncTrack`.
    :type minR: float | None
    :param maxR: Genome-wide upper bound for the replicate-specific observation noise scales.
    :type maxR: float | None
    :param samplingIters: Number of blocks (within-contig) to sample while building the empirical absMean-variance trend in :func:`consenrich.core.fitVarianceFunction`.
    :type samplingIters: int | None
    :param samplingBlockSizeBP: Expected size (in bp) of contiguous blocks that are sampled when fitting AR1 parameters to estimate :math:`(\lvert \mu_b \rvert, \sigma^2_b)` pairs.
      Note, during sampling, each block's size (unit: genomic intervals) is drawn from truncated :math:`\textsf{Geometric}(p=1/\textsf{samplingBlockSize})` to reduce artifacts from fixed-size blocks.
      If `None` or ` < 1`, then this value is inferred using :func:`consenrich.core.getContextSize`.
    :type samplingBlockSizeBP: int | None
    :param binQuantileCutoff: When fitting the variance function, pairs :math:`(\lvert \mu_b \rvert, \sigma^2_b)` are binned by their (absolute) means. This parameter sets the quantile of variances within each bin to use when fitting the global mean-variance trend.
      Increasing this value toward `1.0` can raise the prior trend for measurement uncertainty and yield stiffer signal estimates overall.
    :type binQuantileCutoff: float | None
    :param EB_minLin: Require that the fitted trend in :func:`consenrich.core.getMuncTrack` satisfy: :math:`\textsf{variance} \geq \textsf{minLin} \cdot |\textsf{mean}|`. See :func:`fitVarianceFunction`.
    :type EB_minLin: float | None
    :param EB_use: If True, shrink 'local' noise estimates to a prior trend dependent on amplitude. See  :func:`consenrich.core.getMuncTrack`.
    :type EB_use: bool | None
    :param EB_setNu0: If provided, manually set :math:`\nu_0` to this value (rather than computing via :func:`consenrich.core.EB_computePriorStrength`).
    :type EB_setNu0: int | None
    :param EB_setNuL: If provided, manually set local model df, :math:`\nu_L`, to this value.
    :type EB_setNuL: int | None
    :param damp: Values :math:`> 0` induce a conservative upscaling of the measurement noise covariance matrix with respect to the number of samples, :math:`m`.
       This is akin to a soft damping on the filter gain as the number of samples, :math:`m`, increases.
    :type damp: float | None
    :param pad: A small constant added to the measurement noise variance estimates for numerics.
    :type pad: float | None
    :param EM_tNu: Degrees of freedom :math:`\nu` for the Student-t / Gaussian scale-mixture
        used for robust reweighting in :func:`consenrich.cconsenrich.cblockScaleEM`.
        Larger values push the model toward a Gaussian residual model (less downweighting of apparent outliers),
        and smaller values increase robustness to outliers but can reduce sensitivity to true signal.
        Values in the range ``[5, 15]`` are reasonable, and the default ``10.0`` is sufficient (and not disruptive) for most datasets.
        Users can set to an arbitrarily large value, e.g., :math:`\nu = 1e6` to effectively disable robust reweighting and
        use a standard Gaussian model for residuals.
    :type EM_tNu: float | None
    :param EM_alphaEMA: Used in :func:`consenrich.cconsenrich.cblockScaleEM`. Exponential moving-average (EMA) coefficient applied to per-block scale updates in **log space**.
        After each M-step, we smooth as

        :math:`\log s_b \leftarrow (1-\alpha)\log s_b + \alpha \log \hat{s}_b`,

        where :math:`\hat{s}_b` is the raw per-iteration update. Smaller values give **more smoothing** (slower adaptation).
        ``1.0`` disables smoothing (use the raw update), and values near ``0.0`` give very strong smoothing (slow adaptation).
        Note that smoothing voids *guaranteed* non-increasing behavior of the EM objective, but can be helpful for stability and convergence in practice.
        Values in the range ``[0.05, 0.25]`` are good starting points. The default ``0.1`` corresponds to a halflife of about 7 iterations and is sufficient for most datasets.

    :type EM_alphaEMA: float | None

    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitVarianceFunction`, :func:`consenrich.core.EB_computePriorStrength`, :func:`consenrich.cconsenrich.cblockScaleEM`

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
    damp: float | None
    pad: float | None
    EM_tNu: float | None
    EM_alphaEMA: float | None


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
    :param rescaleStateCovar: If True, the state covariance :math:`\mathbf{P}_{[i]}` is rescaled (in segments) after filtering such that observed
      studentized residuals are consistent with expected values. See :func:`consenrich.cconsenrich.crescaleStateCovar`.
    :type rescaleStateCovar: bool
    """

    stateInit: float
    stateCovarInit: float
    boundState: bool
    stateLowerBound: float
    stateUpperBound: float
    rescaleStateCovar: bool | None


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
    :param inferFragmentLength: Intended for single-end data: if > 0, the maximum correlation lag
       (avg.) between *strand-specific* read tracks is taken as the fragment length estimate and used to
       extend reads from 5'. Ignored if data is paired-end, `countEndsOnly`, or `fragmentLengths` is provided.
       important when targeting broader marks (e.g., ChIP-seq H3K27me3).
    :type inferFragmentLength: int
    :param countEndsOnly: If True, only the 5' read lengths contribute to counting.
    :type countEndsOnly: Optional[bool]
    :param minMappingQuality: Minimum mapping quality (MAPQ) for reads to be counted.
    :type minMappingQuality: Optional[int]
    :param fragmentLengths: If supplied, a list of estimated fragment lengths for each BAM file.
        These are values are used to extend reads. Note, these values will override `TLEN` attributes in paired-end data

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

        - ``SF``: Median of ratios style scale factors (see :func:`consenrich.cconsenrich.cSF`). Restricted to analyses with ``>= 3`` samples (no input control).

        - ``RPKM``: Scale factors based on Reads Per Kilobase per Million mapped reads (see :func:`consenrich.detrorm.getScaleFactorPerMillion`)

    :type normMethod: str
    :param fragmentLengths: List of fragment lengths (bp) to use for extending reads from 5' ends when counting single-end data.
    :type fragmentLengths: List[int], optional
    :param fragmentLengthsControl: List of fragment lengths (bp) to use for extending reads from 5' ends when counting single-end with control data.
    :type fragmentLengthsControl: List[int], optional
    :param useTreatmentFragmentLengths: If True, use fragment lengths estimated from treatment BAM files for control BAM files, too.
    :type useTreatmentFragmentLengths: bool, optional
    :param fixControl: If True, treatment samples are not upscaled, and control samples are not downscaled.
    :type fixControl: bool, optional
    :param globalWeight: Relative weight assigned to the global 'dense' baseline when combining with local baseline estimates. Higher values increase the influence of the global baseline. For instance, ``globalWeight = 2`` results in a weighted average where the global baseline contributes `2/3` of the final baseline estimate; whereas ``globalWeight = 1`` results in equal weighting between global and local baselines.
      Users with input control samples may consider increasing this value to avoid redundancy (artificial local trends have presumably been accounted for in the control, leaving less signal to be modeled locally).
    :type globalWeight: float, optional
    :param asymPos: *Relative* weight assigned to positive residuals to induce asymmetry in reweighting. Used
      during IRLS for the local baseline computation. Using smaller values near `0.0` will downweight peaks more and reduce the
      risk of removing true signal. Typical range is ``(0, 0.75]``.
    :type asymPos: float, optional
    :param logOffset: A small constant added to read normalized counts before log-transforming (pseudocount). For example,  :math:`\log(x + 1)` for ``logOffset = 1``. Default is ``1.0``.
    :type logOffset: float, optional
    :param logMult: Multiplicative factor applied to log-scaled and normalized counts. For example, setting ``logMult = 1 / \log(2)`` will yield log2-scaled counts after transformation, and setting ``logMult = 1.0`` yields natural log-scaled counts.
    :type logMult: float, optional
    :seealso: :func:`consenrich.cconsenrich.cTransform`

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

    intervalSizeBP: int | None
    backgroundBlockSizeBP: int | None
    smoothSpanBP: int | None
    scaleFactors: List[float] | None
    scaleFactorsControl: List[float] | None
    normMethod: str | None
    fragmentLengths: List[int] | None
    fragmentLengthsControl: List[int] | None
    useTreatmentFragmentLengths: bool | None
    fixControl: bool | None
    globalWeight: float | None
    asymPos: float | None
    logOffset: float | None
    logMult: float | None


class matchingParams(NamedTuple):
    r"""Parameters related to the matching algorithm.

    See :ref:`matching` for an overview of the approach.

    :param templateNames: A list of str values -- each entry references a mother wavelet (or its corresponding scaling function). e.g., `[haar, db2]`
    :type templateNames: List[str]
    :param cascadeLevels: Number of cascade iterations, or 'levels', used to define wavelet-based templates
        Must have the same length as `templateNames`, with each entry aligned to the
        corresponding template. e.g., given templateNames `[haar, db2]`, then `[2,2]` would use 2 cascade levels for both templates.
    :type cascadeLevels: List[int]
    :param iters: Number of randomly drawn contiguous blocks used to build
        an empirical null for significance evaluation. See :func:`cconsenrich.csampleBlockStats`.
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
    massQuantileCutoff: Optional[float]
    methodFDR: str | None


class outputParams(NamedTuple):
    r"""Parameters related to output files.

    :param convertToBigWig: If True, output bedGraph files are converted to bigWig format.
    :type convertToBigWig: bool
    :param roundDigits: Number of decimal places to round output values (bedGraph)
    :type roundDigits: int
    :param writeUncertainty: If True, write the posterior state uncertainty :math:`\sqrt{\widetilde{P}_{i,(11)}}` to bedGraph.
    :type writeUncertainty: bool
    :param writeMWSR: If True, write a per-interval mean squared *studentized* post-fit residual (``MWSR``),
        computed using the smoothed state and its posterior variance from the final filter/smoother pass.

        Let :math:`r_{[j,i]} = \texttt{matrixData}_{[j,i]} - \widetilde{x}_{[i]}` denote the post-fit residual for replicate :math:`j` at interval
        :math:`i`, where :math:`\widetilde{x}_{[i]}` is the consensus epigenomic signal level estimate. Let :math:`\widetilde{P}_{[00,i]}`
        be the posterior variance of the first state variable (signal level) and let :math:`R_{[j,i]}` be the observation noise variance for replicate :math:`j` at interval :math:`i`.
        Then, the studentized squared residuals :math:`u^2_{[j,i]}` and the mean weighted squared residuals :math:`\textsf{MWSR}[i]` are recorded as:

        .. math::

          u^2_{[j,i]} = \frac{r_{[j,i]}^2 + \widetilde{P}_{[00,i]}}{R_{[j,i]}},
          \qquad
          \textsf{MWSR}[i] = \frac{1}{m}\sum_{j=1}^{m} u^2_{[j,i]}.

        which is consistent with the EM routine in :func:`consenrich.cconsenrich.cblockScaleEM`
    :type writeMWSR: bool
    """

    convertToBigWig: bool
    roundDigits: int
    writeUncertainty: bool
    writeMWSR: bool


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
    intervalSizeBP: int,
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
        These are values are used to extend reads. Note, these values will override `TLEN` attributes in paired-end data
    :type fragmentLengths: Optional[List[int]]
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

    offsetStr = ((str(offsetStr) or "0,0").replace(" ", "")).split(",")

    numIntervals = ((end - start - 1) // intervalSizeBP) + 1
    counts = np.empty((len(bamFiles), numIntervals), dtype=np.float32)

    if pairedEndMode and fragmentLengths is not None:
        if isinstance(fragmentLengths, list) and len(fragmentLengths) != len(bamFiles):
            if len(fragmentLengths) == 1:
                fragmentLengths = fragmentLengths * len(bamFiles)
            else:
                raise ValueError(
                    f"`fragmentLengths` length must match `bamFiles` length: {len(fragmentLengths)} != {len(bamFiles)}.",
                )

        if isinstance(fragmentLengths, int):
            fragmentLengths = [fragmentLengths] * len(bamFiles)

        pairedEndMode = 0
        inferFragmentLength = 0

    elif pairedEndMode:
        # paired end w/ out fragment lengths provided --> use TLEN attribute for each properly paired read
        fragmentLengths = [0] * len(bamFiles)
        inferFragmentLength = 0

    if not pairedEndMode and (fragmentLengths is None or len(fragmentLengths) == 0):
        # single-end without user-supplied fragment length -->
        # ... estimate fragment lengths as the peak lag_k in
        # ... cross-correlation(forwardReadsTrack,backwardReadsTrack, lag_k)
        inferFragmentLength = 1
        fragmentLengths = [-1] * len(bamFiles)

    if isinstance(countEndsOnly, bool) and countEndsOnly:
        # No fragment length extension, just count 5' ends
        # ... May be preferred for high-resolution analyses in deeply-sequenced HTS
        # ...  data but note the drift in interpretation for processParams.deltaF,
        # ... consider setting deltaF \propto (readLength / intervalSizeBP)
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
            intervalSizeBP,
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
        Q[1, 1] = Q[0, 0] / 4.0

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
    stateInit: float,
    stateCovarInit: float,
    boundState: bool,
    stateLowerBound: float,
    stateUpperBound: float,
    chunkSize: int,
    blockLenIntervals: int,
    covarClip: float = 3.0,
    projectStateDuringFiltering: bool = False,
    pad: float = 1.0e-2,
    calibration_kwargs: dict[str, object] | None = None,
    disableCalibration: bool = False,
    rescaleStateCovar: bool = False,
    damp: float = 0.0,
    EM_tNu: float = 10.0,
    EM_alphaEMA: float = 0.1,
    returnScales: bool = True,
):
    r"""Execute Consenrich given transformed/normalized data and initial measurement and process noise (co)variances"""

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

    m, n = matrixData.shape
    if n < 2:
        raise ValueError("need at least 2 intervals for smoothing")

    if calibration_kwargs is None:
        calibration_kwargs = {}

    blockCount = int(np.ceil(n / float(blockLenIntervals)))
    intervalToBlockMap = (np.arange(n, dtype=np.int32) // blockLenIntervals).astype(
        np.int32
    )
    intervalToBlockMap[intervalToBlockMap >= blockCount] = blockCount - 1

    matrixF = constructMatrixF(float(deltaF)).astype(np.float32, copy=False)
    matrixQ0 = constructMatrixQ(float(minQ), offDiagQ=float(offDiagQ)).astype(
        np.float32, copy=False
    )

    EM_maxIters = int(calibration_kwargs.get("EM_maxIters", 50))
    EM_rtol = float(calibration_kwargs.get("EM_rtol", 1.0e-3))
    EM_multiplierLow = float(calibration_kwargs.get("EM_multiplierLow", 0.1))
    EM_multiplierHigh = float(calibration_kwargs.get("EM_multiplierHigh", 10.0))
    EM_scaleToMedian = bool(calibration_kwargs.get("EM_scaleToMedian", True))

    if damp > 0.0:
        logger.info(
            "Applying damping factor to initial matrixMunc: damp=%.4f",
            float(damp),
        )
        damp_ = 1.0 + (1.0 - np.exp(-float(damp) * float(m + 1)))
        np.multiply(matrixMunc, np.float32(damp_), out=matrixMunc)

    logger.info(
        "m=%d n=%d deltaF=%.6g minQ=%.6g maxQ=%.6g",
        int(m),
        int(n),
        float(deltaF),
        float(minQ),
        float(maxQ),
    )
    logger.info(
        "blockLenIntervals=%d blockCount=%d", int(blockLenIntervals), int(blockCount)
    )

    # Final forward/backward run for _fixed_ noise scales (either from EM calibration or default of 1.0)
    def _run_final_passes(rScale: np.ndarray, qScale: np.ndarray):
        stateForward = np.empty((n, 2), dtype=np.float32)
        stateCovarForward = np.empty((n, 2, 2), dtype=np.float32)
        pNoiseForward = np.empty((n, 2, 2), dtype=np.float32)
        vectorD = np.empty(n, dtype=np.float32)

        phiHat, _, vectorD, sumNLL = cconsenrich.cforwardPass(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixMunc,
            matrixF=matrixF,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            rScale=rScale,
            qScale=qScale,
            blockCount=int(blockCount),
            stateInit=float(stateInit),
            stateCovarInit=float(stateCovarInit),
            covarClip=float(covarClip),
            pad=float(pad),
            projectStateDuringFiltering=bool(projectStateDuringFiltering),
            stateLowerBound=float(stateLowerBound),
            stateUpperBound=float(stateUpperBound),
            chunkSize=0,
            stateForward=stateForward,
            stateCovarForward=stateCovarForward,
            pNoiseForward=pNoiseForward,
            vectorD=vectorD,
            progressBar=None,
            progressIter=0,
            returnNLL=True,
            storeNLLInD=False,
        )

        stateSmoothed, stateCovarSmoothed, _, postFitResiduals = (
            cconsenrich.cbackwardPass(
                matrixData=matrixData,
                matrixF=matrixF,
                stateForward=stateForward,
                stateCovarForward=stateCovarForward,
                pNoiseForward=pNoiseForward,
                covarClip=float(covarClip),
                chunkSize=0,
                stateSmoothed=None,
                stateCovarSmoothed=None,
                lagCovSmoothed=None,
                postFitResiduals=None,
                progressBar=None,
                progressIter=0,
            )
        )

        NIS = vectorD.astype(np.float32, copy=False)
        return (
            phiHat,
            sumNLL,
            stateSmoothed,
            stateCovarSmoothed,
            postFitResiduals,
            NIS,
            intervalToBlockMap,
        )

    if disableCalibration:
        rScale = np.ones(blockCount, dtype=np.float32)
        qScale = np.ones(blockCount, dtype=np.float32)
        (
            phiHat,
            sumNLL,
            stateSmoothed,
            stateCovarSmoothed,
            postFitResiduals,
            NIS,
            intervalToBlockMap,
        ) = _run_final_passes(rScale=rScale, qScale=qScale)
        logger.info("final phiHat=%.4f sumNLL=%.6g", float(phiHat), float(sumNLL))

    else:
        logger.info(
            "\nNoise (co)variance calibration\n\tEM_maxIters=%d EM_rtol=%.3e EM_multiplierLow=%.3e "
            "EM_multiplierHigh=%.3e EM_alphaEMA=%.3f EM_tNu=%.1f\n",
            int(EM_maxIters),
            float(EM_rtol),
            float(EM_multiplierLow),
            float(EM_multiplierHigh),
            float(EM_alphaEMA),
            float(EM_tNu),
        )

        rScale, qScale, emItersDone, emNLL = cconsenrich.cblockScaleEM(
            matrixData=matrixData,
            matrixPluginMuncInit=matrixMunc,
            matrixF=matrixF,
            matrixQ0=matrixQ0,
            intervalToBlockMap=intervalToBlockMap,
            blockCount=int(blockCount),
            stateInit=float(stateInit),
            stateCovarInit=float(stateCovarInit),
            EM_maxIters=int(EM_maxIters),
            EM_rtol=float(EM_rtol),
            covarClip=float(covarClip),
            pad=float(pad),
            EM_multiplierLow=float(EM_multiplierLow),
            EM_multiplierHigh=float(EM_multiplierHigh),
            EM_alphaEMA=float(EM_alphaEMA),
            EM_scaleToMedian=bool(EM_scaleToMedian),
            EM_tNu=float(EM_tNu),
            returnIntermediates=False,
        )

        logger.info(
            "EM summary: iters=%d (EM NLL=%.6g)", int(emItersDone), float(emNLL)
        )

        (
            phiHat,
            sumNLL,
            stateSmoothed,
            stateCovarSmoothed,
            postFitResiduals,
            NIS,
            intervalToBlockMap,
        ) = _run_final_passes(rScale=rScale, qScale=qScale)
        logger.info(
            "final phiHat=%.4f sumNLL=%.6g (EM NLL=%.6g)",
            float(phiHat),
            float(sumNLL),
            float(emNLL),
        )

    outStateSmoothed = np.asarray(stateSmoothed, dtype=np.float32)
    outStateCovarSmoothed = np.asarray(stateCovarSmoothed, dtype=np.float32)
    outPostFitResiduals = np.asarray(postFitResiduals, dtype=np.float32)

    if boundState:
        np.clip(
            outStateSmoothed[:, 0],
            np.float32(stateLowerBound),
            np.float32(stateUpperBound),
            out=outStateSmoothed[:, 0],
        )

    if returnScales:
        return (
            outStateSmoothed,
            outStateCovarSmoothed,
            outPostFitResiduals,
            NIS,
            np.asarray(rScale, dtype=np.float32),
            np.asarray(qScale, dtype=np.float32),
            intervalToBlockMap,
        )

    return (
        outStateSmoothed,
        outStateCovarSmoothed,
        outPostFitResiduals,
        NIS,
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

    intervalSizeBP: int = intervals[1] - intervals[0]
    chromFeatures: bed.BedTool = (
        bed.BedTool(sparseBedFile)
        .sort()
        .merge()
        .filter(
            lambda b: (
                b.chrom == chromosome
                and b.start > intervals[0]
                and b.end < intervals[-1]
                and (b.end - b.start) >= intervalSizeBP
            )
        )
    )
    centeredFeatures: bed.BedTool = chromFeatures.each(
        adjustFeatureBounds, intervalSizeBP=intervalSizeBP
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
                and (b.end - b.start) >= intervalSizeBP
            )
        )
    )
    centeredFeatures: bed.BedTool = chromFeatures.each(
        adjustFeatureBounds, intervalSizeBP=intervalSizeBP
    )
    centeredStarts = []
    for f in centeredFeatures:
        s = int(f.start)
        if start0 <= s <= last and (s - start0) % intervalSizeBP == 0:
            centeredStarts.append(s)
    return np.asarray(centeredStarts, dtype=np.int64)


def adjustFeatureBounds(feature: bed.Interval, intervalSizeBP: int) -> bed.Interval:
    r"""Adjust the start and end positions of a BED feature to be centered around a step."""
    feature.start = cconsenrich.stepAdjustment(
        (feature.start + feature.end) // 2, intervalSizeBP
    )
    feature.end = feature.start + intervalSizeBP
    return feature


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


def autoDeltaF(
    bamFiles: List[str],
    intervalSizeBP: int,
    chromMat: Optional[npt.NDArray[np.float32]] = None,
    fragmentLengths: Optional[List[int]] = None,
    fallBackFragmentLength: int = 147,
    randomSeed: int = 42,
    blockMult: float = 10.0,
    numBlocks: int = 1000,
    maxLagBins: int = 25,
    minDeltaF: float = 1.0e-4,
    maxDeltaF: float = 1.0,
    noiseEps: float = 1.0e-4,
    minBlockWeight: float = 1.0e-4,
) -> float:
    r"""Infer deltaF from the data using short time autocorrelation via FFT frames

    Steps
    - Build a stable 1D track using log1p of the per interval mean
    - Slide a window across the track like an STFT
    - For each window compute autocorrelation from FFT power spectrum
    - Correct the autocorrelation for window taper bias
    - Fit exponential decay in log space to get local L in intervals
    - Aggregate L across windows using a trimmed mean
    - Set deltaF = 1 / L and clip
    """

    def hannWindow(winLen: int) -> np.ndarray:
        # symmetric Hann window to reduce boundary effects
        if winLen <= 1:
            return np.ones((winLen,), dtype=np.float64)
        n = np.arange(winLen, dtype=np.float64)
        return 0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(winLen - 1))

    def nextPow2(x: int) -> int:
        # smallest power of two >= x
        if x <= 1:
            return 1
        return 1 << int((x - 1).bit_length())

    def trimmedMean(values: np.ndarray, trimFrac: float) -> float:
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        trimFrac = float(np.clip(trimFrac, 0.0, 0.49))
        # in case of scipy issues, fallback to manual trim mean implementation
        try:
            from scipy.stats import trim_mean

            return float(trim_mean(v, proportiontocut=trimFrac))
        except Exception:
            v = np.sort(v)
            k = int(np.floor(trimFrac * v.size))
            if v.size - 2 * k <= 0:
                return float(np.mean(v))
            return float(np.mean(v[k : v.size - k]))

    # estimate a representative fragment length across samples
    if (
        fragmentLengths is not None
        and len(fragmentLengths) > 0
        and all(isinstance(x, (int, float)) for x in fragmentLengths)
    ):
        fragLenArr = np.asarray(fragmentLengths, dtype=np.float64)
        medianFragmentLengthBP = trimmedMean(fragLenArr, trimFrac=0.1)
    elif bamFiles is not None and len(bamFiles) > 0:
        tmpFragmentLengths: List[float] = []
        for bamFile in bamFiles:
            fragLen = cconsenrich.cgetFragmentLength(
                bamFile,
                fallBack=fallBackFragmentLength,
                randSeed=randomSeed,
            )
            tmpFragmentLengths.append(float(fragLen))
        medianFragmentLengthBP = trimmedMean(
            np.asarray(tmpFragmentLengths, dtype=np.float64), trimFrac=0.1
        )
    else:
        raise ValueError("one of fragmentLengths or bamFiles is required")

    if (not np.isfinite(medianFragmentLengthBP)) or medianFragmentLengthBP <= 0.0:
        raise ValueError("fraglen estimation failed")

    # interval to fragment ratio fallback
    if chromMat is None:
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    # aggregate pointwise in natural scale then log1p for stability
    if hasattr(chromMat, "ndim") and int(chromMat.ndim) == 1:
        rawMean = np.asarray(chromMat, dtype=np.float64)
    else:
        rawMean = np.mean(chromMat, axis=0, dtype=np.float64)
    rawMean = np.clip(rawMean, 0.0, None)
    meanTrack = np.log1p(rawMean).astype(np.float64, copy=False)

    numIntervals = int(meanTrack.size)
    if numIntervals < 16 or (not np.isfinite(meanTrack).all()):
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    # window length tied to fragment length
    windowLenBP = int(
        max(
            int(intervalSizeBP),
            int(round(float(blockMult) * float(medianFragmentLengthBP))),
        )
    )
    windowLenIntervals = int(np.ceil(float(windowLenBP) / float(intervalSizeBP)))

    # keep windows reasonable for FFT based acorr
    windowLenIntervals = int(max(32, windowLenIntervals))
    windowLenIntervals = int(min(windowLenIntervals, numIntervals))
    if windowLenIntervals < 16:
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    # hop like a typical STFT
    hopIntervals = int(max(1, windowLenIntervals // 2))

    maxStart = int(numIntervals - windowLenIntervals)
    if maxStart < 0:
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    frameStartsAll = np.arange(0, maxStart + 1, hopIntervals, dtype=np.int64)
    if frameStartsAll.size < 2:
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    rng = np.random.default_rng(int(randomSeed))
    if frameStartsAll.size > int(numBlocks):
        chosen = rng.choice(frameStartsAll.size, size=int(numBlocks), replace=False)
        frameStarts = np.sort(frameStartsAll[chosen])
    else:
        frameStarts = frameStartsAll

    fitMaxLag = int(min(int(maxLagBins), windowLenIntervals - 1))
    if fitMaxLag < 4:
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    win = hannWindow(windowLenIntervals)
    nFft = nextPow2(2 * windowLenIntervals)

    # precompute window autocorrelation to correct taper bias
    winFft = np.fft.rfft(win, n=nFft)
    winPower = (winFft.real * winFft.real) + (winFft.imag * winFft.imag)
    winAcf = np.fft.irfft(winPower, n=nFft)
    winAcf0 = float(winAcf[0])

    if (not np.isfinite(winAcf0)) or winAcf0 <= 0.0:
        winRho = None
    else:
        winRho = winAcf[: fitMaxLag + 1] / winAcf0

    frameLengths: List[float] = []

    for startIndex in frameStarts:
        frame = meanTrack[startIndex : startIndex + windowLenIntervals]
        if frame.size != windowLenIntervals:
            continue
        if not np.isfinite(frame).all():
            continue

        # remove DC then apply taper
        x = frame - float(np.mean(frame))
        x = x * win

        # skip nearly constant frames
        xEnergy = float(np.dot(x, x))
        if (not np.isfinite(xEnergy)) or xEnergy <= float(minBlockWeight):
            continue

        # autocorrelation via FFT power spectrum
        X = np.fft.rfft(x, n=nFft)
        power = (X.real * X.real) + (X.imag * X.imag)
        acf = np.fft.irfft(power, n=nFft)

        acf0 = float(acf[0])
        if (not np.isfinite(acf0)) or acf0 <= 0.0:
            continue

        rho = acf[: fitMaxLag + 1] / acf0

        # correct window induced damping
        if winRho is not None:
            denom = np.asarray(winRho, dtype=np.float64)
            denom = np.where(denom > float(noiseEps), denom, float("nan"))
            rho = rho / denom

        # keep rho in a safe range
        rho = np.asarray(rho, dtype=np.float64)
        rho = np.clip(rho, -0.999999, 0.999999)

        # build fit set from positive rho values excluding lag 0
        rhoVals = rho[1:]
        lags = np.arange(1, rhoVals.size + 1, dtype=np.float64)
        posMask = np.isfinite(rhoVals) & (rhoVals > 0.0) & (rhoVals < 0.999999)
        if int(np.sum(posMask)) < 4:
            continue

        rhoPos = rhoVals[posMask]
        lagPos = lags[posMask]

        # fit band restricted to IQR of positive rho
        rhoHigh = float(np.quantile(rhoPos, 0.75))
        rhoLow = float(np.quantile(rhoPos, 0.25))
        fitMask = (rhoPos >= rhoLow) & (rhoPos <= rhoHigh)
        if int(np.sum(fitMask)) < 3:
            continue

        y = np.log(rhoPos[fitMask])
        xFit = lagPos[fitMask]

        # least squares slope for log rho = slope * lag + intercept
        xMean = float(np.mean(xFit))
        yMean = float(np.mean(y))
        xC = xFit - xMean
        yC = y - yMean
        denom = float(np.dot(xC, xC))
        if (not np.isfinite(denom)) or denom <= 0.0:
            continue
        slope = float(np.dot(xC, yC) / denom)

        # exponential decay expects negative slope
        if (not np.isfinite(slope)) or slope >= 0.0:
            continue

        frameL = float(-1.0 / slope)
        if (not np.isfinite(frameL)) or frameL <= 0.0:
            continue

        frameLengths.append(frameL)

    if len(frameLengths) < 8:
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    summaryLBins = trimmedMean(np.asarray(frameLengths, dtype=np.float64), trimFrac=0.1)
    if (not np.isfinite(summaryLBins)) or summaryLBins <= 0.0:
        summaryLBins = float(np.median(np.asarray(frameLengths, dtype=np.float64)))

    if (not np.isfinite(summaryLBins)) or summaryLBins <= 0.0:
        deltaF = float(intervalSizeBP) / float(medianFragmentLengthBP)
        return np.float32(min(deltaF, float(maxDeltaF)))

    deltaF = float(1.0 / summaryLBins)
    deltaF = float(np.clip(deltaF, float(minDeltaF), float(maxDeltaF)))
    return np.float32(deltaF)


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
        import matplotlib as mpl
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
    with mpl.rc_context(MATHFONT):
        plt.figure(figsize=(plotWidthInches, plotHeightInches), dpi=plotDPI)
        x = np.asarray(binnedStateEstimates, dtype=np.float64).ravel()
        binnedStateEstimates = x.astype(np.float32, copy=False)

        plt.hist(
            binnedStateEstimates,
            color="blue",
            alpha=1.0,
            edgecolor="black",
            fill=False,
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


def plotMWSRHistogram(
    chromosome: str,
    plotPrefix: str,
    MWSR: npt.NDArray[np.float32],
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
    r"""(Experimental) Plot a histogram of block-sampled weighted squared residuals (post-fit MWSR).

    :param plotPrefix: Prefixes the output filename
    :type plotPrefix: str
    :param blockSize: Number of contiguous intervals to sample per block.
    :type blockSize: int
    :param numBlocks: Number of samples to draw
    :type numBlocks: int
    :param statFunction: Numpy callable function to compute on each sampled block (e.g., `np.mean`, `np.median`).
    :type statFunction: Callable
    :param plotDirectory: If provided, saves the plot to this directory. The directory should exist.
    :type plotDirectory: str | None

    :seealso: :func:`runConsenrich`, :class:`outputParams`
    """

    if _checkMod("matplotlib"):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
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
        f"consenrichPlot_hist_{chromosome}_{plotPrefix}_MWSR.v{__version__}.png",
    )
    binnedMWSR = _forPlotsSampleBlockStats(
        values_=MWSR,
        blockSize_=blockSize,
        numBlocks_=numBlocks,
        statFunction_=statFunction,
        randomSeed_=randomSeed,
    )
    with mpl.rc_context(MATHFONT):
        plt.figure(figsize=(plotWidthInches, plotHeightInches), dpi=plotDPI)
        x = np.asarray(binnedMWSR, dtype=np.float64).ravel()
        if x.size:
            lowerLim, upperLim = np.quantile(x, [0, 0.99])
            x = x[(x >= lowerLim) & (x <= upperLim)]
        binnedMWSR = x.astype(np.float32, copy=False)

        plt.hist(
            binnedMWSR,
            color="blue",
            alpha=1.0,
            edgecolor="black",
            fill=False,
        )
        plt.title(
            rf"Histogram: {numBlocks} sampled blocks ({blockSize} contiguous intervals each): Weighted Squared Residuals (MWSR)",
        )
        plt.savefig(plotFileName, dpi=plotDPI)
        plt.close()
    if os.path.exists(plotFileName):
        logger.info(f"Wrote MWSR histogram to {plotFileName}")
        return plotFileName
    logger.warning(f"Failed to create histogram. {plotFileName} not written.")
    return None


def fitVarianceFunction(
    jointlySortedMeans: np.ndarray,
    jointlySortedVariances: np.ndarray,
    eps: float = 1.0e-2,
    binQuantileCutoff: float = 0.75,
    EB_minLin: float = 1.0e-2,
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
    EB_minLin: float = 1.0e-2,
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
    binQuantileCutoff: float = 0.75,
    EB_minLin: float = 1.0e-2,
    EB_use: bool = True,
    EB_setNu0: int | None = None,
    EB_setNuL: int | None = None,
    EB_localQuantile: float = 0.0,
    verbose: bool = False,
    eps: float = 5.0e-2,
) -> tuple[npt.NDArray[np.float32], float]:
    r"""Approximate initial sample-specific (**M**)easurement (**unc**)ertainty tracks

    For an individual experimental sample (replicate), quantify *positional* data uncertainty over genomic intervals :math:`i=1,2,\ldots n` spanning ``chromosome``.
    These tracks (per-sample) comprise the ``matrixMunc`` input to :func:`runConsenrich`, :math:`\mathbf{R}[:,:] \in \mathbb{R}^{m \times n}`.

    Variance is modeled as a function of the absolute mean signal level. For ``EB_use=True``, local variance estimates are also
    are integrated with shrinkage using a plug-in empirical Bayes approach.

    :param chromosome: chromosome/contig name
    :type chromosome: str
    :param values: normalized/transformed signal measurements over genomic intervals (e.g., :func:`consenrich.cconsenrich.cTransform` output)
    :type values: np.ndarray
    :param intervals: genomic intervals positions (start positions)
    :type intervals: np.ndarray

    See :class:`consenrich.core.observationParams` for other parameters.

    """

    AR1_PARAMCT = 3
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

    # Global:
    # ... Variance as function of |mean|, globally, as observed in distinct, randomly drawn genomic
    # ... blocks. Within fixed-size blocks, it's assumed that an AR(1) process can, on average,
    # ... account for a large fraction of desired signal, and the (residual) innovation variance
    # ... reflects noise
    blockMeans, blockVars, starts, ends = cconsenrich.cmeanVarPairs(
        intervalsArr,
        valuesArr,
        blockSizeIntervals,
        samplingIters,
        randomSeed,
        excludeMaskArr,
        useInnovationVar=True,
    )

    meanAbs = np.abs(blockMeans)
    mask = np.isfinite(meanAbs) & np.isfinite(blockVars) & (blockVars >= 1.0e-4)

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

    meanTrack = np.abs(valuesArr).copy()
    if useEMA:
        meanTrack = cconsenrich.cEMA(meanTrack, 2 / (localWindowIntervals + 1))
    priorTrack = evalVarianceFunction(opt, meanTrack, EB_minLin=EB_minLin).astype(
        np.float32, copy=False
    )

    if not EB_use:
        return priorTrack.astype(np.float32), np.sum(mask) / float(len(blockMeans))

    # Local:
    # ... 'Rolling' AR(1) innovation variance estimates over a sliding window
    obsVarTrack = cconsenrich.crolling_AR1_IVar(
        valuesArr,
        localWindowIntervals,
        excludeMaskArr,
    ).astype(np.float64, copy=False)

    # Note, negative values are a flag from `cconsenrich.crolling_AR1_IVar`
    # ... -- set as _NaN_ -- and handle later during shrinkage
    obsVarTrack[obsVarTrack < 0.0] = np.nan

    # ~Corresponds~ to `binQuantileCutoff` that is applied in the global/prior fit:
    # ... Optionally, run a quantile filter over the local variance track
    # ...     EB_localQuantile < 0 --> disable
    # ...     EB_localQuantile == 0 --> use binQuantileCutoff
    # ...     EB_localQuantile > 0 --> use supplied quantile value (x100)
    # ... NOTE: Useful heuristic for parity with the global model and tempering effects of
    # ...    spurious measurements in sparse genomic regions where estimated uncertainty
    # ...    is often artificially deflated. Note that the quantile filter _centered_,
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


def _localSlopeAbs(yVals: np.ndarray, xPos: float, halfWindow: int = 2) -> float:
    centerIdx = int(np.clip(int(np.floor(xPos)), 0, yVals.size - 1))
    leftIdx = max(0, centerIdx - halfWindow)
    rightIdx = min(yVals.size - 1, centerIdx + halfWindow)
    if rightIdx - leftIdx < 1:
        return 0.0
    xIdx = np.arange(leftIdx, rightIdx + 1, dtype=np.float64) - float(centerIdx)
    yWin = yVals[leftIdx : rightIdx + 1]
    sumSqX = float(np.sum(xIdx * xIdx))
    if sumSqX <= 0.0:
        return 0.0

    yCentered = yWin - float(np.mean(yWin))
    slope = float(np.sum(xIdx * yCentered) / sumSqX)
    return float(abs(slope))


def getContextSize(
    vals: np.ndarray,
    minSpan: int | None = 3,
    maxSpan: int | None = 64,
    bandZ: float = 1.0,
    maxOrder: int = 5,
) -> tuple[int, int, int]:
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

    # note that these are in log-scale
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        widths, _, leftIps, rightIps = signal.peak_widths(
            yLog,
            featureIndexArray,
            rel_height=0.5,  # half-width at half-maximum
        )

    # For each feature, convert to log-scale and assign per-feature variance using local noise heuristic.
    # ... the local noise heuristic is based on data within `noiseWindow` of the peak
    eps = 1e-4
    noiseWindow = int(min(maxSpan, 32))
    sHatList: list[float] = []
    sigma2List: list[float] = []

    for j, peakIdx in enumerate(featureIndexArray):
        widthHat = float(widths[j])
        leftIp = float(leftIps[j])
        rightIp = float(rightIps[j])

        if not (np.isfinite(leftIp) and np.isfinite(rightIp) and np.isfinite(widthHat)):
            continue
        if widthHat <= 0.0:
            continue
        #   log scale
        widthHat = float(max(1.0, widthHat))
        logWidth = float(np.log(widthHat))

        #   |slope| at left half-max and right half-max
        leftStep = _localSlopeAbs(yLog, float(leftIp), halfWindow=2)
        rightStep = _localSlopeAbs(yLog, float(rightIp), halfWindow=2)

        #   define window around the feature
        wLeft = max(0, int(peakIdx) - noiseWindow)
        wRight = min(n, int(peakIdx) + noiseWindow + 1)

        #   within the local window, use sdev of first-order differences as the local noise estimate (Y-axis)
        #   FFR: it might be more consistent to use the same local noise estimate as in `getMuncTrack`, maybe expensive though
        diffs = np.diff(yLog[wLeft:wRight])
        sigmaY = float(np.std(diffs, ddof=1)) if diffs.size >= 2 else 0.0

        #   propagate to X-axis (width), add contributions from left and right sides
        #   Var[W] ~=~ Var[Y] / (dY/dX)^2, 'independent' contributions from left and right sides
        #   note that variance is decreasing for increasing slope
        sigmaXLeft = sigmaY / (leftStep + eps)
        sigmaXRight = sigmaY / (rightStep + eps)
        sigmaW2 = (sigmaXLeft * sigmaXLeft) + (sigmaXRight * sigmaXRight)

        #   final per-feature variance on --log scale--
        #   Var[log(W)] ~=~ Var[W] / (W^2)
        sigmaS2 = float(max(1e-6, sigmaW2 / (widthHat * widthHat + eps)))
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

    # Take the posterior at each feature given fixed tauSqHat (~parametric EB~)
    # ... (or backup MoM estimate if optimization failed)
    tauSqHat: float = 0.0
    if getattr(res, "success", False):
        tauSqHat = float(res.x)
        logger.info(f"tau^2 MLE plugin = {tauSqHat:.6f}")
    else:
        tauSqHat = tau2Mom
        logger.warning(
            f"Failed to solve for tau^2 MLE...using MoM estimate tau^2 = {tau2Mom:.6f}.",
        )

    #   Posterior variance sigma^2[i] + tau^2 used to compute weights
    vHat = sigma2Arr + tauSqHat
    vHat = np.maximum(vHat, 1e-12)
    wHat = 1.0 / vHat

    # Get point estimate, band limits on natural scale
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

    logger.info(
        f"Natural scale: pointEstimate={pointEstimate:.4f}, Lower={widthLower:.4f}, Upper={widthUpper:.4f}"
    )

    return int(pointEstimate), int(widthLower), int(widthUpper)
