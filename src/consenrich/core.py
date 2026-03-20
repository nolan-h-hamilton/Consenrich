# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
import os
import warnings
from tempfile import NamedTemporaryFile
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
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage, signal, stats, optimize, special, sparse
from scipy.sparse import linalg as sparse_linalg
from tqdm import tqdm
from itrigamma import itrigamma
from . import cconsenrich
from . import ccounts
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

ALIGNMENT_SOURCE_KINDS = ("BAM",)
FRAGMENTS_SOURCE_KIND = "FRAGMENTS"
SUPPORTED_SOURCE_KINDS = ALIGNMENT_SOURCE_KINDS + (FRAGMENTS_SOURCE_KIND,)


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
    With ``fitParams.EM_useIntervalMunc=True``, the outer loop replaces the initial per-replicate
    plugin variances with a shared interval-level track estimated from expected residual squares.


    :param minR: Genome-wide lower bound for replicate-specific observation noise scale.
    :type minR: float | None
    :param maxR: Genome-wide upper bound for the replicate-specific observation noise scale.
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
    :param pad: A small constant added to the observation noise variance estimates for conditioning
    :type pad: float | None
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
    :param conformalRescale: If True, perform replicate-heldout split-conformal calibration for a
        hypothetical new replicate at each genomic interval. The resulting uncertainty is predictive,
        not purely latent-state posterior variance.
    :type conformalRescale: bool | None
    :param conformalAlpha: Target split-conformal miscoverage level :math:`\alpha` in (0, 1)`. When conformalRescale is True,
        the inflation factor is chosen from the :math:`(1-\alpha)` empirical quantile of calibration scores.
    :type conformalAlpha: float | None
    :param conformal_numIters: Number of replicate-heldout calibration rounds used to estimate the inflation factor.
        If set ``> 1``, the routine repeats the replicate split and pools scores across rounds.
    :type conformal_numIters: int | None
    :param conformalFinalRefit: If True, refit the final model after selecting the conformal inflation factor.
        This breaks the strict frozen-predictor split-conformal setup. In all cases, Consenrich only inflates
        predictive uncertainty.
    :type conformalFinalRefit: bool | None
    """

    stateInit: float
    stateCovarInit: float
    boundState: bool
    stateLowerBound: float
    stateUpperBound: float
    conformalRescale: bool | None
    conformalAlpha: float | None
    conformal_numIters: int | None
    conformalFinalRefit: bool | None


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


class inputSource(NamedTuple):
    r"""Describes one input source for counting

    :param path: Path to an alignment or fragments file
    :type path: str
    :param sourceKind: One of ``BAM`` or ``FRAGMENTS``
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
      config may also use ``clusterId`` as a one-group alias
    :type selectGroups: List[str] | None
    :param countMode: Optional counting mode label
      fragments inputs default to `cutsite`
    :type countMode: str | None
    :param fragmentPositionsAreOffset: If True, fragment endpoints are assumed to
      already represent Tn5-adjusted insertion positions
    :type fragmentPositionsAreOffset: bool
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
    fragmentPositionsAreOffset: bool = True


class inputParams(NamedTuple):
    r"""Parameters related to the input data for Consenrich.

    :param bamFiles: A list of paths to distinct coordinate-sorted and indexed BAM files.
    :type bamFiles: List[str]

    :param bamFilesControl: A list of paths to distinct coordinate-sorted and
        indexed control BAM files. e.g., IgG control inputs for ChIP-seq.
    :type bamFilesControl: List[str], optional
    :param pairedEnd: Deprecated: Paired-end/Single-end is inferred automatically from the alignment flags in input BAM files.
    :type pairedEnd: Optional[bool]
    :param treatmentSources: Parsed treatment input sources
    :type treatmentSources: List[inputSource] | None
    :param controlSources: Parsed control input sources
    :type controlSources: List[inputSource] | None
    """

    bamFiles: List[str]
    bamFilesControl: Optional[List[str]]
    pairedEnd: Optional[bool]
    treatmentSources: List[inputSource] | None = None
    controlSources: List[inputSource] | None = None


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
          only appropriate for alignment coverage, not fragments pseudobulks

        - ``SF``: Median of ratios style scale factors (see :func:`consenrich.cconsenrich.cSF`). Restricted to analyses with ``>= 3`` samples (no input control).

        - ``RPKM`` / ``CPM``: Scale factors based on emitted counts per million mapped units
          fragments pseudobulks use emitted insertions rather than raw fragment totals

    :type normMethod: str
    :param fragmentsGroupNorm: Optional extra normalization for fragments pseudobulks
      `NONE` keeps library-size scaling only and `CELLS` additionally divides by selected cell count
    :type fragmentsGroupNorm: str | None
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
    fragmentsGroupNorm: str | None
    fragmentLengths: List[int] | None
    fragmentLengthsControl: List[int] | None
    useTreatmentFragmentLengths: bool | None
    fixControl: bool | None
    globalWeight: float | None
    asymPos: float | None
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
    :param fragmentPositionsAreOffset: If True, fragments are assumed to already encode Tn5-adjusted insertion endpoints
    :type fragmentPositionsAreOffset: bool
    """

    barcodeTag: str | None = "CB"
    defaultCountMode: str | None = "cutsite"
    fragmentsGroupNorm: str | None = "NONE"
    fragmentPositionsAreOffset: bool = True


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
    :param eps: Tolerance parameter for relative maxima detection in the response sequence. Set to zero to enforce strict
        inequalities when identifying discrete relative maxima.
    :type eps: float
    :param massQuantileCutoff: Quantile cutoff for filtering initial (unmerged) matches based on their 'mass' ``((avgSignal*length)/intervalLength)``. To diable, set ``< 0``.
    :type massQuantileCutoff: float
    :param useSplitEmpiricalNull: If True, augment the pooled empirical null with blocked held-out fold nulls
    :type useSplitEmpiricalNull: bool
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
    useSplitEmpiricalNull: bool | None


class outputParams(NamedTuple):
    r"""Parameters related to output files.

    :param convertToBigWig: If True, output bedGraph files are converted to bigWig format.
    :type convertToBigWig: bool
    :param roundDigits: Number of decimal places to round output values (bedGraph)
    :type roundDigits: int
    :param writeUncertainty: If True, write the primary uncertainty track to bedGraph. By default this is
        :math:`\sqrt{\widetilde{P}_{[i,0,0]}}`. With ``conformalRescale=True`` it becomes a calibrated
        future-replicate predictive standard deviation.
    :type writeUncertainty: bool
    :param writeMWSR: If True, write a per-interval mean weighted+squared+*studentized* post-fit residual (``MWSR``),
        computed using the smoothed state and its posterior variance from the final filter/smoother pass.

        Let :math:`g_{[i]}` denote the shared interval background, :math:`b_j` the replicate bias,
        :math:`\widetilde{x}_{[i,0]}` the smoothed signal level, and :math:`\widetilde{P}_{[i,0,0]}`
        the corresponding posterior variance. With observation variance :math:`R_{[j,i]}`, define
        the studentized residual moment

        .. math::

          u^2_{[j,i]} = \frac{(y_{[j,i]} - g_{[i]} - b_j - \widetilde{x}_{[i,0]})^2 + \widetilde{P}_{[i,0,0]}}{R_{[j,i]}}.

        Consenrich then reports the lambda-weighted mean of :math:`u^2_{[j,i]}`, rescaled by the
        Student-:math:`t` reference moment:

        .. math::

          \textsf{MWSR}[i] =
          \frac{\sum_j \lambda_{[j,i]} u^2_{[j,i]}}{\sum_j \lambda_{[j,i]}}
          \cdot \frac{1}{\nu / (\nu - 2)}.
    :type writeMWSR: bool
    :param writeJackknifeSE: If True, write the standard error of the signal level estimates across jackknife replicates to bedGraph. This is only relevant if `applyJackknife` is True.
    :type writeJackknifeSE: bool
    :param applyJackknife: If True, estimate replicate-level sampling variability in the signal level estimates with the jackknife
    :type applyJackknife: bool

    """

    convertToBigWig: bool
    roundDigits: int
    writeUncertainty: bool
    writeMWSR: bool
    writeJackknifeSE: bool
    applyJackknife: bool


class fitParams(NamedTuple):
    r"""Parameters controlling the optimization/fitting procedures.

    These arguments control both the inner routine in :func:`consenrich.cconsenrich.cblockScaleEM`
    and the outer calibration loop in :func:`consenrich.core.runConsenrich`.

    Inner loop:

    1. Filter-smoother state estimation *given* current noise scales
    2. Interval-level Student-t precision reweighting at: \(\lambda_{[j,i]}\) and \(\kappa_{[i]}\)
    3. Block-level process scale updates: \(q_b\)
    4. Replicate-level observation offset/scale updates: \(b_j\) and \(a_j\)

    Outer loop:

    1. update a shared zero-centered background track \(g_i\)
    2. optionally update a shared interval-level plugin variance track \(v_i\)

    Each scale/reweighting resolution is optional. The default fit keeps replicate-level calibration
    and robust precision reweighting on, while leaving the shared interval-level variance update off.


    :param EM_maxIters: Maximum outer EM iterations.
    :type EM_maxIters: int
    :param EM_rtol: Relative improvement tolerance on NLL used in convergence
    :type EM_rtol: float
    :param EM_scaleLOW: Lower clipping bound for block process scales \(q_b\).
    :type EM_scaleLOW: float
    :param EM_scaleHIGH: Upper clipping bound for block process scales \(q_b\).
    :type EM_scaleHIGH: float
    :param EM_alphaEMA: If in ``(0, 1]``, enables log-domain EMA smoothing for block process scales.
    :type EM_alphaEMA: float
    :param EM_scaleToMedian: If True, rescales \(q_b\) by its median after each M-step. This can help preserve between-block scale differences for interpretability, but may interfere with convergence of the EM algorithm since it changes the effective objective.
    :type EM_scaleToMedian: bool
    :param EM_tNu: Student-t df for reweighting strengths (smaller = stronger reweighting)
    :type EM_tNu: float
    :param EM_useProcBlockScale: If True, estimate block process scales \(q_b\); otherwise fix \(q_b\equiv 1\).
    :type EM_useProcBlockScale: bool
    :param EM_useObsPrecReweight: If True, update observation noise precision multipliers \(\lambda_{[j,i]}\) (Student-\(t\) reweighting); otherwise \(\lambda\equiv 1\).
    :type EM_useObsPrecReweight: bool
    :param EM_useProcPrecReweight: If True, update process noise precision multipliers \(\kappa_{[i]}\) (Student-\(t\) reweighting); otherwise \(\kappa\equiv 1\).
    :type EM_useProcPrecReweight: bool
    :param EM_useReplicateBias: If True, estimate additive replicate offsets \(b_j\) in the observation equation.
    :type EM_useReplicateBias: bool
    :param EM_useReplicateScale: If True, estimate multiplicative replicate variance scales \(a_j\) in the observation equation.
    :type EM_useReplicateScale: bool
    :param EM_repBiasShrink: Non-negative shrinkage applied to replicate bias estimates after centering.
    :type EM_repBiasShrink: float
    :param EM_repScaleLOW: Lower clipping bound for replicate variance scales \(a_j\).
    :type EM_repScaleLOW: float
    :param EM_repScaleHIGH: Upper clipping bound for replicate variance scales \(a_j\).
    :type EM_repScaleHIGH: float
    :param EM_outerIters: Number of outer alternations between the inner Kalman-EM fit, shared background update, and interval-level observation variance update.
    :type EM_outerIters: int
    :param EM_outerRtol: Relative tolerance used to stop the outer background/variance loop early.
    :type EM_outerRtol: float
    :param EM_useIntervalMunc: If True, update a shared interval-level observation variance track from smoothed residual second moments.
        Default is False.
    :type EM_useIntervalMunc: bool
    :param EM_intervalMuncEMA: EMA weight used when updating the shared interval-level observation variance track.
    :type EM_intervalMuncEMA: float
    :param EM_useIntervalBackground: If True, update a shared smooth interval-level background with a zero-sum constraint during the outer EM loop.
    :type EM_useIntervalBackground: bool
    :param EM_backgroundSmoothness: Multiplier applied to the second-difference roughness penalty used for the shared background update.
    :type EM_backgroundSmoothness: float


    :seealso: :func:`consenrich.cconsenrich.cblockScaleEM`, :func:`consenrich.core.runConsenrich`, :func:`consenrich.core.getMuncTrack`, :func:`consenrich.core.fitVarianceFunction`
    """

    EM_maxIters: int | None = 50
    EM_rtol: float | None = 1.0e-4
    EM_scaleToMedian: bool | None = False
    EM_tNu: float | None = 10.0
    EM_alphaEMA: float | None = 0.05
    EM_scaleLOW: float | None = 0.5
    EM_scaleHIGH: float | None = 5.0
    EM_useProcBlockScale: bool | None = False
    EM_useObsPrecReweight: bool | None = True
    EM_useProcPrecReweight: bool | None = True
    EM_useReplicateBias: bool | None = True
    EM_useReplicateScale: bool | None = True
    EM_repBiasShrink: float | None = 0.0
    EM_repScaleLOW: float | None = 0.25
    EM_repScaleHIGH: float | None = 4.0
    EM_outerIters: int | None = 3
    EM_outerRtol: float | None = 1.0e-3
    EM_useIntervalMunc: bool | None = False
    EM_intervalMuncEMA: float | None = 0.25
    EM_useIntervalBackground: bool | None = True
    EM_backgroundSmoothness: float | None = 1.0


def _checkMod(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except Exception:
        return False


def _inferAlignmentSourceKind(path: str) -> str:
    if str(path).lower().endswith(".cram"):
        raise ValueError("CRAM inputs are no longer supported.")
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
        raise ValueError(
            f"No barcodes selected for fragments source `{source.path}`"
        )

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
    offsetStr: Optional[str] = "0,0",
    maxInsertSize: Optional[int] = 1000,
    pairedEndMode: Optional[int] = 0,
    inferFragmentLength: Optional[int] = 0,
    countEndsOnly: Optional[bool] = False,
    minMappingQuality: Optional[int] = 0,
    minTemplateLength: Optional[int] = -1,
    fragmentLengths: Optional[List[int]] = None,
) -> npt.NDArray[np.float32]:
    r"""Read binned tracks from generic input sources

    this is the source-agnostic entry point for counting
    """

    if len(sources) == 0:
        raise ValueError("sources list is empty")

    if len(readLengths) != len(sources) or len(scaleFactors) != len(sources):
        raise ValueError("readLengths and scaleFactors must match sources length")

    sourcePaths = getSourcePaths(sources)
    sourceKinds = getSourceKinds(sources)
    offsetParts = ((str(offsetStr) or "0,0").replace(" ", "")).split(",")
    numIntervals = ((end - start - 1) // intervalSizeBP) + 1
    counts = np.empty((len(sources), numIntervals), dtype=np.float32)
    tempPaths: list[str] = []

    if isinstance(fragmentLengths, int):
        fragmentLengths = [fragmentLengths] * len(sources)

    if fragmentLengths is None:
        fragmentLengths = [0] * len(sources)

    if pairedEndMode and fragmentLengths is not None and all(
        sourceKind in ALIGNMENT_SOURCE_KINDS for sourceKind in sourceKinds
    ):
        if isinstance(fragmentLengths, list) and len(fragmentLengths) != len(sources):
            if len(fragmentLengths) == 1:
                fragmentLengths = fragmentLengths * len(sources)
            else:
                raise ValueError(
                    f"`fragmentLengths` length must match `sources` length: {len(fragmentLengths)} != {len(sources)}.",
                )

        pairedEndMode = 0
        inferFragmentLength = 0

    elif pairedEndMode and all(
        sourceKind in ALIGNMENT_SOURCE_KINDS for sourceKind in sourceKinds
    ):
        fragmentLengths = [0] * len(sources)
        inferFragmentLength = 0

    if (
        not pairedEndMode
        and (fragmentLengths is None or len(fragmentLengths) == 0)
        and all(sourceKind in ALIGNMENT_SOURCE_KINDS for sourceKind in sourceKinds)
    ):
        inferFragmentLength = 1
        fragmentLengths = [-1] * len(sources)

    if isinstance(countEndsOnly, bool) and countEndsOnly:
        inferFragmentLength = 0
        pairedEndMode = 0
        fragmentLengths = [0] * len(sources)

    if inferFragmentLength > 0 and all(
        sourceKind in ALIGNMENT_SOURCE_KINDS for sourceKind in sourceKinds
    ) and any(
        fragmentLength <= 0 for fragmentLength in fragmentLengths
    ):
        return readBamSegments(
            bamFiles=sourcePaths,
            chromosome=chromosome,
            start=start,
            end=end,
            intervalSizeBP=intervalSizeBP,
            readLengths=readLengths,
            scaleFactors=scaleFactors,
            oneReadPerBin=oneReadPerBin,
            samThreads=samThreads,
            samFlagExclude=samFlagExclude,
            offsetStr=offsetStr,
            maxInsertSize=maxInsertSize,
            pairedEndMode=pairedEndMode,
            inferFragmentLength=inferFragmentLength,
            countEndsOnly=countEndsOnly,
            minMappingQuality=minMappingQuality,
            minTemplateLength=minTemplateLength,
            fragmentLengths=fragmentLengths,
        )

    try:
        for sourceIndex, sourcePath in enumerate(sourcePaths):
            sourceKind = sourceKinds[sourceIndex]
            source = sources[sourceIndex]
            barcodeAllowListFile, tempPath = _writeFragmentsAllowList(source)
            if tempPath is not None:
                tempPaths.append(tempPath)

            if sourceKind == FRAGMENTS_SOURCE_KIND:
                countMode = str(source.countMode or "").strip().lower()
                if not countMode:
                    countMode = "cutsite"
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
                counts[sourceIndex, :] = ccounts.ccounts_countAlignmentRegion(
                    sourcePath,
                    chromosome,
                    start,
                    end,
                    intervalSizeBP,
                    readLengths[sourceIndex],
                    oneReadPerBin,
                    samThreads,
                    samFlagExclude,
                    shiftForwardStrand53=int(offsetParts[0]),
                    shiftReverseStrand53=int(offsetParts[1]),
                    extendBP=fragmentLengths[sourceIndex],
                    maxInsertSize=maxInsertSize,
                    pairedEndMode=pairedEndMode,
                    inferFragmentLength=inferFragmentLength,
                    minMappingQuality=minMappingQuality,
                    minTemplateLength=minTemplateLength,
                    sourceKind=sourceKind,
                    barcodeAllowListFile="",
                    barcodeGroupMapFile="",
                    countMode="coverage",
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
    blockLenIntervals: int,
    projectStateDuringFiltering: bool = False,
    pad: float = 1.0e-4,
    disableCalibration: bool = False,
    EM_maxIters: int = 50,
    EM_rtol: float = 1.0e-4,
    EM_scaleToMedian: bool = False,
    EM_tNu: float = 10.0,
    EM_alphaEMA: float = 0.05,
    EM_scaleLOW: float = 0.5,
    EM_scaleHIGH: float = 5.0,
    EM_useProcBlockScale: bool = False,
    EM_useObsPrecReweight: bool = True,
    EM_useProcPrecReweight: bool = True,
    EM_useReplicateBias: bool = True,
    EM_useReplicateScale: bool = True,
    EM_repBiasShrink: float = 0.0,
    EM_repScaleLOW: float = 0.25,
    EM_repScaleHIGH: float = 4.0,
    EM_outerIters: int = 3,
    EM_outerRtol: float = 1.0e-3,
    EM_useIntervalMunc: bool = False,
    EM_intervalMuncEMA: float = 0.25,
    EM_useIntervalBackground: bool = True,
    EM_backgroundSmoothness: float = 1.0,
    intervalMuncBinQuantileCutoff: float = 0.5,
    intervalMuncEB_minLin: float = 1.0,
    intervalMuncEB_use: bool = True,
    intervalMuncEB_setNu0: int | None = None,
    intervalMuncEB_setNuL: int | None = None,
    returnScales: bool = True,
    returnReplicateOffsets: bool = False,
    applyJackknife: bool = False,
    jackknifeEM_maxIters: int = 5,
    jackknifeEM_rtol: float = 1.0e-2,
    useWhiteAccel: bool = False,
    useDiscreteConstAccel: bool = False,
    autoDeltaF: bool = True,
    autoDeltaF_low: float = 1.0e-4,
    autoDeltaF_high: float = 2.0,
    autoDeltaF_init: float = 0.01,
    autoDeltaF_maxEvals: int = 25,
    conformalRescale: bool = False,
    conformalAlpha: float = 0.05,
    conformal_numIters: int = 3,
    conformalFinalRefit: bool = True,
):
    r"""Run Consenrich over a contiguous genomic region

    Consenrich estimates a shared signal level from multiple replicate tracks using a two-state
    linear smoother plus an outer calibration loop.

    The observation model is

    .. math::

      y_{[j,i]} = g_{[i]} + x_{[i,0]} + b_j + \epsilon_{[j,i]},
      \qquad
      \mathrm{Var}(\epsilon_{[j,i]}) =
      \frac{a_j (v_{[j,i]} + \mathrm{pad})}{\lambda_{[j,i]}}.

    Here :math:`g_{[i]}` is a shared zero-centered smooth background, :math:`b_j` and :math:`a_j`
    are replicate-level bias and variance factors, and :math:`v_{[j,i]}` is the plugin observation
    variance supplied by ``matrixMunc`` or by the outer interval-level variance update. When
    ``EM_useIntervalMunc=True``, the outer variance update reuses the same EB shrinkage settings
    used to build the initial observation-noise track, unless the caller overrides them here.

    The latent state follows

    .. math::

      \mathbf{x}_{[i+1]} = \mathbf{F}(\delta_F)\mathbf{x}_{[i]} + \eta_{[i]},
      \qquad
      \mathrm{Var}(\eta_{[i]}) = \frac{q_{b(i)} \mathbf{Q}_0}{\kappa_{[i]}}.

    If ``conformalRescale=True``, the primary variance output is calibrated for future-replicate
    prediction rather than interpreted as a pure latent-state posterior variance.

    This wrapper ties together several fundamental routines written in Cython:

    #. :func:`consenrich.cconsenrich.cforwardPass`: Forward filter (predict, update)
    #. :func:`consenrich.cconsenrich.cbackwardPass`: Backward fixed-interval smoother
    #. :func:`consenrich.cconsenrich.cblockScaleEM`: Joint optimization of process and observation model noise scales with robust studentized reweighting

    :seealso: :func:`consenrich.core.getMuncTrack`, :func:`consenrich.cconsenrich.cTransform`, :func:`consenrich.cconsenrich.cforwardPass`, :func:`consenrich.cconsenrich.cbackwardPass`, :func:`consenrich.cconsenrich.cblockScaleEM`
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

    if conformalRescale and applyJackknife:
        raise ValueError(
            "`conformalRescale` is mutually exclusive with `applyJackknife`"
        )

    blockCount = int(np.ceil(intervalCount / float(blockLenIntervals)))
    intervalToBlockMap = (
        np.arange(intervalCount, dtype=np.int32) // blockLenIntervals
    ).astype(np.int32)
    intervalToBlockMap[intervalToBlockMap >= blockCount] = blockCount - 1

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
        nLocal = int(matrixDataLocal.shape[1])
        mLocal = int(matrixDataLocal.shape[0])

        # Note reuse of buffers, each score requires a single filter-smoother iter
        stateForward = np.empty((nLocal, 2), dtype=np.float32)
        stateCovarForward = np.empty((nLocal, 2, 2), dtype=np.float32)
        pNoiseForward = np.empty((nLocal, 2, 2), dtype=np.float32)
        vectorD = np.empty(nLocal, dtype=np.float32)

        stateSmoothed = np.empty((nLocal, 2), dtype=np.float32)
        stateCovarSmoothed = np.empty((nLocal, 2, 2), dtype=np.float32)
        lagCovSmoothed = np.empty((max(nLocal - 1, 1), 2, 2), dtype=np.float32)
        postFitResiduals = np.empty((nLocal, mLocal), dtype=np.float32)

        # ------------------------------------------------------------
        # 'auto' deltaF section
        #
        #   score(deltaF) = NLL(deltaF) + [intervalCount * log(eps + roughness(deltaF))]
        #
        # NLL(deltaF):
        #   Gaussian forward-pass negative log likelihood under fixed qScale=1, no reweighting
        #
        # roughness(deltaF):
        #   Expected squared change in the signal level between adjacent intervals:
        #
        #   E[(x0_{k+1} - x0_k)^2]
        #     = (mu0_{k+1} - mu0_k)^2 + P00_{k+1} + P00_k - 2*C00_{k,k+1}
        #
        # Intuition:
        # - NLL favors matching the observed tracks
        # - roughness penalizes back-and-forth sensitivity to noise
        # - log(.) makes the penalty less dependent on absolute scale
        #
        # ------------------------------------------------------------
        def _penNLL(deltaF_candidate: float):
            deltaF_candidate = float(deltaF_candidate)
            if (not np.isfinite(deltaF_candidate)) or deltaF_candidate <= 0.0:
                return float(1.0e16), float(1.0e16)

            try:
                # update these during search since they're partially determined by deltaF
                matrixF_candidate = buildMatrixF(deltaF_candidate)
                matrixQ0_candidate = buildMatrixQ0(deltaF_candidate)
            except Exception:
                return float(1.0e16), float(1.0e16)

            try:
                out = cconsenrich.cforwardPass(
                    matrixData=matrixDataLocal,
                    matrixPluginMuncInit=matrixMuncLocal,
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
                    # Force the "Gaussian fixed-template" setting for auto-deltaF:
                    EM_useProcBlockScale=False,
                    EM_useObsPrecReweight=False,
                    EM_useProcPrecReweight=False,
                )
                sumNLL = float(out[3])

                cconsenrich.cbackwardPass(
                    matrixData=matrixDataLocal,
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

                if nLocal <= 1:
                    return float(sumNLL), 0.0

                mu = stateSmoothed.astype(np.float64, copy=False)
                P = stateCovarSmoothed.astype(np.float64, copy=False)
                C = lagCovSmoothed.astype(np.float64, copy=False)

                L = nLocal - 1
                deltaMu0 = mu[1:, 0] - mu[:-1, 0]

                # expected roughness
                expDelta2 = (
                    (deltaMu0 * deltaMu0)
                    + P[1:, 0, 0]
                    + P[:-1, 0, 0]
                    - 2.0 * C[:L, 0, 0]
                )
                expDelta2 = np.maximum(expDelta2, 0.0)
                roughnessMean = float(np.mean(expDelta2))

                return float(sumNLL), float(roughnessMean)
            except Exception:
                return float(1.0e16), float(1.0e16)

        # Search over t = log(deltaF)
        tLOW = float(np.log(deltaFMin))
        tHIGH = float(np.log(deltaFMax))

        # warm start grid and set scale refs so w1,w2 are meaningful:
        gridDeltaF = np.exp(np.linspace(tLOW, tHIGH, num=16, dtype=np.float64))
        gridTerms = []
        for d in gridDeltaF:
            sumNLL_g, rough_g = _penNLL(float(d))
            if sumNLL_g >= 1.0e16 or rough_g >= 1.0e16:
                continue

            nll_per_obs = sumNLL_g / (float(mLocal) * float(nLocal))
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
            #   score = w1*(nll_per_obs / nll_ref) + w2*(log1p(roughness)/rough_ref)
            nll_term = (sumNLL / (float(mLocal) * float(nLocal))) / nll_ref
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
            f"autoDeltaF search completed: bestDeltaF={bestDeltaF}\tbestScore={bestScore:.4e}"
        )

        return float(bestDeltaF)

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
        replicateScale: np.ndarray | None,
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
            replicateScale=replicateScale,
            EM_useProcBlockScale=bool(EM_useProcBlockScale),
            EM_useObsPrecReweight=bool(EM_useObsPrecReweight),
            EM_useProcPrecReweight=bool(EM_useProcPrecReweight),
            EM_useReplicateBias=bool(EM_useReplicateBias),
            EM_useReplicateScale=bool(EM_useReplicateScale),
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
            stateSmoothed,
            stateCovarSmoothed,
            lagCovSmoothed,
            postFitResiduals,
            NIS,
        )

    def _fitCalibratedModel(
        *,
        matrixDataLocal: np.ndarray,
        matrixMuncLocal: np.ndarray,
        matrixFLocal: np.ndarray,
        matrixQ0Local: np.ndarray,
        emMaxItersLocal: int,
        emRtolLocal: float,
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
            replicateScaleLocal = np.ones(mLocal, dtype=np.float32)
            (
                _phiHatLocal,
                sumNLLLocal,
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
                replicateScale=replicateScaleLocal,
            )
            return {
                "matrixMunc": currentMunc,
                "background": currentBackground,
                "qScale": qScaleLocal,
                "lambdaExp": lambdaExpLocal,
                "processPrecExp": processPrecExpLocal,
                "replicateBias": replicateBiasLocal,
                "replicateScale": replicateScaleLocal,
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
        if bool(EM_useIntervalMunc):
            sharedTrack = np.mean(matrixMuncLocal, axis=0, dtype=np.float64).astype(
                np.float32, copy=False
            )
            currentMunc = np.broadcast_to(sharedTrack[None, :], (mLocal, nLocal)).copy()
        else:
            currentMunc = np.ascontiguousarray(matrixMuncLocal, dtype=np.float32)

        lambdaExpLocal = None
        processPrecExpLocal = None
        qScaleLocal = np.ones(blockCount, dtype=np.float32)
        replicateBiasLocal = np.zeros(mLocal, dtype=np.float32)
        replicateScaleLocal = np.ones(mLocal, dtype=np.float32)
        stateSmoothedLocal = None
        stateCovarSmoothedLocal = None
        lagCovSmoothedLocal = None
        postFitResidualsLocal = None
        NISLocal = None
        sumNLLLocal = np.nan

        outerIters = max(1, int(EM_outerIters))
        outerAlpha = float(np.clip(EM_intervalMuncEMA, 0.0, 1.0))
        outerTol = float(max(EM_outerRtol, 0.0))

        for outerIter in range(outerIters):
            dataAdjusted = np.ascontiguousarray(
                matrixDataLocal - currentBackground[None, :],
                dtype=np.float32,
            )
            EM_out_local = cconsenrich.cblockScaleEM(
                matrixData=dataAdjusted,
                matrixPluginMuncInit=currentMunc,
                matrixF=matrixFLocal,
                matrixQ0=matrixQ0Local,
                intervalToBlockMap=intervalToBlockMap,
                blockCount=int(blockCount),
                stateInit=float(stateInit),
                stateCovarInit=float(stateCovarInit),
                EM_maxIters=int(emMaxItersLocal),
                EM_rtol=float(emRtolLocal),
                pad=float(pad),
                EM_scaleLOW=float(EM_scaleLOW),
                EM_scaleHIGH=float(EM_scaleHIGH),
                EM_alphaEMA=float(EM_alphaEMA),
                EM_scaleToMedian=bool(EM_scaleToMedian),
                EM_tNu=float(EM_tNu),
                returnIntermediates=True,
                EM_useProcBlockScale=bool(EM_useProcBlockScale),
                EM_useObsPrecReweight=bool(EM_useObsPrecReweight),
                EM_useProcPrecReweight=bool(EM_useProcPrecReweight),
                EM_useReplicateBias=bool(EM_useReplicateBias),
                EM_useReplicateScale=bool(EM_useReplicateScale),
                EM_repBiasShrink=float(EM_repBiasShrink),
                EM_repScaleLOW=float(EM_repScaleLOW),
                EM_repScaleHIGH=float(EM_repScaleHIGH),
            )
            if len(EM_out_local) != 11:
                raise ValueError(
                    "Expected cblockScaleEM(..., returnIntermediates=True) to return 11 values "
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
                replicateScaleLocal,
            ) = EM_out_local

            qScaleLocal = np.asarray(qScaleLocal, dtype=np.float32)
            if lambdaExpLocal is not None:
                lambdaExpLocal = np.asarray(lambdaExpLocal, dtype=np.float32)
            if processPrecExpLocal is not None:
                processPrecExpLocal = np.asarray(processPrecExpLocal, dtype=np.float32)
            replicateBiasLocal = np.asarray(replicateBiasLocal, dtype=np.float32)
            replicateScaleLocal = np.asarray(replicateScaleLocal, dtype=np.float32)
            stateSmoothedLocal = np.asarray(stateSmoothedLocal, dtype=np.float32)
            stateCovarSmoothedLocal = np.asarray(
                stateCovarSmoothedLocal, dtype=np.float32
            )
            lagCovSmoothedLocal = np.asarray(lagCovSmoothedLocal, dtype=np.float32)
            postFitResidualsLocal = np.asarray(postFitResidualsLocal, dtype=np.float32)

            nextBackground = currentBackground
            if bool(EM_useIntervalBackground):
                invVarMatrix = 1.0 / (
                    np.maximum(replicateScaleLocal[:, None], 1.0e-8)
                    * np.maximum(currentMunc, 1.0e-8)
                )
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
                )

            nextMunc = currentMunc
            if bool(EM_useIntervalMunc):
                nextTrack = _estimateIntervalMuncTrack(
                    matrixData=matrixDataLocal,
                    stateSmoothed=stateSmoothedLocal,
                    stateCovarSmoothed=stateCovarSmoothedLocal,
                    replicateBias=replicateBiasLocal,
                    replicateScale=replicateScaleLocal,
                    backgroundTrack=nextBackground,
                    lambdaExp=lambdaExpLocal,
                    pad=float(pad),
                    minR=float(np.maximum(1.0e-8, np.nanmin(currentMunc))),
                    maxR=float(np.maximum(np.nanmax(currentMunc), np.nanmin(currentMunc))),
                    binQuantileCutoff=float(intervalMuncBinQuantileCutoff),
                    EB_minLin=float(intervalMuncEB_minLin),
                    EB_use=bool(intervalMuncEB_use),
                    EB_setNu0=intervalMuncEB_setNu0,
                    EB_setNuL=intervalMuncEB_setNuL,
                )
                if outerAlpha < 1.0:
                    currentTrack = np.asarray(currentMunc[0, :], dtype=np.float32)
                    nextTrack = (
                        ((1.0 - outerAlpha) * currentTrack)
                        + (outerAlpha * nextTrack)
                    ).astype(np.float32, copy=False)
                nextMunc = np.broadcast_to(nextTrack[None, :], (mLocal, nLocal)).copy()

            bgChange = float(
                np.max(np.abs(np.asarray(nextBackground) - np.asarray(currentBackground)))
            )
            muncChange = float(
                np.max(
                    np.abs(
                        np.log(np.maximum(nextMunc[0, :], 1.0e-8))
                        - np.log(np.maximum(currentMunc[0, :], 1.0e-8))
                    )
                )
            )
            currentBackground = np.asarray(nextBackground, dtype=np.float32)
            currentMunc = np.ascontiguousarray(nextMunc, dtype=np.float32)
            logger.info(
                "outerEM[%d/%d]: backgroundShift=%.6g muncShift=%.6g",
                int(outerIter + 1),
                int(outerIters),
                float(bgChange),
                float(muncChange),
            )
            if max(bgChange, muncChange) <= outerTol:
                break

        dataAdjusted = np.ascontiguousarray(
            matrixDataLocal - currentBackground[None, :],
            dtype=np.float32,
        )
        (
            _phiHatLocal,
            sumNLLLocal,
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
            replicateScale=replicateScaleLocal,
        )
        return {
            "matrixMunc": currentMunc,
            "background": currentBackground,
            "qScale": np.asarray(qScaleLocal, dtype=np.float32),
            "lambdaExp": lambdaExpLocal,
            "processPrecExp": processPrecExpLocal,
            "replicateBias": np.asarray(replicateBiasLocal, dtype=np.float32),
            "replicateScale": np.asarray(replicateScaleLocal, dtype=np.float32),
            "stateSmoothed": np.asarray(stateSmoothedLocal, dtype=np.float32),
            "stateCovarSmoothed": np.asarray(stateCovarSmoothedLocal, dtype=np.float32),
            "lagCovSmoothed": np.asarray(lagCovSmoothedLocal, dtype=np.float32),
            "postFitResiduals": np.asarray(postFitResidualsLocal, dtype=np.float32),
            "NIS": np.asarray(NISLocal, dtype=np.float32),
            "sumNLL": float(sumNLLLocal),
        }

    def _conformal_FutureRepQHat(
        deltaFBase: float,
    ) -> tuple[float, np.ndarray | None, float]:
        mLocal = int(trackCount)
        if mLocal < 2:
            logger.warning("conformalRescale: trackCount < 2 --> setting qhat=1.0")
            return 1.0, None, float(deltaFBase)

        alphaLocal = float(conformalAlpha)
        if alphaLocal <= 0.0 or alphaLocal >= 1.0:
            logger.warning(
                "conformalRescale: invalid conformalAlpha=%.3g --> using default 0.1",
                float(conformalAlpha),
            )
            alphaLocal = 0.1
        alphaLocal = float(np.clip(alphaLocal, 1.0e-6, 0.5))

        rng = np.random.default_rng()
        all_scores = []
        idxT0 = None
        deltaF0 = float(deltaFBase)

        iters = int(max(1, int(conformal_numIters)))
        for _ in range(iters):
            perm = rng.permutation(mLocal).astype(np.int32, copy=False)
            splitIdx = int(mLocal // 2)
            idxT = perm[:splitIdx]
            idxC = perm[splitIdx:]

            if idxT0 is None:
                idxT0 = np.asarray(idxT, dtype=np.int32).copy()

            if idxT.size < 1 or idxC.size < 1:
                continue

            matrixDataT = np.ascontiguousarray(matrixData[idxT, :], dtype=np.float32)
            matrixMuncT = np.ascontiguousarray(matrixMunc[idxT, :], dtype=np.float32)

            if inferDeltaF and (not conformalFinalRefit):
                deltaF_iter = _autoDeltaF(matrixDataT, matrixMuncT)
            else:
                deltaF_iter = float(deltaFBase)

            if idxT0 is not None and idxT0.size == idxT.size and np.all(idxT0 == idxT):
                deltaF0 = float(deltaF_iter)

            try:
                matrixF_iter = buildMatrixF(float(deltaF_iter))
                matrixQ0_iter = buildMatrixQ0(float(deltaF_iter))
            except Exception:
                continue

            fitT = _fitCalibratedModel(
                matrixDataLocal=matrixDataT,
                matrixMuncLocal=matrixMuncT,
                matrixFLocal=matrixF_iter,
                matrixQ0Local=matrixQ0_iter,
                emMaxItersLocal=int(min(int(EM_maxIters), 25)),
                emRtolLocal=float(EM_rtol),
            )

            muT = np.asarray(fitT["stateSmoothed"], dtype=np.float32)[:, 0].astype(
                np.float64, copy=False
            )
            vT = np.asarray(fitT["stateCovarSmoothed"], dtype=np.float32)[:, 0, 0].astype(
                np.float64, copy=False
            )
            vT = np.maximum(vT, 0.0)

            conformalMaxPoints = 100_000
            for j in idxC:
                y = matrixData[int(j), :].astype(np.float64, copy=False)
                Rj = matrixMunc[int(j), :].astype(np.float64, copy=False)
                futVar = np.sqrt(np.maximum(vT + Rj + float(pad), 1.0e-8))
                s = np.abs(y - muT) / futVar
                if s.size > int(conformalMaxPoints):
                    stride = int(np.ceil(s.size / float(conformalMaxPoints)))
                    s = s[::stride]
                if s.size:
                    all_scores.append(s.astype(np.float64, copy=False))

        if not all_scores:
            logger.warning("conformalRescale: no finite scores --> setting qhat=1.0")
            return 1.0, idxT0, float(deltaF0)

        scores = np.concatenate(all_scores, axis=0)
        scores = scores[np.isfinite(scores)]

        conformalMaxPoints = 100_000
        if scores.size > int(conformalMaxPoints):
            stride = int(np.ceil(scores.size / float(conformalMaxPoints)))
            scores = scores[::stride]

        N = int(scores.size)
        if N <= 0:
            logger.warning(
                "conformalRescale: empty scores after filtering --> setting qhat=1.0"
            )
            return 1.0, idxT0, float(deltaF0)

        k = int(np.ceil((N + 1) * (1.0 - alphaLocal)))
        q_level = min(1.0, max(0.0, k / float(N)))

        try:
            qhat = float(np.quantile(scores, q_level, method="higher"))
        except TypeError:
            qhat = float(np.quantile(scores, q_level, interpolation="higher"))

        if (not np.isfinite(qhat)) or qhat <= 0.0:
            qhat = 1.0
        qhat = float(max(1.0, qhat))

        logger.info(
            "conformalRescale(future_replicate): alpha=%.3g N=%d qhat=%.6g",
            float(alphaLocal),
            int(N),
            float(qhat),
        )
        return float(qhat), idxT0, float(deltaF0)

    conformalQhat = 1.0
    conformalIdxT = None
    deltaF_fit = float(deltaF)

    if conformalRescale:
        if (not np.isfinite(deltaF_fit)) or (float(deltaF_fit) <= 0.0):
            deltaF_fit = float(autoDeltaF_init)
        deltaF_fit = float(max(deltaF_fit, 1.0e-8))
        if inferDeltaF and conformalFinalRefit:
            deltaF = _autoDeltaF(matrixData, matrixMunc)
            deltaF_fit = float(deltaF)
        conformalQhat, conformalIdxT, deltaF0 = _conformal_FutureRepQHat(deltaF_fit)
        if (
            (not conformalFinalRefit)
            and (conformalIdxT is not None)
            and (conformalIdxT.size >= 1)
        ):
            deltaF_fit = float(deltaF0)
    else:
        if inferDeltaF:
            deltaF = _autoDeltaF(matrixData, matrixMunc)
        deltaF_fit = float(deltaF)

    matrixF = buildMatrixF(float(deltaF_fit))
    matrixQ0 = buildMatrixQ0(float(deltaF_fit))

    matrixDataFit = matrixData
    matrixMuncFit = matrixMunc
    if (
        conformalRescale
        and (not conformalFinalRefit)
        and (conformalIdxT is not None)
        and (conformalIdxT.size >= 1)
    ):
        matrixDataFit = np.ascontiguousarray(
            matrixData[conformalIdxT, :], dtype=np.float32
        )
        matrixMuncFit = np.ascontiguousarray(
            matrixMunc[conformalIdxT, :], dtype=np.float32
        )
        logger.info(
            "conformalRescale with conformalFinalRefit=False: final fit uses %d/%d replicates, so downstream shape/size outputs may differ from original trackCount=%d",
            int(conformalIdxT.size),
            int(trackCount),
            int(trackCount),
        )

    fitFinal = _fitCalibratedModel(
        matrixDataLocal=matrixDataFit,
        matrixMuncLocal=matrixMuncFit,
        matrixFLocal=matrixF,
        matrixQ0Local=matrixQ0,
        emMaxItersLocal=int(EM_maxIters),
        emRtolLocal=float(EM_rtol),
    )
    lambdaExp_final = fitFinal["lambdaExp"]
    processPrecExp_final = fitFinal["processPrecExp"]
    replicateBias_final = np.asarray(fitFinal["replicateBias"], dtype=np.float32)
    replicateScale_final = np.asarray(fitFinal["replicateScale"], dtype=np.float32)
    qScale = np.asarray(fitFinal["qScale"], dtype=np.float32)
    matrixMuncFit = np.asarray(fitFinal["matrixMunc"], dtype=np.float32)
    sumNLL = float(fitFinal["sumNLL"])
    stateSmoothed = np.asarray(fitFinal["stateSmoothed"], dtype=np.float32)
    stateCovarSmoothed = np.asarray(fitFinal["stateCovarSmoothed"], dtype=np.float32)
    postFitResiduals = np.asarray(fitFinal["postFitResiduals"], dtype=np.float32)
    NIS = np.asarray(fitFinal["NIS"], dtype=np.float32)

    outStateSmoothed = np.asarray(stateSmoothed, dtype=np.float32)
    outStateCovarSmoothed = np.asarray(stateCovarSmoothed, dtype=np.float32)
    outPostFitResiduals = np.asarray(postFitResiduals, dtype=np.float32)

    fullState0 = outStateSmoothed[:, 0].astype(np.float64, copy=False)
    outTrack4 = NIS

    # Jackknife:
    # If deltaF was inferred, each leave-one-out fit re-runs the same deltaF search
    # on the reduced replicate set. If deltaF was provided, deltaF stays fixed.
    # - note conformal/jackknife are mutually exclusive

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

            fitLOO = _fitCalibratedModel(
                matrixDataLocal=matrixData_LOO,
                matrixMuncLocal=matrixMunc_LOO,
                matrixFLocal=matrixF_LOO,
                matrixQ0Local=matrixQ0_LOO,
                emMaxItersLocal=int(jackknifeEM_maxIters),
                emRtolLocal=float(jackknifeEM_rtol),
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

    if conformalRescale and conformalQhat != 1.0:
        fittedObsVar = (
            np.asarray(replicateScale_final, dtype=np.float32).reshape(-1, 1)
            * (np.asarray(matrixMuncFit, dtype=np.float32) + np.float32(pad))
        )
        refObsVar = np.nanmedian(fittedObsVar, axis=0).astype(np.float32, copy=False)
        refObsVar = np.maximum(refObsVar, np.float32(0.0))
        predictiveVar = (
            outStateCovarSmoothed[:, 0, 0].astype(np.float32, copy=False) + refObsVar
        )
        predictiveVar = np.maximum(predictiveVar, np.float32(1.0e-12))
        outStateCovarSmoothed[:, 0, 0] = predictiveVar * np.float32(
            conformalQhat * conformalQhat
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
                np.asarray(replicateScale_final, dtype=np.float32),
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
    binQuantileCutoff: float = 0.5,
    EB_minLin: float = 1.0,
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
    EB_minLin: float = 0.01,
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


def _solveZeroCenteredBackground(
    residualMatrix: np.ndarray,
    invVarMatrix: np.ndarray,
    blockLenIntervals: int,
    backgroundSmoothness: float = 1.0,
) -> npt.NDArray[np.float32]:
    residualArr = np.asarray(residualMatrix, dtype=np.float64)
    invVarArr = np.asarray(invVarMatrix, dtype=np.float64)
    if residualArr.ndim != 2 or invVarArr.shape != residualArr.shape:
        raise ValueError("residualMatrix and invVarMatrix must have identical 2D shapes")

    intervalCount = int(residualArr.shape[1])
    if intervalCount < 1:
        return np.zeros(0, dtype=np.float32)

    weightTrack = np.sum(invVarArr, axis=0)
    rhsTrack = np.sum(invVarArr * residualArr, axis=0)
    if not np.any(weightTrack > 0.0):
        return np.zeros(intervalCount, dtype=np.float32)

    systemMat = sparse.diags(weightTrack, offsets=0, format="csr")
    penaltyMat = _buildSecondDiffPenalty(intervalCount)
    lam = _backgroundPenaltyFromSpan(
        blockLenIntervals=blockLenIntervals,
        backgroundSmoothness=backgroundSmoothness,
    )
    systemMat = systemMat + (lam * penaltyMat)

    constraintVec = np.ones(intervalCount, dtype=np.float64)
    kktMat = sparse.bmat(
        [
            [systemMat, constraintVec[:, None]],
            [constraintVec[None, :], None],
        ],
        format="csc",
    )
    rhsVec = np.concatenate([rhsTrack, np.zeros(1, dtype=np.float64)])
    solutionVec = sparse_linalg.spsolve(kktMat, rhsVec)
    backgroundTrack = np.asarray(solutionVec[:-1], dtype=np.float64)
    backgroundTrack -= np.mean(backgroundTrack, dtype=np.float64)
    return backgroundTrack.astype(np.float32, copy=False)


def _estimateIntervalMuncTrack(
    matrixData: np.ndarray,
    stateSmoothed: np.ndarray,
    stateCovarSmoothed: np.ndarray,
    replicateBias: np.ndarray,
    replicateScale: np.ndarray,
    backgroundTrack: np.ndarray,
    lambdaExp: np.ndarray | None,
    pad: float = 1.0e-4,
    minR: float = 1.0e-3,
    maxR: float = 1.0e4,
    binQuantileCutoff: float = 0.5,
    EB_minLin: float = 1.0,
    EB_use: bool = True,
    EB_setNu0: int | None = None,
    EB_setNuL: int | None = None,
) -> npt.NDArray[np.float32]:
    dataArr = np.asarray(matrixData, dtype=np.float64)
    stateArr = np.asarray(stateSmoothed, dtype=np.float64)
    covArr = np.asarray(stateCovarSmoothed, dtype=np.float64)
    biasArr = np.asarray(replicateBias, dtype=np.float64).reshape(-1, 1)
    scaleArr = np.asarray(replicateScale, dtype=np.float64).reshape(-1, 1)
    backgroundArr = np.asarray(backgroundTrack, dtype=np.float64).reshape(1, -1)

    if dataArr.ndim != 2:
        raise ValueError("matrixData must be 2D")
    if stateArr.shape[0] != dataArr.shape[1]:
        raise ValueError("stateSmoothed must align with interval axis")

    scaleArr = np.maximum(scaleArr, 1.0e-8)

    stateLevel = stateArr[:, 0].reshape(1, -1)
    stateVar = np.maximum(covArr[:, 0, 0], 0.0).reshape(1, -1)
    residualSq = (dataArr - backgroundArr - biasArr - stateLevel) ** 2
    residualSq += stateVar

    if lambdaExp is None:
        lambdaArr = np.ones_like(dataArr, dtype=np.float64)
    else:
        lambdaArr = np.asarray(lambdaExp, dtype=np.float64)

    rawVarTrack = np.mean(
        lambdaArr * residualSq / scaleArr,
        axis=0,
        dtype=np.float64,
    )
    rawVarTrack = np.maximum(rawVarTrack, max(float(minR), 1.0e-8)) + float(pad)

    ampTrack = np.abs(
        np.asarray(stateSmoothed[:, 0], dtype=np.float64)
        + np.asarray(backgroundTrack, dtype=np.float64)
    )
    mask = np.isfinite(ampTrack) & np.isfinite(rawVarTrack) & (rawVarTrack > 0.0)
    if np.count_nonzero(mask) < 4:
        clippedTrack = np.clip(rawVarTrack, float(minR), float(maxR))
        return clippedTrack.astype(np.float32, copy=False)

    ampMasked = ampTrack[mask]
    rawMasked = rawVarTrack[mask]
    order = np.argsort(ampMasked)
    priorCoef = fitVarianceFunction(
        ampMasked[order],
        rawMasked[order],
        eps=float(pad),
        binQuantileCutoff=float(binQuantileCutoff),
        EB_minLin=float(EB_minLin),
    )
    priorTrack = evalVarianceFunction(
        priorCoef,
        ampTrack,
        eps=float(pad),
        EB_minLin=float(EB_minLin),
    ).astype(np.float64, copy=False)

    if not EB_use:
        clippedTrack = np.clip(priorTrack, float(minR), float(maxR))
        return clippedTrack.astype(np.float32, copy=False)

    if EB_setNuL is not None and EB_setNuL > 3:
        nuLocal = float(EB_setNuL)
    else:
        nuLocal = float(max(4, dataArr.shape[0]))

    if EB_setNu0 is not None and EB_setNu0 > 4:
        nuPrior = float(EB_setNu0)
    else:
        nuPrior = EB_computePriorStrength(rawMasked, priorTrack[mask], nuLocal)

    posteriorTrack = np.array(priorTrack, dtype=np.float64, copy=True)
    posteriorTrack[mask] = (
        (nuLocal * rawVarTrack[mask]) + (nuPrior * priorTrack[mask])
    ) / (nuLocal + nuPrior)
    posteriorTrack = np.clip(posteriorTrack, float(minR), float(maxR))
    return posteriorTrack.astype(np.float32, copy=False)


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
    EB_minLin: float = 1.0,
    EB_use: bool = True,
    EB_setNu0: int | None = None,
    EB_setNuL: int | None = None,
    EB_localQuantile: float = 0.0,
    verbose: bool = False,
    eps: float = 1.0e-2,
) -> tuple[npt.NDArray[np.float32], float]:
    r"""Approximate initial sample-specific (**M**)easurement (**unc**)ertainty tracks

    For an individual experimental sample (replicate), quantify *positional* data uncertainty over genomic intervals :math:`i=1,2,\ldots n` spanning ``chromosome``.
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

    meanTrack = valuesArr.astype(np.float32, copy=True)
    if useEMA:
        meanTrack = cconsenrich.cEMA(meanTrack, 2 / (localWindowIntervals + 1))
    meanTrack = np.abs(meanTrack)
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
        widthBoot = _estimateLocalPeakWidth(localYBoot, localPeakIdx, peakSearchRadius=2)
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
    r"""Estimate a characteristic feature width from local peak widths

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
