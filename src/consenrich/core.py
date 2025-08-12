# -*- coding: utf-8 -*-
r"""
Consenrich core functions and classes.

"""

import logging
from tempfile import NamedTemporaryFile
from typing import Callable, List, Optional, Tuple, DefaultDict, Any, NamedTuple
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import pybedtools as bed
from scipy import signal, ndimage

from . import cconsenrich

logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(module)s.%(funcName)s -  %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class processParams(NamedTuple):
    r"""Parameters related to the process model of Consenrich.
    The process model governs the signal and variance propagation
    through the state transition :math:`\mathbf{F} \in \mathbb{R}^{2 \times 2}`
    and process noise covariance :math:`\mathbf{Q}_{[i]} \in \mathbb{R}^{2 \times 2}`
    matrices.


    :param deltaF: Propagation length. Informally, how far forward/backward to project the estimate and covariance
        at the previous genomic interval to obtain the initial prediction of the state and covariance at
        the current genomic interval :math:`i`: :math:`x_{[i|i-1]}` and covariance :math:`\mathbf{P}_{[i|i-1]}`.
    :type deltaF: float
    :param minQ: Minimum process noise variance (diagonal in :math:`\mathbf{Q}_{[i]}`)
        for each state variable.
    :type minQ: float
    :param maxQ: Maximum process noise variance (diagonal in :math:`\mathbf{Q}_{[i]}`)
        for each state variable.
    :type maxQ: float
    :param offDiagQ: Initial off-diagonal noise covariance between states.
    :type offDiagQ: float
    :param dStatAlpha: Innovation-based model mismatch threshold :math:`\alpha_D`.
        If we observe :math:`D_{[i]} > \alpha_D`, we consider the process model
        to be unreliable and therefore scale-up the process noise covariance to
        favor the observation model (the data) instead.
    :type dStatAlpha: float
    :param dStatd: Constant :math:`d` in the scaling expression :math:`\sqrt{d|D_{[i]} - \alpha_D| + c}`
        that is used to up/down-scale the process noise covariance in the event of a model mismatch.
    :type dStatd: float
    :param dStatPC: Constant :math:`c` in the scaling expression :math:`\sqrt{d|D_{[i]} - \alpha_D| + c}`
        that is used to up/down-scale the process noise covariance in the event of a model mismatch.
    :type dStatPC: float
    """
    deltaF: float
    minQ: float
    maxQ: float
    offDiagQ: float
    dStatAlpha: float
    dStatd: float
    dStatPC: float


class observationParams(NamedTuple):
    r"""Parameters related to the observation model of Consenrich.
    The observation model is used to integrate sequence alignment count
    data from the multiple input samples and account for region-and-sample-specific
    noise processes corrupting data. The observation model matrix
    :math:`\mathbf{H} \in \mathbb{R}^{m \times 2}` maps from the state dimension (2)
    to the dimension of measurements/data (:math:`m`).

    :param minR: The minimum observation noise variance for each sample
        :math:`j=1\ldots m` in the observation noise covariance
        matrix :math:`\mathbf{R}_{[i, (11:mm)]}`.
    :type minR: float
    :param maxR: The maximum observation noise variance for each sample
        :math:`j=1\ldots m` in the observation noise covariance
        matrix :math:`\mathbf{R}_{[i, (11:mm)]}`.
    :type maxR: float
    :param numNearest: The number of nearest nearby sparse features to use for local
        variance calculation.
    :type numNearest: int
    :param localWeight: The weight for the local variance in the observation model.
    :type localWeight: float
    :param globalWeight: The weight for the global noise level in the observation model.
    :type globalWeight: float
    :param approximationWindowLengthBP: The length of the approximation window in base pairs (BP)
        for the local variance calculation.
    :type approximationWindowLengthBP: int
    :param sparseBedFile: The path to a BED file of 'sparse' regions for the local variance calculation.
    :type sparseBedFile: str, optional
    :param noGlobal: If True, only the 'local' variances are used to approximate observation noise
        covariance :math:`\mathbf{R}_{[:, (11:mm)]}`.
    :type noGlobal: bool
    :param useALV: Whether to use average local variance (ALV) to approximate observation noise
        covariances per-sample, per-interval.
    :type useALV: bool
    :param useConstantNoiseLevel: Whether to use a constant noise level in the observation model.
    :type useConstantNoiseLevel: bool
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
    returnCenter: bool


class stateParams(NamedTuple):
    r"""Parameters related to state and uncertainty bounds and initialization.

    :param stateInit: Initial (primary) state estimate at the first genomic interval: :math:`x_{[1]}`
    :type stateInit: float
    :param stateCovarInit: Initial state covariance (covariance) scale. Note, the *initial* state uncertainty :math:`\mathbf{P}_{[1]}` is a multiple of the identity matrix :math:`\mathbf{I}`
    :type stateCovarInit: float
    :param boundState: If True, the primary state estimate for :math:`x_{[i]}` is constrained within `stateLowerBound` and `stateUpperBound`.
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
    :param offsetStr: Two or four comma-separated integers. With two, NOT strand-specific. With four, first 2 ints correspond to forward strand (start, end), last 2 to reverse strand (start, end).
    :type offsetStr: str, optional

    .. note::
        For an overview of SAM flags, see https://broadinstitute.github.io/picard/explain-flags.html
    """
    samThreads: int
    samFlagExclude: int
    oneReadPerBin: int
    chunkSize: int
    offsetStr: Optional[str] = "0,0"


class detrendParams(NamedTuple):
    r"""Parameters related detrending and background-removal

    :param useOrderStatFilter: Whether to use order statistics for filtering the read density data.
    :type useOrderStatFilter: bool
    :param usePolyFilter: Whether to use polynomial fitting for filtering the read density data.
    :type usePolyFilter: bool
    :param detrendSavitzkyGolayDegree: The polynomial degree of the Savitzky-Golay filter to use for detrending
    :type detrendSavitzkyGolayDegree: int
    :param detrendTrackPercentile: The percentile to use for detrending the read density data.
    :type detrendTrackPercentile: float
    :param detrendWindowLengthBP: The length of the window in base pairs for detrending the read density data.
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

    """
    bamFiles: List[str]
    bamFilesControl: Optional[List[str]]


class genomeParams(NamedTuple):
    genomeName: str
    chromSizesFile: str
    blacklistFile: Optional[str]
    sparseBedFile: Optional[str]
    chromosomes: List[str]
    excludeChroms: List[str]
    excludeForNorm: List[str]


class countingParams(NamedTuple):
    r"""Parameters related to counting reads in genomic intervals.

    :param stepSize: Step size for the genomic intervals.
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
    """
    stepSize: int
    scaleDown: Optional[bool]
    scaleFactors: Optional[List[float]]
    scaleFactorsControl: Optional[List[float]]
    numReads: int


class matchingParams(NamedTuple):
    r"""Parameters related to the (experimental) pattern matching routine packaged with this software.

    See :func:`consenrich.matching.matchWavelet` and :func:`cconsenrich.csampleBlockStats` for implementation.
    """

    templateNames: List[str]
    cascadeLevels: List[int]
    iters: int
    alpha: float
    minMatchLengthBP: Optional[int]
    maxNumMatches: Optional[int]
    minSignalAtMaxima: Optional[float]


def _numIntervals(start: int, end: int, step: int) -> int:
    # helper for consistency
    length = max(0, end - start)
    return (length + step) // step


def getChromRanges(
    bamFile: str,
    chromosome: str,
    chromLength: int,
    samThreads: int,
    samFlagExclude: int
) -> Tuple[int,int]:
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
    start: int = cconsenrich.cgetFirstChromRead(bamFile, chromosome, chromLength, samThreads, samFlagExclude)
    end: int = cconsenrich.cgetLastChromRead(bamFile, chromosome, chromLength, samThreads, samFlagExclude)
    return start,end


def getChromRangesJoint(bamFiles: List[str],
                        chromosome: str,
                        chromSize: int,
                        samThreads: int,
                        samFlagExclude: int
                        ) -> Tuple[int,int]:
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
            samFlagExclude=samFlagExclude
        )
        starts.append(start)
        ends.append(end)
    return min(starts), max(ends)


def getReadLength(bamFile: str,
                numReads: int,
                maxIterations: int,
                samThreads: int,
                samFlagExclude: int
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
    init_rlen = cconsenrich.cgetReadLength(bamFile, numReads, samThreads, maxIterations, samFlagExclude)
    if init_rlen == 0:
        raise ValueError(f"Failed to determine read length in {bamFile}. Revise `numReads`, and/or `samFlagExclude` parameters?")
    return init_rlen


def getReadLengths(bamFiles: List[str], numReads: int, maxIterations: int, samThreads: int, samFlagExclude: int) -> List[int]:
    r"""Get read lengths for a list of BAM files.

    :seealso: :func:`getReadLength`
    """
    return [getReadLength(bamFile, numReads=numReads, maxIterations=maxIterations, samThreads=samThreads, samFlagExclude=samFlagExclude)
     for bamFile in bamFiles]


def readBamSegments(bamFiles:List[str], chromosome: str, start: int,
            end: int,
            stepSize: int,
            readLengths: List[int],
            scaleFactors: List[float],
            oneReadPerBin: int,
            samThreads: int,
            samFlagExclude: int,
            offsetStr: Optional[str] = "0,0") -> npt.NDArray[np.float32]:
    r"""Calculate tracks of read counts (or a function thereof) for each BAM file.

    See :func:`cconsenrich.creadBamSegment` for the underlying implementation in Cython.

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
    :param offsetStr: see :class:`samParams`.
    :type offsetStr: str
    """

    if len(readLengths) != len(bamFiles) or len(scaleFactors) != len(bamFiles):
        raise ValueError(
            "readLengths and scaleFactors must match bamFiles length"
        )

    offsetStr = (str(offsetStr) or "0,0").replace(" ", "")
    numIntervals = _numIntervals(start, end, stepSize)

    counts = np.empty((len(bamFiles), numIntervals), dtype=np.float32)
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
            offsetStr,
        )
        counts[j, :] = arr
        counts[j, :] *= np.float32(scaleFactors[j])

    return counts


def getAverageLocalVarianceTrack(values: np.ndarray,
                                  stepSize: int,
                                  approximationWindowLengthBP: int,
                                  lowPassWindowLengthBP: int,
                                  minR: float,
                                  maxR: float) -> npt.NDArray[np.float32]:
    r"""Approximate local noise levels in a segment using an ALV approach.

    First, computes a segment-length simple moving average of `values` with a
    bp-length window `approximationWindowLengthBP`.

    Second, computes a segment-length simple moving average of squared `values`.

    Between these two averages, the difference between the latter and the square of the former
    approximates the local variance of the segment. These local variances
    are then combined with a median filter of length `lowPassWindowLengthBP`.

    :param values: An array of read-density-based values (typically from a single row in a sample-by-interval matrix)
    :type values: np.ndarray
    :param stepSize: See :class:`countingParams`.
    :type stepSize: int
    :param observationParams: See :class:`observationParams`
    :type observationParams: observationParams
    :param approximationWindowLengthBP: The length of the approximation window in base pairs (BP).
    :type approximationWindowLengthBP: int
    :param lowPassWindowLengthBP: The length of the low-pass filter window in base pairs (BP).
    :type lowPassWindowLengthBP: int

    :seealso: :class:`observationParams`
    """
    values = np.asarray(values, dtype=np.float32)
    windowLength = int(approximationWindowLengthBP / stepSize)
    if windowLength % 2 == 0:
        windowLength += 1
    if len(values)< 3:
        constVar = np.var(values)
        if constVar < minR:
            return np.full_like(values, minR, dtype=np.float32)
        return np.full_like(values, constVar, dtype=np.float32)

    # first get a simple moving average of the values
    localMeanTrack: npt.NDArray[np.float32] = ndimage.uniform_filter(values, size=windowLength, mode='nearest')

    #  ~ E[X_i^2] - E[X_i]^2 ~
    localVarTrack: npt.NDArray[np.float32] = ndimage.uniform_filter(values**2, size=windowLength, mode='nearest')\
        - localMeanTrack**2

    # safe-guard: difference of convolutions returns negative values.
    # shouldn't actually happen, but just in case there are some
    # ...potential artifacts i'm unaware of edge effects, etc.
    localVarTrack = np.maximum(localVarTrack, 0.0)

    # low-pass filter on the local variance track: positional 'noise level' track
    lpassWindowLength = int(lowPassWindowLengthBP / stepSize)
    if lpassWindowLength % 2 == 0:
        lpassWindowLength += 1

    noiseLevel: npt.NDArray[np.float32] = ndimage.median_filter(localVarTrack, size=lpassWindowLength)

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
    initMatrixF[0,1] = np.float32(deltaF)
    return initMatrixF


def constructMatrixQ(minDiagQ: float, offDiagQ: float = 0.0) -> npt.NDArray[np.float32]:
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
    initMatrixQ: npt.NDArray[np.float32] = np.zeros((2, 2), dtype=np.float32)
    initMatrixQ[0, 0] = minDiagQ
    initMatrixQ[1, 1] = minDiagQ
    initMatrixQ[0, 1] = offDiagQ
    initMatrixQ[1, 0] = offDiagQ
    return initMatrixQ


def constructMatrixH(m: int, coefficients: Optional[np.ndarray] = None) -> npt.NDArray[np.float32]:
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
        coefficientsH: Optional[np.ndarray]=None,
        residualCovarInversionFunc: Optional[Callable] = None,
        adjustProcessNoiseFunc: Optional[Callable] = None,
        ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    r"""Run consenrich on a contiguous segment (e.g. a chromosome) of read-density-based data.
    Completes the forward and backward passes given data and approximated observation noise
    covariance matrices :math:`\mathbf{R}_{[1:n, (11:mm)]}`.

    :param matrixData: Read density data for a single chromosome or general contiguous segment,
      possibly preprocessed. Two-dimensional array of shape :math:`m \times n` where :math:`m`
      is the number of samples/tracks and :math:`n` the number of genomic intervals.
    :type matrixData: np.ndarray
    :param matrixMunc: Uncertainty estimates for the read density data, e.g. local variance.
        Two-dimensional array of shape :math:`m \times n` where :math:`m` is the number of samples/tracks
        and :math:`n` the number of genomic intervals. :seealso: :func:`getAverageLocalVarianceTrack`, :func:`getMuncTrack`.
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
    :return: Tuple of three numpy arrays:
        - state estimates :math:`\widetilde{\mathbf{x}}_{[i]}` of shape :math:`n \times 2`
        - state covariance estimates :math:`\widetilde{\mathbf{P}}_{[i]}` of shape :math:`n \times 2 \times 2`
        - post-fit residuals :math:`\widetilde{\mathbf{y}}_{[i]}` of shape :math:`n \times m`
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    :raises ValueError: If the number of samples in `matrixData` is not equal to the number of samples in `matrixMunc`.
    """
    matrixData = matrixData.astype(np.float32)
    matrixMunc = matrixMunc.astype(np.float32)
    m: int = 1 if matrixData.ndim == 1 else matrixData.shape[0]
    n: int = 1 if matrixData.ndim == 1 else matrixData.shape[1]
    scaleQ: float = np.float32(1.0)
    inflatedQ: bool = False
    dStat: float = np.float32(0.0)
    IKH: np.ndarray = np.zeros(shape=(2,2), dtype=np.float32)
    matrixEInverse: np.ndarray = np.zeros(shape=(m, m), dtype=np.float32)

    matrixF: np.ndarray = constructMatrixF(deltaF)
    matrixQ: np.ndarray = constructMatrixQ(minQ, offDiagQ=offDiagQ)
    matrixQCopy: np.ndarray = matrixQ.copy()
    matrixP: np.ndarray = np.eye(2, dtype=np.float32) * np.float32(stateCovarInit)
    matrixH: np.ndarray = constructMatrixH(m, coefficients=coefficientsH)
    matrixK: np.ndarray = np.zeros((2, m), dtype=np.float32)
    vectorX: np.ndarray = np.array([stateInit, 0.0], dtype=np.float32)
    vectorY: np.ndarray = np.zeros(m, dtype=np.float32)
    vectorH: np.ndarray = matrixH[:, 0]

    if residualCovarInversionFunc is None:
        residualCovarInversionFunc = cconsenrich.cinvertMatrixE
    if adjustProcessNoiseFunc is None:
        adjustProcessNoiseFunc = cconsenrich.updateProcessNoiseCovariance

    # ==========================
    # forward: 0,1,2,...,n-1
    # ==========================
    stateForward = np.memmap(
        NamedTemporaryFile(delete=True), dtype=np.float32,
        mode='w+',
        shape=(n,2))
    stateCovarForward = np.memmap(
        NamedTemporaryFile(delete=True), dtype=np.float32,
        mode='w+',
        shape=(n, 2, 2))
    pNoiseForward = np.memmap(
        NamedTemporaryFile(delete=True), dtype=np.float32,
        mode='w+',
        shape=(n, 2, 2))

    for i in range(n):
        if i % progressIter == 0:
            logger.info(f"Forward pass interval: {i+1}/{n}")
        vectorZ = matrixData[:, i]
        vectorX = matrixF @ vectorX
        matrixP = matrixF @ matrixP @ matrixF.T + matrixQ
        vectorY = vectorZ - (matrixH @ vectorX)

        matrixEInverse = residualCovarInversionFunc(matrixMunc[:, i], np.float32(matrixP[0, 0]))
        dStat = np.median((vectorY**2) * np.diag(matrixEInverse))
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
        IKH[0][0] = 1.0 - (matrixK[0,:] @ vectorH)
        IKH[1][0] = -matrixK[1,:] @ vectorH
        IKH[1][1] = 1.0

        vectorX = vectorX + (matrixK @ vectorY)
        matrixP = (IKH) @ matrixP @ (IKH).T + (matrixK * matrixMunc[:, i]) @ matrixK.T
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

    # ==========================
    # backward: n,n-1,n-2,...,0
    # ==========================
    stateSmoothed = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, 2)
        )
    stateCovarSmoothed = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n, 2, 2)
    )
    postFitResiduals = np.memmap(
        NamedTemporaryFile(delete=True),
        dtype=np.float32,
        mode="w+",
        shape=(n,m)
    )

    stateSmoothed[-1] = np.float32(stateForwardArr[-1])
    stateCovarSmoothed[-1] = np.float32(stateCovarForwardArr[-1])
    postFitResiduals[-1] = np.float32(matrixData[:, -1] - (matrixH @ stateSmoothed[-1]))

    for k in range(n - 2, -1, -1):
        if k % progressIter == 0:
            logger.info(f"Backward pass interval: {k+1}/{n}")
        forwardStatePosterior = stateForwardArr[k]
        forwardCovariancePosterior = stateCovarForwardArr[k]
        backwardInitialState = matrixF @ forwardStatePosterior
        backwardInitialCovariance = matrixF @ forwardCovariancePosterior @ matrixF.T + pNoiseForwardArr[k + 1]

        smootherGain = np.linalg.solve(backwardInitialCovariance.T, (forwardCovariancePosterior @ matrixF.T).T).T
        stateSmoothed[k] = (forwardStatePosterior + smootherGain @ (stateSmoothed[k + 1] - backwardInitialState)).astype(np.float32)

        stateCovarSmoothed[k] = (forwardCovariancePosterior
                                + smootherGain @ (stateCovarSmoothed[k + 1] - backwardInitialCovariance) @ smootherGain.T).astype(np.float32)
        postFitResiduals[k] = np.float32(matrixData[:, k] - matrixH @ stateSmoothed[k])

        if k % chunkSize == 0 and k > 0:
            stateSmoothed.flush()
            stateCovarSmoothed.flush()
            postFitResiduals.flush()

    stateSmoothed.flush()
    stateCovarSmoothed.flush()
    postFitResiduals.flush()
    if boundState:
        stateSmoothed[:,0] = np.clip(stateSmoothed[:,0], stateLowerBound, stateUpperBound).astype(np.float32)

    return stateSmoothed[:], stateCovarSmoothed[:], postFitResiduals[:]


def getPrimaryState(stateVectors: np.ndarray, roundPrecision: int = 3) -> npt.NDArray[np.float32]:
    r"""Get the primary state estimate from each vector after running Consenrich.

    :param stateVectors: State vectors from :func:`runConsenrich`.
    :type stateVectors: npt.NDArray[np.float32]
    :return: A one-dimensional numpy array of the primary state estimates.
    :rtype: npt.NDArray[np.float32]
    """
    return np.round(stateVectors[:, 0], decimals=roundPrecision).astype(np.float32)


def getStateCovarTrace(stateCovarMatrices: np.ndarray, roundPrecision: int = 3) -> npt.NDArray[np.float32]:
    r"""Get a one-dimensional array of state covariance traces after running Consenrich

    :param stateCovarMatrices: Estimated state covariance matrices :math:`\widetilde{\mathbf{P}}_{[i]}`
    :type stateCovarMatrices: np.ndarray
    :return: A one-dimensional numpy array of the traces of the state covariance matrices.
    :rtype: npt.NDArray[np.float32]
    """
    return np.round(cconsenrich.cgetStateCovarTrace(stateCovarMatrices.astype(np.float32)), decimals=roundPrecision).astype(np.float32)


def getPrecisionWeightedResidual(postFitResiduals: np.ndarray, matrixMunc: np.ndarray, roundPrecision: int = 3) -> npt.NDArray[np.float32]:
    r"""Get a one-dimensional precision-weighted array residuals after running Consenrich.

    This is essentially an estimate of the residuals with respect to the observation noise covariance
    :math:`\mathbf{R}_{[:, (11:mm)]}`.

    Applies an inverse-variance weighting (with respect to the *observation noise levels*) of the
    post-fit residuals :math:`\widetilde{\mathbf{y}}_{[i]}` and returns a one-dimensional array of
    "precision-weighted residuals".

    :param postFitResiduals: Post-fit residuals from :func:`runConsenrich`.
    :type postFitResiduals: np.ndarray
    :param matrixMunc: an :math:`m \times n` numpy array where each column stores the diagonal entries
        of the observation noise covariance matrix :math:`\mathbf{R}_{[:, (11:mm)]}` for each sample :math:`j=1,2,\ldots,m`
        and each genomic interval :math:`i=1,2,\ldots,n`.
    :type matrixMunc: np.ndarray

    :return: A one-dimensional array of "precision-weighted residuals"
    :rtype: npt.NDArray[np.float32]
    """
    return np.round(cconsenrich.cgetPrecisionWeightedResidual(postFitResiduals.astype(np.float32),
        matrixMunc.astype(np.float32)), decimals=roundPrecision).astype(np.float32)


def getMuncTrack(chromosome: str,
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
        sparseMap: Optional[dict[int, int]] = None) -> npt.NDArray[np.float32]:
    r"""Get observation noise variance :math:`R_{[:,jj]}` for the sample :math:`j`.

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
    :return: A one-dimensional numpy array of the observation noise track for the sample :math:`j`.
    :rtype: npt.NDArray[np.float32]

    """
    trackALV  = getAverageLocalVarianceTrack(rowValues,
                                      stepSize,
                                      approximationWindowLengthBP,
                                      lowPassWindowLengthBP,
                                      minR,
                                      maxR).astype(np.float32)

    globalNoise: float = np.float32(np.mean(trackALV))
    if noGlobal or globalWeight == 0 or useALV:
        return np.clip(trackALV, minR, maxR).astype(np.float32)

    if useConstantNoiseLevel or localWeight == 0 and sparseMap is None:
        return np.clip(globalNoise * np.ones_like(rowValues), minR, maxR).astype(np.float32)

    if sparseMap is not None:
        trackALV = cconsenrich.cSparseAvg(trackALV, sparseMap)

    return np.clip(trackALV*localWeight + np.mean(trackALV)*globalWeight,
                   minR, maxR).astype(np.float32)


def sparseIntersection(chromosome: str, intervals: np.ndarray, sparseBedFile: str) -> npt.NDArray[np.int64]:
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
        .sort().merge().filter(
            lambda b: (
                b.chrom == chromosome
                and b.start > intervals[0]
                and b.end < intervals[-1]
                and (b.end - b.start) >= stepSize
            )
        )
    )
    centeredFeatures: bed.BedTool = chromFeatures.each(
        adjustFeatureBounds,
        stepSize=stepSize
    )

    start0: int = int(intervals[0])
    last:   int = int(intervals[-1])
    chromFeatures: bed.BedTool = (
        bed.BedTool(sparseBedFile)
        .sort().merge().filter(
            lambda b: (
                b.chrom == chromosome
                and b.start > start0
                and b.end   < last
                and (b.end - b.start) >= stepSize
            )
        )
    )
    centeredFeatures: bed.BedTool = chromFeatures.each(adjustFeatureBounds, stepSize=stepSize)
    centeredStarts = []
    for f in centeredFeatures:
        s = int(f.start)
        if start0 <= s <= last and (s - start0) % stepSize == 0:
            centeredStarts.append(s)
    return np.asarray(centeredStarts, dtype=np.int64)


def adjustFeatureBounds(feature: bed.Interval, stepSize: int) -> bed.Interval:
    r"""Adjust the start and end positions of a BED feature to be centered around a step."""
    feature.start = cconsenrich.stepAdjustment(
        (feature.start + feature.end) // 2,
        stepSize
    )
    feature.end = feature.start + stepSize
    return feature


def getSparseMap(chromosome: str,
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
    sparseStarts = sparseIntersection(chromosome, intervals, sparseBedFile)
    idxSparseInIntervals = np.searchsorted(intervals, sparseStarts, side="left")
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