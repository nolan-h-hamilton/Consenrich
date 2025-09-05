# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False, infer_types=True, language_level=3
r"""Cython module for Consenrich core functions.

This module contains Cython implementations of core functions used in Consenrich.
"""

from libc.math cimport abs, fabs, sqrt
from libc.stdint cimport int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from cpython.array cimport array
from pysam.libcalignmentfile cimport AlignedSegment, AlignmentFile
import numpy as np
cimport numpy as cnp

cnp.import_array()

cpdef int stepAdjustment(int value, int stepSize, int pushForward=0):
    r"""Adjusts a value to the nearest multiple of stepSize, optionally pushing it forward.

    :param value: The value to adjust.
    :type value: int
    :param stepSize: The step size to adjust to.
    :type stepSize: int
    :param pushForward: If non-zero, pushes the value forward by stepSize if it is
        not already a multiple of stepSize.
    :type pushForward: int
    :return: The adjusted value.
    :rtype: int
    """
    return max(0, (value-(value % stepSize))) + pushForward*stepSize


cpdef uint64_t cgetFirstChromRead(str bamFile, str chromosome, uint64_t chromLength, uint32_t samThreads, int samFlagExclude):
    r"""Get the start position of the first read in a BAM file for a given chromosome.

    :param bamFile: See :func:`consenrich.core.inputParams`.
    :type bamFile: str
    :param chromosome: Chromosome name.
    :type chromosome: str
    :param chromLength: Length of the chromosome in base pairs.
    :type chromLength: uint64_t
    :param samThreads: Number of threads to use for reading the BAM file.
    :type samThreads: uint32_t
    :param samFlagExclude: SAM flags to exclude reads (e.g., unmapped,
    :type samFlagExclude: int
    :return: Start position of the first read in the chromosome, or 0 if no reads are found.
    :rtype: uint64_t
    """

    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    for read in aln.fetch(contig=chromosome, start=0, end=chromLength):
        if not (read.flag & samFlagExclude):
            aln.close()
            return read.reference_start
    aln.close()
    return 0


cpdef uint64_t cgetLastChromRead(str bamFile, str chromosome, uint64_t chromLength, uint32_t samThreads, int samFlagExclude):
    r"""Get the end position of the last read in a BAM file for a given chromosome.

    :param bamFile: See :func:`consenrich.core.inputParams`.
    :type bamFile: str
    :param chromosome: Chromosome name.
    :type chromosome: str
    :param chromLength: Length of the chromosome in base pairs.
    :type chromLength: uint64_t
    :param samThreads: Number of threads to use for reading the BAM file.
    :type samThreads: uint32_t
    :param samFlagExclude: See :class:`consenrich.core.samParams`.
    :type samFlagExclude: int
    :return: End position of the last read in the chromosome, or 0 if no reads are found.
    :rtype: uint64_t
    """

    cdef uint64_t start_ = chromLength - min((chromLength // 2), 1_000_000)
    cdef uint64_t lastPos = 0
    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    for read in aln.fetch(contig=chromosome, start=start_, end=chromLength):
        if not (read.flag & samFlagExclude):
            lastPos = read.reference_end
    aln.close()
    return lastPos



cpdef uint32_t cgetReadLength(str bamFile, uint32_t minReads, uint32_t samThreads, uint32_t maxIterations, int samFlagExclude):
    r"""Get the median read length from a BAM file after fetching a specified number of reads.

    :param bamFile: see :class:`consenrich.core.inputParams`.
    :type bamFile: str
    :param minReads: Minimum number of reads to consider for the median calculation.
    :type minReads: uint32_t
    :param samThreads: See :class:`consenrich.core.samParams`.
    :type samThreads: uint32_t
    :param maxIterations: Maximum number of reads to iterate over.
    :type maxIterations: uint32_t
    :param samFlagExclude: See :class:`consenrich.core.samParams`.
    :type samFlagExclude: int
    :return: Median read length from the BAM file.
    :rtype: uint32_t
    """
    cdef uint32_t observedReads = 0
    cdef uint32_t currentIterations = 0
    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] readLengths = np.zeros(maxIterations, dtype=np.uint32)
    cdef uint32_t i = 0
    if <uint32_t>aln.mapped < minReads:
        aln.close()
        return 0
    for read in aln.fetch():
        if not (observedReads < minReads and currentIterations < maxIterations):
            break
        if not (read.flag & samFlagExclude):
            # meets critera -> add it
            readLengths[i] = read.query_length
            observedReads += 1
            i += 1
        currentIterations += 1
    aln.close()
    if observedReads < minReads:
        return 0
    return <uint32_t>np.median(readLengths[:observedReads])


cdef inline Py_ssize_t floordiv64(int64_t a, int64_t b) nogil:
    if a >= 0:
        return <Py_ssize_t>(a // b)
    else:
        return <Py_ssize_t>(- ((-a + b - 1) // b))


cpdef inline int64_t getNumIntervals(int64_t start, int64_t end, int64_t step):
    cdef int64_t span = end - start
    if span <= 0:
        return 0
    return (span + step - 1) // step


cpdef cnp.uint32_t[:] creadBamSegment(
    str bamFile,
    str chromosome,
    uint32_t start,
    uint32_t end,
    uint32_t stepSize,
    int64_t readLength,
    uint8_t oneReadPerBin,
    uint16_t samThreads,
    uint16_t samFlagExclude,
    int64_t shiftForwardStrand53 = 0,
    int64_t shiftReverseStrand53 = 0,
    int64_t extendBP = 0,
    int64_t maxInsertSize=1000,
    int64_t pairedEndMode=0,
    int64_t inferFragmentLength=0):
    r"""Count reads in a BAM file for a given chromosome"""

    cdef Py_ssize_t numIntervals = <Py_ssize_t>getNumIntervals(<int64_t>start, <int64_t>end, <int64_t>stepSize)

    cdef cnp.ndarray[cnp.uint32_t, ndim=1] values_np = np.zeros(numIntervals, dtype=np.uint32)
    cdef cnp.uint32_t[::1] values = values_np

    if numIntervals <= 0:
        return values

    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    cdef int64_t start64 = start
    cdef int64_t end64 = end
    cdef int64_t step64 = stepSize
    cdef Py_ssize_t i, index0, index1
    cdef Py_ssize_t lastIndex = numIntervals - 1
    cdef bint readIsForward
    cdef int64_t readStart, readEnd
    cdef int64_t adjStart, adjEnd, fivePrime, mid, midIndex, tlen, atlen
    cdef uint16_t flag
    if inferFragmentLength > 0 and pairedEndMode == 0 and extendBP == 0:
        extendBP = cgetFragmentLength(bamFile,
         chromosome,
         <int64_t>start,
         <int64_t>end,
         samThreads = samThreads,
         samFlagExclude=samFlagExclude,
         maxInsertSize=maxInsertSize,
         minInsertSize=<int64_t>readLength, # xcorr peak > rlen ~~> fraglen
         )
    try:
        with aln:
            for read in aln.fetch(chromosome, start64, end64):
                flag = <uint16_t>read.flag
                if (flag & samFlagExclude) != 0:
                    continue
                if flag & 4:
                    continue

                readIsForward = (flag & 16) == 0
                readStart = <int64_t>read.reference_start
                readEnd   = <int64_t>read.reference_end

                if (flag & 2) and pairedEndMode > 0:
                    # use first in pair + fragment
                    if flag & 128:
                        continue
                    if (flag & 8) or read.next_reference_id != read.reference_id:
                        continue
                    tlen = <int64_t>read.template_length
                    atlen = tlen if tlen >= 0 else -tlen
                    if atlen == 0 or atlen > maxInsertSize:
                        continue
                    if tlen >= 0:
                        adjStart = readStart
                        adjEnd   = readStart + atlen
                    else:
                        adjEnd   = readEnd
                        adjStart = adjEnd - atlen
                    if shiftForwardStrand53 != 0 or shiftReverseStrand53 != 0:
                        if readIsForward:
                            adjStart += shiftForwardStrand53
                            adjEnd   += shiftForwardStrand53
                        else:
                            adjStart -= shiftReverseStrand53
                            adjEnd   -= shiftReverseStrand53
                else:
                    # SE
                    if readIsForward:
                        fivePrime = readStart + shiftForwardStrand53
                    else:
                        fivePrime = (readEnd - 1) - shiftReverseStrand53

                    if extendBP > 0:
                        # from the cut 5' --> 3'
                        if readIsForward:
                            adjStart = fivePrime
                            adjEnd   = fivePrime + extendBP
                        else:
                            adjEnd   = fivePrime + 1
                            adjStart = adjEnd - extendBP
                    elif shiftForwardStrand53 != 0 or shiftReverseStrand53 != 0:
                        if readIsForward:
                            adjStart = readStart + shiftForwardStrand53
                            adjEnd   = readEnd   + shiftForwardStrand53
                        else:
                            adjStart = readStart - shiftReverseStrand53
                            adjEnd   = readEnd   - shiftReverseStrand53
                    else:
                        adjStart = readStart
                        adjEnd   = readEnd

                if adjEnd <= start64 or adjStart >= end64:
                    continue
                if adjStart < start64:
                    adjStart = start64
                if adjEnd > end64:
                    adjEnd = end64

                if oneReadPerBin:
                    # +1 at midpoint of frag.
                    mid = (adjStart + adjEnd) // 2
                    midIndex = <Py_ssize_t>((mid - start64) // step64)
                    if 0 <= midIndex <= lastIndex:
                        values[midIndex] += <uint32_t>1
                else:
                    # +1 every interval intersecting frag
                    index0 = <Py_ssize_t>((adjStart - start64) // step64)
                    index1 = <Py_ssize_t>(((adjEnd - 1) - start64) // step64)
                    if index0 < 0:
                        index0 = <Py_ssize_t>0
                    if index1 > lastIndex:
                        index1 = lastIndex
                    if index0 > lastIndex or index1 < 0 or index0 > index1:
                        continue
                    for b_ in range(index0, index1 + 1):
                        values[b_] += <uint32_t>1

    finally:
        aln.close()

    return values



cpdef cnp.ndarray[cnp.float32_t, ndim=2] cinvertMatrixE(cnp.ndarray[cnp.float32_t, ndim=1] muncMatrixIter, cnp.float32_t priorCovarianceOO):
    r"""Invert the residual covariance matrix during the forward pass.

    :param muncMatrixIter: The diagonal elements of the covariance matrix at a given genomic interval.
    :type muncMatrixIter: cnp.ndarray[cnp.float32_t, ndim=1]
    :param priorCovarianceOO: The a priori 'primary' state variance :math:`P_{[i|i-1,11]}`.
    :type priorCovarianceOO: cnp.float32_t
    :return: The inverted covariance matrix.
    :rtype: cnp.ndarray[cnp.float32_t, ndim=2]
    """

    cdef int m = muncMatrixIter.size
    # we have to invert a P.D. covariance (diagonal) and rank-one (1*priorCovariance) matrix
    cdef cnp.ndarray[cnp.float32_t, ndim=2] inverse = np.empty((m, m), dtype=np.float32)
    # note, not actually an m-dim matrix, just the diagonal elements taken as input
    cdef cnp.ndarray[cnp.float32_t, ndim=1] muncMatrixInverse = np.empty(m, dtype=np.float32)
    cdef float sqrtPrior = sqrt(priorCovarianceOO)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] uVec = np.empty(m, dtype=np.float32)
    cdef float divisor = 1.0
    cdef float scale
    cdef float uVecI
    cdef Py_ssize_t i, j
    for i in range(m):
        # two birds: build up the trace while taking the reciprocals
        muncMatrixInverse[i] = 1.0/(muncMatrixIter[i])
        divisor += priorCovarianceOO*muncMatrixInverse[i]
    # we can combine these two loops, keeping construction
    # of muncMatrixInverse and uVec separate for now in case
    # we want to parallelize this later
    for i in range(m):
        uVec[i] = sqrtPrior*muncMatrixInverse[i]
    scale = 1.0 / divisor
    for i in range(m):
        uVecI = uVec[i]
        inverse[i, i] = muncMatrixInverse[i]-(scale*uVecI*uVecI)
        for j in range(i + 1, m):
            inverse[i, j] = -scale * (uVecI*uVec[j])
            inverse[j, i] = inverse[i, j]
    return inverse


cpdef cnp.ndarray[cnp.float32_t, ndim=1] cgetStateCovarTrace(stateCovarMatrices):
    cdef Py_ssize_t n = stateCovarMatrices.shape[0]
    cdef Py_ssize_t i
    trace = np.empty(n, dtype=np.float32)
    for i in range(n):
        trace[i] = np.float32(stateCovarMatrices[i, 0, 0] + stateCovarMatrices[i, 1, 1])
    return trace


cpdef cgetPrecisionWeightedResidual(postFitResiduals,
                                    matrixMunc):

    cdef Py_ssize_t n = postFitResiduals.shape[0]
    cdef Py_ssize_t m = postFitResiduals.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] precisionWeightedResidual = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] weightsIter = np.empty(m, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] residualsIter = np.empty(m, dtype=np.float32)
    cdef float sumWeights = 0.0
    cdef float sumResiduals = 0.0
    cdef Py_ssize_t i, j
    for i in range(n):
        sumWeights = 0.0
        sumResiduals = 0.0
        for j in range(m):
            weightsIter[j] = np.float32(1.0 / matrixMunc[j, i])
            residualsIter[j] = np.float32(postFitResiduals[i, j])
            sumWeights += weightsIter[j]
            sumResiduals += residualsIter[j] * weightsIter[j]
        if sumWeights > 0.0:
            precisionWeightedResidual[i] = np.float32(sumResiduals / sumWeights)
        else:
            precisionWeightedResidual[i] = np.float32(0.00)
    return precisionWeightedResidual


cpdef tuple updateProcessNoiseCovariance(cnp.ndarray[cnp.float32_t, ndim=2] matrixQ,
        cnp.ndarray[cnp.float32_t, ndim=2] matrixQCopy,
        float dStat,
        float dStatAlpha,
        float dStatd,
        float dStatPC,
        bint inflatedQ,
        float maxQ,
        float minQ):
    r"""Adjust process noise covariance matrix :math:`\mathbf{Q}_{[i]}`

    :param matrixQ: Current process noise covariance
    :param matrixQCopy: A copy of the initial original covariance matrix :math:`\mathbf{Q}_{[.]}`
    :param inflatedQ: Flag indicating if the process noise covariance is inflated
    :return: Updated process noise covariance matrix and inflated flag
    :rtype: tuple
    """

    cdef float scaleQ, fac
    if dStat > dStatAlpha:
        scaleQ = sqrt(dStatd * fabs(dStat-dStatAlpha) + dStatPC)
        if matrixQ[0, 0] * scaleQ <= maxQ:
            matrixQ[0, 0] *= scaleQ
            matrixQ[0, 1] *= scaleQ
            matrixQ[1, 0] *= scaleQ
            matrixQ[1, 1] *= scaleQ
        else:
            fac = maxQ / matrixQCopy[0, 0]
            matrixQ[0, 0] = maxQ
            matrixQ[0, 1] = matrixQCopy[0, 1] * fac
            matrixQ[1, 0] = matrixQCopy[1, 0] * fac
            matrixQ[1, 1] = maxQ
        inflatedQ = True

    elif dStat < dStatAlpha and inflatedQ:
        scaleQ = sqrt(dStatd * fabs(dStat-dStatAlpha) + dStatPC)
        if matrixQ[0, 0] / scaleQ >= minQ:
            matrixQ[0, 0] /= scaleQ
            matrixQ[0, 1] /= scaleQ
            matrixQ[1, 0] /= scaleQ
            matrixQ[1, 1] /= scaleQ
        else:
            # we've hit the minimum, no longer 'inflated'
            fac = minQ / matrixQCopy[0, 0]
            matrixQ[0, 0] = minQ
            matrixQ[0, 1] = matrixQCopy[0, 1] * fac
            matrixQ[1, 0] = matrixQCopy[1, 0] * fac
            matrixQ[1, 1] = minQ
            inflatedQ = False
    return matrixQ, inflatedQ


cdef void _blockMax(double[::1] valuesView,
                    Py_ssize_t[::1] blockStartIndices,
                    Py_ssize_t[::1] blockSizes,
                    double[::1] outputView) noexcept:
    cdef Py_ssize_t iterIndex, elementIndex, startIndex, blockLength
    cdef double currentMax, currentValue
    for iterIndex in range(outputView.shape[0]):
        startIndex = blockStartIndices[iterIndex]
        blockLength = blockSizes[iterIndex] # note, length of blocks affects upcoming loop
        currentMax = valuesView[startIndex]
        for elementIndex in range(1, blockLength):
            currentValue = valuesView[startIndex + elementIndex]
            if currentValue > currentMax:
                currentMax = currentValue
        outputView[iterIndex] = currentMax


cpdef csampleBlockStats(cnp.ndarray[cnp.float64_t, ndim=1] values,
                        int expectedBlockSize,
                        int iters,
                        int randSeed):
    r"""Sample contiguous blocks in the response sequence, record maxima, and repeat.

    Used to build an empirical null distribution and determine significance of response outputs.
    Blocks are drawn randomly from the response sequence. The size of blocks is drawn from a
    geometric distribution (memoryless), maintaining equality in expectation but introducing
    variability for a more robust sampling.

    :param values: The response sequence to sample from.
    :type values: cnp.ndarray[cnp.float64_t, ndim=1]
    :param expectedBlockSize: The expected size (geometric) of the blocks to sample.
    :type expectedBlockSize: int
    :param iters: The number of blocks to sample.
    :type iters: int
    :param randSeed: Random seed for reproducibility.
    :type randSeed: int
    :return: An array of sampled block maxima.
    :rtype: cnp.ndarray[cnp.float64_t, ndim=1]
    :seealso: :func:`consenrich.matching.matchWavelet`
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] valuesArr = np.ascontiguousarray(values, dtype=np.float64)
    cdef double[::1] valuesView = valuesArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1] sizesArr
    cdef cnp.ndarray[cnp.intp_t, ndim=1] startsArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(iters, dtype=np.float64)
    cdef Py_ssize_t maxBlockLength, maxSize, n
    cdef double maxBlockScale = 10.0
    n = valuesView.shape[0]
    np.random.seed(randSeed)
    sizesArr = np.random.geometric(1.0 / expectedBlockSize, size=iters).astype(np.intp, copy=False)
    maxSize = <int>(maxBlockScale * expectedBlockSize)
    np.clip(sizesArr, 1, maxSize if maxSize < n else n, out=sizesArr)
    maxBlockLength = sizesArr.max() # Py_ssize_t <-- intp ok.
    # by construction, shouldn't exceed the length of the response seq.
    # +1 to check case randint(0,0)
    startsArr = np.random.randint(0, int(n-maxBlockLength + 1), size=iters, dtype=np.intp)

    cdef Py_ssize_t[::1] blockStartIndices = startsArr
    cdef Py_ssize_t[::1] blockSizes = sizesArr
    cdef double[::1] outputView = out
    _blockMax(valuesView, blockStartIndices, blockSizes, outputView)
    return out


cpdef cSparseAvg(cnp.float32_t[::1] trackALV, dict sparseMap):
    r"""Fast access and average of `numNearest` sparse elements.

    See :func:`consenrich.core.getMuncTrack`

    :param trackALV: See :func:`consenrich.core.getAverageLocalVarianceTrack`
    :type trackALV: float[::1]
    :param sparseMap: See :func:`consenrich.core.getSparseMap`
    :type sparseMap: dict[int, np.ndarray]
    :return: array of mena('nearest local variances') same length as `trackALV`
    :rtype: cnp.ndarray[cnp.float32_t, ndim=1]
    """
    cdef Py_ssize_t n = <Py_ssize_t>trackALV.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(n, dtype=np.float32)
    cdef Py_ssize_t i, j, m
    cdef float sumNearestVariances = 0.0
    cdef cnp.ndarray[cnp.intp_t, ndim=1] idxs
    cdef cnp.intp_t[::1] idx_view
    for i in range(n):
        idxs = <cnp.ndarray[cnp.intp_t, ndim=1]> sparseMap[i] # FFR: to avoid the cast, create sparseMap as dict[intp, np.ndarray[intp]]
        idx_view = idxs
        m = idx_view.shape[0] # FFR: maybe enforce strict `m == numNearest` in future releases to avoid extra overhead
        if m == 0:
            # this case probably warrants an exception or np.nan
            out[i] = 0.0
            continue
        sumNearestVariances = 0.0
        with nogil:
            for j in range(m):
                sumNearestVariances += trackALV[idx_view[j]]
        out[i] = sumNearestVariances/m

    return out


cpdef int64_t cgetFragmentLength(str bamFile,
                                 str chromosome,
                                 int64_t start,
                                 int64_t end,
                                 uint16_t samThreads=1,
                                 uint16_t samFlagExclude=3844,
                                 int64_t maxInsertSize=1000,
                                 int64_t minInsertSize=20,
                                 int64_t iters=50,
                                 int64_t blockSize=10_000,
                                 int64_t fallBack=147,
                                 int64_t randSeed=42):
    r"""Estimate the fragment length from the maximum (average) correlation lag between forward and reverse strand reads.
    """
    cdef int64_t regionLen = (end - start)
    cdef int64_t lMin = minInsertSize
    cdef int64_t lMax = maxInsertSize
    cdef int64_t pos = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fwd = np.zeros(blockSize, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rev = np.zeros(blockSize, dtype=np.float64)
    cdef int64_t itersInitial = iters

    if regionLen <= 0:
        return fallBack
    if blockSize <= 0 or lMin <= 0 or lMax <= 0 or lMin > lMax:
        return fallBack
    if blockSize <= lMin:
        return fallBack
    if end - start <= blockSize:
        iters = 1

    cdef int64_t maxBlockStart = (end - blockSize - 1)
    if maxBlockStart <= start:
        return fallBack

    cdef list candidates = []
    np.random.seed(randSeed)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] startsArr = np.random.randint(
        low=start,
        high=maxBlockStart,
        size=iters,
        dtype=np.int64)

    cdef AlignmentFile bamFileObj
    try:
        bamFileObj = AlignmentFile(bamFile, "rb", threads=<int>samThreads)
    except Exception:
        return fallBack

    cdef int64_t k, blkStart, blkEnd = 0
    cdef int64_t i, N, l, nTies, med = 0
    cdef float xCorrBest, cSum = -1.0
    cdef list ties = []
    for k in range(iters):
        blkStart = startsArr[k]
        blkEnd = blkStart + blockSize
        cSum = 0.0
        fwd.fill(0.0)
        rev.fill(0.0)
        pos = 0
        try:
            for col in bamFileObj.pileup(chromosome,
                                         blkStart,
                                         blkEnd,
                                         truncate=True,
                                         stepper="all",
                                         max_depth=1000000):
                pos = <int64_t>col.reference_pos
                if pos < blkStart or pos >= blkEnd:
                    continue
                i = pos - blkStart
                for pup in col.pileups:
                    readSeg = pup.alignment
                    if (readSeg.flag & <int>samFlagExclude) != 0:
                        continue
                    if (readSeg.flag & 16) != 0:
                        rev[i] += 1
                    else:
                        fwd[i] += 1
        except Exception:
            continue
        if fwd.sum() == 0 or rev.sum() == 0:
            continue

        xCorrBest = -1.0
        ties = []

        for l in range(lMin, lMax + 1):
            N = blockSize - l
            if N <= 0:
                break
            cSum = 0
            for i in range(N):
                cSum += fwd[i] * rev[i + l]

            if cSum > xCorrBest:
                xCorrBest = cSum
                ties = [l]
            elif cSum == xCorrBest:
                ties.append(l)

        if xCorrBest <= 0 or len(ties) == 0:
            candidates.append(fallBack)
        else:
            nTies = len(ties)
            if nTies & 1:
                med = ties[nTies // 2]
            else:
                med = ties[(nTies // 2) - 1]
            if med < fallBack:
                med = fallBack
            candidates.append(med)

    try:
        bamFileObj.close()
    except Exception:
        pass

    if not candidates:
        return fallBack

    candidates.sort()
    cdef Py_ssize_t n = len(candidates)
    cdef int64_t overall
    if n & 1:
        overall = <int64_t>candidates[n // 2]
    else:
        overall = <int64_t>candidates[(n // 2) - 1]

    if overall < fallBack:
        overall = fallBack
    if overall < minInsertSize:
        overall = minInsertSize
    if overall > maxInsertSize:
        overall = maxInsertSize

    return overall
