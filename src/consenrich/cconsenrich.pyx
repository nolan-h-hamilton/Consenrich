# -*- coding: utf-8 -*-
r"""Cython module for Consenrich core functions.

This module contains Cython implementations of core functions used in Consenrich.
"""
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp
from pysam.libcalignmentfile cimport AlignmentFile, AlignedSegment
cimport cython


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
    return max(0, (value - (value % stepSize))) + pushForward*stepSize


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
    for read in aln.fetch(contig=chromosome, start=0, stop=chromLength):
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

cpdef cnp.uint32_t[:] creadBamSegment(
    str bamFile,
    str chromosome,
    uint32_t start,
    uint32_t end,
    uint32_t stepSize,
    uint32_t readLength,
    uint8_t oneReadPerBin,
    uint16_t samThreads,
    uint16_t samFlagExclude):
    r"""Count reads in a BAM file for a given chromosome and range, returning a numpy array of counts.

    See :func:`consenrich.core.readBamSegments` for the multi-sample python wrapper

    :param bamFile: See :class:`consenrich.core.inputParams`.
    :type bamFile: str
    :param chromosome: Chromosome name.
    :type chromosome: str
    :param start: Start position of the range in base pairs.
    :type start: uint32_t
    :param end: End position of the range in base pairs.
    :type end: uint32_t
    :param stepSize: Size of the bins to count reads in.
    :type stepSize: uint32_t
    :param readLength: Length of the reads. If greater than stepSize, counts reads
        in bins defined by read start and end positions. See :func:`consenrich.core.getReadLength`.
    :type readLength: uint32_t
    :param oneReadPerBin: See :class:`consenrich.core.samParams`.
    :type oneReadPerBin: uint8_t
    :param samThreads: See :class:`consenrich.core.samParams`.
    :type samThreads: uint16_t
    :param samFlagExclude: See :class:`consenrich.core.samParams`.
    :type samFlagExclude: uint16_t
    :return: A numpy array of counts for each bin in the specified range.
    :rtype: cnp.ndarray[cnp.uint32_t, ndim=1]
    """
    cdef Py_ssize_t n = ((end - start) // stepSize) + 1
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] values_np = np.zeros(n, dtype=np.uint32)
    cdef cnp.uint32_t[::1] values = values_np

    cdef AlignmentFile aln = AlignmentFile(bamFile, 'rb', threads=samThreads)
    cdef AlignedSegment read
    cdef uint32_t readStart, readEnd, readMid, readIndex, readIndex0, readIndex1
    cdef Py_ssize_t i

    if oneReadPerBin == 0 and readLength > stepSize:
        for read in aln.fetch(chromosome, start, end):
            if (<uint16_t>read.flag & samFlagExclude) != 0:
                continue
            readStart = <uint32_t>read.reference_start
            readEnd = <uint32_t>read.reference_end
            readIndex0 = (readStart - start) // stepSize
            readIndex1 = (readEnd - start) // stepSize
            for i in range(readIndex0, readIndex1 + 1):
                if 0 <= i < n:
                    values[i] += 1
    else:
        for read in aln.fetch(chromosome, start, end):
            if (<uint16_t>read.flag & samFlagExclude) != 0:
                continue
            readStart = <uint32_t>read.reference_start
            readEnd = <uint32_t>read.reference_end
            readMid = (readStart + readEnd) >> 1
            readIndex = (readMid - start) // stepSize
            if 0 <= readIndex < n:
                values[readIndex] += 1
    aln.close()

    return values


cpdef cnp.ndarray[cnp.float64_t, ndim=2] cinvertMatrixE(cnp.float64_t[:] muncMatrixIter, double priorCovarianceOO):
    r"""Invert the residual covariance matrix during the forward pass.

    :param muncMatrixIter: The diagonal elements of the covariance matrix at a given genomic interval.
    :type muncMatrixIter: cnp.ndarray[cnp.float64_t, ndim=1]
    :param priorCovarianceOO: The a priori 'primary' state variance :math:`P_{[i|i-1,11]}`.
    :type priorCovarianceOO: cnp.float64_t
    :return: The inverted covariance matrix.
    :rtype: cnp.ndarray[cnp.float64_t, ndim=2]
    """

    cdef int m = muncMatrixIter.size
    # we have to invert a P.D. covariance (diagonal) and rank-one (1*priorCovariance) matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=2] inverse = np.empty((m, m), dtype=np.float64)
    # note, not actually an m-dim matrix, just the diagonal elements taken as input
    cdef cnp.ndarray[cnp.float64_t, ndim=1] muncMatrixInverse = np.empty(m, dtype=np.float64)
    cdef double sqrtPrior = sqrt(priorCovarianceOO)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] uVec = np.empty(m, dtype=np.float64)
    cdef double divisor = 1.0
    cdef double scale
    cdef double uVecI
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
        inverse[i, i] = muncMatrixInverse[i] - (scale*uVecI*uVecI)
        for j in range(i + 1, m):
            inverse[i, j] = -scale * uVecI * uVec[j]
            inverse[j, i] = inverse[i, j]
    return inverse


cpdef cnp.ndarray[cnp.float64_t, ndim=1] cgetStateCovarTrace(cnp.ndarray[cnp.float64_t, ndim=3] stateCovarMatrices):
    cdef int n = stateCovarMatrices.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] trace = np.empty(n, dtype=np.float64)
    for i in range(n):
        trace[i] = stateCovarMatrices[i, 0, 0] + stateCovarMatrices[i, 1, 1]
    return trace


cpdef cgetPrecisionWeightedResidual(cnp.ndarray[cnp.float64_t, ndim=2] postFitResiduals,
                                    cnp.ndarray[cnp.float64_t, ndim=2] matrixMunc):

    cdef int n = postFitResiduals.shape[0]
    cdef int m = postFitResiduals.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] precisionWeightedResidual = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] weightsIter = np.empty(m, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] residualsIter = np.empty(m, dtype=np.float64)
    cdef double sumWeights = 0.0
    cdef double sumResiduals = 0.0
    cdef int i, j
    for i in range(n):
        sumWeights = 0.0
        sumResiduals = 0.0
        for j in range(m):
            weightsIter[j] = 1.0 / matrixMunc[j, i]
            residualsIter[j] = postFitResiduals[i, j]
            sumWeights += weightsIter[j]
            sumResiduals += residualsIter[j] * weightsIter[j]
        if sumWeights > 0.0:
            precisionWeightedResidual[i] = sumResiduals / sumWeights
        else:
            precisionWeightedResidual[i] = 0.00
    return precisionWeightedResidual


cpdef tuple updateProcessNoiseCovariance(cnp.ndarray[cnp.float64_t, ndim=2] matrixQ,
        cnp.ndarray[cnp.float64_t, ndim=2] matrixQCopy,
        double dStat,
        double dStatAlpha,
        double dStatd,
        double dStatPC,
        bint inflatedQ,
        double maxQ,
        double minQ):
    r"""Adjust process noise covariance matrix :math:`\mathbf{Q}_{[i]}`

    :param matrixQ: Current process noise covariance
    :param matrixQCopy: A copy of the initial original covariance matrix :math:`\mathbf{Q}_{[.]}`
    :param inflatedQ: Flag indicating if the process noise covariance is inflated
    :return: Updated process noise covariance matrix and inflated flag
    :rtype: tuple
    """

    cdef double scaleQ, fac
    if dStat > dStatAlpha:
        scaleQ = sqrt(dStatd * fabs(dStat - dStatAlpha) + dStatPC)
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
        scaleQ = sqrt(dStatd * fabs(dStat - dStatAlpha) + dStatPC)
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
    return np.asarray(matrixQ), inflatedQ



