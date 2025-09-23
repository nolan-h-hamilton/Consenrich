import pytest
import numpy as np
import scipy.stats as stats

import consenrich.core as core
import consenrich.cconsenrich as cconsenrich


@pytest.mark.correctness
def testConstantGetAverageLocalVarianceTrack(constantValue=10):
    # case: `values` is constant --> noise level should be zero, but due to clipping, `minR`
    values = np.ones(100) * constantValue
    stepSize = 1
    approximationWindowLengthBP = 10
    lowPassWindowLengthBP = 20
    minR = 1.0
    maxR = 100.0
    out = core.getAverageLocalVarianceTrack(
        values,
        stepSize,
        approximationWindowLengthBP,
        lowPassWindowLengthBP,
        minR,
        maxR,
    )
    np.testing.assert_allclose(out, np.ones_like(values) * minR)


@pytest.mark.correctness
def testMaxVarGetAverageLocalVarianceTrack(maxVariance=20):
    # case: values (length 1000) ~ Poisson(maxVariance*2) -->
    # mode(all noise levels) ~=~ maxVariance
    np.random.seed(42)
    values = np.random.poisson(lam=20, size=1000)
    stepSize = 1
    approximationWindowLengthBP = 10
    lowPassWindowLengthBP = 20
    minR = 0.0
    maxR = maxVariance
    out = core.getAverageLocalVarianceTrack(
        values,
        stepSize,
        approximationWindowLengthBP,
        lowPassWindowLengthBP,
        minR,
        maxR,
    )
    np.testing.assert_allclose(stats.mode(out)[0], maxR, rtol=0.001)


@pytest.mark.correctness
def testMatrixConstruction(
    deltaF=0.50, coefficients=[0.1, 0.2, 0.3, 0.4], minQ=0.25, offDiag=0.10
):
    # F
    m = len(coefficients)
    matrixF = core.constructMatrixF(deltaF)
    assert matrixF.shape == (2, 2)
    np.testing.assert_allclose(matrixF, np.array([[1.0, deltaF], [0.0, 1.0]]))

    # H
    matrixH = core.constructMatrixH(m, coefficients)
    assert matrixH.shape == (m, 2)
    np.testing.assert_allclose(matrixH[:, 0], coefficients)
    np.testing.assert_allclose(matrixH[:, 1], np.zeros(m))

    # Q
    matrixQ = core.constructMatrixQ(minQ, offDiag)
    assert matrixQ.shape == (2, 2)
    np.testing.assert_allclose(
        matrixQ, np.array([[minQ, offDiag], [offDiag, minQ]])
    )


@pytest.mark.chelpers
def testResidualCovarianceInversion():
    np.random.seed(42)
    m = 10
    muncMatrixIter = np.random.gamma(shape=2, scale=1.0, size=m) + 1
    priorCovarianceOO = 0.1
    residCovar = np.diag(muncMatrixIter) + (np.ones((m, m)) * priorCovarianceOO)

    invertedMatrix = cconsenrich.cinvertMatrixE(
        muncMatrixIter.astype(np.float32), np.float32(priorCovarianceOO)
    )
    np.testing.assert_allclose(
        invertedMatrix @ residCovar, np.eye(m), atol=1e-8
    )


@pytest.mark.chelpers
def testProcessNoiseAdjustment():
    np.random.seed(42)

    m = 100
    minQ = 0.25
    maxQ = 10.0
    offDiag = 0.0
    dStatAlpha = 3.0
    dStatd = 10.0
    dStatPC = 1.0
    inflatedQ = False

    matrixQ = np.array([[minQ, offDiag], [offDiag, minQ]], dtype=np.float32)
    matrixQCopy = matrixQ.copy()
    vectorY = (np.random.normal(0, 15, size=m)).astype(np.float32)
    dStat = np.mean(vectorY**2).astype(np.float32)
    dStatDiff = np.float32(
        np.sqrt(np.abs(dStat - dStatAlpha) * dStatd + dStatPC)
    )

    matrixQ, inflatedQ = cconsenrich.updateProcessNoiseCovariance(
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

    assert inflatedQ is True
    np.testing.assert_allclose(matrixQ, maxQ * np.eye(2), rtol=0.01)


@pytest.mark.correctness
def testbedMask(tmp_path):
    bedPath = tmp_path / "testTmp.bed"
    bedPath.write_text("chr1\t50\t2000\nchr1\t3000\t5000\nchr1\t10000\t20000\n"
    )
    intervals = np.arange(500, 10_000, 25)
    mask = core.getBedMask('chr1', bedPath, intervals)

    # first test: mask and intervals equal length
    assert len(mask) == len(intervals)

    for i, interval_ in enumerate(intervals):
        if 50 <= interval_ < 2000 or 3000 <= interval_ < 5000:
            assert mask[i] == 1
        else:
            assert mask[i] == 0


@pytest.mark.correctness
def testgetPrecisionWeightedResidualWithCovar():
    np.random.seed(0)
    n, m = 5, 3
    postFitResiduals = np.random.randn(n, m).astype(np.float32)
    matrixMunc = (np.random.rand(m, n).astype(np.float32) * 2.0) + 0.5
    add_vec = (np.random.rand(n).astype(np.float32) * 0.5)
    stateCovarSmoothed = np.zeros((n, 2, 2), dtype=np.float32)
    stateCovarSmoothed[:, 0, 0] = add_vec
    totalUnc = matrixMunc + add_vec
    weights = 1.0 / totalUnc
    expected = (postFitResiduals * weights.T).sum(axis=1) / weights.sum(axis=0)
    out = core.getPrecisionWeightedResidual(
        postFitResiduals=postFitResiduals,
        matrixMunc=matrixMunc,
        roundPrecision=6,
        stateCovarSmoothed=stateCovarSmoothed)
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


@pytest.mark.correctness
def testgetPrimaryStateF64():
    xVec = np.array([
            [1.2349,  0.0],
            [-2.5551, 0.0],
            [10.4446, 0.0],
            [-0.5001, 0.0]],
        dtype=np.float64,
    )
    stateVectors = xVec[:, :]
    out = core.getPrimaryState(stateVectors, roundPrecision=3)
    np.testing.assert_array_equal(out.dtype, np.float32)
    np.testing.assert_allclose(out, np.array([1.235, -2.555, 10.445, -0.500], dtype=np.float32), rtol=0, atol=0)
