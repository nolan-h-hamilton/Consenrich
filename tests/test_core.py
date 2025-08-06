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
    out = core.getAverageLocalVarianceTrack(values,
    stepSize, approximationWindowLengthBP,
    lowPassWindowLengthBP,
    minR,
    maxR)
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
    out = core.getAverageLocalVarianceTrack(values,
    stepSize,
    approximationWindowLengthBP,
    lowPassWindowLengthBP,
    minR,
    maxR)
    np.testing.assert_allclose(stats.mode(out)[0], maxR, rtol=0.001)


@pytest.mark.correctness
def testMatrixConstruction(deltaF = 0.50,
    coefficients = [0.1, 0.2, 0.3, 0.4],
    minQ = 0.25,
    offDiag=0.10):

    # F
    m = len(coefficients)
    matrixF = core.constructMatrixF(deltaF)
    assert matrixF.shape == (2,2)
    np.testing.assert_allclose(matrixF, np.array([[1.0,deltaF],[0.0,1.0]]))

    # H
    matrixH = core.constructMatrixH(m,coefficients)
    assert matrixH.shape == (m,2)
    np.testing.assert_allclose(matrixH[:, 0], coefficients)
    np.testing.assert_allclose(matrixH[:, 1], np.zeros(m))

    # Q
    matrixQ = core.constructMatrixQ(minQ, offDiag)
    assert matrixQ.shape == (2,2)
    np.testing.assert_allclose(matrixQ, np.array([[minQ, offDiag], [offDiag, minQ]]))


@pytest.mark.chelpers
def testResidualCovarianceInversion():
    np.random.seed(42)
    m=10
    muncMatrixIter=np.random.gamma(shape=2, scale=1.0, size=m) + 1
    priorCovarianceOO= 0.1
    residCovar = np.diag(muncMatrixIter) + (np.ones((m,m))*priorCovarianceOO)

    invertedMatrix = cconsenrich.cinvertMatrixE(muncMatrixIter,
    priorCovarianceOO)
    np.testing.assert_allclose(invertedMatrix@residCovar, np.eye(m),
    atol=1e-8)


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

    matrixQ = np.array([[minQ, offDiag], [offDiag, minQ]])
    matrixQCopy = matrixQ.copy()
    vectorY = (np.random.normal(0, 15, size=m))
    dStat = np.mean(vectorY**2)
    dStatDiff = np.sqrt(np.abs(dStat - dStatAlpha)*dStatd + dStatPC)

    matrixQ, inflatedQ = cconsenrich.updateProcessNoiseCovariance(matrixQ,
    matrixQCopy,
    dStat,
    dStatAlpha,
    dStatd,
    dStatPC,
    inflatedQ,
    maxQ,
    minQ)

    assert inflatedQ is True
    np.testing.assert_allclose(matrixQ, maxQ*np.eye(2), rtol=0.01)
