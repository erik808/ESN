import numpy as np
import scipy.io as sio
import pytest
import ESN
from importlib import reload
reload(ESN)
from ESN import ESN

def create_data(dt, init, dataPoints):
    '''Create Lorenz '63 data.
    dt:   timestep size
    init: initial value (x,y,z)'''

    Nu = 3

    # Lorenz '63 parameters
    rho = 28
    sigma = 10
    beta = 8/3

    oldStates = np.zeros((dataPoints, Nu))
    newStates = np.zeros((dataPoints, Nu))

    oldStates[0, :] = init

    x = init[0]
    y = init[1]
    z = init[2]

    for n in range(dataPoints):
        oldStates[n, :] = [x, y, z]
        x = x + dt * sigma * (y - x)
        y = y + dt * (x * (rho - z) - y)
        z = z + dt * (x*y - beta*z)

        newStates[n, :] = [x, y, z]

    return oldStates, newStates

def create_esn_lorenz63():
    dt = 0.02
    Tend = 125
    T = int(Tend / dt)
    U, Y = create_data(dt, [0.1, 0.1, 0.1], T)

    # setup training and testing data
    cutoff = int(np.ceil(0.8 * T))
    trainU = U[:cutoff, :]
    trainY = Y[:cutoff, :]
    testU = U[cutoff:, :]
    testY = Y[cutoff:, :]

    return trainU, trainY

def load_testdata_lorenz63():
    test = sio.loadmat('../matlab/testdata_Lorenz64.mat')
    trainU = test['trainU']
    trainY = test['trainY']
    return trainU, trainY

def setup_esn(trainU, trainY):
    # input/output dimensions
    Nu = trainU.shape[1]
    Ny = trainY.shape[1]

    # create ESN
    Nr = 300
    esn = ESN(Nr, Nu, Ny)

    # parameter settings
    set_default_params(esn)
    return esn

def set_default_params(esn):
    # config for Lorenz63 prediction
    esn.rhoMax = 1.2
    esn.Wconstruction = 'entriesPerRow'
    esn.entriesPerRow = 8
    esn.inputMatrixType = 'full'
    esn.inAmplitude = 1.0
    esn.feedbackMatrixType = 'full'
    esn.ofbAmplitude = 0.0
    esn.feedThrough = False
    esn.pinvTol = 1e-3
    esn.alpha = 1.0
    esn.regressionSolver = 'TikhonovTSVD'

def _test_scaling(scalingType, shiftU_target, shiftY_target,
                  scaleU_target, scaleY_target):

    trainU, trainY = load_testdata_lorenz63()
    esn = setup_esn(trainU, trainY)
    # additional parameter settings
    esn.scalingType = scalingType

    # training
    esn.initialize()
    esn.train(trainU, trainY)

    # test shifts and scalings
    assert isinstance(esn.shiftU, np.ndarray)
    assert isinstance(esn.shiftY, np.ndarray)
    assert isinstance(esn.scaleU, np.ndarray)
    assert isinstance(esn.scaleY, np.ndarray)
    assert esn.shiftU == pytest.approx(shiftU_target, abs=1e-4)
    assert esn.shiftY == pytest.approx(shiftY_target, abs=1e-4)
    assert esn.scaleU == pytest.approx(scaleU_target, abs=1e-4)
    assert esn.scaleY == pytest.approx(scaleY_target, abs=1e-4)

def test_minMax1():
    _test_scaling('minMax1',
                  [-18.3204, -23.7338, 0.0792],
                  [-18.3204, -23.7338, 0.0792],
                  [0.0254, 0.0189, 0.0199],
                  [0.0254, 0.0189, 0.0199])

def test_minMax2():
    _test_scaling('minMax2',
                  [1.4023, 2.7198, 25.1773],
                  [1.4023, 2.7198, 25.1773],
                  [0.0507, 0.0378, 0.0398],
                  [0.0507, 0.0378, 0.0398])

if __name__=='__main__':

    test_minMax1()
    test_minMax2()
