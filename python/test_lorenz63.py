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
    esn.computeScaling(trainU, trainY)

    coldim = np.shape(trainU)[1]
    assert len(esn.shiftU) == coldim, 'incorrect shiftU dimension'
    assert len(esn.shiftY) == coldim, 'incorrect shiftY dimension'
    assert len(esn.scaleU) == coldim, 'incorrect scaleU dimension'
    assert len(esn.scaleY) == coldim, 'incorrect scaleY dimension'

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

def test_minMaxAll():
    _test_scaling('minMaxAll',
                  [13.2708, 13.2708, 13.2708],
                  [13.2708, 13.2708, 13.2708],
                  [0.0270 , 0.0270 , 0.0270],
                  [0.0270 , 0.0270 , 0.0270])

def test_standardize():
    _test_scaling('standardize',
                  [-0.821302058844881,-0.821549171591466, 23.993174055920623],
                  [-0.821351481394199,-0.821629117825128, 23.995399340318425],
                  [0.123450523501821, 0.109698216148058, 0.116749992517157],
                  [0.123450597695880, 0.109698292330504, 0.116814974565204])

def _test_W(Wconstruction, final_entries, test_norm):
    # training data
    trainU, trainY = load_testdata_lorenz63()

    # seeding
    np.random.seed(1)
    esn = setup_esn(trainU, trainY)
    esn.Wconstruction = Wconstruction
    esn.initialize()

    # test matrix-vector product with W on random vector
    np.random.seed(1)
    test_array = np.random.rand(esn.Nr)

    mv_arr = esn.W @ test_array

    # check final 10 entries
    assert mv_arr[-10:] == pytest.approx(final_entries,
                                         abs=1e-8)
    assert np.linalg.norm(mv_arr) == pytest.approx(test_norm,
                                                   abs=1e-6)

def test_W_entriesPerRow():
    final_entries = [
        0.115515876153812,
        0.095260089879517,
        1.251568402435540,
        -0.268846439126479,
        -0.166194137874266,
        0.254244828208688,
        -1.364218695779681,
        -0.391941646487290,
        -0.315976624922663,
        -0.207732148746306]

    test_norm = 12.3712141
    _test_W('entriesPerRow', final_entries, test_norm)

def test_W_sparsity():
    final_entries = [
        -1.114304736887016,
        -0.275501426501573,
        -0.167940289742099,
        -0.332332166370453,
        0.682000132505937,
        -0.403856695391554,
        0.211243072511238,
        -0.289195199687055,
        1.049821790132286,
        0.729814893991415]

    test_norm = 11.4518966
    _test_W('sparsity', final_entries, test_norm)

if __name__=='__main__':
    test_minMax1()
    test_minMax2()
    test_minMaxAll()
    test_standardize()

    test_W_entriesPerRow()
    test_W_sparsity()
