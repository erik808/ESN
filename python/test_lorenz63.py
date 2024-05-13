import numpy as np
import scipy.io as sio
import pytest
import ESN
from importlib import reload
import asciichartpy as acp

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
    esn.scalingType = 'minMax1'
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

def _test_W(Wconstruction, final_entries, test_norm,
            do_value_test=True, norm_tol=1e-6):

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
    if do_value_test:
        assert mv_arr[-10:] == pytest.approx(final_entries,
                                             abs=1e-8)
    assert np.linalg.norm(mv_arr) == pytest.approx(test_norm,
                                                   abs=norm_tol)

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

def test_W_avgDegree():
    # final_entries = [
    #     1.261483629484617,
    #     1.027458587570080,
    #     0.823432495274333,
    #     0.200145110644282,
    #     0.563179096289681,
    #     0.767919358617824,
    #     1.396191514366894,
    #     0.138388026820576,
    #     0.040093351466835,
    #     0.718454690519772]

    # for now disabled, implementation of matlab sprand and
    # sparse.rand are prob quite different.
    final_entries = []

    # Doing a rough test only on the norm instead.
    test_norm = 13.2609592
    _test_W('avgDegree', final_entries, test_norm,
            do_value_test=False, norm_tol=1)

def _test_Win(inputMatrixType, inAmplitude, test_val, test_nrm):
    # training data
    trainU, trainY = load_testdata_lorenz63()

    # seeding
    np.random.seed(1)
    esn = setup_esn(trainU, trainY)
    esn.inputMatrixType = inputMatrixType
    esn.inAmplitude = inAmplitude
    esn.initialize()

    np.random.seed(1)
    test_array = np.random.rand(esn.Nu)
    prod = esn.W_in @ test_array

    assert prod[-1] == pytest.approx(test_val, abs=1e-6)
    assert np.linalg.norm(prod) == pytest.approx(test_nrm, abs=1e-6)

def test_Win_full():
    _test_Win(inputMatrixType = 'full',
              inAmplitude = 1,
              test_val = 0.244289994102627,
              test_nrm = 8.172723040202341)

def test_Win_balancedSparse():
    _test_Win(inputMatrixType = 'balancedSparse',
              inAmplitude = 1,
              test_val = 4.412244631073649e-05,
              test_nrm = 4.890732268750472)

def test_Win_sparse():
    _test_Win(inputMatrixType = 'sparse',
              inAmplitude = 1,
              test_val = 1.326227017595026e-05,
              test_nrm = 4.565567114152793)

def test_Win_sparseOnes():
    _test_Win(inputMatrixType = 'sparseOnes',
              inAmplitude = 1,
              test_val = 1.143748173448866e-04,
              test_nrm = 8.198031104683986)

def test_Win_identity():
    _test_Win(inputMatrixType = 'identity',
              inAmplitude = 1,
              test_val = 0,
              test_nrm = 0.832330908557681)

def test_Win_inAmplitude():
    _test_Win(inputMatrixType = 'identity',
              inAmplitude = 1.234,
              test_val = 0,
              test_nrm = 1.027096341160178)

def _test_Wofb(feedbackMatrixType, ofbAmplitude, test_val, test_nrm):
    # training data
    trainU, trainY = load_testdata_lorenz63()

    # seeding
    np.random.seed(1)
    esn = setup_esn(trainU, trainY)
    esn.feedbackMatrixType = feedbackMatrixType
    esn.ofbAmplitude = ofbAmplitude
    esn.initialize()

    np.random.seed(1)
    test_array = np.random.rand(esn.Nu)
    prod = esn.W_ofb @ test_array

    assert prod[-1] == pytest.approx(test_val, abs=1e-6)
    assert np.linalg.norm(prod) == pytest.approx(test_nrm, abs=1e-6)

def test_Wofb_sparse():
    _test_Wofb(feedbackMatrixType = 'sparse',
               ofbAmplitude = 0.1,
               test_val = -0.038553473052732,
               test_nrm = 0.497486248307358)

def test_Wofb_full():
    _test_Wofb(feedbackMatrixType = 'full',
               ofbAmplitude = 0.1,
               test_val = -0.032721482782959,
               test_nrm = 0.850742658756709)

def _test_training(feedthrough,
                   test_nrm1,
                   test_val,
                   test_nrm2,
                   test_nrm3):
    # training data
    trainU, trainY = load_testdata_lorenz63()

    # seeding
    np.random.seed(1)
    esn = setup_esn(trainU, trainY)
    esn.feedThrough=False
    esn.initialize()
    esn.train(trainU, trainY)
    nrmX = np.linalg.norm(esn.X[-1,:])
    assert nrmX == pytest.approx(test_nrm1, abs=1e-6)

    np.random.seed(1)
    test_array = np.random.rand(esn.Nr)
    prod = esn.W_out @ test_array

    assert prod[-1] == pytest.approx(test_val, abs=1e-6)
    assert np.linalg.norm(prod) == pytest.approx(test_nrm2, abs=1e-6)

    state = esn.X[-1,:]
    assert np.linalg.norm(state) == pytest.approx(test_nrm3, abs=1e-6)
    return esn

def test_training_no_FT():
    _test_training(feedthrough=False,
                   test_nrm1=8.897798472187780,
                   test_val=0.241379038528080,
                   test_nrm2=0.452108247200167,
                   test_nrm3=8.897798472187780)


def _test_prediction(feedThrough,
                     test_nrm1,
                     test_nrm2):

    # training data
    trainU, trainY = load_testdata_lorenz63()

    # seeding
    np.random.seed(1)
    esn = setup_esn(trainU, trainY)
    esn.feedThrough=feedThrough
    esn.initialize()
    esn.train(trainU, trainY)

    state = esn.X[-1,:]

    test = sio.loadmat('../matlab/testdata_Lorenz64.mat')
    testU = test['testU']
    testY = test['testY']
    trainY = test['trainY']
    nPred = np.shape(testU)[0]
    dimY = np.shape(testY)[1]
    predY = np.zeros((nPred, dimY))

    u = (trainY[-1,:] - esn.shiftU) * esn.scaleU;

    assert np.linalg.norm(u) == pytest.approx(test_nrm1, abs=1e-6)

    predY[0,:] = u
    for k in range(1, nPred):
        state = esn.update(state, u, u)
        if esn.feedThrough:
            u = esn.f_out(esn.W_out @ np.append(state,u))
        else:
            u = esn.f_out(esn.W_out @ state)
        predY[k,:] = u

    predY = ( predY / esn.scaleY ) + esn.shiftY

    test_arr = predY[500,:]
    assert np.linalg.norm(test_arr) == pytest.approx(test_nrm2, abs=1e-6)

    print('')
    print(acp.plot([predY[:100,0].tolist(),testY[0:100,0].tolist()]))


def test_prediction_no_FT():

    _test_prediction(feedThrough=False,
                     test_nrm1=0.676582659999415,
                     test_nrm2=24.433495297140841)


if __name__=='__main__':

    # test_minMax1()
    # test_minMax2()
    # test_minMaxAll()
    # test_standardize()

    # test_W_entriesPerRow()
    # test_W_sparsity()
    # test_W_avgDegree()

    # test_Win_full()
    # test_Win_balancedSparse()
    # test_Win_sparse()
    # test_Win_sparseOnes()
    # test_Win_identity()
    # test_Win_inAmplitude()

    # test_Wofb_sparse()
    # test_Wofb_full()

    # test_training()
    test_prediction_no_FT()
