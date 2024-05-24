import numpy as np
import scipy.io as sio
import pytest
import ESN
import KSmodel

from importlib import reload

reload(ESN)
reload(KSmodel)

from ESN import ESN
from KSmodel import KSmodel

def _standard_KS_setup():
    L = 35
    N = 64
    epsilon = 0.1
    ks_prf = KSmodel(L, N) # perfect model
    ks_imp = KSmodel(L, N) # imperfect model
    ks_imp.epsilon = epsilon # set perturbation parameter
    ks_prf.initialize()
    ks_imp.initialize()

    return ks_prf, ks_imp

def _test_KSmodel(component='Jlin'):
    ks_prf, ks_imp = _standard_KS_setup()

    np.random.seed(1)
    test_vec = np.random.rand(ks_prf.N)

    if component == 'Jlin':
        out1 = ks_prf.Jlin @ test_vec
        out2 = ks_imp.Jlin @ test_vec
        testnorm1 = np.linalg.norm(out1)
        testnorm2 = np.linalg.norm(out2)

        truth1 = 2.084188310078326e+02
        truth2 = 2.064645558520709e+02

        assert testnorm1 == pytest.approx(truth1, abs=1e-6)
        assert testnorm2 == pytest.approx(truth2, abs=1e-6)

    test_vec = np.random.rand(ks_prf.N)

    if component == 'f':
        out3 = ks_prf.f(test_vec)
        out4 = ks_imp.f(test_vec)
        testnorm3 = np.linalg.norm(out3)
        testnorm4 = np.linalg.norm(out4)
        truth3 = 2.145306171150842e+02
        truth4 = 2.124926738354342e+02
        assert testnorm3 == pytest.approx(truth3, abs=1e-6)
        assert testnorm4 == pytest.approx(truth4, abs=1e-6)

    test_vec = np.random.rand(ks_prf.N)
    test_vec2 = np.random.rand(ks_prf.N)

    if component == 'J':
        out5 = ks_prf.J(test_vec) @ test_vec2
        out6 = ks_imp.J(test_vec) @ test_vec2
        testnorm5 = np.linalg.norm(out5)
        testnorm6 = np.linalg.norm(out6)
        truth5 = 2.207765006161888e+02
        truth6 = 2.187698503564701e+02
        assert testnorm5 == pytest.approx(truth5, abs=1e-6)
        assert testnorm6 == pytest.approx(truth6, abs=1e-6)


    if component == 'step':
        x_init = ks_prf.x_init
        dt = 0.25
        y, k = ks_prf.step(x_init, dt)
        testnorm = np.linalg.norm(y)
        truth = 5.215431600942750e-01
        assert k == 3
        assert testnorm == pytest.approx(truth, abs=1e-6)

        y, k = ks_imp.step(x_init, dt)
        testnorm = np.linalg.norm(y)
        truth = 5.297663697680071e-01
        assert k == 3
        assert testnorm == pytest.approx(truth, abs=1e-6)

    if component == 'transient':
        y = ks_prf.x_init
        dt = 0.25
        for t in range(100):
            y, k = ks_prf.step(y, dt)

        testnorm = np.linalg.norm(y)
        truth = 10.759000089691213
        assert testnorm == pytest.approx(truth, abs=1e-6)

def test_KSmodel_f():
    _test_KSmodel(component='f')

def test_KSmodel_J():
    _test_KSmodel(component='J')

def test_KSmodel_Jlin():
    _test_KSmodel(component='Jlin')

def test_KSmodel_step():
    _test_KSmodel(component='step')

def test_KSmodel_transient():
    _test_KSmodel(component='transient')



def _test_KS_ESN(feedThrough=True):
    ks_prf, ks_imp = _standard_KS_setup()
    data = sio.loadmat('../matlab/testdata_KS.mat')

    X = data['X']
    Phi = data['Phi']
    NT = data['NT']
    dt = data['dt'][0][0]
    train_range=range(99,2100)
    test_range=range(2100,2500)

    # input data
    # restricted truths and imperfect predictions
    if feedThrough:
        U = np.vstack((X[:,:-1],Phi[:,:-1]))
    else:
        U = X[:,:-1]

    Y = X[:, 1:] # perfect predictions
    trainU = U[:, train_range].T
    trainY = Y[:, train_range].T
    testU  = U[:, test_range].T
    testY  = Y[:, test_range].T

    # ESNc settings:
    esn_pars = {}
    esn_pars['scalingType']        = 'standardize'
    esn_pars['Nr']                 = 100
    esn_pars['rhoMax']             = 0.4
    esn_pars['alpha']              = 1.0
    esn_pars['Wconstruction']      = 'avgDegree'
    esn_pars['avgDegree']          = 3
    esn_pars['tikhonov_lambda']    = 1e-10
    esn_pars['bias']               = 0.0
    esn_pars['squaredStates']      = 'even'
    esn_pars['reservoirStateInit'] = 'random'
    esn_pars['inputMatrixType']    = 'balancedSparse'
    esn_pars['inAmplitude']        = 1.0
    esn_pars['waveletBlockSize']   = 1.0
    esn_pars['waveletReduction']   = 1.0
    esn_pars['feedThrough']        = feedThrough
    esn_pars['ftRange']            = range(N, 2*N)
    esn_pars['fCutoff']            = 0.1

    np.random.seed(1)
    esn = ESN(esn_pars['Nr'], trainU.shape[1], trainY.shape[1])
    esn.setPars(esn_pars)
    esn.initialize()
    esn.train(trainU, trainY)


    # Prediction
    # initialization index
    init_idx = train_range[-1]+1
    # initial state for the predictions
    yk = X[:, init_idx]

    Npred = len(test_range)
    predY = np.zeros((Npred, N))
    esn_state = esn.X[-1,:]
    for i in range(Npred):
        Pyk, Nk = ks_imp.step(yk, dt)
        if feedThrough:
            u_in = np.append(yk.squeeze(), Pyk.squeeze())
        else:
            u_in = yk.squeeze()

    u_in      = np.expand_dims(u_in, axis=0)
    u_in      = esn.scaleInput(u_in)
    esn_state = esn.update(esn_state, u_in)
    u_out     = esn.apply(esn_state, u_in)
    u_out     = np.expand_dims(u_out, axis=0)
    yk        = esn.unscaleOutput(u_out)
    predY[i,:] = yk

    breakpoint()

if __name__=='__main__':
    _test_KS_ESN()
