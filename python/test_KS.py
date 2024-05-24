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

data = sio.loadmat('../matlab/testdata_KS.mat')


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


def test_KSmodel_f():
    _test_KSmodel(component='f')

def test_KSmodel_J():
    _test_KSmodel(component='J')

def test_KSmodel_Jlin():
    _test_KSmodel(component='Jlin')

# def test_KSmodel_step():
#     _test_KSmodel(component='step')




if __name__=='__main__':
    test_KSmodel_J()
    test_KSmodel_f()
    test_KSmodel_Jlin()
    # test_KSmodel_step()
