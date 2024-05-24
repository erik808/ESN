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


L = 35
N = 64
epsilon = 0.1
ks_prf = KSmodel(L, N) # perfect model
ks_imp = KSmodel(L, N) # imperfect model
ks_imp.epsilon = epsilon # set perturbation parameter
# ks_prf.initialize()
# ks_imp.initialize()
