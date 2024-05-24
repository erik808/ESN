import numpy as np
import scipy

from scipy import sparse
from scipy.sparse import linalg


class KSmodel:
    '''KSmodel: Kuramoto-Sivashinsky PDE, normalized to the interval [0,1]
 with periodic boundaries.  d/dt u + u du/dx + (1+epsilon)*(d/dx)^2 u
 + nu * (d/dx)^4 u = 0

    '''

    def __init__(self, L, N):
        ''' constructor '''
        
        self.L  = L
        self.N  = N
        self.nx = N
        self.dx = 1 / self.N
        self.epsilon = 0.0
        
        self.V  = sparse.eye(self.N, self.N)
        
        self.x_init = np.zeros((self.N, 1))
        self.x_init[0] = 1

