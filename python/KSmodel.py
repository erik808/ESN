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

        # grid points in the y-direction
        self.ny  = 1 # not relevant but should be available

        # number of unknowns
        self.nun = 1 # not relevant but should be available

        self.dx = 1 / self.N
        self.epsilon = 0.0

        # Newton tolerance
        self.Ntol = 1e-6

        # Max Newton iterations
        self.Nkmx = 10


        self.V  = sparse.eye(self.N, self.N)

        self.x_init = np.zeros((self.N, 1))
        self.x_init[0] = 1

        self.initialized = False

    def initialize(self):
        self.computeMatrices()
        self.initialized = True

    def computeMatrices(self):
        """discretization operators for first, second and fourth derivative"""
        e = np.ones((self.N))

        # central first deriv
        self.D1 = sparse.spdiags([e, 0*e, -1*e], [-1,0,1],
                                 m=self.N, n=self.N,
                                 format='dok')
        # Periodic boundaries. Using dok format for incremental
        # additions.
        self.D1[0,-1] = [1]
        self.D1[-1,0] = [-1]

        # set dx, convert to csr format.
        self.D1 = (self.D1 / (2*self.dx)).asformat('csr')

        # second derivative, central discretization
        self.D2 = sparse.spdiags([e, -2*e, e], [-1,0,1],
                          m=self.N, n=self.N,
                          format='dok')

        self.D2[0,-1] = [1]
        self.D2[-1,0] = [1]
        self.D2 = (self.D2 / (self.dx**2)).asformat('csr')

        # fourth derivative, central discretization
        self.D4 = sparse.spdiags([e, -4*e, 6*e, -4*e, e],
                                 [-2,-1,0,1,2],
                                 m=self.N, n=self.N,
                                 format='dok')

        self.D4[0,-2:] = [1, -4]
        self.D4[1,-1]  = [1]
        self.D4[-2,0]  = [1]
        self.D4[-1,:2] = [-4, 1]
        self.D4 = (self.D4 / (self.dx**4)).asformat('csr')

        # linear part of the jacobian
        self.Jlin = ((1+self.epsilon)/self.L**2)*self.D2 + (1/self.L**4)*self.D4

    def f(self, y):
        """ RHS """
        assert len(y) == self.N
        assert self.initialized, 'KSmodel not initialized'

        out = (- ((1/self.L)*y) * (self.D1 @ y)
               - ((1+self.epsilon)/self.L**2)*self.D2 @ y
               - (1/self.L**4)*self.D4 @ y)

        return out

    def g(self, yp, ym, dt, frc):
        """ time dependent rhs (backward Euler) """
        assert len(frc) == self.N
        out = yp - ym - dt * (self.f(yp) + frc)
        return out

    def J(self, y):
        """ Jacobian """
        assert len(y) == self.N
        assert y.shape[0] == self.N
        assert self.initialized, 'KSmodel not initialized'

        y = y.squeeze()
        dydx = self.D1 @ y;
        D1y  = sparse.csr_matrix( (dydx,
                                   (range(self.N), range(self.N))),
                                 (self.N, self.N) )
        yD1  = ((sparse.csr_matrix((y,
                                    (range(self.N),
                                     np.append(self.N-1, range(self.N-1))))) +
                 sparse.csr_matrix((-y,
                                    (range(self.N),
                                     np.append(range(1,self.N), 0))))) /
                (2*self.dx))

        out  =  -(1/self.L)*(yD1 + D1y) - self.Jlin
        return out

    def H(self, y, dt):
        """ time dependent Jacobian (backward Euler) """
        out = (sparse.eye(self.N, format='csr') -
               dt * self.J(y))
        return out

    def  step(self, y, dt, frc=0):
        """ perform single step time integration """
        ym = y

        # Newton
        for k in range(self.Nkmx):
            H = self.H(y, dt)
            g = self.g(y, ym, dt, frc)
            breakpoint()
            # dy = H \ -g;
            # y  = y + dy;

            if (np.linalg.norm(dy) < self.Ntol):
                break

            if k == self.Nkmx:
                raise Exception('KS:convergenceError '
                                'no convergence in Newton iteration')


        return y, k
