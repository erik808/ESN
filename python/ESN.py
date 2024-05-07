import time
import numpy as np
import scipy

from scipy import sparse
from scipy.sparse import linalg


class ESN:
    '''ESN: an Echo State Network'''

    def __init__(self, Nr, Nu, Ny):

        self.Nr = Nr  # reservoir state dimension
        self.Nu = Nu  # input dimension
        self.Ny = Ny  # output dimension

        self.W = None  # reservoir weight matrix
        self.W_in = None  # input weight matrix
        self.W_ofb = None  # output weight feedback matrix
        self.W_out = None  # output weight matrix

        # spectral radius of the weight matrix W
        self.rhoMax = 0.95

        # method to construct W: 'entriesPerRow', 'sparsity', 'avgDegree'
        self.Wconstruction = 'entriesPerRow'

        # average degree of the graph W
        self.avgDegree = 3

        # Wconstruction, entriesPerRow option: avg number of entries per row
        self.entriesPerRow = 10

        # Wconstruction, sparsity option: percentage of zeros
        self.sparsity = 0.95

        # control noise
        self.noiseAmplitude = 0.0

        # leaking rate
        self.alpha = 1.0

        # bias
        self.bias = 0.0  # bias vector

        self.shiftU = 0  # input shift
        self.shiftY = 0  # output shift

        self.scaleU = 1  # input scaling
        self.scaleY = 1  # output scaling

        # 'none'        : no scaling or user defined
        # 'minMax1'     : rescale features to [0,1]
        # 'minMax2'     : rescale features to [-1,1]
        # 'minMaxAll'   : rescale data to [-1,1] over all features
        # 'standardize' : rescale feature range to zero mean and unit variance
        self.scalingType = 'none'

        self.X = None  # reservoir state activations

        # for fitting take the square of the even components, this fixes
        # issues with symmetry
        # 'disabled'
        # 'append'
        # 'even'
        self.squaredStates = 'disabled'

        # control feedthrough of u to y: (part of) the input state is appended
        # to the reservoir state activations X and used to fit W_out
        self.feedThrough = False

        # select a subset of the input data u to feedthrough
        self.ftRange = None

        # feedthrough amplitude
        self.ftAmp = 1

        # reservoir amplitude
        self.resAmp = 1

        # mean center reservoir states before fitting
        self.centerX = False

        # mean center training output data before fitting
        self.centerY = False

        # set the activation function f: 'tanh'
        self.activation = 'tanh'

        # activation function
        self.f = lambda x: np.tanh(x)

        # set the output activation: 'identity' or 'tanh'
        self.outputActivation = 'identity'

        # output activation
        self.f_out = lambda y: y

        # inverse output activation
        self.if_out = lambda y: y

        # set the inital state of X: 'random' or 'zero'
        self.reservoirStateInit = 'zero'

        # set the input weight matrix type: 'sparse', 'sparseOnes',
        # 'balancedSparse', 'full', 'identity'.
        self.inputMatrixType = 'sparse'

        # control size input weight matrix
        self.inAmplitude = 1.0

        # set the feedback weight matrix type: 'sparse' or 'full'
        self.feedbackMatrixType = 'sparse'

        # control output feedback
        self.ofbAmplitude = 0.0

        # set the method to solve the linear least squares problem to compute
        # W_out: 'TikhonovTSVD'
        #        'TikhonovNormalEquations
        #        'pinv'
        self.regressionSolver = 'TikhonovTSVD'

        # lambda (when using Tikhonov regularization)
        self.tikhonov_lambda = 1.0e-4

        # tolerance for the pseudo inverse
        self.pinvTol = 1.0e-4

        # For use with TikhonovTSVD. If > 1 we perform a blockwise wavelet
        # projection with blocksize <waveletBlockSize> on the time
        # domain and reduce with a factor <waveletReduction>.
        self.waveletReduction = 1
        self.waveletBlockSize = 1

        # Array that stores the damping coefficients computed after a
        # TikhonovTSVD solve
        self.TikhonovDamping = 0

    def initialize(self):
        '''Create W, W_in, W_ofb and set output activation function'''

        self.createW()
        self.createW_in()
        self.createW_ofb()

        if self.activation == 'tanh':
            self.f = lambda x: np.tanh(x)

        if self.outputActivation == 'tanh':
            self.f_out = lambda y: np.tanh(y)
            self.if_out = lambda y: np.atanh(y)
        elif self.outputActivation == 'identity':
            self.f_out = lambda y: y
            self.if_out = lambda y: y
        else:
            raise Exception('Invalid outputActivation parameter')

        print('ESN initialization done')

    def createW(self):
        '''Create sparse weight matrix with spectral radius rhoMax'''

        if self.Wconstruction == 'entriesPerRow':
            print('ESN avg entries/row in W: %d' % self.entriesPerRow)
            row_ind = []
            col_ind = []
            data = []
            for i in range(self.entriesPerRow):
                row_ind = np.append(row_ind,
                                    np.arange(self.Nr))
                col_ind = np.append(col_ind,
                                    np.ceil(self.Nr * np.random.rand(self.Nr))-1)
                data = np.append(data,
                                 (np.random.rand(self.Nr)-0.5))
            self.W = sparse.csr_matrix((data, (row_ind, col_ind)),
                                       (self.Nr, self.Nr))

        elif self.Wconstruction == 'sparsity':
            self.W = np.random.rand(self.Nr, self.Nr)-0.5;
            rmv_index = np.where(np.random.rand(self.Nr, self.Nr) < self.sparsity)
            self.W[rmv_index] = 0
            self.W = sparse.csr_matrix(self.W.T)

        elif self.Wconstruction == 'avgDegree':
            self.W = sparse.rand(self.Nr, self.Nr,
                                 density=self.avgDegree / self.Nr,
                                 format='csr')
        else:
            raise Exception('Invalid Wconstruction parameter')

        rho, _ = linalg.eigs(self.W, 3)
        mrho = max(abs(rho))

        print('ESN largest eigenvalue of W: %f' % mrho)

        # adjust spectral radius of W
        self.W = self.W * self.rhoMax / mrho

    def createW_in(self):
        '''Create input weight matrix, sparse or full based on
        inputMatrixType'''

        if self.inputMatrixType == 'sparse':
            # Create sparse input weight matrix. This gives a single random
            # connection per row, in a random column.
            row_idx = np.arange(self.Nr)
            col_idx = np.random.randint(0, self.Nu, self.Nr)
            val = np.random.rand(self.Nr) * 2 - 1
            self.W_in = sparse.csc_matrix((val, (row_idx, col_idx)),
                                          (self.Nr, self.Nu))
        elif self.inputMatrixType == 'sparseOnes':
            row_idx = np.arange(self.Nr)
            col_idx = np.random.randint(0, self.Nu, self.Nr)
            val = np.ones(self.Nr)
            self.W_in = sparse.csc_matrix((val, (row_idx, col_idx)),
                                          (self.Nr, self.Nu))
        elif self.inputMatrixType == 'balancedSparse':
            # This implementation makes sure that every input element is
            # connected to roughly the same number of reservoir components.
            # Weights are uniform in [-1,1].

            arM = self.Nr // self.Nu  # floor
            arP = (self.Nr + self.Nu - 1) // self.Nu  # ceil
            l1 = self.Nr - arM * self.Nu
            l2 = self.Nu - l1

            row_idx = np.arange(self.Nr)

            col_idx = []
            for i in range(arP):
                col_idx = np.append(col_idx, np.arange(l1))
            for i in range(arM):
                col_idx = np.append(col_idx, np.arange(l1, l1 + l2))

            val = np.random.rand(self.Nr) * 2 - 1

            self.W_in = sparse.csc_matrix((val, (row_idx, col_idx)),
                                          (self.Nr, self.Nu))
        elif self.inputMatrixType == 'full':
            # Create a random, full input weight matrix. Taking the
            # transpose is to ensure the same behavior as the Matlab
            # code.
            self.W_in = (np.random.rand(self.Nu, self.Nr) * 2 - 1).T
        elif self.inputMatrixType == 'identity':
            self.W_in = sparse.eye(self.Nr, self.Nu)
        else:
            raise Exception('Invalid inputMatrixType parameter')

        self.W_in = self.inAmplitude * self.W_in

    def createW_ofb(self):
        '''Create output feedback weight matrix'''

        if self.feedbackMatrixType == 'sparse':
            # Create sparse output feedback weight matrix. This gives a single
            # random connection per row, in a random column.
            row_idx = np.arange(self.Nr)
            col_idx = np.random.randint(0, self.Ny, self.Nr)
            val = np.random.rand(self.Nr) * 2 - 1
            self.W_ofb = sparse.csc_matrix((val, (row_idx, col_idx)),
                                           (self.Nr, self.Ny))
        elif self.feedbackMatrixType == 'full':
            # Create a random, full output feedback weight matrix
            self.W_ofb = np.random.rand(self.Nr, self.Ny) * 2 - 1
        else:
            raise Exception('Invalid feedbackMatrixType parameter')

        self.W_ofb = self.ofbAmplitude * self.W_ofb

    def train(self, trainU, trainY):
        '''Train the ESN with training data trainU and trainY'''
        assert trainU.shape[1] == self.Nu, \
            'Input training data has incorrect dimension Nu'

        assert trainY.shape[1] == self.Ny, \
            'Output training data has incorrect dimension Ny'

        T = trainU.shape[0]

        assert T == trainY.shape[0], \
            'input and output training data have different number of samples'

        # Now that we have data we can setup the scaling
        self.computeScaling(trainU, trainY)
        print('ESN training, input Nu = %d, output Ny = %d' %
              (self.Nu, self.Ny))

        # Apply scaling
        trainU = self.scaleInput(trainU)
        trainY = self.scaleOutput(trainY)

        # initialize activations X
        if self.reservoirStateInit == 'random':
            Xinit = self.f(10 * np.random.randn(1, self.Nr))
        elif self.reservoirStateInit == 'zero':
            Xinit = np.zeros((1, self.Nr))
        else:
            raise Exception('Invalid reservoirStateInit parameter')

        X = np.append(Xinit, np.zeros((T-1, self.Nr)), 0)

        print('ESN iterate state over %d samples... ' % T)
        tstart = time.time()

        # Iterate the state, save all neuron activations in X
        for k in range(1, T):
            X[k, :] = self.update(X[k-1, :], trainU[k, :], trainY[k-1, :])

        self.X = X
        print('ESN iterate state over %d samples... done (%fs)' %
              (T, time.time() - tstart))

        tstart = time.time()
        print('ESN fitting W_out...')

        extX = self.X
        if self.squaredStates == 'append':
            extX = np.append(self.X, self.X * self.X)
        elif self.squaredStates == 'even':
            extX[:, 1:2:] = extX[:, 1:2:] * extX[:, 1:2:]

        if self.feedThrough:
            if self.ftRange is None:
                self.ftRange = np.arange(self.Nu)

            extX = np.append(self.ftAmp * trainU[:, self.ftRange],
                                self.resAmp * extX)

        if self.centerX:
            extX = extX - np.mean(extX)

        if self.centerY:
            trainY = trainY - np.mean(trainY)

        if self.regressionSolver == 'pinv':
            print('ESN  using pseudo inverse, tol = %1.1e' % self.pinvTol)
            P = np.linalg.pinv(extX, self.pinvTol)
            self.W_out = (P @ self.if_out(trainY)).T
        elif self.regressionSolver == 'TikhonovNormalEquations':
            print('ESN  using Tikhonov regularization, lambda = %1.1e' %
                  self.tikhonov_lambda)
            print(' solving normal equations: %d x %d' %
                  (extX.shape[1], extX.shape[1]))
            Xnormal = extX.T @ extX + self.tikhonov_lambda * \
                sparse.eye(extX.shape[1])
            b = extX.T @ self.if_out(trainY)
            self.W_out = np.linalg.solve(Xnormal, b).T
        elif self.regressionSolver == 'TikhonovTSVD':
            print('ESN using TSVD and Tikhonov regularization, lambda = %1.1e'
                  % self.tikhonov_lambda)
            print(' computing SVD')
            T = extX.shape[0]
            H = sparse.eye(T)

            if self.waveletReduction > 1 and self.waveletBlockSize > 1:
                # create wavelet block
                W = self.haarmat(self.waveletBlockSize)

                # reduce wavelet block
                block_red = round(
                    self.waveletBlockSize / self.waveletReduction)
                W = W[:block_red, :]

                # repeating the block
                Nblocks = T // self.waveletBlockSize
                H = sparse.kron(sparse.eye(Nblocks), W)

                # padding the wavelet block
                Th = Nblocks * self.waveletBlockSize
                rem = T - Th
                Tr = H.shape[0]
                H = sparse.bmat(sparse.csc_matrix(Tr, rem), H).T

            print(' problem size: %d x %d' % (H.shape[1], extX.shape[1]))
            U, s, Vh = scipy.linalg.svd(H.T @ extX, False)

            # # 1500 should be a parameter #FIXME
            # [U,S,V,flag] = svds(extX, 1500, 'largest', ...
            #                     'Tolerance', 1e-6, ...
            #                     'MaxIterations', 5, ...
            #                     'Display', true, ...
            #                     'FailureTreatment','drop')
            # print('  svd flag  %d ', flag)

            f = s * s / (s * s + self.tikhonov_lambda)
            self.TikhonovDamping = f

            # filter cutoff
            fcutoff = 0.01
            fcutoff_ind = np.where(f > fcutoff)[0][-1]

            Vh = Vh[:fcutoff_ind, :]
            U = U[:, :fcutoff_ind]
            s = s[:fcutoff_ind]
            f = s * s / (s * s + self.tikhonov_lambda)
            print('  filter cutoff %1.3e at index %d' % (fcutoff, fcutoff_ind))
            print('  smallest filter coefficient: %1.3e' % f[-1])

            S = sparse.diags(s, shape=(fcutoff_ind, fcutoff_ind), format='csc')
            invReg = sparse.diags(1 / (s * s + self.tikhonov_lambda),
                                  format='csc')
            self.W_out = (Vh.T @ (invReg @ (S @ (U.T @ (H.T @ trainY))))).T
        else:
            raise Exception('Invalid regressionSolver parameter')

        print('ESN fitting W_out... done (%fs)' % (time.time() - tstart))

        # get training error
        predY = self.f_out(extX @ self.W_out.T)

        print('ESN training error: %e' %
              np.sqrt(np.mean((predY - trainY) * (predY - trainY))))

    def update(self, state, u, y=None):
        '''Update the reservoir state'''
        if y is None:
            y = np.zeros((1, self.Ny))

        pre = self.W @ state.T + self.W_in @ u.T + self.W_ofb @ y.T + self.bias

        return self.alpha * self.f(pre) + (1 - self.alpha) * state.T + \
            self.noiseAmplitude * (np.random.rand(self.Nr) - 0.5)

    def apply(self, state, u):
        x = state

        if self.squaredStates == 'append':
            x = np.append(state, state * state)
        elif self.squaredStates == 'even':
            x[1:2:] = state[1:2:] * state[1:2:]

        if self.feedThrough:
            return self.f_out(self.W_out @ np.append(
                self.ftAmp * u(self.ftRange), self.resAmp * x))
        else:
            return self.f_out(self.W_out @ x.T).T

    def computeScaling(self, U, Y):
        if self.scalingType == 'none':
            if (not isinstance(self.scaleU, np.ndarray) or
                not isinstance(self.scaleY, np.ndarray) or
                not isinstance(self.shiftU, np.ndarray) or
                not isinstance(self.shiftY, np.ndarray)):
                self.scaleU = np.tile(self.scaleU, np.shape(U)[1])
                self.scaleY = np.tile(self.scaleY, np.shape(Y)[1])
                self.shiftU = np.tile(self.shiftU, np.shape(U)[1])
                self.shiftY = np.tile(self.shiftY, np.shape(Y)[1])
            print('ESN scaling: none or user specified')

        elif self.scalingType == 'minMax1':
            self.scaleU = 1.0 / (np.max(U, axis=0) - np.min(U, axis=0))
            self.scaleY = 1.0 / (np.max(Y, axis=0) - np.min(Y, axis=0))
            self.shiftU = np.min(U, axis=0)
            self.shiftY = np.min(Y, axis=0)
            print('ESN scaling: minMax1 [0,1]')

        elif self.scalingType == 'minMax2':
            self.scaleU = 2.0 / (np.max(U, axis=0) - np.min(U, axis=0))
            self.scaleY = 2.0 / (np.max(Y, axis=0) - np.min(Y, axis=0))
            self.shiftU = np.min(U, axis=0) + 1 / self.scaleU
            self.shiftY = np.min(Y, axis=0) + 1 / self.scaleY
            print('ESN scaling: minMax2 [-1,1]')

        elif self.scalingType == 'minMaxAll':
            self.scaleU = 2.0 / (np.max(U) - np.min(U))
            self.scaleY = 2.0 / (np.max(Y) - np.min(Y))
            self.shiftU = np.min(U) + 1 / self.scaleU
            self.shiftY = np.min(Y) + 1 / self.scaleY

            self.scaleU = np.tile(self.scaleU, np.shape(U)[1])
            self.scaleY = np.tile(self.scaleY, np.shape(Y)[1])
            self.shiftU = np.tile(self.shiftU, np.shape(U)[1])
            self.shiftY = np.tile(self.shiftY, np.shape(Y)[1])
            print('ESN scaling: minMaxAll [-1,1]')

        elif self.scalingType == 'standardize':
            self.scaleU = 1.0 / np.std(U, axis=0)
            self.scaleY = 1.0 / np.std(Y, axis=0)
            self.shiftU = np.mean(U, axis=0)
            self.shiftY = np.mean(Y, axis=0)
            print('ESN scaling: standardize')
        else:
            raise Exception('invalid scalingType parameter')

        # detect constant datapoints,
        idinfU = np.where(np.isinf(self.scaleU))
        idinfY = np.where(np.isinf(self.scaleY))

        idinfU = idinfU[0]
        idinfY = idinfY[0]

        if len(idinfU) or len(idinfY):
            print(' constant data points found...  these remain unscaled')

        for i in idinfU:
            self.scaleU[i] = 1.0
            self.shiftU[i] = 0.0

        for i in idinfY:
            self.scaleY[i] = 1.0
            self.shiftY[i] = 0.0

    def scaleInput(self, x):
        # only allowing the shifts to have ndim=1 and length
        # corresponding to #columns in x
        assert (self.shiftU.ndim == 1 and
                len(self.shiftU) == x.shape[1]), \
                'incompatible dimensions'
        return (x - self.shiftU) * self.scaleU

    def scaleOutput(self, x):
        assert (self.shiftY.ndim == 1 and
                len(self.shiftY) == x.shape[1]), \
                'incompatible dimensions'
        return (x - self.shiftY) * self.scaleY

    def unscaleInput(self, x):
        assert (self.shiftU.ndim == 1 and
                len(self.shiftU) == x.shape[1]), \
                'incompatible dimensions'
        return (x / self.scaleU) + self.shiftU

    def unscaleOutput(self, x):
        assert (self.shiftY.ndim == 1 and
                len(self.shiftY) == x.shape[1]), \
                'incompatible dimensions'
        return (x / self.scaleY) + self.shiftY


    def haarmat(self, p):
        ''' builds a single orthogonal Haar wavelet block of size p x p'''

        if p == 1:
            return 1

        assert round(np.log2(p)) == np.log2(p), \
            'wavelet block size should be a power of 2'

        W = 1 / np.sqrt(2) * np.array([1, 1, 1, -1])
        dim = 2
        while dim < p:
            W = 1 / np.sqrt(2) * np.append(
                np.kron(W, np.array([1, 1])),
                np.kron(np.eye(dim), np.array([1, -1])))
            dim = W.shape[0]

        return sparse.csc_matrix(W)
