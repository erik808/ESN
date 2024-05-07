classdef ESN < handle
% ESN: an Echo State Network. The ESN can be tricked into doing pure
% DMD by using the feedthrough and disabling the ESN dynamics. Enable
% this with the dmdMode flag.

    properties
        Nr    (1,1) {mustBeInteger} % reservoir state dimension
        Nu    (1,1) {mustBeInteger} % input dimension
        Ny    (1,1) {mustBeInteger} % output dimension

        W     (:,:) double {mustBeNumeric}  % reservoir weight matrix
        W_in  (:,:) double {mustBeNumeric}  % input weight matrix
        W_ofb (:,:) double {mustBeNumeric}  % output weight feedback matrix
        W_out (:,:) double {mustBeNumeric}  % output weight matrix

        % spectral radius of the weight matrix W
        rhoMax (1,1) double {mustBeNonnegative} = 0.95;

        % method to construct W: 'entriesPerRow', 'sparsity', 'avgDegree'
        Wconstruction (1,1) string = 'entriesPerRow';

        % average degree of the graph W
        avgDegree (1,1) double = 3;

        % dmd mode: use the regression solver and input/output interface to
        % create a dmd model.
        dmdMode (1,1) {mustBeNumericOrLogical} = false;

        % Wconstruction, entriesPerRow option: avg number of entries per row
        entriesPerRow (1,1) {mustBeInteger} = 10;

        % Wconstruction, sparsity option: percentage of zeros
        sparsity (1,1) double {mustBeNonnegative} = 0.95;

        % control noise
        noiseAmplitude (1,1) double {mustBeNonnegative} = 0.0;

        % leaking rate
        alpha (1,1) double {mustBeNonnegative} = 1.0;

        % bias
        bias (1,:) double {mustBeNumeric} = 0.0; % bias vector

        shiftU (1,:) double {mustBeNumeric} % input shift
        shiftY (1,:) double {mustBeNumeric} % output shift

        scaleU (1,:) double {mustBeNumeric} % input scaling
        scaleY (1,:) double {mustBeNumeric} % output scaling

        % 'none'        : no scaling or user defined
        % 'minMax1'     : rescale features to [0,1]
        % 'minMax2'     : rescale features to [-1,1]
        % 'minMaxAll'   : rescale data to [-1,1] over all features
        % 'standardize' : rescale feature range to zero mean and unit variance
        scalingType (1,1) string = 'none';

        X (:,:) double {mustBeNumeric} % reservoir state activations

        % for fitting take the square of the even components, this fixes
        % issues with symmetry
        % 'disabled'
        % 'append'
        % 'even'
        squaredStates (1,1) string = 'disabled';

        % control time delay coordinates
        timeDelay (1,1) double {mustBeNonnegative} = 0

        % control time delay coordinates
        timeDelayShift (1,1) double {mustBeNonnegative} = 100

        % control feedthrough of u to y: (part of) the input state is appended
        % to the reservoir state activations X and used to fit W_out
        feedThrough (1,1) {mustBeNumericOrLogical} = false;

        % select a subset of the input data u to feedthrough
        ftRange (1,:) double {mustBeNumeric};

        % feedthrough amplitude
        ftAmp (1,1) double {mustBeNumeric} = 1;

        % reservoir amplitude
        resAmp (1,1) double {mustBeNumeric} = 1;

        % mean center reservoir states before fitting
        centerX (1,1) {mustBeNumericOrLogical} = false;

        % mean center training output data before fitting
        centerY (1,1) {mustBeNumericOrLogical} = false;

        % set the activation function f: 'tanh'
        activation (1,1) string = 'tanh';

        % activation function
        f = @(x) tanh(x);

        % set the output activation: 'identity' or 'tanh'
        outputActivation (1,1) string = 'identity';

        % output activation
        f_out  = @(y) y;

        % inverse output activation
        if_out = @(y) y;

        % set the inital state of X: 'random', 'zero'
        reservoirStateInit (1,1) string = 'zero';

        % set the input weight matrix type: 'sparse', 'sparseOnes',
        % 'balancedSparse', 'full', 'identity'.
        inputMatrixType (1,1) string = 'balancedSparse';

        % control size input weight matrix
        inAmplitude (1,1) double {mustBeNonnegative} = 1.0;

        % set the feedback weight matrix type: 'sparse' or 'full'
        feedbackMatrixType (1,1) string = 'sparse';

        % control output feedback
        ofbAmplitude (1,1) double {mustBeNonnegative} = 0.0;

        % set the method to solve the linear least squares problem to compute
        % W_out: 'TikhonovTSVD'
        %        'TikhonovNormalEquations
        %        'pinv'
        regressionSolver (1,1) string = 'TikhonovTSVD';

        % lambda (when using Tikhonov regularization)
        lambda (1,1) double {mustBeNonnegative} = 1.0e-4;

        % Filter cutoff: Tikhonov regularization dampens the singular values
        % which allows a cutoff at some point.
        fCutoff (1,1) double {mustBeNonnegative} = 1.0e-2;

        % tolerance for the pseudo inverse
        pinvTol (1,1) double {mustBeNonnegative} = 1.0e-4;

        % For use with TikhonovTSVD. If > 1 we perform a blockwise wavelet
        % projection with blocksize <waveletBlockSize> on the time
        % domain and reduce with a factor <waveletReduction>.
        waveletReduction (1,1) double = 1;
        waveletBlockSize (1,1) double = 1;

        % Array that stores the damping coefficients computed after a
        % TikhonovTSVD solve
        TikhonovDamping (1,:) double = 0;
    end

    methods
        %-------------------------------------------------------
        function self = ESN(Nr, Nu, Ny)
        % Constructor

            self.Nr = Nr;
            self.Nu = Nu;
            self.Ny = Ny;

            % default scaling (none)
            self.shiftU = zeros(1, Nu);
            self.shiftY = zeros(1, Ny);
            self.scaleU = ones(1, Nu);
            self.scaleY = ones(1, Ny);
        end

        %-------------------------------------------------------
        function setPars(self, pars)
        % overwrite class params with params in pars struct
            assert(isstruct(pars));
            names = fieldnames(pars);
            for k = 1:numel(names)
                self.(names{k}) = pars.(names{k});
            end
        end

        %-------------------------------------------------------
        function initialize(self)
        % Create W, W_in, W_ofb and set output activation function

            self.createW;
            self.createW_in;
            self.createW_ofb;

            if self.activation == 'tanh'
                self.f = @(x) tanh(x);
            end

            if self.outputActivation == 'tanh'
                self.f_out  = @(y) tanh(y);
                self.if_out = @(y) atanh(y);
            elseif self.outputActivation == 'identity'
                self.f_out = @(y) y;
                self.if_out = @(y) y;
            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid outputActivation parameter');
                throw(ME);
            end

            fprintf('ESN initialization done\n');
        end

        %-------------------------------------------------------
        function createW(self)
        % Create sparse weight matrix with spectral radius rhoMax


            if self.Wconstruction == 'entriesPerRow'
                D = [];
                for i = 1:self.entriesPerRow
                    % This method does not give enough variability in the
                    % nnz per row
                    D = [D; ...
                         [(1:self.Nr)', ceil(self.Nr*rand(self.Nr,1)), ...
                          (rand(self.Nr,1)-0.5)] ];
                end
                self.W = sparse(D(:,1), D(:,2), D(:,3), self.Nr, self.Nr);
                fprintf('ESN avg entries/row in W: %d\n', self.entriesPerRow);

            elseif self.Wconstruction == 'sparsity'
                self.W = rand(self.Nr)-0.5;
                self.W(rand(self.Nr) < self.sparsity) = 0;
                self.W = sparse(self.W);
                fprintf('ESN sparsity W: %f\n', self.sparsity);

            elseif self.Wconstruction == 'avgDegree'
                self.W = sprand(self.Nr, self.Nr, self.avgDegree / self.Nr);
                fprintf('ESN avg degree W: %d\n', self.avgDegree);
            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid Wconstruction parameter');
                throw(ME);
            end

            % try to converge on a few of the largest eigenvalues of W
            opts.maxit=1000;
            rho  = eigs(self.W, 3, 'lm', opts);
            mrho = max(abs(rho));

            if isnan(mrho)
                ME = MException('ESN:convergenceError', ...
                                'eigs did not converge on an eigenvalue');
                throw(ME);
            end
            fprintf('ESN largest eigenvalue of W: %f\n', mrho);

            % adjust spectral radius of W
            self.W = self.W * self.rhoMax / mrho;
        end

        %-------------------------------------------------------
        function createW_in(self)
        % Create input weight matrix, sparse or full based on inputMatrixType.

            if self.inputMatrixType == 'sparse'
                % Create sparse input weight matrix. This gives a single random
                % connection per row, in a random column.
                D = [(1:self.Nr)', ceil(self.Nu * rand(self.Nr, 1)), (rand(self.Nr, 1) * 2 - 1)];
                self.W_in = sparse(D(:,1), D(:,2), D(:,3), self.Nr, self.Nu);

            elseif self.inputMatrixType == 'sparseOnes'
                %
                D = [(1:self.Nr)', ceil(self.Nu * rand(self.Nr, 1)), ones(self.Nr, 1)];
                self.W_in = sparse(D(:,1), D(:,2), D(:,3), self.Nr, self.Nu);

            elseif self.inputMatrixType == 'balancedSparse'
                % This implementation makes sure that every input element is connected
                % to roughly the same number of reservoir components.
                % Weights are uniform in [-1,1].

                arM  = floor(self.Nr/self.Nu);
                arP  = ceil(self.Nr/self.Nu);
                l1   = self.Nr - arM*self.Nu;
                l2   = self.Nu - l1;

                ico  = 1:self.Nr;

                jco1 = (1:l1) .* ones(arP, l1);
                jco2 = (l1+1:l1+l2) .* ones(arM, l2);
                jco  = [jco1(:) ; jco2(:)];

                co = 2*rand(self.Nr,1) - 1;

                self.W_in = sparse(ico,jco,co,self.Nr,self.Nu);

            elseif self.inputMatrixType == 'full'
                % Create a random, full input weight matrix
                self.W_in = (rand(self.Nr, self.Nu) * 2 - 1);

            elseif self.inputMatrixType == 'identity'
                %
                self.W_in = speye(self.Nr, self.Nu);

            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid inputMatrixType parameter');
                throw(ME);
            end
            self.W_in = self.inAmplitude * self.W_in;
        end

        %-------------------------------------------------------
        function createW_ofb(self)
        % Create output feedback weight matrix

            if self.feedbackMatrixType == 'sparse'
                % Create sparse output feedback weight matrix. This gives a single
                % random connection per row, in a random column.
                D = [(1:self.Nr)', ceil(self.Ny * rand(self.Nr, 1)), (rand(self.Nr, 1) * 2 - 1)];
                self.W_ofb = sparse(D(:,1), D(:,2), D(:,3), self.Nr, self.Ny);

            elseif self.feedbackMatrixType == 'full'
                % Create a random, flul output feedback weight matrix
                self.W_ofb = (rand(self.Nr, self.Ny) * 2 - 1);
            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid feedbackMatrixType parameter');
                throw(ME);
            end
            self.W_ofb = self.ofbAmplitude * self.W_ofb;
        end

        %-------------------------------------------------------
        function train(self, trainU, trainY)
        % Train the ESN with training data trainU and trainY
            assert(size(trainU,2) == self.Nu, 'ESN:dimensionError', ...
                   'input training data has incorrect dimension Nu');

            assert(size(trainY,2) == self.Ny, 'ESN:dimensionError', ...
                   'output training data has incorrect dimension Ny');

            T = size(trainU,1);

            assert(T == size(trainY,1), 'ESN:dimensionError', ...
                   'input and output training data have different number of samples');

            % Now that we have data we can setup the scaling
            self.computeScaling(trainU, trainY);

            fprintf('ESN training, input Nu = %d, output Ny = %d\n', self.Nu, self.Ny);

            % Apply scaling
            trainU = self.scaleInput(trainU);
            trainY = self.scaleOutput(trainY);

            % initialize activations X
            if self.reservoirStateInit == 'random'
                Xinit = self.f(10*randn(1, self.Nr));
            elseif self.reservoirStateInit == 'zero'
                Xinit = zeros(1, self.Nr);
            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid reservoirStateInit parameter');
                throw(ME);
            end
            X = [Xinit; zeros(T-1, self.Nr)];

            time = tic;

            if ~self.dmdMode
                % Normal ESN behaviour: iterate the state, save all neuron activations
                % in X
                fprintf('ESN iterate state over %d samples... \n', T);
                for k = 2:T
                    X(k, :) = self.update(X(k-1, :), trainU(k, :), trainY(k-1, :));
                end
                fprintf('ESN iterate state over %d samples... done (%fs)\n', T, toc(time));
            else % DMD mode: disable ESN dynamics
                X = [];
            end

            self.X = X;
            self.computeW_out(X, trainU, trainY);

        end

        %-------------------------------------------------------
        function [] = computeW_out(self, X, trainU, trainY)
        % extend X
            extX = X;
            if self.squaredStates == 'append' &&  ~isempty(extX);
                extX = [self.X, self.X.^2];
            elseif self.squaredStates == 'even' &&  ~isempty(extX);
                extX(:, 2:2:end) = extX(:, 2:2:end).^2;
            end

            if self.feedThrough
                if isempty(self.ftRange)
                    self.ftRange = 1:self.Nu;
                end
                extX = [self.ftAmp*trainU(:,self.ftRange), self.resAmp*extX];
            end
            assert(~isempty(extX), 'enable the feedThrough in pure DMD mode');

            if self.timeDelay > 0
                [extXrows, extXcols] = size(extX);
                extX = self.applyTimeDelay(extX, self.timeDelay, self.timeDelayShift);
                trainY = self.applyTimeDelay(trainY, self.timeDelay, self.timeDelayShift);
            end

            if self.centerX
                extX = extX - mean(extX);
            end

            if self.centerY
                trainY = trainY - mean(trainY);
            end

            time = tic;
            fprintf('ESN fitting W_out...\n')

            if self.regressionSolver == 'pinv'

                fprintf('ESN  using pseudo inverse, tol = %1.1e\n', self.pinvTol)
                P = pinv(extX, self.pinvTol);
                self.W_out = (P*self.if_out(trainY))';

            elseif self.regressionSolver == 'TikhonovNormalEquations'

                fprintf('ESN  using Tikhonov regularization, lambda = %1.1e\n', ...
                        self.lambda)
                fprintf(' solving normal equations: %d x %d\n', size(extX,2), size(extX,2));
                Xnormal    = extX'*extX + self.lambda * speye(size(extX,2));
                b          = extX'*self.if_out(trainY);
                self.W_out = (Xnormal \ b)';

            elseif self.regressionSolver == 'TikhonovTSVD'

                fprintf('ESN using TSVD and Tikhonov regularization, lambda = %1.1e\n', ...
                        self.lambda)
                fprintf(' computing SVD\n');

                T = size(extX, 1);
                H = speye(T,T);

                if (self.waveletReduction > 1) && (self.waveletBlockSize > 1)
                    % create wavelet block
                    W = self.haarmat(self.waveletBlockSize);

                    % reduce wavelet block
                    block_red = round(self.waveletBlockSize / self.waveletReduction);
                    W = W(1:block_red,:);

                    % repeating the block
                    Nblocks = floor(T / self.waveletBlockSize);
                    H = kron(speye(Nblocks), W);

                    % padding the wavelet block
                    Th = Nblocks * self.waveletBlockSize;
                    rem = T - Th;
                    Tr = size(H,1);
                    H = [sparse(zeros(Tr, rem)), H]';
                end

                fprintf(' problem size: %d x %d\n', size(H,2), size(extX,2));
                [U,S,V] = svd(H'*extX, 'econ');

                s = diag(S);
                f = s.^2 ./ (s.^2 + self.lambda);
                self.TikhonovDamping = f;

                % filter cutoff
                fcutoff_ind = find(f > self.fCutoff, 1, 'last');
                V = V(:,1:fcutoff_ind);
                U = U(:,1:fcutoff_ind);
                S = sparse(S(1:fcutoff_ind,1:fcutoff_ind));
                s = diag(full(S));
                f = s.^2 ./ (s.^2 + self.lambda);
                fprintf('  filter cutoff %1.3e at index %d\n', self.fCutoff, fcutoff_ind);
                fprintf('  smallest filter coefficient: %1.3e\n', f(end));

                invReg  = sparse(diag(1./ (s.^2 + self.lambda)));
                self.W_out = (V*(invReg*(S*(U'*(H'*trainY)))))';
            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid regressionSolver parameter');
                throw(ME);
            end
            fprintf('ESN fitting W_out... done (%fs)\n', toc(time))

            % get training error
            predY = self.f_out(extX * self.W_out');

            fprintf('ESN fitting error: %e\n', sqrt(mean((predY(:) - trainY(:)).^2)));

            if self.timeDelay > 0
                % select final diagonal block (current state) W_out
                [m,n] = size(self.W_out);
                self.W_out = self.W_out(m-self.Ny+1:m, n-extXcols+1:n);
            end
        end

        %-------------------------------------------------------
        function [act] = update(self, state, u, y)
        % update the reservoir state
            if nargin < 4
                y = zeros(1, self.Ny);
            end

            if self.dmdMode
                act = u';
            else
                pre = self.W*state' + self.W_in*u' + self.W_ofb*y' + self.bias;
                act = self.alpha * self.f(pre) + (1-self.alpha) * state' + ...
                      self.noiseAmplitude * (rand(self.Nr,1) - 0.5);
            end
        end

        %-------------------------------------------------------
        function [out] = apply(self, state, u)

            if self.dmdMode
                x = [];
            else
                x = state;
            end

            if self.squaredStates == 'append' && ~self.dmdMode
                x = [state, state.^2];
            elseif self.squaredStates == 'even' && ~self.dmdMode
                x(2:2:end) = state(2:2:end).^2;
            end

            if self.feedThrough
                x = [self.ftAmp*u(self.ftRange)'; self.resAmp*x'];
            else
                x = x';
            end

            out = self.f_out(self.W_out * x)';
        end

        %-------------------------------------------------------
        function [] = computeScaling(self, U, Y)

            if self.scalingType == 'none'
                fprintf('ESN scaling: none or user specified\n');

            elseif self.scalingType == 'minMax1'
                self.scaleU = 1.0 ./ (max(U) - min(U));
                self.scaleY = 1.0 ./ (max(Y) - min(Y));
                self.shiftU = min(U);
                self.shiftY = min(Y);
                fprintf('ESN scaling: minMax1 [0,1]\n');

            elseif self.scalingType == 'minMax2'
                self.scaleU = 2.0 ./ (max(U) - min(U));
                self.scaleY = 2.0 ./ (max(Y) - min(Y));
                self.shiftU = min(U) + 1 ./ self.scaleU;
                self.shiftY = min(Y) + 1 ./ self.scaleY;
                fprintf('ESN scaling: minMax2 [-1,1]\n');

            elseif self.scalingType == 'minMaxAll'
                self.scaleU = 2.0 ./ (max(U(:)) - min(U(:)));
                self.scaleY = 2.0 ./ (max(Y(:)) - min(Y(:)));
                self.shiftU = min(U(:)) + 1 ./ self.scaleU;
                self.shiftY = min(Y(:)) + 1 ./ self.scaleY;
                self.scaleU = repmat(self.scaleU, 1, size(U,2));
                self.scaleY = repmat(self.scaleY, 1, size(Y,2));
                self.shiftU = repmat(self.shiftU, 1, size(U,2));
                self.shiftY = repmat(self.shiftY, 1, size(Y,2));
                fprintf('ESN scaling: minMaxAll [-1,1]\n');

            elseif self.scalingType == 'standardize'
                self.scaleU = 1.0 ./ std(U);
                self.scaleY = 1.0 ./ std(Y);
                self.shiftU = mean(U);
                self.shiftY = mean(Y);
                fprintf('ESN scaling: standardize\n');

            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid scalingType parameter');
                throw(ME);
            end

            % detect constant datapoints,
            idinfU = find(isinf(self.scaleU));
            idinfY = find(isinf(self.scaleY));
            if ~isempty(idinfU) || ~isempty(idinfY)
                fprintf(' constant data points found...\n  these remain unscaled\n')
            end
            self.scaleU(idinfU) = 1.0;
            self.scaleY(idinfY) = 1.0;
            self.shiftU(idinfU) = 0.0;
            self.shiftY(idinfY) = 0.0;

        end

        %-------------------------------------------------------
        function [out] = scaleInput(self, in)
            assert(size(in,2) == size(self.shiftU,2), ...
                   'incompatible dimensions');
            out = (in - self.shiftU) .* self.scaleU;
        end

        %-------------------------------------------------------
        function [out] = scaleOutput(self, in)
            assert(size(in,2) == size(self.shiftY,2), ...
                   'incompatible dimensions');
            out = (in - self.shiftY) .* self.scaleY;
        end

        %-------------------------------------------------------
        function [out] = unscaleInput(self, in)
            assert(size(in,2) == size(self.shiftU,2), ...
                   'incompatible dimensions');
            out = (in ./ self.scaleU ) + self.shiftU;
        end

        %-------------------------------------------------------
        function [out] = unscaleOutput(self, in)
            assert(size(in,2) == size(self.shiftY,2), ...
                   'incompatible dimensions');
            out = (in ./ self.scaleY ) + self.shiftY;
        end

        function [W] = haarmat(self, p)
        % builds a single orthogonal Haar wavelet block of size p x p

            if p == 1
                W = 1;
                return
            end

            assert( round(log2(p)) == log2(p) , ...
                    'wavelet block size should be a power of 2');

            W   = 1/sqrt(2)*[1 1; 1 -1];
            dim = 2;
            while dim < p
                W = 1/sqrt(2)*[kron(W,[1 1]); kron(eye(dim),[1 -1])];
                dim = size(W,1);
            end
            W = sparse(W);
        end

        function [Y] = applyTimeDelay(self, X, s, shift)
        % The states in X are extended with delayed states
            if nargin < 4
                shift = 1;
            end

            if s <= 0
                Y = X;
                return
            end

            [T,n] = size(X); % snapshots times state dimension
            m = T-shift*s; % reduce the number of snapshot
            Y = zeros(m, n*(s+1)); % contains s delays and itself
            % set original and append s delays
            for i = 1:s+1
                Y(:,(i-1)*n+1:i*n) =  X(1+(i-1)*shift:m+(i-1)*shift,:);
            end
        end
    end
end
