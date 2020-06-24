classdef ESN < handle
% ESN: an Echo State Network

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

        % Wconstruction parameter: average number of entries per row
        entriesPerRow (1,1) {mustBeInteger} = 10;

        % Wconstruction parameter: percentage of zeros
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

        % control feedthrough of u to y: (part of) the input state is appended
        % to the reservoir state activations X and used to fit W_out
        feedThrough (1,1) {mustBeNumericOrLogical} = false;

        % select a subset of the input data u to feedthrough
        ftRange (1,:) double {mustBeNumeric};

        % feedthrough amplitude
        ftAmp (1,1) double {mustBeNumeric} = 1;

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

        % set the inital state of X: 'random' or 'zero'
        reservoirStateInit (1,1) string = 'zero';

        % set the input weight matrix type: 'sparse' or 'full'
        inputMatrixType (1,1) string = 'sparse';

        % control size input weight matrix
        inAmplitude (1,1) double {mustBeNonnegative} = 1.0;

        % set the feedback weight matrix type: 'sparse' or 'full'
        feedbackMatrixType (1,1) string = 'sparse';

        % control output feedback
        ofbAmplitude (1,1) double {mustBeNonnegative} = 0.0;

        % set the method to solve the linear least squares problem to compute
        % W_out: 'Tikhonov' or 'pinv'
        regressionSolver (1,1) string = 'Tikhonov';

        % lambda (when using Tikhonov regularization)
        lambda (1,1) double {mustBeNonnegative} = 1.0e-4;

        % tolerance for the pseudo inverse
        pinvTol (1,1) double {mustBeNonnegative} = 1.0e-4;
    end

    methods
        %-------------------------------------------------------
        function self = ESN(Nr, Nu, Ny)
        % Constructor

            self.Nr = Nr;
            self.Nu = Nu;
            self.Ny = Ny;

            % default scaling (none)
            self.shiftU = 0;
            self.shiftY = 0;
            self.scaleU = 1;
            self.scaleY = 1;
        end

        %-------------------------------------------------------
        function setPars(self, pars)
        % overwrite class params with params in pars struct
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

            fprintf('ESN avg entries/row in W: %d\n', self.entriesPerRow);

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

            elseif self.Wconstruction == 'sparsity'
                self.W = rand(self.Nr)-0.5;
                self.W(rand(self.Nr) < self.sparsity) = 0;
                self.W = sparse(self.W);
            elseif self.Wconstruction == 'avgDegree'
                self.W = sprand(self.Nr, self.Nr, self.avgDegree / self.Nr);
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

            elseif self.inputMatrixType == 'full'
                % Create a random, full input weight matrix
                self.W_in = (rand(self.Nr, self.Nu) * 2 - 1);
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
            self.computeScaling(trainU, trainY)

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

            fprintf('ESN iterate state over %d samples... \n', T);
            time = tic;

            % iterate the state, save all neuron activations in X
            for k = 2:T
                X(k, :) = self.update(X(k-1, :), trainU(k, :), trainY(k-1, :));
            end

            self.X = X;
            fprintf('ESN iterate state over %d samples... done (%fs)\n', T, toc(time));

            time = tic;
            fprintf('ESN fitting W_out...\n')

            extX = self.X;
            if self.squaredStates == 'append';
                extX = [self.X, self.X.^2];
            elseif self.squaredStates == 'even';
                extX(:,2:2:end) = extX(:,2:2:end).^2;
            end

            if self.feedThrough
                if isempty(self.ftRange)
                    self.ftRange = 1:self.Nu;
                end
                extX = [self.ftAmp*trainU(:,self.ftRange), extX];
            end

            if self.centerX
                extX = extX - mean(extX);
            end

            if self.centerY
                trainY = trainY - mean(trainY);
            end

            if self.regressionSolver == 'pinv'

                fprintf('ESN  using pseudo inverse, tol = %1.1e\n', self.pinvTol)
                P = pinv(extX, self.pinvTol);
                self.W_out = (P*self.if_out(trainY))';

            elseif self.regressionSolver == 'Tikhonov'

                fprintf('ESN  using Tikhonov regularization, lambda = %1.1e\n', ...
                        self.lambda)
                fprintf('     problem size %d x %d\n', size(extX,2), size(extX,2));
                Xnormal    = extX'*extX + self.lambda * speye(size(extX,2));
                b          = extX'*self.if_out(trainY);
                self.W_out = (Xnormal \ b)';

            else
                ME = MException('ESN:invalidParameter', ...
                                'invalid regressionSolver parameter');
                throw(ME);
            end
            fprintf('ESN fitting W_out... done (%fs)\n', toc(time))

            % get training error
            predY = self.f_out(extX * self.W_out');

            fprintf('ESN training error: %e\n', sqrt(mean((predY(:) - trainY(:)).^2)));
        end

        %-------------------------------------------------------
        function [act] = update(self, state, u, y)
        % update the reservoir state
            if nargin < 4
                y = zeros(1, self.Ny);
            end

            pre = self.W*state' + self.W_in*u' + self.W_ofb*y' + self.bias;
            act = self.alpha * self.f(pre) + (1-self.alpha) * state' + ...
                  self.noiseAmplitude * (rand(self.Nr,1) - 0.5);
        end

        %-------------------------------------------------------
        function [out] = apply(self, state, u)

            x = state;

            if self.squaredStates == 'append'
                x = [state, state.^2];
            elseif self.squaredStates == 'even'
                x(2:2:end) = state(2:2:end).^2;
            end

            if self.feedThrough
                out = self.f_out(self.W_out * [self.ftAmp*u(self.ftRange)'; x'])';
            else
                out = self.f_out(self.W_out * x')';
            end
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
        end

        %-------------------------------------------------------
        function [out] = scaleInput(self, in)
            out = (in - self.shiftU) .* self.scaleU;
        end

        %-------------------------------------------------------
        function [out] = scaleOutput(self, in)
            out = (in - self.shiftY) .* self.scaleY;
        end

        %-------------------------------------------------------
        function [out] = unscaleInput(self, in)
            out = (in ./ self.scaleU ) + self.shiftU;
        end

        %-------------------------------------------------------
        function [out] = unscaleOutput(self, in)
            out = (in ./ self.scaleY ) + self.shiftY;
        end

    end
end