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

        % control sparsity in W, average number of entries per row
        entriesPerRow (1,1) {mustBeInteger} = 10;

        % control noise
        noiseAmplitude (1,1) double {mustBeNonnegative} = 0.0;

        % leaking rate
        alpha (1,1) double {mustBeNonnegative} = 1.0;

        % control size input weight matrix
        inAmplitude (1,1) double {mustBeNonnegative} = 1.0;

        % control output feedback
        ofbAmplitude (1,1) double {mustBeNonnegative} = 0.0;

        scaleU (1,:) double {mustBeNumeric} % input scaling
        scaleY (1,:) double {mustBeNumeric} % output scaling

        defaultScaling (1,1) {mustBeNumericOrLogical} = true;

        X (:,:) double {mustBeNumeric} % reservoir state activations

        % control feedthrough of u to y: the input state is appended to the
        % reservoir state activations X and used to fit W_out
        feedThrough (1,1) {mustBeNumericOrLogical} = true;

        % set the output activation: 'identity' or 'tanh'
        outputActivation (1,1) string = 'identity';

        % output activation
        f_out     = @(y) y;

        % inverse output activation
        if_out = @(y) y;

        % set the inital state of X: 'random' or 'zero'
        reservoirStateInit (1,1) string = 'random';

        % set the input weight matrix type: 'sparse' or 'full'
        inputMatrixType (1,1) string = 'sparse';

        % set the feedback weight matrix type: 'sparse' or 'full'
        feedbackMatrixType (1,1) string = 'sparse';

        % set the method to solve the linear least squares problem to compute
        % W_out: 'Tikhonov' or 'pinv'
        regressionSolver (1,1) string = 'Tikhonov';

        % lambda (when using Tikhonov regularization)
        lambda (1,1) double {mustBeNonnegative} = 1.0;

    end

    methods
        function self = ESN(Nr, Nu, Ny)
        % Constructor

            self.Nr = Nr;
            self.Nu = Nu;
            self.Ny = Ny;

        end

        function initialize(self)
        % Create W, W_in, W_ofb and set output activation function

            self.createW;
            self.createW_in;
            self.createW_ofb;

            if self.outputActivation == 'tanh'
                self.f_out     = @(y) tanh(y);
                self.if_out = @(y) atanh(y);
            elseif self.outputActivation == 'identity'
                self.f_out = @(y) y;
                self.if_out = @(y) y;
            end
            
        end

        function createW(self)
        % Create sparse weight matrix with spectral radius rhoMax

            D = [];
            fprintf('ESN avg entries/row: %d\n', self.entriesPerRow);
            for i = 1:self.entriesPerRow
                D = [D; ...
                     [(1:self.Nr)', ceil(self.Nr*rand(self.Nr,1)), ...
                      (rand(self.Nr,1)-0.5)] ];
            end
            self.W = sparse(D(:,1), D(:,2), D(:,3), self.Nr, self.Nr);

            % try to converge on a few of the largest eigenvalues of W
            opts.maxit=500;
            rho  = eigs(self.W, 3, 'lm', opts);
            mrho = max(abs(rho));

            if isnan(mrho)
                ME = MException('ESN:convergenceError', ...
                                'eigs did not converge on an eigenvalue');
                throw(ME);
            end
            fprintf('ESN largest eig: %f\n', mrho);

            % adjust spectral radius of W
            self.W = self.W * self.rhoMax / mrho;
        end

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
            end
            self.W_in = self.inAmplitude * self.W_in;
        end

        function createW_ofb(self)
        % Create output feedback weight matrix

            if self.feedbackMatrixType == 'sparse'
                % Create sparse output feedback weight matrix. This gives a single
                % random connection per row, in a random column.
                D = [(1:self.Nr)', ceil(self.Ny * rand(self.Nr, 1)), (rand(self.Nr, 1) * 2 - 1)];
                self.W_ofb = sparse(D(:,1), D(:,2), D(:,3), self.Nr, self.Ny);

            elseif self.feedbackMatrixType == 'full'
                % Create a random, flul output feedback weight matrix
                self.W_in = (rand(self.Nr, self.Nu) * 2 - 1);
            end
            self.W_ofb = self.ofbAmplitude * self.W_ofb;

        end

        function train(self, trainU, trainY)
        % Train the ESN with training data trainU and trainY

            assert(size(trainU,2) == self.Nu, 'ESN:dimensionError', ...
                   'input training data has incorrect dimension Nu');

            assert(size(trainY,2) == self.Ny, 'ESN:dimensionError', ...
                   'output training data has incorrect dimension Ny');

            T = size(trainU,1);

            assert(T == size(trainY,1), 'ESN:dimensionError', ...
                   'input and output training data have different number of samples');

            fprintf('ESN training samples: %d\n', T);

            % This is the default scaling. Any problem-specific scaling should be
            % done by the user.
            if self.defaultScaling
                self.scaleU = 1.2 * max(abs(trainU));
                self.scaleY = 1.2 * max(abs(trainY));
            end

            % scale training data
            trainU = trainU ./ self.scaleU;
            trainY = trainY ./ self.scaleY;

            % initialize activations X
            if self.reservoirStateInit == 'random'
                Xinit = tanh(10*randn(1, self.Nr));
            elseif self.reservoirStateInit == 'zero'
                Xinit = zeros(1, self.Nr);
            end
            self.X = [Xinit; zeros(T-1, self.Nr)];

            % iterate the state, save all neuron activations in X
            for k = 2:T
                self.X(k, :) = self.update(self.X(k-1, :), trainU(k, :), trainY(k-1, :));
            end

            time = tic;
            fprintf('ESN fitting W_out...\n')

            if self.feedThrough
                extX = [self.X, trainU];
            else
                extX = self.X;
            end

            if self.regressionSolver == 'pinv'
                P     = pinv(extX);
                W_out = (P*self.if_out(trainY))';
            elseif self.regressionSolver == 'Tikhonov'
                Xnormal = extX'*extX + self.lambda * speye(size(extX,2));
                b       = extX'*self.if_out(trainY);
                W_out   = (Xnormal \ b)';
            end
            fprintf('ESN fitting W_out... done (%fs)\n', toc(time))
            
            % get training error
            predY = self.f_out(extX * W_out');
            
            fprintf('ESN training error: %e\n', sqrt(mean((predY(:) - trainY(:)).^2)));
        end

        function [act] = update(self, state, u, y)
            pre = self.W*state' + self.W_in*u' + self.W_ofb*y';
            act = self.alpha * tanh(pre) + (1-self.alpha) * state' + ...
                  self.noiseAmplitude * (rand(self.Nr,1) - 0.5);
        end

    end
end