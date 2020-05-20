classdef ESN < handle
% ESN: an Echo State Network

    properties
        Nr    (1,1) {mustBeInteger} = 500; % reservoir state dimension
        Nu    (1,1) {mustBeInteger} = 10;  % input dimension
        Ny    (1,1) {mustBeInteger} = 10;  % output dimension

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

        % control size input weight matrix
        inAmplitude (1,1) double {mustBeNonnegative} = 1.0;
        
        % control output feedback
        ofbAmplitude (1,1) double {mustBeNonnegative} = 0.0;

        scaleU (1,:) double {mustBeNumeric} % input scaling
        scaleY (1,:) double {mustBeNumeric} % output scaling

        X (:,:) double {mustBeNumeric} % reservoir state activations

        % control feedthrough of u to y: the input state is appended to the
        % reservoir state activations X and used to fit W_out
        feedThrough {mustBeNumericOrLogical} = true;

        % set the output activation: 'identity' or 'tanh'
        outputActivation (1,1) string = 'identity';

        % set the inital state of X: 'random' or 'zero'
        reservoirStateInit (1,1) string = 'random';
        
        % set the input weight matrix type: 'sparse' or 'full'
        inputMatrixType (1,1) string = 'sparse';

        % set the feedback weight matrix type: 'sparse' or 'full'
        feedbackMatrixType (1,1) string = 'sparse';        
    end

    methods
        function self = ESN(Nr, Nu, Ny)
        % constructor

            if nargin > 0
                self.Nr = Nr;
                self.Nu = Nu;
                self.Ny = Ny;
            end
        end

        function initialize(self)
        % create W, W_in, W_ofb

            self.createW;
            self.createW_in;
            self.createW_ofb;;
            
        end
        
        function createW(self)

            D = [];
            % create sparse weight matrix
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
            
           
        % assert(
            
            
        end
    end
end