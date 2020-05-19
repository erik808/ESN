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

        % control output feedback
        ofbAmplitude (1,1) double {mustBeNonnegative} = 0.0;

        scaleU (1,:) double {mustBeNumeric} % input scaling
        scaleY (1,:) double {mustBeNumeric} % output scaling

        X (:,:) double {mustBeNumeric} % reservoir state activations

        % control feedthrough of u to y: the input state is appended to the
        % reservoir state activations X and used to fit W_out
        extendX {mustBeNumericOrLogical} = true;

        % set the output activation: options TODO
        outputActivation (1,1) string = 'identity';

        % set the inital state of X: options TODO
        reservoirStateInit (1,1) string = 'randn';
    end

    methods
        function obj = ESN(Nr, Nu, Ny)
        % constructor

            if nargin > 0
                obj.Nr = Nr;
                obj.Nu = Nu;
                obj.Ny = Ny;
            end
        end

        function initialize(obj)
        % create W, W_in, W_ofb

            D = [];
            % create sparse weight matrix
            fprintf('ESN avg entries/row: %d\n', obj.entriesPerRow);
            for i = 1:obj.entriesPerRow
                D = [D; ...
                     [(1:obj.Nr)', ceil(double(obj.Nr)*rand(obj.Nr,1)), ...
                      (rand(obj.Nr,1)-0.5)] ];
            end
            obj.W = sparse(D(:,1), D(:,2), D(:,3), obj.Nr, obj.Nr);

            % try to converge on a few of the largest eigenvalues of W
            opts.maxit=500;
            rho  = eigs(obj.W, 3, 'lm', opts);
            mrho = max(abs(rho));

            if isnan(mrho)
                ME = MException('ESN:convergenceError', ...
                                'eigs did not converge on an eigenvalue');
                throw(ME);
            end

            fprintf('ESN largest eig: %f\n', mrho);
            % adjust spectral radius of W
            obj.W = obj.W * obj.rhoMax / mrho;
        end
    end
end