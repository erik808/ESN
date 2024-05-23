% Minimal KS test.

% perturbation
epsilon = 0.1;

% Create and initialize two KS models. Perfect prf and imperfect imp.
L = 35;
N = 64;
ks_prf = KSmodel(L, N); % perfect model
ks_imp = KSmodel(L, N); % imperfect model
ks_imp.epsilon = epsilon; % set perturbation parameter
ks_prf.initialize();
ks_imp.initialize();

NT = 300; % timesteps
dt = 0.25; % timestep size
train_range=100:199;
test_range=200:299;

fprintf('Generate time series... \n');
X = zeros(N,NT);
x = ks_prf.x_init;
for i = 1:NT
    X(:,i) = x;
    [x, k] = ks_prf.step(x, dt);
end

Phi = zeros(N, NT);
for i = 1:NT
    x = X(:,i);
    [phi, k] = ks_imp.step(x, dt);
    Phi(:,i) = phi;
end

% input data
% restricted truths and imperfect predictions
U = [X(:, 1:end-1); Phi(:,1:end-1)];
Y = X(:, 2:end); % perfect predictions

trainU = U(:, train_range)';
trainY = Y(:, train_range)';
testU = U(:, test_range)';
testY = Y(:, test_range)';

%ESNc settings:
esn_pars = {};
esn_pars.scalingType        = 'standardize';
esn_pars.Nr                 = 100;
esn_pars.rhoMax             = 0.4;
esn_pars.alpha              = 1.0;
esn_pars.Wconstruction      = 'avgDegree';
esn_pars.avgDegree          = 3;
esn_pars.lambda             = 1e-10;
esn_pars.bias               = 0.0;
esn_pars.squaredStates      = 'even';
esn_pars.reservoirStateInit = 'random';
esn_pars.inputMatrixType    = 'balancedSparse';
esn_pars.inAmplitude        = 1.0;
esn_pars.waveletBlockSize   = 1.0;
esn_pars.waveletReduction   = 1.0;
esn_pars.dmdMode            = false;
esn_pars.feedThrough        = true;
esn_pars.ftRange            = N+1:2*N;
esn_pars.fCutoff            = 0.0;

esn = ESN(esn_pars.Nr, size(trainU,2), size(trainY,2));
esn.setPars(esn_pars);
esn.initialize;
esn.train(trainU, trainY);