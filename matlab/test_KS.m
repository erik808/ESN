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

NT = 5000; % timesteps
dt = 0.25; % timestep size
train_range=100:2100;
test_range=2101:2500;

% generate new data:
generate_data = false;

% use feedthrough connection
with_FT = true;

if generate_data
    fprintf('Generate time series... \n');
    % Perfect evolution
    X = zeros(N,NT);
    x = ks_prf.x_init;
    for i = 1:NT
        X(:,i) = x;
        [x, k] = ks_prf.step(x, dt);
    end
    % Imperfect predictions
    Phi = zeros(N, NT);
    for i = 1:NT
        x = X(:,i);
        [phi, k] = ks_imp.step(x, dt);
        Phi(:,i) = phi;
    end
    save('testdata_KS.mat', 'dt', 'NT', 'dt', ...
         'X', 'Phi', '-v7');
else
    load('testdata_KS.mat');
end

% input data
% restricted truths and imperfect predictions
if with_FT
    U = [X(:, 1:end-1); Phi(:,1:end-1)];
else
    U = [X(:, 1:end-1)];
end
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
esn_pars.feedThrough        = with_FT;
esn_pars.ftRange            = N+1:2*N;
esn_pars.fCutoff            = 0.1;

% seed
rng(1)

esn = ESN(esn_pars.Nr, size(trainU,2), size(trainY,2));
esn.setPars(esn_pars);
esn.initialize();
esn.train(trainU, trainY);

% Prediction
% initialization index
init_idx = train_range(end)+1;
% initial state for the predictions
yk = X(:, init_idx);

Npred = numel(test_range);
predY = zeros(Npred, N);
esn_state = esn.X(end,:);
for i = 1:Npred
    [Pyk, Nk] = ks_imp.step(yk, dt);
    if with_FT
        u_in      = [yk(:); Pyk(:)]';
    else
        u_in      = [yk(:)]';
    end

    u_in      = esn.scaleInput(u_in);
    esn_state = esn.update(esn_state, u_in)';
    u_out     = esn.apply(esn_state, u_in);
    yk        = esn.unscaleOutput(u_out)';
    predY(i,:) = yk;
end


figure(1)
imagesc(predY)
title('predY')

figure(2)
imagesc(testY)
title('testY')

figure(3)
imagesc(abs(predY-testY))
title('testY')
