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

NT = 5000 % timesteps
dt = 0.25 % timestep size

fprintf('Generate time series... \n');
X = zeros(NT, N);
x = ks_prf.x_init;
for i = 1:NT
    X(i,:) = x;
    [x, k] = ks_prf.step(x, dt);
end

imagesc(X')