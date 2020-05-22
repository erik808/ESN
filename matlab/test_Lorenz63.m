% Create Lorenz63 data
dt     = 0.02;
Tend   = 125;
T      = floor(Tend / dt);
[U, Y] = createLorenz63(0.02, [0.1,0.1,0.1], T);

% setup data
cutoff  = ceil(0.8*T);
trainU  = U(1:cutoff,:);
trainY  = Y(1:cutoff,:);
testU   = U(cutoff+1:end,:);
testY   = Y(cutoff+1:end,:);

% input/output dimensions
Nu = size(U,2);
Ny = size(Y,2);

% reservoir size
Nr = 300;

esn = ESN(Nr, Nu, Ny);

% change a few parameters
esn.rhoMax = 1.2;
esn.Wconstruction = 'entriesPerRow';
esn.entriesPerRow = 8;
esn.inputMatrixType = 'full';
esn.feedThrough = false;

esn.initialize;

% training
esn.train(trainU, trainY);

assert(~esn.feedThrough)
assert(size(esn.W_out,2) == esn.Nr);

% prediction
nPred = size(testU,1);

% reservoir state initialization for prediction
state = esn.X(end,:);

% initialize output array
predY = zeros(nPred, size(trainY,2));

% predict, feed results back into network
% begin with final training output as input
u = trainY(end,:) .* esn.scaleU;
predY(1,:) = u;
for k = 2:nPred
    state = esn.update(state, u, u)';
    u = esn.f_out(esn.W_out * state')';
    predY(k,:) = u;
end

% unscale
predY = predY ./ esn.scaleY;

% construct time array for plotting
time = dt*(1:size(predY,1));

% plot reservoir
figure(1)
imagesc(esn.X')
colormap(gray)
colorbar

% plot actual and predicted time series
figure(2);
plot(time, predY(:,1), 'r'); hold on;
plot(time, testY(:,1), 'k'); hold off;
xlim([0,10])
legend('predicted', 'actual')

figure(3);
plot3(predY(:,1),predY(:,2),predY(:,3), 'r'); hold on;
plot3(testY(:,1),testY(:,2),testY(:,3), 'k'); hold off;
campos([-240,240,240]);