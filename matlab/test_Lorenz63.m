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

esn.train(trainU, trainY);




function [oldStates, newStates] = createLorenz63(dt, init, dataPoints)

%  Create Lorenz '63 data.
%  dt:   timestep size
%  init: initial value (x,y,z)

    Nu = 3;

    % Lorenz '63 parameters
    rho   = 28;
    sigma = 10;
    beta  = 8/3;

    oldStates = zeros(dataPoints, Nu);
    newStates = zeros(dataPoints, Nu);

    oldStates(1, :) = init;

    x = init(1);
    y = init(2);
    z = init(3);        
    for n = 1:dataPoints
        oldStates(n,:) = [x,y,z];
        x = x + dt*sigma*(y - x);
        y = y + dt*(x*(rho - z) - y);
        z = z + dt*(x*y - beta*z);    
        
        newStates(n,:) = [x,y,z];
    end
end
