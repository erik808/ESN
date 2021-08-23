import numpy

import matplotlib.pyplot as plt

from ESN import ESN


def createLorenz63(dt, init, dataPoints):
    '''Create Lorenz '63 data.
    dt:   timestep size
    init: initial value (x,y,z)'''

    Nu = 3

    # Lorenz '63 parameters
    rho = 28
    sigma = 10
    beta = 8/3

    oldStates = numpy.zeros((dataPoints, Nu))
    newStates = numpy.zeros((dataPoints, Nu))

    oldStates[0, :] = init

    x = init[0]
    y = init[1]
    z = init[2]

    for n in range(dataPoints):
        oldStates[n, :] = [x, y, z]
        x = x + dt * sigma * (y - x)
        y = y + dt * (x * (rho - z) - y)
        z = z + dt * (x*y - beta*z)

        newStates[n, :] = [x, y, z]

    return oldStates, newStates


def main():
    # Create Lorenz63 data
    dt = 0.02
    Tend = 125
    T = int(Tend / dt)
    U, Y = createLorenz63(0.02, [0.1, 0.1, 0.1], T)

    # setup data
    cutoff = int(numpy.ceil(0.8 * T))
    trainU = U[:cutoff, :]
    trainY = Y[:cutoff, :]
    testU = U[cutoff:, :]
    testY = Y[cutoff:, :]

    # input/output dimensions
    Nu = U.shape[1]
    Ny = Y.shape[1]

    # reservoir size
    Nr = 300

    esn = ESN(Nr, Nu, Ny)

    # change a few parameters
    esn.rhoMax = 1.2
    esn.Wconstruction = 'entriesPerRow'
    esn.entriesPerRow = 8
    esn.inputMatrixType = 'full'
    esn.inAmplitude = 1.0
    esn.feedbackMatrixType = 'full'
    esn.ofbAmplitude = 0.0
    esn.scalingType = 'minMax1'
    esn.feedThrough = False
    esn.pinvTol = 1e-3
    esn.alpha = 1.0
    esn.regressionSolver = 'TikhonovTSVD'

    esn.initialize()

    # training
    esn.train(trainU, trainY)

    # prediction
    nPred = U.shape[0]

    # reservoir state initialization for prediction
    state = esn.X[-1, :]

    # initialize output array
    predY = numpy.zeros((nPred, trainY.shape[1]))

    # predict, feed results back into network
    # begin with final training output as input
    u = (trainY[-1, :] - esn.shiftU) * esn.scaleU
    predY[0, :] = u
    for k in range(1, nPred):
        state = esn.update(state, u, u).T
        if esn.feedThrough:
            u = esn.f_out(esn.W_out @ numpy.append(state, u).T).T
        else:
            u = esn.f_out(esn.W_out @ state.T).T

        predY[k, :] = u

    # unscale
    predY = (predY / esn.scaleY) + esn.shiftY

    # construct time array for plotting
    predT = dt * (numpy.arange(predY.shape[0]))
    testT = dt * (numpy.arange(testY.shape[0]))

    # # plot reservoir
    # figure(1)
    # imagesc(esn.X')
    # colormap(gray)
    # colorbar

    # plot actual and predicted time series
    fig, ax = plt.subplots()
    fig.add_subplot(3, 1, 1)
    plt.plot(predT, predY[:, 0], 'r')
    plt.plot(testT, testY[:, 0], 'k')
    plt.xlim([0, 3])
    fig.legend(['predicted', 'actual'])

    fig.add_subplot(3, 1, 2)
    plt.plot(predT, predY[:, 1], 'r')
    plt.plot(testT, testY[:, 1], 'k')
    plt.xlim([0, 3])

    fig.add_subplot(3, 1, 3)
    plt.plot(predT, predY[:, 2], 'r')
    plt.plot(testT, testY[:, 2], 'k')
    plt.xlim([0, 3])

    plt.show()

    # figure(3)
    # plot3(predY(:,1),predY(:,2),predY(:,3), 'r') hold on
    # plot3(testY(:,1),testY(:,2),testY(:,3), 'k') hold off
    # campos([-240,240,240])


if __name__ == '__main__':
    main()
