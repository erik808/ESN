import numpy
import pytest

from ESN import ESN

def create_data(dt, init, dataPoints):
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

def create_esn_lorenz63():
    dt = 0.02
    Tend = 125
    T = int(Tend / dt)    
    U, Y = create_data(dt, [0.1, 0.1, 0.1], T)

    # setup training and testing data
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

    return esn, trainU, trainY
    
def test_scaling(esn, trainU, trainY):    
    
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
    
    # test minMax1 scaling
    scaleU = esn.scaleU
    scaleY = esn.scaleY

    if esn.Nu > 1:
        assert isinstance(scaleU, list) == True
        scalecol = 1.0 / ( numpy.max(trainU[:,0]) -
                           numpy.min(trainU[:,0]) )
        assert scaleU[0] == pytest.approx(scalecol)

    # todo
    # minmax2
    # minmaxall
    # standardize

if __name__ == '__main__':
    esn, trainU, trainY = create_esn_lorenz63()
    test_scaling(esn, trainU, trainY)

    
    
