import numpy
import pytest

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

def test_lorenz63():
    dt = 0.02
    Tend = 125
    T = int(Tend / dt)    
    U, Y = create_data(dt, [0.1, 0.1, 0.1], T)
    



if __name__ == '__main__':
    test_lorenz63()
