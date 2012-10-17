from pylab import *
from scipy import constants
from scipy.linalg import toeplitz

def main():
    # parameter
    length = 0.1

    # delta values
    dt, dx = 1e-9, 1e-4

    # init arrays
    phi, eField = zeros((length / dx + 1, )), zeros((length / dx + 1, ))

    # init poisson solver
    poissonMatrix = inv(toeplitz([2.0, -1.0] + [0.0] * (phi.shape[0] - 4)) / dx ** 2) * constants.epsilon_0

    # solve
    phi[1:-1] = dot(poissonMatrix, ones(phi.shape)[1:-1])

    plot(phi)
    show()

if __name__ == '__main__':
    main()
