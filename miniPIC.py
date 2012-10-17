from pylab import *
from scipy import constants
from scipy.linalg import toeplitz
from scipy.interpolate import interp1d

def main():
    # parameter
    length, particleCount, dt, dx = 0.1, 1e6, 1e-9, 1e-4
    grid, voltage = linspace(0.0, length, length / dx + 1), zeros((2, ))

    # init particles
    electrones, ions = zeros((particleCount, 2)), zeros((particleCount, 2))
    electrones[:, 0], ions[:, 0] = ogrid[0.0:length:particleCount * 1j], ogrid[0.0:length:particleCount * 1j]

    # init poisson solver
    phi, roh = zeros((length / dx + 1, )), histogram(ions[:, 0], int(length / dx))[0] - histogram(electrones[:, 0], int(length / dx))[0]
    poisson = inv(toeplitz([2.0, -1.0] + [0.0] * (phi.shape[0] - 4)) / dx ** 2) * constants.epsilon_0

    # pic cycle
    for i in range(10):
        # calc field
        phi[1:-1] = dot(poisson, 0.5 * (roh[:-1] + roh[1:])) + grid[1:-1] * (voltage[1] - voltage[0]) / length + voltage[0]
        eField = interp1d(0.5 * (grid[:-1] + grid[:-1]), (phi[1:] - phi[:-1]) / dx)

    plot(phi)
    show()

if __name__ == '__main__':
    main()
