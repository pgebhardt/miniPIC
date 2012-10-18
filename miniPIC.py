from pylab import *
from numpy.random import normal
from scipy import constants
from scipy.linalg import toeplitz
from scipy.signal import convolve
from scipy.interpolate import interp1d

def main():
    # parameter
    length, dt, dx, electrodeArea = 0.05, 5e-11, 0.05 / 500, 2.0 * pi * sqrt(0.05)
    particleCount, particleWeight = 1e5, 2e9
    voltage = array([250.0, 0.0])

    # init grid
    grid, gridMids = linspace(0.0, length, length / dx + 1), linspace(0.5 * dx, length - 0.5 * dx, length / dx)

    # init particles
    electrons, ions = zeros((particleCount, 2)), zeros((particleCount, 2))
    electrons[:, 0], ions[:, 0] = normal(length / 2.0, length / 8.0, particleCount), normal(length / 2.0, length / 8.0, particleCount)
    electrons[:, 1], ions[:, 1] = normal(0.0, 1000.0, particleCount), normal(0.0, 100.0, particleCount)
    chargeWeight = constants.elementary_charge * particleWeight / (dx * electrodeArea)

    # init poisson solver
    phi, roh = zeros((length / dx + 1, )), zeros((length / dx - 1, ))
    poisson = inv(toeplitz([2.0, -1.0] + [0.0] * (phi.shape[0] - 4))) * dx ** 2 / constants.epsilon_0

    # pic cycle
    for i in range(100):
        # calc roh
        roh = chargeWeight * (histogram(ions[:, 0], grid)[0] - histogram(electrons[:, 0], grid)[0])
        for j in range(5):
            roh = convolve(roh, ones((20, )) / 20.0, 'same')

        # calc field
        phi[0], phi[1:-1], phi[-1] = voltage[0], dot(poisson, 0.5 * (roh[:-1] + roh[1:])) + grid[1:-1] * (voltage[1] - voltage[0]) / length + voltage[0], voltage[1]
        eField = interp1d(gridMids, -(phi[1:] - phi[:-1]) / dx, bounds_error=False)

        # update speeds
        electrons[:, 1] -= eField(electrons[:, 0]) * dt * constants.elementary_charge / constants.electron_mass
        ions[:, 1] += eField(ions[:, 0]) * dt * constants.elementary_charge / (constants.neutron_mass * 16)

        # move particles
        electrons[:, 0] += electrons[:, 1] * dt
        ions[:, 0] += ions[:, 1] * dt

        # print step
        if i % 100 == 0:
            print 'step: {}'.format(i)

    # plot electic fiel
    subplot(221)
    plot(eField(gridMids))
    title('Electical Field')

    # plot potential
    subplot(222)
    plot(phi)
    title('Potential')

    # plot charge density
    subplot(223)
    plot(roh)
    title('Charge density')

    # plot particle densit
    subplot(224)
    plot(histogram(ions[:, 0], grid)[0])
    plot(histogram(electrons[:, 0], grid)[0])
    title('Ion and electron density')

    show()

if __name__ == '__main__':
    main()
