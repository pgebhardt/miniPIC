from pylab import *
from scipy import constants
from scipy.linalg import toeplitz
from scipy.interpolate import interp1d
import time

class Timer():
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print 1e3 * (time.time() - self.start)

def main():
    # parameter
    length, particleCount, superparticle, dt, dx = 0.1, 1e6, 1e3, 1e-9, 1e-4
    grid, gridMids, voltage = linspace(0.0, length, length / dx + 1), linspace(0.5 * dx, length - 0.5 * dx, length / dx), array([1.0, 0.0])
    mass = constants.electron_mass

    # init particles
    electrons, ions = zeros((particleCount, 2)), zeros((particleCount, 2))
    electrons[:, 0], ions[:, 0] = ogrid[0.0:length:particleCount * 1j], ogrid[0.0:length:particleCount * 1j]

    # init poisson solver
    phi, roh = zeros((length / dx + 1, )), zeros((length / dx - 1, ))
    poisson = inv(toeplitz([2.0, -1.0] + [0.0] * (phi.shape[0] - 4)) / dx ** 2) * constants.epsilon_0

    # pic cycle
    for i in range(1):
        # calc roh
        print 'calc roh:'
        with Timer():
            roh = superparticle * histogram(ions[:, 0], grid)[0] - histogram(electrons[:, 0], grid)[0]

        # calc field
        print 'calc field:'
        with Timer():
            phi[0], phi[1:-1], phi[-1] = voltage[0], dot(poisson, 0.5 * (roh[:-1] + roh[1:])) + grid[1:-1] * (voltage[1] - voltage[0]) / length + voltage[0], voltage[1]
            eField = interp1d(gridMids, (phi[1:] - phi[:-1]) / dx, bounds_error=False, fill_value=0.0)

        # move particles
        print 'move particles:'
        with Timer():
            electrons[:, 1] -= eField(electrons[:, 0]) * dt * (constants.elementary_charge / mass)
            electrons[:, 0] += electrons[:, 1] * dt
            ions[:, 1] += eField(ions[:, 0]) * dt * (constants.elementary_charge / mass)
            ions[:, 0] += ions[:, 1] * dt

    plot(histogram(electrons[:, 1], grid)[0])
    show()

if __name__ == '__main__':
    main()
