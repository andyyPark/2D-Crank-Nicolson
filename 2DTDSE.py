import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse 
import scipy.sparse.linalg
from scipy.integrate import trapz
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class Schrodinger(object):

    def __init__(self, **kwargs):
        self.set_coordinate(kwargs)
        self.set_constants(kwargs)
        self.psi0 = self.wavepacket(self.X, self.Y, self.xa, self.ya,
                               self.k0x, self.k0y)
        self.psi0 = self.psi0 / np.sqrt(self.normalization(self.psi0))
        self.V = self.set_potential(self.X, self.Y)

    def plot_average_y(self, PSI):
        average_y = np.zeros(self.T)
        average_y2 = np.zeros(self.T)

        for n in range(0, self.T - 1):
            psi = PSI[:, n]
            average_y[n] = trapz(trapz(self.Y * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)
            average_y2[n] = trapz(trapz(self.Y ** 2 * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)

        delta_y = np.sqrt(average_y2 - average_y)

        fig, ax = plt.subplots(1, 3, figsize=(12, 7))
        ax[0].plot(self.t, average_y)
        ax[0].set_title('Average y')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$y$')
        ax[1].plot(self.t, average_y2)
        ax[1].set_title('Average $y^2$')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$y^2$')
        ax[2].plot(self.t, delta_y)
        ax[2].set_title('Uncertainity $\Delta x$')
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$\Delta y$')
        fig.savefig('Uncertainty_y', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.show()

    def plot_average_x(self, PSI):
        average_x = np.zeros(self.T)
        average_x2 = np.zeros(self.T)

        for n in range(0, self.T - 1):
            psi = PSI[:, n]
            average_x[n] = trapz(trapz(self.X * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)
            average_x2[n] = trapz(trapz(self.X ** 2 * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)

        delta_x = np.sqrt(average_x2 - average_x)

        fig, ax = plt.subplots(1, 3, figsize=(12, 7))
        ax[0].plot(self.t, average_x)
        ax[0].set_title('Average $x$')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[1].plot(self.t, average_x2)
        ax[1].set_title('Average $x^2$')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x^2$')
        ax[2].plot(self.t, delta_x)
        ax[2].set_title('Uncertainity $\Delta x$')
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$\Delta x$')
        fig.savefig('Uncertainty_x', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.show()

    def plot_normalization(self, PSI):
        normalization = np.zeros(self.T)

        for n in range(0, self.T - 1):
            psi = PSI[:, n]
            normalization[n] = self.normalization(psi.reshape(self.J, self.L))
        plt.plot(self.t, normalization)
        plt.xlabel('$t$')
        plt.ylabel('$N$')
        plt.title('Normalization')
        plt.savefig('Normalization.png')
        plt.show()

    def solve(self):
        U1, U2 = self.sparse_matrix()
        LU = scipy.sparse.linalg.splu(U1)
        PSI = np.zeros((self.N, self.T), dtype=complex)
        PSI[:, 0] = self.psi0.reshape(self.N)

        for n in range(0, self.T - 1):
            b = U2.dot(PSI[:, n])
            psi = LU.solve(b)
            PSI[:, n + 1] = psi
        return PSI

    def sparse_matrix(self):
        b = 1 + 1j * self.dt * self.hbar ** 2 * (1 / (self.dx ** 2.0) + 1 / (self.dy ** 2.0)) \
        + 1j * self.dt * self.V.reshape(self.N) / (2 * self.hbar)
        c = -1j * self.dt * self.hbar / (4 * self.mass * self.dx ** 2) * np.ones(self.N, dtype=complex)
        a = c
        d = -1j * self.dt * self.hbar / (4 * self.mass * self.dy ** 2) * np.ones(self.N, dtype=complex)
        e = d

        f = 1 - 1j * self.dt * self.hbar ** 2 * (1 / (self.dx ** 2) + 1 / (self.dy ** 2)) \
            - 1j * self.dt * self.V.reshape(self.N) / (2 * self.hbar)
        g = 1j * self.dt * self.hbar / (4 * self.mass * self.dx ** 2) * np.ones(self.N, dtype=complex)
        h = g
        k = 1j * self.dt * self.hbar / (4 * self.mass * self.dy ** 2) * np.ones(self.N, dtype=complex)
        p = k

        U1 = np.array([c, e, b, d, a])
        diags = np.array([-self.J, -1, 0, 1, self.J])
        A = scipy.sparse.spdiags(U1, diags, self.N, self.N)
        A = A.tocsc()

        U2 = np.array([h, p, f, k, g])
        B = scipy.sparse.spdiags(U2, diags, self.N, self.N)
        B = B.tocsc()

        return (A, B)
        
    def set_potential(self, x, y):
        return np.zeros((len(x), len(y)))

    def normalization(self, psi):
        return trapz(trapz(abs(psi ** 2), self.x, dx=self.dx), self.y, dx=self.dy)

    def wavepacket(self, x, y, xa, ya, k0x, k0y):
        N = 1.0 / (2 * np.pi * self.sigmax * self.sigmay)
        e1x = np.exp(-(x - xa) ** 2.0 / (2.0 * self.sigmax ** 2.0))
        e1y = np.exp(-(y - ya) ** 2.0 / (2.0 * self.sigmay ** 2.0))
        e2 = np.exp(1j * k0x * x + 1j * k0y * y)
        return N * e1x * e1y * e2

    def get_shape(self):
        return (self.J, self.L)

    def get_meshgrid(self):
        return (self.X, self.Y)

    def set_constants(self, args):
        self.mass = args['mass']
        self.hbar = args['hbar']

    def set_coordinate(self, args):
        self.nx = args['nx']
        self.ny = args['ny']
        self.x0 = args['x0']
        self.xf = args['xf']
        self.y0 = args['y0']
        self.yf = args['yf']
        self.xa = args['xa']
        self.ya = args['ya']
        self.t0 = args['t0']
        self.tf = args['tf']
        self.dt = args['dt']
        self.sigmax = args['sigmax']
        self.sigmay = args['sigmay']
        self.k0x = args['k0x']
        self.k0y = args['k0y']
        self.Nt = int(round(self.tf / float(self.dt)))
        self.t = np.linspace(self.t0, self.Nt * self.dt, self.Nt)
        self.x = np.linspace(self.x0, self.xf, self.nx + 1)
        self.y = np.linspace(self.y0, self.yf, self.ny + 1)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.J = len(self.x)
        self.L = len(self.y)
        self.N = self.J * self.L
        self.T = len(self.t)


if __name__ == '__main__':
    args = {
        'nx': 100,
        'ny': 100, 
        'x0': -40, 
        'xf': 40, 
        'y0': -40,
        'yf': 40,
        'xa': 4, 
        'ya': 3,
        't0': 0, 
        'tf': 7.0,
        'dt': 0.01, 
        'sigmax': 2.0, 
        'sigmay': 2.0,
        'k0x': 0,
        'k0y': 0,
        'mass': 0.5,
        'hbar': 1
        }
    schrodinger = Schrodinger(**args)
    PSI = schrodinger.solve()

    play_video = False

    if play_video:
        X, Y = schrodinger.get_meshgrid()
        J, L = schrodinger.get_shape()

        def update(i, PSI, surf):
            ax.clear()
            psi = abs(PSI[:, i].reshape(J, L) ** 2)
            surf = ax.plot_surface(X, Y, psi, \
            rstride=1, cstride=1, cmap='plasma')
            ax.set_zlim(0, 0.3)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$|\Psi(x, y)|^2$')
            return surf
        
        frames = PSI.shape[1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$|\Psi(x, y)|^2$')
        surf = ax.plot_surface(X, Y, abs(PSI[:, 0].reshape(J, L) ** 2), \
                rstride=1, cstride=1, cmap='plasma')
        ax.set_zlim(0, 0.3)
        ani = animation.FuncAnimation(fig, update, frames=frames,\
                fargs=(PSI, surf), \
                interval=30, blit=False)

        plt.show()

    schrodinger.plot_normalization(PSI)
    schrodinger.plot_average_x(PSI)
    schrodinger.plot_average_y(PSI)
    