
import numpy as np
from scipy import integrate as integ
import scipy.linalg as la

import linear_response as lr

# References
# ----------
# [1] Origin of the magnetic spin Hall effect:
#     Spin current vorticity in the Fermi sea
#     Physical Rreview R 2, 023065 (2020)
#       DOI: 10.1103/PhysRevResearch.2.023065
# -----------------------------------------------------------------------------


class Rashba_model():
    """
    Electronic model based on Eq. 24 of [1].
    """

    def __init__(self, path, alpha, B=0, th=0, phi=0):
        self.alpha = alpha
        self.set_Zeeman(B, th, phi)

    def set_Zeeman(self, B, th, phi):
        nx = np.sin(th) * np.cos(phi)
        ny = np.sin(th) * np.sin(phi)
        nz = np.cos(th)
        h_zeeman = B * np.array([
            [nz, nx - 1j*ny],
            [nx + 1j*ny, -nz]
        ])
        self.Hzee = h_zeeman

    def hamiltonian(self, kx, ky):
        A = 1
        kinetic = A * (kx**2 + ky**2)
        rashba_01 = self.alpha*(-ky - 1j * kx)
        rashba_10 = self.alpha*(-ky + 1j * kx)
        H_r = np.array([
            [kinetic, rashba_01],
            [rashba_10, kinetic]
        ])
        H = H_r + self.Hzee
        return H

    def velocity_operator(self, kx, ky, dk=1e-10):
        H = self.hamiltonian(kx, ky)
        Hdx = self.hamiltonian(kx + dk, ky)
        Hdy = self.hamiltonian(kx, ky + dk)
        vx = (Hdx - H) / dk
        vy = (Hdy - H) / dk
        return [vx, vy]

    def solve_one(self, kx, ky, eig_vectors=False):
        H = self.hamiltonian(kx, ky)
        diagonalization = la.eigh(H, eigvals_only=not(eig_vectors))
        return diagonalization

    def create_bands_grid(self, nk=100, kmax=10):
        k = np.linspace(-kmax, kmax, num=nk)
        kx, ky = np.meshgrid(k, k)
        bands = np.zeros((2, nk, nk))
        for i1 in range(nk):
            for i2 in range(nk):
                bands[:, i1, i2] = self.solve_one(kx[i1, i2], ky[i1, i2])
        self.bands_grid = bands

    def plot_bands_2d(self, fig, ax, n, kmax=10, cmap="seismic"):
        extent = (-kmax, kmax, -kmax, kmax)
        img = ax.imshow(self.bands_grid[n, :, :],
                        origin="lower", extent=extent, cmap=cmap)
        fig.colorbar(img)
        ax.set_xlabel("$k_x$", fontsize=15)
        ax.set_ylabel("$k_y$", fontsize=15)

    def spin_conductivity_k(self, kx, ky, Ef, i, a, b, Gamma):
        """
        """
        eivals, eivecs = self.solve_one(kx, ky, eig_vectors=True)
        v = self.velocity_operator(kx, ky)
        sigma_k = lr.spin_conductivity_k(
            eivals, eivecs, v, Ef, i, a, b, Gamma)
        return sigma_k

    def spin_conductivity(self, Ef, i, a, b, Gamma, kmax=10):
        opts = {"epsabs": 1e-5}
        ranges = [[-kmax, kmax], [-kmax, kmax]]
        args = (Ef, i, a, b, Gamma)
        result, abserr = integ.nquad(self.spin_conductivity_k, ranges, args=args,
                                     opts=opts)
        return result, abserr
