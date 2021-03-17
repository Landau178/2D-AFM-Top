import pathlib

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import integrate as integ
import scipy.linalg as la

from multiprocessing import Pool, cpu_count


import bz_utilities as bzu
import linear_response as lr
import toy_models as toy
import plot_utils as pltu

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

    def __init__(self, path, alpha, B=0, th=0, phi=0, lamb=0):
        self.A = 0.238  # hbar**2 / m in eV nm**2
        self.alpha = alpha
        self.set_Zeeman(B, th, phi)
        self.lamb = lamb
        self.path = pathlib.Path(path).absolute()
        self.mode_s_cond = "zelezny"

    def set_Zeeman(self, B, th, phi):
        self.B = B
        nx = np.sin(th) * np.cos(phi)
        ny = np.sin(th) * np.sin(phi)
        nz = np.cos(th)
        h_zeeman = B * np.array([
            [nz, nx - 1j*ny],
            [nx + 1j*ny, -nz]
        ])
        self.Hzee = h_zeeman

    def h_warping(self, kx, ky):
        """
        Cubic term in k which breaks the rotational symmetry, and
        reduced it to C3. This lets xz as a reflection plane.
        """
        k_plus = kx + 1j*ky
        warping = 0.5j*self.lamb * (k_plus**3 - np.conjugate(k_plus)**3)
        hw = np.array([
            [warping, 0],
            [0, -warping]
        ])
        return hw

    def hamiltonian(self, kx, ky):
        kinetic = 0.5 * self.A * (kx**2 + ky**2)
        rashba_01 = self.alpha*(-ky - 1j * kx)
        rashba_10 = self.alpha*(-ky + 1j * kx)
        H_r = np.array([
            [kinetic, rashba_01],
            [rashba_10, kinetic]
        ])
        H = H_r + self.Hzee + self.h_warping(kx, ky)
        return H

    def velocity_operator(self, kx, ky, dk=1e-10, reshape=True):
        H = self.hamiltonian(kx, ky)
        Hdx = self.hamiltonian(kx + dk, ky)
        Hdy = self.hamiltonian(kx, ky + dk)
        vx = (Hdx - H) / dk
        vy = (Hdy - H) / dk
        if reshape:
            vx = np.reshape(vx, (1, 2, 1, 2))
            vy = np.reshape(vy, (1, 2, 1, 2))
        return [vx, vy]

    def solve_one(self, kx, ky, eig_vectors=False, reshape=False, fix_gauge=True):
        """
        Diagonalize Hamiltonian for a given k-points.
        Eigenvetors are returned in format:
            eivecs[band, spin_component]
        """
        H = self.hamiltonian(kx, ky)
        diagonalization = la.eigh(H, eigvals_only=not(eig_vectors))
        if eig_vectors:
            eivals, eivecs = diagonalization
            diagonalization = eivals, eivecs.T
        if eig_vectors and fix_gauge:
            eivals, eivecs = diagonalization
            eivecs = bzu.fix_gauge_eigenvector(eivecs)
            diagonalization = eivals, eivecs

        if reshape and eig_vectors:
            eivals, eivecs = diagonalization
            eivecs = np.reshape(eivecs, (2, 1, 2))
            diagonalization = eivals, eivecs
        return diagonalization

    def create_bands_grid(self, nk=100, kmax=10):
        k = np.linspace(-kmax, kmax, num=nk)
        kx, ky = np.meshgrid(k, k)
        bands = np.zeros((2, nk, nk))
        for i1 in range(nk):
            for i2 in range(nk):
                bands[:, i1, i2] = self.solve_one(kx[i1, i2], ky[i1, i2])
        self.bands_grid = bands

    def plot_bands_1d(self, ax, nk=200, kmax=4):
        """
        Plot bands along axis y
        """
        kx = np.zeros(nk)
        ky = np.linspace(-kmax, kmax, nk)
        bands = np.zeros((nk, 2))
        for i in range(nk):
            bands[i, :] = self.solve_one(kx[i], ky[i])
        ax.plot(ky, bands[:, 0], color="purple")
        ax.plot(ky, bands[:, 1], color="purple")
        ax.set_xlabel("$k_y$     $[nm^{-1}]$")
        ax.set_ylabel("Energy   [eV]")
        kd = self.B/self.alpha
        #print("kd = {}".format(kd))
        ks = self.alpha / self.A
        ax.axvline(x=ks, ls="--", color="black", lw=0.5)
        ax.axvline(x=-ks,  ls="--", color="black", lw=0.5)
        ax.axvline(x=kd,  ls="--", color="green", lw=0.5)

    def plot_bands_2d(self, fig, ax, n, kmax=10, cmap="seismic"):
        extent = (-kmax, kmax, -kmax, kmax)
        img = ax.imshow(self.bands_grid[n, :, :],
                        origin="lower", extent=extent, cmap=cmap)
        fig.colorbar(img)
        ax.contour(self.bands_grid[n, :, :],
                   origin="lower", extent=extent)
        ax.set_xlabel("$k_x$", fontsize=15)
        ax.set_ylabel("$k_y$", fontsize=15)

    def plot_bands_3d(self, ax, n, kmax=10, nk=100):
        k = np.linspace(-kmax, kmax, num=nk)
        kx, ky = np.meshgrid(k, k)
        ax.plot_surface(kx, ky, self.bands_grid[n, :, :],
                        linewidth=0, antialiased=False)

# -----------------------------------------------------------------------------

    def spin_conductivity_k(self, kx, ky, Ef, i, a, b, Gamma, mode="zelezny"):
        """
        """
        eivals, eivecs = self.solve_one(kx, ky, eig_vectors=True, reshape=True)
        v = self.velocity_operator(kx, ky)
        spin_cond_function = {
            "zelezny": lr.spin_conductivity_k,
            "zelezny_intra": lr.spin_conductivity_k_zelezny_intra,
            "zelezny_inter": lr.spin_conductivity_k_zelezny_inter,
            "mook": lr.spin_conductivity_k_mook,
            "mook2": lr.spin_conductivity_k_mook_gaussian,
            "mook_intra": lr.spin_conductivity_k_mook_intra,
            "mook_inter": lr.spin_conductivity_k_mook_inter
        }[mode]
        sigma_k = spin_cond_function(
            eivals, eivecs, v, Ef, i, a, b, Gamma)
        return sigma_k

    def spin_conductivity(self, Ef, component, nk=200, Gamma=12.7e-3,
                          integrate=True):
        """
        Calculate the spin conductivity integrating ina regular k-grid.
        """
        kx_min, kx_max, ky_min, ky_max = limits_k_occup(
            Ef, self.alpha, self.B, 0, factor=1.7)
        kkx = np.linspace(kx_min, kx_max, nk)
        kky = np.linspace(ky_min, ky_max, nk)
        kx, ky = np.meshgrid(kkx, kky)
        s_cond = np.zeros_like(kx)
        for x in range(nk):
            for y in range(nk):
                s_cond[x, y] = self.spin_conductivity_k(
                    kx[x, y], ky[x, y], Ef, *component, Gamma)
        if integrate:
            integ_s_cond = integ.simps(
                [integ.simps(s_cond_1d, kky) for s_cond_1d in s_cond], kkx)
        else:
            integ_s_cond = s_cond
        return integ_s_cond

    def spin_conductivity_quad(self, Ef, i, a, b, Gamma, kmax=10):
        opts = {"epsabs": 1e-5}
        ranges = [[-kmax, kmax], [-kmax, kmax]]
        args = (Ef, i, a, b, Gamma)
        result, abserr = integ.nquad(self.spin_conductivity_k, ranges, args=args,
                                     opts=opts)
        return result, abserr

    def spin_conductivity_work(self, Ef, component, Gamma, mode, limits, nk,
                               i_0, i_f):
        """
        Calculate the spin conductivity integrating ina regular k-grid.
        Work for parallel computing in a fraction of the BZ.
        """
        kx_min, kx_max, ky_min, ky_max = limits
        kkx = np.linspace(kx_min, kx_max, nk)
        kky = np.linspace(ky_min, ky_max, nk)
        kx, ky = np.meshgrid(kkx, kky)
        s_cond = np.zeros_like(kx)
        for x in range(i_0, i_f):
            for y in range(nk):
                s_cond[x, y] = self.spin_conductivity_k(
                    kx[x, y], ky[x, y], Ef, *component, Gamma, mode)

        integ_s_cond = integ.simps(
            [integ.simps(s_cond_1d, kky) for s_cond_1d in s_cond], kkx)

        return integ_s_cond

    def spin_conductivity_parallel(self, Ef, component, nk=200, Gamma=12.7e-3,
                                   mode="zelezny", nproc=4):
        """
        Parallel calculation of spin conductivity.
        """
        limits = limits_k_occup(
            Ef, self.alpha, self.B, 0, factor=1.7)
        common_args = (Ef, component, Gamma, mode, limits, nk)
        args_list = self.work_args(common_args, nk, nproc)

        with Pool(nproc) as pool:
            sigma_k_list = pool.starmap(
                self.spin_conductivity_work, args_list)
        sigma_k = np.sum(np.array(sigma_k_list))
        return sigma_k
# -----------------------------------------------------------------------------

    def charge_conductivity_k(self, kx, ky, Ef, a, b, Gamma, mode):
        """
        Total (odd+even) charge conductivity.
        """
        eivals, eivecs = self.solve_one(kx, ky, eig_vectors=True, reshape=True)
        v = self.velocity_operator(kx, ky)
        charge_cond_dict = {
            "odd_z": (lr.charge_conductivity_k, (Gamma,)),
            "odd_m": (lr.charge_conductivity_k_odd_Mook, (Gamma,)),
            "even_z": (lr.charge_conductivity_k_even, ()),
            "even_m": (lr.charge_conductivity_k_even_Mook, (Gamma,))
        }
        charge_cond_func, extra_args = charge_cond_dict[mode]
        sigma_k = charge_cond_func(
            eivals, eivecs, v, Ef, a, b, *extra_args)
        return sigma_k

    def charge_conductivity(self, Ef, component, nk=200, Gamma=12.7e-3,
                            integrate=True, mode="odd_z"):
        """
        Calculate the spin conductivity integrating ina regular k-grid.
        """
        kx_min, kx_max, ky_min, ky_max = limits_k_occup(
            Ef, self.alpha, self.B, 0, factor=1.7)
        kkx = np.linspace(kx_min, kx_max, nk)
        kky = np.linspace(ky_min, ky_max, nk)
        kx, ky = np.meshgrid(kkx, kky)
        c_cond = np.zeros_like(kx)
        for x in range(nk):
            for y in range(nk):
                c_cond[x, y] = self.charge_conductivity_k(
                    kx[x, y], ky[x, y], Ef, *component, Gamma, mode)
        if integrate:
            integ_c_cond = integ.simps(
                [integ.simps(c_cond_1d, kky) for c_cond_1d in c_cond], kkx)
        else:
            integ_c_cond = c_cond
        return integ_c_cond

    def charge_conductivity_work(self, Ef, component, Gamma, mode,  limits, nk, i_0, i_f):
        kx_min, kx_max, ky_min, ky_max = limits
        kkx = np.linspace(kx_min, kx_max, nk)
        kky = np.linspace(ky_min, ky_max, nk)
        kx, ky = np.meshgrid(kkx, kky)
        c_cond = np.zeros_like(kx)
        for x in range(i_0, i_f):
            for y in range(nk):
                c_cond[x, y] = self.charge_conductivity_k(
                    kx[x, y], ky[x, y], Ef, *component, Gamma, mode)

        integ_c_cond = integ.simps(
            [integ.simps(c_cond_1d, kky) for c_cond_1d in c_cond], kkx)
        return integ_c_cond

    def work_args(self, common_args, nk, partitions):
        nodes = []
        length = int(nk / partitions)
        for i in range(partitions):
            nodes.append(length * i)
        nodes.append(nk)
        args_list = []
        for i in range(partitions):
            index_args = (nodes[i], nodes[i+1])
            args_list.append((*common_args, *index_args))
        args_list = (*args_list,)
        return args_list

    def charge_conductivity_parallel(self, Ef, component, nk=200, Gamma=12.7e-3, mode="odd_z", nproc=4):
        """
        Parallel calculation of charge conductivity.
        """
        limits = limits_k_occup(
            Ef, self.alpha, self.B, 0, factor=1.7)
        common_args = (Ef, component, Gamma, mode, limits, nk)
        args_list = self.work_args(common_args, nk, nproc)

        with Pool(nproc) as pool:
            sigma_k_list = pool.starmap(
                self.charge_conductivity_work, args_list)
        sigma_k = np.sum(np.array(sigma_k_list))
        return sigma_k

# -----------------------------------------------------------------------------
# Methods for calculating spin conductivity as funcion of Ef
# -----------------------------------------------------------------------------

    def spin_conductivity_vs_Ef(self, component, Gamma, nE=50, nk=200, mode="zelezny", nproc=4):
        Ef_arr = np.linspace(-0.3, 0.2, nE)
        s_cond = np.zeros_like(Ef_arr)
        for i in range(nE):
            Ef = Ef_arr[i]
            s_cond[i] = self.spin_conductivity_parallel(
                Ef, component, nk=nk, Gamma=Gamma, mode=mode, nproc=nproc)
        dir_Ef, dir_s_cond = self.directory_s_cond(component, Gamma, mode)
        np.save(dir_s_cond, s_cond)
        np.save(dir_Ef, Ef_arr)
        return Ef_arr, s_cond

    def load_spin_conductivity_vs_Ef(self, component, Gamma, mode):
        dir_Ef, dir_s_cond = self.directory_s_cond(component, Gamma, mode)
        s_cond = np.load(dir_s_cond)
        Ef_arr = np.load(dir_Ef)
        return Ef_arr, s_cond

    def directory_s_cond(self, component, Gamma, mode):
        cart_comp = pltu.int2cart(component)
        str_comp = cart_comp[0]+cart_comp[1]+cart_comp[2]
        str_G = np.round(Gamma*1e3, decimals=2)
        ending = "Gamma={}meV_{}.npy".format(str_G, mode)
        dir_cond = self.path / "sigma_{}".format(str_comp)
        toy.mk_dir(dir_cond)
        dir_s_cond = dir_cond / "s_cond_{}_{}".format(str_comp, ending)
        dir_Ef = dir_cond / "Ef_arr_{}".format(ending)
        return dir_Ef, dir_s_cond


# ----------------------------------------------------------------------------
# Methods to claculate charge conductivity as function of the Fermi level
# ----------------------------------------------------------------------------

    def charge_conductivity_vs_Ef(self, component, Gamma, nE=50, nk=200, mode="odd_z", nproc=4):
        Ef_arr = np.linspace(-0.3, 0.2, nE)
        c_cond = np.zeros_like(Ef_arr)
        for i in range(nE):
            Ef = Ef_arr[i]
            c_cond[i] = self.charge_conductivity_parallel(
                Ef, component, nk=nk, Gamma=Gamma, mode=mode, nproc=nproc)
        dir_Ef, dir_c_cond = self.directory_c_cond(component, Gamma, mode)
        np.save(dir_c_cond, c_cond)
        np.save(dir_Ef, Ef_arr)
        return Ef_arr, c_cond

    def load_charge_conductivity_vs_Ef(self, component, Gamma, mode):
        dir_Ef, dir_c_cond = self.directory_c_cond(component, Gamma, mode)
        c_cond = np.load(dir_c_cond)
        Ef_arr = np.load(dir_Ef)
        return Ef_arr, c_cond

    def directory_c_cond(self, component, Gamma, mode):
        cart_comp = pltu.int2cart(component)
        str_comp = cart_comp[0]+cart_comp[1]
        str_G = np.round(Gamma*1e3, decimals=2)
        ending = "Gamma={}meV_{}.npy".format(str_G, mode)
        dir_cond = self.path / "sigma_{}".format(str_comp)
        toy.mk_dir(dir_cond)
        dir_c_cond = dir_cond / "c_cond_{}_{}".format(str_comp, ending)
        dir_Ef = dir_cond / "Ef_arr_{}".format(ending)
        return dir_Ef, dir_c_cond


# -----------------------------------------------------------------------------
# Method for calculating spin current expectation value and its vorticity.
# -----------------------------------------------------------------------------

    def spin_current(self, kx, ky, n, i, a):
        eivecs = self.solve_one(kx, ky, eig_vectors=True,
                                reshape=True, fix_gauge=True)[1]
        v = self.velocity_operator(kx, ky)
        S = bzu.pauli_matrix(i) / 2
        S_eig = np.einsum("nis, st, mit-> nm", eivecs.conj(), S, eivecs)
        vx_eig = np.einsum("nis, isjd, mjd-> nm",
                           eivecs.conj(), v[0], eivecs)
        vy_eig = np.einsum("nis, isjd, mjd-> nm",
                           eivecs.conj(), v[1], eivecs)
        v_eig = [vx_eig, vy_eig]
        js = 0.5 * (S_eig @ v_eig[a] + v_eig[a] @ S_eig)
        return np.real(js[n, n])

    def vorticity(self, kx, ky, i, n):
        """
        Eq 13 of Ref [1]
        Parameters:
        -----------
            kx, ky: (float, float)
                Wave vector components.
            i: (int)
                spin component
            n: (int)
                band index
        """
        dk = 1e-5
        j0_x = self.spin_current(kx, ky, n, i, 0)
        j0_y = self.spin_current(kx, ky, n, i, 1)
        jdx_y = self.spin_current(kx + dk, ky, n, i, 1)
        jdy_x = self.spin_current(kx, ky+dk, n, i, 0)
        dx_jy = (jdx_y-j0_y) / dk
        dy_jx = (jdy_x - j0_x) / dk
        return dx_jy - dy_jx


# ------------------------------------------------------------------------------
def create_path_rashba_model(folder, alpha, B=0, th=0, phi=0, lamb=0):
    """
    """
    str_a = toy.float2str(alpha)
    str_B = toy.float2str(B)
    str_th = toy.float2str(th)
    str_phi = toy.float2str(phi)
    str_lamb = toy.float2str(lamb)
    str_args = (str_a, str_B, str_th, str_phi, str_lamb)
    str_params = "alpha={}_B={}_(th,phi)=({},{})_lamb={}".format(*str_args)
    path_sim = "saved_simulations/toy_model/rashba/{}".format(folder)
    path = toy.ROOT_DIR / path_sim / str_params

    toy.mk_dir(path)
    return pathlib.Path(path)


# -----------------------------------------------------------------------------
# Not completely tested functions to decide the size
# of the integration area in k-space
# in the Rashba model.
# -----------------------------------------------------------------------------


def k_fermi(Ef, alpha, B, signs=(-1, -1, 1)):
    s1, s2, s3 = signs
    A = 0.238  # hbar**2 / m in eV nm**2
    kd = B / alpha
    kmin = alpha / A
    discr = 1+2*(Ef/(alpha*kmin) + s3*kd/kmin)
    if discr < 0:
        s1, s2, s3 = (1, 1, 1)
        discr = 1+2*(Ef/(alpha*kmin) + s3*kd/kmin)
    factor = (s1 + s2*np.sqrt(discr))
    return kmin * factor


def limits_k_occup(Ef, alpha, B, band, factor=1.1):
    signs = {
        0: ((-1, -1, 1), (1, 1, -1)),
        1: ((1, -1, -1), (-1, 1, 1))}[band]
    sign1, sign2 = signs
    k1 = k_fermi(Ef, alpha, B, signs=sign1)
    k2 = k_fermi(Ef, alpha, B, signs=sign2)
    # print(k1,k2)
    center = 0.5*(k1+k2)
    diameter = np.abs(k2-k1)*factor
    kx_min, kx_max = -diameter/2, diameter/2
    ky_min, ky_max = center - diameter/2, center + diameter/2
    limits = (kx_min, kx_max, ky_min, ky_max)
    A = 0.238  # hbar**2 / m in eV nm**2
    e_d = 0.5*A * (B/alpha)**2
    e_min = - 0.5 * alpha**2 / A - B
    if band == 1 and Ef < e_d:
        limits = (-1, 1, -1, 1)
    if band == 0 and Ef < e_min:
        limits = (-1, 1, -1, 1)
    return limits
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Pure analytical results in Rashba model.
# -----------------------------------------------------------------------------


class Analytical_Rashba_model():
    """
    This class groups some analytical results in the model defined by the
    Rashba Hamiltonian (eq. 24 of [1].
    """

    def __init__(self, alpha, B, phi):
        self.A = 0.238  # hbar**2 / m in eV nm**2
        self.alpha = alpha
        self.B = B
        self.phi = phi
        self.kd = B/alpha
        self.kd_v = self.kd * np.array([-np.sin(phi), np.cos(phi)])

    def polar_coord(self, kx, ky):
        k = np.sqrt(kx**2 + ky**2)
        phi_k = np.arctan2(ky, kx)
        return k, phi_k

    def q(self, kx, ky):
        """
        Distance from k-point from the Dirac cone.
        """
        q = np.sqrt((kx-self.kd_v[0])**2 + (ky-self.kd_v[1])**2)
        return q

    def rashba_energy(self, kx, ky):
        """
        Analytical spectrum of rashba Hamiltonian.
        """
        E_kin = 0.5*self.A * (kx**2 + ky**2)
        E_rash = self.alpha * self.q(kx, ky)
        return np.array([E_kin-E_rash, E_kin + E_rash])

    def omega_x(self, kx, ky, band):
        """
        x-component of spin current vorticity.
        """
        sgn_band = {0: -1, 1: 1}[band]
        q = self.q(kx, ky)
        k_cross_kd = kx * self.kd_v[1]-ky*self.kd_v[0]
        term1 = k_cross_kd * (ky - self.kd_v[1])/q**3
        term2 = kx / q
        return sgn_band * 0.5*self.A * (term1 + term2)

    def omega_y(self, kx, ky, band):
        """
        y-component of spin current vorticity.
        """
        sgn_band = {0: -1, 1: 1}[band]
        q = self.q(kx, ky)
        k_cross_kd = kx * self.kd_v[1]-ky*self.kd_v[0]
        term1 = k_cross_kd * (-kx + self.kd_v[0])/q**3
        term2 = ky / q
        return sgn_band * 0.5 * self.A * (term1 + term2)

    def chi_k(self, kx, ky):
        """
        Unitary complex function which appears in eigenvetors.
        """
        k, phi_k = self.polar_coord(kx, ky)
        q = self.q(kx, ky)
        chi = -1j * k * np.exp(-1j*phi_k) + self.kd * np.exp(-1j * self.phi)
        chi = chi / q
        return chi

    def spin_current_exp_val(self, kx, ky, i, a, band):
        """
        Expectation value <kn| J^i_a | kn>,
        where "i" labels the spin component and
        "a" is the velocity component.
        Parameters:
        -----------
            kx, ky: (float, float)
                Wave vector components.
            i, a, band: (int, int int)
                Each one can be 0 or 1,
                the are the spin, velocity and
                band components respectively.
        """
        k_a = {0: kx, 1: ky}[a]
        sign_band = {0: -1, 1: 1}[band]
        spin_sign = {0: 1, 1: -1}[i]
        f = {0: np.real, 1: np.imag}[i]
        chi = spin_sign * f(self.chi_k(kx, ky))
        j = 0.5 * self.A * sign_band * k_a * chi
        return j
