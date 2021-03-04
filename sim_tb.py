
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import linalg as la
from scipy import optimize as opt
from scipy import integrate as integ
import pathlib

import pythtb as pytb

import bz_utilities as bzu
import linear_response as lr
import toy_models as toy
import file_manager as fm


class Simulation_TB():
    """
    This class perform calculations of tight-binding models,
    using the library PythTB. To be initialized, it needs a
    path to the simulation directory, which inside contains a
    config file path/config.json, with some options for initialization.

    config.json
    -----------
        dim_k: (int)
            Dimensionality of reciprocal space.
        dim_r: (int)
            Dimensionality of real space
        lat: (list)
            array containing lattice vectors
        orb: (list)
            Array containing reduced coordinates of all
            tight-binding orbitals.
        nspin: int
            2 if spin degree of freedom is not counted in orb.
            1 if it is.
        Ne: (int)
            Number of electrons in the unit cell.
        k_spoints: (list of floats)
            k special points in reduced coordinates.
        k_sp_labels: (list of str)
            Labels for the k special points.
        hop_files: (list of str)
            Name of the hoppings files

    """

    def __init__(self, path):
        self.path = pathlib.Path(path).absolute()
        # self.save_config()
        self.file_man = fm.Sim_TB_file_manager(self)

        self.model = pytb.tb_model(self.dim_k, self.dim_r,
                                   lat=self.lat, orb=self.orb,
                                   nspin=self.nspin)
        self.file_man.set_Hamiltonian()

    def plot_bands(self, ax, color="green", lw=3):
        """
        Plot the bands in the given path of BZ.

        Parameters:
        -----------
            ax: (matplotlib.axes.Axes)
                Axes for plot.
            color: (str or color, defualt is "green")
                Color of lines in plot.
            lw: (float, default is 3)
                Line width.

        Returns:
        -------
            None
        """
        path = self.k_spoints
        (k_vec, k_dist, k_node) = self.model.k_path(path, 1000, report=False)
        evals = self.model.solve_all(k_vec)
        for i in range(self.nband):
            ax.plot(k_dist, evals[i, :], color=color, lw=lw)

        ax.set_xticks(k_node)
        ax.set_xticklabels(self.k_sp_labels, fontsize=15)
        ax.set_xlim([k_node[0], k_node[-1]])
        for node in range(1, len(k_node)-1):
            ax.axvline(x=k_node[node], color="black", ls="--", lw=0.5)

    def plot_bands_2d(self, j, fig=None, ax=None, nk=100, bands=None, delta_k=0.8*2*np.pi):
        """
        Olny valid for k_dim = 2.
        Parameters:
        -----------
            j: (int)
                Index of band.
            fig: (matplotlib.figure.Figure)
                Figure to plot.
            ax: (matplotlib.axes.Axes)
                Axes to plot.
            nk: (int, default is 10)
                Size of the k-grid.
            bands: (np.ndarray, shape (nbands,nk,nk))
                Bands to plot, if None, grid is calculated.
        Returns:
        --------
            None
        """
        if bands is None:
            self.create_bands_grid(nk=nk, delta_k=delta_k)
        if (ax is None) or (fig is None):
            fig, ax = plt.subplots()

        extent = (-delta_k, delta_k, -delta_k, delta_k)
        img = ax.imshow(self.bands_grid[j, :, :],
                        origin="lower", extent=extent)
        fig.colorbar(img)

        for i in range(1, len(self.k_spoints)):
            kf_red = self.k_spoints[i]
            k0_red = self.k_spoints[i-1]
            kf = kf_red[0] * self.rlat[0] + kf_red[1] * self.rlat[1]
            k0 = k0_red[0] * self.rlat[0] + k0_red[1] * self.rlat[1]
            ax.plot([k0[0], kf[0]], [k0[1], kf[1]], ls="--", color="white")

        for ks_red in self.k_spoints:
            k_vec = ks_red[0] * self.rlat[0] + ks_red[1] * self.rlat[1]
            ax.plot(k_vec[0], k_vec[1], marker="o", markersize=5, color="red")
        ax.set_xlabel("$k_x$", fontsize=15)
        ax.set_ylabel("$k_y$", fontsize=15)
        if (ax is None) or (fig is None):
            return fig, ax

    def create_bands_grid_red_coord(self, nk=10, return_eivec=True, endpoint=True):
        """
        Creates a grid of the eivengavlues and eigenvectors in a grid
        (k1, k2), in such a wat that:
            k = k1 * b1 + k2 * b2
        with b1, b2, reciprocal lattice vectors.
        The grid includes nk x nk values of k1 and k2 in the range (0, 1).

        The solutions of the eigenvalue problem are saved in atributes:
            self.red_bands_grid
            self.red_eivecs_grid

        Note: Only valid for k_dim=2.

        Parameters:
        -----------
            nk: (int, default is 10)
            return_eivec: (bool, default is True)
                If True, eigenvectors are calculated.
            endpoint:
                If True, the last point and the first one are physically
                equivalent. (PB conditions)
        Returns:
        --------
            None
        """
        k = np.linspace(0, 1, num=nk, endpoint=endpoint)
        bands_grid = np.zeros((nk, nk, self.nband))
        eivecs_grid = np.zeros(
            (nk, nk, self.nband, len(self.orb), self.nspin), dtype="complex")
        for i in range(nk):
            for j in range(nk):
                k_red = [k[i], k[j]]
                if return_eivec:
                    eival, eivec = self.model.solve_one(
                        k_red, eig_vectors=True)
                    bands_grid[i, j, :] = eival
                    eivecs_grid[i, j, :, :, :] = eivec
                else:
                    eival = self.model.solve_one(k_red, eig_vectors=False)
                    bands_grid[i, j, :] = eival
        self.red_bands_grid = bands_grid
        if return_eivec:
            self.red_eivecs_grid = eivecs_grid

    def create_bands_grid(self, nk=50, delta_k=1.6*np.pi):
        """
        Creates a grid of the eigernvalues in cartesian coordinates of the
        k-space. Save the energies in atribute self.bands_grid.
        This function is used for ploting the band structure in 2D.

        Parameters:
        ----------
            nk: (int, defalt is 50)
                Size of k_grid.
            delta_k: (float, fedault is 1.6*np.pi)
                Extent of grid is [-delta_k, delta_k] in kx and ky.
        Returns:
        --------
            None
        """
        k = np.linspace(-delta_k, delta_k, nk)
        kx, ky = np.meshgrid(k, k)
        M = bzu.cartesian_to_reduced_k_matrix(self.rlat[0], self.rlat[1])
        bands = np.zeros((self.nband, nk, nk))

        for i1 in range(nk):
            for i2 in range(nk):
                k_vec = np.array([kx[i1, i2], ky[i1, i2]])
                k_red = M @ k_vec
                bands[:, i1, i2] = self.model.solve_one(k_red)
        self.bands_grid = bands

    def berry_flux(self, bands, nk=100):
        """
        Parameters:
        -----------
            bands: (list of ints)
                List containing the indices of the occupied bands.
            nk: (int, default is 100)
                Size of the sample in reciprocal space.

        Returns:
        --------
            plaq: (np.ndarray of shape (nk,nk))
                Array containing the Berry phases on individual plaquettes
                on BZ-grid.

        """
        w_BZ = pytb.wf_array(self.model, [nk, nk])
        w_BZ.solve_on_grid([0, 0])
        plaq = w_BZ.berry_flux(bands, individual_phases=True)
        return plaq

    def plot_berry_flux(self, berry_fluxes, fig, ax):
        """
        Parameters:
        -----------
            berry_fluxes: (np.ndarray)
                2D array containing berry curvatureon 1BZ.
            fig: (matplotlib.figure.Figure)
                Fig of plot.
            ax: (matplotlib.axes.Axes)
                Axes to plot.
        """
        extent = (0, 1.0, 0, 1.0)
        img = ax.imshow(berry_fluxes.T, origin="lower", extent=extent)
        fig.colorbar(img)
        # for i in range(1, len(self.k_spoints)):
        #     kf_red = np.array(self.k_spoints[i]) % 1
        #     k0_red = np.array(self.k_spoints[i-1]) % 1
        #     ax.plot([k0_red[0], kf_red[0]], [k0_red[1],
        #                                      kf_red[1]], ls="--", color="white")

        for ks_red in self.k_spoints:
            ks = np.array(ks_red) % 1  # to keep the point in the 1BZ
            ax.plot(ks[0], ks[1], marker="o",
                    markersize=5, color="red")

        ax.set_title("Berry curvature on 1BZ")
        ax.set_xlabel(r"$k_1$")
        ax.set_ylabel(r"$k_2$")

    def occupations_per_orbital(self, mu, T=0.001):
        """
        Calculate the occupations on each orbital, given a
        chemical potential mu.

        Note: Before calling this method, it is needed to
        create the grid with:
            self.create_bands_grid_red_coord

        Parameters:
        -----------
            mu: (float)
                Chemical Potential in eV.
            T: (float, default is 0.001)
                Temperature in Kelvin degrees.
        Returns:
        --------
            occup: (np.ndarray, shape (norb, nspin))
                Number occupations on each orbital/spin.
        """
        bands = self.red_bands_grid
        eivecs = self.red_eivecs_grid
        nk = np.shape(bands)[0]
        nF = bzu.fermi_dist(bands, mu, T=T)
        occups_k = np.einsum("kqb,kqbos -> kqos", nF, np.abs(eivecs)**2)
        occups = np.sum(occups_k, axis=(0, 1)) / (nk**2)
        return occups

    def occup_upto_chem_pot(self, mu, T=1e-3):
        """
        Calculates the occupation number for a given chemical potential,
        and returns n - self.Ne. This function is used to calculate the
        chemical potential for a given number of electorns

        in the unit cell.
        Parameters:
        -----------
            mu: (float)
                Chemical potential in eV.
            T: (float)
                Temperature in Kelvin degrees.
        Returns:
        --------
            N: (int)
                n-self.Ne, Where n is the occupation and self.Ne
                the number of electrons in the unit cell.
        """
        occups = self.occupations_per_orbital(mu, T=T)
        n = np.sum(occups)
        return n - self.Ne

    def chemical_potential(self, mu_min=None, mu_max=None, T=1e-3, tol=1e-3):
        """
        Calculates the Chemical potential.
        """
        ansatz_auto = (mu_min is None) or (mu_max is None)
        if ansatz_auto:
            eivals = self.model.solve_one([0, 0])
            mu_min = eivals[0]
            mu_max = eivals[-1]
        try:
            mu = opt.brentq(self.occup_upto_chem_pot,
                            mu_min, mu_max, xtol=tol)
        except ValueError:
            mu_min = eivals[0] - 10
            mu_max = eivals[-1] + 10
            mssg = "Exception: Chemical potencial calculated between"
            mssg += str(mu_min) + " and " + str(mu_max) + "."
            print(mssg)
            mu = opt.brentq(self.occup_upto_chem_pot,
                            mu_min, mu_max, xtol=tol)
        return mu

    def set_fermi_lvl(self, tol=1e-4, nk=100):
        """
        Set the fermi level in atribute self.Ef

        Parameters:
        -----------
            tol: (float, default is 1e-4)
                Tolerance in energy.
            nk: (int, default is 100)
        """
        self.create_bands_grid_red_coord(nk=nk, endpoint=False)
        self.Ef = self.chemical_potential(tol=tol)

    def velocity_operator(self, kpt):
        """
        """
        kpt = np.array(kpt)
        norb, ns = len(self.orb), self.nspin
        v = np.zeros((2, norb, ns, norb, ns), dtype="complex")
        for hop in self.hoppings:
            n1, n2, i, j, h = hop
            R = np.array([n1, n2])
            ri = np.array(self.orb[i])[0:2]
            rj = np.array(self.orb[j])[0:2]
            dr = R + rj - ri
            r_real = dr[0] * self.lat[0] + dr[1]*self.lat[1]
            kr = 2 * np.pi * kpt @ dr
            exp = np.exp(1j * kr)
            h_matrix = bzu.pauli_vector(h)*exp
            for x in range(2):
                v[x, i, :, j, :] += 1j * r_real[x] * h_matrix
                v[x, j, :, i, :] += -1j * r_real[x] * h_matrix.T.conjugate()
        return v


# -----------------------------------------------------------------------------
# Berry curvature and Chern's number with the velocity operator.
# -----------------------------------------------------------------------------


    def berry_curvature(self, k1, k2, n, a, b, mode="re"):
        """
        """
        # Geometry dependent factor
        # it should be calculated from the lattice vectors
        factor = 8*np.sqrt(3) * np.pi**2 / 3
        kpt = [k1, k2]
        eivals, eivecs = self.model.solve_one(kpt, eig_vectors=True)
        v = self.velocity_operator(kpt)
        vx_eig = np.einsum("nis, isjd, mjd-> nm", eivecs.conj(), v[0], eivecs)
        vy_eig = np.einsum("nis, isjd, mjd-> nm", eivecs.conj(), v[1], eivecs)
        v_eig = [vx_eig, vy_eig]
        Omega = 0
        for m in range(self.nband):
            if m != n:
                denominator = (eivals[n]-eivals[m])**2
                Omega += v_eig[a][n, m] * v_eig[b][m, n] / denominator
        # ADD REAL PART HERE !!!!!
        casting = {"re": np.real, "im": np.imag}[mode]
        return casting((Omega * 2j)) * factor

    def chern_integrand(self, k1, k2, n):
        Omega_xy = self.berry_curvature(k1, k2, n, 0, 1)
        Omega_yx = 0
        # We have included a 2 factor at the end of berry_curvature
        return (Omega_xy-Omega_yx) / (2 * np.pi)

    def chern_number(self, n):
        opts = {"epsabs": 1e-3}
        ranges = [[0, 1], [0, 1]]
        result, abserr = integ.nquad(self.chern_integrand, ranges,
                                     args=(n,), opts=opts)
        return result, abserr

    def spin_berry_curvature(self, k1, k2, n, i, a, b, mode="re"):
        """
        """
        # Geometry dependent factor
        # it should be calculated from the lattice vectors
        factor = 8*np.sqrt(3) * np.pi**2 / 3
        kpt = [k1, k2]
        eivals, eivecs = self.model.solve_one(kpt, eig_vectors=True)
        S = bzu.pauli_matrix(i) / 2
        S_eig = np.einsum("nis, st, mit-> nm", eivecs.conj(), S, eivecs)
        v = self.velocity_operator(kpt)
        vx_eig = np.einsum("nis, isjd, mjd-> nm", eivecs.conj(), v[0], eivecs)
        vy_eig = np.einsum("nis, isjd, mjd-> nm", eivecs.conj(), v[1], eivecs)
        v_eig = [vx_eig, vy_eig]
        js = 0.5 * (S_eig @ v_eig[a] + v_eig[a] @ S_eig)
        Omega = 0
        for m in range(self.nband):
            if m != n:
                denominator = (eivals[n]-eivals[m])**2
                Omega += js[n, m] * v_eig[b][m, n] / denominator
        # ADD REAL PART HERE !!!!!
        casting = {"re": np.real, "im": np.imag}[mode]
        return casting((Omega * 2j)) * factor


# -----------------------------------------------------------------------------
# Operators in regular k-grid
# -----------------------------------------------------------------------------


    def create_k_grid(self, nk):
        #self.wf_BZ = pytb.wf_array(self.model, [nk, nk])
        #self.wf_BZ.solve_on_grid([0, 0])
        self.nk = nk
        self.create_bands_grid_red_coord(
            nk=self.nk, return_eivec=True, endpoint=False)
        self.velocity_operator_grid(eig_basis=True)
        self.spin_operator_grid()
        self.spin_current_grid()

        np.save(self.path / "k-grid/bands.npy", self.red_bands_grid)
        np.save(self.path / "k-grid/eivecs.npy", self.red_eivecs_grid)
        np.save(self.path / "k-grid/velocity.npy", self.v_grid)
        np.save(self.path / "k-grid/velocity_eig.npy", self.v_grid_eig)
        np.save(self.path / "k-grid/spin_eig.npy", self.S_eig_grid)
        np.save(self.path / "k-grid/js_eig.npy", self.js_grid_eig)

    # @profile
    def load_k_grid(self):
        path = self.path / "k-grid/"
        self.red_bands_grid = np.load(path / "bands.npy")
        self.red_eivecs_grid = np.load(path / "eivecs.npy")
        self.v_grid = np.load(path / "velocity.npy")
        self.v_grid_eig = np.load(path / "velocity_eig.npy")
        self.nk = np.shape(self.red_eivecs_grid)[0]
        self.S_eig_grid = np.load(path / "spin_eig.npy")
        self.js_grid_eig = np.load(path / "js_eig.npy")

    def spin_operator_grid(self):
        eivecs = self.red_eivecs_grid
        subscripts = "kqnis, st, kqmit-> kqnm"
        nk = self.nk
        nband = self.nband
        shape = (3, nk, nk, nband, nband)
        S_eig = np.zeros(shape, dtype="complex")
        for i in range(3):
            S = bzu.pauli_matrix(i) / 2
            S_eig[i] = np.einsum(subscripts, eivecs.conj(), S, eivecs)
        self.S_eig_grid = S_eig

    def velocity_operator_grid(self, eig_basis=False):
        """
        Evaluates the velocity operator in the basis
        of eigenstates, on a grid in the 1BZ.

        """
        nk = self.nk
        norb = len(self.orb)
        shape = (2, nk, nk, norb, 2, norb, 2)
        v_grid = np.zeros(shape, dtype="complex")
        for i in range(nk):
            for j in range(nk):
                kpt = [i/nk, j/nk]
                v_k = self.velocity_operator(kpt)
                v_grid[0, i, j] = v_k[0]
                v_grid[1, i, j] = v_k[1]
        self.v_grid = v_grid
        if eig_basis:
            eivecs = self.red_eivecs_grid
            v_eig = np.einsum("kqnis, akqisjd, kqmjd-> akqnm",
                              eivecs.conj(), v_grid, eivecs)
            self.v_grid_eig = v_eig

    def spin_current_grid(self):
        nk = self.nk
        nband = self.nband
        shape = (3, self.dim_k, nk, nk, nband, nband)
        js_grid = np.zeros(shape, dtype="complex")
        for i in range(3):
            for a in range(self.dim_k):
                S_i = self.S_eig_grid[i]
                v_a = self.v_grid_eig[a]
                js = 0.5 * np.einsum("kqnm, kqmo->kqno", S_i, v_a)
                js += 0.5 * np.einsum("kqnm, kqmo->kqno", v_a, S_i)
                js_grid[i, a] = js
        self.js_grid_eig = js_grid

    def conductivity_grid_old(self, mode, component, extra_arg=()):
        k = np.linspace(0, 1, num=self.nk, endpoint=False)
        dk = k[1] - k[0]
        other_args = (self.Ef, *component, *extra_arg)
        integrator = {
            "s_odd": lr.spin_conductivity_k,
            "s_even": lr.spin_conductivity_k_even,
            "c_odd": lr.charge_conductivity_k,
            "c_even": lr.charge_conductivity_k_even}[mode]
        integ = 0

        for i1 in range(self.nk):
            for i2 in range(self.nk):
                eivals = self.red_bands_grid[i1, i2]
                eivecs = self.red_eivecs_grid[i1, i2]
                velocity = self.v_grid[:, i1, i2]
                args_k = (eivals, eivecs, velocity)
                integ += integrator(*args_k, *other_args)

        return integ * dk**2

    # @profile
    def conductivity_grid(self, mode, component, extra_arg=()):
        other_args = (self.Ef, *component, *extra_arg)
        integrator = {
            "s_odd": lr.spin_conductivity_k_odd_upg,
            "s_even": lr.spin_conductivity_k_even_upg,
            "c_odd": lr.charge_conductivity_k_odd_upg,
            "c_even": lr.charge_conductivity_k_even_upg
        }[mode]
        op_mode = mode[0]
        args_k = {
            "s": (
                self.red_bands_grid,
                self.v_grid_eig,
                self.js_grid_eig),
            "c": (
                self.red_bands_grid,
                self.v_grid_eig
            )}[op_mode]
        integral = integrator(*args_k, *other_args)
        return integral


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Methods that will be deprecated soon
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


    def gradient_kred_Hamiltonian(self, kpt):
        """
        Calculate the k-gradient of the Hamiltonian
        in reduced coordinates.
        Parameters:
        -----------
            kpt: (list of floats)
                Reduced coordinates of k-point
        Returns:
        --------
            Grad_H: (np.ndarray, shape (dim_k, n_orb, nspin, norb,nspin)
                if nspin=1, and (dim_k, n_orb, norb) if nspin=2)
                Gradient of the Hamiltonian, computed as:
                Grad_H[0,:,:,:,:] = (H(k1+eps, k2,..)-H(k1,k2,...))/eps
                Grad_H[1,:,:,:,:] = (H(k1, k2+eps,..)-H(k1,k2,...))/eps
                Here k1, k2,... are reduced coordinates of k,
                in such a way:
                    k = Sum_i ki*bi // (bi are reciprocal lattice vectors)
        """
        eps = 1e-9
        norb = len(self.orb)
        nspin = self.nspin
        if nspin == 1:
            shape_grad = (self.dim_k, norb, norb)
        else:
            shape_grad = (self.dim_k, norb, nspin, norb, nspin)
        grad_H = np.zeros(shape_grad, dtype="complex")
        H0 = self.model._gen_ham(k_input=kpt)
        for i in range(self.dim_k):
            dk = np.zeros(self.dim_k)
            dk[i] += eps
            kpt_i = np.array(kpt) + dk
            H_i = self.model._gen_ham(k_input=kpt_i)
            grad_H[i] = (H_i - H0) / eps
        return grad_H

    def velocity_operator_old(self, kpt):
        """
        Calculate the velocity operator as the k-gradient
        of the Hamiltonian.

        Note: The result should be multiplicated by
        a0/ hbar to recover the physical units.
        """
        grad_H = self.gradient_kred_Hamiltonian(kpt)
        M = bzu.cartesian_to_reduced_k_matrix(self.rlat[0], self.rlat[1])
        v = np.zeros_like(grad_H)
        v[0] = grad_H[0] * M[0, 0] + grad_H[1] * M[1, 0]
        v[1] = grad_H[0] * M[0, 1] + grad_H[1] * M[1, 1]
        return v


if __name__ == "__main__":
    # path = pathlib.Path("tests/")
    # toy.create_hoppings_toy_model(path, 1, 0.0, 0.1)
    # Sim = Simulation_TB(path)
    # Sim.model.display()
    # # fig, ax = plt.subplots(1, 2)
    # # Sim.plot_bands(ax[0])
    # # Sim.plot_bands_2d(0, ax=ax[1])
    # # # plt.colorbar()
    # plt.show()
    print("Done")
