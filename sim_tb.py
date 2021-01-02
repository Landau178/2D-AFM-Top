
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import linalg as la
from scipy import optimize as opt
import pathlib

import pythtb as pytb

import bz_utilities as bzu
import toy_models as toy


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
        self.read_config_file()
        self.init_folders()
        self.model = pytb.tb_model(self.dim_k, self.dim_r,
                                   lat=self.lat, orb=self.orb, nspin=self.nspin)
        for hop_file in self.hop_files:
            self.read_hoppings(name=hop_file)

    def init_folders(self):
        """
        Create some folders in the parent directory.
        """
        toy.mk_dir(self.path / "bands/")

    def save_config(self):
        """
        Just for testing
        """
        a1 = [1, 0, 0]
        a2 = [-1/2, np.sqrt(3)/2, 0]
        a3 = [0, 0, 2/3]
        lat = [a1, a2, a3]
        orb = [[1/3, 1/6, 0],  # spin up
               [5/6, 1/6, 0],  # spin down
               [1/3, 2/3, 0],  # spin down
               [5/6, 2/3, 0]]  # spin up
        nspin = 2
        k_spoints = [[0, 0], [1/3, 1/3], [0, 1/2], [-1/3, 2/3], [0, 0]]
        k_sp_labels = ["$\\Gamma$", "$K$", "$M$", "$K'$", "$\\Gamma$"]
        config = {"dim_k": 2, "dim_r": 3, "lat": lat,
                  "orb": orb, "nspin": nspin, "Ne": 4, "k_spoints": k_spoints,
                  "k_sp_labels": k_sp_labels}
        with open(self.path / 'config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)

    def read_config_file(self):
        """
        Read the config file and set the corresponding atributes.
        """
        with open(self.path / 'config.json', 'r') as fp:
            config = json.load(fp)
        # print(config)
        self.dim_k = config["dim_k"]
        self.dim_r = config["dim_r"]
        self.lat = np.array(config["lat"])
        self.set_recip_lat()
        self.orb = config["orb"]
        self.nspin = config["nspin"]
        self.nband = len(self.orb) * self.nspin
        self.Ne = config["Ne"]
        self.k_spoints = config["k_spoints"]
        self.k_sp_labels = config["k_sp_labels"]
        self.hop_files = config["hop_files"]

    def set_recip_lat(self):
        """
        Set reciprocal vectors. Only valid for r_dim=3.
        """
        a1 = self.lat[0]
        a2 = self.lat[1]
        a3 = self.lat[2]
        vol = a1 @ (np.cross(a2, a3))
        factor = 2 * np.pi / vol
        b1 = factor * np.cross(a2, a3)
        b2 = factor * np.cross(a3, a1)
        b3 = factor * np.cross(a1, a2)
        self.rlat = np.array([b1, b2, b3])

    def read_hoppings(self, name="hoppings.dat", mode="set"):
        """
        Read the hoppings of path/hoppings.dat,
        and load them into the model.
        More info in: self.__add_hopping_from_line

        Parameters:
        -----------
            name: (str, default is "hoppings.dat")
                name of the hopping file.
            mode: (str, default is "set)
                Mode to include hopping.
                See __add_hopping_from_line

        """
        path_to_hops = self.path / name
        with open(path_to_hops, 'r') as reader:
            for line in reader:
                if line[0] == "#":
                    continue
                self.__add_hopping_from_line(line, mode=mode)

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

    def create_bands_grid_red_coord(self, nk=10, return_eivec=True):
        """
        Creates a grid og the eivengavlues and eigenvectors in a grid
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
                    eival = self.model.solve_one(k_red, eig_vectors=True)
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
            self.create_bands_grid_red_coords

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

    def velocity_operator(self, kpt):
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

    def velocity_operator_grid(self):
        """
        Evaluates the velocity operator in the basis
        of eigenstates, on a grid in the 1BZ.

        Note: Before calling this method, a grid with self.create_wf_grid
        should be initialized.
        """
        nk = self.nk
        v_grid = np.zeros((nk, nk, 2, self.nband, self.nband),
                          dtype="complex")
        for i in range(nk):
            for j in range(nk):
                kpt = [i/nk, j/nk]
                eivec = self.wf_BZ[i, j]
                v_orb = self.velocity_operator(kpt)
                vx_U = np.einsum("isjd, ejd->ise", v_orb[0], eivec)
                vy_U = np.einsum("isjd, ejd->ise", v_orb[1], eivec)
                v_x = np.einsum("mis, isn-> mn", eivec.conj(), vx_U)
                v_y = np.einsum("mis, isn-> mn", eivec.conj(), vy_U)
                v_grid[i, j, :, :, :] = np.array([v_x, v_y])
        return v_grid

    def create_wf_grid(self, nk):
        self.wf_BZ = pytb.wf_array(self.model, [nk, nk])
        self.wf_BZ.solve_on_grid([0, 0])
        self.nk = nk - 1


# -----------------------------------------------------------------------------
# Private methods for reading hopping file.
# -----------------------------------------------------------------------------


    def __add_hopping_from_line(self, line, mode="set"):
        """
        Recieves a line of the hopping text file and set the corresponding
        hopping amplitude (or onsite energy) in the tight-binding model.
        More info about the format of "line" in
        self.__extract_hopping_from_line .

        Parameters:
        -----------
            line: (str)
                String containing hopping(onsite) amplitdes.
            mode: (str)
                Mode to include the hopping in the model.
                Can be "set", "add" or "reset".
        Returns:
        --------
            None
        """
        hop, is_onsite = self.__extract_hopping_from_line(line)
        if is_onsite:
            i, h = hop
            self.model.set_onsite(h, ind_i=i, mode=mode)
        else:
            n1, n2, i, j, h = hop
            self.model.set_hop(h, i, j, ind_R=[n1, n2, 0], mode=mode)

    def __extract_hopping_from_line(self, line):
        """
        Recieves a line of the hopping text file and returns the parameters
        that characterizes the hopping/onsite term.

        Input:
        ------
            line: (str)
                Line can be ingresed in one of the following
                formats:
                    1. "n1 n2 i j hr hi"
                    2. "n1 n2 i j s0r s0i sxr sxi syr syi szr szi"
                In format 1. the hopping amplitude (or onsite term) is:
                    h = hr + 1j * hi
                In format 2, the hopping is expressed in a spin resolved basis,
                beeing  "s*r" and "s*i" the real and imaginary parts of the
                coefficients accompaining the pauli matrices s0, sx, sy, sz.

        Returns:
            hop: (tuple)
                If is_onsite is True, it contains the parameters:
                (i, h). Else, the it will contain (n1, n2, i, j, h).
                With i being the orbital index, and h the hopping
                or onsite energy. h can be a complex scalar,
                or a complex array [s0, sx, sy, sz], depending on
                the format of "line".
            is_onsite: (Bool)
                Flag, to determine if the term enconded in line is an
                onsite energy.
        """
        line = self.__reformat_line(line).split(" ")
        if len(line) == 6:
            spin_resolved = False
        elif len(line) == 12:
            spin_resolved = True
        else:
            mssg = "Each line of hoppings file must have"
            mssg += " 6 or 12 quantities separated by white spaces."
            mssg += "\nThis line have {} quantities.".format(len(line))
            raise Exception(mssg)
        n1, n2 = int(line[0]), int(line[1])
        i, j = int(line[2]), int(line[3])
        h = float(line[4]) + 1j * float(line[5])
        if spin_resolved:
            sx = float(line[6]) + 1j * float(line[7])
            sy = float(line[8]) + 1j * float(line[9])
            sz = float(line[10]) + 1j * float(line[11])
            h = [h, sx, sy, sz]
        is_onsite = n1 == 0 and n2 == 0
        is_onsite = is_onsite and i == j
        if is_onsite:
            hop = (i, h)
        else:
            hop = (n1, n2, i, j, h)
        return hop, is_onsite

    def __reformat_line(self, line):
        """
        Recieves a line of the hopings text file,
        replaces double spaces (and up to 6 consecutives spaces),
        by single white space, and removes any final whitespace,
        as the endline character
        Parameters:
        -----------
            line: (str)
                String with parameter separated by white
                spaces.
        Returns:
            formated_line: (str)
                new line, after the formating procedure.
        """
        if line[-1] == "\n":
            formated_line = line[0:-1]
        else:
            formated_line = line

        for _ in range(3):
            formated_line = formated_line.replace("  ", " ")

        while formated_line[-1] == " ":
            formated_line = formated_line[0:-1]
        while formated_line[0] == " ":
            formated_line = formated_line[1::]
        return formated_line


if __name__ == "__main__":
    # path = pathlib.Path("tests/")
    # create_hoppings_toy_model(path, 1, 0.0, 0.1)
    # Sim = Simulation_TB(path)
    # Sim.model.display()
    # # fig, ax = plt.subplots(1, 2)
    # # Sim.plot_bands(ax[0])
    # # Sim.plot_bands_2d(0, ax=ax[1])
    # # # plt.colorbar()
    # plt.show()
    print("Done")
