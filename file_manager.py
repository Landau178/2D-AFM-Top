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


class Sim_TB_file_manager():
    """
    This class is in charge of rerading and manipulating the
    config and input files.
    """

    def __init__(self, Sim):
        """
        Parameters:
        -----------
            Sim: (instance of sim_tb.Simulation_TB)
        """
        self.Sim = Sim
        self.path = Sim.path
        self._read_config_file()
        self.init_folders()

    def set_Hamiltonian(self):
        self.Sim.hoppings = []
        for hop_file in self.Sim.hop_files:
            self._read_hoppings(name=hop_file)

    def init_folders(self):
        """
        Create some folders in the parent directory.
        """
        toy.mk_dir(self.path / "bands/")
        toy.mk_dir(self.path / "k-grid/")

    def _read_hoppings(self, name="hoppings.dat", mode="set"):
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

    def _read_config_file(self):
        """
        Read the config file and set the corresponding atributes.
        """
        with open(self.path / 'config.json', 'r') as fp:
            config = json.load(fp)
        # print(config)
        self.Sim.dim_k = config["dim_k"]
        self.Sim.dim_r = config["dim_r"]
        self.Sim.lat = np.array(config["lat"])
        self.__set_recip_lat()
        self.Sim.orb = config["orb"]
        self.Sim.nspin = config["nspin"]
        self.Sim.nband = len(self.Sim.orb) * self.Sim.nspin
        self.Sim.Ne = config["Ne"]
        self.Sim.k_spoints = config["k_spoints"]
        self.Sim.k_sp_labels = config["k_sp_labels"]
        self.Sim.hop_files = config["hop_files"]

    def __set_recip_lat(self):
        """
        Set reciprocal vectors. Only valid for r_dim=3.
        """
        a1 = self.Sim.lat[0]
        a2 = self.Sim.lat[1]
        a3 = self.Sim.lat[2]
        vol = a1 @ (np.cross(a2, a3))
        factor = 2 * np.pi / vol
        b1 = factor * np.cross(a2, a3)
        b2 = factor * np.cross(a3, a1)
        b3 = factor * np.cross(a1, a2)
        self.Sim.rlat = np.array([b1, b2, b3])

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
            self.Sim.model.set_onsite(h, ind_i=i, mode=mode)
        else:
            n1, n2, i, j, h = hop
            self.Sim.model.set_hop(h, i, j, ind_R=[n1, n2, 0], mode=mode)

    def __extract_hopping_from_line(self, line):
        """
        Recieves a line of the hopping text file and returns the parameters
        that characterizes the hopping/onsite term.

        Note: After last update, the function also save the hoppings in atribute
        self.hoppings: (list)
            each element in format [n1, n2, i, j, hop_sim]
            and hop_sim = [t0, tx, ty, tz]
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
        # hop_sim will be used to sabe hopping in self.hoppings
        hop_sim = h if spin_resolved else [h, 0, 0, 0]
        if is_onsite:
            hop = (i, h)
            self.Sim.hoppings.append([0, 0, i, i, hop_sim])
        else:
            hop = (n1, n2, i, j, h)
            self.Sim.hoppings.append([n1, n2, i, j, hop_sim])
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
