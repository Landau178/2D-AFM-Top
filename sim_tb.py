
import json
import matplotlib.pyplot as plt
import numpy as np
import pathlib

import pythtb as pytb


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
    """

    def __init__(self, path):
        self.path = pathlib.Path(path).absolute()
        self.save_config()
        self.read_config_file()
        self.model = pytb.tb_model(self.dim_k, self.dim_r,
                                   lat=self.lat, orb=self.orb, nspin=2)
        self.read_hoppings()

    def save_config(self):
        """
        Just for testing
        """
        a1 = [1, 0, 0]
        a2 = [-1/2, np.sqrt(3)/2, 0]
        a3 = [0, 0, 2/3]
        # l1 = tbu.vector_SU2(0.5 * a1)
        # l2 = tbu.vector_SU2(0.5 * a1 + 0.5 * a2)
        # l3 = tbu.vector_SU2(0.5 * a2)
        lat = [a1, a2, a3]
        orb = [[1/3, 1/6, 0],  # spin up
               [5/6, 1/6, 0],  # spin down
               [1/3, 2/3, 0],  # spin down
               [5/6, 2/3, 0]]  # spin up
        config = {"dim_k": 2, "dim_r": 3, "lat": lat, "orb": orb}
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
        self.lat = config["lat"]
        self.orb = config["orb"]

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


def list_to_str(foo_list):
    """
    Takes a list, and transform it to a string,
    with elements separated by spaces.
    Parameters:
    -----------
        foo_list: (list)
            List with any content.
            Elements must be convertible to str.
    Returns:
        line: (str)
            if list is [1, 2, 3], then line is:
            "1 2 3 "
    """
    n = len(foo_list)
    line = ""
    for i in range(n):
        line += str(foo_list[i]) + " "
    return line


def create_hoppings_toy_model(path, t, lamb, h):
    """
    Create a text file with the hoppings of the toy model
    of CrAs2.
    Remember to include h !
    """
    a1 = np.array([1, 0, 0])
    a2 = np.array([-0.5, np.sqrt(3)/2, 0])
    l1 = 0.5 * a1
    l2 = 0.5 * (a1 + a2)
    l3 = 0.5 * a2
    alpha = np.sqrt(3)/24 * lamb
    beta = lamb / 36
    onsite = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, h, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, -h, 0],
        [0, 0, 2, 2, 0, 0, 0, 0, 0, 0, -h, 0],
        [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, h, 0]
    ]
    hopps_from0 = [
        [0, 0, 0, 1, t, 0, 0, l1[0]*beta, 0, l1[1]*beta, 0, alpha],
        [0, 0, 0, 2, t, 0, 0, l3[0]*beta, 0, l3[1]*beta, 0, alpha],
        [0, 0, 0, 3, t, 0, 0, l2[0]*beta, 0, l2[1]*beta, 0, -alpha],
        [-1, 0, 0, 1, t, 0, 0, -l1[0]*beta, 0, -l1[1]*beta, 0, -alpha],
        [-1, -1, 0, 3, t, 0, 0, -l2[0]*beta, 0, -l2[1]*beta, 0, alpha],
        [0, -1, 0, 2, t, 0, 0, -l3[0]*beta, 0, -l3[1]*beta, 0, -alpha],
    ]
    hopps_from1 = [
        [0, 0, 1, 3, t, 0, 0, l3[0]*beta, 0, l3[1]*beta, 0, alpha],
        [1, 0, 1, 2, t, 0, 0, l2[0]*beta, 0, l2[1]*beta, 0, -alpha],
        [0, -1, 1, 3, t, 0, 0, -l3[0]*beta, 0, -l3[1]*beta, 0, -alpha],
        [0, -1, 1, 2, t, 0, 0, -l2[0]*beta, 0, -l2[1]*beta, 0, alpha]
    ]
    hopps_from2 = [
        [-1, 0, 2, 3, t, 0, 0, -l1[0]*beta, 0, -l1[1]*beta, 0, -alpha],
        [0, 0, 2, 3, t, 0, 0, l1[0]*beta, 0, l1[1]*beta, 0, alpha]
    ]
    hopps = [*onsite, *hopps_from0, *hopps_from1, *hopps_from2]

    name = path / "hoppings_toy_model.dat"
    with open(name, 'w') as writer:
        writer.write("# Each line is a hopping, with format:\n")
        writer.write("# n1 n2 i j s0r s0i sxr sxi syr syi szr szi\n")
        for hop in hopps:
            line = list_to_str(hop)
            writer.write(line + "\n")


if __name__ == "__main__":
    path = pathlib.Path("tests/")
    create_hoppings_toy_model(path, 1, 1, 1)
    #Sim = Simulation_TB(path)
