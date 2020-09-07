
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
        print(config)
        self.dim_k = config["dim_k"]
        self.dim_r = config["dim_r"]
        self.lat = config["lat"]
        self.orb = config["orb"]

    def read_hoppings(self):
        """
        Read the hoppings of path/hoppings.dat,
        and load them into the model.
        """
        path_to_hops = self.path / "hoppings.dat"


if __name__ == "__main__":
    path = pathlib.Path("tests/")
    Sim = Simulation_TB(path)
