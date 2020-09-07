
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

    def save_config(self):
        """
        Just for testing
        """
        config = {"a": 1, "b": 2}
        with open(self.path / 'config.json', 'w') as fp:
            json.dump(config, fp)

    def read_config_file(self):
        """
        Read the config file and set the corresponding atributes.
        """
        with open(self.path / 'config.json', 'r') as fp:
            config = json.load(fp)
        print(config)


if __name__ == "__main__":
    path = pathlib.Path("tests/")
    Sim = Simulation_TB(path)
