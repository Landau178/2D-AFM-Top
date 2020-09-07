
import numpy as np
import matplotlib.pyplot as plt
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
        dim_r: (dim_r)


    """

    def __init__(self, path):
        self.dir = path
