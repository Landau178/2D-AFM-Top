import pathlib

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import integrate as integ
import scipy.linalg as la


import bz_utilities as bzu
import linear_response as lr
import toy_models as toy
import plot_utils as pltu
import analytical_models as amod

# -----------------------------------------------------------------------------
# This script calculates and save the nonzero components of the spin
# conductivity in the Rashba Hamiltonian.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------
alpha, B = 1.69, 0.07
lamb = 0.0
th, phi = np.pi/2, 0
path = amod.create_path_rashba_model("", alpha, B=B, th=th, phi=phi, lamb=lamb)
Sim_rash = amod.Rashba_model(path, alpha, B=B, th=th, phi=phi, lamb=lamb)
# -----------------------------------------------------------------------------

nk = 400
Gamma_arr = np.array([1e-3])  # np.array([12.7e-3, 20e-3, 30e-3, 40e-3, 50e-3])


for Gamma in Gamma_arr:
    Sim_rash.spin_conductivity_vs_Ef((1, 0, 1), Gamma, nk=nk)
    Sim_rash.spin_conductivity_vs_Ef((1, 1, 0), Gamma, nk=nk)
