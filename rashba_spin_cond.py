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
mode = "c"
nproc = 8
# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------
alpha, B = 0.295, 0.1
lamb = 0.0
th, phi = np.pi/2, 0
path = amod.create_path_rashba_model("", alpha, B=B, th=th, phi=phi, lamb=lamb)
Sim_rash = amod.Rashba_model(path, alpha, B=B, th=th, phi=phi, lamb=lamb)
# -----------------------------------------------------------------------------

nk = 400
nE = 50
# np.array([20e-3, 30e-3, 40e-3, 50e-3]) # 1e-3
Gamma_arr = np.array([12.7e-3])
mode_c_list = ["odd_z", "even_z", "odd_m", "even_m"]

if mode == "s":
    for Gamma in Gamma_arr:
        print("calculating comp: 1,0,1")
        Sim_rash.spin_conductivity_vs_Ef((1, 0, 1), Gamma, nk=nk, nE=nE)
        print("calculating comp: 1,1,0")
        Sim_rash.spin_conductivity_vs_Ef((1, 1, 0), Gamma, nk=nk, nE=nE)

elif mode == "c":
    for Gamma in Gamma_arr:
        for mode_c in mode_c_list:
            mssg = "Calculating charge conductivity: {}-{}"
            opts = {"nk": nk, "nE": nE, "mode": mode_c, "nproc": nproc}
            print(mssg.forma(mode_c, "xx"))
            Sim_rash.charge_conductivity_vs_Ef((0, 0), Gamma, **opts)
            print(mssg.forma(mode_c, "yy"))
            Sim_rash.charge_conductivity_vs_Ef((1, 1), Gamma, **opts)
