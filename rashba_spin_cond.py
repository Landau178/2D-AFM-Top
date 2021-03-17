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
# This script calculates and save the nonzero components of the spin/charge
# conductivity in the Rashba Hamiltonian.
# -----------------------------------------------------------------------------


def main():
    # -----------------------------------------------------------------------------
    # Routine parameters
    # -----------------------------------------------------------------------------
    mode = "s"
    nproc = 6

    nk = 400
    nE = 50

    # -----------------------------------------------------------------------------
    # Simulation parameters
    # -----------------------------------------------------------------------------
    alpha, B = 0.295, 0.1
    lamb = 0.0
    th, phi = np.pi/2, 0
    path = amod.create_path_rashba_model(
        "", alpha, B=B, th=th, phi=phi, lamb=lamb)
    Sim_rash = amod.Rashba_model(path, alpha, B=B, th=th, phi=phi, lamb=lamb)
    # -----------------------------------------------------------------------------
    Gamma_arr = np.array([12.7e-3])
    mode_c_list = ["odd_m"]  # ["odd_z", "even_z", "odd_m", "even_m"]
    mode_s_list = ["mook2"]  # ["zelezny", "mook"]

    if mode == "s":
        routine_spin_cond(Sim_rash, Gamma_arr, mode_s_list, nk, nE, nproc)

    elif mode == "c":
        routine_charge_cond(Sim_rash, Gamma_arr, mode_c_list, nk, nE, nproc)


# np.array([20e-3, 30e-3, 40e-3, 50e-3]) # 1e-3

def routine_spin_cond(Sim_rash, Gamma_arr, mode_s_list, nk, nE, nproc):
    for Gamma in Gamma_arr:
        for mode_s in mode_s_list:
            mssg = "\n\n\nCalculating spin conductivity: {}-{}"
            opts = {"nk": nk, "nE": nE, "mode": mode_s, "nproc": nproc}
            print(mssg.format(mode_s, "yxy"))
            Sim_rash.spin_conductivity_vs_Ef((1, 0, 1), Gamma, **opts)
            print(mssg.format(mode_s, "yyx"))
            Sim_rash.spin_conductivity_vs_Ef((1, 1, 0), Gamma, **opts)


def routine_charge_cond(Sim_rash, Gamma_arr, mode_c_list, nk, nE, nproc):
    for Gamma in Gamma_arr:
        for mode_c in mode_c_list:
            mssg = "\n\n\nCalculating charge conductivity: {}-{}"
            opts = {"nk": nk, "nE": nE, "mode": mode_c, "nproc": nproc}
            print(mssg.format(mode_c, "xx"))
            Sim_rash.charge_conductivity_vs_Ef((0, 0), Gamma, **opts)
            print(mssg.format(mode_c, "yy"))
            Sim_rash.charge_conductivity_vs_Ef((1, 1), Gamma, **opts)


if __name__ == "__main__":
    main()
