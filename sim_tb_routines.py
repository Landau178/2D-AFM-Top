import pathlib
import numpy as np
from scipy import interpolate as interp

import configargparse
import pythtb as pytb


import sim_tb as stb
import bz_utilities as bzu
import toy_models as toy
import plot_utils as pltu
import linear_response as lr

# -----------------------------------------------------------------------------
# This module contains routines  that calculates specific plots of the
# conductivity against some other parameter.
# ------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Routines to calculate odd conductivities vs Gamma parameter.
# ------------------------------------------------------------------------------

def odd_conductivity_vs_Gamma(Sim, mode, component, nG=100):
    """
    Calculates curves for the conductivity as function
    of the band broadening Gamma, and save the results.
    To read the data use:

    gamma, cond = load_odd_conductivity_vs_Gamma(Sim, mode, component)

    Parameters:
    -----------
        Sim: (instance of stb.Simulation_TB)
            Simulation tigh-binding object.
        mode: (str)
            "s" for spin and "c" for charge conductivities.
        component: (tuple of ints,  size 2 or 3)
            Component of the tensor.

    Returns:
    --------
        None
    """
    gamma_arr = np.logspace(-3, 0, num=nG)

    integ_result = np.zeros((nG, 2))
    for g in range(nG):
        args = (mode, component)
        Gamma = gamma_arr[g]
        integ_result[g, 0] = Gamma
        integ_result[g, 1] = Sim.conductivity_grid(*args, extra_arg=(Gamma))

    path_file = directory_odd_cond_vs_Gamma(Sim, mode, component, mkdir=True)
    # Saving result deleting any previous calculation
    np.save(path_file, integ_result)


def load_odd_conductivity_vs_Gamma(Sim, mode, component):
    """
    Load the saved curves for the conductivity as function
    of the band broadening Gamma.

    Parameters:
    -----------
        Sim: (instance of stb.Simulation_TB)
            Simulation tigh-binding object.
        mode: (str)
            "s" for spin and "c" for charge conductivities.
        component: (tuple of ints,  size 2 or 3)
            Component of the tensor.

    Returns:
    --------
        Gamma_arr: (np.ndarray)
            Array with Gamma values.
        cond_arr: (np.ndarray)
            Array with the values of the time-odd
            conductivity.
    """
    path_file = directory_odd_cond_vs_Gamma(Sim, mode, component)
    integ_result = np.load(path_file)
    Gamma_arr = integ_result[:, 0]
    cond_arr = integ_result[:, 1]
    return Gamma_arr, cond_arr


def directory_odd_cond_vs_Gamma(Sim, mode, component, mkdir=False):
    """
    Returns the path to the conductivity curve.

    Parameters:
    -----------
        Sim: (instance of stb.Simulation_TB)
            Simulation tigh-binding object.
        mode: (str)
            "s" for spin and "c" for charge conductivities.
        component: (tuple of ints,  size 2 or 3)
            Component of the tensor.
        mkdir: (bool, default is False)
            Flag to create the parent dir of the file.
    Returns:
    --------
        path_file: (pathlib.Path)
            Path to the .npy file.
    """
    path_result = {"s": Sim.path / "odd_spin_conductivity_vs_Gamma/",
                   "c": Sim.path / "odd_charge_conductivity_vs_Gamma/"}[mode]
    if mkdir:
        toy.mk_dir(path_result)
    name = {
        "s": "SHC_{}{}{}.npy",
        "c": "CHC_{}{}.npy"}[mode]
    name = name.format(*component)
    path_file = path_result / name
    return path_file
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Routines to calculate conductivity vs Fermi level
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def conductivity_vs_Ef(Sim, mode, component, extra_arg=()):
    """
    Calculates curves for the conductivity as function
    of the Fermi level, and save the results.
    To read the data use:

    Ef_arr, cond = load_conductivity_vs_Ef(Sim, mode, component,
                                            extra_arg=extra_arg)

    Parameters:
    -----------
        Sim: (instance of stb.Simulation_TB)
            Simulation tigh-binding object.
        mode: (str)
            One of the group:
            "s_odd", "c_odd", "s_even", "c_even"
        component: (tuple of ints,  size 2 or 3)
            Component of the tensor.
        extra_arg: (tuple, default is ())
            Only for mode equal to "s_odd" or "c_odd".
            It is a tuple containing Gamma parameter.

    Returns:
    --------
        None
    """
    Ef_arr = np.linspace(-4, 5, num=100)
    nE = np.size(Ef_arr)
    integ_result = np.zeros((nE, 2))
    for iE in range(nE):
        Sim.Ef = Ef_arr[iE]
        args = (mode, component)
        integ_result[iE, 0] = Ef_arr[iE]
        integ_result[iE, 1] = Sim.conductivity_grid(*args, extra_arg=extra_arg)

    path_file = directory_cond_vs_Ef(
        Sim, mode, component, extra_arg=extra_arg, mkdir=True)

    # Saving result deleting any previous calculation
    np.save(path_file, integ_result)


def load_cond_vs_Ef(Sim, mode, component, extra_arg=()):
    """
    Load the curves for the conductivity as function
    of the Fermi level.

    Parameters:
    -----------
        Sim: (instance of stb.Simulation_TB)
            Simulation tigh-binding object.
        mode: (str)
            One of the group:
            "s_odd", "c_odd", "s_even", "c_even"
        component: (tuple of ints,  size 2 or 3)
            Component of the tensor.
        extra_arg: (tuple, default is ())
            Only for mode equal to "s_odd" or "c_odd".
            It is a tuple containing Gamma parameter.

    Returns:
    --------
        Ef_arr: (np.ndarray)
            Array with values of the Fermi level.
        cond_arr: (np.ndarray)
            Array with the conductivities.
    """
    path_file = directory_cond_vs_Ef(Sim, mode, component, extra_arg=extra_arg)
    integ_result = np.load(path_file)
    Ef_arr = integ_result[:, 0]
    cond_arr = integ_result[:, 1]
    return Ef_arr, cond_arr


def directory_cond_vs_Ef(Sim, mode, component, extra_arg=(), mkdir=False):
    """
    Returns the path to the save data.

    Parameters:
    -----------
        Sim: (instance of stb.Simulation_TB)
            Simulation tigh-binding object.
        mode: (str)
            One of the group:
            "s_odd", "c_odd", "s_even", "c_even"
        component: (tuple of ints,  size 2 or 3)
            Component of the tensor.
        extra_arg: (tuple, default is ())
            Only for mode equal to "s_odd" or "c_odd".
            It is a tuple containing Gamma parameter.
        mkdir: (bool, default is False)
            Flag to create the parent directory.

    Returns:
    --------
        path_file: (pathlib.Path)
            Path to the *.npy file.
    """
    time_rev = mode.split("_")[1]
    path_result = {
        "s": Sim.path / "{}_spin_conductivity_vs_Ef".format(time_rev),
        "c": Sim.path / "{}_charge_conductivity_vs_Ef".format(time_rev)}[mode[0]]
    if mkdir:
        toy.mk_dir(path_result)
    name = {
        "s": "SHC_{}{}{}",
        "c": "CHC_{}{}"}[mode[0]]
    if time_rev == "odd":
        name_end = "G={}.npy".format(toy.float2str(*extra_arg))
    else:
        name_end = ".npy"
    name = name.format(*component) + name_end
    return path_result / name
