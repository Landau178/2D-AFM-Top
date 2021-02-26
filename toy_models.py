
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import linalg as la
import pathlib

import pythtb as pytb

import bz_utilities as bzu


# ------------------------------------------------------------------------------
# References
# [1] Yang Zhang et al 2018 New J. Phys.20 073028
# ------------------------------------------------------------------------------

# IMPORTANT:
# On this path the simulations are saved, modify this line
# before using this module.
#ROOT_DIR = "/home/orion178/Escritorio/Investigacion/2D-AFM-Top/"
ROOT_DIR = pathlib.Path.home() / "Desktop/Projects_Rodrigo/Project_linear_response/"


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def float2str(x, decimals=3):
    """
    Takes a float, and returns a string justified with an
    specific number of decimals.
    Examples:
    --------
        float2str(-11.32, decimals=3)
        >> "-11.320"
        float2str(11.3234, decimals=3)
        >> "11.323"
    Parameters:
    -----------
        x: (float)
            float to transform.
        decimals: (int, default is 3)
            Number of decimals to include in str expression.

    Returns:
    --------
        str_x: (str)
            String expression of rounded float x.


    """
    lenght_non_decimal = len(str(int(np.abs(x))))
    if x >= 0:
        length = decimals + lenght_non_decimal + 1
    else:
        length = decimals + lenght_non_decimal + 2
    str_x = str(round(x, decimals)).ljust(length, "0")
    return str_x


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


def spherical2cart(r, theta, phi):
    """
    Conversion of shperical to cartesian coordinates.
    Parameters:
    -----------
        r: (float)
            radial coordinate.
        theta: (float)
            Polar angle in  [0, pi].
        phi: (float)
            Azimuthal angle in [0, 2pi].
    Returns:
    --------
        r_cart: (list of floats)
            Cartesian coordinates:
            [x, y, z]
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    r_cart = [x, y, z]
    return r_cart

# -----------------------------------------------------------------------------
# CrAs2 toy model
# -----------------------------------------------------------------------------


def create_path_toy_model(t, alpha, beta, h, folder=""):
    """
    Receives the parameters of the toy model, and creates
    a folder for the simulation on:
    ./saved_simulations/tou_model/str_parameters
    with str_parameters being "t={}_alpha={}_beta={}_h={}"
    with the respective values of the parameters.
    Parameters:
        t: (float)
            Hopping NN.
        alpha: (float)
            SOC parameter on sigma_z
        beta: (float)
            SOC parameter on the link vector.
        h: (float)
            Staggered external magnetic field on z.
    Returns:
    --------
        path: (ppathlib.Path)
                Path to simulation.
    """
    str_t = float2str(t)
    str_alpha = float2str(alpha)
    str_beta = float2str(beta)
    str_h = float2str(h)
    str_parameters = "t={}_alpha={}_beta={}_h={}/".format(
        str_t, str_alpha, str_beta, str_h)
    path = ROOT_DIR / "saved_simulations/toy_model/{}".format(folder)
    path = path / str_parameters
    mk_dir(path)
    return pathlib.Path(path)


def create_hoppings_toy_model(path, t, alpha, beta, h):
    """
    Create a text file with the hoppings of the toy model
    of CrAs2.

    """
    a1 = np.array([1, 0, 0])
    a2 = np.array([-0.5, np.sqrt(3)/2, 0])
    l1 = 0.5 * a1
    l2 = 0.5 * (a1 + a2)
    l3 = 0.5 * a2
    # alpha = np.sqrt(3)/24 * lamb
    # beta = lamb / 36
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


# -----------------------------------------------------------------------------
# Kagome lattice
# -----------------------------------------------------------------------------

def init_kagome_model(t, J, t2, mag_mode, folder=""):
    """
    Init the folder, configfile and hoppings file for a simulation
    fo the Kagome lattice.

    Parameters:
    ----------
        t, J, t2, mag_mode: (float, float, str)
            Parameters of model, see kagome_hoppings.
        folder: (str, default is "")
            Optional parameter of create_path_kafome_model.

    Returns:
    --------
        path: (pathlib.Path)
            Path of the simulation, which contains all neccesary files
            to init sim_tb.Simulation_TB.
    """
    path = create_path_kagome_model(t, J, t2, mag_mode, folder=folder)
    kagome_config(path)
    kagome_hoppings(path, t, J, t2, mag_mode)
    return path


def create_path_kagome_model(t, J, t2, mag_mode, folder=""):
    """
    Parameters:
    -----------
        t, J, t2, mag_mode: (float, float, str)
            Parameters of model, see kagome_hoppings.
        folder: (str, default is "")
            Location of the simulation folder is:
            pathlib.Path(ROOT_DIR / folder)

    Returns:
    --------
        path: (pathlib.Path)
            Path to simulation folder.
    """
    str_t = float2str(t)
    str_J = float2str(J)
    str_t2 = float2str(t2)

    str_parameters = "t={}_J={}_t2={}_mag={}/".format(
        str_t, str_J, str_t2, mag_mode)
    path = ROOT_DIR / "saved_simulations/toy_model/kagome/{}".format(folder)
    path = path / str_parameters
    mk_dir(path)
    return pathlib.Path(path)


def kagome_config(path):
    """
    Save the config file for the kagome lattice in:
        path/config.json

    Parameters:
    -----------
        path: (pathlib.Path)
            Path to the simulation folder.
    Returns:
    ---------
        None
    """
    a1 = [1, 0, 0]
    a2 = [1/2, np.sqrt(3)/2, 0]
    a3 = [0, 0, 1]
    lat = [a1, a2, a3]
    orb = [[0, 0, 0],
           [1/2, 0, 0],
           [0, 1/2, 0],
           ]
    nspin = 2
    k_spoints = [[0, 0], [1/2, 1/2], [1/3, 2/3], [0, 1]]
    k_sp_labels = ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"]
    hop_files = ["hoppings_kagome.dat"]
    config = {"dim_k": 2, "dim_r": 3, "lat": lat,
              "orb": orb, "nspin": nspin, "Ne": 1, "k_spoints": k_spoints,
              "k_sp_labels": k_sp_labels, "hop_files": hop_files}
    with open(path / 'config.json', 'w') as fp:
        json.dump(config, fp, sort_keys=True, indent=4)


def kagome_hoppings(path, t, J, t2, mag_mode):
    """
    Create the hoppping file for the kagome lattice model,
    acording to [1]:

    Parameters:
    -----------
        path: (pathlib.Path)
            path to the simulation folder.
        t: (float)
            NN hopping parameter.
        J: (float)
            Hund's coupling between ininerant electrons
            and local moments.
        t2:(float)
            Strength of the SOC.
        mag_mode: (str)
            String that labels tha magnetization texture.
            See magnetic_texture_kagome.
    """
    mag_angles = magnetic_texture_kagome(mag_mode)
    n = np.array([
        spherical2cart(1, mag_angles[0, 0], mag_angles[0, 1]),
        spherical2cart(1, mag_angles[1, 0], mag_angles[1, 1]),
        spherical2cart(1, mag_angles[2, 0], mag_angles[2, 1])
    ])
    n1 = [0, 1]
    n2 = [-np.sqrt(3)/2, -1/2]
    n3 = [np.sqrt(3)/2, -1/2]

    hoppings = [
        [0, 0, 0, 1, t, 0, 0, t2*n1[0], 0, t2*n1[1], 0, 0],
        [0, 0, 0, 2, t, 0, 0, -t2*n3[0], 0, -t2*n3[1], 0, 0],
        [-1, 0, 0, 1, t, 0, 0, t2*n1[0], 0, t2*n1[1], 0, 0],
        [0, -1, 0, 2, t, 0, 0, -t2*n3[0], 0, -t2*n3[1], 0, 0],
        [0, 0, 1, 2, t, 0, 0, t2*n2[0], 0, t2*n2[1], 0, 0],
        [1, -1, 1, 2, t, 0, 0, t2*n2[0], 0, t2*n2[1], 0, 0]
    ]
    exchange = [
        [0, 0, 0, 0, 0, 0, -J*n[0, 0], 0, -J*n[0, 1], 0, -J*n[0, 2], 0],
        [0, 0, 1, 1, 0, 0, -J*n[1, 0], 0, -J*n[1, 1], 0, -J*n[1, 2], 0],
        [0, 0, 2, 2, 0, 0, -J*n[2, 0], 0, -J*n[2, 1], 0, -J*n[2, 2], 0]
    ]
    hopps = [*hoppings, *exchange]

    name = path / "hoppings_kagome.dat"
    with open(name, 'w') as writer:
        writer.write("# Each line is a hopping, with format:\n")
        writer.write("# n1 n2 i j s0r s0i sxr sxi syr syi szr szi\n")
        for hop in hopps:
            line = list_to_str(hop)
            writer.write(line + "\n")


def magnetic_texture_kagome(mode):
    """
    Returns the magnetization texture in the Kagome
    lattice, according to the modes:
        1. coplanar
        2. coplanar_b 2-fold-y spin rotation of 1.

    Parameters:
    -----------
        mode (str)
            String labeling the magnetic texture.
    Returns:
    --------
        n_arr: (np.ndarray, of shape (3,2))
            Array containing the magnetic texture:
                n[i,:] = [theta, phi]
                polar (theta) and azimuthal (phi)
                angles of site i.

    """
    coplanar_texture = np.array([
        [np.pi/2, np.pi*(1/2 + 2/3)],
        [np.pi/2, np.pi*(1/2 + 4/3)],
        [np.pi/2, np.pi/2]
    ])
    coplanar_b = np.array([
        [np.pi/2, np.pi*(1/2 + 4/3)],
        [np.pi/2, np.pi*(1/2 + 2/3)],
        [np.pi/2, np.pi/2]
    ])
    texture_dict = {
        "coplanar": coplanar_texture,
        "coplanar_b": coplanar_b
    }
    return texture_dict[mode]
