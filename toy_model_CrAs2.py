
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import linalg as la
import pathlib

import pythtb as pytb

import bz_utilities as bzu

ROOT_DIR = "/home/orion178/Escritorio/Investigacion/2D-AFM-Top/"


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
    if x > 0:
        length = decimals + lenght_non_decimal + 1
    else:
        length = decimals + lenght_non_decimal + 2
    str_x = str(round(x, decimals)).ljust(length, "0")
    return str_x


def create_path_toy_model(t, alpha, beta, h):
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
    path = ROOT_DIR + "saved_simulations/toy_model/"+str_parameters
    mk_dir(path)
    return pathlib.Path(path)


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


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass
