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
# This script perform calculations of different components of
# the spin and charge conductivity tensors (odd under time-reversal).
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------
Parser = configargparse.ArgParser()
Parser.add('-i', default=0, action='store',
           type=int, help="Spin polarization of current")
Parser.add('-a', default=0, action='store',
           type=int, help="Direction of the current")
Parser.add('-b', default=0, action='store',
           type=int, help="Direction of the applied electric field")
Parser.add("--mode", default="s", action="store", type=str,
           help="s for spin conductivity, c for charge conductiviy.")
Parser.add("--concat", action="store_true",
           help="Whether to concatenate result with existing calculation.")
Parser.add("--tr", default="odd", action="store", type=str,
           help="If odd, calculates time-odd spin conductivity vs Gamma\
               if even, caculates time-even conductivity vs Ef")


options = Parser.parse_args()
i = options.i
a = options.a
b = options.b
time_rev = options.tr
mode = options.mode
concat = options.concat

component = {
    "s": (i, a, b),
    "c": (a, b)}[mode]

mssg_s = "Calculation of the component\
     ({},{},{}) of the spin conductivity tensor.".format(i, a, b)
mssg_c = "Calculation of the component\
    ({},{}) of the charge conductivity tensor.".format(a, b)
mssg = {"s": mssg_s, "c": mssg_c}[mode]
print(mssg)
# -----------------------------------------------------------------------------


def odd_conductivity_vs_Gamma(Sim, mode, component):
    conductivity_func = {
        "s": Sim.spin_conductivity,
        "c": Sim.charge_conductivity}[mode]

    gamma_arr = np.concatenate(
        (np.logspace(-3, 0, num=50), np.logspace(0, 1, num=11)[1::]))
    nG = np.size(gamma_arr)

    result = []
    for g in range(nG):
        print("Iteration {}".format(g))
        print("Using Gamma={}".format(gamma_arr[g]))
        args = (*component, gamma_arr[g])
        result.append(conductivity_func(*args))

    integ_result = np.zeros((nG, 3))
    integ_result[:, 0] = gamma_arr
    integ_result[:, 1:3] = np.array(result)
    return integ_result


def even_conductivity_vs_Ef(Sim, mode, component):
    conductivity_func = {
        "s": Sim.spin_conductivity_even,
        "c": Sim.charge_conductivity_even}[mode]

    Ef_arr = np.linspace(-4, 5, num=100)
    nE = np.size(Ef_arr)
    result = []
    for iE in range(nE):
        print("Iteration {}".format(iE))
        print("Using Ef={}".format(Ef_arr[iE]))
        Sim.Ef = Ef_arr[iE]
        result.append(conductivity_func(*component))

    integ_result = np.zeros((nE, 3))
    integ_result[:, 0] = Ef_arr
    integ_result[:, 1:3] = np.array(result)
    return integ_result


# Init the simulation
t, J, t2 = 1.0, 1.7, 0.2
mag_mode = "coplanar"
path = toy.init_kagome_model(t, J, t2, mag_mode)
Sim = stb.Simulation_TB(path)
# Sim.set_fermi_lvl()


# Calculation of spin conductivity

if time_rev == "odd":
    Sim.Ef = -2.7535904054700566
    integ_result = odd_conductivity_vs_Gamma(Sim, mode, component)

    path_result = {"s": Sim.path / "spin_conductivity/",
                   "c": Sim.path / "charge_conductivity/"}[mode]
    toy.mk_dir(path_result)


elif time_rev == "even":
    integ_result = even_conductivity_vs_Ef(Sim, mode, component)
    path_result = {
        "s": Sim.path / "even_spin_conductivity_vs_Ef",
        "c": Sim.path / "even_charge_conductivity_vs_Ef"}[mode]
    toy.mk_dir(path_result)


name = {
    "s": "SHC_{}{}{}.npy".format(i, a, b),
    "c": "CHC_{}{}.npy".format(a, b)}[mode]

# Concatenate the new result to an existing one
if concat:
    old_integ_result = np.load(path_result / name)
    integ_result = np.concatenate((old_integ_result, integ_result), axis=0)

# Saving result deleting any previous calculation
np.save(path_result / name, integ_result)


print("")
print("")
