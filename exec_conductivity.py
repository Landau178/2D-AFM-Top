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
import sim_tb_routines as stbr

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

Parser.add("--t2", default=0.2, action="store",
           type=float, help="SOC parameter")

Parser.add("--load", action="store_true",
           help="Flag to determine if load the k-grid\
            instead of calculating it.")

Parser.add("--mag_mode", default="coplanar", action="store",
           type=str, help="mag_mode parameter for  Kagome lattice.")

Parser.add("--nk", default=500, action="store",
           type=int, help="Size of the k-grid")

options = Parser.parse_args()
i = options.i
a = options.a
b = options.b
t2 = options.t2
mag_mode = options.mag_mode
nk = options.nk

time_rev = options.tr
mode = options.mode
concat = options.concat


calc_kgrid = not(options.load)

component = {
    "s": (i, a, b),
    "c": (a, b)}[mode]

cond_mode = mode + "_" + time_rev

# Workflow control
cond_vs_Ef = True
cond_vs_Gamma = False

# only osed if cond_vs_Gamma and time_rev=="odd"
Gamma = 20e-3

mssg_s = "Calculation of the component\
     ({},{},{}) of the spin conductivity tensor.".format(i, a, b)
mssg_c = "Calculation of the component\
    ({},{}) of the charge conductivity tensor.".format(a, b)
mssg = {"s": mssg_s, "c": mssg_c}[mode]
print(mssg)
# -----------------------------------------------------------------------------


# Init the simulation
t, J = 1.0, 1.7
path = toy.init_kagome_model(t, J, t2, mag_mode)
Sim = stb.Simulation_TB(path)
# Sim.set_fermi_lvl()


# Calculation of time-(even/odd) (spin/charge) conductivity


if calc_kgrid:
    Sim.create_k_grid(nk)
else:
    Sim.load_k_grid()

if cond_vs_Gamma:
    Sim.Ef = -2.7535904054700566
    stbr.odd_conductivity_vs_Gamma(Sim, cond_mode, component)



if cond_vs_Ef:
    extra_arg = {"odd": (Gamma,), "even": ()}[time_rev]
    stbr.conductivity_vs_Ef(
        Sim, cond_mode, component, extra_arg=extra_arg)


# Concatenate the new result to an existing one
if concat:
    old_integ_result = np.load(path_result / name)
    integ_result = np.concatenate((old_integ_result, integ_result), axis=0)


print("")
print("")
