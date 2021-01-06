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
# the spin conductivity tensor (odd under time-reversal).
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
options = Parser.parse_args()
i = options.i
a = options.a
b = options.b
mssg = "Calculation of the component\
     ({},{},{}) of the spin conductivity tensor.".format(i, a, b)
print(mssg)
# -----------------------------------------------------------------------------

# Initi the simulation
t, J, t2 = 1.0, 1.7, 0.0
mag_mode = "coplanar"
path = toy.init_kagome_model(t, J, t2, mag_mode)
Sim = stb.Simulation_TB(path)
Sim.set_fermi_lvl()


# Calculation of spin conductivity
gamma_arr = np.logspace(-3, 0, num=50)
nG = np.size(gamma_arr)
result = []
for g in range(nG):
    print("Iteration {}".format(g))
    print("Using Gamma={}".format(gamma_arr[g]))
    result.append(Sim.spin_conductivity(i, a, b, Gamma=gamma_arr[g]))

# Saving result
integ_result = np.zeros((nG, 3))
integ_result[:, 0] = gamma_arr
integ_result[:, 1:3] = np.array(result)
path = Sim.path / "spin_conductivity/"
toy.mk_dir(path)
name_shc = "SHC_{}{}{}.npy".format(i, a, b)
np.save(path / name_shc, integ_result)
