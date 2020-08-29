import pythtb as pytb
import numpy as np
import matplotlib.pyplot as plt
import string as st
from mpl_toolkits.mplot3d import Axes3D
import simulation_tb as sim
import tb_utilities as tbu
import os

root = os.path.dirname(os.path.abspath(__file__)) + '/'
t = 1.0
lamb = 0.0
h = 0.0
sim_name = tbu.simulation_name(t, lamb, h) + '/'
dir = root + 'Simulations/' + sim_name
try:
    os.makedirs(dir)
except OSError:
    pass
parameters = np.array([t, lamb, h])
np.savetxt(dir + 'parameters.dat', parameters, fmt='%.3e')
Sim = sim.Simulation_TB(dir)

Sim.plot_bands_path()
Sim.eigen_functions_grid(n_step=50)
Sim.save_current_grid()
Sim.eigen_functions_grid(n_step=100)

Sim.load_grid(50)
Sim.load_grid(100)
print(np.shape(Sim.bands))
print(Sim.bands[0, 0, :])
#print(np.shape(Sim.wf[:,:]))
