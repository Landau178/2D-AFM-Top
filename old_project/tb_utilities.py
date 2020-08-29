import pythtb as pytb
import numpy as np
import matplotlib.pyplot as plt
import string as st
from mpl_toolkits.mplot3d import Axes3D


def vector_SU2(vector):
      sigma_x = np.array([[0, 1], [1, 0]])
      sigma_y = np.array([[0, -1j], [1j, 0]])
      sigma_z = np.array([[1, 0], [0,-1]])
      return vector[0]*sigma_x +vector[1]*sigma_y + vector[2]*sigma_z

def simulation_name(t, lamb, h):
    str_t = str(round(t, 3)).ljust(5, '0')
    str_lamb = str(round(lamb, 3)).ljust(5, '0')
    str_h = str(round(h, 3)).ljust(5, '0')
    name = 't=' + str_t + '_lambda=' + str_lamb + '_h='+str_h
    return name
