import numpy as np

import pythtb as pytb

import bz_utilities as bzu
import toy_models as toy
import sim_tb as stb

# References
# ----------
# [1] Spin-Polarized Current in Noncollinear Antiferromagnets,
#     PRL119,187204 (2017)
#     DOI: 10.1103/PhysRevLett.119.187204


def Xi_I(velocity, bands, E_F, Gamma=1e-8, alpha=0):
    """
    Calculate Xi_I according to [1].

    Parameters:
    -----------
        Velocity: (np.ndarray, shape (nk,nk, nband,nband))
            Velocity operator in the basis of eigenstates.
            Evaluated on a grid in the 1BZ. (reduced coordinates)
            [k1,k2, band1, band2]
        bands:(np.ndarray, shape (nk, nk, nband))
            Grid of eigenvalues ordered as [k1, k2, band]
        E_F: (float)
            Fermi level.
        Gamma: (float)
            Band broadening, product of disorder.
        alpha: (float)
            Angle of the electric field.
            E = (cos(alpha), sin(alpha))
    Returns:
    --------
        Xi
    """
    nk = np.shape(bands)[0]
    Ex, Ey = np.cos(alpha), np.sin(alpha)
    vE = velocity[0] * Ex + velocity[1] * Ey
    gE = 1. / ((E_F-bands)**2 + Gamma**2)
    XiI_k = np.einsum("ijn, ijnm, ijm, ijmn->ij", gE, velocity[0], gE, vE)
    Xi = np.sum(XiI_k) / (nk**2)
    Xi = - np.real(Xi) * Gamma**2 / np.pi
    return Xi


def spin_conductivity_I(js, vel, bands, E_F, Gamma=1e-8, alpha=0, sum_k=True):
    """
    Parameters:
    -----------
        js: (np.ndarray, shape (nk, nk, nband, nband))
            A component if the spin current operator,
            on a grid in k-space. In the basis of eigenstates.
        vel: (np.ndarray, shape (2, nk, nk, nband, nband))
            Velocity operator, in the basis of eigenstates,
            evaluated on a grid in k-space.
        bands: (np.ndarray, shape (nk, nk, nband))
            Grid of eigenenergies.
        E_F: (float)
            Fermi level
        Gamma: (float default 1e-8)
            Band's broadening. (disorder)
        alpha: (float default 0)
            Angle of the applied electric field.
            (alpha=0 for x axis, alpha=pi/2 for y axis.)
        sum_k: (Bool default True)
            Flag to determin if k-integration is performed or not.
    Returns:
    --------
        If sum_k==True:
            Xi_k: (np.ndarray,shape (nk, nk))
                Spin conducticity, on a k-grid.
        else:
            Xi: (float)
                Spin conductivity.

    """
    nk = np.shape(bands)[0]
    Ex, Ey = np.cos(alpha), np.sin(alpha)
    vE = vel[0] * Ex + vel[1] * Ey
    gE = 1. / ((E_F-bands)**2 + Gamma**2)
    XiI_k = np.einsum("ijn, ijnm, ijm, ijmn->ij", gE, js, gE, vE)
    XiI_k = - np.real(XiI_k) * Gamma**2 / np.pi
    if sum_k:
        Xi = np.sum(XiI_k) / (nk**2)
    else:
        Xi = XiI_k
    return Xi
