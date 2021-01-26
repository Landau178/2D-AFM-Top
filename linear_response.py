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

# -----------------------------------------------------------------------------
# Spin conduictivities, odd and even.
# -----------------------------------------------------------------------------


def spin_conductivity_k(eivals, eivecs, velocity, Ef, i, a, b, Gamma):
    """
    Calculate the k-resolved odd spin conductivity at a given k-point,
    according to [1].
    In the following, n is the number of bands, 

    Parameters:
    ----------
        eivals: (np.ndarray shape is (n,))
            Eigenvals at som k-point.
        eivecs: (np.ndarray shape is (n,norb, 2))
            Eigenvectors at some k-point.
            First component is for bands, second is for
            orbital and third for spin.
        velocity: (np.ndarray, shape is (norb,2,norb, 2))
            Velocity operator. Gradient of the Hamiltonian matrix at k.
        Ef: (float)
            Fermi level.
        i, a, b: (int, int , int)
            Components of spin conductivity tensor.
                i: spin polarization.
                a: direction of current
                b: direction of applied electric field.
        Gamma: (float)
            band broadening. (disorder)

    Returns:
    -------
        sigma_k: (float)
            Contribution to the odd spin conductivity at k.
    """
    S = bzu.pauli_matrix(i) / 2
    S_eig = np.einsum("nis, st, mit-> nm", eivecs.conj(), S, eivecs)
    vx_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[0], eivecs)
    vy_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[1], eivecs)
    v_eig = [vx_eig, vy_eig]
    js = 0.5 * (S_eig @ v_eig[a] + v_eig[a] @ S_eig)

    gE = 1. / ((Ef-eivals)**2 + Gamma**2)
    sigma_k = np.einsum("n, nm, m, mn->", gE, js, gE, v_eig[b])
    sigma_k = - np.real(sigma_k) * Gamma**2 / np.pi
    return sigma_k


def spin_conductivity_k_even(eivals, eivecs, velocity, Ef, i, a, b):
    """
    Calculate the k-resolved even spin conductivity at a given k-point,
    according to [1].

    Parameters:
    ----------
        eivals: (np.ndarray shape is (n,))
            Eigenvals at som k-point.
        eivecs: (np.ndarray shape is (n,norb, 2))
            Eigenvectors at some k-point.
            First component is for bands, second is for
            orbital and third for spin.
        velocity: (np.ndarray, shape is (norb,2,norb, 2))
            Velocity operator. Gradient of the Hamiltonian matrix at k.
        Ef: (float)
            Fermi level.
        i, a, b: (int, int , int)
            Components of spin conductivity tensor.
                i: spin polarization.
                a: direction of current
                b: direction of applied electric field.
        Gamma: (float)
            band broadening. (disorder)

    Returns:
    -------
        sigma_k: (float)
            Contribution to the even spin conductivity at k.
    """
    nband = np.size(eivals)
    S = bzu.pauli_matrix(i) / 2
    S_eig = np.einsum("nis, st, mit-> nm", eivecs.conj(), S, eivecs)
    vx_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[0], eivecs)
    vy_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[1], eivecs)
    v_eig = [vx_eig, vy_eig]
    js = 0.5 * (S_eig @ v_eig[a] + v_eig[a] @ S_eig)
    index_Ef = bzu.index_fermi_level(eivals, Ef)
    sigma_k = 0
    for n in range(index_Ef+1):
        for m in range(index_Ef+1, nband):
            denominator = (eivals[n]-eivals[m])**2
            sigma_k += -2 * js[n, m]*v_eig[b][m, n] / denominator
    return np.imag(sigma_k)
