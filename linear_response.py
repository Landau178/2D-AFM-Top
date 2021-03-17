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
# [2] Origin of the magnetic spin Hall effect:
#     Spin current vorticity in the Fermi sea
#     Physical Rreview R 2, 023065 (2020)
#       DOI: 10.1103/PhysRevResearch.2.023065
# -----------------------------------------------------------------------------
# Spin conduictivities, odd and even.
# -----------------------------------------------------------------------------


# @profile
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


def spin_conductivity_k_zelezny_intra(eivals, eivecs, velocity, Ef, i, a, b, Gamma):
    """
    Same as spin_conductivity_k but only intraband contribution.
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
    sigma_k = 0
    for n in range(np.size(eivals)):
        sigma_k -= gE[n]**2 * js[n, n]*v_eig[b][n, n]
    sigma_k = np.real(sigma_k) * Gamma**2 / np.pi
    return sigma_k


def spin_conductivity_k_zelezny_inter(eivals, eivecs, velocity, Ef, i, a, b, Gamma):
    """
    Same as spin_conductivity_k but only interband contribution.
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
    sigma_k = 0
    for n in range(np.size(eivals)):
        for m in range(np.size(eivals)):
            if not(n == m):
                sigma_k -= gE[n] * gE[m] * js[n, m]*v_eig[b][m, n]
    sigma_k = np.real(sigma_k) * Gamma**2 / np.pi
    return sigma_k


def spin_conductivity_k_mook(eivals, eivecs, velocity, Ef, i, a, b, Gamma):
    """
    Same as spin_conductivity_k, but using the definittion of Mook2020 [2].
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
    sigma_k = 0
    for n in range(nband):
        for m in range(nband):
            if not(n == m):
                f_n = bzu.fermi_dist(eivals[n], Ef)
                f_m = bzu.fermi_dist(eivals[m], Ef)
                factor = -Gamma*(f_m-f_n) / (eivals[n]-eivals[m])
                denominator = (eivals[n]-eivals[m])**2 + Gamma**2
                sigma_k += factor / denominator * js[n, m]*v_eig[b][m, n]
            else:  # ill-defined term
                s = Gamma
                df = -s/np.pi / ((eivals[n]-Ef)**2 + s**2)
                sigma_k += js[n, n]*v_eig[b][n, n] * df / Gamma
    return np.real(sigma_k)


def spin_conductivity_k_mook_gaussian(eivals, eivecs, velocity, Ef, i, a, b, Gamma):
    """
    Same as spin_conductivity_k, but using the definittion of Mook2020 [2],
    and a gaussian approximation fo the dirac delta.
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
    sigma_k = 0
    for n in range(nband):
        for m in range(nband):
            if not(n == m):
                f_n = bzu.fermi_dist(eivals[n], Ef)
                f_m = bzu.fermi_dist(eivals[m], Ef)
                factor = -Gamma*(f_m-f_n) / (eivals[n]-eivals[m])
                denominator = (eivals[n]-eivals[m])**2 + Gamma**2
                sigma_k += factor / denominator * js[n, m]*v_eig[b][m, n]
            else:  # ill-defined term
                s = Gamma
                x = eivals[n]-Ef
                df = -np.exp(- 0.5 * x**2 / s**2) / (s * np.sqrt(2*np.pi))
                sigma_k += js[n, n]*v_eig[b][n, n] * df / Gamma
    return np.real(sigma_k)


def spin_conductivity_k_mook_intra(eivals, eivecs, velocity, Ef, i, a, b, Gamma):
    """
    Same as spin_conductivity_k, but using the definition of Mook2020 [2].
    nly intraband contribution.
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
    sigma_k = 0
    for n in range(nband):
        s = Gamma
        df = -s/np.pi / ((eivals[n]-Ef)**2 + s**2)
        sigma_k += js[n, n]*v_eig[b][n, n] * df / Gamma
    return np.real(sigma_k)


def spin_conductivity_k_mook_inter(eivals, eivecs, velocity, Ef, i, a, b, Gamma):
    """
    Same as spin_conductivity_k, but using the definition of Mook2020 [2].
    Only interband contribution.
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
    sigma_k = 0
    for n in range(nband):
        for m in range(nband):
            if not(n == m):
                f_n = bzu.fermi_dist(eivals[n], Ef)
                f_m = bzu.fermi_dist(eivals[m], Ef)
                factor = -Gamma*(f_m-f_n) / (eivals[n]-eivals[m])
                denominator = (eivals[n]-eivals[m])**2 + Gamma**2
                sigma_k += factor / denominator * js[n, m]*v_eig[b][m, n]
    return np.real(sigma_k)


# @profile
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

# -----------------------------------------------------------------------------
# Charge conductivities, odd and even.
# -----------------------------------------------------------------------------


# @profile
def charge_conductivity_k(eivals, eivecs, velocity, Ef, a, b, Gamma):
    """
    Same as self.spin_conductivity_k, but calculates
    the odd charge conductivity.
    """
    vx_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[0], eivecs)
    vy_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[1], eivecs)
    v_eig = [vx_eig, vy_eig]

    gE = 1. / ((Ef-eivals)**2 + Gamma**2)
    sigma_k = np.einsum("n, nm, m, mn->", gE, v_eig[a], gE, v_eig[b])
    sigma_k = np.real(sigma_k) * Gamma**2 / np.pi
    return sigma_k


# @profile
def charge_conductivity_k_even(eivals, eivecs, velocity, Ef, a, b):
    """
    """
    nband = np.size(eivals)
    vx_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[0], eivecs)
    vy_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[1], eivecs)
    v_eig = [vx_eig, vy_eig]

    index_Ef = bzu.index_fermi_level(eivals, Ef)
    sigma_k = 0
    for n in range(index_Ef+1):
        for m in range(index_Ef+1, nband):
            denominator = (eivals[n]-eivals[m])**2
            sigma_k += 2 * v_eig[a][n, m]*v_eig[b][m, n] / denominator
    return np.imag(sigma_k)


def charge_conductivity_k_odd_Mook(eivals, eivecs, velocity, Ef, a, b, Gamma):
    """
    """
    vx_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[0], eivecs)
    vy_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[1], eivecs)
    v_eig = [vx_eig, vy_eig]
    sigma_k = 0
    nband = np.size(eivals)
    for n in range(nband):
        for m in range(nband):
            va_vb = np.real(v_eig[a][n, m]*v_eig[b][m, n])
            # interband
            if not(n == m):
                f_n = bzu.fermi_dist(eivals[n], Ef)
                f_m = bzu.fermi_dist(eivals[m], Ef)
                factor = Gamma*(f_m-f_n) / (eivals[n]-eivals[m])
                denominator = (eivals[n]-eivals[m])**2 + Gamma**2
                sigma_k += factor / denominator * va_vb
            # intraband
            else:  # ill-defined term
                s = Gamma
                df = s/np.pi / ((eivals[n]-Ef)**2 + s**2)
                sigma_k += va_vb * df / Gamma
    return sigma_k


def charge_conductivity_k_even_Mook(eivals, eivecs, velocity, Ef, a, b, Gamma):
    """
    """
    nband = np.size(eivals)
    vx_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[0], eivecs)
    vy_eig = np.einsum("nis, isjd, mjd-> nm",
                       eivecs.conj(), velocity[1], eivecs)
    v_eig = [vx_eig, vy_eig]

    fermi = bzu.fermi_dist(eivals, Ef)
    sigma_k = 0
    for n in range(nband):
        for m in range(nband):
            va_vb = np.imag(v_eig[a][n, m]*v_eig[b][m, n])
            gap = (eivals[n]-eivals[m])
            factor = (fermi[m]-fermi[n]) / (gap**2 + Gamma**2)
            sigma_k += factor * va_vb
    return sigma_k


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Upgrade of functions that accepts k-grid operators, and perform the integral
# in k-space.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Spin conductivity
# ------------------------------------------------------------------------------
# @profile
def spin_conductivity_k_odd_upg(eivals, v_eig, js_eig, Ef, i, a, b, Gamma):
    nk = np.shape(eivals)[0]
    dk = 1 / nk
    dim_k = len(np.shape(eivals))-1
    v_b = v_eig[b]
    js = js_eig[i, a]

    gE = 1. / ((Ef-eivals)**2 + Gamma**2)
    k_lab = {1: "k", 2: "kq", 3: "kqp"}[dim_k]
    subscripts = f"{k_lab}n, {k_lab}nm, {k_lab}m, {k_lab}mn->{k_lab}s"
    sigma_k = np.einsum(subscripts, gE, js, gE, v_b)
    sigma_k = - np.real(sigma_k) * Gamma**2 / np.pi
    return sigma_k


# @profile
def spin_conductivity_k_even_upg(eivals, v_eig, js_eig, Ef, i, a, b):
    nk = np.shape(eivals)[0]
    dk = 1 / nk
    dim_k = len(np.shape(eivals))-1
    gap_denom = gap_denominator(eivals, Ef)
    v_b = v_eig[b]
    js = js_eig[i, a]
    k_lab = {1: "k", 2: "kq", 3: "kqp"}[dim_k]
    subscripts = f"{k_lab}nm ,{k_lab}nm, {k_lab}mn->{k_lab}"
    sigma_k = np.einsum(subscripts, js, gap_denom, v_b)
    return -2 * np.imag(sigma_k)

# ------------------------------------------------------------------------------
# Charge conductivity
# ------------------------------------------------------------------------------


def charge_conductivity_k_odd_upg(eivals, v_eig, Ef, a, b, Gamma):
    nk = np.shape(eivals)[0]
    dk = 1 / nk
    dim_k = len(np.shape(eivals))-1
    v_b = v_eig[b]
    v_a = v_eig[a]
    gE = 1. / ((Ef-eivals)**2 + Gamma**2)
    k_lab = {1: "k", 2: "kq", 3: "kqp"}[dim_k]
    subscript = f"{k_lab}n, {k_lab}nm, {k_lab}m, {k_lab}mn->{k_lab}"
    sigma_k = np.einsum(subscript, gE, v_a, gE, v_b)
    sigma_k = - np.real(sigma_k) * Gamma**2 / np.pi
    return sigma_k


def charge_conductivity_k_even_upg(eivals, v_eig, Ef, a, b):
    nk = np.shape(eivals)[0]
    dk = 1 / nk
    dim_k = len(np.shape(eivals))-1
    gap_denom = gap_denominator(eivals, Ef)
    v_b = v_eig[b]
    v_a = v_eig[a]
    k_lab = {1: "k", 2: "kq", 3: "kqp"}[dim_k]
    subscripts = f"{k_lab}nm ,{k_lab}nm, {k_lab}mn->{k_lab}"
    sigma_k = np.einsum(subscripts, v_a, gap_denom, v_b)
    return -2 * np.imag(np.sum(sigma_k)) * dk**2

# ------------------------------------------------------------------------------
# Useful utility to calculate time-even integrands
# ------------------------------------------------------------------------------


def gap_denominator(eivals, Ef):
    """
    Calculate the matrix of components:
        M_{nm} = f_n(1-f_m)/(E_n - E_m)**2 
    If not(n==m) and 0 if n==m.

    Parameters:
    -----------
        eivals: (np.ndarray, shape (nk, nk, nbands))
            Grid in k-space of eivengalues

    Returns:
    --------
        matrix: (np.ndarray of shape(nk, nk, nbands, nband))
    """
    fermi = bzu.fermi_dist(eivals, Ef)
    nbands = np.shape(eivals)[-1]
    nk = np.shape(eivals)[0]
    matrix = np.zeros((nk, nk, nbands, nbands))
    for n in range(nbands):
        for m in range(nbands):
            if not(n == m):
                denom_nm = 1 / (eivals[..., n]-eivals[..., m])**2
                fermi_factor = fermi[..., n] * (1-fermi[..., m])
                matrix[..., n, m] = denom_nm * fermi_factor

    return matrix

# -----------------------------------------------------------------------------
# Implementation of SHC with the spin Berry curvature, following zhang2018.
# -----------------------------------------------------------------------------


def spin_conductivity_k_even_zhang(eivals, v_eig, js_eig, Ef, i, a, b):
    nk = np.shape(eivals)[0]
    dk = 1 / nk
    dim_k = len(np.shape(eivals))-1
    gap_denom = gap_denominator_2(eivals, Ef)
    v_b = v_eig[b]
    js = js_eig[i, a]
    k_lab = {1: "k", 2: "kq", 3: "kqp"}[dim_k]
    subscripts = f"{k_lab}nm ,{k_lab}nm, {k_lab}mn->{k_lab}"
    sigma_k = np.einsum(subscripts, js, gap_denom, v_b)
    return 2j * sigma_k


def gap_denominator_2(eivals, Ef):
    """
    Calculate the matrix of components:
        M_{nm} = f_n / (E_n - E_m)**2 
    If not(n==m) and 0 if n==m.

    Parameters:
    -----------
        eivals: (np.ndarray, shape (nk, nk, nbands))
            Grid in k-space of eivengalues

    Returns:
    --------
        matrix: (np.ndarray of shape(nk, nk, nbands, nband))
    """
    fermi = bzu.fermi_dist(eivals, Ef)
    nbands = np.shape(eivals)[-1]
    nk = np.shape(eivals)[0]
    matrix = np.zeros((nk, nk, nbands, nbands))
    for n in range(nbands):
        for m in range(nbands):
            if not(n == m):
                denom_nm = 1 / (eivals[..., n]-eivals[..., m])**2
                fermi_factor = fermi[..., n]
                matrix[..., n, m] = denom_nm * fermi_factor

    return matrix
