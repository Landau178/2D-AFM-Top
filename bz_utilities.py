import numpy as np
from scipy import linalg as la


def cartesian_to_reduced_k_matrix(b1, b2):
    """
    Calculates the transformation matrix between
    cartesian coordinates (kx, ky) anbd reduced coordinates
    (k1,k2). In such a way that:
        k_red = M @ k_cart
    Parameters:
    -----------
        b1: (np.ndarray, shape (2,))
            First basis vector of recip. lattice.
        b2: (np.ndarray, shape (2,))
            Second basis vector of recip. lattice.
    Returns:
    --------
        M: (np.ndarray of shape (2,2))
            Matrix to connect cartian and reduced
            coordinate systems.

    """
    B = np.array([
        [b1[0], b1[1]],
        [b2[0], b2[1]]
    ])
    metric_tensor = np.array([
        [b1@b1, b1@b2],
        [b2@b1, b2 @ b2]
    ])
    M = la.inv(metric_tensor) @ B
    return M


def fermi_dist(energies, mu, T=0.001):
    """
    Calculate the Fermi-Dirac distribution over
    and array of energies, for a fixed value of the
    temperature T.
    Parameters:
    -----------
        energies: (np.ndarray, of any shape)
            Array of energies in eV.
        mu: (float)
            Chemical potential in eV.
        T: (float, default is 0.001)
            Temperature in Kelvin degrees.

    Returns:
    -------
        nF: (np.ndarray, shape shape of energies)
            Fermi-Dirac dist evaluated in the array
            energies.
    """
    kB = 8.6173324e-5
    beta = 1 / (kB*T)
    old_settings = np.seterr(over="ignore")
    nF = 1. / (np.exp((energies-mu) * beta) + 1)
    np.seterr(**old_settings)
    return nF


def index_fermi_level(bands, Ef):
    """
    Returns the index of the last occuped state,
    according to the fermi level.

    Parameters:
    -----------
        bands: (np.ndarray, shape (nband,))
            Energies, ordered from the lowest to the highest.
        Ef: (float)
            Fermi Level.
    Returns:
    -------
        index_Ef: (int)
            Index of the top occupied state.
            -1 means no occupied sate.
    """
    index_Ef = -1
    nband = np.size(bands)
    for i in range(nband):
        if bands[i] < Ef:
            index_Ef += 1
        else:
            break
    return index_Ef


def pauli_matrix(i):
    """
    Returns the i-th pauli Matrix.
    """
    pauli_matrices = np.array([
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]]
    ])
    return pauli_matrices[i]


def pauli_vector(coefs):
    """
    Parameters:
    -----------
        coefs: (float, float, float, float)
            Coefficients (t0, tx, ty, tz).
    Returns:
        matrix: (np.ndarray shape is )
    """
    t0 = coefs[0]
    matrix = np.array([[t0, 0], [0, t0]])
    for i in range(3):
        sigma_i = pauli_matrix(i)
        matrix += coefs[i+1] * sigma_i
    return matrix


def fix_gauge_eigenvector(eivecs):
    """
    Take the matrix of eigenvectors and make real
    the first component of each one.

    Parameters:
    ----------
        eivecs: (np.ndarray, shape (n,n))
            eivecs[i, n] is the i-component
            of the n-eigenvetor.
    Returns:
    --------
        new_eivecs: (np.ndarray, shape (n,n))
            Eigenvectors with the first component
            real.
    """
    n = np.shape(eivecs)[0]
    new_eivecs = np.copy(eivecs)
    for i in range(n):
        phase = np.angle(eivecs[0, i])
        new_eivecs[:, i] = eivecs[:, i] * np.exp(-1j * phase)
    return new_eivecs
