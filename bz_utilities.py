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
