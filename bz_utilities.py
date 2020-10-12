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
