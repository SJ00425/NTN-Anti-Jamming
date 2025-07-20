# beamforming_utils.py
import numpy as np

def svd_bf(h: np.ndarray, tx_antennas):
    """
    Performs SVD-based beamforming to extract w_t (transmit beam) and w_r (receive beam).

    Parameters
    ----------
    h : np.ndarray
        The channel matrix, typically of shape (num_tx_antennas, num_rx_antennas).

    Returns
    -------
    w_t : np.ndarray
        The first column of the left singular vectors, of shape (num_tx_antennas, 1).
    w_r : np.ndarray
        The first column of the right singular vectors, of shape (num_rx_antennas, 1).
    """
    
    # h = ((h)*np.sqrt(tx_antennas)  /np.linalg.norm(h))
    h = (h) /np.linalg.norm(h)
    # h = U * diag(s) * Vh
    U, s, Vh = np.linalg.svd(h)
    # Extract the principal singular-vector components
    w_t = U[:, 0].reshape(-1, 1)
    V = Vh.conj().T
    w_r = V[:, 0].reshape(-1, 1)
    return w_t, w_r


def nulling_bf(h: np.ndarray, 
               w_r: np.ndarray, 
               interference_term: np.ndarray, 
               lambda_: float,
               tx_antennas):
    """
    Calculates the nulling vector v_null based on the interference covariance.

    The nulling vector is the principal eigenvector of 
    Q = h * w_r * w_r^H * h^H - lambda_ * interference_term.

    Parameters
    ----------
    h : np.ndarray
        The channel matrix of shape (num_tx_antennas, num_rx_antennas). 
        Must be compatible with w_r (i.e., h.shape[1] == w_r.shape[0]).
    w_t : np.ndarray
        The transmitte beamforming vector, of shape (num_tx_antennas, 1).
    interference_term : np.ndarray
        The aggregated interference covariance matrix, typically shape (num_tx_antennas, num_tx_antennas).
    lambda_ : float
        A weighting factor that balances the desired signal versus the interference penalty.

    Returns
    -------
    v_null : np.ndarray
        The nulling vector, of shape (num_rx_antennas, 1),
        which is used on the transmit side (or receive side, depending on your convention)
        to mitigate interference.
    """
    
    # h= ((h)*np.sqrt(tx_antennas)  /np.linalg.norm(h))
    # h = ((h) /np.linalg.norm(h))
    # interference_term = (interference_term) /np.linalg.norm(interference_term)
    # Build the matrix Q
    A = h @ w_r @ w_r.conj().T @ h.conj().T
    B = lambda_ * interference_term
    Q = A-B
    # Q = h @ w_r @ w_r.conj().T @ h.conj().T - lambda_ * interference_term
    
    # Eigen-decomposition of Q
    eigen_values, v_nulls = np.linalg.eig(Q)

    # Sort eigenvalues from largest to smallest
    idx = np.argsort(eigen_values)[::-1]
    eigen_values_sorted = eigen_values[idx]
    max_eigen_value = eigen_values_sorted[0]
    v_nulls = v_nulls[:, idx]
    # The nulling vector is the eigenvector corresponding to the largest eigenvalue
    v_null = v_nulls[:, 0].reshape(-1, 1)
    
    return v_null, A, B, max_eigen_value
