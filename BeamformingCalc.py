
import numpy as np
from typing import Tuple

def svd_bf(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    SVD beamforming for y = H x + n with H shape (Nr, Nt).
    Returns:
        w_t: (Nt, 1) principal right singular vector (transmit beam)
        w_r: (Nr, 1) principal left  singular vector (receive combiner)
    """
    if H.ndim != 2:
        raise ValueError("H must be 2D (Nr x Nt)")
    Nr, Nt = H.shape
    # SVD: H = U Σ Vh, where U:(Nr,Nr), Vh:(Nt,Nt)
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    v1 = Vh.conj().T[:, 0].reshape(Nt, 1)  # (Nt,1)
    u1 = U[:, 0].reshape(Nr, 1)            # (Nr,1)
    # ensure unit-norm (guard against numerical drift)
    v1 = v1 / (np.linalg.norm(v1) + 1e-15)
    u1 = u1 / (np.linalg.norm(u1) + 1e-15)
    return v1, u1  # (w_t, w_r)



def nulling_bf(h: np.ndarray, 
               w_r: np.ndarray, 
               interference_term: np.ndarray, 
               lambda_: float,):
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
    # interference_term= hermitize(interference_term)
    B = lambda_ * interference_term
    Q = A-B
    Q = 0.5 * (Q + Q.conj().T)
    # Q = h @ w_r @ w_r.conj().T @ h.conj().T - lambda_ * interference_term
    
    # Eigen-decomposition of Q
    eigen_values, v_nulls = np.linalg.eigh(Q)

    # Sort eigenvalues from largest to smallest
    idx = np.argsort(eigen_values)[::-1]
    eigen_values_sorted = eigen_values[idx]
    max_eigen_value = eigen_values_sorted[0]
    v_nulls = v_nulls[:, idx]
    # The nulling vector is the eigenvector corresponding to the largest eigenvalue
    v_null = v_nulls[:, 0].reshape(-1, 1)
    
    return v_null, A, B, max_eigen_value



def left_singular_u1(H: np.ndarray) -> np.ndarray:
    """
    Fast left singular vector u1 for y = H x + n, H shape (Nr, Nt).
    Returns:
        w_r (Nr,1): principal left singular vector (unit norm).
    """
    # G is small when Nr << Nt
    G = H @ H.conj().T                         # (Nr, Nr), Hermitian PSD
    # eigh is for Hermitian; eigenvalues are ascending
    vals, vecs = np.linalg.eigh(G)
    u1 = vecs[:, -1].reshape(-1, 1)            # principal eigenvector
    u1 /= (np.linalg.norm(u1) + 1e-15)
    return u1
def hermitize(M):
    return 0.5 * (M + M.conj().T)







def nulling_bf_fast_scipy(h, w_r, R, lambda_, tol=1e-8, maxiter=2000, dtype=np.complex128):
    """
    Fast & accurate: v solving max eigenpair of Q = h w_r w_r^H h^H - λ R.
    Shapes:
      h: (Nt, Nr)   (same as your dense code)
      w_r: (Nr, 1)
      R: (Nt, Nt)
    Returns: v_null (Nt,1), eigval (float)
    """
    from scipy.sparse.linalg import eigsh, LinearOperator

    h = h.astype(dtype, copy=False)
    w_r = w_r.astype(dtype, copy=False)
    # R = hermitize(R.astype(dtype, copy=False))
    Nt = h.shape[0]

    a = (h @ w_r).reshape(-1)              # DO NOT normalize; same as dense
    def Qx(x):
        # x is (Nt,), return (Nt,)
        y = a * np.vdot(a, x)             # rank-1 part
        y -= lambda_ * (R @ x)            # interference part
        return y

    Op = LinearOperator((Nt, Nt), matvec=Qx, dtype=dtype)
    vals, vecs = eigsh(Op, k=1, which='LA', tol=tol, maxiter=maxiter)  # LA = largest algebraic
    v = vecs[:, [0]]
    v /= (np.linalg.norm(v) + 1e-30)
    return v, float(vals[0].real)