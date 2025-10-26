import numpy as np

def _build_Q1(P1, N1):
    """
    Interpret P1 into a jammer covariance Q1 (N1xN1):
      - scalar -> P1 * I
      - 1D vector -> ||w|| normalized, Q1 = (||P1|| as scalar power) * w w^H
      - 2D matrix -> returned as is (validated Hermitian)
    """
    P1 = np.asarray(P1)
    if P1.ndim == 0:  # scalar power, isotropic
        power = float(P1)
        return power * np.eye(N1, dtype=complex)
    if P1.ndim == 1:  # beam vector
        w = P1.astype(complex).reshape(-1, 1)
        norm = np.linalg.norm(w)
        if norm == 0:
            raise ValueError("P1 beam vector has zero norm.")
        w = w / norm
        # Use power = 1 by default; if you want E1 scaling pass a scalar times w.
        # If the user passed c*w, absorb |c|^2 into power:
        power = norm**2
        return power * (w @ w.conj().T)
    if P1.ndim == 2:
        if P1.shape[0] != P1.shape[1]:
            raise ValueError("P1 matrix must be square.")
        # Ensure Hermitian numerically
        return 0.5 * (P1 + P1.conj().T)
    raise ValueError("Unsupported P1 shape; must be scalar, (N1,), or (N1,N1).")


def _waterfill(lambdas, E0, eps=1e-12):
    lambdas = np.asarray(lambdas, dtype=float)
    pos = lambdas > eps
    lam = lambdas[pos]
    p = np.zeros_like(lambdas)
    if lam.size == 0 or E0 <= 0:
        return p, 0.0
    a = 1.0 / lam  # 1/lambda_i
    # Sort already in DESC lambda => a ASC
    prefix = np.cumsum(a)
    k_star, mu = 0, 0.0
    for k in range(1, len(lam) + 1):
        mu_k = (E0 + prefix[k-1]) / k
        if mu_k > a[k-1] + eps:
            k_star, mu = k, mu_k
        else:
            break
    if k_star > 0:
        p_active = np.maximum(0.0, mu - a)
        p_active[k_star:] = 0.0
        p[pos] = p_active
    return p, mu


def optimal_Q0(H0, H1, N0, P0, P1):
    """
    Multi-stream optimal Q0* for:
        max_{Q0 >= 0, Tr(Q0)=P0} log2 det(I + H0 Q0 H0^H (H1 Q1 H1^H + N0 I)^{-1})

    Parameters
    ----------
    H0 : (M, N0) complex ndarray
    H1 : (M, N1) complex ndarray
    N0 : float  (noise variance)
    P0 : float  (desired TX power budget)
    P1 : jammer spec:
         - scalar -> isotropic Q1 = P1 * I_{N1}
         - (N1,)  -> beam vector w1 (power absorbed by ||w1||^2), Q1 = (||w1||^2) * (w1hat w1hat^H)
         - (N1,N1)-> full covariance Q1

    Returns
    -------
    Q0 : (N0, N0) complex Hermitian PSD ndarray  (optimal transmit covariance)
    info : dict diagnostics (eigenvalues, powers, rank, rate, etc.)
    """
    H0 = np.asarray(H0, dtype=complex)
    H1 = np.asarray(H1, dtype=complex)

    M, N0_dim = H0.shape
    M1, N1_dim = H1.shape
    if M != M1:
        raise ValueError("H0 and H1 must have the same number of rows (same RX dimension M).")

    # Build jammer covariance Q1 from P1
    Q1 = _build_Q1(P1, N1_dim)

    # P = H1 Q1 H1^H + N0 I
    P = H1 @ Q1 @ H1.conj().T + float(N0) * np.eye(M, dtype=complex)

    # Solve P^{-1} H0 without explicit inversion
    X = np.linalg.solve(P, H0)                 # (M,N0)
    Mmat = H0.conj().T @ X                     # (N0,N0), Hermitian

    # Hermitian eigendecomposition: eigh -> ascending; flip to descending
    w, V = np.linalg.eigh((Mmat + Mmat.conj().T) / 2.0)
    order = np.argsort(w)[::-1]
    lambdas = np.maximum(w[order], 0.0)
    V = V[:, order]

    # Water-filling
    p, mu = _waterfill(lambdas, float(P0))

    # Construct Q0* = V diag(p) V^H
    Q0 = (V * p) @ V.conj().T

    # Achieved rate
    A = H0 @ Q0 @ H0.conj().T
    Y = np.linalg.solve(P, A)
    I_plus = np.eye(M, dtype=complex) + Y
    sign, logdet = np.linalg.slogdet(I_plus)
    rate_bpcu = (logdet / np.log(2.0)) if sign > 0 else np.nan

    info = {
        "lambdas": lambdas,
        "V": V,
        "p": p,
        "mu": mu,
        "rank": int(np.count_nonzero(p > 1e-10)),
        "rate_bpcu": float(rate_bpcu),
    }
    return Q0, Q1 ,info
