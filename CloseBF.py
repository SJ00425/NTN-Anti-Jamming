import numpy as np

def jammeraware_Q(H: np.ndarray, beta: float, P: float = 1.0, N0: float = 1.0,
                  tol: float = 1e-9, maxiter: int = 100):
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    V = Vh.conj().T
    s2 = s**2
    Nt = V.shape[0]
    ln2 = np.log(2.0)

    if P <= 0 or Nt == 0 or s2.size == 0 or np.all(s2 == 0):
        Q0 = np.zeros((Nt, Nt), dtype=complex)
        return Q0, beta*Q0, np.zeros_like(s2), 0.0, 0.0  # + lambda_star

    def q_from_lambda(lmbd):
        if beta <= 1e-12:
            water = N0 / (lmbd * ln2)
            qi = water - np.where(s2 > 0, N0 / s2, np.inf)
            return np.maximum(qi, 0.0)

        a = beta * (1.0 + beta)
        with np.errstate(divide='ignore', invalid='ignore'):
            radicand = 1.0 + 4.0 * a * (s2 / (lmbd * ln2 * N0))
        radicand = np.maximum(radicand, 1.0)
        sqrt_term = np.sqrt(radicand)
        coeff = N0 / (2.0 * a)
        b = 1.0 + 2.0 * beta
        qi = np.zeros_like(s2)
        nz = s2 > 0
        qi[nz] = (coeff / s2[nz]) * (sqrt_term[nz] - b)
        active = s2 > (lmbd * N0 * ln2)      # strict >
        qi = np.where(active, np.maximum(qi, 0.0), 0.0)
        return qi

    # --- bracket (ensure sum(q(lam_low)) > P and sum(q(lam_high)) < P) ---
    s2max = float(np.max(s2))
    lam_low  = 1e-16
    lam_high = s2max / (N0 * ln2) * (1.0 - 1e-12)  # nudge down from threshold

    q_low  = q_from_lambda(lam_low);  S_low  = q_low.sum()
    q_high = q_from_lambda(lam_high); S_high = q_high.sum()

    # If not bracketed, try to fix it
    # make sure S_low > P
    expand = 0
    while S_low <= P and expand < 20:
        lam_low *= 0.1
        q_low = q_from_lambda(lam_low); S_low = q_low.sum()
        expand += 1
    # make sure S_high < P
    shrink = 0
    while S_high >= P and shrink < 20:
        lam_high *= 0.9
        q_high = q_from_lambda(lam_high); S_high = q_high.sum()
        shrink += 1

    # If still not bracketed (very pathological), fall back to scaling q_low
    if not (S_low > P and S_high < P):
        qi = q_low * (P / max(S_low, 1e-16))
        Q0 = V @ np.diag(qi) @ V.conj().T
        Q1 = beta * Q0
        x = s2 * qi
        J = np.sum(np.log2(N0 + (1.0 + beta) * x) - np.log2(N0 + beta * x))
        return Q0, Q1, qi, J, lam_low

    # --- bisection ---
    lamL, lamH = lam_low, lam_high
    qi = None; lambda_star = None
    for _ in range(maxiter):
        lam_mid = 0.5 * (lamL + lamH)
        qi_mid = q_from_lambda(lam_mid)
        S = qi_mid.sum()
        if abs(S - P) <= tol:
            qi = qi_mid; lambda_star = lam_mid
            break
        if S > P:
            lamL = lam_mid
        else:
            lamH = lam_mid
    if qi is None:
        # choose closer side
        qL = q_from_lambda(lamL); SL = qL.sum()
        qH = q_from_lambda(lamH); SH = qH.sum()
        if abs(SL - P) <= abs(SH - P):
            qi, lambda_star = qL, lamL
        else:
            qi, lambda_star = qH, lamH

    # Normalize tiny mismatch on power
    S = qi.sum()
    if S > 0 and abs(S - P) > tol:
        qi *= (P / S)

    Q0 = V @ np.diag(qi) @ V.conj().T
    Q1 = beta * Q0
    x = s2 * qi
    J = np.sum(np.log2(N0 + (1.0 + beta) * x) - np.log2(N0 + beta * x))
    J2 = compute_J_via_logdet(H, Q0, beta, N0)
    return Q0, Q1, qi, J, J2, lambda_star


def compute_J_via_logdet(H: np.ndarray, Q0: np.ndarray, beta: float, N0: float = 1.0):
    """
    J = log2 det( I + H Q0 H^H (N0 I + beta H Q0 H^H)^{-1} )
    """
    Nr = H.shape[0]
    HQH = H @ Q0 @ H.conj().T
    A = np.eye(Nr, dtype=complex) + HQH @ np.linalg.inv(N0*np.eye(Nr) + beta*HQH)
    sign, logdet = np.linalg.slogdet(A)   # numerically stable
    J_mat = (logdet / np.log(2.0)) * sign
    return float(J_mat)