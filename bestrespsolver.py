# ntn_bestresp.py
"""
Unified NTN covariance min-max game — Best-Response (Q0) + EG/MP (Q1)

Saddle problem:
  Max over Q0 ⪰ 0, tr(Q0) ≤ P0
  Min over Q1 ⪰ 0, tr(Q1) ≤ P1
  J(Q0,Q1) = log2 det(N0 I + H1 Q1 H1^H + H0 Q0 H0^H) - log2 det(N0 I + H1 Q1 H1^H)

This file contains ONLY the Best-Response solver and the minimal utilities it depends on:
  - hermitian, chol_inv_apply, gradients, compute_J
  - project_psd_trace (Euclidean spectral projection with trace ≤ or =)
  - classical water-filling on Q0 (closed-form best response)
  - entropy step for Q1 (optional Mirror-Prox geometry on jammer updates)
  - residual for BR scheme
  - solve_game_bestresp_Q0_then_Q1 (main entry)

Author: you :)
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import cholesky, solve, eigh

__all__ = [
    "solve_game_bestresp_Q0_then_Q1",
    "compute_J",
    "gradients",
    "project_psd_trace",
    "waterfilling_Q0",
    "mp_entropy_step",
    "stationarity_residual_BRG",
]

# ----------------------------- Common utilities -----------------------------

def hermitian(X: np.ndarray) -> np.ndarray:
    """Return Hermitian part of X: (X + Xᴴ)/2."""
    return 0.5 * (X + X.conj().T)

def chol_inv_apply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute A^{-1} B for Hermitian positive-(semi)definite A via Cholesky.
    Adds a tiny jitter if A is near-singular.
    """
    M = A.shape[0]
    try:
        L = cholesky(A)
    except np.linalg.LinAlgError:
        eps = 1e-12 * np.trace(A).real / max(M, 1)
        L = cholesky(A + eps * np.eye(M, dtype=complex))
    Y = solve(L, B)              # L Y = B
    return solve(L.conj().T, Y)  # Lᴴ X = Y ⇒ X = A^{-1}B

def gradients(H0: np.ndarray, H1: np.ndarray,
              Q0: np.ndarray, Q1: np.ndarray,
              N0: float):
    """
    Matrix gradients (∇_{Q0}J, ∇_{Q1}J).

    Let P = N0 I + H1 Q1 H1ᴴ, S = H0 Q0 H0ᴴ, A = P + S.
      ∇_{Q0} J = H0ᴴ A^{-1} H0          (ascent direction)
      ∇_{Q1} J = H1ᴴ (A^{-1} − P^{-1}) H1  (descent direction)
    """
    M = H0.shape[0]
    P = N0 * np.eye(M, dtype=complex) + H1 @ Q1 @ H1.conj().T
    S = H0 @ Q0 @ H0.conj().T
    Ainv = chol_inv_apply(P + S, np.eye(M, dtype=complex))
    Pinv = chol_inv_apply(P,       np.eye(M, dtype=complex))
    G0 = hermitian(H0.conj().T @ Ainv @ H0)
    G1 = hermitian(H1.conj().T @ (Ainv - Pinv) @ H1)
    return G0, G1

def compute_J(H0: np.ndarray, H1: np.ndarray,
              Q0: np.ndarray, Q1: np.ndarray,
              N0: float) -> float:
    """J(Q0,Q1) = log2 det(P+S) − log2 det(P), with P=N0I+H1Q1H1ᴴ and S=H0Q0H0ᴴ."""
    M = H0.shape[0]
    P = N0 * np.eye(M, dtype=complex) + H1 @ Q1 @ H1.conj().T
    S = H0 @ Q0 @ H0.conj().T
    A = P + S

    def _logdet(mat: np.ndarray) -> float:
        try:
            L = cholesky(mat)
        except np.linalg.LinAlgError:
            eps = 1e-12 * np.trace(mat).real / max(M, 1)
            L = cholesky(mat + eps * np.eye(M, dtype=complex))
        return 2.0 * np.sum(np.log(np.abs(np.diag(L))))

    return (_logdet(A) - _logdet(P)) / np.log(2.0)

# --------- Spectral projection onto PSD with trace constraints (≤ or =) ---------

def project_psd_trace(
    Z: np.ndarray,
    tau: float,
    eps_floor: float = 0.0,
    mode: str = "le",  # 'le' (≤ tau) or 'eq' (= tau)
) -> np.ndarray:
    """
    Euclidean projection onto:
      C_{≤} := { Q ⪰ eps_floor·I, tr(Q) ≤ τ }
      C_{=} := { Q ⪰ eps_floor·I, tr(Q) = τ }.

    Reduces to eigenvalue projection with a boxed simplex on eigenvalues.
    """
    Zh = hermitian(Z)
    lam, U = eigh(Zh)  # ascending
    n = lam.size

    def _proj_simplex_nonneg(y: np.ndarray, z: float) -> np.ndarray:
        # Duchi et al. (2008) O(n log n)
        if z <= 0:
            return np.zeros_like(y)
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - z))[0]
        if len(rho_idx) == 0:
            theta = (cssv[-1] - z) / n
            w = np.maximum(y - theta, 0.0)
            s = w.sum()
            return w if s == z or s == 0 else (z / s) * w
        r = rho_idx[-1] + 1
        theta = (cssv[r - 1] - z) / r
        return np.maximum(y - theta, 0.0)

    lam = np.asarray(lam, dtype=float)
    lam_f = np.maximum(lam, eps_floor)
    base_sum = lam_f.sum()

    if mode == "le" and base_sum <= tau + 1e-12:
        x = lam_f
    else:
        z = tau - n * eps_floor
        if z < 0:
            x = np.full_like(lam, eps_floor)
        else:
            y = lam - eps_floor
            y_proj = _proj_simplex_nonneg(y, z)
            x = eps_floor + y_proj

        if mode == "le":
            s = x.sum()
            if s > tau + 1e-12:
                y = x - eps_floor
                y = _proj_simplex_nonneg(y, tau - n * eps_floor)
                x = eps_floor + y

    X = hermitian(U @ np.diag(x) @ U.conj().T)
    return X

# ------------------------ Q0 best-response (water-filling) ----------------------

def _waterfill_power(inv_snr: np.ndarray, Ptot: float) -> np.ndarray:
    """
    Classical water-filling on inverse SNRs a_i = inv_snr[i]:
      p_i = max(μ − a_i, 0), with Σ p_i = Ptot.
    """
    a = np.array(inv_snr, dtype=float)
    a_sort = np.sort(a)
    csum = np.cumsum(a_sort)
    mu_cand = (Ptot + csum) / (np.arange(1, a.size + 1))
    idx = np.where(mu_cand > a_sort)[0]
    if len(idx) == 0:
        mu = mu_cand[0]
        p = np.maximum(mu - a, 0.0)
        s = p.sum()
        return p if s == 0 else p * (Ptot / s)
    k = idx[-1] + 1
    mu = (Ptot + csum[k - 1]) / k
    p = np.maximum(mu - a, 0.0)
    s = p.sum()
    return p if s == 0 else p * (Ptot / s)

def waterfilling_Q0(H0: np.ndarray, P: np.ndarray, P0: float, mode: str = "auto"):
    """
    Best response for Q0 with jammer fixed (Gaussian signaling).
      Γ = H0ᴴ P^{-1} H0 = V diag(σ²) Vᴴ
      Q0* = V diag(p) Vᴴ,  p from water-filling on 1/σ².
    mode='rank1' forces single-stream; otherwise multi-stream.
    Returns (Q0, p, V).
    """
    PinvH0 = chol_inv_apply(P, H0)
    Gram = hermitian(H0.conj().T @ PinvH0)
    s2, V = eigh(Gram)
    idx = np.argsort(s2)[::-1]
    s2, V = s2[idx], V[:, idx]
    sigma = np.sqrt(np.maximum(s2, 0.0))
    with np.errstate(divide="ignore"):
        inv_snr = np.where(sigma > 0, 1.0 / (sigma**2 + 1e-300), np.inf)
    if mode == "rank1":
        p = np.zeros_like(s2)
        p[0] = P0
    else:
        p = _waterfill_power(inv_snr, P0)
    Q0 = hermitian(V @ np.diag(p) @ V.conj().T)
    return Q0, p, V

# ------------------ Q1 update (EG or Mirror-Prox entropy step) ------------------

def mp_entropy_step(Q: np.ndarray, grad: np.ndarray,
                    tau: float, eta: float, eps_log: float = 1e-12) -> np.ndarray:
    """
    One entropy-geometry step for jammer:
      Q⁺ = Π_{tr≤τ}(exp(log Q − η·grad)).
    Implemented spectrally with safe log floor and ≤ trace scaling.
    """
    lam, U = eigh(hermitian(Q))
    lam = np.maximum(lam, 0.0)
    lam = lam + eps_log * (lam <= eps_log)
    LogQ = U @ np.diag(np.log(lam)) @ U.conj().T
    H = hermitian(LogQ - eta * grad)
    lamH, UH = eigh(H)
    Y = UH @ np.diag(np.exp(lamH)) @ UH.conj().T
    tr = np.trace(Y).real
    if tr > tau:
        Y = (tau / tr) * Y
    return hermitian(Y)

# -------------------- Residual (for BR + Q1 update scheme) ----------------------

def stationarity_residual_BRG(H0, H1, Q0, Q1, N0, P0, P1,
                              eta_probe: float = 0.1,
                              geometry: str = "euclidean") -> float:
    """
    Gradient-mapping style residual for BR scheme.
    - ascent on Q0 with Euclidean projection
    - descent on Q1 with either Euclidean or entropy geometry
    """
    G0, G1 = gradients(H0, H1, Q0, Q1, N0)
    R0 = project_psd_trace(Q0 + eta_probe * G0, P0, mode="le") - Q0
    if geometry == "euclidean":
        R1 = project_psd_trace(Q1 - eta_probe * G1, P1, mode="le") - Q1
    else:
        Q1p = mp_entropy_step(Q1, G1, P1, eta_probe)
        R1 = Q1p - Q1
    return max(np.linalg.norm(R0, "fro"), np.linalg.norm(R1, "fro"))

# -------------------------- Main Best-Response solver --------------------------

def solve_game_bestresp_Q0_then_Q1(
    H0: np.ndarray, H1: np.ndarray, N0: float, P0: float, P1: float,
    max_outer: int = 200, tol: float = 1e-6, inner_Q1_steps: int = 2,
    geometry: str = "euclidean",              # 'euclidean' (EG) or 'entropy' (MP)
    step_rule: str = "fixed",                 # 'fixed' or 'adp'
    eta: float = 0.3,                         # used if step_rule='fixed'
    eta_init: float = 0.5, eta_min: float = 1e-3, eta_max: float = 1.0,
    beta: float = 0.5, gamma: float = 1.1, max_bt: int = 10,
    min_outer: int = 3,
    eta_probe: float = 0.1,                   # residual thermometer only
    multi_stream: bool = True, verbose: bool = True, track_hist: bool = True,
    Q1_init: np.ndarray | None = None,
    Q0_init: np.ndarray | None = None,
):
    """
    Alternating scheme:
      (i)  Q0 ← argmax_{tr≤P0} J(Q0, Q1) via water-filling (closed form)
      (ii) Q1 ← EG/MP steps minimizing J (with optional backtracking if step_rule='adp')
    Returns (Q0, Q1, it, hist) if track_hist else (Q0, Q1, it).
    """
    M, N0_tx = H0.shape
    _, N1_tx = H1.shape

    # init
    if Q1_init is None:
        Q1 = np.zeros((N1_tx, N1_tx), dtype=complex)
    else:
        Q1 = project_psd_trace(hermitian(Q1_init), P1, mode="le")

    Q0_prev = project_psd_trace(hermitian(Q0_init), P0, mode="le") if Q0_init is not None else None
    eta_var = float(eta_init if step_rule == "adp" else eta)

    hist = {'J': [], 'errQ0': [], 'errQ1': [], 'residual': [],
            'trQ0': [], 'trQ1': [], 'eta': []} if track_hist else None

    def EG_step_Q1_fixed(Q0_fixed, Q1_cur, eta_use):
        _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
        Q1p = project_psd_trace(Q1_cur - eta_use * g, P1, mode="le")
        _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
        Q1n = project_psd_trace(Q1_cur - eta_use * gh, P1, mode="le")
        return Q1n

    def MP_step_Q1_fixed(Q0_fixed, Q1_cur, eta_use):
        _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
        Q1p = mp_entropy_step(Q1_cur, g, P1, eta_use)
        _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
        Q1n = mp_entropy_step(Q1_cur, gh, P1, eta_use)
        return Q1n

    def EG_step_Q1_bt(Q0_fixed, Q1_cur, eta_use, J_cur=None):
        if J_cur is None:
            J_cur = compute_J(H0, H1, Q0_fixed, Q1_cur, N0)
        eta_try = eta_use
        for _ in range(max_bt):
            _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
            Q1p = project_psd_trace(Q1_cur - eta_try * g, P1, mode="le")
            _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
            Q1n = project_psd_trace(Q1_cur - eta_try * gh, P1, mode="le")
            J_new = compute_J(H0, H1, Q0_fixed, Q1n, N0)
            if J_new <= J_cur + 1e-6:  # jammer minimizes J
                return Q1n, min(max(eta_try * gamma, eta_min), eta_max), J_new
            eta_try = max(eta_min, beta * eta_try)
        return Q1n, eta_try, J_new

    def MP_step_Q1_bt(Q0_fixed, Q1_cur, eta_use, J_cur=None):
        if J_cur is None:
            J_cur = compute_J(H0, H1, Q0_fixed, Q1_cur, N0)
        eta_try = eta_use
        for _ in range(max_bt):
            _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
            Q1p = mp_entropy_step(Q1_cur, g, P1, eta_try)
            _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
            Q1n = mp_entropy_step(Q1_cur, gh, P1, eta_try)
            J_new = compute_J(H0, H1, Q0_fixed, Q1n, N0)
            if J_new <= J_cur + 1e-6:
                return Q1n, min(max(eta_try * gamma, eta_min), eta_max), J_new
            eta_try = max(eta_min, beta * eta_try)
        return Q1n, eta_try, J_new

    is_EG = (geometry == "euclidean")

    for it in range(1, max_outer + 1):
        # (i) Q0 best response — water-filling
        Pmat = N0 * np.eye(M, dtype=complex) + H1 @ Q1 @ H1.conj().T
        mode = "auto" if multi_stream else "rank1"
        Q0, p, V = waterfilling_Q0(H0, Pmat, P0, mode=mode)

        # errQ0
        if Q0_prev is None:
            errQ0 = np.nan
        else:
            denom0 = max(np.linalg.norm(Q0_prev, "fro"), 1.0)
            errQ0 = np.linalg.norm(Q0 - Q0_prev, "fro") / denom0

        # (ii) Q1 inner steps
        Q1_old = Q1.copy()
        J_cur = compute_J(H0, H1, Q0, Q1, N0) if step_rule == "adp" else None

        for _ in range(inner_Q1_steps):
            if step_rule == "fixed":
                Q1 = EG_step_Q1_fixed(Q0, Q1, eta_var) if is_EG else MP_step_Q1_fixed(Q0, Q1, eta_var)
            else:
                if is_EG:
                    Q1, eta_var, J_cur = EG_step_Q1_bt(Q0, Q1, eta_var, J_cur)
                else:
                    Q1, eta_var, J_cur = MP_step_Q1_bt(Q0, Q1, eta_var, J_cur)

        # metrics
        denom1 = max(np.linalg.norm(Q1_old, "fro"), 1.0)
        errQ1 = np.linalg.norm(Q1 - Q1_old, "fro") / denom1
        res = stationarity_residual_BRG(H0, H1, Q0, Q1, N0, P0, P1,
                                        eta_probe=eta_probe, geometry=geometry)
        Jval = compute_J(H0, H1, Q0, Q1, N0)

        if track_hist:
            hist['J'].append(Jval)
            hist['errQ0'].append(errQ0)
            hist['errQ1'].append(errQ1)
            hist['residual'].append(res)
            hist['trQ0'].append(np.trace(Q0).real)
            hist['trQ1'].append(np.trace(Q1).real)
            hist['eta'].append(eta_var)

        if verbose and (it <= 20 or it % 50 == 0 or it == max_outer):
            e0 = f"{errQ0:.3e}" if np.isfinite(errQ0) else "nan"
            print(f"[outer {it}] errQ0={e0}, errQ1={errQ1:.3e}, res={res:.3e}, "
                  f"J={Jval:.4f}, eta={eta_var:.3g}, trQ1={np.trace(Q1).real:.6f}")

        if (it >= min_outer) and (errQ1 < tol) and (res < max(1e-3 * tol, tol)):
            break

        Q0_prev = Q0

    return (Q0, Q1, it, hist) if track_hist else (Q0, Q1, it)
