
# Unified NTN covariance min-max game solvers:
# - Mirror-Prox (entropy geometry, trace equality)
# - Min-max (water-filling BR on Q0 + EG/MP on Q1, trace ≤)
# - Proximal Best-Response (trace ≤ with proximal regularization)
# - Extragradient (projected EG with nonmonotone backtracking)
# - PDHG (Condat–Vũ / Chambolle–Pock style)
#
# Problem (informal):
#   Max over desired covariance Q0 ⪰ 0 (tr(Q0) ≤ P0)
#   Min over jammer  covariance Q1 ⪰ 0 (tr(Q1) ≤ P1)
#   of
#       J(Q0,Q1) = log2 det(N0 I + H1 Q1 H1^H + H0 Q0 H0^H)
#                 - log2 det(N0 I + H1 Q1 H1^H)
#   = C_with - C_without (Shannon spectral efficiency gain in bits/s/Hz)
#
# Gradients (matrix calculus): let P := N0 I + H1 Q1 H1^H, S := H0 Q0 H0^H, A := P + S.
#   d/dQ0 J = H0^H A^{-1} H0  (ascent direction)
#   d/dQ1 J = H1^H (A^{-1} - P^{-1}) H1  (descent direction)
# which follow from d log det(X) = tr(X^{-1} dX).
#
# Projection operators used throughout:
#   1) Euclidean projection onto {Q ⪰ ε I, tr(Q) ≤ τ} or {tr(Q) = τ} via eigen-decomp
#   2) Entropic (matrix exponential) projection onto {Q ⪰ 0, tr(Q) = τ}


import numpy as np
from numpy.linalg import cholesky, solve, eigh


# ============================================================
# Common utilities (de-duplicated across implementations)
# ============================================================

__all__ = [

    # Mirror-Prox (trace equality, entropy geometry)
     "solve_game_mirror_prox",
    # Min-max BR+EG/MP (trace ≤)
    "solve_game_bestresp_Q0_then_Q1",
    # Proximal Best-Response (trace ≤)
    "solve_game_proxBR",
    # Extra gradient
    "solve_game_extragradient", 
    # Primal dual hyprid gradient
    "solve_game_pdhg",
    
    "solve_game_proxBR_pp"
    
    "compute_J"
]


def hermitian(X: np.ndarray) -> np.ndarray:
    """Return Hermitian part of X: (X + Xᴴ)/2. Ensures numerical Hermiticity."""
    return 0.5 * (X + X.conj().T)


def chol_inv_apply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A^{-1} B for Hermitian positive-(semi)definite A via Cholesky.

    Robust variant with a tiny diagonal jitter ε I if needed.
    Complexity: one Cholesky + two triangular solves.
    """
    M = A.shape[0]
    try:
        L = cholesky(A)
    except np.linalg.LinAlgError:
        # ε ≍ 1e-12 * tr(A)/n stabilizes nearly singular A without biasing scale
        eps = 1e-12 * np.trace(A).real / max(M, 1)
        L = cholesky(A + eps * np.eye(M, dtype=complex))
    Y = solve(L, B)                # L Y = B
    return solve(L.conj().T, Y)    # Lᴴ X = Y ⇒ X = A^{-1}B


def gradients(H0: np.ndarray, H1: np.ndarray,
              Q0: np.ndarray, Q1: np.ndarray,
              N0: float):
    """Compute the matrix gradients (∇_{Q0}J, ∇_{Q1}J).

    Let P = N0 I + H1 Q1 H1ᴴ, S = H0 Q0 H0ᴴ, A = P + S.
    Using d log det(X) = tr(X^{-1} dX):
        ∇_{Q0} J = H0ᴴ A^{-1} H0  (ascent)
        ∇_{Q1} J = H1ᴴ (A^{-1} - P^{-1}) H1  (descent)
    """
    M = H0.shape[0]
    P = N0 * np.eye(M, dtype=complex) + H1 @ Q1 @ H1.conj().T
    S = H0 @ Q0 @ H0.conj().T
    Ainv = chol_inv_apply(P + S, np.eye(M, dtype=complex))
    Pinv = chol_inv_apply(P,       np.eye(M, dtype=complex))
    G0 = hermitian(H0.conj().T @ Ainv @ H0)               # ascent
    G1 = hermitian(H1.conj().T @ (Ainv - Pinv) @ H1)      # descent
    return G0, G1


def compute_J(H0: np.ndarray, H1: np.ndarray,
              Q0: np.ndarray, Q1: np.ndarray,
              N0: float) -> float:
    """Compute J(Q0,Q1) = log2 det(P+S) − log2 det(P).

    Here P = N0 I + H1 Q1 H1ᴴ and S = H0 Q0 H0ᴴ. We use a Cholesky-based
    log-det: if A = L Lᴴ, then log det(A) = 2 * Σ log diag(L).
    The 1/log(2) converts to bits.
    """
    M = H0.shape[0]
    P = N0 * np.eye(M, dtype=complex) + H1 @ Q1 @ H1.conj().T
    S = H0 @ Q0 @ H0.conj().T
    A = P + S
    B  = 100e6

    def _logdet(mat: np.ndarray) -> float:
        try:
            L = cholesky(mat)
        except np.linalg.LinAlgError:
            eps = 1e-12 * np.trace(mat).real / max(M, 1)
            L = cholesky(mat + eps * np.eye(M, dtype=complex))
        return 2.0 * np.sum(np.log(np.abs(np.diag(L))))

    return ((_logdet(A) - _logdet(P)) / np.log(2.0))


def project_psd_trace(
    Z: np.ndarray,
    tau: float,
    eps_floor: float = 0.0,
    mode: str = 'le'               # 'le' (≤ tau) or 'eq' (= tau)
) -> np.ndarray:
    """
    Euclidean projection onto the convex set
        C_{≤} := { Q ⪰ eps_floor·I,  tr(Q) ≤ τ }
        C_{=} := { Q ⪰ eps_floor·I,  tr(Q) = τ }.

    Reduction to eigenvalues: if Z = U diag(λ) Uᴴ, then the solution is
        X* = U diag(x*) Uᴴ
    where x* solves a *boxed simplex projection* with lower bound eps_floor.

    Parameterization x = eps_floor + y, y ≥ 0 gives
        sum_i x_i = τ  ⇔  sum_i y_i = τ − n·eps_floor.

    For mode='le': return λ clipped to ≥ eps_floor if already feasible (Σ λ⁺ ≤ τ).
    Otherwise (or for mode='eq'), project to the equality simplex using the
    Duchi–Shalev-Shwartz–Singer–Chandra (2008) algorithm.

    Returns X* Hermitian.
    """
    Zh = hermitian(Z)
    lam, U = eigh(Zh)                      # ascending eigenvalues
    n = lam.size

    # helper: project y >= 0 onto simplex {sum y = z}
    def _proj_simplex_nonneg(y: np.ndarray, z: float) -> np.ndarray:
        # Duchi et al. (JMLR 2008) O(n log n) projection
        if z <= 0:
            return np.zeros_like(y)
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - z))[0]
        if len(rho) == 0:
            # all-zero corner; safeguard balance
            theta = (cssv[-1] - z) / n
            w = np.maximum(y - theta, 0.0)
            s = w.sum()
            return w if s == z or s == 0 else (z/s) * w
        r = rho[-1] + 1
        theta = (cssv[r-1] - z) / r
        return np.maximum(y - theta, 0.0)

    # enforce eigenvalue floor via variable substitution:
    # x = eps_floor + y, with y >= 0 and sum x = tau  => sum y = tau - n*eps_floor
    lam = np.asarray(lam, dtype=float)
    lam_f = np.maximum(lam, eps_floor)
    base_sum = lam_f.sum()

    # quick return if feasible for ≤
    if mode == 'le' and base_sum <= tau + 1e-12:
        x = lam_f
    else:
        # equality target on the "y" variables
        z = tau - n * eps_floor
        if z < 0:
            # infeasible target; best we can do is all at floor
            x = np.full_like(lam, eps_floor)
        else:
            y = lam - eps_floor              # can be negative, simplex will clamp to 0
            y_proj = _proj_simplex_nonneg(y, z)
            x = eps_floor + y_proj

        if mode == 'le':
            # numerical safety: if after projection we slightly overshoot, renormalize
            s = x.sum()
            if s > tau + 1e-12:
                y = x - eps_floor
                y = _proj_simplex_nonneg(y, tau - n*eps_floor)
                x = eps_floor + y

    X = hermitian(U @ np.diag(x) @ U.conj().T)
    return X



def kkt_residual_proj(H0, H1, Q0, Q1, N0, P0, P1,
                      eta_probe: float = 0.2,
                      mode: str = 'eq',
                      rho: float | None = None,
                      norm: str = 'fro'):
    """
    Gradient-mapping / Prox-gradient residual for saddle problems.

    modes:
      - 'le'   : projected one-step residual with trace-≤ projection
      - 'eq'   : projected one-step residual with trace-= projection
      - 'both' : returns (res_eq, res_le)
      - 'prox' : PROX-GRADIENT residual (primary merit for true convergence)
                 uses step size 'rho' (>0). If rho is None, fall back to eta_probe.
    """
    G0, G1 = gradients(H0, H1, Q0, Q1, N0)

    # -------- prox-gradient residual (true merit) --------
    if mode == 'prox':
        if rho is None:
            rho = float(eta_probe)  # 合理缺省
        assert rho > 0.0, "rho must be > 0 for mode='prox'"

        # Q0 是“max”块：朝 +G0 方向做近端一步
        Z0 = project_psd_trace(Q0 + rho * G0, P0, mode='le')
        # Q1 是“min”块：朝 -G1 方向做近端一步
        Z1 = project_psd_trace(Q1 - rho * G1, P1, mode='le')

        R0 = (Q0 - Z0) / rho
        R1 = (Q1 - Z1) / rho
        return float(np.sqrt(np.linalg.norm(R0, norm)**2 + np.linalg.norm(R1, norm)**2))

    # -------- 原有的一步投影残差（KKT surrogate）--------
    # ≤ residual
    Q0p_le = project_psd_trace(Q0 + eta_probe*G0, P0, mode='le')
    Q1p_le = project_psd_trace(Q1 - eta_probe*G1, P1, mode='le')
    R0_le  = (Q0p_le - Q0) / eta_probe
    R1_le  = (Q1p_le - Q1) / eta_probe
    res_le = max(np.linalg.norm(R0_le, norm), np.linalg.norm(R1_le, norm))

    # = residual
    Q0p_eq = project_psd_trace(Q0 + eta_probe*G0, P0, mode='eq')
    Q1p_eq = project_psd_trace(Q1 - eta_probe*G1, P1, mode='eq')
    R0_eq  = (Q0p_eq - Q0) / eta_probe
    R1_eq  = (Q1p_eq - Q1) / eta_probe
    res_eq = max(np.linalg.norm(R0_eq, norm), np.linalg.norm(R1_eq, norm))

    if mode == 'both':
        return res_eq, res_le
    return res_eq if mode == 'eq' else res_le






# ============================================================
# Mirror-Prox solver (trace equality, entropy geometry)
# ============================================================

def _safe_floor(X: np.ndarray) -> float:
    """Tiny relative eigenvalue floor ε ≍ 1e-10·tr(X)/n for stable logm."""
    n = X.shape[0]
    return 1e-10 * max(np.trace(hermitian(X)).real, 1.0) / max(n, 1)


def logm_psd(X: np.ndarray, eps_floor: float = None) -> np.ndarray:
    """Matrix logarithm for (approximately) PSD X via eigendecomposition.

    If X = U diag(λ) Uᴴ with λ_i ≥ 0, we compute log X = U diag(log max(λ_i, ε)) Uᴴ.
    The ε floor avoids −∞ when eigenvalues are near zero.
    """
    Xh = hermitian(X)
    lam, U = eigh(Xh)  # ascending
    if eps_floor is None:
        eps_floor = _safe_floor(Xh)
    lam = np.maximum(lam, eps_floor)
    return U @ np.diag(np.log(lam)) @ U.conj().T


def expm_herm_centered(Y: np.ndarray):
    """
    Stable matrix exponential for Hermitian Y using spectral shifting:
        Y = U diag(λ) Uᴴ, m = max_i λ_i.
        exp(Y) = U diag(exp(λ - m)) Uᴴ * exp(m).
    Returns (X, scale) where X = exp(Y)/exp(m) has trace in a safe numeric range.
    """
    Yh = hermitian(Y)
    lam, U = eigh(Yh)
    m = float(np.max(lam))
    e = np.exp(lam - m)
    X = U @ np.diag(e) @ U.conj().T
    return X, np.exp(m)


def normalize_trace_to(X: np.ndarray, tau: float) -> np.ndarray:
    """Scale X so that tr(X) = τ. If tr(X) ≤ 0, return τ·I/n as a fallback."""
    t = np.trace(X).real
    if t <= 0:
        n = X.shape[0]
        return tau * np.eye(n, dtype=complex) / n
    return (tau / t) * X


def entropic_proj_from_dual(Y: np.ndarray, tau: float) -> np.ndarray:
    """Entropy-geometry projection: Π_{tr=τ}(exp(Y)).

    This is the Bregman projection for the von Neumann entropy mirror map
    ψ(Q) = tr(Q log Q − Q), whose gradient is ∇ψ(Q) = log Q.
    """
    Xshift, _ = expm_herm_centered(Y)
    X = normalize_trace_to(Xshift, tau)
    return hermitian(X)


def stationarity_residual_mirror(H0, H1, Q0, Q1, N0, P0, P1, eta_probe=0.1):
    """Residual measured in the entropy geometry (log-domain steps)."""
    G0, G1 = gradients(H0, H1, Q0, Q1, N0)
    L0, L1 = logm_psd(Q0), logm_psd(Q1)
    Q0p = entropic_proj_from_dual(L0 + eta_probe * G0, P0)   # ascent
    Q1p = entropic_proj_from_dual(L1 - eta_probe * G1, P1)   # descent
    return max(np.linalg.norm(Q0p - Q0, 'fro'),
               np.linalg.norm(Q1p - Q1, 'fro'))


def solve_game_mirror_prox(
    H0, H1, N0, P0, P1,
    steps=3000,
    eta=0.25,                   # fixed step in log-domain
    step_rule='fixed',          # kept for API symmetry
    beta=0.7, gamma=1.03,       # unused in fixed mode
    eta_min=1e-3, eta_max=1.0,  # unused in fixed mode
    tol=1e-6, min_steps=15,
    strong_reg=0.0,             # ascent: +μQ0, descent: -μQ1
    Q0_init=None, Q1_init=None,
    use_averaging=True,         # tail averaging
    eta_probe=0.1,              # residual thermometer
    verbose=False, track_hist=True,
    progress=None,            # callable: progress(i, total, metrics:dict, ctx:dict) -> bool(stop?)
    progress_every=1,         # 每多少步调用一次
    progress_ctx=None         # 传上下文信息：方法名/时刻/k/卫星名等
):
    """
    Fixed-step Mirror-Prox (Nemirovski, 2004) under entropy geometry (trace equality).

    Mirror map ψ(Q) = tr(Q log Q − Q) ⇒ ∇ψ(Q) = log Q,  ∇ψ* = exp and normalization.
    Predictor-corrector in dual (log) domain:
        L0 = log Q0,  L1 = log Q1
        \hat Q0 = Π_{tr=P0}(exp(L0 + η ∇_{Q0}J)),   \hat Q1 = Π_{tr=P1}(exp(L1 − η ∇_{Q1}J))
        Q0⁺ = Π_{tr=P0}(exp(L0 + η ∇_{Q0}J(\hat Q))),   Q1⁺ analogously.

    Returns (Q0, Q1, hist) where hist has J / errQ0 / errQ1 / residual / eta / trQ0 / trQ1
    """
    n = H0.shape[1]
    Q0 = (P0/n) * np.eye(n, dtype=complex) if Q0_init is None else hermitian(Q0_init)
    Q1 = (P1/n) * np.eye(n, dtype=complex) if Q1_init is None else hermitian(Q1_init)

    hist = {'J': [], 'errQ0': [], 'errQ1': [], 'residual': [], 'eta': [],
            'trQ0': [], 'trQ1': []} if track_hist else None

    # averaging accumulators
    Q0_acc = np.zeros_like(Q0); Q1_acc = np.zeros_like(Q1); acc_cnt = 0
    burn_in = steps // 3

    for k in range(1, steps+1):
        Q0 = hermitian(Q0); Q1 = hermitian(Q1)
        Q0_old = Q0.copy(); Q1_old = Q1.copy()

        # predictor (dual/log domain)
        G0, G1 = gradients(H0, H1, Q0, Q1, N0)
        if strong_reg > 0:
            G0 = G0 + strong_reg*Q0      # ascent side
            G1 = G1 - strong_reg*Q1      # descent side

        L0, L1  = logm_psd(Q0), logm_psd(Q1)
        Q0_half = entropic_proj_from_dual(L0 + eta*G0, P0)
        Q1_half = entropic_proj_from_dual(L1 - eta*G1, P1)

        # corrector
        G0h, G1h = gradients(H0, H1, Q0_half, Q1_half, N0)
        if strong_reg > 0:
            G0h = G0h + strong_reg*Q0_half
            G1h = G1h - strong_reg*Q1_half

        Q0_new = entropic_proj_from_dual(L0 + eta*G0h, P0)
        Q1_new = entropic_proj_from_dual(L1 - eta*G1h, P1)

        # metrics
        errQ0 = np.linalg.norm(Q0_new - Q0_old, 'fro') / max(np.linalg.norm(Q0_old,'fro'), 1.0)
        errQ1 = np.linalg.norm(Q1_new - Q1_old, 'fro') / max(np.linalg.norm(Q1_old,'fro'), 1.0)
        Jval  = compute_J(H0, H1, Q0_new, Q1_new, N0)
        res   = stationarity_residual_mirror(H0, H1, Q0_new, Q1_new, N0, P0, P1, eta_probe=eta_probe)

        if track_hist:
            hist['J'].append(Jval)
            hist['errQ0'].append(errQ0); hist['errQ1'].append(errQ1)
            hist['residual'].append(res); hist['eta'].append(eta)
            hist['trQ0'].append(np.trace(Q0_new).real); hist['trQ1'].append(np.trace(Q1_new).real)

        if verbose and (k <= 10 or k % 50 == 0 or k == steps):
            print(f"[{k:04d}] J={Jval:.4f}, errQ0={errQ0:.2e}, errQ1={errQ1:.2e}, "
                  f"res={res:.2e}, eta={eta:.3g}, trQ0={hist['trQ0'][-1]:.5f}, trQ1={hist['trQ1'][-1]:.5f}")

        if use_averaging and k >= burn_in:
            Q0_acc += Q0_new; Q1_acc += Q1_new; acc_cnt += 1

        Q0, Q1 = Q0_new, Q1_new
        
        if progress and (k == 1 or (k % progress_every == 0) or k == steps):
            stop = progress(
                i=k, total=steps,
                metrics={
                    "J": float(np.real(Jval)),
                    "residual": float(res),
                    "errQ0": float(errQ0),
                    "errQ1": float(errQ1),
                    "eta":float(eta), 
                },
                ctx=progress_ctx or {}
            )
            if stop: break

        if (k >= min_steps) and (max(errQ0, errQ1) < tol) and (res < max(1e-3*tol, tol)):
            break

    if use_averaging and acc_cnt > 0:
        Q0_out = hermitian(Q0_acc/acc_cnt); Q1_out = hermitian(Q1_acc/acc_cnt)
    else:
        Q0_out, Q1_out = Q0, Q1

    return (Q0_out, Q1_out, hist) if track_hist else (Q0_out, Q1_out)


# ============================================================
# water-MP solver (BR on Q0 + EG/MP on Q1, trace ≤)
# ============================================================


def _waterfill_power(inv_snr: np.ndarray, Ptot: float) -> np.ndarray:
    """Classical water-filling: p_i = max(μ − a_i, 0) to meet Σ p_i = Ptot.

    Here a_i are inverse (effective) SNRs; μ is chosen so that the power constraint holds.
    We compute μ via the cumulative-sum trick on sorted a_i.
    """
    a = np.array(inv_snr, dtype=float)
    a_sort = np.sort(a)
    csum   = np.cumsum(a_sort)
    mu_cand= (Ptot + csum) / (np.arange(1, a.size+1))
    idx = np.where(mu_cand > a_sort)[0]
    if len(idx)==0:
        mu = mu_cand[0]; p = np.maximum(mu - a, 0.0)
        s = p.sum();  p = p if s==0 else p * (Ptot/s)
        return p
    k  = idx[-1] + 1
    mu = (Ptot + csum[k-1]) / k
    p  = np.maximum(mu - a, 0.0)
    s  = p.sum();  p = p if s==0 else p * (Ptot/s)
    return p


def waterfilling_Q0(H0: np.ndarray, P: np.ndarray, P0: float, mode: str = 'auto'):
    """Best response for Q0 with jammer held fixed (Gaussian signaling).

    Effective Gram = H0ᴴ P^{-1} H0. If Γ = V diag(σ²) Vᴴ with σ² ≥ 0, then the
    optimal Q0 aligns with V and allocates powers p via water-filling on 1/σ².

        P := N0 I + H1 Q1 H1ᴴ (held fixed here)
        Γ := H0ᴴ P^{-1} H0
        Q0* = V diag(p) Vᴴ,  p = WF(1/σ²; P0)

    mode='rank1' optionally forces a single-stream solution.
    Returns (Q0, p, V).
    """
    PinvH0 = chol_inv_apply(P, H0)
    Gram   = hermitian(H0.conj().T @ PinvH0)
    s2, V  = eigh(Gram)
    idx    = np.argsort(s2)[::-1]
    s2, V  = s2[idx], V[:, idx]
    sigma  = np.sqrt(np.maximum(s2, 0.0))
    with np.errstate(divide='ignore'):
        inv_snr = np.where(sigma>0, 1.0/(sigma**2+1e-300), np.inf)
    if mode=='rank1':
        p = np.zeros_like(s2); p[0] = P0
    else:
        p = _waterfill_power(inv_snr, P0)
    Q0 = hermitian(V @ np.diag(p) @ V.conj().T)
    active = p > 1e-12

    return Q0, p, V


def mp_entropy_step(Q: np.ndarray, grad: np.ndarray,
                    tau: float, eta: float, eps_log: float = 1e-12) -> np.ndarray:
    """One entropy-geometry step: Q⁺ = Π_{tr≤τ}(exp(log Q − η·grad)).

    Implemented spectrally: log Q via eigendecomp (with small floor), then exponentiate
    and rescale to satisfy tr ≤ τ by a simple global factor if needed.
    """
    lam, U = eigh(hermitian(Q))
    lam = np.maximum(lam, 0.0)
    lam = lam + eps_log*(lam <= eps_log)
    LogQ = U @ np.diag(np.log(lam)) @ U.conj().T
    H = hermitian(LogQ - eta*grad)
    lamH, UH = eigh(H)
    Y = UH @ np.diag(np.exp(lamH)) @ UH.conj().T
    tr = np.trace(Y).real
    if tr > tau: Y = (tau/tr)*Y     # implement ≤ tau
    return hermitian(Y)

    # """
    # 稳定版熵几何一步：在特征域做 softmax，避免 exp 溢出，且强制 tr=τ。
    # """
    # Qh  = hermitian(Q)
    # Gh  = hermitian(grad)

    # # “对偶前进”后直接特征分解
    # lam, U = np.linalg.eigh(Qh - eta*Gh)

    # # 中心化 + 截断，彻底防溢出/下溢
    # lam = lam - np.max(lam)
    # lam = np.clip(lam, -60.0, 60.0)

    # w  = np.exp(lam)
    # sw = np.sum(w)
    # if (not np.isfinite(sw)) or sw <= 0:
    #     n = lam.size
    #     return tau * np.eye(n, dtype=complex) / max(n,1)

    # s = (tau / sw) * w
    # Q_next = U @ np.diag(s) @ U.conj().T

    # # 极小抖动，保证后续 Cholesky 稳定
    # eps = 1e-12 * float(tau) / max(Q.shape[0], 1)
    # return hermitian(Q_next) + eps * np.eye(Q.shape[0], dtype=complex)



def stationarity_residual_BRG(H0, H1, Q0, Q1, N0, P0, P1,
                          eta_probe: float = 0.1,
                          geometry: str = 'euclidean') -> float:
    """Residual for the BR + Q1-updates scheme (EG or MP geometry)."""
    G0, G1 = gradients(H0, H1, Q0, Q1, N0)
    R0 = project_psd_trace(Q0 + eta_probe*G0, P0, mode='le') - Q0      # ascent
    if geometry=='euclidean':
        R1 = project_psd_trace(Q1 - eta_probe*G1, P1, mode='le') - Q1  # descent
    else:
        Q1p = mp_entropy_step(Q1, G1, P1, eta_probe)
        R1  = Q1p - Q1
    return max(np.linalg.norm(R0,'fro'), np.linalg.norm(R1,'fro'))


def solve_game_bestresp_Q0_then_Q1(
    H0, H1, N0, P0, P1,
    max_outer=200, tol=1e-6, inner_Q1_steps=2,
    geometry='euclidean',                # 'euclidean' (EG) or 'entropy' (Mirror-Prox)
    step_rule='fixed',                   # 'fixed' or 'adp'
    eta=0.3,                             # used if step_rule='fixed'
    eta_init=0.5, eta_min=1e-3, eta_max=1.0, beta=0.5, gamma=1.1, max_bt=10,
    min_outer=3,                         # avoid early stop at outer=1
    eta_probe=0.1,                       # for residual "thermometer" only
    multi_stream=True, verbose=True, track_hist=True,
    Q1_init=None,                        # jammer init
    Q0_init=None,                         # desired init (optional)
    progress=None,            # callable: progress(i, total, metrics:dict, ctx:dict) -> bool(stop?)
    progress_every=1,         # 每多少步调用一次
    progress_ctx=None         # 传上下文信息：方法名/时刻/k/卫星名等
    ):
    """
    Alternating scheme:
      (i)  Q0 ← argmax_{tr≤P0, Q0⪰0} J(Q0, Q1) via water-filling (closed form)
      (ii) Update Q1 with inner steps minimizing J: either Euclidean EG or entropy MP.

    Step-rule 'adp' (adaptive): backtracking on J non-increase for the jammer.
    """
    M, N0_tx = H0.shape
    _, N1_tx = H1.shape

    # --- initialize Q1 (jammer) ---
    if Q1_init is None:
        Q1 = np.zeros((N1_tx, N1_tx), dtype=complex)
    else:
        Q1 = project_psd_trace(hermitian(Q1_init), P1, mode='le')

    # --- optional record of previous Q0 for errQ0 ---
    Q0_prev = project_psd_trace(hermitian(Q0_init), P0, mode='le') if Q0_init is not None else None
    if step_rule == 'adp':
        eta_var = float(eta_init)
    else:
        eta_var = float(eta)

    hist = {'J': [], 'errQ0': [], 'errQ1': [], 'residual': [], 'trQ0': [], 'trQ1': [], 'eta': []} if track_hist else None

    # --- inner updaters (fixed-eta) ---
    def EG_step_Q1_fixed(Q0_fixed, Q1_cur, eta_use):
        _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
        Q1p = project_psd_trace(Q1_cur - eta_use*g, P1, mode='le')            # predict
        _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
        Q1n = project_psd_trace(Q1_cur - eta_use*gh, P1, mode='le')           # correct
        return Q1n

    def MP_step_Q1_fixed(Q0_fixed, Q1_cur, eta_use):
        _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
        Q1p = mp_entropy_step(Q1_cur, g, P1, eta_use)                 # predict
        _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
        Q1n = mp_entropy_step(Q1_cur, gh, P1, eta_use)                # correct
        return Q1n

    # --- inner updaters (adaptive eta with backtracking on J non-increase) ---
    def EG_step_Q1_bt(Q0_fixed, Q1_cur, eta_use, J_cur=None):
        if J_cur is None:
            J_cur = compute_J(H0, H1, Q0_fixed, Q1_cur, N0)
        eta_try = eta_use
        for _ in range(max_bt):
            _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
            Q1p  = project_psd_trace(Q1_cur - eta_try*g, P1, mode='le')
            _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
            Q1n  = project_psd_trace(Q1_cur - eta_try*gh, P1, mode='le')
            J_new = compute_J(H0, H1, Q0_fixed, Q1n, N0)
            if J_new <= J_cur + 1e-6:  # jammer minimizes J
                return Q1n, min(max(eta_try*gamma, eta_min), eta_max), J_new
            eta_try = max(eta_min, beta*eta_try)
        # fallback: accept last
        return Q1n, eta_try, J_new

    def MP_step_Q1_bt(Q0_fixed, Q1_cur, eta_use, J_cur=None):
        if J_cur is None:
            J_cur = compute_J(H0, H1, Q0_fixed, Q1_cur, N0)
        eta_try = eta_use
        for _ in range(max_bt):
            _, g = gradients(H0, H1, Q0_fixed, Q1_cur, N0)
            Q1p  = mp_entropy_step(Q1_cur, g, P1, eta_try)
            _, gh = gradients(H0, H1, Q0_fixed, Q1p, N0)
            Q1n  = mp_entropy_step(Q1_cur, gh, P1, eta_try)
            J_new = compute_J(H0, H1, Q0_fixed, Q1n, N0)
            if J_new <= J_cur + 1e-6:
                return Q1n, min(max(eta_try*gamma, eta_min), eta_max), J_new
            eta_try = max(eta_min, beta*eta_try)
        return Q1n, eta_try, J_new

    # choose kernels
    is_EG = (geometry == 'euclidean')

    for it in range(1, max_outer+1):
        # --- best-response on Q0 (water-filling) ---
        Pmat = N0*np.eye(M, dtype=complex) + H1 @ Q1 @ H1.conj().T
        mode = 'auto' if multi_stream else 'rank1'
        Q0, p, V = waterfilling_Q0(H0, Pmat, P0, mode=mode)

        # errQ0
        if Q0_prev is None:
            errQ0 = np.nan
        else:
            denom0 = max(np.linalg.norm(Q0_prev, 'fro'), 1.0)
            errQ0 = np.linalg.norm(Q0 - Q0_prev, 'fro') / denom0

        # --- Q1 inner steps ---
        Q1_old = Q1.copy()
        J_cur  = compute_J(H0, H1, Q0, Q1, N0) if step_rule=='adp' else None

        for _ in range(inner_Q1_steps):
            if step_rule == 'fixed':
                if is_EG:
                    Q1 = EG_step_Q1_fixed(Q0, Q1, eta_var)
                else:
                    Q1 = MP_step_Q1_fixed(Q0, Q1, eta_var)
            else:  # adaptive eta
                if is_EG:
                    Q1, eta_var, J_cur = EG_step_Q1_bt(Q0, Q1, eta_var, J_cur)
                else:
                    Q1, eta_var, J_cur = MP_step_Q1_bt(Q0, Q1, eta_var, J_cur)

        # errQ1
        denom1 = max(np.linalg.norm(Q1_old, 'fro'), 1.0)
        errQ1 = np.linalg.norm(Q1 - Q1_old, 'fro') / denom1

        # residual (use gradient mapping with a fixed probe)
        res  = stationarity_residual_BRG(H0, H1, Q0, Q1, N0, P0, P1,
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
            print(f"[outer {it}] errQ0={e0}, errQ1={errQ1:.3e}, res={res:.3e}, J={Jval:.4f}, eta={eta_var:.3g}, trQ1={hist['trQ1'][-1]:.6f}")
        # --- progress callback（与 MP 版一致） ---
        if progress and (it == 1 or (it % progress_every == 0) or it == max_outer):
            stop = progress(
                i=it, total=max_outer,
                metrics={
                    "J": float(np.real(Jval)),
                    "residual": float(res),
                    "errQ0": (float(errQ0) if np.isfinite(errQ0) else float('nan')),
                    "errQ1": float(errQ1),
                    "eta": float(eta_var),
                },
                ctx=progress_ctx or {}
            )
            if stop:
                break
        # stop: use errQ1 + residual; guard with min_outer
        if (it >= min_outer) and (errQ1 < tol) and (res < max(1e-3*tol, tol)):
            break

        Q0_prev = Q0

    return (Q0, Q1, it, hist) if track_hist else (Q0, Q1, it)


# ============================================================
# Proximal Best-Response solver (trace ≤, prox regularization)
# ============================================================





def _proxBR_Q0(H0, H1, Q0k, Q1k, N0, P0, rho,
               inner_max=200, inner_tol=1e-6,
               eta0=0.25, beta=0.5, gamma=1.1, eta_min=1e-3, eta_max=1.0, proj_mode='le'):
    """
    Ascent-side prox-BR on
        f(Q0) = J(Q0, Q1k) − (ρ/2) ||Q0 − Q0k||_F^2
    using projected gradient ascent with backtracking. Accept if f increases.
    """
    Q0 = Q0k.copy()
    eta = float(eta0)
    f_old = compute_J(H0, H1, Q0, Q1k, N0) - 0.5*rho*np.linalg.norm(Q0-Q0k,'fro')**2
    for _ in range(inner_max):
        G0, _ = gradients(H0, H1, Q0, Q1k, N0)
        grad  = G0 - rho*(Q0 - Q0k)                # ascent gradient
        # backtracking
        tried = False
        for __ in range(20):
            Q0_new = project_psd_trace(Q0 + eta*grad, P0, mode=proj_mode)
            f_new = compute_J(H0, H1, Q0_new, Q1k, N0) - 0.5*rho*np.linalg.norm(Q0_new-Q0k,'fro')**2
            if f_new + 1e-12 >= f_old:             # accept and mild enlarge
                tried = True
                if np.linalg.norm(Q0_new - Q0, 'fro') < inner_tol:
                    Q0 = Q0_new
                    return Q0
                Q0, f_old = Q0_new, f_new
                eta = min(max(eta*gamma, eta_min), eta_max)
                break
            eta = max(eta*beta, eta_min)           # shrink
        if not tried:
            # fallback: accept last
            Q0 = Q0_new
            break
    return hermitian(Q0)


def _proxBR_Q1(H0, H1, Q0k1, Q1k, N0, P1, rho,
               inner_max=200, inner_tol=1e-6,
               eta0=0.25, beta=0.5, gamma=1.1, eta_min=1e-3, eta_max=1.0, proj_mode='le'):
    """
    Descent-side prox-BR on
        g(Q1) = J(Q0k+1, Q1) + (ρ/2) ||Q1 − Q1k||_F^2
    using projected gradient descent with backtracking. Accept if g decreases.
    """
    Q1 = Q1k.copy()
    eta = float(eta0)
    g_old = compute_J(H0, H1, Q0k1, Q1, N0) + 0.5*rho*np.linalg.norm(Q1-Q1k,'fro')**2
    for _ in range(inner_max):
        _, G1 = gradients(H0, H1, Q0k1, Q1, N0)
        grad  = G1 + rho*(Q1 - Q1k)                # descent gradient
        # backtracking
        tried = False
        for __ in range(20):
            Q1_new = project_psd_trace(Q1 - eta*grad, P1, mode=proj_mode)
            g_new = compute_J(H0, H1, Q0k1, Q1_new, N0) + 0.5*rho*np.linalg.norm(Q1_new-Q1k,'fro')**2
            if g_new <= g_old + 1e-12:             # accept and mild enlarge
                tried = True
                if np.linalg.norm(Q1_new - Q1, 'fro') < inner_tol:
                    Q1 = Q1_new
                    return Q1
                Q1, g_old = Q1_new, g_new
                eta = min(max(eta*gamma, eta_min), eta_max)
                break
            eta = max(eta*beta, eta_min)
        if not tried:
            Q1 = Q1_new
            break
    return hermitian(Q1)


def solve_game_proxBR(
    H0, H1, N0, P0, P1,
    rho=1e-2,
    outer_steps=300, outer_tol=1e-6,
    inner_max=200, inner_tol=1e-6,
    eta0_inner=0.25, beta=0.5, gamma=1.1, eta_min=1e-3, eta_max=1.0,
    Q0_init=None, Q1_init=None,
    eta_probe=0.2, min_outer=5,
    verbose=True, track_hist=True
):
    """
    Proximal Best-Response (PBR) iterations:
        Q0^{k+1} ≈ argmax_{tr≤P0} J(Q0, Q1^k) − (ρ/2)||Q0 − Q0^k||²
        Q1^{k+1} ≈ argmin_{tr≤P1} J(Q0^{k+1}, Q1) + (ρ/2)||Q1 − Q1^k||²
    Both subproblems solved by projected gradient with backtracking line search.

    The proximal terms stabilize oscillations and can promote uniqueness of the per-block
    solution maps when ρ is sufficiently large.
    """
    N = H0.shape[1]
    Q0 = np.zeros((N, N), dtype=complex) if Q0_init is None else Q0_init.copy()
    Q1 = np.zeros((N, N), dtype=complex) if Q1_init is None else Q1_init.copy()
    hist = {'J': [], 'errQ0': [], 'errQ1': [], 'residual': [], 'trQ0': [], 'trQ1': []} if track_hist else None

    for k in range(1, outer_steps+1):
        Q0 = hermitian(Q0); Q1 = hermitian(Q1)
        Q0_old = Q0.copy(); Q1_old = Q1.copy()

        # inner prox-BR updates
        Q0_new = _proxBR_Q0(H0, H1, Q0, Q1, N0, P0, rho,
                            inner_max=inner_max, inner_tol=inner_tol,
                            eta0=eta0_inner, beta=beta, gamma=gamma,
                            eta_min=eta_min, eta_max=eta_max)
        Q1_new = _proxBR_Q1(H0, H1, Q0_new, Q1, N0, P1, rho,
                            inner_max=inner_max, inner_tol=inner_tol,
                            eta0=eta0_inner, beta=beta, gamma=gamma,
                            eta_min=eta_min, eta_max=eta_max)

        # relative steps
        errQ0 = np.linalg.norm(Q0_new - Q0_old, 'fro') / max(np.linalg.norm(Q0_old, 'fro'), 1.0)
        errQ1 = np.linalg.norm(Q1_new - Q1_old, 'fro') / max(np.linalg.norm(Q1_old, 'fro'), 1.0)

        # metrics
        Jval = compute_J(H0, H1, Q0_new, Q1_new, N0)
        res  = kkt_residual_proj(H0, H1, Q0_new, Q1_new, N0, P0, P1, eta_probe=eta_probe, mode='le')

        if track_hist:
            hist['J'].append(Jval)
            hist['errQ0'].append(errQ0)
            hist['errQ1'].append(errQ1)
            hist['residual'].append(res)
            hist['trQ0'].append(np.trace(Q0_new).real)
            hist['trQ1'].append(np.trace(Q1_new).real)

        if verbose and (k % 10 == 0 or k <= 5 or k == outer_steps):
            print(f"[{k:03d}] J={Jval:.4f}, errQ0={errQ0:.2e}, errQ1={errQ1:.2e}, res={res:.2e}, trQ1={np.trace(Q1_new).real:.4f}")

        Q0, Q1 = Q0_new, Q1_new

        # stopping
        if (k >= min_outer) and (max(errQ0, errQ1) < outer_tol) and (res < max(1e-3*outer_tol, outer_tol)):
            break

    return (Q0, Q1, hist) if track_hist else (Q0, Q1)

# ============================================================
# Proximal Best-Response solver2 (trace ≤, prox regularization)
# ============================================================

# ---------- 1) 块 prox-gradient 范数（用于相对误差规则） ----------
def _block_pg_norm_Q0(H0, H1, Q0, Q1, N0, P0, eta=0.1, proj_mode='le'):
    G0, _ = gradients(H0, H1, Q0, Q1, N0)
    Z0    = project_psd_trace(Q0 + eta*G0, P0, mode=proj_mode)  # max块：上升
    return np.linalg.norm((Q0 - Z0)/eta, 'fro')

def _block_pg_norm_Q1(H0, H1, Q0, Q1, N0, P1, eta=0.1, proj_mode='le'):
    _, G1 = gradients(H0, H1, Q0, Q1, N0)
    Z1    = project_psd_trace(Q1 - eta*G1, P1, mode=proj_mode)  # min块：下降
    return np.linalg.norm((Q1 - Z1)/eta, 'fro')

# ---------- 2) 相对误差包装器：按需要“加严”内层 ----------
def _solve_Q0_to_relative_error(Q0k, Q1k, target_pg, H0,H1,N0,P0,rho,
                                inner_max, inner_tol,
                                eta0,beta,gamma,eta_min,eta_max,
                                pg_eta=0.1, proj_mode='le', max_refines=3):
    Q0_new = _proxBR_Q0(H0,H1,Q0k,Q1k,N0,P0,rho,
                        inner_max=inner_max, inner_tol=inner_tol,
                        eta0=eta0,beta=beta,gamma=gamma,
                        eta_min=eta_min,eta_max=eta_max, proj_mode=proj_mode)
    ref = 0
    while ref < max_refines:
        pg = _block_pg_norm_Q0(H0,H1,Q0_new,Q1k,N0,P0,eta=pg_eta,proj_mode=proj_mode)
        if pg <= target_pg:
            break
        # 加严
        inner_tol = max(inner_tol*0.2, 1e-12)
        inner_max = int(inner_max*1.5)+5
        Q0_new = _proxBR_Q0(H0,H1,Q0k,Q1k,N0,P0,rho,
                            inner_max=inner_max, inner_tol=inner_tol,
                            eta0=eta0,beta=beta,gamma=gamma,
                            eta_min=eta_min,eta_max=eta_max)
        ref += 1
    return hermitian(Q0_new)

def _solve_Q1_to_relative_error(Q0k1, Q1k, target_pg, H0,H1,N0,P1,rho,
                                inner_max, inner_tol,
                                eta0,beta,gamma,eta_min,eta_max,
                                pg_eta=0.1, proj_mode='le', max_refines=3):
    Q1_new = _proxBR_Q1(H0,H1,Q0k1,Q1k,N0,P1,rho,
                        inner_max=inner_max, inner_tol=inner_tol,
                        eta0=eta0,beta=beta,gamma=gamma,
                        eta_min=eta_min,eta_max=eta_max, proj_mode=proj_mode)
    ref = 0
    while ref < max_refines:
        pg = _block_pg_norm_Q1(H0,H1,Q0k1,Q1_new,N0,P1,eta=pg_eta,proj_mode=proj_mode)
        if pg <= target_pg:
            break
        # 加严
        inner_tol = max(inner_tol*0.2, 1e-12)
        inner_max = int(inner_max*1.5)+5
        Q1_new = _proxBR_Q1(H0,H1,Q0k1,Q1k,N0,P1,rho,
                            inner_max=inner_max, inner_tol=inner_tol,
                            eta0=eta0,beta=beta,gamma=gamma,
                            eta_min=eta_min,eta_max=eta_max)
        ref += 1
    return hermitian(Q1_new)

# ---------- 3) 全新：带 ρ-回溯 + 相对误差 的 ProxBR 外层 ----------
def solve_game_proxBR_pp(
    H0, H1, N0, P0, P1,
    rho=5e-2, outer_steps=300, outer_tol=1e-6,
    inner_max=200, inner_tol=1e-6,
    eta0_inner=0.25, beta=0.5, gamma=1.05, eta_min=1e-3, eta_max=0.3,
    Q0_init=None, Q1_init=None,
    min_outer=5,
    sigma=0.3, tau=1e-2, grow=2.0, lam=1.0,
    pg_eta=0.1, proj_mode='le',
    rho_max=1e3, freeze_backtrack_at=1e-4,  # 新增：ρ上限及末期冻结阈值
    verbose=True, track_hist=True
):
    """ProxBR + 相对误差 + ρ回溯（含上限/冻结）"""
    N = H0.shape[1]
    Q0 = np.zeros((N, N), dtype=complex) if Q0_init is None else Q0_init.copy()
    Q1 = np.zeros((N, N), dtype=complex) if Q1_init is None else Q1_init.copy()
    hist = None
    if track_hist:
        hist = {'J': [], 'res_prox': [], 'res_eq': [], 'res_le': [],
                'errQ0': [], 'errQ1': [], 'trQ0': [], 'trQ1': [], 'rho': []}

    # 自适应 τ
    def adaptive_tau(g_old, base_tau=tau, g_thresh=1e-2):
        if g_old >= g_thresh:
            return base_tau
        return base_tau * (g_old / g_thresh)

    for k in range(1, outer_steps + 1):
        Q0, Q1 = hermitian(Q0), hermitian(Q1)
        Q0_old, Q1_old = Q0.copy(), Q1.copy()

        # 粗略候选
        Q0_tent = _solve_Q0_to_relative_error(Q0, Q1, np.inf,
                                              H0, H1, N0, P0, rho,
                                              inner_max, inner_tol,
                                              eta0_inner, beta, gamma, eta_min, eta_max,
                                              pg_eta, proj_mode)
        Q1_tent = _solve_Q1_to_relative_error(Q0_tent, Q1, np.inf,
                                              H0, H1, N0, P1, rho,
                                              inner_max, inner_tol,
                                              eta0_inner, beta, gamma, eta_min, eta_max,
                                              pg_eta, proj_mode)

        dz = np.sqrt(np.linalg.norm(Q0_tent - Q0, 'fro')**2 +
                     np.linalg.norm(Q1_tent - Q1, 'fro')**2)
        target_pg = sigma * dz / max(rho, 1e-12)

        # 精化候选
        Q0_new = _solve_Q0_to_relative_error(Q0, Q1, target_pg,
                                             H0, H1, N0, P0, rho,
                                             inner_max, inner_tol,
                                             eta0_inner, beta, gamma, eta_min, eta_max,
                                             pg_eta, proj_mode)
        Q1_new = _solve_Q1_to_relative_error(Q0_new, Q1, target_pg,
                                             H0, H1, N0, P1, rho,
                                             inner_max, inner_tol,
                                             eta0_inner, beta, gamma, eta_min, eta_max,
                                             pg_eta, proj_mode)

        Q0_c = hermitian((1 - lam) * Q0 + lam * Q0_new)
        Q1_c = hermitian((1 - lam) * Q1 + lam * Q1_new)

        g_old = kkt_residual_proj(H0, H1, Q0, Q1, N0, P0, P1,
                                  mode='prox', rho=rho, eta_probe=rho)
        g_new = kkt_residual_proj(H0, H1, Q0_c, Q1_c, N0, P0, P1,
                                  mode='prox', rho=rho, eta_probe=rho)

        # ρ回溯（加上上限与冻结）
        tries = 0
        while g_new > max((1 - tau) * g_old, 1e-14) and tries < 4:
            if g_old < freeze_backtrack_at or rho >= rho_max:
                break
            rho = min(rho * grow, rho_max)
            tau_eff = adaptive_tau(g_old, base_tau=tau)
            Q0_c = _solve_Q0_to_relative_error(Q0, Q1, target_pg,
                                               H0, H1, N0, P0, rho,
                                               inner_max, inner_tol,
                                               eta0_inner, beta, gamma, eta_min, eta_max,
                                               pg_eta, proj_mode)
            Q1_c = _solve_Q1_to_relative_error(Q0_c, Q1, target_pg,
                                               H0, H1, N0, P1, rho,
                                               inner_max, inner_tol,
                                               eta0_inner, beta, gamma, eta_min, eta_max,
                                               pg_eta, proj_mode)
            g_new = kkt_residual_proj(H0, H1, Q0_c, Q1_c, N0, P0, P1,
                                      mode='prox', rho=rho, eta_probe=rho)
            tries += 1

        Q0, Q1 = hermitian(Q0_c), hermitian(Q1_c)

        # 记录
        errQ0 = np.linalg.norm(Q0 - Q0_old, 'fro') / max(np.linalg.norm(Q0_old, 'fro'), 1.0)
        errQ1 = np.linalg.norm(Q1 - Q1_old, 'fro') / max(np.linalg.norm(Q1_old, 'fro'), 1.0)
        Jval = compute_J(H0, H1, Q0, Q1, N0)
        res_prox = g_new
        if track_hist:
            hist['J'].append(float(Jval))
            hist['res_prox'].append(float(res_prox))
            res_eq = kkt_residual_proj(H0, H1, Q0, Q1, N0, P0, P1,
                                       mode='eq', eta_probe=pg_eta)
            res_le = kkt_residual_proj(H0, H1, Q0, Q1, N0, P0, P1,
                                       mode='le', eta_probe=pg_eta)
            hist['res_eq'].append(float(res_eq))
            hist['res_le'].append(float(res_le))
            hist['errQ0'].append(float(errQ0))
            hist['errQ1'].append(float(errQ1))
            hist['trQ0'].append(float(np.trace(Q0).real))
            hist['trQ1'].append(float(np.trace(Q1).real))
            hist['rho'].append(float(rho))

        if verbose and (k % 10 == 0 or k <= 5 or k == outer_steps):
            print(f"[{k:03d}] J={Jval:.4f}, ||G||_prox={res_prox:.2e}, "
                  f"errQ0={errQ0:.2e}, errQ1={errQ1:.2e}, rho={rho:.3g}")

        if k >= min_outer and res_prox < outer_tol:
            break

    return (Q0, Q1, hist) if track_hist else (Q0, Q1)



# ============================================================
# Extragradient Solver (with optional nonmonotone backtracking)
# ============================================================

def solve_game_extragradient(
    H0, H1, N0, P0, P1,
    steps: int = 3000,
    eta: float = 0.25,
    step_rule: str = 'fixed',          # 'fixed' | 'adp_res' | 'adp_bal'
    beta: float = 0.7, gamma: float = 1.02,
    eta_min: float = 1e-3, eta_max: float = 1.0,
    tol: float = 1e-6, min_steps: int = 10,
    strong_reg: float = 0.0,           # ascent: -μQ0, descent: +μQ1
    equal_trace: bool = False,         # update uses ≤ by default; True -> equality projection
    eta_probe: float = 0.2,            # residual probe step
    Q0_init=None, Q1_init=None,
    residual_ref: str = 'eq',          # 'eq' | 'le' | 'both'
    window: int = 10,                  # nonmonotone window
    boost_period: int = 10,            # periodic mild growth
    verbose: bool = False, track_hist: bool = True
):
    """
    Projected Extragradient (Korpelevich) on the saddle dynamics with trace constraints.

    One iteration:
        Predict:  \hat Q = Π(Q + η G(Q)) with ascent on Q0 and descent on Q1
        Correct:  Q⁺      = Π(Q + η G(\hat Q))
    with Π either trace-≤ or trace-= projection (Euclidean spectral projection).

    Optional nonmonotone backtracking accepts steps if a residual surrogate does not
    increase beyond the rolling window maximum.
    """
    mode = 'eq' if equal_trace else 'le'
    proj = lambda Z, tau: project_psd_trace(Z, tau, mode=mode)
    N = H0.shape[1]
    Q0 = (np.zeros((N,N), complex) if Q0_init is None else hermitian(Q0_init.copy()))
    Q1 = (np.zeros((N,N), complex) if Q1_init is None else hermitian(Q1_init.copy()))

    hist = {'J': [], 'errQ0': [], 'errQ1': [], 'residual': [], 'res_eq': [], 'res_le': [],
            'eta': [], 'trQ0': [], 'trQ1': []} if track_hist else None

    # initial residual
    if residual_ref == 'both':
        res_eq0, res_le0 = kkt_residual_proj(H0,H1,Q0,Q1,N0,P0,P1,eta_probe,mode='both')
        res_prev = res_eq0
    else:
        res_prev = kkt_residual_proj(H0,H1,Q0,Q1,N0,P0,P1,eta_probe,mode=residual_ref)
    res_buf = [res_prev]
    J_prev  = compute_J(H0, H1, Q0, Q1, N0)

    for k in range(1, steps+1):
        Q0 = hermitian(Q0); Q1 = hermitian(Q1)
        Q0_old = Q0.copy(); Q1_old = Q1.copy()

        # predict
        G0, G1 = gradients(H0, H1, Q0, Q1, N0)
        if strong_reg > 0:
            G0 = G0 - strong_reg*Q0
            G1 = G1 + strong_reg*Q1
        Q0_half = proj(Q0 + eta*G0, P0)
        Q1_half = proj(Q1 - eta*G1, P1)

        # correct
        G0h, G1h = gradients(H0, H1, Q0_half, Q1_half, N0)
        if strong_reg > 0:
            G0h = G0h - strong_reg*Q0_half
            G1h = G1h + strong_reg*Q1_half

        # inner backtracking (nonmonotone via window max)
        eta_try = float(eta)
        accept  = False
        res_ref_threshold = max(res_buf)
        for _ in range(15):
            Q0_new = proj(Q0 + eta_try*G0h, P0)
            Q1_new = proj(Q1 - eta_try*G1h, P1)
            if residual_ref == 'both':
                res_eq, res_le = kkt_residual_proj(H0,H1,Q0_new,Q1_new,N0,P0,P1,eta_probe,mode='both')
                res_new = res_eq
            else:
                res_new = kkt_residual_proj(H0,H1,Q0_new,Q1_new,N0,P0,P1,eta_probe,mode=residual_ref)
            if (step_rule == 'fixed') or (res_new <= res_ref_threshold + 1e-12):
                accept = True
                break
            eta_try = max(eta_min, eta_try*beta)
        if not accept:
            # accept last
            Q0_new = proj(Q0 + eta_try*G0h, P0)
            Q1_new = proj(Q1 - eta_try*G1h, P1)
            if residual_ref == 'both':
                res_eq, res_le = kkt_residual_proj(H0,H1,Q0_new,Q1_new,N0,P0,P1,eta_probe,mode='both')
                res_new = res_eq
            else:
                res_new = kkt_residual_proj(H0,H1,Q0_new,Q1_new,N0,P0,P1,eta_probe,mode=residual_ref)

        # metrics
        errQ0 = np.linalg.norm(Q0_new - Q0_old, 'fro') / max(np.linalg.norm(Q0_old,'fro'), 1.0)
        errQ1 = np.linalg.norm(Q1_new - Q1_old, 'fro') / max(np.linalg.norm(Q1_old,'fro'), 1.0)
        Jval  = compute_J(H0, H1, Q0_new, Q1_new, N0)

        if residual_ref == 'both':
            res_eq, res_le = kkt_residual_proj(H0,H1,Q0_new,Q1_new,N0,P0,P1,eta_probe,mode='both')
            if track_hist:
                hist['res_eq'].append(res_eq); hist['res_le'].append(res_le)
        if track_hist:
            hist['J'].append(Jval)
            hist['errQ0'].append(errQ0); hist['errQ1'].append(errQ1)
            hist['residual'].append(res_new); hist['eta'].append(eta_try)
            hist['trQ0'].append(np.trace(Q0_new).real); hist['trQ1'].append(np.trace(Q1_new).real)

        if verbose and (k <= 10 or k % 50 == 0 or k == steps):
            if residual_ref == 'both':
                print(f"[{k:04d}] J={Jval:.4f}, errQ0={errQ0:.2e}, errQ1={errQ1:.2e}, "
                      f"res_eq={hist['res_eq'][-1]:.2e}, res_le={hist['res_le'][-1]:.2e}, "
                      f"eta={eta_try:.3g}, trQ0={hist['trQ0'][-1]:.5f}, trQ1={hist['trQ1'][-1]:.5f}")
            else:
                print(f"[{k:04d}] J={Jval:.4f}, errQ0={errQ0:.2e}, errQ1={errQ1:.2e}, "
                      f"res={res_new:.2e}, eta={eta_try:.3g}, trQ0={hist['trQ0'][-1]:.5f}, trQ1={hist['trQ1'][-1]:.5f}")

        # outer adaptation
        if step_rule in ('adp_res', 'adp_bal'):
            res_buf.append(res_new)
            if len(res_buf) > window:
                res_buf.pop(0)
            if (k % boost_period) == 0:
                if res_new <= min(res_buf) * 0.995:
                    eta = min(eta_max, eta * gamma)
                else:
                    eta = max(eta_min, eta * beta)

        Q0, Q1 = hermitian(Q0_new), hermitian(Q1_new)
        res_prev, J_prev = res_new, Jval

        if (k >= min_steps) and (max(errQ0, errQ1) < tol) and (res_new < max(1e-3*tol, tol)):
            break

    return (Q0, Q1, hist) if track_hist else (Q0, Q1)


# ============================================================
# PDHG / Primal-Dual Hybrid Gradient (Condat–Vũ / Chambolle–Pock style)
# ============================================================

def solve_game_pdhg(
    H0, H1, N0, P0, P1,
    steps: int = 3000,
    tau: float = 0.05, sigma: float = 0.05, theta: float = 0.5,
    step_rule: str = 'fixed',              # 'fixed' or 'adp'
    beta: float = 0.7, gamma: float = 1.05,
    tau_min: float = 1e-4, tau_max: float = 0.2,
    sigma_min: float = 1e-4, sigma_max: float = 0.2,
    tol: float = 1e-6, strong_reg: float = 0.0,
    Q0_init=None, Q1_init=None,
    use_averaging: bool = True,
    eta_probe: float = 0.2,
    min_steps: int = 10,
    verbose: bool = False, track_hist: bool = True
):
    """
    PDHG-like splitting for min-max with smooth coupling J and simple projections.

    Template:
        Q1^{k+1} = Π_{tr≤P1}(Q1^k − σ ∇_{Q1}J(Q0^k, Q1^k))            (dual/min step)
        \bar Q1  = Q1^{k+1} + θ (Q1^{k+1} − Q1^k)                      (extrapolation)
        Q0^{k+1} = Π_{tr≤P0}(Q0^k + τ ∇_{Q0}J(Q0^k, \bar Q1))          (primal/max step)

    Heuristic adaptive rule (step_rule='adp'): expand (τ,σ) when J↓ and residual↓.
    """
    N = H0.shape[1]
    Q0 = np.zeros((N,N), complex) if Q0_init is None else Q0_init.copy()
    Q1 = np.zeros((N,N), complex) if Q1_init is None else Q1_init.copy()

    if use_averaging:
        Q0_avg = np.zeros_like(Q0); Q1_avg = np.zeros_like(Q1); avg_cnt = 0

    hist = {'J': [], 'errQ0': [], 'errQ1': [], 'residual': [],
            'tau': [], 'sigma': [], 'trQ0': [], 'trQ1': []} if track_hist else None

    J_prev = compute_J(H0, H1, Q0, Q1, N0)
    res_prev = kkt_residual_proj(H0, H1, Q0, Q1, N0, P0, P1, eta_probe=eta_probe, mode='le')

    for k in range(1, steps+1):
        Q0 = hermitian(Q0); Q1 = hermitian(Q1)
        Q0_old = Q0.copy(); Q1_old = Q1.copy()

        # dual (min) step on Q1
        G0k, G1k = gradients(H0, H1, Q0, Q1, N0)
        if strong_reg > 0:
            G1k = G1k + strong_reg * Q1
        Q1_next = project_psd_trace(Q1 - sigma*G1k, P1, mode='le')
        Q1_bar  = Q1_next + theta*(Q1_next - Q1)

        # primal (max) step on Q0 using extrapolated dual
        G0bar, _ = gradients(H0, H1, Q0, Q1_bar, N0)
        if strong_reg > 0:
            G0bar = G0bar - strong_reg * Q0
        Q0_next = project_psd_trace(Q0 + tau*G0bar, P0, mode='le')

        # metrics
        errQ0 = np.linalg.norm(Q0_next - Q0_old, 'fro') / max(np.linalg.norm(Q0_old,'fro'), 1.0)
        errQ1 = np.linalg.norm(Q1_next - Q1_old, 'fro') / max(np.linalg.norm(Q1_old,'fro'), 1.0)
        Jval  = compute_J(H0, H1, Q0_next, Q1_next, N0)
        res   = kkt_residual_proj(H0, H1, Q0_next, Q1_next, N0, P0, P1, eta_probe=eta_probe, mode='le')

        if track_hist:
            hist['J'].append(Jval); hist['errQ0'].append(errQ0); hist['errQ1'].append(errQ1)
            hist['residual'].append(res); hist['tau'].append(tau); hist['sigma'].append(sigma)
            hist['trQ0'].append(np.trace(Q0_next).real); hist['trQ1'].append(np.trace(Q1_next).real)

        if verbose and (k <= 10 or k % 50 == 0 or k == steps):
            print(f"[{k:04d}] J={Jval:.4f}, errQ0={errQ0:.2e}, errQ1={errQ1:.2e}, res={res:.2e}, "
                  f"tau={tau:.3g}, sigma={sigma:.3g}, trQ0={hist['trQ0'][-1]:.5f}, trQ1={hist['trQ1'][-1]:.5f}")

        if step_rule == 'adp' and k > 5:
            improve = (Jval <= J_prev + 1e-12) and (res <= 0.99*res_prev + 1e-12)
            if improve:
                tau   = min(tau * gamma,  tau_max)
                sigma = min(sigma * gamma, sigma_max)
            else:
                tau   = max(tau * beta,   tau_min)
                sigma = max(sigma * beta, sigma_min)
            T = 10
            if (k % T) == 0:
                improve = (Jval <= J_prev + 1e-12) and (res <= 0.995*res_prev + 1e-12)
                if improve:
                    mu = 1.03
                    tau   = min(tau * mu,  tau_max)
                    sigma = min(sigma * mu, sigma_max)

        Q0, Q1 = Q0_next, Q1_next
        J_prev, res_prev = Jval, res

        if use_averaging and k >= steps//3:
            Q0_avg += Q0; Q1_avg += Q1; avg_cnt += 1

        if (k >= min_steps) and (max(errQ0, errQ1) < tol) and (res < max(1e-3*tol, tol)):
            break

    if use_averaging and avg_cnt > 0:
        Q0_out = hermitian(Q0_avg/avg_cnt); Q1_out = hermitian(Q1_avg/avg_cnt)
    else:
        Q0_out, Q1_out = Q0, Q1

    return (Q0_out, Q1_out, hist) if track_hist else (Q0_out, Q1_out)
