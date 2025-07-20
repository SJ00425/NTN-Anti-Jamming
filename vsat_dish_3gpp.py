import numpy as np
import drjit as dr
import mitsuba as mi
from scipy.special import jv
from sionna.rt.antenna_pattern import PolarizedAntennaPattern, register_antenna_pattern

def v_vsat_dish_pattern(theta: mi.Float,
                        phi: mi.Float,
                        *,
                        frequency_hz: float = 10e9,
                        aperture_radius_lambda: float = 10.0,
                        g_peak_dBi: float = 40.0,
                        back_supp_db: float = 30.0) -> mi.Complex2f:
    """
    VSAT vertically-polarized antenna pattern based on 3GPP TR 38.811.
    Pattern: (2J1(u)/u)^2, where u = (2πa/λ)·sin(α), α = angle from +x axis.
    """
    c = 3e8
    lam = c / frequency_hz
    a = aperture_radius_lambda * lam

    # Convert input to numpy
    theta_np = theta.numpy() if hasattr(theta, "numpy") else np.array(theta)
    phi_np = phi.numpy() if hasattr(phi, "numpy") else np.array(phi)

    # Compute angle from main beam (x-axis)
    ray_x = np.sin(theta_np) * np.cos(phi_np)
    alpha = np.arccos(np.clip(ray_x, -1.0, 1.0))

    # Compute u = (2πa/λ)·sin(α)
    u = 2 * np.pi * a * np.sin(alpha) / lam
    u_safe = np.where(np.abs(u) < 1e-10, 1e-10, u)

    # Bessel-based gain pattern
    j1_u = jv(1, u_safe)
    pattern = (2 * j1_u / u_safe) ** 2
    pattern = np.where(np.abs(u) < 1e-10, 1.0, pattern)

    # Normalize gain
    max_gain_lin = 10 ** (g_peak_dBi / 10)
    pattern *= max_gain_lin

    # Backlobe suppression
    pattern = np.where(alpha > (np.pi / 2), pattern * 10 ** (-back_supp_db / 10), pattern)

    # Convert to field amplitude
    field_amp = np.sqrt(pattern)
    return mi.Complex2f(mi.Float(field_amp), mi.Float(0.0))

# ---------- Sionna Integration ----------
def _vsat_factory(*, polarization: str,
                  polarization_model: str = "tr38901_2",
                  frequency_hz: float = 10e9):
    return PolarizedAntennaPattern(
        v_pattern=lambda theta, phi: v_vsat_dish_pattern(theta, phi, frequency_hz=frequency_hz),
        polarization=polarization,
        polarization_model=polarization_model
    )

register_antenna_pattern("vsat_dish", _vsat_factory)
