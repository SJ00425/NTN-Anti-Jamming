# geo_utils.py  –  North-0° / Clockwise azimuth convention
import numpy as np

# ---------- helper: azimuth conversions ----------
def _north_to_east(az_north_deg: float) -> float:
    """
    Convert an azimuth defined with 0° = North, clockwise positive
    to the mathematical convention 0° = East, counter-clockwise positive.
    """
    return (90.0 - az_north_deg) % 360.0


def _east_to_north(az_east_deg: float) -> float:
    """
    Convert an azimuth defined with 0° = East, CCW positive
    back to 0° = North, clockwise positive.
    """
    return (90.0 - az_east_deg) % 360.0


# ---------- 1) surface point along a great-circle arc ----------
def arc_point_on_earth(d_km: float, az_deg: float, earth_radius_km: float = 6371):
    """
    Return the (E, N, U) vector of a point that is `d_km` along
    the Earth’s surface from the origin, in the direction `az_deg`
    (0° = North, clockwise).

    The origin is assumed to be (lat=0°, lon=0°, altitude 0 m).
    Output units are metres.
    """
    R = earth_radius_km
    az_east = np.radians(_north_to_east(az_deg))      # convert to East-0° system
    delta   = d_km / R                                # central angle (rad)

    # Spherical coordinates relative to the origin
    x_sphere = np.sin(delta) * np.cos(az_east)        # East component
    y_sphere = np.sin(delta) * np.sin(az_east)        # North component
    z_sphere = np.cos(delta)                          # Up component

    # Convert to local ENU in metres
    x_local = R * x_sphere * 1e3
    y_local = R * y_sphere * 1e3
    z_local = R * (1 - z_sphere) * 1e3                # “Down” positive

    # Return (East, North, Up)
    return np.array([x_local, y_local, -z_local])


# ---------- 2) ray–sphere intersection with satellite shell ----------
def compute_satellite_intersection_point_enu(
        az_deg: float, el_deg: float, sat_orbit_m: float, tx_pos=None):
    """
    Given azimuth (north-0°, CW) and elevation (0° = horizon, 90° = zenith),
    find the intersection of the ray from `tx_pos` with the spherical
    shell of radius Earth + sat_orbit_m.

    Returns:
        point_enu  – (E, N, U) vector from `tx_pos` to the intersection (m)
        delay_ms   – propagation delay (ms) assuming speed c = 3e8 m/s
        distance_m – geometric distance (m)
    """
    R_e = 6371e3
    sat_r = R_e + sat_orbit_m
    c = 3e8

    # Default TX at (lat=0, lon=0, altitude 0)
    if tx_pos is None:
        tx_pos = np.array([0.0, 0.0, R_e])

    # Direction unit vector in E, N, U
    az_east = np.radians(_north_to_east(az_deg))
    el_rad  = np.radians(el_deg)
    d = np.array([
        np.cos(el_rad) * np.cos(az_east),   # East
        np.cos(el_rad) * np.sin(az_east),   # North
        np.sin(el_rad)                      # Up
    ])

    # Quadratic for ray–sphere intersection: ‖o + t·d‖ = sat_r
    o = tx_pos
    a = np.dot(d, d)
    b = 2 * np.dot(o, d)
    c_quad = np.dot(o, o) - sat_r ** 2
    disc = b * b - 4 * a * c_quad
    if disc < 0:
        raise ValueError("Ray does not intersect satellite shell")

    t1 = (-b - np.sqrt(disc)) / (2 * a)
    t2 = (-b + np.sqrt(disc)) / (2 * a)
    t  = min(t for t in (t1, t2) if t > 0)  # first positive root

    point_ecef = o + t * d
    point_enu  = point_ecef - tx_pos
    dist_m     = np.linalg.norm(point_enu)
    delay_ms   = dist_m / c * 1e3
    return point_enu, delay_ms, dist_m


# ---------- 3) azimuth, elevation, distance between two points ----------
def compute_az_el_dist(sat_pos, gnd_pos, frequency_hz: float | None = None):
    """
    Given satellite and ground positions in the same Cartesian frame
    (E, N, U or ECEF), return:

        az_deg : azimuth (0° = North, clockwise)
        el_deg : elevation
        dist   : distance in metres
        n_waves: number of wavelengths (optional, if frequency_hz given)
    """
    vec  = sat_pos - gnd_pos
    dist = np.linalg.norm(vec)
    dx, dy, dz = vec / dist                            # normalised vector (E, N, U)

    el_rad = np.arcsin(dz)
    az_east = np.arctan2(dy, dx)                       # 0° = East, CCW
    az_deg  =np.degrees(az_east)%360
    # az_deg  = _east_to_north(np.degrees(az_east))      # convert back to North-0°

    el_deg = np.degrees(el_rad)

    if frequency_hz is not None:
        wavelength = 3e8 / frequency_hz
        n_waves = dist / wavelength
        return az_deg, el_deg, dist, n_waves
    return az_deg, el_deg, dist
