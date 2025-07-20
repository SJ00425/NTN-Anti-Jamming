# NTN-Anti-Jamming


## `channel-det-test.ipynb`

This notebook analyzes and simulates the communication channel between a ground-based dish antenna and multiple LEO satellites. It includes both theoretical analysis and Sionna-based simulation.
- **Custom Dish Antenna Pattern**  
  A user-defined dish antenna radiation pattern is implemented and registered into the Sionna framework.

- **Free-Space Path Loss (FSPL) vs. Azimuth and Elevation**  
  FSPL is computed as a function of satellite azimuth and elevation angles.

- **Delay vs. Elevation**  
  Signal propagation delay from the ground dish to LEO satellites is analyzed against elevation angle.

- **SNR vs. Azimuth and Elevation (Theoretical)**  
  Theoretical SNR at the ground dish is calculated as a function of azimuth and elevation, based on antenna gain and FSPL.

- **Sionna Simulation: Ground Dish to LEO Satellites**  
  Simulated SNR performance for a ground dish communicating with three LEO satellites using the Sionna.

## `vsat_dish_3gpp.py`

Implements a VSAT dish antenna model using the 3GPP radiation pattern.  
This module can be directly imported and used within the Sionna environment.

---

## `starlink_tracker.ipynb`

Tracks Starlink satellites from a fixed location near Boulder at a specific timestamp.  

**Note**: Real-time tracking is not displayed within the notebook.

**Note**: Excessive pull requests from Celestrak within a short time may result in temporary IP blocking.

## `geo_utils.py`

Utilities for satellite-ground geometry using North-0° / clockwise azimuth convention.

- `arc_point_on_earth(d_km, az_deg)`:  
  Returns the ENU position of a point `d_km` away along azimuth `az_deg` on Earth's surface.

- `compute_satellite_intersection_point_enu(az_deg, el_deg, sat_orbit_m, tx_pos=None)`:  
  Computes where a ray (from azimuth/elevation) intersects a satellite shell. Returns ENU vector, delay, and distance.

- `compute_az_el_dist(sat_pos, gnd_pos, frequency_hz=None)`:  
  Returns azimuth, elevation, and distance between two points; optionally returns number of wavelengths.

Internally handles azimuth conversion between North-0° (CW) and East-0° (CCW).


