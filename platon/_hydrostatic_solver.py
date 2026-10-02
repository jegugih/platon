import mlx.core as mx
import numpy as np

from . import _mlx
from .constants import k_B, AMU, M_sun, Teff_sun, G
from .errors import AtmosphereError

# The hydrostatic solution involves only ~100 numbers, so it is computed in
# NumPy: MLX's fixed cost per operation (~5 us, vs ~0.5 us in NumPy) would
# dominate.  It has to be float64 anyway (which Metal GPUs lack), because
# radii are ~1e7 m while shells are ~1e4 m thick.


def _solve(P_profile, T_profile, ref_pressure, mu_profile,
           planet_mass, planet_radius, star_radius,
           num_above_cloud, T_star=None):
    '''P_profile must be increasing.  The profiles may be NumPy or MLX
    arrays, and mu_profile may be a single value.  Returns float64 MLX
    arrays of the radii and dr for the first num_above_cloud layers.'''
    P = np.asarray(_mlx.to_numpy(P_profile), dtype=float)
    T = np.asarray(_mlx.to_numpy(T_profile), dtype=float)
    mu = np.broadcast_to(np.asarray(_mlx.to_numpy(mu_profile), dtype=float), P.shape)
    assert(len(P) == len(T))
    if np.any(np.diff(P) <= 0):
        raise ValueError("Pressures in the P/T profile must be increasing")

    # Ensure that the atmosphere is bound by making rough estimates of the
    # Hill radius and atmospheric height
    if T_star is None:
        T_star = Teff_sun

    R_hill = star_radius * \
        (T_star / T[0])**2 * (planet_mass / (3 * M_sun))**(1.0 / 3)
    max_r_estimate = 1.0 / (1 / planet_radius + k_B * np.median(T) * np.log(
        P[0] / ref_pressure) / (G * planet_mass * np.mean(mu) * AMU))
    # The above equation is negative if the real answer is infinity

    if max_r_estimate < 0 or max_r_estimate > R_hill:
        raise AtmosphereError("Atmosphere unbound: height > hill radius")

    # Integrate d(1/r) = k_B T / (G M mu AMU) d(ln P) from ref_pressure, where
    # r = planet_radius, using each layer's mean T and mu.  ref_pressure is
    # inserted into the profile, so a single cumulative sum covers the layers
    # above and below it.
    split = int(np.sum(P <= ref_pressure))
    ln_P = np.log(P)
    ln_Q = np.concatenate([ln_P[:split], [np.log(ref_pressure)], ln_P[split:]])
    T_Q = np.interp(ln_Q, ln_P, T)
    mu_Q = np.interp(ln_Q, ln_P, mu)
    d_inv_r = np.diff(ln_Q) * k_B * (T_Q[1:] + T_Q[:-1]) / 2 / \
        (G * planet_mass * (mu_Q[1:] + mu_Q[:-1]) / 2 * AMU)
    assert(np.all(d_inv_r >= 0))
    integral = np.concatenate([[0.0], np.cumsum(d_inv_r)])
    inv_r = 1.0 / planet_radius + (integral - integral[split])

    radii = 1.0 / np.delete(inv_r, split)
    radii = radii[:num_above_cloud]
    dr = -np.diff(radii)
    return mx.array(radii, dtype=mx.float64), mx.array(dr, dtype=mx.float64)
