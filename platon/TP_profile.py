import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from scipy.special import expn

from . import _mlx
from .constants import h, c, k_B, AMU, G
from .params import NUM_LAYERS, MIN_P, MAX_P

class Profile:
    # Profiles involve only ~100 values, so they are computed in float64
    # NumPy: MLX's fixed cost per operation would dominate.  Parameters are
    # converted to NumPy scalars, so that invalid values give inf/nan
    # temperatures (which a retrieval rejects) rather than exceptions.
    def __init__(self):
        self.pressures = np.logspace(
                np.log10(MIN_P),
                np.log10(MAX_P),
                NUM_LAYERS)

    def get_temperatures(self):
        return self.temperatures

    def get_pressures(self):
        return self.pressures
        
    def set_from_params_dict(self, profile_type, params_dict):
        if profile_type == "isothermal":
            self.set_isothermal(params_dict["T"])
        elif profile_type == "parametric":
            self.set_parametric(
                params_dict["T0"], 10**params_dict["log_P1"],
                params_dict["alpha1"], params_dict["alpha2"],
                10**params_dict["log_P3"], params_dict["T3"])
        elif profile_type == "radiative_solution":
            self.set_from_radiative_solution(**params_dict)
        else:
            assert(False)
                                        
        
    def set_from_arrays(self, P_profile, T_profile):
        P_profile = np.asarray(P_profile, dtype=float)
        T_profile = np.asarray(T_profile, dtype=float)
        self.temperatures = np.interp(np.log10(self.pressures), np.log10(P_profile), T_profile)

    def set_isothermal(self, T_day):
        self.temperatures = np.ones(len(self.pressures)) * np.float64(T_day)

    def set_parametric(self, T0, P1, alpha1, alpha2, P3, T3):
        '''Parametric model from https://arxiv.org/pdf/0910.1347.pdf'''
        T0, P1, alpha1, alpha2, P3, T3 = map(np.float64, (T0, P1, alpha1, alpha2, P3, T3))
        P0 = np.amin(self.pressures)

        ln_P2 = alpha2**2*(T0+np.log(P1/P0)**2/alpha1**2 - T3) - np.log(P1)**2 + np.log(P3)**2
        ln_P2 /= 2 * np.log(P3/P1)
        P2 = np.exp(ln_P2)
        T2 = T3 - np.log(P3/P2)**2/alpha2**2

        P = self.pressures
        self.temperatures = np.where(
            P < P1, T0 + np.log(P/P0)**2 / alpha1**2,
            np.where(P < P3, T2 + np.log(P/P2)**2 / alpha2**2, T3))
        return P2, T2

    def set_from_opacity(self, T_irr, info_dict, visible_cutoff=0.8e-6, T_int=100):
        T_irr, T_int = np.float64(T_irr), np.float64(T_int)
        visible_cutoff = _mlx.as_float(visible_cutoff)
        wavelengths = _mlx.asarray(info_dict["unbinned_wavelengths"])
        d_lambda = mx.diff(wavelengths)
        d_lambda = mx.concatenate([d_lambda[0:1], d_lambda])

        # Convert stellar spectrum from photons/time to energy/time
        stellar_spectrum = _mlx.asarray(info_dict["stellar_spectrum"]) * h * c / wavelengths

        # Convert planetary spectrum from energy/time/wavelength to energy/time
        planet_spectrum = _mlx.asarray(info_dict["planet_spectrum"]) * d_lambda
        absorption_coeffs = _mlx.asarray(info_dict["absorption_coeff_atm"])

        # Equation 49 here: https://arxiv.org/pdf/1006.4702.pdf
        # The averages over wavelength are large, so they are done in MLX,
        # masking the weights instead of selecting columns
        visible = wavelengths < visible_cutoff
        thermal = wavelengths >= visible_cutoff
        n = np.asarray(info_dict["P_profile"])/k_B/np.asarray(info_dict["T_profile"])
        sigmas = absorption_coeffs / _mlx.asarray(n)[:, np.newaxis]
        sigma_v = np.float64(mx.median(_mlx.average(sigmas, stellar_spectrum * visible, axis=1)).item())
        sigma_th = np.float64(mx.median(_mlx.average(sigmas, planet_spectrum * thermal, axis=1)).item())

        gamma = sigma_v / sigma_th

        intermediate_n = (n[0:-1] + n[1:])/2.0
        dr = -np.diff(np.asarray(info_dict["radii"], dtype=float))
        d_taus = sigma_th * intermediate_n * dr
        taus = np.cumsum(d_taus)

        e2 = expn(2, gamma*taus)
        T4 = 3.0/4 * T_int**4 * (2.0/3 + taus) + 3.0/4 * T_irr**4 * (2.0/3 + 2.0/3/gamma * (1 + (gamma*taus/2 - 1)*np.exp(-gamma * taus)) + 2.0*gamma/3 * (1 - taus**2/2) * e2)
        T = T4 ** 0.25
        self.temperatures = np.append(T[0], T)

    def set_from_radiative_solution(self, T_star, Rs, a, Mp, Rp, beta, log_k_th, log_gamma, log_gamma2=None, alpha=0, T_int=100, **ignored_kwargs):
        '''From Line et al. 2013: http://adsabs.harvard.edu/abs/2013ApJ...775..137L, Equation 13 - 16'''
        T_star, Rs, a, Mp, Rp, beta, log_k_th, log_gamma, alpha, T_int = map(
            np.float64, (T_star, Rs, a, Mp, Rp, beta, log_k_th, log_gamma, alpha, T_int))

        k_th = 10.0**log_k_th
        gamma = 10.0**log_gamma
        gamma2 = None if log_gamma2 is None else 10.0**np.float64(log_gamma2)
        
        g = G * Mp / Rp**2
        T_eq = beta * np.sqrt(Rs/(2*a)) * T_star
        taus = k_th * self.pressures / g

        def incoming_stream_contribution(gamma):
            return 3.0/4 * T_eq**4 * (2.0/3 + 2.0/3/gamma * (1 + (gamma*taus/2 - 1)*np.exp(-gamma * taus)) + 2.0*gamma/3 * (1 - taus**2/2) * expn(2, gamma*taus))

        e1 = incoming_stream_contribution(gamma)
        T4 = 3.0/4 * T_int**4 * (2.0/3 + taus) + (1 - alpha) * e1

        if gamma2 is not None:
            e2 = incoming_stream_contribution(gamma2)
            T4 += alpha * e2
        self.temperatures = T4 ** 0.25
