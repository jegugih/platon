'''Day-night heat redistribution on tidally locked rocky planets, from the
four-box model of Wirth, Powell & Wordsworth (2026, arXiv:2601.12556).

The dayside and nightside surfaces and atmospheres each have one temperature.
They exchange energy by radiation (the atmosphere is grey and optically thin
in the infrared, and transparent to starlight), by sensible heat through
surface drag, and by a day-to-night flow whose speed is a fraction alpha of
the sound speed.  Unlike the scaling of Koll (2022), it does not assume weak
temperature gradients, so it also applies to rapidly rotating planets.

The grey opacity and mean molecular weight come from PLATON's opacities and
chemistry.
'''
import warnings

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special

from ._atmosphere_solver import AtmosphereSolver
from . import _mlx
from .constants import h, c, k_B, AMU, G

SIGMA = 2 * np.pi**5 * k_B**4 / (15 * c**2 * h**3)

# Fiducial values of Wirth et al. (Table 1)
DRAG_COEFFICIENT = 3.4e-3      # (0.4 / ln(10 m / 1 cm))^2
TURBULENT_DIFFUSIVITY = 8e-2
CHI = 1.0                      # surface integration factor (Appendix A)
XI = np.pi / 4                 # atmospheric integration factor (Appendix A)


class WirthModel:
    def __init__(self, include_condensation=True,
                 include_opacities=["CH4", "CO2", "CO", "H2O", "H2S", "HCN", "K", "Na", "NH3", "SO2", "TiO", "VO"],
                 downsample=10):
        '''The opacities are loaded over the whole wavelength grid (0.2-30
        um), every `downsample`-th point, which is ample for the
        Planck-weighted means used here.'''
        self.atm = AtmosphereSolver(include_condensation, method="xsec",
                                    include_opacities=include_opacities, downsample=downsample)

    def grey_opacity(self, star_radius, planet_mass, planet_radius, surface_pressure, temperature,
                     logZ=0, CO_ratio=0.53, CH4_mult=1, gases=None, vmrs=None, mmw=None,
                     method="transmission", T_star=None, P_quench=1e-99, zero_opacities=[]):
        '''Grey infrared opacity kappa (m^2/kg) of an isothermal column at
        `temperature` down to surface_pressure (Pa), such that the model's
        optical depth is tau = 2 kappa p_s / g, and its mean molecular weight.

        method="transmission" (default) makes the grey column absorb the same
        fraction of the surface's thermal flux as the real one:
        tau = -ln <2 E3(tau_lambda)>, averaged over the Planck function, with
        tau_lambda the vertical optical depth.  This is what the model's
        optically thin equations use (its absorbed fraction of ground
        emission).  method="planck" is the Planck mean opacity, as Wirth et
        al. define it; for line absorbers such as CO2 it is orders of
        magnitude larger, which makes any atmosphere optically thick.

        Gas and collision-induced absorption count, not Rayleigh scattering.
        PLATON's collision-induced absorption only includes H2 and He.
        '''
        P = np.logspace(np.log10(self.atm.P_grid[0]), np.log10(surface_pressure), 60)
        T = np.full(len(P), float(temperature))
        info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P, T, logZ, CO_ratio, CH4_mult, gases, vmrs,
            add_gas_absorption=True, add_scattering=False, add_collisional_absorption=True,
            T_star=T_star, P_quench=P_quench, zero_opacities=zero_opacities, mmw=mmw)
        coeff = _mlx.to_numpy(info["absorption_coeff_atm"])
        dr = _mlx.to_numpy(info["dr"])
        mu = float(np.mean(_mlx.to_numpy(info["mu_profile"])))
        g = G * planet_mass / planet_radius**2

        lambdas = np.asarray(self.atm._lambda_grid_host, dtype=float)
        planck = 2 * h * c**2 / lambdas**5 / np.expm1(h * c / (k_B * lambdas * temperature))
        trapz = scipy.integrate.trapezoid
        planck_integral = trapz(planck, lambdas)
        coverage = np.pi * planck_integral / (SIGMA * temperature**4)
        if coverage < 0.95:
            warnings.warn("The wavelength grid holds only {:.0%} of the thermal emission at {:.0f} K".format(
                coverage, temperature))

        if method == "transmission":
            tau_lambda = np.sum(0.5 * (coeff[:-1] + coeff[1:]) * dr[:, np.newaxis], axis=0)
            transmission = trapz(planck * 2 * scipy.special.expn(3, tau_lambda), lambdas) / planck_integral
            tau = -np.log(transmission)
        elif method == "planck":
            # Column-mass-weighted Planck mean of the mass opacity
            rho = P * mu * AMU / (k_B * T)
            kappa_lambda = coeff / rho[:, np.newaxis]
            kappa_planck = trapz(planck * kappa_lambda, lambdas, axis=1) / planck_integral
            mass = 0.5 * (rho[:-1] + rho[1:]) * dr
            tau = 2 * np.sum(0.5 * (kappa_planck[:-1] + kappa_planck[1:]) * mass)
        else:
            raise ValueError("method must be 'transmission' or 'planck'")
        return {"kappa": tau * g / (2 * surface_pressure), "tau": tau, "mmw": mu,
                "planck_coverage": coverage}

    @staticmethod
    def solve(star_radius, T_star, semimajor_axis, planet_mass, planet_radius, surface_pressure,
              kappa, mmw, alpha=0.1, rotation_period=None, bond_albedo=0, cp_over_R=4.5,
              drag_coefficient=DRAG_COEFFICIENT, turbulent_diffusivity=TURBULENT_DIFFUSIVITY,
              chi=CHI, xi=XI):
        '''Solves the four-box model (the five dimensionless equations of
        Wirth et al., Sects. 3.1-3.2) for the hemispheric mean temperatures.

        Parameters
        ----------
        kappa : float
            Grey infrared opacity (m^2/kg); tau = 2 kappa p_s / g
        mmw : float
            Mean molecular weight (amu)
        alpha : float or str
            Day-to-night flow speed over the sound speed, or "wtg" for the
            weak temperature gradient scaling alpha = sqrt(1 - T_an/T_ad), or
            "rotating" for the Coriolis-dominated scaling
            alpha = 2^(1/4) gamma^(-1) (T_ad/T_0)^(1/2) Lambda (1 - T_an/T_ad),
            which needs rotation_period (s; the orbital period if tidally
            locked).  Wirth et al. explore 0.05-1.
        cp_over_R : float
            Isobaric heat capacity over the specific gas constant: ~4.5 for
            CO2 (the paper's JANAF values), ~3.5 for N2, O2 or CO, ~4 for H2O
        drag_coefficient, turbulent_diffusivity, chi, xi : float
            The paper's C_D, k_T and integration factors.

        Returns
        -------
        dict with the dayside and nightside surface and atmosphere
        temperatures (T_surface_day, T_atm_day, T_atm_night,
        T_surface_night), brightness temperatures (T_bright_day,
        T_bright_night) and the dayside brightness temperature an observer
        sees at eclipse (T_bright_day_obs), as in Wirth et al. Sect. 5.1, the
        effective temperatures of each hemisphere's emission to space in the
        model's energy budget (T_emit_day, T_emit_night; their sum balances
        the absorbed starlight), the flow speeds (v, w), alpha,
        tau, Lambda, T_0 (the airless dayside temperature), and `converged`.
        '''
        A = bond_albedo
        g = G * planet_mass / planet_radius**2
        F = SIGMA * T_star**4 * (star_radius / semimajor_axis)**2
        T_0 = T_star * ((1 - A) / 2)**0.25 * (semimajor_axis / star_radius)**-0.5
        T_eq = T_star * np.sqrt(star_radius / (2 * semimajor_axis)) * (1 - A)**0.25
        R = k_B / (mmw * AMU)
        c_p = cp_over_R * R
        c_v = c_p - R
        gamma = c_p / c_v
        tau = 2 * kappa * surface_pressure / g
        if tau > 1:
            warnings.warn("tau = {:.2g} > 1, but the model assumes an optically thin atmosphere".format(tau))

        # Dimensionless parameters (Sect. 3.1-3.2)
        W_0 = kappa * R * (1 - A) * F / (chi * drag_coefficient * c_p * g)
        C_T = R * T_0 * turbulent_diffusivity / (planet_radius * g)
        C = xi * C_T / (chi * drag_coefficient)
        L = chi * drag_coefficient * c_p * planet_radius * g / (R**2 * T_0)
        V_per_alpha = np.sqrt(c_p * R * T_0 / c_v) / W_0

        Lambda = None
        if alpha == "rotating":
            if rotation_period is None:
                raise ValueError("alpha='rotating' needs rotation_period")
            Lambda = np.sqrt(k_B * T_eq / (mmw * AMU)) * rotation_period / (2 * np.pi * planet_radius)

        def get_alpha(T0, T1, T2):
            if alpha == "wtg":
                return np.sqrt(1 - T2)
            if alpha == "rotating":
                return 2**0.25 / gamma * np.sqrt(T0 * T1) * Lambda * (1 - T2)
            return alpha

        def residuals(x):
            T0, T1, T2, T3, W = x
            V = get_alpha(T0, T1, T2) * V_per_alpha
            advection = C * V * (T0 * T1)**1.5
            return [
                T0**4 - 1 - tau * T0**4 * T1**4 - tau * W * (1 - 1 / T1),
                # The nightside surface equation, times tau
                T3**4 - tau - tau * W * (1 - T3) / (T0 * T1 * T2)**4,
                2 * T1**4 - 1 - (advection * (T2 - 1) + W * (1 / T1 - 1)) / T0**4,
                2 - T3**4 - (advection * (1 - T2) + W * (T3 - 1)) / (T0 * T1 * T2)**4,
                L / (T0 * T1) * (T0**4 - 2 * T0**4 * T1**4 + 2 * advection * (T2 - 1)
                                 + 2 * W * (1 / T1 - 1))
                - (3 * T2 - 1) * (V * np.sqrt(T0 * T1) - W)]

        # The dimensionless temperatures are between 0 and 1, except the
        # dayside surface's (T0), which can exceed the airless value
        lower = [1e-3, 1e-3, 1e-3, 1e-3, 0]
        upper = [3, 1, 1, 1, np.inf]
        best = None
        for T1 in [0.85, 0.6]:
            for T2 in [0.9, 0.5, 0.15]:
                for W in [0.01, 1, 100]:
                    fit = scipy.optimize.least_squares(
                        residuals, [1, T1, T2, 0.95, W], bounds=(lower, upper), method="trf",
                        xtol=1e-14, ftol=1e-14, gtol=1e-14)
                    if best is None or fit.cost < best.cost:
                        best = fit
            if best.cost < 1e-20:
                break
        T0, T1, T2, T3, W = best.x
        converged = best.cost < 1e-16
        if not converged:
            warnings.warn("The four-box model did not converge (residual {:.2g})".format(
                np.sqrt(2 * best.cost)))

        T_sd = T0 * T_0
        T_ad = T1 * T_sd
        T_an = T2 * T_ad
        T_sn = T3 * T_an
        # Brightness temperatures as in Sect. 5.1, and the observer's
        # weighting of the dayside toward the substellar point (Appendix B).
        # These are nan where 2 tau is too large for that formula.
        with np.errstate(invalid="ignore"):
            bright_day = (T_sd**4 + 2 * tau * (T_ad**4 - T_sd**4))**0.25
            bright_night = (T_sn**4 + 2 * tau * (T_an**4 - T_sn**4))**0.25
        alpha_value = get_alpha(T0, T1, T2)
        return {"T_surface_day": T_sd, "T_atm_day": T_ad, "T_atm_night": T_an,
                "T_surface_night": T_sn,
                "T_bright_day": bright_day, "T_bright_night": bright_night,
                "T_bright_day_obs": bright_day * ((4 - T2**4) / 3)**0.25,
                # Emission to space in the model's energy budget, which
                # balances the absorbed starlight
                "T_emit_day": ((1 - tau) * T_sd**4 + tau * T_ad**4)**0.25,
                "T_emit_night": ((1 - tau) * T_sn**4 + tau * T_an**4)**0.25,
                "v": alpha_value * np.sqrt(gamma * R * T_ad), "w": W * W_0,
                "alpha": alpha_value, "tau": tau, "Lambda": Lambda, "T_0": T_0, "T_eq": T_eq,
                "converged": converged}

    def compute(self, star_radius, T_star, semimajor_axis, planet_mass, planet_radius,
                surface_pressure, logZ=0, CO_ratio=0.53, CH4_mult=1, gases=None, vmrs=None,
                mmw=None, kappa_method="transmission", **solve_kwargs):
        '''grey_opacity at the equilibrium temperature, then solve.  The
        result also has kappa and mmw.'''
        T_eq = T_star * np.sqrt(star_radius / (2 * semimajor_axis)) * \
            (1 - solve_kwargs.get("bond_albedo", 0))**0.25
        grey = self.grey_opacity(star_radius, planet_mass, planet_radius, surface_pressure, T_eq,
                                 logZ, CO_ratio, CH4_mult, gases, vmrs, mmw, kappa_method, T_star)
        result = self.solve(star_radius, T_star, semimajor_axis, planet_mass, planet_radius,
                            surface_pressure, grey["kappa"], grey["mmw"], **solve_kwargs)
        result.update(kappa=grey["kappa"], mmw=grey["mmw"])
        return result
