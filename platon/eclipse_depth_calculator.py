import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import scipy.special
from astropy.io import ascii
import pandas as pd
from pathlib import Path

from .constants import h, c, k_B, R_jup, M_jup, R_sun
from ._atmosphere_solver import AtmosphereSolver
from ._interpolator_3D import interp1d, regular_grid_interp
from . import _mlx
from ._special import expn

class EclipseDepthCalculator:
    def __init__(self, include_condensation=True, method="xsec", include_opacities=["CH4", "CO2", "CO", "H2O", "H2S", "HCN", "K", "Na", "NH3", "SO2", "TiO", "VO"], downsample=1,
                 surface_library="Paragas"):
        '''
        All physical parameters are in SI.

        Parameters
        ----------
        include_condensation : bool
            Whether to use equilibrium abundances that take condensation into
            account.
        num_profile_heights : int
            The number of zones the atmosphere is divided into
        ref_pressure : float
            The planetary radius is defined as the radius at this pressure
        method : string
            "xsec" for opacity sampling, "ktables" for correlated k
        '''
        self.atm = AtmosphereSolver(include_condensation, method=method, include_opacities=include_opacities, downsample=downsample)
        self.tau_cache = _mlx.logspace(-6, 3, 1000)
        # Tabulated in float64.  0.5 - E3 is kept too: in optically thin
        # layers E3 ~ 0.5, and differences of E3 itself cancel in float32.
        with mx.stream(_mlx.FLOAT64_STREAM):
            exp3_cache = expn(3, _mlx.to_float64(np.logspace(-6, 3, 1000)))
            self.exp3_cache = _mlx.from_float64(exp3_cache)
            self._exp3_deficit_cache = _mlx.from_float64(0.5 - exp3_cache)
        self.surface_library = surface_library
        
        if surface_library not in ["HES2012", "Paragas"]:
            raise ValueError("The only surface libraries available are HES2012 and Paragas")

        basedir = Path(__file__).resolve().parent
        self.hemi_refls = pd.read_csv(basedir / f"data/{surface_library}/hemi_refls.csv")
        self.crust_emission_flux = ascii.read(basedir / f"data/{surface_library}/Crust_EmissionFlux.dat", delimiter="\t")
        if surface_library == "HES2012":
            self.redist_factors = {'Metal-rich': 0.6052, 'Ultramafic': 0.5532, 'Feldspathic': 0.5414,
                        'Basaltic': 0.6004, 'Granitoid': 0.5290, 'Clay': 0.5, 'Ice-rich silicate': 0.5,
                        'Fe-oxidized': 0.5978}
        else:
            df = pd.read_csv(basedir / f"data/{surface_library}/f_relation_new_samples.csv")
            self.redist_factors = {col: df[col][0] for col in df.columns}

    def _get_hemi_reflectance(self, surface_type, use_full_lambdas=False):
        '''Hemispherical reflectance of surface_type on the wavelength grid'''
        return self.atm.interp_to_lambda_grid(
            ("hemi_refl", self.surface_library, surface_type),
            self.hemi_refls["Wavelength"], self.hemi_refls[surface_type],
            use_full_lambdas)

    def calc_surface_temp(self, surface_type, stellar_fluxes_orig, a_over_Rs):
        a_over_Rs = _mlx.as_float(a_over_Rs)
        interp_rh = self._get_hemi_reflectance(surface_type, use_full_lambdas=True)
        irrad = float(self.redist_factors[surface_type]) * _mlx.trapz(
            (1 - interp_rh) * _mlx.asarray(stellar_fluxes_orig) / a_over_Rs**2,
            self.atm.orig_lambda_grid).item()
        emission_fluxes = np.asarray(self.crust_emission_flux[surface_type].data, dtype=float)
        if irrad < emission_fluxes[0] or irrad > emission_fluxes[-1]:
            raise ValueError("Cannot compute surface temperature because irradiation is out of range of the data files")

        temperature = np.interp(irrad, emission_fluxes, np.asarray(self.crust_emission_flux["Temperature [K]"], dtype=float))
        return float(temperature)
        
    def calc_surface_flux(self, surface_type, stellar_fluxes, a_over_Rs, temperature):
        a_over_Rs, temperature = _mlx.as_float(a_over_Rs), _mlx.as_float(temperature)
        hemi_reflectance = self._get_hemi_reflectance(surface_type)
        directional_emissivity = 1 - hemi_reflectance
        emitted_fluxes = directional_emissivity * (2 * np.pi * h * c**2) / self.atm.lambda_grid**5 / _mlx.expm1(h*c/k_B/(self.atm.lambda_grid * temperature))
        reflected_fluxes = _mlx.asarray(stellar_fluxes) / a_over_Rs**2 * hemi_reflectance
        fluxes = emitted_fluxes + reflected_fluxes
        return fluxes
        
    def _exp3_lookup(self, x):
        '''E3(x) and 0.5 - E3(x) (computed without cancellation), from the
        tables, sharing one table search'''
        flat = x.reshape(-1)
        lower, upper, frac = _mlx.linear_weights(flat, self.tau_cache)
        below = flat < self.tau_cache[0]
        above = flat > self.tau_cache[-1]

        def lookup(table, left, right):
            result = mx.take(table, lower) * (1 - frac) + mx.take(table, upper) * frac
            return mx.where(below, left, mx.where(above, right, result)).reshape(x.shape)

        return lookup(self.exp3_cache, 0.5, 0), lookup(self._exp3_deficit_cache, 0, 0.5)

    def _exp3(self, x):
        return self._exp3_lookup(x)[0]
        
    def change_wavelength_bins(self, bins):        
        '''Same functionality as :func:`~platon.transit_depth_calculator.TransitDepthCalculator.change_wavelength_bins`'''
        self.atm.change_wavelength_bins(bins)


    def _get_binned_depths(self, depths, stellar_spectrum, n_gauss=10):
        #Step 1: do a first binning if using k-coeffs; first binning is a
        #no-op otherwise
        if self.atm.method == "ktables":
            #Do a first binning based on ktables
            points, weights = scipy.special.roots_legendre(n_gauss)
            weights /= 2
            assert(depths.shape[0] % n_gauss == 0)
            intermediate_lambdas = mx.median(self.atm.lambda_grid.reshape((-1, n_gauss)), axis=1)
            intermediate_depths = depths.reshape((-1, n_gauss)) @ _mlx.asarray(weights)
            intermediate_stellar_spectrum = mx.median(stellar_spectrum.reshape((-1, n_gauss)), axis=1)
        elif self.atm.method == "xsec":
            intermediate_lambdas = self.atm.lambda_grid
            intermediate_depths = depths
            intermediate_stellar_spectrum = stellar_spectrum
        else:
            assert(False)


        if self.atm.wavelength_bins is None:
            return intermediate_lambdas, intermediate_depths, intermediate_lambdas, intermediate_depths

        # Weighted mean in each bin, with the stellar photon spectrum as weights
        intermediate_stellar_photon_spectrum = intermediate_stellar_spectrum / (h * c / intermediate_lambdas)
        bin_matrix, _ = self.atm.get_bin_matrix(intermediate_lambdas)
        binned_wavelengths = (bin_matrix @ intermediate_lambdas) / mx.sum(bin_matrix, axis=1)
        binned_depths = (bin_matrix @ (intermediate_depths * intermediate_stellar_photon_spectrum)) / \
            (bin_matrix @ intermediate_stellar_photon_spectrum)

        return intermediate_lambdas, intermediate_depths, binned_wavelengths, binned_depths

    def _get_photosphere_radii(self, taus, radii, planet_radius):
        result = mx.take(radii, mx.argmin(mx.abs(mx.log(taus)), axis=1))
        # If the atmosphere is optically thin at a wavelength, the
        # photosphere is the planet's surface
        return mx.where(mx.max(taus, axis=1) < 1, planet_radius, result)
              
    def compute_depths(self, t_p_profile, star_radius, planet_mass,
                       planet_radius, T_star, logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       add_gas_absorption=True, add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_spot=None, spot_cov_frac=None,
                       ri = None, frac_scale_height=1,number_density=0,
                       part_size=1e-6, part_size_std=0.5, P_quench=1e-99,
                       stellar_blackbody=False,
                       full_output=False, zero_opacities=[],
                       surface_type=None, semimajor_axis=None, surface_temp=None, surface_pressure=np.inf,
                       mmw=None):
        '''Most parameters are explained in :func:`~platon.transit_depth_calculator.TransitDepthCalculator.compute_depths`

        Parameters
        ----------
        t_p_profile : Profile
            A Profile object from TP_profile
        '''
        (star_radius, planet_radius, T_star, cloudtop_pressure, surface_pressure,
         surface_temp, semimajor_axis) = map(_mlx.as_float, (
             star_radius, planet_radius, T_star, cloudtop_pressure, surface_pressure,
             surface_temp, semimajor_axis))
        T_profile = t_p_profile.temperatures
        P_profile = t_p_profile.pressures
        bot_pressure = min(cloudtop_pressure, surface_pressure)
        
        atm_info = self.atm.compute_params(
            star_radius, planet_mass, planet_radius, P_profile, T_profile,
            logZ, CO_ratio, CH4_mult, gases, vmrs, add_gas_absorption, add_H_minus_absorption, add_scattering,
            scattering_factor, scattering_slope, scattering_ref_wavelength,
            add_collisional_absorption, bot_pressure, custom_abundances,
            T_star, T_spot, spot_cov_frac,
            ri, frac_scale_height, number_density, part_size, part_size_std,
            P_quench, zero_opacities=zero_opacities, mmw=mmw)

        assert(mx.max(atm_info["P_profile"]).item() <= bot_pressure)
        absorption_coeff = atm_info["absorption_coeff_atm"]
        intermediate_coeff = 0.5 * (absorption_coeff[0:-1] + absorption_coeff[1:])
        intermediate_T = 0.5 * (atm_info["T_profile"][0:-1] + atm_info["T_profile"][1:])
        dr = _mlx.from_float64(atm_info["dr"])
        d_taus = intermediate_coeff.T * dr
        taus = mx.cumsum(d_taus, axis=1)

        lambda_grid = self.atm.lambda_grid

        reshaped_lambda_grid = lambda_grid.reshape((-1, 1))
        planck_function = 2*h*c**2/reshaped_lambda_grid**5/_mlx.expm1(h*c/k_B/reshaped_lambda_grid/intermediate_T)

        #padded_taus: ensures 1st layer has 0 optical depth
        padded_taus = mx.concatenate([mx.zeros((taus.shape[0], 1), dtype=taus.dtype), taus], axis=1)
        # diff(E3) == -diff(0.5 - E3).  Difference whichever of the two is
        # small, so float32 keeps the relative precision of each layer's term
        exp3, exp3_deficit = self._exp3_lookup(padded_taus)
        d_exp3 = mx.where(exp3[:, 1:] < 0.25, mx.diff(exp3, axis=1),
                          -mx.diff(exp3_deficit, axis=1))
        integrand = planck_function * d_exp3
        fluxes = -2 * np.pi * mx.sum(integrand, axis=1)

        # x^2 E1(x) - x exp(-x) + exp(-x) is identically 2 E3(x), which does
        # not cancel catastrophically at large x (or give 0 * inf at x = 0)
        if not np.isinf(cloudtop_pressure) and cloudtop_pressure < surface_pressure:
            max_taus = mx.max(taus, axis=1)
            fluxes_from_cloud = np.pi * planck_function[:, -1] * 2 * expn(3, max_taus)
            fluxes += fluxes_from_cloud

        stellar_fluxes, _ = self.atm.get_stellar_spectrum(
            T_star, T_spot, spot_cov_frac, stellar_blackbody)
        
        if surface_pressure < cloudtop_pressure:
            if surface_temp is None:
                stellar_fluxes_orig, _ = self.atm.get_stellar_spectrum(
                    T_star, T_spot, spot_cov_frac, stellar_blackbody, use_full_lambdas=True)
                surface_temp = self.calc_surface_temp(surface_type, stellar_fluxes_orig, semimajor_axis / star_radius)
            surface_flux = self.calc_surface_flux(surface_type, stellar_fluxes, semimajor_axis / star_radius, surface_temp)
            max_taus = mx.max(taus, axis=1)
            fluxes += surface_flux * 2 * expn(3, max_taus)

        photosphere_radii = self._get_photosphere_radii(taus, _mlx.from_float64(atm_info["radii"]), planet_radius)
        eclipse_depths = fluxes / stellar_fluxes * (photosphere_radii/star_radius)**2
        #For correlated k, eclipse_depths has n_gauss points per wavelength, while unbinned_depths has 1 point per wavelength
        unbinned_wavelengths, unbinned_depths, binned_wavelengths, binned_depths = self._get_binned_depths(eclipse_depths, stellar_fluxes)

        if full_output:
            atm_info["surface_temp"] = surface_temp
            atm_info["stellar_spectrum"] = stellar_fluxes
            atm_info["planet_spectrum"] = fluxes
            atm_info["unbinned_wavelengths"] = unbinned_wavelengths
            atm_info["unbinned_eclipse_depths"] = unbinned_depths
            atm_info["taus"] = taus
            atm_info["contrib"] = -integrand / fluxes[:, np.newaxis]

            for key in atm_info:
                if type(atm_info[key]) == dict:
                    for subkey in atm_info[key]:
                        atm_info[key][subkey] = _mlx.to_numpy(atm_info[key][subkey])
                else:
                    atm_info[key] = _mlx.to_numpy(atm_info[key])

            return _mlx.to_numpy(binned_wavelengths), _mlx.to_numpy(binned_depths), atm_info

        return _mlx.to_numpy(binned_wavelengths), _mlx.to_numpy(binned_depths), None
