import hashlib
import math

import mlx.core as mx
import numpy as np
import scipy.interpolate
import warnings

from pathlib import Path
from ._hist import get_num_bins
from . import _mlx
from . import _hydrostatic_solver
from ._loader import load_dict_from_pickle, load_numpy, get_shared
from .abundance_getter import AbundanceGetter
from ._species_data_reader import read_species_data
from . import _interpolator_3D
from .constants import k_B, AMU, M_sun, Teff_sun, G, h, c
from ._get_data import get_data_if_needed
from ._mie_cache import MieCache
from .errors import AtmosphereError
from ._interpolator_3D import regular_grid_interp, interp1d

# Internally, number densities are in units of _mlx.N_REF and cross sections
# in units of 1/N_REF, so that float32 can hold them (see _mlx.py).  Only
# `absorption_coeff_atm` and other outputs are in SI.

_MAX_SMOOTHED_MIE_TABLES = 8


class AtmosphereSolver:
    def __init__(self, include_condensation=True, ref_pressure=1e5, method='xsec', include_opacities=[], downsample=1):
        self.arguments = locals()
        del self.arguments["self"]

        get_data_if_needed()

        basedir = Path(__file__).resolve().parent
        absorption_data, self.mass_data, self.polarizability_data = read_species_data(
            basedir / "data/Absorption",
            basedir / "data/species_info",
            method, include_opacities, downsample)

        self.low_res_lambdas = load_numpy("data/low_res_lambdas.npy")
        stellar_spectra_dict = load_dict_from_pickle("data/stellar_spectra.pkl")

        if method == "xsec":
            lambda_grid = load_numpy("data/wavelengths.npy")[::downsample]
            self.d_ln_lambda = np.median(np.diff(np.log(lambda_grid)))
        else:
            warnings.warn("Correlated-k is not recommended, and will probably be removed in a future version! Please use xsec")
            lambda_grid = load_numpy("data/k_wavelengths.npy")
            self.d_ln_lambda = np.median(np.diff(np.log(np.unique(lambda_grid))))

        self.stellar_spectra_temps = np.array(stellar_spectra_dict["temperatures"], dtype=float)
        # Data on the low-resolution grid is resampled once, in float64
        stellar_spectra = np.array([
            np.interp(lambda_grid, self.low_res_lambdas, spectrum)
            for spectrum in stellar_spectra_dict["spectra"]])

        collisional_absorption_data = load_dict_from_pickle(
            "data/collisional_absorption.pkl")
        low_res_index = np.interp(lambda_grid, self.low_res_lambdas,
                                  np.arange(len(self.low_res_lambdas)))
        lower = np.floor(low_res_index).astype(int)
        upper = np.ceil(low_res_index).astype(int)
        frac = low_res_index - lower
        # CIA coefficients are ~1e-60 m^5; scale by N_REF^2 for float32
        self._scaled_cia = {
            key: _mlx.asarray((data[:, lower] * (1 - frac) + data[:, upper] * frac) * _mlx.N_REF**2)
            for key, data in collisional_absorption_data.items()}

        self.P_grid = load_numpy("data/pressures.npy").astype(float)
        self.T_grid = load_numpy("data/temperatures.npy").astype(float)

        self.N_T = len(self.T_grid)
        self.N_P = len(self.P_grid)

        # One (temperature, pressure, species, wavelength) array, so gas
        # absorption is a single batched matmul instead of a loop over dicts.
        # It is shared with other calculators that load the same opacities.
        self.opacity_species = list(absorption_data)
        self._opacities_key = ("opacities", method, downsample,
                               tuple(self.opacity_species), str(_mlx.dtype()))
        self._opacities_holder = get_shared(
            self._opacities_key, lambda: self._stack_opacities(absorption_data))
        self._opacities = self._opacities_holder.value

        self._set_lambda_grid(lambda_grid)
        self.stellar_spectra = _mlx.asarray(stellar_spectra)
        # In case we truncate with change_wavelength_bins.  MLX arrays are
        # immutable, so these are not affected when the grid is truncated.
        self.orig_lambda_grid = self.lambda_grid
        self._orig_lambda_grid_host = self._lambda_grid_host
        self.orig_stellar_spectra = self.stellar_spectra

        self.wavelength_rebinned = False
        self.wavelength_bins = None

        self.abundance_getter = AbundanceGetter(include_condensation)
        self.min_temperature = max(self.T_grid.min(), self.abundance_getter.min_temperature)
        self.max_temperature = np.amax(self.T_grid)

        self.ref_pressure = ref_pressure
        self.method = method
        self._mie_cache = MieCache()

        self._mie_holder = get_shared(("mie_cross_secs",),
                                      lambda: load_dict_from_pickle("data/all_cross_secs.pkl"))
        self.all_cross_secs = self._mie_holder.value
        self.all_radii = load_numpy("data/mie_radii.npy")
        diffs = np.diff(np.log(self.all_radii))
        self.d_ln_radii = np.median(diffs)
        assert(np.allclose(diffs, self.d_ln_radii))
        self._smoothed_cross_secs = {}

    def _stack_opacities(self, absorption_data):
        np_dtype = np.float32 if _mlx.dtype() == mx.float32 else np.float64
        stacked = np.empty((self.N_T, self.N_P, len(absorption_data),
                            next(iter(absorption_data.values())).shape[2]) if absorption_data
                           else (self.N_T, self.N_P, 0, 0), dtype=np_dtype)
        for i, name in enumerate(absorption_data):
            stacked[:, :, i, :] = absorption_data[name]
        opacities = mx.array(stacked, dtype=_mlx.dtype())
        mx.eval(opacities)
        return opacities

    def _set_lambda_grid(self, lambda_grid):
        self._lambda_grid_host = np.asarray(lambda_grid, dtype=float)
        self.lambda_grid = _mlx.asarray(lambda_grid)
        self.N_lambda = len(lambda_grid)
        self._low_res_resampler = _mlx.LinearResampler(lambda_grid, self.low_res_lambdas)
        self._H_minus_k = None
        self._bin_matrix = None
        self._resampled_tables = {}

    def interp_to_lambda_grid(self, key, table_xs, table_ys, use_full_lambdas=False):
        '''np.interp of a fixed table (e.g. a surface reflectance spectrum)
        to the wavelength grid, cached under `key` until the grid changes.
        Done in float64: some tables are sampled so finely (~1e-12 m) that
        float32 wavelengths would shift them by a noticeable fraction of a
        sample.'''
        cache_key = (key, use_full_lambdas)
        if cache_key not in self._resampled_tables:
            lambdas = self._orig_lambda_grid_host if use_full_lambdas else self._lambda_grid_host
            self._resampled_tables[cache_key] = _mlx.asarray(np.interp(
                lambdas, np.asarray(table_xs, dtype=float), np.asarray(table_ys, dtype=float)))
        return self._resampled_tables[cache_key]

    def get_bin_matrix(self, intermediate_lambdas):
        '''Matrix whose row i selects the wavelengths in wavelength bin i
        (start <= wavelength < end), so that binning is a matmul.  Also
        returns the indices in each bin.  Cached until the grid changes.'''
        if self._bin_matrix is None or self._bin_matrix.shape[1] != intermediate_lambdas.shape[0]:
            lambdas = _mlx.to_numpy(intermediate_lambdas)
            matrix = np.array([np.logical_and(lambdas >= start, lambdas < end)
                               for (start, end) in self.wavelength_bins], dtype=float)
            self._bin_matrix = _mlx.asarray(matrix)
            self._bin_indices = [mx.array(np.nonzero(row)[0].astype(np.int32))
                                 for row in matrix]
        return self._bin_matrix, self._bin_indices

    @property
    def absorption_data(self):
        '''Absorption coefficients of each species, indexed [T, P, wavelength]'''
        return {name: self._opacities[:, :, i, :]
                for i, name in enumerate(self.opacity_species)}

    def get_lambda_grid(self):
        return np.copy(self._lambda_grid_host)

    def change_wavelength_bins(self, bins):
        """Specify wavelength bins, instead of using the full wavelength grid
        in self.lambda_grid.  This makes the code much faster, as
        `compute_depths` will only compute depths at wavelengths that fall
        within a bin.

        Parameters
        ----------
        bins : array_like, shape (N,2)
            Wavelength bins, where bins[i][0] is the start wavelength and
            bins[i][1] is the end wavelength for bin i. If bins is None, resets
            the calculator to its unbinned state.

        Raises
        ------
        NotImplementedError
            Raised when `change_wavelength_bins` is called more than once,
            which is not supported.
        """
        if self.wavelength_rebinned:
            self.__init__(**self.arguments)
            self.wavelength_rebinned = False

        if bins is None:
            return

        bins = np.array(bins, dtype=float)
        lambda_grid = self._lambda_grid_host
        for start, end in bins:
            if start < lambda_grid.min() \
               or start > lambda_grid.max() \
               or end < lambda_grid.min() \
               or end > lambda_grid.max():
                raise ValueError("Invalid wavelength bin: {}-{} meters".format(start, end))
            num_points = np.sum(np.logical_and(lambda_grid > start,
                                               lambda_grid < end))
            if num_points == 0:
                raise ValueError("Wavelength bin too narrow: {}-{} meters".format(start, end))
            if num_points <= 5:
                print("WARNING: only {} points in {}-{} m bin. Results will be inaccurate".format(num_points, start, end))

        self.wavelength_rebinned = True
        self.wavelength_bins = bins

        cond = np.any(np.array([
            np.logical_and(lambda_grid > start, lambda_grid < end) \
            for (start, end) in bins]), axis=0)
        indices = np.nonzero(cond)[0]
        full_opacities = self._opacities

        def take_subset():
            subset = mx.take(full_opacities, mx.array(indices.astype(np.int32)), axis=3)
            mx.eval(subset)
            return subset

        # Shared with other calculators that use the same bins; the full
        # opacities are freed once no calculator holds them
        self._opacities_key += (hashlib.sha1(indices.tobytes()).hexdigest(),)
        self._opacities_holder = get_shared(self._opacities_key, take_subset)
        self._opacities = self._opacities_holder.value
        indices = mx.array(indices.astype(np.int32))
        self.stellar_spectra = mx.take(self.stellar_spectra, indices, axis=1)
        for key in self._scaled_cia:
            self._scaled_cia[key] = mx.take(self._scaled_cia[key], indices, axis=1)
        self._set_lambda_grid(lambda_grid[cond])
        mx.eval(self.stellar_spectra, *self._scaled_cia.values())

    def _get_k(self, T, wavelengths):
        '''H- bound-free plus free-free absorption coefficient (m^4/N), for
        an array of temperatures (rows) and wavelengths (columns).  Runs in
        float64 on the CPU stream; it is tabulated once per wavelength grid.'''
        with mx.stream(_mlx.FLOAT64_STREAM):
            T = _mlx.to_float64(T).reshape(-1, 1)
            wavelengths = 1e6 * _mlx.to_float64(wavelengths)
            alpha = 14391
            lambda_0 = 1.6419

            #Calculate bound-free absorption coefficient
            cond = wavelengths < lambda_0
            x = mx.maximum(1 / wavelengths - 1 / lambda_0, 0)
            C = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
            f_lambda = sum(C[i-1] * x**((i-1)/2) for i in range(1, 7))
            sigma = 1e-18 * wavelengths**3 * x**1.5 * f_lambda
            # T^-2.5 exp(alpha/lambda_0/T) is combined in the exponent
            k_bf = 0.75 * _mlx.exp(alpha / lambda_0 / T - 2.5 * mx.log(T)) * \
                -_mlx.expm1(-alpha / wavelengths / T) * sigma
            k_bf = mx.where(cond, k_bf, 0)

            #Now calculate free-free absorption coefficient
            mid = mx.logical_and(wavelengths > 0.1823, wavelengths < 0.3645)
            red = wavelengths > 0.3645

            ff_matrix_red = [
                [0, 0, 0, 0, 0, 0],
                [2483.346, 285.827, -2054.291, 2827.776, -1341.537, 208.952],
                [-3449.889, -1158.382, 8746.523, -11485.632, 5303.609, -812.939],
                [2200.04, 2427.719, -13651.105, 16755.524, -7510.494, 1132.738],
                [-696.271, -1841.4, 8624.97, -10051.53, 4400.067, -655.02],
                [88.283, 444.517, -1863.864, 2095.288, -901.788, 132.985]]
            ff_matrix_mid = [
                [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
                [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.36],
                [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
                [115.5291, 169.6374, -245.649, 114.243, -21.9972, 1.5097],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]]

            powers = [wavelengths**i for i in (2, 0, -1, -2, -3, -4)]
            k_ff = mx.zeros(k_bf.shape, dtype=k_bf.dtype)
            for n in range(1, 7):
                A_mid = sum(coeff * p for coeff, p in zip(ff_matrix_mid[n-1], powers))
                A_red = sum(coeff * p for coeff, p in zip(ff_matrix_red[n-1], powers))
                A = mx.where(mid, A_mid, mx.where(red, A_red, 0))
                k_ff = k_ff + 1e-29 * (5040/T)**((n+1)/2) * A

            k = k_bf + k_ff

            #1e-3 to convert from cm^4/dyne to m^4/N
            return _mlx.from_float64(k * 1e-3)

    def _get_H_minus_absorption(self, names, log_abund, node_T, node_P):
        '''log_abund has shape (species, nodes); node_T and node_P are the
        grid indices of the nodes.  Returns shape (nodes, wavelengths).'''
        if self._H_minus_k is None:
            self._H_minus_k = self._get_k(self.T_grid, self._lambda_grid_host)

        T = self.T_grid[node_T]
        P = self.P_grid[node_P]
        # k * el * H * P^2 / (k_B T), with the product formed in log space
        log_P_n = _mlx.asarray(np.log10(P * P / (k_B * T)))
        weight = mx.power(10.0, log_abund[names.index("el")] +
                          log_abund[names.index("H")] + log_P_n)
        return mx.take(self._H_minus_k, mx.array(node_T), axis=0) * weight[:, np.newaxis]

    def _get_gas_absorption(self, names, abund, nodes, zero_opacities=[]):
        '''abund has shape (species, nodes); nodes are flattened
        (temperature, pressure) grid indices.  Returns (nodes, wavelengths).'''
        if len(self.opacity_species) == 0:
            return mx.zeros((abund.shape[1], self.N_lambda), dtype=_mlx.dtype())

        rows = {name: i for i, name in enumerate(names)}
        used = [s in rows and s not in zero_opacities for s in self.opacity_species]
        indices = mx.array([rows.get(s, 0) for s in self.opacity_species])
        weights = mx.take(abund, indices, axis=0) * _mlx.asarray(used)[:, np.newaxis]

        # (node, 1, species) @ (species, wavelength) of each node's grid point
        weights = mx.transpose(weights)[:, np.newaxis, :]
        opacities = self._opacities.reshape((self.N_T * self.N_P,) + self._opacities.shape[2:])
        nodes = mx.array(nodes.astype(np.uint32))
        if opacities.dtype == mx.float32:
            # Avoids copying the opacities of the selected grid points
            return mx.gather_mm(weights, opacities, rhs_indices=nodes)[:, 0, :]
        # MLX's gather_mm only supports float32
        return (weights @ mx.take(opacities, nodes, axis=0))[:, 0, :]

    def _get_scattering_absorption(self, names, abund, node_T, node_P,
                                   multiple=1, slope=4, ref_wavelength=1e-6):
        # Polarizabilities are ~1e-30 m^3: work with alpha * 1e30
        pol_sqr = _mlx.asarray([(self.polarizability_data.get(s, 0) * 1e30)**2 for s in names])
        sum_polarizability_sqr = mx.tensordot(pol_sqr, abund, axes=1)

        n = _mlx.asarray(_mlx.n_scaled(self.P_grid[node_P], self.T_grid[node_T]))
        # multiple * 128/3 pi^5 * ref^(slope-4) * n * alpha^2 / lambda^slope
        prefactor = multiple * (128.0 / 3 * np.pi**5) * _mlx.N_REF * 1e-60 / ref_wavelength**4
        spectral = prefactor * mx.power(ref_wavelength / self.lambda_grid, slope)
        return (n * sum_polarizability_sqr)[:, np.newaxis] * spectral

    def _get_collisional_absorption(self, names, abund, node_T, node_P):
        absorption_coeff = mx.zeros((len(node_T), self.N_lambda), dtype=_mlx.dtype())

        rows = {name: i for i, name in enumerate(names)}
        n = _mlx.asarray(_mlx.n_scaled(self.P_grid[node_P], self.T_grid[node_T]))
        T_indices = mx.array(node_T)
        for s1, s2 in self._scaled_cia:
            if s1 in rows and s2 in rows:
                n1 = abund[rows[s1]] * n
                n2 = abund[rows[s2]] * n
                abs_data = mx.take(self._scaled_cia[(s1, s2)], T_indices, axis=0)
                absorption_coeff = absorption_coeff + abs_data * (n1 * n2)[:, np.newaxis]

        return absorption_coeff

    def _get_interpolation_nodes(self, T_profile, P_profile, T_cond, P_cond):
        '''Each layer's cross sections are interpolated bilinearly in (1/T,
        log P) between the four grid points around it, using the same
        fractional indices as before.  Returns the grid indices (node_T,
        node_P) of the grid points that are actually used, and for each
        layer the four corners as indices into those nodes, with weights;
        both of shape (4, layers).'''
        T_grid = self.T_grid[T_cond]
        P_grid = self.P_grid[P_cond]
        x = np.interp(np.log(P_profile), np.log(P_grid), np.arange(len(P_grid)))
        x_lower, x_upper = np.floor(x).astype(int), np.ceil(x).astype(int)
        fx = x - x_lower
        if len(T_grid) == 1:
            t_lower = t_upper = np.zeros(len(T_profile), dtype=int)
            fy = np.zeros(len(T_profile))
        else:
            # Interpolated in 1/T, which increases along the reversed grid
            y = np.interp(1.0 / T_profile, 1.0 / T_grid[::-1], np.arange(len(T_grid)))
            y_lower, y_upper = np.floor(y).astype(int), np.ceil(y).astype(int)
            fy = y - y_lower
            t_lower, t_upper = len(T_grid) - 1 - y_lower, len(T_grid) - 1 - y_upper

        corner_T = T_cond.start + np.array([t_lower, t_upper, t_lower, t_upper])
        corner_P = P_cond.start + np.array([x_lower, x_lower, x_upper, x_upper])
        weights = np.array([(1 - fy) * (1 - fx), fy * (1 - fx), (1 - fy) * fx, fy * fx])
        # A zero weight only occurs where floor == ceil, so it never adds a node
        nodes, corners = np.unique(corner_T * self.N_P + corner_P, return_inverse=True)
        return nodes // self.N_P, nodes % self.N_P, corners.reshape(weights.shape), weights

    def _get_smoothed_mie_table(self, ri, kernel):
        key = (ri, kernel)
        if key not in self._smoothed_cross_secs:
            if len(self._smoothed_cross_secs) >= _MAX_SMOOTHED_MIE_TABLES:
                self._smoothed_cross_secs.pop(next(iter(self._smoothed_cross_secs)))
            table = _mlx.asarray(self.all_cross_secs[ri] * _mlx.N_REF)
            self._smoothed_cross_secs[key] = _mlx.gaussian_filter(table, kernel)
        return self._smoothed_cross_secs[key]

    def _get_mie_scattering_absorption(self, P_cond, T_cond, ri, part_size,
                                       frac_scale_height, max_number_density,
                                       sigma = 0.5, max_zscore = 5, num_integral_points = 100):
        '''Returns the Mie absorption coefficient (m^-1) with shape
        (1, num pressures, num wavelengths)'''
        part_size, frac_scale_height, max_number_density, sigma = map(
            _mlx.as_float, (part_size, frac_scale_height, max_number_density, sigma))
        if isinstance(ri, str):
            kernel = sigma / self.d_ln_radii
            if part_size < self.all_radii[3*int(kernel)] or part_size > self.all_radii[-3*int(kernel)]:
                raise ValueError("part_size out of bounds: {} m".format(part_size))
            cross_secs = self._get_smoothed_mie_table(ri, kernel)
            eff_cross_section = self._low_res_resampler(
                interp1d(part_size, self.all_radii, cross_secs.T))
        else:
            z_scores = -np.logspace(np.log10(0.1), np.log10(max_zscore), int(num_integral_points/2))
            z_scores = _mlx.asarray(np.append(z_scores[::-1], -z_scores))

            probs = _mlx.exp(-z_scores**2/2) / np.sqrt(2 * np.pi)
            radii = part_size * _mlx.exp(z_scores * sigma)
            geometric_cross_section = np.pi * _mlx.N_REF * radii**2

            dense_xs = 2*np.pi*radii[np.newaxis,:] / self.lambda_grid[:,np.newaxis]
            log_dense_xs = mx.log(dense_xs).reshape(-1)

            # Q_ext is computed in NumPy, at the edges of a histogram of
            # log(x), then splined to every x in MLX
            n_bins = get_num_bins(log_dense_xs)
            first_edge = mx.min(log_dense_xs).item()
            last_edge = mx.max(log_dense_xs).item()
            log_x_hist = np.linspace(first_edge, last_edge, n_bins + 1)

            Qext_hist = self._mie_cache.get_and_update(ri, np.exp(log_x_hist))
            spl = scipy.interpolate.make_interp_spline(log_x_hist, Qext_hist)
            Qext_intpl = _mlx.PiecewisePolynomial(spl)(log_dense_xs).reshape((self.N_lambda, radii.shape[0]))
            eff_cross_section = _mlx.trapz(probs*geometric_cross_section*Qext_intpl, z_scores)

        P = self.P_grid[P_cond]
        n = _mlx.asarray(max_number_density / _mlx.N_REF * np.power(P / max(P), 1.0/frac_scale_height))
        return n[np.newaxis, :, np.newaxis] * eff_cross_section[np.newaxis, np.newaxis, :]

    def _get_above_cloud_profiles(self, P_profile, T_profile, names, log_abund,
                                  planet_mass, planet_radius, star_radius,
                                  num_above_cloud, T_star=None, mmw=None):
        '''P_profile and T_profile are NumPy arrays.  If mmw is given, it is
        the mean molecular weight of every layer, instead of the one implied
        by the abundances.'''
        assert(len(P_profile) == len(T_profile))
        # Abundances of all species at once: (T, P, species) -> (layer, species)
        atm_log_abund = regular_grid_interp(
            self.T_grid, np.log10(self.P_grid),
            mx.transpose(log_abund, (1, 2, 0)), T_profile, np.log10(P_profile))
        atm_abund = mx.power(10.0, atm_log_abund)
        if mmw is None:
            masses = _mlx.asarray([self.mass_data[s] for s in names])
            mu_profile = atm_abund @ masses
        else:
            mu_profile = np.full(len(P_profile), mmw)

        radii, dr = _hydrostatic_solver._solve(
            P_profile, T_profile, self.ref_pressure, mu_profile, planet_mass,
            planet_radius, star_radius, num_above_cloud, T_star)

        atm_abundances = {name: atm_abund[:num_above_cloud, i]
                          for i, name in enumerate(names)}
        return radii, dr, atm_abundances, mu_profile

    def _get_log_abundances(self, logZ, CO_ratio, CH4_mult, custom_abundances, gases, vmrs):
        '''Returns (species names, log10 abundances with shape
        (species, N_T, N_P)), before any flooring.'''
        if custom_abundances is None and logZ is not None and CO_ratio is not None:
            names = list(self.abundance_getter.included_species)
            log_abund = self.abundance_getter.get_log(logZ, CO_ratio)
            log_CH4_mult = math.log10(CH4_mult) if CH4_mult > 0 else -math.inf
            offsets = np.zeros((len(names), 1, 1))
            offsets[names.index("CH4")] = log_CH4_mult
            return names, log_abund + _mlx.asarray(offsets)

        if logZ is not None or CO_ratio is not None:
            raise ValueError(
                "Must set logZ=None and CO_ratio=None to use custom_abundances")

        if isinstance(custom_abundances, str):
            # Interpret as filename
            custom_abundances = AbundanceGetter.from_file(custom_abundances)
            return self._stack_log_abundances(custom_abundances)

        if isinstance(custom_abundances, dict):
            for key, value in custom_abundances.items():
                if not isinstance(value, (np.ndarray, mx.array)):
                    raise ValueError(
                        "custom_abundances must map species names to arrays")
                if tuple(value.shape) != (self.N_T, self.N_P):
                    raise ValueError(
                        "custom_abundances has array of invalid size")
            return self._stack_log_abundances(custom_abundances)

        if custom_abundances is None and vmrs is not None and gases is not None:
            names = list(gases)
            with np.errstate(divide="ignore"):
                log_vmrs = np.log10(np.asarray(vmrs, dtype=float))
            log_abund = np.broadcast_to(log_vmrs[:, np.newaxis, np.newaxis],
                                        (len(names), self.N_T, self.N_P))
            return names, _mlx.asarray(log_abund)

        raise ValueError("Unrecognized format for custom_abundances")

    @staticmethod
    def _stack_log_abundances(abundances):
        names = list(abundances)
        logs = []
        for name in names:
            value = abundances[name]
            if isinstance(value, mx.array):
                logs.append(mx.log10(_mlx.asarray(value)))
            else:
                # Take the log in float64, before rounding to float32
                with np.errstate(divide="ignore", invalid="ignore"):
                    logs.append(_mlx.asarray(np.log10(np.asarray(value, dtype=float))))
        return names, mx.stack(logs)

    def _validate_params(self, T_profile, logZ, CO_ratio, cloudtop_pressure):
        T_profile = np.atleast_1d(_mlx.to_numpy(T_profile))
        if T_profile.min() < self.min_temperature or\
           T_profile.max() > self.max_temperature:
            raise AtmosphereError("Invalid temperatures in T/P profile")

        if logZ is not None:
            minimum = self.abundance_getter.logZs.min()
            maximum = self.abundance_getter.logZs.max()
            if logZ < minimum or logZ > maximum:
                raise ValueError(
                    "logZ {} is out of bounds ({} to {})".format(
                        logZ, minimum, maximum))

        if CO_ratio is not None:
            minimum = self.abundance_getter.CO_ratios.min()
            maximum = self.abundance_getter.CO_ratios.max()
            if CO_ratio < minimum or CO_ratio > maximum:
                raise ValueError(
                    "C/O ratio {} is out of bounds ({} to {})".format(CO_ratio, minimum, maximum))

        if not np.isinf(cloudtop_pressure):
            minimum = self.P_grid.min()
            maximum = self.P_grid.max()
            if cloudtop_pressure <= minimum or cloudtop_pressure > maximum:
                raise ValueError(
                    "Cloudtop pressure is {} Pa, but must be between {} and {} Pa unless it is np.inf".format(
                        cloudtop_pressure, minimum, maximum))

    def get_stellar_spectrum(self, T_star, T_spot, spot_cov_frac, blackbody=False, use_full_lambdas=False):
        T_star, T_spot, spot_cov_frac = map(_mlx.as_float, (T_star, T_spot, spot_cov_frac))
        if use_full_lambdas:
            lambdas = self.orig_lambda_grid
            stellar_spectra = self.orig_stellar_spectra
        else:
            lambdas = self.lambda_grid
            stellar_spectra = self.stellar_spectra

        if spot_cov_frac is None:
            spot_cov_frac = 0

        if T_spot is None:
            T_spot = T_star

        if T_star is None:
            unspotted_spectrum = mx.ones(lambdas.shape, dtype=lambdas.dtype)
            spot_spectrum = mx.ones(lambdas.shape, dtype=lambdas.dtype)

        elif T_star >= self.stellar_spectra_temps.min() and T_star <= self.stellar_spectra_temps.max() and not blackbody:
            unspotted_spectrum = interp1d(T_star, self.stellar_spectra_temps, stellar_spectra)
            spot_spectrum = interp1d(T_spot, self.stellar_spectra_temps, stellar_spectra)
            if spot_spectrum.shape[0] != lambdas.shape[0]:
                raise ValueError("Stellar spectra has a different length ({}) than opacities ({})!  If you are using high resolution opacities, pass stellar_blackbody=True to compute_depths".format(spot_spectrum.shape[0], lambdas.shape[0]))
        else:
            # 2 c pi / lambda^4 / (exp(hc/lambda/k_B/T) - 1) * hc/lambda,
            # grouped so the intermediate stays well inside float32 range
            unspotted_spectrum = 2 * np.pi * h * c**2 / lambdas**5 / \
                _mlx.expm1(h * c / k_B / lambdas / T_star)
            spot_spectrum = 2 * np.pi * h * c**2 / lambdas**5 / \
                _mlx.expm1(h * c / k_B / lambdas / T_spot)

        stellar_spectrum = spot_cov_frac * spot_spectrum + \
                           (1 - spot_cov_frac) * unspotted_spectrum
        correction_factors = unspotted_spectrum/stellar_spectrum
        return stellar_spectrum, correction_factors

    def compute_params(self, star_radius, planet_mass, planet_radius,
                       P_profile, T_profile,
                       logZ=0, CO_ratio=0.53, CH4_mult=1,
                       gases=None, vmrs=None,
                       add_gas_absorption=True,
                       add_H_minus_absorption=False,
                       add_scattering=True, scattering_factor=1,
                       scattering_slope=4, scattering_ref_wavelength=1e-6,
                       add_collisional_absorption=True,
                       cloudtop_pressure=np.inf, custom_abundances=None,
                       T_star=None, T_spot=None, spot_cov_frac=None,
                       ri=None, frac_scale_height=1, number_density=0,
                       part_size=1e-6, part_size_std=0.5,
                       P_quench=1e-99,
                       min_abundance=1e-99, min_cross_sec=1e-99, zero_opacities=[],
                       mmw=None):
        (star_radius, planet_mass, planet_radius, logZ, CO_ratio, CH4_mult,
         scattering_factor, scattering_slope, scattering_ref_wavelength,
         cloudtop_pressure, T_star, T_spot, spot_cov_frac, frac_scale_height,
         number_density, part_size, part_size_std, P_quench, min_abundance,
         min_cross_sec, mmw) = map(_mlx.as_float, (
             star_radius, planet_mass, planet_radius, logZ, CO_ratio, CH4_mult,
             scattering_factor, scattering_slope, scattering_ref_wavelength,
             cloudtop_pressure, T_star, T_spot, spot_cov_frac, frac_scale_height,
             number_density, part_size, part_size_std, P_quench, min_abundance,
             min_cross_sec, mmw))
        if mmw is not None and not mmw > 0:
            raise ValueError("Mean molecular weight must be positive, not {}".format(mmw))

        # The (small) profiles are needed on the host for bounds, slicing
        # and the hydrostatic solution
        P_host = np.atleast_1d(np.asarray(_mlx.to_numpy(P_profile), dtype=float))
        T_host = np.atleast_1d(np.asarray(_mlx.to_numpy(T_profile), dtype=float))
        self._validate_params(T_host, logZ, CO_ratio, cloudtop_pressure)
        P_profile = _mlx.asarray(P_host)
        T_profile = _mlx.asarray(T_host)

        names, log_abund = self._get_log_abundances(
            logZ, CO_ratio, CH4_mult, custom_abundances, gases, vmrs)

        # Same as setting NaNs and abundances below min_abundance to
        # min_abundance, but exact in float32
        log_floor = math.log10(min_abundance) if min_abundance > 0 else -math.inf
        log_abund = mx.where(mx.isnan(log_abund), log_floor,
                             mx.maximum(log_abund, log_floor))

        quenched = self.P_grid <= P_quench
        if np.any(quenched):
            T_quench = np.interp(np.log(P_quench), np.log(P_host), T_host)
            quench_log_abund = regular_grid_interp(
                self.T_grid, np.log10(self.P_grid),
                mx.transpose(log_abund, (1, 2, 0)), T_quench, np.log10(P_quench))
            log_abund = mx.where(mx.array(quenched)[np.newaxis, np.newaxis, :],
                                 quench_log_abund[:, np.newaxis, np.newaxis],
                                 log_abund)

        # P_profile is increasing, so the layers above the clouds come first
        num_above_cloud = int(np.sum(P_host < cloudtop_pressure))

        radii, dr, atm_abundances, mu_profile = self._get_above_cloud_profiles(
            P_host, T_host, names, log_abund, planet_mass, planet_radius,
            star_radius, num_above_cloud, T_star, mmw)

        P_profile = P_profile[:num_above_cloud]
        T_profile = T_profile[:num_above_cloud]

        T_cond = _interpolator_3D.get_condition_slice(T_host[:num_above_cloud], self.T_grid)
        P_cond = _interpolator_3D.get_condition_slice(
            P_host[:num_above_cloud], self.P_grid, cloudtop_pressure)
        # Opacities are only computed at the grid points that the layers'
        # interpolation uses (~3x fewer than the whole T_cond x P_cond block
        # for a non-isothermal profile)
        node_T, node_P, corners, corner_weights = self._get_interpolation_nodes(
            T_host[:num_above_cloud], P_host[:num_above_cloud], T_cond, P_cond)
        nodes = node_T * self.N_P + node_P
        log_abund = mx.take(log_abund.reshape((len(names), -1)), mx.array(nodes), axis=1)
        abund = mx.power(10.0, log_abund)

        absorption_coeff = mx.zeros((len(nodes), self.N_lambda), dtype=_mlx.dtype())
        if add_gas_absorption:
            absorption_coeff += self._get_gas_absorption(names, abund, nodes, zero_opacities=zero_opacities)
        if add_H_minus_absorption:
            absorption_coeff += self._get_H_minus_absorption(names, log_abund, node_T, node_P)
        if add_scattering:
            if ri is not None:
                if scattering_factor != 1 or scattering_slope != 4:
                    raise ValueError("Cannot use both parametric and Mie scattering at the same time")

                mie_absorption = self._get_mie_scattering_absorption(
                    P_cond, T_cond, ri, part_size,
                    frac_scale_height, number_density, sigma=part_size_std)
                absorption_coeff += mx.take(mie_absorption[0], mx.array(node_P - P_cond.start), axis=0)
                absorption_coeff += self._get_scattering_absorption(
                    names, abund, node_T, node_P)

            else:
                absorption_coeff += self._get_scattering_absorption(names, abund,
                node_T, node_P, scattering_factor, scattering_slope,
                scattering_ref_wavelength)

        if add_collisional_absorption:
            absorption_coeff += self._get_collisional_absorption(
                names, abund, node_T, node_P)

        # Cross sections vary less than absorption coefficients by pressure
        # and temperature, so interpolation should be done with cross sections
        n_nodes = _mlx.asarray(_mlx.n_scaled(self.P_grid[node_P], self.T_grid[node_T]))
        cross_secs = absorption_coeff / n_nodes[:, np.newaxis]
        cross_secs = mx.maximum(cross_secs, max(min_cross_sec * _mlx.N_REF, _mlx.tiny()))
        ln_cross = mx.log(cross_secs)

        corners = mx.array(corners)
        corner_weights = _mlx.asarray(corner_weights)
        ln_cross_atm = sum(corner_weights[i][:, np.newaxis] * mx.take(ln_cross, corners[i], axis=0)
                           for i in range(4))

        cross_secs_atm = _mlx.exp(ln_cross_atm)
        absorption_coeff_atm = cross_secs_atm * _mlx.n_scaled(P_profile, T_profile)[:, np.newaxis]
        output_dict = {"absorption_coeff_atm": absorption_coeff_atm,
                       "radii": radii,
                       "dr": dr,
                       "P_profile": P_profile,
                       "T_profile": T_profile,
                       "mu_profile": mu_profile,
                       "atm_abundances": atm_abundances}

        return output_dict
