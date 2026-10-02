import numpy as np
from ._params import _UniformParam, _GaussianParam, _Param


class FitInfo:
    def __init__(self, guesses_dict):
        self.fit_param_names = []
        self.all_params = dict()
        self.gas_remainder = False

        for key in guesses_dict:
            self.all_params[key] = _Param(guesses_dict[key])

    def add_uniform_fit_param(self, name, low_lim, high_lim,
                              low_guess=None, high_guess=None):
        '''Fit for the parameter `name` using a uniform prior between `low_lim`
        and `high_lim`.  If using emcee, the walkers' initial values for this
        parameter are randomly selected to be between `low_guess` and
        `high_guess`.  If not specified, `low_guess` is set to `low_lim`, and
        similarly with `high_guess`.'''

        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))

        if low_guess is None:
            low_guess = low_lim
        if high_guess is None:
            high_guess = high_lim
        best_guess = self.all_params[name].best_guess

        self.fit_param_names.append(name)
        self.all_params[name] = _UniformParam(best_guess, low_lim, high_lim,
                                              low_guess, high_guess)

    def add_gaussian_fit_param(self, name, std, low_guess=None, high_guess=None):
        '''Fit for the parameter `name` using a Gaussian prior with standard
        deviation `std`.  If using emcee, the walkers' initial values for this
        parameter are randomly selected to be between `low_guess` and
        `high_guess`.  If `low_guess` is None, it is set to mean-2*std; if
        `high_guess` is None, it is set to mean+2*std.'''

        if name in self.fit_param_names:
            raise ValueError("Already fitting for {0}".format(name))

        mean = self.all_params[name].best_guess
        if low_guess is None:
            low_guess = mean - 2 * std
        if high_guess is None:
            high_guess = mean + 2 * std

        self.fit_param_names.append(name)
        self.all_params[name] = _GaussianParam(
            mean, std, low_guess, high_guess)

    def add_gases_clr(self, gases, low_lim=1e-12, remainder=False):
        '''Fit for the centered log-ratios (CLRs) of the abundances of
        `gases`, for a free retrieval with fit_clr=True.  The CLR of the
        last gas is minus the sum of the others.

        If `remainder` is True, the CLRs of all of `gases` are fitted, and an
        unretrieved remainder gas takes the place of the last CLR component.
        See `add_gases_vmr` for what this requires.'''
        self.gases = gases
        self.clr_low_lim = low_lim
        self.gas_remainder = remainder
        fitted = gases if remainder else gases[:-1]
        ln_limit = np.log(np.min(low_lim))
        n = len(fitted) + 1
        clr_min =  (n-1)/n * (ln_limit + np.log(n-1))
        clr_max = -(n-1)/n * ln_limit
        for g in fitted:
            self.all_params[f'clr_{g}'] = _Param(0)
            self.add_uniform_fit_param(f'clr_{g}', clr_min, clr_max)

    def add_gases_vmr(self, gases, low_lim, high_lim, remainder=False):
        '''Fit for the log10 volume mixing ratios (VMRs) of `gases`, with
        uniform priors between log10(low_lim) and log10(high_lim), for a free
        retrieval with fit_vmr=True.  The last gas is the background gas: it
        is not fitted, and its VMR is 1 minus the sum of the others.

        If `remainder` is True, the VMRs of all of `gases` are fitted, and
        they need not sum to 1.  The rest of the atmosphere is an unretrieved
        remainder gas, with no opacity, whose molecular weight follows from
        the mean molecular weight.  The mean molecular weight must therefore
        be set or fitted, as either "mmw" or "inv_mmw" (see
        CombinedRetriever.get_default_fit_info).  Samples where the VMRs sum
        to 1 or more, or where the remainder gas would have a negative
        molecular weight, are rejected.'''
        self.gases = gases
        self.gas_remainder = remainder
        fitted = gases if remainder else gases[:-1]
        for g in fitted:
            self.all_params[f'log_{g}'] = _Param(np.log10(low_lim))
            self.add_uniform_fit_param(f'log_{g}', np.log10(low_lim), np.log10(high_lim))
        
    def _interpret_param_array(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueError("Fit array invalid")

        result = dict()
        for i, key in enumerate(self.fit_param_names):
            result[key] = array[i]

        for key in self.all_params:
            if key not in result:
                result[key] = self.all_params[key].best_guess

        return result

    def _within_limits(self, array):
        if len(array) != len(self.fit_param_names):
            raise ValueError("Fit array invalid")

        for i, key in enumerate(self.fit_param_names):
            if not self.all_params[key].within_limits(array[i]):
                return False

        return True

    def _generate_rand_param_arrays(self, num_arrays):
        result = []

        for i in range(num_arrays):
            row = []
            for name in self.fit_param_names:
                if i == 0:
                    # Have one walker with fiducial value
                    row.append(self.all_params[name].best_guess)
                else:
                    row.append(self.all_params[name].get_random_value())
            result.append(row)

        return np.array(result)

    def _get(self, name):
        return self.all_params[name].best_guess

    def _get_num_fit_params(self):
        return len(self.fit_param_names)

    def _from_unit_interval(self, index, u):
        name = self.fit_param_names[index]
        return self.all_params[name].from_unit_interval(u)

    def _ln_prior(self, array):
        result = 0
        for i, name in enumerate(self.fit_param_names):
            result += self.all_params[name].ln_prior(array[i])

        return result

    def __repr__(self):
        return "Params to fit: {}; all params: {}".format(self.fit_param_names,
                                                          self.all_params)
