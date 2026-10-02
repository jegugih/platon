'''MLX backend for PLATON's forward model.

Data files are read with NumPy, converted to MLX arrays once, and all
per-call radiative transfer then runs in MLX.  Public methods still return
NumPy arrays, so samplers and plotting code are unaffected.

Precision
---------
Metal GPUs have no float64, so by default spectral arrays are float32 and
live on the GPU.  To keep float32 safe:

* Number densities are carried in units of N_REF (see `n_scaled`) and cross
  sections in units of 1/N_REF, which keeps CIA coefficients (~1e-60),
  squared polarizabilities (~1e-60) and their products in range.
* Abundances are carried as log10 values, so floors such as 1e-99 are exact.
* Small arrays that are prone to cancellation are always float64: ray path
  lengths on the CPU stream (FLOAT64_STREAM), where MLX supports float64,
  and the hydrostatic radii and T/P profiles in NumPy, because for ~100
  values MLX's fixed cost per operation would dominate.

Call ``set_precision("float64")`` before creating a calculator to run
everything in float64 on the CPU, e.g. to validate against NumPy PLATON.
'''
import math

import mlx.core as mx
import numpy as np

from .constants import k_B

N_REF = 1e25
'''Reference number density (m^-3) used to scale densities and cross sections'''

FLOAT64_STREAM = mx.cpu

_dtype = mx.float32

# Smallest value that floors may take; anything smaller flushes to zero on the
# GPU (Metal does not keep float32 denormals).
_TINY = {mx.float32: 1e-37, mx.float64: 1e-300}


def set_precision(precision):
    '''Choose "float32" (GPU, default) or "float64" (CPU) for spectral arrays.

    Must be called before any calculator is created, because opacity tables
    are converted to the chosen dtype when they are loaded.  This also changes
    MLX's default device for the whole process.
    '''
    global _dtype
    if precision == "float32":
        _dtype = mx.float32
        mx.set_default_device(mx.gpu)
    elif precision == "float64":
        _dtype = mx.float64
        mx.set_default_device(mx.cpu)
    else:
        raise ValueError("precision must be 'float32' or 'float64'")


def dtype():
    return _dtype


def tiny():
    return _TINY[_dtype]


def asarray(x, dtype=None):
    '''Convert x to an MLX array of the spectral dtype (or `dtype`).'''
    if dtype is None:
        dtype = _dtype
    if isinstance(x, mx.array):
        return x if x.dtype == dtype else x.astype(dtype)
    # MLX cannot import NumPy views with negative strides, e.g. a[::-1]
    return mx.array(np.ascontiguousarray(x, dtype=np.float64), dtype=dtype)


def to_numpy(x):
    '''Evaluate x and return it as a float64 NumPy array (or pass through).'''
    if isinstance(x, mx.array):
        if x.dtype == mx.float32:
            return np.array(x).astype(np.float64)
        return np.array(x)
    return x


def to_float64(x):
    '''Cast to float64 on the CPU stream.'''
    if isinstance(x, mx.array):
        return x.astype(mx.float64, stream=FLOAT64_STREAM)
    return mx.array(np.ascontiguousarray(x, dtype=np.float64), dtype=mx.float64)


def from_float64(x):
    '''Cast a float64 array back to the spectral dtype.'''
    return x.astype(_dtype, stream=FLOAT64_STREAM)


def n_scaled(P, T):
    '''Number density P / (k_B T) in units of N_REF.'''
    return P / (k_B * N_REF * T)


def as_float(x):
    '''Python float (None passes through).  Parameters must not stay NumPy
    scalars: np.float64(2) * mx.array(...) is computed by NumPy and returns a
    NumPy array, while 2.0 * mx.array(...) stays in MLX.'''
    return None if x is None else float(x)


# MLX 0.32's float64 exp (CPU) is only float32-accurate, while its float64
# power is accurate; and its float32 expm1 (GPU) loses accuracy above |x| ~ 0.1,
# where exp(x) - 1 is fine.  Use these instead of mx.exp / mx.expm1.
def exp(x):
    x = x if isinstance(x, mx.array) else asarray(x)
    if x.dtype == mx.float64:
        return mx.power(math.e, x)
    return mx.exp(x)


def expm1(x):
    x = x if isinstance(x, mx.array) else asarray(x)
    if x.dtype == mx.float64:
        return mx.expm1(x)
    return mx.where(mx.abs(x) < 0.1, mx.expm1(x), mx.exp(x) - 1)


def logspace(start, stop, num):
    # Constant grids are built in float64, then rounded once
    return asarray(np.logspace(start, stop, num))


def linear_weights(targets, xs):
    '''Lower/upper indices and fractions for linear interpolation of the
    sorted 1D grid `xs` at `targets`, clamped at the edges like np.interp.

    Exact grid points (and clamped targets) get upper == lower, so the
    neighbouring value is never touched, as with floor/ceil indexing.
    '''
    n = xs.shape[0]
    if n == 1:
        zero = mx.zeros(targets.shape, dtype=mx.int32)
        return zero, zero, mx.zeros(targets.shape, dtype=targets.dtype)

    i = mx.searchsorted(xs, targets, side="right").astype(mx.int32) - 1
    i = mx.clip(i, 0, n - 2)
    x0 = mx.take(xs, i)
    x1 = mx.take(xs, i + 1)
    # Fraction is computed directly rather than from a fractional index, so
    # it keeps full precision regardless of how far along the grid it is
    frac = mx.clip((targets - x0) / (x1 - x0), 0, 1)
    at_upper = frac >= 1
    lower = mx.where(at_upper, i + 1, i)
    frac = mx.where(at_upper, 0, frac)
    upper = mx.where(frac > 0, lower + 1, lower)
    return lower, upper, frac


def interp(x, xp, fp, left=None, right=None):
    '''np.interp for MLX: 1D piecewise-linear interpolation, any shape x.
    Works in x's dtype, so float64 inputs stay float64.'''
    x = x if isinstance(x, mx.array) else asarray(x)
    xp = asarray(xp, x.dtype)
    fp = asarray(fp, x.dtype)
    shape = x.shape
    flat = x.reshape(-1)
    lower, upper, frac = linear_weights(flat, xp)
    result = mx.take(fp, lower) * (1 - frac) + mx.take(fp, upper) * frac
    if left is not None:
        result = mx.where(flat < xp[0], left, result)
    if right is not None:
        result = mx.where(flat > xp[-1], right, result)
    return result.reshape(shape)


class LinearResampler:
    '''Linear interpolation between two fixed grids.  Weights are computed
    once, in float64, so applying it is two gathers and a blend.'''

    def __init__(self, new_xs, old_xs):
        idx = np.interp(np.asarray(new_xs, dtype=np.float64),
                        np.asarray(old_xs, dtype=np.float64),
                        np.arange(len(old_xs)))
        lower = np.floor(idx).astype(np.int32)
        upper = np.ceil(idx).astype(np.int32)
        self.lower = mx.array(lower)
        self.upper = mx.array(upper)
        self.frac = asarray(idx - lower)

    def __call__(self, data, axis=-1):
        shape = [1] * data.ndim
        shape[axis] = -1
        frac = self.frac.reshape(shape)
        return mx.take(data, self.lower, axis=axis) * (1 - frac) + \
            mx.take(data, self.upper, axis=axis) * frac


def trapz(y, x, axis=-1):
    '''np.trapz along `axis`, with 1D sample points x.'''
    y = mx.moveaxis(y, axis, -1)
    return mx.sum((y[..., 1:] + y[..., :-1]) * mx.diff(x) / 2, axis=-1)


def average(a, weights, axis):
    '''np.average with 1D weights along `axis`.'''
    shape = [1] * a.ndim
    shape[axis] = -1
    return mx.sum(a * weights.reshape(shape), axis=axis) / mx.sum(weights)


def percentile(a, q):
    '''np.percentile of the flattened array (default "linear" method).'''
    a = mx.sort(a.reshape(-1))
    pos = q / 100.0 * (a.shape[0] - 1)
    lower = int(math.floor(pos))
    upper = min(lower + 1, a.shape[0] - 1)
    frac = pos - lower
    return a[lower] * (1 - frac) + a[upper] * frac


def _reflect_indices(n, radius):
    # scipy.ndimage mode="reflect": (d c b a | a b c d | d c b a)
    idx = np.arange(-radius, n + radius) % (2 * n)
    idx = np.where(idx >= n, 2 * n - 1 - idx, idx)
    return mx.array(idx.astype(np.int32))


def gaussian_filter(a, sigma, truncate=4.0):
    '''scipy.ndimage.gaussian_filter with mode="reflect", applied along every
    axis with the same sigma.'''
    radius = int(truncate * float(sigma) + 0.5)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 / sigma**2 * offsets**2)
    weights /= weights.sum()

    for axis in range(a.ndim):
        n = a.shape[axis]
        padded = mx.take(a, _reflect_indices(n, radius), axis=axis)
        padded = mx.moveaxis(padded, axis, 0)
        result = mx.zeros(padded[radius:radius + n].shape, dtype=a.dtype)
        for k, w in enumerate(weights):
            result = result + float(w) * padded[k:k + n]
        a = mx.moveaxis(result, 0, axis)
    return a


class PiecewisePolynomial:
    '''MLX evaluation of a scipy spline, via scipy.interpolate.PPoly.

    Out-of-range points are extrapolated from the end polynomials, as
    BSpline does by default.
    '''

    def __init__(self, spline):
        import scipy.interpolate
        ppoly = scipy.interpolate.PPoly.from_spline(spline)
        breaks = ppoly.x
        coeffs = ppoly.c
        # Drop the zero-width intervals at the padded ends of the knot vector
        keep = np.diff(breaks) > 0
        starts = breaks[:-1][keep]
        self.breaks = asarray(starts)
        self.coeffs = asarray(coeffs[:, keep])
        self.order = coeffs.shape[0]

    def __call__(self, x):
        n = self.breaks.shape[0]
        i = mx.searchsorted(self.breaks, x, side="right").astype(mx.int32) - 1
        i = mx.clip(i, 0, n - 1)
        dx = x - mx.take(self.breaks, i)
        result = mx.take(self.coeffs[0], i)
        for k in range(1, self.order):
            result = result * dx + mx.take(self.coeffs[k], i)
        return result
