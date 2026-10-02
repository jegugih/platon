import mlx.core as mx
import numpy as np

from . import _mlx


def get_condition_slice(target_data, interp_data, max_cutoff=np.inf):
    '''Returns the slice of interp_data (a sorted NumPy grid) that brackets
    every value in target_data, stopping at max_cutoff.  This is the same
    range of grid points as the boolean mask the NumPy version used.'''
    target_min = float(np.min(_mlx.to_numpy(target_data)))
    target_max = float(np.max(_mlx.to_numpy(target_data)))

    start = np.searchsorted(interp_data, target_min, side="left")
    if start == len(interp_data):
        start = 0
    elif interp_data[start] != target_min:
        start = max(0, start - 1)

    end = np.searchsorted(interp_data, min(target_max, max_cutoff),
                          side="left")
    end = len(interp_data) if end == len(interp_data) else end + 1
    return slice(int(start), int(end))


def _host_weights(target, xs):
    # Scalar targets: same float64 arithmetic as the NumPy version
    index = np.interp(target, np.asarray(xs, dtype=np.float64),
                      np.arange(len(xs)))
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    return lower, upper, float(index - lower)


def _is_scalar(x):
    return np.isscalar(x) or (isinstance(x, (np.ndarray, mx.array)) and x.ndim == 0)


def interp1d(target_xs, xs, data, assume_sorted=True):
    '''Linear interpolation of data (first axis indexed by xs) to target_xs.'''
    assert(xs.shape[0] == data.shape[0])
    if not assume_sorted:
        sort = np.argsort(_mlx.to_numpy(xs))
        xs = xs[sort]
        data = mx.take(data, mx.array(sort.astype(np.int32)), axis=0)

    if _is_scalar(target_xs) and not isinstance(target_xs, mx.array):
        lower, upper, frac = _host_weights(target_xs, _mlx.to_numpy(xs))
        return data[lower] * (1 - frac) + data[upper] * frac

    target_xs = mx.atleast_1d(_mlx.asarray(target_xs))
    lower, upper, frac = _mlx.linear_weights(target_xs, _mlx.asarray(xs))
    frac = frac.reshape((-1,) + (1,) * (data.ndim - 1))
    return mx.take(data, lower, axis=0) * (1 - frac) + \
        mx.take(data, upper, axis=0) * frac


def regular_grid_interp(ys, xs, data, target_ys, target_xs):
    '''Bilinear interpolation of data, whose first two axes are indexed by
    ys and xs, to the points (target_ys, target_xs).  Trailing axes of data
    are carried along.  Scalar targets return a single point.'''
    assert(data.shape[0] == len(ys) and data.shape[1] == len(xs))

    if _is_scalar(target_ys) and not isinstance(target_ys, mx.array) and \
       _is_scalar(target_xs) and not isinstance(target_xs, mx.array):
        y0, y1, fy = _host_weights(target_ys, _mlx.to_numpy(ys))
        x0, x1, fx = _host_weights(target_xs, _mlx.to_numpy(xs))
        return data[y0, x0] * (1 - fy) * (1 - fx) + \
            data[y1, x0] * fy * (1 - fx) + \
            data[y0, x1] * (1 - fy) * fx + \
            data[y1, x1] * fy * fx

    isscalar = _is_scalar(target_ys)
    target_ys = mx.atleast_1d(_mlx.asarray(target_ys))
    target_xs = mx.atleast_1d(_mlx.asarray(target_xs))
    assert(target_ys.shape == target_xs.shape)

    y0, y1, fy = _mlx.linear_weights(target_ys, _mlx.asarray(ys))
    x0, x1, fx = _mlx.linear_weights(target_xs, _mlx.asarray(xs))

    # Gather whole trailing blocks (e.g. every wavelength) per target point
    trailing = data.shape[2:]
    flat = data.reshape((data.shape[0] * data.shape[1],) + trailing)
    n_x = data.shape[1]

    def corner(y, x):
        return mx.take(flat, y * n_x + x, axis=0)

    shape = (-1,) + (1,) * len(trailing)
    fy = fy.reshape(shape)
    fx = fx.reshape(shape)
    result = corner(y0, x0) * (1 - fy) * (1 - fx) + \
        corner(y1, x0) * fy * (1 - fx) + \
        corner(y0, x1) * (1 - fy) * fx + \
        corner(y1, x1) * fy * fx
    if isscalar:
        return result[0]
    return result
