'''Special functions missing from MLX.'''
import mlx.core as mx

from . import _mlx

_EULER = 0.5772156649015329

# Power series below `split`, continued fraction above, with fixed iteration
# counts so every element does the same work.  The continued fraction
# converges slowest at the split, the series cancels more the larger it is.
# These give a maximum relative error of ~2e-6 (float32) and ~1e-14
# (float64).  The loops are not passed to mx.compile: JIT-compiling the
# unrolled graph takes over 10 s, while the uncompiled version runs in ~2 ms
# for 1e5 points.
#                    split, continued fraction iterations, series terms
_ITERATIONS = {mx.float32: (1.0, 30, 12), mx.float64: (2.0, 48, 24)}


def expn(n, x):
    '''Exponential integral E_n(x) for integer n >= 1 and x >= 0, like
    scipy.special.expn.'''
    x = x if isinstance(x, mx.array) else _mlx.asarray(x)
    n = int(n)
    split, cf_iters, series_iters = _ITERATIONS[x.dtype]
    nm1 = n - 1
    psi = -_EULER + sum(1.0 / i for i in range(1, nm1 + 1))

    # Continued fraction (modified Lentz) for x > split
    x_cf = mx.maximum(x, split)
    b = x_cf + n
    c = mx.ones_like(x) * 1e30
    d = 1 / b
    h = d
    for i in range(1, cf_iters + 1):
        an = -i * (nm1 + i)
        b = b + 2
        d = 1 / (an * d + b)
        c = b + an / c
        h = h * c * d
    cf = h * _mlx.exp(-x_cf)

    # Power series for 0 < x <= split
    x_s = mx.where(x > 0, mx.minimum(x, split), split)
    log_x = mx.log(x_s)
    series = -log_x - _EULER if nm1 == 0 else mx.ones_like(x) / nm1
    fact = mx.ones_like(x)
    for i in range(1, series_iters + 1):
        fact = fact * (-x_s / i)
        if i != nm1:
            series = series - fact / (i - nm1)
        else:
            series = series + fact * (-log_x + psi)

    result = mx.where(x > split, cf, series)
    at_zero = mx.inf if n == 1 else 1.0 / nm1
    return mx.where(x == 0, at_zero, result)
