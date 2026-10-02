import math

import mlx.core as mx

from . import _mlx


def _hist_bin_fd(x):
    iqr = (_mlx.percentile(x, 75) - _mlx.percentile(x, 25)).item()
    return 2 * iqr * x.size**(-1./3)


def _hist_bin_sturges(x):
    return (mx.max(x).item() - mx.min(x).item()) / (math.log2(x.size) + 1.)


def _hist_bin_auto(x):
    fd_bw = _hist_bin_fd(x)
    sturges_bw = _hist_bin_sturges(x)
    if fd_bw > 0:
        return min(fd_bw, sturges_bw)

    else:
        return sturges_bw


def get_num_bins(x):
    width = _hist_bin_auto(x)
    n = int(math.ceil((mx.max(x).item() - mx.min(x).item()) / width))
    return n
