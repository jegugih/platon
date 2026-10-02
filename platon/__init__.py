name = "platon"
__version__ = "6.3.1"
__md5sum__ = "4deb845a63909ed8d0a74de2af5f0a48"
__data_url__ = "https://astro.uchicago.edu/~mz/data_{}.zip".format(__md5sum__)


def set_precision(precision):
    '''Choose "float32" (GPU, the default) or "float64" (CPU) for spectral
    arrays.  Call before creating any calculator.  See platon/_mlx.py.'''
    from ._mlx import set_precision as _set_precision
    _set_precision(precision)
