import mlx.core as mx

from . import _mlx


def get_dl(radii):
    '''Radii must be in descending order.  Returns dl[i,j], the distance travelled by a ray with impact parameter radii[j+1] from the shell at radii[i+1] to radii[i].
    Computed in float64 on the CPU stream.  r'^2 - r^2 is factored as
    (r' - r)(r' + r): squares of ~1e7 m radii cancel badly, and MLX's x**2 is
    not exactly rounded, so equal radii could give a nonzero length.'''
    with mx.stream(_mlx.FLOAT64_STREAM):
        radii = _mlx.to_float64(radii)
        r_mesh, r_prime_mesh = mx.meshgrid(radii, radii)
        r_prime = r_prime_mesh[:, 0:-1]
        r = r_mesh[:, 1:]
        sqr_length = (r_prime - r) * (r_prime + r)
        sqr_length = mx.maximum(sqr_length, 0)
        lengths = 2 * mx.sqrt(sqr_length)

        dl = lengths[0:-1] - lengths[1:]
    return dl


def get_line_of_sight_tau(absorption_coeff, radii):
    '''radii must be in descending order.  absorption_coeff must be an array whose first dimension is the number of wavelengths and second dimension is the number of radius points, which must be 1 less than the size of radii.'''
    with mx.stream(_mlx.FLOAT64_STREAM):
        geometry_radii = _mlx.to_float64(radii)
        assert(mx.all(mx.diff(geometry_radii) <= 0).item())
    absorption_coeff = _mlx.asarray(absorption_coeff)
    intermediate_coeff = 0.5 * \
        (absorption_coeff[0:-1] + absorption_coeff[1:])
    dl = _mlx.from_float64(get_dl(geometry_radii))
    return intermediate_coeff.T @ dl
