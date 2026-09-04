import adora_precision
import jax
import jax.numpy as jnp
from jax.lax import fori_loop


PRECISION = adora_precision.configured_precision()


def _validate_inputs(dz, emis, opac):
    dz = jnp.asarray(dz)
    emis = jnp.asarray(emis)
    opac = jnp.asarray(opac)
    if dz.ndim != 1 or emis.ndim != 1 or opac.ndim != 1:
        raise ValueError("dz, emis, and opac must be one-dimensional arrays")
    if dz.shape != emis.shape or dz.shape != opac.shape:
        raise ValueError("dz, emis, and opac must have identical shapes")
    if dz.shape[0] == 0:
        raise ValueError("dz, emis, and opac must contain at least one layer")
    return dz, emis, opac


def _one_minus_exp_neg_over_x(x):
    """Evaluate ``(1 - exp(-x)) / x`` with a finite value and derivative at 0."""
    threshold = jnp.asarray(1e-4, dtype=x.dtype)
    small = jnp.abs(x) < threshold
    safe_x = jnp.where(small, jnp.ones_like(x), x)
    direct = -jnp.expm1(-x) / safe_x
    x2 = x * x
    one = jnp.ones_like(x)
    series = one - x / 2 + x2 / 6 - x2 * x / 24 + x2 * x2 / 120
    return jnp.where(small, series, direct)


def _prepare_inputs(dz, emis, opac, I_start):
    dz, emis, opac = _validate_inputs(dz, emis, opac)
    dtype = jnp.result_type(dz, emis, opac, I_start, jnp.float32)
    dz = dz.astype(dtype)
    emis = emis.astype(dtype)
    opac = opac.astype(dtype)
    I_start = jnp.asarray(I_start, dtype=dtype)
    if I_start.ndim != 0:
        raise ValueError("I_start must be a scalar")
    return dz, emis, opac, I_start


def cumsum_fs(dz, emis, opac, I_start=0.0):
    """Integrate bottom-to-top scalar transfer with piecewise-constant layers.

    ``dz``, ``emis``, and ``opac`` describe successive layers in propagation
    order. ``I_start`` is the intensity incident on the first layer.
    """
    dz, emis, opac, I_start = _prepare_inputs(dz, emis, opac, I_start)
    dtau = opac * dz
    # Build a genuinely exclusive reverse sum.  Subtracting each layer from an
    # inclusive cumulative sum loses the optical depth above it when the local
    # layer is many orders of magnitude thicker than its neighbours.
    tau_above_layer = jnp.concatenate(
        (jnp.cumsum(dtau[::-1])[:-1][::-1], jnp.zeros(1, dtype=dtau.dtype))
    )
    transmittance = jnp.exp(-tau_above_layer)
    local_contribution = emis * dz * _one_minus_exp_neg_over_x(dtau)
    outgoing_contribution = local_contribution * transmittance
    I = I_start * jnp.exp(-jnp.sum(dtau)) + jnp.sum(outgoing_contribution)
    return I


def nearest_fs(dz, emis, opac, I_start=0.0):
    """Iteratively integrate the same layer model as :func:`cumsum_fs`."""
    dz, emis, opac, I_start = _prepare_inputs(dz, emis, opac, I_start)

    def body(i, intens):
        dtau = opac[i] * dz[i]
        local_contribution = emis[i] * dz[i] * _one_minus_exp_neg_over_x(dtau)
        return intens * jnp.exp(-dtau) + local_contribution

    result = fori_loop(
        0,
        dz.shape[0],
        body,
        I_start,
    )
    return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is None:
        plt.ion()
    else:
        ipython.run_line_magic("matplotlib", "")

    n_depth = 20
    n_wave = 101
    wave = jnp.linspace(-5, 5, n_wave)
    profile = jnp.exp(-wave**2)
    eta = jnp.ones((n_wave, n_depth)) * 1e-5 * profile[:, None] + 1e-6
    chi = jnp.ones((n_wave, n_depth)) * 1e-3 * profile[:, None] + 1e-6
    # eta = jnp.ones((n_wave, n_depth)) * 1e-5 * profile[:, None]
    # chi = jnp.ones((n_wave, n_depth)) * 1e-3 * profile[:, None]
    dz = jnp.ones(n_depth) * 1e2

    fs_fn = cumsum_fs
    fs_fn = nearest_fs
    fs = jax.jit(
        jax.vmap(
            fs_fn,
            in_axes=[None, 0, 0],
            out_axes=0
        )
    )
    dfs = jax.jit(
        jax.vmap(
            jax.jacrev(
                fs_fn,
                argnums=(1, 2)
            ),
            in_axes=[None, 0, 0],
            out_axes=0,
        )
    )

    I = fs(dz, eta, chi)
    dI = dfs(dz, eta, chi)

    plt.figure()
    plt.plot(wave, I)

    plt.figure()
    plt.imshow(dI[0])
    plt.colorbar()
    plt.title("dI/deta")

    plt.figure()
    plt.imshow(dI[1])
    plt.colorbar()
    plt.title("dI/dchi")
