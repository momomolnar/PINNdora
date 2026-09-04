import adora_precision
adora_precision.configure_precision()
import jax
import jax.numpy as jnp
from adora_data import FE_I_6301_6302_LINE_LIST
from atmosphere import atmosphere_from_falc
from lineop import read_kurucz
from response_fn import lte_rt
import jaxopt
from jaxopt import LevenbergMarquardt
# NOTE(cmo): Didn't have much success with optimistix, even after downgrading jax to 0.6.2 to work around a bug
# https://github.com/jax-ml/jax/issues/31448
# https://github.com/patrick-kidger/optimistix/issues/151
# The LM approach _does_ work, but it computes the jacobian/hessian in a way
# that ends up being costly per iteration vs the faster jacobian used for the
# response functions. We may need our own

def pack_params(temperature, ne, nhtot, vz, vturb):
    return jnp.stack([
        temperature,
        ne,
        nhtot,
        vz,
        vturb,
    ]).reshape(-1)

def params_to_tuple(params):
    params = jnp.asarray(params)
    if params.ndim != 1 or params.shape[0] % 5 != 0:
        raise ValueError("params must be a one-dimensional array with length divisible by 5")
    p = params.reshape(5, -1)
    return (
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
    )


def physical_to_unconstrained(params):
    """Map physical atmospheric parameters to finite optimizer coordinates.

    Temperature, densities, velocity, and microturbulence are mapped onto the
    same conservative bounds used by the bounded SciPy example.  Densities are
    transformed logarithmically because their useful ranges span many orders
    of magnitude.
    """
    temperature, ne, nhtot, vz, vturb = params_to_tuple(params)
    dtype = jnp.result_type(params, jnp.float32)
    eps = jnp.asarray(1e-6, dtype=dtype)

    def logit(fraction):
        fraction = jnp.clip(fraction, eps, 1.0 - eps)
        return jnp.log(fraction) - jnp.log1p(-fraction)

    def bounded(value, lower, upper):
        return logit((value - lower) / (upper - lower))

    def log_bounded(value, lower, upper):
        safe_value = jnp.clip(value, lower, upper)
        fraction = (jnp.log(safe_value) - jnp.log(lower)) / (
            jnp.log(upper) - jnp.log(lower)
        )
        return logit(fraction)

    return pack_params(
        bounded(temperature, 2.5e3, 100e3),
        log_bounded(ne, 1e16, 1e22),
        log_bounded(nhtot, 1e16, 1e24),
        bounded(vz, -20e3, 20e3),
        bounded(vturb, 0.0, 20e3),
    )


def unconstrained_to_physical(params):
    """Decode optimizer coordinates into a finite, physically valid atmosphere."""
    temperature, ne, nhtot, vz, vturb = params_to_tuple(params)

    def bounded(value, lower, upper):
        return lower + (upper - lower) * jax.nn.sigmoid(value)

    def log_bounded(value, lower, upper):
        log_value = bounded(value, jnp.log(lower), jnp.log(upper))
        return jnp.exp(log_value)

    return pack_params(
        bounded(temperature, 2.5e3, 100e3),
        log_bounded(ne, 1e16, 1e22),
        log_bounded(nhtot, 1e16, 1e24),
        bounded(vz, -20e3, 20e3),
        bounded(vturb, 0.0, 20e3),
    )


lte_rt_jit = jax.jit(
        lte_rt,
)

lte_rt_wave = jax.vmap(
    lte_rt_jit,
    in_axes=[None, 0, None, None, None, None, None, None]
)

def compute_residuals(params, target, lines, wavelengths, dz):
    params = params.reshape(5, -1)
    temperature = params[0]
    ne = params[1]
    nhtot = params[2]
    vz = params[3]
    vturb = params[4]

    lte_rt_wave = jax.vmap(
        lte_rt_jit,
        in_axes=[None, 0, None, None, None, None, None, None]
    )

    model = lte_rt_wave(lines, wavelengths, dz, temperature, ne, nhtot, vz, vturb)
    return model - target


def compute_residuals_unconstrained(params, target, lines, wavelengths, dz):
    physical_params = unconstrained_to_physical(params)
    return compute_residuals(physical_params, target, lines, wavelengths, dz)

def vmap_residuals(params, target, lines, wavelength, dz):
    params = params.reshape(5, -1)
    temperature = params[0]
    ne = params[1]
    nhtot = params[2]
    vz = params[3]
    vturb = params[4]

    model = lte_rt_jit(lines, wavelength, dz, temperature, ne, nhtot, vz, vturb)
    return model - target


compute_residuals_vmap = jax.jit(
    jax.vmap(
        vmap_residuals,
        in_axes=[None, 0, None, 0, None]
    )
)

# NOTE(cmo): Swapping the order of the jacobian application appears to be much
# faster as it prevents "false sharing" over wavelengths
compute_residuals_jac_vmap = jax.jit(
    jax.vmap(
        jax.jacrev(
            vmap_residuals,
            argnums=0
        ),
        in_axes=[None, 0, None, 0, None]
    )
)


if __name__ == "__main__":
    import lightweaver as lw
    from lightweaver.fal import Falc82
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

    lines = read_kurucz(FE_I_6301_6302_LINE_LIST)

    fal = Falc82()
    _, dz, temperature, ne, nhtot, vz, vturb = atmosphere_from_falc(fal)

    temperature_target = temperature.copy()
    temperature_target = temperature_target + jnp.sin(jnp.linspace(0.0, 6.0 * jnp.pi, temperature.shape[0])) * 400
    ne_target = jnp.where(
        jnp.arange(ne.shape[0]) < 30,
        ne * 1.04,
        ne
    )
    nhtot_target = nhtot.copy()
    vturb_target = vturb.copy()
    vz_target = jnp.linspace(1.0, 0.0, temperature.shape[0]) * 1.2e3

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    start = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
    target = lte_rt_wave(lines, waves, dz, temperature_target, ne_target, nhtot_target, vz_target, vturb_target)

    initial_params = pack_params(
        temperature=temperature,
        ne=ne,
        nhtot=nhtot,
        vz=vz,
        vturb=vturb
    )

    plt.figure()
    plt.plot(waves, start)
    plt.plot(waves, target)

    initial_unconstrained = physical_to_unconstrained(initial_params)
    solver = LevenbergMarquardt(
        residual_fun=jax.jit(compute_residuals_unconstrained),
        solver=jaxopt.linear_solve.solve_cg,
    )
    result = solver.run(initial_unconstrained, target, lines, waves, dz)
    fitted_params = unconstrained_to_physical(result.params)

    plt.plot(waves, lte_rt_wave(lines, waves, dz, *params_to_tuple(fitted_params)), '--')
