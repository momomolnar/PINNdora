import adora_precision
import jax
import jax.numpy as jnp
from jax.lax import cond, fori_loop
from jax.scipy.linalg import expm


PRECISION = adora_precision.configured_precision()

def stokes_K(opac):
    opac = jnp.asarray(opac)
    if opac.shape != (7,):
        raise ValueError("opac must have shape (7,)")
    eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V = opac
    # NOTE(cmo): Propagation matrix
    K = jnp.stack([
        jnp.stack([eta_I, eta_Q, eta_U, eta_V]),
        jnp.stack([eta_Q, eta_I, rho_V, -rho_U]),
        jnp.stack([eta_U, -rho_V, eta_I, rho_Q]),
        jnp.stack([eta_V, rho_U, -rho_Q, eta_I]),
    ])
    return K


def _prepare_inputs(dz, I_start, emis, opac):
    dz = jnp.asarray(dz)
    I_start = jnp.asarray(I_start)
    emis = jnp.asarray(emis)
    opac = jnp.asarray(opac)
    if dz.ndim != 1:
        raise ValueError("dz must be a one-dimensional array")
    if dz.shape[0] == 0:
        raise ValueError("the polarized depth grid must contain at least one point")
    if I_start.shape != (4,):
        raise ValueError("I_start must have shape (4,)")
    if emis.shape != (dz.shape[0], 4):
        raise ValueError("emis must have shape (len(dz), 4)")
    if opac.shape != (dz.shape[0], 7):
        raise ValueError("opac must have shape (len(dz), 7)")

    dtype = jnp.result_type(dz, I_start, emis, opac, jnp.float32)
    dz = dz.astype(dtype)
    I_start = I_start.astype(dtype)
    emis = emis.astype(dtype)
    opac = opac.astype(dtype)
    return dz, I_start, emis, opac, dtype


def _delo_constant_fs_impl(
    dz,
    I_start,
    emis,
    opac,
    *,
    singular_fallback,
):
    dz, I_start, emis, opac, dtype = _prepare_inputs(
        dz, I_start, emis, opac
    )
    Id = jnp.eye(4, dtype=dtype)
    low_tau_threshold = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    opacity_floor = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype))

    def body(i, intens):
        Km = stokes_K(opac[i - 1])
        K = stokes_K(opac[i])
        eta_I_m = Km[0, 0]
        eta_I = K[0, 0]
        dtau = 0.5 * (eta_I + eta_I_m) * dz[i]

        def constant_coefficient_step(current_intens):
            # DELO normalization is undefined when either endpoint eta_I is
            # zero.  Integrate the interval with averaged physical
            # coefficients instead.  The augmented matrix exponential solves
            # dI/ds = epsilon - K I without dividing by K, so it remains exact
            # for constant coefficients in vacuum, pure emission, and thick
            # absorption and stays well behaved for singular matrices.
            mean_K = 0.5 * (Km + K)
            mean_emis = 0.5 * (emis[i - 1] + emis[i])
            generator = jnp.zeros((5, 5), dtype=dtype)
            generator = generator.at[:4, :4].set(-mean_K)
            generator = generator.at[:4, 4].set(mean_emis)
            # JAX's default of 16 scaling/squaring steps returns NaNs for very
            # thick but otherwise benign intervals.  Sixty-four covers optical
            # depths far beyond the atmospheric regime while retaining the
            # correct underflow-to-zero limit.
            propagator = expm(generator * dz[i], max_squarings=64)
            augmented_intensity = jnp.concatenate(
                (current_intens, jnp.ones(1, dtype=dtype))
            )
            return (propagator @ augmented_intensity)[:4]

        def geometrical_trapezoid(current_intens):
            # For optically thin intervals this second-order direct form avoids
            # DELO normalization roundoff without the cost of a matrix
            # exponential.  It is selected only by small |dtau|, never merely
            # because an endpoint opacity is singular.
            Phi_m = Id - 0.5 * dz[i] * Km
            Phi = Id + 0.5 * dz[i] * K
            source = 0.5 * dz[i] * (emis[i - 1] + emis[i])
            return jnp.linalg.solve(Phi, Phi_m @ current_intens + source)

        def delo_step(current_intens):
            # Keep this branch finite even when a surrounding vmap lowers cond
            # to a select and evaluates both branches.
            safe_eta_I_m = jnp.where(
                jnp.abs(eta_I_m) <= opacity_floor,
                jnp.ones_like(eta_I_m),
                eta_I_m,
            )
            safe_eta_I = jnp.where(
                jnp.abs(eta_I) <= opacity_floor,
                jnp.ones_like(eta_I),
                eta_I,
            )
            source_m = emis[i - 1] / safe_eta_I_m
            source = emis[i] / safe_eta_I

            # Modified propagation matrices for DELO-constant.
            K_prime = K / safe_eta_I - Id
            Km_prime = Km / safe_eta_I_m - Id

            # Janett et al. (2017), Appendix B.
            E_k = jnp.exp(-dtau)
            F_k = -jnp.expm1(-dtau)
            Phi_k = E_k * Id - 0.5 * F_k * Km_prime
            Phi_kp = Id + 0.5 * F_k * K_prime
            Psi_k = 0.5 * F_k * source_m
            Psi_kp = 0.5 * F_k * source
            return jnp.linalg.solve(Phi_kp, Phi_k @ current_intens + Psi_k + Psi_kp)

        def nonsingular_step(current_intens):
            return cond(
                jnp.abs(dtau) <= low_tau_threshold,
                geometrical_trapezoid,
                delo_step,
                current_intens,
            )

        # ``singular_fallback`` is a Python boolean, so the non-singular entry
        # point does not trace (and therefore does not compile or execute) the
        # costly matrix exponential.  This matters under an outer vmap: JAX
        # batches a data-dependent cond into selection logic that otherwise
        # evaluates the fallback for every ordinary atmosphere in the batch.
        if not singular_fallback:
            return nonsingular_step(intens)

        singular_endpoint = (
            (jnp.abs(eta_I_m) <= opacity_floor)
            | (jnp.abs(eta_I) <= opacity_floor)
        )
        return cond(
            singular_endpoint,
            constant_coefficient_step,
            nonsingular_step,
            intens,
        )


    # NOTE(cmo): Loop from a starting index of 1 assuming I_start at lower boundary
    intens = fori_loop(
        1,
        dz.shape[0],
        body,
        I_start
    )
    return intens


def delo_constant_fs(dz, I_start, emis, opac):
    """Integrate polarized transfer, including singular-opacity intervals.

    Arrays are ordered from the lower boundary toward the observer. ``dz[i]``
    is the geometrical distance from point ``i - 1`` to point ``i``; ``dz[0]``
    is unused because ``I_start`` is already defined at the first point.

    Use :func:`delo_constant_fs_nonsingular` in performance-sensitive batched
    synthesis when Stokes-I opacity is known to be nonzero at every depth.
    """
    return _delo_constant_fs_impl(
        dz,
        I_start,
        emis,
        opac,
        singular_fallback=True,
    )


def delo_constant_fs_nonsingular(dz, I_start, emis, opac):
    """Fast polarized transfer for atmospheres with nonzero Stokes-I opacity.

    This specialization has the same DELO-constant and optically thin steps as
    :func:`delo_constant_fs`, but deliberately omits its matrix-exponential
    fallback.  The caller must guarantee that ``abs(opac[:, 0])`` is greater
    than ``sqrt(finfo(dtype).tiny)`` at every depth point.  In return, an outer
    :func:`jax.vmap` cannot speculatively execute the expensive fallback for
    otherwise ordinary atmospheres.
    """
    return _delo_constant_fs_impl(
        dz,
        I_start,
        emis,
        opac,
        singular_fallback=False,
    )

if __name__ == "__main__":
    # Phi_{k+1} I_{k+1} = Phi_k I_k + Psi_{k+1} + Psi_k
    opac_grid = jnp.array([
        [1.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
        [1.1, 0.1, 0.1, 0.1, 0.04, 0.04, 0.04],
        [1.2, 0.1, 0.1, 0.1, 0.03, 0.03, 0.03],
        [1.3, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02],
        [1.4, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
    ])

    epsilon_grid = jnp.array([
        [0.5, 0.05, 0.05, 0.05],
        [0.6, 0.06, 0.06, 0.06],
        [0.7, 0.07, 0.07, 0.07],
        [0.8, 0.08, 0.08, 0.08],
        [0.9, 0.09, 0.09, 0.09]
    ])

    dz = jnp.ones(5) * 0.5
    I_start = jnp.array([1.0, 0.0, 0.0, 0.0])

    I_final = delo_constant_fs(dz, I_start, epsilon_grid, opac_grid)
