import adora_precision
adora_precision.configure_precision()
import jax
import jax.numpy as jnp
import astropy.constants as const

DEBROGLIE_CONST = (const.h / (2 * jnp.pi * const.k_B) * const.h / const.m_e).value
K_B_EV = const.k_B.to("eV / K").value


def lte_pops(
    energy,
    g,
    stage,
    temperature,
    ne,
    ntot,
):
    """Return normalized LTE level populations.

    Parameters use SI units except for ``energy``, which is in eV.  ``g`` is
    the statistical weight and ``stage`` is an integer ionization-stage index.
    Temperature, electron density, statistical weights, and total population
    must be positive (``ntot`` may also be zero).

    Relative Saha--Boltzmann weights are normalized in log space so highly
    ionized or weakly populated levels do not overflow before normalization.
    """
    energy = jnp.asarray(energy)
    g = jnp.asarray(g)
    stage = jnp.asarray(stage)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("energy must be a non-empty one-dimensional array")
    if g.shape != energy.shape or stage.shape != energy.shape:
        raise ValueError("energy, g, and stage must have identical shapes")

    k_B_T = temperature * K_B_EV
    log_saha_term = (
        jnp.log(0.5)
        + jnp.log(ne)
        + 1.5 * (jnp.log(DEBROGLIE_CONST) - jnp.log(temperature))
    )
    log_weights = (
        jnp.log(g) - jnp.log(g[0])
        - (energy - energy[0]) / k_B_T
        - (stage - stage[0]) * log_saha_term
    )
    log_norm = jax.scipy.special.logsumexp(log_weights)
    return ntot * jnp.exp(log_weights - log_norm)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import lightweaver as lw
    from lightweaver.rh_atoms import CaII_atom, H_6_atom
    from lightweaver.fal import Falc82
    import astropy.units as u

    Ca = CaII_atom()
    energies = jnp.array([
        (level.E_SI << u.Unit("J")).to("eV").value for level in Ca.levels
    ])
    gs = jnp.array([level.g for level in Ca.levels])
    stages = jnp.array([level.stage + 1 for level in Ca.levels])

    fal = Falc82()

    rad_set = lw.RadiativeSet([H_6_atom(), CaII_atom()])
    eq_pops = rad_set.compute_eq_pops(fal)
    ref = eq_pops.atomicPops["Ca"].nStar

    ntot = lw.DefaultAtomicAbundance['Ca'] * fal.nHTot

    lte_pops_jit = jax.jit(jax.vmap(lte_pops, in_axes=[None, None, None, 0, 0, 0], out_axes=1))
    nstar = lte_pops_jit(energies, gs, stages, fal.temperature, fal.ne, ntot)

    dnstar_datmos = jax.jit(
        jax.vmap(
            jax.jacfwd(lte_pops, argnums=(3,4,5)),
            in_axes=[None, None, None, 0, 0, 0],
            out_axes=1,
        )
    )
    # dnstar_datmos = jax.vmap(
    #         jax.jacfwd(lte_pops, argnums=(3,4,5)),
    #         in_axes=[None, None, None, 0, 0, 0],
    #         out_axes=1,
    #     )
    nstar_response = dnstar_datmos(energies, gs, stages, fal.temperature, fal.ne, ntot)

    temperature_resp = nstar_response[0]
    ne_resp = nstar_response[1]

    ne_pert = fal.ne * 1e-5
    temp_pert = fal.temperature * 1e-4

    fal_ne_plus = Falc82()
    fal_ne_minus = Falc82()
    fal_ne_plus.ne += ne_pert
    fal_ne_minus.ne -= ne_pert
    ne_plus = rad_set.compute_eq_pops(fal_ne_plus).atomicPops["Ca"].nStar
    ne_minus = rad_set.compute_eq_pops(fal_ne_minus).atomicPops["Ca"].nStar
    ne_resp_fd = (ne_plus - ne_minus) / (2.0 * ne_pert)

    fal_temp_plus = Falc82()
    fal_temp_minus = Falc82()
    fal_temp_plus.temperature += temp_pert
    fal_temp_minus.temperature -= temp_pert
    temp_plus = rad_set.compute_eq_pops(fal_temp_plus).atomicPops["Ca"].nStar
    temp_minus = rad_set.compute_eq_pops(fal_temp_minus).atomicPops["Ca"].nStar
    temperature_resp_fd = (temp_plus - temp_minus) / (2.0 * temp_pert)

    plt.figure()

    for i in range(nstar.shape[0]):
        plt.plot(nstar[i])

    for i in range(nstar.shape[0]):
        plt.plot(ref[i], '--', c=f"C{i}")
    plt.yscale('log')

    plt.figure()
    plt.plot(temperature_resp.T)
    for i in range(nstar.shape[0]):
        plt.plot(temperature_resp_fd[i], '--', c=f"C{i}")
    plt.title("T response")
    plt.yscale("symlog")

    plt.figure()
    plt.plot(ne_resp.T)
    for i in range(nstar.shape[0]):
        plt.plot(ne_resp_fd[i], '--', c=f"C{i}")
    plt.title("ne response")
    plt.yscale("symlog", linthresh=1e-7)
