from time import perf_counter

import adora_precision
adora_precision.configure_precision()
import jax
import jax.numpy as jnp

from adora_data import FE_I_6301_6302_LINE_LIST
from atmosphere import atmosphere_from_falc
from lineop import AtomicData, read_kurucz, emis_opac_polarised, planck
from vector_formal_solver import delo_constant_fs, delo_constant_fs_nonsingular


def _lte_polarised_rt(
    formal_solver,
    adata: AtomicData,
    wave,
    dz,
    temperature,
    ne,
    nhtot,
    vz,
    vturb,
    b,
    gamma_b,
    chi_b,
):
    eta, chi = jax.vmap(
        emis_opac_polarised,
        in_axes=[None, None, 0, 0, 0, 0, 0, 0, 0, 0]
    )(adata, wave, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)

    I_start = jnp.zeros(4, dtype=eta.dtype).at[0].set(planck(wave, temperature[0]))
    I = formal_solver(dz, I_start, eta, chi)
    return I


def lte_polarised_rt(
    adata: AtomicData,
    wave,
    dz,
    temperature,
    ne,
    nhtot,
    vz,
    vturb,
    b,
    gamma_b,
    chi_b,
):
    """Polarized LTE synthesis with the robust singular-opacity fallback."""

    return _lte_polarised_rt(
        delo_constant_fs,
        adata,
        wave,
        dz,
        temperature,
        ne,
        nhtot,
        vz,
        vturb,
        b,
        gamma_b,
        chi_b,
    )


def lte_polarised_rt_nonsingular(
    adata: AtomicData,
    wave,
    dz,
    temperature,
    ne,
    nhtot,
    vz,
    vturb,
    b,
    gamma_b,
    chi_b,
):
    """Fast polarized LTE synthesis when Stokes-I opacity is nonzero."""

    return _lte_polarised_rt(
        delo_constant_fs_nonsingular,
        adata,
        wave,
        dz,
        temperature,
        ne,
        nhtot,
        vz,
        vturb,
        b,
        gamma_b,
        chi_b,
    )


if __name__ == "__main__":
    import lightweaver as lw
    from lightweaver.fal import Falc82
    import numpy as np
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
    b = jnp.ones(temperature.shape[0]) * 0.05
    # gamma_b = jnp.zeros(temperature.shape[0])
    # ~ 45 degrees
    gamma_b = jnp.ones(temperature.shape[0]) * 0.785
    chi_b = jnp.zeros(temperature.shape[0])

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    lte_rt_wave = jax.jit(
        jax.vmap(
            lte_polarised_rt_nonsingular,
            in_axes=[None, 0, None, None, None, None, None, None, None, None, None],
            out_axes=1,
        )
    )
    intens = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)

    plt.figure()
    plt.plot(waves, intens[0] / intens[0, 0], label="I")
    plt.plot(waves, intens[3] / intens[0, 0], label="V")

    from lightweaver.rh_atoms import H_atom, Fe23_atom
    fal_lw = Falc82()
    fal_lw.B = np.ones(temperature.shape[0]) * 0.05
    fal_lw.gammaB = np.ones(temperature.shape[0]) * 0.785
    fal_lw.chiB = np.zeros(temperature.shape[0])
    fal_lw.quadrature(3)

    rad_set = lw.RadiativeSet([H_atom(), Fe23_atom()])
    rad_set.set_detailed_static("Fe")
    spect = rad_set.compute_wavelength_grid()
    eq_pops = rad_set.compute_eq_pops(fal_lw)

    ctx = lw.Context(fal_lw, spect, eq_pops)
    Iquv_lw = ctx.compute_rays(wavelengths=np.array(waves), mus=[1.0], stokes=True)
    plt.plot(waves, Iquv_lw[0] / Iquv_lw[0, 0], '--', label="I Lw")
    plt.plot(waves, Iquv_lw[3] / Iquv_lw[0, 0], '--', label="V Lw")
    plt.legend()

    response_start = perf_counter()
    lte_polarised_rt_response = jax.jit(
        jax.vmap(
            jax.jacrev(
                lte_polarised_rt_nonsingular,
                argnums=(3, 4, 5, 6, 7, 8, 9, 10),
            ),
            in_axes=[None, 0, None, None, None, None, None, None, None, None, None],
            out_axes=1,
        )
    )
    resp = lte_polarised_rt_response(
        lines, waves, dz, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b
    )
    jax.block_until_ready(resp)
    response_elapsed = perf_counter() - response_start
    print(
        "Response-function first call (JIT compilation + evaluation): "
        f"{response_elapsed:.3f} s"
    )
    cached_start = perf_counter()
    resp = lte_polarised_rt_response(
        lines, waves, dz, temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b
    )
    jax.block_until_ready(resp)
    cached_elapsed = perf_counter() - cached_start
    print(f"Response-function cached evaluation: {cached_elapsed:.3f} s")
    # dIdT = resp[0]
    # dIdne = resp[1]
    # dIdnhtot = resp[2]
    # dIdv = resp[3]
    # dIdvt = resp[4]

    # def maxabs(a):
    #     return max(np.abs(a).max(), a.max())

    # fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(8, 8))

    # mappable = ax[0, 0].imshow(dIdT.T, aspect='auto')
    # ax[0, 0].set_title('dI / dT')
    # fig.colorbar(mappable, ax=ax[0, 0])

    # m = maxabs(dIdne)
    # mappable = ax[0, 1].imshow(dIdne.T, aspect='auto', vmin=-m, vmax=m, cmap='RdBu_r')
    # ax[0, 1].set_title('dI / dne')
    # fig.colorbar(mappable, ax=ax[0, 1])

    # m = maxabs(dIdnhtot)
    # mappable = ax[1, 0].imshow(dIdnhtot.T, aspect='auto', vmin=-m, vmax=m, cmap='PuOr_r')
    # ax[1, 0].set_title('dI / dnhtot')
    # fig.colorbar(mappable, ax=ax[1, 0])

    # m = maxabs(dIdvt)
    # mappable = ax[1, 1].imshow(dIdvt.T, aspect='auto', vmin=-m, vmax=m, cmap="RdYlBu_r")
    # ax[1, 1].set_title('dI / dvturb')
    # fig.colorbar(mappable, ax=ax[1, 1])
