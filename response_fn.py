import adora_precision
adora_precision.configure_precision()
import jax
import jax.numpy as jnp
from adora_data import FE_I_6301_6302_LINE_LIST
from atmosphere import atmosphere_from_falc
from lineop import AtomicData, read_kurucz, emis_opac, planck
from scalar_formal_solver import nearest_fs

def lte_rt(adata: AtomicData, wave, dz, temperature, ne, nhtot, vz, vturb):
    eta, chi = jax.vmap(
        emis_opac,
        in_axes=[None, None, 0, 0, 0, 0, 0]
    )(adata, wave, temperature, ne, nhtot, vz, vturb)

    I_start = planck(wave, temperature[0])
    I = nearest_fs(dz, eta, chi, I_start=I_start)
    return I

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

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    lte_rt_wave = jax.jit(
        jax.vmap(
            lte_rt,
            in_axes=[None, 0, None, None, None, None, None, None]
        )
    )
    intens = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb)

    plt.figure()
    plt.plot(waves, intens)

    lte_rt_response = jax.jit(
        jax.vmap(
            jax.jacrev(
                lte_rt,
                argnums=(3, 4, 5, 6, 7),
            ),
            in_axes=[None, 0, None, None, None, None, None, None]
        )
    )
    resp = lte_rt_response(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
    dIdT = resp[0]
    dIdne = resp[1]
    dIdnhtot = resp[2]
    dIdv = resp[3]
    dIdvt = resp[4]

    def maxabs(a):
        return max(np.abs(a).max(), a.max())

    fig, ax = plt.subplots(2, 2, layout='constrained', figsize=(8, 8))

    mappable = ax[0, 0].imshow(dIdT.T, aspect='auto')
    ax[0, 0].set_title('dI / dT')
    fig.colorbar(mappable, ax=ax[0, 0])

    m = maxabs(dIdne)
    mappable = ax[0, 1].imshow(dIdne.T, aspect='auto', vmin=-m, vmax=m, cmap='RdBu_r')
    ax[0, 1].set_title('dI / dne')
    fig.colorbar(mappable, ax=ax[0, 1])

    m = maxabs(dIdnhtot)
    mappable = ax[1, 0].imshow(dIdnhtot.T, aspect='auto', vmin=-m, vmax=m, cmap='PuOr_r')
    ax[1, 0].set_title('dI / dnhtot')
    fig.colorbar(mappable, ax=ax[1, 0])

    m = maxabs(dIdvt)
    mappable = ax[1, 1].imshow(dIdvt.T, aspect='auto', vmin=-m, vmax=m, cmap="RdYlBu_r")
    ax[1, 1].set_title('dI / dvturb')
    fig.colorbar(mappable, ax=ax[1, 1])

    fig.savefig('blah.png')
    plt.close()
