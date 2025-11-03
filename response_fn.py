import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from lineop import AtomicData, read_kurucz, emis_opac
from scalar_formal_solver import nearest_fs

def lte_rt(adata: AtomicData, wave, dz, temperature, ne, nhtot, vz, vturb):
    eta, chi = jax.vmap(
        emis_opac,
        in_axes=[None, None, 0, 0, 0, 0, 0]
    )(adata, wave, temperature, ne, nhtot, vz, vturb)

    I = nearest_fs(dz, eta, chi)
    return I

if __name__ == "__main__":
    import lightweaver as lw
    from lightweaver.fal import Falc82
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        get_ipython().run_line_magic("matplotlib", "")
    except:
        plt.ion()

    lines = read_kurucz("kurucz_6301_6302.linelist")

    fal = Falc82()
    dz = jnp.array(
        np.concatenate(
            [
                [fal.z[::-1][0] - fal.z[::-1][1]],
                fal.z[::-1][1:] - fal.z[::-1][:-1]
            ]
        )
    )
    temperature = jnp.array(fal.temperature[::-1])
    ne = jnp.array(fal.ne[::-1])
    nhtot = jnp.array(fal.nHTot[::-1])
    vturb = jnp.array(fal.vturb[::-1])
    vz = jnp.zeros(temperature.shape[0])

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
