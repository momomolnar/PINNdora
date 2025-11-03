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
    print(dz.shape, eta.shape, chi.shape)

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
    temperature   = jnp.array(fal.temperature[::-1])
    ne            = jnp.array(fal.ne[::-1])
    nhtot         = jnp.array(fal.nHTot[::-1])
    vturb         = jnp.array(fal.vturb[::-1])
    vz            = jnp.zeros(temperature.shape[0])

    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

    # Define the x and y dimensions of the 3D array
    x_dim = 10  # For example, 100 rows in the 2nd dimension
    y_dim = 5  # For example, 50 columns in the 3rd dimension

    # Step 1: Add new axes to the 1D array (along the x and y dimensions)
    temperature_broadcasted = temperature[:, None, None]  # Shape: (n, 1, 1)
    ne_broadcasted          = ne[:, None, None]
    nhtot_broadcasted       = nhtot[:, None, None]
    vturb_broadcasted       = vturb[:, None, None]
    vz_broadcasted          = vz[:, None, None]
    # Step 2: Broadcast along the x and y dimensions
    temperature_3D = jnp.broadcast_to(temperature_broadcasted,
                                      (temperature.shape[0], x_dim, y_dim))  # Shape: (n, x_dim, y_dim)
    ne_3D = jnp.broadcast_to(ne_broadcasted,
                             (ne.shape[0], x_dim, y_dim))  # Shape: (n, x_dim, y_dim)
    nhtot_3D = jnp.broadcast_to(nhtot_broadcasted,
                                (nhtot.shape[0], x_dim, y_dim))  # Shape: (n, x_dim, y_dim)
    vturb_3D = jnp.broadcast_to(vturb_broadcasted,
                                (vturb.shape[0], x_dim, y_dim))  # Shape: (n, x_dim, y_dim)
    vz_3D = jnp.broadcast_to(vz_broadcasted,
                             (vz.shape[0], x_dim, y_dim))  # Shape: (n, x_dim, y_dim)


    lte_rt_wave = jax.jit(
        jax.vmap(
            lte_rt,
            in_axes=[None, 0, None, None, None, None, None, None]
        )
    )

    lte_rt_wave_3D = jax.jit(
        jax.vmap(
        jax.vmap(  # Outer vmap: Loops over spatial grid (x_dim, y_dim)
            jax.vmap(  # Inner vmap: Loops over wavelengths for a single (x, y) column
                lte_rt,
                in_axes=[None, 0, None, None, None, None, None, None]  # Vectorize over `waves` and `vertical_profiles`
            ),
            in_axes=[None, None, None, 1, 1, 1, 1, 1]  # Vectorize over spatial dimensions (x, y)
        ), in_axes=[None, None, None, 2, 2, 2, 2, 2])
    )

    intens = lte_rt_wave(lines, waves, dz, temperature, ne, nhtot, vz, vturb)
    intens2D = lte_rt_wave_3D(lines, waves, dz, temperature_3D, ne_3D, nhtot_3D, vz_3D, vturb_3D)

    plt.figure()
    plt.plot(waves, intens)
    plt.savefig("example_spectrum.png")
    plt.clf()
    import numpy as np

    np.savez('data/spectrum_2D.npz', intens2D)