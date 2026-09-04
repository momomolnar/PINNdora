"""Generate a small spatially broadcast LTE spectrum dataset."""

from pathlib import Path

import adora_precision
adora_precision.configure_precision()
import jax
import jax.numpy as jnp
import numpy as np

from adora_data import FE_I_6301_6302_LINE_LIST
from atmosphere import atmosphere_from_falc
from lineop import AtomicData, emis_opac, planck, read_kurucz
from scalar_formal_solver import nearest_fs

DEFAULT_DATASET = Path("data") / "spectrum_2D.npz"


def lte_rt(adata: AtomicData, wave, dz, temperature, ne, nhtot, vz, vturb):
    """Synthesize one wavelength through one bottom-to-top column."""
    eta, chi = jax.vmap(
        emis_opac,
        in_axes=(None, None, 0, 0, 0, 0, 0),
    )(adata, wave, temperature, ne, nhtot, vz, vturb)
    lower_boundary = planck(wave, temperature[0])
    return nearest_fs(dz, eta, chi, I_start=lower_boundary)


_LTE_RT_WAVELENGTHS = jax.vmap(
    lte_rt,
    in_axes=(None, 0, None, None, None, None, None, None),
)
_LTE_RT_GRID = jax.jit(
    jax.vmap(
        jax.vmap(
            _LTE_RT_WAVELENGTHS,
            in_axes=(None, None, None, 0, 0, 0, 0, 0),
        ),
        in_axes=(None, None, None, 0, 0, 0, 0, 0),
    )
)


def broadcast_atmosphere(profiles, x_dim, y_dim):
    """Broadcast depth profiles to canonical ``(x, y, depth)`` arrays."""
    if not isinstance(x_dim, int) or not isinstance(y_dim, int):
        raise TypeError("x_dim and y_dim must be integers")
    if x_dim <= 0 or y_dim <= 0:
        raise ValueError("x_dim and y_dim must be positive")

    profiles = tuple(jnp.asarray(profile) for profile in profiles)
    if not profiles or profiles[0].ndim != 1 or profiles[0].size == 0:
        raise ValueError("profiles must contain non-empty one-dimensional arrays")
    depth_shape = profiles[0].shape
    if any(profile.ndim != 1 or profile.shape != depth_shape for profile in profiles):
        raise ValueError("all profiles must have the same one-dimensional shape")
    if any(not bool(jnp.all(jnp.isfinite(profile))) for profile in profiles):
        raise ValueError("all profiles must contain only finite values")
    return tuple(
        jnp.broadcast_to(profile, (x_dim, y_dim, profile.shape[0]))
        for profile in profiles
    )


def synthesize_grid(
    adata,
    waves,
    dz,
    temperature,
    ne,
    nhtot,
    vz,
    vturb,
    x_dim,
    y_dim,
):
    """Synthesize a broadcast atmosphere with shape ``(x, y, wavelength)``."""
    waves = jnp.asarray(waves)
    dz = jnp.asarray(dz)
    if waves.ndim != 1 or waves.size == 0:
        raise ValueError("waves must be a non-empty one-dimensional array")
    if not bool(jnp.all(jnp.isfinite(waves))) or not bool(jnp.all(waves > 0.0)):
        raise ValueError("waves must contain only positive finite values")
    if dz.ndim != 1 or dz.size == 0:
        raise ValueError("dz must be a non-empty one-dimensional array")
    if not bool(jnp.all(jnp.isfinite(dz))) or not bool(jnp.all(dz > 0.0)):
        raise ValueError("dz must contain positive finite cell widths")

    spatial_profiles = broadcast_atmosphere(
        (temperature, ne, nhtot, vz, vturb), x_dim, y_dim
    )
    if spatial_profiles[0].shape[-1] != dz.shape[0]:
        raise ValueError("dz and atmospheric profiles must have the same depth")

    return _LTE_RT_GRID(adata, waves, dz, *spatial_profiles)


def save_dataset(path, intensity, wavelength):
    """Save named arrays, creating the output directory when necessary."""
    path = Path(path)
    intensity = np.asarray(intensity)
    wavelength = np.asarray(wavelength)
    if intensity.ndim != 3:
        raise ValueError("intensity must have shape (x, y, wavelength)")
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("wavelength must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength must contain only positive finite values")
    if intensity.shape[-1] != wavelength.shape[0]:
        raise ValueError("intensity and wavelength dimensions do not match")
    if not np.all(np.isfinite(intensity)):
        raise ValueError("intensity must contain only finite values")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, intensity=intensity, wavelength=wavelength)


def generate_dataset(output_path=DEFAULT_DATASET, x_dim=10, y_dim=5):
    """Generate and save the default broadcast FAL-C dataset."""
    import lightweaver as lw
    from lightweaver.fal import Falc82

    fal = Falc82()
    _, dz, temperature, ne, nhtot, vz, vturb = atmosphere_from_falc(fal)
    waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)
    lines = read_kurucz(FE_I_6301_6302_LINE_LIST)

    intensity = synthesize_grid(
        lines,
        waves,
        dz,
        temperature,
        ne,
        nhtot,
        vz,
        vturb,
        x_dim,
        y_dim,
    )
    save_dataset(output_path, intensity, waves)
    return intensity, waves


def main():
    """Generate the dataset and save an example spectrum plot."""
    import matplotlib.pyplot as plt

    intensity, waves = generate_dataset()
    plt.figure()
    plt.plot(waves, intensity[0, 0])
    plt.savefig("example_spectrum.png")
    plt.close()


if __name__ == "__main__":
    main()
