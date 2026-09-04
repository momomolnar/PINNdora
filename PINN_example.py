"""Train the experimental spatial PINN against a generated 2-D spectrum."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import random

from adora_data import FE_I_6301_6302_LINE_LIST
from atmosphere import atmosphere_from_falc
from PINN_machinery import PINNDora_MLP


DEFAULT_DATASET = Path("data") / "spectrum_2D.npz"


def load_dataset(path=DEFAULT_DATASET):
    """Load a named intensity/wavelength archive and validate its dimensions."""
    path = Path(path)
    with np.load(path) as archive:
        missing = {"intensity", "wavelength"}.difference(archive.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"{path} is missing {names}; regenerate it with create_2D_testset.py"
            )
        intensity = np.asarray(archive["intensity"])
        wavelength = np.asarray(archive["wavelength"])

    if intensity.ndim != 3:
        raise ValueError("dataset intensity must have shape (nx, ny, n_wave)")
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("dataset wavelength must be a non-empty 1-D array")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("dataset wavelength must contain only positive finite values")
    if intensity.shape[-1] != wavelength.shape[0]:
        raise ValueError(
            "the intensity wavelength axis must match the saved wavelength grid"
        )
    if not np.all(np.isfinite(intensity)):
        raise ValueError("dataset intensity must contain only finite values")
    return jnp.asarray(intensity), jnp.asarray(wavelength)


def coordinate_grid(nx, ny, height):
    """Return depth-fastest normalized ``(x, y, z)`` coordinates."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    height = jnp.asarray(height)
    if height.ndim != 1 or height.size < 2:
        raise ValueError("height must be a one-dimensional grid with two points")

    height_range = height[-1] - height[0]
    if (
        not bool(jnp.all(jnp.isfinite(height)))
        or not bool(jnp.all(jnp.diff(height) > 0.0))
        or not bool(height_range > 0.0)
    ):
        raise ValueError("height must be finite and strictly increasing")
    z = (height - height[0]) / height_range
    x = jnp.linspace(0.0, 1.0, num=nx)
    y = jnp.linspace(0.0, 1.0, num=ny)
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    return jnp.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1)


def main(dataset_path=DEFAULT_DATASET, num_epochs=1000):
    """Load the generated dataset and train a PINN model."""
    from lightweaver.fal import Falc82

    from lineop import read_kurucz

    intensity, waves = load_dataset(dataset_path)
    nx, ny, n_wave = intensity.shape

    fal = Falc82()
    height, dz, *_ = atmosphere_from_falc(fal)
    coordinates = coordinate_grid(nx, ny, height)
    observed = intensity.reshape((nx * ny, n_wave))

    lines = read_kurucz(FE_I_6301_6302_LINE_LIST)
    model = PINNDora_MLP(
        [3, 64, 64, 64, 5],
        random.PRNGKey(0),
        waves,
        lines,
        dz,
        fal,
        lr=1e-3,
    )
    model.train(coordinates, observed, num_epochs=num_epochs)
    return model


if __name__ == "__main__":
    main()
