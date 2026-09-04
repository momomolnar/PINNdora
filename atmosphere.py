"""Helpers for constructing consistently ordered one-dimensional atmospheres."""

from adora_precision import REAL_DTYPE
import jax.numpy as jnp
import numpy as np


def cell_widths(height):
    """Return positive cell widths for an increasing grid of cell centres.

    The first cell uses the same one-sided width as the interval to the second
    centre.  Subsequent cells use the distance from the preceding centre.  The
    returned array therefore has the same length as ``height``.

    This is a host-side construction/validation helper, not a differentiable
    JAX kernel; call it before entering ``jax.jit`` or AD transformations.
    """
    height_array = np.asarray(height, dtype=float)
    if height_array.ndim != 1:
        raise ValueError("height must be a one-dimensional array")
    if height_array.size < 2:
        raise ValueError("height must contain at least two points")

    spacing = np.diff(height_array)
    if not np.all(np.isfinite(spacing)):
        raise ValueError("height must contain only finite values")
    if np.any(spacing <= 0.0):
        raise ValueError("height must be strictly increasing")

    return jnp.asarray(
        np.concatenate((spacing[:1], spacing)), dtype=REAL_DTYPE
    )


def atmosphere_from_falc(fal):
    """Convert a Lightweaver FAL atmosphere to bottom-to-top JAX arrays.

    Returns ``(height, dz, temperature, ne, nhtot, vz, vturb)``.  Lightweaver
    stores FAL models from the top of the atmosphere toward the lower
    boundary, while Adora's formal solvers integrate in the opposite order.
    """
    height_host = np.asarray(fal.z, dtype=float)[::-1].copy()
    # Derive widths from the same selected-precision centres that are returned
    # and archived.  In fp32, subtracting the original fp64 centres and only
    # then rounding the widths can otherwise disagree with differences of the
    # rounded height array.
    height = jnp.asarray(height_host, dtype=REAL_DTYPE)
    dz = cell_widths(np.asarray(height))

    def bottom_to_top(name, values):
        values = np.asarray(values)
        if values.ndim != 1 or values.shape != height.shape:
            raise ValueError(
                f"fal.{name} must be one-dimensional and match the height grid"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"fal.{name} must contain only finite values")
        return jnp.asarray(values[::-1].copy(), dtype=REAL_DTYPE)

    return (
        height,
        dz,
        bottom_to_top("temperature", fal.temperature),
        bottom_to_top("ne", fal.ne),
        bottom_to_top("nHTot", fal.nHTot),
        bottom_to_top("vz", fal.vz),
        bottom_to_top("vturb", fal.vturb),
    )
