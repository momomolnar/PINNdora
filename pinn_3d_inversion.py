"""Three-dimensional neural-field inversion of polarized Fe I spectra.

The neural atmosphere is a shared implicit representation of ``(x, y, z)``.
Radiative transfer is evaluated independently along each vertical column with
Adora's LTE polarized formal solver, so the forward model is commonly called
1.5-D rather than horizontally coupled 3-D radiative transfer.

Running this module without mode flags performs the complete workflow:

1. Broadcast FAL-C over a 50 x 50 horizontal grid, apply a small smooth 3-D
   perturbation, and synthesize full-Stokes observations for every Fe I line
   in the selected Kurucz file.
2. Pretrain a vertical neural field to reproduce FAL-C thermodynamics plus a
   weak, non-axis-aligned magnetic seed field at every horizontal location.
3. Optimize a shared spatial correction field from the spectral residuals.

The default problem is intentionally substantial.  Use small ``--nx``,
``--ny``, and ``--n-wave`` values for a quick smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# Precision is process-global in JAX and must be selected before importing any
# physics module that creates constants or lookup tables.  Direct CLI launches
# are pre-scanned here; library callers can select the same mode with the
# ADORA_PRECISION environment variable before importing this module.
import adora_precision

adora_precision.configure_precision(adora_precision.precision_from_argv())

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from adora_data import FE_I_6301_6302_LINE_LIST
from adora_precision import NUMPY_REAL_DTYPE, REAL_DTYPE
from atmosphere import atmosphere_from_falc
from lineop import (
    AtomicData,
    emis_opac_polarised,
    emis_opac_polarised_offset,
    planck,
    read_kurucz,
    wavelength_offsets,
)
from vector_formal_solver import delo_constant_fs_nonsingular


SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
STOKES_LABELS = ("I", "Q", "U", "V")
FIELD_NAMES = (
    "temperature",
    "ne",
    "nhtot",
    "vz",
    "vturb",
    "b",
    "gamma_b",
    "chi_b",
)
LATENT_CHANNEL_NAMES = (
    "bounded_log_temperature",
    "bounded_log_ne",
    "bounded_log_nhtot",
    "bounded_los_velocity",
    "bounded_log_vturb",
    "bounded_log_magnetic_field",
    "bounded_inclination",
    "bounded_azimuth",
)

DEFAULT_DATASET = Path("data") / "pinn_3d_test_cube.npz"
DEFAULT_RESULT = Path("data") / "pinn_3d_inversion.npz"
DEFAULT_CHECKPOINT = Path("data") / "pinn_3d_checkpoint.npz"
DEFAULT_WAVELENGTH_PARALLELISM = 48
DEFAULT_VALIDATION_COLUMNS = 8

# Canonical host values are kept separately from their selected-precision JAX
# forms.  Checkpoint compatibility is therefore independent of whether a run
# happens in fp32 or fp64.
_POSITIVE_LOWER_VALUES = (2500.0, 1.0e14, 1.0e14, 50.0)
_POSITIVE_UPPER_VALUES = (120000.0, 1.0e23, 1.0e25, 30000.0)
_SPATIAL_SCALE_VALUES = (0.35, 0.35, 0.35, 0.08, 0.35, 0.20, 0.20, 0.20)

# The network represents positive quantities through a bounded logarithmic
# transform.  The bounds cover FAL-C with room for the deliberately small
# perturbations used here while preventing invalid atmospheric states.
_POSITIVE_LOWER = jnp.asarray(_POSITIVE_LOWER_VALUES, dtype=REAL_DTYPE)
_POSITIVE_UPPER = jnp.asarray(_POSITIVE_UPPER_VALUES, dtype=REAL_DTYPE)
_VELOCITY_LIMIT = 2.0e4  # m / s
_MAGNETIC_FIELD_LOWER = 1.0e-5  # tesla
_MAGNETIC_FIELD_UPPER = 0.50  # tesla
_INCLINATION_MARGIN = 1.0e-5  # radians away from singular vertical axes
_AZIMUTH_LIMIT = 0.5 * jnp.pi  # principal interval for the pi-periodic azimuth

# Maximum latent displacement introduced by the spatial network.  The first
# five entries control T, ne, nHTot, vz, and vturb; the last three control the
# magnetic strength, inclination, and azimuth. tanh makes every correction
# smooth/bounded.
DEFAULT_SPATIAL_SCALE = jnp.asarray(_SPATIAL_SCALE_VALUES, dtype=REAL_DTYPE)


class Atmosphere(NamedTuple):
    """Eight physical profiles used by polarized LTE synthesis."""

    temperature: jax.Array
    ne: jax.Array
    nhtot: jax.Array
    vz: jax.Array
    vturb: jax.Array
    b: jax.Array
    gamma_b: jax.Array
    chi_b: jax.Array


@dataclass(frozen=True)
class PerturbationConfig:
    """Amplitudes of the deterministic Gaussian test-cube perturbation."""

    center_x: float = 0.55
    center_y: float = 0.45
    center_z: float = 0.28
    sigma_x: float = 0.16
    sigma_y: float = 0.16
    sigma_z: float = 0.12
    log_temperature: float = 0.04
    log_ne: float = 0.025
    log_nhtot: float = -0.015
    velocity_m_s: float = 700.0
    log_vturb: float = 0.04
    log_b: float = 0.12
    inclination_rad: float = 0.08
    azimuth_rad: float = 0.12

    def validate(self) -> None:
        values = tuple(vars(self).values())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("perturbation parameters must be finite")
        if min(self.sigma_x, self.sigma_y, self.sigma_z) <= 0.0:
            raise ValueError("perturbation widths must be positive")


@dataclass(frozen=True)
class NeuralFieldConfig:
    """Architecture and bounded correction scale for the neural field."""

    base_hidden: tuple[int, ...] = (128, 128, 128)
    spatial_hidden: tuple[int, ...] = (96, 96, 96)
    base_frequencies: int = 6
    spatial_scale: tuple[float, ...] = _SPATIAL_SCALE_VALUES

    def validate(self) -> None:
        if not self.base_hidden or not self.spatial_hidden:
            raise ValueError("base and spatial networks need hidden layers")
        if any(size <= 0 for size in (*self.base_hidden, *self.spatial_hidden)):
            raise ValueError("all hidden-layer sizes must be positive")
        if len(self.spatial_scale) != 8:
            raise ValueError("spatial_scale must contain eight values")
        if not isinstance(self.base_frequencies, int) or self.base_frequencies < 0:
            raise ValueError("base_frequencies must be a non-negative integer")
        if not all(
            math.isfinite(value) and value > 0.0 for value in self.spatial_scale
        ):
            raise ValueError("spatial correction scales must be positive and finite")

    @property
    def base_layers(self) -> tuple[int, ...]:
        n_features = 1 + 2 * self.base_frequencies
        return (n_features, *self.base_hidden, 8)

    @property
    def spatial_layers(self) -> tuple[int, ...]:
        return (3, *self.spatial_hidden, 8)


def _validate_positive_integer(name: str, value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _as_real(value) -> jax.Array:
    """Move a numerical input to the active Adora compute precision."""

    return jnp.asarray(value, dtype=REAL_DTYPE)


def _canonical_line_centers(adata: AtomicData) -> np.ndarray:
    """Return absolute line centres without narrowing them through JAX."""

    canonical = getattr(adata, "canonical_lambda0_nm", None)
    if canonical is not None:
        return np.asarray(canonical, dtype=np.float64)
    return np.asarray(adata.lambda0, dtype=np.float64)


def _host_wavelength_axis(wavelengths) -> np.ndarray:
    """Validate and retain an absolute wavelength axis in host fp64."""

    source = np.asarray(wavelengths)
    if (
        adora_precision.configured_precision() == "fp32"
        and np.issubdtype(source.dtype, np.floating)
        and source.dtype.itemsize < np.dtype(np.float64).itemsize
    ):
        raise ValueError(
            "fp32 synthesis requires absolute wavelengths retained as host "
            "float64 until centering; pass the archive/NumPy float64 axis, "
            "not an already-cast JAX float32 array"
        )
    host = np.asarray(wavelengths, dtype=np.float64)
    if host.ndim != 1 or host.size == 0:
        raise ValueError("wavelengths must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(host)) or np.any(host <= 0.0):
        raise ValueError("wavelengths must be positive and finite")
    return host


def centered_wavelengths(adata: AtomicData, wavelengths) -> jax.Array:
    """Return host-precentered wavelengths in the selected compute dtype.

    Call this at the Python boundary, before an absolute wavelength array is
    transferred to an fp32 device.  Near 630 nm the centered coordinates keep
    orders of magnitude more line-profile resolution than absolute fp32 nm.
    """

    host = _host_wavelength_axis(wavelengths)
    return wavelength_offsets(host, adata.wavelength_reference_nm)


def _unit_axis(size: int) -> jax.Array:
    _validate_positive_integer("axis size", size)
    if size == 1:
        return jnp.asarray((0.5,), dtype=REAL_DTYPE)
    return jnp.linspace(0.0, 1.0, size, dtype=REAL_DTYPE)


def normalized_coordinate_grid(nx: int, ny: int, height) -> tuple[jax.Array, ...]:
    """Return axes and a depth-fast ``(nx, ny, nz, 3)`` coordinate cube.

    Saved horizontal axes and normalized height are in ``[0, 1]``.  Network
    inputs are mapped to ``[-1, 1]`` for better conditioning.
    """

    _validate_positive_integer("nx", nx)
    _validate_positive_integer("ny", ny)
    height = _as_real(height)
    if height.ndim != 1 or height.size < 2:
        raise ValueError("height must be a one-dimensional grid with two points")
    if not bool(jnp.all(jnp.isfinite(height))) or not bool(
        jnp.all(jnp.diff(height) > 0.0)
    ):
        raise ValueError("height must be finite and strictly increasing")

    x = _unit_axis(nx)
    y = _unit_axis(ny)
    z_normalized = (height - height[0]) / (height[-1] - height[0])
    xx, yy, zz = jnp.meshgrid(x, y, z_normalized, indexing="ij")
    coordinates = 2.0 * jnp.stack((xx, yy, zz), axis=-1) - 1.0
    return x, y, z_normalized, coordinates


def _broadcast_profile(profile, nx: int, ny: int) -> jax.Array:
    profile = _as_real(profile)
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("reference profiles must be non-empty one-dimensional arrays")
    return jnp.broadcast_to(profile, (nx, ny, profile.shape[0]))


def create_falc_reference_cube(
    nx: int = 50,
    ny: int = 50,
    magnetic_field_t: float = 0.05,
    inclination_rad: float = 0.8,
    azimuth_rad: float = 0.3,
):
    """Broadcast FAL-C thermodynamics plus a weak magnetic seed over x/y."""

    _validate_positive_integer("nx", nx)
    _validate_positive_integer("ny", ny)
    if (
        not math.isfinite(magnetic_field_t)
        or not _MAGNETIC_FIELD_LOWER < magnetic_field_t < _MAGNETIC_FIELD_UPPER
    ):
        raise ValueError(
            "magnetic_field_t must lie inside the neural transform bounds "
            f"({_MAGNETIC_FIELD_LOWER}, {_MAGNETIC_FIELD_UPPER}) T"
        )
    if (
        not math.isfinite(inclination_rad)
        or not _INCLINATION_MARGIN < inclination_rad < math.pi - _INCLINATION_MARGIN
    ):
        raise ValueError("inclination_rad must lie inside the neural transform margins")
    if not math.isfinite(azimuth_rad):
        raise ValueError("azimuth_rad must be finite")

    from lightweaver.fal import Falc82

    height, dz, temperature, ne, nhtot, vz, vturb = atmosphere_from_falc(Falc82())
    n_depth = int(height.shape[0])
    b = jnp.full(n_depth, magnetic_field_t, dtype=REAL_DTYPE)
    gamma_b = jnp.full(n_depth, inclination_rad, dtype=REAL_DTYPE)
    chi_b = _wrap_azimuth(jnp.full(n_depth, azimuth_rad, dtype=REAL_DTYPE))
    reference = Atmosphere(temperature, ne, nhtot, vz, vturb, b, gamma_b, chi_b)
    cube = Atmosphere(*(_broadcast_profile(profile, nx, ny) for profile in reference))
    x, y, z_normalized, coordinates = normalized_coordinate_grid(nx, ny, height)
    return x, y, height, z_normalized, dz, coordinates, reference, cube


def _wrap_azimuth(angle):
    """Wrap the magnetic azimuth to its pi-periodic principal interval."""

    return jnp.mod(angle + 0.5 * jnp.pi, jnp.pi) - 0.5 * jnp.pi


def perturb_falc_cube(
    reference_cube: Atmosphere,
    x,
    y,
    z_normalized,
    config: PerturbationConfig = PerturbationConfig(),
) -> tuple[Atmosphere, jax.Array]:
    """Apply a deterministic, smooth, bounded perturbation to a FAL-C cube."""

    config.validate()
    x = _as_real(x)
    y = _as_real(y)
    z_normalized = _as_real(z_normalized)
    expected_shape = (x.size, y.size, z_normalized.size)
    if any(jnp.asarray(field).shape != expected_shape for field in reference_cube):
        raise ValueError(f"every reference field must have shape {expected_shape}")

    xx, yy, zz = jnp.meshgrid(x, y, z_normalized, indexing="ij")
    envelope = jnp.exp(
        -0.5
        * (
            ((xx - config.center_x) / config.sigma_x) ** 2
            + ((yy - config.center_y) / config.sigma_y) ** 2
            + ((zz - config.center_z) / config.sigma_z) ** 2
        )
    )
    perturbed = Atmosphere(
        reference_cube.temperature * jnp.exp(config.log_temperature * envelope),
        reference_cube.ne * jnp.exp(config.log_ne * envelope),
        reference_cube.nhtot * jnp.exp(config.log_nhtot * envelope),
        reference_cube.vz + config.velocity_m_s * envelope,
        reference_cube.vturb * jnp.exp(config.log_vturb * envelope),
        reference_cube.b * jnp.exp(config.log_b * envelope),
        jnp.clip(
            reference_cube.gamma_b + config.inclination_rad * envelope,
            1.0e-6,
            jnp.pi - 1.0e-6,
        ),
        _wrap_azimuth(reference_cube.chi_b + config.azimuth_rad * envelope),
    )
    return perturbed, envelope


def build_wavelength_grid(
    lines: AtomicData,
    n_wave: int = 201,
    padding_nm: float = 0.05,
) -> np.ndarray:
    """Build a vacuum-nm grid covering every line in ``lines``."""

    _validate_positive_integer("n_wave", n_wave)
    if not math.isfinite(padding_nm) or padding_nm <= 0.0:
        raise ValueError("padding_nm must be positive and finite")
    centers = _canonical_line_centers(lines)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("the Kurucz data must contain at least one line")
    if not np.all(np.isfinite(centers)) or np.any(centers <= 0.0):
        raise ValueError("line wavelengths must be positive and finite")
    start = float(np.min(centers) - padding_nm)
    stop = float(np.max(centers) + padding_nm)
    if start <= 0.0:
        raise ValueError("padding_nm produces non-positive wavelengths")
    required = np.unique(np.concatenate(([start], centers, [stop])))
    if n_wave < required.size:
        raise ValueError(
            "n_wave must accommodate every unique line center and both wings; "
            f"need at least {required.size}"
        )

    # Bisect the largest remaining interval so the final grid stays close to
    # uniform while retaining every requested line center exactly.
    points = required.tolist()
    while len(points) < n_wave:
        points.sort()
        spacing = np.diff(points)
        interval = int(np.argmax(spacing))
        points.append(0.5 * (points[interval] + points[interval + 1]))
    # Absolute wavelengths remain host fp64.  They are centered in host
    # precision immediately before transfer to a JAX compute kernel.
    return np.sort(np.asarray(points, dtype=np.float64))


def atomic_data_fingerprint(adata: AtomicData) -> str:
    """Hash the canonical parsed atomic data, excluding derived accelerators."""

    digest = hashlib.sha256()
    # Keep this list stable across internal representations.  The packed
    # Zeeman component arrays are derived exactly from the three canonical
    # rectangular arrays below; hashing them would invalidate existing cube
    # files whenever that acceleration metadata changes.
    canonical_fields = (
        "mass",
        "elem",
        "stage",
        "abund",
        "lambda0",
        "log_grad",
        "log_gs",
        "log_gw",
        "gi",
        "gj",
        "ei",
        "ej",
        "Aji",
        "line_weight",
        "zeeman_alphas",
        "zeeman_strengths",
        "zeeman_shifts",
    )
    for name in canonical_fields:
        array = np.ascontiguousarray(np.asarray(getattr(adata, name)))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    runtime_digest = digest.hexdigest()
    if (
        getattr(adata, "canonical_sha256", None) is not None
        and runtime_digest == getattr(adata, "runtime_sha256", None)
    ):
        return adata.canonical_sha256
    return runtime_digest


def polarized_lte_rt(
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
    """Synthesize one vacuum wavelength through one bottom-to-top column."""

    eta, chi = jax.vmap(
        emis_opac_polarised,
        in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        adata,
        wave,
        temperature,
        ne,
        nhtot,
        vz,
        vturb,
        b,
        gamma_b,
        chi_b,
    )
    lower_boundary = (
        jnp.zeros(4, dtype=eta.dtype).at[0].set(planck(wave, temperature[0]))
    )
    # The bounded PINN decoder guarantees a positive thermodynamic atmosphere;
    # its Stokes-I opacity is nonzero throughout the supported domain.  Using
    # the specialized solver keeps the singular matrix-exponential fallback
    # out of the vmapped accelerator graph.
    return delo_constant_fs_nonsingular(dz, lower_boundary, eta, chi)


def polarized_lte_rt_offset(
    adata: AtomicData,
    wave_offset,
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
    """Synthesize at a wavelength precentered around the atomic reference."""

    eta, chi = jax.vmap(
        emis_opac_polarised_offset,
        in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        adata,
        wave_offset,
        temperature,
        ne,
        nhtot,
        vz,
        vturb,
        b,
        gamma_b,
        chi_b,
    )
    absolute_wave = (
        jnp.asarray(adata.wavelength_reference_nm, dtype=eta.dtype) + wave_offset
    )
    lower_boundary = jnp.zeros(4, dtype=eta.dtype).at[0].set(
        planck(absolute_wave, temperature[0])
    )
    return delo_constant_fs_nonsingular(dz, lower_boundary, eta, chi)


def _synthesize_one_column(
    adata: AtomicData,
    wavelengths,
    dz,
    column: Atmosphere,
    wavelength_parallelism: int = 1,
):
    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    # ``lax.map(..., batch_size=...)`` maps chunks sequentially while vmapping
    # the independent wavelengths within each chunk.  This supplies enough
    # parallel work to an accelerator without materializing the reverse-mode
    # intermediates for the complete wavelength axis at once.  Capping at the
    # static wavelength extent also vectorizes short/remainder-only batches.
    chunk_size = min(wavelength_parallelism, int(wavelengths.shape[0]))

    def solve_wavelength(wave):
        return polarized_lte_rt(adata, wave, dz, *column)

    if chunk_size == 1:
        spectra_wave_first = jax.lax.map(solve_wavelength, wavelengths)
    else:
        spectra_wave_first = jax.lax.map(
            solve_wavelength,
            wavelengths,
            batch_size=chunk_size,
        )
    return jnp.swapaxes(spectra_wave_first, 0, 1)


_REMATERIALIZED_COLUMN_SYNTHESIS = jax.checkpoint(
    _synthesize_one_column,
    static_argnums=(4,),
)


def _synthesize_columns_core(
    adata: AtomicData,
    wavelengths,
    dz,
    atmosphere: Atmosphere,
    wavelength_parallelism: int = 1,
):
    """JAX core returning ``(n_columns, 4, n_wave)``."""

    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )

    def solve_column(column):
        return _REMATERIALIZED_COLUMN_SYNTHESIS(
            adata,
            wavelengths,
            dz,
            Atmosphere(*column),
            wavelength_parallelism,
        )

    # Columns are independent 1.5-D transfer problems. Vectorizing this
    # outer dimension lets an accelerator solve the complete column batch in
    # parallel. ``_synthesize_one_column`` separately bounds wavelength
    # parallelism to keep the reverse-mode memory footprint predictable. On
    # CPU the vectorized column form is substantially slower for this
    # small-matrix workload, so retain the sequential map there. The caller
    # controls both accelerator dimensions explicitly.
    if jax.default_backend() == "cpu":
        return jax.lax.map(solve_column, atmosphere)
    return jax.vmap(solve_column)(atmosphere)


_SYNTHESIZE_COLUMNS_JIT = jax.jit(
    _synthesize_columns_core,
    static_argnames=("wavelength_parallelism",),
)


def _synthesize_one_column_offset(
    adata: AtomicData,
    wavelength_offsets_nm,
    dz,
    column: Atmosphere,
    wavelength_parallelism: int = 1,
):
    """Internal column synthesis using precision-safe centered wavelengths."""

    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    chunk_size = min(wavelength_parallelism, int(wavelength_offsets_nm.shape[0]))

    def solve_wavelength(wave_offset):
        return polarized_lte_rt_offset(adata, wave_offset, dz, *column)

    if chunk_size == 1:
        spectra_wave_first = jax.lax.map(solve_wavelength, wavelength_offsets_nm)
    else:
        spectra_wave_first = jax.lax.map(
            solve_wavelength,
            wavelength_offsets_nm,
            batch_size=chunk_size,
        )
    return jnp.swapaxes(spectra_wave_first, 0, 1)


_REMATERIALIZED_COLUMN_SYNTHESIS_OFFSET = jax.checkpoint(
    _synthesize_one_column_offset,
    static_argnums=(4,),
)


def _synthesize_columns_offset_core(
    adata: AtomicData,
    wavelength_offsets_nm,
    dz,
    atmosphere: Atmosphere,
    wavelength_parallelism: int = 1,
):
    """JAX core for ``(column, Stokes, wave)`` from centered nm offsets."""

    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )

    def solve_column(column):
        return _REMATERIALIZED_COLUMN_SYNTHESIS_OFFSET(
            adata,
            wavelength_offsets_nm,
            dz,
            Atmosphere(*column),
            wavelength_parallelism,
        )

    if jax.default_backend() == "cpu":
        return jax.lax.map(solve_column, atmosphere)
    return jax.vmap(solve_column)(atmosphere)


_SYNTHESIZE_COLUMNS_OFFSET_JIT = jax.jit(
    _synthesize_columns_offset_core,
    static_argnames=("wavelength_parallelism",),
)


def _validate_atmosphere(
    atmosphere: Atmosphere, expected_shape=None
) -> tuple[int, ...]:
    arrays = tuple(jnp.asarray(field) for field in atmosphere)
    shape = arrays[0].shape
    if not shape or any(array.shape != shape for array in arrays[1:]):
        raise ValueError("all eight atmospheric fields must have identical shapes")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"atmospheric fields must have shape {expected_shape}")
    for name, array in zip(FIELD_NAMES, arrays):
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError(f"{name} must contain only finite values")
    for name, array in zip(
        ("temperature", "ne", "nhtot", "vturb", "b"),
        (arrays[0], arrays[1], arrays[2], arrays[4], arrays[5]),
    ):
        if not bool(jnp.all(array > 0.0)):
            raise ValueError(f"{name} must be strictly positive")
    if not bool(jnp.all((arrays[6] >= 0.0) & (arrays[6] <= jnp.pi))):
        raise ValueError("gamma_b must lie between 0 and pi")
    return shape


def synthesize_atmosphere_batch(
    adata: AtomicData,
    wavelengths,
    dz,
    atmosphere: Atmosphere,
    wavelength_parallelism: int = 1,
) -> jax.Array:
    """Synthesize complete columns and return ``(column, Stokes, wavelength)``."""

    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    absolute_wavelengths = _host_wavelength_axis(wavelengths)
    offsets = centered_wavelengths(adata, absolute_wavelengths)
    dz = _as_real(dz)
    shape = _validate_atmosphere(atmosphere)
    if len(shape) != 2:
        raise ValueError("a synthesis batch must have shape (columns, depth)")
    if shape[1] != dz.size:
        raise ValueError("atmosphere depth and dz must match")
    if (
        dz.ndim != 1
        or not bool(jnp.all(jnp.isfinite(dz)))
        or not bool(jnp.all(dz > 0.0))
    ):
        raise ValueError("dz must contain positive finite cell widths")
    return _SYNTHESIZE_COLUMNS_OFFSET_JIT(
        adata,
        offsets,
        dz,
        Atmosphere(*(_as_real(field) for field in atmosphere)),
        wavelength_parallelism=wavelength_parallelism,
    )


def synthesize_atmosphere_cube(
    adata: AtomicData,
    wavelengths,
    dz,
    atmosphere: Atmosphere,
    batch_columns: int = 16,
    show_progress: bool = True,
    description: str = "Synthesizing columns",
    wavelength_parallelism: int = 1,
) -> jax.Array:
    """Forward-synthesize ``(nx, ny, depth)`` fields in bounded-memory chunks."""

    batch_columns = _validate_positive_integer("batch_columns", batch_columns)
    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    absolute_wavelengths = _host_wavelength_axis(wavelengths)
    offsets = centered_wavelengths(adata, absolute_wavelengths)
    dz = _as_real(dz)
    if dz.ndim != 1 or dz.size == 0:
        raise ValueError("dz must be a non-empty one-dimensional array")
    shape = _validate_atmosphere(atmosphere)
    if len(shape) != 3:
        raise ValueError("a cube atmosphere must have shape (nx, ny, depth)")
    nx, ny, n_depth = shape
    flat = Atmosphere(
        *(_as_real(field).reshape((-1, n_depth)) for field in atmosphere)
    )
    n_columns = nx * ny
    n_wave = int(absolute_wavelengths.shape[0])
    output = np.empty((n_columns, 4, n_wave), dtype=NUMPY_REAL_DTYPE)
    starts = range(0, n_columns, batch_columns)
    iterator = tqdm(
        starts,
        total=math.ceil(n_columns / batch_columns),
        desc=description,
        disable=not show_progress,
    )
    for start in iterator:
        stop = min(start + batch_columns, n_columns)
        count = stop - start
        batch = Atmosphere(*(field[start:stop] for field in flat))
        if count < batch_columns:
            padding = batch_columns - count
            batch = Atmosphere(
                *(
                    jnp.concatenate(
                        (field, jnp.repeat(field[-1:], padding, axis=0)), axis=0
                    )
                    for field in batch
                )
            )
        spectra = _SYNTHESIZE_COLUMNS_OFFSET_JIT(
            adata,
            offsets,
            dz,
            batch,
            wavelength_parallelism=wavelength_parallelism,
        ).block_until_ready()
        output[start:stop] = np.asarray(spectra[:count])
    return jnp.asarray(
        output.reshape((nx, ny, 4, n_wave)), dtype=REAL_DTYPE
    )


def _logit(probability):
    probability = _as_real(probability)
    # 1 - 1e-9 rounds back to exactly one in fp32.  A dtype-aware open
    # interval prevents infinite logits and arctanh values in either mode.
    epsilon = jnp.asarray(
        max(1.0e-9, 4.0 * jnp.finfo(probability.dtype).eps),
        dtype=probability.dtype,
    )
    probability = jnp.clip(probability, epsilon, 1.0 - epsilon)
    return jnp.log(probability) - jnp.log1p(-probability)


def _clip_open_unit_interval(value):
    value = _as_real(value)
    margin = jnp.asarray(
        max(1.0e-9, 4.0 * jnp.finfo(value.dtype).eps), dtype=value.dtype
    )
    return jnp.clip(value, -1.0 + margin, 1.0 - margin)


def validate_neural_transform_domain(atmosphere: Atmosphere) -> None:
    """Reject physical profiles that the bounded decoder cannot represent."""

    _validate_atmosphere(atmosphere)
    positive = jnp.stack(
        (
            atmosphere.temperature,
            atmosphere.ne,
            atmosphere.nhtot,
            atmosphere.vturb,
        ),
        axis=-1,
    )
    if not bool(jnp.all((positive > _POSITIVE_LOWER) & (positive < _POSITIVE_UPPER))):
        raise ValueError("positive atmosphere fields exceed neural transform bounds")
    if not bool(jnp.all(jnp.abs(atmosphere.vz) < _VELOCITY_LIMIT)):
        raise ValueError("LOS velocity exceeds neural transform bounds")
    if not bool(
        jnp.all(
            (atmosphere.b > _MAGNETIC_FIELD_LOWER)
            & (atmosphere.b < _MAGNETIC_FIELD_UPPER)
        )
    ):
        raise ValueError("magnetic field exceeds neural transform bounds")
    if not bool(
        jnp.all(
            (atmosphere.gamma_b > _INCLINATION_MARGIN)
            & (atmosphere.gamma_b < jnp.pi - _INCLINATION_MARGIN)
        )
    ):
        raise ValueError("inclination exceeds neural transform bounds")
    if not bool(jnp.all(jnp.abs(_wrap_azimuth(atmosphere.chi_b)) < _AZIMUTH_LIMIT)):
        raise ValueError("azimuth lies on a neural transform boundary")


def atmosphere_to_latent(atmosphere: Atmosphere) -> jax.Array:
    """Encode a physical atmosphere into the network's eight latent channels."""

    if not isinstance(atmosphere.b, jax.core.Tracer):
        validate_neural_transform_domain(atmosphere)

    positive = jnp.stack(
        (
            atmosphere.temperature,
            atmosphere.ne,
            atmosphere.nhtot,
            atmosphere.vturb,
        ),
        axis=-1,
    )
    log_lower = jnp.log(_POSITIVE_LOWER)
    log_span = jnp.log(_POSITIVE_UPPER) - log_lower
    positive_unit = (jnp.log(positive) - log_lower) / log_span
    positive_latent = _logit(positive_unit)
    velocity_latent = jnp.arctanh(
        _clip_open_unit_interval(atmosphere.vz / _VELOCITY_LIMIT)
    )
    magnetic_log_lower = jnp.log(_MAGNETIC_FIELD_LOWER)
    magnetic_log_span = jnp.log(_MAGNETIC_FIELD_UPPER) - magnetic_log_lower
    magnetic_unit = (jnp.log(atmosphere.b) - magnetic_log_lower) / magnetic_log_span
    magnetic_latent = _logit(magnetic_unit)
    inclination_span = jnp.pi - 2.0 * _INCLINATION_MARGIN
    inclination_unit = (atmosphere.gamma_b - _INCLINATION_MARGIN) / inclination_span
    inclination_latent = _logit(inclination_unit)
    azimuth_latent = jnp.arctanh(
        _clip_open_unit_interval(
            _wrap_azimuth(atmosphere.chi_b) / _AZIMUTH_LIMIT
        )
    )
    return jnp.stack(
        (
            positive_latent[..., 0],
            positive_latent[..., 1],
            positive_latent[..., 2],
            velocity_latent,
            positive_latent[..., 3],
            magnetic_latent,
            inclination_latent,
            azimuth_latent,
        ),
        axis=-1,
    )


def latent_to_atmosphere(latent) -> Atmosphere:
    """Smoothly decode latent channels to finite, physical atmospheric fields."""

    latent = _as_real(latent)
    if latent.shape[-1] != 8:
        raise ValueError("latent atmosphere must have eight channels")
    positive_latent = jnp.stack(
        (latent[..., 0], latent[..., 1], latent[..., 2], latent[..., 4]),
        axis=-1,
    )
    log_lower = jnp.log(_POSITIVE_LOWER)
    log_span = jnp.log(_POSITIVE_UPPER) - log_lower
    positive = jnp.exp(log_lower + log_span * jax.nn.sigmoid(positive_latent))
    vz = _VELOCITY_LIMIT * jnp.tanh(latent[..., 3])
    magnetic_log_lower = jnp.log(_MAGNETIC_FIELD_LOWER)
    magnetic_log_span = jnp.log(_MAGNETIC_FIELD_UPPER) - magnetic_log_lower
    b = jnp.exp(magnetic_log_lower + magnetic_log_span * jax.nn.sigmoid(latent[..., 5]))
    inclination_span = jnp.pi - 2.0 * _INCLINATION_MARGIN
    gamma_b = _INCLINATION_MARGIN + inclination_span * jax.nn.sigmoid(latent[..., 6])
    chi_b = _AZIMUTH_LIMIT * jnp.tanh(latent[..., 7])
    return Atmosphere(
        positive[..., 0],
        positive[..., 1],
        positive[..., 2],
        vz,
        positive[..., 3],
        b,
        gamma_b,
        chi_b,
    )


def validate_spatial_reachability(
    reference: Atmosphere,
    truth: Atmosphere,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
) -> None:
    """Ensure a frozen FAL base can reach every truth latent channel."""

    reference_latent = atmosphere_to_latent(reference)
    truth_latent = atmosphere_to_latent(truth)
    if truth_latent.ndim != reference_latent.ndim + 2 or truth_latent.shape[-2:] != (
        reference_latent.shape[0],
        8,
    ):
        raise ValueError("truth and reference depth/channel dimensions do not match")
    spatial_scale = _as_real(spatial_scale)
    if spatial_scale.shape != (8,) or not bool(jnp.all(spatial_scale > 0.0)):
        raise ValueError("spatial_scale must contain eight positive values")
    required = jnp.max(
        jnp.abs(truth_latent - reference_latent[None, None, ...]),
        axis=(0, 1, 2),
    )
    if not bool(jnp.all(required < spatial_scale)):
        ratios = np.asarray(required / spatial_scale)
        failing = [
            LATENT_CHANNEL_NAMES[index] for index in np.flatnonzero(ratios >= 1.0)
        ]
        raise ValueError(
            "truth perturbation exceeds the frozen spatial correction range for: "
            + ", ".join(failing)
        )


def init_mlp(layer_sizes, key, zero_output: bool = False):
    """Initialize a GELU MLP as a tuple PyTree."""

    layer_sizes = tuple(int(size) for size in layer_sizes)
    if len(layer_sizes) < 2 or any(size <= 0 for size in layer_sizes):
        raise ValueError("layer_sizes must contain at least two positive sizes")
    keys = jax.random.split(key, len(layer_sizes) - 1)
    layers = []
    for index, (n_in, n_out, layer_key) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:], keys)
    ):
        weight = jax.random.normal(
            layer_key, (n_in, n_out), dtype=REAL_DTYPE
        ) * jnp.sqrt(
            2.0 / (n_in + n_out)
        )
        bias = jnp.zeros(n_out, dtype=REAL_DTYPE)
        if zero_output and index == len(layer_sizes) - 2:
            weight = jnp.zeros_like(weight)
            bias = jnp.zeros_like(bias)
        layers.append({"w": weight, "b": bias})
    return tuple(layers)


def apply_mlp(params, coordinates):
    """Evaluate an MLP on an arbitrary leading coordinate shape."""

    activation = _as_real(coordinates)
    for layer in params[:-1]:
        activation = jax.nn.gelu(activation @ layer["w"] + layer["b"])
    return activation @ params[-1]["w"] + params[-1]["b"]


def vertical_features(z_coordinates, n_frequencies: int):
    """Fourier-encode height so the base MLP can resolve FAL-C's transition."""

    z_coordinates = _as_real(z_coordinates)
    if z_coordinates.shape[-1] != 1:
        raise ValueError("vertical coordinates must have one feature")
    if not isinstance(n_frequencies, int) or n_frequencies < 0:
        raise ValueError("n_frequencies must be a non-negative integer")
    if n_frequencies == 0:
        return z_coordinates
    frequencies = 2.0 ** jnp.arange(n_frequencies, dtype=z_coordinates.dtype)
    angles = jnp.pi * z_coordinates * frequencies
    return jnp.concatenate((z_coordinates, jnp.sin(angles), jnp.cos(angles)), axis=-1)


def _base_features(params, z_coordinates):
    input_features = int(params[0]["w"].shape[0])
    if input_features < 1 or (input_features - 1) % 2:
        raise ValueError("base-network input size is not a valid Fourier encoding")
    return vertical_features(z_coordinates, (input_features - 1) // 2)


def initialize_neural_field(
    key,
    config: NeuralFieldConfig = NeuralFieldConfig(),
):
    """Initialize the learned vertical base and zero spatial correction."""

    config.validate()
    base_key, spatial_key = jax.random.split(key)
    return {
        "base": init_mlp(config.base_layers, base_key),
        "spatial": init_mlp(config.spatial_layers, spatial_key, zero_output=True),
    }


def neural_field_latent(
    params,
    coordinates,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
):
    """Return combined latent field and the bounded spatial correction."""

    coordinates = _as_real(coordinates)
    if coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have a final (x, y, z) axis")
    spatial_scale = _as_real(spatial_scale)
    if spatial_scale.shape != (8,):
        raise ValueError("spatial_scale must have shape (8,)")
    base_latent = apply_mlp(
        params["base"], _base_features(params["base"], coordinates[..., 2:3])
    )
    spatial_correction = spatial_scale * jnp.tanh(
        apply_mlp(params["spatial"], coordinates)
    )
    return base_latent + spatial_correction, spatial_correction


def evaluate_neural_field(
    params,
    coordinates,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
) -> Atmosphere:
    """Evaluate the neural atmosphere at normalized ``(x, y, z)`` coordinates."""

    latent, _ = neural_field_latent(params, coordinates, spatial_scale)
    return latent_to_atmosphere(latent)


_EVALUATE_NEURAL_FIELD_JIT = jax.jit(evaluate_neural_field)


def evaluate_neural_field_cube(
    params,
    coordinates,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    batch_columns: int = 16,
    show_progress: bool = True,
) -> Atmosphere:
    """Evaluate a full ``(x,y,z)`` neural cube in bounded-memory chunks."""

    batch_columns = _validate_positive_integer("batch_columns", batch_columns)
    coordinates = _as_real(coordinates)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (nx, ny, depth, 3)")
    nx, ny, n_depth, _ = coordinates.shape
    n_columns = nx * ny
    flat_coordinates = coordinates.reshape((n_columns, n_depth, 3))
    outputs = [
        np.empty((n_columns, n_depth), dtype=NUMPY_REAL_DTYPE)
        for _ in FIELD_NAMES
    ]
    iterator = tqdm(
        range(0, n_columns, batch_columns),
        total=math.ceil(n_columns / batch_columns),
        desc="Evaluating neural atmosphere",
        disable=not show_progress,
    )
    for start in iterator:
        stop = min(start + batch_columns, n_columns)
        count = stop - start
        batch = flat_coordinates[start:stop]
        if count < batch_columns:
            batch = jnp.concatenate(
                (
                    batch,
                    jnp.repeat(batch[-1:], batch_columns - count, axis=0),
                ),
                axis=0,
            )
        atmosphere = _EVALUATE_NEURAL_FIELD_JIT(
            params, batch, _as_real(spatial_scale)
        )
        jax.block_until_ready(atmosphere.temperature)
        for output, field in zip(outputs, atmosphere):
            output[start:stop] = np.asarray(field[:count])
    return Atmosphere(
        *(
            jnp.asarray(output.reshape((nx, ny, n_depth)), dtype=REAL_DTYPE)
            for output in outputs
        )
    )


def pretrain_falc(
    params,
    z_coordinates,
    reference: Atmosphere,
    steps: int = 2000,
    learning_rate: float = 2.0e-3,
    tolerance: float | None = None,
    show_progress: bool = True,
):
    """Supervise the vertical network on FAL-C and leave spatial output zero."""

    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0.0):
        raise ValueError("tolerance must be positive and finite when supplied")
    z_coordinates = _as_real(z_coordinates)
    if z_coordinates.ndim != 1:
        raise ValueError("z_coordinates must be one-dimensional")
    _validate_atmosphere(reference, expected_shape=(z_coordinates.size,))
    target = atmosphere_to_latent(reference)
    network_input = _base_features(
        params["base"], (2.0 * z_coordinates - 1.0).reshape((-1, 1))
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    base_params = params["base"]
    opt_state = optimizer.init(base_params)

    def loss_fn(candidate):
        prediction = apply_mlp(candidate, network_input)
        return jnp.mean((prediction - target) ** 2)

    def step(base, state):
        loss_value, grads = jax.value_and_grad(loss_fn)(base)
        updates, next_state = optimizer.update(grads, state, base)
        return optax.apply_updates(base, updates), next_state, loss_value

    def run_pretraining(base, state):
        initial_loss = loss_fn(base)

        def scan_step(carry, _):
            current, current_state, best, best_loss = carry
            next_params, next_state, loss_value = step(current, current_state)
            improved = loss_value < best_loss
            best = jax.tree.map(
                lambda best_leaf, current_leaf: jnp.where(
                    improved, current_leaf, best_leaf
                ),
                best,
                current,
            )
            best_loss = jnp.where(improved, loss_value, best_loss)
            return (next_params, next_state, best, best_loss), loss_value

        (base, state, best, best_loss), loss_history = jax.lax.scan(
            scan_step,
            (base, state, base, initial_loss),
            xs=None,
            length=steps,
        )
        final_loss = loss_fn(base)
        improved = final_loss < best_loss
        best = jax.tree.map(
            lambda best_leaf, final_leaf: jnp.where(
                improved, final_leaf, best_leaf
            ),
            best,
            base,
        )
        best_loss = jnp.where(improved, final_loss, best_loss)
        return best, jnp.concatenate((loss_history, final_loss[None])), best_loss

    compiled_pretraining = jax.jit(run_pretraining)
    iterator = tqdm(
        total=steps,
        desc="Pretraining FAL-C",
        disable=not show_progress,
    )
    best_params, all_losses, _ = compiled_pretraining(base_params, opt_state)
    # Materialize the complete trace once; the previous loop synchronized the
    # accelerator for every scalar loss value.
    all_losses = np.asarray(all_losses, dtype=float)
    history = all_losses[:-1]
    best_loss = float(np.min(all_losses))
    iterator.update(steps)
    if steps:
        iterator.set_postfix(loss=f"{history[-1]:.3e}")
    iterator.close()
    if tolerance is not None and steps > 0 and best_loss > tolerance:
        raise RuntimeError(
            "FAL-C pretraining did not reach the requested latent MSE: "
            f"best={best_loss:.3e}, tolerance={tolerance:.3e}; increase "
            "--pretrain-steps or network capacity"
        )

    return {"base": best_params, "spatial": params["spatial"]}, history


def continuum_normalization(observed_stokes):
    """Estimate a fixed continuum scale from the two Stokes-I window edges."""

    observed_stokes = _as_real(observed_stokes)
    if observed_stokes.ndim != 3 or observed_stokes.shape[1] != 4:
        raise ValueError("observed_stokes must have shape (columns, 4, wavelength)")
    continuum = 0.5 * (observed_stokes[:, 0, 0] + observed_stokes[:, 0, -1])
    tiny = jnp.finfo(observed_stokes.dtype).tiny
    return jnp.maximum(jnp.abs(continuum), tiny)


def weighted_stokes_mse(
    synthetic,
    observed,
    continuum,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    column_mask=None,
):
    """Continuum-normalized weighted full-Stokes mean-square error."""

    synthetic = _as_real(synthetic)
    observed = _as_real(observed)
    continuum = _as_real(continuum)
    weights = _as_real(stokes_weights)
    if synthetic.shape != observed.shape or synthetic.ndim != 3:
        raise ValueError("synthetic and observed must share (column, 4, wave) shape")
    if synthetic.shape[1] != 4:
        raise ValueError("the middle spectral axis must contain I, Q, U, V")
    if continuum.shape != (synthetic.shape[0],):
        raise ValueError("continuum must have one value per column")
    if weights.shape != (4,):
        raise ValueError("stokes_weights must have shape (4,)")
    residual = (synthetic - observed) / continuum[:, None, None]
    per_column = jnp.mean((residual * weights[None, :, None]) ** 2, axis=(1, 2))
    if column_mask is None:
        return jnp.mean(per_column)
    column_mask = _as_real(column_mask)
    if column_mask.shape != (synthetic.shape[0],):
        raise ValueError("column_mask must have one value per column")
    return jnp.sum(per_column * column_mask) / jnp.maximum(jnp.sum(column_mask), 1.0)


def neural_spectral_loss(
    params,
    coordinates,
    wavelengths,
    observed_stokes,
    continuum,
    dz,
    adata,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    column_mask=None,
    prior_weight: float = 0.0,
    wavelength_parallelism: int = 1,
):
    """Physics-informed objective from absolute wavelengths (legacy API).

    Production fp32 training uses :func:`neural_spectral_loss_offset`, because
    an absolute wavelength that has already reached a float32 device cannot be
    centered without recovering precision that was lost during that transfer.
    """

    latent, spatial_correction = neural_field_latent(params, coordinates, spatial_scale)
    return _latent_spectral_loss_impl(
        latent,
        spatial_correction,
        wavelengths,
        observed_stokes,
        continuum,
        dz,
        adata,
        stokes_weights,
        column_mask,
        prior_weight,
        wavelength_parallelism,
        wavelengths_are_offsets=False,
    )


def neural_spectral_loss_offset(
    params,
    coordinates,
    wavelength_offsets_nm,
    observed_stokes,
    continuum,
    dz,
    adata,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    column_mask=None,
    prior_weight: float = 0.0,
    wavelength_parallelism: int = 1,
):
    """Physics-informed objective from host-precentered wavelength offsets."""

    latent, spatial_correction = neural_field_latent(params, coordinates, spatial_scale)
    return _latent_spectral_loss_impl(
        latent,
        spatial_correction,
        wavelength_offsets_nm,
        observed_stokes,
        continuum,
        dz,
        adata,
        stokes_weights,
        column_mask,
        prior_weight,
        wavelength_parallelism,
        wavelengths_are_offsets=True,
    )


def _latent_spectral_loss_impl(
    latent,
    spatial_correction,
    wavelengths,
    observed_stokes,
    continuum,
    dz,
    adata,
    stokes_weights,
    column_mask,
    prior_weight,
    wavelength_parallelism,
    *,
    wavelengths_are_offsets,
):
    atmosphere = latent_to_atmosphere(latent)
    synthesis = (
        _synthesize_columns_offset_core
        if wavelengths_are_offsets
        else _synthesize_columns_core
    )
    synthetic = synthesis(
        adata,
        wavelengths,
        dz,
        atmosphere,
        wavelength_parallelism,
    )
    spectral_loss = weighted_stokes_mse(
        synthetic,
        observed_stokes,
        continuum,
        stokes_weights,
        column_mask,
    )
    per_column_prior = jnp.mean(spatial_correction**2, axis=(1, 2))
    if column_mask is None:
        prior = jnp.mean(per_column_prior)
    else:
        prior = jnp.sum(per_column_prior * column_mask) / jnp.maximum(
            jnp.sum(column_mask), 1.0
        )
    return spectral_loss + prior_weight * prior, (spectral_loss, prior)


def frozen_base_spectral_loss(
    spatial_params,
    base_latent,
    coordinates,
    wavelengths,
    observed_stokes,
    continuum,
    dz,
    adata,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    column_mask=None,
    prior_weight: float = 0.0,
    wavelength_parallelism: int = 1,
):
    """Spectral loss that differentiates only the trainable spatial network."""

    spatial_correction = _as_real(spatial_scale) * jnp.tanh(
        apply_mlp(spatial_params, coordinates)
    )
    latent = jax.lax.stop_gradient(base_latent)[None, ...] + spatial_correction
    return _latent_spectral_loss_impl(
        latent,
        spatial_correction,
        wavelengths,
        observed_stokes,
        continuum,
        dz,
        adata,
        stokes_weights,
        column_mask,
        prior_weight,
        wavelength_parallelism,
        wavelengths_are_offsets=False,
    )


def frozen_base_spectral_loss_offset(
    spatial_params,
    base_latent,
    coordinates,
    wavelength_offsets_nm,
    observed_stokes,
    continuum,
    dz,
    adata,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    column_mask=None,
    prior_weight: float = 0.0,
    wavelength_parallelism: int = 1,
):
    """Frozen-base loss using precision-safe centered wavelengths."""

    spatial_correction = _as_real(spatial_scale) * jnp.tanh(
        apply_mlp(spatial_params, coordinates)
    )
    latent = jax.lax.stop_gradient(base_latent)[None, ...] + spatial_correction
    return _latent_spectral_loss_impl(
        latent,
        spatial_correction,
        wavelength_offsets_nm,
        observed_stokes,
        continuum,
        dz,
        adata,
        stokes_weights,
        column_mask,
        prior_weight,
        wavelength_parallelism,
        wavelengths_are_offsets=True,
    )


def padded_column_batches(n_columns: int, batch_size: int, rng):
    """Return shuffled fixed-size indices, masks, and exact one-pass coverage."""

    n_columns = _validate_positive_integer("n_columns", n_columns)
    batch_size = _validate_positive_integer("batch_size", batch_size)
    permutation = np.asarray(rng.permutation(n_columns), dtype=np.int64)
    batches = []
    for start in range(0, n_columns, batch_size):
        indices = permutation[start : start + batch_size]
        count = indices.size
        mask = np.zeros(batch_size, dtype=float)
        mask[:count] = 1.0
        if count < batch_size:
            indices = np.concatenate(
                (indices, np.repeat(indices[-1], batch_size - count))
            )
        batches.append((indices, mask, count))
    return batches


def sample_wavelength_indices(
    rng,
    wavelengths,
    batch_size: int,
):
    """Uniformly sample wavelengths for an unbiased full-spectrum MSE."""

    shape = getattr(wavelengths, "shape", None)
    if shape is None:
        shape = np.shape(wavelengths)
    n_wave = math.prod(shape)
    batch_size = _validate_positive_integer("wavelength batch_size", batch_size)
    if batch_size >= n_wave:
        return np.arange(n_wave, dtype=np.int64)
    return np.sort(rng.choice(n_wave, size=batch_size, replace=False))


def _spectral_epoch_batches(
    n_columns: int,
    column_batch_size: int,
    wavelengths,
    wavelength_batch_size: int,
    rng,
):
    """Build fixed-shape epoch indices in the legacy RNG-consumption order."""

    column_batches = padded_column_batches(n_columns, column_batch_size, rng)
    column_indices = np.stack([indices for indices, _, _ in column_batches])
    masks = np.stack([mask for _, mask, _ in column_batches])
    wavelength_indices = np.stack(
        [
            sample_wavelength_indices(
                rng,
                wavelengths,
                wavelength_batch_size,
            )
            for _ in column_batches
        ]
    )
    return column_indices, wavelength_indices, masks


def train_spectral_inversion(
    params,
    coordinates,
    observed_stokes,
    wavelengths,
    line_centers,
    dz,
    adata,
    epochs: int = 10,
    batch_columns: int = 8,
    wavelength_batch: int = 48,
    learning_rate: float = 3.0e-4,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    prior_weight: float = 1.0e-5,
    fine_tune_base: bool = False,
    require_improvement: bool = True,
    seed: int = 0,
    show_progress: bool = True,
    wavelength_parallelism: int = DEFAULT_WAVELENGTH_PARALLELISM,
    validation_columns: int = DEFAULT_VALIDATION_COLUMNS,
):
    """Fit the shared neural field using shuffled complete-column batches."""

    if not isinstance(epochs, int) or epochs < 0:
        raise ValueError("epochs must be a non-negative integer")
    batch_columns = _validate_positive_integer("batch_columns", batch_columns)
    wavelength_batch = _validate_positive_integer("wavelength_batch", wavelength_batch)
    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    validation_columns = _validate_positive_integer(
        "validation_columns", validation_columns
    )
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(prior_weight) or prior_weight < 0.0:
        raise ValueError("prior_weight must be non-negative and finite")

    absolute_wavelengths = _host_wavelength_axis(wavelengths)
    host_line_centers = np.asarray(line_centers, dtype=np.float64)
    coordinates = _as_real(coordinates)
    observed_stokes = _as_real(observed_stokes)
    wavelengths = centered_wavelengths(adata, absolute_wavelengths)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (columns, depth, 3)")
    if observed_stokes.shape != (
        coordinates.shape[0],
        4,
        absolute_wavelengths.size,
    ):
        raise ValueError("observed_stokes must have shape (columns, 4, n_wave)")
    if jnp.asarray(dz).ndim != 1 or int(jnp.asarray(dz).size) != coordinates.shape[1]:
        raise ValueError("coordinate depth and dz must match")
    if host_line_centers.ndim != 1 or host_line_centers.size == 0:
        raise ValueError("line_centers must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(host_line_centers)) or not np.all(
        (host_line_centers >= absolute_wavelengths[0])
        & (host_line_centers <= absolute_wavelengths[-1])
    ):
        raise ValueError("every line center must be covered by wavelengths")
    dz = _as_real(dz)
    spatial_scale = _as_real(spatial_scale)
    stokes_weights = _as_real(stokes_weights)
    eager_arrays = {
        "coordinates": coordinates,
        "observed_stokes": observed_stokes,
        "wavelength_offsets": wavelengths,
        "dz": dz,
        "spatial_scale": spatial_scale,
        "stokes_weights": stokes_weights,
    }
    for name, array in eager_arrays.items():
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError(f"{name} must contain only finite values")
    if not bool(jnp.all(dz > 0.0)):
        raise ValueError("dz must be strictly positive")
    if spatial_scale.shape != (8,) or not bool(jnp.all(spatial_scale > 0.0)):
        raise ValueError("spatial_scale must contain eight positive values")
    if stokes_weights.shape != (4,) or not bool(jnp.all(stokes_weights > 0.0)):
        raise ValueError("stokes_weights must contain four positive values")
    if not bool(
        jnp.allclose(
            coordinates[..., 2],
            coordinates[0, :, 2][None, :],
            rtol=0.0,
            atol=8.0 * jnp.finfo(coordinates.dtype).eps,
        )
    ):
        raise ValueError("all columns must use the same ordered height coordinates")
    continuum = continuum_normalization(observed_stokes)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))

    base_params = params["base"]
    if fine_tune_base:
        trainable = params

        def batch_loss(
            candidate,
            batch_coordinates,
            batch_wavelengths,
            batch_observed,
            batch_continuum,
            batch_mask,
        ):
            return neural_spectral_loss_offset(
                candidate,
                batch_coordinates,
                batch_wavelengths,
                batch_observed,
                batch_continuum,
                dz,
                adata,
                spatial_scale,
                stokes_weights,
                batch_mask,
                prior_weight,
                wavelength_parallelism,
            )

        def assemble(candidate):
            return candidate

    else:
        trainable = params["spatial"]
        z_coordinates = coordinates[0, :, 2:3]
        base_latent = apply_mlp(base_params, _base_features(base_params, z_coordinates))

        def batch_loss(
            candidate,
            batch_coordinates,
            batch_wavelengths,
            batch_observed,
            batch_continuum,
            batch_mask,
        ):
            return frozen_base_spectral_loss_offset(
                candidate,
                base_latent,
                batch_coordinates,
                batch_wavelengths,
                batch_observed,
                batch_continuum,
                dz,
                adata,
                spatial_scale,
                stokes_weights,
                batch_mask,
                prior_weight,
                wavelength_parallelism,
            )

        def assemble(candidate):
            return {"base": base_params, "spatial": candidate}

    opt_state = optimizer.init(trainable)

    def train_step(
        candidate,
        state,
        batch_coordinates,
        batch_wavelengths,
        batch_observed,
        batch_continuum,
        batch_mask,
    ):
        (loss_value, (spectral_value, prior_value)), grads = jax.value_and_grad(
            batch_loss, has_aux=True
        )(
            candidate,
            batch_coordinates,
            batch_wavelengths,
            batch_observed,
            batch_continuum,
            batch_mask,
        )
        updates, next_state = optimizer.update(grads, state, candidate)
        next_params = optax.apply_updates(candidate, updates)
        metrics = jnp.stack(
            (loss_value, spectral_value, prior_value),
        )
        return next_params, next_state, metrics

    rng = np.random.default_rng(seed)
    n_columns = coordinates.shape[0]
    history = np.empty((epochs, 3), dtype=float)

    validation_count = min(validation_columns, n_columns)
    validation_indices = np.linspace(
        0, n_columns - 1, num=validation_count, dtype=np.int64
    )
    validation_mask = np.zeros(validation_columns, dtype=float)
    validation_mask[:validation_count] = 1.0
    if validation_count < validation_columns:
        validation_indices = np.concatenate(
            (
                validation_indices,
                np.repeat(
                    validation_indices[-1],
                    validation_columns - validation_count,
                ),
            )
        )
    validation_coordinates = coordinates[validation_indices]
    validation_observed = observed_stokes[validation_indices]
    validation_continuum = continuum[validation_indices]

    def validation_loss(
        candidate,
        batch_coordinates,
        full_wavelengths,
        batch_observed,
        batch_continuum,
        batch_mask,
    ):
        return batch_loss(
            candidate,
            batch_coordinates,
            full_wavelengths,
            batch_observed,
            batch_continuum,
            batch_mask,
        )[1][0]

    compiled_validation = jax.jit(validation_loss)
    validation_arguments = (
        validation_coordinates,
        wavelengths,
        validation_observed,
        validation_continuum,
        jnp.asarray(validation_mask),
    )

    def train_epoch(
        candidate,
        state,
        all_coordinates,
        all_observed_stokes,
        all_continuum,
        all_wavelengths,
        epoch_column_indices,
        epoch_wavelength_indices,
        epoch_masks,
    ):
        """Run every ordered optimizer update in one accelerator dispatch."""

        def scan_step(carry, batch):
            current, current_state, metric_totals = carry
            column_indices, wave_indices, column_mask = batch
            batch_observed = jnp.take(
                jnp.take(all_observed_stokes, column_indices, axis=0),
                wave_indices,
                axis=2,
            )
            current, current_state, metrics = train_step(
                current,
                current_state,
                jnp.take(all_coordinates, column_indices, axis=0),
                jnp.take(all_wavelengths, wave_indices, axis=0),
                batch_observed,
                jnp.take(all_continuum, column_indices, axis=0),
                column_mask,
            )
            metric_totals = metric_totals + jnp.sum(column_mask) * metrics
            return (current, current_state, metric_totals), None

        initial_totals = jnp.zeros(3, dtype=all_observed_stokes.dtype)
        (candidate, state, metric_totals), _ = jax.lax.scan(
            scan_step,
            (candidate, state, initial_totals),
            (epoch_column_indices, epoch_wavelength_indices, epoch_masks),
        )
        return candidate, state, metric_totals

    compiled_epoch = jax.jit(train_epoch)
    validation_history = np.empty(epochs + 1, dtype=float)
    validation_history[0] = float(
        compiled_validation(trainable, *validation_arguments)
    )
    best_validation = validation_history[0]
    best_trainable = trainable
    epoch_iterator = tqdm(
        range(epochs),
        desc="Inverting Stokes cube",
        disable=not show_progress,
    )
    for epoch in epoch_iterator:
        (
            epoch_column_indices,
            epoch_wavelength_indices,
            epoch_masks,
        ) = _spectral_epoch_batches(
            n_columns,
            batch_columns,
            wavelengths,
            wavelength_batch,
            rng,
        )
        trainable, opt_state, metric_totals = compiled_epoch(
            trainable,
            opt_state,
            coordinates,
            observed_stokes,
            continuum,
            wavelengths,
            jnp.asarray(epoch_column_indices),
            jnp.asarray(epoch_wavelength_indices),
            jnp.asarray(epoch_masks),
        )
        validation_value = compiled_validation(trainable, *validation_arguments)
        # One transfer/synchronization per epoch replaces three scalar device
        # reads for every column batch.  The optimizer updates, RNG order, and
        # count-weighted history are unchanged.
        epoch_metrics = np.asarray(
            jnp.concatenate(
                (
                    metric_totals / n_columns,
                    jnp.reshape(validation_value, (1,)),
                )
            ),
            dtype=float,
        )
        history[epoch] = epoch_metrics[:3]
        validation_history[epoch + 1] = epoch_metrics[3]
        if validation_history[epoch + 1] < best_validation:
            best_validation = validation_history[epoch + 1]
            best_trainable = trainable
        epoch_iterator.set_postfix(
            loss=f"{history[epoch, 0]:.3e}",
            spectral=f"{history[epoch, 1]:.3e}",
            validation=f"{validation_history[epoch + 1]:.3e}",
        )
    if (
        require_improvement
        and epochs > 0
        and not best_validation < validation_history[0]
    ):
        raise RuntimeError(
            "spectral inversion did not improve the fixed full-wavelength "
            "validation loss; adjust optimization settings"
        )
    return assemble(best_trainable), history, validation_history


_DATASET_FIELD_KEYS = {
    "temperature": "temperature_K",
    "ne": "electron_density_m3",
    "nhtot": "hydrogen_density_m3",
    "vz": "los_velocity_m_s",
    "vturb": "microturbulence_m_s",
    "b": "magnetic_field_T",
    "gamma_b": "inclination_rad",
    "chi_b": "azimuth_rad",
}


def _atmosphere_payload(prefix: str, atmosphere: Atmosphere):
    return {
        f"{prefix}{_DATASET_FIELD_KEYS[name]}": np.asarray(field)
        for name, field in zip(FIELD_NAMES, atmosphere)
    }


def _atmosphere_from_payload(payload, prefix: str) -> Atmosphere:
    return Atmosphere(
        *(
            _as_real(payload[f"{prefix}{_DATASET_FIELD_KEYS[name]}"])
            for name in FIELD_NAMES
        )
    )


def _stored_precision_tolerance(array, *, minimum_atol=0.0):
    """Return validation tolerances appropriate for an archived float dtype."""

    dtype = np.asarray(array).dtype
    if np.issubdtype(dtype, np.floating) and dtype.itemsize <= 4:
        epsilon = np.finfo(dtype).eps
        return 8.0 * epsilon, max(minimum_atol, 8.0 * epsilon)
    return 1.0e-12, minimum_atol


def _validate_precision_metadata(payload, key: str) -> None:
    if key not in payload:
        return
    value = str(np.asarray(payload[key]))
    if value not in adora_precision.SUPPORTED_PRECISIONS:
        raise ValueError(
            f"{key} must be one of: {', '.join(adora_precision.SUPPORTED_PRECISIONS)}"
        )


def validate_test_cube(payload) -> tuple[int, int, int, int]:
    """Validate the named NPZ-compatible test-cube schema."""

    required = {
        "schema_version",
        "stokes_labels",
        "x_normalized",
        "y_normalized",
        "height_m",
        "height_normalized",
        "cell_width_m",
        "wavelength_nm",
        "line_center_nm",
        "atomic_data_sha256",
        "observed_stokes",
        "perturbation_envelope",
        *(f"reference_{key}" for key in _DATASET_FIELD_KEYS.values()),
        *(f"truth_{key}" for key in _DATASET_FIELD_KEYS.values()),
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"test cube is missing keys: {', '.join(sorted(missing))}")
    if int(np.asarray(payload["schema_version"])) != SCHEMA_VERSION:
        raise ValueError("unsupported test-cube schema version")
    _validate_precision_metadata(payload, "synthesis_precision")
    x = np.asarray(payload["x_normalized"])
    y = np.asarray(payload["y_normalized"])
    height = np.asarray(payload["height_m"])
    height_normalized = np.asarray(payload["height_normalized"])
    dz = np.asarray(payload["cell_width_m"])
    wavelengths = np.asarray(payload["wavelength_nm"])
    if x.ndim != 1 or y.ndim != 1 or height.ndim != 1 or wavelengths.ndim != 1:
        raise ValueError("coordinate and wavelength axes must be one-dimensional")
    nx, ny, n_depth, n_wave = x.size, y.size, height.size, wavelengths.size
    if min(nx, ny, n_depth, n_wave) <= 0:
        raise ValueError("test-cube axes must be non-empty")
    if n_depth < 2:
        raise ValueError("test-cube height must contain at least two points")
    for name, axis in (
        ("x_normalized", x),
        ("y_normalized", y),
        ("height_m", height),
        ("height_normalized", height_normalized),
    ):
        if np.any(~np.isfinite(axis)):
            raise ValueError(f"{name} must be finite")
    if height_normalized.shape != (n_depth,):
        raise ValueError("height_normalized must match height_m")
    if np.any(np.diff(height) <= 0.0) or np.any(np.diff(height_normalized) <= 0.0):
        raise ValueError("height axes must be strictly increasing")
    if (
        np.any(x < 0.0)
        or np.any(x > 1.0)
        or np.any(y < 0.0)
        or np.any(y > 1.0)
        or np.any(height_normalized < 0.0)
        or np.any(height_normalized > 1.0)
    ):
        raise ValueError("normalized coordinate axes must lie in [0, 1]")
    if (nx > 1 and np.any(np.diff(x) <= 0.0)) or (ny > 1 and np.any(np.diff(y) <= 0.0)):
        raise ValueError("horizontal coordinate axes must be strictly increasing")
    expected_height_normalized = (height - height[0]) / (height[-1] - height[0])
    height_rtol, height_atol = _stored_precision_tolerance(height_normalized)
    if not np.allclose(
        height_normalized,
        expected_height_normalized,
        rtol=height_rtol,
        atol=height_atol,
    ):
        raise ValueError("height_normalized is inconsistent with height_m")
    labels = tuple(str(label) for label in np.asarray(payload["stokes_labels"]))
    if labels != STOKES_LABELS:
        raise ValueError("stokes_labels must be I, Q, U, V")
    fingerprint = str(np.asarray(payload["atomic_data_sha256"]))
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("atomic_data_sha256 must be a lowercase SHA-256 digest")
    if dz.shape != (n_depth,) or np.any(~np.isfinite(dz)) or np.any(dz <= 0.0):
        raise ValueError("cell_width_m must be positive, finite, and match height")
    expected_dz = np.concatenate((np.diff(height)[:1], np.diff(height)))
    dz_rtol, dz_atol = _stored_precision_tolerance(dz, minimum_atol=1.0e-9)
    if not np.allclose(
        dz,
        expected_dz,
        rtol=dz_rtol,
        atol=dz_atol,
    ):
        raise ValueError("cell_width_m is inconsistent with height_m")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelength_nm must be positive and finite")
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelength_nm must be strictly increasing")
    line_centers = np.asarray(payload["line_center_nm"])
    if (
        line_centers.ndim != 1
        or line_centers.size == 0
        or np.any(~np.isfinite(line_centers))
        or np.any(line_centers < wavelengths[0])
        or np.any(line_centers > wavelengths[-1])
    ):
        raise ValueError("line centers must be finite and covered by wavelength_nm")
    observed = np.asarray(payload["observed_stokes"])
    if observed.shape != (nx, ny, 4, n_wave):
        raise ValueError("observed_stokes must have shape (nx, ny, 4, n_wave)")
    if np.any(~np.isfinite(observed)):
        raise ValueError("observed_stokes must be finite")
    reference = _atmosphere_from_payload(payload, "reference_")
    truth = _atmosphere_from_payload(payload, "truth_")
    _validate_atmosphere(reference, expected_shape=(n_depth,))
    _validate_atmosphere(truth, expected_shape=(nx, ny, n_depth))
    envelope = np.asarray(payload["perturbation_envelope"])
    if envelope.shape != (nx, ny, n_depth) or np.any(~np.isfinite(envelope)):
        raise ValueError("perturbation_envelope has an invalid shape or values")
    return nx, ny, n_depth, n_wave


def save_test_cube(path, payload) -> None:
    """Validate and save a compressed synthetic observation cube."""

    validate_test_cube(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: np.asarray(value) for key, value in payload.items()}
    )


def load_test_cube(path=DEFAULT_DATASET):
    """Load a synthetic observation cube without enabling pickle."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    validate_test_cube(payload)
    return payload


def validate_inversion_result(payload) -> tuple[int, int, int, int]:
    """Validate a self-contained fitted-cube result archive."""

    nx, ny, n_depth, n_wave = validate_test_cube(payload)
    _validate_precision_metadata(payload, "inversion_precision")
    required = {
        "synthetic_stokes",
        "pretrain_loss",
        "inversion_loss_total_spectral_prior",
        "inversion_loss_columns",
        "validation_full_wavelength_loss",
        "best_validation_epoch",
        "best_validation_loss",
        "final_full_cube_spectral_loss",
        "base_layers",
        "spatial_layers",
        "spatial_scale",
        "stokes_weights",
        "prior_weight",
        "seed",
        *(f"inferred_{key}" for key in _DATASET_FIELD_KEYS.values()),
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"inversion result is missing keys: {', '.join(sorted(missing))}"
        )
    synthetic = np.asarray(payload["synthetic_stokes"])
    if synthetic.shape != (nx, ny, 4, n_wave) or np.any(~np.isfinite(synthetic)):
        raise ValueError("synthetic_stokes has an invalid shape or values")
    inferred = _atmosphere_from_payload(payload, "inferred_")
    _validate_atmosphere(inferred, expected_shape=(nx, ny, n_depth))
    pretrain_history = np.asarray(payload["pretrain_loss"])
    inversion_history = np.asarray(payload["inversion_loss_total_spectral_prior"])
    validation_history = np.asarray(payload["validation_full_wavelength_loss"])
    if pretrain_history.ndim != 1 or np.any(~np.isfinite(pretrain_history)):
        raise ValueError("pretrain_loss must be a finite one-dimensional history")
    if (
        inversion_history.ndim != 2
        or inversion_history.shape[1] != 3
        or np.any(~np.isfinite(inversion_history))
    ):
        raise ValueError("inversion loss history must have three finite columns")
    if tuple(str(value) for value in payload["inversion_loss_columns"]) != (
        "total",
        "spectral",
        "prior",
    ):
        raise ValueError("inversion_loss_columns must label total/spectral/prior")
    if validation_history.shape != (inversion_history.shape[0] + 1,) or np.any(
        ~np.isfinite(validation_history)
    ):
        raise ValueError("validation loss history must bracket all inversion epochs")
    best_epoch = int(np.asarray(payload["best_validation_epoch"]))
    best_loss = float(np.asarray(payload["best_validation_loss"]))
    if best_epoch != int(np.argmin(validation_history)) or not np.isclose(
        best_loss, np.min(validation_history), rtol=0.0, atol=0.0
    ):
        raise ValueError("best validation metadata do not match the loss history")
    if not np.isfinite(float(np.asarray(payload["final_full_cube_spectral_loss"]))):
        raise ValueError("final_full_cube_spectral_loss must be finite")
    return nx, ny, n_depth, n_wave


def generate_test_cube(
    output_path=DEFAULT_DATASET,
    kurucz_path=FE_I_6301_6302_LINE_LIST,
    nx: int = 50,
    ny: int = 50,
    n_wave: int = 201,
    wavelength_padding_nm: float = 0.05,
    synthesis_batch_columns: int = 16,
    magnetic_field_t: float = 0.05,
    inclination_rad: float = 0.8,
    azimuth_rad: float = 0.3,
    perturbation: PerturbationConfig = PerturbationConfig(),
    show_progress: bool = True,
    wavelength_parallelism: int = DEFAULT_WAVELENGTH_PARALLELISM,
):
    """Create, perturb, synthesize, and save the requested FAL-C test cube."""

    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    perturbation.validate()
    log_b_min = math.log(magnetic_field_t) + min(0.0, perturbation.log_b)
    log_b_max = math.log(magnetic_field_t) + max(0.0, perturbation.log_b)
    if log_b_min <= math.log(_MAGNETIC_FIELD_LOWER) or log_b_max >= math.log(
        _MAGNETIC_FIELD_UPPER
    ):
        raise ValueError(
            "the reference field plus perturbation exceeds the neural magnetic "
            "transform bounds"
        )
    gamma_min = inclination_rad + min(0.0, perturbation.inclination_rad)
    gamma_max = inclination_rad + max(0.0, perturbation.inclination_rad)
    if gamma_min <= _INCLINATION_MARGIN or gamma_max >= math.pi - _INCLINATION_MARGIN:
        raise ValueError(
            "the reference inclination plus perturbation exceeds the neural "
            "inclination bounds"
        )
    lines = read_kurucz(kurucz_path)
    wavelengths = build_wavelength_grid(lines, n_wave, wavelength_padding_nm)
    (
        x,
        y,
        height,
        z_normalized,
        dz,
        _,
        reference,
        reference_cube,
    ) = create_falc_reference_cube(
        nx,
        ny,
        magnetic_field_t,
        inclination_rad,
        azimuth_rad,
    )
    positive_perturbations = (
        perturbation.log_temperature,
        perturbation.log_ne,
        perturbation.log_nhtot,
        perturbation.log_vturb,
    )
    positive_profiles = (
        reference.temperature,
        reference.ne,
        reference.nhtot,
        reference.vturb,
    )
    for name, profile, amplitude, lower, upper in zip(
        ("temperature", "ne", "nhtot", "vturb"),
        positive_profiles,
        positive_perturbations,
        _POSITIVE_LOWER_VALUES,
        _POSITIVE_UPPER_VALUES,
    ):
        profile_log = np.log(np.asarray(profile, dtype=np.float64))
        perturbed_log_min = float(np.min(profile_log)) + min(0.0, amplitude)
        perturbed_log_max = float(np.max(profile_log)) + max(0.0, amplitude)
        if perturbed_log_min <= math.log(lower) or perturbed_log_max >= math.log(
            upper
        ):
            raise ValueError(
                f"the reference {name} plus perturbation exceeds the neural "
                "transform bounds"
            )
    truth, envelope = perturb_falc_cube(
        reference_cube, x, y, z_normalized, perturbation
    )
    validate_neural_transform_domain(reference)
    validate_neural_transform_domain(truth)
    observed = synthesize_atmosphere_cube(
        lines,
        wavelengths,
        dz,
        truth,
        batch_columns=synthesis_batch_columns,
        show_progress=show_progress,
        description="Synthesizing truth cube",
        wavelength_parallelism=wavelength_parallelism,
    )
    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "stokes_labels": np.asarray(STOKES_LABELS),
        "x_normalized": np.asarray(x),
        "y_normalized": np.asarray(y),
        "height_m": np.asarray(height),
        "height_normalized": np.asarray(z_normalized),
        "cell_width_m": np.asarray(dz),
        "wavelength_nm": np.asarray(wavelengths),
        "line_center_nm": _canonical_line_centers(lines),
        "atomic_data_sha256": np.asarray(atomic_data_fingerprint(lines)),
        "synthesis_precision": np.asarray(adora_precision.configured_precision()),
        "observed_stokes": np.asarray(observed),
        "perturbation_envelope": np.asarray(envelope),
        **_atmosphere_payload("reference_", reference),
        **_atmosphere_payload("truth_", truth),
    }
    save_test_cube(output_path, payload)
    return payload


def _validate_neural_parameters(params, config: NeuralFieldConfig) -> None:
    config.validate()
    for network_name, layer_sizes in (
        ("base", config.base_layers),
        ("spatial", config.spatial_layers),
    ):
        if (
            network_name not in params
            or len(params[network_name]) != len(layer_sizes) - 1
        ):
            raise ValueError(f"{network_name} parameters do not match the architecture")
        for index, (layer, n_in, n_out) in enumerate(
            zip(params[network_name], layer_sizes[:-1], layer_sizes[1:])
        ):
            weight = jnp.asarray(layer["w"])
            bias = jnp.asarray(layer["b"])
            if weight.shape != (n_in, n_out) or bias.shape != (n_out,):
                raise ValueError(
                    f"{network_name} layer {index} has incompatible shapes"
                )
            if not bool(jnp.all(jnp.isfinite(weight))) or not bool(
                jnp.all(jnp.isfinite(bias))
            ):
                raise ValueError(
                    f"{network_name} layer {index} contains non-finite parameters"
                )


def save_checkpoint(path, params, config: NeuralFieldConfig) -> None:
    """Save a versioned warm-start archive for both MLPs and their transforms."""

    _validate_neural_parameters(params, config)
    payload = {
        "checkpoint_schema_version": np.asarray(CHECKPOINT_SCHEMA_VERSION),
        "checkpoint_precision": np.asarray(adora_precision.configured_precision()),
        "base_layers": np.asarray(config.base_layers, dtype=np.int64),
        "spatial_layers": np.asarray(config.spatial_layers, dtype=np.int64),
        "spatial_scale": np.asarray(config.spatial_scale, dtype=np.float64),
        "latent_channel_names": np.asarray(LATENT_CHANNEL_NAMES),
        "positive_lower": np.asarray(_POSITIVE_LOWER_VALUES, dtype=np.float64),
        "positive_upper": np.asarray(_POSITIVE_UPPER_VALUES, dtype=np.float64),
        "velocity_limit": np.asarray(_VELOCITY_LIMIT, dtype=np.float64),
        "magnetic_field_lower": np.asarray(
            _MAGNETIC_FIELD_LOWER, dtype=np.float64
        ),
        "magnetic_field_upper": np.asarray(
            _MAGNETIC_FIELD_UPPER, dtype=np.float64
        ),
        "inclination_margin": np.asarray(_INCLINATION_MARGIN, dtype=np.float64),
        "azimuth_limit": np.asarray(_AZIMUTH_LIMIT, dtype=np.float64),
    }
    for network_name in ("base", "spatial"):
        for index, layer in enumerate(params[network_name]):
            payload[f"{network_name}_{index}_weight"] = np.asarray(layer["w"])
            payload[f"{network_name}_{index}_bias"] = np.asarray(layer["b"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_checkpoint(path):
    """Load neural-field parameters and architecture from a saved checkpoint."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if (
            int(np.asarray(archive["checkpoint_schema_version"]))
            != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported checkpoint schema version")
        expected_transform = {
            "positive_lower": np.asarray(_POSITIVE_LOWER_VALUES, dtype=np.float64),
            "positive_upper": np.asarray(_POSITIVE_UPPER_VALUES, dtype=np.float64),
            "velocity_limit": np.asarray(_VELOCITY_LIMIT, dtype=np.float64),
            "magnetic_field_lower": np.asarray(
                _MAGNETIC_FIELD_LOWER, dtype=np.float64
            ),
            "magnetic_field_upper": np.asarray(
                _MAGNETIC_FIELD_UPPER, dtype=np.float64
            ),
            "inclination_margin": np.asarray(
                _INCLINATION_MARGIN, dtype=np.float64
            ),
            "azimuth_limit": np.asarray(_AZIMUTH_LIMIT, dtype=np.float64),
        }
        if "checkpoint_precision" in archive.files:
            _validate_precision_metadata(archive, "checkpoint_precision")
        if tuple(str(value) for value in archive["latent_channel_names"]) != (
            LATENT_CHANNEL_NAMES
        ):
            raise ValueError("checkpoint latent-channel semantics do not match")
        for name, expected in expected_transform.items():
            if not np.array_equal(np.asarray(archive[name]), expected):
                raise ValueError(f"checkpoint {name} transform does not match")
        base_layers = tuple(int(value) for value in archive["base_layers"])
        spatial_layers = tuple(int(value) for value in archive["spatial_layers"])
        config = NeuralFieldConfig(
            base_hidden=base_layers[1:-1],
            spatial_hidden=spatial_layers[1:-1],
            base_frequencies=(base_layers[0] - 1) // 2,
            spatial_scale=tuple(float(value) for value in archive["spatial_scale"]),
        )
        params = {}
        for network_name, layers in (
            ("base", base_layers),
            ("spatial", spatial_layers),
        ):
            params[network_name] = tuple(
                {
                    "w": _as_real(archive[f"{network_name}_{index}_weight"]),
                    "b": _as_real(archive[f"{network_name}_{index}_bias"]),
                }
                for index in range(len(layers) - 1)
            )
    _validate_neural_parameters(params, config)
    return params, config


def _coordinate_cube_from_payload(payload):
    x = _as_real(payload["x_normalized"])
    y = _as_real(payload["y_normalized"])
    z = _as_real(payload["height_normalized"])
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    return 2.0 * jnp.stack((xx, yy, zz), axis=-1) - 1.0


def run_inversion(
    payload,
    kurucz_path=FE_I_6301_6302_LINE_LIST,
    result_path=DEFAULT_RESULT,
    checkpoint_path=DEFAULT_CHECKPOINT,
    field_config: NeuralFieldConfig = NeuralFieldConfig(),
    initial_params=None,
    pretrain_steps: int = 2000,
    pretrain_learning_rate: float = 2.0e-3,
    pretrain_tolerance: float | None = 1.0e-4,
    inversion_epochs: int = 10,
    inversion_learning_rate: float = 3.0e-4,
    training_batch_columns: int = 8,
    wavelength_batch: int = 48,
    synthesis_batch_columns: int = 16,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    prior_weight: float = 1.0e-5,
    fine_tune_base: bool = False,
    seed: int = 0,
    show_progress: bool = True,
    wavelength_parallelism: int = DEFAULT_WAVELENGTH_PARALLELISM,
    validation_columns: int = DEFAULT_VALIDATION_COLUMNS,
):
    """Pretrain and spectrally invert an already generated test cube."""

    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    validation_columns = _validate_positive_integer(
        "validation_columns", validation_columns
    )
    nx, ny, n_depth, _ = validate_test_cube(payload)
    field_config.validate()
    lines = read_kurucz(kurucz_path)
    saved_centers = np.asarray(payload["line_center_nm"], dtype=np.float64)
    expected_centers = _canonical_line_centers(lines)
    if saved_centers.shape != expected_centers.shape or not np.allclose(
        saved_centers, expected_centers, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("the inversion Kurucz data do not match the test cube")
    if str(np.asarray(payload["atomic_data_sha256"])) != atomic_data_fingerprint(lines):
        raise ValueError("the inversion atomic data do not match the test cube")
    reference = _atmosphere_from_payload(payload, "reference_")
    truth = _atmosphere_from_payload(payload, "truth_")
    if not fine_tune_base:
        validate_spatial_reachability(
            reference, truth, _as_real(field_config.spatial_scale)
        )
    coordinates = _coordinate_cube_from_payload(payload)
    flat_coordinates = coordinates.reshape((nx * ny, n_depth, 3))
    observed = _as_real(payload["observed_stokes"]).reshape((nx * ny, 4, -1))
    # Keep absolute wavelengths on the host until train_spectral_inversion
    # precenters them.  Casting this array here would lose fp32 line resolution.
    wavelengths = _host_wavelength_axis(payload["wavelength_nm"])
    dz = _as_real(payload["cell_width_m"])
    if pretrain_steps == 0 and initial_params is None:
        raise ValueError("zero pretraining steps require warm-start parameters")
    if initial_params is None:
        params = initialize_neural_field(jax.random.PRNGKey(seed), field_config)
    else:
        _validate_neural_parameters(initial_params, field_config)
        params = jax.tree.map(_as_real, initial_params)
    params, pretrain_history = pretrain_falc(
        params,
        payload["height_normalized"],
        reference,
        steps=pretrain_steps,
        learning_rate=pretrain_learning_rate,
        tolerance=pretrain_tolerance,
        show_progress=show_progress,
    )
    # Persist the expensive pretrained state before entering the long spectral
    # loop.  This archive is a warm start (Adam/RNG state is intentionally not
    # claimed to be an exact interrupted-run resume).
    save_checkpoint(checkpoint_path, params, field_config)
    params, inversion_history, validation_history = train_spectral_inversion(
        params,
        flat_coordinates,
        observed,
        wavelengths,
        saved_centers,
        dz,
        lines,
        epochs=inversion_epochs,
        batch_columns=training_batch_columns,
        wavelength_batch=wavelength_batch,
        learning_rate=inversion_learning_rate,
        stokes_weights=stokes_weights,
        spatial_scale=_as_real(field_config.spatial_scale),
        prior_weight=prior_weight,
        fine_tune_base=fine_tune_base,
        seed=seed + 1,
        show_progress=show_progress,
        wavelength_parallelism=wavelength_parallelism,
        validation_columns=validation_columns,
    )
    # Preserve the best fitted iterate before any expensive full-cube rendering
    # or archive compression can fail.
    save_checkpoint(checkpoint_path, params, field_config)
    inferred = evaluate_neural_field_cube(
        params,
        coordinates,
        _as_real(field_config.spatial_scale),
        batch_columns=synthesis_batch_columns,
        show_progress=show_progress,
    )
    _validate_atmosphere(inferred, expected_shape=(nx, ny, n_depth))
    final_stokes = synthesize_atmosphere_cube(
        lines,
        wavelengths,
        dz,
        inferred,
        batch_columns=synthesis_batch_columns,
        show_progress=show_progress,
        description="Synthesizing fitted cube",
        wavelength_parallelism=wavelength_parallelism,
    )
    observed_array = np.asarray(payload["observed_stokes"])
    synthetic_array = np.asarray(final_stokes)
    continuum = 0.5 * (observed_array[:, :, 0, 0] + observed_array[:, :, 0, -1])
    continuum = np.maximum(np.abs(continuum), np.finfo(NUMPY_REAL_DTYPE).tiny)
    residual = (synthetic_array - observed_array) / continuum[:, :, None, None]
    final_full_loss = np.mean(
        (residual * np.asarray(stokes_weights)[None, None, :, None]) ** 2
    )
    result = {
        **{key: np.asarray(value) for key, value in payload.items()},
        "synthetic_stokes": np.asarray(final_stokes),
        "pretrain_loss": pretrain_history,
        "inversion_loss_total_spectral_prior": inversion_history,
        "inversion_loss_columns": np.asarray(("total", "spectral", "prior")),
        "validation_full_wavelength_loss": validation_history,
        "best_validation_epoch": np.asarray(int(np.argmin(validation_history))),
        "best_validation_loss": np.asarray(np.min(validation_history)),
        "final_full_cube_spectral_loss": np.asarray(final_full_loss),
        "base_layers": np.asarray(field_config.base_layers, dtype=np.int64),
        "spatial_layers": np.asarray(field_config.spatial_layers, dtype=np.int64),
        "spatial_scale": np.asarray(field_config.spatial_scale),
        "stokes_weights": np.asarray(stokes_weights),
        "prior_weight": np.asarray(prior_weight),
        "wavelength_parallelism": np.asarray(wavelength_parallelism),
        "validation_columns": np.asarray(validation_columns),
        "inversion_precision": np.asarray(adora_precision.configured_precision()),
        "seed": np.asarray(seed),
        **_atmosphere_payload("inferred_", inferred),
    }
    validate_inversion_result(result)
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_path, **result)
    save_checkpoint(checkpoint_path, params, field_config)
    return params, result


def _parse_hidden_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden layers must be comma-separated integers"
        ) from exc
    if not layers or any(size <= 0 for size in layers):
        raise argparse.ArgumentTypeError("hidden-layer sizes must be positive")
    return layers


def _parse_stokes_weights(value: str) -> tuple[float, float, float, float]:
    try:
        weights = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Stokes weights must be numeric") from exc
    if len(weights) != 4 or any(
        not math.isfinite(weight) or weight <= 0.0 for weight in weights
    ):
        raise argparse.ArgumentTypeError(
            "Stokes weights must be four positive comma-separated values"
        )
    return weights


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the import-safe command-line interface."""

    parser = argparse.ArgumentParser(
        description="Generate and invert a 3-D FAL-C full-Stokes test cube."
    )
    parser.add_argument(
        "--precision",
        choices=adora_precision.SUPPORTED_PRECISIONS,
        default=adora_precision.configured_precision(),
        help=(
            "JAX compute precision (default: fp64). This is selected before "
            "physics tables are imported; fp32 is intended for consumer GPUs."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="create the synthetic test cube and stop",
    )
    mode.add_argument(
        "--invert-only",
        action="store_true",
        help="load --dataset and run pretraining/inversion without regeneration",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        help="warm-start parameters from a model archive (optimizer restarts)",
    )
    parser.add_argument("--kurucz", type=Path, default=FE_I_6301_6302_LINE_LIST)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--n-wave", type=int, default=201)
    parser.add_argument("--wavelength-padding-nm", type=float, default=0.05)
    parser.add_argument("--magnetic-field-t", type=float, default=0.05)
    parser.add_argument("--inclination-rad", type=float, default=0.8)
    parser.add_argument("--azimuth-rad", type=float, default=0.3)
    parser.add_argument(
        "--base-hidden", type=_parse_hidden_layers, default=(128, 128, 128)
    )
    parser.add_argument(
        "--spatial-hidden", type=_parse_hidden_layers, default=(96, 96, 96)
    )
    parser.add_argument("--base-frequencies", type=int, default=6)
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--pretrain-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--pretrain-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--skip-pretraining",
        action="store_true",
        help="use --checkpoint-in directly and skip additional FAL-C pretraining",
    )
    parser.add_argument("--inversion-epochs", type=int, default=10)
    parser.add_argument("--inversion-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--training-batch-columns", type=int, default=8)
    parser.add_argument(
        "--validation-columns",
        type=int,
        default=DEFAULT_VALIDATION_COLUMNS,
        help="fixed columns used for full-spectrum validation each epoch",
    )
    parser.add_argument("--wavelength-batch", type=int, default=48)
    parser.add_argument(
        "--wavelength-parallelism",
        type=int,
        default=DEFAULT_WAVELENGTH_PARALLELISM,
        help=(
            "wavelengths evaluated concurrently inside each bounded-memory "
            "radiative-transfer chunk; lower this value if accelerator memory "
            "is constrained"
        ),
    )
    parser.add_argument("--synthesis-batch-columns", type=int, default=16)
    parser.add_argument(
        "--stokes-weights",
        type=_parse_stokes_weights,
        default=(1.0, 5.0, 5.0, 2.0),
        metavar="I,Q,U,V",
    )
    parser.add_argument("--prior-weight", type=float, default=1.0e-5)
    parser.add_argument("--fine-tune-base", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv=None):
    """Run generation, inversion, or the complete requested workflow."""

    args = build_argument_parser().parse_args(argv)
    active_precision = adora_precision.configured_precision()
    if args.precision != active_precision:
        raise ValueError(
            f"requested --precision {args.precision}, but this Python process "
            f"already imported Adora in {active_precision}. Start a fresh process "
            f"with ADORA_PRECISION={args.precision}, or invoke the command-line "
            "entry point directly."
        )
    show_progress = not args.no_progress
    initial_params = None
    config = NeuralFieldConfig(
        base_hidden=args.base_hidden,
        spatial_hidden=args.spatial_hidden,
        base_frequencies=args.base_frequencies,
    )
    config.validate()
    if not args.generate_only:
        if args.skip_pretraining and args.checkpoint_in is None:
            raise ValueError("--skip-pretraining requires --checkpoint-in")
        if args.checkpoint_in is not None:
            # Load before generation so a missing/corrupt warm start cannot
            # waste the expensive default truth-cube synthesis.
            initial_params, config = load_checkpoint(args.checkpoint_in)
    if args.invert_only:
        payload = load_test_cube(args.dataset)
    else:
        payload = generate_test_cube(
            output_path=args.dataset,
            kurucz_path=args.kurucz,
            nx=args.nx,
            ny=args.ny,
            n_wave=args.n_wave,
            wavelength_padding_nm=args.wavelength_padding_nm,
            synthesis_batch_columns=args.synthesis_batch_columns,
            magnetic_field_t=args.magnetic_field_t,
            inclination_rad=args.inclination_rad,
            azimuth_rad=args.azimuth_rad,
            show_progress=show_progress,
            wavelength_parallelism=args.wavelength_parallelism,
        )
    if args.generate_only:
        return None
    run_inversion(
        payload,
        kurucz_path=args.kurucz,
        result_path=args.result,
        checkpoint_path=args.checkpoint,
        field_config=config,
        initial_params=initial_params,
        pretrain_steps=0 if args.skip_pretraining else args.pretrain_steps,
        pretrain_learning_rate=args.pretrain_learning_rate,
        pretrain_tolerance=(None if args.skip_pretraining else args.pretrain_tolerance),
        inversion_epochs=args.inversion_epochs,
        inversion_learning_rate=args.inversion_learning_rate,
        training_batch_columns=args.training_batch_columns,
        wavelength_batch=args.wavelength_batch,
        synthesis_batch_columns=args.synthesis_batch_columns,
        stokes_weights=args.stokes_weights,
        prior_weight=args.prior_weight,
        fine_tune_base=args.fine_tune_base,
        seed=args.seed,
        show_progress=show_progress,
        wavelength_parallelism=args.wavelength_parallelism,
        validation_columns=args.validation_columns,
    )
    return None


if __name__ == "__main__":
    main()
