"""Three-dimensional neural-field inversion of polarized Fe I spectra.

The neural atmosphere is a shared implicit representation of ``(x, y, z)``.
Radiative transfer is evaluated independently along each vertical column with
Adora's LTE polarized formal solver, so the forward model is commonly called
1.5-D rather than horizontally coupled 3-D radiative transfer.

Running this module without mode flags performs the complete workflow:

1. Broadcast FAL-C over a 50 x 50 horizontal grid, apply a small smooth 3-D
   perturbation, and synthesize full-Stokes observations for every Fe I line
   in the selected Kurucz file.
2. Initialize a shared spatial network with exactly zero output, reproducing
   the stored FAL-C reference (including its magnetic seed) without pretraining.
3. Fit bounded log10 ratios and signed offsets relative to that fixed reference.

The default problem is intentionally substantial.  Use small ``--nx``,
``--ny``, and ``--n-wave`` values for a quick smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

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
CHECKPOINT_SCHEMA_VERSION = 2
PARAMETERIZATION = "fixed_reference_log10_ratios_v2"
SPECTRA_SCHEMA_VERSION = 1
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
CORRECTION_CHANNEL_NAMES = (
    "log10_temperature_ratio",
    "log10_ne_ratio",
    "log10_nhtot_ratio",
    "los_velocity_offset_over_20_km_s",
    "log10_vturb_ratio",
    "log10_magnetic_field_ratio",
    "inclination_offset_over_pi",
    "azimuth_offset_over_half_pi",
)

DEFAULT_DATASET = Path("data") / "pinn_3d_test_cube.npz"
DEFAULT_RESULT = Path("data") / "pinn_3d_inversion.npz"
DEFAULT_CHECKPOINT = Path("data") / "pinn_3d_checkpoint.npz"
DEFAULT_SPECTRA_OUTPUT = Path("data") / "pinn_3d_spectra.npz"
DEFAULT_WAVELENGTH_PARALLELISM = 48
DEFAULT_VALIDATION_COLUMNS = 8
DEFAULT_INVERSION_INITIAL_LEARNING_RATE = 1.0e-3
DEFAULT_INVERSION_FINAL_LEARNING_RATE = 1.0e-5
DEFAULT_WANDB_PROJECT = "adora-pinn-3d"
DEFAULT_WANDB_EVALUATION_EVERY = 50

# Canonical host values are kept separately from their selected-precision JAX
# forms.  Checkpoint compatibility is therefore independent of whether a run
# happens in fp32 or fp64.
_SPATIAL_SCALE_VALUES = (1.0,) * 8
_VELOCITY_SCALE = 2.0e4  # m/s per unit signed correction
_INCLINATION_SCALE = math.pi
_AZIMUTH_SCALE = 0.5 * math.pi

# tanh outputs are always in [-1, 1]. Scales multiply these dimensionless
# outputs: dex for positive fields, 20 km/s for velocity, pi and pi/2 for angles.
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

    spatial_hidden: tuple[int, ...] = (96, 96, 96)
    spatial_scale: tuple[float, ...] = _SPATIAL_SCALE_VALUES

    def validate(self) -> None:
        if not self.spatial_hidden:
            raise ValueError("the spatial network needs hidden layers")
        if any(not isinstance(size, int) or size <= 0 for size in self.spatial_hidden):
            raise ValueError("all hidden-layer sizes must be positive")
        if len(self.spatial_scale) != 8:
            raise ValueError("spatial_scale must contain eight values")
        if not all(
            math.isfinite(value) and value > 0.0 for value in self.spatial_scale
        ):
            raise ValueError("spatial correction scales must be positive and finite")

    @property
    def spatial_layers(self) -> tuple[int, ...]:
        return (3, *self.spatial_hidden, 8)


class WandbRunLogger:
    """Small optional adapter around a W&B run.

    The W&B package is imported only when logging is explicitly requested, so
    the core synthesis and inversion APIs retain no mandatory tracking
    dependency. Device metrics are already materialized at epoch boundaries;
    logging them here does not add accelerator synchronization points.
    """

    def __init__(
        self,
        wandb_module,
        run,
        *,
        evaluation_every: int = DEFAULT_WANDB_EVALUATION_EVERY,
        log_artifacts: bool = False,
    ):
        self.wandb = wandb_module
        self.run = run

        self.evaluation_every = _validate_positive_integer(
            "wandb evaluation interval", evaluation_every
        )
        self.log_artifacts = bool(log_artifacts)

        self.run.define_metric("inversion/epoch")
        for metric in (
            "inversion/train_total_loss",
            "inversion/train_spectral_loss",
            "inversion/train_prior",
            "inversion/learning_rate",
            "inversion/validation_full_wavelength_loss",
            "inversion/best_validation_loss",
        ):
            self.run.define_metric(
                metric,
                step_metric="inversion/epoch",
                summary="min",
            )
        self.run.define_metric(
            "inversion/evaluation_figure",
            step_metric="inversion/epoch",
        )

    def log_inversion_metrics(self, metrics: dict[str, float | int]) -> None:
        """Log one already-synchronized inversion epoch record."""

        self.run.log(dict(metrics))

    def log_evaluation_figure(self, epoch: int, figure) -> None:
        """Upload a current spectra/atmosphere comparison figure."""

        self.run.log(
            {
                "inversion/epoch": int(epoch),
                "inversion/evaluation_figure": self.wandb.Image(figure),
            }
        )

    def update_config(self, values: dict) -> None:
        self.run.config.update(_wandb_serializable(values))

    def update_summary(self, values: dict) -> None:
        for key, value in _wandb_serializable(values).items():
            self.run.summary[key] = value

    def track_file(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        role: str,
    ) -> None:
        """Optionally upload a local input or output file as a W&B Artifact."""

        if not self.log_artifacts:
            return
        if role not in {"input", "output"}:
            raise ValueError("W&B artifact role must be input or output")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"W&B artifact file does not exist: {path}")
        artifact = self.wandb.Artifact(
            name=f"adora-pinn-{role}-{artifact_type}-{self.run.id}",
            type=artifact_type,
            metadata={"source_path": str(path.resolve()), "role": role},
        )
        artifact.add_file(str(path.resolve()), name=path.name)
        if role == "input":
            self.run.use_artifact(artifact)
        else:
            self.run.log_artifact(artifact)

    def finish(self, exit_code: int) -> None:
        self.run.finish(exit_code=exit_code)


def _validate_positive_integer(name: str, value: int) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_learning_rate_range(initial: float, final: float) -> None:
    """Validate a positive, monotonically decreasing learning-rate range."""

    if not math.isfinite(initial) or initial <= 0.0:
        raise ValueError("initial learning_rate must be positive and finite")
    if not math.isfinite(final) or final <= 0.0:
        raise ValueError("final_learning_rate must be positive and finite")
    if final > initial:
        raise ValueError("final_learning_rate must not exceed learning_rate")


def sin_squared_learning_rate(
    step,
    total_steps: int,
    initial_learning_rate: float = DEFAULT_INVERSION_INITIAL_LEARNING_RATE,
    final_learning_rate: float = DEFAULT_INVERSION_FINAL_LEARNING_RATE,
):
    """Return a clipped sin-squared decay from ``initial`` to ``final``.

    Optimizer update zero uses the initial rate and update
    ``total_steps - 1`` uses the final rate. A one-update run necessarily uses
    the initial rate because there is no interval over which to decay.
    """

    if not isinstance(total_steps, int):
        raise TypeError("total_steps must be an integer")
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative")
    _validate_learning_rate_range(initial_learning_rate, final_learning_rate)
    if total_steps <= 1:
        return jnp.asarray(initial_learning_rate, dtype=REAL_DTYPE)
    progress = jnp.clip(
        jnp.asarray(step, dtype=REAL_DTYPE) / (total_steps - 1),
        0.0,
        1.0,
    )
    amplitude = initial_learning_rate - final_learning_rate
    return (
        final_learning_rate + amplitude * jnp.sin(0.5 * jnp.pi * (1.0 - progress)) ** 2
    )


def sin_squared_learning_rate_schedule(
    total_steps: int,
    initial_learning_rate: float = DEFAULT_INVERSION_INITIAL_LEARNING_RATE,
    final_learning_rate: float = DEFAULT_INVERSION_FINAL_LEARNING_RATE,
):
    """Build the JAX-compatible learning-rate callable consumed by Optax."""

    # Validate eagerly so bad CLI values fail before an expensive compilation.
    sin_squared_learning_rate(
        0,
        total_steps,
        initial_learning_rate,
        final_learning_rate,
    )

    def schedule(step):
        return sin_squared_learning_rate(
            step,
            total_steps,
            initial_learning_rate,
            final_learning_rate,
        )

    return schedule


def _wandb_serializable(value):
    """Convert CLI and NumPy values into W&B configuration primitives."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _wandb_serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wandb_serializable(item) for item in value]
    return value


def _initialize_wandb(args, field_config: NeuralFieldConfig, wandb_module=None):
    """Lazily initialize W&B for a parsed CLI invocation."""

    if not args.wandb_project.strip():
        raise ValueError("--wandb-project must be non-empty")

    _validate_positive_integer("wandb evaluation interval", args.wandb_evaluation_every)
    if wandb_module is None:
        try:
            wandb_module = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was requested but wandb is not installed; install "
                "it with `python -m pip install -e '.[wandb]'` or "
                "`python -m pip install wandb`"
            ) from exc
    if args.wandb_dir is not None:
        args.wandb_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_login:
        login_succeeded = wandb_module.login()
        if login_succeeded is False:
            raise RuntimeError("wandb.login() did not authenticate successfully")

    if args.generate_only:
        job_type = "generate"
    elif args.invert_only:
        job_type = "invert"
    else:
        job_type = "generate-and-invert"
    config = _wandb_serializable(vars(args))
    config.update(
        {
            "workflow": job_type,
            "parameterization": PARAMETERIZATION,
            "effective_spatial_hidden": list(field_config.spatial_hidden),
            "effective_spatial_scale": list(field_config.spatial_scale),
        }
    )
    run = wandb_module.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags or None,
        mode=args.wandb_mode,
        dir=None if args.wandb_dir is None else str(args.wandb_dir.resolve()),
        job_type=job_type,
        config=config,
    )
    if run is None:
        raise RuntimeError("wandb.init() did not return a run")
    try:
        return WandbRunLogger(
            wandb_module,
            run,
            evaluation_every=args.wandb_evaluation_every,
            log_artifacts=args.wandb_log_artifacts,
        )
    except BaseException:
        try:
            run.finish(exit_code=1)
        except Exception:
            pass
        raise


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
    if not math.isfinite(magnetic_field_t) or magnetic_field_t <= 0.0:
        raise ValueError("magnetic_field_t must be positive for a log10 ratio")
    if not math.isfinite(inclination_rad) or not 0.0 <= inclination_rad <= math.pi:
        raise ValueError("inclination_rad must lie in [0, pi]")
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
    if getattr(
        adata, "canonical_sha256", None
    ) is not None and runtime_digest == getattr(adata, "runtime_sha256", None):
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
    lower_boundary = (
        jnp.zeros(4, dtype=eta.dtype).at[0].set(planck(absolute_wave, temperature[0]))
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
    flat = Atmosphere(*(_as_real(field).reshape((-1, n_depth)) for field in atmosphere))
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
    return jnp.asarray(output.reshape((nx, ny, 4, n_wave)), dtype=REAL_DTYPE)


def validate_neural_transform_domain(atmosphere: Atmosphere) -> None:
    """Require finite physical fields and positive multiplicative quantities."""

    _validate_atmosphere(atmosphere)


def atmosphere_to_corrections(atmosphere: Atmosphere, reference: Atmosphere):
    """Encode log10(q/q_ref) and scaled additive velocity/angle changes.

    The reference may be one depth profile or a broadcast-compatible sampled
    atmosphere. Azimuth differences use the shortest pi-periodic displacement.
    """

    return jnp.stack(
        tuple(
            jnp.log10(_as_real(value) / _as_real(baseline))
            if index in (0, 1, 2, 4, 5)
            else (
                _wrap_azimuth(_as_real(value) - _as_real(baseline)) / _AZIMUTH_SCALE
                if index == 7
                else (_as_real(value) - _as_real(baseline))
                / (_VELOCITY_SCALE if index == 3 else _INCLINATION_SCALE)
            )
            for index, (value, baseline) in enumerate(zip(atmosphere, reference))
        ),
        axis=-1,
    )


def corrections_to_atmosphere(corrections, reference: Atmosphere) -> Atmosphere:
    """Perturb the exact reference at each height using physical corrections.

    Positive quantities multiply by 10**correction. Signed velocity and angles
    use additive offsets, which also work at zero reference velocity/azimuth.
    Inclinations outside [0, pi] are reflected to the equivalent inclination;
    the polarized solver is invariant to this reflection (sin² gamma, cos gamma).
    The usual [0, pi] branch is unchanged, including at zero correction.
    """

    corrections = _as_real(corrections)
    if corrections.shape[-1] != 8:
        raise ValueError("atmosphere corrections must have eight channels")
    reference = jax.tree.map(jax.lax.stop_gradient, reference)
    fields = [
        _as_real(baseline) * jnp.power(10.0, corrections[..., index])
        if index in (0, 1, 2, 4, 5)
        else _as_real(baseline)
        + corrections[..., index]
        * (
            _VELOCITY_SCALE
            if index == 3
            else _INCLINATION_SCALE
            if index == 6
            else _AZIMUTH_SCALE
        )
        for index, baseline in enumerate(reference)
    ]
    inclination = fields[6]
    fields[6] = jnp.where(
        (inclination >= 0.0) & (inclination <= jnp.pi),
        inclination,
        jnp.pi - jnp.abs(jnp.pi - jnp.mod(inclination, 2.0 * jnp.pi)),
    )
    return Atmosphere(*fields)


def validate_spatial_reachability(
    reference: Atmosphere,
    truth: Atmosphere,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
) -> None:
    """Ensure the truth fits within the bounded reference-relative outputs."""

    validate_neural_transform_domain(reference)
    validate_neural_transform_domain(truth)
    if (
        reference.temperature.ndim != 1
        or truth.temperature.shape[-1:] != (reference.temperature.size,)
        or truth.temperature.ndim != 3
    ):
        raise ValueError("truth and reference depth/channel dimensions do not match")
    spatial_scale = _as_real(spatial_scale)
    if spatial_scale.shape != (8,) or not bool(
        jnp.all(jnp.isfinite(spatial_scale) & (spatial_scale > 0.0))
    ):
        raise ValueError("spatial_scale must contain eight positive finite values")
    required = jnp.max(
        jnp.abs(atmosphere_to_corrections(truth, reference)), axis=(0, 1, 2)
    )
    if not bool(jnp.all(required < spatial_scale)):
        failing = [
            CORRECTION_CHANNEL_NAMES[index]
            for index in np.flatnonzero(np.asarray(required >= spatial_scale))
        ]
        raise ValueError(
            "truth perturbation exceeds the reference-relative correction range for: "
            + ", ".join(failing)
            + "; increase --spatial-scale"
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
        ) * jnp.sqrt(2.0 / (n_in + n_out))
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


def _validate_reference(reference: Atmosphere, height_normalized) -> None:
    z = np.asarray(height_normalized)
    if (
        z.ndim != 1
        or z.size < 2
        or not np.all(np.isfinite(z))
        or np.any(np.diff(z) <= 0.0)
        or z[0] != 0.0
        or z[-1] != 1.0
    ):
        raise ValueError("reference heights must increase from 0 to 1")
    _validate_atmosphere(reference, expected_shape=z.shape)


def initialize_neural_field(
    key,
    reference: Atmosphere,
    height_normalized,
    config: NeuralFieldConfig = NeuralFieldConfig(),
):
    """Bind a fixed reference and initialize exactly zero spatial outputs.

    Hidden layers retain random weights; only the final layer is zero. This
    gives an exact identity atmosphere without a pretraining optimizer pass.
    Reference profiles are data, never trainable parameters.
    """

    config.validate()
    _validate_reference(reference, height_normalized)
    return {
        "reference": {
            "height_normalized": _as_real(height_normalized),
            "atmosphere": Atmosphere(*(_as_real(field) for field in reference)),
        },
        "spatial": init_mlp(config.spatial_layers, key, zero_output=True),
    }


def reference_at_coordinates(params, coordinates) -> Atmosphere:
    """Interpolate fixed reference profiles at the requested normalized heights.

    At the stored depth grid this returns the original reference exactly. The
    physical profiles are linearly interpolated for intermediate heights.
    """

    coordinates = _as_real(coordinates)
    if coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have a final (x, y, z) axis")
    reference = jax.tree.map(jax.lax.stop_gradient, params["reference"])
    z = 2.0 * reference["height_normalized"] - 1.0
    return Atmosphere(
        *(
            jnp.interp(coordinates[..., 2], z, field)
            for field in reference["atmosphere"]
        )
    )


def neural_field_output(params, coordinates):
    """Return eight dimensionless correction channels bounded to [-1, 1]."""

    return jnp.tanh(apply_mlp(params["spatial"], coordinates))


def evaluate_neural_field(
    params,
    coordinates,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
) -> Atmosphere:
    """Evaluate the fixed reference perturbed by the current neural outputs."""

    reference = reference_at_coordinates(params, coordinates)
    corrections = _as_real(spatial_scale) * neural_field_output(params, coordinates)
    return corrections_to_atmosphere(corrections, reference)


def _evaluate_field_and_output(params, coordinates, spatial_scale):
    output = neural_field_output(params, coordinates)
    reference = reference_at_coordinates(params, coordinates)
    return corrections_to_atmosphere(spatial_scale * output, reference), output


_EVALUATE_NEURAL_FIELD_JIT = jax.jit(_evaluate_field_and_output)


def evaluate_neural_field_cube(
    params,
    coordinates,
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    batch_columns: int = 16,
    show_progress: bool = True,
    return_corrections: bool = False,
):
    """Evaluate a full ``(x,y,z)`` neural cube in bounded-memory chunks."""

    batch_columns = _validate_positive_integer("batch_columns", batch_columns)
    coordinates = _as_real(coordinates)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (nx, ny, depth, 3)")
    nx, ny, n_depth, _ = coordinates.shape
    n_columns = nx * ny
    flat_coordinates = coordinates.reshape((n_columns, n_depth, 3))
    outputs = [
        np.empty((n_columns, n_depth), dtype=NUMPY_REAL_DTYPE) for _ in FIELD_NAMES
    ]
    correction_output = (
        np.empty((n_columns, n_depth, 8), dtype=NUMPY_REAL_DTYPE)
        if return_corrections
        else None
    )
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
        atmosphere, normalized_output = _EVALUATE_NEURAL_FIELD_JIT(
            params, batch, _as_real(spatial_scale)
        )
        jax.block_until_ready(atmosphere.temperature)
        if return_corrections:
            correction_output[start:stop] = np.asarray(normalized_output[:count])
        for output, field in zip(outputs, atmosphere):
            output[start:stop] = np.asarray(field[:count])
    atmosphere = Atmosphere(
        *(
            jnp.asarray(output.reshape((nx, ny, n_depth)), dtype=REAL_DTYPE)
            for output in outputs
        )
    )
    if return_corrections:
        return atmosphere, correction_output.reshape((nx, ny, n_depth, 8))
    return atmosphere


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

    normalized_output = neural_field_output(params, coordinates)
    reference = reference_at_coordinates(params, coordinates)
    return _relative_spectral_loss_impl(
        reference,
        normalized_output,
        spatial_scale,
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

    normalized_output = neural_field_output(params, coordinates)
    reference = reference_at_coordinates(params, coordinates)
    return _relative_spectral_loss_impl(
        reference,
        normalized_output,
        spatial_scale,
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


def _relative_spectral_loss_impl(
    reference,
    normalized_output,
    spatial_scale,
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
    atmosphere = corrections_to_atmosphere(
        _as_real(spatial_scale) * normalized_output, reference
    )
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
    per_column_prior = jnp.mean(normalized_output**2, axis=(1, 2))
    if column_mask is None:
        prior = jnp.mean(per_column_prior)
    else:
        prior = jnp.sum(per_column_prior * column_mask) / jnp.maximum(
            jnp.sum(column_mask), 1.0
        )
    return spectral_loss + prior_weight * prior, (spectral_loss, prior)


def reference_spectral_loss(
    spatial_params,
    reference,
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

    normalized_output = jnp.tanh(apply_mlp(spatial_params, coordinates))
    return _relative_spectral_loss_impl(
        reference,
        normalized_output,
        spatial_scale,
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


def reference_spectral_loss_offset(
    spatial_params,
    reference,
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
    """Fixed-reference loss using precision-safe centered wavelengths."""

    normalized_output = jnp.tanh(apply_mlp(spatial_params, coordinates))
    return _relative_spectral_loss_impl(
        reference,
        normalized_output,
        spatial_scale,
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
    learning_rate: float = DEFAULT_INVERSION_INITIAL_LEARNING_RATE,
    final_learning_rate: float = DEFAULT_INVERSION_FINAL_LEARNING_RATE,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    spatial_scale=DEFAULT_SPATIAL_SCALE,
    prior_weight: float = 1.0e-5,
    require_improvement: bool = True,
    seed: int = 0,
    show_progress: bool = True,
    wavelength_parallelism: int = DEFAULT_WAVELENGTH_PARALLELISM,
    validation_columns: int = DEFAULT_VALIDATION_COLUMNS,
    metrics_callback: Callable[[dict[str, float | int]], None] | None = None,
    evaluation_callback: Callable[[int, dict], None] | None = None,
    evaluation_every: int = DEFAULT_WANDB_EVALUATION_EVERY,
):
    """Fit the shared neural field using shuffled complete-column batches.

    The Adam learning rate follows a sin-squared decay at optimizer-update
    granularity. Optional evaluations run at epoch zero, every
    ``evaluation_every`` epochs, and at the final epoch.
    """

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
    _validate_learning_rate_range(learning_rate, final_learning_rate)
    if evaluation_callback is not None:
        evaluation_every = _validate_positive_integer(
            "evaluation_every", evaluation_every
        )
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
    n_columns = coordinates.shape[0]
    updates_per_epoch = math.ceil(n_columns / batch_columns)
    total_optimizer_steps = epochs * updates_per_epoch
    learning_rate_schedule = sin_squared_learning_rate_schedule(
        total_optimizer_steps,
        learning_rate,
        final_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate_schedule),
    )

    trainable = params["spatial"]
    reference = reference_at_coordinates(params, coordinates[0])

    def batch_loss(
        candidate,
        batch_coordinates,
        batch_wavelengths,
        batch_observed,
        batch_continuum,
        batch_mask,
    ):
        return reference_spectral_loss_offset(
            candidate,
            reference,
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
        return {"reference": params["reference"], "spatial": candidate}

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
    validation_history[0] = float(compiled_validation(trainable, *validation_arguments))
    best_validation = validation_history[0]
    best_trainable = trainable
    if metrics_callback is not None:
        metrics_callback(
            {
                "inversion/epoch": 0,
                "inversion/learning_rate": float(np.asarray(learning_rate_schedule(0))),
                "inversion/validation_full_wavelength_loss": float(
                    validation_history[0]
                ),
                "inversion/best_validation_loss": float(best_validation),
            }
        )
    if evaluation_callback is not None:
        evaluation_callback(0, assemble(trainable))
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
        completed_update = min(
            (epoch + 1) * updates_per_epoch - 1,
            max(total_optimizer_steps - 1, 0),
        )
        current_learning_rate = float(
            np.asarray(learning_rate_schedule(completed_update))
        )
        if validation_history[epoch + 1] < best_validation:
            best_validation = validation_history[epoch + 1]
            best_trainable = trainable
        epoch_iterator.set_postfix(
            loss=f"{history[epoch, 0]:.3e}",
            spectral=f"{history[epoch, 1]:.3e}",
            validation=f"{validation_history[epoch + 1]:.3e}",
            lr=f"{current_learning_rate:.2e}",
        )
        if metrics_callback is not None:
            metrics_callback(
                {
                    "inversion/epoch": int(epoch + 1),
                    "inversion/train_total_loss": float(history[epoch, 0]),
                    "inversion/train_spectral_loss": float(history[epoch, 1]),
                    "inversion/train_prior": float(history[epoch, 2]),
                    "inversion/learning_rate": current_learning_rate,
                    "inversion/validation_full_wavelength_loss": float(
                        validation_history[epoch + 1]
                    ),
                    "inversion/best_validation_loss": float(best_validation),
                }
            )
        completed_epoch = epoch + 1
        if evaluation_callback is not None and (
            completed_epoch % evaluation_every == 0 or completed_epoch == epochs
        ):
            evaluation_callback(completed_epoch, assemble(trainable))
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


def validate_spectra_output(payload) -> tuple[int, int, int]:
    """Validate the compact input/output spectra interchange archive."""

    required = {
        "schema_version",
        "stokes_labels",
        "x_normalized",
        "y_normalized",
        "wavelength_nm",
        "input_stokes",
        "output_stokes",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            "spectra output is missing keys: " + ", ".join(sorted(missing))
        )
    if int(np.asarray(payload["schema_version"])) != SPECTRA_SCHEMA_VERSION:
        raise ValueError("unsupported spectra output schema version")
    if tuple(str(value) for value in payload["stokes_labels"]) != STOKES_LABELS:
        raise ValueError("stokes_labels must be ordered as I, Q, U, V")
    x = np.asarray(payload["x_normalized"], dtype=float)
    y = np.asarray(payload["y_normalized"], dtype=float)
    wavelength = np.asarray(payload["wavelength_nm"], dtype=float)
    for name, axis in (("x_normalized", x), ("y_normalized", y)):
        if axis.ndim != 1 or axis.size == 0 or np.any(~np.isfinite(axis)):
            raise ValueError(f"{name} must be a non-empty finite axis")
        if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{name} must be strictly increasing")
    if (
        wavelength.ndim != 1
        or wavelength.size == 0
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or (wavelength.size > 1 and np.any(np.diff(wavelength) <= 0.0))
    ):
        raise ValueError("wavelength_nm must be a positive increasing finite axis")
    expected_shape = (x.size, y.size, 4, wavelength.size)
    for key in ("input_stokes", "output_stokes"):
        spectra = np.asarray(payload[key])
        if spectra.shape != expected_shape or np.any(~np.isfinite(spectra)):
            raise ValueError(f"{key} must have finite shape {expected_shape}")
    return x.size, y.size, wavelength.size


def save_spectra_output(
    path: str | Path,
    *,
    x_normalized,
    y_normalized,
    wavelength_nm,
    input_stokes,
    output_stokes,
) -> dict[str, np.ndarray]:
    """Write input and fitted Stokes cubes without atmosphere/checkpoint data."""

    payload = {
        "schema_version": np.asarray(SPECTRA_SCHEMA_VERSION),
        "stokes_labels": np.asarray(STOKES_LABELS),
        "x_normalized": np.asarray(x_normalized),
        "y_normalized": np.asarray(y_normalized),
        "wavelength_nm": np.asarray(wavelength_nm, dtype=np.float64),
        "input_stokes": np.asarray(input_stokes),
        "output_stokes": np.asarray(output_stokes),
        "intensity_unit": np.asarray("kW m-2 nm-1 sr-1"),
        "wavelength_unit": np.asarray("nm"),
    }
    validate_spectra_output(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def load_spectra_output(path: str | Path = DEFAULT_SPECTRA_OUTPUT):
    """Load and validate a compact spectra archive."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"spectra output archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    validate_spectra_output(payload)
    return payload


def validate_inversion_result(payload) -> tuple[int, int, int, int]:
    """Validate a self-contained fitted-cube result archive."""

    nx, ny, n_depth, n_wave = validate_test_cube(payload)
    _validate_precision_metadata(payload, "inversion_precision")
    required = {
        "synthetic_stokes",
        "inversion_loss_total_spectral_prior",
        "inversion_loss_columns",
        "validation_full_wavelength_loss",
        "best_validation_epoch",
        "best_validation_loss",
        "final_full_cube_spectral_loss",
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
    _validate_correction_metadata(payload)
    if "inferred_normalized_corrections" not in payload:
        raise ValueError("inversion result is missing inferred_normalized_corrections")
    corrections = np.asarray(payload["inferred_normalized_corrections"])
    if corrections.shape != (nx, ny, n_depth, 8) or not np.all(
        np.isfinite(corrections) & (np.abs(corrections) <= 1.0)
    ):
        raise ValueError(
            "normalized corrections must be finite (nx, ny, depth, 8) values in [-1, 1]"
        )
    synthetic = np.asarray(payload["synthetic_stokes"])
    if synthetic.shape != (nx, ny, 4, n_wave) or np.any(~np.isfinite(synthetic)):
        raise ValueError("synthetic_stokes has an invalid shape or values")
    inferred = _atmosphere_from_payload(payload, "inferred_")
    _validate_atmosphere(inferred, expected_shape=(nx, ny, n_depth))
    inversion_history = np.asarray(payload["inversion_loss_total_spectral_prior"])
    validation_history = np.asarray(payload["validation_full_wavelength_loss"])
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
    schedule_keys = {
        "inversion_learning_rate_by_epoch",
        "inversion_learning_rate_schedule",
        "inversion_initial_learning_rate",
        "inversion_final_learning_rate",
    }
    present_schedule_keys = schedule_keys.intersection(payload)
    if present_schedule_keys and present_schedule_keys != schedule_keys:
        missing_schedule_keys = schedule_keys.difference(payload)
        raise ValueError(
            "inversion learning-rate metadata is incomplete: "
            + ", ".join(sorted(missing_schedule_keys))
        )
    if present_schedule_keys:
        learning_rates = np.asarray(
            payload["inversion_learning_rate_by_epoch"], dtype=float
        )
        initial_learning_rate = float(
            np.asarray(payload["inversion_initial_learning_rate"])
        )
        final_learning_rate = float(
            np.asarray(payload["inversion_final_learning_rate"])
        )
        _validate_learning_rate_range(initial_learning_rate, final_learning_rate)
        if str(np.asarray(payload["inversion_learning_rate_schedule"])) != (
            "sin_squared"
        ):
            raise ValueError("unsupported inversion learning-rate schedule")
        if (
            learning_rates.shape != validation_history.shape
            or np.any(~np.isfinite(learning_rates))
            or np.any(np.diff(learning_rates) > initial_learning_rate * 1.0e-6)
            or not np.isclose(
                learning_rates[0], initial_learning_rate, rtol=1.0e-6, atol=0.0
            )
            or np.any(learning_rates < final_learning_rate * (1.0 - 1.0e-6))
            or np.any(learning_rates > initial_learning_rate * (1.0 + 1.0e-6))
        ):
            raise ValueError("inversion learning-rate history is inconsistent")
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
    spatial_scale=DEFAULT_SPATIAL_SCALE,
):
    """Create, perturb, synthesize, and save the requested FAL-C test cube."""

    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    perturbation.validate()
    requested = np.asarray(
        (
            perturbation.log_temperature / math.log(10.0),
            perturbation.log_ne / math.log(10.0),
            perturbation.log_nhtot / math.log(10.0),
            perturbation.velocity_m_s / _VELOCITY_SCALE,
            perturbation.log_vturb / math.log(10.0),
            perturbation.log_b / math.log(10.0),
            perturbation.inclination_rad / _INCLINATION_SCALE,
            perturbation.azimuth_rad / _AZIMUTH_SCALE,
        )
    )
    if np.any(np.abs(requested) >= np.asarray(spatial_scale)):
        raise ValueError(
            "test perturbation exceeds the reference-relative correction bounds"
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
    truth, envelope = perturb_falc_cube(
        reference_cube, x, y, z_normalized, perturbation
    )
    validate_neural_transform_domain(reference)
    validate_neural_transform_domain(truth)
    validate_spatial_reachability(reference, truth, spatial_scale)
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
    if set(params) != {"reference", "spatial"}:
        raise ValueError(
            "parameters must contain a fixed reference and spatial network"
        )
    reference = params["reference"]
    _validate_reference(reference["atmosphere"], reference["height_normalized"])
    layer_sizes = config.spatial_layers
    if len(params["spatial"]) != len(layer_sizes) - 1:
        raise ValueError("spatial parameters do not match the architecture")
    for index, (layer, n_in, n_out) in enumerate(
        zip(params["spatial"], layer_sizes[:-1], layer_sizes[1:])
    ):
        weight, bias = np.asarray(layer["w"]), np.asarray(layer["b"])
        if weight.shape != (n_in, n_out) or bias.shape != (n_out,):
            raise ValueError(f"spatial layer {index} has incompatible shapes")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError(f"spatial layer {index} contains non-finite parameters")


def _correction_metadata():
    return {
        "parameterization": np.asarray(PARAMETERIZATION),
        "correction_channel_names": np.asarray(CORRECTION_CHANNEL_NAMES),
        "velocity_offset_scale_m_s": np.asarray(_VELOCITY_SCALE, dtype=np.float64),
        "inclination_offset_scale_rad": np.asarray(
            _INCLINATION_SCALE, dtype=np.float64
        ),
        "azimuth_offset_scale_rad": np.asarray(_AZIMUTH_SCALE, dtype=np.float64),
    }


def _validate_correction_metadata(payload):
    for name, expected in _correction_metadata().items():
        if name not in payload or not np.array_equal(
            np.asarray(payload[name]), expected
        ):
            raise ValueError(
                f"{name} does not match the reference-relative parameterization"
            )


def save_checkpoint(path, params, config: NeuralFieldConfig) -> None:
    """Save the spatial MLP, its scales, and the exact fixed reference."""

    _validate_neural_parameters(params, config)
    payload = {
        "checkpoint_schema_version": np.asarray(CHECKPOINT_SCHEMA_VERSION),
        "checkpoint_precision": np.asarray(adora_precision.configured_precision()),
        "spatial_layers": np.asarray(config.spatial_layers, dtype=np.int64),
        "spatial_scale": np.asarray(config.spatial_scale, dtype=np.float64),
        "reference_height_normalized": np.asarray(
            params["reference"]["height_normalized"]
        ),
        **_atmosphere_payload("reference_", params["reference"]["atmosphere"]),
        **_correction_metadata(),
    }
    for index, layer in enumerate(params["spatial"]):
        payload[f"spatial_{index}_weight"] = np.asarray(layer["w"])
        payload[f"spatial_{index}_bias"] = np.asarray(layer["b"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_checkpoint(path):
    """Load a reference-relative model; old learned-base models need a fresh run."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if (
            int(np.asarray(archive["checkpoint_schema_version"]))
            != CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported checkpoint schema version: learned-base checkpoints "
                "cannot be used by the fixed-reference log10 model; start without "
                "--checkpoint-in or regenerate the initial checkpoint"
            )
        _validate_correction_metadata(archive)
        _validate_precision_metadata(archive, "checkpoint_precision")
        layers = tuple(int(value) for value in archive["spatial_layers"])
        config = NeuralFieldConfig(
            spatial_hidden=layers[1:-1],
            spatial_scale=tuple(float(value) for value in archive["spatial_scale"]),
        )
        if layers != config.spatial_layers:
            raise ValueError(
                "checkpoint spatial architecture must have 3 inputs and 8 outputs"
            )
        params = {
            "reference": {
                "height_normalized": _as_real(archive["reference_height_normalized"]),
                "atmosphere": _atmosphere_from_payload(archive, "reference_"),
            },
            "spatial": tuple(
                {
                    "w": _as_real(archive[f"spatial_{index}_weight"]),
                    "b": _as_real(archive[f"spatial_{index}_bias"]),
                }
                for index in range(len(layers) - 1)
            ),
        }
    _validate_neural_parameters(params, config)
    return params, config


def _validate_checkpoint_reference(params, reference, height_normalized):
    stored = params["reference"]
    for name, actual, expected in zip(
        ("height_normalized", *FIELD_NAMES),
        (stored["height_normalized"], *stored["atmosphere"]),
        (height_normalized, *reference),
    ):
        actual, expected = np.asarray(actual), np.asarray(expected)
        tolerance = max(
            8.0 * np.finfo(actual.dtype).eps,
            8.0 * np.finfo(expected.dtype).eps,
            1.0e-12,
        )
        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance if name == "height_normalized" else 0.0,
        ):
            raise ValueError(f"checkpoint reference {name} does not match the dataset")


def _coordinate_cube_from_payload(payload):
    x = _as_real(payload["x_normalized"])
    y = _as_real(payload["y_normalized"])
    z = _as_real(payload["height_normalized"])
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    return 2.0 * jnp.stack((xx, yy, zz), axis=-1) - 1.0


def _display_atmosphere_quantities(atmosphere: Atmosphere) -> tuple[np.ndarray, ...]:
    """Convert an atmosphere cube into T, log10(ne), Bx, and Bz display maps."""

    temperature = np.asarray(atmosphere.temperature, dtype=float)
    log_ne = np.log10(np.asarray(atmosphere.ne, dtype=float))
    strength_g = 1.0e4 * np.asarray(atmosphere.b, dtype=float)
    inclination = np.asarray(atmosphere.gamma_b, dtype=float)
    azimuth = np.asarray(atmosphere.chi_b, dtype=float)
    bx = strength_g * np.sin(inclination) * np.cos(azimuth)
    bz = strength_g * np.cos(inclination)
    return temperature, log_ne, bx, bz


def create_inversion_evaluation_figure(
    wavelength_nm,
    input_stokes,
    output_stokes,
    x_normalized,
    y_normalized,
    height_m,
    input_atmosphere: Atmosphere,
    output_atmosphere: Atmosphere,
    *,
    x_index: int,
    y_index: int,
    height_index: int,
    epoch: int | None = None,
):
    """Create the 4x3 linked-view layout as a non-interactive snapshot.

    This lightweight renderer is used for periodic W&B evaluations. The
    interactive GUI in :mod:`compare_inversion_gui` uses the same quantities
    and panel ordering.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    input_stokes = np.asarray(input_stokes, dtype=float)
    output_stokes = np.asarray(output_stokes, dtype=float)
    x_normalized = np.asarray(x_normalized, dtype=float)
    y_normalized = np.asarray(y_normalized, dtype=float)
    height_m = np.asarray(height_m, dtype=float)
    if input_stokes.shape != output_stokes.shape or input_stokes.shape != (
        4,
        wavelength_nm.size,
    ):
        raise ValueError("evaluation Stokes profiles must have shape (4, n_wave)")
    if not (0 <= x_index < x_normalized.size and 0 <= y_index < y_normalized.size):
        raise IndexError("evaluation horizontal index is outside the FOV")
    if not 0 <= height_index < height_m.size:
        raise IndexError("evaluation height index is outside the atmosphere")

    input_quantities = _display_atmosphere_quantities(input_atmosphere)
    output_quantities = _display_atmosphere_quantities(output_atmosphere)
    expected_shape = (x_normalized.size, y_normalized.size, height_m.size)
    all_quantities = (*input_quantities, *output_quantities)
    if any(values.shape != expected_shape for values in all_quantities):
        raise ValueError(
            f"evaluation atmosphere fields must have shape {expected_shape}"
        )

    figure, axes = plt.subplots(4, 3, figsize=(14.5, 12.0), constrained_layout=True)
    labels = (r"$I/I_c$", r"$Q/I_c$", r"$U/I_c$", r"$V/I_c$")
    quantity_titles = (
        "Temperature [K]",
        r"$\log_{10}(n_e\,[\mathrm{m}^{-3}])$",
        r"$B_x$ [G]",
        r"$B_z$ [G]",
    )
    cmaps = ("inferno", "viridis", "RdBu_r", "RdBu_r")
    continuum = max(
        abs(0.5 * (input_stokes[0, 0] + input_stokes[0, -1])),
        np.finfo(float).tiny,
    )
    for row, label in enumerate(labels):
        spectrum_axis = axes[row, 0]
        spectrum_axis.plot(
            wavelength_nm,
            input_stokes[row] / continuum,
            color="black",
            linewidth=1.5,
            label="Input",
        )
        spectrum_axis.plot(
            wavelength_nm,
            output_stokes[row] / continuum,
            color="tab:orange",
            linewidth=1.25,
            linestyle="--",
            label="Current inversion",
        )
        spectrum_axis.set_ylabel(label)
        spectrum_axis.grid(alpha=0.22)
        if row == 0:
            spectrum_axis.legend(loc="best", fontsize="small")
        if row == 3:
            spectrum_axis.set_xlabel("Wavelength [nm]")

        truth_slice = input_quantities[row][:, :, height_index]
        lower, upper = np.percentile(truth_slice, (5.0, 95.0))
        if not upper > lower:
            padding = max(abs(float(lower)) * 0.01, 1.0e-12)
            lower, upper = lower - padding, upper + padding
        normalization = Normalize(vmin=float(lower), vmax=float(upper))
        x_extent = (
            (float(x_normalized[0] - 0.5), float(x_normalized[0] + 0.5))
            if x_normalized.size == 1
            else (float(x_normalized[0]), float(x_normalized[-1]))
        )
        y_extent = (
            (float(y_normalized[0] - 0.5), float(y_normalized[0] + 0.5))
            if y_normalized.size == 1
            else (float(y_normalized[0]), float(y_normalized[-1]))
        )
        extent = (*x_extent, *y_extent)
        row_images = []
        row_quantities = (input_quantities[row], output_quantities[row])
        for column, values in enumerate(row_quantities, 1):
            map_axis = axes[row, column]
            image = map_axis.imshow(
                values[:, :, height_index].T,
                origin="lower",
                extent=extent,
                aspect="equal",
                interpolation="nearest",
                cmap=cmaps[row],
                norm=normalization,
            )
            row_images.append(image)
            map_axis.plot(
                x_normalized[x_index],
                y_normalized[y_index],
                marker="+",
                color="white",
                markeredgewidth=1.5,
                markersize=9,
            )
            if row == 0:
                map_axis.set_title(
                    "Input / truth" if column == 1 else "Current inversion"
                )
            if row == 3:
                map_axis.set_xlabel("x (normalized)")
            if column == 1:
                map_axis.set_ylabel(f"{quantity_titles[row]}\ny (normalized)")
        figure.colorbar(
            row_images[-1],
            ax=axes[row, 1:].tolist(),
            shrink=0.72,
            pad=0.02,
        )

    location = (
        f"x={x_normalized[x_index]:.3f}, y={y_normalized[y_index]:.3f}; "
        f"z={height_m[height_index] / 1000.0:.1f} km"
    )
    prefix = "Evaluation" if epoch is None else f"Evaluation at epoch {epoch}"
    figure.suptitle(f"{prefix} — {location}", fontweight="bold")
    return figure


def run_inversion(
    payload,
    kurucz_path=FE_I_6301_6302_LINE_LIST,
    result_path=DEFAULT_RESULT,
    checkpoint_path=DEFAULT_CHECKPOINT,
    field_config: NeuralFieldConfig = NeuralFieldConfig(),
    initial_params=None,
    inversion_epochs: int = 10,
    inversion_learning_rate: float = DEFAULT_INVERSION_INITIAL_LEARNING_RATE,
    inversion_final_learning_rate: float = DEFAULT_INVERSION_FINAL_LEARNING_RATE,
    training_batch_columns: int = 8,
    wavelength_batch: int = 48,
    synthesis_batch_columns: int = 16,
    stokes_weights=(1.0, 5.0, 5.0, 2.0),
    prior_weight: float = 1.0e-5,
    seed: int = 0,
    show_progress: bool = True,
    wavelength_parallelism: int = DEFAULT_WAVELENGTH_PARALLELISM,
    validation_columns: int = DEFAULT_VALIDATION_COLUMNS,
    experiment_logger: WandbRunLogger | None = None,
    spectra_output_path: str | Path | None = None,
):
    """Invert spectra with bounded corrections to an exact fixed reference."""

    inversion_run_started = time.perf_counter()
    wavelength_parallelism = _validate_positive_integer(
        "wavelength_parallelism", wavelength_parallelism
    )
    validation_columns = _validate_positive_integer(
        "validation_columns", validation_columns
    )
    _validate_learning_rate_range(
        inversion_learning_rate, inversion_final_learning_rate
    )
    if spectra_output_path is not None:
        spectra_path = Path(spectra_output_path).resolve()
        protected_paths = {
            Path(result_path).resolve(): "result",
            Path(checkpoint_path).resolve(): "checkpoint",
        }
        if spectra_path in protected_paths:
            raise ValueError(
                "spectra_output_path must differ from the "
                f"{protected_paths[spectra_path]} path"
            )
    nx, ny, n_depth, n_wave = validate_test_cube(payload)
    if experiment_logger is not None:
        experiment_logger.update_config(
            {
                "dataset_nx": nx,
                "dataset_ny": ny,
                "dataset_n_depth": n_depth,
                "dataset_n_wave": n_wave,
                "dataset_n_columns": nx * ny,
                "inversion_learning_rate_schedule": "sin_squared",
                "inversion_initial_learning_rate": inversion_learning_rate,
                "inversion_final_learning_rate": inversion_final_learning_rate,
            }
        )
        experiment_logger.update_summary(
            {
                "outputs/result_path": Path(result_path),
                "outputs/checkpoint_path": Path(checkpoint_path),
                "outputs/spectra_path": (
                    None if spectra_output_path is None else Path(spectra_output_path)
                ),
            }
        )
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
    if initial_params is None:
        params = initialize_neural_field(
            jax.random.PRNGKey(seed),
            reference,
            payload["height_normalized"],
            field_config,
        )
    else:
        _validate_neural_parameters(initial_params, field_config)
        _validate_checkpoint_reference(
            initial_params, reference, payload["height_normalized"]
        )
        params = jax.tree.map(_as_real, initial_params)
    # Save the exact-reference initialization (or compatible warm start).
    save_checkpoint(checkpoint_path, params, field_config)
    spectral_inversion_started = time.perf_counter()

    evaluation_callback = None
    if experiment_logger is not None:
        evaluation_x_index = nx // 2
        evaluation_y_index = ny // 2
        evaluation_height_index = n_depth // 2

        def evaluation_callback(epoch, current_params):
            current_atmosphere = evaluate_neural_field_cube(
                current_params,
                coordinates,
                _as_real(field_config.spatial_scale),
                batch_columns=synthesis_batch_columns,
                show_progress=False,
            )
            current_column = Atmosphere(
                *(
                    field[
                        evaluation_x_index : evaluation_x_index + 1,
                        evaluation_y_index : evaluation_y_index + 1,
                        :,
                    ]
                    for field in current_atmosphere
                )
            )
            current_stokes = synthesize_atmosphere_cube(
                lines,
                wavelengths,
                dz,
                current_column,
                batch_columns=1,
                show_progress=False,
                description="Synthesizing W&B evaluation spectrum",
                wavelength_parallelism=wavelength_parallelism,
            )
            figure = create_inversion_evaluation_figure(
                wavelengths,
                np.asarray(payload["observed_stokes"])[
                    evaluation_x_index, evaluation_y_index
                ],
                np.asarray(current_stokes)[0, 0],
                payload["x_normalized"],
                payload["y_normalized"],
                payload["height_m"],
                truth,
                current_atmosphere,
                x_index=evaluation_x_index,
                y_index=evaluation_y_index,
                height_index=evaluation_height_index,
                epoch=epoch,
            )
            try:
                experiment_logger.log_evaluation_figure(epoch, figure)
            finally:
                import matplotlib.pyplot as plt

                plt.close(figure)

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
        final_learning_rate=inversion_final_learning_rate,
        stokes_weights=stokes_weights,
        spatial_scale=_as_real(field_config.spatial_scale),
        prior_weight=prior_weight,
        seed=seed + 1,
        show_progress=show_progress,
        wavelength_parallelism=wavelength_parallelism,
        validation_columns=validation_columns,
        metrics_callback=(
            None
            if experiment_logger is None
            else experiment_logger.log_inversion_metrics
        ),
        evaluation_callback=evaluation_callback,
        evaluation_every=(
            DEFAULT_WANDB_EVALUATION_EVERY
            if experiment_logger is None
            else experiment_logger.evaluation_every
        ),
    )
    spectral_inversion_seconds = time.perf_counter() - spectral_inversion_started
    if experiment_logger is not None:
        experiment_logger.update_summary(
            {"timing/spectral_inversion_seconds": spectral_inversion_seconds}
        )
    # Preserve the best fitted iterate before any expensive full-cube rendering
    # or archive compression can fail.
    save_checkpoint(checkpoint_path, params, field_config)
    atmosphere_evaluation_started = time.perf_counter()
    inferred, normalized_corrections = evaluate_neural_field_cube(
        params,
        coordinates,
        _as_real(field_config.spatial_scale),
        batch_columns=synthesis_batch_columns,
        show_progress=show_progress,
        return_corrections=True,
    )
    _validate_atmosphere(inferred, expected_shape=(nx, ny, n_depth))
    atmosphere_evaluation_seconds = time.perf_counter() - atmosphere_evaluation_started
    final_synthesis_started = time.perf_counter()
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
    synthetic_array = np.asarray(final_stokes)
    final_synthesis_seconds = time.perf_counter() - final_synthesis_started
    observed_array = np.asarray(payload["observed_stokes"])
    continuum = 0.5 * (observed_array[:, :, 0, 0] + observed_array[:, :, 0, -1])
    continuum = np.maximum(np.abs(continuum), np.finfo(NUMPY_REAL_DTYPE).tiny)
    residual = (synthetic_array - observed_array) / continuum[:, :, None, None]
    final_full_loss = np.mean(
        (residual * np.asarray(stokes_weights)[None, None, :, None]) ** 2
    )
    updates_per_epoch = math.ceil((nx * ny) / training_batch_columns)
    total_optimizer_steps = inversion_epochs * updates_per_epoch
    learning_rate_schedule = sin_squared_learning_rate_schedule(
        total_optimizer_steps,
        inversion_learning_rate,
        inversion_final_learning_rate,
    )
    learning_rate_history = np.asarray(
        [
            float(
                np.asarray(
                    learning_rate_schedule(
                        0
                        if epoch == 0
                        else min(
                            epoch * updates_per_epoch - 1,
                            max(total_optimizer_steps - 1, 0),
                        )
                    )
                )
            )
            for epoch in range(inversion_epochs + 1)
        ]
    )
    result = {
        **{key: np.asarray(value) for key, value in payload.items()},
        "synthetic_stokes": synthetic_array,
        "inferred_normalized_corrections": normalized_corrections,
        **_correction_metadata(),
        "inversion_loss_total_spectral_prior": inversion_history,
        "inversion_loss_columns": np.asarray(("total", "spectral", "prior")),
        "validation_full_wavelength_loss": validation_history,
        "inversion_learning_rate_by_epoch": learning_rate_history,
        "inversion_learning_rate_schedule": np.asarray("sin_squared"),
        "inversion_initial_learning_rate": np.asarray(inversion_learning_rate),
        "inversion_final_learning_rate": np.asarray(inversion_final_learning_rate),
        "best_validation_epoch": np.asarray(int(np.argmin(validation_history))),
        "best_validation_loss": np.asarray(np.min(validation_history)),
        "final_full_cube_spectral_loss": np.asarray(final_full_loss),
        "spatial_layers": np.asarray(field_config.spatial_layers, dtype=np.int64),
        "spatial_scale": np.asarray(field_config.spatial_scale),
        "stokes_weights": np.asarray(stokes_weights),
        "prior_weight": np.asarray(prior_weight),
        "training_batch_columns": np.asarray(training_batch_columns),
        "wavelength_parallelism": np.asarray(wavelength_parallelism),
        "validation_columns": np.asarray(validation_columns),
        "inversion_precision": np.asarray(adora_precision.configured_precision()),
        "seed": np.asarray(seed),
        **_atmosphere_payload("inferred_", inferred),
    }
    validate_inversion_result(result)
    result_path = Path(result_path)
    result_save_started = time.perf_counter()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_path, **result)
    result_save_seconds = time.perf_counter() - result_save_started
    spectra_save_started = time.perf_counter()
    if spectra_output_path is not None:
        save_spectra_output(
            spectra_output_path,
            x_normalized=payload["x_normalized"],
            y_normalized=payload["y_normalized"],
            wavelength_nm=wavelengths,
            input_stokes=observed_array,
            output_stokes=synthetic_array,
        )
    spectra_save_seconds = time.perf_counter() - spectra_save_started
    save_checkpoint(checkpoint_path, params, field_config)
    if experiment_logger is not None:
        experiment_logger.update_summary(
            {
                "inversion/best_validation_epoch": int(np.argmin(validation_history)),
                "inversion/best_validation_loss": float(np.min(validation_history)),
                "inversion/final_full_cube_spectral_loss": float(final_full_loss),
                "timing/atmosphere_evaluation_seconds": (atmosphere_evaluation_seconds),
                "timing/final_synthesis_seconds": final_synthesis_seconds,
                "timing/result_save_seconds": result_save_seconds,
                "timing/spectra_save_seconds": spectra_save_seconds,
                "timing/run_inversion_seconds": (
                    time.perf_counter() - inversion_run_started
                ),
            }
        )
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


def _parse_spatial_scale(value: str) -> tuple[float, ...]:
    try:
        scales = tuple(float(piece.strip()) for piece in value.split(","))
        NeuralFieldConfig(spatial_scale=scales).validate()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "spatial scale needs eight positive finite values"
        ) from exc
    return scales


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


def _parse_wandb_tags(value: str) -> tuple[str, ...]:
    tags = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tags:
        raise argparse.ArgumentTypeError("W&B tags must not be empty")
    return tags


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
        help="load --dataset and invert without regeneration",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--spectra-output",
        "--output-spectra",
        type=Path,
        nargs="?",
        const=DEFAULT_SPECTRA_OUTPUT,
        metavar="PATH",
        help=(
            "optionally write a compact NPZ containing wavelength plus input "
            "and fitted Stokes cubes; without PATH, use "
            f"{DEFAULT_SPECTRA_OUTPUT}"
        ),
    )
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
        "--spatial-hidden", type=_parse_hidden_layers, default=(96, 96, 96)
    )
    parser.add_argument(
        "--spatial-scale",
        type=_parse_spatial_scale,
        default=_SPATIAL_SCALE_VALUES,
        metavar="T,NE,NH,V,VT,B,GAMMA,CHI",
        help="positive scales for bounded outputs: dex for T/NE/NH/VT/B; units of 20 km/s, pi, pi/2 for V/GAMMA/CHI (default: all 1)",
    )

    parser.add_argument("--inversion-epochs", type=int, default=10)
    parser.add_argument(
        "--inversion-learning-rate",
        "--inversion-initial-learning-rate",
        type=float,
        default=DEFAULT_INVERSION_INITIAL_LEARNING_RATE,
        help="initial sin-squared schedule rate (default: 1e-3)",
    )
    parser.add_argument(
        "--inversion-final-learning-rate",
        type=float,
        default=DEFAULT_INVERSION_FINAL_LEARNING_RATE,
        help="final sin-squared schedule rate (default: 1e-5)",
    )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        help="enable optional Weights & Biases experiment logging",
    )
    wandb_group.add_argument(
        "--wandb-login",
        action="store_true",
        help="run wandb.login() interactively before starting an online run",
    )
    wandb_group.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project name (default: {DEFAULT_WANDB_PROJECT})",
    )
    wandb_group.add_argument("--wandb-entity", help="W&B user or team")
    wandb_group.add_argument("--wandb-name", help="W&B run display name")
    wandb_group.add_argument("--wandb-group", help="optional W&B run group")
    wandb_group.add_argument(
        "--wandb-tags",
        type=_parse_wandb_tags,
        default=(),
        metavar="TAG[,TAG...]",
        help="comma-separated W&B tags",
    )
    wandb_group.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B synchronization mode (default: online)",
    )
    wandb_group.add_argument(
        "--wandb-dir",
        type=Path,
        help="directory for local W&B metadata (default: ./wandb)",
    )

    wandb_group.add_argument(
        "--wandb-evaluation-every",
        type=int,
        default=DEFAULT_WANDB_EVALUATION_EVERY,
        metavar="EPOCHS",
        help=(
            "plot current spectra and atmosphere maps every N inversion epochs "
            "(default: 50; epoch zero and the final epoch are also logged)"
        ),
    )
    wandb_group.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="upload dataset, result, and checkpoint NPZ files as W&B Artifacts",
    )
    return parser


def main(argv=None):
    """Run generation, inversion, or the complete requested workflow."""

    args = build_argument_parser().parse_args(argv)
    if args.wandb_login and not args.wandb:
        raise ValueError("--wandb-login requires --wandb")
    if args.wandb_login and args.wandb_mode != "online":
        raise ValueError("--wandb-login requires --wandb-mode online")
    if args.generate_only and args.spectra_output is not None:
        raise ValueError("--spectra-output requires an inversion run")
    if args.spectra_output is not None:
        spectra_path = args.spectra_output.resolve()
        protected_paths = {
            args.dataset.resolve(): "dataset",
            args.result.resolve(): "result",
            args.checkpoint.resolve(): "checkpoint",
        }
        if spectra_path in protected_paths:
            raise ValueError(
                "--spectra-output must differ from the "
                f"--{protected_paths[spectra_path]} path"
            )
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
        spatial_hidden=args.spatial_hidden,
        spatial_scale=args.spatial_scale,
    )
    config.validate()
    if not args.generate_only:
        if args.checkpoint_in is not None:
            # Load before generation so a missing/corrupt warm start cannot
            # waste the expensive default truth-cube synthesis.
            initial_params, config = load_checkpoint(args.checkpoint_in)
    experiment_logger = _initialize_wandb(args, config) if args.wandb else None
    workflow_started = time.perf_counter()
    try:
        if experiment_logger is not None and args.checkpoint_in is not None:
            experiment_logger.track_file(
                args.checkpoint_in,
                artifact_type="model",
                role="input",
            )

        dataset_started = time.perf_counter()
        if args.invert_only:
            payload = load_test_cube(args.dataset)
            dataset_timing_name = "timing/dataset_load_seconds"
            dataset_artifact_role = "input"
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
                spatial_scale=config.spatial_scale,
                wavelength_parallelism=args.wavelength_parallelism,
            )
            dataset_timing_name = "timing/generation_seconds"
            dataset_artifact_role = "output"

        if experiment_logger is not None:
            nx, ny, n_depth, n_wave = validate_test_cube(payload)
            if args.generate_only:
                experiment_logger.update_config(
                    {
                        "dataset_nx": nx,
                        "dataset_ny": ny,
                        "dataset_n_depth": n_depth,
                        "dataset_n_wave": n_wave,
                        "dataset_n_columns": nx * ny,
                    }
                )
            experiment_logger.update_summary(
                {
                    dataset_timing_name: time.perf_counter() - dataset_started,
                    "outputs/dataset_path": args.dataset,
                }
            )
            experiment_logger.track_file(
                args.dataset,
                artifact_type="dataset",
                role=dataset_artifact_role,
            )

        if not args.generate_only:
            run_inversion(
                payload,
                kurucz_path=args.kurucz,
                result_path=args.result,
                checkpoint_path=args.checkpoint,
                field_config=config,
                initial_params=initial_params,
                inversion_epochs=args.inversion_epochs,
                inversion_learning_rate=args.inversion_learning_rate,
                inversion_final_learning_rate=args.inversion_final_learning_rate,
                training_batch_columns=args.training_batch_columns,
                wavelength_batch=args.wavelength_batch,
                synthesis_batch_columns=args.synthesis_batch_columns,
                stokes_weights=args.stokes_weights,
                prior_weight=args.prior_weight,
                seed=args.seed,
                show_progress=show_progress,
                wavelength_parallelism=args.wavelength_parallelism,
                validation_columns=args.validation_columns,
                experiment_logger=experiment_logger,
                spectra_output_path=args.spectra_output,
            )
            if experiment_logger is not None:
                experiment_logger.track_file(
                    args.checkpoint,
                    artifact_type="model",
                    role="output",
                )
                experiment_logger.track_file(
                    args.result,
                    artifact_type="inversion-result",
                    role="output",
                )
                if args.spectra_output is not None:
                    experiment_logger.track_file(
                        args.spectra_output,
                        artifact_type="spectra",
                        role="output",
                    )
        if experiment_logger is not None:
            experiment_logger.update_summary(
                {
                    "timing/total_workflow_seconds": (
                        time.perf_counter() - workflow_started
                    )
                }
            )
    except BaseException:
        if experiment_logger is not None:
            try:
                experiment_logger.finish(exit_code=1)
            except Exception:
                pass
        raise
    else:
        if experiment_logger is not None:
            experiment_logger.finish(exit_code=0)
    return None


if __name__ == "__main__":
    main()
