"""Create a reproducible center-to-boundary full-Stokes PINN test set.

The default cube has a 10 percent center enhancement in temperature and
electron density, LOS velocity changing from -5 km/s on every horizontal
boundary to +5 km/s at the center, and signed LOS magnetic field changing from
-500 G to +500 G.  Magnetic-field strength remains positive; the signed LOS
component is represented through the field inclination.

An exactly zero-output neural-field checkpoint is written with the fixed
boundary reference and correction scales large enough for the requested truth
cube. Load it with ``pinn_3d_inversion.py --checkpoint-in`` to preserve that
reference and those scales. No FAL-C pretraining is needed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Configure process-wide precision before importing JAX or any physics module
# that constructs module-level arrays.
import adora_precision

adora_precision.configure_precision(adora_precision.precision_from_argv())

import jax
import jax.numpy as jnp
import numpy as np

import pinn_3d_inversion as pinn3d
from adora_data import FE_I_6301_6302_LINE_LIST
from lineop import read_kurucz


DEFAULT_DATASET = Path("data") / "pinn_contrast_cube.npz"
DEFAULT_CHECKPOINT = Path("data") / "pinn_contrast_initial.npz"
DEFAULT_NX = 51
DEFAULT_NY = 51
DEFAULT_N_WAVE = 201
DEFAULT_TEMPERATURE_CENTER_RATIO = 1.10
DEFAULT_NE_CENTER_RATIO = 1.10
DEFAULT_BOUNDARY_VELOCITY_KM_S = -5.0
DEFAULT_CENTER_VELOCITY_KM_S = 5.0
DEFAULT_BOUNDARY_B_LOS_GAUSS = -500.0
DEFAULT_CENTER_B_LOS_GAUSS = 500.0
DEFAULT_FIELD_STRENGTH_GAUSS = 1000.0
DEFAULT_PROFILE_POWER = 1.0
DEFAULT_THERMODYNAMIC_SIGMA_KM = 100.0
DEFAULT_SCALE_SAFETY_FACTOR = 1.25
DEFAULT_SCALE_PADDING = 0.05
GAUSS_PER_TESLA = 1.0e4
METERS_PER_KILOMETER = 1.0e3


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _greater_than_one_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be greater than one")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _odd_grid_size(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed < 3 or parsed % 2 == 0:
        raise argparse.ArgumentTypeError(
            "horizontal grid sizes must be odd integers of at least three"
        )
    return parsed


def _hidden_layers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden layers must be comma-separated integers"
        ) from exc
    if not result or any(width <= 0 for width in result):
        raise argparse.ArgumentTypeError("hidden-layer sizes must be positive")
    return result


def _spatial_scale(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "spatial scale must contain comma-separated numbers"
        ) from exc
    if len(result) != 8 or any(
        not math.isfinite(item) or item <= 0.0 for item in result
    ):
        raise argparse.ArgumentTypeError(
            "spatial scale must contain eight positive finite values"
        )
    return result


def center_boundary_envelope(
    x,
    y,
    n_depth: int,
    profile_power: float = DEFAULT_PROFILE_POWER,
) -> jax.Array:
    """Return a height-independent envelope that is zero at edges and one centrally."""

    if not isinstance(n_depth, int) or n_depth <= 0:
        raise ValueError("n_depth must be a positive integer")
    if not math.isfinite(profile_power) or profile_power <= 0.0:
        raise ValueError("profile_power must be positive and finite")
    x = jnp.asarray(x, dtype=adora_precision.REAL_DTYPE)
    y = jnp.asarray(y, dtype=adora_precision.REAL_DTYPE)
    for name, axis in (("x", x), ("y", y)):
        if axis.ndim != 1 or axis.size < 3 or axis.size % 2 == 0:
            raise ValueError(f"{name} must contain an odd number of at least 3 points")
        if not bool(jnp.all(jnp.isfinite(axis))) or not bool(
            jnp.all(jnp.diff(axis) > 0.0)
        ):
            raise ValueError(f"{name} must be finite and strictly increasing")
        tolerance = 16.0 * jnp.finfo(axis.dtype).eps
        expected = jnp.asarray((0.0, 0.5, 1.0), dtype=axis.dtype)
        actual = jnp.stack((axis[0], axis[axis.size // 2], axis[-1]))
        if not bool(jnp.allclose(actual, expected, rtol=0.0, atol=tolerance)):
            raise ValueError(f"{name} must span 0 to 1 and contain center 0.5")

    x_profile = jnp.clip(4.0 * x * (1.0 - x), 0.0, 1.0)
    y_profile = jnp.clip(4.0 * y * (1.0 - y), 0.0, 1.0)
    horizontal = jnp.power(
        x_profile[:, None] * y_profile[None, :],
        profile_power,
    )
    return jnp.broadcast_to(horizontal[..., None], (*horizontal.shape, n_depth))


def height_localized_envelope(
    horizontal_envelope,
    height_m,
    *,
    target_height_km: float,
    sigma_km: float = DEFAULT_THERMODYNAMIC_SIGMA_KM,
) -> tuple[jax.Array, jax.Array, int, float]:
    """Localize a horizontal contrast around the nearest sampled height.

    The returned vertical Gaussian is exactly one at the selected FAL-C depth
    point.  Consequently, requested center/boundary endpoints are exact on
    that stored layer even when the original height grid does not contain the
    requested height verbatim.
    """

    if not math.isfinite(target_height_km):
        raise ValueError("target_height_km must be finite")
    if not math.isfinite(sigma_km) or sigma_km <= 0.0:
        raise ValueError("sigma_km must be positive and finite")
    height_host = np.asarray(height_m, dtype=float)
    if (
        height_host.ndim != 1
        or height_host.size < 2
        or np.any(~np.isfinite(height_host))
        or np.any(np.diff(height_host) <= 0.0)
    ):
        raise ValueError("height_m must be finite, increasing, and one-dimensional")
    target_height_m = target_height_km * METERS_PER_KILOMETER
    if not height_host[0] <= target_height_m <= height_host[-1]:
        raise ValueError(
            f"target height {target_height_km:g} km lies outside the FAL-C "
            f"grid [{height_host[0] / METERS_PER_KILOMETER:g}, "
            f"{height_host[-1] / METERS_PER_KILOMETER:g}] km"
        )

    horizontal_envelope = jnp.asarray(
        horizontal_envelope,
        dtype=adora_precision.REAL_DTYPE,
    )
    if (
        horizontal_envelope.ndim != 3
        or horizontal_envelope.shape[-1] != height_host.size
        or not bool(jnp.all(jnp.isfinite(horizontal_envelope)))
    ):
        raise ValueError(
            "horizontal_envelope must be finite with shape (nx, ny, n_depth)"
        )
    target_index = int(np.argmin(np.abs(height_host - target_height_m)))
    effective_height_m = float(height_host[target_index])
    height = jnp.asarray(height_host, dtype=adora_precision.REAL_DTYPE)
    sigma_m = sigma_km * METERS_PER_KILOMETER
    vertical = jnp.exp(-0.5 * ((height - effective_height_m) / sigma_m) ** 2)
    vertical = vertical.at[target_index].set(1.0)
    return (
        horizontal_envelope * vertical[None, None, :],
        vertical,
        target_index,
        effective_height_m,
    )


def anchor_reference_temperature(
    reference: pinn3d.Atmosphere,
    reference_cube: pinn3d.Atmosphere,
    vertical_envelope,
    *,
    target_index: int,
    boundary_temperature_k: float,
) -> tuple[pinn3d.Atmosphere, pinn3d.Atmosphere]:
    """Smoothly anchor the boundary profile to an exact temperature layer."""

    if not math.isfinite(boundary_temperature_k) or boundary_temperature_k <= 0.0:
        raise ValueError("boundary_temperature_k must be positive and finite")
    reference_shape = pinn3d._validate_atmosphere(reference)
    cube_shape = pinn3d._validate_atmosphere(reference_cube)
    if len(reference_shape) != 1 or len(cube_shape) != 3:
        raise ValueError(
            "reference and reference_cube must be one- and three-dimensional"
        )
    if cube_shape[-1] != reference_shape[0]:
        raise ValueError("reference and reference_cube depth dimensions must match")
    if not isinstance(target_index, int) or not 0 <= target_index < reference_shape[0]:
        raise ValueError("target_index lies outside the reference atmosphere")
    vertical_envelope = jnp.asarray(
        vertical_envelope,
        dtype=adora_precision.REAL_DTYPE,
    )
    if (
        vertical_envelope.shape != reference_shape
        or not bool(jnp.all(jnp.isfinite(vertical_envelope)))
        or not bool(jnp.all((vertical_envelope >= 0.0) & (vertical_envelope <= 1.0)))
    ):
        raise ValueError(
            "vertical_envelope must be finite and lie between zero and one"
        )

    original_temperature = float(np.asarray(reference.temperature)[target_index])
    log_anchor_ratio = math.log(boundary_temperature_k / original_temperature)
    scale = jnp.exp(log_anchor_ratio * vertical_envelope)
    profile_temperature = (
        (reference.temperature * scale).at[target_index].set(boundary_temperature_k)
    )
    cube_temperature = (
        (reference_cube.temperature * scale[None, None, :])
        .at[..., target_index]
        .set(boundary_temperature_k)
    )
    return (
        reference._replace(temperature=profile_temperature),
        reference_cube._replace(temperature=cube_temperature),
    )


def _validate_magnetic_geometry(
    boundary_b_los_gauss: float,
    center_b_los_gauss: float,
    field_strength_gauss: float,
) -> tuple[float, float, float]:
    values = (boundary_b_los_gauss, center_b_los_gauss, field_strength_gauss)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("magnetic-field values must be finite")
    if field_strength_gauss <= 0.0:
        raise ValueError("field_strength_gauss must be positive")
    if max(abs(boundary_b_los_gauss), abs(center_b_los_gauss)) >= (
        field_strength_gauss
    ):
        raise ValueError(
            "field_strength_gauss must be strictly greater than the absolute "
            "boundary and center LOS fields"
        )
    field_strength_t = field_strength_gauss / GAUSS_PER_TESLA
    boundary_inclination = math.acos(boundary_b_los_gauss / field_strength_gauss)
    center_inclination = math.acos(center_b_los_gauss / field_strength_gauss)
    return field_strength_t, boundary_inclination, center_inclination


def build_contrast_atmosphere(
    reference_cube: pinn3d.Atmosphere,
    envelope,
    *,
    thermodynamic_envelope=None,
    temperature_center_ratio: float = DEFAULT_TEMPERATURE_CENTER_RATIO,
    ne_center_ratio: float = DEFAULT_NE_CENTER_RATIO,
    boundary_velocity_km_s: float = DEFAULT_BOUNDARY_VELOCITY_KM_S,
    center_velocity_km_s: float = DEFAULT_CENTER_VELOCITY_KM_S,
    boundary_b_los_gauss: float = DEFAULT_BOUNDARY_B_LOS_GAUSS,
    center_b_los_gauss: float = DEFAULT_CENTER_B_LOS_GAUSS,
    field_strength_gauss: float = DEFAULT_FIELD_STRENGTH_GAUSS,
) -> pinn3d.Atmosphere:
    """Apply exact center/boundary endpoint values to a reference cube."""

    if not math.isfinite(temperature_center_ratio) or temperature_center_ratio <= 0.0:
        raise ValueError("temperature_center_ratio must be positive and finite")
    if not math.isfinite(ne_center_ratio) or ne_center_ratio <= 0.0:
        raise ValueError("ne_center_ratio must be positive and finite")
    if not all(
        math.isfinite(value) for value in (boundary_velocity_km_s, center_velocity_km_s)
    ):
        raise ValueError("velocity endpoints must be finite")
    field_strength_t, _, _ = _validate_magnetic_geometry(
        boundary_b_los_gauss,
        center_b_los_gauss,
        field_strength_gauss,
    )
    shape = pinn3d._validate_atmosphere(reference_cube)
    if len(shape) != 3:
        raise ValueError("reference_cube must have shape (nx, ny, depth)")
    envelope = jnp.asarray(envelope, dtype=adora_precision.REAL_DTYPE)
    if envelope.shape != shape or not bool(jnp.all(jnp.isfinite(envelope))):
        raise ValueError("envelope must be finite and match reference_cube")
    tolerance = 16.0 * jnp.finfo(envelope.dtype).eps
    if not bool(jnp.all((envelope >= -tolerance) & (envelope <= 1.0 + tolerance))):
        raise ValueError("envelope values must lie between zero and one")
    envelope = jnp.clip(envelope, 0.0, 1.0)

    if thermodynamic_envelope is None:
        thermodynamic_envelope = envelope
    else:
        thermodynamic_envelope = jnp.asarray(
            thermodynamic_envelope,
            dtype=adora_precision.REAL_DTYPE,
        )
        if thermodynamic_envelope.shape != shape or not bool(
            jnp.all(jnp.isfinite(thermodynamic_envelope))
        ):
            raise ValueError(
                "thermodynamic_envelope must be finite and match reference_cube"
            )
        if not bool(
            jnp.all(
                (thermodynamic_envelope >= -tolerance)
                & (thermodynamic_envelope <= 1.0 + tolerance)
            )
        ):
            raise ValueError(
                "thermodynamic_envelope values must lie between zero and one"
            )
        thermodynamic_envelope = jnp.clip(thermodynamic_envelope, 0.0, 1.0)

    temperature_multiplier = (
        1.0 + (temperature_center_ratio - 1.0) * thermodynamic_envelope
    )
    ne_multiplier = 1.0 + (ne_center_ratio - 1.0) * thermodynamic_envelope
    boundary_velocity = boundary_velocity_km_s * METERS_PER_KILOMETER
    center_velocity = center_velocity_km_s * METERS_PER_KILOMETER
    velocity = boundary_velocity + (center_velocity - boundary_velocity) * envelope
    b_los_gauss = (
        boundary_b_los_gauss + (center_b_los_gauss - boundary_b_los_gauss) * envelope
    )
    inclination = jnp.arccos(jnp.clip(b_los_gauss / field_strength_gauss, -1.0, 1.0))
    magnetic_strength = jnp.full_like(reference_cube.b, field_strength_t)

    return pinn3d.Atmosphere(
        reference_cube.temperature * temperature_multiplier,
        reference_cube.ne * ne_multiplier,
        reference_cube.nhtot,
        velocity,
        reference_cube.vturb,
        magnetic_strength,
        inclination,
        reference_cube.chi_b,
    )


def select_spatial_scale(
    reference: pinn3d.Atmosphere,
    truth: pinn3d.Atmosphere,
    *,
    explicit_scale: tuple[float, ...] | None = None,
    safety_factor: float = DEFAULT_SCALE_SAFETY_FACTOR,
    padding: float = DEFAULT_SCALE_PADDING,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Return required and selected reference-relative correction scales for inversion."""

    if not math.isfinite(safety_factor) or safety_factor <= 1.0:
        raise ValueError("safety_factor must be finite and greater than one")
    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError("padding must be finite and non-negative")
    corrections = np.asarray(pinn3d.atmosphere_to_corrections(truth, reference))
    required = np.max(
        np.abs(corrections),
        axis=(0, 1, 2),
    )
    if explicit_scale is None:
        default_scale = np.asarray(
            pinn3d.NeuralFieldConfig().spatial_scale,
            dtype=float,
        )
        selected_array = np.maximum(
            default_scale,
            safety_factor * required + padding,
        )
        selected = tuple(float(value) for value in selected_array)
    else:
        selected = tuple(float(value) for value in explicit_scale)

    config = pinn3d.NeuralFieldConfig(spatial_scale=selected)
    config.validate()
    pinn3d.validate_spatial_reachability(reference, truth, selected)
    return required, selected


def _validated_output_paths(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    dataset_path = Path(dataset_path).expanduser()
    checkpoint_path = Path(checkpoint_path).expanduser()
    if dataset_path.resolve() == checkpoint_path.resolve():
        raise ValueError("dataset and checkpoint paths must be different")
    for description, path in (
        ("dataset", dataset_path),
        ("checkpoint", checkpoint_path),
    ):
        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"{description} output is not a file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{description} output already exists: {path}; pass --overwrite "
                "to replace it"
            )
    return dataset_path, checkpoint_path


def create_test_set(
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    kurucz_path: str | Path = FE_I_6301_6302_LINE_LIST,
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    n_wave: int = DEFAULT_N_WAVE,
    wavelength_padding_nm: float = 0.05,
    synthesis_batch_columns: int = 16,
    wavelength_parallelism: int = pinn3d.DEFAULT_WAVELENGTH_PARALLELISM,
    profile_power: float = DEFAULT_PROFILE_POWER,
    thermodynamic_height_km: float | None = None,
    thermodynamic_sigma_km: float = DEFAULT_THERMODYNAMIC_SIGMA_KM,
    temperature_boundary_k: float | None = None,
    temperature_center_ratio: float = DEFAULT_TEMPERATURE_CENTER_RATIO,
    ne_center_ratio: float = DEFAULT_NE_CENTER_RATIO,
    boundary_velocity_km_s: float = DEFAULT_BOUNDARY_VELOCITY_KM_S,
    center_velocity_km_s: float = DEFAULT_CENTER_VELOCITY_KM_S,
    boundary_b_los_gauss: float = DEFAULT_BOUNDARY_B_LOS_GAUSS,
    center_b_los_gauss: float = DEFAULT_CENTER_B_LOS_GAUSS,
    field_strength_gauss: float = DEFAULT_FIELD_STRENGTH_GAUSS,
    magnetic_azimuth_deg: float = 0.0,
    spatial_hidden: tuple[int, ...] = (96, 96, 96),
    spatial_scale: tuple[float, ...] | None = None,
    scale_safety_factor: float = DEFAULT_SCALE_SAFETY_FACTOR,
    scale_padding: float = DEFAULT_SCALE_PADDING,
    seed: int = 0,
    show_progress: bool = True,
    overwrite: bool = False,
) -> tuple[dict[str, np.ndarray], pinn3d.NeuralFieldConfig]:
    """Synthesize the requested cube and write its compatible checkpoint."""

    dataset_path, checkpoint_path = _validated_output_paths(
        dataset_path,
        checkpoint_path,
        overwrite=overwrite,
    )
    if not isinstance(nx, int) or nx < 3 or nx % 2 == 0:
        raise ValueError("nx must be an odd integer of at least three")
    if not isinstance(ny, int) or ny < 3 or ny % 2 == 0:
        raise ValueError("ny must be an odd integer of at least three")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not math.isfinite(magnetic_azimuth_deg):
        raise ValueError("magnetic_azimuth_deg must be finite")
    if not math.isfinite(thermodynamic_sigma_km) or thermodynamic_sigma_km <= 0.0:
        raise ValueError("thermodynamic_sigma_km must be positive and finite")
    if temperature_boundary_k is not None and thermodynamic_height_km is None:
        raise ValueError("temperature_boundary_k requires thermodynamic_height_km")

    field_strength_t, boundary_inclination, _ = _validate_magnetic_geometry(
        boundary_b_los_gauss,
        center_b_los_gauss,
        field_strength_gauss,
    )
    lines = read_kurucz(Path(kurucz_path).expanduser())
    wavelengths = pinn3d.build_wavelength_grid(
        lines,
        n_wave=n_wave,
        padding_nm=wavelength_padding_nm,
    )
    (
        x,
        y,
        height,
        z_normalized,
        dz,
        _,
        reference,
        reference_cube,
    ) = pinn3d.create_falc_reference_cube(
        nx,
        ny,
        magnetic_field_t=field_strength_t,
        inclination_rad=boundary_inclination,
        azimuth_rad=math.radians(magnetic_azimuth_deg),
    )
    envelope = center_boundary_envelope(
        x,
        y,
        int(height.size),
        profile_power,
    )
    if thermodynamic_height_km is None:
        thermodynamic_envelope = envelope
        vertical_envelope = jnp.ones_like(height)
        thermodynamic_target_index = -1
        effective_thermodynamic_height_m = math.nan
        thermodynamic_mode = "height_independent"
    else:
        (
            thermodynamic_envelope,
            vertical_envelope,
            thermodynamic_target_index,
            effective_thermodynamic_height_m,
        ) = height_localized_envelope(
            envelope,
            height,
            target_height_km=thermodynamic_height_km,
            sigma_km=thermodynamic_sigma_km,
        )
        thermodynamic_mode = "height_localized_gaussian"
        if temperature_boundary_k is not None:
            reference, reference_cube = anchor_reference_temperature(
                reference,
                reference_cube,
                vertical_envelope,
                target_index=thermodynamic_target_index,
                boundary_temperature_k=temperature_boundary_k,
            )

    boundary_velocity_m_s = boundary_velocity_km_s * METERS_PER_KILOMETER
    reference = reference._replace(
        vz=jnp.full_like(reference.vz, boundary_velocity_m_s)
    )
    reference_cube = reference_cube._replace(
        vz=jnp.full_like(reference_cube.vz, boundary_velocity_m_s)
    )
    truth = build_contrast_atmosphere(
        reference_cube,
        envelope,
        thermodynamic_envelope=thermodynamic_envelope,
        temperature_center_ratio=temperature_center_ratio,
        ne_center_ratio=ne_center_ratio,
        boundary_velocity_km_s=boundary_velocity_km_s,
        center_velocity_km_s=center_velocity_km_s,
        boundary_b_los_gauss=boundary_b_los_gauss,
        center_b_los_gauss=center_b_los_gauss,
        field_strength_gauss=field_strength_gauss,
    )
    pinn3d.validate_neural_transform_domain(reference)
    pinn3d.validate_neural_transform_domain(truth)
    required_scale, selected_scale = select_spatial_scale(
        reference,
        truth,
        explicit_scale=spatial_scale,
        safety_factor=scale_safety_factor,
        padding=scale_padding,
    )
    field_config = pinn3d.NeuralFieldConfig(
        spatial_hidden=spatial_hidden,
        spatial_scale=selected_scale,
    )
    field_config.validate()

    observed = pinn3d.synthesize_atmosphere_cube(
        lines,
        wavelengths,
        dz,
        truth,
        batch_columns=synthesis_batch_columns,
        show_progress=show_progress,
        description="Synthesizing contrast truth cube",
        wavelength_parallelism=wavelength_parallelism,
    )
    payload = {
        "schema_version": np.asarray(pinn3d.SCHEMA_VERSION),
        "contrast_schema_version": np.asarray(3),
        "correction_parameterization": np.asarray(pinn3d.PARAMETERIZATION),
        "stokes_labels": np.asarray(pinn3d.STOKES_LABELS),
        "x_normalized": np.asarray(x),
        "y_normalized": np.asarray(y),
        "height_m": np.asarray(height),
        "height_normalized": np.asarray(z_normalized),
        "cell_width_m": np.asarray(dz),
        "wavelength_nm": np.asarray(wavelengths),
        "line_center_nm": pinn3d._canonical_line_centers(lines),
        "atomic_data_sha256": np.asarray(pinn3d.atomic_data_fingerprint(lines)),
        "synthesis_precision": np.asarray(adora_precision.configured_precision()),
        "observed_stokes": np.asarray(observed),
        "perturbation_envelope": np.asarray(envelope),
        "thermodynamic_envelope": np.asarray(thermodynamic_envelope),
        "thermodynamic_vertical_envelope": np.asarray(vertical_envelope),
        "thermodynamic_mode": np.asarray(thermodynamic_mode),
        "requested_thermodynamic_height_m": np.asarray(
            math.nan
            if thermodynamic_height_km is None
            else thermodynamic_height_km * METERS_PER_KILOMETER
        ),
        "effective_thermodynamic_height_m": np.asarray(
            effective_thermodynamic_height_m
        ),
        "thermodynamic_target_index": np.asarray(thermodynamic_target_index),
        "thermodynamic_sigma_m": np.asarray(
            thermodynamic_sigma_km * METERS_PER_KILOMETER
        ),
        "temperature_boundary_anchor_K": np.asarray(
            math.nan if temperature_boundary_k is None else temperature_boundary_k
        ),
        "contrast_profile": np.asarray("center_boundary_polynomial"),
        "profile_power": np.asarray(profile_power),
        "generation_wavelength_padding_nm": np.asarray(wavelength_padding_nm),
        "generation_synthesis_batch_columns": np.asarray(synthesis_batch_columns),
        "generation_wavelength_parallelism": np.asarray(wavelength_parallelism),
        "temperature_center_ratio": np.asarray(temperature_center_ratio),
        "electron_density_center_ratio": np.asarray(ne_center_ratio),
        "boundary_velocity_m_s": np.asarray(boundary_velocity_m_s),
        "center_velocity_m_s": np.asarray(center_velocity_km_s * METERS_PER_KILOMETER),
        "boundary_b_los_T": np.asarray(boundary_b_los_gauss / GAUSS_PER_TESLA),
        "center_b_los_T": np.asarray(center_b_los_gauss / GAUSS_PER_TESLA),
        "magnetic_field_strength_T": np.asarray(field_strength_t),
        "magnetic_azimuth_rad": np.asarray(reference.chi_b[0]),
        "requested_magnetic_azimuth_rad": np.asarray(
            math.radians(magnetic_azimuth_deg)
        ),
        "required_spatial_scale": np.asarray(required_scale),
        "suggested_spatial_scale": np.asarray(selected_scale),
        "spatial_scale_mode": np.asarray(
            "automatic" if spatial_scale is None else "explicit"
        ),
        "spatial_scale_safety_factor": np.asarray(scale_safety_factor),
        "spatial_scale_padding": np.asarray(scale_padding),
        "initial_spatial_hidden": np.asarray(spatial_hidden, dtype=np.int64),
        "checkpoint_seed": np.asarray(seed),
        **pinn3d._atmosphere_payload("reference_", reference),
        **pinn3d._atmosphere_payload("truth_", truth),
    }
    pinn3d.save_test_cube(dataset_path, payload)

    initial_params = pinn3d.initialize_neural_field(
        jax.random.PRNGKey(seed),
        reference,
        z_normalized,
        field_config,
    )
    pinn3d.save_checkpoint(checkpoint_path, initial_params, field_config)
    return payload, field_config


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the documented command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a full-Stokes FAL-C test cube with exact horizontal "
            "center/boundary contrasts and a compatible PINN checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "The horizontal envelope is "
            "[4*x*(1-x)*4*y*(1-y)]**profile_power, equals zero on every "
            "boundary, and equals one at x=y=0.5. Temperature and electron "
            "density can additionally be localized in height. Magnetic inputs "
            "describe signed B_LOS; the stored field strength is positive and "
            "B_LOS=B*cos(inclination)."
        ),
    )

    outputs = parser.add_argument_group("outputs and runtime")
    outputs.add_argument(
        "--precision",
        choices=adora_precision.SUPPORTED_PRECISIONS,
        default=adora_precision.configured_precision(),
        help="JAX compute precision selected before importing the physics code",
    )
    outputs.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="output NPZ containing truth atmosphere and synthesized Stokes cube",
    )
    outputs.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="initial neural-field checkpoint with compatible correction scales",
    )
    outputs.add_argument(
        "--kurucz",
        type=Path,
        default=FE_I_6301_6302_LINE_LIST,
        help="Kurucz-format line list used for synthesis and later inversion",
    )
    outputs.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed used only to initialize the saved neural checkpoint",
    )
    outputs.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing dataset and checkpoint files",
    )
    outputs.add_argument(
        "--no-progress",
        action="store_true",
        help="disable synthesis progress bars",
    )

    grid = parser.add_argument_group("spatial and spectral grid")
    grid.add_argument(
        "--nx",
        type=_odd_grid_size,
        default=DEFAULT_NX,
        help="odd x-grid size; oddness puts x=0.5 exactly on the grid",
    )
    grid.add_argument(
        "--ny",
        type=_odd_grid_size,
        default=DEFAULT_NY,
        help="odd y-grid size; oddness puts y=0.5 exactly on the grid",
    )
    grid.add_argument(
        "--n-wave",
        type=_positive_integer,
        default=DEFAULT_N_WAVE,
        help="number of wavelengths including all exact line centers",
    )
    grid.add_argument(
        "--wavelength-padding-nm",
        type=_positive_float,
        default=0.05,
        help="wavelength coverage beyond the outermost line center in nm",
    )
    grid.add_argument(
        "--synthesis-batch-columns",
        type=_positive_integer,
        default=16,
        help="horizontal columns synthesized concurrently",
    )
    grid.add_argument(
        "--wavelength-parallelism",
        type=_positive_integer,
        default=pinn3d.DEFAULT_WAVELENGTH_PARALLELISM,
        help="wavelengths evaluated concurrently in each synthesis chunk",
    )
    grid.add_argument(
        "--profile-power",
        type=_positive_float,
        default=DEFAULT_PROFILE_POWER,
        help="positive exponent controlling concentration toward the center",
    )

    thermodynamics = parser.add_argument_group("thermodynamic and velocity contrast")
    thermodynamics.add_argument(
        "--thermodynamic-height-km",
        type=_finite_float,
        help=(
            "localize temperature and electron-density contrasts around the "
            "nearest FAL-C layer to this height in km"
        ),
    )
    thermodynamics.add_argument(
        "--thermodynamic-sigma-km",
        type=_positive_float,
        default=DEFAULT_THERMODYNAMIC_SIGMA_KM,
        help="Gaussian height width for localized thermodynamic contrasts in km",
    )
    thermodynamics.add_argument(
        "--temperature-boundary-k",
        type=_positive_float,
        help=(
            "exact boundary temperature on the selected thermodynamic layer; "
            "requires --thermodynamic-height-km"
        ),
    )
    thermodynamics.add_argument(
        "--temperature-center-ratio",
        type=_positive_float,
        default=DEFAULT_TEMPERATURE_CENTER_RATIO,
        help="center temperature divided by boundary temperature",
    )
    thermodynamics.add_argument(
        "--ne-center-ratio",
        type=_positive_float,
        default=DEFAULT_NE_CENTER_RATIO,
        help="center electron density divided by boundary FAL-C electron density",
    )
    thermodynamics.add_argument(
        "--boundary-velocity-km-s",
        type=_finite_float,
        default=DEFAULT_BOUNDARY_VELOCITY_KM_S,
        help="LOS velocity on every horizontal boundary in km/s",
    )
    thermodynamics.add_argument(
        "--center-velocity-km-s",
        type=_finite_float,
        default=DEFAULT_CENTER_VELOCITY_KM_S,
        help="LOS velocity at x=y=0.5 in km/s",
    )

    magnetic = parser.add_argument_group("magnetic geometry")
    magnetic.add_argument(
        "--boundary-b-los-gauss",
        type=_finite_float,
        default=DEFAULT_BOUNDARY_B_LOS_GAUSS,
        help="signed LOS magnetic component on every horizontal boundary in G",
    )
    magnetic.add_argument(
        "--center-b-los-gauss",
        type=_finite_float,
        default=DEFAULT_CENTER_B_LOS_GAUSS,
        help="signed LOS magnetic component at x=y=0.5 in G",
    )
    magnetic.add_argument(
        "--field-strength-gauss",
        type=_positive_float,
        default=DEFAULT_FIELD_STRENGTH_GAUSS,
        help=(
            "constant unsigned |B| in G; must strictly exceed both absolute "
            "LOS endpoints"
        ),
    )
    magnetic.add_argument(
        "--magnetic-azimuth-deg",
        type=_finite_float,
        default=0.0,
        help="constant magnetic azimuth in degrees, wrapped modulo 180 degrees",
    )

    checkpoint = parser.add_argument_group("checkpoint architecture and reach")
    checkpoint.add_argument(
        "--spatial-hidden",
        type=_hidden_layers,
        default=(96, 96, 96),
        metavar="WIDTH[,WIDTH...]",
        help="hidden widths of the three-dimensional correction MLP",
    )
    checkpoint.add_argument(
        "--spatial-scale",
        type=_spatial_scale,
        metavar="T,NE,NH,V,VT,B,GAMMA,CHI",
        help=(
            "explicit eight-channel reference-relative correction limits; when omitted "
            "they are derived from the truth cube"
        ),
    )
    checkpoint.add_argument(
        "--spatial-scale-safety-factor",
        type=_greater_than_one_float,
        default=DEFAULT_SCALE_SAFETY_FACTOR,
        help="factor applied to each required reference-relative correction in auto mode",
    )
    checkpoint.add_argument(
        "--spatial-scale-padding",
        type=_nonnegative_float,
        default=DEFAULT_SCALE_PADDING,
        help="additive correction margin applied after the auto safety factor",
    )
    return parser


def main(argv=None):
    """Create the configured test set and its initialization checkpoint."""

    args = build_argument_parser().parse_args(argv)
    active_precision = adora_precision.configured_precision()
    if args.precision != active_precision:
        raise ValueError(
            f"requested --precision {args.precision}, but this Python process "
            f"already imported Adora in {active_precision}; start a fresh "
            "process or set ADORA_PRECISION before importing this module"
        )
    dataset_path = args.dataset.expanduser()
    checkpoint_path = args.checkpoint.expanduser()
    payload, field_config = create_test_set(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        kurucz_path=args.kurucz,
        nx=args.nx,
        ny=args.ny,
        n_wave=args.n_wave,
        wavelength_padding_nm=args.wavelength_padding_nm,
        synthesis_batch_columns=args.synthesis_batch_columns,
        wavelength_parallelism=args.wavelength_parallelism,
        profile_power=args.profile_power,
        thermodynamic_height_km=args.thermodynamic_height_km,
        thermodynamic_sigma_km=args.thermodynamic_sigma_km,
        temperature_boundary_k=args.temperature_boundary_k,
        temperature_center_ratio=args.temperature_center_ratio,
        ne_center_ratio=args.ne_center_ratio,
        boundary_velocity_km_s=args.boundary_velocity_km_s,
        center_velocity_km_s=args.center_velocity_km_s,
        boundary_b_los_gauss=args.boundary_b_los_gauss,
        center_b_los_gauss=args.center_b_los_gauss,
        field_strength_gauss=args.field_strength_gauss,
        magnetic_azimuth_deg=args.magnetic_azimuth_deg,
        spatial_hidden=args.spatial_hidden,
        spatial_scale=args.spatial_scale,
        scale_safety_factor=args.spatial_scale_safety_factor,
        scale_padding=args.spatial_scale_padding,
        seed=args.seed,
        show_progress=not args.no_progress,
        overwrite=args.overwrite,
    )
    print(f"Created dataset: {dataset_path}")
    print(f"Created checkpoint: {checkpoint_path}")
    print(
        "Checkpoint spatial scale: "
        + ",".join(f"{value:.8g}" for value in field_config.spatial_scale)
    )
    if str(np.asarray(payload["thermodynamic_mode"])) == "height_localized_gaussian":
        requested_height_km = (
            float(np.asarray(payload["requested_thermodynamic_height_m"]))
            / METERS_PER_KILOMETER
        )
        effective_height_km = (
            float(np.asarray(payload["effective_thermodynamic_height_m"]))
            / METERS_PER_KILOMETER
        )
        target_index = int(np.asarray(payload["thermodynamic_target_index"]))
        print(
            f"Thermodynamic target: requested {requested_height_km:.6g} km; "
            f"using layer {target_index} at {effective_height_km:.6g} km"
        )
    return None


if __name__ == "__main__":
    main()
