"""Compare FP32 and FP64 PINN inversions in isolated JAX processes.

JAX precision is process-global and must be selected before importing JAX.
Consequently, the public command in this module deliberately does not import
JAX or :mod:`pinn_3d_inversion`.  It starts one fresh worker for each precision
and combines their numerical artifacts into a JSON report.

The two measured workers load the same initial checkpoint and use identical
NumPy RNG seeds.  They report synchronized first-call and cached timings for a
representative spectrum and gradient, complete convergence histories, final
spectra, and recovered atmospheric fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = 1
PRECISIONS = ("fp64", "fp32")
DEFAULT_DATASET = Path("data") / "pinn_3d_test_cube.npz"
DEFAULT_OUTPUT = Path("data") / "pinn_precision_comparison.json"
DEFAULT_KURUCZ = (
    Path(__file__).resolve().parent / "adora_data" / "kurucz_6301_6302.linelist"
)
ATMOSPHERE_FIELDS = (
    "temperature",
    "ne",
    "nhtot",
    "vz",
    "vturb",
    "b",
    "gamma_b",
    "chi_b",
)
STOKES_LABELS = ("I", "Q", "U", "V")
_INTERNAL_WORKER_MODES = ("initialize", "compare")


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


def _stokes_weights(value: str) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Stokes weights must be numeric") from exc
    if len(result) != 4 or any(not math.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "Stokes weights must be four positive comma-separated values"
        )
    return result


def _optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off", "disabled"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive number or 'none'"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def _spatial_scale(value: str) -> tuple[float, ...]:
    try:
        scales = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "spatial scale must be eight positive finite values"
        ) from exc
    if len(scales) != 8 or any(not math.isfinite(item) or item <= 0 for item in scales):
        raise argparse.ArgumentTypeError(
            "spatial scale must be eight positive finite values"
        )
    return scales


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the public precision-comparison command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the same PINN inversion in fresh FP64 and FP32 processes and "
            "write a machine-readable accuracy/performance comparison."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--kurucz", type=Path, default=DEFAULT_KURUCZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="directory for per-precision arrays (defaults beside --output)",
    )
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        help=(
            "optional shared starting checkpoint; otherwise one deterministic "
            "FP64 checkpoint is created before both measured runs"
        ),
    )
    parser.add_argument("--spatial-hidden", type=_hidden_layers, default=(96, 96, 96))
    parser.add_argument("--spatial-scale", type=_spatial_scale, default=(1.0,) * 8)
    parser.add_argument("--inversion-epochs", type=_nonnegative_integer, default=10)
    parser.add_argument("--inversion-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--inversion-final-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--training-batch-columns", type=_positive_integer, default=8)
    parser.add_argument("--validation-columns", type=_positive_integer, default=8)
    parser.add_argument("--wavelength-batch", type=_positive_integer, default=48)
    parser.add_argument("--wavelength-parallelism", type=_positive_integer, default=48)
    parser.add_argument("--synthesis-batch-columns", type=_positive_integer, default=16)
    parser.add_argument(
        "--stokes-weights",
        type=_stokes_weights,
        default=(1.0, 5.0, 5.0, 2.0),
        metavar="I,Q,U,V",
    )
    parser.add_argument("--prior-weight", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timing-repeats", type=_positive_integer, default=3)
    parser.add_argument(
        "--platform",
        choices=("gpu", "cpu", "auto"),
        default="gpu",
        help="JAX platform required in each worker (default: gpu)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU comparison (GPU execution is required by default)",
    )
    parser.add_argument(
        "--require-device-substring",
        default="",
        help="fail unless the selected JAX device name contains this text",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report and its named per-run artifacts",
    )
    return parser


def _build_internal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_worker-mode", choices=_INTERNAL_WORKER_MODES, required=True)
    parser.add_argument("--_precision", choices=PRECISIONS, required=True)
    parser.add_argument("--_config", type=Path, required=True)
    parser.add_argument("--_output", type=Path, required=True)
    parser.add_argument("--_summary", type=Path)
    return parser


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_ready(payload), indent=2, sort_keys=True, allow_nan=False
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_arrays(named_arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _worker_command(
    mode: str,
    precision: str,
    config_path: Path,
    output_path: Path,
    summary_path: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-mode",
        mode,
        "--_precision",
        precision,
        "--_config",
        str(config_path),
        "--_output",
        str(output_path),
    ]
    if summary_path is not None:
        command.extend(("--_summary", str(summary_path)))
    return command


def _invoke_worker(
    mode: str,
    precision: str,
    config_path: Path,
    output_path: Path,
    summary_path: Path | None,
    platform: str,
) -> None:
    """Start a worker with precision configured before any JAX import."""

    environment = os.environ.copy()
    environment["ADORA_PRECISION"] = precision
    if platform == "auto":
        environment.pop("JAX_PLATFORMS", None)
    else:
        # JAX reports the backend as ``gpu`` but registers CUDA under the
        # ``cuda`` platform name used by JAX_PLATFORMS.
        environment["JAX_PLATFORMS"] = "cuda" if platform == "gpu" else platform
    command = _worker_command(mode, precision, config_path, output_path, summary_path)
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{precision} {mode} worker failed with exit code "
            f"{completed.returncode}:\n{details}"
        )


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    result = numerator / denominator
    return float(result) if math.isfinite(result) else None


def array_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Return stable FP64-reference error metrics for two equal-shaped arrays."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"comparison shape mismatch: {reference.shape} != {candidate.shape}"
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("comparison arrays must contain only finite values")
    difference = candidate - reference
    reference_l2 = float(np.linalg.norm(reference.ravel()))
    difference_l2 = float(np.linalg.norm(difference.ravel()))
    reference_max = float(np.max(np.abs(reference))) if reference.size else 0.0
    max_absolute = float(np.max(np.abs(difference))) if difference.size else 0.0
    return {
        "shape": list(reference.shape),
        "elements": int(reference.size),
        "max_absolute": max_absolute,
        "mean_absolute": (
            float(np.mean(np.abs(difference))) if difference.size else 0.0
        ),
        "rmse": (
            float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0
        ),
        "relative_l2": _finite_ratio(difference_l2, reference_l2),
        "normalized_max": _finite_ratio(max_absolute, reference_max),
    }


def _gradient_comparison(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    result = array_comparison(reference, candidate)
    reference64 = np.asarray(reference, dtype=np.float64).ravel()
    candidate64 = np.asarray(candidate, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(reference64) * np.linalg.norm(candidate64))
    cosine = _finite_ratio(float(np.dot(reference64, candidate64)), denominator)
    result.update(
        {
            "fp64_l2_norm": float(np.linalg.norm(reference64)),
            "fp32_l2_norm": float(np.linalg.norm(candidate64)),
            "cosine_similarity": cosine,
        }
    )
    return result


def _spectral_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    continuum: np.ndarray,
    stokes_axis: int,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    continuum = np.asarray(continuum, dtype=np.float64)
    if (
        reference.shape != candidate.shape
        or reference.shape[stokes_axis] != 4
        or stokes_axis != reference.ndim - 2
        or continuum.shape != reference.shape[:-2]
    ):
        raise ValueError("spectral arrays must have matching four-component axes")
    scale = np.maximum(np.abs(continuum), np.finfo(np.float64).tiny)[..., None, None]
    normalized_difference = (candidate - reference) / scale
    result = array_comparison(reference, candidate)
    result.update(
        {
            "continuum_normalized_max_absolute": float(
                np.max(np.abs(normalized_difference))
            ),
            "continuum_normalized_rmse": float(
                np.sqrt(np.mean(np.square(normalized_difference)))
            ),
            "continuum_denominator": (
                "absolute Stokes-I mean at the first and last observed wavelengths"
            ),
            "per_stokes": {},
        }
    )
    for index, label in enumerate(STOKES_LABELS):
        component_difference = np.take(candidate - reference, index, axis=stokes_axis)
        normalized_component = component_difference / scale[..., 0, :]
        component = array_comparison(
            np.take(reference, index, axis=stokes_axis),
            np.take(candidate, index, axis=stokes_axis),
        )
        component.update(
            {
                "absolute_rmse": float(
                    np.sqrt(np.mean(np.square(component_difference)))
                ),
                "continuum_normalized_rmse": float(
                    np.sqrt(np.mean(np.square(normalized_component)))
                ),
            }
        )
        result["per_stokes"][label] = component
    return result


def _timing_speedups(
    fp64: dict[str, Any], fp32: dict[str, Any]
) -> dict[str, float | None]:
    common = sorted(set(fp64).intersection(fp32))
    result: dict[str, float | None] = {}
    for name in common:
        first = fp64[name]
        second = fp32[name]
        if (
            isinstance(first, (int, float))
            and not isinstance(first, bool)
            and isinstance(second, (int, float))
            and not isinstance(second, bool)
        ):
            result[name] = _finite_ratio(float(first), float(second))
    return result


def build_comparison_report(
    config: dict[str, Any],
    artifact_paths: dict[str, Path],
    summary_paths: dict[str, Path],
    initial_checkpoint: Path,
) -> dict[str, Any]:
    """Combine worker artifacts and validate the reproducibility contract."""

    summaries = {
        precision: _read_json(summary_paths[precision]) for precision in PRECISIONS
    }
    archives: dict[str, dict[str, np.ndarray]] = {}
    for precision in PRECISIONS:
        with np.load(artifact_paths[precision], allow_pickle=False) as archive:
            archives[precision] = {
                name: np.asarray(archive[name]) for name in archive.files
            }

    schedule_hashes = {
        precision: summaries[precision].get("schedule_sha256")
        for precision in PRECISIONS
    }
    if len(set(schedule_hashes.values())) != 1 or None in schedule_hashes.values():
        raise ValueError(
            f"precision workers used different schedules: {schedule_hashes}"
        )
    for name in (
        "diagnostic_column_indices",
        "diagnostic_wavelength_indices",
        "diagnostic_column_mask",
    ):
        if not np.array_equal(archives["fp64"][name], archives["fp32"][name]):
            raise ValueError(f"precision workers used different {name}")

    fp64 = archives["fp64"]
    fp32 = archives["fp32"]
    convergence = {
        "inversion": {
            "columns": ["total", "spectral", "prior"],
            "fp64": fp64["inversion_loss"].tolist(),
            "fp32": fp32["inversion_loss"].tolist(),
            "difference": array_comparison(
                fp64["inversion_loss"], fp32["inversion_loss"]
            ),
        },
        "validation": {
            "fp64": fp64["validation_loss"].tolist(),
            "fp32": fp32["validation_loss"].tolist(),
            "difference": array_comparison(
                fp64["validation_loss"], fp32["validation_loss"]
            ),
        },
    }
    recovered = {
        name: array_comparison(fp64[f"recovered_{name}"], fp32[f"recovered_{name}"])
        for name in ATMOSPHERE_FIELDS
    }
    initial_spectra = _spectral_comparison(
        fp64["diagnostic_spectra"],
        fp32["diagnostic_spectra"],
        fp64["diagnostic_continuum"],
        stokes_axis=1,
    )
    final_spectra = _spectral_comparison(
        fp64["final_synthetic_stokes"],
        fp32["final_synthetic_stokes"],
        fp64["final_continuum"],
        stokes_axis=2,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_precision": "fp64",
        "candidate_precision": "fp32",
        "config": _json_ready(config),
        "reproducibility": {
            "initial_checkpoint": str(initial_checkpoint.resolve()),
            "initial_checkpoint_sha256": _file_sha256(initial_checkpoint),
            "initialization_seed": int(config["seed"]),
            "inversion_schedule_seed": int(config["seed"]) + 1,
            "schedule_sha256": schedule_hashes["fp64"],
            "schedule_match": True,
        },
        "runs": summaries,
        "metric_definitions": {
            "speedup_fp64_over_fp32": (
                "FP64 elapsed seconds / FP32 elapsed seconds; values above one "
                "mean FP32 is faster"
            ),
            "relative_l2": "L2(FP32 - FP64) / L2(FP64)",
            "normalized_max": "max_abs(FP32 - FP64) / max_abs(FP64)",
            "gradient_cosine_similarity": ("dot(FP64, FP32) / (L2(FP64) * L2(FP32))"),
        },
        "speedup_fp64_over_fp32": _timing_speedups(
            summaries["fp64"]["timings_seconds"],
            summaries["fp32"]["timings_seconds"],
        ),
        "comparisons": {
            "spectra": {
                "before_inversion": initial_spectra,
                "after_inversion": final_spectra,
            },
            "gradient_before_inversion": _gradient_comparison(
                fp64["diagnostic_gradient"], fp32["diagnostic_gradient"]
            ),
            "shared_initial_parameters_after_precision_cast": array_comparison(
                fp64["initial_parameters"], fp32["initial_parameters"]
            ),
            "gradient_loss_before_inversion": {
                "fp64": float(fp64["diagnostic_loss"]),
                "fp32": float(fp32["diagnostic_loss"]),
                "difference": array_comparison(
                    fp64["diagnostic_loss"], fp32["diagnostic_loss"]
                ),
            },
            "convergence": convergence,
            "recovered_atmosphere": recovered,
        },
        "artifacts": {
            precision: str(artifact_paths[precision].resolve())
            for precision in PRECISIONS
        },
    }


def _public_config(args: argparse.Namespace) -> dict[str, Any]:
    numeric_positive = (
        "inversion_learning_rate",
        "inversion_final_learning_rate",
    )
    for name in numeric_positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
    if not math.isfinite(args.prior_weight) or args.prior_weight < 0:
        raise ValueError("--prior-weight must be non-negative and finite")
    if args.inversion_final_learning_rate > args.inversion_learning_rate:
        raise ValueError(
            "--inversion-final-learning-rate must not exceed --inversion-learning-rate"
        )
    if args.platform == "cpu" and not args.allow_cpu:
        raise ValueError("--platform cpu requires the explicit --allow-cpu flag")
    dataset = args.dataset.resolve()
    kurucz = args.kurucz.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset does not exist: {dataset}")
    if not kurucz.is_file():
        raise FileNotFoundError(f"Kurucz line list does not exist: {kurucz}")
    if args.checkpoint_in is not None and not args.checkpoint_in.resolve().is_file():
        raise FileNotFoundError(
            f"starting checkpoint does not exist: {args.checkpoint_in.resolve()}"
        )
    return {
        "dataset": str(dataset),
        "kurucz": str(kurucz),
        "spatial_hidden": list(args.spatial_hidden),
        "spatial_scale": list(args.spatial_scale),
        "inversion_epochs": args.inversion_epochs,
        "inversion_learning_rate": args.inversion_learning_rate,
        "inversion_final_learning_rate": args.inversion_final_learning_rate,
        "training_batch_columns": args.training_batch_columns,
        "validation_columns": args.validation_columns,
        "wavelength_batch": args.wavelength_batch,
        "wavelength_parallelism": args.wavelength_parallelism,
        "synthesis_batch_columns": args.synthesis_batch_columns,
        "stokes_weights": list(args.stokes_weights),
        "prior_weight": args.prior_weight,
        "seed": args.seed,
        "timing_repeats": args.timing_repeats,
        "platform": args.platform,
        "allow_cpu": args.allow_cpu,
        "require_device_substring": args.require_device_substring,
    }


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run both precisions and return the combined report."""

    config = _public_config(args)
    output_path = args.output.resolve()
    artifacts_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else output_path.with_suffix("").with_name(output_path.stem + "_artifacts")
    )
    config_path = artifacts_dir / "config.json"
    initial_checkpoint = (
        args.checkpoint_in.resolve()
        if args.checkpoint_in is not None
        else artifacts_dir / "shared_initial_checkpoint.npz"
    )
    artifact_paths = {
        precision: artifacts_dir / f"{precision}_diagnostics.npz"
        for precision in PRECISIONS
    }
    summary_paths = {
        precision: artifacts_dir / f"{precision}_summary.json"
        for precision in PRECISIONS
    }
    production_paths = [
        artifacts_dir / f"{precision}_{suffix}.npz"
        for precision in PRECISIONS
        for suffix in ("inversion_result", "final_checkpoint")
    ]
    generated_paths = [
        output_path,
        config_path,
        *artifact_paths.values(),
        *summary_paths.values(),
        *production_paths,
    ]
    if args.checkpoint_in is None:
        generated_paths.append(initial_checkpoint)
    existing = [path for path in generated_paths if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists (use --overwrite): {joined}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config["shared_initial_checkpoint"] = str(initial_checkpoint)
    _write_json(config_path, config)
    if args.checkpoint_in is None:
        _invoke_worker(
            "initialize",
            "fp64",
            config_path,
            initial_checkpoint,
            None,
            args.platform,
        )
    for precision in PRECISIONS:
        _invoke_worker(
            "compare",
            precision,
            config_path,
            artifact_paths[precision],
            summary_paths[precision],
            args.platform,
        )
    report = build_comparison_report(
        config,
        artifact_paths,
        summary_paths,
        initial_checkpoint,
    )
    _write_json(output_path, report)
    return report


def _synchronize(jax_module: Any, value: Any) -> Any:
    for leaf in jax_module.tree.leaves(value):
        blocker = getattr(leaf, "block_until_ready", None)
        if blocker is not None:
            blocker()
    return value


def _benchmark(
    jax_module: Any, function: Any, repeats: int
) -> tuple[Any, dict[str, Any]]:
    import time

    started = time.perf_counter()
    result = _synchronize(jax_module, function())
    first = time.perf_counter() - started
    cached = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = _synchronize(jax_module, function())
        cached.append(time.perf_counter() - started)
    return result, {
        "first_call_seconds": first,
        "cached_seconds": cached,
        "cached_median_seconds": statistics.median(cached),
        "cached_min_seconds": min(cached),
    }


def _device_metadata(jax_module: Any) -> dict[str, Any]:
    devices = jax_module.local_devices()
    metadata = []
    for device in devices:
        entry: dict[str, Any] = {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        try:
            stats = device.memory_stats()
        except (AttributeError, RuntimeError):
            stats = None
        if stats:
            entry["memory_stats_bytes"] = {
                key: int(value)
                for key, value in stats.items()
                if isinstance(value, (int, np.integer))
                and ("byte" in key or "limit" in key)
            }
        metadata.append(entry)
    return {
        "jax_version": str(jax_module.__version__),
        "default_backend": str(jax_module.default_backend()),
        "devices": metadata,
    }


def _verify_worker_environment(
    precision: str, config: dict[str, Any], jax_module: Any
) -> None:
    selected = os.environ.get("ADORA_PRECISION")
    if selected != precision:
        raise RuntimeError(
            f"worker expected ADORA_PRECISION={precision}, found {selected!r}"
        )
    expected_x64 = precision == "fp64"
    actual_x64 = bool(jax_module.config.x64_enabled)
    if actual_x64 != expected_x64:
        raise RuntimeError(
            f"{precision} worker has jax_enable_x64={actual_x64}; precision must "
            "be configured before importing JAX"
        )
    expected_platform = config["platform"]
    if (
        expected_platform != "auto"
        and jax_module.default_backend() != expected_platform
    ):
        raise RuntimeError(
            f"requested JAX platform {expected_platform!r}, got "
            f"{jax_module.default_backend()!r}"
        )
    if jax_module.default_backend() == "cpu" and not config["allow_cpu"]:
        raise RuntimeError(
            "precision performance comparisons require a GPU; pass --allow-cpu "
            "only for testing or numerical checks"
        )
    required = config["require_device_substring"].lower()
    if required and not any(
        required in str(device.device_kind).lower()
        for device in jax_module.local_devices()
    ):
        kinds = ", ".join(
            str(device.device_kind) for device in jax_module.local_devices()
        )
        raise RuntimeError(
            f"no selected JAX device contains {required!r}; available: {kinds}"
        )


def _initialize_worker(config: dict[str, Any], output_path: Path) -> None:
    import jax

    import pinn_3d_inversion as pinn

    _verify_worker_environment("fp64", config, jax)
    field_config = pinn.NeuralFieldConfig(
        spatial_hidden=tuple(config["spatial_hidden"]),
        spatial_scale=tuple(config["spatial_scale"]),
    )
    payload = pinn.load_test_cube(config["dataset"])
    reference = pinn._atmosphere_from_payload(payload, "reference_")
    params = pinn.initialize_neural_field(
        jax.random.PRNGKey(int(config["seed"])),
        reference,
        payload["height_normalized"],
        field_config,
    )
    _synchronize(jax, params)
    pinn.save_checkpoint(output_path, params, field_config)


def _schedule_diagnostics(
    pinn: Any,
    n_columns: int,
    wavelengths: Any,
    config: dict[str, Any],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], str]:
    rng = np.random.default_rng(int(config["seed"]) + 1)
    hashed: list[tuple[str, np.ndarray]] = []
    first: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    # A diagnostic batch is needed even for a zero-epoch smoke comparison.
    epochs_to_build = max(1, int(config["inversion_epochs"]))
    for epoch in range(epochs_to_build):
        columns, waves, masks = pinn._spectral_epoch_batches(
            n_columns,
            int(config["training_batch_columns"]),
            wavelengths,
            int(config["wavelength_batch"]),
            rng,
        )
        if first is None:
            first = (columns[0], waves[0], masks[0])
        if epoch < int(config["inversion_epochs"]):
            hashed.extend(
                (
                    (f"epoch_{epoch}_columns", columns),
                    (f"epoch_{epoch}_wavelengths", waves),
                    (f"epoch_{epoch}_masks", masks),
                )
            )
    assert first is not None
    if not hashed:
        hashed.append(
            (
                "zero_epoch_seed",
                np.asarray((int(config["seed"]) + 1,), dtype=np.int64),
            )
        )
    return first, _hash_arrays(hashed)


def _comparison_worker(
    precision: str,
    config: dict[str, Any],
    output_path: Path,
    summary_path: Path,
) -> None:
    import time

    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    import pinn_3d_inversion as pinn
    from lineop import read_kurucz

    _verify_worker_environment(precision, config, jax)
    payload = pinn.load_test_cube(config["dataset"])
    nx, ny, n_depth, _ = pinn.validate_test_cube(payload)
    lines = read_kurucz(config["kurucz"])
    saved_centers = np.asarray(payload["line_center_nm"], dtype=np.float64)
    expected_centers = pinn._canonical_line_centers(lines)
    if saved_centers.shape != expected_centers.shape or not np.allclose(
        saved_centers,
        expected_centers,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError("the Kurucz data do not match the comparison dataset")
    if str(np.asarray(payload["atomic_data_sha256"])) != pinn.atomic_data_fingerprint(
        lines
    ):
        raise ValueError("the atomic-data fingerprint does not match the dataset")

    params, field_config = pinn.load_checkpoint(config["shared_initial_checkpoint"])
    initial_parameters, _ = ravel_pytree(params["spatial"])
    _synchronize(jax, initial_parameters)
    timings: dict[str, float] = {}
    coordinate_cube = pinn._coordinate_cube_from_payload(payload)
    coordinates = coordinate_cube.reshape((nx * ny, n_depth, 3))
    observed = pinn._as_real(payload["observed_stokes"]).reshape((nx * ny, 4, -1))
    absolute_wavelengths = pinn._host_wavelength_axis(payload["wavelength_nm"])
    wavelength_offsets_nm = pinn.centered_wavelengths(lines, absolute_wavelengths)
    dz = pinn._as_real(payload["cell_width_m"])
    spatial_scale = pinn._as_real(field_config.spatial_scale)
    continuum = pinn.continuum_normalization(observed)
    diagnostic_indices, schedule_hash = _schedule_diagnostics(
        pinn, nx * ny, absolute_wavelengths, config
    )
    column_indices, wavelength_indices, column_mask = diagnostic_indices
    batch_coordinates = coordinates[column_indices]
    batch_wavelength_offsets = wavelength_offsets_nm[wavelength_indices]
    batch_observed = jnp.take(
        jnp.take(observed, column_indices, axis=0), wavelength_indices, axis=2
    )
    batch_continuum = continuum[column_indices]
    batch_mask = pinn._as_real(column_mask)
    stokes_weights = pinn._as_real(config["stokes_weights"])

    def diagnostic_spectrum(candidate):
        atmosphere = pinn.evaluate_neural_field(
            candidate, batch_coordinates, spatial_scale
        )
        return pinn._synthesize_columns_offset_core(
            lines,
            batch_wavelength_offsets,
            dz,
            atmosphere,
            int(config["wavelength_parallelism"]),
        )

    compiled_spectrum = jax.jit(diagnostic_spectrum)
    diagnostic_spectra, spectrum_timing = _benchmark(
        jax,
        lambda: compiled_spectrum(params),
        int(config["timing_repeats"]),
    )
    timings.update(
        {
            "spectrum_first_call": spectrum_timing["first_call_seconds"],
            "spectrum_cached_median": spectrum_timing["cached_median_seconds"],
            "spectrum_cached_min": spectrum_timing["cached_min_seconds"],
        }
    )

    gradient_candidate = params["spatial"]
    reference = pinn.reference_at_coordinates(params, coordinates[0])

    def diagnostic_loss(candidate):
        return pinn.reference_spectral_loss_offset(
            candidate,
            reference,
            batch_coordinates,
            batch_wavelength_offsets,
            batch_observed,
            batch_continuum,
            dz,
            lines,
            spatial_scale,
            stokes_weights,
            batch_mask,
            float(config["prior_weight"]),
            int(config["wavelength_parallelism"]),
        )[0]

    compiled_gradient = jax.jit(jax.value_and_grad(diagnostic_loss))
    (diagnostic_loss_value, gradient_tree), gradient_timing = _benchmark(
        jax,
        lambda: compiled_gradient(gradient_candidate),
        int(config["timing_repeats"]),
    )
    diagnostic_gradient, _ = ravel_pytree(gradient_tree)
    _synchronize(jax, diagnostic_gradient)
    timings.update(
        {
            "gradient_first_call": gradient_timing["first_call_seconds"],
            "gradient_cached_median": gradient_timing["cached_median_seconds"],
            "gradient_cached_min": gradient_timing["cached_min_seconds"],
        }
    )

    # Both precisions start from the same spatial weights and fixed reference.
    # Use the production path for training, checkpoints and final diagnostics.
    production_result_path = output_path.with_name(f"{precision}_inversion_result.npz")
    production_checkpoint_path = output_path.with_name(
        f"{precision}_final_checkpoint.npz"
    )
    started = time.perf_counter()
    fitted, result = pinn.run_inversion(
        payload,
        kurucz_path=config["kurucz"],
        result_path=production_result_path,
        checkpoint_path=production_checkpoint_path,
        field_config=field_config,
        initial_params=params,
        inversion_epochs=int(config["inversion_epochs"]),
        inversion_learning_rate=float(config["inversion_learning_rate"]),
        inversion_final_learning_rate=float(config["inversion_final_learning_rate"]),
        training_batch_columns=int(config["training_batch_columns"]),
        wavelength_batch=int(config["wavelength_batch"]),
        synthesis_batch_columns=int(config["synthesis_batch_columns"]),
        stokes_weights=tuple(config["stokes_weights"]),
        prior_weight=float(config["prior_weight"]),
        seed=int(config["seed"]),
        show_progress=False,
        wavelength_parallelism=int(config["wavelength_parallelism"]),
        validation_columns=int(config["validation_columns"]),
    )
    _synchronize(jax, fitted)
    final_spectra = jnp.asarray(result["synthetic_stokes"])
    recovered = pinn._atmosphere_from_payload(result, "inferred_")
    _synchronize(jax, (final_spectra, recovered))
    timings["production_run_inversion"] = time.perf_counter() - started
    timings["total"] = timings["production_run_inversion"]
    inversion_history = np.asarray(result["inversion_loss_total_spectral_prior"])
    validation_history = np.asarray(result["validation_full_wavelength_loss"])

    observed_cube = np.asarray(payload["observed_stokes"])
    final_continuum = 0.5 * (observed_cube[:, :, 0, 0] + observed_cube[:, :, 0, -1])

    arrays: dict[str, np.ndarray] = {
        "diagnostic_column_indices": np.asarray(column_indices),
        "diagnostic_wavelength_indices": np.asarray(wavelength_indices),
        "diagnostic_column_mask": np.asarray(column_mask),
        "diagnostic_spectra": np.asarray(diagnostic_spectra),
        "diagnostic_continuum": np.asarray(batch_continuum),
        "diagnostic_loss": np.asarray(diagnostic_loss_value),
        "diagnostic_gradient": np.asarray(diagnostic_gradient),
        "initial_parameters": np.asarray(initial_parameters),
        "inversion_loss": np.asarray(inversion_history),
        "validation_loss": np.asarray(validation_history),
        "final_synthetic_stokes": np.asarray(final_spectra),
        "final_continuum": np.asarray(final_continuum),
    }
    arrays.update(
        {
            f"recovered_{name}": np.asarray(field)
            for name, field in zip(ATMOSPHERE_FIELDS, recovered)
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    parameter_dtype = str(np.asarray(jax.tree.leaves(params)[0]).dtype)
    summary = {
        "precision": precision,
        "parameter_dtype": parameter_dtype,
        "atomic_float_dtype": str(np.asarray(lines.lambda0).dtype),
        "spectrum_dtype": str(arrays["diagnostic_spectra"].dtype),
        "gradient_dtype": str(arrays["diagnostic_gradient"].dtype),
        "gradient_elements": int(arrays["diagnostic_gradient"].size),
        "schedule_sha256": schedule_hash,
        "timings_seconds": timings,
        "timing_samples_seconds": {
            "spectrum_cached": spectrum_timing["cached_seconds"],
            "gradient_cached": gradient_timing["cached_seconds"],
        },
        "device": _device_metadata(jax),
        "final_validation_loss": float(np.asarray(validation_history)[-1]),
        "production_result": str(production_result_path.resolve()),
        "production_checkpoint": str(production_checkpoint_path.resolve()),
    }
    _write_json(summary_path, summary)


def _worker_main(argv: Sequence[str]) -> None:
    args = _build_internal_parser().parse_args(argv)
    # The parent also sets this in the child environment.  Assigning it here
    # protects direct/manual worker invocations, and still happens before JAX.
    os.environ["ADORA_PRECISION"] = args._precision
    config = _read_json(args._config)
    if config["platform"] == "auto":
        os.environ.pop("JAX_PLATFORMS", None)
    else:
        os.environ["JAX_PLATFORMS"] = (
            "cuda" if config["platform"] == "gpu" else config["platform"]
        )
    if args._worker_mode == "initialize":
        if args._precision != "fp64":
            raise ValueError("shared checkpoint initialization must use fp64")
        _initialize_worker(config, args._output)
        return
    if args._summary is None:
        raise ValueError("comparison workers require --_summary")
    _comparison_worker(
        args._precision,
        config,
        args._output,
        args._summary,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a worker or the public two-precision comparison."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--_worker-mode" in arguments:
        _worker_main(arguments)
        return 0
    args = build_argument_parser().parse_args(arguments)
    report = run_comparison(args)
    concise = {
        "report": str(args.output.resolve()),
        "speedup_fp64_over_fp32": report["speedup_fp64_over_fp32"],
        "final_spectra_relative_l2": report["comparisons"]["spectra"][
            "after_inversion"
        ]["relative_l2"],
        "gradient_relative_l2": report["comparisons"]["gradient_before_inversion"][
            "relative_l2"
        ],
    }
    print(json.dumps(concise, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
