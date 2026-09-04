import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import compare_pinn_precision as comparison


def _write_worker_fixture(tmp_path, precision, schedule="same-schedule"):
    artifact = tmp_path / f"{precision}.npz"
    summary = tmp_path / f"{precision}.json"
    offset = 0.0 if precision == "fp64" else 1.0
    diagnostic = np.arange(24.0).reshape(2, 4, 3) + 10.0
    final = np.arange(24.0).reshape(1, 2, 4, 3) + 10.0
    gradient = np.asarray((1.0, 2.0, 3.0))
    arrays = {
        "diagnostic_column_indices": np.asarray((1, 0)),
        "diagnostic_wavelength_indices": np.asarray((0, 2, 4)),
        "diagnostic_column_mask": np.asarray((1.0, 1.0)),
        "diagnostic_spectra": diagnostic + offset,
        "diagnostic_continuum": np.asarray((10.0, 20.0)),
        "diagnostic_loss": np.asarray(0.5 + 0.1 * offset),
        "diagnostic_gradient": gradient + 0.1 * offset,
        "initial_parameters": np.asarray((0.25, -0.5, 0.75)) + 0.01 * offset,
        "pretrain_loss": np.asarray((2.0, 1.0)) + 0.1 * offset,
        "inversion_loss": np.asarray(((0.8, 0.7, 0.1),)) + 0.01 * offset,
        "validation_loss": np.asarray((0.9, 0.7)) + 0.01 * offset,
        "final_synthetic_stokes": final + offset,
        "final_continuum": np.asarray(((10.0, 20.0),)),
    }
    arrays.update(
        {
            f"recovered_{name}": np.full((1, 2, 3), index + 1.0 + 0.1 * offset)
            for index, name in enumerate(comparison.ATMOSPHERE_FIELDS)
        }
    )
    np.savez(artifact, **arrays)
    summary.write_text(
        json.dumps(
            {
                "precision": precision,
                "schedule_sha256": schedule,
                "timings_seconds": {
                    "gradient_cached_median": 4.0 if precision == "fp64" else 1.0,
                    "total": 12.0 if precision == "fp64" else 2.0,
                },
                "device": {
                    "default_backend": "gpu",
                    "devices": [{"device_kind": "NVIDIA GeForce RTX 4090"}],
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact, summary


def test_array_comparison_uses_fp64_as_the_relative_denominator():
    actual = comparison.array_comparison(
        np.asarray((3.0, 4.0)), np.asarray((0.0, 8.0))
    )

    assert actual["max_absolute"] == pytest.approx(4.0)
    assert actual["rmse"] == pytest.approx(np.sqrt(12.5))
    assert actual["relative_l2"] == pytest.approx(1.0)
    assert actual["normalized_max"] == pytest.approx(1.0)


def test_report_contains_speed_accuracy_convergence_and_recovery(tmp_path):
    artifacts = {}
    summaries = {}
    for precision in comparison.PRECISIONS:
        artifacts[precision], summaries[precision] = _write_worker_fixture(
            tmp_path, precision
        )
    checkpoint = tmp_path / "initial.npz"
    checkpoint.write_bytes(b"shared-parameters")
    config = {"seed": 9, "platform": "gpu"}

    report = comparison.build_comparison_report(
        config, artifacts, summaries, checkpoint
    )

    assert report["reproducibility"]["schedule_match"] is True
    assert report["reproducibility"]["initialization_seed"] == 9
    assert report["reproducibility"]["inversion_schedule_seed"] == 10
    assert report["speedup_fp64_over_fp32"]["total"] == pytest.approx(6.0)
    assert report["speedup_fp64_over_fp32"][
        "gradient_cached_median"
    ] == pytest.approx(4.0)

    spectra = report["comparisons"]["spectra"]["before_inversion"]
    assert spectra["continuum_normalized_max_absolute"] == pytest.approx(0.1)
    assert set(spectra["per_stokes"]) == set(comparison.STOKES_LABELS)
    assert spectra["per_stokes"]["I"]["absolute_rmse"] == pytest.approx(1.0)

    gradient = report["comparisons"]["gradient_before_inversion"]
    assert 0.99 < gradient["cosine_similarity"] <= 1.0
    assert gradient["relative_l2"] > 0.0
    convergence = report["comparisons"]["convergence"]
    assert convergence["pretraining"]["fp64"] == [2.0, 1.0]
    assert convergence["inversion"]["columns"] == ["total", "spectral", "prior"]
    recovered = report["comparisons"]["recovered_atmosphere"]
    assert set(recovered) == set(comparison.ATMOSPHERE_FIELDS)
    assert all("relative_l2" in values for values in recovered.values())


def test_report_rejects_different_precision_schedules(tmp_path):
    fp64_artifact, fp64_summary = _write_worker_fixture(tmp_path, "fp64", "first")
    fp32_artifact, fp32_summary = _write_worker_fixture(tmp_path, "fp32", "second")
    checkpoint = tmp_path / "initial.npz"
    checkpoint.write_bytes(b"parameters")

    with pytest.raises(ValueError, match="different schedules"):
        comparison.build_comparison_report(
            {"seed": 0},
            {"fp64": fp64_artifact, "fp32": fp32_artifact},
            {"fp64": fp64_summary, "fp32": fp32_summary},
            checkpoint,
        )


def test_worker_is_a_fresh_process_with_precision_and_cuda_set_first(
    monkeypatch, tmp_path
):
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command
        recorded.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(comparison.subprocess, "run", fake_run)
    comparison._invoke_worker(
        "compare",
        "fp32",
        tmp_path / "config.json",
        tmp_path / "arrays.npz",
        tmp_path / "summary.json",
        "gpu",
    )

    assert recorded["command"][0] == comparison.sys.executable
    assert "--_worker-mode" in recorded["command"]
    assert recorded["env"]["ADORA_PRECISION"] == "fp32"
    assert recorded["env"]["JAX_PLATFORMS"] == "cuda"
    assert recorded["check"] is False


def test_cli_requires_gpu_unless_cpu_is_explicitly_allowed():
    parser = comparison.build_argument_parser()
    defaults = parser.parse_args([])
    assert defaults.platform == "gpu"
    assert not defaults.allow_cpu
    permitted = parser.parse_args(["--platform", "cpu", "--allow-cpu"])
    assert permitted.platform == "cpu"
    assert permitted.allow_cpu


def test_public_module_does_not_import_jax_or_pinn_at_module_scope():
    assert "jax" not in comparison.__dict__
    assert "pinn" not in comparison.__dict__
    assert Path(comparison.__file__).name == "compare_pinn_precision.py"
