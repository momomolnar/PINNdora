import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


_PROBE = r"""
import json
import numpy as np
import jax
import jax.numpy as jnp

import adora_precision
from adora_data import FE_I_6301_6302_LINE_LIST
from lineop import emis_opac_polarised_offset, read_kurucz, wavelength_offsets
from pinn_3d_inversion import atomic_data_fingerprint
from voigt import voigt_H

lines = read_kurucz(FE_I_6301_6302_LINE_LIST)
center = lines.canonical_lambda0_nm[0]
wave_offset = wavelength_offsets(
    np.asarray([center + 1.0e-5], dtype=np.float64),
    lines.wavelength_reference_nm,
)[0]
base = jnp.asarray(
    [5777.0, 1.0e19, 1.0e23, 250.0, 1500.0, 0.1, 0.7, 0.3],
    dtype=adora_precision.REAL_DTYPE,
)
scale = jnp.asarray(
    [100.0, 1.0e18, 1.0e22, 100.0, 100.0, 0.01, 0.05, 0.05],
    dtype=adora_precision.REAL_DTYPE,
)

def coefficients(q):
    values = base + q * scale
    eta, chi = emis_opac_polarised_offset(lines, wave_offset, *values)
    return jnp.concatenate((eta, chi))

q0 = jnp.zeros(8, dtype=adora_precision.REAL_DTYPE)
values = coefficients(q0)
jacobian = jax.jacrev(coefficients)(q0)
voigt = jnp.asarray(voigt_H(jnp.asarray(0.1), jnp.asarray(0.25)))
complex_value = jax.lax.complex(
    jnp.asarray(0.1, dtype=adora_precision.REAL_DTYPE),
    jnp.asarray(0.2, dtype=adora_precision.REAL_DTYPE),
)
print(json.dumps({
    "precision": adora_precision.configured_precision(),
    "x64": bool(jax.config.x64_enabled),
    "real_dtype": str(np.asarray(values).dtype),
    "atomic_dtype": str(np.asarray(lines.lambda0).dtype),
    "offset_dtype": str(np.asarray(lines.lambda0_offset).dtype),
    "voigt_dtype": str(np.asarray(voigt).dtype),
    "complex_dtype": str(np.asarray(complex_value).dtype),
    "fingerprint": atomic_data_fingerprint(lines),
    "values": np.asarray(values, dtype=np.float64).tolist(),
    "jacobian": np.asarray(jacobian, dtype=np.float64).tolist(),
    "finite": bool(jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(jacobian))),
}))
"""


def _run_probe(precision):
    environment = os.environ.copy()
    environment["ADORA_PRECISION"] = precision
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_precision_modes_cast_atomic_voigt_and_gradients_consistently():
    fp64 = _run_probe("fp64")
    fp32 = _run_probe("fp32")

    assert fp64["precision"] == "fp64"
    assert fp32["precision"] == "fp32"
    assert fp64["x64"] is True
    assert fp32["x64"] is False
    assert fp64["real_dtype"] == fp64["atomic_dtype"] == "float64"
    assert fp32["real_dtype"] == fp32["atomic_dtype"] == "float32"
    assert fp64["offset_dtype"] == fp64["voigt_dtype"] == "float64"
    assert fp32["offset_dtype"] == fp32["voigt_dtype"] == "float32"
    assert fp64["complex_dtype"] == "complex128"
    assert fp32["complex_dtype"] == "complex64"
    assert fp64["fingerprint"] == fp32["fingerprint"]
    assert fp64["finite"] and fp32["finite"]

    values64 = np.asarray(fp64["values"])
    values32 = np.asarray(fp32["values"])
    scale = max(np.linalg.norm(values64), np.finfo(float).tiny)
    assert np.linalg.norm(values32 - values64) / scale < 5.0e-4

    gradient64 = np.asarray(fp64["jacobian"]).ravel()
    gradient32 = np.asarray(fp32["jacobian"]).ravel()
    relative_l2 = np.linalg.norm(gradient32 - gradient64) / np.linalg.norm(gradient64)
    cosine = np.dot(gradient64, gradient32) / (
        np.linalg.norm(gradient64) * np.linalg.norm(gradient32)
    )
    assert relative_l2 < 5.0e-4
    assert cosine > 0.99999


def test_direct_cli_selects_fp32_before_importing_physics():
    environment = os.environ.copy()
    environment.pop("ADORA_PRECISION", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), environment.get("PYTHONPATH", "")))
    )
    command = [
        sys.executable,
        str(ROOT / "pinn_3d_inversion.py"),
        "--precision",
        "fp32",
        "--help",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "--precision {fp32,fp64}" in completed.stdout
    assert "truncated to dtype float32" not in completed.stderr
