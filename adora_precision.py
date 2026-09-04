"""Central numerical-precision configuration for Adora.

Precision has to be selected before importing physics modules because several
of them construct JAX constants and lookup tables at import time.  Set the
``ADORA_PRECISION`` environment variable to ``fp32`` or ``fp64`` before
starting Python.  ``fp64`` remains the default for backward compatibility.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np


PRECISION_ENV_VAR = "ADORA_PRECISION"
DEFAULT_PRECISION = "fp64"
SUPPORTED_PRECISIONS = ("fp32", "fp64")


def _normalize_precision(value: str | None) -> str:
    if value is None:
        value = os.environ.get(PRECISION_ENV_VAR, DEFAULT_PRECISION)
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_PRECISIONS:
        choices = ", ".join(SUPPORTED_PRECISIONS)
        raise ValueError(
            f"unsupported precision {value!r}; expected one of: {choices}"
        )
    return normalized


def configure_precision(value: str | None = None) -> str:
    """Configure JAX and return the selected precision name.

    Call this before importing modules that construct JAX arrays.  Passing an
    explicit value takes precedence over ``ADORA_PRECISION`` and updates the
    environment so subsequently imported modules observe the same setting.
    """

    selected = _normalize_precision(value)
    os.environ[PRECISION_ENV_VAR] = selected
    jax.config.update("jax_enable_x64", selected == "fp64")
    # Preserve the full selected mantissa for neural-network dot products.
    # In particular, fp32 must not silently become TF32 on NVIDIA hardware.
    jax.config.update("jax_default_matmul_precision", "highest")

    global PRECISION, REAL_DTYPE, COMPLEX_DTYPE, NUMPY_REAL_DTYPE
    global NUMPY_COMPLEX_DTYPE
    PRECISION = selected
    if selected == "fp64":
        REAL_DTYPE = jnp.float64
        COMPLEX_DTYPE = jnp.complex128
        NUMPY_REAL_DTYPE = np.dtype(np.float64)
        NUMPY_COMPLEX_DTYPE = np.dtype(np.complex128)
    else:
        REAL_DTYPE = jnp.float32
        COMPLEX_DTYPE = jnp.complex64
        NUMPY_REAL_DTYPE = np.dtype(np.float32)
        NUMPY_COMPLEX_DTYPE = np.dtype(np.complex64)
    return selected


def configured_precision() -> str:
    """Return the active Adora precision name."""

    return PRECISION


def precision_from_argv(argv=None) -> str:
    """Read ``--precision`` from command-line arguments without importing a CLI.

    Both ``--precision fp32`` and ``--precision=fp32`` are accepted.  When the
    option is absent, the environment value (or the fp64 default) is returned.
    This lightweight scan lets entry points configure JAX before importing any
    module-level numerical arrays.
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    for index, argument in enumerate(arguments):
        if argument == "--precision":
            if index + 1 == len(arguments):
                raise ValueError("--precision requires fp32 or fp64")
            return _normalize_precision(arguments[index + 1])
        if argument.startswith("--precision="):
            return _normalize_precision(argument.partition("=")[2])
    return _normalize_precision(None)


# Importing this small module is the single configuration point used by all
# numerical modules.  No JAX arrays are allocated before this call.
configure_precision()


__all__ = [
    "COMPLEX_DTYPE",
    "DEFAULT_PRECISION",
    "NUMPY_COMPLEX_DTYPE",
    "NUMPY_REAL_DTYPE",
    "PRECISION",
    "PRECISION_ENV_VAR",
    "REAL_DTYPE",
    "SUPPORTED_PRECISIONS",
    "configure_precision",
    "configured_precision",
    "precision_from_argv",
]
