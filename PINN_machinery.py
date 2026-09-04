"""Neural atmospheric corrections coupled to Adora's LTE synthesis."""

import math

import adora_precision
adora_precision.configure_precision()
import jax
import jax.numpy as jnp
import optax
from jax import random
from tqdm import trange

from atmosphere import atmosphere_from_falc
from lineop import emis_opac, planck
from scalar_formal_solver import nearest_fs

def _lte_rt(lines, wave, dz, temperature, ne, nhtot, vz, vturb):
    """Synthesize one wavelength through one bottom-to-top column."""
    eta, chi = jax.vmap(
        emis_opac,
        in_axes=(None, None, 0, 0, 0, 0, 0),
    )(lines, wave, temperature, ne, nhtot, vz, vturb)
    lower_boundary = planck(wave, temperature[0])
    return nearest_fs(dz, eta, chi, I_start=lower_boundary)


_LTE_RT_WAVELENGTHS = jax.vmap(
    _lte_rt,
    in_axes=(None, 0, None, None, None, None, None, None),
)
_LTE_RT_COLUMNS = jax.jit(
    jax.vmap(
        _LTE_RT_WAVELENGTHS,
        in_axes=(None, None, None, 0, 0, 0, 0, 0),
    )
)


class PINNDora_MLP:
    """MLP predicting depth-dependent corrections for atmospheric columns.

    ``x`` is a two-dimensional coordinate array with depth varying fastest:
    all depth points for the first spatial column, followed by all depth points
    for the second column, and so on.  The network has five output channels in
    the order temperature, electron density, total hydrogen density, LOS
    velocity, and microturbulence.
    """

    n_outputs = 5

    def __init__(
        self,
        layer_sizes,
        key,
        waves,
        lines,
        dz,
        fal,
        lr=1e-3,
        velocity_scale=1e3,
        correction_limit=10.0,
    ):
        layer_sizes = tuple(int(size) for size in layer_sizes)
        if len(layer_sizes) < 2 or any(size <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive sizes")
        if layer_sizes[-1] != self.n_outputs:
            raise ValueError(
                f"the PINN output layer must have {self.n_outputs} units, "
                f"got {layer_sizes[-1]}"
            )
        if not math.isfinite(velocity_scale) or velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be a positive finite value")
        if not math.isfinite(correction_limit) or correction_limit <= 0.0:
            raise ValueError("correction_limit must be a positive finite value")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite value")

        self.layer_sizes = layer_sizes
        self.key = key
        self.waves = jnp.asarray(waves)
        if self.waves.ndim != 1 or self.waves.size == 0:
            raise ValueError("waves must be a non-empty one-dimensional array")
        if not bool(jnp.all(jnp.isfinite(self.waves))) or not bool(
            jnp.all(self.waves > 0.0)
        ):
            raise ValueError("waves must contain only positive finite values")
        self.lines = lines
        self.fal = fal
        self.velocity_scale = float(velocity_scale)
        self.correction_limit = float(correction_limit)

        (
            _,
            fal_dz,
            self.temperature,
            self.ne,
            self.nhtot,
            self.vz,
            self.vturb,
        ) = atmosphere_from_falc(fal)

        self.dz = fal_dz if dz is None else jnp.asarray(dz)
        if self.dz.ndim != 1:
            raise ValueError("dz must be a one-dimensional array")
        self.n_depth = int(self.dz.shape[0])
        if self.n_depth != int(self.temperature.shape[0]):
            raise ValueError(
                "dz and the FAL atmosphere must contain the same number of depths"
            )
        base_profiles = {
            "temperature": self.temperature,
            "ne": self.ne,
            "nhtot": self.nhtot,
            "vz": self.vz,
            "vturb": self.vturb,
        }
        for name, profile in base_profiles.items():
            if profile.shape != (self.n_depth,):
                raise ValueError(
                    f"the FAL {name} profile must have shape ({self.n_depth},)"
                )
            if not bool(jnp.all(jnp.isfinite(profile))):
                raise ValueError(f"the FAL {name} profile must be finite")
        for name in ("temperature", "ne", "nhtot", "vturb"):
            if not bool(jnp.all(base_profiles[name] > 0.0)):
                raise ValueError(f"the FAL {name} profile must be strictly positive")
        if not bool(jnp.all(jnp.isfinite(self.dz))):
            raise ValueError("dz must contain only finite values")
        if not bool(jnp.all(self.dz > 0.0)):
            raise ValueError("dz must contain strictly positive cell widths")

        self.params = self.init_params(layer_sizes, key)
        self.optimizer = optax.adam(learning_rate=lr)
        self.opt_state = self.optimizer.init(self.params)
        self.loss_and_grad = jax.jit(
            jax.value_and_grad(self.mse_loss, argnums=2)
        )

        # Construct this transformation once.  Recreating a jitted nested-vmap
        # inside every loss evaluation defeats JAX's compilation cache.
        def compiled_train_step(params, opt_state, x, observed):
            loss_value, grads = self.loss_and_grad(x, observed, params)
            updates, next_opt_state = self.optimizer.update(
                grads, opt_state, params
            )
            next_params = optax.apply_updates(params, updates)
            return loss_value, next_opt_state, next_params

        self._compiled_train_step = jax.jit(compiled_train_step)

    @staticmethod
    def init_params(layer_sizes, key):
        """Initialize a dictionary PyTree of Xavier-scaled layer parameters."""
        keys = random.split(key, len(layer_sizes) - 1)
        params = {}
        for i, (m, n, layer_key) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ):
            params[f"layer_{i}"] = {
                "w": random.normal(layer_key, (m, n))
                * jnp.sqrt(2.0 / (m + n)),
                "b": jnp.zeros(n),
            }
        return params

    def _validate_coordinates(self, x):
        x = jnp.asarray(x)
        if x.ndim != 2:
            raise ValueError("x must have shape (n_samples, input_dim)")
        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"x has {x.shape[1]} features, expected {self.layer_sizes[0]}"
            )
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one sample")
        if not isinstance(x, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(x))
        ):
            raise ValueError("x must contain only finite values")
        return x

    def _validate_column_coordinates(self, x):
        x = self._validate_coordinates(x)
        if x.shape[0] % self.n_depth != 0:
            raise ValueError(
                "the number of coordinate rows must be a positive multiple of n_depth"
            )
        return x

    def _expected_observed_shape(self, x):
        return (x.shape[0] // self.n_depth, self.waves.shape[0])

    def _validate_observed(self, x, observed):
        observed = jnp.asarray(observed)
        expected = self._expected_observed_shape(x)
        if observed.shape != expected:
            raise ValueError(
                f"observed spectra must have shape {expected}, got {observed.shape}"
            )
        if not isinstance(observed, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(observed))
        ):
            raise ValueError("observed spectra must contain only finite values")
        return observed

    def forward(self, x, params=None):
        """Return five flat correction channels for the supplied coordinates."""
        x = self._validate_coordinates(x)
        if params is None:
            params = self.params

        activations = x
        for i in range(len(self.layer_sizes) - 2):
            layer = params[f"layer_{i}"]
            activations = jax.nn.gelu(
                jnp.dot(activations, layer["w"]) + layer["b"]
            )

        final_layer = params[f"layer_{len(self.layer_sizes) - 2}"]
        output = jnp.dot(activations, final_layer["w"]) + final_layer["b"]
        return tuple(output[..., i] for i in range(self.n_outputs))

    def apply_corrections(self, corrections):
        """Convert correction logits into physical bottom-to-top profiles.

        Positive quantities use multiplicative exponential corrections.  LOS
        velocity is additive so a zero-velocity reference atmosphere still has
        a nonzero velocity gradient.
        """
        corrections = jnp.asarray(corrections)
        if corrections.ndim != 3 or corrections.shape[1:] != (
            self.n_depth,
            self.n_outputs,
        ):
            raise ValueError(
                "corrections must have shape (n_columns, n_depth, 5)"
            )
        if not isinstance(corrections, jax.core.Tracer) and not bool(
            jnp.all(jnp.isfinite(corrections))
        ):
            raise ValueError("corrections must contain only finite values")

        # A runaway optimizer step must not turn otherwise finite coordinates
        # into zero/inf physical profiles.  Inside the limit this is exactly the
        # original parameterization; outside it saturates to a conservative
        # multiplicative range and a bounded velocity perturbation.
        corrections = jnp.clip(
            corrections, -self.correction_limit, self.correction_limit
        )
        temperature = self.temperature * jnp.exp(corrections[..., 0])
        ne = self.ne * jnp.exp(corrections[..., 1])
        nhtot = self.nhtot * jnp.exp(corrections[..., 2])
        vz = self.vz + self.velocity_scale * corrections[..., 3]
        vturb = self.vturb * jnp.exp(corrections[..., 4])
        return temperature, ne, nhtot, vz, vturb

    def corrected_atmosphere(self, x, params=None):
        """Evaluate the network and return corrected atmospheric columns."""
        x = self._validate_column_coordinates(x)
        channels = self.forward(x, params=params)
        corrections = jnp.stack(channels, axis=-1).reshape(
            (-1, self.n_depth, self.n_outputs)
        )
        return self.apply_corrections(corrections)

    @staticmethod
    def lte_rt(lines, waves, dz, temperature, ne, nhtot, vz, vturb):
        """Backward-compatible wrapper for one wavelength and one column."""
        return _lte_rt(lines, waves, dz, temperature, ne, nhtot, vz, vturb)

    def compute_lte_rt_3D(
        self, temperature_3D, ne_3D, nhtot_3D, vz_3D, vturb_3D
    ):
        """Synthesize profiles whose final axis is depth.

        Both flattened ``(n_columns, n_depth)`` inputs and true spatial grids
        such as ``(nx, ny, n_depth)`` are accepted.  The returned shape is the
        input spatial shape followed by wavelength.
        """
        profiles = tuple(
            jnp.asarray(profile)
            for profile in (temperature_3D, ne_3D, nhtot_3D, vz_3D, vturb_3D)
        )
        profile_shape = profiles[0].shape
        if len(profile_shape) < 2 or profile_shape[-1] != self.n_depth:
            raise ValueError(
                "atmospheric profiles must have shape (..., n_depth)"
            )
        if any(profile.shape != profile_shape for profile in profiles[1:]):
            raise ValueError("all atmospheric profiles must have identical shapes")

        spatial_shape = profile_shape[:-1]
        flat_profiles = tuple(
            profile.reshape((-1, self.n_depth)) for profile in profiles
        )
        intensity = _LTE_RT_COLUMNS(
            self.lines,
            self.waves,
            self.dz,
            *flat_profiles,
        )
        return intensity.reshape(spatial_shape + (self.waves.shape[0],))

    def mse_loss(self, x, I_obs, params):
        """Return spectral MSE without mutating model state."""
        x = self._validate_column_coordinates(x)
        I_obs = self._validate_observed(x, I_obs)
        atmosphere = self.corrected_atmosphere(x, params=params)
        synthetic = self.compute_lte_rt_3D(*atmosphere)
        return jnp.mean((synthetic - I_obs) ** 2)

    def train_step(self, x, y):
        """Perform one compiled optimizer step and return its pre-update loss."""
        x = self._validate_column_coordinates(x)
        y = self._validate_observed(x, y)
        loss_value, self.opt_state, self.params = self._compiled_train_step(
            self.params, self.opt_state, x, y
        )
        return loss_value

    def _train_step(
        self,
        params,
        opt_state,
        x,
        y,
        optimizer=None,
        loss_fn=None,
    ):
        """Compatibility wrapper around the model's reusable compiled step."""
        x = self._validate_column_coordinates(x)
        y = self._validate_observed(x, y)
        optimizer = self.optimizer if optimizer is None else optimizer
        loss_fn = self.mse_loss if loss_fn is None else loss_fn
        if optimizer is self.optimizer and loss_fn == self.mse_loss:
            return self._compiled_train_step(params, opt_state, x, y)

        loss_value, grads = jax.value_and_grad(loss_fn, argnums=2)(x, y, params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return loss_value, next_opt_state, next_params

    def train(self, x_train, y_train, num_epochs=1000):
        """Train on complete atmospheric columns."""
        if num_epochs < 0:
            raise ValueError("num_epochs must be non-negative")
        x_train = self._validate_column_coordinates(x_train)
        y_train = self._validate_observed(x_train, y_train)
        progress_bar = trange(
            num_epochs, desc="Training Progress", leave=True, position=0
        )
        for epoch in progress_bar:
            loss_value = self.train_step(x_train, y_train)
            progress_bar.set_postfix({"loss": loss_value.item()})
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value.item():.4f}")

    def predict(self, x):
        """Return the five correction channels using the current parameters."""
        return self.forward(x)
