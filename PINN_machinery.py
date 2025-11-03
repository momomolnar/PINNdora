import jax
import optax
from jax import random
from tqdm import trange  # Import tqdm's range object for iteration with progress bar

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from lineop import emis_opac
from scalar_formal_solver import nearest_fs
from jax import grad

class PINNDora_MLP:
    def __init__(self, layer_sizes, key, waves, lines, dz, fal, lr=1e-3):
        self.layer_sizes = layer_sizes
        self.key = key
        self.waves = waves
        self.lines = lines
        self.dz = dz
        self.fal = fal
        self.temperature = jnp.array(fal.temperature)
        self.nhtot       = jnp.array(fal.nHTot)
        self.vz          = jnp.array(fal.vz)
        self.ne          = jnp.array(fal.ne)
        self.vturb       = jnp.array(fal.vturb)

        self.params = self.init_params(layer_sizes, key)
        self.optimizer = optax.adam(learning_rate=lr)
        self.opt_state = self.optimizer.init(self.params)

    def init_params(self, layer_sizes, key):
        """
        Convert parameters into a PyTree of nested dictionaries for JAX/Optax compatibility.
        Args:
            layer_sizes: List of integers defining the layer dimensions.
            key: PRNGKey for randomness.

        Returns:
            A dictionary of parameters (trainable weights and biases for each layer).
        """
        keys = random.split(key, len(layer_sizes) - 1)

        params = {}
        for i, (m, n, k) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:], keys)):
            params[f"layer_{i}"] = {
                "w": random.normal(k, (m, n)) * jnp.sqrt(2.0 / (m + n)),  # Xavier init
                "b": jnp.zeros(n)  # Bias initialized to zero
            }
        return params

    def forward(self, x, params=None):
        """
        Feed-forward pass through the MLP.
        Args:
            x: Input data, shape (batch_size, input_dim).

        Returns:
            Output: Atmospheric parameters, shape (batch_size, output_dim).
        """
        if params is None:
            params = self.params
        x = jnp.asarray(x)  # Ensure x is a JAX array

        # Pass through all layers except the last one (apply activation here)
        for i in range(len(self.layer_sizes) - 2):  # Skip the final layer
            layer_params = params[f"layer_{i}"]  # Get parameters for layer i
            x = jax.nn.gelu(jnp.dot(x, layer_params["w"]) + layer_params["b"])  # Linear plus activation

        # Final output layer (no activation, reshape to match output dimensions)
        final_layer_params = params[f"layer_{len(self.layer_sizes) - 2}"]
        flat_output = jnp.dot(x, final_layer_params["w"]) + final_layer_params["b"]

        # Reshape the final output to (batch_size, 5, len(self.dz))
        output = flat_output.reshape(-1, 5)
        return output[..., 0], output[..., 1], output[..., 2], output[..., 3], output[..., 4]

    def lte_rt(self, lines, waves, dz, temperature, ne, nhtot, vz, vturb):
        eta, chi = (jax.vmap(
            emis_opac,
            in_axes=[None, None, 0, 0, 0, 0, 0]
        )(lines, waves, temperature, ne, nhtot, vz, vturb))

        I = nearest_fs(dz, eta, chi)
        return I

    def compute_lte_rt_3D(self, temperature_3D, ne_3D, nhtot_3D, vz_3D, vturb_3D):

        # Define the transformation using jax.vmap

        lte_rt_wave_3D = jax.jit(jax.vmap(
                    jax.vmap(
                        self.lte_rt,
                        in_axes=[None, 0, None, None, None, None, None, None]  # Vectorize over `waves`
                    ),
                    in_axes=[None, None, None, 0, 0, 0, 0, 0]  # Vectorize over spatial grid (x_dim, y_dim)
                ))


        # Compute the 3D intensities using lte_rt_wave_3D
        intens_3D = lte_rt_wave_3D(self.lines, self.waves, self.dz,
            temperature_3D,
            ne_3D,
            nhtot_3D,
            vz_3D,
            vturb_3D
        )
        return intens_3D

    def mse_loss(self, x, I_obs, params):
        temperature_corr, ne_corr, nhtot_corr, vz_corr, vturb_corr = self.forward(x, params=params)
        print(temperature_corr.shape)
        # breakpoint()
        self.t_temperature = self.temperature * (1 + jnp.reshape(temperature_corr, (-1, 82)))
        self.t_nhtot       = self.nhtot       * (1 + jnp.reshape(nhtot_corr, (-1, 82)))
        self.t_vz          = self.vz          * (1 + jnp.reshape(vz_corr, (-1, 82)))
        self.t_ne          = self.ne          * (1 + jnp.reshape(ne_corr, (-1, 82)))
        self.t_vturb       = self.vturb       * (1 + jnp.reshape(vturb_corr, (-1, 82)))

        # print(f"temp 1: {temperature[1, ...]}")
        self.I_synthetic = self.compute_lte_rt_3D(
            self.t_temperature,
            self.t_ne,
            self.t_nhtot,
            self.t_vz,
            self.t_vturb)
        # print(f"Isynth shape: {I_synthetic.shape}")
        print(jnp.mean(self.I_synthetic))

        # return jnp.mean(I_synthetic)
        return jnp.mean((self.I_synthetic - I_obs[0, ...]) ** 2)

    def train_step(self, x, y):
        # print(f"Params before update: {self.params['layer_2']['w'][0:3, 0:3]}")
        loss_value, self.opt_state, self.params = self._train_step(self.params, self.opt_state, x, y, self.optimizer,
                                                                   self.mse_loss)
        # print(f"grad: {grad(self.mse_loss)(x, y, self.params)}")
        # print(f"Params after update: {self.params['layer_2']['w'][0:3, 0:3]}")

        return loss_value

    def _train_step(self, params, opt_state, x, y, optimizer, loss_fn):
        def loss_function(params):
            return loss_fn(x, y, self.params)  # Make params explicit in loss_fn

        # Compute loss and gradients
        # loss_value, grads = jax.value_and_grad(loss_fn)(x, y)
        loss_value = self.mse_loss(x, y, params)
        grads = jax.grad(loss_function)(params)
        # breakpoint()
        # Update parameters using the optimizer
        print(f"Params after update: {params['layer_2']['w'][0:3, 0:3]}")
        breakpoint()
        updates, opt_state = optimizer.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)
        print(f"Params after update: {params['layer_2']['w'][0:3, 0:3]}")
        return loss_value, opt_state, params

    def train(self, x_train, y_train, num_epochs=1000):
        """
        Train the MLP with a progress bar.
        Args:
            x_train: Input training data (e.g., spatial grid coordinates).
            y_train: Target data (e.g., observed spectrum/intensities).
            num_epochs: Number of training epochs.
        """
        # Use tqdm's `trange` object to wrap the training loop
        progress_bar = trange(num_epochs, desc="Training Progress", leave=True, position=0)

        for epoch in progress_bar:
            loss_value = self.train_step(x_train, y_train)  # Perform one training step

            # Update progress bar with the current loss
            progress_bar.set_postfix({"loss": loss_value.item()})

            # Perform train_step
            # Optionally log progress every 100 epochs (or more)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.4f}")

    def predict(self, x):
        return self.forward(x)
