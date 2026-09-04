"""A small, import-safe Optax MLP example."""

import math

import jax
import jax.numpy as jnp
import optax
from jax import random, value_and_grad


class MLP:
    """Fully connected ReLU network with a reusable compiled training step."""

    def __init__(self, layer_sizes, key, learning_rate=1e-3):
        layer_sizes = tuple(int(size) for size in layer_sizes)
        if len(layer_sizes) < 2 or any(size <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive sizes")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")

        self.layer_sizes = layer_sizes
        self.key = key
        self.learning_rate = learning_rate
        self.params = self.init_params(layer_sizes, key)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # A closure keeps the Python optimizer static without passing the MLP
        # instance as a dynamic argument to jax.jit.
        def compiled_step(params, opt_state, x, y):
            loss_value, grads = value_and_grad(self.mse_loss)(params, x, y)
            updates, next_opt_state = self.optimizer.update(
                grads, opt_state, params
            )
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, loss_value

        self._compiled_step = jax.jit(compiled_step)

    @staticmethod
    def init_params(layer_sizes, key):
        keys = random.split(key, len(layer_sizes) - 1)
        return [
            {
                "w": random.normal(layer_key, (n_in, n_out)) * 0.01,
                "b": jnp.zeros(n_out),
            }
            for n_in, n_out, layer_key in zip(
                layer_sizes[:-1], layer_sizes[1:], keys
            )
        ]

    def _validate_input(self, x):
        x = jnp.asarray(x)
        if x.ndim != 2:
            raise ValueError("x must have shape (n_samples, input_dim)")
        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"x has {x.shape[1]} features, expected {self.layer_sizes[0]}"
            )
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one sample")
        return x

    def _validate_target(self, x, y):
        y = jnp.asarray(y)
        expected = (x.shape[0], self.layer_sizes[-1])
        if y.shape != expected:
            raise ValueError(f"y must have shape {expected}, got {y.shape}")
        return y

    def forward(self, params, x):
        x = self._validate_input(x)
        for layer in params[:-1]:
            x = jax.nn.relu(jnp.dot(x, layer["w"]) + layer["b"])
        return jnp.dot(x, params[-1]["w"]) + params[-1]["b"]

    def mse_loss(self, params, x, y):
        x = self._validate_input(x)
        y = self._validate_target(x, y)
        predictions = self.forward(params, x)
        return jnp.mean((predictions - y) ** 2)

    def train_step(self, params, opt_state, x, y):
        """Perform one step without treating ``self`` as a jitted argument."""
        x = self._validate_input(x)
        y = self._validate_target(x, y)
        return self._compiled_step(params, opt_state, x, y)

    def train(self, x_train, y_train, num_epochs=1000):
        if num_epochs < 0:
            raise ValueError("num_epochs must be non-negative")
        x_train = self._validate_input(x_train)
        y_train = self._validate_target(x_train, y_train)

        params, opt_state = self.params, self.opt_state
        for epoch in range(num_epochs):
            params, opt_state, loss_value = self.train_step(
                params, opt_state, x_train, y_train
            )
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value.item():.4f}")
        self.params = params
        self.opt_state = opt_state

    def predict(self, x):
        return self.forward(self.params, x)


def main():
    """Train and plot the scalar sine-wave example."""
    import matplotlib.pyplot as plt

    layer_sizes = [1, 64, 64, 1]
    learning_rate = 1e-3
    num_epochs = 1000

    x_train = jnp.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y_train = jnp.sin(2.0 * jnp.pi * x_train)

    mlp = MLP(layer_sizes, random.PRNGKey(0), learning_rate)
    mlp.train(x_train, y_train, num_epochs)

    y_pred = mlp.predict(x_train)
    plt.plot(x_train, y_pred, label="MLP Prediction")
    plt.plot(x_train, y_train, label="Exact Solution")
    plt.legend()
    plt.show()
    return mlp


if __name__ == "__main__":
    main()
