import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax
import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layer_sizes, key, learning_rate=1e-3):
        self.layer_sizes = layer_sizes
        self.key = key
        self.learning_rate = learning_rate
        self.params = self.init_params(layer_sizes, key)
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def init_params(self, layer_sizes, key):
        keys = random.split(key, len(layer_sizes))

        params = []
        for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
            params.append({
                "w": random.normal(k, (m, n)) * 0.01,
                "b": jnp.zeros(n)
            })
        return params

    def forward(self, params, x):
        for layer in params[:-1]:
            x = jax.nn.relu(jnp.dot(x, layer["w"]) + layer["b"])
        return jnp.dot(x, params[-1]["w"]) + params[-1]["b"]

    def mse_loss(self, params, x, y):
        predictions = self.forward(params, x)
        return jnp.mean((predictions - y) ** 2)

    @jit
    def train_step(self, params, opt_state, x, y):
        loss_value, grads = value_and_grad(self.mse_loss)(params, x, y)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    def train(self, x_train, y_train, num_epochs=1000):
        params, opt_state = self.params, self.opt_state
        for epoch in range(num_epochs):
            params, opt_state, loss_value = self.train_step(params, opt_state, x_train, y_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.4f}")
        self.params = params

    def predict(self, x):
        return self.forward(self.params, x)


# Hyperparameters
layer_sizes = [3, 64, 64, 5]  # Example: Input layer, two hidden layers, output layer
learning_rate = 1e-3
num_epochs = 1000

# Initialize parameters
key = random.PRNGKey(0)

# Create a dataset
# Here we use y = sin(2πx) as an example, but you can replace these with any generic NumPy arrays
x_train = jnp.linspace(-1, 1, 100).reshape(-1, 1)
y_train = jnp.sin(2 * jnp.pi * x_train)

# Initialize and train the MLP
mlp = MLP(layer_sizes, key, learning_rate)
mlp.train(x_train, y_train, num_epochs)

# Predict
x_test = jnp.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = mlp.predict(x_test)

# Plot results
plt.plot(x_test, y_pred, label='MLP Prediction')
plt.plot(x_test, jnp.sin(2 * jnp.pi * x_test), label='Exact Solution')
plt.legend()
plt.show()

# General case: Predicting with another np.array output
# Example:
# y_train_new = np.random.random((100, 1))
# mlp.train(x_train, jnp.array(y_train_new), num_epochs)
# y_pred_new = mlp.predict(x_test)
# plt.plot(x_test, y_pred_new, label='New MLP Prediction')
# plt.plot(x_test, y_train_new, label='New Exact Solution')
# plt.legend()
# plt.show()