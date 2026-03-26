import jax.numpy as jnp
import lightweaver as lw
from jax import random
import numpy as np
from lightweaver.fal import Falc82
from lineop import read_kurucz
import matplotlib.pyplot as plt

from PINN_machinery import PINNDora_MLP

try:
    get_ipython().run_line_magic("matplotlib", "")
except:
    plt.ion()


lines = read_kurucz("kurucz_6301_6302.linelist")
fal = Falc82()

spectrum_np_array = np.load("data/spectrum_2D.npz")  # This will be a NumPy array
spectrum = jnp.array(spectrum_np_array['arr_0'])  # Convert it back to JAX
nx = spectrum.shape[0]
ny = spectrum.shape[1]
nwv = spectrum.shape[2]

spectrum = jnp.reshape(spectrum, (1, nx*ny, nwv))

x_array = jnp.linspace(0, 1, num=nx)  # 50 points along the x-axis
y_array = jnp.linspace(0, 1, num=ny)  # 100 points along the y-axis
z_array = jnp.linspace(0, 1, num=len(fal.z))  # 100 points along the y-axis
xx, yy, zz = jnp.meshgrid(x_array, y_array, z_array, indexing="ij")  # Use 'ij' indexing for matrix-style indexing
xx_flat = xx.ravel()  # Shape: (M*N*P,)
yy_flat = yy.ravel()  # Shape: (M*N*P,)
zz_flat = zz.ravel()  # Shape: (M*N*P,)

# Stack the flattened coordinates into a single (N_points, 3) array
xyz_flat = jnp.stack([xx_flat, yy_flat, zz_flat], axis=-1)

dz = jnp.array(
    np.concatenate(
        [
            [fal.z[::-1][0] - fal.z[::-1][1]],
            fal.z[::-1][1:] - fal.z[::-1][:-1]
        ]
    )
)

waves = jnp.linspace(lw.air_to_vac(630.1), lw.air_to_vac(630.3), 201)

learning_rate = 1e-3
num_epochs = 1000
output_dim = 5
layer_sizes = [3, 64, 64, 64, output_dim]
# Initialize parameters
key = random.PRNGKey(0)
# Initialize and train the MLP
model = PINNDora_MLP(layer_sizes, key, waves, lines, dz, fal,
                     lr=learning_rate)

# def test_fn(p):
#     temperature_corr, ne_corr, nhtot_corr, vz_corr, vturb_corr = model.forward(xyz_flat, params=p)
#     t_temperature = model.temperature * (1 + jnp.reshape(temperature_corr, (-1, 82)))
#     t_nhtot       = model.nhtot       * (1 + jnp.reshape(nhtot_corr, (-1, 82)))
#     t_vz          = model.vz          * (1 + jnp.reshape(vz_corr, (-1, 82)))
#     t_ne          = model.ne          * (1 + jnp.reshape(ne_corr, (-1, 82)))
#     t_vturb       = model.vturb       * (1 + jnp.reshape(vturb_corr, (-1, 82)))

#     # # print(f"temp 1: {temperature[1, ...]}")
#     I_synthetic = model.compute_lte_rt_3D(
#         t_temperature,
#         t_ne,
#         t_nhtot,
#         t_vz,
#         t_vturb)

#     return jnp.mean((I_synthetic - spectrum.squeeze())**2)

model.train(xyz_flat, spectrum, num_epochs)


# # Testing
x_test = jnp.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_test)