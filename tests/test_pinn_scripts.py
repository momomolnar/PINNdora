from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import PINN_example
import PINN_machinery
import create_2D_testset
import pretrain_model
from atmosphere import cell_widths
from lineop import planck, read_kurucz
from lte_pops import DEBROGLIE_CONST, K_B_EV, lte_pops


class FakeFal:
    def __init__(self):
        # Lightweaver's public ordering is top to bottom.
        self.z = np.array([30.0, 20.0, 10.0])
        self.temperature = np.array([300.0, 200.0, 100.0])
        self.ne = np.array([3.0e16, 2.0e16, 1.0e16])
        self.nHTot = np.array([3.0e20, 2.0e20, 1.0e20])
        self.vz = np.zeros(3)
        self.vturb = np.array([3.0e3, 2.0e3, 1.0e3])


def make_pinn(n_columns=2):
    fal = FakeFal()
    dz = cell_widths(fal.z[::-1])
    model = PINN_machinery.PINNDora_MLP(
        [3, 5],
        jax.random.PRNGKey(0),
        jnp.array([630.2]),
        None,
        dz,
        fal,
    )
    coordinates = jnp.zeros((n_columns * model.n_depth, 3))
    zero_params = jax.tree.map(jnp.zeros_like, model.params)
    return model, coordinates, zero_params


def test_pinn_uses_dynamic_depth_reversed_base_and_physical_transforms():
    model, coordinates, zero_params = make_pinn()

    temperature, ne, nhtot, vz, vturb = model.corrected_atmosphere(
        coordinates, zero_params
    )
    assert temperature.shape == (2, 3)
    np.testing.assert_allclose(temperature[0], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(ne[0], [1.0e16, 2.0e16, 3.0e16])
    np.testing.assert_allclose(nhtot[0], [1.0e20, 2.0e20, 3.0e20])
    np.testing.assert_allclose(vz, 0.0)
    np.testing.assert_allclose(vturb[0], [1.0e3, 2.0e3, 3.0e3])

    corrections = jnp.zeros((2, 3, 5)).at[..., (0, 1, 2, 4)].set(-5.0)
    corrected = model.apply_corrections(corrections)
    for positive_profile in (*corrected[:3], corrected[4]):
        assert bool(jnp.all(positive_profile > 0.0))

    velocity_correction = corrections.at[..., 3].set(1.0)
    corrected_velocity = model.apply_corrections(velocity_correction)[3]
    np.testing.assert_allclose(corrected_velocity, model.velocity_scale)

    extreme = jnp.full((2, model.n_depth, 5), 1.0e3)
    positive_extreme = model.apply_corrections(extreme)
    assert all(bool(jnp.all(jnp.isfinite(profile))) for profile in positive_extreme)
    negative_extreme = model.apply_corrections(-extreme)
    assert all(bool(jnp.all(jnp.isfinite(profile))) for profile in negative_extreme)
    for positive_profile in (*negative_extreme[:3], negative_extreme[4]):
        assert bool(jnp.all(positive_profile > 0.0))


def test_pinn_and_dataset_reject_nonpositive_wavelengths(tmp_path):
    fal = FakeFal()
    with pytest.raises(ValueError, match="positive finite"):
        PINN_machinery.PINNDora_MLP(
            [3, 5],
            jax.random.PRNGKey(0),
            jnp.array([0.0]),
            None,
            cell_widths(fal.z[::-1]),
            fal,
        )

    profiles = (jnp.ones(3),) * 5
    with pytest.raises(ValueError, match="positive finite"):
        create_2D_testset.synthesize_grid(
            None, jnp.array([-1.0]), jnp.ones(3), *profiles, x_dim=1, y_dim=1
        )

    invalid_archive = tmp_path / "invalid.npz"
    np.savez(invalid_archive, intensity=np.ones((1, 1, 1)), wavelength=[0.0])
    with pytest.raises(ValueError, match="positive finite"):
        PINN_example.load_dataset(invalid_archive)


def test_pinn_requires_depth_fast_coordinates_and_exact_observed_shape(monkeypatch):
    model, coordinates, zero_params = make_pinn()
    with pytest.raises(ValueError, match="multiple of n_depth"):
        model.corrected_atmosphere(jnp.zeros((5, 3)), zero_params)
    assert model.forward(jnp.zeros((5, 3)), zero_params)[0].shape == (5,)
    with pytest.raises(ValueError, match="observed spectra must have shape"):
        model.mse_loss(coordinates, jnp.zeros((1, 2, 1)), zero_params)
    with pytest.raises(ValueError, match="x must contain only finite"):
        model.predict(jnp.full((1, 3), jnp.nan))
    with pytest.raises(ValueError, match="observed spectra must contain only finite"):
        model.mse_loss(coordinates, jnp.full((2, 1), jnp.nan), zero_params)

    monkeypatch.setattr(
        model,
        "compute_lte_rt_3D",
        lambda *profiles: jnp.mean(profiles[0], axis=-1, keepdims=True),
    )
    observed = jnp.broadcast_to(jnp.mean(model.temperature), (2, 1))
    loss, grads = jax.value_and_grad(model.mse_loss, argnums=2)(
        coordinates, observed, zero_params
    )
    assert loss == pytest.approx(0.0)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))
    assert not hasattr(model, "I_synthetic")
    assert not hasattr(model, "t_temperature")

    model.params = zero_params
    model.opt_state = model.optimizer.init(model.params)
    assert model.train_step(coordinates, observed) == pytest.approx(0.0)


def test_pinn_compute_preserves_true_spatial_shape(monkeypatch):
    model, _, _ = make_pinn()

    def fake_columns(lines, waves, dz, temperature, ne, nhtot, vz, vturb):
        return jnp.broadcast_to(
            jnp.sum(temperature, axis=-1, keepdims=True),
            (temperature.shape[0], waves.shape[0]),
        )

    monkeypatch.setattr(PINN_machinery, "_LTE_RT_COLUMNS", fake_columns)
    profile = jnp.ones((2, 4, model.n_depth))
    result = model.compute_lte_rt_3D(
        profile, profile, profile, profile, profile
    )
    assert result.shape == (2, 4, model.waves.shape[0])


@pytest.mark.parametrize("module", [PINN_machinery, create_2D_testset])
def test_scalar_pinn_paths_use_planck_lower_boundary(module, monkeypatch):
    monkeypatch.setattr(
        module,
        "emis_opac",
        lambda adata, wave, temperature, ne, nhtot, vz, vturb: (1.0, 1.0),
    )
    monkeypatch.setattr(
        module,
        "nearest_fs",
        lambda dz, eta, chi, I_start=0.0: I_start,
    )
    temperature = jnp.array([5000.0, 6000.0])
    ones = jnp.ones(2)
    scalar_lte_rt = (
        module._lte_rt if module is PINN_machinery else module.lte_rt
    )
    result = scalar_lte_rt(
        None, 630.2, ones, temperature, ones, ones, ones, ones
    )
    np.testing.assert_allclose(result, planck(630.2, temperature[0]))


def test_pretrain_model_trains_scalar_shapes_and_persists_optimizer_state():
    model = pretrain_model.MLP([1, 8, 1], jax.random.PRNGKey(1))
    x = jnp.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    y = jnp.sin(x)
    initial_params = jax.tree.map(lambda value: value.copy(), model.params)
    initial_state = jax.tree.map(lambda value: value.copy(), model.opt_state)

    model.train(x, y, num_epochs=1)
    assert model.predict(x).shape == y.shape
    assert any(
        not np.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(initial_params), jax.tree.leaves(model.params)
        )
    )
    assert any(
        not np.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(initial_state), jax.tree.leaves(model.opt_state)
        )
    )

    state_after_first_train = jax.tree.map(
        lambda value: value.copy(), model.opt_state
    )
    model.train(x, y, num_epochs=1)
    assert any(
        not np.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(state_after_first_train),
            jax.tree.leaves(model.opt_state),
        )
    )

    with pytest.raises(ValueError, match="y must have shape"):
        model.train(x, jnp.zeros((16, 5)), num_epochs=1)


def test_demo_modules_are_import_safe():
    assert not hasattr(PINN_example, "model")
    assert not hasattr(pretrain_model, "mlp")
    assert not hasattr(create_2D_testset, "intensity")


def test_dataset_uses_xy_depth_order_and_named_roundtrip(tmp_path):
    profiles = create_2D_testset.broadcast_atmosphere(
        (jnp.arange(4.0),) * 5, x_dim=2, y_dim=3
    )
    assert all(profile.shape == (2, 3, 4) for profile in profiles)
    np.testing.assert_allclose(profiles[0][1, 2], jnp.arange(4.0))

    intensity = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    wavelength = np.linspace(630.1, 630.2, 4)
    path = tmp_path / "nested" / "spectrum.npz"
    create_2D_testset.save_dataset(path, intensity, wavelength)
    assert path.exists()
    with np.load(path) as archive:
        assert set(archive.files) == {"intensity", "wavelength"}

    loaded_intensity, loaded_wavelength = PINN_example.load_dataset(path)
    np.testing.assert_allclose(loaded_intensity, intensity)
    np.testing.assert_allclose(loaded_wavelength, wavelength)


def test_dataset_synthesis_returns_x_then_y_axes():
    lines = read_kurucz(
        Path(__file__).resolve().parents[1] / "kurucz_6301_6302.linelist"
    )
    depth = 3
    intensity = create_2D_testset.synthesize_grid(
        lines,
        jnp.array([630.2]),
        jnp.ones(depth) * 10.0,
        jnp.array([5000.0, 5500.0, 6000.0]),
        jnp.ones(depth) * 1.0e17,
        jnp.ones(depth) * 1.0e20,
        jnp.zeros(depth),
        jnp.ones(depth) * 1.0e3,
        x_dim=2,
        y_dim=3,
    )
    assert intensity.shape == (2, 3, 1)
    assert bool(jnp.all(jnp.isfinite(intensity)))


def test_coordinate_grid_is_depth_fastest():
    coordinates = PINN_example.coordinate_grid(2, 3, jnp.array([10.0, 20.0, 40.0]))
    assert coordinates.shape == (18, 3)
    np.testing.assert_allclose(coordinates[:3, :2], 0.0)
    np.testing.assert_allclose(coordinates[:3, 2], [0.0, 1.0 / 3.0, 1.0])


def test_lte_pops_matches_analytic_ratios_and_conserves_population():
    temperature = 6000.0
    ne = 1.0e17
    ntot = 4.0e12
    energy = jnp.array([0.0, 1.0])
    g = jnp.array([2.0, 4.0])

    same_stage = lte_pops(
        energy, g, jnp.array([0, 0]), temperature, ne, ntot
    )
    expected_ratio = 2.0 * np.exp(-1.0 / (K_B_EV * temperature))
    np.testing.assert_allclose(same_stage[1] / same_stage[0], expected_ratio)
    np.testing.assert_allclose(jnp.sum(same_stage), ntot)

    adjacent_stage = lte_pops(
        energy, g, jnp.array([0, 1]), temperature, ne, ntot
    )
    saha_term = 0.5 * ne * (DEBROGLIE_CONST / temperature) ** 1.5
    np.testing.assert_allclose(
        adjacent_stage[1] / adjacent_stage[0],
        expected_ratio / saha_term,
    )
    np.testing.assert_allclose(jnp.sum(adjacent_stage), ntot)


def test_lte_pops_log_normalization_stays_finite_under_extreme_weights():
    populations = jax.jit(lte_pops)(
        jnp.zeros(3),
        jnp.ones(3),
        jnp.array([0, 50, 100]),
        1.0e4,
        1.0e6,
        1.0e20,
    )
    assert bool(jnp.all(jnp.isfinite(populations)))
    assert bool(jnp.all(populations >= 0.0))
    np.testing.assert_allclose(jnp.sum(populations), 1.0e20, rtol=1e-12)

    with pytest.raises(ValueError, match="identical shapes"):
        lte_pops(jnp.ones(2), jnp.ones(3), jnp.ones(2), 5000.0, 1e16, 1e10)
