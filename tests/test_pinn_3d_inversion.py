from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pinn_3d_inversion as pinn3d
from adora_data import FE_I_6301_6302_LINE_LIST
from lineop import read_kurucz


@pytest.fixture(scope="module")
def tiny_polarized_cube():
    lines = read_kurucz(FE_I_6301_6302_LINE_LIST)
    x, y, height, z_norm, dz, coordinates, reference, cube = (
        pinn3d.create_falc_reference_cube(1, 2)
    )
    perturbation = pinn3d.PerturbationConfig(
        center_x=0.5,
        center_y=0.0,
        sigma_x=0.2,
        sigma_y=0.2,
    )
    truth, envelope = pinn3d.perturb_falc_cube(cube, x, y, z_norm, perturbation)
    centers = pinn3d._canonical_line_centers(lines)
    wavelengths = np.asarray(
        (
            centers[0] - 0.04,
            centers[0],
            0.5 * (centers[0] + centers[1]),
            centers[1],
            centers[1] + 0.04,
        )
    )
    observed = pinn3d.synthesize_atmosphere_cube(
        lines,
        wavelengths,
        dz,
        truth,
        batch_columns=2,
        show_progress=False,
    )
    payload = {
        "schema_version": np.asarray(pinn3d.SCHEMA_VERSION),
        "stokes_labels": np.asarray(pinn3d.STOKES_LABELS),
        "x_normalized": np.asarray(x),
        "y_normalized": np.asarray(y),
        "height_m": np.asarray(height),
        "height_normalized": np.asarray(z_norm),
        "cell_width_m": np.asarray(dz),
        "wavelength_nm": np.asarray(wavelengths),
        "line_center_nm": centers,
        "atomic_data_sha256": np.asarray(pinn3d.atomic_data_fingerprint(lines)),
        "observed_stokes": np.asarray(observed),
        "perturbation_envelope": np.asarray(envelope),
        **pinn3d._atmosphere_payload("reference_", reference),
        **pinn3d._atmosphere_payload("truth_", truth),
    }
    return {
        "lines": lines,
        "coordinates": coordinates,
        "reference": reference,
        "truth": truth,
        "payload": payload,
    }


def test_cli_defaults_to_requested_50_by_50_cube():
    args = pinn3d.build_argument_parser().parse_args([])
    assert (args.nx, args.ny, args.n_wave) == (50, 50, 201)
    assert args.wavelength_parallelism == pinn3d.DEFAULT_WAVELENGTH_PARALLELISM == 48
    assert args.validation_columns == pinn3d.DEFAULT_VALIDATION_COLUMNS == 8
    assert args.precision == pinn3d.adora_precision.configured_precision()
    assert args.stokes_weights == (1.0, 5.0, 5.0, 2.0)
    assert not args.generate_only
    assert not args.invert_only


def test_falc_cube_and_coordinates_have_canonical_depth_fast_order():
    x, y, height, z_norm, dz, coordinates, reference, cube = (
        pinn3d.create_falc_reference_cube(2, 3)
    )
    assert height.shape == (82,)
    assert coordinates.shape == (2, 3, 82, 3)
    assert all(field.shape == (2, 3, 82) for field in cube)
    assert bool(jnp.all(jnp.diff(height) > 0.0))
    assert bool(jnp.all(dz > 0.0))
    np.testing.assert_allclose(coordinates[0, 0, :, :2], -1.0)
    np.testing.assert_allclose(coordinates[0, 0, :, 2], 2.0 * z_norm - 1.0)
    for base_profile, cube_field in zip(reference, cube):
        np.testing.assert_allclose(cube_field[1, 2], base_profile)
    assert x.shape == (2,)
    assert y.shape == (3,)
    *_, wrapped_reference, _ = pinn3d.create_falc_reference_cube(
        1, 1, azimuth_rad=100.0
    )
    assert bool(jnp.all(jnp.abs(wrapped_reference.chi_b) < 0.5 * jnp.pi))


def test_perturbation_is_deterministic_small_and_physical():
    x, y, _, z_norm, _, _, _, cube = pinn3d.create_falc_reference_cube(3, 3)
    config = pinn3d.PerturbationConfig()
    first, first_envelope = pinn3d.perturb_falc_cube(cube, x, y, z_norm, config)
    second, second_envelope = pinn3d.perturb_falc_cube(cube, x, y, z_norm, config)
    np.testing.assert_array_equal(first_envelope, second_envelope)
    for first_field, second_field in zip(first, second):
        np.testing.assert_array_equal(first_field, second_field)
        assert bool(jnp.all(jnp.isfinite(first_field)))
    assert float(jnp.max(jnp.abs(jnp.log(first.temperature / cube.temperature)))) <= (
        abs(config.log_temperature) + 1.0e-12
    )
    assert float(jnp.max(jnp.abs(first.vz - cube.vz))) <= (
        abs(config.velocity_m_s) + 1.0e-12
    )
    assert bool(jnp.all(first.temperature > 0.0))
    assert bool(jnp.all(first.ne > 0.0))
    assert bool(jnp.all(first.nhtot > 0.0))
    assert bool(jnp.all(first.vturb > 0.0))
    assert bool(jnp.all(first.b > 0.0))
    assert bool(jnp.all((first.gamma_b > 0.0) & (first.gamma_b < jnp.pi)))
    assert not np.allclose(first.temperature[0, 0], first.temperature[1, 1])


def test_unreachable_custom_perturbation_is_rejected_before_synthesis(tmp_path):
    with pytest.raises(ValueError, match="transform bounds"):
        pinn3d.generate_test_cube(
            tmp_path / "unreachable.npz",
            nx=1,
            ny=1,
            n_wave=4,
            perturbation=pinn3d.PerturbationConfig(log_temperature=100.0),
            show_progress=False,
        )


def test_wavelength_grid_covers_every_packaged_kurucz_line():
    lines = read_kurucz(FE_I_6301_6302_LINE_LIST)
    wavelengths = pinn3d.build_wavelength_grid(lines, n_wave=21)
    assert lines.lambda0.shape == (2,)
    assert bool(jnp.all(jnp.diff(wavelengths) > 0.0))
    assert float(wavelengths[0]) < float(jnp.min(lines.lambda0))
    assert float(wavelengths[-1]) > float(jnp.max(lines.lambda0))
    for center in pinn3d._canonical_line_centers(lines):
        assert np.count_nonzero(np.asarray(wavelengths) == center) == 1
    with pytest.raises(ValueError, match="every unique line center"):
        pinn3d.build_wavelength_grid(lines, n_wave=3)


def test_real_tiny_cube_has_finite_full_stokes_and_spatial_signal(
    tiny_polarized_cube,
):
    observed = tiny_polarized_cube["payload"]["observed_stokes"]
    assert observed.shape == (1, 2, 4, 5)
    assert np.all(np.isfinite(observed))
    assert np.all(observed[:, :, 0] > 0.0)
    assert np.max(np.abs(observed[:, :, 1:])) > 0.0
    assert not np.allclose(observed[0, 0], observed[0, 1], rtol=1.0e-10)


def test_accelerator_column_and_wavelength_vectorization_match_sequential_solver(
    monkeypatch, tiny_polarized_cube
):
    payload = tiny_polarized_cube["payload"]
    truth = pinn3d.Atmosphere(
        *(
            field.reshape((2, -1))
            for field in pinn3d._atmosphere_from_payload(payload, "truth_")
        )
    )
    lines = tiny_polarized_cube["lines"]
    wavelengths = np.asarray(payload["wavelength_nm"], dtype=np.float64)
    offsets = pinn3d.centered_wavelengths(lines, wavelengths)
    dz = jnp.asarray(payload["cell_width_m"])

    monkeypatch.setattr(pinn3d.jax, "default_backend", lambda: "cpu")
    sequential = pinn3d._synthesize_columns_offset_core(
        lines, offsets, dz, truth, wavelength_parallelism=1
    )
    monkeypatch.setattr(pinn3d.jax, "default_backend", lambda: "gpu")
    vectorized = pinn3d._synthesize_columns_offset_core(
        lines, offsets, dz, truth, wavelength_parallelism=2
    )

    synthesis_tolerance = max(
        1.0e-12, 64.0 * np.finfo(np.asarray(offsets).dtype).eps
    )
    np.testing.assert_allclose(
        vectorized,
        sequential,
        rtol=synthesis_tolerance,
        atol=synthesis_tolerance,
    )
    with pytest.raises(ValueError, match="wavelength_parallelism"):
        pinn3d._synthesize_columns_offset_core(
            lines, offsets, dz, truth, wavelength_parallelism=0
        )


def test_latent_transform_roundtrips_reference_and_bounds_extremes():
    *_, reference, _ = pinn3d.create_falc_reference_cube(1, 1)
    decoded = pinn3d.latent_to_atmosphere(pinn3d.atmosphere_to_latent(reference))
    transform_tolerance = max(
        1.0e-8, 64.0 * np.finfo(np.asarray(decoded.temperature).dtype).eps
    )
    for expected, actual in zip(reference, decoded):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=transform_tolerance,
            atol=transform_tolerance,
        )

    for extreme in (-1.0e3, 1.0e3):
        atmosphere = pinn3d.latent_to_atmosphere(jnp.full((2, 3, 8), extreme))
        assert all(bool(jnp.all(jnp.isfinite(field))) for field in atmosphere)
        for positive in (
            atmosphere.temperature,
            atmosphere.ne,
            atmosphere.nhtot,
            atmosphere.vturb,
            atmosphere.b,
        ):
            assert bool(jnp.all(positive > 0.0))
        assert bool(
            jnp.all((atmosphere.gamma_b >= 0.0) & (atmosphere.gamma_b <= jnp.pi))
        )

    magnetic_jacobian = jax.jacrev(
        lambda latent: jnp.stack(pinn3d.latent_to_atmosphere(latent)[5:])
    )(jnp.zeros(8))
    assert bool(jnp.all(jnp.isfinite(magnetic_jacobian)))
    with pytest.raises(ValueError, match="transform bounds"):
        pinn3d.create_falc_reference_cube(1, 1, magnetic_field_t=0.5)


def test_pretraining_reduces_falc_loss_and_preserves_xy_invariance():
    _, _, _, z_norm, _, coordinates, reference, _ = pinn3d.create_falc_reference_cube(
        1, 2
    )
    config = pinn3d.NeuralFieldConfig()
    params = pinn3d.initialize_neural_field(jax.random.PRNGKey(0), config)
    initial_spatial = jax.tree.map(lambda value: value.copy(), params["spatial"])
    params, history = pinn3d.pretrain_falc(
        params,
        z_norm,
        reference,
        steps=2000,
        learning_rate=2.0e-3,
        tolerance=1.0e-4,
        show_progress=False,
    )
    assert history.shape == (2000,)
    assert np.isfinite(history).all()
    assert np.min(history) < 1.0e-4
    target = pinn3d.atmosphere_to_latent(reference)
    network_input = pinn3d._base_features(
        params["base"], (2.0 * z_norm - 1.0).reshape((-1, 1))
    )
    returned_loss = jnp.mean(
        (pinn3d.apply_mlp(params["base"], network_input) - target) ** 2
    )
    assert float(returned_loss) <= float(np.min(history)) + 1.0e-14
    for before, after in zip(
        jax.tree.leaves(initial_spatial), jax.tree.leaves(params["spatial"])
    ):
        np.testing.assert_array_equal(after, before)
    atmosphere = pinn3d.evaluate_neural_field(
        params, coordinates, jnp.asarray(config.spatial_scale)
    )
    for field in atmosphere:
        np.testing.assert_allclose(field[0, 0], field[0, 1], rtol=1.0e-12, atol=1.0e-10)
    relative_limits = {
        "temperature": 0.03,
        "ne": 0.05,
        "nhtot": 0.05,
        "vturb": 0.02,
        "b": 0.02,
    }
    for name, target, inferred in zip(pinn3d.FIELD_NAMES, reference, atmosphere):
        inferred = inferred[0, 0]
        if name in relative_limits:
            error = float(jnp.max(jnp.abs(inferred / target - 1.0)))
            assert error < relative_limits[name]
        elif name == "vz":
            assert float(jnp.max(jnp.abs(inferred - target))) < 100.0
        else:
            assert float(jnp.max(jnp.abs(inferred - target))) < 0.02


def test_weighted_stokes_loss_uses_each_polarization_component():
    observed = jnp.zeros((2, 4, 3))
    continuum = jnp.asarray((2.0, 4.0))
    weights = jnp.asarray((1.0, 2.0, 3.0, 4.0))
    for stokes_index in range(4):
        synthetic = observed.at[0, stokes_index, 1].set(1.0)
        loss = pinn3d.weighted_stokes_mse(synthetic, observed, continuum, weights)
        expected = (weights[stokes_index] / continuum[0]) ** 2 / 24.0
        np.testing.assert_allclose(loss, expected)
    assert pinn3d.weighted_stokes_mse(
        observed, observed, continuum, weights
    ) == pytest.approx(0.0)


def test_fixed_column_batches_cover_every_column_once_and_pad_only_masked():
    batches = pinn3d.padded_column_batches(
        n_columns=6,
        batch_size=4,
        rng=np.random.default_rng(7),
    )
    assert [batch[2] for batch in batches] == [4, 2]
    real_indices = np.concatenate([indices[:count] for indices, _, count in batches])
    np.testing.assert_array_equal(np.sort(real_indices), np.arange(6))
    np.testing.assert_array_equal(batches[-1][1], (1.0, 1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="positive"):
        pinn3d.sample_wavelength_indices(np.random.default_rng(0), np.arange(5.0), 0)


def test_epoch_batching_preserves_rng_order_without_materializing_wavelengths():
    class ShapeOnlyWavelengths:
        shape = (7,)

        def __array__(self, *args, **kwargs):
            raise AssertionError("wavelength values must stay on the accelerator")

    actual_rng = np.random.default_rng(17)
    actual = pinn3d._spectral_epoch_batches(
        5,
        2,
        ShapeOnlyWavelengths(),
        3,
        actual_rng,
    )

    expected_rng = np.random.default_rng(17)
    column_batches = pinn3d.padded_column_batches(5, 2, expected_rng)
    expected = (
        np.stack([indices for indices, _, _ in column_batches]),
        np.stack(
            [
                pinn3d.sample_wavelength_indices(
                    expected_rng, np.empty(7), 3
                )
                for _ in column_batches
            ]
        ),
        np.stack([mask for _, mask, _ in column_batches]),
    )
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert actual_rng.random() == expected_rng.random()


def test_dataset_and_checkpoint_roundtrip(tmp_path, tiny_polarized_cube):
    dataset_path = tmp_path / "cube" / "test_cube.npz"
    pinn3d.save_test_cube(dataset_path, tiny_polarized_cube["payload"])
    loaded = pinn3d.load_test_cube(dataset_path)
    assert pinn3d.validate_test_cube(loaded) == (1, 2, 82, 5)
    np.testing.assert_array_equal(
        loaded["observed_stokes"],
        tiny_polarized_cube["payload"]["observed_stokes"],
    )

    config = pinn3d.NeuralFieldConfig(
        base_hidden=(8,), spatial_hidden=(9,), base_frequencies=2
    )
    params = pinn3d.initialize_neural_field(jax.random.PRNGKey(5), config)
    checkpoint_path = tmp_path / "checkpoint" / "field.npz"
    pinn3d.save_checkpoint(checkpoint_path, params, config)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        assert str(archive["checkpoint_precision"]) == (
            pinn3d.adora_precision.configured_precision()
        )
        assert archive["positive_lower"].dtype == np.float64
    restored, restored_config = pinn3d.load_checkpoint(checkpoint_path)
    assert restored_config == config
    for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(expected, actual)

    coordinates = tiny_polarized_cube["coordinates"]
    direct = pinn3d.evaluate_neural_field(params, coordinates)
    chunked = pinn3d.evaluate_neural_field_cube(
        params, coordinates, batch_columns=1, show_progress=False
    )
    evaluation_tolerance = max(
        1.0e-12, 64.0 * np.finfo(np.asarray(coordinates).dtype).eps
    )
    for expected, actual in zip(direct, chunked):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=evaluation_tolerance,
            atol=evaluation_tolerance,
        )

    result = {
        **loaded,
        "synthetic_stokes": loaded["observed_stokes"],
        "pretrain_loss": np.asarray((1.0, 0.5)),
        "inversion_loss_total_spectral_prior": np.asarray(((0.2, 0.2, 0.0),)),
        "inversion_loss_columns": np.asarray(("total", "spectral", "prior")),
        "validation_full_wavelength_loss": np.asarray((0.2, 0.1)),
        "best_validation_epoch": np.asarray(1),
        "best_validation_loss": np.asarray(0.1),
        "final_full_cube_spectral_loss": np.asarray(0.1),
        "base_layers": np.asarray(config.base_layers),
        "spatial_layers": np.asarray(config.spatial_layers),
        "spatial_scale": np.asarray(config.spatial_scale),
        "stokes_weights": np.ones(4),
        "prior_weight": np.asarray(0.0),
        "seed": np.asarray(0),
        **pinn3d._atmosphere_payload("inferred_", tiny_polarized_cube["truth"]),
    }
    assert pinn3d.validate_inversion_result(result) == (1, 2, 82, 5)

    inconsistent = dict(loaded)
    inconsistent["height_normalized"] = loaded["height_normalized"] ** 2
    with pytest.raises(ValueError, match="inconsistent"):
        pinn3d.validate_test_cube(inconsistent)


def test_atomic_fingerprint_prevents_mismatched_inversion(tiny_polarized_cube):
    incompatible = dict(tiny_polarized_cube["payload"])
    incompatible["atomic_data_sha256"] = np.asarray("0" * 64)
    with pytest.raises(ValueError, match="atomic data do not match"):
        pinn3d.run_inversion(
            incompatible,
            field_config=pinn3d.NeuralFieldConfig(
                base_hidden=(8,), spatial_hidden=(8,), base_frequencies=1
            ),
            pretrain_steps=0,
            inversion_epochs=0,
            show_progress=False,
        )


def test_atomic_fingerprint_excludes_derived_packed_zeeman_metadata():
    lines = read_kurucz(FE_I_6301_6302_LINE_LIST)
    fingerprint = pinn3d.atomic_data_fingerprint(lines)

    repacked = replace(
        lines,
        zeeman_component_lines=jnp.flip(lines.zeeman_component_lines),
    )
    assert pinn3d.atomic_data_fingerprint(repacked) == fingerprint

    physically_changed = replace(
        lines,
        line_weight=lines.line_weight.at[0].multiply(2.0),
    )
    assert pinn3d.atomic_data_fingerprint(physically_changed) != fingerprint


def test_real_spectral_gradient_and_one_inversion_epoch_are_finite(
    tiny_polarized_cube,
):
    case = tiny_polarized_cube
    payload = case["payload"]
    config = pinn3d.NeuralFieldConfig(
        base_hidden=(12,), spatial_hidden=(12,), base_frequencies=2
    )
    params = pinn3d.initialize_neural_field(jax.random.PRNGKey(8), config)
    coordinates = case["coordinates"].reshape((2, 82, 3))
    observed = jnp.asarray(payload["observed_stokes"]).reshape((2, 4, 5))
    wavelengths = np.asarray(payload["wavelength_nm"], dtype=np.float64)
    offsets = pinn3d.centered_wavelengths(case["lines"], wavelengths)
    continuum = pinn3d.continuum_normalization(observed)

    def loss_fn(candidate):
        return pinn3d.neural_spectral_loss_offset(
            candidate,
            coordinates,
            offsets,
            observed,
            continuum,
            payload["cell_width_m"],
            case["lines"],
            jnp.asarray(config.spatial_scale),
            wavelength_parallelism=2,
        )[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    gradient_norm = sum(float(jnp.sum(leaf**2)) for leaf in jax.tree.leaves(grads))
    assert gradient_norm > 0.0
    assert bool(jnp.all(jnp.abs(grads["spatial"][-1]["b"]) > 0.0))

    initial_spatial = jax.tree.map(lambda value: value.copy(), params["spatial"])
    fitted, history, validation_history = pinn3d.train_spectral_inversion(
        params,
        coordinates,
        observed,
        wavelengths,
        payload["line_center_nm"],
        payload["cell_width_m"],
        case["lines"],
        epochs=1,
        batch_columns=2,
        wavelength_batch=5,
        learning_rate=1.0e-4,
        spatial_scale=jnp.asarray(config.spatial_scale),
        seed=3,
        show_progress=False,
        wavelength_parallelism=2,
        validation_columns=2,
    )
    assert history.shape == (1, 3)
    assert np.isfinite(history).all()
    assert validation_history.shape == (2,)
    assert np.isfinite(validation_history).all()
    assert validation_history[-1] < validation_history[0]
    fitted_loss = float(loss_fn(fitted))
    loss_rtol = 8.0 * np.finfo(np.asarray(offsets).dtype).eps
    np.testing.assert_allclose(
        fitted_loss, np.min(validation_history), rtol=loss_rtol
    )
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(initial_spatial), jax.tree.leaves(fitted["spatial"])
        )
    )


def test_module_is_import_safe_and_packaged_line_list_is_used():
    assert Path(pinn3d.FE_I_6301_6302_LINE_LIST).name.endswith(".linelist")
    assert not hasattr(pinn3d, "observed_stokes")
    assert not hasattr(pinn3d, "trained_params")


def test_console_entrypoint_returns_success_value(monkeypatch, tiny_polarized_cube):
    monkeypatch.setattr(
        pinn3d,
        "generate_test_cube",
        lambda **kwargs: tiny_polarized_cube["payload"],
    )
    assert pinn3d.main(["--generate-only", "--no-progress"]) is None

    called = False

    def should_not_generate(**kwargs):
        nonlocal called
        called = True
        return tiny_polarized_cube["payload"]

    monkeypatch.setattr(pinn3d, "generate_test_cube", should_not_generate)
    with pytest.raises(ValueError, match="requires --checkpoint-in"):
        pinn3d.main(["--skip-pretraining", "--no-progress"])
    assert not called
