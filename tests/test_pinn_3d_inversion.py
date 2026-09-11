from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pinn_3d_inversion as pinn3d
from adora_data import FE_I_6301_6302_LINE_LIST
from lineop import read_kurucz


class _FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name):
        self.files.append((path, name))


class _FakeWandbRun:
    def __init__(self):
        self.id = "test-run-id"
        self.config = {}
        self.summary = {}
        self.metric_definitions = []
        self.logged = []
        self.input_artifacts = []
        self.output_artifacts = []
        self.finish_codes = []

    def define_metric(self, name, **kwargs):
        self.metric_definitions.append((name, kwargs))

    def log(self, values):
        self.logged.append(dict(values))

    def use_artifact(self, artifact):
        self.input_artifacts.append(artifact)

    def log_artifact(self, artifact):
        self.output_artifacts.append(artifact)

    def finish(self, exit_code):
        self.finish_codes.append(exit_code)


class _FakeWandb:
    Artifact = _FakeArtifact

    @staticmethod
    def Image(figure):
        return ("image", figure)

    def __init__(self):
        self.run = _FakeWandbRun()
        self.init_kwargs = None
        self.login_calls = 0

    def login(self):
        self.login_calls += 1
        return True

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run.config.update(kwargs["config"])
        return self.run


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
    assert args.inversion_learning_rate == pytest.approx(1.0e-3)
    assert args.inversion_final_learning_rate == pytest.approx(1.0e-5)
    assert args.spectra_output is None
    assert args.precision == pinn3d.adora_precision.configured_precision()
    assert args.stokes_weights == (1.0, 5.0, 5.0, 2.0)
    assert not args.generate_only
    assert not args.invert_only
    assert not args.wandb
    assert args.wandb_project == pinn3d.DEFAULT_WANDB_PROJECT
    assert args.wandb_mode == "online"
    assert args.wandb_tags == ()
    assert args.wandb_evaluation_every == pinn3d.DEFAULT_WANDB_EVALUATION_EVERY == 50


def test_sin_squared_learning_rate_has_requested_endpoints_and_shape():
    schedule = pinn3d.sin_squared_learning_rate_schedule(101, 1.0e-3, 1.0e-5)
    values = np.asarray([schedule(step) for step in range(101)], dtype=float)
    assert values[0] == pytest.approx(1.0e-3)
    assert values[-1] == pytest.approx(1.0e-5)
    assert values[50] == pytest.approx(0.5 * (1.0e-3 + 1.0e-5))
    assert np.all(np.diff(values) <= 0.0)
    assert float(schedule(-10)) == pytest.approx(1.0e-3)
    assert float(schedule(1000)) == pytest.approx(1.0e-5)
    with pytest.raises(ValueError, match="must not exceed"):
        pinn3d.sin_squared_learning_rate_schedule(10, 1.0e-5, 1.0e-3)


def test_spectra_output_flag_accepts_default_or_explicit_path(tmp_path):
    default_args = pinn3d.build_argument_parser().parse_args(["--spectra-output"])
    assert default_args.spectra_output == pinn3d.DEFAULT_SPECTRA_OUTPUT
    requested = tmp_path / "spectra.npz"
    explicit_args = pinn3d.build_argument_parser().parse_args(
        ["--output-spectra", str(requested)]
    )
    assert explicit_args.spectra_output == requested


def test_wandb_logger_records_configuration_metrics_and_artifacts(tmp_path):
    wandb_dir = tmp_path / "wandb-metadata"
    args = pinn3d.build_argument_parser().parse_args(
        [
            "--generate-only",
            "--wandb",
            "--wandb-project",
            "adora-test",
            "--wandb-entity",
            "research-team",
            "--wandb-name",
            "tiny-run",
            "--wandb-group",
            "unit-tests",
            "--wandb-tags",
            "fp32, strong-field",
            "--wandb-mode",
            "offline",
            "--wandb-dir",
            str(wandb_dir),
            "--wandb-log-artifacts",
        ]
    )
    fake_wandb = _FakeWandb()
    field_config = pinn3d.NeuralFieldConfig(
        spatial_hidden=(9,),
    )

    logger = pinn3d._initialize_wandb(args, field_config, fake_wandb)

    assert wandb_dir.is_dir()
    assert fake_wandb.init_kwargs["project"] == "adora-test"
    assert fake_wandb.init_kwargs["entity"] == "research-team"
    assert fake_wandb.init_kwargs["name"] == "tiny-run"
    assert fake_wandb.init_kwargs["group"] == "unit-tests"
    assert fake_wandb.init_kwargs["tags"] == ("fp32", "strong-field")
    assert fake_wandb.init_kwargs["mode"] == "offline"
    assert fake_wandb.init_kwargs["job_type"] == "generate"
    assert fake_wandb.init_kwargs["dir"] == str(wandb_dir.resolve())
    assert fake_wandb.init_kwargs["config"]["dataset"] == str(args.dataset)
    assert fake_wandb.run.config["effective_spatial_hidden"] == [9]
    assert fake_wandb.run.config["workflow"] == "generate"
    metric_names = {name for name, _ in fake_wandb.run.metric_definitions}
    assert {
        "inversion/epoch",
        "inversion/train_total_loss",
        "inversion/validation_full_wavelength_loss",
    } <= metric_names

    inversion_record = {
        "inversion/epoch": 1,
        "inversion/train_total_loss": 0.25,
        "inversion/validation_full_wavelength_loss": 0.2,
    }
    logger.log_inversion_metrics(inversion_record)
    assert fake_wandb.run.logged[-1] == inversion_record
    figure = object()
    logger.log_evaluation_figure(50, figure)
    assert fake_wandb.run.logged[-1] == {
        "inversion/epoch": 50,
        "inversion/evaluation_figure": ("image", figure),
    }
    logger.update_config({"custom_path": tmp_path / "custom.npz"})
    assert fake_wandb.run.config["custom_path"] == str(tmp_path / "custom.npz")

    artifact_path = tmp_path / "result.npz"
    artifact_path.write_bytes(b"test artifact")
    logger.track_file(artifact_path, artifact_type="dataset", role="input")
    logger.track_file(artifact_path, artifact_type="result", role="output")
    assert len(fake_wandb.run.input_artifacts) == 1
    assert len(fake_wandb.run.output_artifacts) == 1
    assert fake_wandb.run.input_artifacts[0].files[0][1] == "result.npz"
    assert fake_wandb.run.output_artifacts[0].metadata["role"] == "output"

    logger.finish(exit_code=0)
    assert fake_wandb.run.finish_codes == [0]


def test_wandb_missing_dependency_has_actionable_error(monkeypatch):
    args = pinn3d.build_argument_parser().parse_args(["--wandb"])

    def missing_wandb(_name):
        raise ModuleNotFoundError("No module named 'wandb'")

    monkeypatch.setattr(pinn3d.importlib, "import_module", missing_wandb)
    with pytest.raises(RuntimeError, match="wandb is not installed"):
        pinn3d._initialize_wandb(args, pinn3d.NeuralFieldConfig())


def test_wandb_login_is_explicit():
    args = pinn3d.build_argument_parser().parse_args(["--wandb", "--wandb-login"])
    fake_wandb = _FakeWandb()
    logger = pinn3d._initialize_wandb(args, pinn3d.NeuralFieldConfig(), fake_wandb)
    assert fake_wandb.login_calls == 1
    logger.finish(exit_code=0)


def test_wandb_tags_reject_empty_lists():
    with pytest.raises(SystemExit):
        pinn3d.build_argument_parser().parse_args(["--wandb-tags", ", ,"])


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
    with pytest.raises(ValueError, match="correction bounds"):
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


def test_wandb_evaluation_figure_uses_requested_4_by_3_layout(
    tiny_polarized_cube,
):
    import matplotlib.pyplot as plt

    payload = tiny_polarized_cube["payload"]
    truth = tiny_polarized_cube["truth"]
    figure = pinn3d.create_inversion_evaluation_figure(
        payload["wavelength_nm"],
        payload["observed_stokes"][0, 1],
        payload["observed_stokes"][0, 1] * 0.99,
        payload["x_normalized"],
        payload["y_normalized"],
        payload["height_m"],
        truth,
        truth,
        x_index=0,
        y_index=1,
        height_index=40,
        epoch=50,
    )
    try:
        assert len(figure.axes) == 16  # 12 panels plus four shared colorbars.
        assert figure.axes[0].get_title() == ""
        assert figure.axes[1].get_title() == "Input / truth"
        assert figure.axes[2].get_title() == "Current inversion"
        assert "epoch 50" in figure._suptitle.get_text()
    finally:
        plt.close(figure)


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

    synthesis_tolerance = max(1.0e-12, 64.0 * np.finfo(np.asarray(offsets).dtype).eps)
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


def test_relative_transform_roundtrips_known_ratios_and_signed_offsets():
    *_, reference, _ = pinn3d.create_falc_reference_cube(1, 1)
    reference = reference._replace(chi_b=jnp.zeros_like(reference.chi_b))
    corrections = jnp.broadcast_to(
        jnp.asarray((0.1, -0.5, 0.3, -0.25, 0.2, -0.1, 0.05, -0.2)), (82, 8)
    )
    atmosphere = pinn3d.corrections_to_atmosphere(corrections, reference)
    for index in (0, 1, 2, 4, 5):
        np.testing.assert_allclose(
            atmosphere[index] / reference[index],
            10.0 ** np.asarray(corrections[:, index]),
            rtol=2e-6,
        )
    np.testing.assert_allclose(atmosphere.vz, reference.vz - 5000.0)
    np.testing.assert_allclose(atmosphere.gamma_b, reference.gamma_b + 0.05 * np.pi)
    np.testing.assert_allclose(atmosphere.chi_b, -0.1 * np.pi)
    np.testing.assert_allclose(
        pinn3d.atmosphere_to_corrections(atmosphere, reference),
        corrections,
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.parametrize("seed", [0, 7])
def test_zero_initialization_reproduces_the_exact_reference(seed):
    _, _, _, z_norm, _, coordinates, reference, cube = (
        pinn3d.create_falc_reference_cube(2, 3)
    )
    config = pinn3d.NeuralFieldConfig(spatial_hidden=(8, 8))
    params = pinn3d.initialize_neural_field(
        jax.random.PRNGKey(seed), reference, z_norm, config
    )
    np.testing.assert_array_equal(pinn3d.neural_field_output(params, coordinates), 0.0)
    for expected, actual in zip(
        cube, pinn3d.evaluate_neural_field(params, coordinates)
    ):
        np.testing.assert_array_equal(actual, expected)
    assert set(params) == {"reference", "spatial"}
    assert not hasattr(pinn3d, "pretrain_falc")
    # At arbitrary heights the reference is interpolated, not learned.
    points = jnp.asarray(((0.1, -0.3, -0.2), (-0.7, 0.8, 0.45)))
    atmosphere = pinn3d.evaluate_neural_field(params, points)
    for profile, actual in zip(reference, atmosphere):
        expected = np.interp(
            np.asarray(points[:, 2]),
            2.0 * np.asarray(z_norm) - 1.0,
            np.asarray(profile),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_network_output_is_bounded_and_all_decoder_channels_have_gradients():
    _, _, _, z_norm, _, coordinates, reference, _ = pinn3d.create_falc_reference_cube(
        1, 1
    )
    config = pinn3d.NeuralFieldConfig(spatial_hidden=(8,))
    params = pinn3d.initialize_neural_field(
        jax.random.PRNGKey(0), reference, z_norm, config
    )
    for extreme in (-1000.0, 1000.0):
        candidate = {
            **params,
            "spatial": (
                *params["spatial"][:-1],
                {
                    **params["spatial"][-1],
                    "b": jnp.full(8, extreme),
                },
            ),
        }
        output = pinn3d.neural_field_output(candidate, coordinates)
        assert bool(jnp.all(jnp.abs(output) <= 1.0))
        atmosphere = pinn3d.evaluate_neural_field(candidate, coordinates)
        pinn3d._validate_atmosphere(atmosphere)
    single = pinn3d.Atmosphere(*(value[0] for value in reference))
    normalizers = jnp.asarray(
        (
            single.temperature,
            single.ne,
            single.nhtot,
            20000.0,
            single.vturb,
            single.b,
            np.pi,
            np.pi / 2.0,
        )
    )
    jacobian = jax.jacrev(
        lambda delta: jnp.stack(pinn3d.corrections_to_atmosphere(delta, single))
        / normalizers
    )(jnp.zeros(8))
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert bool(jnp.all(jnp.diag(jacobian) > 0.0))
    np.testing.assert_allclose(
        np.diag(jacobian)[[0, 1, 2, 4, 5]], np.log(10), rtol=2e-6
    )


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
                pinn3d.sample_wavelength_indices(expected_rng, np.empty(7), 3)
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
    spectra_path = tmp_path / "spectra" / "comparison.npz"
    pinn3d.save_spectra_output(
        spectra_path,
        x_normalized=loaded["x_normalized"],
        y_normalized=loaded["y_normalized"],
        wavelength_nm=loaded["wavelength_nm"],
        input_stokes=loaded["observed_stokes"],
        output_stokes=loaded["observed_stokes"] * 0.99,
    )
    spectra = pinn3d.load_spectra_output(spectra_path)
    assert pinn3d.validate_spectra_output(spectra) == (1, 2, 5)
    np.testing.assert_allclose(
        spectra["output_stokes"], loaded["observed_stokes"] * 0.99
    )

    config = pinn3d.NeuralFieldConfig(spatial_hidden=(9,))
    params = pinn3d.initialize_neural_field(
        jax.random.PRNGKey(5),
        tiny_polarized_cube["reference"],
        loaded["height_normalized"],
        config,
    )
    checkpoint_path = tmp_path / "checkpoint" / "field.npz"
    pinn3d.save_checkpoint(checkpoint_path, params, config)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        assert str(archive["checkpoint_precision"]) == (
            pinn3d.adora_precision.configured_precision()
        )
        assert archive["spatial_scale"].dtype == np.float64
        assert str(archive["parameterization"]) == pinn3d.PARAMETERIZATION
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
        "inferred_normalized_corrections": np.asarray(
            pinn3d.atmosphere_to_corrections(
                tiny_polarized_cube["truth"], tiny_polarized_cube["reference"]
            )
        ),
        **pinn3d._correction_metadata(),
        "inversion_loss_total_spectral_prior": np.asarray(((0.2, 0.2, 0.0),)),
        "inversion_loss_columns": np.asarray(("total", "spectral", "prior")),
        "validation_full_wavelength_loss": np.asarray((0.2, 0.1)),
        "inversion_learning_rate_by_epoch": np.asarray((1.0e-3, 1.0e-5)),
        "inversion_learning_rate_schedule": np.asarray("sin_squared"),
        "inversion_initial_learning_rate": np.asarray(1.0e-3),
        "inversion_final_learning_rate": np.asarray(1.0e-5),
        "best_validation_epoch": np.asarray(1),
        "best_validation_loss": np.asarray(0.1),
        "final_full_cube_spectral_loss": np.asarray(0.1),
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
            field_config=pinn3d.NeuralFieldConfig(spatial_hidden=(8,)),
            inversion_epochs=0,
            show_progress=False,
        )


def test_inversion_starts_at_reference_without_pretraining_and_saves_outputs(
    tmp_path,
    tiny_polarized_cube,
):
    case = tiny_polarized_cube
    config = pinn3d.NeuralFieldConfig(spatial_hidden=(8,))
    checkpoint = tmp_path / "relative.npz"
    params, result = pinn3d.run_inversion(
        case["payload"],
        field_config=config,
        inversion_epochs=0,
        result_path=tmp_path / "result.npz",
        checkpoint_path=checkpoint,
        synthesis_batch_columns=2,
        validation_columns=2,
        wavelength_parallelism=2,
        show_progress=False,
    )
    assert "pretrain_loss" not in result and "base_layers" not in result
    assert pinn3d.validate_inversion_result(result) == (1, 2, 82, 5)
    np.testing.assert_array_equal(result["inferred_normalized_corrections"], 0.0)
    for field, profile in zip(
        pinn3d._atmosphere_from_payload(result, "inferred_"), case["reference"]
    ):
        np.testing.assert_array_equal(field, np.broadcast_to(profile, field.shape))
    restored, restored_config = pinn3d.load_checkpoint(checkpoint)
    assert restored_config == config
    for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(actual, expected)
    changed_reference = case["reference"]._replace(ne=case["reference"].ne * 2.0)
    with pytest.raises(ValueError, match="reference ne does not match"):
        pinn3d._validate_checkpoint_reference(
            restored, changed_reference, case["payload"]["height_normalized"]
        )
    with pytest.raises(ValueError, match="reference height_normalized does not match"):
        pinn3d._validate_checkpoint_reference(
            restored, case["reference"], case["payload"]["height_normalized"] ** 2
        )


def test_legacy_checkpoints_are_rejected_with_migration_guidance(tmp_path):
    checkpoint = tmp_path / "old.npz"
    np.savez(checkpoint, checkpoint_schema_version=1)
    with pytest.raises(ValueError, match="start without --checkpoint-in"):
        pinn3d.load_checkpoint(checkpoint)


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
    config = pinn3d.NeuralFieldConfig(spatial_hidden=(12,))
    params = pinn3d.initialize_neural_field(
        jax.random.PRNGKey(8), case["reference"], payload["height_normalized"], config
    )
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

    assert all(
        np.all(np.asarray(leaf) == 0.0) for leaf in jax.tree.leaves(grads["reference"])
    )
    initial_spatial = jax.tree.map(lambda value: value.copy(), params["spatial"])
    reported_metrics = []
    reported_evaluations = []
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
        metrics_callback=reported_metrics.append,
        evaluation_callback=lambda epoch, _params: reported_evaluations.append(epoch),
        evaluation_every=50,
    )
    for original, current in zip(
        jax.tree.leaves(params["reference"]), jax.tree.leaves(fitted["reference"])
    ):
        np.testing.assert_array_equal(current, original)
    assert history.shape == (1, 3)
    assert np.isfinite(history).all()
    assert validation_history.shape == (2,)
    assert np.isfinite(validation_history).all()
    assert [record["inversion/epoch"] for record in reported_metrics] == [0, 1]
    assert reported_evaluations == [0, 1]
    assert reported_metrics[0]["inversion/best_validation_loss"] == pytest.approx(
        validation_history[0]
    )
    assert reported_metrics[1]["inversion/train_total_loss"] == pytest.approx(
        history[0, 0]
    )
    assert reported_metrics[1][
        "inversion/validation_full_wavelength_loss"
    ] == pytest.approx(validation_history[1])
    assert validation_history[-1] < validation_history[0]
    fitted_loss = float(loss_fn(fitted))
    loss_rtol = 8.0 * np.finfo(np.asarray(offsets).dtype).eps
    np.testing.assert_allclose(fitted_loss, np.min(validation_history), rtol=loss_rtol)
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

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        pinn3d.importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else None,
    )
    assert (
        pinn3d.main(
            ["--generate-only", "--no-progress", "--wandb", "--wandb-mode", "offline"]
        )
        is None
    )
    assert fake_wandb.run.finish_codes == [0]
    assert fake_wandb.run.config["dataset_n_columns"] == 2
    assert "timing/generation_seconds" in fake_wandb.run.summary

    called = False

    def should_not_generate(**kwargs):
        nonlocal called
        called = True
        return tiny_polarized_cube["payload"]

    monkeypatch.setattr(pinn3d, "generate_test_cube", should_not_generate)
    with pytest.raises(SystemExit):
        pinn3d.main(["--skip-pretraining", "--no-progress"])
    assert not called


def test_wandb_run_is_finished_when_workflow_fails(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        pinn3d.importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else None,
    )

    def fail_generation(**_kwargs):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(pinn3d, "generate_test_cube", fail_generation)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        pinn3d.main(
            ["--generate-only", "--no-progress", "--wandb", "--wandb-mode", "offline"]
        )
    assert fake_wandb.run.finish_codes == [1]
