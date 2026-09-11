from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import create_test_set as generator
import pinn_3d_inversion as pinn3d


def _reference_cube(nx=5, ny=5, n_depth=2):
    shape = (nx, ny, n_depth)
    return pinn3d.Atmosphere(
        jnp.full(shape, 6000.0),
        jnp.full(shape, 1.0e18),
        jnp.full(shape, 1.0e20),
        jnp.full(shape, -5000.0),
        jnp.full(shape, 1000.0),
        jnp.full(shape, 0.1),
        jnp.full(shape, 2.0 * jnp.pi / 3.0),
        jnp.zeros(shape),
    )


def test_cli_defaults_reproduce_requested_contrast():
    args = generator.build_argument_parser().parse_args([])

    assert (args.nx, args.ny, args.n_wave) == (51, 51, 201)
    assert args.temperature_center_ratio == pytest.approx(1.10)
    assert args.ne_center_ratio == pytest.approx(1.10)
    assert args.boundary_velocity_km_s == pytest.approx(-5.0)
    assert args.center_velocity_km_s == pytest.approx(5.0)
    assert args.boundary_b_los_gauss == pytest.approx(-500.0)
    assert args.center_b_los_gauss == pytest.approx(500.0)
    assert args.field_strength_gauss == pytest.approx(1000.0)
    assert args.profile_power == pytest.approx(1.0)
    assert args.thermodynamic_height_km is None
    assert args.thermodynamic_sigma_km == pytest.approx(100.0)
    assert args.temperature_boundary_k is None
    assert args.spatial_scale is None
    assert args.dataset == generator.DEFAULT_DATASET
    assert args.checkpoint == generator.DEFAULT_CHECKPOINT
    assert not args.overwrite


def test_cli_help_lists_every_public_option():
    help_text = generator.build_argument_parser().format_help()
    options = {
        "--precision",
        "--dataset",
        "--checkpoint",
        "--kurucz",
        "--seed",
        "--overwrite",
        "--no-progress",
        "--nx",
        "--ny",
        "--n-wave",
        "--wavelength-padding-nm",
        "--synthesis-batch-columns",
        "--wavelength-parallelism",
        "--profile-power",
        "--thermodynamic-height-km",
        "--thermodynamic-sigma-km",
        "--temperature-boundary-k",
        "--temperature-center-ratio",
        "--ne-center-ratio",
        "--boundary-velocity-km-s",
        "--center-velocity-km-s",
        "--boundary-b-los-gauss",
        "--center-b-los-gauss",
        "--field-strength-gauss",
        "--magnetic-azimuth-deg",
        "--spatial-hidden",
        "--spatial-scale",
        "--spatial-scale-safety-factor",
        "--spatial-scale-padding",
    }
    assert all(option in help_text for option in options)


def test_cli_rejects_even_grids_and_invalid_spatial_scale():
    parser = generator.build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--nx", "50"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--spatial-scale", "1,2,3"])


def test_center_boundary_envelope_has_exact_endpoints_and_no_height_change():
    x = jnp.linspace(0.0, 1.0, 5)
    y = jnp.linspace(0.0, 1.0, 7)
    envelope = np.asarray(generator.center_boundary_envelope(x, y, 3))

    assert envelope.shape == (5, 7, 3)
    np.testing.assert_allclose(envelope[0], 0.0)
    np.testing.assert_allclose(envelope[-1], 0.0)
    np.testing.assert_allclose(envelope[:, 0], 0.0)
    np.testing.assert_allclose(envelope[:, -1], 0.0)
    np.testing.assert_allclose(envelope[2, 3], 1.0)
    np.testing.assert_allclose(envelope[..., 0], envelope[..., 2])

    concentrated = np.asarray(
        generator.center_boundary_envelope(x, y, 3, profile_power=2.0)
    )
    assert np.all(concentrated <= envelope)
    np.testing.assert_allclose(concentrated[2, 3], 1.0)


def test_contrast_atmosphere_matches_all_requested_center_and_edge_values():
    reference_cube = _reference_cube()
    x = jnp.linspace(0.0, 1.0, 5)
    y = jnp.linspace(0.0, 1.0, 5)
    envelope = generator.center_boundary_envelope(x, y, 2)
    truth = generator.build_contrast_atmosphere(reference_cube, envelope)

    edge = (0, 2, 0)
    center = (2, 2, 0)
    assert float(truth.temperature[edge]) == pytest.approx(6000.0)
    assert float(truth.temperature[center]) == pytest.approx(6600.0)
    assert float(truth.ne[edge]) == pytest.approx(1.0e18)
    assert float(truth.ne[center]) == pytest.approx(1.1e18)
    assert float(truth.vz[edge]) == pytest.approx(-5000.0)
    assert float(truth.vz[center]) == pytest.approx(5000.0)

    b_los_gauss = np.asarray(
        generator.GAUSS_PER_TESLA * truth.b * jnp.cos(truth.gamma_b)
    )
    assert b_los_gauss[edge] == pytest.approx(-500.0)
    assert b_los_gauss[center] == pytest.approx(500.0)
    np.testing.assert_allclose(truth.b, 0.1)
    np.testing.assert_array_equal(truth.nhtot, reference_cube.nhtot)
    np.testing.assert_array_equal(truth.vturb, reference_cube.vturb)


def test_height_localized_thermodynamics_hit_the_selected_layer_exactly():
    reference_cube = _reference_cube(n_depth=5)
    temperature_profile = jnp.asarray((6000.0, 5500.0, 6000.0, 7000.0, 100_000.0))
    reference_cube = reference_cube._replace(
        temperature=jnp.broadcast_to(temperature_profile, (5, 5, 5))
    )
    reference = pinn3d.Atmosphere(*(field[0, 0] for field in reference_cube))
    x = jnp.linspace(0.0, 1.0, 5)
    y = jnp.linspace(0.0, 1.0, 5)
    height = jnp.asarray((-100_000.0, 0.0, 100_000.0, 200_000.0, 300_000.0))
    horizontal = generator.center_boundary_envelope(x, y, 5)
    localized, vertical, target_index, effective_height = (
        generator.height_localized_envelope(
            horizontal,
            height,
            target_height_km=100.0,
            sigma_km=100.0,
        )
    )

    assert target_index == 2
    assert effective_height == pytest.approx(100_000.0)
    assert float(vertical[target_index]) == pytest.approx(1.0)
    assert float(localized[2, 2, target_index]) == pytest.approx(1.0)
    assert float(localized[2, 2, 0]) < 1.0
    np.testing.assert_allclose(localized[0], 0.0)

    reference, reference_cube = generator.anchor_reference_temperature(
        reference,
        reference_cube,
        vertical,
        target_index=target_index,
        boundary_temperature_k=5000.0,
    )
    truth = generator.build_contrast_atmosphere(
        reference_cube,
        horizontal,
        thermodynamic_envelope=localized,
        temperature_center_ratio=1.2,
        ne_center_ratio=10.0,
        boundary_velocity_km_s=0.0,
        center_velocity_km_s=0.0,
        boundary_b_los_gauss=0.0,
        center_b_los_gauss=500.0,
        field_strength_gauss=1000.0,
    )
    pinn3d.validate_neural_transform_domain(reference)
    pinn3d.validate_neural_transform_domain(truth)

    edge = (0, 2, target_index)
    center = (2, 2, target_index)
    assert float(reference.temperature[target_index]) == pytest.approx(5000.0)
    assert float(truth.temperature[edge]) == pytest.approx(5000.0)
    assert float(truth.temperature[center]) == pytest.approx(6000.0)
    assert float(jnp.max(truth.temperature)) < 120_000.0
    assert float(truth.ne[center] / truth.ne[edge]) == pytest.approx(10.0)
    assert float(truth.ne[2, 2, 0] / reference_cube.ne[2, 2, 0]) < 10.0

    b_los_gauss = np.asarray(
        generator.GAUSS_PER_TESLA * truth.b * jnp.cos(truth.gamma_b)
    )
    assert b_los_gauss[edge] == pytest.approx(0.0, abs=1.0e-4)
    assert b_los_gauss[center] == pytest.approx(500.0)


def test_temperature_anchor_requires_a_thermodynamic_height(tmp_path):
    with pytest.raises(ValueError, match="requires thermodynamic_height_km"):
        generator.create_test_set(
            dataset_path=tmp_path / "cube.npz",
            checkpoint_path=tmp_path / "checkpoint.npz",
            temperature_boundary_k=5000.0,
        )


def test_auto_spatial_scale_is_reachable_and_explicit_scale_is_respected():
    reference_cube = _reference_cube()
    x = jnp.linspace(0.0, 1.0, 5)
    y = jnp.linspace(0.0, 1.0, 5)
    envelope = generator.center_boundary_envelope(x, y, 2)
    truth = generator.build_contrast_atmosphere(reference_cube, envelope)
    reference = pinn3d.Atmosphere(*(field[0, 0] for field in reference_cube))

    required, selected = generator.select_spatial_scale(reference, truth)

    assert required.shape == (8,)
    assert np.all(np.asarray(selected) > required)
    pinn3d.validate_spatial_reachability(reference, truth, selected)

    explicit = tuple(float(value) for value in required + 0.2)
    _, returned = generator.select_spatial_scale(
        reference,
        truth,
        explicit_scale=explicit,
    )
    assert returned == explicit
    with pytest.raises(ValueError, match="correction range"):
        generator.select_spatial_scale(
            reference,
            truth,
            explicit_scale=(0.01,) * 8,
        )


def test_magnetic_strength_must_exceed_signed_los_endpoints():
    with pytest.raises(ValueError, match="strictly greater"):
        generator._validate_magnetic_geometry(-500.0, 500.0, 500.0)


def test_existing_outputs_require_explicit_overwrite(tmp_path):
    dataset = tmp_path / "cube.npz"
    checkpoint = tmp_path / "checkpoint.npz"
    dataset.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--overwrite"):
        generator._validated_output_paths(
            dataset,
            checkpoint,
            overwrite=False,
        )
    assert generator._validated_output_paths(
        dataset,
        checkpoint,
        overwrite=True,
    ) == (dataset, checkpoint)
    with pytest.raises(ValueError, match="must be different"):
        generator._validated_output_paths(dataset, dataset, overwrite=True)


def test_main_forwards_cli_values_and_prints_outputs(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_create_test_set(**kwargs):
        captured.update(kwargs)
        return {
            "thermodynamic_mode": np.asarray("height_localized_gaussian"),
            "requested_thermodynamic_height_m": np.asarray(100_000.0),
            "effective_thermodynamic_height_m": np.asarray(95_000.0),
            "thermodynamic_target_index": np.asarray(12),
        }, SimpleNamespace(spatial_scale=(1.0,) * 8)

    monkeypatch.setattr(generator, "create_test_set", fake_create_test_set)
    dataset = tmp_path / "custom-cube.npz"
    checkpoint = tmp_path / "custom-checkpoint.npz"
    assert (
        generator.main(
            [
                "--dataset",
                str(dataset),
                "--checkpoint",
                str(checkpoint),
                "--temperature-center-ratio",
                "0.9",
                "--center-velocity-km-s",
                "3.5",
                "--profile-power",
                "2",
                "--thermodynamic-height-km",
                "100",
                "--thermodynamic-sigma-km",
                "75",
                "--temperature-boundary-k",
                "5000",
                "--overwrite",
                "--no-progress",
            ]
        )
        is None
    )

    assert captured["dataset_path"] == dataset
    assert captured["checkpoint_path"] == checkpoint
    assert captured["temperature_center_ratio"] == pytest.approx(0.9)
    assert captured["center_velocity_km_s"] == pytest.approx(3.5)
    assert captured["profile_power"] == pytest.approx(2.0)
    assert captured["thermodynamic_height_km"] == pytest.approx(100.0)
    assert captured["thermodynamic_sigma_km"] == pytest.approx(75.0)
    assert captured["temperature_boundary_k"] == pytest.approx(5000.0)
    assert captured["overwrite"]
    assert not captured["show_progress"]
    output = capsys.readouterr().out
    assert str(dataset) in output
    assert str(checkpoint) in output
    assert "requested 100 km" in output
    assert "layer 12 at 95 km" in output
