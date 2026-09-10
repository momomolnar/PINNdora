from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import visualize_pinn_3d_inversion as viz


@pytest.fixture
def visualization_archive(tmp_path):
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 1.0, 5)
    height_m = 1000.0 * np.asarray((0.0, 100.0, 300.0, 600.0, 1000.0, 1500.0, 2200.0))
    xx, yy, zz = np.meshgrid(
        x,
        y,
        (height_m - height_m[0]) / (height_m[-1] - height_m[0]),
        indexing="ij",
    )
    perturbation = (
        np.sin(np.pi * xx) * np.cos(np.pi * yy) * np.exp(-(((zz - 0.45) / 0.25) ** 2))
    )
    truth = {
        "temperature_K": 5000.0 + 1800.0 * zz + 300.0 * perturbation,
        "electron_density_m3": 1.0e20
        * np.exp(-3.2 * zz)
        * np.exp(-0.08 * perturbation),
        "hydrogen_density_m3": 1.0e23
        * np.exp(-3.0 * zz)
        * np.exp(0.1 * perturbation),
        "los_velocity_m_s": 1800.0 * perturbation,
        "magnetic_field_T": 0.05 * np.exp(0.08 * perturbation),
        "inclination_rad": 0.8 + 0.12 * perturbation,
        "azimuth_rad": 0.3 + 0.15 * perturbation,
    }
    inferred = {
        "temperature_K": truth["temperature_K"] * (1.0 + 0.01 * perturbation),
        "electron_density_m3": truth["electron_density_m3"]
        * np.exp(0.03 * perturbation),
        "hydrogen_density_m3": truth["hydrogen_density_m3"]
        * np.exp(-0.02 * perturbation),
        "los_velocity_m_s": 0.92 * truth["los_velocity_m_s"] + 40.0,
        "magnetic_field_T": truth["magnetic_field_T"] * np.exp(0.01 * perturbation),
        "inclination_rad": truth["inclination_rad"] - 0.01 * perturbation,
        "azimuth_rad": truth["azimuth_rad"] + 0.02 * perturbation,
    }
    payload = {
        "x_normalized": x,
        "y_normalized": y,
        "height_m": height_m,
        **{f"truth_{key}": value for key, value in truth.items()},
        **{f"inferred_{key}": value for key, value in inferred.items()},
    }
    path = tmp_path / "inversion.npz"
    np.savez_compressed(path, **payload)
    return path


def test_load_inversion_slices_validates_shapes_and_physics(visualization_archive):
    data = viz.load_inversion_slices(visualization_archive)
    assert data.shape == (4, 5, 7)
    assert set(data.truth) == {
        "temperature",
        "ne",
        "nhtot",
        "vz",
        "b",
        "gamma",
        "chi",
    }
    assert set(data.inferred) == set(data.truth)
    assert np.all(data.truth["temperature"] > 0.0)
    assert np.all(data.truth["ne"] > 0.0)
    assert np.all(data.truth["nhtot"] > 0.0)
    components = viz.thermodynamic_components(data.truth)
    np.testing.assert_allclose(components["log_ne"], np.log10(data.truth["ne"]))
    np.testing.assert_allclose(components["log_nhtot"], np.log10(data.truth["nhtot"]))


def test_height_selection_supports_indices_km_and_fractions(visualization_archive):
    height_m = viz.load_inversion_slices(visualization_archive).height_m
    np.testing.assert_array_equal(
        viz.resolve_height_indices(height_m, indices=(6, 0, 3)), (0, 3, 6)
    )
    np.testing.assert_array_equal(
        viz.resolve_height_indices(height_m, heights_km=(0.0, 600.0, 2200.0)),
        (0, 3, 6),
    )
    np.testing.assert_array_equal(viz.resolve_height_indices(height_m), (2, 4, 5))
    with pytest.raises(ValueError, match="fewer than three distinct"):
        viz.resolve_height_indices(height_m, fractions=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="inside"):
        viz.resolve_height_indices(height_m, heights_km=(-1.0, 600.0, 2200.0))


def test_magnetic_components_follow_inclination_azimuth_convention():
    fields = {
        "b": np.asarray((0.01, 0.02, 0.03)),
        "gamma": np.asarray((0.5 * np.pi, 0.5 * np.pi, 0.0)),
        "chi": np.asarray((0.0, 0.5 * np.pi, 0.7)),
    }
    components = viz.magnetic_components(fields)
    np.testing.assert_allclose(components["bx"], (100.0, 0.0, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(components["by"], (0.0, 200.0, 0.0), atol=1.0e-12)
    np.testing.assert_allclose(components["bz"], (0.0, 0.0, 300.0), atol=1.0e-12)


def test_component_grid_has_three_rows_columns_and_shared_colorbars(
    visualization_archive,
):
    data = viz.load_inversion_slices(visualization_archive)
    indices = viz.resolve_height_indices(data.height_m)
    truth = viz.thermodynamic_components(data.truth)
    inferred = viz.thermodynamic_components(data.inferred)
    norms = viz.shared_component_normalizations(
        truth, inferred, indices, viz.THERMODYNAMIC_SPECS
    )
    figure = viz.plot_component_grid(
        data,
        truth,
        indices,
        viz.THERMODYNAMIC_SPECS,
        norms,
        title="Test atmosphere",
    )
    try:
        n_columns = len(viz.THERMODYNAMIC_SPECS)
        n_panels = len(indices) * n_columns
        assert len(figure.axes) == n_panels + n_columns
        assert all(len(axis.images) == 1 for axis in figure.axes[:n_panels])
        assert [axis.get_title() for axis in figure.axes[:n_columns]] == [
            spec.title for spec in viz.THERMODYNAMIC_SPECS
        ]
    finally:
        plt.close(figure)


def test_generate_visualizations_writes_all_four_requested_figures(
    visualization_archive, tmp_path
):
    output_dir = tmp_path / "figures"
    paths, selected = viz.generate_visualizations(
        visualization_archive,
        output_dir,
        heights_km=(100.0, 600.0, 1500.0),
        prefix="experiment",
        dpi=50,
    )
    assert selected.tolist() == [1, 3, 5]
    assert {path.name for path in paths} == {
        "experiment_truth_thermodynamics.png",
        "experiment_inferred_thermodynamics.png",
        "experiment_truth_magnetic.png",
        "experiment_inferred_magnetic.png",
    }
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)


def test_cli_can_render_one_requested_subset(visualization_archive, tmp_path, capsys):
    output_dir = tmp_path / "subset"
    assert (
        viz.main(
            [
                "--result",
                str(visualization_archive),
                "--output-dir",
                str(output_dir),
                "--state",
                "inferred",
                "--quantity",
                "magnetic",
                "--height-indices",
                "0",
                "3",
                "6",
                "--dpi",
                "40",
            ]
        )
        is None
    )
    assert (output_dir / "pinn_3d_inferred_magnetic.png").is_file()
    output = capsys.readouterr().out
    assert "index 0 = 0.0 km" in output
    assert "Wrote" in output


def test_missing_visualization_fields_are_reported(tmp_path):
    path = Path(tmp_path) / "missing.npz"
    np.savez(path, x_normalized=np.asarray((0.0, 1.0)))
    with pytest.raises(ValueError, match="missing visualization fields"):
        viz.load_inversion_slices(path)
