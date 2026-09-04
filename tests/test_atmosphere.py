import jax.numpy as jnp
import numpy as np
import pytest

from atmosphere import atmosphere_from_falc, cell_widths


class FakeFalc:
    z = np.array([3.0, 2.0, 1.0, 0.0])
    temperature = np.array([30.0, 20.0, 10.0, 0.0])
    ne = np.array([300.0, 200.0, 100.0, 0.0])
    nHTot = np.array([3000.0, 2000.0, 1000.0, 0.0])
    vz = np.array([3.0, 2.0, 1.0, 0.0])
    vturb = np.array([0.3, 0.2, 0.1, 0.0])


def test_cell_widths_are_positive_and_match_grid_length():
    widths = cell_widths([0.0, 2.0, 5.0, 9.0])
    np.testing.assert_array_equal(widths, [2.0, 2.0, 3.0, 4.0])
    assert widths.shape == (4,)
    assert bool(jnp.all(widths > 0.0))


@pytest.mark.parametrize("height", [[0.0], [0.0, 0.0], [1.0, 0.0], [0.0, np.nan]])
def test_cell_widths_reject_invalid_grids(height):
    with pytest.raises(ValueError):
        cell_widths(height)


def test_atmosphere_from_falc_reverses_every_depth_array_consistently():
    height, dz, temperature, ne, nhtot, vz, vturb = atmosphere_from_falc(FakeFalc())
    np.testing.assert_array_equal(height, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dz, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(temperature, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_array_equal(ne, [0.0, 100.0, 200.0, 300.0])
    np.testing.assert_array_equal(nhtot, [0.0, 1000.0, 2000.0, 3000.0])
    np.testing.assert_array_equal(vz, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(vturb, [0.0, 0.1, 0.2, 0.3])


@pytest.mark.parametrize("attribute", ["temperature", "ne", "nHTot", "vz", "vturb"])
def test_atmosphere_from_falc_rejects_mismatched_profiles(attribute):
    class InvalidFalc:
        z = FakeFalc.z
        temperature = FakeFalc.temperature
        ne = FakeFalc.ne
        nHTot = FakeFalc.nHTot
        vz = FakeFalc.vz
        vturb = FakeFalc.vturb

    setattr(InvalidFalc, attribute, np.ones(3))
    with pytest.raises(ValueError, match=attribute):
        atmosphere_from_falc(InvalidFalc())
