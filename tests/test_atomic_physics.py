from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import wofz

from adora_data import FE_I_6301_6302_LINE_LIST
import lineop
from contop import (
    HC,
    E_RYD,
    K_B,
    M_TO_NM,
    SAHA_CONST,
    continuum_opacity,
    gaunt_bf,
    h_bf_cont,
    hminus_bf_gray,
    lte_h_ion_fracs,
)
from lineop import (
    fe_pops,
    gamma_from_broadening,
    planck,
    read_kurucz,
)
from voigt import voigt_H


ROOT = Path(__file__).resolve().parents[1]
LINE_LIST = ROOT / "kurucz_6301_6302.linelist"


def test_packaged_line_list_matches_source_fixture():
    assert (
        FE_I_6301_6302_LINE_LIST.read_text().splitlines()
        == LINE_LIST.read_text().splitlines()
    )


def _replace_field(record, start, stop, value):
    width = stop - start
    if len(value) > width:
        raise ValueError(f"{value!r} does not fit in a {width}-character field")
    return record[:start] + value.rjust(width) + record[stop:]


def _write_modified_line_list(tmp_path, transform):
    record = LINE_LIST.read_text().splitlines()[0].ljust(154)
    path = tmp_path / "test.linelist"
    path.write_text("# parser must ignore comments\n\n" + transform(record) + "\n")
    return path


def test_kurucz_fixture_metadata_and_zeeman_components():
    lines = read_kurucz(LINE_LIST)

    np.testing.assert_allclose(
        lines.lambda0,
        [630.324277423091, 630.423704112446],
        rtol=0.0,
        atol=5e-13,
    )
    np.testing.assert_array_equal(lines.gi, [5.0, 3.0])
    np.testing.assert_array_equal(lines.gj, [5.0, 1.0])
    np.testing.assert_allclose(
        lines.Aji,
        [7_145_689.18541003, 12_413_010.612455],
        rtol=2e-13,
    )
    np.testing.assert_array_equal(lines.line_weight, [1.0, 1.0])
    assert np.all(np.asarray(lines.ej) > np.asarray(lines.ei))
    photon_energy = lineop.HC / (
        np.asarray(lines.lambda0) * lineop.NM_TO_M
    ) / lineop.Q_ELE
    np.testing.assert_allclose(lines.ej - lines.ei, photon_energy, rtol=2e-15)

    # The first transition has 13 anomalous components, including one
    # physically zero-strength pi component.  The second has three components
    # and is padded to the common width.
    np.testing.assert_array_equal(
        lines.zeeman_alphas[0],
        [0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0],
    )
    np.testing.assert_allclose(
        lines.zeeman_strengths[0],
        [0.4, 0.2, 0.2, 0.1, 0.3, 0.3, 0.0, 0.3, 0.3, 0.1, 0.2, 0.2, 0.4],
    )
    np.testing.assert_array_equal(lines.zeeman_alphas[1, :3], [-1, 0, 1])
    np.testing.assert_allclose(lines.zeeman_strengths[1, :3], 1.0)
    np.testing.assert_allclose(lines.zeeman_shifts[1, :3], [-2.5, 0.0, 2.5])
    np.testing.assert_allclose(lines.zeeman_strengths[1, 3:], 0.0)
    np.testing.assert_array_equal(
        lines.zeeman_component_lines,
        [0] * 12 + [1] * 3,
    )
    np.testing.assert_array_equal(
        lines.zeeman_component_alphas,
        [0, -1, 1, 0, -1, 1, -1, 1, 0, -1, 1, 0, -1, 0, 1],
    )
    np.testing.assert_allclose(
        lines.zeeman_component_strengths,
        [0.4, 0.2, 0.2, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.2, 0.2, 0.4,
         1.0, 1.0, 1.0],
    )
    for line_index in range(2):
        for alpha in (-1, 0, 1):
            mask = np.asarray(lines.zeeman_alphas[line_index]) == alpha
            assert np.asarray(lines.zeeman_strengths[line_index])[mask].sum() == pytest.approx(1.0)


def test_kurucz_negative_energies_fractions_and_zero_damping(tmp_path):
    def transform(record):
        record = _replace_field(record, 24, 36, "-45333.872")
        record = _replace_field(record, 80, 86, "0.00")
        record = _replace_field(record, 86, 92, "0.00")
        record = _replace_field(record, 92, 98, "0.00")
        record = _replace_field(record, 109, 115, "-0.301")
        return _replace_field(record, 118, 124, "-0.699")

    lines = read_kurucz(_write_modified_line_list(tmp_path, transform))
    assert float(lines.ei[0]) >= 0.0
    assert float(lines.ej[0]) > float(lines.ei[0])
    assert float(lines.line_weight[0]) == pytest.approx(0.1)
    assert np.isneginf(float(lines.log_grad[0]))
    assert np.isneginf(float(lines.log_gs[0]))
    assert np.isneginf(float(lines.log_gw[0]))
    gamma = gamma_from_broadening(
        lines.log_grad[0], lines.log_gs[0], lines.log_gw[0], 5000.0, 1e20, 1e23
    )
    assert float(gamma) == 0.0


def test_kurucz_rejects_unsupported_species_and_empty_files(tmp_path):
    unsupported = _write_modified_line_list(
        tmp_path,
        lambda record: _replace_field(record, 18, 24, "12.00"),
    )
    with pytest.raises(NotImplementedError, match="only Fe I"):
        read_kurucz(unsupported)

    empty = tmp_path / "empty.linelist"
    empty.write_text("# no records\n\n")
    with pytest.raises(ValueError, match="contains no records"):
        read_kurucz(empty)


def test_indeterminate_terms_fall_back_to_unpolarized_profile(tmp_path):
    def transform(record):
        record = _replace_field(record, 42, 52, "unknown")
        return _replace_field(record, 70, 80, "unknown")

    lines = read_kurucz(_write_modified_line_list(tmp_path, transform))
    np.testing.assert_array_equal(lines.zeeman_alphas[0], [-1, 0, 1])
    np.testing.assert_array_equal(lines.zeeman_strengths[0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(lines.zeeman_shifts[0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "temperature, ne, nhtot",
    [(3_000.0, 1e22, 1e24), (10_000.0, 1e18, 1e20), (100_000.0, 1e12, 1e20)],
)
def test_fe_populations_conserve_abundance_and_are_finite(temperature, ne, nhtot):
    abundance = 3e-5
    populations = jnp.asarray(fe_pops(abundance, temperature, ne, nhtot))
    assert bool(jnp.all(jnp.isfinite(populations)))
    assert bool(jnp.all(populations >= 0.0))
    np.testing.assert_allclose(populations.sum(), abundance * nhtot, rtol=2e-14)

    jacobian = jax.jacrev(lambda temp: jnp.asarray(fe_pops(abundance, temp, ne, nhtot)))(
        temperature
    )
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_fe_population_ratios_use_kurucz_ionization_potentials():
    temperature = 6000.0
    ne = 1e19
    populations = jnp.asarray(fe_pops(3e-5, temperature, ne, 1e23))
    ionpot = np.asarray(lineop._PARTITION_TABLE.ionpot[26 - 1][:2]) / lineop.Q_ELE
    saha = 2.0 * lineop.SAHA_CONST * temperature**1.5 / ne
    expected_ii_i = (
        lineop.Q_FeII(temperature)
        / lineop.Q_FeI(temperature)
        * saha
        * np.exp(-ionpot[0] / (lineop.K_B_EV * temperature))
    )
    expected_iii_ii = (
        lineop.Q_FeIII(temperature)
        / lineop.Q_FeII(temperature)
        * saha
        * np.exp(-ionpot[1] / (lineop.K_B_EV * temperature))
    )
    np.testing.assert_allclose(populations[1] / populations[0], expected_ii_i)
    np.testing.assert_allclose(populations[2] / populations[1], expected_iii_ii)


def test_planck_wien_tail_has_finite_zero_derivative():
    wavelength = 100.0
    temperature = 10.0
    value = planck(wavelength, temperature)
    derivative = jax.grad(lambda temp: planck(wavelength, temp))(temperature)
    assert float(value) == 0.0
    assert float(derivative) == 0.0

    moderate_temperature = 5000.0
    exponent = lineop.HC_KB_NM / (630.0 * moderate_temperature)
    expected = lineop.TWOHC2_NM5 / (630.0**5 * np.expm1(exponent))
    np.testing.assert_allclose(planck(630.0, moderate_temperature), expected)


def test_hydrogen_bound_free_profile_has_cubic_wavelength_scaling_and_edge():
    level = 2
    n = level + 1
    edge = HC * n**2 / E_RYD * M_TO_NM
    wavelength_1 = 0.7 * edge
    wavelength_2 = 0.9 * edge
    cross_section_1 = h_bf_cont(wavelength_1, level)
    cross_section_2 = h_bf_cont(wavelength_2, level)
    gaunt_ratio = gaunt_bf(wavelength_1, n, 1.0) / gaunt_bf(wavelength_2, n, 1.0)
    expected_ratio = gaunt_ratio * (wavelength_1 / wavelength_2) ** 3
    np.testing.assert_allclose(cross_section_1 / cross_section_2, expected_ratio, rtol=2e-14)
    assert float(h_bf_cont(edge, level)) > 0.0
    assert float(h_bf_cont(jnp.nextafter(edge, jnp.inf), level)) == 0.0


def test_weak_hydrogen_ionization_does_not_cancel():
    temperature = 3000.0
    ne = 1e22
    nhtot = 1e24
    ratio = (
        SAHA_CONST
        * temperature**1.5
        * np.exp(-E_RYD / (K_B * temperature))
        / ne
    )
    expected_nhii = nhtot * ratio / (1.0 + ratio)
    nhi, nhii = lte_h_ion_fracs(temperature, ne, nhtot)

    np.testing.assert_allclose(nhi + nhii, nhtot, rtol=2e-15)
    np.testing.assert_allclose(nhii, expected_nhii, rtol=2e-14)
    assert float(nhii) > 0.0

    lyman_continuum = continuum_opacity(90.0, temperature, ne, nhtot)
    assert bool(jnp.isfinite(lyman_continuum))
    assert float(lyman_continuum) > 0.0
    derivative = jax.grad(lambda temp: continuum_opacity(90.0, temp, ne, nhtot))(
        temperature
    )
    assert bool(jnp.isfinite(derivative))


def test_hminus_bound_free_wishart_polynomial_reference():
    wavelength = 630.0
    temperature = 6000.0
    ne = 1e20
    nhi = 1e23
    wavelength_a = wavelength * 10.0
    alpha = 35.57800518069265
    pressure_e = ne * 1.380649e-23 * temperature * 10.0
    theta = 5040.0 / temperature
    expected_cm = (
        4.158e-10
        * alpha
        * 1e-18
        * pressure_e
        * theta**2.5
        * 10 ** (0.754 * theta)
        * (nhi * 1e-6)
    )
    assert wavelength_a == 6300.0
    np.testing.assert_allclose(
        hminus_bf_gray(wavelength, temperature, ne, nhi), expected_cm * 1e2, rtol=2e-14
    )


def test_line_profile_uses_line_centre_for_damping(monkeypatch):
    lines = read_kurucz(LINE_LIST)
    seen_wavelengths = []
    original = lineop.damping_from_gamma

    def capture(gamma, lambda0, doppler_width):
        seen_wavelengths.append(lambda0)
        return original(gamma, lambda0, doppler_width)

    monkeypatch.setattr(lineop, "damping_from_gamma", capture)
    lineop.emis_opac_line(
        lines.mass[0],
        lines.abund[0],
        lines.lambda0[0],
        lines.log_grad[0],
        lines.log_gs[0],
        lines.log_gw[0],
        lines.gj[0],
        lines.ej[0],
        lines.Aji[0],
        lines.line_weight[0],
        lines.lambda0[0] + 0.1,
        5000.0,
        1e20,
        1e23,
        0.0,
        2e3,
    )
    assert seen_wavelengths == [lines.lambda0[0]]


def test_zero_field_polarised_coefficients_reduce_to_scalar_coefficients():
    lines = read_kurucz(LINE_LIST)
    scalar_eta, scalar_chi = lineop.emis_opac(
        lines, 630.3243, 5777.0, 1e19, 1e23, 250.0, 1500.0
    )
    polarised_eta, polarised_chi = lineop.emis_opac_polarised(
        lines,
        630.3243,
        5777.0,
        1e19,
        1e23,
        250.0,
        1500.0,
        0.0,
        0.7,
        0.3,
    )
    np.testing.assert_allclose(
        polarised_eta, [scalar_eta, 0.0, 0.0, 0.0], rtol=2e-15, atol=1e-18
    )
    np.testing.assert_allclose(
        polarised_chi,
        [scalar_chi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        rtol=2e-15,
        atol=1e-20,
    )


def test_packed_zeeman_kernel_matches_padded_values_and_atmospheric_gradients():
    lines = read_kurucz(LINE_LIST)
    padded_lines = replace(
        lines,
        zeeman_component_lines=None,
        zeeman_component_alphas=None,
        zeeman_component_strengths=None,
        zeeman_component_shifts=None,
    )
    wavelength = 630.3243
    atmosphere = (5777.0, 1e19, 1e23, 250.0, 1500.0, 0.1, 0.7, 0.3)

    def coefficients(atomic_data, *atmospheric_args):
        eta, chi = lineop.emis_opac_polarised(
            atomic_data, wavelength, *atmospheric_args
        )
        return jnp.concatenate((eta, chi))

    packed_values = coefficients(lines, *atmosphere)
    padded_values = coefficients(padded_lines, *atmosphere)
    np.testing.assert_allclose(
        packed_values, padded_values, rtol=2e-15, atol=1e-30
    )

    atmospheric_argnums = tuple(range(1, 9))
    jacobian = jax.jacrev(coefficients, argnums=atmospheric_argnums)
    packed_jacobian = jacobian(lines, *atmosphere)
    padded_jacobian = jacobian(padded_lines, *atmosphere)
    for packed_derivative, padded_derivative in zip(
        packed_jacobian, padded_jacobian
    ):
        np.testing.assert_allclose(
            packed_derivative,
            padded_derivative,
            rtol=2e-15,
            atol=1e-42,
        )


def test_voigt_matches_scipy_and_has_finite_derivatives():
    for damping in (0.0, 1e-4, 0.1, 1.0, 10.0):
        for offset in (-20.0, -5.0, 0.0, np.sqrt(0.5), 5.0, 20.0):
            absorption, dispersion = voigt_H(damping, offset)
            reference = wofz(offset + 1j * damping)
            np.testing.assert_allclose(absorption, reference.real, rtol=5e-5, atol=2e-7)
            np.testing.assert_allclose(dispersion, reference.imag, rtol=5e-5, atol=2e-7)
            derivative = jax.jacrev(lambda a, v: jnp.asarray(voigt_H(a, v)), argnums=(0, 1))(
                damping, offset
            )
            assert all(bool(jnp.all(jnp.isfinite(part))) for part in derivative)


def test_voigt_core_to_wing_transition_is_continuous():
    damping = 7.405
    transition_offset = 15.0 - damping
    epsilon = 1e-8
    left = jnp.asarray(voigt_H(damping, transition_offset - epsilon))
    right = jnp.asarray(voigt_H(damping, transition_offset + epsilon))
    np.testing.assert_allclose(left, right, rtol=2e-8, atol=2e-9)
