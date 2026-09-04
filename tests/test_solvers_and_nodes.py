import jax
import jax.numpy as jnp
import numpy as np
import pytest

import iterate
import iterate_adam
import iterate_scipy
from nodes import NodeSpec
from scalar_formal_solver import cumsum_fs, nearest_fs
from vector_formal_solver import (
    delo_constant_fs,
    delo_constant_fs_nonsingular,
    stokes_K,
)


@pytest.mark.parametrize("solver", [nearest_fs, cumsum_fs])
def test_scalar_constant_source_and_lower_boundary(solver):
    dz = jnp.array([0.3, 0.7, 1.1, 0.4])
    opacity = jnp.full(dz.shape, 0.8)
    source = 6.5
    emission = source * opacity
    lower_boundary = 2.0

    actual = jax.jit(solver)(dz, emission, opacity, lower_boundary)
    total_tau = jnp.sum(opacity * dz)
    expected = lower_boundary * jnp.exp(-total_tau) + source * (
        1.0 - jnp.exp(-total_tau)
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_scalar_cumsum_matches_iterative_for_depth_dependent_atmosphere():
    dz = jnp.array([0.2, 0.6, 1.4, 0.9, 0.3])
    emission = jnp.array([1.0, 8.0, 2.5, 11.0, 0.4])
    opacity = jnp.array([0.03, 0.7, 0.2, 1.4, 0.09])
    lower_boundary = 3.2

    iterative = nearest_fs(dz, emission, opacity, lower_boundary)
    vectorized = cumsum_fs(dz, emission, opacity, lower_boundary)

    np.testing.assert_allclose(vectorized, iterative, rtol=1e-12, atol=1e-12)


def test_scalar_cumsum_preserves_small_overlying_optical_depth():
    dz = jnp.ones(2)
    opacity = jnp.array([1e16, 1.0])
    emission = jnp.array([100.0e16, 0.0])

    iterative = nearest_fs(dz, emission, opacity)
    vectorized = cumsum_fs(dz, emission, opacity)
    expected = 100.0 * jnp.exp(-1.0)

    np.testing.assert_allclose(iterative, expected, rtol=2e-15)
    np.testing.assert_allclose(vectorized, expected, rtol=2e-15)


@pytest.mark.parametrize("solver", [nearest_fs, cumsum_fs])
def test_scalar_zero_opacity_preserves_emission_and_has_finite_derivatives(solver):
    dz = jnp.array([0.2, 0.5, 1.3])
    emission = jnp.array([2.0, 3.0, 5.0])
    opacity = jnp.zeros_like(dz)
    lower_boundary = 7.0

    actual = solver(dz, emission, opacity, lower_boundary)
    expected = lower_boundary + jnp.sum(emission * dz)
    derivatives = jax.jacrev(solver, argnums=(1, 2))(
        dz, emission, opacity, lower_boundary
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert all(bool(jnp.all(jnp.isfinite(derivative))) for derivative in derivatives)


@pytest.mark.parametrize("solver", [nearest_fs, cumsum_fs])
def test_scalar_solver_rejects_incompatible_shapes(solver):
    with pytest.raises(ValueError, match="identical shapes"):
        solver(jnp.ones(2), jnp.ones(3), jnp.ones(2))
    with pytest.raises(ValueError, match="one-dimensional"):
        solver(jnp.ones((2, 1)), jnp.ones(2), jnp.ones(2))
    with pytest.raises(ValueError, match="scalar"):
        solver(jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.ones(1))
    with pytest.raises(ValueError, match="at least one layer"):
        solver(jnp.array([]), jnp.array([]), jnp.array([]))


def test_stokes_propagation_matrix_layout():
    opacity = jnp.arange(1.0, 8.0)
    expected = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 7.0, -6.0],
            [3.0, -7.0, 1.0, 5.0],
            [4.0, 6.0, -5.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(stokes_K(opacity), expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_vector_solver_vacuum_pure_emission_is_finite(dtype):
    dz = jnp.array([99.0, 2.0, 3.0], dtype=dtype)
    lower_boundary = jnp.ones(4, dtype=dtype)
    emission = jnp.arange(12, dtype=dtype).reshape(3, 4)
    opacity = jnp.zeros((3, 7), dtype=dtype)

    actual = jax.jit(delo_constant_fs)(dz, lower_boundary, emission, opacity)
    expected = (
        lower_boundary
        + 0.5 * dz[1] * (emission[0] + emission[1])
        + 0.5 * dz[2] * (emission[1] + emission[2])
    )
    derivatives = jax.jacrev(delo_constant_fs, argnums=(2, 3))(
        dz, lower_boundary, emission, opacity
    )

    tolerance = 1e-6 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.dtype == dtype
    assert all(bool(jnp.all(jnp.isfinite(derivative))) for derivative in derivatives)


def test_vector_solver_unpolarized_constant_source_matches_analytic_solution():
    n_depth = 5
    dz = jnp.ones(n_depth)
    opacity_i = 0.4
    source_i = 8.0
    lower_boundary = jnp.array([2.0, 0.0, 0.0, 0.0])
    opacity = jnp.zeros((n_depth, 7)).at[:, 0].set(opacity_i)
    emission = jnp.zeros((n_depth, 4)).at[:, 0].set(opacity_i * source_i)

    actual = delo_constant_fs(dz, lower_boundary, emission, opacity)
    total_tau = opacity_i * jnp.sum(dz[1:])
    expected_i = lower_boundary[0] * jnp.exp(-total_tau) + source_i * (
        1.0 - jnp.exp(-total_tau)
    )

    np.testing.assert_allclose(actual[0], expected_i, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1:], 0.0, atol=1e-12)


def test_nonsingular_vector_solver_matches_robust_solver_when_vmapped():
    dz = jnp.array([0.0, 0.2, 0.7, 1.1])
    lower_boundary = jnp.array(
        [
            [2.0, 0.1, -0.2, 0.3],
            [3.0, -0.2, 0.4, -0.1],
        ]
    )
    emission = jnp.arange(32.0).reshape(2, 4, 4) / 20.0
    opacity = jnp.zeros((2, 4, 7))
    opacity = opacity.at[..., 0].set(
        jnp.array([[0.2, 0.3, 0.4, 0.5], [0.7, 0.6, 0.5, 0.4]])
    )
    opacity = opacity.at[..., 1:].set(
        jnp.arange(48.0).reshape(2, 4, 6) / 1000.0
    )

    def batched(solver, batch_emission, batch_opacity):
        return jax.vmap(solver, in_axes=(None, 0, 0, 0))(
            dz,
            lower_boundary,
            batch_emission,
            batch_opacity,
        )

    robust = jax.jit(lambda eta, chi: batched(delo_constant_fs, eta, chi))
    fast = jax.jit(
        lambda eta, chi: batched(delo_constant_fs_nonsingular, eta, chi)
    )
    np.testing.assert_allclose(
        fast(emission, opacity),
        robust(emission, opacity),
        rtol=1e-12,
        atol=1e-12,
    )

    def loss(solver, batch_emission, batch_opacity):
        return jnp.sum(batched(solver, batch_emission, batch_opacity) ** 2)

    robust_grad = jax.jit(jax.grad(loss, argnums=(1, 2)), static_argnums=0)(
        delo_constant_fs,
        emission,
        opacity,
    )
    fast_grad = jax.jit(jax.grad(loss, argnums=(1, 2)), static_argnums=0)(
        delo_constant_fs_nonsingular,
        emission,
        opacity,
    )
    for fast_part, robust_part in zip(fast_grad, robust_grad):
        np.testing.assert_allclose(
            fast_part,
            robust_part,
            rtol=1e-11,
            atol=1e-12,
        )


def test_vector_solver_thick_to_vacuum_interval_remains_exponential():
    dz = jnp.array([0.0, 10.0])
    lower_boundary = jnp.array([1.0, 0.0, 0.0, 0.0])
    emission = jnp.zeros((2, 4))
    opacity = jnp.zeros((2, 7)).at[0, 0].set(1.0)

    actual = delo_constant_fs(dz, lower_boundary, emission, opacity)
    expected = jnp.array([jnp.exp(-5.0), 0.0, 0.0, 0.0])
    derivatives = jax.jacrev(delo_constant_fs, argnums=(2, 3))(
        dz, lower_boundary, emission, opacity
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    assert all(bool(jnp.all(jnp.isfinite(part))) for part in derivatives)

    extreme_opacity = opacity.at[0, 0].set(1e10)
    extreme = delo_constant_fs(
        jnp.array([0.0, 1.0]), lower_boundary, emission, extreme_opacity
    )
    assert bool(jnp.all(jnp.isfinite(extreme)))
    np.testing.assert_array_equal(extreme, jnp.zeros(4))


def test_vector_solver_rejects_incompatible_shapes():
    dz = jnp.ones(3)
    lower_boundary = jnp.ones(4)
    emission = jnp.ones((3, 4))
    opacity = jnp.ones((3, 7))

    with pytest.raises(ValueError, match="I_start"):
        delo_constant_fs(dz, jnp.ones(3), emission, opacity)
    with pytest.raises(ValueError, match="emis"):
        delo_constant_fs(dz, lower_boundary, emission[:-1], opacity)
    with pytest.raises(ValueError, match="opac"):
        delo_constant_fs(dz, lower_boundary, emission, opacity[:, :-1])
    with pytest.raises(ValueError, match=r"shape \(7,\)"):
        stokes_K(jnp.ones(6))
    with pytest.raises(ValueError, match="at least one point"):
        delo_constant_fs(
            jnp.array([]), lower_boundary, jnp.empty((0, 4)), jnp.empty((0, 7))
        )


def test_single_point_vector_grid_returns_lower_boundary():
    lower_boundary = jnp.arange(4.0)
    actual = delo_constant_fs(
        jnp.ones(1), lower_boundary, jnp.zeros((1, 4)), jnp.zeros((1, 7))
    )
    np.testing.assert_array_equal(actual, lower_boundary)


def make_node_spec(**overrides):
    values = dict(
        z_min=0.0,
        z_max=10.0,
        n_temperature=2,
        n_ne=2,
        n_nhtot=2,
        n_vz=1,
        n_vturb=1,
        n_b=1,
        n_gamma_b=1,
        n_chi_b=1,
        n_interp=3,
    )
    values.update(overrides)
    return NodeSpec(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"z_max": 0.0}, "z_max"),
        ({"n_temperature": 0}, "n_temperature"),
        ({"n_interp": 1}, "n_interp"),
    ],
)
def test_node_spec_rejects_invalid_configuration(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        make_node_spec(**overrides)


def valid_node_parameters():
    return (
        jnp.array([5_000.0, 7_000.0]),
        jnp.array([1e16, 1e18]),
        jnp.array([1e20, 1e22]),
        jnp.array([100.0]),
        jnp.array([1_000.0]),
        jnp.array([0.05]),
        jnp.array([0.7]),
        jnp.array([1.2]),
    )


def test_node_pack_unpack_and_reconstruction_are_jittable():
    spec = make_node_spec()
    parameters = valid_node_parameters()
    packed = spec.pack_nodes(*parameters)

    assert packed.shape == (spec.n_nodes,)
    for actual, expected in zip(spec.unpack_nodes(packed), parameters):
        np.testing.assert_array_equal(actual, expected)

    reconstructed = jax.jit(spec.reconstruct_from_nodes)(packed)
    dz, temperature, ne, nhtot, *_ = reconstructed
    np.testing.assert_allclose(dz, 5.0)
    np.testing.assert_allclose(temperature, [5_000.0, 6_000.0, 7_000.0])
    np.testing.assert_allclose(ne, [1e16, 1e17, 1e18], rtol=1e-12)
    np.testing.assert_allclose(nhtot, [1e20, 1e21, 1e22], rtol=1e-12)
    jacobian = jax.jacrev(lambda value: spec.reconstruct_from_nodes(value)[1])(packed)
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_node_shapes_and_positive_domains_are_validated():
    spec = make_node_spec()
    parameters = list(valid_node_parameters())

    with pytest.raises(ValueError, match="nodes must have shape"):
        spec.unpack_nodes(jnp.ones(spec.n_nodes - 1))
    with pytest.raises(ValueError, match="temperature must have shape"):
        spec.pack_nodes(jnp.ones(1), *parameters[1:])

    parameters[1] = jnp.array([1e16, 0.0])
    with pytest.raises(ValueError, match="ne nodes must be strictly positive"):
        spec.reconstruct_atmos(*parameters)


@pytest.mark.parametrize(
    ("z", "message"),
    [
        (jnp.array([0.0, 5.0, 4.0, 10.0]), "strictly increasing"),
        (jnp.array([1.0, 5.0, 10.0]), "complete"),
    ],
)
def test_interp_to_nodes_validates_height_grid(z, message):
    spec = make_node_spec()
    values = jnp.ones(z.shape) * 5_000.0
    with pytest.raises(ValueError, match=message):
        spec.interp_to_nodes(z, values, values, values, values, values)


def test_single_nodes_interpolate_when_interval_contains_no_grid_sample():
    spec = make_node_spec(
        z_min=4.0,
        z_max=6.0,
        n_temperature=1,
        n_ne=1,
        n_nhtot=1,
    )
    z = jnp.array([0.0, 10.0])
    temperature = jnp.array([4_000.0, 8_000.0])
    ne = jnp.array([1e16, 1e18])
    nhtot = jnp.array([1e20, 1e22])
    velocity = jnp.array([0.0, 10.0])
    vturb = jnp.array([1_000.0, 2_000.0])

    result = spec.interp_to_nodes(
        z, temperature, ne, nhtot, velocity, vturb
    )
    assert all(bool(jnp.all(jnp.isfinite(part))) for part in result)
    np.testing.assert_allclose(result[0], [6_000.0])
    np.testing.assert_allclose(result[1], [5.05e17])


@pytest.mark.parametrize("module", [iterate, iterate_scipy, iterate_adam])
def test_imported_pack_params_uses_jax_and_is_jittable(module):
    arrays = tuple(jnp.arange(3.0) + index for index in range(5))
    packed = jax.jit(module.pack_params)(*arrays)
    expected = jnp.stack(arrays).reshape(-1)
    np.testing.assert_array_equal(packed, expected)


@pytest.mark.parametrize("module", [iterate, iterate_adam])
def test_unconstrained_parameterization_stays_finite_and_in_domain(module):
    parameters = module.pack_params(
        jnp.array([5_000.0, 8_000.0]),
        jnp.array([1e17, 1e20]),
        jnp.array([1e19, 1e23]),
        jnp.array([-2_000.0, 3_000.0]),
        jnp.array([500.0, 4_000.0]),
    )
    encoded = module.physical_to_unconstrained(parameters)
    decoded = module.unconstrained_to_physical(encoded)
    np.testing.assert_allclose(decoded, parameters, rtol=2e-14, atol=1e-8)

    extreme = module.unconstrained_to_physical(
        jnp.tile(jnp.array([-1e6, 1e6]), 5)
    ).reshape(5, -1)
    assert bool(jnp.all(jnp.isfinite(extreme)))
    assert bool(jnp.all(extreme[0] > 0.0))
    assert bool(jnp.all(extreme[1] > 0.0))
    assert bool(jnp.all(extreme[2] > 0.0))
    assert bool(jnp.all(extreme[4] >= 0.0))
