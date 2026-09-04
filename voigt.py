from adora_precision import REAL_DTYPE
import jax
import jax.numpy as jnp


def _real_inputs(a, v):
    a = jnp.asarray(a)
    v = jnp.asarray(v)
    dtype = jnp.result_type(a, v)
    if not jnp.issubdtype(dtype, jnp.floating) or dtype.itemsize < 4:
        dtype = jnp.dtype(REAL_DTYPE)
    return jnp.broadcast_arrays(a.astype(dtype), v.astype(dtype))


def voigt_H_impl(a, v):
    a, v = _real_inputs(a, v)
    dtype = a.dtype
    l = jnp.asarray(4.1195342878142354, dtype=dtype)
    # l=sqrt(n/sqrt(2.))  ! L = 2**(-1/4) * N**(1/2)
    ac = jnp.asarray((
        -1.5137461654527820e-10, 4.9048215867870488e-09,
        1.3310461806370372e-09,  -3.0082822811202271e-08,
        -1.9122258522976932e-08, 1.8738343486619108e-07,
        2.5682641346701115e-07,  -1.0856475790698251e-06,
        -3.0388931839840047e-06, 4.1394617248575527e-06,
        3.0471066083243790e-05,  2.4331415462641969e-05,
        -2.0748431511424456e-04, -7.8166429956142650e-04,
        -4.9364269012806686e-04, 6.2150063629501763e-03,
        3.3723366855316413e-02,  1.0838723484566792e-01,
        2.6549639598807689e-01,  5.3611395357291292e-01,
        9.2570871385886788e-01,  1.3948196733791203e+00,
        1.8562864992055408e+00,  2.1978589365315417e+00
    ), dtype=dtype)

    s = jnp.abs(v) + a
    # Region I
    z = jax.lax.complex(a, -v)
    # Blend the rational wing approximation into the Weideman core rather than
    # hard-switching at s=15.  The two approximations differ at the few-ppm
    # level there; a hard branch creates a real discontinuity in inversion
    # objectives and their AD derivatives.
    blend_start = jnp.asarray(14.5, dtype=dtype)
    blend_stop = jnp.asarray(15.5, dtype=dtype)
    evaluate_reg1 = s >= blend_start
    reg1_denominator = jnp.asarray(0.5, dtype=dtype) + z * z
    # The asymptotic branch has poles inside the region where it is inactive.
    # Mask them before division so reverse-mode AD never sees an undefined
    # value from the unselected branch of jnp.where.
    safe_reg1_denominator = jnp.where(
        evaluate_reg1,
        reg1_denominator,
        jnp.ones_like(reg1_denominator),
    )
    sqrt_pi_inverse = jnp.asarray(0.5641896, dtype=dtype)
    reg1 = (z * sqrt_pi_inverse) / safe_reg1_denominator

    recLmZ = 1.0 / jax.lax.complex(l + a, -v)
    t = jax.lax.complex(l - a, v) * recLmZ
    wei24 = recLmZ * (
        sqrt_pi_inverse + 2.0 * recLmZ * (
        ac[23]+(ac[22]+(ac[21]+(ac[20]+(ac[19]+(ac[18]+(ac[17]+(ac[16]+(ac[15]+(ac[14]+(ac[13]+(ac[12]+(ac[11]+(ac[10]+(ac[9]+(ac[8]+
        (ac[7]+(ac[6]+(ac[5]+(ac[4]+(ac[3]+(ac[2]+(ac[1]+ac[0]*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t)*t
    ))
    blend_coordinate = jnp.clip(
        (s - blend_start) / (blend_stop - blend_start), 0.0, 1.0
    )
    blend_weight = blend_coordinate**2 * (3.0 - 2.0 * blend_coordinate)
    result = (1.0 - blend_weight) * wei24 + blend_weight * reg1
    return result.real, result.imag

# @jax.custom_jvp
def voigt_H(a, v):
    return voigt_H_impl(a, v)

# @voigt_H.defjvp
# def voigt_H_jvp(primals, tangents):
#     a, v = primals
#     a_dot, v_dot = tangents
#     grad = jax.jacrev(voigt_H_impl, argnums=(0, 1), holomorphic=True)
#     primal_out = voigt_H_impl(a, v)
#     jac = grad(jnp.complex_(a), jnp.complex_(v))
#     tangent_out = jac[0] * a_dot + jac[1] * v_dot
#     breakpoint()
#     return primal_out, tangent_out

def voigt_H_re(a, v):
    return voigt_H_impl(a, v)[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        from IPython import get_ipython
        ipython = get_ipython()
    except ImportError:
        ipython = None
    if ipython is None:
        plt.ion()
    else:
        ipython.run_line_magic("matplotlib", "")

    a = jnp.array([0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0])
    v = jnp.linspace(-400, 400, 101)

    ia = 3
    voigt_H_jit = jax.jit(jax.vmap(voigt_H))
    HF = voigt_H_jit(jnp.ones(101) * a[ia], v)
    H = HF[0]
    F = HF[1]
    plt.figure()
    plt.plot(v, H)
    plt.plot(v, F)

    dvoigt = jax.jit(jax.vmap(jax.jacrev(voigt_H, argnums=(0, 1))))
    grads = dvoigt(jnp.ones(101) * a[ia], v)
    dHda = grads[0][0]
    dFda = grads[0][1]
    dHdv = grads[1][0]
    dFdv = grads[1][1]

    dvoigt_re = jax.jit(jax.vmap(jax.jacrev(voigt_H_re, argnums=(0, 1))))
    grads_re = dvoigt_re(jnp.ones(101) * a[ia], v)
    dvda = grads_re[0]
    dvdv = grads_re[1]

    plt.figure()
    plt.plot(v, dHda)
    plt.plot(v, dHdv)
    plt.plot(v, dFda)
    plt.plot(v, dFdv)
    plt.plot(v, dvda, '--')
    plt.plot(v, dvdv, '--')
