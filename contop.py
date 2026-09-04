from adora_precision import REAL_DTYPE
import jax
import jax.numpy as jnp
import astropy.constants as const
import astropy.units as u
import numpy as np


def _real(value):
    return jnp.asarray(value, dtype=REAL_DTYPE)


HC = _real(const.h.value * const.c.value)
NM_TO_M = _real(u.Unit('nm').to('m'))
M_TO_NM = _real(u.Unit('m').to('nm'))
E_RYD = _real(const.Ryd.to('J', equivalencies=u.spectral()).value)
E_RYD_OVER_K_B = _real(
    const.Ryd.to('J', equivalencies=u.spectral()).value / const.k_B.value
)
HC_OVER_E_RYD_NM = _real(
    const.h.value
    * const.c.value
    / const.Ryd.to('J', equivalencies=u.spectral()).value
    * float(u.Unit('m').to('nm'))
)
Q_ELE = _real(u.eV.to(u.J))
EPS_0 = _real(const.eps0.value)
M_ELE = _real(const.m_e.value)
K_B = _real(const.k_B.value)
H_CROSS_SECTION_C0 = _real(
    32.0
    / (3.0 * np.sqrt(3.0))
    * (float(u.eV.to(u.J)) / np.sqrt(4.0 * np.pi * const.eps0.value)) ** 2
    / (const.m_e.value * const.c.value)
    * const.h.value
    / (2.0 * const.Ryd.to('J', equivalencies=u.spectral()).value)
)
N_H_CONT = 5
SAHA_CONST = _real(
    ((2 * np.pi * const.m_e.value * const.k_B.value) / const.h.value**2) ** 1.5
)

def gaunt_bf(wvl, nEff, charge) -> float:
    '''
    Gaunt factor for bound-free transitions, from Seaton (1960), Rep. Prog.
    Phys. 23, 313, as used in RH.

    Parameters
    ----------
    wvl : float or array-like
        The wavelength at which to compute the Gaunt factor [nm].
    nEff : float
        Principal quantum number.
    charge : float
        Charge of free state.

    Returns
    -------
    result : float or array-like
        Gaunt factor for bound-free transitions.
    '''
    # /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313 -- ----------- */
    x = HC_OVER_E_RYD_NM / (wvl * charge**2)
    x3 = x**(1.0/3.0)
    nsqx = 1.0 / (nEff**2 * x)

    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) - 0.0496 * x3**2 \
            * (1.0 - (1.0 - nsqx) * (2.0 / 3.0) * nsqx)

def h_bf_cont(wvl, i):
    """
    wvl: float
        The wavelength in nm
    i: int
        The lower level of the continuum (n), 0-indexed
    """

    Z = 1.0
    n = i + 1
    lambda_edge = HC_OVER_E_RYD_NM * n**2
    alpha0 = H_CROSS_SECTION_C0 * n * gaunt_bf(lambda_edge, n, 1.0)
    gbf0 = gaunt_bf(lambda_edge, n, Z)
    gbf = gaunt_bf(wvl, n, Z)
    alpha = jnp.where(
        wvl <= lambda_edge,
        alpha0 * gbf / gbf0 * (wvl / lambda_edge)**3,
        0.0
    )
    return alpha

def hminus_ff_gray(wvl, temperature, ne, nhi):
    """
    wvl: float
        The wavelength in nm
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhi: float
        The number density of neutral H in m-3

    Computes the absorption in m-1 due to H- ff

    Follows Gray p.141 (2021 online)
    https://www.cambridge.org/highereducation/books/the-observation-and-analysis-of-stellar-photospheres/67B340445C56F4421BCBA0AFFAAFDEE0#contents
    """
    wvl_a = wvl * 10.0
    x1 = jnp.log10(wvl_a)
    x2 = x1 * x1
    x3 = x2 * x1
    f0 = -2.2763 - 1.6850 * x1 + 0.76661 * x2 - 0.0533464 * x3
    f1 = 15.2827 - 9.2846 * x1 + 1.99381 * x2 - 0.142631 * x3
    f2 = -197.789 + 190.266 * x1 - 67.9775 * x2 + 10.6913 * x3 - 0.625151 * x3 * x1
    thermal_log = jnp.log10(5040.0 / temperature)
    p_e = ne * K_B * temperature * 10 # to dyn/cm2
    sigma = jnp.where(
        (wvl > 260.0) & (wvl < 11390.0),
        1e-26 * p_e * 10**(f0 + f1 * thermal_log + f2 * thermal_log**2) * (nhi * 1e-6),
        0.0
    ) # in cm-1
    return sigma * 1e2

def hminus_bf_gray(wvl, temperature, ne, nhi):
    """
    wvl: float
        The wavelength in nm
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhi: float
        The number density of neutral H in m-3

    Computes the absorption in m-1 due to H- bf

    Follows Gray p.140 (2021 online)
    https://www.cambridge.org/highereducation/books/the-observation-and-analysis-of-stellar-photospheres/67B340445C56F4421BCBA0AFFAAFDEE0#contents
    """

    wvl_a = wvl * 10.0
    # Wishart's cross-section fit, in units of 1e-18 cm2 per H- ion.
    alpha = 1.99654 + (-1.18267e-5 + (2.64243e-6 + (-4.40524e-10 + (3.23992e-14 + (-1.39568e-18 + 2.78701e-23 * wvl_a) * wvl_a) * wvl_a) * wvl_a) * wvl_a) * wvl_a
    p_e = ne * K_B * temperature * 10 # dyn/cm2
    theta = 5040.0 / temperature

    sigma = jnp.where(
        (wvl > 150.0) & (wvl < 1605.0),
        4.158e-10 * alpha * 1e-18 * p_e * theta**(2.5) * 10 ** (0.754 * theta) * (nhi * 1e-6),
        0.0
    ) # in cm-1
    return sigma * 1e2

def lte_h_ion_fracs(temperature, ne, nhtot):
    """
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhtot: float
        The number density of H (I and II) in m-3

    Computes nhi and nhii for the given point
    """
    # 2 g_hii / g_hi = 1
    log_ratio = (
        jnp.log(SAHA_CONST)
        + 1.5 * jnp.log(temperature)
        - E_RYD_OVER_K_B / temperature
        - jnp.log(ne)
    )
    # Computing nhii as nhtot - nhi erases the ionized population when the gas
    # is weakly ionized.  Logistic fractions remain accurate in both tails and
    # sum to nhtot without subtracting nearly equal large numbers.
    nhi = nhtot * jax.nn.sigmoid(-log_ratio)
    nhii = nhtot * jax.nn.sigmoid(log_ratio)
    return nhi, nhii

def continuum_opacity(wvl, temperature, ne, nhtot):
    """
    wvl: float
        The wavelength in nm
    temperature: float
        The temperature in K
    ne: float
        The electron density in m-3
    nhtot: float
        The number density of H (I and II) in m-3

    Computes the continuum opacity in m-1
    """
    stimulated_exponent = (
        -1.2398e3 * 5040.0 * jnp.log(10.0) / (wvl * temperature)
    )
    stimulated_correction = -jnp.expm1(stimulated_exponent)
    nhi, _ = lte_h_ion_fracs(temperature, ne, nhtot)

    abs_h_bf = 0.0
    for i in range(0, N_H_CONT):
        n = i + 1
        # Algebraically equivalent to the level-specific Saha expression, but
        # referenced to stable neutral-H population.  In particular the n=1
        # population is exactly nhi even when nhii is vanishingly small.
        excitation_over_kT = (
            E_RYD_OVER_K_B * (1.0 - 1.0 / n**2) / temperature
        )
        pop = nhi * n**2 * jnp.exp(-excitation_over_kT)
        abs_h_bf = abs_h_bf + h_bf_cont(wvl, i) * pop

    abs_hm_bf = hminus_bf_gray(wvl, temperature, ne, nhi)
    abs_hm_ff = hminus_ff_gray(wvl, temperature, ne, nhi)

    return (abs_h_bf + abs_hm_bf) * stimulated_correction + abs_hm_ff

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

    plt.figure()

    wave = jnp.linspace(50, 2000, 1001)
    temperature = 7715.0
    pe = 10**2.5 / 10
    # temperature = 6429.0
    # pe = 10**1.77 / 10
    ne = pe / (temperature * K_B)
    nhtot = 1e3 * ne
    continuum_opacity_jit = jax.jit(jax.vmap(continuum_opacity, in_axes=[0, None, None, None]))
    cop = continuum_opacity_jit(wave, temperature, ne, nhtot)

    plt.semilogy(wave, cop)

    import lightweaver as lw
    from lightweaver.rh_atoms import H_atom, CaII_atom
    import numpy as np
    n_depth = 2
    atmos = lw.Atmosphere.make_1d(
        lw.ScaleType.Geometric,
        depthScale=np.linspace(1, 0, n_depth) * 1e3,
        temperature=np.ones(n_depth) * temperature,
        vlos=np.zeros(n_depth),
        vturb=np.ones(n_depth) * 2e3,
        ne=np.ones(n_depth) * ne,
        nHTot=np.ones(n_depth) * nhtot
    )
    atmos.quadrature(3)
    rad_set = lw.RadiativeSet([H_atom(), CaII_atom()])
    rad_set.set_active("Ca")
    eq_pops = rad_set.compute_eq_pops(atmos)
    spect = rad_set.compute_wavelength_grid(extraWavelengths=wave)

    ctx = lw.Context(atmos, spect, eq_pops)
    plt.plot(ctx.spect.wavelength, ctx.background.chi[:, 0])


    dcont_op = jax.jacrev(continuum_opacity_jit, argnums=(1, 2, 3))
    dcop = dcont_op(wave, temperature, ne, nhtot)

    plt.figure()
    plt.plot(wave, dcop[0], label='dchi/dT')
    plt.plot(wave, dcop[1], label='dchi/dne')
    plt.plot(wave, dcop[2], label='dchi/dnhtot')
    plt.yscale('symlog', linthresh=1e-30)
    plt.legend()
