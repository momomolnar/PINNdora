from fractions import Fraction
import hashlib
import re

from adora_precision import REAL_DTYPE
import jax
import jax.numpy as jnp
import astropy.constants as const
import astropy.units as u
import numpy as np
from contop import lte_h_ion_fracs, continuum_opacity
from voigt import voigt_H_re as voigt, voigt_H as voigt_HF
import jax_dataclasses as jdc
import lightweaver as lw
from lightweaver.zeeman import lande_factor, fraction_range, zeeman_strength
from adora_data import FE_I_6301_6302_LINE_LIST

def _real(value):
    return jnp.asarray(value, dtype=REAL_DTYPE)


HC = _real(const.h.value * const.c.value)
NM_TO_M = _real(u.Unit('nm').to('m'))
M_TO_NM = _real(u.Unit('m').to('nm'))
E_RYD = _real(const.Ryd.to('J', equivalencies=u.spectral()).value)
Q_ELE = _real(u.eV.to(u.J))
EPS_0 = _real(const.eps0.value)
M_ELE = _real(const.m_e.value)
K_B = _real(const.k_B.value)
K_B_EV = _real(const.k_B.to('eV / K').value)
K_B_U = _real((const.k_B / const.u).value)
INV_C = _real(1.0 / const.c.value)
INV_FOURPI_C = _real(1.0 / (4.0 * np.pi * const.c.value))
HC_FOURPI_KJ_NM = _real(
    (const.h * const.c).to('kJ nm').value / (4.0 * np.pi)
)
SQRT_PI = _real(np.sqrt(np.pi))
SAHA_CONST = _real(
    ((2 * np.pi * const.m_e.value * const.k_B.value) / const.h.value**2) ** 1.5
)
TWOHC2_NM5 = _real(
    2.0 * (const.h.to('kJ s') * const.c**2 / (1e-9**4)).value
)
HC_KB_NM = _real((const.h * const.c / const.k_B).to('K nm').value)
DLAMBDA_B_CONST = _real(
    float(u.eV.to(u.J))
    / (4.0 * np.pi * const.m_e.value * const.c.to('nm/s').value)
)

# Interpolate logarithmic Kurucz partition functions over the full temperature
# range used by FAL-C.  The previous Irwin polynomial extrapolation becomes
# unphysical above its 16,000 K validity limit.
_PARTITION_TABLE = lw.KuruczPfTable()
_FE_PARTITION_TEMPERATURE = jnp.asarray(
    _PARTITION_TABLE.Tpf, dtype=REAL_DTYPE
)
_FE_LOG_PARTITION = jnp.asarray(
    _PARTITION_TABLE.pf[26 - 1][:3], dtype=REAL_DTYPE
)
_FE_IONIZATION_POTENTIAL = jnp.asarray(
    _PARTITION_TABLE.ionpot[26 - 1][:2], dtype=REAL_DTYPE
) / _real(u.eV.to(u.J))

@jdc.pytree_dataclass
class AtomicData:
    mass: jax.Array
    elem: jax.Array
    stage: jax.Array
    abund: jax.Array
    lambda0: jax.Array
    log_grad: jax.Array
    log_gs: jax.Array
    log_gw: jax.Array
    gi: jax.Array
    gj: jax.Array
    ei: jax.Array
    ej: jax.Array
    Aji: jax.Array
    line_weight: jax.Array
    zeeman_alphas: jax.Array
    zeeman_strengths: jax.Array
    zeeman_shifts: jax.Array
    # Packed nonzero Zeeman components used by the accelerated polarized
    # kernel.  These are optional so callers that construct ``AtomicData``
    # directly with the original padded fields remain compatible.
    zeeman_component_lines: jax.Array | None = None
    zeeman_component_alphas: jax.Array | None = None
    zeeman_component_strengths: jax.Array | None = None
    zeeman_component_shifts: jax.Array | None = None
    # Absolute wavelengths have only ~6e-5 nm spacing in fp32 near 630 nm.
    # Host-precentered offsets retain line-profile resolution on consumer GPUs.
    lambda0_offset: jax.Array | None = None
    wavelength_reference_nm: jdc.Static[float] = 0.0
    # Hash of the canonical fp64 host representation, independent of the
    # selected compute precision and derived acceleration fields.
    canonical_sha256: jdc.Static[str | None] = None
    # Preserve exact absolute line centres for archive metadata and host-side
    # wavelength-grid construction.  Reconstructing them from an fp32
    # ``lambda0`` array would throw away the precision that offset synthesis is
    # specifically designed to retain.
    canonical_lambda0_nm: jdc.Static[tuple[float, ...] | None] = None
    # Fingerprint of the initially cast runtime fields.  Consumers can use it
    # to distinguish an untouched parsed table (and return canonical_sha256)
    # from a dataclass replacement that changed physical atomic data.
    runtime_sha256: jdc.Static[str | None] = None


_CANONICAL_ATOMIC_FIELDS = (
    "mass",
    "elem",
    "stage",
    "abund",
    "lambda0",
    "log_grad",
    "log_gs",
    "log_gw",
    "gi",
    "gj",
    "ei",
    "ej",
    "Aji",
    "line_weight",
    "zeeman_alphas",
    "zeeman_strengths",
    "zeeman_shifts",
)


def _canonical_atomic_fingerprint(arrays) -> str:
    digest = hashlib.sha256()
    for name in _CANONICAL_ATOMIC_FIELDS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def wavelength_offsets(wavelengths, reference_nm: float):
    """Precenter absolute nm wavelengths on the host, then cast for compute.

    ``wavelengths`` should retain fp64 host precision until this function is
    called.  The returned offsets can safely be stored and evaluated in fp32.
    """

    host_wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if not np.all(np.isfinite(host_wavelengths)):
        raise ValueError("wavelengths must contain only finite values")
    host_offsets = host_wavelengths - float(reference_nm)
    return jnp.asarray(host_offsets, dtype=REAL_DTYPE)


# https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/database/atomllapi.py
def air_to_vac(wlair):
    """Convert wavelengths [AA] in air into those in vacuum.

    * See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Args:
        wlair:  wavelengthe in air [Angstrom]
        n:  Refractive Index in dry air at 1 atm pressure and 15ºC with 0.045% CO2 by volume (Birch and Downs, 1994, Metrologia, 31, 315)

    Returns:
        wlvac:  wavelength in vacuum [Angstrom]
    """
    s = 1e4 / wlair
    n = (
        1.0
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s * s)
        + 0.0001599740894897 / (38.92568793293 - s * s)
    )
    wlvac = wlair * n
    return wlvac

# Modified from https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/database/atomllapi.py
def read_kurucz(kuruczf):
    """Input Kurucz line list (http://kurucz.harvard.edu/linelists/)

    Args:
        kuruczf: file path

    Returns:
        AtomicData, containing some of:
        A:  Einstein coefficient in [s-1]
        wavelength:  vacuum transition wavelength in [nm]
        elower: lower excitation potential [eV]
        eupper: upper excitation potential [eV]
        glower: lower statistical weight
        gupper: upper statistical weight
        jlower: lower J (rotational quantum number, total angular momentum)
        jupper: upper J
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly)
        gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta: log of gamma of Stark damping (s-1 / m-3)
        gamvdW:  log of (van der Waals damping constant / neutral hydrogen number) (s-1 / m-3)
    """
    ccgs = 29979245800.0  # c in cgs
    ecgs = 4.80320450e-10  # [esu]=[dyn^0.5*cm] #elementary charge
    mecgs = 9.10938356e-28  # [g] !electron mass
    with open(kuruczf) as f:
        lines = [
            line.rstrip("\n").ljust(154)
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not lines:
        raise ValueError(f"Kurucz line list {kuruczf!s} contains no records")
    n_lines = len(lines)
    wlnmair = np.zeros(n_lines)
    loggf = np.zeros(n_lines)
    species = np.full(n_lines, "", dtype=object)
    elower = np.zeros(n_lines)
    jlower = np.zeros(n_lines)
    labellower = np.full(n_lines, "", dtype=object)
    eupper = np.zeros(n_lines)
    jupper = np.zeros(n_lines)
    labelupper = np.full(n_lines, "", dtype=object)
    gamRad = np.zeros(n_lines)
    gamSta = np.zeros(n_lines)
    gamvdW = np.zeros(n_lines)
    hyperfrac = np.zeros(n_lines)
    isofrac = np.zeros(n_lines)
    ielem, iion = np.zeros(len(lines), dtype=int), np.zeros(len(lines), dtype=int)

    for i, line in enumerate(lines):
        wlnmair[i] = float(line[0:11])
        loggf[i] = float(line[11:18])
        species[i] = str(line[18:24])
        ielem[i] = int(species[i].split(".")[0])
        iion[i] = int(species[i].split(".")[1]) + 1
        # Negative Kurucz energies mark predicted/extrapolated levels; their
        # magnitude is still the physical excitation energy.
        elower[i] = abs(float(line[24:36]))
        jlower[i] = float(line[36:41])
        labellower[i] = str(line[42:52])
        eupper[i] = abs(float(line[52:64]))
        jupper[i] = float(line[64:69])
        labelupper[i] = str(line[70:80])
        gamRad[i] = float(line[80:86])
        gamSta[i] = float(line[86:92])
        gamvdW[i] = float(line[92:98])
        hyperfrac[i] = float(line[109:115].strip() or 0.0)
        isofrac[i] = float(line[118:124].strip() or 0.0)

    if not np.all(np.isfinite(wlnmair)) or np.any(wlnmair <= 0.0):
        raise ValueError("Kurucz wavelengths must be positive and finite")

    if np.any(ielem != 26) or np.any(iion != 1):
        unsupported = sorted(set(zip(ielem.tolist(), iion.tolist())))
        raise NotImplementedError(
            "LTE synthesis currently supports only Fe I (element=26, "
            f"stage=1); found {unsupported}"
        )

    L_map = {
        symbol: value
        for value, symbol in enumerate("SPDFGHIKLMNOQRTU")
    }

    def parse_term(label):
        match = re.search(r"(\d+)([SPDFGHIKLMNOQRTU])\s*$", label)
        if match is None:
            return -1, -1
        return int(match.group(1)), L_map[match.group(2)]

    lower_terms = [parse_term(label) for label in labellower]
    upper_terms = [parse_term(label) for label in labelupper]
    multiplicity_lower = np.array([term[0] for term in lower_terms])
    multiplicity_upper = np.array([term[0] for term in upper_terms])
    L_lower = np.array([term[1] for term in lower_terms])
    L_upper = np.array([term[1] for term in upper_terms])

    not_flip = (eupper - elower) > 0
    elower_inverted = np.where(not_flip, elower, eupper)
    eupper_inverted = np.where(not_flip, eupper, elower)
    jlower_inverted = np.where(not_flip, jlower, jupper)
    jupper_inverted = np.where(not_flip, jupper, jlower)
    L_lower_inverted = np.where(not_flip, L_lower, L_upper)
    L_upper_inverted = np.where(not_flip, L_upper, L_lower)
    multiplicity_lower_inverted = np.where(not_flip, multiplicity_lower, multiplicity_upper)
    multiplicity_upper_inverted = np.where(not_flip, multiplicity_upper, multiplicity_lower)

    elower = elower_inverted
    eupper = eupper_inverted
    J_lower = jlower_inverted
    J_upper = jupper_inverted
    L_lower = L_lower_inverted
    L_upper = L_upper_inverted
    multiplicity_lower = multiplicity_lower_inverted
    multiplicity_upper = multiplicity_upper_inverted

    zeeman_alpha_list = []
    zeeman_strength_list = []
    zeeman_shift_list = []

    def unpolarized_components():
        return (
            np.array([-1, 0, 1], dtype=np.int32),
            np.ones(3),
            np.zeros(3),
        )

    for line_index in range(len(lines)):
        Jl = Fraction.from_float(J_lower[line_index]).limit_denominator(2)
        Ju = Fraction.from_float(J_upper[line_index]).limit_denominator(2)
        Ll = int(L_lower[line_index])
        Lu = int(L_upper[line_index])
        lower_mult = int(multiplicity_lower[line_index])
        upper_mult = int(multiplicity_upper[line_index])

        determinate = lower_mult > 0 and upper_mult > 0 and Ll >= 0 and Lu >= 0
        if determinate:
            Sl = Fraction(lower_mult - 1, 2)
            Su = Fraction(upper_mult - 1, 2)
            determinate = (
                abs(Ll - Sl) <= Jl <= Ll + Sl
                and abs(Lu - Su) <= Ju <= Lu + Su
                and abs(Ju - Jl) <= 1
                and not (Ju == 0 and Jl == 0)
            )

        if not determinate:
            alpha, strength, shift = unpolarized_components()
        else:
            # Adapted from Lightweaver's LS-coupling implementation.
            gLl = lande_factor(Jl, Ll, Sl)
            gLu = lande_factor(Ju, Lu, Su)
            alpha_values = []
            strength_values = []
            shift_values = []
            norm = np.zeros(3)

            for ml in fraction_range(-Jl, Jl + 1):
                for mu in fraction_range(-Ju, Ju + 1):
                    if abs(ml - mu) <= 1.0:
                        component_alpha = int(ml - mu)
                        component_strength = zeeman_strength(Ju, mu, Jl, ml)
                        alpha_values.append(component_alpha)
                        shift_values.append(gLl * ml - gLu * mu)
                        strength_values.append(component_strength)
                        norm[component_alpha + 1] += component_strength

            alpha = np.array(alpha_values, dtype=np.int32)
            strength = np.array(strength_values)
            shift = np.array(shift_values)
            if np.any(norm <= 0.0):
                alpha, strength, shift = unpolarized_components()
            else:
                strength /= norm[alpha + 1]

        zeeman_alpha_list.append(alpha)
        zeeman_strength_list.append(strength)
        zeeman_shift_list.append(shift)
    max_zeeman_components = max([z.shape[0] for z in zeeman_alpha_list])

    zeeman_alphas = np.zeros((len(lines), max_zeeman_components), dtype=np.int32)
    zeeman_strengths = np.zeros((len(lines), max_zeeman_components))
    zeeman_shifts = np.zeros((len(lines), max_zeeman_components))
    for i, (alpha, strength, shift) in enumerate(zip(zeeman_alpha_list, zeeman_strength_list, zeeman_shift_list)):
        length = zeeman_alpha_list[i].shape[0]
        zeeman_alphas[i, :length] = alpha
        zeeman_strengths[i, :length] = strength
        zeeman_shifts[i, :length] = shift

    # Keep the original rectangular tables as public metadata, but synthesize
    # from a packed representation.  For the bundled Fe I pair this avoids 11
    # of 26 Voigt evaluations: ten padding entries and one exactly zero-strength
    # physical component.  A zero constant contributes neither a value nor an
    # atmospheric derivative, so removing it is exact for the inversion.
    zeeman_component_lines, zeeman_component_slots = np.nonzero(
        zeeman_strengths != 0.0
    )
    zeeman_component_alphas = zeeman_alphas[
        zeeman_component_lines, zeeman_component_slots
    ]
    zeeman_component_strengths = zeeman_strengths[
        zeeman_component_lines, zeeman_component_slots
    ]
    zeeman_component_shifts = zeeman_shifts[
        zeeman_component_lines, zeeman_component_slots
    ]

    wlaa = np.where(wlnmair < 200, wlnmair * 10, air_to_vac(wlnmair * 10))
    wl_vac = np.where(wlnmair < 200, wlnmair, air_to_vac(wlnmair * 10) * 0.1)
    nu_lines = 1e8 / wlaa  # [cm-1]<-[AA]
    elower = (elower << u.Unit('cm-1')).to('eV', equivalencies=u.spectral()).value
    # Kurucz level energies are rounded independently of the tabulated line
    # wavelength.  As in RH, retain the lower excitation energy but derive the
    # upper energy from the wavelength used by the profile.  This keeps the LTE
    # upper-level population, Aji, and line photon energy internally consistent.
    eupper = (
        elower
        + const.h.value
        * const.c.value
        / (wl_vac * float(u.Unit('nm').to('m')))
        / float(u.eV.to(u.J))
    )
    glower = J_lower * 2 + 1
    gupper = J_upper * 2 + 1
    A = (
        10**loggf
        / gupper
        * (ccgs * nu_lines) ** 2
        * (8 * np.pi**2 * ecgs**2)
        / (mecgs * ccgs**3)
    )
    # A zero Kurucz damping field is a sentinel for an absent process, not
    # log10(gamma)=0.  Collisional coefficients are tabulated per cm^-3;
    # subtract six only for actual values to convert them to per m^-3.
    has_collisional_damping = (gamSta != 0.0) | (gamvdW != 0.0)
    gamRad = np.where(
        gamRad != 0.0,
        gamRad,
        np.where(has_collisional_damping, np.log10(A), -np.inf),
    )
    gamSta = np.where(gamSta != 0.0, gamSta - 6.0, -np.inf)
    gamvdW = np.where(gamvdW != 0.0, gamvdW - 6.0, -np.inf)
    line_weight = 10.0 ** (hyperfrac + isofrac)

    canonical = {
        "mass": np.asarray(
            [lw.PeriodicTable[lw.Element(Z=z)].mass for z in ielem],
            dtype=np.float64,
        ),
        "elem": np.asarray(ielem, dtype=np.int64),
        "stage": np.asarray(iion, dtype=np.int64),
        "abund": np.asarray(
            [lw.DefaultAtomicAbundance[lw.Element(Z=z)] for z in ielem],
            dtype=np.float64,
        ),
        "lambda0": np.asarray(wl_vac, dtype=np.float64),
        "log_grad": np.asarray(gamRad, dtype=np.float64),
        "log_gs": np.asarray(gamSta, dtype=np.float64),
        "log_gw": np.asarray(gamvdW, dtype=np.float64),
        "gi": np.asarray(glower, dtype=np.float64),
        "gj": np.asarray(gupper, dtype=np.float64),
        "ei": np.asarray(elower, dtype=np.float64),
        "ej": np.asarray(eupper, dtype=np.float64),
        "Aji": np.asarray(A, dtype=np.float64),
        "line_weight": np.asarray(line_weight, dtype=np.float64),
        "zeeman_alphas": np.asarray(zeeman_alphas, dtype=np.int32),
        "zeeman_strengths": np.asarray(zeeman_strengths, dtype=np.float64),
        "zeeman_shifts": np.asarray(zeeman_shifts, dtype=np.float64),
    }
    wavelength_reference_nm = float(
        0.5 * (canonical["lambda0"].min() + canonical["lambda0"].max())
    )
    runtime_integer_dtype = jnp.int64 if REAL_DTYPE == jnp.float64 else jnp.int32
    runtime_numpy_float = np.float64 if REAL_DTYPE == jnp.float64 else np.float32
    runtime_numpy_integer = np.int64 if REAL_DTYPE == jnp.float64 else np.int32
    runtime_for_hash = {
        name: np.asarray(
            canonical[name],
            dtype=(
                np.int32
                if name == "zeeman_alphas"
                else runtime_numpy_integer
                if name in ("elem", "stage")
                else runtime_numpy_float
            ),
        )
        for name in _CANONICAL_ATOMIC_FIELDS
    }

    return AtomicData(
        mass=jnp.asarray(canonical["mass"], dtype=REAL_DTYPE),
        elem=jnp.asarray(canonical["elem"], dtype=runtime_integer_dtype),
        stage=jnp.asarray(canonical["stage"], dtype=runtime_integer_dtype),
        abund=jnp.asarray(canonical["abund"], dtype=REAL_DTYPE),
        lambda0=jnp.asarray(canonical["lambda0"], dtype=REAL_DTYPE),
        log_grad=jnp.asarray(canonical["log_grad"], dtype=REAL_DTYPE),
        log_gs=jnp.asarray(canonical["log_gs"], dtype=REAL_DTYPE),
        log_gw=jnp.asarray(canonical["log_gw"], dtype=REAL_DTYPE),
        gi=jnp.asarray(canonical["gi"], dtype=REAL_DTYPE),
        gj=jnp.asarray(canonical["gj"], dtype=REAL_DTYPE),
        ei=jnp.asarray(canonical["ei"], dtype=REAL_DTYPE),
        ej=jnp.asarray(canonical["ej"], dtype=REAL_DTYPE),
        Aji=jnp.asarray(canonical["Aji"], dtype=REAL_DTYPE),
        line_weight=jnp.asarray(canonical["line_weight"], dtype=REAL_DTYPE),
        zeeman_alphas=jnp.asarray(canonical["zeeman_alphas"], dtype=jnp.int32),
        zeeman_strengths=jnp.asarray(
            canonical["zeeman_strengths"], dtype=REAL_DTYPE
        ),
        zeeman_shifts=jnp.asarray(canonical["zeeman_shifts"], dtype=REAL_DTYPE),
        zeeman_component_lines=jnp.asarray(
            zeeman_component_lines, dtype=jnp.int32
        ),
        zeeman_component_alphas=jnp.asarray(
            zeeman_component_alphas, dtype=jnp.int32
        ),
        zeeman_component_strengths=jnp.asarray(
            zeeman_component_strengths, dtype=REAL_DTYPE
        ),
        zeeman_component_shifts=jnp.asarray(
            zeeman_component_shifts, dtype=REAL_DTYPE
        ),
        lambda0_offset=jnp.asarray(
            canonical["lambda0"] - wavelength_reference_nm,
            dtype=REAL_DTYPE,
        ),
        wavelength_reference_nm=wavelength_reference_nm,
        canonical_sha256=_canonical_atomic_fingerprint(canonical),
        canonical_lambda0_nm=tuple(float(value) for value in canonical["lambda0"]),
        runtime_sha256=_canonical_atomic_fingerprint(runtime_for_hash),
    )

def thermal_vel(mass, temperature):
    r"""
    /** Compute mean thermal velocity
    * \param mass [u]
    * \param temperature [K]
    * \return mean thermal velocity [m/s]
    */
    """
    return jnp.sqrt(2.0 * temperature / mass * K_B_U)

def doppler_width(lambda0, mass, temperature, vturb):
    r"""
    /** Compute doppler width
    * \param lambda0 wavelength [nm]
    * \param mass [u]
    * \param temperature [K]
    * \param vturb microturbulent velocity [m / s]
    * \return Doppler width [nm]
    */
    """
    return lambda0 * jnp.sqrt(2.0 * K_B_U * temperature / mass + vturb**2) * INV_C

def damping_from_gamma(gamma, lambda0, dop_width):
    r"""
    /** Compute damping coefficient for Voigt profile from gamma
    * \param gamma [rad / s]
    * \param lambda0 line-centre wavelength [nm]
    * \param dop_width [nm]
    * \return gamma / (4 pi dnu_D) = gamma / (4 pi dlambda_D) * lambda0**2
    */
    """
    # NOTE(cmo): Extra 1e-9 to convert c to nm / s (all lengths here in nm)
    return INV_FOURPI_C * gamma * 1e-9 * lambda0**2 / dop_width

def gamma_from_broadening(log_grad, log_gs, log_gw, temperature, ne, nhi):
    """
    Compute damping gamma from the line parameters and atmospheric parameters
    """
    grad = 10**log_grad
    gstark = 10**log_gs * ne
    gvdw = 10**log_gw * nhi

    gamma = grad + gstark + gvdw
    return gamma

def planck(wave, temperature):
    """
    Compute the Planck function (in kW/(nm m2 sr))

    wave : float
        Wavelength [nm]
    temperature : float
        Temperature [K]
    """
    exponent = HC_KB_NM / (wave * temperature)
    exp_negative = jnp.exp(-exponent)
    inverse_expm1 = exp_negative / (-jnp.expm1(-exponent))
    return TWOHC2_NM5 / wave**5 * inverse_expm1

def emis_opac_line(
        mass,
        abund,
        lambda0,
        log_grad,
        log_gs,
        log_gw,
        gj,
        ej,
        Aji,
        line_weight,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        wavelength_delta=None,
):
    """
    Compute emissivity/opacity for a single LTE line
    """
    nhi, nhii = lte_h_ion_fracs(temperature, ne, nhtot)
    dop_width = doppler_width(lambda0, mass, temperature, vturb)
    gamma = gamma_from_broadening(log_grad, log_gs, log_gw, temperature, ne, nhi)
    adamp = damping_from_gamma(gamma, lambda0, dop_width)
    if wavelength_delta is None:
        wavelength_delta = wave - lambda0
    v = (wavelength_delta + (vel * lambda0) * INV_C) / dop_width
    p = voigt(adamp, v) / (SQRT_PI * dop_width)

    hnu_4pi = HC_FOURPI_KJ_NM / wave
    Uji = line_weight * hnu_4pi * Aji * p
    Sfn = planck(wave, temperature)
    nj = fei_pop_i(abund, temperature, ne, nhtot, gj, ej)
    eta = nj * Uji
    chi = eta / Sfn
    return eta, chi

def polarised_line_component(
        alpha,
        strength,
        shift,
        adamp,
        v_scalar,
        v_b,
):
    # phi_sb, phi_pi, phi_sr = 0.0, 0.0, 0.0
    # psi_sb, psi_pi, psi_sr = 0.0, 0.0, 0.0
    components = jnp.zeros((2, 3))

    vh, vf = voigt_HF(adamp, v_scalar - shift * v_b)
    components = components.at[0, alpha+1].set(strength * vh)
    components = components.at[1, alpha+1].set(strength * vf)
    return components


def _polarised_line_state(
        mass,
        abund,
        lambda0,
        log_grad,
        log_gs,
        log_gw,
        gj,
        ej,
        Aji,
        line_weight,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        source_function,
        wavelength_delta,
):
    """Compute the per-line state shared by every Zeeman component."""
    mag_width = DLAMBDA_B_CONST * lambda0**2
    nhi, _ = lte_h_ion_fracs(temperature, ne, nhtot)
    dop_width = doppler_width(lambda0, mass, temperature, vturb)
    gamma = gamma_from_broadening(
        log_grad, log_gs, log_gw, temperature, ne, nhi
    )
    adamp = damping_from_gamma(gamma, lambda0, dop_width)
    v = (wavelength_delta + (vel * lambda0) * INV_C) / dop_width
    v_b = mag_width * b / dop_width
    voigt_norm = 1.0 / (SQRT_PI * dop_width)

    nj = fei_pop_i(abund, temperature, ne, nhtot, gj, ej)
    hnu_4pi = HC_FOURPI_KJ_NM / wave
    eta_no_prof = line_weight * nj * hnu_4pi * Aji
    chi_no_prof = eta_no_prof / source_function
    return adamp, v, v_b, voigt_norm, eta_no_prof, chi_no_prof


def _polarised_profiles(
        components,
        voigt_norm,
        cos_gamma,
        sin_2chi,
        cos_2chi,
):
    """Convert grouped absorption/dispersion components to Stokes profiles."""
    sin2_gamma = 1.0 - cos_gamma**2

    phi_sigma = components[..., 0, 0] + components[..., 0, 2]
    phi_delta = 0.5 * components[..., 0, 1] - 0.25 * phi_sigma
    phi = (phi_delta * sin2_gamma + 0.5 * phi_sigma) * voigt_norm

    phi_q = phi_delta * sin2_gamma * cos_2chi * voigt_norm
    phi_u = phi_delta * sin2_gamma * sin_2chi * voigt_norm
    phi_v = (
        0.5
        * (components[..., 0, 2] - components[..., 0, 0])
        * cos_gamma
        * voigt_norm
    )

    psi_sigma = components[..., 1, 0] + components[..., 1, 2]
    psi_delta = 0.5 * components[..., 1, 1] - 0.25 * psi_sigma
    psi_q = psi_delta * sin2_gamma * cos_2chi * voigt_norm
    psi_u = psi_delta * sin2_gamma * sin_2chi * voigt_norm
    psi_v = (
        0.5
        * (components[..., 1, 2] - components[..., 1, 0])
        * cos_gamma
        * voigt_norm
    )
    return jnp.stack(
        (phi, phi_q, phi_u, phi_v, psi_q, psi_u, psi_v), axis=-1
    )


def emis_opac_polarised_line(
        mass,
        abund,
        lambda0,
        log_grad,
        log_gs,
        log_gw,
        gj,
        ej,
        Aji,
        line_weight,
        zeeman_alpha,
        zeeman_strength,
        zeeman_shift,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        cos_gamma,
        sin_2chi,
        cos_2chi,
        wavelength_delta=None,
):
    if wavelength_delta is None:
        wavelength_delta = wave - lambda0
    source_function = planck(wave, temperature)
    (
        adamp,
        v,
        v_b,
        voigt_norm,
        eta_no_prof,
        chi_no_prof,
    ) = _polarised_line_state(
        mass,
        abund,
        lambda0,
        log_grad,
        log_gs,
        log_gw,
        gj,
        ej,
        Aji,
        line_weight,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        source_function,
        wavelength_delta,
    )

    components = jax.vmap(
        polarised_line_component,
        in_axes=[0, 0, 0, None, None, None]
    )(
        zeeman_alpha,
        zeeman_strength,
        zeeman_shift,
        adamp,
        v,
        v_b
    ).sum(axis=0)
    profiles = _polarised_profiles(
        components, voigt_norm, cos_gamma, sin_2chi, cos_2chi
    )
    return eta_no_prof * profiles[:4], chi_no_prof * profiles


# Backward-compatible alias for the original misspelling.
polarised_line_compoment = polarised_line_component


def _spectral_coordinates(adata: AtomicData, wave, *, wave_is_offset):
    wave = jnp.asarray(wave)
    if not wave_is_offset:
        return wave, wave - adata.lambda0
    if adata.lambda0_offset is None:
        raise ValueError(
            "offset synthesis requires AtomicData.lambda0_offset; "
            "construct atomic data with read_kurucz"
        )
    reference = jnp.asarray(adata.wavelength_reference_nm, dtype=wave.dtype)
    return reference + wave, wave - adata.lambda0_offset


def _emis_opac_impl(
        adata: AtomicData,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        *,
        wave_is_offset,
):
    absolute_wave, wavelength_delta = _spectral_coordinates(
        adata, wave, wave_is_offset=wave_is_offset
    )
    chi_cont = continuum_opacity(absolute_wave, temperature, ne, nhtot)
    B_planck = planck(absolute_wave, temperature)
    eta_cont = chi_cont * B_planck

    axis_spec = (*[0] * 10, *[None] * 6, 0)
    line_eta, line_chi = jax.vmap(emis_opac_line, in_axes=axis_spec)(
        adata.mass,
        adata.abund,
        adata.lambda0,
        adata.log_grad,
        adata.log_gs,
        adata.log_gw,
        adata.gj,
        adata.ej,
        adata.Aji,
        adata.line_weight,
        absolute_wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        wavelength_delta,
    )

    eta = eta_cont + line_eta.sum()
    chi = chi_cont + line_chi.sum()
    return eta, chi


def emis_opac(adata: AtomicData, wave, temperature, ne, nhtot, vel, vturb):
    """
    Compute total emissivity/opacity for a atmospheric point

    adata : AtomicData
        The dataclass containing the atomic data
    wave : float
        The wavelength at which to compute [nm]
    temperature : float
        Temperature [K]
    ne : float
        Electron density [m-3]
    nhtot : float
        Total H density [m-3]
    vel : float
        LOS velocity [m/s]
    vturb : float
        microturbulent velocity [m/s]

    Returns
    -------
    Total emissivity, total opacity, from continuum and all lines.
    """
    return _emis_opac_impl(
        adata,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        wave_is_offset=False,
    )


def emis_opac_offset(
        adata: AtomicData,
        wave_offset,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
):
    """Evaluate scalar coefficients from a host-precentered wavelength."""

    return _emis_opac_impl(
        adata,
        wave_offset,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        wave_is_offset=True,
    )

def _emis_opac_polarised_impl(
        adata: AtomicData,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        gamma_b,
        chi_b,
        *,
        wave_is_offset,
    ):
    """
    Compute total emissivity/opacity for a atmospheric point

    adata : AtomicData
        The dataclass containing the atomic data
    wave : float
        The wavelength at which to compute [nm]
    temperature : float
        Temperature [K]
    ne : float
        Electron density [m-3]
    nhtot : float
        Total H density [m-3]
    vel : float
        LOS velocity [m/s]
    vturb : float
        microturbulent velocity [m/s]
    b : float
        magnetic field [T]
    gamma_b : float
        inclination of magnetic field [radians]
    chi_b : float
        azimuth of magnetic field [radians]

    Returns
    -------
    Total emissivity, total opacity, from continuum and all lines. (Stokes form)
    [eps_I, eps_Q, eps_U, eps_V], [eta_I, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V]
    """
    absolute_wave, wavelength_delta = _spectral_coordinates(
        adata, wave, wave_is_offset=wave_is_offset
    )
    chi_cont = continuum_opacity(absolute_wave, temperature, ne, nhtot)
    B_planck = planck(absolute_wave, temperature)
    eta_cont = chi_cont * B_planck

    # NOTE(cmo): This is only correct in the muz=1 case
    cos_gamma = jnp.cos(gamma_b)
    cos_2chi = jnp.cos(2.0 * chi_b)
    sin_2chi = jnp.sin(2.0 * chi_b)

    # ``AtomicData`` instances created before packed Zeeman metadata was added
    # retain the original rectangular execution path.  Parsed line lists use
    # the packed path below, which evaluates only physically nonzero components.
    if adata.zeeman_component_lines is None:
        axis_spec = (*[0] * 13, *[None] * 10, 0)
        line_eta, line_chi = jax.vmap(
            emis_opac_polarised_line, in_axes=axis_spec
        )(
            adata.mass,
            adata.abund,
            adata.lambda0,
            adata.log_grad,
            adata.log_gs,
            adata.log_gw,
            adata.gj,
            adata.ej,
            adata.Aji,
            adata.line_weight,
            adata.zeeman_alphas,
            adata.zeeman_strengths,
            adata.zeeman_shifts,
            absolute_wave,
            temperature,
            ne,
            nhtot,
            vel,
            vturb,
            b,
            cos_gamma,
            sin_2chi,
            cos_2chi,
            wavelength_delta,
        )
        eta = line_eta.sum(axis=0)
        chi = line_chi.sum(axis=0)
        eta = eta.at[0].set(eta[0] + eta_cont)
        chi = chi.at[0].set(chi[0] + chi_cont)
        return eta, chi

    line_state_axes = (*[0] * 10, *[None] * 8, 0)
    (
        adamp,
        v,
        v_b,
        voigt_norm,
        eta_no_prof,
        chi_no_prof,
    ) = jax.vmap(_polarised_line_state, in_axes=line_state_axes)(
        adata.mass,
        adata.abund,
        adata.lambda0,
        adata.log_grad,
        adata.log_gs,
        adata.log_gw,
        adata.gj,
        adata.ej,
        adata.Aji,
        adata.line_weight,
        absolute_wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        B_planck,
        wavelength_delta,
    )

    component_lines = adata.zeeman_component_lines
    components = jax.vmap(
        polarised_line_component,
        in_axes=(0, 0, 0, 0, 0, 0),
    )(
        adata.zeeman_component_alphas,
        adata.zeeman_component_strengths,
        adata.zeeman_component_shifts,
        adamp[component_lines],
        v[component_lines],
        v_b[component_lines],
    )
    component_to_line = jax.nn.one_hot(
        component_lines,
        adata.mass.shape[0],
        dtype=components.dtype,
    )
    components_by_line = jnp.einsum(
        "cl,cvp->lvp", component_to_line, components
    )
    profiles = _polarised_profiles(
        components_by_line,
        voigt_norm,
        cos_gamma,
        sin_2chi,
        cos_2chi,
    )
    eta = (eta_no_prof[:, None] * profiles[:, :4]).sum(axis=0)
    chi = (chi_no_prof[:, None] * profiles).sum(axis=0)

    eta = eta.at[0].set(eta[0] + eta_cont)
    chi = chi.at[0].set(chi[0] + chi_cont)
    return eta, chi


def emis_opac_polarised(
        adata: AtomicData,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        gamma_b,
        chi_b,
    ):
    """Evaluate polarized coefficients at an absolute wavelength in nm."""

    return _emis_opac_polarised_impl(
        adata,
        wave,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        gamma_b,
        chi_b,
        wave_is_offset=False,
    )


def emis_opac_polarised_offset(
        adata: AtomicData,
        wave_offset,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        gamma_b,
        chi_b,
    ):
    """Evaluate polarized coefficients from a host-precentered wavelength.

    Use :func:`wavelength_offsets` to construct ``wave_offset``.  This avoids
    catastrophic loss of line-profile resolution when bulk physics uses fp32.
    """

    return _emis_opac_polarised_impl(
        adata,
        wave_offset,
        temperature,
        ne,
        nhtot,
        vel,
        vturb,
        b,
        gamma_b,
        chi_b,
        wave_is_offset=True,
    )


def _log_fe_partition(stage_index, temperature):
    """Interpolate log partition functions on Lightweaver's Kurucz grid."""
    return jnp.interp(
        temperature,
        _FE_PARTITION_TEMPERATURE,
        _FE_LOG_PARTITION[stage_index],
    )


def Q_FeI(T):
    return jnp.exp(_log_fe_partition(0, T))


def Q_FeII(T):
    return jnp.exp(_log_fe_partition(1, T))


def Q_FeIII(T):
    return jnp.exp(_log_fe_partition(2, T))

def fe_pops(abund, temperature, ne, nhtot):
    """
    Compute Fe I, II and III populations using Kurucz partition functions.

    ionisation potentials from https://srd.nist.gov/jpcrdreprint/1.555659.pdf

    abund : float
        The decimal abundance of Fe relative to H
    temperature : float
        The temperature [K]
    ne : float
        The electron density [m-3]
    nhtot : float
        Total H density [m-3]
    """
    kBT = K_B_EV * temperature
    log_saha = (
        jnp.log(2.0 * SAHA_CONST)
        + 1.5 * jnp.log(temperature)
        - jnp.log(ne)
    )
    log_n1_n0 = (
        _log_fe_partition(1, temperature)
        - _log_fe_partition(0, temperature)
        + log_saha
        - _FE_IONIZATION_POTENTIAL[0] / kBT
    )
    log_n2_n1 = (
        _log_fe_partition(2, temperature)
        - _log_fe_partition(1, temperature)
        + log_saha
        - _FE_IONIZATION_POTENTIAL[1] / kBT
    )

    # n_II/n_I=r01 and n_III/n_II=r12, so the three relative
    # populations are [1, r01, r01*r12].  Softmax evaluates the
    # normalization without overflow or loss of abundance conservation.
    log_weights = jnp.array([0.0, log_n1_n0, log_n1_n0 + log_n2_n1])
    fractions = jax.nn.softmax(log_weights)
    populations = abund * nhtot * fractions
    return populations[0], populations[1], populations[2]

def fei_pop_i(abund, temperature, ne, nhtot, gi, ei):
    """
    Compute the population of a level i of Fe I with statistical weight gi and energy ei

    abund : float
        The decimal abundance of Fe relative to H
    temperature : float
        The temperature [K]
    ne : float
        The electron density [m-3]
    nhtot : float
        Total H density [m-3]
    gi : float
        Statistical weight g
    ei : float
        Energy level in [eV]
    """
    nfei, nfeii, nfeiii = fe_pops(abund, temperature, ne, nhtot)
    kBT = K_B_EV * temperature
    ni = nfei * gi * jnp.exp(-_log_fe_partition(0, temperature) - ei / kBT)
    return ni



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

    # plt.figure()
    kd = read_kurucz(FE_I_6301_6302_LINE_LIST)

    # wave = np.linspace(50, 1000.0, 100)
    # b = planck(wave, 5000)
    # import lightweaver as lw
    # b_lw = (lw.planck(5000, wave) << u.Unit('W/(m2 Hz sr)')).to('kW/(m2 nm sr)', equivalencies=u.spectral_density(wav=wave << u.nm)).value
    # plt.figure()
    # plt.plot(wave, b)
    # plt.plot(wave, b_lw, '--')

    eta_s, chi_s = emis_opac(kd, 630.324, 5000, 1e22, 1e25, 0.0, 2e3)

    eta, chi = emis_opac_polarised(kd, 630.31, 5000, 1e22, 1e25, 0.0, 2e3, 0.1, 0.0, 0.0)
