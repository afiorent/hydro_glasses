"""Hydrodynamic thermal transport analysis for glasses.

Functions for extracting disorder linewidths from dynamic structure
factors, fitting dispersion relations, and computing hydrodynamic
contributions to thermal conductivity.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fitting lineshapes
# ---------------------------------------------------------------------------


def _lorentzian_fit(w: NDArray, w0: float, gamma: float, norm: float) -> NDArray:
    return norm * (gamma / (gamma**2 + (w - w0) ** 2))


def _gaussian_fit(w: NDArray, w0: float, sigma: float, norm: float) -> NDArray:
    return norm * np.exp(-((w - w0) ** 2) / 2 / sigma**2)


def _dho_fit(w: NDArray, w0: float, tau: float, norm: float) -> NDArray:
    return norm * (w * 2 * tau / ((w * tau) ** 2 + (w**2 - w0**2) ** 2))


_FIT_FUNCS = {
    "lorentzian": _lorentzian_fit,
    "gaussian": _gaussian_fit,
    "dho": _dho_fit,
}


# ---------------------------------------------------------------------------
# Debye DOS helpers
# ---------------------------------------------------------------------------


def dos_omega_T(omega: NDArray[np.floating], cT: float) -> NDArray[np.floating]:
    r"""Transverse DOS in the Debye model.

    .. math:: g_T(\omega) = \omega^2 / (\pi^2 c_T^3)

    Parameters
    ----------
    omega : array
        Angular frequencies.
    cT : float
        Transverse speed of sound.
    """
    return omega**2 / np.pi**2 / cT**3


def dos_omega_L(omega: NDArray[np.floating], cL: float) -> NDArray[np.floating]:
    r"""Longitudinal DOS in the Debye model.

    .. math:: g_L(\omega) = \omega^2 / (2 \pi^2 c_L^3)

    Parameters
    ----------
    omega : array
        Angular frequencies.
    cL : float
        Longitudinal speed of sound.
    """
    return omega**2 / 2 / np.pi**2 / cL**3


# ---------------------------------------------------------------------------
# Heat capacity
# ---------------------------------------------------------------------------


def calculate_heat_capacity(omega: NDArray[np.floating], temp: float) -> NDArray[np.floating]:
    r"""Heat capacity per mode.

    Uses the quantum expression from the kaldo paper.  The result is in
    J/K.

    Parameters
    ----------
    omega : array
        Angular frequencies (rad/ps).
    temp : float
        Temperature in Kelvin.

    Returns
    -------
    NDArray
        Heat capacity per mode (J/K).
    """
    kB = 1.380649e-23  # J/K
    h = 6.62607015e-22  # J*ps
    result = np.ones_like(omega) * kB
    nonzero = ~np.isclose(omega, 0)
    freq = omega[nonzero] / (2 * np.pi)  # 1/ps
    result[nonzero] = (h * freq) ** 2 / 4 / kB / temp**2 / np.sinh(h * freq / 2 / kB / temp) ** 2
    return result


# ---------------------------------------------------------------------------
# Disorder linewidths
# ---------------------------------------------------------------------------


def compute_Gamma(spectrum: dict) -> dict[str, NDArray[np.floating]]:
    r"""Compute linewidth :math:`\Gamma` from the integral width of S(Q, omega).

    Parameters
    ----------
    spectrum : dict
        Must contain ``'omega'`` and ``'S'`` (with branches ``'L'``, ``'T'``).

    Returns
    -------
    dict
        ``{branch: Gamma_array}`` for each branch.
    """
    w = spectrum["omega"]
    S = spectrum["S"]

    Gamma: dict[str, NDArray] = {}
    for branch in S:
        N = trapezoid(S[branch].mean(axis=2), w, axis=1)
        sq_int = trapezoid(S[branch].mean(axis=2) ** 2, w, axis=1)
        Gamma[branch] = N**2 / (2 * np.pi * sq_int)
    return Gamma


def compute_disorder_widths(
    spectrum: dict,
    *,
    qmax: float = 1.0,
    qmax_c: float = 0.4,
    fit_func: str = "DHO",
) -> dict:
    """Extract disorder linewidths by fitting S(Q, omega) peaks.

    Parameters
    ----------
    spectrum : dict
        Must contain ``'q'``, ``'omega'``, and ``'S'``.
    qmax : float
        Maximum wavevector for fitting.
    qmax_c : float
        Maximum wavevector for sound-velocity fit.
    fit_func : ``'lorentzian'`` | ``'gaussian'`` | ``'DHO'``
        Lineshape model.

    Returns
    -------
    dict
        Keys: ``'q'``, ``'Gamma'``, ``'Gamma_std'``, ``'omega'``,
        ``'c_sound'``.

    Raises
    ------
    ValueError
        If inputs are invalid or empty.
    """
    func_key = fit_func.lower()
    if func_key not in _FIT_FUNCS:
        raise ValueError(f"Fitting function should be one of {list(_FIT_FUNCS)}")
    func = _FIT_FUNCS[func_key]

    q = spectrum["q"]
    w = spectrum["omega"]
    S = spectrum["S"]

    if not isinstance(q, np.ndarray) or q.ndim != 1:
        raise ValueError('spectrum["q"] should be a 1D numpy array')
    if len(q) == 0:
        raise ValueError('spectrum["q"] cannot be empty')
    if not np.all(q >= 0):
        raise ValueError('spectrum["q"] should contain non-negative wavevector norms')

    q_for_fit: dict[str, list] = {}
    freq: dict[str, list] = {}
    width: dict[str, list] = {}
    width_std: dict[str, list] = {}
    c_sound: dict[str, NDArray] = {}

    for branch in S:
        q_for_fit[branch] = []
        width[branch] = []
        width_std[branch] = []
        freq[branch] = []

        for i in range(len(q[q < qmax])):
            try:
                if S[branch][i].ndim == 2:
                    popt, pcov = curve_fit(func, w, S[branch][i].mean(axis=1))
                else:
                    popt, pcov = curve_fit(func, w, S[branch][i])
                q_for_fit[branch].append(q[i])
                freq[branch].append([popt[0], pcov[0, 0]])
                width[branch].append(popt[1])
                width_std[branch].append(np.sqrt(pcov[1, 1]))
            except RuntimeError:
                log.warning("Fit failure for Q index %d in branch %s", i, branch)
                continue

        q_for_fit[branch] = np.array(q_for_fit[branch])
        if func_key == "dho":
            width[branch] = np.array(width[branch]) / 2
            width_std[branch] = np.array(width_std[branch]) / 2
        else:
            width[branch] = np.array(width[branch])
            width_std[branch] = np.array(width_std[branch])
        freq[branch] = np.array(freq[branch])

        mask = q_for_fit[branch] <= qmax_c
        popt, pcov = curve_fit(
            lambda k, c: c * k,
            q_for_fit[branch][mask],
            freq[branch][mask, 0],
            sigma=freq[branch][mask, 1],
        )
        c_sound[branch] = np.array([popt[0], np.sqrt(pcov[0, 0])])

    return {
        "q": q_for_fit,
        "Gamma": width,
        "Gamma_std": width_std,
        "omega": freq,
        "c_sound": c_sound,
    }


def fit_disorder_widths(
    fit_dict: dict,
    *,
    qmax: float = 0.5,
    power: int = 2,
    eta: float = 0,
) -> dict[str, NDArray]:
    r"""Fit linewidths as :math:`\Gamma(q) = A q^n`.

    Parameters
    ----------
    fit_dict : dict
        Output of :func:`compute_disorder_widths`.
    qmax : float
        Maximum wavevector for fitting.
    power : int
        Exponent *n*.
    eta : float
        Constant offset to subtract.

    Returns
    -------
    dict
        ``{branch: [A, sigma_A]}`` for each branch.
    """
    q = fit_dict["q"]
    G = fit_dict["Gamma"]
    coeffs: dict[str, NDArray] = {}

    for branch in q:
        mask = q[branch] <= qmax
        popt, pcov = curve_fit(
            lambda k, a: a * k**power,
            q[branch][mask],
            G[branch][mask] - eta,
        )
        coeffs[branch] = np.array([popt[0], pcov[0, 0]])
    return coeffs


# ---------------------------------------------------------------------------
# Thermal conductivity
# ---------------------------------------------------------------------------


def compute_hydro_integral(
    omega_T: NDArray[np.floating],
    omega_L: NDArray[np.floating],
    gamma_T: NDArray[np.floating],
    gamma_L: NDArray[np.floating],
    c_sound: dict[str, float],
    temp: float,
) -> tuple[float, float]:
    r"""Compute the hydrodynamic thermal conductivity integral.

    Parameters
    ----------
    omega_T, omega_L : arrays
        Transverse and longitudinal frequency grids.
    gamma_T, gamma_L : arrays
        Linewidths on the respective grids.
    c_sound : dict
        ``{'T': cT, 'L': cL}`` sound velocities.
    temp : float
        Temperature in Kelvin.

    Returns
    -------
    kappa_T, kappa_L : float
        Transverse and longitudinal contributions (W/m/K).
    """
    cT = c_sound["T"]
    cL = c_sound["L"]
    cvT = calculate_heat_capacity(omega_T, temp)
    cvL = calculate_heat_capacity(omega_L, temp)
    dosT = dos_omega_T(omega_T, cT)
    dosL = dos_omega_L(omega_L, cL)
    integrand_T = 1e22 * cT**2 * cvT * dosT / 2 / gamma_T / 3
    integrand_L = 1e22 * cL**2 * cvL * dosL / 2 / gamma_L / 3
    return float(trapezoid(integrand_T, omega_T)), float(trapezoid(integrand_L, omega_L))


def hydrodynamic_contribution(
    spectrum: dict,
    temp: float,
    *,
    q_max: float | None = None,
    omega_max: float | dict[str, float] | None = None,
    omega_min: float = 0,
    is_interpolate: str = "spline",
    harmonic_eta: float | None = None,
    anharmonic_gamma: object | None = None,
    linear_dispersion: bool = True,
    fit_func_Gamma: str = "lorentzian",
) -> tuple[float, float, float, dict]:
    r"""Compute hydrodynamic contribution to thermal conductivity.

    Parameters
    ----------
    spectrum : dict
        Dynamic structure factor dictionary.
    temp : float
        Temperature in Kelvin.
    q_max : float, optional
        Wavevector cutoff (provide *either* this or *omega_max*).
    omega_max : float or dict, optional
        Frequency cutoff.  Can be a dict ``{'L': ..., 'T': ...}`` for
        Ioffe-Regel cutoffs.
    omega_min : float
        Lower frequency bound.
    is_interpolate : str
        Interpolation scheme for linewidths.
    harmonic_eta : float, optional
        Harmonic broadening to subtract.
    anharmonic_gamma : callable, optional
        Function returning anharmonic linewidths.
    linear_dispersion : bool
        Use linear (Debye) dispersion.
    fit_func_Gamma : str
        Lineshape for linewidth extraction.

    Returns
    -------
    kappa_T : float
        Transverse contribution (W/m/K).
    kappa_L : float
        Longitudinal contribution (W/m/K).
    kappa_total : float
        Total contribution.
    debug : dict
        Intermediate quantities for inspection.
    """
    if omega_max is not None and q_max is not None:
        raise ValueError("Provide either `q_max` or `omega_max`, not both.")

    fit = compute_disorder_widths(spectrum, qmax=1, qmax_c=0.6, fit_func=fit_func_Gamma)
    cT = fit["c_sound"]["T"][0]
    cL = fit["c_sound"]["L"][0]
    log.info("Speed of sound: c_L = %.2f m/s, c_T = %.2f m/s", cL * 100, cT * 100)

    if omega_max is not None:
        if isinstance(omega_max, dict):
            omega_max_T = omega_max["T"]
            omega_max_L = omega_max["L"]
        else:
            omega_max_T = omega_max_L = omega_max
    elif q_max is not None:
        omega_max_T = cT * q_max
        omega_max_L = cL * q_max
    else:
        raise ValueError("Provide one of `q_max` or `omega_max`.")

    omega_T = np.linspace(omega_min, omega_max_T, 5000, endpoint=True)
    omega_L = np.linspace(omega_min, omega_max_L, 5000, endpoint=True)

    dosT = dos_omega_T(omega_T, cT)
    dosL = dos_omega_L(omega_L, cL)

    # Build Gamma arrays
    if harmonic_eta is None:
        xT = np.insert(fit["q"]["T"], 0, 0)
        yT = np.insert(fit["Gamma"]["T"], 0, 0)
        xL = np.insert(fit["q"]["L"], 0, 0)
        yL = np.insert(fit["Gamma"]["L"], 0, 0)
    else:
        xT = np.insert(fit["q"]["T"], 0, 0)
        yT = np.insert(fit["Gamma"]["T"] - harmonic_eta, 0, 0)
        xL = np.insert(fit["q"]["L"], 0, 0)
        yL = np.insert(fit["Gamma"]["L"] - harmonic_eta, 0, 0)
        if anharmonic_gamma is not None:
            yT += anharmonic_gamma(xT / cT / 2 / np.pi) ** 2
            yL += anharmonic_gamma(xL / cL / 2 / np.pi) ** 2

    gammaT, gammaL = _interpolate_gamma(
        xT,
        yT,
        xL,
        yL,
        omega_T,
        omega_L,
        cT,
        cL,
        omega_max_T,
        omega_max_L,
        is_interpolate,
    )

    cvT = calculate_heat_capacity(omega_T, temp)
    cvL = calculate_heat_capacity(omega_L, temp)

    if linear_dispersion:
        integrand_T = 1e22 * cT**2 * cvT * dosT / 2 / gammaT / 3
        integrand_L = 1e22 * cL**2 * cvL * dosL / 2 / gammaL / 3
    else:
        raise NotImplementedError(
            "Non-linear dispersion is not yet supported in the refactored API"
        )

    kappa_T = float(trapezoid(integrand_T, omega_T))
    kappa_L = float(trapezoid(integrand_L, omega_L))

    debug = {
        "omega_L": omega_L,
        "omega_T": omega_T,
        "gamma_L": gammaL,
        "gamma_T": gammaT,
        "dos_L": dosL,
        "dos_T": dosT,
        "cv_L": cvL,
        "cv_T": cvT,
        "omega_max": {"T": omega_max_T, "L": omega_max_L},
        "kappa_vs_omega_L": integrand_L,
        "kappa_vs_omega_T": integrand_T,
    }
    return kappa_T, kappa_L, kappa_T + kappa_L, debug


def _interpolate_gamma(
    xT: NDArray,
    yT: NDArray,
    xL: NDArray,
    yL: NDArray,
    omega_T: NDArray,
    omega_L: NDArray,
    cT: float,
    cL: float,
    omega_max_T: float,
    omega_max_L: float,
    method: str,
) -> tuple[NDArray, NDArray]:
    """Interpolate linewidths from fitted data points."""
    if method == "spline":
        splT = InterpolatedUnivariateSpline(xT, np.sqrt(yT), k=1)
        gammaT = splT(omega_T / cT) ** 2
        splL = InterpolatedUnivariateSpline(xL, np.sqrt(yL), k=1)
        gammaL = splL(omega_L / cL) ** 2
    elif method == "square":
        w_fit = xT * cT
        mask = w_fit / 2 / np.pi < 3.5
        popt, _pcov = curve_fit(lambda q, a: a * q**2, w_fit[mask], yT[mask])
        gammaT = popt[0] * omega_T**2

        w_fit = xL * cL
        mask = w_fit / 2 / np.pi < 3.5
        popt, _pcov = curve_fit(lambda q, a: a * q**2, w_fit[mask], yL[mask])
        gammaL = popt[0] * omega_L**2
    elif method == "2&4":

        def sq_and_qu(w: NDArray, a: float, b: float) -> NDArray:
            return np.abs(a) * w**2 + np.abs(b) * w**4

        w_fit = xT * cT
        mask = w_fit < omega_max_T
        sigma = w_fit[mask] ** 2
        sigma[0] = sigma[1]
        popt, _pcov = curve_fit(sq_and_qu, w_fit[mask], yT[mask], sigma=sigma)
        gammaT = sq_and_qu(omega_T, *popt)

        w_fit = xL * cL
        mask = w_fit < omega_max_L
        sigma = w_fit[mask] ** 2
        sigma[0] = sigma[1]
        popt, _pcov = curve_fit(sq_and_qu, w_fit[mask], yL[mask], sigma=sigma)
        gammaL = sq_and_qu(omega_L, *popt)
    else:
        raise ValueError(f"Unknown interpolation method: '{method}'")

    return gammaT, gammaL
