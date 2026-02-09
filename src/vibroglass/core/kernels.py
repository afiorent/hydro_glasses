"""Frequency smoothing kernels for VDOS and spectral interpolation.

Provides Lorentzian, Gaussian, and Damped Harmonic Oscillator (DHO)
kernels, plus convenience functions for computing kernel-smoothed
spectra and vibrational densities of states.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lorentzian(x: NDArray[np.floating] | float, eta: float) -> NDArray[np.floating]:
    """Lorentzian (Cauchy) kernel.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    eta : float
        Half-width at half-maximum.

    Returns
    -------
    NDArray
        ``eta / (pi * (x**2 + eta**2))``.
    """
    return eta / np.pi / (np.asarray(x) ** 2 + eta**2)


def gauss(x: NDArray[np.floating] | float, sigma: float) -> NDArray[np.floating]:
    """Normalised Gaussian kernel.

    Parameters
    ----------
    x : array_like
        Evaluation points.
    sigma : float
        Standard deviation.

    Returns
    -------
    NDArray
        Gaussian probability density evaluated at *x*.
    """
    return np.exp(-0.5 * (np.asarray(x) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def dho(
    w: NDArray[np.floating],
    w0: float,
    sigma: float,
    norm: float,
) -> NDArray[np.floating]:
    """Damped Harmonic Oscillator kernel.

    Parameters
    ----------
    w : array_like
        Frequency grid.
    w0 : float
        Resonance frequency.
    sigma : float
        Damping parameter.
    norm : float
        Overall normalisation factor.

    Returns
    -------
    NDArray
        DHO lineshape evaluated on *w*.
    """
    w = np.asarray(w)
    return norm * (w * 2 * sigma / ((w * sigma) ** 2 + (w**2 - w0**2) ** 2))


def _wdho(w: NDArray[np.floating], w0: float, sigma: float) -> NDArray[np.floating]:
    """Weighted DHO kernel (used internally by :func:`compute_vdos`)."""
    w = np.asarray(w)
    return w**2 * 2 * sigma / ((w * sigma) ** 2 + (w**2 - w0**2) ** 2)


def compute_kernel(
    frequencies: NDArray[np.floating],
    values: NDArray[np.floating],
    freq_range: NDArray[np.floating],
    sigma: float,
    kernel: str = "lorentz",
    normalize: bool = True,
) -> NDArray[np.floating]:
    r"""Compute :math:`y(\omega) = \sum_i y_i\,K(\omega - \omega_i)`.

    Parameters
    ----------
    frequencies : 1-D array
        Mode frequencies :math:`\omega_i`.
    values : 1-D array
        Weights :math:`y_i` (same length as *frequencies*).
    freq_range : 1-D array
        Output frequency grid.
    sigma : float
        Kernel width (``eta`` for Lorentz, ``sigma`` for Gauss).
    kernel : ``'lorentz'`` | ``'DHO'`` | ``'gauss'``
        Kernel type.
    normalize : bool
        If ``True``, divide result by its integral over *freq_range*.

    Returns
    -------
    NDArray
        Smoothed spectrum on *freq_range*.

    Raises
    ------
    ValueError
        If inputs are not 1-D or have mismatched lengths, or if *kernel*
        is unrecognised.
    """
    freqs = np.asarray(frequencies, dtype=float)
    vals = np.asarray(values, dtype=float)
    omega = np.asarray(freq_range, dtype=float)

    if freqs.ndim != 1 or vals.ndim != 1:
        raise ValueError("`frequencies` and `values` must be 1-D arrays")
    if freqs.size != vals.size:
        raise ValueError("`frequencies` and `values` must have the same length")

    y_out = np.zeros_like(omega, dtype=float)

    if kernel == "lorentz":
        for f, wgt in zip(freqs, vals, strict=True):
            x = omega - f
            y_out += wgt * (sigma / np.pi) / (x * x + sigma * sigma)
    elif kernel == "DHO":
        for f, wgt in zip(freqs, vals, strict=True):
            y_out += (
                wgt * (omega**2 * 2.0 * sigma) / ((omega * sigma) ** 2 + (omega**2 - f**2) ** 2)
            )
    elif kernel == "gauss":
        norm_prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        for f, wgt in zip(freqs, vals, strict=True):
            x = omega - f
            y_out += wgt * norm_prefactor * np.exp(-0.5 * (x / sigma) ** 2)
    else:
        raise ValueError("Unknown kernel, choose 'lorentz', 'DHO' or 'gauss'")

    if normalize:
        integral = np.trapezoid(y_out, omega)
        if integral != 0.0:
            y_out /= integral

    return y_out


def compute_vdos(
    frequencies: NDArray[np.floating],
    freq_range: NDArray[np.floating],
    sigma: float,
    kernel: str = "lorentz",
) -> NDArray[np.floating]:
    """Compute the vibrational density of states (VDOS).

    Parameters
    ----------
    frequencies : 1-D array
        Mode frequencies.
    freq_range : 1-D array
        Frequency grid for evaluation.
    sigma : float
        Kernel width parameter.
    kernel : ``'lorentz'`` | ``'DHO'`` | ``'gauss'``
        Kernel type.

    Returns
    -------
    NDArray
        Normalised VDOS on *freq_range*.
    """
    freqs = np.asarray(frequencies, dtype=float)
    omega = np.asarray(freq_range, dtype=float)
    vdos = np.zeros_like(omega)

    if kernel == "lorentz":
        for freq in freqs:
            vdos += lorentzian(omega - freq, sigma)
    elif kernel == "DHO":
        for freq in freqs:
            vdos += _wdho(omega, freq, sigma)
    else:
        for freq in freqs:
            vdos += gauss(omega - freq, sigma)

    vdos /= np.trapezoid(vdos, omega)
    return vdos
