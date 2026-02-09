"""Adapters for ingesting kaldo phonon objects.

Provides :func:`v2_matrix` for computing velocity-operator matrices and
:func:`Ctt_correlator` for current-current correlation functions from
``kaldo.Phonons`` objects.

``kaldo`` is an optional dependency.  If it is not installed, these
functions raise an :class:`ImportError` with installation instructions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from numpy.typing import NDArray

from vibroglass.analysis.structure_factor import compute_braQketn

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

try:
    from kaldo.observables.harmonic_with_q import HarmonicWithQ
except ImportError:
    HarmonicWithQ = None  # type: ignore[assignment,misc]


def _require_kaldo() -> None:
    """Raise a friendly error if kaldo is not installed."""
    if HarmonicWithQ is None:
        raise ImportError(
            "kaldo is required for this function. Install it with: pip install kaldo"
        )


def v2_matrix(
    phonons: object,
    *,
    cutoff_n: int = 3000,
    nomega: int = 1000,
    delta_width: float = 0.2,
    vectorize: bool = True,
    discrete_delta: bool = False,
    normalize: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    r"""Compute the velocity-operator matrix :math:`v^2(\omega, \omega')`.

    Parameters
    ----------
    phonons : kaldo.Phonons
        A kaldo ``Phonons`` object with force constants and eigenvectors.
    cutoff_n : int
        Number of physical modes to include (starting from mode 3).
    nomega : int
        Number of frequency grid points.
    delta_width : float
        Width of the Gaussian (or discrete) delta approximation.
    vectorize : bool
        Use vectorised tensor contraction (faster).
    discrete_delta : bool
        Use a discrete box-car delta instead of Gaussian.
    normalize : bool
        Normalise the result by the delta-function norms.

    Returns
    -------
    omegas : (nomega,) array
        Frequency grid.
    v2_omega : (nomega, nomega) array
        Velocity-operator matrix on the frequency grid.

    Raises
    ------
    ImportError
        If ``kaldo`` is not installed.
    ValueError
        If the computed :math:`v_{ij}^2` matrix is not symmetric.
    """
    _require_kaldo()

    q_points = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    n_modes = phonons.n_modes

    for ik in range(len(q_points)):
        q_point = q_points[ik]
        hwq_kwargs = {
            "q_point": q_point,
            "second": phonons.forceconstants.second,
            "distance_threshold": phonons.forceconstants.distance_threshold,
            "folder": phonons.folder,
            "storage": phonons.storage,
            "is_nw": phonons.is_nw,
            "is_unfolding": phonons.is_unfolding,
        }
        sij_x = HarmonicWithQ(**hwq_kwargs)._sij_x
        sij_y = HarmonicWithQ(**hwq_kwargs)._sij_y
        sij_z = HarmonicWithQ(**hwq_kwargs)._sij_z

    # Symmetrise and isotropic-average
    sij2_x = 0.5 * (np.abs(sij_x) ** 2 + np.abs(sij_x.T) ** 2)
    sij2_y = 0.5 * (np.abs(sij_y) ** 2 + np.abs(sij_y.T) ** 2)
    sij2_z = 0.5 * (np.abs(sij_z) ** 2 + np.abs(sij_z.T) ** 2)
    sij2 = (sij2_x + sij2_y + sij2_z) / 3

    omega = phonons.omega.flatten()
    omega_n_m = omega[:, np.newaxis] * omega[np.newaxis, :]
    vij2 = np.reshape(sij2 / omega_n_m, (n_modes, n_modes))

    if not np.allclose(vij2, vij2.T, atol=1e-6):
        raise ValueError("v_ij^2 matrix is not symmetric (tolerance 1e-6)")

    def _delta_vec(w: NDArray, s: float, *, discrete: bool = False) -> NDArray:
        if discrete:
            return np.where(np.abs(w) < s, 0.5 / s, 0.0)
        return np.exp(-(w**2) / (2 * s**2)) / (s * np.sqrt(2 * np.pi))

    omega_n = phonons.omega.flatten()[3 : cutoff_n + 3]
    omegas = np.linspace(0, omega_n[-1], nomega)
    v2nm = vij2[3 : cutoff_n + 3, 3 : cutoff_n + 3]

    if vectorize:
        delta_omega = _delta_vec(
            omegas[:, np.newaxis] - omega_n[np.newaxis, :],
            delta_width,
            discrete=discrete_delta,
        )
        if normalize:
            norm = delta_omega.sum(axis=1)[:, np.newaxis] * delta_omega.sum(axis=1)[np.newaxis, :]
            v2_omega = np.zeros((nomega, nomega))
            nonzero = norm != 0
            v2_omega[nonzero] = (delta_omega @ v2nm @ delta_omega.T)[nonzero] / norm[nonzero]
        else:
            v2_omega = delta_omega @ v2nm @ delta_omega.T
    else:
        v2_omega = np.zeros((nomega, nomega))
        for i in range(nomega):
            for j in range(nomega):
                d_i = _delta_vec(omegas[i] - omega_n, delta_width)
                d_j = _delta_vec(omegas[j] - omega_n, delta_width)
                v2_omega[i, j] = oe.contract("nm,n,m", v2nm, d_i, d_j)

    return omegas, v2_omega


def Ctt_correlator(
    phonon: object,
    Q_list: NDArray[np.integer],
    *,
    nomega: int = 1000,
    normalize: bool = False,
    debug: bool = False,
    return_gamma: bool = False,
    bandwidths: NDArray[np.floating] | None = None,
) -> tuple:
    r"""Compute the current-current correlator :math:`C_{tt}(Q, \omega)`.

    Parameters
    ----------
    phonon : kaldo.Phonons
        A kaldo ``Phonons`` object.
    Q_list : (n_q, 3) int array
        Wavevector indices in reciprocal-lattice units.
    nomega : int
        Number of frequency grid points.
    normalize : bool
        Normalise each :math:`C(Q, \omega)` in :math:`\omega`.
    debug : bool
        Use a fixed broadening instead of phonon bandwidths.
    return_gamma : bool
        Return additional diagnostic arrays.
    bandwidths : array, optional
        Override the phonon bandwidths.

    Returns
    -------
    Q : (n_q, 3) float array
        Cartesian wavevectors.
    omega_domain : (nomega,) float array
        Angular-frequency grid.
    C : dict
        ``{'L': ..., 'T': ...}`` correlator arrays of shape ``(n_q, nomega)``.

    If *return_gamma* is ``True``, additional arrays are appended to the
    return tuple (gamma, n_omega_delta, delta1, delta2, pop_over_omega).
    """
    const = phonon.hbar / 1.66054e-27 * 1e8  # hbar / 1 a.m.u.

    def _lorentzian(w: NDArray, g: NDArray | float) -> NDArray:
        return g / (w**2 + g**2)

    natm = phonon.n_atoms
    pm = phonon.physical_mode
    nmodes = phonon.eigenvalues[pm].size

    # Reshape eigenvectors
    reshaped_eigvec = np.transpose(
        np.transpose(np.transpose(phonon.eigenvectors.real, axes=(0, 2, 1))[pm]).reshape(
            natm, 3, nmodes
        ),
        axes=(1, 2, 0),
    )

    pos = phonon.atoms.get_positions().T
    omegas = 2 * np.pi * phonon.frequency[pm].flatten()

    bw = phonon.bandwidth[pm].flatten() if bandwidths is None else bandwidths[pm].flatten()

    omin = omegas.min()
    omax = omegas.max()
    omega_domain = np.linspace(omin, omax, nomega)

    reciprocal_cell = phonon.atoms.cell.reciprocal()
    inv_sqrtmass = 1 / np.sqrt(phonon.atoms.get_masses())

    Q, bran_ketQ_sq_L, bran_ketQ_sq_T = compute_braQketn(
        Q_list,
        reciprocal_cell,
        np.array([pos]),
        np.array([reshaped_eigvec]),
        weights=np.array([inv_sqrtmass]),
    )
    bran_ketQ_sq_L = bran_ketQ_sq_L[0]
    bran_ketQ_sq_T = bran_ketQ_sq_T[0]

    C: dict[str, NDArray] = {}

    if debug:
        gamma = 0.01 * (omax - omin)
        delta1 = _lorentzian(omega_domain[:, np.newaxis] + omegas[np.newaxis, :], gamma)
        delta2 = _lorentzian(omega_domain[:, np.newaxis] - omegas[np.newaxis, :], gamma)
        n_omega_delta = delta2.T
        C["L"] = bran_ketQ_sq_L @ n_omega_delta
        C["T"] = bran_ketQ_sq_T @ n_omega_delta
        if normalize:
            for branch in C:
                norms = np.trapezoid(C[branch], omega_domain, axis=1)
                C[branch] = (C[branch].T / norms).T
    else:
        gamma = np.zeros_like(omega_domain)[:, np.newaxis] + bw[np.newaxis, :]
        delta1 = _lorentzian(omega_domain[:, np.newaxis] + omegas[np.newaxis, :], gamma)
        delta2 = _lorentzian(omega_domain[:, np.newaxis] - omegas[np.newaxis, :], gamma)

        pop = phonon.population.copy()
        pop[~phonon.physical_mode] = 0
        pop = pop[pm].flatten()

        n_omega_delta = ((delta1 * pop + delta2 * (1 + pop)) / omegas).T
        C["L"] = const * bran_ketQ_sq_L @ n_omega_delta
        C["T"] = const * bran_ketQ_sq_T @ n_omega_delta

        if normalize:
            for branch in C:
                norms = np.trapezoid(C[branch], omega_domain, axis=1)
                C[branch] = (C[branch].T / norms).T

    if return_gamma:
        return (
            Q,
            omega_domain,
            C,
            gamma,
            n_omega_delta,
            delta1,
            delta2,
            phonon.population[pm].flatten() / omegas,
        )
    return Q, omega_domain, C
