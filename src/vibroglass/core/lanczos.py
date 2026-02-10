"""Haydock-Lanczos algorithm for computing vibrational spectra.

Provides memory-efficient Lanczos tridiagonalisation, continued-fraction
evaluation, and a high-level :func:`spectrum` function that combines both.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

log = logging.getLogger(__name__)


def continued_fraction(
    a_coefficients: NDArray[np.floating] | list[float],
    b_coefficients: NDArray[np.floating] | list[float],
) -> float | complex:
    """Evaluate a continued fraction from *a* and *b* coefficients.

    Parameters
    ----------
    a_coefficients : array_like
        Diagonal coefficients.
    b_coefficients : array_like
        Off-diagonal coefficients (same length as *a_coefficients*).

    Returns
    -------
    float or complex
        Value of the continued fraction.
    """
    n = len(a_coefficients)
    if n == 0:
        return 0
    if n == 1:
        return a_coefficients[0] + b_coefficients[0]
    return a_coefficients[0] + b_coefficients[0] / continued_fraction(
        a_coefficients[1:], b_coefficients[1:]
    )


def compute_b2(beta: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """Build the *b*-squared array for the continued fraction.

    Parameters
    ----------
    beta : 1-D array
        Off-diagonal Lanczos coefficients.

    Returns
    -------
    NDArray
        Array of length ``len(beta) + 1`` with ``b2[0] = 1`` and
        ``b2[1:] = -beta**2``.
    """
    b2 = np.zeros(len(beta) + 1, dtype=complex)
    b2[0] = 1
    b2[1:] = -(beta**2)
    return b2


def recompute_spectrum(
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    z2: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Recompute a spectrum from saved Lanczos coefficients.

    Useful for anharmonic post-processing where broadening is modified
    without re-running the Lanczos iteration.

    Parameters
    ----------
    alpha : 1-D array
        Diagonal Lanczos coefficients.
    beta : 1-D array
        Off-diagonal Lanczos coefficients.
    z2 : 1-D array
        Squared complex frequencies ``(omega + i*eta)**2``.

    Returns
    -------
    NDArray
        ``|Im(y)|`` evaluated on *z2*.
    """
    b2 = compute_b2(beta)
    y = np.zeros(len(z2), dtype=complex)
    for i, z2_ in enumerate(z2):
        y[i] = continued_fraction(np.insert(z2_ - alpha, 0, 0), b2)
    return np.abs(np.imag(y))


def lanczos_cheap(
    A: NDArray | spmatrix,
    v: NDArray[np.floating],
    k: int,
    *,
    last2: bool = False,
) -> (
    tuple[NDArray[np.floating], NDArray[np.floating]]
    | tuple[NDArray[np.floating], NDArray[np.floating], NDArray, NDArray]
):
    """Memory-efficient Lanczos tridiagonalisation.

    Parameters
    ----------
    A : (n, n) array or sparse matrix
        Symmetric matrix.
    v : (n,) array
        Starting vector (will be normalised internally).
    k : int
        Number of Lanczos steps.
    last2 : bool, optional
        If ``True``, also return the last two Lanczos vectors (useful
        for restarting).

    Returns
    -------
    alpha : (k,) array
        Diagonal coefficients.
    beta : (k,) array
        Off-diagonal coefficients.
    v, v_minus : (n,) arrays
        Returned only when *last2* is ``True``.
    """
    alpha_array = np.zeros(k)
    beta_array = np.zeros(k)
    v = v / np.linalg.norm(v)
    v_minus = np.zeros_like(v, dtype=complex)
    beta = 0.0
    for j in range(k):
        w = A @ v
        alpha = np.real(np.vdot(v, w))
        if j == k - 1:
            break
        w = w - alpha * v - (beta * v_minus if j > 0 else 0)
        beta = np.linalg.norm(w)
        if beta == 0:
            break
        v_minus = v
        v = w / beta
        alpha_array[j] = alpha
        beta_array[j] = beta
    if not last2:
        return alpha_array, beta_array
    return alpha_array, beta_array, v, v_minus


def lanczos_restart(
    A: NDArray | spmatrix,
    k: int,
    alpha_old: NDArray[np.floating],
    beta_old: NDArray[np.floating],
    v: NDArray,
    v_minus: NDArray,
    *,
    last2: bool = False,
) -> (
    tuple[NDArray[np.floating], NDArray[np.floating]]
    | tuple[NDArray[np.floating], NDArray[np.floating], NDArray, NDArray]
):
    """Restart Lanczos from a previous state.

    Parameters
    ----------
    A : (n, n) array or sparse matrix
        Symmetric matrix.
    k : int
        Number of *additional* Lanczos steps.
    alpha_old, beta_old : 1-D arrays
        Coefficients from a previous :func:`lanczos_cheap` call.
    v, v_minus : (n,) arrays
        Last two Lanczos vectors from the previous run.
    last2 : bool, optional
        Return last two vectors for further restarts.

    Returns
    -------
    alpha, beta : 1-D arrays
        Combined old + new coefficients.
    v, v_minus : (n,) arrays
        Only if *last2* is ``True``.
    """
    alpha_old = alpha_old[:-1]
    beta_old = beta_old[:-1]
    k_old = len(alpha_old)

    alpha_array = np.zeros(k + k_old)
    beta_array = np.zeros(k + k_old)
    alpha_array[:k_old] = alpha_old
    beta_array[:k_old] = beta_old

    v = v / np.linalg.norm(v)
    beta = beta_old[-1]
    for j in range(k_old, k_old + k):
        w = A @ v
        alpha = np.real(np.vdot(v, w))
        if j == k_old + k - 1:
            break
        w = w - alpha * v - (beta * v_minus if j > 0 else 0)
        beta = np.linalg.norm(w)
        if beta == 0:
            break
        v_minus = v
        v = w / beta
        alpha_array[j] = alpha
        beta_array[j] = beta
    if not last2:
        return alpha_array, beta_array
    return alpha_array, beta_array, v, v_minus


def lanczos_ortho(
    A: NDArray | spmatrix,
    u_in: NDArray,
    nsteps: int,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Lanczos with forced orthogonalisation.

    Parameters
    ----------
    A : (n, n) array or sparse matrix
        Symmetric matrix.
    u_in : (n,) array
        Starting vector.
    nsteps : int
        Number of Lanczos steps.

    Returns
    -------
    T : (nsteps, nsteps) array
        Tridiagonal matrix.
    U : (n, nsteps) array
        Orthonormal Lanczos vectors.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if u_in.shape[0] != A.shape[0]:
        raise ValueError("Incompatible matrix and vector shapes.")

    dim_a = A.shape[0]
    U = np.zeros((dim_a, nsteps), dtype=complex)
    T = np.zeros((nsteps, nsteps))

    U[:, 0] = u_in / np.linalg.norm(u_in)
    alpha = np.real(U[:, 0].conj() @ (A @ U[:, 0]))
    T[0, 0] = alpha

    W = A @ U[:, 0] - alpha * U[:, 0]

    for i in range(1, nsteps):
        beta = np.linalg.norm(W)
        if beta == 0:
            break
        W = W / beta
        W = W - U @ (U.T.conj() @ W)
        U[:, i] = W

        alpha = np.real(U[:, i].conj() @ (A @ U[:, i]))
        T[i, i] = alpha
        T[i - 1, i] = beta
        T[i, i - 1] = beta

        W = A @ U[:, i] - beta * U[:, i - 1] - alpha * U[:, i]

    return T, U


def lanczos(
    A: NDArray | spmatrix,
    v: NDArray,
    k: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray]:
    """Full Lanczos with explicit Q matrix (memory-intensive).

    Parameters
    ----------
    A : (n, n) array or sparse matrix
        Symmetric matrix.
    v : (n,) array
        Starting vector.
    k : int
        Number of Lanczos steps.

    Returns
    -------
    alpha : (k,) array
        Diagonal coefficients.
    beta : (k,) array
        Off-diagonal coefficients.
    Q : (n, k) array
        Lanczos vectors.
    """
    n = A.shape[0]
    Q = np.zeros((n, k), dtype=complex)
    T = np.zeros((k, k))

    Q[:, 0] = v / np.linalg.norm(v)

    for j in range(k):
        w = np.matmul(A, Q[:, j], dtype=complex)
        alpha = np.vdot(Q[:, j], w)
        if j == k - 1:
            break
        w = w - alpha * Q[:, j] - (T[j, j - 1] * Q[:, j - 1] if j > 0 else 0)
        beta = np.linalg.norm(w)
        if beta == 0:
            break
        Q[:, j + 1] = w / beta
        T[j, j] = alpha
        T[j, j + 1] = beta
        T[j + 1, j] = beta

    alpha_array = np.diagonal(T)
    beta_array = np.zeros(k)
    beta_array[:-1] = np.diagonal(T, offset=1)
    return alpha_array, beta_array, Q


def spectrum(
    A: NDArray | spmatrix,
    v: NDArray[np.floating],
    k: int,
    omega_array: NDArray[np.floating],
    eta: float,
    *,
    return_chain: bool = False,
    use_ortho: bool = False,
    norm_A: float = 1,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray, NDArray]:
    """Compute a spectrum via the Haydock-Lanczos continued fraction.

    Parameters
    ----------
    A : (n, n) array or sparse matrix
        Symmetric dynamical matrix.
    v : (n,) array
        Initial (ket) vector.
    k : int
        Number of Lanczos iterations.
    omega_array : 1-D array
        Frequency grid.
    eta : float
        Lorentzian broadening (imaginary part of ``z``).
    return_chain : bool, optional
        Also return ``(alpha, beta)`` Lanczos coefficients.
    use_ortho : bool, optional
        Use forced orthogonalisation variant.
    norm_A : float, optional
        Rescale *A* coefficients by this factor.

    Returns
    -------
    spec : 1-D array
        ``|Im(1 / (z² - D²))|`` on *omega_array*.
    alpha, beta : 1-D arrays
        Only returned when *return_chain* is ``True``.
    """
    if not use_ortho:
        alpha, beta = lanczos_cheap(A, v, k)
    else:
        T, _U = lanczos_ortho(A, v, nsteps=k)
        alpha = np.diagonal(T)
        beta = np.append(np.diagonal(T, offset=1), 0)

    alpha = alpha * norm_A
    beta = beta * norm_A

    b2 = compute_b2(beta)

    y = np.zeros_like(omega_array, dtype=complex)
    for i, omega in enumerate(omega_array):
        z2 = (omega + 1j * eta) ** 2
        y[i] = continued_fraction(np.insert(z2 - alpha, 0, 0), b2)

    if not return_chain:
        return np.abs(np.imag(y))
    return np.abs(np.imag(y)), alpha, beta


def convert_spectrum_lz2hydro(spec: dict) -> dict:
    """Convert Lanczos spectrum dict to hydrodynamic tools format.

    Parameters
    ----------
    spec : dict
        Spectrum dictionary as returned by
        :meth:`~vibroglass.analysis.spectra.VibrationalSpectra.compute_vdsf`.

    Returns
    -------
    dict
        Dictionary with keys ``'omega'``, ``'q'``, ``'S'``.
    """
    new = {}
    new["omega"] = spec["omega"]
    new["q"] = np.linalg.norm(spec["Q"], axis=1)
    new["S"] = {}
    for b in ("L", "T"):
        new["S"][b] = np.array([spec[i][b]["S"] for i in range(len(new["q"]))])[:, :, np.newaxis]
    return new
