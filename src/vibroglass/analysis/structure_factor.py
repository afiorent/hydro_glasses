"""Dynamic structure factor and related amorphous-material observables.

Core functions for computing pseudo-plane-wave projections, dynamic
structure factors S(Q, omega), inverse participation ratios, and IR
vectors for amorphous systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ase import Atoms


def compute_phi_Q(
    Q_list: NDArray[np.integer],
    reciprocal_cell: NDArray[np.floating],
    pos: NDArray[np.floating],
    *,
    fix_eperp: NDArray[np.floating] | None = None,
    cell_sym: bool = False,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    r"""Compute pseudo-plane-wave states for wavevectors in *Q_list*.

    Parameters
    ----------
    Q_list : (n_q, 3) int array
        Wavevector indices in reciprocal-lattice units.
    reciprocal_cell : (3, 3) float array
        Reciprocal-cell matrix.
    pos : (n_snapshots, 3, n_atoms) float array
        Atomic positions (transposed layout).
    fix_eperp : (3,) float array, optional
        Fixed transverse polarisation direction.
    cell_sym : bool
        Use cell-symmetry-adapted transverse polarisations.

    Returns
    -------
    phi_L : (n_atoms, 3, n_q) complex array
        Longitudinal plane-wave states.
    phi_T : (n_atoms, 3, n_q) complex array
        Transverse plane-wave states.
    """
    Q = np.array([2 * np.pi * reciprocal_cell @ Q_ for Q_ in Q_list])
    Qn = oe.contract("qa,q->qa", Q, 1 / np.linalg.norm(Q, axis=1))

    if not cell_sym:
        if fix_eperp is None:
            eperp = np.random.rand(*Q.shape)
            eperp -= np.array([Q_ * eperp_.dot(Q_) for eperp_, Q_ in zip(eperp, Qn, strict=True)])
        else:
            eperp = fix_eperp * np.ones_like(Q_list, dtype=float)
    else:
        eT = 2 * np.pi * reciprocal_cell @ np.array([1, 0, 0])
        eT /= np.linalg.norm(eT)
        eT_2 = 2 * np.pi * reciprocal_cell @ np.array([0, 1, 0])
        eT_2 /= np.linalg.norm(eT_2)
        eperp = np.array(
            [
                np.cross(Q_, eT) if np.linalg.norm(np.cross(Q_, eT)) > 1e-6 else np.cross(Q_, eT_2)
                for Q_ in Q_list
            ]
        )

    eperp = oe.contract("qa,q->qa", eperp, 1 / np.linalg.norm(eperp, axis=1))

    exp_i_Q_dot_R = np.exp(1j * np.transpose(Q @ pos, axes=(0, 2, 1)))

    exp_i_Q_dot_R_L = oe.contract("qa,Iq->Iaq", Qn, exp_i_Q_dot_R[0, :, :])
    exp_i_Q_dot_R_T = oe.contract("qa,Iq->Iaq", eperp, exp_i_Q_dot_R[0, :, :])
    return exp_i_Q_dot_R_L, exp_i_Q_dot_R_T


def compute_Q_dot_product(
    Q_list: NDArray[np.integer],
    positions: NDArray[np.floating],
    reciprocal_cell: NDArray[np.floating],
) -> dict:
    r"""Compute dot products between pseudo-plane-wave states.

    Parameters
    ----------
    Q_list : (n_q, 3) int array
        Wavevector indices.
    positions : (n_snapshots, n_atoms, 3) float array
        Atomic positions.
    reciprocal_cell : (3, 3) float array
        Reciprocal-cell matrix.

    Returns
    -------
    dict
        Nested dictionary ``Q_dot_product[Q1][Q2]`` with overlap values.
    """
    from itertools import product

    natm = positions.shape[1]
    exp_dict: dict[str, NDArray] = {}
    for Q_ in Q_list:
        lbl = str(list(Q_))
        Q = 2 * np.pi * reciprocal_cell @ Q_
        exp_dict[lbl] = np.exp(1j * np.einsum("a,tIa->tI", Q, positions)) / np.sqrt(natm)

    result: dict[str, dict[str, NDArray]] = {}
    for Q1, Q2 in product(Q_list, Q_list):
        lbl1, lbl2 = str(list(Q1)), str(list(Q2))
        if lbl1 not in result:
            result[lbl1] = {}
        result[lbl1][lbl2] = np.einsum("tI,tI->t", exp_dict[lbl1], exp_dict[lbl2].conj())
    return result


def inv_participation_ratio(eigvec: NDArray[np.floating]) -> NDArray[np.floating]:
    r"""Inverse participation ratio per mode.

    .. math::

        1/p_n = \sum_i |e_{i,n}|^4

    Parameters
    ----------
    eigvec : (n_modes, n_dof) array
        Eigenvectors (rows are modes).

    Returns
    -------
    (n_modes,) array
        Inverse participation ratios.
    """
    return np.sum(eigvec**4, axis=1)


def compute_braQketn(
    Q_list: NDArray[np.integer],
    reciprocal_cell: NDArray[np.floating],
    pos: NDArray[np.floating],
    reshaped_eigvec: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    r"""Compute :math:`|\langle Q | n \rangle|^2` projections.

    Parameters
    ----------
    Q_list : (n_q, 3) int array
        Wavevector indices.
    reciprocal_cell : (3, 3) float array
        Reciprocal-cell matrix.
    pos : (T, 3, n_atoms) float array
        Transposed positions for each snapshot.
    reshaped_eigvec : (T, 3, n_modes, n_atoms) float array
        Reshaped eigenvectors.
    weights : (T, n_atoms) float array, optional
        Per-atom weights (e.g. inverse square-root masses).

    Returns
    -------
    Q : (n_q, 3) float array
        Cartesian wavevectors.
    braQketn_L : (T, n_q, n_modes) float array
        Squared longitudinal projections.
    braQketn_T : (T, n_q, n_modes) float array
        Squared transverse projections.
    """
    Q = np.array([2 * np.pi * reciprocal_cell @ Q_ for Q_ in Q_list])
    Qn = oe.contract("qa,q->qa", Q, 1 / np.linalg.norm(Q, axis=1))

    eperp = np.random.rand(*Q.shape)
    eperp -= np.array([Q_ * eperp_.dot(Q_) for eperp_, Q_ in zip(eperp, Qn, strict=True)])
    eperp = oe.contract("qa,q->qa", eperp, 1 / np.linalg.norm(eperp, axis=1))

    natm = pos.shape[2]
    if weights is not None:
        reshaped_eigvec = oe.contract("tani,ti->tani", reshaped_eigvec, weights)

    exp_i_Q_dot_R = np.exp(1j * np.transpose(Q @ pos, axes=(0, 2, 1))) / np.sqrt(3 * natm)
    bran_ketQ = np.array([r @ e for r, e in zip(reshaped_eigvec, exp_i_Q_dot_R, strict=True)])
    bran_ketQ_sq_L = np.abs(oe.contract("qa,tanq->tqn", Qn, bran_ketQ)) ** 2
    bran_ketQ_sq_T = np.abs(oe.contract("qa,tanq->tqn", eperp, bran_ketQ)) ** 2
    return Q, bran_ketQ_sq_L, bran_ketQ_sq_T


def compute_SQomega(
    Q_list: NDArray[np.integer],
    eigval: NDArray[np.floating],
    eigvec: NDArray[np.floating],
    positions: NDArray[np.floating],
    reciprocal_cell: NDArray[np.floating],
    *,
    cutoff: float | None = None,
    nomega: int = 1000,
    use_soft: bool = True,
    domega: float = 0.01,
    is_anharmonic: bool = False,
    gammas: NDArray[np.floating] | None = None,
    vectorize: bool = True,
) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    r"""Compute the dynamic structure factor :math:`S(Q, \omega)`.

    Parameters
    ----------
    Q_list : (n_q, 3) int array
        Wavevector indices in reciprocal-lattice units.
    eigval : (T, n_modes) float array
        Eigenvalues (squared angular frequencies).
    eigvec : (T, n_atoms, 3, n_modes) float array
        Eigenvectors.
    positions : (T, n_atoms, 3) float array
        Atomic positions for each snapshot.
    reciprocal_cell : (3, 3) float array
        Reciprocal-cell matrix.
    cutoff : float, optional
        Frequency cutoff in THz.
    nomega : int
        Number of frequency points.
    use_soft : bool
        Include soft (acoustic) modes.
    domega : float
        Lorentzian broadening width.
    is_anharmonic : bool
        Use mode-dependent anharmonic linewidths.
    gammas : (T, n_modes) float array, optional
        Anharmonic linewidths (required when *is_anharmonic* is ``True``).
    vectorize : bool
        Use vectorised computation (faster).

    Returns
    -------
    Q : (n_q, 3) float array
        Cartesian wavevectors.
    omega_domain : (nomega,) float array
        Angular frequency grid.
    SQ : dict
        ``{'L': ..., 'T': ...}`` arrays of shape ``(T, n_q, nomega)``.
    """
    if is_anharmonic and gammas is None:
        raise ValueError("Anharmonic linewidths (`gammas`) are required when is_anharmonic=True")

    def _lorentzian(omega: NDArray, gamma: NDArray | float) -> NDArray:
        return gamma / np.pi / (gamma**2 + omega**2)

    SQ: dict[str, NDArray] = {}

    T = positions.shape[0]
    natm = positions.shape[1]

    pos = np.transpose(positions, axes=(0, 2, 1))
    reshaped_eigvec = np.transpose(eigvec.reshape(T, natm, 3, 3 * natm), axes=(0, 2, 3, 1))

    omegas = np.sqrt(np.abs(eigval)) * np.sign(eigval)

    if cutoff is not None:
        mask = omegas.flatten() <= 2 * np.pi * cutoff
        omegas = omegas[:, mask]
        reshaped_eigvec = reshaped_eigvec[:, :, mask, :]
        if is_anharmonic:
            gammas = gammas[:, mask]

    if not use_soft:
        omegas = omegas[:, 3:]
        reshaped_eigvec = reshaped_eigvec[:, :, 3:, :]
        if is_anharmonic:
            gammas = gammas[:, 3:]

    omax = np.max(omegas)
    omega_domain = np.linspace(0, omax, nomega)

    Q, bran_ketQ_sq_L, bran_ketQ_sq_T = compute_braQketn(
        Q_list, reciprocal_cell, pos, reshaped_eigvec
    )

    if vectorize:
        if is_anharmonic:
            delta = _lorentzian(
                np.array([omega_domain[:, np.newaxis] - om[np.newaxis, :] for om in omegas]),
                gammas,
            )
        else:
            delta = _lorentzian(
                np.array([omega_domain[:, np.newaxis] - om[np.newaxis, :] for om in omegas]),
                domega,
            )
        SQ["L"] = oe.contract("tqn,twn->tqw", bran_ketQ_sq_L, delta)
        SQ["T"] = oe.contract("tqn,twn->tqw", bran_ketQ_sq_T, delta)
    else:
        SQ["L"] = np.zeros((T, Q.shape[0], nomega))
        SQ["T"] = np.zeros((T, Q.shape[0], nomega))
        for i, omega in enumerate(omega_domain):
            delta = _lorentzian(omega - omegas, domega)
            SQ["L"][:, :, i] = oe.contract("tqn,tn->tq", bran_ketQ_sq_L, delta)
            SQ["T"][:, :, i] = oe.contract("tqn,tn->tq", bran_ketQ_sq_T, delta)

    return Q, omega_domain, SQ


def make_IR_vector(
    atoms: Atoms,
    polarizations: str | int | tuple[str | int, ...] = ("x", "y"),
    charges: NDArray[np.floating] | None = None,
    charges_dict: dict[str, float] | None = None,
) -> NDArray[np.floating]:
    r"""Build normalised IR ket vector(s).

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure.
    polarizations : str, int, or sequence thereof
        Polarisation directions (``'x'``, ``'y'``, ``'z'`` or ``0``, ``1``, ``2``).
    charges : array, optional
        Effective charges.  Accepted shapes: ``(N,)``, ``(N, 3)``,
        ``(N, 3, P)``, or flat ``3N``.
    charges_dict : dict, optional
        Element-symbol to scalar-charge mapping.  Used when *charges*
        is ``None``.

    Returns
    -------
    (n_vectors, 3*N) float array
        Normalised IR vectors.

    Raises
    ------
    ValueError
        If neither *charges* nor *charges_dict* is provided, or if
        shapes are incompatible.
    KeyError
        If an element is missing from *charges_dict*.
    """
    # Normalise polarizations to a tuple
    if isinstance(polarizations, (str, int)):
        pols: tuple[str | int, ...] = (polarizations,)
    else:
        pols = tuple(polarizations)

    _pol_map = {"x": 0, "y": 1, "z": 2}
    pol_indices: list[int] = []
    for p in pols:
        if isinstance(p, str):
            key = p.lower()
            if key not in _pol_map:
                raise ValueError(f"Unknown polarization '{p}'")
            pol_indices.append(_pol_map[key])
        elif isinstance(p, int):
            if p not in (0, 1, 2):
                raise ValueError("Polarization index must be 0, 1, or 2")
            pol_indices.append(p)
        else:
            raise ValueError("Polarizations must be strings 'x','y','z' or ints 0..2")

    N = len(atoms)
    if N == 0:
        return np.zeros((0, 0), dtype=float)

    masses = np.asarray(atoms.get_masses(), dtype=float)
    if np.any(masses <= 0):
        raise ValueError("All atomic masses must be positive")

    # Build charges_arr with shape (N, 3, P)
    if charges is None:
        if charges_dict is None:
            raise ValueError("Provide either `charges` or `charges_dict`")
        charges_arr = np.zeros((N, 3, 3), dtype=float)
        for i, atom in enumerate(atoms):
            sym = atom.symbol
            if sym not in charges_dict:
                raise KeyError(f"Element {sym} not found in charges_dict")
            charges_arr[i] = float(charges_dict[sym]) * np.eye(3)
    else:
        ch = np.asarray(charges, dtype=float)
        if ch.ndim == 1 and ch.size == N:
            charges_arr = ch[:, None, None] * np.eye(3)[None, :, :]
        elif ch.ndim == 2 and ch.shape == (N, 3):
            charges_arr = ch[:, :, None]
        elif ch.ndim == 3 and ch.shape[0] == N and ch.shape[1] == 3:
            charges_arr = ch
        elif ch.size == 3 * N:
            charges_arr = ch.reshape(N, 3)[:, :, None]
        else:
            raise ValueError(
                "`charges` must have shape (N,), (N,3), (N,3,P) or be compatible with N atoms"
            )

    P = charges_arr.shape[2]
    inv_sqrt_m = 1.0 / np.sqrt(masses)

    phi_N3P = np.einsum("ijp,i->ijp", charges_arr, inv_sqrt_m)

    result = np.zeros((P, N * 3), dtype=float)
    for p in range(P):
        vec = phi_N3P[:, :, p].reshape(N * 3)
        norm = np.linalg.norm(vec)
        if norm != 0.0:
            vec /= norm
        result[p] = vec
    return result
