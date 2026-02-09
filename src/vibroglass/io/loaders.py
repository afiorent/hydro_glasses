"""Load dynamical matrices and atomic structures from various file formats.

Supported formats
-----------------
- ``dynmat.npz`` — SciPy sparse matrix (preferred).
- ``second.npy`` — kaldo-style dense, mass-weighted force constants.
- ``Dyn.form``  — LAMMPS ESKM 3-column flattened dense matrix.

In all cases the directory must also contain ``replicated_atoms.xyz``
readable by ASE.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from ase import Atoms, units
from ase.io import read
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)


def load_dynmat_and_atoms(
    root: str | os.PathLike[str],
    *,
    save_dynmat_npz: bool = True,
    use_tril: bool = False,
) -> tuple[sparse.spmatrix, Atoms]:
    """Load a dynamical matrix and ASE :class:`~ase.Atoms` from *root*.

    The directory must contain ``replicated_atoms.xyz``.  The dynamical
    matrix is loaded from the first file found among ``dynmat.npz``,
    ``second.npy``, and ``Dyn.form`` (in that order).

    Parameters
    ----------
    root : str or path-like
        Directory containing the input files.
    save_dynmat_npz : bool, optional
        When loading from a non-npz source, save the sparse matrix as
        ``dynmat.npz`` for faster future loads.  Default ``True``.
    use_tril : bool, optional
        If ``True``, symmetrise via lower triangle instead of averaging.
        Default ``False``.

    Returns
    -------
    dynmat : scipy.sparse.spmatrix
        Symmetrised dynamical matrix in CSR format.
    atoms : ase.Atoms
        Atomic structure read from ``replicated_atoms.xyz``.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist, ``replicated_atoms.xyz`` is missing, or
        no dynamical-matrix file is found.
    """
    root = str(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"root path `{root}` does not exist or is not a directory")

    atoms_path = os.path.join(root, "replicated_atoms.xyz")
    if not os.path.exists(atoms_path):
        raise FileNotFoundError(f"Atoms file `{atoms_path}` not found")

    dyn_npz_path = os.path.join(root, "dynmat.npz")
    second_npy_path = os.path.join(root, "second.npy")
    dyn_form_path = os.path.join(root, "Dyn.form")

    atoms = read(atoms_path)
    n_atoms = len(atoms)
    mass = atoms.get_masses()

    # --- 1) sparse npz (preferred) ---
    if os.path.exists(dyn_npz_path):
        dynmat_sparse = sparse.load_npz(dyn_npz_path)
        dynmat_sparse = (dynmat_sparse + dynmat_sparse.transpose()) / 2
        return dynmat_sparse, atoms

    # --- 2) kaldo second.npy ---
    if os.path.exists(second_npy_path):
        return _load_from_second_npy(
            second_npy_path,
            dyn_npz_path,
            atoms,
            mass,
            save_dynmat_npz=save_dynmat_npz,
            use_tril=use_tril,
        )

    # --- 3) LAMMPS Dyn.form ---
    if os.path.exists(dyn_form_path):
        return _load_from_dyn_form(
            dyn_form_path,
            dyn_npz_path,
            atoms,
            n_atoms,
            save_dynmat_npz=save_dynmat_npz,
            use_tril=use_tril,
        )

    raise FileNotFoundError(
        f"Neither `{dyn_npz_path}`, `{second_npy_path}`, nor `{dyn_form_path}` found in `{root}`"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _symmetrise(mat: NDArray[np.floating], *, use_tril: bool) -> NDArray[np.floating]:
    if use_tril:
        return np.tril(mat) + np.tril(mat, -1).T
    return (mat + mat.T) / 2


def _load_from_second_npy(
    second_npy_path: str,
    dyn_npz_path: str,
    atoms: Atoms,
    mass: NDArray[np.floating],
    *,
    save_dynmat_npz: bool,
    use_tril: bool,
) -> tuple[sparse.spmatrix, Atoms]:
    dynmat = np.load(second_npy_path)
    # mass-weight
    dynmat = dynmat * (
        1.0 / np.sqrt(mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
    )
    dynmat = dynmat * (
        1.0 / np.sqrt(mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
    )
    n_modes = dynmat.shape[1] * dynmat.shape[2]
    log.info("n modes= %s", n_modes)
    dynmat *= units.mol / (10 * units.J)
    dynmat = dynmat.reshape(n_modes, n_modes)
    dynmat = _symmetrise(dynmat, use_tril=use_tril)
    dynmat_sparse = csr_matrix(dynmat)
    if save_dynmat_npz:
        sparse.save_npz(dyn_npz_path, dynmat_sparse)
    return dynmat_sparse, atoms


def _load_from_dyn_form(
    dyn_form_path: str,
    dyn_npz_path: str,
    atoms: Atoms,
    n_atoms: int,
    *,
    save_dynmat_npz: bool,
    use_tril: bool,
) -> tuple[sparse.spmatrix, Atoms]:
    log.info("ESKM LAMMPS units are assumed")
    try:
        df = pd.read_csv(
            dyn_form_path,
            sep=r"\s+",
            header=None,
            usecols=[0, 1, 2],
            names=["c0", "c1", "c2"],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read `Dyn.form` at `{dyn_form_path}`: {exc}") from exc

    flat = df.to_numpy().ravel()
    n_modes = 3 * n_atoms
    if flat.size != n_modes * n_modes:
        raise ValueError(
            f"Unexpected data size in `Dyn.form` (got {flat.size}, expected {n_modes * n_modes})."
        )

    mat = flat.reshape(n_modes, n_modes)
    mat = _symmetrise(mat, use_tril=use_tril)
    dynmat_sparse = csr_matrix(mat)
    if save_dynmat_npz:
        sparse.save_npz(dyn_npz_path, dynmat_sparse)
    return dynmat_sparse, atoms
