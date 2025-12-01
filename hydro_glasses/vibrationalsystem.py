import os
import logging
from typing import Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from ase.io import read
from ase import units
import pandas as pd


log = logging.getLogger(__name__)

class VibrationalSystem:
    """
    Container for a dynamical matrix and atoms.
    Can be constructed either from `dynmat` + `atoms` or from a `root` directory.
    """
    def __init__(self, dynmat: Optional[sparse.spmatrix] = None, atoms=None, root: Optional[str] = None):
        self.dynmat = None
        self.atoms = None
        self.root = None

        if (dynmat is not None and atoms is not None) and root is None:
            self.dynmat = dynmat
            self.atoms = atoms
        elif root is not None:
            self.root = root
            # perform initialization immediately
            self.initialize_from_root()
        else:
            raise ValueError("Provide either both `dynmat` and `atoms`, or provide `root` (directory path)")

    @classmethod
    def from_root(cls, root: str, **loader_kwargs):
        """
        Alternate constructor using `root` directory.
        Loads data with `load_dynmat_and_atoms` and returns an instance constructed
        from the loaded `dynmat` and `atoms` (avoids double-initialization).
        """
        if not os.path.isdir(root):
            raise FileNotFoundError(f"root path `{root}` does not exist or is not a directory")
        dynmat_sparse, atoms = load_dynmat_and_atoms(root, **loader_kwargs)
        return cls(dynmat=dynmat_sparse, atoms=atoms)

    def initialize_from_root(self):
        """Load data from `self.root`. Raises FileNotFoundError if root invalid."""
        if not self.root:
            raise ValueError("`root` is not set on this VibrationalSystem instance")
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"root path `{self.root}` does not exist or is not a directory")
        dynmat_sparse, atoms = load_dynmat_and_atoms(self.root)
        self.dynmat = dynmat_sparse
        self.atoms = atoms



def load_dynmat_and_atoms(root: str, save_dynmat_npz: bool = True, use_tril: bool = False) -> Tuple[sparse.spmatrix, object]:
    """
    Load dynamical matrix and ASE Atoms from a directory `root`.
    Expects files:
      - `replicated_atoms.xyz`
      - `dynmat.npz` or `second.npy`
    If those are missing, will try to parse `Dyn.form` using the same approach
    as `lammps_dynmat2sparse` (three whitespace-separated columns -> flattened dense).
    Prints that ESKM LAMMPS units are assumed when parsing `Dyn.form`.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"root path `{root}` does not exist or is not a directory")

    atoms_path = os.path.join(root, "replicated_atoms.xyz")
    if not os.path.exists(atoms_path):
        raise FileNotFoundError(f"Atoms file `{atoms_path}` not found")

    dyn_npz_path = os.path.join(root, "dynmat.npz")
    second_npy_path = os.path.join(root, "second.npy")
    dyn_form_path = os.path.join(root, "Dyn.form")

    atoms = read(atoms_path)
    # fallback for ASE API differences
    n_atoms = atoms.get_global_number_of_atoms() if hasattr(atoms, "get_global_number_of_atoms") else len(atoms)
    mass = atoms.get_masses()

    # 1) existing sparse npz
    if os.path.exists(dyn_npz_path):
        dynmat_sparse = sparse.load_npz(dyn_npz_path)
        dynmat_sparse = (dynmat_sparse + dynmat_sparse.transpose()) / 2
        return dynmat_sparse, atoms

    # 2) second.npy path (existing logic)
    if os.path.exists(second_npy_path):
        dynmat = np.load(second_npy_path)
        dynmat = dynmat * (1.0 / np.sqrt(mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]))
        dynmat = dynmat * (1.0 / np.sqrt(mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]))
        n_modes = dynmat.shape[1] * dynmat.shape[2]
        log.info("n modes= %s", n_modes)
        dynmat *= units.mol / (10 * units.J)
        dynmat = dynmat.reshape([n_modes, n_modes])
        if use_tril:
            dynmat = np.tril(dynmat) + np.tril(dynmat, -1).T
        else:
            dynmat = (dynmat + dynmat.T) / 2
        dynmat_sparse = sparse.csr_matrix(dynmat)
        if save_dynmat_npz:
            sparse.save_npz(dyn_npz_path, dynmat_sparse)
        return dynmat_sparse, atoms

    # 3) new: parse Dyn.form using the lammps_dynmat2sparse approach
    if os.path.exists(dyn_form_path):
        print("ESKM LAMMPS units are assumed")
        try:
            # read exactly three whitespace-separated columns (matches lammps_dynmat2sparse)
            df = pd.read_csv(dyn_form_path, sep=r'\s+', header=None, usecols=[0, 1, 2], names=['c0', 'c1', 'c2'])
        except Exception as exc:
            raise RuntimeError(f"Failed to read `Dyn.form` at `{dyn_form_path}`: {exc}")

        arr = df.to_numpy()
        flat = arr.ravel()
        n_modes = 3 * n_atoms

        if flat.size != n_modes * n_modes:
            raise ValueError(
                f"Unexpected data size in `Dyn.form` (got {flat.size}, expected {n_modes*n_modes}). "
                f"Ensure `Dyn.form` follows the `lammps_dynmat2sparse` three-column flattened dense format."
            )

        mat = flat.reshape((n_modes, n_modes))
        # symmetrize unless explicitly using tril
        if use_tril:
            mat = np.tril(mat) + np.tril(mat, -1).T
        else:
            mat = (mat + mat.T) / 2

        dynmat_sparse = csr_matrix(mat)
        if save_dynmat_npz:
            sparse.save_npz(dyn_npz_path, dynmat_sparse)
        return dynmat_sparse, atoms

    raise FileNotFoundError(f"Neither `{dyn_npz_path}`, `{second_npy_path}`, nor `{dyn_form_path}` found in `{root}`")