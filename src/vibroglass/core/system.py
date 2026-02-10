"""Container for a dynamical matrix and atomic structure."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from vibroglass.io.loaders import load_dynmat_and_atoms

if TYPE_CHECKING:
    from ase import Atoms
    from scipy.sparse import spmatrix


class VibrationalSystem:
    """Dynamical matrix plus atomic structure.

    Can be constructed either from explicit ``dynmat`` + ``atoms`` or
    from a *root* directory that contains the required files (see
    :func:`~vibroglass.io.loaders.load_dynmat_and_atoms`).

    Parameters
    ----------
    dynmat : scipy.sparse.spmatrix, optional
        Sparse dynamical matrix.
    atoms : ase.Atoms, optional
        Atomic structure.
    root : str or path-like, optional
        Directory from which to load data.

    Raises
    ------
    ValueError
        If neither ``(dynmat, atoms)`` nor ``root`` is provided.
    FileNotFoundError
        If *root* does not exist or is not a directory.
    """

    dynmat: spmatrix | None
    atoms: Atoms | None
    root: str | None

    def __init__(
        self,
        dynmat: spmatrix | None = None,
        atoms: Atoms | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.dynmat = None
        self.atoms = None
        self.root = None

        if dynmat is not None and atoms is not None and root is None:
            self.dynmat = dynmat
            self.atoms = atoms
        elif root is not None:
            self.root = str(root)
            self._initialize_from_root()
        else:
            raise ValueError(
                "Provide either both `dynmat` and `atoms`, or provide `root` (directory path)"
            )

    @classmethod
    def from_root(cls, root: str | os.PathLike[str], **loader_kwargs) -> VibrationalSystem:
        """Construct from a directory path.

        Parameters
        ----------
        root : str or path-like
            Directory containing input files.
        **loader_kwargs
            Forwarded to :func:`~vibroglass.io.loaders.load_dynmat_and_atoms`.

        Returns
        -------
        VibrationalSystem
        """
        root_str = str(root)
        if not os.path.isdir(root_str):
            raise FileNotFoundError(f"root path `{root_str}` does not exist or is not a directory")
        dynmat_sparse, atoms = load_dynmat_and_atoms(root_str, **loader_kwargs)
        return cls(dynmat=dynmat_sparse, atoms=atoms)

    def _initialize_from_root(self) -> None:
        if not self.root or not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"root path `{self.root}` does not exist or is not a directory"
            )
        dynmat_sparse, atoms = load_dynmat_and_atoms(self.root)
        self.dynmat = dynmat_sparse
        self.atoms = atoms
