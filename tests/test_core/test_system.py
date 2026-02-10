"""Tests for vibroglass.core.system."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from numpy.testing import assert_allclose
from scipy import sparse

from vibroglass.core.system import VibrationalSystem


@pytest.fixture()
def system_on_disk(tmp_path):
    """Create a minimal system with dynmat.npz + replicated_atoms.xyz."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=[5, 5, 5], pbc=True)
    write(str(tmp_path / "replicated_atoms.xyz"), atoms)
    n_modes = 3 * len(atoms)
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n_modes, n_modes))
    mat = (mat + mat.T) / 2
    dynmat_sparse = sparse.csr_matrix(mat)
    sparse.save_npz(str(tmp_path / "dynmat.npz"), dynmat_sparse)
    return tmp_path, atoms, mat


class TestVibrationalSystem:
    def test_construct_from_dynmat_and_atoms(self, system_on_disk):
        root, atoms, _mat = system_on_disk
        dynmat = sparse.load_npz(str(root / "dynmat.npz"))
        vs = VibrationalSystem(dynmat=dynmat, atoms=atoms)
        assert vs.dynmat is not None
        assert vs.atoms is not None
        assert vs.root is None

    def test_construct_from_root(self, system_on_disk):
        root, _atoms, mat = system_on_disk
        vs = VibrationalSystem(root=str(root))
        assert vs.dynmat is not None
        assert len(vs.atoms) == 2
        assert_allclose(vs.dynmat.toarray(), mat, atol=1e-12)

    def test_from_root_classmethod(self, system_on_disk):
        root, _atoms, mat = system_on_disk
        vs = VibrationalSystem.from_root(str(root))
        assert vs.dynmat is not None
        assert_allclose(vs.dynmat.toarray(), mat, atol=1e-12)

    def test_invalid_args_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            VibrationalSystem()

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VibrationalSystem(root=str(tmp_path / "nonexistent"))

    def test_from_root_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VibrationalSystem.from_root(str(tmp_path / "nonexistent"))
