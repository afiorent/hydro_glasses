"""Tests for vibroglass.io.loaders."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from numpy.testing import assert_allclose
from scipy import sparse

from vibroglass.io.loaders import load_dynmat_and_atoms


@pytest.fixture()
def small_system(tmp_path):
    """Create a minimal 2-atom system with known dynamical matrix."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=[5, 5, 5], pbc=True)
    write(str(tmp_path / "replicated_atoms.xyz"), atoms)
    n_modes = 3 * len(atoms)
    # symmetric positive-definite matrix
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n_modes, n_modes))
    mat = mat @ mat.T  # make symmetric PD
    return tmp_path, atoms, mat


class TestLoadFromNpz:
    """Test loading from sparse dynmat.npz."""

    def test_loads_sparse_npz(self, small_system):
        root, _atoms, mat = small_system
        dynmat_sparse = sparse.csr_matrix(mat)
        sparse.save_npz(str(root / "dynmat.npz"), dynmat_sparse)

        result_dynmat, result_atoms = load_dynmat_and_atoms(str(root))

        assert sparse.issparse(result_dynmat)
        assert_allclose(result_dynmat.toarray(), mat, atol=1e-12)
        assert len(result_atoms) == 2

    def test_symmetrizes_npz(self, small_system):
        root, _atoms, mat = small_system
        # make a non-symmetric sparse matrix
        upper = np.triu(mat)
        dynmat_sparse = sparse.csr_matrix(upper)
        sparse.save_npz(str(root / "dynmat.npz"), dynmat_sparse)

        result_dynmat, _ = load_dynmat_and_atoms(str(root))
        result_dense = result_dynmat.toarray()

        assert_allclose(result_dense, result_dense.T, atol=1e-12)


class TestLoadFromDynForm:
    """Test loading from LAMMPS Dyn.form file."""

    def test_loads_dyn_form(self, small_system):
        root, _atoms, mat = small_system
        # Write 3-column format
        flat = mat.ravel()
        with open(root / "Dyn.form", "w") as f:
            for i in range(0, len(flat), 3):
                f.write(f"{flat[i]:.15e} {flat[i + 1]:.15e} {flat[i + 2]:.15e}\n")

        result_dynmat, _result_atoms = load_dynmat_and_atoms(str(root), save_dynmat_npz=False)

        assert sparse.issparse(result_dynmat)
        expected = (mat + mat.T) / 2
        assert_allclose(result_dynmat.toarray(), expected, atol=1e-12)

    def test_auto_saves_npz(self, small_system):
        root, _atoms, mat = small_system
        flat = mat.ravel()
        with open(root / "Dyn.form", "w") as f:
            for i in range(0, len(flat), 3):
                f.write(f"{flat[i]:.15e} {flat[i + 1]:.15e} {flat[i + 2]:.15e}\n")

        npz_path = root / "dynmat.npz"
        assert not npz_path.exists()

        load_dynmat_and_atoms(str(root), save_dynmat_npz=True)

        assert npz_path.exists()


class TestLoadFromSecondNpy:
    """Test loading from kaldo-style second.npy."""

    def test_loads_second_npy(self, small_system):
        root, atoms, _mat = small_system
        n_atoms = len(atoms)
        # second.npy shape: (1, n_atoms, 3, 1, n_atoms, 3) for kaldo format
        rng = np.random.default_rng(99)
        second = rng.standard_normal((1, n_atoms, 3, 1, n_atoms, 3))
        # make symmetric
        second_flat = second.reshape(n_atoms * 3, n_atoms * 3)
        second_flat = (second_flat + second_flat.T) / 2
        second = second_flat.reshape(1, n_atoms, 3, 1, n_atoms, 3)
        np.save(str(root / "second.npy"), second)

        result_dynmat, _result_atoms = load_dynmat_and_atoms(str(root), save_dynmat_npz=False)

        assert sparse.issparse(result_dynmat)
        assert result_dynmat.shape == (n_atoms * 3, n_atoms * 3)
        # verify symmetry
        dense = result_dynmat.toarray()
        assert_allclose(dense, dense.T, atol=1e-12)


class TestErrors:
    """Test error handling."""

    def test_missing_root(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_dynmat_and_atoms(str(tmp_path / "nonexistent"))

    def test_missing_atoms_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"replicated_atoms\.xyz"):
            load_dynmat_and_atoms(str(tmp_path))

    def test_no_dynmat_files(self, small_system):
        root, _, _ = small_system
        # root has atoms but no dynmat files
        with pytest.raises(FileNotFoundError, match="Neither"):
            load_dynmat_and_atoms(str(root))
