"""Tests for vibroglass.analysis.spectra."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from scipy import sparse

from vibroglass.analysis.spectra import LanczosOptions, VibrationalSpectra


@pytest.fixture()
def small_vibrational_system(tmp_path):
    """Create a minimal system on disk for VibrationalSpectra tests."""
    n_atoms = 4
    atoms = Atoms(
        "Si" * n_atoms,
        positions=[[i, 0, 0] for i in range(n_atoms)],
        cell=[n_atoms, 5, 5],
        pbc=True,
    )
    write(str(tmp_path / "replicated_atoms.xyz"), atoms)
    n_modes = 3 * n_atoms
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n_modes, n_modes))
    mat = mat @ mat.T
    dynmat_sparse = sparse.csr_matrix(mat)
    sparse.save_npz(str(tmp_path / "dynmat.npz"), dynmat_sparse)
    return tmp_path, atoms, mat


class TestLanczosOptions:
    def test_defaults(self):
        opts = LanczosOptions()
        assert opts.eta == 1.0
        assert opts.hl_steps == 200
        assert opts.use_ortho is False
        assert len(opts.omega_array) == 20000

    def test_from_mapping(self):
        opts = LanczosOptions.from_mapping({"eta": 2.0, "hl_steps": 50})
        assert opts.eta == 2.0
        assert opts.hl_steps == 50

    def test_invalid_hl_steps_raises(self):
        with pytest.raises(ValueError, match="hl_steps"):
            LanczosOptions(hl_steps=-1)

    def test_invalid_eta_raises(self):
        with pytest.raises(ValueError, match="eta"):
            LanczosOptions(eta=-1.0)

    def test_to_dict(self):
        opts = LanczosOptions(eta=3.0, hl_steps=100)
        d = opts.to_dict()
        assert d["eta"] == 3.0
        assert d["hl_steps"] == 100


class TestVibrationalSpectra:
    def test_spectrum_for_ket(self, small_vibrational_system):
        root, _atoms, mat = small_vibrational_system
        n_modes = mat.shape[0]
        omega = np.linspace(0.1, 20, 100)
        opts = LanczosOptions(eta=1.0, omega_array=omega, hl_steps=n_modes)

        vs = VibrationalSpectra(str(root), options=opts)
        ket = np.random.default_rng(0).standard_normal(n_modes)
        result = vs.spectrum_for_ket(ket)

        assert result.shape == omega.shape
        assert np.all(result >= 0)

    def test_spectrum_for_ket_list_serial(self, small_vibrational_system):
        root, _atoms, _mat = small_vibrational_system
        n_modes = _mat.shape[0]
        omega = np.linspace(0.1, 20, 50)
        opts = LanczosOptions(eta=1.0, omega_array=omega, hl_steps=8)

        vs = VibrationalSpectra(str(root), options=opts)
        rng = np.random.default_rng(1)
        kets = [rng.standard_normal(n_modes) for _ in range(3)]
        results = vs.spectrum_for_ket_list(kets, parallel=False)

        assert len(results) == 3
        # Worker returns (spectrum, alpha, beta) tuples
        for r in results:
            spec, alpha, _beta = r
            assert spec.shape == omega.shape
            assert alpha.shape == (8,)

    def test_compute_stochastic_vdos(self, small_vibrational_system):
        root, _atoms, _mat = small_vibrational_system
        omega = np.linspace(0.1, 20, 50)
        opts = LanczosOptions(eta=1.0, omega_array=omega, hl_steps=8)

        vs = VibrationalSpectra(str(root), options=opts)
        result = vs.compute_stochastic_vdos(nstoc=2, parallel=False, save=False)

        assert len(result) == 2
        for i in range(2):
            assert "S" in result[i]
            assert "alpha" in result[i]
            assert "beta" in result[i]
