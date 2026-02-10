"""Tests for vibroglass.io.kaldo."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_allclose

from vibroglass.io.kaldo import Ctt_correlator, v2_matrix


def _make_mock_phonons(n_atoms: int = 4, n_k: int = 1, *, seed: int = 0):
    """Build a minimal mock ``kaldo.Phonons``-like object."""
    rng = np.random.default_rng(seed)
    n_modes = 3 * n_atoms

    phonons = MagicMock()
    phonons.n_atoms = n_atoms
    phonons.n_modes = n_modes
    phonons.n_k_points = n_k

    # Eigenvalues / frequencies / omega - 1D arrays for Gamma-point
    freqs = np.sort(rng.uniform(0.5, 10.0, size=n_modes))
    freqs[:3] = 0.0  # acoustic modes
    phonons.frequency = freqs
    phonons.omega = 2 * np.pi * freqs
    phonons.eigenvalues = freqs**2

    # Eigenvectors - kaldo Gamma-point shape: (n_modes, n_atoms, 3)
    eigvec = rng.standard_normal((n_modes, n_atoms, 3))
    phonons.eigenvectors = SimpleNamespace(real=eigvec)

    # Physical modes mask (1D, size n_modes)
    pm = np.ones(n_modes, dtype=bool)
    pm[:3] = False
    phonons.physical_mode = pm

    # Populations (1D)
    phonons.population = rng.uniform(0.01, 1.0, size=n_modes)

    # Bandwidths (1D)
    phonons.bandwidth = rng.uniform(0.01, 0.5, size=n_modes)

    # Atoms-like
    atoms = MagicMock()
    atoms.get_positions.return_value = rng.uniform(0, 5, size=(n_atoms, 3))
    atoms.get_masses.return_value = np.full(n_atoms, 28.0)
    cell = MagicMock()
    cell.reciprocal.return_value = np.eye(3) * 0.2
    atoms.cell = cell
    phonons.atoms = atoms

    # hbar constant used by Ctt_correlator
    phonons.hbar = 1.0545718e-22  # J*ps

    # Force constants / reciprocal grid (for v2_matrix)
    phonons.forceconstants = MagicMock()
    phonons.forceconstants.second = MagicMock()
    phonons.forceconstants.distance_threshold = 5.0
    phonons.folder = "/tmp"
    phonons.storage = "numpy"
    phonons.is_nw = False
    phonons.is_unfolding = False

    grid = MagicMock()
    grid.unitary_grid.return_value = rng.uniform(0, 1, size=(n_k, 3))
    phonons._reciprocal_grid = grid

    return phonons


class TestV2Matrix:
    def test_returns_omegas_and_matrix(self):
        """v2_matrix returns (omegas, v2_omega) with correct shapes."""
        n_atoms = 4
        n_modes = 3 * n_atoms
        phonons = _make_mock_phonons(n_atoms)
        nomega = 50
        cutoff_n = n_modes - 3  # all physical modes

        rng = np.random.default_rng(42)
        sij = rng.standard_normal((n_modes, n_modes)) + 0j

        mock_hwq = MagicMock()
        mock_hwq._sij_x = sij
        mock_hwq._sij_y = sij
        mock_hwq._sij_z = sij

        with patch("vibroglass.io.kaldo.HarmonicWithQ", return_value=mock_hwq):
            omegas, v2_omega = v2_matrix(
                phonons, cutoff_n=cutoff_n, nomega=nomega, delta_width=0.5
            )

        assert omegas.shape == (nomega,)
        assert v2_omega.shape == (nomega, nomega)
        assert np.all(v2_omega >= -1e-10)

    def test_symmetric_output(self):
        """v2_omega matrix should be symmetric."""
        n_atoms = 4
        n_modes = 3 * n_atoms
        phonons = _make_mock_phonons(n_atoms)
        nomega = 30
        cutoff_n = n_modes - 3

        rng = np.random.default_rng(7)
        sij = rng.standard_normal((n_modes, n_modes)) + 0j

        mock_hwq = MagicMock()
        mock_hwq._sij_x = sij
        mock_hwq._sij_y = sij
        mock_hwq._sij_z = sij

        with patch("vibroglass.io.kaldo.HarmonicWithQ", return_value=mock_hwq):
            _omegas, v2_omega = v2_matrix(
                phonons, cutoff_n=cutoff_n, nomega=nomega, delta_width=0.5
            )

        assert_allclose(v2_omega, v2_omega.T, atol=1e-12)


class TestCttCorrelator:
    def test_returns_Q_omega_C(self):
        """Ctt_correlator returns (Q, omega_domain, C) with L and T branches."""
        phonons = _make_mock_phonons(n_atoms=4)
        Q_list = np.array([[1, 0, 0], [0, 1, 0]])
        nomega = 50

        Q, omega_domain, C = Ctt_correlator(phonons, Q_list, nomega=nomega)

        assert Q.shape == (2, 3)
        assert omega_domain.shape == (nomega,)
        assert "L" in C
        assert "T" in C
        assert C["L"].shape[0] == 2  # n_q
        assert C["T"].shape[0] == 2
        assert C["L"].shape[1] == nomega
        assert C["T"].shape[1] == nomega

    def test_debug_mode(self):
        """Debug mode should return correlator without populations."""
        phonons = _make_mock_phonons(n_atoms=4)
        Q_list = np.array([[1, 0, 0]])
        nomega = 30

        _Q, _omega_domain, C = Ctt_correlator(phonons, Q_list, nomega=nomega, debug=True)

        assert C["L"].shape == (1, nomega)

    def test_normalize(self):
        """Normalized correlator should integrate to 1 for each Q."""
        phonons = _make_mock_phonons(n_atoms=4)
        Q_list = np.array([[1, 0, 0]])
        nomega = 200

        _Q, omega_domain, C = Ctt_correlator(
            phonons, Q_list, nomega=nomega, debug=True, normalize=True
        )

        integral = np.trapezoid(C["L"][0], omega_domain)
        assert_allclose(integral, 1.0, atol=0.05)

    def test_return_gamma(self):
        """return_gamma=True returns extra diagnostic arrays."""
        phonons = _make_mock_phonons(n_atoms=4)
        Q_list = np.array([[1, 0, 0]])

        result = Ctt_correlator(phonons, Q_list, nomega=30, return_gamma=True)

        # Should return 8-element tuple
        assert len(result) == 8
