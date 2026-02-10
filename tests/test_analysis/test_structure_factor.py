"""Tests for vibroglass.analysis.structure_factor."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from numpy.testing import assert_allclose

from vibroglass.analysis.structure_factor import (
    compute_phi_Q,
    inv_participation_ratio,
    make_IR_vector,
)


def _make_small_cubic_atoms(n: int = 4) -> Atoms:
    """Create a small cubic cell with *n* atoms on a grid."""
    positions = np.array(
        [[i, j, k] for i in range(n) for j in range(1) for k in range(1)], dtype=float
    )
    return Atoms(
        symbols="Si" * len(positions),
        positions=positions,
        cell=[n, 1, 1],
        pbc=True,
    )


class TestComputePhiQ:
    def test_output_shapes(self):
        atoms = _make_small_cubic_atoms(4)
        pos = atoms.get_positions().T[np.newaxis, :, :]  # (1, 3, N)
        rcell = atoms.cell.reciprocal().array
        Q_list = np.array([[1, 0, 0], [0, 1, 0]])
        phi_L, phi_T = compute_phi_Q(Q_list, rcell, pos)
        n_atoms = len(atoms)
        n_q = len(Q_list)
        # Shape: (n_atoms, 3, n_q) for longitudinal
        assert phi_L.shape == (n_atoms, 3, n_q)
        assert phi_T.shape == (n_atoms, 3, n_q)

    def test_longitudinal_parallel_to_q(self):
        """Longitudinal polarisation vectors should be parallel to Q."""
        atoms = _make_small_cubic_atoms(4)
        pos = atoms.get_positions().T[np.newaxis, :, :]
        rcell = atoms.cell.reciprocal().array
        Q_list = np.array([[1, 0, 0]])
        phi_L, _phi_T = compute_phi_Q(Q_list, rcell, pos)
        # For a single Q along x, the polarisation direction (axis=1) should be ~ (1,0,0)
        # Check that each atom's vector is proportional to Q direction
        Q_cart = 2 * np.pi * rcell @ Q_list[0]
        Q_hat = Q_cart / np.linalg.norm(Q_cart)
        for i in range(len(atoms)):
            vec = phi_L[i, :, 0]
            if np.linalg.norm(vec) > 1e-10:
                vec_hat = vec / np.abs(np.linalg.norm(vec))
                # Should be parallel (cross product ~ 0)
                cross = np.linalg.norm(np.cross(np.real(vec_hat), Q_hat))
                assert cross < 1e-6


class TestInvParticipationRatio:
    def test_localised_mode(self):
        """A mode localised on one site should have IPR = 1."""
        n = 10
        eigvec = np.zeros(n)
        eigvec[0] = 1.0
        ipr = inv_participation_ratio(eigvec[np.newaxis, :])
        assert_allclose(ipr[0], 1.0)

    def test_delocalised_mode(self):
        """A fully delocalised mode should have IPR = 1/N."""
        n = 100
        eigvec = np.ones(n) / np.sqrt(n)
        ipr = inv_participation_ratio(eigvec[np.newaxis, :])
        assert_allclose(ipr[0], 1.0 / n, atol=1e-10)


class TestMakeIRVector:
    def test_output_shape(self):
        atoms = Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5])
        charges_dict = {"Si": 3.2, "O": -1.6}
        result = make_IR_vector(atoms, polarizations=("x", "y"), charges_dict=charges_dict)
        n_atoms = len(atoms)
        # charges_dict produces (N,3,3) charges -> 3 vectors
        assert result.shape[1] == 3 * n_atoms

    def test_normalisation(self):
        atoms = Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5])
        charges_dict = {"Si": 3.2, "O": -1.6}
        result = make_IR_vector(atoms, charges_dict=charges_dict)
        for row in result:
            if np.linalg.norm(row) > 0:
                assert_allclose(np.linalg.norm(row), 1.0, atol=1e-12)

    def test_missing_element_raises(self):
        atoms = Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5])
        with pytest.raises(KeyError, match="O"):
            make_IR_vector(atoms, charges_dict={"Si": 1.0})

    def test_scalar_charges(self):
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5])
        charges = np.array([1.0, 2.0])
        result = make_IR_vector(atoms, charges=charges)
        assert result.shape[1] == 6  # 2 atoms * 3 directions

    def test_no_charges_raises(self):
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5])
        with pytest.raises(ValueError, match="charges"):
            make_IR_vector(atoms)
