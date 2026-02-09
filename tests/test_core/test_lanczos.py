"""Tests for vibroglass.core.lanczos."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse

from vibroglass.core.lanczos import (
    compute_b2,
    continued_fraction,
    lanczos_cheap,
    recompute_spectrum,
    spectrum,
)


def _make_small_dynmat(n: int = 6, seed: int = 42) -> np.ndarray:
    """Create a small symmetric positive-definite matrix."""
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, n))
    return mat @ mat.T


class TestContinuedFraction:
    def test_single_element(self):
        result = continued_fraction([3.0], [0.0])
        assert_allclose(result, 3.0)

    def test_two_elements(self):
        # a0 + b0 / (a1 + b1)
        # = 1 + 2 / (3 + 4) = 1 + 2/7
        result = continued_fraction([1.0, 3.0], [2.0, 4.0])
        assert_allclose(result, 1.0 + 2.0 / 7.0)

    def test_empty(self):
        result = continued_fraction([], [])
        assert result == 0


class TestComputeB2:
    def test_shape_and_values(self):
        beta = np.array([2.0, 3.0, 4.0])
        b2 = compute_b2(beta)
        assert b2.shape == (4,)
        assert_allclose(b2[0], 1.0)
        assert_allclose(b2[1:], -(beta**2))


class TestLanczosCheap:
    def test_tridiagonal_coefficients(self):
        mat = _make_small_dynmat(10)
        v = np.random.default_rng(0).standard_normal(10)
        k = 8
        alpha, beta = lanczos_cheap(mat, v, k)
        assert alpha.shape == (k,)
        assert beta.shape == (k,)

    def test_last2_option(self):
        mat = _make_small_dynmat(10)
        v = np.random.default_rng(0).standard_normal(10)
        result = lanczos_cheap(mat, v, 5, last2=True)
        assert len(result) == 4
        _alpha, _beta, v_last, v_prev = result
        assert v_last.shape == (10,)
        assert v_prev.shape == (10,)


class TestSpectrum:
    def test_spectrum_vs_direct(self):
        """Compare Lanczos spectrum to direct eigenvalue computation."""
        n = 20
        mat = _make_small_dynmat(n, seed=7)
        rng = np.random.default_rng(1)
        v = rng.standard_normal(n)

        omega = np.linspace(0.1, 15, 500)
        eta = 0.5

        # Lanczos spectrum (use all n steps for exact result)
        spec_lanczos = spectrum(mat, v, k=n, omega_array=omega, eta=eta)

        # Direct: compute via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(mat)
        v_normed = v / np.linalg.norm(v)
        projections = eigvecs.T @ v_normed
        spec_direct = np.zeros_like(omega)
        for i in range(n):
            z = omega + 1j * eta
            spec_direct += np.abs(projections[i]) ** 2 * np.abs(np.imag(1.0 / (z**2 - eigvals[i])))

        assert_allclose(spec_lanczos, spec_direct, atol=1e-4)

    def test_return_chain(self):
        mat = _make_small_dynmat(10)
        v = np.random.default_rng(0).standard_normal(10)
        omega = np.linspace(0.1, 10, 50)
        result = spectrum(mat, v, k=8, omega_array=omega, eta=1.0, return_chain=True)
        assert len(result) == 3
        spec, alpha, _beta = result
        assert spec.shape == omega.shape
        assert alpha.shape == (8,)

    def test_sparse_matrix(self):
        """Spectrum should work with sparse matrices too."""
        mat = _make_small_dynmat(10)
        mat_sparse = sparse.csr_matrix(mat)
        v = np.random.default_rng(0).standard_normal(10)
        omega = np.linspace(0.1, 10, 50)

        spec_dense = spectrum(mat, v, k=8, omega_array=omega, eta=1.0)
        spec_sparse = spectrum(mat_sparse, v, k=8, omega_array=omega, eta=1.0)

        assert_allclose(spec_dense, spec_sparse, atol=1e-10)


class TestRecomputeSpectrum:
    def test_roundtrip(self):
        """recompute_spectrum with same z2 should match original spectrum."""
        mat = _make_small_dynmat(10)
        v = np.random.default_rng(0).standard_normal(10)
        omega = np.linspace(0.1, 10, 50)
        eta = 1.0

        spec_orig, alpha, beta = spectrum(
            mat, v, k=8, omega_array=omega, eta=eta, return_chain=True
        )
        z2 = (omega + 1j * eta) ** 2
        spec_recomp = recompute_spectrum(alpha, beta, z2)

        assert_allclose(spec_recomp, spec_orig, atol=1e-10)
