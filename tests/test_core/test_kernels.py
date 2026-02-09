"""Tests for vibroglass.core.kernels."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from vibroglass.core.kernels import (
    compute_kernel,
    compute_vdos,
    dho,
    gauss,
    lorentzian,
)


class TestLorentzian:
    def test_peak_value(self):
        # At x=0, lorentzian = 1/(pi*eta)
        eta = 2.0
        assert_allclose(lorentzian(0.0, eta), 1.0 / (np.pi * eta))

    def test_normalisation(self):
        eta = 1.5
        x = np.linspace(-1000, 1000, 500_000)
        integral = np.trapezoid(lorentzian(x, eta), x)
        assert_allclose(integral, 1.0, atol=1e-3)


class TestGauss:
    def test_peak_value(self):
        sigma = 2.0
        expected = 1.0 / (sigma * np.sqrt(2 * np.pi))
        assert_allclose(gauss(0.0, sigma), expected)

    def test_normalisation(self):
        sigma = 1.0
        x = np.linspace(-10, 10, 100_000)
        integral = np.trapezoid(gauss(x, sigma), x)
        assert_allclose(integral, 1.0, atol=1e-6)


class TestDHO:
    def test_positive(self):
        w = np.linspace(0.1, 10, 100)
        result = dho(w, w0=5.0, sigma=1.0, norm=1.0)
        assert np.all(result >= 0)


class TestComputeKernel:
    def test_single_delta(self):
        omega = np.linspace(-5, 15, 10_000)
        freqs = np.array([5.0])
        vals = np.array([1.0])
        result = compute_kernel(freqs, vals, omega, sigma=0.5, kernel="lorentz", normalize=False)
        # Peak should be near omega=5
        peak_idx = np.argmax(result)
        assert_allclose(omega[peak_idx], 5.0, atol=omega[1] - omega[0])

    def test_normalised_integral(self):
        omega = np.linspace(-10, 30, 50_000)
        freqs = np.array([5.0, 10.0])
        vals = np.array([1.0, 2.0])
        result = compute_kernel(freqs, vals, omega, sigma=1.0, normalize=True)
        integral = np.trapezoid(result, omega)
        assert_allclose(integral, 1.0, atol=1e-4)

    def test_gauss_kernel(self):
        omega = np.linspace(-10, 30, 50_000)
        freqs = np.array([5.0])
        vals = np.array([1.0])
        result = compute_kernel(freqs, vals, omega, sigma=1.0, kernel="gauss", normalize=False)
        peak_idx = np.argmax(result)
        assert_allclose(omega[peak_idx], 5.0, atol=omega[1] - omega[0])

    def test_unknown_kernel_raises(self):
        omega = np.linspace(0, 5, 10)
        with pytest.raises(ValueError, match="Unknown kernel"):
            compute_kernel(np.array([1.0]), np.array([1.0]), omega, 1.0, kernel="bad")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_kernel(np.array([1.0, 2.0]), np.array([1.0]), np.linspace(0, 5, 10), 1.0)


class TestComputeVdos:
    def test_normalised(self):
        freqs = np.array([3.0, 5.0, 7.0])
        omega = np.linspace(0, 15, 50_000)
        vdos = compute_vdos(freqs, omega, sigma=0.5)
        integral = np.trapezoid(vdos, omega)
        assert_allclose(integral, 1.0, atol=1e-4)

    def test_three_peaks(self):
        freqs = np.array([3.0, 7.0, 11.0])
        omega = np.linspace(0, 15, 50_000)
        vdos = compute_vdos(freqs, omega, sigma=0.2)
        # Should have local maxima near each frequency
        for f in freqs:
            idx = np.argmin(np.abs(omega - f))
            # value at frequency should be > neighbours far away
            assert vdos[idx] > vdos[0]
