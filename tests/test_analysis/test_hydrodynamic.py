"""Tests for vibroglass.analysis.hydrodynamic."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from vibroglass.analysis.hydrodynamic import (
    calculate_heat_capacity,
    compute_Gamma,
    dos_omega_L,
    dos_omega_T,
)


class TestCalculateHeatCapacity:
    def test_classical_limit(self):
        """At high T, heat capacity per mode should approach k_B."""
        kB = 1.380649e-23
        omega = np.array([10.0, 20.0, 30.0])
        cv = calculate_heat_capacity(omega, temp=1e6)
        assert_allclose(cv, kB, rtol=1e-3)

    def test_zero_frequency(self):
        """At omega=0, heat capacity should be k_B (classical)."""
        kB = 1.380649e-23
        omega = np.array([0.0])
        cv = calculate_heat_capacity(omega, temp=300.0)
        assert_allclose(cv, kB)

    def test_shape(self):
        omega = np.linspace(0, 50, 100)
        cv = calculate_heat_capacity(omega, temp=300.0)
        assert cv.shape == omega.shape


class TestDosOmega:
    def test_dos_T_positive(self):
        omega = np.linspace(0.1, 10, 50)
        dos = dos_omega_T(omega, cT=3.0)
        assert np.all(dos >= 0)

    def test_dos_L_positive(self):
        omega = np.linspace(0.1, 10, 50)
        dos = dos_omega_L(omega, cL=5.0)
        assert np.all(dos >= 0)

    def test_dos_scaling(self):
        """DOS should scale as omega^2."""
        omega = np.array([1.0, 2.0])
        dos = dos_omega_T(omega, cT=1.0)
        assert_allclose(dos[1] / dos[0], 4.0)


class TestComputeGamma:
    def test_compute_Gamma_keys(self):
        """compute_Gamma should return a dict with same keys as S."""
        omega = np.linspace(0.1, 10, 100)
        # Simulate a spectrum dict
        rng = np.random.default_rng(42)
        S_L = rng.random((5, 100, 1))
        S_T = rng.random((5, 100, 1))
        spectrum = {"omega": omega, "S": {"L": S_L, "T": S_T}}

        result = compute_Gamma(spectrum)
        assert "L" in result
        assert "T" in result
        assert result["L"].shape == (5,)
        assert result["T"].shape == (5,)
