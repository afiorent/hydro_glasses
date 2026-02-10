"""Compute stochastic VDOS and compare with direct diagonalisation."""

import time

import matplotlib.pyplot as plt
import numpy as np

from vibroglass import VibrationalSpectra, VibrationalSystem
from vibroglass.core.kernels import compute_vdos

if __name__ == "__main__":
    root = "N1728"
    nstoc = 10

    system = VibrationalSystem(root=root)
    spectra = VibrationalSpectra(system, options={"hl_steps": 300, "eta": 1.0})

    # Compute stochastic VDOS
    start = time.time()
    VDOS = spectra.compute_stochastic_vdos(nstoc=nstoc)
    end = time.time()
    print(f"Haydock wall time: {end - start:.1f} s")

    # Average and error bars
    w = spectra.options.omega_array
    y = np.mean([w * VDOS[i]["S"] for i in range(nstoc)], axis=0)
    yerr = 3 * np.std([w * VDOS[i]["S"] for i in range(nstoc)], axis=0) / np.sqrt(nstoc)
    norm = np.trapezoid(y, w)
    y /= norm
    yerr /= norm

    # Direct diagonalisation for comparison
    dynmat_dense = spectra.vs.dynmat.toarray()
    start = time.time()
    w2, eig = np.linalg.eigh(dynmat_dense)
    end = time.time()
    print(f"Diagonalisation time: {end - start:.1f} s")

    eta = spectra.options.eta
    w_true = np.sign(w2) * np.sqrt(np.abs(w2))
    vdos_direct = compute_vdos(w_true, w, sigma=eta, kernel="DHO")

    # Plot comparison
    fig, ax = plt.subplots()
    ax.plot(w, y, label=f"stochastic n={nstoc}")
    ax.fill_between(w, y + yerr, y - yerr, alpha=0.5, label=r"3$\sigma$")
    ax.plot(w, vdos_direct, "-.", label="direct (DHO)")

    symbol = spectra.vs.atoms.get_chemical_symbols()[0]
    N = len(spectra.vs.atoms)
    ax.set_title(rf"a{symbol} $N={N}$ $\eta={eta}$")
    ax.set_ylabel("VDOS")
    ax.set_xlabel(r"$\omega$ (rad/ps)")
    ax.legend()
    plt.tight_layout()
    plt.show()
