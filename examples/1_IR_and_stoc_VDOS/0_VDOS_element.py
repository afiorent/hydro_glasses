"""Compute element-resolved stochastic VDOS for amorphous SiO2."""

import time

import matplotlib.pyplot as plt
import numpy as np

from vibroglass import VibrationalSpectra, VibrationalSystem
from vibroglass.core.kernels import compute_vdos

if __name__ == "__main__":
    root = "aSiO2_648/"
    omega_array = np.linspace(0.1, 250, 10000)
    nstoc = 10

    system = VibrationalSystem(root=root)
    spectra = VibrationalSpectra(
        system, options={"hl_steps": 300, "eta": 1.0, "omega_array": omega_array}
    )

    # Total stochastic VDOS
    start = time.time()
    VDOS = spectra.compute_stochastic_vdos(nstoc=nstoc)
    end = time.time()
    print(f"Haydock wall time: {end - start:.1f} s")

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

    # Element-resolved VDOS
    VDOS_element = {}
    for element in ["Si", "O"]:
        start = time.time()
        VDOS_el = spectra.compute_stochastic_vdos(nstoc=nstoc, element=element)
        end = time.time()
        print(f"Haydock wall time ({element}): {end - start:.1f} s")
        y_el = np.mean([w * VDOS_el[i]["S"] for i in range(nstoc)], axis=0)
        norm_el = np.trapezoid(y_el, w)
        VDOS_element[element] = y_el / norm_el

    fig, ax = plt.subplots()
    for element in ["Si", "O"]:
        ax.plot(w, VDOS_element[element], label=f"{element}")
    tot_vdos = VDOS_element["Si"] * 1 + VDOS_element["O"] * 2
    ax.plot(w, tot_vdos, "-", label="total", color="black")
    ax.plot(w, vdos_direct * 3, "-.", label="total direct")
    ax.set_title(rf"a{symbol} $N={N}$ $\eta={eta}$")
    ax.set_ylabel("VDOS")
    ax.set_xlabel(r"$\omega$ (rad/ps)")
    ax.legend()
    plt.tight_layout()
    plt.show()
