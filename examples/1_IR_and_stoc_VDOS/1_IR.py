"""Compute IR spectra for amorphous SiO2 and compare with direct method."""

from time import time

import matplotlib.pyplot as plt
import numpy as np

from vibroglass import VibrationalSpectra, VibrationalSystem
from vibroglass.analysis.structure_factor import make_IR_vector
from vibroglass.core.kernels import compute_kernel

if __name__ == "__main__":
    root = "aSiO2_648/"
    omega_array = np.linspace(0.1, 250, 10000)

    system = VibrationalSystem(root=root)
    spectra = VibrationalSpectra(
        system, options={"hl_steps": 300, "eta": 1.0, "omega_array": omega_array}
    )

    # Haydock IR spectrum
    start = time()
    spectrum = spectra.compute_IR_spectrum(
        polarizations=[0, 1, 2], charges_dict={"Si": 3.2, "O": -1.6}
    )
    end = time()
    print(f"Haydock wall time: {end - start:.1f} s")

    # Direct diagonalisation for comparison
    dynmat_dense = spectra.vs.dynmat.toarray()
    w2, eig = np.linalg.eigh(dynmat_dense)

    atoms = spectra.vs.atoms
    phi = make_IR_vector(atoms=atoms, polarizations=[0, 1, 2], charges_dict={"Si": 3.2, "O": -1.6})
    w_true = np.sign(w2) * np.sqrt(np.abs(w2))

    thz2cm1 = 33.35641
    x = spectra.options.omega_array

    fig, ax = plt.subplots()
    for i, pol in enumerate(["x"]):
        # Direct method
        Z_mu = np.einsum("i,in->n", phi[i], eig)
        y_direct = np.abs(Z_mu) ** 2 / w_true
        y_direct = compute_kernel(
            w_true,
            y_direct,
            x,
            sigma=spectra.options.eta,
            kernel="lorentz",
            normalize=True,
        )
        ax.plot(
            x * thz2cm1 / 2 / np.pi,
            y_direct * 2 * np.pi / thz2cm1,
            label=f"IR direct pol. {pol}",
            marker=".",
        )

        # Haydock result
        y_hl = spectrum[i]["S"]
        norm = np.trapezoid(y_hl, x)
        y_hl = y_hl / norm
        ax.plot(
            x * thz2cm1 / 2 / np.pi,
            y_hl * 2 * np.pi / thz2cm1,
            label=f"IR Haydock pol. {pol}",
        )

    ax.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax.set_ylabel("IR (a.u.)")
    ax.set_xlim(0, 1300)
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.show()
