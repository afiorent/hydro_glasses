"""Compute the vibrational dynamic structure factor S(Q, omega) via Haydock-Lanczos."""

import time

import matplotlib.pyplot as plt
import numpy as np

from vibroglass import VibrationalSpectra, VibrationalSystem

if __name__ == "__main__":
    root = "N1728"
    system = VibrationalSystem(root=root)

    start = time.time()
    spectra = VibrationalSpectra(system, options={"hl_steps": 200, "eta": 1.0})
    spectrum = spectra.compute_vdsf(nq=8)
    end = time.time()
    print(f"Haydock wall time: {end - start:.1f} s")

    # Plotting
    q = np.linalg.norm(spectrum["Q"], axis=1)
    w = spectrum["omega"]
    n_q = len(q)

    S_qw = np.zeros((2, n_q, len(w)))
    for ib, branch in enumerate(["L", "T"]):
        for iq in range(n_q):
            S_qw[ib, iq, :] = spectrum[iq][branch]["S"]

    fig, axes = plt.subplots(ncols=2, sharey=True)
    vmax = 1e-2
    for imode in range(2):
        ax = axes[imode]
        ax.pcolormesh(
            q,
            w / 2 / np.pi,
            w[:, np.newaxis] / 2 / np.pi * S_qw[imode, :, :].T,
            shading="nearest",
            vmin=0,
            vmax=vmax,
            cmap="inferno",
        )
        dispersion = w[np.argmax(w[np.newaxis, :] * S_qw[imode, :, :], axis=1)]
        ax.plot(q, dispersion / 2 / np.pi, "--", color="b")
        ax.set_title("Longitudinal" if imode == 0 else "Transverse")
    fig.suptitle(r"$S(q, \nu)$")
    plt.tight_layout()
    plt.show()
