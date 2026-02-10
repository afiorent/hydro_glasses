# vibroglass

A Python toolkit for computing lattice dynamics and hydrodynamic properties of glasses and amorphous materials.

vibroglass provides an efficient, parallelised pipeline for computing vibrational spectra using the **Haydock-Lanczos** algorithm -- without the need for full matrix diagonalisation.

## Features

- **Vibrational Dynamic Structure Factor** $S(Q, \omega)$ via the Haydock-Lanczos recursion
- **Stochastic VDOS** with element-resolved projections
- **IR spectra** from Born effective charges
- **Hydrodynamic thermal transport** analysis (disorder widths, heat capacity, $\kappa$)
- **kaldo interoperability** for importing phonon data
- Supports LAMMPS, NumPy, and kaldo input formats

```{toctree}
:maxdepth: 2
:caption: User Guide

getting-started
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
