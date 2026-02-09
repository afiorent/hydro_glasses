# Getting Started

## Installation

Install vibroglass from the repository root:

```bash
pip install -e .
```

### Dependencies

**Core:** `numpy`, `scipy`, `opt_einsum`, `ase`, `pandas`

**Optional:**
- `matplotlib` -- plotting: `pip install -e ".[plot]"`
- `kaldo` -- phonon interoperability: `pip install -e ".[kaldo]"`

## Quick start

### Load a system and compute VDOS

```python
from vibroglass import VibrationalSystem, VibrationalSpectra, LanczosOptions
import numpy as np

# Load dynamical matrix and atoms from a directory containing
# dynmat.npz and replicated_atoms.xyz
system = VibrationalSystem(root="path/to/data")

# Configure the Lanczos computation
options = LanczosOptions(hl_steps=300, eta=1.0)
spectra = VibrationalSpectra(system, options=options)

# Compute stochastic VDOS with 10 random vectors
vdos = spectra.compute_stochastic_vdos(nstoc=10)

# Average spectra
omega = spectra.options.omega_array
mean_spectrum = np.mean([omega * vdos[i]["S"] for i in range(10)], axis=0)
```

### Compute $S(Q, \omega)$

```python
spectrum = spectra.compute_vdsf(nq=8)

# Access longitudinal and transverse components
Q = spectrum["Q"]
omega = spectrum["omega"]
S_L = spectrum[0]["L"]["S"]  # first Q-point, longitudinal
S_T = spectrum[0]["T"]["S"]  # first Q-point, transverse
```

### Compute IR spectrum

```python
ir = spectra.compute_IR_spectrum(
    polarizations=[0, 1, 2],
    charges_dict={"Si": 3.2, "O": -1.6},
)
```

### Load from different formats

```python
from vibroglass import load_dynmat_and_atoms

# From sparse .npz (recommended)
dynmat, atoms = load_dynmat_and_atoms("path/to/data")

# Also supports kaldo's second.npy and LAMMPS Dyn.form
```

## Data format

vibroglass expects a directory containing:

- **`dynmat.npz`** -- sparse dynamical matrix in scipy sparse format (or `second.npy` from kaldo, or `Dyn.form` from LAMMPS)
- **`replicated_atoms.xyz`** -- atomic structure in extended XYZ format (readable by ASE)

## Running tests

```bash
tox
```

This runs linting (`ruff`) and the test suite (`pytest`).
