# Examples

Worked examples demonstrating core `vibroglass` workflows.

## 0 -- Amorphous Silicon VDSF (`0_aSi_DSF/`)

Compute the vibrational dynamic structure factor S(Q, omega) for a 1728-atom
amorphous silicon system.

| Script | Description |
|--------|-------------|
| `0a_diagonalize.py` | Direct eigendecomposition of the dynamical matrix |
| `0b_direct_DSF.py` | Direct S(Q, omega) computation |
| `1_haydock_DSF.py` | **Haydock-Lanczos** S(Q, omega) (recommended) |
| `2_VDOS.py` | Stochastic VDOS with comparison to direct method |
| `lammps_dynmat2sparse.py` | Convert LAMMPS dynamical matrices to sparse `.npz` |
| `plot_all.ipynb` | Notebook comparing direct vs Haydock results |

**Data:** uncompress `data_N1728.tar.gz` before running.

## 1 -- IR and element-resolved VDOS (`1_IR_and_stoc_VDOS/`)

Compute infrared spectra and element-resolved VDOS for a 648-atom amorphous
SiO2 system.

| Script | Description |
|--------|-------------|
| `0_VDOS_element.py` | Total and element-resolved (Si, O) stochastic VDOS |
| `1_IR.py` | IR spectrum with Born effective charges |
| `plot_aSiO2.ipynb` | Visualization notebook |

## Quick start

```bash
# From the examples/0_aSi_DSF directory:
tar xzf data_N1728.tar.gz
python 1_haydock_DSF.py
```
