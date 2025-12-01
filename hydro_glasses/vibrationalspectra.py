# python
from dataclasses import dataclass, field
import numpy as np
from typing import Sequence, Optional,Any, Mapping
try:
    from multiprocessing import Pool
except:
    try:
        from multiprocess import Pool
    except ImportError:
        raise ImportError("multiprocessing or multiprocess module is required for parallel execution")
import os
import logging
from hydro_glasses.vibrationalsystem import VibrationalSystem
from hydro_glasses import lanczos
from hydro_glasses import amorphous_tools as at

log = logging.getLogger(__name__)

@dataclass
class LanczosOptions:
    eta: float = 1.0
    omega_array: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 130, 20000))
    hl_steps: int = 200
    use_ortho: bool = False

    def __post_init__(self):
        self.finalize()

    def finalize(self) -> None:
        # normalize types
        if not isinstance(self.omega_array, np.ndarray):
            self.omega_array = np.asarray(self.omega_array)
        # basic validation
        if not isinstance(self.hl_steps, int) or self.hl_steps <= 0:
            raise ValueError("`hl_steps` must be a positive integer")
        if not (isinstance(self.eta, (int, float)) and self.eta > 0):
            raise ValueError("`eta` must be a positive number")

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]):
        """
        Construct from a mapping (e.g. a dict). If `mapping` is already
        a LanczosOptions instance, return it.
        """
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        allowed = {"eta", "omega_array", "hl_steps", "use_ortho"}
        kwargs = {k: mapping[k] for k in mapping.keys() & allowed}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return {
            "eta": float(self.eta),
            "omega_array": np.asarray(self.omega_array),
            "hl_steps": int(self.hl_steps),
            "use_ortho": bool(self.use_ortho),
        }

# module-level global and worker (must be module-level for multiprocessing)
_GLOBAL_DYNMAT = None

def _lanczos_worker(args):
    """
    Module-level worker that uses module global `_GLOBAL_DYNMAT`.
    Expects args: (ket, hl_steps, omega_array, eta, normalize, use_ortho)
    """
    ket, hl_steps, omega_array, eta, normalize, use_ortho = args
    global _GLOBAL_DYNMAT
    if _GLOBAL_DYNMAT is None:
        raise RuntimeError("_GLOBAL_DYNMAT is not set in worker process")
    return lanczos.spectrum(_GLOBAL_DYNMAT, ket, hl_steps, omega_array, eta, normalize, use_ortho)


class VibrationalSpectra:
    """
    High-level object that holds a VibrationalSystem and Lanczos defaults.
    Usage:
      vsys = VibrationalSystem.from_root(root)
      opts = LanczosOptions() or a dict with options
      vspec = VibrationalSpectra(vsys, opts)
      result = vspec.spectrum_for_ket(ket_vector)
      batch = vspec.spectrum_for_ket_list([ket1, ket2], parallel=True)
    """
    def __init__(self, vibrational_system, options: Optional[LanczosOptions] = None):
        # accept either VibrationalSystem instance or a root path
        if isinstance(vibrational_system, str):
            if not os.path.isdir(vibrational_system):
                raise FileNotFoundError(f"root path `{vibrational_system}` does not exist or is not a directory")
            self.vs = VibrationalSystem.from_root(vibrational_system)
        else:
            self.vs = vibrational_system

        if options is None:
            options = LanczosOptions()
        elif not isinstance(options, LanczosOptions):
            options = LanczosOptions.from_mapping(options)
        options.finalize()
        options.finalize()
        self.options = options
        self.last_result = None

    def spectrum_for_ket(self, ket: np.ndarray, hl_steps: Optional[int] = None,
                         eta: Optional[float] = None, omega_array: Optional[np.ndarray] = None,
                         use_ortho: Optional[bool] = None, normalize: bool = True):
        if self.vs.dynmat is None:
            if getattr(self.vs, "root", None):
                self.vs.initialize_from_root()
            else:
                raise RuntimeError("VibrationalSystem has no `dynmat` loaded")

        hl_steps = self.options.hl_steps if hl_steps is None else hl_steps
        eta = self.options.eta if eta is None else eta
        omega_array = self.options.omega_array if omega_array is None else omega_array
        use_ortho = self.options.use_ortho if use_ortho is None else use_ortho

        result = lanczos.spectrum(self.vs.dynmat, ket, hl_steps, omega_array, eta, normalize, use_ortho)
        self.last_result = result
        return result

    def spectrum_for_ket_list(self, ket_list: Sequence[np.ndarray], parallel: bool = True,
                              ncpus: Optional[int] = None, **per_call_kwargs):
        if self.vs.dynmat is None:
            if getattr(self.vs, "root", None):
                self.vs.initialize_from_root()
            else:
                raise RuntimeError("VibrationalSystem has no `dynmat` loaded")

        if ncpus is None:
            ncpus = os.cpu_count() or 1
        ncpus = min(ncpus, len(ket_list))
        print('Number of processes used', ncpus,flush=True)
        hl_steps = per_call_kwargs.get("hl_steps", self.options.hl_steps)
        eta = per_call_kwargs.get("eta", self.options.eta)
        omega_array = per_call_kwargs.get("omega_array", self.options.omega_array)
        use_ortho = per_call_kwargs.get("use_ortho", self.options.use_ortho)
        normalize = per_call_kwargs.get("normalize", True)

        # prepare picklable inputs: (ket, hl_steps, omega_array, eta, normalize, use_ortho)
        inputs = [(ket, hl_steps, omega_array, eta, normalize, use_ortho) for ket in ket_list]

        # set module global so worker processes (created via fork on Linux) inherit the dynmat
        global _GLOBAL_DYNMAT
        _GLOBAL_DYNMAT = self.vs.dynmat

        try:
            if parallel and ncpus > 1:
                with Pool(ncpus) as p:
                    results = p.map(_lanczos_worker, inputs)
            else:
                results = [_lanczos_worker(inp) for inp in inputs]
        finally:
            # clear global to avoid accidental reuse
            _GLOBAL_DYNMAT = None

        return results

    def compute_stochastic_vdos(self, nstoc=10, normalize: bool = True, parallel: bool = True,
                               ncpus: Optional[int] = None, save=True):
        """
        Compute stochastic VDOS using random kets.
        Returns a list of results for each stochastic sample.
        """
        natoms = self.vs.atoms.get_global_number_of_atoms()
        dim = 3 * natoms

        ket_list = [np.random.randn(dim) for _ in range(nstoc)]

        results = self.spectrum_for_ket_list(ket_list, parallel=parallel, ncpus=ncpus,
                                             normalize=normalize)
        results = np.array(results, dtype='object')

        spectrum = {}
        for i in range(nstoc):
            spectrum[i] = {}
            spectrum[i]['S'] = results[i][0]
            spectrum[i]['alpha'] = results[i][1]
            spectrum[i]['beta'] = results[i][2]
        if save:
            root = self.vs.root
            hl_steps = self.options.hl_steps
            eta = self.options.eta
            np.save(root + '/spectrum_hlsteps{}_eta{}.npy'.format(hl_steps, eta), spectrum)

        return spectrum

    def compute_vdfs(self,
                     Q_list: Optional[np.ndarray] = None,
                     isotropic_minimal: bool = True,
                     nq: Optional[int] = None,
                     normalize: bool = True,
                     parallel: bool = True,
                     ncpus: Optional[int] = None,
                     save: bool = True):
        """
        Compute VDFS (S(Q, omega)) and return a `spectrum` dict.

        - If `Q_list` is provided it will be used (expects shape (M,3)).
        - If `Q_list` is None and `isotropic_minimal` is True, `nq` must be provided and
          the isotropic-minimal Q set is built from `nq`.
        - If `Q_list` is None and `isotropic_minimal` is False, raise ValueError.
        """
        # ensure dynmat/atoms available
        if self.vs.dynmat is None:
            if getattr(self.vs, "root", None):
                self.vs.initialize_from_root()
            else:
                raise RuntimeError("VibrationalSystem has no `dynmat` loaded")

        # Build or validate Q_list
        if Q_list is None:
            if not isotropic_minimal:
                raise ValueError("No `Q_list` provided: set `isotropic_minimal=True` to auto-generate Qs or pass a `Q_list`")
            # isotropic_minimal requested -> require nq
            if nq is None:
                raise ValueError("`nq` must be provided when `isotropic_minimal` is True")
            # build isotropic-minimal Q_list (integer triplets)
            Q_list = np.array(list(set(tuple(sorted(l)) for l in [[i, j, k] for i in range(nq) for j in range(nq) for k in range(nq)])))
            zero_idx = np.argwhere((Q_list == [0, 0, 0]).all(axis=1))
            if zero_idx.size:
                Q_list = np.delete(Q_list, zero_idx, axis=0)
            Q_list = Q_list[np.argsort(np.linalg.norm(Q_list, axis=1))]
            Q_list = Q_list[np.unique(np.linalg.norm(Q_list, axis=1), return_index=True)[1]]
        else:
            Q_list = np.asarray(Q_list)
            if Q_list.ndim != 2 or Q_list.shape[1] != 3:
                raise ValueError("`Q_list` must be an array with shape (M, 3)")

        # Prepare positions and reciprocal cell
        positions = np.array([self.vs.atoms.get_positions()])
        pos = np.transpose(positions, axes=(0, 2, 1))
        reciprocal_cell = np.linalg.inv(self.vs.atoms.cell)

        # Compute kets for longitudinal and transverse polarizations
        ket_Q_L, ket_Q_T = at.compute_phi_Q(Q_list, reciprocal_cell, pos)

        natoms = self.vs.atoms.get_global_number_of_atoms()
        ket_Q_L = ket_Q_L.reshape([3 * natoms, Q_list.shape[0]])
        ket_Q_T = ket_Q_T.reshape([3 * natoms, Q_list.shape[0]])

        # determine ncpus
        if ncpus is None:
            ncpus = os.cpu_count() or 1
        ncpus = min(ncpus, max(1, Q_list.shape[0]))

        # prepare ket lists and compute spectra (L and T)
        ket_list_L = [ket_Q_L[:, i] for i in range(ket_Q_L.shape[1])]
        ket_list_T = [ket_Q_T[:, i] for i in range(ket_Q_T.shape[1])]

        results_L = self.spectrum_for_ket_list(ket_list_L, parallel=parallel, ncpus=ncpus, normalize=normalize)
        results_T = self.spectrum_for_ket_list(ket_list_T, parallel=parallel, ncpus=ncpus, normalize=normalize)

        # assemble output into a new spectrum dict
        spectrum = {}
        omega_array = self.options.omega_array
        Q_array = np.array([2 * np.pi * np.matmul(reciprocal_cell, Q_) for Q_ in Q_list])

        spectrum['omega'] = omega_array
        spectrum['Q'] = Q_array

        for iq in range(Q_array.shape[0]):
            spectrum[iq] = {}
            S_L, alpha_L, beta_L = results_L[iq]
            spectrum[iq]['L'] = {'S': S_L, 'alpha': alpha_L, 'beta': beta_L}
            S_T, alpha_T, beta_T = results_T[iq]
            spectrum[iq]['T'] = {'S': S_T, 'alpha': alpha_T, 'beta': beta_T}

        # optional save
        if save and getattr(self.vs, "root", None):
            root = self.vs.root
            hl_steps = getattr(self.options, "hl_steps", None)
            eta = getattr(self.options, "eta", None)
            fname = os.path.join(root, f'vdfs_hlsteps{hl_steps}_eta{eta}_isomin{int(bool(isotropic_minimal))}.npy')
            np.save(fname, spectrum)

        return spectrum