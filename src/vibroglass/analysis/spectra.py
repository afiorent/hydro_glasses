"""High-level API for computing vibrational spectra via Lanczos.

Provides :class:`LanczosOptions` for configuration and
:class:`VibrationalSpectra` for parallel spectrum computations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vibroglass.analysis import structure_factor as at
from vibroglass.core import lanczos
from vibroglass.core.system import VibrationalSystem

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)


@dataclass
class LanczosOptions:
    """Configuration for the Haydock-Lanczos spectrum computation.

    Parameters
    ----------
    eta : float
        Lorentzian broadening parameter.
    omega_array : NDArray
        Frequency grid.
    hl_steps : int
        Number of Lanczos iterations.
    use_ortho : bool
        Use forced orthogonalisation.
    """

    eta: float = 1.0
    omega_array: NDArray[np.floating] = field(default_factory=lambda: np.linspace(0.1, 130, 20000))
    hl_steps: int = 200
    use_ortho: bool = False

    def __post_init__(self) -> None:
        self.finalize()

    def finalize(self) -> None:
        """Validate and normalise option values."""
        if not isinstance(self.omega_array, np.ndarray):
            self.omega_array = np.asarray(self.omega_array)
        if not isinstance(self.hl_steps, int) or self.hl_steps <= 0:
            raise ValueError("`hl_steps` must be a positive integer")
        if not (isinstance(self.eta, (int, float)) and self.eta > 0):
            raise ValueError("`eta` must be a positive number")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object] | None) -> LanczosOptions:
        """Construct from a dict-like mapping."""
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        allowed = {"eta", "omega_array", "hl_steps", "use_ortho"}
        kwargs = {k: mapping[k] for k in mapping.keys() & allowed}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return {
            "eta": float(self.eta),
            "omega_array": np.asarray(self.omega_array),
            "hl_steps": int(self.hl_steps),
            "use_ortho": bool(self.use_ortho),
        }


# Module-level globals for multiprocessing workers
_GLOBAL_DYNMAT = None


def _init_worker(dynmat: object) -> None:
    """Initialiser for worker processes: set module-level dynmat."""
    global _GLOBAL_DYNMAT
    _GLOBAL_DYNMAT = dynmat


def _lanczos_worker(args: tuple) -> NDArray:
    """Worker function for parallel Lanczos computation."""
    ket, hl_steps, omega_array, eta, use_ortho = args
    global _GLOBAL_DYNMAT
    if _GLOBAL_DYNMAT is None:
        raise RuntimeError("_GLOBAL_DYNMAT is not set in worker process")
    return lanczos.spectrum(
        _GLOBAL_DYNMAT,
        ket,
        hl_steps,
        omega_array,
        eta,
        return_chain=True,
        use_ortho=use_ortho,
    )


class VibrationalSpectra:
    """High-level API for vibrational spectrum computations.

    Parameters
    ----------
    vibrational_system : VibrationalSystem or str
        A :class:`~vibroglass.core.system.VibrationalSystem` instance or
        a root directory path.
    options : LanczosOptions or dict, optional
        Lanczos computation parameters.
    """

    def __init__(
        self,
        vibrational_system: VibrationalSystem | str | os.PathLike[str],
        options: LanczosOptions | dict | None = None,
    ) -> None:
        if isinstance(vibrational_system, (str, os.PathLike)):
            self.vs = VibrationalSystem.from_root(vibrational_system)
        else:
            self.vs = vibrational_system

        if options is None:
            options = LanczosOptions()
        elif not isinstance(options, LanczosOptions):
            options = LanczosOptions.from_mapping(options)
        self.options = options
        self.last_result: NDArray | None = None

    def spectrum_for_ket(
        self,
        ket: NDArray[np.floating],
        hl_steps: int | None = None,
        eta: float | None = None,
        omega_array: NDArray[np.floating] | None = None,
        use_ortho: bool | None = None,
    ) -> NDArray[np.floating]:
        """Compute a spectrum for a single ket vector.

        Parameters
        ----------
        ket : (n_dof,) array
            Input ket vector.
        hl_steps, eta, omega_array, use_ortho
            Override defaults from :attr:`options`.

        Returns
        -------
        (n_omega,) array
            Spectrum values.
        """
        self._ensure_dynmat()
        hl_steps = self.options.hl_steps if hl_steps is None else hl_steps
        eta = self.options.eta if eta is None else eta
        omega_array = self.options.omega_array if omega_array is None else omega_array
        use_ortho = self.options.use_ortho if use_ortho is None else use_ortho

        result = lanczos.spectrum(
            self.vs.dynmat, ket, hl_steps, omega_array, eta, use_ortho=use_ortho
        )
        self.last_result = result
        return result

    def spectrum_for_ket_list(
        self,
        ket_list: Sequence[NDArray[np.floating]],
        *,
        parallel: bool = True,
        ncpus: int | None = None,
        **per_call_kwargs: object,
    ) -> list[NDArray | tuple]:
        """Compute spectra for a list of ket vectors.

        Parameters
        ----------
        ket_list : sequence of arrays
            Input ket vectors.
        parallel : bool
            Use multiprocessing.
        ncpus : int, optional
            Number of worker processes.
        **per_call_kwargs
            Override ``hl_steps``, ``eta``, ``omega_array``, ``use_ortho``.

        Returns
        -------
        list
            One result per ket.
        """
        self._ensure_dynmat()

        if ncpus is None:
            ncpus = os.cpu_count() or 1
        ncpus = min(ncpus, len(ket_list))
        log.info("Number of processes used: %d", ncpus)

        hl_steps = per_call_kwargs.get("hl_steps", self.options.hl_steps)
        eta = per_call_kwargs.get("eta", self.options.eta)
        omega_array = per_call_kwargs.get("omega_array", self.options.omega_array)
        use_ortho = per_call_kwargs.get("use_ortho", self.options.use_ortho)

        inputs = [(ket, hl_steps, omega_array, eta, use_ortho) for ket in ket_list]

        global _GLOBAL_DYNMAT
        _GLOBAL_DYNMAT = self.vs.dynmat

        try:
            if parallel and ncpus > 1:
                from multiprocessing import Pool

                with Pool(ncpus, initializer=_init_worker, initargs=(self.vs.dynmat,)) as p:
                    results = p.map(_lanczos_worker, inputs)
            else:
                results = [_lanczos_worker(inp) for inp in inputs]
        finally:
            _GLOBAL_DYNMAT = None

        return results

    def compute_stochastic_vdos(
        self,
        nstoc: int = 10,
        *,
        parallel: bool = True,
        ncpus: int | None = None,
        save: bool = True,
        element: str | None = None,
    ) -> dict:
        """Compute stochastic VDOS using random kets.

        Parameters
        ----------
        nstoc : int
            Number of stochastic samples.
        parallel : bool
            Use multiprocessing.
        ncpus : int, optional
            Number of worker processes.
        save : bool
            Save results to *root* directory.
        element : str, optional
            Restrict to a specific element (experimental).

        Returns
        -------
        dict
            ``{i: {'S': ..., 'alpha': ..., 'beta': ...}}`` for each sample.
        """
        self._ensure_dynmat()
        natoms = len(self.vs.atoms)
        dim = 3 * natoms
        rng = np.random.default_rng()
        ket_list = [rng.standard_normal(dim) for _ in range(nstoc)]

        if element is not None:
            log.info("Extracting VDOS for element: %s (experimental)", element)
            symbols = np.array(self.vs.atoms.get_chemical_symbols())
            mask = symbols == element
            if not mask.any():
                raise ValueError(f"element `{element}` not found in system")
            mask3 = np.repeat(mask, 3).astype(float)
            ket_list = [k * mask3 for k in ket_list]

        results_raw = self.spectrum_for_ket_list(ket_list, parallel=parallel, ncpus=ncpus)

        spectrum: dict = {}
        for i in range(nstoc):
            spec, alpha, beta = results_raw[i]
            spectrum[i] = {"S": spec, "alpha": alpha, "beta": beta}

        if save and self.vs.root:
            fname = os.path.join(
                self.vs.root,
                f"spectrum_hlsteps{self.options.hl_steps}_eta{self.options.eta}.npy",
            )
            np.save(fname, spectrum)

        return spectrum

    def compute_vdsf(
        self,
        Q_list: NDArray[np.integer] | None = None,
        *,
        isotropic_minimal: bool = True,
        nq: int | None = None,
        parallel: bool = True,
        ncpus: int | None = None,
        save: bool = True,
    ) -> dict:
        """Compute the vibrational dynamic structure factor S(Q, omega).

        Parameters
        ----------
        Q_list : (M, 3) int array, optional
            Wavevector indices. If ``None``, auto-generated.
        isotropic_minimal : bool
            Auto-generate an isotropic-minimal Q set.
        nq : int, optional
            Range for auto-generated Q set.
        parallel : bool
            Use multiprocessing.
        ncpus : int, optional
            Number of worker processes.
        save : bool
            Save results to *root* directory.

        Returns
        -------
        dict
            Structured spectrum dictionary.
        """
        self._ensure_dynmat()

        if Q_list is None:
            if not isotropic_minimal:
                raise ValueError("No Q_list provided: set isotropic_minimal=True or pass a Q_list")
            if nq is None:
                raise ValueError("`nq` must be provided when isotropic_minimal is True")
            Q_list = _build_isotropic_minimal_qlist(nq)
        else:
            Q_list = np.asarray(Q_list)
            if Q_list.ndim != 2 or Q_list.shape[1] != 3:
                raise ValueError("`Q_list` must be an array with shape (M, 3)")

        positions = np.array([self.vs.atoms.get_positions()])
        pos = np.transpose(positions, axes=(0, 2, 1))
        reciprocal_cell = np.linalg.inv(self.vs.atoms.cell)

        ket_Q_L, ket_Q_T = at.compute_phi_Q(Q_list, reciprocal_cell, pos)

        natoms = len(self.vs.atoms)
        ket_Q_L = ket_Q_L.reshape(3 * natoms, Q_list.shape[0])
        ket_Q_T = ket_Q_T.reshape(3 * natoms, Q_list.shape[0])

        if ncpus is None:
            ncpus = os.cpu_count() or 1
        ncpus = min(ncpus, max(1, Q_list.shape[0]))

        ket_list_L = [ket_Q_L[:, i] for i in range(ket_Q_L.shape[1])]
        ket_list_T = [ket_Q_T[:, i] for i in range(ket_Q_T.shape[1])]

        results_L = self.spectrum_for_ket_list(ket_list_L, parallel=parallel, ncpus=ncpus)
        results_T = self.spectrum_for_ket_list(ket_list_T, parallel=parallel, ncpus=ncpus)

        Q_array = np.array([2 * np.pi * reciprocal_cell @ Q_ for Q_ in Q_list])
        spectrum: dict = {"omega": self.options.omega_array, "Q": Q_array}
        for iq in range(Q_array.shape[0]):
            S_L, alpha_L, beta_L = results_L[iq]
            S_T, alpha_T, beta_T = results_T[iq]
            spectrum[iq] = {
                "L": {"S": S_L, "alpha": alpha_L, "beta": beta_L},
                "T": {"S": S_T, "alpha": alpha_T, "beta": beta_T},
            }

        if save and self.vs.root:
            fname = os.path.join(
                self.vs.root,
                f"vdsf_hlsteps{self.options.hl_steps}_eta{self.options.eta}.npy",
            )
            np.save(fname, spectrum)

        return spectrum

    def compute_IR_spectrum(
        self,
        polarizations: str | int | tuple[str | int, ...] = ("x", "y"),
        charges: NDArray[np.floating] | None = None,
        charges_dict: dict[str, float] | None = None,
        *,
        parallel: bool = True,
        ncpus: int | None = None,
        save: bool = True,
    ) -> dict:
        """Compute IR spectra for the system.

        Parameters
        ----------
        polarizations : str, int, or tuple
            Polarisation directions.
        charges, charges_dict
            Passed to :func:`~vibroglass.analysis.structure_factor.make_IR_vector`.
        parallel : bool
            Use multiprocessing.
        ncpus : int, optional
            Number of worker processes.
        save : bool
            Save results to *root* directory.

        Returns
        -------
        dict
            ``{'omega': ..., i: {'S': ..., 'alpha': ..., 'beta': ...}}``.
        """
        self._ensure_dynmat()

        phi = at.make_IR_vector(
            atoms=self.vs.atoms,
            polarizations=polarizations,
            charges=charges,
            charges_dict=charges_dict,
        )
        phi = np.asarray(phi)
        if phi.ndim == 1:
            phi = phi[None, :]

        ket_list = [phi[i] for i in range(phi.shape[0])]
        results_raw = self.spectrum_for_ket_list(ket_list, parallel=parallel, ncpus=ncpus)

        spectrum: dict = {"omega": self.options.omega_array}
        for i, res in enumerate(results_raw):
            S, alpha, beta = res
            spectrum[i] = {"S": S, "alpha": alpha, "beta": beta}

        if save and self.vs.root:
            fname = os.path.join(
                self.vs.root,
                f"IR_hlsteps{self.options.hl_steps}_eta{self.options.eta}.npy",
            )
            np.save(fname, spectrum)

        return spectrum

    def _ensure_dynmat(self) -> None:
        if self.vs.dynmat is None:
            if self.vs.root:
                self.vs._initialize_from_root()
            else:
                raise RuntimeError("VibrationalSystem has no `dynmat` loaded")


def _build_isotropic_minimal_qlist(nq: int) -> NDArray[np.integer]:
    """Build an isotropic-minimal set of Q-point indices."""
    raw = set(
        tuple(sorted(triplet))
        for triplet in ((i, j, k) for i in range(nq) for j in range(nq) for k in range(nq))
    )
    Q_list = np.array([list(q) for q in raw])
    zero_idx = np.argwhere((Q_list == [0, 0, 0]).all(axis=1))
    if zero_idx.size:
        Q_list = np.delete(Q_list, zero_idx, axis=0)
    norms = np.linalg.norm(Q_list, axis=1)
    Q_list = Q_list[np.argsort(norms)]
    _, unique_idx = np.unique(np.linalg.norm(Q_list, axis=1), return_index=True)
    return Q_list[unique_idx]
