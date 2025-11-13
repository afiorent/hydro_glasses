import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe
import sys

from time import time
from ase.io import read,write
from ase import units
from hydro_glasses import lanczos,test_lanczos
from hydro_glasses import amorphous_tools as at
import logging
import os
from scipy import sparse
#multiprocessing
from multiprocess import Pool
from typing import Sequence, Union, Dict
from ase import Atoms

def DHO(w, w0, sigma, norm):
    return norm*(w*2*sigma/((w*sigma)**2 + (w**2-w0**2)**2))
def wDHO(w, w0, sigma):
    return w**2*2*sigma/((w*sigma)**2 + (w**2-w0**2)**2)
def gauss(x,sigma):
    return np.exp(-((x) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
def lorenztian(x,eta):
    return eta/np.pi/((x)**2+eta**2)




def make_IR_vector(atoms: Atoms,
                   polarizations: Union[str, int, Sequence[Union[str, int]]] = ('x', 'y'),
                   charges: Union[None, Sequence, np.ndarray] = None,
                   charges_dict: Dict[str, float] = None) -> np.ndarray:
    """
    Build normalized IR vector(s) and return an array of shape (n_vectors, 3*N).

    - `polarizations`: 'x','y','z' or 0,1,2 or a sequence thereof. A single polarization
      may be provided as a string or int.
    - `charges`: None or array. Accepted shapes:
        * (N,)       -> per-atom scalar Z (becomes diagonal)
        * (N,3)      -> per-atom per-direction
        * (N,3,P)    -> per-atom per-direction with P polarizations
        * flat 3N -> treated as (N,3)
    - `charges_dict`: element symbol -> scalar charge. If `charges` is None, builds
      (N,3,3) with Z_i * I_3.

    Returns:
    - numpy.ndarray of shape (n_vectors, 3*N). `n_vectors = len(polarizations) * P`.
      If only one vector is created, the shape is `(1, 3*N)` so it is iterable.
    """
    # normalize polarizations input to a list of indices
    if isinstance(polarizations, (str, int)):
        pols = (polarizations,)
    else:
        pols = tuple(polarizations)

    pol_indices = []
    for p in pols:
        if isinstance(p, str):
            mp = p.lower()
            if mp == 'x':
                pol_indices.append(0)
            elif mp == 'y':
                pol_indices.append(1)
            elif mp == 'z':
                pol_indices.append(2)
            else:
                raise ValueError(f"Unknown polarization '{p}'")
        elif isinstance(p, int):
            if p not in (0, 1, 2):
                raise ValueError("Polarization index must be 0,1,2")
            pol_indices.append(p)
        else:
            raise ValueError("Polarizations must be strings 'x','y','z' or ints 0..2")

    N = len(atoms)
    if N == 0:
        return np.zeros((0, 0), dtype=float)

    masses = np.asarray(atoms.get_masses(), dtype=float)
    if masses.size != N:
        raise ValueError("Unexpected number of masses from atoms")
    if np.any(masses <= 0):
        raise ValueError("All atomic masses must be positive")

    # Build charges_arr with shape (N,3,P)
    if charges is None:
        if charges_dict is None:
            raise ValueError("Provide either `charges` or `charges_dict`")
        charges_arr = np.zeros((N, 3, 3), dtype=float)
        for i, atom in enumerate(atoms):
            sym = atom.symbol
            if sym not in charges_dict:
                raise KeyError(f"Element {sym} not found in charges_dict")
            charges_arr[i] = float(charges_dict[sym]) * np.eye(3)
    else:
        ch = np.asarray(charges, dtype=float)
        if ch.ndim == 1 and ch.size == N:
            # (N,) -> diagonal per-atom -> (N,3,3)
            charges_arr = ch[:, None, None] * np.eye(3)[None, :, :]
        elif ch.ndim == 2 and ch.shape == (N, 3):
            # (N,3) -> (N,3,1)
            charges_arr = ch[:, :, None]
        elif ch.ndim == 3 and ch.shape[0] == N and ch.shape[1] == 3:
            # (N,3,P)
            charges_arr = ch
        elif ch.size == 3 * N:
            charges_arr = ch.reshape(N, 3)[:, :, None]
        else:
            raise ValueError("`charges` must have shape (N,), (N,3), (N,3,P) or be compatible with N atoms")

    P = charges_arr.shape[2]
    inv_sqrt_m = 1.0 / np.sqrt(masses)
    # phi_N3P shape: (N,3,P)# phi_N3P shape: (N,3,P)# phi_N3P shape: (N,3,P)
    phi_N3P = np.einsum('ijp,i->ijp', charges_arr, inv_sqrt_m)

    # produce one 3N vector per p (flatten per-atom 3 components)
    n_vectors = P
    result = np.zeros((n_vectors, N * 3), dtype=float)

    for p in range(P):
        vec = phi_N3P[:, :, p].reshape(N * 3)
        norm = np.linalg.norm(vec)
        if norm != 0.0:
            vec /= norm
        result[p] = vec# phi_N3P shape: (N,3,P)
    return result

root='aSiO2_648/'
dynmat_sparse=sparse.load_npz(root+'dynmat.npz')
dynmat_sparse=(dynmat_sparse+dynmat_sparse.transpose())/2 #sometimes it is not symmetrical for numerical reasons
atoms=read(root+'replicated_atoms.xyz')
natoms=atoms.get_global_number_of_atoms()


# Prepare inputs for haydock algorithm

eta=1
k=200
spectrum={}
omega_array=np.linspace(0.1,300,10000)
spectrum['omega']=omega_array


# Generating vectors for IR calculation
# let assume ionic charges instead of Born charges for simplicity
phi=make_IR_vector(atoms=atoms,polarizations=[0,1,2],charges_dict={'Si':3.2,'O':-1.6})
print(phi[0,:30],atoms.get_chemical_symbols()[:10])

ncpus=os.cpu_count()
print('ncpus:',ncpus)
if ncpus>20:
    ncpus=12
start=time()
with Pool(ncpus) as p:
    inputs = [(dynmat_sparse,phi_,k,omega_array,eta,True) for phi_ in phi ]
    result = p.starmap(lanczos.spectrum, inputs)
end=time()
print('Elapsed time (s):',end-start)
#saving
result=np.array(result,dtype='object')
for i in range(3):
    spectrum[i]={}
    spectrum[i]['S']=result[i][0]
    spectrum[i]['alpha']=result[i][1]
    spectrum[i]['beta']=result[i][2]

np.save(root+'IR_k{}_eta{}.npy'.format(k,eta),spectrum)


fig,ax=plt.subplots()
spectrum=np.load(root+'IR_k{}_eta{}.npy'.format(k,eta),allow_pickle=True).item()
for i,pol in enumerate(['x','y','z']):
    w=spectrum['omega']
    y=spectrum[i]['S']
    norm=np.trapz(y, w)
    y/=norm
    ax.plot(w/2/np.pi,y*2*np.pi,label='IR pol. {}'.format(pol))
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('IR (a.u.)')
plt.title('IR Spectrum via Haydock')
plt.show()





# Comparison haydock vs direct method


# dynmat_dense=dynmat_sparse.toarray()
# w2,eig=np.linalg.eigh(dynmat_dense)
#
#
#
#
# w_true=np.sign(w2)*np.sqrt(np.abs(w2))
# fig,ax=plt.subplots()
# # for i,pol in enumerate(['x']):
# #     w=spectrum['omega']
# #     y=spectrum[i]['S']
# #     norm=np.trapz(y, w)
# #     y/=norm
# #     ax.plot(w,y,label='IR pol. {}'.format(pol))
# for i, pol in enumerate(['x']):
#     Z_mu=np.einsum('i,in->n',phi[i],eig)
#     print(Z_mu.shape)
#     ax.plot(w_true, Z_mu**2/w_true, label='IR direct pol. {}'.format(pol))
# plt.show()
