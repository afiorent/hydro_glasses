import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe
import sys
sys.path.append('/g100/home/userexternal/afiorent/hydro_glasses/')
from time import time
from ase.io import read,write
from ase import units
from hydro_glasses import lanczos,test_lanczos
from hydro_glasses import amorphous_tools as at



root='N10/'



dynmat=np.load(root+'second.npy')
dynmat=dynmat[0,:,:,0,:,:]/28.085 * units.mol / (10 * units.J)

n_modes=dynmat.shape[0]*dynmat.shape[1]
dynmat=dynmat.reshape([n_modes,n_modes])


Q_list = np.array(list(set(tuple(sorted(l)) for l in [[i, j, k] for i in range(12) for j in range(12) for k in range(12)])))
Q_list = np.delete(Q_list, np.argwhere((Q_list == [0, 0, 0]).all(axis = 1)), axis = 0)
Q_list = Q_list[np.argsort(np.linalg.norm(Q_list, axis = 1))]
Q_list = Q_list[np.unique(np.linalg.norm(Q_list, axis = 1), return_index = True)[1]]
atoms=read(root+'replicated_atoms.xyz')
positions = np.array([atoms.get_positions()])
pos = np.transpose(positions, axes = (0, 2, 1))
reciprocal_cell = atoms.cell.reciprocal()
print(reciprocal_cell)
ket_Q_L,ket_Q_T=at.compute_phi_Q(Q_list,reciprocal_cell,pos)



ket_Q_T=ket_Q_T.reshape([3*atoms.get_global_number_of_atoms(),Q_list.shape[0] ])
Q_array=np.array([2*np.pi*np.matmul(reciprocal_cell, Q_) for Q_ in Q_list])



eta=1
spectrum={}
omega_array=np.linspace(0.1,120,10000)
spectrum['omega']=omega_array
spectrum['Q']=Q_array
for iq,Q in enumerate(Q_list):
    phi=ket_Q_T[:,iq]
    #y=np.zeros_like(omega_array,dtype=complex)
    x,y=lanczos.spectrum(A=dynmat, v=phi, k=100,omega_array=omega_array,eta=eta)
    spectrum[iq]=y
np.save(root+'spectrum.npy',spectrum)





