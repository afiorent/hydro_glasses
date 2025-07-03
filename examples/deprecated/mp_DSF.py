import numpy as np
#import matplotlib.pyplot as plt
import opt_einsum as oe
import sys
sys.path.append('/path/to/hydro_glasses/')
from time import time
from ase.io import read,write
from ase import units
from hydro_glasses import lanczos,test_lanczos
from hydro_glasses import amorphous_tools as at
#import tensorflow as tf
import logging
import os
from scipy import sparse
#multiprocessing
from multiprocessing import Pool

root='N10/'



dynmat=np.load(root+'second.npy')
dynmat=dynmat[0,:,:,0,:,:]/28.085 * units.mol / (10 * units.J)

n_modes=dynmat.shape[0]*dynmat.shape[1]
dynmat=dynmat.reshape([n_modes,n_modes])
dynmat=(dynmat+dynmat.T)/2
dynmat_sparse=sparse.csr_matrix(dynmat)
#dynmat_tf = tf.sparse.from_dense(np.complex128(dynmat) )

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
k=100
#spectrum['omega']=omega_array
#spectrum['Q']=Q_array

ncpus=os.cpu_count()
print('ncpus:',ncpus)
if ncpus>20:
    ncpus=12
start=time()
with Pool(ncpus) as p:
    inputs = [(dynmat_sparse,ket_Q_T[:,i],k,omega_array,eta) for i in range(ket_Q_T.shape[1]) ]
    result = p.starmap(lanczos.spectrum, inputs)
end=time()
print(end-start)
result=np.array(result)
np.save(root+'result.npy',result)






