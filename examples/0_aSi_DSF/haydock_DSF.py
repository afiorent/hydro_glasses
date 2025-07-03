import numpy as np
import opt_einsum as oe
import sys
sys.path.append('../hydro_glasses/')
from time import time
from ase.io import read,write
from ase import units
from hydro_glasses import lanczos,test_lanczos
from hydro_glasses import amorphous_tools as at
import logging
logging.basicConfig(level=logging.DEBUG)
import os
from scipy import sparse
#multiprocessing
from multiprocessing import Pool
#raise length limit for lanczos chain #default is 999
sys.setrecursionlimit(2000)

#User inputs: number of lanczos steps, ortho Lanczos, root directory

k=200#lanczos step
ortho=False#True slows down the code, but it makes the algorithm more robust
#ortho may reduce the number of steps for convergence. The scaling goes from O(Nk)-> O(Nk^2)
if ortho:
    print('using ortho')
else:
    print('not using ortho')
root='N1728/' #default root

use_tril=False #use lower triangular part of the matrix to symmetrize it. If False, it symmetrizes the matrix by averaging upper and lower triangular parts.
logging.info('root= '+str(root) )
atoms=read(root+'replicated_atoms.xyz')
Lx=atoms.cell[0][0]
print(Lx)
nq=10 #number of q-points
mass = atoms.get_masses()
print('mass=',np.mean(mass), 'nq=',nq)
#loading the dynmat as a sparse matrix
try:
    dynmat_sparse=sparse.load_npz(root+'dynmat.npz')
    dynmat_sparse=(dynmat_sparse+dynmat_sparse.transpose())/2 #sometimes it is not symmetrical for numerical reasons
except:
    dynmat=np.load(root+'second.npy')
    dynmat =dynmat* 1 / np.sqrt(mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
    dynmat = dynmat * 1 / np.sqrt(mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
    n_modes=dynmat.shape[1]*dynmat.shape[2]
    logging.info('n modes= '+str( n_modes) )
    dynmat *= units.mol / (10 * units.J)
    dynmat=dynmat.reshape([n_modes,n_modes])
    if use_tril: #it symmetrizes by replicating the lower triangular part of the matrix
        dynmat=np.tril(dynmat)+np.tril(dynmat,-1).T
    else:
        dynmat=(dynmat+dynmat.T)/2
    dynmat_sparse=sparse.csr_matrix(dynmat)
    sparse.save_npz(root+'dynmat.npz',dynmat_sparse)

#generation of the acoustic plane-wave basis
Q_list = np.array(list(set(tuple(sorted(l)) for l in [[i, j, k] for i in range(nq) for j in range(nq) for k in range(nq)])))
Q_list = np.delete(Q_list, np.argwhere((Q_list == [0, 0, 0]).all(axis = 1)), axis = 0)
Q_list = Q_list[np.argsort(np.linalg.norm(Q_list, axis = 1))]
Q_list = Q_list[np.unique(np.linalg.norm(Q_list, axis = 1), return_index = True)[1]]
positions = np.array([atoms.get_positions()])
pos = np.transpose(positions, axes = (0, 2, 1))
reciprocal_cell = np.linalg.inv(atoms.cell)
ket_Q_L,ket_Q_T=at.compute_phi_Q(Q_list,reciprocal_cell,pos)

print(reciprocal_cell,ket_Q_T.shape)


ket_Q_T=ket_Q_T.reshape([3*atoms.get_global_number_of_atoms(),Q_list.shape[0] ])
ket_Q_L=ket_Q_L.reshape([3*atoms.get_global_number_of_atoms(),Q_list.shape[0] ])
Q_array=np.array([2*np.pi*np.matmul(reciprocal_cell, Q_) for Q_ in Q_list])


##HAYDOCK ALGORITHM
#eta and omega array can be recomputed in post-processing with negligible cost
eta=1
#ortho=False
spectrum={}
omega_array=np.linspace(0.1,120,20000)
spectrum['omega']=omega_array
spectrum['Q']=Q_array

ncpus=os.cpu_count()
logging.info('ncpus:{}'.format(ncpus) )
### There is no a priori perfect number of CPUs for this task.
### Each process should have enough memory to store the Lanczos chain and the Hamiltonian matrix. For small systems, it may use nprocess=ncpus.
if ncpus>20:
    ncpus=8
start=time()
with Pool(ncpus) as p:
    inputs = [(dynmat_sparse,ket_Q_L[:,i],k,omega_array,eta,True,ortho) for i in range(ket_Q_L.shape[1]) ]
    result_L = p.starmap(lanczos.spectrum, inputs)
end=time()
logging.info('first polarization done in {} s'.format(end-start) )
start=time()
with Pool(ncpus) as p:
    inputs = [(dynmat_sparse,ket_Q_T[:,i],k,omega_array,eta,True,ortho) for i in range(ket_Q_T.shape[1]) ]
    result_T = p.starmap(lanczos.spectrum, inputs)
end=time()
logging.info('second polarization done in {} s'.format(end-start) )
result=[result_L,result_T]


#saving
for iq in range(Q_array.shape[0]):
    spectrum[iq]={}
    for ib,b in enumerate(['L','T']):
        spectrum[iq][b]={}
        spectrum[iq][b]['S']=result[ib][iq][0]
        spectrum[iq][b]['alpha']=result[ib][iq][1]
        spectrum[iq][b]['beta']=result[ib][iq][2]

if ortho:
    np.save(root+'spectrum_ortho_k{}_eta{}.npy'.format(k,eta),spectrum)
else:
    np.save(root+'spectrum_k{}_eta{}.npy'.format(k,eta),spectrum)






