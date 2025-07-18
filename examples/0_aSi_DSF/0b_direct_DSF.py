
import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe
import sys
sys.path.append('../../../hydro_glasses/')
from time import time
from ase.io import read,write
from ase import units
from hydro_glasses import lanczos,test_lanczos,hydrodynamic
from hydro_glasses import amorphous_tools as at
import os
import logging
from scipy import sparse
#multiprocessing
from multiprocessing import Pool
from scipy.optimize import curve_fit




root_spectrum='N1728/'
atoms = read(root_spectrum+'replicated_atoms.xyz')
positions = np.array([atoms.get_positions()])
pos = np.transpose(positions, axes = (0, 2, 1))
reciprocal_cell = np.linalg.inv(atoms.cell)
Lx=atoms.cell[0][0]
nq=8 #number of q-points #increase for larger systems

Q_list = np.array(list(set(tuple(sorted(l)) for l in [[i, j, k] for i in range(nq) for j in range(nq) for k in range(nq)])))
Q_list = np.delete(Q_list, np.argwhere((Q_list == [0, 0, 0]).all(axis = 1)), axis = 0)
Q_list = Q_list[np.argsort(np.linalg.norm(Q_list, axis = 1))]
Q_list = Q_list[np.unique(np.linalg.norm(Q_list, axis = 1), return_index = True)[1]]
N=atoms.get_global_number_of_atoms()
print(N)
nmodes=N*3
eigval=np.load(root_spectrum+'eigval.npy')
eigvec=np.load(root_spectrum+'eigvec.npy')



eta=1
Q, omega, S = at.compute_SQomega(Q_list, 
                                 np.array([eigval]), 
                                 np.array([eigvec]), 
                                 positions, 
                                 reciprocal_cell,
                                 cutoff = 120/2/np.pi, # THz
                                 domega = eta, # rad/ps
                                 nomega = 20000,
                                 use_soft = False,#True include the few modes with negative frequency, if there are any
                                 is_anharmonic = False,
                                vectorize=True)#for debugging set false
qnorm=np.linalg.norm(Q,axis=1)
S_Qw = {'Q': Q, 'omega': omega, 'S': S}
np.save(root_spectrum +'/S_Qw_har_direct.eta{}.no2pi.npy'.format(eta), S_Qw)







