import numpy as np
from scipy import sparse
from ase.io import read,write
root='N1728/'
atoms=read(root+'replicated_atoms.xyz')
N=atoms.get_global_number_of_atoms()
print(N)
dynmat_sparse=sparse.load_npz(root+'dynmat.npz')
dynmat_sparse=(dynmat_sparse+dynmat_sparse.transpose())/2 #sometimes it is not symmetrical for numerical reasons
dynmat_dense=dynmat_sparse.toarray()
print(dynmat_dense.shape)
dynmat_dense=dynmat_dense.reshape([3*N,3*N])
w2,eigvec=np.linalg.eigh(dynmat_dense)
np.save(root+'eigval.npy',w2)
np.save(root+'eigvec.npy',eigvec)
