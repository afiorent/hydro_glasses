import numpy as np
from pathlib import Path
import sys
import pandas as pd
from scipy.sparse import save_npz, csr_matrix
from ase.io import read,write
atoms=read('path/to/replicated_atoms.xyz'.format(j))
n=atoms.get_global_number_of_atoms()
print(n)
infile = './path/to/Dyn.form'.format(j)
outfile ='./path/for/dynmat.npz'.format(j)
mat = pd.read_csv(infile, 
                  sep = '\s+',
                  usecols = [0, 1, 2],
                  names = ['1', '2', '3']).to_numpy()
mat = csr_matrix(mat.reshape(3*n,3*n))
save_npz(outfile, mat)
