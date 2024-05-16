import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe
import sys
sys.path.append('/path/to/hydro_glasses/')
from time import time
from ase.io import read,write
from ase import units
from hydro_glasses import lanczos
from hydro_glasses import amorphous_tools as at
import os
import logging
logging.basicConfig(level=logging.DEBUG)
from scipy import sparse

#multiprocessing
from multiprocessing import Pool


root='./N14/'
k=200
eta=1
#THIS IS ALL POST PROCESSING
spectrum=np.load(root+'spectrum_k{}_eta{}.npy'.format(k,eta),allow_pickle=True).item()
q=spectrum['Q']
q=np.linalg.norm(q,axis=1)

ncpus=os.cpu_count()
logging.info('ncpus:{}'.format(ncpus) )

S_qw_T={}
omega_array=spectrum['omega']

#recover the splines
root_int='/path/to/anharmonic/bandwidth_splines/'
for T in [100,200,300,400,500]:
    spl=np.load(root_int+'spline.N3.T{}K.npy'.format(T),allow_pickle=True).item()
    eta_array=spl(omega_array/2/np.pi)
    z=(omega_array+1j*eta_array+1j*eta )

    start=time()
    with Pool(ncpus) as p:
        inputs = [(spectrum[iq]['T']['alpha'],spectrum[iq]['T']['beta'],z) for iq in range(len(q)) ]
        result_T = p.starmap(lanczos.recompute_spectrum, inputs)
    end=time()
    logging.info(end-start)
    
    start=time()
    with Pool(ncpus) as p:
        inputs = [(spectrum[iq]['L']['alpha'],spectrum[iq]['L']['beta'],z) for iq in range(len(q)) ]
        result_L = p.starmap(lanczos.recompute_spectrum, inputs)
    end=time()
    logging.info(end-start)

    S_qw_T[T]=np.array([result_L,result_T])




S_qw_T['omega']=omega_array
S_qw_T['Q']=q
np.save(root+'spectrum_T_k{}_eta{}.npy'.format(k,eta),S_qw_T)
