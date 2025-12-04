import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../hydro_glasses')
from time import time
from ase.io import read,write
from ase import units
from hydro_glasses import lanczos,test_lanczos,hydrodynamic
from hydro_glasses import vibrationalsystem
from hydro_glasses import vibrationalspectra
from hydro_glasses.delta_kernel import compute_kernel

if __name__ == '__main__':
    root='aSiO2_648/'
    omega_array=np.linspace(0.1,250,10000)
    system=vibrationalsystem.VibrationalSystem(root=root) ### Read a dynamical matrix and an ASE atoms object
    spectra=vibrationalspectra.\
        VibrationalSpectra(vibrational_system=system,options={'hl_steps':300,'eta':1.0,'omega_array':omega_array}) ### Initialize Vibrational Spectra object
    start=time()
    spectrum=spectra.compute_IR_spectrum(polarizations=[0,1,2],charges_dict={'Si':3.2,'O':-1.6})
    end=time()
    print('HL wall time in seconds:',end-start)


    thz2cm1=33.35641
    # Plotting IR spectrum
    ###Comparison haydock vs direct method


    dynmat_dense=spectra.vs.dynmat.toarray()
    w2,eig=np.linalg.eigh(dynmat_dense)

    ### generate vectors for IR
    from hydro_glasses.amorphous_tools import make_IR_vector
    atoms=spectra.vs.atoms
    phi=make_IR_vector(atoms=atoms,polarizations=[0,1,2],charges_dict={'Si':3.2,'O':-1.6})


    w_true=np.sign(w2)*np.sqrt(np.abs(w2))
    fig,ax=plt.subplots()
    for i, pol in enumerate(['x']):
        #### Here we need the eigenvectors 'eig' and the IR vectors 'phi'
        Z_mu=np.einsum('i,in->n',phi[i],eig)
        print(Z_mu.shape)
        y=np.abs(Z_mu)**2/w_true
        x=spectra.options.omega_array
        y=compute_kernel(w_true, y, x, sigma=spectra.options.eta, kernel='lorentz', normalize=True)
        ax.plot(x*thz2cm1/2/np.pi, y*2*np.pi/thz2cm1, label='IR direct pol. {}'.format(pol),marker='.')
        ### Haydock result
        y=spectrum[i]['S']
        norm=np.trapz(y, x)
        y/=norm
        ax.plot(x*thz2cm1/2/np.pi,y*2*np.pi/thz2cm1,label='IR Haydock pol. {}'.format(pol))
    ax.set_xlabel('Frequency (cm$^{-1}$)')
    ax.set_ylabel('IR (a.u.)')
    ax.set_xlim(0,1300)
    ax.set_yticks([])
    plt.legend()
    plt.show()
