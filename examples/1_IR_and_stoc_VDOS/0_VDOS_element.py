import numpy as np
import sys

from hydro_glasses.delta_kernel import compute_vdos

sys.path.append("../../hydro_glasses")
from hydro_glasses import vibrationalsystem
from hydro_glasses import vibrationalspectra
import matplotlib.pyplot as plt
import time

### Initialize Vibrational System and Spectra
if __name__ == '__main__':
    root='aSiO2_648/'
    omega_array=np.linspace(0.1,250,10000)
    nstoc=10## Number of stochastic vectors
    system=vibrationalsystem.VibrationalSystem(root=root)
    spectra=vibrationalspectra.\
        VibrationalSpectra(vibrational_system=system, options={'hl_steps':300,'eta':1.0,'omega_array':omega_array}) ### Initialize Vibrational Spectra object

    ### Compute Stochastic VDOS# of nstoc random vectors
    start=time.time()
    VDOS=spectra.compute_stochastic_vdos(nstoc=nstoc)
    end=time.time()
    print('Haydock wall time:{} s'.format(end-start) )

    ### Plotting
    prefactor_error=3

    w=spectra.options.omega_array
    y=np.mean(np.array([w*VDOS[i]['S'] for i in range(nstoc)]),axis=0)
    yerr=prefactor_error*np.std(np.array([w*VDOS[i]['S'] for i in range(nstoc)]),axis=0)/np.sqrt(nstoc)
    norm=np.trapz(y, w)
    y/=norm
    yerr/=norm

    #### Compute direct VDOS for comparison
    dynmat_dense=spectra.vs.dynmat.toarray()
    start=time.time()
    w2,eig=np.linalg.eigh(dynmat_dense)
    end=time.time()
    print('diagonalization time:{} s'.format(end-start) )
    eta=spectra.options.eta
    w_true=np.sign(w2)*np.sqrt(np.abs(w2))

    fig,ax=plt.subplots()

    ax.plot(w,y,label='stochastic n={}'.format(nstoc))
    ax.fill_between(w,y+yerr,y-yerr,alpha=0.5,label=f'{prefactor_error} $\sigma$')

    # vdos=compute_vdos(w_true,w,sigma=eta/2,kernel='lorentz')
    # ax.plot(w,vdos,'--',label='direct Lor.')
    vdos=compute_vdos(w_true,w,sigma=eta,kernel='DHO')
    ax.plot(w,vdos,'-.',label='direct') #DHO
    chemical_f=spectra.vs.atoms.get_chemical_symbols()[0]
    N=spectra.vs.atoms.get_global_number_of_atoms()
    plt.title(f'a{chemical_f} $N={N}$ $\eta={eta}$' )
    plt.legend()
    plt.ylabel('VDOS')
    plt.xlabel('$\omega\mathrm{~(rad/ps)}$')
    plt.show()

    ### Compute Stochastic VDOS per element
    w=spectra.options.omega_array
    VDOS_element={}
    for element in ['Si','O']:
        start=time.time()
        VDOS=spectra.compute_stochastic_vdos(nstoc=nstoc,element=element)
        end=time.time()
        print('Haydock wall time:{} s'.format(end-start) )
        y = np.mean(np.array([w * VDOS[i]['S'] for i in range(nstoc)]), axis=0)
        norm=np.trapz(y, w)
        y/=norm
        VDOS_element[element]=y
    ### Plotting
    w=spectra.options.omega_array


    fig,ax=plt.subplots()
    for element in ['Si','O']:
        y=VDOS_element[element]
        ax.plot(w,y,label='element {}'.format(nstoc,element))
    tot_vdos=VDOS_element['Si']*1+VDOS_element['O']*2
    ax.plot(w,tot_vdos,'-',label='total',color='black')
    ax.plot(w,vdos*3,'-.',label='total direct')
    plt.title(f'a{chemical_f} $N={N}$ $\eta={eta}$' )
    plt.legend()
    plt.ylabel('VDOS')
    plt.xlabel('$\omega\mathrm{~(rad/ps)}$')
    plt.show()