import numpy as np
import sys
sys.path.append("../../hydro_glasses")
from hydro_glasses import vibrationalsystem
from hydro_glasses import vibrationalspectra
import matplotlib.pyplot as plt
import time
sys.setrecursionlimit(2000)

### Initialize Vibrational System and Spectra

root='N1728'
system=vibrationalsystem.VibrationalSystem(root=root)
start=time.time()
spectra=vibrationalspectra.VibrationalSpectra(vibrational_system=system,options={'hl_steps':200,'eta':1.0})
spectrum=spectra.compute_vdsf(nq=8)
end=time.time()
print('diagonalization time:{} s'.format(end-start) )

### Plotting
n_modes=2  # Longitudinal and Transverse
q=spectrum['Q']
q=np.linalg.norm(q,axis=1)
w=spectrum['omega']
S_qw=np.zeros((n_modes,len(q),len(w)))
for ib,b in enumerate(['L','T']):
    for iq in range(len(q)):
        S_qw[ib,iq,:]=spectrum[iq][b]['S']

fig,axes=plt.subplots(ncols=2,sharey=True)
vmax=1e-2
for imode in range(n_modes):
    ax=axes[imode]
    p=ax.pcolormesh(q,w/2/np.pi,w[:,np.newaxis]/2/np.pi*S_qw[imode,:,:].T,
                          shading = 'nearest',
                          vmin = 0,
                          vmax = vmax,
                          cmap = 'inferno'
                 )
    dispersion=w[np.argmax(w[np.newaxis,:]*S_qw[imode,:,:],axis=1)] ### it should give the acoustic dispersion in the lower frequency part
    # print(T_dispersion.shape)
    ax.plot(q,dispersion/2/np.pi,'--',color='b')
    if imode==1:
        fig.colorbar(p, ax=ax)
fig.suptitle('$S(q,\\nu)$')
plt.show()
