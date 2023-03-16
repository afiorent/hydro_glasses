import numpy as np
import matplotlib.pyplot as plt
def lanczos(A,u_in,nsteps):
    '''
    Algoritmo di Lanczos per matrice simmetrica e reale e un vettore di input
    
    Args:
    A: matrice di cui trovare gli autovalori
    u_in: vettore di input
    nsteps: numero di step per l'algoritmo
    
    Return:
    T: matrice tridiagonale 
    U: matrice unitaria tale per cui U.T@A@U == T a meno di errore numerico
    '''
    if A.shape[0]==A.shape[1]:
        if u_in.shape[0]==A.shape[0]:
            pass
        else: 
            print(u_in.shape,A.shape[0])
            raise ValueError('il vettore e la matrice non hanno dimensioni compatibili')
    else:
        raise ValueError('A non è quadrata!')
    dimA=A.shape[0]
    U=np.zeros((dimA, nsteps))
    T=np.zeros((nsteps,nsteps))
    
    U[:,0]=u_in/np.linalg.norm(u_in)
    alpha=U[:,0]@A@U[:,0]
    T[0,0]=alpha
    W=A@U[:,0]-alpha*U[:,0]
    for i in range(1,nsteps):
        beta=np.linalg.norm(W)
        U[:,i]=W/beta
        alpha=U[:,i]@A@U[:,i]
        W=A@U[:,i] - alpha*U[:,i] - beta*U[:,i-1]
        T[i,i]=alpha
        T[i-1,i]=beta
        T[i,i-1]=beta
    return T, U

def compute_spectrum(A, u_in, nsteps, fmin=-20, fmax=20, nfreq=1000, eta=1.0e-3, plot=False):
    '''
    Calcola lo spettro in frequenza di 1/(z-A) calcolato su un vettore di input u_in
    '''
    T, U=lanczos(A, u, nsteps=nsteps)
    freq=np.linspace(fmin,fmax,nfreq) - 1j*eta*np.ones(nfreq)
    sp=((U.T@u)[ :]@np.linalg.inv(freq[:,np.newaxis, np.newaxis]*np.identity(nsteps)-T[:,:]))[:,0]
    if plot:
        f=plt.figure()
        plt.plot(np.real(freq), np.imag(sp), label='imaginary part')
        plt.legend()
        f=plt.figure()
        plt.plot(np.real(freq), np.real(sp), label='real part')
        plt.legend()
    return np.real(freq), sp
