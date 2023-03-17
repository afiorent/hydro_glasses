import numpy as np
import matplotlib.pyplot as plt
def lanczos(A,u_in,nsteps):
    '''
    Lanczos algorithm for a symmetric matrix `A` and an initial vector `v`
    
    Args:
    A: n*n matrix
    u_in: n*1 input vector
    nsteps: number of iterations
    
    Return:
    T: n*n Tridiagonal matrix
    U: n*n Unitary matrix such that U.T@A@U == T 
    '''
    if A.shape[0]==A.shape[1]:
        if u_in.shape[0]==A.shape[0]:
            pass
        else: 
            print(u_in.shape,A.shape[0])
            raise ValueError('Incompatible matrix and vector shapes')
    else:
        raise ValueError('A must be a square matrix')
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

def compute_spectrum(A, u, nsteps, fmin=-20, fmax=20, nfreq=1000, eta=1.0e-3, plot=False):
    '''
    Compute the frequency spectrum of 1/(z-A) on a vector input `u_in`
    '''
    T, U = lanczos(A, u, nsteps=nsteps)
    freq = np.linspace(fmin, fmax, nfreq) - 1j*eta*np.ones(nfreq)
    sp=((U.T@u)[ :]@np.linalg.inv(freq[:,np.newaxis, np.newaxis]*np.identity(nsteps)-T[:,:]))[:,0]
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.real(freq), np.imag(sp), label = 'Im')
        ax.legend()
        fig, ax = plt.subplots()
        ax.plot(np.real(freq), np.real(sp), label = 'Re')
        ax.legend()
    return np.real(freq), sp
