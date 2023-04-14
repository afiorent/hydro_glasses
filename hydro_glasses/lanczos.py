import numpy as np
import matplotlib.pyplot as plt
#from ase.io import read,write
import opt_einsum as oe
#from scipy.optimize import curve_fit
import logging
def continued_fraction(a_coefficients, b_coefficients):
    """
    Calcola il valore di una frazione continua data la lista dei suoi coefficienti a e b.
    
    Args:
    a_coefficients: lista dei coefficienti a della frazione continua
    b_coefficients: lista dei coefficienti b della frazione continua
    
    Returns:
    Il valore della frazione continua
    """
    n = len(a_coefficients)
    if n == 0:
        return 0
    elif n == 1:
        return a_coefficients[0] + b_coefficients[0]
    else:
        return a_coefficients[0] + b_coefficients[0] / continued_fraction(a_coefficients[1:], b_coefficients[1:])



def recompute_spectrum(alpha,beta,z2):
    b2=compute_b2(beta)
    y=np.zeros(len(z2),dtype=complex)
    #z2=z**2
    for i,z2_ in enumerate(z2):
        y[i]=continued_fraction(np.insert(z2_-alpha,0,0),b2)
    return np.abs(np.imag(y))
def lanczos_cheap(A, v, k,last2=False):
    """
    Implementazione dell'algoritmo di Lanczos per una matrice simmetrica A e un vettore iniziale v.
    
    Args:
    A: matrice simmetrica di dimensione n x n
    v: vettore di dimensione n x 1
    k: numero di iterazioni
    
    Returns:
    alpha array, beta array
    if last2=True it returns also the last two vectors
    """
    n = A.shape[0]
    #T = np.zeros((k, k))
    alpha_array=np.zeros(k)
    beta_array=np.zeros(k)
    v = v / np.linalg.norm(v)
    v_minus=np.zeros_like(v,dtype=complex)
    for j in range(k):
        w =A@v # np.matmul(A, v,dtype=complex)
        alpha = np.real( np.vdot(v, w) ) 
        if j == k-1:
            break
        w = w - alpha * v - (beta * v_minus if j > 0 else 0)
        beta = np.linalg.norm(w)
        #print(beta)
        if beta == 0:
            break
        v_minus=v
        v=w / beta
        alpha_array[j] = alpha
        beta_array[j] = beta
    if not last2:
        return alpha_array, beta_array
    else:
        return alpha_array, beta_array, v, v_minus
def lanczos_restart(A, k,alpha_old, beta_old, v, v_minus,last2=False):
    """
    Implementazione dell'algoritmo di Lanczos per una matrice simmetrica A e un vettore iniziale v.
    
    Args:
    A: matrice simmetrica di dimensione n x n
    v: vettore di dimensione n x 1
    k: numero di iterazioni
    
    Returns:
    alpha array, beta array
    if last2=True it returns also the last two vectors
    """
    n = A.shape[0]
    #T = np.zeros((k, k))
    alpha_old=alpha_old[:-1]
    beta_old=beta_old[:-1]
    k_old=len(alpha_old)
    
    alpha_array=np.zeros(k+k_old)
    beta_array=np.zeros(k+k_old)
    alpha_array[:k_old]=alpha_old
    beta_array[:k_old]=beta_old
    
    v = v / np.linalg.norm(v)
    beta=beta_old[-1]
    for j in range(k_old,k_old+k):
        w =A@v # np.matmul(A, v,dtype=complex)
        alpha = np.real( np.vdot(v, w) ) 
        if j == k_old+k-1:
            break
        w = w - alpha * v - (beta * v_minus if j > 0 else 0)
        beta = np.linalg.norm(w)
        #print(beta)
        if beta == 0:
            break
        v_minus=v
        v=w / beta
        print(alpha,beta)
        alpha_array[j] = alpha
        beta_array[j] = beta
    if not last2:
        return alpha_array, beta_array
    else:
        return alpha_array, beta_array, v, v_minus

#def lanczos_cheap(A, v, k):
#    """
#    Implementazione dell'algoritmo di Lanczos per una matrice simmetrica A e un vettore iniziale v.
#    
#    Args:
#    A: matrice simmetrica di dimensione n x n
#    v: vettore di dimensione n x 1
#    k: numero di iterazioni
#    
#    Returns:
#    alpha array, beta array
#    """
#    n = A.shape[0]
#    #T = np.zeros((k, k))
#    alpha_array=np.zeros(k)
#    beta_array=np.zeros(k)
#    v = v / np.linalg.norm(v)
#    v_minus=np.zeros_like(v,dtype=complex)
#    for j in range(k):
#        w =A@v # np.matmul(A, v,dtype=complex)
#        alpha = np.real( np.vdot(v, w) ) 
#        if j == k-1:
#            break
#        w = w - alpha * v - (beta * v_minus if j > 0 else 0)
#        beta = np.linalg.norm(w)
#        #print(beta)
#        if beta == 0:
#            break
#        v_minus=v
#        v=w / beta
#        alpha_array[j] = alpha
#        beta_array[j] = beta
#
#    return alpha_array, beta_array#T, Q
def lanczos_ortho(A,u_in,nsteps):
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
    U=np.zeros((dimA, nsteps),dtype=complex)
    T=np.zeros((nsteps,nsteps))
    
    U[:,0]=u_in/np.linalg.norm(u_in)
    alpha=np.real(U[:,0].conj()@(A@U[:,0]) )
    T[0,0]=alpha
    W=A@U[:,0]-alpha*U[:,0]
    for i in range(1,nsteps):
        beta=np.linalg.norm(W)
        #modifica enri
        W=W/beta
        W=W-U@(U.transpose().conjugate()@W)
        U[:,i]=W
        #fine modifica
        alpha=np.real(U[:,i].conj()@ A@U[:,i] )
        W=A@U[:,i] - beta*U[:,i-1]- alpha*U[:,i]
#        U[:,i]=W/beta
#        alpha=np.real(U[:,i].conj()@ A@U[:,i] )
#        W=A@U[:,i] - beta*U[:,i-1]
#        #alpha=np.real(np.vdot(W,U[:,i]) )
#        W=W-U@(U.transpose().conjugate()@W)
        T[i,i]=alpha
        T[i-1,i]=beta
        T[i,i-1]=beta
    return T, U

def lanczos(A, v, k):
    """
    Implementazione dell'algoritmo di Lanczos per una matrice simmetrica A e un vettore iniziale v.
    
    Args:
    A: matrice simmetrica di dimensione n x n
    v: vettore di dimensione n x 1
    k: numero di iterazioni
    
    Returns:
    T: matrice tridiagonale di dimensione k x k
    Q: matrice di dimensione n x k, contenente i vettori di Lanczos
    """
    n = A.shape[0]
    Q = np.zeros((n, k),dtype=complex)
    T = np.zeros((k, k))
    
    Q[:, 0] = v / np.linalg.norm(v)
    
    for j in range(k):
        w = np.matmul(A, Q[:, j],dtype=complex)
        alpha = np.vdot(Q[:, j], w)
        if j == k-1:
            break
        w = w - alpha * Q[:, j] - (T[j, j-1] * Q[:, j-1] if j > 0 else 0)
        beta = np.linalg.norm(w)
        if beta == 0:
            break
        Q[:, j+1] = w / beta
        T[j, j] = alpha
        T[j, j+1] = beta
        T[j+1, j] = beta

    alpha_array=np.diagonal(T)
    beta_array=np.zeros(k)    
    beta_array[:-1]=np.diagonal(T,offset=1)
    return alpha_array, beta_array,Q#T, Q
    #return T,Q
def compute_b2(beta):
    b2=np.zeros(len(beta)+1,dtype=complex)
    b2[0]=1
    b2[1:]=-beta**2
    return b2

def spectrum(A, v, k,omega_array,eta,return_chain=False,use_ortho=False,norm_A=1):
    """
    Compute spectrum using lanczos method

    
    Args:
    A: matrice simmetrica di dimensione n x n
    v: vettore di dimensione n x 1
    k: numero di iterazioni
    omega_array: array di angular frequency
    eta: parte immaginaria di z=omega+i eta
    
    Returns:
    omega_array, spectrum
    """
    if not use_ortho:
        alpha,beta=lanczos_cheap(A,v,k)
    else:
        T, U=lanczos_ortho(A,v,nsteps=k)
        alpha=np.diagonal(T)
        beta=np.diagonal(T,offset=1)
        beta=np.append(beta,0)
        #logging.warning(alpha.shape)
        #logging.warning(beta.shape)
    alpha=alpha*norm_A
    beta=beta*norm_A
    b2=compute_b2(beta)
    y=np.zeros_like(omega_array,dtype=complex)
    for i,omega in enumerate(omega_array):
        z2=(omega+1j*eta)**2
        y[i]=continued_fraction(np.insert(z2-alpha,0,0),b2)
    if not return_chain:
        return np.abs(np.imag(y))
    else:
        return np.abs(np.imag(y)),alpha,beta



def convert_spectrum_lz2hydro(spectrum):
    """
    Convert the  spectrum computed through  lanczos method in a spectrum suitable for hydrodynamic tools
    Args:
    dictionary: spectrum
    Return:
    dictionary: spectrum
    """
    new_spectrum={}
    new_spectrum['omega']=spectrum['omega']
    new_spectrum['q']=np.linalg.norm(spectrum['Q'],axis=1 )
    new_spectrum['S']={}
    for ib,b in enumerate(['L','T']):
        new_spectrum['S'][b]=np.array( [spectrum[i][b]['S'] for i in range( len(new_spectrum['q']) ) ] )[:,:,np.newaxis]
    return new_spectrum
