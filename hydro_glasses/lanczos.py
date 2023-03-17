import numpy as np
import matplotlib.pyplot as plt
#from ase.io import read,write
import opt_einsum as oe
#from scipy.optimize import curve_fit

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



def lanczos_cheap(A, v, k):
    """
    Implementazione dell'algoritmo di Lanczos per una matrice simmetrica A e un vettore iniziale v.
    
    Args:
    A: matrice simmetrica di dimensione n x n
    v: vettore di dimensione n x 1
    k: numero di iterazioni
    
    Returns:
    alpha array, beta array
    """
    n = A.shape[0]
    #T = np.zeros((k, k))
    alpha_array=np.zeros(k)
    beta_array=np.zeros(k)
    v = v / np.linalg.norm(v)
    v_minus=np.zeros_like(v,dtype=complex)
    for j in range(k):
        w = np.matmul(A, v,dtype=complex)
        alpha = np.vdot(v, w)
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

    return alpha_array, beta_array#T, Q



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
    b2[0]=1+0*1j
    b2[1:]=-beta**2
    return b2

def spectrum(A, v, k,omega_array,eta):
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
    alpha,beta=lanczos_cheap(A,v,k)
    b2=compute_b2(beta)
    y=np.zeros_like(omega_array,dtype=complex)
    for i,omega in enumerate(omega_array):
        z2=omega**2+1j*eta
        y[i]=continued_fraction(np.insert(z2-alpha,0,0),b2)
    return omega_array, np.abs(np.imag(y))

