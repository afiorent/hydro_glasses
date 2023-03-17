import numpy as np
import matplotlib.pyplot as plt
#from ase.io import read,write
import opt_einsum as oe
#from scipy.optimize import curve_fit

def continued_fraction(a_coefficients, b_coefficients):
    """
    Compute the value of a continued fraction from the lists of coefficients `a` and `b`
    
    Args:
    a_coefficients: list of coefficients `a`
    b_coefficients: list of coefficients `b`
    
    Returns:
    The value of the continued fraction
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
    Lanczos algorithm for a symmetric matrix `A` and an initial vector `v`
    
    Args:
    A: n * n (float) Symmetric matrix
    v: n * 1 (float) vector
    k: (int) number of iterations
    
    Returns:
    alpha array, beta array
    """
    n = A.shape[0]
    #T = np.zeros((k, k))
    alpha_array = np.zeros(k)
    beta_array = np.zeros(k)
    v = v / np.linalg.norm(v)
    v_minus = np.zeros_like(v, dtype = complex)
    for j in range(k):
        w = np.matmul(A, v, dtype = complex)
        alpha = np.vdot(v, w)
        if j == k-1:
            break
        w = w - alpha * v - (beta * v_minus if j > 0 else 0)
        beta = np.linalg.norm(w)
        #print(beta)
        if beta == 0:
            break
        v_minus = v
        v = w / beta
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





