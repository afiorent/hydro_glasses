import numpy as np
from scipy.integrate import trapz
from itertools import product
import opt_einsum as oe

if __name__ == '__main__':
    pass

def compute_Q_dot_product(Q_list, positions, reciprocal_cell):
    """
    Computes the dot product between <Q|Q'> pseudo-plane-wave states.
    Q_list.

            Parameters:
                Q_list (int): list of triplets of integers that correspond to the Q-points in reciprocal space.
                positions (float): list of positions of the atoms
                reciprocal_cell (float): 3x3 matrix with the reciprocal cell 

            Returns:
                Q_dot_product (float): dictionary with the longitudinal ('L') and transverse ('T') components of the Q.R products, one for each snapshot.
    """
    natm = positions.shape[1]
    exp_i_Q_dot_R_list={}
    for Q_ in Q_list:
        lbl = str(list(Q_))
        Q = 2*np.pi*reciprocal_cell@Q_
        exp_i_Q_dot_R = np.exp(1j*np.einsum('a,tIa->tI', Q, positions))/np.sqrt(natm)
        exp_i_Q_dot_R_list[lbl] = exp_i_Q_dot_R

    Q_dot_product = {}

    for Q1, Q2 in product(Q_list, Q_list):
        lbl1 = str(list(Q1))
        lbl2 = str(list(Q2))
        e1 = exp_i_Q_dot_R_list[lbl1]
        e2 = exp_i_Q_dot_R_list[lbl2]

        if lbl1 not in Q_dot_product:
            Q_dot_product[lbl1] = {}
        Q_dot_product[lbl1][lbl2] =  np.einsum('tI,tI->t', e1, e2.conj())
    return Q_dot_product
    
def inv_participation_ratio(eigvec):
    """Compute the inverse participation ratio per mode `n`:
                        1/p_n = \sum_{i} |<i|n>|^4
        where |i> is the i-th element of the canonical basis, i.e., |i> = (0, 0, ...,   0, 1,   0, ..., 0)
                                                                           1, 2, ..., i-1, i, i+1, ..., N)
    """
    return np.sum(eigvec**4, axis = 1)

def compute_braQketn(Q_list, reciprocal_cell, pos, reshaped_eigvec, weights = None):
    #if len(reciprocal_cell.shape) > 1:
    #    Q = np.array([[2*np.pi*np.matmul(reciprocal_cell[i], Q_) for Q_ in Q_list] for i in range(reciprocal_cell.shape[0])])
    #    print('Q shape = ', Q.shape)
    #    Qn = oe.contract('tqa,tq->tqa', Q, 1/np.linalg.norm(Q, axis = 2))
    #    #Qn = np.einsum('qa,q->qa', Q, 1/np.linalg.norm(Q, axis = 1))
    #    
    #    eperp = np.random.rand(*Q.shape)
    #    eperp -= np.array([[Q_[t]*eperp_[t].dot(Q_[t]) for eperp_, Q_ in zip(eperp, Qn)] for t in range(Q_.shape[0])])
    #    eperp = oe.contract('tqa,tq->tqa', eperp, 1/np.linalg.norm(eperp, axis = 2))
    #    
    #    if weights is not None:
    #        reshaped_eigvec = oe.contract('tani,ti->tani', reshaped_eigvec, weights)
    #    #exp_i_Q_dot_R = np.array([np.exp(1j*np.transpose(Q@pos, axes = (0, 2, 1))) for t in range(Q.shape[0])]) #/ np.sqrt(3*natm) # 'qa,taI->tIq'
    #    exp_i_Q_dot_R = np.exp(1j*oe.contract('tqa,taI->tIq', Q, pos))
    #    bran_ketQ = np.array([r@e for r, e in zip(reshaped_eigvec, exp_i_Q_dot_R)]) # 'tanI,tIq->tanq'
    #    bran_ketQ_sq_L = np.abs(oe.contract('qa,tanq->tqn', Qn, bran_ketQ))**2
    #    bran_ketQ_sq_T = np.abs(oe.contract('qa,tanq->tqn', eperp, bran_ketQ))**2
    #    return Q, bran_ketQ_sq_L, bran_ketQ_sq_T
    #else:
    Q = np.array([2*np.pi*np.matmul(reciprocal_cell, Q_) for Q_ in Q_list])
    Qn = oe.contract('qa,q->qa', Q, 1/np.linalg.norm(Q, axis = 1))
    
    eperp = np.random.rand(*Q.shape)
    eperp -= np.array([Q_*eperp_.dot(Q_) for eperp_, Q_ in zip(eperp, Qn)])
    eperp = oe.contract('qa,q->qa', eperp, 1/np.linalg.norm(eperp, axis = 1))
   
    natm = pos.shape[2]
    if weights is not None:
        reshaped_eigvec = oe.contract('tani,ti->tani', reshaped_eigvec, weights)
    exp_i_Q_dot_R = np.exp(1j*np.transpose(Q@pos, axes = (0, 2, 1))) / np.sqrt(3*natm) # 'qa,taI->tIq'
    bran_ketQ = np.array([r@e for r, e in zip(reshaped_eigvec, exp_i_Q_dot_R)]) # 'tanI,tIq->tanq'
    bran_ketQ_sq_L = np.abs(oe.contract('qa,tanq->tqn', Qn, bran_ketQ))**2
    bran_ketQ_sq_T = np.abs(oe.contract('qa,tanq->tqn', eperp, bran_ketQ))**2
    return Q, bran_ketQ_sq_L, bran_ketQ_sq_T

def compute_SQomega(Q_list,
                    eigval,
                    eigvec,
                    positions,
                    reciprocal_cell,
                    cutoff = None,
                    nomega = 1000,
                    use_soft = True,
                    domega = 0.01,
                    is_anharmonic = False,
                    gammas = None,
                    vectorize = True,
                    print_timing = False):
    """Compute the dynamic structure factor."""
    
    def lorentzian(omega, gamma):
        return gamma/np.pi/(gamma**2 + omega**2)

    if is_anharmonic and gammas is None:
        raise ValueError('Anharmonic linewidths are required to compute the anharmonic dynamic structure factor!')
    
    #########################################################################################################
    if print_timing:
        from timeit import default_timer
        print('Compute initial stuff', end=' ')
        start = start_ = default_timer()        
    
    SQ = {}
        
    T = positions.shape[0]
    natm = positions.shape[1]
   
    # Position vector
    pos = np.transpose(positions, axes = (0, 2, 1))
    reshaped_eigvec = np.transpose(eigvec.reshape(T, natm, 3, 3*natm), axes = (0, 2, 3, 1)) # new shape is (T, 3, 3*natm, natm)
    
    # Angular frequencies for each mode
    omegas = np.sqrt(np.abs(eigval))*np.sign(eigval)

    if cutoff is not None:
        mask = omegas.flatten() <= 2*np.pi*cutoff # the value of cutoff is in THz, while omega in rad/ps
    #    print(mask.shape)
    #    print(gammas.shape)
    #    print(reshaped_eigvec.shape)
        omegas = omegas[:, mask]
        reshaped_eigvec = reshaped_eigvec[:, :, mask, :]
        if is_anharmonic:
            gammas = gammas[:, mask]

    if not use_soft:
        # Do not consider soft modes
        omegas = omegas[:,3:]
        reshaped_eigvec = reshaped_eigvec[:, :, 3:, :]
        if is_anharmonic:
            gammas = gammas[:,3:]
    
    omin = 0 #np.min(omegas)
    omax = np.max(omegas)
    #domega = domega*(omax-omin)
    omega_min = (omax-omin)/nomega
    
    # For testing purposes
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))

    #########################################################################################################
    
    # Compute the projections of the normal modes onto amorphous plane waves for the longitudinal and transverse branches
    Q, bran_ketQ_sq_L, bran_ketQ_sq_T = compute_braQketn(Q_list, reciprocal_cell, pos, reshaped_eigvec)
    
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))
        
    if print_timing:
        print('Begin loop over omega...', end=' ')
        start = end = default_timer()

    # The angular frequency output domain
    omega_domain = np.linspace(omin, omax, nomega)

    if vectorize:
        # Optimized calculation
        if is_anharmonic:
            # The "delta function" centered on the mode angular frequency is a lorentzian with width equal to the mode anharmonic bandwidth
            #delta = lorentzian(np.array([omega_domain[:,np.newaxis] - om[np.newaxis, :] for om in omegas]), 2*np.pi*gammas) # shape = T x nomega x nmodes
            delta = lorentzian(np.array([omega_domain[:,np.newaxis] - om[np.newaxis, :] for om in omegas]), gammas) # shape = T x nomega x nmodes
        else:
            # The "delta function" is a smeared Dirac-delta, chosen to be a lorentzian with fixed width
            delta = lorentzian(np.array([omega_domain[:,np.newaxis] - om[np.newaxis, :] for om in omegas]), domega) # shape = T x nomega x nmodes

        # Compute the sum over modes
        SQ['L'] = oe.contract('tqn,twn->tqw', bran_ketQ_sq_L, delta)
        SQ['T'] = oe.contract('tqn,twn->tqw', bran_ketQ_sq_T, delta)
    else:
        # Unoptimized calculation (for debug)
        SQ['L'] = np.zeros((T, Q.shape[0], nomega))
        SQ['T'] = np.zeros((T, Q.shape[0], nomega))
        loop = 0
        for i, omega in enumerate(omega_domain):

            # if not use_soft:
            #     where = np.logical_and(np.logical_and(omegas>omega-domega, omegas<=omega+domega), omegas>0)
            # else:
            #     where = np.logical_and(omegas>omega-domega, omegas<=omega+domega)
            # delta = np.zeros(eigval.shape)
            # delta[where] = 1
            
            delta = lorentzian(omega-omegas, domega)

            SQ['L'][:, :, i] = oe.contract('tqn,tn->tq', bran_ketQ_sq_L, delta)
            SQ['T'][:, :, i] = oe.contract('tqn,tn->tq', bran_ketQ_sq_T, delta)
            
            if print_timing:
                print('{}'.format(i), end=' ')
                ti = default_timer()
                if loop:
                    loop = 0.5*(loop + ti - end)
                else:
                    loop = ti - end
                end = ti
            
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))
        print('Avg step performed in {} s'.format(loop))
        
    if print_timing:
        print('Normalize SQ...', end=' ')
        start = end = default_timer()

    # Normalize the spectrum along omega
    for branch in SQ:
        inverse_norms = 1 # /trapz(SQ[branch], omega_domain, axis = 2)
        SQ[branch] = np.transpose(np.array([SQ[branch][:, :, i]*inverse_norms for i in range(SQ[branch].shape[2])]), axes = (2, 0, 1))
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))
        print('Total elapsed time = {} s'.format(end - start_))
        
    # Return the Q vectors, the angular frequency, and the dynamical structure factor    
    return Q, omega_domain, SQ


def compute_SQQpomega(Q_list, Qp_list,
                    eigval,
                    eigvec,
                    positions,
                    reciprocal_cell,
                    nomega = 100, use_soft = True, domega = 0.01, print_timing = False):
    
    from scipy.integrate import trapz
    from itertools import product
    
    
    #########################################################################################################
    if print_timing:
        from timeit import default_timer
        print('Compute initial stuff', end=' ')
        start = timeit.default_timer()
    
    SQ = {}
        
    T = positions.shape[0]
    natm = positions.shape[1]
    
    reshaped_eigvec = eigvec.reshape(T, natm, 3, 3*natm)
    
    omegas = np.sqrt(np.abs(eigval))*np.sign(eigval)
            
    omin = np.min(omegas)
    omax = np.max(omegas)
    domega = domega*(omax-omin)
    omega_min = (omax-omin)/nomega
    
    # For testing purposes
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))

    #########################################################################################################
    
    Q = np.array([2*np.pi*np.matmul(reciprocal_cell, Q_) for Q_ in Q_list])
    Qn = oe.contract('qa,q->qa', Q, 1/np.linalg.norm(Q, axis = 1))
    eperp = np.random.rand(*Q.shape)
    eperp -= np.array([Q_*eperp_.dot(Q_) for eperp_, Q_ in zip(eperp, Qn)])
    eperp = oe.contract('qa,q->qa', eperp, 1/np.linalg.norm(eperp, axis = 1))
    
    Qp = np.array([2*np.pi*np.matmul(reciprocal_cell, Q_) for Q_ in Qp_list])
    Qpn = oe.contract('qa,q->qa', Qp, 1/np.linalg.norm(Qp, axis = 1))
    eperpp = np.random.rand(*Q.shape)
    eperpp -= np.array([Q_*eperp_.dot(Q_) for eperp_, Q_ in zip(eperpp, Qpn)])
    eperpp = oe.contract('qa,q->qa', eperpp, 1/np.linalg.norm(eperpp, axis = 1))

    # Aggiornato fino a qui
    if weights is not None:
        reshaped_eigvec = oe.contract('tani,ti->tani', reshaped_eigvec, weights)
    exp_i_Q_dot_R = np.exp(1j*np.transpose(Q@pos, axes = (0, 2, 1))) #/ np.sqrt(3*natm) # 'qa,taI->tIq'
    bran_ketQ = np.array([r@e for r, e in zip(reshaped_eigvec, exp_i_Q_dot_R)]) # 'tanI,tIq->tanq'
    bran_ketQ_sq_L = np.abs(oe.contract('qa,tanq->tqn', Qn, bran_ketQ))**2
    bran_ketQ_sq_T = np.abs(oe.contract('qa,tanq->tqn', eperp, bran_ketQ))**2
    # Copiato fino a qui da SQomega
    
    #>>>> da aggiornare
    for Q_, Qp_ in product(Q_list, Qp_list):
        if print_timing:
            print('Compute bran_ketQ...', end = ' ')
            start = default_timer()
            
        lbl = '{}-{}'.format(str(Q_), str(Qp_))
        
        Q  = 2*np.pi*np.matmul(reciprocal_cell, Q_)
        Qp = 2*np.pi*np.matmul(reciprocal_cell, Qp_)
        
        exp_i_Q_dot_R    = np.exp( 1j*np.einsum('a,tIa->tI', Q,  positions))/np.sqrt(natm)
        exp_i_Qp_dot_R_c = np.exp(-1j*np.einsum('a,tIa->tI', Qp, positions))/np.sqrt(natm)
        
        bran_ketQ    = np.einsum('tIan,tI->tan', reshaped_eigvec, exp_i_Q_dot_R)
        bran_ketQp_c = np.einsum('tIan,tI->tan', reshaped_eigvec, exp_i_Qp_dot_R_c)
                
        bran_ketQ_sq = np.einsum('tan,tan->tn', bran_ketQ, bran_ketQp_c)
        
        # For testing purposes 
        if print_timing:
            end = default_timer()
            print('Done in {} s'.format(end - start))        
    
        SQ[lbl] = np.zeros((T, nomega), dtype = np.complex128)
        
        for i, omega in enumerate(np.linspace(omin, omax, nomega)):
            
            # For testing purposes
            if print_timing:
                print('Compute {}th omega loop...'.format(i), end = ' ')
                start = default_timer()
            
            if not use_soft:
                where = np.logical_and(np.logical_and(omegas>omega-domega, omegas<=omega+domega), omegas>0)
            else:
                where = np.logical_and(omegas>omega-domega, omegas<=omega+domega)
            delta = np.zeros(eigval.shape)
            delta[where] = 1
            
            SQ[lbl][:, i] = np.einsum('tn,tn->t', bran_ketQ_sq, delta)

            # For testing purposes
            if print_timing:
                end = default_timer()
                print('Done in {} s'.format(end - start))

        # norms = np.array([trapz(SQ[lbl][t, :], np.linspace(omin, omax, nomega)) for t in range(T)])
        # SQ[lbl] = np.array([s/n for s, n in zip(SQ[lbl], norms)])
        
    return np.linspace(omin, omax, nomega), SQ

def v2_matrix(phonons, cutoff_n = 3000, nomega = 1000, delta_width = 0.2, vectorize = True, discrete_delta = False, normalize = False):

    from kaldo.observables.harmonic_with_q import HarmonicWithQ
    
    q_points = phonons._reciprocal_grid.unitary_grid(is_wrapping=False)
    frequency = np.zeros((phonons.n_k_points, phonons.n_modes))
    for ik in  range(len(q_points)):
        q_point = q_points[ik]
        sij_x = HarmonicWithQ(q_point=q_point,
                               second=phonons.forceconstants.second,
                               distance_threshold=phonons.forceconstants.distance_threshold,
                               folder=phonons.folder,
                               storage=phonons.storage,
                               is_nw=phonons.is_nw,
                               is_unfolding=phonons.is_unfolding)._sij_x
        sij_y = HarmonicWithQ(q_point=q_point,
                               second=phonons.forceconstants.second,
                               distance_threshold=phonons.forceconstants.distance_threshold,
                               folder=phonons.folder,
                               storage=phonons.storage,
                               is_nw=phonons.is_nw,
                               is_unfolding=phonons.is_unfolding)._sij_y
        sij_z = HarmonicWithQ(q_point=q_point,
                               second=phonons.forceconstants.second,
                               distance_threshold=phonons.forceconstants.distance_threshold,
                               folder=phonons.folder,
                               storage=phonons.storage,
                               is_nw=phonons.is_nw,
                               is_unfolding=phonons.is_unfolding)._sij_z
    
    n_modes = phonons.n_modes
    omega = phonons.omega.flatten()

    sij2_x = 0.5*(np.abs(sij_x)**2+np.abs(sij_x.T)**2)
    sij2_y = 0.5*(np.abs(sij_y)**2+np.abs(sij_y.T)**2)
    sij2_z = 0.5*(np.abs(sij_z)**2+np.abs(sij_z.T)**2)
    sij2_x = (sij2_x+sij2_y+sij2_z)/3

    omega_n_m = (omega[:,np.newaxis]*omega[np.newaxis,:])
    vij2_x = np.reshape(sij2_x/omega_n_m, (n_modes, n_modes))

    def check_symmetric(a, tol=1e-6):
        return np.all(np.abs(a-a.T) < tol)

    if not check_symmetric(vij2_x):
        print('v_ij^2 is NOT symmetric up to 1e-6')
        return


    def delta_vec(w, s, discrete = False):
        if discrete:
            return np.where(np.abs(w) < s, 0.5/s, 0)
        else:
            # Gaussian approximation to the delta function
            return 1/(s*np.sqrt(2*np.pi)) * np.exp(-w**2/(2*s**2)) 
    
    def create_v2_omega(v2nm, omegas, omega_n, sigma, normalize = False):

        if vectorize:
            # Vectorized form
            delta_omega  = delta_vec(omegas[:, np.newaxis] - omega_n[np.newaxis, :], sigma, discrete = discrete_delta)
            print('Created delta function with shape {}'.format(delta_omega.shape))
            print('Contracting tensors...', end=' ')
            
            if normalize:
                #norm = oe.contract('wn,um->wu', delta_omega, delta_omega)
                #v2_omega = oe.contract('nm,wn,um,wu->wu', v2nm, delta_omega, delta_omega, norm)
                norm = delta_omega.sum(axis = 1)[:, np.newaxis]*delta_omega.sum(axis = 1)[np.newaxis, :]
                v2_omega = np.zeros((omegas.size, omegas.size))
                v2_omega[norm!=0] = (delta_omega@v2nm@delta_omega.T)[norm!=0]/norm[norm!=0]
            else:
                #v2_omega = oe.contract('nm,wn,um->wu', v2nm, delta_omega, delta_omega)
                v2_omega = delta_omega@v2nm@delta_omega.T
            print('Done.')
        else:
            # C-like implementation 
            v2_omega = np.zeros((len(omegas), len(omegas)))
            for i in range(len(omegas)):
                for j in range(len(omegas)):
                    omega  = omegas[i]
                    omega1 = omegas[j]

                    # Discrete delta
                    #delta_omega  = delta_bin(omega,  omega_n, sigma)
                    #delta_omega1 = delta_bin(omega1, omega_n, sigma)
                    #if np.sum(delta_omega) != 0:
                    #    delta_omega = delta_omega/np.sum(delta_omega)
                    #if np.sum(delta_omega1) != 0:
                    #    delta_omega1 = delta_omega1/np.sum(delta_omega1)
                    
                    # Gaussian delta
                    delta_omega  = delta(omega,  omega_n, sigma)
                    delta_omega1 = delta(omega1, omega_n, sigma)

                    v2_omega[i, j] = oe.contract('nm, n, m', v2nm, delta_omega, delta_omega1)
        
        return v2_omega
    
    omega_n = phonons.omega.flatten()[3:cutoff_n+3]
    omegas = np.linspace(0, omega_n[-1], nomega)
    v2nm = vij2_x[3:cutoff_n+3,3:cutoff_n+3]
    
    print('Entering `create_v2_omega`.')
    v2_omega = create_v2_omega(v2nm, omegas, omega_n, delta_width, normalize)
    print('`create_v2_omega` Done.')

    return omegas, v2_omega

def Ctt_correlator(phonon, Q_list, nomega = 1000, normalize = False, debug = False, return_gamma = False, bandwidths = None):


    const = phonon.hbar/1.66054e-27*1e8 # hbar / 1 a.m.u. in kg
    #const = phonon.hbar/1.66054e-27*(4*np.pi**2)*1e8 # hbar / 1 a.m.u. in kg / (angfreq to freq)**2 * (factor to get result in Ang^2*ps)

    def lorentzian(w, g):
        # Lorentzia functions for angular frequencies
        return g/(w**2 + g**2)
    
    C = {}
    
    # Get number of atoms, physical modes indices, and number of physical normal modes
    natm = phonon.n_atoms
    pm = phonon.physical_mode
    nmodes = phonon.eigenvalues[pm].size
    
    # Reshape the eigenvector matrix
    reshaped_eigvec = np.transpose(np.transpose(np.transpose(phonon.eigenvectors.real, axes = (0, 2, 1))[pm]).reshape(natm, 3, nmodes), axes = (1, 2, 0))
    # Get the atomic positions
    pos = phonon.atoms.get_positions().T
    
    # Angular frequency in rad/ps
    omegas = 2*np.pi * phonon.frequency[pm].flatten()
    # Phonon bandwidths in rad/ps
    if bandwidths is None:
        bandwidths = phonon.bandwidth[pm].flatten() 
    else:
        bandwidths = bandwidths[pm].flatten()
    # Define the angular frequency domain
    omin = omegas.min()
    omax = omegas.max()
    omega_min = (omax-omin)/nomega
    omega_domain = np.linspace(omin, omax, nomega)

    # Compute the projections of the normal modes into plane waves
    reciprocal_cell = phonon.atoms.cell.reciprocal()
    inv_sqrtmass = 1/np.sqrt(phonon.atoms.get_masses())
    Q, bran_ketQ_sq_L, bran_ketQ_sq_T = compute_braQketn(Q_list, reciprocal_cell, np.array([pos]), np.array([reshaped_eigvec]), weights = np.array([inv_sqrtmass]))
    bran_ketQ_sq_L = bran_ketQ_sq_L[0]
    bran_ketQ_sq_T = bran_ketQ_sq_T[0]

    if debug:
        # This should compute the dynamic structure factor
        gamma = 0.01*(omax-omin)
        delta1 = lorentzian(omega_domain[:, np.newaxis] + omegas[np.newaxis, :], gamma)
        delta2 = lorentzian(omega_domain[:, np.newaxis] - omegas[np.newaxis, :], gamma)
        print('Debug value')
        n_omega_delta = ((delta2)).T
        C['L'] = bran_ketQ_sq_L@n_omega_delta
        C['T'] = bran_ketQ_sq_T@n_omega_delta
        if normalize:
            for branch in C:
                inverse_norms = 1/trapz(C[branch], omega_domain, axis = 1)
                C[branch] = np.array([C[branch][:, i]*inverse_norms for i in range(C[branch].shape[1])]).T
    else:
        # Matrix of bandwidths
        gamma = np.zeros_like(omega_domain)[:, np.newaxis] + bandwidths[np.newaxis, :]

        # Lorentzian with the negative frequency peak
        delta1 = lorentzian(omega_domain[:, np.newaxis] + omegas[np.newaxis, :], gamma)
        # Lorentzian with the positive frequency peak
        delta2 = lorentzian(omega_domain[:, np.newaxis] - omegas[np.newaxis, :], gamma)

        # Phonon populations
        pop = phonon.population
        pop[~phonon.physical_mode] = 0
        pop = pop[pm].flatten()

        # Double peaked term in the sum
        n_omega_delta = ((delta1*pop + delta2*(1+pop))/omegas).T

        # Put everything together
        C['L'] = const*bran_ketQ_sq_L@n_omega_delta
        C['T'] = const*bran_ketQ_sq_T@n_omega_delta

        if normalize:
            # Normalize each C(k,w) in w
            for branch in C:
                norms = trapz(C[branch], omega_domain, axis = 1)
                C[branch] = np.array([C[branch][:, i]/norms for i in range(C[branch].shape[1])]).T

    if return_gamma:
        # For debug purposes: return the ingredients
        return Q, omega_domain, C, gamma, n_omega_delta, delta1, delta2, phonon.population[pm].flatten()/omegas
    else:
        return Q, omega_domain, C

######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################

# Old functions and various tests

def compute_SQomega_OLD(Q_list,
                    eigval,
                    eigvec,
                    positions,
                    reciprocal_cell,
                    nomega = 100, use_soft = True, domega = 0.01, print_timing = False):
    """Compute the dinamic structure factor."""
    
    #########################################################################################################
    if print_timing:
        from timeit import default_timer
        print('Compute initial stuff', end=' ')
        start = timeit.default_timer()
    
    SQ = {'L': {}, 'T': {}}
        
    T = positions.shape[0]
    natm = positions.shape[1]
    
    reshaped_eigvec = eigvec.reshape(T, natm, 3, 3*natm)
    
    omegas = np.sqrt(np.abs(eigval))*np.sign(eigval)
            
    omin = np.min(omegas)
    omax = np.max(omegas)
    domega = domega*(omax-omin)
    omega_min = (omax-omin)/nomega
    
    # For testing purposes
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))

    #########################################################################################################
    
    for Q_ in Q_list:
        if print_timing:
            print('Compute bran_ketQ...', end = ' ')
            start = default_timer()
        
        lbl = str(list(Q_))
        Q = 2*np.pi*np.matmul(reciprocal_cell, Q_)   
        eperp = np.random.rand(3)
        eperp -= Q*eperp.dot(Q)/Q.dot(Q)
        eperp /= np.linalg.norm(eperp)

        # t: time index; I: atomic index; a: cartesian index; n; eigenvector index
    
        # Compute exp(i R_I \cdot Q) for each step (t) and for each atom (I)
        exp_i_Q_dot_R = np.exp(1j*np.einsum('a,tIa->tI', Q, positions))/np.sqrt(3*natm)
        
        # Compute <n|Q> = \sum_{I} Evec_{Ia,n} exp(i R_I \cdot Q) for each time, eigenvector and cartesian component
        bran_ketQ = np.einsum('tIan,tI->tan', reshaped_eigvec, exp_i_Q_dot_R)
        
        bran_ketQ_sq_L = np.abs(np.einsum('a,tan->tn', Q, bran_ketQ))**2 / np.sqrt(Q.dot(Q))
        bran_ketQ_sq_T = np.abs(np.einsum('a,tan->tn', eperp, bran_ketQ))**2
        
        # printnp.einsum('a,tan->tn', Q, bran_ketQ).real / Q.dot(Q)        
        
        # For testing purposes 
        if print_timing:
            end = default_timer()
            print('Done in {} s'.format(end - start))        
    
        SQ['T'][lbl] = np.zeros((T, nomega))
        SQ['L'][lbl] = np.zeros((T, nomega))
        
        for i, omega in enumerate(np.linspace(omin, omax, nomega)):
            
            # For testing purposes
            if print_timing:
                print('Compute {}th omega loop...'.format(i), end = ' ')
                start = default_timer()
            
            if not use_soft:
                where = np.logical_and(np.logical_and(omegas>omega-domega, omegas<=omega+domega), omegas>0)
            else:
                where = np.logical_and(omegas>omega-domega, omegas<=omega+domega)
            delta = np.zeros(eigval.shape)
            delta[where] = 1
            
            SQ['L'][lbl][:, i] = np.einsum('tn,tn->t', bran_ketQ_sq_L, delta)
            SQ['T'][lbl][:, i] = np.einsum('tn,tn->t', bran_ketQ_sq_T, delta)
            
            # For testing purposes
            if print_timing:
                end = default_timer()
                print('Done in {} s'.format(end - start))

        for branch in SQ:
            norms = np.array([trapz(SQ[branch][lbl][t, :], np.linspace(omin, omax, nomega)) for t in range(T)])
            SQ[branch][lbl] = np.array([s/n for s, n in zip(SQ[branch][lbl], norms)])
        
    return np.linspace(omin, omax, nomega), SQ

def compute_SQomega_delta(Q_list,
                    eigval,
                    eigvec,
                    positions,
                    reciprocal_cell,
                    nomega = 100, use_soft = True, domega = 0.01, print_timing = False):
    """Compute the dinamic structure factor."""
    
    def lorentzian(omega, omega_n, gamma):
        return gamma/np.pi/(gamma**2 + (omega-omega_n)**2)
    
    #########################################################################################################
    if print_timing:
        from timeit import default_timer
        print('Compute initial stuff', end=' ')
        start = timeit.default_timer()
    
    SQ = {'L': {}, 'T': {}}
        
    T = positions.shape[0]
    natm = positions.shape[1]
    
    reshaped_eigvec = eigvec.reshape(T, natm, 3, 3*natm)
    
    omegas = np.sqrt(np.abs(eigval))*np.sign(eigval)
            
    omin = np.min(omegas)
    omax = np.max(omegas)
    domega = domega*(omax-omin)
    omega_min = (omax-omin)/nomega
    
    # For testing purposes
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))

    #########################################################################################################
    
    for Q_ in Q_list:
        if print_timing:
            print('Compute bran_ketQ...', end = ' ')
            start = default_timer()
        
        lbl = str(list(Q_))
        Q = 2*np.pi*np.matmul(reciprocal_cell, Q_)   
        eperp = np.random.rand(3)
        eperp -= Q*eperp.dot(Q)/Q.dot(Q)
        eperp /= np.linalg.norm(eperp)

        # t: time index; I: atomic index; a: cartesian index; n; eigenvector index
    
        # Compute exp(i R_I \cdot Q) for each step (t) and for each atom (I)
        exp_i_Q_dot_R = np.exp(1j*np.einsum('a,tIa->tI', Q, positions))/np.sqrt(3*natm)
        
        # Compute <n|Q> = \sum_{I} Evec_{Ia,n} exp(i R_I \cdot Q) for each time, eigenvector and cartesian component
        bran_ketQ = np.einsum('tIan,tI->tan', reshaped_eigvec, exp_i_Q_dot_R)
        
        bran_ketQ_sq_L = np.abs(np.einsum('a,tan->tn', Q, bran_ketQ))**2 / np.sqrt(Q.dot(Q))
        bran_ketQ_sq_T = np.abs(np.einsum('a,tan->tn', eperp, bran_ketQ))**2
        
        # printnp.einsum('a,tan->tn', Q, bran_ketQ).real / Q.dot(Q)        
        
        # For testing purposes 
        if print_timing:
            end = default_timer()
            print('Done in {} s'.format(end - start))        
    
        SQ['T'][lbl] = np.zeros((T, nomega))
        SQ['L'][lbl] = np.zeros((T, nomega))
        
        for i, omega in enumerate(np.linspace(omin, omax, nomega)):
            
            # For testing purposes
#             if print_timing:
#                 print('Compute {}th omega loop...'.format(i), end = ' ')
#                 start = default_timer()
            
#             if not use_soft:
#                 where = np.logical_and(np.logical_and(omegas>omega-domega, omegas<=omega+domega), omegas>0)
#             else:
#                 where = np.logical_and(omegas>omega-domega, omegas<=omega+domega)
#             delta = np.zeros(eigval.shape)
#             delta[where] = 1
            
            delta = lorentzian(omega, omegas, domega)
            
            
            SQ['L'][lbl][:, i] = np.einsum('tn,tn->t', bran_ketQ_sq_L, delta)
            SQ['T'][lbl][:, i] = np.einsum('tn,tn->t', bran_ketQ_sq_T, delta)
            
            # For testing purposes
            if print_timing:
                end = default_timer()
                print('Done in {} s'.format(end - start))

        for branch in SQ:
            norms = np.array([trapz(SQ[branch][lbl][t, :], np.linspace(omin, omax, nomega)) for t in range(T)])
            SQ[branch][lbl] = np.array([s/n for s, n in zip(SQ[branch][lbl], norms)])
        
    return np.linspace(omin, omax, nomega), SQ

def compute_SQomega_opt(Q_list,
                    eigval,
                    eigvec,
                    positions,
                    reciprocal_cell,
                    nomega = 100, use_soft = True, domega = 0.01, print_timing = False):
    """Compute the dinamic structure factor."""
    
    #########################################################################################################
    if print_timing:
        from timeit import default_timer
        print('Compute initial stuff', end=' ')
        start = timeit.default_timer()
    
    SQ = {}
        
    T = positions.shape[0]
    natm = positions.shape[1]
    
    reshaped_eigvec = np.transpose(eigvec.reshape(T, natm, 3, 3*natm), axes = (0, 2, 3, 1))
    pos = np.transpose(positions, axes = (0, 2, 1))
    
    omegas = np.sqrt(np.abs(eigval))*np.sign(eigval)
            
    omin = np.min(omegas)
    omax = np.max(omegas)
    domega = domega*(omax-omin)
    omega_min = (omax-omin)/nomega
    
    # For testing purposes
    if print_timing:
        end = default_timer()
        print('Done in {} s'.format(end - start))

    #########################################################################################################
    
    # Create an array with the Q vectors
    Q = np.array([2*np.pi*np.matmul(reciprocal_cell, Q_) for Q_ in Q_list])
    # Create an array with the normalized Q vectors
    Qn = np.einsum('qa,q->qa', Q, 1/np.linalg.norm(Q, axis = 1))
    
    # Create and array with the unit vectors perpendicular to Q
    eperp = np.random.rand(*Q.shape)
    eperp -= np.array([Q_*eperp_.dot(Q_) for eperp_, Q_ in zip(eperp, Q)])
    eperp = np.einsum('qa,q->qa', eperp, 1/np.linalg.norm(eperp, axis = 1))
        
    # Compute exp(i R Q) for each time t, atom I, Q vector q 
    exp_i_Q_dot_R = np.exp(1j*np.transpose(Q@pos/np.sqrt(3*natm), axes = (0, 2, 1))) # 'qa,taI->tIq'
    # Compute the scalar product between eigenvectors and |Q> states for each time t, cartesian direction a, eigenvector n, Q vector q
    bran_ketQ = np.array([r@e for r,e in zip(reshaped_eigvec, exp_i_Q_dot_R)]) # 'tanI,tIq->tanq'
    
    # Project the scalar vector onto the Q direction and compute its squared modulus
    bran_ketQ_sq_L = np.abs(np.einsum('qa,tanq->tqn', Qn, bran_ketQ))**2
    
    # Project the scalar vector onto the direction perpendicular to Q and compute its squared modulus
    bran_ketQ_sq_T = np.abs(np.einsum('qa,tanq->tqn', eperp, bran_ketQ))**2
    
    # Prepare the empty arrays for the longitudinal and transverse components of S(w, Q)
    SQ['T'] = np.zeros((T, Q.shape[0], nomega))
    SQ['L'] = np.zeros((T, Q.shape[0], nomega))
        
    omega_domain = np.linspace(omin, omax, nomega)
    for i, omega in enumerate(omega_domain):

        # For testing purposes
        if print_timing:
            print('Compute {}th omega loop...'.format(i), end = ' ')
            start = default_timer()

        if not use_soft:
            where = np.logical_and(np.logical_and(omegas>omega-domega, omegas<=omega+domega), omegas>0)
        else:
            where = np.logical_and(omegas>omega-domega, omegas<=omega+domega)
        delta = np.zeros(eigval.shape)
        delta[where] = 1

        SQ['L'][:, :, i] = np.einsum('tqn,tn->tq', bran_ketQ_sq_L, delta)
        SQ['T'][:, :, i] = np.einsum('tqn,tn->tq', bran_ketQ_sq_T, delta)

        for branch in SQ:
            norms = 1/trapz(SQ[branch], omega_domain, axis = 2)
            SQ[branch] = np.einsum('tqn,tq->tqn', SQ[branch], norms)
        
    return omega_domain, SQ

