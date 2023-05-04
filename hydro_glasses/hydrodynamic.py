import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm
from scipy.integrate import trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline, pchip_interpolate
from scipy.optimize import curve_fit
import sys
sys.path.append('/u/p/ppegolo/Software/scriptelli')
#from simple_bayesian_fit import *

def plot_spectrum(spectrum, vmax = 10, N = None):
    q = spectrum['q']
    f = spectrum['omega']/2/np.pi
    S = spectrum['S']
    titles = ['Longitudinal', 'Transverse']

    fig, axes = plt.subplots(ncols = 2, figsize = (10, 4))

    for b, ax, title in zip(S, axes, titles):
    
        ax.pcolormesh(q, f, S[b][:, :, 0].T,
                      shading = 'nearest',
                      vmin = 0,
                      vmax = vmax,
                      cmap = 'inferno')

        ax.set_ylim(0, 10)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r'Wavevector $Q$ ($\mathrm{\AA}^{-1}$)')
        ax.set_ylabel(r'Frequency $\nu$ (THz)')

        ax.set_title(title)
        ax.set_facecolor('black')
    
    if N is not None:
        fig.suptitle('Anharmonic dynamic structure factor($N={}$)'.format(8*N**3))
    else:
        fig.suptitle('Anharmonic dynamic structure factor')
        
    fig.tight_layout()
    return fig, axes    
  
def plot_spectrum_vs_frequency(spectrum, qmax = 1, units = 'THz', is_integral = False, fit_func = 'lorentzian'):
    from scipy.optimize import curve_fit
    def lorentzian(w, w0, gamma, norm):
        return norm*(gamma/(gamma**2 + (w-w0)**2))
    def gaussian(w, w0, sigma2, norm):
        return norm*np.exp(-(w-w0)**2/2/sigma2)
    if fit_func.lower() == 'lorentzian':
        func = lorentzian
    elif fit_func.lower() == 'gaussian':
        func = gaussian
    else:
        raise ValueError('Fitting function should be either lorentzian or gaussian')
    
    if units.lower() == 'thz':
        conv = 1
        lbl = 'THz'
    elif units.lower() == 'mev':
        conv = 4.135665538536
        lbl = 'meV'
    else:
        raise NotImplementedError('"{}" units are not implemented!'.format(units))
        
    q = spectrum['q']
    f = spectrum['omega']/2/np.pi
    S = spectrum['S']

    cmap = matplotlib.cm.get_cmap('inferno')
    norm = plt.Normalize(vmin=0, vmax=qmax)
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    
    if is_integral:
        Gamma_integ = compute_Gamma(spectrum)

    fig, axes = plt.subplots(ncols = 2, figsize = (8, 4))

    for branch, ax in zip(S, axes):

        for i in range(len(q[q<qmax])):
            try:
                popt, pcov = curve_fit(func, f, S[branch][i].mean(axis = 1))
                pl = ax.plot(f*conv, S[branch][i].mean(axis = 1), 
                             label = '{:.2f}'.format(q[i]), color = cmap(norm(q[i])))
                ax.plot(f*conv, func(f, *popt), ls = '--', color = 'green', lw = 2) # pl[0].get_color())
                if is_integral:
                    ax.plot(f*conv, func(f, popt[0], Gamma_integ[branch][i]/2/np.pi, 
                                           popt[2]), ls = 'dotted', color = 'blue', lw = 2) # pl[0].get_color())
                
            except:
                print(i, end = ' ')
                continue
        ax.set_title(branch)
#         ax.set_xlim(0, 100)
        ax.set_xlabel(r'Frequency $\nu$ ({})'.format(lbl))
        ax.set_ylabel(r'$S(Q, \nu)$')    

    fig.tight_layout()
    return fig, axes

def compute_disorder_widths(spectrum, qmax = 1, qmax_c = 0.4, fit_func = 'DHO', eta = 0):
    from scipy.optimize import curve_fit
    def lorentzian(w, w0, gamma, norm):
        return norm*(gamma/(gamma**2 + (w-w0)**2))
    def gaussian(w, w0, sigma, norm):
        return norm*np.exp(-(w-w0)**2/2/sigma**2)
    # TODO: uncomment the new DHO and remove the old one once the Lanczos alg. is updated
    #def DHO(w, w0, gamma, norm, eta = eta):
    #    num = norm*np.pi*(2*G+eta)*w0
    #    den = (w**2 - w0**2 - 0.5*(G + eta)**2)**2 + w**2*(2*G + eta)**2
    #    return num/den
    def DHO(w, w0, tau, norm): # old version
        return norm*(w*2*tau/((w*tau)**2 + (w**2-w0**2)**2))

    if fit_func.lower() == 'lorentzian':
        func = lorentzian
    elif fit_func.lower() == 'gaussian':
        func = gaussian
    elif fit_func.lower() == 'dho':
        #func = lambda w, w0, gamma, norm: DHO(w, w0, gamma, norm, eta = eta) # TODO: for the new version
        func = DHO  # TODO: old
    else:
        raise ValueError('Fitting function should be either lorentzian or gaussian')
    
    q = spectrum['q']
    w = spectrum['omega']
    S = spectrum['S']
    
    q_for_fit = {}
    freq = {}
    width = {}
    width_std = {}
    c_sound = {}

    for branch in S:
        q_for_fit[branch] = []
        width[branch] = []
        width_std[branch] = []
        freq[branch] = []
        
        for i in range(len(q[q<qmax])):
            try:
                if len(S[branch][i].shape) == 2:
                    popt, pcov = curve_fit(func, w, S[branch][i].mean(axis = 1))
                else:
                    popt, pcov = curve_fit(func, w, S[branch][i])
                q_for_fit[branch].append(q[i])
                freq[branch].append([popt[0], pcov[0,0]])
                width[branch].append(popt[1])
                width_std[branch].append(np.sqrt(pcov[1,1]))
                
            except:
                print(i, end = ' ')
                continue
        
        q_for_fit[branch] = np.array(q_for_fit[branch])
        width[branch] = np.array(width[branch])
        width_std[branch] = np.array(width_std[branch])
        freq[branch] = np.array(freq[branch])
        
        popt, pcov = curve_fit(lambda k, c: c*k, q_for_fit[branch][q_for_fit[branch]<=qmax_c], 
                               freq[branch][q_for_fit[branch]<=qmax_c, 0],
                               sigma = freq[branch][q_for_fit[branch]<=qmax_c, 1])
        c_sound[branch] = np.array([popt[0], np.sqrt(pcov[0,0])])

    return {'q': q_for_fit, 
            'Gamma': width,
            'Gamma_std': width_std,
            'omega': freq, 
            'c_sound': c_sound}


def fit_disorder_widths(fit_dict, qmax = 0.5, power = 2, eta = 0):
    from scipy.optimize import curve_fit
    
    q = fit_dict['q']
    G = fit_dict['Gamma']
    
    coeffs = {}
    
    for branch in q:
        
        popt, pcov = curve_fit(lambda k, a: a*k**power, 
                               q[branch][q[branch]<=qmax],
                               G[branch][q[branch]<=qmax]-eta)
        
        coeffs[branch] = np.array([popt[0], pcov[0,0]])
    
    return coeffs

def plot_fitted_widths(spectrum, qmax = 1, qmax_fit = 0.5, power = 2, fit_func_Gamma = 'lorentzian'):
    
    fit = compute_disorder_widths(spectrum, qmax = qmax, qmax_c=0.6, fit_func = fit_func_Gamma)

    fig, ax = plt.subplots()
    plots = []
    for b in fit['q']:
        pl, = ax.plot(fit['q'][b], fit['Gamma'][b], '--o')
        plots.append(pl)

    coeffs = fit_disorder_widths(fit, qmax = qmax_fit, power = power)
    x = np.linspace(0, qmax, 100)
    handles = []
    labels = []
    for b, pl in zip(coeffs, plots):
        y = coeffs[b][0]*x**2
        yerr = coeffs[b][1]*x**2
        pl1, = ax.plot(x[x<=qmax_fit], y[x<=qmax_fit], color = pl.get_color(), ls = '-')
        pl1_, = ax.plot(x[x>qmax_fit], y[x>qmax_fit], color = pl.get_color(), ls = 'dotted')
        
        pl2 = ax.fill_between(x, y-yerr, y+yerr, color = pl.get_color(), alpha = 0.5)
        handles.append((pl1, pl2))
        labels.append(b)

    ax.set_xlim(0,)
    ax.set_ylim(0,)

    ax.legend(handles, labels)
    ax.set_xlabel(r'Wavevector $Q$ ($\mathrm{\AA}^{-1}$)')
    ax.set_ylabel(r'Disorder linewidth $\Gamma_{\mathrm{dis}}$ (THz)')
    
    return fig, ax, coeffs

def compute_Gamma(spectrum):
    from scipy.integrate import trapezoid
    
    w = spectrum['omega']
    S = spectrum['S']
    
    Gamma = {}
    
    for branch in S:
        N = trapezoid(S[branch].mean(axis=2), w, axis=1)
        I = trapezoid(S[branch].mean(axis=2)**2, w, axis=1)
#         print(norm, integ)
        Gamma[branch] = N**2/(2*np.pi*I)
    return Gamma

import ase.units as units
from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline

def calculate_heat_capacity(omega,temp):
    '''Calculates the heat capacity per mode as written in the kaldo paper. The output is in J/K.'''
    kB = 1.380649e-23 # J/K
    h = 6.62607015e-22 # J*ps
    result = np.ones_like(omega)*kB
    where = ~np.isclose(omega, 0)
    frequency = omega[where]/(2*np.pi) # 1/ps
    result[where] = (h*frequency)**2/4/kB/temp**2 / np.sinh(h*frequency/2/kB/temp)**2
    return result

def gamma_dis_omega(omega,coeff_k,c):
    '''Computes gamma for propagons'''
    k=omega/(c)
    return coeff_k*(k**2)

def dos_omega_T(omega,cT):
    '''Transverse DOS in the Debye Model'''
    return omega**2/np.pi**2/cT**3

def dos_omega_L(omega,cL):
    '''Longitudinal DOS in the Debye Model'''
    return omega**2/2/np.pi**2/cL**3

def bayesian_fit_gamma(x_data, y_data, x_infer, y_err = None):
    basis_kind = 'monomial'
    n_min = 2
    M_tot = 10
    log_evidence_vP, alpha_vP, beta_vP = minimize_logevidence(x_data, 
                                                              y_data, 
                                                              M_tot, 
                                                              y_err = y_err,
                                                              n_min = n_min, 
                                                              basis_kind = basis_kind)
    i_optP = np.nanargmax(log_evidence_vP) + n_min
    M_optP = i_optP+1
    meanN_optPtot, sigmaN_optPtot = bayesian_fit(x_data, 
                                                 y_data,
                                                 x_infer, 
                                                 log_evidence_vP,
                                                 M_optP, 
                                                 alpha_vP, 
                                                 beta_vP = beta_vP, 
                                                 n_min = n_min, 
                                                 basis_kind = basis_kind)
    
    return meanN_optPtot, sigmaN_optPtot
    

def general_fit(q,a,b,c):
    return a*q**2 + b*q**4 + c*q**6

def polytanh_fit(q,a,b,c):
    return a*q**(np.tanh((q-b)/c)+3)

def inverse(y_curve,y_range,x_range):
    x_inv=[]
    for y in y_range:
        diff=np.abs(y_curve(x_range)-y )
        i_min=np.argmin(diff)
        x_inv.append(x_range[i_min])
    return np.array(x_inv)

def fit_q_w(w,a,b,c,d):
    return a*w+b*w**2+c*w**3+d*w**4
def cq_factor(fit,b,omega):
    q=np.linspace(0,1,1000)
    w_q_L_spl=UnivariateSpline(fit['q'][b],fit['omega'][b][:,0])
    c_q_spl=w_q_L_spl.derivative()
    q_w=inverse(w_q_L_spl,y_range=omega,x_range=q)
    popt,conv=curve_fit(fit_q_w,omega,q_w)
    q_w_fit=fit_q_w(omega,*popt)
    c_sound=fit['c_sound'][b][0]
    factor_b={'T':2,'L':1}
    dos_c2=factor_b[b]*c_q_spl(q_w_fit)*(q_w_fit**2)/2/np.pi**2
    return  dos_c2
    
def hydrodynamic_contribution(spectrum,
                              T,
                              q_max = None, 
                              omega_max = None,
                              omega_min = 0,
                              is_interpolate = False,
                              harmonic_eta = None,
                              boundary_thickness = None,
                              anharmonic_gamma = None,
                              linear_dispersion=True,
                              fit_func_Gamma = 'lorentzian',
                              first_omega_max = None
                             ):
    '''
    Computes the hydrodynamic contribution to the thermal conductivity.
    
    Parameters:
        - spectrum:              Dictionary with the dynamic structure factors.
        - T:                     Temperature in Kelvin.
        - q_max:                 Cutoff wavevector for propagons.
        - omega_max:             Cutoff frequecncy for propagons. If it is a dictionary,
                                 the keys must be 'L' for longitudinal and 'T' for transverse modes.
        - is_interpolate:        If True, the bandwiths are interpolated with a spline function. 
                                 If False, the bandwiths are assumed to be quadratic in omega and are
                                 fitted accordingly.
                                
        
    
    Returns:
        - integT: Transverse hydrodinamic contribution
        - integL: Longitudinal hydrodinamic contribution
        - tot:    Total hydrodinamic contribution
        - d:      A dictionary with various quantities for debug purposes.
    '''
    
    from scipy.integrate import trapezoid
    from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline
       
    if omega_max is not None and q_max is not None:
        raise ValueError('Provide either `q_max` or `omega_max`, not both.')
        
    fit = compute_disorder_widths(spectrum, qmax=1, qmax_c=0.6, fit_func = fit_func_Gamma)
    cT = fit['c_sound']['T'][0]
    cL = fit['c_sound']['L'][0]
    print('Speed of sound:')
    print('c_L = {} m/s'.format(cL*100))
    print('c_T = {} m/s'.format(cT*100))
    
    if omega_max is not None:
        if isinstance(omega_max, dict):
            print('Using Ioffe-Regel cutoffs')
            omega_max_T = omega_max['T']
            omega_max_L = omega_max['L']
        else:
            print('Fixed cutoff', end = ' ')
            omega_max_T = omega_max_L = omega_max
            print('omega_max = {}'.format(omega_max))
    elif q_max is not None:
        omega_max_T = cT*q_max
        omega_max_L = cL*q_max
        print('omega max T and L',omega_max_T,omega_max_L)
    else:
        raise ValueError('Provide one between `q_max` and `omega_max`.')
        
    if first_omega_max is None and is_interpolate == '2&4bis':
        first_omega_max = (omega_max_T + omega_max_L)/4
        
    # Angular frequency in rad/ps
    omega_T = np.linspace(omega_min, omega_max_T, 5000, endpoint = True)
    omega_L = np.linspace(omega_min, omega_max_L, 5000, endpoint = True)

    # Compute acoustic dos up to cutoff
    dos_T = dos_omega_T(omega_T, cT)
    dos_L = dos_omega_L(omega_L, cL)
    
    # Gamma definition
    if harmonic_eta is None:
        # Manually put (0,0) in the array Gamma vs q
        xT = np.insert(fit['q']['T'],0,0)
        yT = np.insert(fit['Gamma']['T'],0,0)
        yT_std = np.insert(fit['Gamma_std']['T'],0,1e-10)
        xL = np.insert(fit['q']['L'],0,0)
        yL = np.insert(fit['Gamma']['L'],0,0)
        yL_std = np.insert(fit['Gamma_std']['L'],0,1e-10)
    else:
        # Manually put (0,0) in the array Gamma vs q and subtract the finite broadening 
        xT = np.insert(fit['q']['T'],0,0)
        yT = np.insert(fit['Gamma']['T']-harmonic_eta,0,0)
        xL = np.insert(fit['q']['L'],0,0)
        yL = np.insert(fit['Gamma']['L']-harmonic_eta,0,0)
        if anharmonic_gamma is not None:
            # Add the anharmonic bandwidth to the harmonic-disorder one
            # The spline is assumed to be for the square root of gamma, hence the squares below
            yT += anharmonic_gamma(xT/cT/2/np.pi)**2
            yL += anharmonic_gamma(xL/cL/2/np.pi)**2
            
            
            
    # Gamma fitting
    if is_interpolate == 'spline':
        # Interpolate Gamma vs q 
        # TODO: is this the best way to get a positive vector? Probably not...
        
        splT = InterpolatedUnivariateSpline(xT, np.sqrt(yT), k=1)
        gammaT = splT(omega_T/cT)**2

# #         q = np.insert(fit['q']['L'], 0, 0)
#         spl = InterpolatedUnivariateSpline(q, np.insert(fit['Gamma']['L'], 0, 0), k = 2)
#         gammaL = spl(omega_L/cL)
    
        splL = InterpolatedUnivariateSpline(xL, np.sqrt(yL), k=1)
        gammaL = splL(omega_L/cL)**2

#         gammaL = pchip_interpolate(xL, yL, x = omega_L/cL)
#         gammaT = pchip_interpolate(xT, yT, x = omega_T/cT)
        
#         popt, pcov=curve_fit(polytanh_fit, omega_L/cL, gammaL, p0 = [1, 1, 0.5]) #,sigma=weight_T[1:])
#         gammaL = polytanh_fit(omega_L/cL,*popt)
#         popt, pcov=curve_fit(polytanh_fit, omega_T/cT, gammaT, p0 = [1, 1, 0.5]) #,sigma=weight_T[1:])
#         gammaT = polytanh_fit(omega_T/cT,*popt)

    elif is_interpolate == 'square':
        # Assume Gamma=A*q**2 and fit A
        w_fit = xT*cT
        where = w_fit/2/np.pi < 3.5
        popt, pcov = curve_fit(lambda q,a: a*q**2, w_fit[where], yT[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xT[xT<=0.5], yT[xT<=0.5])
        gammaT = popt[0]*(omega_T)**2
        
        w_fit = xL*cL
        where = w_fit/2/np.pi < 3.5
        popt, pcov = curve_fit(lambda q,a: a*q**2, w_fit[where], yL[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xL[xL<=0.5], yL[xL<=0.5])
        gammaL = popt[0]*(omega_L)**2

    elif is_interpolate == '2&4':
        # Assume Gamma=A*q**2+B*q**4 and fit A and B
        def sq_and_qu(w,a,b):
            return np.abs(a)*w**2+np.abs(b)*w**4
        
        w_fit = xT*cT
        where = w_fit < omega_max_T
#         where = w_fit/2/np.pi < 3.5
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt, pcov = curve_fit(sq_and_qu, w_fit[where], yT[where], 
                               sigma = sigma)
#                                sigma = yT_std[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xT[xT<=0.5], yT[xT<=0.5])
        gammaT = sq_and_qu(omega_T, *popt)
        
        w_fit = xL*cL
        where = w_fit < omega_max_L
#         where = w_fit/2/np.pi < 3.5
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt, pcov = curve_fit(sq_and_qu, w_fit[where], yL[where],
                               sigma = sigma)
#                                sigma = yL_std[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xL[xL<=0.5], yL[xL<=0.5])
        gammaL = sq_and_qu(omega_L, *popt)
        
    elif is_interpolate == '2&4bis':
        # Assume Gamma=A*q**2+B*q**4 and fit A and B separately. Requires two cutoffs
        def qu(w, a):
            return np.abs(a)*w**4
        
        w_fit = xT*cT        
        where = (w_fit > first_omega_max) & (w_fit < omega_max_T)
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt_, pcov = curve_fit(qu, w_fit[where], yT[where], 
                                sigma = sigma)
        
        b = popt_[0]
        a = b*first_omega_max**2
        gammaT = a*omega_T**2 + b*omega_T**4
        
        w_fit = xL*cL
        where = (w_fit > first_omega_max) & (w_fit < omega_max_L)
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt_, pcov = curve_fit(qu, w_fit[where], yL[where], 
                                sigma = sigma)
        
        b = popt_[0]
        a = b*first_omega_max**2
        gammaL = a*omega_L**2 + b*omega_L**4
        
    elif is_interpolate == '2&4ter':
        # Assume Gamma=A*q**2+B*q**4. Fit only A up to an intermediate cutoff, then find B accordingly
        def qu(w, a):
            return np.abs(a)*w**2
        
        w_fit = xT*cT        
        where = (w_fit < first_omega_max)
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt_, pcov = curve_fit(qu, w_fit[where], yT[where], 
                                sigma = sigma)
        
        a = popt_[0]
        b = a/first_omega_max**2
        gammaT = np.concatenate([a*omega_T[omega_T<first_omega_max]**2, b*omega_T[omega_T>=first_omega_max]**4])
        
        w_fit = xL*cL
        where = (w_fit < first_omega_max)
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt_, pcov = curve_fit(qu, w_fit[where], yL[where], 
                                sigma = sigma)
        
        a = popt_[0]
        b = a/first_omega_max**2
        gammaL = np.concatenate([a*omega_L[omega_L<first_omega_max]**2, b*omega_L[omega_L>=first_omega_max]**4])
        
    elif is_interpolate == '2&4quater':
        # Assume Gamma=A*q**2+B*q**4. Fit only B from an intermediate cutoff, then find A accordingly
        def qu(w, a):
            return np.abs(a)*w**4
        
        w_fit = xT*cT        
        where = (w_fit >= first_omega_max) & (w_fit < omega_max_T)
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt_, pcov = curve_fit(qu, w_fit[where], yT[where], 
                                sigma = sigma)
        
        b = popt_[0]
        a = b*first_omega_max**2
        gammaT = np.concatenate([a*omega_T[omega_T<first_omega_max]**2, b*omega_T[omega_T>=first_omega_max]**4])
        
        w_fit = xL*cL
        where = (w_fit >= first_omega_max) & (w_fit < omega_max_L)
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt_, pcov = curve_fit(qu, w_fit[where], yL[where], 
                                sigma = sigma)
        
        b = popt_[0]
        a = b*first_omega_max**2
        gammaL = np.concatenate([a*omega_L[omega_L<first_omega_max]**2, b*omega_L[omega_L>=first_omega_max]**4])
        
    elif is_interpolate == '2&4quinquies':
        # Assume Gamma=A*q**2+B*q**4. Fit A up to an intermediate cutoff, then fit B with A fixed
       
        w_fit = xT*cT        
        where = (w_fit > 0) & (w_fit < first_omega_max)
        sigma = w_fit[where]**2
#         sigma[0] = sigma[1]
        popt_, pcov = curve_fit(lambda x, a: a+2*x, np.log(w_fit[where]), np.log(yT[where]),
                              sigma = sigma)

        a = np.exp(popt_[0])
        where = (w_fit >= first_omega_max) & (w_fit < omega_max_T)
#         where = (w_fit >= first_omega_max) & (w_fit < omega_max_T)
        sigma = w_fit[where]**2
#         sigma[0] = sigma[1]
        popt_, pcov = curve_fit(lambda x, a: a+4*x, np.log(w_fit[where]), np.log(yT[where]), 
                                sigma = sigma)
        b = np.exp(popt_[0])
        print(a,b)
        gammaT = np.concatenate([a*omega_T[omega_T<first_omega_max]**2, b*omega_T[omega_T>=first_omega_max]**4])
#         gammaT = a*omega_T**2 + b*omega_T**4
        
        w_fit = xL*cL
        where = (w_fit > 0) & (w_fit < first_omega_max)
        sigma = w_fit[where]**2
#         sigma[0] = sigma[1]
        popt_, pcov = curve_fit(lambda x, a: a+2*x, np.log(w_fit[where]), np.log(yL[where]),
                              sigma = sigma)

        a = np.exp(popt_[0])
        where = (w_fit >= first_omega_max) & (w_fit < omega_max_L)
#         where = (w_fit >= first_omega_max) & (w_fit < omega_max_T)
        sigma = w_fit[where]**2
#         sigma[0] = sigma[1]
        popt_, pcov = curve_fit(lambda x, a: a+4*x, np.log(w_fit[where]), np.log(yL[where]), 
                                sigma = sigma)
        b = np.exp(popt_[0])
        print(a,b)
        
        gammaL = np.concatenate([a*omega_L[omega_L<first_omega_max]**2, b*omega_L[omega_L>=first_omega_max]**4])
        
    elif is_interpolate == '2&4bound':
        # Assume Gamma=A*q**2+B*q**4 and fit A and B
        def sq_and_qu(w,bou,a,b):
            return np.abs(a)*w**2+np.abs(b)*w**4 + bou
        
        w_fit = xT*cT
        where = w_fit < omega_max_T
#         where = w_fit/2/np.pi < 3.5
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt, pcov = curve_fit(lambda w,a,b: sq_and_qu(w, cT*100/boundary_thickness*1e-12,a,b), 
                               w_fit[where], yT[where]+cT*100/boundary_thickness*1e-12, 
                               sigma = sigma,maxfev = 6000)
#                                sigma = yT_std[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xT[xT<=0.5], yT[xT<=0.5])
        print('T',popt[0],popt[1],np.sqrt(popt[0]/popt[1]))
        gammaT = sq_and_qu(omega_T, cT*100/boundary_thickness*1e-12, *popt)
        
        w_fit = xL*cL
        where = w_fit < omega_max_L
#         where = w_fit/2/np.pi < 3.5
        sigma = w_fit[where]**2
        sigma[0] = sigma[1]
        popt, pcov = curve_fit(lambda w,a,b: sq_and_qu(w,cL*100/boundary_thickness*1e-12,a,b),
                               w_fit[where], yL[where]+cL*100/boundary_thickness*1e-12,
                               sigma = sigma,maxfev = 6000)
#                                sigma = yL_std[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xL[xL<=0.5], yL[xL<=0.5])
        print('L',popt[0],popt[1],np.sqrt(popt[0]/popt[1]))
        gammaL = sq_and_qu(omega_L, cL*100/boundary_thickness*1e-12, *popt)
    
    elif is_interpolate == 'harm':
        
        w_fit = xT*cT
        where = w_fit < omega_max_T
#         where = w_fit/2/np.pi < 3.5
        popt, pcov = curve_fit(lambda w, a: a*w**4, w_fit[where], yT[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xT[xT<=0.5], yT[xT<=0.5])
        gammaT = harmonic_eta + popt[0]*omega_T**4
        
        w_fit = xL*cL
        where = w_fit < omega_max_L
#         where = w_fit/2/np.pi < 3.5
        popt, pcov = curve_fit(lambda w, a: a*w**4, w_fit[where], yL[where])
#         popt, pcov = curve_fit(lambda q,a: a*q**2, xL[xL<=0.5], yL[xL<=0.5])
        gammaL = harmonic_eta + popt[0]*omega_L**4
    
    elif is_interpolate == 'hybrid':
        threshold=0.35
        weight_T=xT**2
        weight_L=xL**2
        
        popt, pcov=curve_fit(general_fit, xT[1:], yT[1:], sigma = weight_T[1:], absolute_sigma=False)
        gammaT=general_fit(omega_T/cT,*popt)
        #splT = InterpolatedUnivariateSpline(xT, yT)
        #stop_T=xT[xT<=threshold][-1]*cT
        #gammaT=np.concatenate( (general_fit(omega_T[omega_T<=stop_T]/cT,popt[0],popt[1],popt[2]),splT(omega_T[omega_T>stop_T]/cT) ) )  
        popt, pcov=curve_fit(general_fit, xL[1:], yL[1:], sigma = weight_L[1:], absolute_sigma=False)
#         splL = InterpolatedUnivariateSpline(xL, yL)
#         stop_L=xL[xL<=threshold][-1]*cL
#         gammaL=np.concatenate( (general_fit(omega_L[omega_L<=stop_L]/cL,popt[0],popt[1],popt[2]),splL(omega_L[omega_L>stop_L]/cL) ) )  
        gammaL=general_fit(omega_L/cL,*popt)
        
        
        #yL_tmp=np.concatenate( (gammaL[omega_L/cL<=threshold],yL[xL>threshold]) )
        #xL_tmp=np.concatenate( (omega_L[omega_L/cL<=threshold]/cL,xL[xL>threshold]) )
        #splL=UnivariateSpline(xL_tmp, yL_tmp,s=len(xL_tmp)/500,k=2)
        #gammaL=splL(omega_L/cL)
        
    elif is_interpolate == 'bayesian':
        
        gammaL, gammaL_std = bayesian_fit_gamma(xL, yL, omega_L/cL) #, y_err = xL**2/50)
        gammaT, gammaT_std = bayesian_fit_gamma(xT, yT, omega_T/cT) #, y_err = xT**2/50)

    else:
        print('there is not this option for the linewidths. Only spline, square, hybrid')
        

    # Compute the heat capacity per mode
    cvT = calculate_heat_capacity(omega_T, T)
    cvL = calculate_heat_capacity(omega_L, T)
        
    # Compute the hydrodynamic contributions
    if linear_dispersion:
        integrandT = 1e22*cT**2*cvT*dos_T/2/gammaT/3
        integrandL = 1e22*cL**2*cvL*dos_L/2/gammaL/3
#         integT = 1e22*cT**2*trapezoid(cvT*dos_T/2/gammaT, omega_T)/3
#         integL = 1e22*cL**2*trapezoid(cvL*dos_L/2/gammaL, omega_L)/3
    else:
        dos_c2_T=cq_factor(fit,'T',omega_T)
        dos_c2_L=cq_factor(fit,'L',omega_L)
        #print(factor_T,factor_L)
#         integT = 1e22*trapezoid(dos_c2_T*cvT/2/gammaT, omega_T)/3
#         integL = 1e22*trapezoid(dos_c2_L*cvL/2/gammaL, omega_L)/3
        
        integrandT = 1e22*dos_c2_T*cvT/2/gammaT/3
        integrandL = 1e22*dos_c2_L*cvL/2/gammaL/3
        
        print('non linear dispersion')
        
    integT = trapezoid(integrandT, omega_T)
    integL = trapezoid(integrandL, omega_L)

#     low_cut_T = omega_T>=xT[1]*cT
#     low_cut_L = omega_L>=xL[1]*cL
    
#     integT = 1e22*cT**2*trapezoid(cvT[low_cut_T]*dos_T[low_cut_T]/2/gammaT[low_cut_T], omega_T[low_cut_T])/3
#     integL = 1e22*cL**2*trapezoid(cvL[low_cut_L]*dos_L[low_cut_L]/2/gammaL[low_cut_L], omega_L[low_cut_L])/3
    
    return integT, integL, integT + integL, {'omega_L': omega_L, 
                                             'omega_T': omega_T, 
                                             'gamma_L': gammaL, 
                                             'gamma_T': gammaT,
                                             'dos_L': dos_L,
                                             'dos_T': dos_T,
                                             'cv_L': cvL,
                                             'cv_T': cvT,
                                             'omega_max': {'T': omega_max_T, 'L': omega_max_L},
                                             'kappa_vs_omega_L': integrandL,
                                             'kappa_vs_omega_T': integrandT}
