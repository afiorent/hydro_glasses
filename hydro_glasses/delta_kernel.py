import numpy as np
### # Kernel functions for frequency interpolation/smoothing

def DHO(w, w0, sigma, norm):
    return norm*(w*2*sigma/((w*sigma)**2 + (w**2-w0**2)**2))
def wDHO(w, w0, sigma):
    return w**2*2*sigma/((w*sigma)**2 + (w**2-w0**2)**2)
def gauss(x,sigma):
    return np.exp(-((x) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
def lorenztian(x,eta):
    return eta/np.pi/((x)**2+eta**2)
def compute_kernel(frequencies, values, freq_range, sigma, kernel='lorentz', normalize=True):
    """
    Compute y(omega) = sum_i y_i K(omega - omega_i) on `freq_range`.

    Parameters:
    - frequencies: array-like, omega_i
    - values: array-like, y_i (same length as frequencies)
    - freq_range: numpy array where the result is evaluated
    - sigma: kernel width parameter (eta for Lorentz, sigma for Gaussian, etc.)
    - kernel: 'lorentz' | 'DHO' | 'gauss'
    - normalize: if True, divide result by integral over freq_range

    Returns:
    - numpy array of same shape as freq_range with y(omega)
    """
    freqs = np.asarray(frequencies, dtype=float)
    vals = np.asarray(values, dtype=float)
    omega = np.asarray(freq_range, dtype=float)

    if freqs.ndim != 1 or vals.ndim != 1:
        raise ValueError("`frequencies` and `values` must be 1-D arrays")
    if freqs.size != vals.size:
        raise ValueError("`frequencies` and `values` must have the same length")

    y_out = np.zeros_like(omega, dtype=float)

    if kernel == 'lorentz':
        for f, wgt in zip(freqs, vals):
            x = omega - f
            y_out += wgt * (sigma / np.pi) / (x * x + sigma * sigma)
    elif kernel == 'DHO':
        # using wDHO(w, w0, sigma) = w**2*2*sigma/((w*sigma)**2 + (w**2-w0**2)**2)
        for f, wgt in zip(freqs, vals):
            y_out += wgt * (omega**2 * 2.0 * sigma) / ((omega * sigma)**2 + (omega**2 - f**2)**2)
    elif kernel == 'gauss':
        norm_prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        for f, wgt in zip(freqs, vals):
            x = omega - f
            y_out += wgt * norm_prefactor * np.exp(-0.5 * (x / sigma) ** 2)
    else:
        raise ValueError("Unknown kernel, choose 'lorentz', 'DHO' or 'gauss'")

    if normalize:
        integral = np.trapz(y_out, omega)
        if integral != 0.0:
            y_out /= integral

    return y_out

def compute_vdos(frequencies,freq_range, sigma,kernel='lorentz'):
    """
    Compute the vibrational density of states (VDOS) using a Gaussian kernel.

    Parameters:
    - frequencies: array-like
        List of angular frequencies (rad/s).
    - sigma: float
        Standard deviation of the Gaussian kernel.
    - freq_range: numpy array
        Frequency range over which the VDOS is computed.
    -kernel: str
        The kind of kernel used to compute the VDOS. Default is Lorentzian

    Returns:
    - vdos: numpy array
        Vibrational density of states.
    """
#     # Define the frequency range for the VDOS calculation
#     freq_min = np.min(frequencies) - 3 * sigma
#     freq_max = np.max(frequencies) + 3 * sigma
#     freq_range = np.linspace(freq_min, freq_max, bins)

    # Initialize the VDOS array
    vdos = np.zeros_like(freq_range)

    # Compute the kernel contribution from each frequency
    if kernel=='lorentz':
        for freq in frequencies:
            vdos += lorenztian((freq_range - freq),sigma)
    elif kernel=='DHO':
        for freq in frequencies:
            vdos += wDHO(freq_range,freq,sigma)
    else:
        for freq in frequencies:
            vdos += gauss((freq_range - freq),sigma)

    # Normalize the VDOS
    vdos /= np.trapz(vdos, freq_range)

    return  vdos