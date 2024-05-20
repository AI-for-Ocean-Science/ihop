""" Methods related to noise """

from importlib import resources
import os

import numpy as np

from IPython import embed

def calc_pace_sig(wave:np.ndarray, Rrs_scale:float=1e4):
    """
    Calculate the PACE signal based on the given wave array and Rrs scale.

    Parameters:
        wave (np.ndarray): Array of wave values.
        Rrs_scale (float): Scaling factor for Rrs.

    Returns:
        pace_sig (np.ndarray): Array of PACE error values.
    """

    # Load PACE median error
    pace_file = os.path.join(resources.files('ihop'), 
                             'data', 'PACE' , 'PACE_error.npz')
    pace = np.load(pace_file)

    # Reduce by sqrt(pixels)
    scl = np.sqrt(wave.size/pace['wave'].size)
    # Interpolate
    pace_sig = Rrs_scale * scl * np.interp(
        wave, pace['wave'], pace['Rrs_u'])
    return pace_sig


def add_noise(Rs, perc:int=None, abs_sig:float=None,
              wave:np.ndarray=None, correlate:bool=False):
    """
    Add random noise to the input array Rs.

    Parameters:
        Rs (np.ndarray): Input array.
        perc (int, optional): Percentage of noise to be added as a fraction of Rs. Default is None.
        abs_sig (float, str, optional): Absolute value of noise to be added. Default is None.
        correlate (bool, optional): Whether to correlate the noise. Default is False.

    Returns:
        ndarray: Array with noise added.
    """
    use_Rs = Rs.copy()

    # Random draws
    if correlate:
        npix = Rs.shape[1]
        # Genearte the covariance matrix
        vals = {0: 1., 1: 0.5, 2: 0.3, 3: 0.1}
        cov_m = np.zeros((npix,npix))
        for jj in range(npix):
            i0 = max(0, jj-3)
            i1 = min(jj+4, npix)
            for ii in range(i0, i1):
                diff = int(np.abs(ii-jj))
                cov_m[jj,ii] = vals[diff] 
        # Generate the noise
        r_sig = np.random.multivariate_normal(
            np.zeros(npix), cov_m, size=use_Rs.shape[0])
    else:
        r_sig = np.random.normal(size=Rs.shape)

    # Truncate to 3 sigma
    r_sig = np.minimum(r_sig, 3.)
    r_sig = np.maximum(r_sig, -3.)

    if perc is not None:
        use_Rs += (perc/100.) * use_Rs * r_sig
    elif abs_sig  == 'PACE':
        if wave is None:
            raise ValueError("Need wavelength array for PACE noise.")
        # Add it in
        pace_sig = calc_pace_sig(wave)
        use_Rs += r_sig * pace_sig
    elif abs_sig  == 'PACE_CORR':
        if wave is None:
            raise ValueError("Need wavelength array for PACE noise.")
        # Add it in
        pace_sig = calc_pace_sig(wave)
        use_Rs += r_sig * pace_sig
    elif abs_sig  == 'PACE_TRUNC':
        if wave is None:
            raise ValueError("Need wavelength array for PACE noise.")
        pace_sig = calc_pace_sig(wave)
        # Boost the noise at the edges
        ok_wv = (wave > 380.) & (wave < 700.)
        pace_sig[~ok_wv] *= 100.   
        # Add it in
        use_Rs += r_sig * pace_sig
    elif isinstance(abs_sig, (float,int,np.ndarray)):
        use_Rs += r_sig * abs_sig
    else:
        raise ValueError("Bad abs_sig")
    
    # Return
    return use_Rs