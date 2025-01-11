""" Methods for the hybrid decomposition of a """

import numpy as np

# Prep
def a_func(wave, params, W1=None, W2=None):
    """
    Calculate the total absorption coefficient (a) for a given wavelength (wave) using the hybrid model.

    Parameters:
        wave (np.ndarray): The wavelength of the light.
        params (np.ndarray): The parameters for the hybrid model.  Not in log 10
            Adg (float): The absorption coefficient of CDOM/detritus.
            Sdg (float): The slope of the CDOM/detritus absorption spectrum.
            H1 (float): The coefficient of the first component of phytoplankton.
            H2 (float): The second coefficient for phytoplankton.
        W1 (float, optional): The specific absorption coefficient of the first component of phytoplankton.
        W2 (float, optional): The specific absorption coefficient of the second component of phytoplankton.

    Returns:
        np.ndarray: The total absorption coefficient (a) at the given wavelength.
    """
    # TODO -- Worry about blowing up the memory

    # CDOM/detritus
    #a_dg = Adg * np.exp(-Sdg*(wave-440)) 
    a_dg = np.outer(params[...,0], np.ones_like(wave)) *\
        np.exp(np.outer(-params[...,1],wave-440.))

    # a_ph
    a_ph = np.outer(params[...,2], W1) + np.outer(params[...,3],W2)

    #
    if len(params.shape) == 1:
        return (a_dg + a_ph).flatten()
    else:
        return a_dg + a_ph