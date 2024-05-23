""" Methods for the hybrid decomposition of a """

import numpy as np

# Prep
def a_func(wave, Adg, Sdg, H1, H2, W1=None, W2=None):
    """
    Calculate the total absorption coefficient (a) for a given wavelength (wave) using the hybrid model.

    Parameters:
        wave (np.ndarray): The wavelength of the light.
        Adg (float): The absorption coefficient of CDOM/detritus.
        Sdg (float): The slope of the CDOM/detritus absorption spectrum.
        H1 (float): The coefficient of the first component of phytoplankton.
        H2 (float): The second coefficient for phytoplankton.
        W1 (float, optional): The specific absorption coefficient of the first component of phytoplankton.
        W2 (float, optional): The specific absorption coefficient of the second component of phytoplankton.

    Returns:
        np.ndarray: The total absorption coefficient (a) at the given wavelength.
    """
    # CDOM/detritus
    a_dg = Adg * np.exp(-Sdg*(wave-440)) 
    # a_ph
    a_ph = H1*W1 + H2*W2
    #
    return a_dg + a_ph