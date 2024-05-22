""" Methods for the hybrid decomposition of a """

import numpy as np

# Prep
def a_func(wave, Adg, Sdg, H1, H2, W1=None, W2=None):
    # CDOM/detritus
    a_dg = Adg * np.exp(-Sdg*(wave-440)) 
    # a_ph
    a_ph = H1*W1 + H2*W2
    #
    return a_dg + a_ph