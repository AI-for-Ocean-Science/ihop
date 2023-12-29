""" Load IOP data """

from oceancolor.hydrolight import loisel23
from oceancolor.iop import cross

import numpy as np

def load_loisel23(iop:str, X:int=4, Y:int=0, 
    min_wv:float=300., high_cut:float=1000.,
    remove_water:bool=False):

    # Load
    ds = loisel23.load_ds(X, Y)

    # Unpack and cut
    spec = ds[iop].data
    wave = ds.Lambda.data 
    Rs = ds.Rrs.data

    cut = (wave >= min_wv) & (wave <= high_cut)
    spec = spec[:,cut]
    wave = wave[cut]
    Rs = Rs[:,cut]

    # Remove water
    if iop == 'a' and remove_water:
        a_w = cross.a_water(wave, data='IOCCG')
        spec = spec - np.outer(np.ones(3320), a_w)

    # Return
    return spec, wave, Rs