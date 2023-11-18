""" NMF analysis of IOPs """

import os

import numpy as np
from importlib import resources

from oceancolor.iop import cross
from ihop.hydrolight import loisel23

def load_loisel_2023(N_NMF:int=4):
    """ Load the NMF-based parameterization of IOPs from Loisel 2023

    Returns:
        tuple: 
            - **ab** (*np.ndarray*) -- PCA coefficients
            - **Rs** (*np.ndarray*) -- Rrs values
            - **d_a** (*dict*) -- dict of PCA 
            - **d_bb** (*dict*) -- dict of PCA
    """

    # Load up data
    l23_path = os.path.join(resources.files('ihop'),
                            'data', 'NMF')
    l23_a_file = os.path.join(l23_path, f'L23_NMF_a_{N_NMF}.npz')
    l23_bb_file = os.path.join(l23_path, f'L23_NMF_bb_{N_NMF}.npz')

    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)
    nparam = d_a['coeff'].shape[1]+d_bb['coeff'].shape[1]
    ab = np.zeros((d_a['coeff'].shape[0], nparam))
    ab[:,0:d_a['coeff'].shape[1]] = d_a['coeff']
    ab[:,d_a['coeff'].shape[1]:] = d_bb['coeff']

    Rs = d_a['Rs']

    # Return
    return ab, Rs, d_a, d_bb



def prep_loisel23(iop:str, min_wv:float=400., sigma:float=0.05,
                  X:int=4, Y:int=0):
    """ Prep L23 data for NMF analysis

    Args:
        iop (str): IOP to use
        min_wv (float, optional): Minimum wavelength for analysis. Defaults to 400..
        sigma (float, optional): Error to use. Defaults to 0.05.
        X (int, optional): _description_. Defaults to 4.
        Y (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    # Load
    ds = loisel23.load_ds(X, Y)

    # Unpack and cut
    spec = ds[iop].data
    wave = ds.Lambda.data 
    Rs = ds.Rrs.data

    cut = wave >= min_wv
    spec = spec[:,cut]
    wave = wave[cut]
    Rs = Rs[:,cut]

    # Remove water
    if iop == 'a':
        a_w = cross.a_water(wave, data='IOCCG')
        spec_nw = spec - np.outer(np.ones(3320), a_w)
    else:
        spec_nw = spec

    # Reshape
    spec_nw = np.reshape(spec_nw, (spec_nw.shape[0], 
                     spec_nw.shape[1], 1))

    # Build mask and error
    mask = (spec_nw >= 0.).astype(int)
    err = np.ones_like(mask)*sigma

    # Return
    return spec_nw, mask, err, wave, Rs