""" Load Training set(s) for Rs """

import numpy as np

from oceancolor.hydrolight import loisel23

from IPython import embed

def loisel23_rs(X:int=4, Y:int=0, 
                min_wv:float=None, 
                max_wv:float=None):
    """
    Load remote sensing data from the Loisel23 dataset.

    Args:
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.
        min_wv (float, optional): The minimum wavelength to include in the data. Defaults to None.
        max_wv (float, optional): The maximum wavelength to include in the data. Defaults to None.

    Returns:
        dict: A dictionary containing the inputs, targets, and extras.
    """

    d = {}

    # Load up the data
    ds = loisel23.load_ds(X, Y)

    # Wavelength cuts?
    gd_wv = np.ones_like(ds.Lambda.data, dtype=bool)
    if min_wv is not None:
        gd_wv = gd_wv & (ds.Lambda.data >= min_wv)
    if max_wv is not None:
        gd_wv = gd_wv & (ds.Lambda.data <= max_wv)

    # Inputs (a, bb, Chl)
    d['inputs'] = {}
    for iop in ['a', 'bb']:
        data = ds[iop].data
        d['inputs'][iop] = data[:,gd_wv]
    d['inputs']['Chl'] = loisel23.calc_Chl(ds)

    # bb_water
    d['bb_w'] = ds.bb.data[0,:] - ds.bbnw.data[0,:]
    
    # Targets
    d['Rs'] = ds.Rrs.data[:,gd_wv]

    # Extras
    d['wave'] = ds.Lambda.data[gd_wv]
    d['param'] = [X, Y, min_wv, max_wv]

    # Return
    return d
    