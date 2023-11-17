""" NMF Analysis """

import os
from importlib import resources

import numpy as np

from oceancolor.iop import cross

from ihop.hydrolight import loisel23

import nmf_imaging

# TODO -- remove this dependency!
from astropy.io import fits


def loisel23_components(min_wv:float=400.,
                        sigma:float=0.05,
                        N_NMF:int=10, X:int=4, Y:int=0):

    path = os.path.join(resources.files('ihop'), 
                    'data', 'NMF')
    outroot = os.path.join(path, f'L23_NMF_{N_NMF}')
    
    # Load
    ds = loisel23.load_ds(X, Y)

    # Unpack and cut
    spec = ds['a'].data
    wave = ds.Lambda.data 

    cut = wave >= min_wv
    spec = spec[:,cut]
    wave = wave[cut]

    # Remove water
    a_w = cross.a_water(wave, data='IOCCG')
    spec_nw = spec - np.outer(np.ones(3320), a_w)

    # Build mask and error
    mask = (spec_nw >= 0.).astype(int)
    err = np.ones_like(mask)*sigma

    # Do it
    comps = nmf_imaging.NMFcomponents(
        ref=spec_nw, mask=mask, ref_err=err, n_components=N_NMF,
        path_save=outroot, oneByOne=True)

    # Load
    hdul = fits.open(outroot+'_comp.fits')
    M = hdul[0].data.T

    hdul2 = fits.open(outroot+'_coef.fits')
    coeff = hdul2[0].data.T

    outfile = outroot+'_M.npz'
    np.savez(outfile, M=M, coeff=coeff)

    print(f'Wrote: {outfile}')

if __name__ == '__main__':

    loisel23_components()