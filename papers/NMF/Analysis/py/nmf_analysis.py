""" NMF Analysis """

import os
from importlib import resources

import numpy as np

from oceancolor.iop import cross

from ihop.hydrolight import loisel23

from ihop.iops import nmf


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

    # Reshape
    spec_nw = np.reshape(spec_nw, (spec_nw.shape[0], 
                     spec_nw.shape[1], 1))

    # Build mask and error
    mask = (spec_nw >= 0.).astype(int)
    err = np.ones_like(mask)*sigma

    # Do it
    comps = nmf.NMFcomponents(
        ref=spec_nw, mask=mask, ref_err=err, n_components=N_NMF,
        path_save=outroot, oneByOne=True)

    # Load
    M = np.load(outroot+'_comp.npy').T
    coeff = np.load(outroot+'_coef.npy').T

    outfile = outroot+'.npz'
    np.savez(outfile, M=M, coeff=coeff,
             spec=spec_nw[...,0],
             mask=mask[...,0],
             err=err[...,0],
             wave=wave)

    print(f'Wrote: {outfile}')

if __name__ == '__main__':

    for n in range(1,10):
        loisel23_components(N_NMF=n+1)