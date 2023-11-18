""" NMF Analysis """

import os
from importlib import resources

import numpy as np

from oceancolor.iop import cross


from ihop.utils import nmf as nmf_utils
from ihop.iops import nmf as iop_nmf

from IPython import embed

def loisel23_components(iop:str, N_NMF:int=10):

    path = os.path.join(resources.files('ihop'), 
                    'data', 'NMF')
    outroot = os.path.join(path, f'L23_NMF_{iop}_{N_NMF}')

    # Load
    spec_nw, mask, err, wave, Rs = iop_nmf.prep_loisel23(iop)

    # Do it
    comps = nmf_utils.NMFcomponents(
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
             wave=wave,
             Rs=Rs)

    print(f'Wrote: {outfile}')

if __name__ == '__main__':

    for n in range(1,10):
        loisel23_components('a', N_NMF=n+1)
        loisel23_components('bb', N_NMF=n+1)