""" Decompose IOPs """

import os
import numpy as np

from oceancolor.iop import cross

from cnmf.oceanography import iops as cnmf_iops

from ihop.training_sets import load_rs
from ihop.iops.decompose import generate_pca
from ihop.iops.decompose import generate_nmf
from ihop.iops.decompose import loisel23_filenames

        
def pca_loisel23(X:int=4, Y:int=0, Ncomp:int=3,
                 clobber:bool=False):

    # Load training data
    d = load_rs.loisel23_rs(X=X, Y=Y)

    # Loop on IOP
    for iop in ['a', 'bb']:
        # Prep
        # TODO -- Replace this with loisel23_filenames in iops.decompose.py
        outroot = f'pca_L23_X{X}Y{Y}_{iop}'
        # Do it
        generate_pca(d['inputs'][iop], outroot, Ncomp,
                     extras={'Rs':d['Rs'], 'wave':d['wave']},
                     clobber=clobber)

def nmf_loisel23(X:int=4, Y:int=0, Ncomp:int=3,
                 clobber:bool=False): 
    """
    Perform Non-negative Matrix Factorization (NMF) on Loisel23 data.

    Args:
        X (int): X-coordinate of the training data.
        Y (int): Y-coordinate of the training data.
        Ncomp (int): Number of components for NMF.
        clobber (bool): Flag indicating whether to overwrite existing files.

    """

    # Load training data
    d = load_rs.loisel23_rs(X=X, Y=Y)

    # Loop on IOP
    outfiles = loisel23_filenames('nmf', Ncomp, X, Y)
    for outfile, iop in zip(outfiles, ['a', 'bb']):

        # Remove water
        print("Removing water")
        if iop == 'a':
            iop_w = cross.a_water(d['wave'], data='IOCCG')
        else:
            iop_w = d['bb_w']
        spec = d['inputs'][iop]
        nspec, _ = spec.shape
        spec = spec - np.outer(np.ones(nspec), iop_w)
            
        # Prep for NMF
        new_spec, mask, err  = cnmf_iops.prep(
            spec, sigma=0.05)

        # Do it
        generate_nmf(new_spec, mask, err, outfile, Ncomp, 
                     clobber=clobber,
                     normalize=True,
                     wave=d['wave'],
                     Rs=d['Rs'])
            
    
if __name__ == '__main__':

    # L23
    #pca_loisel23(clobber=True)
    #nmf_loisel23(clobber=True)
    nmf_loisel23(Ncomp=4, clobber=True)
    #nmf_loisel23(Ncomp=5, clobber=True)
    #generate_l23_tara_pca()  # Broken