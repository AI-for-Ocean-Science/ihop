""" Decompose Rrs """

import os
import numpy as np

from oceancolor.iop import cross
from oceancolor.hydrolight import loisel23

from cnmf.oceanography import iops as cnmf_iops

from ihop.training_sets import load_rs
from ihop.iops.decompose import generate_pca
from ihop.iops.decompose import generate_nmf
from ihop.iops.decompose import generate_int
from ihop.iops import io as iops_io

from IPython import embed
        

def decompose_loisel23_Rrs(decomp:str, Ncomp:int, 
                           X:int=4, Y:int=0, 
                           clobber:bool=False): 
    """
    Decompose Loisel+23 Rrs

    Args:
        decomp (str): Decomposition method ('pca' or 'nmf').    
        Ncomp (int): Number of components for NMF.
        X (int, optional): X-coordinate of the training data.
        Y (int, optional): Y-coordinate of the training data.
        clobber (bool): Flag indicating whether to overwrite existing files.

    """
    comp = 'Rrs'
    outfile = iops_io.loisel23_filename(decomp, comp, Ncomp, X, Y)

    # Load
    ds = loisel23.load_ds(X, Y)

    # Unpack 
    wave = ds.Lambda.data 
    Rrs = ds.Rrs.data

    # Go
    if decomp == 'nmf':
        # Prep for NMF
        new_spec, mask, err  = cnmf_iops.prep(
            Rrs, sigma=0.05)

        # Do it
        generate_nmf(new_spec, mask, err, outfile, Ncomp, 
                    clobber=clobber,
                    normalize=True,
                    wave=wave)
    elif decomp == 'pca':
        generate_pca(spec, outfile, Ncomp,
                    extras={'Rs':Rs, 'wave':wave},
                    clobber=clobber)
    elif decomp == 'npca':
        generate_pca(spec, outfile, Ncomp, norm=True,
                    extras={'Rs':Rs, 'wave':wave},
                    clobber=clobber)
    elif decomp == 'int': # interpolate
        generate_int(spec, outfile, Ncomp, wave,
                    extras={'Rs':Rs, 'wave':wave},
                    clobber=clobber)
    else:
        raise ValueError("Bad decomp")
                
def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # NMF``
    if flg & (2**0):
        decompose_loisel23_Rrs('nmf', 2, clobber=True)  # for a

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- NMF, 2

        
    else:
        flg = sys.argv[1]

    main(flg)
