""" Decompose IOPs """

import os
import numpy as np

from oceancolor.iop import cross

from cnmf.oceanography import iops as cnmf_iops

from ihop.training_sets import load_rs
from ihop.iops.decompose import generate_pca
from ihop.iops.decompose import generate_nmf
from ihop.iops.decompose import generate_int
from ihop.iops import io as iops_io

from IPython import embed
        

def decompose_loisel23_iop(decomp:str, Ncomp:int, iop:str,
                           X:int=4, Y:int=0, 
                       clobber:bool=False): 
    """
    Decompose one of the Loisel+23 IOPs.

    Args:
        decomp (str): Decomposition method ('pca' or 'nmf').    
        Ncomp (int): Number of components for NMF.
        iop (tuple): The IOP to decompose. 
        X (int, optional): X-coordinate of the training data.
        Y (int, optional): Y-coordinate of the training data.
        clobber (bool): Flag indicating whether to overwrite existing files.

    """
    # Loop on IOP
    outfile = iops_io.loisel23_filename(decomp, iop, Ncomp, X, Y)

    # Load training data
    spec, wave, Rs, d = iops_io.load_loisel23_iop(
        iop, X=X, Y=Y, remove_water=True)

    # Go
    if decomp == 'nmf':
        # Prep for NMF
        new_spec, mask, err  = cnmf_iops.prep(
            spec, sigma=0.05)

        # Do it
        generate_nmf(new_spec, mask, err, outfile, Ncomp, 
                    clobber=clobber,
                    normalize=True,
                    wave=wave,
                    Rs=Rs)
    elif decomp == 'pca':
        generate_pca(spec, outfile, Ncomp,
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

    # L23 + PCA
    if flg & (2**0):
        decompose_loisel23_iop('pca', 4, 'a', clobber=True)  # for a
        decompose_loisel23_iop('pca', 2, 'bb', clobber=True)  # for bb

    # L23 + Interpolate,NMF
    if flg & (2**1):
        decompose_loisel23_iop('int', 40, 'a', clobber=True)  # for a
        #decompose_loisel23_iop('nmf', 2, 'bb', clobber=True)  # for bb
    
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- L23 + PCA
        #flg += 2 ** 1  # 2 -- L23 + Int on a
        #flg += 2 ** 2  # 4 -- L23 + NMF 4
        #flg += 2 ** 3  # 8 -- L23 + NMF 4,3

        #flg += 2 ** 4  # 16 -- L23 + PCA 4,2 + norm_Rs=False

        
    else:
        flg = sys.argv[1]

    main(flg)

    # L23
    # PCA

    #nmf_loisel23(clobber=True)
    #nmf_loisel23(Ncomp=2, clobber=True)
    #nmf_loisel23(Ncomp=3, clobber=True)
    #nmf_loisel23(Ncomp=4, clobber=True)
    #nmf_loisel23(Ncomp=5, clobber=True)
    #generate_l23_tara_pca()  # Broken