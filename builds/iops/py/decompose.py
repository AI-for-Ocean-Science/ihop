""" Decompose IOPs """

import os
import numpy as np

from oceancolor.iop import cross

from cnmf.oceanography import iops as cnmf_iops

from ihop.training_sets import load_rs
from ihop.iops.decompose import generate_pca
from ihop.iops.decompose import generate_nmf
from ihop.iops.decompose import loisel23_filename

        

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

    # Load training data
    d = load_rs.loisel23_rs(X=X, Y=Y)

    # Loop on IOP
    outfile = loisel23_filename(decomp, iop, Ncomp, X, Y)

    # Remove water
    print("Removing water")
    if iop == 'a':
        iop_w = cross.a_water(d['wave'], data='IOCCG')
    else:
        iop_w = d['bb_w']
    spec = d['inputs'][iop]
    nspec, _ = spec.shape
    spec = spec - np.outer(np.ones(nspec), iop_w)
        
    # Go
    if decomp == 'nmf':
        # Prep for NMF
        new_spec, mask, err  = cnmf_iops.prep(
            spec, sigma=0.05)

        # Do it
        generate_nmf(new_spec, mask, err, outfile, Ncomp, 
                    clobber=clobber,
                    normalize=True,
                    wave=d['wave'],
                    Rs=d['Rs'])
    elif decomp == 'pca':
        generate_pca(d['inputs'][iop], outfile, Ncomp,
                    extras={'Rs':d['Rs'], 'wave':d['wave']},
                    clobber=clobber)
    elif decomp == 'int': # interpolate
        generate_int(d['inputs'][iop], outfile, Ncomp,
                    extras={'Rs':d['Rs'], 'wave':d['wave']},
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
        decompose_loisel23_iop('pca', clobber=True, Ncomp=2)  # for bb
        decompose_loisel23_iop('pca', clobber=True, Ncomp=4)  # for a
    
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- L23 + PCA
        #flg += 2 ** 1  # 2 -- L23 + NMF
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