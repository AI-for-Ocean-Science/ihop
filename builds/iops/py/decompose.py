""" Decompose IOPs """

import os

#from ihop.iops.pca import generate_l23_pca
#from ihop.iops.pca import generate_l23_tara_pca

from ihop.training_sets import load_rs
from ihop.iops.decompose import generate_pca

def load_training(datasets:list):

    # L23?
    if 'L23' in datasets:
    # Load
        spec_nw, mask, err, wave, Rs = iops.prep_loisel23(
        iop, min_wv=min_wv, remove_water=True,
        high_cut=high_cut)
        
def pca_loisel23(X:int=4, Y:int=0, Ncomp:int=3,
                 clobber:bool=False):

    # Load training data
    d = load_rs.loisel23_rs(X=X, Y=Y)

    # PCA
    for iop in ['a', 'bb']:
        # Prep
        outroot = f'pca_L23_X{X}Y{Y}_{iop}'
        # Do it
        generate_pca(d['inputs'][iop], outroot, Ncomp,
                     extras={'Rs':d['Rs'], 'wave':d['wave']},
                     clobber=clobber)

    
if __name__ == '__main__':

    # L23
    pca_loisel23(clobber=True)
    #generate_l23_tara_pca()  # Broken