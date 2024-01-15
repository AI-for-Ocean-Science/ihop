""" Decompose IOPs """

import os


from cnmf.oceanography import iops as cnmf_iops

from ihop.training_sets import load_rs
from ihop.iops.decompose import generate_pca
from ihop.iops.decompose import generate_nmf

        
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

    # Load training data
    d = load_rs.loisel23_rs(X=X, Y=Y)

    # Loop on IOP
    for iop in ['a', 'bb']:
        # Prep
        # TODO -- Replace this with loisel23_filenames in iops.decompose.py
        outfile = f'nmf_L23_X{X}Y{Y}_{iop}_N{Ncomp:02d}.npz'
        new_spec, mask, err  = cnmf_iops.prep(d['inputs'][iop],
                                              sigma=0.05)
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
    #nmf_loisel23(Ncomp=4, clobber=True)
    nmf_loisel23(Ncomp=5, clobber=True)
    #generate_l23_tara_pca()  # Broken