""" PCA-based parameterization of IOPs """

import os
import numpy as np

from importlib import resources

from oceancolor.utils import pca

from ihop.hydrolight import loisel23

pca_path = os.path.join(resources.files('ihop'),
                            'data', 'PCA')

def load(pca_file:str):
    return np.load(os.path.join(pca_path, pca_file))

def load_loisel_2023_pca():
    """ Load the PCA-based parameterization of IOPs from Loisel 2023

    Returns:
        tuple: 
            - **ab** (*np.ndarray*) -- PCA coefficients
            - **Rs** (*np.ndarray*) -- Rrs values
            - **d_a** (*dict*) -- dict of PCA 
            - **d_bb** (*dict*) -- dict of PCA
    """

    # Load up data
    l23_path = os.path.join(resources.files('ihop'),
                            'data', 'PCA')
    l23_a_file = os.path.join(l23_path, 'pca_L23_X4Y0_a_N3.npz')
    l23_bb_file = os.path.join(l23_path, 'pca_L23_X4Y0_bb_N3.npz')

    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)
    nparam = d_a['Y'].shape[1]+d_bb['Y'].shape[1]
    ab = np.zeros((d_a['Y'].shape[0], nparam))
    ab[:,0:d_a['Y'].shape[1]] = d_a['Y']
    ab[:,d_a['Y'].shape[1]:] = d_bb['Y']

    Rs = d_a['Rs']

    # Return
    return ab, Rs, d_a, d_bb




def generate_l23_pca(clobber:bool=False, Ncomp:int=3,
                     X:int=4, Y:int=0):
    """ Generate PCA models for IOPs in Loisel 2023 data

    Args:
        clobber (bool, optional): Clobber existing model? Defaults to False.
        Ncomp (int, optional): Number of PCA components. Defaults to 3.
        X (int, optional): X. Defaults to 4.
        Y (int, optional): Y. Defaults to 0.
    """

    # Load up the data
    ds = loisel23.load_ds(X, Y)

    # Loop on IOPs
    for iop in ['a', 'b', 'bb']:
        outfile = os.path.join(pca_path, f'pca_L23_X{X}Y{Y}_{iop}_N{Ncomp}.npz')
        if not os.path.exists(outfile) or clobber:
            pca.fit_normal(ds[iop].data, Ncomp, save_outputs=outfile,
                           extra_arrays={'Rs':ds.Rrs.data,
                                         'wavelength':ds.Lambda.data})

    # L23 positive, definite

def generate_l23_tara_pca(clobber:bool=False, return_N:int=None):
    """ Generate a PCA for L23 + Tara
        Restricted to 400-705nm

    Args:
        clobber (bool, optional): _description_. Defaults to False.
        return_N (int, optional): _description_. Defaults to None.

    Returns:
        None or tuple:  if return_N is not None, returns
            - **data** (*np.ndarray*) -- data array
            - **wave_grid** (*np.ndarray*) -- wavelength grid
            - **pca_fit** (*dict*) -- PCA fit
    """

    # Load up
    wave_grid, tara_a_water, l23_a = loisel23.tara_matched_to_l23()

    # N components
    data = np.append(l23_a, tara_a_water, axis=0)
    for N in [3,5,20]:
        outfile = os.path.join(pca_path, f'pca_L23_X4Y0_Tara_a_N{N}.npz')
        if not os.path.exists(outfile) or clobber or ( (return_N is not None) 
                                                      and (return_N == N) ):
            print(f"Fit PCA with N={N}")
            pca_fit = pca.fit_normal(data, N, save_outputs=outfile,
                           extra_arrays={'wavelength':wave_grid})
            if return_N is not None and return_N == N:
                return data, wave_grid, pca_fit

def reconstruct(Y, pca_dict, idx):
    # Grab the orginal
    orig = pca_dict['data'][idx]

    # Reconstruct
    recon = np.dot(Y, pca_dict['M']) + pca_dict['mean']

    return orig, recon 


if __name__ == '__main__':
    #generate_l23_pca(clobber=clobber)
    generate_l23_tara_pca()