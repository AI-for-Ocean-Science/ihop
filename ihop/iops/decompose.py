""" Parameterizations of IOPs, i.e. decompositions """

import os
import numpy as np

from importlib import resources

from oceancolor.utils import pca
from oceancolor.hydrolight import loisel23

from cnmf.oceanography.iops import tara_matched_to_l23

from IPython import embed

pca_path = os.path.join(resources.files('ihop'),
                            'data', 'PCA')

def load(pca_file:str, pca_path:str=pca_path):
    return np.load(os.path.join(pca_path, pca_file))

def load_loisel_2023_pca(N_PCA:int=3, l23_path:str=None,
    X:int=4, Y:int=0):
    """ Load the PCA-based parameterization of IOPs from Loisel 2023

    Args:
        N_PCA (int, optional): Number of PCA components. Defaults to 3.
        l23_path (str, optional): Path to PCA files. Defaults to None.
            If None, uses the default path of ihop/data/PCA
        X (int, optional): X. Defaults to 4.
        Y (int, optional): Y. Defaults to 0.

    Returns:
        tuple: 
            - **ab** (*np.ndarray*) -- PCA coefficients
            - **Rs** (*np.ndarray*) -- Rrs values
            - **d_a** (*dict*) -- dict of PCA 
            - **d_bb** (*dict*) -- dict of PCA
    """
    # Load up data
    if l23_path is None:
        l23_path = os.path.join(resources.files('ihop'),
                            'data', 'PCA')
    l23_a_file = os.path.join(l23_path, f'pca_L23_X{X}Y{Y}_a_N{N_PCA}.npz')
    l23_bb_file = os.path.join(l23_path, f'pca_L23_X{X}Y{Y}_bb_N{N_PCA}.npz')

    # Load up
    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)
    nparam = d_a['Y'].shape[1]+d_bb['Y'].shape[1]
    ab = np.zeros((d_a['Y'].shape[0], nparam))
    ab[:,0:d_a['Y'].shape[1]] = d_a['Y']
    ab[:,d_a['Y'].shape[1]:] = d_bb['Y']

    Rs = d_a['Rs']

    # Return
    return ab, Rs, d_a, d_bb

def load_data(data_path, back_scatt:str='bb'):
    # Load up data
    data = np.load(data_path)
    features_data = data["abb"]
    labels_data = data["Rs"]
    return features_data, labels_data


def generate_pca(iop_data:np.ndarray,
                 outroot:str,
                 Ncomp:int,
                 clobber:bool=False, 
                 extras:dict=None,
                 pca_path:str=pca_path):
    """ Generate PCA model for input IOP 

    Args:
        iop_data (np.ndarray): IOP data (n_samples, n_features)
        outroot (str): Output root.
        Ncomp (int): Number of PCA components. Defaults to 3.
        clobber (bool, optional): Clobber existing model? Defaults to False.
        pca_path (str, optional): Path for output PCA files. Defaults to pca_path.
        extras (dict, optional): Extra arrays to save. Defaults to None.
    """

    # Prep
    outfile = os.path.join(pca_path, f'{outroot}_N{Ncomp}.npz')

    # Do it
    if not os.path.exists(outfile) or clobber:
        pca.fit_normal(iop_data, Ncomp, save_outputs=outfile,
                       extra_arrays=extras)

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
    wave_grid, tara_a_water, l23_a = tara_matched_to_l23()

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
    """
    Reconstructs the original data point from the PCA-encoded representation.

    Args:
        Y (ndarray): The PCA-encoded representation of the data point.
        pca_dict (dict): A dictionary containing the PCA transformation parameters.
        idx (int): The index of the data point to reconstruct.

    Returns:
        tuple: A tuple containing the original data point and its reconstructed version.
    """
    # Grab the original
    orig = pca_dict['data'][idx]

    # Reconstruct
    recon = np.dot(Y, pca_dict['M']) + pca_dict['mean']

    return orig, recon
