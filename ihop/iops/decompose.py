""" Parameterizations of IOPs, i.e. decompositions """

import os
import numpy as np

from importlib import resources

from oceancolor.utils import pca
from oceancolor.hydrolight import loisel23

from cnmf.oceanography.iops import tara_matched_to_l23
from cnmf import nmf_imaging
from cnmf import io as cnmf_io

from IPython import embed

# Globals
pca_path = os.path.join(resources.files('ihop'),
                            'data', 'PCA')
nmf_path = os.path.join(resources.files('ihop'),
                            'data', 'NMF')

def loisel23_filenames(decomp:str, Ncomp:int,
                       X:int, Y:int):
    """
    Generate filenames for Loisel23 decomposition.

    Args:
        decomp (str): The decomposition type. pca, nmf
        Ncomp (int): The number of components.
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 

    Returns:
        tuple: A tuple containing the filenames for L23_a and L23_bb.
    """
    # Load up data
    d_path = os.path.join(resources.files('ihop'),
                            'data', decomp.upper())
    l23_a_file = os.path.join(d_path, f'{decomp}_L23_X{X}Y{Y}_a_N{Ncomp:02d}.npz')
    l23_bb_file = os.path.join(d_path, f'{decomp}_L23_X{X}Y{Y}_bb_N{Ncomp:02d}.npz')

    # Return
    return l23_a_file, l23_bb_file

def load_loisel2023(decomp:str, Ncomp:int, X:int=4, Y:int=0, 
                    scale_Rs:float=1.e4):
    """ Load the NMF or PCA-based parameterization of IOPs from Loisel 2023

    Args:
        decomp (str): The decomposition type. pca, nmf
        Ncomp (int): Number of components.
        X (int, optional): simulation scenario   
        Y (int, optional):  solar zenith angle used in the simulation, and 

    Returns:
        tuple: 
            - **ab** (*np.ndarray*) -- PCA coefficients
            - **Rs** (*np.ndarray*) -- Rrs values scaled by 1e4
            - **d_a** (*dict*) -- dict of PCA 
            - **d_bb** (*dict*) -- dict of PCA
    """
    # Filenames
    l23_a_file, l23_bb_file = loisel23_filenames(
        decomp, Ncomp, X, Y)

    # Load up
    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)
    key = 'Y' if decomp == 'pca' else 'coeff'

    nparam = d_a[key].shape[1]+d_bb[key].shape[1]
    ab = np.zeros((d_a[key].shape[0], nparam))
    ab[:,0:d_a[key].shape[1]] = d_a[key]
    ab[:,d_a[key].shape[1]:] = d_bb[key]

    # Rs
    Rs = d_a['Rs'] * scale_Rs

    # Return
    return ab, Rs, d_a, d_bb

def generate_nmf(iop_data:np.ndarray, mask:np.ndarray, 
                 err:np.ndarray,
                 outfile:str,
                 Ncomp:int,
                 clobber:bool=False, 
                 nmf_path:str=nmf_path,
                 normalize:bool=True,
                 wave:np.ndarray=None,
                 Rs:np.ndarray=None,
                 seed:int=12345):
    """ Generate NMF model for input IOP 

    Args:
        iop_data (np.ndarray): IOP data (n_samples, n_features, 1)
        mask (np.ndarray): Mask (n_samples, n_features, 1)
        err (np.ndarray): Error (n_samples, n_features, 1)
        outroot (str): Output root.
        Ncomp (int): Number of PCA components. Defaults to 3.
        clobber (bool, optional): Clobber existing model? Defaults to False.
        nmf_path (str, optional): Path for output NMF files. Defaults to nmf_path.
        seed (int, optional): Random seed. Defaults to 12345.
        wave (np.ndarray, optional): Wavelengths. Defaults to None.
            Only for convenience in the saved file
        Rs (np.ndarray, optional): Rrs. Defaults to None.
            Only for convenience in the saved file
    """
    # Prep
    outfile = os.path.join(nmf_path, outfile)
    outroot = outfile.replace('.npz','')
    # Decompose
    comps = nmf_imaging.NMFcomponents(
        ref=iop_data, mask=mask, ref_err=err,
        n_components=Ncomp,
        path_save=outroot, oneByOne=True,
        normalize=normalize,
        seed=seed)
    # Gather up
    M = np.load(outroot+'_comp.npy').T
    coeff = np.load(outroot+'_coef.npy').T

    # Save
    cnmf_io.save_nmf(outfile, M, coeff, iop_data[...,0],
                     mask[...,0], err[...,0], wave, Rs)

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



def reconstruct_pca(Y:np.ndarray, pca_dict:dict, idx:int):
    """
    Reconstructs the original data point from the PCA-encoded representation.

    Args:
        Y (np.ndarray): The PCA-encoded representation of the data point.
        pca_dict (dict): A dictionary containing the PCA transformation parameters.
        idx (int): The index of the data point to reconstruct.

    Returns:
        tuple: A tuple containing the original data and its reconstructed version.
    """
    # Grab the original
    orig = pca_dict['data'][idx]

    # Reconstruct
    recon = np.dot(Y, pca_dict['M']) + pca_dict['mean']

    return orig, recon

def reconstruct_nmf(Y:np.ndarray, nmf_dict:dict, idx:int):
    """
    Reconstructs the original data point from the PCA-encoded representation.

    Args:
        Y (np.ndarray): The NMF-encoded representation of the data point.
        pca_dict (dict): A dictionary containing the NMF transformation parameters.
        idx (int): The index of the data point to reconstruct.

    Returns:
        tuple: A tuple containing the original data and its reconstructed version.
    """
    # Grab the original
    orig = nmf_dict['spec'][idx]

    # Reconstruct
    recon = np.dot(Y, nmf_dict['M'])

    return orig, recon