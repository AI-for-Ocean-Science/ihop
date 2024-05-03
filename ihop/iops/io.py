""" Load IOP data """
import os

from importlib import resources

from oceancolor.hydrolight import loisel23
from oceancolor.water import absorption

import numpy as np


def loisel23_filename(decomp:str, iop:str, Ncomp:int,
                       X:int, Y:int):
    """
    Generate filenames for Loisel23 decomposition.

    Args:
        decomp (str): The decomposition type. pca, nmf, int
        iop (str): The IOP decomposed. a, bb
        Ncomp (int): The number of components
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 

    Returns:
        tuple: A tuple containing the filenames for L23_a and L23_bb.
    """
    root = f'{decomp}_L23_X{X}Y{Y}_{iop}_N{Ncomp:02d}'
    # Load up data
    d_path = os.path.join(resources.files('ihop'),
                            'data', decomp.upper())
    l23_file = os.path.join(d_path, f'{root}.npz')
    return l23_file

def loisel23_filenames(decomps:tuple, Ncomps:tuple,
                       X:int, Y:int):
    l23_a_file = loisel23_filename(decomps[0], 'a', Ncomps[0], X, Y)
    l23_bb_file = loisel23_filename(decomps[1], 'bb', Ncomps[1], X, Y)
    # Return
    return l23_a_file, l23_bb_file


def load_loisel23_iop(iop:str, X:int=4, Y:int=0, 
    min_wv:float=300., high_cut:float=1000.,
    remove_water:bool=False):

    # Load
    ds = loisel23.load_ds(X, Y)

    # Unpack and cut
    spec = ds[iop].data
    wave = ds.Lambda.data 
    Rs = ds.Rrs.data

    cut = (wave >= min_wv) & (wave <= high_cut)
    spec = spec[:,cut]
    wave = wave[cut]
    Rs = Rs[:,cut]

    # Remove water
    if iop == 'a' and remove_water:
        iop_w = absorption.a_water(wave, data='IOCCG')
    elif iop == 'bb' and remove_water:
        iop_w = d['bb_w']

    nspec, _ = spec.shape
    spec = spec - np.outer(np.ones(nspec), iop_w)

    # Return
    return spec, wave, Rs, ds

def load_loisel2023_decomp(decomps:tuple, 
                           Ncomps:tuple, X:int=4, 
                           Y:int=0, scale_Rs:float=1.e4):
    """ Load the NMF or PCA-based parameterization of IOPs from Loisel 2023

    Args:
        decomps (tuple): The decomposition type. pca, nmf
            for (a,bb)
        Ncomps (tuple): Number of components. (a,bb)
        X (int, optional): simulation scenario   
        Y (int, optional):  solar zenith angle used in the simulation, and 

    Returns:
        tuple: 
            - **ab** (*np.ndarray*) -- coefficients
            - **Rs** (*np.ndarray*) -- Rrs values scaled by 1e4
            - **d_a** (*dict*) -- dict of PCA 
            - **d_bb** (*dict*) -- dict of PCA
    """
    # Filenames
    l23_a_file, l23_bb_file = loisel23_filenames(
        decomps, Ncomps, X, Y)

    # Load up
    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)
    keya = 'Y' if decomps[0] == 'pca' else 'coeff'
    keyb = 'Y' if decomps[1] == 'pca' else 'coeff'

    nparam = d_a[keya].shape[1]+d_bb[keya].shape[1]
    ab = np.zeros((d_a[keya].shape[0], nparam))
    ab[:,0:d_a[keya].shape[1]] = d_a[keya]
    ab[:,d_a[keya].shape[1]:] = d_bb[keyb]

    # Rs
    Rs = d_a['Rs'] * scale_Rs

    # Return
    return ab, Rs, d_a, d_bb