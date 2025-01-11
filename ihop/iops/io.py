""" Load IOP data """
import os

from importlib import resources

from ocpy.hydrolight import loisel23
from ocpy.water import absorption

import numpy as np

from IPython import embed


def loisel23_filename(decomp:str, iop:str, Ncomp:int,
                       X:int, Y:int, d_path:str=None):
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
    if d_path is None:
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
    """
    Load Loisel23 IOP data.

    Parameters:
        iop (str): The type of IOP data to load ('a' for absorption or 'bb' for backscattering).
        X (int): The X-coordinate of the data to load.
        Y (int): The Y-coordinate of the data to load.
        min_wv (float): The minimum wavelength to include in the loaded data.
        high_cut (float): The maximum wavelength to include in the loaded data.
        remove_water (bool): Whether to remove the water contribution from the loaded data.

    Returns:
        tuple: A tuple containing the loaded IOP data, the corresponding wavelengths, the remote sensing reflectances, and the original dataset.
    """
    # Load
    ds = loisel23.load_ds(X, Y)

    # Unpack 
    spec = ds[iop].data
    wave = ds.Lambda.data 
    Rs = ds.Rrs.data

    # Remove water?
    if remove_water:
        if iop == 'a':
            iop_w = absorption.a_water(wave, data='IOCCG')
            nspec, _ = spec.shape
            spec = spec - np.outer(np.ones(nspec), iop_w)
        elif iop == 'bb':
            spec = ds['bbnw']
        
    # Cut
    cut = (wave >= min_wv) & (wave <= high_cut)
    spec = spec[:,cut]
    wave = wave[cut]
    Rs = Rs[:,cut]

    # Return
    return spec, wave, Rs, ds

def load_loisel2023_decomp(decomps:tuple, 
                           Ncomps:tuple, X:int=4, 
                           Y:int=0, scale_Rs:float=1.e4):
    """ Load the INT, NMF or PCA-based parameterization of IOPs from Loisel 2023

    Args:
        decomps (tuple): The decomposition type. pca, nmf
            for (a,bb)
        Ncomps (tuple): Number of components. (a,bb)
        X (int, optional): simulation scenario   
        Y (int, optional):  solar zenith angle used in the simulation, and 
        scale_Rs (float, optional): Scaling factor for Rs. Defaults to 1.e4.

    Returns:
        tuple: 
            - **ab** (*np.ndarray*) -- coefficients
            - **Rs** (*np.ndarray*) -- Rrs values scaled by 1e4
            - **d_a** (*dict*) -- dict of decomposition
            - **d_bb** (*dict*) -- dict of decomposition
    """
    # Filenames
    l23_a_file, l23_bb_file = loisel23_filenames(
        decomps, Ncomps, X, Y)
    print(f"Loading decomps from {l23_a_file} and {l23_bb_file}")

    # Load up
    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)

    # Prep
    keys = dict(pca='Y', nmf='coeff', int='new_spec', 
                npca='Y', bsp='coeffs', hyb='coeff')
    keya = keys[decomps[0]]
    keyb = keys[decomps[1]]

    nparam = d_a[keya].shape[1]+d_bb[keyb].shape[1]
    ab = np.zeros((d_a[keya].shape[0], nparam))

    ab[:,0:d_a[keya].shape[1]] = d_a[keya]
    ab[:,d_a[keya].shape[1]:] = d_bb[keyb]

    # Rs
    Rs = d_a['Rs'] * scale_Rs

    # Return
    return ab, Rs, d_a, d_bb