""" Module to analyze the Inference outputs (i.e. chains) """
import datetime

import numpy as np

import torch

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ihop.iops.decompose import reconstruct_nmf
from ihop.iops.decompose import reconstruct_pca
from ihop.iops.decompose import reconstruct_int

from IPython import embed

def chop_chains(chains:np.ndarray, burn:int=7000, thin:int=1):
    """
    Chop the chains to remove the burn-in and thin.

    Args:
        chains (np.ndarray): The chains to be chopped.
            nsample, nsteps, nwalkers, ndim
        burn (int, optional): The number of burn-in samples to be removed. Defaults to 3000.
        thin (int, optional): The thinning factor. Defaults to 1.

    Returns:
        np.ndarray: The chopped chains.
    """
    # Chop
    nspec = chains.shape[0]
    chains = chains[:, burn::thin, :, :].reshape(nspec, -1, chains.shape[-1])
    return chains

def calc_Rrs(emulator, chains:np.ndarray, quick_and_dirty:bool=False,
             verbose:bool=False):
    """
    Calculate Rrs values for each chain in the given emulator.

    We take simple median here for the chains
        but should take the full distribution for correlations. 

    Args:
        emulator: The emulator object used for prediction.
        chains (np.ndarray): The chains containing coefficients for each spectrum.
        quick_and_dirty (bool, optional): Flag to use a quick and dirty method. Defaults to False.

    Returns:
        tuple: 
            np.ndarray: An array of Rrs values calculated for each chain.
            np.ndarray: An array of uncertainty in Rrs values 
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find the median coefficients
    nspec = chains.shape[0]

    list_Rrs = []
    list_stdRrs = []
    if quick_and_dirty:
        coeff_med = np.median(chains, axis=1)
        # Calc Rrs
        for ss in range(nspec):
            Rs = emulator.prediction(coeff_med[ss,:], device)
            list_Rrs.append(Rs)
            list_stdRrs.append(np.zeros(Rs.size))
    else:
        start = datetime.datetime.now()
        for ss in range(nspec):
            tmp = []
            for item in chains[ss]:
                pred_Rs = emulator.prediction(item, device)
                tmp.append(pred_Rs)
            list_Rrs.append(np.median(np.array(tmp), axis=0))
            list_stdRrs.append(np.std(np.array(tmp), axis=0))
            if verbose and (ss % 10) == 0:
                print(f'{ss} in {(datetime.datetime.now()-start).seconds}s')

    # Turn into a numpy array
    return np.array(list_Rrs), np.array(list_stdRrs)


def calc_iop(iop_chains:np.ndarray, decomp:str, d_iop:dict):
    """
    Calculate the mean and standard deviation of the reconstructed IOPs.

    Parameters:
        iop_chains (np.ndarray): Array of IOP chains.
        decomp (str): Decomposition method ('pca' or 'nmf').
        d_iop (dict): Dictionary of IOP values.

    Returns:
        tuple: 
            np.ndarray: Array of mean IOP values.
            np.ndarray: Array of standard deviation of IOP values.
    """

    # Prep
    if decomp == 'pca':
        rfunc = reconstruct_pca
    elif decomp == 'nmf':
        rfunc = reconstruct_nmf
    elif decomp == 'int':
        rfunc = reconstruct_int
    else:
        raise ValueError("Bad decomp")

    # a
    all_mean = []
    all_std = []
    for idx in range(iop_chains.shape[0]):
        _, iop_recon = rfunc(iop_chains[idx], d_iop, idx)
        iop_mean = np.median(iop_recon, axis=0)
        iop_std = np.std(iop_recon, axis=0)
        # Save
        all_mean.append(iop_mean)
        all_std.append(iop_std)

    # Return
    return np.array(all_mean), np.array(all_std)