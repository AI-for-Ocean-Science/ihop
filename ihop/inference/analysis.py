""" Module to analyze the Inference outputs (i.e. chains) """

import numpy as np

import torch

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

def calc_Rrs(emulator, chains:np.ndarray):
    """
    Calculate Rrs values for each chain in the given emulator.

    We take simple median here for the chains
        but should take the full distribution for correlations. 

    Args:
        emulator: The emulator object used for prediction.
        chains (np.ndarray): The chains containing coefficients for each spectrum.

    Returns:
        np.ndarray: An array of Rrs values calculated for each chain.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find the median coefficients
    nspec = chains.shape[0]
    coeff_med = np.median(chains, axis=1)

    # Calc Rrs
    list_Rrs = []
    for ss in range(nspec):
        Rs = emulator.prediction(coeff_med[ss,:], device)
        list_Rrs.append(Rs)

    # Turn into a numpy array
    return np.array(list_Rrs)

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