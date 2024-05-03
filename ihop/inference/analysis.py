""" Module to analyze the Inference outputs (i.e. chains) """

import numpy as np

import torch

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
