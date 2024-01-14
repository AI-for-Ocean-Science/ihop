""" Methods for IHOP I/O """

import os

import numpy as np
import torch

from oceancolor.hydrolight import loisel23

from ihop.iops.decompose import load_loisel2023
from ihop.emulators import io as ihop_io

from IPython import embed

def load_l23_emulator(Ncomp:int, X:int=4, Y:int=0, decomp:str='pca'):
    """
    Load the L23 emulator model.

    Args:
        Ncomp (int): The number of components.
        X (int, optional): simulation scenario   
        Y (int, optional):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.
        decomp (str, optional): The decomposition method. Can be 'pca' or 'nmf'.

    Returns:
        emulator (object): The loaded emulator model.
        emulator_file (str): The file path of the loaded emulator model.
    """

    # Load model
    path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 
                        'Emulators', f'DenseNet_{decomp.upper()}') 
    if decomp == 'pca':
        arch = '512_512_512_256'
    elif decomp == 'nmf':
        arch = '512_512_512_256'
    else:
        raise ValueError("Bad decomp")
    emulator_file = os.path.join(path,
        f'dense_l23_{decomp}_X{X}Y{Y}_N{Ncomp:02d}_{arch}_chl.pth')

    # Load
    emulator = ihop_io.load_nn(emulator_file)

    # Return
    return emulator, emulator_file


def load_l23_data(X:int=4, Y:int=0, decomp:str='pca'):
    """
    Load L23 data

    Args:
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.
        decomp (str): Decomposition method.

    Returns:
        tuple: Tuple containing ab, Chl, Rs, d_a, d_bb, and model.
            - ab (numpy.ndarray): Absorption coefficients.
            - Chl (numpy.ndarray): Chlorophyll concentration.
            - Rs (numpy.ndarray): Remote sensing reflectance.
            - d_a (dict): Hydrolight data for absorption coefficients.
            - d_bb (dict): Hydrolight data for backscattering coefficients.
    """
    print("Loading... ")
    ab, Rs, d_a, d_bb = load_loisel2023(decomp)
    # Chl
    ds_l23 = loisel23.load_ds(X, Y)
    Chl = loisel23.calc_Chl(ds_l23)

    # Return
    return ab, Chl, Rs, d_a, d_bb

# #############################################
def load_l23_fit(in_idx:int, chop_burn = -3000,
            iop_type:str='nmf', 
            chains_only:bool=False, perc:int=10,
            X:int=4, Y:int=0):
    """
    Load data and perform calculations for the IHOP project.

    Args:
        in_idx (int): The index of the data to work on.
        chop_burn (int, optional): The number of burn-in samples to discard from the MCMC chains. Defaults to -3000.
        iop_type (str, optional): The type of IOP (Inherent Optical Property) model to use. Defaults to 'nmf'.
        chains_only (bool, optional): Whether to return only the MCMC chains or not. Defaults to False.
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.
        perc (int, optional): The percentile to use for the MCMC chains. Defaults to 10.

    Returns:
        tuple: A tuple containing various data arrays and values:
            - d_a (dict): Hydrolight data for absorption coefficients.
            - idx (int): The L23 index.
            - orig (numpy.ndarray): Original data.
            - a_mean (numpy.ndarray): Mean of the reconstructed absorption coefficients.
            - a_std (numpy.ndarray): Standard deviation of the reconstructed absorption coefficients.
            - a_pca (numpy.ndarray): Reconstructed absorption coefficients using PCA.
            - obs_Rs (numpy.ndarray): Observed remote sensing reflectance.
            - pred_Rs (numpy.ndarray): Predicted remote sensing reflectance.
            - std_pred (numpy.ndarray): Standard deviation of the predicted remote sensing reflectance.
            - NN_Rs (numpy.ndarray): Remote sensing reflectance predicted by the neural network model.
            - Rs (numpy.ndarray): Remote sensing reflectance.
            - ab (numpy.ndarray): Absorption coefficients.
            - allY (numpy.ndarray): All MCMC samples.
            - wave (numpy.ndarray): Wavelength data.
            - orig_bb (numpy.ndarray): Original data for backscattering coefficients.
            - bb_mean (numpy.ndarray): Mean of the reconstructed backscattering coefficients.
            - bb_std (numpy.ndarray): Standard deviation of the reconstructed backscattering coefficients.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Chains
    out_path = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'Fits', 'L23')
    chain_file = f'fit_L23_{iop_type.upper()}_NN_Rs{perc:02d}.npz'

    # Start
    ab, Chl, Rs, d_a, d_bb, model = load_l23_data_model(iop_type=iop_type,
                                             X=X, Y=Y)
           
    # MCMC
    print(f"Loading MCMC: {chain_file}")
    d = np.load(os.path.join(out_path, chain_file))
    chains = d['chains']
    if in_idx is not None:
        chains = chains[in_idx]
    l23_idx = d['idx']
    obs_Rs = d['obs_Rs']
    if chains_only:
        return chains, d, ab, Chl, d_a

    idx = l23_idx[in_idx]
    print(f'Working on: L23 index={idx}')

    # Parse
    nwave = d_a['wave'].size
    wave = d_a['wave']
    ncomp = 3

    # Prep
    if iop_type == 'pca':
        rfunc = ihop_pca.reconstruct
    elif iop_type == 'nmf':
        rfunc = ihop_nmf.reconstruct
    
    # a
    Y = chains[chop_burn:, :, 0:ncomp].reshape(-1,ncomp)
    orig, a_recon = rfunc(Y, d_a, idx)
    a_mean = np.mean(a_recon, axis=0)
    a_std = np.std(a_recon, axis=0)
    _, a_pca = rfunc(ab[idx][:ncomp], d_a, idx)

    # bb
    Y = chains[chop_burn:, :, ncomp:].reshape(-1,ncomp)
    orig_bb, bb_recon = rfunc(Y, d_bb, idx)
    bb_mean = np.mean(bb_recon, axis=0)
    bb_std = np.std(bb_recon, axis=0)
    #_, a_pca = rfunc(ab[idx][:ncomp], d_a, idx)

    # Rs
    allY = chains[chop_burn:, :, :].reshape(-1,ncomp*2+1) # Chl
    all_pred = np.zeros((allY.shape[0], nwave))
    for kk in range(allY.shape[0]):
        Ys = allY[kk]
        pred_Rs = model.prediction(Ys, device)
        all_pred[kk,:] = pred_Rs

    pred_Rs = np.median(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)
    NN_Rs = model.prediction(ab[idx].tolist() + [Chl[idx]], device)

    return d_a, idx, orig, a_mean, a_std, a_pca, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, Rs, ab, allY, wave,\
        orig_bb, bb_mean, bb_std