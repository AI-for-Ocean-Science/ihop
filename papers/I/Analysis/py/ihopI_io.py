""" I/O for IHOP paper I """

import os

import numpy as np
import torch

from oceancolor.hydrolight import loisel23

from ihop.iops.pca import load_loisel_2023_pca
from ihop.emulators import io as ihop_io
from ihop.iops import pca as ihop_pca
from ihop.iops import nmf as ihop_nmf

from IPython import embed

def load_l23_data_model(X:int=4, Y:int=0, iop_type:str='pca'):
    """
    Load L2.3 data and model.

    Args:
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.
        iop_type (str): Decomposition method.

    Returns:
        tuple: Tuple containing ab, Chl, Rs, d_a, d_bb, and model.
            - ab (numpy.ndarray): Absorption coefficients.
            - Chl (numpy.ndarray): Chlorophyll concentration.
            - Rs (numpy.ndarray): Remote sensing reflectance.
            - d_a (dict): Hydrolight data for absorption coefficients.
            - d_bb (dict): Hydrolight data for backscattering coefficients.
            - model (torch.nn.Module): Neural network model.
    """
    print("Loading... ")
    ab, Rs, d_a, d_bb = load_loisel_2023_pca()
    # Chl
    ds_l23 = loisel23.load_ds(X, Y)
    Chl = loisel23.calc_Chl(ds_l23)

    # Load model
    if iop_type == 'pca':
        model_file = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators',
            'DenseNet_PCA',
            'dense_l23_pca_X4Y0_512_512_512_256_chl.pth')
    else:
        raise ValueError("Bad decomp")
    # Load
    model = ihop_io.load_nn(model_file)
    # Return
    return ab, Chl, Rs, d_a, d_bb, model

# #############################################
# Load
def load_l23_fit(in_idx:int, chop_burn = -3000,
            iop_type:str='nmf', use_quick:bool=False,
            chains_only:bool=False, perc:int=10,
            X:int=4, Y:int=0):
    """
    Load data and perform calculations for the IHOP project.

    Args:
        in_idx (int): The index of the data to work on.
        chop_burn (int, optional): The number of burn-in samples to discard from the MCMC chains. Defaults to -3000.
        iop_type (str, optional): The type of IOP (Inherent Optical Property) model to use. Defaults to 'nmf'.
        use_quick (bool, optional): Whether to use a quick fit or not. Defaults to False.
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

    if use_quick:
        out_path = './'
        chain_file = 'quick_fit_nmf.npz'

    # Start
    ab, Chl, Rs, d_a, d_bb, model = load_l23_data_model(iop_type=iop_type,
                                             X=X, Y=Y)
           
    # MCMC
    print(f"Loading MCMC: {chain_file}")
    d = np.load(os.path.join(out_path, chain_file))
    chains = d['chains']
    if use_quick:
        pass
    else:
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