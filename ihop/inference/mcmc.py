""" MCMC module for IHOP """

import os
from importlib import resources

import numpy as np

import matplotlib.pyplot as plt

import emcee
import corner

import torch

from ihop.emulators import io 


def log_prob(ab, Rs, model, device, scl_sig):
    pred = model.prediction(ab, device)
    #
    sig = scl_sig * Rs
    #
    return -1*0.5 * np.sum( (pred-Rs)**2 / sig**2)


def run_emcee_nn(nn_model, Rs, nwalkers:int=32, nsteps:int=20000,
                 save_file:str=None, p0=None, scl_sig:float=0.05):

    # Device for NN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init
    if hasattr(nn_model, 'ninput'):
        ndim = nn_model.ninput
    else:
        # TODO -- remove this
        ndim = 8
    if p0 is None:
        p0 = np.random.rand(nwalkers, ndim)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    if save_file is not None:
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, 
                                    args=[Rs, nn_model, device, scl_sig],
                                    backend=backend)

    # Burn in
    print("Running burn-in")
    state = sampler.run_mcmc(p0, 1000)
    sampler.reset()

    # Run
    print("Running full model")
    sampler.run_mcmc(state, nsteps)

    print(f"All done: Wrote {save_file}")

    # Return
    return sampler

if __name__ == '__main__':

    from ihop.iops.nmf import load_loisel_2023
    # Load Hydrolight
    print("Loading Hydrolight data")
    #ab, Rs, d_l23 = ihop_io.load_loisel_2023_pca()
    ab, Rs, _, _ = load_loisel_2023()

    # Load model
    em_path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators')
    model_file = os.path.join(em_path, 'densenet_NMF3_L23', 
                       'densenet_NMF_[512, 128, 128]_batchnorm_epochs_2500_p_0.05_lr_0.001.pth')
    print(f"Loading model: {model_file}")
    model = io.load_nn(model_file)
    

    # idx=200
    idx = 200
    save_file = f'MCMC_NN_NMF_i{idx}.h5'

    run_emcee_nn(model, Rs[idx], save_file=save_file)