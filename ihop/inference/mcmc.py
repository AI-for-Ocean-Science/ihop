""" MCMC module for IHOP """

import os
import warnings

import numpy as np
import emcee

import torch
from ihop.emulators import io

from IPython import embed


def log_prob(ab, Rs, model, device, scl_sig, abs_sig):
    """
    Calculate the logarithm of the probability of the given parameters.

    Args:
        ab (array-like): The parameters to be used in the model prediction.
        Rs (array-like): The observed values.
        model (object): The model object with a `prediction` method.
        device (str): The device to be used for the model prediction.
        scl_sig (float or None): The scaling factor for the error. If None, absolute error is used.
        abs_sig (float): The absolute error.

    Returns:
        float: The logarithm of the probability.
    """
    pred = model.prediction(ab, device)
    # Error
    sig = scl_sig * Rs if scl_sig is not None else abs_sig
    #
    prob = -1*0.5 * np.sum( (pred-Rs)**2 / sig**2)
    if np.isnan(prob):
        return -np.inf
    else:
        return prob


def run_emcee_nn(nn_model, Rs, nwalkers:int=32, nsteps:int=20000,
                 save_file:str=None, p0=None, scl_sig:float=None,
                 abs_sig:float=None,
                 skip_check:bool=False):
    """
    Run the emcee sampler for neural network inference.

    Args:
        nn_model (torch.nn.Module): The neural network model.
        Rs (numpy.ndarray): The input data.
        nwalkers (int, optional): The number of walkers in the ensemble. Defaults to 32.
        nsteps (int, optional): The number of steps to run the sampler. Defaults to 20000.
        save_file (str, optional): The file path to save the backend. Defaults to None.
        p0 (numpy.ndarray, optional): The initial positions of the walkers. Defaults to None.
        scl_sig (float, optional): The scaling factor for the sigma parameter. Defaults to None.
        abs_sig (float, optional): The absolute value of the sigma parameter. Defaults to None.
        skip_check (bool, optional): Whether to skip the initial state check. Defaults to False.

    Returns:
        emcee.EnsembleSampler: The emcee sampler object.
    """
    # Device for NN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init
    if hasattr(nn_model, 'ninput'):
        ndim = nn_model.ninput
    else:
        # TODO -- remove this
        warnings.warn("Assuming 8 inputs. REMOVE THIS")
        ndim = 8


    # Initialize
    if p0 is None:
        p0 = np.random.rand(nwalkers, ndim)
    else:
        if len(p0) != ndim:
            raise ValueError("Bad p0")
        # Replicate for nwalkers
        p0 = np.tile(p0, (nwalkers, 1))
        # Perturb a tiny bit
        p0 += p0*np.random.uniform(-1e-4, 1e-4, size=p0.shape)

    #embed(header='run_emcee_nn 47')
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    if save_file is not None:
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, 
        args=[Rs, nn_model, device, scl_sig, abs_sig],
        backend=backend)

    # Burn in
    print("Running burn-in")
    state = sampler.run_mcmc(p0, 1000,
        skip_initial_state_check=skip_check)
    sampler.reset()

    # Run
    print("Running full model")
    sampler.run_mcmc(state, nsteps,
        skip_initial_state_check=skip_check)

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