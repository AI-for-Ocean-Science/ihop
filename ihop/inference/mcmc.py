""" MCMC module for IHOP """

import os
import warnings

from multiprocessing import Pool
import numpy as np
import emcee

import torch
from ihop.emulators import io

from IPython import embed


#def lnprior(ab):
#    m, b, lnf = theta
#    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
#        return 0.0
#    return -np.inf

def log_prob(ab, Rs, model, device, scl_sig, abs_sig, priors,
             cut):
    """
    Calculate the logarithm of the probability of the given parameters.

    Args:
        ab (array-like): The parameters to be used in the model prediction.
        Rs (array-like): The observed values.
        model (object): The model object with a `prediction` method.
        device (str): The device to be used for the model prediction.
        scl_sig (float or None): The scaling factor for the error. If None, absolute error is used.
        abs_sig (float or np.ndarray): The absolute error.
        prior (tuple): The prior information.
        cut (array-like): Limit the likelihood calculation
            to a subset of the values

    Returns:
        float: The logarithm of the probability.
    """
    # Priors
    use_ab = ab
    if priors is not None:
        # Check for NMF positivity
        if 'NMFpos' in priors.keys() and priors['NMFpos']:
            if np.min(ab) < 0:
                return -np.inf
        elif 'use_log_ab' in priors.keys() and priors['use_log_ab']:
            use_ab = 10**ab
            if (np.min(ab) < -4) or (np.max(ab) > 2):
                return -np.inf
        #lp = lnprior(ab, priors)

    # Proceed
    pred = model.prediction(use_ab, device)
    # Error
    sig = scl_sig * Rs if scl_sig is not None else abs_sig
    #
    eeval = (pred-Rs)**2 / sig**2
    # Cut?
    if cut is not None:
        eeval = eeval[cut]
    # Finish
    prob = -1*0.5 * np.sum(eeval)
    if np.isnan(prob):
        return -np.inf
    else:
        return prob


def run_emcee_nn(nn_model, Rs, nwalkers:int=32, nsteps:int=20000,
                 save_file:str=None, p0=None, scl_sig:float=None,
                 abs_sig:float=None, 
                 priors:tuple=None,
                 cut:np.ndarray=None,
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
        prior (dict, optional): The prior information. Defaults to None.
        cut (np.ndarray, optional): Limit the likelihood calculation
            to a subset of the values

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
        p0 += p0*np.random.uniform(-1e-2, 1e-2, size=p0.shape)

    #embed(header='run_emcee_nn 47')
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    if save_file is not None:
        backend = emcee.backends.HDFBackend(save_file)
        backend.reset(nwalkers, ndim)
    else:
        backend = None

    # Init
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=[Rs, nn_model, device, scl_sig, abs_sig, priors, cut],
        backend=backend)#, pool=pool)

    # Burn in
    print("Running burn-in")
    state = sampler.run_mcmc(p0, 1000,
        skip_initial_state_check=skip_check,
        progress=True)
    sampler.reset()

    # Run
    print("Running full model")
    sampler.run_mcmc(state, nsteps,
        skip_initial_state_check=skip_check,
        progress=True)

    if save_file is not None:
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