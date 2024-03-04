""" Methods for fitting """

import numpy as np

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ihop.inference import mcmc

from IPython import embed


def fit_one(items:list, pdict:dict=None, chains_only:bool=False):
    """
    Fits a model to a set of input data using the MCMC algorithm.

    Args:
        items (list): A list containing the input data, response data, and index.
        pdict (dict, optional): A dictionary containing the model and fitting parameters. Defaults to None.
        chains_only (bool, optional): If True, only the chains are returned. Defaults to False.

    Returns:
        tuple: A tuple containing the MCMC sampler object and the index.
    """
    # Unpack
    Rs, inputs, idx = items
    ndim = pdict['model'].ninput

    # Run
    #embed(header='fit_one 208')
    sampler = mcmc.run_emcee_nn(
        pdict['model'], Rs,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        scl_sig=pdict['scl_sig']/100. if pdict['scl_sig'] is not None else None,
        abs_sig=pdict['abs_sig'] if pdict['abs_sig'] is not None else None,
        p0=inputs,
        save_file=pdict['save_file'])

    # Return
    if chains_only:
        return sampler.get_chain().astype(np.float32), idx
    else:
        return sampler, idx

def fit_batch(pdict:dict, items:list, n_cores:int=1): 
    """
    Fits a batch of items using parallel processing.

    Args:
        pdict (dict): A dictionary containing the parameters for fitting.
        items (list): A list of items to be fitted.
        n_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.

    Returns:
        tuple: A tuple containing the fitted samples and their corresponding indices.
    """

    # Setup for parallel
    map_fn = partial(fit_one, pdict=pdict, chains_only=True)
    
    # Parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    # Need to avoid blowing up the memory!
    # Slurp
    all_idx = np.array([item[1] for item in answers])
    #answers = [item[0].get_chain() for item in answers]
    answers = np.array([item[0].astype(np.float32) for item in answers])

    # Chains
    #all_samples = np.zeros((len(samples), samples[0].shape[0], 
    #    samples[0].shape[1], samples[0].shape[2]))
    #for ss in range(len(all_idx)):
    #    all_samples[ss,:,:,:] = samples[ss]

    return answers[all_idx], all_idx