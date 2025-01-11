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
    print(f"idx={idx}")
    sampler = mcmc.run_emcee_nn(
        pdict['model'], Rs,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        scl_sig=pdict['scl_sig']/100. if pdict['scl_sig'] is not None else None,
        abs_sig=pdict['abs_sig'] if pdict['abs_sig'] is not None else None,
        cut=pdict['cut'] if pdict['cut'] is not None else None,
        priors=pdict['priors'],
        p0=inputs,
        save_file=pdict['save_file'])

    # Return
    if chains_only:
        return sampler.get_chain().astype(np.float32), idx
    else:
        return sampler, idx

def fit_batch(pdict:dict, items:list, n_cores:int=1, fit_method=None): 
    """
    Fits a batch of items using parallel processing.

    Args:
        pdict (dict): A dictionary containing the parameters for fitting.
        items (list): A list of items to be fitted.
        n_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        fit_method (function, optional): The fitting method to be used. Defaults to None.

    Returns:
        tuple: A tuple containing the fitted samples and their corresponding indices.
    """
    if fit_method is None:
        fit_method = fit_one

    # Setup for parallel
    map_fn = partial(fit_method, pdict=pdict, chains_only=True)
    
    # Parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    # Need to avoid blowing up the memory!
    # Slurp
    all_idx = np.array([item[1] for item in answers])
    answers = np.array([item[0].astype(np.float32) for item in answers])

    return answers, all_idx