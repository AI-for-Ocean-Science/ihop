""" Modules to perform inference using the Gordon method. """

import numpy as np

import emcee
from emcee import ensemble

from oceancolor.hydrolight import loisel23

from IPython import embed

# Conversion from rrs to Rrs
A_Rrs, B_Rrs = 0.52, 1.7

# Gordon factors
G1, G2 = 0.0949, 0.0794  # Gordon

# Hack for now
ds = loisel23.load_ds(4,0)
bbw = ds.bb.data[0,:]-ds.bbnw.data[0,:]
bbw = bbw[::2]
aw = ds.a.data[0,:]-ds.anw.data[0,:]
aw = aw[::2]


def calc_ab(model:str, params:np.ndarray):
    """
    Calculate the a and b values from the input parameters.

    Args:
        model (str): The model name.
        params (np.ndarray): The input parameters.

    Returns:
        tuple: A tuple containing the a and b values.
    """
    if model == 'Indiv':
        #a = params[:41]/1000
        #bb = params[41:]/1000
        a = 10**params[:41]
        bb = 10**params[41:]
    elif model == 'bbwater':
        a = 10**params[:41]
        bb = 10**params[41:] + bbw
    else:
        raise ValueError(f"Bad model: {model}")
    # Return
    return a, bb

def grab_priors(model:str):
    if model in ['Indiv', 'bbwater']:
        ndim = 82
        priors = np.zeros((ndim, 2))
        priors[:,0] = -6
        priors[:,1] = 5
        #priors[:,0] = 0.
        #priors[:,1] = np.inf
    else:
        raise ValueError(f"Bad model: {model}")
    # Return
    return priors

def fit_one(items:list, pdict:dict=None, chains_only:bool=False):
    """
    Fits a model to a set of input data using the MCMC algorithm.

    Args:
        items (list): A list containing the input data, errors, response data, and index.
        pdict (dict, optional): A dictionary containing the model and fitting parameters. Defaults to None.
        chains_only (bool, optional): If True, only the chains are returned. Defaults to False.

    Returns:
        tuple: A tuple containing the MCMC sampler object and the index.
    """
    # Unpack
    Rs, varRs, params, idx = items

    # Run
    print(f"idx={idx}")
    sampler = run_emcee(
        pdict['model'], Rs, varRs,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        nburn=pdict['nburn'],
        skip_check=True,
        p0=params,
        save_file=pdict['save_file'])

    # Return
    if chains_only:
        return sampler.get_chain().astype(np.float32), idx
    else:
        return sampler, idx

def init_mcmc(model:str, ndim:int, wave:np.ndarray,
              nsteps:int=10000, nburn:int=1000):
    """
    Initializes the MCMC parameters.

    Args:
        emulator: The emulator model.
        ndim (int): The number of dimensions.
        priors (dict): The prior information (optional).

    Returns:
        dict: A dictionary containing the MCMC parameters.
    """
    pdict = dict(model=model)
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['nsteps'] = nsteps
    pdict['nburn'] = nburn
    pdict['wave'] = wave
    pdict['save_file'] = None
    #
    return pdict

def log_prob(params, model:str, Rs, varRs):
    """
    Calculate the logarithm of the probability of the given parameters.

    Args:
        params (array-like): The parameters to be used in the model prediction.
        model (str): The model name
        Rs (array-like): The observed values.

    Returns:
        float: The logarithm of the probability.
    """
    # Priors
    priors = grab_priors(model)
    if np.any(params < priors[:,0]) or np.any(params > priors[:,1]):
        return -np.inf


    # Proceed
    a, bb = calc_ab(model, params)
    u = bb / (a+bb)
    rrs = G1 * u + G2 * u*u
    pred = A_Rrs*rrs / (1 - B_Rrs*rrs)

    # Evaluate
    eeval = (pred-Rs)**2 / varRs
    # Finish
    prob = -1*0.5 * np.sum(eeval)
    if np.isnan(prob):
        return -np.inf
    else:
        return prob


def run_emcee(model:str, Rrs, varRrs, nwalkers:int=32, 
              nburn:int=1000,
              nsteps:int=20000, save_file:str=None, 
              p0=None, skip_check:bool=False, ndim:int=None):
    """
    Run the emcee sampler for neural network inference.

    Args:
        nn_model (torch.nn.Module): The neural network model.
        Rrs (numpy.ndarray): The input data.
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

    # Initialize
    if p0 is None:
        priors = grab_priors(model)
        ndim = priors.shape[0]
        p0 = np.random.uniform(priors[:,0], priors[:,1], size=(nwalkers, ndim))
    else:
        # Replicate for nwalkers
        ndim = len(p0)
        p0 = np.tile(p0, (nwalkers, 1))
        # Perturb 
        p0 += p0*np.random.uniform(-1e-2, 1e-2, size=p0.shape)
        #r = 10**np.random.uniform(-0.5, 0.5, size=p0.shape[0])
        #for ii in range(p0.shape[0]):
        #    p0[ii] *= r[ii]
        #embed(header='108 of fgordon')

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
        args=[model, Rrs, varRrs],
        backend=backend)#, pool=pool)

    # Burn in
    print("Running burn-in")
    state = sampler.run_mcmc(p0, nburn,
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