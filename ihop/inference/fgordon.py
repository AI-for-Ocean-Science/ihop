""" Modules to perform inference using the Gordon method. """

import numpy as np

import emcee

from scipy.interpolate import interp1d

from oceancolor.hydrolight import loisel23
from oceancolor.ph import absorption as ph_absorption

from ihop.iops import io as iops_io

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
ds_wave = ds.Lambda.data[::2]

# ##################################
# Bricaud
b1998 = ph_absorption.load_bricaud1998()

# Interpolate
f_b1998_A = interp1d(b1998['lambda'], b1998.Aphi, bounds_error=False, fill_value=0.)
f_b1998_E = interp1d(b1998['lambda'], b1998.Ephi, bounds_error=False, fill_value=0.)

# Apply
L23_A = f_b1998_A(ds_wave)
L23_E = f_b1998_E(ds_wave)

# Normalize at 440
iwave = np.argmin(np.abs(ds_wave-440))
L23_A /= L23_A[iwave]

# ##################################
# NMF for aph
# Load the decomposition of aph
aph_file = iops_io.loisel23_filename('nmf', 'aph', 2, 4, 0)
d_aph = np.load(aph_file)
NMF_W1=d_aph['M'][0][::2]
NMF_W2=d_aph['M'][1][::2]

# ##################################
# NMF for bb
bb_file = iops_io.loisel23_filename('nmf', 'bb', 2, 4, 0)
d_bb = np.load(bb_file)
NMF_bbW1=d_bb['M'][0][::2]
NMF_bbW2=d_bb['M'][1][::2]

def calc_ab(model:str, params:np.ndarray, pdict:dict):
    """
    Calculate the a and b values from the input parameters.

    Args:
        model (str): The model name.
        params (np.ndarray): The input parameters.
            1D or 2D

    Returns:
        tuple: A tuple containing the a and b values.
    """
    if model == 'Indiv':
        #a = params[:41]/1000
        #bb = params[41:]/1000
        a = 10**params[...,:41]
        bb = 10**params[...,41:]
    elif model == 'bbwater':
        a = 10**params[...,:41]
        bb = 10**params[...,41:] + bbw
    elif model == 'water':
        a = 10**params[...,:41] + aw
        bb = 10**params[...,41:] + bbw
    elif model == 'bp':
        a = 10**params[...,:41] + aw
        # Lee+2002
        bbp = np.outer(10**params[...,-1],
                       (550./pdict['wave'])**pdict['Y'])
        # Add water
        bb = bbp + bbw
    elif model == 'explee':
        # anw exponential
        anw = np.outer(10**params[...,0], np.ones_like(pdict['wave'])) *\
            np.exp(np.outer(-10**params[...,1],pdict['wave']-400.))
        a = anw + aw
                       
        # Lee+2002 bbp
        bbp = np.outer(10**params[...,-1],
                       (550./pdict['wave'])**pdict['Y'])
        # Add water
        bb = bbp + bbw
    elif model == 'cstcst':
        # Constant anw
        anw = np.outer(10**params[...,0], np.ones_like(pdict['wave']))
        a = anw + aw
                       
        # Cosntant bpp
        bbp = np.outer(10**params[...,1], np.ones_like(pdict['wave']))

        # Add water
        bb = bbp + bbw
    elif model == 'expcst':
        # anw exponential
        anw = np.outer(10**params[...,0], np.ones_like(pdict['wave'])) *\
            np.exp(np.outer(-10**params[...,1],pdict['wave']-400.))
        a = anw + aw
                       
        # Cosntant bpp
        bbp = np.outer(10**params[...,2], np.ones_like(pdict['wave']))

        # Add water
        bb = bbp + bbw
    elif model == 'exppow':
        # anw exponential
        anw = np.outer(10**params[...,0], np.ones_like(pdict['wave'])) *\
            np.exp(np.outer(-10**params[...,1],pdict['wave']-400.))
        a = anw + aw
                       
        # Power-law with free exponent
        bbp = np.outer(10**params[...,2], np.ones_like(pdict['wave'])) *\
                       (550./pdict['wave'])**(10**params[...,3]).reshape(-1,1)
        # Add water
        bb = bbp + bbw
    elif model == 'giop':
        # adg exponential
        adg = np.outer(10**params[...,0], np.ones_like(pdict['wave'])) *\
            np.exp(np.outer(-10**params[...,1],pdict['wave']-400.))
        # aph Briaud
        aph = np.outer(10**params[...,2], L23_A*pdict['Chl']**(L23_E))

        a = adg + aph + aw
                       
        # Lee+2002
        bbp = np.outer(10**params[...,-1],
                       (550./pdict['wave'])**pdict['Y'])
        # Add water
        bb = bbp + bbw
    elif model == 'giop+':  # adg, aph Bricaud, free bbp
        # adg exponential
        adg = np.outer(10**params[...,0], np.ones_like(pdict['wave'])) *\
            np.exp(np.outer(-10**params[...,1],pdict['wave']-400.))
        # aph Briaud
        aph = np.outer(10**params[...,2], L23_A*pdict['Chl']**(L23_E))

        a = adg + aph + aw
                       
        # Power-law with free exponent
        bbp = np.outer(10**params[...,3], np.ones_like(pdict['wave'])) *\
                       (550./pdict['wave'])**(10**params[...,4]).reshape(-1,1)
        # Add water
        bb = bbp + bbw
    elif model == 'hybpow':
        # adg exponent
        adg = np.outer(10**params[...,0], np.ones_like(pdict['wave'])) *\
            np.exp(np.outer(-10**params[...,1],pdict['wave']-400.))
        aph = np.outer(10**params[...,2], NMF_W1) + np.outer(10**params[...,3], NMF_W2)

        a = adg + aph + aw
                       
        # Power-law with free exponent
        bbp = np.outer(10**params[...,4], np.ones_like(pdict['wave'])) *\
                       (550./pdict['wave'])**(10**params[...,5]).reshape(-1,1)
        # Add water
        bb = bbp + bbw
    elif model == 'hybnmf':
        # adg exponent
        adg = np.outer(10**params[...,0], np.ones_like(pdict['wave'])) *\
            np.exp(np.outer(-10**params[...,1],pdict['wave']-400.))
        aph = np.outer(10**params[...,2], NMF_W1) +\
            np.outer(10**params[...,3], NMF_W2)

        a = adg + aph + aw
                       
        # NMF bb
        bnw = np.outer(10**params[...,4], NMF_bbW1) +\
            np.outer(10**params[...,5], NMF_bbW2)

        # Add water
        bb = bnw + bbw
    else:
        raise ValueError(f"Bad model: {model}")
    # Return
    return a, bb

def calc_Rrs(a, bb, in_G1=None, in_G2=None):
    """
    Calculates the Remote Sensing Reflectance (Rrs) using the given absorption (a) and backscattering (bb) coefficients.

    Parameters:
        a (float or array-like): Absorption coefficient.
        bb (float or array-like): Backscattering coefficient.

    Returns:
        float or array-like: Remote Sensing Reflectance (Rrs) value.
    """
    # u
    u = bb / (a+bb)
    # rrs
    if in_G1 is not None:
        t1 = in_G1 * u
    else: 
        t1 = G1 * u
    if in_G2 is not None:
        t2 = in_G2 * u*u
    else:
        t2 = G2 * u*u
    rrs = t1 + t2
    # Done
    Rrs = A_Rrs*rrs / (1 - B_Rrs*rrs)
    return Rrs
    
def init_mcmc(model:str, ndim:int, wave:np.ndarray,
              nsteps:int=10000, nburn:int=1000, Y:float=None,
              Chl:float=None):
    """
    Initializes the MCMC parameters.

    Args:
        emulator: The emulator model.
        ndim (int): The number of dimensions.
        nsteps (int, optional): The number of steps to run the sampler. Defaults to 10000.
        nburn (int, optional): The number of steps to run the burn-in. Defaults to 1000.
        wave (np.ndarray): The wavelengths.
        Y (float, optional): The Y parameter for the bp model. Defaults to None.
        Chl (float, optional): The chlorophyll value. Defaults to None.

    Returns:
        dict: A dictionary containing the MCMC parameters.
    """
    pdict = dict(model=model)
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['nsteps'] = nsteps
    pdict['nburn'] = nburn
    pdict['wave'] = wave
    pdict['Y'] = Y # for bp (Lee+2002)
    pdict['save_file'] = None
    pdict['Chl'] = Chl # for aph (Brichaud+1995)
    #
    return pdict

def grab_priors(model:str):
    # Set em
    if model in ['Indiv', 'bbwater', 'water']:
        ndim = 82
    elif model in ['bp']:
        ndim = 42
    elif model == 'cstcst':
        ndim = 2
    elif model == 'explee':
        ndim = 3
    elif model == 'expcst':
        ndim = 3
    elif model == 'exppow':
        ndim = 4
    elif model == 'giop':
        ndim = 4
    elif model == 'giop+':
        ndim = 5
    elif model == 'hybpow':
        ndim = 6
    elif model == 'hybnmf':
        ndim = 6
    else:
        raise ValueError(f"Bad model: {model}")
    # Return
    priors = np.zeros((ndim, 2))
    priors[:,0] = -6
    priors[:,1] = 5
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
        save_file=pdict['save_file'],
        pdict=pdict)

    # Return
    if chains_only:
        return sampler.get_chain().astype(np.float32), idx
    else:
        return sampler, idx



def log_prob(params, model:str, Rs:np.ndarray, varRs, pdict:dict):
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
    a, bb = calc_ab(model, params, pdict)
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
              p0=None, skip_check:bool=False, ndim:int=None,
              pdict:dict=None):
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
        args=[model, Rrs, varRrs, pdict],
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