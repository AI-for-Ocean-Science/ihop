""" Generic fitting routines for IHOP """
import os
import numpy as np

from ihop.inference import mcmc
from ihop.emulators import io as emu_io
from ihop.inference import noise
from ihop.inference import fitting 

from IPython import embed


def set_priors(edict, use_log_ab=False, use_NMF_pos=False):
    # Priors
    priors = None
    if use_log_ab:
        priors = {}
        priors['use_log_ab'] = True
    elif use_NMF_pos:
        priors = {}
        priors['NMFpos'] = True
    return priors

def fit(edict:dict, params:np.ndarray, wave:np.ndarray,
        priors:dict, Rrs:np.ndarray, outfile:str, abs_sig:float,
        Nspec:int=None, debug:bool=False, n_cores:int=1,
        max_wv:float=None, extras:dict=None):
    """
    Fits the data with or without considering any errors.

    Args:
        edict (dict): A dictionary containing the necessary information for fitting.
        Nspec (int): The number of spectra to fit. Default is None = all
        abs_sig (float): The absolute value of the error to consider. Default is None.
            if None, use no error!
        debug (bool): Whether to run in debug mode. Default is False.
        n_cores (int): The number of CPU cores to use for parallel processing. Default is 1.
        max_wv (float): The maximum wavelength to consider. Default is None.
        use_log_ab (bool): Whether to use log(ab) in the priors. Default is False.
        use_NMF_pos (bool): Whether to use positive priors for NMF. Default is False.

    """
    emulator, _ = emu_io.load_emulator_from_dict(edict)

    # Init MCMC
    if abs_sig in ['PACE', 'PACE_CORR', 'PACE_TRUNC']:
        pace_sig = noise.calc_pace_sig(wave)
        pdict = mcmc.init_mcmc(emulator, params.shape[1], 
                      abs_sig=pace_sig, priors=priors)
    else:
        pdict = mcmc.init_mcmc(emulator, params.shape[1]+1, 
                      abs_sig=abs_sig, priors=priors)

    # Include a non-zero error to avoid bad chi^2 behavior
    if abs_sig is None:
        pdict['abs_sig'] = 1.

    # Max wave?
    if max_wv is not None:
        cut = wave < max_wv
        pdict['cut'] = cut

    # No noise
    if abs_sig is None or abs_sig=='PACE_TRUNC':
        use_Rrs = Rrs.copy()
    else:
        correlate = abs_sig=='PACE_CORR'
        use_Rrs = noise.add_noise(
            Rrs, abs_sig=abs_sig, wave=wave,
            correlate=correlate)

    # Prep
    if Nspec is None:
        idx = np.arange(Rrs.shape[0])
    else:
        idx = np.arange(Nspec)
    if debug:
        #idx = idx[0:2]
        idx = [170, 180]
    if priors is not None and 'use_log_ab' in priors and priors['use_log_ab']:
        items = [(use_Rrs[i], np.log10(params[i,:]).tolist(), i) for i in idx]
    else:
        items = [(use_Rrs[i], params[i,:].tolist(), i) for i in idx]

    #if debug:
    #    embed(header='fit 88')

    # Fit
    all_samples, all_idx = fitting.fit_batch(pdict, items,
                                             n_cores=n_cores)
    # Save
    save_fits(all_samples, all_idx, Rrs, use_Rrs, outfile,
              extras=extras)


def save_fits(all_samples, all_idx, Rs, use_Rs, outroot, extras:dict=None):
    """
    Save the fitting results to a file.

    Parameters:
        all_samples (numpy.ndarray): Array of fitting chains.
        all_idx (numpy.ndarray): Array of indices.
        Rs (numpy.ndarray): Array of Rs values.
        use_Rs (numpy.ndarray): Array of observed Rs values.
        outroot (str): Root name for the output file.
    """  
    # Save
    outdir = 'Fits/L23'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, outroot)
    # Outdict
    outdict = dict()
    outdict['chains'] = all_samples
    outdict['idx'] = all_idx
    outdict['obs_Rs'] = use_Rs[all_idx]
    outdict['Rs'] = Rs[all_idx]
    
    # Extras
    if extras is not None:
        for key in extras.keys():
            outdict[key] = extras[key]
    np.savez(outfile, **outdict)
    print(f"Saved: {outfile}")