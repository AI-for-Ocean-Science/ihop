""" Perform Gordon analyses """
import os

import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.inference import fgordon
from ihop.inference import fitting

from IPython import embed

def prep_data(idx:int, scl_noise:float=0.01):
    """ Prepare the data for the Gordon analysis """

    # Load
    ds = loisel23.load_ds(4,0)

    # Grab
    Rrs = ds.Rrs.data[idx,:]
    Rrs_true = Rrs.copy()
    wave = ds.Lambda.data
    true_wave = ds.Lambda.data.copy()
    a = ds.a.data[idx,:]
    bb = ds.bb.data[idx,:]

    # Cut down to 40 bands
    Rrs = Rrs[::2]
    wave = wave[::2]

    # Error
    varRrs = (scl_noise * Rrs)**2

    # Dict me
    odict = dict(wave=wave, Rrs=Rrs, varRrs=varRrs, a=a, bb=bb, 
                 true_wave=true_wave, Rrs_true=Rrs_true,
                 bbw=ds.bb.data[idx,:]-ds.bbnw.data[idx,:],
                 aw=ds.a.data[idx,:]-ds.anw.data[idx,:])

    return odict

def fit_model(model:str, n_cores=20, idx:int=170):

    odict = prep_data(idx)
    # Unpack
    wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a = odict['a']
    bb = odict['bb']
    bbw = odict['bbw']
    
    # Grab the priors (as a test and for ndim)
    priors = fgordon.grab_priors(model)
    ndim = priors.shape[0]
    # Initialize the MCMC
    pdict = fgordon.init_mcmc(model, ndim, wave)
    
    # Hack for now
    if model == 'Indiv':
        p0_a = a[::2]
        p0_b = bb[::2]
    elif model == 'bbwater':
        p0_a = a[::2]
        p0_b = np.maximum(bb[::2] - bbw[::2], 1e-4)
    elif model == 'water':
        p0_a = a[::2] - aw[::2]
        p0_b = bb[::2] - bbw[::2]
    else:
        raise ValueError(f"51 of gordon.py -- Deal with this model: {model}")

    p0 = np.concatenate((np.log10(p0_a), np.log10(p0_b)))

    # Set the items
    items = [(Rrs, varRrs, p0, idx)]

    # Test
    chains, idx = fgordon.fit_one(items[0], pdict=pdict, chains_only=True)
    
    # Save
    outfile = f'FGordon_{model}_170'
    save_fits(chains, idx, outfile)

def reconstruct(model:str, chains, burn=7000, thin=1):
    chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    # Burn the chains
    if model in ['Indiv', 'bbwater', 'water']:
        a = 10**chains[:, :41]
        bb = 10**chains[:, 41:]
    else:
        raise ValueError(f"Bad model: {model}")

    # Calculate the mean and standard deviation
    a_mean = np.mean(a, axis=0)
    a_std = np.std(a, axis=0)
    bb_mean = np.mean(bb, axis=0)
    bb_std = np.std(bb, axis=0)

    # Calculate the model Rrs
    u = bb/(a+bb)
    rrs = fgordon.G1 * u + fgordon.G2 * u*u
    Rrs = fgordon.A_Rrs*rrs / (1 - fgordon.B_Rrs*rrs)

    # Stats
    sigRs = np.std(Rrs, axis=0)
    Rrs = np.mean(Rrs, axis=0)

    # Return
    return a_mean, bb_mean, a_std, bb_std, Rrs, sigRs 

def save_fits(all_samples, all_idx, outroot, extras:dict=None):
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
    outdir = 'Fits/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, outroot)
    # Outdict
    outdict = dict()
    outdict['chains'] = all_samples
    outdict['idx'] = all_idx
    
    # Extras
    if extras is not None:
        for key in extras.keys():
            outdict[key] = extras[key]
    np.savez(outfile, **outdict)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Testing
    if flg & (2**0):
        wv, Rrs, varRrs = prep_data(170)

    # Indiv
    if flg & (2**1):
        fit_model('Indiv')

    # bb_water
    if flg & (2**2):
        fit_model('bbwater')

    # water
    if flg & (2**3):
        fit_model('water')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Testing
        #flg += 2 ** 1  # 2 -- No priors
        #flg += 2 ** 2  # 4 -- bb_water

    else:
        flg = sys.argv[1]

    main(flg)