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
    wave = ds.Lambda.data

    # Cut down to 40 bands
    Rrs = Rrs[::2]
    wave = wave[::2]

    # Error
    varRrs = (scl_noise * Rrs)**2

    return wave, Rrs, varRrs

def fit_model(model:str, n_cores=20):

    wave, Rrs, varRrs = prep_data(170)
    # Grab the priors (as a test and for ndim)
    priors = fgordon.grab_priors(model)
    ndim = priors.shape[0]
    # Initialize the MCMC
    pdict = fgordon.init_mcmc(model, ndim, wave)
    
    # Set the items
    items = [(Rrs, varRrs, None, 170)]

    # Test
    chains, idx = fgordon.fit_one(items[0], pdict=pdict, chains_only=True)
    
    # Save
    outfile = f'FGordon_{model}_170'
    save_fits(chains, idx, outfile)

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

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Testing
        #flg += 2 ** 1  # 2 -- Figure 2 L23: PCA vs NMF Explained variance

    else:
        flg = sys.argv[1]

    main(flg)