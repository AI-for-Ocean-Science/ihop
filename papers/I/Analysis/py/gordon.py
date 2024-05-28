""" Perform Gordon analyses """
import os

import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.inference import fgordon
from ihop.inference import fitting

from IPython import embed

def prep_data(idx:int, scl_noise:float=0.02):
    """ Prepare the data for the Gordon analysis """

    # Load
    ds = loisel23.load_ds(4,0)

    # Grab
    Rrs = ds.Rrs.data[idx,:]
    true_Rrs = Rrs.copy()
    wave = ds.Lambda.data
    true_wave = ds.Lambda.data.copy()
    a = ds.a.data[idx,:]
    bb = ds.bb.data[idx,:]

    # For bp
    rrs = Rrs / (fgordon.A_Rrs + fgordon.B_Rrs*Rrs)
    i440 = np.argmin(np.abs(true_wave-440))
    i555 = np.argmin(np.abs(true_wave-555))
    Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * rrs[i440]/rrs[i555]))

    # For aph
    aph = ds.aph.data[idx,:]
    Chl = aph[i440] / 0.05582

    # Cut down to 40 bands
    Rrs = Rrs[::2]
    wave = wave[::2]

    # Error
    varRrs = (scl_noise * Rrs)**2

    # Dict me
    odict = dict(wave=wave, Rrs=Rrs, varRrs=varRrs, a=a, bb=bb, 
                 true_wave=true_wave, true_Rrs=true_Rrs,
                 bbw=ds.bb.data[idx,:]-ds.bbnw.data[idx,:],
                 aw=ds.a.data[idx,:]-ds.anw.data[idx,:],
                 Y=Y, Chl=Chl)

    return odict

def fit_model(model:str, n_cores=20, idx:int=170, 
              nsteps:int=10000, nburn:int=1000):

    odict = prep_data(idx)
    # Unpack
    wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a = odict['a']
    bb = odict['bb']
    bbw = odict['bbw']
    aw = odict['aw']
    
    # Grab the priors (as a test and for ndim)
    priors = fgordon.grab_priors(model)
    ndim = priors.shape[0]
    # Initialize the MCMC
    pdict = fgordon.init_mcmc(model, ndim, wave, Y=odict['Y'], Chl=odict['Chl'],
                              nsteps=nsteps, nburn=nburn)
    
    # Hack for now
    if model == 'Indiv':
        p0_a = a[::2]
        p0_b = bb[::2]
    elif model == 'bbwater':
        scl = 5.
        p0_a = scl*a[::2]
        p0_b = 2*np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
    elif model == 'water':
        scl = 5.
        p0_a = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_b = 2*np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
    elif model == 'bp':
        scl = 5.
        p0_a = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        # bbp
        bnw = 2*np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        i500 = np.argmin(np.abs(wave-500))
        p0_b = bnw[i500] 
    elif model == 'exppow':
        i400 = np.argmin(np.abs(wave-400))
        i500 = np.argmin(np.abs(wave-500))
        scl = 1.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i400], 0.017] 
        # bbp
        bnw = 2*np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = bnw[i500] 
    elif model == 'giop':
        i440 = np.argmin(np.abs(wave-440))
        i500 = np.argmin(np.abs(wave-500))
        scl = 1.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i440]/2., 0.017, anw[i440]/2.] 
        # bbp
        bnw = 2*np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = bnw[i500] 
    else:
        raise ValueError(f"51 of gordon.py -- Deal with this model: {model}")

    p0 = np.concatenate((np.log10(p0_a), 
                         np.log10(np.atleast_1d(p0_b))))
    #p0 = np.concatenate([p0_a, p0_b])

    # Chk initial guess
    ca,cbb = fgordon.calc_ab(model, p0, pdict)
    pRrs = fgordon.calc_Rrs(ca, cbb)
    print(f'Initial Rrs guess: {np.mean((Rrs-pRrs)/Rrs)}')
    embed(header='100 of gordon')

    # Set the items
    #items = [(Rrs, varRrs, None, idx)]
    items = [(Rrs, varRrs, p0, idx)]

    # Test
    chains, idx = fgordon.fit_one(items[0], pdict=pdict, chains_only=True)
    
    # Save
    outfile = f'FGordon_{model}_170'
    save_fits(chains, idx, outfile)

def reconstruct(model:str, chains, pdict:dict, burn=7000, thin=1):
    # Burn the chains
    chains = chains[burn::thin, :, :].reshape(-1, chains.shape[-1])
    # Calc
    a, bb = fgordon.calc_ab(model, chains, pdict)
    del chains

    # Calculate the mean and standard deviation
    a_mean = np.median(a, axis=0)
    a_5, a_95 = np.percentile(a, [5, 95], axis=0)
    #a_std = np.std(a, axis=0)
    bb_mean = np.median(bb, axis=0)
    bb_5, bb_95 = np.percentile(bb, [5, 95], axis=0)
    #bb_std = np.std(bb, axis=0)

    # Calculate the model Rrs
    u = bb/(a+bb)
    rrs = fgordon.G1 * u + fgordon.G2 * u*u
    Rrs = fgordon.A_Rrs*rrs / (1 - fgordon.B_Rrs*rrs)

    # Stats
    sigRs = np.std(Rrs, axis=0)
    Rrs = np.median(Rrs, axis=0)

    # Return
    return a_mean, bb_mean, a_5, a_95, bb_5, bb_95, Rrs, sigRs 

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
        odict = prep_data(170)

    # Indiv
    if flg & (2**1):
        fit_model('Indiv')

    # bb_water
    if flg & (2**2):
        fit_model('bbwater', nsteps=50000, nburn=5000)

    # water
    if flg & (2**3):
        fit_model('water', nsteps=50000, nburn=5000)

    # bp
    if flg & (2**4): # 16
        fit_model('bp', nsteps=50000, nburn=5000)

    # Exponential power-law
    if flg & (2**5): # 32
        fit_model('exppow', nsteps=10000, nburn=1000)

    # GIOP-like:  adg, aph, bbp
    if flg & (2**6): # 64
        fit_model('giop', nsteps=10000, nburn=1000)

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