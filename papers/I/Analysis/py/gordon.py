""" Perform Gordon analyses """
import os

import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.inference import fgordon
from ihop.inference import noise

from IPython import embed

def prep_data(idx:int, scl_noise:float=0.02, add_noise:bool=False):
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
    adg = ds.ag.data[idx,:] + ds.ad.data[idx,:]
    aph = ds.aph.data[idx,:]

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
                 adg=adg, aph=aph,
                 Y=Y, Chl=Chl)

    return odict

def fit_model(model:str, n_cores=20, idx:int=170, 
              nsteps:int=10000, nburn:int=1000, 
              scl_noise:float=0.02,
              scl:float=None,  # Scaling for the priors
              add_noise:bool=False):

    odict = prep_data(idx, scl_noise=scl_noise, add_noise=add_noise)

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
    elif model == 'cstcst':
        # Exponential adg, power-law bpp with free exponent
        i400 = np.argmin(np.abs(wave-400))
        i500 = np.argmin(np.abs(wave-500))
        if scl is None:
            scl = 5.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i400]]
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = [bnw[i500]]
    elif model == 'exppow':
        # Exponential adg, power-law bpp with free exponent
        i400 = np.argmin(np.abs(wave-400))
        i500 = np.argmin(np.abs(wave-500))
        if scl is None:
            scl = 5.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i400], 0.017] 
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = [bnw[i500], odict['Y']]
    elif model == 'explee':
        # Exponential adg, power-law bpp with free exponent
        i400 = np.argmin(np.abs(wave-400))
        i500 = np.argmin(np.abs(wave-500))
        if scl is None:
            scl = 5.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i400], 0.017] 
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = [bnw[i500]]
    elif model == 'expcst':
        # Exponential adg, power-law bpp with free exponent
        i400 = np.argmin(np.abs(wave-400))
        i500 = np.argmin(np.abs(wave-500))
        if scl is None:
            scl = 5.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i400], 0.017] 
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = [bnw[i500]]
    elif model == 'giop':
        i440 = np.argmin(np.abs(wave-440))
        i500 = np.argmin(np.abs(wave-500))
        scl = 5.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i440]/2., 0.017, anw[i440]/2.] 
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = bnw[i500] 
    elif model == 'giop+':
        i440 = np.argmin(np.abs(wave-440))
        i500 = np.argmin(np.abs(wave-500))
        if scl is None:
            scl = 5.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i440]/2., 0.017, anw[i440]/2.] 
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = [bnw[i500], odict['Y']]
    elif model == 'hybpow':
        i440 = np.argmin(np.abs(wave-440))
        i500 = np.argmin(np.abs(wave-500))
        scl = 1.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i440]/2., 0.017, anw[i440]/4., anw[i440]/4.] 
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = [bnw[i500], odict['Y']]
    elif model == 'hybnmf':
        i440 = np.argmin(np.abs(wave-440))
        i500 = np.argmin(np.abs(wave-500))
        scl = 1.
        anw = np.maximum(scl*a[::2] - aw[::2], 1e-5)
        p0_a = [anw[i440]/2., 0.017, anw[i440]/4., anw[i440]/4.] 
        # bbp
        bnw = np.maximum(scl*bb[::2] - bbw[::2], 1e-5)
        p0_b = [bnw[i500]/2., bnw[i500]/2.]
    else:
        raise ValueError(f"51 of gordon.py -- Deal with this model: {model}")

    p0 = np.concatenate((np.log10(p0_a), 
                         np.log10(np.atleast_1d(p0_b))))
    #p0 = np.concatenate([p0_a, p0_b])

    # Gordon Rrs
    gordon_Rrs = fgordon.calc_Rrs(odict['a'][::2], odict['bb'][::2])
    if add_noise:
        gordon_Rrs = noise.add_noise(gordon_Rrs, perc=scl_noise*100)

    # Chk initial guess
    ca,cbb = fgordon.calc_ab(model, p0, pdict)
    pRrs = fgordon.calc_Rrs(ca, cbb)
    print(f'Initial Rrs guess: {np.mean((gordon_Rrs-pRrs)/gordon_Rrs)}')
    embed(header='196 of gordon')

    # Set the items
    #items = [(Rrs, varRrs, None, idx)]
    #items = [(Rrs, varRrs, p0, idx)]
    items = [(gordon_Rrs, varRrs, p0, idx)]

    # 
    chains, idx = fgordon.fit_one(items[0], pdict=pdict, chains_only=True)
    
    # Save
    outfile = f'FGordon_{model}_{idx}'
    if add_noise:
        # Add noise to the outfile with padding of 2
        outfile += f'_N{int(100*scl_noise):02d}'
    else:
        outfile += f'_n{int(100*scl_noise):02d}'
    save_fits(chains, idx, outfile,
              extras=dict(wave=wave, obs_Rrs=gordon_Rrs, varRrs=varRrs))

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
        fit_model('exppow', nsteps=80000, nburn=8000, scl=1., idx=170)
        #fit_model('exppow', nsteps=80000, nburn=8000, scl=1., idx=1032)
        #fit_model('exppow', nsteps=80000, nburn=8000, scl=1., idx=1032,
        #          scl_noise=0.05, add_noise=False)
        #fit_model('explee', nsteps=80000, nburn=8000, scl=1., idx=1032,
        #          scl_noise=0.05, add_noise=False)
        #fit_model('exppow', nsteps=80000, nburn=8000, scl=1.)
        #fit_model('exppow', nsteps=80000, nburn=8000,
        #          scl_noise=0.2, add_noise=False, scl=1.)

    # GIOP-like:  adg, aph, bbp
    if flg & (2**6): # 64
        fit_model('giop', nsteps=80000, nburn=8000, scl=1.)

    # GIOP-like:  adg, aph Bricaud, bbp with free exponent
    if flg & (2**7): # 128
        #fit_model('giop+', nsteps=80000, nburn=8000, scl=1.)
        #fit_model('giop+', nsteps=80000, nburn=8000, scl=1., idx=1032)
        fit_model('giop+', nsteps=80000, nburn=8000, scl=1., idx=1032,
                  scl_noise=0.07, add_noise=False)

    # NMF aph
    if flg & (2**8): # 256
        fit_model('hybpow', nsteps=80000, nburn=8000)
        fit_model('hybpow', nsteps=80000, nburn=8000,
                  scl_noise=0.07, add_noise=True)

    # Exp anw, constant bbp
    if flg & (2**9): # 512
        #fit_model('expcst', nsteps=80000, nburn=8000, scl=1.)
        fit_model('expcst', nsteps=80000, nburn=8000, scl=1., idx=1032)

    # NMF aph
    if flg & (2**10): # 1024
        fit_model('hybnmf', nsteps=80000, nburn=8000)

    # Exponential + power-law with Lee+2002
    if flg & (2**11): # 2048
        #fit_model('explee', nsteps=80000, nburn=8000, scl=1., idx=1032)
        fit_model('explee', nsteps=80000, nburn=8000, scl=1., idx=1032,
                  scl_noise=0.05, add_noise=False)
        #fit_model('explee', nsteps=80000, nburn=8000, scl=1., idx=180)
        #fit_model('explee', nsteps=80000, nburn=8000, scl=1.)

    # Constant and consant
    if flg & (2**12): # 4096
        fit_model('cstcst', nsteps=80000, nburn=8000, scl=1.)
        fit_model('cstcst', nsteps=80000, nburn=8000, scl=1., idx=1032)

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