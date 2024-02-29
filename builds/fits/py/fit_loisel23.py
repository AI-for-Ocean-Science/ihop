""" Fits to Loisel+2023 """
import os
import numpy as np

from ihop.emulators import io as emu_io
from ihop import io as ihop_io
from ihop.inference import fitting 

def init_mcmc(emulator, ndim, perc:int=None, abs_sig:float=None):
    # MCMC
    pdict = dict(model=emulator)
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['nsteps'] = 10000
    pdict['save_file'] = None
    pdict['scl_sig'] = perc
    pdict['abs_sig'] = abs_sig
    pdict['priors'] = None
    #
    return pdict

def select_spectra(Nspec:int, ntot:int, seed:int=71234):
    # Select a random sample
    np.random.seed(seed)
    idx = np.random.choice(np.arange(ntot), Nspec, replace=False)

    return idx

def add_noise(Rs, perc:int=None, abs_sig:float=None):
    # Add in random noise
    use_Rs = Rs.copy()
    r_sig = np.random.normal(size=Rs.shape)
    r_sig = np.minimum(r_sig, 3.)
    r_sig = np.maximum(r_sig, -3.)
    if perc is not None:
        use_Rs += (perc/100.) * use_Rs * r_sig
    else:
        use_Rs += r_sig * abs_sig
    # Return
    return use_Rs

def fit_without_error(edict:dict, Nspec:str='all',
                      debug:bool=False, n_cores:int=1): 
    # Load data
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(
        edict['decomp'], edict['Ncomp'])
    # Load emulator
    emulator, e_file = emu_io.load_emulator_from_dict(edict, use_s3=True)
    root = emu_io.set_l23_emulator_root(edict)

    outroot = f'fit_Rs_01_{root}'
    # Init MCMC
    pdict = init_mcmc(emulator, ab.shape[1]+1)
    # Include a non-zero error to avoid bad chi^2 behavior
    pdict['abs_sig'] = 1.

    # Add noise -- not for noiseless
    #use_Rs = add_noise(Rs, abs_sig=pdict['abs_sig'])
    use_Rs = Rs.copy()

    # Prep
    if Nspec == 'all':
        idx = np.arange(len(Chl))
    else:
        raise ValueError("Bad Nspec")
    if debug:
        idx = idx[0:2]
    items = [(use_Rs[i], ab[i].tolist()+[Chl[i]], i) for i in idx]

    # Fit
    all_samples, all_idx = fitting.fit_batch(pdict, items,
                                             n_cores=n_cores)

    # Save
    outdir = 'Fits/L23'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, outroot)
    np.savez(outfile, chains=all_samples, idx=all_idx,
             obs_Rs=Rs[all_idx], use_Rs=use_Rs[all_idx])


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Noiseless
    if flg & (2**0):
        hidden_list=[512, 512, 512, 256]
        decomp = 'nmf'
        Ncomp = (4,2)
        X, Y = 4, 0
        n_cores = 2
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomp, Ncomp, 'Rrs',
            'dense', hidden_list=hidden_list, include_chl=True, 
            X=X, Y=Y)

        fit_without_error(edict, n_cores=n_cores)# debug=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- L23 + PCA
        #flg += 2 ** 1  # 2 -- L23 + NMF
        #flg += 2 ** 2  # 4 -- L23 + NMF 4
        #flg += 2 ** 3  # 8 -- L23 + NMF 4,3
        #flg += 2 ** 4  # 16 -- L23 + NMF 4,2

        
    else:
        flg = sys.argv[1]

    main(flg)