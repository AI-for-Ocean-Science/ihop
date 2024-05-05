""" Fits to Loisel+2023 """
import os

# For emcee
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from ihop.emulators import io as emu_io
from ihop import io as ihop_io
from ihop.inference import fitting 
from ihop.inference import io as fitting_io

from IPython import embed

def init_mcmc(emulator, ndim, perc:int=None, abs_sig:float=None):
    # MCMC
    pdict = dict(model=emulator)
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['nsteps'] = 10000
    pdict['save_file'] = None
    pdict['scl_sig'] = perc
    pdict['abs_sig'] = abs_sig
    pdict['priors'] = None
    pdict['cut'] = None
    #
    return pdict

def select_spectra(Nspec:int, ntot:int, seed:int=71234):
    # Select a random sample
    np.random.seed(seed)
    idx = np.random.choice(np.arange(ntot), Nspec, replace=False)

    return idx

def add_noise(Rs, perc:int=None, abs_sig:float=None):
    """
    Add random noise to the input array Rs.

    Parameters:
        Rs (ndarray): Input array.
        perc (int, optional): Percentage of noise to be added as a fraction of Rs. Default is None.
        abs_sig (float, optional): Absolute value of noise to be added. Default is None.

    Returns:
        ndarray: Array with noise added.
    """
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

def load(edict:dict):
    """
    Load data and emulator from the given dictionary.

    Parameters:
        edict (dict): A dictionary containing the necessary information for loading data and emulator.

    Returns:
        tuple: A tuple containing the loaded data and emulator in the following order: 
            ab, Chl, Rs, emulator, d_a.
    """
    # Load data
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        edict['decomps'], edict['Ncomps'])
    # Load emulator
    emulator, e_file = emu_io.load_emulator_from_dict(
        edict, use_s3=True)

    # Return
    return ab, Chl, Rs, emulator, d_a

def fit(edict:dict, Nspec:int=None, abs_sig:float=None,
                      debug:bool=False, n_cores:int=1,
                      max_wv:float=None):
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

    """
    # Load
    ab, Chl, Rs, emulator, d_a = load(edict)

    # Output
    root = emu_io.set_l23_emulator_root(edict)
    outroot = f'fit_Rs01_{root}'
    if abs_sig is None:
        outroot.replace('fit', 'fitN')
    if max_wv is not None:
        outroot += f'_max{int(max_wv)}'

    # Init MCMC
    pdict = init_mcmc(emulator, ab.shape[1]+1)

    # Include a non-zero error to avoid bad chi^2 behavior
    if abs_sig is None:
        pdict['abs_sig'] = 1.
    else:
        pdict['abs_sig'] = abs_sig

    # Max wave?
    if max_wv is not None:
        cut = d_a['wave'] < max_wv
        pdict['cut'] = cut

    # No noise
    if abs_sig is None:
        use_Rs = Rs.copy()
    else:
        use_Rs = add_noise(Rs, abs_sig=abs_sig)

    # Prep
    if Nspec is None:
        idx = np.arange(len(Chl))
    else:
        idx = np.arange(Nspec)
    if debug:
        idx = idx[0:2]
    items = [(use_Rs[i], ab[i].tolist()+[Chl[i]], i) for i in idx]

    # Fit
    all_samples, all_idx = fitting.fit_batch(pdict, items,
                                             n_cores=n_cores)
    # Save
    save_fits(all_samples, all_idx, Rs, use_Rs, outroot)

def test_fit(edict:dict, Nspec:int=100, abs_sig:float=None,
             debug:bool=False, n_cores:int=1): 
    """
    Fits the data without considering any errors.

    Args:
        edict (dict): A dictionary containing the necessary information for fitting.
        Nspec (str): The number of spectra to fit. Default is 'all'.
        debug (bool): Whether to run in debug mode. Default is False.
        n_cores (int): The number of CPU cores to use for parallel processing. Default is 1.

    Returns:

    """
    # Load
    ab, Chl, Rs, emulator, d_a = load(edict)

    # Output
    outfile = fitting_io.l23_chains_filename(
        edict, int(abs_sig), test=True)
    outfile = os.path.basename(outfile)

    # Init MCMC
    pdict = init_mcmc(emulator, ab.shape[1]+1)
    # Include a non-zero error to avoid bad chi^2 behavior
    pdict['abs_sig'] = abs_sig

    # Select a random sample
    idx = select_spectra(Nspec, len(Chl))

    # Add noise fit_Rs01_L23_X4_Y0_pcapca_42_chl_Rrs_dense_512_512_512_256.npz
    use_Rs = add_noise(Rs, abs_sig=pdict['abs_sig'])

    # Prep                           #
    items = [(use_Rs[i], ab[i].tolist()+[Chl[i]], i) for i in idx]
    # Fit
    all_samples, all_idx = fitting.fit_batch(pdict, items,
                                             n_cores=n_cores)
    # Save
    save_fits(all_samples, all_idx, Rs, use_Rs, outfile)



def save_fits(all_samples, all_idx, Rs, use_Rs, outroot):
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
    np.savez(outfile, chains=all_samples, idx=all_idx,
             obs_Rs=use_Rs[all_idx], Rs=Rs[all_idx])
    print(f"Saved: {outfile}")


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
        n_cores = 20
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomp, Ncomp, 'Rrs',
            'dense', hidden_list=hidden_list, include_chl=True, 
            X=X, Y=Y)

        fit(edict, n_cores=n_cores)#, debug=True)

    # Noiseless, cut at 600nm
    if flg & (2**1):

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomp = 'nmf'
        Ncomp = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomp, Ncomp, 'Rrs',
            'dense', hidden_list=hidden_list, include_chl=True, 
            X=X, Y=Y)

        # Analysis params
        max_wv=600.

        fit(edict, n_cores=n_cores, max_wv=max_wv)#, debug=True)

    # Noiseless, PCA
    if flg & (2**2):

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('pca', 'pca')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        # Analysis params

        fit(edict, n_cores=n_cores)

    # Noiseless, INT/NMF
    if flg & (2**3):

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('int', 'nmf')
        Ncomps = (40,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        # Analysis params
        fit(edict, n_cores=n_cores, 
                          Nspec=100)#, debug=True)

    # PCA, abs_sig=1
    if flg & (2**4): # 16

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('pca', 'pca')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 1.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig)#, debug=True)

    # PCA, abs_sig=2
    if flg & (2**5): # 32

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('pca', 'pca')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 2.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig)#, debug=True)

    # PCA, abs_sig=5
    if flg & (2**6): # 64

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('pca', 'pca')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 5.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig)#, debug=True)

    # Testing
    if flg & (2**30):
        hidden_list=[512, 512, 512, 256]
        decomp = 'nmf'
        Ncomp = (4,2)
        X, Y = 4, 0
        #n_cores = 2
        n_cores = 16
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomp, Ncomp, 'Rrs',
            'dense', hidden_list=hidden_list, include_chl=True, 
            X=X, Y=Y)
        # Test fit
        test_fit(edict, abs_sig=2., n_cores=n_cores)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        # NOISELESS
        #flg += 2 ** 0  # 1 -- Noiseless
        #flg += 2 ** 1  # 2 -- Noiseless + cut at 600nm
        #flg += 2 ** 2  # 4 -- Noiseless, PCA 
        #flg += 2 ** 3  # 8 -- Noiseless, INT/NMF

        # PCA with Noise
        #flg += 2 ** 4  # 16 -- PCA, abs_sig=1
        #flg += 2 ** 5  # 32 -- PCA, abs_sig=2
        #flg += 2 ** 6  # 64 -- PCA, abs_sig=5

        # Tests
        flg += 2 ** 30  # 16 -- L23 + NMF 4,2

        
    else:
        flg = sys.argv[1]

    main(flg)