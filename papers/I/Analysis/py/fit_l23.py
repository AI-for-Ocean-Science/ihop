""" Code to fit L23 with PCA Emulator """

import os

import numpy as np
import time

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torch

import corner


from ihop.inference import mcmc
from ihop import io as ihop_io
from ihop.emulators import io as emu_io
#from ihop.iops import pca as ihop_pca

from IPython import embed

out_path = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'Fits', 'L23')


def do_all_fits(n_cores:int=4, decomp:str='pca', Ncomp:int=3):
    """
    Perform fits for different percentages.

    Args:
        n_cores (int): Number of CPU cores to use for parallel processing.
        decomp (str): Type of IOP (Input/Output Parameters) to use.
        fake (bool): Flag indicating whether to use fake data.

    Returns:
        None
    """

    for perc in [0, 5, 10, 15, 20]:
        print(f"Working on: perc={perc}")
        fit_fixed_perc(perc=perc, n_cores=n_cores, Nspec=100,
                       decomp=decomp, Ncomp=Ncomp)

def analyze_l23(chain_file, chop_burn:int=-4000,
                iop_type:str='pca'):

    # #############################################
    # Load

    # Load Hydrolight
    print("Loading Hydrolight data")
    if iop_type == 'pca':
        ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()
    elif iop_type == 'nmf':
        ab, Rs, _, _ = load_loisel_2023()

    # MCMC
    print("Loading MCMC")
    d = np.load(os.path.join(out_path, chain_file))
    chains = d['chains']
    l23_idx = d['idx']

    all_medchi, all_stdchi, all_rms, all_maxdev = [], [], [], []
    all_mxwv = []
    
    # Loop
    for ss, idx in enumerate(l23_idx):
        # a
        Y = chains[ss, chop_burn:, :, 0:3].reshape(-1,3)
        orig, a_recon = ihop_pca.reconstruct(Y, d_a, idx)
        a_mean = np.mean(a_recon, axis=0)
        a_std = np.std(a_recon, axis=0)

        # Stats
        rms = np.std(a_mean-orig)
        chi = np.abs(a_mean-orig)/a_std
        dev = np.abs(a_mean-orig)/a_mean
        imax_dev = np.argmax(dev)
        max_dev = dev[imax_dev]
        mxwv = d_a['wave'][imax_dev]

        # Save
        all_rms.append(rms)
        all_maxdev.append(max_dev)
        all_medchi.append(np.median(chi))
        all_stdchi.append(np.std(chi))
        all_mxwv.append(mxwv)

    # Return
    stats = dict(rms=all_rms,
                 max_dev=all_maxdev,
                 med_chi=all_medchi,
                 std_chi=all_stdchi,
                 mx_wave=all_mxwv)
    return stats


def check_one(chain_file:str, in_idx:int, chop_burn:int=-3000):

    # #############################################
    # Load

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ihop_io.load_nn('model_100000')

    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()

    # MCMC
    print("Loading MCMC")
    d = np.load(os.path.join(out_path, chain_file))
    chains = d['chains']
    l23_idx = d['idx']
    obs_Rs = d['obs_Rs']

    idx = l23_idx[in_idx]
    print(f'Working on: L23 index={idx}')
    
    # a
    Y = chains[in_idx, chop_burn:, :, 0:3].reshape(-1,3)
    orig, a_recon = ihop_pca.reconstruct(Y, d_a, idx)
    a_mean = np.mean(a_recon, axis=0)
    a_std = np.std(a_recon, axis=0)
    _, a_pca = ihop_pca.reconstruct(ab[idx][:3], d_a, idx)

    # Rs
    allY = chains[in_idx, chop_burn:, :, :].reshape(-1,6)
    all_pred = np.zeros((allY.shape[0], 81))
    for kk in range(allY.shape[0]):
        Ys = allY[kk]
        pred_Rs = model.prediction(Ys, device)
        all_pred[kk,:] = pred_Rs

    pred_Rs = np.median(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)
    NN_Rs = model.prediction(ab[idx], device)

    # #########################################################
    # Plot the solution
    plt.clf()
    ax = plt.gca()
    ax.plot(d_a['wave'], orig, 'bo', label='True')
    ax.plot(d_a['wave'], a_mean, 'r-', label='Fit')
    ax.plot(d_a['wave'], a_pca, 'k:', label='PCA')
    ax.fill_between(
        d_a['wave'], a_mean-a_std, a_mean+a_std, 
        color='r', alpha=0.5) 

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$a(\lambda)$')

    plt.show()

    '''
    # #########################################################
    # Plot the residuals
    plt.clf()
    ax = plt.gca()
    ax.plot(d_a['wavelength'], a_mean-orig, 'bo', label='True')
    ax.fill_between(d_a['wavelength'], -a_std, a_std, 
        color='r', alpha=0.5) 

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$a(\lambda)$ [Fit-Orig]')

    plt.show()
    '''

    # #########################################################
    # Compare Rs
    plt.clf()
    ax = plt.gca()
    ax.plot(d_a['wave'], Rs[idx], 'bo', label='True')
    ax.plot(d_a['wave'], obs_Rs[in_idx], 'ks', label='Obs')
    ax.plot(d_a['wave'], pred_Rs, 'rx', label='Model')
    ax.plot(d_a['wave'], NN_Rs, 'g-', label='NN+True')

    ax.fill_between(
        d_a['wave'], pred_Rs-std_pred, pred_Rs+std_pred,
        color='r', alpha=0.5) 

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$R_s$')

    ax.legend()

    plt.show()

    # Corner
    fig = corner.corner(
        allY, labels=['a0', 'a1', 'a2', 'b0', 'b1', 'b2'],
        truths=ab[idx])
    plt.show() 



def fit_fixed(Ncomp, perc:int=None, abs_sig:float=None,
              n_cores:int=1, seed:int=2234, Nspec:int=100, 
              decomp:str='pca', test:bool=False):
                   
    """
    Fits a model using fixed percentage or absolute
    error on the input data.

    Args:
        Ncomp (tuple, optional): The number of components. (a, bb)
        perc (int, optional): The percentage of perturbation to apply to the input data.
        abs_sig (float, optional): The absolute error to apply to the input data.
        n_cores (int): The number of CPU cores to use for parallel processing.
        seed (int, optional): The random seed for reproducibility. Defaults to 1234.
        Nspec (int, optional): The number of random samples to select. Defaults to 100.
        decomp (str, optional): The type of IOP (Inherent Optical Property) to use. Defaults to 'pca'.
        test (bool, optiona0): If true, this is only a test

    """
    if perc is None and abs_sig is None:
        raise ValueError("Must specify either perc or abs_sig")
    #os.environ["OMP_NUM_THREADS"] = "1"
    hidden_list=[512, 512, 512, 256]
    X, Y = 4, 0
    dataset = 'L23'


    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)
    edict = emu_io.set_emulator_dict(dataset, decomp, Ncomp, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)
    ndim = ab.shape[1]+1
    emulator, e_file = emu_io.load_emulator_from_dict(edict)
    nwave = Rs.shape[1]

    root = emu_io.set_l23_emulator_root(edict)
    # Outfile
    if perc is not None:
        outroot = f'fit_Rs{perc:02d}_{root}'
    else:
        outroot = f'fit_Rs{int(abs_sig):02d}_{root}'
    if test:
        outroot = 'test_'+outroot
    outfile = os.path.join(out_path, outroot)

    use_Rs = Rs

    # Select a random sample
    np.random.seed(seed)
    idx = np.random.choice(np.arange(use_Rs.shape[0]),
                           Nspec, replace=False)

    # Add in random noise
    r_sig = np.random.normal(size=use_Rs.shape)
    r_sig = np.minimum(r_sig, 3.)
    r_sig = np.maximum(r_sig, -3.)
    if perc is not None:
        use_Rs += (perc/100.) * use_Rs * r_sig
    else:
        use_Rs += r_sig * abs_sig

    # MCMC
    pdict = dict(model=emulator)
    pdict['nwalkers'] = max(16,ndim*2)
    pdict['nsteps'] = 10000
    pdict['save_file'] = None
    pdict['scl_sig'] = perc
    pdict['abs_sig'] = abs_sig
    pdict['priors'] = None

    # Setup for parallel
    map_fn = partial(fit_one, pdict=pdict)

    # Prep
    items = [(use_Rs[i], ab[i].tolist()+[Chl[i]], i) for i in idx]
    
    # Parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    # Slurp
    samples = [item[0].get_chain() for item in answers]
    all_idx = np.array([item[1] for item in answers])

    # Chains
    all_samples = np.zeros((len(samples), samples[0].shape[0], 
        samples[0].shape[1], samples[0].shape[2]))
    for ss in range(len(all_idx)):
        all_samples[ss,:,:,:] = samples[ss]

    # Save
    #embed(header='fit_l23.py 311')
    np.savez(outfile, chains=all_samples, idx=all_idx,
             obs_Rs=Rs[all_idx], use_Rs=use_Rs[all_idx])
    print(f"Wrote: {outfile}")

def another_test(Ncomp, decomp:str='pca',
                 n_cores:int=4,
                 perc:int=None, abs_sig:float=None,
                 Nspec=8):
    """
    Perform another test using the specified IOP type, fake flag, and number of cores.

    Args:
        iop_type (str, optional): The type of IOP to use for the test. Default is 'pca'.
        fake (bool, optional): Flag indicating whether to use fake data for the test. Default is False.
        n_cores (int, optional): The number of CPU cores to use for parallel processing. Default is 4.

    """
    fit_fixed(Ncomp, n_cores=n_cores, perc=perc, abs_sig=abs_sig,
                   Nspec=Nspec, decomp=decomp, test=True)

def quick_test(iop_type:str='pca', fake:bool=False,
               perc:int=None, idx:int=1000,
               max_perc:float=None, 
               seed=None):
    """
    Perform a quick test for fitting using the specified parameters.

    Args:
        iop_type (str): The type of IOP (Inherent Optical Property) to use for fitting. Default is 'pca'.
        fake (bool): Whether to use fake data for testing. Default is False.
        perc (int or float): The percentage of random noise to add to the data. Default is None.
        idx (int): The index of the data to use for fitting. Default is 1000.
        max_perc (float): The maximum percentage of absolute error to add to the data. Default is None.
        seed: The seed for random number generation. Default is None.
    """
    if perc is None and max_perc is None:
        raise ValueError("Must specify either perc or max_perc")
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 150

    if seed is not None:
        np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load Hydrolight
    ab, Chl, Rs, d_a, d_bb, model = load()#iop_type=iop_type)
    wave = d_a['wave'] if iop_type == 'pca' else d_a['wave']
    nwave = wave.size
    ncomp = ab.shape[1]//2

    outfile = f'quick_fit_{iop_type}'

    pdict = dict(model=model)
    pdict['nwalkers'] = 16
    pdict['nsteps'] = 20000
    pdict['save_file'] = 'tmp.h5'
    pdict['scl_sig'] = perc 
    pdict['abs_sig'] = None

    if fake:
        use_Rs = model.prediction(ab[idx], device)
    else:
        use_Rs = Rs[idx]

    # Add in random noise
    r_sig = np.random.normal(size=use_Rs.shape)
    r_sig = np.minimum(r_sig, 3.)
    r_sig = np.maximum(r_sig, -3.)
    if max_perc is not None:
        # Absolute error
        mx_Rs = max(use_Rs)
        pdict['abs_sig'] = mx_Rs * max_perc / 100.
        use_Rs += r_sig * pdict['abs_sig']
        # Keep a floor -- FAKE!
        #use_Rs = np.maximum(use_Rs, 1e-4)
    else:
        use_Rs += (perc/100.) * use_Rs * r_sig

    inputs = ab[idx].tolist()+[Chl[idx]]
    items = [use_Rs, inputs, idx]

    t0 = time.time()
    sampler, _ = fit_one(items, pdict=pdict)
    t1 = time.time()
    dt = t1-t0
    print(f'Time: {dt} sec')
    samples = sampler.get_chain()
    all_samples = samples.copy()
    samples = samples[-4000:,:,:].reshape((-1, samples.shape[-1]))

    # Save
    all_idx = [idx]
    np.savez(outfile, 
             chains=all_samples,
             idx=all_idx,
             ab=ab[idx],
             obs_Rs=use_Rs.reshape((1,use_Rs.size)),
    )
    print(f"Wrote: {outfile}")

    # ###########################################
    # ###########################################
    # ###########################################
    # ###########################################
    # Plots
    #
    cidx = 0
    plt.clf()
    plt.hist(samples[:,cidx], 100, color='k', histtype='step')
    ax = plt.gca()
    ax.axvline(ab[idx][cidx])
    #
    plt.show()

    # Corner
    if iop_type == 'pca':
        labels=['a0', 'a1', 'a2', 'b0', 'b1', 'b2', 'Chl']
    elif iop_type == 'nmf':
        labels=['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3']


    fig = corner.corner(samples, labels=labels, truths=inputs)
    plt.show()

    # Plot a
    mean_ans = np.mean(samples, axis=0)
    median_ans = np.median(samples, axis=0)
    a_true = np.dot(ab[idx][0:ncomp], d_a['M'])
    a_fits = np.dot(samples[:,0:ncomp], d_a['M'])

    med_fits = np.median(a_fits, axis=0)
    std_fits = np.std(a_fits, axis=0)

    plt.clf()
    ax = plt.gca()
    ax.plot(wave, a_true, 'ko', label='True')
    #ax.plot(wave, a_fit, 'b-', label='Fit')
    ax.plot(wave, med_fits, 'b-', label='Fit')

    ax.fill_between(wave,
        med_fits-std_fits, med_fits+std_fits,
        color='b', alpha=0.5) 
    ax.legend()
    plt.show()

    # ###########################################
    # Plot Rs
    model_Rs = model.prediction(inputs, device)
    all_pred = np.zeros((samples.shape[0], nwave))
    for kk in range(samples.shape[0]):
        Ys = samples[kk]
        pred_Rs = model.prediction(Ys, device)
        all_pred[kk,:] = pred_Rs

    fit_Rs = np.median(all_pred, axis=0)

    plt.clf()
    ax = plt.gca()
    ax.scatter(wave, Rs[idx], color='k', label='True Rs', 
        s=5, facecolors='none')
    ax.plot(wave, use_Rs, 'kx', label='Used Rs')
    ax.plot(wave, model_Rs, 'b-', label='Model Rs')
    ax.plot(wave, fit_Rs, 'r-', label='Fit Rs')

    ax.set_ylabel('Rs')
    ax.set_xlabel('Wavelenegth (nm)')

    #ax.fill_between(wave,
    #    med_fits-std_fits, med_fits+std_fits,
    #    color='b', alpha=0.5) 
    ax.legend()
    plt.show()

    embed(header='342 of fit_l23.py')


if __name__ == '__main__':

    # Testing
    #quick_test(perc=5)
    #quick_test(iop_type='nmf', fake=True, max_perc=2,
    #           idx=1000, seed=12345)

    #another_test(n_cores=2)
    #another_test((4,2), decomp='nmf', n_cores=1, Nspec=1, perc=10.)
    #another_test((4,2), decomp='nmf', n_cores=1, Nspec=1, abs_sig=1.)
    another_test((4,2), decomp='nmf', n_cores=1, Nspec=1, abs_sig=2.)
    #another_test((4,3), decomp='nmf', n_cores=1, Nspec=1)

    # All of em
    #do_all_fits(decomp='pca', n_cores=1)
    #do_all_fits(decomp='nmf', n_cores=1)
    #do_all_fits(decomp='nmf', n_cores=1, Ncomp=4)
    #do_all_fits(iop_type='nmf', n_cores=4, fake=True)


    # Analysis
    #stats = analyze_l23('fit_a_L23_NN_Rs10.npz')
    #check_one('fit_a_L23_NN_Rs10.npz', 0)
