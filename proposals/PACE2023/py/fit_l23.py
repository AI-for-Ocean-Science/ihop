""" Code to fit L23 with random Rs error """

import os

import numpy as np
import time

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torch

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

import seaborn as sns

import corner

from ihop.inference import mcmc
from ihop.emulators import io as ihop_io
from ihop.emulators import nn
from ihop.emulators.nn import SimpleNet, DenseNet
from ihop.iops import pca as ihop_pca
from ihop.iops.nmf import load_loisel_2023
from ihop.iops.pca import load_loisel_2023_pca

from IPython import embed

out_path = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'L23')

def load_hydro(iop_type:str='pca'):
    print("Loading Hydrolight data")
    if iop_type == 'pca':
        ab, Rs, d_a, d_bb = load_loisel_2023_pca()
        # Load model
        model_file = '/home/xavier/Projects/Oceanography/Color/IHOP/Emulators/SimpleNet_PCA'
        model_file += '/model_100000.pth'
        model = ihop_io.load_nn(model_file)
    elif iop_type == 'nmf':
        ab, Rs, d_a, d_bb = load_loisel_2023()
        em_path = os.path.join(os.getenv('OS_COLOR'), 
                               'IHOP', 'Emulators')
        model_file = os.path.join(
            em_path, 'DenseNet_NM4',
            'densenet_NMF_[512, 512, 512, 256]_epochs_25000_p_0.0_lr_0.01.pth')
                                  
                       
        print(f"Loading model: {model_file}")
        model = ihop_io.load_nn(model_file)

    else:
        raise IOError(f"Bad iop_type: {iop_type}")
    # Return
    return ab, Rs, d_a, d_bb, model

def do_all_fits(n_cores:int=4, iop_type:str='pca',
                fake:bool=False):

    for perc in [0, 5, 10, 15, 20]:
        print(f"Working on: perc={perc}")
        fit_fixed_perc(perc=perc, n_cores=n_cores, Nspec=100,
                       iop_type=iop_type, fake=fake)

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
        mxwv = d_a['wavelength'][imax_dev]

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
    ax.plot(d_a['wavelength'], orig, 'bo', label='True')
    ax.plot(d_a['wavelength'], a_mean, 'r-', label='Fit')
    ax.plot(d_a['wavelength'], a_pca, 'k:', label='PCA')
    ax.fill_between(
        d_a['wavelength'], a_mean-a_std, a_mean+a_std, 
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
    ax.plot(d_a['wavelength'], Rs[idx], 'bo', label='True')
    ax.plot(d_a['wavelength'], obs_Rs[in_idx], 'ks', label='Obs')
    ax.plot(d_a['wavelength'], pred_Rs, 'rx', label='Model')
    ax.plot(d_a['wavelength'], NN_Rs, 'g-', label='NN+True')

    ax.fill_between(
        d_a['wavelength'], pred_Rs-std_pred, pred_Rs+std_pred,
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

def fit_one(items:list, pdict:dict=None):
    # Unpack
    Rs, ab, idx = items

    # Init (cheating, but do it)
    if hasattr(pdict['model'], 'ninput'):
        ndim = pdict['model'].ninput
    else:
        ndim = 8 # NMF

    # Run
    sampler = mcmc.run_emcee_nn(
        pdict['model'], Rs,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        scl_sig=pdict['scl_sig']/100. if pdict['scl_sig'] is not None else None,
        abs_sig=pdict['abs_sig'] if pdict['abs_sig'] is not None else None,
        p0=ab,
        save_file=pdict['save_file'])

    # Return
    return sampler, idx

def fit_fixed_perc(perc:int, n_cores:int, seed:int=1234,
                   Nspec:int=100, iop_type:str='pca',
                   fake:bool=False):
    # Outfile
    outfile = os.path.join(out_path,
        f'fit_a_L23_NN_Rs{perc:02d}')

    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb, model = load_hydro(iop_type=iop_type)
    nwave = Rs.shape[1]
    #ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()

    if fake:
        print("Using fake Rs")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_Rs = np.zeros((ab.shape[0], nwave))
        for kk in range(ab.shape[0]):
            pred_Rs = model.prediction(ab[kk], device)
            use_Rs[kk,:] = pred_Rs
    else:
        use_Rs = Rs

    # Select a random sample
    np.random.seed(seed)
    idx = np.random.choice(np.arange(use_Rs.shape[0]), 
                           Nspec, replace=False)

    # Add in random noise
    r_sig = np.random.normal(size=use_Rs.shape)
    r_sig = np.minimum(r_sig, 3.)
    r_sig = np.maximum(r_sig, -3.)
    use_Rs += (perc/100.) * use_Rs * r_sig

    # MCMC
    pdict = dict(model=model)
    pdict['nwalkers'] = 16
    pdict['nsteps'] = 10000
    pdict['save_file'] = None
    pdict['perc'] = perc

    # Setup for parallel
    map_fn = partial(fit_one, pdict=pdict)

    # Prep
    items = [(use_Rs[i], ab[i], i) for i in idx]
    
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
    np.savez(outfile, chains=all_samples, idx=all_idx,
             obs_Rs=Rs[all_idx], use_Rs=use_Rs[all_idx])
    print(f"Wrote: {outfile}")

def another_test(iop_type:str='pca',
                 fake:bool=False,
                 n_cores:int=4):
    fit_fixed_perc(perc=10, n_cores=n_cores, Nspec=8, 
                   iop_type=iop_type, fake=fake)

def quick_test(iop_type:str='pca', fake:bool=False,
               perc:int=None, idx:int=1000,
               max_perc:float=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb, model = load_hydro(iop_type=iop_type)
    wave = d_a['wavelength'] if iop_type == 'pca' else d_a['wave']
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
    if perc is not None or max_perc is not None:
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

    items = [use_Rs, ab[idx], idx]

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
    np.savez(outfile, chains=all_samples,
             idx=all_idx,
             obs_Rs=use_Rs.reshape((1,use_Rs.size)),
    )
    print(f"Wrote: {outfile}")

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
        labels=['a0', 'a1', 'a2', 'b0', 'b1', 'b2']
    elif iop_type == 'nmf':
        labels=['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3']

    fig = corner.corner(samples, labels=labels, truths=ab[idx])
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
    model_Rs = model.prediction(ab[idx], device)
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


def fig_bspline_tara(outfile='fig_bspline_tara.png'):

    # Load up
    wv_grid, tara_a_water, l23_a = ihop_pca.load_tara()
    wv_grid = wv_grid.astype(np.float64)

    L23_Tara_pca = ihop_pca.load_pca(f'pca_L23_X4Y0_Tara_a_N3.npz')

    imx = np.argmax(tara_a_water[:,3])


    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # #####################################################
    # RMS
    ax_rms= plt.subplot(gs[0])

    # #####################################################
    # Spectra
    ax_spec= plt.subplot(gs[1])

    # B-spline it
    ifit = 1
    everyns = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for clr, idx in zip(['b', 'g'], [100000, imx]):
        orig, recon = ihop_pca.reconstruct(L23_Tara_pca, idx)
        orig = orig.astype(np.float64)
        sv_fits = [] 
        rmss = []
        for everyn in everyns:
            my_bspline = bspline(wv_grid, nord=3,
                        everyn=everyn)
            code, yfit = my_bspline.fit(wv_grid, orig, np.ones_like(orig))
            sv_fits += [yfit]
            # RMS
            rmss += [np.sqrt(np.mean((orig - yfit)**2))]

        # Plot RMS vs. everyn
        ax_rms.plot(everyns, rmss, 'o', color=clr)

        ax_rms.set_xlabel('Every n')
        ax_rms.set_ylabel(r'RMS')
        #ax_rms.set_ylim(-1., 1.1*np.max(orig))


        ax_spec.plot(wv_grid, orig,  'o', color=clr, label=f'Tara: idx={idx}')
        # Fit
        ax_spec.plot(wv_grid, sv_fits[ifit],  ls='-', color=clr,
                     label=f'B-Spline: everyn={everyns[ifit]}')

        ax_spec.set_xlabel('Wavelength (nm)')
        ax_spec.set_ylabel(r'$a(\lambda)$')
        ax_spec.set_ylim(0., 1.1*np.max(orig))
        ax_spec.legend(fontsize=15.)
        

    # Finish
    for ax in [ax_rms, ax_spec]:
        plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


if __name__ == '__main__':

    # Testing
    #quick_test()
    quick_test(iop_type='nmf', fake=True, max_perc=2,
               idx=1000)

    #another_test()
    #another_test(iop_type='nmf', fake=True, n_cores=1)

    # All of em
    #do_all_fits(iop_type='pca')
    #do_all_fits(iop_type='nmf', n_cores=4, fake=True)

    # Analysis
    #stats = analyze_l23('fit_a_L23_NN_Rs10.npz')
    #check_one('fit_a_L23_NN_Rs10.npz', 0)
