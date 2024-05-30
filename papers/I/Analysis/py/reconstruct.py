""" Methods for paper specfici reconstruction """

from importlib import reload 

import os
import numpy as np
import torch

import datetime

from ihop.iops.decompose import reconstruct_nmf
from ihop.iops.decompose import reconstruct_pca
from ihop.iops.decompose import reconstruct_int
from ihop.iops.decompose import reconstruct_hyb

from ihop.emulators import io as emu_io
from ihop import io as ihop_io
from ihop.inference import io as inf_io
from ihop.inference import analysis as inf_analysis
from ihop.inference import io as fitting_io

from IPython import embed

def reconstruct_bsp(coeffs, d_a, idx):
    from pypeit.bspline.utilc import bspline_model
    from pypeit.bspline import bspline

    wv64 = d_a['wave'].astype(np.float64)

    # Build the bspline
    bspline_dict = {}
    bspline_dict['xmin'] = 0.
    bspline_dict['xmax'] = 1.
    bspline_dict['npoly'] = 1
    bspline_dict['nord'] = 3
    bspline_dict['funcname'] = 'legendre'
    #
    bspline_dict['coeff'] = coeffs[0]
    bspline_dict['icoeff'] = np.zeros_like(bspline_dict['coeff'])

    bspline_dict['breakpoints'] = d_a['breakpoints']
    bspline_dict['mask'] = np.ones_like(bspline_dict['breakpoints'], dtype=bool)
#
    #
    new_bspline = bspline(wv64, from_dict=bspline_dict)

    a1, lower, upper = new_bspline.action(wv64)
    n = new_bspline.mask.sum() - new_bspline.nord

    yfits = []
    for goodcoeff in coeffs.astype(np.float64):
        yfit = bspline_model(wv64, a1, lower, upper, 
                          goodcoeff, n, new_bspline.nord, 
                          new_bspline.npoly)
        yfits.append(yfit)
    # Orig
    orig = d_a['data'][idx]
    return orig, np.array(yfits)

def one_spectrum(in_idx:int, ab, Chl, d_chains, d_a, d_bb, emulator,
                 decomps:tuple, Ncomp:tuple, use_Chl:bool=False,
                 chop_burn:int=-3000, in_log10:bool=False):
    chains = d_chains['chains'][in_idx]
    if in_log10:
        chains = 10**chains
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l23_idx = d_chains['idx']
    obs_Rs = d_chains['obs_Rs']

    idx = l23_idx[in_idx]
    print(f'Working on: L23 index={idx}')

    # Parse
    nwave = d_a['wave'].size
    wave = d_a['wave']

    # Prep
    decompd = dict(pca=reconstruct_pca, nmf=reconstruct_nmf, int=reconstruct_int,
                   bsp=reconstruct_bsp, hyb=reconstruct_hyb)
    rfunc = decompd[decomps[0]]
    
    # a
    Y = chains[chop_burn:, :, 0:Ncomp[0]].reshape(-1,Ncomp[0])
    orig, a_recon = rfunc(Y, d_a, idx)
    #_, a_nmf = rfunc(d_a['coeff'][idx], d_a, idx)
    a_mean = np.median(a_recon, axis=0)
    a_std = np.std(a_recon, axis=0)
    _, tmp = rfunc(np.atleast_2d(ab[idx][:Ncomp[0]]), d_a, idx)
    a_pca = tmp[0]
    print("Done with a")

    rfunc = decompd[decomps[1]]
    # bb
    Y = chains[chop_burn:, :, Ncomp[0]:-1].reshape(-1,Ncomp[1])
    orig_bb, bb_recon = rfunc(Y, d_bb, idx)
    #_, bb_nmf = rfunc(d_bb['coeff'][idx], d_bb, idx)
    bb_mean = np.median(bb_recon, axis=0)
    bb_std = np.std(bb_recon, axis=0)
    #_, a_pca = rfunc(ab[idx][:ncomp], d_a, idx)
    print("Done with bb")

    # Rs
    add_Chl = 1 if use_Chl else 0
    allY = chains[chop_burn:, :, :].reshape(-1,Ncomp[0]+Ncomp[1]+add_Chl) # Chl
    all_pred = np.zeros((allY.shape[0], nwave))
    for kk in range(allY.shape[0]):
        Ys = allY[kk]
        pred_Rs = emulator.prediction(Ys, device)
        all_pred[kk,:] = pred_Rs

    pred_Rs = np.median(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)
    if use_Chl:
        NN_Rs = emulator.prediction(ab[idx].tolist() + [Chl[idx]], device)
    else:
        NN_Rs = emulator.prediction(ab[idx].tolist(), device)

    a_nmf, bb_nmf = None, None
    return idx, orig, a_mean, a_std, a_pca, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, allY, wave,\
        orig_bb, bb_mean, bb_std, a_nmf, bb_nmf

# #############################################################################3
def all_spectra(decomps:tuple, Ncomps:tuple, 
                hidden_list:list=[512, 512, 512, 256], 
                dataset:str='L23', perc:int=None, 
                use_log_ab:bool=False, include_Chl:bool=True,
                abs_sig:float=None, nchains:int=None,
                X:int=4, Y:int=0, quick_and_dirty:bool=False):

    priors = None
    if use_log_ab:
        priors = {}
        priors['use_log_ab'] = True

    d_keys = dict(pca='Y', nmf='coeff', int='new_spec', hyb='coeff')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = {}

    # Load
    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=include_Chl, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    wave = d_a['wave']

    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    chain_file = inf_io.l23_chains_filename(
        edict, perc if perc is not None else abs_sig,
        priors=priors) 
    outfile = os.path.basename(chain_file).replace('fit', 'recon')

    # Chains
    d_chains = inf_io.load_chains(chain_file)

    if nchains is not None:
        chains = d_chains['chains'][:nchains]
    else:
        chains = d_chains['chains']
        nchains = chains.shape[0]
    chain_idx = d_chains['idx'][:nchains]
    outputs['idx'] = chain_idx
    
    # Chop off the burn
    chains = inf_analysis.chop_chains(chains)

    if use_log_ab:
        chains = 10**chains

    # ##############
    # Rrs
    print("Starting Rrs")
    fit_Rrs, std_Rrs = inf_analysis.calc_Rrs(
        emulator, chains, quick_and_dirty=quick_and_dirty, verbose=False)
    outputs['fit_Rrs'] = fit_Rrs
    outputs['fit_Rrs_std'] = std_Rrs

    # Correct estimates
    if include_Chl:
        items = [ab[i].tolist()+[Chl[i]] for i in chain_idx]
    else:
        items = [ab[i].tolist() for i in chain_idx]
    corr_Rrs = []
    for item in items:
        iRs = emulator.prediction(item, device)
        corr_Rrs.append(iRs)
    corr_Rrs = np.array(corr_Rrs)

    outputs['corr_Rrs'] = corr_Rrs
    print("Done with Rrs")

    # ##############
    # a
    print("Starting a")
    a_means, a_stds = inf_analysis.calc_iop(
        chains[...,:Ncomps[0]], decomps[0], d_a)
    outputs['fit_a_mean'] = a_means
    outputs['fit_a_std'] = a_stds

    # Decomposed
    if decomps[0] == 'pca':
        rfunc = reconstruct_pca
    elif decomps[0] == 'nmf':
        rfunc = reconstruct_nmf
    elif decomps[0] == 'int':
        rfunc = reconstruct_int
    elif decomps[0] == 'hyb':
        rfunc = reconstruct_hyb
    else:
        raise ValueError(f"Your decomps={decomps[0]} is not supported.")

    # Original reconstruction 
    if decomps[0] == 'int':
        _, a_recons = rfunc(d_a[d_keys[decomps[0]]], d_a, 0)
    else:
        a_recons = []
        for idx in chain_idx:
            _, a_recon = rfunc(d_a[d_keys[decomps[0]]][idx], d_a, idx)
            a_recons.append(a_recon.flatten())
    outputs['decomp_a'] = np.array(a_recons)
    print("Done with a")

    # ##############
    # bb
    bb_means, bb_stds = inf_analysis.calc_iop(
        chains[...,Ncomps[0]:Ncomps[0]+Ncomps[1]], 
        decomps[1], d_bb)
    outputs['fit_bb_mean'] = bb_means
    outputs['fit_bb_std'] = bb_stds

    # Decomposed
    if decomps[1] == 'pca':
        rfunc = reconstruct_pca
    elif decomps[1] == 'nmf':
        rfunc = reconstruct_nmf
    else:
        raise ValueError(f"Your decomps={decomps[1]} is not supported.")

    bb_recons = []
    for idx in chain_idx:
        _, bb_recon = rfunc(d_bb[d_keys[decomps[1]]][idx], d_bb, idx)
        bb_recons.append(bb_recon)
    outputs['decomp_bb'] = np.array(bb_recons)

    # Save!
    np.savez(outfile, **outputs)
    print(f'Saved to: {outfile}')

# Command line execution
if __name__ == '__main__':

    # Noiseless
    #all_spectra(('nmf', 'nmf'), (4,2), abs_sig=None, quick_and_dirty=True)#, nchains=300)
    #all_spectra(('pca', 'pca'), (4,2), abs_sig=None, quick_and_dirty=True)#, nchains=500)
    #all_spectra(('int', 'nmf'), (40,2), abs_sig=None)#, nchains=100)
    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig=None, quick_and_dirty=True)#, nchains=300)

    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig=None, quick_and_dirty=True,
    #            use_log_ab=True)#, nchains=300)
    #all_spectra(('nmf', 'nmf'), (3,2), abs_sig=None, quick_and_dirty=True,
    #            use_log_ab=True)#, nchains=300)
    #all_spectra(('nmf', 'nmf'), (4,2), abs_sig=None, quick_and_dirty=True,
    #            use_log_ab=True)#, nchains=300)
    #all_spectra(('hyb', 'nmf'), (4,2), abs_sig=None, quick_and_dirty=True,
    #    include_Chl=False, use_log_ab=True)#, nchains=300)

    # PCA with noise
    #all_spectra(('pca', 'pca'), (4,2), abs_sig=1., quick_and_dirty=True)#, nchains=500)
    #all_spectra(('pca', 'pca'), (4,2), abs_sig=2., quick_and_dirty=True)#, nchains=500)
    #all_spectra(('pca', 'pca'), (4,2), abs_sig=5., quick_and_dirty=True)#, nchains=500)

    # NMF with noise
    #all_spectra(('nmf', 'nmf'), (4,2), abs_sig=1., quick_and_dirty=True)#, nchains=500)
    #all_spectra(('nmf', 'nmf'), (4,2), abs_sig=2., quick_and_dirty=True)#, nchains=500)
    #all_spectra(('nmf', 'nmf'), (4,2), abs_sig=5., quick_and_dirty=True)#, nchains=500)

    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig=2., quick_and_dirty=True)#, nchains=500)
    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig=5., quick_and_dirty=True)#, nchains=500)

    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig=2., quick_and_dirty=True,
    #            use_log_ab=True)#, nchains=500)
    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig=5., quick_and_dirty=True,
    #            use_log_ab=True)#, nchains=500)

    #all_spectra(('nmf', 'nmf'), (3,2), abs_sig=2., quick_and_dirty=True,
    #            use_log_ab=True)#, nchains=500)
    #all_spectra(('nmf', 'nmf'), (4,2), abs_sig=2., quick_and_dirty=True,
    #            use_log_ab=True)#, nchains=500)

    # PACE
    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig='PACE', quick_and_dirty=True, use_log_ab=True)#, nchains=500)
    #all_spectra(('nmf', 'nmf'), (2,2), abs_sig='PACE_CORR', quick_and_dirty=True, use_log_ab=True)#, nchains=500)

    # Hybrid
    all_spectra(('hyb', 'nmf'), (4,2), abs_sig=2., quick_and_dirty=True,
        include_Chl=False, use_log_ab=True)#, nchains=300)