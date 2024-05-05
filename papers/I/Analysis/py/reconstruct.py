""" Methods for paper specfici reconstruction """

import os
import numpy as np
import torch

from ihop.iops.decompose import reconstruct_nmf
from ihop.iops.decompose import reconstruct_pca
from ihop.iops.decompose import reconstruct_int

from ihop.emulators import io as emu_io
from ihop import io as ihop_io
from ihop.inference import io as inf_io
from ihop.inference import analysis as inf_analysis
from ihop.inference import io as fitting_io

from IPython import embed

def one_spectrum(in_idx:int, ab, Chl, d_chains, d_a, d_bb, emulator,
                             decomp:str, Ncomp:tuple,
                             chop_burn:int=-3000):
    chains = d_chains['chains'][in_idx]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    l23_idx = d_chains['idx']
    obs_Rs = d_chains['obs_Rs']

    idx = l23_idx[in_idx]
    print(f'Working on: L23 index={idx}')

    # Parse
    nwave = d_a['wave'].size
    wave = d_a['wave']

    # Prep
    if decomp == 'pca':
        rfunc = reconstruct_pca
    elif decomp == 'nmf':
        rfunc = reconstruct_nmf
    
    # a
    Y = chains[chop_burn:, :, 0:Ncomp[0]].reshape(-1,Ncomp[0])
    orig, a_recon = rfunc(Y, d_a, idx)
    _, a_nmf = rfunc(d_a['coeff'][idx], d_a, idx)
    a_mean = np.median(a_recon, axis=0)
    a_std = np.std(a_recon, axis=0)
    _, a_pca = rfunc(ab[idx][:Ncomp[0]], d_a, idx)

    # bb
    Y = chains[chop_burn:, :, Ncomp[0]:-1].reshape(-1,Ncomp[1])
    orig_bb, bb_recon = rfunc(Y, d_bb, idx)
    _, bb_nmf = rfunc(d_bb['coeff'][idx], d_bb, idx)
    bb_mean = np.median(bb_recon, axis=0)
    bb_std = np.std(bb_recon, axis=0)
    #_, a_pca = rfunc(ab[idx][:ncomp], d_a, idx)

    # Rs
    allY = chains[chop_burn:, :, :].reshape(-1,Ncomp[0]+Ncomp[1]+1) # Chl
    all_pred = np.zeros((allY.shape[0], nwave))
    for kk in range(allY.shape[0]):
        Ys = allY[kk]
        pred_Rs = emulator.prediction(Ys, device)
        all_pred[kk,:] = pred_Rs

    pred_Rs = np.median(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)
    NN_Rs = emulator.prediction(ab[idx].tolist() + [Chl[idx]], device)

    return idx, orig, a_mean, a_std, a_pca, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, allY, wave,\
        orig_bb, bb_mean, bb_std, a_nmf, bb_nmf

def all_spectra(decomps:tuple, Ncomps:tuple, 
                hidden_list:list=[512, 512, 512, 256], 
                dataset:str='L23', perc:int=None, 
                abs_sig:float=None, nchains:int=None,
                X:int=4, Y:int=0):

    d_keys = dict(pca='Y', nmf='coeff', int='new_spec')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = {}

    # Load
    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=True, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    wave = d_a['wave']

    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    outfile = os.path.basename(fitting_io.l23_chains_filename(
        edict, int(abs_sig)).replace('fit', 'recon'))

    # Chains
    chain_file = inf_io.l23_chains_filename(
        edict, perc if perc is not None else int(abs_sig)) 
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

    # ##############
    # Rrs
    fit_Rrs = inf_analysis.calc_Rrs(emulator, chains)
    outputs['fit_Rrs'] = fit_Rrs

    # Correct estimates
    items = [ab[i].tolist()+[Chl[i]] for i in chain_idx]
    corr_Rrs = []
    for item in items:
        iRs = emulator.prediction(item, device)
        corr_Rrs.append(iRs)
    corr_Rrs = np.array(corr_Rrs)

    outputs['corr_Rrs'] = corr_Rrs

    # ##############
    # a
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
    else:
        raise ValueError(f"Your decomps={decomps[0]} is not supported.")

    # Original reconstruction 
    if decomps[0] == 'int':
        _, a_recons = rfunc(d_a[d_keys[decomps[0]]], d_a, 0)
    else:
        a_recons = []
        for idx in chain_idx:
            _, a_recon = rfunc(d_a[d_keys[decomps[0]]][idx], d_a, idx)
            a_recons.append(a_recon)
    outputs['decomp_a'] = np.array(a_recons)

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

    # NMF
    all_spectra(('nmf', 'nmf'), (4,2), abs_sig=1.0,
                nchains=500)
    # PCA
    all_spectra(('pca', 'pca'), (4,2), abs_sig=1.0,
                nchains=500)
    # INT/NMF
    all_spectra(('int', 'nmf'), (40,2), abs_sig=1.0,
                nchains=25)