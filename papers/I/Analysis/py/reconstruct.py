""" Methods for paper specfici reconstruction """

import numpy as np
import torch

from ihop.iops.decompose import reconstruct_nmf
from ihop.iops.decompose import reconstruct_pca

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