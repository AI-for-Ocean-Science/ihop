""" Figures for Paper I on IHOP """

# imports
import os
import sys
from importlib import resources

import numpy as np

import torch
import corner

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

from oceancolor.hydrolight import loisel23
from oceancolor.utils import plotting 
from oceancolor.iop import cross

from ihop import io as ihop_io
from ihop.iops import decompose 
from ihop.emulators import io as emu_io
from ihop.inference import io as inf_io
from ihop.training_sets import load_rs

from cnmf import stats as cnmf_stats


mpl.rcParams['font.family'] = 'stixgeneral'

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import reconstruct

from IPython import embed

# Number of components
Ncomp = 4



def fig_basis_functions(decomp:str,
                        outfile:str='fig_basis_functions.png', 
                        norm:bool=False):

    X, Y = 4, 0

    # Load
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)
    wave = d_a['wave']

    # Load training data
    d_train = load_rs.loisel23_rs(X=X, Y=Y)

    # Seaborn
    sns.set(style="whitegrid",
            rc={"lines.linewidth": 2.5})
            # 'axes.edgecolor': 'black'
    sns.set_palette("pastel")
    sns.set_context("paper")
    #sns.set_context("poster", linewidth=3)
    #sns.set_palette("husl")

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)


    for ss, IOP in enumerate(['a', 'bb']):
        ax = plt.subplot(gs[ss])

        d = d_a if IOP == 'a' else d_bb
        if IOP == 'a':
            iop_w = cross.a_water(wave, data='IOCCG')
        else:
            iop_w = d_train['bb_w']
        iop_w /= np.sum(iop_w)

        # Plot water first
        sns.lineplot(x=wave, y=iop_w,
                            label=r'$W_'+f'{1}'+r'^{\rm '+IOP+r'}$',
                            ax=ax, lw=2)#, drawstyle='steps-pre')

        # Now the rest
        M = d['M']
        # Variance
        evar_i = cnmf_stats.evar_computation(
            d['spec'], d['coeff'], d['M'])
        # Plot
        for ii in range(Ncomp):
            # Normalize
            if norm:
                iwv = np.argmin(np.abs(wave-440.))
                nrm = M[ii][iwv]
            else:
                nrm = 1.
            # Step plot
            sns.lineplot(x=wave, y=M[ii]/nrm, 
                            label=r'$W_'+f'{ii+2}'+r'^{\rm '+IOP+r'}$',
                            ax=ax, lw=2)#, drawstyle='steps-pre')
            #ax.step(wave, M[ii]/nrm, label=f'{itype}:'+r'  $\xi_'+f'{ii+1}'+'$')

        # Thick line around the border of the axis
        ax.spines['top'].set_linewidth(2)
        
        # Horizontal line at 0
        ax.axhline(0., color='k', ls='--')

        # Labels
        ax.set_xlabel('Wavelength (nm)')
        #ax.set_xlim(400., 720.)

        lIOP = r'$a(\lambda)$' if IOP == 'a' else r'$b_b(\lambda)$'
        ax.set_ylabel(f'NMF Basis Functions for {lIOP}')

        # Variance
        ax.text(0.5, 0.5, f'Explained variance: {100*evar_i:.2f}%',
            transform=ax.transAxes,
            fontsize=13, ha='center')

        loc = 'upper right' if ss == 1 else 'upper left'
        ax.legend(fontsize=15, loc=loc)

        plotting.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_emulator_rmse(dataset:str, Ncomp:int, hidden_list:list,
                      outfile:str='fig_emulator_rmse.png',
                      log_rrmse:bool=False,
                      X:int=4, Y:int=0, decomp:str='nmf'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edict = emu_io.set_emulator_dict(dataset, decomp, Ncomp, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)

    # Init the Plot
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(4,1)
    ax_rel = plt.subplot(gs[2])
    ax_bias = plt.subplot(gs[3])
    ax_abs = plt.subplot(gs[0:2])

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)
    emulator, e_file = emu_io.load_emulator_from_dict(edict)
    print(f"Using: {e_file} for the emulator")
    wave = d_a['wave']

    clrs = ['k', 'b', 'r', 'g']
    for ss, dset in enumerate(['Training', 'Validation']):
        if ss > 0:
            continue
        clr = clrs[ss]

        # Concatenate
        inputs = np.concatenate((ab, Chl.reshape(Chl.size,1)), axis=1)
        targets = Rs

        # Predict and compare
        dev = np.zeros_like(targets)
        for ss in range(targets.shape[0]):
            dev[ss,:] = targets[ss] - emulator.prediction(inputs[ss],
                                                    device)
        
        # Bias?
        bias = np.mean(dev, axis=0)
        
        # RMSE
        rmse = np.sqrt(np.mean(dev**2, axis=0))

        # Mean Rs
        mean_Rs = np.mean(targets, axis=0)

        # #####################################################
        # Absolute

        ax_abs.plot(wave, rmse, 'o', color=clr, label=f'{dset}')

        ax_abs.set_ylabel(r'RMSE  (10$^{-4} \, \rm sr^{-1}$)')
        ax_abs.tick_params(labelbottom=False)  # Hide x-axis labels

        ax_abs.legend(fontsize=15)

        #ax.set_xlim(1., 10)
        #ax.set_ylim(1e-5, 0.01)
        #ax.set_yscale('log')
        #ax.legend(fontsize=15)

        #ax_abs.text(0.95, 0.90, model, color='k',
        #    transform=ax_abs.transAxes,
        #    fontsize=22, ha='right')

        # #####################################################
        # Relative
        rRMSE = rmse/mean_Rs
        ax_rel.plot(wave, rRMSE, 'o', color=clr)
        ax_rel.set_ylabel('rRMSE')
        #ax_rel.set_ylim(0., rRMSE.max()*1.05)

        ax_rel.tick_params(labelbottom=False)  # Hide x-axis labels

        if log_rrmse:
            ax_rel.set_yscale('log')
        else:
            ax_rel.set_ylim(0., 0.10)

        # #####################################################
        # Bias
        ax_bias.plot(wave, bias/mean_Rs, 'o', color=clr)
        ax_bias.set_ylabel('Relative Bias')
        ax_bias.set_ylim(-0.02, 0.02)


    # Finish
    for ss, ax in enumerate([ax_abs, ax_rel, ax_bias]):
        plotting.set_fontsize(ax, 15)
        if ss == 2:
            ax.set_xlabel('Wavelength [nm]')
        # Grid
        ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    # A few stats on Chl
    min_chl = np.min(Chl)
    max_chl = np.max(Chl)
    print(f'Min Chl: {np.log10(min_chl):.2f}, Max Chl: {np.log10(max_chl):.2f}')


# ############################################################
def fig_mcmc_fit(outfile='fig_mcmc_fit.png', decomp:str='nmf',
        hidden_list:list=[512, 512, 512, 256], dataset:str='L23', use_quick:bool=False,
        X:int=4, Y:int=0, show_zoom:bool=False, perc:int=10,
        test:bool=False):

    in_idx = 0
    Ncomp = 4
    # Load
    edict = emu_io.set_emulator_dict(dataset, decomp, Ncomp, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)

    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    chain_file = inf_io.l23_chains_filename(edict, perc, test=test)
    d_chains = inf_io.load_chains(chain_file)

    # Reconstruct
    items = reconstruct.one_spectrum(in_idx, ab, Chl, d_chains, 
                                     d_a, d_bb, 
                                     emulator, decomp, Ncomp)
    idx, orig, a_mean, a_std, a_iop, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, allY, wave,\
        orig_bb, bb_mean, bb_std = items
    print(f"L23 index = {idx}")

    # #########################################################
    # Plot the solution
    lgsz = 18.

    fig = plt.figure(figsize=(10,12))
    plt.clf()
    gs = gridspec.GridSpec(3,1)
    
    xpos, ypos, ypos2 = 0.95, 0.15, 0.95

    # #########################################################
    # a
    ax_a = plt.subplot(gs[1])
    def plot_spec(ax):
        ax.plot(wave, orig, 'ko', label='True')
        ax.plot(wave, a_mean, 'r-', label='Retrieval')
        #ax.plot(wave, a_iop, 'k:', label='PCA')
        ax.fill_between(wave, a_mean-a_std, a_mean+a_std, 
            color='r', alpha=0.5, label='Uncertainty') 
    plot_spec(ax_a)
    #ax_a.set_xlabel('Wavelength (nm)')
    ax_a.set_ylabel(r'$a(\lambda) \; [{\rm m}^{-1}]$')

    ax_a.text(xpos, ypos2,  '(b)', color='k',
            transform=ax_a.transAxes,
              fontsize=18, ha='left')

    ax_a.legend(fontsize=lgsz)
    ax_a.tick_params(labelbottom=False)  # Hide x-axis labels
    ax_a.tick_params(labelbottom=False)  # Hide x-axis labels

    # Zoom in
    # inset axes....
    if show_zoom:
        asz = 0.4
        ax_zoom = ax_a.inset_axes(
            [0.15, 0.5, asz, asz])
        plot_spec(ax_zoom)
        ax_zoom.set_xlim(340.,550)
        ax_zoom.set_ylim(0., 0.25)
        #ax_zoom.set_ylim(340.,550)
        ax_zoom.set_xlabel('Wavelength (nm)')
        ax_zoom.set_ylabel(r'$a(\lambda)$')

    # #########################################################
    # b
    ax_bb = plt.subplot(gs[2])
    ax_bb.plot(wave, orig_bb, 'ko', label='True')
    ax_bb.plot(wave, bb_mean, 'g-', label='Retrieval')
    ax_bb.fill_between(wave, bb_mean-bb_std, bb_mean+bb_std, 
            color='g', alpha=0.5, label='Uncertainty') 

    ax_bb.set_xlabel('Wavelength (nm)')
    ax_bb.set_ylabel(r'$b_b(\lambda) \; [{\rm m}^{-1}]$')

    ax_bb.text(xpos, ypos2,  '(c)', color='k',
            transform=ax_bb.transAxes,
              fontsize=18, ha='right')
    ax_bb.legend(fontsize=lgsz)

    # #########################################################
    # Rs
    ax_R = plt.subplot(gs[0])
    ax_R.plot(wave, Rs[idx], 'kx', label='True')
    if use_quick:
        ax_R.plot(wave, obs_Rs[0], 'bs', label='"Observed"')
    else:
        ax_R.plot(wave, obs_Rs[in_idx], 'bs', label='"Observed"')
    ax_R.plot(wave, pred_Rs, 'r-', label='Fit', zorder=10)
    ax_R.fill_between(wave, pred_Rs-std_pred, pred_Rs+std_pred, 
            color='r', alpha=0.5, zorder=10) 
    #ax_R.plot(wave, NN_Rs, 'g-', label='NN+True')

    ax_R.fill_between(wave,
        pred_Rs-std_pred, pred_Rs+std_pred,
        color='r', alpha=0.5) 

    #ax_R.set_xlabel('Wavelength (nm)')
    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [{\rm sr}^{-1}$]')
    ax_R.tick_params(labelbottom=False)  # Hide x-axis labels

    ax_R.text(xpos, ypos, '(a)', color='k',
            transform=ax_R.transAxes,
              fontsize=18, ha='right')

    ax_R.legend(fontsize=lgsz)
    ax_R.tick_params(labelbottom=False)  # Hide x-axis labels
    
    # axes
    for ax in [ax_a, ax_R, ax_bb]:
        plotting.set_fontsize(ax, 15)

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_corner(outfile='fig_corner.png', decomp:str='nmf',
        hidden_list:list=[512, 512, 512, 256], dataset:str='L23', 
        chop_burn:int=-3000, perc:int=10,
        X:int=4, Y:int=0,
        test:bool=False):

    in_idx = 0
    Ncomp = 4
    # Load
    edict = emu_io.set_emulator_dict(dataset, decomp, Ncomp, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)

    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    chain_file = inf_io.l23_chains_filename(edict, perc, test=test)
    d_chains = inf_io.load_chains(chain_file)

    chains = d_chains['chains'][in_idx]
    coeff = chains[chop_burn:, :, :].reshape(-1,2*Ncomp+1)

    #print(f"L23 index = {idx}")
    # Labels
    lbls = [r'$H_'+f'{ii+2}'+r'^{a}$' for ii in range(Ncomp)]
    lbls += [r'$H_'+f'{ii+2}'+r'^{bb}$' for ii in range(Ncomp)]
    lbls += ['Chl']

    fig = corner.corner(
        coeff, labels=lbls,
        label_kwargs={'fontsize':17},
        color='blue',
        #axes_scale='log',
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
# ############################################################
def fig_rmse_vs_sig(outroot:str='fig_rmse_vs_sig',
                    decomp:str='pca', chop_burn:int=-3000):

    # Prep
    if decomp == 'pca':
        rfunc = decompose.reconstruct_pca
    elif decomp == 'nmf':
        rfunc = decompose.reconstruct_nmf

    outfile = outroot + f'_{decomp}.png'

    all_l23_rmse = []
    all_l23_sig = []
    #all_perc = [0, 5, 10, 15, 20]
    all_perc = [0, 5, 10, 20]
    for perc in all_perc:
        print(f"Working on: {perc}%")
        # L23
        ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)
        d_chains = ihop_io.load_l23_chains(decomp, perc=perc)
        chains = d_chains['chains']
        l23_idx = d_chains['idx']
        ncomp = 3
        wave = d_a['wave']
        dev = np.zeros((chains.shape[0], wave.size))
        mcmc_std = np.zeros((chains.shape[0], wave.size))
        #embed(header='242 of figs')
        for in_idx in range(chains.shape[0]):
            idx = l23_idx[in_idx]
            ichains = chains[in_idx]
            Y = ichains[chop_burn:, :, 0:ncomp].reshape(-1,ncomp)
            orig, a_recon = rfunc(Y, d_a, idx)
            a_mean = np.mean(a_recon, axis=0)
            #
            mcmc_std[in_idx,:] = np.std(a_recon, axis=0)
            dev[in_idx,:] = orig - a_mean
            #_, a_pca = rfunc(ab[idx][:ncomp], d_a, idx)
        # RMSE
        rmse_l23 = np.sqrt(np.mean(dev**2, axis=0))
        # Save
        all_l23_rmse.append(rmse_l23)
        all_l23_sig.append(np.mean(mcmc_std, axis=0))
        

    # Plot
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # #####################################################
    # Absolute
    #ax_abs = plt.subplot(gs[0:2])
    all_ax = []
    for ss, rmse_l23 in enumerate(all_l23_rmse):
        ax= plt.subplot(gs[ss])

        ax.plot(wave, rmse_l23, 'o', label=f'{all_perc[ss]}%: RMSE')
        ax.plot(wave, all_l23_sig[ss], '*', label=f'{all_perc[ss]}%: MCMC std')

        ax.set_ylabel(r'RMSE in $a(\lambda)$ (m$^{-1}$)')
    #ax_abs.tick_params(labelbottom=False)  # Hide x-axis labels
        all_ax.append(ax)
        ax.legend(fontsize=10)


    #ax.set_xlim(1., 10)
    #ax.set_ylim(1e-5, 0.01)
    #ax.set_yscale('log')

    # Finish
    for ss, ax in enumerate(all_ax):
        plotting.set_fontsize(ax, 13)
        ax.set_xlabel('Wavelength [nm]')
        # Grid
        ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Decomposition
    if flg & (2**0):
        #fig_emulator_rmse('L23_PCA')
        fig_basis_functions('nmf')

    # Example spectra
    if flg & (2**20):
        #fig_emulator_rmse('L23', 3, [512, 512, 512, 256],
        #                  outfile='fig_emulator_rmse_3.png')
        fig_emulator_rmse('L23', 4, [512, 512, 512, 256],
                          log_rrmse=True)
        #fig_emulator_rmse(['L23_NMF', 'L23_PCA'], [3, 3])

    # L23 IHOP performance vs. perc error
    if flg & (2**21):
        fig_mcmc_fit(test=True)

    # L23 IHOP performance vs. perc error
    if flg & (2**22):
        #fig_rmse_vs_sig()
        fig_rmse_vs_sig(decomp='nmf')

    # L23 IHOP performance vs. perc error
    if flg & (2**23):
        fig_corner(test=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg += 2 ** 0  # Basis functions of the decomposition
        #flg += 2 ** 20  # RMSE of emulators
        #flg += 2 ** 21  # Single MCMC fit (example)
        #flg += 2 ** 22  # RMSE of L23 fits
        flg += 2 ** 23  # corner

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
        
    else:
        flg = sys.argv[1]

    main(flg)