""" Figures for Paper I on IHOP """

# imports
import os
import sys
from importlib import resources

import numpy as np

import torch

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec


from oceancolor.hydrolight import loisel23
from oceancolor.utils import plotting 

from ihop import io as ihop_io
from ihop.iops import decompose 


mpl.rcParams['font.family'] = 'stixgeneral'

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import reconstruct

from IPython import embed


def fig_emulator_rmse(models:list, 
                      outfile:str='fig_emulator_rmse.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init the Plot
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(3,1)
    ax_rel = plt.subplot(gs[2])
    ax_abs = plt.subplot(gs[0:2])

    clrs = ['k', 'b', 'r', 'g']
    for ss, model in enumerate(models):
        clr = clrs[ss]
        if model[0:3] == 'L23':
            decomp = model[4:].lower()
            ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_data(decomp=decomp)
            Ncomp = ab.shape[1]//2
            emulator, e_file = ihop_io.load_l23_emulator(Ncomp, decomp=decomp)
            print(f"Using: {e_file} for the emulator")
            wave = d_a['wave']

        # Concatenate
        inputs = np.concatenate((ab, Chl.reshape(Chl.size,1)), axis=1)
        targets = Rs

        # Predict and compare
        dev = np.zeros_like(targets)
        for ss in range(targets.shape[0]):
            dev[ss,:] = targets[ss] - emulator.prediction(inputs[ss],
                                                    device)
        
        # RMSE
        rmse = np.sqrt(np.mean(dev**2, axis=0))

        # Mean Rs
        mean_Rs = np.mean(targets, axis=0)

        # #####################################################
        # Absolute

        ax_abs.plot(wave, rmse, 'o', color=clr, label=f'{model}')

        ax_abs.set_ylabel(r'Absolute RMSE (m$^{-1}$)')
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
        ax_rel.plot(wave, rmse/mean_Rs, '*', color=clr)
        ax_rel.set_ylabel('Relative RMSE')
        ax_rel.set_ylim(0., 0.023)

    # Finish
    for ss, ax in enumerate([ax_abs, ax_rel]):
        plotting.set_fontsize(ax, 17)
        if ss == 1:
            ax.set_xlabel('Wavelength [nm]')
        # Grid
        ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_mcmc_fit(outfile='fig_mcmc_fit.png', decomp='pca',
                 use_quick:bool=False,
                 show_zoom:bool=False):
    # Load
    in_idx = 0
    Ncomp = 3
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_data(decomp=decomp)
    d_chains = ihop_io.load_l23_chains(decomp, perc=10)
    emulator, e_file = ihop_io.load_l23_emulator(Ncomp, decomp=decomp)

    # Reconstruct
    items = reconstruct.one_spectrum(in_idx, ab, Chl, d_chains, d_a, d_bb, emulator, decomp)
    idx, orig, a_mean, a_std, a_iop, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, allY, wave,\
        orig_bb, bb_mean, bb_std = items

    # #########################################################
    # Plot the solution
    lgsz = 18.

    fig = plt.figure(figsize=(10,12))
    plt.clf()
    gs = gridspec.GridSpec(3,1)
    
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
    ax_a.set_xlabel('Wavelength (nm)')
    ax_a.set_ylabel(r'$a(\lambda)$')

    ax_a.text(0.05, 0.05, '(b)', color='k',
            transform=ax_a.transAxes,
              fontsize=18, ha='left')

    ax_a.legend(fontsize=lgsz)

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
    ax_bb.set_ylabel(r'$b_b(\lambda)$')

    ax_bb.text(0.05, 0.05, '(c)', color='k',
            transform=ax_bb.transAxes,
              fontsize=18, ha='left')
    ax_bb.legend(fontsize=lgsz)

    # #########################################################
    # Rs
    ax_R = plt.subplot(gs[0])
    ax_R.plot(wave, Rs[idx], 'kx', label='True')
    if use_quick:
        ax_R.plot(wave, obs_Rs[0], 'bs', label='Obs')
    else:
        ax_R.plot(wave, obs_Rs[in_idx], 'bs', label='Obs')
    ax_R.plot(wave, pred_Rs, 'r-', label='Model', zorder=10)
    ax_R.fill_between(wave, pred_Rs-std_pred, pred_Rs+std_pred, 
            color='r', alpha=0.5, zorder=10) 
    #ax_R.plot(wave, NN_Rs, 'g-', label='NN+True')

    ax_R.fill_between(wave,
        pred_Rs-std_pred, pred_Rs+std_pred,
        color='r', alpha=0.5) 

    ax_R.set_xlabel('Wavelength (nm)')
    ax_R.set_ylabel(r'$R_s$')

    ax_R.text(0.05, 0.05, '(a)', color='k',
            transform=ax_R.transAxes,
              fontsize=18, ha='left')

    ax_R.legend(fontsize=lgsz)
    
    # axes
    for ax in [ax_a, ax_R, ax_bb]:
        plotting.set_fontsize(ax, 15)

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
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
        ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_data(decomp=decomp)
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

    # Example spectra
    if flg & (2**20):
        #fig_emulator_rmse('L23_PCA')
        fig_emulator_rmse(['L23_NMF', 'L23_PCA'])

    # L23 IHOP performance vs. perc error
    if flg & (2**21):
        fig_mcmc_fit()#use_quick=True)

    # L23 IHOP performance vs. perc error
    if flg & (2**22):
        #fig_rmse_vs_sig()
        fig_rmse_vs_sig(decomp='nmf')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Example spectra
        #flg += 2 ** 1  # 2 -- L23: PCA vs NMF Explained variance
        #flg += 2 ** 2  # 4 -- L23: PCA and NMF basis
        #flg += 2 ** 3  # 8 -- L23: Fit NMF basis functions with CDOM, Chl
        #flg += 2 ** 4  # 16 -- L23+Tara; a1, a2 contours
        #flg += 2 ** 5  # 32 -- L23,Tara compare NMF basis functions
        #flg += 2 ** 6  # 64 -- Fit l23 basis functions

        #flg += 2 ** 20  # RMSE of emulators
        flg += 2 ** 21  # Single MCMC fit (example)
        #flg += 2 ** 22  # RMSE of L23 fits

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
        
    else:
        flg = sys.argv[1]

    main(flg)