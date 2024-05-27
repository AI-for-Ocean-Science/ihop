""" Figs for Gordon Analyses """
import os, sys

import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

from oceancolor.utils import plotting 
from oceancolor.water import absorption

from ihop.inference import io as inf_io

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import gordon



# ############################################################
def fig_mcmc_fit(model:str, idx:int=170, chain_file=None,
                 outroot='fig_mcmc_fit'): 

    if chain_file is None:
        chain_file = f'../Analysis/Fits/FGordon_{model}_170.npz'
    d_chains = inf_io.load_chains(chain_file)

    # Load the data
    wave, Rrs, varRrs = gordon.prep_data(idx)

    # Reconstruct
    a, b, siga, sigb, model_Rrs, sigRs = gordon.reconstruct(
        model, d_chains['chains']) 

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')

    # Outfile
    outfile = outroot + f'_{idx}.png'

    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    

    # #########################################################
    # a with water

    ax_aw = plt.subplot(gs[0])
    ax_aw.plot(wave, orig+a_w, 'ko', label='True', zorder=1)
    ax_aw.plot(wave, a_nmf+a_w, 'r:', label='Real Recon')
    ax_aw.fill_between(wave, a_w+a_mean-a_std, a_w+a_mean+a_std, 
            color='r', alpha=0.5, label='Uncertainty') 
    #ax_a.set_xlabel('Wavelength (nm)')
    ax_aw.set_ylabel(r'$a(\lambda) \; [{\rm m}^{-1}]$')
    #else:
    #    ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    ax_a.legend(fontsize=lgsz)
    #ax_a.tick_params(labelbottom=False)  # Hide x-axis labels

    if wvmnx is not None:
        ax_a.set_xlim(wvmnx[0], wvmnx[1])

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
    if water:
        bb_w = d_train['bb_w']
    else:
        bb_w = 0
    ax_bb = plt.subplot(gs[3])
    ax_bb.plot(wave, bb_w+orig_bb, 'ko', label='True')
    ax_bb.plot(wave, bb_w+bb_mean, 'g-', label='Retrieval')
    if show_NMF:
        ax_bb.plot(wave, bb_w+bb_nmf, 'g:', label='True NMF')
    ax_bb.fill_between(wave, bb_w+bb_mean-bb_std, bb_w+bb_mean+bb_std, 
            color='g', alpha=0.5, label='Uncertainty') 

    #ax_bb.set_xlabel('Wavelength (nm)')
    ax_bb.set_ylabel(r'$b_b(\lambda) \; [{\rm m}^{-1}]$')

    ax_bb.text(xpos, ypos2,  '(c)', color='k',
            transform=ax_bb.transAxes,
              fontsize=18, ha='left')
    ax_bb.legend(fontsize=lgsz)
    ax_bb.set_ylim(bottom=0., top=None)

    # #########################################################
    # Rs
    ax_R = plt.subplot(gs[0:2])
    ax_R.plot(wave, Rs[idx], 'kx', label='True')
    if true_only:
        pass
    elif use_quick:
        ax_R.plot(wave, obs_Rs[0], 'bs', label='"Observed"')
    else:
        ax_R.plot(wave, obs_Rs[in_idx], 'bs', label='"Observed"')
    if (not true_only) and (not true_obs_only):
        ax_R.plot(wave, pred_Rs, 'r-', label='Fit', zorder=10)
        ax_R.fill_between(wave, pred_Rs-std_pred, pred_Rs+std_pred, 
            color='r', alpha=0.5, zorder=10) 
        ax_R.fill_between(wave,
            pred_Rs-std_pred, pred_Rs+std_pred,
            color='r', alpha=0.5) 

    #ax_R.set_xlabel('Wavelength (nm)')
    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')
    #ax_R.tick_params(labelbottom=False)  # Hide x-axis labels

    ax_R.text(xpos, ypos, '(a)', color='k',
            transform=ax_R.transAxes,
              fontsize=18, ha='right')

    #ax_R.set_yscale('log')
    ax_R.legend(fontsize=lgsz)
    
    # axes
    for ss, ax in enumerate([ax_a, ax_R, ax_bb]):
        plotting.set_fontsize(ax, 14)
        if ss != 1:
            ax.set_xlabel('Wavelength (nm)')

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Decomposition
    if flg & (2**0):
        fig_mcmc_fit('Indiv')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg += 2 ** 0  # Basis functions of the decomposition
        #flg += 2 ** 1  # RMSE of emulators
        
    else:
        flg = sys.argv[1]

    main(flg)