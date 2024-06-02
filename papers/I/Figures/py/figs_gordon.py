""" Figs for Gordon Analyses """
import os, sys

import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

import corner

from oceancolor.utils import plotting 
from oceancolor.water import absorption

from ihop.inference import io as inf_io
from ihop.inference import fgordon

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import gordon


from IPython import embed

# ############################################################
def fig_mcmc_fit(model:str, idx:int=170, chain_file=None,
                 outroot='fig_gordon_fit', show_bbnw:bool=False,
                 set_abblim:bool=True, scl_noise:float=None): 

    if chain_file is None:
        noises = '' if scl_noise is None else f'_n{int(100*scl_noise):02d}'
        chain_file = f'../Analysis/Fits/FGordon_{model}_170{noises}.npz'
    d_chains = inf_io.load_chains(chain_file)

    # Load the data
    odict = gordon.prep_data(idx)
    wave = odict['wave']
    Rrs = odict['Rrs']
    varRrs = odict['varRrs']
    a_true = odict['a']
    bb_true = odict['bb']
    aw = odict['aw']
    bbw = odict['bbw']
    bbnw = bb_true - bbw
    wave_true = odict['true_wave']
    Rrs_true = odict['true_Rrs']

    gordon_Rrs = fgordon.calc_Rrs(odict['a'][::2], odict['bb'][::2])

    # Interpolate
    aw_interp = np.interp(wave, wave_true, aw)
    bbw_interp = np.interp(wave, wave_true, bbw)

    # Reconstruc
    pdict = fgordon.init_mcmc(model, d_chains['chains'].shape[-1], 
                              wave, Y=odict['Y'], Chl=odict['Chl'])
    a_mean, bb_mean, a_5, a_95, bb_5, bb_95,\
        model_Rrs, sigRs = gordon.reconstruct(
        model, d_chains['chains'], pdict) 

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')

    # Outfile
    outfile = outroot + f'_{model}_{idx}.png'

    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    

    # #########################################################
    # a with water

    ax_aw = plt.subplot(gs[0])
    ax_aw.plot(wave_true, a_true, 'ko', label='True', zorder=1)
    ax_aw.plot(wave, a_mean, 'r-', label='Retreival')
    ax_aw.fill_between(wave, a_5, a_95,
            color='r', alpha=0.5, label='Uncertainty') 
    #ax_a.set_xlabel('Wavelength (nm)')
    ax_aw.set_ylabel(r'$a(\lambda) \; [{\rm m}^{-1}]$')
    #else:
    #    ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    ax_aw.legend(fontsize=lgsz)
    ax_aw.set_ylim(bottom=0., top=2*a_true.max())
    #ax_a.tick_params(labelbottom=False)  # Hide x-axis labels


    # #########################################################
    # a without water

    ax_anw = plt.subplot(gs[1])
    ax_anw.plot(wave_true, a_true-aw, 'ko', label='True', zorder=1)
    ax_anw.plot(wave, a_mean-aw_interp, 'r-', label='Retreival')
    ax_anw.fill_between(wave, a_5-aw_interp, a_95-aw_interp, 
            color='r', alpha=0.5, label='Uncertainty') 
    
    ax_anw.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')
    #else:
    #    ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    #ax_anw.legend(fontsize=lgsz)
    if set_abblim:
        ax_anw.set_ylim(bottom=0., top=2*(a_true-aw).max())
    #ax_a.tick_params(labelbottom=False)  # Hide x-axis labels


    # #########################################################
    # b
    ax_bb = plt.subplot(gs[3])
    if show_bbnw:
        use_bbw = bbw[::2]
        show_bb = bbnw
    else:
        use_bbw = 0.
        show_bb = bbnw
    ax_bb.plot(wave_true, show_bb, 'ko', label='True')
    ax_bb.plot(wave, bb_mean-use_bbw, 'g-', label='Retrieval')
    ax_bb.fill_between(wave, bb_5-use_bbw, bb_95-use_bbw,
            color='g', alpha=0.5, label='Uncertainty') 

    #ax_bb.set_xlabel('Wavelength (nm)')
    if show_bbnw:
        ax_bb.set_ylabel(r'$b_bnw(\lambda) \; [{\rm m}^{-1}]$')
    else:
        ax_bb.set_ylabel(r'$b_b(\lambda) \; [{\rm m}^{-1}]$')

    ax_bb.legend(fontsize=lgsz)
    if set_abblim:
        ax_bb.set_ylim(bottom=0., top=2*bb_true.max())


    # #########################################################
    # Rs
    ax_R = plt.subplot(gs[2])
    ax_R.plot(wave_true, Rrs_true, 'kx', label='True L23')
    ax_R.plot(wave, gordon_Rrs, 'ko', label='L23 + Gordon')
    ax_R.plot(wave, model_Rrs, 'r-', label='Fit', zorder=10)
    ax_R.fill_between(wave, model_Rrs-sigRs, model_Rrs+sigRs, 
            color='r', alpha=0.5, zorder=10) 

    if scl_noise is not None:
        ax_R.plot(d_chains['wave'], d_chains['obs_Rrs'], 'bs', label='Observed')

    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')
    ax_R.set_ylim(bottom=0., top=1.1*Rrs_true.max())

    ax_R.legend(fontsize=lgsz)
    
    
    # axes
    for ss, ax in enumerate([ax_aw, ax_anw, ax_R, ax_bb]):
        plotting.set_fontsize(ax, 14)
        if ss > 1:
            ax.set_xlabel('Wavelength (nm)')

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_corner(model, outroot:str='fig_gordon_corner', idx:int=170,
        chain_file:str=None, scl_noise:float=None): 

    if chain_file is None:
        noises = '' if scl_noise is None else f'_n{int(100*scl_noise):02d}'
        chain_file = f'../Analysis/Fits/FGordon_{model}_170{noises}.npz'
    d_chains = inf_io.load_chains(chain_file)

    # Outfile
    outfile = outroot + f'_{model}_{idx}.png'

    burn = 7000
    thin = 1
    chains = d_chains['chains']
    coeff = 10**(chains[burn::thin, :, :].reshape(-1, chains.shape[-1]))

    if model == 'hybpow':
        clbls = ['H0', 'g', 'H1', 'H2', 'B1', 'b']
    elif model == 'hybnmf':
        clbls = ['H0', 'g', 'H1', 'H2', 'B1', 'B2']
    else:
        clbls = None

    fig = corner.corner(
        coeff, labels=clbls,
        label_kwargs={'fontsize':17},
        color='k',
        #axes_scale='log',
        #truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Indiv
    if flg == 1:
        fig_mcmc_fit('Indiv')

    # bbwater
    if flg == 2:
        fig_mcmc_fit('bbwater')

    # water
    if flg == 3:
        fig_mcmc_fit('water')

    # water
    if flg == 4:
        fig_mcmc_fit('bp')

    # exppow
    if flg == 5:
        fig_mcmc_fit('exppow')

    # GIOP
    if flg == 6:
        fig_mcmc_fit('giop', show_bbnw=True, set_abblim=False)

    # GIOP
    if flg == 7:
        fig_corner('giop')

    # GIOP+
    if flg == 8:
        fig_mcmc_fit('giop+', show_bbnw=True, set_abblim=False)

    # NMF aph + power-law bb
    if flg == 9:
        fig_mcmc_fit('hybpow', show_bbnw=True, set_abblim=False)

    # NMF aph + power-law bb + 7% noise
    if flg == 10:
        fig_mcmc_fit('hybpow', show_bbnw=True, set_abblim=False,
                     scl_noise=0.07)

    # HybPow without noise
    if flg == 11:
        fig_corner('hybpow')

    # HybPow with noise
    if flg == 12:
        fig_corner('hybpow', scl_noise=0.07)

    # NMF aph + power-law bb
    if flg == 13:
        fig_mcmc_fit('hybnmf', show_bbnw=True, set_abblim=False)

    # HybNMF
    if flg == 14:
        fig_corner('hybnmf')
        # B1, B2 are highly degenerate, no bueno

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)