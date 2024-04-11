""" Figures for IHOP talks """

import os
import sys

import numpy as np
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit

import seaborn as sns
import pandas

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy
mpl.rcParams['font.family'] = 'stixgeneral'

import corner

from oceancolor.utils import plotting 
from oceancolor.utils import cat_utils
from oceancolor.iop import cdom
from oceancolor.ph import pigments
from oceancolor.hydrolight import loisel23
from oceancolor.tara import io as tara_io
from oceancolor.water import absorption

from ihop import io as ihop_io
from ihop.iops import decompose 
from ihop.emulators import io as emu_io
from ihop.inference import io as inf_io
from ihop.training_sets import load_rs

from cnmf import io as cnmf_io
from cnmf import stats as cnmf_stats

from IPython import embed

# Number of components
#Ncomp = (4,3)
Ncomp = (4,2)

# Local
sys.path.append(os.path.abspath("../papers/I/Analysis/py"))
import reconstruct

# #############################################
def fig_abR(outfile='fig_abR.png',
    nmf_fit:str='L23', N_NMF:int=4, iop:str='a',
    norm:bool=True):

    # Load
    X,Y = 4,0
    ds = loisel23.load_ds(X, Y)
    da_l23 = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    db_l23 = cnmf_io.load_nmf(nmf_fit, N_NMF, 'bb')
    L23_wave = ds.Lambda.data

    a_w = absorption.a_water(da_l23['wave'])

    # bb water
    bb_w = ds.bb.data[0,:] - ds.bbnw.data[0,:]
    f_bb = interp1d(L23_wave, bb_w)
    #embed(header='52 of talks')

    idxs = [0, 1000, 2000]

    # #########################################################
    # Figure
    figsize=(10,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # #########################################################
    # Absorption Spectra
    ax_a = plt.subplot(gs[0])
    ax_a.grid(True)

    # Water
    ax_a.plot(da_l23['wave'], a_w/10, ls=':', 
              label=r'$a_{\rm w}/10$', color='k')

    # Others
    for ss, idx in enumerate(idxs):
        nrm = 1.
        lbl = '' if ss > 0 else r'$a_{\rm nw}$'
        ax_a.plot(da_l23['wave'], da_l23['spec'][idx]/nrm, 
                 label=lbl, ls='-')
    # Label
    ax_a.set_ylabel(r'Absorption Coefficient (m$^{-1}$)')
    #ax_spec.set_yscale('log')
    # Legend at top center
    ax_a.legend(fontsize=13., loc='upper center')#, ncol=3)

    # #########################################################
    # Backscattering
    ax_b = plt.subplot(gs[1])

    # Water
    ax_b.plot(db_l23['wave'], f_bb(db_l23['wave']), 
              ls=':', label=r'$b_{b, \rm w}$', color='k')
    ax_b.legend(fontsize=13., loc='upper center')#, ncol=3)

    # Others
    for ss, idx in enumerate(idxs):
        nrm = 1.
        lbl = '' if ss > 0 else r'$b_{b, \rm nw}$'
        ax_b.plot(da_l23['wave'], db_l23['spec'][idx]/nrm, 
                 label=lbl, ls='-') 
    # Label
    ax_b.set_ylabel(r'Backscattering (m$^{-1}$)')
    #ax_spec.set_yscale('log')
    # Legend at top center
    ax_b.legend(fontsize=13., loc='upper center')#, ncol=3)


    # #########################################################
    # Reflectances
    ax_R = plt.subplot(gs[2:])


    for ss, idx in enumerate(idxs):
        ax_R.plot(da_l23['wave'], da_l23['Rs'][idx], 'o', ms=2)
    ax_R.set_ylabel(r'Reflectances (sr$^{-1}$)')

    # Axes
    for ax in [ax_a, ax_b, ax_R]:
        plotting.set_fontsize(ax, 15.)
        ax.set_xlabel('Wavelength (nm)')
        ax.grid(True)

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_mcmc_fit(outroot='fig_mcmc_fit', decomp:str='nmf',
        hidden_list:list=[512, 512, 512, 256], dataset:str='L23', use_quick:bool=False,
        X:int=4, Y:int=0, show_zoom:bool=False, 
        perc:int=None, abs_sig:float=None,
        wvmnx:tuple=None, show_NMF:bool=False,
        water:bool=False, in_idx:int=0,
        test:bool=False):

    # Load
    edict = emu_io.set_emulator_dict(dataset, decomp, Ncomp, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)

    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    chain_file = inf_io.l23_chains_filename(edict, 
                                            perc if perc is not None else int(abs_sig), 
                                            test=test)
    d_chains = inf_io.load_chains(chain_file)

    # Reconstruct
    items = reconstruct.one_spectrum(in_idx, ab, Chl, d_chains, 
                                     d_a, d_bb, 
                                     emulator, decomp, Ncomp)
    idx, orig, a_mean, a_std, a_iop, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, allY, wave,\
        orig_bb, bb_mean, bb_std, a_nmf, bb_nmf = items
    print(f"L23 index = {idx}")

    # Outfile
    outfile = outroot + f'_{idx}.png'
    if water:
        outfile = outfile.replace('.png', '_water.png')
        # Load training data
        d_train = load_rs.loisel23_rs(X=X, Y=Y)


    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=(9,8))
    plt.clf()
    gs = gridspec.GridSpec(2,1)
    
    xpos, ypos, ypos2 = 0.05, 0.10, 0.10

    # #########################################################
    # a
    if water:
        a_w = cross.a_water(wave, data='IOCCG')
    else:
        a_w = 0
    ax_a = plt.subplot(gs[1])
    def plot_spec(ax):
        ax.plot(wave, orig+a_w, 'ko', label='True')
        ax.plot(wave, a_mean+a_w, 'r-', label='Retrieval')
        if show_NMF:
            ax.plot(wave, a_nmf+a_w, 'r:', label='Real Recon')
        ax.fill_between(wave, a_w+a_mean-a_std, a_w+a_mean+a_std, 
            color='r', alpha=0.5, label='Uncertainty') 
    plot_spec(ax_a)
    #ax_a.set_xlabel('Wavelength (nm)')
    if water:
        ax_a.set_ylabel(r'$a(\lambda) \; [{\rm m}^{-1}]$')
    else:
        ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    #ax_a.text(xpos, ypos2,  '(b)', color='k',
    #        transform=ax_a.transAxes,
    #          fontsize=18, ha='left')

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

    '''
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
    '''

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
    ax_R.set_ylabel(r'$R_{rs}(\lambda) \; [10^{-4} \, {\rm sr}^{-1}$]')
    #ax_R.tick_params(labelbottom=False)  # Hide x-axis labels

    #ax_R.text(xpos, ypos, '(a)', color='k',
    #        transform=ax_R.transAxes,
    #          fontsize=18, ha='right')

    #ax_R.set_yscale('log')
    ax_R.legend(fontsize=lgsz)
    
    # axes
    for ss, ax in enumerate([ax_a, ax_R]):
        plotting.set_fontsize(ax, 16)
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

    # Example spectra
    if flg & (2**0):
        fig_abR()

    # MCMC fit example;  R, a only
    if flg & (2**1):
        #fig_mcmc_fit(test=True, perc=10)
        fig_mcmc_fit(test=True, abs_sig=2.)
        #fig_mcmc_fit(test=True, abs_sig=2., water=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Example spectra
        flg += 2 ** 1  # 2 -- MCMC fit; R, a only
    else:
        flg = sys.argv[1]

    main(flg)