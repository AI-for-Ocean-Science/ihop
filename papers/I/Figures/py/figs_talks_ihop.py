""" Figures for Paper I on IHOP """

# imports
import os
import sys
from importlib import resources

import numpy as np

import torch
import corner
import xarray

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

from oceancolor.hydrolight import loisel23
from oceancolor.utils import plotting 
from oceancolor.water import absorption
from oceancolor.water import scattering
from oceancolor.pace import io as pace_io

from ihop.iops.decompose import reconstruct_nmf
from ihop.iops.decompose import reconstruct_pca

from ihop import io as ihop_io
from ihop.iops import decompose 
from ihop.iops import io as iops_io
from ihop.emulators import io as emu_io
from ihop.inference import io as inf_io
from ihop.inference import analysis as inf_analysis
from ihop.training_sets import load_rs
from ihop.inference import io as fitting_io

from cnmf import stats as cnmf_stats


mpl.rcParams['font.family'] = 'stixgeneral'

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import reconstruct
import ls2 as anly_ls2
import calc_stats

# Fits
sys.path.append(os.path.abspath("../../../builds/fits/py"))
import fits

from IPython import embed

# Number of components
#Ncomp = (4,3)
Ncomps = (4,2)


clbls = [r'$H_'+f'{ii+2}'+r'^{a}$' for ii in range(Ncomps[0])]
clbls += [r'$H_'+f'{ii+2}'+r'^{bb}$' for ii in range(Ncomps[1])]
clbls += ['Chl']


def fig_basis_all_a(outfile='fig_ihop_talk_basis_all_a.png', 
                    decomps:tuple=('nmf', 'nmf')):

    X, Y = 4, 0

    # Seaborn
    sns.set(style="whitegrid",
            rc={"lines.linewidth": 2.5})
            # 'axes.edgecolor': 'black'
    sns.set_palette("pastel")
    sns.set_context("paper")
    #sns.set_context("poster", linewidth=3)
    #sns.set_palette("husl")

    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1,3)


    for ss, m in enumerate([2,3,4]):
        # Load
        IOP = 'a'
        ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
            decomps, (m,2))
        wave = d_a['wave']
        d = d_a


        ax = plt.subplot(gs[ss])

        #d = d_a if IOP == 'a' else d_bb
        #if IOP == 'a':
        #    iop_w = absorption.a_water(wave, data='IOCCG')
        #else:
        #    iop_w = d_train['bb_w']
        #iop_w /= np.sum(iop_w)

        # Plot water first
        #sns.lineplot(x=wave, y=iop_w,
        #                    label=r'$W_'+f'{1}'+r'^{\rm '+IOP+r'}$',
        #                    ax=ax, lw=2)#, drawstyle='steps-pre')

        # Now the rest
        M = d['M']
        # Variance
        if decomps[0] == 'nmf':
            evar_i = cnmf_stats.evar_computation(
                d['spec'], d['coeff'], d['M'])
        else:
            evar_i = np.sum(d['explained_variance'])
        # Plot
        for ii in range(m):
            nrm = 1.
            # Step plot
            sns.lineplot(x=wave, y=M[ii]/nrm, 
                            label=r'$W_'+f'{ii+1}'+r'^{\rm '+IOP+r'}$',
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
        ax.text(0.5, 0.5, f'{100*evar_i:.2f}%',
            transform=ax.transAxes,
            fontsize=13, ha='center')

        loc = 'upper right' #if ss == 1 else 'upper left'
        ax.legend(fontsize=15, loc=loc)

        plotting.set_fontsize(ax, 16)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Decomposition, m=2,3,4
    if flg & (2**0):
        #fig_emulator_rmse('L23_PCA')
        #fig_basis_functions(('nmf', 'nmf'), in_Ncomps=(2,2),
        #                    outfile='fig_basis_functions_nmf_22.png')
        fig_basis_all_a()
        #fig_basis_functions(('pca', 'pca'),
        #                    outfile='fig_basis_functions_pca.png')
        #fig_basis_functions(('npca', 'npca'), in_Ncomps=(4,2),
        #                    outfile='fig_basis_functions_npca.png')


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg += 2 ** 0  # Basis functions of the decomposition, m=2,3,4
        
    else:
        flg = sys.argv[1]

    main(flg)