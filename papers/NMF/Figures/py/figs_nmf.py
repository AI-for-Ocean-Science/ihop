""" Figuers for the NMF paper"""

import os
import xarray
from importlib import resources

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

import torch

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator 

import corner

from oceancolor.utils import plotting 
from oceancolor.iop import cdom


mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

import pandas


from IPython import embed

def load_nmf(nmf_fit:str, N_NMF:int=None):

    # Load
    if nmf_fit == 'l23':
        if N_NMF is None:
            N_NMF = 5
        path = os.path.join(resources.files('ihop'), 
                    'data', 'NMF')
        outroot = os.path.join(path, f'L23_NMF_{N_NMF}')
        nmf_file = outroot+'.npz'
        #
        d = np.load(nmf_file)
    else:
        raise IOError("Bad input")

    return d

def fig_nmf_rmse(outfile:str='fig_nmf_rmse.png',
                 nmf_fit:str='l23'):

    # RMSE
    rmss = []
    for n in range(1,10):
        # load
        d = load_nmf(nmf_fit, N_NMF=n+1)
        N_NMF = d['M'].shape[0]
        recon = np.dot(d['coeff'],
                       d['M'])
        #
        dev = recon - d['spec']
        rms = np.std(dev, axis=1)
        # Average
        avg_rms = np.mean(rms)
        rmss.append(avg_rms)

    # Plot

    fig = plt.figure(figsize=(6,6))
    plt.clf()
    ax = plt.gca()

    ax.plot(2+np.arange(N_NMF-1), rmss, 'o')

    ax.set_xlabel('Number of Components')
    ax.set_ylabel(r'Average RMSE (m$^{-1}$)')

    ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_nmf_indiv(outfile:str='fig_nmf_indiv.png',
                 nmf_fit:str='l23'):

    # load
    d = load_nmf(nmf_fit)
    M = d['M']
    wave = d['wave']

    fig = plt.figure(figsize=(7,5))
    gs = gridspec.GridSpec(2,2)

    all_ax = []
    for clr, ss in zip(['b', 'orange', 'g', 'r'], range(4)):
        ax = plt.subplot(gs[ss])

        ax.step(wave, M[ss], label=r'$\xi_'+f'{ss}'+'$',
                color=clr)
        #
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Basis vector')

        # Axis specific
        if ss == 0:
            pass
        elif ss == 1: # CDOM
            # Expoential
            a_cdom_exp = cdom.a_exp(wave)
            iwv = np.argmin(np.abs(wave-400.))
            a_cdom_exp *= M[ss][iwv] / a_cdom_exp[iwv]

            # Power law
            a_cdom_pow = cdom.a_pow(wave)
            a_cdom_pow *= M[ss][iwv] / a_cdom_pow[iwv]

            # Power law fit
            # TODO -- Fit!!!
            S_fit = -10.
            a_cdom_pow_fit = cdom.a_pow(wave, S=S_fit)
            a_cdom_pow_fit *= M[ss][iwv] / a_cdom_pow_fit[iwv]

            #
            ax.plot(wave, a_cdom_exp, color='gray', label='CDOM exp', ls='--')
            ax.plot(wave, a_cdom_pow, color='gray', label='CDOM pow', ls=':')
            ax.plot(wave, a_cdom_pow_fit, color='gray', 
                    label=f'CDOM pow: S={S_fit}', ls='-')

            ax.legend()
        else:
            pass

        # Save
        all_ax.append(ax)
    
    # Axes
    for ax in all_ax:
        plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_nmf_coeff(outfile:str='fig_nmf_coeff.png',
                 nmf_fit:str='l23'):

    # load
    d = load_nmf(nmf_fit)
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    fig = corner.corner(
        coeff[:,:4], labels=['a0', 'a1', 'a2', 'a3'],
        label_kwargs={'fontsize':17},
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

    # NMF RMSE
    if flg & (2**0):
        fig_nmf_rmse()

    # NMF basis
    if flg & (2**1):
        fig_nmf_basis()

    # Individual
    if flg & (2**2):
        fig_nmf_indiv()

    # Coeff
    if flg & (2**3):
        fig_nmf_coeff()



# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 1  # 1 -- 
    else:
        flg = sys.argv[1]

    main(flg)