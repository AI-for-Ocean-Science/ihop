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

import corner

from oceancolor.utils import plotting 
from oceancolor.iop import cdom
from oceancolor.ph import pigments


mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

import pandas


from IPython import embed



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

def fig_explained_variance(
    outfile:str='fig_explained_variance.png',
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

def fig_nmf_basis(outroot:str='fig_nmf_basis',
                 nmf_fit:str='l23', N_NMF:int=4):

    outfile = f'{outroot}_{N_NMF}.png'
    # RMSE
    rmss = []
    # load
    d = load_nmf(nmf_fit, N_NMF=N_NMF)
    M = d['M']
    wave = d['wave']

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    ax = plt.gca()

    # Plot
    for ss in range(N_NMF):
        ax.step(wave, M[ss], label=r'$\xi_'+f'{ss}'+'$')


    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Basis vector')

    ax.legend()

    #ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_fit_cdom(outfile:str='fig_fit_cdom.png',
                 nmf_fit:str='l23', N_NMF:int=4,
                 wv_max:float=600.):

    # load
    d = load_nmf(nmf_fit, N_NMF=N_NMF)
    M = d['M']
    wave = d['wave']


    # load
    d = load_nmf(nmf_fit, N_NMF=N_NMF)
    M = d['M']
    wave = d['wave']

    if N_NMF==5: 
        ss = 1
    elif N_NMF==4: 
        ss = 0
    else:
        raise IOError("Bad N_NMF")
    a_cdom = M[ss]

    wv_cut = wave < wv_max
    cut_wv = wave[wv_cut]

    # Fit exponentials
    exp_tot_coeff, cov = cdom.fit_exp_tot(wave[wv_cut], 
                                            a_cdom[wv_cut])
    a_cdom_totexp_fit = exp_tot_coeff[0] * cdom.a_exp(
        wave[wv_cut], S_CDOM=exp_tot_coeff[1],
        wave0=exp_tot_coeff[2])
    print(f'Tot exp coeff: {exp_tot_coeff}')
    exp_norm_coeff, cov = cdom.fit_exp_norm(wave[wv_cut], 
                                            a_cdom[wv_cut])
    a_cdom_exp_fit = exp_norm_coeff[0] * cdom.a_exp(wave[wv_cut])

    # Fit power-law
    pow_coeff, pow_cov = cdom.fit_pow(cut_wv, a_cdom[wv_cut])
    a_cdom_pow_fit = pow_coeff[0] * cdom.a_pow(cut_wv, S=pow_coeff[1])

    fig = plt.figure(figsize=(11,5))
    gs = gridspec.GridSpec(1,2)

    # #########################################################
    # Fits as normal
    ax_fits = plt.subplot(gs[0])

    # NMF
    ax_fits.step(wave, M[ss], label=r'$\xi_'+f'{ss}'+'$', color='k')

    ax_fits.plot(cut_wv, a_cdom_exp_fit, 
            color='b', label='CDOM exp', ls='-')
    ax_fits.plot(cut_wv, a_cdom_totexp_fit, 
            color='b', label='CDOM Tot exp', ls='--')
    ax_fits.plot(cut_wv, a_cdom_pow_fit, 
            color='r', label='CDOM pow', ls='-')

    ax_fits.axvline(wv_max, ls='--', color='gray')

    ax_fits.legend()

    # #########################################################
    # CDF
    cdf_NMF = np.cumsum(a_cdom[wv_cut])
    cdf_NMF /= cdf_NMF[-1]
    
    cdf_exp = np.cumsum(a_cdom_exp_fit)
    cdf_exp /= cdf_exp[-1]

    cdf_exptot = np.cumsum(a_cdom_totexp_fit)
    cdf_exptot /= cdf_exptot[-1]

    cdf_pow = np.cumsum(a_cdom_pow_fit)
    cdf_pow /= cdf_pow[-1]

    ax_cdf = plt.subplot(gs[1])

    # Plot
    ax_cdf.step(cut_wv, cdf_NMF, label=r'$\xi_'+f'{ss}'+'$', color='k')
    ax_cdf.plot(cut_wv, cdf_exp, color='b', label='CDOM exp', ls='-')
    ax_cdf.plot(cut_wv, cdf_exptot, color='b', label='CDOM exp', ls='--')
    ax_cdf.plot(cut_wv, cdf_pow, color='r', label='CDOM pow', ls='-')

    # Finish
    for ax in [ax_fits, ax_cdf]:
        plotting.set_fontsize(ax, 15)
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_nmf_indiv(outfile:str='fig_nmf_indiv.png',
                 nmf_fit:str='l23', N_NMF:int=4):
    pass
    '''
    #
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Basis vector')
        ax.minorticks_on()

        elif (N_NMF==5 and ss == 1) or (N_NMF==4 and ss == 0): # CDOM
            #embed(header='fig_nmf_indiv 150')
            # Expoential
            a_cdom_exp = cdom.a_exp(wave)
            iwv = np.argmin(np.abs(wave-400.))
            a_cdom_exp *= M[ss][iwv] / a_cdom_exp[iwv]

            # Power law
            a_cdom_pow = cdom.a_pow(wave)
            a_cdom_pow *= M[ss][iwv] / a_cdom_pow[iwv]

            # Power law fit
            coeff, cov = cdom.fit_pow(wave, M[ss])
            a_cdom_pow_fit = coeff[0] * cdom.a_pow(wave, S=coeff[1])

            #
            ax.plot(wave, a_cdom_exp, color='gray', label='CDOM exp', ls='--')
            ax.plot(wave, a_cdom_pow, color='gray', label='CDOM pow', ls=':')
            ax.plot(wave, a_cdom_pow_fit, color='gray', 
                    label=f'CDOM pow: S={coeff[1]:0.2f}', ls='-')

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
    '''

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
        fig_nmf_basis(N_NMF=5)

    # Individual
    if flg & (2**2):
        fig_nmf_indiv()
        fig_nmf_indiv(outfile='fig_nmf_indiv_N5.png',
            N_NMF=5)

    # Coeff
    if flg & (2**3):
        fig_nmf_coeff()

    # Fit CDOM
    if flg & (2**4):
        fig_fit_cdom()

    # Explained variance
    if flg & (2**5):
        fig_explain_variance()



# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- RMSE
        #flg += 2 ** 1  # 2 -- NMF basis
        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
    else:
        flg = sys.argv[1]

    main(flg)