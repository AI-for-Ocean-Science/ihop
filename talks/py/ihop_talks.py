""" Figures for IHOP talks """

import os
from importlib import resources

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

from cnmf import io as cnmf_io
from cnmf import stats as cnmf_stats

from IPython import embed



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

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Example spectra
    if flg & (2**0):
        fig_abR()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Example spectra
        #flg += 2 ** 1  # 2 -- L23: PCA vs NMF Explained variance
    else:
        flg = sys.argv[1]

    main(flg)