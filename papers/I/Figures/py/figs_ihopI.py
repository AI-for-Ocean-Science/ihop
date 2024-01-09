""" Figures for Paper I on IHOP """

# imports
import os
from importlib import resources

import numpy as np

import torch
import h5py

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import corner

from oceancolor.hydrolight import loisel23
from oceancolor.utils import plotting 

from ihop.emulators import io as ihop_io
from ihop.iops import pca as ihop_pca
#from ihop.iops import nmf as ihop_nmf
from ihop.iops.pca import load_loisel_2023_pca
#from ihop.iops.nmf import evar_computation

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas


def fig_emulator_rmse(emulator:str,
                      outfile:str='fig_emulator_rmse.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if emulator == 'L23_PCA':
        emulator_file = os.path.join(os.getenv('OS_COLOR'),
                                     'IHOP',
                                     'Emulators',
                                     'DenseNet_PCA',
                                     'dense_l23_pca_X4Y0_512_512_512_256_chl.pth')
        #
        X,Y = 4,0
        ab, Rs, d_a, _ = load_loisel_2023_pca()
        wave = d_a['wave']
        # Data
        ds_l23 = loisel23.load_ds(X, Y)
        # Chl
        Chl = loisel23.calc_Chl(ds_l23)
        # Concatenate
        inputs = np.concatenate((ab, Chl.reshape(Chl.size,1)), axis=1)
        targets = Rs

    # Load emulator
    model = ihop_io.load_nn(emulator_file)

    # Predict and compare
    dev = np.zeros_like(targets)
    for ss in range(targets.shape[0]):
        dev[ss,:] = targets[ss] - model.prediction(inputs[ss],
                                                   device)
    
    # RMSE
    rmse = np.sqrt(np.mean(dev**2, axis=0))

    # Plot
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # #####################################################
    # PCA
    ax= plt.subplot(gs[0])

    ax.plot(wave, rmse, 'o')

    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('RMSE')

    plotting.set_fontsize(ax, 17)
    #ax.set_xlim(1., 10)
    #ax.set_ylim(1e-5, 0.01)
    #ax.set_yscale('log')
    #ax.legend(fontsize=15)

    ax.text(0.95, 0.90, emulator, color='k',
        transform=ax.transAxes,
        fontsize=22, ha='right')

    
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
        fig_emulator_rmse('L23_PCA')



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

        flg += 2 ** 20  # 1 -- RMSE

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
        
    else:
        flg = sys.argv[1]

    main(flg)