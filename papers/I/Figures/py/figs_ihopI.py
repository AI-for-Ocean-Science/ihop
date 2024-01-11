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

from ihop.emulators import io as ihop_io
from ihop.iops.pca import load_loisel_2023_pca


mpl.rcParams['font.family'] = 'stixgeneral'

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ihopI_io



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

    # Mean Rs
    mean_Rs = np.mean(targets, axis=0)

    # Plot
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(3,1)

    # #####################################################
    # Absolute
    ax_abs = plt.subplot(gs[0:2])

    ax_abs.plot(wave, rmse, 'o')

    ax_abs.set_ylabel(r'Absolute RMSE (m$^{-1}$)')
    ax_abs.tick_params(labelbottom=False)  # Hide x-axis labels


    #ax.set_xlim(1., 10)
    #ax.set_ylim(1e-5, 0.01)
    #ax.set_yscale('log')
    #ax.legend(fontsize=15)

    ax_abs.text(0.95, 0.90, emulator, color='k',
        transform=ax_abs.transAxes,
        fontsize=22, ha='right')

    # Relative
    ax_rel = plt.subplot(gs[2])
    ax_rel.plot(wave, rmse/mean_Rs, 'o', color='k')
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
def fig_mcmc_fit(outfile='fig_mcmc_fit.png', iop_type='nmf',
                 use_quick:bool=False,
                 show_zoom:bool=False):
    # Load
    in_idx = 0
    items = ihopI_io.load_l23_fit( in_idx, iop_type=iop_type, use_quick=use_quick)
    d_a, idx, orig, a_mean, a_std, a_iop, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, Rs, ab, allY, wave,\
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

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Example spectra
    if flg & (2**20):
        fig_emulator_rmse('L23_PCA')

    # L23 IHOP performance vs. perc error
    if flg & (2**21):
        fig_mcmc_fit()#use_quick=True)

    # L23 IHOP performance vs. perc error
    if flg & (2**22):
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

        #flg += 2 ** 20  # RMSE
        flg += 2 ** 21  # Single MCMC fit (example)

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
        
    else:
        flg = sys.argv[1]

    main(flg)