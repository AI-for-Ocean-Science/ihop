""" Figures for the PACE 2023 proposal """

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

from oceancolor.utils import plotting 

from ihop.emulators import io as ihop_io
from ihop.iops import pca as ihop_pca
from ihop.iops import nmf as ihop_nmf
from ihop.iops.pca import load_loisel_2023_pca
from ihop.iops.nmf import load_loisel_2023

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas


from IPython import embed

def load_nmf(nmf_fit:str, N_NMF:int=None, iop:str='a'):

    # Load
    if nmf_fit == 'l23':
        if N_NMF is None:
            N_NMF = 5
        path = os.path.join(resources.files('ihop'), 
                    'data', 'NMF')
        outroot = os.path.join(path, f'L23_NMF_{iop}_{N_NMF}')
        nmf_file = outroot+'.npz'
        #
        d = np.load(nmf_file)
    else:
        raise IOError("Bad input")

    return d


# #############################################
# Load
def load_up(in_idx:int, chop_burn = -3000,
            iop_type:str='nmf'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Chains
    out_path = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'L23',
        iop_type.upper())
    chain_file = 'fit_a_L23_NN_Rs10.npz'
           
    # Load prep
    if iop_type == 'pca':
        lfunc = load_loisel_2023_pca
        em_path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 
                               'Emulators', 'SimpleNet_PCA')
        model_file += '/model_100000.pth'
        ncomp = 3
    elif iop_type == 'nmf':
        # Model
        em_path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 
                               'Emulators')
        model_file = os.path.join(
            em_path, 'DenseNet_NM4',
            'densenet_NMF_[512, 512, 512, 256]_epochs_25000_p_0.0_lr_0.01.pth')
        lfunc = load_loisel_2023
        ncomp = 4

    # Do it
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb = lfunc()
    print(f"Loading model: {model_file}")
    model = ihop_io.load_nn(model_file)

    nwave = d_a['wave'].size

    # MCMC
    print("Loading MCMC")
    d = np.load(os.path.join(out_path, chain_file))
    chains = d['chains']
    l23_idx = d['idx']
    obs_Rs = d['obs_Rs']

    idx = l23_idx[in_idx]
    print(f'Working on: L23 index={idx}')

    # Prep
    if iop_type == 'pca':
        rfunc = ihop_pca.reconstruct
        wave = d_a['wavelength']
    elif iop_type == 'nmf':
        rfunc = ihop_nmf.reconstruct
        wave = d_a['wave']
    
    # a
    Y = chains[in_idx, chop_burn:, :, 0:ncomp].reshape(-1,ncomp)
    orig, a_recon = rfunc(Y, d_a, idx)
    a_mean = np.mean(a_recon, axis=0)
    a_std = np.std(a_recon, axis=0)
    _, a_pca = rfunc(ab[idx][:ncomp], d_a, idx)

    # Rs
    allY = chains[in_idx, chop_burn:, :, :].reshape(-1,ncomp*2)
    all_pred = np.zeros((allY.shape[0], nwave))
    for kk in range(allY.shape[0]):
        Ys = allY[kk]
        pred_Rs = model.prediction(Ys, device)
        all_pred[kk,:] = pred_Rs

    pred_Rs = np.median(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)
    NN_Rs = model.prediction(ab[idx], device)

    return d_a, idx, orig, a_mean, a_std, a_pca, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, Rs, ab, allY, wave

def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)


def fig_l23_tara_pca_nmf(
    outfile='fig_l23_tara_pca_nmf.png',
    show_spec:bool=False,
    nmf_fit:str='l23'):

    # Load up
    L23_Tara_pca_N20 = ihop_pca.load('pca_L23_X4Y0_Tara_a_N20.npz')
    N=3
    L23_Tara_pca = ihop_pca.load(f'pca_L23_X4Y0_Tara_a_N{N}.npz')
    wave = L23_Tara_pca['wavelength']

    # NMF
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

    # Figure
    clrs = ['b', 'g']
    figsize=(12,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # #####################################################
    # PCA
    ax_var = plt.subplot(gs[0])

    ax_var.plot(np.arange(L23_Tara_pca_N20['explained_variance'].size)+1,
        np.cumsum(L23_Tara_pca_N20['explained_variance']), 'o-',
        color=clrs[0])
    ax_var.set_xlabel('Number of Components')
    ax_var.set_ylabel('Cumulative Explained Variance')
    # Horizontal line at 1
    ax_var.axhline(1., color='k', ls=':')


    axes = [ax_var]

    # #####################################################
    # NMF
    ax_nmf = plt.subplot(gs[1])

    ax_nmf.plot(2+np.arange(N_NMF-1), rmss, 'o', color=clrs[1])

    ax_nmf.set_xlabel('Number of Components')
    ax_nmf.set_ylabel(r'Average RMSE (m$^{-1}$)')
        
    ax_nmf.set_ylim(0., 0.0007)

    #ax_nmf.set_yscale('log')

    axes.append(ax_nmf)

    # Finish
    for ss, ax in enumerate(axes):
        plotting.set_fontsize(ax, 17)
        ax.set_xlim(0., 10)
        lbl = 'PCA' if ss == 0 else 'NMF'
        ax.text(0.95, 0.05, lbl, color=clrs[ss],
            transform=ax.transAxes,
              fontsize=18, ha='right')


    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")





def fig_mcmc_fit(outfile='fig_mcmc_fit.png', iop_type='nmf'):
    # Load
    in_idx = 0
    items = load_up(in_idx, iop_type=iop_type)
    d_a, idx, orig, a_mean, a_std, a_pca, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, Rs, ab, allY, wave = items

    # #########################################################
    # Plot the solution

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)
    
    # a
    ax_a = plt.subplot(gs[0])
    def plot_spec(ax):
        ax.plot(wave, orig, 'bo', label='True')
        ax.plot(wave, a_mean, 'r-', label='Fit')
        ax.plot(wave, a_pca, 'k:', label='PCA')
        ax.fill_between(wave, a_mean-a_std, a_mean+a_std, 
            color='r', alpha=0.5) 
    plot_spec(ax_a)
    ax_a.set_xlabel('Wavelength (nm)')
    ax_a.set_ylabel(r'$a(\lambda)$')

    # Zoom in
    # inset axes....
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
    # Rs
    ax_R = plt.subplot(gs[1])
    ax_R.plot(wave, Rs[idx], 'bo', label='True')
    ax_R.plot(wave, obs_Rs[in_idx], 'ks', label='Obs')
    ax_R.plot(wave, pred_Rs, 'rx', label='Model')
    ax_R.plot(wave, NN_Rs, 'g-', label='NN+True')

    ax_R.fill_between(wave,
        pred_Rs-std_pred, pred_Rs+std_pred,
        color='r', alpha=0.5) 

    ax_R.set_xlabel('Wavelength (nm)')
    ax_R.set_ylabel(r'$R_s$')

    ax_R.legend()
    
    # axes
    for ax in [ax_a, ax_R]:
        plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_corner(outfile='fig_corner.png'):
    in_idx = 0
    items = load_up(in_idx)
    d_a, idx, orig, a_mean, a_std, a_pca, obs_Rs,\
        pred_Rs, std_pred, NN_Rs, Rs, ab, allY = items
    
    fig = corner.corner(
        allY, labels=['a0', 'a1', 'a2', 'b0', 'b1', 'b2'],
        truths=ab[idx],
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

    # PCA
    if flg & (2**0):
        fig_l23_tara_pca_nmf()

    # MCMC fit
    if flg & (2**1):
        fig_mcmc_fit()

    # MCMC fit
    if flg & (2**2):
        fig_corner()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- PCA + NMF perf.
        #flg += 2 ** 1  # 2 -- MCMC fit
    else:
        flg = sys.argv[1]

    main(flg)