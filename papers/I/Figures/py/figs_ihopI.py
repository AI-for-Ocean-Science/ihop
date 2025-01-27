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




def fig_basis_functions(decomps:tuple,
                        outfile:str='fig_basis_functions.png', 
                        in_Ncomps=None,
                        norm:bool=False):

    X, Y = 4, 0
    if in_Ncomps is not None:
        Ncomps = in_Ncomps

    # Load
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    wave = d_a['wave']

    # Load training data
    d_train = load_rs.loisel23_rs(X=X, Y=Y)

    # Seaborn
    sns.set(style="whitegrid",
            rc={"lines.linewidth": 2.5})
            # 'axes.edgecolor': 'black'
    sns.set_palette("pastel")
    sns.set_context("paper")
    #sns.set_context("poster", linewidth=3)
    #sns.set_palette("husl")

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)


    for ss, IOP in enumerate(['a', 'bb']):
        ax = plt.subplot(gs[ss])

        d = d_a if IOP == 'a' else d_bb
        if IOP == 'a':
            iop_w = absorption.a_water(wave, data='IOCCG')
        else:
            iop_w = d_train['bb_w']
        iop_w /= np.sum(iop_w)

        # Plot water first
        sns.lineplot(x=wave, y=iop_w,
                            label=r'$W_'+f'{1}'+r'^{\rm '+IOP+r'}$',
                            ax=ax, lw=2)#, drawstyle='steps-pre')

        # Now the rest
        if decomps[ss] == 'hyb':
            #embed(header='114 of figs ihop')
            M = np.concatenate([d['W1'], d['W2']]).reshape(2, d['W1'].size)
            spec = d['data']
            coeff = d['coeff'][:,2:]
            Ncomp = 2
        else:
            M = d['M']
            spec = d['spec']
            coeff = d['coeff']
            Ncomp = Ncomps[ss]
        # Variance
        if decomps[ss] in ['nmf', 'hyb']:
            evar_i = cnmf_stats.evar_computation(
                spec, coeff, M)
        else:
            evar_i = np.sum(d['explained_variance'])
        # Plot
        for ii in range(Ncomp):
            # Normalize
            if norm:
                iwv = np.argmin(np.abs(wave-440.))
                nrm = M[ii][iwv]
            else:
                nrm = 1.
            # Step plot
            sns.lineplot(x=wave, y=M[ii]/nrm, 
                            label=r'$W_'+f'{ii+2}'+r'^{\rm '+IOP+r'}$',
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
        ax.text(0.5, 0.5, f'Explained variance: {100*evar_i:.2f}%',
            transform=ax.transAxes,
            fontsize=13, ha='center')

        loc = 'upper right' if ss == 1 else 'upper left'
        ax.legend(fontsize=15, loc=loc)

        plotting.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_nmf_corner(outroot='fig_nmf_corner', decomps:tuple=('nmf','nmf'),
        dataset:str='L23', X:int=4, Y:int=0):

    in_idx = 2663
    # Load
    edict = emu_io.set_emulator_dict(dataset, decomps, Ncomps, 'Rrs',
        'dense', include_chl=True, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(decomps, Ncomps)

    for ss, iop in enumerate(['a', 'bb']):

        lbls = [r'$H_'+f'{ii+2}'+r'^'+f'{iop}'+'$' for ii in range(Ncomps[ss])]
        if ss == 0:
            coeff = ab[:, :Ncomps[0]]
        else:
            coeff = ab[:, Ncomps[0]:]

        # Set minimum
        for ii in range(Ncomps[ss]):
            coeff[:,ii] = np.maximum(coeff[:,ii], 1e-3)

        plt.clf()
        fig = corner.corner(
            coeff, labels=lbls,
            label_kwargs={'fontsize':17},
            color='k',
            axes_scale='log',
            #truths=truths,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            )

        outfile = f'{outroot}_{iop}.png'
        plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
        plt.savefig(outfile, dpi=300)
        print(f"Saved: {outfile}")


def fig_emulator_rmse(dataset:str, Ncomps:tuple, hidden_list:list,
                      decomps:tuple,
                      outfile:str='fig_emulator_rmse.png',
                      log_rrmse:bool=False,
                      include_chl:bool=True,
                      preproc_Rs:str=None,
                      X:int=4, Y:int=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        preproc_Rs=preproc_Rs,
        include_chl=include_chl, X=X, Y=Y)

    # Init the Plot
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(4,1)
    ax_rel = plt.subplot(gs[2])
    ax_bias = plt.subplot(gs[3])
    ax_abs = plt.subplot(gs[0:2])

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    emulator, e_file = emu_io.load_emulator_from_dict(edict)
    print(f"Using: {e_file} for the emulator")
    wave = d_a['wave']

    clrs = ['k', 'b', 'r', 'g']
    for ss, dset in enumerate(['Training', 'Validation']):
        if ss > 0:
            continue
        clr = clrs[ss]

        # Concatenate
        if include_chl:
            inputs = np.concatenate((ab, Chl.reshape(Chl.size,1)), axis=1)
        else:
            inputs = ab
        targets = Rs

        # Predict and compare
        dev = np.zeros_like(targets)
        for ss in range(targets.shape[0]):
            dev[ss,:] = emulator.prediction(inputs[ss], device) - targets[ss]  
        
        # Bias?
        bias = np.mean(dev, axis=0)
        
        # RMSE
        rmse = np.sqrt(np.mean(dev**2, axis=0))
        rRMSE = np.sqrt(np.mean((dev/targets)**2, axis=0))
        #rRMSE = rmse/mean_Rs

        # Mean Rs
        mean_Rs = np.mean(targets, axis=0)

        # #####################################################
        # Absolute

        ax_abs.plot(wave, rmse, 'o', color=clr, label=f'{dset}')

        ax_abs.set_ylabel(r'RMSE  (10$^{-4} \, \rm sr^{-1}$)')
        ax_abs.tick_params(labelbottom=False)  # Hide x-axis labels

        ax_abs.legend(fontsize=15)

        #ax.set_xlim(1., 10)
        #ax.set_ylim(1e-5, 0.01)
        #ax.set_yscale('log')
        #ax.legend(fontsize=15)

        #ax_abs.text(0.95, 0.90, model, color='k',
        #    transform=ax_abs.transAxes,
        #    fontsize=22, ha='right')

        # #####################################################
        # Relative
        ax_rel.plot(wave, rRMSE, 'o', color=clr)
        ax_rel.set_ylabel('rRMSE')
        #ax_rel.set_ylim(0., rRMSE.max()*1.05)

        ax_rel.tick_params(labelbottom=False)  # Hide x-axis labels

        if log_rrmse:
            ax_rel.set_yscale('log')
        else:
            ax_rel.set_ylim(0., 0.10)

        # #####################################################
        # Bias
        ax_bias.plot(wave, bias/mean_Rs, 'o', color=clr)
        ax_bias.set_ylabel('Relative Bias')
        ax_bias.set_ylim(-0.02, 0.02)


    # Finish
    for ss, ax in enumerate([ax_abs, ax_rel, ax_bias]):
        plotting.set_fontsize(ax, 15)
        if ss == 2:
            ax.set_xlabel('Wavelength [nm]')
        # Grid
        ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    # A few stats on Chl
    min_chl = np.min(Chl)
    max_chl = np.max(Chl)
    print(f'Min Chl: {np.log10(min_chl):.2f}, Max Chl: {np.log10(max_chl):.2f}')


# ############################################################
def fig_rmse_Rrs_a(decomps:tuple, Ncomps:tuple, outfile:str, 
                   hidden_list:list=[512, 512, 512, 256], dataset:str='L23', 
            X:int=4, Y:int=0, abs_sig:float=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ######################
    # Load
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    wave = d_a['wave']
    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=True, X=X, Y=Y)

    # Load chains
    recon_file = os.path.join(
        '../Analysis/',
        os.path.basename(fitting_io.l23_chains_filename(
        edict, abs_sig).replace('fit', 'recon')))
    d_recon = np.load(recon_file)
    chain_idx = d_recon['idx']

    # #############################
    # a
    tkey = 'spec' if decomps[0] == 'nmf' else 'data'
    a_true = d_a[tkey][chain_idx]

    fit_diff = d_recon['fit_a_mean'] - a_true
    a_fit_RMSE = np.sqrt(np.mean((fit_diff)**2, axis=0))
    a_fit_MAD = np.median(np.abs(fit_diff), axis=0)

    # Set min
    a_true_min = np.maximum(a_true, 1e-4)
    a_fit_rMAD = np.median(np.abs(fit_diff/a_true_min), axis=0)
    a_fit_rRMSE = np.sqrt(np.mean((fit_diff/a_true_min)**2, axis=0))

    # ############################
    # Calc Rrs

    # RMSE
    fit_diff = d_recon['fit_Rrs'] - Rs[chain_idx]
    fit_rrmse = np.sqrt(np.mean((fit_diff/Rs[chain_idx])**2, 
                                axis=0))

    corr_diff = d_recon['corr_Rrs'] - Rs[chain_idx]
    corr_rrmse = np.sqrt(np.mean((corr_diff/Rs[chain_idx])**2, axis=0))
    
    #ii = 275
    #frmse = np.sum(fit_diff[ii,:]**2)
    #crmse = np.sum(corr_diff[ii,:]**2)

    # ######################################################
    # ######################################################
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(3,1)

    aaxes = [] 

    # ##############################
    # Rrs
    ax_R = plt.subplot(gs[0])
    aaxes.append(ax_R)

    ax_R.plot(wave, fit_rrmse, 'ro', label='Fit')
    ax_R.plot(wave, corr_rrmse, 'kx', label='Correct')

    ax_R.set_ylim(0., 0.05)


    ax_R.legend()
    ax_R.set_ylabel(r'RMSE $R_{\rm rs}$')
    ax_R.tick_params(labelbottom=False)  # Hide x-axis labels


    # ##############################
    # Absolute a
    ax_a = plt.subplot(gs[1])
    aaxes.append(ax_a)

    ax_a.plot(wave, a_fit_MAD, 'bo', label='Fit MAD')
    ax_a.plot(wave, a_fit_RMSE, 'ro', label='Fit RMSE')

    ax_a.set_ylabel(r'Absolute $a_{\rm nw}(\lambda)$ Error')
    ax_a.tick_params(labelbottom=False)  # Hide x-axis labels

    # ##############################
    # Relative a
    ax_ra = plt.subplot(gs[2])
    aaxes.append(ax_ra)

    ax_ra.set_ylabel(r'Relative $a_{\rm nw}(\lambda)$ Error')

    ax_ra.plot(wave, a_fit_rMAD, 'ks', label='Fit rMAD')
    ax_ra.plot(wave, a_fit_rRMSE, 'gs', label='Fit rRMSE')

    ax_ra.set_ylim(0., 0.2)
    ax_ra.set_xlabel('Wavelength (nm)')

    # All
    for ax in aaxes:
        plotting.set_fontsize(ax, 18)
        ax.legend(fontsize=17.)
        ax.grid()

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_rmse_a_error(decomps:tuple, Ncomps:tuple, outfile:str, 
                     abs_sigs:list, hidden_list:list=[512, 512, 512, 256], 
                     dataset:str='L23', X:int=4, Y:int=0, show_bias:bool=False,
                     use_log_ab:bool=False):

    # ######################
    # Load
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    wave = d_a['wave']
    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=True, X=X, Y=Y)

    priors = None
    if use_log_ab:
        priors = {}
        priors['use_log_ab'] = True

    # Noiseless
    recon_file = os.path.join(
        '../Analysis/',
        os.path.basename(fitting_io.l23_chains_filename(
        edict, None, priors=priors).replace('fit', 'recon')))
    d_nless = np.load(recon_file)
    a_nless = calc_rmses(d_a, d_nless, decomps[0])

    # Loop on abs_sigs
    a_rmses = []
    a_biases = []
    for abs_sig in abs_sigs:

        # #############################

        # Load chains
        recon_file = os.path.join(
            '../Analysis/',
            os.path.basename(fitting_io.l23_chains_filename(
            edict, abs_sig, priors=priors).replace('fit', 'recon')))
        d_recon = np.load(recon_file)
        chain_idx = d_recon['idx']

        a_recon = calc_rmses(d_a, d_recon, decomps[0])
        a_rmses.append(a_recon)

    # ######################################################
    # ######################################################
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    aaxes = [] 
    akeys = ['a_fit_MAD', 'a_fit_RMSE',
             'a_fit_rMAD', 'a_fit_rRMSE']
    ylbls = [r'Absolute $a_{\rm nw}(\lambda)$ MAD', 
             r'Absolute $a_{\rm nw}(\lambda)$ RMSE',
                r'Relative $a_{\rm nw}(\lambda)$ MAD',
                r'Relative $a_{\rm nw}(\lambda)$ RMSE']
    if show_bias:
        akeys[0] = 'a_fit_BIAS'
        akeys[2] = 'a_fit_rBIAS'
        #
        ylbls[0] = r'Absolute $a_{\rm nw}(\lambda)$ Bias'
        ylbls[2] = r'Relative $a_{\rm nw}(\lambda)$ Bias'

    for tt, akey, ylbl in zip(range(len(akeys)), akeys, ylbls):
        ax= plt.subplot(gs[tt])
        aaxes.append(ax)

        ax.plot(wave, a_nless[akey], 'kx', label='Noiseless')
        for ss, abs_sig in enumerate(abs_sigs):
            ax.plot(wave, a_rmses[ss][akey], 'o', label=f'abs_sig={abs_sig}')

        ax.set_ylabel(ylbl)

    # All
    for ii, ax in enumerate(aaxes):
        plotting.set_fontsize(ax, 18)
        ax.grid()
        # 
        if ii == 2:
            ax.legend(fontsize=17.)
        if ii > 1:
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylim(0., 0.3)
        else:
            ax.tick_params(labelbottom=False)  # Hide x-axis labels

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_summary_a(items:list, outfile:str, 
                    b_dN=('nmf', 2),
                    hidden_list:list=[512, 512, 512, 256], 
                    dataset:str='L23', X:int=4, Y:int=0, 
                    show_bias:bool=False):
                    
    # Load wavelength
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        ('nmf', 'nmf'), (2,2))
    wave = d_a['wave']

    # Work on the RMSEs
    a_rmses = []
    for item in items:
        # Unpack
        abs_sig, decomp0, N0, priors, clr, ls = item
        # Calculate
        a_rmses.append(
            calc_stats.calc_a_stats(
                abs_sig, (decomp0, b_dN[0]), (N0, b_dN[1]), 
                priors))

    # ######################################################
    # ######################################################
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    aaxes = [] 
    akeys = ['a_fit_MAD', 'a_fit_RMSE',
             'a_fit_rMAD', 'a_fit_rRMSE']
    ylbls = [r'Absolute $a_{\rm nw}(\lambda)$ MAD', 
             r'Absolute $a_{\rm nw}(\lambda)$ RMSE',
                r'Relative $a_{\rm nw}(\lambda)$ MAD',
                r'Relative $a_{\rm nw}(\lambda)$ RMSE']

    n_clrs = {2:'k', 3:'b', 4:'g'}

    for tt, akey, ylbl in zip(range(len(akeys)), akeys, ylbls):
        ax= plt.subplot(gs[tt])
        aaxes.append(ax)

        #ax.plot(wave, a_nless[akey], 'kx', label='Noiseless')
        for ss, a_rmse in enumerate(a_rmses):
            abs_sig, decomp0, N0, priors, clr, ls = items[ss]
            # Parse clr
            if clr == 'use_n':
                clr = n_clrs[N0]
            lasig = abs_sig if abs_sig is not None else 0
            label=f'N0={N0}, abs_sig={lasig}'
            ax.plot(wave, a_rmse[akey], color=clr, ls=ls,
                    label=label)

        ax.set_ylabel(ylbl)

    # All
    for ii, ax in enumerate(aaxes):
        plotting.set_fontsize(ax, 18)
        ax.grid()
        # 
        if ii == 2:
            ax.legend(fontsize=17.)
        if ii > 1:
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylim(0., 0.3)
        else:
            ax.tick_params(labelbottom=False)  # Hide x-axis labels

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ############################################################
def fig_a_examples(decomps:tuple, Ncomps:tuple, outfile:str, 
                     abs_sigs:list, hidden_list:list=[512, 512, 512, 256], 
                     dataset:str='L23', X:int=4, Y:int=0,
                     use_log_ab:bool=False, include_chl:bool=True,
                     skip_fits:bool=False, show_LS2:bool=False,
                     show_noiseless_error:bool=False,
                     show_absig_error:list=None):

    # ######################
    # Load
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    wave = d_a['wave']
    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=include_chl, X=X, Y=Y)

    priors = None
    if use_log_ab:
        priors = {}
        priors['use_log_ab'] = True


    tkey = 'spec' if decomps[0] == 'nmf' else 'data'

    # Hack for normalized PCA
    if decomps[0] == 'npca':
        sdata = d_a['data'] * np.outer(d_a['norm_vals'], 
                                             np.ones(d_a['data'].shape[1]))
    else:
        sdata = d_a[tkey]

    # Set the indices
    i440 = np.argmin(np.abs(wave-440.))
    i_min = np.argmin(sdata[:,i440])
    i_max = np.argmax(sdata[:,i440])
    a_med = np.median(sdata[:,i440])
    i_med = np.argmin(np.abs(sdata[:,i440] - a_med))
    print(f'Min={i_min}, Max={i_max}, Med={i_med}')

    i_random = np.random.choice(sdata.shape[0], 1)[0]



    # Decomposed
    if skip_fits:
        d_keys = dict(pca='Y', nmf='coeff', int='new_spec', npca='Y')
        if decomps[0] in ['pca', 'npca']:
            rfunc = reconstruct_pca
        elif decomps[0] == 'nmf':
            rfunc = reconstruct_nmf
        elif decomps[0] == 'int':
            rfunc = reconstruct_int
        a_recons = []
        for idx in range(d_a[tkey].shape[0]):
            _, a_recon = rfunc(d_a[d_keys[decomps[0]]][idx], d_a, idx)
            a_recons.append(a_recon)
        a_recons = np.array(a_recons)
    else:
        # Noiseless
        recon_file = os.path.join(
            '../Analysis/',
            os.path.basename(fitting_io.l23_chains_filename(
            edict, None, priors=priors).replace('fit', 'recon')))
        d_nless = np.load(recon_file)
        a_recons = d_nless['decomp_a']

        # Load recons
        d_recons = []
        for abs_sig in abs_sigs:
            recon_file = os.path.join(
                '../Analysis/',
                os.path.basename(fitting_io.l23_chains_filename(
                edict, abs_sig, priors=priors).replace('fit', 'recon')))
            print(f'Loading: {recon_file}')
            d_recon = np.load(recon_file)
            # Append
            d_recons.append(d_recon)


    # ######################################################
    # ######################################################
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    aaxes = [] 
    idxs = [i_min, i_max, i_med, i_random]
    ylbls = [r'$a_{\rm nw}(\lambda)$ min', 
                r'$a_{\rm nw}(\lambda)$ max',
                    r'$a_{\rm nw}(\lambda)$ median',
                    r'$a_{\rm nw}(\lambda)$ random']

    for tt, idx, ylbl in zip(range(len(idxs)), idxs, ylbls):
        # 
        ax= plt.subplot(gs[tt])
        aaxes.append(ax)

        scale = d_a['norm_vals'][idx] if decomps[0] == 'npca' else 1.

        # True
        ax.plot(wave, d_a[tkey][idx]*scale, 'ko', label='True', ms=4)
    
        # True Decomposition
        ax.plot(wave, a_recons[idx], ':', color='gray', label='Decomposition')

        # LS2?
        if show_LS2:
            waves, all_a, all_anw, all_bb, all_bbp = anly_ls2.calc_ls2(idx)
            ax.plot(waves, all_anw, 'o', color='red',ms=2., label='LS2')

        if not skip_fits:
            ii = np.where(d_nless['idx'] == idx)[0][0]
            # Noiseless
            ax.plot(wave, d_nless['fit_a_mean'][ii], 'k-', label='Noiseless')


            if show_noiseless_error:
                # Noiseless
                ax.fill_between(wave, 
                    d_nless['fit_a_mean'][ii] - d_nless['fit_a_std'][ii],
                    d_nless['fit_a_mean'][ii] + d_nless['fit_a_std'][ii],
                    color='gray', alpha=0.5)

            # Loop on abs_sigs
            for jj, abs_sig in enumerate(abs_sigs):
                ax.plot(wave, d_recons[jj]['fit_a_mean'][ii], '-', 
                        label=f'abs_sig={abs_sig}')
                if show_absig_error is not None and abs_sig in show_absig_error:
                    ax.fill_between(wave, 
                        d_recons[jj]['fit_a_mean'][ii] - d_recons[jj]['fit_a_std'][ii],
                        d_recons[jj]['fit_a_mean'][ii] + d_recons[jj]['fit_a_std'][ii],
                        alpha=0.5)

        ax.set_ylabel(ylbl)
        #embed(header='fig_a_examples 592')

    # All
    for ii, ax in enumerate(aaxes):
        plotting.set_fontsize(ax, 18)
        ax.grid()
        # 
        if ii == 0:
            ax.legend(fontsize=17.)
        if ii > 1:
            ax.set_xlabel('Wavelength (nm)')
            #ax.set_ylim(0., 0.3)
        else:
            ax.tick_params(labelbottom=False)  # Hide x-axis labels

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


# ############################################################
def fig_mcmc_decompose(outroot='fig_mcmc_decompose', decomp:str='nmf',
        hidden_list:list=[512, 512, 512, 256], dataset:str='L23', use_quick:bool=False,
        X:int=4, Y:int=0, show_zoom:bool=False, 
        perc:int=None, abs_sig:float=None,
        wvmnx:tuple=None,
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
    lgsz = 18.

    fig = plt.figure(figsize=(10,12))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax_a = plt.subplot(gs[0])
    ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')
    ax_a.set_xlabel('Wavelength (nm)')

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ############################################################
def fig_mcmc_fit(outroot='fig_mcmc_fit', decomps:str=('nmf','nmf'),
        hidden_list:list=[512, 512, 512, 256], dataset:str='L23', 
        use_quick:bool=False,
        X:int=4, Y:int=0, show_zoom:bool=False, 
        in_Ncomps=None,
        perc:int=None, abs_sig:float=None,
        wvmnx:tuple=None, show_NMF:bool=False,
        water:bool=False, in_idx:int=0, use_reconstruct:bool=False,
        use_Chl:bool=True,
        chain_file:str=None, in_log10:bool=False,
        test:bool=False, true_obs_only:bool=False,
        true_only:bool=False):

    if in_Ncomps is not None:
        Ncomps = in_Ncomps
    # Load
    edict = emu_io.set_emulator_dict(dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=use_Chl, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(decomps, Ncomps)

    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    if chain_file is None:
        chain_file = inf_io.l23_chains_filename(edict, 
                                            perc if perc is not None else int(abs_sig), 
                                            test=test)
    d_chains = inf_io.load_chains(chain_file)

    if use_reconstruct:
        # Reconstruct
        if decomps[0] == 'bsp': # HACK
            Ncomps = (Ncomps[0]+5, Ncomps[1])
        items = reconstruct.one_spectrum(
            in_idx, ab, Chl, d_chains, 
            d_a, d_bb, emulator, decomps, Ncomps,
            in_log10=in_log10, use_Chl=use_Chl)
        idx, orig, a_mean, a_std, a_iop, obs_Rs,\
            pred_Rs, std_pred, NN_Rs, allY, wave,\
            orig_bb, bb_mean, bb_std, a_nmf, bb_nmf = items
        print(f"L23 index = {idx}")
    else:
        priors = None
        if in_log10:
            priors = {}
            priors['use_log_ab'] = True

        recon_file = os.path.join(
            '../Analysis/',
            os.path.basename(fitting_io.l23_chains_filename(
            edict, abs_sig, priors=priors).replace('fit', 'recon')))
        d_recon = np.load(recon_file)
        idx = np.where(d_recon['idx'] == in_idx)[0][0]

        # Unpack here
        wave = d_a['wave']
        orig = d_a['spec'][in_idx]
        a_mean = d_recon['fit_a_mean'][idx]
        a_std = d_recon['fit_a_std'][idx]
        orig_bb = d_bb['spec'][in_idx]
        bb_mean = d_recon['fit_bb_mean'][idx]
        bb_std = d_recon['fit_bb_std'][idx]

        obs_Rs = d_chains['obs_Rs']
        pred_Rs = d_recon['fit_Rrs'][idx]
        std_pred = 0.01

    # Outfile
    outfile = outroot + f'_{idx}.png'
    if water:
        outfile = outfile.replace('.png', '_water.png')
        # Load training data
        d_train = load_rs.loisel23_rs(X=X, Y=Y)


    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    
    xpos, ypos, ypos2 = 0.05, 0.10, 0.10

    # #########################################################
    # a
    if water:
        a_w = absorption.a_water(wave, data='IOCCG')
    else:
        a_w = 0
    ax_a = plt.subplot(gs[2])
    def plot_spec(ax):
        ax.plot(wave, orig+a_w, 'ko', label='True', zorder=1)
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

    ax_a.text(xpos, ypos2,  '(b)', color='k',
            transform=ax_a.transAxes,
              fontsize=18, ha='left')

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

# ############################################################
def fig_mcmc_pace(pace_file:str=None, outroot='fig_mcmc_pace', 
                  decomps:str=('nmf','nmf'),
        hidden_list:list=[512, 512, 512, 256], dataset:str='L23', 
        use_quick:bool=False,
        X:int=4, Y:int=0, show_zoom:bool=False, 
        in_Ncomps=None,
        perc:int=None, abs_sig:float=None,
        show_giop:bool=False,
        show_detailed_giop:bool=False,
        wvmnx:tuple=None, show_NMF:bool=False,
        water:bool=False, in_idx:int=0, use_reconstruct:bool=False,
        chain_file:str=None, in_log10:bool=False,
        test:bool=False, true_obs_only:bool=False,
        true_only:bool=False):

    if in_Ncomps is not None:
        Ncomps = in_Ncomps

    # Load chains
    if chain_file is None:
        chain_file = inf_io.l23_chains_filename(edict, 
                                            perc if perc is not None else int(abs_sig), 
                                            test=test)
    d_chains = inf_io.load_chains(chain_file)

    # Load data
    ihop_file = os.path.join(os.getenv('OS_COLOR'), 'data', 
                         'PACE', 'early', pace_file)
    xds = xarray.open_dataset(ihop_file)
    x,y = d_chains['pace_idx'][in_idx]
    pace_Rrs = xds.Rrs.data[x,y,:]
    #spec_err = xds.Rrs_unc.data[x,y,:]

    # GIOP?
    if show_giop:
        giop_file = os.path.join(os.getenv('OS_COLOR'), 'data', 
                         'PACE', 'early', pace_file.replace('AOP', 'IOP').replace('_IHOP', ''))
        xds_giop, _ = pace_io.load_iop_l2(giop_file)
        # Unpack
        pace_a = xds_giop.a.data[x,y,:]
        pace_bb = xds_giop.bb.data[x,y,:]
        pace_aph = np.maximum(xds_giop.aph.data[x,y,:], 0.)
        pace_adgN = np.maximum(xds_giop.adg_442.data[x,y], 0.)
        pace_adgS = np.maximum(xds_giop.adg_s.data[x,y], 0.)
        pace_iop_wl = xds_giop.wavelength.data

    # Emulator
    edict = emu_io.set_emulator_dict(dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)
    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    # NMF
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(decomps, Ncomps)

    if use_reconstruct:
        # Reconstruct
        items = reconstruct.one_spectrum(in_idx, ab, Chl, d_chains, 
                                     d_a, d_bb, 
                                     emulator, decomps, Ncomps,
                                     in_log10=in_log10)
        idx, orig, a_mean, a_std, a_iop, obs_Rs,\
            pred_Rs, std_pred, NN_Rs, allY, wave,\
            orig_bb, bb_mean, bb_std, a_nmf, bb_nmf = items
        print(f"L23 index = {idx}")
    else:
        priors = None
        if in_log10:
            priors = {}
            priors['use_log_ab'] = True

        recon_file = os.path.join(
            '../Analysis/',
            os.path.basename(fitting_io.l23_chains_filename(
            edict, abs_sig, priors=priors).replace('fit', 'recon')))
        d_recon = np.load(recon_file)
        idx = np.where(d_recon['idx'] == in_idx)[0][0]

        # Unpack here
        wave = d_a['wave']
        orig = d_a['spec'][in_idx]
        a_mean = d_recon['fit_a_mean'][idx]
        a_std = d_recon['fit_a_std'][idx]
        orig_bb = d_bb['spec'][in_idx]
        bb_mean = d_recon['fit_bb_mean'][idx]
        bb_std = d_recon['fit_bb_std'][idx]

        obs_Rs = d_chains['obs_Rs']
        pred_Rs = d_recon['fit_Rrs'][idx]
        std_pred = 0.01

    # Outfile
    outfile = outroot + f'_{idx}.png'
    if water:
        outfile = outfile.replace('.png', '_water.png')
        # Load training data
        d_train = load_rs.loisel23_rs(X=X, Y=Y)


    # #########################################################
    # Plot the solution
    lgsz = 14.

    fig = plt.figure(figsize=(10,6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    
    xpos, ypos, ypos2 = 0.05, 0.10, 0.10

    # #########################################################
    # a
    if water:
        a_w = absorption.a_water(wave, data='IOCCG')
    else:
        a_w = 0
    ax_a = plt.subplot(gs[2])

    # NMF
    ax_a.plot(wave, a_mean+a_w, 'r-', label='Retrieval')
    if show_NMF:
        ax_a.plot(wave, a_nmf+a_w, 'r:', label='Real Recon')
    ax_a.fill_between(wave, a_w+a_mean-a_std, a_w+a_mean+a_std, 
            color='r', alpha=0.5, label='Uncertainty') 

    #GIOP
    if show_giop:
        pace_aw = absorption.a_water(pace_iop_wl, data='IOCCG')
        ax_a.plot(pace_iop_wl, pace_a-pace_aw, 'ko', label='GIOP', zorder=1)
        if show_detailed_giop:
            # aph
            ax_a.plot(pace_iop_wl, pace_aph, '*', color='green',
                    label='GIOP, aph', zorder=2, ms=3)
            # adg
            pace_adg = pace_adgN * np.exp(-1*pace_adgS*(pace_iop_wl-442.))
            ax_a.plot(pace_iop_wl, pace_adg, '*', color='cyan',
                    label='GIOP, adg', zorder=2, ms=3)
            # NMF
            if Ncomps[0] == 2:
                chains = d_chains['chains'][in_idx]
                med_dg = np.median(chains[...,0])
                med_ph = np.median(chains[...,1])
                if in_log10:
                    med_dg = 10**med_dg
                    med_ph = 10**med_ph
                #
                ax_a.plot(wave, d_a['M'][0]*med_dg, ':', color='cyan')#, label='IHOP adg')
                ax_a.plot(wave, d_a['M'][1]*med_ph, ':', color='green')#, label='IHOP aph')
    #ax_a.set_xlabel('Wavelength (nm)')
    if water:
        ax_a.set_ylabel(r'$a(\lambda) \; [{\rm m}^{-1}]$')
    else:
        ax_a.set_ylabel(r'$a_{\rm nw}(\lambda) \; [{\rm m}^{-1}]$')

    # Horizontal line at 0
    ax_a.axhline(0., color='gray', ls='--')

    ax_a.text(xpos, ypos2,  '(b)', color='k',
            transform=ax_a.transAxes,
              fontsize=18, ha='left')

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
    # NMF
    ax_bb.plot(wave, bb_w+bb_mean, 'g-', label='Retrieval')
    if show_NMF:
        ax_bb.plot(wave, bb_w+bb_nmf, 'g:', label='True NMF')
    ax_bb.fill_between(wave, bb_w+bb_mean-bb_std, bb_w+bb_mean+bb_std, 
            color='g', alpha=0.5, label='Uncertainty') 
    # GIOP
    if show_giop:
        _, _, pace_b_w = scattering.betasw_ZHH2009(pace_iop_wl, 24., 
                                              np.atleast_1d(0.), 35.)
        pace_bb_w = pace_b_w / 2.
        #ax_bb.plot(pace_iop_wl, pace_bb-pace_bb_w, 'ko', label='True')
        ax_bb.plot(pace_iop_wl, pace_bb, 'ko', label='GIOP')
        #embed(header='fig_mcmc_pace 1158')

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
    #ax_R.plot(wave, pace_Rrs, 'kx', label='True')
    if true_only:
        pass
    elif use_quick:
        ax_R.plot(wave, obs_Rs[0], 'bs', label='Observed')
    else:
        ax_R.plot(wave, obs_Rs[in_idx], 'bs', label='Observed')
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

    # Gray out unanalyzed region
    ymin, ymax = ax_R.get_ylim()
    ax_R.fill_between([350, 380], ymin, ymax, color='black', alpha=0.7)
    ax_R.fill_between([700, 750], ymin, ymax, color='black', alpha=0.7)

    
    # axes
    for ss, ax in enumerate([ax_a, ax_R, ax_bb]):
        plotting.set_fontsize(ax, 14)
        if ss != 1:
            ax.set_xlabel('Wavelength (nm)')

    #plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")



def fig_corner(decomps:tuple, outroot:str='fig_corner', 
        hidden_list:list=[512, 512, 512, 256], dataset:str='L23', 
        chop_burn:int=-3000, perc:int=None, abs_sig:float=None,
        in_Ncomps=None, no_labels:bool=False,
        include_chl:bool=True,
        chain_file:str=None, in_log10:bool=False,
        X:int=4, Y:int=0, in_idx:int=0):

    if in_Ncomps is not None:
        Ncomps = in_Ncomps

    # Load
    edict = emu_io.set_emulator_dict(dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=include_chl, X=X, Y=Y)

    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(decomps, Ncomps)

    emulator, e_file = emu_io.load_emulator_from_dict(edict)

    priors = None
    if in_log10:
        priors = {}
        priors['use_log_ab'] = True

    if chain_file is None:
        chain_file = inf_io.l23_chains_filename(
            edict, abs_sig, priors=priors)
        #perc if perc is not None else int(abs_sig), test=test)
    d_chains = inf_io.load_chains(chain_file)

    chains = d_chains['chains'][in_idx]
    # Hack for bspline
    if decomps[0] == 'bsp':
        buff = 5
    else:
        buff = 0
    add_Chl = 1 if edict['include_chl'] else 0
    coeff = chains[chop_burn:, :, :].reshape(
        -1,Ncomps[0]+Ncomps[1]+add_Chl+buff)

    if in_log10:
        coeff = 10**coeff

    #embed(header='919 of figs')

    idx = d_chains['idx'][in_idx]
    # Outfile
    outfile = outroot + f'_{idx}.png'

    if edict['include_chl']:
        truths = np.concatenate((ab[idx], Chl[idx].reshape(1,)))
    else:
        truths = ab[idx]
    #if in_log10:
    #    truths = np.log10(truths)

    #if no_labels:
    #    clbls = None

    fig = corner.corner(
        coeff, labels=clbls,
        label_kwargs={'fontsize':17},
        color='k',
        #axes_scale='log',
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    # Compare answers
    median_coeff = np.median(coeff, axis=0)

    print(f"True a: {ab[idx, :Ncomps[0]]}")
    print(f"Fitted a: {median_coeff[:Ncomps[0]]}")
    print('---')
    print(f"True b: {ab[idx, Ncomps[0]:]}")
    print(f"Fitted b: {median_coeff[Ncomps[0]:-1]}")
    print('---')
    print(f"True Chl: {Chl[idx]}")
    print(f"Fitted Chl: {median_coeff[-1]}")



# ############################################################
# ############################################################
def fig_rmse_vs_sig(outroot:str='fig_rmse_vs_sig', 
                    hidden_list:list=[512, 512, 512, 256], dataset:str='L23', 
                    X:int=4, Y:int=0, debug:bool=False,
                    decomp:str='pca', chop_burn:int=-3000):

    edict = emu_io.set_emulator_dict(dataset, decomp, Ncomp, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)
    # Prep
    if decomp == 'pca':
        rfunc = decompose.reconstruct_pca
    elif decomp == 'nmf':
        rfunc = decompose.reconstruct_nmf

    outfile = outroot + f'_{decomp}.png'

    all_l23_rmse = []
    all_l23_sig = []
    #all_perc = [0, 5, 10, 15, 20]
    #all_perc = [0, 5, 10, 20]
    all_perc = [1]
    for perc in all_perc:
        perc = None
        abs_sig = 1.
        print(f"Working on: {perc}%")
        # L23
        ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)
        chain_file = inf_io.l23_chains_filename(
            edict, perc if perc is not None else int(abs_sig)) 
        d_chains = inf_io.load_chains(chain_file)
        chains = d_chains['chains']
        l23_idx = d_chains['idx']
        
        nobj = 100 if debug else chains.shape[0]

        wave = d_a['wave']
        dev = np.zeros((nobj, wave.size))
        mcmc_std = np.zeros((nobj, wave.size))
        for in_idx in range(nobj):
            idx = l23_idx[in_idx]
            ichains = chains[in_idx]
            Y = ichains[chop_burn:, :, 0:Ncomp[0]].reshape(-1,Ncomp[0])
            orig, a_recon = rfunc(Y, d_a, idx)
            _, a_nmf = rfunc(d_a['coeff'][idx], d_a, idx)
            a_mean = np.mean(a_recon, axis=0)
            #
            mcmc_std[in_idx,:] = np.std(a_recon, axis=0) / orig
            #dev[in_idx,:] = (orig - a_mean)/orig
            dev[in_idx,:] = (a_nmf - a_mean)/a_nmf
            #_, a_pca = rfunc(ab[idx][:ncomp], d_a, idx)
            if debug:
                embed(header='535 of figs')
        # RMSE
        #rmse_l23 = np.sqrt(np.mean(dev**2, axis=0))
        rmse_l23 = np.sqrt(np.mean(dev**2, axis=0))
        # Save
        all_l23_rmse.append(rmse_l23)
        all_l23_sig.append(np.mean(mcmc_std, axis=0))

        if debug:
            embed(header='544 of figs')
        

    # Plot
    figsize=(8,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # #####################################################
    # Absolute
    #ax_abs = plt.subplot(gs[0:2])
    all_ax = []
    for ss, rmse_l23 in enumerate(all_l23_rmse):
        ax= plt.subplot(gs[ss])

        ax.plot(wave, rmse_l23, 'o', label=f'{all_perc[ss]}%: RMSE')
        ax.plot(wave, all_l23_sig[ss], '*', label=f'{all_perc[ss]}%: MCMC std')

        ax.set_ylabel(r'Relative error in $a_{\rm nw}(\lambda)$')# (m$^{-1}$)')
    #ax_abs.tick_params(labelbottom=False)  # Hide x-axis labels
        all_ax.append(ax)
        ax.legend(fontsize=10)


    #ax.set_xlim(1., 10)
    #ax.set_yscale('log')

    # Finish
    for ss, ax in enumerate(all_ax):
        plotting.set_fontsize(ax, 13)
        ax.set_xlabel('Wavelength [nm]')
        # Grid
        ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
        # Limits
        ax.set_ylim(0, 0.2)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ############################################################
def fig_decompose_error(outfile:str='fig_decompose_error.png', 
                    hidden_list:list=[512, 512, 512, 256], dataset:str='L23', 
                    X:int=4, Y:int=0, debug:bool=False,
                    decomp:str='nmf', chop_burn:int=-3000,
                    perc:float=None, abs_sig:float=None):

    # Prep
    edict = emu_io.set_emulator_dict(dataset, decomp, Ncomp, 'Rrs',
        'dense', hidden_list=hidden_list, include_chl=True, X=X, Y=Y)
    # Prep
    if decomp == 'pca':
        rfunc = decompose.reconstruct_pca
    elif decomp == 'nmf':
        rfunc = decompose.reconstruct_nmf

    # Load the chains
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_decomposition(decomp, Ncomp)
    chain_file = inf_io.l23_chains_filename(
            edict, perc if perc is not None else int(abs_sig)) 
    d_chains = inf_io.load_chains(chain_file)

    # Run stats
    chains = d_chains['chains']
    Nspec = chains.shape[0]
    l23_idx = d_chains['idx']
    # Quick chk on index
    srt = np.argsort(l23_idx)
    assert np.all(srt == l23_idx)

    chains = chains[:, chop_burn:, :, :].reshape(
        chains.shape[0], -1, chains.shape[-1])

    comp_med = np.median(chains, axis=1)

    # Offsets
    rdiff_comp = np.zeros((Nspec, chains.shape[-1]))
    # a, b
    rdiff_comp[:, :ab.shape[1]] = (comp_med[:,:-1] - ab)/ab
    # Chl
    rdiff_comp[:, -1] = (comp_med[:,-1] - Chl)/Chl

    # Percentiles
    #comp_per_16 = np.percentile(chains, 16, axis=1)
    #embed(header='fig_decompose_error 675')

    # Plot
    figsize=(6,12)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(4,2)

    for ss, lbl in enumerate(clbls): 

        # Plot the coefficients
        ax = plt.subplot(gs[ss])
        ax.hist(rdiff_comp[:,ss], bins=70, histtype='step', color='k', lw=2)

        ax.set_xlim(-0.5, 0.5)

        # Labels
        ax.set_xlabel(f'Error in {lbl}')
        ax.set_ylabel('Count')

        # Stats
        bias = np.median(rdiff_comp[:,ss])
        std = np.std(rdiff_comp[:,ss])
        print(std)
        lsz = 15.
        ax.text(0.95, 0.90, f'Bias: {bias:.3f}',
                color='k',
            transform=ax.transAxes,
            fontsize=lsz, ha='right')
        ax.text(0.95, 0.80, f'RMS: {std:.3f}',
                color='k',
            transform=ax.transAxes,
            fontsize=lsz, ha='right')

        plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Decomposition
    if flg & (2**0):
        #fig_emulator_rmse('L23_PCA')
        #fig_basis_functions(('nmf', 'nmf'), in_Ncomps=(2,2),
        #                    outfile='fig_basis_functions_nmf_22.png')
        #fig_basis_functions(('nmf', 'nmf'), in_Ncomps=(3,2),
        #                    outfile='fig_basis_functions_nmf_32.png')
        #fig_basis_functions(('pca', 'pca'),
        #                    outfile='fig_basis_functions_pca.png')
        #fig_basis_functions(('npca', 'npca'), in_Ncomps=(4,2),
        #                    outfile='fig_basis_functions_npca.png')
        fig_basis_functions(('hyb', 'nmf'), in_Ncomps=(4,2),
                            outfile='fig_basis_functions_hybnmf_22.png')

    # Emulator RMSE
    if flg & (2**1):
        # NMF
        #fig_emulator_rmse('L23', 3, [512, 512, 512, 256],
        #                  outfile='fig_emulator_rmse_3.png')
        #fig_emulator_rmse('L23', 4, [512, 512, 512, 256],
        #                  log_rrmse=True)
        #fig_emulator_rmse('L23', (4,3), [512, 512, 512, 256],
        #                  log_rrmse=True)
        #fig_emulator_rmse('L23', (4,2), [512, 512, 512, 256],
        #                  log_rrmse=True)
        #fig_emulator_rmse(['L23_NMF', 'L23_PCA'], [3, 3])
        #fig_emulator_rmse('L23', (3,2), [512, 512, 512, 256],
        #                  ('nmf', 'nmf'), log_rrmse=True, 
        #                  outfile='fig_emulator_rmse_nmf_32.png')
        #fig_emulator_rmse('L23', (2,2), [512, 512, 512, 256],
        #                  ('nmf', 'nmf'), log_rrmse=True, 
        #                  outfile='fig_emulator_rmse_nmf_22.png')
        # BSP
        #fig_emulator_rmse('L23', (10,2), [512, 512, 512, 256],
        #                  ('bsp', 'nmf'), log_rrmse=True, 
        #                  outfile='fig_emulator_rmse_bsp_102.png')
        # PCA
        #fig_emulator_rmse('L23', (4,2), [512, 512, 512, 256],
        #                  ('pca', 'pca'), log_rrmse=True, 
        #                  preproc_Rs='lin-5', 
        #                  outfile='fig_emulator_rmse_pca_lin-5.png') 
        # INT
        #fig_emulator_rmse('L23', (40,2), [512, 512, 512, 256],
        #                  ('int', 'nmf'), log_rrmse=True,
        #                  outfile='fig_emulator_rmse_intnmf.png') 
        # HYB
        fig_emulator_rmse('L23', (4,2), [512, 512, 512, 256],
                          ('hyb', 'nmf'), log_rrmse=True, 
                          include_chl=False,
                          outfile='fig_emulator_rmse_hyb_42.png')

    # MCMC evaluation
    if flg & (2**2):
        #fig_mcmc_fit(test=True, perc=10)
        #fig_mcmc_fit(test=True, abs_sig=2.)
        #fig_mcmc_fit(test=True, abs_sig=2., water=True)
        #fig_mcmc_fit(test=True, abs_sig=2., water=True)
        #fig_mcmc_fit(abs_sig=5., in_idx=275) # Turbid
        #fig_mcmc_fit(abs_sig=5., in_idx=2663,
        # in_Ncomps=(2,2), in_log10=True) # Minimum
        #fig_mcmc_fit(abs_sig=5., in_idx=170) # Median
        #fig_mcmc_fit(abs_sig=5., in_idx=180) # Median
        #fig_mcmc_fit(abs_sig=2., in_idx=2949, in_log10=True, in_Ncomps=(4,2)) # Maximum absorption
        #fig_mcmc_fit(abs_sig=1., in_idx=0)#, wvmnx=[500, 600.]) # Clear
        #fig_mcmc_fit(abs_sig=1., in_idx=99) # Clear

        #fig_mcmc_fit(outroot='fig_mcmc_fit_true',
        #             test=True, abs_sig=2., true_only=True)
        #fig_mcmc_fit(outroot='fig_mcmc_fit_trueobs',
        #             test=True, abs_sig=2., true_obs_only=True)
        #
        #fig_mcmc_fit(abs_sig=5., in_idx=1, use_reconstruct=True, in_log10=True, # Median
                   #chain_file='../../../builds/fits/Fits/L23/fit_Rs05_L23_X4_Y0_nmfnmf_42_chl_Rrs_dense_512_512_512_256.npz')
        #fig_mcmc_fit(abs_sig=2., in_idx=1, use_reconstruct=True, in_log10=False, 
        #             in_Ncomps=(3,2), 
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs02_L23_X4_Y0_nmfnmf_32_chl_Rrs_dense_512_512_512_256.npz')
        #fig_mcmc_fit(abs_sig=5., in_idx=0, use_reconstruct=True, 
        #             in_log10=True, in_Ncomps=(2,2), 
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs05_L23_X4_Y0_nmfnmf_22_chl_Rrs_dense_512_512_512_256_logab.npz')
        #fig_mcmc_fit(abs_sig=2., in_idx=1, use_reconstruct=True, in_log10=True, in_Ncomps=(3,2), 
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs02_L23_X4_Y0_nmfnmf_32_chl_Rrs_dense_512_512_512_256_logab.npz')
        #fig_mcmc_fit(abs_sig=1., in_idx=1, use_reconstruct=True, in_log10=True, # Median
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs01_L23_X4_Y0_nmfnmf_42_chl_Rrs_dense_512_512_512_256.npz')
        #fig_mcmc_fit(abs_sig=None, in_idx=1, decomps=('bsp', 'nmf'), 
        #             use_reconstruct=True, in_Ncomps=(10,2), 
        #           chain_file='../../../builds/fits/Fits/L23/fitN_Rs01_L23_X4_Y0_bspnmf_102_chl_Rrs_dense_512_512_512_256.npz')
        #fig_mcmc_fit(abs_sig='PACE', in_idx=0, decomps=('nmf', 'nmf'), 
        #             use_reconstruct=True, in_Ncomps=(2,2), in_log10=True,
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs99_L23_X4_Y0_nmfnmf_22_chl_Rrs_dense_512_512_512_256_logab.npz')
        #fig_mcmc_fit(abs_sig='PACE_CORR', in_idx=1, decomps=('nmf', 'nmf'), 
        #             use_reconstruct=True, in_Ncomps=(2,2), in_log10=True,
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs98_L23_X4_Y0_nmfnmf_22_chl_Rrs_dense_512_512_512_256_logab.npz')
        fig_mcmc_fit(abs_sig=None, in_idx=1, decomps=('hyb', 'nmf'), 
                     use_reconstruct=True, in_Ncomps=(4,2), in_log10=True,
                     use_Chl=False,
                   chain_file='../../../builds/fits/Fits/L23/fitN_Rs01_L23_X4_Y0_hybnmf_42_Rrs_dense_512_512_512_256_logab.npz')

    # L23 IHOP performance vs. perc error
    if flg & (2**22):
        #fig_rmse_vs_sig()
        fig_rmse_vs_sig(decomp='nmf')#, debug=True)

    # L23 IHOP performance vs. perc error
    if flg & (2**23):
        #fig_corner(test=True, perc=10)
        #fig_corner(test=True, abs_sig=2.)
        #fig_corner(abs_sig=1., in_idx=275) # Turbid
        #fig_corner(abs_sig=1., in_idx=0) # Clear
        #fig_corner(('pca', 'pca'), abs_sig=1., in_idx=0) # 
        #fig_corner(('nmf', 'nmf'), abs_sig=None, in_idx=2663) # Minimum
        #fig_corner(('nmf', 'nmf'), abs_sig=None, in_idx=2949) # Maximum
        #fig_corner(('nmf', 'nmf'), abs_sig=2., in_idx=170, in_Ncomps=(4,2)) # Maximum
        #fig_corner(('nmf', 'nmf'), abs_sig=5., in_idx=180) #
        #fig_corner(('nmf', 'nmf'), abs_sig=5., in_idx=170) # 
        #
        #fig_corner(('nmf', 'nmf'), abs_sig=5., in_idx=1, in_log10=True,
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs05_L23_X4_Y0_nmfnmf_42_chl_Rrs_dense_512_512_512_256.npz')
        #fig_corner(('nmf', 'nmf'), abs_sig=2., in_idx=1, in_Ncomps=(3,2), #in_log10=True,
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs02_L23_X4_Y0_nmfnmf_32_chl_Rrs_dense_512_512_512_256.npz')
        #fig_corner(('nmf', 'nmf'), abs_sig=2., in_idx=1, in_Ncomps=(3,2), in_log10=True,
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs02_L23_X4_Y0_nmfnmf_32_chl_Rrs_dense_512_512_512_256_logab.npz')
        #fig_corner(('nmf', 'nmf'), abs_sig=1., in_idx=1, in_log10=True,
        #           chain_file='../../../builds/fits/Fits/L23/fit_Rs01_L23_X4_Y0_nmfnmf_42_chl_Rrs_dense_512_512_512_256.npz')
        #fig_corner(('bsp', 'nmf'), abs_sig=None, no_labels=True,
        #           in_idx=0, in_Ncomps=(10,2),
        #           chain_file='../../../builds/fits/Fits/L23/fitN_Rs01_L23_X4_Y0_bspnmf_102_chl_Rrs_dense_512_512_512_256.npz')
        #fig_corner(('nmf', 'nmf'), abs_sig=5., in_idx=2663, 
        #           in_Ncomps=(2,2), in_log10=True)
        fig_corner(('hyb', 'nmf'), abs_sig=None, no_labels=True,
                   in_log10=True,
                   in_idx=0, in_Ncomps=(4,2), include_chl=False,
                   chain_file='../../../builds/fits/Fits/L23/fitN_Rs01_L23_X4_Y0_hybnmf_42_Rrs_dense_512_512_512_256_logab.npz')

    # 
    if flg & (2**24):
        fig_nmf_corner()

    # Decompose
    if flg & (2**25):
        fig_mcmc_decompose(abs_sig=1., in_idx=0)

    # Decompose
    if flg & (2**26):
        fig_decompose_error(abs_sig=1.)

    # RMSE of Rrs and a
    if flg & (2**27):
        #fig_rmse_Rrs_a(('nmf', 'nmf'), (4,2),'fig_rmse_Rrs_a_nmfnmf.png',
        #               abs_sig=None)
        fig_rmse_Rrs_a(('nmf', 'nmf'), (2,2),'fig_rmse_Rrs_a_nmfnmf_22.png',
                       abs_sig=None)
        #fig_rmse_Rrs_a(('pca', 'pca'), (4,2),'fig_rmse_Rrs_a_pcapca.png',
                      #abs_sig=None)
        #fig_rmse_Rrs_a(('int', 'nmf'), (40,2), 'fig_rmse_Rrs_a_intnmf.png',
        #               abs_sig=None)

    # RMSE of Rrs and a
    if flg & (2**28):
        #fig_rmse_a_error(('nmf', 'nmf'), (4,2), 
        #                 'fig_rmse_a_error_nmfnmf.png', [1, 2., 5.],
        #                 show_bias=True)
        fig_rmse_a_error(('nmf', 'nmf'), (2,2), 
                         'fig_rmse_a_error_nmfnmf.png', 
                         [2., 5.],
                         use_log_ab=True,
                         show_bias=False)
        #fig_rmse_a_error(('pca', 'pca'), (4,2), 
        #                 'fig_rmse_a_error_pcapca.png', [1., 2., 5.])

    # MAD, RMSE of a
    if flg & (2**29):
        #fig_a_examples(('nmf', 'nmf'), (4,2), 
        #                 'fig_a_examples_nmf.png', [1.0, 2., 5.],
        #                 show_noiseless_error=True,
        #                 show_LS2=True)
        #fig_a_examples(('nmf', 'nmf'), (4,2), 
        #                 'fig_a_examples_nmf.png', [2.],
        #                 use_log_ab=True,
        #                 show_noiseless_error=True,
        #                 show_absig_error=[2.],
        #                 show_LS2=True)
        #fig_a_examples(('pca', 'pca'), (4,2), 
        #                 'fig_a_examples.png', [1., 5.], skip_fits=True)
        #fig_a_examples(('npca', 'pca'), (4,2), 
        #                 'fig_a_examples_npca.png', [1., 5.],
        #                 skip_fits=True)
        #fig_a_examples(('pca', 'pca'), (4,2), 
        #                 'fig_a_examples.png', [2.]) 
                         #[1., 5.], skip_fits=True)
        #fig_a_examples(('npca', 'pca'), (4,2), 
        #                 'fig_a_examples_npca.png', [1., 5.],
        #                 skip_fits=True)
        fig_a_examples(('hyb', 'nmf'), (4,2), 
                         'fig_a_examples_hyb.png', [2., 99],
                         include_chl=False,
                         use_log_ab=True,
                         show_noiseless_error=True,
                         show_absig_error=[2., 99])

    # RMSE of a
    if flg & (2**30):
        # Decomposed only
        #fig_summary_a(
        #    [
        #        (-1, 'nmf', 2, 'logab', 'use_n', '-'),
        #        (-1, 'nmf', 4, 'logab', 'use_n', '-'),
        #    ], 'fig_summary_decompose.png') 

        # Noiseless only
        #fig_summary_a(
        #    [
        #        (-1, 'nmf', 2, 'logab', 'use_n', ':'),
        #        (-1, 'nmf', 4, None, 'use_n', ':'),   # FIX THIS!
        #        (None, 'nmf', 2, 'logab', 'use_n', '-'),
        #        (None, 'nmf', 4, None, 'use_n', '-'), # FIX THIS!!!!
        #    ], 'fig_summary_noiseless.png') 

        # abs_sig=2.
        fig_summary_a(
            [
                (-1, 'nmf', 2, 'logab', 'use_n', ':'),
                (-1, 'nmf', 3, 'logab', 'use_n', ':'),   
                (-1, 'nmf', 4, 'logab', 'use_n', ':'),   
                (-1, 'hyb', 4, 'logab', 'use_n', ':'),   
                (2., 'nmf', 2, 'logab', 'use_n', '-'),
                (2., 'nmf', 3, 'logab', 'use_n', '-'),
                (2., 'nmf', 4, 'logab', 'use_n', '-'), 
            ], 'fig_summary_abssig2.png') 

        # 2,2 with PACE
        #fig_summary_a(
        #    [
        #        (-1, 'nmf', 2, 'logab', 'use_n', ':'),
        #        (2., 'nmf', 2, 'logab', 'use_n', '-'),
        #        ('PACE', 'nmf', 2, 'logab', 'blue', '-'),
        #        ('PACE_CORR', 'nmf', 2, 'logab', 'red', '-'),
        #    ], 'fig_summary_PACE22.png') 

    # MCMC evaluation
    if flg & (2**31):
        fig_mcmc_pace('PACE_OCI.20240413T175656.L2.OC_AOP.V1_0_0.NRT_IHOP.nc',
            abs_sig='PACE_TRUNC', in_idx=0, decomps=('nmf', 'nmf'), 
            use_reconstruct=True, in_Ncomps=(2,2), in_log10=True,
            show_giop=True,
            chain_file='../../../builds/fits/Fits/L23/fit_Rs97_PACE_OCI.20240413T175656.L2_NN_22_chl_logab.npz')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg += 2 ** 0  # Basis functions of the decomposition
        #flg += 2 ** 1  # RMSE of emulators
        #flg += 2 ** 2  # Single MCMC fit (example)

        #flg += 2 ** 22  # RMSE of L23 fits
        #flg += 2 ** 23  # Fit corner
        #flg += 2 ** 24  # NMF corner plots (decomposition only)

        #flg += 2 ** 26  # Decompose error

        #flg += 2 ** 27  # RMSE on Rrs and a
        #flg += 2 ** 28  # RMSE on a vs. abs_sig
        flg += 2 ** 29  # Examples

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance

         
        #flg += 2 ** 30  # Summary
        #flg += 2 ** 31  # PACE MCMC
        
    else:
        flg = sys.argv[1]

    main(flg)