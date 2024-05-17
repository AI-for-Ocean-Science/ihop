""" Methods to calcualte stats """
import os
import numpy as np 

from ihop.emulators import io as emu_io
from ihop.inference import io as fitting_io
from ihop import io as ihop_io

def calc_rmses(d_a, d_recon, a_decomp,
               a_decomposed=None):
    
    a_items = {}
    # #############################
    # a
    tkey = 'spec' if a_decomp == 'nmf' else 'data'

    # Diff
    if a_decomposed is None:
        chain_idx = d_recon['idx']
        a_true = d_a[tkey][chain_idx]
        fit_diff = d_recon['fit_a_mean'] - a_true
    else:
        a_true = a_decomposed
        fit_diff = d_a[tkey] - a_true

    a_fit_RMSE = np.sqrt(np.mean((fit_diff)**2, axis=0))
    a_fit_MAD = np.median(np.abs(fit_diff), axis=0)
    a_fit_bias = np.median(fit_diff, axis=0)

    # Set min
    a_true_min = np.maximum(a_true, 1e-4)
    a_fit_rMAD = np.median(np.abs(fit_diff/a_true_min), axis=0)
    a_fit_rBIAS = np.median(fit_diff/a_true_min, axis=0)
    a_fit_rRMSE = np.sqrt(np.mean((fit_diff/a_true_min)**2, axis=0))

    a_items['a_fit_RMSE'] = a_fit_RMSE
    a_items['a_fit_MAD'] = a_fit_MAD
    a_items['a_fit_BIAS'] = a_fit_bias
    a_items['a_fit_rMAD'] = a_fit_rMAD
    a_items['a_fit_rBIAS'] = a_fit_rBIAS
    a_items['a_fit_rRMSE'] = a_fit_rRMSE

    return a_items

def calc_a_stats(abs_sig:float, 
                 decomps:tuple,
                 Ncomps:tuple,
                 priors:str, 
                 hidden_list:list=[512, 512, 512, 256], 
                 dataset:str='L23', X:int=4, Y:int=0):

    # ######################
    # Load
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        decomps, Ncomps)
    wave = d_a['wave']
    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=True, X=X, Y=Y)

    if priors is not None:
        priors = {}
        if priors == 'logab':
            priors['use_log_ab'] = True

    if abs_sig == -1.:
        # Noiseless
        recon_file = os.path.join(
            '../Analysis/',
            os.path.basename(fitting_io.l23_chains_filename(
            edict, None, priors=priors).replace('fit', 'recon')))
        d_nless = np.load(recon_file)
        a_recons = d_nless['decomp_a']
        return calc_rmses(d_a, None, decomps[0],
                          a_decomposed=a_recons)
