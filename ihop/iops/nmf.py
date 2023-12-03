""" NMF analysis of IOPs """

import os
from importlib import resources

import numpy as np

from oceancolor.iop import cross
from ihop.hydrolight import loisel23

def prep_loisel23(iop:str, min_wv:float=400., sigma:float=0.05,
                  X:int=4, Y:int=0):
    """ Prep L23 data for NMF analysis

    Args:
        iop (str): IOP to use
        min_wv (float, optional): Minimum wavelength for analysis. Defaults to 400..
        sigma (float, optional): Error to use. Defaults to 0.05.
        X (int, optional): _description_. Defaults to 4.
        Y (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    # Load
    ds = loisel23.load_ds(X, Y)

    # Unpack and cut
    spec = ds[iop].data
    wave = ds.Lambda.data 
    Rs = ds.Rrs.data

    cut = wave >= min_wv
    spec = spec[:,cut]
    wave = wave[cut]
    Rs = Rs[:,cut]

    # Remove water
    if iop == 'a':
        a_w = cross.a_water(wave, data='IOCCG')
        spec_nw = spec - np.outer(np.ones(3320), a_w)
    else:
        spec_nw = spec

    # Reshape
    spec_nw = np.reshape(spec_nw, (spec_nw.shape[0], 
                     spec_nw.shape[1], 1))

    # Build mask and error
    mask = (spec_nw >= 0.).astype(int)
    err = np.ones_like(mask)*sigma

    # Return
    return spec_nw, mask, err, wave, Rs

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

def single_exp_var(nmf_fit:str, N_NMF:int, iop:str,
                   col_wise=False):
    data = load_nmf(nmf_fit, N_NMF=N_NMF, iop=iop)
    
    # transpose data matrix if it is col_wise.
    if col_wise:
        data = np.transpose(data)

    # Calculate covariance matrix
    cov_data = np.cov(data)

    # Eigen decomposition 
    values, vectors = np.linalg.eig(cov_data)

    # Get explained variances
    explained_variances = values / np.sum(values)
    return explained_variances

def exp_var():

    print("Explained Variance Computation Starts.")

    # bb
    exp_var_list = []
    index_list = []
    for i in range(2, 11):
        data_path = f"../data/L23_NMF_bb_{i}_coef.npy"
        exp_var_i = exp_var(data_path)
        exp_var_list.append(exp_var_i)
        index_list.append(f"bb_{i}")
    result_dict = {
        "index_list": index_list,
        "exp_var": exp_var_list,
    }
    df_exp_var = pd.DataFrame(result_dict)
    df_exp_var.set_index("index_list", inplace=True)
    file_save = "../data/exp_var_coef_L23_NMF_bb.csv"
    df_exp_var.to_csv(file_save, header=False)

    # a
    exp_var_list = []
    index_list = []
    for i in range(2, 11):
        data_path = f"../data/L23_NMF_a_{i}_coef.npy"
        exp_var_i = exp_var(data_path)
        exp_var_list.append(exp_var_i)
        index_list.append(f"a_{i}")
    result_dict = {
        "index_list": index_list,
        "exp_var": exp_var_list,
    }
    df_exp_var = pd.DataFrame(result_dict)
    df_exp_var.set_index("index_list", inplace=True)
    file_save = "../data/exp_var_coef_L23_NMF_a.csv"
    df_exp_var.to_csv(file_save, header=False)    
    print("Computation Ends Successfully!")
    
    
    