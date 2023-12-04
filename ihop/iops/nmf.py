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
    if nmf_fit == 'L23':
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

def load_nmf_coef(nmf_fit:str, N_NMF:int=None, iop:str='a'):

    # Load
    data_dir = "/home/jovyan/ihop/ihop"
    if nmf_fit == 'L23':
        if N_NMF is None:
            N_NMF = 5
        
        path = os.path.join(data_dir, 'data', 'NMF')
        outroot = os.path.join(path, f'L23_NMF_{iop}_{N_NMF}_coef')
        nmf_file = outroot+'.npy'
        #
        h = np.load(nmf_file)
        print(h.shape)
    else:
        raise IOError("Bad input")

    return h

def load_nmf_comp(nmf_fit:str, N_NMF:int=None, iop:str='a'):

    # Load
    data_dir = "/home/jovyan/ihop/ihop"
    if nmf_fit == 'L23':
        if N_NMF is None:
            N_NMF = 5
        path = os.path.join(data_dir, 'data', 'NMF')
        outroot = os.path.join(path, f'L23_NMF_{iop}_{N_NMF}_comp')
        nmf_file = outroot+'.npy'
        #
        w = np.load(nmf_file)
        print(w.shape)
    else:
        raise IOError("Bad input")

    return w

# To define the explained variance for NMF, we should 
# use the following fomula (https://rdrr.io/cran/NMF/man/rss.html):
# RSS = \sum_{i,j} (v_{ij} - V_{ij})^2
# evar = 1 - \frac{RSS}{\sum_{i,j} v_{ij}^2}
# where, V_{ij} is the target, and v_{ij} is the estimation.

def evar_nmf(v_target:np.array, w:np.array, h:np.array):
    v_est = np.dot(w, h)
    rss = np.sum(np.square(v_est - v_target))
    evar = 1 - rss / np.sum(np.square(v_est))
    return (rss, evar)

def evar_computation(nmf_fit:str, N_NMF:int=None, iop:str='a'):
    d_npz = load_nmf(nmf_fit, N_NMF, iop)
    v_target = d_npz['spec']
    #######################################
    h = d_npz['M']
    w = d_npz['coeff']
    # we think it should be, but given data
    # requires above code
    # h = d_npz['coeff']
    # w = d_npz['M']
    ######################################
    rss, evar = evar_nmf(v_target, w, h)
    return (rss, evar)

def evar_for_all(save_path, iop:str='a'):
    print("Computation Starts.")
    evar_list, index_list = [], []
    for i in range(2, 11):
        _, evar_i = evar_computation("L23", i, "a")
        evar_list.append(evar_i)
        index_list.append(i)
    result_dict = {
        "index_list": index_list,
        "exp_var": exp_var_list,
    }
    df_exp_var = pd.DataFrame(result_dict)
    df_exp_var.set_index("index_list", inplace=True)
    df_exp_var.to_csv(save_path, header=False)    
    print("Computation Ends Successfully!")

if __name__ == "__main__":
    path = os.path.join(resources.files('ihop'), 
                        'data', 'NMF')
    save_path_a = os.path.join(path, f'evar_L23_NMF_a.csv')
    evar_for_all(save_path_a, 'a')
    save_path_bb = os.path.join(path, f'evar_L23_NMF_bb.csv')
    evar_for_all(save_path_bb, 'bb')
