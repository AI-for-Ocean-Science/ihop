""" NMF analysis of IOPs """

import os
from importlib import resources

import numpy as np
from importlib import resources

from oceancolor.iop import cross
from ihop.hydrolight import loisel23

def load_loisel_2023(N_NMF:int=4):
    """ Load the NMF-based parameterization of IOPs from Loisel 2023

    Returns:
        tuple: 
            - **ab** (*np.ndarray*) -- PCA coefficients
            - **Rs** (*np.ndarray*) -- Rrs values
            - **d_a** (*dict*) -- dict of PCA 
            - **d_bb** (*dict*) -- dict of PCA
    """

    # Load up data
    l23_path = os.path.join(resources.files('ihop'),
                            'data', 'NMF')
    l23_a_file = os.path.join(l23_path, f'L23_NMF_a_{N_NMF}.npz')
    l23_bb_file = os.path.join(l23_path, f'L23_NMF_bb_{N_NMF}.npz')

    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)
    nparam = d_a['coeff'].shape[1]+d_bb['coeff'].shape[1]
    ab = np.zeros((d_a['coeff'].shape[0], nparam))
    ab[:,0:d_a['coeff'].shape[1]] = d_a['coeff']
    ab[:,d_a['coeff'].shape[1]:] = d_bb['coeff']

    Rs = d_a['Rs']

    # Return
    return ab, Rs, d_a, d_bb



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

def reconstruct(Y, nmf_dict, idx):
    # Grab the orginal
    orig = nmf_dict['spec'][idx]

    # Reconstruct
    recon = np.dot(Y, nmf_dict['M']) 

    # Return
    return orig, recon 

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
        _, evar_i = evar_computation("L23", i, iop)
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

def evar_plot(save_path, iop:str='a'):
    print("Computation Starts.")
    evar_list, index_list = [], []
    for i in range(2, 11):
        _, evar_i = evar_computation("L23", i, iop)
        evar_list.append(evar_i)
        index_list.append(i)
    plt.figure(figsize=(10, 8))
    plt.plot(index_list, evar_list, '-o', color='blue')
    plt.axhline(y = 1.0, color ="red", linestyle ="--") 
    plt.xlabel("Dim of Feature space", fontsize=15)
    plt.ylabel("Explained Variance", fontsize=15)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Plot Ends Successfully!")

if __name__ == "__main__":
    path = os.path.join(
        resources.files('ihop'), 
        'data', 'NMF'
    )
    save_path_a = os.path.join(path, f'evar_L23_NMF_a.csv')
    evar_for_all(save_path_a, 'a')
    save_path_bb = os.path.join(path, f'evar_L23_NMF_bb.csv')
    evar_for_all(save_path_bb, 'bb')
    plot_path_a = os.path.join(path, f'evar_L23_NMF_a.pdf')
    evar_plot(plot_path_a, iop='a')
    plot_path_bb = os.path.join(path, f'evar_L23_NMF_bb.pdf')
    evar_plot(plot_path_bb, iop='bb')
