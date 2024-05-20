""" Methods related to fitting PACE data"""
import os

import numpy as np
import xarray

from matplotlib import pyplot as plt

from ihop.pace import prep 
from ihop.inference import io as fitting_io
from ihop.emulators import io as emu_io
from ihop.iops import io as iops_io
from ihop import io as ihop_io

from cnmf import apply as cnmf_apply

# Internal
import fits

from IPython import embed

def prep_one():
    basename = 'PACE_OCI.20240413T175656.L2.OC_AOP.V1_0_0.NRT.nc'
    pfile = os.path.join(os.getenv('OS_COLOR'), 'data', 
                         'PACE', 'early', basename)
    outfile = pfile.replace('.nc', '_IHOP.nc')
    prep.process_l2_for_l23(pfile, outfile, minval=-10., maxval=100.)


def load_one_example(lon:float=-75.5, # W
                    lat:float=34.5): # N
    basename = 'PACE_OCI.20240413T175656.L2.OC_AOP.V1_0_0.NRT_IHOP.nc'
    ihop_file = os.path.join(os.getenv('OS_COLOR'), 'data', 
                         'PACE', 'early', basename)
    #
    xds = xarray.open_dataset(ihop_file)
    idx = np.argmin( (xds.longitude.data-lon)**2 + (xds.latitude.data-lat)**2)
    x,y = np.unravel_index(idx, xds.longitude.shape)

    spec = xds.Rrs.data[x,y,:]
    spec_err = xds.Rrs_unc.data[x,y,:]
    #
    return basename, xds.wavelength.data, spec, spec_err


def load(edict:dict):
    # Load data
    pace_file, pace_wave, pace_Rrs, pace_err = load_one_example()

    # Reshape
    pace_Rrs = pace_Rrs.reshape(1, pace_Rrs.size)
    pace_err = pace_err.reshape(1, pace_err.size)

    # Load emulator
    emulator, e_file = emu_io.load_emulator_from_dict(
        edict, use_s3=True)

    # Return
    return pace_file, pace_wave, pace_Rrs, pace_err, emulator

def find_closest(edict:dict, pace_wave:np.ndarray, 
                 pace_Rrs:np.ndarray, pace_err:np.ndarray, 
                 debug:bool=False):

    # Load the L23 solution and data
    decomp_file = iops_io.loisel23_filename('nmf', 'Rrs', 2, 4, 0)
    d_Rrs = np.load(decomp_file)
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        edict['decomps'], edict['Ncomps'])

    # Fit
    # Insure the right shape
    if pace_Rrs.ndim == 1:
        pace_Rrs = pace_Rrs.reshape(1, pace_Rrs.size)
        pace_err = pace_err.reshape(1, pace_err.size)

    # Avoid bad data and even out 
    ok_wv = (pace_wave > 380.) & (pace_wave < 700.)
    pace_err[:] = 1.
    pace_err[0,ok_wv] = 2.e-4


    # Fit
    nmf_coeff = cnmf_apply.calc_coeff(d_Rrs['M'], pace_Rrs, 1./(pace_err)**2)

    # Find the closest solution in L23
    nspec = pace_Rrs.shape[0]
    all_idx = []
    for kk in range(nspec):
        L23_idx = np.argmin( (d_Rrs['coeff'][:,0] - nmf_coeff[0,kk])**2 + 
            (d_Rrs['coeff'][:,1] - nmf_coeff[1,kk])**2)
        # Debug?
        if debug:
            recon = np.dot(nmf_coeff[:,kk], d_Rrs['M'])
            fig = plt.figure(figsize=(5,5))
            ax = plt.gca()
            #
            ax.plot(pace_wave, pace_Rrs[kk,:], 'b-', label='Rebin')
            ax.plot(pace_wave, recon, 'g-', label='Recon')
            ax.plot(d_Rrs['wave'], d_Rrs['spec'][L23_idx], 'r-', label='L23')
            ax.legend()
            #
            plt.show()
            embed(header='103 of fit_pace')

        # Save
        all_idx.append(L23_idx)
    all_idx = np.array(all_idx)

    # Grab the parameters
    ab = np.atleast_2d(ab[L23_idx])
    Chl = np.atleast_1d(Chl[L23_idx])

    params = np.concatenate((ab, Chl[:,None]), axis=1)

    # Return
    return params



def fit(edict:dict, Nspec:int=None, abs_sig:float=None,
                      debug:bool=False, n_cores:int=1,
                      use_log_ab:bool=False,
                      use_NMF_pos:bool=False,
                      max_wv:float=None):
    """
    Fits the data with or without considering any errors.

    Args:
        edict (dict): A dictionary containing the necessary information for fitting.
        Nspec (int): The number of spectra to fit. Default is None = all
        abs_sig (float): The absolute value of the error to consider. Default is None.
            if None, use no error!
        debug (bool): Whether to run in debug mode. Default is False.
        n_cores (int): The number of CPU cores to use for parallel processing. Default is 1.
        max_wv (float): The maximum wavelength to consider. Default is None.
        use_log_ab (bool): Whether to use log(ab) in the priors. Default is False.
        use_NMF_pos (bool): Whether to use positive priors for NMF. Default is False.

    """
    # Priors
    priors = fits.set_priors(edict, use_log_ab=use_log_ab, use_NMF_pos=use_NMF_pos)
    
    # Load data
    pace_file, pace_wave, pace_Rrs, pace_err, emulator = load(edict)

    # Find the closest L23 solution
    params = find_closest(edict, pace_wave, pace_Rrs, pace_err)#, debug=True)
    
    # Output file
    outfile = os.path.basename(fitting_io.pace_chains_filename(
        pace_file, edict, abs_sig, priors=priors))

    # Scale Rrs
    pace_Rrs *= 1e4

    # Fit
    fits.fit(edict, params, pace_wave, priors, pace_Rrs, outfile, abs_sig,
                Nspec=Nspec, debug=debug, n_cores=n_cores, max_wv=max_wv)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Prep for IHOP
    if flg & (2**0):
        prep_one()

    # Fit one
    if flg & (2**1):
                # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (2,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 'PACE_TRUNC'
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, abs_sig=abs_sig, n_cores=n_cores, use_log_ab=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0


        # Prep
        #flg += 2 ** 0  # 1 

    else:
        flg = sys.argv[1]

    main(flg)