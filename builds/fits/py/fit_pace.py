""" Methods related to fitting PACE data"""
import os

import numpy as np
import xarray

from oceancolor.pace import io as pace_io

from ihop.pace import prep 
from ihop.emulators import io as emu_io

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
    return xds.wavelength.data, spec, spec_err


def load(edict:dict):
    """
    Load data and emulator from the given dictionary.

    Parameters:
        edict (dict): A dictionary containing the necessary information for loading data and emulator.

    Returns:
        tuple: A tuple containing the loaded data and emulator in the following order: 
            ab, Chl, Rs, emulator, d_a.
    """
    # Load data
    pace_wave, pace_spec, pace_err = load_one_example()

    # Load emulator
    emulator, e_file = emu_io.load_emulator_from_dict(
        edict, use_s3=True)

    # Return
    return pace_wave, pace_spec, pace_err, emulator


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Noiseless NMF
    if flg & (2**0):
        prep_one()

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