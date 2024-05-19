""" Methods related to fitting PACE data"""
import os

import numpy as np
import xarray

from oceancolor.pace import io as pace_io

from ihop import io as ihop_io

def load_one_example(lon:float=-75.5, # W
                    lat:float=34.5): # N
    pfile = os.path.join(os.getenv('OS_COLOR'), 'data', 'PACE', 'early',
                     'PACE_OCI.20240413T175656.L2.OC_AOP.V1_0_0.NRT.nc')
    xds, flags = pace_io.load_oci_l2(pfile)                    

    # Polish
    xds['Rrs'] = xarray.where(xds['Rrs']>-10, xds['Rrs'], np.nan)
    xds['Rrs'] = xarray.where(xds['Rrs']<100, xds['Rrs'], np.nan)
    # Error too
    xds['Rrs_unc'] = xarray.where(xds['Rrs']>-10, xds['Rrs_unc'], np.nan)
    xds['Rrs_unc'] = xarray.where(xds['Rrs']<100, xds['Rrs_unc'], np.nan)

    #
    idx = np.argmin( (xds.longitude.data-lon)**2 + (xds.latitude.data-lat)**2)
    x,y = np.unravel_index(idx, xds.longitude.shape)

    spec = xds.Rrs.data[x,y,:]
    #
    return xds.wavelength.data, spec


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
    pace_wave, pace_spec = load_one_example()

    # Load emulator
    emulator, e_file = emu_io.load_emulator_from_dict(
        edict, use_s3=True)

    # Return
    return ab, Chl, Rs, emulator, d_a