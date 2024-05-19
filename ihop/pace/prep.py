""" Methods to prep data for IHOP """

import numpy as np

from oceancolor.pace import io as pace_io
from oceancolor.utils import spectra

import xarray

from IPython import embed

def process_l2_for_l23(pfile:str, outfile:str, 
               minval:float=None, maxval:float=None):

    # Load L23
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(('nmf','nmf'), (2,2))
    L23_waves = np.array((d_a['wave']-2.5).tolist() + [d_a['wave'][-1] + 2.5])

    # Load
    xds, flags = pace_io.load_oci_l2(pfile)                    

    # Polish
    if minval is not None:
        xds['Rrs'] = xarray.where(xds['Rrs']>minval, xds['Rrs'], np.nan)
        xds['Rrs_unc'] = xarray.where(xds['Rrs']>minval, xds['Rrs_unc'], np.nan)
    if minval is not None:
        xds['Rrs'] = xarray.where(xds['Rrs']<maxval, xds['Rrs'], np.nan)
        xds['Rrs_unc'] = xarray.where(xds['Rrs']<maxval, xds['Rrs_unc'], np.nan)

    # Rebin
    nx, ny, nw = xds.Rrs.data.shape
    Rrs = np.resize(xds.Rrs.data, (nx*ny,nw))
    Rrs_u = np.resize(xds.Rrs_u.data, (nx*ny, nw))

    rebin_wave, rebin_values, rebin_err = spectra.rebin_to_grid(
        xds.wavelength.data, Rrs.T, Rrs.u.T, L23_waves)

    # 