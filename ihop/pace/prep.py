""" Methods to prep data for IHOP """
import os

import numpy as np

import xarray

from oceancolor.pace import io as pace_io
from oceancolor.utils import spectra

from ihop import io as ihop_io

from IPython import embed

def process_l2_for_l23(pfile:str, outfile:str, 
               minval:float=None, maxval:float=None):
            
    # Check for file
    if os.path.isfile(outfile):
        print(f'File exists: {outfile}')
        print('Remove it if you want to write a new one')
        return
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
    Rrs_unc = np.resize(xds.Rrs_unc.data, (nx*ny, nw))

    print(f"Rebinning to {L23_waves.size} wavelengths")
    rebin_wave, rebin_values, rebin_err = spectra.rebin_to_grid(
        xds.wavelength.data, Rrs.T, Rrs_unc.T, L23_waves)

    # Back to 3D
    Rrs = np.reshape(rebin_values, (nx,ny,rebin_wave.size))
    Rrs_unc = np.reshape(rebin_err, (nx,ny,rebin_wave.size))

    # Construct the nc file

    rrs_xds = xarray.Dataset(
        {'Rrs':(('x', 'y', 'wl'),Rrs.astype(np.float32))},
               coords = {'latitude': (('x', 'y'), xds.latitude.data),
                                  'longitude': (('x', 'y'), xds.longitude.data),
                                  'wavelength' : ('wl', rebin_wave)},
               attrs={'variable':'Remote sensing reflectance'})
    rrsu_xds = xarray.Dataset(
        {'Rrs_unc':(('x', 'y', 'wl'),Rrs_unc.astype(np.float32))},
               coords = {'latitude': (('x', 'y'), xds.latitude.data),
                                  'longitude': (('x', 'y'), xds.longitude.data),
                                  'wavelength' : ('wl', rebin_wave)},
               attrs={'variable':'Remote sensing reflectance error'})

    new_xds = xarray.merge([rrs_xds, rrsu_xds])

    # Write
    new_xds.to_netcdf(outfile)
    print(f'Wrote: {outfile}')