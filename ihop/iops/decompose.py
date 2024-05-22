""" Parameterizations of IOPs, i.e. decompositions """

import os
import numpy as np

from importlib import resources

from functools import partial

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from oceancolor.utils import pca

from cnmf.oceanography.iops import tara_matched_to_l23
from cnmf import nmf_imaging
from cnmf import io as cnmf_io

from ihop.iops import io as iops_io
from ihop.iops import hybrid

from IPython import embed

# Globals
pca_path = os.path.join(resources.files('ihop'),
                            'data', 'PCA')
nmf_path = os.path.join(resources.files('ihop'),
                            'data', 'NMF')


def generate_nmf(iop_data:np.ndarray, mask:np.ndarray, 
                 err:np.ndarray,
                 outfile:str,
                 Ncomp:int,
                 clobber:bool=False, 
                 nmf_path:str=nmf_path,
                 normalize:bool=True,
                 wave:np.ndarray=None,
                 Rs:np.ndarray=None,
                 seed:int=12345):
    """ Generate NMF model for input IOP 

    Args:
        iop_data (np.ndarray): IOP data (n_samples, n_features, 1)
        mask (np.ndarray): Mask (n_samples, n_features, 1)
        err (np.ndarray): Error (n_samples, n_features, 1)
        outroot (str): Output root.
        Ncomp (int): Number of PCA components. Defaults to 3.
        clobber (bool, optional): Clobber existing model? Defaults to False.
        nmf_path (str, optional): Path for output NMF files. Defaults to nmf_path.
        seed (int, optional): Random seed. Defaults to 12345.
        wave (np.ndarray, optional): Wavelengths. Defaults to None.
            Only for convenience in the saved file
        Rs (np.ndarray, optional): Rrs. Defaults to None.
            Only for convenience in the saved file
    """
    # Prep
    outfile = os.path.join(nmf_path, outfile)
    outroot = outfile.replace('.npz','')
    # Decompose
    comps = nmf_imaging.NMFcomponents(
        ref=iop_data, mask=mask, ref_err=err,
        n_components=Ncomp,
        path_save=outroot, oneByOne=True,
        normalize=normalize,
        seed=seed)
    # Gather up
    M = np.load(outroot+'_comp.npy').T
    coeff = np.load(outroot+'_coef.npy').T

    # Save
    cnmf_io.save_nmf(outfile, M, coeff, iop_data[...,0],
                     mask[...,0], err[...,0], wave, Rs)

def generate_pca(iop_data:np.ndarray,
                 outfile:str,
                 Ncomp:int,
                 norm:bool=False,
                 clobber:bool=False, 
                 extras:dict=None):
    """ Generate PCA model for input IOP 

    Args:
        iop_data (np.ndarray): IOP data (n_samples, n_features)
        outfile (str): 
        Ncomp (int): Number of PCA components. Defaults to 3.
        norm (bool, optional): Normalize the data? Defaults to False.
        clobber (bool, optional): Clobber existing model? Defaults to False.
        extras (dict, optional): Extra arrays to save. Defaults to None.
    """
    # Normalize?
    if norm:
        norm_vals = np.sum(iop_data, axis=1)
        iop_data = iop_data / np.outer(norm_vals, np.ones(iop_data.shape[1]))
        # Save to extras
        if extras is None:
            extras = dict()
        extras['norm_vals'] = norm_vals 
        #embed(header='97 of decompose')
    # Do it
    if not os.path.exists(outfile) or clobber:
        pca.fit_normal(iop_data, Ncomp, save_outputs=outfile,
                       extra_arrays=extras)
    else:
        print("File exists.  Use clobber=True to overwrite")

def generate_hybrid(iop_data:np.ndarray,
                 outfile:str, Ncomp:int, wave:np.ndarray,
                 clobber:bool=False, 
                 extras:dict=None):

    if not os.path.exists(outfile) or clobber:
        # Load the decomposition of aph
        aph_file = iops_io.loisel23_filename('nmf', 'aph', 2, 4, 0)
        d_aph = np.load(aph_file)

        if Ncomp != 4:
            raise ValueError("Ncomp must be 4 for hybrid decomposition")

        # Prep
        partial_func = partial(hybrid.a_func, W1=d_aph['M'][0], W2=d_aph['M'][1])

        # Do it
        params = []
        for idx in range(iop_data.shape[0]):
            ans, cov = curve_fit(partial_func, wave, 
                                iop_data[idx],
                        p0=[0.01, 0.01, 0.05, 0.05])
            # Save
            params.append(ans)
        params = np.array(params)
        # Prep
        outputs = dict(data=iop_data, 
                       coeff=params)
        if extras:
            outputs.update(extras)
        # Save
        np.savez(outfile, **outputs)
        print(f'Wrote: {outfile}')


def generate_int(iop_data:np.ndarray,
                 outfile:str, Ncomp:int, wave:np.ndarray,
                 clobber:bool=False, 
                 extras:dict=None):
    """ Generate Interpolation model for input IOP 

    Args:
        iop_data (np.ndarray): IOP data (n_samples, n_features)
        outfile (str): 
        Ncomp (int): Number of points to interpolate down to. Defaults to 3.
        wave (np.ndarray): Wavelengths of original
        clobber (bool, optional): Clobber existing model? Defaults to False.
        pca_path (str, optional): Path for output PCA files. Defaults to pca_path.
        extras (dict, optional): Extra arrays to save. Defaults to None.
    """
    # Do it
    if not os.path.exists(outfile) or clobber:
        new_wave = np.linspace(wave[0], wave[-1], Ncomp)
        new_spec = np.zeros((iop_data.shape[0], Ncomp))
#
        for ss in range(new_spec.shape[0]):
            f = interp1d(wave, iop_data[ss,:], kind='cubic')
            new_spec[ss,:] = f(new_wave)
        # Save
        outputs = dict(data=iop_data, new_wave=new_wave, 
                       new_spec=new_spec)
        if extras:
            outputs.update(extras)
        # Save
        np.savez(outfile, **outputs)
        print(f'Wrote: {outfile}')


def generate_l23_tara_pca(clobber:bool=False, return_N:int=None):
    """ Generate a PCA for L23 + Tara
        Restricted to 400-705nm

    Args:
        clobber (bool, optional): _description_. Defaults to False.
        return_N (int, optional): _description_. Defaults to None.

    Returns:
        None or tuple:  if return_N is not None, returns
            - **data** (*np.ndarray*) -- data array
            - **wave_grid** (*np.ndarray*) -- wavelength grid
            - **pca_fit** (*dict*) -- PCA fit
    """

    # Load up
    wave_grid, tara_a_water, l23_a = tara_matched_to_l23()

    # N components
    data = np.append(l23_a, tara_a_water, axis=0)
    for N in [3,5,20]:
        outfile = os.path.join(pca_path, f'pca_L23_X4Y0_Tara_a_N{N}.npz')
        if not os.path.exists(outfile) or clobber or ( (return_N is not None) 
                                                      and (return_N == N) ):
            print(f"Fit PCA with N={N}")
            pca_fit = pca.fit_normal(data, N, save_outputs=outfile,
                           extra_arrays={'wavelength':wave_grid})
            if return_N is not None and return_N == N:
                return data, wave_grid, pca_fit



def reconstruct_pca(Y:np.ndarray, pca_dict:dict, idx:int):
    """
    Reconstructs the original data point from the PCA-encoded representation.

    Args:
        Y (np.ndarray): The PCA-encoded representation of the data point.
        pca_dict (dict): A dictionary containing the PCA transformation parameters.
        idx (int): The index of the data point to reconstruct.

    Returns:
        tuple: A tuple containing the original data and its reconstructed version.
    """
    # Grab the original
    orig = pca_dict['data'][idx]

    # Reconstruct
    recon = np.dot(Y, pca_dict['M']) + pca_dict['mean']

    # Scale?
    if 'norm_vals' in pca_dict:
        orig *= pca_dict['norm_vals'][idx]
        recon *= pca_dict['norm_vals'][idx]

    return orig, recon

def reconstruct_nmf(Y:np.ndarray, nmf_dict:dict, idx:int):
    """
    Reconstructs the original data point from the PCA-encoded representation.

    Args:
        Y (np.ndarray): The NMF-encoded representation of the data point.
        pca_dict (dict): A dictionary containing the NMF transformation parameters.
        idx (int): The index of the data point to reconstruct.

    Returns:
        tuple: A tuple containing the original data and its reconstructed version.
    """
    # Grab the original
    orig = nmf_dict['spec'][idx]

    # Reconstruct
    recon = np.dot(Y, nmf_dict['M'])

    return orig, recon

def reconstruct_int(Y:np.ndarray, int_dict:dict, idx:int,
                    n_cores:int=10, single:bool=False):
    """
    Reconstructs the original data point from the INT-encoded representation.

    Args:
        Y (np.ndarray): The INT-encoded representation of the data point.
            nchains, nfeatures or nfeatures
        int_dict (dict): A dictionary containing the Int transformation parameters.
        idx (int): The index of the data point to reconstruct.

    Returns:
        tuple: A tuple containing the original data and its reconstructed version.
    """
    # Grab the original
    orig = int_dict['data'][idx]

    # Do it (slowly!)
    # TODO -- Parallelize this
    map_fn = partial(partial_int)

    if single:
        items = [(int_dict['new_wave'], Y, int_dict['wave'])]
    else:
        items = [(int_dict['new_wave'], Y[ichain,:], int_dict['wave']) 
             for ichain in range(Y.shape[0])]

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        recon = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    recon = np.array(recon)
    #for ichain in range(Y.shape[0]):
    #    f = interp1d(int_dict['new_wave'], Y[ichain,:], kind='cubic')
    #    recon[ichain,:] = f(int_dict['wave'])

    return orig, recon


def partial_int(items:list):
    # Unpack
    new_wave = items[0]
    new_spec = items[1]
    wave = items[2]

    # Do it (slowly!)
    f = interp1d(new_wave, new_spec, kind='cubic')
    return f(wave)
    