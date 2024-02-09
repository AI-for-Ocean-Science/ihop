""" Methods for IHOP I/O """

import os

import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.iops.decompose import load_loisel2023
from ihop.iops.decompose import reconstruct_nmf
from ihop.iops.decompose import reconstruct_pca
from ihop.emulators import io as ihop_io

from IPython import embed




def load_l23_decomposition(decomp:str, Ncomp:int, X:int=4, Y:int=0):
    """
    Load L23 data and decomposition

    Args:
        decomp (str): Decomposition method. pca, nmf
        Ncomp (int): The number of components.
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.

    Returns:
        tuple: Tuple containing ab, Chl, Rs, d_a, d_bb, and model.
            - ab (numpy.ndarray): Absorption coefficients.
            - Chl (numpy.ndarray): Chlorophyll concentration.
            - Rs (numpy.ndarray): Remote sensing reflectance.
            - d_a (dict): Hydrolight data for absorption coefficients.
            - d_bb (dict): Hydrolight data for backscattering coefficients.
    """
    print("Loading... ")
    ab, Rs, d_a, d_bb = load_loisel2023(decomp, Ncomp)
    # Chl
    ds_l23 = loisel23.load_ds(X, Y)
    Chl = loisel23.calc_Chl(ds_l23)

    # Return
    return ab, Chl, Rs, d_a, d_bb

# #############################################
def load_l23_chains(decomp:str, perc:int=None, X:int=4, Y:int=0):
    """
    Load data and chains for L23

    Args:
        decomp (str, optional): The type of IOP (Inherent Optical Property) model to use. Defaults to 'nmf'.
        X (int): simulation scenario   
        Y (int):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.
        perc (int, optional): The percentile to use for the MCMC chains. Defaults to 10.

    Returns:
        dict-like: The MCMC chains.
    """
    # Chains
    out_path = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'Fits', 'L23')
    if perc is None:
        raise IOError("Must specify percentile for now")
    else:
        chain_file = f'fit_L23_{decomp.upper()}_NN_Rs{perc:02d}.npz'

    # MCMC
    print(f"Loading MCMC: {chain_file}")
    d = np.load(os.path.join(out_path, chain_file))
    return d
