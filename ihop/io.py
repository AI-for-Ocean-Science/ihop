""" Methods for IHOP I/O """

import os

import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.iops.io import load_loisel2023_decomp

from IPython import embed


def load_l23_full(decomps:tuple, Ncomp:tuple, X:int=4, Y:int=0):
    """
    Load L23 data and decomposition

    Args:
        decomps (tuple): Decomposition methods. pca, nmf
        Ncomp (tuple): The number of components (a,bb)
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
    ab, Rs, d_a, d_bb = load_loisel2023_decomp(decomps, Ncomp)
    # Chl
    ds_l23 = loisel23.load_ds(X, Y)
    Chl = loisel23.calc_Chl(ds_l23)

    # Return
    return ab, Chl, Rs, d_a, d_bb
