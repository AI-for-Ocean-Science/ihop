""" Inference I/O """
import os
import warnings

import numpy as np

from ihop.emulators import io as emu_io

def path_to_emulator(dataset:str):
    if os.getenv('OS_COLOR') is not None:
        path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 
                            'Fits', dataset)
    else:
        warnings.warn("OS_COLOR not set. Using current directory.")
        path = './'
    return path

def l23_chains_filename(edict:dict, error:int, test:bool=False,
                        out_path:str=None, priors:dict=None):
    """
    Generates the filename for the L23 chains file.

    Args:
        edict (dict): The input dictionary containing dataset information.
        error (int, str, None): The error value.
            If None, the file is assumed to be noiseless.
                And the error value is set to 1.
                And the filename is prefixed with 'N'.
            If PACE, we set the error value to 99.
            If PACE_CORR, we set the error value to 98.
        test (bool, optional): Flag indicating if it is a test file. Defaults to False.
        out_path (str, optional): The output path. Defaults to None.
        priors (dict, optional): The priors dictionary. Defaults to None.

    Returns:
        str: The generated filename for the L23 chains file.
    """
    if out_path is None:
        out_path = path_to_emulator(edict['dataset'])
    # Root
    root = emu_io.set_l23_emulator_root(edict)
    # Build it
    prefix = ''
    if error is None:
        # Noiseless!
        prefix = 'N'
        error = 1
    elif error == 'PACE':
        error = 99
    elif error == 'PACE_CORR':
        error = 98
    else: # abs_sig = float
        pass
    # 
    chain_file = f'fit{prefix}_Rs{int(error):02d}_{root}.npz'
    if priors is not None:
        if 'use_log_ab' in priors and priors['use_log_ab']:
            chain_file = f'{chain_file[:-4]}_logab.npz'
    if test:
        chain_file = 'test_'+chain_file
    # Return
    return os.path.join(out_path, chain_file)

# #############################################
def load_chains(chain_file:str):
    """
    Load data and chains for L23

    Args:
        edict (dict): A dictionary containing the emulator information.

    Returns:
        dict-like: The MCMC chains.
    """
    # MCMC
    print(f"Loading MCMC chains: {chain_file}")
    d = np.load(os.path.join(chain_file))
    return d