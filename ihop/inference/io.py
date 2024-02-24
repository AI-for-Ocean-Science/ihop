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

def l23_chains_filename(edict:dict, perc:int, test:bool=False):
    # perc (int, optional): The percentile to use for the MCMC chains. Defaults to 10.
    out_path = path_to_emulator(edict['dataset'])
    # Root
    root = emu_io.set_l23_emulator_root(edict)
    # Build it
    chain_file = f'fit_Rs{int(perc):02d}_{root}.npz'
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