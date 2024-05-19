""" Methods related to fitting PACE data"""

from ihop import io as ihop_io

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
    ab, Chl, Rs, d_a, d_bb = ihop_io.load_l23_full(
        edict['decomps'], edict['Ncomps'])
    # Load emulator
    emulator, e_file = emu_io.load_emulator_from_dict(
        edict, use_s3=True)

    # Return
    return ab, Chl, Rs, emulator, d_a