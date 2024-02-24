""" I/O for IHOP Emulators """

import os
import warnings

import torch

from IPython import embed

def path_to_emulator(dataset:str):
    if os.getenv('OS_COLOR') is not None:
        path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators', dataset)
    else:
        warnings.warn("OS_COLOR not set. Using current directory.")
        path = './'
    return path

def load_emulator_from_dict(edict:dict):
    """
    Load an emulator from a dictionary.

    Args:
        edict (dict): A dictionary containing the emulator information.

    Returns:
        object: The loaded emulator.
    """
    # Load model
    path = path_to_emulator(edict['dataset'])
    if edict['dataset'] == 'L23':
        emulator_root = set_l23_emulator_root(edict)
    else:
        raise ValueError(f"Dataset {edict['dataset']} not supported.")
    emulator_file = os.path.join(path, emulator_root)+'.pth'

    # Load
    print(f'Loading: {emulator_file}')
    emulator = load_nn(emulator_file)

    # Return
    return emulator, emulator_file
        
def set_l23_emulator_root(edict:dict):
    # Dataset
    root = f'{edict["dataset"]}'

    # INPUTS
    # L23?
    if edict['dataset'] == 'L23':
        root += f'_X{edict["X"]}_Y{edict["Y"]}'
    # Decomp
    root += f'_{edict["decomp"]}'
    # Ncomp (tuple)
    root += f'_{edict["Ncomp"][0]}{edict["Ncomp"][1]}'
    # Chl?
    if edict['include_chl']:
        root += '_chl'
    # Outputs
    root += f'_{edict["outputs"]}'

    # Emulator
    root += f'_{edict["emulator"]}'
    # Hidden list?
    if edict['emulator'] == 'dense':
        for item in edict['hidden_list']:
            root += f'_{item}'
    # Return
    return root

def set_emulator_dict(dataset, decomp, Ncomp, outputs:str,
        emulator:str, hidden_list:list=None, 
        include_chl:bool=False, X:int=None, Y:int=None): 
    """
    Set the dictionary of emulator files for the Loisel+2023 emulator.

    Returns:
        dict: A dictionary containing the emulator files.
    """
    # Dict
    emulator_dict = {
        'dataset': dataset,
        'decomp': decomp,
        'Ncomp': Ncomp,
        'outputs': outputs,
        'emulator': emulator,
        'hidden_list': hidden_list,
        'include_chl': include_chl,
        'X': X,
        'Y': Y,
    }
    # Return
    return emulator_dict


def load_nn(model_file:str):
    """
    Load a neural network model from a file.

    Args:
        model_file (str): The path to the model file.
            Should end in .pth

    Returns:
        model: The loaded neural network model.
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        map_location=torch.device('cpu')
    else:
        map_location=None

    # Load
    model = torch.load(model_file, map_location=map_location)
    # Return
    return model


def save_nn(model, root:str, epoch:int, optimizer, losses:list, path:str=None):
    """
    Save the neural network model, optimizer state, and training information to 
    two files.
        .pt -- state
        .pth -- model

    Args:
        model (torch.nn.Module): The neural network model to be saved.
        root (str): The root name of the output file.
        epoch (int): The current epoch number.
        optimizer: The optimizer object used for training.
        losses (list): A list of training losses.
        path (str, optional): The directory path where the file will be saved. Defaults to None.

    Returns:
        tuple: A tuple containing the paths to the saved files.

    """
    # Outfile
    outfile = f"{root}.pt"
    if path is not None:
        # Make sure the folder exists
        if not os.path.exists(path):
            os.makedirs(path)
        # Update outfile
        outfile = os.path.join(path, outfile)
    # Save
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,}, 
               outfile)
    pth_outfile = outfile.replace('.pt', '.pth')
    torch.save(model, pth_outfile)
    print(f"Wrote")
    print(f"  {outfile}")
    print(f"  {pth_outfile}")

    # Return
    return outfile, pth_outfile