""" Basic I/O for IHOP """

import os
import warnings

import torch

def path_to_emulator(dataset:str):
    if os.getenv('OS_COLOR') is not None:
        path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators', dataset)
    else:
        warnings.warn("OS_COLOR not set. Using current directory.")
        path = './'
    return path
        
def set_l23_emulator_root(edict:dict):
    # Dataset
    root = f'{edict["dataset"]}'

    # INPUTS
    # L23?
    if edict['dataset'] == 'L23':
        root += f'_X{edict["X"]}_Y{edict["Y"]}'
    # Decomp
    root += f'_{edict["decomp"]}'
    # Ncomp
    root += f'_{edict["Ncomp"]}'
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
        outfile = os.path.join(path, outfile)
    # Save
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses,}, 
               outfile)
    pth_outfile = outfile.replace('.pt', '.pth')
    torch.save(model, pth_outfile)
    print(f"Wrote: {outfile}, {pth_outfile}")

    # Return
    return outfile, pth_outfile