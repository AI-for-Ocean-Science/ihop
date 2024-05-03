""" I/O for IHOP Emulators """

import os
import warnings

import torch

from IPython import embed

def path_to_emulator(dataset:str, use_s3:bool=False):
    if use_s3:
        path = f's3://ihop/Emulators/{dataset}'
    elif os.getenv('OS_COLOR') is not None:
        path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators', dataset)
    else:
        warnings.warn("OS_COLOR not set. Using current directory.")
        path = './'
    return path

def load_emulator_from_dict(edict:dict, use_s3:bool=False): 
    """
    Load an emulator from a dictionary.

    Args:
        edict (dict): A dictionary containing the emulator information.
        use_s3 (bool, optional): Flag indicating whether to use S3. Defaults to False.

    Returns:
        object: The loaded emulator.
    """
    # Load model
    path = path_to_emulator(edict['dataset'], use_s3=use_s3)
    if edict['dataset'] == 'L23':
        emulator_root = set_l23_emulator_root(edict)
    else:
        raise ValueError(f"Dataset {edict['dataset']} not supported.")
    emulator_file = os.path.join(path, emulator_root)+'.pth'

    # Download from s3?
    if use_s3:
        from ulmo import io as ulmo_io
        local_file = os.path.basename(emulator_file)
        ulmo_io.download_file_from_s3(local_file, emulator_file)
        emulator_file = local_file

    # Load
    print(f'Loading: {emulator_file}')
    emulator = load_nn(emulator_file)

    # Return
    return emulator, emulator_file
        
def set_l23_emulator_root(edict: dict):
    """
    Constructs the root path for the L23 emulator based on the provided parameters.

    Parameters:
    - edict (dict): A dictionary containing the parameters for constructing the root path.

    Returns:
    - root (str): The constructed root path for the L23 emulator.
    """
    # Dataset
    root = f'{edict["dataset"]}'

    # INPUTS
    # L23?
    if edict['dataset'] == 'L23':
        root += f'_X{edict["X"]}_Y{edict["Y"]}'
    # Decomp
    root += f'_{edict["decomps"][0]}{edict["decomps"][1]}'
    # Ncomp (tuple)
    root += f'_{edict["Ncomps"][0]}{edict["Ncomps"][1]}'
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

def set_emulator_dict(dataset:str, decomps:tuple, 
                      Ncomps:tuple, outputs:str,
        emulator:str, hidden_list:list=None, 
        include_chl:bool=False, X:int=None, Y:int=None): 
    """
    Create a dictionary containing the parameters for an emulator.

    Args:
        dataset (str): The dataset used for training the emulator.
        decomp (tuple): The decomposition methods used for the dataset.
        Ncomp (tuple): The number of components used in the decomposition.
        outputs (str): The output variables predicted by the emulator.
        emulator (str): The type of emulator used.
        hidden_list (list, optional): List of hidden layers for the emulator. Defaults to None.
        include_chl (bool, optional): Whether to include chlorophyll as an input variable. Defaults to False.
        X (int, optional): The X coordinate of the emulator. Defaults to None.
        Y (int, optional): The Y coordinate of the emulator. Defaults to None.

    Returns:
        dict: A dictionary containing the emulator parameters.
    """
    # Dict
    emulator_dict = {
        'dataset': dataset,
        'decomps': decomps,
        'Ncomps': Ncomps,
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