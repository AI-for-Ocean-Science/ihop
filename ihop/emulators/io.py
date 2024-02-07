""" Basic I/O for IHOP """

import os

import torch


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
    print(f"Wrote: {outfile}.pt, {outfile}.pth")