""" Basic I/O for IHOP """

import torch


def load_nn(model_file:str):

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        map_location=torch.device('cpu')
    else:
        map_location=None

    # Load
    model = torch.load(model_file, map_location=map_location)

    return model