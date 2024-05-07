""" Preprocess data for training and testing. """

import numpy as np

from IPython import embed

def normalize(data):
    """
    Preprocesses the input data by normalizing it.

    Both demeand and set to unit variance.

    Args:
        data (numpy.ndarray): The input data to be preprocessed.

    Returns:
        tuple: A tuple containing the preprocessed data, mean, and standard deviation.

    """
    # Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean)/std

    return data.astype(np.float32), mean, std

def linear(data, max_scale:float):
    """
    Preprocesses the input data by balancing the targets
        with a linear relation.  If positive, increase 
        with array size, otherwise decrease

    Args:
        data (numpy.ndarray): The input data to be preprocessed.
        max_scale (float): The maximum scaling factor to use.

    Returns:
        tuple: A tuple containing the preprocessed data
            and the scalings used for balancing.

    """
    offset = 0.
    scale = np.linspace(1., np.abs(max_scale), data.shape[1])
    if max_scale < 0:
        scale = 1./scale
    # Preproc
    data = (data - offset)/scale
    # Return
    return data.astype(np.float32), offset, scale