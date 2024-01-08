""" Preprocess data for training and testing. """

import numpy as np

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
