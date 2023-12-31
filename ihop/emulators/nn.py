""" Generate/Use NN to 'predict' Rs"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


from IPython import embed

# Erdong's Notebook
#   https://github.com/AI-for-Ocean-Science/ulmo/blob/F_S/ulmo/fs_reg_dense/fs_dense_train.ipynb

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

class SimpleNet(nn.Module):
    def __init__(self, ninput:int, noutput:int,
                 nhidden1:int, nhidden2:int,
                 ab_norm:tuple, 
                 Rs_parm:tuple):
        super(SimpleNet, self).__init__()
        # Save
        self.ninput = ninput
        self.noutput = noutput
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2

        self.ab_norm = ab_norm
        self.Rs_parm = Rs_parm

        # Architecture
        self.fc1 = nn.Linear(self.ninput, nhidden1)
        self.fc2 = nn.Linear(nhidden1, noutput)
        self.fc2b = nn.Linear(nhidden1, nhidden2)
        self.fc3 = nn.Linear(nhidden2, noutput)

        # Normalization terms
        

    def forward(self, x):
        '''
        # 1 hidden layers -- 0.23 at 1000
        x = self.fc1(x)
        x = F.relu(x)  # 0.23 at 1000
        #x = F.leaky_relu(x)  # 0.23 at 1000
        #x = F.sigmoid(x)  # Much worse
        x = self.fc2(x)
        '''

        # 2 hidden layers -- 9,17 at 1000
        x = self.fc1(x)
        x = F.relu(x)  
        x = self.fc2b(x)
        x = F.relu(x)  
        x = self.fc3(x)
        #x = F.relu(x)
        #x = self.fc3(x)

        return x

    def prediction(self, sample, device):
        # Normalize the inputs
        norm_sample = (sample - self.ab_norm[0])/self.ab_norm[1]
        tensor = torch.Tensor(norm_sample)

        self.eval()
        with torch.no_grad():
            batch_features = tensor.view(-1, 6).to(device)
            outputs = self(batch_features)

        outputs.cpu()
        pred = outputs * self.Rs_parm[1] + self.Rs_parm[0]

        # Convert to numpy
        pred = pred.numpy()

        return pred.flatten()

class DenseNet(nn.Module):
    """
    A class representing a DenseNet neural network model.

    Args:
        d_input (int): The number of input features.
        d_output (int): The number of output features.
        hidden_list (list): A list of integers representing the number of hidden units in each layer.
        drop_on (bool): Whether to apply dropout regularization.
        p_drop (float): The probability of dropping a neuron during dropout.
        batchnorm (bool): Whether to apply batch normalization.
        ab_norm (tuple): A tuple containing the mean and standard deviation used for input normalization.
        Rs_parm (tuple): A tuple containing the scaling parameters used for output denormalization.

    Attributes:
        ab_norm (tuple): A tuple containing the mean and standard deviation used for input normalization.
        Rs_parm (tuple): A tuple containing the scaling parameters used for output denormalization.
        ninput (int): The number of input features.
        noutput (int): The number of output features.
        num_layers (int): The total number of layers in the network.
        block_layers (nn.ModuleList): A list of layer blocks in the network.

    Methods:
        layer_block: Creates a layer block with the specified configuration.
        forward: Performs forward propagation through the network.
        prediction: Generates predictions for a given input sample.

    """

    def __init__(self, d_input:int, d_output:int, 
                 hidden_list:list, drop_on:bool, 
                 p_drop:float, batchnorm:bool, 
                 ab_norm:tuple, Rs_parm:tuple):
        # Init
        super(DenseNet, self).__init__()

        # Save
        self.ab_norm = ab_norm
        self.Rs_parm = Rs_parm
        self.ninput = d_input
        self.noutput = d_output
        self.num_layers = len(hidden_list) + 1

        # Build the layers
        block_layers = []
        d_in = d_input
        d_out = hidden_list[0]
        block_layers.append(self.layer_block(d_in, d_out, drop_on, p_drop, batchnorm))
        for i in range(self.num_layers-2):
            d_in = hidden_list[i]
            d_out = hidden_list[i+1]
            block_layers.append(self.layer_block(d_in, d_out, drop_on, p_drop, batchnorm))
        d_in = hidden_list[-1]
        d_out = d_output
        head_layer = nn.Sequential(
            nn.Linear(in_features=d_in, out_features=d_out)
        )
        block_layers.append(head_layer)
        self.block_layers = nn.ModuleList(block_layers)
        
    def layer_block(self, d_in, d_out, drop_on, p_drop, batchnorm):
        """
        Creates a layer block with the specified configuration.

        Args:
            d_in (int): The number of input features for the layer block.
            d_out (int): The number of output features for the layer block.
            drop_on (bool): Whether to apply dropout regularization.
            p_drop (float): The probability of dropping a neuron during dropout.
            batchnorm (bool): Whether to apply batch normalization.

        Returns:
            nn.Sequential: The layer block.

        """
        if drop_on and batchnorm:
            block_i = nn.Sequential(
                nn.Linear(in_features=d_in, out_features=d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(),
                nn.Dropout(p_drop)
            )
        elif drop_on:
            block_i = nn.Sequential(
                nn.Linear(in_features=d_in, out_features=d_out),
                nn.ReLU(),
                nn.Dropout(p_drop)
            )
        else:
            block_i = nn.Sequential(
                nn.Linear(in_features=d_in, out_features=d_out),
                nn.ReLU()
            )
        return block_i

    def forward(self, x):
        """
        Performs forward propagation through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        for i in range(self.num_layers):
            layer_i = self.block_layers[i]
            x = layer_i(x)
        return x

    def prediction(self, sample, device):
        """
        Generates predictions for a given input sample.

        Args:
            sample (numpy.ndarray): The input sample.
            device: The device to perform the prediction on.

        Returns:
            numpy.ndarray: The predicted output.

        """
        # Normalize the inputs
        norm_sample = (sample - self.ab_norm[0])/self.ab_norm[1]
        nparam = norm_sample.size
        tensor = torch.Tensor(norm_sample)

        self.eval()
        with torch.no_grad():
            batch_features = tensor.view(-1, nparam).to(device)
            outputs = self(batch_features)

        outputs.cpu()
        pred = outputs * self.Rs_parm[1] + self.Rs_parm[0]

        # Convert to numpy
        pred = pred.numpy()

        return pred.flatten()