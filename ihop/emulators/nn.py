""" Generate/Use NN to 'predict' Rs"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from ihop.iops.pca import load_loisel_2023_pca
from ihop.iops.pca import load_data
from ihop.iops.nmf import load_loisel_2023

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

def preprocess_data(data):
    """
    Preprocesses the input data by normalizing it.

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

def perform_training(model, dataset, ishape:int, tshape:int,
                     train_kwargs, lr, nepochs:int=100):
    """
    Perform training of a neural network model using the given dataset.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        dataset (torch.utils.data.Dataset): The dataset used for training.
        ishape (int): The input shape of the data.
        tshape (int): The target shape of the data.
        train_kwargs (dict): Additional keyword arguments for the data loader.
        lr (float): The learning rate for the optimizer.
        nepochs (int, optional): The number of training epochs. Defaults to 100.

    Returns:
        tuple: A tuple containing the final epoch number, the final loss value, and the optimizer.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset, **train_kwargs)

    for epoch in range(nepochs):
        loss = 0
        for batch_features, targets in train_loader:

            # load it to the active device
            batch_features = batch_features.view(-1, ishape).to(device)
            targets = targets.view(-1, tshape).to(device)

            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training loss
            train_loss = criterion(outputs, targets)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, nepochs, loss))

    # Return
    return epoch, loss, optimizer

def build_quick_nn_l23(nepochs:int,
                       root:str='model',
                       back_scatt:str='bb'):

    # ##############################
    # Quick NN on L23

    # Load up data
    ab, Rs, _, _ = load_loisel_2023_pca()
    

    target = Rs
    nparam = ab.shape[1]

    # Preprocess
    pre_ab, mean_ab, std_ab = preprocess_data(ab)
    pre_targ, mean_targ, std_targ = preprocess_data(target)

    # Dataset
    dataset = MyDataset(pre_ab, pre_targ)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nhidden1 = 128
    nhidden2 = 128
    # Instantiate
    model = SimpleNet(nparam, target.shape[1],
                      nhidden1, nhidden2,
                      (mean_ab, std_ab),
                      (mean_targ, std_targ)
                      ).to(device)
    nbatch = 64
    train_kwargs = {'batch_size': nbatch}

    lr = 1e-3
    epoch, loss, optimizer = perform_training(
        model, dataset, nparam, target.shape[1], train_kwargs, 
        lr, nepochs=nepochs)


    # Save
    PATH = f"{root}.pt"
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
    torch.save(model, f'{root}.pth')
    print(f"Wrote: {root}.pt, {root}.pth")

def build_densenet(hidden_list:list,
                   nepochs:int,
                   dataset:str,
                   lr:float,
                   dropout_on:bool,
                   p_dropout:float,
                   batchnorm:bool,
                   save:bool,
                   root:str='model'):
    """
    Builds a DenseNet model for training and saves the model if specified.

    Args:
        hidden_list (list): List of integers specifying the number of hidden units in each layer.
        nepochs (int): Number of training epochs.
        dataset (str): Name of the dataset to load.
        lr (float): Learning rate for the optimizer.
        dropout_on (bool): Flag indicating whether to use dropout layers.
        p_dropout (float): Dropout probability.
        batchnorm (bool): Flag indicating whether to use batch normalization layers.
        save (bool): Flag indicating whether to save the trained model.
        root (str, optional): Root name for the saved model files. Defaults to 'model'.

    Returns:
        float: The loss value after training.
    """
    ### The DenseNet with dropout and batchnorm layers.
    
    # Load up data
    if dataset == 'L23_PCA':
        print("Loading L23_PCA")
        ab, Rs, _, _ = load_loisel_2023_pca()
    elif dataset == 'L23_NMF':
        print("Loading L23_NMF")
        ab, Rs, _, _ = load_loisel_2023()
    else:
        print("Load Data from path of dataset")    
        ab, Rs = load_data(dataset)

    target = Rs
    nparam = ab.shape[1]

    # Preprocess
    pre_ab, mean_ab, std_ab = preprocess_data(ab)
    pre_targ, mean_targ, std_targ = preprocess_data(target)

    # Dataset
    dataset = MyDataset(pre_ab, pre_targ)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate
    model = DenseNet(nparam, target.shape[1],
                     hidden_list, dropout_on, p_dropout, batchnorm,
                     (mean_ab, std_ab), (mean_targ, std_targ)).to(device)
    nbatch = 64
    train_kwargs = {'batch_size': nbatch}

    epoch, loss, optimizer = perform_training(model, dataset, nparam,
        target.shape[1], train_kwargs, lr, nepochs=nepochs)

    # Save
    if save:
        PATH = f"{root}.pt"
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,}, PATH)
        torch.save(model, f'{root}.pth')
        print(f"Wrote: {root}.pt, {root}.pth")
    # Return
    return loss

if __name__ == '__main__':

    # Train
    build_quick_nn_l23(100, root='model_100')
    #build_quick_nn_l23(20000, root='model_20000')
    build_quick_nn_l23(100000, root='model_100000')
    #####################################################################
    hidden_list, epochs, lr, p_drop = [512, 512, 256], 2500, 1e-2, 0.0
    build_densenet(hidden_list, epochs, lr, True, p_drop, True, False)
    ### loss for above model is: 0.001996453170879529.
    #####################################################################
    
    # Test loading and prediction
    test = False
    if test:
        model = torch.load('model_100.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmp = model.prediction(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), device)
        embed(header='215 of nn')