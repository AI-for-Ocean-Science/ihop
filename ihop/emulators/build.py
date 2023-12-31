""" Routines to build emulators. """

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ihop.iops.pca import load_loisel_2023_pca
from ihop.iops.pca import load_data
from ihop.iops.nmf import load_loisel_2023
from ihop.emulators.nn import MyDataset, DenseNet, SimpleNet

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