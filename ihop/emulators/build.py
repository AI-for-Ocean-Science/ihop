""" Routines to build emulators. """

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ihop.emulators.nn import MyDataset, DenseNet, SimpleNet
from ihop.emulators import preprocess
from ihop.emulators import io as emu_io

from IPython import embed

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

def densenet(hidden_list:list,
                   nepochs:int,
                   inputs:np.ndarray,
                   targets:np.ndarray,
                   lr:float,
                   dropout_on:bool=False,
                   p_dropout:float=0.,
                   batchnorm:bool=True,
                   save:bool=True,
                   root:str='model',
                   out_path:str=None):
    """
    Builds a DenseNet model for training and saves the model if specified.

    Args:
        hidden_list (list): List of integers specifying the number of hidden units in each layer.
        nepochs (int): Number of training epochs.
        inputs (np.ndarray): The input data. shape=(n_samples, n_features)
        targets (np.ndarray): The output data. shape=(n_samples, n_target_features)
        lr (float): Learning rate for the optimizer.
        dropout_on (bool): Flag indicating whether to use dropout layers.
        p_dropout (float): Dropout probability.
        batchnorm (bool): Flag indicating whether to use batch normalization layers.
        save (bool): Flag indicating whether to save the trained model.
        root (str, optional): Root name for the saved model files. Defaults to 'model'.
        out_path (str, optional): Path to the output directory. Defaults to None.

    Returns:
        float: The loss value after training.
    """
    nparam = inputs.shape[1]

    # Preprocess
    pre_inputs, mean_ab, std_ab = preprocess.normalize(inputs)
    pre_targets, mean_targ, std_targ = preprocess.normalize(targets)

    # Dataset
    dataset = MyDataset(pre_inputs, pre_targets)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate
    model = DenseNet(inputs.shape[1], pre_targets.shape[1],
                     hidden_list, dropout_on, p_dropout, batchnorm,
                     (mean_ab, std_ab), (mean_targ, std_targ)).to(device)
    nbatch = 64
    train_kwargs = {'batch_size': nbatch}

    epoch, losses, optimizer = perform_training(model, dataset, nparam,
        pre_targets.shape[1], train_kwargs, lr, nepochs=nepochs)

    # Save
    if save:
        pth_file, pt_file = emu_io.save_nn(
            model, root, epoch, optimizer, losses, path=out_path)
        
    # Return
    return losses


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
        tuple: A tuple containing the final epoch number, 
            a list of all the losses, and the optimizer.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset, **train_kwargs)

    losses = []
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

            # Convert to real space from normalized
            # new_output = outputs * model.Rs_parm[1] + model.Rs_parm[0]
            # new_targets = targets * model.Rs_parm[1] + model.Rs_parm[0]
          
            # compute training loss in real space
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
        losses.append(loss)

    # Return
    return epoch, losses, optimizer

