""" Emulator builds for Hydrolight and PCA """
import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.iops.decompose import load_loisel2023
from ihop.emulators import build

def emulate_l23(decomp:str, include_chl:bool=True, X:int=4, Y:int=0,
    hidden_list:list=[512, 512, 256], 
    nepochs:int=100, lr:float=1e-2, p_drop:float=0.):
    """
    Generate an emulator for a decomposition
    of the Loisel+23 dataset.

    Args:
        decomp (str): The decomposition type. pca, nmf
        include_chl (bool, optional): Flag indicating whether to include chlorophyll in the input data. Defaults to True.
        X (int, optional): X-coordinate of the dataset. Defaults to 4.
        Y (int, optional): Y-coordinate of the dataset. Defaults to 0.
        hidden_list (list, optional): List of hidden layer sizes for the dense neural network. Defaults to [512, 512, 256].
        nepochs (int, optional): Number of training epochs. Defaults to 100.
        lr (float, optional): Learning rate for the neural network. Defaults to 1e-2.
        p_drop (float, optional): Dropout probability for the neural network. Defaults to 0.
    """
    # Load data
    ab, Rs, _, _ = load_loisel2023(decomp)
    Ncomp = ab.shape[1]//2

    # Outfile
    root = f'dense_l23_{decomp}_X{X}Y{Y}_N{Ncomp:02d}'
    for item in hidden_list:
        root += f'_{item}'
    if include_chl:
        root += '_chl'


    if include_chl: 
        ds_l23 = loisel23.load_ds(X, Y)
        Chl = loisel23.calc_Chl(ds_l23)
        inputs = np.concatenate((ab, Chl.reshape(Chl.size,1)), axis=1)
    else:
        inputs = ab

    build.densenet(hidden_list, nepochs, inputs, Rs,
                   lr, dropout_on=False,
                   batchnorm=True, save=True, root=root)
    
def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # L23 + PCA
    if flg & (2**0):
        emulate_l23('pca', hidden_list=[512, 512, 512, 256],
            nepochs=25000)
        #  Ran on Nautilus Jupyter
        # epoch : 2500/2500, loss = 0.001642
        # epoch : 25000/25000, loss = 0.000570

    # L23 + NMF 
    if flg & (2**1):
        emulate_l23('nmf', hidden_list=[512, 512, 512, 256],
            nepochs=25000)
            #nepochs=100)
        #  Ran on Nautilus Jupyter
        # epoch : 25000/25000, loss = 0.000885


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- L23 + PCA
        #flg += 2 ** 1  # 2 -- L23 + NMF

        
    else:
        flg = sys.argv[1]

    main(flg)