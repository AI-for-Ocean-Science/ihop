""" Emulator builds for Hydrolight and PCA """
import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.iops.pca import load_loisel_2023_pca
from ihop.emulators import build

def l23_pca(include_chl:bool=True, X:int=4, Y:int=0,
    hidden_list:list=[512, 512, 256], 
    nepochs:int=100, lr:float=1e-2, p_drop:float=0.):

    #include_chl (bool, optional): Flag indicating whether to include chlorophyll in the input data. Defaults to True.
    root = f'dense_l23_pca_X{X}Y{Y}'
    for item in hidden_list:
        root += f'_{item}'
    if include_chl:
        root += '_chl'

    ab, Rs, _, _ = load_loisel_2023_pca()

    # Chl
    if include_chl: 
        # Load
        ds_l23 = loisel23.load_ds(X, Y)
        # Chl
        Chl = loisel23.calc_Chl(ds_l23)
        # Concatenate
        inputs = np.concatenate((ab, Chl.reshape(Chl.size,1)), axis=1)
    else:
        inputs = ab



    # Build
    build.densenet(hidden_list, nepochs, inputs, Rs,
                   lr, dropout_on=False,
                   batchnorm=True, save=True, root=root)
    
if __name__ == '__main__':

    # L23
    l23_pca(hidden_list=[512, 512, 512, 256],
            nepochs=2500)
