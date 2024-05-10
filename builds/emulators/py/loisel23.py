""" Emulator builds for Hydrolight and PCA """
import os
import numpy as np

from oceancolor.hydrolight import loisel23

from ihop.iops import io as iops_io
from ihop.emulators import build
from ihop.emulators import io as emu_io

from ulmo import io as ulmo_io

from IPython import embed


def emulate_l23(decomps:tuple, Ncomps:tuple, include_chl:bool=True, 
                X:int=4, Y:int=0, hidden_list:list=[512, 512, 256], 
                real_loss:bool=False, preproc_Rs:str=None,
                nepochs:int=100, lr:float=1e-2, 
                push_to_s3:bool=False):
    """
    Generate an emulator for a decomposition
    of the Loisel+23 dataset.

    Args:
        decomps (tuple): The decomposition type. pca, nmf
        Ncomps (tuple): The number of components. (a,bb)
        include_chl (bool, optional): Flag indicating whether to include chlorophyll in the input data. Defaults to True.
        X (int, optional): X-coordinate of the dataset. Defaults to 4.
        Y (int, optional): Y-coordinate of the dataset. Defaults to 0.
        hidden_list (list, optional): List of hidden layer sizes for the dense neural network. Defaults to [512, 512, 256].
        nepochs (int, optional): Number of training epochs. Defaults to 100.
        lr (float, optional): Learning rate for the neural network. Defaults to 1e-2.
        push_to_s3 (bool, optional): Flag indicating whether to push the model to S3. Defaults to False.
        skip_Chl (bool, optional): Skip the chlorophyll input. Defaults to False.
        preproc_Rs (str, optional): Preprocessing method for Rrs. Defaults to None.
            norm -- Normalize Rrs
            lin## -- Linear scaling for Rrs
    """
    dataset = 'L23'
    # Load data
    ab, Rs, _, _ = iops_io.load_loisel2023_decomp(
        decomps, Ncomps, X=X, Y=Y)

    # Emulator dict
    edict = emu_io.set_emulator_dict(
        dataset, decomps, Ncomps, 'Rrs',
        'dense', hidden_list=hidden_list, 
        include_chl=include_chl, X=X, Y=Y,
        preproc_Rs=preproc_Rs)

    # Outfile (local)
    root = emu_io.set_l23_emulator_root(edict)
    path = emu_io.path_to_emulator(dataset)
    root = os.path.join(path, root)
    print(f"Building emulator: {root}")

    if include_chl: 
        ds_l23 = loisel23.load_ds(X, Y)
        Chl = loisel23.calc_Chl(ds_l23)
        inputs = np.concatenate((ab, Chl.reshape(Chl.size,1)), axis=1)
    else:
        inputs = ab

    build.densenet(hidden_list, nepochs, inputs, Rs,
                   lr, dropout_on=False,
                   preproc_targets=preproc_Rs,
                   batchnorm=True, save=True, root=root,
                   out_path=path, real_loss=real_loss)

    # Push to S3?
    if push_to_s3:
        s3_path = f's3://ihop/Emulators/L23'
        for ext in ['pt', 'pth']:
            # s3 Filename
            basef = os.path.basename(root)
            s3_file = os.path.join(s3_path, f'{basef}.{ext}')
            # Do it
            ulmo_io.upload_file_to_s3(f'{root}.{ext}', s3_file)
    

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # L23 + PCA
    if flg & (2**0):
        emulate_l23('pca', 3, hidden_list=[512, 512, 512, 256],
            nepochs=25000)
        #  Ran on Nautilus Jupyter
        # epoch : 2500/2500, loss = 0.001642
        # epoch : 25000/25000, loss = 0.000570

    # L23 + NMF, m=3
    if flg & (2**1):
        emulate_l23('nmf', 3, hidden_list=[512, 512, 512, 256],
            nepochs=25000, #norm_Rs=False,
            push_to_s3=True)

    # flg=4;  L23 + NMF, m=4
    if flg & (2**2):
        # Now without water
        emulate_l23('nmf', 4, hidden_list=[512, 512, 512, 256],
            nepochs=25000, #norm_Rs=False,
            push_to_s3=True)

    # flg=4;  L23 + NMF, m=4,3
    if flg & (2**3):
        emulate_l23('nmf', (4,3), hidden_list=[512, 512, 512, 256],
            nepochs=25000, #norm_Rs=False,
            push_to_s3=True)

    # L23 + PCA, m=4,2
    if flg & (2**4):
        emulate_l23(('pca','pca'), (4,2), 
                    hidden_list=[512, 512, 512, 256], 
                    nepochs=25000, #norm_Rs=False, 
                    push_to_s3=True)

    # L23 + Int,NMF, m=40,2; Chl
    if flg & (2**5):
        emulate_l23(('int','nmf'), (40,2), 
                    hidden_list=[512, 512, 512, 256], 
                    nepochs=25000, #norm_Rs=False, 
                    push_to_s3=True)

    # L23 + PCA, balance Rs 
    if flg & (2**6): # 64
        emulate_l23(('pca','pca'), (4,2), 
                    hidden_list=[512, 512, 512, 256], 
                    nepochs=25000, preproc_Rs='lin-5', 
                    push_to_s3=True)

    # flg=4;  L23 + NMF, m=3,2
    if flg & (2**7):
        emulate_l23(('nmf', 'nmf'), (3,2), 
                    hidden_list=[512, 512, 512, 256],
            nepochs=25000, 
            push_to_s3=True)

    # flg=4;  L23 + NMF, m=4,2, no Chl
    if flg & (2**8):
        emulate_l23(('nmf', 'nmf'), (4,2), 
                    hidden_list=[512, 512, 512, 256],
            nepochs=25000, include_chl=False,
            push_to_s3=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- L23 + PCA
        #flg += 2 ** 1  # 2 -- L23 + NMF
        #flg += 2 ** 2  # 4 -- L23 + NMF 4
        #flg += 2 ** 3  # 8 -- L23 + NMF 4,3

        #flg += 2 ** 4  # 16 -- L23 + PCA 4,2 + norm_Rs=False
        #flg += 2 ** 5  # 32 -- L23 + INT,NMF 40,2 + norm_Rs=False

        #flg += 2 ** 6  # 64 -- L23 + PCA 4,2; linear scale Rs (-5)
        
        #flg += 2 ** 7  # 128 -- L23 + NMF 3,2 + norm_Rs=False
        #flg += 2 ** 8  # 256 -- L23 + NMF 4,2 + norm_Rs=False, no Chl

        
    else:
        flg = sys.argv[1]

    main(flg)