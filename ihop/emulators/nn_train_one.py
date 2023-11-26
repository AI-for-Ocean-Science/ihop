""" Generate/Use NN to 'predict' Rs"""


#from io_test import *
from IPython import embed
from ihop.emulators.nn import build_densenet
from ihop.emulators.nn_arch_search import archt_search


if __name__ == "__main__":
    epochs = 25000
    lr_list = [1e-2]#, 1e-3]
    hidden_lists = [
        [512, 512, 512, 256],
    ]
    p_drop_list = [0.0]#, 0.05]
    print("search starts.")
    #opt_result = archt_search(hidden_lists, lr_list, p_drop_list, epochs, 'L23_PCA')
    opt_result = archt_search(hidden_lists, lr_list, p_drop_list, epochs, 'L23_NMF')
    print("search ends.")
    print("opt loss is: ", opt_result[-1])
    print("opt hyperparas are: ", opt_result[:-1])
    #############################################
    ### best results provided by above models:
    ### epochs: 2500
    ### opt_loss: 0.001484
    ### model: (True,  [512, 512, 512, 256], 0.01, 0.0)
    #############################################
    
