""" Test module for emulators. """
import os

import numpy as np
import timeit
import torch

from ihop.emulators import io as ihop_io

from IPython import embed

model = None

def benchmark(in_model):
    global model
    model = in_model

    setup_code = '''
import torch
import numpy as np
global model
ninput = model.ninput
values = np.zeros(ninput)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''

    main_code = '''
model.prediction(values, device)
'''

    # Do it
    number = 10000
    tmp = timeit.timeit(stmt=main_code, setup=setup_code, 
        globals=globals(), number=number)
    print(f'Time per evaluation: {1000*tmp/number} ms')

if __name__ == '__main__':
    em_path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators')
    model_file = os.path.join(em_path, 'densenet_NMF3_L23', 
        'densenet_NMF_[512, 128, 128]_batchnorm_epochs_2500_p_0.05_lr_0.001.pth')
    print(f"Loading model: {model_file}")
    model = ihop_io.load_nn(model_file)
    model.ninput = 8
    benchmark(model)