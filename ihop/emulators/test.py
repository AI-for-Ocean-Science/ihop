""" Test module for emulators. """
import os
import glob
from importlib import resources

import numpy as np
import timeit
import torch

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from oceancolor.utils import plotting 

from ihop.emulators import io as ihop_io
from ihop.iops.nmf import load_loisel_2023
from ihop.iops.pca import load_loisel_2023_pca

from IPython import embed

model = None

def benchmark(in_model, number:int=50000):
    global model
    model = in_model

    setup_code = '''
import torch
import numpy as np
global model
try:
    ninput = model.ninput
except:
    ninput = 8
values = np.zeros(ninput)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''

    main_code = '''
model.prediction(values, device)
'''

    # Do it
    tot_time = timeit.timeit(stmt=main_code, setup=setup_code, 
        globals=globals(), number=number)
    time_per_eval = tot_time/number
    print(f'Time per evaluation: {1000*time_per_eval} ms')

    return time_per_eval

def calc_stats(model_file:str, iop_type:str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load
    model = ihop_io.load_nn(model_file)
    if iop_type == 'pca':
        ab, Rs, d_a, d_bb = load_loisel_2023_pca()
    elif iop_type == 'nmf':
        ab, Rs, d_a, d_bb = load_loisel_2023()

    # Loop through em
    devs = []
    for idx in range(ab.shape[0]):
        pred_Rs = model.prediction(ab[idx], device)
        dev = pred_Rs - Rs[idx] 
        #
        devs.append(dev)
    # array
    devs = np.array(devs)

    # Stats
    mean_Rs = np.mean(Rs, axis=0)
    rmse_wave = np.std(devs, axis=0)
    avg_rmse = np.std(devs, axis=1)
    per_rmse = rmse_wave/mean_Rs*100

    max_dev_per = np.max(np.abs(devs/Rs), axis=0)
    bias = np.mean(devs, axis=0)
    per_bias = bias/mean_Rs*100

    # Return
    stat_dict = {}
    stat_dict['mean_Rs'] = mean_Rs
    stat_dict['rmse_wave'] = rmse_wave
    stat_dict['avg_rmse'] = avg_rmse
    stat_dict['per_rmse'] = per_rmse
    stat_dict['max_dev_per'] = max_dev_per
    stat_dict['bias'] = bias
    stat_dict['per_bias'] = per_bias
    return stat_dict

def nn_emulator_plot(model_file:str, iop_type:str,
                     clobber:bool=False):

    out_path = os.path.join(
        resources.files('ihop'), 'emulators', 'QA')
    out_root = os.path.basename(model_file).split('.pt')[0]
    out_file = os.path.join(out_path, f'{out_root}.png')

    if os.path.exists(out_file) and not clobber:
        print(f"Skipping: File exists {out_file}")
        return

    # Calculations
    model = ihop_io.load_nn(model_file)
    time_per = benchmark(model)
    stat_dict = calc_stats(model_file, iop_type)

    if iop_type == 'pca':
        ab, Rs, d_a, d_bb = load_loisel_2023_pca()
    elif iop_type == 'nmf':
        ab, Rs, d_a, d_bb = load_loisel_2023()

    # ########################
    # Plots
    fig = plt.figure(figsize=(11,7))
    gs = gridspec.GridSpec(2,2)

    # ########################
    # Data
    ax0 = plt.subplot(gs[0])
    ax0.text(0.95, 0.9, out_root, transform=ax0.transAxes,
              fontsize=9, ha='right', color='k')
    ax0.text(0.95, 0.8, f'Time per eval: {time_per*1000:0.2f} ms', transform=ax0.transAxes,
              fontsize=14, ha='right', color='k')
    ax0.text(0.95, 0.7, f'Avg RMSE: {np.mean(stat_dict["avg_rmse"]):0.5f}', 
             transform=ax0.transAxes, fontsize=14, ha='right', color='k')
    ax0.set_axis_off()

    # ########################
    # RMSE
    ax1 = plt.subplot(gs[1])

    ax1.plot(d_a['wave'], stat_dict['per_rmse'])
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('RMSE (%)')
    ax1.set_ylim(0, 30)

    ax1b = ax1.twinx()
    oclr = 'g'
    ax1b.plot(d_a['wave'], stat_dict['rmse_wave'], color=oclr)
    ax1b.set_ylabel('RMSE', color=oclr)

    # ########################
    # Bias
    ax2 = plt.subplot(gs[2])

    ax2.plot(d_a['wave'], stat_dict['bias'], color='k')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Bias (%)')

    # ########################
    # Max dev
    ax3 = plt.subplot(gs[3])

    ax3.plot(d_a['wave'], stat_dict['max_dev_per'], color='r')
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Max Dev (%)')

    # Finish
    for ax in [ax1, ax1b, ax2, ax3]:
        plotting.set_fontsize(ax, 15)
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(out_file, dpi=300)
    print(f"Saved: {out_file}")

if __name__ == '__main__':

    '''
    em_path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators')
    model_file = os.path.join(em_path, 'densenet_NMF3_L23', 
        'densenet_NMF_[512, 128, 128]_batchnorm_epochs_2500_p_0.05_lr_0.001.pth')
    print(f"Loading model: {model_file}")
    model = ihop_io.load_nn(model_file)
    model.ninput = 8
    benchmark(model)
    '''

    # Test em
    em_path = os.path.join(os.getenv('OS_COLOR'), 'IHOP', 'Emulators',
                           'DenseNet_NM4')
    model_files = glob.glob(os.path.join(em_path, '*.pth'))

    for model_file in model_files:
        nn_emulator_plot(model_file, 'nmf')
        