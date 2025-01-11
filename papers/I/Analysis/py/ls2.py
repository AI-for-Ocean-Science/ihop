""" Methods related to LS2 """

import numpy as np

from oceancolor.ls2 import kd_nn
from oceancolor.ls2 import ls2_main
from oceancolor.ls2.io import load_LUT
from oceancolor.water import absorption
from oceancolor.water import scattering
from oceancolor.hydrolight import loisel23

def calc_ls2(l23_idx:int, sza:float=0.):

    # Load
    ds = loisel23.load_ds(4,0)
    wave = ds.Lambda.data
    Rs = ds.Rrs.data

    # Grab the MODIS wavelengths
    modis_wvs = [443, 488, 531, 547, 667] # [nm].
    i_modis = []
    for modis_wv in modis_wvs:
        idx = np.argmin(np.abs(wave-modis_wv))
        i_modis.append(idx)

    # #######################
    # Kd
    Rrs_MODIS = Rs[l23_idx, i_modis] 

    Kds = []
    for iwave in wave:
        Kd = kd_nn.Kd_NN_MODIS(Rrs_MODIS, sza, iwave)
        Kds.append(Kd[0][0])
    # array me
    Kds = np.array(Kds)

    # LUT
    LS2_LUT = load_LUT()

    # Water
    a_w = absorption.a_water(wave, data='IOCCG')
    _, _, b_w = scattering.betasw_ZHH2009(wave, 20., [0], 35.)

    # b_p
    b_p = ds.bnw[l23_idx, :].data


    # Run it!
    #
    all_a = []
    all_bb = []
    all_bbp = []
    all_anw = []
    waves = []
    for jj, iwave in enumerate(wave):
        items = ls2_main.LS2_main(
            sza, iwave, Rs[l23_idx, jj], 
            Kds[jj], a_w[jj], b_w[jj], b_p[jj], LS2_LUT, False)
        if items is None:
            continue
        a, anw, bb, bbp, kappa = items
        # Save
        waves.append(iwave)
        all_a.append(a)
        all_anw.append(anw)
        all_bb.append(bb)
        all_bbp.append(bbp)

    # Return
    return waves, all_a, all_anw, all_bb, all_bbp

# Test
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    in_idx:int=2949 # Max

    ds = loisel23.load_ds(4,0)
    wave = ds.Lambda.data
    true_a_nw = ds.anw[in_idx, :].data

    # Run
    waves, all_a, all_anw, all_bb, all_bbp = calc_ls2(in_idx)


    fig = plt.figure(figsize=(4,4))
    ax = plt.gca()
    #
    ax.plot(wave, true_a_nw, label='True a_nw')
    ax.plot(waves, all_anw, 'o', label='LS2')
    #
    ax.legend()
    #
    plt.show()