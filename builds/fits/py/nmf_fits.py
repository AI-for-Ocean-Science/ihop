""" Fits to Loisel+2023 """
import os

# For emcee
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from ihop.emulators import io as emu_io
from ihop import io as ihop_io
from ihop.inference import fitting 
from ihop.inference import io as fitting_io

from IPython import embed

from fit_loisel23 import fit


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Noiseless NMF
    if flg == 0:
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, include_chl=True, 
            X=X, Y=Y)

        fit(edict, n_cores=n_cores)#, debug=True)

    # Noiseless, cut at 600nm
    if flg == 1:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomp = 'nmf'
        Ncomp = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomp, Ncomp, 'Rrs',
            'dense', hidden_list=hidden_list, include_chl=True, 
            X=X, Y=Y)

        # Analysis params
        max_wv=600.

        fit(edict, n_cores=n_cores, max_wv=max_wv)#, debug=True)

    # Noiseless, PCA
    if flg == 2:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('pca', 'pca')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        # Analysis params

        fit(edict, n_cores=n_cores)#, Nspec=300)#debug=True)

    # NMF, abs_sig=1, 4,2
    if flg == 3:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 1.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, debug=True)

    # NMF, abs_sig=2, 4,2
    if flg == 4:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 2.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig)#, debug=True)

    # NMF, abs_sig=5, 4.2
    if flg == 5:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 5.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, debug=True)

    # NMF, abs_sig=2, log prior
    if flg == 6:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (4,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 2.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, 
            use_log_ab=True)#, debug=True)


    # NMF, abs_sig=2, log prior
    if flg == 7:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (3,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 2.
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, 
            use_log_ab=True, debug=True)
            #use_NMF_pos=True, debug=True)


    # NMF, 2,2 Noiseless, NMF_pos
    if flg == 8:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (2,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = None
        edict = emu_io.set_emulator_dict(
            dataset, decomps, Ncomps, 'Rrs',
            'dense', hidden_list=hidden_list, 
            include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, 
            use_log_ab=False, use_NMF_pos=True)

    # NMF, 2,2 abs_sig=1,
    if flg == 9:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (2,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 1.

        edict = emu_io.set_emulator_dict(
                dataset, decomps, Ncomps, 'Rrs',
                'dense', hidden_list=hidden_list, 
                include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, 
                use_log_ab=False, use_NMF_pos=True)

    # NMF, 2,2 abs_sig=2,
    if flg == 10:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (2,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 2.

        edict = emu_io.set_emulator_dict(
                dataset, decomps, Ncomps, 'Rrs',
                'dense', hidden_list=hidden_list, 
                include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, 
                use_log_ab=False, use_NMF_pos=True)

    # NMF, 2,2 abs_sig=5,
    if flg == 11:

        # Emulator
        hidden_list=[512, 512, 512, 256]
        decomps = ('nmf', 'nmf')
        Ncomps = (2,2)
        X, Y = 4, 0
        n_cores = 20
        dataset = 'L23'
        abs_sig = 5.

        edict = emu_io.set_emulator_dict(
                dataset, decomps, Ncomps, 'Rrs',
                'dense', hidden_list=hidden_list, 
                include_chl=True, X=X, Y=Y)

        fit(edict, n_cores=n_cores, abs_sig=abs_sig, 
                use_log_ab=True, use_NMF_pos=False,
                debug=True)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        # NOISELESS
        #flg += 2 ** 0  # 1 -- Noiseless
        #flg += 2 ** 1  # 2 -- Noiseless + cut at 600nm
        #flg += 2 ** 2  # 4 -- Noiseless, PCA 
        #flg += 2 ** 3  # 8 -- Noiseless, INT/NMF

        # PCA with Noise
        #flg += 2 ** 4  # 16 -- PCA, abs_sig=1
        #flg += 2 ** 5  # 32 -- PCA, abs_sig=2
        #flg += 2 ** 6  # 64 -- PCA, abs_sig=5

        # NMF with Noise
        #flg += 2 ** 7  # 128 -- NMF, abs_sig=1
        #flg += 2 ** 8  # 256 -- NMF, abs_sig=2
        #flg += 2 ** 9  # 512 -- NMF, abs_sig=5

        # NMF with Noise + log prior
        #flg += 2 ** 10  # 1024 -- NMF, abs_sig=2, log prior

        # Tests
        #flg += 2 ** 30  # 16 -- L23 + NMF 4,2
        #flg += 2 ** 31  # 16 -- L23 + NMF 4,2
        #flg += 2 ** 32  # 16 -- L23 + NMF 4,2
        flg += 2 ** 33  # 16 -- L23 + NMF 2,2

        
    else:
        flg = sys.argv[1]

    main(flg)