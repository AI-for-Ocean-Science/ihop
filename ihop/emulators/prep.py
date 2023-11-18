""" Prep for emulator action """

import numpy as np

from ihop.hydrolight import loisel23

from IPython import embed

def full_loisel23(X:int=4, Y:int=0):
    
    ds = loisel23.load_ds(X, Y)

    a = ds.a.data
    bb = ds.bb.data
    Rs = ds.Rrs.data

    # Concatenate a and bb
    abb = np.concatenate((a, bb), axis=1)

    # Save
    outfile = f'loisel23_X{X}Y{Y}_full.npz'
    np.savez(outfile, abb=abb, Rs=Rs)
    print(f"Wrote: {outfile}")


if __name__ == '__main__':

    full_loisel23()