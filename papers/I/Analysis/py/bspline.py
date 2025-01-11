""" B-spline Decomposition """
import numpy as np
import matplotlib.pyplot as plt

from ihop.iops import io as iops_io

from pypeit.core import fitting
from pypeit.bspline import bspline

from ihop.iops import io as iops_io

from IPython import embed

def decompose_loisel23(iop:str,
    decomp:str='bsp', Ncomp:int=10, 
    X:int=4, Y:int=0, debug:bool=False,
    plot:bool=False): 
    """
    Decompose one of the Loisel+23 IOPs.

    Args:
        decomp (str): Decomposition method ('pca' or 'nmf').    
        Ncomp (int): Number of components for NMF.
        iop (tuple): The IOP to decompose. 
        X (int, optional): X-coordinate of the training data.
        Y (int, optional): Y-coordinate of the training data.

    """
    # Loop on IOP
    outfile = iops_io.loisel23_filename(decomp, iop, Ncomp, X, Y)

    # Breakpoints
    bkpts3 = [350.] + np.linspace(352., 660., 10).tolist() + [
        670., 680., 690, 705, 725., 745., 755]
    bkpts3 = np.array(bkpts3)

    # Load training data
    spec, wave, Rs, ds = iops_io.load_loisel23_iop(
        iop, X=X, Y=Y, remove_water=True)

    wv64 = wave.astype(np.float64)

    # Init
    my_bspline = bspline(wave, nord=3, fullbkpt=bkpts3)

    # Go
    sv_coeffs = []
    sv_fits = []
    diffs = []
    rdiffs = []

    for ss in range(spec.shape[0]):
        _, yfit = my_bspline.fit(wv64, spec[ss], 
                                    np.ones_like(wv64)) # Equal weights
        # RMSE
        diffs.append(spec[ss]-yfit)
        rdiffs.append((spec[ss]-yfit)/np.maximum(spec[ss],1e-4))
        # Save
        sv_coeffs.append(my_bspline.coeff.copy())
        sv_fits.append(yfit)
        #if debug and ss == 0:
        #    embed(header='62 of bspline')

    # Stats
    if plot:
        rmses = np.sqrt(np.mean(np.array(diffs)**2, axis=0))
        plt.clf()
        plt.plot(wave, rmses)
        plt.show()
        rrmses = np.sqrt(np.mean(np.array(rdiffs)**2, axis=0))
        plt.clf()
        plt.plot(wave, rrmses)
        plt.show()
        embed(header='77 of bspline')

    # Save
    if debug:
        embed(header='81 of bspline')
        return
    outfile = iops_io.loisel23_filename(
        decomp, iop, Ncomp, X, Y)
    outputs = dict(
        data=spec, 
        breakpoints=bkpts3,
        wave=wave,
        Rs=Rs,
        coeffs=np.array(sv_coeffs))
    np.savez(outfile, **outputs)
    print(f"Wrote: {outfile}")

if __name__ == '__main__':
    decompose_loisel23('a')#, plot=False, debug=True)