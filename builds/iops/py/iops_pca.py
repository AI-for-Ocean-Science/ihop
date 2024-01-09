""" IOP PCA builds """

from ihop.iops.pca import generate_l23_pca
from ihop.iops.pca import generate_l23_tara_pca

    
if __name__ == '__main__':

    # L23
    generate_l23_pca(clobber=True)
    #generate_l23_tara_pca()  # Broken