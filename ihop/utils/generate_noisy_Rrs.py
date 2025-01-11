import numpy as np

# Rrs is the input Rrs with no noise, n x m dimension, 
# n is the total number of Rrs spectra, m is the number of wavelength
# Rrs_noisy = Rrs * (1 + U * e1 + C * e2)
# U * e1: uncorrelated noise, U is typically set to be 0.01
# e1 is an array (size m) of gaussian distributed variables with mean of 0 and standard deviation of 0.01
# C * e2: correlated noise, C is typically set to be 0.005
# e2 is a gaussian distributed variable with mean of 0 and standard deviation of 1 (same for all wavelength)
# seed (integer) is used to initialize the random number generator; with the same seed the random values are the same

def generate_noisy_Rrs(Rrs,U,C,seed):
    np.random.seed(seed)
    if isinstance(Rrs,list):
        Rrs=np.array(Rrs)
        
    if Rrs.ndim==2:
        e1=np.random.normal(0,1,Rrs.shape[1])
    else:
        e1=np.random.normal(0,1,Rrs.size)
        
    e2=np.random.normal(0,1,1)
    
    Rrs_noisy=Rrs*(1+U*e1+C*e2)
    
    return Rrs_noisy


# example 1

Rrs=np.array([0.0059,0.0074,0.0111,0.0150,0.0165,0.0172,0.0084,0.0062,0.0058])

Rrs_noisy=generate_noisy_Rrs(Rrs,0.01,0.005,10)

print(Rrs_noisy)

print('============================')
# example 2

Rrs=[0.0059,0.0074,0.0111,0.0150,0.0165,0.0172,0.0084,0.0062,0.0058]

Rrs_noisy=generate_noisy_Rrs(Rrs,0.01,0.005,10)

print(Rrs_noisy)

print('============================')

# example 3
Rrs=np.array([[0.0059,0.0074,0.0111,0.0150,0.0165,0.0172,0.0084,0.0062,0.0058],[0.0059,0.0074,0.0111,0.0150,0.0165,0.0172,0.0084,0.0062,0.0058]])

Rrs_noisy=generate_noisy_Rrs(Rrs,0.01,0.005,10)

print(Rrs_noisy)

print('============================')