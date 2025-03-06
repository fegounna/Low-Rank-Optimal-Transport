import numpy as np
import matplotlib.pylab as pl
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from ot.bregman import screenkhorn


def generate_data_1D(n, gaussian_from, gaussian_to):
    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=20, s=5)  # m= mean, s= std
    b = gauss(n, m=60, s=10)

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    M /= M.max()
    
    return x, a, b, M

def plot_distrib_1D(x, a, b, M):
    pl.figure(1, figsize=(6.4, 3))
    pl.plot(x, a, "b", label="Source distribution")
    pl.plot(x, b, "r", label="Target distribution")
    pl.legend()

    # plot distributions and loss matrix
    pl.figure(2, figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, M, "Cost matrix M")

def benchmark_1D(liste_n, gaussian_from=(10, 20), gaussian_to=(40, 30)):
    
    for n in liste_n:

        
        # test of our method
    
    