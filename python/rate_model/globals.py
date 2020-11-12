import numpy as np

dir = "test"

n_pop = 1
n_neurons = 1
K = 500

m0 = .01

from get_param import *
ext_inputs, J, Tsyn = get_param(n_pop, dir) ;
ext_inputs = ext_inputs * m0 ;

if(K!=np.inf):
    path = '../cpp/rate_model/simulations/%dpop/%s/N%d/K%d/' %(n_pop, dir, n_neurons, K)
    print('reading simulations data from:')
    print(path)
