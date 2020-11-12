import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

gv.init_param()

filter_rates = np.loadtxt(gv.path + 'filter_rates.dat') ;
print('filter_rates')
print(filter_rates.shape)

time = filter_rates[:,0]
print(time)

rates = np.mean(np.delete(filter_rates,[0],axis=1),axis=0)
print(rates.shape)

xi = np.loadtxt(gv.path + 'low_rank_xi.dat') ;
print('xi')
print(xi.shape)

kappa = np.dot(xi, rates)/10000.
print(kappa)
