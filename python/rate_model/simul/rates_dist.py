import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

gv.init_param()

filter_rates = np.loadtxt(gv.path + 'filter_rates.dat') ;
print(filter_rates.shape)

time = filter_rates[:,0]
print(time)

rates = np.mean(np.delete(filter_rates,[0],axis=1),axis=0)
print(rates.shape)

mean_rates = np.mean(rates)
print(mean_rates)

logrates = [np.log(i) for i in rates if i>=.01]

plt.hist(logrates)
plt.xlabel('rates (Hz)')
plt.ylabel('count')
plt.show()
