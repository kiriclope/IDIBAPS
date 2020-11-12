import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

gv.init_param()

filter_rates = np.loadtxt(gv.path + 'filter_rates.dat') ;

time = filter_rates[:,0]
print(time)

rates = np.delete(filter_rates,[0],axis=1)
print(rates.shape)

for i in range(0,5):
    plt.plot(time, rates[:,i])

mean_rates = np.mean(rates,axis=1) 
print(mean_rates.shape)

plt.plot(time, mean_rates)
plt.ylabel('rates (Hz)')
plt.xlabel('time (ms)')

avg_rates = np.mean(rates, axis=0)
print('mean', np.mean(avg_rates), 'var', np.var(avg_rates)) 
