import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

gv.init_param()

filter_inputs = np.loadtxt(gv.path + 'inputs.dat') ;

time = filter_inputs[:,0]
print(time)

inputs = np.mean(np.delete(filter_inputs,[0],axis=1),axis=0)
print(inputs.shape)

mean_inputs = np.mean(inputs)
print(mean_inputs)

plt.hist(inputs)
plt.xlabel('inputs (a.u.)')
plt.ylabel('count')
