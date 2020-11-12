import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

trials = []
kappas = []

for gv.TRIAL_ID in np.arange(1,11,1): 
    gv.init_param()
    
    filename = 'trial_%d/overlap.dat' % gv.TRIAL_ID
    print(filename)
    
    try :
        overlap = np.loadtxt(gv.path + filename)
        kappa = np.mean(np.delete(overlap,[0],axis=1),axis=0)

        kappas.append(kappa[0])
        trials.append(gv.TRIAL)

    except :
        pass
        
fig = plt.figure('kappa Vs Trials')
plt.hist(kappas)

plt.xlabel('$\kappa$')
plt.ylabel('count')

plt.show()
print('number of trials:', len(kappas))
print('mean kappa', np.mean(kappas))
