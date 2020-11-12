import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

sigma_1 = []
kappas = []

for gv.VAR_XI in np.arange(0,1.1,.05): 
    gv.init_param()
    try:
        overlap = np.loadtxt(gv.path + 'overlap.dat')
        print(overlap.shape)
        print(overlap)
        kappa = np.mean(np.delete(overlap,[0],axis=1),axis=0)
        print(kappa.shape)
        
        kappas.append(kappa[0])
        sigma_1.append(gv.VAR_XI)
    except:
        pass

print(sigma_1)
print(kappas)

fig = plt.figure('kappa Vs sigma_1')
plt.plot(sigma_1,kappas,'o')

kappa_pos = [abs(i) for i in kappas]
kappa_neg = [-abs(i) for i in kappas]

plt.plot(sigma_1,kappa_pos,'-')
plt.plot(sigma_1,kappa_neg,'-')

plt.xlabel('$\sigma_1$')
plt.ylabel('$\kappa$')

plt.show()
