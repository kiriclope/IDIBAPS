import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

J1 = []
kappas = []

for gv.MEAN_XI in np.arange(0,1.1,.1):
    gv.init_param()

    try:
        overlap = np.loadtxt(gv.path + 'overlap.dat')
        print(overlap.shape)
        print(overlap)
        kappa = np.mean(np.delete(overlap,[0],axis=1),axis=0)
        print(kappa.shape)
        
        kappas.append(kappa[0])
        J1.append(gv.MEAN_XI)
    except:
        pass
    
print(J1)
print(kappas)

fig = plt.figure('kappa Vs J1')
plt.plot(J1,kappas,'o')

kappa_pos = [abs(i) for i in kappas]
kappa_neg = [-abs(i) for i in kappas]

plt.plot(J1,kappa_pos,'-')
plt.plot(J1,kappa_neg,'-')

plt.xlabel('$J_1$')
plt.ylabel('$\kappa$')

plt.show()
