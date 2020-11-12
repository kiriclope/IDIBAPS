import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

gv.J0 = 1.0
gv.MEAN_XI = -0.0
gv.VAR_XI = 5.0

for gv.TRIAL_ID in np.arange(0,10,1):
    var = []
    kappas = []
    
    for gv.VAR_ORTHO in np.arange(0,11,1): 
        gv.init_param()
        
        try :
            if gv.IF_TRIALS : 
                filename = 'trial_%d/overlap.dat' % gv.TRIAL_ID ;
                overlap = np.loadtxt(gv.path + filename) 
            else :
                overlap = np.loadtxt(gv.path + 'overlap.dat') ;
            
            kappa = np.mean(np.delete(overlap,[0],axis=1),axis=0) ;            
            kappas.append(kappa[0])
            var.append(gv.VAR_ORTHO)
                
        except :
            pass
    
    print(var)
    print(kappas)

    fig = plt.figure('kappa Vs var ortho')
    plt.plot(var,kappas,'-o')

    # kappa_pos = [abs(i) for i in kappas]
    # kappa_neg = [-abs(i) for i in kappas]

    # plt.plot(J0,kappa_pos,'-')
    # plt.plot(J0,kappa_neg,'-')

    plt.xlabel('$\sigma_{ortho}$')
    plt.ylabel('$\kappa$')

plt.show()
# plt.ylim([-1,1])
