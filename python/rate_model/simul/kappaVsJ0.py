import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(sys.modules['constants'])

gv.IF_TRIALS = 1
gv.IF_FF = 0

gv.MEAN_XI = -0.0 
gv.VAR_XI = 5.0 

gv.IF_LEFT_RIGHT = 0
gv.MEAN_XI_LEFT = 0.0
gv.VAR_XI_LEFT = 5.0 
gv.MEAN_XI_RIGHT = 0.0
gv.VAR_XI_RIGHT = 5.0 
gv.RHO = 1.

gv.IF_FF = 0
gv.MEAN_FF = 0.0 
gv.VAR_FF = 5.0 
gv.VAR_ORTHO = 0.0 

for gv.TRIAL_ID in np.arange(0,9,1):
    J0 = []
    kappas = []
    rates = []
    sigmas = []
    
    for gv.J0 in np.concatenate((np.arange(0,.1,.01),np.arange(.1,1.8,.1))):
        gv.folder = 'I0_%2.f_J0_%.2f' % (gv.I0, gv.J0)
        gv.init_param()
    
        try :
            overlap = np.loadtxt(gv.path + 'overlap.dat') ;
            filter_rates = np.loadtxt(gv.path + 'filter_rates.dat') ;

            kappa = np.mean(np.delete(overlap,[0],axis=1),axis=0) ; 
            rate = np.mean(np.mean(np.delete(filter_rates,[0],axis=1),axis=1),axis=0) ;
            var = np.var(np.mean(np.delete(filter_rates,[0],axis=1),axis=1),axis=0) ;
            
            kappas.append(kappa[0])
            rates.append(rate)
            sigmas.append(np.fabs(var-gv.VAR_XI*kappa[0]**2))
            J0.append(gv.J0)
                
        except :
            pass
    
    print(J0)
    print(kappas)

    plt.figure('kappa Vs J0')
    plt.plot(J0,kappas,'o')

    # kappa_pos = [abs(i) for i in kappas]
    # kappa_neg = [-abs(i) for i in kappas]

    # plt.plot(J0,kappa_pos,'-')
    # plt.plot(J0,kappa_neg,'-')

    plt.xlabel('$J_0$')
    plt.ylabel('$\kappa$')
    
    plt.figure('r0 Vs J0')
    plt.plot(J0,rates,'o')

    plt.figure('sigma0 Vs J0')
    plt.plot(J0,sigmas,'o')

plt.show()
# plt.ylim([-1,1])
