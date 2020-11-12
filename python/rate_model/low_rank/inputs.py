import importlib, sys
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy.special as special

from scipy.optimize import fsolve, root

import random as rand

import constants
from constants import *
importlib.reload(sys.modules['constants'])

global J0
J0 = -1
J1 = mean_xi
sigma_1 = var_xi

global TOL
TOL = 1e-3

global MAX_ITER
MAX_ITER = 100 ;

def inputs_dist():
    x0 = [rand.random()/2.0 for i in range(0,3)]
    # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ...
    y = root(self_consistent_eqs,x0,method='lm')

    u = y.x[0]
    sigma_0 = y.x[1]
    kappa = y.x[2]
    print('mean_rate %f' %(mean_rate(u, sigma_0, kappa))) 
    print('variance %.3f' % variance(u, sigma_0, kappa))
    print('kappa %f', overlap(u, sigma_0, kappa) )
    return y

def self_consistent_eqs(x):
    global J0, J1, sigma_1
    u = x[0]
    sigma_0 = x[1]
    kappa = x[2]

    mean_eq = u/np.sqrt(K)  - ( ext_inputs[0] + J0 * mean_rate(u, sigma_0, kappa ) + J1 * kappa/np.sqrt(K) )  
    var_eq = sigma_0 - variance(u,sigma_0,kappa)
    kappa_eq = kappa - overlap(u,sigma_0,kappa)
    # kappa_eq = 1 - overlap(u,sigma_0,kappa)
    
    eqs = np.array([mean_eq, var_eq, kappa_eq])
    return eqs.flatten()

def branch_kappa_0_eqs(x):
    global J0, J1, sigma_1
    u_0 = x[0]
    sigma_0 = x[1]

    mean_eq = u_0/np.sqrt(K) - ( ext_inputs[0] + J0 * mean_rate(u_0, sigma_0, 0 ) )  
    var_eq = sigma_0 - variance(u_0, sigma_0, 0)
    
    eqs = np.array([mean_eq, var_eq])
    return eqs.flatten()

def phi(x): # normal distribution
    return np.exp(-0.5*x*x) / np.sqrt(2*np.pi)

def Phi(x): # CDF of the normal distribution
    return 0.5 *(1.0 + special.erf(x/np.sqrt(2)) )

def mean_rate(u, sigma_0, kappa):
    global J0, J1, sigma_1
    return Phi( u/np.sqrt(1.0 + sigma_0 + sigma_1 * kappa * kappa) )

def variance(u, sigma_0, kappa):
    global J0, J1, sigma_1
    return J0*J0 * mean_rate(u, sigma_0, kappa) - 2 * special.owens_t(u/np.sqrt(1.0 + sigma_0 + sigma_1 * kappa * kappa), 1/np.sqrt(1.0 + 2*sigma_0 + 2*sigma_1 * kappa * kappa))

def overlap(u,sigma_0,kappa):
    global J0, J1, sigma_1
    return J1 * mean_rate(u,sigma_0,kappa) + sigma_1*kappa / np.sqrt(1+sigma_0+sigma_1*kappa*kappa) * phi(u/np.sqrt(1+sigma_0+sigma_1*kappa*kappa))
    #return sigma_1 / np.sqrt(1+sigma_0+sigma_1*kappa*kappa) * phi(u/np.sqrt(1+sigma_0+sigma_1*kappa*kappa))

def plot_kappaVsJ0():
    global J0, J1, sigma_1
    global TOL, MAX_ITER

    J0s = []
    kappas = []
    
    for J0 in np.arange(-.1,-1.,-.1):
        x0 = [rand.random() for i in range(0,3)]
        counter = 0

        print(TOL)
        print(MAX_ITER)
        
        while (any(t>=TOL for t in self_consistent_eqs(x0)) and counter<MAX_ITER) :
            counter += 1
            x0 = [rand.random() for i in range(0,3)]
        
            # y = fsolve(self_consistent_eqs, x0) # to fix fsolve is not always converging ...
            # u = y[0]
            # sigma_0 = y[1]
            # kappa = y[2]
        
            y = root(branch_kappa_0_eqs,x0,method='lm')
            u = y.x[0]
            sigma_0 = y.x[1]
            kappa = y.x[2]
            
        print('J0 %.2f iter %d' % (J0,counter) )
        print('x0 %.3f %.3f %.3f' % (x0[0],x0[1],x0[2]) )
        print('ERR0R %.3f %.3f %.3f' % (self_consistent_eqs(x0)[0],self_consistent_eqs(x0)[1],self_consistent_eqs(x0)[2]) )

        J0s.append(abs(J0))
        kappas.append(kappa)
        
        # print('mean_rate %.3f' % (mean_rate(u, sigma_0, kappa)) ) 
        # print('variance %.3f' % sigma_0)
        # print('kappa %f' % kappa ) 
            
    plt.plot(J0s,kappas,'-o')

# if __name__ == "__main__":    
#     # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ...
#     # u = y[0]
#     # sigma_0 = y[1]
#     # kappa = y[2]
    
#     J0 = []
#     kappas = []
    
#     for J[0][0] in np.arange(-0.1,-.1,-.1):
#         x0 = [rand.random() for i in range(0,3)]

#         print(J[0][0])
#         y = root(self_consistent_eqs,x0,method='lm')
#         u = y.x[0]
#         sigma_0 = y.x[1]
#         kappa = y.x[2]

#         J0.append(J0)
#         kappas.append(kappa)
        
#         print('mean_rate %.3f' % (mean_rate(u, sigma_0, kappa)) ) 
#         print('variance %.3f' % (variance(u, sigma_0, kappa)))
#         print('kappa %f' % (overlap(u, sigma_0, kappa)) ) 

#     # plt.plot(J0,kappas,'-o')
