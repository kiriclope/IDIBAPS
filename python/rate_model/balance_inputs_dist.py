import numpy as np

import scipy.integrate as integrate
import scipy.special as special

from scipy.optimize import fsolve, root

import random as rand

import globals
from globals import *

import importlib, sys
from importlib import reload
importlib.reload(sys.modules['globals'])

def inputs_dist():
    x0 = [rand.random()/2.0 for i in range(0,2*n_pop)]
    # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ...
    y = root(self_consistent_eqs,x0,method='lm')

    mean_inputs = y.x[0:n_pop]
    var_inputs = y.x[n_pop:2*n_pop]

    print( vec_quench_avg_Phi(mean_inputs,var_inputs) )
    return y

def self_consistent_eqs(x):
    mean_input = x[0:n_pop]
    var_input = x[n_pop:2*n_pop]

    mean_eq = mean_input / np.sqrt(K) - ( ext_inputs + J.dot(vec_quench_avg_Phi(mean_input, var_input) ) ) # add [0] if using quad 
    var_eq = var_input - (J*J).dot(vec_quench_avg_Phi2(mean_input, var_input)[0])

    eqs = np.array([mean_eq, var_eq])
    return eqs.flatten()

def phi(x): # normal distribution
    return np.exp(-0.5*x*x) / np.sqrt(2*np.pi)

def Phi(x): # CDF of the normal distribution
    return 0.5 *(1.0 + special.erf(x/np.sqrt(2)) ) 
    # if(x>0):
    #     return x
    # else:
    #     return 0
    
def integrand(x,a,b):
    return Phi(a + np.sqrt(b) * x) * phi(x)

def integrand2(x,a,b):
    return Phi(a + np.sqrt(b) * x) * Phi(a + np.sqrt(b) * x) * phi(x)

def quench_avg_Phi(a,b):
    # return integrate.quad(integrand,-np.inf,np.inf,args=(a,b))
    return Phi(a/np.sqrt(1.0 + b))

def quench_avg_Phi2(a, b):
    return integrate.quad(integrand2,-np.inf,np.inf,args=(a,b))

vec_quench_avg_Phi = np.vectorize(quench_avg_Phi)
vec_quench_avg_Phi2 = np.vectorize(quench_avg_Phi2)

