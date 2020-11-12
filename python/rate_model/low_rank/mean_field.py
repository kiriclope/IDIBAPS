import sys
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as integrate
import scipy.special as special
from scipy.optimize import fsolve, root

import random as rand

import constants as gv
reload(gv)

import mf_utils
from mf_utils import *
reload(sys.modules['mf_utils'])

def branch_kappa_0_eqs(x):
    
    u = x[0] 
    sigma = abs(x[1]) 

    mean_eq = u/np.sqrt(gv.K) - ( gv.I0 + gv.J0 * mean_rate(u, sigma) ) 
    var_eq = sigma - variance(u, sigma) 

    eqs = np.array([mean_eq, var_eq]) 
    return eqs.flatten() 

def branch_kappa_pos_eqs(x):
    
    u = x[0]
    sigma = abs(x[1])
    
    mean_eq = u/np.sqrt(gv.K)  - ( gv.I0 + gv.J0 * mean_rate(u, sigma) ) 
    var_eq = 1 - gv.sigma_1/np.sqrt(1 + sigma) * phi( u/np.sqrt(1+sigma) ) 

    eqs = np.array([mean_eq, var_eq])
    return eqs.flatten()

def branch_kappa_neg_eqs(x): 
    u = x[0] 
    sigma = abs(x[1]) 
    
    mean_eq = u/np.sqrt(gv.K)  - ( gv.I0 + gv.J0 * mean_rate(u, sigma) ) 
    var_eq = u + np.sqrt( (2*np.log(gv.sigma_1) - np.log(1+sigma) - np.log(2*np.pi) )*(1+sigma) ) 
    
    eqs = np.array([mean_eq, var_eq]) 
    return eqs.flatten() 

def SolveStatic (y0, tolerance = 1e-8, backwards = 1):

    # The variable y contains the mean-field variables mu and delta0
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(2)
    eps = 0.1

    while (again==1):

        # Take a step

        new0 = np.sqrt(gv.K) * ( gv.I0 + gv.J0 * mean_rate(y[0], y[1]) )  
        new1 = gv.sigma_1**2 * phi(y[0]/np.sqrt(1+y[1]))**2 - 1 

        y_new[0] = (1-backwards*eps)*y[0] + eps*backwards*new0
        y_new[1] = (1-eps)*y[1] + eps*new1

        # Stop if the variables converge to a number, or zero
        # If it becomes nan, or explodes

        if ( np.fabs(y[1]-y_new[1]) < tolerance*np.fabs(y[1]) ):
            again = 0

        if ( np.fabs(y[1]-y_new[1]) < tolerance ):
            again = 0

        if np.isnan(y_new[0]) == True:
            again = 0
            y_new = [0,0]

        if( np.fabs(y[0])> 1/tolerance  ):
            again = 0
            y_new = [0,0]

        if( y[1]<0 ):
            again = 1
            y_new = [rand.random() for i in range(0,2)]

        y[0] = y_new[0]
        y[1] = y_new[1]

    return y_new

def solve_mf(branch):
    print_globals()

    counter = 0
    x0 = [rand.random()/2 for i in range(0,2)]
    u = x0[0]
    sigma = abs(x0[1])

    if(branch==0):        
        while (any(t>=gv.TOL for t in branch_kappa_0_eqs(x0)) and counter<gv.MAX_ITER):
            x0 = [rand.random()/2 for i in range(0,2)]
            x0[1] = abs(x0[1])
            y = root(branch_kappa_0_eqs, x0, method='lm')

            u = y.x[0]
            sigma = abs(y.x[1])

            x0 = [u, sigma]
            print('iter ', counter, ' sol ', x0, ' error ', branch_kappa_0_eqs(x0))
            if any(t==True for t in np.isnan(x0)):
                x0 = [rand.random() for i in range(0,3)]            
            counter +=1
            
    if(branch==1):
        while (any(t>=gv.TOL for t in branch_kappa_pos_eqs(x0)) and counter<gv.MAX_ITER):
            x0 = [rand.random() for i in range(0,2)]
            x0[1] = abs(x0[1])

            y = root(branch_kappa_pos_eqs, x0)
            u = y.x[0]
            sigma = abs(y.x[1])

            # y = SolveStatic(x0)
            # u = y[0]
            # sigma = y[1]
            
            x0 = [u, sigma]
            print('iter ', counter, ' sol ', x0, ' error ', branch_kappa_pos_eqs(x0))
            if any(t==True for t in np.isnan(x0)):
                x0 = [rand.random() for i in range(0,2)] 
            counter +=1

    if(branch==-1):        
        while (any(t>=gv.TOL for t in branch_kappa_neg_eqs(x0)) and counter<gv.MAX_ITER):
            x0 = [rand.random()/2 for i in range(0,2)]
            x0[1] = abs(x0[1])
            
            y = root(branch_kappa_neg_eqs, x0, method='lm')

            u = y.x[0]
            sigma = abs(y.x[1])

            x0 = [u, sigma]
            print('iter ', counter, ' sol ', x0, ' error ', branch_kappa_neg_eqs(x0))
            if any(t==True for t in np.isnan(x0)):
                x0 = [rand.random() for i in range(0,2)]            
                
            counter +=1
    
    print('mean input %f' % u ) 
    print('total variance %f' % sigma)
    print('sigma_0', variance(u,sigma))
    print('mean rate', mean_rate(u,sigma))
    print('kappa',np.sqrt(np.maximum(0,sigma-variance(u,sigma))/gv.sigma_1))
    return y

<
