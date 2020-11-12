import numpy as np 

import scipy 
from scipy.optimize import fsolve 
import scipy.special as special

from functools import partial 

import fct_facilities as fac
from fct_integrals import *

import constants as gv

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Solve mean-field equations

### Zero-mean solutions, corresponding to the central solution, solved through iteration

def SolveStaticZero (y0, tolerance = 1e-8, backwards = 1):

    # The variable y contains the mean-field variables mu and delta0
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(2)
    eps = 0.2
    n_iter = 0
    
    while (again==1 and n_iter<gv.MAX_ITER):
        # Take a step
        n_iter = n_iter + 1  
        
        new0 = np.sqrt(gv.K) * ( gv.I0 - gv.J0 * intPhi(y[0],y[1]) ) # net input 
        new1 = gv.J0 * gv.J0 * intPhiSq(y[0], y[1]) # total variance 
        
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
    
        y[0] = y_new[0]
        y[1] = y_new[1]

    return y_new

### Non-zero solutions, solved through iteration
def SolveStaticNonZero (y0, tolerance = 1e-8, backwards = 1):

    # The variable y contains the mean-field variables mu, delta0 and deltainf
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0) 
    y_new = np.ones(2) 
    eps = .2 
    n_iter = 0 
    
    while (again==1 and n_iter<gv.MAX_ITER):
        # Take a step
        n_iter = n_iter + 1 

        new0 = np.sqrt(gv.K) * ( gv.I0 - gv.J0 * intPhi(y[0],y[1]) ) # net input 
        new1 = gv.VAR_XI**2 * phi(y[0]/np.sqrt(1+y[1]))**2 - 1 # kappa 
        
        y_new[0] = (1-backwards*eps)*y[0] + eps*backwards*new0 
        y_new[1] = (1-eps)*y[1] + eps*new1 
        
        # Stop if the variables converge to a number, or zero 
        # If it becomes nan, or explodes

        if( np.fabs(y[0])> 1/tolerance ):
            again = 0
            y_new = [0,0]

        if ( np.fabs(y[1]-y_new[1]) < tolerance*np.fabs(y[1]) ):
            again = 0

        if ( np.fabs(y[1]-y_new[1]) < tolerance ):
            again = 0

        if np.isnan(y_new[0]) == True:
            again = 0
            y_new = [0,0]
        
        y[0] = y_new[0]
        y[1] = np.fabs(y_new[1])

        # print('iter', n_iter, 'y[0]', y[0],'y_new[0]', y_new[0], 'err', np.fabs(y[0]-y_new[0]) ) 
        # print('iter', n_iter, 'y[1]', y[1],'y_new[1]', y_new[1], 'err', np.fabs(y[1]-y_new[1]) ) 

    return y_new

def mean_rate(u, sigma):
    return intPhi(u, sigma)

def variance(u, sigma): 
    var = np.maximum(0, gv.J0**2 * (intPhi(u, sigma) - 2 * special.owens_t(u/np.sqrt(1.0 + sigma), 1/np.sqrt(1.0 + 2*sigma) ) ) )
    return var 

def overlap(u,sigma):
    kappa = np.sqrt(np.maximum(0,sigma-variance(u,sigma))/gv.VAR_XI)
    return kappa 

def SolveStatic (y0, tolerance = 1e-10, backwards = 1):  # y[0]=mu, y[1]=Delta0

    # The variable y contains the mean-field variables mu, delta0 and kappa
    # Note that, for simplicity, only delta0 and one first-order statistics (kappa) get iterated
    # The variable backwards can be set to (-1) to invert the flow of iteration and reach unstable solutions

    again = 1
    y = np.array(y0)
    y_new = np.ones(3)
    eps = 0.2

    Mm, Mn, Mi, Sim, Sin, Sini, Sip = VecPar
    Sii = np.sqrt( (Sini/Sin)**2 + Sip**2 )

    while (again==1):

        # Take a step

        mu = Mm * y[2] + Mi
        new1 = g*g * PhiSq(mu, y[1]) + Sim**2 * y[2]**2 + Sii**2
        new2 =  Mn * Phi(mu, y[1]) + Sini * Prime(mu, y[1])

        y_new[0] = Mm * new2 + Mi
        y_new[1] = (1-eps)*y[1] + eps*new1
        y_new[2] = (1-backwards*eps)*y[2] + backwards*eps*new2

        # Stop if the variables converge to a number, or zero
        # If it becomes nan, or explodes
        
        if( np.fabs(y[1]-y_new[1]) < tolerance*np.fabs(y[1]) and np.fabs(y[2]-y_new[2]) < tolerance*np.fabs(y[2]) ):
            again = 0

        if( np.fabs(y[1]-y_new[1]) < tolerance and np.fabs(y[2]-y_new[2]) < tolerance ):
            again = 0

        if np.isnan(y_new[0]) == True:
            again = 0
            y_new = [0,0,0]

        if( np.fabs(y[2])> 1/tolerance ):
            again = 0
            y_new = [0,0,0]
    
        y[0] = y_new[0]
        y[1] = y_new[1]
        y[2] = y_new[2]

    return y_new
