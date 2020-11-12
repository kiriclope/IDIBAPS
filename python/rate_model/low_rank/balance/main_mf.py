#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

#### code for the mf of one balanced population with low rank xi: 

#### CODE 1a: spontaneous activity in rank-one networks: DMF theory (related to Fig. 1C)
#### This code computes the DMF solutions (and their stability) for increasing values of the random strength g
#### The overlap direction is defined along the unitary direction (rho = 0, see Methods)
#### Within the DMF theory, activity is then described in terms of mean (mu) and variance (delta) of x

#### Note that the Data/ folder is empty to begin; this code needs to be run with the flag doCompute = 1
#### at least once

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Import functions

import sys
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import random as rand

import fct_integrals as integ
import fct_facilities as fac
import fct_mf as mf

import constants as gv
reload(gv)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# We solve separately the DMF equations corresponding to stationary and chaotic states

J0_list = []
kappa_list = []
sigma_0_list = []
rate_list = []

for gv.J0 in np.linspace(.001,1.26,200):
    y0 = [rand.random() for i in range(0,2)]
    sol = mf.SolveStaticNonZero(y0)

    u = sol[0]
    delta_0 = sol[1]
    
    kappa = mf.overlap(u,delta_0)
    sigma_0 = mf.variance(u,delta_0)
    rate = mf.mean_rate(u, delta_0)
    
    J0_list.append(gv.J0)
    kappa_list.append(kappa) 
    sigma_0_list.append(sigma_0)
    rate_list.append(rate)

for gv.J0 in np.linspace(1.26,1.8,200):
    y0 = [rand.random() for i in range(0,2)]
    sol = mf.SolveStaticZero(y0)

    u = sol[0]
    delta_0 = sol[1]
    
    kappa = mf.overlap(u,delta_0)
    sigma_0 = mf.variance(u,delta_0)
    rate = mf.mean_rate(u, delta_0)
    
    J0_list.append(gv.J0)
    kappa_list.append(kappa) 
    sigma_0_list.append(sigma_0)
    rate_list.append(rate)

kappas_neg = [-abs(i) for i in kappa_list]

fac.SetPlotParams()

plt.figure('kappa Vs J0')
# ax0 = plt.axes(frameon=True)

plt.xlabel('$J_0$')
plt.ylabel('$\kappa$')

plt.plot(J0_list, kappa_list,'-k')
plt.plot(J0_list, kappas_neg,'-k')

# ax0.spines['top'].set_visible(False)
# ax0.spines['right'].set_visible(False)
# ax0.yaxis.set_ticks_position('left')
# ax0.xaxis.set_ticks_position('bottom')

plt.savefig('balance_kappaVsJ0.svg',format='svg')

plt.figure('sigma0 Vs J0')
plt.xlabel('$J_0$')
plt.ylabel('$\sigma_0$')

plt.plot(J0_list, sigma_0_list,'-k')

plt.savefig('balance_sigma0VsJ0.svg',format='svg')

plt.figure('r0 Vs J0')
plt.xlabel('$J_0$')
plt.ylabel('$r_0$')

plt.plot(J0_list, rate_list,'-k')
plt.savefig('balance_r0VsJ0.svg',format='svg')

