import importlib, sys
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import constants as gv
importlib.reload(gv)

import fct_facilities as fac
importlib.reload(fac)

fac.SetPlotParams()

def sigma_0(g):
    return 2*g**2/np.pi*np.arcsin(1-1/4/gv.sigma_1**2)

def kappa(g,eps=1):
    return eps*np.sqrt( np.maximum(0, 4*gv.sigma_1**2-(1+sigma_0(g)) ) /gv.sigma_1)
        
def variance(g):
    return sigma_0(g) + gv.sigma_1 * kappa(g,1)**2

def print_globals():
    print('I0', gv.I0, 'J0', gv.J0, 'J1', gv.J1, 'sigma_1', gv.sigma_1)
    print('TOL', gv.TOL, 'MAX_ITER', gv.MAX_ITER)

print_globals()

g = np.arange(0,3,.01)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot(g,kappa(g,1), color='k')
plt.plot(g,kappa(g,-1), color='k')
plt.xlabel('Random strength $g$')
plt.ylabel('$\kappa$')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

plt.savefig('vanilla_kappaVsg.svg',format='svg')

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot(g, sigma_0(g), color='k')
plt.xlabel('Random strength $g$')
plt.ylabel('$\sigma_0$')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

plt.savefig('vanilla_sig0Vsg.svg',format='svg')

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot(g,variance(g), color='k')
plt.xlabel('Random strength $g$')
plt.ylabel('Variance')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

plt.savefig('vanilla_varVsg.svg',format='svg')

plt.show()
