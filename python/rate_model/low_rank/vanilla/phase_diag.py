import importlib, sys 
from importlib import reload 

import numpy as np 
import matplotlib.pyplot as plt 

import constants as gv 
importlib.reload(gv) 

import fct_facilities as fac 
importlib.reload(fac) 

fac.SetPlotParams() 

def gVsSig1(sigma_1): 
    g = np.sqrt(np.pi/2 * (4*sigma_1**2-1)/np.arcsin(1-1/4/sigma_1**2)) 
    g[np.isnan(g)]=1 
    return g 

sigma_1 = np.arange(0.5,4,.01)
g = gVsSig1(sigma_1)

print('g', g)
print('sigma_1', sigma_1)

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot(g, sigma_1, color='k')
plt.xlabel('Random strength $g$')
plt.ylabel('$\sigma_1$')

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')

plt.axvline(x=1.25,ymin=0, ymax=0.25, color='k')
plt.axhline(y=0.5,xmin=0, xmax=0.5, color='k')

plt.xlim([0,4])
plt.ylim([0,4])

# plt.savefig('vanilla_kappaVsg.svg',format='svg')
