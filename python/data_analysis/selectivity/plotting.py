import os, sys
import matplotlib.pyplot as plt

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 
import data.constants as gv 

def figDir():
    script_dir = os.path.dirname(__file__) 
    figdir = script_dir + '/figs'
    print('saving in', figdir)

    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    return figdir

def add_vlines(figname):
    plt.figure(figname) 
    plt.ahvline(y=gv.t_sample[0]-2, c='k', ls='-') # sample onset

    plt.ahvline(y=gv.t_early_delay[0]-2, c='k', ls='--') 
    plt.ahvline(y=gv.t_early_delay[1]-2, c='k', ls='--') # DPA early delay
    
    plt.ahvline(y=gv.t_DRT_delay[0]-2, c='r', ls='--') #DRT delay
    plt.ahvline(y=gv.t_DRT_delay[1]-2, c='r', ls='--') 
        
    plt.ahvline(y=gv.t_late_delay[0]-2, c='k', ls='--')
    plt.ahvline(y=gv.t_late_delay[1]-2, c='k', ls='--') # DPA late delay

def add_hlines(figname):
    plt.figure(figname) 
    plt.axvline(x=gv.t_sample[0]-2, c='k', ls='-') # sample onset

    plt.axvline(x=gv.t_early_delay[0]-2, c='k', ls='--') 
    plt.axvline(x=gv.t_early_delay[1]-2, c='k', ls='--') # DPA early delay
    
    plt.axvline(x=gv.t_DRT_delay[0]-2, c='r', ls='--') #DRT delay
    plt.axvline(x=gv.t_DRT_delay[1]-2, c='r', ls='--') 
        
    plt.axvline(x=gv.t_late_delay[0]-2, c='k', ls='--')
    plt.axvline(x=gv.t_late_delay[1]-2, c='k', ls='--') # DPA late delay

def save_fig(figname, figdir):
    plt.figure(figname) 
    plt.savefig(figdir + figname +'.svg',format='svg') 
