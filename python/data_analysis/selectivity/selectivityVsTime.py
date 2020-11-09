from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

from scipy.ndimage.filters import gaussian_filter1d

import data.fct_facilities as fac
importlib.reload(fac) ; 
fac.SetPlotParams() 

import preprocessing as pre
from plotting import *

for gv.mouse in [gv.mice[2]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times()
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_frame_rate() 
        data.get_bins(t_start=0) 
        
        gv.duration = X.shape[2]/gv.frame_rate 
        gv.time = np.linspace(0, gv.duration, X.shape[2]) 
        
        trial_averages = []
        for gv.trial in gv.trials: 
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1)
            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 

            X_S1 = np.mean(X_S1, axis=0) 
            X_S2 = np.mean(X_S2, axis=0) 

            # idx = np.where(np.mean(X_S1[:,gv.bins_ED],axis=1)<=0.01) 
            # X_S1 = np.delete(X_S1, idx, axis=0) 
            # X_S2 = np.delete(X_S2, idx, axis=0)

            # idx = np.where(np.mean(X_S1[:,gv.bins_ED],axis=1)<=0.01) 
            # X_S1 = np.delete(X_S1, idx, axis=0) 
            # X_S2 = np.delete(X_S2, idx, axis=0) 

            sel_idx = (X_S1-X_S2)/(X_S1+X_S2+.000000001) 
            sel_idx = sel_idx.flatten()
            
            sel_idx = sel_idx.reshape(X_S1.shape[0], X_S1.shape[1]) 
            print(sel_idx.shape)
            
            idx = np.argsort(np.mean(sel_idx[:,gv.bins_stim], axis=1)) 
            sel_idx_sort = gaussian_filter1d(sel_idx[idx],3) 
            print(sel_idx_sort.shape)
            
            figname = '%s_%s_%s_selectivity_idx' % (gv.mouse, gv.session, gv.trial)
            ax = plt.figure(figname).add_subplot() 
            im = ax.imshow(sel_idx_sort, cmap='jet', vmin=-1., vmax=1., origin='lower', extent = [-2, gv.duration-2, 0, sel_idx.shape[0]], aspect='auto') 
            
            plt.xlabel('Time (s)')
            plt.ylabel('neuron #')
                
            plt.xlim([-2,gv.t_test[1]-2])
        
            ax.grid(False)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('selectivity index', rotation=90) 
            
            add_hlines(figname) 
            figdir = figDir() 
            save_fig(figname, figdir) 
