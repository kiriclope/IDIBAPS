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

gv.data_type = 'rates'

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

            X_S1_S2 = np.array([X_S1, X_S2])
            
            for i, sample in enumerate(gv.samples):

                # average over trials 
                X_avg = np.mean(X_S1_S2[i], axis=0)
                X_avg = pre.z_score(X_avg) 
            
                #average over ED to sort
                X_ED = np.mean(X_avg[:,gv.bins_MD], axis=1) 

                # sorting rates
                if i==0 and 'ND' in gv.trial:
                    idx = np.argsort(X_ED)
                X_sort = gaussian_filter1d(X_avg[idx],3)

                figname = '%s_%s_%s_%s_z_scores' % (gv.mouse, gv.session, gv.trial, gv.samples[i]) 
                ax = plt.figure(figname).add_subplot() 
                im = ax.imshow(X_sort, cmap='jet', vmin=-1, vmax=1, origin='lower', extent = [-2, gv.duration-2, 0, gv.n_neurons], aspect='auto')
                plt.xlabel('Time (s)')
                plt.ylabel('neuron #')
                
                plt.xlim([-2,gv.t_test[1]-2])
                
                ax.grid(False)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Norm. rate (z-score)', rotation=90) 

                add_hlines(figname)
                figdir = figDir()
                save_fig(figname, figdir) 
