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

pal = ['r','b']

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_frame_rate() 
        data.get_bins(t_start=1) 
        
        gv.duration = X.shape[2]/gv.frame_rate 
        gv.time = np.linspace(0, gv.duration, X.shape[2]) 

        theta = np.linspace(0, 2*np.pi, gv.n_neurons)
        trial_averages = []
        
        for gv.trial in gv.trials:
            
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1)
            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 

            X_S1_S2 = np.array([X_S1, X_S2]) 
            print(X_S1_S2.shape) 

            X_stim = []  
            X_ED = []
            X_MD = []
            X_LD = []
            
            for i, sample in enumerate(gv.samples):

                # average over trials 
                X_avg = np.mean(X_S1_S2[i], axis=0) 
                
                #average over epoch to sort
                X_stim_avg = np.mean(X_avg[:,gv.bins_stim], axis=1)
                X_ED_avg = np.mean(X_avg[:,gv.bins_ED], axis=1) 
                X_MD_avg = np.mean(X_avg[:,gv.bins_MD], axis=1) 
                X_LD_avg = np.mean(X_avg[:,gv.bins_LD], axis=1) 
                
                X_stim.append(X_stim_avg)
                X_ED.append(X_ED_avg)
                X_MD.append(X_MD_avg)
                X_LD.append(X_LD_avg)
                
            X_stim = np.asarray(X_stim) 
            X_ED = np.asarray(X_ED) 
            X_MD = np.asarray(X_MD) 
            X_LD = np.asarray(X_LD) 
            
            sel_stim = (X_stim[0]-X_stim[1])/(X_stim[0]+X_stim[1]+.000000001)
            sel_ED = (X_ED[0]-X_ED[1])/(X_ED[0]+X_ED[1]+.000000001)
            sel_MD = (X_MD[0]-X_MD[1])/(X_MD[0]+X_MD[1]+.000000001)
            sel_LD = (X_LD[0]-X_LD[1])/(X_LD[0]+X_LD[1]+.000000001)
            
            if 'ND' in gv.trials: 
                idx = np.argsort(sel_ED) 
            
            sel_stim = sel_stim[idx]
            sel_ED = sel_ED[idx]
            sel_LD = sel_LD[idx]
            
            figname = '%s_%s_%s_polar' % (gv.mouse, gv.session, gv.trial) 
            ax = plt.figure(figname).add_subplot() 
            # plt.polar( theta, gaussian_filter1d(sel_stim,1) , color='k') 
            plt.polar( theta, gaussian_filter1d(sel_ED, 30) , color='b') 
            plt.polar( theta , gaussian_filter1d(sel_MD, 30) , color='y') 
            plt.polar( theta, gaussian_filter1d(sel_LD, 30) , color='r') 
