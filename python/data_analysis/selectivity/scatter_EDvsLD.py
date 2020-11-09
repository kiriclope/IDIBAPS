from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

import data.fct_facilities as fac
importlib.reload(fac) ; 
fac.SetPlotParams() 

import preprocessing as pre
from plotting import *

from matplotlib.ticker import PercentFormatter

pal = ['r','b','y']
gv.data_type = 'rates'

for gv.mouse in [gv.mice[2]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times()
    
    for gv.session in [gv.sessions[0]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_frame_rate() 
        data.get_bins(t_start=1) 
        
        gv.duration = X.shape[2]/gv.frame_rate 
        gv.time = np.linspace(0, gv.duration, X.shape[2]) 
        
        trial_averages = []
        for i, gv.trial in enumerate(gv.trials): 
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1)

            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 

            X_S1_ED = np.mean(X_S1[:,:,gv.bins_ED], axis=2) 
            X_S2_ED = np.mean(X_S2[:,:,gv.bins_ED], axis=2) 

            X_S1_LD = np.mean(X_S1[:,:,gv.bins_LD], axis=2) 
            X_S2_LD = np.mean(X_S2[:,:,gv.bins_LD], axis=2) 

            X_S1_ED = np.mean(X_S1_ED, axis=0) 
            X_S2_ED = np.mean(X_S2_ED, axis=0) 

            X_S1_LD = np.mean(X_S1_LD, axis=0) 
            X_S2_LD = np.mean(X_S2_LD, axis=0) 

            # idx = np.where(X_S1_ED<.0001) 
            # X_S1_ED = np.delete(X_S1_ED, idx) 
            # X_S2_ED = np.delete(X_S2_ED, idx) 
            # X_S1_LD = np.delete(X_S1_LD, idx) 
            # X_S2_LD = np.delete(X_S2_LD, idx) 

            # idx = np.where(X_S2_ED<.0001) 
            # X_S1_ED = np.delete(X_S1_ED, idx) 
            # X_S2_ED = np.delete(X_S2_ED, idx) 
            # X_S1_LD = np.delete(X_S1_LD, idx) 
            # X_S2_LD = np.delete(X_S2_LD, idx) 
            
            sel_idx_ED = (X_S1_ED - X_S2_ED)/(X_S1_ED + X_S2_ED + .00000000001) 
            sel_idx_ED = sel_idx_ED.flatten() 

            # idx = np.where(abs(sel_idx_ED)<0.25) 
            # sel_idx_ED = sel_idx_ED[idx] 
            
            sel_idx_LD = (X_S1_LD - X_S2_LD)/(X_S1_LD + X_S2_LD  + .00000000001) 
            sel_idx_LD = sel_idx_LD.flatten() 
            # sel_idx_LD = sel_idx_LD[idx]
            
            # idx = np.where(abs(sel_idx_ED)<=0.5) 
            # ED = np.delete(sel_idx_ED, idx) 
            # sel_idx_LD = np.delete(sel_idx_LD, idx) 

            # idx = np.where(abs(sel_idx_LD)<=0.5) 
            # # sel_idx_ED = np.delete(sel_idx_ED, idx) 
            # LD = np.delete(sel_idx_LD, idx) 

            # print('ED', ED.shape, 'LD', LD.shape, 'LD/ED', LD.shape[0]/ED.shape[0]) 

            idx = np.where(abs(sel_idx_LD-sel_idx_ED)>0.25) 
            sel_idx_ED = np.delete(sel_idx_ED, idx) 
            sel_idx_LD = np.delete(sel_idx_LD, idx) 

            print('ED', sel_idx_ED.shape, 'LD', sel_idx_LD.shape) 
            figname = '%s_%s_%s_selectivity_idx_EDvsLD' % (gv.mouse, gv.session, gv.trial) 
            ax = plt.figure(figname).add_subplot()

            corr = np.round(np.corrcoef(sel_idx_ED, sel_idx_LD)[0,1], 2)
            print('corr', corr) 
            
            plt.scatter(sel_idx_ED, sel_idx_LD, color=pal[i]) 
            plt.xlabel('selectivity index ED') 
            plt.ylabel('selectivity index LD') 
            
            plt.xlim([-1,1]) 
            plt.ylim([-1,1]) 
            
            figdir = figDir() 
            save_fig(figname, figdir) 
