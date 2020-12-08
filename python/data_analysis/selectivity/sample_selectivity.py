from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

import data.fct_facilities as fac 
importlib.reload(fac) ; 
fac.SetPlotParams() 

import data.preprocessing as pp
import data.plotting as pl 

from matplotlib.ticker import PercentFormatter 

pal = ['r','b','y'] 
gv.data_type = 'fluo' 

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times()
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_bins(t_start=0) 
        
        trial_averages = [] 
        for i, gv.trial in enumerate(gv.trials): 
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1) 

            X_S1 = pp.dFF0(X_S1) 
            X_S2 = pp.dFF0(X_S2) 
            
            for j in range(X_S1.shape[0]):
                # X_S1[j] = pp.z_score(X_S1[j]) 
                # X_S2[j] = pp.z_score(X_S2[j]) 

                X_S1[j] = pp.normalize(X_S1[j]) 
                X_S2[j] = pp.normalize(X_S2[j]) 
            
            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
            
            X_S1 = np.mean(X_S1[:,:,gv.bins_ED], axis=2) 
            X_S2 = np.mean(X_S2[:,:,gv.bins_ED], axis=2) 
            
            X_S1 = np.mean(X_S1, axis=0) 
            X_S2 = np.mean(X_S2, axis=0) 
            
            sel_idx = (X_S1 - X_S2)/(X_S1 + X_S2 + gv.eps) 
            sel_idx = sel_idx.flatten() 
            print(sel_idx.shape) 
            
            figname = '%s_%s_selectivity_idx' % (gv.mouse, gv.session) 
            ax = plt.figure(figname).add_subplot() 
            plt.hist(sel_idx, bins=1000, alpha=1, color=pal[i], histtype='step', label=gv.trial,  weights=np.ones(len(sel_idx)) / len(sel_idx)) 
            # ax.yaxis.set_major_formatter(PercentFormatter(1)) 
            
        plt.legend()
        plt.xlim([-1,1])
        plt.ylim([0,1]) 
        plt.xlabel('selectivity index') 
        plt.ylabel('fraction')
        # figdir = pl.figDir()
        # pl.save_fig(figname, figdir)


