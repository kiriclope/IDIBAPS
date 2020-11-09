from std_lib import *
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')
from sklearn_lib import *

import data.constants as gv
importlib.reload(gv) ;

import data.utils as data
importlib.reload(data) ;

import utils as fct
importlib.reload(fct) ;

import data.fct_facilities as fac
importlib.reload(fac) ;
fac.SetPlotParams()

for gv.mouse in [gv.mice[-1]]:
    print('mouse:', gv.mouse)

    data.get_sessions_mouse()
    data.get_stimuli_times()
    data.get_delays_times() 

    print('t_early_delay', gv.t_early_delay, 't_DRT_delay', gv.t_DRT_delay, 't_late_delay', gv.t_late_delay) 
    
    for gv.session in [gv.sessions[-1]] : 
    
        X_data, y_labels = data.get_fluo_data() 
        print('session:', gv.session, 'X', X_data.shape,'y', y_labels.shape) 

        data.get_bins(t_start=1.5) 

        cos_alp_trials = []
        alpha_trials = []

        for gv.trial in gv.trials :
            X_S1_trials, X_S2_trials = data.get_S1_S2_trials(X_data, y_labels) 
            X_trials, y_trials = data.get_X_y_bins(X_S1_trials, X_S2_trials) 

            coefs = fct.cross_val_clf(X_trials, y_trials) 
            alpha, cos_alp = fct.get_cos(coefs) 
            
            alpha_trials.append(alpha[0]) 
            cos_alp_trials.append(cos_alp[0])
            
        print('cos_alp', cos_alp_trials) 
        fct.plot_cos_bar(cos_alp_trials, [0,0,0]) 
        gv.dum=1 

        mat_alp = [] 
        mat_cos = []
        
        X_S1_all, X_S2_all = data.get_S1_S2_all(X_data, y_labels) 
        X_all, y_all = data.get_X_y_bins(X_S1_all, X_S2_all) 

        for i in range(100): 
            coefs_shuffle = fct.cross_val_clf(X_all, y_all, shuffle=1) 
            alpha_shuffle, cos_alp_shuffle = fct.get_cos(coefs_shuffle) 

            mat_alp.append(alpha_shuffle[0]) 
            mat_cos.append(cos_alp_shuffle[0]) 

        mat_alp = np.asarray(mat_alp) 
        mean_alp = np.mean(mat_alp, axis=0) 
        std_alp = np.std(mat_alp, axis=0) 
        print('cos(<alp>)', np.cos(mean_alp), 'std_alp', std_alp) 

        mat_cos = np.asarray(mat_cos) 
        # print(mat_cos.shape) 
    
        mean_cos = [ np.mean(mat_cos, axis=0) ]
        std_cos = [ np.std(mat_cos, axis=0) ]
        print('<cos(alp)>', mean_cos, 'std_cos', std_cos) 

        fct.plot_cos_bar([mean_cos[0], mean_cos[0], mean_cos[0]], [std_cos[0], std_cos[0], std_cos[0]]) 

        gv.dum=-1 

        figname = '%s_%s_cos_alpha' % (gv.mouse, gv.session)
        plt.figure(figname)
        plt.savefig(figname + '.svg', format='svg')
    
        z_score_alp = fct.get_z_score_cos_alp(cos_alp_trials, mean_cos, std_cos) 

        figname = '%s_%s_z_score' % (gv.mouse, gv.session) 
        plt.figure(figname) 
        plt.savefig(figname+ '.svg', format='svg') 

        p_value_alp = fct.get_p_value_alp(z_score_alp) 

        figname = '%s_%s_p_value' % (gv.mouse, gv.session) 
        plt.figure(figname)
        plt.savefig(figname + '.svg', format='svg')
