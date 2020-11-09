#!/usr/bin/env python3 
""" Compute angle of coding direction between two different epochs""" 

from std_lib import * 
from sklearn_lib import * 
from mne_lib import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv
importlib.reload(gv) ;

import data.utils as data 
importlib.reload(data) ; 

import utils as fct 
importlib.reload(fct) ; 

import data.fct_facilities as fac
importlib.reload(fac) ;
fac.SetPlotParams()

clf = LogisticRegression(solver='liblinear', penalty='l2') 
# clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto') 

# figs_dir = os.path.join(figs_dir, clf.__class__.__name__+'/') 
# clf_name = clf.__class__.__name__

# if gv.laser_on: 
#     figs_dir = os.path.join(figs_dir, 'laser_on/') 
#     data_dir = os.path.join(data_dir, 'laser_on/') 

IF_SHUFFLE=0
IF_GRID=0

gv.epochs=['ED', 'MD', 'LD']
gv.laser_on=0

for gv.mouse in gv.mice:
    data.get_sessions_mouse()
    data.get_stimuli_times()
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[-1]] : 
        
        X_data, y_labels = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X_data.shape,'y', y_labels.shape)
        data.get_bins(t_start=0)
        
        cos_alp_trials = [] 
        alpha_trials = []
        
        for gv.trial in gv.trials : 
            X_S1_trials, X_S2_trials = data.get_S1_S2_trials(X_data, y_labels) 
            print(X_S1_trials.shape) 
            
            X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials) 
            print('X', X_trials.shape,'y', y_trials.shape)
            
            if IF_GRID :
                coefs, clf = fct.grid_search_cv_clf(X_trials, y_trials, shuffle=0, cv=10)
            else :
                X_detrend = []
                for n_trial in range(0,X_trials.shape[0]): 
                    fit_values = fct.detrend_data(X_trials[n_trial], poly_fit=1, degree=10) 
                    X_detrend.append(fit_values) 
                    
                X_detrend = np.asarray(X_detrend) 
                coefs = fct.coefs_clf(X_trials-X_detrend, y_trials, clf=clf) 
            print('coefs', coefs.shape)
            
            alpha, cos_alp = fct.get_cos(coefs) 
            print('trial', gv.trial, 'cos_alp', cos_alp) 
            
            alpha_trials.append(alpha) 
            cos_alp_trials.append(cos_alp)
            
            if IF_SHUFFLE: 
                mat_cos = [] 
                
                for i in range(1000): 
                    coefs_shuffle = fct.coefs_clf(X_trials, y_trials, clf=clf, shuffle=1) 
                    alpha_shuffle, cos_alp_shuffle = fct.get_cos(coefs_shuffle) 
                    
                    mat_cos.append(cos_alp_shuffle) 
                    
                mat_cos = np.asarray(mat_cos) 
                
                mean_cos = np.mean(mat_cos, axis=0) 
                std_cos = np.std(mat_cos, axis=0)
                q1 = np.percentile(mat_cos, 25, axis=0)
                q3 = np.percentile(mat_cos, 75, axis=0)
                
                print('<cos(alp)>', mean_cos, 'std_cos', std_cos, 'q1', q1)
                
            if gv.laser_on:
                figtitle = '%s_%s_cos_alpha_trials_laser_on' % (gv.mouse, gv.session)
            else:
                figtitle = '%s_%s_cos_alpha_trials' % (gv.mouse, gv.session)
                
            ax = plt.figure(figtitle).add_subplot()
            # ax = plt.figure(figtitle).subplot() 
            xticks = np.arange(0,len(gv.epochs)-1) 
            
            if IF_SHUFFLE: 
                width = 1/10. 
            else: 
                width = 2/10. 
                
            if('ND' in gv.trial): 
                ax.bar(xticks - 4/10, cos_alp[1:], width, label=gv.trial, color='r') ; 
                if IF_SHUFFLE: 
                    ax.bar(xticks - 3/10, mean_cos[1:], width, yerr=[q1[1:], q3[1:]],color='r', alpha=0.5) ; 
                    
            if('D1' in gv.trial): 
                ax.bar(xticks - 1/10, cos_alp[1:], width, label=gv.trial, color='b') ; 
                if IF_SHUFFLE:
                    ax.bar(xticks, mean_cos[1:], width, yerr=[q1[1:],q3[1:]], color='b', alpha=0.5) ;                 
            if('D2' in gv.trial):
                ax.bar(xticks + 2/10 , cos_alp[1:], width, label=gv.trial, color='g') ; 
                if IF_SHUFFLE:
                    ax.bar(xticks + 3/10, mean_cos[1:], width, yerr=[q1[1:],q3[1:]], color='g', alpha=0.5) ; 
                
            plt.ylabel('cos($\\alpha$)') 
            plt.xlabel('epochs') 
            labels = gv.epochs ; 
            ax.set_xticks(xticks) ; 
            ax.set_xticklabels(labels[1:]) ; 
            
            ax.legend()
            plt.ylim([0,1]) ;
>
