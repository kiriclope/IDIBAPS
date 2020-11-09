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

import plotting as plot
reload(plot)

gv.data_type = 'fluo'
gv.epochs = ['Stim', 'ED', 'MD', 'LD']
gv.laser_on = 0 
gv.n_boot = 1
IF_SHUFFLE = 0 

C=1
penalty = 'l1' 
solver = 'liblinear' 

clf = LogisticRegression(C=C, solver='liblinear', penalty=penalty, tol=1e-6, max_iter=int(1e6)) 
# clf = svm.LinearSVC(C=1, penalty='l1', loss='squared_hinge', dual=False) 

for gv.mouse in [gv.mice[2]]:
    
    data.get_sessions_mouse()
    data.get_stimuli_times()
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[-1]] :
        
        X_data, y_labels = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X_data.shape,'y', y_labels.shape)
        data.get_bins(t_start=0)

        F0 = np.mean(X_data[:,:,gv.bins_baseline],axis=2)
        F0 = F0[:,:, np.newaxis] 
        X_data = (X_data-F0) / (F0 + 0.0000000001) 

        X_trials = []
        for gv.trial in gv.trials : 
            X_S1, X_S2 = data.get_S1_S2_trials(X_data, y_labels) 
            data.get_trial_types(X_S1)
            print('X_S1', X_S1.shape) 

            X_S1_S2 = np.vstack((X_S1, X_S2))
            print('X_S1_S2', X_S1_S2.shape)
            X_trials.append(X_S1_S2)

        X_trials = np.vstack(X_trials)
        X_epochs = data.get_X_epochs(X_trials)
        print('X_epochs', X_trials.shape) 
        
        X_epochs = np.reshape(X_epochs, (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), gv.n_neurons, len(gv.epochs))) 
        print(X_trials.shape) 

        y_early = np.array([np.zeros(int(3*gv.n_trials/2)), np.ones(int(3*gv.n_trials/2))]).flatten() 
        y_trials = np.array([np.zeros(int(gv.n_trials/2)), np.ones(int(gv.n_trials/2))]).flatten() 

        coefs_boot = [] 
        for i, gv.epoch in enumerate(gv.epochs):
            
            if 'Stim' in gv.epoch or 'ED' in gv.epoch:
                X_S1 = []
                X_S2 = []                    
                for j in range(len(gv.trials)):
                    X_S1.append(X_epochs[j,0,:,:,i])
                    X_S2.append(X_epochs[j,1,:,:,i])
                X_S1 = np.vstack(X_S1) 
                X_S2 = np.vstack(X_S2) 
                X_S1_S2 = np.vstack([X_S1, X_S2])
                coefs_boot.append( fct.bootstrap_clf_epoch(X_S1_S2, y_early, clf, shuffle=0) )
                
            else:
                for j, gv.trial in enumerate(gv.trials): 
                    X_S1 = X_epochs[j,0,:,:,i] 
                    X_S2 = X_epochs[j,1,:,:,i] 
                    X_S1_S2 = np.vstack([X_S1, X_S2])                    
                    coefs_boot.append(fct.bootstrap_clf_epoch(X_S1_S2, y_trials, clf, shuffle=0))
                        
        coefs_boot = np.array(coefs_boot) 
        print('coefs', coefs_boot.shape)        
        break
    
        alpha_samples = [] 
        cos_alp_samples = [] 
        for sample in range(coefs_boot.shape[0]):
            alpha, cos_alp = fct.get_cos(coefs_boot[sample]) 

            alpha_samples.append(alpha)
            cos_alp_samples.append(cos_alp)

        alpha_samples = np.asarray(alpha_samples)
        cos_alp_samples = np.asarray(cos_alp_samples)

        print('samples', cos_alp_samples.shape)                
        mean_cos_samples = np.mean(cos_alp_samples,axis=0) 
        
        q1 = mean_cos_samples - np.percentile(cos_alp_samples, 25, axis=0) 
        q3 = np.percentile(cos_alp_samples, 75, axis=0) - mean_cos_samples
        y_err = np.asarray([q1[1:],q3[1:]]) 
                
        print('trial', gv.trial, 'cos_alp', mean_cos_samples, 'q1', q1, 'q3', q3) 
        cos_alp = mean_cos_samples

        plot.plot_cosine_bars(cos_alp, [], y_err, 0) 
