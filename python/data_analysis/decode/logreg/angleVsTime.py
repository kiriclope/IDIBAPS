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
gv.epochs = [ 'ED', 'MD', 'LD']
gv.laser_on = 0 
gv.n_boot = 100 
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
        
        for gv.trial in gv.trials : 
            X_S1_trials, X_S2_trials = data.get_S1_S2_trials(X_data, y_labels) 
            print(X_S1_trials.shape) 
            
            X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials) 
            print('X', X_trials.shape,'y', y_trials.shape) 

            coefs_boot = fct.bootstrap_clf(X_trials, y_trials, clf, shuffle=0) 
            print('coefs', coefs_boot.shape)

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
