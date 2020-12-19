from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.utils as fct
import data.plotting as pl
import data.preprocessing as pp 
import data.angle as agl
import data.progressbar as pg
import data.fct_facilities as fac
fac.SetPlotParams() 

import bootstrap as bs 
from statistics import t_test 

def is_pca(X_trials): 
    gv.IF_PCA = 0 
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
        gv.IF_PCA = 1 
    return X_trials 

def get_epochs(): 
    if gv.EDvsLD: 
        gv.epochs = ['ED', 'MD', 'LD'] 
        print('angle btw ED and other epochs') 
    else: 
        gv.epochs = ['Stim', 'ED', 'MD', 'LD'] 
        print('angle btw STIM and other epochs') 

def SVD_trick(X):
    # SVD trick from The elements of statistical learning Data Mining, ..., Friedman et al., 2009 
    U, D, Vh = np.linalg.svd(X, full_matrices=False) 
    X = (U*D[..., None, :]) 
    print('X', X.shape, 'Vh', Vh.shape) 
    return X, Vh 

def get_clf(C=1, penalty='l2', solver='liblinear', loss='squared_hinge', cv=None, l1_ratio=0, shrinkage='auto'): 
    
    if 'LogisticRegressionCV' in gv.clf_name:
        clf = LogisticRegressionCV(Cs=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), 
                                   fit_intercept=True, n_jobs=None , intercept_scaling=1e2 ) 
        
    elif 'LogisticRegression' in gv.clf_name:
        clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8),
                                 fit_intercept=True, l1_ratio=l1_ratio, intercept_scaling=1e2 ) 
        
    elif 'LDA' in gv.clf_name:        
        clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage=shrinkage)
        
    elif 'PLS' in gv.clf_name:
        gv.num_cores = int(multiprocessing.cpu_count()/6) 
        clf = PLSRegression(scale=False)

    elif 'ReLASSO' in gv.clf_name:
        clf = relassoCV = RelaxedLassoLarsCV( fit_intercept=True, verbose=False, max_iter=500,
                                              normalize=True, precompute='auto', cv=None, max_n_alphas=1000,
                                              n_jobs=None, eps=np.finfo(np.float).eps, copy_X=True) 
        
    return clf

def bootstrap_coefs_epochs(X_trials, bootstrap_method='standard', C=1e0, penalty='l2', solver='liblinear', loss='squared_hinge', cv=10, l1_ratio=None, shrinkage='auto'): 

    gv.n_boots = int(1e3) 
    gv.num_cores =  int(multiprocessing.cpu_count()) - 1 
    
    clf = get_clf(C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage) 
    
    print('bootstrap samples', gv.n_boots, 'clf', gv.clf_name) 
    boots_model = bs.bootstrap(clf, bootstrap_method=bootstrap_method, n_boots=gv.n_boots, n_jobs=gv.num_cores) 

    X_trials = is_pca(X_trials) 
    get_epochs() 
    coefs = np.empty((len(gv.trials), len(gv.epochs), gv.n_boots, X_trials.shape[3])) 
    
    for n_trial, gv.trial in enumerate(gv.trials):         
        X_S1_S2 = np.vstack( ( X_trials[n_trial,0], X_trials[n_trial,1] ) ) 
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 

        X_S1_S2 = pp.avg_epochs(X_S1_S2, y) 
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
        
        for n_epochs in range(X_S1_S2.shape[2]): 
            X = X_S1_S2[:,:,n_epochs]
            Vh = None 

            if gv.TIBSHIRANI_TRICK and (penalty=='l2' or 'LDA' in gv.clf_name or 'PLS' in gv.clf_name): 
                X, Vh = SVD_trick(X) 
                
            if gv.BAYES_BOOTSTRAP: 
                boots_coefs = boots_model.bayesian_bootstrap(X, y, Vh) 
            elif gv.BAGGING_BOOTSTRAP: 
                boots_coefs = boots_model.bagging_bootstrap(X, y, Vh) 
            else: 
                boots_coefs = boots_model.my_bootstrap(X, y, Vh) 
                
            coefs[n_trial, n_epochs,:, 0:boots_coefs.shape[1]] = boots_coefs 
            
    return coefs 

def get_cos_epochs(coefs):

    cos_boot = np.empty( (len(gv.trials), gv.n_boots, len(gv.epochs) ) ) 
    
    mean = np.empty((len(gv.trials), len(gv.epochs)))
    upper = np.empty( (len(gv.trials), len(gv.epochs)) )
    lower = np.empty((len(gv.trials), len(gv.epochs)))
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        for boot in range(coefs.shape[2]): 
            cos_alp = agl.get_cos(coefs[n_trial,:,boot,:], coefs[n_trial,0,boot,:]) # bins x neurons 
            cos_boot[n_trial, boot] = np.array(cos_alp) 
        
        mean[n_trial] = np.mean(cos_boot[n_trial], axis=0) 
        lower[n_trial] = mean[n_trial] - np.percentile(cos_boot[n_trial], 25, axis=0) 
        upper[n_trial] = np.percentile(cos_boot[n_trial], 75, axis=0) - mean[n_trial]
        
        print('trial', gv.trial, 'cos', mean[n_trial], 'lower', lower[n_trial], 'upper', upper[n_trial]) 

    return mean, lower, upper, cos_boot 

def get_p_values(cos_boot):

    p_values = np.empty( ( cos_boot.shape[0]-1, cos_boot.shape[2]-1) ) 
    for n_trial in range(1, cos_boot.shape[0]): # trials 
        for n_epoch in range(1, cos_boot.shape[2]): # epochs 
            sample_1  = cos_boot[0,:,n_epoch] # boots 
            sample_2  = cos_boot[n_trial,:,n_epoch]
            p_values[n_trial-1, n_epoch-1] = t_test(sample_2, sample_1, alternative='both-sided')
            # note sample_2 then sample_1 for H0: S2>=S1, Ha S1>S2
    return p_values

def add_pvalue(p_values): 
    cols = 0.25*np.arange(len(gv.trials)) 
    high = [1.0, 0.9] 

    for n_cols in range(1, len(cols)):        
        for n_epoch in range(p_values.shape[1]): 
            
            plt.plot( [n_epoch + cols[0], n_epoch + cols[n_cols]] , [high[n_cols-1], high[n_cols-1]] , lw=.8, c='k') 
            
            if p_values[n_cols-1,n_epoch]<=.001: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "***", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]<=.01: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "**", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]<=.05: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "*", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]>.05: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1], "ns", ha='center', va='bottom', color='k', fontsize=6) 

def create_fig_dir(C=1, penalty='l1', solver='liblinear', cv=0, loss='lsqr', l1_ratio=0, shrinkage='auto'): 
    
    pl.figDir() 
    clf_param = '' 
    
    if 'LogisticRegression' in gv.clf_name:
        if 'liblinear' in solver:
            clf_param = '/C_%.3f_penalty_%s_solver_%s/' % (C, penalty, solver)
        if 'sag' in solver:
            clf_param = '/C_%.3f_penalty_%s_solver_%s_l1_ratio_%.2f/' % (C, penalty, solver, l1_ratio)
            
    elif gv.clf_name in 'LinearSVC':
        clf_param = '/C_%.3f_penalty_%s_loss_%s/' % (C, penalty, loss)
    elif gv.clf_name in 'LinearDiscriminantAnalysis':
        clf_param = '/shrinkage_%s_solver_lsqr/' % shrinkage
        
    # if cv!=0: 
    #     gv.figdir = gv.figdir + '/gridsearchCV_%d' % cv 
    
    gv.figdir = gv.figdir +'/'+ gv.clf_name + clf_param 
    
    if not os.path.isdir(gv.figdir): 
        os.makedirs(gv.figdir) 

def get_corr_epochs(coefs): 
    
    corr = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    lower = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    upper = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    
    corr_boot = np.empty( (len(gv.trials), gv.n_boots, coefs.shape[1], coefs.shape[1]) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        
        for boot in range(gv.n_boots): 
            corr_boot[n_trial, boot] = np.corrcoef(coefs[n_trial,:,boot,:]) # bins x coefficients 
            
        corr[n_trial] = np.mean(corr_boot[n_trial], axis=0) 
        lower[n_trial] = corr[n_trial] - np.percentile(corr_boot[n_trial], 25, axis=0) 
        upper[n_trial] = np.percentile(corr_boot[n_trial], 75, axis=0) - corr[n_trial] 
        
    return corr, lower, upper 

def plot_cos_epochs(X_trials, bootstrap_method='standard', C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=10, l1_ratio=None, shrinkage='auto'): 
    
    coefs = bootstrap_coefs_epochs(X_trials, bootstrap_method, C, penalty, solver, loss, cv, l1_ratio, shrinkage) 
    mean, lower, upper, cos_boot = get_cos_epochs(coefs) 
    p_values = get_p_values(cos_boot) 
    print('p_values', p_values) 
    
    create_fig_dir(C=C, penalty=penalty, solver=solver, cv=cv, loss=loss, l1_ratio=l1_ratio, shrinkage=shrinkage) 
    
    pl.bar_trials_epochs(mean, lower, upper) 
    add_pvalue(p_values) 
    plt.ylim([-0.1, 1.1]) 
    
    figtitle = '%s_%s_bars_cos_alp' % (gv.mouse, gv.session) 
    pl.save_fig(figtitle) 

def plot_loop_mice_sessions(C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=10, l1_ratio=None, shrinkage='auto'): 

    gv.T_WINDOW = 0.0 
    gv.IF_SAVE = 1 
    gv.EDvsLD = 1 
    gv.SAVGOL = 0 
    
    gv.clf_name = 'LogisticRegressionCV'  
    # gv.clf_name = 'ReLASSO'
    
    gv.BAYES_BOOTSTRAP = 0 
    gv.BAGGING_BOOTSTRAP = 1 
    gv.FEATURE_SELECTION = 0 
    
    gv.TIBSHIRANI_TRICK = 0 
    
    # for gv.mouse in gv.mice : 
    #     fct.get_sessions_mouse() 
    #     fct.get_stimuli_times() 
    #     fct.get_delays_times() 
        
    #     for gv.session in gv.sessions : 
    #         X_trials = fct.get_X_y_mouse_session() 
    #         print(X_trials.shape)
            
    #         matplotlib.use('Agg') # so that fig saves when in the in the background 
    #         plot_cos_epochs(X_trials, C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage) 
    #         plt.close('all') 

    for gv.mouse in [gv.mice[1]] : 
        fct.get_sessions_mouse() 
        fct.get_stimuli_times() 
        fct.get_delays_times() 
        
        for gv.session in [gv.sessions[-1]] : 
            X_trials = fct.get_X_y_mouse_session() 
            print(X_trials.shape)
            matplotlib.use('GTK3cairo')
            plot_cos_epochs(X_trials, C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage) 

