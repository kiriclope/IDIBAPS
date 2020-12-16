from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis') 
from scipy.spatial import distance 

import data.constants as gv 
import data.utils as fct
import data.plotting as pl
import data.preprocessing as pp 
import data.angle as agl
import data.progressbar as pg

import data.fct_facilities as fac 
importlib.reload(fac) 
fac.SetPlotParams() 

from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical
from pls import optimise_pls_cv 
from spls import *

from bayesian_bootstrap import bootstrap as bayes 

from joblib import Parallel, delayed, parallel_backend
import multiprocessing

def t_test(x,y,alternative='both-sided'):
    _, double_p = stats.ttest_ind(x,y,equal_var = False) 
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval

def pls_transform(X,y,verbose):
    n_comp = optimise_pls_cv(X, y, 2*X.shape[0], verbose=verbose) 
    pls = PLSRegression(n_components=n_comp) 
    X, _ = pls.fit_transform(X,y) 
    return X, pls.coef_ 

def grid_search_cv_clf(loss, X, y, cv=10): 
    
    pipe = Pipeline([('scale', StandardScaler()), ('clf', loss)]) 
    
    param_grid = [{'clf': [loss], 'clf__C' : np.logspace(-4, 4, 1000)}] 
    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=False, n_jobs=gv.num_cores) 
    best_model = search.fit(X, y) 
    
    return best_model 

def bayesian_boot(clf, X, y, n_boot, n_jobs=1):
    model = bayes.BayesianBootstrapBagging(clf, n_boot, X.shape[1], n_jobs=n_jobs) 
    model.fit(X, y) 
    coefs = [ estimator.coef_.flatten() for estimator in model.base_models_ ] 
    coefs = np.array(coefs) 
    return coefs 

def bagging_boot(clf, X, y, n_estimators):
    model = BaggingRegressor(base_estimator=clf, n_estimators=n_estimators, n_jobs=gv.num_cores, bootstrap_features=False) 
    model.fit(X, y) 
    coefs = [ estimator.coef_.flatten() for estimator in model.estimators_ ] 
    coefs = np.array(coefs) 
    return coefs 

def bootstrap_clf_par(X, y, clf, dum, cv): 
    # note: joblib messes up my global variables ... 

    if dum==1: 
        print('no boot') 
        idx_trials = np.arange(0, X.shape[0]) 
    else:
        #standard bootstrap
        idx_trials = np.random.randint(0, X.shape[0], X.shape[0]) 
        # block bootstrap 
        # idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)), 
        #                           np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 
        #trialxepochs 
        # X is (trials x times , neurons) and we want to sample over trials 
        # so we pick n_trials with replacement ie if trial k, then idx_k = [k*gv.bins_ED, (k+1)*gv.bins_ED] 
        # idx = np.random.randint(0, 40, 40) 
        # idx_trials = np.array( [np.arange(k*7, (k+1)*7) for k in idx ] ).flatten() 
        
        # idx = np.random.randint(0, 32, 32) 
        # idx_trials = np.array( [np.arange(k*6, (k+1)*6) for k in idx ] ).flatten() 
        
    X_sample = X[idx_trials] 
    y_sample = y[idx_trials] 
    
    # hierarchical bootstrap 
    # for trial in idx_trials: 
    #     idx_neurons = np.random.randint(0, X.shape[1], X.shape[1]) 
    #     X_sample[trial] = X[trial, idx_neurons] 
    
    # scaler = StandardScaler().fit(X) 
    # X_sample = scaler.transform(X_sample) 
    
    if cv==0:
        X_sample = StandardScaler().fit_transform(X_sample)

        _, coefs = pls_transform(X_sample, y_sample, verbose=False)        
        # _, coefs = spls_cv(X_sample, y_sample, n_comp_max = X_sample.shape[0], n_jobs=5) 

        coefs_samples = coefs.flatten() 

        # clf.fit(X_sample, y_sample) 
        # coefs_samples = clf.coef_.flatten() 
    else:
        best_model = grid_search_cv_clf(clf, X_sample, y_sample, cv=cv) 
        coefs_samples = best_model.best_estimator_['clf'].coef_.flatten() 
        # C = best_model.best_estimator_['clf__C'] 
        
    return coefs_samples 

def bootCoefs(X_proj, C=1e0, penalty='l2', solver='liblinear', loss='squared_hinge', cv=10, l1_ratio=None, shrinkage='auto'): 

    gv.n_boot = int(1e3) 
    gv.num_cores = 56 # int(1*multiprocessing.cpu_count()/2) 

    # clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8),
    #                          fit_intercept=bool(not gv.standardize), l1_ratio=l1_ratio) 

    gv.num_cores = int(multiprocessing.cpu_count()/5) 
    clf = LogisticRegressionCV(Cs=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), 
                               fit_intercept=bool(not gv.standardize), n_jobs=5) 
    
    # clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', dual=False, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(not gv.standardize) ) 
    # clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage=shrinkage) 
   
    # clf = Lasso(alpha=1.0/C, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=int(1e8), tol=1e-6, warm_start=False, positive=False, random_state=None, selection='cyclic') 

    # clf = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=int(1e8), tol=1e-6, copy_X=True, cv=None, verbose=False, n_jobs=10, positive=False, random_state=None, selection='random') 

    # gv.num_cores = 4 
    # clf = PLSRegression()
    
    gv.clf_name = clf.__class__.__name__ 
    if 'CV' in 'gv.clf_name': 
        cv=0
        
    print('bootstrap samples', gv.n_boot, 'clf', gv.clf_name) 
    
    gv.IF_PCA = 0 
    if X_proj.shape[3]!=gv.n_neurons: 
        X_proj = X_proj[:,:,:,0:gv.n_components,:] 
        gv.IF_PCA = 1 

    if gv.EDvsLD: 
        gv.epochs = ['ED', 'MD', 'LD'] 
        print('angle btw ED and other epochs') 
    else: 
        gv.epochs = ['Stim', 'ED', 'MD', 'LD'] 
        print('angle btw STIM and other epochs') 
        
    coefs = np.empty((len(gv.trials), len(gv.epochs), gv.n_boot, X_proj.shape[3])) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        X_S1 = X_proj[n_trial,0] 
        X_S2 = X_proj[n_trial,1] 

        # X_S1, X_S2, idx = pp.selectiveNeurons(X_S1, X_S2, .25) 
        # print(X_S1.shape) 
        
        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        X_S1_S2 = pp.avg_epochs(X_S1_S2) 
        
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
        
        for n_epochs in range(X_S1_S2.shape[2]): 
            X = X_S1_S2[:,:,n_epochs] 
            # X = StandardScaler().fit_transform(X) 
            # X = pls_transform(X, y) 
            # X = spls_cv(X, y, n_comp_max = X.shape[0]) 
            # # X = spls_red(X, y, n_comp=3, eta=.9) 

            if 'PLS' in gv.clf_name:
                with pg.tqdm_joblib(pg.tqdm(desc= gv.trial + ' ' + gv.epochs[n_epochs] , total=gv.n_boot)) as progress_bar:             
                    boot_coefs = Parallel(n_jobs=gv.num_cores, verbose=False)(delayed(bootstrap_clf_par)(X, y, clf, gv.n_boot, cv) 
                                                                              for _ in range(gv.n_boot))
            else: 
                X = StandardScaler().fit_transform(X) 
                if gv.BAYES_BOOTSTRAP: 
                    boot_coefs = bayesian_boot(clf, X, y, gv.n_boot, n_jobs=gv.num_cores) 
                else: 
                    boot_coefs = bagging_boot(clf, X, y, gv.n_boot) 
            
            coefs[n_trial, n_epochs,:, 0:X.shape[1]] = np.array(boot_coefs) 
            
    return coefs 

def cosVsEpochs(coefs):

    cos_boot = np.empty( (len(gv.trials), gv.n_boot, len(gv.epochs) ) ) 
    
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

    if cv!=0: 
        gv.figdir = gv.figdir + '/gridsearchCV_%d' % cv 

    gv.figdir = gv.figdir +'/'+ gv.clf_name + clf_param 
    
    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)

def corrVsTime(coefs): 
    
    corr = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    lower = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    upper = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    
    corr_boot = np.empty( (len(gv.trials), gv.n_boot, coefs.shape[1], coefs.shape[1]) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        
        for boot in range(gv.n_boot): 
            corr_boot[n_trial, boot] = np.corrcoef(coefs[n_trial,:,boot,:]) # bins x coefficients 
            
        corr[n_trial] = np.mean(corr_boot[n_trial], axis=0) 
        lower[n_trial] = corr[n_trial] - np.percentile(corr_boot[n_trial], 25, axis=0) 
        upper[n_trial] = np.percentile(corr_boot[n_trial], 75, axis=0) - corr[n_trial]
        
    return corr, lower, upper 

def EDvsLD(X_proj, C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=10, l1_ratio=None, shrinkage='auto'):

    coefs = bootCoefs(X_proj, C, penalty, solver, loss, cv, l1_ratio, shrinkage) 
    mean, lower, upper, cos_boot = cosVsEpochs(coefs) 
    p_values = get_p_values(cos_boot) 
    print('p_values', p_values) 
    
    create_fig_dir(C=C, penalty=penalty, solver=solver, cv=cv, loss=loss, l1_ratio=l1_ratio, shrinkage=shrinkage) 
    
    pl.barCosAlp(mean, lower, upper) 
    add_pvalue(p_values) 
    plt.ylim([-0.1, 1.1]) 
    
    figtitle = '%s_%s_cos_alpha' % (gv.mouse, gv.session) 
    pl.save_fig(figtitle) 

    return coefs 

def loop_mice_sessions(C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=10, l1_ratio=None, shrinkage='auto'):

    gv.T_WINDOW = 0.5
    gv.IF_SAVE = 1 
    gv.EDvsLD = 1 
    for gv.mouse in [gv.mice[1]] :
        fct.get_sessions_mouse() 
        fct.get_stimuli_times() 
        fct.get_delays_times() 
        
        for gv.session in [gv.sessions[-1]] : 
            X_trials = fct.get_X_y_mouse_session() 
            EDvsLD(X_trials, C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage) 
            # plt.close('all') 
