from .libs import * 

import data.constants as gv 
reload(gv)
import data.utils as fct 
reload(fct)
import data.plotting as pl 
reload(pl)
import data.preprocessing as pp 
reload(pp)
import data.angle as agl 
reload(agl)
import data.progressbar as pg 
reload(pg)
import data.fct_facilities as fac 
reload(fac)
fac.SetPlotParams() 

import warnings
warnings.filterwarnings("ignore")

import models.glms 
reload(models.glms) 
from models.glms import get_clf 

import dim_red.pca.pca_decomposition 
reload(dim_red.pca.pca_decomposition) 
from dim_red.pca.pca_decomposition import pca_methods 

import dim_red.spca
reload(dim_red.spca)
from dim_red.spca import supervisedPCA_CV 

import dim_red.pls
reload(dim_red.pls)
from dim_red.pls import plsCV

from . import bootstrap
reload(bootstrap)
from .bootstrap import bootstrap

import statistics
reload(statistics)
from .statistics import t_test 

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

def bootstrap_coefs_epochs(X_trials, bootstrap_method='standard', C=1e0, penalty='l2', solver='liblinear', loss='squared_hinge', cv=None, l1_ratio=None, shrinkage='auto', fit_intercept=True, intercept_scaling=1e2): 

    # if gv.pls_method is not None: 
    #     my_pls = plsCV(cv=gv.pls_cv, pls_method=gv.pls_method, max_comp=gv.pls_max_comp, n_jobs=gv.num_cores, verbose=True) 
    
    get_clf(C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio,
            shrinkage=shrinkage, fit_intercept=fit_intercept, intercept_scaling= intercept_scaling) 
    
    boots_model = bootstrap(gv.clf, bootstrap_method=bootstrap_method, n_boots=gv.n_boots, scaling=gv.scaling, n_jobs=gv.num_cores) 
    
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
            
            # if gv.pls_method is not None: 
            #     # print('pls decomposition') 
            #     X = my_pls.fit_transform(X, y) 
            #     print('X', X.shape)
            
            if gv.LASSOCV:
                gv.lassoCV.fit(X, y) 
                selected = np.argwhere(gv.lassoCV[-1].coef_==0)
                if len(selected)<X.shape[1]: 
                    X = np.delete(X, selected, axis=1) 
                
            print('X', X.shape, 'y', y.shape) 
            
            if gv.TIBSHIRANI_TRICK and (penalty=='l2' or 'LDA' in gv.clf_name or 'PLS' in gv.clf_name): 
                X, Vh = SVD_trick(X)
                
            boots_coefs = boots_model.get_coefs(X, y, Vh) 
            coefs[n_trial, n_epochs,:, 0:boots_coefs.shape[1]] = boots_coefs 
            
    return coefs 

def get_cos_epochs(coefs):

    cos_boot = np.empty( (len(gv.trials), gv.n_boots, len(gv.epochs) ) ) 
    
    mean_cos = np.empty((len(gv.trials), len(gv.epochs)))
    var_cos = np.empty((len(gv.trials), len(gv.epochs)))
    upper_cos = np.empty( (len(gv.trials), len(gv.epochs)) )
    lower_cos = np.empty((len(gv.trials), len(gv.epochs)))

    # # THIS IS WRONG !!!!!!!!!
    # for n_trial, gv.trial in enumerate(gv.trials): 
    #     for boot in range(coefs.shape[2]): 
    #         cos_alp = agl.get_cos(coefs[n_trial,:,boot,:], coefs[n_trial,0,boot,:]) # bins x neurons 
    #         cos_boot[n_trial, boot] = np.array(cos_alp) 
    
    #     mean_cos[n_trial] = np.mean(cos_boot[n_trial], axis=0) 
    #     lower_cos[n_trial] = mean_cos[n_trial] - np.percentile(cos_boot[n_trial], 25, axis=0) 
    #     upper_cos[n_trial] = np.percentile(cos_boot[n_trial], 75, axis=0) - mean_cos[n_trial] 
        
    #     print('trial', gv.trial, 'cos', mean_cos[n_trial], 'lower', lower_cos[n_trial], 'upper', upper_cos[n_trial])

    # using the delta method
    mean_coefs = np.mean(coefs, axis=2) 
    std_coefs = np.std(coefs, axis=2) 
    
    for n_epoch, gv.epoch in enumerate(gv.epochs):
        # mean_cos = cos(mean_coefs) 
        mean_cos[n_epoch] = agl.get_cos(mean_coefs[:,n_epoch,:], mean_coefs[0,n_epoch,:]) 
        # var_cos = nabla(coefsToCos)(coefs).T cov(coefs) nabla(coefsToCos)(coefs) 
        var_cos[n_epoch] = agl.get_cos(mean_coefs[:,n_epoch,:], mean_coefs[0,n_epoch,:]) 
        
        lower_cos[n_epoch] = mean_cos[n_epoch] - 0.675 * var_cos[n_epoch] 
        upper_cos[n_epoch] = 0.675 * var_cos[n_epoch] - mean_cos[n_epoch] 
        
        print('epoch', gv.epoch, 'cos', mean_cos[n_epoch], 'lower', lower_cos[n_epoch], 'upper', upper_cos[n_epoch]) 
    
    return mean_cos, lower_cos, upper_cos, cos_boot 

def get_cos_trials(coefs):

    cos_boot = np.empty( ( len(gv.epochs), gv.n_boots, len(gv.trials) ) ) 
    
    mean = np.empty( ( len(gv.epochs) , len(gv.trials) ) )
    upper = np.empty( ( len(gv.epochs),  len(gv.trials) ) )
    lower = np.empty( ( len(gv.epochs), len(gv.trials) ) ) 
    
    # for n_epoch, gv.epoch in enumerate(gv.epochs): 
    #     for boot in range(coefs.shape[2]): 
    #         cos_alp = agl.get_cos(coefs[:,n_epoch,boot,:], coefs[0,n_epoch,boot,:]) # trials x neurons 
    #         cos_boot[n_epoch, boot] = np.array(cos_alp) 
            
    #     mean[n_epoch] = np.mean(cos_boot[n_epoch], axis=0) 
    #     lower[n_epoch] = mean[n_epoch] - np.percentile(cos_boot[n_epoch], 25, axis=0) 
    #     upper[n_epoch] = np.percentile(cos_boot[n_epoch], 75, axis=0) - mean[n_epoch]
    
        # print('epoch', gv.epoch, 'cos', mean[n_epoch], 'lower', lower[n_epoch], 'upper', upper[n_epoch]) 

    mean_coefs = np.mean(coefs, axis=2) 
    lower_coefs = mean_coefs - np.percentile(coefs, 25, axis=2) 
    upper_coefs = np.percentile(coefs, 75, axis=2) - mean_coefs 
    
    for n_epoch, gv.epoch in enumerate(gv.epochs): 
        mean[n_epoch] = agl.get_cos(mean_coefs[:,n_epoch,:], mean_coefs[0,n_epoch,:]) 
        lower[n_epoch] = agl.get_cos(lower_coefs[:,n_epoch,:], lower_coefs[0,n_epoch,:]) 
        upper[n_epoch] = agl.get_cos(upper_coefs[:,n_epoch,:], upper_coefs[0,n_epoch,:]) 
    
        print('epoch', gv.epoch, 'cos', mean[n_epoch], 'lower', lower[n_epoch], 'upper', upper[n_epoch]) 
        
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

def create_fig_dir(C=1, penalty='l1', solver='liblinear', cv=0, loss='lsqr', l1_ratio=0, shrinkage='auto', fit_intercept=True, intercept_scaling=1e2): 
    
    pl.figDir() 
    clf_param = ''

    if gv.cos_trials:
        gv.figdir = gv.figdir + '/cos_trials' 
    else:
        gv.figdir = gv.figdir + '/cos_epochs' 
        
    if 'LogisticRegressionCV' in gv.clf_name: 
        if 'liblinear' in solver:
            if cv is not None:
                clf_param = '/C_%.3f_penalty_%s_solver_%s_cv_%d' % (C, penalty, solver, cv) 
                if fit_intercept:
                    clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
            else: 
                clf_param = '/C_%.3f_penalty_%s_solver_%s_cv_%d' % (C, penalty, solver, 5) 
                if fit_intercept:
                    clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
        
        if 'sag' in solver:
            if cv is not None:
                clf_param = '/C_%.3f_penalty_%s_solver_%s_cv_%d_intercept_fit_%d_scaling_%d' % (C, penalty, solver, cv)
                if fit_intercept:
                    clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
            else: 
                clf_param = '/C_%.3f_penalty_%s_solver_%s_cv_%d' % (C, penalty, solver, 5) 
                if fit_intercept:
                    clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
            # if cv is not None:
            #     clf_param = '/C_%.3f_penalty_%s_solver_%s_cv_%d_l1_ratio_%.2f' % (C, penalty, solver, cv, l1_ratio[-1])
            #     if fit_intercept:
            #         clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
            # else:
            #     clf_param = '/C_%.3f_penalty_%s_solver_%s_cv_%d_l1_ratio_%.2f' % (C, penalty, solver, 5, l1_ratio[-1]) 
            #     if fit_intercept:
            #         clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )

        clf_param = clf_param + '/%s' % gv.scoring

    if 'glmnet' in gv.clf_name:
        clf_param = '/C_%.3f_l1_ratio_%.2f' % (C, l1_ratio) 
        if fit_intercept:
            clf_param = clf_param + '_fit_intercept'
            
        clf_param = clf_param + '/%s' % gv.scoring
        
    elif 'LogisticRegression' in gv.clf_name:
        if 'liblinear' in solver:
            clf_param = '/C_%.3f_penalty_%s_solver_%s' % (C, penalty, solver)
            if fit_intercept:
                clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
        if 'sag' in solver:
            clf_param = '/C_%.3f_penalty_%s_solver_%s_l1_ratio_%.2f' % (C, penalty, solver, l1_ratio)        
            if fit_intercept:
                clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
                
    elif gv.clf_name in 'LinearSVC':
        clf_param = '/C_%.3f_penalty_%s_loss_%s' % (C, penalty, loss)
        if fit_intercept:
            clf_param = clf_param + '_intercept_fit_%d_scaling_%d' % (fit_intercept, intercept_scaling )
            
    elif gv.clf_name in 'LDA': 
        clf_param = '/shrinkage_%s_solver_lsqr' % shrinkage 
    
    gv.figdir = gv.figdir +'/'+ gv.clf_name + clf_param 
    
    gv.figdir = gv.figdir + '/%s_boots' % gv.bootstrap_method 

    if not os.path.isdir(gv.figdir): 
        os.makedirs(gv.figdir) 

def get_corr_epochs(coefs): 
    
    corr = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    lower = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    upper = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    
    corr_boot = np.empty( (len(gv.trials), gv.n_boots, coefs.shape[1], coefs.shape[1]) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials):

        # for n_epoch, epoch in enumerate(gv.epochs):
        #     mean_coefs = np.mean(coefs[n_trial, n_epoch], axis=1) 
        #     print('epoch', epoch, 'non zero coefs', np.count_nonzero(mean_coefs) ) 
        
        for boot in range(gv.n_boots): 
            corr_boot[n_trial, boot] = np.corrcoef(coefs[n_trial,:,boot,:]) # bins x coefficients 
            
        corr[n_trial] = np.mean(corr_boot[n_trial], axis=0) 
        lower[n_trial] = corr[n_trial] - np.percentile(corr_boot[n_trial], 25, axis=0) 
        upper[n_trial] = np.percentile(corr_boot[n_trial], 75, axis=0) - corr[n_trial] 

        print('trial', gv.trial, 'cos', corr[n_trial,:,0], 'lower', lower[n_trial,:,0], 'upper', upper[n_trial,:,0]) 
        
    return corr[:,:,0], lower[:,:,0], upper[:,:,0], corr_boot[:,:,0] 

def plot_cos_epochs(X_trials, bootstrap_method='block', C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=None, l1_ratio=None, shrinkage='auto', fit_intercept=False, intercept_scaling=1e2): 

    create_fig_dir(C=C, penalty=penalty, solver=solver, cv=cv, loss=loss, l1_ratio=l1_ratio, shrinkage=shrinkage, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling) 

    coefs = bootstrap_coefs_epochs(X_trials, bootstrap_method, C, penalty, solver, loss, cv, l1_ratio, shrinkage, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling) 

    if gv.cos_trials:
        mean_cos, lower_cos, upper_cos, cos_boot = get_cos_trials(coefs)
    else:
        mean_cos, lower_cos, upper_cos, cos_boot = get_cos_epochs(coefs)
        
    pl.bar_trials_epochs(mean_cos, lower_cos, upper_cos) 
    # p_values_cos = get_p_values(cos_boot)  
    # print('p_values', p_values) 
    # add_pvalue(p_values_cos)
    
    if np.all(mean_cos>-0.1): 
        plt.ylim([-0.1, 1.1]) 
    else:
        plt.ylim([-1, 1.1])  
       
    figtitle = '%s_%s_bars_cos_alp' % (gv.mouse, gv.session) 
    pl.save_fig(figtitle) 
    
    # mean_corr, lower_corr, upper_corr, corr_boot = get_corr_epochs(coefs) 
    # p_values_corr = get_p_values(corr_boot) 
    # pl.bar_trials_epochs(mean_corr, lower_corr, upper_corr, var_name='corr') 
    # add_pvalue(p_values_corr) 
    # if np.all(mean_cos>-0.1): 
    #     plt.ylim([-0.1, 1.1]) 
    # else:
    #     plt.ylim([-1, 1.1])  
    
    # figtitle = '%s_%s_bars_corr' % (gv.mouse, gv.session) 
    # pl.save_fig(figtitle) 

def plot_loop_mice_sessions(clf=None, C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=None, l1_ratio=None, shrinkage='auto', fit_intercept=False, intercept_scaling=1e2): 
    
    gv.num_cores =  int(0.9*multiprocessing.cpu_count()) 
    # gv.num_cores =  int( np.sqrt(0.9*multiprocessing.cpu_count()) ) 
    gv.IF_SAVE = 1 
    gv.correct_trial = 0
    gv.pair_trials = 0
    
    gv.cos_trials = 0 
    # gv.trials = ['ND_D1', 'ND_D2'] 
    
    # classification parameters
    if clf is None:
        gv.clf_name = 'glmnet'
    else:
        gv.clf_name = clf 
        
    gv.scoring = 'f1' # 'accuracy', 'f1', 'roc_auc' or 'neg_log_loss' 'r2' 
    gv.TIBSHIRANI_TRICK = 0  
    
    # bootstrap parameters
    gv.n_boots = int(1e3) 
    gv.bootstrap_method='block' # 'bayes', 'bagging', 'standard', 'block' or 'hierarchical' 
    
    # preprocessing parameters 
    gv.T_WINDOW = 0.5 
    gv.EDvsLD = 1 # average over epochs ED, MD and LD 
    
    # only useful with dim red methods 
    gv.ED_MD_LD = 1 
    gv.DELAY_ONLY = 0 
    
    gv.SAVGOL = 0 # sav_gol filter 
    gv.Z_SCORE = 0 # z_score with BL mean and std 
    
    # feature selection 
    gv.FEATURE_SELECTION = 0 
    gv.LASSOCV = 0 
    
    # scaling before clf, when using pca use None 
    gv.scaling = 'standardize_sample' # 'standardize_sample' # 'standardize', 'normalize', 'standardize_sample', 'normalize_sample' or None 
    
    # PCA parameters 
    gv.explained_variance = .90 
    gv.n_components = None 
    gv.inflection = False 
    gv.minka_mle = False 
    gv.pca_method = None # 'hybrid', 'concatenated', 'averaged', 'supervised' or None 
    gv.max_threshold = 10 
    gv.n_thresholds = 100 
    gv.spca_scoring = 'roc_auc' # 'mse', 'log_loss' or 'roc_auc' 
    gv.spca_cv = 5 
    
    if gv.pca_method is not None: 
        # gv.scaling = None # safety for dummies 
        if gv.pca_method in 'supervised': 
            my_pca = supervisedPCA_CV(n_components=gv.n_components, explained_variance=gv.explained_variance, cv=gv.spca_cv, max_threshold=gv.max_threshold, n_thresholds=gv.n_thresholds, verbose=True, n_jobs=gv.num_cores, scoring=gv.spca_scoring) 
        else: 
            my_pca = pca_methods(pca_method=gv.pca_method,
                                 explained_variance=gv.explained_variance, inflection=gv.inflection,
                                 minka_mle=gv.minka_mle, verbose=True)     
    # PLS parameters 
    # gv.pls_n_comp = None 
    gv.pls_max_comp = 30 # 'full', int or None 
    gv.pls_method = None # 'PLSRegression', 'PLSCanonical', 'PLSSVD', CCA or None 
    gv.pls_cv = 5 
    
    if gv.pls_method is not None: 
        gv.pca_method = None  # safety for dummies 
        # gv.scaling = None # safety for dummies 
        my_pls = plsCV(cv=gv.pls_cv, pls_method=gv.pls_method, max_comp=gv.pls_max_comp, n_jobs=gv.num_cores, verbose=True) 
        
    for gv.mouse in [gv.mice[1]] : 
        fct.get_sessions_mouse() 
        fct.get_stimuli_times() 
        fct.get_delays_times() 
        
        for gv.session in gv.sessions : 
            X_trials, y = fct.get_X_y_mouse_session() 

            if gv.ED_MD_LD: 
                X_trials = X_trials[:,:,:,:,gv.bins_ED_MD_LD] 
            if gv.DELAY_ONLY: 
                X_trials = X_trials[:,:,:,:,gv.bins_delay] 
                gv.bin_start = gv.bins_delay[0] 
            
            if (gv.pca_method is not None) or (gv.pls_method is not None): 
                                    
                if gv.pca_method is not None: 
                    X_trials = my_pca.fit_transform(X_trials, y) 
                elif gv.pls_method is not None: 
                    X_trials = my_pls.trial_hybrid(X_trials, y) 
                    
            print('bootstrap samples:', gv.n_boots, ', clf:', gv.clf_name,
                  ', scaling:', gv.scaling, ', scoring:', gv.scoring, ', cv:', cv, 
                  ', pca_method:', gv.pca_method, ', pls_method:', gv.pls_method, ', n_components', X_trials.shape[3]) 
            
            matplotlib.use('Agg') # so that fig saves when in the in the background 
            # matplotlib.use('GTK3cairo') 
            plot_cos_epochs(X_trials, C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, bootstrap_method=gv.bootstrap_method) 
            plt.close('all') 
