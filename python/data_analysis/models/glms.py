from .libs import * 

from .glmnet_wrapper import logitnet, logitnetCV, logitnetAlphaCV, iterAlphaCV 
from python_glmnet import LogitNet, LogitNetAlphaCV

from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical, CCA 
import multiprocessing 

import data.constants as gv 

def set_globals(**opts):
    
    gv.num_cores = opts['n_jobs']
    gv.IF_SAVE = opts['IF_SAVE']
    gv.SYNTHETIC = opts['SYNTHETIC'] 
    
    # parameters 
    gv.mouse = gv.mice[opts['i_mice']] 
    gv.trial = gv.trials[opts['i_trial']] 
    gv.day = gv.days[opts['i_day']] 
    gv.epoch = gv.epochs[opts['i_epoch']] 

    gv.cos_trials = opts['cos_trials']
    gv.scores_trials = opts['scores_trials']
    
    # preprocessing
    gv.DECONVOLVE= opts['DECONVOLVE']
    gv.DCV_THRESHOLD = opts['DCV_THRESHOLD']
    
    gv.F0_THRESHOLD = opts['F0_THRESHOLD']
    gv.AVG_F0_TRIALS = opts['F0_AVG_TRIALS']

    gv.Z_SCORE = opts['Z_SCORE']
    gv.Z_SCORE_BL = opts['Z_SCORE_BL']

    gv.DETREND = 0 # detrend the data 
    gv.SAVGOL = 0 # sav_gol filter 

    gv.T_WINDOW = opts['T_WINDOW'] 
    gv.EDvsLD = opts['EDvsLD']
    gv.ED_MD_LD = opts['ED_MD_LD'] 
    gv.DELAY_ONLY = 0 

    # feature selection 
    gv.FEATURE_SELECTION = 0 
    gv.LASSOCV = 0 
        
    # bootstrap 
    gv.n_boots = opts['n_boots'] 
    gv.bootstrap_method = opts['bootstrap_method'] 
    gv.bootstrap_cos = opts['boot_cos'] 
    gv.n_cos_boots = opts['n_cos_boots'] 

    gv.correct_trial = opts['correct_trials']
    gv.pair_trials = opts['pair_trials']    

    # temporal decoder
    gv.my_decoder = opts['my_decoder']
    gv.fold_type = opts['fold_type']
    gv.n_iter = opts['n_iter']
    
    # classification parameters 
    gv.clf_name = opts['clf_name']    
    gv.scoring = opts['scoring'] 
    gv.TIBSHIRANI_TRICK = 0 

    # dimensionality reduction 
    gv.AVG_BEFORE_PCA = 1 
    gv.pca_model = None # PCA, sparsePCA, supervisedPCA or None 
    gv.pls_method = None # 'PLSRegression', 'PLSCanonical', 'PLSSVD', CCA or None 

    gv.fix_alpha_lbd = opts['fix_alpha_lbd']
    
def set_options(**kwargs): 
    
    opts = dict()
    opts['verbose'] = 0 

    opts['n_jobs'] = int(0.9*multiprocessing.cpu_count()) 
    opts['IF_SAVE'] = 1 
    opts['SYNTHETIC'] = 0

    opts['fix_alpha_lbd'] = 0
    
    # globals 
    opts['i_mice'] = 1
    opts['i_day'] = -1
    opts['i_trial'] = 0  
    opts['i_epoch'] = 0 

    opts['laser_on']=0

    # bootstrap
    opts['boots'] = False 
    opts['n_boots'] = int(1e3) 
    opts['bootstrap_method'] = 'block' # 'bayes', 'bagging', 'standard', 'block' or 'hierarchical' 
    opts['boot_cos'] = 0
    opts['n_cos_boots'] = int(1e3)

    opts['cos_trials']=0
    opts['correct_trials']=0
    opts['pair_trials']=0

    # temporal decoder 
    opts['scores_trials']=0
    opts['n_iter']=100
    opts['my_decoder'] = 0 
    opts['fold_type'] = 'stratified' 
    
    # preprocessing parameters 
    opts['T_WINDOW'] = 0 
    opts['EDvsLD'] = 1 # average over epochs ED, MD and LD 
    opts['ED_MD_LD'] = 0 
    
    opts['DECONVOLVE']=0 
    opts['DCV_THRESHOLD']=0.5 
    
    opts['F0_THRESHOLD']=None 
    opts['F0_AVG_TRIALS'] = 1
    
    opts['Z_SCORE'] = 0 
    opts['Z_SCORE_BL'] = 0 
    
    # classification parameters 
    opts['clf_name']='logitnetCV' 
    opts['scoring'] = 'roc_auc' # 'accuracy', 'f1', 'roc_auc' or 'neg_log_loss' 'r2' 
    opts['inner_scoring'] = 'accuracy' # 'accuracy', 'f1', 'roc_auc' or 'neg_log_loss' 'r2' 
    opts['inner_splits'] = 10 # 'accuracy', 'f1', 'roc_auc' or 'neg_log_loss' 'r2' 
    
    # sklearn LogisticRegression, LogisticRegressionCV
    opts['C']=1e0 
    opts['Cs']=100 
    opts['penalty']='elasticnet' 
    opts['solver']='saga'
    
    # LDA
    opts['loss']='lsqr' 
    opts['shrinkage']='auto'

    # LassoLarsIC
    opts['criterion']='bic'

    opts['fit_intercept'] = True 
    opts['intercept_scaling']=1e2

    # for glmnet only 
    opts['n_splits']=3 
    opts['alpha'] = 1 
    opts['n_alpha'] = 10 
    opts['n_lambda'] = 100 
    opts['alpha_path']= None # -np.sort(-np.logspace(-4, -2, opts['Cs'])) 
    opts['min_lambda_ratio'] = 1e-4 
    opts['prescreen'] = False 

    opts['lbd'] = 'lambda_1se'
    
    opts['off_diag']=True 
    opts['standardize']=True 
    opts['lambda_path']= None # -np.sort(-np.linspace(-3, -1, opts['Cs'])) 
    opts['cut_point']=1 
    
    opts['shuffle'] = True     
    opts['random_state'] = None 
    opts['tol']=1e-7 
    opts['max_iter']= int(1e5) 

    opts.update(kwargs) 
    
    return opts 

def get_clf(**kwargs):
    
    options = set_options(**kwargs)
    globals().update(options) 
    
    # sklearn
    
    if 'LDA' in gv.clf_name: 
        gv.clf = LinearDiscriminantAnalysis(tol=tol, solver='lsqr', shrinkage=shrinkage) 
        
    if 'PLS' in gv.clf_name:
        gv.clf = PLSRegression(scale=False) 
        
    if 'LinearSVC' in gv.clf_name:
        gv.clf = LinearSVC(C=C, penalty=penalty, loss=loss, dual=False,
                           tol=tol, max_iter=int(max_iter), multi_class='ovr',
                           fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                           class_weight=None, verbose=0, random_state=None) 

    if 'LogisticRegressionCV' in gv.clf_name:
        gv.clf = LogisticRegressionCV(Cs=np.logspace(0,4,Cs), solver=solver, penalty=penalty, l1_ratios=np.linspace(0,1,10), 
                                      tol=tol, max_iter=int(max_iter), scoring=gv.scoring, 
                                      fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
                                      cv=n_splits, n_jobs=None) 
        
    elif 'LogisticRegression' in gv.clf_name:
        gv.clf = LogisticRegression(C=C, solver=solver, penalty=penalty, l1_ratio=l1_ratio,
                                    tol=tol, max_iter=int(max_iter),
                                    fit_intercept=fit_intercept,  intercept_scaling=intercept_scaling,
                                    n_jobs=None) 
    if 'lassolarsIC':
        LassoLarsIC(criterion=criterion, fit_intercept=fit_intercept, verbose=False,
                    normalize=standardize, precompute='auto', max_iter=500, eps=2.220446049250313e-16, copy_X=True, positive=False) 

    # python_glmnet 
    if 'lognetAlphaCV' in gv.clf_name: 
        gv.clf = LogitNetAlphaCV(n_alpha=n_alpha, alpha_path=alpha_path,
                                 n_lambda=n_lambda, min_lambda_ratio=min_lambda_ratio, lambda_path=lambda_path,
                                 standardize=standardize, fit_intercept=fit_intercept,
                                 lower_limits=-np.inf, upper_limits=np.inf, cut_point=cut_point,
                                 n_splits=inner_splits, scoring=inner_scoring, n_jobs=None, tol=tol,
                                 max_iter=100000, shuffle=shuffle, random_state=None, max_features=None, verbose=False) 
        
    elif 'lognetCV' in gv.clf_name: 
        gv.clf = LogitNet(alpha=alpha, n_lambda=n_lambda, min_lambda_ratio=min_lambda_ratio, lambda_path=lambda_path,
                          standardize=standardize, fit_intercept=fit_intercept,
                          lower_limits=-np.inf, upper_limits=np.inf, cut_point=cut_point,
                          n_splits=inner_splits, scoring=inner_scoring, n_jobs=None, tol=tol,
                          max_iter=max_iter, shuffle=shuffle, random_state=None, max_features=None, verbose=False)    

    # glmnet_python
    if 'logitnetAlphaCV' in gv.clf_name:
        gv.clf = logitnetAlphaCV(lbd=lbd, n_alpha=n_alpha, n_lambda=n_lambda, 
                                 n_splits=inner_splits, fold_type=fold_type, scoring=inner_scoring,
                                 standardize=standardize, fit_intercept=fit_intercept, prescreen=prescreen, 
                                 thresh=tol, maxit=max_iter, n_jobs=None, verbose=verbose)
        
    elif 'logitnetCV' in gv.clf_name:
        gv.clf = logitnetCV(lbd=lbd, alpha=alpha, n_lambda=n_lambda, n_splits=inner_splits,
                            standardize=standardize, fit_intercept=fit_intercept, prescreen=prescreen,
                            fold_type=fold_type, shuffle=shuffle, random_state=random_state,
                            scoring=inner_scoring, thresh=tol , maxit=max_iter, n_jobs=None) 
                
    elif 'logitnet' in gv.clf_name: 
        gv.clf = logitnet(lbd=lbd, alpha=alpha, n_lambda=n_lambda, prescreen=prescreen, 
                          standardize=standardize, fit_intercept=fit_intercept, 
                          scoring=gv.scoring, thresh=tol , maxit=max_iter) 
        
    if 'iterAlphaCV' in gv.clf_name:
        gv.clf = iterAlphaCV(n_iter=n_iter, lbd=lbd, n_alpha=n_alpha, n_lambda=n_lambda, 
                                 n_splits=inner_splits, fold_type=fold_type, scoring=inner_scoring,
                                 standardize=standardize, fit_intercept=fit_intercept, prescreen=prescreen, 
                                 thresh=tol, maxit=max_iter, n_jobs=None, verbose=verbose)
        
