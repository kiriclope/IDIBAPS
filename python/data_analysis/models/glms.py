from .libs import *

import data.constants as gv 

from .glmnet_wrapper import logitnet, logitnetCV, logitnetStratCV

from python_glmnet import LogitNet, LogitNetOffDiag, LogitNetAlphaCV 
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical, CCA 
import multiprocessing 

def set_globals(**opts):
    
    gv.num_cores = opts['num_cores']
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
    gv.AVG_F0_TRIALS = 0 

    gv.Z_SCORE = opts['Z_SCORE']
    gv.Z_SCORE_BL = opts['Z_SCORE_BL']

    gv.DETREND = 0 # detrend the data 
    gv.SAVGOL = 0 # sav_gol filter 

    gv.T_WINDOW = opts['T_WINDOW'] 
    gv.EDvsLD = opts['EDvsLD'] 
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
    
def set_options(**kwargs): 
    
    opts = dict()

    opts['num_cores'] = int(0.9*multiprocessing.cpu_count()) 
    opts['IF_SAVE'] = 1 
    opts['SYNTHETIC'] = 0
    
    # globals 
    opts['i_mice'] = 1
    opts['i_day'] = -1
    opts['i_trial'] = 0  
    opts['i_epoch'] = 0 

    opts['laser_on']=0

    # bootstrap 
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

    opts['DECONVOLVE']=0 
    opts['DCV_THRESHOLD']=0.5 

    opts['F0_THRESHOLD']=None 

    opts['Z_SCORE'] = 0 
    opts['Z_SCORE_BL'] = 0 

    # classification parameters 
    opts['clf_name']='logitnetCV' 
    opts['scoring'] = 'accuracy' # 'accuracy', 'f1', 'roc_auc' or 'neg_log_loss' 'r2' 

    # sklearn LogisticRegression, LogisticRegressionCV
    opts['C']=1e0 
    opts['Cs']=100 
    opts['penalty']='l1' 
    opts['solver']='liblinear'

    # LDA
    opts['loss']='lsqr' 
    opts['shrinkage']='auto'

    # LassoLarsIC
    opts['criterion']='bic'

    opts['fit_intercept']=False
    opts['intercept_scaling']=1e2

    # for glmnet only 
    opts['n_splits']=3 
    opts['alpha'] = None 
    opts['n_alpha'] = 10 
    opts['n_lambda'] = 100 
    opts['alpha_path']= None # -np.sort(-np.logspace(-4, -2, opts['Cs'])) 
    opts['min_lambda_ratio'] = 1e-3

    opts['off_diag']=True 
    opts['standardize']=False
    opts['lambda_path']= None # -np.sort(-np.linspace(-3, -1, opts['Cs'])) 
    opts['cut_point']=1
    
    opts['shuffle'] = True     
    opts['random_state'] = None 
    opts['tol']=1e-4
    opts['max_iter']= int(1e5) 

    opts.update(kwargs)
    
    return opts 

def get_clf(**kwargs):
    options = set_options(**kwargs)
    globals().update(options) 
    
    if 'LogisticRegressionCV' in gv.clf_name:
        gv.clf = LogisticRegressionCV(Cs=np.logspace(-4,4,Cs), solver=solver, penalty=penalty, l1_ratios=l1_ratio, 
                                      tol=tol, max_iter=int(max_iter), scoring=gv.scoring, 
                                      fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
                                      cv=n_splits, n_jobs=None) 
        
    elif 'LogisticRegression' in gv.clf_name:
        gv.clf = LogisticRegression(C=C, solver=solver, penalty=penalty, l1_ratio=l1_ratio,
                                    tol=tol, max_iter=int(max_iter),
                                    fit_intercept=fit_intercept,  intercept_scaling=intercept_scaling,
                                    n_jobs=None) 
        
    elif 'LDA' in gv.clf_name: 
        gv.clf = LinearDiscriminantAnalysis(tol=tol, solver='lsqr', shrinkage=shrinkage) 
        
    elif 'PLS' in gv.clf_name:
        gv.clf = PLSRegression(scale=False) 
        
    elif 'ReLASSO' in gv.clf_name:
        gv.clf = relassoCV = RelaxedLassoLarsCV( fit_intercept=False, verbose=False, max_iter=500,
                                              normalize=False, precompute='auto', cv=n_splits, max_n_alphas=1000,
                                              n_jobs=None, eps=np.finfo(np.float).eps, copy_X=True) 
    elif 'LinearSVC' in gv.clf_name:
        gv.clf = LinearSVC(C=C, penalty=penalty, loss=loss, dual=False,
                           tol=tol, max_iter=int(max_iter), multi_class='ovr',
                           fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                           class_weight=None, verbose=0, random_state=None) 

    elif 'glmnet_off_diag' in gv.clf_name: 
        gv.clf = LogitNetOffDiag(alpha=alpha, n_lambda=n_lambda, min_lambda_ratio=min_lambda_ratio,
                                 lambda_path=lambda_path, standardize=standardize, fit_intercept=fit_intercept,
                                 lower_limits=-np.inf, upper_limits=np.inf,
                                 cut_point=cut_point, n_splits=n_splits, scoring=gv.scoring, n_jobs=None, tol=1e-7,
                                 max_iter=100000, random_state=None, max_features=None, verbose=False)
        
    elif 'glmnetCV' in gv.clf_name: 
        gv.clf = LogitNetAlphaCV(alpha=alpha, off_diag=off_diag,n_alpha=n_alpha, alpha_path=alpha_path,
                                 n_lambda=n_lambda, min_lambda_ratio=min_lambda_ratio, lambda_path=lambda_path,
                                 standardize=standardize, fit_intercept=fit_intercept,
                                 lower_limits=-np.inf, upper_limits=np.inf, cut_point=cut_point,
                                 n_splits=n_splits, scoring=gv.scoring, n_jobs=None, tol=tol,
                                 max_iter=100000, random_state=None, max_features=None, verbose=False) 
    
    elif 'lognet' in gv.clf_name: 
        gv.clf = LogitNet(alpha=alpha, n_lambda=n_lambda, min_lambda_ratio=min_lambda_ratio, lambda_path=lambda_path,
                          standardize=standardize, fit_intercept=fit_intercept,
                          lower_limits=-np.inf, upper_limits=np.inf, cut_point=cut_point,
                          n_splits=n_splits, scoring=gv.scoring, n_jobs=None, tol=tol,
                          max_iter=max_iter, random_state=None, max_features=None, verbose=False)    

    elif 'logitnetCV' in gv.clf_name:
        gv.clf = logitnetCV(alpha=alpha, n_lambda=n_lambda, n_splits=n_splits,
                            standardize=standardize, fit_intercept=fit_intercept, 
                            fold_type=fold_type, shuffle=shuffle, random_state=random_state,
                            scoring=gv.scoring, thresh=1e-4 , maxit=1e5, n_jobs=gv.num_cores)
        
    elif 'logitnetStratCV' in gv.clf_name:
        gv.clf = logitnetStratCV(alpha=alpha, n_lambda=n_lambda, n_splits=n_splits,
                                 standardize=standardize, fit_intercept=fit_intercept, 
                                 scoring=gv.scoring, shuffle=shuffle, random_state=random_state,
                                 thresh=tol , maxit=max_iter, n_jobs=None) 

    elif 'logitnet' in gv.clf_name:
        gv.clf = logitnet(alpha=l1_ratio, nlambda=Cs, standardize=False, fit_intercept=fit_intercept,
                          scoring=gv.scoring, thresh=1e-4 , maxit=1e5, n_jobs=gv.num_cores)        
        
    elif 'pycasso' in gv.clf_name:
        gv.clf = pycasso.Solver(X,Y, lambdas=(100,0.05), family="binomial", penalty="l1")         

    elif 'CCA' in gv.clf_name:
        gv.clf = CCA(n_components=20, scale=False, max_iter=500, tol=1e-06, copy=True) 

    elif 'lassolarsIC':
        LassoLarsIC(criterion=criterion, fit_intercept=fit_intercept, verbose=False,
                    normalize=standardize, precompute='auto', max_iter=500, eps=2.220446049250313e-16, copy_X=True, positive=False) 
    elif 'sgd':
        SGDClassifier(loss='log', penalty=penalty, alpha=0.0001, l1_ratio=l1_ratio, fit_intercept=False, max_iter=1000, tol=0.00001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False) 
        
    clf = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=False, normalize=False, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=10, verbose=False, n_jobs=None, positive=False, random_state=None, selection='random') 
    gv.lassoCV = Pipeline([('scaler', StandardScaler()), ('clf', clf)]) 
        
