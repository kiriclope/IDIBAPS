from .libs import *

import data.constants as gv 

# from .glmnet_wrapper import logitnet, logitnetCV

from glmnet import LogitNet, LogitNetOffDiag
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical, CCA 

def set_options(**kwargs): 
    
    opts = dict() 
    opts['clf_name']='glmnet'
    opts['F0_THRESHOLD']=None
    opts['cos_trials']=0
    opts['scores_trials']=0
    opts['laser_on']=0
    
    opts['C']=1e0 
    opts['Cs']=100 
    opts['penalty']='l1' 
    opts['solver']='liblinear'
    opts['criterion']='bic'
    
    # for glmnet only
    opts['standardize']=False
    opts['l1_ratio']=None 
    opts['lambda_path']= -np.sort(-np.logspace(-4, -2, opts['Cs'])) 
    opts['cut_point']=1
    
    opts['loss']='lsqr' 
    opts['shrinkage']='auto' 
    opts['n_splits']=3
    opts['fit_intercept']=False
    opts['Normalize']=False 
    opts['intercept_scaling']=1e2
    opts['scoring']='roc_auc'
    opts['tol']=1e-4
    opts['max_iter']=1e5 
    opts['bootstrap_method']='block'
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
        gv.clf = LogitNetOffDiag(alpha=l1_ratio, n_lambda=Cs, min_lambda_ratio=1e-4,
                                 lambda_path=lambda_path, standardize=standardize, fit_intercept=fit_intercept,
                                 lower_limits=-np.inf, upper_limits=np.inf,
                                 cut_point=cut_point, n_splits=n_splits, scoring=gv.scoring, n_jobs=None, tol=1e-7,
                                 max_iter=100000, random_state=None, max_features=None, verbose=False)
        
    elif 'glmnet' in gv.clf_name: 
        gv.clf = LogitNet(alpha=l1_ratio, n_lambda=Cs, min_lambda_ratio=1e-4, lambda_path=lambda_path,
                          standardize=standardize, fit_intercept=fit_intercept,
                          lower_limits=-np.inf, upper_limits=np.inf, cut_point=cut_point,
                          n_splits=n_splits, scoring=gv.scoring, n_jobs=None, tol=1e-7,
                          max_iter=100000, random_state=None, max_features=None, verbose=False)

    elif 'logitnetCV' in gv.clf_name:
        gv.clf = logitnetCV(alpha=l1_ratio, nlambda=Cs, nfolds=n_splits, standardize=False, fit_intercept=fit_intercept, 
                            scoring=gv.scoring, thresh=1e-4 , maxit=1e5, n_jobs=gv.num_cores) 

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
        
    clf = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=False, normalize=False, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=10, verbose=False, n_jobs=None, positive=False, random_state=None, selection='random') 
    gv.lassoCV = Pipeline([('scaler', StandardScaler()), ('clf', clf)]) 
        
