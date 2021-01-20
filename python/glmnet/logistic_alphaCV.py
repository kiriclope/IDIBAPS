import sys 
sys.setrecursionlimit(10**6) 

import numpy as np
from scipy.special import expit
from scipy.sparse import issparse, csc_matrix
from scipy import stats

from copy import deepcopy
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from joblib import Parallel, delayed

from .errors import _check_error_flag
from _glmnet import lognet, splognet, lsolns
from glmnet.util import (_fix_lambda_path,
                         _check_user_lambda,
                         _interpolate_model,
                         _score_lambda_path)

from .logistic import LogitNet 

class LogitNetAlphaCV(LogitNet): 
    
    def __init__(self, n_alpha=100, alpha_path=None, n_lambda=100, min_lambda_ratio=1e-4, 
                 lambda_path=None, standardize=True, fit_intercept=True,
                 lower_limits=-np.inf, upper_limits=np.inf, shuffle=True,
                 cut_point=1.0, n_splits=3, scoring=None, n_jobs=1, tol=1e-7,
                 max_iter=100000, random_state=None, max_features=None, verbose=False): 

        # super().__init__(alpha=1, n_lambda=n_lambda, min_lambda_ratio=min_lambda_ratio, lambda_path=lambda_path, cut_point=cut_point,
        #                  standardize=standardize, fit_intercept=fit_intercept, lower_limits=lower_limits, upper_limits=upper_limits, 
        #                  n_splits=n_splits, scoring=scoring, n_jobs=n_jobs, tol=tol, max_iter= max_iter,
        #                  shuffle=shuffle, random_state=random_state, max_features=max_features, verbose=verbose)

        self.base_model = LogitNet(alpha=1, n_lambda=n_lambda, min_lambda_ratio=min_lambda_ratio, lambda_path=lambda_path, cut_point=cut_point,
                                   standardize=standardize, fit_intercept=fit_intercept, lower_limits=lower_limits, upper_limits=upper_limits, 
                                   n_splits=n_splits, scoring=scoring, n_jobs=n_jobs, tol=tol, max_iter= max_iter,
                                   shuffle=shuffle, random_state=random_state, max_features=max_features, verbose=verbose)
        
        self.base_model.random_state = np.random.randint(0, 1e6)
        self.n_jobs = n_jobs
        self.verbose = verbose 
        # self.shuffle = shuffle 
        self.n_alpha = n_alpha 
        self.alpha_path = alpha_path 

    def fit(self, X, y):
                
        if self.n_alpha==0 | self.n_alpha is None : 
            raise ValueError("n_alpha must be positive setting n_alpha to 100") 
            self.n_alpha = 100 
            
        if self.alpha_path is None: 
            self.alpha_path = np.linspace(0,1, self.n_alpha) 
        
        # fix seed of the folds for all alphas 
        # if self.shuffle :
        #     self.random_state = np.random.randint(0, 1e6) 
        
        # self.scores_alpha = np.empty(self.n_alpha) 

        model_alpha = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='loky')(
            delayed(self.fit_fixed_alpha)(X, y, self.alpha_path[i_alpha]) 
            for i_alpha in range(self.n_alpha) ) 
        self.model_alpha = np.array(model_alpha) 
        
        # self.model_alpha = np.empty(self.n_alpha)
        # self.model_alpha = []
        self.scores_alpha = []        
        for i_alpha in range(self.n_alpha): 
            # self.model_alpha.append( self.fit_fixed_alpha(X, y, self.alpha_path[i_alpha]) ) 
            self.scores_alpha.append( self.model_alpha[i_alpha].scores_lambda_best_ )

            if self.verbose:
                print('model', self.model_alpha[i_alpha], 'score', self.scores_alpha[i_alpha])
            
        self.best_score_idx_ = np.argmax(self.scores_alpha) 
        self.best_score_ = self.scores_alpha[self.best_score_idx_] 
        self.best_estimator_ = self.model_alpha[self.best_score_idx_] 
        
        self.coef_ = self.best_estimator_.coef_ 
        self.best_alpha_ = self.best_estimator_.alpha 
        self.best_lambda_ = self.best_estimator_.lambda_best_ 
        
        return self 
    
    def fit_fixed_alpha(self, X, y, alpha=1):
        # copy base model 
        # model = deepcopy(super()) 
        model = deepcopy(self.base_model)
        # set value of alpha  
        model.alpha = alpha        
        # fit estimator 
        model.fit(X, y) 

        return model 
    
