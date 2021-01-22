import warnings 
warnings.filterwarnings('ignore') 

from copy import deepcopy

import scipy
import scipy.stats as stats
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin 
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_squared_error 
from sklearn.model_selection import KFold, StratifiedKFold 

from glmnet_python.glmnetSet import glmnetSet 

from glmnet_python.glmnet import glmnet 
from glmnet_python.glmnetCoef import glmnetCoef 
from glmnet_python.glmnetPredict import glmnetPredict 

from glmnet_python.cvglmnet import cvglmnet 
from glmnet_python.cvglmnetCoef import cvglmnetCoef 
from glmnet_python.cvglmnetPredict import cvglmnetPredict 

from glmnet_python.cvglmnetPlot import cvglmnetPlot

from joblib import Parallel, delayed

import data.progressbar as pg

class logitnet(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, alpha=1, n_lambda=100, standardize=False, fit_intercept=False, thresh=1e-4 , maxit=1e6):
        
        self.alpha = alpha 
        self.n_lambda = n_lambda 
        self.fit_intercept = fit_intercept 
        self.thresh = thresh 
        self.maxit = maxit 
        
        opts = dict() 
        opts['alpha'] = scipy.float64(alpha)
        opts['nlambda'] = scipy.int32(n_lambda)
        
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        
        opts['thresh'] = scipy.float64(thresh) 
        opts['maxit'] = scipy.int32(maxit) 
        
        self.options = glmnetSet(opts)        
        self.lbd_best_ = None 
        
    def fit(self, X, y):
        model_ = glmnet(x = X.copy(), y = y.copy(), family = 'binomial', **self.options) 
        self.model_ = model_ # for some reason I have to pass it like that a = funcnet() then self.a = a         
        return self 
    
    def get_coefs(self, obj=None, lbd=None, exact=False):
        if lbd is None: 
            lbd= self.lbd_best_
        if obj is None:
            coefs = glmnetCoef(self.model_, s = scipy.float64([lbd]), exact = exact)
        else:
            coefs = glmnetCoef(obj, s = scipy.float64([lbd]), exact = exact)
            
        self.intercept_ = coefs[0] 
        self.coef_ = coefs[1:] 
        
        return coefs
    
    def predict(self, X, lbd=None): 
        if lbd is None: 
            lbd= self.lbd_best_ 
        return glmnetPredict(self.model_, newx=X, ptype='class', s = scipy.float64([lbd]) ) 
    
    def predict_proba(self, X, lbd=None): 
        if lbd is None: 
            lbd= self.lbd_best_             
        return glmnetPredict(self.model_, newx=X, ptype='response', s = scipy.float64([lbd]) ) 
    
    def score(self, X, y, lbd=None): 
        if self.scoring=='class': 
            y_pred = self.predict(X, lbd) 
            return accuracy_score(y, y_pred) 
        if self.scoring=='auc': 
            y_pred = self.predict_proba(X, lbd) 
            return roc_auc_score(y, y_pred) 
        if self.scoring=='deviance': 
            y_pred = self.predict_proba(X, lbd) 
            return log_loss(y, y_pred) 
        if self.scoring=='mse': 
            y_pred = self.predict(X) 
            return mean_squared_error(y, y_pred) 
    
class logitnetCV(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, alpha=1, n_lambda=100, n_splits=10, standardize=False, fit_intercept=False, shuffle=True,
                 random_state=None, scoring='class', thresh=1e-4 , maxit=1e6, n_jobs=1):

        opts = dict() 
        opts['alpha'] = scipy.float64(alpha)
        opts['nlambda'] = scipy.int32(n_lambda)
        
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept
        
        opts['thresh'] = scipy.float64(thresh)
        opts['maxit'] = scipy.int32(maxit)
        
        self.options = glmnetSet(opts) 
        self.scoring = scoring # 'deviance', 'class', 'auc', 'mse' or 'mae'
        
        if 'accuracy' in scoring: 
            self.scoring = 'class'  
        if 'roc_auc' in scoring: 
            self.scoring = 'auc' 
        if 'log_loss' in scoring: 
            self.scoring = 'deviance' 
            
        self.scoring = scoring
        self.n_splits = 16  
        self.n_jobs = n_jobs 
        
    def fit(self, X, y):
        # y = y[:,np.newaxis]
        model_ = cvglmnet(x = X.copy(), y = y.copy(), family = 'binomial', ptype = self.scoring,
                          nfolds=self.n_splits, parallel=self.n_jobs, **self.options) 
        
        self.model_ = model_ # for some reason I have to pass it like that a = funcnet() then self.a = a 
        
        cv_mean_score = model_['cvm'] 
        cv_standard_error = model_['cvsd'] 
        lbd_min_ = model_['lambda_min']  
        lbd_1se_ = model_['lambda_1se'] 
        
        self.cv_mean_score_ = cv_mean_score 
        self.cv_standard_error_ = cv_standard_error         
        
        self.lbd_min_ = lbd_min_
        self.lbd_1se_ = lbd_1se_
        
        coef_ = cvglmnetCoef(model_, s='lambda_1se')[1:] 
        self.coef_ = coef_
        
        return self
    
    def lasso_path(self, cvfit): 
        cvglmnetPlot(cvfit)
        print('lambda_min', cvfit['lambda_min'], 'lambda_1se', cvfit['lambda_1se']) 

    def predict(self, X): 
        return cvglmnetPredict(self.model_, newx=X, ptype='class') 
    
    def predict_proba(self, X): 
        return cvglmnetPredict(self.model_,newx=X, ptype='response') 

    def score(self, X, y):
        y = y[:,np.newaxis]
        if self.scoring=='class':
            y_pred = cvglmnetPredict(self.model_, newx=X, ptype='class')
            return accuracy_score(y, y_pred) 
        if self.scoring=='auc':            
            y_pred = cvglmnetPredict(self.model_, newx=X, ptype='response') # equivalent to predictProba 
            return roc_auc_score(y, y_pred) 
        if self.scoring=='deviance': 
            y_pred = cvglmnetPredict(self.model_, newx=X, ptype='response') # equivalent to predictProba 
            return log_loss(y, y_pred) 
        if self.scoring=='mse': 
            y_pred = cvglmnetPredict(self.model_, newx=X, ptype='class') 
            return mean_squared_error(y, y_pred) 
        
class logitnetStratCV(logitnet, BaseEstimator, ClassifierMixin): 
    def __init__(self, alpha=1, n_lambda=100, n_splits=10, standardize=False, fit_intercept=False, shuffle=True,
                 random_state=None, scoring='class', thresh=1e-4 , maxit=1e6, n_jobs=1, verbose=True):
        
        # init logitnet 
        super().__init__(alpha=alpha, n_lambda=n_lambda, 
                         standardize=standardize, fit_intercept=fit_intercept, 
                         thresh=thresh, maxit=maxit) 
        
        self.scoring = scoring 
        # redefines scoring to match glmnet notations: 'deviance', 'class', 'auc', 'mse' or 'mae'        
        if 'accuracy' in scoring: 
            self.scoring = 'class'  
        if 'roc_auc' in scoring: 
            self.scoring = 'auc' 
        if 'log_loss' in scoring: 
            self.scoring = 'deviance' 
                
        # parameters of the cv 
        self.n_splits = n_splits 
        self.shuffle = shuffle 
        self.random_state = random_state 
        self.n_jobs = n_jobs 
        self.verbose = verbose 
        
    def scoresCV(self, X, y, cv_splits): 
        
        # with pg.tqdm_joblib(pg.tqdm(desc='cross validation', total= int(self.n_splits * self.lbd_path_.size) )) as progress_bar: 
        cv_scores = Parallel(n_jobs=self.n_jobs)(delayed(self.scoresFoldLambda)(X, y, idx_train, idx_test, lbd) 
                                                 for (idx_train, idx_test) in cv_splits for lbd in self.lbd_path_) 
            
        cv_scores = np.array(cv_scores).reshape( self.n_splits, self.n_lambda) 
        # cv_scores = np.empty((self.n_splits, self.n_lambda)) 
        # i_splits = -1 
        
        # for idx_train, idx_test in cv_splits: 
        #     i_splits = i_splits + 1 
        #     i_lbd = -1
            
        #     for lbd in self.lbd_path_ :
        #         i_lbd = i_lbd + 1                
        #         print(i_splits, i_lbd)
        #         cv_scores[i_splits, i_lbd] = self.scoresFoldLambda(X, y, idx_train, idx_test, lbd) 
                
        # if self.verbose :
            # print('cv_scores', cv_scores.shape)
            
        return cv_scores 
    
    def scoresFoldLambda(self, X, y, idx_train, idx_test, lbd):
        '''For a given lambda, fits lognet on X_train and return score for X_test'''
        X_train, y_train = X[idx_train], y[idx_train] 
        X_test, y_test = X[idx_test], y[idx_test] 
        
        super().fit(X_train, y_train) 
        fold_score = super().score(X_test, y_test, lbd)
        
        return fold_score 
        
    def fit(self, X, y):
        
        # initial fit on all the data 
        super().fit(X, y) 
        self.base_model = self.model_
        coefs = self.base_model['beta'] 
        
        # self.intercept_ = coefs[0] 
        self.coef_ = coefs 
        
        self.lbd_path_ = self.model_['lambdau'] 
        
        folds = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state) 
        cv_splits = folds.split(X, y) 
        
        # compute cv scores for all lambdas 
        cv_scores = self.scoresCV(X, y, cv_splits) 
        
        if self.scoring == 'auc':
            cv_scores = -cv_scores 
            
        self.cv_mean_score_ = np.atleast_1d(np.mean(cv_scores, axis=0)) 
        self.cv_std_error_ = np.atleast_1d(stats.sem(cv_scores)) 
        
        self.lbd_min_ = np.amax(self.lbd_path_[self.cv_mean_score_ <= np.amin(self.cv_mean_score_)]).reshape([1]) 
        self.idx_lbd_min_ = self.lbd_path_ ==  self.lbd_min_ 
        
        self.cv_min_std_ = self.cv_mean_score_[self.idx_lbd_min_] + self.cv_std_error_[self.idx_lbd_min_] 
        self.lbd_best_ = np.amax(self.lbd_path_[self.cv_mean_score_ <= self.cv_min_std_]).reshape([1]) 
        self.idx_lbd_best_ = self.lbd_path_ ==  self.lbd_best_ 
        
        self.coef_ = self.coef_[:, self.idx_lbd_best_].flatten()
        
        # self.idx_lbd_min = np.argmin(self.cv_mean_score_) 
        # self.lbd_min_ = self.lbd_path_[self.idx_lbd_min_] 
        
        # self.cv_min_score_ = self.cv_mean_score_[self.idx_lbd_min_] 
        # self.cv_min_std_ = self.cv_std_error_[self.idx_lbd_min_] 
        
        # target_score = self.cv_min_score_ - self.cut_point * self.cv_min_std_ 
        # self.idx_lbd_best_ = np.argwhere(self.cv_mean_score_ >= target_score)[0] 
        # self.lbd_best_ = self.lbd_path_[self.idx_lbd_best_] 
        
        # super().get_coefs(self.base_model, lbd=self.lbd_best_, exact=False) 
        
        return self 
    
