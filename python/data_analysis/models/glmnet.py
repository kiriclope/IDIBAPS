import warnings 
warnings.filterwarnings('ignore') 

from sklearn.base import BaseEstimator, ClassifierMixin 

import glmnet_py 

from glmnet_py.glmnetSet import glmnetSet 

from glmnet_py.glmnet import glmnet 
from glmnet_py.glmnetCoef import glmnetCoef 

from glmnet_py.cvglmnet import cvglmnet 
from glmnet_py.cvglmnetCoef import cvglmnetCoef 

class logitnet(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, alpha=1, nlambda=100, standardize=False, fit_intercept=False,
                 scoring='auc', thresh=1e-4 , maxit=1e5, n_jobs=1): 
        
        opts = dict()
        opts['alpha'] = alpha 
        opts['nlambda'] = nlambda 
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        opts['thresh'] = thresh 
        opts['maxit'] = maxit 
        
        self.options = glmnetSet(opts)
        
        if 'accuracy' in scoring:
            self.scoring = 'class'
        if 'roc_auc' in scoring:
            self.scoring = 'auc'
            
        self.n_jobs = n_jobs 
        
        self.model_ = None 
        self.lbd_min_ = None 
        self.lbd_1se_ = None 
        self.coef_ = None 
        
    def fit(self, X, y): 
        self.model_ = glmnet(x = X.copy(), y = y.copy(), family = 'binomial', options=self.options) 
        self.lbd_min_ = self.model_['lambda_min'] 
        self.lbd_1se_ = self.model_['lambda_1se'] 
        self.coef_ = glmnetCoef(self.model_, s = 'lambda_1se') 
        
        return self 
    
    def predict(self, X): 
        return glmnetPredict(self.model_, X, ptype='class') 
    
    def predict_proba(self, X): 
        return glmnetPredict(self.model_, X, ptype='response') 
    
class logitnetCV(BaseEstimator, ClassifierMixin): 
    
    def __init__(self, alpha=1, nlambda=100, nfolds=None,
                 standardize=False, fit_intercept=False,
                 scoring='auc', thresh=1e-4 , maxit=1e5, n_jobs=1): 
        
        opts = dict()
        opts['alpha'] = alpha 
        opts['nlambda'] = nlambda 
        opts['standardize'] = standardize 
        opts['intr'] = fit_intercept 
        opts['thresh'] = thresh 
        opts['maxit'] = maxit 
        
        self.options = glmnetSet(opts) 
        
        # self.scoring = scoring # 'deviance', 'class', 'auc', 'mse' or 'mae'
        
        if 'accuracy' in scoring:
            self.scoring = 'class'
        if 'roc_auc' in scoring:
            self.scoring = 'auc'
            
        self.nfolds = nfolds 
        self.n_jobs = n_jobs 
        
        self.model_ = None 
        self.lbd_min_ = None 
        self.lbd_1se_ = None 
        self.coef_ = None 
        
    def fit(self, X, y): 
        self.model_ = cvglmnet(x = X.copy(), y = y.copy(), family = 'binomial', ptype = self.scoring,
                               nfolds=self.nfolds, parallel=self.n_jobs, options=self.options) 
        
        self.lbd_min_ = self.model_['lambda_min']  
        self.lbd_1se_ = self.model_['lambda_1se'] 
        
        self.coef_ = cvglmnetCoef(self.model_, s='lambda_1se') 
        
    def predict(self, X): 
        return cvglmnetPredict(self.model_, X, ptype='class') 
    
    def predict_proba(self, X): 
        return cvglmnetPredict(self.model_, X, ptype='response')     
