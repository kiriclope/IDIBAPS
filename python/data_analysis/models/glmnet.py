import warnings 
warnings.filterwarnings('ignore')

import glmnet_python

from glmnet_python.glmnetSet import glmnetSet 

from glmnet_python.glmnet import glmnet 
from glmnet_python.glmnetCoef import glmnetCoef 

from glmnet_python.cvglmnet import cvglmnet
from glmnet_python.cvglmnetCoef import cvglmnetCoef

class glmnet_Logit():

    def __init__(self, alpha=1, nlambda=100, standardize=True, intr=True, nfolds=None, ptype='class', thresh=1e-4 , maxit=1e5, n_jobs=1):
        
        opts = dict()
        opts['alpha'] = alpha
        opts['nlambda'] = nlambda
        opts['standardize'] = standardize
        opts['intr'] = intr
        opts['thresh'] = thresh
        opts['maxit'] = maxit
        
        self.options = glmnetSet(opts) 
        
        self.ptype = ptype 
        self.nfolds = n_folds 
        self.n_jobs = n_jobs
        
        self.model_ = None 
        self.lbd_min_ = None 
        self.lbd_1se_ = None 
        self.coef_ = None 
        
    def fit(self, X, y):
        
        if self.nfolds is None : 
            self.model_ = glmnet(x = X.copy(), y = y.copy(), family = 'binomial') 
            self.lbd_min_ = self.model_['lambda_min'] 
            self.lbd_1se_ = self.model_['lambda_1se'] 
            self.coef_ = glmnetCoef(self.model_, s = 'lambda_min') 
        else:            
            self.model_ = cvglmnet(x = X.copy(), y = y.copy(), family = 'binomial', ptype = self.ptype, nfolds=self.nfolds, parallel=self.n_jobs) 
            self.lbd_min_ = self.model_['lambda_min'] 
            self.lbd_1se_ = self.model_['lambda_1se'] 
            self.coef_ = cvglmnetCoef(self.model_, s = 'lambda_min') 
        
        return self 
