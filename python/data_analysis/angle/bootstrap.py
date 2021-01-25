from copy import deepcopy
import warnings 
warnings.filterwarnings("ignore") 

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from bayesian_bootstrap import bootstrap as bayes 
from joblib import Parallel, delayed

import data.progressbar as pg
import data.constants as gv

class bootstrap():
    
    def __init__(self, clf, bootstrap_method='standard', n_boots=1000, scaling='standardize', gridsearch= None, Vh=None, n_jobs=1):
        
        if scaling is None:
            self.pipe = Pipeline([('clf', clf)]) 
        elif scaling in 'standardize': 
            self.scaler =  StandardScaler() 
            self.pipe = Pipeline([('clf', clf)]) 
        elif scaling in 'standardize_sample':
            self.pipe = Pipeline([('scale', StandardScaler()), ('clf', clf)]) 
        elif scaling in 'normalize': 
            self.scaler =  MinMaxScaler()
            self.pipe = Pipeline([('clf', clf)])
        elif scaling in 'normalize_sample': 
            self.pipe = Pipeline([('scale', MinMaxScaler()), ('clf', clf)]) 
            
        if gridsearch is not None: 
            param_grid = [{'C' : np.logspace(-4, 4, 10)} ] 
            self.pipe = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=False, n_jobs=None) 
            
        self.scaling = scaling 
        self.bootstrap_method = bootstrap_method 
        self.n_boots = n_boots 
        self.Vh = Vh 
        self.n_jobs = n_jobs 
        
        self._coefs = None
        
    def my_bootstrap_loop(self, X, y): 
        
        if self.n_boots==1: 
            X_sample = X 
            y_sample = y 
        else: 
            if 'standard' in self.bootstrap_method :
                idx_trials = np.random.randint(0, X.shape[0], X.shape[0]) 
                
            if 'block' in self.bootstrap_method :
                idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)), 
                                          np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) )
                
            X_sample = X[idx_trials] 
            y_sample = y[idx_trials] 
            
            if 'hierarchical' in self.bootstrap_method :
                for trial in idx_trials: 
                    idx_neurons = np.random.randint(0, X.shape[1], X.shape[1]) 
                    X_sample[trial] = X[trial, idx_neurons] 
                
        # if self.scaling is not None:
        #     if 'sample' in self.scaling: 
        #         X_sample = self.scaler.transform(X_sample) 
        
        pipe_copy = deepcopy(self.pipe)
        pipe_copy.fit(X_sample, y_sample) 
        boots_coefs = pipe_copy[-1].coef_.flatten() 
        
        if self.Vh is not None: 
            boots_coefs = self.Vh.T.dot(boots_coefs).flatten() 
            
        return boots_coefs 
    
    def my_bootstrap(self, X, y): 
        if self.scaling is not None:
            if not 'sample' in self.scaling: 
                self.scaler.fit(X) 

        if self.n_boots>1:
            bar_name = gv.trial + ' ' + gv.epoch +', '+ self.bootstrap_method + ' bootstrap' 
            with pg.tqdm_joblib(pg.tqdm(desc=bar_name, total=self.n_boots)) as progress_bar: 
                boots_coefs = Parallel(n_jobs=self.n_jobs)(delayed(self.my_bootstrap_loop)(X, y) for _ in range(self.n_boots) ) 
            self._coefs = np.array(boots_coefs)
        else:
            # for _ in range(self.n_boots):
            boots_coefs = self.my_bootstrap_loop(X, y) 
            self._coefs = np.array(boots_coefs) 
            self._coefs = self._coefs[np.newaxis] 
            
    def bayesian_bootstrap(self, X, y): 
        model = bayes.BayesianBootstrapBagging(self.pipe, self.n_boots, X.shape[1], n_jobs=self.n_jobs) 
        model.fit(X, y) 
        
        if self.Vh is not None: 
            coefs = [ self.Vh.T.dot(estimator[-1].coef_.flatten() ).flatten() for estimator in model.base_models_ ] 
        else: 
            coefs = [ estimator[-1].coef_.flatten() for estimator in model.base_models_ ] 
            
        self._coefs = np.array(coefs) 

    def bagging_bootstrap(self, X, y): 
        model = BaggingRegressor(base_estimator=self.pipe, n_estimators=self.n_boots, n_jobs=self.n_jobs, bootstrap_features=False) 
        model.fit(X, y)
        
        if self.Vh is not None:
            coefs = [ self.Vh.T.dot(estimator[-1].coef_.flatten() ).flatten() for estimator in model.estimators_ ] 
        else:
            coefs = [ estimator[-1].coef_.flatten() for estimator in model.estimators_ ] 

        self._coefs = np.array(coefs) 

    def get_coefs(self, X, y, Vh): 
        self.Vh = Vh
        
        if self.bootstrap_method in 'bagging':
            self.bagging_bootstrap(X, y) 
        elif self.bootstrap_method in 'bayesian': 
            self.bayesian_bootstrap(X, y)
        else:
            self.my_bootstrap(X, y)
            
        return self._coefs 
