import sys
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis')
import data.progressbar as pg

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline

from bayesian_bootstrap import bootstrap as bayes 

from joblib import Parallel, delayed

class bootstrap():

    def __init__(self, clf, bootstrap_method='standard', n_boots=1000, scaling='standardize', n_jobs=1):
        
        if 'standardize' in scaling:
            self.pipe = Pipeline([('scale', StandardScaler()), ('clf', clf)]) 
        elif 'normalize' in scaling: 
            self.pipe = Pipeline([('scale', MinMaxScaler()), ('clf', clf)]) 
        else: 
            self.pipe = Pipeline([('clf', clf)]) 
            
        self.bootstrap_method = bootstrap_method
        self.n_boots = n_boots
        self.n_jobs = n_jobs
        
        self._coefs = None
        
    def my_bootstrap_loop(self, X, y, Vh=None): 
        
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
                
        self.pipe.fit(X_sample, y_sample) 
        boots_coefs = self.pipe[-1].coef_.flatten() 

        if Vh is not None: 
            boots_coefs = Vh.T.dot(boots_coefs).flatten() 
            
        return boots_coefs

    def my_bootstrap(self, X, y, Vh=None):
        with pg.tqdm_joblib(pg.tqdm(desc= 'my_bootstrap ', total=self.n_boots)) as progress_bar: 
            boots_coefs = Parallel(n_jobs=self.n_jobs)(delayed(self.my_bootstrap_loop)(X, y, Vh) for _ in range(self.n_boots) ) 
        boots_coefs = np.array(boots_coefs) 
        return boots_coefs 
    
    def bayesian_bootstrap(self, X, y, Vh=None): 
        model = bayes.BayesianBootstrapBagging(self.pipe, self.n_boots, X.shape[1], n_jobs=self.n_jobs) 
        model.fit(X, y) 
        
        if Vh is not None: 
            coefs = [ Vh.T.dot(estimator[-1].coef_.flatten() ).flatten() for estimator in model.base_models_ ] 
        else: 
            coefs = [ estimator[-1].coef_.flatten() for estimator in model.base_models_ ] 
        
        self.coefs = np.array(coefs)
        return self.coefs 

    def bagging_bootstrap(self, X, y, n_estimators, Vh=None):
        model = BaggingRegressor(base_estimator=self.pipe, n_estimators=self.n_boots, n_jobs=self.n_jobs, bootstrap_features=False) 
        model.fit(X, y)
        
        if Vh is not None:
            coefs = [ Vh.T.dot(estimator[-1].coef_.flatten() ).flatten() for estimator in model.estimators_ ] 
        else:
            coefs = [ estimator[-1].coef_.flatten() for estimator in model.estimators_ ] 

        self._coefs = np.array(coefs) 
        return self._coefs 
