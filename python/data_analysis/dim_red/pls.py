import warnings
warnings.filterwarnings("ignore")

import numpy as np 

from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, GridSearchCV 
from sklearn.metrics import mean_squared_error 
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical 

from joblib import Parallel, delayed

class plsCV():

    def __init__(self, cv, pls_method='PLSRegression', max_comp=3, n_jobs=None, verbose=None): 

        self.cv = cv
        self.max_comp = max_comp 
        self.n_jobs = n_jobs 
        self.pls_method = pls_method 
        
        self.clf = PLSRegression()
        # self.clf = PLSSVD()
        # self.clf = PLSCanonical()
        
        self.scaler = StandardScaler() 
        self.verbose = verbose
        
        self._coefs = None
        self._opt_n_comp = None
        self._X_proj = None
        self._best_model = None
        
        self.pipe = Pipeline(steps=[('preprocessor', self.scaler), ('estimator', self.clf)])
        self.search = None 
        
    def cross_val_mse(X, y): 
        '''Run PLS including a variable number of components, up to n_comp_max, and calculate MSE ''' 
        mse = [] 
        
        n_components = np.arange(1, self.max_comp) 
        for n_comp in (pg.tqdm(n_components, desc='pls') if self.verbose else n_components): 
            pls = PLSRegression(n_components=n_comp) 
            y_cv = cross_val_predict(pls, X, y, cv=self.cv) 
            mse.append(mean_squared_error(y, y_cv))
            
        # def loop_comp(X, y, n_comp) :
        #     pls = PLSRegression(n_components=n_comp) 
        #     # pls = PLSCanonical(n_components=n_comp) 
        #     y_cv = cross_val_predict(pls, X, y, cv=10) 
        #     mse = mean_squared_error(y, y_cv) 
        #     return mse 
    
        # with pg.tqdm_joblib(pg.tqdm(desc= 'pls optimization', total=n_comp_max-1)) as progress_bar: 
        #     mse = Parallel(n_jobs=20)(delayed(loop_comp)(X, y, n_comp) for n_comp in range(1, n_comp_max)) 
    
        min_mse = np.argmin(mse) 
        if self.verbose : 
            print("Suggested number of components: ", min_mse+1) 
            
        self._opt_n_comp = min_mse + 1 
        
        return self._opt_n_comp 
    
    def fit(self, X, y):
        if isinstance(self.max_comp, str):
            self.max_comp = X.shape[1]
            
        # X=self.scaler.fit_transform(X)
        # self._opt_n_comp = self.cross_val_mse(X, y) 
        # self.clf=PLSRegression(n_components=self._opt_n_comp) 
        # self.X_proj = self.clf.fit(X,y) 
        # self._coefs = self.clf.coef_
        
        param_grid = [{'estimator__n_components' : np.arange(1, self.max_comp)}] 
        search = GridSearchCV(self.pipe, scoring='neg_root_mean_squared_error', param_grid=param_grid, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs) 
        search_result = search.fit(X, y)
        
        self._best_model = search_result.best_estimator_['estimator'] 
        self.opt_n_comp = search_result.best_params_['estimator__n_components']
        
        if self.verbose:
            print('model', self._best_model, 'n_comp', self.opt_n_comp) 
        
        self._best_model.fit(X,y)        
        self._coefs = self._best_model.coef_ 
        
        return self 
    
    def fit_transform(self, X, y):  
        if isinstance(self.max_comp, str):
            self.max_comp = X.shape[1] 
            
        # self._opt_n_comp = self.cross_val_mse(X, y) 
        # self.clf=PLSRegression(n_components=self._opt_n_comp) 
        # self.X_proj, _ = self.clf.fit_transform(X,y) 
        # self._coefs = self.clf.coef_ 
        
        param_grid = [{'estimator__n_components' : np.arange(1, self.max_comp)}] 
        search = GridSearchCV(self.pipe, scoring='neg_root_mean_squared_error', param_grid=param_grid, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs) 
        
        search_result = search.fit(X, y) 
        self._best_model = search_result.best_estimator_['estimator'] 
        self.opt_n_comp = search_result.best_params_['estimator__n_components']
        self._X_proj, _ = self._best_model.fit_transform(X,y) 
        
        return self._X_proj

    def trial_hybrid(self, X_trials, y): 
        # trial_type, neurons, samples x time 
        X_avg = np.empty( (X_trials.shape[0], X_trials.shape[3], X_trials.shape[1] * X_trials.shape[-1] ) ) 
        for n_trial in range(X_trials.shape[0]) : 
            X_avg[n_trial] = np.hstack( ( np.mean(X_trials[n_trial,0], axis=0), np.mean(X_trials[n_trial,1], axis=0) ) ) 
            
        X_avg = np.hstack(X_avg) 
        y_avg = np.array([np.zeros(int(X_avg.shape[1]/2)), np.ones(int(X_avg.shape[1]/2))]).flatten()
        
        if self.verbose: 
            print('X_avg', X_avg.shape, 'y_avg', y_avg.shape) 
            
        # standardize neurons/features across trials/samples 
        self.scaler.fit(X_avg.T) 
        X_avg = self.scaler.transform(X_avg.T).T 
        
        # PLS the trial averaged data 
        self.pipe = Pipeline(steps=[('estimator', self.clf)]) 
        self.fit(X_avg.T, y_avg) 
        
        if self.verbose :
            print('n_pc', self.opt_n_comp) 
            
        X_proj = np.empty( (X_trials.shape[0], X_trials.shape[1], X_trials.shape[2], self.opt_n_comp, X_trials.shape[-1]) ) 
        for i in range(X_trials.shape[0]): 
            for j in range(X_trials.shape[1]): 
                for k in range(X_trials.shape[2]): 
                    trial = self.scaler.transform(X_trials[i,j,k,:,:].T).T # neurons x time = features x samples 
                    trial_proj = self._best_model.transform(trial.T, y)[0].T
                    X_proj[i,j,k,0:trial_proj.shape[0]] = trial_proj 
                    
        if self.verbose : 
            print('X_proj', X_proj.shape)
            
        return X_proj 
