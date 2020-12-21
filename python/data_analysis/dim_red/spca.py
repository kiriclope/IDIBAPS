import sys
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis')
import data.constants as gv
import data.progressbar as pg

import warnings
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.metrics import log_loss 
from sklearn.model_selection import cross_val_predict 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class supervisedPCA():
    
    def __init__(self, model=LogisticRegression(), explained_variance=0.9, threshold=10, verbose=False): 
        self._model = model 
        self._explained_variance = explained_variance 
        self._threshold = threshold 
        self._verbose = verbose 

        self.score = 'neg_log_loss'
        
        self._X_proj = None 
        self._dropouts = None
        self._pca = None 
        self._n_components = None 
        
    def get_optimal_number_of_components(self, X): 
        cov = np.dot(X,X.transpose())/float(X.shape[0]) 
        U,s,v = np.linalg.svd(cov) 
        S_nn = sum(s) 
        
        for n_components in range(0,s.shape[0]): 
            temp_s = s[0:n_components] 
            S_ii = sum(temp_s)
            if (1 - S_ii/float(S_nn)) <= 1 - self._explained_variance: 
                return n_components 
            
        self._n_components = s.shape[0] 
        
        return self._n_components 
    
    def fit(self, X, y):
        X_extra_dim = X[:, np.newaxis]  # Change dimension from (x,y) to (x,1,y)        
        self._dropouts = [] 
        
        # Fit each feature with the target (predictor) and if the coefficient is less than the threshold, drop it 
        for i in range(0, X_extra_dim.shape[2]): # iterate over all the features 
            X_one_feature = X_extra_dim[:, :, i] # n_samples, 1, n_neurons
            self._model.fit(X_one_feature, y) 
            
            if (all([abs(self._model.coef_[0]) < self._threshold])): # Return True if True for all values in the list 
                self._dropouts.append(i) 
                
        if (len(self._dropouts) == X_extra_dim.shape[2]):  # all features have coef less than the threshold 
            warnings.warn('All features_coefs below threshold: %.2f, try a smaller threshold' % self._threshold ) 
        else:
            if(len(self._dropouts)>0): 
                X_extra_dim = np.delete(X_extra_dim, self._dropouts,2) 
                
        self._n_components = self.get_optimal_number_of_components(X_extra_dim[:, 0, :]) # n_samples X n_features 
        self._pca = PCA(n_components=self._n_components)
        self._X_proj = self._pca.fit_transform(X_extra_dim[:, 0, :]) # n_samples X n_features 
        # self._pca = self._pca.fit(X_extra_dim[:, 0, :]) 
        
        return self

    def fit_transform(self, X, y): 
        self.fit(X,y)
        return self._X_proj 

    def transform(self, X): 
        return self._pca.transform(X) 

    def get_coefs(self): 
        return self._n_components
    
    def get_pca(self): 
        return self._pca 
    
class supervisedPCA_CV(supervisedPCA):
    
    def __init__(self, model=LogisticRegression(), explained_variance=0.9, cv=10, max_threshold=10, Cs=10, verbose=False, n_jobs=None): 
        super().__init__(model=model, explained_variance=explained_variance, threshold=None, verbose=verbose) 
        
        self._Cs = Cs 
        self._max_threshold = max_threshold 
        self._cv = cv 
        self._n_jobs = n_jobs
        self.scaler = StandardScaler() 
        
        self._best_model = None         
        
    def fit(self, X, y):
        
        # param_grid = [{'threshold' : np.logspace(-4, 4, self._Cs)}] 
        # search = GridSearchCV( super(), param_grid=param_grid, cv=self._cv, verbose=self._verbose, n_jobs=self._n_jobs) 
        # search_result = search.fit(X, y)      
        # self._best_model = search_result.best_estimator_ 
        
        mlhs = [] 
        thresholds = np.linspace(0, self._max_threshold, self._Cs)
        
        for self._threshold in ( pg.tqdm(thresholds, desc='spca_cv') if self._verbose else thresholds): 
            super().fit(X, y) 
            y_cv = cross_val_predict(self._model, self._X_proj, y, cv=self._cv) 
            mlhs.append(log_loss(y, y_cv)) 
        mlh = np.argmax(mlhs) 
        
        print(mlhs)
        
        if self._verbose: 
            print('threshold_opt', thresholds[mlh]) 
            
        self._threshold = thresholds[mlh] 
        super().fit(X, y) 
        
        self._best_model = super()
        
        # print('threshold', self._threshold) 
        # self._best_model.fit(X,y) 
        # self._n_components = super().get_coefs()
        # self._pca = super().get_pca()
        
        if self._verbose:
            print('n_components', self._n_components, 'pca', self._pca) 
        
        return self 
    
    def fit_transform(self, X, y): 
        self.fit(X,y) 
        return self._best_model._X_proj 
    
    def trial_hybrid(self, X_trials, y):  
        
        X_avg = np.empty( (len(gv.trials), gv.n_neurons, len(gv.samples) * X_trials.shape[-1] ) ) 
        for n_trial in range(len(gv.trials)) : 
            X_avg[n_trial] = np.hstack( ( np.mean(X_trials[n_trial,0], axis=0), np.mean(X_trials[n_trial,1], axis=0) ) ) 
            
        X_avg = np.hstack(X_avg) # n_features X n_trials
        y_avg = np.array([np.zeros(int(X_avg.shape[1]/2)), np.ones(int(X_avg.shape[1]/2))]).flatten() 

        if self._verbose : 
            print('X_avg', X_avg.shape, 'y_avg', y_avg.shape) 
            
        # standardize neurons/features across trials/samples 
        self.scaler.fit(X_avg.T, y_avg)  # n_trials X n_features 
        X_avg = self.scaler.transform(X_avg.T).T # n_trials X n_features 
        
        # supervised PCA the trial averaged data 
        self.fit(X_avg.T, y_avg) # n_trials X n_features

        if (len(self._dropouts) == X_trials.shape[3]):  # all features have coef less than the threshold 
            warnings.warn('All features_coefs below threshold: %.2f, try a smaller threshold' % self._threshold ) 
        else:
            if(len(self._dropouts)>0): 
                X_trials = np.delete(X_trials, self._dropouts, 3) 
        
        if self._verbose : 
            print('n_pc', self._n_components,'explained_variance', self._explained_variance, 'pca', self._pca) 
            
        X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), self._n_components, X_trials.shape[-1]) ) 
        for i in range(X_trials.shape[0]): 
            for j in range(X_trials.shape[1]): 
                for k in range(X_trials.shape[2]): 
                    trial = self.scaler.transform(X_trials[i,j,k,:,:].T).T # neurons x time = features x samples
                    X_proj[i,j,k] = self._pca.transform(trial.T).T 
                    
        if self._verbose:            
            print('X_proj', X_proj.shape) 
            
        return X_proj 
