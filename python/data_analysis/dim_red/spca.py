import data.constants as gv 
import data.progressbar as pg 

import warnings
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA 
from sklearn.metrics import log_loss, mean_squared_error , roc_auc_score 
from sklearn.model_selection import cross_val_predict 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

class supervisedPCA(BaseEstimator, ClassifierMixin):
    
    def __init__(self, model=LogisticRegression(), explained_variance=0.9, n_components=None, threshold=10, scaling=None, verbose=False):

        # if scaling is None:
        #     self.pipe = Pipeline([('clf', clf)]) 
        # elif scaling in 'standardize': 
        #     self.scaler =  StandardScaler() 
        #     self.pipe = Pipeline([('scale', StandardScaler()), ('clf', clf)]) 

        self._model = model 
        self._explained_variance = explained_variance 
        self._n_components = n_components 
        self._threshold = threshold 
        self._verbose = verbose 

        self.score = 'neg_log_loss'
        
        self._X_proj = None 
        self._dropouts = None
        self._pca = None
        self.list_n_components = np.empty(len(gv.trials)) 
        
    def get_inflexion_point(self, explained_variance): 
        d2_var = np.gradient(np.gradient(explained_variance)) 
        inflection_point = np.argwhere(np.diff(np.sign(d2_var)))[0][0]
        return np.maximum(inflection_point,1) 
    
    def get_optimal_number_of_components(self, X): 
        cov = np.dot(X,X.transpose())/float(X.shape[0]) 
        U,s,v = np.linalg.svd(cov) 
        S_nn = sum(s) 
        
        for n_components in range(0, s.shape[0]): 
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
        elif(len(self._dropouts)>0): 
            X_extra_dim = np.delete(X_extra_dim, self._dropouts, 2) 
            
        self._n_components = self.get_optimal_number_of_components(X_extra_dim[:, 0, :]) # n_samples X n_features 
        # self._n_components = np.min( X_extra_dim.shape[0], X_extra_dim.shape[2]) 
        
        # if self._verbose:
        #     print(self._n_components, X_extra_dim[:, 0, :].shape) 
        
        self._pca = PCA(n_components=self._n_components) 
        
        self._X_proj = self._pca.fit_transform(X_extra_dim[:, 0, :]) # n_samples X n_features 
        
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
    
    def __init__(self, model=LogisticRegression(), explained_variance=0.9, n_components=None, cv=5, max_threshold=100, n_thresholds=100, scaling=None, verbose=False, n_jobs=None, scoring='mse'): 
        super().__init__(model=model, explained_variance=explained_variance, n_components=n_components, threshold=None, scaling=scaling, verbose=verbose) 
        
        self._n_thresholds = n_thresholds 
        self._max_threshold = max_threshold 
        self._cv = cv 
        self._n_jobs = n_jobs 
        self.scaler = StandardScaler() 
        self.scoring = scoring
        
        self._best_model = None 
        
    def fit(self, X, y):
        
        # param_grid = [{'threshold' : np.logspace(-4, 4, self._n_thresholds)}] 
        # search = GridSearchCV( super(), param_grid=param_grid, cv=self._cv, verbose=self._verbose, n_jobs=self._n_jobs) 
        # search_result = search.fit(X, y)      
        # self._best_model = search_result.best_estimator_ 
        
        mlhs = [] 
        thresholds = np.linspace(0, self._max_threshold, self._n_thresholds) 
        
        if self.scoring in 'mse': 
            scorer = mean_squared_error 
        elif self.scoring in 'log_loss' : 
            scorer = log_loss 
        elif self.scoring in 'roc_auc':
            scorer = roc_auc_score 
            
        for self._threshold in ( pg.tqdm(thresholds, desc='spca_cv') if self._verbose else thresholds): 
            super().fit(X, y) 
            y_cv = cross_val_predict(self._model, self._X_proj, y, cv=3) 
            mlhs.append(scorer(y, y_cv)) 
            
        mlh = np.argmin(mlhs) 
            
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
        
    def trial_hybrid(self, X_trials, y): 
        
        X_avg = np.empty( (len(gv.trials), gv.n_neurons, len(gv.samples) * X_trials.shape[-1] ) ) 
        X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), X_trials.shape[-2], X_trials.shape[-1]) ) 
        for n_trial in range(len(gv.trials)) : 
            X_avg[n_trial] = np.hstack( ( np.mean(X_trials[n_trial,0], axis=0), np.mean(X_trials[n_trial,1], axis=0) ) ) 
            
            # X_avg = np.hstack(X_avg) # n_features X n_trials
            y_avg = np.array([np.zeros(int(X_avg[n_trial].shape[1]/2)), np.ones(int(X_avg[n_trial].shape[1]/2))]).flatten() 
        
            if self._verbose : 
                print('X_avg', X_avg[n_trial].shape, 'y_avg', y_avg.shape) 
            
            # standardize neurons/features across trials/samples 
            self.scaler.fit(X_avg[n_trial].T, y_avg)  # n_trials X n_features 
            X_avg[n_trial] = self.scaler.transform(X_avg[n_trial].T).T # n_trials X n_features 
        
            # supervised PCA the trial averaged data 
            self.fit(X_avg[n_trial].T, y_avg) # n_trials X n_features 
            self.list_n_components[n_trial] = self._n_components 
            if self._verbose : 
                print('n_pc', self._n_components,'explained_variance', self._explained_variance, 'pca', self._pca) 
            
            # X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), self._n_components, X_trials.shape[-1]) ) 
            # for i in range(X_trials.shape[0]): 
            for j in range(X_trials.shape[1]): 
                for k in range(X_trials.shape[2]): 
                    # scaling
                    trial = self.scaler.transform(X_trials[n_trial,j,k,:,:].T).T # neurons x time = features x samples 
                    # remove dropouts 
                    if (len(self._dropouts) == X_trials.shape[3]):  # all features have coef less than the threshold 
                        warnings.warn('All features_coefs below threshold: %.2f, try a smaller threshold' % self._threshold ) 
                    elif(len(self._dropouts)>0): 
                        trial = np.delete(trial, self._dropouts, 0)
                    # decomposition
                    trial_proj = self._pca.transform(trial.T).T
                    X_proj[n_trial,j,k,0:trial_proj.shape[0]] = trial_proj 
                    
        if self._verbose:            
            print('X_proj', X_proj.shape) 
            
        return X_proj 

    def fit_transform(self, X, y):
        return self.trial_hybrid(X, y) 
