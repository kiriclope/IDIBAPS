import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_predict

class supervisedPCA():
    
    def __init__(self, model=None, threshold=0, n_components=0, verbose=False): 
        self._model = model
        self._threshold = threshold
        self._n_components = n_components
        self._verbose = verbose
        
        self._X_transform = None
        self._dropouts = None
        self._pca = None
        self._coef = None
        
    def fit(self, X, y):
        X_extra_dim = X[:, np.newaxis]  # Change dimension from (x,y) to (x,1,y)        
        self._dropouts = []
        
        # Fit each feature with the target (predictor) and if the coefficient is less than the threshold, drop it
        for i in range(0, X_extra_dim.shape[2]): # iterate over all the features
            X_one_feature = X_extra_dim[:, :, i]
            self._model.fit(X_one_feature, y)
            
            if (all([abs(self._model.coef_[0]) < self._threshold])): # Return True if True for all values in the list
                self._dropouts.append(i)
                
        # if (len(self._dropouts) == X_extra_dim.shape[2]):  # all features have coef less than the threshold
        #     raise ValueError('Try a smaller threshold') 
        
        if (self._n_components > 0):  # calculate the most important n_components
            self._pca = PCA(n_components=self._n_components) 
        else: # if n_components is not more than 0. then all components are kept 
            self._pca = PCA(n_components=X_extra_dim.shape[2]) 
            
        self._X_transform = self._pca.fit_transform(X_extra_dim[:, 0, :]) 
        
        self._model = self._model.fit(self._X_transform, y) 
        self.coef_ = self._model.coef_ 
        
        return self
        
    def spca_cv(self, X, y, cv=10, threshold_max=10, Cs=10): 
        mlhs = [] 
        thresholds = np.linspace(0, threshold_max, Cs) 
        for self._threshold in ( pg.tqdm(thresholds, desc='spca_cv') if self._verbose else thresholds): 
            self = self.fit(X, y) 
            y_cv = cross_val_predict(self._model, self._X_transform, y, cv=cv) 
            mlhs.append(log_loss(y, y_cv)) 

        mlh = np.argmax(mlhs) 
        
        if self._verbose: 
            print('threshold_opt', thresholds[mlh]) 
            
        self._threshold = thresholds[mlh] 
        self = self.fit(X, y) 
        
        return self.coef_ 
