import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

class spca_model():
    
    def __init__(model=None, threshold=0, n_components=0):    
        self._model = model
        self._threshold = threshold
        self._n_components = n_components
        
        self._dropouts = None
        self._pca = None
        
    
    def fit_spca_model(self, x, y):
        x_extra_dim = x[:, np.newaxis]  # Change dimension from (x,y) to (x,1,y)
        # x.reshape(x.shape + (1,))
        
        self._dropouts = []
        
        # Fit each feature with the target (predictor) and if the coefficient is less than the threshold, drop it
        for i in range(0, x_extra_dim.shape[2]): # iterate over all the features
            x_one_feature = x_extra_dim[:, :, i]
            self._model.fit(x_one_feature, y)
            
            if (all([abs(self._model.coef_[0]) < self._threshold])): # Return True if True for all values in the list
                self._dropouts.append(i)
                
        if (len(self._dropouts) == x_extra_dim.shape[2]):  # all features have coef less than the threshold
            raise ValueError('Try a smaller threshold')
            
        if (self._n_components > 0):  # calculate the most important n_components
            self._pca = PCA(n_components=self._n_components)
        else: # if n_components is not more than 0. then all components are kept
            self._pca = PCA(n_components=x_extra_dim.shape[2])
        
        x_tranform = self._pca.fit_transform(x_extra_dim[:, 0, :])
        
        self._model = self._model.fit(x_transform, y)
        
        return self

    
    def spca_predict(self, x):
        # delete from x those features which had coef less than the threshold, and were in the dropout set
        trans_x = np.delete(x, self._dropouts, axis=1)  
        
        # the remaining features have been transformed into components using PCA weights 
        trans_x = self._pca.transform(trans_x)
        
        # predict using the spca model
        return self._model.predict(trans_x)
    
    
    def get_num_components(self):
        # return the number of components after the pca fit
        return self._pca.n_components_

    
    def get_components(self):
        # shape : (n_components, n_features)
        # Principal axes in feature space, representing the directions of maximum variance in the data.
        # Should be run only after the PCA has been run on the training dataset
        return self._pca.components_

    
    def get_coefficients(self):
        # return the coefs of the model the data has been fit on
        return self._model.coef_
    
    
    def get_score(self, x, y):
        # return score of the reduced dimensional data on the model
        return self._model.score(x, y)
