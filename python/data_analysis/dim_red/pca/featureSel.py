import numpy as np                                                                                                                     
from sklearn.feature_selection import VarianceThreshold

class featSel():

    def __init__(self):
        self._X_feat = None
        
    def corrFit(self, X):
        
        corr = self.X.corr() 
        
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
                        
        selected_columns = X.columns[columns] 
        self._X_feat = X[selected_columns]

        return self 

    def varFit(self, X, threshold=0.1):
        thresh_filter = VarianceThreshold(threshold)
        self._X_feat = thresh_filter.fit_transform(X) 
