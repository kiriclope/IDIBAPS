import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, f_regression, mutual_info_classif, f_classif

class featSel():
    
    def __init__(self):
        self._X_feat = None
        
    def select_best(X, y, scoring=f_classif): 
        model = SelectKBest(score_func=scoring, k=X.shape[1]) 
        model.fit(X,y)         
        pval = model.pvalues_.flatten()
        non_selected = np.argwhere(pval>0.05)
        X_selected = np.delete(X, non_selected, axis=1)
        return X_selected
    
    def select_indep(X, threshold=0.9):
        
        X = pd.DataFrame(X) 
        corr = X.corr() 
        
        columns = np.full((corr.shape[0],), True, dtype=bool) 
        for i in range(corr.shape[0]): 
            for j in range(i+1, corr.shape[0]): 
                if corr.iloc[i,j] >= threshold: 
                    if columns[j]: 
                        columns[j] = False 
                        
        selected_columns = X.columns[columns] 
        
        return selected_columns 
    
    def cor_selector(X, y, num_feats):
        cor_list = []
        feature_name = X.columns.tolist()
        # calculate the correlation with y for each feature
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        # replace NaN with 0
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        # feature name
        cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
        # feature selection? 0 for not select, 1 for select
        cor_support = [True if i in cor_feature else False for i in feature_name]
        return cor_support, cor_feature
    
    def var_fit_transform(X, threshold=0.1): 
        thresh_filter = VarianceThreshold(threshold) 
        thresh_filter.fit(X) 
        return thresh_filter.get_support(True) 

