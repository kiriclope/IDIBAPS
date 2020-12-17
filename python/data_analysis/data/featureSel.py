import numpy as np                                                                                                                     
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold 

class featSel():

    def __init__(self):
        self._X_feat = None

    def select_best(X, y, percentage=0.9):
        model = SelectKBest(score_func=chi2, k=int(percentage*X.shape[1]) ) 
        return model.fit_transform(X,y)
    
    def select_indep( X, threshold=0.9):
        
        corr = X.corr() 
        
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= threshold:
                    if columns[j]:
                        columns[j] = False
                        
        selected_columns = X.columns[columns] 
        
        return X[selected_columns] 

    def cor_selector(X, y,num_feats):
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
        return thresh_filter.fit_transform(X)         

