import numpy as np 

from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler 

from joblib import Parallel, delayed

class plsCV():

    def __init__(self, cv, pls_method='PLSRegression', max_comp=3, n_jobs=None, verbose=None): 

        self.cv = cv
        self.max_comp = max_comp 
        self.n_jobs = n_jobs 
        self.pls_method = pls.method 
        
        self.clf = PLSRegression()
        # self.clf = PLSSVD()
        # self.clf = PLSCanonical()
        
        self.scaler = StandardScaler() 
        self.verbose = verbose
        
        self._coefs = None
        self._opt_n_comp = None
        self._X_proj = None 
        
        self.pipe = Pipeline(steps=[('preprocessor', self.scaler), ('estimator', self.clf)]) 
        param_grid = [{'estimator__n_components' : np.arange(0, self.max_comp)}] 
        self.search = GridSearchCV(self.pipe, scoring='neg_mean_squared_error', param_grid=param_grid, cv=self.cv, verbose=False, n_jobs=self.n_jobs)
        
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
        # self._opt_n_comp = self.cross_val_mse(X, y) 
        # self.clf=PLSRegression(n_components=self._opt_n_comp) 
        # self.X_proj = self.clf.fit(X,y) 
        # self._coefs = self.clf.coef_
        
        search_result = self.search.fit(X, y) 
        self._best_model = search_result.best_estimator_ 
        self._best_model.fit(X,y) 
        self._coefs = self._best_model.coef_ 
        
        return self
    
    def fit_transform(self, X, y): 
        # self._opt_n_comp = self.cross_val_mse(X, y) 
        # self.clf=PLSRegression(n_components=self._opt_n_comp) 
        # self.X_proj = self.clf.fit_transform(X,y) 
        # self._coefs = self.clf.coef_
        
        search_result = self.search.fit(X, y) 
        self._best_model = search_result.best_estimator_ 
        self._X_proj = self._best_model.fit_transform(X,y) 
        
        return self._X_proj 
