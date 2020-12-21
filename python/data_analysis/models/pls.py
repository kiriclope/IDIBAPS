import numpy as np 
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed

class plsCV():

    def __init__(self, cv, pls_method='PLSRegression', max_comp=3, n_jobs=None, verbose=None): 
        self.cv = cv
        self.n_jobs = n_jobs
        self.pls_method = pls.method

        self.clf = PLSRegression()
        # self.clf = PLSSVD()
        # self.clf = PLSCanonical()

        self._coefs = None
        self.max_comp = max_comp
        
    def cross_val_mse(X, y): 
        '''Run PLS including a variable number of components, up to n_comp_max, and calculate MSE ''' 
        mse = [] 
        n_components = np.arange(1, self.max_comp) 
        for n_comp in (pg.tqdm(n_components, desc='pls') if verbose else n_components): 
            pls = PLSRegression(n_components=n_comp) 
            y_cv = cross_val_predict(pls, X, y, cv=cv) 
            mse.append(mean_squared_error(y, y_cv))
            if verbose :
                print("Suggested number of components: ", msemin+1) 
    
        # def loop_comp(X, y, n_comp) :
        #     pls = PLSRegression(n_components=n_comp) 
        #     # pls = PLSCanonical(n_components=n_comp) 
        #     y_cv = cross_val_predict(pls, X, y, cv=10) 
        #     mse = mean_squared_error(y, y_cv) 
        #     return mse 
    
        # with pg.tqdm_joblib(pg.tqdm(desc= 'pls optimization', total=n_comp_max-1)) as progress_bar: 
        #     mse = Parallel(n_jobs=20)(delayed(loop_comp)(X, y, n_comp) for n_comp in range(1, n_comp_max)) 
    
        msemin = np.argmin(mse)    
        return msemin+1

    def fit(self, X, y):
        self.n_comp = self.cross_val_mse(X, y) 
        self.clf.fit(X,y)
        self._coefs = self.clf.coef_ 

    def fit_transform(self, X, y):
        self.n_comp = self.cross_val_mse(X, y) 
        X_transf = self.clf.fit_transform(X,y) 
        self._coefs = self.clf.coef_ 
        return X_transf 
