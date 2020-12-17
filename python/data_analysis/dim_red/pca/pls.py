import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed

def pls_cv_mse(X, y, n_comp_max, cv=10, verbose=True): 
    '''Run PLS including a variable number of components, up to n_comp_max, and calculate MSE ''' 
    mse = [] 
    n_components = np.arange(1, n_comp_max) 
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

def pls_transform(X, y, verbose=True):
    n_comp = pls_cv_mse(X, y, 2*X.shape[0], verbose=verbose) 
    pls = PLSRegression(n_components=n_comp) 
    X, _ = pls.fit_transform(X,y) 
    return X, pls.coef_ 
