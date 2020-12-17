import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

from direpack import sprm, snipls
from direpack.cross_validation._cv_support_functions import robust_loss 
from direpack.sprm._m_support_functions import Hampel

def spls_cv(X, y, n_comp_max=3, etas=10, cv=10, n_jobs=1):
    HampelScore = make_scorer(robust_loss, greater_is_better=False, needs_proba=False, needs_threshold=False, lfun=mean_squared_error,fun=Hampel)
    res_sprm_cv = GridSearchCV(snipls(verbose=False), cv=cv, param_grid={"n_components": np.arange(1, n_comp_max).tolist(), "eta": np.linspace(0, 1, etas).tolist()}, scoring=HampelScore, n_jobs=n_jobs) 
    
    res_sprm_cv.fit(X,y) 
    
    best_model = res_sprm_cv.best_estimator_ 
    X_red = best_model.fit_transform(X,y)
    
    return X_red, best_model.coef_ 

def spls_red(X, y, n_comp=3, eta=0.5):
    model = sprm(verbose=False, n_components=n_comp, eta=eta)
    X_red = model.fit_transform(X, y)
    
    return X_red
