from .libs import * 
import data.constants as gv 
# from .glmnet import glmnet_Logit 
from glmnet import LogitNet 
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical, CCA 

def get_clf(C=1, penalty='l2', solver='liblinear', loss='squared_hinge', cv=None, l1_ratio=None, shrinkage='auto', normalize=True, fit_intercept=True, intercept_scaling=1e2, tol=1e-4, max_iter=1e5): 
    
    if 'LogisticRegressionCV' in gv.clf_name:
        gv.clf = LogisticRegressionCV(Cs=np.logspace(-4,4,C), solver=solver, penalty=penalty, l1_ratios=l1_ratio,
                                      tol=tol, max_iter=int(max_iter), scoring=gv.scoring,
                                      fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
                                      cv=cv, n_jobs=None) 
        
    elif 'LogisticRegression' in gv.clf_name:
        gv.clf = LogisticRegression(C=C, solver=solver, penalty=penalty, l1_ratio=l1_ratio,
                                    tol=tol, max_iter=int(max_iter),
                                    fit_intercept=fit_intercept,  intercept_scaling=intercept_scaling,
                                    n_jobs=None) 
        
    elif 'LDA' in gv.clf_name: 
        gv.clf = LinearDiscriminantAnalysis(tol=tol, solver='lsqr', shrinkage=shrinkage)
        
    elif 'PLS' in gv.clf_name:
        gv.clf = PLSRegression(scale=False) 
        
    elif 'ReLASSO' in gv.clf_name:
        gv.clf = relassoCV = RelaxedLassoLarsCV( fit_intercept=False, verbose=False, max_iter=500,
                                              normalize=False, precompute='auto', cv=cv, max_n_alphas=1000,
                                              n_jobs=None, eps=np.finfo(np.float).eps, copy_X=True) 
    elif 'LinearSVC' in gv.clf_name:
        gv.clf = LinearSVC(C=C, penalty=penalty, loss=loss, dual=False,
                           tol=tol, max_iter=int(max_iter), multi_class='ovr',
                           fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                           class_weight=None, verbose=0, random_state=None) 

    elif 'glmnet' in gv.clf_name:
        gv.clf = LogitNet(alpha=1, n_lambda=100, min_lambda_ratio=1e-4,
                          lambda_path=None, standardize=False, fit_intercept=fit_intercept,
                          lower_limits=-np.inf, upper_limits=np.inf,
                          cut_point=1.0, n_splits=cv, scoring=gv.scoring, n_jobs=-1, tol=1e-7,
                          max_iter=100000, random_state=None, max_features=None, verbose=False)

    elif 'pycasso' in gv.clf_name:
        gv.clf = pycasso.Solver(X,Y, lambdas=(100,0.05), family="binomial", penalty="l1")         

    elif 'CCA' in gv.clf_name:
        gv.clf = CCA(n_components=20, scale=False, max_iter=500, tol=1e-06, copy=True) 
        
    clf = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=False, normalize=False, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=10, verbose=False, n_jobs=None, positive=False, random_state=None, selection='random') 
    gv.lassoCV = Pipeline([('scaler', StandardScaler()), ('clf', clf)]) 
        
