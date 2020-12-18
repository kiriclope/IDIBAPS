from .libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis') 

import data.constants as gv 

def get_clf(C=1, penalty='l2', solver='liblinear', loss='squared_hinge', cv=None, l1_ratio=0, shrinkage='auto', normalize=True): 
    
    if 'LogisticRegressionCV' in gv.clf_name:
        gv.clf = LogisticRegressionCV(Cs=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), 
                                   fit_intercept=True, n_jobs=None , intercept_scaling=1e2 ) 
        
    elif 'LogisticRegression' in gv.clf_name:
        gv.clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8),
                                    fit_intercept=True, l1_ratio=l1_ratio, intercept_scaling=1e2, cv=cv ) 
        
    elif 'LDA' in gv.clf_name: 
        gv.clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage=shrinkage)
        
    elif 'PLS' in gv.clf_name:
        gv.clf = PLSRegression(scale=False) 

    elif 'ReLASSO' in gv.clf_name:
        gv.clf = relassoCV = RelaxedLassoLarsCV( fit_intercept=True, verbose=False, max_iter=500,
                                              normalize=normalize, precompute='auto', cv=cv, max_n_alphas=1000,
                                              n_jobs=None, eps=np.finfo(np.float).eps, copy_X=True) 
