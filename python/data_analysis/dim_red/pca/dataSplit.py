from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')

import data.constants as gv 
import data.plotting as pl 
import data.preprocessing as pp 

from sklearn.model_selection import train_test_split

def avg_epochs(X):
    
    X_STIM = np.mean(X[:,:,gv.bins_STIM[-3:]-gv.bin_start],axis=2) 
    X_ED = np.mean(X[:,:,gv.bins_ED[-3:]-gv.bin_start],axis=2) 
    X_MD = np.mean(X[:,:,gv.bins_MD[-3:]-gv.bin_start],axis=2) 
    X_LD = np.mean(X[:,:,gv.bins_LD[-3:]-gv.bin_start],axis=2) 

    X_epochs = np.array([X_STIM, X_ED, X_MD, X_LD]) 
        
    X_epochs = np.moveaxis(X_epochs,0,2) 
    return X_epochs 

def decision(coefs, X, intercept):
    return np.dot(X, coefs) + intercept

def get_se(X, y, clf):
    """StdErr per variable estimation.
    https://en.wikipedia.org/wiki/Ordinary_least_squares 
    """
    MSE = np.mean((y - clf.predict(X).T)**2) 
    # numerically unstable below with openblas if rcond is less than that
    var_est = MSE * np.diag(np.linalg.pinv(np.dot(X.T, X), rcond=1e-10))
    SE_est = np.sqrt(var_est)
    return SE_est

def get_coefs_ci(clf, X, SE_est, z=1.96):
    """Estimate CI given data, StdErrors and model."""
    coefs = np.ravel(clf.coef_)
    upper = coefs + (z * SE_est)
    lower = coefs - (z * SE_est)

def get_prob_ci(clf, X, SE_est, z=1.96):
    """Estimate CI given data, StdErrors and model."""
    coefs = np.ravel(clf.coef_)
    upper = coefs + (z * SE_est)
    lower = coefs - (z * SE_est)
    prob = 1. / (1. + np.exp(-decision(coefs, X, clf.intercept_)))
    upper_prob = 1. / (1. + np.exp(-decision(upper, X, clf.intercept_)))
    lower_prob = 1. / (1. + np.exp(-decision(lower, X, clf.intercept_)))

    stacked = np.vstack((lower_prob, upper_prob))
    up = np.max(stacked, axis=0)
    lo = np.min(stacked, axis=0)
    return prob, up, lo

def splitDataClf(X, y, clf): 
    # split the data into two samples 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) 

    # fit logistic lasso on the training set 
    clf.fit(X_train, y_train) 
    
    # perform inference on the test set 
    coefs, upper, lower = get_coefs_ci(clf, X_test, SE_est, z=1.96) 

    return coefs, upper, lower


def getCoefsTrials(X_trials, C=1e0, penalty='l1', solver='liblinear', cv=10): 

    clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(gv.standardize)) 
 
    gv.AVG_EPOCHS = 1 
    gv.trial_size = X_trials.shape[-1]
 
    # if pca reduced data 
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:]    
    
    if gv.AVG_EPOCHS: 
        gv.trial_size = len(['STIM','ED','MD','LD']) 
        
    mean_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    upper_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    lower_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    
    y = np.array([np.zeros(X_trials.shape[2]), np.ones(X_trials.shape[2])]).flatten() 

    for n_trial, gv.trial in enumerate(gv.trials): 
    
        X_S1 = X_trials[n_trial,0] 
        X_S2 = X_trials[n_trial,1] 
        X_S1_S2 = np.vstack((X_S1, X_S2)) 

        if gv.AVG_EPOCHS: 
            X_S1_S2 = avg_epochs(X_S1_S2) 

        print('X_S1_S2', X_S1_S2.shape) 

        for n_bins in range(gv.trial_size): 
            X = X_S1_S2[:,:,n_bins]
            coefs, upper, lower = splitDataClf(X, y, clf)
            
            mean_coefs[n_trials, n_bins] = coefs
            upper_coefs[n_trials, n_bins] = upper
            lower_coefs[n_trials, n_bins] = lower
            
    return mean_coefs, upper_coefs, lower_coefs 

