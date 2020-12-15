from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.plotting as pl 
import data.preprocessing as pp 

from joblib import Parallel, delayed 
import multiprocessing 

from sklearn.model_selection import train_test_split 

def get_X_y_sessions_trials(n_trial, X_trials): 
        
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
    
    gv.AVG_EPOCHS = 0 
    gv.trial_size = X_trials.shape[-1] 
    
    if gv.AVG_EPOCHS: 
        gv.trial_size = len(['ED','MD','LD']) 
    
    y = np.array([np.zeros(X_trials.shape[2]), np.ones(X_trials.shape[2])]).flatten() 
    
    X_S1 = X_trials[n_trial,0] 
    X_S2 = X_trials[n_trial,1] 
    X_S1_S2 = np.vstack((X_S1, X_S2)) 

    return X_S1_S2, y 

def get_X_y_trials(n_trial, X_trials): 
        
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
    
    gv.AVG_EPOCHS = 0 
    gv.trial_size = X_trials.shape[-1] 
    
    if gv.AVG_EPOCHS: 
        gv.trial_size = len(['ED','MD','LD']) 
    
    y = np.array([np.zeros(X_trials.shape[2]), np.ones(X_trials.shape[2])]).flatten() 
    
    X_S1 = X_trials[n_trial,0] 
    X_S2 = X_trials[n_trial,1] 
    X_S1_S2 = np.vstack((X_S1, X_S2)) 

    return X_S1_S2, y 

def datasplit(X, y, C):

    # split the data into two samples 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) 
    print('X_train, y_train', X_train.shape, y_train.shape)
    
    # fit logistic lasso on the training set
    model = sm.Logit(y_train, X_train)
    results = model.fit_regularized(alpha=1/C) 
    print(results.summary()) 

    # perform inference on the test set
    # y_pred = results.predict(X_test) 
    predictions = result.get_prediction(X_test)
    print(predictions.summary())

def avg_epochs(X):
    
    X_STIM = np.mean(X[:,:,gv.bins_STIM[:]-gv.bin_start],axis=2) 
    X_ED = np.mean(X[:,:,gv.bins_ED[:]-gv.bin_start],axis=2) 
    X_MD = np.mean(X[:,:,gv.bins_MD[:]-gv.bin_start],axis=2) 
    X_LD = np.mean(X[:,:,gv.bins_LD[:]-gv.bin_start],axis=2) 

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
    print(MSE)
    # numerically unstable below with openblas if rcond is less than that 
    var_est = MSE * np.diag(np.linalg.pinv(np.dot(X.T, X), rcond=1.e-8)) 
    SE_est = np.sqrt(var_est)
    return SE_est

def get_coefs_ci(clf, X, SE_est, z=1.96):
    """Estimate CI given data, StdErrors and model."""
    coefs = np.ravel(clf.coef_) 
    upper = coefs + (z * SE_est) 
    lower = coefs - (z * SE_est) 
    return coefs, upper, lower 

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

def splitDataClf(X, y, clf, C=0 ,cv=10, prop=0.5): 
    # split the data into two samples 
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=prop)
    # X_train, X_test, y_train, y_test = X, X, y, y 
    
    # split data into two samples with equal number of trial types 
    X_train_S1, X_test_S1, y_train_S1, y_test_S1 = train_test_split(X[0:int(gv.n_trials/2)], y[0:int(gv.n_trials/2)], test_size=prop) 
    X_train_S2, X_test_S2, y_train_S2, y_test_S2 = train_test_split(X[int(gv.n_trials/2):], y[int(gv.n_trials/2):], test_size=prop) 

    X_train = np.vstack( (X_train_S1, X_train_S2))
    X_test = np.vstack( (X_test_S1, X_test_S2))
    
    y_train = np.hstack( (y_train_S1, y_train_S2)) 
    y_test = np.hstack( (y_test_S1, y_test_S2)) 
    
    if gv.standardize: 
        scaler = StandardScaler().fit(X_train) 
        X_train = scaler.transform(X_train) 
        X_test = scaler.transform(X_test)

    # fit logistic lasso on the training set 
    print('fit X_train') 
    # if cv!=0:
    #     clf = gridSearch(clf, X_train, y_train, cv=cv) 
    clf.fit(X_train, y_train) 
    
    print('compute SE') 
    SE_est = get_se(X_test, y_test, clf) 
    
    # perform inference on the test set 
    print('ci X_test') 
    coefs, upper, lower = get_coefs_ci(clf, X_test, SE_est, z=1.96) 

    return coefs, upper, lower

def bootStrapClf(X, y, clf):
    num_cores = -int(1*multiprocessing.cpu_count()/4) 

    def parForBoot(X, y, clf): 
        idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)),
                                  np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 

        X_boot = X[idx_trials] 
        y_boot = y[idx_trials] 

        if gv.standardize:
            scaler = StandardScaler().fit(X) 
            X_boot = scaler.transform(X_boot)

        clf.fit(X_boot, y_boot) 
        return clf.coef_.flatten() 

    coefs_boot = Parallel(n_jobs=num_cores, verbose=True)(delayed(parForBoot)(X, y, clf) for _ in range(gv.n_boot) ) 
    coefs_boot = np.array(coefs_boot) 
    
    coefs = np.mean(coefs_boot, axis=0) 
    upper = np.percentile(coefs_boot, 75, axis=0) 
    lower = np.percentile(coefs_boot, 25, axis=0) 

    return coefs, upper, lower

def gridSearch(loss, X, y, cv=10): 
    num_cores = -int(1*multiprocessing.cpu_count()/4) 
    
    pipe = Pipeline([('scale', StandardScaler()), ('classifier', loss)])
    
    param_grid = [{'classifier': [loss], 'classifier__C' : np.logspace(-4, 4, 100)}] 
    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=False, n_jobs=num_cores) 
    
    best_model = search.fit(X, y) 
    C_cv = best_model.best_estimator_.get_params()['classifier__C'] 

    return best_model.best_estimator_['classifier'] 

def getCoefsTrials(X_trials, C=1e0, penalty='l1', solver='liblinear', cv=10): 
    num_cores = -int(1*multiprocessing.cpu_count()/4) 

    gv.n_boots = int(1e4) 
    
    clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-4, max_iter=int(1e8), fit_intercept=bool(gv.standardize)) 
    # if cv!=0 & C==0: 
    # clf = LogisticRegressionCV(Cs=np.logspace(-1,1,100), solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(gv.standardize), n_jobs=num_cores) 
    
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

            print('trial', n_trial, 'bin', n_bins) 
            
            X = X_S1_S2[:,:,n_bins]
            coefs, upper, lower = splitDataClf(X, y, clf, C=C, cv=cv, prop=0.5) 
            # coefs, upper, lower = bootStrapClf(X, y, clf) 
            
            mean_coefs[n_trial, n_bins] = coefs 
            upper_coefs[n_trial, n_bins] = upper 
            lower_coefs[n_trial, n_bins] = lower 
            
    return mean_coefs, upper_coefs, lower_coefs 
