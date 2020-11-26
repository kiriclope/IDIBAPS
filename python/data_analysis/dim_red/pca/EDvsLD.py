from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 
from scipy.spatial import distance 

import data.constants as gv 
import data.plotting as pl
import data.preprocessing as pp 

from joblib import Parallel, delayed, parallel_backend
import multiprocessing 
    
def grid_search_cv_clf(loss, X, y, cv=10): 
    
    pipe = Pipeline([('scale', StandardScaler()), ('clf', loss)])
    
    param_grid = [{'clf': [loss], 'clf__C' : np.logspace(-4, 4, 1000)}] 
    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=False, n_jobs=gv.num_cores) 
    best_model = search.fit(X, y) 
    
    return best_model

def bootstrap_clf_par(X, y, clf, dum, cv): 

    if dum==1: 
        print('no boot') 
        idx_trials = np.arange(0, X.shape[0]) 
    else:
        #standard bootstrap
        idx_trials = np.random.randint(0, X.shape[0], X.shape[0])
        #block bootstrap
        # idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)),
        #                           np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 
        
    X_sample = X[idx_trials] 
    y_sample = y[idx_trials] 

    # hierarchical bootstrap
    # for trial in idx_trials: 
    #     idx_neurons = np.random.randint(0, X.shape[1], X.shape[1])
    #     X_sample[trial] = X[trial, idx_neurons] 

    # scaler = StandardScaler().fit(X) 
    # X_sample = scaler.transform(X_sample) 

    if cv==0:
        X_sample = StandardScaler().fit_transform(X_sample) 
        clf.fit(X_sample, y_sample) 
        coefs_samples = clf.coef_.flatten() 
    else:
        best_model = grid_search_cv_clf(clf, X_sample, y_sample, cv=cv)        
        coefs_samples = best_model.best_estimator_['clf'].coef_.flatten() 
        # C = best_model.best_estimator_['clf__C']
        
    return coefs_samples

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    u = vector / (np.linalg.norm(vector) + gv.eps)
    return u 

def angle_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """ 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 

def cos_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """ 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)

def get_cos(coefs): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """ 
    cos_alp=[] 
    for j in np.arange(0, coefs.shape[0]):  
        cos_alp.append( cos_between(coefs[0], coefs[j]) )         
    return cos_alp

def bootCoefs(X_proj, C=1e0, penalty='l2', solver='liblinear', loss='squared_hinge', cv=10, l1_ratio=None): 

    gv.n_boot = int(1e3) 
    gv.num_cores = int(1*multiprocessing.cpu_count()/2) 

    clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8),
                             fit_intercept=bool(not gv.standardize), n_jobs=gv.num_cores, l1_ratio=l1_ratio) 

    # clf = LogisticRegressionCV(solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), 
    #                            fit_intercept=bool(not gv.standardize), n_jobs=2) 
    
    # clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', dual=False, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(not gv.standardize) ) 
    # clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage='auto') 
    gv.clf_name = clf.__class__.__name__ 
    if 'CV' in 'gv.clf_name': 
        cv=0
    
    print('bootstrap samples', gv.n_boot, 'clf', gv.clf_name) 
    
    gv.IF_PCA = 0
    if X_proj.shape[3]!=gv.n_neurons: 
        X_proj = X_proj[:,:,:,0:gv.n_components,:] 
        gv.IF_PCA = 1 

    if gv.EDvsLD: 
        gv.epochs = ['ED', 'MD', 'LD'] 
        print('angle btw ED and other epochs')
    else:
        gv.epochs = ['Stim', 'ED', 'MD', 'LD'] 
        print('angle btw STIM and other epochs')
        
    coefs = np.empty((len(gv.trials), len(gv.epochs), gv.n_boot, X_proj.shape[3])) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        X_S1 = X_proj[n_trial,0] 
        X_S2 = X_proj[n_trial,1]
        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        
        X_S1_S2 = pp.avg_epochs(X_S1_S2) 
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
        
        for n_epochs in range(X_S1_S2.shape[2]):
            X = X_S1_S2[:,:,n_epochs] 
            boot_coefs = Parallel(n_jobs=gv.num_cores, verbose=True)(delayed(bootstrap_clf_par)(X, y, clf, gv.n_boot, cv) for _ in range(gv.n_boot)) 
            coefs[n_trial, n_epochs] = np.array(boot_coefs) 

    return coefs 

def cosVsEpochs(coefs):

    cos_boot = np.empty( (len(gv.trials), gv.n_boot, len(gv.epochs) ) ) 
    
    mean = np.empty((len(gv.trials), len(gv.epochs)))
    upper = np.empty( (len(gv.trials), len(gv.epochs)) )
    lower = np.empty((len(gv.trials), len(gv.epochs)))
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        for boot in range(coefs.shape[2]): 
            cos_alp = get_cos(coefs[n_trial,:,boot,:]) # bins x neurons 
            cos_boot[n_trial, boot] = np.array(cos_alp)
        
        mean[n_trial] = np.mean(cos_boot[n_trial], axis=0) 
        lower[n_trial] = mean[n_trial] - np.percentile(cos_boot[n_trial], 25, axis=0) 
        upper[n_trial] = np.percentile(cos_boot[n_trial], 75, axis=0) - mean[n_trial]
        
        print('trial', gv.trial, 'cos', mean[n_trial], 'lower', lower[n_trial], 'upper', upper[n_trial]) 

    return mean, lower, upper, cos_boot 

def get_p_values(cos_boot):

    p_values = np.empty( ( cos_boot.shape[0]-1, cos_boot.shape[2]-1) ) 
    for n_trial in range(1, cos_boot.shape[0]): # trials 
        for n_epoch in range(1, cos_boot.shape[2]): # epochs 
            sample_1  = cos_boot[0,:,n_epoch] 
            sample_2  = cos_boot[n_trial,:,n_epoch] 
            t_score, p_value = stats.ttest_ind(sample_1, sample_2, equal_var=False) 
            # if t_score>0:
            #     p_value = p_value/2
            # else:
            #     p_value = 1-p_value/2
            p_values[n_trial-1, n_epoch-1] = p_value 

    return p_values

def add_pvalue(p_values): 
    cols = 0.25*np.arange(len(gv.trials)) 
    high = [0.9, 0.8] 
    print(cols)
    for n_cols in range(1, len(cols)):        
        for n_epoch in range(p_values.shape[1]):
            
            plt.plot( [n_epoch + cols[0], n_epoch + cols[n_cols]] , [high[n_cols-1], high[n_cols-1]] , lw=.8, c='k') 
            
            if p_values[n_cols-1,n_epoch]<=.001: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "***", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]<=.01: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "**", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]<=.05: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "*", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]>.05: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1], "ns", ha='center', va='bottom', color='k', fontsize=6) 

def create_fig_dir(C=1, penalty='l1', solver='liblinear', cv=0, loss='lsqr'): 
    
    pl.figDir() 
    
    if 'LogisticRegression' in gv.clf_name:
        clf_param = '/C_%.3f_penalty_%s_solver_%s/' % (C, penalty, solver)
    elif gv.clf_name in 'LinearSVC':
        clf_param = '/C_%.3f_penalty_%s_loss_%s/' % (C, penalty, loss)
    elif gv.clf_name in 'LinearDiscriminantAnalysis':
        clf_param = '/shrinkage_auto_solver_lsqr/'

    if cv!=0: 
        gv.figdir = gv.figdir + '/gridsearchCV_%d' % cv 

    gv.figdir = gv.figdir +'/'+ gv.clf_name + clf_param 
    
    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)

def corrVsTime(coefs): 
    
    corr = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    lower = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) ) 
    upper = np.empty( (len(gv.trials), coefs.shape[1], coefs.shape[1]) )
    
    corr_boot = np.empty( (len(gv.trials), gv.n_boot, coefs.shape[1], coefs.shape[1]) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        
        for boot in range(gv.n_boot): 
            corr_boot[n_trial, boot] = np.corrcoef(coefs[n_trial,:,boot,:]) # bins x coefficients 
            
        corr[n_trial] = np.mean(corr_boot[n_trial], axis=0) 
        lower[n_trial] = corr[n_trial] - np.percentile(corr_boot[n_trial], 25, axis=0) 
        upper[n_trial] = np.percentile(corr_boot[n_trial], 75, axis=0) - corr[n_trial]
        
    return corr, lower, upper 

def EDvsLD(X_proj, C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=10, l1_ratio=None):

    coefs = bootCoefs(X_proj, C, penalty, solver, loss, cv, l1_ratio) 
    mean, lower, upper, cos_boot = cosVsEpochs(coefs) 
    p_values = get_p_values(cos_boot) 
    
    create_fig_dir(C=C, penalty=penalty, solver=solver, cv=cv, loss=loss) 
    
    pl.barCosAlp(mean, lower, upper) 
    add_pvalue(p_values)
    plt.ylim([0, 1]) 
    
    figtitle = '%s_%s_cos_alpha' % (gv.mouse, gv.session) 
    pl.save_fig(figtitle) 

    return coefs 

