from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')
sys.path.insert(2, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis/pkg/merged/python') 

from scipy.spatial import distance 

import data.constants as gv 
import data.plotting as pl 
import data.preprocessing as pp 

from joblib import Parallel, delayed 
import multiprocessing 

from joblib import parallel_backend

pal = ['r','b','y']
    
def angleEDvsTime(alpha, dum=0): 
    ax = plt.figure('angleEDvsTime').add_subplot()

    binED = gv.bins_ED-gv.bin_start
    alphaED = np.mean(alpha[:,binED], axis=1)

    # average over all trial types
    if dum :
        alphaED_avg = np.mean(alphaED)
        alphaED = np.array([alphaED_avg, alphaED_avg, alphaED_avg]) 
        
    x = gv.time[binED[-1]:-1]
    for n_trial in range(len(gv.trials)):
        y = alphaED[n_trial, np.newaxis] - alpha[n_trial, binED[-1]:-1]
        plt.plot(x, y, '-o', color=pal[n_trial])

    plt.xlabel('time (s)') 
    plt.ylabel('$\\alpha_{i,ED}$ (deg)') 
    pl.vlines_delay(ax) 
    plt.xlim([gv.t_ED[-1],gv.t_LD[-1]])

def cosEDvsTime(alpha, q1_alpha, q3_alpha, sigma=1):
    figname = '%s_%s_cosEDvsTime' % (gv.mouse, gv.session) 
    ax = plt.figure(figname).add_subplot() 
    
    x = gv.time-np.array(2) 
    for n_trial in range(len(gv.trials)):

        y = gaussian_filter1d( np.cos( alpha[n_trial] ), sigma )
        plt.plot(x, y, '-', color=pal[n_trial]) 
        q1 = gaussian_filter1d( np.cos( q1_alpha[n_trial] ) , sigma) 
        q3 = gaussian_filter1d( np.cos( q3_alpha[n_trial] ), sigma) 
        ax.fill_between(x, q1, q3 , color=pal[n_trial], alpha=.1) 
        
    plt.xlabel('time (s)') 
    plt.ylabel('cos($\\alpha$)') 
    pl.vlines_delay(ax) 
    plt.xlim([-2,gv.t_LD[-1]+1])
    plt.ylim([-.2,1])
    ax.axhline(0, color='k', ls='--')
    ax.axvline(0, color='k', ls='-')
    figdir = pl.figDir('pca') 
    pl.save_fig(figname)         

def grid_search_cv_clf(loss, X, y, cv=10): 
    num_cores = -int(1*multiprocessing.cpu_count()/4) 
    
    pipe = Pipeline([('scale', StandardScaler()), ('classifier', loss)])
    
    param_grid = [{'classifier': [loss], 'classifier__C' : np.logspace(-3, 3, 100)}] 
    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=True, n_jobs=num_cores, refit=True) 
    
    # best_model = search.fit(X, y) 
    # C_cv = best_model.best_estimator_.get_params()['classifier__C'] 
    
    return best_model.best_estimator_['classifier'].coef_ 

def bootstrap_clf_par(X, y, clf, dum, cv): 

    if dum==1: 
        idx_trials = np.arange(0, X.shape[0]) 
    else: 
        idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)),
                                  np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 
        # idx_trials = np.arange(0, X.shape[0]) 
    
    X_sample = X[idx_trials] 
    y_sample = y[idx_trials]
    
    # for trial in idx_trials: 
    #     idx_neurons = np.random.randint(0, X.shape[1], X.shape[1]) 
    #     X_sample[trial] = X[trial, idx_neurons] 

    if gv.standardize:
        X_sample = StandardScaler().fit_transform(X_sample)
        # scaler = StandardScaler().fit(X) 
        # X_sample = scaler.transform(X_sample) 
    
    clf.fit(X_sample, y_sample) 
    coefs_samples = clf.coef_.flatten() 
 
    # coefs_samples = grid_search_cv_clf(clf, X_sample, y_sample, cv=cv).flatten() 
    
    return coefs_samples 

def parforbinsboots(X_S1_S2, y, clf, n_bins, dum, cv): 
    X = X_S1_S2[:,:,n_bins] 

    # h = np.random.uniform(-10,10,(X.shape[0],X.shape[1]))

    if dum==1: 
        idx_trials = np.arange(0, X.shape[0]) 
    else: 
        idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)), 
                           np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 
    
    X_sample = X[idx_trials] # + h[idx_trials] 
    y_sample = y[idx_trials] 
    
    # for trial in idx_trials: 
    #     idx_neurons = np.random.randint(0, X.shape[1], X.shape[1]) 
    #     X_sample[trial] = X[trial, idx_neurons] 
        
    if gv.standardize: 
        X_sample = StandardScaler().fit_transform(X_sample) 
        # scaler = StandardScaler().fit(X) 
        # X_sample = scaler.transform(X_sample) 
    
    clf.fit(X_sample, y_sample) 
    coefs_samples = clf.coef_.flatten() 

    # coefs_samples = grid_search_cv_clf(clf, X_sample, y_sample, cv=cv).flatten()
    
    return coefs_samples 

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    u = vector / ( np.linalg.norm(vector) + gv.eps ) 
    return u 

def angle_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """ 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.arccos( np.clip( np.dot(v1_u, v2_u), -1.0, 1.0) ) 

def get_angle(coefs, v0): 

    alpha=np.empty(coefs.shape[0])
    for i in np.arange(0, coefs.shape[0]):     
        alpha[i] = angle_between(v0, coefs[i]) 
        
    return alpha 

def parforangles(coefs, n_bins, n_boot, v0): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """ 
    alpha = angle_between(v0, coefs[n_bins, n_boot])*180.0/np.pi     
    return alpha

def bootCoefs(X_trials, C=1e0, penalty='l2', solver='liblinear', cv=10): 
    
    gv.n_boot = int(1e3) 
    num_cores = -int(multiprocessing.cpu_count()/2) 

    gv.IF_PCA=0
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
        gv.IF_PCA=1 
    
    # print('n_boot', gv.n_boot) 
    clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8),
                             fit_intercept=bool(not gv.standardize)) 
    # clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', tol=1e-6, max_iter=int(1e8), fit_intercept=bool(not gv.standardize)) 
    # clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage='auto') 
    
    gv.AVG_EPOCHS = 1 
    gv.trial_size = X_trials.shape[-1] 

    pl.figDir() 
    gv.clf_name = clf.__class__.__name__ 
    clf_param = '/C_%.3f_penalty_%s_solver_%s/' % (C, penalty, solver) 
    gv.figdir = gv.figdir +'/'+ gv.clf_name + clf_param 
    
    if gv.AVG_EPOCHS: 
        gv.figdir = gv.figdir + '/avg_epochs'

    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)
    
    if gv.AVG_EPOCHS: 
        # if gv.STIM_AND_DELAY: 
        #     gv.trial_size = len(['STIM','ED','MD','LD']) 
        # if gv.DELAY_ONLY: 
        gv.trial_size = len(gv.epochs) 
    else:
        gv.trial_size = len(gv.bins_delay) 
        X_trials = X_trials[:,:,:,:,gv.bins_delay] 
        gv.time =  gv.t_delay 
        
    coefs = np.empty( (len(gv.trials), gv.trial_size,  gv.n_boot, X_trials.shape[3]) ) 
    
    # mean_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    # q1_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    # q3_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
    
        X_S1 = X_trials[n_trial,0] 
        X_S2 = X_trials[n_trial,1]

        if gv.SELECTIVE:
            if 'ND' in gv.trial: 
                X_S1, X_S2, idx = pp.selectiveNeurons(X_S1, X_S2, .1) 
                coefs = np.delete(coefs, idx, axis=-1) 
            else:
                X_S1 = np.delete(X_S1, idx, axis=1) 
                X_S2 = np.delete(X_S2, idx, axis=1) 

        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        
        if gv.AVG_EPOCHS: 
            X_S1_S2 = pp.avg_epochs(X_S1_S2) 

        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
            
        print('X_S1_S2', X_S1_S2.shape, 'y', y.shape) 

        for n_bins in range(gv.trial_size): 
            X = X_S1_S2[:,:,n_bins] 
            
            # for boot in range(gv.n_boot):
            #     coefs_boot =  bootstrap_clf_par(X, y, clf, gv.n_boot, cv) 
            #     coefs[n_trial, n_bins, boot] = np.asarray(coefs_boot) 
                
            coefs_boot = Parallel(n_jobs=num_cores, verbose=True)(delayed(bootstrap_clf_par)(X, y, clf, gv.n_boot, cv) for _ in range(gv.n_boot)) 
            coefs[n_trial, n_bins] = np.asarray(coefs_boot)            
        
        # coefs_boot = Parallel(n_jobs=num_cores, verbose=True)(delayed(parforbinsboots)(X_S1_S2, y, clf, n_bins, gv.n_boot, cv) 
        #                                                       for _ in range(gv.n_boot) 
        #                                                       for n_bins in range(gv.trial_size) ) 
        
        # coefs[n_trial] = np.asarray(coefs_boot).reshape(gv.trial_size, gv.n_boot, X_S1_S2.shape[1]) 
        
        # mean_coefs[n_trial] = np.mean(coefs[n_trial], axis=1) 
        # q1_coefs[n_trial] = np.percentile(coefs[n_trial], 25, axis=1) 
        # q3_coefs[n_trial] = np.percentile(coefs[n_trial], 75, axis=1) 

    return coefs

def angleVsTime(coefs): 

    mean_coefs = np.mean(coefs, axis=2) 
    q1_coefs = np.percentile(coefs, 25, axis=2) 
    q3_coefs = np.percentile(coefs, 75, axis=2) 

    gv.trial_size = coefs.shape[1]
    
    alpha = np.empty( (len(gv.trials), gv.trial_size) ) 
    q1_alpha = np.empty( (len(gv.trials), gv.trial_size) ) 
    q3_alpha = np.empty( (len(gv.trials), gv.trial_size) )
    
    alpha_boot = np.empty( (len(gv.trials), gv.n_boot, gv.trial_size) ) 

    for n_trial, gv.trial in enumerate(gv.trials): 
        
        alpha[n_trial] = get_angle(mean_coefs[n_trial], mean_coefs[n_trial,0]) 
        q1_alpha[n_trial] = get_angle(mean_coefs[n_trial], mean_coefs[n_trial,0]) 
        q3_alpha[n_trial] = get_angle(mean_coefs[n_trial], mean_coefs[n_trial,0]) 
        
        # for boot in range(gv.n_boot):
        #     if not gv.AVG_EPOCHS: 
        #         if gv.DELAY_ONLY: 
        #             v0 = np.mean(coefs[n_trial, gv.bins_ED[:]-gv.bin_start, boot], axis=0) 
        #         else: 
        #             # v0 = np.mean( np.mean(coefs[n_trial, gv.bins_STIM[-3:]-gv.bin_start, :], axis=0), axis=0) 
        #             # v0 = np.mean(coefs[n_trial, np.hstack( (gv.bins_STIM[-1]-gv.bin_start, gv.bins_ED[0]-gv.bin_start) ), boot], axis=0) 
        #             v0 = np.mean(coefs[n_trial, gv.bins_STIM[-2:]-gv.bin_start, boot], axis=0) 
        #             # v0 = coefs[n_trial, gv.bins_ED[-1]-gv.bin_start, boot] 
        #     else: 
        #         v0 = coefs[n_trial, 0, boot] 
                
        #     alpha_boot[n_trial, boot] = get_angle(coefs[n_trial,:,boot,:], v0) # bins x coefficients 
            
        # alpha[n_trial] = np.mean(alpha_boot[n_trial], axis=0) 
        # q1_alpha[n_trial] = np.percentile(alpha_boot[n_trial], 25, axis=0) 
        # q3_alpha[n_trial] = np.percentile(alpha_boot[n_trial], 75, axis=0) 

        # # std_alpha = np.std(alpha_boot[n_trial], axis=0) 
        # # q1_alpha[n_trial] = alpha[n_trial] - 1.96 * std_alpha/np.sqrt(gv.n_boot) 
        # # q3_alpha[n_trial] = alpha[n_trial] + 1.96 * std_alpha/np.sqrt(gv.n_boot) 
       
    return alpha, q1_alpha, q3_alpha 

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

def barCorr(corr, lower, upper):

    labels = np.arange(len(gv.epochs)-1) 
    width=0.25
    
    figtitle = '%s_%s_corr_beta' % (gv.mouse, gv.session)
    ax = plt.figure(figtitle).add_subplot() 

    for n_trial, trial in enumerate(gv.trials):
        values = corr[n_trial][0][1:]
        error = np.array([ lower[n_trial][0][1:], upper[n_trial][0][1:] ] ) 
        plt.bar(labels + n_trial*width, values , yerr=error, color = pal[n_trial], width = width) 
            
    plt.xticks([i + width for i in range(len(gv.epochs)-1)], gv.epochs[1:]) 

    plt.xlabel('Epochs') 
    plt.ylabel('Corr($\\beta_i $,$\\beta_j $)') 
    plt.ylim([0,1])
