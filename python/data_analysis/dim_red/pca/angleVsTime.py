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

def avg_epochs(X):
    
    X_STIM = np.mean(X[:,:,gv.bins_STIM[:]-gv.bin_start],axis=2) 
    X_ED = np.mean(X[:,:,gv.bins_ED[:]-gv.bin_start],axis=2) 
    X_MD = np.mean(X[:,:,gv.bins_MD[:]-gv.bin_start],axis=2) 
    X_LD = np.mean(X[:,:,gv.bins_LD[:]-gv.bin_start],axis=2) 

    # if gv.STIM_AND_DELAY:
    #     X_STIM = np.mean(X[:,:,gv.bins_STIM-gv.bin_start],axis=-1) 
    #     X_epochs = np.array([X_STIM, X_ED, X_MD, X_LD])
    # elif gv.DELAY_ONLY:
    X_epochs = np.array([X_STIM, X_ED, X_MD, X_LD])
        
    X_epochs = np.moveaxis(X_epochs,0,2) 
    return X_epochs 

def angleAllVsTime(alpha):
    ax = plt.figure('angleAllvsTime').add_subplot() 
    for n_trial in range(0,len(gv.trials)): 
        plt.plot(gv.time, alpha[n_trial], '-o', color=pal[n_trial]) 
        plt.xlabel('time (s)') 
        plt.ylabel('$\\alpha$ (deg)') 
    plt.xlim([gv.t_start,gv.t_LD[-1]]) 
    pl.vlines_delay(ax) 

def angleNDvsTime(alpha):
    ax = plt.figure('angleNDvsTime').add_subplot()
    for n_trial in range(1,len(gv.trials)):
        plt.plot(gv.time, alpha[0]-alpha[n_trial], '-o', color=pal[n_trial]) 
        plt.xlabel('time (s)') 
        plt.ylabel('$\\alpha_{i,ND}$ (deg)') 
    plt.xlim([gv.t_start,gv.t_LD[-1]]) 
    pl.vlines_delay(ax) 

def cosNDvsTime(alpha):
    ax = plt.figure('cosNDvsTime').add_subplot() 
    for n_trial in range(1,len(gv.trials)):
        plt.plot(gv.time, np.cos( (alpha[0]-alpha[n_trial])*np.pi/180.), '-o', color=pal[n_trial]) 
        plt.xlabel('time (s)') 
        plt.ylabel(' cos($\\alpha_{i,ND}$) (deg)') 
    plt.xlim([gv.t_start,gv.t_LD[-1]]) 
    pl.vlines_delay(ax) 
    
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

def cosStimVsTime(alpha, dum=0): 
    ax = plt.figure('cosStimVsTime').add_subplot()

    binSTIM = gv.bins_STIM-gv.bin_start
    alphaSTIM = np.mean(alpha[:,binSTIM[-3:-1]], axis=1) 

    # average over all trial types
    if dum :
        alphaSTIM_avg = np.mean(alphaSTIM)
        alphaSTIM = np.array([alphaSTIM_avg, alphaSTIM_avg, alphaSTIM_avg]) 
        
    x = gv.t_stim_delay[binSTIM[-1]:-1] 
    for n_trial in range(len(gv.trials)): 
        y = np.cos( (alphaSTIM[n_trial, np.newaxis] - alpha[n_trial, binSTIM[-1]:-1]) *np.pi/180.) 
        plt.plot(x, y, '-o', color=pal[n_trial]) 
        
    plt.xlabel('time (s)') 
    plt.ylabel('cos($\\alpha_{i,ED}$)') 
    pl.vlines_delay(ax) 
    plt.xlim([gv.t_ED[0],gv.t_LD[-1]]) 

def cosVsEpochs(alpha, dum=0): 
    ax = plt.figure('cosStimVsEpochs').add_subplot()
    
    alphaSTIM = alpha[:,0]
    # average over all trial types
    if dum : 
        alpha_avg = np.mean(alpha[:,0], axis=0) 
        alphaSTIM = alpha_avg*np.ones(alpha.shape[1])

    q = np.zeros(alpha.shape[1])
    for n_trial, gv.trial in enumerate(gv.trials): 
        y = np.cos( (alphaSTIM[n_trial] - alpha[n_trial, :]) *np.pi/180.)
        print(y)
        # pl.plot_cosine_bars(y, [], q, q) 

    plt.ylim([0,1.1]) 

def grid_search_cv_clf(loss, X, y, cv=10): 
    num_cores = -int(1*multiprocessing.cpu_count()/4) 
    
    pipe = Pipeline([('scale', StandardScaler()), ('classifier', loss)])
    
    param_grid = [{'classifier': [loss], 'classifier__C' : np.logspace(-3, 3, 1000)}] 
    search = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=True, n_jobs=num_cores) 
    
    best_model = search.fit(X, y) 
    C_cv = best_model.best_estimator_.get_params()['classifier__C'] 

    return best_model.best_estimator_['classifier'].coef_

def bootstrap_clf_par(X, y, clf, dum, cv): 

    if dum==1: 
        idx_trials = np.arange(0, X.shape[0]) 
    else: 
        idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)), np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 
        # idx_trials = np.arange(0, X.shape[0]) 
    
    X_sample = X[idx_trials] 
    y_sample = y[idx_trials]
    
    # for trial in idx_trials: 
    #     idx_neurons = np.random.randint(0, X.shape[1], X.shape[1]) 
    #     X_sample[trial] = X[trial, idx_neurons] 

    # X_sample = StandardScaler().fit_transform(X_sample)
    if gv.standardize:
        scaler = StandardScaler().fit(X.T) 
        X_sample = scaler.transform(X_sample.T).T 
    
    # clf.fit(X_sample, y_sample) 
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
        
    # # X_sample = StandardScaler().fit_transform(X_sample) 
    # if gv.standardize: 
    #     scaler = StandardScaler().fit(X.T) 
    #     X_sample = scaler.transform(X_sample.T).T 
    
    # clf.fit(X_sample, y_sample) 
    # coefs_samples = clf.coef_.flatten()  

    coefs_samples = grid_search_cv_clf(clf, X_sample, y_sample, cv=cv).flatten()
    
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
    
    gv.n_boot = int(1e0) 
    num_cores = -int(1*multiprocessing.cpu_count()/4) 
    
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
        
    # print('n_boot', gv.n_boot) 
    clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(gv.standardize)) 
    # clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', dual=False, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(not gv.standardize)) 
    # clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage='auto') 
    
    gv.AVG_EPOCHS = 1
    gv.trial_size = X_trials.shape[-1] 
    
    if gv.AVG_EPOCHS: 
        # if gv.STIM_AND_DELAY: 
        #     gv.trial_size = len(['STIM','ED','MD','LD']) 
        # if gv.DELAY_ONLY: 
        gv.trial_size = len(['STIM','ED','MD','LD']) 
    # else:
    #     gv.trial_size = len(gv.bins_delay) 
    #     X_trials = X_trials[:,:,:,:,gv.bins_delay] 
    #     gv.time =  gv.t_delay 
        
    coefs = np.empty( (len(gv.trials), gv.trial_size,  gv.n_boot, X_trials.shape[3]) )
    
    # mean_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    # q1_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    # q3_coefs = np.empty( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
        
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
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            model = grid_search_cv_clf(clf, X_train, y_train, clf, 1, cv) 

            for boot in range(gv.n_boot):
                # coefs_boot =  bootstrap_clf_par(X_test, y_test, clf, gv.n_boot, cv) 
                coefs_boot =  bootstrap_clf_par(X_test, y_test, model, gv.n_boot, cv) 
                # print(coefs_boot.shape)
                coefs[n_trial, n_bins, boot] = np.asarray(coefs_boot) 

            # coefs_boot = Parallel(n_jobs=num_cores)(delayed(bootstrap_clf_par)(X, y, clf, gv.n_boot, cv) for i in range(gv.n_boot)) 
            # coefs[n_trial, n_bins] = np.asarray(coefs_boot) 
        
        # coefs_boot = Parallel(n_jobs=num_cores, verbose=True)(delayed(parforbinsboots)(X_S1_S2, y, clf, n_bins, gv.n_boot, cv) 
        #                                                       for n_bins in range(gv.trial_size) 
        #                                                       for _ in range(gv.n_boot) ) 
        
        # coefs[n_trial] = np.asarray(coefs_boot).reshape(gv.trial_size, gv.n_boot, X_S1_S2.shape[1]) 
        
        # mean_coefs[n_trial] = np.mean(coefs[n_trial], axis=1) 
        # q1_coefs[n_trial] = np.percentile(coefs[n_trial], 25, axis=1) 
        # q3_coefs[n_trial] = np.percentile(coefs[n_trial], 75, axis=1) 

    return coefs

def angleVsTime(coefs): 

    # mean_coefs = np.mean(coefs, axis=2) 
    # q1_coefs = np.percentile(coefs, 25, axis=2) 
    # q3_coefs = np.percentile(coefs, 75, axis=2) 

    gv.trial_size = coefs.shape[1]
    
    alpha = np.empty( (len(gv.trials), gv.trial_size) ) 
    q1_alpha = np.empty( (len(gv.trials), gv.trial_size) ) 
    q3_alpha = np.empty( (len(gv.trials), gv.trial_size) )
    
    alpha_boot = np.empty( (len(gv.trials), gv.n_boot, gv.trial_size) ) 

    for n_trial, gv.trial in enumerate(gv.trials): 
        
        # alpha[n_trial] = get_angle(mean_coefs[n_trial], np.mean( mean_coefs[n_trial, gv.bins_STIM[-2:-1] ], axis=0) ) 
        # q1_alpha[n_trial] = get_angle(q1_coefs[n_trial], np.mean( q1_coefs[n_trial, gv.bins_STIM[-2:-1] ], axis=0) ) 
        # q3_alpha[n_trial] = get_angle(q3_coefs[n_trial], np.mean( q3_coefs[n_trial, gv.bins_STIM[-2:-1] ], axis=0) ) 
        
        for boot in range(gv.n_boot):
            if not gv.AVG_EPOCHS: 
                if gv.DELAY_ONLY: 
                    v0 = np.mean(coefs[n_trial, gv.bins_ED[:]-gv.bin_start, boot], axis=0) 
                else: 
                    # v0 = np.mean( np.mean(coefs[n_trial, gv.bins_STIM[-3:]-gv.bin_start, :], axis=0), axis=0) 
                    # v0 = np.mean(coefs[n_trial, np.hstack( (gv.bins_STIM[-1]-gv.bin_start, gv.bins_ED[0]-gv.bin_start) ), boot], axis=0) 
                    v0 = np.mean(coefs[n_trial, gv.bins_STIM[-2:]-gv.bin_start, boot], axis=0) 
                    # v0 = coefs[n_trial, gv.bins_ED[-1]-gv.bin_start, boot] 
            else: 
                v0 = coefs[n_trial, 0, boot] 
                
            alpha_boot[n_trial, boot] = get_angle(coefs[n_trial,:,boot,:], v0) # bins x coefficients 
            
        alpha[n_trial] = np.mean(alpha_boot[n_trial], axis=0) 
        q1_alpha[n_trial] = np.percentile(alpha_boot[n_trial], 25, axis=0) 
        q3_alpha[n_trial] = np.percentile(alpha_boot[n_trial], 75, axis=0) 

        # # std_alpha = np.std(alpha_boot[n_trial], axis=0) 
        # # q1_alpha[n_trial] = alpha[n_trial] - 1.96 * std_alpha/np.sqrt(gv.n_boot) 
        # # q3_alpha[n_trial] = alpha[n_trial] + 1.96 * std_alpha/np.sqrt(gv.n_boot) 
       
    return alpha, q1_alpha, q3_alpha 

def corrVsTime(coefs, penalty): 
    
    corr = np.empty( (len(gv.trials), gv.trial_size, gv.trial_size) ) 
    corr_boot = np.empty( (len(gv.trials), gv.n_boot, gv.trial_size, gv.trial_size) ) 

    for n_trial, gv.trial in enumerate(gv.trials): 
        
        for boot in range(gv.n_boot):
            if not gv.AVG_EPOCHS:
                if gv.DELAY_ONLY:
                    v0 = np.mean(coefs[n_trial, gv.bins_ED[:]-gv.bin_start, boot], axis=0) 
                else: 
                    # v0 = np.mean( np.mean(coefs[n_trial, gv.bins_STIM[-3:-1]-gv.bin_start, :], axis=0), axis=0) 
                    # v0 = np.mean(coefs[n_trial, gv.bins_STIM[-2:-1]-gv.bin_start, boot], axis=0) 
                    v0 = coefs[n_trial, gv.bins_STIM[-1]-gv.bin_start, boot] 
            else: 
                v0 = coefs[n_trial, 0, boot] 
                
            corr_boot[n_trial, boot] = np.corrcoef(coefs[n_trial,:,boot,:]) # bins x coefficients 
        
        corr[n_trial] = np.mean(corr_boot[n_trial], axis=0)

        if gv.AVG_EPOCHS:

            figtitle = '%s_session_%s_trial_%s_corrEpochs_%s_penalty' % (gv.mouse,gv.session,gv.trial,penalty)        
            ax = plt.figure(figtitle).add_subplot() 
        
            im = ax.imshow(corr[n_trial], cmap='jet', vmin=0.0, vmax=1.0, origin='lower')
            
            labels = ['STIM','ED','MD','LD']
            xticks = np.arange(0,len(labels)) 
            yticks = np.arange(0,len(labels))
        
            ax.set_xticks(xticks) ; 
            ax.set_xticklabels(labels) ; 

            ax.set_yticks(yticks) ; 
            ax.set_yticklabels(labels) ; 

        else:
            figtitle = '%s_session_%s_trial_%s_corr_%s_penalty' % (gv.mouse,gv.session,gv.trial,penalty) 
            ax = plt.figure(figtitle).add_subplot() 
            
            im = ax.imshow(corr[n_trial], cmap='jet', origin='lower', vmin=0, vmax=1, extent = [-2 , gv.duration-2, -2 , gv.duration-2])

            pl.vlines_delay(ax) 
            pl.hlines_delay(ax) 
        
            plt.xlim([gv.t_delay[0]-2, gv.t_delay[-1]-2]); 
            plt.ylim([gv.t_delay[0]-2, gv.t_delay[-1]-2]);
            
        ax.set_title(gv.trial) 
        ax.set_xlabel('Time (s)') 
        ax.set_ylabel('Time (s)') 
        ax.grid(False) 
        cbar = plt.colorbar(im, ax=ax) 
        cbar.set_label('corr', rotation=90) 
            
        figdir = pl.figDir('pca') 
        pl.save_fig(figtitle) 

    return corr
