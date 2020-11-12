from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 
from scipy.spatial import distance

import data.constants as gv 
import data.plotting as pl

from joblib import Parallel, delayed
import multiprocessing

pal = ['r','b','y']

def avg_epochs(X):
    
    X_ED = np.mean(X[:,:,gv.bins_ED-gv.bin_start],axis=-1) 
    X_MD = np.mean(X[:,:,gv.bins_MD-gv.bin_start],axis=-1) 
    X_LD = np.mean(X[:,:,gv.bins_LD-gv.bin_start],axis=-1) 

    if gv.STIM_AND_DELAY:
        X_STIM = np.mean(X[:,:,gv.bins_STIM-gv.bin_start],axis=-1) 
        X_epochs = np.array([X_STIM, X_ED, X_MD, X_LD])
    elif gv.DELAY_ONLY:
        X_epochs = np.array([X_ED, X_MD, X_LD])
        
    X_epochs = np.moveaxis(X_epochs,0,2) 
    return X_epochs 

def angleAllvsTime(alpha):
    ax = plt.figure('angleAllvsTime').add_subplot()
    for n_trial in range(0,len(gv.trials)):
        plt.plot(gv.t_delay, alpha[n_trial], '-o', color=pal[n_trial]) 
        plt.xlabel('time (s)') 
        plt.ylabel('$\\alpha$ (deg)') 
    plt.xlim([gv.t_start,gv.t_LD[-1]]) 
    pl.vlines_delay(ax) 

def angleNDvsTime(alpha):
    ax = plt.figure('angleNDvsTime').add_subplot()
    for n_trial in range(1,len(gv.trials)):
        plt.plot(gv.t_delay, alpha[0]-alpha[n_trial], '-o', color=pal[n_trial]) 
        plt.xlabel('time (s)') 
        plt.ylabel('$\\alpha_{i,ND}$ (deg)') 
    plt.xlim([gv.t_start,gv.t_LD[-1]]) 
    pl.vlines_delay(ax) 

def cosNDvsTime(alpha):
    ax = plt.figure('cosNDvsTime').add_subplot() 
    for n_trial in range(1,len(gv.trials)):
        plt.plot(gv.t_delay, np.cos( (alpha[0]-alpha[n_trial])*np.pi/180.), '-o', color=pal[n_trial]) 
        plt.xlabel('time (s)') 
        plt.ylabel(' cos($\\alpha_{i,ND}$) (deg)') 
    plt.xlim([gv.t_start,gv.t_LD[-1]]) 
    pl.vlines_delay(ax) 
    
def angleEDvsTime(agl, dum=0): 
    ax = plt.figure('angleEDvsTime').add_subplot()

    binED = gv.bins_ED-gv.bin_start
    aglED = np.mean(agl[:,binED], axis=1)

    # average over all trial types
    if dum :
        aglED_avg = np.mean(aglED)
        aglED = np.array([aglED_avg, aglED_avg, aglED_avg]) 
        
    x = gv.t_delay[binED[-1]:-1]
    for n_trial in range(len(gv.trials)):
        y = aglED[n_trial, np.newaxis] - agl[n_trial, binED[-1]:-1]
        plt.plot(x, y, '-o', color=pal[n_trial])
    
    plt.xlabel('time (s)') 
    plt.ylabel('$\\alpha_{i,ED}$ (deg)') 
    pl.vlines_delay(ax) 
    plt.xlim([gv.t_ED[-1],gv.t_LD[-1]])

def cosEDvsTime(agl, dum=0): 
    ax = plt.figure('cosEDvsTime').add_subplot()

    binED = gv.bins_ED-gv.bin_start 
    aglED = np.mean(agl[:,binED], axis=1) 

    # average over all trial types
    if dum :
        aglED_avg = np.mean(aglED)
        aglED = np.array([aglED_avg, aglED_avg, aglED_avg]) 
        
    x = gv.t_delay[binED[-1]:-1] 
    for n_trial in range(len(gv.trials)): 
        y = np.cos( (aglED[n_trial, np.newaxis] - agl[n_trial, binED[-1]:-1]) *np.pi/180.) 
        plt.plot(x, y, '-o', color=pal[n_trial]) 
        
    plt.xlabel('time (s)') 
    plt.ylabel('cos($\\alpha_{i,ED}$)') 
    pl.vlines_delay(ax) 
    plt.xlim([gv.t_ED[-1],gv.t_LD[-1]])

def cosStimVsTime(agl, dum=0): 
    ax = plt.figure('cosStimVsTime').add_subplot()

    binSTIM = gv.bins_STIM-gv.bin_start
    aglSTIM = np.mean(agl[:,binSTIM[-3:-1]], axis=1) 

    # average over all trial types
    if dum :
        aglSTIM_avg = np.mean(aglSTIM)
        aglSTIM = np.array([aglSTIM_avg, aglSTIM_avg, aglSTIM_avg]) 
        
    x = gv.t_stim_delay[binSTIM[-1]:-1] 
    for n_trial in range(len(gv.trials)): 
        y = np.cos( (aglSTIM[n_trial, np.newaxis] - agl[n_trial, binSTIM[-1]:-1]) *np.pi/180.) 
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
    
def bootstrap_clf_par(X, y, clf, dum): 

    if dum==1: 
        idx = np.arange(0, X.shape[0]) 
    else: 
        # idx = np.random.randint(0, X.shape[0], X.shape[0]) 
        idx = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)), np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 
        
    X_sample = X[idx] 
    y_sample = y[idx] 

    # X_sample = StandardScaler().fit_transform(X_sample) 
    scaler = StandardScaler().fit(X) 
    X_sample = scaler.transform(X_sample) 
    
    clf.fit(X_sample, y_sample) 
    coefs_samples = clf.coef_.flatten() 

    return coefs_samples 

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    u = vector / np.linalg.norm(vector) 
    return u 

def angle_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """ 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 

def get_angle(coefs): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """ 

    alp=np.empty(coefs.shape[0]) 
    dum = np.ones(coefs.shape[1])
    for i in np.arange(0, coefs.shape[0]): 
        alpha = angle_between(dum, coefs[i]) 
        alp[i] = alpha*180.0/np.pi 
        
    return alp 

def angleVsTime(X_trials, C=1e0, penalty='l2', solver='liblinear'): 
    
    gv.n_boot = int(1e3) 
    num_cores = multiprocessing.cpu_count() 

    if X_trials.shape[3]!=gv.n_neurons:
        X_trials = X_trials[:,:,:,0:gv.n_components,:]

    # print('n_boot', gv.n_boot)
    clf = LogisticRegression(C=C, solver=solver, penalty=penalty,tol=1e-6, max_iter=int(1e6), fit_intercept=False) 
    # clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage='auto') 
    # clf = LinearDiscriminantAnalysis(tol=1e-6) 

    gv.AVG_EPOCHS = 0 
    gv.trial_size = X_trials.shape[-1]
    
    if gv.AVG_EPOCHS: 
        if gv.STIM_AND_DELAY: 
            gv.trial_size = len(['STIM','ED','MD','LD']) 
        elif gv.DELAY_ONLY: 
            gv.trial_size = len(['ED','MD','LD']) 
    
    print(gv.trial_size) 
    coefs = np.empty( (len(gv.trials), gv.trial_size,  gv.n_boot, X_trials.shape[3]) ) 
    agl_boot = np.empty( (len(gv.trials), gv.n_boot, gv.trial_size) ) 
    
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
            
            coefs_boot = Parallel(n_jobs=num_cores)(delayed(bootstrap_clf_par)(X, y, clf, gv.n_boot) for i in range(gv.n_boot)) 
            
            coefs[n_trial, n_bins] = np.asarray(coefs_boot) 
            
        for boot in range(gv.n_boot): 
            agl_boot[n_trial, boot] = get_angle(coefs[n_trial,:,boot,:]) # bins x coefficients 

    print(agl_boot.shape) 
    mean_agl = np.mean(agl_boot, axis=1) 
    print(mean_agl.shape) 
    
    return mean_agl 
