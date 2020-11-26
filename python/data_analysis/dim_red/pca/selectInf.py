from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')

import data.constants as gv 
import data.plotting as pl 
import data.preprocessing as pp 

from selectinf.algorithms import lasso, cv
import regreg.api as rr
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 

pal = ['r','b','y']

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
    
def selectInfCrossVal(X, y, K):

    loss = rr.glm.logistic(X,y) 
    # lam_seq = np.exp(np.linspace(np.log(1.e-6), np.log(1), 100)) * np.fabs(np.dot(X.T,y)).max() 
    lam_seq = np.logspace(-2, 2, 100) 
    
    folds = np.arange(X.shape[0]) % K 
    # folds = K 
    CV_compute = cv.CV(loss, folds, lam_seq) 
    lam_CV, CV_val, SD_val, lam_CV_randomized, CV_val_randomized, SD_val_randomized = CV_compute.choose_lambda_CVr() 

    minimum_CV = np.min(CV_val)
    lam_idx = list(lam_seq).index(lam_CV)
    SD_min = SD_val[lam_idx]
    lam_1SD = lam_seq[max([i for i in range(lam_seq.shape[0]) if CV_val[i] <= minimum_CV + SD_min])]
    
    print(lam_1SD, lam_CV, lam_CV_randomized) 
    return lam_1SD

def selectInfLogistic(X, y, C):

    L = lasso.lasso.logistic(X, y, C) 
    L.fit()

    Cst = L.constraints
        
    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L.onestep_estimator),
        L.constraints.offset)
    
    df = L.summary(compute_intervals=True)
    df = df.reset_index(drop=True)
    
    variable = df['variable']
    beta = df['lasso'] 
    lower = df['lower_confidence'] 
    upper = df['upper_confidence'] 
    # p_value = df['pval'] 

    return variable, beta, lower, upper

def avg_epochs(X):

    if not gv.ED_MD_LD:
        X_ED = np.mean(X[:,:,gv.bins_ED[:]-gv.bin_start],axis=2) 
        X_MD = np.mean(X[:,:,gv.bins_MD[:]-gv.bin_start],axis=2) 
        X_LD = np.mean(X[:,:,gv.bins_LD[:]-gv.bin_start],axis=2) 
    else:
        X_ED = np.mean(X[:,:,0:len(gv.bins_ED)],axis=-1) 
        X_MD = np.mean(X[:,:,len(gv.bins_ED):len(gv.bins_ED)+len(gv.bins_MD)],axis=-1) 
        X_LD = np.mean(X[:,:,len(gv.bins_ED)+len(gv.bins_MD):len(gv.bins_ED)+len(gv.bins_MD)+len(gv.bins_LD)],axis=-1) 
        X_STIM = X_ED 
    
    X_epochs = np.array([X_ED, X_MD, X_LD])        
    X_epochs = np.moveaxis(X_epochs,0,2) 
    return X_epochs 

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

def selectInfCoefs(X_trials, C=1e0, K=5): 
        
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
    
    gv.AVG_EPOCHS = 1 
    gv.trial_size = X_trials.shape[-1] 
    
    if gv.AVG_EPOCHS: 
        gv.trial_size = len(['ED','MD','LD']) 
        
    coefs = np.zeros( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    q1 = np.zeros( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
    q3 = np.zeros( (len(gv.trials), gv.trial_size,  X_trials.shape[3]) ) 
      
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
            X = StandardScaler().fit_transform(X)

            C = selectInfCrossVal(X, y, K)
            
            idx_bin, coefs_bin, q1_bin, q3_bin = selectInfLogistic(X, y, C) 
            
            # print(coefs_bin.shape) 
            
            for idx in range(len(idx_bin)):
                coefs[n_trial, n_bins, idx] = coefs_bin[idx]
                q1[n_trial, n_bins, idx] = q1_bin[idx] 
                q3[n_trial, n_bins, idx] = q3_bin[idx] 
                
    return coefs, q1, q3

def selectInfAngle(coefs, q1, q3): 

    gv.trial_size = coefs.shape[1]
    
    alpha = np.empty( (len(gv.trials), gv.trial_size) ) 
    q1_alpha = np.empty( (len(gv.trials), gv.trial_size) ) 
    q3_alpha = np.empty( (len(gv.trials), gv.trial_size) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 

        if gv.AVG_EPOCHS: 
            alpha[n_trial] = get_angle(coefs[n_trial], coefs[n_trial, 0] ) 
            q1_alpha[n_trial] = get_angle(q1[n_trial], q1[n_trial, 0] ) 
            q3_alpha[n_trial] = get_angle(q3[n_trial], q3[n_trial, 0] ) 
        else:
            alpha[n_trial] = get_angle(coefs[n_trial], np.mean( coefs[n_trial, gv.bins_STIM[:] ], axis=0) ) 
            q1_alpha[n_trial] = get_angle(q1[n_trial], np.mean( q1[n_trial, gv.bins_STIM[:] ], axis=0) ) 
            q3_alpha[n_trial] = get_angle(q3[n_trial], np.mean( q3[n_trial, gv.bins_STIM[:] ], axis=0) ) 
            
    return alpha, q1_alpha, q3_alpha 

def selectInfCorr(coefs,C=1e0): 
    
    corrCoef = np.empty( (len(gv.trials), gv.trial_size, gv.trial_size) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        
        corrCoef[n_trial] = np.corrcoef(coefs[n_trial,:,:]) # bins x coefficients 
        
        if gv.AVG_EPOCHS: 
            figtitle = '%s_session_%s_trial_%s_selectInfCorr_%.2fC' % (gv.mouse,gv.session,gv.trial,C) 
            ax = plt.figure(figtitle).add_subplot()
        
            im = ax.imshow(corrCoef[n_trial], cmap='jet', vmin=0.0, vmax=1.0, origin='lower') 
            
            labels = gv.epochs
            xticks = np.arange(0,len(labels)) 
            yticks = np.arange(0,len(labels))
        
            ax.set_xticks(xticks) ; 
            ax.set_xticklabels(labels) ; 

            ax.set_yticks(yticks) ; 
            ax.set_yticklabels(labels) ; 

        else: 
            figtitle = '%s_session_%s_trial_%s_selectInfCorrEpochs_%.2fC' % (gv.mouse,gv.session,gv.trial,C) 
            ax = plt.figure(figtitle).add_subplot()
            
            im = ax.imshow(corrCoef[n_trial], cmap='jet', origin='lower', vmin=-1, vmax=1, extent = [-2 , gv.duration-2, -2 , gv.duration-2])

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
        
    # return corrCoef

def barCos(alpha, lower, upper):    
    return 0
