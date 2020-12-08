from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis')

import data.constants as gv 
import data.plotting as pl 
import data.preprocessing as pp 
import data.angle as agl

from selectinf.algorithms import lasso, cv
import regreg.api as rr
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 

def selectInfCrossVal(X, y, K):

    loss = rr.glm.logistic(X,y) 
    lam_seq = np.exp(np.linspace(np.log(1.e-6), np.log(1), 100)) * np.fabs(np.dot(X.T,y)).max() 
    # lam_seq = np.logspace(-2, 2, 100) 
    
    folds = np.arange(X.shape[0]) % K 
    # folds = K 
    CV_compute = cv.CV(loss, folds, lam_seq) 
    lam_CV, CV_val, SD_val, lam_CV_randomized, CV_val_randomized, SD_val_randomized = CV_compute.choose_lambda_CVr() 

    minimum_CV = np.min(CV_val)
    lam_idx = list(lam_seq).index(lam_CV)
    SD_min = SD_val[lam_idx]
    lam_1SD = lam_seq[max([i for i in range(lam_seq.shape[0]) if CV_val[i] <= minimum_CV + SD_min])]
    
    print('lambdas', lam_1SD, lam_CV, lam_CV_randomized) 
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

def cosEDvsTime(alpha, lower_alpha, upper_alpha, sigma=1):
    figname = '%s_%s_cosEDvsTime' % (gv.mouse, gv.session) 
    ax = plt.figure(figname).add_subplot() 
    
    x = gv.time-np.array(2) 
    for n_trial in range(len(gv.trials)):

        y = gaussian_filter1d( np.cos( alpha[n_trial] ), sigma )
        plt.plot(x, y, '-', color=gv.pal[n_trial]) 
        lower = gaussian_filter1d( np.cos( lower_alpha[n_trial] ) , sigma) 
        upper = gaussian_filter1d( np.cos( upper_alpha[n_trial] ), sigma) 
        ax.fill_between(x, lower, upper , color=gv.pal[n_trial], alpha=.1) 
        
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
    
def selectInfCoefs(X_trials, C=1e0, K=5): 
        
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
    
    gv.trial_size = X_trials.shape[-1] 
    
    gv.IF_PCA = 0
    if X_trials.shape[3]!=gv.n_neurons: 
        X_proj = X_trials[:,:,:,0:gv.n_components,:] 
        gv.IF_PCA = 1 
        
    if gv.AVG_EPOCHS: 
        
        if gv.EDvsLD: 
            gv.epochs = ['ED', 'MD', 'LD'] 
            print('angle btw ED and other epochs')
        else:
            gv.epochs = ['Stim', 'ED', 'MD', 'LD'] 
            print('angle btw STIM and other epochs') 

    coefs = np.zeros( (len(gv.trials), len(gv.epochs),  X_trials.shape[3]) ) 
    lower = np.zeros( (len(gv.trials), len(gv.epochs),  X_trials.shape[3]) ) 
    upper = np.zeros( (len(gv.trials), len(gv.epochs),  X_trials.shape[3]) ) 
    
    # y = np.array([np.zeros(X_trials.shape[2]), np.ones(X_trials.shape[2])]).flatten() 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
    
        X_S1 = X_trials[n_trial,0] 
        X_S2 = X_trials[n_trial,1] 
        X_S1_S2 = np.vstack((X_S1, X_S2)) 

        if gv.AVG_EPOCHS: 
            X_S1_S2 = pp.avg_epochs(X_S1_S2)
            
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 

        print('X_S1_S2', X_S1_S2.shape) 

        for n_bins in range(X_S1_S2.shape[2]): 
            X = X_S1_S2[:,:,n_bins] 
            X = StandardScaler().fit_transform(X)

            C = selectInfCrossVal(X, y, K)
            
            idx_bin, coefs_bin, lower_bin, upper_bin = selectInfLogistic(X, y, C) 
            
            # print(coefs_bin.shape) 
            
            for idx in range(len(idx_bin)):
                coefs[n_trial, n_bins, idx] = coefs_bin[idx] 
                lower[n_trial, n_bins, idx] = lower_bin[idx] 
                upper[n_trial, n_bins, idx] = upper_bin[idx] 
                
    return coefs, lower, upper

def selectInfCos(coefs, lower, upper): 

    gv.trial_size = coefs.shape[1]
    
    cos = np.empty( (len(gv.trials), gv.trial_size) ) 
    lower_cos = np.empty( (len(gv.trials), gv.trial_size) ) 
    upper_cos = np.empty( (len(gv.trials), gv.trial_size) ) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 

        if gv.AVG_EPOCHS: 
            cos[n_trial] = agl.get_cos(coefs[n_trial], coefs[n_trial, 0] ) 
            lower_cos[n_trial] = cos[n_trial] - agl.get_cos(lower[n_trial], lower[n_trial, 0] ) 
            upper_cos[n_trial] = agl.get_cos(upper[n_trial], upper[n_trial, 0] ) - cos[n_trial]
        else:
            cos[n_trial] = agl.get_cos(coefs[n_trial], np.mean( coefs[n_trial, gv.bins_STIM[:] ], axis=0) ) 
            lower_cos[n_trial] = cos[n_trial] - agl.get_cos(lower[n_trial], np.mean( lower[n_trial, gv.bins_STIM[:] ], axis=0) ) 
            upper_cos[n_trial] = agl.get_cos(upper[n_trial], np.mean( upper[n_trial, gv.bins_STIM[:] ], axis=0) ) - cos[n_trial] 
            
        print('trial', gv.trial, 'cos', cos[n_trial], 'lower', lower_cos[n_trial], 'upper', upper_cos[n_trial]) 
            
    return cos, lower_cos, upper_cos 

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

def selectInf(X_trials, C=1e0, K=5):
    
    coefs, lower, upper = selectInfCoefs(X_trials, C=C, K=K) 
    mean, lower, upper = selectInfCos(coefs, lower, upper) 

    pl.barCosAlp(mean, lower, upper) 
    plt.ylim([-.1, 1.1]) 
