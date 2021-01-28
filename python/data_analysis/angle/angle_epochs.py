from .libs import * 
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold 

import data.constants as gv 
reload(gv) 
import data.utils as fct 
reload(fct) 
import data.plotting as pl 
reload(pl) 
import data.preprocessing as pp 
reload(pp) 
import data.angle as agl 
reload(agl)
import data.progressbar as pg 
reload(pg)
import data.fct_facilities as fac 
reload(fac)
fac.SetPlotParams() 

import warnings
warnings.filterwarnings("ignore")

import models.glms 
reload(models.glms) 
from models.glms import * 

import dim_red.pca.pca_decomposition 
reload(dim_red.pca.pca_decomposition) 
from dim_red.pca.pca_decomposition import pca_methods 

import dim_red.spca
reload(dim_red.spca)
from dim_red.spca import supervisedPCA_CV 

import dim_red.pls 
reload(dim_red.pls) 
from dim_red.pls import plsCV 

from . import bootstrap
reload(bootstrap)
from .bootstrap import bootstrap

import statistics
reload(statistics) 
from .statistics import get_p_values, add_pvalue 

from .utils import *

def bootstrap_coefs(X_trials, **kwargs): 
    
    options = set_options(**kwargs)
    
    # if gv.pls_method is not None: 
    #     my_pls = plsCV(cv=gv.pls_cv, pls_method=gv.pls_method, max_comp=gv.pls_max_comp, n_jobs=gv.num_cores, verbose=True) 
    
    get_clf(**options) 
    
    boots_model = bootstrap(gv.clf, bootstrap_method=gv.bootstrap_method, n_boots=gv.n_boots, scaling=gv.scaling, n_jobs=gv.num_cores) 

    get_epochs()
    
    if not gv.AVG_BEFORE_PCA: 
        X_trials = pp.avg_epochs(X_trials) 
        
    coefs = np.empty((len(gv.trials), len(gv.epochs), gv.n_boots, X_trials.shape[3])) 

    # concatenated trials 
    X_concat = np.vstack( ( X_trials[0,0], X_trials[1,0],
                            X_trials[2,0], X_trials[0,1],
                            X_trials[1,1], X_trials[2,1] ) )
    
    y_concat = np.array([np.zeros(int(X_concat.shape[0]/2)), np.ones(int(X_concat.shape[0]/2))]).flatten()    
    print('X_concat', X_concat.shape, 'y_concat', y_concat.shape)
    
    # fix random seed 
    seed = np.random.randint(0,1e6)
        
    for n_trial, gv.trial in enumerate(gv.trials): 
        
        X_S1_S2 = np.vstack( ( X_trials[n_trial,0], X_trials[n_trial,1] ) )         
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
        
        if gv.list_n_components is not None: 
            X_S1_S2 = X_S1_S2[:,0:int(gv.list_n_components[n_trial])] 
            
        if not gv.cos_trials: 
            # same seed for each epoch but different for each trial
            seed = np.random.randint(0,1e6) 
            
        for n_epochs, gv.epoch in enumerate(gv.epochs): 
            if gv.cos_trials: 
                # same seed for each trial but different for each epoch 
                np.random.seed(seed * n_trial) 
            else: 
                # same seed for each epoch but different for each trial 
                np.random.seed(seed) 
                
            X = X_S1_S2[:,:,n_epochs] 
            y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
            
            if n_epochs == 0:
                X = X_concat[:,:,n_epochs] 
                y = y_concat
                
            # thresh_filter = VarianceThreshold(0.001) 
            # thresh_filter.fit(X)      
            # X = thresh_filter.transform(X) 
            # print(X.shape) 
                
            Vh = None 
            
            # if gv.pls_method is not None: 
            #     # print('pls decomposition') 
            #     X = my_pls.fit_transform(X, y) 
            #     print('X', X.shape) 
            
            # if gv.LASSOCV: 
            #     gv.lassoCV.fit(X, y) 
            #     selected = np.argwhere(gv.lassoCV[-1].coef_==0) 
            #     if len(selected)<X.shape[1]: 
            #         X = np.delete(X, selected, axis=1) 
                
            # print('X', X.shape, 'y', y.shape) 
            
            # if gv.TIBSHIRANI_TRICK and (penalty=='l2' or 'LDA' in gv.clf_name or 'PLS' in gv.clf_name): 
            #     X, Vh = SVD_trick(X) 
                
            boots_coefs = boots_model.get_coefs(X, y, Vh) 
            # print(boots_coefs.shape) 
            coefs[n_trial, n_epochs,:, 0:boots_coefs.shape[1]] = boots_coefs 
            
    return coefs 

def boot_cos(x,y):
    idx = np.random.choice(np.arange(len(x)), size=len(x)) 
    x_sample = x[idx] 
    y_sample = y[idx]    
    return agl.cos_between(x_sample, y_sample) 

def get_cos_epochs(coefs): 
    ''' Input: coefs: N_trials x N_epochs x N_boots x N_neurons
        Output: mean_cos, lower_cos and upper_cos: N_trials x N_epochs 
                cos_sample: N_trials x N_epochs x N_boots 
    '''
    
    cos_sample = np.empty( (len(gv.trials), len(gv.epochs), gv.n_boots ) ) 
    
    mean_cos = np.empty((len(gv.trials), len(gv.epochs))) 
    upper_cos = np.empty( (len(gv.trials), len(gv.epochs)))  
    lower_cos = np.empty((len(gv.trials), len(gv.epochs))) 
    
    for n_trial, gv.trial in enumerate(gv.trials):  # 'ND', 'D1', 'D2'
        if gv.list_n_components is not None:
            x =  coefs[n_trial, 0, :, 0:int(gv.list_n_components[n_trial])] # 'ED' 
        else:
            x =  coefs[n_trial, 0] # 'ED'
        
        for n_epoch, epoch in enumerate(gv.epochs): # 'ED', 'MD', 'LD' 
            if gv.list_n_components is not None:
                y =  coefs[n_trial, n_epoch, :, 0:int(gv.list_n_components[n_trial])] 
            else:
                y =  coefs[n_trial, n_epoch] 

            if gv.n_boots<10:
                for n_boot in range(gv.n_boots): 
                    cos_sample[n_trial, n_epoch, n_boot] = agl.cos_between(x[n_boot], y[n_boot]) 
            else:
                cos_name = '%s cos ED vs %s' % (gv.trial, gv.epoch) 
                with pg.tqdm_joblib(pg.tqdm(desc=cos_name, total=gv.n_boots)) as progress_bar: 
                    dum = Parallel(n_jobs=gv.num_cores)(delayed(agl.cos_between)(x[n_boot],y[n_boot]) for n_boot in range(gv.n_boots) ) 
                    cos_sample[n_trial, n_epoch] = np.array(dum) 
            
    mean_cos = np.mean(cos_sample, axis=-1) 
    lower_cos = mean_cos - np.percentile(cos_sample, 25, axis=-1) 
    upper_cos = np.percentile(cos_sample, 75, axis=-1) - mean_cos 
        
    for n_trial, gv.trial in enumerate(gv.trials): 
        print('trial', gv.trial, 'cos', mean_cos[n_trial], 'lower', lower_cos[n_trial], 'upper', upper_cos[n_trial]) 
        
    return mean_cos, lower_cos, upper_cos, cos_sample 

def get_cos_trials(coefs): 
            
    cos_sample = np.empty( (len(gv.epochs), len(gv.trials), gv.n_boots ) ) 
    
    mean_cos = np.empty((len(gv.trials), len(gv.epochs))).T 
    upper_cos = np.empty( (len(gv.trials), len(gv.epochs))).T 
    lower_cos = np.empty((len(gv.trials), len(gv.epochs))).T 
    
    for n_epoch, gv.epoch in enumerate(gv.epochs): # 'ED', 'MD', 'LD' 
        x =  coefs[0, n_epoch] # 'ND' 
        
        for n_trial, gv.trial in enumerate(gv.trials):  # 'ND', 'D1', 'D2' 
            y =  coefs[n_trial, n_epoch] 
            
            if gv.n_boots<10:
                for n_boot in range(gv.n_boots): 
                    cos_sample[n_epoch, n_trial, n_boot] = agl.cos_between(x[n_boot], y[n_boot]) 
            else:                
                cos_name = '%s cos ND vs %s' % (gv.epoch, gv.trial) 
                with pg.tqdm_joblib(pg.tqdm(desc=cos_name, total=gv.n_boots)) as progress_bar: 
                    dum = Parallel(n_jobs=gv.num_cores)(delayed(agl.cos_between)(x[n_boot],y[n_boot]) for n_boot in range(gv.n_boots) ) 
                cos_sample[n_epoch, n_trial] = np.array(dum) 
            
    mean_cos = np.mean(cos_sample, axis=-1) 
    lower_cos = mean_cos - np.percentile(cos_sample, 25, axis=-1) 
    upper_cos = np.percentile(cos_sample, 75, axis=-1) - mean_cos 
        
    for n_epoch, gv.epoch in enumerate(gv.epochs): 
        print('epoch', gv.epoch, 'cos', mean_cos[n_epoch], 'lower', lower_cos[n_epoch], 'upper', upper_cos[n_epoch]) 
        
    return mean_cos, lower_cos, upper_cos, cos_sample 

def get_corr_epochs(coefs, n_boots): 
    ''' Compute the correlation between the coefficient vector of the glm 
    fitted on the average normalized fluo 
    coefs is N_condition x N_time x N_boots x N_neurons
    mean_coefs is N_condition x N_time x N_neurons 
    for example, we compute corrcoef(mean_coefs['ND','ED'], mean_coefs['ND','MD']) 
    we bootstrap with replacement to get some statistics
    ''' 
    mean_coefs = np.mean(coefs, axis=2) 
    
    mean_corr = np.empty( (len(gv.trials), coefs.shape[1]) ) 
    lower_corr = np.empty( (len(gv.trials), coefs.shape[1] ) ) 
    upper_corr = np.empty( (len(gv.trials), coefs.shape[1] ) ) 
    
    corr_sample = np.empty( (len(gv.trials), len(gv.epochs), n_boots) ) 
    
    def boot_corr(x,y):
        idx = np.random.choice(np.arange(len(x)), size=len(x)) 
        x_sample = x[idx] 
        y_sample = y[idx]                 
        return np.corrcoef(x_sample, y_sample)[1,0] 
            
    # resampling (x_i, y_i) pairs 
    for n_trial, gv.trial in enumerate(gv.trials): 
        x =  mean_coefs[n_trial, 0] # 'ED'
        
        for n_epoch, gv.epoch in enumerate(gv.epochs): # 'ED', 'MD', 'LD' 
            y =  mean_coefs[n_trial, n_epoch] 
            
            # for n_boot in range(n_boots): 
            #     idx = np.random.choice(np.arange(len(x)), size=len(x)) 
            #     x_sample = x[idx] 
            #     y_sample = y[idx] 
            #     corr_sample[n_trial, n_epoch, n_boot] = np.corrcoef(x_sample, y_sample)[1,0]
            
            corr_name = '%s corr ND vs %s' % (gv.trial, gv.epoch) 
            with pg.tqdm_joblib(pg.tqdm(desc=corr_name, total=n_boots)) as progress_bar: 
                dum = Parallel(n_jobs=gv.num_cores)(delayed(boot_corr)(x,y) for _ in range(n_boots) ) 
            corr_sample[n_trial, n_epoch] = np.array(dum) 
                
    mean_corr = np.mean(corr_sample, axis=-1) 
    lower_corr = mean_corr - np.percentile(corr_sample, 25, axis=-1) 
    upper_corr = np.percentile(corr_sample, 75, axis=-1) - mean_corr 
        
    for n_trial, gv.trial in enumerate(gv.trials): 
        print('trial', gv.trial, 'corr', mean_corr[n_trial], 'lower', lower_corr[n_trial], 'upper', upper_corr[n_trial]) 
        
    return mean_corr, lower_corr, upper_corr, corr_sample 


def get_corr_trials(coefs, n_boots): 
    ''' Compute the correlation between the coefficient vector of the glm 
    fitted on the average normalized fluo 
    coefs is N_condition x N_time x N_boots x N_neurons
    mean_coefs is N_condition x N_time x N_neurons 
    for example, we compute corrcoef(mean_coefs['ND','ED'], mean_coefs['ND','MD']) 
    we bootstrap with replacement to get some statistics
    ''' 
    
    mean_coefs = np.mean(coefs, axis=2) 
    mean_coefs = np.swapaxes(mean_coefs, 0, 1) 
    
    mean_corr = np.empty( (len(gv.epochs), coefs.shape[1]) ) 
    lower_corr = np.empty( (len(gv.epochs), coefs.shape[1] ) ) 
    upper_corr = np.empty( (len(gv.epochs), coefs.shape[1] ) ) 
    
    corr_sample = np.empty( (len(gv.epochs), len(gv.trials), n_boots) ) 
    
    def boot_corr(x,y):
        idx = np.random.choice(np.arange(len(x)), size=len(x)) 
        x_sample = x[idx] 
        y_sample = y[idx]                 
        return np.corrcoef(x_sample, y_sample)[1,0] 
            
    # resampling (x_i, y_i) pairs 
    for n_epoch, gv.epoch in enumerate(gv.epochs): 
        x =  mean_coefs[n_epoch, 0] # 'ND' 
        
        for n_trial, gv.trial in enumerate(gv.trials): # 'ND', 'D1', 'D2' 
            y =  mean_coefs[n_epoch, n_trial] 
            
            # for n_boot in range(n_boots): 
            #     idx = np.random.choice(np.arange(len(x)), size=len(x)) 
            #     x_sample = x[idx] 
            #     y_sample = y[idx] 
            #     corr_sample[n_trial, n_epoch, n_boot] = np.corrcoef(x_sample, y_sample)[1,0]
            
            corr_name = '%s corr ND vs %s' % (gv.trial, gv.epoch) 
            with pg.tqdm_joblib(pg.tqdm(desc=corr_name, total=n_boots)) as progress_bar: 
                dum = Parallel(n_jobs=gv.num_cores)(delayed(boot_corr)(x,y) for _ in range(n_boots) ) 
            corr_sample[n_epoch, n_trial] = np.array(dum) 
                
    mean_corr = np.mean(corr_sample, axis=-1) 
    lower_corr = mean_corr - np.percentile(corr_sample, 25, axis=-1) 
    upper_corr = np.percentile(corr_sample, 75, axis=-1) - mean_corr 
        
    for n_epoch, gv.epoch in enumerate(gv.epochs): 
        print('epoch', gv.epoch, 'corr', mean_corr[n_epoch], 'lower', lower_corr[n_epoch], 'upper', upper_corr[n_epoch]) 
        
    return mean_corr, lower_corr, upper_corr, corr_sample 

def plot_cos_epochs(X_trials, **kwargs):
    
    options = set_options(**kwargs) 
    create_fig_dir(**options) 
    coefs = bootstrap_coefs(X_trials, **options) 
    
    if gv.cos_trials:
        mean_cos, lower_cos, upper_cos, cos_sample = get_cos_trials(coefs) 
    else: 
        mean_cos, lower_cos, upper_cos, cos_sample = get_cos_epochs(coefs) 
        
    pl.bar_trials_epochs(mean_cos, lower_cos, upper_cos) 
    p_values_cos = get_p_values(cos_sample) 
    add_pvalue(p_values_cos) 
    
    if np.all(mean_cos>-0.1): 
        plt.ylim([-0.1, 1.5]) 
    else: 
        plt.ylim([-1.1, 1.5]) 
       
    figtitle = '%s_%s_bars_cos_alp' % (gv.mouse, gv.session) 
    pl.save_fig(figtitle) 
    
    # if gv.cos_trials: 
    #     mean_corr, lower_corr, upper_corr, corr_sample = get_corr_trials(coefs, 1000) 
    # else: 
    #     mean_corr, lower_corr, upper_corr, corr_sample = get_corr_epochs(coefs, 1000) 
        
    # p_values_corr = get_p_values(corr_sample) 
    # pl.bar_trials_epochs(mean_corr, lower_corr, upper_corr, var_name='corr') 
    # add_pvalue(p_values_corr) 
    # if np.all(mean_corr>-0.1): 
    #     plt.ylim([-0.1, 1.1]) 
    # else: 
    #     plt.ylim([-1, 1.1]) 
    
    # figtitle = '%s_%s_bars_corr' % (gv.mouse, gv.session) 
    # pl.save_fig(figtitle) 
    
def plot_loop_mice_sessions(**kwargs): 

    options = set_options(**kwargs) 
    set_globals(**options) 
                                    
    # scaling before clf, when using pca use None 
    gv.scaling = 'standardize_sample' # 'standardize_sample' # 'standardize', 'normalize', 'standardize_sample', 'normalize_sample' or None 
    
    # PCA parameters
    gv.AVG_BEFORE_PCA = 1
    gv.DELAY_ONLY = 0 
    gv.explained_variance = .9 
    gv.n_components = 10 
    gv.list_n_components = None 
    gv.inflection = None 
    gv.minka_mle = False 
    gv.pca_model = None # PCA, sparsePCA, supervisedPCA or None 
    gv.sparse_alpha = 1 
    gv.ridge_alpha = .01 
    gv.pca_method = 'concatenated' # 'hybrid', 'concatenated', 'averaged' or None 
    gv.max_threshold = 10 
    gv.n_thresholds = 100 
    gv.spca_scoring = 'roc_auc' # 'mse', 'log_loss' or 'roc_auc'  
    gv.spca_cv = 5 
    
    if gv.pca_model is not None: 
        # gv.scaling = None # safety for dummies 
        if 'supervisedPCA'==gv.pca_model: 
            my_pca = supervisedPCA_CV(n_components=gv.n_components, explained_variance=gv.explained_variance,
                                      cv=gv.spca_cv, max_threshold=gv.max_threshold, n_thresholds=gv.n_thresholds,
                                      verbose=options['verbose'], n_jobs=gv.num_cores, scoring=gv.spca_scoring) 
        else: 
            # my_pca = pca_methods(pca_method=gv.pca_method, explained_variance=gv.explained_variance,
            #                      inflection=gv.inflection, minka_mle=gv.minka_mle, verbose=options['verbose']) 
            my_pca = pca_methods(pca_model=gv.pca_model, pca_method=gv.pca_method, n_components= gv.n_components,
                                 total_explained_variance=gv.explained_variance, inflection=gv.inflection,
                                 minka_mle=gv.minka_mle, verbose=options['verbose'], ridge_alpha=gv.ridge_alpha, alpha=gv.sparse_alpha) 
    # PLS parameters 
    # gv.pls_n_comp = None 
    gv.pls_max_comp = 30 # 'full', int or None 
    gv.pls_method = None # 'PLSRegression', 'PLSCanonical', 'PLSSVD', CCA or None 
    gv.pls_cv = 5 
    
    if gv.pls_method is not None: 
        gv.pca_method = None  # safety for dummies 
        # gv.scaling = None # safety for dummies 
        my_pls = plsCV(cv=gv.pls_cv, pls_method=gv.pls_method, max_comp=gv.pls_max_comp, n_jobs=gv.num_cores, verbose=options['verbose']) 
        
    for gv.mouse in [gv.mice[1]]:
        fct.get_days() 
        
        for gv.day in [gv.days[-1]]: 
            if gv.SYNTHETIC: 
                X_trials, y = syn.synthetic_data(0.5) 
            else: 
                X_all, y_all = fct.get_fluo_data()
                X_all = pp.preprocess_X(X_all) 
                X_trials = fct.get_X_S1_S2(X_all, y_all) 
                
            if gv.ED_MD_LD: 
                X_trials = X_trials[...,gv.bins_ED_MD_LD] 
            if gv.DELAY_ONLY: 
                X_trials = X_trials[...,gv.bins_delay] 
                gv.bin_start = gv.bins_delay[0] 
                
            if gv.AVG_BEFORE_PCA: 
                X_trials = pp.avg_epochs(X_trials)
                print('X_trials', X_trials.shape)  
            
            if (gv.pca_model is not None) or (gv.pls_method is not None): 
                                    
                if gv.pca_model is not None: 
                    X_trials = my_pca.fit_transform(X_trials) 
                    gv.list_n_components = my_pca.list_n_components 
                    # gv.n_neurons = X_trials.shape[3]
                    # print('X_trials', X_trials.shape) 
                    print(gv.list_n_components)
                    
                elif gv.pls_method is not None: 
                    X_trials = my_pls.trial_hybrid(X_trials, y) 
                    
            print('bootstrap samples:', gv.n_boots, ', clf:', gv.clf_name, 
                  ', scaling:', gv.scaling, ', scoring:', gv.scoring, ', n_splits:', options['n_splits'])
            
            # print('pca_method:', gv.pca_method, ', pls_method:', gv.pls_method, ', n_components', X_trials.shape[3]) 
            
            matplotlib.use('Agg') # so that fig saves when in the in the background 
            # matplotlib.use('GTK3cairo') 
            plot_cos_epochs(X_trials, **options) 
            plt.close('all') 

def cosDaysEpochs(**kwargs): 
    matplotlib.use('Agg') # so that fig saves when in the in the background 
    options = set_options(**kwargs) 
    set_globals(**options) 

    # scaling before clf 
    gv.AVG_BEFORE_PCA = 1 
    gv.pca_model = None # PCA, sparsePCA, supervisedPCA or None 
    gv.pls_method = None # 'PLSRegression', 'PLSCanonical', 'PLSSVD', CCA or None 
    gv.scaling = 'standardize_sample' # 'standardize_sample' # 'standardize', 'normalize', 'standardize_sample', 'normalize_sample' or None 
    epoch_str = ['middle', 'late'] 
    
    create_fig_dir(**options) 

    print('bootstrap samples:', gv.n_boots, ', clf:', gv.clf_name, 
          ', scaling:', gv.scaling, ', scoring:', gv.scoring, ', n_splits:', options['n_splits'])
    
    for gv.mouse in [gv.mice[0]]: 
        fct.get_days() 

        mean_cos = np.empty( ( len(gv.days), len(gv.trials), len(gv.epochs) ) ) 
        lower_cos = np.empty( ( len(gv.days), len(gv.trials), len(gv.epochs) ) ) 
        upper_cos = np.empty( ( len(gv.days), len(gv.trials), len(gv.epochs) ) ) 
        
        for i_day, gv.day in enumerate(gv.days): 
            X_all, y_all = fct.get_fluo_data()
            X_all = pp.preprocess_X(X_all) 
            X_trials = fct.get_X_S1_S2(X_all, y_all) 
                
            X_trials = pp.avg_epochs(X_trials) 
            
            coefs = bootstrap_coefs(X_trials, **options) 
            mean_cos[i_day], lower_cos[i_day], upper_cos[i_day], _ = get_cos_epochs(coefs) 

        for i_epoch, gv.epoch in enumerate(gv.epochs[1:]): # epochs MD and LD 

            figtitle = '%s_cosDays_%s' % (gv.mouse, gv.epoch) 
            plt.figure(figtitle)
            
            for i_trial, gv.trial in enumerate(gv.trials): # all trials, ND, D1 and D2 

                mean = mean_cos[:, i_trial, i_epoch+1] # all days
                print(gv.trial, mean) 
                error = np.absolute(np.vstack([ lower_cos[:, i_trial, i_epoch+1], upper_cos[:, i_trial, i_epoch+1] ] )) # all days 
                
                plt.errorbar(gv.days, mean, yerr=error, color=gv.pal[i_trial] ) 
                # plt.title(epoch_str[i_epoch]) 
                plt.ylabel('cos($\\beta_{DPA}$,$\\beta_{%s}$)' % epoch_str[i_epoch]) 
                plt.xlabel('Day')
                plt.xticks(gv.days) 
                plt.ylim([-.1, 1.1]) 
                
            pl.save_fig(figtitle)                 
            plt.close('all') 
            

def cosDaysTasks(**kwargs): 
    matplotlib.use('Agg') # so that fig saves when in the in the background 
    options = set_options(**kwargs) 
    set_globals(**options) 
    
    # scaling before clf
    gv.cos_trials = 1
    gv.AVG_BEFORE_PCA = 1 
    gv.pca_model = None # PCA, sparsePCA, supervisedPCA or None 
    gv.pls_method = None # 'PLSRegression', 'PLSCanonical', 'PLSSVD', CCA or None 
    gv.scaling = 'standardize_sample' # 'standardize_sample' # 'standardize', 'normalize', 'standardize_sample', 'normalize_sample' or None 
    trial_str = ['dual Go', 'dual NoGo']     
    
    create_fig_dir(**options) 
    
    print('bootstrap samples:', gv.n_boots, ', clf:', gv.clf_name, 
          ', scaling:', gv.scaling, ', scoring:', gv.scoring, ', n_splits:', options['n_splits'])
    
    for gv.mouse in [gv.mice[1]]: 
        fct.get_days() 
        mean_cos = np.empty( ( len(gv.days), len(gv.trials), len(gv.epochs) ) ) 
        lower_cos = np.empty( ( len(gv.days), len(gv.trials), len(gv.epochs) ) ) 
        upper_cos = np.empty( ( len(gv.days), len(gv.trials), len(gv.epochs) ) ) 
        
        for i_day, gv.day in enumerate(gv.days): 
            X_all, y_all = fct.get_fluo_data()
            X_all = pp.preprocess_X(X_all) 
            X_trials = fct.get_X_S1_S2(X_all, y_all) 
                
            X_trials = pp.avg_epochs(X_trials) 
            
            coefs = bootstrap_coefs(X_trials, **options) 
            mean_cos[i_day], lower_cos[i_day], upper_cos[i_day], _ = get_cos_trials(coefs) 
            
        for i_trial, gv.trial in enumerate(gv.trials[1:]): # all trials, ND, D1 and D2 
            
            figtitle = '%s_cosDays_%s' % (gv.mouse, gv.trial) 
            plt.figure(figtitle) 
            
            for i_epoch, gv.epoch in enumerate(gv.epochs): # epochs in ED, MD and LD 
                
                mean = mean_cos[:, i_epoch, i_trial+1] # all days 
                print(gv.trial, mean) 
                error = np.absolute(np.vstack([ lower_cos[:, i_epoch, i_trial+1], upper_cos[:, i_epoch, i_trial+1] ] )) # all days 
                
                plt.errorbar(gv.days, mean, yerr=error, color=gv.pal[i_epoch] ) 
                plt.ylabel('cos($\\beta_{DPA}$,$\\beta_{%s}$)' % trial_str[i_trial]) 
                plt.xlabel('Day')
                plt.xticks(gv.days) 
                plt.ylim([-.1, 1.1]) 
                
            pl.save_fig(figtitle)                 
            plt.close('all') 
            
            
