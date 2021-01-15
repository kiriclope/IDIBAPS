from .libs import * 

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
import data.synthetic as syn
reload(agl)
import data.fct_facilities as fac 
reload(fac)
fac.SetPlotParams() 

import warnings 
warnings.filterwarnings("ignore") 

import models.glms 
reload(models.glms) 
from models.glms import set_options, get_clf 

import dim_red.pca.pca_decomposition 
reload(dim_red.pca.pca_decomposition) 
from dim_red.pca.pca_decomposition import pca_methods 

import dim_red.spca
reload(dim_red.spca)
from dim_red.spca import supervisedPCA_CV 

import dim_red.pls
reload(dim_red.pls)
from dim_red.pls import plsCV

from . import decoder
reload(decoder)
from .decoder import cross_temp_decoder 

from . import utils
reload(utils)
from .utils import *

def get_scores(X_trials, **kwargs): 
    
    options = set_options(**kwargs) 
    get_clf(**kwargs) 
    
    decoder = cross_temp_decoder(gv.clf, scoring=gv.scoring, cv=kwargs['n_splits'], shuffle=gv.shuffle, random_state=gv.random_state, mne_decoder=not(gv.my_decoder), fold_type=gv.fold_type, standardize=gv.standardize, n_jobs=gv.num_cores, n_iter=gv.n_iter) 
    
    get_epochs() 
    
    if gv.scores_trials:
        X_trials = np.moveaxis(X_trials, 0, -1) 

    scores = np.empty((X_trials.shape[0], X_trials.shape[-1], X_trials.shape[-1] ))
    
    for n_cond in range(X_trials.shape[0]): 
        X_S1 = X_trials[n_cond,0] 
        X_S2 = X_trials[n_cond,1] 
        
        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        if gv.list_n_components is not None: 
            X_S1_S2 = X_S1_S2[:,0:int(gv.list_n_components[n_cond])] 
            
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
        
        if gv.scores_trials:
            print('epoch:', gv.epochs[n_cond], 'X', X_S1_S2.shape,'y', y.shape) 
        else: 
            print('trial:', gv.trials[n_cond], 'X', X_S1_S2.shape,'y', y.shape) 
            
        scores[n_cond] = decoder.fit(X_S1_S2, y) 
        
    return scores 
    
def plot_scores_epochs(X_trials, **kwargs):
    
    options = set_options(**kwargs) 
    create_fig_dir(**options) 
    
    scores = get_scores(X_trials, **options) 
    
    for n_cond in range(X_trials.shape[0]):
        gv.trial = gv.trials[n_cond] 
        plot_scores_mat(scores[n_cond]) 
        figtitle = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse, gv.session, gv.trials[n_cond]) 
        pl.save_fig(figtitle) 
        
def plot_loop_mice_sessions(**kwargs):

    options = set_options(**kwargs) 

    gv.num_cores =  int(0.9*multiprocessing.cpu_count()) 
    gv.my_decoder = 1 
    gv.n_iter = 100 
    
    gv.shuffle= True 
    gv.random_state= None  
    
    gv.IF_SAVE = 1 
    gv.SYNTHETIC = 0 
    gv.correct_trial = 0 
    
    # classification parameters 
    gv.clf_name = options['clf_name'] 
    n_splits = options['n_splits'] 
    gv.scoring =  'roc_auc' # 'accuracy' 'roc_auc' 
    gv.fold_type = 'stratified' 
    gv.standardize = True # safety for dummies 
    
    gv.TIBSHIRANI_TRICK = 0 
    
    # preprocessing parameters 
    gv.T_WINDOW = 0. 
    gv.EDvsLD = 1 # average over epochs ED, MD and LD 
    
    # only useful with dim red methods 
    gv.ED_MD_LD = 1 
    gv.DELAY_ONLY = 0 
    
    gv.SAVGOL = 0 # sav_gol filter 
    gv.Z_SCORE = 0 # z_score with BL mean and std 
    
    # feature selection 
    gv.FEATURE_SELECTION = 0 
    gv.LASSOCV = 0 
        
    # PCA parameters 
    gv.explained_variance = .9 
    gv.list_n_components = None 
    gv.inflection = False 
    gv.pca_model = None 
    gv.pca_method = 'concatenated' # 'hybrid', 'concatenated', 'averaged', 'supervised' or None 
    gv.max_threshold = 10 
    gv.n_thresholds = 100 
    
    if gv.pca_model is not None: 
        if 'supervised' in gv.pca_model: 
            my_pca = supervisedPCA_CV(explained_variance=gv.explained_variance, cv=5, max_threshold=gv.max_threshold, Cs=gv.n_thresholds, verbose=True, n_jobs=gv.num_cores) 
        else: 
            my_pca = pca_methods(pca_model=gv.pca_model, pca_method=gv.pca_method, explained_variance=gv.explained_variance, inflection=gv.inflection) 
            
    # PLS parameters 
    gv.pls_max_comp = 100 # 'full', int or None 
    gv.pls_method = None # 'PLSRegression' # 'PLSRegression', 'PLSCanonical', 'PLSSVD' or None 
    gv.pls_cv = 5 
    
    if gv.pls_method is not None: 
        gv.pca_method = None  # safety for dummies 
        # gv.scaling = None # safety for dummies 
        my_pls = plsCV(cv=gv.pls_cv, pls_method=gv.pls_method, max_comp=gv.pls_max_comp, n_jobs=gv.num_cores, verbose=True) 
        
    for gv.mouse in gv.mice : 
        fct.get_sessions_mouse() 
        fct.get_stimuli_times() 
        fct.get_delays_times() 
        
        for gv.session in gv.sessions : 
            if gv.SYNTHETIC: 
                X_trials, y = syn.synthetic_data(0.5) 
            else: 
                X_trials, y = fct.get_X_y_mouse_session() 

            if gv.ED_MD_LD: 
                X_trials = X_trials[:,:,:,:,gv.bins_ED_MD_LD] 
            if gv.DELAY_ONLY: 
                X_trials = X_trials[:,:,:,:,gv.bins_delay] 
                gv.bin_start = gv.bins_delay[0] 
                
            X_trials = pp.avg_epochs(X_trials) 
            # print('X_trials', X_trials.shape) 
            
            if gv.pca_model is not None: 
                X_trials = my_pca.fit_transform(X_trials, y) 
                gv.list_n_components = my_pca.list_n_components
                print(gv.list_n_components)
            elif gv.pls_method is not None: 
                X_trials = my_pls.trial_hybrid(X_trials, y) 
                    
            print('decoder:', gv.my_decoder, 'clf:', gv.clf_name, 
                  ', scaling:', gv.scaling, ', scoring:', gv.scoring, ', n_splits:', n_splits, 
                  ', pca_method:', gv.pca_method, ', pls_method:', gv.pls_method,
                  ', n_components', X_trials.shape[3]) 
            
            matplotlib.use('Agg') # so that fig saves when in the in the background 
            # matplotlib.use('GTK3cairo') 
            plot_scores_epochs(X_trials, **options) 
            plt.close('all') 
