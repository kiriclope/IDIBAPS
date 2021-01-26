from .libs import * 

import data.constants as gv 
reload(gv) 
import data.utils as fct 
reload(fct) 
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

from . import decoder
reload(decoder)
from .decoder import cross_temp_decoder 

from . import utils
reload(utils)
from .utils import *

def get_scores(X_trials, **kwargs): 
    
    options = set_options(**kwargs) 
    get_clf(**options)
    
    decoder = cross_temp_decoder(gv.clf, scoring=options['scoring'], cv=options['n_splits'],
                                 shuffle=options['shuffle'], random_state=options['random_state'],
                                 my_decoder=options['my_decoder'], fold_type=options['fold_type'],
                                 standardize=options['standardize'], n_jobs=options['num_cores'],
                                 n_iter=options['n_iter'], figdir=options['figdir']) 
    
    get_epochs() 
    
    if gv.scores_trials: 
        X_trials = np.swapaxes(X_trials, 0, -1) 
        
    if not gv.AVG_BEFORE_PCA: 
        X_trials = pp.avg_epochs(X_trials) 
        
    scores = np.empty((X_trials.shape[0], X_trials.shape[-1], X_trials.shape[-1] ))
    
    for n_cond in range(X_trials.shape[0]): 
        X_S1 = X_trials[n_cond,0] 
        X_S2 = X_trials[n_cond,1] 
        
        X_S1_S2 = np.vstack((X_S1, X_S2))
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten()
        
        if gv.list_n_components is not None: 
            X_S1_S2 = X_S1_S2[:,0:int(gv.list_n_components[n_cond])] 
                    
        if gv.scores_trials:
            print('epoch:', gv.epochs[n_cond], 'X', X_S1_S2.shape,'y', y.shape) 
        else: 
            print('trial:', gv.trials[n_cond], 'X', X_S1_S2.shape,'y', y.shape) 
            
        scores[n_cond] = decoder.fit(X_S1_S2, y) 
        
    return scores 
    
def plot_scores_epochs(X_trials, **kwargs):
    
    options = set_options(**kwargs) 
    
    scores = get_scores(X_trials, **options) 
    
    for n_cond in range(X_trials.shape[0]):

        if gv.scores_trials: 
            gv.epoch = gv.epochs[n_cond]
        else:
            gv.trial = gv.trials[n_cond]
            
        plot_scores_mat(scores[n_cond]) 
        
def plot_loop_mice_sessions(**kwargs):

    options = set_options(**kwargs) 
    set_globals(**options)     
    
    # PCA parameters 
    gv.AVG_BEFORE_PCA = 1 
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
    gv.spca_scoring = 'accuracy' # 'mse', 'log_loss' or 'roc_auc' 
    gv.spca_cv = 5 
    
    if gv.pca_model is not None: 
        if 'supervisedPCA'==gv.pca_model: 
            my_pca = supervisedPCA_CV(explained_variance=gv.explained_variance, cv=5, max_threshold=gv.max_threshold, Cs=gv.n_thresholds,
                                      verbose=options['verbose'], n_jobs=gv.num_cores) 
        else: 
            my_pca = pca_methods(pca_model=gv.pca_model, pca_method=gv.pca_method, n_components= gv.n_components, 
                                 total_explained_variance=gv.explained_variance, inflection=gv.inflection, 
                                 minka_mle=gv.minka_mle, verbose=True, ridge_alpha=gv.ridge_alpha, alpha=gv.sparse_alpha) 
            
    # PLS parameters 
    gv.pls_max_comp = 100 # 'full', int or None 
    gv.pls_method = None # 'PLSRegression' # 'PLSRegression', 'PLSCanonical', 'PLSSVD' or None 
    gv.pls_cv = 5 
    
    if gv.pls_method is not None: 
        gv.pca_method = None  # safety for dummies 
        # gv.scaling = None # safety for dummies 
        my_pls = plsCV(cv=gv.pls_cv, pls_method=gv.pls_method, max_comp=gv.pls_max_comp, n_jobs=gv.num_cores, verbose=options['verbose']) 
    
    for gv.mouse in [gv.mice[-2]] : 
        fct.get_days() 
        for gv.day in [gv.days[-1]] : 
            if gv.SYNTHETIC: 
                X_trials, y = syn.synthetic_data(0.5) 
            else: 
                X_all, y_all = fct.get_fluo_data() 
                X_all = pp.preprocess_X(X_all) 
                X_trials = fct.get_X_S1_S2(X_all, y_all) 
                y = np.array([np.zeros(X_trials.shape[2]), np.ones(X_trials.shape[2])]).flatten() 
                
            options['figdir'] = create_fig_dir(**options) 
            print(options['figdir']) 
            
            if gv.ED_MD_LD: 
                X_trials = X_trials[...,gv.bins_ED_MD_LD] 
            if gv.DELAY_ONLY: 
                X_trials = X_trials[...,gv.bins_delay] 
                gv.bin_start = gv.bins_delay[0] 

            if gv.AVG_BEFORE_PCA: 
                X_trials = pp.avg_epochs(X_trials) 
                
            print('X_trials', X_trials.shape) 
            
            if gv.pca_model is not None: 
                X_trials = my_pca.fit_transform(X_trials, y) 
                gv.list_n_components = my_pca.list_n_components 
                print(gv.list_n_components) 
            elif gv.pls_method is not None: 
                X_trials = my_pls.trial_hybrid(X_trials, y) 
            
            print('decoder:', gv.my_decoder, 'clf:', gv.clf_name, 
                  ', standardize:', options['standardize'], ', scoring:', options['scoring'], ', n_splits:', options['n_splits'] ) 
                  # print('pca_method:', gv.pca_method, ', pls_method:', gv.pls_method, ', n_components', X_trials.shape[3]) 
            
            matplotlib.use('Agg') # so that fig saves when in the in the background 
            # matplotlib.use('GTK3cairo') 
            plot_scores_epochs(X_trials, **options) 
            plt.close('all') 
