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
import data.progressbar as pg 
reload(pg)
import data.fct_facilities as fac 
reload(fac)
fac.SetPlotParams() 

import warnings 
warnings.filterwarnings("ignore") 

import models.glms 
reload(models.glms) 
from models.glms import get_clf 

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

def get_epochs():
    if gv.EDvsLD: 
        gv.epochs = ['ED', 'MD', 'LD'] 
        print('angle btw ED and other epochs') 
    else: 
        gv.epochs = ['Stim', 'ED', 'MD', 'LD'] 
        print('angle btw STIM and other epochs') 

def create_fig_dir(C=1, penalty='l1', solver='liblinear', cv=0, loss='lsqr', shrinkage='auto'): 
    pl.figDir() 

    clf_param = ''
    if 'LogisticRegression' in gv.clf_name : 
        clf_param = '/C_%.3f_penalty_%s_solver_%s/' % (C, penalty, solver) 
    elif gv.clf_name in 'LinearSVC': 
        clf_param = '/C_%.3f_penalty_%s_loss_%s/' % (C, penalty, loss) 
    elif gv.clf_name in 'LinearDiscriminantAnalysis': 
        clf_param = '/shrinkage_%s_solver_lsqr/' % shrinkage 
        
    gv.figdir = gv.figdir +'/'+ gv.clf_name + clf_param + '/' + gv.scoring 
    
    if gv.my_decoder: 
        gv.figdir = gv.figdir + '/kfold_%d' % cv 
    else: 
        gv.figdir = gv.figdir + '/stratified_kfold_%d' % cv 
        
    if gv.AVG_EPOCHS: 
        gv.figdir = gv.figdir + '/avg_epochs' 
        
    if not os.path.isdir(gv.figdir): 
        os.makedirs(gv.figdir) 
        print('created: ', gv.figdir) 

def get_scores(X_trials, C=1e0, penalty='l1', solver='liblinear', cv=8, l1_ratio=None, loss='lsqr', shrinkage='auto'): 

    get_clf(C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage, normalize=False) 
    decoder = cross_temp_decoder(gv.clf, scoring=gv.scoring, cv=cv, shuffle=True, mne_decoder=not(gv.my_decoder), n_jobs=gv.num_cores) 
    
    get_epochs() 
    
    scores = np.empty((len(gv.trials), len(gv.epochs), len(gv.epochs) ))
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        X_S1 = X_trials[n_trial,0] 
        X_S2 = X_trials[n_trial,1] 
        
        # if gv.SELECTIVE: 
        #     X_S1, X_S2, idx = pp.selectiveNeurons(X_S1, X_S2, .1)
        
        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        X_S1_S2 = pp.avg_epochs(X_S1_S2)        
        
        y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
        
        print('trial:', gv.trial, 'X', X_S1_S2.shape,'y', y.shape) 
        scores[n_trial] = decoder.fit(X_S1_S2, y) 
        
    return scores 

def plot_scores_mat(scores):

    figtitle = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse,gv.session,gv.trial)
    ax = plt.figure(figtitle).add_subplot() 
    
    if gv.EDvsLD:
        im = ax.imshow(scores, cmap='jet', vmin=0.5, vmax=1, origin='lower')

        xticks = np.arange(0,len(gv.epochs)) 
        yticks = np.arange(0,len(gv.epochs))
        
        ax.set_xticks(xticks) ; 
        ax.set_xticklabels(gv.epochs) ; 

        ax.set_yticks(yticks) ; 
        ax.set_yticklabels(gv.epochs) ; 
    else:
        im = ax.imshow(scores, cmap='jet', origin='lower', vmin=0.5, vmax=1, extent = [-2 , gv.duration-2, -2 , gv.duration-2]) 
        pl.vlines_delay(ax) 
        pl.hlines_delay(ax) 
        
        plt.xlim([gv.t_delay[0]-2, gv.t_delay[-1]-2]); 
        plt.ylim([gv.t_delay[0]-2, gv.t_delay[-1]-2]); 
        
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')    
    ax.set_title(gv.trial) 
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax) 
    cbar.set_label('accuracy', rotation=90)
    
def plot_scores_epochs(X_trials, C=1e0, penalty='l1', solver='liblinear', cv=8, l1_ratio=None, loss='lsqr', shrinkage='auto'):

    create_fig_dir(C=C, penalty=penalty, solver=solver, cv=cv, loss=loss, shrinkage=shrinkage)
    
    scores = get_scores(X_trials, C=C, penalty=penalty, solver=solver, cv=cv, l1_ratio=l1_ratio, loss=loss, shrinkage=shrinkage)

    for n_trial, gv.trial in enumerate(gv.trials):
        plot_scores_mat(scores[n_trial]) 
        figtitle = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse,gv.session,gv.trial)
        pl.save_fig(figtitle) 

def plot_loop_mice_sessions(C=1e0, penalty='l2', solver = 'liblinear', loss='squared_hinge', cv=10, l1_ratio=None, shrinkage='auto'): 
    
    gv.num_cores =  int(0.9*multiprocessing.cpu_count()) 
    gv.my_decoder = 1 
    gv.IF_SAVE = 1 
    gv.correct_trial = 0 
    
    # classification parameters 
    gv.clf_name = 'LogisticRegressionCV'
    gv.scoring = 'roc_auc' 
    gv.TIBSHIRANI_TRICK = 0 
        
    # preprocessing parameters 
    gv.T_WINDOW = 0.5 
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
    gv.explained_variance = 0.90  
    gv.pca_method = 'hybrid' # 'hybrid', 'concatenated', 'averaged', 'supervised' or None 
    gv.max_threshold = 10 
    gv.n_thresholds = 100 
    
    if gv.pca_method is not None: 
        if gv.pca_method in 'supervised': 
            my_pca = supervisedPCA_CV(explained_variance=gv.explained_variance, cv=5, max_threshold=gv.max_threshold, Cs=gv.n_thresholds, verbose=True, n_jobs=gv.num_cores) 
        else: 
            my_pca = pca_methods(pca_method=gv.pca_method, explained_variance=gv.explained_variance) 
            
    # PLS parameters 
    gv.pls_max_comp = 100 # 'full', int or None 
    gv.pls_method = None # 'PLSRegression' # 'PLSRegression', 'PLSCanonical', 'PLSSVD' or None 
    gv.pls_cv = 5 
    
    if gv.pls_method is not None: 
        gv.pca_method = None  # safety for dummies 
        # gv.scaling = None # safety for dummies 
        my_pls = plsCV(cv=gv.pls_cv, pls_method=gv.pls_method, max_comp=gv.pls_max_comp, n_jobs=gv.num_cores, verbose=True) 
        
    for gv.mouse in [gv.mice[2]] : 
        fct.get_sessions_mouse() 
        fct.get_stimuli_times() 
        fct.get_delays_times() 
        
        for gv.session in [gv.sessions[3]] : 
            X_trials, y = fct.get_X_y_mouse_session() 
            
            if (gv.pca_method is not None) or (gv.pls_method is not None): 
                
                if gv.ED_MD_LD: 
                    X_trials = X_trials[:,:,:,:,gv.bins_ED_MD_LD] 
                if gv.DELAY_ONLY: 
                    X_trials = X_trials[:,:,:,:,gv.bins_delay] 
                    gv.bin_start = gv.bins_delay[0] 
                    
                if gv.pca_method is not None: 
                    X_trials = my_pca.fit_transform(X_trials, y) 
                elif gv.pls_method is not None : 
                    X_trials = my_pls.trial_hybrid(X_trials, y) 
                    
            print('clf:', gv.clf_name, ', pca_method:', gv.pca_method, ', pls_method:', gv.pls_method, ', n_components', X_trials.shape[3]) 
            
            matplotlib.use('Agg') # so that fig saves when in the in the background 
            # matplotlib.use('GTK3cairo') 
            plot_scores_epochs(X_trials, C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage) 
            plt.close('all') 
            
