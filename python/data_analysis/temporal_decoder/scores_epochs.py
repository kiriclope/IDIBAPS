from .libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.utils as fct
import data.utils as data 
import data.plotting as pl 
import data.preprocessing as pp 
import data.fct_facilities as fac
fac.SetPlotParams() 

from models.glms import get_clf
from .decoder import cross_temp_decoder 

def is_pca(X_trials): 
    gv.IF_PCA = 0 
    if X_trials.shape[3]!=gv.n_neurons: 
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
        gv.IF_PCA = 1 
    return X_trials

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
        
    gv.figdir = gv.figdir +'/'+ gv.clf_name + clf_param 
    
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

    gv.num_cores =  int(multiprocessing.cpu_count()) - 1     
    get_clf(C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage, normalize=False) 
    decoder = cross_temp_decoder(gv.clf, scoring='accuracy', cv=cv, shuffle=True, mne_decoder=not(gv.my_decoder), n_jobs=gv.num_cores) 

    X_trials = is_pca(X_trials) 
    get_epochs() 

    scores = np.empty((len(gv.trials), len(gv.epochs), len(gv.epochs) ))
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        X_S1 = X_trials[n_trial,0] 
        X_S2 = X_trials[n_trial,1] 
        
        if gv.SELECTIVE: 
            X_S1, X_S2, idx = pp.selectiveNeurons(X_S1, X_S2, .1)
        
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

    gv.T_WINDOW = 0.0 
    gv.IF_SAVE = 1 
    gv.EDvsLD = 1 
    gv.SAVGOL = 0 
    
    gv.clf_name = 'LogisticRegressionCV' 
    gv.my_decoder = 0 
    
    gv.FEATURE_SELECTION = 0     
    gv.TIBSHIRANI_TRICK = 0
    
    for gv.mouse in [gv.mice[1]] : 
        fct.get_sessions_mouse() 
        fct.get_stimuli_times() 
        fct.get_delays_times() 
        
        for gv.session in [gv.sessions[4]] : 
            X_trials = fct.get_X_y_mouse_session() 
            print(X_trials.shape)
            matplotlib.use('GTK3cairo')
            plot_scores_epochs(X_trials, C=C, penalty=penalty, solver=solver, loss=loss, cv=cv, l1_ratio=l1_ratio, shrinkage=shrinkage)
