from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.utils as data 

import data.plotting as pl 
import data.preprocessing as pp 

import decode.cross_temp.utils as decode 
importlib.reload(decode) 

def create_fig_dir(C=1, penalty='l1', solver='liblinear', cv=0, loss='lsqr', shrinkage='auto'): 
    pl.figDir() 
    
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
        
def temporal_decoder(X_trials, C=1e0, penalty='l1', solver='liblinear', cv=8, l1_ratio=None, loss='lsqr', shrinkage='auto'): 

    gv.AVG_EPOCHS=1
    gv.SELECTIVE=0 

    gv.num_cores = 5 
    
    if gv.EDvsLD: 
        gv.epochs = ['ED','MD','LD'] 
    else:
        gv.epochs = ['STIM','ED','MD','LD'] 

    gv.IF_PCA=0
    if X_trials.shape[3]!=gv.n_neurons : 
        gv.IF_PCA=1
        X_trials = X_trials[:,:,:,0:gv.n_components,:] 
    
    # clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8),
    #                          fit_intercept=bool(not gv.standardize), l1_ratio=l1_ratio) 
    
    clf = LogisticRegressionCV(Cs=C, solver=solver, penalty=penalty, tol=1e-6, max_iter=int(1e8), 
                               fit_intercept=bool(not gv.standardize), n_jobs=10) 
    
    # clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage=1) 
    
    gv.clf_name = clf.__class__.__name__ 
    create_fig_dir(C=C, penalty=penalty, solver=solver, cv=cv, loss=loss, shrinkage=shrinkage) 
    
    for n_trial, gv.trial in enumerate(gv.trials): 
        X_S1 = X_trials[n_trial,0] 
        X_S2 = X_trials[n_trial,1] 
        
        if gv.SELECTIVE: 
            X_S1, X_S2, idx = pp.selectiveNeurons(X_S1, X_S2, .1) 
            
        X_S1_S2 = np.vstack((X_S1, X_S2)) 
        
        print(X_S1_S2.shape)
        
        y = np.array([np.zeros(X_trials.shape[2]), np.ones(X_trials.shape[2])]).flatten() 
        
        # if gv.AVG_EPOCHS: 
        #     X_S1_S2 = pp.avg_epochs(X_S1_S2) 
        # y = np.array([np.zeros(int(X_S1_S2.shape[0]/2)), np.ones(int(X_S1_S2.shape[0]/2))]).flatten() 
        
        print('trial:', gv.trial, 'X', X_S1_S2.shape,'y', y.shape) 
        
        if gv.my_decoder: 
            # scores = decode.cross_temp_clf_par(clf, X_S1_S2, y, cv=cv) 
            scores = decode.cross_temp_clf_par_epochs(clf, X_S1_S2, y, cv=cv) 
        else: 
            scores, scores_std = decode.mne_cross_temp_clf(X_S1_S2, y, clf, cv=cv) 
            
        decode.cross_temp_plot_mat(scores, gv.AVG_EPOCHS) 
        
        figname = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse,gv.session,gv.trial) 
        pl.save_fig(figname) 

