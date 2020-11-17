from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.utils as data 

import data.plotting as plot 

import decode.cross_temp.utils as decode 
importlib.reload(decode) 

def temporal_decoder(X_proj, IF_EPOCHS=0, C=1e0, penalty='l2', cv=8, my_decoder=0): 

    NO_PCA=1 
    if X_proj.shape[3]!=gv.n_neurons : 
        NO_PCA=0 
        X_proj = X_proj[:,:,:,0:gv.n_components,:] 
        
    if IF_EPOCHS: 
        gv.epochs = ['ED','MD','LD'] 
    else: 
        gv.epochs = ['all'] 
    
    gv.my_decoder= my_decoder 
    # clf = LogisticRegression(C=C, solver='liblinear', penalty=penalty, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(not gv.standardize)) 
    clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', dual=False, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(not gv.standardize)) 
    # clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage='auto') 
    # clf = LinearDiscriminantAnalysis(tol=1e-6) 
    
    for i, gv.trial in enumerate(gv.trials): 
        X_S1_trials = X_proj[i,0] 
        X_S2_trials = X_proj[i,1] 
        
        X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials) 
        print('trial:', gv.trial, 'X', X_trials.shape,'y', y_trials.shape) 
        
        if gv.my_decoder: 
            scores = decode.cross_temp_clf_par(clf, X_trials, y_trials, cv=cv) 
        else: 
            scores, scores_std = decode.mne_cross_temp_clf(X_trials, y_trials, clf, cv=cv) 
        
        print('scores', scores.shape) 
        decode.cross_temp_plot_mat(scores, IF_EPOCHS) 
        
        figname = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse,gv.session,gv.trial) 

        if NO_PCA: 
            plot.figDir('cross_temp') 
        else:
            plot.figDir('pca_cross_temp') 

        clf_name = clf.__class__.__name__ 
        
        if gv.my_decoder:
            gv.figdir = gv.figdir + '/my_decoder' 

        gv.figdir = gv.figdir + '/' + clf_name + '/C_%.2f_penalty_%s_cv_%d' % (C, penalty, cv) 
        
        if IF_EPOCHS:            
            gv.figdir = gv.figdir + '/epochs'
        
        if not os.path.isdir(gv.figdir):
            os.makedirs(gv.figdir)
            
        plot.save_fig(figname) 
