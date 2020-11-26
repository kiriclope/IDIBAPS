from .std_lib import * 
from .sklearn_lib import * 
from .mne_lib import * 

from joblib import Parallel, delayed
import multiprocessing
 
sys.path.append('../../')

import data.constants as gv
import data.plotting as pl

def K_fold_clf(clf, X_t_train, X_t_test, y, cv): 
    scores = [] 
    folds = KFold(n_splits=cv, shuffle=True) 
    
    for idx_train, idx_test in folds.split(X_t_train): 
        X_train, y_train = X_t_train[idx_train], y[idx_train] 
        X_test, y_test = X_t_test[idx_test], y[idx_test] 

        if gv.standardize:
            scaler =  StandardScaler().fit(X_train) 
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test) 
        
        clf.fit(X_train, y_train) 
        
        scores.append(clf.score(X_test, y_test)) 
        
    return np.mean(scores) 
    
def mne_cross_temp_clf( X, y, clf=None, cv=10, scoring='accuracy'):
    num_cores = int(1*multiprocessing.cpu_count()/8) 

    print('clf', clf.__class__.__name__ )
    if(clf==None): 
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1,solver='liblinear',penalty='l1')) 
    else:
        if gv.standardize:
            pipe = make_pipeline(StandardScaler(), clf)
        else:
            pipe = clf 
    print('standardize', gv.standardize) 
    time_gen = GeneralizingEstimator(pipe, n_jobs=num_cores, scoring=scoring, verbose=False) 
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=num_cores) 
    scores = np.mean(scores, axis=0) 
    scores_std= np.std(scores, axis=0) 

    return scores, scores_std

def cross_temp_clf_par(clf, X, y, cv=10): 

    num_cores = -int(multiprocessing.cpu_count()/4) 

    def loop(t_train, t_test, clf, X, y, cv): 
        X_t_train = X[:,:,t_train] 
        X_t_test = X[:,:,t_test]
        
        score = K_fold_clf(clf, X_t_train,  X_t_test, y, cv) 
        return score 
    
    scores = Parallel(n_jobs=num_cores, verbose=True)(delayed(loop)(t_train, t_test, clf, X, y, cv) 
                                                      for t_train in range(0, X.shape[2]) 
                                                      for t_test in range(0, X.shape[2]) ) 
    scores = np.asarray(scores) 
    scores = scores.reshape( X.shape[2], X.shape[2] ) 
    return scores 

def cross_temp_plot_diag(scores,scores_std): 

    time = np.linspace(0, gv.duration, scores.shape[0]); 
    diag_scores = np.diag(scores) 
    diag_scores_std = scores_std 

    figtitle = 'cross_temp_diag_%s_session_%s_trial_%s' % (gv.mouse,gv.session,gv.trial) 
    ax = plt.figure(figtitle).add_subplot()
    plt.plot(time, diag_scores) 
    plt.fill_between(time, diag_scores - diag_scores_std, diag_scores + diag_scores_std, alpha=0.25, color='green') 

    y_for_chance = np.repeat(0.50, len(diag_scores) ) ;
    plt.plot(time, y_for_chance, '--', c='black') 
    plt.ylim([0, 1]) 

    plt.axvline(x=2, c='black', linestyle='dashed')
    plt.axvline(x=3, c='black', linestyle='dashed')

    plt.axvline(x=4.5, c='r', linestyle='dashed')
    plt.axvline(x=5.5, c='r', linestyle='dashed')

    plt.axvline(x=6.5, c='r', linestyle='dashed')
    plt.axvline(x=7, c='r', linestyle='dashed')
    
    plt.text(2., 1., 'Sample', rotation=0)
    plt.text(9., 1., 'Test', rotation=0)

    plt.axvline(x=9, c='black', linestyle='dashed')
    plt.axvline(x=10, c='black', linestyle='dashed')
    
    plt.xlim([0,gv.duration]) ;

def cross_temp_plot_mat(scores, IF_EPOCHS=0, IF_MEAN=0):

    figtitle = '%s_session_%s_trial_%s_cross_temp_decoder' % (gv.mouse,gv.session,gv.trial)
    ax = plt.figure(figtitle).add_subplot() 

    if IF_EPOCHS or IF_MEAN: 
        im = ax.imshow(scores, cmap='jet', vmin=0.5, vmax=1, origin='lower')
    elif gv.DELAY_ONLY:
        im = ax.imshow(scores, cmap='jet', origin='lower', vmin=0.5, vmax=1, extent = [gv.t_delay[0]-2 , gv.t_delay[-1]-2, gv.t_delay[0]-2 , gv.t_delay[-1]-2])
    else: 
        im = ax.imshow(scores, cmap='jet', origin='lower', vmin=0.5, vmax=1, extent = [-2 , gv.duration-2, -2 , gv.duration-2]) 
        
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    
    ax.set_title(gv.trial) 

    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax) 
    cbar.set_label('accuracy', rotation=90) 

    if(IF_EPOCHS or IF_MEAN):
        labels = gv.epochs
        xticks = np.arange(0,len(labels)) 
        yticks = np.arange(0,len(labels))
        
        ax.set_xticks(xticks) ; 
        ax.set_xticklabels(labels) ; 

        ax.set_yticks(yticks) ; 
        ax.set_yticklabels(labels) ; 

    else:

        pl.vlines_delay(ax) 
        pl.hlines_delay(ax) 
        
        plt.xlim([gv.t_delay[0]-2, gv.t_delay[-1]-2]); 
        plt.ylim([gv.t_delay[0]-2, gv.t_delay[-1]-2]); 
