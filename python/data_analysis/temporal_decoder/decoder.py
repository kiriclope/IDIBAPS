import sys
sys.path.append('../')
import data.progressbar as pg 

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.pipeline import make_pipeline
from mne.decoding import GeneralizingEstimator, cross_val_multiscore

from joblib import Parallel, delayed

class cross_temp_decoder():

    def __init__(self, clf, scoring='accuracy', cv=10, shuffle=True, mne_decoder=False, folds_type='stratified', n_jobs=1):
        self.clf = clf
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.shuffle = shuffle
        self.mne_decoder = mne_decoder 
        self.cv = cv
        self.folds_type = folds_type
        self.scores = None
        
    def mne_cross_temp_scores(self, X, y):

        pipe = make_pipeline(StandardScaler(), self.clf)
        time_gen = GeneralizingEstimator(pipe, n_jobs=self.n_jobs, scoring=self.scoring, verbose=False) 
        scores = cross_val_multiscore(time_gen, X, y, cv=self.cv, n_jobs=self.n_jobs) 
        self.scores = np.mean(scores, axis=0) 
        return self.scores

    def cross_val_loop(self, X, y, t_train, t_test): 
        
        if 'stratified' in self.folds_type:
            folds = StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle) 
        else: 
            folds = KFold(n_splits=self.cv, shuffle=self.shuffle)

        X_t_train = X[:,:,t_train] 
        X_t_test = X[:,:,t_test] 
        
        scores = []
        for idx_train, idx_test in folds.split(X_t_train, y): 
            X_train, y_train = X_t_train[idx_train], y[idx_train] 
            X_test, y_test = X_t_test[idx_test], y[idx_test] 
            
            scaler =  StandardScaler().fit(X_train) 
            X_train = scaler.transform(X_train) 
            X_test = scaler.transform(X_test) 
            
            self.clf.fit(X_train, y_train)             
            scores.append(self.clf.score(X_test, y_test))
            
        return np.mean(scores)
    
    def cross_temp_scores(self, X, y): 
        
        with pg.tqdm_joblib(pg.tqdm(desc="cross validation", total=int(X.shape[2]*X.shape[2]))) as progress_bar: 
            scores = Parallel(n_jobs=self.n_jobs)(delayed(self.cross_val_loop)(X, y, t_train, t_test) 
                                                   for t_train in range(0, X.shape[2]) 
                                                   for t_test in range(0, X.shape[2]) )
            
        self.scores = np.asarray(scores).reshape( X.shape[2], X.shape[2] ) 
        return self.scores 
    
    def fit(self, X, y):

        if self.mne_decoder:
            self.scores = self.mne_cross_temp_scores(X, y) 
        else: 
            self.scores = self.cross_temp_scores(X, y) 

        return self.scores 
