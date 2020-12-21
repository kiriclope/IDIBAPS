import sys, importlib 
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis') 
import data.constants as gv 

import numpy as np
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 

class pca_methods():
    
    def __init__(self, pca_method='hybrid', explained_variance=0.9):
        self.pca_method = pca_method
        self.explained_variance = explained_variance 
        self.scaler = StandardScaler() 

    def get_optimal_number_of_components(self, X): 
        cov = np.dot(X,X.transpose())/float(X.shape[0]) 
        U,s,v = np.linalg.svd(cov)
        S_nn = sum(s) 
        
        for num_components in range(0,s.shape[0]):
            temp_s = s[0:num_components]
            S_ii = sum(temp_s)
            if (1 - S_ii/float(S_nn)) <= 1 - self.explained_variance: 
                return num_components
            
        return s.shape[0] 

    def trial_hybrid(self, X_trials): 
        
        X_avg = np.empty( (len(gv.trials), gv.n_neurons, len(gv.samples) * gv.trial_size ) ) 
        for n_trial in range(len(gv.trials)) :
            X_avg[n_trial] = np.hstack( ( np.mean(X_trials[n_trial,0], axis=0), np.mean(X_trials[n_trial,1], axis=0) ) ) 
    
        X_avg = np.hstack(X_avg) 
        print('X_avg', X_avg.shape) 
    
        # standardize neurons/features across trials/samples 
        self.scaler.fit(X_avg.T) 
        X_avg = self.scaler.transform(X_avg.T).T 
    
        # PCA the trial averaged data 
        n_components = self.get_optimal_number_of_components(X_avg) 
        pca = PCA(n_components=n_components) 
        X_avg = pca.fit_transform(X_avg.T).T 
        
        explained_variance = pca.explained_variance_ratio_ 
        print('n_pc', n_components,'explained_variance', explained_variance[0:3], 'total' , np.cumsum(explained_variance)[-1]) 
        
        X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), n_components, gv.trial_size) ) 
        for i in range(X_trials.shape[0]): 
            for j in range(X_trials.shape[1]): 
                for k in range(X_trials.shape[2]): 
                    trial = self.scaler.transform(X_trials[i,j,k,:,:].T).T # neurons x time = features x samples 
                    X_proj[i,j,k] = pca.transform(trial.T).T 
                
        print('X_proj', X_proj.shape) 
        return X_proj 

    def trial_concatenated(self, X_trials):

        for n_trial in range(len(gv.trials)) : 
            X_S1_S2 = np.hstack( (np.hstack(X_trials[n_trial,0]), np.hstack(X_trials[n_trial,1])) ) 
            trials.append(X_S1_S2)
            
        X_concat = np.hstack(trials) 

        # standardize neurons/features across trials/samples 
        self.scaler.fit(X_concat.T) 
        X_concat = self.scaler.transform(X_concat.T).T 
        
        n_components = self.get_optimal_number_of_components(X_concat) 
        pca = PCA(n_components=n_components) 
        # pca on X: trials x neurons 
        X_concat = pca.fit_transform(X_concat.T).T 
        
        print('X_concat', X_concat.shape) 
        
        explained_variance = pca.explained_variance_ratio_ 
        print('n_pc', n_components,'explained_variance', explained_variance[0:3], 'total' , np.cumsum(explained_variance)[-1]*100) 
        
        X_proj = np.empty( ( len(gv.trials), len(gv.samples), int( gv.n_trials/len(gv.samples) ), n_components , gv.trial_size) ) 
        for i in range( len(gv.trials) ): 
            for j in range( len(gv.samples) ) : 
                for k in range( int( gv.n_trials/len(gv.samples) ) ) : 
                    for l in range(n_components) : 
                        m = i*len(gv.samples)* int( gv.n_trials/len(gv.samples) ) + j * int( gv.n_trials/len(gv.samples) )  + k 
                        X_proj[i,j,k,l] = X_concat[l, gv.trial_size * m: gv.trial_size * (m + 1)].flatten() 
            
        print('X_proj', X_proj.shape) 
        return X_proj 

    def trial_averaged(self, X_trials):
        
        for n_trial in range(len(gv.trials)) : 
            X_S1 = np.mean(X_trials[n_trial,0], axis=0) 
            X_S2 = np.mean(X_trials[n_trial,1], axis=0) 
            X_S1_S2 = np.hstack((X_S1, X_S2))             
            trial_averages.append(X_S1_S2) 
        
        X_avg = np.hstack(trial_averages)
        # standardize neurons/features across trials/samples 
        X_avg = self.scaler.fit_transform(X_avg.T).T 
        
        pca = PCA(n_components=n_components) 
        X_proj = pca.fit_transform(X_avg.T).T
        
        explained_variance = pca.explained_variance_ratio_ 
        print('n_pc', gv.n_components,'explained_variance', explained_variance, 'total' , np.cumsum(explained_variance)[-1]) 
        
        X_proj = np.asarray(X_proj) 
        X_proj = X_proj.reshape(gv.n_components, len(gv.trials), len(gv.samples), gv.trial_size) 
        print('X_proj', X_proj.shape) 
        
        return X_proj
    
    def fit_transform(self, X_trials):
        if self.pca_method in 'hybrid':
            X_proj = self.trial_hybrid(self, X_trials)
        if self.pca_method in 'concatenated':
            X_proj = self.trial_concatenated(self, X_trials)
        if self.pca_method in 'averaged': 
            X_proj = self.trial_concatenated(self, X_trials)
        return X_proj
