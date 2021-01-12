import data.constants as gv 

import numpy as np
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 

class pca_methods():
    
    def __init__(self, pca_method='hybrid', explained_variance=0.9, inflection=False, minka_mle=False, verbose=True):
        self.pca_method = pca_method
        self.explained_variance = explained_variance 
        self.scaler = StandardScaler(with_mean=True, with_std=False) 
        self.inflection = inflection 
        self.minka_mle = minka_mle 
        self.list_n_components = np.empty(len(gv.trials)) 
        self.verbose = verbose
        
    def get_inflection_point(self, explained_variance): 
        d2_var = np.gradient(np.gradient(explained_variance))
        try :
            inflection_point = np.argwhere(np.diff(np.sign(d2_var)))[0][0]
        except:
            inflection_point=1 
        return np.maximum(inflection_point,1) 
    
    def get_optimal_number_of_components(self, X): 
        cov = np.dot(X,X.transpose())/float(X.shape[0]) 
        U,s,v = np.linalg.svd(cov)
        S_nn = sum(s) 
        
        for num_components in range(0, s.shape[0] ):
            temp_s = s[0:num_components]
            S_ii = sum(temp_s)
            if (1 - S_ii/float(S_nn)) <= 1 - self.explained_variance: 
                return num_components
            
        return np.maximum(s.shape[0], 1) 

    def trial_hybrid(self, X_trials): 
        # concatenate average over trials 
        X_avg = np.empty( (len(gv.trials), gv.n_neurons, len(gv.samples) * X_trials.shape[-1] ) ) 
        for n_trial in range(len(gv.trials)) :
            X_avg[n_trial] = np.hstack( ( np.mean(X_trials[n_trial,0], axis=0), np.mean(X_trials[n_trial,1], axis=0) ) ) 
            
        # X_avg = np.hstack(X_avg) 
        if self.verbose :
            print('X_avg', X_avg.shape) 

        X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), X_trials.shape[-2], X_trials.shape[-1]) )
            
        for n_trial in range(X_trials.shape[0]): 
            # standardize neurons/features across trials/samples 
            self.scaler.fit(X_avg[n_trial].T) 
            X_avg[n_trial] = self.scaler.transform(X_avg[n_trial].T).T 
        
            # PCA the trial averaged data
            if self.minka_mle:
                n_components = 'mle'
            else:
                n_components = self.get_optimal_number_of_components(X_avg[n_trial].T)
            
            pca = PCA(n_components=n_components) 
            pca.fit(X_avg[n_trial].T) 
        
            n_components = pca.n_components_  
            explained_variance = pca.explained_variance_ratio_
            self.list_n_components[n_trial] = n_components 
            
            if self.inflection:
                n_components = self.get_inflection_point(explained_variance) 
                pca = PCA(n_components=n_components) 
                pca.fit(X_avg[n_trial].T) 
            
                n_components = pca.n_components_ 
                explained_variance = pca.explained_variance_ratio_ 
                self.list_n_components[n_trial] = int(n_components) 
            
            if self.verbose :
                print('trial', gv.trials[n_trial], 'n_pc', n_components,
                      'explained_variance', explained_variance[0:3], 'total' , np.cumsum(explained_variance)[-1]) 
        
            # X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), n_components, X_trials.shape[-1]) ) 
            for j in range(X_trials.shape[1]): 
                for k in range(X_trials.shape[2]): 
                    trial = self.scaler.transform(X_trials[n_trial,j,k,:,:].T).T # neurons x time = features x samples 
                    X_proj[n_trial,j,k,0:n_components] = pca.transform(trial.T).T 
                
        if self.verbose :
            print('X_proj', X_proj.shape)
            
        return X_proj 
    
    def trial_concatenated(self, X_trials):

        # concatenate individual trials 
        X_proj = np.empty( ( len(gv.trials), len(gv.samples), int( gv.n_trials/len(gv.samples) ), X_trials.shape[-2] , X_trials.shape[-1]) ) 
        # trials = []
        for n_trial in range(len(gv.trials)) : 
            X_S1_S2 = np.hstack( (np.hstack(X_trials[n_trial,0]), np.hstack(X_trials[n_trial,1])) ) 
            # trials.append(X_S1_S2) 
            
            # X_concat = np.hstack(trials) 
            X_concat = X_S1_S2
            
            # standardize neurons/features across trials/samples 
            self.scaler.fit(X_concat.T) 
            X_concat = self.scaler.transform(X_concat.T).T 
         
            if self.minka_mle: 
                n_components = 'mle' 
            else: 
                n_components = self.get_optimal_number_of_components(X_concat.T) 
            
            print(X_concat.shape, n_components) 
        
            # pca on X: trials x neurons 
            pca = PCA(n_components=n_components) 
            pca.fit(X_concat.T)
        
            n_components = pca.n_components_ 
            self.list_n_components[n_trial] = n_components 
            explained_variance = pca.explained_variance_ratio_ 
        
            if self.inflection:
                n_components = self.get_inflection_point(explained_variance) 
                pca = PCA(n_components=n_components) 
                X_concat = pca.fit_transform(X_concat.T).T 
            
                n_components = pca.n_components_ 
                self.list_n_components[n_trial] = n_components 
                explained_variance = pca.explained_variance_ratio_ 
            else:
                X_concat = pca.transform(X_concat.T).T 
        
            if self.verbose :
                print('X_concat', X_concat.shape) 
        
            if self.verbose :
                print('n_pc', n_components,'explained_variance', explained_variance[0:3], 'total' , np.cumsum(explained_variance)[-1]*100) 
        
            # X_proj = np.empty( ( len(gv.trials), len(gv.samples), int( gv.n_trials/len(gv.samples) ), n_components , X_trials.shape[-1]) ) 
            # for i in range( len(gv.trials) ): 
            for j in range( len(gv.samples) ) : 
                for k in range( int( gv.n_trials/len(gv.samples) ) ) : 
                    for l in range(n_components) : 
                        # m = i*len(gv.samples)* int( gv.n_trials/len(gv.samples) ) + j * int( gv.n_trials/len(gv.samples) )  + k 
                        m = j * int( gv.n_trials/len(gv.samples) ) + k 
                        X_proj[n_trial,j,k,l] = X_concat[l, X_trials.shape[-1] * m: X_trials.shape[-1] * (m + 1)].flatten() 
            
            if self.verbose :
                print('X_proj', X_proj.shape) 
            
        return X_proj 
    
    def trial_averaged(self, X_trials):
        
        trial_averages = []
        for n_trial in range(len(gv.trials)) : 
            X_S1 = np.mean(X_trials[n_trial,0], axis=0) 
            X_S2 = np.mean(X_trials[n_trial,1], axis=0) 
            X_S1_S2 = np.hstack((X_S1, X_S2))             
            trial_averages.append(X_S1_S2) 
        
        X_avg = np.hstack(trial_averages) 
        # standardize neurons/features across trials/samples 
        X_avg = self.scaler.fit_transform(X_avg.T).T 
        
        if self.minka_mle:
            n_components = 'mle'
        else:
            n_components = self.get_optimal_number_of_components(X_avg.T) 
            
        pca = PCA(n_components=n_components) 
        pca.fit(X_avg.T)
        
        n_components = pca.n_components_ 
        explained_variance = pca.explained_variance_ratio_ 
        
        if self.inflection:
            n_components = self.get_inflection_point(explained_variance) 
            pca = PCA(n_components=n_components) 
            X_proj = pca.fit_transform(X_avg.T).T
            
            n_components = pca.n_components_ 
            explained_variance = pca.explained_variance_ratio_ 
        else:
            X_proj = pca.transform(X_avg.T).T 
            
        if self.verbose :
            print('n_pc', n_components,'explained_variance', explained_variance, 'total' , np.cumsum(explained_variance)[-1]) 
        
        X_proj = np.asarray(X_proj) 
        X_proj = X_proj.reshape(n_components, len(gv.trials), len(gv.samples), X_trials.shape[-1]) 
        X_proj = np.rollaxis(X_proj, 0, -1) 
        
        X_proj = X_proj[:,:,np.newaxis] 
        
        if self.verbose : 
            print('X_proj', X_proj.shape) 
        
        return X_proj
    
    def fit_transform(self, X_trials, y=None): 
        X_proj = [] 
        if self.pca_method in 'hybrid': 
            X_proj = self.trial_hybrid(X_trials) 
        if self.pca_method in 'concatenated': 
            X_proj = self.trial_concatenated(X_trials) 
        if self.pca_method in 'averaged': 
            X_proj = self.trial_averaged(X_trials) 
        return X_proj 
